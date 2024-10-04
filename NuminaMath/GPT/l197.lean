import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Linear.Equations
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.AbsoluteValue
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Monotone
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificFunctions.Log
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Card
import Mathlib.Data.Multiplicity
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Simplex
import Mathlib.Tactic
import data.nat.prime

namespace equilibrium_stability_l197_197527

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 2)

theorem equilibrium_stability (x : ℝ) :
  (x = 0 → HasDerivAt f (-1) 0 ∧ (-1 < 0)) ∧
  (x = Real.log 2 → HasDerivAt f (2 * Real.log 2) (Real.log 2) ∧ (2 * Real.log 2 > 0)) :=
by
  sorry

end equilibrium_stability_l197_197527


namespace max_books_single_student_borrowed_l197_197499

-- Define the given conditions
def total_students := 20
def zero_books_students := 2
def single_book_students := 10
def two_books_students := 5
def remaining_students := total_students - (zero_books_students + single_book_students + two_books_students)
def average_books_per_student := 2
def total_books_borrowed := total_students * average_books_per_student

-- Proving the maximum number of books a single student could borrow
theorem max_books_single_student_borrowed : 
  let max_books_borrowed := 14 in
  ∀ student_books : Fin remaining_students → ℕ,
  (∀ i, student_books i ≥ 3) → 
  (∑ i, student_books i = total_books_borrowed - (0*zero_books_students + single_book_students + 2*two_books_students)) → 
  ∃ i, student_books i = max_books_borrowed := 
sorry

end max_books_single_student_borrowed_l197_197499


namespace tree_height_after_n_years_l197_197265

-- Define the initial conditions: initial height and yearly growth
def initialHeight : ℝ := 1.8
def yearlyGrowth : ℝ := 0.3

-- Prove the function relationship between height L and years n
theorem tree_height_after_n_years (n : ℕ) : 
  let L := initialHeight + yearlyGrowth * n
  in L = 0.3 * n + 1.8 :=
by
  sorry

end tree_height_after_n_years_l197_197265


namespace speed_in_still_water_l197_197701

-- Define the velocities (speeds)
def speed_downstream (V_w V_s : ℝ) : ℝ := V_w + V_s
def speed_upstream (V_w V_s : ℝ) : ℝ := V_w - V_s

-- Define the given conditions
def downstream_condition (V_w V_s : ℝ) : Prop := speed_downstream V_w V_s = 9
def upstream_condition (V_w V_s : ℝ) : Prop := speed_upstream V_w V_s = 1

-- The main theorem to prove
theorem speed_in_still_water (V_s V_w : ℝ) (h1 : downstream_condition V_w V_s) (h2 : upstream_condition V_w V_s) : V_w = 5 :=
  sorry

end speed_in_still_water_l197_197701


namespace common_chord_is_linear_l197_197622

-- Defining the equations of two intersecting circles
noncomputable def circle1 : ℝ → ℝ → ℝ := sorry
noncomputable def circle2 : ℝ → ℝ → ℝ := sorry

-- Defining a method to eliminate quadratic terms
noncomputable def eliminate_quadratic_terms (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Defining the linear equation representing the common chord
noncomputable def common_chord (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Statement of the problem
theorem common_chord_is_linear (circle1 circle2 : ℝ → ℝ → ℝ) :
  common_chord circle1 circle2 = eliminate_quadratic_terms circle1 circle2 := sorry

end common_chord_is_linear_l197_197622


namespace parallel_vectors_l197_197387

variables (α : ℝ)

def a : ℝ × ℝ := (Real.sin α, 1 - 4 * Real.cos (2 * α))
def b : ℝ × ℝ := (1, 3 * Real.sin α - 2)

theorem parallel_vectors (hα : α ∈ set.Ioo 0 (Real.pi / 2)) (h_parallel : a α = b α) :
  (Real.sin (2 * α)) / (2 + Real.cos α ^ 2) = 4 / 11 := sorry

end parallel_vectors_l197_197387


namespace proof_problem_l197_197430

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then |x - 1| else 3 ^ x

theorem proof_problem :
  (f (f (-2)) = 27) ∧ (∃ a, f a = 2 ∧ a = -1) :=
by
  -- Definitions and conditions are enough for the proof problem statement
  sorry

end proof_problem_l197_197430


namespace max_value_m_l197_197859

-- Given conditions
variables {x y : ℝ}
variable hxy_pos : 0 < x ∧ 0 < y
variable hsum : x + y = 20

-- Define the function m
noncomputable def m := log (x * y) / log 10

-- The theorem stating the maximum value
theorem max_value_m : m ≤ 2 :=
by
  sorry

end max_value_m_l197_197859


namespace gcd_of_X_and_Y_l197_197914

theorem gcd_of_X_and_Y (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : 5 * X = 4 * Y) :
  Nat.gcd X Y = 9 := 
sorry

end gcd_of_X_and_Y_l197_197914


namespace part1_inequality_part2_range_l197_197433

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

-- Part 1: Prove that f(x) ≥ f(0) for all x
theorem part1_inequality : ∀ x : ℝ, f x ≥ f 0 :=
sorry

-- Part 2: Prove that the range of a satisfying 2f(x) ≥ f(a+1) for all x is -4.5 ≤ a ≤ 1.5
theorem part2_range (a : ℝ) (h : ∀ x : ℝ, 2 * f x ≥ f (a + 1)) : -4.5 ≤ a ∧ a ≤ 1.5 :=
sorry

end part1_inequality_part2_range_l197_197433


namespace arithmetic_seq_8th_term_l197_197071

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197071


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197672

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197672


namespace sum_infinite_partial_fraction_l197_197304

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197304


namespace sum_all_values_x_l197_197223

-- Define the problem's condition
def condition (x : ℝ) : Prop := Real.sqrt ((x - 2) ^ 2) = 9

-- Define the theorem to prove the sum of all solutions equals 4
theorem sum_all_values_x : ∑ x in {x : ℝ | condition x}, x = 4 := by
  -- Introduce the definition of condition
  sorry

end sum_all_values_x_l197_197223


namespace b_95_mod_49_l197_197022

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := 
by sorry

end b_95_mod_49_l197_197022


namespace complex_numbers_same_abs_value_l197_197968

variable (a b c : ℂ) (n : ℕ)

theorem complex_numbers_same_abs_value (h1 : a + b + c = 0) (h2 : a^n + b^n + c^n = 0) (h3 : n > 1) :
  ∃ (i j : {x : ℂ // x = a ∨ x = b ∨ x = c}), i ≠ j ∧ complex.abs (i.val) = complex.abs (j.val) :=
by sorry

end complex_numbers_same_abs_value_l197_197968


namespace train_passes_jogger_in_46_seconds_l197_197700

-- Definitions directly from conditions
def jogger_speed_kmh : ℕ := 10
def train_speed_kmh : ℕ := 46
def initial_distance_m : ℕ := 340
def train_length_m : ℕ := 120

-- Additional computed definitions based on conditions
def relative_speed_ms : ℕ := (train_speed_kmh - jogger_speed_kmh) * 1000 / 3600
def total_distance_m : ℕ := initial_distance_m + train_length_m

-- Prove that the time it takes for the train to pass the jogger is 46 seconds
theorem train_passes_jogger_in_46_seconds : total_distance_m / relative_speed_ms = 46 := by
  sorry

end train_passes_jogger_in_46_seconds_l197_197700


namespace original_area_to_enlarged_area_l197_197256

theorem original_area_to_enlarged_area {d_original d_enlarged : ℝ} 
  (h : d_enlarged = 3 * d_original) : 
  let r_original := d_original / 2
      r_enlarged := d_enlarged / 2
      A_original := π * r_original^2
      A_enlarged := π * r_enlarged^2
  in A_original / A_enlarged = 1 / 9 := 
by 
  sorry

end original_area_to_enlarged_area_l197_197256


namespace arithmetic_sequence_8th_term_l197_197119

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197119


namespace sum_fraction_series_l197_197301

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197301


namespace probability_AB_next_to_each_other_CD_not_next_to_each_other_l197_197211

theorem probability_AB_next_to_each_other_CD_not_next_to_each_other :
  ∃ (p : ℚ), p = (4:ℚ) / 21 ∧ ∀ (students : list ℕ), 
    (length students = 7) →
    (∃ ab_pos, (students.nth ab_pos = some A) ∧ (students.nth (ab_pos+1) = some B)) ∧
    (∀ cd_pos, (abs (cd_pos - (cd_pos + 1)) ≠ 1) → 
    (students.nth cd_pos ≠ some C ∧ students.nth (cd_pos + 1) ≠ some D)) :=
sorry

end probability_AB_next_to_each_other_CD_not_next_to_each_other_l197_197211


namespace sqrt_D_is_always_odd_integer_l197_197544

variables {x : ℤ} -- Define 'x' as an integer

def a := x
def b := x + 3
def c := x * (x + 3)
def D := a^2 + b^2 + c^2

-- Theorem statement expressing that √D is always an odd integer
theorem sqrt_D_is_always_odd_integer : (sqrt D = a + b + 3) ∧ (∀ x : ℤ, (x^2 + 3 * x + 3) % 2 = 1) := sorry

end sqrt_D_is_always_odd_integer_l197_197544


namespace odd_divisors_perfect_squares_below_100_l197_197469

theorem odd_divisors_perfect_squares_below_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k * k)}.card = 9 :=
by
  sorry

end odd_divisors_perfect_squares_below_100_l197_197469


namespace louisa_average_speed_l197_197704

-- Problem statement
theorem louisa_average_speed :
  ∃ v : ℝ, (250 / v * v = 250 ∧ 350 / v * v = 350) ∧ ((350 / v) = (250 / v) + 3) ∧ v = 100 / 3 := by
  sorry

end louisa_average_speed_l197_197704


namespace max_area_rectangle_fence_l197_197187

noncomputable def optimal_area : ℝ :=
  let l_min := 80
  let w_min := 40
  let perimeter := 300
  let l := 80
  let w := perimeter / 2 - l
  in l * w

theorem max_area_rectangle_fence : optimal_area = 5600 :=
by
  have perimeter := 300
  have l_min := 80
  have w_min := 40
  have l := 80
  have w := 70 -- from the condition l + w = 150, l = 80 => w = 70
  have area := l * w -- area = 80 * 70
  show optimal_area = 80 * 70 -- corresponds to optimal_area
  calc optimal_area = 80 * 70 : by rw [optimal_area]
                  ... = 5600 : by norm_num
  sorry

end max_area_rectangle_fence_l197_197187


namespace matrix_transform_correct_l197_197815

-- Define a 3x3 matrix
def mat3 := Matrix (Fin 3) (Fin 3) ℝ

-- The specific transformation matrix we expect
def M_expected : mat3 := 
  ![![0, 1, 0], ![1, 0, 0], ![0, 0, -2]]

-- The transformation condition
def transforms (M N : mat3) : Prop := 
  M ⬝ N = ![![N 1 0, N 1 1, N 1 2], ![N 0 0, N 0 1, N 0 2], ![-2 * N 2 0, -2 * N 2 1, -2 * N 2 2]]

-- The theorem to be proven
theorem matrix_transform_correct : 
  ∃ (M : mat3), ∀ (N : mat3), transforms M N ∧ M = M_expected :=
by
  sorry

end matrix_transform_correct_l197_197815


namespace series_sum_l197_197337

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197337


namespace prob_xi_max_eq_xi1_independence_xi_max_I_l197_197031
open ProbabilityTheory

variable {Ω : Type*} {ι : Type*} [Fintype ι] [Nonempty ι]

noncomputable def xi (i : ι) : Ω → ℝ := sorry -- i.i.d random variable
noncomputable def xi_max := λ ω, Finset.univ.sup (λ i, xi i ω)

theorem prob_xi_max_eq_xi1 :
  (∀ i j, i ≠ j → xi i = xi j → 0) → 
  ∀ ω, MeasureTheory.Measure.prob (xi_max ω = xi (Fintype.fintype.choose $ λ i, true)) = (1 : ℝ) / (Fintype.card ι) :=
sorry

theorem independence_xi_max_I :
  ∀ ω, @Independence Ω ℝ _ (MeasureTheory.Measure.prob <$> xi_max) (λ w, (xi_max w = xi (Fintype.fintype.choose $ λ i, true))) :=
sorry

end prob_xi_max_eq_xi1_independence_xi_max_I_l197_197031


namespace runs_last_match_26_l197_197261

noncomputable def runs_given_in_last_match (initial_avg : ℚ) (wickets_before : ℕ) (wickets_last_match : ℕ) (decrease_avg : ℚ) : ℚ :=
  let initial_runs := initial_avg * wickets_before
  let new_avg := initial_avg - decrease_avg
  let total_wickets := wickets_before + wickets_last_match
  let new_total_runs := new_avg * total_wickets
  new_total_runs - initial_runs

theorem runs_last_match_26 (initial_avg wickets_before wickets_last_match : ℕ) (decrease_avg : ℚ) :
  initial_avg = 124 / 10 → wickets_before = 115 → wickets_last_match = 6 → decrease_avg = 4 / 10 →
  runs_given_in_last_match initial_avg wickets_before wickets_last_match decrease_avg = 26 := by
  intros h1 h2 h3 h4
  unfold runs_given_in_last_match
  rw [h1, h2, h3, h4]
  norm_num
  rw [← sub_add_eq_add_sub, add_sub_assoc]
  norm_num
  sorry

end runs_last_match_26_l197_197261


namespace pyramid_is_regular_l197_197589

variables {Pyramid Point : Type}

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def pyramid_regular (A B C D : Point) : Prop :=
  dist D A = dist D B ∧ dist D B = dist D C ∧ ∀ θ : ℕ, θ ∈ {angle D A B, angle D B C, angle D C A} -> θ = angle D A B

-- Main statement to prove
theorem pyramid_is_regular (A B C D : Point) (h1 : equilateral_triangle A B C)
  (h2 : angle D A B = angle D B C ∧ angle D B C = angle D C A ∧ angle D C A = angle D A B) : pyramid_regular A B C D :=
sorry

end pyramid_is_regular_l197_197589


namespace max_circles_tangent_bounds_l197_197192

theorem max_circles_tangent_bounds (R r : ℝ) (h : R > r) (n : ℕ) :
  (3 / 2) * (sqrt R + sqrt r) / (sqrt R - sqrt r) - 1 ≤ n ∧ 
  n ≤ (63 / 20) * (R + r) / (R - r) :=
sorry

end max_circles_tangent_bounds_l197_197192


namespace odd_divisors_perfect_squares_below_100_l197_197468

theorem odd_divisors_perfect_squares_below_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k * k)}.card = 9 :=
by
  sorry

end odd_divisors_perfect_squares_below_100_l197_197468


namespace g_sum_value_l197_197559

def g (x : ℝ) : ℝ :=
  if x > 3 then x^2 - 2 * x + 1
  else if -3 ≤ x ∧ x ≤ 3 then -x + 4
  else 5

theorem g_sum_value : g (-4) + g (0) + g (4) = 18 := by
  sorry

end g_sum_value_l197_197559


namespace gratuity_percent_correct_l197_197734

-- Define the number of investors and clients
def num_investors : ℕ := 3
def num_clients : ℕ := 3

-- Define the total bill including gratuity
def total_bill : ℝ := 720

-- Define the average cost per individual before gratuity
def cost_per_individual : ℝ := 100

-- Define the total number of individuals
def total_individuals : ℕ := num_investors + num_clients

-- Define the total cost before gratuity
def total_cost_before_gratuity : ℝ := total_individuals * cost_per_individual

-- Define the gratuity amount
def gratuity_amount : ℝ := total_bill - total_cost_before_gratuity

-- Define the gratuity percentage
def gratuity_percentage : ℝ := (gratuity_amount / total_cost_before_gratuity) * 100

-- Theorem to prove the gratuity percentage
theorem gratuity_percent_correct : gratuity_percentage = 20 := by
  sorry

end gratuity_percent_correct_l197_197734


namespace mowing_lawn_each_week_l197_197533

-- Definitions based on the conditions
def riding_speed : ℝ := 2 -- acres per hour with riding mower
def push_speed : ℝ := 1 -- acre per hour with push mower
def total_hours : ℝ := 5 -- total hours

-- The problem we want to prove
theorem mowing_lawn_each_week (A : ℝ) :
  (3 / 4) * A / riding_speed + (1 / 4) * A / push_speed = total_hours → 
  A = 15 :=
by
  sorry

end mowing_lawn_each_week_l197_197533


namespace find_ratio_l197_197786

theorem find_ratio (x y c d : ℝ) (h₁ : 4 * x - 2 * y = c) (h₂ : 5 * y - 10 * x = d) (h₃ : d ≠ 0) : c / d = 0 :=
sorry

end find_ratio_l197_197786


namespace part1_probability_expectation_X_l197_197993

-- Definitions for Part 1
def homestays : List (ℕ × ℕ) :=
  [(16, 6), (8, 16), (12, 4), (14, 10), (13, 11), (18, 10), (9, 9), (20, 12)]

def qualify_ordinary (n : ℕ) : Prop := n >= 10
def qualify_quality (n : ℕ) : Prop := n >= 10

-- Theorem statement for Part 1
theorem part1_probability :
  let selected_ordinary := filter (fun p => qualify_ordinary p.1) homestays
  let selected_quality := filter (fun p => qualify_quality p.2) selected_ordinary
  (selected_ordinary.length.choose 3 : ℚ) / (homestays.length.choose 3) *
  (selected_quality.length.choose 3 : ℚ) / (selected_ordinary.length.choose 3) = 1 / 5 :=
sorry

-- Definitions for Part 2
def qualify_ordinary_15 (n : ℕ) : Prop := n >= 15

-- Theorem statement for Part 2
theorem expectation_X :
  let selected_ordinary_15 := filter (fun p => qualify_ordinary_15 p.1) homestays
  let total_selected := homestays.length.choose 4
  let p_0 := ((selected_ordinary_15.length.choose 0) * ((homestays.length - selected_ordinary_15.length).choose 4) : ℚ) / total_selected
  let p_1 := ((selected_ordinary_15.length.choose 1) * ((homestays.length - selected_ordinary_15.length).choose 3) : ℚ) / total_selected
  let p_2 := ((selected_ordinary_15.length.choose 2) * ((homestays.length - selected_ordinary_15.length).choose 2) : ℚ) / total_selected
  let p_3 := ((selected_ordinary_15.length.choose 3) * ((homestays.length - selected_ordinary_15.length).choose 1) : ℚ) / total_selected
  (0 * p_0 + 1 * p_1 + 2 * p_2 + 3 * p_3) = 3 / 2 :=
sorry

end part1_probability_expectation_X_l197_197993


namespace CA_eq_A_intersection_CB_eq_l197_197839

-- Definitions as per conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | x > 1 }

-- Proof problems as per questions and answers
theorem CA_eq : (U \ A) = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem A_intersection_CB_eq : (A ∩ (U \ B)) = { x : ℝ | x ≤ 1 } :=
by
  sorry

end CA_eq_A_intersection_CB_eq_l197_197839


namespace percent_of_number_l197_197806

theorem percent_of_number (x : ℝ) (h : 18 = 0.75 * x) : x = 24 := by
  sorry

end percent_of_number_l197_197806


namespace pairs_of_skew_lines_in_cube_l197_197384

theorem pairs_of_skew_lines_in_cube (cube_vertices : Finset ℕ) (h_cube : cube_vertices.card = 8)
  (non_coplanar_points : ℕ) (h_non_coplanar_points : non_coplanar_points = 58)
  (skew_lines_per_tetrahedron : ℕ) (h_skew_lines_per_tetrahedron : skew_lines_per_tetrahedron = 3) :
  let pairs_of_skew_lines := non_coplanar_points * skew_lines_per_tetrahedron
  pairs_of_skew_lines = 174 := 
  by
    intro pairs_of_skew_lines
    have h_pairs : pairs_of_skew_lines = 58 * 3
      := by rw [h_non_coplanar_points, h_skew_lines_per_tetrahedron]
    rw h_pairs
    norm_num
    exact h_pairs

end pairs_of_skew_lines_in_cube_l197_197384


namespace minimum_bailing_rate_l197_197999

theorem minimum_bailing_rate (distance speed : ℝ) (water_admission_rate max_water_capacity : ℝ)
    (row_time minutes_to_reach : ℝ) :
  distance = 2 ∧ speed = 3 ∧ water_admission_rate = 8 ∧ max_water_capacity = 50 ∧ row_time = distance / speed ∧
  minutes_to_reach = row_time * 60 →
  ∀ r : ℝ, (water_admission_rate - r) * minutes_to_reach ≤ max_water_capacity → r ≥ 7 :=
by
  intros h r hr
  have : minutes_to_reach = 40 := by
    rw [← h.3, ← h.4]
    norm_num
  sorry

end minimum_bailing_rate_l197_197999


namespace steven_arrangements_l197_197358

theorem steven_arrangements : 
  (∃ (arr : Finset (List Char)), arr.card = 120 ∧ (∀ a ∈ arr, List.last a 'S' = 'E')) :=
sorry

end steven_arrangements_l197_197358


namespace least_value_f_l197_197750

theorem least_value_f {A B C D P : Type}
  (f : P → Real) (a b c : Real)
  (AD BC AC BD AB CD : Real)
  (condition1 : AD = BC ∧ AD = a)
  (condition2 : AC = BD ∧ AC = b)
  (condition3 : AB * CD = c^2) :
  ∃ P, f P = AP + BP + CP + DP :=
  ∃ A B C D P, AP + BP + CP + DP = sqrt ((a^2 + b^2 + c^2) / 2) :=
sorry

end least_value_f_l197_197750


namespace infinite_series_converges_l197_197313

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197313


namespace find_number_of_students_l197_197238

theorem find_number_of_students 
    (N T : ℕ) 
    (h1 : T = 80 * N)
    (h2 : (T - 350) / (N - 5) = 90) : 
    N = 10 :=
sorry

end find_number_of_students_l197_197238


namespace determinant_S_l197_197539

/-- Let n be a positive integer. 
For i and j in {1, 2, ..., n}, let s(i, j) be the number of pairs (a, b) of nonnegative integers satisfying a * i + b * j = n.
Let S be the n-by-n matrix whose (i, j) -entry is s(i, j). -/
def s (n i j : ℕ) : ℕ :=
  (finset.Icc 0 n).filter (λ ab, ab.1 * i + ab.2 * j = n).card

def S (n : ℕ) : matrix (fin n) (fin n) ℕ :=
  λ i j, s n (i + 1) (j + 1)

theorem determinant_S (n : ℕ) : 
  matrix.det (S n) = if n % 2 = 1 then -(n + 1) else -n :=
sorry

end determinant_S_l197_197539


namespace triangle_inscribed_angle_l197_197190

theorem triangle_inscribed_angle 
  (y : ℝ)
  (arc_PQ arc_QR arc_RP : ℝ)
  (h1 : arc_PQ = 2 * y + 40)
  (h2 : arc_QR = 3 * y + 15)
  (h3 : arc_RP = 4 * y - 40)
  (h4 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_P : ℝ, angle_P = 64.995 := 
by 
  sorry

end triangle_inscribed_angle_l197_197190


namespace min_value_sum_reciprocal_l197_197548

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l197_197548


namespace relationship_y1_relationship_y2_store_choice_1581_cost_effectiveness_l197_197255

def price_per_pair := 30
def discount_A := 0.7
def discount_B := 0.85

def cost_A (x : ℕ) : ℝ :=
  if x > 10 then price_per_pair * 10 + (price_per_pair * discount_A) * (x - 10)
  else price_per_pair * x

def cost_B (x : ℕ) : ℝ :=
  (price_per_pair * discount_B) * x

theorem relationship_y1 (x : ℕ) (h : x > 10) : cost_A x = 21 * x + 90 := by
  sorry

theorem relationship_y2 (x : ℕ) : cost_B x = 25.5 * x := by
  sorry

theorem store_choice_1581 : (cost_A 71 = 1581 ∧ cost_B 62 = 1581) ∧ (71 > 62) := by
  sorry

theorem cost_effectiveness (x : ℕ) (h : x > 12) : 
    (12 < x ∧ x < 20 → cost_B x < cost_A x) ∧ (x > 20 → cost_A x < cost_B x) := by
  sorry

end relationship_y1_relationship_y2_store_choice_1581_cost_effectiveness_l197_197255


namespace min_value_sum_reciprocal_l197_197549

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l197_197549


namespace largest_angle_in_pentagon_l197_197194

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
    (hA : A = 60) 
    (hB : B = 85) 
    (hCD : C = D) 
    (hE : E = 2 * C + 15) 
    (sum_angles : A + B + C + D + E = 540) : 
    E = 205 := 
by 
    sorry

end largest_angle_in_pentagon_l197_197194


namespace min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l197_197719

-- Problem (Ⅰ)
theorem min_value_f1 (x : ℝ) (h : x > 0) : (12 / x + 3 * x) ≥ 12 :=
sorry

theorem min_value_f1_achieved : (12 / 2 + 3 * 2) = 12 :=
by norm_num

-- Problem (Ⅱ)
theorem max_value_f2 (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

theorem max_value_f2_achieved : (1 / 6) * (1 - 3 * (1 / 6)) = 1 / 12 :=
by norm_num

end min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l197_197719


namespace simplify_expression_l197_197543

variables {a b : ℝ}

-- Define the conditions
def condition (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a^4 + b^4 = a + b)

-- Define the target goal
def goal (a b : ℝ) : Prop := 
  (a / b + b / a - 1 / (a * b^2)) = (-a - b) / (a * b^2)

-- Statement of the theorem
theorem simplify_expression (h : condition a b) : goal a b :=
by 
  sorry

end simplify_expression_l197_197543


namespace circles_intersect_l197_197166

def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem circles_intersect :
  (∃ (x y : ℝ), circle_eq1 x y ∧ circle_eq2 x y) :=
sorry

end circles_intersect_l197_197166


namespace relationship_among_values_l197_197417

-- Define the properties of the function f
variables (f : ℝ → ℝ)

-- Assume necessary conditions
axiom domain_of_f : ∀ x : ℝ, f x ≠ 0 -- Domain of f is ℝ
axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_function : ∀ x y : ℝ, (0 ≤ x) → (x ≤ y) → (f x ≤ f y) -- f is increasing for x in [0, + ∞)

-- Define the main theorem based on the problem statement
theorem relationship_among_values : f π > f (-3) ∧ f (-3) > f (-2) :=
by
  sorry

end relationship_among_values_l197_197417


namespace decreasing_interval_g_min_value_a_no_zeros_l197_197875

open Real

-- Define the functions from the given conditions
noncomputable def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x
noncomputable def g (a x : ℝ) : ℝ := f a x + x

-- Lean statement for the first problem
theorem decreasing_interval_g :
  ∃ (a : ℝ), (∀ x ∈ Ioo 0 2, (3 - a - 2 / x) < 0) :=
sorry

-- Lean statement for the second problem
theorem min_value_a_no_zeros :
  ∃ (a : ℝ), (a ≥ 2 - 4 * log 2) ∧ (∀ x ∈ Ioo 0 (1/2), f a x > 0) :=
sorry

end decreasing_interval_g_min_value_a_no_zeros_l197_197875


namespace game_winning_strategy_l197_197250

-- Define the game setup
def initial_tuple (n : ℕ) : ℕ × ℕ := (n, 0)

def set_S : set (ℤ × ℤ) := {(-1, 2), (-1, 0)}

-- Define the condition for valid moves
def valid_move (current_tuple new_tuple : ℤ × ℤ) : Prop :=
  ∃ (b : ℤ × ℤ), b ∈ set_S ∧ new_tuple = (current_tuple.1 + b.1, current_tuple.2 + b.2)

-- Define what it means for a player to lose
def loses (tuple : ℤ × ℤ) : Prop :=
  tuple.1 < 0 ∨ tuple.2 < 0

-- Define the winning condition for the first player based on the initial value of n
theorem game_winning_strategy (n : ℕ) :
  (∃ k : ℕ, k = 2) ∧ (∃ S : set (ℤ × ℤ), S = set_S) ∧
  ((∃ move_strategy : ℕ → ℤ × ℤ → ℤ × ℤ,
    (∀ m : ℕ, ¬loses (move_strategy m (initial_tuple n))) ∧
    (n = 2 ^ n → (∃ move_strategy : ℕ → ℤ × ℤ → ℤ × ℤ,
        ∀ m : ℕ, ¬loses (move_strategy m (initial_tuple n))))) ∧
    (¬(n = 2 ^ n) → (∃ move_strategy : ℕ → ℤ × ℤ → ℤ × ℤ,
        ∀ m : ℕ, loses (move_strategy m (initial_tuple n))))) :=
sorry

end game_winning_strategy_l197_197250


namespace garden_strawberry_area_l197_197952

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l197_197952


namespace eccentricity_of_ellipse_l197_197737

noncomputable def eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (l : ℝ → ℝ) (hl : l 0 = 0)
  (h_intersects : ∃ M N : ℝ × ℝ, M ≠ N ∧ (M.1 / a)^2 + (M.2 / b)^2 = 1 ∧ (N.1 / a)^2 + (N.2 / b)^2 = 1 ∧ l M.1 = M.2 ∧ l N.1 = N.2)
  (P : ℝ × ℝ) (hP : (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ P ≠ (0, 0))
  (h_product_slopes : ∀ (Mx Nx Px : ℝ) (k : ℝ),
    l Mx = k * Mx →
    l Nx = k * Nx →
    l Px ≠ k * Px →
    ((k * Mx - P.2) / (Mx - P.1)) * ((k * Nx - P.2) / (Nx - P.1)) = -1/3) :
  eccentricity a b h1 h2 = Real.sqrt (2 / 3) :=
by
  sorry

end eccentricity_of_ellipse_l197_197737


namespace abs_neg_three_l197_197125

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l197_197125


namespace detect_fake_coins_in_3_weighings_l197_197176

theorem detect_fake_coins_in_3_weighings
    (n: Nat) (coins: Fin n → ℝ)
    (h1: n = 1000)
    (h2: ∃ k, k ≤ 2 ∧ 
              ∀ i j, i ≠ j → 
              (coins i = coins j ∨ (coins i ≠ coins j ∧ (coins i = coins 0 ∨ coins j = coins 0))))
    (h3: ∀ i j, coins i = coins j ∨ (coins i ≠ coins j ∧ (coins i = coins 0 ∨ coins j = coins 0)))
   : ∃ three_weighings: Fin 3 → List (List (Fin n)), 
     (∀ i, List.length (three_weighings i) = 2) ∧ 
     (∃ fake_detected: Bool, 
       (if fake_detected then 
        ∃ lighter: Bool, -- if detected, whether it's lighter
        true
        else
        true)) :=
sorry

end detect_fake_coins_in_3_weighings_l197_197176


namespace dice_probability_l197_197225

noncomputable def probability_same_face (throws : ℕ) (dice : ℕ) : ℚ :=
  1 - (1 - (1 / 6) ^ dice) ^ throws

theorem dice_probability : 
  probability_same_face 5 10 = 1 - (1 - (1 / 6) ^ 10) ^ 5 :=
by 
  sorry

end dice_probability_l197_197225


namespace sum_max_min_on_interval_l197_197913

open Real

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

theorem sum_max_min_on_interval : 
  let M := sup (f '' (set.Icc (-1 : ℝ) 3))
  let N := inf (f '' (set.Icc (-1 : ℝ) 3))
  M + N = 39 / 4 :=
by
  sorry

end sum_max_min_on_interval_l197_197913


namespace find_constants_to_divide_l197_197965

universe u
variables {V : Type u} [AddCommGroup V] [Module ℝ V]
-- Conditions
variables (C D Q : V)
variable (hCQ_QD_ratio : ∃ (a b : ℝ), a / b = 3 / 5 ∧ 
  (∀ R : V, R = a / (a + b) • C + b / (a + b) • D ↔ R = Q))

theorem find_constants_to_divide : 
  ∃ (r s : ℝ), (r = 5 / 8 ∧ s = 3 / 8) ∧ Q = r • C + s • D :=
begin
  use [5/8, 3/8],
  split,
  { split; refl },
  { sorry }
end

end find_constants_to_divide_l197_197965


namespace infinite_curious_numbers_l197_197563

def is_curious (N : ℕ) : Prop :=
  (N * 9).digitRev = N

def curious_sequence (k : ℕ) : ℕ :=
  10 ^ (k + 3) + 9 * (10 ^ (k + 2) + 10 ^ (k + 1) + 10 ^ k - 1) / 9 + 8 * 10 + 1

theorem infinite_curious_numbers :
  ∀ k : ℕ, is_curious (curious_sequence k) :=
by
  sorry

end infinite_curious_numbers_l197_197563


namespace gcd_lcm_of_consecutive_naturals_l197_197858

theorem gcd_lcm_of_consecutive_naturals (m : ℕ) (h : m > 0) (n : ℕ) (hn : n = m + 1) :
  gcd m n = 1 ∧ lcm m n = m * n :=
by
  sorry

end gcd_lcm_of_consecutive_naturals_l197_197858


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197674

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197674


namespace shift_left_l197_197879

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - π / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_left : ∀ x, f (x + π / 12) = g x := 
by
  intro x
  simp [f, g, Real.sin]
  sorry

end shift_left_l197_197879


namespace simplify_expression_equals_100_l197_197699

def a : ℚ := 3 + 4 / 7
def x : ℚ := 28 / 100

theorem simplify_expression_equals_100 : 
  let expr := (-4 * a^3 * Real.sqrt (Real.sqrt (a * x) / a^2))^3 + 
              (-10 * a * Real.sqrt x * Real.sqrt ((a * x)^(-1)))^2 + 
              (-2 * (Real.cbrt (a^4 * Real.sqrt (x / a)))^2)^3 in
  expr = 100 := 
by 
  sorry  -- Placeholder for the actual proof steps

end simplify_expression_equals_100_l197_197699


namespace algebra_problem_l197_197862

open Classical

noncomputable def expr1 (x y : ℝ) (b : ℕ) : ℝ :=
- x * y ^ (b - 1)

noncomputable def expr2 (x y : ℝ) (a : ℕ) : ℝ :=
3 * x ^ (a + 2) * y ^ 3

theorem algebra_problem (a b : ℕ) (x y : ℝ) :
  (expr1 x y b - expr2 x y a = c * x ^ d * y ^ e) → 
  (a = -1) ∧ 
  (b = 4) → 
  a ^ b = 1 := 
sorry

end algebra_problem_l197_197862


namespace summer_discount_percentage_l197_197136

/--
Given:
1. The original cost of the jeans (original_price) is $49.
2. On Wednesdays, there is an additional $10.00 off on all jeans after the summer discount is applied.
3. Before the sales tax applies, the cost of a pair of jeans (final_price) is $14.50.

Prove:
The summer discount percentage (D) is 50%.
-/
theorem summer_discount_percentage (original_price final_price : ℝ) (D : ℝ) :
  original_price = 49 → 
  final_price = 14.50 → 
  (original_price - (original_price * D / 100) - 10 = final_price) → 
  D = 50 :=
by intros h_original h_final h_discount; sorry

end summer_discount_percentage_l197_197136


namespace exists_strictly_increasing_sequence_l197_197356

theorem exists_strictly_increasing_sequence 
  (N : ℕ) : 
  (∃ (t : ℕ), t^2 ≤ N ∧ N < t^2 + t) →
  (∃ (s : ℕ → ℕ), (∀ n : ℕ, s n < s (n + 1)) ∧ 
   (∃ k : ℕ, ∀ n : ℕ, s (n + 1) - s n = k) ∧
   (∀ n : ℕ, s (s n) - s (s (n - 1)) ≤ N 
      ∧ N < s (1 + s n) - s (s (n - 1)))) :=
by
  sorry

end exists_strictly_increasing_sequence_l197_197356


namespace arithmetic_sequence_eighth_term_l197_197096

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197096


namespace eliot_account_balance_l197_197237

theorem eliot_account_balance 
  (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 :=
by 
  sorry

end eliot_account_balance_l197_197237


namespace part1_part2_part3_l197_197039

open Set

variable (x : ℝ)

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}

theorem part1 : A ∩ B = {x | 3 ≤ x ∧ x < 7} :=
sorry

theorem part2 : (Aᶜ : Set ℝ) = {x | x < 3 ∨ x ≥ 7} :=
sorry

theorem part3 : (A ∪ B)ᶜ = {x | x ≤ 2 ∨ x ≥ 10} :=
sorry

end part1_part2_part3_l197_197039


namespace steps_in_staircase_using_210_toothpicks_l197_197350

-- Define the conditions
def first_step : Nat := 3
def increment : Nat := 2
def total_toothpicks_5_steps : Nat := 55

-- Define required theorem
theorem steps_in_staircase_using_210_toothpicks : ∃ (n : ℕ), (n * (n + 2) = 210) ∧ n = 13 :=
by
  sorry

end steps_in_staircase_using_210_toothpicks_l197_197350


namespace slope_asymptotes_of_hyperbola_l197_197035

noncomputable theory

open Real

variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def slope_of_line (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

def point_A (x y : ℝ) : Prop := hyperbola a b x y ∧ y = 2 * b

theorem slope_asymptotes_of_hyperbola :
  (∃ (x : ℝ), point_A a b x (2*b)) → slope_of_line (0, 0) (-(sqrt 5 * a), 2 * b) = -1 →
  (1 / a : ℝ) = (sqrt 5 / (2 * b)) :=
by
  sorry

end slope_asymptotes_of_hyperbola_l197_197035


namespace max_k_for_10_pow_divides_90_fact_l197_197481

-- Define the mathematical conditions and the final proof statement.
theorem max_k_for_10_pow_divides_90_fact : 
  ∃ k : ℕ, (90.factorial % (10^k) = 0) ∧ (∀ m > k, 90.factorial % (10^m) ≠ 0) :=
sorry

end max_k_for_10_pow_divides_90_fact_l197_197481


namespace smallest_common_multiple_gt_50_l197_197213

theorem smallest_common_multiple_gt_50 (a b : ℕ) (h1 : a = 15) (h2 : b = 20) : 
    ∃ x, x > 50 ∧ Nat.lcm a b = x := by
  have h_lcm : Nat.lcm a b = 60 := by sorry
  use 60
  exact ⟨by decide, h_lcm⟩

end smallest_common_multiple_gt_50_l197_197213


namespace minimum_log_function_l197_197427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (3 - x) + log a (x + 1)

theorem minimum_log_function (a : ℝ) (h : 0 < a ∧ a < 1) 
  (hx : ∀ x : ℝ, -1 < x ∧ x < 3 → f a x ≥ -2 ) : a = 1 / 2 :=
sorry

end minimum_log_function_l197_197427


namespace count_valid_selections_l197_197801

-- Conditions
def circumference := 24
def segment_points := 24
def chosen_points := 8
def forbidden_arcs := [3, 8]

-- Problem Statement
theorem count_valid_selections : 
  ∃ (selections : Finset (Finset (Fin 24))),
    selections.card = 258 ∧ 
    ∀ sel ∈ selections, sel.card = chosen_points ∧ 
    ∀ p1 ∈ sel, ∀ p2 ∈ sel, 
      p1 ≠ p2 → 
      let dist := (p2 - p1 + 24) % 24 in 
      dist ∉ forbidden_arcs :=
sorry

end count_valid_selections_l197_197801


namespace calculate_expression_l197_197597

variable (x y : ℝ)

theorem calculate_expression (h1 : x + y = 5) (h2 : x * y = 3) : 
   x + (x^4 / y^3) + (y^4 / x^3) + y = 27665 / 27 :=
by
  sorry

end calculate_expression_l197_197597


namespace floor_exponents_eq_l197_197955

theorem floor_exponents_eq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_inf_k : ∃ᶠ k in at_top, ∃ (k : ℕ), ⌊a ^ k⌋ + ⌊b ^ k⌋ = ⌊a⌋ ^ k + ⌊b⌋ ^ k) :
  ⌊a ^ 2014⌋ + ⌊b ^ 2014⌋ = ⌊a⌋ ^ 2014 + ⌊b⌋ ^ 2014 := by
  sorry

end floor_exponents_eq_l197_197955


namespace consecutive_vertices_product_l197_197349

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l197_197349


namespace sin_neg_135_eq_neg_sqrt_2_over_2_l197_197779

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_135_eq_neg_sqrt_2_over_2_l197_197779


namespace sum_naturals_formula_l197_197805

noncomputable def sum_naturals : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_naturals n + (n + 1)

theorem sum_naturals_formula (n : ℕ) :
  sum_naturals n = n * (n + 1) / 2 :=
sorry

end sum_naturals_formula_l197_197805


namespace multiplier_for_average_grade_l197_197979

/-- Conditions -/
def num_of_grades_2 : ℕ := 3
def num_of_grades_3 : ℕ := 4
def num_of_grades_4 : ℕ := 1
def num_of_grades_5 : ℕ := 1
def cash_reward : ℕ := 15

-- Definitions for sums and averages based on the conditions
def sum_of_grades : ℕ :=
  num_of_grades_2 * 2 + num_of_grades_3 * 3 + num_of_grades_4 * 4 + num_of_grades_5 * 5

def total_grades : ℕ :=
  num_of_grades_2 + num_of_grades_3 + num_of_grades_4 + num_of_grades_5

def average_grade : ℕ :=
  sum_of_grades / total_grades

/-- Proof statement -/
theorem multiplier_for_average_grade : cash_reward / average_grade = 5 := by
  sorry

end multiplier_for_average_grade_l197_197979


namespace number_of_ways_to_form_team_l197_197574

-- Define the problem conditions
def num_boys : ℕ := 7
def num_girls : ℕ := 9
def team_size : ℕ := 6

-- Define the requirement for at least 2 boys
def at_least_two_boys (num_boys_selected : ℕ) : Prop :=
  num_boys_selected >= 2

-- Our main theorem statement
theorem number_of_ways_to_form_team :
  ∃ ways : ℕ, ways = 6846 ∧ 
    (∃ num_boys_selected : ℕ, at_least_two_boys(num_boys_selected) ∧
      (num_boys_selected + (team_size - num_boys_selected) = team_size) ∧
      (num_boys_selected <= num_boys) ∧
      ((team_size - num_boys_selected) <= num_girls)) :=
sorry

end number_of_ways_to_form_team_l197_197574


namespace count_valid_c_l197_197821

theorem count_valid_c : ∃ (count : ℕ), count = 670 ∧ 
  ∀ (c : ℤ), (-2007 ≤ c ∧ c ≤ 2007) → 
    (∃ (x : ℤ), (x^2 + c) % (2^2007) = 0) ↔ count = 670 :=
sorry

end count_valid_c_l197_197821


namespace probability_non_edge_unit_square_l197_197251

theorem probability_non_edge_unit_square : 
  let total_squares := 100
  let perimeter_squares := 36
  let non_perimeter_squares := total_squares - perimeter_squares
  let probability := (non_perimeter_squares : ℚ) / total_squares
  probability = 16 / 25 :=
by
  sorry

end probability_non_edge_unit_square_l197_197251


namespace distinct_combinations_of_three_numbers_prime_condition_and_divisors_l197_197243

-- Definition of numbers being in the set and greater than n/2
def numbers_in_set_and_greater_half (n : ℕ) (a b c : ℕ) : Prop :=
  2 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ a > n / 2 ∧ b > n / 2 ∧ c > n / 2

-- The first proof statement
theorem distinct_combinations_of_three_numbers (n a b c : ℕ)
  (h : numbers_in_set_and_greater_half n a b c) :
  a + b + c ≠ a * b + c ∧
  a + b + c ≠ a * c + b ∧
  a + b + c ≠ b * c + a ∧
  a + b + c ≠ a * (b + c) ∧
  a + b + c ≠ b * (a + c) ∧
  a + b + c ≠ c * (a + b) ∧
  a + b + c ≠ a * b * c ∧
  a * b + c ≠ a * c + b ∧
  a * b + c ≠ b * c + a ∧
  a * b + c ≠ a * (b + c) ∧
  a * b + c ≠ b * (a + c) ∧
  a * b + c ≠ c * (a + b) ∧
  a * b + c ≠ a * b * c ∧
  a * c + b ≠ b * c + a ∧
  a * c + b ≠ a * (b + c) ∧
  a * c + b ≠ b * (a + c) ∧
  a * c + b ≠ c * (a + b) ∧
  a * c + b ≠ a * b * c ∧
  b * c + a ≠ a * (b + c) ∧
  b * c + a ≠ b * (a + c) ∧
  b * c + a ≠ c * (a + b) ∧
  b * c + a ≠ a * b * c ∧
  a * (b + c) ≠ b * (a + c) ∧
  a * (b + c) ≠ c * (a + b) ∧
  a * (b + c) ≠ a * b * c ∧
  b * (a + c) ≠ c * (a + b) ∧
  b * (a + c) ≠ a * b * c ∧
  c * (a + b) ≠ a * b * c :=
sorry

-- The second proof statement
theorem prime_condition_and_divisors (n p : ℕ)
  (hp : p.prime)
  (hpn : p ≤ nat.sqrt n)
  (three_numbers : list ℕ)
  (hselect : ∀ a ∈ three_numbers, ∃ b c, 
    three_numbers = [p, b, c] ∧ 
    b > c ∧
    a * b + c = a * (b + c) ∨ b * c + a = a * b * c ∨ a * (b + c) = c * (a + b)) :
  ∃ (k : ℕ), k = nat.divisors (p - 1).card :=
sorry

end distinct_combinations_of_three_numbers_prime_condition_and_divisors_l197_197243


namespace fewest_coach_handshakes_l197_197772

theorem fewest_coach_handshakes (n_A n_B k_A k_B : ℕ) (h1 : n_A = n_B + 2)
    (h2 : ((n_A * (n_A - 1)) / 2) + ((n_B * (n_B - 1)) / 2) + (n_A * n_B) + k_A + k_B = 620) :
  k_A + k_B = 189 := 
sorry

end fewest_coach_handshakes_l197_197772


namespace binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l197_197028

-- Definition of power of two
def is_power_of_two (n : ℕ) := ∃ m : ℕ, n = 2^m

-- Theorems to be proven
theorem binom_even_if_power_of_two (n : ℕ) (h : is_power_of_two n) :
  ∀ k : ℕ, 1 ≤ k ∧ k < n → Nat.choose n k % 2 = 0 := sorry

theorem binom_odd_if_not_power_of_two (n : ℕ) (h : ¬ is_power_of_two n) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ Nat.choose n k % 2 = 1 := sorry

end binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l197_197028


namespace ratio_percent_l197_197729

theorem ratio_percent (x : ℕ) (h : (15 / x : ℚ) = 60 / 100) : x = 25 := 
sorry

end ratio_percent_l197_197729


namespace arithmetic_sequence_8th_term_is_71_l197_197099

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197099


namespace cos_C_of_isosceles_triangle_l197_197494

theorem cos_C_of_isosceles_triangle (A B C : ℝ) (a b c : ℝ) (hAeqB : A = B) (ha : a = 3) (hc : c = 2) : 
  cos C = 7 / 9 := by
  -- Definitions
  let b : ℝ := 3 -- since A = B, and a = 3
  -- Use the cosine rule
  have hcos : cos C = (a^2 + b^2 - c^2) / (2 * a * b),
  -- Substitute the values
  rw [ha, hc], -- rw with right hand side
  have hab : a = 3 := ha,
  have hb : b = 3 := by rfl,
  sorry

end cos_C_of_isosceles_triangle_l197_197494


namespace better_quality_l197_197181

open ProbabilityTheory

def machine_quality (A B : Type) [ProbabilitySpace A] [ProbabilitySpace B]
  (ξ1 : A → ℝ) (ξ2 : B → ℝ) : Prop :=
  ∃ (E_ξ1 E_ξ2 : ℝ) (D_ξ1 D_ξ2 : ℝ),
    (Expectation ξ1 = E_ξ1 ∧ Expectation ξ2 = E_ξ2) ∧
    (Variance ξ1 = D_ξ1 ∧ Variance ξ2 = D_ξ2) ∧ E_ξ1 = E_ξ2 ∧ D_ξ1 > D_ξ2

theorem better_quality :
  ∀ (A B : Type) [ProbabilitySpace A] [ProbabilitySpace B]
    (ξ1 : A → ℝ) (ξ2 : B → ℝ),
    machine_quality A B ξ1 ξ2 → 
    ∃ (B_quality : Bool), B_quality = true :=
by
  intro A B _ _ ξ1 ξ2 h
  cases h with E_ξ1 h
  cases h with E_ξ2 h
  cases h with D_ξ1 h
  cases h with D_ξ2 h
  exists true
  sorry

end better_quality_l197_197181


namespace max_good_set_integer_l197_197408

namespace proof

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2016 }

def is_good_set (X : Set ℕ) : Prop :=
  ∃ x y, x ∈ X ∧ y ∈ X ∧ x < y ∧ x ∣ y

def is_target_subset (X : Set ℕ) : Prop :=
  Set.card X = 1008 ∧
  ∀ x, x ∈ X → (∃ y, y ∈ X ∧ x < y ∧ x ∣ y)

theorem max_good_set_integer :
  (∀ X : Set ℕ, Set.card X = 1008 → 671 ∈ X → is_good_set X) ∧
  (∀ a : ℕ, a > 671 → ∃ X : Set ℕ, Set.card X = 1008 ∧ a ∈ X ∧ ¬ is_good_set X) :=
sorry

end proof

end max_good_set_integer_l197_197408


namespace sum_fraction_series_l197_197299

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197299


namespace toms_animal_robots_l197_197978

theorem toms_animal_robots (h : ∀ (m t : ℕ), t = 2 * m) (hmichael : 8 = m) : ∃ t, t = 16 := 
by
  sorry

end toms_animal_robots_l197_197978


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197663

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197663


namespace series_sum_l197_197340

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197340


namespace general_term_formula_sum_of_first_n_terms_l197_197887

noncomputable def seq : ℕ → ℤ
| 1       := 1
| 2       := 3
| 3       := 7
| (n + 1) := 2 * seq n + 1

example : seq 1 = 1 ∧ seq 2 = 3 ∧ seq 3 = 7 ∧ seq 4 = 15 :=
by {
  split,
  { dsimp [seq], refl },
  split,
  { dsimp [seq], refl },
  split,
  { dsimp [seq], refl },
  { dsimp [seq], refl }
}

theorem general_term_formula (n : ℕ) : seq n = 2^n - 1 :=
sorry

theorem sum_of_first_n_terms (n : ℕ) : 
  (∑ i in Finset.range n, seq (i + 1)) = 2^(n + 1) - n - 2 :=
sorry

end general_term_formula_sum_of_first_n_terms_l197_197887


namespace arithmetic_sequence_8th_term_is_71_l197_197105

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197105


namespace polygon_vertices_product_at_least_2014_l197_197346

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l197_197346


namespace greatest_four_digit_number_divisible_by_six_l197_197203

theorem greatest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split, { norm_num },
  split, { norm_num },
  split,
  { norm_num }, -- proving 9996 % 6 = 0 
  { intros m hm1 hm2 hm3, -- proving no larger number satisfies the conditions
    sorry } -- proof of this statement would go here
end

end greatest_four_digit_number_divisible_by_six_l197_197203


namespace range_of_a_for_monotonic_intervals_l197_197847

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * a * x^3 + x^2 + a * x + 1

theorem range_of_a_for_monotonic_intervals (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
  f(a)(x) = 0 ∧ f(a)(y) = 0 ∧ x ≠ 0 ∧ y ≠ 0) ↔ a ∈ set.Ioo (-1:ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

end range_of_a_for_monotonic_intervals_l197_197847


namespace f_expression_decreasing_interval_correct_l197_197877

noncomputable def max_point : ℝ := π / 12
noncomputable def min_point : ℝ := 7 * π / 12
noncomputable def A : ℝ := 3
noncomputable def ω : ℝ := 2
noncomputable def φ : ℝ := π / 3

-- Define the function
noncomputable def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Given conditions
axiom A_pos : A > 0
axiom ω_pos : ω > 0
axiom φ_bound : |φ| < π
axiom f_max : f max_point = A
axiom f_min : f min_point = -A

-- Theorem: the analytical expression is correct
theorem f_expression : f (x : ℝ) = 3 * Real.sin (2 * x + π / 3) :=
sorry

-- Define the decreasing interval
def decreasing_interval (k : ℤ) : Set ℝ := {x | k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12}

-- Theorem: the monotonically decreasing interval is correct
theorem decreasing_interval_correct (x : ℝ) (k : ℤ) : x ∈ decreasing_interval k ↔ k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12 :=
sorry

end f_expression_decreasing_interval_correct_l197_197877


namespace joel_strawberries_area_l197_197950

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end joel_strawberries_area_l197_197950


namespace solve_equation_l197_197366

open Real

noncomputable def verify_solution (x : ℝ) : Prop :=
  1 / ((x - 3) * (x - 4)) +
  1 / ((x - 4) * (x - 5)) +
  1 / ((x - 5) * (x - 6)) = 1 / 8

theorem solve_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6) :
  verify_solution x ↔ (x = (9 + sqrt 57) / 2 ∨ x = (9 - sqrt 57) / 2) := 
by
  sorry

end solve_equation_l197_197366


namespace complete_magic_square_l197_197286

def is_magic_square (square : ℕ → ℕ → ℕ) (n : ℕ) :=
  (∀ i, ∑ j in finset.range n, square i j = 15) ∧                     -- each row sums to 15
  (∀ j, ∑ i in finset.range n, square i j = 15) ∧                     -- each column sums to 15
  (∑ i in finset.range n, square i i = 15) ∧                          -- main diagonal sums to 15
  (∑ i in finset.range n, square i (n - i - 1) = 15)                  -- anti-diagonal sums to 15

theorem complete_magic_square :
  ∃ (square : ℕ → ℕ → ℕ), 
    (square 0 0 = 12) ∧ (square 0 2 = 4) ∧
    (square 1 0 = 7)  ∧ 
    (square 2 1 = 1)  ∧ 
    is_magic_square square 3 :=
sorry

end complete_magic_square_l197_197286


namespace triangle_bc_length_l197_197558

-- Define the problem in Lean
theorem triangle_bc_length :
  ∀ (A B C D E F : Point) (BC : ℝ),
  angle A B C = 90 ∧
  (foot_of_altitude A BC D) ∧
  (foot_of_angle_bisector A B C E) ∧
  (foot_of_median A BC F) ∧
  dist D E = 3 ∧
  dist E F = 5 →
  dist B C = 20 :=
by
  intros A B C D E F BC h
  sorry

end triangle_bc_length_l197_197558


namespace arithmetic_seq_8th_term_l197_197074

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197074


namespace sum_fraction_series_l197_197298

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197298


namespace parabola_intersection_sum_l197_197152

theorem parabola_intersection_sum : 
  ∃ x_0 y_0 : ℝ, (y_0 = x_0^2 + 15 * x_0 + 32) ∧ (x_0 = y_0^2 + 49 * y_0 + 593) ∧ (x_0 + y_0 = -33) :=
by
  sorry

end parabola_intersection_sum_l197_197152


namespace tan_22_5_decomposition_l197_197161

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  tan (22.5 * Real.pi / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d ∧
  a + b + c + d = 3 :=
by
  sorry

end tan_22_5_decomposition_l197_197161


namespace S1_S2_inequality_l197_197026

noncomputable def S_1 (n : ℕ) : ℕ :=
  ∑ i in finset.range(n + 1),
    (finset.filter (λ d, d % 2 = 1) (finset.divisors i)).card

noncomputable def S_2 (n : ℕ) : ℕ :=
  ∑ i in finset.range(n + 1),
    (finset.filter (λ d, d % 2 = 0) (finset.divisors i)).card

theorem S1_S2_inequality (n : ℕ) : 
  |(S_1 n : ℝ) - (S_2 n) - (n : ℝ) * Real.log 2| < Real.sqrt (n) + 1 := 
sorry

end S1_S2_inequality_l197_197026


namespace arithmetic_sequence_terms_l197_197357

-- Math proof problem definition
theorem arithmetic_sequence_terms (k : ℤ) : 
  ∀ (a₁ d n : ℤ), a₁ = -3 → d = 2 → (a₁ + (n - 1) * d) = 2k - 1 → n = k + 2 := 
by 
  intros _ _ _ h₁ h₂ h₃
  sorry

end arithmetic_sequence_terms_l197_197357


namespace combined_age_71_in_6_years_l197_197495

-- Given conditions
variable (combinedAgeIn15Years : ℕ) (h_condition : combinedAgeIn15Years = 107)

-- Define the question
def combinedAgeIn6Years : ℕ := combinedAgeIn15Years - 4 * (15 - 6)

-- State the theorem to prove the question == answer given conditions
theorem combined_age_71_in_6_years (h_condition : combinedAgeIn15Years = 107) : combinedAgeIn6Years combinedAgeIn15Years = 71 := 
by 
  sorry

end combined_age_71_in_6_years_l197_197495


namespace number_of_distinguishable_icosahedrons_l197_197656

noncomputable def distinguishable_constructions_of_icosahedron : Nat :=
  11! / 5

theorem number_of_distinguishable_icosahedrons :
  distinguishable_constructions_of_icosahedron = 7983360 := sorry

end number_of_distinguishable_icosahedrons_l197_197656


namespace number_of_positive_integers_l197_197829

theorem number_of_positive_integers (n : ℕ) : 
  (0 < n ∧ n < 36 ∧ (∃ k : ℕ, n = k * (36 - k))) → 
  n = 18 ∨ n = 24 ∨ n = 30 ∨ n = 32 ∨ n = 34 ∨ n = 35 :=
sorry

end number_of_positive_integers_l197_197829


namespace largest_of_numbers_l197_197273

theorem largest_of_numbers (a b c d : ℝ) (hₐ : a = 0) (h_b : b = -1) (h_c : c = -2) (h_d : d = Real.sqrt 3) :
  d = Real.sqrt 3 ∧ d > a ∧ d > b ∧ d > c :=
by
  -- Using sorry to skip the proof
  sorry

end largest_of_numbers_l197_197273


namespace equation_holds_iff_b_eq_c_l197_197924

theorem equation_holds_iff_b_eq_c (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a * a + 100 * a + b + c ↔ b = c :=
by sorry

end equation_holds_iff_b_eq_c_l197_197924


namespace hyperbola_eccentricity_l197_197884

variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1))
variable (h_distance : b * c / Math.sqrt(a^2 + b^2) = (Math.sqrt 3) / 2 * c)

theorem hyperbola_eccentricity : 
  (Math.sqrt (1 + b^2 / a^2)) = 2 := 
sorry

end hyperbola_eccentricity_l197_197884


namespace Kim_nail_polishes_l197_197006

-- Define the conditions
variable (K : ℕ)
def Heidi_nail_polishes (K : ℕ) : ℕ := K + 5
def Karen_nail_polishes (K : ℕ) : ℕ := K - 4

-- The main statement to prove
theorem Kim_nail_polishes (K : ℕ) (H : Heidi_nail_polishes K + Karen_nail_polishes K = 25) : K = 12 := by
  sorry

end Kim_nail_polishes_l197_197006


namespace magnitude_sum_l197_197441

variables {R : Type*} [inner_product_space ℝ R]

variables (a b : R)
variables (h₁ : inner_product_space.is_orthonormal some_fun a b)
variables (h₂ : ∥a∥ = 1)
variables (h₃ : ∥b∥ = 2)

theorem magnitude_sum (h₁ : inner_product_space.is_orthonormal some_fun a b) (h₂ : ∥a∥ = 1) (h₃ : ∥b∥ = 2) : 
  ∥a + b∥ = 3 :=
begin
  -- proof steps will go here
  sorry
end

end magnitude_sum_l197_197441


namespace second_certificate_interest_rate_approx_l197_197277

-- Define initial investment
def initial_investment : ℝ := 8000
-- Define first certificate's annual interest rate
def first_annual_interest_rate : ℝ := 10 / 100
-- Define duration of each investment in months
def duration_months : ℕ := 4
-- Total amount after all investments
def final_amount : ℝ := 8840

-- Calculate the value after the first 4-month certificate with simple interest rate
def first_certificate_value : ℝ :=
  initial_investment * (1 + (first_annual_interest_rate * duration_months / 12))

-- Define the annual interest rate of the second certificate
noncomputable def second_annual_interest_rate (s : ℝ) : Prop :=
  first_certificate_value * (1 + (s / 3 / 100)) = final_amount

-- State the theorem that the second interest rate is approximately 20.79%
theorem second_certificate_interest_rate_approx : ∃ s : ℝ, second_annual_interest_rate s ∧ abs (s - 20.79) < 0.01 :=
by
  sorry

end second_certificate_interest_rate_approx_l197_197277


namespace sum_series_l197_197325

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197325


namespace number_of_integers_satisfying_abs_x_lt_4pi_l197_197456

theorem number_of_integers_satisfying_abs_x_lt_4pi : 
    ∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, (abs x < 4 * Real.pi) → x ∈ Icc (-12:ℤ) (12:ℤ) := by
  sorry

end number_of_integers_satisfying_abs_x_lt_4pi_l197_197456


namespace sum_infinite_series_eq_l197_197289

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197289


namespace sequence_properties_l197_197521

theorem sequence_properties
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2) ∧ (a 2 = 3) ∧
  (∀ n, a (2 * n - 1) = 2 * (2 * n - 1)) ∧
  (∀ n, a (2 * n) = 2 * 2 * n - 1) :=
by
  sorry

end sequence_properties_l197_197521


namespace rectangle_area_l197_197145

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 250) : l * w = 2500 :=
  sorry

end rectangle_area_l197_197145


namespace DE_bisects_BC_l197_197520

open Real EuclideanGeometry

/-- Definition of the circle and given conditions: -/
variables (O A B C D E M N : Point)
variable (circleO : Circle O)
variable (perpendicular_CD_AB : Perpendicular CD AB)
variable (AE_bisects_OC : Bisects AE OC)
variable (diameter_AB : Diameter AB circleO)

/-- Proof that DE bisects BC, given the conditions: -/
theorem DE_bisects_BC :
  Bisects DE BC :=
by
  sorry

end DE_bisects_BC_l197_197520


namespace david_has_15_shells_l197_197569

-- Definitions from the conditions
def mia_shells (david_shells : ℕ) : ℕ := 4 * david_shells
def ava_shells (david_shells : ℕ) : ℕ := mia_shells david_shells + 20
def alice_shells (david_shells : ℕ) : ℕ := (ava_shells david_shells) / 2

-- Total number of shells
def total_shells (david_shells : ℕ) : ℕ := david_shells + mia_shells david_shells + ava_shells david_shells + alice_shells david_shells

-- Proving the number of shells David has is 15 given the total number of shells is 195
theorem david_has_15_shells : total_shells 15 = 195 :=
by
  sorry

end david_has_15_shells_l197_197569


namespace probability_red_face_l197_197756

def edge_length_large : ℕ := 6
def edge_length_small : ℕ := 1

def total_num_small_cubes : ℕ := edge_length_large ^ 3

def num_corner_cubes : ℕ := 8
def num_edge_cubes : ℕ := 48
def num_face_cubes : ℕ := 24

def num_cubes_with_red_faces : ℕ := num_corner_cubes + num_edge_cubes + num_face_cubes

theorem probability_red_face :
  (num_cubes_with_red_faces.to_nat / total_num_small_cubes.to_nat : ℚ) = 10 / 27 := by
  sorry

end probability_red_face_l197_197756


namespace parabola_focus_l197_197816

-- Definitions and conditions from the original problem
def parabola_eq (x y : ℝ) : Prop := x^2 = (1/2) * y 

-- Define the problem to prove the coordinates of the focus
theorem parabola_focus (x y : ℝ) (h : parabola_eq x y) : (x = 0 ∧ y = 1/8) :=
sorry

end parabola_focus_l197_197816


namespace square_midpoints_area_l197_197984

theorem square_midpoints_area (A : ℝ) (s : ℝ) :
  A = 80 →
  s^2 = A →
  let P : ℝ := s / (2 * sqrt 2) -- Side of the smaller square
  in P^2 = 40 :=
by
  intros h1 h2
  sorry

end square_midpoints_area_l197_197984


namespace correct_characters_total_l197_197697

theorem correct_characters_total (total_chars : ℕ) (mistakes_Yuan mistakes_Fang : ℕ → ℕ)
  (correct_ratio : ∀ {x : ℕ}, mistakes_Yuan x = x / 10 ∧ mistakes_Fang x = 2 * (x / 10))
  (correct_yuan_twice_fang : ∀ {y f : ℕ}, correct_chars_Yuan y = 2 * correct_chars_Fang f)
  (task : total_chars = 10000) :
  correct_chars_Yuan + correct_chars_Fang = 8640 := by
  sorry

end correct_characters_total_l197_197697


namespace arithmetic_seq_8th_term_l197_197078

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197078


namespace integer_values_less_than_4pi_l197_197458

theorem integer_values_less_than_4pi : 
  {x : ℤ | abs x < 4 * Real.pi}.card = 25 := 
by
  sorry

end integer_values_less_than_4pi_l197_197458


namespace sum_infinite_series_eq_l197_197287

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197287


namespace infinitely_many_decimals_l197_197450

theorem infinitely_many_decimals (a b : ℝ) (ha : a = 3.3) (hb : b = 3.6) : set.infinite {x : ℝ | a < x ∧ x < b} :=
sorry

end infinitely_many_decimals_l197_197450


namespace geometric_sequence_fifth_term_l197_197141

theorem geometric_sequence_fifth_term (x y : ℝ) (r : ℝ) 
  (h1 : x + y ≠ 0) (h2 : x - y ≠ 0) (h3 : x ≠ 0) (h4 : y ≠ 0)
  (h_ratio_1 : (x - y) / (x + y) = r)
  (h_ratio_2 : (x^2 * y) / (x - y) = r)
  (h_ratio_3 : (x * y^2) / (x^2 * y) = r) :
  (x * y^2 * ((y / x) * r)) = y^3 := 
by 
  sorry

end geometric_sequence_fifth_term_l197_197141


namespace sin_double_minus_cos_sq_l197_197853

theorem sin_double_minus_cos_sq {α : ℝ} (h : tan α = 2) : sin (2 * α) - cos (α) ^ 2 = 3 / 5 :=
sorry

end sin_double_minus_cos_sq_l197_197853


namespace magnitude_of_z_l197_197545

noncomputable def problem_statement (r : ℝ) (z : ℂ) :=
  |r| < 4 ∧ z + 1 / z + 2 = r

theorem magnitude_of_z (r : ℝ) (z : ℂ) (h : problem_statement r z) : |z| = 1 :=
sorry

end magnitude_of_z_l197_197545


namespace least_integer_in_list_is_minus_one_l197_197565

noncomputable def least_integer_in_list (K : List ℤ) : ℤ :=
  if h : K.length = 12 ∧ ∃ m n : ℕ, (m ≤ n ∧ K.all (λ k, m ≤ k ∧ k ≤ n) ∧ n - m = 7) then
    let m := K.min' (by sorry)
    let n := K.max' (by sorry)
    let smallest := K.min'.get (by sorry)
    smallest
  else
    0

theorem least_integer_in_list_is_minus_one (K : List ℤ) (h1 : K.length = 12) 
  (h2 : ∃ m n : ℕ, (m ≤ n ∧ K.all (λ k, m ≤ k ∧ k ≤ n) ∧ n - m = 7)) : 
  least_integer_in_list K = -1 :=
begin
  sorry
end

end least_integer_in_list_is_minus_one_l197_197565


namespace distinct_3_element_subsets_l197_197792

theorem distinct_3_element_subsets (n : ℕ) 
  (h1 : 0 < n)
  (A : finset (finset ℕ))
  (h2 : ∀ (i j : fin n), i ≠ j → ∃ k l m : ℕ, {k, l, m} ∈ A ∧ k ≠ l ∧ l ≠ m ∧ m ≠ k ∧ k ∈ finset.range n \ 
  ∧ l ∈ finset.range n ∧ m ∈ finset.range n)
  (h3 : ∀ (i j : fin n), i ≠ j → (A i ∩ A j).card ≠ 1) :
  ∃ (m : ℕ), n = 4 * m :=
by
  sorry

end distinct_3_element_subsets_l197_197792


namespace arithmetic_sequence_8th_term_l197_197113

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197113


namespace magnitude_of_vector_l197_197551

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - complex.I) : complex.abs (2 / z + z^2) = 2 :=
by
  -- Assume z = 1 - i
  have hz : z = 1 - complex.I := h
  -- The proof would go here
  sorry

end magnitude_of_vector_l197_197551


namespace arithmetic_sequence_8th_term_l197_197085

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197085


namespace polynomial_irreducible_by_eisenstein_l197_197027

theorem polynomial_irreducible_by_eisenstein (f : ℕ → ℤ) (n : ℕ) (a : ℕ → ℤ) (p : ℕ) [hp : prime p] 
   (h_1 : a n ≠ 0)
   (h_2 : ∀ i : ℕ, i < n → p ∣ a i)
   (h_3 : ¬ (p ∣ a n))
   (h_4 : p^2 ∣/ a 0) : irreducible (λ x : ℕ, ∑ i in range (n + 1), a i * x^i) := 
sorry

end polynomial_irreducible_by_eisenstein_l197_197027


namespace largest_c_3_in_range_l197_197368

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end largest_c_3_in_range_l197_197368


namespace magnitude_of_b_l197_197863

variables (a b : ℝ × ℝ)
variable (theta : ℝ)
variable (dot_product : ℝ)

-- Given conditions
def angle_between_vectors := theta = (3 * Real.pi) / 4
def vector_a := a = (-3, 4)
def dot_product_ab := dot_product = -10

-- Prove |b| = 2√2
theorem magnitude_of_b 
    (h1 : angle_between_vectors theta)
    (h2 : vector_a a)
    (h3 : dot_product_ab dot_product) :
  ||b|| = 2 * Real.sqrt 2 :=
sorry

end magnitude_of_b_l197_197863


namespace inverse_isosceles_triangle_l197_197628

theorem inverse_isosceles_triangle (T : Type) [triangle T] (ABC : T) (A B C : ℕ) :
  (angles_equal A B = true) → (isosceles_triangle ABC) :=
by
  sorry

end inverse_isosceles_triangle_l197_197628


namespace root_of_polynomial_gt_zero_l197_197036

theorem root_of_polynomial_gt_zero (a : ℝ) (x1 x2 x3 : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (hx1 : x1 < x2) (hx2 : x2 < x3) (h_roots : (x^3 - 4 * x - a).is_root x1)
  (h_roots : (x^3 - 4 * x - a).is_root x2) (h_roots : (x^3 - 4 * x - a).is_root x3) : 
  x2 > 0 :=
sorry

end root_of_polynomial_gt_zero_l197_197036


namespace arithmetic_sequence_8th_term_l197_197080

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197080


namespace cyclic_quad_circumcenter_on_OB_OD_l197_197557

noncomputable def cyclicQuadrilateral (A B C D : Point) : Prop :=
  ∃ (Ω : Circle), A ∈ Ω ∧ B ∈ Ω ∧ C ∈ Ω ∧ D ∈ Ω

noncomputable def circumcenter (A B C D : Point) : Point :=
  sorry  -- Point should be defined as the circumcenter of the circle through A, B, C, D.

noncomputable def angleBisectorIntersect (A B C D : Point) : (Point × Point) :=
  sorry  -- Points B1, D1 where angle bisectors intersect AC

noncomputable def circleThroughAndTangent (P Q : Point) (L : Line) : Circle :=
  sorry  -- Defines a circle through P and tangent to line L at Q

noncomputable def parallel (L1 L2 : Line) : Prop :=
  sorry  -- Defines parallelism of two lines

theorem cyclic_quad_circumcenter_on_OB_OD
  (A B C D O : Point)
  (ABCD_cyclic : cyclicQuadrilateral A B C D)
  (O_circumcenter : O = circumcenter A B C D)
  (B1 D1 : Point)
  (B1_D1_intersect : (B1, D1) = angleBisectorIntersect A B C D)
  (cO_B : Circle)
  (O_B_tangent : cO_B = circleThroughAndTangent B D1 (line AC))
  (cO_D : Circle)
  (O_D_tangent : cO_D = circleThroughAndTangent D B1 (line AC))
  (BD1_par_D_B1 : parallel (line BD1) (line DB1)) :
  O ∈ lineThrough (center cO_B) (center cO_D) := sorry

end cyclic_quad_circumcenter_on_OB_OD_l197_197557


namespace intervals_of_monotonicity_l197_197876

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi / 3)

theorem intervals_of_monotonicity :
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) → (f x ≤ f (7 * Real.pi / 12 + k * Real.pi)))) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi) → (f x ≥ f (Real.pi / 12 + k * Real.pi)))) ∧
  (f (Real.pi / 2) = -Real.sqrt 3) ∧
  (f (Real.pi / 12) = 1 - Real.sqrt 3 / 2) := sorry

end intervals_of_monotonicity_l197_197876


namespace alpha_abs_value_l197_197540

theorem alpha_abs_value (α β : ℂ) 
  (h_conjugate : α.im = -β.im ∧ α.re = β.re)
  (h_real_ratio : ∃ (r : ℝ), α^2 / β^3 = r)
  (h_distance : complex.abs (α - β) = 4) :
  complex.abs α = real.sqrt (8 + 2 * real.sqrt 5) :=
by sorry

end alpha_abs_value_l197_197540


namespace simplify_expression_l197_197994

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end simplify_expression_l197_197994


namespace upper_bound_exists_l197_197830

theorem upper_bound_exists (n : ℤ) (h1 : (4 * n + 7 > 1)) (h2 : ∃ (n_min n_max : ℤ), (n_max - n_min = 14 ∧ ∀ k : ℤ, (n_min ≤ k ∧ k ≤ n_max) → 4 * k + 7 < 64)) : 64 := 
sorry

end upper_bound_exists_l197_197830


namespace average_of_three_l197_197890

-- Definitions of Conditions
variables (A B C : ℝ)
variables (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132)

-- The proof problem stating the goal
theorem average_of_three (A B C : ℝ) 
    (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132) : 
    (A + B + C) / 3 = 67 := 
sorry

end average_of_three_l197_197890


namespace round_3_147_to_nearest_hundredth_l197_197991

theorem round_3_147_to_nearest_hundredth : (real.round_to 0.01 3.147) = 3.15 :=
by
  sorry

end round_3_147_to_nearest_hundredth_l197_197991


namespace karen_has_32_quarters_l197_197954

variable (k : ℕ)  -- the number of quarters Karen has

-- Define the number of quarters Christopher has
def christopher_quarters : ℕ := 64

-- Define the value of a single quarter in dollars
def quarter_value : ℚ := 0.25

-- Define the amount of money Christopher has
def christopher_money : ℚ := christopher_quarters * quarter_value

-- Define the monetary difference between Christopher and Karen
def money_difference : ℚ := 8

-- Define the amount of money Karen has
def karen_money : ℚ := christopher_money - money_difference

-- Define the number of quarters Karen has
def karen_quarters := karen_money / quarter_value

-- The theorem we need to prove
theorem karen_has_32_quarters : k = 32 :=
by
  sorry

end karen_has_32_quarters_l197_197954


namespace problem_statement_l197_197892

-- Definitions based on the conditions
def vector_oa := (3, -4)
def vector_ob := (6, -3)
def vector_oc (m : ℝ) := (5 - m, -3 - m)

-- Condition for triangle formation
def not_collinear (m : ℝ) := m ≠ 1 / 2

-- Definition of vector AC
def vector_ac (m : ℝ) := (2 - m, 1 - m)

-- Square of the length of vector AC
def ac_squared (m : ℝ) := (2 - m)^2 + (1 - m)^2

-- Condition given for all m in [1, 2]
def ac_squared_condition (x : ℝ) := ∀ (m : ℝ), 1 ≤ m ∧ m ≤ 2 → ac_squared(m) ≤ -x^2 + x + 3

-- Statement of the mathematical proof problem in Lean
theorem problem_statement : 
  (∀ (m : ℝ), vector_oa, vector_ob, vector_oc m, not_collinear m) → 
  (∀ (x : ℝ), ac_squared_condition x → -1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end problem_statement_l197_197892


namespace find_norm_a_plus_b_l197_197448

variables (a b : ℝ^3)

-- Given conditions
axiom length_a : ∥a∥ = 2
axiom length_b : ∥b∥ = 1
axiom perp_condition : (b - 2 • a) ⬝ b = 0

-- The proof objective
theorem find_norm_a_plus_b : ∥a + b∥ = real.sqrt 6 := by
  sorry

end find_norm_a_plus_b_l197_197448


namespace talia_age_in_seven_years_l197_197933

variable {T : ℕ}

theorem talia_age_in_seven_years (h₁ : 3 * T = 39) (h₂ : Talia_father_age_in_3_years := 36 + 3) :
  T + 7 = 20 :=
by
  have T_curr : T = 13 := by linarith
  show T + 7 = 20 from by linarith

end talia_age_in_seven_years_l197_197933


namespace geometric_series_first_term_l197_197276

theorem geometric_series_first_term (r a S : ℝ) (hr : r = 1 / 8) (hS : S = 60) (hS_formula : S = a / (1 - r)) : 
  a = 105 / 2 := by
  rw [hr, hS] at hS_formula
  sorry

end geometric_series_first_term_l197_197276


namespace area_of_triangle_l197_197851

-- Define the hyperbola and its foci
def hyperbola (a b : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def focus1 (a : ℝ) := (-real.sqrt (a^2 + 4), 0)
def focus2 (a : ℝ) := (real.sqrt (a^2 + 4), 0)
def point_on_hyperbola (a b : ℝ) (P : ℝ × ℝ) := hyperbola a b P.1 P.2

noncomputable def cos_60 := real.cos (real.pi / 3)
noncomputable def sin_60 := real.sin (real.pi / 3)

-- Lean statement for the proof problem
theorem area_of_triangle (a : ℝ) (P : ℝ × ℝ) (h1 : a > 0)
  (h2 : point_on_hyperbola a 2 P) (h3 : ∃ θ : ℝ, θ = real.pi / 3) :
  (1/2) * (16 : ℝ) * real.sin (real.pi / 3) = 4 * real.sqrt 3 :=
by {
  sorry,
}

end area_of_triangle_l197_197851


namespace test_question_count_l197_197174

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end test_question_count_l197_197174


namespace sequence_arithmetic_and_sum_l197_197402

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) + a n) / 2

theorem sequence_arithmetic_and_sum :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  (a 1 = 1) →
  (a 2 = 2) →
  (is_arithmetic_sequence a) ∧
  ((∑ i in range n, (2:ℝ)^i * (1 - a i) / (a i * a (i + 1))) = 2 - (2:ℝ)^(n + 1) / (n + 1)) :=
by
  sorry

end sequence_arithmetic_and_sum_l197_197402


namespace correct_propositions_l197_197873

def prop1 : Prop :=
  ∀ (P : Point) (α : Plane) (θ : Real), (P ∉ α) → ∃ (L : Line), (P ∈ L) ∧ (angle L α = θ)

def prop2 : Prop :=
  ∀ (L : Line) (α β : Plane), (L ∥ α) → (L ∥ β) → (L ∥ intersection_line α β)

def prop3 : Prop :=
  ∀ (L1 L2 : Line) (P : Point), (L1 ∦ L2) → ∃! (α : Plane), (P ∈ α) ∧ (L1 ∥ α) ∧ (L2 ∥ α)

def prop4 : Prop :=
  ∀ (L1 L2 : Line), (L1 ∦ L2) → ∃ (α : Plane), (angle α L1 = angle α L2) ∧ (∀ (β : Plane), (angle β L1 = angle α L1) → (angle β L2 = angle α L2))

theorem correct_propositions :
  prop2 ∧ prop4 :=
by 
  sorry

end correct_propositions_l197_197873


namespace sampling_methods_correct_l197_197257

def company_sales_outlets (A B C D : ℕ) : Prop :=
  A = 150 ∧ B = 120 ∧ C = 180 ∧ D = 150 ∧ A + B + C + D = 600

def investigation_samples (total_samples large_outlets region_C_sample : ℕ) : Prop :=
  total_samples = 100 ∧ large_outlets = 20 ∧ region_C_sample = 7

def appropriate_sampling_methods (investigation1_method investigation2_method : String) : Prop :=
  investigation1_method = "Stratified sampling" ∧ investigation2_method = "Simple random sampling"

theorem sampling_methods_correct :
  company_sales_outlets 150 120 180 150 →
  investigation_samples 100 20 7 →
  appropriate_sampling_methods "Stratified sampling" "Simple random sampling" :=
by
  intros h1 h2
  sorry

end sampling_methods_correct_l197_197257


namespace tan_beta_solution_l197_197390

theorem tan_beta_solution
  (α β : ℝ)
  (h₁ : Real.tan α = 2)
  (h₂ : Real.tan (α + β) = -1) :
  Real.tan β = 3 := 
sorry

end tan_beta_solution_l197_197390


namespace exists_not_perfect_square_l197_197834

variable (L : List ℕ)
variable (hL_size : L.length = 2021)
variable (hL_distinct : L.Nodup)
variable (hL_cond : ∀ a ∈ L, ¬ (∃ k : ℕ, k ≥ 1010 ∧ 2^k ∣ a))

theorem exists_not_perfect_square :
  ∃ a b c ∈ L, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬ ∃ k : ℕ, k^2 = |b^2 - 4 * a * c| :=
by
  sorry

end exists_not_perfect_square_l197_197834


namespace quadrilateral_parallel_sides_bisectors_rhombus_eq_sides_l197_197507

open EuclideanGeometry

theorem quadrilateral_parallel_sides_bisectors_rhombus_eq_sides
  (A B C D : Point)
  (h1 : parallel (line A D) (line B C))
  (h2 : rhombus (bisector_angle A D C) (bisector_angle D B C) (bisector_angle A C B) (bisector_angle A D B))
  : dist A B = dist C D :=
by
  sorry

end quadrilateral_parallel_sides_bisectors_rhombus_eq_sides_l197_197507


namespace four_divides_n_if_subsets_exist_l197_197794

theorem four_divides_n_if_subsets_exist :
  ∀ n : ℕ, (∃ (A : Finset (Finset (Fin n))) (h : A.card = n),
  (∀ (i j : Fin n), i ≠ j → ∃ (Ai Aj : Finset n), Ai ∈ A ∧ Aj ∈ A ∧ i ≠ j → (Ai ∩ Aj).card ≠ 1)) → 4 | n :=
by
  intros n h
  sorry

end four_divides_n_if_subsets_exist_l197_197794


namespace subset_max_elements_l197_197011



open Set

theorem subset_max_elements {S : Finset ℕ} (A : Finset ℕ) 
  (h : ∀ x ∈ A, ∀ y ∈ A, x ≠ y → x + y ≠ 552) 
  (S_def : S = Finset.range' 1 56 |>.map (λ n, 10 * n - 9)) : 
  A ⊆ S → A.card ≤ 28 := 
by 
  sorry

end subset_max_elements_l197_197011


namespace profit_or_loss_percentage_correct_l197_197253

/-- 
Let's define all necessary parameters and prove that the profit or loss percentage is 44.26%.
-/
def purchase_price : ℝ := 32
def purchase_tax_rate : ℝ := 0.05
def shipping_fee : ℝ := 2.50
def selling_price : ℝ := 56
def trading_tax_rate : ℝ := 0.07

def total_purchase_price_with_tax : ℝ := purchase_price * (1 + purchase_tax_rate)
def total_cost : ℝ := total_purchase_price_with_tax + shipping_fee
def net_revenue : ℝ := selling_price * (1 - trading_tax_rate)
def profit_or_loss : ℝ := net_revenue - total_cost
def profit_or_loss_percentage : ℝ := (profit_or_loss / total_cost) * 100

theorem profit_or_loss_percentage_correct : profit_or_loss_percentage ≈ 44.26 :=
by
  sorry

end profit_or_loss_percentage_correct_l197_197253


namespace arithmetic_sequence_8th_term_l197_197123

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197123


namespace B_participated_Huangmei_Opera_l197_197996

-- Definitions using given conditions
def participated_A (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∨ 
  (c "Huangmei Flower Picking" ∧ ¬ c "Yue Family Boxing")

def participated_B (c : String → Prop) : Prop :=
  (c "Huangmei Opera" ∧ ¬ c "Huangmei Flower Picking") ∨
  (c "Yue Family Boxing" ∧ ¬ c "Huangmei Flower Picking")

def participated_C (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∧ c "Huangmei Flower Picking" ∧ c "Yue Family Boxing" ->
  (c "Huangmei Opera" ∨ c "Huangmei Flower Picking" ∨ c "Yue Family Boxing")

-- Proving the special class that B participated in
theorem B_participated_Huangmei_Opera :
  ∃ c : String → Prop, participated_A c ∧ participated_B c ∧ participated_C c → c "Huangmei Opera" :=
by
  -- proof steps would go here
  sorry

end B_participated_Huangmei_Opera_l197_197996


namespace correct_judgments_count_l197_197985

open Set

def p : Prop := ¬(∈) {2} {1, 2, 3}
def q : Prop := Subset {2} {1, 2, 3}

theorem correct_judgments_count : 
  (if (p ∨ q) then 1 else 0) +
  (if ¬(p ∨ q) then 1 else 0) +
  (if (p ∧ q) then 1 else 0) +
  (if ¬(p ∧ q) then 1 else 0) +
  (if ¬p then 1 else 0) +
  (if ¬q then 1 else 0) = 4 :=
  by
    sorry

end correct_judgments_count_l197_197985


namespace find_angle_A_find_area_triangle_l197_197506

variable (a b c A B : ℝ)
variable (A_acute : A < π / 2)
variable (cond1 : 2 * a * sin B = sqrt 3 * b)

-- (I) Prove A = π / 3
theorem find_angle_A : 
  A = π / 3 :=
sorry

variable (a_eq_6 : a = 6)
variable (bc_eq_8 : b + c = 8)

-- (II) Prove the area of triangle ABC is 7 * sqrt 3 / 3
theorem find_area_triangle : 
  let bc := (b * c) in
  let angle_A := π / 3 in
  (1 / 2) * b * c * sin angle_A = 7 * sqrt 3 / 3 :=
sorry

end find_angle_A_find_area_triangle_l197_197506


namespace greatest_integer_a_for_domain_of_expression_l197_197817

theorem greatest_integer_a_for_domain_of_expression :
  ∃ a : ℤ, (a^2 < 60 ∧ (∀ b : ℤ, b^2 < 60 → b ≤ a)) :=
  sorry

end greatest_integer_a_for_domain_of_expression_l197_197817


namespace arithmetic_sequence_8th_term_l197_197118

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197118


namespace robert_monthly_expenses_l197_197990

def robert_basic_salary : ℝ := 1250
def robert_sales : ℝ := 23600
def first_tier_limit : ℝ := 10000
def second_tier_limit : ℝ := 20000
def first_tier_rate : ℝ := 0.10
def second_tier_rate : ℝ := 0.12
def third_tier_rate : ℝ := 0.15
def savings_rate : ℝ := 0.20

def first_tier_commission : ℝ :=
  first_tier_limit * first_tier_rate

def second_tier_commission : ℝ :=
  (second_tier_limit - first_tier_limit) * second_tier_rate

def third_tier_commission : ℝ :=
  (robert_sales - second_tier_limit) * third_tier_rate

def total_commission : ℝ :=
  first_tier_commission + second_tier_commission + third_tier_commission

def total_earnings : ℝ :=
  robert_basic_salary + total_commission

def savings : ℝ :=
  total_earnings * savings_rate

def monthly_expenses : ℝ :=
  total_earnings - savings

theorem robert_monthly_expenses :
  monthly_expenses = 3192 := by
  sorry

end robert_monthly_expenses_l197_197990


namespace problem_statement_l197_197886

def a : ℕ → ℕ
| 1 := 2
| 2 := 6
| (n + 3) := 2 * a (n + 2) - a (n + 1) + 2

theorem problem_statement :
    (⌊ ∑ i in finset.range 2017, 2017 / a (i + 1) ⌋) = 2016 :=
sorry

end problem_statement_l197_197886


namespace donation_amounts_l197_197614

theorem donation_amounts (d1 d2 d3 d4 d5 : ℕ) (h_avg : (d1 + d2 + d3 + d4 + d5) / 5 = 560) 
  (h_mult100 : ∀ d, d ∈ {d1, d2, d3, d4, d5} → d % 100 = 0)
  (h_least : ∃ d, d ∈ {d1, d2, d3, d4, d5} ∧ d = 200)
  (h_most : ∃! d, d ∈ {d1, d2, d3, d4, d5} ∧ d = 800)
  (h_median : ∃ d, d ∈ {d1, d2, d3, d4, d5} ∧ d = 600 ∧ list.nth (list.sort (≤) [d1, d2, d3, d4, d5]) 2 = 600) :
  (∃ x y, (x, y) ∈ {(500, 700), (600, 600)} ∧ x + y + 200 + 800 + 600 = 2800) :=
sorry

end donation_amounts_l197_197614


namespace smallest_w_correct_l197_197480

-- Define the conditions
def is_factor (a b : ℕ) : Prop := ∃ k, a = b * k

-- Given conditions
def cond1 (w : ℕ) : Prop := is_factor (2^6) (1152 * w)
def cond2 (w : ℕ) : Prop := is_factor (3^4) (1152 * w)
def cond3 (w : ℕ) : Prop := is_factor (5^3) (1152 * w)
def cond4 (w : ℕ) : Prop := is_factor (7^2) (1152 * w)
def cond5 (w : ℕ) : Prop := is_factor (11) (1152 * w)
def is_positive (w : ℕ) : Prop := w > 0

-- The smallest possible value of w given all conditions
def smallest_w : ℕ := 16275

-- Proof statement
theorem smallest_w_correct : 
  ∀ (w : ℕ), cond1 w ∧ cond2 w ∧ cond3 w ∧ cond4 w ∧ cond5 w ∧ is_positive w ↔ w = smallest_w := sorry

end smallest_w_correct_l197_197480


namespace parade_ground_problem_l197_197981

theorem parade_ground_problem :
  ∃ (ways : ℕ), ways = 30 ∧ ways = (nat.factorial 8 / (nat.factorial 7 * nat.factorial 1)) :=
  sorry

end parade_ground_problem_l197_197981


namespace sin_neg_135_eq_neg_sqrt_2_over_2_l197_197778

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_135_eq_neg_sqrt_2_over_2_l197_197778


namespace least_integer_of_quadratic_solution_l197_197688

theorem least_integer_of_quadratic_solution :
  ∃ x : ℤ, x * x = 3 * (2 * x) + 50 ∧ x = -4 :=
by
  have eq : ∀ x, x * x = 3 * (2 * x) + 50 ↔ x * x - 6 * x - 50 = 0 := by
  {
    intro x,
    rw [←sub_eq_zero, mul_assoc, ←eq_sub_iff_add_eq],
    apply eq.symm,
    ring,
  }
  sorry

end least_integer_of_quadratic_solution_l197_197688


namespace find_b_l197_197424

def z1 : ℂ := 1 + I
def z2 (b : ℂ) : ℂ := 2 + b * I

theorem find_b (b : ℝ) (h : (z1 * z2 b).im = 0) : b = -2 :=
by sorry

end find_b_l197_197424


namespace percentage_increase_l197_197727

theorem percentage_increase (regular_rate : ℝ) (regular_hours total_compensation total_hours_worked : ℝ)
  (h1 : regular_rate = 20)
  (h2 : regular_hours = 40)
  (h3 : total_compensation = 1000)
  (h4 : total_hours_worked = 45.714285714285715) :
  let overtime_hours := total_hours_worked - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_compensation - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  let percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100
  percentage_increase = 75 := 
by
  sorry

end percentage_increase_l197_197727


namespace continuous_at_three_l197_197034

noncomputable def f (b : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 3 then 2 * x^2 - 3 else b * x - 5

theorem continuous_at_three (b : ℝ) :
  ContinuousAt (f b) 3 ↔ b = 20 / 3 := by
sorry

end continuous_at_three_l197_197034


namespace arithmetic_sequence_8th_term_l197_197117

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197117


namespace relationship_0_2_with_M_l197_197746

variable (M : Set ℤ)

-- Conditions
variable (h1 : ∃ x ∈ M, x > 0 ∧ ∃ y ∈ M, y < 0) 
variable (h2 : ∃ a ∈ M, a % 2 = 1 ∧ ∃ b ∈ M, b % 2 = 0) 
variable (h3 : -1 ∉ M)
variable (h4 : ∀ x y ∈ M, x + y ∈ M)

theorem relationship_0_2_with_M : (0 ∈ M) ∧ (2 ∉ M) := by
  sorry

end relationship_0_2_with_M_l197_197746


namespace zipToBarcode_correct_barcodeToZip_correct_l197_197708

def digitToBinary : ℕ → String
| 0 => "11000"
| 1 => "00011"
| 2 => "01101"
| 3 => "00101"
| 4 => "00110"
| 5 => "01010"
| 6 => "10100"
| 7 => "00001"
| 8 => "01011"
| 9 => "10010"
| _ => "XXXXX"  -- invalid input

def binaryToDigit (s : String) : ℕ :=
match s with
| "11000" => 0
| "00011" => 1
| "01101" => 2
| "00101" => 3
| "00110" => 4
| "01010" => 5
| "10100" => 6
| "00001" => 7
| "01011" => 8
| "10010" => 9
| _ => 10  -- essentially an invalid output

def zipToBarcode (zip : List ℕ) : String :=
zip.map digitToBinary |> String.concat " "

def barcodeToZip (barcode : List String) : List ℕ :=
barcode.map binaryToDigit

theorem zipToBarcode_correct :
  zipToBarcode [3,6,4,7,0,1,3,0] = "00101 10100 00110 00001 11000 00011 00101 11000" :=
sorry

theorem barcodeToZip_correct :
  barcodeToZip ["01010", "10010", "00001", "00011", "11000", "00110", "10010", "00001"] = [2,9,7,1,0,4,9,7] :=
sorry

end zipToBarcode_correct_barcodeToZip_correct_l197_197708


namespace rectangle_area_l197_197989

/-!
# Area of Rectangle Inscribed in Parabola

Given:
1. Rectangle ABCD is inscribed in the region bounded by the parabola y = x^2 - 12x + 32 and the x-axis.
2. The vertex A of the rectangle lies on the x-axis.
3. The rectangle's two sides are parallel and two sides are perpendicular to the x-axis.
4. Each vertical side of the rectangle has a length equal to one third of the rectangle's base.

Prove:
The area of the rectangle ABCD is 91 + 25 * sqrt 13.
-/

open Real

theorem rectangle_area : 
  ∃ (A B C D : ℝ × ℝ) (base height : ℝ),
  (∃ (x : ℝ), A = (6-x, 0) ∧ B = (6+x, 0) ∧ C = (6+x, -2*x / 3) ∧ D = (6-x, -2*x / 3)) ∧
  (∀ p ∈ {A, B, C, D}, p.2 = (p.1)^2 - 12*(p.1) + 32 ∨ p.2 = 0) ∧
  (height = 2 * (base / 3)) ∧ 
  (base = 2 * x) ∧
  (4 * (x^2) / 3 = 91 + 25 * sqrt 13) := sorry

end rectangle_area_l197_197989


namespace minimum_work_to_remove_cube_l197_197757

namespace CubeBuoyancy

def edge_length (ℓ : ℝ) := ℓ = 0.30 -- in meters
def wood_density (ρ : ℝ) := ρ = 750  -- in kg/m^3
def water_density (ρ₀ : ℝ) := ρ₀ = 1000 -- in kg/m^3

theorem minimum_work_to_remove_cube 
  {ℓ ρ ρ₀ : ℝ} 
  (h₁ : edge_length ℓ)
  (h₂ : wood_density ρ)
  (h₃ : water_density ρ₀) : 
  ∃ W : ℝ, W = 22.8 := 
sorry

end CubeBuoyancy

end minimum_work_to_remove_cube_l197_197757


namespace integer_solution_l197_197363

theorem integer_solution (n : ℤ) (hneq : n ≠ -2) :
  ∃ (m : ℤ), (n^3 + 8) = m * (n^2 - 4) ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end integer_solution_l197_197363


namespace solve_system_eq_l197_197998

theorem solve_system_eq (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) → 
  ( ∃ t : ℝ, (x = (1 + t) * b) ∧ (y = (1 - t) * b) ∧ (z = 0) ∧ t^2 = -1/2 ) :=
by
  -- proof will be filled in here
  sorry

end solve_system_eq_l197_197998


namespace sum_fraction_series_l197_197297

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197297


namespace proof_problem_l197_197013

noncomputable def solution : ℤ :=
let x : ℝ := -0.879385 in
let y : ℝ := -0.879385 in
let z : ℝ := -0.879385 in
let m := ⌊x^3 + y^3 + z^3⌋ in
(m % 2007)

theorem proof_problem :
  (∃ x y z : ℝ, 
    (x + y^2 + z^4 = 0) ∧ 
    (y + z^2 + x^4 = 0) ∧ 
    (z + x^2 + y^4 = 0) ∧ 
    (solution = 2004)) :=
by
  let x := -0.879385
  let y := -0.879385
  let z := -0.879385
  let m := ⌊x^3 + y^3 + z^3⌋
  have h1 : x + y^2 + z^4 = 0 := sorry
  have h2 : y + z^2 + x^4 = 0 := sorry
  have h3 : z + x^2 + y^4 = 0 := sorry
  have h_m : m = -3 := sorry
  have h_mod : -3 % 2007 = 2004 := sorry
  use [x, y, z]
  tauto

end proof_problem_l197_197013


namespace arithmetic_sequence_8th_term_l197_197081

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197081


namespace recipe_sugar_cups_l197_197977

theorem recipe_sugar_cups (total_flour : ℕ) (flour_added : ℕ) (sugar_more_than_flour : ℕ) :
  total_flour = 9 → flour_added = 4 → sugar_more_than_flour = 6 → 
  let flour_needed := total_flour - flour_added in
  let sugar_needed := flour_needed + sugar_more_than_flour in
  sugar_needed = 11 :=
by
  intros h1 h2 h3
  unfold flour_needed sugar_needed
  rw [h1, h2, h3]
  norm_num
  sorry

end recipe_sugar_cups_l197_197977


namespace prism_problem_l197_197267

-- Given conditions
def n : ℕ := 5
def a : ℕ := 7
def b : ℕ := 10
def edges : ℕ := 15

theorem prism_problem (h : 3 * n = edges) : n - a * b = -65 :=
by 
  sorry

end prism_problem_l197_197267


namespace evaluate_expression_l197_197809

theorem evaluate_expression (x : ℕ) (h : x = 3) : 
  (∏ i in (finset.range 20).map (function.embedding.sigma_mk ∅) (function.embedding.sigma_mk ∅ (λ _, finset.range 20))) /
  (∏ i in (finset.range 10).map (function.embedding.sigma_mk ∅) (function.embedding.sigma_mk ∅ (λ _, finset.range 10)).image (λ i, 2 * i + 1)) = 3^110 := 
sorry

end evaluate_expression_l197_197809


namespace find_YQ_length_l197_197000

-- Definition of a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of a line segment
structure Segment :=
  (start : Point)
  (end : Point)

-- Definitions based on given conditions
def XY := 10
def segmentPQ := 6

-- Predicate to represent collinearity
def Collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (B.y - A.y) * (C.x - A.x)

-- Predicate to represent parallelism of segments
def Parallel (P Q : Segment) : Prop :=
  (P.end.x - P.start.x) * (Q.end.y - Q.start.y) = (P.end.y - P.start.y) * (Q.end.x - Q.start.x)

-- Main theorem statement
theorem find_YQ_length (P Q R X Y Z W : Point) (PQ : Segment) :
  XY = 10 ∧
  PQ = 6 ∧
  Segment.start PQ = P ∧
  Segment.end PQ = Q ∧
  Collinear X P Z ∧
  Collinear Y Q Z ∧
  Segment // this needs to match with definition from conditions ∧ 
  AngleBisector XW QWR ∧
  Parallel PQ XY →
  Segment.length YQ = 15 :=
sorry

end find_YQ_length_l197_197000


namespace sin_double_angle_l197_197389

theorem sin_double_angle (x : ℝ) (h : Real.tan (π / 4 - x) = 2) : Real.sin (2 * x) = -3 / 5 :=
by
  sorry

end sin_double_angle_l197_197389


namespace normal_distribution_highest_point_l197_197420

def normal_curve_highest_point (f : ℝ → ℝ) (μ : ℝ) :=
  ∀ x : ℝ, f(x) ≤ f(μ)

theorem normal_distribution_highest_point :
  (∀ z : ℝ, (∫ x in 0.2..z, pdf_normal 0 x) = 0.5) →
  (∃ x0 : ℝ, normal_curve_highest_point (λ x, pdf_normal 0 x) 0.2) :=
by
  intro h
  use 0.2
  sorry

end normal_distribution_highest_point_l197_197420


namespace sum_of_series_l197_197327

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197327


namespace solve_cryptarithm_l197_197813

-- Defining the constraints that each character represents a unique digit in range (0-9)
def is_valid_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Defining the digit variables
variables (C O W M I L K : ℕ)

-- Ensuring all variables are unique digits
axiom distinct_digits : C ≠ O ∧ C ≠ W ∧ C ≠ M ∧ C ≠ I ∧ C ≠ L ∧ C ≠ K ∧
                        O ≠ W ∧ O ≠ M ∧ O ≠ I ∧ O ≠ L ∧ O ≠ K ∧
                        W ≠ M ∧ W ≠ I ∧ W ≠ L ∧ W ≠ K ∧
                        M ≠ I ∧ M ≠ L ∧ M ≠ K ∧
                        I ≠ L ∧ I ≠ K ∧
                        L ≠ K

-- Definition of the cryptarithm puzzle and the proof statement
theorem solve_cryptarithm :
    (is_valid_digit C) ∧ (is_valid_digit O) ∧ (is_valid_digit W) ∧ 
    (is_valid_digit M) ∧ (is_valid_digit I) ∧ (is_valid_digit L) ∧ 
    (is_valid_digit K) ∧ distinct_digits ∧ 
    ((2 * (C * 100000 + O * 10000 + W * 1000 + O * 100 + V * 10 + A)) = 
     (M * 100000 + O * 10000 + L * 1000 + O * 100 + K * 10 + I)) :=
    (C = 3 ∧ O = 0 ∧ W = 5 ∧ M = 6 ∧ I = 0 ∧ L = 2 ∧ K = 1) ∨
    sorry

end solve_cryptarithm_l197_197813


namespace coin_flip_points_l197_197263

theorem coin_flip_points {a b : ℕ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_gt_b : a > b)
  (h_unattainable_scores : (nat_gcd a b = 1) → (∃ n, n < 35 ∧ ¬∃ x y, x * a + y * b = n)) :
  a = 11 ∧ b = 8 :=
by
  sorry

end coin_flip_points_l197_197263


namespace max_length_of_chord_l197_197491

theorem max_length_of_chord 
  (a : ℝ) 
  (h₀ : 0 < a ∧ a ≤ 4) 
  (h₁ : ∀ x y: ℝ, (x + a) ^ 2 + (y - a) ^ 2 = 4 * a → y = x + 4 → ∃ A B: ℝ × ℝ, A ≠ B ∧ A.1 = B.1 ∧ (A.2 - B.2) ^ 2 = 4 * (2 * a - ((2 * a - 4) ^ 2 / 2 - 2 * (a - 3) ^ 2 + 10)))
  : ∀ A B: ℝ × ℝ, A ≠ B ∧ (A.2 - B.2) ^ 2 = (2 * sqrt 10)^2 :=
by
  sorry

end max_length_of_chord_l197_197491


namespace arithmetic_seq_8th_term_l197_197066

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197066


namespace arithmetic_sequence_8th_term_is_71_l197_197102

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197102


namespace arithmetic_sequence_8th_term_l197_197087

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197087


namespace arithmetic_sequence_8th_term_l197_197107

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197107


namespace mod_of_complex_eq_l197_197394

open Complex

theorem mod_of_complex_eq (z : ℂ) (h : (z + 2) / (z - 2) = Complex.i) : Complex.abs z = 2 :=
sorry

end mod_of_complex_eq_l197_197394


namespace packages_needed_to_label_apartments_l197_197144

-- Define the conditions for apartment labeling ranging from 100 to 135, and 300 to 335
def apartments_floor_1 : List ℕ := List.range' 100 136
def apartments_floor_3 : List ℕ := List.range' 300 336

-- Define the function to count the digit occurrences in a list of apartment numbers
def count_digit (n : ℕ) (l : List ℕ) : ℕ :=
  l.foldr (λ (num acc) => acc + num.digits.count (λ d => d = n)) 0

-- Define the digit package size
def package_size : ℕ := 10

-- Define the function to compute the number of required packages
def required_packages (l1 l2 : List ℕ) : ℕ :=
  let max_count := max (count_digit 1 l1 + count_digit 1 l2) (count_digit 3 l1 + count_digit 3 l2)
  (max_count + package_size - 1) / package_size  -- ceiling division

-- Lean statement to prove that the number of packages needed is 46
theorem packages_needed_to_label_apartments : required_packages apartments_floor_1 apartments_floor_3 = 46 := by
  sorry

end packages_needed_to_label_apartments_l197_197144


namespace possible_medians_count_l197_197018

def is_possible_median (n : ℤ) (S : Set ℤ) : Prop :=
  ∃ T : Set ℤ, T ⊆ S ∧ T.card = 11 ∧ multiset.sort (≤) T = [(T.to_list.sort (≤)).nth 5]

theorem possible_medians_count : Finset.card
  (Finset.filter (λ x, is_possible_median x {1, 3, 5, 7, 11, 13, 17, a, b, c, d | a < b ∧ b < c ∧ c < d}) 
  (Finset.range 22)) = 4 :=
sorry

end possible_medians_count_l197_197018


namespace minimum_distance_l197_197596

def RationalManPath (t : ℝ) : ℝ × ℝ :=
  (2 * Real.cos t, Real.sin t)

def IrrationalManPath (t : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos (t / 2), 3 * Real.sin (t / 2))

theorem minimum_distance {A B : ℝ × ℝ} :
  (∃ t₁, A = RationalManPath t₁) →
  (∃ t₂, B = IrrationalManPath t₂) →
  ∃ d, d = 2 ∧ ∀ t₁ t₂, dist (RationalManPath t₁) (IrrationalManPath t₂) ≥ d :=
sorry

end minimum_distance_l197_197596


namespace clock_hands_overlap_l197_197233

theorem clock_hands_overlap (t : ℝ) :
  (∀ (h_angle m_angle : ℝ), h_angle = 30 + 0.5 * t ∧ m_angle = 6 * t ∧ h_angle = m_angle ∧ h_angle = 45) → t = 8 :=
by
  intro h
  sorry

end clock_hands_overlap_l197_197233


namespace sum_all_values_x_l197_197221

-- Define the problem's condition
def condition (x : ℝ) : Prop := Real.sqrt ((x - 2) ^ 2) = 9

-- Define the theorem to prove the sum of all solutions equals 4
theorem sum_all_values_x : ∑ x in {x : ℝ | condition x}, x = 4 := by
  -- Introduce the definition of condition
  sorry

end sum_all_values_x_l197_197221


namespace largest_c_range_3_l197_197369

theorem largest_c_range_3 (c : ℝ) : (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ (c ≤ 61 / 4) :=
begin
  sorry
end

end largest_c_range_3_l197_197369


namespace smallest_amount_received_is_14493_l197_197755

noncomputable def smallest_amount_received (range : ℝ) : ℝ := range / 0.69

theorem smallest_amount_received_is_14493 :
  let amount := smallest_amount_received 10000 in
  14492 < amount ∧ amount < 14494 :=
by
  let amount := smallest_amount_received 10000
  have h : amount = 10000 / 0.69 := rfl
  have h_approx : amount ≈ 14492.75 := sorry  -- approximation step using real arithmetic
  sorry

end smallest_amount_received_is_14493_l197_197755


namespace arithmetic_seq_8th_term_l197_197079

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197079


namespace train_speed_l197_197752

-- Definitions
def train_length : ℕ := 400-- in meters
def crossing_time : ℝ := 9.99920006399488 -- in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6 -- from m/s to km/hr

-- Required proof statement
theorem train_speed : 
  (train_length : ℝ) / crossing_time * conversion_factor ≈ 144.03 :=
sorry

end train_speed_l197_197752


namespace worker_payment_l197_197735

theorem worker_payment (x : ℕ) (daily_return : ℕ) (non_working_days : ℕ) (total_days : ℕ) 
    (net_earning : ℕ) 
    (H1 : daily_return = 25) 
    (H2 : non_working_days = 24) 
    (H3 : total_days = 30) 
    (H4 : net_earning = 0) 
    (H5 : ∀ w, net_earning = w * x - non_working_days * daily_return) : 
  x = 100 :=
by
  sorry

end worker_payment_l197_197735


namespace polynomial_coeff_sum_l197_197837

theorem polynomial_coeff_sum :
  (∃ (a : ℕ → ℤ), (x^3 - 1) * (x + 1)^7 = ∑ i in finset.range 11, a i * (x + 3)^i) →
  (∑ i in finset.range 11, a i = 9) :=
by {
  sorry
}

end polynomial_coeff_sum_l197_197837


namespace total_distance_travelled_l197_197698

noncomputable def cycle_distance (time_minutes : ℕ) (speed_mph : ℝ) : ℝ :=
  (time_minutes / 60) * speed_mph

noncomputable def walk_distance (time_minutes : ℕ) (speed_mph : ℝ) : ℝ :=
  (time_minutes / 60) * speed_mph

noncomputable def jog_distance (time_minutes : ℕ) (speed_mph : ℝ) : ℝ :=
  (time_minutes / 60) * speed_mph

theorem total_distance_travelled :
  let cycle_time := 20
  let cycle_speed := 12
  let walk_time := 40
  let walk_speed := 3
  let jog_time := 50
  let jog_speed := 7
  
  let cycle_dist := cycle_distance cycle_time cycle_speed
  let walk_dist := walk_distance walk_time walk_speed
  let jog_dist := jog_distance jog_time jog_speed
  
  cycle_dist + walk_dist + jog_dist = 11.8333 :=
by 
  let cycle_time := 20
  let cycle_speed := 12
  let walk_time := 40
  let walk_speed := 3
  let jog_time := 50
  let jog_speed := 7
  
  let cycle_dist := cycle_distance cycle_time cycle_speed
  let walk_dist := walk_distance walk_time walk_speed
  let jog_dist := jog_distance jog_time jog_speed

  have h1 := rfl : (cycle_dist + walk_dist + jog_dist = 11.8333)

  exact h1

end total_distance_travelled_l197_197698


namespace red_blue_points_counterexample_exists_l197_197406

/-- Formal statement: It is not necessarily true that K₁ is the closest to P₁ among the blue points
even if for every i from 1 to 8, Pᵢ is the closest to Kᵢ. -/
theorem red_blue_points_counterexample_exists (K P : Fin 8 → ℝ × ℝ)
  (h : ∀ i : Fin 8, ∀ j : Fin 8, dist (P i) (K i) ≤ dist (P i) (K j)) :
  ∃ i₁ i₂ : Fin 8, i₁ ≠ i₂ ∧ dist (P 0) (K i₂) < dist (P 0) (K 0) :=
begin
  -- Given the definition of positions and distances, we assume distances can create a counterexample.
  sorry 
end

end red_blue_points_counterexample_exists_l197_197406


namespace tour_guide_groupings_l197_197196

-- Definitions for the conditions in the problem
noncomputable def num_tour_guides := 2
noncomputable def num_tourists := 8

-- Definition for the number of ways to choose k tourists from 8 where 1 <= k <= 7
noncomputable def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The theorem we want to prove
theorem tour_guide_groupings :
  (∑ k in (Finset.range (num_tourists)).filter (λ k => k > 0 ∧ k < num_tourists), choose num_tourists k) = 254 :=
by
  sorry

end tour_guide_groupings_l197_197196


namespace sum_fraction_series_l197_197295

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197295


namespace largest_consecutive_odd_sum_7500_l197_197645

theorem largest_consecutive_odd_sum_7500 :
  ∃ y : ℤ, (∀ i : ℕ, 0 ≤ i ∧ i < 30 → (1 ≤ 2 * y + 2 * i + 1)) ∧
           (∑ i in finset.range 30, (2 * y + 2 * ↑i + 1)) = 7500 ∧
           (2 * y + 2 * 29 + 1) = 279 :=
by
  sorry

end largest_consecutive_odd_sum_7500_l197_197645


namespace general_term_sum_of_sequence_l197_197400

noncomputable theory

-- Definition of the sequence a_n
def a : ℕ+ → ℕ
| ⟨1, _⟩      := 0
| ⟨n+1, h⟩  := 2 * a ⟨n, h⟩ + n * 2^n

-- Problem 1: Prove the general term of the sequence
theorem general_term (n : ℕ+) : a n = 2^(n-2) * (n * (n-1)) :=
sorry

-- Definition of the sum of the first n terms S_n
def S (n : ℕ+) : ℕ :=
∑ k in Finset.range n, a k

-- Problem 2: Prove the sum of the first n terms of the sequence
theorem sum_of_sequence (n : ℕ+) : S n = 2^(n-1) * (n^2 - 3 * n + 4) - 2 :=
sorry

end general_term_sum_of_sequence_l197_197400


namespace count_flippant_integers_between_10_and_1000_l197_197743

def is_reverse_div_by_7 (n : ℕ) : Prop :=
  let rev := (n.digits 10).reverse.foldl (λ (acc : ℕ) d, acc * 10 + d) 0
  rev % 7 = 0

def is_flippant (n : ℕ) : Prop :=
  n % 10 ≠ 0 ∧ n % 7 = 0 ∧ is_reverse_div_by_7 n

theorem count_flippant_integers_between_10_and_1000 :
  (Finset.filter is_flippant (Finset.Ico 10 1000)).card = 17 :=
sorry

end count_flippant_integers_between_10_and_1000_l197_197743


namespace min_value_f_on_interval_l197_197371

open Real

noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 - 4 * tan x - 8 * cot x + 4 * cot x ^ 2 + 5

theorem min_value_f_on_interval :
  ∃ x ∈ Ioo (π / 2) π, ∀ y ∈ Ioo (π / 2) π, f y ≥ f x ∧ f x = 9 - 8 * sqrt 2 := 
by
  -- proof omitted
  sorry

end min_value_f_on_interval_l197_197371


namespace sin_double_angle_l197_197386

theorem sin_double_angle (x : ℝ) (h : Math.cos (Real.pi / 4 + x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
sorry

end sin_double_angle_l197_197386


namespace problem_statement_l197_197556

theorem problem_statement 
  (n d : ℕ) 
  (h1 : 2 ≤ n) 
  (h2 : 1 ≤ d) 
  (h3 : d ∣ n) 
  (x : Fin n → ℝ) 
  (h4 : ∑ i : Fin n, x i = 0) :
  ∃ (s : Finset (Fin n)) (hₛ : s.card = d), (∑ i in s, x i) ≥ 0 := 
sorry

end problem_statement_l197_197556


namespace sequence_value_2015_l197_197640

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 / 5 ∧ ∀ n : ℕ, 
    (a (n + 1) = 2 * a n ∧ (0 ≤ a n ∧ a n < 1 / 2)) ∨ 
    (a (n + 1) = 2 * a n - 1 ∧ (1 / 2 ≤ a n ∧ a n < 1))

theorem sequence_value_2015 : ∃ (a : ℕ → ℝ), sequence a ∧ a 2015 = 2 / 5 :=
by {
  sorry
}

end sequence_value_2015_l197_197640


namespace third_circle_ratio_l197_197037

universe u

variables {x : ℝ}

def circle_radius_small := x / 2
def circle_radius_large := 2 * x
def ratio_radius_third_circle (CB_radius : ℝ) : ℝ := 
  (CB_radius / 2) / CB_radius

theorem third_circle_ratio (x : ℝ) (hx: x > 0) :
  ratio_radius_third_circle x = 1 / 2 :=
by
  let r1 := circle_radius_small x
  let r2 := circle_radius_large x
  have hr3 : CB_radius / 2 = x / 2 := sorry
  have key : ratio_radius_third_circle x = (CB_radius / 2) / CB_radius := sorry
  rw [key, hr3, div_self] <|> simp; linarith
  sorry

end third_circle_ratio_l197_197037


namespace expression_for_even_quadratic_function_l197_197917

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def range_leq_4 (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≤ 4
  
-- Given conditions
variables (a b : ℝ)
noncomputable def f : ℝ → ℝ := λ x, (x + a) * (b * x + 2 * a)

-- Proof statement
theorem expression_for_even_quadratic_function :
  is_even_function (f a (-2 / a)) → (range_leq_4 (f a (-2 / a))) → f a (-2 / a) = λ x, -2 * x^2 + 4 :=
by
  sorry

end expression_for_even_quadratic_function_l197_197917


namespace series_sum_l197_197335

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197335


namespace distance_from_reflected_point_l197_197584

theorem distance_from_reflected_point
  (P : ℝ × ℝ) (P' : ℝ × ℝ)
  (hP : P = (3, 2))
  (hP' : P' = (3, -2))
  : dist P P' = 4 := sorry

end distance_from_reflected_point_l197_197584


namespace round_34_865_to_nearest_tenth_l197_197992

theorem round_34_865_to_nearest_tenth : Real.round_to_tenth 34.865 = 34.9 := by
  sorry

end round_34_865_to_nearest_tenth_l197_197992


namespace passengers_at_third_station_l197_197268

def initial_passengers : ℕ := 270
def passengers_dropped_first_station (p : ℕ) : ℕ := p / 3
def passengers_taken_first_station : ℕ := 280
def passengers_dropped_second_station (p : ℕ) : ℕ := p / 2
def passengers_taken_second_station : ℕ := 12

theorem passengers_at_third_station : 
  let first_station_passengers := initial_passengers - passengers_dropped_first_station initial_passengers + passengers_taken_first_station,
      second_station_passengers := first_station_passengers - passengers_dropped_second_station first_station_passengers + passengers_taken_second_station
  in
  second_station_passengers = 242 := by
  sorry

end passengers_at_third_station_l197_197268


namespace arithmetic_sequence_8th_term_l197_197083

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197083


namespace arithmetic_sequence_8th_term_is_71_l197_197103

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197103


namespace largest_decimal_of_4bit_binary_l197_197205

-- Define the maximum 4-bit binary number and its interpretation in base 10
def max_4bit_binary_value : ℕ := 2^4 - 1

-- The theorem to prove the statement
theorem largest_decimal_of_4bit_binary : max_4bit_binary_value = 15 :=
by
  -- Lean tactics or explicitly writing out the solution steps can be used here.
  -- Skipping proof as instructed.
  sorry

end largest_decimal_of_4bit_binary_l197_197205


namespace abs_neg_three_l197_197133

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l197_197133


namespace find_sp_l197_197147

theorem find_sp (s p : ℝ) (t x y : ℝ) (h1 : x = 3 + 5 * t) (h2 : y = 3 + p * t) 
  (h3 : y = 4 * x - 9) : 
  s = 3 ∧ p = 20 := 
by
  -- Proof goes here
  sorry

end find_sp_l197_197147


namespace limit_kx_div_x_zero_l197_197538

variables {a : ℕ → ℕ}
hypothesis (h1 : ∀ n, a n < a (n + 1))
hypothesis (h2 : ∃ S, ∑' i, 1 / (a i : ℝ) < S)
def k (x : ℝ) : ℕ := {i | (a i : ℝ) ≤ x}.to_finset.card

theorem limit_kx_div_x_zero : tendsto (λ x : ℝ, (k x) / x) at_top (𝓝 0) :=
sorry

end limit_kx_div_x_zero_l197_197538


namespace girls_not_playing_soccer_l197_197236

-- Define the given conditions
def students_total : Nat := 420
def boys_total : Nat := 312
def soccer_players_total : Nat := 250
def percent_boys_playing_soccer : Float := 0.78

-- Define the main goal based on the question and correct answer
theorem girls_not_playing_soccer : 
  students_total = 420 → 
  boys_total = 312 → 
  soccer_players_total = 250 → 
  percent_boys_playing_soccer = 0.78 → 
  ∃ (girls_not_playing_soccer : Nat), girls_not_playing_soccer = 53 :=
by 
  sorry

end girls_not_playing_soccer_l197_197236


namespace minimum_lambda_value_l197_197561

noncomputable def meets_inequality_for_all_x : Prop :=
  ∀ (λ : ℝ), λ > 0 →
    (∀ (x : ℝ), x > 0 → e^(λ * x) - (Real.log x) / λ >= 0) →
    λ ≥ 1 / Real.exp 1

theorem minimum_lambda_value : meets_inequality_for_all_x :=
by
  sorry

end minimum_lambda_value_l197_197561


namespace find_d_l197_197024

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3
def h (x : ℝ) (c : ℝ) (d : ℝ) : Prop := f (g x c) c = 15 * x + d

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l197_197024


namespace relationship_among_a_b_c_l197_197391

noncomputable def a : ℝ := 2^0.8
noncomputable def b : ℝ := (1/2)^(-1.2)
noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 5)

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l197_197391


namespace correlated_relationships_l197_197474

-- Define relationships
inductive Relationship
| AgeWealth
| CurveCoordinates
| AppleProductionClimate
| TreeDiameterHeight

def is_correlational : Relationship → Prop
| Relationship.AgeWealth := True
| Relationship.CurveCoordinates := False
| Relationship.AppleProductionClimate := True
| Relationship.TreeDiameterHeight := True

theorem correlated_relationships :
  ∀ r, r = Relationship.AgeWealth ∨ r = Relationship.AppleProductionClimate ∨ r = Relationship.TreeDiameterHeight ↔ is_correlational r := 
by
  intro r
  cases r
  case AgeWealth         => exact Iff.intro (λ _, True.intro) (λ _, True.intro)
  case CurveCoordinates  => exact Iff.intro (λ h, by contradiction) (λ h, False.elim h)
  case AppleProductionClimate => exact Iff.intro (λ _, True.intro) (λ _, True.intro)
  case TreeDiameterHeight => exact Iff.intro (λ _, True.intro) (λ _, True.intro)

#check correlated_relationships

end correlated_relationships_l197_197474


namespace probability_units_digit_meets_condition_proof_l197_197732

   noncomputable def probability_units_digit_meets_condition : ℚ :=
     let possible_digits := finset.range 10
     let satisfying_digits := finset.filter (λ n, n < 7 ∨ (n % 2 = 0)) possible_digits
     (satisfying_digits.card : ℚ) / possible_digits.card

   theorem probability_units_digit_meets_condition_proof :
     probability_units_digit_meets_condition = 4 / 5 := 
   by
     sorry
   
end probability_units_digit_meets_condition_proof_l197_197732


namespace number_of_irrationals_l197_197939

-- Definitions for the given numbers
def zero : ℝ := 0
def pi : ℝ := Real.pi
def frac22div7 : ℝ := 22 / 7
def sqrt2 : ℝ := Real.sqrt 2
def neg_sqrt9 : ℝ := -Real.sqrt 9

-- Definitions for rational and irrational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

def is_irrational (x : ℝ) : Prop := ¬is_rational x

-- The problem statement
theorem number_of_irrationals : ({zero, pi, frac22div7, sqrt2, neg_sqrt9}.filter is_irrational).card = 2 := 
sorry

end number_of_irrationals_l197_197939


namespace ratio_of_roots_l197_197029

noncomputable def m_values : (m1 m2 : ℝ) :=
∃ (m1 m2 : ℝ), 9 * m1^2 - 40 * m1 + 4 = 0 ∧ 9 * m2^2 - 40 * m2 + 4 = 0

theorem ratio_of_roots (m1 m2 : ℝ) (h : m_values m1 m2) :
  m1 = m2 → (m1 / m2 + m2 / m1 = 42.5) :=
sorry

end ratio_of_roots_l197_197029


namespace common_ratio_geometric_sequence_l197_197421

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Define the second term, third term, and sixth term as parts of the arithmetic sequence
def a2 (a1 d : ℝ) : ℝ := arithmetic_sequence a1 d 2
def a3 (a1 d : ℝ) : ℝ := arithmetic_sequence a1 d 3
def a6 (a1 d : ℝ) : ℝ := arithmetic_sequence a1 d 6

-- Assume conditions and prove that the common ratio r is 3
theorem common_ratio_geometric_sequence (a1 d : ℝ) (h : d ≠ 0) :
  let r := a3 a1 d / a2 a1 d 
  in r = 3 :=
by
  let a₁ := a1
  let b := d
  have hₑq : r = 3 := sorry
  exact hₑq

end common_ratio_geometric_sequence_l197_197421


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197669

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197669


namespace arithmetic_sequence_8th_term_l197_197082

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197082


namespace intervals_of_increase_range_of_function_l197_197878

-- Definitions and lemmas related to trigonometric functions and derivatives would be typically imported from Mathlib

-- Question 1: Proving the intervals of increase for the given function when ω = 1
theorem intervals_of_increase 
  (k : ℤ) 
  : 
    ∀ x : ℝ, 
    let f : ℝ → ℝ := λ x, sin (2 * x - π / 6) + 2 * (cos x) ^ 2 - 1,
    let intervals := (-π/3 + k * π, π/6 + k * π),
    ∃ x1 x2, (x1 ∈ intervals ∧ x2 ∈ intervals) ∧ (∀ x ∈ intervals, f(x) < f(x + ε)).

-- Question 2: Proving the range of the given function on [0, π/8] when omega is 16/3
theorem range_of_function 
  :
    let ω := (8 : ℝ) / 3,
    let f : ℝ → ℝ := λ x, sin (omega * x + π / 6),
    ∀ x ∈ Icc (0 : ℝ) (π / 8), f(x) ∈ Icc (1 / 2) 1 :=
by 
  sorry

-- End of Lean 4 statements


end intervals_of_increase_range_of_function_l197_197878


namespace student_movement_l197_197712

theorem student_movement :
  let students_total := 10
  let students_front := 3
  let students_back := 7
  let chosen_students := 2
  let ways_to_choose := Nat.choose students_back chosen_students
  let ways_to_place := (students_front + 1) * (students_front + chosen_students)
  in ways_to_choose * ways_to_place = 420 := by
  sorry

end student_movement_l197_197712


namespace minimum_value_ineq_l197_197546

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l197_197546


namespace arithmetic_sequence_8th_term_l197_197111

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197111


namespace triangle_angle_bisector_theorem_l197_197919

theorem triangle_angle_bisector_theorem
  (A B C Q : Type)
  [HasLabel A] [HasLabel B] [HasLabel C] [HasLabel Q]
  (h_ratio : ∀ (AC CB AB : ℕ), AC = 4 * AB ∧ CB = 5 * AB)
  (h_ab : ∀ (AB : ℕ), AB = 18)
  (h_intersection : ∀ (AQ QB : ℕ), AQ / QB = 4 / 5) :
  AQ / QB = 4 / 5 :=
sorry

end triangle_angle_bisector_theorem_l197_197919


namespace probability_even_integer_division_l197_197612

theorem probability_even_integer_division (r k : ℤ)
  (hr : -5 < r ∧ r < 7)
  (hk : 2 ≤ k ∧ k ≤ 9) :
  let R_even := {r : ℤ | -5 < r ∧ r < 7 ∧ even r}
      K := {k : ℤ | 2 ≤ k ∧ k ≤ 9}
      valid_pairs := { (r, k) : ℤ × ℤ | r ∈ R_even ∧ k ∈ K ∧ k ∣ r }
      total_pairs := R_even.prod K
  in (valid_pairs.card : ℚ) / (total_pairs.card : ℚ) = 17 / 48 :=
by
  sorry

end probability_even_integer_division_l197_197612


namespace square_of_number_ending_in_5_l197_197578

theorem square_of_number_ending_in_5 (a : ℤ) :
  (10 * a + 5) * (10 * a + 5) = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_number_ending_in_5_l197_197578


namespace prod_ineq_l197_197593

theorem prod_ineq (n : ℕ) : 
  (\prod k in finset.range n, (2 * k + 1)) / (\prod k in finset.range n, (2 * (k + 1))) ≤ 1 / real.sqrt (2 * n + 1) := 
sorry

end prod_ineq_l197_197593


namespace balls_evenly_spaced_probability_l197_197831

noncomputable def probability_evenly_spaced_balls : ℚ :=
  let total_probability := ∑' a, ∑' n, (3:ℚ) ^ (-4 * a - 6 * n) in
  let permutations := (1 / total_probability) * (4!) in
  permutations

theorem balls_evenly_spaced_probability :
  probability_evenly_spaced_balls = 1 / 2430 :=
by
  sorry

end balls_evenly_spaced_probability_l197_197831


namespace arithmetic_seq_8th_term_l197_197065

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197065


namespace smallest_subset_coprime_l197_197825

def minimal_n {S : set ℕ} (hS : S = {x | 1 ≤ x ∧ x ≤ 150 ∧ prime x ∨ ¬prime x}) : ℕ :=
  111

theorem smallest_subset_coprime (S : set ℕ) (hS : S = {x | 1 ≤ x ∧ x ≤ 150}) (h_prime : ∀ x ∈ S, prime x ∨ ¬prime x) :
  ∃ n, (∀ subset : finset ℕ, subset.card = n → ∀ A ⊂ subset, A.card = 4 → pairwise coprime A) ∧ n = 111 :=
by {
  use minimal_n hS,
  sorry
}

end smallest_subset_coprime_l197_197825


namespace queue_adjustments_l197_197709

theorem queue_adjustments (students_front : ℕ) (students_back : ℕ) (students_move : ℕ) :
  students_front = 3 → students_back = 7 → students_move = 2 → 
  (∃ ways_to_adjust : ℕ, ways_to_adjust = 420) :=
by
  intros h1 h2 h3
  use 420
  sorry

end queue_adjustments_l197_197709


namespace cannot_be_reflection_l197_197191

-- Definitions based on given conditions
variable {T : Type*} [triangle T]
variable {A B C D E : Type*} [triangle A] [triangle B] [triangle C] [triangle D] [triangle E]

-- Statement to prove
theorem cannot_be_reflection (T_reflect : ∀ t : T, ∃ a : A, ∃ b : B, ∃ d : D, ∃ e : E,
     (reflection t a) ∧ (reflection t b) ∧ (reflection t d) ∧ (reflection t e)) : 
    ¬ (reflection T C) :=
sorry

end cannot_be_reflection_l197_197191


namespace parabola_focus_l197_197137

theorem parabola_focus (a : ℝ) :
  (∀ (x : ℝ), (x, a * x^2) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2}) ∧ 
  (∀ (p : ℝ × ℝ), p ∈ {p : ℝ × ℝ | p.2 = 1}) → 
  (0, -1) ∈ {f : ℝ × ℝ | is_focus f (y = a * x^2) (y = 1)} :=
sorry

end parabola_focus_l197_197137


namespace number_of_chameleons_l197_197936

noncomputable def lizard_species : Type := { b : Bool // b = true ∨ b = false }
def is_gecko (l : lizard_species) : Prop := l.val
def is_chameleon (l : lizard_species) : Prop := ¬l.val

variables (Brian Chris LeRoy Mike Sam : lizard_species)

-- Conditions
axiom Brian_statement : Brian.val = Mike.val
axiom Chris_statement : is_chameleon LeRoy ∧ is_chameleon Sam
axiom LeRoy_statement : is_chameleon Chris
axiom Mike_statement : (is_gecko Brian ∧ is_gecko Mike ∧ is_gecko Chris) ∨ 
                       (is_gecko Brian ∧ is_gecko Mike ∧ is_gecko LeRoy) ∨
                       (is_gecko Brian ∧ is_gecko Mike ∧ is_gecko Sam) ∨
                       (is_gecko Brian ∧ is_gecko Chris ∧ is_gecko LeRoy) ∨
                       (is_gecko Brian ∧ is_gecko Chris ∧ is_gecko Sam) ∨
                       (is_gecko Brian ∧ is_gecko LeRoy ∧ is_gecko Sam) ∨
                       (is_gecko Mike ∧ is_gecko Chris ∧ is_gecko LeRoy) ∨
                       (is_gecko Mike ∧ is_gecko Chris ∧ is_gecko Sam) ∨
                       (is_gecko Mike ∧ is_gecko LeRoy ∧ is_gecko Sam) ∨
                       (is_gecko Chris ∧ is_gecko LeRoy ∧ is_gecko Sam)
axiom Sam_statement : Brian.val = Chris.val

theorem number_of_chameleons : 
  (∃ S : finset lizard_species, S.card = 5 ∧ S.filter is_chameleon).card = 2 :=
sorry

end number_of_chameleons_l197_197936


namespace right_triangle_bc_length_l197_197932

noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_bc_length (A B C M : Point) 
                           (h_right_angle : ∠(A, B, C) = π / 2)
                           (h_AB : dist A B = 2)
                           (h_AC : dist A C = 4)
                           (h_median : dist A M = dist B C / 2)
                           (h_midpoint_M : midpoint B C = M) :
  dist B C = 2 * sqrt_5 :=
by
  sorry

end right_triangle_bc_length_l197_197932


namespace bridge_length_l197_197260

theorem bridge_length 
  (walking_speed_km_per_hr : ℝ)
  (time_to_cross_minutes : ℝ)
  (km_to_m : ℝ := 1000)
  (hr_to_min : ℝ := 60) :
      walking_speed_km_per_hr = 10 → time_to_cross_minutes = 3 →
      let walking_speed_m_per_min := walking_speed_km_per_hr * km_to_m / hr_to_min in
      let length_of_bridge_m := walking_speed_m_per_min * time_to_cross_minutes in
      length_of_bridge_m = 500 :=
by
  intros h_speed h_time
  let walking_speed_m_per_min := walking_speed_km_per_hr * km_to_m / hr_to_min
  have h_speed_conv : walking_speed_m_per_min = 166.67 := by sorry
  let length_of_bridge_m := walking_speed_m_per_min * time_to_cross_minutes
  have h_length : length_of_bridge_m = 500 := by sorry
  exact h_length

end bridge_length_l197_197260


namespace solution_set_of_inequality_l197_197170

theorem solution_set_of_inequality :
  {x : ℝ | |x + 1| - |x - 5| < 4} = {x : ℝ | x < 4} :=
sorry

end solution_set_of_inequality_l197_197170


namespace range_of_2x_minus_y_l197_197838

theorem range_of_2x_minus_y (x y : ℝ) (hx : 0 < x ∧ x < 4) (hy : 0 < y ∧ y < 6) : -6 < 2 * x - y ∧ 2 * x - y < 8 := 
sorry

end range_of_2x_minus_y_l197_197838


namespace volume_within_one_unit_l197_197352

theorem volume_within_one_unit 
  (length width height : ℝ) 
  (length_eq : length = 2) 
  (width_eq : width = 5) 
  (height_eq : height = 6) 
  (m n p : ℕ) 
  (vol : ℝ) 
  (vol_eq : vol = (492 + 43 * Real.pi) / 3)
  (sum_eq : m + n + p = 538): 
  ∃ vol : ℝ, ∃ m n p : ℕ, (vol = (492 + 43 * Real.pi) / 3) ∧ (m + n + p = 538) := 
by 
  use (492 + 43 * Real.pi) / 3
  use 492, 43, 3
  sorry

end volume_within_one_unit_l197_197352


namespace true_discount_correct_l197_197648

-- Definitions based on conditions
def amount_of_bill : ℝ := 2360
def bankers_discount : ℝ := 424.8

-- Definition of true discount based on the given formula
def true_discount (BD A : ℝ) : ℝ := BD / (1 + BD / A)

-- The math proof problem statement
theorem true_discount_correct (BD A : ℝ) (hBD : BD = bankers_discount) (hA : A = amount_of_bill) :
  true_discount BD A = 360 := 
by
  simp [true_discount, hBD, hA]
  -- add necessary calculation steps to verify
  sorry

end true_discount_correct_l197_197648


namespace observe_three_cell_types_l197_197725

def biology_experiment
  (material : Type) (dissociation_fixative : material) (acetic_orcein_stain : material) (press_slide : Prop) : Prop :=
  ∃ (testes : material) (steps : material → Prop),
    steps testes ∧ press_slide ∧ (steps dissociation_fixative) ∧ (steps acetic_orcein_stain)

theorem observe_three_cell_types (material : Type)
  (dissociation_fixative acetic_orcein_stain : material)
  (press_slide : Prop)
  (steps : material → Prop) :
  biology_experiment material dissociation_fixative acetic_orcein_stain press_slide →
  ∃ (metaphase_of_mitosis metaphase_of_first_meiosis metaphase_of_second_meiosis : material), 
    steps metaphase_of_mitosis ∧ steps metaphase_of_first_meiosis ∧ steps metaphase_of_second_meiosis :=
sorry

end observe_three_cell_types_l197_197725


namespace perfect_squares_less_than_100_l197_197463

theorem perfect_squares_less_than_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k^2)}.card = 9 :=
begin
  sorry
end

end perfect_squares_less_than_100_l197_197463


namespace lateral_surface_area_of_cylinder_l197_197850

theorem lateral_surface_area_of_cylinder (side_length : ℝ) (area : ℝ) (h : ℝ) (circumference : ℝ) : 
  area = 1 ∧ side_length ^ 2 = area ∧ h = side_length ∧ circumference = 1 → 
  (2 * side_length * circumference = 2 * π) :=
by
  intro h1
  simp at h1
  sorry

end lateral_surface_area_of_cylinder_l197_197850


namespace sandy_friday_hours_l197_197056

-- Define the conditions
def hourly_rate := 15
def saturday_hours := 6
def sunday_hours := 14
def total_earnings := 450

-- Define the proof problem
theorem sandy_friday_hours (F : ℝ) (h1 : F * hourly_rate + saturday_hours * hourly_rate + sunday_hours * hourly_rate = total_earnings) : F = 10 :=
sorry

end sandy_friday_hours_l197_197056


namespace shaded_region_area_l197_197745

/-- A rectangle measuring 12cm by 8cm has four semicircles drawn with their diameters as the sides
of the rectangle. Prove that the area of the shaded region inside the rectangle but outside
the semicircles is equal to 96 - 52π (cm²). --/
theorem shaded_region_area (A : ℝ) (π : ℝ) (hA : A = 96 - 52 * π) : 
  ∀ (length width r1 r2 : ℝ) (hl : length = 12) (hw : width = 8) 
  (hr1 : r1 = length / 2) (hr2 : r2 = width / 2),
  (length * width) - (2 * (1/2 * π * r1^2 + 1/2 * π * r2^2)) = A := 
by 
  sorry

end shaded_region_area_l197_197745


namespace proof_problem_l197_197392

variables (x y : ℝ)

theorem proof_problem
  (hx : x > 0)
  (hy : y > 0)
  (hxy : x - y > log (y / x))
  : x > y ∧ x + 1 / y > y + 1 / x ∧ 1 / (2 ^ x) < 2 ^ (-y) :=
by
  sorry

end proof_problem_l197_197392


namespace express_y_in_terms_of_x_l197_197885

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 1) : y = -3 * x + 1 := 
by
  sorry

end express_y_in_terms_of_x_l197_197885


namespace max_value_of_f_l197_197635

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (π / 2 - x)

theorem max_value_of_f : ∃ x : ℝ, f x = 5 :=
by 
  sorry

end max_value_of_f_l197_197635


namespace part_a_part_b_part_c_l197_197576

-- Part (a)
theorem part_a (p q r : ℕ) (h : p + q + r = 9) (choices: ℕ := 3) :
  let P := (Nat.choose 9 3) * 2^6 / (3^9 : ℝ) in P = 0.27385 :=
sorry

-- Part (b)
theorem part_b (p q r : ℕ) (h : p + q + r = 9) (choices: ℕ := 3) :
  let P := (Nat.factorial 9 / ((Nat.factorial 3)^3)) / (3^9 : ℝ) in P = 0.68265 :=
sorry

-- Part (c)
theorem part_c (p q r : ℕ) (h : p + q + r = 9) (choices: ℕ := 3) :
  let P := (Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)) / (3^9 : ℝ) in P = 0.064 :=
sorry

end part_a_part_b_part_c_l197_197576


namespace even_perfect_square_factors_count_l197_197454

theorem even_perfect_square_factors_count :
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card)) in
  factors_count = 30 :=
by
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card))
  have h_factors_count : factors_count = 30 := sorry
  exact h_factors_count

end even_perfect_square_factors_count_l197_197454


namespace arithmetic_seq_8th_term_l197_197068

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197068


namespace necessary_but_not_sufficient_l197_197517

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l197_197517


namespace general_formula_sum_first_n_terms_l197_197405

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}
axiom a1_a2 : a 1 + a 2 = 6
axiom a2_a3 : a 2 + a 3 = 10

-- The result for question I
theorem general_formula : (∀ n : ℕ, a n = 2 * n) :=
by sorry

-- Sequence of sums of consecutive terms
def b (a : ℕ → ℝ) (n : ℕ) := a n + a (n + 1)

-- Prove the sum of first n terms of the sequence {a_n + a_{n+1}}
theorem sum_first_n_terms (n : ℕ) : (∑ i in finset.range n, b a i) = 2 * n^2 + 4 * n :=
by sorry

end general_formula_sum_first_n_terms_l197_197405


namespace max_people_not_in_any_club_l197_197178

theorem max_people_not_in_any_club (total_people : ℕ) 
  (members_M : ℕ) (members_S : ℕ) (members_Z : ℕ) 
  (exclusive_M : ∀ x, x ∈ M → x ∉ S ∧ x ∉ Z) :
  total_people = 60 →
  members_M = 16 →
  members_S = 18 →
  members_Z = 11 →
  (∀ x, x ∈ M → x ∉ S ∧ x ∉ Z) →
  let max_people_nonmembers := total_people - (members_M + members_S) in
  max_people_nonmembers = 26 :=
by 
  intros h1 h2 h3 h4 h5;
  rw [h1, h2, h3, h4];
  let max_nonmembers := 60 - (16 + 18);
  show max_nonmembers = 26;
  sorry

end max_people_not_in_any_club_l197_197178


namespace sum_infinite_partial_fraction_l197_197303

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197303


namespace eval_9xp1_l197_197899

theorem eval_9xp1 (x : ℝ) (h : 3^(2 * x) = 5) : 9^(x + 1) = 45 :=
by sorry

end eval_9xp1_l197_197899


namespace ratio_minimum_l197_197404

-- Define points on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distances between points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Prove the inequality
theorem ratio_minimum (A B C D O : Point)
  (h_square : A.x ≠ B.x ∧ A.y = B.y ∧ C.x ≠ D.x ∧ C.y = D.y
    ∧ distance A B = distance B C ∧ distance C D = distance D A) :
  (distance O A + distance O C) / (distance O B + distance O D) ≥ 1 / real.sqrt 2 :=
by
  sorry

end ratio_minimum_l197_197404


namespace largest_good_number_is_576_smallest_bad_number_is_443_l197_197828

def is_good_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℤ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

def largest_good_number : ℕ := 576

def smallest_bad_number : ℕ := 443

theorem largest_good_number_is_576 : ∀ M : ℕ, is_good_number M → M ≤ 576 := 
by
  sorry

theorem smallest_bad_number_is_443 : ∀ M : ℕ, ¬ is_good_number M → 443 ≤ M :=
by
  sorry

end largest_good_number_is_576_smallest_bad_number_is_443_l197_197828


namespace combined_tax_rate_is_approx_30_68_l197_197004

def johns_initial_tax_rate := 0.30
def ingrids_initial_tax_rate := 0.40
def johns_income := 58000
def ingrids_income := 72000
def johns_tax_reduction := 2000
def ingrids_exemption_rate := 0.15

def combined_tax_rate (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (john_income : ℕ) (ingrid_income : ℕ) (john_reduction : ℕ) (ingrid_exemption : ℝ) : ℝ := 
  let john_initial_tax := john_tax_rate * john_income 
  let ingrid_initial_tax := ingrid_tax_rate * ingrid_income
  let john_final_tax := john_initial_tax - john_reduction
  let ingrid_income_after_exemption := ingrid_income - (ingrid_exemption * ingrid_income)
  let ingrid_final_tax := ingrid_tax_rate * ingrid_income_after_exemption
  let combined_tax := john_final_tax + ingrid_final_tax
  let combined_income := john_income + ingrid_income
  (combined_tax / combined_income) * 100

theorem combined_tax_rate_is_approx_30_68 :
  combined_tax_rate johns_initial_tax_rate ingrids_initial_tax_rate johns_income ingrids_income johns_tax_reduction ingrids_exemption_rate ≈ 30.68 := 
sorry

end combined_tax_rate_is_approx_30_68_l197_197004


namespace main_theorem_l197_197403

-- Defining the sequence and properties in conditions
noncomputable def a : ℕ+ → ℝ := sorry

def condition1 (n : ℕ+) : ℝ :=
a (n + 1) = a 1 ^ 2 * a 2 ^ 2 * a n ^ 2 - 3

def condition2 : Prop :=
(1 / 2) * (a 1 + real.sqrt (a 2 - 1)) ∈ ℕ+

-- Main theorem that needs to be proven
theorem main_theorem (n : ℕ+) (h1 : ∀ n, condition1 n) (h2 : condition2) :
  (1 / 2) * (finset.prod (finset.range n) (λ i, a (i + 1)) + real.sqrt (a (n + 1) - 1)) ∈ ℕ+ :=
sorry

end main_theorem_l197_197403


namespace complex_number_condition_l197_197871

theorem complex_number_condition (z1 z2 : ℂ) (h : abs (z1 - conj z2) = abs (1 - z1 * z2)) :
  abs z1 = 1 ∨ abs z2 = 1 := 
by sorry

end complex_number_condition_l197_197871


namespace bus_seating_capacity_l197_197498

-- Conditions
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seat_capacity : ℕ := 3
def back_seat_capacity : ℕ := 9
def total_seats : ℕ := left_side_seats + right_side_seats

-- Proof problem statement
theorem bus_seating_capacity :
  (total_seats * seat_capacity) + back_seat_capacity = 90 := by
  sorry

end bus_seating_capacity_l197_197498


namespace number_of_cars_in_section_H_l197_197575

theorem number_of_cars_in_section_H :
  (let rows_G := 15 in
   let cars_per_row_G := 10 in
   let rows_H := 20 in
   let cars_per_row_H := 9 in
   let minutes := 30 in
   let cars_per_minute := 11 in
   let total_cars_G := rows_G * cars_per_row_G in
   let total_cars_walked := minutes * cars_per_minute in
   let total_cars_H := rows_H * cars_per_row_H in
   total_cars_H = total_cars_walked - total_cars_G) := {
  sorry
}

end number_of_cars_in_section_H_l197_197575


namespace range_of_a_l197_197972

def f : ℝ → ℝ :=
  λ x, if x > 1 then 2 * f (x - 2) else (if -1 ≤ x ∧ x ≤ 1 then 1 - |x| else 0)

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∃ s : finset ℝ, (∀ x ∈ s, 1 ≤ x ∧ x ≤ 5) ∧ s.card = 5 ∧ ∀ x ∈ s, f x = real.log x / real.log a) →
  sqrt 2 < a ∧ a < sqrt real.exp 1 := 
sorry

end range_of_a_l197_197972


namespace shift_graph_cos_eq_l197_197188

theorem shift_graph_cos_eq :
  ∀ x, cos (2 * (x - π / 8)) = cos (2 * x - π / 4) :=
begin
  intro x,
  sorry
end

end shift_graph_cos_eq_l197_197188


namespace relationship_among_p_q_a_b_l197_197020

open Int

variables (a b p q : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = p) (h3 : Nat.lcm a b = q)

theorem relationship_among_p_q_a_b : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end relationship_among_p_q_a_b_l197_197020


namespace hyperbola_focus_property_l197_197410

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

noncomputable def left_focus : (ℝ × ℝ) := (-2, 0)
noncomputable def right_focus : (ℝ × ℝ) := (2, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem hyperbola_focus_property (P : ℝ × ℝ) (h₁ : point_on_hyperbola P)
  (h₂ : distance P left_focus = 5) : distance P right_focus = 3 ∨ distance P right_focus = 7 :=
by sorry

end hyperbola_focus_property_l197_197410


namespace probability_telepathically_linked_l197_197583

def telepathically_linked (a b : ℕ) : Prop :=
  |a - b| ≤ 1

theorem probability_telepathically_linked :
  (∑ a in Finset.range 10, ∑ b in Finset.range 10, if telepathically_linked a b then (1 : ℚ) else 0) / (10 * 10) = 7 / 25 := by
  sorry

end probability_telepathically_linked_l197_197583


namespace min_value_of_frac_l197_197849

open Real

theorem min_value_of_frac (x : ℝ) (hx : x > 0) : 
  ∃ (t : ℝ), t = 2 * sqrt 5 + 2 ∧ (∀ y, y > 0 → (x^2 + 2 * x + 5) / x ≥ t) :=
by
  sorry

end min_value_of_frac_l197_197849


namespace conjugate_of_z_l197_197619

-- Define the given complex number
def z : ℂ := (1 - Complex.i) * Complex.i

-- Define the expected conjugate result
def expected_conjugate : ℂ := 1 - Complex.i

-- The problem statement: Prove that the conjugate of z equals the expected result
theorem conjugate_of_z :
  Complex.conj z = expected_conjugate := by 
  -- Placeholder for the proof
  sorry

end conjugate_of_z_l197_197619


namespace area_of_triangle_AMB_l197_197437

noncomputable def hyperbola := {x : ℝ × ℝ | (x.1^2 / 4) - (x.2^2 / 9) = 1}

variable (A B : ℝ × ℝ)
variable (M : {x : ℝ × ℝ // hyperbola x})
variable (angle_AMB : ∠ A M.1 B = 120)

theorem area_of_triangle_AMB 
  (hyp1: ∃ A B : ℝ × ℝ, is_focus_of_hyperbola A ∧ is_focus_of_hyperbola B) 
  (hyp2: M ∈ hyperbola) 
  (hyp3: angle_AMB = 120) :
  ∃ area : ℝ, area = 2 * sqrt 3 :=
sorry

end area_of_triangle_AMB_l197_197437


namespace zorbs_of_60_deg_l197_197044

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg_l197_197044


namespace proposition_1_proposition_3_main_proof_l197_197413

variables {l m : Type} {α β : Type}
variables (IsPerpendicular : α → l → Prop) (IsParallel : α → β → Prop)
variables (IsContainedIn : m → β → Prop)

-- Given conditions:
axiom l_perpendicular_to_alpha : IsPerpendicular α l
axiom m_contained_in_beta : IsContainedIn m β

-- The propositions:
theorem proposition_1 : IsParallel α β → IsPerpendicular l m :=
sorry

theorem proposition_3 : IsPerpendicular l m → IsParallel α β :=
sorry

theorem main_proof : (IsParallel α β → IsPerpendicular l m) ∧ (IsPerpendicular l m → IsParallel alpha β) :=
begin
  split,
  { exact proposition_1, },
  { exact proposition_3,}
end

end proposition_1_proposition_3_main_proof_l197_197413


namespace cube_surface_area_including_inside_l197_197270

theorem cube_surface_area_including_inside 
  (original_edge_length : ℝ) 
  (hole_side_length : ℝ) 
  (original_cube_surface_area : ℝ)
  (removed_hole_area : ℝ)
  (newly_exposed_internal_area : ℝ) 
  (total_surface_area : ℝ) 
  (h1 : original_edge_length = 3)
  (h2 : hole_side_length = 1)
  (h3 : original_cube_surface_area = 6 * (original_edge_length * original_edge_length))
  (h4 : removed_hole_area = 6 * (hole_side_length * hole_side_length))
  (h5 : newly_exposed_internal_area = 6 * 4 * (hole_side_length * hole_side_length))
  (h6 : total_surface_area = original_cube_surface_area - removed_hole_area + newly_exposed_internal_area) : 
  total_surface_area = 72 :=
by
  sorry

end cube_surface_area_including_inside_l197_197270


namespace necessary_but_not_sufficient_condition_l197_197513

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l197_197513


namespace arithmetic_sequence_eighth_term_l197_197097

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197097


namespace trigonometric_identity_l197_197714

theorem trigonometric_identity
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos70 : cos (70 * π / 180))
  (cos20 : cos (20 * π / 180))
  (sin225 : sin (225 * π / 180)) :
  (cos70 * cos20) / (1 - 2 * sin225^2) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l197_197714


namespace limit_of_sequence_l197_197588

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (3 * n - 2) / (2 * n - 1)) :
  Tendsto a_n atTop (𝓝 (3 / 2)) :=
by
  sorry

end limit_of_sequence_l197_197588


namespace abs_neg_three_l197_197132

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l197_197132


namespace intersect_EG_QR_midpoint_l197_197930

variables {E F G H Q R S : Type}
variables [parallelogram E F G H]

variables (EF_length EH_length : ℝ)
variable (k : ℝ)

-- Conditions
def EQ_length := (13 : ℝ) * k
def ER_length := (13 : ℝ) * k

variable (S_divides_EG : E → G → S → Prop) -- S divides EG at midpoint

theorem intersect_EG_QR_midpoint
  (H1 : EF_length = 200 * k)
  (H2 : EH_length = 500 * k)
  (H3 : ∃ S, (S_divides_EG Q R S) ∧ S = midpoint E G) :
  ∃ S, (distance E S) / (distance E G) = 1 / 2 :=
begin
  sorry
end

end intersect_EG_QR_midpoint_l197_197930


namespace cone_volume_l197_197414

theorem cone_volume (r l: ℝ) (h: ℝ) (hr : r = 1) (hl : l = 2) (hh : h = Real.sqrt (l^2 - r^2)) : 
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by 
  sorry

end cone_volume_l197_197414


namespace tangent_curve_l197_197867

theorem tangent_curve (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  a = 0 ∨ a = 2 := 
sorry

end tangent_curve_l197_197867


namespace hi_mom_box_office_revenue_scientific_notation_l197_197766

def box_office_revenue_scientific_notation (billion : ℤ) (revenue : ℤ) : Prop :=
  revenue = 5.396 * 10^9

theorem hi_mom_box_office_revenue_scientific_notation :
  box_office_revenue_scientific_notation 53.96 53960000000 :=
by
  sorry

end hi_mom_box_office_revenue_scientific_notation_l197_197766


namespace distinct_3_element_subsets_l197_197793

theorem distinct_3_element_subsets (n : ℕ) 
  (h1 : 0 < n)
  (A : finset (finset ℕ))
  (h2 : ∀ (i j : fin n), i ≠ j → ∃ k l m : ℕ, {k, l, m} ∈ A ∧ k ≠ l ∧ l ≠ m ∧ m ≠ k ∧ k ∈ finset.range n \ 
  ∧ l ∈ finset.range n ∧ m ∈ finset.range n)
  (h3 : ∀ (i j : fin n), i ≠ j → (A i ∩ A j).card ≠ 1) :
  ∃ (m : ℕ), n = 4 * m :=
by
  sorry

end distinct_3_element_subsets_l197_197793


namespace X_M_H_collinear_l197_197526

theorem X_M_H_collinear
  (A B C E F H M I_b I_c X : Point)
  (triangle_ABC : Triangle A B C)
  (altitude_BE : Altitude B E H)
  (altitude_CF : Altitude C F H)
  (midpoint_M : Midpoint M B C)
  (incenter_BMF : Incenter I_b (Triangle B M F))
  (incenter_CME : Incenter I_c (Triangle C M E))
  (internal_tangents_intersection : Intersection X (TangentCircle (CircumscribedCircle (Triangle B M F)) (CircumscribedCircle (Triangle C M E))))
  : Collinear X M H :=
sorry

end X_M_H_collinear_l197_197526


namespace abs_neg_three_l197_197127

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l197_197127


namespace triangle_perimeter_l197_197486

theorem triangle_perimeter (x : ℕ) 
  (h1 : x % 2 = 1) 
  (h2 : 7 - 2 < x)
  (h3 : x < 2 + 7) :
  2 + 7 + x = 16 := 
sorry

end triangle_perimeter_l197_197486


namespace perfect_squares_less_than_100_l197_197461

theorem perfect_squares_less_than_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k^2)}.card = 9 :=
begin
  sorry
end

end perfect_squares_less_than_100_l197_197461


namespace smallest_part_in_ratio_l197_197802

variable (b : ℝ)

theorem smallest_part_in_ratio (h : b = -2620) : 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  smallest_part = 100 :=
by 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  sorry

end smallest_part_in_ratio_l197_197802


namespace sequence_divisibility_l197_197444

theorem sequence_divisibility (a b c : ℤ) (u v : ℕ → ℤ) (N : ℕ)
  (hu0 : u 0 = 1) (hu1 : u 1 = 1)
  (hu : ∀ n ≥ 2, u n = 2 * u (n - 1) - 3 * u (n - 2))
  (hv0 : v 0 = a) (hv1 : v 1 = b) (hv2 : v 2 = c)
  (hv : ∀ n ≥ 3, v n = v (n - 1) - 3 * v (n - 2) + 27 * v (n - 3))
  (hdiv : ∀ n ≥ N, u n ∣ v n) : 3 * a = 2 * b + c :=
by
  sorry

end sequence_divisibility_l197_197444


namespace calculate_S_l197_197025

noncomputable def z : ℂ := (1/2 : ℂ) + (complex.I * (sqrt 3) / 2)

theorem calculate_S : z + 2 * z^2 + 3 * z^3 + 4 * z^4 + 5 * z^5 + 6 * z^6 = 3 - 3 * (complex.I * sqrt 3) :=
by
  sorry

end calculate_S_l197_197025


namespace sum_series_l197_197326

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197326


namespace arithmetic_mean_of_18_24_42_l197_197201

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l197_197201


namespace num_people_watched_last_week_l197_197940

variable (s f t : ℕ)
variable (h1 : s = 80)
variable (h2 : f = s - 20)
variable (h3 : t = s + 15)
variable (total_last_week total_this_week : ℕ)
variable (h4 : total_this_week = f + s + t)
variable (h5 : total_this_week = total_last_week + 35)

theorem num_people_watched_last_week :
  total_last_week = 200 := sorry

end num_people_watched_last_week_l197_197940


namespace rectangle_area_l197_197630

theorem rectangle_area (r l b: ℝ) (A_sq: ℝ) (A_rect: ℝ) 
  (h1 : l = (2/5) * r)
  (h2 : r = real.sqrt A_sq)
  (h3 : b = 10)
  (h4 : A_sq = 900)
  (h5 : A_rect = l * b) : 
  A_rect = 120 :=
by {
  sorry
}

end rectangle_area_l197_197630


namespace percentage_went_camping_l197_197907

-- Given conditions
variables {students : ℕ}
variables (N : ℕ) (P : ℕ)

-- 20 percent of the students went to the camping trip (P percent of N students)
-- 75 percent of the students who went to the camping trip did not take more than $100.
def went_camping : Prop := P = 20
def not_take_more_than_100 : Prop := 75 * P / 100 = 15

-- We want to prove that
theorem percentage_went_camping (N : ℕ) (h1 : went_camping N P) (h2 : not_take_more_than_100 N P) : P = 20 :=
by
  exact h1

end percentage_went_camping_l197_197907


namespace poly_inequality_l197_197956

noncomputable def poly (coeffs : List ℝ) : ℝ → ℝ :=
  λ x => coeffs.enum.sum (λ ⟨i, a_i⟩ => a_i * x ^ i)

def f (coeffs_f : List ℝ) := poly coeffs_f
def g (coeffs_g : List ℝ) := poly coeffs_g

lemma leading_coeff_pos (coeffs : List ℝ) (degree : ℕ) (h : degree < coeffs.length) :
  coeffs.nth_le degree h > 0 := sorry

def poly_ge (coeffs_f coeffs_g : List ℝ) (f g : ℝ → ℝ) :=
  ∃ r, (∀ i > r, coeffs_f.nth_le i sorry = coeffs_g.nth_le i sorry) ∧ (coeffs_f.nth_le r sorry > coeffs_g.nth_le r sorry) ∨ (f = g)

theorem poly_inequality
  (coeffs_f coeffs_g : List ℝ)
  (f := f coeffs_f)
  (g := g coeffs_g)
  (n_f : ℕ)
  (n_g : ℕ)
  (h_f : leading_coeff_pos coeffs_f n_f)
  (h_g : leading_coeff_pos coeffs_g n_g)
  (h1 : poly_ge coeffs_f coeffs_g f g) :
  ∀ x, f (f x) + g (g x) ≥ f (g x) + g (f x) := sorry

end poly_inequality_l197_197956


namespace max_of_min_values_l197_197796

def f (x : ℝ) : ℝ := -x^2 + 2 * x - 3
def interval (a : ℝ) : set ℝ := set.Icc (2 * a - 1) 2

theorem max_of_min_values (a : ℝ) (h1 : a ≤ 1.5) (h2 : 0 ≤ 2 * a - 1) :
  ∃ x, x ∈ interval a → ∀ y, y ∈ interval a → f y ≥ f x ∧ f x = -3 :=
sorry

end max_of_min_values_l197_197796


namespace infinite_series_converges_l197_197311

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197311


namespace floor_expression_value_l197_197381

theorem floor_expression_value (y : ℝ) (h : y = 6.2) : (Int.floor 6.5) * (Int.floor (2 / 3)) + (Int.floor 2) * 7.2 + (Int.floor 8.4) - y = 16.2 := by
  sorry

end floor_expression_value_l197_197381


namespace find_k_range_l197_197883

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1 / 3)

def g (x k : ℝ) : ℝ :=
abs (x - k) + abs (x - 1)

theorem find_k_range (k : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g x2 k) → (k ≤ 3 / 4 ∨ k ≥ 5 / 4) :=
by
  sorry

end find_k_range_l197_197883


namespace even_perfect_square_factors_count_l197_197453

theorem even_perfect_square_factors_count :
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card)) in
  factors_count = 30 :=
by
  let x := 2^6 * 5^3 * 7^8,
      factors_count := (({i | 0 ≤ i ∧ i ≤ 6 ∧ even i ∧ 2 ≤ i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 3 ∧ even i}.card) *
                       ({i | 0 ≤ i ∧ i ≤ 8 ∧ even i}.card))
  have h_factors_count : factors_count = 30 := sorry
  exact h_factors_count

end even_perfect_square_factors_count_l197_197453


namespace smallest_x_for_palindrome_l197_197214

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

theorem smallest_x_for_palindrome (x : ℕ) : (x + 7321 = 7447) → (is_palindrome (x + 7321)) → x = 126 :=
by
  sorry

end smallest_x_for_palindrome_l197_197214


namespace algebra_trigonometric_identitities_l197_197388

theorem algebra_trigonometric_identitities (α : ℝ) :
  (sin α + cos α = (√2) / 3) ∧ (π / 2 < α) ∧ (α < π) →
  (sin α - cos α = 4 / 3) ∧ 
  (sin^2 (π / 2 - α) - cos^2 (π / 2 + α) = -4 * √2 / 9) :=
by sorry

end algebra_trigonometric_identitities_l197_197388


namespace series_sum_l197_197339

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197339


namespace leon_total_payment_l197_197010

noncomputable def total_payment : ℕ :=
let toy_organizers_cost := 78 * 3 in
let gaming_chairs_cost := 83 * 2 in
let total_orders := toy_organizers_cost + gaming_chairs_cost in
let delivery_fee := total_orders * 5 / 100 in
total_orders + delivery_fee

theorem leon_total_payment : total_payment = 420 :=
by
  sorry

end leon_total_payment_l197_197010


namespace proof_problem_l197_197934

-- Parametric equation of the line:
def parametric_line (t : ℝ) : ℝ × ℝ := 
  ( (↑(sqrt 2) / 2) * t, (↑(sqrt 2) / 2) * t )

-- Equation of the circle:
def circle (x y : ℝ) : Prop := 
  x^2 + y^2 - 4 * x - 2 * y + 4 = 0

-- Polar equation of the circle
def polar_circle (ρ θ : ℝ) : Prop := 
  ρ^2 - 4 * ρ * cos θ - 2 * ρ * sin θ + 4 = 0

-- Distance between intersection points P and Q
def distance_PQ (t1 t2 : ℝ) : ℝ := 
  abs (t1 - t2)

theorem proof_problem :
  (forall t1 t2 : ℝ, 
    (parametric_line t1).1^2 + (parametric_line t1).2^2 - 4 * (parametric_line t1).1 - 2 * (parametric_line t1).2 + 4 = 0 → 
    (parametric_line t2).1^2 + (parametric_line t2).2^2 - 4 * (parametric_line t2).1 - 2 * (parametric_line t2).2 + 4 = 0 → 
    distance_PQ t1 t2 = sqrt 2) ∧
  ∀ ρ θ : ℝ, polar_circle ρ θ ↔ circle (ρ * cos θ) (ρ * sin θ) := 
sorry

end proof_problem_l197_197934


namespace min_value_of_f_on_interval_l197_197373

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan x ^ 2 - 4 * Real.tan x - 8 * (1 / Real.tan x) + 4 * (1 / Real.tan x) ^ 2 + 5

theorem min_value_of_f_on_interval :
  is_min (f x) (9 - 8 * Real.sqrt 2) (Ioo (Real.pi / 2) Real.pi) :=
sorry

end min_value_of_f_on_interval_l197_197373


namespace arithmetic_seq_8th_term_l197_197070

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197070


namespace oppose_estimation_l197_197647

-- Define the conditions
def survey_total : ℕ := 50
def favorable_attitude : ℕ := 15
def total_population : ℕ := 9600

-- Calculate the proportion opposed
def proportion_opposed : ℚ := (survey_total - favorable_attitude) / survey_total

-- Define the statement to be proved
theorem oppose_estimation : 
  proportion_opposed * total_population = 6720 := by
  sorry

end oppose_estimation_l197_197647


namespace possible_omega_values_l197_197435

theorem possible_omega_values (ω : ℕ) (φ : ℝ) 
  (h₀ : 0 < ω) (h₁ : ω ≤ 12) (h₂ : ω ∈ Nat.primes.map Nat.succ)
  (h₃ : 0 < φ ∧ φ < π)
  (h_sym : φ = π / 2)
  (h_not_mono : ¬ monotone_on (λ x : ℝ, cos (ω * x)) (Icc (π / 4) (π / 2))) :
  ω ∈ {3, 5, 6, 7, 8, 9, 10, 11, 12} :=
sorry

end possible_omega_values_l197_197435


namespace value_of_y_at_x_8_l197_197909

theorem value_of_y_at_x_8 
  (k : ℝ)
  (h1 : ∀ x : ℝ, y = k * x ^ (1/3))
  (h2 : y = 4 * sqrt 3 ∧ x = 64) : 
  y = 2 * sqrt 3 :=
sorry

end value_of_y_at_x_8_l197_197909


namespace solve_triangle_l197_197609

theorem solve_triangle (a c b : ℝ) (angle_ABC : ℝ) : Prop :=
  ∀ (triangle : Type)
    (BC : triangle -> ℝ)
    (AB : triangle -> ℝ)
    (CA : triangle -> ℝ)
    (angle : triangle -> ℝ -> Prop),
    (BC triangle = a) ∧
    (AB triangle - CA triangle = c - b) ∧
    (angle triangle ∠ABC = angle_ABC) →
    (∃ (A B C : triangle),
      ∃ (AB BC CA : ℝ), 
      ∃ (angles : triangle -> ℝ -> Prop), 
      BC = a ∧ 
      (AB - CA) = c - b ∧ 
      angles ∠ABC = angle_ABC)

end solve_triangle_l197_197609


namespace arithmetic_sequence_8th_term_l197_197109

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197109


namespace solve_quadratic_completing_square_l197_197997

theorem solve_quadratic_completing_square (x : ℝ) :
  (2 * x^2 - 4 * x - 1 = 0) ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
by
  sorry

end solve_quadratic_completing_square_l197_197997


namespace earth_unfit_in_cube_of_side_3_km_l197_197695

-- Definitions from conditions above
def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

def earth_population : ℝ :=
  7 * 10^9

-- Formulating the problem as a theorem
theorem earth_unfit_in_cube_of_side_3_km
  (side_length_km : ℝ)
  (h_side_length : side_length_km = 3)
  (h_earth_population : earth_population > 7 * 10^9)
  (significant_space_buildings : Bool) :
  let volume := cube_volume (3000 * side_length_km) in
  (volume / earth_population) < 4 → false := 
begin
  sorry
end

end earth_unfit_in_cube_of_side_3_km_l197_197695


namespace max_sum_sequence_reached_at_14_l197_197484

def sequence (n : ℕ) : ℕ := 43 - 3 * n

noncomputable def sum_sequence (n : ℕ) : ℕ := (n * (40 + 43 - 3 * n)) / 2

theorem max_sum_sequence_reached_at_14 : 
  ∀ (n : ℕ), (∃ k : ℕ, n = k) → (∃ m : ℕ in set.univ, sum_sequence 14 = sum_sequence m) :=
by
  sorry

end max_sum_sequence_reached_at_14_l197_197484


namespace matrix_mult_correct_l197_197345

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-2, 5]
]

def vector1 : Matrix (Fin 2) (Fin 1) ℤ := ![
  ![4],
  ![-3]
]

def result : Matrix (Fin 2) (Fin 1) ℤ := ![
  ![6],
  ![-23]
]

theorem matrix_mult_correct : (matrix1 ⬝ vector1) = result := by
  sorry

end matrix_mult_correct_l197_197345


namespace range_of_a_l197_197888

def A : Set ℝ := {x | (1 / (x - 2) + 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | (1/2)^a * 2^x = 4}

theorem range_of_a :
  (∀ x, x ∈ A ∪ B a ↔ x ∈ A) →
  a ∈ Set.Ico (-1 : ℝ) 0 :=
by
  intro h
  sorry

end range_of_a_l197_197888


namespace find_gamma_k_l197_197642

noncomputable def alpha (n d : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def beta (n r : ℕ) : ℕ := r^(n - 1)
noncomputable def gamma (n d r : ℕ) : ℕ := alpha n d + beta n r

theorem find_gamma_k (k d r : ℕ) (hk1 : gamma (k-1) d r = 200) (hk2 : gamma (k+1) d r = 2000) :
    gamma k d r = 387 :=
sorry

end find_gamma_k_l197_197642


namespace sum_series_l197_197323

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197323


namespace queue_adjustments_l197_197710

theorem queue_adjustments (students_front : ℕ) (students_back : ℕ) (students_move : ℕ) :
  students_front = 3 → students_back = 7 → students_move = 2 → 
  (∃ ways_to_adjust : ℕ, ways_to_adjust = 420) :=
by
  intros h1 h2 h3
  use 420
  sorry

end queue_adjustments_l197_197710


namespace composite_sum_l197_197053

theorem composite_sum (x y n : ℕ) (hx : x > 1) (hy : y > 1) (h : x^2 + x * y - y = n^2) :
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = x + y + 1 :=
sorry

end composite_sum_l197_197053


namespace ratio_of_w_to_y_l197_197164

theorem ratio_of_w_to_y (w x y z : ℚ)
  (h1 : w / x = 5 / 4)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 4) :
  w / y = 10 / 3 :=
sorry

end ratio_of_w_to_y_l197_197164


namespace infinite_series_converges_l197_197314

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197314


namespace sum_infinite_series_eq_l197_197288

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197288


namespace triangle_angle_and_side_conditions_l197_197942

variables {A B C a b c : ℝ}

theorem triangle_angle_and_side_conditions 
  (h1 : c * sin A - sqrt 3 * a * cos C = 0)
  (h2 : cos A = (2 * sqrt 7) / 7)
  (h3 : c = sqrt 14) :
  C = π / 3 ∧ (sin (π - A - C) = 3 * sqrt 21 / 14) ∧ 
  (b = 3 * sqrt 2) :=
by sorry

end triangle_angle_and_side_conditions_l197_197942


namespace arithmetic_seq_8th_term_l197_197072

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197072


namespace _l197_197560

-- Define the given problem conditions
variables (O A T B P : Type) [MetricSpace O]
variable [MetricSpace T]
variable [MetricSpace B]
variable [MetricSpace P]

-- Define constants
variables (OA OP AP : ℝ)
variables (circle_radius OP_value PT_value : ℝ)
variable h1 : OA = 8
variable h2 : OP = 3

-- Assume tangent and chord properties, right angle, etc.
axiom right_angle_OAT : ⦃x y z : O⦄ → Metric.angle O A T = π / 2

-- Power of a point theorem for the configuration
lemma length_of_PT
  (h_circle_radius : circle_radius = 8)
  (h_OP_value : OP_value = 3)
  (h_PT_value : PT_value = 25 / 3)
  (h_OAT_right_angle : Metric.angle O A T = π / 2) :
  PT_value = 25 / 3 :=
by
  sorry

end _l197_197560


namespace equation_of_parallel_line_l197_197139

theorem equation_of_parallel_line (m : ℝ) :
  (∀ x y, y = (3 / 2) * x + m ∧ (3, -1) ∈ (λ p : ℝ × ℝ, p.snd = (3 / 2) * p.fst + m)) →
  (3 * 3 - 2 * (-1) - 11 = 0) :=
by
  sorry

end equation_of_parallel_line_l197_197139


namespace water_fraction_after_replacements_l197_197723

theorem water_fraction_after_replacements (initial_volume : ℕ) (removed_volume : ℕ) (replacements : ℕ) :
  initial_volume = 16 → removed_volume = 4 → replacements = 4 →
  (3 / 4 : ℚ) ^ replacements = 81 / 256 :=
by
  intros h_initial_volume h_removed_volume h_replacements
  sorry

end water_fraction_after_replacements_l197_197723


namespace correct_calculation_l197_197227

theorem correct_calculation :
  (∀ (a b c d : ℝ),
    (2 * sqrt 5 * 3 * sqrt 5 ≠ 6 * sqrt 5) ∧
    (5 * sqrt 2 * 5 * sqrt 3 ≠ 5) ∧
    (sqrt 12 * sqrt 8 = 4 * sqrt 6) ∧
    (3 * sqrt 2 * 2 * sqrt 3 ≠ 6 * sqrt 5)) :=
by
  intros a b c d
  sorry

end correct_calculation_l197_197227


namespace sqrt_fraction_arith_sqrt_16_l197_197644

-- Prove that the square root of 4/9 is ±2/3
theorem sqrt_fraction (a b : ℕ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (h_a : a = 4) (h_b : b = 9) : 
    (Real.sqrt (a / (b : ℝ)) = abs (Real.sqrt a / Real.sqrt b)) :=
by
    rw [h_a, h_b]
    sorry

-- Prove that the arithmetic square root of √16 is 4.
theorem arith_sqrt_16 : Real.sqrt (Real.sqrt 16) = 4 :=
by
    sorry

end sqrt_fraction_arith_sqrt_16_l197_197644


namespace angle_measure_max_area_l197_197943

noncomputable def angle_A (a b c : ℝ) (h : b^2 + c^2 - a^2 + b * c = 0) : ℝ :=
  if h : ∀ a b c, b^2 + c^2 - a^2 + b * c = 0 then (2 * Real.pi) / 3 else 0

noncomputable def max_area_triangle (a b c : ℝ) (h : b^2 + c^2 - a^2 + b * c = 0) : ℝ :=
  if ha : a = Real.sqrt 3 then (Real.sqrt 3) / 4 else 0

theorem angle_measure (a b c : ℝ) (h : b^2 + c^2 - a^2 + b * c = 0) : 
  angle_A a b c h = (2 * Real.pi) / 3 :=
sorry

theorem max_area (a b c : ℝ) (h : b^2 + c^2 - a^2 + b * c = 0) (ha : a = Real.sqrt 3) : 
  max_area_triangle a b c h = (Real.sqrt 3) / 4 :=
sorry

end angle_measure_max_area_l197_197943


namespace find_t_from_line_l197_197362

theorem find_t_from_line (t : ℝ) : ((t = 1) ↔ ((t, 7) = λ x : ℝ, (((x + 7) - (0 + 1)) / (-6 + 0)))) := 
by 
  sorry

end find_t_from_line_l197_197362


namespace necessary_but_not_sufficient_l197_197516

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l197_197516


namespace longest_factor_link_and_count_l197_197957

-- Definitions for natural numbers and the factorization of x
variables (k m n : ℕ)
def x : ℕ := 5^k * 31^m * 1990^n

-- The prime factorization of 1990
def factorize_1990 := 2 * 5 * 199

-- Redefine x using the prime factorization of 1990
def x_factorized : ℕ := 2^n * 5^(k + n) * 31^m * 199^n

-- Lean proof statement
theorem longest_factor_link_and_count (h : x = x_factorized) : 
  L x = 3 * n + k + m + 3 ∧
  R x = (3 * n + k + m).factorial / ((n.factorial) ^ 2 * (k + n).factorial * m.factorial) :=
by {sorry}

end longest_factor_link_and_count_l197_197957


namespace sum_of_a_b_l197_197905

theorem sum_of_a_b (f : ℝ → ℝ) (a b : ℝ) (x1 x2 : ℝ) (h_f : f = λ x, (1/3) * x^3 + a * x^2 + b * x + 1)
  (h_deriv : deriv f = λ x, x^2 + 2 * a * x + b) (h_zeros : f' x1 = 0 ∧ f' x2 = 0) (h_distinct : x1 ≠ x2)
  (h_sorted : (x1, x2, 2) forms arithmetic progression ∨ (x1, x2, 2) forms geometric progression)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0) : a + b = 13 / 2 := by
  -- Placeholder for the proof
  sorry

end sum_of_a_b_l197_197905


namespace arithmetic_sequence_8th_term_l197_197121

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197121


namespace cube_edge_length_l197_197259

theorem cube_edge_length (edge_length : ℝ) 
    (base_length : ℝ) (base_width : ℝ) (rise : ℝ) 
    (h_base : base_length = 10) (h_width : base_width = 5) (h_rise : rise = 2.5)
    (h_volume : base_length * base_width * rise = edge_length^3) :
    edge_length = 5 :=
begin
  sorry
end

end cube_edge_length_l197_197259


namespace incorrect_value_of_Reema_marks_l197_197739

theorem incorrect_value_of_Reema_marks 
  (n : ℕ) (avg_incorrect : ℚ) (avg_correct : ℚ) (correct_mark : ℚ) (incorrect_sum : ℚ) (correct_sum : ℚ) (incorrect_mark : ℚ) :
  n = 35 →
  avg_incorrect = 72 →
  avg_correct = 71.71 →
  correct_mark = 56 →
  incorrect_sum = n * avg_incorrect →
  correct_sum = n * avg_correct →
  incorrect_sum - correct_mark + incorrect_mark = correct_sum →
  incorrect_mark = 46.85 :=
by
  intros n_val avg_inc val_avg_inc val_avg_corr val_correct val_inc_sum val_correct_sum eqn
  sorry

end incorrect_value_of_Reema_marks_l197_197739


namespace hannah_starting_time_l197_197449

-- Definitions based on conditions
def total_movie_duration : ℕ := 3 * 60 -- 3 hours in minutes.
def laptop_turn_off_time : ℕ × ℕ := (17, 44) -- 5:44 pm in 24-hour format as (hours, minutes).
def remaining_movie_time : ℕ := 36 -- 36 minutes.

-- Definition for the proving statement
theorem hannah_starting_time :
  let watched_time := total_movie_duration - remaining_movie_time in
  let turn_off_hour := laptop_turn_off_time.fst in
  let turn_off_minutes := laptop_turn_off_time.snd in
  let watched_hours := watched_time / 60 in
  let watched_minutes := watched_time % 60 in
  let start_hour := turn_off_hour - watched_hours in
  let start_minutes := turn_off_minutes - watched_minutes in
  start_hour = 15 ∧ start_minutes = 20 :=
by
  sorry

end hannah_starting_time_l197_197449


namespace find_integer_n_l197_197200

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 :=
by
  sorry

end find_integer_n_l197_197200


namespace equal_circumcircle_radii_l197_197048

theorem equal_circumcircle_radii
  (A B C O : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (hABC_acute : ∀ X Y Z : A, Angle X Y Z < 90)
  (hO_orthocenter : ∀ X Y Z : A, OrthoCenter O X Y Z) :
  ∀ R1 R2 R3 : ℝ,
  (IsCircumcircleRadius O A B R1) →
  (IsCircumcircleRadius O B C R2) →
  (IsCircumcircleRadius O C A R3) →
  R1 = R2 ∧ R2 = R3 := by
    sorry

end equal_circumcircle_radii_l197_197048


namespace calculate_fraction_l197_197776

theorem calculate_fraction :
  (5 * 6.factorial + 30 * 5.factorial) / 7.factorial = 30 / 7 :=
by sorry

end calculate_fraction_l197_197776


namespace central_angle_of_cone_is_192_degrees_l197_197918

-- Define the given constants
def radius : ℝ := 8
def slant_height : ℝ := 15

-- Define the circumference of the base circle
def circumference (r : ℝ) : ℝ := 2 * real.pi * r

-- Define the central angle calculation
def central_angle (circ : ℝ) (l : ℝ) : ℝ := (circ * 180) / (l * real.pi)

-- The theorem to be proven
theorem central_angle_of_cone_is_192_degrees :
  central_angle (circumference radius) slant_height = 192 :=
by
  sorry -- Proof is skipped as instructed

end central_angle_of_cone_is_192_degrees_l197_197918


namespace correct_parallelism_proposition_l197_197274

def parallel {α : Type} (a b : α) : Prop := sorry
def perpendicular {α : Type} (a b : α) : Prop := sorry
def within {α : Type} (a : α) (b : set α) : Prop := sorry

theorem correct_parallelism_proposition (a b c : Type) (α : set Type) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  ¬(perpendicular a b ∧ perpendicular b c → parallel a c) ∧
  ¬(parallel a α ∧ within b α → parallel a b) ∧
  ¬(parallel a b ∧ parallel b α → parallel a α) :=
by {
  -- Proof is skipped
  sorry
}

end correct_parallelism_proposition_l197_197274


namespace digit_place_value_ratio_l197_197938

theorem digit_place_value_ratio (n : ℚ) (h1 : n = 85247.2048) (h2 : ∃ d1 : ℚ, d1 * 0.1 = 0.2) (h3 : ∃ d2 : ℚ, d2 * 0.001 = 0.004) : 
  100 = 0.1 / 0.001 :=
by
  sorry

end digit_place_value_ratio_l197_197938


namespace find_angle_BAC_l197_197014

noncomputable def center (ω : Circle) : Point := sorry
noncomputable def midpoint_arc_not_containing (ω : Circle) (A : Point) (B : Point) (C : Point) : Point := sorry
noncomputable def orthocenter (ABC : Triangle) : Point := sorry
noncomputable def circumcircle (ABC : Triangle) : Circle := sorry
noncomputable def angle (A B C : Point) : ℝ := sorry

structure Triangle :=
(A B C : Point)

def example_triangle : Triangle := sorry

theorem find_angle_BAC (ABC : Triangle)
    (O : Point) (ω : Circle)
    (W H : Point)
    (h₁ : O = center (circumcircle ABC))
    (h₂ : W = midpoint_arc_not_containing ω ABC.ABC.A ABC.B ABC.C)
    (h₃ : H = orthocenter ABC)
    (h₄ : dist W O = dist W H) :
    angle ABC.A ABC.B ABC.C = 60 :=
sorry

end find_angle_BAC_l197_197014


namespace alternating_sum_l197_197343

theorem alternating_sum : ∑ k in Finset.range 100, (-1)^(k+1) * (k + 1) = -50 := by
  sorry

end alternating_sum_l197_197343


namespace least_changes_to_different_sums_l197_197439

theorem least_changes_to_different_sums :
  let M : Matrix (Fin 3) (Fin 3) ℕ := ![![3, 5, 0], ![6, 2, 4], ![7, 1, 8]]
  ∃ (changes : Fin 3 → Fin 3 → ℕ), 
  (∀ i, finset.univ.sum (λ j, (M i j + changes i j % M i j) % M i j) ≠ 
       finset.univ.sum (λ i, (M i j + changes i j % M i j) % M i j)) ∧ 
  finset.univ.sum (λ i, finset.univ.sum (λ j, ite ((changes i j) != 0) 1 0)) = 3 :=
by
  sorry

end least_changes_to_different_sums_l197_197439


namespace rogers_cookie_cost_l197_197832

/-- 
Given:
- Art makes 15 trapezoid cookies with bases of 3 units and 7 units, and height of 4 units.
- Roger makes 20 rectangle cookies using the same total dough as Art.
- Art's cookies sell for 75 cents each.

Prove:
- One of Roger's cookies should cost 56 cents to earn the same amount from a single batch.
-/
theorem rogers_cookie_cost :
  let art_trapezoid_area := (1 / 2: ℝ) * (3 + 7) * 4,
      art_total_dough := 15 * art_trapezoid_area,
      roger_cookie_area := art_total_dough / 20,
      art_total_earnings := 15 * 75,
      roger_cookie_cost := art_total_earnings / 20
  in roger_cookie_cost = 56 :=
by
  sorry

end rogers_cookie_cost_l197_197832


namespace student_movement_l197_197711

theorem student_movement :
  let students_total := 10
  let students_front := 3
  let students_back := 7
  let chosen_students := 2
  let ways_to_choose := Nat.choose students_back chosen_students
  let ways_to_place := (students_front + 1) * (students_front + chosen_students)
  in ways_to_choose * ways_to_place = 420 := by
  sorry

end student_movement_l197_197711


namespace sum_infinite_partial_fraction_l197_197310

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197310


namespace total_call_charges_l197_197283

-- Definitions based on conditions
def base_fee : ℝ := 39
def included_minutes : ℕ := 300
def excess_charge_per_minute : ℝ := 0.19

-- Given variables
variable (x : ℕ) -- excess minutes
variable (y : ℝ) -- total call charges

-- Theorem stating the relationship between y and x
theorem total_call_charges (h : x > 0) : y = 0.19 * x + 39 := 
by sorry

end total_call_charges_l197_197283


namespace compute_percent_errors_l197_197535

noncomputable def true_diameter : ℝ := 30
noncomputable def error_percent : ℝ := 10 / 100
noncomputable def actual_area : ℝ := real.pi * (true_diameter / 2)^2

def min_diameter := true_diameter * (1 - error_percent)
def max_diameter := true_diameter * (1 + error_percent)

noncomputable def min_area := real.pi * (min_diameter / 2)^2
noncomputable def max_area := real.pi * (max_diameter / 2)^2

def smallest_percent_error := ((actual_area - min_area) / actual_area) * 100
def largest_percent_error := ((max_area - actual_area) / actual_area) * 100

theorem compute_percent_errors :
  smallest_percent_error ≈ 19 ∧ largest_percent_error ≈ 21 :=
by
  compute_exact_spe := ((225*real.pi - 182.25*real.pi) / (225*real.pi)) * 100
  compute_exact_lpe := ((272.25*real.pi - 225*real.pi) / (225*real.pi)) * 100
  exact compute_exact_spe = 19 ∧ compute_exact_lpe = 21

end compute_percent_errors_l197_197535


namespace sum_infinite_partial_fraction_l197_197308

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197308


namespace total_first_half_points_l197_197500

-- Define the sequences for Tigers and Lions
variables (a ar b d : ℕ)
-- Defining conditions
def tied_first_quarter : Prop := a = b
def geometric_tigers : Prop := ∃ r : ℕ, ar = a * r ∧ ar^2 = a * r^2 ∧ ar^3 = a * r^3
def arithmetic_lions : Prop := b+d = b + d ∧ b+2*d = b + 2*d ∧ b+3*d = b + 3*d
def tigers_win_by_four : Prop := (a + ar + ar^2 + ar^3) = (b + (b + d) + (b + 2*d) + (b + 3*d)) + 4
def score_limit : Prop := (a + ar + ar^2 + ar^3) ≤ 120 ∧ (b + (b + d) + (b + 2*d) + (b + 3*d)) ≤ 120

-- Goal: The total number of points scored by the two teams in the first half is 23
theorem total_first_half_points : tied_first_quarter a b ∧ geometric_tigers a ar ∧ arithmetic_lions b d ∧ tigers_win_by_four a ar b d ∧ score_limit a ar b d → 
(a + ar) + (b + d) = 23 := 
by {
  sorry
}

end total_first_half_points_l197_197500


namespace sum_of_series_l197_197330

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197330


namespace sum_of_series_l197_197332

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197332


namespace possible_values_of_a_l197_197733

noncomputable def hookFunction (a x : ℝ) := x + a / x

theorem possible_values_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, 0 < x ∧ x < real.sqrt a → hookFunction a x > hookFunction a (x + 1)) 
  (h3 : ∀ x, real.sqrt a < x ∧ x < x + 1 → hookFunction a x < hookFunction a (x + 1))
  (h4 : ∀ x, 2 ≤ x ∧ x ≤ 4 → abs (hookFunction a x - hookFunction a 2) = 1 ∨ abs (hookFunction a x - hookFunction a 4) = 1) : 
  a = 4 ∨ a = 6 + 4 * real.sqrt 2 :=
by
  sorry

end possible_values_of_a_l197_197733


namespace cats_in_village_l197_197751

theorem cats_in_village (C : ℕ) (h1 : 1 / 3 * C = (1 / 4) * (1 / 3) * C)
  (h2 : (1 / 12) * C = 10) : C = 120 :=
sorry

end cats_in_village_l197_197751


namespace rectangle_area_12_l197_197488

theorem rectangle_area_12
  (L W : ℝ)
  (h1 : L + W = 7)
  (h2 : L^2 + W^2 = 25) :
  L * W = 12 :=
by
  sorry

end rectangle_area_12_l197_197488


namespace times_faster_l197_197911

theorem times_faster (A B W : ℝ) (h1 : A = 3 * B) (h2 : (A + B) * 21 = A * 28) : A = 3 * B :=
by sorry

end times_faster_l197_197911


namespace chocolate_bars_in_box_l197_197804

theorem chocolate_bars_in_box (x : ℕ) (h1 : 2 * (x - 4) = 18) : x = 13 := 
by {
  sorry
}

end chocolate_bars_in_box_l197_197804


namespace number_chosen_l197_197702

theorem number_chosen (x : ℤ) (h : x / 4 - 175 = 10) : x = 740 := by
  sorry

end number_chosen_l197_197702


namespace find_a_l197_197407

theorem find_a (a : ℝ) : 
  let A := (3, 2)
  let B := (-2, a)
  let C := (8, 12)
  -- Condition: A, B, and C are collinear
  (slope A C) = (slope A B) →
  -- Prove that a = -8
  a = -8 :=
sorry

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

end find_a_l197_197407


namespace matrix_multiplication_correct_l197_197344

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![4, -2]
]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![9, -3],
  ![2, 2]
]

def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![27, -9],
  ![32, -16]
]

theorem matrix_multiplication_correct :
  matrix1 ⬝ matrix2 = result_matrix :=
by
  sorry

end matrix_multiplication_correct_l197_197344


namespace sum_of_consecutive_integers_l197_197471

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l197_197471


namespace sufficient_not_necessary_condition_l197_197835

theorem sufficient_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x > 0 ∧ y > 0) → (x > 0 ∧ y > 0 ↔ (y/x + x/y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l197_197835


namespace infinite_series_converges_l197_197312

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197312


namespace manuscript_fee_tax_l197_197148

theorem manuscript_fee_tax (fee : ℕ) (tax_paid : ℕ) :
  (tax_paid = 0 ∧ fee ≤ 800) ∨ 
  (tax_paid = (14 * (fee - 800) / 100) ∧ 800 < fee ∧ fee ≤ 4000) ∨ 
  (tax_paid = 11 * fee / 100 ∧ fee > 4000) →
  tax_paid = 420 →
  fee = 3800 :=
by 
  intro h_eq h_tax;
  sorry

end manuscript_fee_tax_l197_197148


namespace arithmetic_sequence_eighth_term_l197_197095

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197095


namespace angle_DEF_75_l197_197503

/-- Given:
  - O is the center of the circle that circumscribes triangle DEF
  - ∠DOF = 150°
  - ∠EOD = 130°
Prove: ∠DEF = 75°
-/
theorem angle_DEF_75
  (O D E F : Type)
  [is_center_circumscribed O D E F]
  (angle_DOF : ℝ) (angle_EOD : ℝ)
  (h_dof : angle_DOF = 150) (h_eod : angle_EOD = 130) : 
  ∃ (angle_DEF : ℝ), angle_DEF = 75 := 
by
  sorry

end angle_DEF_75_l197_197503


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197664

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197664


namespace multinomial_theorem_l197_197601

theorem multinomial_theorem (x : Fin n → ℝ) (N : ℕ) :
    (∑ i, x i)^N = ∑ (n : Fin n → ℕ) in Fintype.piFinset (λ _, Finset.range (N + 1)),
      if (∑ i, n i = N) then (Nat.factorial N / ∏ i, Nat.factorial (n i)) * ∏ i, (x i)^(n i) else 0 :=
by sorry

end multinomial_theorem_l197_197601


namespace men_women_arrangement_l197_197249

theorem men_women_arrangement :
  let n := 3 in
  ∑ k in {0, 1}, nat.factorial n * nat.factorial n = 72 :=
by
  let n := 3
  exact calc
    ∑ k in {0, 1}, nat.factorial n * nat.factorial n = (nat.factorial n * nat.factorial n) + (nat.factorial n * nat.factorial n) : by sorry
    ... = 2 * (nat.factorial n * nat.factorial n) : by sorry
    ... = 2 * (6 * 6) : by sorry
    ... = 2 * 36 : by sorry
    ... = 72 : by sorry

end men_women_arrangement_l197_197249


namespace number_of_integer_pairs_mn_l197_197460

theorem number_of_integer_pairs_mn (m n : ℤ) : 
  let equation_holds := m^2 + n^2 = m * n + 3 
  in ∃ (count : ℕ), count = {p : ℤ × ℤ | equation_holds}.card := sorry

end number_of_integer_pairs_mn_l197_197460


namespace overall_profit_percentage_l197_197748

theorem overall_profit_percentage :
  let SP_A := 900
  let SP_B := 1200
  let SP_C := 1500
  let P_A := 300
  let P_B := 400
  let P_C := 500
  let CP_A := SP_A - P_A
  let CP_B := SP_B - P_B
  let CP_C := SP_C - P_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let ProfitPercentage := (TP / TCP) * 100
  ProfitPercentage = 50 := by
  sorry

end overall_profit_percentage_l197_197748


namespace value_of_a_l197_197489

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 20 - 7 * a) : a = 20 / 11 := by
  sorry

end value_of_a_l197_197489


namespace number_of_non_zero_digits_in_decimal_fraction_l197_197900

-- Definition of the problem conditions
def decimal_fraction := 120 / (2^4 * 5^9)

-- The theorem we need to prove
theorem number_of_non_zero_digits_in_decimal_fraction : 
  (how_many_non_zero_digits (to_decimal decimal_fraction)) = 3 :=
sorry

end number_of_non_zero_digits_in_decimal_fraction_l197_197900


namespace pond_length_is_4_l197_197633

noncomputable def field_length : ℝ := 16
noncomputable def field_width : ℝ := field_length / 2
noncomputable def field_area : ℝ := field_length * field_width
noncomputable def pond_area : ℝ := field_area / 8
noncomputable def pond_length : ℝ := real.sqrt pond_area

theorem pond_length_is_4 :
  pond_length = 4 :=
by
  -- Definitions of field_length, field_width, and field_area
  have h1 : field_length = 16 := rfl
  have h2 : field_width = field_length / 2 := rfl
  have h3 : field_area = field_length * field_width := rfl
  -- Definition of pond_area and pond_length
  have h4 : pond_area = field_area / 8 := rfl
  have h5 : pond_length = real.sqrt pond_area := rfl
  -- Calculation
  have h6 : field_width = 8 := by 
    rw [h1, h2] 
    exact (div_eq_iff (ne_of_gt (by norm_num))).mpr rfl
  have h7 : field_area = 128 := by 
    rw [h1, h6, h3] 
    exact (mul_eq_iff (ne_of_gt (by norm_num))).mpr rfl
  have h8 : pond_area = 16 := by 
    rw [h7, h4] 
    exact (div_eq_iff (ne_of_gt (by norm_num))).mpr rfl
  have h9 : pond_length = 4 := by 
    rw [h8, h5] 
    exact sqrt_eq_iff_sq_eq.mpr rfl
  exact h9

end pond_length_is_4_l197_197633


namespace product_of_roots_quadratic_eq_l197_197479

variable {x₁ x₂ : ℝ}

theorem product_of_roots_quadratic_eq (hx₁ : x₁ ∈ polynomial.roots (polynomial.C 2 + polynomial.C (-3) * polynomial.X + polynomial.X ^ 2))
      (hx₂ : x₂ ∈ polynomial.roots (polynomial.C 2 + polynomial.C (-3) * polynomial.X + polynomial.X ^ 2)) :
  x₁ * x₂ = 2 := 
sorry

end product_of_roots_quadratic_eq_l197_197479


namespace area_of_bounded_region_l197_197142

open Real

noncomputable def bounded_region_area : ℝ :=
  let f := λ x y : ℝ, y^2 + 4 * x * y + 80 * (abs x) = 800
  -- Assuming the graph defined by the equation forms a bounded region.
  let vertices := [(0, 20), (0, -20), (20, -20), (-20, 20)]
  let height := dist (0, 20) (0, -20)
  let base := dist (0, 20) (-20, 20)
  height * base

theorem area_of_bounded_region :
  bounded_region_area = 1600 := by
  sorry

end area_of_bounded_region_l197_197142


namespace num_spacy_subsets_15_l197_197354

def spacy_subsets (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | 3     => 4
  | n + 1 => spacy_subsets n + if n ≥ 2 then spacy_subsets (n - 2) else 1

theorem num_spacy_subsets_15 : spacy_subsets 15 = 406 := by
  sorry

end num_spacy_subsets_15_l197_197354


namespace min_ab_value_l197_197861

theorem min_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 / a + 20 / b = 4) : ab = 25 :=
sorry

end min_ab_value_l197_197861


namespace box_volume_l197_197740

theorem box_volume (l w s : ℕ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 := by
  -- Let's introduce the values derived from the conditions
  have h_length : l - 2 * s = 32 := by
    rw [hl, hs]
    norm_num
  have h_width : w - 2 * s = 20 := by
    rw [hw, hs]
    norm_num
  -- We now prove the volume
  calc
    (l - 2 * s) * (w - 2 * s) * s
      = 32 * 20 * 8 : by rw [h_length, h_width, hs]
  ... = 5120 : by norm_num

end box_volume_l197_197740


namespace probability_of_two_boys_given_one_l197_197582

-- Define the probability space for two children and the necessary conditions
def gender := {Boy, Girl}
def gender_combinations := [(Boy, Boy), (Boy, Girl), (Girl, Boy), (Girl, Girl)]

-- Event: one child is a boy
def one_child_is_a_boy (children : (gender × gender)) : Prop :=
  children.1 = Boy ∨ children.2 = Boy

-- Event: both children are boys
def both_children_are_boys (children : (gender × gender)) : Prop :=
  children.1 = Boy ∧ children.2 = Boy

-- The probability of an event A in a given finite probability space S
noncomputable def prob (S : finset (gender × gender)) (A : (gender × gender) → Prop):
  ℚ := (S.filter A).card / S.card

-- Define the conditional probability
def conditional_probability :=
  prob (finset.of_list gender_combinations)
    (fun children => both_children_are_boys children) /
  prob (finset.of_list gender_combinations)
    (fun children => one_child_is_a_boy children)

-- Prove the conditional probability result
theorem probability_of_two_boys_given_one (h : one_child_is_a_boy children) :
  conditional_probability = 1 / 2 :=
by
  sorry

end probability_of_two_boys_given_one_l197_197582


namespace sum_fraction_series_l197_197302

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197302


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197667

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197667


namespace sum_series_l197_197322

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197322


namespace units_digit_5_pow_17_mul_4_l197_197691

theorem units_digit_5_pow_17_mul_4 : ((5 ^ 17) * 4) % 10 = 0 :=
by
  sorry

end units_digit_5_pow_17_mul_4_l197_197691


namespace series_sum_l197_197342

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197342


namespace redLightDuration_mathBookThickness_l197_197361

-- Definition and axioms based on the conditions.
def isUnitOfTime (u : String) : Prop := u = "seconds" ∨ u = "minutes" ∨ u = "hours"
def isUnitOfLength (u : String) : Prop := u = "millimeters" ∨ u = "centimeters" ∨ u = "meters"

-- Statements to prove
theorem redLightDuration :
  ∃ u, isUnitOfTime u ∧ u = "seconds" := 
by { use "seconds", split, { left, refl }, { refl } }

theorem mathBookThickness :
  ∃ u, isUnitOfLength u ∧ u = "millimeters" := 
by { use "millimeters", split, { left, refl }, { refl } }

end redLightDuration_mathBookThickness_l197_197361


namespace rectangle_area_increase_l197_197134

variable {L W : ℝ} -- Define variables for length and width

theorem rectangle_area_increase (p : ℝ) (hW : W' = 0.4 * W) (hA : A' = 1.36 * (L * W)) :
  L' = L + (240 / 100) * L :=
by
  sorry

end rectangle_area_increase_l197_197134


namespace exponential_log3_range_l197_197550

theorem exponential_log3_range (x : ℝ) (h : log 3 (x - 2) < 0) : 
  0 < exp (log 3 (x - 2)) ∧ exp (log 3 (x - 2)) < 1 :=
by
  sorry

end exponential_log3_range_l197_197550


namespace polygon_vertices_product_at_least_2014_l197_197347

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l197_197347


namespace sum_infinite_series_eq_l197_197291

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197291


namespace volume_of_parallelepiped_l197_197926

theorem volume_of_parallelepiped (a b : ℝ) (h60 : Real.cos (Real.pi / 3) = 1 / 2) :
  let d1 := Real.sqrt (a^2 + b^2 + a * b)
  let d2 := Real.sqrt (a^2 + b^2 - a * b)
  let h := Real.sqrt (2 * a * b)
  let S := a * b * (Real.sin (Real.pi / 3))
  V := S * h 
  V = √6 / 2 * a^2 * b^2 := by
    sorry

end volume_of_parallelepiped_l197_197926


namespace power_of_two_plus_one_is_power_of_integer_l197_197383

theorem power_of_two_plus_one_is_power_of_integer (n : ℕ) (hn : 0 < n) (a k : ℕ) (ha : 2^n + 1 = a^k) (hk : 1 < k) : n = 3 :=
by
  sorry

end power_of_two_plus_one_is_power_of_integer_l197_197383


namespace determine_c_l197_197881

variable (a c : ℝ)

def f (x : ℝ) : ℝ := x^3 + a * x^2 + (c - a)

theorem determine_c (H : ∀ a, f a = 0 → a = −3 ∨ (1 < a ∧ a < 3/2) ∨ (3/2 < a)) :
  c = 1 :=
sorry

end determine_c_l197_197881


namespace alex_total_earnings_l197_197360

def total_earnings (hours_w1 hours_w2 wage : ℕ) : ℕ :=
  (hours_w1 + hours_w2) * wage

theorem alex_total_earnings
  (hours_w1 hours_w2 wage : ℕ)
  (h1 : hours_w1 = 28)
  (h2 : hours_w2 = hours_w1 - 10)
  (h3 : wage * 10 = 80) :
  total_earnings hours_w1 hours_w2 wage = 368 :=
by
  sorry

end alex_total_earnings_l197_197360


namespace arithmetic_sequence_eighth_term_l197_197090

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197090


namespace cricket_run_target_l197_197519

theorem cricket_run_target
  (run_rate_1st_period : ℝ)
  (overs_1st_period : ℕ)
  (run_rate_2nd_period : ℝ)
  (overs_2nd_period : ℕ)
  (target_runs : ℝ)
  (h1 : run_rate_1st_period = 3.2)
  (h2 : overs_1st_period = 10)
  (h3 : run_rate_2nd_period = 5)
  (h4 : overs_2nd_period = 50) :
  target_runs = (run_rate_1st_period * overs_1st_period) + (run_rate_2nd_period * overs_2nd_period) :=
by
  sorry

end cricket_run_target_l197_197519


namespace isosceles_triangle_tangent_circumcircle_l197_197969

variables {A B C D E : Type}
variables [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] 
variables (ABC_circ : circle A B C) 
variables (D_arc : arc A B C)
variables (D_distinct : D ∈ D_arc ∧ D ≠ B ∧ D ≠ C)
variables (E_int : E ∈ (line CD ∩ line AB)) 

theorem isosceles_triangle_tangent_circumcircle (h1 : isosceles A C B) 
  (h2 : on_circle D ABC_circ) 
  (h3 : distinct_points [D ≠ B, D ≠ C]) 
  (h4 : intersection_point E (line CD) (line AB)) :
  tangent (line BC) (circumcircle B D E) :=
sorry

end isosceles_triangle_tangent_circumcircle_l197_197969


namespace algebraic_expression_value_l197_197692

theorem algebraic_expression_value (a b c : ℝ) (x : ℝ) : 
  (a * (-5)^4 + b * (-5)^2 + c = 3) → (a * 5^4 + b * 5^2 + c = 3) := 
begin
  sorry
end

end algebraic_expression_value_l197_197692


namespace number_of_cans_needed_for_32_rooms_l197_197272

theorem number_of_cans_needed_for_32_rooms :
  ∀ (total_rooms_with_all_paint : ℕ) (lost_cans : ℕ) (rooms_with_remaining_paint : ℕ),
  total_rooms_with_all_paint = 40 →
  lost_cans = 4 →
  rooms_with_remaining_paint = 32 →
  (rooms_with_remaining_paint * lost_cans) / (total_rooms_with_all_paint - rooms_with_remaining_paint) = 16 :=
by
  intros total_rooms_with_all_paint lost_cans rooms_with_remaining_paint
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end number_of_cans_needed_for_32_rooms_l197_197272


namespace coloring_one_third_of_square_l197_197285

theorem coloring_one_third_of_square :
  nat.choose 18 6 = 18564 := by
  sorry

end coloring_one_third_of_square_l197_197285


namespace prank_combinations_l197_197279

theorem prank_combinations : 
  let monday := 1
  let tuesday := 2
  let wednesday := 6
  let thursday := 3
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday = 36) := 
by
  let monday := 1
  let tuesday := 2
  let wednesday := 6
  let thursday := 3
  let friday := 1
  have h_comb := monday * tuesday * wednesday * thursday * friday
  have : h_comb = 36 := sorry
  exact this

end prank_combinations_l197_197279


namespace solve_system_l197_197608

theorem solve_system : 
  { (x, y) : ℝ × ℝ | (x^3 + y = 2 * x) ∧ (y^3 + x = 2 * y) } = 
  { (0, 0), (1, 1), (-1, -1), (sqrt 3, -sqrt 3), (-sqrt 3, sqrt 3) } :=
by
  sorry

end solve_system_l197_197608


namespace quadratic_inequality_solution_set_l197_197643

theorem quadratic_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Ioo (-4 : ℝ) 0 :=
by
  sorry

end quadratic_inequality_solution_set_l197_197643


namespace min_diff_l197_197396

noncomputable def geometric_sequence_sum (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 + 1 / (2 ^ n) else 1 - 1 / (2 ^ n)

noncomputable def bounds (n : ℕ) : ℚ :=
  geometric_sequence_sum n - 1 / (geometric_sequence_sum n)

theorem min_diff (s t : ℚ) :
  (∀ n : ℕ, n > 0 → bounds n ∈ set.Icc s t) → t - s = 17 / 12 :=
by
  sorry

end min_diff_l197_197396


namespace proof_problem_l197_197240

def eq1 (x : ℝ) : Prop :=
  log 4 (4 ^ (real.sqrt 2 * real.sin x) + 4 ^ (real.sqrt 2 * real.cos x)) +
  log ((real.tan x)^4 + 1)^2 (real.sqrt 2) =
  log 16 (real.cot x^4 / (real.cot x^4 + 1))

theorem proof_problem (x : ℝ) (k : ℤ) :
  eq1 x → (x = (5 * real.pi / 4) + 2 * real.pi * k) :=
begin
  sorry
end

end proof_problem_l197_197240


namespace arithmetic_sequence_8th_term_l197_197110

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197110


namespace new_average_weight_l197_197617

theorem new_average_weight (avg_weight: ℕ) (num_students: ℕ) (new_student_weight: ℕ) :
  (avg_weight = 28) ∧ (num_students = 29) ∧ (new_student_weight = 4) → 
  (let total_weight := num_students * avg_weight in
   let new_total_weight := total_weight + new_student_weight in
   let new_total_students := num_students + 1 in
   let new_avg_weight := new_total_weight / new_total_students in
   new_avg_weight = 27.2) :=
begin
  sorry
end

end new_average_weight_l197_197617


namespace graph_shift_cos_sin_l197_197654

theorem graph_shift_cos_sin :
  function.graph_shift (λ x, cos (3 * x - π / 3))
    (λ x, sin (3 * x)) = shift_left π / 18 :=
by
  sorry

end graph_shift_cos_sin_l197_197654


namespace find_a3_l197_197351

-- Definitions from conditions
def arithmetic_sum (a1 a3 : ℕ) := (3 / 2) * (a1 + a3)
def common_difference := 2
def S3 := 12

-- Theorem to prove that a3 = 6
theorem find_a3 (a1 a3 : ℕ) (h₁ : arithmetic_sum a1 a3 = S3) (h₂ : a3 = a1 + common_difference * 2) : a3 = 6 :=
by
  sorry

end find_a3_l197_197351


namespace tan_22_5_decomposition_l197_197160

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  tan (22.5 * Real.pi / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d ∧
  a + b + c + d = 3 :=
by
  sorry

end tan_22_5_decomposition_l197_197160


namespace total_cost_is_correct_percentage_of_each_is_correct_average_price_is_correct_l197_197568

def roses_count := 50
def lilies_count := 40
def sunflowers_count := 30
def daisies_count := 20
def orchids_count := 10
def tulips_count := 15

def rose_price := 2.0
def lily_price := 1.5
def sunflower_price := 1.0
def daisy_price := 0.75
def orchid_price := 3.0
def tulip_price := 2.5

def total_cost := roses_count * rose_price 
                  + lilies_count * lily_price 
                  + sunflowers_count * sunflower_price 
                  + daisies_count * daisy_price 
                  + orchids_count * orchid_price 
                  + tulips_count * tulip_price

def total_flowers := roses_count + lilies_count 
                     + sunflowers_count + daisies_count 
                     + orchids_count + tulips_count

def rose_percentage := (roses_count / total_flowers.toFloat) * 100
def lily_percentage := (lilies_count / total_flowers.toFloat) * 100
def sunflower_percentage := (sunflowers_count / total_flowers.toFloat) * 100
def daisy_percentage := (daisies_count / total_flowers.toFloat) * 100
def orchid_percentage := (orchids_count / total_flowers.toFloat) * 100
def tulip_percentage := (tulips_count / total_flowers.toFloat) * 100

def average_price := total_cost / total_flowers.toFloat

theorem total_cost_is_correct : total_cost = 272.50 := 
by 
  sorry

theorem percentage_of_each_is_correct : 
  rose_percentage = 30.30 
  ∧ lily_percentage = 24.24 
  ∧ sunflower_percentage = 18.18 
  ∧ daisy_percentage = 12.12 
  ∧ orchid_percentage = 6.06 
  ∧ tulip_percentage = 9.09 := 
by 
  sorry

theorem average_price_is_correct : average_price = 1.65 := 
by 
  sorry

end total_cost_is_correct_percentage_of_each_is_correct_average_price_is_correct_l197_197568


namespace sum_of_x_for_sqrt_eq_nine_l197_197216

theorem sum_of_x_for_sqrt_eq_nine :
  (∑ x in Finset.filter (λ x, (abs (x - 2) = 9)) (Finset.range 100), x) = 4 :=
sorry

end sum_of_x_for_sqrt_eq_nine_l197_197216


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197665

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197665


namespace diameter_of_intersecting_circles_l197_197281

noncomputable def is_diameter (Γ : Circle) (X Y : Point) : Prop :=
  distance X Y = 2 * radius Γ

noncomputable def tangent_perpendicular (Γ1 Γ2 : Circle) (A B : Point) : Prop :=
  tangent Γ1 A ⊥ tangent Γ2 A ∧ tangent Γ1 B ⊥ tangent Γ2 B

variables (Γ1 Γ2 : Circle) (A B M X Y : Point)

theorem diameter_of_intersecting_circles
  (h1 : Intersects Γ1 Γ2 A B)
  (h2 : tangent_perpendicular Γ1 Γ2 A B)
  (h3 : OnCircle Γ1 M)
  (h4 : InsideCircle Γ2 M)
  (h5 : ExtendsToCircle Γ1 M A X)
  (h6 : ExtendsToCircle Γ1 M B Y) :
  is_diameter Γ2 X Y :=
sorry

end diameter_of_intersecting_circles_l197_197281


namespace triangle_weight_l197_197509

variables (S C T : ℕ)

def scale1 := (S + C = 8)
def scale2 := (S + 2 * C = 11)
def scale3 := (C + 2 * T = 15)

theorem triangle_weight (h1 : scale1 S C) (h2 : scale2 S C) (h3 : scale3 C T) : T = 6 :=
by 
  sorry

end triangle_weight_l197_197509


namespace sum_of_x_for_sqrt_eq_nine_l197_197217

theorem sum_of_x_for_sqrt_eq_nine :
  (∑ x in Finset.filter (λ x, (abs (x - 2) = 9)) (Finset.range 100), x) = 4 :=
sorry

end sum_of_x_for_sqrt_eq_nine_l197_197217


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197670

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197670


namespace area_of_ABIJKFGD_eq_62_5_l197_197508

-- Definitions for the basic squares and setup
def ABCD := Square (pointA pointB pointC pointD : EuclideanSpace ℝ ℝ) :=
  side_length : ℝ, sideLength = 5, side_square := 25 -- area 25 square with side length 5
def EFGD := Square (pointE pointF pointG pointD : EuclideanSpace ℝ ℝ) :=
  side_length : ℝ, sideLength = 5, side_square := 25 -- area 25 square with side length 5

-- Condition for midpoint and intersection of squares
def IsMidpoint (P Q R : EuclideanSpace ℝ ℝ) := 
  dist P R = dist Q R / 2 

def EachLines (Square1 Square2 Square3 : EuclideanSpace ℝ ℝ) :=
  L_exists : Line (L: EuclideanSpace ℝ ℝ),
     L = Midpoint P Q ∧ L liesIn (Line pointE pointF) -- midpoint conditions and point L on EF
  
-- Definition and proof of the total area of polygon
theorem area_of_ABIJKFGD_eq_62_5 :
  ∀ (pointA pointB pointC pointD pointE pointF pointG pointL: EuclideanSpace ℝ ℝ), 
  side_square ABCD = 25 ∧ side_square EFGD = 25 ∧
  side_length sqIJKL = 5 ∧ 
  IsMidpoint pointH pointBC pointEF ∧ IsMidpoint pointD pointJK ∧
  LExists_line_midpoint_L pointE pointF pointL → 
  total_area_of_polygon_ABIJKFGD = 62.5 := 
begin
  -- proof
  sorry, -- proof omitted
end

end area_of_ABIJKFGD_eq_62_5_l197_197508


namespace diagonals_in_nonagon_l197_197258

-- Define the properties of the polygon
def convex : Prop := true
def sides (n : ℕ) : Prop := n = 9
def right_angles (count : ℕ) : Prop := count = 2

-- Define the formula for the number of diagonals in a polygon with 'n' sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem definition
theorem diagonals_in_nonagon :
  convex →
  (sides 9) →
  (right_angles 2) →
  number_of_diagonals 9 = 27 :=
by
  sorry

end diagonals_in_nonagon_l197_197258


namespace isosceles_right_triangle_fold_l197_197741

theorem isosceles_right_triangle_fold (BC : ℝ) (AB AC BP : ℝ) (AP_square : ℝ) :
  ∠BAC = 90 ∧ AB = 10 ∧ AC = 10 ∧ BP = 6 ∧ BC = sqrt (100 + 100) 
    ∧ AP_square = AB^2 - BP^2 → AP_square = 64 :=
by
  intro h,
  sorry

end isosceles_right_triangle_fold_l197_197741


namespace range_of_a_l197_197485

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f a x ≤ f a y) ↔ (- (1/4:ℝ) ≤ a ∧ a ≤ 0) := by
  sorry

end range_of_a_l197_197485


namespace minimum_area_triangle_segment_bisected_l197_197186

theorem minimum_area_triangle_segment_bisected 
  (A L N : Point) (h : is_interior A (angle L N)) :
  (∃ line, ∀ line_a, area (triangle A (line_intersect line_a L) (line_intersect line_a N))
   ≤ area (triangle A (line_intersect line L) (line_intersect line N)) ∧ 
   bisects (segment (line_intersect line L) (line_intersect line N)) A) → 
  bisects (segment (line_intersect line L) (line_intersect line N)) A :=
begin
  sorry
end

end minimum_area_triangle_segment_bisected_l197_197186


namespace second_chick_eats_52_l197_197983

theorem second_chick_eats_52 (days : ℕ) (first_chick_eats : ℕ → ℕ) (second_chick_eats : ℕ → ℕ) :
  (∀ n, first_chick_eats n + second_chick_eats n = 12) →
  (∃ a b, first_chick_eats a = 7 ∧ second_chick_eats a = 5 ∧
          first_chick_eats b = 7 ∧ second_chick_eats b = 5 ∧
          12 * days = first_chick_eats a * 2 + first_chick_eats b * 6 + second_chick_eats a * 2 + second_chick_eats b * 6) →
  (first_chick_eats a * 2 + first_chick_eats b * 6 = 44) →
  (second_chick_eats a * 2 + second_chick_eats b * 6 = 52) :=
by
  sorry

end second_chick_eats_52_l197_197983


namespace length_of_one_side_is_approximately_18_17_l197_197694

noncomputable def length_of_one_side_of_box : ℝ :=
  let cost_per_box := 1.2
  let total_volume := 3.06 * 10^6
  let min_amount_spent := 612
  let number_of_boxes := min_amount_spent / cost_per_box
  let volume_per_box := total_volume / number_of_boxes
  real.cbrt volume_per_box

theorem length_of_one_side_is_approximately_18_17 : abs (length_of_one_side_of_box - 18.17) < 0.01 := by
  sorry

end length_of_one_side_is_approximately_18_17_l197_197694


namespace hyperbola_focus_distance_asymptote_l197_197436

theorem hyperbola_focus_distance_asymptote :
  ∀ (b : ℝ), (b > 0) → 
  ((∃ (x y : ℝ), y^2 = 12 * x ∧ x = 3 ∧ y = 0) ∧ 
  (∃ (hx hy : ℝ), (hx, hy) = (3, 0) ∧ (hx - 2 * hy / sqrt 5) = 0)) →
  (b^2 = 5 → 
  (∃ f : ℝ, f = sqrt 5 ∧ (sqrt 5 * 3 - 2 * 0) / sqrt (5 * 3^2 + 2 * 0^2) = sqrt 5)) :=
by
  intros b h b_pos
  intro h
  sorry

end hyperbola_focus_distance_asymptote_l197_197436


namespace largest_c_3_in_range_l197_197367

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end largest_c_3_in_range_l197_197367


namespace determine_a_l197_197866

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x / Real.log a - 3

theorem determine_a (a : ℝ): (∀ x : ℝ, f x a > 2 ↔ x > a^2 - 3) → (1, +∞) = set_of (λ x, f x a > 2) → a = 2 :=
by
  intros h_conditions h_interval
  sorry

end determine_a_l197_197866


namespace fruit_eating_problem_l197_197721

theorem fruit_eating_problem (a₀ p₀ o₀ : ℕ) (h₀ : a₀ = 5) (h₁ : p₀ = 8) (h₂ : o₀ = 11) :
  ¬ ∃ (d : ℕ), (a₀ - d) = (p₀ - d) ∧ (p₀ - d) = (o₀ - d) ∧ ∀ k, k ≤ d → ((a₀ - k) + (p₀ - k) + (o₀ - k) = 24 - 2 * k ∧ a₀ - k ≥ 0 ∧ p₀ - k ≥ 0 ∧ o₀ - k ≥ 0) :=
by
  sorry

end fruit_eating_problem_l197_197721


namespace player_A_always_wins_l197_197580

theorem player_A_always_wins (a b c : ℤ) :
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x - x1) * (x - x2) * (x - x3) = x^3 + a*x^2 + b*x + c :=
sorry

end player_A_always_wins_l197_197580


namespace part1_part2_part3_l197_197836

noncomputable def polynomial_expansion := (1 - 2 * x) ^ 7

theorem part1 (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) :
  polynomial_expansion = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 -> 
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -1 :=
by
  sorry

theorem part2 (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) :
  polynomial_expansion = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 ->
  a = 1 ->
  a + a_2 + a_4 + a_6 = 1094 ∧ a_1 + a_3 + a_5 + a_7 = 1093 :=
by
  sorry

theorem part3 : 2^7 = 128 :=
by
  sorry

end part1_part2_part3_l197_197836


namespace unique_solution_l197_197823

theorem unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1 :=
by
  sorry

end unique_solution_l197_197823


namespace basketball_team_points_l197_197497

theorem basketball_team_points :
  ∃ q : ℕ, 
    q % 2 = 1 ∧                 -- q is an odd number
    q ≥ 7 ∧ 
    (∑ i in finset.range 11, 7) + q = 100 ∧
    7 ≤ q ∧ 
    q = 23 := 
sorry

end basketball_team_points_l197_197497


namespace arithmetic_sequence_8th_term_l197_197120

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197120


namespace triangle_side_c_sin_2A_minus_pi_over_4_l197_197920

theorem triangle_side_c (a b : ℝ) (A C : ℝ) (h₁ : a = sqrt 5) (h₂ : b = 3) (h₃ : sin C = 2 * sin A) : 
  ∃ c : ℝ, c = 2 * sqrt 5 := 
by
  use 2 * sqrt 5
  sorry

theorem sin_2A_minus_pi_over_4 (a b c A C : ℝ) (h₁ : a = sqrt 5) (h₂ : b = 3) (h₃ : c = 2 * sqrt 5) (h₄ : sin C = 2 * sin A) : 
  sin (2 * A - π / 4) = sqrt 2 / 10 := 
by
  sorry

end triangle_side_c_sin_2A_minus_pi_over_4_l197_197920


namespace number_of_integers_with_1_left_of_3_l197_197620

theorem number_of_integers_with_1_left_of_3 : 
  (card { l : List ℕ | l = [1, 2, 3, 4, 5, 6].perm l ∧ 1 ∈ l ∧ 3 ∈ l ∧ (index_of 1 l < index_of 3 l) }) = 360 := 
sorry

end number_of_integers_with_1_left_of_3_l197_197620


namespace music_logarithms_proof_l197_197980

theorem music_logarithms_proof :
  let α := 9 / 8
  let β := 256 / 243
  log (α^5) + log (β^2) - log 2 = 0 :=
by
  sorry

end music_logarithms_proof_l197_197980


namespace min_distance_ellipse_to_line_l197_197860

open Real

noncomputable def ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

noncomputable def line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x + y = 2 * sqrt 5

theorem min_distance_ellipse_to_line :
  let d_min := (1 / 2) * sqrt 10 in
  ∀ (P : ℝ × ℝ), ellipse P → ∃ d, 
  (∀ Q : ℝ × ℝ, line Q → ∂(P, Q) = d) ∧ d = d_min :=
sorry

end min_distance_ellipse_to_line_l197_197860


namespace find_a_l197_197429

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1)
  (h_min : ∀ x : ℝ, (-1 < x ∧ x < 3) → log a (3 - x) + log a (x + 1) ≥ -2)
  : a = 1/2 :=
  sorry

end find_a_l197_197429


namespace ab_value_l197_197856

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f x

noncomputable def function_definition (f : ℝ → ℝ) : Prop := 
  ∀ x, (x ≥ 0 → f x = 2 * x - x ^ 2)

noncomputable def range_condition (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y, y ∈ set.range (λ x, f x) → (a ≤ x ∧ x ≤ b) → (1 / b) ≤ y ∧ y ≤ 1 / a

theorem ab_value (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : is_odd_function f)
  (h2 : function_definition f)
  (h3 : range_condition f a b) :
  a * b = (1 + sqrt 5) / 2 := sorry

end ab_value_l197_197856


namespace calculate_expression_l197_197775
open Complex

-- Define the given values for a and b
def a := 3 + 2 * Complex.I
def b := 2 - 3 * Complex.I

-- Define the target expression
def target := 3 * a + 4 * b

-- The statement asserts that the target expression equals the expected result
theorem calculate_expression : target = 17 - 6 * Complex.I := by
  sorry

end calculate_expression_l197_197775


namespace arithmetic_sequence_8th_term_l197_197084

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197084


namespace shortest_distance_parabola_l197_197824

theorem shortest_distance_parabola (x y : ℝ) (h1 : x = 5) (h2 : y = 10) :
  ∃ P : ℝ × ℝ, P.1 = 3 ∧ P.2 = 3 ∧ (dist (x, y) P = sqrt 53) :=
by
  sorry

end shortest_distance_parabola_l197_197824


namespace character_of_c_l197_197812

noncomputable def exists_function_satisfying_condition (c : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → f (x + y^2) ≥ c * f x + y)

theorem character_of_c :
  ∀ c, (∃ f : ℝ → ℝ, (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → f (x + y^2) ≥ c * f x + y)) ↔ c < 1 :=
begin
  intro c,
  split,
  { intro h,
    sorry, -- Forward direction proof placeholder
  },
  { intro h,
    sorry, -- Backward direction proof placeholder
  }
end

end character_of_c_l197_197812


namespace limit_tg_cos_log_l197_197242

theorem limit_tg_cos_log:
  ∀ x : ℝ, 
  (∀ε > 0, ∃ δ > 0, ∀ x, 0 < |x| < δ → 
    |((tan x * cos (1/x) + real.log (2 + x)) / real.log (4 + x)) - 1/2| < ε) := 
begin
  sorry
end

end limit_tg_cos_log_l197_197242


namespace conditions_for_a_and_b_l197_197717

variables (a b x y : ℝ)

theorem conditions_for_a_and_b (h1 : x^2 + x * y + y^2 - y = 0) (h2 : a * x^2 + b * x * y + x = 0) :
  (a + 1)^2 = 4 * (b + 1) ∧ b ≠ -1 :=
sorry

end conditions_for_a_and_b_l197_197717


namespace ratio_sheep_to_horses_is_correct_l197_197163

-- Definitions of given conditions
def ounces_per_horse := 230
def total_ounces_per_day := 12880
def number_of_sheep := 16

-- Express the number of horses and the ratio of sheep to horses
def number_of_horses : ℕ := total_ounces_per_day / ounces_per_horse
def ratio_sheep_to_horses := number_of_sheep / number_of_horses

-- The main statement to be proved
theorem ratio_sheep_to_horses_is_correct : ratio_sheep_to_horses = 2 / 7 :=
by
  sorry

end ratio_sheep_to_horses_is_correct_l197_197163


namespace breadth_of_landscape_l197_197706

noncomputable def landscape_breadth (L : ℕ) (playground_area : ℕ) (total_area : ℕ) (B : ℕ) : Prop :=
  B = 6 * L ∧ playground_area = 4200 ∧ playground_area = (1 / 7) * total_area ∧ total_area = L * B

theorem breadth_of_landscape : ∃ (B : ℕ), ∀ (L : ℕ), landscape_breadth L 4200 29400 B → B = 420 :=
by
  intros
  sorry

end breadth_of_landscape_l197_197706


namespace quadratic_has_real_roots_range_l197_197916

noncomputable def has_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := 2
  let c := -1
  b^2 - 4 * a * c ≥ 0

theorem quadratic_has_real_roots_range (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 ∧ k ≠ 0 := by
sorry

end quadratic_has_real_roots_range_l197_197916


namespace simplify_expression_l197_197058

theorem simplify_expression : (- (1 : ℝ) / 16) ^ (-3 / 4) = -8 := 
by
  sorry

end simplify_expression_l197_197058


namespace three_of_clubs_initial_edge_position_l197_197738

/-- There is a deck of 52 cards placed in a row, and 51 cards will be discarded, leaving only the
three of clubs. At each step, the spectator indicates the position of the card to be discarded from
the edge, and the magician decides whether to count from the left or the right edge and discards
the corresponding card. Prove that the initial position of the three of clubs must be on the edge
to guarantee the magician's success in the trick. -/
theorem three_of_clubs_initial_edge_position
    (deck : list ℕ) (h_deck_length : deck.length = 52)
    (three_of_clubs : ℕ) (h_three_of_clubs : three_of_clubs = 3)
    (last_card : ℕ) (h_last_card : last_card = 3)
    (discard_strategy : ℕ → option ℕ) :
    (three_of_clubs = deck.head ∨ three_of_clubs = deck.last) :=
by
  sorry

end three_of_clubs_initial_edge_position_l197_197738


namespace train_length_490_l197_197753

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_490 :
  train_length 63 28 = 490 := by
  -- Proof goes here
  sorry

end train_length_490_l197_197753


namespace cyclic_quadrilateral_l197_197017

variables {α : Type*} [MetricSpace α]

-- Points on the circle
variables (A B C D S E F : α)
variable  is_circle : Convex ℝ (Set.insert A (Set.insert B (Set.insert C (Set.insert D ∅))))
variable  midpoint_arc_AB : IsArcMidpoint S A B (Not (C ∈ Interval S A B)) (Not (D ∈ Interval S A B))
variable  E_on_ab : IsIntersectionPoint E S D A B
variable  F_on_ab : IsIntersectionPoint F S C A B

theorem cyclic_quadrilateral : Concyclic C D E F :=
by
  sorry

end cyclic_quadrilateral_l197_197017


namespace arithmetic_sequence_8th_term_is_71_l197_197104

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197104


namespace sin_neg_135_degree_l197_197781

theorem sin_neg_135_degree : sin (-(135 * Real.pi / 180)) = - (Real.sqrt 2 / 2) :=
by 
  -- Here, we need to use the known properties and the equivalences given
  sorry

end sin_neg_135_degree_l197_197781


namespace vasya_improved_example1_vasya_improved_example2_l197_197683

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l197_197683


namespace problem_statement_l197_197536

open EuclideanGeometry

noncomputable def given_triangle (A B C Y Z U V L : Point) : Prop :=
  isAltitude Y B C ∧ isAltitude Z C B ∧
  isPerpendicularFoot U Y B C ∧ isPerpendicularFoot V Z B C ∧
  areConcurrent [line_through Y V, line_through Z U] L

theorem problem_statement (A B C Y Z U V L : Point) (h : given_triangle A B C Y Z U V L) :
  perpendicular (line_through A L) (line_through B C) :=
sorry

end problem_statement_l197_197536


namespace arithmetic_sequence_8th_term_is_71_l197_197100

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197100


namespace probability_of_factors_less_than_seven_is_half_l197_197208

open nat

def positive_factors (n : ℕ) : list ℕ :=
  list.filter (λ d, d > 0 ∧ d ∣ n) (list.range (n + 1))

def probability_less_than_seven (n : ℕ) : ℚ :=
  let factors := positive_factors n in
  let favorable := list.filter (λ d, d < 7) factors in
  (favorable.length : ℚ) / (factors.length : ℚ)

theorem probability_of_factors_less_than_seven_is_half :
  probability_less_than_seven 60 = 1 / 2 :=
by sorry

end probability_of_factors_less_than_seven_is_half_l197_197208


namespace f_g_of_3_l197_197903

def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := x^2 + 2 * x + 1

theorem f_g_of_3 : f (g 3) = 61 :=
by
  sorry

end f_g_of_3_l197_197903


namespace vasya_improved_example1_vasya_improved_example2_l197_197685

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l197_197685


namespace exists_infinite_sets_S_T_l197_197803

theorem exists_infinite_sets_S_T :
  ∃ (S T : set ℕ), (S ≠ ∅) ∧ (T ≠ ∅) ∧ (∀ n : ℕ, n > 0 → ∃ (k : ℕ) (s : fin k → ℕ) (t : fin k → ℕ),
    (∀ i j, i < j → s i < s j) ∧ (∀ i, s i ∈ S) ∧ (∀ i, t i ∈ T) ∧
    n = (finset.univ.sum (λ i, s i * t i))) :=
sorry

end exists_infinite_sets_S_T_l197_197803


namespace parade_ground_problem_l197_197982

theorem parade_ground_problem :
  ∃ (ways : ℕ), ways = 30 ∧ ways = (nat.factorial 8 / (nat.factorial 7 * nat.factorial 1)) :=
  sorry

end parade_ground_problem_l197_197982


namespace appropriate_sampling_method_l197_197650

/--
Given there are 40 products in total, consisting of 10 first-class products,
25 second-class products, and 5 defective products, if we need to select
8 products for quality analysis, then the appropriate sampling method is
the stratified sampling method.
-/
theorem appropriate_sampling_method
  (total_products : ℕ)
  (first_class_products : ℕ)
  (second_class_products : ℕ)
  (defective_products : ℕ)
  (selected_products : ℕ)
  (stratified_sampling : ℕ → ℕ → ℕ → ℕ → Prop) :
  total_products = 40 →
  first_class_products = 10 →
  second_class_products = 25 →
  defective_products = 5 →
  selected_products = 8 →
  stratified_sampling total_products first_class_products second_class_products defective_products →
  stratified_sampling total_products first_class_products second_class_products defective_products :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end appropriate_sampling_method_l197_197650


namespace four_divides_n_if_subsets_exist_l197_197795

theorem four_divides_n_if_subsets_exist :
  ∀ n : ℕ, (∃ (A : Finset (Finset (Fin n))) (h : A.card = n),
  (∀ (i j : Fin n), i ≠ j → ∃ (Ai Aj : Finset n), Ai ∈ A ∧ Aj ∈ A ∧ i ≠ j → (Ai ∩ Aj).card ≠ 1)) → 4 | n :=
by
  intros n h
  sorry

end four_divides_n_if_subsets_exist_l197_197795


namespace infinite_series_converges_l197_197318

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197318


namespace log_simplification_l197_197995

open Real

theorem log_simplification (a b d e z y : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (ha : a ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) :
  log (a / b) + log (b / e) + log (e / d) - log (az / dy) = log (dy / z) :=
by
  sorry

end log_simplification_l197_197995


namespace limit_tg_cos_log_l197_197241

theorem limit_tg_cos_log:
  ∀ x : ℝ, 
  (∀ε > 0, ∃ δ > 0, ∀ x, 0 < |x| < δ → 
    |((tan x * cos (1/x) + real.log (2 + x)) / real.log (4 + x)) - 1/2| < ε) := 
begin
  sorry
end

end limit_tg_cos_log_l197_197241


namespace general_form_of_quadratic_equation_l197_197157

noncomputable def quadratic_equation_general_form (x : ℝ) : Prop :=
  (x + 3) * (x - 1) = 2 * x - 4

theorem general_form_of_quadratic_equation (x : ℝ) :
  quadratic_equation_general_form x → x^2 + 1 = 0 :=
sorry

end general_form_of_quadratic_equation_l197_197157


namespace average_of_middle_two_numbers_l197_197615

theorem average_of_middle_two_numbers 
  (a b c d : ℕ) 
  (ha : a = 3) 
  (h_diff : 3 < b ∧ b < c ∧ c < d ∧ (d - a) = max (d - a) for all permutations)
  (h_avg : (a + b + c + d) / 4 = 6) : 
  ((b + c) / 2 : ℝ) = 4.5 :=
sorry

end average_of_middle_two_numbers_l197_197615


namespace mutually_exclusive_pairs_l197_197872

def event (outcome : Type) := outcome → Prop

variables (Out : Type) 
variables (E1 E2 : event Out → Prop)
variables (A1 A2 : event Out)
variables (A_hits_7th_ring A_hits_8th_ring : event Out)
variables (B_hits_8th_ring : event Out)
variables (both_hit_target neither_hit_target : event Out)
variables (at_least_one_hits A_hits_not_B : event Out)

-- Conditions:
def pair1_is_mutually_exclusive : Prop := ∀ o, ¬(A_hits_7th_ring o ∧ A_hits_8th_ring o)
def pair3_is_mutually_exclusive : Prop := ∀ o, ¬(both_hit_target o ∧ neither_hit_target o)

-- Question: Proving the number of mutually exclusive pairs equals 2.
theorem mutually_exclusive_pairs : pair1_is_mutually_exclusive A_hits_7th_ring A_hits_8th_ring ∧ pair3_is_mutually_exclusive both_hit_target neither_hit_target := 
by {
  -- Proof omitted
  sorry
}

end mutually_exclusive_pairs_l197_197872


namespace solve_for_x_l197_197492

theorem solve_for_x {x : ℤ} (h : 3 * x + 7 = -2) : x = -3 := 
by
  sorry

end solve_for_x_l197_197492


namespace length_AK_angle_CAB_l197_197284

-- Definitions and conditions
variable (O1 O2 O A B C K1 K2 K3 K : Point)
variable (r R : ℝ)
variable (triangle_ABC : Triangle A B C)
variable (circle_inscribed_A : Circle O1 r)
variable (circle_inscribed_B : Circle O2 r)
variable (circle_inscribed_ABC : Circle O R)
variable (tangent_AK1 : isTangentAt circle_inscribed_A A B K1)
variable (tangent_BK2 : isTangentAt circle_inscribed_B B A K2)
variable (tangent_K : isTangentAt circle_inscribed_ABC A B K)
variable (AK1_len : AK1 = 4)
variable (BK2_len : BK2 = 6)
variable (AB_len : AB = 16)

-- Part (a): Prove the length of segment AK
theorem length_AK : AK = 32 / 5 := 
sorry

-- Additional definition for part (b)
variable (tangent_AK3 : isTangentAt circle_inscribed_A A C K3)
variable (circumcenter_OK1K3 : isCircumcenter O1 O K1 K3)

-- Part (b): Prove the angle CAB
theorem angle_CAB : ∠CAB = 2 * arcsin (3 / 5) :=
sorry

end length_AK_angle_CAB_l197_197284


namespace area_of_sector_approximation_l197_197705

/--
Theorem:
Given a circle with a radius of 12 meters and a central angle of 40 degrees,
the area of the sector is approximately 50.26544 square meters.
-/
theorem area_of_sector_approximation (r : ℝ) (θ : ℝ) (π : ℝ) (A : ℝ) 
  (r_eq : r = 12) (θ_eq : θ = 40) (π_approx: π = 3.14159) : 
  A ≈ 50.26544 :=
by
  sorry

end area_of_sector_approximation_l197_197705


namespace arithmetic_sequence_8th_term_l197_197116

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197116


namespace price_of_8_inch_portrait_l197_197598

/-- Define the conditions given in the problem -/
section
variables {P : ℝ} -- P is the price of an 8-inch portrait
variables (daily_earnings_3_days : ℝ) -- Earnings every 3 days
variables (daily_earnings : ℝ) -- Daily earnings

/-- Given: 
1. Price of an 8-inch portrait is P.
2. Price of a 16-inch portrait is 2P.
3. Sansa sells three 8-inch portraits and five 16-inch portraits per day.
4. Earnings every 3 days is $195.
We need to prove that the price of an 8-inch portrait is $5 
-/
theorem price_of_8_inch_portrait 
  (h1 : daily_earnings_3_days = 195) 
  (h2 : daily_earnings = daily_earnings_3_days / 3)
  (h3 : daily_earnings = 3 * P + 5 * (2 * P)) :
  P = 5 :=
by
  sorry
end

end price_of_8_inch_portrait_l197_197598


namespace distance_to_nearest_edge_of_picture_l197_197742

def wall_width : ℕ := 26
def picture_width : ℕ := 4
def distance_from_end (wall picture : ℕ) : ℕ := (wall - picture) / 2

theorem distance_to_nearest_edge_of_picture :
  distance_from_end wall_width picture_width = 11 :=
sorry

end distance_to_nearest_edge_of_picture_l197_197742


namespace largest_integer_solution_l197_197629

theorem largest_integer_solution (x : ℤ) (h : 3 - 2 * x > 0) : x ≤ 1 :=
by sorry

end largest_integer_solution_l197_197629


namespace f2_plus_fprime2_l197_197143

noncomputable def f : ℝ → ℝ := sorry
def tangent_equation := ∀ x y, 2 * x + y - 3 = 0

lemma f_at_2 (h : ∀ x y, (2 * x + y - 3 = 0) → f 2 = y) : f 2 = -1 :=
by {
  specialize h 2 (-1),
  simp at h,
  exact h,
}

lemma f_prime_at_2 (h : ∀ x y, (2 * x + y - 3 = 0) → deriv f 2 = 2) : deriv f 2 = -2 :=
by {
  specialize h 2 (-1),
  simp at h,
  exact h,
}

theorem f2_plus_fprime2 :
  (∃ f : ℝ → ℝ, tangent_equation) →
  (f 2 + deriv f 2 = -3) :=
by {
  intros h,
  have h1 : f 2 = -1 := f_at_2 h,
  have h2 : deriv f 2 = -2 := f_prime_at_2 h,
  simp [h1, h2],
}

end f2_plus_fprime2_l197_197143


namespace perpendicular_parallel_situations_l197_197554

def conditions_2 (X Y Z : Type) [line X] [line Y] [plane Z] : Prop :=
  (X ⊥ Z) ∧ (Y ⊥ Z) → X ∥ Y

def conditions_3 (X Y Z : Type) [plane X] [plane Y] [line Z] : Prop :=
  (X ⊥ Z) ∧ (Y ⊥ Z) → X ∥ Y

theorem perpendicular_parallel_situations (X Y Z : Type)
  (h1 : (X ⊥ Z) ∧ (Y ⊥ Z) → X ∥ Y)
  (h2 : (X ⊥ Z) ∧ (Y ⊥ Z) → X ∥ Y) :
  (conditions_2 X Y Z ∨ conditions_3 X Y Z) :=
by
  sorry

end perpendicular_parallel_situations_l197_197554


namespace arithmetic_expression_value_l197_197570

theorem arithmetic_expression_value :
  68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 :=
by
  sorry

end arithmetic_expression_value_l197_197570


namespace sum_binomial_coeff_l197_197713

open Nat

theorem sum_binomial_coeff (n : ℕ) : 
  (∑ k in Finset.range n.succ, if k = 0 then 0 else 2^k * Nat.choose n k) = 3^n - 1 := by
  sorry

end sum_binomial_coeff_l197_197713


namespace max_a2b3c4_l197_197962

noncomputable def maximum_value (a b c : ℝ) : ℝ := a^2 * b^3 * c^4

theorem max_a2b3c4 (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  maximum_value a b c ≤ 143327232 / 386989855 := sorry

end max_a2b3c4_l197_197962


namespace sum_first_60_terms_l197_197168

-- Define the sequence according to the given recurrence relation
noncomputable def seq (n : ℕ) : ℕ → ℕ
| 0 => a_0  -- Need to decide initial value a_0
| (n + 1) => seq n + (-1)^n * seq n + 2*n - 1  -- Use the given recurrence relation

-- State the problem of the sum of the first 60 terms
theorem sum_first_60_terms (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n + 1) + (-1)^n * a n = 2*n - 1) :
  (∑ n in finset.range 60, a n) = 1830 :=
sorry

end sum_first_60_terms_l197_197168


namespace total_seeds_eaten_l197_197773

-- Definitions and conditions
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds
def first_four_players_seeds : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds
def average_seeds : ℕ := first_four_players_seeds / 4
def fifth_player_seeds : ℕ := average_seeds

-- Statement to prove
theorem total_seeds_eaten :
  first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds + fifth_player_seeds = 475 :=
by {
  sorry
}

end total_seeds_eaten_l197_197773


namespace part1_part2_l197_197720

def my_mul (x y : Int) : Int :=
  if x = 0 then abs y
  else if y = 0 then abs x
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then abs x + abs y
  else - (abs x + abs y)

theorem part1 : my_mul (-15) (my_mul 3 0) = -18 := 
  by
  sorry

theorem part2 (a : Int) : 
  my_mul 3 a + a = 
  if a < 0 then 2 * a - 3 
  else if a = 0 then 3
  else 2 * a + 3 :=
  by
  sorry

end part1_part2_l197_197720


namespace arithmetic_seq_8th_term_l197_197069

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197069


namespace minimum_log_function_l197_197426

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (3 - x) + log a (x + 1)

theorem minimum_log_function (a : ℝ) (h : 0 < a ∧ a < 1) 
  (hx : ∀ x : ℝ, -1 < x ∧ x < 3 → f a x ≥ -2 ) : a = 1 / 2 :=
sorry

end minimum_log_function_l197_197426


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197671

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197671


namespace arithmetic_sequence_8th_term_l197_197115

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197115


namespace slopes_sum_zero_parabola_equation_l197_197398

-- Problem 1: Sum of slopes k1 and k2 equals 0
theorem slopes_sum_zero
  (p : ℝ)
  (A B : ℝ × ℝ)
  (h_inter_A : A.1 = (A.2)^2 / (2*p))
  (h_inter_B : B.1 = (B.2)^2 / (2*p))
  (Q : ℝ × ℝ)
  (h_Q : Q = (-2, 0)) :
  let k1 := A.2 / (A.1 + 2) in
  let k2 := B.2 / (B.1 + 2) in
  k1 + k2 = 0 :=
sorry

-- Problem 2: Equation of the parabola
theorem parabola_equation
  (C : ℝ → ℝ)
  (P A B M N : ℝ × ℝ)
  (h_parabola : ∀ (x : ℝ), C x = (x / 2).sqrt)
  (h_inter_A : A.1 = (A.2)^2 / 2)
  (h_inter_B : B.1 = (B.2)^2 / 2)
  (h_not_AB : P ≠ A ∧ P ≠ B)
  (h_M : M.1 = -2)
  (h_N : N.1 = -2)
  (h_dot_product : M.2 * N.2 = 2 - 4) :
  ∀ (x : ℝ), (C x)^2 = x :=
sorry

end slopes_sum_zero_parabola_equation_l197_197398


namespace triangle_angles_l197_197046

theorem triangle_angles (A B C : ℝ) (hA : A = 120)
  (h_incenter_centroid : ∃ (I G : ℝ×ℝ), is_incenter I ⟨A, B, C⟩ ∧ is_centroid G ⟨A, B, C⟩ ∧ I = G) :
  B = 19 + 34/60 ∧ C = 40 + 26/60 := 
sorry

end triangle_angles_l197_197046


namespace vasya_problem_l197_197676

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l197_197676


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197661

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197661


namespace sum_of_x_for_sqrt_eq_nine_l197_197215

theorem sum_of_x_for_sqrt_eq_nine :
  (∑ x in Finset.filter (λ x, (abs (x - 2) = 9)) (Finset.range 100), x) = 4 :=
sorry

end sum_of_x_for_sqrt_eq_nine_l197_197215


namespace ratio_difference_l197_197184

theorem ratio_difference (x : ℕ) (h : 7 * x = 70) : 70 - 3 * x = 40 :=
by
  -- proof would go here
  sorry

end ratio_difference_l197_197184


namespace intersection_A_B_l197_197446

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := by
  -- Proof to be filled
  sorry

end intersection_A_B_l197_197446


namespace coefficient_x_squared_l197_197171

-- Define the conditions
variables (a : ℝ) (n : ℕ)

-- Condition 1: Sum of all coefficients in the binomial expansion of (1 + a * sqrt(x))^n is -1
def condition1 : (1 + a)^n = -1 := sorry

-- Condition 2: Sum of all binomial coefficients in (1 + a * sqrt(x))^n is 32
def condition2 : 2^n = 32 := sorry

-- Theorem: Coefficient of the term containing x^2 in the expansion of (1 + x) * (1 + a * sqrt(x))^n is 120
theorem coefficient_x_squared (h1 : condition1) (h2 : condition2) : 
  let exp := (1 + x) * (1 + a * sqrt x)^n in
  coeff exp (x^2) = 120 := sorry

end coefficient_x_squared_l197_197171


namespace line_intersects_circle_l197_197638

-- Define the line y = kx + 1
def line (k : ℝ) : ℝ → ℝ := λ x => k * x + 1

-- Define the circle x^2 + y^2 = 4
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- The point (0, 1) is on the line y = kx + 1
lemma point_on_line (k : ℝ) : line k 0 = 1 := by
  simp [line]

-- The point (0, 1) is inside the circle x^2 + y^2 = 4
lemma point_inside_circle : ¬ circle 0 1 := by
  simp [circle]
  norm_num

-- The line y = kx + 1 intersects the circle x^2 + y^2 = 4
theorem line_intersects_circle (k : ℝ) : ∃ x y : ℝ, line k x = y ∧ circle x y := by
  use 0, 1
  simp [line, circle, point_on_line, point_inside_circle]
  norm_num
  sorry

end line_intersects_circle_l197_197638


namespace simplify_expr_to_polynomial_l197_197231

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l197_197231


namespace scientific_notation_of_53_96_billion_l197_197769

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end scientific_notation_of_53_96_billion_l197_197769


namespace midpoint_of_centers_coincides_with_circumcenter_l197_197964

open EuclideanGeometry Real

variables {A B C I_A I_B P : Point} (Ω : Circle)

def excenter_of_incircle_tangent_to_side (A B C : Point) : Point := sorry
def circumcenter (A B C : Point) : Point := sorry
def circumcircle (A B C : Point) : Circle := sorry

theorem midpoint_of_centers_coincides_with_circumcenter :
  excenter_of_incircle_tangent_to_side A B C = I_A →
  excenter_of_incircle_tangent_to_side B C A = I_B →
  P ∈ circumcircle A B C →
  let O_A := circumcenter I_A C P,
      O_B := circumcenter I_B C P,
      M := midpoint O_A O_B
  in M = circumcenter A B C :=
  sorry

end midpoint_of_centers_coincides_with_circumcenter_l197_197964


namespace sequence_count_l197_197641

theorem sequence_count (a : ℕ → ℤ) (h₁ : a 1 = 0) (h₂ : a 11 = 4) 
  (h₃ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → |a (k + 1) - a k| = 1) : 
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end sequence_count_l197_197641


namespace paul_walking_time_l197_197530

variable (P : ℕ)

def is_walking_time (P : ℕ) : Prop :=
  P + 7 * (P + 2) = 46

theorem paul_walking_time (h : is_walking_time P) : P = 4 :=
by sorry

end paul_walking_time_l197_197530


namespace positive_sum_minus_terms_gt_zero_l197_197966

theorem positive_sum_minus_terms_gt_zero 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 1) : 
  a^2 + a * b + b^2 - a - b > 0 := 
by
  sorry

end positive_sum_minus_terms_gt_zero_l197_197966


namespace arithmetic_seq_8th_term_l197_197076

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197076


namespace arithmetic_sequence_8th_term_l197_197086

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197086


namespace determine_b_l197_197937

-- Definitions of points P, R, S, and area calculations based on the given conditions
def Point (x y : ℝ) : Type := {x: ℝ, y: ℝ}

def line (b : ℝ) (x : ℝ) : ℝ := b - x

def P (b : ℝ) : Point 0 b := ⟨0, b⟩

def S (b : ℝ) : Point 4 (b - 4) := ⟨4, b - 4⟩

def area_triangle (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) + P2.x * (P3.y - P1.y) + P3.x * (P1.y - P2.y))

theorem determine_b (b : ℝ) (h_pos : 0 < b) (h_lt : b < 4) 
  (h_ratio : (area_triangle (S b) ⟨4, b - 4⟩ ⟨b, 0⟩) / (area_triangle ⟨0,0⟩ (P b) ⟨b, 0⟩) = 9 / 25) : b = 5 / 2 :=
by
  sorry

end determine_b_l197_197937


namespace necessary_but_not_sufficient_l197_197518

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l197_197518


namespace monotonic_increasing_f_maximum_value_f_l197_197610

def f (x : ℝ) : ℝ := sin (1 / 2 * x) - 1 / 2 * sin x

theorem monotonic_increasing_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) : 
  ∀ {a b: ℝ}, (0 ≤ a ∧ a ≤ π ∧ 0 ≤ b ∧ b ≤ π) → a ≤ b → f a ≤ f b :=
sorry

theorem maximum_value_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∃ h_max : ℝ, h_max = f (4 * π / 3) ∧ h_max = (3 * Real.sqrt 3) / 4 :=
sorry

end monotonic_increasing_f_maximum_value_f_l197_197610


namespace maximum_k_l197_197842

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- Prove that the maximum integer value k satisfying k(x - 2) < f(x) for all x > 2 is 4.
theorem maximum_k (x : ℝ) (hx : x > 2) : ∃ k : ℤ, k = 4 ∧ (∀ x > 2, k * (x - 2) < f x) :=
sorry

end maximum_k_l197_197842


namespace biased_coin_four_heads_prob_l197_197252

/-- A biased coin is flipped seven times, and the probability of 
getting heads exactly twice is equal to the probability of getting 
heads exactly three times. The probability that the coin comes up 
heads in exactly four out of seven flips is equal to 675/3999. -/
theorem biased_coin_four_heads_prob : 
  ∀ (h : ℝ), 
  (0 < h) → (h < 1) → 
  (7.choose 2 * h^2 * (1 - h)^5 = 7.choose 3 * h^3 * (1 - h)^4) →
  (∃ (i j : ℕ), gcd i j = 1 ∧ 
    i + j = 4674 ∧ 
    (7.choose 4 * (3 / 8)^4 * (5 / 8)^3 = (i : ℝ) / j)) :=
by sorry

end biased_coin_four_heads_prob_l197_197252


namespace count_implications_l197_197785

def r : Prop := sorry
def s : Prop := sorry

def statement_1 := ¬r ∧ ¬s
def statement_2 := ¬r ∧ s
def statement_3 := r ∧ ¬s
def statement_4 := r ∧ s

def neg_rs : Prop := r ∨ s

theorem count_implications : (statement_2 → neg_rs) ∧ 
                             (statement_3 → neg_rs) ∧ 
                             (statement_4 → neg_rs) ∧ 
                             (¬(statement_1 → neg_rs)) -> 
                             3 = 3 := by
  sorry

end count_implications_l197_197785


namespace vasya_improved_example1_vasya_improved_example2_l197_197684

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l197_197684


namespace geo_sequence_ratio_l197_197038

theorem geo_sequence_ratio
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (q : ℝ)
  (hq1 : q = 1 → S_8 = 8 * a_n 0 ∧ S_4 = 4 * a_n 0 ∧ S_8 = 2 * S_4)
  (hq2 : q ≠ 1 → S_8 = 2 * S_4 → false)
  (hS : ∀ n, S_n n = a_n 0 * (1 - q^n) / (1 - q))
  (h_condition : S_8 = 2 * S_4) :
  a_n 2 / a_n 0 = 1 := sorry

end geo_sequence_ratio_l197_197038


namespace true_proposition_is_q_l197_197891

-- Defining the propositions p, q, and m
variables (p q m : Prop)

-- Defining the judgments A, B, and C
def A : Prop := p
def B : Prop := ¬ (p ∨ q)
def C : Prop := m

-- Stating the theorem
theorem true_proposition_is_q (h1 : (p ∨ q ∨ m) ∧ ¬ (p ∧ q) ∧ ¬ (p ∧ m) ∧ ¬ (q ∧ m))
                             (h2 : (A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ C)) : q :=
sorry

end true_proposition_is_q_l197_197891


namespace range_of_a_l197_197431

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → (x^2 + 2*x + a) / x > 0) ↔ a > -3 :=
by
  sorry

end range_of_a_l197_197431


namespace find_valid_trios_l197_197269

noncomputable def valid_trio (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ((a ∣ b ∨ b ∣ a) ∨ (a ∣ c ∨ c ∣ a) ∨ (b ∣ c ∨ c ∣ b)) ∧
  a + b + c = 4004

theorem find_valid_trios :
  ∀ (a b c : ℕ), a ∈ {1, ..., 2002} → b ∈ {1, ..., 2002} → c ∈ {1, ..., 2002} → 
  valid_trio a b c →
  (a, b, c) = (1, 2001, 2002) ∨
  (a, b, c) = (2, 2000, 2002) ∨
  (a, b, c) = (7, 1995, 2002) ∨
  (a, b, c) = (11, 1991, 2002) ∨
  (a, b, c) = (13, 1989, 2002) ∨
  (a, b, c) = (14, 1988, 2002) ∨
  (a, b, c) = (22, 1980, 2002) ∨
  (a, b, c) = (26, 1976, 2002) ∨
  (a, b, c) = (77, 1925, 2002) ∨
  (a, b, c) = (91, 1911, 2002) ∨
  (a, b, c) = (143, 1859, 2002) ∨
  (a, b, c) = (154, 1848, 2002) ∨
  (a, b, c) = (182, 1820, 2002) ∨
  (a, b, c) = (286, 1716, 2002) :=
by
  sorry

end find_valid_trios_l197_197269


namespace parallel_lines_m_eq_one_l197_197915

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y + (m - 2) = 0 ∧ 2 * m * x + 4 * y + 16 = 0 → m = 1) :=
by
  sorry

end parallel_lines_m_eq_one_l197_197915


namespace radii_difference_of_concentric_circles_l197_197165

theorem radii_difference_of_concentric_circles 
  (r : ℝ) 
  (h_area_ratio : (π * (2 * r)^2) / (π * r^2) = 4) : 
  (2 * r) - r = r :=
by
  sorry

end radii_difference_of_concentric_circles_l197_197165


namespace annuity_future_value_relation_l197_197378

theorem annuity_future_value_relation (n : ℕ) (P r K : ℝ) (h : K = P * ((1 + r / 100)^n - 1) / (r / 100) * (1 + r / 100)) : 
  let K' := P * ((1 + r / 100)^n - 1) / (r / 100)
  in K' = K / (1 + r / 100) :=
by
  sorry

end annuity_future_value_relation_l197_197378


namespace inequality_solution_l197_197607

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x else 0

def g (x : ℝ) : ℝ :=
  if x < 2 then 2 - x else 0

theorem inequality_solution (x : ℝ) : f(g(x)) > g(f(x)) → x < 0 :=
by
  sorry

end inequality_solution_l197_197607


namespace probability_of_factors_less_than_seven_is_half_l197_197207

open nat

def positive_factors (n : ℕ) : list ℕ :=
  list.filter (λ d, d > 0 ∧ d ∣ n) (list.range (n + 1))

def probability_less_than_seven (n : ℕ) : ℚ :=
  let factors := positive_factors n in
  let favorable := list.filter (λ d, d < 7) factors in
  (favorable.length : ℚ) / (factors.length : ℚ)

theorem probability_of_factors_less_than_seven_is_half :
  probability_less_than_seven 60 = 1 / 2 :=
by sorry

end probability_of_factors_less_than_seven_is_half_l197_197207


namespace sequence_seventh_term_l197_197505

theorem sequence_seventh_term :
  ∃ (t : ℕ → ℕ), 
    t 1 = 3 ∧
    t 5 = 18 ∧
    (∀ n ≥ 2, t n = (t (n - 1) + t (n + 1)) / 4) ∧
    t 7 = 54 :=
begin
  sorry
end

end sequence_seventh_term_l197_197505


namespace prob_factors_less_than_seven_l197_197210

theorem prob_factors_less_than_seven (n : ℕ) (h₁ : n = 60) :
  let factors := {d ∈ finset.range (n + 1) | n % d = 0}
  let prob := (factors.filter (< 7)).card.toRat / factors.card.toRat
  prob = 1 / 2 :=
by
  sorry

end prob_factors_less_than_seven_l197_197210


namespace original_number_l197_197040

theorem original_number (n : ℚ) (h : (3 * (n + 3) - 2) / 3 = 10) : n = 23 / 3 := 
sorry

end original_number_l197_197040


namespace trajectory_midpoint_l197_197482

theorem trajectory_midpoint {x y : ℝ} (hx : 2 * y + 1 = 2 * (2 * x)^2 + 1) :
  y = 4 * x^2 := 
by sorry

end trajectory_midpoint_l197_197482


namespace exp_increasing_l197_197418

-- Define the exponential function
noncomputable def exp (a x : ℝ) := a ^ x

-- Statement to prove
theorem exp_increasing (a : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < x₂ → exp a x₁ < exp a x₂) ↔ (1 < a) :=
begin
  sorry
end

end exp_increasing_l197_197418


namespace sum_of_solutions_l197_197218

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l197_197218


namespace minimum_value_of_a_l197_197438

variable (a x y : ℝ)

-- Condition
def condition (x y : ℝ) (a : ℝ) : Prop := 
  (x + y) * ((1/x) + (a/y)) ≥ 9

-- Main statement
theorem minimum_value_of_a : (∀ x > 0, ∀ y > 0, condition x y a) → a ≥ 4 :=
sorry

end minimum_value_of_a_l197_197438


namespace sum_of_solutions_l197_197220

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l197_197220


namespace raman_profit_percentage_l197_197988

-- Define the constants and conditions.
def butter_quantity_a : ℝ := 34 -- kg
def butter_quantity_b : ℝ := 36 -- kg
def price_per_kg_a : ℝ := 150 -- Rs./kg
def price_per_kg_b : ℝ := 125 -- Rs./kg
def selling_price_per_kg : ℝ := 192 -- Rs./kg

-- Define the proof statement.
theorem raman_profit_percentage :
  let cost_a := butter_quantity_a * price_per_kg_a in
  let cost_b := butter_quantity_b * price_per_kg_b in
  let total_cost := cost_a + cost_b in
  let total_quantity := butter_quantity_a + butter_quantity_b in
  let cost_price_per_kg := total_cost / total_quantity in
  let profit_percentage := ((selling_price_per_kg - cost_price_per_kg) / cost_price_per_kg) * 100 in
  profit_percentage ≈ 40 :=
sorry

end raman_profit_percentage_l197_197988


namespace arithmetic_mean_of_18_24_42_l197_197202

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l197_197202


namespace cannot_inscribe_convex_heptagon_l197_197528

def convex_heptagon_angles : Prop := 
  let angles := [140, 120, 130, 120, 130, 110, 150]
  let central_angles := angles.map (λ a, 2 * (180 - a))
  central_angles.sum <= 360

theorem cannot_inscribe_convex_heptagon : ¬ convex_heptagon_angles :=
by
  sorry

end cannot_inscribe_convex_heptagon_l197_197528


namespace parallel_lines_l197_197487

theorem parallel_lines (a : ℝ) :
  ((3 * a + 2) * x + a * y + 6 = 0) ↔
  (a * x - y + 3 = 0) →
  a = -1 :=
by sorry

end parallel_lines_l197_197487


namespace max_handshakes_l197_197722

theorem max_handshakes (n m : ℕ) (cond1 : n = 30) (cond2 : m = 5) 
                       (cond3 : ∀ (i : ℕ), i < 30 → ∀ (j : ℕ), j < 30 → i ≠ j → true)
                       (cond4 : ∀ (k : ℕ), k < 5 → ∃ (s : ℕ), s ≤ 10) : 
  ∃ (handshakes : ℕ), handshakes = 325 :=
by
  sorry

end max_handshakes_l197_197722


namespace sum_infinite_partial_fraction_l197_197307

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197307


namespace sum_of_coefficients_l197_197826

noncomputable def P (x : ℤ) : ℤ := (x ^ 2 - 3 * x + 1) ^ 100

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

end sum_of_coefficients_l197_197826


namespace abs_val_neg_three_l197_197128

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l197_197128


namespace q_correct_q_conditions_l197_197626

noncomputable def q (x : ℝ) : ℝ := (1 / 5) * (x + 2) * (x + 1) * (x - 1)

theorem q_correct :
  ∀ x, q(x) = (1 / 5) * x^3 + (2 / 5) * x^2 - (1 / 5) * x - (2 / 5) :=
  by
    sorry

theorem q_conditions :
  (q(3) = 8) :=
  by
    sorry

end q_correct_q_conditions_l197_197626


namespace renata_lottery_winnings_l197_197041

def initial_money : ℕ := 10
def donation : ℕ := 4
def prize_won : ℕ := 90
def water_cost : ℕ := 1
def lottery_ticket_cost : ℕ := 1
def final_money : ℕ := 94

theorem renata_lottery_winnings :
  ∃ (lottery_winnings : ℕ), 
  initial_money - donation + prize_won 
  - water_cost - lottery_ticket_cost 
  = final_money ∧ 
  lottery_winnings = 2 :=
by
  -- Proof steps will go here
  sorry

end renata_lottery_winnings_l197_197041


namespace suggested_bacon_students_l197_197603

-- Definitions based on the given conditions
def students_mashed_potatoes : ℕ := 330
def students_tomatoes : ℕ := 76
def difference_bacon_mashed_potatoes : ℕ := 61

-- Lean 4 statement to prove the correct answer
theorem suggested_bacon_students : ∃ (B : ℕ), students_mashed_potatoes = B + difference_bacon_mashed_potatoes ∧ B = 269 := 
by
  sorry

end suggested_bacon_students_l197_197603


namespace circle_area_from_diameter_l197_197049

-- Defining the points C and D
structure Point where
  x : ℝ
  y : ℝ

def C : Point := { x := 3, y := 5 }
def D : Point := { x := 8, y := 12 }

-- Function to compute the distance between two points
def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

-- Function to compute the area of a circle given its radius
def circle_area (r : ℝ) : ℝ :=
  Real.pi * r^2

-- The theorem statement
theorem circle_area_from_diameter :
  circle_area ((distance C D) / 2) = (74 * Real.pi) / 4 := 
by sorry

end circle_area_from_diameter_l197_197049


namespace sum_fraction_values_l197_197224

/-- Given x takes values from the set {-2023, -2022, -2021, ..., -2, -1, 0, 1, 1/2, 1/3, ..., 1/2021, 1/2022, 1/2023}, 
the sum of the expressions (x^2 - 1) / (x^2 + 1) for all such x values is -1. -/
theorem sum_fraction_values :
  let S := {x : ℚ | x ∈ finset.range (2023 + 2023 + 1) \ 2023},
      sum := (∑ x in S, (x * x - 1) / (x * x + 1) : ℚ) in
  sum = -1 := sorry

end sum_fraction_values_l197_197224


namespace arithmetic_seq_8th_term_l197_197077

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197077


namespace quadratic_function_value_l197_197552

theorem quadratic_function_value (x1 x2 a b : ℝ) (h1 : a ≠ 0)
  (h2 : 2012 = a * x1^2 + b * x1 + 2009)
  (h3 : 2012 = a * x2^2 + b * x2 + 2009) :
  (a * (x1 + x2)^2 + b * (x1 + x2) + 2009) = 2009 :=
by
  sorry

end quadratic_function_value_l197_197552


namespace parameter_a_cdf_correct_probability_interval_binomial_probability_l197_197744

noncomputable def pdf (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 3 * x^2 else 0

noncomputable def cdf (x : ℝ) : ℝ :=
if x ≤ 0 then 0
else if 0 < x ∧ x < 1 then x^3
else 1

theorem parameter_a : ∫ x in 0..1, pdf x = 1 :=
by sorry

theorem cdf_correct : ∀ x : ℝ, cdf x = 
  if x ≤ 0 then 0
  else if 0 < x ∧ x < 1 then x^3
  else 1 :=
by sorry

theorem probability_interval : ∫ x in 0.25..0.75, pdf x = 0.4 :=
by sorry

theorem binomial_probability : 
   ((nat.choose 4 3) * (0.4)^3 * (0.6) ≈ 0.1536) :=
by sorry


end parameter_a_cdf_correct_probability_interval_binomial_probability_l197_197744


namespace avg_weight_sec_B_is_correct_l197_197649

def avg_weight_sec_B {n_A n_B : ℕ} (W_A : ℝ) (W_total_avg : ℝ) : ℝ :=
  let W_total := W_total_avg * (n_A + n_B)
  let W_A_total := n_A * W_A
  (W_total - W_A_total) / n_B

theorem avg_weight_sec_B_is_correct :
  avg_weight_sec_B 36 44 40 37.25 = 35 :=
by
  sorry

end avg_weight_sec_B_is_correct_l197_197649


namespace cookies_per_pan_l197_197894

theorem cookies_per_pan 
  (pans : Nat)
  (total_cookies : Nat)
  (h1 : pans = 5)
  (h2 : total_cookies = 40) : 
  (total_cookies / pans = 8) :=
by
  rw [h1, h2]
  exact rfl

end cookies_per_pan_l197_197894


namespace sum_of_angles_approx_l197_197399

noncomputable def parallelogram_larger_angle_ratio : ℝ := 3 / 11 * 180
noncomputable def quadrilateral_largest_angle_ratio : ℝ := 150

theorem sum_of_angles_approx :
  (parallelogram_larger_angle_ratio + quadrilateral_largest_angle_ratio) ≈ 280.91 :=
by sorry

end sum_of_angles_approx_l197_197399


namespace nat_gt_10_is_diff_of_hypotenuse_numbers_l197_197199

def is_hypotenuse_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem nat_gt_10_is_diff_of_hypotenuse_numbers (n : ℕ) (h : n > 10) : 
  ∃ (n₁ n₂ : ℕ), is_hypotenuse_number n₁ ∧ is_hypotenuse_number n₂ ∧ n = n₁ - n₂ :=
by
  sorry

end nat_gt_10_is_diff_of_hypotenuse_numbers_l197_197199


namespace quadrilateral_concyclic_bisectors_l197_197987

theorem quadrilateral_concyclic_bisectors
  {A B C D : Type*}
  (Q1 : ∀ (α β γ φ : ℝ), α + β + γ + φ = 360)
  (Q2 : ∀ (α β γ φ : ℝ), 180 - (α / 2 + φ / 2) + 180 - (β / 2 + γ / 2) = 180)
  (Q3 : ∀ (A B C D P Q R S : Type*), true) : -- Dummy variables for quadrilateral vertex and intersection points
  ∃ (P Q R S : Type*), concyclic P S R Q :=
begin
  sorry
end

end quadrilateral_concyclic_bisectors_l197_197987


namespace fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l197_197382

open Complex

def inFourthQuadrant (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) > 0 ∧ (m^2 + 3*m - 28) < 0

def onNegativeHalfXAxis (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) < 0 ∧ (m^2 + 3*m - 28) = 0

def inUpperHalfPlaneIncludingRealAxis (m : ℝ) : Prop :=
  (m^2 + 3*m - 28) ≥ 0

theorem fourth_quadrant_for_m (m : ℝ) :
  (-7 < m ∧ m < 3) ↔ inFourthQuadrant m := 
sorry

theorem negative_half_x_axis_for_m (m : ℝ) :
  (m = 4) ↔ onNegativeHalfXAxis m :=
sorry

theorem upper_half_plane_for_m (m : ℝ) :
  (m ≤ -7 ∨ m ≥ 4) ↔ inUpperHalfPlaneIncludingRealAxis m :=
sorry

end fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l197_197382


namespace revenue_decreases_6_5_percent_l197_197707

variables (T C : ℝ)

theorem revenue_decreases_6_5_percent (h1 : T > 0) (h2 : C > 0) :
  let T_new := T * 0.85,
      C_new := C * 1.10,
      R := T * C,
      R_new := T_new * C_new in
  (R_new / R) = 0.935 ∧ (100 - (R_new / R * 100) = 6.5) :=
by
  let T_new := T * 0.85
  let C_new := C * 1.10
  let R := T * C
  let R_new := T_new * C_new
  have h3 : R_new / R = 0.935, by sorry
  have h4 : 100 - (R_new / R * 100) = 6.5, by sorry
  exact and.intro h3 h4

end revenue_decreases_6_5_percent_l197_197707


namespace div_seven_and_sum_factors_l197_197587

theorem div_seven_and_sum_factors (a b c : ℤ) (h : (a = 0 ∨ b = 0 ∨ c = 0) ∧ ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * 7 * (a + b) * (b + c) * (c + a) :=
by
  sorry

end div_seven_and_sum_factors_l197_197587


namespace infinite_series_converges_l197_197317

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197317


namespace max_stores_visited_l197_197179

theorem max_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ)
  (people_two_stores : ℕ) (visits_two_stores_each : ℕ)
  (visits_unaccounted : ℕ) (remaining_people : ℕ)
  (minimum_visits_remaining_people : ℕ) :
  stores = 8 →
  total_visits = 21 →
  total_shoppers = 12 →
  people_two_stores = 8 →
  visits_two_stores_each = 2 →
  (people_two_stores * visits_two_stores_each + visits_unaccounted) = total_visits →
  remaining_people = (total_shoppers - people_two_stores) →
  visits_unaccounted = 5 →
  minimum_visits_remaining_people = 1 →
  (remaining_people -- 1 and each of the remaining must visit at least 1 store) →
  (remaining_people - 1) * minimum_visits_remaining_people + 2 visits = visits_unaccounted →
  max_stores_visited = 3 :=
begin
  sorry
end

end max_stores_visited_l197_197179


namespace obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197662

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l197_197662


namespace sum_series_l197_197324

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197324


namespace intercepts_sum_l197_197172

theorem intercepts_sum (x y : ℝ) (h : x - 2 * y + 1 = 0) :
  let y_intercept := (λ x, -((x + 1) / 2)) 0 in
  let x_intercept := (λ y, -1 - y) 0 in
  y_intercept + x_intercept = -1 / 2 :=
sorry

end intercepts_sum_l197_197172


namespace hyperbola_eccentricity_range_l197_197782

-- Define the hyperbola and its characteristics.
variables (a b c : ℝ)

-- Define the eccentricity e of the hyperbola.
def eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (1 + (a^2) / (b^2))

-- Variables for points A, B, and the focus F.
variables (A B F : ℝ × ℝ)

-- Conditions given in the problem.
axiom hyperbola_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1
axiom asymptotes : ∀ (x : ℝ), ∃ y : ℝ, y = ((b * x) / a) ∨ y = -((b * x) / a)
axiom line_intersection : ∃ A B : ℝ × ℝ,
  A = (a^2 / c, b * a / c) ∧ B = (a^2 / c, -b * a / c)
axiom focus_F : ∃ F : ℝ × ℝ, -- Define coordinates for F if necessary
axiom angle_condition : ∀ (A B F : ℝ × ℝ) (θ : ℝ),
  60 < θ ∧ θ < 90 → ∃ k_FB : ℝ, (1 / √3) < k_FB ∧ k_FB < 1

-- The Lean version of the proof problem.
theorem hyperbola_eccentricity_range
  (A B F : ℝ × ℝ) (a b c : ℝ) (e : ℝ)
  (h1 : hyperbola_eq a b)
  (h2 : asymptotes a b)
  (h3 : line_intersection a b c A B)
  (h4 : focus_F F)
  (h5 : angle_condition A B F) :
  (real.sqrt 2) < e ∧ e < 2 :=
sorry

end hyperbola_eccentricity_range_l197_197782


namespace polynomial_representation_l197_197228

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l197_197228


namespace cube_root_simplification_l197_197057

theorem cube_root_simplification : (∛(30^3 + 40^3 + 50^3) = 60) :=
by 
  sorry

end cube_root_simplification_l197_197057


namespace find_ab_over_10_l197_197790

def double_fac_odds (n : ℕ) : ℕ :=
if n % 2 = 1 then ((n + 1) * (n - 1) * ... * 3) else 1

def double_fac_evens (n : ℕ) : ℕ :=
if n % 2 = 0 then (n * (n - 2) * ... * 2) else 1

noncomputable def double_factorial (n : ℕ) : ℕ :=
if n % 2 = 1 then double_fac_odds n else double_fac_evens n

noncomputable def S : ℚ :=
∑ i in (finset.range 2009).map (λ x, x + 1), (double_factorial (2 * i - 1)) / (double_factorial (2 * i))

theorem find_ab_over_10 : (∃ a b : ℕ, 
  let denom := int.nat_abs S.denom,
  let b := denom / (2^a),
  denom = 2^a * b ∧ @has_zero.zero nat _ (b % 2 = 1)
) ∧ (ab : ℚ := a * b) ∧ (ab / 10 = 401) :=
by
  sorry

end find_ab_over_10_l197_197790


namespace trig_identity_l197_197841

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 :=
by 
  sorry

end trig_identity_l197_197841


namespace count_even_sum_subsets_l197_197820

open Finset

-- Given conditions
def even_sum_subset_property (s : Finset ℕ) :=
  ∃ a b c d, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ a ≠ b ∧ c ≠ d ∧ a + b = 16 ∧ c + d = 24

-- Main theorem statement
theorem count_even_sum_subsets :
  (univ.filter even_sum_subset_property).card = 210 :=
sorry

end count_even_sum_subsets_l197_197820


namespace length_of_st_l197_197944

theorem length_of_st (P Q R S T : Type) [EuclideanGeometry P Q R S T] 
  (QR_eq : dist Q R = 30)
  (angle_R_eq : ∠ PQR = 45)
  (midpoint_S : is_midpoint S Q R)
  (perpendicular_bisector_st : is_perpendicular (line_from S R) (line_from T S)) :
  dist S T = 15 * Real.sqrt 2 / 2 :=
by
  sorry

end length_of_st_l197_197944


namespace final_value_of_A_l197_197693

theorem final_value_of_A :
  let A := 1 in
  let A := A * 2 in
  let A := A * 3 in
  let A := A * 4 in
  let A := A * 5 in
  A = 120 :=
by
  sorry

end final_value_of_A_l197_197693


namespace number_of_odd_divisors_lt_100_is_9_l197_197465

theorem number_of_odd_divisors_lt_100_is_9 :
  (finset.filter (λ n : ℕ, ∃ k : ℕ, n = k * k) (finset.range 100)).card = 9 :=
sorry

end number_of_odd_divisors_lt_100_is_9_l197_197465


namespace minimum_value_l197_197868

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y = 1) :
  ∀ (z : ℝ), z = (1/x + 1/y) → z ≥ 3 + 2*Real.sqrt 2 :=
by
  sorry

end minimum_value_l197_197868


namespace counterexamples_sum_of_digits_five_l197_197819

theorem counterexamples_sum_of_digits_five : 
  ∃ (count : ℕ), count = 7 ∧ 
  ∀ N : ℕ, (sum_of_digits N = 5 ∧ (∀ d : ℕ, d ∈ digits N → d ≠ 0)) → 
    (is_prime N ∨ count_of_non_prime N = count) := 
    sorry

end counterexamples_sum_of_digits_five_l197_197819


namespace number_drawn_from_first_group_l197_197659

theorem number_drawn_from_first_group :
  ∀ (students : ℕ) (groups : ℕ) (group_size : ℕ) (sixteenth_group_start : ℕ) (sixteenth_group_draw : ℕ),
  students = 160 →
  groups = 20 →
  group_size = students / groups →
  sixteenth_group_start = (16 - 1) * group_size + 1 →
  sixteenth_group_draw = 126 →
  ∃ (first_group_draw : ℕ), first_group_draw = 6 :=
by
  intros students groups group_size sixteenth_group_start sixteenth_group_draw
  assume h_students h_groups h_group_size h_sixteenth_group_start h_sixteenth_group_draw
  let first_group_start := 1
  let first_group_draw := first_group_start + (sixteenth_group_draw - sixteenth_group_start)
  existsi first_group_draw
  sorry

end number_drawn_from_first_group_l197_197659


namespace cos_C_of_right_triangle_l197_197493

theorem cos_C_of_right_triangle (A B C : Point) (h : ∠A = 90°) (tanC : tan ∠C = 5/2) :
  cos ∠C = 2 * sqrt 29 / 29 :=
sorry

end cos_C_of_right_triangle_l197_197493


namespace sequence_term_100_eq_l197_197167

noncomputable def a : ℕ → ℝ
| 1 => 2
| 2 => 1
| n => (a n * a (n-1)) / (a (n-1) - a n) * (a n - (a (n+1)))

theorem sequence_term_100_eq :
  a 100 = 1 / 50 := sorry

end sequence_term_100_eq_l197_197167


namespace pasture_feeding_days_l197_197627

-- Definitions: Number of cows, days, and the growth rate of grass
def grass_consistent_growth (n_cows_1 n_cows_2 days_1 days_2 : ℕ) (rate : ℕ) : Prop :=
  ∀ C R G, 
    (n_cows_1 * C * days_1 = G + days_1 * R) ∧
    (n_cows_2 * C * days_2 = G + days_2 * R) ∧
    (rate = R)

-- Statement of the proof problem
theorem pasture_feeding_days
  (n_cows_1 : ℕ) (days_1 : ℕ)
  (n_cows_2 : ℕ) (days_2 : ℕ)
  (n_cows_3 : ℕ) (expected_days : ℕ)
  (rate : ℕ)
  (h_growth : grass_consistent_growth n_cows_1 n_cows_2 days_1 days_2 rate) :
  n_cows_3 * days_1 = expected_days :=
by {
  sorry
}

-- Parameter values for the specific problem
example : pasture_feeding_days 20 40 35 10 25 20 15 (by apply_instance) :=
by {
  sorry
}

end pasture_feeding_days_l197_197627


namespace AB_length_possibilities_l197_197522

theorem AB_length_possibilities (ABC : Triangle) (A B C : Point) (h_AC_eq_1 : side_length A C = 1) 
  (h_median_divides_angle : ∃ (epsilon phi : Angle), ∠CAB = epsilon + phi ∧ phi = 2 * epsilon ∨ epsilon = 2 * phi) :
  ∃ (AB : ℝ), (1 / 2) < AB ∧ AB < 2 ∧ AB ≠ 1 :=
by
  sorry

end AB_length_possibilities_l197_197522


namespace arithmetic_seq_8th_term_l197_197075

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197075


namespace number_of_paths_l197_197896

theorem number_of_paths (n m : ℕ) (h : m ≤ n) : 
  ∃ paths : ℕ, paths = Nat.choose n m := 
sorry

end number_of_paths_l197_197896


namespace hi_mom_box_office_revenue_scientific_notation_l197_197767

def box_office_revenue_scientific_notation (billion : ℤ) (revenue : ℤ) : Prop :=
  revenue = 5.396 * 10^9

theorem hi_mom_box_office_revenue_scientific_notation :
  box_office_revenue_scientific_notation 53.96 53960000000 :=
by
  sorry

end hi_mom_box_office_revenue_scientific_notation_l197_197767


namespace train_cross_tunnel_in_one_minute_l197_197754

-- Definitions for the conditions
def train_length : ℝ := 800 -- in meters
def train_speed_kmh : ℝ := 78 -- in km/hr
def tunnel_length : ℝ := 500 -- in meters

-- Auxiliary definition to convert speed from km/hr to m/s
def kmh_to_mps (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600

-- Definition for total distance to be covered by the train
def total_distance : ℝ := train_length + tunnel_length

-- Definition for train speed in meters per second
def train_speed_mps : ℝ := kmh_to_mps train_speed_kmh

-- Definition for time to cross the tunnel in seconds
def time_seconds : ℝ := total_distance / train_speed_mps

-- Definition for time to cross the tunnel in minutes
def time_minutes : ℝ := time_seconds / 60

theorem train_cross_tunnel_in_one_minute : time_minutes = 1 := by
  sorry

end train_cross_tunnel_in_one_minute_l197_197754


namespace gcd_max_value_l197_197765

theorem gcd_max_value : ∀ (n : ℕ), n > 0 → ∃ (d : ℕ), d = 9 ∧ d ∣ gcd (13 * n + 4) (8 * n + 3) :=
by
  sorry

end gcd_max_value_l197_197765


namespace exam_time_allocation_l197_197921

theorem exam_time_allocation : 
    ∀ (total_time_minutes num_questions num_typeA num_typeB: ℕ) 
      (time_typeA_per_problem time_typeB_per_problem: ℝ), 
    total_time_minutes = 180 → 
    num_questions = 200 → 
    num_typeA = 50 → 
    num_typeB = num_questions - num_typeA → 
    time_typeA_per_problem = 2 * time_typeB_per_problem → 
    (num_typeA * time_typeA_per_problem + num_typeB * time_typeB_per_problem = total_time_minutes) →
    num_typeA * time_typeA_per_problem = 72 :=
by
  intros total_time_minutes num_questions num_typeA num_typeB time_typeA_per_problem time_typeB_per_problem 
  ht hn hA hB tA_eq_2_tB htotal
  sorry

end exam_time_allocation_l197_197921


namespace find_length_PQ_l197_197935

-- Define the points, medians, and segment lengths according to the problem's conditions.

variable (P Q R S T M Q' R' : Point)
variable (PS SR QT : ℝ)

-- Given conditions
def cond1 : PS = 8 := sorry
def cond2 : SR = 16 := sorry
def cond3 : QT = 12 := sorry
def cond4 : midpoint M Q R := sorry
def cond5 : reflect_over_median PQ' R' P := sorry

-- The main theorem stating the problem and its correct answer.
theorem find_length_PQ (h1 : PS = 8) (h2 : SR = 16) (h3 : QT = 12) 
  (h4 : midpoint M Q R) (h5 : reflect_over_median PQ' R' P) : 
  PQ = 4 * sqrt 17 := 
sorry

end find_length_PQ_l197_197935


namespace sum_infinite_partial_fraction_l197_197305

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197305


namespace newspapers_ratio_l197_197946

theorem newspapers_ratio :
  (∀ (j m : ℕ), j = 234 → m = 4 * j + 936 → (m / 4) / j = 2) :=
by
  sorry

end newspapers_ratio_l197_197946


namespace Igor_lied_l197_197763

-- Define the players
inductive Player
| Andrey
| Maxim
| Igor
| Kolya

open Player

-- Define the places
inductive Place
| First
| Second
| Third
| Last

open Place

-- Define the statements made by each player
def Statements : Player → Place → Prop
| Andrey, place => place ≠ First ∧ place ≠ Last
| Maxim, place  => place ≠ Last
| Igor, place   => place = First
| Kolya, place  => place = Last

-- Given conditions
def isHonest (p : Player) (place : Place) : Prop := Statements p place

-- Define that exactly three boys are honest and one is lying
def exactlyThreeHonest (plc : Player → Place) : Prop :=
(count (λ p => isHonest p (plc p))) 3

-- Lean 4 statement to prove Igor is lying
theorem Igor_lied (places : Player → Place) (H : exactlyThreeHonest places) : ¬ isHonest Igor (places Igor) :=
sorry

end Igor_lied_l197_197763


namespace equation_of_line_passing_through_point_and_intersects_circle_smallest_circle_through_P_and_C_l197_197397

open Real EuclideanGeometry

-- Problem 1
theorem equation_of_line_passing_through_point_and_intersects_circle (P : Point ℝ) 
    (C : Circle ℝ) (A B : Point ℝ) (k₁ k₂ : ℝ) : 
    (P = (2, 1)) →
    (C = Circle.mk (Equiv.prod (Equiv.refl ℝ) (Equiv.refl ℝ)) ⟨-1, 2⟩ 4) →
    ((A ≠ B) ∧ (A ∈ C.points) ∧ (B ∈ C.points) ∧ (angle A C B = π / 2)) →
    (¬ collinear [P, A, B]) →
    (∀ k, k = k₁ ∨ k = k₂) →
    (k₁ = 1 ∨ k₁ = -7) ∧ (k₂ = 1 ∨ k₂ = -7) →
    (equation_of_line P k₁ = "x - y - 1 = 0") ∨ 
    (equation_of_line P k₂ = "7x + y - 15 = 0") :=
begin
  intros hP hC hA hB hK hk,
  sorry
end

-- Problem 2
theorem smallest_circle_through_P_and_C (P C : Point ℝ) (r : ℝ) :
    P = (2, 1) →
    C = (-1, 2) →
    let center := (1/2, 3/2) in
    r = √5/2 →
    equation_of_circle center r = "(x - 1/2)^2 + (y - 3/2)^2 = 5/2" :=
begin
  intros hP hC center hr,
  sorry
end

end equation_of_line_passing_through_point_and_intersects_circle_smallest_circle_through_P_and_C_l197_197397


namespace circumcenter_on_Euler_line_l197_197553

-- Define the basic setup of the problem with the given conditions
variables {A B C K_a L_a M_a X_a X_b X_c : Type} 

def triangle_scalene (ABC : Type) : Prop := sorry -- Scalene property placeholder

def intersects_internal_angle_bisector (A B C K_a : Type) : Prop := sorry -- Intersection placeholder
def intersects_external_angle_bisector (A B C L_a : Type) : Prop := sorry -- Intersection placeholder
def intersects_median (A B C M_a : Type) : Prop := sorry -- Intersection placeholder

def circumcircle_intersects (A K_a L_a M_a X_a : Type) : Prop := sorry -- Circumcircle intersection placeholder

-- Define the setup of the Euler line
def Euler_line (A B C : Type) : Type := sorry

-- Define the circumcenter of the triangle
def circumcenter (X_a X_b X_c : Type) : Type := sorry

-- The proof statement in Lean
theorem circumcenter_on_Euler_line
    (h1 : triangle_scalene ABC)
    (h2 : intersects_internal_angle_bisector A B C K_a)
    (h3 : intersects_external_angle_bisector A B C L_a)
    (h4 : intersects_median A B C M_a)
    (h5 : circumcircle_intersects A K_a L_a M_a X_a)
    (h6 : circumcircle_intersects B K_b L_b M_b X_b)
    (h7 : circumcircle_intersects C K_c L_c M_c X_c) : 
    (circumcenter X_a X_b X_c) ∈ (Euler_line A B C) := 
sorry

end circumcenter_on_Euler_line_l197_197553


namespace tan_22_5_expression_l197_197159

-- Define the problem in Lean
theorem tan_22_5_expression : ∃ (a b c d : ℤ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a > 0 ∧ b > 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 3 ∧ (tan(π / 8) = real.sqrt a - real.sqrt b + real.sqrt c - d) :=
by
  have := real.tan_pi_div_eight -- gives us that tan(π / 8) = √2 - 1
  use [2, 1, 0, 0]
  split; norm_num -- ensure positivity and ordering of integers (a = 2, b = 1, c = 0, d = 0)
  split; linarith -- finalize the proof of the form
  sorry -- to be filled in

end tan_22_5_expression_l197_197159


namespace remainder_when_x14_plus_1_divided_by_x_plus_1_l197_197359

theorem remainder_when_x14_plus_1_divided_by_x_plus_1 :
  (polynomial.eval (-1) (polynomial.C 1 + polynomial.X ^ 14)) = 2 :=
by sorry

end remainder_when_x14_plus_1_divided_by_x_plus_1_l197_197359


namespace arithmetic_sequence_8th_term_is_71_l197_197106

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197106


namespace sum_fraction_bounded_l197_197393

theorem sum_fraction_bounded (n : ℕ) (x : Fin n → ℝ) 
  (h1 : 2 ≤ n) 
  (h2 : ∑ i, |x i| = 1) 
  (h3 : ∑ i, x i = 0) : 
  abs (∑ i in Finset.range n, x ⟨i, Finset.mem_range.mpr (Nat.lt_of_lt_succ (Nat.lt_of_lt_succ h1))⟩ / (i + 1)) ≤ 1 / 2 - 1 / (2 * n) :=
by
  sorry

end sum_fraction_bounded_l197_197393


namespace rate_of_paving_l197_197631

theorem rate_of_paving (length width : ℝ) (total_cost : ℝ) (h_length : length = 5.5) (h_width : width = 3.75) (h_total_cost : total_cost = 12375) :
  total_cost / (length * width) = 600 :=
by
  calc
    total_cost / (length * width) = 12375 / (5.5 * 3.75) : by rw [h_length, h_width, h_total_cost]
                                ... = 12375 / 20.625       : by norm_num
                                ... = 600                  : by norm_num

end rate_of_paving_l197_197631


namespace sum_series_l197_197319

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197319


namespace arithmetic_sequence_8th_term_l197_197114

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197114


namespace arithmetic_sequence_8th_term_l197_197124

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197124


namespace negation_proposition_l197_197637

theorem negation_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l197_197637


namespace number_of_integers_satisfying_abs_x_lt_4pi_l197_197455

theorem number_of_integers_satisfying_abs_x_lt_4pi : 
    ∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, (abs x < 4 * Real.pi) → x ∈ Icc (-12:ℤ) (12:ℤ) := by
  sorry

end number_of_integers_satisfying_abs_x_lt_4pi_l197_197455


namespace evaluate_expression_l197_197908

variable y : ℝ
variable Q : ℝ

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.pi) = Q

-- Target expression evaluation
def target_expression : ℝ := 10 * (6 * y + 14 * Real.pi + y^2)

-- Required proof statement
theorem evaluate_expression (h : condition) : target_expression = 4 * Q + 10 * y^2 := 
by sorry

end evaluate_expression_l197_197908


namespace vector_projection_l197_197375

theorem vector_projection (a : ℝ) (b : ℝ) (c : ℝ) :
    let u := (3, -5, -2)
    let v := (1, -3, 2)
    u • v / (v • v) * v = (1, -3, 2) :=
by
  let u : ℝ × ℝ × ℝ := (3, -5, -2)
  let v : ℝ × ℝ × ℝ := (1, -3, 2)
  sorry

end vector_projection_l197_197375


namespace percentage_calculation_l197_197206

def part : ℝ := 12.356
def whole : ℝ := 12356
def expected_percentage : ℝ := 0.1

theorem percentage_calculation (p w : ℝ) (h_p : p = part) (h_w : w = whole) : 
  (p / w) * 100 = expected_percentage :=
sorry

end percentage_calculation_l197_197206


namespace infinite_series_converges_l197_197316

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197316


namespace scalar_v_cross_products_l197_197652

noncomputable def scalar_d : ℝ := 4

theorem scalar_v_cross_products (v : ℝ^3) (i j k : ℝ^3) (H1: (i.dot i = 1)) (H2: (j.dot j = 1)) (H3: (k.dot k = 1)) 
  (H4: (i.dot j = 0)) (H5: (i.dot k = 0)) (H6: (j.dot k = 0)) :
  i × (v × (2 • i)) + j × (v × (2 • j)) + k × (v × (2 • k)) = scalar_d • v := sorry

end scalar_v_cross_products_l197_197652


namespace length_segment_AB_l197_197632

variable {x : ℝ}
def line := λ x : ℝ, x + 1
def curve := λ x : ℝ, (1 / 2) * x^2 - 1

theorem length_segment_AB : 
  let A := (1 + Real.sqrt 5, line (1 + Real.sqrt 5)),
      B := (1 - Real.sqrt 5, line (1 - Real.sqrt 5)) in
  Real.dist A B = 2 * Real.sqrt 10 := 
by
  sorry

end length_segment_AB_l197_197632


namespace min_white_cells_to_paint_l197_197929

theorem min_white_cells_to_paint {n : ℕ} (h : 2 ≤ n) :
  let grid := matrice.init n n (λ i j, if (i = 0 ∧ j = 0) ∨ (i = n-1 ∧ j = n-1) then 1 else 0) in
  let transformation := ∀ i j, grid i j = 1 in
  ∃ k, k = 2 * n - 4 := 
sorry

end min_white_cells_to_paint_l197_197929


namespace polynomial_representation_l197_197229

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l197_197229


namespace divisor_exists_l197_197728

open Nat

noncomputable def added_number : ℝ := 48.00000000010186
def n₁ : ℕ := 1782452
noncomputable def n₂ : ℝ := n₁ + added_number
def rounded_n₂ : ℕ := 1782500

theorem divisor_exists :
  ∃ d : ℕ, d = 500 ∧ rounded_n₂ % d = 0 :=
by
  use 500
  split
  · rfl
  · simp
  sorry

end divisor_exists_l197_197728


namespace joel_strawberries_area_l197_197949

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end joel_strawberries_area_l197_197949


namespace correct_option_D_l197_197941

def traffic_flow (Q: ℝ) (V: ℝ) (K: ℝ) := Q = V * K
def vehicle_flow_speed (V: ℝ) (v_0: ℝ) (K: ℝ) (k_0: ℝ) := V = v_0 * (1 - K / k_0)
def quadratic_behavior (K: ℝ) (v_0: ℝ) (k_0: ℝ) : Prop :=
  ∀ Q V: ℝ, vehicle_flow_speed V v_0 K k_0 → traffic_flow Q V K → (∃ K_max, (0 < K_max) ∧
  (∀ K', 0 ≤ K' ∧ K' ≤ K_max → Q' Q K') ∧ (∀ K', K_max ≤ K' → decreasing_flow Q K'))

theorem correct_option_D (v_0 k_0: ℝ) (hv0: 0 < v_0) (hk0: 0 < k_0):
  quadratic_behavior K v_0 k_0
:= sorry

end correct_option_D_l197_197941


namespace product_of_solutions_eq_zero_l197_197798

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → (x = 0 ∨ x = -4 / 7)) → (0 = 0) := 
by
  intros h
  sorry

end product_of_solutions_eq_zero_l197_197798


namespace probability_of_yellow_or_green_l197_197212

def bag : List (String × Nat) := [("yellow", 4), ("green", 3), ("red", 2), ("blue", 1)]

def total_marbles (bag : List (String × Nat)) : Nat := bag.foldr (fun (_, n) acc => n + acc) 0

def favorable_outcomes (bag : List (String × Nat)) : Nat :=
  (bag.filter (fun (color, _) => color = "yellow" ∨ color = "green")).foldr (fun (_, n) acc => n + acc) 0

theorem probability_of_yellow_or_green :
  (favorable_outcomes bag : ℚ) / (total_marbles bag : ℚ) = 7 / 10 := by
  sorry

end probability_of_yellow_or_green_l197_197212


namespace initial_investment_l197_197571

theorem initial_investment (P : ℝ) (x : ℝ) (final_amount : ℝ) (years : ℝ) (rate : ℝ) (tripling_period : ℝ)
    (h1 : rate = 8) 
    (h2 : tripling_period = 112 / rate) 
    (h3 : final_amount = 18000) 
    (h4 : years = 28) 
    (h5 : (years / tripling_period).ceil = 2) :
  P * (3 ^ 2) = final_amount → P = 2000 :=
by
  sorry

end initial_investment_l197_197571


namespace meeting_probability_l197_197577

theorem meeting_probability :
  let C := (0, 0)
  let D := (6, 8)
  let moves_C := [{1, 0}, {0, 1}]
  let moves_D := [{-1, 0}, {0, -1}]
  let steps := 8
  -- Probability calculation 
  -- Using combinatorial methods to calculate individual probabilities
  let c_i (i : ℕ) : ℕ := Nat.choose 7 i
  let d_i (i : ℕ) : ℕ := Nat.choose 8 (i + 1)
  -- Summing over all possible meeting points
  let meeting_prob : ℚ := (Finset.sum (Finset.range 7) (fun i => (c_i i * d_i i))) / (2 ^ 15)
  -- Expected probability result
  meeting_prob = 203 / 32768

end meeting_probability_l197_197577


namespace pairs_of_boys_girls_l197_197581

theorem pairs_of_boys_girls (a_g b_g a_b b_b : ℕ) 
  (h1 : a_b = 3 * a_g)
  (h2 : b_b = 4 * b_g) :
  ∃ c : ℕ, b_b = 7 * b_g :=
sorry

end pairs_of_boys_girls_l197_197581


namespace compute_a_b_difference_square_l197_197960

noncomputable def count_multiples (m n : ℕ) : ℕ :=
  (n - 1) / m

theorem compute_a_b_difference_square :
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  (a - b) ^ 2 = 0 :=
by
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  show (a - b) ^ 2 = 0
  sorry

end compute_a_b_difference_square_l197_197960


namespace smallest_value_of_S_l197_197150

theorem smallest_value_of_S :
  ∃ (a_1 a_2 a_3 b_1 b_2 b_3 c_1 c_2 c_3 d_1 d_2 d_3 : ℕ),
  set.univ = {a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3, d_1, d_2, d_3} ∧
  (∀ i ∈ {a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3, d_1, d_2, d_3}, 1 ≤ i ∧ i ≤ 10) ∧
  a_1 * a_2 * a_3 + b_1 * b_2 * b_3 + c_1 * c_2 * c_3 + d_1 * d_2 * d_3 = 613 :=
by
  sorry

end smallest_value_of_S_l197_197150


namespace find_x₀_l197_197848

variables {x₀ y₀ : ℝ}

def parabola (x y : ℝ) : Prop := y^2 = 2 * x

noncomputable def focus : (ℝ × ℝ) := (1/2, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem find_x₀ (h_point_on_parabola : parabola x₀ y₀)
  (h_distance_focus : distance (x₀, y₀) focus = 2 * x₀) :
  x₀ = 1 / 2 :=
sorry

end find_x₀_l197_197848


namespace find_z_range_m_l197_197865
  
noncomputable def z (a b : ℂ) : ℂ := a - 2 * Complex.i

theorem find_z 
    (a b : ℂ)
    (hz1 : (z a b + 2 * Complex.i).im = 0) 
    (hz2 : (z a b / (2 - Complex.i)).im = 0) :
  z a b = 4 - 2 * Complex.i := 
  sorry

theorem range_m 
    (m : ℝ)
    (hz : (Complex.abs (4 + (m - 2) * Complex.i) ≤ 5)) :
  -1 ≤ m ∧ m ≤ 5 := 
  sorry

end find_z_range_m_l197_197865


namespace probability_of_one_defective_l197_197922

open Finset

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω] {products : Finset Ω} {quality defective : Finset Ω}

/--
  In a batch of 10 products, there are 7 quality products and 3 defective ones.
  If 4 products are randomly selected, the probability of exactly getting 1 defective product is 1/2.
-/
theorem probability_of_one_defective 
  (h1 : products.card = 10)
  (h2 : quality.card = 7)
  (h3 : defective.card = 3)
  (h4 : Disjoint quality defective)
  (h5 : quality ∪ defective = products)
  (h6 : (Finset.choose 4 products).Nonempty ) :
  (card { s ∈ Finset.choose 4 products | card (s ∩ defective) = 1 } : ℚ) / 
  (card (Finset.choose 4 products) : ℚ) = 1 / 2 :=
sorry

end probability_of_one_defective_l197_197922


namespace abs_neg_three_l197_197131

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l197_197131


namespace cube_volume_split_l197_197845

theorem cube_volume_split (x y z : ℝ) (h : x > 0) :
  ∃ y z : ℝ, y > 0 ∧ z > 0 ∧ y^3 + z^3 = x^3 :=
sorry

end cube_volume_split_l197_197845


namespace equivalent_n_mod_3_l197_197364

noncomputable def exists_permutation_sum (n : ℕ) : Prop :=
  ∃ σ : (Fin n) → (Fin n), bij_on σ (Set.univ) ∧ 
  (∑ i in Finset.range n, (σ ⟨i, by simp [Fin.is_lt]⟩).val * (-2) ^ i = 0)

theorem equivalent_n_mod_3 (n : ℕ) : 
  exists_permutation_sum n ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end equivalent_n_mod_3_l197_197364


namespace solve_c_l197_197409

theorem solve_c (a b c d e : ℝ) (h1 : a - (-1) = b - a) (h2 : b - a = -4 - b)
                (h3 : c / -1 = d / c) (h4 : d / c = e / d) (h5 : e / d = -4 / e) :
                c = - real.sqrt 2 :=
by sorry

end solve_c_l197_197409


namespace functions_are_same_l197_197760

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_same_l197_197760


namespace zora_is_shorter_by_eight_l197_197531

noncomputable def zora_height (z : ℕ) (b : ℕ) (i : ℕ) (zara : ℕ) (average_height : ℕ) : Prop :=
  i = z + 4 ∧
  zara = b ∧
  average_height = 61 ∧
  (z + i + zara + b) / 4 = average_height

theorem zora_is_shorter_by_eight (Z B : ℕ)
  (h1 : zora_height Z B (Z + 4) 64 61) : (B - Z) = 8 :=
by
  sorry

end zora_is_shorter_by_eight_l197_197531


namespace is_even_function_l197_197529

def f (x : ℝ) : ℝ := 3^(x^2 - 3) - real.sqrt (x^2)

theorem is_even_function : ∀ x : ℝ, f (-x) = f x := by
  intros x
  unfold f
  sorry

end is_even_function_l197_197529


namespace sum_of_a_l197_197811

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.sin x) + 2 * Real.arccos (Real.cos x)

theorem sum_of_a :
  let F (a : ℝ) (x : ℝ) := (2 * Real.pi * a + Real.arcsin (Real.sin x) + 2 * Real.arccos (Real.cos x) - a * x) / (Real.tan x ^ 2 + 1)
  ∃ a₁ a₂ a₃ : ℝ,
  F a₁ x = 0 ∧ F a₂ x = 0 ∧ F a₃ x = 0 ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧
  set.card (set_of (λ x, F a₁ x = 0)) = 3 ∧
  set.card (set_of (λ x, F a₂ x = 0)) = 3 ∧
  set.card (set_of (λ x, F a₃ x = 0)) = 3 ∧
  set.sum ({a₁, a₂, a₃} : set ℝ) (λ a, a) = 1.6 := sorry

end sum_of_a_l197_197811


namespace min_distance_midpoint_to_x_axis_l197_197963

noncomputable def shortest_distance_to_x_axis_from_midpoint (x1 y1 x2 y2 : ℝ) : ℝ :=
  if h : (x1^2 = 4 * y1) ∧ (x2^2 = 4 * y2) ∧ (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 6) then
    let y_mid := (y1 + y2) / 2 in abs y_mid
  else 0

theorem min_distance_midpoint_to_x_axis (x1 y1 x2 y2 : ℝ)
  (h : (x1^2 = 4 * y1) ∧ (x2^2 = 4 * y2) ∧ (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 6)) :
  shortest_distance_to_x_axis_from_midpoint x1 y1 x2 y2 = 2 :=
sorry

end min_distance_midpoint_to_x_axis_l197_197963


namespace Jean_money_correct_l197_197947

-- Given conditions
variables (J : ℝ) -- Jane's money in dollars
variables (exchange_rate : ℝ := 1.18) -- exchange rate from euro to dollars
variables (Jack_money : ℝ := 120) -- Jack's money in dollars
variables (total_money : ℝ := 256) -- total combined money in dollars

-- Jean's money in Euros
noncomputable def Jean_money_euros := 3 * J / exchange_rate

-- Combine conditions to prove the required total is consistent with the given answer
theorem Jean_money_correct (h : J + 3 * J * exchange_rate + Jack_money = total_money) :
  Jean_money_euros J = 76.17 := 
begin
  sorry,
end

end Jean_money_correct_l197_197947


namespace necessary_but_not_sufficient_condition_l197_197514

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l197_197514


namespace num_of_permutations_l197_197278

namespace PermutationsProof

-- Definitions based on the conditions
def nums : List Nat := [1, 2, 3, 4, 5, 6]

-- Function to calculate the number of valid permutations
def count_valid_permutations (arr : List (List Nat)) : Nat :=
  let N1 := arr.head!.head!
  let N2 := arr.tail!.head!.max!
  let N3 := arr.tail!.tail!.head!.max!
  if N1 < N2 ∧ N2 < N3 then 1 else 0

-- Proof statement with the required conditions and question
theorem num_of_permutations : let perms := List.permutations nums in
  let valid_perms := perms.filter (λ perm, 
    count_valid_permutations [perm.take 1, perm.drop 1 |>.take 2, perm.drop 3] = 1) in
  valid_perms.length = 240 :=
by
  sorry

end num_of_permutations_l197_197278


namespace solve_triangle_l197_197927

theorem solve_triangle (a b : ℝ) (A B : ℝ) : ((A + B < π ∧ A > 0 ∧ B > 0 ∧ a > 0) ∨ (a > 0 ∧ b > 0 ∧ (π > A) ∧ (A > 0))) → ∃ c C, c > 0 ∧ (π > C) ∧ C > 0 :=
sorry

end solve_triangle_l197_197927


namespace angle_ACB_orthocenter_distance_l197_197621

theorem angle_ACB_orthocenter_distance (H C: Point) (A B: Point) (O: Point) (R: ℝ) 
    (h_orthocenter: is_orthocenter H A B C) 
    (h_circumcenter: is_circumcenter O A B C) 
    (h_radius: distance O A = R) 
    (h_distance: distance H C = R) 
    : (angle A C B = 60) ∨ (angle A C B = 120) :=
sorry

end angle_ACB_orthocenter_distance_l197_197621


namespace vasya_problem_l197_197679

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l197_197679


namespace correct_number_of_statements_l197_197247

def statement1 : Prop := (0:ℕ) ∈ set.univ ℕ 
def statement2 : Prop := ¬ (sqrt 2).is_rational
def statement3 : Prop := (∅ : set ℕ) ⊆ {0}
def statement4 : Prop := (0 ∉ (∅ : set ℕ))

def statement5 : Prop := let intersection := {p | ∃ (x : ℝ), p = (x, x + 3) ∧ p = (x, -2*x + 6)} in intersection = {(1, 4)}

def number_correct_statements : ℕ := 
if statement1 then 1 else 0 +
if statement2 then 1 else 0 +
if statement3 then 1 else 0 +
if statement4 then 1 else 0 +
if statement5 then 1 else 0

theorem correct_number_of_statements : number_correct_statements = 4 :=
by {
  sorry
}

end correct_number_of_statements_l197_197247


namespace trigonometric_identity_proof_l197_197602

theorem trigonometric_identity_proof :
  (sin 40 + sin 80) / (cos 40 + cos 80) = tan 60 := 
by
  sorry

end trigonometric_identity_proof_l197_197602


namespace kamal_chemistry_marks_l197_197953

-- Definitions of the marks
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def num_subjects : ℕ := 5

-- Statement to be proved
theorem kamal_chemistry_marks : ∃ (chemistry_marks : ℕ), 
  76 + 60 + 72 + 82 + chemistry_marks = 71 * 5 :=
by
sorry

end kamal_chemistry_marks_l197_197953


namespace problem_2goal_3_l197_197246

open EuclideanGeometry

-- Define the problem in Lean
variables {A B C M N : Point}
variable [noncomputable_instance] : CharZero ℝ

-- Given conditions
axiom angle_BAM_eq_angle_CAN : ∠BAM = ∠CAN
axiom angle_ABM_eq_angle_CBN : ∠ABM = ∠CBN
axiom product_condition : ∃ k : ℝ, AM * AN * BC = k ∧ BM * BN * CA = k ∧ CM * CN * AB = k

theorem problem_2goal_3 (ABC_non_equilateral : ¬ congruent (triangle A B C) (triangle A C B)) :
  (∃ k : ℝ, 3 * k = A.dist B * B.dist C * C.dist A) ∧
  (is_medicenter (midpoint M N) A B C) := 
by
  sorry

end problem_2goal_3_l197_197246


namespace shirts_per_minute_l197_197275

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (shirts_per_min : ℕ) 
  (h : total_shirts = 12 ∧ total_minutes = 6) :
  shirts_per_min = 2 :=
sorry

end shirts_per_minute_l197_197275


namespace necessary_but_not_sufficient_condition_l197_197515

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l197_197515


namespace sum_of_solutions_l197_197219

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l197_197219


namespace album_ways_10_l197_197749

noncomputable def total_album_ways : ℕ := 
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := 4
  let friends := 4
  ((total_albums.choose photo_albums) * (total_albums - photo_albums).choose stamp_albums) / friends

theorem album_ways_10 :
  total_album_ways = 10 := 
by sorry

end album_ways_10_l197_197749


namespace tan_sum_identity_l197_197827

theorem tan_sum_identity :
  tan (70 * (π / 180)) + tan (50 * (π / 180)) - sqrt 3 * tan (70 * (π / 180)) * tan (50 * (π / 180)) = - sqrt 3 :=
by sorry

end tan_sum_identity_l197_197827


namespace number_of_odd_divisors_lt_100_is_9_l197_197466

theorem number_of_odd_divisors_lt_100_is_9 :
  (finset.filter (λ n : ℕ, ∃ k : ℕ, n = k * k) (finset.range 100)).card = 9 :=
sorry

end number_of_odd_divisors_lt_100_is_9_l197_197466


namespace midpoint_on_ac_l197_197594

open EuclideanGeometry

variables (A B C D P Q M : Point)

-- Definitions based on the conditions
def cyclic_quadrilateral (A B C D : Point) : Prop := 
  ∃ ω : Circle, A ∈ ω ∧ B ∈ ω ∧ C ∈ ω ∧ D ∈ ω

def points_on_rays (A B P D Q : Point) : Prop :=
  (∃ k1 : ℝ, P = A + k1 • (B - A) ∧ k1 ≥ 0) ∧ 
  (∃ k2 : ℝ, Q = A + k2 • (D - A) ∧ k2 ≥ 0)

def lengths_equal (P Q C D : Point) : Prop :=
  dist A P = dist C D ∧ dist A Q = dist B C

-- The statement to prove
theorem midpoint_on_ac
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : points_on_rays A B P D Q)
  (h3 : lengths_equal P Q C D) :
  let M := midpoint P Q in
  collinear {A, C, M} := 
sorry

end midpoint_on_ac_l197_197594


namespace sum_series_l197_197320

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197320


namespace geometry_problem_l197_197959

theorem geometry_problem 
  (P A B Q : Type) [AffineSpace P A B Q]
  (M : Type) (midpoint_PA : A = midpoint P M) (midpoint_PB : B = midpoint P N)
  (PQ_eq_PB : PQ = PB) (PQ_parallel_AB : parallel PQ AB)
  (P_moves_perpendicular_to_AB : ∀ P, P ⊥ AB) :
  {MN_length_constant : constant_length MN}
  {perimeter_PAQ_changes : changes_perimeter (triangle P A Q)}
  {area_PAB_changes : changes_area (triangle P A B)}
  {area_trapezoid_ABNM_changes : changes_area (trapezoid A B N M)} :
  3 = (quantity_changes [
    MN_length_constant,
    perimeter_PAQ_changes,
    area_PAB_changes,
    area_trapezoid_ABNM_changes
  ]) :=
sorry

end geometry_problem_l197_197959


namespace find_k_l197_197634

variable (x y k : ℝ)

-- Definition: the line equations and the intersection condition
def line1_eq (x y k : ℝ) : Prop := 3 * x - 2 * y = k
def line2_eq (x y : ℝ) : Prop := x - 0.5 * y = 10
def intersect_at_x (x : ℝ) : Prop := x = -6

-- The theorem we need to prove
theorem find_k (h1 : line1_eq x y k)
               (h2 : line2_eq x y)
               (h3 : intersect_at_x x) :
               k = 46 :=
sorry

end find_k_l197_197634


namespace expected_voters_percentage_l197_197235

theorem expected_voters_percentage (total_voters : ℕ) (democrat_percentage republican_percentage : ℕ) 
  (democrat_voting_for_A republican_voting_for_A : ℚ):
  democrat_percentage = 60 →
  republican_percentage = 40 →
  democrat_voting_for_A = 65 →
  republican_voting_for_A = 20 →
  democrat_percentage + republican_percentage = 100 →
  total_voters > 0 →
  let democrat_voters := total_voters * democrat_percentage / 100 in
  let republican_voters := total_voters * republican_percentage / 100 in
  let democrat_voters_for_A := democrat_voters * democrat_voting_for_A / 100 in
  let republican_voters_for_A := republican_voters * republican_voting_for_A / 100 in
  democrat_voters + republican_voters = total_voters →
  let total_voters_for_A := democrat_voters_for_A + republican_voters_for_A in
  total_voters_for_A / total_voters * 100 = 47 :=
by 
  intros h1 h2 h3 h4 h5 h6;
  sorry

end expected_voters_percentage_l197_197235


namespace max_mu_value_l197_197898

noncomputable def maxValue (x y z : ℝ) : ℝ :=
  sqrt (2 * x + 1) + sqrt (3 * y + 4) + sqrt (5 * z + 6)

theorem max_mu_value (x y z : ℝ) (h : 2 * x + 3 * y + 5 * z = 29) : 
  maxValue x y z ≤ 2 * sqrt 30 := 
by 
  sorry

end max_mu_value_l197_197898


namespace arithmetic_sequence_8th_term_l197_197122

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l197_197122


namespace closest_integer_sqrt_l197_197783

-- Define the sequence of non-square numbers F_n
def non_square_sequence : ℕ → ℕ
| 0     := 2
| (n+1) := let k := non_square_sequence n + 1 in if ∃ m : ℕ, m * m = k then k + 1 else k

-- The main theorem to prove
theorem closest_integer_sqrt {n m : ℕ} (h : m^2 ≤ non_square_sequence n ∧ non_square_sequence n < (m+1)^2) : 
  m = Nat.floor (Real.sqrt n) := 
sorry

end closest_integer_sqrt_l197_197783


namespace cos_36_degree_l197_197624

theorem cos_36_degree :
  (2 * sin (real.pi / 10) = (real.sqrt 5 - 1) / 2) →
  cos (real.pi / 5) = (real.sqrt 5 + 1) / 4 :=
begin
  intro h,
  -- proof steps go here
  sorry -- placeholder for the proof
end

end cos_36_degree_l197_197624


namespace largest_prime_factor_expression_l197_197696

open Nat

def expression := 16^4 + 2 * 16^2 + 1 - 13^4

theorem largest_prime_factor_expression :
  largest_prime_factor expression = 71 := by sorry

end largest_prime_factor_expression_l197_197696


namespace number_of_consecutive_sum_sets_eq_18_l197_197472

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l197_197472


namespace wand_original_price_l197_197686

theorem wand_original_price :
  (∃ x : ℝ, (x / 8 = 4) ∧ x = 32) :=
begin
  use 32,
  split,
  { norm_num },
  { refl }
end

end wand_original_price_l197_197686


namespace unique_solution_l197_197822

theorem unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1 :=
by
  sorry

end unique_solution_l197_197822


namespace card_number_combinations_l197_197182

theorem card_number_combinations : 
  let card1 := {1, 2},
      card2 := {3, 4},
      card3 := {5, 6},
      cards := {card1, card2, card3}
  in (∀ (d1 d2 d3 : ℕ), d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3) →
     (∀ (d : ℕ), d ∈ card1 ∨ d ∈ card2 ∨ d ∈ card3) →
     (∀ (d : ℕ), d ≠ 9) →
     (∀ c1 c2 c3, c1 ∈ card1 → c2 ∈ card2 → c3 ∈ card3) →
     6 * 4 * 2 = 48 := 
by 
  intro card1 card2 card3 cards h1 h2 h3 h4
  sorry

end card_number_combinations_l197_197182


namespace radius_of_smallest_circle_l197_197376

-- Define the sides of the triangle
def a : ℕ := 7
def b : ℕ := 9
def c : ℕ := 12

-- Define the condition for the triangle being obtuse
def isObtuseTriangle (a b c : ℕ) : Prop := c^2 > a^2 + b^2

-- Define the function to calculate the radius of the smallest circle containing the triangle
def smallestCircumscribedRadius (a b c : ℕ) : ℕ := c / 2

-- Define the main theorem that combines these ideas
theorem radius_of_smallest_circle (a b c : ℕ) (h : isObtuseTriangle a b c) : smallestCircumscribedRadius a b c = 6 := by
  -- the proof is omitted
  sorry

-- Applying the conditions to our specific problem
example : radius_of_smallest_circle 7 9 12 (by { unfold isObtuseTriangle, norm_num, linarith }) = 6 := by
  -- the proof is omitted
  sorry

end radius_of_smallest_circle_l197_197376


namespace max_dot_product_l197_197440

-- Definitions of vectors OA, OB, OC with given conditions
variables {V : Type*} [inner_product_space ℝ V] {OA OB OC : V}
variables (h1 : ⟪OA, OB⟫ = 0) -- OA · OB = 0
variables (h2 : ∥OA∥ = 1) (h3 : ∥OC∥ = 1) -- |OA| = |OC| = 1
variables (h4 : ∥OB∥ = real.sqrt 3) -- |OB| = √3

-- Definition of CA and CB
noncomputable def CA : V := sorry
noncomputable def CB : V := sorry

-- Proposition: The maximum value of the dot product CA · CB
theorem max_dot_product : ∃ θ : ℝ, (⟪CA h2 h3 θ, CB h1 h2 h4 h3 h2 θ⟫) = 3 :=
  sorry

end max_dot_product_l197_197440


namespace sum_infinite_series_eq_l197_197290

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197290


namespace proj_3u_eq_l197_197019

variables {u z : ℝ × ℝ}
variable (proj_zu : ℝ × ℝ)
variable (proj_zu_eq : proj_zu = (4, 1))

noncomputable def proj (z : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * z.1 + v.2 * z.2 in
  let norm_sq := z.1 * z.1 + z.2 * z.2 in
  (dot / norm_sq) • z

theorem proj_3u_eq :
  proj z (3 • u) = (12, 3) :=
by
  have h₁ : proj_zu_u = proj z u := proj_zu_eq
  rw [← h₁, ← proj_zu_eq]
  sorry

end proj_3u_eq_l197_197019


namespace bertha_no_daughters_count_l197_197282

open Nat

-- Definitions for the conditions
def daughters : ℕ := 8
def total_women : ℕ := 42
def granddaughters : ℕ := total_women - daughters
def daughters_who_have_daughters := granddaughters / 6
def daughters_without_daughters := daughters - daughters_who_have_daughters
def total_without_daughters := granddaughters + daughters_without_daughters

-- The theorem to prove
theorem bertha_no_daughters_count : total_without_daughters = 37 := by
  sorry

end bertha_no_daughters_count_l197_197282


namespace number_of_ten_digit_palindromic_numbers_l197_197180

theorem number_of_ten_digit_palindromic_numbers : 
  let palindromic (n : ℕ) := ∀ d : ℕ, d < 10 → Nat.digits 10 n = List.reverse (Nat.digits 10 n) 
  let ten_digit (n : ℕ) := 10^9 ≤ n ∧ n < 10^10
  let valid_last_digit (n : ℕ) := ∀ d : ℕ, d < 10 → List.head (Nat.digits 10 n) ≠ 0
  ∃ num : ℕ, num = 9 * 10^4 := 
by
  sorry

end number_of_ten_digit_palindromic_numbers_l197_197180


namespace necessary_but_not_sufficient_condition_l197_197511

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (1 < k) ∧ (k < 5) → 
  (k - 1 > 0) ∧ (5 - k > 0) ∧ ((k ≠ 3) → (k < 5 ∧ 1 < k)) :=
by
  intro h
  have hk_gt_1 := h.1
  have hk_lt_5 := h.2
  refine ⟨hk_gt_1, hk_lt_5, λ hk_neq_3, ⟨hk_lt_5, hk_gt_1⟩⟩
  sorry

end necessary_but_not_sufficient_condition_l197_197511


namespace no_five_coin_combination_for_70_cents_l197_197377

/-- Define the values of each coin type -/
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

/-- Prove that it is not possible to achieve a total value of 70 cents with exactly five coins -/
theorem no_five_coin_combination_for_70_cents :
  ¬ ∃ a b c d e : ℕ, a + b + c + d + e = 5 ∧ a * penny + b * nickel + c * dime + d * quarter + e * quarter = 70 :=
sorry

end no_five_coin_combination_for_70_cents_l197_197377


namespace total_weight_full_l197_197226

variables (c d : ℝ)
def bucket_weight_third_full (c : ℝ) : ℝ := c
def bucket_weight_three_quarters_full (d : ℝ) : ℝ := d

theorem total_weight_full (c d : ℝ) :
  let x := (9 * c - 4 * d) / 5 in
  let y := (12 * (d - c)) / 5 in
  x + y = (8 * d - 3 * c) / 5 :=
by
  sorry

end total_weight_full_l197_197226


namespace arithmetic_sequence_8th_term_l197_197112

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197112


namespace find_multiple_l197_197173

-- Defining the conditions
variables (A B k : ℕ)

-- Given conditions
def sum_condition : Prop := A + B = 77
def bigger_number_condition : Prop := A = 42

-- Using the conditions and aiming to prove that k = 5
theorem find_multiple
  (h1 : sum_condition A B)
  (h2 : bigger_number_condition A) :
  6 * B = k * A → k = 5 :=
by
  sorry

end find_multiple_l197_197173


namespace total_travel_time_l197_197005

-- Define the given conditions
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 30
def distance_to_school : ℝ := 6.857142857142858

-- State the theorem to prove
theorem total_travel_time :
  (distance_to_school / speed_jogging) + (distance_to_school / speed_bus) = 1.6 :=
by
  sorry

end total_travel_time_l197_197005


namespace sum_of_consecutive_integers_l197_197470

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l197_197470


namespace simplify_expression_l197_197477

theorem simplify_expression (a b : ℝ) (h : a + b < 0) : 
  |a + b - 1| - |3 - (a + b)| = -2 :=
by 
  sorry

end simplify_expression_l197_197477


namespace number_of_consecutive_sum_sets_eq_18_l197_197473

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l197_197473


namespace div_1959_l197_197986

theorem div_1959 (n : ℕ) : ∃ k : ℤ, 5^(8 * n) - 2^(4 * n) * 7^(2 * n) = k * 1959 := 
by 
  sorry

end div_1959_l197_197986


namespace sum_fraction_series_l197_197300

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197300


namespace leon_total_payment_l197_197008

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end leon_total_payment_l197_197008


namespace minimum_value_ineq_l197_197547

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l197_197547


namespace abs_neg_three_l197_197126

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l197_197126


namespace solve_system_of_equations_l197_197606

theorem solve_system_of_equations :
  ∃ x y : ℚ, (4 * x - 6 * y = -3) ∧ (8 * x + 3 * y = 6) ∧ (x + y = 1.25) :=
by
  use 9/20, 8/10  -- providing the solutions for x and y
  split
  { -- proving the first equation holds
    norm_num
    exact rfl },
  split
  { -- proving the second equation holds
    norm_num
    exact rfl },
  { -- proving the sum of the solutions is correct
    norm_num
    exact rfl }

end solve_system_of_equations_l197_197606


namespace fraction_calculation_l197_197777

theorem fraction_calculation : (8 / 24) - (5 / 72) + (3 / 8) = 23 / 36 :=
by
  sorry

end fraction_calculation_l197_197777


namespace vasya_problem_l197_197677

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l197_197677


namespace smallest_possible_area_square_l197_197379

theorem smallest_possible_area_square : 
  ∃ (c : ℝ), (∀ (x y : ℝ), ((y = 3 * x - 20) ∨ (y = x^2)) ∧ 
      (10 * (9 + 4 * c) = ((c + 20) / Real.sqrt 10) ^ 2) ∧ 
      (c = 80) ∧ 
      (10 * (9 + 4 * c) = 3290)) :=
by {
  use 80,
  sorry
}

end smallest_possible_area_square_l197_197379


namespace part1_part2_l197_197353

-- Problem Statement and Conditions
namespace TriangleProblems

variables (a b c : ℝ)
def S (a b c : ℝ) : ℝ := (Math.sqrt 3 / 4) * (a^2 + c^2 - b^2)

-- Proof for part (Ⅰ)
theorem part1 (hS : S a b c = (1 / 2) * a * c * Real.sin (π / 3)) : 
  Real.arctan (Math.sqrt 3) = π / 3 := 
by sorry

-- Proof for part (Ⅱ)
theorem part2 (b_val : b = Math.sqrt 3) : 
  ∃ A : ℝ, 0 < A ∧ A < (2 * π / 3) ∧ 
  (2 * (Math.sqrt 3 - 1) * Real.sin A + 4 * Real.sin ((2 * π / 3) - A)) ≤ 2 * Math.sqrt 6 := 
by sorry

end TriangleProblems

end part1_part2_l197_197353


namespace ezekiel_faces_painted_l197_197810

theorem ezekiel_faces_painted (cuboids : ℕ) (faces_per_cuboid : ℕ) (painted_faces_total : ℕ) :
  cuboids = 5 →
  faces_per_cuboid = 6 →
  painted_faces_total = 30 ↔ painted_faces_total = cuboids * faces_per_cuboid :=
begin
  intros h_cuboids h_faces,
  split,
  { intro h_total,
    rw [h_total, h_cuboids, h_faces],
    exact dec_trivial, },
  { intro h_eq,
    rw [h_cuboids, h_faces] at h_eq,
    exact h_eq, },
end

end ezekiel_faces_painted_l197_197810


namespace shane_gum_left_l197_197807

def elyse_initial_gum : ℕ := 100
def half (x : ℕ) := x / 2
def rick_gum : ℕ := half elyse_initial_gum
def shane_initial_gum : ℕ := half rick_gum
def chewed_gum : ℕ := 11

theorem shane_gum_left : shane_initial_gum - chewed_gum = 14 := by
  sorry

end shane_gum_left_l197_197807


namespace water_added_for_alcohol_solution_water_added_for_sugar_water_solution_l197_197459

namespace Solution

-- Part 1: Alcohol Solution
theorem water_added_for_alcohol_solution
  (V₁ : ℕ) (C₁ : ℚ) (C₂ : ℚ) (H₁ : V₁ = 50) (H₂ : C₁ = 0.4) (H₃ : C₂ = 0.25) :
  ∃ w₁ : ℕ, w₁ = 30 :=
begin
  existsi 30,
  rw [H₁, H₂, H₃],
  sorry -- proving the calculation steps here
end

-- Part 2: Sugar Water Solution
theorem water_added_for_sugar_water_solution
  (V₂ : ℕ) (C₃ : ℚ) (C₄ : ℚ) (H₄ : V₂ = 60) (H₅ : C₃ = 0.3) (H₆ : C₄ = 0.15) :
  ∃ w₂ : ℕ, w₂ = 60 :=
begin
  existsi 60,
  rw [H₄, H₅, H₆],
  sorry -- proving the calculation steps here
end

end Solution

end water_added_for_alcohol_solution_water_added_for_sugar_water_solution_l197_197459


namespace alex_time_to_entrance_l197_197925

theorem alex_time_to_entrance :
  let rate := 80 / 20 in
  let conversion_factor := 3.28084 in
  let remaining_distance := 90 * conversion_factor in
  let time_needed := remaining_distance / rate in
  time_needed ≈ 73.819 :=
by 
  let rate := 80 / 20
  let conversion_factor := 3.28084
  let remaining_distance := 90 * conversion_factor
  let time_needed := remaining_distance / rate
  -- the real proof will go here, but we use sorry for now
  sorry

end alex_time_to_entrance_l197_197925


namespace arithmetic_sequence_8th_term_l197_197108

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l197_197108


namespace compositeShapeSum_is_42_l197_197262

-- Define the pentagonal prism's properties
structure PentagonalPrism where
  faces : ℕ := 7
  edges : ℕ := 15
  vertices : ℕ := 10

-- Define the pyramid addition effect
structure PyramidAddition where
  additional_faces : ℕ := 5
  additional_edges : ℕ := 5
  additional_vertices : ℕ := 1
  covered_faces : ℕ := 1

-- Definition of composite shape properties
def compositeShapeSum (prism : PentagonalPrism) (pyramid : PyramidAddition) : ℕ :=
  (prism.faces - pyramid.covered_faces + pyramid.additional_faces) +
  (prism.edges + pyramid.additional_edges) +
  (prism.vertices + pyramid.additional_vertices)

-- The theorem to be proved: that the total sum is 42
theorem compositeShapeSum_is_42 : compositeShapeSum ⟨7, 15, 10⟩ ⟨5, 5, 1, 1⟩ = 42 := by
  sorry

end compositeShapeSum_is_42_l197_197262


namespace sum_infinite_series_eq_l197_197294

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197294


namespace a2009_equals_7_l197_197443

def sequence_element (n k : ℕ) : ℚ :=
  if k = 0 then 0 else (n - k + 1) / k

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem a2009_equals_7 : 
  let n := 63
  let m := 2009
  let subset_cumulative_count := cumulative_count n
  (2 * m = n * (n + 1) - 14 ∧
   m = subset_cumulative_count - 7 ∧ 
   sequence_element n 8 = 7) →
  sequence_element n (subset_cumulative_count - m + 1) = 7 :=
by
  -- proof steps to be filled here
  sorry

end a2009_equals_7_l197_197443


namespace find_a6_plus_a8_l197_197419

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

constant a_n : ℕ → ℝ
constant b : ℝ

axiom h1 : ∀ x, f x / g x = b^x
axiom h2 : ∀ x, f' x * g x < f x * g' x
axiom h3 : f 1 / g 1 + f (-1) / g (-1) = 5 / 2
axiom h4 : a_n 5 * a_n 7 + 2 * a_n 6 * a_n 8 + a_n 4 * a_n 12 = f 4 / g 4
axiom h5 : ∀ n, a_n n > 0

theorem find_a6_plus_a8 : a_n 6 + a_n 8 = 1 / 4 :=
sorry

end find_a6_plus_a8_l197_197419


namespace range_of_a_l197_197880

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) := x^3 - x^2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (1/2) 2 → x2 ∈ Set.Icc (1/2) 2 → f x1 a - g x2 ≥ 0) →
  1 ≤ a :=
by
  sorry

end range_of_a_l197_197880


namespace prove_angles_l197_197045

axiom Triangle (A B C M X : Type) : Type

axiom on_median (A B C M X : Type) (t: Triangle A B C M X) : Prop
axiom segment_inequality (A B C M X : Type) (t: Triangle A B C M X) : (A < B) -> Prop
axiom compare_angles (A B C M X : Type) (t: Triangle A B C M X) : Prop

theorem prove_angles (A B C M X : Type) (t: Triangle A B C M X) 
(om: on_median A B C M X t) 
(si: segment_inequality A B C M X t (A < C)) 
: compare_angles A B C M X t := 
sorry

end prove_angles_l197_197045


namespace monotonic_intervals_proof_l197_197797

noncomputable def tangent_function_decreasing_intervals (x : ℝ) : Prop :=
  ∀ (y : ℝ), y = 3 * Real.tan (Real.pi / 4 - 2 * x) →
  (0 ≤ x ∧ x < 3 * Real.pi / 8 ∨ 3 * Real.pi / 8 < x ∧ x ≤ Real.pi / 2) →
  ∃ (a b : ℝ), (a < x ∧ x < b) → 
  a ∈ [0, 3 * Real.pi / 8) ∧ b ∈ (3 * Real.pi / 8, Real.pi / 2] ∧
  Real.tan (Real.pi / 4 - 2 * a) > Real.tan (Real.pi / 4 - 2 * x) ∧
  Real.tan (Real.pi / 4 - 2 * x) > Real.tan (Real.pi / 4 - 2 * b)

theorem monotonic_intervals_proof (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  tangent_function_decreasing_intervals x :=
begin
  sorry
end

end monotonic_intervals_proof_l197_197797


namespace correct_triangle_set_l197_197146

/-- Definition of triangle inequality -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Sets of lengths for checking the triangle inequality -/
def Set1 : ℝ × ℝ × ℝ := (5, 8, 2)
def Set2 : ℝ × ℝ × ℝ := (5, 8, 13)
def Set3 : ℝ × ℝ × ℝ := (5, 8, 5)
def Set4 : ℝ × ℝ × ℝ := (2, 7, 5)

/-- The correct set of lengths that can form a triangle according to the triangle inequality -/
theorem correct_triangle_set : satisfies_triangle_inequality 5 8 5 :=
by
  -- Proof would be here
  sorry

end correct_triangle_set_l197_197146


namespace solution_set_of_inequalities_l197_197857

theorem solution_set_of_inequalities (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : ∀ x, mx + n > 0 ↔ x < (1/3)) : ∀ x, nx - m < 0 ↔ x < -3 :=
by
  sorry

end solution_set_of_inequalities_l197_197857


namespace simplify_expression_l197_197716

-- Define the conditions in the problem
def eight_to_two_thirds : ℝ := 8^(2/3)
def log_25 : ℝ := log 25
def log_one_fourth : ℝ := log (1 / 4)

-- State the question as a theorem which claims the expression equals 6
theorem simplify_expression : eight_to_two_thirds + log_25 - log_one_fourth = 6 := sorry

end simplify_expression_l197_197716


namespace min_max_ab_bc_cd_de_l197_197967

theorem min_max_ab_bc_cd_de (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) (h_sum : a + b + c + d + e = 2018) : 
  ∃ a b c d e, 
  a > 0 ∧ 
  b > 0 ∧ 
  c > 0 ∧ 
  d > 0 ∧ 
  e > 0 ∧ 
  a + b + c + d + e = 2018 ∧ 
  ∀ M, M = max (max (max (a + b) (b + c)) (max (c + d) (d + e))) ↔ M = 673  :=
sorry

end min_max_ab_bc_cd_de_l197_197967


namespace arithmetic_seq_8th_term_l197_197073

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l197_197073


namespace number_composite_l197_197590

theorem number_composite (k : ℕ) (hk : k ≥ 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = ∑ i in Finset.range (k + 1), 100 ^ i) :=
by
  sorry

end number_composite_l197_197590


namespace simplify_expr_to_polynomial_l197_197230

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l197_197230


namespace find_m_if_f_is_odd_f_is_decreasing_l197_197015

def f (x : ℝ) (m : ℝ) := (2 / (2 ^ x + 1)) + m

theorem find_m_if_f_is_odd (h : ∀ x : ℝ, f x = -f (-x)) : m = -1 := by
  sorry

theorem f_is_decreasing (m : ℝ) (x1 x2 : ℝ) (hx : x1 < x2) : f x1 m > f x2 m := by
  sorry

end find_m_if_f_is_odd_f_is_decreasing_l197_197015


namespace problem1_l197_197248

theorem problem1 (a : ℝ) (x : ℝ) (h : x = (a^2 + 1) / (2 * a)) (ha : a > 0) :
  (let expr := (x + 1)^(-1/2) / ((x - 1)^(-1/2) - (x + 1)^(-1/2)) in
  if h1 : a < 1 then expr = (1 - a) / (2 * a)
  else if h2 : a > 1 then expr = (a - 1) / 2
  else false) := sorry

end problem1_l197_197248


namespace count_six_digit_numbers_divisible_by_12_l197_197747

def six_digit_numbers_divisible_by_12 : ℕ :=
  (finset.range 10).sum (λ a, 
    (finset.filter (λ b, 12 ∣ (100000 * a + 19880 + b))
                   (finset.range 10)).card)

theorem count_six_digit_numbers_divisible_by_12 :
  six_digit_numbers_divisible_by_12 = 9 :=
  sorry

end count_six_digit_numbers_divisible_by_12_l197_197747


namespace sum_after_50_rounds_l197_197502

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds_l197_197502


namespace impossible_to_construct_center_l197_197844

theorem impossible_to_construct_center (C : set (ℝ × ℝ)) (hC : (∃ O r, C = { P | dist O P = r })) :
  ¬(∃ O, C_center C O) :=
sorry

-- Additional definitions required for the theorem statement
def C_center (C : set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
∀ P ∈ C, dist O P = dist (classical.some (classical.some_spec hC).1) P

end impossible_to_construct_center_l197_197844


namespace min_operations_identify_buttons_l197_197423

theorem min_operations_identify_buttons :
  ∀ (buttons lights : ℕ), (buttons = 64) → (lights = 64) → (∀ b, b < 64 → ∃! k, k < 6 ∧ ∀ i, i < 6 → (b nbitOn i) = (k mod 2))
  → 6 :=
begin
  intros buttons lights H1 H2 H3,
  -- Conditions as premises
  have premise1 : buttons = 64 := H1,
  have premise2 : lights = 64 := H2,
  -- Define captured state and uniqueness
  have capture_unique : ∀ b, b < 64 → ∃! k, k < 6 ∧ ∀ i, i < 6 → (b nbitOn i) = (k mod 2),
  
  -- Proof steps would follow, simplified by sorry
  sorry,
end

end min_operations_identify_buttons_l197_197423


namespace consecutive_vertices_product_l197_197348

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l197_197348


namespace factorization_l197_197425

def f (a b c : ℝ) : ℝ := a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

def q (a b c : ℝ) : ℝ := a^3 * b^2 + a^2 * b^3 + b^3 * c^2 + b^2 * c^3 + c^3 * a^2 + c^2 * a^3

theorem factorization (a b c : ℝ) : f a b c = (a - b) * (b - c) * (c - a) * q a b c :=
by
  sorry

end factorization_l197_197425


namespace sum_infinite_series_eq_l197_197293

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197293


namespace chemical_equilibrium_l197_197923

theorem chemical_equilibrium (
  (SO₂_initial : ℝ) (O₂_initial : ℝ) (SO₃_initial : ℝ)
  (percent_conversion_SO₂ : ℝ) :
  2 * SO₂_initial + O₂_initial = 2 * SO₃_initial →
  percent_conversion_SO₂ = 0.8 →
  SO₂_initial = 0.4 →
  O₂_initial = 1 →
  SO₃_initial = 0 →
  let Δ_SO₂ := -SO₂_initial * percent_conversion_SO₂ in
  let Δ_O₂ := Δ_SO₂ / 2 in
  let Δ_SO₃ := -Δ_SO₂ in
  [SO₂_eq] = SO₂_initial + Δ_SO₂ →
  [O₂_eq] = O₂_initial + Δ_O₂ →
  [SO₃_eq] = SO₃_initial + Δ_SO₃ →
  [SO₂_eq] = 0.08 ∧ [O₂_eq] = 0.84 ∧ [SO₃_eq] = 0.32
:= sorry

end chemical_equilibrium_l197_197923


namespace test_question_count_l197_197175

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end test_question_count_l197_197175


namespace arithmetic_sequence_8th_term_is_71_l197_197098

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197098


namespace least_faces_combined_l197_197193

noncomputable def num_faces_dice_combined : ℕ :=
  let a := 11
  let b := 7
  a + b

/-- Given the conditions on the dice setups for sums of 8, 11, and 15,
the least number of faces on the two dice combined is 18. -/
theorem least_faces_combined (a b : ℕ) (h1 : 6 < a) (h2 : 6 < b)
  (h_sum_8 : ∃ (p : ℕ), p = 7)  -- 7 ways to roll a sum of 8
  (h_sum_11 : ∃ (q : ℕ), q = 14)  -- half probability means 14 ways to roll a sum of 11
  (h_sum_15 : ∃ (r : ℕ), r = 2) : a + b = 18 :=
by
  sorry

end least_faces_combined_l197_197193


namespace max_area_triangle_l197_197244

-- Definition of the ellipse and the chord passing through its center
def ellipse_eq (x y a b : ℝ) := (x^2)/(a^2) + (y^2)/(b^2) = 1

-- Definition of the maximum area of triangle FAB
theorem max_area_triangle (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ A B F : (ℝ × ℝ), 
    (∃ x₀ y₀ : ℝ, A = (x₀, y₀) ∧ B = (-x₀, -y₀) ∧ y₀ ∈ Icc (-b) b ∧ ellipse_eq x₀ y₀ a b) ∧ 
    F = (c, 0) ∧ 
    let S := c * b in
    S = b * c :=
by
  sorry

end max_area_triangle_l197_197244


namespace sum_of_series_l197_197334

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197334


namespace lines_concurrent_l197_197001

noncomputable def triangle_concurrent (A B C D E F P X Y Z : Point)
  (h_tangent_D : incircle_tangent D ⟶ BC)
  (h_tangent_E : incircle_tangent E ⟶ CA)
  (h_tangent_F : incircle_tangent F ⟶ AB)
  (h_inter_AD_BE : ∃ P, AD ∩ BE = P)
  (h_reflection_X : reflection P EF = X)
  (h_reflection_Y : reflection P FD = Y)
  (h_reflection_Z : reflection P DE = Z) : Prop := 
concurrent AX BY CZ

theorem lines_concurrent (A B C D E F P X Y Z : Point)
  (h_tangent_D : incircle_tangent D ⟶ BC)
  (h_tangent_E : incircle_tangent E ⟶ CA)
  (h_tangent_F : incircle_tangent F ⟶ AB)
  (h_inter_AD_BE : ∃ P, AD ∩ BE = P)
  (h_reflection_X : reflection P EF = X)
  (h_reflection_Y : reflection P FD = Y)
  (h_reflection_Z : reflection P DE = Z) : 
  triangle_concurrent A B C D E F P X Y Z h_tangent_D h_tangent_E h_tangent_F h_inter_AD_BE h_reflection_X h_reflection_Y h_reflection_Z :=
sorry

end lines_concurrent_l197_197001


namespace probability_one_marble_each_color_l197_197724

theorem probability_one_marble_each_color :
  let total_marbles := 9
  let total_ways := Nat.choose total_marbles 3
  let favorable_ways := 3 * 3 * 3
  let probability := favorable_ways / total_ways
  probability = 9 / 28 :=
by
  sorry

end probability_one_marble_each_color_l197_197724


namespace cyclic_sum_equality_l197_197586

theorem cyclic_sum_equality (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i in finset.range n, a i / (a ((i + 1) % n) * (a i + a ((i + 1) % n)))) =
  (∑ i in finset.range n, a ((i + 1) % n) / (a i * (a ((i + 1) % n) + a i))) :=
sorry

end cyclic_sum_equality_l197_197586


namespace polynomial_divisible_by_x_minus_two_l197_197791

-- Define the polynomial g(x)
def g (x : ℝ) (n : ℝ) : ℝ := 3 * x^2 + 5 * x + n

-- Statement of the theorem
theorem polynomial_divisible_by_x_minus_two (n : ℝ) :
  (∀ x, g x n = g(x) → g 2 n = 0) → n = -22 :=
by
  sorry

end polynomial_divisible_by_x_minus_two_l197_197791


namespace odd_divisors_perfect_squares_below_100_l197_197467

theorem odd_divisors_perfect_squares_below_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k * k)}.card = 9 :=
by
  sorry

end odd_divisors_perfect_squares_below_100_l197_197467


namespace rectangular_field_area_l197_197135

theorem rectangular_field_area (L B : ℝ) (h1 : B = 0.6 * L) (h2 : 2 * L + 2 * B = 800) : L * B = 37500 :=
by
  -- Proof will go here
  sorry

end rectangular_field_area_l197_197135


namespace sum_of_series_l197_197333

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197333


namespace exists_n_no_prime_less_2008_l197_197054

theorem exists_n_no_prime_less_2008 :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℤ, ∀ p : ℕ, Nat.Prime p → p < 2008 → ¬ p ∣ (k^2 + k + n) :=
sorry

end exists_n_no_prime_less_2008_l197_197054


namespace no_pos_int_sequence_exists_pos_irr_seq_l197_197245

-- Part (1)
theorem no_pos_int_sequence 
  (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f(n + 1) ^ 2 ≥ 2 * f(n) * f(n + 2)) : 
  ∃ N, f(N) <(1 : ℕ) := sorry

-- Part (2)
theorem exists_pos_irr_seq 
  (f : ℕ → ℝ) 
  (h : (∀ n : ℕ, f(n + 1) ^ 2 ≥ 2 * f(n) * f(n + 2))
  ∧ (∀ n : ℕ, f(n) > 0)) : 
  ∃ g : ℕ → ℝ, (∀ n : ℕ, g(n + 1) ^ 2 ≥ 2 * g(n) * g(n + 2) ∧ (irrational (g(n)))) := sorry

end no_pos_int_sequence_exists_pos_irr_seq_l197_197245


namespace pieces_of_wood_for_table_l197_197385

theorem pieces_of_wood_for_table :
  ∀ (T : ℕ), (24 * T + 48 * 8 = 672) → T = 12 :=
by
  intro T
  intro h
  sorry

end pieces_of_wood_for_table_l197_197385


namespace proof_m_n_sum_l197_197658

-- Definitions based on conditions
def m : ℕ := 2
def n : ℕ := 49

-- Problem statement as a Lean theorem
theorem proof_m_n_sum : m + n = 51 :=
by
  -- This is where the detailed proof would go. Using sorry to skip the proof.
  sorry

end proof_m_n_sum_l197_197658


namespace angle_A_and_bc_range_l197_197928

variables (a b c A B C : ℝ) (h1 : a = Real.sqrt 2) (acute_ABC : A + B + C = π ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h2 : (b^2 - a^2 - c^2)/(a*c) = Real.cos(A + C) / (Real.sin A * Real.cos A))

theorem angle_A_and_bc_range (h : (b^2 - a^2 - c^2)/(a*c) = Real.cos(A + C) / (Real.sin A * Real.cos A)) :
  A = π / 4 ∧ 2*Real.sin B*2*Real.sin C ∈ (2*Real.sqrt 2, 2 + Real.sqrt 2] :=
sorry

end angle_A_and_bc_range_l197_197928


namespace sum_fraction_series_l197_197296

open scoped BigOperators

-- Define the infinite sum
noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in Set.univ, if (n = 0) then 0 else (3 * (n : ℝ) - 2) / ((n : ℝ) * ((n + 1) : ℝ) * ((n + 3) : ℝ))

-- The theorem stating the sum
theorem sum_fraction_series : series_sum = -7 / 24 := 
  sorry

end sum_fraction_series_l197_197296


namespace distance_increase_formula_l197_197657

noncomputable def increase_distance
  (m : ℝ) (Π : ℝ) (μ : ℝ) (g : ℝ) : ℝ :=
  Π / (μ * m * g)

theorem distance_increase_formula
  (m Π μ g : ℝ) (h1 : m > 0) (h2 : Π > 0) (h3 : μ > 0) (h4 : g > 0) :
  increase_distance m Π μ g = Π / (μ * m * g) :=
by
  -- Add the proof here
  sorry

end distance_increase_formula_l197_197657


namespace lukas_averages_points_l197_197975

theorem lukas_averages_points (total_points : ℕ) (num_games : ℕ) (average_points : ℕ)
  (h_total: total_points = 60) (h_games: num_games = 5) : average_points = total_points / num_games :=
sorry

end lukas_averages_points_l197_197975


namespace cubic_roots_identity_l197_197023

theorem cubic_roots_identity :
  ∀ (d e f : ℝ), (Polynomial.root (Polynomial.C 5 - Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - 3 * Polynomial.X) d) ∧
                 (Polynomial.root (Polynomial.C 5 - Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - 3 * Polynomial.X) e) ∧
                 (Polynomial.root (Polynomial.C 5 - Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - 3 * Polynomial.X) f) →
  (d - 1) * (e - 1) * (f - 1) = 3 :=
by
  intros d e f h
  sorry

end cubic_roots_identity_l197_197023


namespace solution_set_of_x_squared_gt_x_l197_197169

theorem solution_set_of_x_squared_gt_x :
  { x : ℝ | x^2 > x } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end solution_set_of_x_squared_gt_x_l197_197169


namespace vasya_problem_l197_197678

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l197_197678


namespace sets_equal_l197_197445

theorem sets_equal (M N : Set ℝ) (hM : M = { x | x^2 = 1 }) (hN : N = { a | ∀ x ∈ M, a * x = 1 }) : M = N :=
sorry

end sets_equal_l197_197445


namespace arithmetic_sequence_eighth_term_l197_197089

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197089


namespace find_intersecting_lines_l197_197818

theorem find_intersecting_lines (x y : ℝ) : 
  (2 * x - y)^2 - (x + 3 * y)^2 = 0 ↔ x = 4 * y ∨ x = - (2 / 3) * y :=
by
  sorry

end find_intersecting_lines_l197_197818


namespace quadrilateral_diagonals_identity_l197_197970

theorem quadrilateral_diagonals_identity
  (a b c d m n : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hm : 0 < m) (hn : 0 < n)
  (H : ∃ A B C D : ℝ, -- existence of quadrilateral with given side lengths
         (A - B = a) ∧ (B - C = b) ∧ (C - D = c) ∧ (D - A = d)
         ∧ (m = (A - C)) ∧ (n = (B - D))) :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos(A + C) :=
sorry

end quadrilateral_diagonals_identity_l197_197970


namespace tangents_parallel_l197_197592

noncomputable theory
open_locale classical

variables {C1 C2 : Type} [circle C1] [circle C2]
variables {K A B : C1} {A' B' : C2}

theorem tangents_parallel (h1 : tangent_at C1 C2 K)
                          (h2 : on_circle A C1)
                          (h3 : on_circle B C1)
                          (h4 : corresponding_point K A A' C1 C2)
                          (h5 : corresponding_point K B B' C1 C2)
                          (h6 : homothety K C1 C2) :
  parallel (tangent A C1) (tangent A' C2) ∧
  parallel (tangent B C1) (tangent B' C2) :=
sorry

end tangents_parallel_l197_197592


namespace largest_c_range_3_l197_197370

theorem largest_c_range_3 (c : ℝ) : (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ (c ≤ 61 / 4) :=
begin
  sorry
end

end largest_c_range_3_l197_197370


namespace find_100k_l197_197060

/-
Definitions:
- ABCD is a square with side length 4.
- S is the set of all line segments of length 3 with endpoints on adjacent sides of the square.
- k is the area of the region enclosed by the midpoints of the segments in S.

Statement to prove:
- 100k = 893
-/
theorem find_100k (A B C D : Point)
  (side_length : ℝ)
  (h_square : square A B C D side_length)
  (segment_length : ℝ)
  (h_length : segment_length = 3)
  (S : set (LineSegment))
  (h_S : ∀ s ∈ S, (length s = segment_length ∧ s.endpoints ⊂ {side A B, side B C, side C D, side D A}))
  (k : ℝ)
  (h_k : k = enclosed_area_by_midpoints S) :
  100 * k = 893 := 
sorry

end find_100k_l197_197060


namespace solve_fractional_equation_l197_197604

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2) : 
  (4 * x ^ 2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 := by 
  sorry

end solve_fractional_equation_l197_197604


namespace at_least_two_circles_equal_given_inscribed_circle_l197_197945

theorem at_least_two_circles_equal_given_inscribed_circle 
  (A B C D : Type) [algebra : algebraic_geom] 
  (circles : A → radius) (adjacent: (A → B) → Prop) (externally: (A → B) → Prop)
  (inscribed_circle: existInscribed Circle: B)
  (inside_quadrilateral: quadrilateral: Q): 
  ∃ (x y: radius), x = y ∧ (adjacent x y) ∧ (externally x y) :=
sorry

end at_least_two_circles_equal_given_inscribed_circle_l197_197945


namespace largest_even_among_consecutives_l197_197646

theorem largest_even_among_consecutives (x : ℤ) (h : (x + (x + 2) + (x + 4) = x + 18)) : x + 4 = 10 :=
by
  sorry

end largest_even_among_consecutives_l197_197646


namespace abs_val_neg_three_l197_197130

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l197_197130


namespace not_dynamic_3470_dynamic_1530_dynamic_number_algebraic_sum_original_swapped_multiple_of_3_l197_197055

-- Definition: a four-digit number where each digit is non-zero
def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∀ d, d ∈ (n.digits 10) → d ≠ 0

-- Definition: a "dynamic number"
def is_dynamic_number (n : ℕ) : Prop :=
  ∃ t h te u, n = 1000 * t + 100 * h + 10 * te + u ∧
  10 * t + h = a ∧ 10 * te + u = 2 * a

-- Prove that 3470 is not a "dynamic number"
theorem not_dynamic_3470 : ¬ is_dynamic_number 3470 :=
by
  sorry

-- Example: Prove that 1530 is a "dynamic number"
theorem dynamic_1530 : is_dynamic_number 1530 :=
by
  sorry

-- Algebraic expression: Prove a "dynamic number" can be represented as 102a
theorem dynamic_number_algebraic (a : ℕ) : is_dynamic_number (102 * a) :=
by
  sorry

-- Sum of the original and swapped: Prove that the sum is a multiple of 3
theorem sum_original_swapped_multiple_of_3 (a : ℕ) :
  (102 * a) + (200 * a + a) % 3 = 0 :=
by
  sorry

end not_dynamic_3470_dynamic_1530_dynamic_number_algebraic_sum_original_swapped_multiple_of_3_l197_197055


namespace sum_of_series_l197_197328

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197328


namespace arithmetic_sequence_8th_term_is_71_l197_197101

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l197_197101


namespace find_y_given_conditions_l197_197611

theorem find_y_given_conditions (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 5 * t + 9) (h3 : x = 0) : y = 33 / 2 := by
  sorry

end find_y_given_conditions_l197_197611


namespace partial_fractions_sum_zero_l197_197774

theorem partial_fractions_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, 
     x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 →
     1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4)) →
  A + B + C + D + E = 0 :=
by
  intros h
  sorry

end partial_fractions_sum_zero_l197_197774


namespace sum_of_youngest_and_oldest_nephews_l197_197636

theorem sum_of_youngest_and_oldest_nephews 
    (n1 n2 n3 n4 n5 n6 : ℕ) 
    (mean_eq : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 10) 
    (median_eq : (n3 + n4) / 2 = 12) : 
    n1 + n6 = 12 := 
by 
    sorry

end sum_of_youngest_and_oldest_nephews_l197_197636


namespace line_is_tangent_l197_197415

-- Define the problem conditions
def diameter_of_circle (O : Type) [metric_space O] (c : ℝ) := c = 10
def shortest_distance_to_center (P O : Type) [metric_space O] (d : ℝ) := d = 5

-- Define the positional relationship
def is_tangent_to_circle (l O : Type) [metric_space O] := l.dist O = 5

-- The main theorem stating the positional relationship between line l and circle O
theorem line_is_tangent {O P : Type} [metric_space O]
  (h1 : diameter_of_circle O 10)
  (h2 : shortest_distance_to_center P O 5) :
  is_tangent_to_circle P O :=
sorry

end line_is_tangent_l197_197415


namespace one_half_percent_as_decimal_l197_197895

theorem one_half_percent_as_decimal :
  ∃ d : ℝ, d = (1/2) / 100 ∧ d = 0.005 :=
by
  have h : (1/2:ℝ) / 100 = 0.005,
  { norm_num },
  use 0.005,
  exact ⟨h, rfl⟩

end one_half_percent_as_decimal_l197_197895


namespace integer_values_less_than_4pi_l197_197457

theorem integer_values_less_than_4pi : 
  {x : ℤ | abs x < 4 * Real.pi}.card = 25 := 
by
  sorry

end integer_values_less_than_4pi_l197_197457


namespace parking_space_unpainted_side_l197_197264

theorem parking_space_unpainted_side 
  (L W : ℝ) 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 125) : 
  L = 8.90 := 
by 
  sorry

end parking_space_unpainted_side_l197_197264


namespace arithmetic_seq_middle_term_l197_197204

theorem arithmetic_seq_middle_term :
  let a := (2:ℕ) ^ 3 in
  let b := (2:ℕ) ^ 5 in
  let z := (a + b) / 2 in
  z = 20 :=
by
  let a := (2:ℕ) ^ 3
  let b := (2:ℕ) ^ 5
  let z := (a + b) / 2
  show z = 20
  sorry

end arithmetic_seq_middle_term_l197_197204


namespace min_value_fx_l197_197478

theorem min_value_fx : ∀ x : ℝ, x > 0.5 → (∃ c : ℝ, (∀ y : ℝ, y > 0.5 → f y ≥ c) ∧ c = 2.5) :=
by
  let f : ℝ → ℝ := λ x, x + (2 / (2 * x - 1))
  sorry

end min_value_fx_l197_197478


namespace sum_series_l197_197321

theorem sum_series : 
  (∑ n in (Finset.range ∞).filter (λ n, n > 0), (3 * n - 2) / (n * (n + 1) * (n + 3))) = 31 / 24 := by
  sorry

end sum_series_l197_197321


namespace cos_identity_l197_197852

theorem cos_identity (θ : ℝ) (hcos : cos θ = 1/3) (hθ : θ ∈ Ioo (-π) (0)) : 
  cos (π / 2 + θ) = 2 * real.sqrt 2 / 3 :=
by
  sorry

end cos_identity_l197_197852


namespace prob_factors_less_than_seven_l197_197209

theorem prob_factors_less_than_seven (n : ℕ) (h₁ : n = 60) :
  let factors := {d ∈ finset.range (n + 1) | n % d = 0}
  let prob := (factors.filter (< 7)).card.toRat / factors.card.toRat
  prob = 1 / 2 :=
by
  sorry

end prob_factors_less_than_seven_l197_197209


namespace sqrt_40_between_6_and_7_l197_197808

theorem sqrt_40_between_6_and_7 : (6 : ℤ) < real.sqrt 40 ∧ real.sqrt 40 < 7 := by
  have h1 : real.sqrt 36 < real.sqrt 40 := real.sqrt_lt real.rpow_pos_of_pos (by norm_num) (by norm_num : (36 : ℝ) < 40)
  have h2 : real.sqrt 40 < real.sqrt 49 := real.sqrt_lt real.sqrt_rpow_pos_of_pos (by norm_num) (by norm_num : (40 : ℝ) < 49)
  exact ⟨by linarith, by linarith⟩

end sqrt_40_between_6_and_7_l197_197808


namespace trigonometric_identity_l197_197504

theorem trigonometric_identity (α : ℝ) : 
  cos(α) ^ 2 + cos(α + 60 * real.pi / 180) ^ 2 - cos(α) * cos(α + 60 * real.pi / 180) = 3 / 4 := 
by 
  sorry

end trigonometric_identity_l197_197504


namespace tan_cot_sixth_power_l197_197475

section
variables {θ a b : Real}
hypothesis h : (tan θ) ^ 2 / a + (cot θ) ^ 2 / b = 1 / (2 * (a + b))

theorem tan_cot_sixth_power :
  (tan θ) ^ 6 / (a * a) + (cot θ) ^ 6 / (b * b) = (1 + 64 * b ^ 5 * a ^ 2) / (8 * b ^ 3 * a ^ 2) :=
by
  sorry
end

end tan_cot_sixth_power_l197_197475


namespace triangle_area_correct_l197_197542

def vector_a : ℝ × ℝ := (4, -3)
def vector_b : ℝ × ℝ := (-6, 5)
def vector_c : ℝ × ℝ := (2 * -6, 2 * 5)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * |a.1 * c.2 - a.2 * c.1|

theorem triangle_area_correct :
  area_of_triangle (4, -3) (0, 0) (-12, 10) = 2 := by
  sorry

end triangle_area_correct_l197_197542


namespace find_fourth_number_l197_197906

theorem find_fourth_number (x y : ℝ) (h1 : 0.25 / x = 2 / y) (h2 : x = 0.75) : y = 6 :=
by
  sorry

end find_fourth_number_l197_197906


namespace sequence_general_term_l197_197401

-- Define the function f and integral
noncomputable def f : ℝ → ℝ := λ x, ∫ t in 1..x, (2 * t + 1)

-- Define the sequence S_n
def S (n : ℕ) : ℝ := n^2 + n - 2

-- Define the sequence a_n and the equality to be proven
def a (n : ℕ) : ℝ :=
  if n = 1 then 0 else 2 * n

-- Define the theorem that proves the general formula
theorem sequence_general_term (n : ℕ) : a n = 
  if n = 1 then 0 else 2*n := 
sorry

end sequence_general_term_l197_197401


namespace polynomial_divisible_by_x_l197_197971

noncomputable def polynomial := { f : ℤ → ℝ // ∃ n : ℤ, f n ≠ 0 }

theorem polynomial_divisible_by_x
  (f : polynomial)
  (h_non_constant : ∃ n : ℤ, f.1 n ≠ 0)
  (h_divisibility : ∀ (n k : ℤ), k > 0 →
    (f.1 (n + 1) * f.1 (n + 2) * ... * f.1 (n + k)) / (f.1 1 * f.1 2 * ... * f.1 k) ∈ ℤ) :
  f.1 0 = 0 := 
  sorry

end polynomial_divisible_by_x_l197_197971


namespace exists_fi_l197_197846

theorem exists_fi (f : ℝ → ℝ) (h_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧ 
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧ 
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧ 
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧ 
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
by
  sorry

end exists_fi_l197_197846


namespace series_sum_l197_197341

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197341


namespace total_cost_correct_l197_197948

def cost_of_cat_toy := 10.22
def cost_of_cage := 11.73
def cost_of_cat_food := 7.50
def cost_of_leash := 5.15
def cost_of_cat_treats := 3.98

theorem total_cost_correct : 
  cost_of_cat_toy + cost_of_cage + cost_of_cat_food + cost_of_leash + cost_of_cat_treats = 38.58 := 
by
  sorry

end total_cost_correct_l197_197948


namespace arithmetic_seq_8th_term_l197_197067

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197067


namespace system_exactly_two_solutions_l197_197814

theorem system_exactly_two_solutions (a : ℝ) : 
  (∃ x y : ℝ, |y + x + 8| + |y - x + 8| = 16 ∧ (|x| - 15)^2 + (|y| - 8)^2 = a) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, |y₁ + x₁ + 8| + |y₁ - x₁ + 8| = 16 ∧ (|x₁| - 15)^2 + (|y₁| - 8)^2 = a → 
                      |y₂ + x₂ + 8| + |y₂ - x₂ + 8| = 16 ∧ (|x₂| - 15)^2 + (|y₂| - 8)^2 = a → 
                      x₁ = x₂ ∧ y₁ = y₂) → 
  (a = 49 ∨ a = 289) :=
sorry

end system_exactly_two_solutions_l197_197814


namespace num_even_perfect_square_factors_of_2_6_5_3_7_8_l197_197451

def num_even_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 2^6 * 5^3 * 7^8 then
    let valid_a := [2, 4, 6]
    let valid_c := [0, 2]
    let valid_b := [0, 2, 4, 6, 8]
    valid_a.length * valid_c.length * valid_b.length
  else 0

theorem num_even_perfect_square_factors_of_2_6_5_3_7_8 :
  num_even_perfect_square_factors (2^6 * 5^3 * 7^8) = 30 :=
by
  sorry

end num_even_perfect_square_factors_of_2_6_5_3_7_8_l197_197451


namespace perfect_squares_less_than_100_l197_197462

theorem perfect_squares_less_than_100 :
  {n : ℕ | n < 100 ∧ (∃ k : ℕ, n = k^2)}.card = 9 :=
begin
  sorry
end

end perfect_squares_less_than_100_l197_197462


namespace range_of_f_l197_197162

noncomputable def f : ℝ → ℝ := sorry -- Define f appropriately

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} :=
sorry

end range_of_f_l197_197162


namespace find_erased_number_l197_197042

noncomputable def average_remaining (n : ℕ) (erased : ℕ) : ℚ :=
  (∑ i in finset.range n, (i + 1) - erased : ℚ) / (n - 1)

theorem find_erased_number (n : ℕ) (h1 : n = 71)
  (h2 : average_remaining n 36 + 2 / 5) : erased = 8 := sorry

end find_erased_number_l197_197042


namespace koschei_never_escapes_l197_197660

-- Define a structure for the initial setup
structure Setup where
  koschei_initial_room : Nat -- Initial room of Koschei
  guard_positions : List (Bool) -- Guards' positions, True for West, False for East

-- Example of the required setup:
def initial_setup : Setup :=
  { koschei_initial_room := 1, guard_positions := [true, false, true] }

-- Function to simulate the movement of guards
def move_guards (guards : List Bool) (room : Nat) : List Bool :=
  guards.map (λ g => not g)

-- Function to check if all guards are on the same wall
def all_guards_same_wall (guards : List Bool) : Bool :=
  List.all guards id ∨ List.all guards (λ g => ¬g)

-- Main statement: 
theorem koschei_never_escapes (setup : Setup) :
  ∀ room : Nat, ¬(all_guards_same_wall (move_guards setup.guard_positions room)) :=
  sorry

end koschei_never_escapes_l197_197660


namespace inequality_for_positive_integers_l197_197052

theorem inequality_for_positive_integers (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b + b * c + a * c ≤ 3 * a * b * c :=
sorry

end inequality_for_positive_integers_l197_197052


namespace bake_sale_cookies_l197_197280

theorem bake_sale_cookies (R O C : ℕ) (H1 : R = 42) (H2 : R = 6 * O) (H3 : R = 2 * C) : R + O + C = 70 := by
  sorry

end bake_sale_cookies_l197_197280


namespace operation_T_reduction_B9_result_l197_197395

/-- A Γ sequence is a finite sequence of real numbers within the interval (-1, 1). -/
def Γ_seq (B : List ℝ) : Prop :=
  ∀ x ∈ B, -1 < x ∧ x < 1

/-- The operation T on a Γ sequence, which combines two terms into one as defined. -/
def T (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

/-- Define a function performing the operation T on a sequence B, removing two elements and adding the resultant. -/
def T_on_seq (B : List ℝ) : List ℝ :=
  match B with
  | [] => []
  | [a] => [a]
  | a :: b :: rest => T a b :: rest

/-- Prove that any Γ sequence of length n can eventually be reduced to one term through n-1 operations of T. -/
theorem operation_T_reduction (B : List ℝ) (hB : Γ_seq B) (n : ℕ) (h_length : B.length = n) (h_n : 2 ≤ n) :
  ∃ x : ℝ, Γ_seq [x] ∧ (iterate T_on_seq (n-1) B) = [x] :=
sorry

/-- Given sequence and its final term after 9 operations -/
theorem B9_result :
  let B := [-5/7, -1/6, -1/5, -1/4, 5/6, 1/2, 1/3, 1/4, 1/5, 1/6] in
  Γ_seq B → (iterate T_on_seq 9 B) = [5/6] :=
sorry

end operation_T_reduction_B9_result_l197_197395


namespace sum_of_divisors_of_24_l197_197185

theorem sum_of_divisors_of_24 : 
  ∑ (d ∈ {d : ℕ | d ∣ 24}) d = 60 := 
by 
{
  -- Define the specific divisors based on prime factorization.
  have h : {d : ℕ | d ∣ 24} = {1, 2, 3, 4, 6, 8, 12, 24}, sorry,
  -- Apply the sum addition for positive divisors.
  rw h,
  -- Sum of the divisors.
  have hsum : ∑ (d ∈ {1, 2, 3, 4, 6, 8, 12, 24}) d = 60, from sorry,
  exact hsum,
}

end sum_of_divisors_of_24_l197_197185


namespace equilateral_triangle_angles_l197_197525

theorem equilateral_triangle_angles
  (A B C X Y : Type)
  [linear_ordered_field A]
  [linear_ordered_field B]
  [linear_ordered_field C]
  (h1 : ∠ ABX = ∠ YAC)
  (h2 : ∠ AYB = ∠ BXC)
  (h3 : XC = YB)
  : ∠ BAC = 60 ∧ ∠ ABC = 60 ∧ ∠ ACB = 60 :=
sorry

end equilateral_triangle_angles_l197_197525


namespace arithmetic_sequence_eighth_term_l197_197092

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197092


namespace cubic_polynomial_real_roots_l197_197555

-- Definitions from conditions
variables (P Q R : Polynomial ℝ)

-- Given conditions in the problem
def is_quadratic (P : Polynomial ℝ) : Prop :=
  degree P = 2

def is_cubic (Q : Polynomial ℝ) : Prop :=
  degree Q = 3

-- Main theorem statement
theorem cubic_polynomial_real_roots (P Q R : Polynomial ℝ)
  (hPq : is_quadratic P)
  (hQc : is_cubic Q)
  (hRc : is_cubic R)
  (hRel : P^2 + Q^2 = R^2) : 
  ∃ (Q_1 : Polynomial ℝ), is_cubic Q_1 ∧ ∀ root_val ∈ (Q_1.roots : set ℝ), true := 
sorry

end cubic_polynomial_real_roots_l197_197555


namespace sum_of_series_l197_197329

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197329


namespace james_spent_270_dollars_l197_197532

-- Define the conditions
def tablespoons_per_pound : ℝ := 1.5
def cubic_feet_of_bathtub : ℝ := 6
def gallons_per_cubic_foot : ℝ := 7.5
def pounds_per_gallon : ℝ := 8
def cost_per_tablespoon : ℝ := 0.50

-- Calculate total gallons from cubic feet
def total_gallons (ft³ : ℝ) (gallon_per_ft³ : ℝ) : ℝ :=
  ft³ * gallon_per_ft³

-- Calculate total pounds from gallons
def total_pounds (gallons : ℝ) (pounds_per_gall : ℝ) : ℝ :=
  gallons * pounds_per_gall

-- Calculate total tablespoons from pounds
def total_tablespoons (pounds : ℝ) (tablespoons_per_pound : ℝ) : ℝ :=
  pounds * tablespoons_per_pound

-- Calculate total cost from tablespoons
def total_cost (tablespoons : ℝ) (cost_per_tablespoon : ℝ) : ℝ :=
  tablespoons * cost_per_tablespoon

-- Prove that the total cost is equal to the calculated value
theorem james_spent_270_dollars :
  total_cost (total_tablespoons (total_pounds (total_gallons cubic_feet_of_bathtub gallons_per_cubic_foot) pounds_per_gallon) 
              tablespoons_per_pound) 
             cost_per_tablespoon = 270 := 
by 
  sorry

end james_spent_270_dollars_l197_197532


namespace find_x_l197_197239

theorem find_x (x : ℝ) (h : 2 * x = 26 - x + 19) : x = 15 :=
by
  sorry

end find_x_l197_197239


namespace probability_one_fails_out_of_three_l197_197154

-- Defining the problem conditions in Lean.

def probability_lasts (p : ℝ) : Prop := p = 0.2

def probability_fails (p : ℝ) : Prop := p = 0.8

noncomputable def failure_binomial_probability (n k : ℕ) (p q : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * (q ^ (n - k))

-- The theorem that encapsulates the problem question and expected result.

theorem probability_one_fails_out_of_three 
  (p_lasts p_fails : ℝ) 
  (h₁ : probability_lasts p_lasts) 
  (h₂ : probability_fails p_fails) : 
  failure_binomial_probability 3 1 p_fails p_lasts = 0.096 :=
by 
  rw [h₁, h₂]
  sorry -- Proof not required as per instructions

end probability_one_fails_out_of_three_l197_197154


namespace normal_price_of_ticket_l197_197003

variable (P : ℚ)
variable (x : ℚ)
variable (total_cost : ℚ)

def ticket_cost_website := x * P
def ticket_cost_scalper := 4.8 * P - 10
def ticket_cost_discount := 0.6 * P

axiom total_payment : total_cost = 360
axiom total_tickets : x + 2 + 1 = 5
axiom cost_equation : ticket_cost_website + ticket_cost_scalper + ticket_cost_discount = total_cost

theorem normal_price_of_ticket : P = 50 := by
  sorry

end normal_price_of_ticket_l197_197003


namespace CD_eq_DE_l197_197524

-- Define the key points and their relationships in the triangle
variables {A B C D E O : Type}
variables {AC AB : Set (Set A)}
variables [affine_space A] -- To work with points in an affine space

-- Define the conditions from the problem
variables (triangleABC : affine_triangle A)
variables (D_on_AC : D ∈ AC) 
variables (E_on_AB : E ∈ AB)
variables (BE_eq_CD : dist B E = dist C D)
variables (BD_CE_intersect_O : line_intersects BD CE O)
variables (angle_BOC_condition : ∠ B O C = π / 2 + 1 / 2 * (∠ B A C))

-- Define the proof statement
theorem CD_eq_DE [metric_space A] : dist C D = dist D E :=
by
  sorry -- Proof can be filled in by interactive use within Lean

end CD_eq_DE_l197_197524


namespace sin_cos_equation_solution_l197_197234

theorem sin_cos_equation_solution (k : ℤ) :
  ∀ x : ℝ, (sin (3 * x) * (sin x)^3 + cos (3 * x) * (cos x)^3 = 1 / 8) ↔ 
  (∃ n : ℤ, x = n * π + π / 6 ∨ x = n * π - π / 6) := 
sorry

end sin_cos_equation_solution_l197_197234


namespace cone_volume_ratio_l197_197689

noncomputable def volume (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (r^2) * h

theorem cone_volume_ratio
  (rC hC rD hD : ℝ)
  (hrC : rC = 15) (hhC : hC = 30)
  (hrD : rD = 30) (hhD : hD = 15) : 
  volume rC hC / volume rD hD = 1 / 2 := by
sory

end cone_volume_ratio_l197_197689


namespace sin_neg_135_degree_l197_197780

theorem sin_neg_135_degree : sin (-(135 * Real.pi / 180)) = - (Real.sqrt 2 / 2) :=
by 
  -- Here, we need to use the known properties and the equivalences given
  sorry

end sin_neg_135_degree_l197_197780


namespace original_price_each_tv_l197_197573

-- Given Definitions
variables (P : ℝ) (original_cost : ℝ) (discounted_cost : ℝ)
def bought_two_televisions (P : ℝ) : Prop := original_cost = 2 * P
def twenty_five_percent_discount : Prop := discounted_cost = 0.75 * original_cost
def paid_975 (discounted_cost : ℝ) : Prop := discounted_cost = 975

-- Target to Prove
theorem original_price_each_tv (h1 : bought_two_televisions P)
                               (h2 : twenty_five_percent_discount)
                               (h3 : paid_975 discounted_cost) : P = 650 :=
  sorry

end original_price_each_tv_l197_197573


namespace sum_infinite_partial_fraction_l197_197309

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197309


namespace arithmetic_seq_8th_term_l197_197062

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197062


namespace B_work_days_l197_197726

theorem B_work_days
  (A_work_rate : ℝ) (B_work_rate : ℝ) (A_days_worked : ℝ) (B_days_worked : ℝ)
  (total_work : ℝ) (remaining_work : ℝ) :
  A_work_rate = 1 / 15 →
  B_work_rate = total_work / 18 →
  A_days_worked = 5 →
  remaining_work = total_work - A_work_rate * A_days_worked →
  B_days_worked = 12 →
  remaining_work = B_work_rate * B_days_worked →
  total_work = 1 →
  B_days_worked = 12 →
  B_work_rate = total_work / 18 →
  B_days_alone = total_work / B_work_rate →
  B_days_alone = 18 := 
by
  intro hA_work_rate hB_work_rate hA_days_worked hremaining_work hB_days_worked hremaining_work_eq htotal_work hB_days_worked_again hsry_mul_inv hB_days_we_alone_eq
  sorry

end B_work_days_l197_197726


namespace area_of_GHCD_l197_197523

-- Define the conditions given in the problem
def trapezoid := Type
def parallelogram := Type
variables (ABCD : trapezoid)
variables (AB CD : ℝ) (AB_len CD_len : ℝ) (h : ℝ)
variables (A B C D G H : ℝ → ℝ)
variables (GHCD : parallelogram)
variables (AB_midpoint AD_midpoint BC_midpoint CD_midpoint : ℝ)

def conditions : Prop := 
  (AB_len = 12 ∧ CD_len = 18 ∧ h = 15 ∧
   G = AD_midpoint / 2 ∧ 
   H = BC_midpoint / 2)

-- Define the resulting area we need to find
noncomputable def area_GHCD : ℝ := 123.75

-- The theorem to prove the area given the conditions
theorem area_of_GHCD (H : conditions) : 
  ∃ (area_GHCD : ℝ), area_GHCD = 123.75 :=
begin
  use 123.75,
  sorry
end

end area_of_GHCD_l197_197523


namespace sum_first_seven_terms_is_28_l197_197613

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence 
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_a4_a6_sum : a 2 + a 4 + a 6 = 12

-- Prove that the sum of the first seven terms is 28
theorem sum_first_seven_terms_is_28 (h : is_arithmetic_seq a d) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
sorry

end sum_first_seven_terms_is_28_l197_197613


namespace cardinality_of_B_l197_197718

-- Condition definitions
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {x | ∃ (a ∈ A) (b ∈ A), x = a + b}

-- Main theorem statement
theorem cardinality_of_B : B.card = 6 := by
  sorry

end cardinality_of_B_l197_197718


namespace find_a_l197_197428

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1)
  (h_min : ∀ x : ℝ, (-1 < x ∧ x < 3) → log a (3 - x) + log a (x + 1) ≥ -2)
  : a = 1/2 :=
  sorry

end find_a_l197_197428


namespace square_remainder_l197_197061

-- Define the conditions
def isValidPlacement (x y : ℝ) : Prop :=
  x > 0 ∧ x < 8 ∧ y > 0 ∧ y < 8 ∧ (8 * x) % 1 = 0 ∧ (8 * y) % 1 = 0 ∧ 
  (8 * (1 - x / 8)) % 1 = 0 ∧ (8 * (1 - y / 8)) % 1 = 0

def N : ℕ :=
  set.toFinset {xy : ℝ × ℝ | isValidPlacement xy.1 xy.2}.card

-- Statement of the problem
theorem square_remainder : (N ^ 2) % 100 = 41 :=
  sorry

end square_remainder_l197_197061


namespace find_five_l197_197562

theorem find_five (f : ℝ → ℝ) (h_even : ∀ x, f(x) = f(-x)) (h1 : f(1) = 2)
  (h2 : ∀ x y, x * y ≠ 0 → f(Real.sqrt (x^2 + y^2)) = (f(x) * f(y)) / (f(x) + f(y))) :
  f 5 = 2 / 25 :=
sorry

end find_five_l197_197562


namespace find_c_value_l197_197799

variable {x: ℝ}

theorem find_c_value (d e c : ℝ) (h₁ : 6 * d = 18) (h₂ : -15 + 6 * e = -5)
(h₃ : (10 / 3) * c = 15) :
  c = 4.5 :=
by
  sorry

end find_c_value_l197_197799


namespace coloringSchemeExists_l197_197800

noncomputable def cubeFaceColoringExists : Prop :=
∃ (coloring : Face → Color),
  (∀ (o1 : Orientation), appearsAsInDiagramInOrientation coloring o1) ∧
  (∀ (o2 : Orientation), appearsAsInDiagramInOrientation coloring o2) ∧
  (∀ (o3 : Orientation), appearsAsInDiagramInOrientation coloring o3)

axiom orientation1 : Orientation
axiom orientation2 : Orientation
axiom orientation3 : Orientation

theorem coloringSchemeExists :
  cubeFaceColoringExists :=
by
  sorry

end coloringSchemeExists_l197_197800


namespace max_next_person_weight_l197_197599

def avg_weight_adult := 150
def avg_weight_child := 70
def max_weight_elevator := 1500
def num_adults := 7
def num_children := 5

def total_weight_adults := num_adults * avg_weight_adult
def total_weight_children := num_children * avg_weight_child
def current_weight := total_weight_adults + total_weight_children

theorem max_next_person_weight : 
  max_weight_elevator - current_weight = 100 := 
by 
  sorry

end max_next_person_weight_l197_197599


namespace convert_rectangular_to_polar_l197_197789

noncomputable def polar_coordinates (x : ℝ) (y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := real.arctan (y / x)
  (r, θ)

theorem convert_rectangular_to_polar :
  polar_coordinates 8 (2 * real.sqrt 2) = (6 * real.sqrt 2, 0.349) :=
by
  sorry

end convert_rectangular_to_polar_l197_197789


namespace solution_set_inequality_l197_197434

variable {f : ℝ → ℝ}

-- Assumptions
axiom domain_condition : ∀ x : ℝ, -6 ≤ x → x ≤ 2 → x + 1 ∈ [-5, 3]
axiom odd_function : ∀ x : ℝ, f(x) + f(-x) = 0
axiom f_neg1_zero : f(-1) = 0
axiom decreasing_on_neg4_to_0 : ∀ {x y : ℝ}, -4 ≤ x → x < y → y < 0 → f(x) > f(y)
axiom no_max_value : ¬∃ b, ∀ x : ℝ, f(x) ≤ b

-- Goal
theorem solution_set_inequality : 
  {x | x^3 * f(x) ≤ 0} = 
  {x | -4 ≤ x ∧ x ≤ -1 ∨ x = 0 ∨ 1 ≤ x ∧ x ≤ 4} :=
sorry

end solution_set_inequality_l197_197434


namespace leon_total_payment_l197_197009

noncomputable def total_payment : ℕ :=
let toy_organizers_cost := 78 * 3 in
let gaming_chairs_cost := 83 * 2 in
let total_orders := toy_organizers_cost + gaming_chairs_cost in
let delivery_fee := total_orders * 5 / 100 in
total_orders + delivery_fee

theorem leon_total_payment : total_payment = 420 :=
by
  sorry

end leon_total_payment_l197_197009


namespace measure_angle_FAE_eq_60_l197_197762

theorem measure_angle_FAE_eq_60 (ABC : Type) (BCDE : Type) (AEF : Type)
  [equilateral_triang ABC] [shares_side_with_square BCDE ABC] [equilateral_triang AEF]
  (AF_parallel_BC : parallel AF BC) :
  measure_angle FAE = 60 :=
by
  sorry

end measure_angle_FAE_eq_60_l197_197762


namespace cone_surface_area_correct_l197_197730

def cone_surface_area (slant_height base_radius : ℝ) : ℝ :=
  let base_circumference := 2 * Real.pi * base_radius
  let lateral_surface_area := (base_circumference * slant_height) / 2
  let base_area := Real.pi * base_radius^2
  lateral_surface_area + base_area

theorem cone_surface_area_correct : cone_surface_area 2 1 = 3 * Real.pi := by
  sorry


end cone_surface_area_correct_l197_197730


namespace tan_alpha_eq_7_5_l197_197901

theorem tan_alpha_eq_7_5 (α : ℝ) (h : tan (π - π / 4) = 1 / 6) : tan α = 7 / 5 :=
sorry

end tan_alpha_eq_7_5_l197_197901


namespace series_sum_l197_197336

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197336


namespace find_t_l197_197189

-- Given vertices of triangle ABC
def A : (ℝ × ℝ) := (1, 10)
def B : (ℝ × ℝ) := (3, 0)
def C : (ℝ × ℝ) := (10, 0)

-- A horizontal line with equation y = t intersects line segment AB at T and line segment AC at U
-- Forming triangle ATU with area 15
def area_triangle (A T U : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (T.2 - U.2) + T.1 * (U.2 - A.2) + U.1 * (A.2 - T.2))

theorem find_t (t : ℝ) (T U : ℝ × ℝ) 
  (hT : T.2 = t) 
  (hU : U.2 = t) 
  (hT_eq : T = ((15 - t)/5, t)) 
  (hU_eq : U = (10 - (9 * t)/10, t)) 
  (h_area : area_triangle A T U = 15) :
  t ≈ 3.22 :=
begin
  -- Proof would go here, which is omitted as stated in the instructions
  sorry
end

end find_t_l197_197189


namespace abs_val_neg_three_l197_197129

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l197_197129


namespace min_value_f_on_interval_l197_197372

open Real

noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 - 4 * tan x - 8 * cot x + 4 * cot x ^ 2 + 5

theorem min_value_f_on_interval :
  ∃ x ∈ Ioo (π / 2) π, ∀ y ∈ Ioo (π / 2) π, f y ≥ f x ∧ f x = 9 - 8 * sqrt 2 := 
by
  -- proof omitted
  sorry

end min_value_f_on_interval_l197_197372


namespace cone_slant_height_is_correct_cone_volume_is_correct_l197_197864

-- Definition of constants
def base_radius : ℝ := 2
def lateral_surface_is_semicircle : Prop := true  -- Placeholder for condition on lateral surface

-- Definition of slant height
def slant_height (r : ℝ) : ℝ := 4  -- Since l is derived directly as 4 from condition

-- Definition of height using Pythagorean theorem
def height (r l : ℝ) : ℝ := real.sqrt(l^2 - r^2)

-- Definition of volume
def volume (r h : ℝ) : ℝ := (1/3) * real.pi * r^2 * h

-- Theorem statement for slant height
theorem cone_slant_height_is_correct : 
  lateral_surface_is_semicircle → slant_height base_radius = 4 :=
by
  intros
  sorry

-- Theorem statement for volume
theorem cone_volume_is_correct :
  lateral_surface_is_semicircle → 
  volume base_radius (height base_radius (slant_height base_radius)) = (8 * real.sqrt 3) / 3 * real.pi :=
by
  intros
  sorry

end cone_slant_height_is_correct_cone_volume_is_correct_l197_197864


namespace find_a_n_find_T_n_l197_197870

-- Definitions for the problem
def a_n (n : ℕ) : ℕ := 2 * n - 1 -- General term formula for the arithmetic sequence
def b_n (n : ℕ) : ℕ := 3 ^ (a_n n) -- Definition of the sequence b_n

-- Sum of the first n terms of the arithmetic sequence a_n
def S_n (n : ℕ) : ℕ := n * a_n n + (n * (n - 1)) / 2

-- Sum of the first n terms of the sequence b_n
def T_n (n : ℕ) : ℕ := (1 / 3 : ℚ) * (9 * (1 - 9 ^ n) / (1 - 9))

-- Theorem statements
theorem find_a_n (a_3 : ℕ := 5) (S_4 : ℕ := 16) : ∀ n : ℕ, a_n n = 2 * n - 1 := sorry

theorem find_T_n (a_3 : ℕ := 5) (S_4 : ℕ := 16) : ∀ n : ℕ, T_n n = (3 * (9 ^ n - 1) / 8) := sorry

end find_a_n_find_T_n_l197_197870


namespace concyclic_A_l197_197770

-- Definitions for points and circle ω
variables {A B C D E P Q T A' : Type} [Inhabited A] [Inhabited B] [Inhabited C] 
          [Inhabited D] [Inhabited E] [Inhabited P] [Inhabited Q] [Inhabited T] [Inhabited A']
variables (circle_omega : Set (A × ℝ))

-- Conditions
axiom points_on_circle : (A ∈ circle_omega) ∧ (B ∈ circle_omega) ∧ (C ∈ circle_omega) ∧ (D ∈ circle_omega) ∧ (E ∈ circle_omega)
axiom AB_eq_BD : (dist A B = dist B D)
axiom BC_eq_CE : (dist B C = dist C E)
axiom chord_intersect : intersecting_chord A C B E P
axiom parallel_through_A : parallel (line_through A (line_through B E)) (extension D E Q)
axiom circle_APQ : on_circle (circle_through A P Q) T ∧ minor_arc T D E

-- Reflection property
axiom reflection_A' : reflection_over_line A C A'

-- Goal
theorem concyclic_A'_B_P_T :
  concyclic {A', B, P, T} :=
begin
  sorry
end

end concyclic_A_l197_197770


namespace lily_initial_money_l197_197974

def cost_celery := 5
def cost_cereal := 12 * 0.5
def cost_bread := 8
def cost_milk := 10 - (10 * 0.1)
def cost_potato := 1
def potatoes_quantity := 6
def remaining_money_for_coffee := 26

theorem lily_initial_money :
  (cost_celery + cost_cereal + cost_bread + cost_milk + (cost_potato * potatoes_quantity) + remaining_money_for_coffee) = 60 := by
  sorry

end lily_initial_money_l197_197974


namespace painted_cubes_l197_197271

theorem painted_cubes (n : ℕ) (h₁ : n = 5) :
  let total_cubes := n * n * n,
      painted_three := 8,
      painted_two := 12 * (n - 2),
      painted_one := 6 * ((n - 2) * (n - 2)),
      not_painted := (n - 2) * (n - 2) * (n - 2)
  in total_cubes = 125 ∧ painted_three = 8 ∧ painted_two = 36 ∧ painted_one = 54 ∧ not_painted = 27 :=
by
  sorry

end painted_cubes_l197_197271


namespace arithmetic_sequence_eighth_term_l197_197093

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197093


namespace tom_earns_more_l197_197564

noncomputable def linda_investment (P : ℕ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

noncomputable def tom_investment (P : ℕ) (r : ℝ) (n k : ℕ) : ℝ :=
  P * (1 + r / k)^(n * k)

def difference_in_earnings (P : ℕ) (r : ℝ) (n k : ℕ) : ℝ :=
  tom_investment P r n k - linda_investment P r n

theorem tom_earns_more (P : ℕ) (r : ℝ) (n k : ℕ) :
  difference_in_earnings P r n k ≈ 98.94 :=
by
  have h₁ : linda_investment 60000 0.05 3 = 69457.5,
  have h₂ : tom_investment 60000 0.05 3 2 ≈ 69556.44,
  exact sorry

end tom_earns_more_l197_197564


namespace bill_score_l197_197043

theorem bill_score
  (J B S : ℕ)
  (h1 : B = J + 20)
  (h2 : B = S / 2)
  (h3 : J + B + S = 160) : 
  B = 45 := 
by 
  sorry

end bill_score_l197_197043


namespace quadratic_form_abc_sum_l197_197639

theorem quadratic_form (x : ℝ) : 
  let a := -3
  let b := -4
  let c := 192
  -3 * x^2 + 24 * x + 144 = a * (x + b)^2 + c := 
by 
  sorry

theorem abc_sum :
  let a := -3
  let b := -4
  let c := 192
  a + b + c = 185 := 
by 
  simp [a, b, c]
  norm_num

end quadratic_form_abc_sum_l197_197639


namespace number_of_odd_divisors_lt_100_is_9_l197_197464

theorem number_of_odd_divisors_lt_100_is_9 :
  (finset.filter (λ n : ℕ, ∃ k : ℕ, n = k * k) (finset.range 100)).card = 9 :=
sorry

end number_of_odd_divisors_lt_100_is_9_l197_197464


namespace revenue_times_l197_197703

noncomputable def revenue_ratio (D : ℝ) : ℝ :=
  let revenue_Nov := (2 / 5) * D
  let revenue_Jan := (1 / 3) * revenue_Nov
  let average := (revenue_Nov + revenue_Jan) / 2
  D / average

theorem revenue_times (D : ℝ) (hD : D ≠ 0) : revenue_ratio D = 3.75 :=
by
  -- skipped proof
  sorry

end revenue_times_l197_197703


namespace angle_ABC_is_83_degrees_l197_197595

theorem angle_ABC_is_83_degrees (A B C D K : Type)
  (angle_BAC : Real) (angle_CAD : Real) (angle_ACD : Real)
  (AB AC AD : Real) (angle_ABC : Real) :
  angle_BAC = 60 ∧ angle_CAD = 60 ∧ angle_ACD = 23 ∧ AB + AD = AC → 
  angle_ABC = 83 :=
by
  sorry

end angle_ABC_is_83_degrees_l197_197595


namespace sum_div_1000_remainder_l197_197958

open BigOperators

def is_distinct_4_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ (let digits := (n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10) in
                        digits.fst ≠ digits.snd ∧ digits.fst ≠ digits.2 ∧ digits.fst ≠ digits.3 ∧ 
                        digits.snd ≠ digits.2 ∧ digits.snd ≠ digits.3 ∧ 
                        digits.2 ≠ digits.3)

def four_digit_distinct_sum : ℕ :=
  ∑ n in (finset.range 10000).filter is_distinct_4_digits, n

theorem sum_div_1000_remainder :
  four_digit_distinct_sum % 1000 = 960 :=
  sorry

end sum_div_1000_remainder_l197_197958


namespace total_pizza_slices_l197_197198

theorem total_pizza_slices (p s : ℕ) (h_p : p = 36) (h_s : s = 12) : p * s = 432 :=
by
  rw [h_p, h_s]
  exact rfl

end total_pizza_slices_l197_197198


namespace total_seeds_l197_197232

-- Definitions and conditions
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds
def Eun_seeds : ℕ := 2 * Gwi_seeds

-- Theorem statement
theorem total_seeds : Bom_seeds + Gwi_seeds + Yeon_seeds + Eun_seeds = 2340 :=
by
  -- Skipping the proof steps with sorry
  sorry

end total_seeds_l197_197232


namespace sum_of_divisors_multiple_of_3_l197_197591

theorem sum_of_divisors_multiple_of_3 (k : ℕ) (n : ℕ) (h : n = 3 * k + 2) :
  ∃ m : ℕ, σ n = 3 * m := sorry

end sum_of_divisors_multiple_of_3_l197_197591


namespace exponent_multiplication_l197_197902

theorem exponent_multiplication (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 4) :
  a^(m + n) = 8 := by
  sorry

end exponent_multiplication_l197_197902


namespace solution_bound_l197_197537

variables {a b c x y z : ℝ} {λ : ℝ}

theorem solution_bound (h₁ : a^2 + b^2 + c^2 = 1) 
  (h₂ : λ > 0) (h₃ : λ ≠ 1) 
  (h₄ : x - λ * y = a) 
  (h₅ : y - λ * z = b) 
  (h₆ : z - λ * x = c) :
  x^2 + y^2 + z^2 ≤ 1 / (λ - 1)^2 :=
by
  sorry

end solution_bound_l197_197537


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197666

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197666


namespace tan_22_5_expression_l197_197158

-- Define the problem in Lean
theorem tan_22_5_expression : ∃ (a b c d : ℤ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a > 0 ∧ b > 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 3 ∧ (tan(π / 8) = real.sqrt a - real.sqrt b + real.sqrt c - d) :=
by
  have := real.tan_pi_div_eight -- gives us that tan(π / 8) = √2 - 1
  use [2, 1, 0, 0]
  split; norm_num -- ensure positivity and ordering of integers (a = 2, b = 1, c = 0, d = 0)
  split; linarith -- finalize the proof of the form
  sorry -- to be filled in

end tan_22_5_expression_l197_197158


namespace max_sum_of_products_l197_197759

theorem max_sum_of_products : 
  ∃ (faces : Fin 6 → ℕ), 
    (faces ⟨0, by norm_num⟩ = 3) ∧
    (faces ⟨1, by norm_num⟩ = 4) ∧
    (faces ⟨4, by norm_num⟩ = 7) ∧
    (faces ⟨5, by norm_num⟩ = 8) ∧
    (∀ i, 
        faces i ∈ {3, 4, 5, 6, 7, 8}) ∧
    let a := faces ⟨0, by norm_num⟩,
        b := faces ⟨1, by norm_num⟩,
        c := faces ⟨2, by norm_num⟩,
        d := faces ⟨3, by norm_num⟩,
        e := faces ⟨4, by norm_num⟩,
        f := faces ⟨5, by norm_num⟩ in 
    a + b = 7 ∧
    e + f = 15 ∧
    a + b = 7 ∧ 
    e = 7 ∧
    f = 8 ∧ 
    (a + b) * (c + d) * (e + f) = 1155 :=
by
  let faces := 
    fun
      | ⟨0, _⟩ => 3
      | ⟨1, _⟩ => 4
      | ⟨2, _⟩ => 5
      | ⟨3, _⟩ => 6
      | ⟨4, _⟩ => 7
      | ⟨5, _⟩ => 8
  existsi faces
  -- enter the rest of proof to justify according to conditions stated
  sorry

end max_sum_of_products_l197_197759


namespace find_a3_plus_inv_a3_l197_197476

variable (a : ℝ)

theorem find_a3_plus_inv_a3 (h : (a + 1 / a)^2 = 5) : 
  (a^3 + 1 / a^3 = 2 * real.sqrt 5) ∨ (a^3 + 1 / a^3 = -2 * real.sqrt 5) :=
begin
  sorry
end

end find_a3_plus_inv_a3_l197_197476


namespace sum_infinite_partial_fraction_l197_197306

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l197_197306


namespace min_value_of_f_on_interval_l197_197374

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan x ^ 2 - 4 * Real.tan x - 8 * (1 / Real.tan x) + 4 * (1 / Real.tan x) ^ 2 + 5

theorem min_value_of_f_on_interval :
  is_min (f x) (9 - 8 * Real.sqrt 2) (Ioo (Real.pi / 2) Real.pi) :=
sorry

end min_value_of_f_on_interval_l197_197374


namespace judgments_correct_l197_197412

variables {l m : Line} (a : Plane)

def is_perpendicular (l : Line) (a : Plane) : Prop := -- Definition of perpendicularity between a line and a plane
sorry

def is_parallel (l m : Line) : Prop := -- Definition of parallel lines
sorry

def is_contained_in (m : Line) (a : Plane) : Prop := -- Definition of a line contained in a plane
sorry

theorem judgments_correct 
  (hl : is_perpendicular l a)
  (hm : l ≠ m) :
  (∀ m, is_perpendicular m l → is_parallel m a) ∧ 
  (is_perpendicular m a → is_parallel m l) ∧
  (is_contained_in m a → is_perpendicular m l) ∧
  (is_parallel m l → is_perpendicular m a) :=
sorry

end judgments_correct_l197_197412


namespace ariel_years_fencing_l197_197764

-- Definitions based on given conditions
def fencing_start_year := 2006
def birth_year := 1992
def current_age := 30

-- To find: The number of years Ariel has been fencing
def current_year : ℕ := birth_year + current_age
def years_fencing : ℕ := current_year - fencing_start_year

-- Proof statement
theorem ariel_years_fencing : years_fencing = 16 := by
  sorry

end ariel_years_fencing_l197_197764


namespace garden_strawberry_area_l197_197951

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l197_197951


namespace system1_solution_l197_197059

theorem system1_solution (x y : ℝ) (h1 : 4 * x - 3 * y = 1) (h2 : 3 * x - 2 * y = -1) : x = -5 ∧ y = 7 :=
sorry

end system1_solution_l197_197059


namespace perfect_square_last_two_digits_even_l197_197051

/-!
  # Theorem
  The product of the last two digits of any perfect square is even.
-/

theorem perfect_square_last_two_digits_even (n : ℕ) :
  (∃ m : ℤ, n = m * m) → (let d0 := n % 10,
                              d1 := (n / 10) % 10 in
                              (d0 * d1) % 2 = 0) :=
by
  sorry

end perfect_square_last_two_digits_even_l197_197051


namespace even_function_a_equals_one_l197_197432

theorem even_function_a_equals_one 
  (a : ℝ) 
  (h : ∀ x : ℝ, 2^(-x) + a * 2^x = 2^x + a * 2^(-x)) : 
  a = 1 := 
by
  sorry

end even_function_a_equals_one_l197_197432


namespace average_age_in_club_l197_197501

theorem average_age_in_club :
  let women_avg_age := 32
  let men_avg_age := 38
  let children_avg_age := 10
  let women_count := 12
  let men_count := 18
  let children_count := 10
  let total_ages := (women_avg_age * women_count) + (men_avg_age * men_count) + (children_avg_age * children_count)
  let total_people := women_count + men_count + children_count
  let overall_avg_age := (total_ages : ℝ) / (total_people : ℝ)
  overall_avg_age = 29.2 := by
  sorry

end average_age_in_club_l197_197501


namespace necessary_but_not_sufficient_condition_l197_197021

noncomputable theory

open Real

def f (a x : ℝ) : ℝ := -x^3 + 2 * a * x

def derivative_f (a x : ℝ) : ℝ := -3 * x^2 + 2 * a

def monotonic_decreasing (f' : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x ∈ I, f' x ≤ 0

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  a < 3/2 → (monotonic_decreasing (derivative_f a) {x | x ≤ 1} ↔ a ≤ 0) :=
by sorry

end necessary_but_not_sufficient_condition_l197_197021


namespace rainfall_difference_l197_197653

-- Define the conditions
def day1_rainfall := 26
def day2_rainfall := 34
def average_rainfall := 140
def less_rainfall := 58

-- Calculate the total rainfall this year in the first three days
def total_rainfall_this_year := average_rainfall - less_rainfall

-- Calculate the total rainfall in the first two days
def total_first_two_days := day1_rainfall + day2_rainfall

-- Calculate the rainfall on the third day
def day3_rainfall := total_rainfall_this_year - total_first_two_days

-- The proof problem
theorem rainfall_difference : day2_rainfall - day3_rainfall = 12 := 
by
  sorry

end rainfall_difference_l197_197653


namespace tangent_line_equation_l197_197140

noncomputable def func : ℝ → ℝ := λ x, Real.exp x

theorem tangent_line_equation :
  ∃ (slope intercept : ℝ), slope = Real.exp 0 ∧ intercept = func 0 ∧ ∀ x y : ℝ, y = slope * x + intercept ↔ x - y + 1 = 0 :=
by {
    let slope := Real.exp 0,
    let intercept := func 0,
    use slope,
    use intercept,
    split,
    { exact rfl },
    split,
    { exact rfl },
    intro x,
    intro y,
    split,
    { intro h,
      rw [h],
      simp },
    { intro h,
      rw [← h],
      simp },
}

end tangent_line_equation_l197_197140


namespace interesting_quadruples_count_l197_197355

/-- Definition of interesting ordered quadruples (a, b, c, d) where 1 ≤ a < b < c < d ≤ 15 and a + b > c + d --/
def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + b > c + d

/-- The number of interesting ordered quadruples (a, b, c, d) is 455 --/
theorem interesting_quadruples_count : 
  (∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s.card = 455 ∧ ∀ (a b c d : ℕ), 
    ((a, b, c, d) ∈ s ↔ is_interesting_quadruple a b c d)) :=
sorry

end interesting_quadruples_count_l197_197355


namespace arithmetic_seq_8th_term_l197_197064

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197064


namespace green_beans_weight_l197_197567

theorem green_beans_weight (G : ℝ) 
  (rice_remains : G - 30 : ℝ)
  (sugar_remains : G - 10 : ℝ)
  (lost_weights_remain : (2 / 3) * (G - 30) + G + (4 / 5) * (G - 10) = 120) :
  G = 60 := 
by 
  sorry

end green_beans_weight_l197_197567


namespace no_tiling_possible_l197_197787

-- Defining the standard 8x8 chessboard
def chessboard : Type := finset (ℕ × ℕ)

-- Define a function to color the chessboard in a checkerboard pattern
def is_white (coord : ℕ × ℕ) : Prop :=
  (coord.1 + coord.2) % 2 = 0

-- Define the modified chessboard with the top-left and bottom-right squares removed
def modified_chessboard : chessboard :=
  (finset.univ.filter (λ coord : ℕ × ℕ, coord ≠ (0, 0) ∧ coord ≠ (7, 7)))

-- Define the property of dominos covering one white and one black square
inductive domino : Type
| mk : (ℕ × ℕ) → (ℕ × ℕ) → domino

def domino.covers (d : domino) (sq1 sq2 : ℕ × ℕ) : Prop :=
  (sq1 = d.1 ∧ sq2 = d.2) ∨ (sq1 = d.2 ∧ sq2 = d.1)

def domino_tiling (tiles : finset domino) (board : chessboard) : Prop :=
  ∀ sq ∈ board, ∃ d ∈ tiles, domino.covers d sq (sq.snd - sq.fst)

-- Now, the proof statement
theorem no_tiling_possible :
  ¬ ∃ (tiles : finset domino), domino_tiling tiles modified_chessboard := sorry

end no_tiling_possible_l197_197787


namespace sum_of_series_l197_197331

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l197_197331


namespace no_distinct_integers_cycle_l197_197050

theorem no_distinct_integers_cycle (p : ℤ → ℤ) 
  (x : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (n : ℕ) (h_n_ge_3 : n ≥ 3)
  (hx_cycle : ∀ i, i < n → p (x i) = x (i + 1) % n) :
  false :=
sorry

end no_distinct_integers_cycle_l197_197050


namespace not_possible_values_l197_197153

theorem not_possible_values (t h d : ℕ) (ht : 3 * t - 6 * h = 2001) (hd : t - h = d) (hh : 6 * h > 0) :
  ∃ n, n = 667 ∧ ∀ d : ℕ, d ≤ 667 → ¬ (t = h + d ∧ 3 * (h + d) - 6 * h = 2001) :=
by
  sorry

end not_possible_values_l197_197153


namespace difference_Q_R_l197_197761

variables {T Ps Qs Rs : ℕ}
variables {x : ℕ}
variable h1 : T = Ps + Qs + Rs
variable h2 : Ps = 3 * x
variable h3 : Qs = 7 * x
variable h4 : Rs = 12 * x
variable h5 : Qs - Ps = 2800
variable h6 : 50000 ≤ T ∧ T ≤ 75000
variable h7 : 5000 ≤ Ps
variable h8 : Rs ≤ 45000

theorem difference_Q_R : 14000 = Rs - Qs :=
sorry

end difference_Q_R_l197_197761


namespace series_sum_l197_197338

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l197_197338


namespace joe_bought_3_oranges_l197_197534

theorem joe_bought_3_oranges 
  (x : ℕ)
  (cost_orange : ℝ := 4.50)
  (cost_juice : ℝ := 0.50)
  (cost_honey : ℝ := 5.00)
  (cost_plants : ℝ := 18.00)
  (total_cost : ℝ := 68.00)
  (num_juices : ℕ := 7)
  (num_honeys : ℕ := 3)
  (num_plants : ℕ := 4) :

  let cost_for_juices := num_juices * cost_juice,
      cost_for_honeys := num_honeys * cost_honey,
      cost_for_plants := num_plants / 2 * cost_plants,
      total_other_cost := cost_for_juices + cost_for_honeys + cost_for_plants,
      remaining_cost := total_cost - total_other_cost,
      oranges_bought := remaining_cost / cost_orange in
    oranges_bought = 3 := 
by 
  sorry

end joe_bought_3_oranges_l197_197534


namespace possible_knight_counts_l197_197651

-- Define the types of natives
inductive Native
| Knight
| Liar
| Bore

-- Define the behavior of each native regarding their neighbors
def is_truthful_about_neighbors (n : Native) (left right : Native) : Prop :=
  match n with
  | Native.Knight => left = Native.Bore ∧ right = Native.Bore
  | Native.Liar => ¬(left = Native.Bore ∧ right = Native.Bore)
  | Native.Bore => true -- Bores can say anything unless they are next to a Knight

-- Define the conditions: the number of knights seated at the table
def valid_knight_count (natives : List Native) : Prop :=
  let num_knights := natives.count Native.Knight
  num_knights = 1 ∨ num_knights = 2

-- Define the setting of 7 natives seating in a circle where each responds that both neighbors are bores
def valid_configuration (natives : List Native) : Prop :=
  natives.length = 7 ∧
  (∀ i, is_truthful_about_neighbors 
    (natives.get! (i % 7)) 
    (natives.get! ((i - 1) % 7)) 
    (natives.get! ((i + 1) % 7)))

-- The main theorem statement
theorem possible_knight_counts (natives : List Native) :
  valid_configuration natives → valid_knight_count natives :=
sorry

end possible_knight_counts_l197_197651


namespace probability_not_paired_shoes_l197_197833

noncomputable def probability_not_pair (total_shoes : ℕ) (pairs : ℕ) (shoes_drawn : ℕ) : ℚ :=
  let total_ways := Nat.choose total_shoes shoes_drawn
  let pair_ways := pairs * Nat.choose 2 2
  let not_pair_ways := total_ways - pair_ways
  not_pair_ways / total_ways

theorem probability_not_paired_shoes (total_shoes pairs shoes_drawn : ℕ) (h1 : total_shoes = 6) 
(h2 : pairs = 3) (h3 : shoes_drawn = 2) :
  probability_not_pair total_shoes pairs shoes_drawn = 4 / 5 :=
by 
  rw [h1, h2, h3]
  simp [probability_not_pair, Nat.choose]
  sorry

end probability_not_paired_shoes_l197_197833


namespace customer_paid_correct_amount_l197_197151

theorem customer_paid_correct_amount (cost_price : ℕ) (markup_percentage : ℕ) (total_price : ℕ) :
  cost_price = 6500 → 
  markup_percentage = 30 → 
  total_price = cost_price + (cost_price * markup_percentage / 100) → 
  total_price = 8450 :=
by
  intros h_cost_price h_markup_percentage h_total_price
  sorry

end customer_paid_correct_amount_l197_197151


namespace area_OPF_l197_197416

def parabola (x y : ℝ) : Prop := x^2 = 16 * y

def distance (a b : ℝ) := (a^2 + b^2).sqrt

def point_on_parabola (P : ℝ × ℝ) :=
  ∃ m n : ℝ, P = (m, n) ∧ parabola m n

def focus : ℝ × ℝ := (0, 4)

theorem area_OPF (P : ℝ × ℝ) (h1 : point_on_parabola P) (h2 : distance (P.1 - focus.1) (P.2 - focus.2) = 8) : 
  ∃ A : ℝ, A = 16 :=
sorry

end area_OPF_l197_197416


namespace two_lines_perpendicular_to_same_line_l197_197195

open EuclideanGeometry

noncomputable def lines_perpendicular_to_same_line (l: Line) (l1 l2: Line) : Prop := 
  (Line.perpendicular l1 l ∧ Line.perpendicular l2 l)

-- In 2D, lines perpendicular to the same line are parallel.
axiom perpendicular_lines_parallel_2D (l l1 l2: Line) (plane: Plane) :
  in_plane l plane → in_plane l1 plane → in_plane l2 plane → lines_perpendicular_to_same_line l l1 l2 → parallel l1 l2

-- In 3D, lines perpendicular to the same line can be parallel, intersecting, or skew.
axiom perpendicular_lines_relations_3D (l l1 l2: Line) :
  (lines_perpendicular_to_same_line l l1 l2 → (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2)) 

theorem two_lines_perpendicular_to_same_line (l l1 l2: Line) (plane: Plane) :
  in_plane l plane → in_plane l1 plane → in_plane l2 plane → lines_perpendicular_to_same_line l l1 l2 →
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) :=
by
  intro h1 h2 h3 h4
  exact Or.inl (perpendicular_lines_parallel_2D l l1 l2 plane h1 h2 h3 h4)
  -- Using the perpendicular_lines_parallel_2D axiom for 2D and placeholder sorry for 3D part
  sorry

end two_lines_perpendicular_to_same_line_l197_197195


namespace correct_number_of_true_statements_l197_197149

-- Define the problem and conditions.
def problem_statement : Prop :=
  (∃ α β : ℝ, sin (α - β) = sin α * cos β + cos α * sin β) ∧
  ¬(("∀ A : ℝ, A > π / 6 ↔ sin A > 1 / 2")) ∧
  (∀ {A B : ℝ}, sin A = sin B → A = B) ∧
  (∀ {α : ℝ}, ¬ (α = π / 6 → sin α = 1 / 2) ↔ (α ≠ π / 6 → sin α ≠ 1 / 2))

-- Prove that exactly 3 statements are true.
theorem correct_number_of_true_statements : problem_statement → true_statements = 3 :=
  by
    intros h
    sorry -- Proof step is omitted.

end correct_number_of_true_statements_l197_197149


namespace convex_ngon_equal_segments_sum_l197_197731

theorem convex_ngon_equal_segments_sum (N : ℕ) (x : ℝ) (a b : Fin N → ℝ) 
  (convex_ngon_inscribed : ∀ k : Fin N, a k * (x + b k) = b ((k : ℕ + N - 1) % N) * (x + a ((k : ℕ + N - 1) % N))) :
  ∑ k in Finset.range N, a k = ∑ k in Finset.range N, b k :=
by
  sorry

end convex_ngon_equal_segments_sum_l197_197731


namespace A_iff_B_l197_197411

-- Define Proposition A: ab > b^2
def PropA (a b : ℝ) : Prop := a * b > b ^ 2

-- Define Proposition B: 1/b < 1/a < 0
def PropB (a b : ℝ) : Prop := 1 / b < 1 / a ∧ 1 / a < 0

theorem A_iff_B (a b : ℝ) : (PropA a b) ↔ (PropB a b) := sorry

end A_iff_B_l197_197411


namespace domain_of_f_l197_197138

def domain_f (x : ℝ) : Prop := x ≤ 4 ∧ x ≠ 1

theorem domain_of_f :
  {x : ℝ | ∃(h1 : 4 - x ≥ 0) (h2 : x - 1 ≠ 0), true} = {x : ℝ | domain_f x} :=
by
  sorry

end domain_of_f_l197_197138


namespace finite_operations_l197_197600

theorem finite_operations (seq : List ℕ) (h_pos : ∀ x ∈ seq, x > 0) :
  ∃ N, ∀ seq', (seq', h_pos') ~> seq, seq' = seq' :=  -- ~> denotes a single valid operation step
sorry

end finite_operations_l197_197600


namespace problem_proof_l197_197758

variables (E F G H : Prop)

axiom F_H_mut_exclusive_complementary : (F ∧ H) → False ∧ (¬F → H) ∧ (¬H → F)
axiom P_E_or_H : P (E ∨ H) = P E + P H

theorem problem_proof :
  (F ∧ H -> False ∧ (¬F → H) ∧ (¬H → F)) ∧ (P (E ∨ H) = P E + P H) :=
by
  exact (F_H_mut_exclusive_complementary, P_E_or_H)

end problem_proof_l197_197758


namespace arithmetic_mean_multiplied_by_three_l197_197483

theorem arithmetic_mean_multiplied_by_three (b : Fin 5 → ℝ) :
  (∑ i, 3 * b i) / 5 = 3 * (∑ i, b i) / 5 :=
by
  sorry

end arithmetic_mean_multiplied_by_three_l197_197483


namespace value_of_x_l197_197904

theorem value_of_x (x : ℤ) : (x + 1) * (x + 1) = 16 ↔ (x = 3 ∨ x = -5) := 
by sorry

end value_of_x_l197_197904


namespace range_of_x_in_expansion_l197_197490

theorem range_of_x_in_expansion :
  ∀ x : ℝ, (binomial 6 2 * 2^4 * (-x)^2 ≤ binomial 6 1 * 2^5 * (-x) ∧ 
             binomial 6 1 * 2^5 * (-x) < binomial 6 0 * 2^6) ↔ (-1/3 < x ∧ x ≤ 0) :=
by
  sorry

end range_of_x_in_expansion_l197_197490


namespace expensive_coffee_cost_l197_197266

theorem expensive_coffee_cost (x : ℝ) :
  let cost_regular := 6.42 * 7
  let weight_expensive := 68.25
  let weight_total := 75.25
  let price_per_pound := 7.20
  let total_cost := weight_total * price_per_pound
  cost_regular + weight_expensive * x = total_cost → x = 7.28 :=
by
  intros
  let cost_regular := 6.42 * 7
  let weight_expensive := 68.25
  let weight_total := 75.25
  let price_per_pound := 7.20
  let total_cost := weight_total * price_per_pound
  have equation : cost_regular + weight_expensive * x = total_cost, by assumption
  sorry

end expensive_coffee_cost_l197_197266


namespace y_coord_of_equidistant_point_eq_19_over_8_l197_197687

theorem y_coord_of_equidistant_point_eq_19_over_8 :
  ∃ y : ℚ, (∀ A B : ℝ × ℝ, A = (-3, 1) ∧ B = (2, 5) →
  (dist (0, y) A = dist (0, y) B)) ↔ y = 19/8 :=
begin
  sorry
end

end y_coord_of_equidistant_point_eq_19_over_8_l197_197687


namespace tan_sum_identity_l197_197840

variable (α : ℝ)

theorem tan_sum_identity 
  (h1 : α ∈ set.Ioo (π / 2) π)
  (h2 : sin α = (√5) / 5) :
  tan (2 * α + (π / 4)) = -1 / 7 := by
  sorry

end tan_sum_identity_l197_197840


namespace find_b_l197_197882

theorem find_b (a b : ℝ) (h1 : (1 : ℝ)^3 + a*(1)^2 + b*1 + a^2 = 10)
    (h2 : 3*(1 : ℝ)^2 + 2*a*(1) + b = 0) : b = -11 :=
sorry

end find_b_l197_197882


namespace sequence_formulas_and_sum_l197_197854

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1
noncomputable def b (n : ℕ) : ℕ := 2 ^ n
def c (n : ℕ) : ℕ := a n - b n
def S_a (n : ℕ) : ℕ := n * (2 + (3 * n - 1)) / 2
def S_b (n : ℕ) : ℕ := 2 ^ (n + 1) - 2
def S_c (n : ℕ) : ℕ := S_a n - S_b n

theorem sequence_formulas_and_sum (n : ℕ) :
  ∀ n ≥ 1, a n = 3 * n - 1 ∧ b n = 2 ^ n ∧
  S_c n = (3 * n^2 + n) / 2 - 2 ^ (n + 1) + 2 :=
by
  sorry

end sequence_formulas_and_sum_l197_197854


namespace bouncy_balls_total_l197_197976

theorem bouncy_balls_total :
  let red_packs := 6
  let red_per_pack := 12
  let yellow_packs := 10
  let yellow_per_pack := 8
  let green_packs := 4
  let green_per_pack := 15
  let blue_packs := 3
  let blue_per_pack := 20
  let red_balls := red_packs * red_per_pack
  let yellow_balls := yellow_packs * yellow_per_pack
  let green_balls := green_packs * green_per_pack
  let blue_balls := blue_packs * blue_per_pack
  red_balls + yellow_balls + green_balls + blue_balls = 272 := 
by
  sorry

end bouncy_balls_total_l197_197976


namespace solution_to_problem_l197_197365

-- Definitions of conditions
def condition_1 (x : ℝ) : Prop := 2 * x - 6 ≠ 0
def condition_2 (x : ℝ) : Prop := 5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10

-- Definition of solution set
def solution_set (x : ℝ) : Prop := 3 < x ∧ x < 60 / 19

-- The theorem to be proven
theorem solution_to_problem (x : ℝ) (h1 : condition_1 x) : condition_2 x ↔ solution_set x :=
by sorry

end solution_to_problem_l197_197365


namespace arithmetic_seq_8th_term_l197_197063

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l197_197063


namespace symmetric_difference_cardinality_l197_197973

variable {α : Type}

def symmetric_difference (A B : set α) : set α := (A \ B) ∪ (B \ A)

theorem symmetric_difference_cardinality (x y : set ℕ) (hx : x.to_finset.card = 25) (hy : y.to_finset.card = 30) (hxy : (x ∩ y).to_finset.card = 12) :
  (symmetric_difference x y).to_finset.card = 31 :=
sorry

end symmetric_difference_cardinality_l197_197973


namespace workshop_workers_l197_197616

theorem workshop_workers (W : ℕ):
  (average_all : 9000 * W) →
  (average_technicians : 6 * 12000) →
  (average_nontechnicians : 6000 * (W - 6)) →
  (9000 * W = 6 * 12000 + 6000 * (W - 6)) →
  W = 12 :=
by
  sorry

end workshop_workers_l197_197616


namespace infinite_series_converges_l197_197315

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end infinite_series_converges_l197_197315


namespace value_of_coins_is_77_percent_l197_197197

theorem value_of_coins_is_77_percent :
  let pennies := 2 * 1  -- value of two pennies in cents
  let nickel := 5       -- value of one nickel in cents
  let dimes := 2 * 10   -- value of two dimes in cents
  let half_dollar := 50 -- value of one half-dollar in cents
  let total_cents := pennies + nickel + dimes + half_dollar
  let dollar_in_cents := 100
  (total_cents / dollar_in_cents) * 100 = 77 :=
by
  sorry

end value_of_coins_is_77_percent_l197_197197


namespace probability_of_one_fork_one_spoon_one_knife_l197_197897

theorem probability_of_one_fork_one_spoon_one_knife 
  (num_forks : ℕ) (num_spoons : ℕ) (num_knives : ℕ) (total_pieces : ℕ)
  (h_forks : num_forks = 7) (h_spoons : num_spoons = 8) (h_knives : num_knives = 5)
  (h_total : total_pieces = num_forks + num_spoons + num_knives) :
  (∃ (prob : ℚ), prob = 14 / 57) :=
by
  sorry

end probability_of_one_fork_one_spoon_one_knife_l197_197897


namespace sum_all_values_x_l197_197222

-- Define the problem's condition
def condition (x : ℝ) : Prop := Real.sqrt ((x - 2) ^ 2) = 9

-- Define the theorem to prove the sum of all solutions equals 4
theorem sum_all_values_x : ∑ x in {x : ℝ | condition x}, x = 4 := by
  -- Introduce the definition of condition
  sorry

end sum_all_values_x_l197_197222


namespace final_results_l197_197855

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (2, (1 / x) + a)

-- Given condition that f passes through the point (1, 1)
lemma pass_through_point (a : ℝ) : f 1 a = 1 :=
by sorry

-- Given condition that g(x) = f(x) + 2*log_2(x) has only one zero point
def g (x : ℝ) (a : ℝ) : ℝ := f x a + 2 * log (2, x)

lemma one_zero_point (a : ℝ) : ∃! x, g x a = 0 :=
by sorry

-- Given condition that a > 0 and difference between max and min of f(x) on [t, t+1] <= 1
noncomputable def Q (t : ℝ) : ℝ := ((-t + 1) / (t^2 + t))

lemma max_min_difference (a : ℝ) (t : ℝ) (ht1 : t ≥ (1 / 3)) (ht2 : t ≤ 1) : a > 0 → a ≥ Q (1 / 3) :=
by sorry

-- Stating final values and ranges
theorem final_results : (∃ (a : ℝ), a = 1) ∧ (∃ (a : ℝ), a = 0 ∨ a = -1/4) ∧ (∀a, a ≥ 3/2) :=
by
  split; sorry
  split; sorry
  sorry with sorry


end final_results_l197_197855


namespace parallel_vectors_l197_197893

variables (x : ℝ)

theorem parallel_vectors (h : (1 + x) / 2 = (1 - 3 * x) / -1) : x = 3 / 5 :=
by {
  sorry
}

end parallel_vectors_l197_197893


namespace positive_numbers_arrangement_l197_197447

theorem positive_numbers_arrangement (m n : ℕ) (R C : list ℕ) 
(hR_len : R.length = m) (hC_len : C.length = n) 
(hR_sum_eq : R.sum = C.sum) :
  ∃ (A : list ℕ), A.length < m + n ∧ 
  (∃ (matrix : matrix (fin m) (fin n) ℕ), 
  (∀ (i : fin m), (finset.univ.sum (λ j, matrix i j) = R[i])) ∧ 
  (∀ (j : fin n), (finset.univ.sum (λ i, matrix i j) = C[j]))) := 
begin
  sorry
end

end positive_numbers_arrangement_l197_197447


namespace basket_weight_l197_197047

variables 
  (B : ℕ) -- Weight of the basket
  (L : ℕ) -- Lifting capacity of one balloon

-- Condition: One balloon can lift a basket with contents weighing not more than 80 kg
axiom one_balloon_lifts (h1 : B + L ≤ 80) : Prop

-- Condition: Two balloons can lift a basket with contents weighing not more than 180 kg
axiom two_balloons_lift (h2 : B + 2 * L ≤ 180) : Prop

-- The proof problem: Determine B under the given conditions
theorem basket_weight (B : ℕ) (L : ℕ) (h1 : B + L ≤ 80) (h2 : B + 2 * L ≤ 180) : B = 20 :=
  sorry

end basket_weight_l197_197047


namespace quadratic_bound_l197_197442

theorem quadratic_bound {a b c : ℝ} (f : ℝ → ℝ) (h_f : ∀ x, |x| ≤ 1 → |f x| ≤ 1) :
  f = λ x, (a * x^2 + b * x + c) → (∀ x, |x| ≤ 1 → |2 * a * x + b| ≤ 4) :=
begin
  intros,
  sorry,
end

end quadratic_bound_l197_197442


namespace min_value_fraction_l197_197843

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ln : Real.log (a + b) = 0) :
  (2 / a + 3 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_fraction_l197_197843


namespace percentage_reduction_in_price_of_oil_l197_197736

theorem percentage_reduction_in_price_of_oil :
  ∀ P : ℝ, ∀ R : ℝ, P = 800 / (800 / R - 5) ∧ R = 40 →
  (P - R) / P * 100 = 25 := by
  -- Assumptions
  intros P R h
  have hP : P = 800 / (800 / R - 5) := h.1
  have hR : R = 40 := h.2
  -- Result to be proved
  sorry

end percentage_reduction_in_price_of_oil_l197_197736


namespace ratio_of_trapezoids_l197_197183

theorem ratio_of_trapezoids (AD AO OB BC AB DO OC : ℝ)
  (AD_eq_AO : AD = AO) (AO_eq_OB : AO = OB) (OB_eq_BC : OB = BC)
  (AD_val : AD = 15) (DO_val : DO = 20)
  (AOB_is_isosceles : triangle.isosceles AO OB = true)
  (XY_divides_AD_in_half : XY = 17.5) 
  (XY_divides_BC_in_half : XY = 17.5) :
  (let OP := sqrt (AO^2 - (AB/2)^2) in
  let height_of_ABYX := OP / 2
  ∧ let area_ABYX := 1/2 * height_of_ABYX * (AB + 35)
  ∧ let area_XYCD := 1/2 * height_of_ABYX * (35 + DO)
  ∧ let ratio := area_ABYX / area_XYCD in
  ratio = 11/15 ∧ p + q = 26) :=
begin
  sorry
end

end ratio_of_trapezoids_l197_197183


namespace chicken_legs_baked_l197_197771

theorem chicken_legs_baked (L : ℕ) (H₁ : 144 / 16 = 9) (H₂ : 224 / 16 = 14) (H₃ : 16 * 9 = 144) :  L = 144 :=
by
  sorry

end chicken_legs_baked_l197_197771


namespace problem_proof_l197_197625

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem problem_proof :
  (∀ x, g (x + Real.pi) = g x) ∧ (∀ y, g (2 * (Real.pi / 12) - y) = g y) :=
by
  sorry

end problem_proof_l197_197625


namespace cyclic_quadrilateral_APCD_l197_197030

variables (A B C H A₁ C₁ D P D' : Type) [metric_space H] [metric_space A₁] [metric_space C₁] [metric_space D] [metric_space P] [metric_space D']

def is_midpoint (B H P : H) : Prop :=
  dist B P = dist P H

def is_reflection (D AC D' : H) : Prop :=
  dist D AC = dist D' AC ∧ (⟪D, AC⟫ = ⟪AC, D'⟫)

axiom AC_eq : dist A C = dist A₁ C₁

theorem cyclic_quadrilateral_APCD' :
  (∃ (A B C H A₁ C₁ D P D' : H), 
    acute_triangle A B C ∧ orthocenter A B C H ∧
    ∃ (A₁ C₁ : H), line H A₁ ∧ B C ∧
    ∃ (D P D' : H), meets BH A₁C₁ D ∧
    is_midpoint B H P ∧
    is_reflection D AC D' ∧ 
    cyclic_quadrilateral A P C D') :=
sorry

end cyclic_quadrilateral_APCD_l197_197030


namespace A_finish_work_in_6_days_l197_197254

theorem A_finish_work_in_6_days :
  ∃ (x : ℕ), (1 / (12:ℚ) + 1 / (x:ℚ) = 1 / (4:ℚ)) → x = 6 :=
by
  sorry

end A_finish_work_in_6_days_l197_197254


namespace leon_total_payment_l197_197007

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end leon_total_payment_l197_197007


namespace max_value_of_expression_max_value_achieved_l197_197032

theorem max_value_of_expression (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
    8 * x + 3 * y + 10 * z ≤ Real.sqrt 173 :=
sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)
    (hx : x = Real.sqrt 173 / 30)
    (hy : y = Real.sqrt 173 / 20)
    (hz : z = Real.sqrt 173 / 50) :
    8 * x + 3 * y + 10 * z = Real.sqrt 173 :=
sorry

end max_value_of_expression_max_value_achieved_l197_197032


namespace max_value_of_z_plus_2_plus_i_l197_197618

-- Definition of the problem's conditions
def condition (z : ℂ) : Prop :=
  abs (z - complex.i) = 1

-- The mathematical problem rewritten as a Lean statement
theorem max_value_of_z_plus_2_plus_i (z : ℂ) (hz : condition z) :
  abs (z + 2 + complex.i) ≤ 2 * real.sqrt 2 + 1 := 
sorry

end max_value_of_z_plus_2_plus_i_l197_197618


namespace fraction_ordering_l197_197784

theorem fraction_ordering :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  (b < c) ∧ (c < a) :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  sorry

end fraction_ordering_l197_197784


namespace achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197668

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l197_197668


namespace integral_ge_2_l197_197012

open Real

variable {f : ℝ → ℝ}

noncomputable def integral_problem (f : ℝ → ℝ) :=
  ∫ (x : ℝ) in 0..2, (f (1 + x) / f x)

theorem integral_ge_2
  (hf_pos : ∀ x, 0 < f x)
  (hf_cont : Continuous f)
  (hf_per : ∀ x, f (x + 2) = f x) :
  integral_problem f ≥ 2 ∧ (integral_problem f = 2 ↔ ∀ x, f (x + 1) = f x) :=
by
  sorry

end integral_ge_2_l197_197012


namespace solve_eq1_solve_eq2_l197_197605

-- Define the theorem for the first equation
theorem solve_eq1 (x : ℝ) (h : 2 * x - 7 = 5 * x - 1) : x = -2 :=
sorry

-- Define the theorem for the second equation
theorem solve_eq2 (x : ℝ) (h : (x - 2) / 2 - (x - 1) / 6 = 1) : x = 11 / 2 :=
sorry

end solve_eq1_solve_eq2_l197_197605


namespace projection_computation_l197_197156

def vector_projection (u v : ℝ^3) : ℝ^3 :=
  let uv_dot := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
  let vv_dot := v.1 * v.1 + v.2 * v.2 + v.3 * v.3 in
  (uv_dot / vv_dot) • v

theorem projection_computation :
  let v := ⟨ -2, 3, -1.5 ⟩ : ℝ^3 in
  let w := ⟨ 4, 1, -3 ⟩ : ℝ^3 in
  vector_projection w v = ⟨ 1 / 15.25, -1.5 / 15.25, 0.75 / 15.25 ⟩ :=
by sorry

end projection_computation_l197_197156


namespace initial_red_marbles_l197_197496

theorem initial_red_marbles
    (r g : ℕ)
    (h1 : 3 * r = 5 * g)
    (h2 : 2 * (r - 15) = g + 18) :
    r = 34 := by
  sorry

end initial_red_marbles_l197_197496


namespace arithmetic_sequence_eighth_term_l197_197094

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197094


namespace scientific_notation_of_53_96_billion_l197_197768

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end scientific_notation_of_53_96_billion_l197_197768


namespace solution_set_is_R_l197_197869

theorem solution_set_is_R 
  (b c : ℝ) 
  (h1 : ∀ x, x ∈ Set.Ioo (-1) 2 ↔ (x^2 + b * x + c < 0)) : 
  ∀ x, x ∈ Set.univ ↔ (b * x^2 + x + c < 0) := 
begin
  sorry
end

end solution_set_is_R_l197_197869


namespace interval_of_monotonic_increasing_l197_197912

def piecewise_function (a : ℝ) : ℝ → ℝ :=
  λ x, if x > 1 then a ^ x else (4 - a / 2) * x + 2

theorem interval_of_monotonic_increasing (a : ℝ) (hf : ∀ x y, x ≤ y → piecewise_function a x ≤ piecewise_function a y) : 
  4 ≤ a ∧ a < 8 := 
sorry

end interval_of_monotonic_increasing_l197_197912


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197675

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197675


namespace constructible_triangle_l197_197788

theorem constructible_triangle (k c delta : ℝ) (h1 : 2 * c < k) :
  ∃ (a b : ℝ), a + b + c = k ∧ a + b > c ∧ ∃ (α β : ℝ), α - β = delta :=
by
  sorry

end constructible_triangle_l197_197788


namespace largest_distance_l197_197016

noncomputable def parabola (t : ℝ) : ℝ × ℝ := (4*t^2, 4*t)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

theorem largest_distance 
  (t : ℝ) :
  let A := (5, 0)
  let B := (4 * t^2, 4 * t)
  let AB := distance A B
  ∃ t, AB = 5 :=
begin
  use 0,
  simp only [parabola],
  unfold distance,
  simp,
  norm_num,
  sorry
end

end largest_distance_l197_197016


namespace find_f_at_1_l197_197874

noncomputable def f (y : ℝ) : ℝ := logBase 2 (sqrt ((9 * y + 5) / 2))

theorem find_f_at_1 : f 1 = 1 :=
by
  have h : f 1 = logBase 2 (sqrt ((9 * 1 + 5) / 2)) :=
    rfl
  rw [h]
  sorry

end find_f_at_1_l197_197874


namespace closest_estimate_l197_197155

theorem closest_estimate (a b : ℤ) (h₀ : a = 58) (h₁ : b = 41) :
  ∃ c d : ℤ, c = 60 ∧ d = 40 ∧ abs ((a * b) - (c * d)) ≤ abs ((a * b) - (x * y)) ∀ (x y : ℤ), ((x = 50 ∧ y = 40) ∨ (x = 60 ∧ y = 50) ∨ (x = 60 ∧ y = 40)) :=
by
  sorry

end closest_estimate_l197_197155


namespace least_value_a_l197_197910

theorem least_value_a (a : ℤ) :
  (∃ a : ℤ, a ≥ 0 ∧ (a ^ 6) % 1920 = 0) → a = 8 ∧ (a ^ 6) % 1920 = 0 :=
by
  sorry

end least_value_a_l197_197910


namespace arithmetic_sequence_8th_term_l197_197088

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l197_197088


namespace necessary_but_not_sufficient_condition_l197_197512

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (1 < k) ∧ (k < 5) → 
  (k - 1 > 0) ∧ (5 - k > 0) ∧ ((k ≠ 3) → (k < 5 ∧ 1 < k)) :=
by
  intro h
  have hk_gt_1 := h.1
  have hk_lt_5 := h.2
  refine ⟨hk_gt_1, hk_lt_5, λ hk_neq_3, ⟨hk_lt_5, hk_gt_1⟩⟩
  sorry

end necessary_but_not_sufficient_condition_l197_197512


namespace find_m_plus_n_l197_197715

theorem find_m_plus_n (PQ QR RP : ℕ) (x y : ℕ) 
  (h1 : PQ = 26) 
  (h2 : QR = 29) 
  (h3 : RP = 25) 
  (h4 : PQ = x + y) 
  (h5 : QR = x + (QR - x))
  (h6 : RP = x + (RP - x)) : 
  30 = 29 + 1 :=
by
  -- assumptions already provided in problem statement
  sorry

end find_m_plus_n_l197_197715


namespace focus_parabola_directrix_l197_197623

open Real

theorem focus_parabola_directrix 
  (F : ℝ × ℝ) (P Q : ℝ × ℝ)
  (hF : F = (2, 0))
  (C : ℝ × ℝ → Prop := λ x y, y^2 = 8*x)
  (dir : ℝ × ℝ → Prop := λ x y, x = -2)
  (h1 : dir P)
  (h2 : C Q)
  (h3 : ∃ s : ℝ, Q = (s, real.sqrt (8*s)))
  (h4 : ∃ t : ℝ, P = (-2, t))
  (hFP_FQ : (fst P - fst F)^2 + (snd P - snd F)^2 = 16 * ((fst Q - fst F)^2 + (snd Q - snd F)^2))
: |FP| = 3 := 
by
  sorry

end focus_parabola_directrix_l197_197623


namespace smallest_n_for_terminating_decimal_l197_197690

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m, m < n → (∃ k1 k2 : ℕ, (m + 150 = 2^k1 * 5^k2 ∧ m > 0) → false)) ∧ (∃ k1 k2 : ℕ, (n + 150 = 2^k1 * 5^k2) ∧ n > 0) :=
sorry

end smallest_n_for_terminating_decimal_l197_197690


namespace problem_solution_l197_197541

variable (α β : Type) [plane α] [plane β] (l m : Type) [line l] [line m]
variable (h1 : l ⊆ α) (h2 : m ⊆ β)

/- Propositions Definitions -/
def proposition1 := α ≠ β → l ≠ m
def proposition2 := l ⊥ m → α ⊥ β

theorem problem_solution :
  ¬proposition1 ∧ ¬proposition2 := by
  sorry

end problem_solution_l197_197541


namespace arithmetic_sequence_eighth_term_l197_197091

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l197_197091


namespace jake_has_more_apples_l197_197002

variables (steven_peaches steven_apples jake_peaches jake_apples : ℕ)
variables (more_apples : Prop)

-- Define the conditions
def steven_conditions :=
  steven_peaches = 19 ∧ 
  steven_apples = 14 ∧ 
  jake_peaches = steven_peaches - 12 ∧ 
  more_apples

-- Define the statement that should be proved
def apples_comparison : Prop :=
  ∃ jake_apples, jake_apples > steven_apples

-- The theorem statement
theorem jake_has_more_apples (cond : steven_conditions) : apples_comparison :=
by { sorry }

end jake_has_more_apples_l197_197002


namespace sum_infinite_series_eq_l197_197292

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l197_197292


namespace tom_total_miles_l197_197655

def monday_run_morning := 6 -- miles
def monday_run_evening := 4 -- miles
def wednesday_run_morning := 5.25 -- miles
def wednesday_run_evening := 5 -- miles
def friday_run_first := 3 -- miles
def friday_run_second := 4.5 -- miles
def friday_run_third := 2 -- miles

def tuesday_cycle_morning := 10 -- miles
def tuesday_cycle_evening := 8 -- miles
def thursday_cycle_morning := 7 -- miles
def thursday_cycle_evening := 12 -- miles

def total_running := 
  monday_run_morning + monday_run_evening + 
  wednesday_run_morning + wednesday_run_evening + 
  friday_run_first + friday_run_second + friday_run_third

def total_cycling := 
  tuesday_cycle_morning + tuesday_cycle_evening + 
  thursday_cycle_morning + thursday_cycle_evening

def total_miles := total_running + total_cycling

theorem tom_total_miles : total_miles = 66.75 :=
by
  simp [total_running, total_cycling, monday_run_morning, monday_run_evening, 
        wednesday_run_morning, wednesday_run_evening, 
        friday_run_first, friday_run_second, friday_run_third,
        tuesday_cycle_morning, tuesday_cycle_evening,
        thursday_cycle_morning, thursday_cycle_evening]
  sorry

end tom_total_miles_l197_197655


namespace defective_products_selection_l197_197422
noncomputable def possible_values_of_xi (total_products defective_products selected_products : ℕ) : Set ℕ :=
  { n | n ≤ defective_products ∧ n ≤ selected_products ∧ n ≥ 0 }

theorem defective_products_selection :
  let total_products := 8 in
  let defective_products := 2 in
  let selected_products := 3 in
  possible_values_of_xi total_products defective_products selected_products = {0, 1, 2} :=
by
  simp [possible_values_of_xi]
  sorry

end defective_products_selection_l197_197422


namespace sphere_surface_area_l197_197585

theorem sphere_surface_area (A B C O : ℝ → ℝ → ℝ → Prop)
  (h1 : dist A B = 1)
  (h2 : dist B C = 2)
  (h3 : ∠ A B C = 120 * (Real.pi / 180))
  (h4 : dist O (threePointsPlane A B C) = 3) : 
  4 * Real.pi * (sqrt (55/3))^2 = (220/3) * Real.pi :=
sorry

end sphere_surface_area_l197_197585


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197673

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l197_197673


namespace number_of_players_in_each_game_l197_197177

theorem number_of_players_in_each_game 
  (n : ℕ) (Hn : n = 30)
  (total_games : ℕ) (Htotal : total_games = 435) :
  2 = 2 :=
sorry

end number_of_players_in_each_game_l197_197177


namespace casper_initial_candies_l197_197579

theorem casper_initial_candies (x : ℕ) :
  let after_first_day := (2 * x / 3) - 2 in
  let after_second_day := (2 * after_first_day / 3) - 4 in
  after_second_day = 8 →
  x = 57 :=
by
  sorry

end casper_initial_candies_l197_197579


namespace necessary_but_not_sufficient_condition_l197_197510

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (1 < k) ∧ (k < 5) → 
  (k - 1 > 0) ∧ (5 - k > 0) ∧ ((k ≠ 3) → (k < 5 ∧ 1 < k)) :=
by
  intro h
  have hk_gt_1 := h.1
  have hk_lt_5 := h.2
  refine ⟨hk_gt_1, hk_lt_5, λ hk_neq_3, ⟨hk_lt_5, hk_gt_1⟩⟩
  sorry

end necessary_but_not_sufficient_condition_l197_197510


namespace proof_problem_l197_197033

variables {n : ℕ} (x : Fin (n) → ℝ) (S : ℝ)

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem proof_problem
  (h₀ : ∀ i, (1 ≤ i ∧ i < n) → 0 ≤ x ⟨i, sorry⟩ )
  (h₁ : ∑ k in Finset.range (n-1), (x ⟨k+1, sorry⟩) ^ (6/5) = 2 / (n * (n-1) ))
  (h₂ : ∑ k in Finset.range (n-1), x ⟨k+1, sorry⟩ = S)
  (h₃ : x ⟨0, sorry⟩ = 0) :
  ∑ k in Finset.range (n-1), (∏ i in Finset.range (k+1), (S - ∑ j in Finset.range i, x ⟨ j, sorry ⟩ )) ^ (2/(k+1))
  < (11 / 4) * binom n 2⁻¹ :=
sorry

end proof_problem_l197_197033


namespace number_of_true_propositions_is_zero_l197_197961

theorem number_of_true_propositions_is_zero (a b c : Type) 
  (H1 : ∀ (a b c : Type), (a ⊥ b) → (b ⊥ c) → (a ∥ c))
  (H2 : ∀ (a b c : Type), skew a b → skew b c → skew a c)
  (H3 : ∀ (a b c : Type), intersects a b → intersects b c → intersects a c)
  (H4 : ∀ (a b c : Type), coplanar a b → coplanar b c → coplanar a c) : 
  0 = 0 := 
sorry

end number_of_true_propositions_is_zero_l197_197961


namespace dolls_given_to_girls_correct_l197_197572

-- Define the total number of toys given
def total_toys_given : ℕ := 403

-- Define the number of toy cars given to boys
def toy_cars_given_to_boys : ℕ := 134

-- Define the number of dolls given to girls
def dolls_given_to_girls : ℕ := total_toys_given - toy_cars_given_to_boys

-- State the theorem to prove the number of dolls given to girls
theorem dolls_given_to_girls_correct : dolls_given_to_girls = 269 := by
  sorry

end dolls_given_to_girls_correct_l197_197572


namespace vasya_improved_example1_vasya_improved_example2_l197_197682

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l197_197682


namespace complement_intersection_l197_197889

namespace SetProof

variable U : Set ℕ := {1, 2, 3, 4, 5}
variable A : Set ℕ := {1, 2, 3}
variable B : Set ℕ := {1, 4}

theorem complement_intersection (U A B : Set ℕ) : ((U \ A) ∩ B) = {4} := by
  sorry

end SetProof

end complement_intersection_l197_197889


namespace louise_green_pencils_l197_197566

theorem louise_green_pencils:
  (each_box_holds : ℕ = 20)
  (red_pencils : ℕ = 20)
  (blue_pencils : ℕ = 2 * red_pencils)
  (yellow_pencils : ℕ = 40)
  (total_boxes : ℕ = 8)
  (combined_red_and_blue : ℕ = red_pencils + blue_pencils)
  (green_pencils : ℕ = combined_red_and_blue) :
  (green_pencils = (20 * 8) - (red_pencils + blue_pencils + yellow_pencils)) :=
by
  sorry

end louise_green_pencils_l197_197566


namespace num_even_perfect_square_factors_of_2_6_5_3_7_8_l197_197452

def num_even_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 2^6 * 5^3 * 7^8 then
    let valid_a := [2, 4, 6]
    let valid_c := [0, 2]
    let valid_b := [0, 2, 4, 6, 8]
    valid_a.length * valid_c.length * valid_b.length
  else 0

theorem num_even_perfect_square_factors_of_2_6_5_3_7_8 :
  num_even_perfect_square_factors (2^6 * 5^3 * 7^8) = 30 :=
by
  sorry

end num_even_perfect_square_factors_of_2_6_5_3_7_8_l197_197452


namespace vasya_problem_l197_197680

theorem vasya_problem : 
  ∃ (e : ℝ), 
  (e = 333 / 3 - 33 / 3 ∨ e = 33 * 3 + 3 / 3) ∧
  (count threes in expression e < 10) ∧
  (e = 100) :=
by sorry

end vasya_problem_l197_197680


namespace wilson_theorem_factorial_mod_13_l197_197380

-- Definition of Wilson's Theorem
theorem wilson_theorem (p : ℕ) [fact (nat.prime p)] : (p - 1)! ≡ -1 [MOD p] :=
  sorry

-- Given conditions
def p := 13

-- Main proof statement
theorem factorial_mod_13 : (10! % 13) = 6 :=
  by
    -- Apply Wilson's theorem and given conditions
    have h1 : 12! % 13 = 12! % 13 := sorry
    have h2 : 12! ≡ -1 [MOD 13] := wilson_theorem p
    sorry

end wilson_theorem_factorial_mod_13_l197_197380


namespace vasya_improved_example1_vasya_improved_example2_l197_197681

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l197_197681


namespace distance_A_to_line_l_l197_197931

-- Define the polar equation of the line
def line_eq (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = √2 / 2

-- Define point A in polar coordinates
def A_polar := (2 : ℝ, 3 * π / 4)

-- Convert point A to Cartesian coordinates
def A_cartesian : (ℝ × ℝ) :=
  let r := 2 in
  let θ := 3 * π / 4 in
  (r * cos θ, r * sin θ)

-- Define the Cartesian equation of the line
def line_cartesian (x y : ℝ) : Prop := x + y = 1

-- Define the distance function from a point to a line in Cartesian coordinates
def distance_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

-- Prove the distance from point A to line l is √2 / 2
theorem distance_A_to_line_l : 
  let (xA, yA) := A_cartesian in
  distance_to_line xA yA 1 1 (-1) = √2 / 2 :=
by
  -- Translate A to Cartesian coordinates
  let xA : ℝ := (A_cartesian).1
  let yA : ℝ := (A_cartesian).2

  -- Calculate the distance
  show distance_to_line xA yA 1 1 (-1) = √2 / 2

  -- We skip the proof here, this is a placeholder
  sorry

end distance_A_to_line_l_l197_197931
