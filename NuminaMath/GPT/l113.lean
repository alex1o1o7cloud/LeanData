import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Algebra.Time
import Mathlib.Analysis.Geometry.Ellipse
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Set
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Incenter
import Mathlib.Init.Data.Int.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Topology.Infinite.Sum
import ProbabilityTheory

namespace length_of_congruent_side_l113_113704

theorem length_of_congruent_side (b h : ℝ) (base_area_eq: b = 30) (area_eq : h = 120): ∃ (a : ℝ), a = 17 :=
by
  -- Base of the isosceles triangle
  have base_length : b = 30 := base_area_eq

  -- Given area
  have triangle_area : 120 = (1/2) * b * h := by
    rw [base_length, mul_assoc, ←mul_div_assoc, ←mul_div_assoc]
    exact area_eq
    
  -- Height of the triangle
  have height : h = 8 := by sorry

  -- Congruent side calculation using Pythagorean theorem
  use 17
  sorry

end length_of_congruent_side_l113_113704


namespace distance_between_A_and_B_l113_113412

theorem distance_between_A_and_B 
  (v t t1 : ℝ)
  (h1 : 5 * v * t + 4 * v * t = 9 * v * t)
  (h2 : t1 = 10 / (4.8 * v))
  (h3 : 10 / 4.8 = 25 / 12):
  (9 * v * t + 4 * v * t1) = 450 :=
by 
  -- Proof to be completed
  sorry

end distance_between_A_and_B_l113_113412


namespace total_feet_is_correct_l113_113797

-- definitions according to conditions
def number_of_heads := 46
def number_of_hens := 24
def number_of_cows := number_of_heads - number_of_hens
def hen_feet := 2
def cow_feet := 4
def total_hen_feet := number_of_hens * hen_feet
def total_cow_feet := number_of_cows * cow_feet
def total_feet := total_hen_feet + total_cow_feet

-- proof statement with sorry
theorem total_feet_is_correct : total_feet = 136 :=
by
  sorry

end total_feet_is_correct_l113_113797


namespace new_player_weight_l113_113958

theorem new_player_weight
  (n : ℕ) (replacement_count : ℕ) (original_weight : ℕ) (new_avg_increase : ℕ) 
  (player1_weight player2_weight : ℕ) (headcount : ℕ)
  (original_avg_weight : ℕ)
  (h1 : n = 12)
  (h2 : original_avg_weight = 80)
  (h3 : player1_weight = 65)
  (h4 : player2_weight = 75)
  (h5 : replacement_count = 1)
  (h6 : new_avg_increase = 2.5)
  : (replacement_count * (player1_weight + player2_weight) + (replacement_count * headcount * new_avg_increase)) = 170 := 
sorry

end new_player_weight_l113_113958


namespace tangent_line_equation_at_point_l113_113911

/-- Given a function f : ℝ → ℝ defined by f(x) = x^2 + x - 1
    and a point (1,1) on its graph, prove that the tangent line
    at this point has the equation 3x - y - 2 = 0. -/
theorem tangent_line_equation_at_point :
  ∀ (f : ℝ → ℝ), f = (λ x, x^2 + x - 1) →
  ∀ (x y : ℝ), (x = 1 ∧ y = 1 ∧ y = f x) →
  (∃ (a b c : ℝ), (3*x - y - 2 = 0)) :=
by
  sorry

end tangent_line_equation_at_point_l113_113911


namespace vector_line_form_to_slope_intercept_l113_113537

variable (x y : ℝ)

theorem vector_line_form_to_slope_intercept :
  (∀ (x y : ℝ), ((-1) * (x - 3) + 2 * (y + 4) = 0) ↔ (y = (-1/2) * x - 11/2)) :=
by
  sorry

end vector_line_form_to_slope_intercept_l113_113537


namespace automobile_credit_percentage_at_end_of_year_x_l113_113166

noncomputable def calculate_percentage (auto_finance_credit : ℝ) (consumer_credit : ℝ) : ℝ :=
  let total_auto_credit := 3 * auto_finance_credit
  (total_auto_credit / consumer_credit) * 100

theorem automobile_credit_percentage_at_end_of_year_x :
  let auto_finance_credit := 35
  let total_consumer_credit := 291.6666666666667
  calculate_percentage auto_finance_credit total_consumer_credit = 36 := by
sorry

end automobile_credit_percentage_at_end_of_year_x_l113_113166


namespace xi_expectation_minimum_rounds_needed_l113_113041

-- Definition and conditions for Part (1)
def xi_prob (C : ℕ → ℕ → ℚ) (n₁ n₂ k : ℕ) : ℚ :=
  if k = 0 then (C 7 3) / (C 12 3)
  else if k = 1 then (C 5 1) * (C 7 2) / (C 12 3)
  else if k = 2 then (C 5 2) * (C 7 1) / (C 12 3)
  else (C 5 3) / (C 12 3)

noncomputable def E_xi : ℚ :=
  0 * (xi_prob (λ x y => (nat.factorial x) / ((nat.factorial y) * (nat.factorial (x - y)))) 12 5 0) +
  1 * (xi_prob (λ x y => (nat.factorial x) / ((nat.factorial y) * (nat.factorial (x - y)))) 12 5 1) +
  2 * (xi_prob (λ x y => (nat.factorial x) / ((nat.factorial y) * (nat.factorial (x - y)))) 12 5 2) +
  3 * (xi_prob (λ x y => (nat.factorial x) / ((nat.factorial y) * (nat.factorial (x - y)))) 12 5 3)

theorem xi_expectation : E_xi = 5 / 4 :=
  sorry

-- Definition and conditions for Part (2)
def win_probability (p1 p2 : ℚ) : ℚ :=
  2 * p1 * p2 * (p1 + p2) - 3 * (p1 * p2)^2

noncomputable def minimum_rounds (p1 p2 : ℚ) (w : ℚ) : ℚ :=
  6 / (win_probability p1 p2)

theorem minimum_rounds_needed : minimum_rounds (2 / 3) (2 / 3) 6 = 11 :=
  sorry

end xi_expectation_minimum_rounds_needed_l113_113041


namespace blankets_collected_l113_113230

theorem blankets_collected (team_size : ℕ) (first_day_each_person : ℕ) (multiplier_second_day : ℕ) (third_day_total : ℕ) :
  team_size = 15 → first_day_each_person = 2 → multiplier_second_day = 3 → third_day_total = 22 →
  (team_size * first_day_each_person + (team_size * first_day_each_person * multiplier_second_day) + third_day_total) = 142 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end blankets_collected_l113_113230


namespace paintable_wall_area_correct_l113_113989

noncomputable def paintable_wall_area : Nat :=
  let length := 15
  let width := 11
  let height := 9
  let closet_width := 3
  let closet_length := 4
  let unused_area := 70
  let room_wall_area :=
    2 * (length * height) +
    2 * (width * height)
  let closet_wall_area := 
    2 * (closet_width * height)
  let paintable_area_per_bedroom := 
    room_wall_area - (unused_area + closet_wall_area)
  4 * paintable_area_per_bedroom

theorem paintable_wall_area_correct : paintable_wall_area = 1376 := by
  sorry

end paintable_wall_area_correct_l113_113989


namespace avg_salary_rest_of_workers_l113_113046

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_technicians : ℝ) (total_workers : ℕ) (n_technicians : ℕ) (avg_rest : ℝ) :
  avg_all = 8000 ∧ avg_technicians = 20000 ∧ total_workers = 49 ∧ n_technicians = 7 →
  avg_rest = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l113_113046


namespace cos_identity_l113_113421

theorem cos_identity (α : ℝ) : 
  3.4028 * (Real.cos α)^4 + 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 - 3 * (Real.cos α) + 1 = 
  2 * (Real.cos (7 * α / 2)) * (Real.cos (α / 2)) := 
by sorry

end cos_identity_l113_113421


namespace proof_inequality_l113_113354

noncomputable def g (x : ℝ) : ℝ := 
  (3 * x^2 - x) / (1 + x^2)

theorem proof_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_xyz : x + y + z = 1) : 
  g(x) + g(y) + g(z) ≥ 0 := 
sorry

end proof_inequality_l113_113354


namespace intersection_of_A_and_B_l113_113258

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l113_113258


namespace find_leftmost_vertex_l113_113563

theorem find_leftmost_vertex {n : ℕ} (h₁ : 4 ≤ n) (h₂ : 4 > 0)
  (h₃ : log (n + 1) + log (n + 2) - log (n * (n + 3)) = log (91 / 90)) :
  n = 12 :=
by
  sorry

end find_leftmost_vertex_l113_113563


namespace xiaoqiang_xiaohua_meeting_probability_l113_113106

noncomputable def probability_of_meeting : ℝ :=
  17 / 24

theorem xiaoqiang_xiaohua_meeting_probability :
  let xiaoqiang_arrival_range := 100 < x ∧ x < 120,
      xiaohua_arrival_range := 60 < y ∧ y < 120,
      waiting_condition := |x - y| ≤ 10,
      total_area := 1200,
      meeting_area := (1/2) * 10 * 10 + (30 + 50) * 20 / 2 in
  probability_of_meeting = meeting_area / total_area :=
by
  sorry

end xiaoqiang_xiaohua_meeting_probability_l113_113106


namespace fleet_total_distance_l113_113131

theorem fleet_total_distance (n : ℕ) (a d : ℤ) (speed : ℤ) (departure_interval : ℤ) (total_hours : ℤ) :
  n = 15 → a = 300 → d = -10 → speed = 60 → departure_interval = 10 → total_hours = 5 →
  let total_distance : ℤ := n * a + (n * (n-1) / 2) * d in
  total_distance = 3450 := by
  sorry

end fleet_total_distance_l113_113131


namespace correct_derivative_statements_l113_113395

theorem correct_derivative_statements :
  let y1 := (λ x : ℝ, Real.log 2)
  let y2 := (λ x : ℝ, Real.sqrt x)
  let y3 := (λ x : ℝ, Real.exp (-x))
  let y4 := (λ x : ℝ, Real.cos x)
  let dy1 := (λ x : ℝ, 0)
  let dy2 := (λ x : ℝ, 1 / (2 * Real.sqrt x))
  let dy3 := (λ x : ℝ, -Real.exp (-x))
  let dy4 := (λ x : ℝ, -Real.sin x)
  (y1' 0 = dy1 0) = false ∧ 
  (y2' = dy2) = true ∧ 
  (y3' = dy3) = true ∧ 
  (y4' = -Real.sin) = false -> 2 = 2 :=
by {
  let y1 := (λ x : ℝ, Real.log 2),
  let y2 := (λ x : ℝ, Real.sqrt x),
  let y3 := (λ x : ℝ, Real.exp (-x)),
  let y4 := (λ x : ℝ, Real.cos x),
  let dy1 := (λ x : ℝ, 0),
  let dy2 := (λ x : ℝ, 1 / (2 * Real.sqrt x)),
  let dy3 := (λ x : ℝ, -Real.exp (-x)),
  let dy4 := (λ x : ℝ, -Real.sin x),
  have h1 : deriv y1 = 0,
  { apply differentiable_const, },
  have h2 : ∀ x : ℝ, deriv y2 x = 1 / (2 * Real.sqrt x),
  { intro x, exact differentiable_at.sqrt, },
  have h3 : ∀ x : ℝ, deriv y3 x = -Real.exp (-x),
  { intro x, exact differentiable_at.exp_neg, },
  have h4 : ∀ x : ℝ, deriv y4 x = -Real.sin x,
  { intro x, exact differentiable_at.sin, },
  have correct_count : 2 = 2, sorry
}

end correct_derivative_statements_l113_113395


namespace sequence_general_term_l113_113982

theorem sequence_general_term {a : ℕ → ℚ} 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n ≥ 2, a n = 3 * a (n - 1) / (a (n - 1) + 3)) : 
  ∀ n, a n = 3 / (n + 2) :=
by
  sorry

end sequence_general_term_l113_113982


namespace apples_left_l113_113648

theorem apples_left (initial_apples : ℕ)
  (percent_sold_to_Alice : ℕ)
  (percent_given_to_Bob : ℕ)
  (donated_to_charity : ℕ)
  (h_initial_apples : initial_apples = 150)
  (h_percent_sold_to_Alice : percent_sold_to_Alice = 30)
  (h_percent_given_to_Bob : percent_given_to_Bob = 20)
  (h_donated_to_charity : donated_to_charity = 2) :
  let sold_to_Alice := percent_sold_to_Alice * initial_apples / 100,
    remaining_after_Alice := initial_apples - sold_to_Alice,
    given_to_Bob := percent_given_to_Bob * remaining_after_Alice / 100,
    remaining_after_Bob := remaining_after_Alice - given_to_Bob,
    final_remaining := remaining_after_Bob - donated_to_charity
  in final_remaining = 82 :=
by {
  sorry
}

end apples_left_l113_113648


namespace farm_animal_count_l113_113799

theorem farm_animal_count :
  (∃ c d a : ℕ, 
    d = 3 * c ∧ 
    a = 5 * (d + c) ∧ 
    a = 130 ∧ 
    ∃ s h t : ℕ, 
      s = 1/4 * a ∧ 
      h = 3/5 * d ∧ 
      t + s < 130 ∧
      h = (3/5 * d).toNat ∧
      ∃ d' d'' : ℕ,
        d'' = d - h ∧
        d' = d'' + h ∧ 
        t = 1/2 * d' ∧
        ∃ c' : ℕ,
          c' = 2 * (a + d' + c) ∧
          ∃ c'' : ℕ,
            c'' = c' + 1/2 * c' ∧
            555 = a + d' + c + t + c'' ) :=
sorry

end farm_animal_count_l113_113799


namespace diff_eq_40_l113_113611

theorem diff_eq_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end diff_eq_40_l113_113611


namespace bridge_length_proof_l113_113056

/-- Train and bridge length problem -/
def train_bridge_length (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℕ) : ℝ :=
  let train_speed_mps := (train_speed_kmph * 1000) / 3600
  let total_distance := train_speed_mps * time_seconds
  total_distance - train_length

theorem bridge_length_proof :
  train_bridge_length 130 54 30 = 320 :=
by
  sorry

end bridge_length_proof_l113_113056


namespace coefficient_x3_in_expansion_l113_113207

theorem coefficient_x3_in_expansion : 
  nat.choose 5 3 * (1 : ℤ)^(5 - 3) * (-2)^3 = -80 :=
by
  sorry

end coefficient_x3_in_expansion_l113_113207


namespace solve_for_x_l113_113874

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 6 * x + 5

theorem solve_for_x : ∀ x : ℝ, delta (phi x) = -1 → x = - 16 / 15 :=
by
  intro x
  intro h
  -- Proof skipped
  sorry

end solve_for_x_l113_113874


namespace find_k_l113_113591

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b (k : ℝ) : ℝ × ℝ := (-3, k)

-- Define dot product for 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Condition for perpendicular vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The statement to prove
theorem find_k : ∀ k : ℝ, perpendicular vector_a (vector_b k) → k = 9 :=
by
  intro k h,
  -- proof will be inserted here
  sorry

end find_k_l113_113591


namespace find_number_l113_113785

theorem find_number (N : ℕ) (hN : (200 ≤ Nat.digits 10 N.length ∧ Nat.digits 10 N.length ≤ 200)) :
  (∃ a : ℕ, a ∈ {1, 2, 3} ∧ N = 125 * a * 10 ^ 197) :=
by
  sorry

end find_number_l113_113785


namespace two_pow_1000_mod_17_l113_113769

theorem two_pow_1000_mod_17 : 2^1000 % 17 = 0 :=
by {
  sorry
}

end two_pow_1000_mod_17_l113_113769


namespace math_proof_problem_l113_113279

noncomputable def f (x α : ℝ) := sin(x - α) + 2 * cos x
noncomputable def h (α : ℝ) := 5 - 4 * sin α

theorem math_proof_problem : 
  (∃ α : ℝ, ∀ x : ℝ, f x α = f (-x) α) ∧          -- Existence of α such that f(x) is even function
  (¬ ∃ α : ℝ, ∀ x : ℝ, f x α = -f (-x) α) ∧       -- Non-existence of α such that f(x) is odd function
  (¬ ∃ x α : ℝ, f x α = -3) ∧                     -- The minimum value of f(x) is not -3
  (∃ α : ℝ, ∃ x : ℝ, h α = 3) ∧                   -- The maximum value of h(α) is 3
  (∃ α : ℝ, (α = π / 6) → f (- π / 3) α = 0)      -- For α = π / 6, (-π/3, 0) is symmetry center
:= sorry

end math_proof_problem_l113_113279


namespace min_n_minus_m_l113_113578

noncomputable def f : ℝ → ℝ := λ x => Real.exp (4 * x - 1)
noncomputable def g : ℝ → ℝ := λ x => 1 / 2 + Real.log (2 * x)

theorem min_n_minus_m : 
  ∀ (m n : ℝ), f m = g n → ∃ t > 0, f m = g n = t ∧ (n - m = (1 + Real.log 2) / 4) := 
by 
  intros m n h,
  rw [f, g] at h,
  sorry

end min_n_minus_m_l113_113578


namespace find_m_n_sum_l113_113002

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end find_m_n_sum_l113_113002


namespace piecewise_function_value_l113_113356

def g (x : ℝ) : ℝ :=
if x ≤ 0 then Real.exp x else Real.log x

theorem piecewise_function_value :
  g (g (1 / 2)) = 1 / 2 :=
by
  -- Assuming the conditions provided
  sorry

end piecewise_function_value_l113_113356


namespace max_segments_one_length_max_nine_segments_one_length_l113_113988

-- Definitions related to the problem
structure Pentagonal (A B C D E O : Type*) :=
(convex : convex A B C D E)
(internal_point : O ∈ interior (convex_hull ∪{A, B, C, D, E}))
-- Note: Here you might need to define convex Hull and interior accurately for Lean definitions.

-- Theorem statement without the proof
theorem max_segments_one_length (A B C D E O : Type*) (h : Pentagonal A B C D E O) : 
  (∃ s : set (segment (A ∪ B ∪ C ∪ D ∪ E ∪ O)), 
     card s = 10 ∧ (∀ seg ∈ s, length seg = 1)) → false := 
sorry

-- The conclusion part can explicitly state the proposition that no more than 9 segments
theorem max_nine_segments_one_length (A B C D E O : Type*) (h : Pentagonal A B C D E O) : ∃ s : set (segment (A ∪ B ∪ C ∪ D ∪ E ∪ O)), 
  card s = 9 ∧ (∀ seg ∈ s, length seg = 1) := 
sorry

end max_segments_one_length_max_nine_segments_one_length_l113_113988


namespace sin_graph_shift_l113_113409

theorem sin_graph_shift (x : ℝ) : (∃ k : ℝ, (∀ x, sin (3 * (x - k)) = sin (3 * x - 2)) ∧ k = 2/3) :=
by {
  use 2/3,
  split,
  {
    intro x,
    calc
      sin (3 * (x - 2 / 3)) = sin (3 * x - 2) : by rw [mul_sub, mul_div_cancel' 3 (by norm_num)]
  },
  {
    simp
  }
}

end sin_graph_shift_l113_113409


namespace value_of_x_plus_y_l113_113949

theorem value_of_x_plus_y (x y : ℤ) (hx : x = -3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end value_of_x_plus_y_l113_113949


namespace sampling_scheme_exists_l113_113838

theorem sampling_scheme_exists : 
  ∃ (scheme : List ℕ → List (List ℕ)), 
    ∀ (p : List ℕ), p.length = 100 → (scheme p).length = 20 :=
by
  sorry

end sampling_scheme_exists_l113_113838


namespace frequency_total_students_l113_113414

noncomputable def total_students (known : ℕ) (freq : ℝ) : ℝ :=
known / freq

theorem frequency_total_students (known : ℕ) (freq : ℝ) (h1 : known = 40) (h2 : freq = 0.8) :
  total_students known freq = 50 :=
by
  rw [total_students, h1, h2]
  norm_num

end frequency_total_students_l113_113414


namespace botanical_garden_heights_l113_113466

def tree_heights_arithmetic (h2 : ℕ) (inc : ℕ) (dec : ℕ) (h1 h2 h3 h4 h5 : ℕ) : Prop :=
  (h1 + inc = h2 ∨ h1 - dec = h2) ∧ (h2 + inc = h3 ∨ h2 - dec = h3) ∧ 
  (h3 + inc = h4 ∨ h3 - dec = h4) ∧ (h4 + inc = h5 ∨ h4 - dec = h5)

noncomputable def average_height_condition (h1 h2 h3 h4 h5 : ℕ) (avg_decimal : ℚ) : Prop :=
  (h1 + h2 + h3 + h4 + h5) / 5 = k + avg_decimal

theorem botanical_garden_heights :
  ∃ h1 h3 h4 h5, tree_heights_arithmetic 15 5 3 h1 15 h3 h4 h5 ∧ average_height_condition h1 15 h3 h4 h5 0.4 := sorry

end botanical_garden_heights_l113_113466


namespace cosine_identity_example_l113_113598

theorem cosine_identity_example {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 3) : Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by sorry

end cosine_identity_example_l113_113598


namespace train_length_l113_113391
-- Import all necessary libraries from Mathlib

-- Define the given conditions and prove the target
theorem train_length (L_t L_p : ℝ) (h1 : L_t = L_p) (h2 : 54 * (1000 / 3600) * 60 = 2 * L_t) : L_t = 450 :=
by
  -- Proof goes here
  sorry

end train_length_l113_113391


namespace max_area_rect_bamboo_fence_l113_113750

theorem max_area_rect_bamboo_fence (a b : ℝ) (h : a + b = 10) : a * b ≤ 24 :=
by
  sorry

end max_area_rect_bamboo_fence_l113_113750


namespace friend_spending_l113_113420

-- Definitions based on conditions
def total_spent (you friend : ℝ) : Prop := you + friend = 15
def friend_spent (you friend : ℝ) : Prop := friend = you + 1

-- Prove that the friend's spending equals $8 given the conditions
theorem friend_spending (you friend : ℝ) (htotal : total_spent you friend) (hfriend : friend_spent you friend) : friend = 8 :=
by
  sorry

end friend_spending_l113_113420


namespace area_IXJY_l113_113653

variable (A B C D I J X Y : Point)
variable (AB CD AD BC AJ BI DJ CI : Line)
variable [IsRectangle ABCD]
variable [Midpoint I AD]
variable [Midpoint J BC]
variable [Intersection X AJ BI]
variable [Intersection Y DJ CI]
variable [Area ABCD = 4]

theorem area_IXJY : Area IXJY = 1 := by
  sorry

end area_IXJY_l113_113653


namespace meaning_of_r_abs_greater_than_r_0_05_l113_113394

-- Definitions for the terms used
def r : ℝ := sorry
def r_0_05 : ℝ := sorry

-- Statement of the proof problem
theorem meaning_of_r_abs_greater_than_r_0_05 (r : ℝ) (r_0_05 : ℝ) :
  |r| > r_0_05 → "An event with a probability of less than 5% occurred in another trial." :=
sorry

end meaning_of_r_abs_greater_than_r_0_05_l113_113394


namespace traffic_to_driving_ratio_l113_113408

-- Define the given conditions
def driving_time : ℝ := 5
def total_trip_time : ℝ := 15

-- Define the time spent in traffic as total trip time minus driving time
def traffic_time : ℝ := total_trip_time - driving_time

-- Define the ratio as traffic time divided by driving time
def ratio : ℝ := traffic_time / driving_time

-- Theorem stating that the ratio is 2
theorem traffic_to_driving_ratio : ratio = 2 := 
by
  -- Use the given conditions and perform the arithmetic operations
  have h1 : traffic_time = 10 := by
    unfold traffic_time
    exact sub_eq_add_neg _ _
    exact rfl
  
  have h2 : ratio = traffic_time / driving_time := by rw [ratio]
  have h3 : ratio = 10 / 5 := by rw [h2, h1]
  have h4 : ratio = 2 := by
    rw [h3]
    norm_num
  exact h4

end traffic_to_driving_ratio_l113_113408


namespace inequality_proof_equality_condition_l113_113682

theorem inequality_proof (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) ≥ (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) := 
sorry

theorem equality_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) = (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) ↔ a = b ∧ b = c := 
sorry

end inequality_proof_equality_condition_l113_113682


namespace eccentricity_of_ellipse_l113_113666

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
variables (x y : ℝ) (P : ℝ × ℝ) (F1 F2 I : ℝ × ℝ)

def ellipse_eq (a b : ℝ) (P : ℝ × ℝ) := 
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1

def incenter (I : ℝ × ℝ) := 
  S_triangle_IPF1 + S_triangle_IPF2 = 2 * S_triangle_IF1F2

theorem eccentricity_of_ellipse (hP : ellipse_eq a b P) (h_incenter : incenter I)
    : e = 1 / 2 :=
sorry

end eccentricity_of_ellipse_l113_113666


namespace domain_of_f_l113_113218

def domain_f (x : ℝ) : Prop := f x = Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, (domain_f x → -1 ≤ x ∧ x ≤ 1) ∧ (0 ≤ Real.arcsin (x^2) ∧ Real.arcsin (x^2) ≤ Real.pi / 2) :=
by 
  sorry -- Proof to be done.

end domain_of_f_l113_113218


namespace number_of_people_l113_113463

open Nat

theorem number_of_people (n : ℕ) (h : n^2 = 100) : n = 10 := by
  sorry

end number_of_people_l113_113463


namespace count_pairs_satisfying_sum_property_l113_113841

theorem count_pairs_satisfying_sum_property :
  ∃ n, n = 6 ∧ (∃ (pairs : List (ℕ × ℕ)),
    (∀ p ∈ pairs, let (a, b) := p in a + b = 13 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
    pairs.length = n) :=
by
  sorry

end count_pairs_satisfying_sum_property_l113_113841


namespace log21_requires_additional_information_l113_113533

noncomputable def log3 : ℝ := 0.4771
noncomputable def log5 : ℝ := 0.6990

theorem log21_requires_additional_information
  (log3_given : log3 = 0.4771)
  (log5_given : log5 = 0.6990) :
  ¬ (∃ c₁ c₂ : ℝ, log21 = c₁ * log3 + c₂ * log5) :=
sorry

end log21_requires_additional_information_l113_113533


namespace triangle_AMN_equilateral_l113_113514

-- Define the given conditions
variables {A B C D M N : Type*} [geom : EuclideanGeometry A B C D M N]
include geom

-- Parallelogram properties
axiom parallelogram_properties (hABCD : Parallelogram A B C D) :
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  (A - B) = (D - C) ∧ (A - D) = (B - C) ∧
  Angle A B C = Angle D C A ∧
  DiagonalBisects A B C D

-- Equilateral triangles on sides BC and CD
axiom equilateral_triangles_on_sides (hBCM : EquilateralTriangle B C M)
                                      (hCDN : EquilateralTriangle C D N) :
  (B - C) = (C - M) ∧ (C - D) = (D - N) ∧ 
  (angle B C M = 60° ∧ angle C D N = 60°)

-- Prove that triangle AMN is equilateral
theorem triangle_AMN_equilateral (hABCD : Parallelogram A B C D)
                                  (hBCM : EquilateralTriangle B C M)
                                  (hCDN : EquilateralTriangle C D N) :
  EquilateralTriangle A M N :=
by
  sorry

end triangle_AMN_equilateral_l113_113514


namespace moving_line_passes_through_fixed_point_given_ratio_PQ_l113_113286

variable (k : ℝ) (C : ℝ → ℝ) (L : ℝ → ℝ)

-- Define the Parabola C
def parabola (x : ℝ) : ℝ := 1 / 2 * x^2

-- Define the Line L
def line (x : ℝ) : ℝ := k * x - 1

theorem moving_line_passes_through_fixed_point (P : ℝ → ℝ) :
  ∀ t : ℝ, ∃ Q : ℝ × ℝ, (Q = (k, 1)) ∧ ∀ A B : ℝ × ℝ, 
  (parabola A.1 = A.2) ∧ (parabola B.1 = B.2) ∧ 
  let L : ℝ → ℝ := λ x, P x
  in L t = parabola t ∧ L t = A.2 ∧ L t = B.2 ∧ 
  ∃ y : ℝ, y = L t → Q =
  (k, 1) → ∃ AB : ℝ → ℝ, AB t = parabola t := sorry

theorem given_ratio_PQ (P Q : ℝ → ℝ) (M N : ℝ × ℝ) :
  ∀ t : ℝ, P t = k * t - 1 → Q t = k ->
  ∃ M N : ℝ × ℝ, (parabola M.1 = M.2) ∧ (parabola N.1 = N.2) ∧
  M ∈ parabola ∧ N ∈ parabola ∧
  let PQ : ℝ → ℝ := λ x, (P x - Q x)
  in (P Q M N : ℝ → ℝ), 
  ( M.1 - N.1 ≠ 0 ) ∧ 
  ( P M ≠ P N / M.2 ≠ N.2 ) ∧
  ( P M / P N =  Q M / Q N ) := sorry

end moving_line_passes_through_fixed_point_given_ratio_PQ_l113_113286


namespace minimum_expression_value_l113_113118

theorem minimum_expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_sum : a + b + c + d = 4) :
  (∃ x, x = (1/2) ∨
   (∃ y, y ≤ (∑ e in [a, b, c, d], e^8 / prod (λ f, if f = e then (e^2 + 1) else f^2 + e) [a, b, c, d]) ∧ y = x)) := sorry

end minimum_expression_value_l113_113118


namespace samantha_routes_l113_113690

theorem samantha_routes : 
    ∃ (house_swpark_ways northeast_school_ways central_park_ways : ℕ),
        house_swpark_ways = Nat.choose 5 2 ∧ 
        northeast_school_ways = Nat.choose 6 3 ∧ 
        central_park_ways = 2 ∧
        house_swpark_ways * central_park_ways * northeast_school_ways = 400 := 
by
  use Nat.choose 5 2, Nat.choose 6 3, 2,
  simp,
  sorry

end samantha_routes_l113_113690


namespace length_of_AC_l113_113128

-- Define the variables and context
variables (O A B C : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C]
variables (r : ℝ) (AC : ℝ) (OB : ℝ)

-- Define the conditions
def circle_area (r : ℝ) : Prop := π * r ^ 2 = 50 * π
def right_angled_triangle (A B C : Type) (hAC : A ≠ C) (H : A ≠ B) (H1 : B ≠ C) : Prop :=
  ∃ O: A, ∃ OA OB OC: ℝ, AC = 2 * r ∧ OB = 5 ∧ right_o = true

-- The theorem to prove
theorem length_of_AC (O A B C : Type) (r AC OB : ℝ) :
  circle_area r ∧ right_angled_triangle A B C ∧ OB = 5 → AC = 10 * sqrt(2) :=
sorry

end length_of_AC_l113_113128


namespace max_cities_visited_l113_113364

theorem max_cities_visited (n k : ℕ) (h0 : 1 ≤ n) (h1 : k < n) : 
  (if k ≤ n - 3 then n - k else 2) :=
sorry

end max_cities_visited_l113_113364


namespace find_a_values_l113_113188

theorem find_a_values (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_lt : a < b)
  (h_eq : Real.sqrt a + Real.sqrt b = Real.sqrt 2160) :
  a ∈ {15, 60, 135, 240, 375} := by
  sorry

end find_a_values_l113_113188


namespace solution_l113_113872

def statement_A : Prop := ¬ (A = 1)
def statement_B : Prop := (D = 1)
def statement_C : Prop := (B = 1)
def statement_D : Prop := ¬ (D = 1)
def one_truth (A B C D : Prop) : Prop :=
  A ∨ B ∨ C ∨ D ∧ ¬(A ∧ B) ∧ ¬(A ∧ C) ∧ ¬(A ∧ D) ∧ ¬(B ∧ C) ∧ ¬(B ∧ D) ∧ ¬(C ∧ D)

axiom statements : one_truth statement_A statement_B statement_C statement_D

theorem solution : A = 1 :=
by
  sorry

end solution_l113_113872


namespace polygon_vertices_l113_113034

theorem polygon_vertices (D n : ℕ) (h : D = 6) (h_formula : D = n - 3) : n = 9 := by
  rw [h, h_formula]
  linarith

end polygon_vertices_l113_113034


namespace solve_ab_find_sqrt_l113_113554

variable (a b : ℝ)

-- Given Conditions
axiom h1 : real.cbrt (2 * b - 2 * a) = -2
axiom h2 : real.sqrt (4 * a + 3 * b) = 3

-- Goal: Prove that a = 3 and b = -1
theorem solve_ab : a = 3 ∧ b = -1 := by
  sorry

-- Given a = 3 and b = -1, find the square root of 5a - b
theorem find_sqrt : a = 3 ∧ b = -1 → real.sqrt (5 * a - b) = 4 ∨ real.sqrt (5 * a - b) = -4 := by
  sorry

end solve_ab_find_sqrt_l113_113554


namespace right_triangle_area_l113_113047

theorem right_triangle_area (base hypotenuse : ℕ) (h_base : base = 8) (h_hypotenuse : hypotenuse = 10) :
  ∃ height : ℕ, height^2 = hypotenuse^2 - base^2 ∧ (base * height) / 2 = 24 :=
by
  sorry

end right_triangle_area_l113_113047


namespace limit_derivative_l113_113899

noncomputable def f : ℝ → ℝ := sorry

theorem limit_derivative (h_diff : differentiable ℝ f) :
  (Real.limit (fun Δx : ℝ => (f (1 + Δx) - f 1) / (3 * Δx)) 0) = (1 / 3) * (deriv f 1) :=
by
  sorry

end limit_derivative_l113_113899


namespace sheep_ratio_l113_113010

theorem sheep_ratio (s : ℕ) (h1 : s = 400) (h2 : s / 4 + 150 = s - s / 4) : (s / 4 * 3 - 150) / 150 = 1 :=
by {
  sorry
}

end sheep_ratio_l113_113010


namespace man_l113_113460

theorem man's_salary (S : ℝ) 
  (h_food : S * (1 / 5) > 0)
  (h_rent : S * (1 / 10) > 0)
  (h_clothes : S * (3 / 5) > 0)
  (h_left : S * (1 / 10) = 19000) : 
  S = 190000 := by
  sorry

end man_l113_113460


namespace constant_term_binomial_expansion_l113_113610

theorem constant_term_binomial_expansion (n : ℕ) (x : ℝ) : 
  (∑ k in finset.range (n + 1), binomial n k) = 64 → 
  (∑ r in finset.range (n + 1), binomial n r * (-2)^r * x^(n - 2 * r)) = (-160 : ℝ) :=
by
  sorry

end constant_term_binomial_expansion_l113_113610


namespace intersect_A_B_l113_113260

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l113_113260


namespace between_200_and_250_has_11_integers_with_increasing_digits_l113_113293

/-- How many integers between 200 and 250 have three different digits in increasing order --/
theorem between_200_and_250_has_11_integers_with_increasing_digits :
  ∃ n : ℕ, n = 11 ∧ ∀ (x : ℕ),
    200 ≤ x ∧ x ≤ 250 →
    (∀ i j k : ℕ, (x = 100 * i + 10 * j + k) →
    i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) ↔ x ∈ {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249} :=
by {
  let s := {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249},
  existsi 11,
  simp [s],
  intro x,
  split,
  intros h _,
  have hx : x ∈ s, sorry, -- proof needed
  exact hx,
  intros h _,
  split,
  intros i j k hx _,
  have hi : i = 2, sorry, -- proof needed
  cases h with j_values _,
  case or.inl : { rw j_values, exact sorry }, -- proof needed
  case or.inr : { rw j_values, exact sorry }, -- proof needed
  exact sorry -- scaffolding, proof steps go here
}

end between_200_and_250_has_11_integers_with_increasing_digits_l113_113293


namespace find_b_value_find_maximum_area_l113_113613

-- Given conditions
variables {A B C : ℝ}
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C respectively

-- Conditions of the problem
axiom angle_condition : cos B + sqrt 3 * sin B = 2
axiom side_angle_relation : (cos B / b) + (cos C / c) = (sqrt 3 * sin A) / (3 * sin C)

-- Proving b = sqrt(3)
theorem find_b_value (h1 : angle_condition) (h2 : side_angle_relation) : b = sqrt 3 := sorry

-- Proving maximum area
theorem find_maximum_area 
  (h1 : angle_condition)
  (h2 : side_angle_relation) 
  (triangle_ineq : a * c <= ((a + c)^2) / 4) 
  (area_formula : S = (1 / 2) * a * c * sin B) : 
  S <= (3 * sqrt 3) / 4 := sorry

end find_b_value_find_maximum_area_l113_113613


namespace equation_transformation_correct_l113_113101

theorem equation_transformation_correct :
  ∀ (x : ℝ), 
  6 * ((x - 1) / 2 - 1) = 6 * ((3 * x + 1) / 3) → 
  (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by
  intro x
  intro h
  sorry

end equation_transformation_correct_l113_113101


namespace value_of_m_l113_113547

-- Definition of a perfect square trinomial
def is_perfect_square_trinomial (a b x : ℝ) : Prop :=
  (a * x + b) ^ 2 = a ^ 2 * x ^ 2 + 2 * a * b * x + b ^ 2

-- Given condition
def given_condition (x m : ℝ) : Prop := 
  is_perfect_square_trinomial 1 b x ∧ b^2 = 36

-- Theorem to prove
theorem value_of_m (x m : ℝ) (h : given_condition x m) : m = 12 ∨ m = -12 :=
by sorry

end value_of_m_l113_113547


namespace range_of_a_l113_113261

-- Definitions of sets and the problem conditions
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}
def condition (a : ℝ) : Prop := P ∪ M a = P

-- The theorem stating what needs to be proven
theorem range_of_a (a : ℝ) (h : condition a) : -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l113_113261


namespace platinum_earrings_percentage_l113_113619

theorem platinum_earrings_percentage
  (rings_percentage ornaments_percentage : ℝ)
  (rings_percentage_eq : rings_percentage = 0.30)
  (earrings_percentage_eq : ornaments_percentage - rings_percentage = 0.70)
  (platinum_earrings_percentage : ℝ)
  (platinum_earrings_percentage_eq : platinum_earrings_percentage = 0.70) :
  ornaments_percentage * platinum_earrings_percentage = 0.49 :=
by 
  have earrings_percentage := 0.70
  have ornaments_percentage := 0.70
  sorry

end platinum_earrings_percentage_l113_113619


namespace original_number_of_men_l113_113107

theorem original_number_of_men 
  (x : ℕ) 
  (H1 : x * 15 = (x - 8) * 18) : 
  x = 48 := 
sorry

end original_number_of_men_l113_113107


namespace ruel_usable_stamps_l113_113028

def totalStamps (books10 books15 books25 books30 : ℕ) (stamps10 stamps15 stamps25 stamps30 : ℕ) : ℕ :=
  books10 * stamps10 + books15 * stamps15 + books25 * stamps25 + books30 * stamps30

def damagedStamps (damaged25 damaged30 : ℕ) : ℕ :=
  damaged25 + damaged30

def usableStamps (books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 damaged25 damaged30 : ℕ) : ℕ :=
  totalStamps books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 - damagedStamps damaged25 damaged30

theorem ruel_usable_stamps :
  usableStamps 4 6 3 2 10 15 25 30 5 3 = 257 := by
  sorry

end ruel_usable_stamps_l113_113028


namespace area_of_PSR_l113_113622

-- Definitions of the problem's conditions
noncomputable def triangle_is_similar (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] := 
  PQR ∼ PSR ∧ PQ = PR

def smallest_triangle_area : ℝ := 2

def number_of_smallest_triangles : ℕ := 8

def PQR_area : ℝ := 72

def PSR_is_formed_by_four_smallest_triangles : Prop := 
  ∃ (smallest_triangles : Π (i : fin 4), triangle),
    (∀ i, smallest_triangles i ⊂ PSR) ∧ 
    (∀ i j, i ≠ j → disjoint (smallest_triangles i) (smallest_triangles j))

-- The statement we need to prove
theorem area_of_PSR :
  triangle_is_similar P Q R S → smallest_triangle_area = 2 → 
  PQR_area = 72 → PSR_is_formed_by_four_smallest_triangles → 
  area PSR = 8 :=
by
  -- The proof goes here
  sorry

end area_of_PSR_l113_113622


namespace ratio_area_of_polygons_l113_113963

-- Definitions of the problem setup
variable {A B C D E F G H I J K L : Point}

-- The decagon is regular, meaning all sides and angles are equal
def decagon_regular (A B C D E F G H I J : Point) : Prop := 
  regular_decagon A B C D E F G H I J

-- K is the midpoint of CD
def midpoint_CD (K : Point) (C D : Point) : Prop := 
  K = midpoint C D

-- L is the midpoint of GH
def midpoint_GH (L : Point) (G H : Point) : Prop := 
  L = midpoint G H

-- Areas of the polygons ABJKL and EFLKJ
def area (A B J K L : Point) : ℝ := 
  polygon_area A B J K L

def area (E F L K J : Point) : ℝ := 
  polygon_area E F L K J

-- Proof statement
theorem ratio_area_of_polygons :
  ∀ {A B C D E F G H I J K L : Point},
  decagon_regular A B C D E F G H I J →
  midpoint_CD K C D →
  midpoint_GH L G H →
  (area A B J K L) / (area E F L K J) = 1 :=
sorry

end ratio_area_of_polygons_l113_113963


namespace trigonometric_identity_l113_113597

theorem trigonometric_identity 
  (α : ℝ) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := 
sorry

end trigonometric_identity_l113_113597


namespace product_sequence_l113_113804

noncomputable def a : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 2 + (a n - 2)^2

theorem product_sequence : (∏ i in (Finset.range ∞), a i) = 1 := 
by
  sorry

end product_sequence_l113_113804


namespace blueberry_pies_count_l113_113957

-- Definitions and conditions
def total_pies := 30
def ratio_parts := 10
def pies_per_part := total_pies / ratio_parts
def blueberry_ratio := 3

-- Problem statement
theorem blueberry_pies_count :
  blueberry_ratio * pies_per_part = 9 := by
  -- The solution step that leads to the proof
  sorry

end blueberry_pies_count_l113_113957


namespace monotonic_increase_in_interval_l113_113913

theorem monotonic_increase_in_interval :
  (∀ x : ℝ, -π/4 ≤ x ∧ x ≤ π/4 → (differentiable_at ℝ (λ x, sin x + cos x) x ∧ has_deriv_at (λ x, sin x + cos x) (cos x - sin x) x 
  → 0 < cos x - sin x)) ∧ 
  (∀ x : ℝ, -π/4 ≤ x ∧ x ≤ π/4 → (differentiable_at ℝ (λ x, 2 * sqrt 2 * sin x * cos x) x ∧ has_deriv_at (λ x, 2 * sqrt 2 * sin x * cos x) 
  (2 * sqrt 2 * (cos x ^ 2 - sin x ^ 2)) x → 0 < 2 * sqrt 2 * (cos x ^ 2 - sin x ^ 2))) :=
sorry

end monotonic_increase_in_interval_l113_113913


namespace find_number_l113_113777

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
sorry

end find_number_l113_113777


namespace loss_percentage_is_15_l113_113109

def cost_price : ℝ := 1500
def selling_price : ℝ := 1275
def loss := cost_price - selling_price
def loss_percentage := (loss / cost_price) * 100

theorem loss_percentage_is_15 : loss_percentage = 15 := by
  sorry

end loss_percentage_is_15_l113_113109


namespace polynomial_identity_l113_113505

open Function

-- Define the polynomial terms
def f1 (x : ℝ) := 2*x^5 + 4*x^3 + 3*x + 4
def f2 (x : ℝ) := x^4 - 2*x^3 + 3
def g (x : ℝ) := -2*x^5 + x^4 - 6*x^3 - 3*x - 1

-- Lean theorem statement
theorem polynomial_identity :
  ∀ x : ℝ, f1 x + g x = f2 x :=
by
  intros x
  sorry

end polynomial_identity_l113_113505


namespace selection_plans_count_l113_113843

def num_selection_plans : ℕ :=
  let students := ["A", "B", "C", "D", "E"]
  let restrictions := {("A", "physics"), ("A", "chemistry")}
  sorry

theorem selection_plans_count (students : List String) (restrictions : Set (String × String)) : num_selection_plans = 72 := 
  sorry

end selection_plans_count_l113_113843


namespace area_of_triangle_ABC_l113_113161

theorem area_of_triangle_ABC 
  (BD DC : ℕ) 
  (h_ratio : BD / DC = 4 / 3)
  (S_BEC : ℕ) 
  (h_BEC : S_BEC = 105) :
  ∃ (S_ABC : ℕ), S_ABC = 315 := 
sorry

end area_of_triangle_ABC_l113_113161


namespace probability_x_gt_5y_l113_113677

def point_in_rectangle (x y : ℝ) := 
  0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 3005

theorem probability_x_gt_5y :
  (∫ x in 0..3000, ∫ y in 0..min (3005 : ℝ) (x / 5), 1) / (3000 * 3005) = 200 / 20033 := 
sorry

end probability_x_gt_5y_l113_113677


namespace sqrt_a_2b_c_eq_pm4_l113_113902

theorem sqrt_a_2b_c_eq_pm4 (a b c : ℤ) 
  (h1 : Real.sqrt (2 * a - 1) = 3 ∨ Real.sqrt (2 * a - 1) = -3) 
  (h2 : Int.cbrt (3 * a + b - 9) = 2) 
  (h3 : c = Int.floor (Real.sqrt 57)) :
  Real.sqrt (a + 2 * b + c) = 4 ∨ Real.sqrt (a + 2 * b + c) = -4 := 
by
  sorry

end sqrt_a_2b_c_eq_pm4_l113_113902


namespace vector_em_l113_113891

variables {A B C E M : Type}

-- Assuming points are in a vector space
variables [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C E M : V)
variables (h1 : midpoint V A B C M)
variables (h2 : E ∈ lineSegment V A C)
variables (h3 : E - C = 2 • (E - A))

-- The statement to prove
theorem vector_em (h1 : midpoint B C M) (h2 : E ∈ lineSegment A C) (h3 : C - E = 2 • (A - E)) :
  E - M = (1 / 6 : ℝ) • (A - C) + (1 / 2 : ℝ) • (B - A) :=
sorry

end vector_em_l113_113891


namespace discount_rate_500_discount_rate_1000_min_marked_price_for_at_least_one_third_discount_max_discount_rate_in_range_l113_113198

-- Conditions from the problem
def discount_rate (marked_price : ℕ) (voucher : ℕ) : ℚ :=
  (marked_price * 0.2 + voucher) / marked_price

def voucher_amount (spent : ℕ) : ℕ :=
  if spent >= 200 ∧ spent < 400 then 30
  else if spent >= 400 ∧ spent < 500 then 60
  else if spent >= 500 ∧ spent < 700 then 100
  else if spent >= 700 ∧ spent < 900 then 130
  else 0 -- assuming the pattern continues and other ranges are handled differently

-- Discount rate calculations for 500 and 1000 yuan
theorem discount_rate_500 : discount_rate 500 (voucher_amount 400) = 0.32 := sorry

theorem discount_rate_1000 : discount_rate 1000 (voucher_amount 800) = 0.33 := sorry

-- For items priced between 500 and 1000 yuan
theorem min_marked_price_for_at_least_one_third_discount { x : ℕ } (h₁ : 500 ≤ x) (h₂ : x ≤ 1000) : x = 625 ∨ x = 875  := sorry

theorem max_discount_rate_in_range : ∀ (x : ℕ), (500 ≤ x ∧ x ≤ 1000) → discount_rate x (voucher_amount (x * 0.8)) ≤ 0.36 := sorry

end discount_rate_500_discount_rate_1000_min_marked_price_for_at_least_one_third_discount_max_discount_rate_in_range_l113_113198


namespace actual_price_of_good_before_discounts_l113_113818

noncomputable def original_price (sold_price : ℝ) (discount_applied : ℝ) : ℝ := sold_price / discount_applied

theorem actual_price_of_good_before_discounts :
  original_price 6600 0.684 = 9649.12 :=
by
  have : 6600 / 0.684 = 9649.12 := by sorry
  exact this

end actual_price_of_good_before_discounts_l113_113818


namespace sum_of_first_ten_prime_units_digit_7_is_810_l113_113225

def is_unit_digit_7 (n : ℕ) : Prop :=
  n % 10 = 7

def is_prime (n : ℕ) : Prop :=
  Prime n

def sum_of_first_ten_prime_units_digit_7 : ℕ :=
  (List.filter is_prime [7, 17, 37, 47, 67, 97, 107, 127, 137, 157, 167]).take 10).sum

theorem sum_of_first_ten_prime_units_digit_7_is_810 :
  sum_of_first_ten_prime_units_digit_7 = 810 :=
  by
    sorry

end sum_of_first_ten_prime_units_digit_7_is_810_l113_113225


namespace min_expression_n_12_l113_113072

theorem min_expression_n_12 : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (n = 12 → (n / 3 + 50 / n ≤ 
                        m / 3 + 50 / m))) :=
by
  sorry

end min_expression_n_12_l113_113072


namespace sum_of_seven_smallest_multiples_of_twelve_l113_113765

theorem sum_of_seven_smallest_multiples_of_twelve : 
  let multiples := [12, 24, 36, 48, 60, 72, 84] in
  ∑ i in multiples, i = 336 := 
by sorry

end sum_of_seven_smallest_multiples_of_twelve_l113_113765


namespace percent_decrease_price_l113_113649

-- Define the original price and sale price
def originalPrice : ℝ := 100
def salePrice : ℝ := 40

-- Define the percent decrease formula
def percentDecrease (original : ℝ) (sale : ℝ) : ℝ :=
  ((original - sale) / original) * 100

-- State the theorem to prove
theorem percent_decrease_price : 
  percentDecrease originalPrice salePrice = 60 := 
by
  sorry

end percent_decrease_price_l113_113649


namespace quadratic_equal_real_roots_l113_113245

theorem quadratic_equal_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (x = a*x / 2)) ↔ a = 2 ∨ a = -2 :=
by sorry

end quadratic_equal_real_roots_l113_113245


namespace best_model_is_model1_l113_113980

noncomputable def model_best_fitting (R1 R2 R3 R4 : ℝ) :=
  R1 = 0.975 ∧ R2 = 0.79 ∧ R3 = 0.55 ∧ R4 = 0.25

theorem best_model_is_model1 (R1 R2 R3 R4 : ℝ) (h : model_best_fitting R1 R2 R3 R4) :
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  cases h with
  | intro h1 h_rest =>
    cases h_rest with
    | intro h2 h_rest2 =>
      cases h_rest2 with
      | intro h3 h4 =>
        sorry

end best_model_is_model1_l113_113980


namespace probability_two_diff_color_chips_l113_113079

theorem probability_two_diff_color_chips (blue_chips yellow_chips : ℕ) (total_chips : ℕ) :
  blue_chips = 7 → yellow_chips = 5 → total_chips = blue_chips + yellow_chips →
  (2 * (blue_chips / total_chips) * (yellow_chips / total_chips) = 35 / 72) :=
by 
  intros h1 h2 h3
  rw [h1, h2] at *
  have h4 : total_chips = 12 := by rw [h1, h2]; exact h3
  rw h4
  sorry

end probability_two_diff_color_chips_l113_113079


namespace speed_of_cyclist_l113_113725

theorem speed_of_cyclist 
  (L B : ℕ) 
  (h1 : B = 4 * L) 
  (h2 : L * B = 102400) 
  (h3 : ∀ t : ℕ, t = 8) 
  : let perimeter := 2 * L + 2 * B,
        time := (8 : ℚ) / 60,
        distance := (perimeter : ℚ) / 1000,
        speed := distance / time in
    speed = 12 := 
by {
  sorry
}

end speed_of_cyclist_l113_113725


namespace find_fraction_l113_113126

-- Let f be a real number representing the fraction
theorem find_fraction (f : ℝ) (h : f * 12 + 5 = 11) : f = 1 / 2 := 
by
  sorry

end find_fraction_l113_113126


namespace hypotenuse_length_of_right_triangle_with_given_medians_l113_113058

theorem hypotenuse_length_of_right_triangle_with_given_medians (a b : ℝ) 
    (h1 : b^2 + (a^2 / 4) = 52) (h2 : a^2 + (b^2 / 4) = 36) : 
    √((2 * a)^2 + (2 * b)^2) = 16.8 := 
by 
  sorry

end hypotenuse_length_of_right_triangle_with_given_medians_l113_113058


namespace second_smallest_packs_hot_dogs_l113_113513

theorem second_smallest_packs_hot_dogs 
    (n : ℕ) 
    (k : ℤ) 
    (h1 : 10 * n ≡ 4 [MOD 8]) 
    (h2 : n = 4 * k + 2) : 
    n = 6 :=
by sorry

end second_smallest_packs_hot_dogs_l113_113513


namespace sin_neg_390_eq_neg_half_l113_113781

theorem sin_neg_390_eq_neg_half : sin (-390 * real.pi / 180) = -1/2 := by
  sorry

end sin_neg_390_eq_neg_half_l113_113781


namespace number_of_true_statements_l113_113867

theorem number_of_true_statements (a b c d : ℝ) :
  (if (a > b ∧ c > d) then (a + c > b + d) else true) ∧
  (if (a > b ∧ c > d) then (a * c > b * d) else true) ∧
  (if (a > b) then (1/a > 1/b) else true) ∧
  (if (c ≠ 0 ∧ a * c^2 > b * c^2) then (a > b) else true) ↔ 
  2 := by 
  sorry

end number_of_true_statements_l113_113867


namespace integer_tangent_values_zero_l113_113233

noncomputable def circle_circumference : ℝ := 15 * Real.pi

def is_mean_proportional (t₁ m n : ℝ) : Prop := t₁ = Real.sqrt (m * n)

def tangent_secant_condition (m t₁ : ℝ) : Prop := ∃ n, n = circle_circumference - m ∧ is_mean_proportional t₁ m n

theorem integer_tangent_values_zero (m t₁ : ℝ) (hm : m ∈ ℤ) (ht₁ : t₁ ∈ ℤ) : ¬ (tangent_secant_condition m t₁) :=
by sorry

end integer_tangent_values_zero_l113_113233


namespace average_percentage_reduction_l113_113739

theorem average_percentage_reduction (x : ℝ) :
  (140 * (1 - x) * (1 - x) = 35) :=
begin
  sorry
end

end average_percentage_reduction_l113_113739


namespace fixed_points_and_zeros_no_fixed_points_range_b_l113_113571

def f (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem fixed_points_and_zeros (b c : ℝ) (h1 : f b c (-3) = -3) (h2 : f b c 2 = 2) :
  ∃ x1 x2 : ℝ, f b c x1 = 0 ∧ f b c x2 = 0 ∧ x1 = -1 + Real.sqrt 7 ∧ x2 = -1 - Real.sqrt 7 :=
sorry

theorem no_fixed_points_range_b {b : ℝ} (h : ∀ x : ℝ, f b (b^2 / 4) x ≠ x) : 
  b > 1 / 3 ∨ b < -1 :=
sorry

end fixed_points_and_zeros_no_fixed_points_range_b_l113_113571


namespace red_trace_larger_sphere_area_l113_113144

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end red_trace_larger_sphere_area_l113_113144


namespace probability_heads_l113_113384

def coin_flip (penny nickel dime quarter half_dollar : bool) : Prop :=
  (penny = true) ∧ (nickel = true) ∧ (quarter = true)

theorem probability_heads : 
  let total_outcomes := 2^5 in
  let favorable_outcomes := 2 * 2 in
  let probability := favorable_outcomes / total_outcomes in
  ∃ (penny nickel dime quarter half_dollar : bool),
    coin_flip penny nickel dime quarter half_dollar →
    probability = (1 : ℚ) / 8 :=
by
  sorry

end probability_heads_l113_113384


namespace g_value_l113_113054

theorem g_value :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, x ≠ 0 → g(x) - 2 * g(1 / x) = 3^x + x) →
  ∃ g : ℝ → ℝ, g(2) = -4 - (2 * Real.sqrt 3) / 3 :=
begin
  sorry
end

end g_value_l113_113054


namespace cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l113_113450

variable (x : ℕ) (x_ge_4 : x ≥ 4)

-- Total cost under scheme ①
def scheme_1_cost (x : ℕ) : ℕ := 5 * x + 60

-- Total cost under scheme ②
def scheme_2_cost (x : ℕ) : ℕ := 9 * (80 + 5 * x) / 10

theorem cost_scheme_1 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_1_cost x = 5 * x + 60 :=  
sorry

theorem cost_scheme_2 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_2_cost x = (80 + 5 * x) * 9 / 10 := 
sorry

-- When x = 30, compare which scheme is more cost-effective
variable (x_eq_30 : x = 30)
theorem cost_comparison_scheme (x_eq_30 : x = 30) : 
  scheme_1_cost 30 > scheme_2_cost 30 := 
sorry

-- When x = 30, a more cost-effective combined purchasing plan
def combined_scheme_cost : ℕ := scheme_1_cost 4 + scheme_2_cost (30 - 4)

theorem more_cost_effective_combined_plan (x_eq_30 : x = 30) : 
  combined_scheme_cost < scheme_1_cost 30 ∧ combined_scheme_cost < scheme_2_cost 30 := 
sorry

end cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l113_113450


namespace integral_x_div_sin_squared_l113_113186

-- Problem Statement: Prove the indefinite integral of x / sin^2 x.
theorem integral_x_div_sin_squared : 
  ∫ (x : ℝ) in 0..1, x / (sin x)^2 = -x * cot x + log (abs (sin x)) + C :=
sorry

end integral_x_div_sin_squared_l113_113186


namespace find_ratio_l113_113661

noncomputable def complex_numbers_are_non_zero (x y z : ℂ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0

noncomputable def sum_is_30 (x y z : ℂ) : Prop :=
x + y + z = 30

noncomputable def expanded_equality (x y z : ℂ) : Prop :=
((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z

theorem find_ratio (x y z : ℂ)
  (h1 : complex_numbers_are_non_zero x y z)
  (h2 : sum_is_30 x y z)
  (h3 : expanded_equality x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3.5 :=
sorry

end find_ratio_l113_113661


namespace new_messages_per_day_l113_113670

-- Definitions based on conditions
def initial_unread := 98
def daily_read := 20
def total_days := 7

-- Definition of the unknown
def x : ℕ := 6

-- Statement of the problem to prove
theorem new_messages_per_day : (initial_unread - total_days * daily_read + total_days * x = 0) ↔ (x = 6) :=
by
  have h : initial_unread - total_days * daily_read + total_days * x = 0 ↔ -42 + total_days * x = 0 := by
    rw [initial_unread, daily_read, total_days]
    norm_num
  exact Iff.trans h (by norm_cast)

end new_messages_per_day_l113_113670


namespace count_integer_solutions_less_than_zero_l113_113933

theorem count_integer_solutions_less_than_zero : 
  ∃ k : ℕ, k = 4 ∧ (∀ n : ℤ, n^4 - n^3 - 3 * n^2 - 3 * n - 17 < 0 → k = 4) :=
by
  sorry

end count_integer_solutions_less_than_zero_l113_113933


namespace molecular_weight_correct_l113_113755

noncomputable def molecular_weight : ℝ := 
  let N_count := 2
  let H_count := 6
  let Br_count := 1
  let O_count := 1
  let C_count := 3
  let N_weight := 14.01
  let H_weight := 1.01
  let Br_weight := 79.90
  let O_weight := 16.00
  let C_weight := 12.01
  N_count * N_weight + 
  H_count * H_weight + 
  Br_count * Br_weight + 
  O_count * O_weight +
  C_count * C_weight

theorem molecular_weight_correct :
  molecular_weight = 166.01 := 
by
  sorry

end molecular_weight_correct_l113_113755


namespace meeting_occurs_probability_l113_113447

-- Define the conditions for the problem
def valid_times (x y z : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 ∧ 
  0 ≤ y ∧ y ≤ 2 ∧ 
  0 ≤ z ∧ z ≤ 2 

def boss_meets_engineers (x y z : ℝ) : Prop :=
  z > x ∧ z > y

def engineers_wait (x y : ℝ) : Prop :=
  |x - y| ≤ 1.5

noncomputable def meeting_probability : ℝ :=
  ∫ x in 0..2, ∫ y in 0..2, ∫ z in 0..2, 
    (if boss_meets_engineers x y z ∧ engineers_wait x y then 1 else 0) / 8

theorem meeting_occurs_probability : meeting_probability = 7 / 24 := 
sorry

end meeting_occurs_probability_l113_113447


namespace caleb_caught_trouts_l113_113180

theorem caleb_caught_trouts (C : ℕ) (h1 : 3 * C = C + 4) : C = 2 :=
by {
  sorry
}

end caleb_caught_trouts_l113_113180


namespace find_set_A_l113_113921

def set_A (x : ℤ) : Prop := (2*x - 3)/(x + 1) ≤ 0

theorem find_set_A : ∀ x, set_A x ↔ x ∈ {0, 1} := by
  sorry

end find_set_A_l113_113921


namespace num_cards_needed_l113_113080

-- Define the range of three-digit numbers
def three_digit_numbers := {x : ℕ | x >= 100 ∧ x <= 999}

-- Define the set of reversible digits
def reversible_digits := {0, 1, 6, 8, 9}

-- Define the set of valid first and last digits for reversible numbers
def valid_edge_digits := {1, 6, 8, 9}

-- Define the set of valid middle digits for symmetrical numbers
def valid_symmetrical_middle_digits := {0, 1, 8}

-- Define the set of symmetrical digits
def symmetrical_digits := valid_edge_digits

-- Count the number of three-digit numbers that are reversible
noncomputable def num_reversible (n : ℕ) : ℕ := 
  (finset.filter (λ n, (n / 100 ∈ valid_edge_digits) ∧ 
                         (n % 10 ∈ valid_edge_digits) ∧ 
                         ((n / 10) % 10 ∈ reversible_digits)) (finset.range 999)).card

-- Count the number of symmetrical numbers that remain the same when reversed
noncomputable def num_symmetrical : ℕ := 
  (finset.filter (λ n, (n / 100 ∈ symmetrical_digits) ∧ 
                         (n % 10 ∈ symmetrical_digits) ∧ 
                         ((n / 10) % 10 ∈ valid_symmetrical_middle_digits)) (finset.range 999)).card

/-- Prove that the number of cards needed is 46 by the given conditions --/
theorem num_cards_needed : num_symmetrical + ((num_reversible 999 - num_symmetrical) / 2) = 46 :=
sorry

end num_cards_needed_l113_113080


namespace smallest_possible_z_l113_113330

theorem smallest_possible_z :
  ∃ z : ℕ, ∃ w x y : ℕ, w < x ∧ x < y ∧ y < z ∧
  even w ∧ even x ∧ even y ∧ even z ∧
  (w^4 + x^4 + y^4 = z^4) ∧ (z = 14) :=
by
  use 14
  sorry

end smallest_possible_z_l113_113330


namespace neg_ex_of_exists_gt_l113_113722

theorem neg_ex_of_exists_gt :
  ¬(∃ x : ℝ, x > 0 ∧ 2 ^ x > 3 ^ x) ↔ ∀ x : ℝ, x > 0 → 2 ^ x ≤ 3 ^ x := by 
  sorry

end neg_ex_of_exists_gt_l113_113722


namespace simplify_expr_eq_l113_113381

theorem simplify_expr_eq :
  let a := x^2 - 4x + 3
  let b := x^2 - 6x + 9
  let c := x^2 - 6x + 8
  let d := x^2 - 8x + 15
  ∃ x, 
    a = (x-3)*(x-1) ∧ 
    b = (x-3)^2 ∧ 
    c = (x-4)*(x-2) ∧ 
    d = (x-5)*(x-3)
  →  (frac (a/b) / frac (c/d)) = frac ((x-1)*(x-5)) ((x-3)*(x-4)*(x-2))
by
  sorry

end simplify_expr_eq_l113_113381


namespace average_retail_price_of_products_l113_113831

theorem average_retail_price_of_products :
  ∃ A : ℝ, (
    let num_products := 25 in
    let min_price := 400 in
    let max_cheap_products := 12 in
    let max_expensive_product_price := 13200 in
    let total_product_cost := (max_cheap_products * min_price) + ((num_products - max_cheap_products - 1) * min_price) + max_expensive_product_price in
    A = total_product_cost / num_products
  ) → A = 912 :=
by
  let num_products := 25
  let min_price := 400
  let max_cheap_products := 12
  let max_expensive_product_price := 13200
  let total_product_cost := (max_cheap_products * min_price) + ((num_products - max_cheap_products - 1) * min_price) + max_expensive_product_price
  let A := total_product_cost / num_products
  have h1 : total_product_cost = (12 * 400) + (12 * 400) + 13200,
  { sorry },
  have h2 : total_product_cost = 22800,
  { sorry },
  have h3 : A = 22800 / 25,
  { sorry },
  have h4 : A = 912,
  { sorry },
  use A,
  exact h4

end average_retail_price_of_products_l113_113831


namespace water_wasted_per_hour_l113_113708

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l113_113708


namespace real_z_10_thirtieth_roots_of_unity_real_z_10_l113_113074

noncomputable def thirtieth_roots_of_unity : Set ℂ :=
  {z : ℂ | z^30 = 1}

theorem real_z_10 (z : thirtieth_roots_of_unity) :
  ((z : ℂ)^10).re ∈ ({1, -1} : Set ℝ) :=
sorry

theorem thirtieth_roots_of_unity_real_z_10 :
  card {z ∈ thirtieth_roots_of_unity | ((z : ℂ)^10).re ∈ ({1, -1} : Set ℝ)} = 20 :=
sorry

end real_z_10_thirtieth_roots_of_unity_real_z_10_l113_113074


namespace diapers_per_pack_l113_113671

def total_boxes := 30
def packs_per_box := 40
def price_per_diaper := 5
def total_revenue := 960000

def total_packs_per_week := total_boxes * packs_per_box
def total_diapers_sold := total_revenue / price_per_diaper

theorem diapers_per_pack :
  total_diapers_sold / total_packs_per_week = 160 :=
by
  -- Placeholder for the actual proof
  sorry

end diapers_per_pack_l113_113671


namespace min_value_of_expression_l113_113117

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end min_value_of_expression_l113_113117


namespace area_inequality_l113_113120

open Real

variables (AB CD AD BC S : ℝ) (alpha beta : ℝ)
variables (α_pos : 0 < α ∧ α < π) (β_pos : 0 < β ∧ β < π)
variables (S_pos : 0 < S) (H1 : ConvexQuadrilateral AB CD AD BC S)

theorem area_inequality :
  AB * CD * sin α + AD * BC * sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
sorry

end area_inequality_l113_113120


namespace intersection_eq_interval_l113_113590

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

theorem intersection_eq_interval : M ∩ N = {x | 1 < x ∧ x < 5} :=
sorry

end intersection_eq_interval_l113_113590


namespace problem_solution_l113_113303

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_even_number_of_divisors (n : ℕ) : Prop := is_even ((finset.range n).filter (λ d => n % d = 0)).card

def problem_statement : Prop :=
  ∀ (n : ℕ), n < 100 →
  ¬ (n % 3 = 0) →
  has_even_number_of_divisors n →
  (finset.range 100).filter (λ x => x < 100 ∧ ¬ (x % 3 = 0) ∧ has_even_number_of_divisors x).card = 59

theorem problem_solution : problem_statement :=
sorry

end problem_solution_l113_113303


namespace domain_of_f_l113_113219

def domain_f (x : ℝ) : Prop := f x = Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, (domain_f x → -1 ≤ x ∧ x ≤ 1) ∧ (0 ≤ Real.arcsin (x^2) ∧ Real.arcsin (x^2) ≤ Real.pi / 2) :=
by 
  sorry -- Proof to be done.

end domain_of_f_l113_113219


namespace exist_scientist_one_acquaintance_l113_113136

-- Introduce the concepts of scientists and acquaintances.
variable {Scientist : Type} (acquainted : Scientist → Scientist → Prop)

-- Condition: There are scientists with acquaintances.
variable (nonempty : ∃ (A A' : Scientist), acquainted A A')

-- Condition: Acquainted relation is symmetric.
variable (symm_acquainted : ∀ {A B : Scientist}, acquainted A B → acquainted B A)

-- Condition: No two scientists with the same number of acquaintances share a common acquaintance.
variable (distinct_acquaintances : ∀ {A B : Scientist} [finite (Set {x | acquainted A x})],
  ∀ {C : Scientist}, (A ≠ B ∧ Set.card (Set {x | acquainted C x}) = Set.card (Set {x | acquainted A x}) →
  ∀ {D : Scientist}, acquainted A D → acquainted B D → false))

-- Prove: There exists a scientist who is acquainted with exactly one other participant.
theorem exist_scientist_one_acquaintance :
  ∃ (A : Scientist), (Set.card (Set {x | acquainted A x}) = 1) :=
sorry

end exist_scientist_one_acquaintance_l113_113136


namespace S_n_mod_10_eq_5_at_18_l113_113194

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

def S_n (n : ℕ) : ℕ :=
  (List.range n).sumBy (fun k => fibonacci (k + 1))

theorem S_n_mod_10_eq_5_at_18 (n : ℕ) : 
    (S_n 18) % 10 = 5 :=
sorry

end S_n_mod_10_eq_5_at_18_l113_113194


namespace max_m_value_l113_113351

noncomputable def a_n (n : ℕ) : ℕ := (n + 1) * 2^n

noncomputable def b_n (n : ℕ) : ℕ := log2 (a_n n / (n + 1))

def satisfies_inequality (n : ℕ) (m : ℝ) : Prop :=
  (List.prod (List.map (λ k, 1 + 1 / (b_n (2 * k))) (List.range n))) ≥ m * Real.sqrt (b_n (2 * n + 2))

theorem max_m_value : (∀ n : ℕ, satisfies_inequality n (3/4)) :=
by sorry

end max_m_value_l113_113351


namespace interval_for_k_l113_113318

theorem interval_for_k (k : ℝ) : (∃ x : ℝ, |x - 2| - |x - 5| > k) → k ∈ set.Iio 3 :=
by
  sorry

end interval_for_k_l113_113318


namespace quadratic_function_through_point_l113_113557

theorem quadratic_function_through_point : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = a * x ^ 2 ∧ ((x, y) = (-1, 4)) → y = 4 * x ^ 2) :=
sorry

end quadratic_function_through_point_l113_113557


namespace expected_pourings_l113_113371

/-- On the New Year's table, there are 4 glasses in a row. Initially, the first and third glasses 
    contain orange juice (1), and the second and fourth glasses are empty (0). While waiting for guests, 
    Valya pour juice randomly from one glass to another. Prove that the expected number of pourings 
    required to have the first and third glasses empty (0), and the second and fourth glasses full (1) is 6. -/

def initial_state : Vector ℕ 4 := ⟨[1, 0, 1, 0], by simp⟩
def target_state : Vector ℕ 4 := ⟨[0, 1, 0, 1], by simp⟩

theorem expected_pourings : 
  ∀ (count_pourings : (Vector ℕ 4) → (Vector ℕ 4) → ℕ),
  count_pourings initial_state target_state = 6 := 
sorry

end expected_pourings_l113_113371


namespace pass_rate_l113_113782

theorem pass_rate (total_students : ℕ) (students_not_passed : ℕ) (pass_rate : ℚ) :
  total_students = 500 → 
  students_not_passed = 40 → 
  pass_rate = (total_students - students_not_passed) / total_students * 100 →
  pass_rate = 92 :=
by 
  intros ht hs hpr 
  sorry

end pass_rate_l113_113782


namespace triangle_one_interior_angle_61_degrees_l113_113745

theorem triangle_one_interior_angle_61_degrees
  (x : ℝ) : 
  (x + 75 + 2 * x + 25 + 3 * x - 22 = 360) → 
  (1 / 2 * (2 * x + 25) = 61 ∨ 
   1 / 2 * (3 * x - 22) = 61 ∨ 
   1 / 2 * (x + 75) = 61) :=
by
  intros h_sum
  sorry

end triangle_one_interior_angle_61_degrees_l113_113745


namespace simplify_and_evaluate_l113_113693

theorem simplify_and_evaluate (x : ℝ) (h : x = 3 / 2) : 
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := 
by
  sorry

end simplify_and_evaluate_l113_113693


namespace number_of_ks_divisible_l113_113859

theorem number_of_ks_divisible (n : ℕ) (hn : n = 353500) 
  (h_div : ∀ k : ℕ, k^2 + k % 505 = 0 → k <= n) : 
  (finset.filter (λ k, (k^2 + k) % 505 = 0) (finset.range (n + 1))).card = 2800 := 
sorry

end number_of_ks_divisible_l113_113859


namespace green_shirts_percentage_l113_113624

def total_students : ℕ := 600
def percent_blue : ℝ := 0.45
def percent_red : ℝ := 0.23
def other_students : ℕ := 102

theorem green_shirts_percentage :
    ( ( total_students - (percent_blue * total_students).to_nat - (percent_red * total_students).to_nat - other_students ).to_nat / total_students : ℝ ) * 100 = 15 := 
by
  sorry

end green_shirts_percentage_l113_113624


namespace triangle_third_side_l113_113095

-- Define the sides of the triangle
def AC : ℝ := 6
def BC : ℝ := 8
def medians_perpendicular (B K A N : Type) [inner_product_space ℝ (B → ℝ)] (O : B → A → ℝ) : Prop :=
  O K N = 0 -- Medians are perpendicular at O

-- Define the theorem
theorem triangle_third_side (AB : ℝ) (h1 : AB > 0) 
  (median_property : ∃ (B K A N : Type), medians_perpendicular B K A N) : 
  AB = 2 * ℝ.sqrt 5 := 
sorry

end triangle_third_side_l113_113095


namespace area_APGQ_l113_113411

/-- Triangle ABC with sides BC = 7, CA = 8, AB = 9 and
D, E, F are midpoints of BC, CA, AB respectively.
G is the intersection of AD and BE,
G' is the reflection of G across D,
G'E meets CG at P, 
G'F meets BG at Q.
The area of APG'Q is 16√5 / 3.
-/

theorem area_APGQ (BC CA AB : ℝ)
  (hBC : BC = 7) (hCA : CA = 8) (hAB : AB = 9)
  (D E F : ℝ × ℝ)
  (G G' P Q: ℝ × ℝ)
  (hD : midpoint D (BC/2) (BC/2))
  (hE : E = (CA/2, 0, CA/2))
  (hF : F = (AB/2, AB/2, 0))
  (hG : G = (1/3, 1/3, 1/3))
  (hG' : G' = (−1/3, 2/3, 2/3))
  -- Defining the positions for P and Q can involve further detail in actual proofs.
  (h_P : (P = (2/9, 2/9, 5/9)))
  (harea : ∀ s a b c : ℝ, s = (a + b + c)/2 → triangle_area ABC BC CA AB = 12 * √5): 
  -- Output statement
  area_APGQ : area (APG'Q) = (16 * √5 / 3) := 
sorry

end area_APGQ_l113_113411


namespace a_seq_101_l113_113983

noncomputable def a_sequence : ℕ → ℚ
| 0       := 2
| (n + 1) := a_sequence n + (1 / 2)

theorem a_seq_101 : a_sequence 100 = 52 := sorry

end a_seq_101_l113_113983


namespace triangle_QRS_area_l113_113617

-- Define the Euclidean plane
open EuclideanGeometry

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def distance (p1 p2 : Point2D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def right_angle (p1 p2 p3 : Point2D) : Prop :=
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0

noncomputable def area_of_triangle (p1 p2 p3 : Point2D) : ℝ :=
  0.5 * real.abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem triangle_QRS_area :
  ∀ (P Q R S T : Point2D),
    distance P Q = 3 ∧
    distance Q R = 3 ∧
    distance R S = 3 ∧
    distance S T = 3 ∧
    distance T P = 3 ∧
    right_angle P Q R ∧
    right_angle R S T ∧
    right_angle S T P ∧
    (R.y = S.y ∧ Q.x = R.x) →  -- Line QR parallel to line ST

  -- Show that the area of triangle QRS is 4.5
  area_of_triangle Q R S = 4.5 :=
begin
  sorry
end

end triangle_QRS_area_l113_113617


namespace true_proposition_is_C_l113_113052

-- Given conditions as definitions
def prop1 (x y : ℝ) : Prop := (x^2 + y^2 = 0) → (x = 0 ∧ y = 0)
def prop2 : Prop := ¬ (∀ (T₁ T₂ : Triangle), T₁.similar T₂ → T₁.area = T₂.area)
def prop3 {A B : Set α} : Prop := (A ∩ B = A) → (A ⊆ B)

-- The proposition to be proven
theorem true_proposition_is_C : (prop1 0 0) ∧ (prop3 (arbitrary Set α) (arbitrary Set α)) :=
by
  -- Proof not included
  sorry

end true_proposition_is_C_l113_113052


namespace find_lambda_l113_113243

def line (λ : ℝ) : (ℝ × ℝ) → Prop := λ x y, x - 2*y + λ = 0 
def translated_line (λ : ℝ) : (ℝ × ℝ) → Prop := λ x y, x - 2*y + λ + 3 = 0
def is_tangent (L : (ℝ × ℝ) → Prop) : Prop :=
(∃ (c : ℝ × ℝ), L c ∧ L = tangent)

def circle : (ℝ × ℝ) → Prop := λ x y, (x + 1)^2 + (y - 2)^2 = 9

theorem find_lambda (λ : ℝ) :
  (translated_line λ is_tangent circle) ↔ (λ = -13 ∨ λ = 3) :=
sorry

end find_lambda_l113_113243


namespace butterflies_count_l113_113404

noncomputable def numberOfButterflies (totalBlackDots butterfliesBlackDots : Real) : Real :=
  totalBlackDots / butterfliesBlackDots

theorem butterflies_count :
  ∀ (totalBlackDots butterfliesBlackDots : Real), totalBlackDots = 397.0 → 
  butterfliesBlackDots = 33.08333333 → 
  abs (numberOfButterflies totalBlackDots butterfliesBlackDots - 12) < 1 :=
by
  intros totalBlackDots butterfliesBlackDots hTotal hEach
  rw [numberOfButterflies, hTotal, hEach]
  have hCalc : 397.0 / 33.08333333 ≈ 12 := sorry
  sorry

end butterflies_count_l113_113404


namespace probability_abc_divisible_by_4_l113_113372

theorem probability_abc_divisible_by_4 
(a b c : ℕ) 
(h1 : a ∈ finset.range 2016 ∪ ∅)
(h2 : b ∈ finset.range 2016 ∪ ∅)
(h3 : c ∈ finset.range 2016 ∪ ∅) : 
  (nat.divisible (a * b * c + a * b + a) 4) → 
  (11 / 32) := 
sorry

end probability_abc_divisible_by_4_l113_113372


namespace desired_result_l113_113433

noncomputable def alpha (A : set X) : ℝ := sorry
noncomputable def beta (A : set X) : ℝ := sorry
noncomputable def gamma (A : set X) : ℝ := sorry
noncomputable def delta (A : set X) : ℝ := sorry

axiom non_neg_alpha : ∀ (A : set X), 0 ≤ alpha A
axiom non_neg_beta  : ∀ (A : set X), 0 ≤ beta A
axiom non_neg_gamma : ∀ (A : set X), 0 ≤ gamma A
axiom non_neg_delta : ∀ (A : set X), 0 ≤ delta A
axiom main_cond : ∀ (A B : set X), alpha A * beta B ≤ gamma (A ∪ B) * delta (A ∩ B)

def alpha_𝒜 (𝒜 : set (set X)) : ℝ :=
  ∑ A in 𝒜, alpha A

def beta_𝒜 (𝒜 : set (set X)) : ℝ :=
  ∑ A in 𝒜, beta A

def gamma_𝒜 (𝒜 : set (set X)) : ℝ :=
  ∑ A in 𝒜, gamma A

def delta_𝒜 (𝒜 : set (set X)) : ℝ :=
  ∑ A in 𝒜, delta A

theorem desired_result (𝒜 𝒝 : set (set X)) :
  alpha_𝒜 𝒜 * beta_𝒜 𝒝 ≤ gamma_𝒜 (𝒜 ∪ 𝒝) * delta_𝒜 (𝒜 ∩ 𝒝) :=
sorry

end desired_result_l113_113433


namespace blankets_collected_l113_113231

theorem blankets_collected (team_size : ℕ) (first_day_each_person : ℕ) (multiplier_second_day : ℕ) (third_day_total : ℕ) :
  team_size = 15 → first_day_each_person = 2 → multiplier_second_day = 3 → third_day_total = 22 →
  (team_size * first_day_each_person + (team_size * first_day_each_person * multiplier_second_day) + third_day_total) = 142 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end blankets_collected_l113_113231


namespace prob_4_consecutive_baskets_prob_exactly_4_baskets_l113_113810

theorem prob_4_consecutive_baskets 
  (p : ℝ) (h : p = 1/2) : 
  (p^4 * (1 - p) + (1 - p) * p^4) = 1/16 :=
by sorry

theorem prob_exactly_4_baskets 
  (p : ℝ) (h : p = 1/2) : 
  5 * p^4 * (1 - p) = 5/32 :=
by sorry

end prob_4_consecutive_baskets_prob_exactly_4_baskets_l113_113810


namespace factorization_of_1386_l113_113304

theorem factorization_of_1386 :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b : ℕ), a * b = 1386 → 10 ≤ a ∧ a < 100 → 10 ≤ b ∧ b < 100 → a ≤ b → n = 1 :=
begin
  sorry
end

end factorization_of_1386_l113_113304


namespace inequality_proof_l113_113940

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a ^ 3 / (a ^ 2 + a * b + b ^ 2)) + (b ^ 3 / (b ^ 2 + b * c + c ^ 2)) + (c ^ 3 / (c ^ 2 + c * a + a ^ 2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_proof_l113_113940


namespace parabola_point_ordinate_l113_113244

-- The definition of the problem as a Lean 4 statement
theorem parabola_point_ordinate (a : ℝ) (x₀ y₀ : ℝ) 
  (h₀ : 0 < a)
  (h₁ : x₀^2 = (1 / a) * y₀)
  (h₂ : dist (0, 1 / (4 * a)) (0, -1 / (4 * a)) = 1)
  (h₃ : dist (x₀, y₀) (0, 1 / (4 * a)) = 5) :
  y₀ = 9 / 2 := 
sorry

end parabola_point_ordinate_l113_113244


namespace limit_of_expectation_l113_113359
open scoped Classical

variable {Ω : Type} [ProbabilitySpace Ω]
variable (ξ : Ω → ℝ)
variable {k : ℕ} (x : ℕ → ℝ) (p : ℕ → ℝ)

noncomputable def max (x : ℕ → ℝ) (k : ℕ) := 
  Finset.sup (Finset.range k) x

def takes_values (ξ : Ω → ℝ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ ω, ∃ i, i < k ∧ ξ ω = x i

theorem limit_of_expectation (hξ : takes_values ξ x k) (hx : ∀ i, i < k → 0 ≤ x i) :
  (lim (λ n : ℕ, (Expectation (λ ω, ξ ω ^ n)) ^ (1 / (n : ℝ))) atTop) = max x k :=
sorry

end limit_of_expectation_l113_113359


namespace equation_of_line_AC_equation_of_altitude_from_B_to_AB_l113_113561

structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 4 0
def B := Point.mk 6 7
def C := Point.mk 0 3

-- Define the line equation function passing through two points
def line_equation (P Q : Point) : ℝ → ℝ → Bool :=
  λ x y => ∃ a b c, a * P.x + b * P.y + c = 0 ∧ a * Q.x + b * Q.y + c = 0 ∧ a * x + b * y + c = 0

-- Define the altitudes function
def altitude_line (P Q R : Point) : ℝ → ℝ → Bool :=
  λ x y => 
    let slope := (Q.y - P.y) / (Q.x - P.x)
    let perp_slope := -1 / slope
    ∃ a b c, b = -a / perp_slope ∧ a * R.x + b * R.y + c = 0 ∧ a * x + b * y + c = 0

theorem equation_of_line_AC :
  ∀ x y, line_equation A C x y ↔ 3 * x + 4 * y - 12 = 0 :=
by
  sorry

theorem equation_of_altitude_from_B_to_AB :
  ∀ x y, altitude_line A B B x y ↔ 2 * x + 7 * y - 21 = 0 :=
by
  sorry

end equation_of_line_AC_equation_of_altitude_from_B_to_AB_l113_113561


namespace numberOfBlackBalls_l113_113618

noncomputable def totalBalls : ℕ := 15
noncomputable def totalDraws : ℕ := 100
noncomputable def blackDraws : ℕ := 60

theorem numberOfBlackBalls (h : totalBalls = 15) (h1 : totalDraws = 100) (h2 : blackDraws = 60) : 
  ∃ (blackBalls : ℕ), blackBalls = 9 :=
by
  have ratio : Rational := (blackDraws : ℚ) / (totalDraws : ℚ)
  have totalBallsRatio : Rational := (3 : ℚ) / (5 : ℚ)
  have blackBalls : ℚ := (totalBalls : ℚ) * totalBallsRatio
  exact ⟨9, by norm_num [blackBalls]⟩

end numberOfBlackBalls_l113_113618


namespace group_of_5_more_men_than_women_l113_113793

-- Conditions: There are 6 men and 4 women.
def men : ℕ := 6
def women : ℕ := 4

/-
We want to prove the number of ways to choose a group of 5 people
such that there are always more men than women.
-/

theorem group_of_5_more_men_than_women (d : ℕ) :
  d = ∑ i in ({3, 4, 5} : Finset ℕ), (Nat.choose men i) * (Nat.choose women (5 - i)) := by
  have H1: (Nat.choose men 3) * (Nat.choose women 2) = 20 * 6 := rfl
  have H2: (Nat.choose men 4) * (Nat.choose women 1) = 15 * 4 := rfl
  have H3: (Nat.choose men 5) * 0 = 6 := rfl
  let d := H1 + H2 + H3
  exact d

end group_of_5_more_men_than_women_l113_113793


namespace derivative_of_y_l113_113115

noncomputable def y (x : ℝ) : ℝ :=
  (real.cot 2)^(1/3) - (1/20) * ((cos (10 * x))^2 / sin (20 * x))

theorem derivative_of_y :
  ∀ x : ℝ, deriv y x = 1 / (4 * (sin (10 * x))^2) :=
by
  assume x,
  sorry

end derivative_of_y_l113_113115


namespace earth_surface_inhabitable_fraction_l113_113947

theorem earth_surface_inhabitable_fraction:
  (let water_covered := 3 / 4 in
   let land_exposed := 1 / 4 in
   let inhabitable_for_humans := 1 / 3 in
   inhabitable_for_humans * land_exposed = 1 / 12) := 
by
  sorry

end earth_surface_inhabitable_fraction_l113_113947


namespace sum_of_squares_l113_113248

theorem sum_of_squares (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range n, (2 * 3^k)^2) = (1 / 2) * (9^n - 1) :=
by
  sorry

end sum_of_squares_l113_113248


namespace correct_options_b_c_d_l113_113042

open Nat

-- Definitions (Conditions)
def five_volunteers := {XiaoZhao, XiaoLi, XiaoLuo, XiaoWang, XiaoZhang}
def four_tasks := {translation, security, etiquette, service}
def different_heights := {h1, h2, h3, h4, h5 : ℕ}

-- Proof problem (Statements to verify)
theorem correct_options_b_c_d :
  (5.choose 2 * 4.factorial = 240) ∧
  (5.choose 2 * 3.factorial = 60) ∧
  ((5 * 4 / 2) * 2.factorial = 40) :=
by
  sorry

end correct_options_b_c_d_l113_113042


namespace sum_of_seven_smallest_multiples_of_twelve_l113_113763

theorem sum_of_seven_smallest_multiples_of_twelve : 
  let multiples := [12, 24, 36, 48, 60, 72, 84] in
  ∑ i in multiples, i = 336 := 
by sorry

end sum_of_seven_smallest_multiples_of_twelve_l113_113763


namespace correct_opinions_count_l113_113540

noncomputable def h (m : ℝ) : ℝ := m^2 - real.log (2 * m) - 1

theorem correct_opinions_count :
  (∀ (m : ℝ), 0 < m → h m = 0 ↔ sqrt 2 < m ∧ m < sqrt 3) →
  (∀ (m : ℝ), h' m = 2 * m - 1 / m → 0 < h' m) →
  (∀ (m : ℝ), 0 < m → h (sqrt 2) < 0 ∧ h (sqrt 3) > 0) →
  ∃! l : line, tangent_to C1 l ∧ tangent_to C2 l →
  ∃! m : ℝ, sqrt 2 < m ∧ m < sqrt 3 :=
sorry

#print correct_opinions_count

end correct_opinions_count_l113_113540


namespace wire_length_l113_113521

theorem wire_length (A : ℝ) (h : A = 53824) : ∃ (length_of_wire : ℝ), length_of_wire = 9280 :=
by
  let s := Real.sqrt A
  let perimeter := 4 * s
  let length_of_wire := 10 * perimeter
  use length_of_wire
  rw [h]
  calc
    s = 232 := sorry -- sqrt(53824)
    perimeter = 928 := by
      rw [h]
      calc
        4 * 232 = 928 := sorry
    length_of_wire = 10 * 928 := rfl
    10 * 928 = 9280 := sorry

end wire_length_l113_113521


namespace smallest_n_value_existence_l113_113154

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l113_113154


namespace general_term_formula_l113_113055

theorem general_term_formula (n : ℕ) (a : ℕ → ℚ) :
  (∀ n, a n = (-1)^n * (n^2)/(2 * n - 1)) :=
sorry

end general_term_formula_l113_113055


namespace triangle_YZ_length_l113_113985

theorem triangle_YZ_length 
  (X Z : ℝ) 
  (XY : ℝ) 
  (cos_add_sin_eq_two : cos (2 * X - Z) + sin (X + Z) = 2)
  (XY_val : XY = 6) : 
  YZ = 3 := 
sorry

end triangle_YZ_length_l113_113985


namespace sitting_break_frequency_l113_113340

theorem sitting_break_frequency (x : ℕ) (h1 : 240 % x = 0) (h2 : 240 / 20 = 12) (h3 : 240 / x + 10 = 12) : x = 120 := 
sorry

end sitting_break_frequency_l113_113340


namespace nth_term_pattern_l113_113766

theorem nth_term_pattern (a : ℕ → ℕ) (h : ∀ n, a n = n * (n - 1)) : 
  (a 0 = 0) ∧ (a 1 = 2) ∧ (a 2 = 6) ∧ (a 3 = 12) ∧ (a 4 = 20) ∧ 
  (a 5 = 30) ∧ (a 6 = 42) ∧ (a 7 = 56) ∧ (a 8 = 72) ∧ (a 9 = 90) := sorry

end nth_term_pattern_l113_113766


namespace bacon_needed_l113_113823

def eggs_per_plate : ℕ := 2
def bacon_per_plate : ℕ := 2 * eggs_per_plate
def customers : ℕ := 14
def bacon_total (eggs_per_plate bacon_per_plate customers : ℕ) : ℕ := customers * bacon_per_plate

theorem bacon_needed : bacon_total eggs_per_plate bacon_per_plate customers = 56 :=
by
  sorry

end bacon_needed_l113_113823


namespace smallest_N_exists_l113_113862

theorem smallest_N_exists :
  ∃ (N : ℕ), N > 0 ∧ 
  (∃ (i : ℕ), i ∈ {0, 1, 2, 3} ∧ (N + i) % 9 = 0) ∧
  (∃ (i : ℕ), i ∈ {0, 1, 2, 3} ∧ (N + i) % 25 = 0) ∧
  (∃ (i : ℕ), i ∈ {0, 1, 2, 3} ∧ (N + i) % 49 = 0) ∧
  (∃ (i : ℕ), i ∈ {0, 1, 2, 3} ∧ (N + i) % 121 = 0) :=
by {
  use 363,
  split,
  { linarith, },
  split,
  { use 0, split, { repeat {linarith <|> norm_num, }, }, },
  split,
  { use 2, split, { repeat {linarith <|> norm_num, }, }, },
  split,
  { use 1, split, { repeat {linarith <|> norm_num, }, }, },
  { use 3, split, { repeat {linarith <|> norm_num, }, }, },
}

end smallest_N_exists_l113_113862


namespace max_value_a_plus_b_minus_c_l113_113752

-- Define an auxiliary function to check if a number is three-digit.
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the problem conditions
def uses_digits_exactly_once (a b c : ℕ) : Prop :=
  let digits : List ℕ := 
    (List.of_fn (λ i, (a / 10^(2 - i)) % 10)) 
    ++ (List.of_fn (λ i, (b / 10^(2 - i)) % 10)) 
    ++ (List.of_fn (λ i, (c / 10^(2 - i)) % 10))
  in digits.erase_dup.length = 9 ∧ digits.all (λ d, 1 ≤ d ∧ d ≤ 9)

-- Define the proof problem statement
theorem max_value_a_plus_b_minus_c (a b c : ℕ) 
  (ha : is_three_digit a) 
  (hb : is_three_digit b) 
  (hc : is_three_digit c) 
  (h : uses_digits_exactly_once a b c) : 
  a + b - c = 1716 := 
sorry

end max_value_a_plus_b_minus_c_l113_113752


namespace find_ratio_of_isosceles_triangle_l113_113480

noncomputable def isosceles_triangle_ratio {A B C D E : Point} {k : ℝ} (h_triangle: IsIsoscelesTriangle A B C)
  (h_inscribed : InscribedInCircle A B C)
  (h_diameter : DiameterIntersects AD BC E)
  (h_de_vs_ea : DE / EA = k) : ℝ :=
  let CE := SegmentLength E C
  let BC := SegmentLength B C
  CE / BC

theorem find_ratio_of_isosceles_triangle {A B C D E : Point} {k : ℝ} (h_triangle: IsIsoscelesTriangle A B C)
  (h_inscribed : InscribedInCircle A B C)
  (h_diameter : DiameterIntersects AD BC E)
  (h_de_vs_ea : DE / EA = k) : 
  isosceles_triangle_ratio h_triangle h_inscribed h_diameter h_de_vs_ea = 2 * k / (1 + k) :=
by
  sorry

end find_ratio_of_isosceles_triangle_l113_113480


namespace pool_width_l113_113043

-- Define the given conditions
def hose_rate : ℝ := 60 -- cubic feet per minute
def drain_time : ℝ := 2000 -- minutes
def pool_length : ℝ := 150 -- feet
def pool_depth : ℝ := 10 -- feet

-- Calculate the total volume drained
def total_volume := hose_rate * drain_time -- cubic feet

-- Define a variable for the pool width
variable (W : ℝ)

-- The statement to prove
theorem pool_width :
  (total_volume = pool_length * W * pool_depth) → W = 80 :=
by
  sorry

end pool_width_l113_113043


namespace rahuls_share_l113_113776

theorem rahuls_share (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)
  (rahul_work_one_day : rahul_days > 0) (rajesh_work_one_day : rajesh_days > 0)
  (total_payment_eq : total_payment = 105) 
  (rahul_days_eq : rahul_days = 3) 
  (rajesh_days_eq : rajesh_days = 2) :
  rahul_share = 42 := 
by
  sorry

end rahuls_share_l113_113776


namespace complement_of_B_in_A_l113_113234

def setA : set ℝ := {x | x > -2}
def setB : set ℝ := {y | -1 ≤ y ∧ y ≤ 1}

theorem complement_of_B_in_A : (setA \ {x | -1 ≤ real.log (x + 2) ∧ real.log (x + 2) ≤ 1}) = ((-2 : ℝ), -1) ∪ ((1 : ℝ), ⊤) :=
by
  sorry

end complement_of_B_in_A_l113_113234


namespace avg_speed_trip_l113_113457

/-- Given a trip with total distance of 70 kilometers, with the first 35 kilometers traveled at
    48 kilometers per hour and the remaining 35 kilometers at 24 kilometers per hour, 
    prove that the average speed is 32 kilometers per hour. -/
theorem avg_speed_trip (d1 d2 : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) (total_distance : ℝ)
  (H1 : d1 = 35) (H2 : d2 = 35) (H3 : s1 = 48) (H4 : s2 = 24)
  (H5 : total_distance = 70)
  (T1 : t1 = d1 / s1) (T2 : t2 = d2 / s2) :
  70 / (t1 + t2) = 32 :=
by
  sorry

end avg_speed_trip_l113_113457


namespace parabola_through_points_with_h_l113_113901

noncomputable def quadratic_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_through_points_with_h (
    a h k : ℝ) 
    (H0 : quadratic_parabola a h k 0 = 4)
    (H1 : quadratic_parabola a h k 6 = 5)
    (H2 : a < 0)
    (H3 : 0 < h)
    (H4 : h < 6) : 
    h = 4 := 
sorry

end parabola_through_points_with_h_l113_113901


namespace no_solution_log_eq_l113_113508

theorem no_solution_log_eq (x : ℝ)
  (h1 : 0 < x + 5)
  (h2 : 0 < x - 3)
  (h3 : 0 < x^2 - 8x + 15) :
  ¬ (log (x + 5) + log (x - 3) = log (x^2 - 8x + 15)) :=
by
  sorry

end no_solution_log_eq_l113_113508


namespace repeating_decimal_to_fraction_l113_113211

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end repeating_decimal_to_fraction_l113_113211


namespace extreme_value_problem_1_extreme_value_problem_2_exponential_inequality_l113_113568

-- Problem 1
def f1 (x : ℝ) : ℝ := log (1 + (1 / 2) * x) - (2 * x) / (x + 2)
theorem extreme_value_problem_1 : ∃ x : ℝ, f1(x) = log 2 - 1 :=
sorry

-- Problem 2
def f2 (a x : ℝ) : ℝ := log (1 + a * x) - (2 * x) / (x + 2)
theorem extreme_value_problem_2 (a : ℝ) (h : 1/2 < a ∧ a < 1) (x1 x2 : ℝ) (hx1 : f2 a x1 = 0) (hx2 : f2 a x2 = 0) :
  f2 a x1 + f2 a x2 > f2 a 0 :=
sorry

-- Problem 3
theorem exponential_inequality (n : ℕ) (hn : n ≥ 2) : exp (n * (n - 1) / 2) > nat.factorial n :=
sorry

end extreme_value_problem_1_extreme_value_problem_2_exponential_inequality_l113_113568


namespace correct_statements_l113_113925

variables {a b : ℝ}
variables (m n : ℝ × ℝ)

/-- Let vectors $\overrightarrow{m}=(2,\frac{1}{a})$ and $\overrightarrow{n}=(\frac{1}{b},4)$, where $a > 0$ and $b > 0$. -/
noncomputable def vector_m := (2 : ℝ, 1 / a)
noncomputable def vector_n := (1 / b, 4 : ℝ)

theorem correct_statements (h1 : a > 0) (h2 : b > 0) :
  ((¬(2 * 4 - (1 / a) * (1 / b) = 0)) → (Real.logb 2 (a * b) ≠ -3)) ∧
  ((a + b = 1) → (2 * (1 / b) + (1 / a) * 4 ≥ 6 + 4 * Real.sqrt 2)) ∧
  ((Real.sqrt (2 ^ 2 + (1 / a) ^ 2) = Real.sqrt ((1 / b) ^ 2 + 4 ^ 2) ∧ Real.sqrt ((1 / (a^2)) + 4) > 4 * Real.sqrt 2) → (1 < b / a ∧ b / a < Real.sqrt 7 / 2)) :=
sorry

end correct_statements_l113_113925


namespace least_whole_number_l113_113098

theorem least_whole_number (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : 7 ∣ n) : 
  n = 301 := 
sorry

end least_whole_number_l113_113098


namespace extremum_areas_extremum_areas_case_b_equal_areas_l113_113674

variable (a b x : ℝ)
variable (h1 : b > 0) (h2 : a ≥ b) (h_cond : 0 < x ∧ x ≤ b)

def area_t1 (a b x : ℝ) : ℝ := 2 * x^2 - (a + b) * x + a * b
def area_t2 (a b x : ℝ) : ℝ := -2 * x^2 + (a + b) * x

noncomputable def x0 (a b : ℝ) : ℝ := (a + b) / 4

-- Problem 1
theorem extremum_areas :
  b ≥ a / 3 → area_t1 a b (x0 a b) ≤ area_t1 a b x ∧ area_t2 a b (x0 a b) ≥ area_t2 a b x :=
sorry

theorem extremum_areas_case_b :
  b < a / 3 → (area_t1 a b b = b^2) ∧ (area_t2 a b b = a * b - b^2) :=
sorry

-- Problem 2
theorem equal_areas :
  b ≤ a ∧ a ≤ 2 * b → (area_t1 a b (a / 2) = area_t2 a b (a / 2)) ∧ (area_t1 a b (b / 2) = area_t2 a b (b / 2)) :=
sorry

end extremum_areas_extremum_areas_case_b_equal_areas_l113_113674


namespace prove_equations_and_PA_PB_l113_113631

noncomputable def curve_C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ + 7 / ρ = 4 * Real.cos θ + 4 * Real.sin θ

theorem prove_equations_and_PA_PB :
  (∀ (α : ℝ), 0 ≤ α ∧ α < π → 
    (∃ (C1_cart : ℝ → ℝ → Prop), ∀ x y, C1_cart x y ↔ x^2 = 4 * y) ∧
    (∃ (C1_polar : ℝ → ℝ → Prop), ∀ ρ θ, C1_polar ρ θ ↔ ρ^2 * Real.cos θ^2 = 4 * ρ * Real.sin θ) ∧
    (∃ (C2_cart : ℝ → ℝ → Prop), ∀ x y, C2_cart x y ↔ (x - 2)^2 + (y - 2)^2 = 1)) ∧
  (∃ (P A B : ℝ × ℝ), P = (0, 1) ∧ 
    curve_C1_parametric t (Real.pi / 2) = A ∧ 
    curve_C1_parametric t (Real.pi / 2) = B ∧ 
    |P - A| * |P - B| = 4) :=
sorry

end prove_equations_and_PA_PB_l113_113631


namespace speed_in_kmph_l113_113517

noncomputable def speed_conversion (speed_mps: ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_in_kmph : speed_conversion 18.334799999999998 = 66.00528 :=
by
  -- proof steps would go here
  sorry

end speed_in_kmph_l113_113517


namespace third_recipe_soy_sauce_l113_113036

theorem third_recipe_soy_sauce :
  let bottle_ounces := 16
  let cup_ounces := 8
  let first_recipe_cups := 2
  let second_recipe_cups := 1
  let total_bottles := 3
  (total_bottles * bottle_ounces) / cup_ounces - (first_recipe_cups + second_recipe_cups) = 3 :=
by
  sorry

end third_recipe_soy_sauce_l113_113036


namespace common_ratio_of_geometric_sequence_l113_113528

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable (q : ℝ)

-- Conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = q * a n
def first_term := a_n 1 = -1
def sum_first_10_terms_eq := (S_n 10) / (S_n 5) = 31 / 32

-- Question to be proven
theorem common_ratio_of_geometric_sequence
  (h1 : geometric_sequence a_n)
  (h2 : first_term)
  (h3 : sum_first_10_terms_eq) :
  q = - 1 / 2 :=
sorry

end common_ratio_of_geometric_sequence_l113_113528


namespace g_increasing_on_neg_infty_0_l113_113895

theorem g_increasing_on_neg_infty_0 {f : ℝ → ℝ} (h1 : ∀ x y : ℝ, x < y → f(x) < f(y)) 
  (h2 : ∀ x : ℝ, f(x) < 0) : ∀ x y : ℝ, x < y ∧ y < 0 → (x^2 * f(x)) < (y^2 * f(y)) := 
by 
  sorry

end g_increasing_on_neg_infty_0_l113_113895


namespace coefficient_a_eq_2_l113_113550

theorem coefficient_a_eq_2 (a : ℝ) (h : (a^3 * (4 : ℝ)) = 32) : a = 2 :=
by {
  -- Proof will need to be filled in here
  sorry
}

end coefficient_a_eq_2_l113_113550


namespace find_a_b_l113_113570

noncomputable def f (x a b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3 * b * x

theorem find_a_b 
  (a b : ℝ)
  (hf : (f 1 a b = -11))
  (hf' : (∂ x, f x a b) 1 = -12) :
  a = 1 ∧ b = -3 :=
sorry

end find_a_b_l113_113570


namespace joshua_average_speed_l113_113995

theorem joshua_average_speed 
  (total_distance : ℤ) 
  (total_time : ℤ) 
  (h1 : total_distance = 28) 
  (h2 : total_time = 8) : 
  total_distance / total_time = 3.5 := 
by
  sorry

end joshua_average_speed_l113_113995


namespace lance_read_yesterday_l113_113347

-- Definitions based on conditions
def total_pages : ℕ := 100
def pages_tomorrow : ℕ := 35
def pages_yesterday (Y : ℕ) : ℕ := Y
def pages_today (Y : ℕ) : ℕ := Y - 5

-- The statement that we need to prove
theorem lance_read_yesterday (Y : ℕ) (h : pages_yesterday Y + pages_today Y + pages_tomorrow = total_pages) : Y = 35 :=
by sorry

end lance_read_yesterday_l113_113347


namespace ratio_of_areas_l113_113469

theorem ratio_of_areas 
  (t : ℝ) (q : ℝ)
  (h1 : t = 1 / 4)
  (h2 : q = 1 / 2) :
  q / t = 2 :=
by sorry

end ratio_of_areas_l113_113469


namespace xyz_value_l113_113544

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 26 / 3 := 
by
  sorry

end xyz_value_l113_113544


namespace result_after_subtraction_l113_113148

-- Define the conditions
def x : ℕ := 40
def subtract_value : ℕ := 138

-- The expression we will evaluate
def result (x : ℕ) : ℕ := 6 * x - subtract_value

-- The theorem stating the evaluated result
theorem result_after_subtraction : result 40 = 102 :=
by
  unfold result
  rw [← Nat.mul_comm]
  simp
  sorry -- Proof placeholder

end result_after_subtraction_l113_113148


namespace coeff_x3_expansion_l113_113634

noncomputable def binom_coeff (n k : ℕ) := Nat.choose n k

theorem coeff_x3_expansion :
  let x := (1 : ℝ) -- Choosing x as a floating-point number for simplicity
  let expr := (x + 2 / (x^2)) * (1 + 2 * x)^5
  let coeff_x3 := 4 * binom_coeff 5 2 + 2 * 32 * binom_coeff 5 5
  coeff_x3 = 104 := 
by
  let x := (1 : ℝ)

  -- The expressions for simplicity
  let _ := (x + 2 / (x^2)) * (1 + 2 * x)^5
  let _ := binom_coeff 5 2
  let _ := binom_coeff 5 5

  -- The coefficient of x^3
  let coeff_x3 := 4 * binom_coeff 5 2 + 2 * 32 * binom_coeff 5 5
  show coeff_x3 = 104 from sorry

end coeff_x3_expansion_l113_113634


namespace intersection_of_A_and_B_l113_113257

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l113_113257


namespace probability_prime_sum_l113_113199

noncomputable def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def isPrimeSum (a b : ℕ) : Bool :=
  Nat.prime (a + b)

def validPairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17)]

def totalPairs : ℕ :=
  Nat.choose 8 2

def favorablePairs : ℕ :=
  4 -- Manually counted from solution step 3

theorem probability_prime_sum :
  Rat.mk favorablePairs totalPairs = Rat.mk 1 7 := by
  sorry

end probability_prime_sum_l113_113199


namespace find_values_of_a_and_b_find_square_root_l113_113553

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end find_values_of_a_and_b_find_square_root_l113_113553


namespace alex_potatoes_peeled_l113_113931

theorem alex_potatoes_peeled 
  (initial_potatoes : ℕ)
  (homer_rate : ℕ)
  (alex_rate : ℕ)
  (homer_peeling_time : ℕ)
  (total_peeling_time : ℝ) :
  initial_potatoes = 60 →
  homer_rate = 4 →
  alex_rate = 6 →
  homer_peeling_time = 6 →
  total_peeling_time = 9.6 →
  alex_peeled = 22 := by

  intros h_init h_homer_r h_alex_r h_homer_t h_total_t
  let remaining_potatoes := initial_potatoes - (homer_rate * homer_peeling_time)
  let combined_rate := homer_rate + alex_rate
  have remaining_time : remaining_potatoes / combined_rate = 3.6 := by sorry
  have alex_peeled : alex_rate * remaining_time = 21.6 := by sorry
  exact alex_peeled = 22

end alex_potatoes_peeled_l113_113931


namespace painting_cost_of_cube_l113_113050

-- Define the relevant parameters
def volume_of_cube := 9261 -- cubic centimeters
def cost_per_sq_cm := 13 -- paise

-- Problem statement:
theorem painting_cost_of_cube : 
  let side_length := Real.cbrt volume_of_cube in
  let surface_area := 6 * side_length^2 in
  let total_cost_paise := surface_area * cost_per_sq_cm in
  let total_cost_rupees := total_cost_paise / 100 in
  total_cost_rupees = 344.98 :=
sorry

end painting_cost_of_cube_l113_113050


namespace lateral_surface_area_of_rotated_triangle_l113_113884

theorem lateral_surface_area_of_rotated_triangle :
  let AC := 3
  let BC := 4
  let AB := Real.sqrt (AC ^ 2 + BC ^ 2)
  let radius := BC
  let slant_height := AB
  let lateral_surface_area := Real.pi * radius * slant_height
  lateral_surface_area = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_rotated_triangle_l113_113884


namespace S_20_value_l113_113538

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (∀ n, a (2 * n + 1) = 2 * a (2 * n - 1)) ∧
  (∀ n, a (2 * n) = a (2 * n - 1) + 1)

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range n, a i

theorem S_20_value
  (a : ℕ → ℕ)
  (h1 : seq a) :
  S a 20 = 2056 :=
sorry

end S_20_value_l113_113538


namespace composite_probability_l113_113314

-- Definitions of composite numbers and the problem space
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

-- Rolling a die results in a number from 1 to 6
def roll_die : Type := {n : ℕ // n ∈ ({1, 2, 3, 4, 5, 6} : set ℕ)}

-- Considering 5 dice rolls
def rolls (n : ℕ) : Type := vector roll_die n

-- The main theorem
theorem composite_probability :
  ∃ n : ℚ, n = 485 / 486 ∧
  ∀ (r : rolls 5), 
    (is_composite (r.to_list.map subtype.val).prod ↔ 
    realizability of r.to_list in computed space,)
    -- This clause ensures that the definition of composite numbers 
    -- and rolling dice outputs are respected in the context of product analysis.
    sorry

end composite_probability_l113_113314


namespace parts_of_sqrt_cube_root_of_sum_l113_113025

theorem parts_of_sqrt (a b : ℝ) : 
  (⌊real.sqrt 2⌋ = 1) ∧ ((real.sqrt 2 - 1) = real.sqrt 2 - 1) ∧
  (⌊real.sqrt 11⌋ = 3) ∧ ((real.sqrt 11 - 3) = real.sqrt 11 - 3) :=
by sorry

theorem cube_root_of_sum (a b : ℝ) (h1 : a = real.sqrt 5 - 2) (h2 : b = 10) :
  real.cbrt (a + b - real.sqrt 5) = 2 :=
by sorry

end parts_of_sqrt_cube_root_of_sum_l113_113025


namespace optionA_not_identical_optionB_not_identical_optionC_not_identical_optionD_not_identical_no_functions_identical_l113_113419

-- Define the first option functions and their properties
def optionA_f1 (x : ℝ) : ℝ := (x^2) / x
def optionA_f2 (x : ℝ) : ℝ := x

def optionB_f1 (x : ℝ) : ℝ := sqrt (x^2)
def optionB_f2 (x : ℝ) : ℝ := x

def optionC_f1 (x : ℝ) : ℝ := 3 * x^3
def optionC_f2 (x : ℝ) : ℝ := x

def optionD_f1 (x : ℝ) : ℝ := (sqrt x)^2
def optionD_f2 (x : ℝ) : ℝ := x

-- Prove that the functions pairs are not identical
theorem optionA_not_identical : (∃ x : ℝ, x ≠ 0 ∧ optionA_f1 x ≠ optionA_f2 x) :=
by {
  use 1,
  split,
  { norm_num, },
  { simp [optionA_f1, optionA_f2], }
}

theorem optionB_not_identical : (∃ x : ℝ, optionB_f1 x ≠ optionB_f2 x) :=
by {
  use (-1),
  simp [optionB_f1, optionB_f2],
  norm_num,
}

theorem optionC_not_identical : (∃ x : ℝ, optionC_f1 x ≠ optionC_f2 x) :=
by {
  use 1,
  simp [optionC_f1, optionC_f2],
}

theorem optionD_not_identical : (∃ x : ℝ, x < 0 ∧ optionD_f1 x ≠ optionD_f2 x) :=
by {
  use -1,
  split,
  { norm_num, },
  { simp [optionD_f1, optionD_f2], }
}

-- Overall result statement
theorem no_functions_identical : 
  (∃ x : ℝ, x ≠ 0 ∧ optionA_f1 x ≠ optionA_f2 x) ∧ 
  (∃ x : ℝ, optionB_f1 x ≠ optionB_f2 x) ∧ 
  (∃ x : ℝ, optionC_f1 x ≠ optionC_f2 x) ∧ 
  (∃ x : ℝ, x < 0 ∧ optionD_f1 x ≠ optionD_f2 x) :=
by {
  exact ⟨optionA_not_identical, optionB_not_identical, optionC_not_identical, optionD_not_identical⟩,
  sorry
}

end optionA_not_identical_optionB_not_identical_optionC_not_identical_optionD_not_identical_no_functions_identical_l113_113419


namespace num_integers_between_200_and_250_with_increasing_digits_l113_113301

theorem num_integers_between_200_and_250_with_increasing_digits : 
  card {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ (n.digits).length = 3 
    ∧ (∀ i j, i < j → (n.digits.nth i < n.digits.nth j))} = 11 := 
sorry

end num_integers_between_200_and_250_with_increasing_digits_l113_113301


namespace rectangular_to_polar_coordinates_l113_113498

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end rectangular_to_polar_coordinates_l113_113498


namespace find_tricycles_l113_113966

noncomputable def number_of_tricycles (w b t : ℕ) : ℕ := t

theorem find_tricycles : ∃ (w b t : ℕ), 
  (w + b + t = 10) ∧ 
  (2 * b + 3 * t = 25) ∧ 
  (number_of_tricycles w b t = 5) :=
  by 
    sorry

end find_tricycles_l113_113966


namespace row_time_to_100_yards_l113_113027

theorem row_time_to_100_yards :
  let init_width_yd := 50
  let final_width_yd := 100
  let increase_width_yd_per_10m := 2
  let rowing_speed_mps := 5
  let current_speed_mps := 1
  let yard_to_meter := 0.9144
  let init_width_m := init_width_yd * yard_to_meter
  let final_width_m := final_width_yd * yard_to_meter
  let width_increase_m_per_10m := increase_width_yd_per_10m * yard_to_meter
  let total_width_increase := (final_width_m - init_width_m)
  let num_segments := total_width_increase / width_increase_m_per_10m
  let total_distance := num_segments * 10
  let effective_speed := rowing_speed_mps + current_speed_mps
  let time := total_distance / effective_speed
  time = 41.67 := by
  sorry

end row_time_to_100_yards_l113_113027


namespace sum_fibonacci_series_l113_113999

theorem sum_fibonacci_series :
  (∑ n in Finset.range 1000, (fib (n + 1)) / 4^(n + 2)) = 1 / 11 :=
by
  sorry

end sum_fibonacci_series_l113_113999


namespace min_real_part_of_z_l113_113006

-- Defining the complex number and the given condition
variables {z : ℂ}
def condition := (z + z⁻¹).re ∈ set.Icc 1 2

-- Goal: Proving the minimum value of the real part of z
theorem min_real_part_of_z (h : condition) : ∃ x : ℝ, x = ℜ z ∧ x = 1/2 :=
by
  sorry

end min_real_part_of_z_l113_113006


namespace marked_cells_dividable_l113_113019

theorem marked_cells_dividable (n : ℕ) (marked_cells : set (ℕ × ℕ))
  (h_size : marked_cells.size = 2 * n)
  (h_rook : ∀ p q ∈ marked_cells, ∃ path : list (ℕ × ℕ), 
    path.head = p ∧ path.last = q ∧ ∀ r ∈ path, r ∈ marked_cells ∧ 
    (r = p ∨ r = q ∨ adjacent_in_grid(r, marked_cells))) :
  ∃ rectangles : list (list (ℕ × ℕ)), list.length rectangles = n ∧ 
  disjoint_union rectangles = marked_cells ∧
  ∀ rect ∈ rectangles, ∃ x1 y1 x2 y2, rect = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)] :=
sorry

def adjacent_in_grid (cell : (ℕ × ℕ), grid : set (ℕ × ℕ)) : Prop := 
  ∃ (dx dy : ℤ), (dx,dy) ∈ [(0,1), (1,0), (0,-1), (-1,0)] ∧ 
    (cell.1 + dx, cell.2 + dy) ∈ grid 

end marked_cells_dividable_l113_113019


namespace train_passing_time_l113_113936

noncomputable def length_of_train : ℝ := 150
noncomputable def speed_of_train_kmh : ℝ := 75
noncomputable def speed_of_car_kmh : ℝ := 45
noncomputable def conversion_factor : ℝ := 5 / 18

theorem train_passing_time :
  let relative_speed_kmh := speed_of_train_kmh - speed_of_car_kmh
  let relative_speed_ms := relative_speed_kmh * conversion_factor
  let time := length_of_train / relative_speed_ms
  time ≈ 18 := 
sorry

end train_passing_time_l113_113936


namespace total_oranges_in_pyramid_l113_113676

theorem total_oranges_in_pyramid
    (base_width : ℕ) (base_length : ℕ) (h_base_width : base_width = 5)
    (h_base_length : base_length = 7)
    (layer_count : ℕ) (h_layer_count: layer_count = 5) :
    (∑ i in finset.range layer_count, (base_width - i) * (base_length - i)) = 85 :=
by
  sorry

end total_oranges_in_pyramid_l113_113676


namespace sequence_formula_comparison_Sn_l113_113272

variable {n : ℕ}
variable {a : ℕ → ℝ} -- Sequence of terms

def S (n : ℕ) : ℝ := -a n - (1 / 2)^(n - 1) + 2

theorem sequence_formula (n : ℕ) (h : ∀ n, a n = n / 2^n) : a n = n / 2^n :=
by sorry

theorem comparison_Sn (n : ℕ) (h : ∀ n, a n = n / 2^n):
  (n = 3 ∨ n = 4 → S n < 2 - 1 / (n - 1)) ∧ (n ≥ 5 → S n > 2 - 1 / (n - 1)) :=
by sorry

end sequence_formula_comparison_Sn_l113_113272


namespace hyperbola_equation_l113_113582

theorem hyperbola_equation (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : 2 * b / a = 1) : 
  a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ (∀ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ↔ x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l113_113582


namespace cassie_water_intake_l113_113183

theorem cassie_water_intake :
  ∀ (ounces_per_bottle ounces_per_cup refill_per_day : ℕ),
    ounces_per_bottle = 16 →
    ounces_per_cup = 8 →
    refill_per_day = 6 →
    (refill_per_day * (ounces_per_bottle / ounces_per_cup) = 12) :=
by
  intros ounces_per_bottle ounces_per_cup refill_per_day h1 h2 h3
  rw [h1, h2, h3]
  decide

end cassie_water_intake_l113_113183


namespace low_degree_polys_condition_l113_113214

theorem low_degree_polys_condition :
  ∃ (f : Polynomial ℤ), ∃ (g : Polynomial ℤ), ∃ (h : Polynomial ℤ),
    (f = Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.X + 1 ∨
          f = Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2 * Polynomial.X + 2 ∨
          f = 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 2 * Polynomial.X + 1 ∨
          f = 2 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + Polynomial.X + 2) ∧
          f ^ 4 + 2 * f + 2 = (Polynomial.X ^ 4 + 2 * Polynomial.X ^ 2 + 2) * g + 3 * h := 
sorry

end low_degree_polys_condition_l113_113214


namespace broken_line_length_l113_113857

noncomputable def length_broken_line (x y : ℝ) :=
  abs(2 * y - abs x) - x = 2 ∧ -2 ≤ x ∧ x ≤ 1

theorem broken_line_length : 
  ∃ L, ∀ x y : ℝ, length_broken_line x y → (L ≈ 5.89 : ℝ) :=
sorry

end broken_line_length_l113_113857


namespace angle_of_inclination_of_line_l113_113701

theorem angle_of_inclination_of_line :
  let line_eq := (∀ x y : ℝ, sqrt 3 * x + 3 * y + 1 = 0) in
  let slope : ℝ := - (sqrt 3 / 3) in
  let θ : ℝ := π - arctan (real.abs (sqrt 3 / 3)) in
  θ = (5 * π / 6) :=
by
  sorry

end angle_of_inclination_of_line_l113_113701


namespace incorrect_statement_D_l113_113357

def f (x : ℝ) := Real.cos (x + Real.pi / 3)

theorem incorrect_statement_D :
  let A := ∀ x : ℝ, f (x + 2 * Real.pi) = f x
  let B := ∀ x : ℝ, f (8 * Real.pi / 3 - x) = f (8 * Real.pi / 3 + x)
  let C := f (Real.pi / 6 + Real.pi) = 0
  A → B → C → ¬ ∀ x, Real.pi / 2 < x ∧ x < Real.pi → f(x) < f(x + 1) :=
by
  sorry

end incorrect_statement_D_l113_113357


namespace chess_grandmaster_time_l113_113991

noncomputable def total_time (t_rules : ℕ) (t_proficiency : ℕ) (t_master : ℕ) : ℕ :=
  t_rules + t_proficiency + t_master

theorem chess_grandmaster_time : 
  let t_rules := 2 in
  let t_proficiency := 49 * t_rules in
  let t_master := 100 * (t_rules + t_proficiency) in
  total_time t_rules t_proficiency t_master = 10_100 := 
by 
  -- We need to prove that:
  -- t_rules + t_proficiency + t_master = 10_100
  sorry

end chess_grandmaster_time_l113_113991


namespace midpoint_locus_quartic_curve_l113_113065

variable (p : ℝ)

-- The parabola equation
def parabola (x y : ℝ) := y^2 - 2 * p * x = 0

-- The equation of the locus of the midpoint of the chord MN
def midpoint_locus (x y : ℝ) := y^4 - (p * (x - p) * y^2) + (p^4 / 2) = 0 ∧ y ≠ 0 ∧ x ≥ p * (1 + sqrt 2)

-- Proving the locus of the midpoint of the chord MN
theorem midpoint_locus_quartic_curve : 
  ∀ x y : ℝ, parabola x y → (∃ m : ℝ, y - m * x + p * (m + m^3 / 2) = 0) → midpoint_locus x y :=
by 
  intros x y parabola_eq normal_eq
  sorry

end midpoint_locus_quartic_curve_l113_113065


namespace new_person_weight_l113_113703

theorem new_person_weight (avg_increase : ℝ) (n : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 3 -> n = 4 -> old_weight = 70 -> 
  new_weight = old_weight + n * avg_increase -> 
  new_weight = 82 :=
by
  intros h_avg h_n h_old h_eq
  rw [h_avg, h_n, h_old] at h_eq
  linarith

end new_person_weight_l113_113703


namespace find_a7_of_arithmetic_sequence_l113_113970

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem find_a7_of_arithmetic_sequence (a d : ℤ)
  (h : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 +
       arithmetic_sequence a d 12 + arithmetic_sequence a d 13 = 24) :
  arithmetic_sequence a d 7 = 6 :=
by
  sorry

end find_a7_of_arithmetic_sequence_l113_113970


namespace apple_juice_less_than_cherry_punch_l113_113696

def orange_punch : ℝ := 4.5
def total_punch : ℝ := 21
def cherry_punch : ℝ := 2 * orange_punch
def combined_punch : ℝ := orange_punch + cherry_punch
def apple_juice : ℝ := total_punch - combined_punch

theorem apple_juice_less_than_cherry_punch : cherry_punch - apple_juice = 1.5 := by
  sorry

end apple_juice_less_than_cherry_punch_l113_113696


namespace remaining_frustum_fraction_l113_113809

/--
A square pyramid has a base edge of 64 inches and an altitude of 18 inches. A smaller square pyramid,
whose altitude is one-third of the original pyramid's altitude, is cut away from the apex.
Determine the volume of the remaining frustum as a fractional part of the volume of the original pyramid.
-/
def volume_of_pyramid (base_edge : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

def volume_of_frustum_fraction (base_edge : ℝ) (height : ℝ) (fraction : ℝ) : Prop :=
  let V := volume_of_pyramid base_edge height in
  let V_s := volume_of_pyramid (base_edge / 3) (height / 3) in
  let V_f := V - V_s in
  V_f / V = fraction

theorem remaining_frustum_fraction :
  volume_of_frustum_fraction 64 18 (263 / 272) :=
by
  sorry

end remaining_frustum_fraction_l113_113809


namespace has_inverse_l113_113844

noncomputable def p (x : ℝ) : ℝ := x^2
noncomputable def q (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def r (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)
noncomputable def s (x : ℝ) : ℝ := x / (x + 1)
noncomputable def t (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def u (x : ℝ) : ℝ := Real.log x + Real.log (10 - x)
noncomputable def v (x : ℝ) : ℝ := 2 * x - 3

-- Proving that p, s, and v have inverses
theorem has_inverse :
  (∀ x ∈ set.Icc (0 : ℝ) 3, function.injective p) ∧
  (∀ x ∈ set.Ici (0 : ℝ), function.injective s) ∧
  (∀ x ∈ set.Icc (-1 : ℝ) 2, function.injective v) :=
by
  -- Proof skipped
  sorry

end has_inverse_l113_113844


namespace trapezia_smaller_side_length_l113_113465

-- Define parameters for the problem
def side_length := 4 -- side length of equilateral triangle

-- Define the conditions in Lean
def height_equilateral_triangle := 2 * Real.sqrt 3
def area_equilateral_triangle := (4 * height_equilateral_triangle) / 2
def total_area_rectangle := 3 * area_equilateral_triangle
def height_rectangle := height_equilateral_triangle
def length_rectangle := total_area_rectangle / height_rectangle

-- Proof statement: length of the smaller parallel side of the trapezia
theorem trapezia_smaller_side_length :
  let a := Real.sqrt 3 in
  a = √3 :=
sorry

end trapezia_smaller_side_length_l113_113465


namespace bug_closest_point_l113_113125

theorem bug_closest_point :
  let r : ℝ := - 1 / 4,
      S_x : ℝ := 2 + sum (λ n : ℕ, r ^ (2 * n + 1) / 2),
      S_y : ℝ := 1 + sum (λ n : ℕ, r ^ (2 * n + 1) * (n + 1) / 2)
  in S_x = 8 / 5 ∧ S_y = 6 / 5 :=
by {
  sorry -- proof steps here
}

end bug_closest_point_l113_113125


namespace projection_locus_circle_l113_113629

noncomputable def locus_of_projections (l1 l2 : Set Point) (A : Point) (pi1 pi2 : Set Point) (l : Set Point) : Set Point :=
  sorry

theorem projection_locus_circle (l1 l2 : Set Point) (A : Point) (pi1 pi2 : Set Point) (l : Set Point)
  (h1 : skew l1 l2) 
  (h2 : A ∈ l1) 
  (h3 : ⟂ π1 π2) 
  (h4 : l = π1 ∩ π2)
  (h5 : right_dihedral_angle π1 π2) :
  ∃ B : Point, locus_of_projections l1 l2 A pi1 pi2 l = circle A B :=
    sorry

end projection_locus_circle_l113_113629


namespace part1_part2_l113_113885

variable {a : ℕ → ℝ}

-- Problem conditions
def a_seq (a : ℕ → ℝ) :=
  ∀ n : ℕ, a n = a (n + 1) + 2 * a n * a (n + 1)

def a_initial (a : ℕ → ℝ) := a 1 = 1 

-- Part 1: Prove {1 / a_n} is an arithmetic sequence
theorem part1 (h1 : a_seq a) (h2 : a_initial a) :
  ∃ d : ℝ, ∀ n : ℕ, (1 / a (n + 1)) - (1 / a n) = d := 
by 
  sorry

-- Create the sequence b_n = a_n * a_{n+1}
noncomputable def b : ℕ → ℝ := λ n, a n * a (n+1)

-- Part 2: Find S_n = sum of first n terms of b_n
theorem part2 (h1 : a_seq a) (h2 : a_initial a) (n : ℕ) :
  (Finset.range n).sum b = n / (2 * n + 1) := 
by 
  sorry

end part1_part2_l113_113885


namespace hexagon_perimeter_l113_113719

theorem hexagon_perimeter (s : ℕ) (P : ℕ) (h1 : s = 8) (h2 : 6 > 0) 
                          (h3 : P = 6 * s) : P = 48 := by
  sorry

end hexagon_perimeter_l113_113719


namespace lowest_score_on_one_of_last_two_tests_l113_113692

-- define conditions
variables (score1 score2 : ℕ) (total_score average desired_score : ℕ)

-- Shauna's scores on the first two tests are 82 and 75
def shauna_score1 := 82
def shauna_score2 := 75

-- Shauna wants to average 85 over 4 tests
def desired_average := 85
def number_of_tests := 4

-- total points needed for desired average
def total_points_needed := desired_average * number_of_tests

-- total points from first two tests
def total_first_two_tests := shauna_score1 + shauna_score2

-- total points needed on last two tests
def points_needed_last_two_tests := total_points_needed - total_first_two_tests

-- Prove the lowest score on one of the last two tests
theorem lowest_score_on_one_of_last_two_tests : 
  (∃ (score3 score4 : ℕ), score3 + score4 = points_needed_last_two_tests ∧ score3 ≤ 100 ∧ score4 ≤ 100 ∧ (score3 ≥ 83 ∨ score4 ≥ 83)) :=
sorry

end lowest_score_on_one_of_last_two_tests_l113_113692


namespace geometric_sequence_count_l113_113595

theorem geometric_sequence_count : 
  let c := 8000 in
  ∃ (a b : ℕ), a < b ∧ b < c ∧ (∃ r : ℚ, a = c / (r^2) ∧ b = c / r) ∧ ∀ c = 8000, 
  ∃ n : ℕ, n = 39 :=
by
  sorry

end geometric_sequence_count_l113_113595


namespace Olivia_money_left_l113_113017

theorem Olivia_money_left (initial_amount spend_amount : ℕ) (h1 : initial_amount = 128) 
  (h2 : spend_amount = 38) : initial_amount - spend_amount = 90 := by
  sorry

end Olivia_money_left_l113_113017


namespace determine_n_l113_113767

variable (a b n : ℕ)
variable (ab_nonzero : a * b ≠ 0)
variable (n_ge_2 : n ≥ 2)
variable (a_eq_2b : a = 2 * b)
variable (b_nonzero : b ≠ 0)

theorem determine_n (sum_terms_zero : (choose n 3) * b ^ (n - 3) + (choose n 4) * b ^ (n - 4) = 0) :
  n = 4 := by
  sorry

end determine_n_l113_113767


namespace set_intersection_l113_113920

open Set Real

theorem set_intersection (A : Set ℝ) (hA : A = {-1, 0, 1}) (B : Set ℝ) (hB : B = {y | ∃ x ∈ A, y = cos (π * x)}) :
  A ∩ B = {-1, 1} :=
by
  rw [hA, hB]
  -- remaining proof should go here
  sorry

end set_intersection_l113_113920


namespace tiffany_total_bags_l113_113741

def initial_bags : ℕ := 10
def found_on_tuesday : ℕ := 3
def found_on_wednesday : ℕ := 7
def total_bags : ℕ := 20

theorem tiffany_total_bags (initial_bags : ℕ) (found_on_tuesday : ℕ) (found_on_wednesday : ℕ) (total_bags : ℕ) :
    initial_bags + found_on_tuesday + found_on_wednesday = total_bags :=
by
  sorry

end tiffany_total_bags_l113_113741


namespace train_passing_time_l113_113413

/-
Problem:
Two trains of equal length are running on parallel lines in the same direction at 48 km/hr and 36 km/hr. 
The faster train passes the slower train in some time. 
The length of each train is 60 meters. 
Prove that the time it takes for the faster train to pass the slower train is 36 seconds.
-/

theorem train_passing_time
  (length : ℕ) 
  (speed_faster : ℕ) 
  (speed_slower : ℕ) 
  (length_eq : length = 60) 
  (speed_faster_eq : speed_faster = 48) 
  (speed_slower_eq : speed_slower = 36) :
  let relative_speed := (speed_faster - speed_slower) * (5 / 18 : ℝ) in
  let distance := (2 * length : ℝ) in
  time := distance / relative_speed ->
  time = 36 :=
  sorry

end train_passing_time_l113_113413


namespace items_sold_l113_113339

theorem items_sold (houses_day1 : ℕ) (items_per_house_day1 : ℕ) 
  (houses_day2 : ℕ) (sold_rate_day2 : ℚ) (items_per_house_day2 : ℕ) 
  (total_items_sold : ℕ) : 
  houses_day1 = 20 → 
  items_per_house_day1 = 2 →
  houses_day2 = 2 * houses_day1 → 
  sold_rate_day2 = 0.8 → 
  items_per_house_day2 = 2 → 
  total_items_sold = (houses_day1 * items_per_house_day1) + 
  (((houses_day2 : ℚ) * sold_rate_day2).toNat * items_per_house_day2) →
  total_items_sold = 104 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end items_sold_l113_113339


namespace john_trip_duration_l113_113993

theorem john_trip_duration :
  ∃ start_time end_time : ℕ,
  (start_time = 7 * 60 + 38) ∧  -- Start time at 7:38 a.m.
  (end_time = 15 * 60 + 45) ∧  -- End time at 3:45 p.m.
  (end_time - start_time = 8 * 60 + 7) :=  -- Duration Calculation
by {
  use 7 * 60 + 38,
  use 15 * 60 + 45,
  simp,
  split,
  refl,
  split,
  refl,
  simp,
  sorry
}

end john_trip_duration_l113_113993


namespace find_distance_l113_113343

noncomputable def distance_to_office (T : ℚ) :=
  let D := 40 * (T - 8/60) in
  D

theorem find_distance :
  ∀ (T : ℚ), 
  40 * (T - 8/60) = 30 * (T + 4/60) ∧
  50 * (T - 12/60) = 40 * (T - 8/60) ∧
  35 * (T + 2/60) = 40 * (T - 8/60) →
  distance_to_office T = 24 :=
by
  intro T
  intros h1 h2 h3
  sorry

end find_distance_l113_113343


namespace chessboard_equal_area_enclosed_l113_113407

theorem chessboard_equal_area_enclosed (polygonal_chain : set (ℕ × ℕ))
  (hchain : ∀ p ∈ polygonal_chain, (1 ≤ p.1 ∧ p.1 ≤ 8) ∧ (1 ≤ p.2 ∧ p.2 ≤ 8))
  (adjacency : ∀ p1 p2 ∈ polygonal_chain, p1 ≠ p2 → 
              (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 = p2.2 - 1)) ∨ 
              (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 = p2.1 - 1)) ∨ 
              ((p1.1 = p2.1 + 1 ∨ p1.1 = p2.1 - 1) ∧ (p1.2 = p2.2 + 1 ∨ p1.2 = p2.2 - 1)))
  (closed : ∀ p ∈ polygonal_chain, ∃ q ∈ polygonal_chain, p ≠ q) :
  let black_segments_area := ∑ p in polygonal_chain, if (p.1 + p.2) % 2 = 0 then 1 else 0,
      white_segments_area := ∑ p in polygonal_chain, if (p.1 + p.2) % 2 ≠ 0 then 1 else 0
  in black_segments_area = white_segments_area :=
sorry

end chessboard_equal_area_enclosed_l113_113407


namespace tennis_tournament_n_value_l113_113325

theorem tennis_tournament_n_value (n : ℕ) :
  let total_players := n + 4 * n,
      total_matches := total_players * (total_players - 1) / 2,
      ratio := 8 / 3 in
  ¬∃ x : ℕ, total_matches = 11 * x :=
by
  sorry

end tennis_tournament_n_value_l113_113325


namespace surface_area_of_revolution_volume_of_revolution_l113_113790

theorem surface_area_of_revolution (a b c t : ℝ) (r : ℝ := a / (2 * (Real.sin (Real.arccos (2 * t / (a * c))))))
  (h_cos_y : a * c * Real.cos (Real.arccos (2 * t / (a * c))) = 2 * t)
  (h_cos_x : a * b * Real.cos (Real.arccos (2 * t / (a * b))) = 2 * t)
  (h_radius : r = (a * b * c) / (4 * t)) :
  (2 * Math.pi * t * (b^2 + c^2) / (b * c) = a^2 * Math.pi) :=
by sorry

theorem volume_of_revolution (a b c t : ℝ) (h : b^2 + c^2 = a^2) (bc_two_t : b * c = 2 * t) :
  (4 * Math.pi * t^2 * (b^2 + c^2) / (3 * a * b * c) = (a * b * c * Math.pi) / 3) :=
by sorry

end surface_area_of_revolution_volume_of_revolution_l113_113790


namespace find_point_q_l113_113987

variable {A B C F G Q : Type}
variable [add_comm_group A] [add_comm_group B] [add_comm_group C]
variable [module ℝ A] [module ℝ B] [module ℝ C]
variable [module ℝ F] [module ℝ G] [module ℝ Q]

noncomputable def is_f_update (BF_ratio FC_ratio : ℝ) (B F C : A) : Prop := 
  BF_ratio * (F - B) = FC_ratio * (F - C)

noncomputable def is_g_update (AG_ratio GC_ratio : ℝ) (A G C : A) : Prop := 
  AG_ratio * (G - A) = GC_ratio * (G - C)

noncomputable def point_q (a b c : ℝ) (Q A B C : A) : Prop :=
  Q = a • A + b • B + c • C

theorem find_point_q
  {A B C : A} { F G Q : A}
  (hF : is_f_update 2 1 B F C)
  (hG : is_g_update 4 1 A G C)
{a b c : ℝ} (ha : a = 5 / 8) (hb : b = 3 / 8) (hc : c = 1 / 2) :
  point_q a b c Q A B C :=
begin
  have ha' : 5 / 8 + 3 / 8 + 1 / 2 = 1 := by norm_num,
  sorry
end

end find_point_q_l113_113987


namespace rhombus_area_l113_113266

theorem rhombus_area (x y : ℝ)
  (h1 : x^2 + y^2 = 113) 
  (h2 : x = y + 8) : 
  1 / 2 * (2 * y) * (2 * (y + 4)) = 97 := 
by 
  -- Assume x and y are the half-diagonals of the rhombus
  sorry

end rhombus_area_l113_113266


namespace place_5_5_in_A_makes_sum_div_3_l113_113992

def slip_numbers : List ℝ := [2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 4.5, 5, 5, 5.5, 6]

def cups_label : List String := ["A", "B", "C", "D", "E", "F"]

def placed_slips : List (String × ℝ) := [("A", 4.5), ("C", 3)]

def cup_sum_divisible_by_3 (cup : String) (slips : List (String × ℝ)) : Prop :=
  let cup_slips := slips.filter (λ pair, pair.1 = cup)
  (cup_slips.map (λ pair, pair.2)).sum % 3 = 0

theorem place_5_5_in_A_makes_sum_div_3 :
  cup_sum_divisible_by_3 "A" (("A", 5.5) :: placed_slips) :=
by
  -- skipped proof
  sorry

end place_5_5_in_A_makes_sum_div_3_l113_113992


namespace yolanda_walking_rate_l113_113770

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate_l113_113770


namespace shortest_distance_eq_l113_113652

noncomputable def shortest_distance_AB : ℝ :=
  let distance (a : ℝ) := abs (-a^2 + 6*a - 7) / real.sqrt 5 in
  real.inf (set.range distance)

theorem shortest_distance_eq :
  shortest_distance_AB = 2 * real.sqrt 5 / 5 :=
by
  sorry

end shortest_distance_eq_l113_113652


namespace equation_of_line_AC_equation_of_altitude_from_B_to_AB_l113_113562

structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 4 0
def B := Point.mk 6 7
def C := Point.mk 0 3

-- Define the line equation function passing through two points
def line_equation (P Q : Point) : ℝ → ℝ → Bool :=
  λ x y => ∃ a b c, a * P.x + b * P.y + c = 0 ∧ a * Q.x + b * Q.y + c = 0 ∧ a * x + b * y + c = 0

-- Define the altitudes function
def altitude_line (P Q R : Point) : ℝ → ℝ → Bool :=
  λ x y => 
    let slope := (Q.y - P.y) / (Q.x - P.x)
    let perp_slope := -1 / slope
    ∃ a b c, b = -a / perp_slope ∧ a * R.x + b * R.y + c = 0 ∧ a * x + b * y + c = 0

theorem equation_of_line_AC :
  ∀ x y, line_equation A C x y ↔ 3 * x + 4 * y - 12 = 0 :=
by
  sorry

theorem equation_of_altitude_from_B_to_AB :
  ∀ x y, altitude_line A B B x y ↔ 2 * x + 7 * y - 21 = 0 :=
by
  sorry

end equation_of_line_AC_equation_of_altitude_from_B_to_AB_l113_113562


namespace find_a_for_odd_function_l113_113948

theorem find_a_for_odd_function (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (f x = x / ((3 * x + 1) * (x - a))) ∧ f (-x) = -f x) → a = 1 / 3 :=
by
  let f := λ x : ℝ, x / ((3 * x + 1) * (x - a))
  sorry

end find_a_for_odd_function_l113_113948


namespace face_opposite_x_is_E_l113_113495

theorem face_opposite_x_is_E :
  let faces : Fin 6 → String := ![ "x", "A", "B", "C", "D", "E" ],
      face_x := 0,
      face_A := 1,
      face_B := 2,
      face_C := 3,
      face_D := 4,
      face_E := 5
  in
  -- Conditions
  (faces face_A = "A") ∧
  (faces face_B = "B") ∧
  (faces face_C = "C") ∧
  (faces face_D = "D") ∧
  (faces face_E = "E") ∧
  (faces face_A ∈ ["A", "B"]) ∧ -- Face with x is surrounded by A & B
  (faces face_B ∈ ["A", "B"]) ∧ -- Face with x is surrounded by A & B
  (faces face_A = "A") -- A is folded upwards from x
  (faces face_B = "B") -- B is folded rightwards from x
  (faces face_C = "C") -- C becomes the top face
  →
  -- Conclusion
  faces 5 = "E" := sorry

end face_opposite_x_is_E_l113_113495


namespace sample_size_is_50_l113_113088

theorem sample_size_is_50 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 800) (h2 : sample_students = 50) :
  sample_students = 50 :=
by
  -- Conditions given in the problem
  have h1 : total_students = 800 := h1,
  have h2 : sample_students = 50 := h2,
  -- The correct answer, based on the conditions
  exact h2

end sample_size_is_50_l113_113088


namespace find_values_of_x_y_l113_113442

theorem find_values_of_x_y (x y : ℝ) (I A C : Set ℝ)
  (hI : I = {2, 3, x^2 + 2*x - 3})
  (hA : A = {5})
  (hC_sub_I : C ⊆ I)
  (hCA : C \ A = {2, y}) :
  ((x = -4 ∨ x = 2) ∧ y = 3) :=
begin
  sorry
end

end find_values_of_x_y_l113_113442


namespace problem_statement_l113_113532

-- Define the given function
def f (x : ℝ) := (Real.log x) / (1 + x) - Real.log x

-- Define the auxiliary function g
def g (x : ℝ) := x + 1 + Real.log x

-- Prove the statements
theorem problem_statement {x₀ : ℝ} (h₀ : ∃! x, g x = 0) (h_max : ∀ x, f x₀ ≥ f x) :
  f x₀ = x₀ ∧ f x₀ < 1 / 2 :=
sorry

end problem_statement_l113_113532


namespace residue_calculation_mod_17_l113_113174

theorem residue_calculation_mod_17 :
  (∃ a b c : ℤ, a ≡ 220 [MOD 17] ∧ b ≡ 18 [MOD 17] ∧ c ≡ 28 [MOD 17]) →
  (((12 * 1 - 11 * 5 + 4) % 17) = 12) :=
by
  intros h
  rcases h with ⟨a, b, c, ha, hb, hc⟩
  have h1 : a * b % 17 = (12 * 1) % 17, by sorry
  have h2 : c * 5 % 17 = (11 * 5) % 17, by sorry
  have h3 : (12 * 1 - 11 * 5 + 4) % 17 = 12, by sorry
  exact h3

end residue_calculation_mod_17_l113_113174


namespace diminished_value_160_l113_113851

theorem diminished_value_160 (x : ℕ) (n : ℕ) : 
  (∀ m, m > 200 ∧ (∀ k, m = k * 180) → n = m) →
  (200 + x = n) →
  x = 160 :=
by
  sorry

end diminished_value_160_l113_113851


namespace tan_alpha_eq_neg2_l113_113069

theorem tan_alpha_eq_neg2 {α : ℝ} {x y : ℝ} (hx : x = -2) (hy : y = 4) (hM : (x, y) = (-2, 4)) :
  Real.tan α = -2 :=
by
  sorry

end tan_alpha_eq_neg2_l113_113069


namespace bread_slices_remaining_l113_113013

theorem bread_slices_remaining 
  (total_slices : ℕ)
  (third_eaten: ℕ)
  (slices_eaten_breakfast : total_slices / 3 = third_eaten)
  (slices_after_breakfast : total_slices - third_eaten = 8)
  (slices_used_lunch : 2)
  (slices_remaining : 8 - slices_used_lunch = 6) : 
  total_slices = 12 → third_eaten = 4 → slices_remaining = 6 := by 
  sorry

end bread_slices_remaining_l113_113013


namespace find_first_term_and_common_ratio_find_S_n_l113_113253

-- Definition of the geometric sequence and conditions
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n

-- Definition of the arithmetic sequence
def arithmetic_seq (b : ℕ → ℝ) (d : ℝ) := ∀ n, b (n + 1) = b n + d

-- Main theorem
theorem find_first_term_and_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : geometric_seq a q)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_product : a 2 * a 4 * a 6 = 512)
  (h_arith : arithmetic_seq (λ n, a (2 * n) - [1, 3, 9].nth n.get) (a 3 - a 1)) :
  a 0 = 2 ∧ q = real.sqrt 2 :=
by
  sorry

-- Another theorem to find S_n
theorem find_S_n (q : ℝ) :
  q = real.sqrt 2 →
  ∀ n, (∑ i in finset.range n, (λ n, (real.sqrt 2)^(n + 1)) i ^ 2) = 2^(n + 2) - 4 :=
by
  sorry

end find_first_term_and_common_ratio_find_S_n_l113_113253


namespace pentagon_parallel_AE_BD_l113_113638

-- Define the vertices and segments in a pentagon
variables {A B C D E : Type}

-- Conditions
variables {parallel_ab_ce : ∀ A B C E, Parallel AB CE}
variables {parallel_bc_ad : ∀ B C A D, Parallel BC AD}
variables {parallel_cd_be : ∀ C D B E, Parallel CD BE}
variables {parallel_de_ac : ∀ D E A C, Parallel DE AC}

-- Prove that AE is parallel to BD
theorem pentagon_parallel_AE_BD 
  (h1 : Parallel AB CE)
  (h2 : Parallel BC AD)
  (h3 : Parallel CD BE)
  (h4 : Parallel DE AC) : Parallel AE BD :=
by { sorry }

end pentagon_parallel_AE_BD_l113_113638


namespace option_D_is_linear_l113_113103

-- Define the conditions
def option_A : Prop := ¬(∃ (c : ℝ), ∀ (x y : ℝ), 2 * x - y = c)
def option_B : Prop := ¬(∃ (a b c : ℝ), a * x * y + b * x - 2 = c)
def option_C : Prop := ¬(∃ (a b c : ℝ), 2 / x - y = c)
def option_D : Prop := ∃ (a b c : ℝ), a * x + b * y = c

-- Main theorem statement
theorem option_D_is_linear : 
  option_A → option_B → option_C → option_D :=
by sorry

end option_D_is_linear_l113_113103


namespace average_salary_feb_mar_apr_may_l113_113386

theorem average_salary_feb_mar_apr_may 
  (average_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_months_1 : ℤ)
  (total_months_2 : ℤ)
  (total_sum_jan_apr : average_jan_feb_mar_apr * (total_months_1:ℝ) = 32000)
  (january_salary: salary_jan = 4700)
  (may_salary: salary_may = 6500)
  (total_months_1_eq: total_months_1 = 4)
  (total_months_2_eq: total_months_2 = 4):
  average_jan_feb_mar_apr * (total_months_1:ℝ) - salary_jan + salary_may/total_months_2 = 8450 :=
by
  sorry

end average_salary_feb_mar_apr_may_l113_113386


namespace sum_of_seven_smallest_multiples_of_12_l113_113761

theorem sum_of_seven_smallest_multiples_of_12 : 
  ∑ i in (Finset.range 7).map (λ i, 12 * (i + 1)) = 336 :=
by
  sorry

end sum_of_seven_smallest_multiples_of_12_l113_113761


namespace derivative_of_function_l113_113113

noncomputable def derivative_expr (x : ℝ) : ℝ :=
  (    (derivative (λ x : ℝ, (∛(Real.cot 2)) - (1/20) * (cos (10*x))^2 / (sin (20*x))))
         x) 

theorem derivative_of_function (x : ℝ) : 
 (derivative_expr x) = (1/(4 * (sin (10 * x))^2)) :=
 by
  sorry

end derivative_of_function_l113_113113


namespace calvin_wins_l113_113348

-- Definitions based on the conditions
variables 
  (k N : ℕ)
  (h_k : k ≥ 1)
  (h_N : N > 1)

-- 2N + 1 coins on a circle, all initially showing heads
def coins := list.repeat tt (2 * N + 1)

-- Calvin can turn any coin from heads to tails, Hobbes can turn at most one adjacent coin from tails to heads
-- Calvin wins if at any moment there are k coins showing tails after Hobbes has made his move
theorem calvin_wins : k ≤ N + 1 :=
by
  sorry

end calvin_wins_l113_113348


namespace max_min_abs_diff_eq_one_div_n_max_sum_abs_diff_eq_two_sub_two_div_n_l113_113482

variables {n : ℕ}
variables (x y : Fin n → ℝ)
variables (hx : ∀ i j : Fin n, i ≤ j → x i ≤ x j)
variables (hy : ∀ i j : Fin n, i ≤ j → y i ≤ y j)
variables (hx_nonneg : ∀ i : Fin n, 0 ≤ x i)
variables (hy_nonneg : ∀ i : Fin n, 0 ≤ y i)
variables (hx_sum : Finset.sum Finset.univ x = 1)
variables (hy_sum : Finset.sum Finset.univ y = 1)

theorem max_min_abs_diff_eq_one_div_n :
  min (Fin n) (λ i, abs (x i - y i)) = 1 / n := sorry

theorem max_sum_abs_diff_eq_two_sub_two_div_n :
  Finset.sum Finset.univ (λ i, abs (x i - y i)) = 2 - 2 / n := sorry

end max_min_abs_diff_eq_one_div_n_max_sum_abs_diff_eq_two_sub_two_div_n_l113_113482


namespace stock_yield_percentage_l113_113479

def annualDividend (parValue : ℕ) (rate : ℕ) : ℕ :=
  (parValue * rate) / 100

def yieldPercentage (dividend : ℕ) (marketPrice : ℕ) : ℕ :=
  (dividend * 100) / marketPrice

theorem stock_yield_percentage :
  let par_value := 100
  let rate := 8
  let market_price := 80
  yieldPercentage (annualDividend par_value rate) market_price = 10 :=
by
  sorry

end stock_yield_percentage_l113_113479


namespace limit_geometric_sum_l113_113865

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem limit_geometric_sum :
  ∀ (a : ℝ) (q : ℝ), a = 1 → q = 1/3 → (tendsto (λ n, geometric_sum a q n) at_top (𝓝 (3 / 2))) :=
by
  intros a q ha hq
  
  have h : geometric_sum a q = λ n, a * (1 - q^n) / (1 - q),
  { sorry },
  
  rw [ha, hq] at *,
  
  sorry

end limit_geometric_sum_l113_113865


namespace repeating_decimal_to_fraction_l113_113210

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end repeating_decimal_to_fraction_l113_113210


namespace find_number_l113_113601

theorem find_number (x : ℚ) : (35 / 100) * x = (20 / 100) * 50 → x = 200 / 7 :=
by
  intros h
  sorry

end find_number_l113_113601


namespace meal_cost_per_person_l113_113162

/-
Problem Statement:
Prove that the cost per meal is $3 given the conditions:
- There are 2 adults and 5 children.
- The total bill is $21.
-/

theorem meal_cost_per_person (total_adults : ℕ) (total_children : ℕ) (total_bill : ℝ) 
(total_people : ℕ) (cost_per_meal : ℝ) : 
total_adults = 2 → total_children = 5 → total_bill = 21 → total_people = total_adults + total_children →
cost_per_meal = total_bill / total_people → 
cost_per_meal = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end meal_cost_per_person_l113_113162


namespace smallest_n_exists_l113_113886

variables (V : Finset Point) (E : Finset (Set Point))
open Nat

-- Define the points in V and specify there are 2019 points
def V_set_properties (V : Finset Point) : Prop :=
  V.card = 2019 ∧ (∀ P1 P2 P3 P4 ∈ V, ¬ coplanar P1 P2 P3 P4)

-- Define the edges in E and specify properties
def E_set_properties (E : Finset (Set Point)) (V : Finset Point) : Prop :=
  ∀ e ∈ E, ∃ P1 P2 ∈ V, e = {P1, P2}

theorem smallest_n_exists
  (V : Finset Point) (E : Finset (Set Point)) 
  (hV : V_set_properties V) (hE : E_set_properties E V) :
  ∃ (n : ℕ), n = 2795 ∧ ∀ E' ⊆ E, E'.card ≥ n → 
    (∃ T : Finset (Set (Set Point)), T.card = 908 ∧ (∀ t ∈ T, ∃ a b c ∈ V, t = {a, b} ∧ {a, c} ∈ E')) :=
sorry

end smallest_n_exists_l113_113886


namespace question_digits_to_the_right_l113_113932

noncomputable def digits_to_the_right_of_decimal (r : ℚ) : ℕ :=
  (r.num : ℚ).denom

theorem question_digits_to_the_right :
  digits_to_the_right_of_decimal (5^7 / (10^5 * 15625)) = 5 :=
by
  have h0 : 15625 = 5 ^ 6 := by norm_num
  have h1 : 5^7 / (10^5 * 15625) = 1 / 20000 := by
    rw [h0]
    norm_num
  have h2 : (1 / 20000 : ℚ) = 0.00005 := by norm_num
  sorry

end question_digits_to_the_right_l113_113932


namespace time_for_a_and_b_to_complete_work_l113_113772

noncomputable def work_rate_b : ℚ := 1 / 30
noncomputable def work_rate_a : ℚ := 2 * work_rate_b
noncomputable def combined_work_rate : ℚ := work_rate_a + work_rate_b
noncomputable def time_to_complete_work : ℚ := 1 / combined_work_rate

theorem time_for_a_and_b_to_complete_work :
  (a_rate_is_twice_b_rate : work_rate_a = 2 * work_rate_b)
  (b_completes_in_30_days : work_rate_b = 1 / 30) :
  time_to_complete_work = 10 := sorry

end time_for_a_and_b_to_complete_work_l113_113772


namespace num_correct_propositions_l113_113155

theorem num_correct_propositions (z : ℂ) (a b : ℂ) (i : ℂ) (h1 : z = a + b * I) 
(h2 : ‖z + 1‖ = ‖z - 2 * I‖) 
(h3 : ¬ (‖v‖^2 = v^2 → (‖z‖^2 = z^2))) 
(h4 : 1 + i + i^2 + ... + i^2015 = 0) : 
finset.count true [false, true, false, false] = 1 := by
  sorry

end num_correct_propositions_l113_113155


namespace vector_subtraction_l113_113290

def a : Real × Real := (2, -1)
def b : Real × Real := (-2, 3)

theorem vector_subtraction :
  a.1 - 2 * b.1 = 6 ∧ a.2 - 2 * b.2 = -7 := by
  sorry

end vector_subtraction_l113_113290


namespace work_duration_l113_113946

variable (a b c : ℕ)
variable (daysTogether daysA daysB daysC : ℕ)

theorem work_duration (H1 : daysTogether = 4)
                      (H2 : daysA = 12)
                      (H3 : daysB = 18)
                      (H4: a = 1 / 12)
                      (H5: b = 1 / 18)
                      (H6: 1 / daysTogether = 1 / daysA + 1 / daysB + 1 / daysC) :
                      daysC = 9 :=
sorry

end work_duration_l113_113946


namespace sum_of_seven_smallest_multiples_of_12_l113_113760

theorem sum_of_seven_smallest_multiples_of_12 : 
  ∑ i in (Finset.range 7).map (λ i, 12 * (i + 1)) = 336 :=
by
  sorry

end sum_of_seven_smallest_multiples_of_12_l113_113760


namespace perpendicular_line_plane_l113_113240

variable {P : Type*} [EuclideanGeometry P]

-- Definitions of lines m, n, and plane α
variable (m n : Line P) (α : Plane P)

-- Definitions of perpendicularity (⊥) and subset (⊆)
def perp_to_plane (l : Line P) (π : Plane P) : Prop := 
  -- definition here, denoting l ⊥ π
  sorry

def line_in_plane (l : Line P) (π : Plane P) : Prop :=
  -- definition here, denoting l ⊆ π
  sorry

-- Problem statement
theorem perpendicular_line_plane (hmα : perp_to_plane m α) 
  (hnα : line_in_plane n α) : perp m n :=
  sorry

end perpendicular_line_plane_l113_113240


namespace quadrilateral_area_l113_113246

noncomputable def AB : ℝ := 3
noncomputable def BC : ℝ := 3
noncomputable def CD : ℝ := 4
noncomputable def DA : ℝ := 8
noncomputable def angle_DAB_add_angle_ABC : ℝ := 180

theorem quadrilateral_area :
  AB = 3 ∧ BC = 3 ∧ CD = 4 ∧ DA = 8 ∧ angle_DAB_add_angle_ABC = 180 →
  ∃ area : ℝ, area = 13.2 :=
by {
  sorry
}

end quadrilateral_area_l113_113246


namespace maximal_correlation_coefficient_representation_independence_iff_maximal_correlation_is_zero_maximal_correlation_coefficient_of_pairs_l113_113435

noncomputable def maximal_correlation_coefficient (X : ℝ^n) (Y : ℝ^m) : ℝ :=
  sorry  -- Sup over all Borel functions f,g of ρ(f(X), g(Y))

variables (X Y : vector ℝ n) (X₁ X₂ : vector ℝ n) (Y₁ Y₂ : vector ℝ m)

-- Part (a)
theorem maximal_correlation_coefficient_representation :
  ∀X Y, maximal_correlation_coefficient X Y = 
    (sup (λ f g, E[λ x y, f(x) * g(y)])) ∧ 
    (sup (λ f, √(E[λ y, (E[λ x, f(x)]|y)^2])))) :=
sorry

-- Part (b)
theorem independence_iff_maximal_correlation_is_zero :
  ∀X Y, maximal_correlation_coefficient X Y = 0 ↔ independent X Y :=
sorry

-- Part (c)
theorem maximal_correlation_coefficient_of_pairs :
  ∀X₁ Y₁ X₂ Y₂, independent (X₁, Y₁) (X₂, Y₂) → 
  maximal_correlation_coefficient (X₁, X₂) (Y₁, Y₂) = 
    max (maximal_correlation_coefficient X₁ Y₁) (maximal_correlation_coefficient X₂ Y₂) :=
sorry

end maximal_correlation_coefficient_representation_independence_iff_maximal_correlation_is_zero_maximal_correlation_coefficient_of_pairs_l113_113435


namespace olga_grandchildren_l113_113928

theorem olga_grandchildren :
  (∃ num_daughters_per_son : ℕ,
    (3 * 6) + (3 * num_daughters_per_son) = 33) :=
by
  use 5
  sorry

end olga_grandchildren_l113_113928


namespace area_large_sphere_trace_l113_113145

-- Define the conditions
def radius_small_sphere : ℝ := 4
def radius_large_sphere : ℝ := 6
def area_small_sphere_trace : ℝ := 37

-- Define the mathematically equivalent proof problem
theorem area_large_sphere_trace :
  let r1 := radius_small_sphere,
      r2 := radius_large_sphere,
      a1 := area_small_sphere_trace,
      ratio := (r2 / r1) ^ 2 in
  a1 * ratio = 83.25 := by
sorry

end area_large_sphere_trace_l113_113145


namespace problem_1_problem_2_l113_113262

noncomputable def minValueFunc (x : ℝ) : ℝ := (5 / x) + (9 / (1 - 5 * x))

theorem problem_1 (m n x y : ℝ) (h_mn : m ≠ n) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
                   (h_x_pos : 0 < x) (h_y_pos : 0 < y) :
  (m^2 / x) + (n^2 / y) > ((m + n)^2 / (x + y)) := by
  sorry

theorem problem_2 (x : ℝ) (h_x_bounds : 0 < x ∧ x < 1/5) :
  minValueFunc x ≥ 64 ∧ (minValueFunc x = 64 ↔ x = 1/8) := by
  have h_min := problem_1 5 3 (5 * x) (1 - 5 * x) by sorry
  sorry

end problem_1_problem_2_l113_113262


namespace evaluate_expression_l113_113179

-- Define the numerator and denominator
def num := 2^2 + 2^1 + 2^0
def den := 2^(-1) + 2^(-2) + 2^(-3)

-- Prove that the given expression evaluates to 8
theorem evaluate_expression : (num / den) = 8 := by
  sorry

end evaluate_expression_l113_113179


namespace message_spread_in_24_hours_l113_113461

theorem message_spread_in_24_hours : ∃ T : ℕ, (T = (2^25 - 1)) :=
by 
  let T := 2^24 - 1
  use T
  sorry

end message_spread_in_24_hours_l113_113461


namespace ordered_pair_condition_l113_113067

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def psi : ℝ := (1 - Real.sqrt 5) / 2

def c : ℕ → ℝ
| 0     := 1
| 1     := 0
| (n+2) := c (n+1) + c n

def S : Set (ℕ × ℕ) := { p | ∃ (J : Finset ℕ), p = (J.sum (λ j, c j), J.sum (λ j, c (j - 1))) }

theorem ordered_pair_condition (x y : ℕ) :
  ∃ (alpha beta m M : ℝ), alpha = psi ∧ beta = 1 ∧ m = -1 ∧ M = phi ∧ 
  (-1 < alpha * x + beta * y ∧ alpha * x + beta * y < phi ↔ (x, y) ∈ S) := 
sorry

end ordered_pair_condition_l113_113067


namespace cos_phi_between_u_and_v_l113_113289

open Real

variables (u v : EuclideanAffineSpace 2)

def norm (x : EuclideanAffineSpace 2) : Real := sqrt (dot x x)

theorem cos_phi_between_u_and_v
    (h1 : norm u = 5)
    (h2 : norm v = 7)
    (h3 : norm (u + v) = 9) : 
    (dot u v) / (norm u * norm v) = 1 / 10 := 
by
 sorry

end cos_phi_between_u_and_v_l113_113289


namespace adam_action_figures_per_shelf_l113_113816

-- Define the number of shelves and the total number of action figures
def shelves : ℕ := 4
def total_action_figures : ℕ := 44

-- Define the number of action figures per shelf
def action_figures_per_shelf : ℕ := total_action_figures / shelves

-- State the theorem to be proven
theorem adam_action_figures_per_shelf : action_figures_per_shelf = 11 :=
by sorry

end adam_action_figures_per_shelf_l113_113816


namespace incorrect_reasoning_error_l113_113083

-- Define the conditions
variables (M P S : Type) 
variables (isP : M → P) (isP_S : S → P)

-- Define the form of reasoning
def incorrect_reasoning (m : M) (s : S) : M :=
  sorry

-- State the theorem that the conclusion is wrong due to incorrect reasoning
theorem incorrect_reasoning_error :
  (∀ m: M, isP m) → (∀ s: S, isP_S s) → (incorrect_reasoning ≠ (by {assume m: M, m })) :=
sorry

end incorrect_reasoning_error_l113_113083


namespace crayons_lost_or_given_away_total_l113_113868

def initial_crayons_box1 := 479
def initial_crayons_box2 := 352
def initial_crayons_box3 := 621

def remaining_crayons_box1 := 134
def remaining_crayons_box2 := 221
def remaining_crayons_box3 := 487

def crayons_lost_or_given_away_box1 := initial_crayons_box1 - remaining_crayons_box1
def crayons_lost_or_given_away_box2 := initial_crayons_box2 - remaining_crayons_box2
def crayons_lost_or_given_away_box3 := initial_crayons_box3 - remaining_crayons_box3

def total_crayons_lost_or_given_away := crayons_lost_or_given_away_box1 + crayons_lost_or_given_away_box2 + crayons_lost_or_given_away_box3

theorem crayons_lost_or_given_away_total : total_crayons_lost_or_given_away = 610 :=
by
  -- Proof should go here
  sorry

end crayons_lost_or_given_away_total_l113_113868


namespace hyperbola_with_foci_on_y_axis_l113_113976

variable (m n : ℝ)

-- condition stating that mn < 0
def mn_neg : Prop := m * n < 0

-- the main theorem statement
theorem hyperbola_with_foci_on_y_axis (h : mn_neg m n) : 
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, m * x^2 - m * y^2 = n ↔ y^2 - x^2 = a) :=
sorry

end hyperbola_with_foci_on_y_axis_l113_113976


namespace max_f_val_l113_113523

noncomputable def f (x : ℝ) : ℝ := (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_f_val : ∃ (x : ℝ), f(x) = 6 / 5 :=
by
  sorry

end max_f_val_l113_113523


namespace martha_prob_exactly_10_correct_l113_113363

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def probability_correct (right wrong : ℚ) (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (right ^ k) * (wrong ^ (n - k))

theorem martha_prob_exactly_10_correct :
  let n := 20
  let k := 10
  let right := (1 : ℚ) / 4
  let wrong := (3 : ℚ) / 4
  probability_correct right wrong n k = 93350805 / 1073741824 :=
by
  sorry

end martha_prob_exactly_10_correct_l113_113363


namespace profit_margin_A_cost_price_B_units_purchased_l113_113127

variables (cost_price_A selling_price_A selling_price_B profit_margin_B total_units total_cost : ℕ)
variables (units_A units_B : ℕ)

-- Conditions
def condition1 : cost_price_A = 40 := sorry
def condition2 : selling_price_A = 60 := sorry
def condition3 : selling_price_B = 80 := sorry
def condition4 : profit_margin_B = 60 := sorry
def condition5 : total_units = 50 := sorry
def condition6 : total_cost = 2200 := sorry

-- Proof statements 
theorem profit_margin_A (h1 : cost_price_A = 40) (h2 : selling_price_A = 60) :
  (selling_price_A - cost_price_A) * 100 / cost_price_A = 50 :=
by sorry

theorem cost_price_B (h3 : selling_price_B = 80) (h4 : profit_margin_B = 60) :
  (selling_price_B * 100) / (100 + profit_margin_B) = 50 :=
by sorry

theorem units_purchased (h5 : 40 * units_A + 50 * units_B = 2200)
  (h6 : units_A + units_B = 50) :
  units_A = 30 ∧ units_B = 20 :=
by sorry


end profit_margin_A_cost_price_B_units_purchased_l113_113127


namespace range_of_x_l113_113567

def f (x : ℝ) : ℝ := 3 * x - 2

theorem range_of_x {k : ℕ} (hk : k > 0) (x : ℝ) :
  (∀ n, 1 ≤ n ∧ n < k → f^[n] x ≤ 244) ∧ (f^[k] x > 244) → 
  x ∈ (3^(5 - k) + 1 : ℝ, 3^(6 - k) + 1 : ℝ] :=
sorry

end range_of_x_l113_113567


namespace temperature_change_l113_113616

theorem temperature_change
    (rate1 : ℕ := 3) (years1 : ℕ := 300)
    (rate2 : ℕ := 5) (years2 : ℕ := 200)
    (rate3 : ℕ := 2) (years3 : ℕ := 200)
    (century : ℕ := 100)
    (rate1_years1_century : 3 * 3 = 9)
    (rate2_years2_century : 5 * 2 = 10)
    (rate3_years3_century : 2 * 2 = 4)
    (conversion_factor : ℕ := 1.8)
    (conversion_offset : ℕ := 32): (rate1 * (years1 / century) + rate2 * (years2 / century) + rate3 * (years3 / century) = 23) ∧ (23 * conversion_factor + conversion_offset = 73.4) :=
by
  sorry

end temperature_change_l113_113616


namespace evaluate_expression_l113_113440

theorem evaluate_expression : (1.2^3 - (0.9^3 / 1.2^2) + 1.08 + 0.9^2 = 3.11175) :=
by
  sorry -- Proof goes here

end evaluate_expression_l113_113440


namespace ratio_of_packets_to_tent_stakes_l113_113688

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes_l113_113688


namespace calculate_f_g_cubic_root_of_2_l113_113281

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_f_g_cubic_root_of_2 : f (g (real.cbrt 2)) = -11 := by
  sorry

end calculate_f_g_cubic_root_of_2_l113_113281


namespace div_condition_nat_l113_113853

theorem div_condition_nat (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 :=
by
  sorry

end div_condition_nat_l113_113853


namespace sum_of_common_divisors_l113_113406

theorem sum_of_common_divisors :
  ∀ (d : ℕ), d ∈ ({1, 2, 4, 5} : Finset ℕ) →
    d ∣ 48 ∧
    d ∣ 80 ∧
    d ∣ -20 ∧
    d ∣ 180 ∧
    d ∣ 120 →
  ∑ d in ({1, 2, 4, 5} : Finset ℕ), d = 12 := by
  -- The proof goes here
  sorry

end sum_of_common_divisors_l113_113406


namespace temperature_decrease_time_l113_113732

theorem temperature_decrease_time
  (T_initial T_final T_per_hour : ℤ)
  (h_initial : T_initial = -5)
  (h_final : T_final = -25)
  (h_decrease : T_per_hour = -5) :
  (T_final - T_initial) / T_per_hour = 4 := by
sorry

end temperature_decrease_time_l113_113732


namespace solve_other_endpoint_l113_113062

structure Point where
  x : ℤ
  y : ℤ

def midpoint : Point := { x := 3, y := 1 }
def known_endpoint : Point := { x := 7, y := -3 }

def calculate_other_endpoint (m k : Point) : Point :=
  let x2 := 2 * m.x - k.x;
  let y2 := 2 * m.y - k.y;
  { x := x2, y := y2 }

theorem solve_other_endpoint : calculate_other_endpoint midpoint known_endpoint = { x := -1, y := 5 } :=
  sorry

end solve_other_endpoint_l113_113062


namespace balls_removed_l113_113200

theorem balls_removed (original current removed : ℕ) (h₀ : original = 8) (h₁ : current = 6) (h₂ : removed = original - current) :
  removed = 2 :=
by
  rw [h₂, h₀, h₁]
  norm_num

end balls_removed_l113_113200


namespace chord_length_of_intercepted_line_l113_113639

-- Definitions of polar to rectangular conversions and distance formula.
def line_in_rectangular (x y : ℝ) : Prop := x + y = 2 * Real.sqrt 2
def circle_in_rectangular (x y : ℝ) : Prop := x^2 + y^2 = 8

theorem chord_length_of_intercepted_line :
  ∃ length : ℝ, length = 4 ∧ (∀ x y : ℝ, circle_in_rectangular x y → line_in_rectangular x y → 2 * Real.sqrt ((2 * Real.sqrt 2)^2 - 2^2) = 4) :=
begin
  sorry
end

end chord_length_of_intercepted_line_l113_113639


namespace shekar_math_marks_l113_113029

variable (science socialStudies english biology average : ℕ)

theorem shekar_math_marks 
  (h1 : science = 65)
  (h2 : socialStudies = 82)
  (h3 : english = 67)
  (h4 : biology = 95)
  (h5 : average = 77) :
  ∃ M, average = (science + socialStudies + english + biology + M) / 5 ∧ M = 76 :=
by
  sorry

end shekar_math_marks_l113_113029


namespace division_remainder_l113_113108

theorem division_remainder (q d r : ℕ) (h₁ : q = 432) (h₂ : d = 44) (h₃ : r = 0) :
  let a := q * d + r in
  (a % 31) = 5 :=
by
  let a := q * d + r
  have h₄ : a = 19008 := by rw [h₁, h₂, h₃]; norm_num
  have h₅ : 19008 % 31 = 5 := by norm_num
  rw [h₄, h₅]; reflexivity

end division_remainder_l113_113108


namespace find_f3_l113_113897

noncomputable def f : ℝ → ℝ := λ x =>
  if h : 0 ≤ x ∧ x ≤ 2 then 4^x + 3 / x else 4^(x % 2) + 3 / (x % 2)

theorem find_f3 : f 3 = 7 :=
by
  have h₁ : f x = f (x % 2),
  { intro x,
    dsimp [f],
    split_ifs,
    sorry },
  have h₂ : f 1 = 4^1 + 3 / 1 := rfl,
  calc
    f 3 = f (3 % 2) := h₁ 3
      ... = f 1 := by norm_num
      ... = 7     := h₂

end find_f3_l113_113897


namespace rational_sqrt_rational_l113_113651

noncomputable theory

open Real

theorem rational_sqrt_rational {m n p : ℚ} 
  (h1 : ∃ q : ℚ, q = (sqrt m.to_real + sqrt n.to_real + sqrt p.to_real)) : 
  (∃ r1 r2 r3 : ℚ, sqrt m.to_real = r1 ∧ sqrt n.to_real = r2 ∧ sqrt p.to_real = r3) :=
sorry

end rational_sqrt_rational_l113_113651


namespace find_QP_squared_l113_113968

theorem find_QP_squared :
  ∀ (O1 O2 P Q R : Point)
    (r1 r2 d : ℝ),
    circle O1 r1 →
    circle O2 r2 →
    dist O1 O2 = d →
    r1 = 9 →
    r2 = 7 →
    d = 14 →
    dist Q P = dist P R →
    (dist Q P)^2 = 170 :=
by
  intros
  sorry

end find_QP_squared_l113_113968


namespace vectors_reorder_l113_113773

/-- Given n vectors in ℝ², each of length 1 and sum to zero,
there exists a reordering such that the length of the sum of the 
first k vectors does not exceed √5 for any k. -/
theorem vectors_reorder (n : ℕ) (v : fin n → ℝ × ℝ)
  (h1 : ∀ i, ‖v i‖ = 1) 
  (h2 : finset.univ.sum v = (0, 0)) :
  ∃ σ : equiv.perm (fin n), 
    ∀ k, k ∈ finset.range n → ‖(finset.range (k + 1)).sum (λ i, v (σ i))‖ ≤ real.sqrt 5 :=
begin
  sorry
end

end vectors_reorder_l113_113773


namespace area_comet_tail_region_calculation_l113_113977

noncomputable def area_of_comet_tail_region : ℝ :=
  let r1 := 6
  let r2 := 2
  let side := 6
  let larger_quarter_circle_area := (π * r1^2) / 4
  let smaller_quarter_circle_area := (π * r2^2) / 4
  let quarter_square_area := (side^2) / 4
  larger_quarter_circle_area - (smaller_quarter_circle_area + quarter_square_area)

theorem area_comet_tail_region_calculation : area_of_comet_tail_region = 8 * π - 9 :=
  sorry

end area_comet_tail_region_calculation_l113_113977


namespace bread_slices_remaining_l113_113014

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end bread_slices_remaining_l113_113014


namespace john_initial_socks_l113_113344

theorem john_initial_socks (X : ℕ) (threw_away : ℕ) (bought : ℕ) (currently_has : ℕ) 
  (h1 : threw_away = 19) (h2 : bought = 13) (h3 : currently_has = 27) :
  X - threw_away + bought = currently_has → X = 33 :=
by
  intro h
  have : X - 19 + 13 = 27 := by
    rw [h1, h2, h3] at h
    exact h
  linarith

end john_initial_socks_l113_113344


namespace exists_composite_expression_l113_113685

-- Define what it means for a number to be composite
def is_composite (m : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m

-- Main theorem statement
theorem exists_composite_expression :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → is_composite (n * 2^k + 1) :=
sorry

end exists_composite_expression_l113_113685


namespace total_blankets_collected_l113_113229

/-- 
Freddie and his team are collecting blankets for three days to be donated to the Children Shelter Organization.
- There are 15 people on the team.
- On the first day, each of them gave 2 blankets.
- On the second day, they tripled the number they collected on the first day by asking door-to-door.
- On the last day, they set up boxes at schools and got a total of 22 blankets.

Prove that the total number of blankets collected for the three days is 142.
-/
theorem total_blankets_collected:
  let people := 15 in
  let blankets_per_person_first_day := 2 in
  let blankets_first_day := people * blankets_per_person_first_day in
  let blankets_second_day := blankets_first_day * 3 in
  let blankets_third_day := 22 in
  blankets_first_day + blankets_second_day + blankets_third_day = 142 :=
by
  sorry

end total_blankets_collected_l113_113229


namespace set_problems_l113_113355

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_problems :
  (A ∩ B = ({4} : Set ℤ)) ∧
  (A ∪ B = ({1, 2, 4, 5, 6, 7, 8, 9, 10} : Set ℤ)) ∧
  (U \ (A ∪ B) = ({3} : Set ℤ)) ∧
  ((U \ A) ∩ (U \ B) = ({3} : Set ℤ)) :=
by
  sorry

end set_problems_l113_113355


namespace hexagon_area_l113_113689

/-- Prove that the area of the regular hexagon XYZVWU with vertices at (0,0) and (8,2) is 17√3 / 2 --/
theorem hexagon_area :
  let X := (0 : ℝ, 0 : ℝ)
  let V := (8 : ℝ, 2 : ℝ)
  ∀ (s : ℝ), s = (∥X - V∥ / 2) → 
  (hexagon_area_with_side_length s) = (17 * Real.sqrt 3 / 2) :=
by
  let X := (0 : ℝ, 0 : ℝ)
  let V := (8 : ℝ, 2 : ℝ)
  let s := (Real.sqrt ((8-0)^2 + (2-0)^2) / 2 : ℝ)
  sorry

def hexagon_area_with_side_length (s : ℝ) : ℝ :=
  3 * Real.sqrt 3 * s^2 / 2

end hexagon_area_l113_113689


namespace find_functions_l113_113852

theorem find_functions (f g : ℚ → ℚ) (h : ∀ x y : ℚ, f(x + g(y)) = g(x) + 2 * y + f(y)) :
  (∃ a : ℚ, (∀ x : ℚ, f x = 2 * x + a) ∧ (∀ x : ℚ, g x = 2 * x)) ∨
  (∃ a : ℚ, (∀ x : ℚ, f x = a - x) ∧ (∀ x : ℚ, g x = - x)) :=
sorry

end find_functions_l113_113852


namespace zinc_in_combined_mass_l113_113641

def mixture1_copper_zinc_ratio : ℕ × ℕ := (13, 7)
def mixture2_copper_zinc_ratio : ℕ × ℕ := (5, 3)
def mixture1_mass : ℝ := 100
def mixture2_mass : ℝ := 50

theorem zinc_in_combined_mass :
  let zinc1 := (mixture1_copper_zinc_ratio.2 : ℝ) / (mixture1_copper_zinc_ratio.1 + mixture1_copper_zinc_ratio.2) * mixture1_mass
  let zinc2 := (mixture2_copper_zinc_ratio.2 : ℝ) / (mixture2_copper_zinc_ratio.1 + mixture2_copper_zinc_ratio.2) * mixture2_mass
  zinc1 + zinc2 = 53.75 :=
by
  sorry

end zinc_in_combined_mass_l113_113641


namespace min_value_a_l113_113410

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * Real.cos (2 * x) + Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem min_value_a : ∃ a, (∀ x, |g x| ≤ a) ∧ a = 3 :=
by
  use 3
  split
  · intro x
    have h : 1 ≤ g x ∧ g x ≤ 3 := by
      -- Here, you would include the detailed proof steps
      sorry
    exact abs_le_of_le_of_neg_le h.2 (neg_le_of_neg_le h.1)
  · rfl

end min_value_a_l113_113410


namespace count_units_digit_0_l113_113222

theorem count_units_digit_0 : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ n % 10 = 0}.card = 10 := sorry

end count_units_digit_0_l113_113222


namespace kadin_total_volume_l113_113346

def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem kadin_total_volume (V₁ V₂ V₃ Vₜotal : ℝ) (h₁ : V₁ = volume_of_sphere 2) 
                          (h₂ : V₂ = volume_of_sphere 3) (h₃ : V₃ = volume_of_sphere 5) 
                          (h_total : Vₜotal = V₁ + V₂ + V₃) : Vₜotal = (640 / 3) * Real.pi :=
by
  rw [h₁, h₂, h₃, volume_of_sphere, volume_of_sphere, volume_of_sphere]
  simp only [Real.pi]
  sorry

end kadin_total_volume_l113_113346


namespace number_of_exercise_books_l113_113427

theorem number_of_exercise_books (pencils pens exercise_books : ℕ) (h_ratio : (14 * pens = 4 * pencils) ∧ (14 * exercise_books = 3 * pencils)) (h_pencils : pencils = 140) : exercise_books = 30 :=
by
  sorry

end number_of_exercise_books_l113_113427


namespace find_k_l113_113592

section VectorProof

variables {k : ℝ}

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (2, -1)

def vec1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def vec2 : ℝ × ℝ := (vec_a.1 - 2 * vec_b.1, vec_a.2 - 2 * vec_b.2)

theorem find_k (h : vec1 k.1 * vec2.1 + vec1 k.2 * vec2.2 = 0) : k = 2 :=
by
  {sorry}

end VectorProof

end find_k_l113_113592


namespace num_integers_between_200_and_250_with_increasing_digits_l113_113300

theorem num_integers_between_200_and_250_with_increasing_digits : 
  card {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ (n.digits).length = 3 
    ∧ (∀ i j, i < j → (n.digits.nth i < n.digits.nth j))} = 11 := 
sorry

end num_integers_between_200_and_250_with_increasing_digits_l113_113300


namespace wang_ming_bmi_zhang_yu_bmi_category_l113_113365

-- Part 1: BMI calculation
theorem wang_ming_bmi (w h : ℝ) (h_pos : h > 0) : 
  (p : ℝ) = w / (h^2) := 
sorry

-- Part 2: Zhang Yu's BMI and weight category
theorem zhang_yu_bmi_category :
  let h : ℝ := 1.80,
      w : ℝ := 81, 
      p := w / (h^2) in
  p = 25 ∧ p > 24 :=
by 
  have p_eq : p = w / (h^2) := by sorry
  have p_val : p = 25 := by sorry
  have p_cat : p > 24 := by sorry
  exact ⟨p_val, p_cat⟩

end wang_ming_bmi_zhang_yu_bmi_category_l113_113365


namespace infinite_sqrt_converges_to_3_l113_113632

noncomputable def infinite_sqrt : ℝ := sqrt (3 + 2 * infinite_sqrt)

theorem infinite_sqrt_converges_to_3 : infinite_sqrt > 0 → infinite_sqrt = 3 :=
by
  assume h : infinite_sqrt > 0
  have eqn : 3 + 2 * infinite_sqrt = infinite_sqrt ^ 2 :=
    calc
      3 + 2 * infinite_sqrt = infinite_sqrt ^ 2 : 
      sorry -- Detailed proof omitted here
  have sol := by sorry -- Solve the equation 3 + 2 * m = m^2 to show m = 3
  sorry -- Full proof omitted, include proper logical steps to avoid this

end infinite_sqrt_converges_to_3_l113_113632


namespace problem1_l113_113124

theorem problem1
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : (3*x - 2)^(6) = a₀ + a₁ * (2*x - 1) + a₂ * (2*x - 1)^2 + a₃ * (2*x - 1)^3 + a₄ * (2*x - 1)^4 + a₅ * (2*x - 1)^5 + a₆ * (2*x - 1)^6)
  (h₂ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1)
  (h₃ : a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ = 64) :
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63 / 65 := by
  sorry

end problem1_l113_113124


namespace volunteer_and_elderly_arrangements_l113_113133

def num_volunteers : ℕ := 5
def num_elderly : ℕ := 2

theorem volunteer_and_elderly_arrangements :
  (factorial (num_volunteers + num_elderly - 1)) * (factorial num_elderly) = 1440 :=
by
  sorry

end volunteer_and_elderly_arrangements_l113_113133


namespace distance_between_points_l113_113973

noncomputable def complex_distance (z1 z2 : Complex) : Real :=
  Complex.abs (z2 - z1)

theorem distance_between_points :
  complex_distance (-3 + Complex.i) (1 - Complex.i) = Real.sqrt 20 := by
sorry

end distance_between_points_l113_113973


namespace light_exit_angle_approx_l113_113771

/-
Given:
- The refractive index of the prism is n_2 = 1.5
- The cross-section of the prism is an equilateral triangle
- A ray of light enters the prism horizontally from the air (n_1 = 1)
- The light exits the prism at an angle θ

Prove:
- θ ≈ 13 degrees
-/

theorem light_exit_angle_approx (n₁ n₂ : ℝ) (θ₁ θ₂ r θ : ℝ)
  (h₁ : n₁ = 1) 
  (h₂ : n₂ = 1.5) 
  (h₃ : θ₁ = 30)
  (h₄ : sin θ₁ * n₁ = sin r * n₂) 
  (h₅ : r = arcsin (1 / 3)) 
  (h₆ : θ₂ = 60 - r)
  (h₇ : sin θ * n₁ = sin θ₂ * n₂)
  : θ ≈ 13 :=
sorry

end light_exit_angle_approx_l113_113771


namespace pow_sum_inv_int_l113_113004

theorem pow_sum_inv_int {x : ℝ} (h1 : x ≠ 0) (h2 : (x + x⁻¹) ∈ ℤ) : ∀ n : ℕ, (x^n + x^(-n)) ∈ ℤ :=
by
  sorry

end pow_sum_inv_int_l113_113004


namespace sum_of_seven_smallest_multiples_of_twelve_l113_113764

theorem sum_of_seven_smallest_multiples_of_twelve : 
  let multiples := [12, 24, 36, 48, 60, 72, 84] in
  ∑ i in multiples, i = 336 := 
by sorry

end sum_of_seven_smallest_multiples_of_twelve_l113_113764


namespace non_organic_chicken_price_l113_113468

theorem non_organic_chicken_price :
  ∀ (x : ℝ), (0.75 * x = 9) → (2 * (0.9 * x) = 21.6) :=
by
  intro x hx
  sorry

end non_organic_chicken_price_l113_113468


namespace solution_l113_113383

section
variables (x y : ℕ)
variables (price_red_black_pen price_black_refill price_red_refill discounted_price_black discounted_price_red total_purchase total_spent original_price total_savings : ℝ)

-- Given conditions
def original_prices  :=
  price_red_black_pen = 10 ∧ 
  price_black_refill = 6 ∧ 
  price_red_refill = 8

def discount_prices :=
  discounted_price_black = price_black_refill * 0.5 ∧ 
  discounted_price_red = price_red_refill * 0.75 

def purchased_items :=
  total_purchase = x + y ∧ 
  total_purchase = 10 ∧ 
  total_spent = discounted_price_black * x + discounted_price_red * y + 2 * price_red_black_pen ∧ 
  total_spent = 74

-- Proof problem 1
def number_of_refills :=
  3 * x + 6 * y = 54 ∧ 
  x + y = 10

-- Proof problem 2
def saved_amount :=
  original_price = 2 * price_red_black_pen + 2 * price_black_refill + 8 * price_red_refill ∧ 
  original_price = 96 ∧ 
  total_savings = original_price - total_spent ∧ 
  total_savings = 22

theorem solution : x = 2 ∧ y = 8 ∧ total_savings = 22 :=
by
  unfold original_prices discount_prices purchased_items number_of_refills saved_amount
  sorry
end

end solution_l113_113383


namespace inequality_for_positive_reals_l113_113686

open Real

theorem inequality_for_positive_reals 
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) :=
sorry

end inequality_for_positive_reals_l113_113686


namespace factor_expression_l113_113493

theorem factor_expression (b : ℤ) : 
  (8 * b ^ 3 + 120 * b ^ 2 - 14) - (9 * b ^ 3 - 2 * b ^ 2 + 14) 
  = -1 * (b ^ 3 - 122 * b ^ 2 + 28) := 
by {
  sorry
}

end factor_expression_l113_113493


namespace largest_dimension_of_crate_l113_113791

theorem largest_dimension_of_crate:
  ∀ (l w : ℝ), l = 8 ∧ w = 3 ∧ (∃ r : ℝ, r = 3) →
  sqrt (l^2 + w^2) = sqrt 73 :=
by
  intros l w h
  cases h with hl hw_r
  cases hw_r with hw hr
  rw [hl, hw]
  simp
  sorry

end largest_dimension_of_crate_l113_113791


namespace probability_equal_AB_AC_l113_113620

-- Define the structure of the problem
structure School :=
  (students_per_class : Nat)
  (num_classes : Nat)
  (boys_per_class : Nat)

-- Example problem setup
def liberal_arts_school : School :=
  { students_per_class := 40, num_classes := 4, boys_per_class := 8 }

variables (school : School)
def total_students : Nat := school.students_per_class * school.num_classes
def num_girls_per_class : Nat := school.students_per_class - school.boys_per_class
def total_boys : Nat := school.num_classes * school.boys_per_class
def total_girls : Nat := school.num_classes * num_girls_per_class school
def sample_size : Nat := 20

-- Defining students A, B, C belonging to respective classes
variables (A B C : Fin total_students)
variables (class_A class_B : Nat)
variables (from_same_class : class_A = class_B)
variables (from_diff_class : class_A ≠ class_B)

-- The proof problem statement
theorem probability_equal_AB_AC :
  (prob_selected A ∧ prob_selected B) = (prob_selected A ∧ prob_selected C) :=
by
  sorry

end probability_equal_AB_AC_l113_113620


namespace probability_same_row_or_column_l113_113636

theorem probability_same_row_or_column :
  let A := fin 3 × fin 3 in 
  let total_selections := (A.fintype.card.choose 3) in
  let valid_selections := 3 * 2 in -- since 3 methods to select from the first row, 2 from the second, 1 from third
  (total_selections - valid_selections) / total_selections.to_num = 13 / 14 :=
by sorry

end probability_same_row_or_column_l113_113636


namespace sum_seven_smallest_multiples_of_12_l113_113758

theorem sum_seven_smallest_multiples_of_12 :
  (Finset.sum (Finset.range 7) (λ n, 12 * (n + 1))) = 336 :=
by
  -- proof (sorry to skip)
  sorry

end sum_seven_smallest_multiples_of_12_l113_113758


namespace exterior_angle_of_octagon_is_45_degrees_l113_113964

noncomputable def exterior_angle_of_regular_octagon : ℝ :=
  let n : ℝ := 8
  let interior_angle_sum := 180 * (n - 2) -- This is the sum of interior angles of any n-gon
  let each_interior_angle := interior_angle_sum / n -- Each interior angle in a regular polygon
  let each_exterior_angle := 180 - each_interior_angle -- Exterior angle is supplement of interior angle
  each_exterior_angle

theorem exterior_angle_of_octagon_is_45_degrees :
  exterior_angle_of_regular_octagon = 45 := by
  sorry

end exterior_angle_of_octagon_is_45_degrees_l113_113964


namespace find_m_n_sum_l113_113001

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end find_m_n_sum_l113_113001


namespace same_function_C_l113_113477

def f_C (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

def g_C (t : ℝ) : ℝ := 
  |t|

theorem same_function_C : 
  f_C = g_C :=
sorry

end same_function_C_l113_113477


namespace jason_additional_manager_months_l113_113646

def additional_manager_months (bartender_years manager_years total_exp_months : ℕ) : ℕ :=
  let bartender_months := bartender_years * 12
  let manager_months := manager_years * 12
  total_exp_months - (bartender_months + manager_months)

theorem jason_additional_manager_months : 
  additional_manager_months 9 3 150 = 6 := 
by 
  sorry

end jason_additional_manager_months_l113_113646


namespace line_intersects_at_least_one_of_skew_lines_l113_113288

-- Definitions
variables {R : Type*} [LinearOrderedField R]
variables (a b l : Line R)
variables (α β : Plane R)

-- Conditions
def skew_lines : Prop :=
  ∀ (p : Point R), ¬(p ∈ a ∧ p ∈ b)

def contains_lines : Prop :=
  (∀ p, p ∈ a → p ∈ α) ∧ (∀ p, p ∈ b → p ∈ β)

def intersect_planes : Prop :=
  ∀ (p : Point R), p ∈ α ∧ p ∈ β ↔ p ∈ l

-- Theorem
theorem line_intersects_at_least_one_of_skew_lines
  (h1 : skew_lines a b)
  (h2 : contains_lines a b α β)
  (h3 : intersect_planes α β l) :
  (∃ p, p ∈ l ∧ p ∈ a) ∨ (∃ p, p ∈ l ∧ p ∈ b) :=
sorry

end line_intersects_at_least_one_of_skew_lines_l113_113288


namespace snack_eaters_initial_count_l113_113795

-- Define all variables and conditions used in the problem
variables (S : ℕ) (initial_people : ℕ) (new_outsiders_1 : ℕ) (new_outsiders_2 : ℕ) (left_after_first_half : ℕ) (left_after_second_half : ℕ) (remaining_snack_eaters : ℕ)

-- Assign the specific values according to conditions
def conditions := 
  initial_people = 200 ∧
  new_outsiders_1 = 20 ∧
  new_outsiders_2 = 10 ∧
  left_after_first_half = (S + new_outsiders_1) / 2 ∧
  left_after_second_half = left_after_first_half + new_outsiders_2 - 30 ∧
  remaining_snack_eaters = left_after_second_half / 2 ∧
  remaining_snack_eaters = 20

-- State the theorem to prove
theorem snack_eaters_initial_count (S : ℕ) (initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters : ℕ) :
  conditions S initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters → S = 100 :=
by sorry

end snack_eaters_initial_count_l113_113795


namespace total_prime_factors_l113_113775

theorem total_prime_factors : 
  let expr := (4 ^ 11) * (7 ^ 3) * (11 ^ 2) in
  (prime_factors expr).length = 27 := 
by {
  let expr := (4 ^ 11) * (7 ^ 3) * (11 ^ 2),
  sorry
}

end total_prime_factors_l113_113775


namespace equivalent_sets_l113_113441

def P : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

def Q : Set ℝ := { x | ∃ y, y = log 2 (x^2 - 2*x - 15) }

def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ x ∉ (P ∩ Q) }

theorem equivalent_sets :
  P_plus_Q = { x | x < -3 ∨ (-1 ≤ x ∧ x ≤ 4) ∨ x > 5 } := by
  sorry

end equivalent_sets_l113_113441


namespace problem_statement_l113_113575

variable {α : Type*} [field α] [decidable_eq α] 

def f (a b x : α) := a * x^2 + (b - 8) * x - a - a * b

theorem problem_statement : 
  (∀ x, f (-3) 5 x > 0 → x ∈ Ioo (-3 : α) 2) ∧
  (∀ x, f (-3) 5 x < 0 → x ∈ Iio (-3) ∨ x ∈ Ioi 2) ∧
  (∀ a b, has_roots (a * x^2 + (b - 8) * x - a - ab) (-3) (2)) →
  (f (-3) 5 = λ x, -3 * x^2 + 3 * x + 18) ∧
  ( ∀ c, (∀ x, -3 * x^2 + 5 * x + c ≤ 0) → c ≤ - 25 / 12 ) ∧
  ( ∀ x, x > -1 → max (λ y, y = -3) (λ x, (f (-3) 5 x - 21) / (x + 1)) = - 3 )
  :=
sorry

end problem_statement_l113_113575


namespace eval_expression_l113_113206

theorem eval_expression (x y z : ℝ) 
  (h1 : z = y - 11) 
  (h2 : y = x + 3) 
  (h3 : x = 5)
  (h4 : x + 2 ≠ 0) 
  (h5 : y - 3 ≠ 0) 
  (h6 : z + 7 ≠ 0) : 
  ( (x + 3) / (x + 2) * (y - 1) / (y - 3) * (z + 9) / (z + 7) ) = 2.4 := 
by
  sorry

end eval_expression_l113_113206


namespace compare_rental_fees_l113_113071

namespace HanfuRental

def store_A_rent_price : ℝ := 120
def store_B_rent_price : ℝ := 160
def store_A_discount : ℝ := 0.20
def store_B_discount_limit : ℕ := 6
def store_B_excess_rate : ℝ := 0.50
def x : ℕ := 40 -- number of Hanfu costumes

def y₁ (x : ℕ) : ℝ := (store_A_rent_price * (1 - store_A_discount)) * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ store_B_discount_limit then store_B_rent_price * x
  else store_B_rent_price * store_B_discount_limit + store_B_excess_rate * store_B_rent_price * (x - store_B_discount_limit)

theorem compare_rental_fees (x : ℕ) (hx : x = 40) :
  y₂ x ≤ y₁ x :=
sorry

end HanfuRental

end compare_rental_fees_l113_113071


namespace expression_value_l113_113545

theorem expression_value
  (x y a b : ℤ)
  (h1 : x = 1)
  (h2 : y = 2)
  (h3 : a + 2 * b = 3) :
  2 * a + 4 * b - 5 = 1 := 
by sorry

end expression_value_l113_113545


namespace find_divisor_l113_113801

theorem find_divisor (x y : ℝ) (hx : x > 0) (hx_val : x = 1.3333333333333333) (h : 4 * x / y = x^2) : y = 3 :=
by 
  sorry

end find_divisor_l113_113801


namespace simplify_and_evaluate_l113_113379

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  ( ( (2 * x + 1) / x - 1 ) / ( (x^2 - 1) / x ) ) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l113_113379


namespace sum_of_seven_smallest_multiples_of_12_l113_113762

theorem sum_of_seven_smallest_multiples_of_12 : 
  ∑ i in (Finset.range 7).map (λ i, 12 * (i + 1)) = 336 :=
by
  sorry

end sum_of_seven_smallest_multiples_of_12_l113_113762


namespace right_triangle_x_sum_l113_113803

noncomputable def right_triangle_x (x : ℝ) : Prop :=
  let side1 := x
  let side2 := x + 1
  let hypotenuse := 2 * x - 1
  hypotenuse^2 = side1^2 + side2^2

theorem right_triangle_x_sum (x : ℝ) (pos_lengths : x > 0) :
  right_triangle_x x → (x = 3) → (3 : ℝ) = (a + b + c) :=
by
  assume (h1 : right_triangle_x x)
  assume (h2 : x = 3)
  have a_b_c_sum : 3 = (a + b + c)
    by
      sorry
  exact a_b_c_sum

end right_triangle_x_sum_l113_113803


namespace cannot_tile_remaining_with_dominoes_l113_113368

def can_tile_remaining_board (pieces : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j

theorem cannot_tile_remaining_with_dominoes : 
  ∃ (pieces : List (ℕ × ℕ)), (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) ∧ (1 ≤ j ∧ j ≤ 10) → ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j) ∧ ¬ can_tile_remaining_board pieces :=
sorry

end cannot_tile_remaining_with_dominoes_l113_113368


namespace inequality_range_of_a_l113_113583

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| - |x + a| ≥ a) → (a ≤ - 1 / 4) :=
begin
  sorry
end

end inequality_range_of_a_l113_113583


namespace overtaking_time_correct_l113_113470

noncomputable def time_to_overtake (speed_a speed_b : ℝ) (time_a : ℝ) (distance_a distance_b : ℝ) : ℝ :=
  distance_a / (speed_b - speed_a)

theorem overtaking_time_correct :
  ∀ (speed_a speed_b : ℝ) (time_a : ℝ),
  speed_a = 5 ∧ time_a = 0.5 ∧ speed_b = 5.555555555555555 →
  time_to_overtake speed_a speed_b 0.5 (speed_a * 0.5) (speed_b * ?) = 4.5 :=
begin
  intros,
  sorry,
end

end overtaking_time_correct_l113_113470


namespace garden_area_not_covered_by_flower_beds_l113_113806

-- Define the parameters and the objects:
def side_length_garden : ℝ := 16
def radius_flower_bed : ℝ := 8

-- Define the areas computed:
def area_square : ℝ := side_length_garden ^ 2
def area_circle : ℝ := Real.pi * (radius_flower_bed ^ 2)
def area_flower_beds : ℝ := 4 * (area_circle / 4)
def area_shaded : ℝ := area_square - area_flower_beds

-- The proof statement:
theorem garden_area_not_covered_by_flower_beds :
  area_shaded = 256 - 64 * Real.pi := 
sorry

end garden_area_not_covered_by_flower_beds_l113_113806


namespace cone_volume_ratio_l113_113497

theorem cone_volume_ratio : 
  ∀ (rC hC rD hD : ℝ), 
  rC = 10 → hC = 20 → rD = 20 → hD = 10 → 
  (volume_C : ℝ) = (1 / 3) * π * (rC * rC) * hC →
  (volume_D : ℝ) = (1 / 3) * π * (rD * rD) * hD →
  volume_C / volume_D = 1 / 2 :=
by
  intros rC hC rD hD hC_eq rC_eq hD_eq rD_eq vol_C vol_D
  rw [hC_eq, rC_eq, hD_eq, rD_eq] at vol_C vol_D
  sorry

end cone_volume_ratio_l113_113497


namespace probability_equal_men_women_l113_113431

noncomputable def combination (n k : ℕ) : ℚ :=
  (n! / (k! * (n - k)!))

theorem probability_equal_men_women :
  let total_students := 8 in
  let men := 4 in
  let women := 4 in
  let selected_students := 4 in
  let favorable_ways := (combination men 2) * (combination women 2) in
  let total_ways := combination total_students selected_students in
  (favorable_ways / total_ways) = (18 / 35) :=
by
  sorry

end probability_equal_men_women_l113_113431


namespace ramanujan_number_l113_113022

theorem ramanujan_number (h r : ℂ) (cond1 : h = 7 + 4i) (cond2 : r * h = 60 - 18i) :
  r = 174/65 - (183/65) * I := by
sorry

end ramanujan_number_l113_113022


namespace students_per_class_l113_113341

theorem students_per_class (total_cupcakes : ℕ) (num_classes : ℕ) (pe_students : ℕ) 
  (h1 : total_cupcakes = 140) (h2 : num_classes = 3) (h3 : pe_students = 50) : 
  (total_cupcakes - pe_students) / num_classes = 30 :=
by
  sorry

end students_per_class_l113_113341


namespace fraction_used_for_peanut_butter_cookies_l113_113360

theorem fraction_used_for_peanut_butter_cookies :
  ∀ (x : ℝ), 
    (10 - (10 / 2) - (10 - (10 / 2)) * x - (1 / 3) * ((10 - (10 / 2)) * (1 - x)) = 2)
    → x = 2 / 5 :=
by
  intro x
  assume h
  sorry

end fraction_used_for_peanut_butter_cookies_l113_113360


namespace polar_coordinates_of_point_l113_113501

noncomputable def polarCoordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y >= 0 then Real.atan (y / x)
           else if x > 0 ∧ y < 0 then 2 * Real.pi - Real.atan (|y / x|)
           else if x < 0 then Real.pi + Real.atan (y / x)
           else if y > 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ)

theorem polar_coordinates_of_point :
  polarCoordinates 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_point_l113_113501


namespace sin_double_angle_l113_113893

theorem sin_double_angle (x : ℝ) (h : sin x + cos x = 3 * real.sqrt 2 / 5) : sin (2 * x) = -7 / 25 := by
  sorry

end sin_double_angle_l113_113893


namespace domain_of_function_l113_113576

theorem domain_of_function :
  ∀ x, (x - 2 > 0) ∧ (3 - x ≥ 0) ↔ 2 < x ∧ x ≤ 3 :=
by 
  intros x 
  simp only [and_imp, gt_iff_lt, sub_lt_iff_lt_add, sub_nonneg, le_iff_eq_or_lt, add_comm]
  exact sorry

end domain_of_function_l113_113576


namespace complex_pure_imaginary_l113_113905

theorem complex_pure_imaginary (a : ℝ) (h : (((1 : ℂ) + (a : ℂ) * complex.I) / ((1 : ℂ) - complex.I)).re = 0) : a = 1 :=
by sorry

end complex_pure_imaginary_l113_113905


namespace sum_of_possible_s_values_l113_113526

noncomputable def isosceles_area_condition (s : ℝ) : Prop :=
  let A := (Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180))
  let B := (Real.cos (45 * Real.pi / 180), Real.sin (45 * Real.pi / 180))
  let C := (Real.cos (s * Real.pi / 180), Real.sin (s * Real.pi / 180))
  (dist A B = dist A C ∨ dist A B = dist B C ∨ dist A C = dist B C) ∧
  (abs ((fst B - fst A) * (snd C - snd A) - (snd B - snd A) * (fst C - fst A)) / 2 > 0.1)

theorem sum_of_possible_s_values : (∑ s in {s : ℝ | 0 ≤ s ∧ s ≤ 360 ∧ isosceles_area_condition s}, s) = 60 :=
by
  sorry

end sum_of_possible_s_values_l113_113526


namespace relationship_among_abc_l113_113237

noncomputable def a : ℝ := (1 / 3 : ℝ) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3 : ℝ) ^ (2 / 5)
noncomputable def c : ℝ := Real.logBase (1 / 3) (1 / 5)

theorem relationship_among_abc : c > b ∧ b > a := by
  sorry

end relationship_among_abc_l113_113237


namespace min_ab_min_expr_min_a_b_l113_113255

-- Define the conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hln : Real.log a + Real.log b = Real.log (a + 9 * b))

-- 1. The minimum value of ab
theorem min_ab : ab = 36 :=
sorry

-- 2. The minimum value of (81 / a^2) + (1 / b^2)
theorem min_expr : (81 / a^2) + (1 / b^2) = (1 / 2) :=
sorry

-- 3. The minimum value of a + b
theorem min_a_b : a + b = 16 :=
sorry

end min_ab_min_expr_min_a_b_l113_113255


namespace replace_asterisks_l113_113417

theorem replace_asterisks (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end replace_asterisks_l113_113417


namespace max_tied_teams_for_most_wins_l113_113965

   -- Definition of a round-robin tournament with 8 teams
   def round_robin_tournament (teams : Finset ℕ) : Prop :=
     teams.card = 8 ∧
     ∀ t1 t2 ∈ teams, t1 ≠ t2 → (∃ winner, winner = t1 ∨ winner = t2)

   -- Definition to calculate the number of games in the tournament
   def number_of_games (n : ℕ) : ℕ :=
     (n * (n - 1)) / 2

   theorem max_tied_teams_for_most_wins (teams : Finset ℕ)
     (h_tournament : round_robin_tournament teams) :
     ∃ max_tied_teams, max_tied_teams = 7 :=
   by
     -- Omitting the step where we really calculate that the maximum number of tied teams is 7
     sorry
   
end max_tied_teams_for_most_wins_l113_113965


namespace tangent_slope_l113_113756

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

theorem tangent_slope
  (center : ℝ × ℝ) (tangent_point : ℝ × ℝ)
  (h1 : center = (2, 3)) (h2 : tangent_point = (7, 8)) :
  slope tangent_point center ≠ 0 →
  perpendicular_slope (slope tangent_point center) = -1 :=
by
  sorry

end tangent_slope_l113_113756


namespace smallest_positive_period_l113_113509

-- Definitions
def is_period (T : ℝ) (f : ℝ → ℝ) := ∀ x, f(x + T) = f x

-- Given function and known period formula
def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem smallest_positive_period : ∃ T > 0, is_period T f ∧ ∀ T' > 0, is_period T' f → T ≤ T' :=
by
  use Real.pi
  sorry

end smallest_positive_period_l113_113509


namespace minimum_moves_to_construct_equilateral_triangle_l113_113821

-- Define the initial setup of the problem
def initial_setup (O P : Point) (C : Circle) : Prop := 
  C.center = O ∧ P ∈ C

-- Define the types of moves available to Anita
inductive Move : Type
| draw_line (A B : Point) : Line
| draw_circle (center : Point) (radius : ℝ) : Circle

-- Define a known point as a property
def known_point (pt : Point) (known_points : set Point) : Prop :=
  pt ∈ known_points

-- Define the construct_equilateral_triangle property
def construct_equilateral_triangle (O P : Point) (C : Circle) (n : ℕ) : Prop :=
  ∀ moves : list Move,
    initial_setup O P C →
    length moves = n →
    (∀ move ∈ moves,
      ∃ intersection_points : set Point,
      known_point O intersection_points ∧
      known_point P intersection_points ∧
      -- Define all intersections properties forming an equilateral triangle
      ∃ A B C : Point, A ∈ C ∧ B ∈ C ∧ 
      distance A B = distance B C ∧ distance B C = distance C A) →
    n = 5

-- Lean statement for the proof problem
theorem minimum_moves_to_construct_equilateral_triangle (O P : Point) (C : Circle) :
  initial_setup O P C → construct_equilateral_triangle O P C 5 :=
by
  sorry

end minimum_moves_to_construct_equilateral_triangle_l113_113821


namespace derivative_of_f_at_i_l113_113605

noncomputable def f (x : ℂ) : ℂ := x^4 - x^2

theorem derivative_of_f_at_i : (deriv f) complex.I = -6 * complex.I :=
by
  -- Proof goes here
  sorry

end derivative_of_f_at_i_l113_113605


namespace sin_cos_sum_l113_113602

theorem sin_cos_sum (θ a b : ℝ) (h1 : 0 < θ ∧ θ < real.pi / 2)
  (h2 : real.sin (2 * θ) = a) (h3 : real.cos (2 * θ) = b) :
  real.sin θ + real.cos θ = real.sqrt (1 + a) :=
sorry

end sin_cos_sum_l113_113602


namespace ratio_of_inscribed_sphere_radius_to_height_l113_113883

theorem ratio_of_inscribed_sphere_radius_to_height 
    (H : ℝ) 
    (R : ℝ) 
    (tetrahedron_regular : true) 
    (height_of_tetrahedron : ℝ := H) 
    (radius_of_inscribed_sphere : ℝ := R) 
    (volume_tetrahedron : ℝ := (4 * (1 / 3) * (1 / 2) * (sqrt 3 / 4) * (H / sqrt 3) ^ 2 * R)) :
  (R / H = 1 / 4) := 
by
  sorry

end ratio_of_inscribed_sphere_radius_to_height_l113_113883


namespace factor_square_difference_l113_113212

theorem factor_square_difference (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := 
  sorry

end factor_square_difference_l113_113212


namespace cost_of_goods_l113_113736

-- Define variables and conditions
variables (x y z : ℝ)

-- Assume the given conditions
axiom h1 : x + 2 * y + 3 * z = 136
axiom h2 : 3 * x + 2 * y + z = 240

-- Statement to prove
theorem cost_of_goods : x + y + z = 94 := 
sorry

end cost_of_goods_l113_113736


namespace count_multiples_of_12_l113_113934

theorem count_multiples_of_12 (a b : ℤ) (h1 : 15 < a) (h2 : b < 205) (h3 : ∃ k : ℤ, a = 12 * k) (h4 : ∃ k : ℤ, b = 12 * k) : 
  ∃ n : ℕ, n = 16 := 
by 
  sorry

end count_multiples_of_12_l113_113934


namespace xy_difference_squared_l113_113310

theorem xy_difference_squared (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  -- the proof goes here
  sorry

end xy_difference_squared_l113_113310


namespace animal_population_survival_l113_113959

theorem animal_population_survival (m : ℕ) :
  (∀ (p : ℝ), (p = 1 - 1/10) →
   (∀ (n : ℝ), (n = 200) →
    (∀ (s : ℝ), (s = 145.8) →
     (n * p^m = s) → 
     m ≈ 3))) := 
sorry

end animal_population_survival_l113_113959


namespace repeating_decimal_arithmetic_l113_113176

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l113_113176


namespace count_pairs_satisfying_sum_property_l113_113842

theorem count_pairs_satisfying_sum_property :
  ∃ n, n = 6 ∧ (∃ (pairs : List (ℕ × ℕ)),
    (∀ p ∈ pairs, let (a, b) := p in a + b = 13 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
    pairs.length = n) :=
by
  sorry

end count_pairs_satisfying_sum_property_l113_113842


namespace time_pipe_A_alone_fills_tank_l113_113149

noncomputable def time_for_pipe_A_to_fill_tank (A B C: ℝ) : ℝ :=
  if h : B = 2 * A ∧ C = 2 * B ∧ A + B + C = 1 / 3 then
    21
  else
    0

theorem time_pipe_A_alone_fills_tank :
  ∀ (A B C: ℝ), B = 2 * A → C = 2 * B → A + B + C = 1 / 3 → time_for_pipe_A_to_fill_tank A B C = 21 :=
begin
  intros A B C hB hC hSum,
  simp [time_for_pipe_A_to_fill_tank, hB, hC, hSum],
  sorry
end

end time_pipe_A_alone_fills_tank_l113_113149


namespace union_of_A_and_B_l113_113890

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := 
by 
  sorry

end union_of_A_and_B_l113_113890


namespace probability_at_least_two_out_of_four_l113_113864

/-
We define the probability of one seed germinating and the number of seeds.
Then we establish that the probability of at least 2 out of 4 seeds germinating equals 608/625.
-/

def prob_germinating : ℚ := 4 / 5
def num_seeds : ℕ := 4

theorem probability_at_least_two_out_of_four : 
  ∃ (p : ℚ), p = 608 / 625 ∧ (p = (nat.choose 4 2 * (prob_germinating ^ 2) * ((1 - prob_germinating) ^ 2)) +
                                (nat.choose 4 3 * (prob_germinating ^ 3) * ((1 - prob_germinating) ^ 1)) + 
                                (prob_germinating ^ 4)) :=
by {
  sorry
}

end probability_at_least_two_out_of_four_l113_113864


namespace relationship_among_a_b_c_l113_113270

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3
def a : ℝ := f (real.sqrt (1 / 2))
def b : ℝ := f (2 ^ 0.2)
def c : ℝ := f (real.log2 (1 / 2))

-- Statement to prove the relationship among a, b, and c
theorem relationship_among_a_b_c : b > a ∧ a > c := by
  -- Proof not given
  sorry

end relationship_among_a_b_c_l113_113270


namespace between_200_and_250_has_11_integers_with_increasing_digits_l113_113295

/-- How many integers between 200 and 250 have three different digits in increasing order --/
theorem between_200_and_250_has_11_integers_with_increasing_digits :
  ∃ n : ℕ, n = 11 ∧ ∀ (x : ℕ),
    200 ≤ x ∧ x ≤ 250 →
    (∀ i j k : ℕ, (x = 100 * i + 10 * j + k) →
    i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) ↔ x ∈ {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249} :=
by {
  let s := {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249},
  existsi 11,
  simp [s],
  intro x,
  split,
  intros h _,
  have hx : x ∈ s, sorry, -- proof needed
  exact hx,
  intros h _,
  split,
  intros i j k hx _,
  have hi : i = 2, sorry, -- proof needed
  cases h with j_values _,
  case or.inl : { rw j_values, exact sorry }, -- proof needed
  case or.inr : { rw j_values, exact sorry }, -- proof needed
  exact sorry -- scaffolding, proof steps go here
}

end between_200_and_250_has_11_integers_with_increasing_digits_l113_113295


namespace find_polynomials_satisfy_piecewise_l113_113855

def f (x : ℝ) : ℝ := 0
def g (x : ℝ) : ℝ := -x
def h (x : ℝ) : ℝ := -x + 2

theorem find_polynomials_satisfy_piecewise :
  ∀ x : ℝ, abs (f x) - abs (g x) + h x = 
    if x < -1 then -1
    else if x <= 0 then 2
    else -2 * x + 2 :=
by
  sorry

end find_polynomials_satisfy_piecewise_l113_113855


namespace sum_tan_sq_l113_113529

-- Define the polynomial P(x) = 16x^3 - 21x
def P (x : ℝ) : ℝ := 16 * x^3 - 21 * x

-- State the main theorem to prove the sum of all possible values of tan^2(θ)
theorem sum_tan_sq (θ : ℝ) (h : P (real.sin θ) = P (real.cos θ)) : Σ x, x = (real.tan θ)^2 := 
sorry

end sum_tan_sq_l113_113529


namespace exchange_rate_change_2014_l113_113485

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_2014_l113_113485


namespace degree_f_plus_g_l113_113698

-- Let f and g be polynomials where the degree of f is 3 and the degree of g is 2.
-- The leading coefficients of f and g at their respective highest degrees do not zero out when summed.

variable {R : Type} [Ring R]
variables (f g : R[X])

def degree_f_eq_3 (f : R[X]) : Prop := degree f = 3
def degree_g_eq_2 (g : R[X]) : Prop := degree g = 2
def leading_coeff_nonzero_sum (f g : R[X]) : Prop := 
  leadingCoeff f + leadingCoeff (g * X)

theorem degree_f_plus_g (f g : R[X])
  (hf : degree f = 3)
  (hg : degree g = 2)
  (hfg : leadingCoeff f + leadingCoeff (g * X) ≠ 0) : 
  degree (f + g) = 3 := 
by
  sorry

end degree_f_plus_g_l113_113698


namespace train_cross_time_l113_113151

-- Definitions based on the conditions in part a)
def train_speed_kmh : ℝ := 40
def train_length_meters : ℝ := 220.0176
def conversion_factor : ℝ := 1000 / 3600

-- Essential intermediate calculation not using solution steps but using definitions from conditions.
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor

-- Theorem statement based on the equivalent problem definition
theorem train_cross_time :
  (220.0176 / (40 * (1000 / 3600))) ≈ 19.80176 :=
by
  sorry

end train_cross_time_l113_113151


namespace intersection_points_between_line_and_hyperbola_l113_113242

structure Focus :=
  (x : ℝ)
  (y : ℝ)

structure Hyperbola :=
  (focus1 : Focus)
  (focus2 : Focus)
  (a : ℝ)

structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

def num_intersection_points (h : Hyperbola) (l : Line) : ℕ :=
  sorry

theorem intersection_points_between_line_and_hyperbola :
  ∀ (h : Hyperbola) (l : Line), num_intersection_points h l ∈ {0, 1, 2} :=
by
  sorry

end intersection_points_between_line_and_hyperbola_l113_113242


namespace arithmetic_sequence_sum_first_15_l113_113971

def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_sum_first_15
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h₂ : a 7 + a 8 + a 9 = 3) :
  arithmetic_sequence_sum a 15 = 15 :=
by
  sorry

end arithmetic_sequence_sum_first_15_l113_113971


namespace number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l113_113163

-- Define the number of classes in each grade.
def num_classes_first_year : ℕ := 14
def num_classes_second_year : ℕ := 14
def num_classes_third_year : ℕ := 15

-- Prove the number of different ways to choose students from 1 class.
theorem number_of_ways_to_choose_one_class :
  (num_classes_first_year + num_classes_second_year + num_classes_third_year) = 43 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from one class in each grade.
theorem number_of_ways_to_choose_one_class_each_grade :
  (num_classes_first_year * num_classes_second_year * num_classes_third_year) = 2940 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from 2 classes from different grades.
theorem number_of_ways_to_choose_two_classes_different_grades :
  (num_classes_first_year * num_classes_second_year + num_classes_first_year * num_classes_third_year + num_classes_second_year * num_classes_third_year) = 616 := 
by {
  -- Numerical calculation
  sorry
}

end number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l113_113163


namespace area_large_sphere_trace_l113_113146

-- Define the conditions
def radius_small_sphere : ℝ := 4
def radius_large_sphere : ℝ := 6
def area_small_sphere_trace : ℝ := 37

-- Define the mathematically equivalent proof problem
theorem area_large_sphere_trace :
  let r1 := radius_small_sphere,
      r2 := radius_large_sphere,
      a1 := area_small_sphere_trace,
      ratio := (r2 / r1) ^ 2 in
  a1 * ratio = 83.25 := by
sorry

end area_large_sphere_trace_l113_113146


namespace wire_not_used_l113_113204

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l113_113204


namespace polygon_smallest_angle_l113_113707

theorem polygon_smallest_angle 
  (n : ℕ) (angles : Finₓ (n+1) → ℕ) (h1 : n = 15) 
  (h2 : ∀ (i j : Finₓ (n+1)), i < j → angles i < angles j)
  (h3 : angles ⟨14, Nat.lt_succ_self 14⟩ = 176)
  (h4 : angles.sum = 2340) : 
  ∃ a, angles ⟨0, Nat.zero_lt_succ 14⟩ = a ∧ a = 136 :=
by
  sorry

end polygon_smallest_angle_l113_113707


namespace find_x_coordinate_l113_113800

-- Define the conditions for the point (x, y) to be equally distant from the y-axis, the line x = 2, and the line y = 2x + 4
def equidistant_from_lines (x y : ℝ) : Prop :=
  abs (x - 0) = abs (x - 2) ∧ abs (x - 2) = abs (2 * x + 4 - y) / real.sqrt (5)

-- Prove that the point (x, y) that satisfies the equidistant condition has x-coordinate 2
theorem find_x_coordinate (x y : ℝ) (h : equidistant_from_lines x y) : 
  x = 2 :=
  sorry

end find_x_coordinate_l113_113800


namespace equality_of_distances_l113_113822

theorem equality_of_distances
  (Γ : Circle)
  (tangent_E : Point → Tangent AB Γ)
  (tangent_D : Point → Tangent AC Γ)
  (P : Point)
  (P_midpoint : IsMidpoint P BC)
  (PR : Line)
  (PR_tangent : ∀ (R : Point), Tangent PR Γ R)
  (PS : Line)
  (PS_tangent : ∀ (S : Point), Tangent PS Γ S)
  (AR : Line)
  (AR_intersects : ∀ (M : Point), Intersection AR BC M)
  (AS : Line)
  (AS_intersects : ∀ (N : Point), Intersection AS BC N) :
  Distance P M = Distance P N :=
sorry

end equality_of_distances_l113_113822


namespace S_2011_eq_0_l113_113981

noncomputable def a (n : ℕ) : ℝ :=
  Real.sin (2 * n * Real.pi / 3) + Real.sqrt 3 * Real.cos (2 * n * Real.pi / 3)

noncomputable def S (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i, a i)

theorem S_2011_eq_0 : S 2011 = 0 := 
by
  sorry

end S_2011_eq_0_l113_113981


namespace product_sequence_l113_113654

noncomputable def sequence (G : ℕ → ℕ) := 
  G 1 = 1 ∧ 
  G 2 = 3 ∧ 
  ∀ n, n ≥ 2 → G (n + 1) = G n + 2 * G (n - 1)

theorem product_sequence (G : ℕ → ℕ) (hG : sequence G) :
  (∏ k in (finset.range 48).filter (λ k, 3 ≤ k+3), 
    (G (k+3) / G (k + 2) - G (k + 3) / G (k + 4))) = 
  (G 50 / G 51) := 
by 
  sorry

end product_sequence_l113_113654


namespace find_length_GH_l113_113328

-- Define the conditions as constants
def angle_G : ℝ := 30
def angle_H : ℝ := 90
def length_HI : ℝ := 10

-- The expected solution, rounded to the nearest tenth
def length_GH_expected : ℝ := 17.3

-- State the problem in Lean
theorem find_length_GH :
  angle_G = 30 ∧ angle_H = 90 ∧ length_HI = 10 →
  abs ((length_HI / (real.tan (angle_G * real.pi / 180))) - length_GH_expected) < 0.1 :=
by
  intros h
  sorry

end find_length_GH_l113_113328


namespace range_of_a_max_value_of_m_l113_113278

noncomputable def ln : ℝ → ℝ := sorry

def g (x a : ℝ) : ℝ := ln x - a * x + 1/2 * x^2

def has_two_extreme_points (g : ℝ → ℝ) : Prop := sorry

def equation_has_real_solution (m : ℤ) : Prop := 
  ∃ (x : ℝ), x > 0 ∧ ln x = m * (x + 1)

theorem range_of_a (a : ℝ) : 
  has_two_extreme_points (λ x, g(x, a)) → a > 2 :=
sorry

theorem max_value_of_m (m : ℤ) :
  equation_has_real_solution m → m ≤ 0 :=
sorry

end range_of_a_max_value_of_m_l113_113278


namespace minimum_cars_with_racing_stripes_l113_113623

variable (T A B : ℕ)

-- Conditions
axiom TotalCars : T = 100
axiom WithoutAC : A = 47
axiom MaxACNoRacingStripes : B = 47

-- Question to prove
theorem minimum_cars_with_racing_stripes :
  ∃ R : ℕ, T = 100 ∧ A = 47 ∧ B = 47 ∧ R = 6 :=
begin
  use 6,
  split, exact TotalCars,
  split, exact WithoutAC,
  split, exact MaxACNoRacingStripes,
  exact Nat.eq_refl 6,
end

end minimum_cars_with_racing_stripes_l113_113623


namespace abs_sum_zero_eq_neg_one_l113_113600

theorem abs_sum_zero_eq_neg_one (a b : ℝ) (h : |3 + a| + |b - 2| = 0) : a + b = -1 :=
sorry

end abs_sum_zero_eq_neg_one_l113_113600


namespace neg_p_sufficient_not_necessary_neg_q_l113_113878

variable (x : ℝ)

def p : Prop := 0 < x ∧ x < 2
def q : Prop := 1 / x ≥ 1

theorem neg_p_sufficient_not_necessary_neg_q : 
  (¬ p → ¬ q) ∧ ¬ (¬ q → ¬ p) := by 
sorry

end neg_p_sufficient_not_necessary_neg_q_l113_113878


namespace second_percentage_increase_l113_113472

theorem second_percentage_increase 
  (P : ℝ) 
  (x : ℝ) 
  (h1: 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := 
  sorry

end second_percentage_increase_l113_113472


namespace necessary_but_not_sufficient_condition_l113_113063

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 2 → x^2 - a > 0) → (a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l113_113063


namespace flow_rate_ratio_is_two_l113_113645

namespace Streams

-- Define the capacities of the canisters
def capacity_larger : ℕ := 10
def capacity_smaller : ℕ := 8

-- Variables for the streams' flow rates
variable {flow_rate_stronger flow_rate_weaker : ℚ}

-- Theorem statement: Given the conditions, the flow rate of the stronger stream is 2 times the flow rate of the weaker stream.
theorem flow_rate_ratio_is_two 
  (h: ∀ (x : ℚ), x ≠ 0 → (4 / x) = ((10 - x) / 4) → (4 * (10 - x)) = x * x) :
  flow_rate_stronger = 2 * flow_rate_weaker :=
begin
  -- The variable x is the amount of water in the larger canister when 4 liters have filled the smaller canister.
  let x : ℚ := 2,
  have h1 : x * (10 - x) = 16,  
  { sorry },
  have h2 : x = 2 ∨ x = 8, 
  { -- Solve the quadratic equation x^2 - 10x + 16 = 0
    sorry },
  cases h2,
  case or.inl {
    have ratio : flow_rate_stronger = 2 * flow_rate_weaker,
    { sorry },
    exact ratio,
  },
  case or.inr {
    have ratio : flow_rate_stronger = 2 * flow_rate_weaker,
    { sorry },
    exact ratio,
  }
end

end Streams

end flow_rate_ratio_is_two_l113_113645


namespace magnitude_of_angle_A_range_of_f_C_l113_113967

noncomputable def f_C (C : ℝ) : ℝ := 1 - (2 * real.cos (2 * C)) / (1 + real.tan C)

open real

variables {a b c : ℝ}
variables (A B C : ℝ)
variables (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π/2)
variables (h_eq1 : (2 * a) * cos C + 1 * (c - 2 * b) = 0)

theorem magnitude_of_angle_A :
  A = π / 3 :=
sorry

theorem range_of_f_C :
  set.Ioo ((√3 - 1) / 2) √2 = set.range f_C :=
sorry

end magnitude_of_angle_A_range_of_f_C_l113_113967


namespace negation_of_prop_p_is_correct_l113_113585

-- Define the proposition p
def prop_p : Prop := ∃ x : ℝ, tan x = 1

-- Define the negation of the proposition p
def neg_p : Prop := ∀ x : ℝ, tan x ≠ 1

-- Proof statement verifying that neg_p is the correct negation of p
theorem negation_of_prop_p_is_correct : (¬prop_p) = neg_p := by
  sorry

end negation_of_prop_p_is_correct_l113_113585


namespace wire_not_used_l113_113205

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l113_113205


namespace sin_sq_minus_2sin_eq_0_solution_l113_113729

theorem sin_sq_minus_2sin_eq_0_solution :
  {x : Real | sin x ^ 2 - 2 * sin x = 0} = {x : Real | ∃ k : ℤ, x = k * Real.pi} :=
by
  sorry

end sin_sq_minus_2sin_eq_0_solution_l113_113729


namespace right_triangles_in_rectangle_l113_113713

noncomputable def right_triangles_count (A P B C Q D : Type) [LinearOrderedRing A]
  (AB CD AD BC : A)
  (midpoint_AD : Q = (AD + D) / 2)
  (midpoint_BC : Q = (BC + C) / 2)
  (square_ADQP : AD = DQ ∧ AD = DP)
  (rhombus_BCQP : BC = BQ ∧ BC = CQ)
  (AB_ne_AD : AB ≠ AD) : ℕ :=
12

theorem right_triangles_in_rectangle (A P B C Q D : Type) [LinearOrderedRing A]
  (AB CD AD BC : A)
  (midpoint_AD : Q = (AD + D) / 2)
  (midpoint_BC : Q = (BC + C) / 2)
  (square_ADQP : AD = DQ ∧ AD = DP)
  (rhombus_BCQP : BC = BQ ∧ BC = CQ)
  (AB_ne_AD : AB ≠ AD) : 
  right_triangles_count A P B C Q D AB CD AD BC midpoint_AD midpoint_BC square_ADQP rhombus_BCQP AB_ne_AD = 12 := 
sorry

end right_triangles_in_rectangle_l113_113713


namespace inv_89_mod_90_l113_113213

theorem inv_89_mod_90 : (∃ x, 0 ≤ x ∧ x ≤ 89 ∧ 89 * x ≡ 1 [MOD 90]) :=
by {
  use 89,
  split, -- split the conjunctions
  { exact le_refl 89, }, -- 0 ≤ 89
  split, 
  { exact le_of_lt (by norm_num), }, -- 89 ≤ 89
  { rw [←Int.coe_nat_mul, Int.coe_nat_mod],
    exact Int.modeq.modeq_of_dvd (by use -88), -- because 89 * 89 - 1 = 90 * (-88), which means 89 * 89 ≡ 1 [MOD 90]
}}; sorry

end inv_89_mod_90_l113_113213


namespace unique_A3_zero_l113_113657

variable {F : Type*} [Field F]

theorem unique_A3_zero (A : Matrix (Fin 2) (Fin 2) F) 
  (h1 : A ^ 4 = 0) 
  (h2 : Matrix.trace A = 0) : 
  A ^ 3 = 0 :=
sorry

end unique_A3_zero_l113_113657


namespace solution_exists_l113_113910

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem solution_exists (a λ : ℝ) :
  f (a + 2) = 27 →
  (λ * 2 ^ (a * x) - 4 ^ x ≤ λ)
  (∀ x ∈ (Set.Icc (0 : ℝ) (2 : ℝ)), λ * 2 ^ (a * x) - 4 ^ x  ≤ λ)  → (∃ a = 1) ∧ (∃ λ = 4/3) := 
begin 
  sorry,
end

end solution_exists_l113_113910


namespace cn_is_geometric_l113_113881

open Nat

variable {α : Type*} [CommSemiring α] (a : ℕ → α) (q : α) (m : ℕ)

def geometric_seq (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

def b (a : ℕ → α) (m : ℕ) (n : ℕ) : α :=
∑ i in range m, a (m * (n - 1) + i + 1)

def c (a : ℕ → α) (m : ℕ) (n : ℕ) : α :=
∏ i in range m, a (m * (n - 1) + i + 1)

theorem cn_is_geometric (a : ℕ → α) (q : α) (m : ℕ) [fact (geometric_seq a q)] :
  ∃ r, ∀ n, c a m (n + 1) = c a m n * r :=
sorry

end cn_is_geometric_l113_113881


namespace folded_triangle_line_square_length_l113_113464

noncomputable def triangle_side_length : ℝ := 15
noncomputable def distance_AB_A_touch_point : ℝ := 11
noncomputable def folding_line_square_length : ℝ := 28561.25 / 1225

theorem folded_triangle_line_square_length :
  ∃ P Q : ℝ, P = distance_AB_A_touch_point ∧ Q = triangle_side_length - distance_AB_A_touch_point ∧ P^2 + Q^2 - 2*P*Q*real.cos (real.pi / 3) = folding_line_square_length :=
begin
  sorry
end

end folded_triangle_line_square_length_l113_113464


namespace rotate_A_to_B_l113_113633

-- Define rotation of a point (x, y) counterclockwise by 270 degrees
def rotate_270_counterclockwise (x y : ℝ) : ℝ × ℝ :=
  (y, -x)

theorem rotate_A_to_B :
  let A := (6, 8)
  let B := rotate_270_counterclockwise 6 8
  (B.1 + B.2) = 2 :=
by
  -- Definitions for clarity
  let A := (6, 8)
  let B := rotate_270_counterclockwise 6 8
  
  -- Calculate the coordinates of B
  have hB : B = (8, -6) := rfl

  -- Calculate p + q
  have hSum : B.1 + B.2 = 2 :=
    by
      rw [hB]
      -- verify the calculation
      simp

  -- Combine the results
  exact hSum

end rotate_A_to_B_l113_113633


namespace jogging_path_diameter_l113_113323

theorem jogging_path_diameter 
  (d_pond : ℝ)
  (w_flowerbed : ℝ)
  (w_jogging_path : ℝ)
  (h_pond : d_pond = 20)
  (h_flowerbed : w_flowerbed = 10)
  (h_jogging_path : w_jogging_path = 12) :
  2 * (d_pond / 2 + w_flowerbed + w_jogging_path) = 64 :=
by
  sorry

end jogging_path_diameter_l113_113323


namespace trees_left_after_typhoon_l113_113930

theorem trees_left_after_typhoon (trees_grown : ℕ) (trees_died : ℕ) (h1 : trees_grown = 17) (h2 : trees_died = 5) : (trees_grown - trees_died = 12) :=
by
  -- The proof would go here
  sorry

end trees_left_after_typhoon_l113_113930


namespace cube_volume_space_diagonal_l113_113712

theorem cube_volume_space_diagonal (s : ℝ) (h: s * √3 = 9) : s^3 = 81 * √3 := by
  sorry

end cube_volume_space_diagonal_l113_113712


namespace seventh_grade_median_eighth_grade_mean_eighth_grade_mode_eighth_grade_better_scores_l113_113743

def seventh_scores := [90, 95, 95, 80, 90, 80, 85, 90, 85, 100]
def eighth_scores := [85, 85, 95, 80, 95, 90, 90, 90, 100, 90]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

def median (l : List ℕ) : ℕ :=
  let sorted := List.sort l
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

def mode (l : List ℕ) : List ℕ :=
  let freq := l.foldl (λm a, m.insert a (m.find a |>.getD 0 + 1)) (RBMap.empty _ _)
  let max_freq := freq.values.foldl max 0
  freq.toList.filterMap (λ ⟨k, v⟩, if v = max_freq then some k else none)

def variance (l : List ℕ) : ℕ :=
  let m := mean l
  (l.map (λx, (x - m)^2)).sum / l.length

theorem seventh_grade_median :
  median seventh_scores = 90 := by sorry

theorem eighth_grade_mean :
  mean eighth_scores = 90 := by sorry

theorem eighth_grade_mode :
  mode eighth_scores = [90] := by sorry

theorem eighth_grade_better_scores :
  (mean eighth_scores > mean seventh_scores) ∧ (variance eighth_scores < variance seventh_scores) := by sorry

end seventh_grade_median_eighth_grade_mean_eighth_grade_mode_eighth_grade_better_scores_l113_113743


namespace problem_statement_l113_113572

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x then (Real.logb 4 x) else (2 ^ (-x))

theorem problem_statement : 
  f (f (-4)) + f (Real.logb 2 (1 / 6)) = 8 :=
by
  sorry

end problem_statement_l113_113572


namespace custom_star_2008_l113_113836

noncomputable def custom_star (n : ℕ) : ℕ :=
if n = 1 then 1 else sorry

theorem custom_star_2008 :
  custom_star 2004 = 3 ^ 1003 :=
by
  have h1 : custom_star 1 = 1 := sorry,
  suffices ∀ (n : ℕ), custom_star (2 * (n + 1)) = 3^(n),
  from sorry,
  intro n,
  induction n with k hk,
  {
    rw [custom_star, if_pos rfl],
    exact h1
  },
  {
    rw [custom_star, if_neg (nat.succ.ne_zero k)]
    exact sorry
  }

end custom_star_2008_l113_113836


namespace prism_diagonals_not_valid_l113_113102

theorem prism_diagonals_not_valid
  (a b c : ℕ)
  (h3 : a^2 + b^2 = 3^2 ∨ b^2 + c^2 = 3^2 ∨ a^2 + c^2 = 3^2)
  (h4 : a^2 + b^2 = 4^2 ∨ b^2 + c^2 = 4^2 ∨ a^2 + c^2 = 4^2)
  (h6 : a^2 + b^2 = 6^2 ∨ b^2 + c^2 = 6^2 ∨ a^2 + c^2 = 6^2) :
  False := 
sorry

end prism_diagonals_not_valid_l113_113102


namespace price_per_packet_of_biscuits_l113_113749

def price_cupcake := 1.5  -- price of a cupcake in dollars
def price_cookie := 2  -- price of a cookie packet in dollars
def daily_cupcakes := 20  -- number of cupcakes baked daily
def daily_cookies := 10  -- number of cookie packets baked daily
def daily_biscuits := 20  -- number of biscuit packets baked daily
def total_earnings_5_days := 350  -- total earnings in dollars for 5 days

theorem price_per_packet_of_biscuits :
  let price_biscuit := (total_earnings_5_days / 5 - (daily_cupcakes * price_cupcake + daily_cookies * price_cookie)) / daily_biscuits in
  price_biscuit = 1 :=
  by 
    sorry

end price_per_packet_of_biscuits_l113_113749


namespace find_magnitude_l113_113892

noncomputable def unit_vectors (a b : ℝ^3) : Prop := 
  ‖a‖ = 1 ∧ ‖b‖ = 1

noncomputable def angle_between (a b : ℝ^3) : Prop := 
  real.angle a b = real.pi / 3

theorem find_magnitude 
  {a b : ℝ^3} 
  (ha : unit_vectors a b) 
  (hab : angle_between a b) : 
  ‖(3 : ℝ) • a + b‖ = real.sqrt 13 := 
sorry

end find_magnitude_l113_113892


namespace length_of_AB_in_right_triangle_l113_113319

theorem length_of_AB_in_right_triangle
  (A B C : Type)
  (h_triangle : ∃ (angle_ABC : ℝ) (BC AB : ℝ), ∠ C = 90 ∧ BC = 3 ∧ cos angle_ABC = 1 / 3)
  (angle_ABC : ℝ) : ℝ :=
by
  sorry

end length_of_AB_in_right_triangle_l113_113319


namespace union_A_B_complement_intersection_A_B_range_of_a_l113_113588

-- Definitions
def A := {x : ℝ | 4 ≤ x ∧ x < 8}
def B := {x : ℝ | 5 < x ∧ x < 10}
def C (a : ℝ) := {x : ℝ | x > a}

-- Theorem statements
theorem union_A_B : A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 10} := 
by sorry

theorem complement_intersection_A_B : (compl A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 10} := 
by sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).nonempty → a < 8 := 
by sorry

end union_A_B_complement_intersection_A_B_range_of_a_l113_113588


namespace sum_of_arithmetic_sequence_l113_113833

variable (k : ℕ)

def first_term : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def num_terms : ℕ := 2 * k + 2

def nth_term (n : ℕ) : ℕ := first_term k + (n - 1) * common_difference k

theorem sum_of_arithmetic_sequence :
  let S := (num_terms k) * (first_term k + nth_term k (num_terms k)) / 2
  in S = 2 * k^3 + 6 * k^2 + 8 * k + 3 :=
by
  sorry

end sum_of_arithmetic_sequence_l113_113833


namespace maximum_car_washes_within_budget_l113_113342

def normal_price : ℝ := 15
def budget : ℝ := 250

def discounted_price (washes : ℕ) : ℝ :=
  if washes ≥ 10 ∧ washes ≤ 14 then 0.9 * normal_price
  else if washes ≥ 15 ∧ washes ≤ 19 then 0.8 * normal_price
  else if washes ≥ 20 then 0.7 * normal_price
  else normal_price

def max_washes (budget : ℝ) : ℕ :=
  let candidate1 := if (budget / (0.9 * normal_price) : ℚ).toNat ≤ 14 then (budget / (0.9 * normal_price) : ℚ).toNat else 14
  let candidate2 := if (budget / (0.8 * normal_price) : ℚ).toNat ≤ 19 then (budget / (0.8 * normal_price) : ℚ).toNat else 19
  let candidate3 := (budget / (0.7 * normal_price) : ℚ).toNat
  max (max candidate1 candidate2) candidate3

theorem maximum_car_washes_within_budget : max_washes budget = 23 :=
by sorry

end maximum_car_washes_within_budget_l113_113342


namespace domain_of_function_l113_113220

theorem domain_of_function :
  { x : ℝ | -2 ≤ x ∧ x < 4 } = { x : ℝ | (x + 2 ≥ 0) ∧ (4 - x > 0) } :=
by
  sorry

end domain_of_function_l113_113220


namespace product_of_c_values_distinct_real_roots_l113_113223

theorem product_of_c_values_distinct_real_roots :
  (∏ c in (Finset.range 16).filter (λ c, c > 0), c) = Nat.factorial 15 :=
by
  sorry

end product_of_c_values_distinct_real_roots_l113_113223


namespace proof_mean_median_modes_l113_113961

def data : List Nat :=
  List.replicate 12 1 ++ List.replicate 12 2 ++ List.replicate 12 3 ++ List.replicate 12 4 ++
  List.replicate 12 5 ++ List.replicate 12 6 ++ List.replicate 12 7 ++ List.replicate 12 8 ++
  List.replicate 12 9 ++ List.replicate 10 29 ++ List.replicate 11 30 ++ List.replicate 7 31

def median (l : List Nat) : Nat :=
  let len := l.length
  if len % 2 = 0 then
    (l[len / 2 - 1] + l[len / 2]) / 2
  else
    l[len / 2]

def mean (l : List Nat) : Float :=
  l.foldl (fun acc x => acc + x) 0 / l.length.toFloat

def modes (l : List Nat) : List Nat :=
  let freq := l.foldr (fun x m => m.insertWith Nat.add x 1) (Std.RBMap.empty)
  let maxFreq := freq.fold (fun _ v acc => max v acc) 0
  freq.fold (fun k v acc => if v = maxFreq then k :: acc else acc) []

def median_of_modes (l : List Nat) : Nat :=
  median (modes l)

theorem proof_mean_median_modes :
  let l := data
  let M := median (l.qsort Nat.le)
  let μ := mean l
  let d := median_of_modes (l.qsort Nat.le)
  d < μ ∧ μ < M :=
by
  let l := data
  let M := median (l.qsort Nat.le)
  let μ := mean l
  let d := median_of_modes (l.qsort Nat.le)
  sorry

end proof_mean_median_modes_l113_113961


namespace neg_p_iff_l113_113951

variable {x : ℝ}

def p : Prop := ∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), x^2 - 1 ≥ 0

theorem neg_p_iff : ¬ p ↔ ∃ x ∈ set.Icc (1 : ℝ) (2 : ℝ), x^2 - 1 ≤ 0 :=
by
  sorry

end neg_p_iff_l113_113951


namespace subset_sum_divisible_by_n_l113_113373

theorem subset_sum_divisible_by_n (n : ℕ) (a : Fin n → ℕ) (h_pos : ∀ i, a i > 0) : 
    ∃ (s : Finset (Fin n)), (s.sum (λ i, a i)) % n = 0 := 
  sorry

end subset_sum_divisible_by_n_l113_113373


namespace girls_attending_ball_l113_113008

theorem girls_attending_ball (g b : ℕ) 
    (h1 : g + b = 1500) 
    (h2 : 3 * g / 4 + 2 * b / 3 = 900) : 
    g = 1200 ∧ 3 * 1200 / 4 = 900 := 
by
  sorry

end girls_attending_ball_l113_113008


namespace reservoir_original_content_l113_113474

noncomputable def original_content (T O : ℝ) : Prop :=
  (80 / 100) * T = O + 120 ∧
  O = (50 / 100) * T

theorem reservoir_original_content (T : ℝ) (h1 : (80 / 100) * T = (50 / 100) * T + 120) : 
  (50 / 100) * T = 200 :=
by
  sorry

end reservoir_original_content_l113_113474


namespace range_of_a_l113_113919

theorem range_of_a (a : ℝ) (h : (∃ x : ℝ, x ∈ Iic 0 ∧ x ∈ ({1, 3, a} : set ℝ))) : a ≤ 0 :=
sorry

end range_of_a_l113_113919


namespace max_hats_l113_113492

theorem max_hats (r : ℝ) (h : ℝ) (θ : ℝ) (total_angle : ℝ) :
  (∀ t : ℕ, total_angle = 2 * Real.pi)
  → (∀ t : ℕ, h = r)
  → (∀ t : ℕ, θ = 2 * Real.pi * r / 360)
  → (∃ t : ℕ, t = 6) :=
by
  intro ht hr hθ
  use 6
  sorry

end max_hats_l113_113492


namespace apples_in_each_box_l113_113405

theorem apples_in_each_box (x : ℕ) :
  (5 * x - (60 * 5)) = (2 * x) -> x = 100 :=
by
  sorry

end apples_in_each_box_l113_113405


namespace can_form_polygon_l113_113402

-- Definition of the polygon problem
def total_match_length := 24 -- 12 matches each 2 cm long
def match_length := 2
def num_matches := 12
def required_area := 16

-- Proposition that we can form a polygon with the given conditions
theorem can_form_polygon 
  (h1 : num_matches = 12)
  (h2 : match_length = 2)
  (h3 : total_match_length = num_matches * match_length)
  (h4 : required_area = 16) :
  ∃ (polygon : Type) (P : polygon), 
    (polygon_perimeter P = total_match_length) ∧ 
    (polygon_area P = required_area) ∧ 
    (uses_all_matches P) :=
sorry

end can_form_polygon_l113_113402


namespace increasing_condition_sufficient_not_necessary_l113_113437

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0) → (a ≥ 0) ∧ ¬ (a > 0 ↔ (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0)) :=
by
  sorry

end increasing_condition_sufficient_not_necessary_l113_113437


namespace initial_eggs_ben_l113_113484

-- Let's define the conditions from step a):
def eggs_morning := 4
def eggs_afternoon := 3
def eggs_left := 13

-- Define the total eggs Ben ate
def eggs_eaten := eggs_morning + eggs_afternoon

-- Now we define the initial eggs Ben had
def initial_eggs := eggs_left + eggs_eaten

-- The theorem that states the initial number of eggs
theorem initial_eggs_ben : initial_eggs = 20 :=
  by sorry

end initial_eggs_ben_l113_113484


namespace AE_perp_BF_l113_113039

open EuclideanGeometry -- Open needed namespaces

noncomputable def triangle ABC := {A B C : Point | right_angled B A C}
def altitude_BD := {D : Point | line_through B D ∧ perp B D}
def midpoint_BD := {E : Point | midpoint B D E}
def midpoint_CD := {F : Point | midpoint C D F}

theorem AE_perp_BF
    (A B C D E F : Point)
    (hABC : triangle ABC)
    (hBD : altitude_BD D)
    (hE : midpoint_BD E)
    (hF : midpoint_CD F) :
    perp (line_through A E) (line_through B F) :=
sorry -- Proof to be filled in

end AE_perp_BF_l113_113039


namespace vertex_of_parabola_l113_113706

theorem vertex_of_parabola :
  ∀ (x : ℝ), (3 * (x - 4)^2 + 9) = 3 * (x - 4)^2 + 9 → ∃ (h k : ℝ), h = 4 ∧ k = 9 ∧ y = 3 * (x - h)^2 + k :=
by
  intros x h k
  existsi 4
  existsi 9
  apply and.intro
  { sorry }
  apply and.intro
  { sorry }
  { sorry }

end vertex_of_parabola_l113_113706


namespace valid_three_digit_numbers_count_l113_113596

theorem valid_three_digit_numbers_count : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let is_invalid n := (∃ (a b : ℕ), a ≠ b ∧ (100*a + 10*b + a = n ∨ 100*a + 10*a + b = n ∨ 100*b + 10*a + a = n))
  let valid_numbers := {n ∈ three_digit_numbers | ¬is_invalid n}
  (finset.card valid_numbers = 738) := 
by 
  sorry

end valid_three_digit_numbers_count_l113_113596


namespace arithmetic_sequence_findings_l113_113656

-- Definitions and conditions
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := (n * (a 1 + a n)) / 2

def a (n : ℕ) : ℝ := 2 * n - 1

def b (n : ℕ) : ℝ := 1 / (a (n + 1) * a (n + 2))

def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), b i

-- The main theorem: proving the findings from the conditions
theorem arithmetic_sequence_findings :
  (S 9 a = 81) → 
  (a 1 + a 13 = 26) →
  a n = 2 * n - 1 ∧ 
  ∀ n : ℕ, 30 * T n - m ≤ 0 := sorry

end arithmetic_sequence_findings_l113_113656


namespace projection_b_on_a_l113_113926

open Real

-- Definitions directly from the problem
def a : ℝ × ℝ := (1, real.sqrt 3)

-- Defining the dot product for convenience
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Defining the length for convenience
def length (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2)

-- Given condition
axiom a_perp_a_minus_3b (b : ℝ × ℝ) : dot_product a (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0

-- The projection of b on a
def projection (b : ℝ × ℝ) : ℝ :=
  dot_product a b / length a

-- The theorem to be proven
theorem projection_b_on_a (b : ℝ × ℝ) (h : a_perp_a_minus_3b b) : projection b = 2 / 3 :=
by
  sorry

end projection_b_on_a_l113_113926


namespace unique_scalar_multiple_l113_113418

variables (a b : V) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ∥ b)

theorem unique_scalar_multiple : ∃! λ : ℝ, a = λ • b := 
sorry

end unique_scalar_multiple_l113_113418


namespace probability_heads_before_tails_l113_113000

noncomputable def solve_prob : ℚ := 
  let p := λ n : ℕ, if n = 4 then 1 else if n = 3 then (1 / 2 + 1 / 2 * t 1) else
                        if n = 2 then (1 / 2 * (1/2 + 1/2 * t 1) + 1 / 2 * t 1) else
                        if n = 1 then (1 / 2 * (1 / 4 + 3 / 4 * t 1) + 1 / 2 * t 1) else
                                   (1 / 2 * (1 / 8 + 7 / 8 * t 1) + 1 / 2 * t 1)
  and t := λ n : ℕ, if n = 2 then 0 else
                        if n = 1 then 1 / 2 * (t 1 + 1 / 2) else
                                   1 / 2 * (t 1 + t n)
  in p 0

theorem probability_heads_before_tails : solve_prob = 15/23 :=
sorry

end probability_heads_before_tails_l113_113000


namespace greatest_divisor_of_three_consecutive_odds_l113_113350

theorem greatest_divisor_of_three_consecutive_odds (n : ℕ) : 
  ∃ (d : ℕ), (∀ (k : ℕ), k = 2*n + 1 ∨ k = 2*n + 3 ∨ k = 2*n + 5 → d ∣ (2*n + 1) * (2*n + 3) * (2*n + 5)) ∧ d = 3 :=
by
  sorry

end greatest_divisor_of_three_consecutive_odds_l113_113350


namespace count_valid_integers_l113_113297

def is_valid_integer (n : ℕ) : Prop :=
  200 <= n ∧ n <= 250 ∧
  let d2 := (n / 100) % 10,
      d1 := (n / 10) % 10,
      d0 := n % 10 in
  d2 = 2 ∧ 2 < d1 ∧ d1 < d0

theorem count_valid_integers : 
  (Finset.filter is_valid_integer (Finset.range (250 + 1))).card = 11 := 
by 
  sorry

end count_valid_integers_l113_113297


namespace log_ineq_solution_l113_113730

theorem log_ineq_solution (x : ℝ) : 
  (log (1/2) (2*x + 1) ≥ log (1/2) 3) ↔ (x > -1/2 ∧ x ≤ 1) :=
by
  sorry

end log_ineq_solution_l113_113730


namespace length_of_MN_l113_113073

noncomputable def square_side_length (a : ℝ) : Prop := 
  ∃ M N : ℝ, 
  let BM := a * (Real.sqrt 6 - Real.sqrt 2), 
      BN := a * (Real.sqrt 6 - Real.sqrt 2) in
  BM = BN

theorem length_of_MN (a : ℝ) (h : square_side_length a) : 
  let BM := a * (Real.sqrt 6 - Real.sqrt 2) in
  square_side_length a → BM = a * (Real.sqrt 6 - Real.sqrt 2) := 
by {
  sorry 
}

end length_of_MN_l113_113073


namespace min_PF1_dot_PF2_l113_113581

noncomputable def hyperbola_min_PF1_dot_PF2 (x y b : ℝ) (H : 0 < b) : ℝ :=
  let e := (Real.sqrt 5 / 2)
  let a := 2
  let c := Real.sqrt 5
  (x, y) ∈ Set.Icc (0, 0) (0, b) -> (x^2 + y^2 - 5)

theorem min_PF1_dot_PF2 (x y b : ℝ) (H : 0 < b):
  let e := (Real.sqrt 5 / 2)
  let a := 2
  let c := Real.sqrt 5
  let condition : (x, y) ∈ Set.Icc (0, 0) (0, b)
  let PF1_dot_PF2 := (x^2 + y^2 - 5)
  condition -> PF1_dot_PF2 = -21 / 5 :=
by
  sorry

end min_PF1_dot_PF2_l113_113581


namespace trigonometric_identities_l113_113548

theorem trigonometric_identities (α : Real) (h1 : 3 * π / 2 < α ∧ α < 2 * π) (h2 : Real.sin α = -3 / 5) :
  Real.tan α = 3 / 4 ∧ Real.tan (α - π / 4) = -1 / 7 ∧ Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end trigonometric_identities_l113_113548


namespace irreducible_polynomial_l113_113349

open Polynomial

theorem irreducible_polynomial 
  (p : ℕ) (n : ℕ → ℕ) (d : ℕ) 
  (hp : Prime p) 
  (hn_pos : ∀ i, 0 < n i) 
  (hd : d = gcd_list (List.ofFn n)) :
  Irreducible (C ((-p:ℚ)) + (∑ i in Finset.range p, X^(n i))) / (X^d - 1 : ℚ[X]) :=
sorry

end irreducible_polynomial_l113_113349


namespace angle_of_inclination_l113_113507

theorem angle_of_inclination (x y: ℝ) (h: x + real.sqrt 3 * y - 5 = 0) : 
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ real.tan θ = -1/real.sqrt 3 ∧ θ = 150 :=
sorry

end angle_of_inclination_l113_113507


namespace between_200_and_250_has_11_integers_with_increasing_digits_l113_113294

/-- How many integers between 200 and 250 have three different digits in increasing order --/
theorem between_200_and_250_has_11_integers_with_increasing_digits :
  ∃ n : ℕ, n = 11 ∧ ∀ (x : ℕ),
    200 ≤ x ∧ x ≤ 250 →
    (∀ i j k : ℕ, (x = 100 * i + 10 * j + k) →
    i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) ↔ x ∈ {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249} :=
by {
  let s := {234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249},
  existsi 11,
  simp [s],
  intro x,
  split,
  intros h _,
  have hx : x ∈ s, sorry, -- proof needed
  exact hx,
  intros h _,
  split,
  intros i j k hx _,
  have hi : i = 2, sorry, -- proof needed
  cases h with j_values _,
  case or.inl : { rw j_values, exact sorry }, -- proof needed
  case or.inr : { rw j_values, exact sorry }, -- proof needed
  exact sorry -- scaffolding, proof steps go here
}

end between_200_and_250_has_11_integers_with_increasing_digits_l113_113294


namespace pretty_18_sum_div_18_l113_113835

def is_pretty_18 (n : ℕ) : Prop :=
  n % 18 = 0 ∧ ∃ d, d > 0 ∧ d = 18 ∧ ( nat.divisors n ).card = 18 

noncomputable def sum_pretty_18_below_1000 :=
  ∑ n in (finset.range 1000).filter is_pretty_18, n

theorem pretty_18_sum_div_18 : sum_pretty_18_below_1000 / 18 = 112 :=
by 
  sorry

end pretty_18_sum_div_18_l113_113835


namespace range_of_a_perpendicular_tangent_l113_113604

def has_vertical_tangent (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x, deriv f x = real.infinity

theorem range_of_a_perpendicular_tangent (a : ℝ) :
  (∃ x, (f : ℝ → ℝ) = λ x, a * x^3 + real.log x ∧ deriv f x = real.infinity) ↔ a ∈ Iio 0 := 
begin
  sorry
end

end range_of_a_perpendicular_tangent_l113_113604


namespace find_vector_b_l113_113558

-- Define the given vectors and their properties
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ
def angle_b_a : ℝ := 180
def mag_b : ℝ := 3 * Real.sqrt 5

-- Assert the conditions
axiom angle_condition : angle_b_a = 180
axiom magnitude_condition : ∥b∥ = mag_b -- ∥.∥ is the magnitude (norm) function.

-- Prove that vector b is (-3, 6)
theorem find_vector_b : b = (-3, 6) :=
  by
    sorry

end find_vector_b_l113_113558


namespace probability_circle_containment_l113_113697

theorem probability_circle_containment :
  let a_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let circle_C_contained (a : ℕ) : Prop := a > 3
  let m : ℕ := (a_set.filter circle_C_contained).card
  let n : ℕ := a_set.card
  let p : ℚ := m / n
  p = 4 / 7 := 
by
  sorry

end probability_circle_containment_l113_113697


namespace base_conversion_example_l113_113516

theorem base_conversion_example :
  let n1 := 2 * 8^2 + 5 * 8^1 + 4 * 8^0,
      d1 := 1 * 4^1 + 3 * 4^0,
      n2 := 1 * 5^2 + 3 * 5^1 + 2 * 5^0,
      d2 := 3 * 4^1 + 2 * 4^0 in
  n1 / d1 + n2 / d2 = 28 :=
by
  sorry

end base_conversion_example_l113_113516


namespace number_of_smaller_triangles_l113_113794
-- Import necessary libraries

-- Define the parameters for the large and small triangles
def large_triangle_sidelength := 15
def small_triangle_sidelength := 3

-- Define the areas for the geometric calculations
def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- Calculate the areas of the large and small triangles
def area_large_triangle := area_equilateral_triangle large_triangle_sidelength
def area_small_triangle := area_equilateral_triangle small_triangle_sidelength

-- Define the final statement to be proven
theorem number_of_smaller_triangles : (area_large_triangle / area_small_triangle) = 25 := 
by
  sorry

end number_of_smaller_triangles_l113_113794


namespace function_is_even_l113_113714

def f (x : ℝ) : ℝ := Real.sin ((2005 / 2) * Real.pi - 2004 * x)

theorem function_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end function_is_even_l113_113714


namespace total_students_l113_113035

theorem total_students (teams students_per_team : ℕ) (h1 : teams = 9) (h2 : students_per_team = 18) :
  teams * students_per_team = 162 := by
  sorry

end total_students_l113_113035


namespace polynomial_satisfies_condition_l113_113195

-- Define P as a real polynomial
def P (a : ℝ) (X : ℝ) : ℝ := a * X

-- Define a statement that needs to be proven
theorem polynomial_satisfies_condition (P : ℝ → ℝ) :
  (∀ X : ℝ, P (2 * X) = 2 * P X) ↔ ∃ a : ℝ, ∀ X : ℝ, P X = a * X :=
by
  sorry

end polynomial_satisfies_condition_l113_113195


namespace carl_city_mileage_l113_113182

noncomputable def city_mileage (miles_city mpg_highway cost_per_gallon total_cost miles_highway : ℝ) : ℝ :=
  let total_gallons := total_cost / cost_per_gallon
  let gallons_highway := miles_highway / mpg_highway
  let gallons_city := total_gallons - gallons_highway
  miles_city / gallons_city

theorem carl_city_mileage :
  city_mileage 60 40 3 42 200 = 20 / 3 := by
  sorry

end carl_city_mileage_l113_113182


namespace find_x_l113_113937

-- Given condition
def condition (x : ℝ) : Prop := 3 * x - 5 * x + 8 * x = 240

-- Statement (problem to prove)
theorem find_x (x : ℝ) (h : condition x) : x = 40 :=
by 
  sorry

end find_x_l113_113937


namespace hexagon_perimeter_l113_113415

noncomputable def side_lengths : (real × real × real × real × real) := (1, 1, 2, 2, 1)

theorem hexagon_perimeter (AB BC CD DE EF : ℝ) :
  AB = 1 → BC = 1 → CD = 2 → DE = 2 → EF = 1 →
  AB + BC + CD + DE + EF + real.sqrt 10 = 7 + real.sqrt 10 :=
by
  intros hAB hBC hCD hDE hEF
  simp [hAB, hBC, hCD, hDE, hEF]
  sorry

end hexagon_perimeter_l113_113415


namespace solution_l113_113193

noncomputable def F (a b c : ℝ) := a * (b ^ 3) + c

theorem solution (a : ℝ) (h : F a 2 3 = F a 3 10) : a = -7 / 19 := sorry

end solution_l113_113193


namespace flavors_remaining_to_try_l113_113929

def total_flavors : ℕ := 100
def flavors_tried_two_years_ago (total_flavors : ℕ) : ℕ := total_flavors / 4
def flavors_tried_last_year (flavors_tried_two_years_ago : ℕ) : ℕ := 2 * flavors_tried_two_years_ago

theorem flavors_remaining_to_try
  (total_flavors : ℕ)
  (flavors_tried_two_years_ago : ℕ)
  (flavors_tried_last_year : ℕ) :
  flavors_tried_two_years_ago = total_flavors / 4 →
  flavors_tried_last_year = 2 * flavors_tried_two_years_ago →
  total_flavors - (flavors_tried_two_years_ago + flavors_tried_last_year) = 25 :=
by
  sorry

end flavors_remaining_to_try_l113_113929


namespace collinear_P_A_Z_l113_113778

variables {A B C P Q R X Y Z : Type} 
variables [c : CyclicQuadrilateral B C Q R]
variables [t : Triangle A B C]
variables [p : InsideTriangle P A B C]
variables [on_AB : OnSegment Q A B]
variables [on_AC : OnSegment R A C]

def circumcircle (X Y Z : Type) := sorry

def intersection (X  circXY circYZ: Type) := sorry

def collinear (P A Z : Type) := sorry

theorem collinear_P_A_Z :
  ∀ (t : Triangle A B C) (p : InsideTriangle P A B C) 
  (on_AB : OnSegment Q A B) (on_AC : OnSegment R A C)
  (circBCQR : CyclicQuadrilateral B C Q R)
  (circQAP := circumcircle Q A P)
  (circPRA := circumcircle P R A)
  (circBCQR := circumcircle B C Q R)
  (X := intersection circQAP circBCQR)
  (Y := intersection circPRA circBCQR),
  ∃ Z, intersection (lineBY B Y) (lineCX C X) = Z → collinear P A Z := sorry

end collinear_P_A_Z_l113_113778


namespace shaded_region_area_l113_113135

def area_shaded : ℝ :=
  let r1 := 2
  let r2 := 4
  let R := 8
  let area_large := π * R^2
  let area_small1 := π * r1^2
  let area_small2 := π * r2^2
  area_large - area_small1 - area_small2

theorem shaded_region_area (R r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 4) (hR : R = 8) :
  let area_large := π * R^2
  let area_small1 := π * r1^2
  let area_small2 := π * r2^2
  area_large - area_small1 - area_small2 = 44 * π :=
by sorry

end shaded_region_area_l113_113135


namespace monotonic_increasing_exists_odd_function_l113_113912

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem monotonic_increasing (a : ℝ) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

theorem exists_odd_function : ∃ a : ℝ, a = 1 ∧ (∀ x : ℝ, f a (-x) = -f a x) :=
sorry

end monotonic_increasing_exists_odd_function_l113_113912


namespace gardening_project_cost_l113_113170

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end gardening_project_cost_l113_113170


namespace students_enrolled_both_english_and_german_l113_113621

def total_students : ℕ := 32
def enrolled_german : ℕ := 22
def only_english : ℕ := 10
def students_enrolled_at_least_one_subject := total_students

theorem students_enrolled_both_english_and_german :
  ∃ (e_g : ℕ), e_g = enrolled_german - only_english :=
by
  sorry

end students_enrolled_both_english_and_german_l113_113621


namespace total_floor_area_l113_113444

theorem total_floor_area
    (n : ℕ) (a_cm : ℕ)
    (num_of_slabs : n = 30)
    (length_of_slab_cm : a_cm = 130) :
    (30 * ((130 * 130) / 10000)) = 50.7 :=
by
  sorry

end total_floor_area_l113_113444


namespace simplify_to_linear_form_l113_113378

theorem simplify_to_linear_form (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := 
by 
  sorry

end simplify_to_linear_form_l113_113378


namespace distance_proof_l113_113978

-- Definitions based on conditions
def polar_point : ℝ × ℝ := (sqrt 2, π / 4)
def polar_line (ρ θ : ℝ) : Prop := ρ * cos θ - ρ * sin θ - 1 = 0

-- Conversion functions
def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Point and Line in rectangular coordinates
def rect_point : ℝ × ℝ := (1, 1)
def rect_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Distance function
def distance_point_to_line (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  abs (P.fst - P.snd - 1) / sqrt 2

-- Theorem statement
theorem distance_proof :
  distance_point_to_line rect_point rect_line = sqrt 2 / 2 := by
  sorry

end distance_proof_l113_113978


namespace students_more_than_pets_l113_113515

theorem students_more_than_pets
    (num_classrooms : ℕ)
    (students_per_classroom : ℕ)
    (rabbits_per_classroom : ℕ)
    (hamsters_per_classroom : ℕ)
    (total_students : ℕ)
    (total_pets : ℕ)
    (difference : ℕ)
    (classrooms_eq : num_classrooms = 5)
    (students_eq : students_per_classroom = 20)
    (rabbits_eq : rabbits_per_classroom = 2)
    (hamsters_eq : hamsters_per_classroom = 1)
    (total_students_eq : total_students = num_classrooms * students_per_classroom)
    (total_pets_eq : total_pets = num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom)
    (difference_eq : difference = total_students - total_pets) :
  difference = 85 := by
  sorry

end students_more_than_pets_l113_113515


namespace arithmetic_sequence_n_l113_113969

theorem arithmetic_sequence_n 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3_plus_a5 : a 3 + a 5 = 14)
  (Sn_eq_100 : S n = 100) :
  n = 10 :=
sorry

end arithmetic_sequence_n_l113_113969


namespace derivative_of_function_l113_113112

noncomputable def derivative_expr (x : ℝ) : ℝ :=
  (    (derivative (λ x : ℝ, (∛(Real.cot 2)) - (1/20) * (cos (10*x))^2 / (sin (20*x))))
         x) 

theorem derivative_of_function (x : ℝ) : 
 (derivative_expr x) = (1/(4 * (sin (10 * x))^2)) :=
 by
  sorry

end derivative_of_function_l113_113112


namespace question1_invalid_question2_minimum_a_l113_113160

-- Question (1) verification
theorem question1_invalid (x : ℝ) (h1 : 10 ≤ x ∧ x ≤ 1000) :
    (f : ℝ → ℝ) → (∀ x, f(x) = x / 150 + 2 → f(x) ≤ x / 5) → False :=
by {
  sorry
}

-- Question (2) minimum positive integer a
theorem question2_minimum_a : ∃ (a : ℕ), (a ≥ 328) ∧ 
  (∀ x, 10 ≤ x ∧ x ≤ 1000 → 
  (let f(x) := (10 * x - 3 * a) / (x + 2)
  in f(x) ≤ 9 ∧ f(x) = 10 - (3 * a + 20) / (x + 2) ∧ f(x) ≤ x / 5)) :=
by {
  sorry
}

end question1_invalid_question2_minimum_a_l113_113160


namespace cond_expectation_equals_g_l113_113007

noncomputable def independent_random_variable
  (ξ : ℕ → ℝ) (G : Set (Set ℝ)) : Prop := sorry

noncomputable def expectation_finite
  (f : ℝ → ℝ → ℝ) (ξ : ℕ → ℝ) (ζ : ℝ) : Prop :=
  ∫⁻ (xy : ℝ × ℝ), ∥f (xy.1) (xy.2)∥ ∂measure.prod volume volume < ∞

theorem cond_expectation_equals_g
  {ξ : ℕ → ℝ}
  {G : Set (Set ℝ)}
  {f : ℝ → ℝ → ℝ}
  {ζ : ℝ}
  (h_independent : independent_random_variable ξ G)
  (h_finite : expectation_finite f ξ ζ) :
  ∫⁻ (x : ℝ), f (ξ x) ζ ∂volume = ∫⁻ (x : ℝ), (fun y => ∫⁻ (z : ℝ), f z y ∂volume) ζ ∂measure.of G :=
sorry

end cond_expectation_equals_g_l113_113007


namespace probability_one_neighborhood_unassigned_l113_113327

-- Defining the problem parameters
def neighborhoods : Finset (Fin 3) := Finset.univ
def staffMembers : Fin 4 := Fin.mk 4 (by decide)

-- Calculate the number of ways to assign the staff members
def totalAssignments : ℕ := 3^4

-- Define a function that counts the valid assignments where exactly one neighborhood gets no staff members
-- This will be in two scenarios: (2, 2, 0) and (3, 1, 0)
def validAssignments : ℕ := (3 * Nat.choose 4 2 * Nat.choose 2 2 * 2) + (3 * Nat.choose 4 1 * 1 * 2)

-- The probability computation
def probabilityOfOneUnassignedNeighborhood : ℚ :=
  validAssignments / totalAssignments

-- The theorem to prove
theorem probability_one_neighborhood_unassigned :
  probabilityOfOneUnassignedNeighborhood = 14 / 27 :=
by
  sorry

end probability_one_neighborhood_unassigned_l113_113327


namespace lemon_heads_per_package_l113_113650

theorem lemon_heads_per_package (total_lemon_heads : ℕ) (total_boxes : ℕ) 
  (ate_total : total_lemon_heads = 54) (total_is_boxes : total_boxes = 9) : 
  (total_lemon_heads / total_boxes = 6) := 
by
  rw [ate_total, total_is_boxes]
  norm_num

end lemon_heads_per_package_l113_113650


namespace repeating_decimals_expr_as_fraction_l113_113178

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l113_113178


namespace height_of_fixed_point_l113_113085

-- Define lengths of the rods
def L1 := 1
def L2 := 2
def L3 := 3

-- Given the condition that the three rods form a right angle at their common end,
-- and their free ends lie on a plane, prove the height of the fixed point.

theorem height_of_fixed_point : 
  let d1 := Real.sqrt (L1^2 + L2^2)
  let d2 := Real.sqrt (L1^2 + L3^2)
  let d3 := Real.sqrt (L2^2 + L3^2)
  let base_area := (d1 * d2 * Real.sqrt (d3^2 + d1^2 - d2^2) / 4) 
  let volume := L1 * L2 * L3 / 6
  volume = (base_area * (6 / 7)) / 3 → 
  (6 / 7)
:= by
  sorry

end height_of_fixed_point_l113_113085


namespace age_of_son_l113_113805

theorem age_of_son (D S : ℕ) (h₁ : S = D / 4) (h₂ : D - S = 27) (h₃ : D = 36) : S = 9 :=
by
  sorry

end age_of_son_l113_113805


namespace a_plus_b_eq_one_l113_113317

open Real

noncomputable def circle_center : Point := (-1, -1)

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 1 = 0

noncomputable def line_eq (a b x y : ℝ) : Prop :=
  a * x + b * y + 1 = 0

theorem a_plus_b_eq_one (a b : ℝ) :
  (∀ x y, circle_eq x y → line_eq a b x y → (x, y) = circle_center) → a + b = 1 :=
by
  intro h
  sorry

end a_plus_b_eq_one_l113_113317


namespace sum_first_ten_terms_l113_113430

theorem sum_first_ten_terms (a d : ℝ) (h : a + 7 * d = 10) : 
  let S₁₀ : ℝ := 5 * (2 * a + 9 * d)
  in S₁₀ = 100 - 25 * d :=
by
  sorry

end sum_first_ten_terms_l113_113430


namespace infinite_power_tower_equation_l113_113388

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ x ^ x ^ x ^ x -- continues infinitely

theorem infinite_power_tower_equation (x : ℝ) (h_pos : 0 < x) (h_eq : infinite_power_tower x = 2) : x = Real.sqrt 2 :=
  sorry

end infinite_power_tower_equation_l113_113388


namespace monotonic_intervals_and_range_of_f_l113_113566

def f (x : ℝ) : ℝ :=
  (sin (x / 2) + cos (x / 2))^2 - 2 * sqrt 3 * (cos (x / 2))^2 + sqrt 3

theorem monotonic_intervals_and_range_of_f :
  (∃ k : ℤ, ∀ x : ℝ, 2 * k * π - π / 6 ≤ x ∧ x ≤ 5 * π / 6 + 2 * k * π → monotone (λ y, f y)) ∧
  (∃ k : ℤ, ∀ x : ℝ, 5 * π / 6 + 2 * k * π ≤ x ∧ x ≤ 11 * π / 6 + 2 * k * π → antitone (λ y, f y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → 1 - sqrt 3 ≤ f x ∧ f x ≤ 3) := 
sorry

end monotonic_intervals_and_range_of_f_l113_113566


namespace derivative_of_y_l113_113114

noncomputable def y (x : ℝ) : ℝ :=
  (real.cot 2)^(1/3) - (1/20) * ((cos (10 * x))^2 / sin (20 * x))

theorem derivative_of_y :
  ∀ x : ℝ, deriv y x = 1 / (4 * (sin (10 * x))^2) :=
by
  assume x,
  sorry

end derivative_of_y_l113_113114


namespace cistern_fill_time_with_leak_l113_113129

theorem cistern_fill_time_with_leak {R L : ℝ} (hR : R = 1 / 2) (hL : L = 1 / 4) :
  let effective_rate := R - L
  let time_without_leak := 2
  let time_with_leak := 1 / effective_rate
  let extra_time := time_with_leak - time_without_leak
  in extra_time = 2 :=
by
  let effective_rate := R - L
  let time_with_leak := 1 / effective_rate
  let time_without_leak := 2
  let extra_time := time_with_leak - time_without_leak
  sorry

end cistern_fill_time_with_leak_l113_113129


namespace find_m_l113_113717

-- Define the conditions of the problem
def hyperbola_eq (x y m : ℝ) : Prop := x^2 - (y^2 / m) = 1

def foci_eq (c : ℝ) : Prop := c = 3

-- Define the equivalent proof problem
theorem find_m (m : ℝ) 
  (H_hyperbola : ∀ x y, hyperbola_eq x y m)
  (H_foci : foci_eq (real.sqrt (m - 1))) : m = 8 := 
sorry

end find_m_l113_113717


namespace log_sum_range_l113_113534

theorem log_sum_range (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  (Real.log y / Real.log x + Real.log x / Real.log y) ∈ Set.union (Set.Iic (-2)) (Set.Ici 2) :=
sorry

end log_sum_range_l113_113534


namespace sum_seven_smallest_multiples_of_12_l113_113759

theorem sum_seven_smallest_multiples_of_12 :
  (Finset.sum (Finset.range 7) (λ n, 12 * (n + 1))) = 336 :=
by
  -- proof (sorry to skip)
  sorry

end sum_seven_smallest_multiples_of_12_l113_113759


namespace estimate_product_l113_113847

/-- 
Given approximations 819 ≈ 800 and 32 ≈ 30,
prove that 819 × 32 ≈ 24000.
-/
theorem estimate_product : 
  ∀ (a b c d : ℕ), 
    a ≈ 800 → b ≈ 30 → c = 819 → d = 32 → (c * d) ≈ 24000 :=
by
  intros a b c d ha hb hc hd
  -- Insert proof steps here
  sorry

end estimate_product_l113_113847


namespace correct_prism_description_l113_113476

/-- Definitions for the conditions regarding a prism -/
def option_A (p : Prism) : Prop := p.has_only_two_parallel_faces
def option_B (p : Prism) : Prop := p.all_edges_equal
def option_C (p : Prism) : Prop := p.all_faces_are_parallelograms
def option_D (p : Prism) : Prop := p.base_faces_parallel_and_all_lateral_edges_equal

/-- Main theorem stating that the correct description of a prism is given by option D -/
theorem correct_prism_description (p : Prism) : 
  (option_D p) ∧ ¬(option_A p) ∧ ¬(option_B p) ∧ ¬(option_C p) := 
by 
  sorry

end correct_prism_description_l113_113476


namespace michael_robots_l113_113167

-- Conditions
def tom_robots := 3
def times_more := 4

-- Theorem to prove
theorem michael_robots : (times_more * tom_robots) + tom_robots = 15 := by
  sorry

end michael_robots_l113_113167


namespace ratio_of_trapezoid_area_to_triangle_area_l113_113026

theorem ratio_of_trapezoid_area_to_triangle_area :
  ∀ (A B C D E F G : Type)
    [HasRightTriangle A B C]
    (h_right : right_angle A B C)
    (h_parallel_1 : parallel DE AB)
    (h_parallel_2 : parallel FG AB)
    (h_length_AD_DF_FB : AD = DF ∧ DF = FB ∧ AD = (AB / 3))
    (h_length_AB : AB = 8)
    (h_length_BC : BC = 6), 
  (area FGAB / area ABC) = 5 / 9 := 
sorry


end ratio_of_trapezoid_area_to_triangle_area_l113_113026


namespace spherical_coordinates_standard_equivalence_l113_113630

def std_spherical_coords (ρ θ φ: ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_standard_equivalence :
  std_spherical_coords 5 (11 * Real.pi / 6) (2 * Real.pi - 5 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_standard_equivalence_l113_113630


namespace sum_first_6_terms_l113_113972

variable (a : ℕ → ℚ)
variable (d : ℚ)

def arithmetic_sequence (a d : ℚ) : ℕ → ℚ
| 0     := a
| (n+1) := arithmetic_sequence a d n + d

def S₆ (a₁ d : ℚ) := 6 * a₁ + 15 * d

theorem sum_first_6_terms (h₁ : arithmetic_sequence 0 d 4 = 1/2) 
    (h₂ : 8 * arithmetic_sequence 0 d 5 + 2 * arithmetic_sequence 0 d 3 = arithmetic_sequence 0 d 1)
    : S₆ (5/2) (-1/2) = 15/2 := by
  sorry

end sum_first_6_terms_l113_113972


namespace proof_problem_l113_113952

-- Definitions and conditions
variable (a b c A B C : ℝ)
variable (cosA cosB cosC : ℝ)
variable (pi : ℝ) (hpi : pi = Real.pi)

-- Conditions based on the problem
variable (h1 : a * cosC + (1/2) * c = b)
variable (hb1 : b = 4)
variable (hc1 : c = 6)
variable (hA : A = pi / 3)

-- Specify the goals for proof
theorem proof_problem : 
  (cosA = 1 / 2 ∧ A = pi / 3) ∧ 
  (b = 4 ∧ c = 6 ∧ A = pi / 3 → cosB = 2 / Real.sqrt 7 ∧ cos (A + 2 * B) = -11 / 14) :=
  by
  sorry

end proof_problem_l113_113952


namespace find_angle_B_l113_113962

noncomputable def angle_triangle_sum {α : Type} [linear_order_α : linear_order α] 
  (a b c : α) : Prop := a + b + c = 180

variables (A B C D : ℕ)
variables (E : ℕ)
variables (x : ℕ)

-- Conditions
def angle_A : Prop := A = 60
def angle_B : Prop := B = 2 * C
def angle_division : Prop := A / 2 = 30
def angle_BEC : Prop := E = 20
def angle_triangle_BEC : Prop := angle_triangle_sum E C (180 - B)

-- Proof goal
theorem find_angle_B :
  angle_A A → angle_B B C → angle_division A → angle_BEC E →
  angle_triangle_BEC E C B → B = 40 := 
  by intros; sorry

end find_angle_B_l113_113962


namespace max_likelihood_estimates_l113_113802

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {a : ℝ}
variable {σ2 : ℝ}

noncomputable def mean_estimate (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  (finset.sum (finset.univ) (λ i, x i)) / n

noncomputable def variance_estimate (n : ℕ) (x : Fin n → ℝ) (a : ℝ) : ℝ :=
  (finset.sum (finset.univ) (λ i, (x i - a) ^ 2)) / n

theorem max_likelihood_estimates (n : ℕ) (x : Fin n → ℝ) :
  a = mean_estimate n x ∧ σ2 = variance_estimate n x a :=
sorry

end max_likelihood_estimates_l113_113802


namespace circle_diameter_l113_113044

theorem circle_diameter (A : ℝ) (h : A = 100 * Real.pi) : ∃ D : ℝ, D = 20 :=
by
  let r := Real.sqrt (A / Real.pi)
  have hr : r = 10 := sorry
  have D : ℝ := 2 * r
  have hD : D = 20 := sorry
  use D
  exact hD

end circle_diameter_l113_113044


namespace domain_of_f_l113_113216

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end domain_of_f_l113_113216


namespace price_of_candied_grape_l113_113491

theorem price_of_candied_grape (x : ℝ) (h : 15 * 2 + 12 * x = 48) : x = 1.5 :=
by
  sorry

end price_of_candied_grape_l113_113491


namespace Hillary_deposit_l113_113165

theorem Hillary_deposit (price_per_craft : ℕ) (crafts_sold : ℕ) (extra_money : ℕ) (remaining_money : ℕ) 
  (total_earned := price_per_craft * crafts_sold + extra_money) 
  (deposit := total_earned - remaining_money) :
  price_per_craft = 12 →
  crafts_sold = 3 →
  extra_money = 7 →
  remaining_money = 25 →
  deposit = 18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold total_earned deposit
  norm_num
  sorry

end Hillary_deposit_l113_113165


namespace total_distance_l113_113345

open Real

theorem total_distance :
  let jonathan_distance := 7.5
  let mercedes_distance := 2.5 * jonathan_distance
  let davonte_distance := real.sqrt (3.25 * mercedes_distance)
  let felicia_distance := davonte_distance - 1.75
  let average_distance := (jonathan_distance + davonte_distance + felicia_distance) / 3
  let emilia_distance := average_distance ^ 2
  in
      mercedes_distance + davonte_distance + felicia_distance + emilia_distance ≈ 83.321 := 
by
  let jonathan_distance := 7.5
  let mercedes_distance := 2.5 * jonathan_distance
  let davonte_distance := real.sqrt (3.25 * mercedes_distance)
  let felicia_distance := davonte_distance - 1.75
  let average_distance := (jonathan_distance + davonte_distance + felicia_distance) / 3
  let emilia_distance := average_distance ^ 2
  have mercedes_dist_calc : mercedes_distance = 18.75 := by norm_num [jonathan_distance]
  -- More steps to simplify and verify the distances
  have davonte_dist_calc : davonte_distance = sqrt 60.9375 := by norm_num [mercedes_distance]
  -- More steps to simplify and verify the distances
  have felicia_dist_calc : felicia_distance = davonte_distance - 1.75 := by norm_num [davonte_distance]
  have avg_dist_calc : average_distance = (jonathan_distance + davonte_distance + felicia_distance) / 3 := by norm_num
  have emilia_dist_calc : emilia_distance = average_distance ^ 2 := by norm_num [average_distance]
  have total_distance := mercedes_distance + davonte_distance + felicia_distance + emilia_distance
  -- Verify total_distance is approximately equal to 83.321
  have : abs (total_distance - 83.321) < 0.001,
  sorry

end total_distance_l113_113345


namespace f_periodic_and_specific_value_l113_113504

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 2 then Real.log2 (x + 1) else 
if 2 ≤ x then f (x - 2) else 
-f (x + 2)

theorem f_periodic_and_specific_value : 
  (∀ x : ℝ, f (x + 1) = -f x) → 
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log2 (x + 1)) → 
  f (2012) - f (2011) = -1 := 
by
  intros Hf Hf2
  sorry

end f_periodic_and_specific_value_l113_113504


namespace mass_percentage_Al_is_35_94_l113_113858

-- Declare the given problem conditions
def molar_mass_Al : ℝ := 26.98  -- given in g/mol
def molar_mass_S : ℝ := 32.06   -- given in g/mol

-- Define the total molar mass of Al2S3
def molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)

-- Define the total mass of Al in Al2S3
def total_mass_Al_in_Al2S3 : ℝ := 2 * molar_mass_Al

-- Define the mass percentage of Al in Al2S3
def mass_percentage_Al_in_Al2S3 : ℝ := (total_mass_Al_in_Al2S3 / molar_mass_Al2S3) * 100

-- The main statement to be proved
theorem mass_percentage_Al_is_35_94 :
  mass_percentage_Al_in_Al2S3 = 35.94 :=
by 
  -- since this is only for generating the statement, insert 'sorry' to skip proof
  sorry

end mass_percentage_Al_is_35_94_l113_113858


namespace coeff_x2_expansion_eq_160_l113_113908

theorem coeff_x2_expansion_eq_160
  (a : ℝ)
  (h₁ : ∑ c in (3*x + (a / (2*x))) * (2*x - 1/x)^5) = 4) :
  coefficient of x^2 ((3*x + (a / (2*x))) * (2*x - 1/x)^5) = 160 := by
sorry

end coeff_x2_expansion_eq_160_l113_113908


namespace triangle_incenter_distance_l113_113746

open EuclideanGeometry

theorem triangle_incenter_distance
  (P Q R : Point)
  (hPQ : dist P Q = 31)
  (hPR : dist P R = 29)
  (hQR : dist Q R = 30)
  (J : Point)
  (hJ : is_incenter P Q R J) :
  dist P J = Real.sqrt 233 := by
  sorry

end triangle_incenter_distance_l113_113746


namespace ellipse_focal_distance_correct_l113_113906

noncomputable def ellipse_focal_distance (x y : ℝ) (θ : ℝ) : ℝ :=
  let a := 5 -- semi-major axis
  let b := 2 -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2) -- calculate focal distance
  2 * c -- return 2c

theorem ellipse_focal_distance_correct (θ : ℝ) :
  ellipse_focal_distance (-4 + 2 * Real.cos θ) (1 + 5 * Real.sin θ) θ = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_focal_distance_correct_l113_113906


namespace SequenceProblem_l113_113249

section SequenceProblem
variable (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ)

-- Given initial conditions
axiom a1 : a 1 = 2
axiom a2 : a 2 = 8
axiom Sn_relation : ∀ n ≥ 1, S (n + 1) + 4 * S (n - 1) = 5 * S n
axiom T_def : ∀ n, T (n) = ∑ i in range (n + 1), log2 (a i)

-- Definitions of sequences
def an (n : ℕ) : ℕ := 2 ^ (2 * n - 1)

-- The main theorem to prove
theorem SequenceProblem : 
  (∀ n, a n = an n) ∧
  (∃ n : ℕ, ∏ k in finset.range (n - 1).succ, (1 - 1 / (T (k + 2))) ≥ 1009 / 2016 ∧ n = 1008) :=
  sorry
end SequenceProblem

end SequenceProblem_l113_113249


namespace circle_diameter_l113_113045

theorem circle_diameter (A : ℝ) (h : A = 100 * Real.pi) : ∃ D : ℝ, D = 20 :=
by
  let r := Real.sqrt (A / Real.pi)
  have hr : r = 10 := sorry
  have D : ℝ := 2 * r
  have hD : D = 20 := sorry
  use D
  exact hD

end circle_diameter_l113_113045


namespace batsman_runs_needed_l113_113960

theorem batsman_runs_needed 
  (A : ℕ) (H1 : A + 3 = (16 * A + 85) / 17)
  (R : ℕ) (H2 : R = 7.5) : 
  needs_to_score : ℕ :=
by 
  let total_runs_before_17 = 16 * A
  let total_runs_after_17 = total_runs_before_17 + 85
  let avg_after_17 = total_runs_after_17 / 17
  let new_avg = avg_after_17 + 2
  let total_runs_needed = new_avg * 18
  let runs_needed_18th = total_runs_needed - total_runs_after_17
  needs_to_score = runs_needed_18th
  sorry

end batsman_runs_needed_l113_113960


namespace maximize_profit_l113_113130

def revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - 0.5 * x^2
  else if x > 400 then 80000
  else 0 -- This ensures the piecewise function is well-defined for all x ∈ ℝ

def cost (x : ℝ) : ℝ := 20000 + 100 * x

def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 300 * x - 0.5 * x^2 - 20000
  else if x > 400 then 60000 - 100 * x
  else 0 -- This ensures the piecewise function is well-defined for all x ∈ ℝ

theorem maximize_profit :
  ∃ x : ℝ, profit 300 = 25000 ∧ (∀ y : ℝ, 0 ≤ y → y ≤ 400 → profit y ≤ profit 300) ∧ (∀ z : ℝ, z > 400 → profit z ≤ profit 300) :=
by
  sorry

end maximize_profit_l113_113130


namespace alice_speed_exceed_l113_113429

theorem alice_speed_exceed (d : ℝ) (t₁ t₂ : ℝ) (t₃ : ℝ) :
  d = 220 →
  t₁ = 220 / 40 →
  t₂ = t₁ - 0.5 →
  t₃ = 220 / t₂ →
  t₃ = 44 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_speed_exceed_l113_113429


namespace bread_slices_remaining_l113_113015

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end bread_slices_remaining_l113_113015


namespace color_count_l113_113184

-- Defining the set of numbers
def numbers := {2, 3, 4, 5, 6, 7, 8, 9}

-- Defining the set of colors
inductive Color
| red
| green
| blue

open Color

-- Defining the condition that each number and its factors are colored differently
def valid_coloring (coloring : ℕ → Color) : Prop :=
  ∀ (n : ℕ), n ∈ numbers → ∀ (m : ℕ), m ∈ numbers → m < n → n % m = 0 → coloring n ≠ coloring m

-- Statement of the problem in Lean
theorem color_count :
  ∃ (f : ℕ → Color), valid_coloring f ∧ (card (set_of (λ (coloring : ℕ → Color), valid_coloring coloring)) = 432) :=
sorry

end color_count_l113_113184


namespace vector_dot_product_proof_l113_113939

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (2, 0, 3)
def vector_c : ℝ × ℝ × ℝ := (3, 4, 2)

noncomputable def vector_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem vector_dot_product_proof : dot_product vector_a (vector_add vector_b vector_c) = 3 :=
by
  sorry

end vector_dot_product_proof_l113_113939


namespace greatest_monthly_drop_l113_113721

-- Definition of monthly price changes
def price_change_jan : ℝ := -1.00
def price_change_feb : ℝ := 2.50
def price_change_mar : ℝ := 0.00
def price_change_apr : ℝ := -3.00
def price_change_may : ℝ := -1.50
def price_change_jun : ℝ := 1.00

-- Proving the month with the greatest monthly drop in price
theorem greatest_monthly_drop :
  (price_change_apr < price_change_jan) ∧
  (price_change_apr < price_change_feb) ∧
  (price_change_apr < price_change_mar) ∧
  (price_change_apr < price_change_may) ∧
  (price_change_apr < price_change_jun) :=
by
  sorry

end greatest_monthly_drop_l113_113721


namespace profit_without_discount_l113_113423

theorem profit_without_discount
  (CP SP_with_discount : ℝ) 
  (H1 : CP = 100) -- Assume cost price is 100
  (H2 : SP_with_discount = CP + 0.216 * CP) -- Selling price with discount
  (H3 : SP_with_discount = 0.95 * SP_without_discount) -- SP with discount is 95% of SP without discount
  : (SP_without_discount - CP) / CP * 100 = 28 := 
by
  -- proof goes here
  sorry

end profit_without_discount_l113_113423


namespace faction_with_more_liars_than_truth_tellers_l113_113334

theorem faction_with_more_liars_than_truth_tellers 
  (r1 r2 r3 l1 l2 l3 : ℕ) 
  (H1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016)
  (H2 : r1 + l2 + l3 = 1208)
  (H3 : r2 + l1 + l3 = 908)
  (H4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end faction_with_more_liars_than_truth_tellers_l113_113334


namespace students_in_both_competitions_l113_113956

variable (TotalStudents MathParticipants PhysicsParticipants NoParticipants BothParticipants : ℕ)

theorem students_in_both_competitions :
  TotalStudents = 37 ∧ MathParticipants = 30 ∧ PhysicsParticipants = 20 ∧ NoParticipants = 4 →
  MathParticipants + PhysicsParticipants + NoParticipants - BothParticipants = TotalStudents →
  BothParticipants = 17 := 
by
  intro h1 h2
  have h3 : 30 + 20 + 4 - BothParticipants = 37 := h2
  have h4 : 54 - BothParticipants = 37 := by rwa [show 30 + 20 + 4 = 54 by norm_num]
  have h5 : BothParticipants = 17 := by linarith
  exact h5

end students_in_both_competitions_l113_113956


namespace typist_original_salary_l113_113727

theorem typist_original_salary (S : ℝ) :
  (1.10 * S * 0.95 * 1.07 * 0.97 = 2090) → (S = 2090 / (1.10 * 0.95 * 1.07 * 0.97)) :=
by
  intro h
  sorry

end typist_original_salary_l113_113727


namespace complex_number_in_first_quadrant_l113_113904

theorem complex_number_in_first_quadrant :
  let z := (2 : ℂ) - (1 : ℂ) * complex.I * ((1 : ℂ) + (2 : ℂ) * complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  let z := (2 : ℂ) - (1 : ℂ) * complex.I * ((1 : ℂ) + (2 : ℂ) * complex.I)
  sorry

end complex_number_in_first_quadrant_l113_113904


namespace marcia_blouses_l113_113669

def cost_skirts (n_skirts : Nat) (price_skirt : Nat) : Nat := n_skirts * price_skirt

def cost_pants (first_pair : Nat) (discount_rate : Rat) : Nat := 
  first_pair + (first_pair * discount_rate.denom / discount_rate.num)

def remaining_budget (total_budget : Nat) (cost_skirts : Nat) (cost_pants : Nat) : Nat := 
  total_budget - (cost_skirts + cost_pants)

def num_blouses (remaining_budget : Nat) (price_blouse : Nat) : Nat := 
  remaining_budget / price_blouse

theorem marcia_blouses : 
  cost_skirts 3 20 + 
  cost_pants 30 (1/2) +
  num_blouses (remaining_budget 180 (cost_skirts 3 20) (cost_pants 30 (1/2))) 15 = 180 :=
by 
  sorry 

end marcia_blouses_l113_113669


namespace price_of_each_apple_l113_113448

-- Define the constants and conditions
def price_banana : ℝ := 0.60
def total_fruits : ℕ := 9
def total_cost : ℝ := 5.60

-- Declare the variables for number of apples and price of apples
variables (A : ℝ) (x y : ℕ)

-- Define the conditions in Lean
axiom h1 : x + y = total_fruits
axiom h2 : A * x + price_banana * y = total_cost

-- Prove that the price of each apple is $0.80
theorem price_of_each_apple : A = 0.80 :=
by sorry

end price_of_each_apple_l113_113448


namespace proof_inverse_prop1_proof_negation_prop1_proof_inverse_prop2_proof_negation_prop3_proof_contrapositive_prop3_proof_inverse_prop4_l113_113105

-- Problem Setup
variable (A B : Set α)
variable (nat_divisible_by_six : ℕ → Prop) (nat_divisible_by_two : ℕ → Prop)
variable (a : ℝ)
variable (x : ℝ)

-- Definitions of Propositions
def prop1 : Prop := ∀ (x : α), x ∈ (A ∪ B) → x ∈ B
def inverse_prop1 : Prop := ∀ (x : α), x ∈ B → x ∈ (A ∪ B)
def negation_prop1 : Prop := ∀ (x : α), x ∉ (A ∪ B) → x ∉ B

def prop2 : Prop := ∀ (n : ℕ), nat_divisible_by_six n → nat_divisible_by_two n
def inverse_prop2 : Prop := ∀ (n : ℕ), nat_divisible_by_two n → nat_divisible_by_six n

def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 5 → |x - 2| < 3
def negation_prop3 : Prop := ∀ (x : ℝ), ¬ (0 < x ∧ x < 5) → |x - 2| ≥ 3
def contrapositive_prop3 : Prop := ∀ (x : ℝ), |x - 2| ≥ 3 → ¬ (0 < x ∧ x < 5)

def prop4 : Prop := ∀ (x : ℝ), (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0 → a ∈ (-2 : ℝ, 2)
def inverse_prop4 : Prop := ∀ (a : ℝ), a ∈ (-2 : ℝ, 2) → ∀ (x : ℝ), (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Proof Statements
theorem proof_inverse_prop1 : inverse_prop1 A B := sorry
theorem proof_negation_prop1 : negation_prop1 A B := sorry
theorem proof_inverse_prop2 : ¬ inverse_prop2 nat_divisible_by_six nat_divisible_by_two := sorry
theorem proof_negation_prop3 : ¬ negation_prop3 := sorry
theorem proof_contrapositive_prop3 : contrapositive_prop3 := sorry
theorem proof_inverse_prop4 : inverse_prop4 := sorry

end proof_inverse_prop1_proof_negation_prop1_proof_inverse_prop2_proof_negation_prop3_proof_contrapositive_prop3_proof_inverse_prop4_l113_113105


namespace three_numbers_can_be_written_as_difference_of_primes_l113_113935

/-
  Define the function that produces the sequence {10n - 1 | n ≥ 1}
-/
def sequence (n : ℕ) : ℕ := 10 * n + 9

/-
  Define the set S of these numbers {9, 19, 29, 39, ...} up to a certain term k
-/
def setS (k : ℕ) : finset ℕ := (finset.range k).image sequence

/-
  The property that the number can be written as the difference of two primes
-/
def can_be_written_as_difference_of_primes (x : ℕ) : Prop :=
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p ≠ q ∧ (p - q = x ∨ q - p = x)

/-
  Main statement:
  There are exactly three numbers in the set that can be written as the difference of two primes where one of the primes is 2
-/
theorem three_numbers_can_be_written_as_difference_of_primes :
  ∃ (k : nat), k ∈ {9, 19, 29, 39} ∧ can_be_written_as_difference_of_primes k :=
begin
  sorry
end

end three_numbers_can_be_written_as_difference_of_primes_l113_113935


namespace crushing_load_example_l113_113201

variable (T H : ℝ)

noncomputable def L (T H : ℝ) : ℝ := (30 * T^3) / H

theorem crushing_load_example (hT : T = 5) (hH : H = 10) : L T H = 375 :=
by
  rw [hT, hH]
  have eq1 : L 5 10 = (30 * 5^3) / 10 := by rfl
  have eq2 : (30 * 5^3) / 10 = (30 * 125) / 10 := by norm_num
  have eq3 : (30 * 125) / 10 = 375 := by norm_num
  rw [eq1, eq2, eq3]
  exact rfl

end crushing_load_example_l113_113201


namespace sum_of_products_lt_zero_l113_113315

theorem sum_of_products_lt_zero (a b c d e f : ℤ) (h : ∃ (i : ℕ), i ≤ 6 ∧ i ≠ 6 ∧ (∀ i ∈ [a, b, c, d, e, f], i < 0 → i ≤ i)) :
  ab + cdef < 0 :=
sorry

end sum_of_products_lt_zero_l113_113315


namespace proof_example_l113_113481

variables {A B C P J : Point}
variables {p a b c r R : ℝ}
variables {PA PB PC PJ : ℝ}

-- Conditions
def incenter (ABC : Triangle) (J : Point) : Prop := -- some definition of incenter
def point_in_triangle (P : Point) (ABC : Triangle) : Prop := -- some definition of point being in triangle
def side_lengths (ABC : Triangle) : ℝ × ℝ × ℝ := -- some definition of side lengths
def semiperimeter (ABC : Triangle) : ℝ := -- some definition of semiperimeter
noncomputable def inradius (ABC : Triangle) : ℝ := -- some definition of inradius
noncomputable def circumradius (ABC : Triangle) : ℝ := -- some definition of circumradius

-- The mathematical statement to be proved
theorem proof_example (ABC : Triangle) (P : Point) (J : Point) (a b c r R : ℝ) (PA PB PC PJ : ℝ) 
  (h1 : point_in_triangle P ABC)
  (h2 : incenter ABC J)
  (h3 : (a, b, c) = side_lengths ABC)
  (h4 : semiperimeter ABC = p)
  (h5 : inradius ABC = r)
  (h6 : circumradius ABC = R)
  (h7 : PA = dist P A)
  (h8 : PB = dist P B)
  (h9 : PC = dist P C)
  (h10 : PJ = dist P J) :
  PJ^2 = (p - a) / p * PA^2 + (p - b) / p * PB^2 + (p - c) / p * PC^2 + 4 * r^2 - 4 * R * r :=
sorry

end proof_example_l113_113481


namespace angle_sine_relation_l113_113986

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) (R : ℝ) : Prop :=
  a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

theorem angle_sine_relation (A B C a b c R : ℝ) (h : triangle_ABC A B C a b c R) :
  (A > B ↔ sin A > sin B) :=
sorry

end angle_sine_relation_l113_113986


namespace cost_price_of_watch_l113_113815

theorem cost_price_of_watch
  (C : ℝ)
  (h1 : 0.9 * C + 225 = 1.05 * C) :
  C = 1500 :=
by sorry

end cost_price_of_watch_l113_113815


namespace sum_sqrt_and_kn_inequality_l113_113436

theorem sum_sqrt_and_kn_inequality (n : ℕ) (x : Fin n → ℝ) (hn : 2 ≤ n) 
  (hx_sum : ∑ i, (x i)^2 = 1) : 
  ∑ i, Real.sqrt (1 - (x i)^2) + (2 - 2 * Real.sqrt (1 + 1 / (n - 1))) * 
    ∑ (i : Fin n) (j : Fin n) (h : i.val < j.val), (x i) * (x j) ≥ n - 1 := 
by 
  sorry

end sum_sqrt_and_kn_inequality_l113_113436


namespace shaded_region_area_l113_113400

def square_side : ℝ := 6
def circle_radius : ℝ := 2 * Real.sqrt 3

-- Area of the square
def square_area := square_side ^ 2

-- Area of one right triangle
def triangle_area := 1 / 2 * (square_side / 2) * circle_radius

-- Area of one circular sector
def sector_area := (1 / 12) * Real.pi * (circle_radius ^ 2)

-- Total area to be subtracted
def subtract_area := 8 * triangle_area + 4 * sector_area

-- The area of the shaded region
def shaded_area := square_area - subtract_area

theorem shaded_region_area :
  shaded_area = 36 - 12 * Real.sqrt 3 - 4 * Real.pi :=
by
  sorry

end shaded_region_area_l113_113400


namespace customers_left_tip_l113_113152

section
variable (total_customers_breakfast total_customers_lunch total_customers_evening 
          early_morning_customers priority_customers regular_customers : ℕ)
variable (percentage_no_tip_early percentage_no_tip_priority percentage_no_tip_regular percentage_no_tip_remaining : ℚ)

def total_customers_served := total_customers_breakfast + total_customers_lunch + total_customers_evening

def no_tip_early := early_morning_customers * percentage_no_tip_early
def no_tip_priority := priority_customers * percentage_no_tip_priority
def no_tip_regular := regular_customers * percentage_no_tip_regular

def total_remaining_customers := (total_customers_breakfast - early_morning_customers) + 
                                 (total_customers_lunch - priority_customers) + 
                                 (total_customers_evening - regular_customers)

def no_tip_remaining := total_remaining_customers * percentage_no_tip_remaining

def total_no_tip := no_tip_early + no_tip_priority + no_tip_regular + no_tip_remaining
def total_tip := total_customers_served - total_no_tip

theorem customers_left_tip (h1 : total_customers_breakfast = 70)
                           (h2 : early_morning_customers = 20)
                           (h3 : total_customers_lunch = 100)
                           (h4 : priority_customers = 60)
                           (h5 : total_customers_evening = 45)
                           (h6 : regular_customers = 22)
                           (h7 : percentage_no_tip_early = 0.30)
                           (h8 : percentage_no_tip_priority = 0.60)
                           (h9 : percentage_no_tip_regular = 0.50)
                           (h10 : percentage_no_tip_remaining = 0.25) :
total_tip = 134 := sorry

end

end customers_left_tip_l113_113152


namespace apples_sold_in_1_half_hour_l113_113366

theorem apples_sold_in_1_half_hour (total_apples : ℕ) (total_hours : ℕ) (hours_passed : ℕ) (minutes_passed : ℕ)
  (h1 : total_apples = 108)
  (h2 : total_hours = 6)
  (h3 : hours_passed = 1)
  (h4 : minutes_passed = 30) :
  let rate := total_apples / total_hours in
  let time := hours_passed + (minutes_passed / 60 : ℕ) in
  rate * time = 27 := 
by 
  -- constants used in the formalization
  have rate_eq : rate = 18 :=
    by {
      calc
        rate = total_apples / total_hours    : by rfl
        ...   = 108 / 6                      : by rw [h1, h2]
        ...   = 18                          : by norm_num,
    },

  have time_eq : time = 1.5 :=
    calc 
      time = hours_passed + (minutes_passed / 60 : ℕ) : by rfl
      ...  = 1 + (30 / 60 : ℕ)                      : by { rw [h3, h4] }
      ...  = 1 + (0.5 : ℕ)                         : by norm_num,
        ...  = 1.5                               : by norm_num,

  show (rate * time) = 27,
  calc (rate * time) = 18 * 1.5 : by rw [rate_eq, time_eq]
                 ... = 27       : by norm_num

end apples_sold_in_1_half_hour_l113_113366


namespace tile_difference_correct_l113_113338

def initial_blue_tiles := 23
def initial_green_tiles := 16
def first_border_green_tiles := 6 * 1
def second_border_green_tiles := 6 * 2
def total_green_tiles := initial_green_tiles + first_border_green_tiles + second_border_green_tiles
def difference_tiling := total_green_tiles - initial_blue_tiles

theorem tile_difference_correct : difference_tiling = 11 := by
  sorry

end tile_difference_correct_l113_113338


namespace wholesale_cost_l113_113141

theorem wholesale_cost (W : ℝ) (h1 : 1.20 * W - 204) = 0.85 * (1.20 * W) = true :
  W = 200 := by sorry

end wholesale_cost_l113_113141


namespace boys_girls_relationship_l113_113164

theorem boys_girls_relationship (b g : ℕ): (4 + 2 * b = g) → (b = (g - 4) / 2) :=
by
  intros h
  sorry

end boys_girls_relationship_l113_113164


namespace rectangle_diagonals_equal_but_not_rhombus_l113_113397

-- Definitions
structure Rectangle :=
  (sides_equal : ∀ (a b : ℕ), a ≠ b → sides a = sides b → false) -- property that opposite sides are equal
  (diagonals_equal : ∀ (d1 d2 : ℕ), d1 = d2) -- property that diagonals are equal

structure Rhombus :=
  (all_sides_equal : ∀ (a b : ℕ), sides a = sides b) -- property that all sides are equal
  (diagonals_perpendicular_bisect : ∀ (d : ℕ), d ┴ d) -- property that diagonals are perpendicular and bisect each other

-- Theorem to prove
theorem rectangle_diagonals_equal_but_not_rhombus :
  ∀ (rect : Rectangle) (rhom : Rhombus), rect.diagonals_equal ∧ ¬ rhom.diagonals_equal :=
  sorry

end rectangle_diagonals_equal_but_not_rhombus_l113_113397


namespace lucille_total_revenue_l113_113009

theorem lucille_total_revenue (salary_ratio stock_ratio : ℕ) (salary_amount : ℝ) (h_ratio : salary_ratio / stock_ratio = 4 / 11) (h_salary : salary_amount = 800) : 
  ∃ total_revenue : ℝ, total_revenue = 3000 :=
by
  sorry

end lucille_total_revenue_l113_113009


namespace max_log_sum_l113_113950

theorem max_log_sum (a b : ℝ) (h_positive : 0 < a ∧ 0 < b) (h_sum : a + b = 4) : 
  log 2 a + log 2 b ≤ 2 :=
by
  sorry

end max_log_sum_l113_113950


namespace curve_equation_l113_113542

theorem curve_equation :
  (F : ℝ × ℝ) (M : ℝ × ℝ) (P : ℝ × ℝ) 
  (hF : F = (0,1)) 
  (hl : ∃ x, M = (x,-1))
  (hP : ∃ x y, P = (x, y) ∧ ∃ x_M, M = (x_M, -1) ∧ (sqrt (x^2 + (y - 1)^2) = abs (y + 1))) :
  P.1^2 = 4 * P.2 :=
sorry

end curve_equation_l113_113542


namespace part_one_part_two_l113_113496

section ellipse_properties

variables {a b : ℝ} (h1 : a > b > 0)

def is_ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_line (x y : ℝ) : Prop :=
  x + y = 1

def orthogonal (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem part_one (h : ∃ P Q : ℝ × ℝ, is_ellipse P.1 P.2 ∧ is_ellipse Q.1 Q.2 ∧ is_line P.1 P.2 ∧ is_line Q.1 Q.2 ∧ orthogonal P.1 P.2 Q.1 Q.2) :
  (1 / a^2) + (1 / b^2) = 2 := sorry

theorem part_two (h2 : (sqrt 3 / 3) ≤ sqrt (1 - b^2 / a^2) ∧ sqrt (1 - b^2 / a^2) ≤ sqrt 2 / 2) :
  ∃ (a_val : ℝ), (a = a_val) ∧ (sqrt 5 / 2) ≤ a_val ∧ a_val ≤ (sqrt 6 / 2) :=
sorry

end ellipse_properties

end part_one_part_two_l113_113496


namespace polynomial_division_correct_l113_113224

def p (x : ℝ) := x^3 + 5*x^2 + 3*x + 9
def d (x : ℝ) := x - 2
def q (x : ℝ) := x^2 + 7*x + 17
def r : ℝ := 43

theorem polynomial_division_correct : ∀ x : ℝ, p(x) = d(x) * q(x) + r := by
  -- The proof is omitted
  sorry

end polynomial_division_correct_l113_113224


namespace x_coord_sum_l113_113367

noncomputable def sum_x_coordinates (x : ℕ) : Prop :=
  (0 ≤ x ∧ x < 20) ∧ (∃ y, y ≡ 7 * x + 3 [MOD 20] ∧ y ≡ 13 * x + 18 [MOD 20])

theorem x_coord_sum : ∃ (x : ℕ), sum_x_coordinates x ∧ x = 15 := by 
  sorry

end x_coord_sum_l113_113367


namespace asymptote_equation_l113_113241

variables {a b c x y : ℝ}
variables (C : a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1))
variables (area_triangle_MF1F2 : 4 * a^2)

theorem asymptote_equation : (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1) ∧ (4 * a^2) → (y = ± 4 * x) :=
by
  intro C area_triangle_MF1F2
  sorry

end asymptote_equation_l113_113241


namespace two_F_cover_circle_l113_113455

-- Definitions for the problem
variable (R : ℝ) (F : Type)
variable [ConvexShape F]
variable (segment : Circle R → Circle R → F)

-- Hypothesis that a convex shape F cannot cover a semicircle of radius R
hypothesis (h_convex_semi : ¬(∃ f : F, covers_semicircle f R))

-- Theorem statement
theorem two_F_cover_circle : 
  (∃ (f1 f2 : F), covers_circle (f1, f2) R) :=
sorry

end two_F_cover_circle_l113_113455


namespace shaded_area_semicircles_l113_113975

open Real

/-- Points A, B, C, D, E, F, G lie on a straight line where AB = BC = CD = DE = EF = FG = 10.
    Semicircles with diameters AG, AB, BC, CD, DE, EF, and FG are drawn.
    Prove that the area of the shaded region created by these semicircles is 500π. -/
theorem shaded_area_semicircles : 
    let d := 10 in
    let r := d / 2 in
    let area_semicircle_d (d : ℝ) := (π * (d / 2)^2) / 2 in
    let area_lg_semicircle := (π * (60 / 2)^2) / 2 in
    (area_lg_semicircle - 2 * area_semicircle_d d + 3 * area_semicircle_d d - area_semicircle_d d + area_semicircle_d d) = 500 * π :=
by
  let d := 10
  let r := d / 2
  let area_semicircle_d := fun d : ℝ => (π * (d / 2)^2) / 2
  let area_lg_semicircle := (π * (60 / 2)^2) / 2
  sorry

end shaded_area_semicircles_l113_113975


namespace expression_value_l113_113099

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 :=
by
  rw [hx, hy, hz]
  -- After substituting the values, we will need to simplify the expression to reach 20.
  sorry

end expression_value_l113_113099


namespace smallest_prime_sum_l113_113871

theorem smallest_prime_sum (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (H1 : Prime (a + b + c + d))
  (H2 : Prime (a + b)) (H3 : Prime (a + c)) (H4 : Prime (a + d)) (H5 : Prime (b + c)) (H6 : Prime (b + d)) (H7 : Prime (c + d))
  (H8 : Prime (a + b + c)) (H9 : Prime (a + b + d)) (H10 : Prime (a + c + d)) (H11 : Prime (b + c + d))
  : a + b + c + d = 31 :=
sorry

end smallest_prime_sum_l113_113871


namespace value_of_f_m_plus_one_l113_113941

variable (a m : ℝ)

def f (x : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one 
  (h : f a (-m) < 0) : f a (m + 1) < 0 := by
  sorry

end value_of_f_m_plus_one_l113_113941


namespace find_smallest_k_l113_113454

def colombian_configuration (points : Finset Point) : Prop :=
  points.card = 4027 ∧
  (∃ reds blues : Finset Point,
    reds.card = 2013 ∧
    blues.card = 2014 ∧
    points = reds ∪ blues ∧
    ∀ p q r : Point, p ∈ points → q ∈ points → r ∈ points → 
      p ≠ q → q ≠ r → p ≠ r → 
      ¬ collinear p q r)

def good_drawing (k : ℕ) (points : Finset Point) (lines : Finset Line) : Prop :=
  (∀ l : Line, l ∈ lines → ¬ ∃ p : Point, p ∈ points ∧ p ∈ l) ∧
  (∀ reg : Region,
    (reg.part_of_plane lines ∧ ∃ p q : Point, p ∈ points → q ∈ points → 
      (collinear p q r) → (p ∈ reg ∧ q ∈ reg ∧ r ∈ reg)))

theorem find_smallest_k :
  ∀ points : Finset Point,
  colombian_configuration points →
  ∃ k : ℕ, (∀ lines : Finset Line, good_drawing k points lines) ↔ k = 2013 :=
sorry

end find_smallest_k_l113_113454


namespace plane_divides_KP_l113_113625

-- Define points in the tetrahedron
variables (A B C D M K P N : Type) [AffineSpace ℝ A] [AffineSpace ℝ B]
          [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ M]
          [AffineSpace ℝ K] [AffineSpace ℝ P] [AffineSpace ℝ N]

-- Conditions of the problem
axiom midpoint_AD (hM : midpoint ℝ A D M) : AffineSpace.span ℝ {A, D}
axiom point_BD (hN : ratio_point ℝ B D N (2 / 3)) : AffineSpace.span ℝ {B, D}
axiom midpoint_AB (hK : midpoint ℝ A B K) : AffineSpace.span ℝ {A, B}
axiom midpoint_CD (hP : midpoint ℝ C D P) : AffineSpace.span ℝ {C, D}

-- Statement of the proof problem
theorem plane_divides_KP (hPlane : AffineSpace.span ℝ {M, C, N}) :
  divides_ratio (segment K P) (segment P : ratio 1 3) :=
sorry

end plane_divides_KP_l113_113625


namespace find_a_for_min_l113_113277

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 - 6 * a * x + 2

theorem find_a_for_min {a x0 : ℝ} (hx0 : 1 < x0 ∧ x0 < 3) (h : ∀ x : ℝ, deriv (f a) x0 = 0) : a = -2 :=
by
  sorry

end find_a_for_min_l113_113277


namespace correct_operation_l113_113104

-- Define the operations as predicates
def op_A : Prop := sqrt 3 + 2 * sqrt 3 = 2 * sqrt 6
def op_B (a : ℝ) : Prop := (-a^2)^3 = a^6
def op_C (a : ℝ) : Prop := (1 / (2 * a)) + (1 / a) = (2 / (3 * a))
def op_D (a b : ℝ) : Prop := (1 / (3 * a * b)) / (b / (3 * a)) = 1 / (b^2)

-- Theorem stating option D is the correct one
theorem correct_operation (a b : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) : 
  ¬ op_A ∧ ¬ op_B a ∧ ¬ op_C a ∧ op_D a b :=
by
  sorry

end correct_operation_l113_113104


namespace hexagon_side_lengths_l113_113502

open Nat

/-- Define two sides AB and BC of a hexagon with their given lengths -/
structure Hexagon :=
  (AB BC AD BE CF DE: ℕ)
  (distinct_lengths : AB ≠ BC ∧ (AB = 7 ∧ BC = 8))
  (total_perimeter : AB + BC + AD + BE + CF + DE = 46)

-- Define a theorem to prove the number of sides measuring 8 units
theorem hexagon_side_lengths (h: Hexagon) :
  ∃ (n : ℕ), n = 4 ∧ n * 8 + (6 - n) * 7 = 46 :=
by
  -- Assume the proof here
  sorry

end hexagon_side_lengths_l113_113502


namespace find_f_log2_3_l113_113606

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = (1 / 3)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = (1 / 2) :=
by
  sorry

end find_f_log2_3_l113_113606


namespace largest_power_of_two_divides_a2013_l113_113038

noncomputable def a_2013 (n : ℕ) : ℤ := -1007 * 2013 * Nat.factorial 2013

noncomputable def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  Nat.factorial.find_greatest_pow 2 n

theorem largest_power_of_two_divides_a2013 :
  largest_power_of_two_dividing 2013 = 2004 :=
by
  sorry

end largest_power_of_two_divides_a2013_l113_113038


namespace inequality_convex_l113_113679

theorem inequality_convex (x y a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : a + b = 1) : 
  (a * x + b * y) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 := 
sorry

end inequality_convex_l113_113679


namespace water_distribution_l113_113078

theorem water_distribution :
  ∀ (total_players : ℕ) (initial_water_l : ℕ) (spilled_water_ml left_over_water_ml : ℕ), 
  total_players = 30 →
  initial_water_l = 8 →
  spilled_water_ml = 250 →
  left_over_water_ml = 1750 →
  let initial_water_ml := initial_water_l * 1000
  let distributed_water_ml := initial_water_ml - spilled_water_ml - left_over_water_ml
  (distributed_water_ml / total_players) = 200 :=
by
  intros total_players initial_water_l spilled_water_ml left_over_water_ml 
  intro h_tp intro h_iw intro h_sw intro h_lw
  let initial_water_ml := initial_water_l * 1000
  let distributed_water_ml := initial_water_ml - spilled_water_ml - left_over_water_ml
  show (distributed_water_ml / total_players) = 200
  sorry

end water_distribution_l113_113078


namespace solution_set_l113_113663

variable {f : ℝ → ℝ}

/-- Given a convex function f defined on 0 ≤ x ≤ 1 -/
def convex_on_interval (f : ℝ → ℝ) (a b : ℝ) [convex ℝ (set.Icc a b)] :=
∀ ⦃x1 x2⦄ (hx1 : a ≤ x1) (hx1₂ : x1 ≤ b) (hx2 : a ≤ x2) (hx2₃ : x2 ≤ b) (λ : ℝ) 
  (hλ0 : 0 ≤ λ) (hλ1 : λ ≤ 1),
  f(λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2

/-- Prove the locus of points P(u, v) with 0 ≤ v ≤ 2 -/
theorem solution_set {u v : ℝ} (hf : convex_on_interval f 0 1):
  ( v ∈ set.Icc 0 2 ∧ ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ a + b = v ∧ f a + f b = u ) ↔
  ∃ (x y : ℝ), f (sin^2 x) + f (cos^2 y) = u ∧ sin^2 x + cos^2 y = v :=
sorry

end solution_set_l113_113663


namespace integral_rational_function_l113_113519

theorem integral_rational_function :
  ∫ (x^3 + 6 * x^2 + 8 * x + 8) / ((x + 2) ^ 2 * (x^2 + 4)) dx
  = - (1 / (x + 2)) + (1 / 2) * log (abs (x^2 + 4)) + (1 / 2) * arctan (x / 2) + C :=
sorry

end integral_rational_function_l113_113519


namespace count_convex_m_gons_formula_correct_l113_113443

noncomputable def count_convex_m_gons (m n : ℕ) (hm : 5 ≤ m) : ℕ :=
  let term1 := (2 * n + 1) * ( Nat.choose n (m - 1) + n * Nat.choose n (m - 2) )
  let term2 := (2 * n + 1) * (Finset.sum (Finset.range (n - 1)) (λ x, 
               Finset.sum (Finset.range (2 * n + 1 - (x + n + 1))) (λ y,
               (y - x - 1) * Nat.choose (2 * n + 1 - y) (m - 4))))
  term1 - term2

theorem count_convex_m_gons_formula_correct (M : Finset (Euclidean 2)) (m n : ℕ) (hm : 5 ≤ m) :
  count_convex_m_gons m n hm = 
  (2 * n + 1) * ( Nat.choose n (m - 1) + n * Nat.choose n (m - 2) ) -
  (2 * n + 1) * (Finset.sum (Finset.range (n - 1)) (λ x, 
                Finset.sum (Finset.range (2 * n + 1 - (x + n + 1))) (λ y, 
                (y - x - 1) * Nat.choose (2 * n + 1 - y) (m - 4)))) :=
sorry

end count_convex_m_gons_formula_correct_l113_113443


namespace quadratic_polynomials_min_distinct_roots_l113_113375

theorem quadratic_polynomials_min_distinct_roots (P Q : ℝ → ℝ)
  (hP : ∃ a b c ∈ ℝ, P = λ x, a*x^2 + b*x + c)   -- P(x) is a quadratic polynomial
  (hQ : ∃ d e f ∈ ℝ, Q = λ x, d*x^2 + e*x + f)   -- Q(x) is a quadratic polynomial
  (h_roots_distinct : ∃ x₁ x₂ x₃ x₄ ∈ ℝ, P(x₁) = 0 ∧ P(x₂) = 0 ∧ Q(x₃) = 0 ∧ Q(x₄) = 0 ∧ x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄)
  (h_PQ_roots : ∃ y₁ y₂ y₃ y₄ ∈ ℝ, P(Q(y₁)) = 0 ∧ P(Q(y₂)) = 0 ∧ P(Q(y₃)) = 0 ∧ P(Q(y₄)) = 0 ∧ y₁ ≠ y₂ ∧ y₃ ≠ y₄ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄)
  (h_QP_roots : ∃ z₁ z₂ z₃ z₄ ∈ ℝ, Q(P(z₁)) = 0 ∧ Q(P(z₂)) = 0 ∧ Q(P(z₃)) = 0 ∧ Q(P(z₄)) = 0 ∧ z₁ ≠ z₂ ∧ z₃ ≠ z₄ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄) :
  ∃ n ∈ ℕ, n = 6 ∧ (∀ r ∈ ℝ, (P(r) = 0 ∨ Q(r) = 0 ∨ P(Q(r)) = 0 ∨ Q(P(r)) = 0) → r ∈ (∑ i in range n, singleton_set i ∩ ℝ)) :=
  sorry

end quadratic_polynomials_min_distinct_roots_l113_113375


namespace four_not_divides_n_l113_113037

theorem four_not_divides_n (n : ℕ) (x : ℕ → ℤ) 
  (hx : ∀ i, 1 ≤ i ∧ i ≤ n → (x i = 1 ∨ x i = -1))
  (h : (∑ i in finset.range n, x i * x (i + 1)) + x n * x 1 = 0) : 
  ¬ 4 ∣ n :=
by sorry

end four_not_divides_n_l113_113037


namespace intersect_A_B_l113_113259

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l113_113259


namespace pairs_of_ping_pong_rackets_sold_l113_113147

theorem pairs_of_ping_pong_rackets_sold : 
  ∀ (total_amount : ℝ) (average_price : ℝ), 
  total_amount = 735 → average_price = 9.8 → (total_amount / average_price = 75) :=
by
  intros total_amount average_price ht ha
  rw [ht, ha]
  norm_num
  sorry

end pairs_of_ping_pong_rackets_sold_l113_113147


namespace probability_spinner_lands_in_shaded_region_l113_113820

theorem probability_spinner_lands_in_shaded_region :
  ∀ (total_regions shaded_regions : ℕ)
  (h_total : total_regions = 6)
  (h_shaded : shaded_regions = 3),
  (shaded_regions : ℚ) / (total_regions : ℚ) = 1 / 2 :=
by
  intros total_regions shaded_regions h_total h_shaded
  rw [h_total, h_shaded]
  norm_num
  sorry

end probability_spinner_lands_in_shaded_region_l113_113820


namespace calorie_count_l113_113089

/-- Constants representing the essential variables based on the problem conditions --/
constant C : ℤ
constant carrot_calories : ℤ := C
constant broccoli_calories : ℤ := C / 3
constant total_carrot_pounds : ℤ := 1
constant total_broccoli_pounds : ℤ := 2
constant total_calories : ℤ := 85

/-- Statement representing the total calories consumed from carrots and broccoli --/
theorem calorie_count :
  C + 2 * (C / 3) = 85 →
  C = 51 :=
by
  intro h
  sorry

end calorie_count_l113_113089


namespace bill_milk_problem_l113_113826

theorem bill_milk_problem 
  (M : ℚ) 
  (sour_cream_milk : ℚ := M / 4)
  (butter_milk : ℚ := M / 4)
  (whole_milk : ℚ := M / 2)
  (sour_cream_gallons : ℚ := sour_cream_milk / 2)
  (butter_gallons : ℚ := butter_milk / 4)
  (butter_revenue : ℚ := butter_gallons * 5)
  (sour_cream_revenue : ℚ := sour_cream_gallons * 6)
  (whole_milk_revenue : ℚ := whole_milk * 3)
  (total_revenue : ℚ := butter_revenue + sour_cream_revenue + whole_milk_revenue)
  (h : total_revenue = 41) :
  M = 16 :=
by
  sorry

end bill_milk_problem_l113_113826


namespace area_of_rectangle_l113_113326

variables {A B C D : Type*} [linear_ordered_field A] [metric_space A] [cosine_space A]
variables {AB BC AD : A} (ha : 90 = 90) (hb : AB = 4) (hc : BC = 5) (hd : AD = 3)

theorem area_of_rectangle (hape_eq : ha ∧ hd) : 
  (AD * BC = (3 : A) * (5 : A)) := 
begin
  sorry
end

end area_of_rectangle_l113_113326


namespace function_property_l113_113574

-- Definitions and conditions
def f (a b x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

noncomputable def y (a b x : ℝ) : ℝ := f a b (x + Real.pi / 4)

-- Property definitions
def is_even (g : ℝ → ℝ) := ∀ x : ℝ, g x = g (-x)
def is_symmetric_about (g : ℝ → ℝ) (p : ℝ × ℝ) := 
  ∀ x : ℝ, g x = 2 * p.2 - g (2 * p.1 - x)

-- The statement to prove
theorem function_property (a b : ℝ) (h : a ≠ 0) (h_max : ∀ x, f a b x ≤ f a b (Real.pi / 4)) :
  is_even (y a b) ∧ is_symmetric_about (y a b) (3 * Real.pi / 2, 0) :=
by
  sorry

end function_property_l113_113574


namespace tangent_y_axis_tangent_both_axes_l113_113274

variables (a b r : ℝ)
#check (x : ℝ) (y : ℝ)

-- Conditions: r > 0, Circle equation
def circle_eqn (x y : ℝ) := (x - a)^2 + (y - b)^2 = r^2 

-- Theorem for condition 1: Circle tangent to the y-axis implies |a| = r
theorem tangent_y_axis (h_circle : ∀ x y, circle_eqn a b r x y → r > 0) (tangent_y : ∃ x, (x - a)^2 = r^2 ∧ (0 - b)^2 = 0) : |a| = r :=
sorry

-- Theorem for condition 2: Circle tangent to both coordinate axes implies |a| = |b| = r
theorem tangent_both_axes (h_circle : ∀ x y, circle_eqn a b r x y → r > 0) (tangent_both : ∃ (x : ℝ), (x - a)^2 = r^2 ∧ a = r ∧ b = r) : |a| = |b| = r :=
sorry

end tangent_y_axis_tangent_both_axes_l113_113274


namespace journey_length_l113_113511

/-- Define the speed in the urban area as 55 km/h. -/
def urban_speed : ℕ := 55

/-- Define the speed on the highway as 85 km/h. -/
def highway_speed : ℕ := 85

/-- Define the time spent in each area as 3 hours. -/
def travel_time : ℕ := 3

/-- Define the distance traveled in the urban area as the product of the speed and time. -/
def urban_distance : ℕ := urban_speed * travel_time

/-- Define the distance traveled on the highway as the product of the speed and time. -/
def highway_distance : ℕ := highway_speed * travel_time

/-- Define the total distance of the journey. -/
def total_distance : ℕ := urban_distance + highway_distance

/-- The theorem that the total distance is 420 km. -/
theorem journey_length : total_distance = 420 := by
  -- Prove the equality by calculating the distances and summing them up
  sorry

end journey_length_l113_113511


namespace simplify_expression_l113_113380

variable (α : ℝ)

-- Conditions: Trigonometric identities used in the solution
lemma cos_2x_identity (x : ℝ) : cos (2 * x) = 2 * (cos x)^2 - 1 :=
by sorry

lemma cos_squared_half_x_identity (x : ℝ) : (cos (x / 2))^2 = (1 + cos x) / 2 :=
by sorry

-- Theorem statement
theorem simplify_expression : 3 - 4 * cos (4 * α) + cos (8 * α) - 8 * (cos (2 * α))^4 = -8 * cos (4 * α) :=
by
  sorry

end simplify_expression_l113_113380


namespace problem_l113_113535

def f (x : ℝ) : ℝ := x^2 + 3 * x * (deriv (f) 2)

theorem problem (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 3 * x * deriv f 2) : 
  1 + deriv f 1 = -3 :=
by
  have h_deriv : ∀ x, deriv f x = 2 * x + 3 * deriv f 2 := 
    by { intro x, rw h x, simp, }
  have f_prime_2_eq : deriv f 2 = -2 := 
    by { rw h_deriv 2, linarith, }
  have f_prime_1_eq : deriv f 1 = -4 := 
    by { rw h_deriv 1, rw f_prime_2_eq, linarith, }
  show 1 + deriv f 1 = -3,
  rw f_prime_1_eq,
  linarith

end problem_l113_113535


namespace smallest_k_divisible_by_200_l113_113644

theorem smallest_k_divisible_by_200 : 
  ∃ (k : ℕ), (∑ i in finset.range k.succ, i^2) = 112 :=
by
  let sum_squares := λ k : ℕ, k * (k + 1) * (2 * k + 1) / 6
  have h: ∃ k, sum_squares k % 200 = 0 := sorry
  exact h

end smallest_k_divisible_by_200_l113_113644


namespace compare_f_values_l113_113003

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (h_pos : 0 < a) :
  (a > 2 * Real.sqrt 2 → f a > f (a / 2) * f (a / 2)) ∧
  (a = 2 * Real.sqrt 2 → f a = f (a / 2) * f (a / 2)) ∧
  (0 < a ∧ a < 2 * Real.sqrt 2 → f a < f (a / 2) * f (a / 2)) :=
by
  sorry

end compare_f_values_l113_113003


namespace max_sum_l113_113845

-- Let's define the six numbers.
variables (a b c d e f : ℕ)
variables (h1 : {a, b, c, d, e, f} = {2, 5, 8, 11, 14, 17})

-- Conditions
def rows_equal := a + b + c = d + e + f
def col1_equal := a + d
def col2_equal := b + e
def col3_equal := c + f
def cols_equal := col1_equal = col2_equal ∧ col2_equal = col3_equal

-- Theorem statement
theorem max_sum := ∃ a b c d e f : ℕ, {a, b, c, d, e, f} = {2, 5, 8, 11, 14, 17} ∧ rows_equal a b c d e f ∧ cols_equal a b c d e f → a + b + c = 33

end max_sum_l113_113845


namespace Tom_age_ratio_l113_113090

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio_l113_113090


namespace decodeMINT_l113_113068

def charToDigit (c : Char) : Option Nat :=
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'M' => some 5
  | 'I' => some 6
  | 'N' => some 7
  | 'D' => some 8
  | 'S' => some 9
  | _   => none

def decodeWord (word : String) : Option Nat :=
  let digitsOption := word.toList.map charToDigit
  if digitsOption.all Option.isSome then
    let digits := digitsOption.map Option.get!
    some (digits.foldl (λ acc d => 10 * acc + d) 0)
  else
    none

theorem decodeMINT : decodeWord "MINT" = some 5674 := by
  sorry

end decodeMINT_l113_113068


namespace minimum_fraction_l113_113880

theorem minimum_fraction (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + 2 * n = 8) : 2 / m + 1 / n = 1 :=
by
  sorry

end minimum_fraction_l113_113880


namespace square_of_99_l113_113428

theorem square_of_99 : 99 * 99 = 9801 :=
by sorry

end square_of_99_l113_113428


namespace micah_total_envelopes_l113_113672

-- Define the conditions as hypotheses
def weight_threshold := 5
def stamps_for_heavy := 5
def stamps_for_light := 2
def total_stamps := 52
def light_envelopes := 6

-- Noncomputable because we are using abstract reasoning rather than computational functions
noncomputable def total_envelopes : ℕ :=
  light_envelopes + (total_stamps - light_envelopes * stamps_for_light) / stamps_for_heavy

-- The theorem to prove
theorem micah_total_envelopes : total_envelopes = 14 := by
  sorry

end micah_total_envelopes_l113_113672


namespace area_is_300_l113_113724

variable (l w : ℝ) -- Length and Width of the playground

-- Conditions
def condition1 : Prop := 2 * l + 2 * w = 80
def condition2 : Prop := l = 3 * w

-- Question and Answer
def area_of_playground : ℝ := l * w

theorem area_is_300 (h1 : condition1 l w) (h2 : condition2 l w) : area_of_playground l w = 300 := 
by
  sorry

end area_is_300_l113_113724


namespace monotonicity_of_f1_g_above_f2_inequality_proof_l113_113573

-- Problem (1)
def f1 (x: ℝ) : ℝ := - (3 / 2) * log x + (1 / 2) * x - (1 / x)

theorem monotonicity_of_f1 :
  (∀ x, 0 < x ∧ x < 1 → increasing_on f1 (Ioo 0 1)) ∧
  (∀ x, 1 < x ∧ x < 2 → decreasing_on f1 (Ioo 1 2)) ∧
  (∀ x, 2 < x → increasing_on f1 (Ioi 2)) :=
begin
  sorry
end

-- Problem (2)
def f2 (x: ℝ) : ℝ := log x - 2 * x - (1 / x)
def g (x: ℝ) : ℝ := -x - (1 / x) - 1

theorem g_above_f2 (x: ℝ) (hx: 1 < x) : g x > f2 x :=
begin
  sorry
end

-- Problem (3)
theorem inequality_proof (n: ℕ) (hn: 2 ≤ n) :
  (finset.range n).sum (λ k, log (k + 2) / ((k + 2) ^ 2)) < (2 * n ^ 2 - n - 1) / (4 * (n + 1)) :=
begin
  sorry
end

end monotonicity_of_f1_g_above_f2_inequality_proof_l113_113573


namespace valid_inequalities_l113_113389

section inequalities

-- Assumptions and definitions
variables {a b c m : ℝ}
variable h1 : a > b > c > 0
variable h2 : a > b > c > 0
variable h3 : a > b > c > 0
variable h4 : a > b > c > 0

noncomputable theory

-- Statements of inequalities
def inequality_1 : Prop := c / a < c / b 
def inequality_2 : Prop := a + m / b + m > a / b 
def inequality_3 : Prop := (a ^ 2 + b ^ 2) / 2 ≥ ((a + b) / 2) ^ 2
def inequality_4 : Prop := a + b ≤ sqrt (2 * (a ^ 2 + b ^ 2))

-- Theorem stating which inequalities always hold
theorem valid_inequalities : inequality_1 ∧ inequality_3 ∧ inequality_4 :=
by
  split
  all_goals { sorry }

end inequalities

end valid_inequalities_l113_113389


namespace smallest_pos_real_x_solution_l113_113525

def smallest_pos_real_x (x : ℝ) : Prop :=
  x > 0 ∧ (floor (x^2) - x * floor x = 12)

theorem smallest_pos_real_x_solution :
  ∃ x > 0, smallest_pos_real_x x ∧ x = 169 / 13 :=
by
  sorry

end smallest_pos_real_x_solution_l113_113525


namespace problem_1_problem_2_problem_3_problem_4_l113_113122

theorem problem_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = 14 * Real.sqrt 5 / 5 :=
by sorry

theorem problem_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 :=
by sorry

theorem problem_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 :=
by sorry

theorem problem_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3) ^ 2 = 2 * Real.sqrt 15 - 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l113_113122


namespace lychee_production_increase_l113_113733

variable (x : ℕ) -- percentage increase as a natural number

def lychee_increase_2006 (x : ℕ) : ℕ :=
  (1 + x)*(1 + x)

theorem lychee_production_increase (x : ℕ) :
  lychee_increase_2006 x = (1 + x) * (1 + x) :=
by
  sorry

end lychee_production_increase_l113_113733


namespace trigonometric_inequality_equality_conditions_l113_113239

theorem trigonometric_inequality
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) ≥ 9 :=
sorry

theorem equality_conditions
  (α β : ℝ)
  (hα : α = Real.arctan (Real.sqrt 2))
  (hβ : β = π / 4) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) = 9 :=
sorry

end trigonometric_inequality_equality_conditions_l113_113239


namespace numberOfTrueProps_l113_113894

def prop1 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop2 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop3 (a b c : ℝ) : Prop := ¬(a > b) → ¬(a * c^2 > b * c^2)
def prop4 (a b c : ℝ) : Prop := ¬(a * c^2 > b * c^2) → ¬(a > b)

def countTrueProps (a b c : ℝ) : ℕ :=
  [prop1 a b c, prop2 a b c, prop3 a b c, prop4 a b c].count (λ p => p)

theorem numberOfTrueProps (a b c : ℝ) : countTrueProps a b c = 2 := sorry

end numberOfTrueProps_l113_113894


namespace value_of_x_l113_113531

def is_whole_number (n : ℝ) : Prop := ∃ (k : ℤ), n = k

theorem value_of_x (n : ℝ) (x : ℝ) :
  n = 1728 →
  is_whole_number (Real.log n / Real.log x + Real.log n / Real.log 12) →
  x = 12 :=
by
  intro h₁ h₂
  sorry

end value_of_x_l113_113531


namespace matrix_transformation_and_eigenvalues_l113_113914

theorem matrix_transformation_and_eigenvalues :
  ∀ (a : ℝ),
  let A := !![3, a; 0, -1],
      P := !![2; -3],
      P' := !![3; 3] in
  (A ⬝ P = P' → a = 1) ∧
  (eigenvalues A = (3, -1) ∧
   ∃ v1 v2, eigenvector A 3 v1 ∧ eigenvector A (-1) v2 ∧ 
   v1 = !![1; 0] ∧ v2 = !![1; -4]) := sorry

end matrix_transformation_and_eigenvalues_l113_113914


namespace car_sharing_problem_l113_113329

theorem car_sharing_problem 
  (x : ℕ)
  (cond1 : ∃ c : ℕ, x = 4 * c + 4)
  (cond2 : ∃ c : ℕ, x = 3 * c + 9):
  (x / 4 + 1 = (x - 9) / 3) :=
by sorry

end car_sharing_problem_l113_113329


namespace perimeter_ratio_l113_113140

theorem perimeter_ratio (w l : ℕ) (hfold : w = 8) (lfold : l = 6) 
(folded_w : w / 2 = 4) (folded_l : l / 2 = 3) 
(hcut : w / 4 = 1) (lcut : l / 2 = 3) 
(perimeter_small : ℕ) (perimeter_large : ℕ)
(hperim_small : perimeter_small = 2 * (3 + 4)) 
(hperim_large : perimeter_large = 2 * (6 + 4)) :
(perimeter_small : ℕ) / (perimeter_large : ℕ) = 7 / 10 := sorry

end perimeter_ratio_l113_113140


namespace percentage_of_work_day_in_meetings_l113_113362

theorem percentage_of_work_day_in_meetings
  (work_day_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (multiplier : ℕ)
  (total_minutes : ℕ := work_day_hours * 60)
  (second_meeting_minutes : ℕ := multiplier * first_meeting_minutes)
  (total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes)
  (percentage_spent_in_meetings : ℕ := (total_meeting_minutes * 100) / total_minutes) :
  work_day_hours = 10 ∧ first_meeting_minutes = 45 ∧ multiplier = 3 → percentage_spent_in_meetings = 30 :=
by
  intros conditions
  have h1 : work_day_hours = 10 := conditions.left.left
  have h2 : first_meeting_minutes = 45 := conditions.left.right
  have h3 : multiplier = 3 := conditions.right
  subst h1
  subst h2
  subst h3
  sorry

end percentage_of_work_day_in_meetings_l113_113362


namespace sarah_probability_l113_113031

noncomputable def probability_odd_product_less_than_20 : ℚ :=
  let total_possibilities := 36
  let favorable_pairs := [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3)]
  let favorable_count := favorable_pairs.length
  let probability := favorable_count / total_possibilities
  probability

theorem sarah_probability : probability_odd_product_less_than_20 = 2 / 9 :=
by
  sorry

end sarah_probability_l113_113031


namespace prime_sum_count_15_l113_113817

def prime_sums (n : ℕ) : ℕ := sorry  -- This function should compute the nth sum of consecutive primes.

theorem prime_sum_count_15 (h1 : prime_sums 1 = 2) 
                          (h2 : prime_sums 2 = 5) 
                          (h3 : prime_sums 3 = 10) 
                          (h4 : prime_sums 4 = 17)
                          (h5 : prime_sums 5 = 28)
                          (h6 : prime_sums 6 = 41)
                          (h7 : prime_sums 7 = 58)
                          (h8 : prime_sums 8 = 77)
                          (h9 : prime_sums 9 = 100)
                          (h10 : prime_sums 10 = 129)
                          (h11 : prime_sums 11 = 160)
                          (h12 : prime_sums 12 = 197)
                          (h13 : prime_sums 13 = 238)
                          (h14 : prime_sums 14 = 281)
                          (h15 : prime_sums 15 = 328)
                          (pr1 : nat.prime (prime_sums 1))
                          (pr2 : nat.prime (prime_sums 2))
                          (pr3 : nat.prime (prime_sums 4))
                          (pr4 : nat.prime (prime_sums 6))
                          (pr5 : nat.prime (prime_sums 7))
                          (pr6 : nat.prime (prime_sums 11))
                          (pr7 : nat.prime (prime_sums 14))
                          (npr1 : ¬nat.prime (prime_sums 3))
                          (npr2 : ¬nat.prime (prime_sums 5))
                          (npr3 : ¬nat.prime (prime_sums 8))
                          (npr4 : ¬nat.prime (prime_sums 9))
                          (npr5 : ¬nat.prime (prime_sums 10))
                          (npr6 : ¬nat.prime (prime_sums 12))
                          (npr7 : ¬nat.prime (prime_sums 13))
                          (npr8 : ¬nat.prime (prime_sums 15)) :
  (list.nth_le (list_of_prime_sums (range 15) 5 sorry.typed.bool.shuffle_shuffle {x | nat.prime x}) = 6) := sorry

end prime_sum_count_15_l113_113817


namespace proof_part1_proof_part2_l113_113615

variable {A B C a b c : Real}

-- Conditions
def condition1 : Prop := 4 * sin^2 ((A + B) / 2) - cos (2 * C) = 7 / 2
def condition2 : Prop := c = Real.sqrt 7

-- The first conclusion
def conclusion1 : Prop := C = Real.pi / 3

-- The second conclusion (proof of maximum area)
def conclusion2 : Prop := ∃ a b : ℝ, a = b ∧ ∀ S : ℝ, 
  S = (1/2) * a * b * sin C → S ≤ (7 * Real.sqrt 3) / 4

theorem proof_part1 (h1 : condition1) (h2 : condition2) : conclusion1 := 
by sorry

theorem proof_part2 (h1 : condition1) (h2 : condition2) : conclusion2 :=
by sorry

end proof_part1_proof_part2_l113_113615


namespace smallest_m_divisible_by_31_l113_113227

theorem smallest_m_divisible_by_31 (n : ℕ) (hn: 0 < n) :
  ∃ m : ℕ, (m + 2 ^ (5 * n)) % 31 = 0 ∧ ∀ m' : ℕ, (m' + 2 ^ (5 * n)) % 31 = 0 → m ≤ m' :=
begin
  use 30,
  split,
  { sorry }, 
  { 
    intros m' h,
    sorry 
  }
end

end smallest_m_divisible_by_31_l113_113227


namespace prime_number_condition_l113_113506

theorem prime_number_condition (n : ℕ) (h1 : n ≥ 2) :
  (∀ d : ℕ, d ∣ n → d > 1 → d^2 + n ∣ n^2 + d) → Prime n :=
sorry

end prime_number_condition_l113_113506


namespace largest_number_of_stamps_per_page_l113_113647

theorem largest_number_of_stamps_per_page :
  Nat.gcd (Nat.gcd 1200 1800) 2400 = 600 :=
sorry

end largest_number_of_stamps_per_page_l113_113647


namespace find_xy_solution_l113_113215

theorem find_xy_solution : ∃ x y : ℝ, (x - 9) ^ 2 + (y - 10) ^ 2 + (x - y) ^ 2 = 1 / 3 ∧ 
                             x = 9 + 1 / 3 ∧ y = 9 + 2 / 3 :=
by {
  use [9 + 1 / 3, 9 + 2 / 3],
  split,
  { sorry },
  { split; refl }
}

end find_xy_solution_l113_113215


namespace minimum_luxury_owners_l113_113626

-- Defining the basic structure of the village population
variable {V : Type} -- The population of the village 
variable [Fintype V] -- Assuming a finite type for the population

-- Percentage of people having certain luxuries
constant has_refrigerator : V → Prop
constant has_television : V → Prop 
constant has_computer : V → Prop
constant has_air_conditioner : V → Prop
constant has_washing_machine : V → Prop 
constant has_microwave_oven : V → Prop 
constant has_high_speed_internet : V → Prop

-- Assume the percentage of each luxury owner
axiom percent_refrigerator : (fintype.card {v : V // has_refrigerator v}) = 0.67 * (fintype.card V)
axiom percent_television : (fintype.card {v : V // has_television v}) = 0.74 * (fintype.card V)
axiom percent_computer : (fintype.card {v : V // has_computer v}) = 0.77 * (fintype.card V)
axiom percent_air_conditioner : (fintype.card {v : V // has_air_conditioner v}) = 0.83 * (fintype.card V)
axiom percent_washing_machine : (fintype.card {v : V // has_washing_machine v}) = 0.55 * (fintype.card V)
axiom percent_microwave : (fintype.card {v : V // has_microwave_oven v}) = 0.48 * (fintype.card V)
axiom percent_internet : (fintype.card {v : V // has_high_speed_internet v}) = 0.42 * (fintype.card V)

-- Additional population overlaps
axiom percent_tv_and_computer : (fintype.card {v : V // has_television v ∧ has_computer v}) = 0.35 * (fintype.card V)
axiom percent_wm_and_microwave : (fintype.card {v : V // has_washing_machine v ∧ has_microwave_oven v}) = 0.30 * (fintype.card V)
axiom percent_ac_and_refrigerator : (fintype.card {v : V // has_air_conditioner v ∧ has_refrigerator v}) = 0.27 * (fintype.card V)

-- Top 10% highest income earners having all luxuries
axiom top_10_percent_luxuries : (fintype.card {v : V // has_refrigerator v ∧ has_television v ∧ has_computer v ∧ has_air_conditioner v ∧ has_washing_machine v ∧ has_microwave_oven v ∧ has_high_speed_internet v}) = 0.10 * (fintype.card V)

theorem minimum_luxury_owners : (fintype.card {v : V // has_refrigerator v ∧ has_television v ∧ has_computer v ∧ has_air_conditioner v ∧ has_washing_machine v ∧ has_microwave_oven v ∧ has_high_speed_internet v}) = 0.10 * (fintype.card V) :=
sorry

end minimum_luxury_owners_l113_113626


namespace triangle_inequality_l113_113251

theorem triangle_inequality
  (ABC : Type) [triangle ABC]
  (A B C : point ABC)
  (I : incenter A B C)
  (A' B' C' : point ABC)
  (HA' : angle_bisector A B C A')
  (HB' : angle_bisector B A C B')
  (HC' : angle_bisector C A B C') :
  1 / 4 < (distance A I * distance B I * distance C I) / (distance A A' * distance B B' * distance C C') ∧ 
  (distance A I * distance B I * distance C I) / (distance A A' * distance B B' * distance C C') ≤ 8 / 27 :=
by 
  sorry

end triangle_inequality_l113_113251


namespace complex_numbers_magnitude_squared_l113_113048

theorem complex_numbers_magnitude_squared (z : ℂ) (a b : ℝ) :
  z = a + b * I ∧ z + complex.abs z = 4 + 5 * I ∧ complex.abs z = real.sqrt (a^2 + b^2)
  → complex.norm_sq z = 1681 / 64 := 
by
  sorry

end complex_numbers_magnitude_squared_l113_113048


namespace distance_between_parallel_lines_l113_113738

theorem distance_between_parallel_lines (r d : ℝ)
  (h1 : ∀ x y z : ℝ, x^2 + y^2 = z^2 → x = cos(z) \lor y = sin(z)) :
  (42 * 42 + (d / 2)^2 = r^2) ∧ 
  (40 * 40 + (3 * d / 2)^2 = r^2) →
  d = Real.sqrt (92 / 11) :=
by
  sorry

end distance_between_parallel_lines_l113_113738


namespace max_sum_sin_cos_real_l113_113997

theorem max_sum_sin_cos_real (θ : Fin 2008 → ℝ) : 
  ∃ θ : Fin 2008 → ℝ, (finset.univ.sum (λ i : Fin 2008, Real.sin (θ i) * Real.cos (θ ((i + 1) % 2008)))) = 1004 :=
sorry

end max_sum_sin_cos_real_l113_113997


namespace tan_315_deg_l113_113850

theorem tan_315_deg : Real.tan (315 * Real.pi / 180) = -1 := sorry

end tan_315_deg_l113_113850


namespace repeating_decimal_arithmetic_l113_113175

def x : ℚ := 0.234 -- repeating decimal 0.234
def y : ℚ := 0.567 -- repeating decimal 0.567
def z : ℚ := 0.891 -- repeating decimal 0.891

theorem repeating_decimal_arithmetic :
  x - y + z = 186 / 333 := 
sorry

end repeating_decimal_arithmetic_l113_113175


namespace remove_terms_yield_desired_sum_l113_113863

-- Define the original sum and the terms to be removed
def originalSum : ℚ := 1/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
def termsToRemove : List ℚ := [1/9, 1/12, 1/15, 1/18]

-- Definition of the desired remaining sum
def desiredSum : ℚ := 1/2

noncomputable def sumRemainingTerms : ℚ :=
originalSum - List.sum termsToRemove

-- Lean theorem to prove
theorem remove_terms_yield_desired_sum : sumRemainingTerms = desiredSum :=
by 
  sorry

end remove_terms_yield_desired_sum_l113_113863


namespace smallest_k_for_covering_rectangles_l113_113353

theorem smallest_k_for_covering_rectangles (n : ℕ) (hn : 0 < n)
  (S : Set (ℝ × ℝ)) (hS : S ⊆ Set.Ioo 0 1 ×ˢ Set.Ioo 0 1) (hS_card : S.card = n) :
  ∃ k : ℕ, k = 2 * n + 2 ∧ 
  ∀ (R : Finset (Set (ℝ × ℝ))), R.card = k →
  (∀ r ∈ R, ∃ x1 x2 y1 y2, r = Set.Ioo x1 x2 ×ˢ Set.Ioo y1 y2 ∧
    ((∀ p ∈ S, p ∉ Set.Ioo x1 x2 ×ˢ Set.Ioo y1 y2) ∧ 
     ∀ q ∈ Set.Ioo 0 1 ×ˢ Set.Ioo 0 1, q ∉ S → q ∈ ⋃₀ R)) := sorry

end smallest_k_for_covering_rectangles_l113_113353


namespace solve_other_endpoint_l113_113061

structure Point where
  x : ℤ
  y : ℤ

def midpoint : Point := { x := 3, y := 1 }
def known_endpoint : Point := { x := 7, y := -3 }

def calculate_other_endpoint (m k : Point) : Point :=
  let x2 := 2 * m.x - k.x;
  let y2 := 2 * m.y - k.y;
  { x := x2, y := y2 }

theorem solve_other_endpoint : calculate_other_endpoint midpoint known_endpoint = { x := -1, y := 5 } :=
  sorry

end solve_other_endpoint_l113_113061


namespace symmetry_axis_of_sine_sub_pi_over_3_l113_113398

theorem symmetry_axis_of_sine_sub_pi_over_3 :
  ∃ k : ℤ, x = -π/6 + k * π ↔ y = sin (x - π/3) := 
sorry

end symmetry_axis_of_sine_sub_pi_over_3_l113_113398


namespace find_y_l113_113111

variable (y k : ℝ)

-- Conditions
def inverse_relationship (y : ℝ) (k : ℝ) : ℝ := k / y^2
def given_x_1 := (inverse_relationship y k = 1)
def given_x_0_25 := (inverse_relationship 6 k = 0.25)

-- Proof goal
theorem find_y : given_x_1 ∧ given_x_0_25 → y = 3 :=
by
  done
  sorry

end find_y_l113_113111


namespace tank_capacity_l113_113796

-- Definitions based on conditions
def emptying_rate_due_to_leak (C : ℝ) : ℝ := C / 6
def filling_rate_due_to_inlet : ℝ := 3 * 60
def net_emptying_rate_due_to_both_open (C : ℝ) : ℝ := C / 8

-- The theorem we need to prove
theorem tank_capacity : 
  ∃ (C : ℝ), filling_rate_due_to_inlet - emptying_rate_due_to_leak C = net_emptying_rate_due_to_both_open C ∧ C = 617 := by
  sorry

end tank_capacity_l113_113796


namespace slope_probability_unit_square_l113_113655

theorem slope_probability_unit_square:
  let P := { (x,y) : ℝ × ℝ | 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 } in
  let point := (3/4, 1/4) in
  let slope := 1/3 in
  let probability := 1/3 in
  (Prob (λ (p : ℝ × ℝ), (p.snd - point.snd) / (p.fst - point.fst) > slope)) = probability
  → ∃ (m n : ℕ), m + n = 4 ∧ p = m / n ∧ gcd m n = 1 :=
by
  sorry

end slope_probability_unit_square_l113_113655


namespace exists_multiple_all_ones_l113_113665

open Nat

theorem exists_multiple_all_ones (n : ℕ) (hn_pos : 0 < n) (hn_coprime : gcd n 10 = 1) :
  ∃ k : ℕ, k > 0 ∧ 10^k ≡ 1 [MOD 9 * n] :=
by
  sorry

end exists_multiple_all_ones_l113_113665


namespace br_eq_o1o2_l113_113998

variables {A B C P Q R O O1 O2 : Type*}
variables [IsoscelesTriangle ABC] [PointOn P AC] [PointOn Q AB] [PointOn R BC]
variables (AP_eq_QB : AP = QB) (angle_PBC_eq : ∠PBC = 90 - ∠BAC) (RP_eq_RQ : RP = RQ)
variables (circumcenter_O1 : Circumcenter (Triangle APQ) = O1)
variables (circumcenter_O2 : Circumcenter (Triangle CRP) = O2)

theorem br_eq_o1o2 (h1 : IsoscelesTriangle ABC) (h2 : PointOn P AC) (h3 : PointOn Q AB)
                  (h4 : PointOn R BC) (h5 : AP = QB) (h6 : ∠PBC = 90 - ∠BAC)
                  (h7 : RP = RQ) (h8 : Circumcenter (Triangle APQ) = O1)
                  (h9 : Circumcenter (Triangle CRP) = O2) : 
                  Length (Segment BR) = Length (Segment O1O2) :=
by
  sorry

end br_eq_o1o2_l113_113998


namespace fifth_inequality_proof_l113_113016

theorem fifth_inequality_proof :
  (1 + 1 / (2^2 : ℝ) + 1 / (3^2 : ℝ) + 1 / (4^2 : ℝ) + 1 / (5^2 : ℝ) + 1 / (6^2 : ℝ) < 11 / 6) 
  := 
sorry

end fifth_inequality_proof_l113_113016


namespace sum_geometric_sequence_l113_113927

variable (a₁ q : ℝ) (n : ℕ)

def Sn (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem sum_geometric_sequence :
  ∀ (a₁ q : ℝ) (n : ℕ), Sn a₁ q n =
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q) := by
  sorry

end sum_geometric_sequence_l113_113927


namespace sufficient_but_not_necessary_condition_l113_113536

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : x > 2 ∨ x < -1 :=
by
  sorry

example (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : (x > 2) ∨ (x < -1) := 
by 
  apply sufficient_but_not_necessary_condition; exact h

end sufficient_but_not_necessary_condition_l113_113536


namespace trains_clear_time_l113_113096

-- Define constants for the lengths of the trains.
def length_train1 : ℝ := 150 -- in meters
def length_train2 : ℝ := 200 -- in meters

-- Define constants for the speeds of the trains in km/h.
def speed_train1_kmh : ℝ := 100 -- in km/h
def speed_train2_kmh : ℝ := 120 -- in km/h

-- Convert the speeds from km/h to m/s.
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (1000 / 3600)

def speed_train1_ms : ℝ := kmh_to_ms speed_train1_kmh
def speed_train2_ms : ℝ := kmh_to_ms speed_train2_kmh

-- Define the relative speed as the sum of the two speeds (opposite directions).
def relative_speed_ms : ℝ := speed_train1_ms + speed_train2_ms

-- Define the total distance to be covered.
def total_distance : ℝ := length_train1 + length_train2

-- Calculate the time it takes for the trains to be completely clear of each other.
def time_to_clear : ℝ := total_distance / relative_speed_ms

-- The theorem to be proved.
theorem trains_clear_time : time_to_clear ≈ 5.72 := by sorry

end trains_clear_time_l113_113096


namespace find_a_value_l113_113564

noncomputable def tangent_slope_perpendicular_to_line (a : ℝ) : Prop :=
  ∀ (x : ℝ), a * x + (3 * x^2 - 2) * (-1) = a

theorem find_a_value : ∃ a : ℝ, (tangent_slope_perpendicular_to_line a) ∧ a = 1 :=
begin
  use 1,
  split,
  { intros x,
    calc
      1 * x + (3 * x^2 - 2) * (-1) = x - (3 * x^2 - 2)  : by ring
      ... = - (3 * x^2 - 2 - x)  : by ring },
  { refl }
end

end find_a_value_l113_113564


namespace union_of_sets_l113_113668

theorem union_of_sets :
  let A := { x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) }
  let B := { y : ℝ | ∃ x : ℝ, y = 2 ^ x }
  (A ∪ B) = (set.Ioi 0) :=
by
  sorry

end union_of_sets_l113_113668


namespace AE_eq_AF_l113_113612

-- Define the problem's context within Lean.

variable {A B C M N P E F K H Q : Type}
variable [triangle: triangle A B C]
variable [h_midpoints: is_midpoint M B C ∧ is_midpoint N A C ∧ is_midpoint P A B]
variable [h_angles: angle B N E C = (1 / 2 * angle A M B) ∧ angle B P F = (1 / 2 * angle A M C)]

-- Define the proof goal using the above conditions.
theorem AE_eq_AF 
  (h1 : is_midpoint M B C)
  (h2 : is_midpoint N A C)
  (h3 : is_midpoint P A B)
  (h4 : angle B N E C = (1 / 2 * angle A M B))
  (h5 : angle B P F = (1 / 2 * angle A M C)) :
  distance A E = distance A F :=
by sorry

end AE_eq_AF_l113_113612


namespace find_a_l113_113308

theorem find_a (a : ℕ) (h : a * 2 * 2^3 = 2^6) : a = 4 := 
by 
  sorry

end find_a_l113_113308


namespace unit_prices_cost_function_max_trees_l113_113742

-- 1. Prove unit prices
theorem unit_prices (pA pB : ℝ) :
  (pA = 2 * pB) →
  (3 * pA + 2 * pB = 320) →
  (pB = 40) ∧ (pA = 80) :=
by
  intros h₁ h₂
  have : pA = 80 := sorry
  have : pB = 40 := sorry
  tauto

-- 2. Prove cost function
theorem cost_function (a : ℕ) :
  (∀ a, w = 40 * a + 8000)}} :=
by
  have : w = 40 * a + 8000 := sorry
  tauto

-- 3. Prove maximum number of type A trees
theorem max_trees (a : ℕ) :
  (72 * a + 40 * (200 - a) ≤ 12000) → 
  (a ≤ 125) :=
by
  have : a ≤ 125 := sorry
  tauto

end unit_prices_cost_function_max_trees_l113_113742


namespace Ivan_increases_share_more_than_six_times_l113_113020

theorem Ivan_increases_share_more_than_six_times
  (p v s i : ℝ)
  (hp : p / (v + s + i) = 3 / 7)
  (hv : v / (p + s + i) = 1 / 3)
  (hs : s / (p + v + i) = 1 / 3) :
  ∃ k : ℝ, k > 6 ∧ i * k > 0.6 * (p + v + s + i * k) :=
by
  sorry

end Ivan_increases_share_more_than_six_times_l113_113020


namespace root_quadrant_l113_113607

theorem root_quadrant (m n : ℝ) (h1 : m + n = 2) (h2 : m * n = -3) (h3 : m < n) :
    ∃ q, q = 2 ∧ ((m, n) ∈ Quadrant.two) :=
sorry

end root_quadrant_l113_113607


namespace total_hours_A_ascending_and_descending_l113_113788

theorem total_hours_A_ascending_and_descending
  (ascending_speed_A ascending_speed_B descending_speed_A descending_speed_B distance summit_distance : ℝ)
  (h1 : descending_speed_A = 1.5 * ascending_speed_A)
  (h2 : descending_speed_B = 1.5 * ascending_speed_B)
  (h3 : ascending_speed_A > ascending_speed_B)
  (h4 : 1/ascending_speed_A + 1/ascending_speed_B = 1/hour - 600/summit_distance)
  (h5 : 0.5 * summit_distance/ascending_speed_A = (summit_distance - 600)/ascending_speed_B) :
  (summit_distance / ascending_speed_A) + (summit_distance / descending_speed_A) = 1.5 := 
sorry

end total_hours_A_ascending_and_descending_l113_113788


namespace proof_problem_1_proof_problem_2_l113_113438

noncomputable def problem_1 (a b : ℝ) : Prop :=
  ((2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3))) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6)

noncomputable def problem_2 : Prop :=
  ((2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 2^(3/4 - 1) - (-2005)^0) = 100

theorem proof_problem_1 (a b : ℝ) : problem_1 a b := 
  sorry

theorem proof_problem_2 : problem_2 := 
  sorry

end proof_problem_1_proof_problem_2_l113_113438


namespace find_special_n_l113_113609

-- Define perfect number
def is_perfect_number (m : ℕ) : Prop :=
  (∑ i in (finset.filter (λ d, d ∣ m) (finset.range (m + 1))), i) = 2 * m

-- The main statement we need to prove
theorem find_special_n (n : ℕ) (h1 : is_perfect_number (n - 1)) (h2 : is_perfect_number ((n * (n + 1)) / 2)) : n = 7 :=
sorry

end find_special_n_l113_113609


namespace cool_31_32_cool_consecutive_n_digit_l113_113753

-- Define the concept of a "cool" number
def isCool (n : ℕ) : Prop :=
  ∃ f : ℕ → ℕ, (∀ k, f (sumSquaresOfDigits (f k)) = if sumSquaresOfDigits (f k) = 1 then 1 else f (sumSquaresOfDigits (f k) - 1)) ∧ (f n = 1)

-- Definition for sum of squares of digits in Lean
def sumSquaresOfDigits (n : ℕ) : ℕ := 
  n.digits 10 |>.foldl (fun acc d => acc + d^2) 0

-- Prove that 31 and 32 are cool two-digit numbers
theorem cool_31_32 : isCool 31 ∧ isCool 32 :=
by sorry

-- For every n > 2020, there exist two consecutive n-digit cool numbers
theorem cool_consecutive_n_digit (n : ℕ) (hn : n > 2020) : 
  ∃ a b : ℕ, (a < b) ∧ (a.digits.length = n) ∧ (b.digits.length = n) ∧ isCool a ∧ isCool b :=
by sorry

end cool_31_32_cool_consecutive_n_digit_l113_113753


namespace equation_solution_l113_113032

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l113_113032


namespace gardening_project_cost_l113_113171

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end gardening_project_cost_l113_113171


namespace message_channels_encryption_l113_113075

theorem message_channels_encryption :
  ∃ (assign_key : Fin 105 → Fin 105 → Fin 100),
  ∀ (u v w x : Fin 105), 
  u ≠ v → u ≠ w → u ≠ x → v ≠ w → v ≠ x → w ≠ x →
  (assign_key u v = assign_key u w ∧ assign_key u v = assign_key u x ∧ 
   assign_key u v = assign_key v w ∧ assign_key u v = assign_key v x ∧ 
   assign_key u v = assign_key w x) → False :=
by
  sorry

end message_channels_encryption_l113_113075


namespace circle_area_ratio_l113_113452

theorem circle_area_ratio (d : ℝ) :
  let r := d / 2 in
  let A := π * r^2 in
  let new_d := 3 * d in
  let new_r := new_d / 2 in
  let A' := π * new_r^2 in
  A / A' = (1 : ℚ) / 9 :=
by
  sorry

end circle_area_ratio_l113_113452


namespace num_new_students_l113_113385

theorem num_new_students 
  (original_avg_age : ℕ) 
  (original_num_students : ℕ) 
  (new_avg_age : ℕ) 
  (age_decrease : ℕ) 
  (total_age_orginal : ℕ := original_num_students * original_avg_age) 
  (total_new_students : ℕ := (original_avg_age - age_decrease) * (original_num_students + 12))
  (x : ℕ := total_new_students - total_age_orginal) :
  original_avg_age = 40 → 
  original_num_students = 12 →
  new_avg_age = 32 →
  age_decrease = 4 →
  x = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end num_new_students_l113_113385


namespace proof_problem_l113_113917

noncomputable def parabola : Type := sorry

-- 1. The parabola is given by the equation y² = 4x
def equation_of_parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Point M(1, 2) is on the parabola, lines MP and MQ are perpendicular and intersect the parabola at points P and Q
def point_on_parabola (x y : ℝ) (h : equation_of_parabola x y) : Prop := x = 1 ∧ y = 2
def perpendicular_lines (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation_of_parabola x₁ y₁) (h₂ : equation_of_parabola x₂ y₂) : Prop :=
  (y₁ - 2) * (y₂ - 2) + (x₁ - 1) * (x₂ - 1) = 0

-- Points A and B on the parabola, with |AB| = 9/2
def points_AB (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation_of_parabola x₁ y₁) (h₂ : equation_of_parabola x₂ y₂) : Prop :=
  ((x₂ - x₁)^2 + (y₂ - y₁)^2) = (9 / 2) 

-- 2. The line PQ passes through the fixed point B(5, -2)
def fixed_point_B (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation_of_parabola x₁ y₁) (h₂ : equation_of_parabola x₂ y₂) : Prop :=
  let y₁' := (4 - (4 * 1)) in
  let y₂' := (4 - (4 * 1)) in
  let line_PQ := (λ (x : ℝ), (4 * x) / (y₂ + y₁) -2) in
  line_PQ 5 = -2

-- 3. The maximum distance from the origin to the line PQ is √29
def max_distance_origin_to_PQ (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation_of_parabola x₁ y₁) (h₂ : equation_of_parabola x₂ y₂) : Prop :=
  @Real.sqrt 29

theorem proof_problem :
  (∀ x y, equation_of_parabola x y) ∧
  (perpendicular_lines 1 2  y₁ y₂) ∧
  ∃ x₁ y₁ x₂ y₂, points_AB x₁ y₁ x₂ y₂  → 
  (∃ x₁ y₁ x₂ y₂, fixed_point_B x₁ y₁ x₂ y₂) →
  (max_distance_origin_to_PQ x₁ y₁ x₂ y₂) :=
begin
  sorry
end

end proof_problem_l113_113917


namespace percentage_of_24_eq_0_12_l113_113449

theorem percentage_of_24_eq_0_12 : (p : ℝ) → (h : p * 24 = 0.12) → p * 100 = 0.5 :=
by
  intro p h
  calc
    p * 24 = 0.12 : h
    have h1 : p = 0.12 / 24 := by field_simp; exact (div_eq_iff (24 ≠ 0)).mpr h
    calc
      p * 100 = (0.12 / 24) * 100 : by rw [h1]
            ... = 0.5         : by field_simp [division_def]; norm_num

end percentage_of_24_eq_0_12_l113_113449


namespace exists_indices_l113_113250

-- Define the sequence condition
def is_sequence_of_all_positive_integers (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a m = n) ∧ (∀ n m1 m2 : ℕ, a m1 = n ∧ a m2 = n → m1 = m2)

-- Main theorem statement
theorem exists_indices 
  (a : ℕ → ℕ) 
  (h : is_sequence_of_all_positive_integers a) :
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ (a 0 + a m = 2 * a ℓ) :=
by
  sorry

end exists_indices_l113_113250


namespace poisson_pmf_3_poisson_expectation_4_poisson_variance_4_l113_113138

noncomputable def poisson_pmf (a k : ℕ) : ℝ :=
  (a^k * Real.exp (-a : ℝ)) / Nat.fact k

def poisson_expectation (a : ℝ) : ℝ := a

def poisson_variance (a : ℝ) : ℝ := a

theorem poisson_pmf_3 (a : ℕ) (h : a = 4) : poisson_pmf a 3 = 0.1952 := sorry

theorem poisson_expectation_4 (a : ℝ) (h : a = 4) : poisson_expectation a = 4 := sorry

theorem poisson_variance_4 (a : ℝ) (h : a = 4) : poisson_variance a = 4 := sorry

end poisson_pmf_3_poisson_expectation_4_poisson_variance_4_l113_113138


namespace eccentricity_range_l113_113358

-- Definitions and Conditions
variables {a b c : ℝ}
variable {e : ℝ} -- eccentricity

def ellipse_eq (x y : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def valid_bounds : Prop := a > b ∧ b > 0 ∧ 1 ≤ λ ∧ λ ≤ 2 ∧ e = c / a
def tan_product_condition {Q A B : Point} : Prop :=
  if Q ≠ A ∧ Q ≠ B then
    let θ1 := angle Q A B
    let θ2 := angle Q B A
    (tan θ1) * (tan θ2) < 2 / 3
  else true

-- Proof Problem
theorem eccentricity_range
  (h1 : valid_bounds)
  (h2 : tan_product_condition Q A B)
  (h3 : 1 ≤ λ ∧ λ ≤ 2)
  : e ∈ set.Ioc (sqrt 3 / 3) (2 / 3) :=
  sorry

end eccentricity_range_l113_113358


namespace geometric_sequence_a5_l113_113283

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a3 : a 3 = -1)
  (h_a7 : a 7 = -9) : a 5 = -3 := 
sorry

end geometric_sequence_a5_l113_113283


namespace sum_constants_l113_113734

theorem sum_constants (a b x : ℝ) 
  (h1 : (x - a) / (x + b) = (x^2 - 50 * x + 621) / (x^2 + 75 * x - 3400))
  (h2 : x^2 - 50 * x + 621 = (x - 27) * (x - 23))
  (h3 : x^2 + 75 * x - 3400 = (x - 40) * (x + 85)) :
  a + b = 112 :=
sorry

end sum_constants_l113_113734


namespace length_of_AB_is_correct_l113_113580

-- Definitions of the hyperbola and circle provided in the conditions
def hyperbola (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1)

def circle (center : ℝ × ℝ) (radius : ℝ) : Prop := radius > 0 ∧ ∃ x y, ((x - center.1)^2 + (y - center.2)^2 = radius^2)

-- Definition of the problem's conditions
def conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧
(∃ e : ℝ, e = sqrt 5 ∧ e = sqrt (1 + (b^2 / a^2))) ∧
(hyperbola a b) ∧
(circle (2, 3) 1)

-- The main theorem we need to prove
theorem length_of_AB_is_correct (a b : ℝ) : conditions a b → (∃ AB : ℝ, AB = 4 * sqrt 5 / 5) :=
by {
    intros h,
    have h1 : a > 0 ∧ b > 0, from h.1,
    sorry
}

end length_of_AB_is_correct_l113_113580


namespace fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l113_113434

theorem fraction_inequalities (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1 / 2 ≤ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ∧ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ≤ 1 :=
sorry

theorem fraction_inequality_equality_right (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1 - a) * (1 - b) = 0 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) :=
sorry

theorem fraction_inequality_equality_left (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  a = b ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l113_113434


namespace hexagon_parallelogram_l113_113637

open EuclideanGeometry

noncomputable def is_symmetric_point (A A' B F : Point) : Prop := ∃ (M : Point), midpoint B F M ∧ reflection M A = A'

theorem hexagon_parallelogram
  (A B C D E F A' : Point)
  (h₁ : angle A B F = angle C B D)
  (h₂ : angle A F B = angle E F D)
  (h₃ : angle A = angle C ∧ angle A = angle E ∧ angle A ≤ 180)
  (h₄ : is_symmetric_point A A' B F)
  (h₅ : ¬ collinear C E A') :
  parallelogram A' C D E := 
sorry

end hexagon_parallelogram_l113_113637


namespace sum_is_56_l113_113541

noncomputable def sum_of_four_numbers (a b c d : ℝ) : ℝ :=
  a + b + c + d

-- Given pairs
def pairs (a b c d : ℝ) : list ℝ :=
  [a + b, b + c, c + d, a + c, b + d, a + d].sort (λ x y, x < y)

-- Main theorem
theorem sum_is_56 (a b c d : ℝ) (h_pairs : pairs a b c d = [7, 15, 43, 47, _, _]) :
  sum_of_four_numbers a b c d = 56 :=
by sorry

end sum_is_56_l113_113541


namespace find_y_when_x_is_twelve_l113_113599

variables (x y k : ℝ)

theorem find_y_when_x_is_twelve
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = 12) :
  y = 56.25 :=
sorry

end find_y_when_x_is_twelve_l113_113599


namespace prob_1_prob_2_prob_3_l113_113276

-- Problem (I)
theorem prob_1 (x : ℝ) : (e^x - 1 - x) ≥ 0 :=
sorry

-- Problem (II)
theorem prob_2 (a : ℝ) (x : ℝ) (hx : x ≥ 0) (h : (e^x - 1 - x - a * x^2) ≥ 0) : a ≤ 1 / 2 :=
sorry

-- Problem (III)
theorem prob_3 (x : ℝ) (hx : x > 0) : (e^x - 1) * log (x + 1) > x^2 :=
sorry

end prob_1_prob_2_prob_3_l113_113276


namespace range_of_k_for_obtuse_triangle_l113_113887

theorem range_of_k_for_obtuse_triangle (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) : 
  2 < k ∧ k < 6 :=
by
  sorry

end range_of_k_for_obtuse_triangle_l113_113887


namespace angle_A_at_max_value_of_c_over_b_plus_b_over_c_l113_113614

theorem angle_A_at_max_value_of_c_over_b_plus_b_over_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_triangle : a = b * math.sin A + c * math.sqrt (1 - (math.sin A)^2))
  (h_altitude : a = b * math.sin A + c * math.sqrt (1 - (math.sin A)^2))
  (h_max : ∀ x, (b/c + c/b) ≤ (2/ℚ√3 * (math.sin A + ℚ√3 * math.cos A))) :
  A = π/6 :=
by
  sorry

end angle_A_at_max_value_of_c_over_b_plus_b_over_c_l113_113614


namespace no_infinite_primes_l113_113021

theorem no_infinite_primes (p : ℕ → ℕ) (h_prime: ∀ n : ℕ, Nat.Prime (p n))
    (h_order: ∀ n: ℕ, (p n) < (p (n + 1)))
    (h_relation: ∀ k: ℕ, (p (k + 1) = 2 * (p k) + 1) ∨ (p (k + 1) = 2 * (p k) - 1)) :
    ¬ ∀ k : ℕ, Nat.Prime (p k) :=
begin
  sorry
end

end no_infinite_primes_l113_113021


namespace find_total_values_l113_113735

theorem find_total_values (n : ℕ) (S : ℝ) 
  (h1 : S / n = 150) 
  (h2 : (S + 25) / n = 151.25) 
  (h3 : 25 = 160 - 135) : n = 20 :=
by
  sorry

end find_total_values_l113_113735


namespace minor_arc_MB_is_60_l113_113635

-- Assume we have a circle with points M, B, S on the circumference.
variables {P : Type} [circle : circle P] 
variables {M B S Q Q} -- Points on the circle

-- Given angle MBS is 60 degrees.
constant angle_MBS : ∠MBS = 60

-- Prove minor arc MB is 60 degrees.
theorem minor_arc_MB_is_60 : 
  arc_measure (arc M B) = 60 :=
sorry

end minor_arc_MB_is_60_l113_113635


namespace inclination_of_line_l113_113551

noncomputable def inclination_angle (v : ℝ × ℝ) : ℝ :=
  let (x, y) := v
  if x = 0 then 
    if y > 0 then π / 2 else 3 * π / 2
  else
    real.atan (y / x) + (if x < 0 then π else 0)

theorem inclination_of_line 
  (v : ℝ × ℝ) (hv : v = (-1, real.sqrt 3)) : inclination_angle v = 2 * π / 3 := 
by
  rw [hv, inclination_angle]
  -- setting up calculations
  simp only
  -- known result for atan and simplifications
  sorry

end inclination_of_line_l113_113551


namespace increase_in_area_400ft2_l113_113787

theorem increase_in_area_400ft2 (l w : ℝ) (h₁ : l = 60) (h₂ : w = 20)
  (h₃ : 4 * (l + w) = 4 * (4 * (l + w) / 4 / 4 )):
  (4 * (l + w) / 4) ^ 2 - l * w = 400 := by
  sorry

end increase_in_area_400ft2_l113_113787


namespace two_digit_numbers_summing_to_143_l113_113840

-- We are to prove that the number of two-digit numbers which sum to 143 with their reverse number is 6.
theorem two_digit_numbers_summing_to_143 : 
  {n : ℕ | n / 10 + n % 10 = 13 ∧ n / 10 < 10 ∧ n / 10 > 0 ∧ n % 10 < 10}.card = 6 :=
sorry

end two_digit_numbers_summing_to_143_l113_113840


namespace consecutive_days_sum_to_100_l113_113333

theorem consecutive_days_sum_to_100 :
  ∃ (days : List ℕ), days.length = 7 ∧
                      (days.nth 0 = some 29) ∧ (days.nth 1 = some 30) ∧
                      (days.nth 2 = some 31) ∧ (days.nth 3 = some 1) ∧
                      (days.nth 4 = some 2) ∧ (days.nth 5 = some 3) ∧
                      (days.nth 6 = some 4) ∧
                      days.sum = 100 :=
by
  sorry

end consecutive_days_sum_to_100_l113_113333


namespace gardening_project_total_cost_l113_113169

theorem gardening_project_total_cost :
  let 
    num_rose_bushes := 20
    cost_per_rose_bush := 150
    gardener_hourly_rate := 30
    gardener_hours_per_day := 5
    gardener_days := 4
    soil_volume := 100
    soil_cost_per_cubic_foot := 5

    cost_of_rose_bushes := num_rose_bushes * cost_per_rose_bush
    gardener_total_hours := gardener_hours_per_day * gardener_days
    cost_of_gardener := gardener_hourly_rate * gardener_total_hours
    cost_of_soil := soil_volume * soil_cost_per_cubic_foot

    total_cost := cost_of_rose_bushes + cost_of_gardener + cost_of_soil
  in
    total_cost = 4100 := 
  by
    intros
    simp [num_rose_bushes, cost_per_rose_bush, gardener_hourly_rate, gardener_hours_per_day, gardener_days, soil_volume, soil_cost_per_cubic_foot]
    rw [mul_comm num_rose_bushes, mul_comm gardener_total_hours, mul_comm soil_volume]
    sorry -- place for proof steps

end gardening_project_total_cost_l113_113169


namespace eval_expression_solve_system_1_solve_system_2_l113_113489

-- First problem
theorem eval_expression : 
  (π - 3.14)^0 + real.sqrt 16 + abs (1 - real.sqrt 2) = 4 + real.sqrt 2 :=
sorry

-- Second problem
theorem solve_system_1 (x y : ℝ) :
  x - y = 2 ∧ 2 * x + 3 * y = 9 ↔ x = 3 ∧ y = 1 :=
sorry

-- Third problem
theorem solve_system_2 (x y : ℝ) :
  (5 * (x - 1) + 2 * y = 4 * (1 - y) + 3) ∧ (x / 3 + y / 2 = 1) ↔ x = 0 ∧ y = 2 :=
sorry

end eval_expression_solve_system_1_solve_system_2_l113_113489


namespace max_value_of_3cosx_minus_sinx_l113_113522

noncomputable def max_cosine_expression : ℝ :=
  Real.sqrt 10

theorem max_value_of_3cosx_minus_sinx : 
  ∃ x : ℝ, ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := 
by {
  sorry
}

end max_value_of_3cosx_minus_sinx_l113_113522


namespace total_pieces_of_art_l113_113425

-- Definitions for the conditions provided
variable (A : ℕ) 
def pieces_displayed := (1 / 3 : ℚ) * A
def sculptures_displayed := (1 / 6 : ℚ) * pieces_displayed
def pieces_not_displayed := (2 / 3 : ℚ) * A
def paintings_not_displayed := (1 / 3 : ℚ) * pieces_not_displayed
def sculptures_not_displayed := (2 / 3 : ℚ) * pieces_not_displayed

-- Given condition
axiom condition_not_displayed_sculptures : sculptures_not_displayed = 1200

-- Proof statement
theorem total_pieces_of_art : A = 2700 :=
by
  sorry

end total_pieces_of_art_l113_113425


namespace video_votes_count_l113_113473

noncomputable def total_votes (S : ℕ) (like_ratio dislike_ratio neutral_ratio : ℚ) : ℚ :=
  S / (like_ratio - dislike_ratio)

theorem video_votes_count (S : ℕ) (like_ratio dislike_ratio neutral_ratio : ℚ) :
  S = 180 → like_ratio = 0.60 → dislike_ratio = 0.20 → neutral_ratio = 0.20 →
  total_votes S like_ratio dislike_ratio neutral_ratio = 450 :=
by 
  intros hS hLike hDislike hNeutral,
  simp [total_votes, hS, hLike, hDislike],
  norm_num,
  sorry

end video_votes_count_l113_113473


namespace part_I_part_II_l113_113909

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) := abs (x + m) + abs (2 * x - 1)

-- Part (I)
theorem part_I (x : ℝ) : (f x (-1) ≤ 2) ↔ (0 ≤ x ∧ x ≤ (4 / 3)) :=
by sorry

-- Part (II)
theorem part_II (m : ℝ) : (∀ x, (3 / 4) ≤ x ∧ x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 :=
by sorry

end part_I_part_II_l113_113909


namespace min_diff_numbers_l113_113789

noncomputable def minNumbers {R : Type*} [Ring R] {P : R[X]} (degP : P.degree = 10) : ℕ :=
  10

theorem min_diff_numbers (P : Polynomial ℝ) (h : P.degree = 10) :
  minNumbers h = 10 := 
sorry

end min_diff_numbers_l113_113789


namespace no_partition_exists_l113_113642

theorem no_partition_exists :
  ¬ ∃ (S : Finset (Finset ℕ)), 
      S.card = 11 ∧
      (∀ s ∈ S, s.card = 3 ∧ (∃ a b c ∈ s, a + b = c)) ∧
      (⋃ s ∈ S, s) = {1, 2, 3, ..., 33} := 
by 
  sorry

end no_partition_exists_l113_113642


namespace avg_sq_feet_per_person_l113_113984

def population : ℕ := 30690000
def total_area_sq_miles : ℕ := 3855103
def sq_feet_per_sq_mile : ℕ := 5280 * 5280
def options := [1000000, 2000000, 3000000, 4000000, 5000000]

theorem avg_sq_feet_per_person :
  let total_sq_feet := total_area_sq_miles * sq_feet_per_sq_mile
  let avg_sq_feet_per_person := total_sq_feet / population
  abs (avg_sq_feet_per_person - 3000000) < abs (avg_sq_feet_per_person - option)
  for option in options
    if option ≠ 3000000
  := sorry

end avg_sq_feet_per_person_l113_113984


namespace discount_problem_exists_l113_113471

-- Define the problem conditions
def discount_constraints (x y : ℕ) : Prop := (1 ≤ x ∧ x < 10) ∧ (1 ≤ y ∧ y < 10) ∧ (100 * (x + y) - x * y = 1260)

-- Define the main theorem statement
theorem discount_problem_exists : ∃ x y : ℕ, discount_constraints x y :=
by { use [5, 8], sorry }

end discount_problem_exists_l113_113471


namespace geometric_sequence_first_term_l113_113070

theorem geometric_sequence_first_term (a : ℕ) (r : ℕ)
    (h1 : a * r^2 = 27) 
    (h2 : a * r^3 = 81) : 
    a = 3 :=
by
  sorry

end geometric_sequence_first_term_l113_113070


namespace equation_solution_l113_113033

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l113_113033


namespace solve_inequality_l113_113854

open Set

theorem solve_inequality :
  { x : ℝ | (2 * x - 2) / (x^2 - 5*x + 6) ≤ 3 } = Ioo (5/3) 2 ∪ Icc 3 4 :=
by
  sorry

end solve_inequality_l113_113854


namespace change_in_polynomial_l113_113040

variable {x b : ℝ}

theorem change_in_polynomial (h_pos : 0 < b) :
  let poly := 2 * x^2 - 5
      poly_incr := 2 * (x + b)^2 - 5
      poly_decr := 2 * (x - b)^2 - 5
  in (poly_incr - poly = 4 * b * x + 2 * b^2) ∨ (poly_decr - poly = -4 * b * x + 2 * b^2) :=
by
  let poly := 2 * x^2 - 5
  let poly_incr := 2 * (x + b)^2 - 5
  let poly_decr := 2 * (x - b)^2 - 5
  sorry

end change_in_polynomial_l113_113040


namespace remainder_zero_l113_113082

theorem remainder_zero (x : ℕ) (h1 : x = 1680) :
  (x % 5 = 0) ∧ (x % 6 = 0) ∧ (x % 7 = 0) ∧ (x % 8 = 0) :=
by
  sorry

end remainder_zero_l113_113082


namespace max_min_value_sum_l113_113393

theorem max_min_value_sum : 
  ∀ (M N : ℝ), 
    (∀ x ∈ Icc (0 : ℝ) 4, x + 2 ≤ M) ∧ (∃ x ∈ Icc (0 : ℝ) 4, M = x + 2) →
    (∀ x ∈ Icc (0 : ℝ) 4, x + 2 ≥ N) ∧ (∃ x ∈ Icc (0 : ℝ) 4, N = x + 2) →
    M + N = 8 :=
by
  sorry

end max_min_value_sum_l113_113393


namespace domain_of_f_l113_113856

def f (x : ℝ) : ℝ := x^(-0.2) + 2 * x^(0.5)

theorem domain_of_f : { x : ℝ | 0 < x } = set.Ioi 0 := 
sorry

end domain_of_f_l113_113856


namespace ratio_of_cows_sold_l113_113792

-- Condition 1: The farmer originally has 51 cows.
def original_cows : ℕ := 51

-- Condition 2: The farmer adds 5 new cows to the herd.
def new_cows : ℕ := 5

-- Condition 3: The farmer has 42 cows left after selling a portion of the herd.
def remaining_cows : ℕ := 42

-- Defining total cows after adding new cows
def total_cows_after_addition : ℕ := original_cows + new_cows

-- Defining cows sold
def cows_sold : ℕ := total_cows_after_addition - remaining_cows

-- The theorem states the ratio of 'cows sold' to 'total cows after addition' is 1 : 4
theorem ratio_of_cows_sold : (cows_sold : ℚ) / (total_cows_after_addition : ℚ) = 1 / 4 := by
  -- Proof would go here
  sorry


end ratio_of_cows_sold_l113_113792


namespace f_monotonically_increasing_g_max_min_l113_113898

noncomputable theory

-- Condition: Define the function f.
def f (x : ℝ) : ℝ := 2 * Real.sin(2 * x - Real.pi / 3)

-- Condition: Define the function g.
def g (x : ℝ) : ℝ := 2 * Real.sin(2 * x) + 1

-- Proof problem for monotonically increasing interval.
theorem f_monotonically_increasing (k : ℤ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12), 
  f.derivative x > 0 := 
sorry

-- Proof problem for maximum and minimum values of g on the given interval.
theorem g_max_min :
  ∀ x ∈ Set.Icc (- Real.pi / 12) (Real.pi / 3), 
  (g x ≤ 3 ∧ g x ≥ 0) :=
sorry

end f_monotonically_increasing_g_max_min_l113_113898


namespace count_pairs_sum_greater_than_100_l113_113157

theorem count_pairs_sum_greater_than_100 :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 100 }
  in (card { p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2 ∧ p.1 + p.2 > 100 }) = 2500 :=
sorry

end count_pairs_sum_greater_than_100_l113_113157


namespace range_of_a_l113_113285

/--
Given the parabola \(x^2 = y\), points \(A\) and \(B\) are on the parabola and located on both sides of the y-axis,
and the line \(AB\) intersects the y-axis at point \((0, a)\). If \(\angle AOB\) is an acute angle (where \(O\) is the origin),
then the real number \(a\) is greater than 1.
-/
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : (x1^2 = x2^2) → (x1 * x2 = -a) → ((-a + a^2) > 0) → (1 < a) :=
by 
  sorry

end range_of_a_l113_113285


namespace number_of_people_prefer_soda_l113_113321

-- Given conditions
def total_people : ℕ := 600
def central_angle_soda : ℝ := 198
def full_circle_angle : ℝ := 360

-- Problem statement
theorem number_of_people_prefer_soda : 
  (total_people : ℝ) * (central_angle_soda / full_circle_angle) = 330 := by
  sorry

end number_of_people_prefer_soda_l113_113321


namespace intersection_M_N_l113_113922

-- Defining the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | real.log x / real.log 2 < 1} -- using log base change property

-- The statement to prove that the intersection of M and N is as described
theorem intersection_M_N :
  (M ∩ N = {x | 0 < x ∧ x ≤ 1}) :=
  sorry

end intersection_M_N_l113_113922


namespace syllogism_problem_l113_113273

variable (Square Rectangle : Type) 
variable (is_rectangle : Square → Rectangle)
variable (equal_diagonals_square : ∀ (s : Square), (diagonals_equal s))
variable (equal_diagonals_rectangle : ∀ (r : Rectangle), (diagonals_equal r))

theorem syllogism_problem : 
  (equal_diagonals_rectangle (is_rectangle s)) ∧ (is_rectangle s) → (equal_diagonals_square s) :=
by
  sorry

end syllogism_problem_l113_113273


namespace true_proposition_l113_113256

noncomputable def f (a x : ℝ) : ℝ := a ^ x

def prop_p (a : ℝ) : Prop := 0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x < f a y

def prop_q : Prop := ∀ x : ℝ, (π / 4) < x ∧ x < (5 * π / 4) → sin x > cos x

theorem true_proposition (a : ℝ) (h : 0 < a ∧ a ≠ 1):
  (¬(prop_p a) ∧ prop_q) :=
sorry

end true_proposition_l113_113256


namespace negation_of_existential_l113_113918

theorem negation_of_existential:
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 = 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existential_l113_113918


namespace conditional_probability_l113_113254

-- Given probabilities:
def p_a : ℚ := 5/23
def p_b : ℚ := 7/23
def p_c : ℚ := 1/23
def p_a_and_b : ℚ := 2/23
def p_a_and_c : ℚ := 1/23
def p_b_and_c : ℚ := 1/23
def p_a_and_b_and_c : ℚ := 1/23

-- Theorem statement to prove:
theorem conditional_probability : p_a_and_b_and_c / p_a_and_c = 1 :=
by
  sorry

end conditional_probability_l113_113254


namespace compare_charges_l113_113996

/-
Travel agencies A and B have group discount methods with the original price being $200 per person.
- Agency A: Buy 4 full-price tickets, the rest are half price.
- Agency B: All customers get a 30% discount.
Prove the given relationships based on the number of travelers.
-/

def agency_a_cost (x : ℕ) : ℕ :=
  if 0 < x ∧ x < 4 then 200 * x
  else if x ≥ 4 then 100 * x + 400
  else 0

def agency_b_cost (x : ℕ) : ℕ :=
  140 * x

theorem compare_charges (x : ℕ) :
  (agency_a_cost x < agency_b_cost x -> x > 10) ∧
  (agency_a_cost x = agency_b_cost x -> x = 10) ∧
  (agency_a_cost x > agency_b_cost x -> x < 10) :=
by
  sorry

end compare_charges_l113_113996


namespace number_of_members_is_44_l113_113458

-- Define necessary parameters and conditions
def paise_per_rupee : Nat := 100

def total_collection_in_paise : Nat := 1936

def number_of_members_in_group (n : Nat) : Prop :=
  n * n = total_collection_in_paise

-- Proposition to prove
theorem number_of_members_is_44 : number_of_members_in_group 44 :=
by
  sorry

end number_of_members_is_44_l113_113458


namespace solution_l113_113942

noncomputable def prove_a_greater_than_3 : Prop :=
  ∀ (x : ℝ) (a : ℝ), (a > 0) → (|x - 2| + |x - 3| + |x - 4| < a) → a > 3

theorem solution : prove_a_greater_than_3 :=
by
  intros x a h_pos h_ineq
  sorry

end solution_l113_113942


namespace arithmetic_sequence_sum_l113_113731

-- Define arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℕ → ℕ) :=
  ∀ n, a (n + 1) = a n + d 1

def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Conditions from the problem
variables {a : ℕ → ℕ} {d : ℕ}

axiom condition : a 3 + a 7 + a 11 = 6

-- Definition of a_7 as derived in the solution
def a_7 : ℕ := 2

-- Proof problem equivalent statement
theorem arithmetic_sequence_sum : arithmetic_sum a d 13 = 26 :=
by
  -- These steps would involve setting up and proving the calculation details
  sorry

end arithmetic_sequence_sum_l113_113731


namespace fibonacci_inequality_l113_113700

open Finset
open BigOperators

-- Fibonacci sequence definition
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 1) + 1 => fib (n + 1) + fib n

-- Binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Statement of the problem
theorem fibonacci_inequality
    (n : ℕ)
    (hn : 0 < n) : 
    ∑ i in range (n + 1), binom n i * fib i < (2 * n + 2)^n / n! :=
sorry

end fibonacci_inequality_l113_113700


namespace linear_function_paralle_passes_through_point_l113_113716

theorem linear_function_paralle_passes_through_point :
  ∃ (k : ℝ) (b : ℝ), k = 2 ∧ ((-1, 1) ∈ set_of (λ (p : ℝ × ℝ), p.2 = 2 * p.1 + b)) ∧ (∀ x y, y = 2 * x + 3) :=
begin
  sorry
end

end linear_function_paralle_passes_through_point_l113_113716


namespace relation_between_x_and_y_l113_113311

open Real

noncomputable def x (t : ℝ) : ℝ := t^(1 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 1))

theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : (y t)^(x t) = (x t)^(y t) :=
by sorry

end relation_between_x_and_y_l113_113311


namespace bridge_length_l113_113812

theorem bridge_length (l1 : ℝ) (v1 : ℝ) (t : ℝ) (l2 : ℝ) (v2 : ℝ) :
  l1 = 175 → v1 = 60 * 1000 / 3600 → t = 45 →
  l2 = 125 → v2 = 50 * 1000 / 3600 →
  let relative_speed := v1 + v2 in
  let total_distance_covered := relative_speed * t in
  let bridge_length := total_distance_covered - (l1 + l2) in 
  bridge_length = 1075.2 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end bridge_length_l113_113812


namespace solve_for_x_l113_113382

theorem solve_for_x (x : ℝ) (hx₁ : x ≠ 3) (hx₂ : x ≠ -2) 
  (h : (x + 5) / (x - 3) = (x - 2) / (x + 2)) : x = -1 / 3 :=
by
  sorry

end solve_for_x_l113_113382


namespace amy_worked_hours_l113_113478

variable {h : ℕ} -- Number of hours as a natural number

-- Define the conditions
def hourly_wage := 2
def tips := 9
def total_earnings := 23

-- Define the equation representing the total earnings
def earnings_eq : Prop := hourly_wage * h + tips = total_earnings

-- The theorem to be proved
theorem amy_worked_hours : earnings_eq → h = 7 := by
  sorry

end amy_worked_hours_l113_113478


namespace tiling_impossible_if_b_odd_l113_113683

theorem tiling_impossible_if_b_odd 
  (m n b : ℕ) 
  (hm : 4 ∣ m ∨ 4 ∤ m)
  (hn : n % 2 = 1)
  (hb : b % 2 = 1) :
  ¬(∃ f : ℕ × ℕ → bool, ∀ i j, 
    (i + j) % 2 = (f (i, j) : ℕ) ∧ -- Checkerboard condition
    ((i < m ∧ j < n) ∨ (i < 2 * b ∧ j < n)) → 
    ((f (i, j) = f (i + 1, j) ∧ i + 1 < m) ∨ 
     (f (i, j) = f (i, j + b) ∧ j + b < n))) :=
by
  sorry

end tiling_impossible_if_b_odd_l113_113683


namespace scientific_notation_of_1_35_billion_l113_113332

theorem scientific_notation_of_1_35_billion :
  ∃ n : ℕ, 1.35 * 10^n = 1.35 * 10^9 :=
by
  use 9
  sorry

end scientific_notation_of_1_35_billion_l113_113332


namespace root_of_p_eq_root_of_q_l113_113861

def polynomial := Polynomial ℚ

def p : polynomial := (Polynomial.C 2) * Polynomial.X ^ 4 
                      + (Polynomial.C (-5)) * Polynomial.X ^ 3 
                      + (Polynomial.C (-7)) * Polynomial.X ^ 2 
                      + (Polynomial.C 34) * Polynomial.X 
                      + Polynomial.C (-24)

def q : polynomial := (Polynomial.C 2) * Polynomial.X ^ 3 
                      + (Polynomial.C (-3)) * Polynomial.X ^ 2 
                      + (Polynomial.C (-12)) * Polynomial.X 
                      + Polynomial.C 10

theorem root_of_p_eq_root_of_q (x : ℚ) : Polynomial.eval x p = 0 ↔ x = 1 ∨ Polynomial.eval x q = 0 := 
by
  sorry

end root_of_p_eq_root_of_q_l113_113861


namespace number_of_correct_conclusions_is_two_l113_113565

section AnalogicalReasoning
  variable (a b c : ℝ) (x y : ℂ)

  -- Condition 1: The analogy for distributive property over addition in ℝ and division
  def analogy1 : (c ≠ 0) → ((a + b) * c = a * c + b * c) → (a + b) / c = a / c + b / c := by
    sorry

  -- Condition 2: The analogy for equality of real and imaginary parts in ℂ
  def analogy2 : (x - y = 0) → x = y := by
    sorry

  -- Theorem stating that the number of correct conclusions is 2
  theorem number_of_correct_conclusions_is_two : 2 = 2 := by
    -- which implies that analogy1 and analogy2 are valid, and the other two analogies are not
    sorry

end AnalogicalReasoning

end number_of_correct_conclusions_is_two_l113_113565


namespace ellipse_major_minor_axis_condition_l113_113252

theorem ellipse_major_minor_axis_condition (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1) 
                                          (h2 : ∀ a b : ℝ, a = 2 * b) :
  m = 1 / 4 :=
sorry

end ellipse_major_minor_axis_condition_l113_113252


namespace find_g_2023_l113_113390

def g : ℕ → ℕ := sorry -- Function definition placeholder

theorem find_g_2023 (h1 : ∀ n, g(g(n)) = 3 * n) (h2 : ∀ n, g(3 * n + 2) = 3 * n + 1) : 
  g 2023 = 2019 :=
sorry

end find_g_2023_l113_113390


namespace log_f2_l113_113579

noncomputable def f (x : ℝ) : ℝ := x^(1/2)

theorem log_f2 :
  f (1/2) = (sqrt 2) / 2 → log 2 (f 2) = 1 / 2 :=
by
  intro h
  unfold f at *
  sorry

end log_f2_l113_113579


namespace quadrilateral_angles_inscribed_in_circle_l113_113049

theorem quadrilateral_angles_inscribed_in_circle
  (ABCD : Type)
  [convex_quadrilateral ABCD]
  (inscribed_in_circle : ABCD.inscribed_in_circle)
  (BD : diagonal ABCD)
  (angle_bisector_B : BD.angle_bisector (vertex B))
  (angle_BD_with_AC : BD.angle_with (diagonal AC) = 72)
  (angle_BD_with_AD : BD.angle_with (side AD) = 53) :
  (Quadrilateral.angles ABCD = (72, 110, 108, 70)) ∨
  (Quadrilateral.angles ABCD = (108, 38, 72, 142)) :=
sorry

end quadrilateral_angles_inscribed_in_circle_l113_113049


namespace find_trousers_l113_113313

variables (S T Ti : ℝ) -- Prices of shirt, trousers, and tie respectively
variables (x : ℝ)      -- The number of trousers in the first scenario

-- Conditions given in the problem
def condition1 : Prop := 6 * S + x * T + 2 * Ti = 80
def condition2 : Prop := 4 * S + 2 * T + 2 * Ti = 140
def condition3 : Prop := 5 * S + 3 * T + 2 * Ti = 110

-- Theorem to prove
theorem find_trousers : condition1 S T Ti x ∧ condition2 S T Ti ∧ condition3 S T Ti → x = 4 :=
by
  sorry

end find_trousers_l113_113313


namespace geometric_sequence_common_ratio_l113_113331

theorem geometric_sequence_common_ratio (a_1 q : ℝ) 
  (h1 : a_1 * q^2 = 9) 
  (h2 : a_1 * (1 + q) + 9 = 27) : 
  q = 1 ∨ q = -1/2 := 
by
  sorry

end geometric_sequence_common_ratio_l113_113331


namespace verify_total_cost_l113_113011

-- Define the ages and discounts
def ages := {Mrs_Lopez := 40, Husband := 40, Parents := (72, 75), Children := [3, 10, 14], Nephews := [6, 17], Aunt := 65, Friends := [56, 33], Sister := 40}
def adult_price := 11
def child_price := 8
def senior_price := 9

def discounts := {Husband := 0.25, Parents := 0.15, Nephew_17 := 0.10, Sister := 0.30, Aunt_BOGO := true}

-- Define the ticket price calculation after discounts
def ticket_cost (age : ℕ) (discount : ℕ → ℝ) : ℝ :=
  if age ≥ 60 then senior_price * (1 - discount age)
  else if age < 13 then child_price
  else adult_price * (1 - discount age)

-- Define the discount function
def discount_fn (ages : ℕ) : ℝ := 
  if ages = 40 then 0.25
  else if ages = 72 ∨ ages = 75 then 0.15
  else if ages = 17 then 0.10
  else if ages = 40 then 0.30
  else 0.0

-- Calculate the total cost for all tickets
def total_cost : ℝ := 
  let n_children := 3
  let n_adults := 1 + 1 + 2 + 1
  let n_seniors := 2 + 2
  let cost_mr_lopez_husband := ticket_cost 40 discount_fn + ticket_cost 40 discount_fn
  let cost_parents := ticket_cost 72 discount_fn + ticket_cost 75 discount_fn
  let cost_children_nephews := ticket_cost 3 discount_fn + ticket_cost 10 discount_fn + ticket_cost 14 discount_fn + ticket_cost 6 discount_fn + ticket_cost 17 discount_fn
  let cost_aunt := ticket_cost 65 discount_fn
  let cost_friends := ticket_cost 56 discount_fn + ticket_cost 33 discount_fn
  let real_cost_sister := ticket_cost 40 discount_fn
  cost_mr_lopez_husband - total_cost

-- Prove the total cost is $115.15
theorem verify_total_cost : total_cost 115.15 := by {
    sorry
}

end verify_total_cost_l113_113011


namespace P_one_div_P_neg_one_eq_l113_113066

-- Define the polynomial f(x) = x^2007 + 17*x^2006 + 1
def f (x : ℝ) := x^2007 + 17 * x^2006 + 1

-- Assume distinct zeroes r_1, r_2, ..., r_2007 of the polynomial f(x)
axiom distinct_roots : Π (j : ℕ), 1 ≤ j ∧ j ≤ 2007 → f (r j) = 0

-- Define the polynomial P of degree 2007 with the given property
noncomputable def P (x : ℝ) := (Π (j : ℕ), 1 ≤ j ∧ j ≤ 2007 → (x - (r j + 1 / (r j))))

-- The conjecture to be proved
theorem P_one_div_P_neg_one_eq : 
  ∀ (c : ℝ), 
    (∏ (j in finset.range 2007), (1 - (r j + 1 / (r j)))) =
    (∏ (j in finset.range 2007), (-1 - (r j + 1 / (r j)))) →
  ∃ (c : ℝ), P 1 / P (-1) = 289 / 259 := sorry

end P_one_div_P_neg_one_eq_l113_113066


namespace solve_for_x_l113_113695

theorem solve_for_x :
  ∃ x : ℝ, 2 ^ x + 10 = 4 * 2 ^ x - 34 ↔ x = Real.log2 (44 / 3) :=
by
  sorry

end solve_for_x_l113_113695


namespace exponent_properties_l113_113121

theorem exponent_properties {a b c : ℝ} :
  (9 ^ a * 9 ^ b) / 9 ^ c = 9 ^ (a + b - c) :=
by sorry

example : (9 ^ 5.6 * 9 ^ 10.3) / 9 ^ 2.56256 = 9 ^ 13.33744 :=
by
  have h : 13.33744 = 5.6 + 10.3 - 2.56256 := by norm_num
  rw [←h, exponent_properties]
  exact rfl

end exponent_properties_l113_113121


namespace negate_prop_l113_113586

theorem negate_prop (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) :
  ¬ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) ↔ ∃ x_0 : ℝ, 0 ≤ x_0 ∧ x_0 ≤ 2 * Real.pi ∧ |Real.sin x_0| > 1 :=
by sorry

end negate_prop_l113_113586


namespace number_of_glasses_l113_113593

theorem number_of_glasses (oranges_per_glass total_oranges : ℕ) 
  (h1 : oranges_per_glass = 2) 
  (h2 : total_oranges = 12) : 
  total_oranges / oranges_per_glass = 6 := by
  sorry

end number_of_glasses_l113_113593


namespace k_is_constant_k_value_l113_113882

-- Define the initial set of numbers
def initialNumbers := List.range' 1 15

-- Define the resulting number after all the operations
def finalNumber := 120

-- Sum of the first n natural numbers
def sum_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the squares of the first n natural numbers
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Sum of the cubes of the first n natural numbers
def sum_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

-- k is the sum of all expressions xy(x + y) performed by Bobby
def k : ℕ := sum_squares 15 ^ 2 - sum_squares 15 * sum_nat 15 + 2 * sum_cubes 15

theorem k_is_constant : (∀ ops : List (ℕ × ℕ), -- operations in any order
    (∀ (op ∈ ops), (op.fst ∈ initialNumbers ∧ op.snd ∈ initialNumbers ∧ op.fst ≠ op.snd)) →
    (∑ op in ops, op.fst * op.snd * (op.fst + op.snd)) = k) :=
sorry

theorem k_value : k = 49140 :=
sorry

end k_is_constant_k_value_l113_113882


namespace min_value_of_function_l113_113768

open Real

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  let y := -cos x ^ 2 - 2 * sin x + 4 in 
  ∃ (c : ℝ), y = (sin x - 1) ^ 2 + 2 ∧ y ≥ 2 :=
by
  sorry

end min_value_of_function_l113_113768


namespace ratio_pond_to_field_l113_113720

-- Definition of variables and given conditions
def width_of_field (l : ℝ) : ℝ := l / 2
def area_of_field (l w : ℝ) : ℝ := l * w
def area_of_pond (side_length : ℝ) : ℝ := side_length * side_length
def ratio_of_areas (area_pond area_field : ℝ) : ℝ := area_pond / area_field

-- Constants given in the problem
def length_of_field := 80.0
def side_length_of_pond := 8.0

theorem ratio_pond_to_field :
  ratio_of_areas (area_of_pond side_length_of_pond) (area_of_field length_of_field (width_of_field length_of_field)) = 1 / 50 :=
by
  sorry

end ratio_pond_to_field_l113_113720


namespace adult_ticket_cost_is_19_l113_113740

variable (A : ℕ) -- the cost for an adult ticket
def child_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400
def adults_attendance : ℕ := 280
def children_attendance : ℕ := 120

-- The equation representing the total receipts
theorem adult_ticket_cost_is_19 (h : total_receipts = 280 * A + 120 * child_ticket_cost) : A = 19 :=
  by sorry

end adult_ticket_cost_is_19_l113_113740


namespace spadesuit_calculation_l113_113869

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 2 (spadesuit 6 1) = -1221 := by
  sorry

end spadesuit_calculation_l113_113869


namespace ruler_perpendicular_line_exists_l113_113084

-- Definitions based on conditions.
def line3D := ℝ → ℝ × ℝ × ℝ
def line2D := ℝ → ℝ × ℝ
def perp_2d (l1 l2 : line2D) : Prop := ∃ (x1 y1 x2 y2 : ℝ), l1 = (λ t, (x1 * t + y1)) ∧ l2 = (λ t, (x2 * t + y2)) ∧ x1 * x2 + y1 * y2 = 0
def perp_3d_to_2d_projection (l1 : line3D) (l2 : line2D) (proj_line : line2D) : Prop :=
  ∃ (proj : ℝ × ℝ × ℝ → ℝ × ℝ), (∀ t, proj (l1 t) = proj_line t) ∧ perp_2d proj_line l2

-- Main statement.
theorem ruler_perpendicular_line_exists (l : line3D) : ∃ l' : line2D, perp_3d_to_2d_projection l l' (λ t, (0, t)) :=
sorry

end ruler_perpendicular_line_exists_l113_113084


namespace remaining_tickets_equation_l113_113483

-- Define the constants and variables
variables (x y : ℕ)

-- Conditions from the problem
def tickets_whack_a_mole := 32
def tickets_skee_ball := 25
def tickets_space_invaders : ℕ := x

def spent_hat := 7
def spent_keychain := 10
def spent_toy := 15

-- Define the condition for the total number of tickets spent
def total_tickets_spent := spent_hat + spent_keychain + spent_toy
-- Prove the remaining tickets equation
theorem remaining_tickets_equation : y = (tickets_whack_a_mole + tickets_skee_ball + tickets_space_invaders) - total_tickets_spent ->
                                      y = 25 + x :=
by
  sorry

end remaining_tickets_equation_l113_113483


namespace product_seq_geometric_sum_seq_geometric_l113_113053

variable {b : ℕ → ℝ}  -- Define the sequence b
variable {q : ℝ}     -- Define the common ratio q

-- Geometric sequence definition
def is_geometric_seq (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, b (n+1) = q * b n

-- Sequence based on product of three terms
def product_seq (b : ℕ → ℝ) : ℕ → ℝ :=
  λ n, b n * b (n+1) * b (n+2)

-- Sequence based on sum of three terms
def sum_seq (b : ℕ → ℝ) : ℕ → ℝ :=
  λ n, b n + b (n+1) + b (n+2)

-- The statement to be proven about the product sequence being geometric
theorem product_seq_geometric (h : is_geometric_seq b q) : 
  is_geometric_seq (product_seq b) (q^3) :=
by sorry

-- The statement to be proven about the sum sequence being geometric
theorem sum_seq_geometric (h : is_geometric_seq b q) : 
  is_geometric_seq (sum_seq b) q :=
by sorry

end product_seq_geometric_sum_seq_geometric_l113_113053


namespace sign_of_slope_equals_sign_of_correlation_l113_113699

-- Definitions for conditions
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ t, y t = a + b * x t

def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  r > -1 ∧ r < 1 ∧ ∀ t t', (y t - y t').sign = (x t - x t').sign

def regression_line_slope (b : ℝ) : Prop := True

-- Theorem to prove the sign of b is equal to the sign of r
theorem sign_of_slope_equals_sign_of_correlation (x y : ℝ → ℝ) (r b : ℝ) 
  (h1 : linear_relationship x y) 
  (h2 : correlation_coefficient x y r) 
  (h3 : regression_line_slope b) : 
  b.sign = r.sign := 
sorry

end sign_of_slope_equals_sign_of_correlation_l113_113699


namespace time_display_unique_digits_l113_113819

theorem time_display_unique_digits : 
  ∃ n : ℕ, n = 840 ∧ ∀ h : Fin 10, h = 5 →
  5 * 7 * 4 * 6 = n :=
by
  use 840
  simp
  sorry

end time_display_unique_digits_l113_113819


namespace find_y_in_triangle_problem_l113_113275

theorem find_y_in_triangle_problem
  (ABD_45_45_90 : is_45_45_90 △ABD)
  (AB_BD_eq_10 : length AB = 10 ∧ length BD = 10)
  (ACD_30_60_90 : is_30_60_90 △ACD)
  (angle_CAD_30 : angle CAD = 30) : y = 10 * real.sqrt 3 :=
sorry

end find_y_in_triangle_problem_l113_113275


namespace num_integers_between_200_and_250_with_increasing_digits_l113_113299

theorem num_integers_between_200_and_250_with_increasing_digits : 
  card {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ (n.digits).length = 3 
    ∧ (∀ i j, i < j → (n.digits.nth i < n.digits.nth j))} = 11 := 
sorry

end num_integers_between_200_and_250_with_increasing_digits_l113_113299


namespace rectangular_to_polar_coordinates_l113_113499

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end rectangular_to_polar_coordinates_l113_113499


namespace fraction_ratio_x_div_y_l113_113306

theorem fraction_ratio_x_div_y (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : y / (x + z) = (x - y) / z) 
(h5 : y / (x + z) = x / (y + 2 * z)) :
  x / y = 2 / 3 := 
  sorry

end fraction_ratio_x_div_y_l113_113306


namespace proof_problem_l113_113889

variable (p q : Prop)

-- Define the propositions
def prop_p : Prop := ∀ x y : ℝ, x < y → 2^x - 2^(-x) > 2^y - 2^(-y)
def prop_q : Prop := ∀ x y : ℝ, x < y → 2^x + 2^(-x) < 2^y + 2^(-y)

-- The proof problem statement
theorem proof_problem (h₁ : ¬prop_p) (h₂ : ¬prop_q) : (¬prop_p ∨ prop_q) :=
by {
  sorry
}

end proof_problem_l113_113889


namespace cos_value_correct_l113_113236

noncomputable def cos_value (α : ℝ) := cos (π / 3 - α)

theorem cos_value_correct (α : ℝ) (h : sin (π / 6 + α) = 1 / 3) :
  cos_value α = 1 / 3 :=
by
  sorry

end cos_value_correct_l113_113236


namespace range_of_m_l113_113577

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

-- Define the function g
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2)^2 + m

-- Statement to prove
theorem range_of_m (m : ℝ) :
  (∀ x₁ ∈ set.Icc (1 : ℝ) 2, ∃ x₂ ∈ set.Icc (-1) 1, f x₁ ≥ g x₂ m) → m ≤ 5 / 2 :=
begin
  sorry
end

end range_of_m_l113_113577


namespace solution_form_l113_113784

noncomputable def is_solution (N : ℕ) : Prop :=
  ∃ (a : ℕ), (a ∈ {1, 2, 3}) ∧ N = 125 * a * 10^197

theorem solution_form (N : ℕ) :
  (∃ (a m n k : ℕ), (a < 10) ∧ m < 10^k ∧ k = 199 ∧ N = m + 10^k * a + 10^(k+1) * n ∧
    m + 10^k * a + 10^(k+1) * n = 5 * (m + 10^(k+1) * n)) ↔ is_solution N :=
by
  sorry

end solution_form_l113_113784


namespace solution_form_l113_113783

noncomputable def is_solution (N : ℕ) : Prop :=
  ∃ (a : ℕ), (a ∈ {1, 2, 3}) ∧ N = 125 * a * 10^197

theorem solution_form (N : ℕ) :
  (∃ (a m n k : ℕ), (a < 10) ∧ m < 10^k ∧ k = 199 ∧ N = m + 10^k * a + 10^(k+1) * n ∧
    m + 10^k * a + 10^(k+1) * n = 5 * (m + 10^(k+1) * n)) ↔ is_solution N :=
by
  sorry

end solution_form_l113_113783


namespace remaining_number_is_six_l113_113018

-- Define the conditions as Lean definitions
def numbers_on_blackboard := list.range (2013) -- Natural numbers from 1 to 2012

def erase_and_add_units_digit (nums : list ℕ) : list ℕ :=
  sorry -- Function to represent the operation (to be implemented)

-- The problem statement in Lean
theorem remaining_number_is_six (h1 : ∃ n, n ∈ erase_and_add_units_digit (numbers_on_blackboard) ∧ n = 12)
  (h2 : list.length (erase_and_add_units_digit (numbers_on_blackboard)) = 2) :
  ∃ m, m ∈ erase_and_add_units_digit (numbers_on_blackboard) ∧ m = 6 :=
by
  sorry

end remaining_number_is_six_l113_113018


namespace inductive_inequality_l113_113751

theorem inductive_inequality (n : ℕ) (h1 : n > 1) :
  (∑ i in finset.range (2^n - 1 + 1), 1 / (i + 1) : ℝ) < n :=
sorry

end inductive_inequality_l113_113751


namespace find_ramanujan_number_l113_113324

noncomputable def hardy_number : ℂ := 3 + 7i
noncomputable def product : ℂ := 40 + 24i
noncomputable def ramanujan_number : ℂ := 4 - (104 / 29)*i

theorem find_ramanujan_number :
  hardy_number * ramanujan_number = product := 
by
  -- This is where the proof would be elaborated
  sorry

end find_ramanujan_number_l113_113324


namespace find_PC_length_l113_113955

constant Point : Type
constant Triangle : Type
constant similarity : Triangle → Triangle → Prop
constant length_of_side : Triangle → Point → Point → ℝ

axiom AB : Point
axiom BC : Point
axiom CA : Point
axiom P : Point
axiom A : Point
axiom B : Point
axiom C : Point
axiom T1 : Triangle
axiom T2 : Triangle

axiom T1_is_PAB : T1 = ⟨P, A, B⟩
axiom T2_is_PCA : T2 = ⟨P, C, A⟩

axiom similarity_condition : similarity T1 T2
axiom AB_length : length_of_side T1 A B = 8
axiom BC_length : length_of_side T1 B C = 7
axiom CA_length : length_of_side T1 C A = 6

theorem find_PC_length : length_of_side T2 P C = 9 := 
sorry

end find_PC_length_l113_113955


namespace range_of_m_condition_l113_113907

theorem range_of_m_condition (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ * x₁ - 2 * m * x₁ + m - 3 = 0) 
  (h₂ : x₂ * x₂ - 2 * m * x₂ + m - 3 = 0)
  (hx₁ : x₁ > -1 ∧ x₁ < 0)
  (hx₂ : x₂ > 3) :
  m > 6 / 5 ∧ m < 3 :=
sorry

end range_of_m_condition_l113_113907


namespace red_trace_larger_sphere_area_l113_113143

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end red_trace_larger_sphere_area_l113_113143


namespace dennis_taught_46_years_l113_113432

-- Variables for the number of years taught by Adrienne, Virginia, and Dennis.
variables (A V D : ℕ)

-- Conditions in the problem.
def condition1 : Prop := V + A + D = 102
def condition2 : Prop := V = A + 9
def condition3 : Prop := V = D - 9

-- The theorem to prove Dennis has taught for 46 years.
theorem dennis_taught_46_years (h1 : condition1) (h2 : condition2) (h3 : condition3) : D = 46 := 
sorry

end dennis_taught_46_years_l113_113432


namespace problem_statement_l113_113287

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 4 / Real.log 0.8
noncomputable def c : ℝ := 2 ^ 0.3

theorem problem_statement : c > a ∧ a > b :=
by
  sorry

end problem_statement_l113_113287


namespace range_of_a_if_sqrt2a_is_meaningful_l113_113307

variable (a : ℝ)

theorem range_of_a_if_sqrt2a_is_meaningful : (∃ r : ℝ, r = √(2 + a)) → a ≥ -2 :=
by sorry

end range_of_a_if_sqrt2a_is_meaningful_l113_113307


namespace proof_problem_l113_113681

theorem proof_problem 
  {a b c : ℝ} (h_cond : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (n : ℕ) :
  1/a^(2*n+1) + 1/b^(2*n+1) + 1/c^(2*n+1) = 1/(a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) :=
sorry

end proof_problem_l113_113681


namespace angle_QZY_l113_113974

theorem angle_QZY (P Q R S Z X Y : Point) 
  (hPQ_parallel_RS : parallel PQ RS) 
  (hZ_on_PQ : On Z PQ) 
  (hX_on_RS : On X RS) 
  (hY_between_PQ_RS : Between Y PQ RS)
  (hAngle_YXS : ∠ Y X S = 20)
  (hAngle_ZYX : ∠ Z Y X = 50) : 
  ∠ Q Z Y = 30 :=
sorry

end angle_QZY_l113_113974


namespace num_packs_blue_tshirts_l113_113834

def num_white_tshirts_per_pack : ℕ := 6
def num_packs_white_tshirts : ℕ := 5
def num_blue_tshirts_per_pack : ℕ := 9
def total_num_tshirts : ℕ := 57

theorem num_packs_blue_tshirts : (total_num_tshirts - num_white_tshirts_per_pack * num_packs_white_tshirts) / num_blue_tshirts_per_pack = 3 := by
  sorry

end num_packs_blue_tshirts_l113_113834


namespace minimize_regions_l113_113662

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem minimize_regions (c : ℝ) :
  (∀ (x : ℝ), f x = c + x → exists_finite_areas (f x (c + x))) →
  c = -11 / 27 :=
by
-- Dummy existence of finite areas function (for condition encapsulation)
def exists_finite_areas (graph1 : ℝ) (graph2 : ℝ) : Prop := sorry
sorry

end minimize_regions_l113_113662


namespace sum_series_l113_113487

theorem sum_series :
  ∑ n in Finset.range (99 - 1).succ.succ \ Finset.range 1, (1 : ℚ) / ((2 * (n + 1 : ℕ) - 1) * (2 * (n + 1 : ℕ) + 1)) = 33 / 201 := by
  sorry

end sum_series_l113_113487


namespace remainder_is_x_squared_l113_113860

noncomputable def remainder_of_poly_div : Polynomial ℝ :=
  let p : Polynomial ℝ := Polynomial.X ^ 1004
  let q : Polynomial ℝ := (Polynomial.X ^ 2 + 1) * (Polynomial.X - 1)
  let (_, r) := Polynomial.divMod p q
  r

theorem remainder_is_x_squared :
  remainder_of_poly_div = Polynomial.X ^ 2 :=
by
  sorry

end remainder_is_x_squared_l113_113860


namespace sufficient_but_not_necessary_l113_113780

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end sufficient_but_not_necessary_l113_113780


namespace angle_between_diagonals_l113_113064

variables (α β : ℝ) 

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_diagonals_l113_113064


namespace significant_relationship_exists_probability_is_correct_l113_113718

noncomputable def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![34, 16], ![10, 40]]

def N := 100
def chi_square_test_statistic : ℝ :=
  let a := contingency_table[0, 0]
  let b := contingency_table[0, 1]
  let c := contingency_table[1, 0]
  let d := contingency_table[1, 1]
  N * abs (a * d - b * c - N / 2)^2 /
  ((a + b) * (c + d) * (a + c) * (b + d))

def significant_relationship (chi_square_value : ℝ) : Prop :=
  chi_square_value > 6.635

theorem significant_relationship_exists :
  significant_relationship chi_square_test_statistic :=
by {
  -- This part uses the predetermined calculation and values
  have : chi_square_test_statistic = 23.377, sorry,
  show 23.377 > 6.635, from by norm_num,
  sorry
}

def C (n k : ℕ) : ℕ := Nat.choose n k

def probability_exactly_one_long_panicle (total_samples : ℕ) (long_panicle_plants : ℕ)
    (sample_size : ℕ) : ℝ :=
  let total_ways := C total_samples sample_size
  let favorable_ways := C long_panicle_plants 1 * C (total_samples - long_panicle_plants) 2
  favorable_ways / total_ways

theorem probability_is_correct :
  probability_exactly_one_long_panicle 5 1 3 = 2 / 5 :=
by {
  -- This part uses the predetermined calculation and values
  have : probability_exactly_one_long_panicle 5 1 3 = 4 / 10, sorry,
  show 4 / 10 = 2 / 5, from by norm_num,
  sorry
}

end significant_relationship_exists_probability_is_correct_l113_113718


namespace relationship_between_problems_geometry_problem_count_steve_questions_l113_113370

variable (x y W A G : ℕ)

def word_problems (x : ℕ) : ℕ := x / 2
def addition_and_subtraction_problems (x : ℕ) : ℕ := x / 3
def geometry_problems (x W A : ℕ) : ℕ := x - W - A

theorem relationship_between_problems :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x ∧
  G = geometry_problems x W A →
  W + A + G = x :=
by
  sorry

theorem geometry_problem_count :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x →
  G = geometry_problems x W A →
  G = x / 6 :=
by
  sorry

theorem steve_questions :
  y = x / 2 - 4 :=
by
  sorry

end relationship_between_problems_geometry_problem_count_steve_questions_l113_113370


namespace standard_deviation_transformed_data_l113_113608

variable {x : Fin 10 → ℝ}

-- Given Condition: Standard deviation of the sample data
def stddev_sample : ℝ := 2

-- Given assumption
def stddev (x : Fin 10 → ℝ) : ℝ := sorry  -- Define the standard deviation function (assuming necessary properties)

-- Required to prove: Standard deviation of the transformed data
theorem standard_deviation_transformed_data (h : stddev x = stddev_sample) :
    stddev (λ i, 2 * x i - 1) = 4 :=
sorry

end standard_deviation_transformed_data_l113_113608


namespace dog_travel_distance_l113_113456

noncomputable def total_distance_travelled_by_dog 
  (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  let D := (time * (speed1 * speed2)) / (speed1 + speed2) in
  D

theorem dog_travel_distance
  (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h_time : time = 2) 
  (h_speed1 : speed1 = 10) 
  (h_speed2 : speed2 = 5) : 
  total_distance_travelled_by_dog time speed1 speed2 = 40 / 3 := 
by
  unfold total_distance_travelled_by_dog
  rw [h_time, h_speed1, h_speed2]
  norm_num
  sorry

end dog_travel_distance_l113_113456


namespace x_eq_y_sufficient_not_necessary_abs_l113_113879

theorem x_eq_y_sufficient_not_necessary_abs (x y : ℝ) : (x = y → |x| = |y|) ∧ (|x| = |y| → x = y ∨ x = -y) :=
by {
  sorry
}

end x_eq_y_sufficient_not_necessary_abs_l113_113879


namespace frank_hamburger_goal_l113_113873

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal_l113_113873


namespace sum_seven_smallest_multiples_of_12_l113_113757

theorem sum_seven_smallest_multiples_of_12 :
  (Finset.sum (Finset.range 7) (λ n, 12 * (n + 1))) = 336 :=
by
  -- proof (sorry to skip)
  sorry

end sum_seven_smallest_multiples_of_12_l113_113757


namespace probability_f_ge_1_in_interval_l113_113569

-- Definitions and conditions
def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x
def interval_x : Set ℝ := Set.Icc 0 Real.pi

-- The probability question transformed into a proof problem
theorem probability_f_ge_1_in_interval : 
  (Set.countable_measure {x | f x ≥ 1 ∧ x ∈ interval_x}) / (Set.countable_measure interval_x) = (1 / 2) :=
by 
  sorry

end probability_f_ge_1_in_interval_l113_113569


namespace find_m_l113_113594

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -11.6 :=
sorry

end find_m_l113_113594


namespace two_digit_numbers_summing_to_143_l113_113839

-- We are to prove that the number of two-digit numbers which sum to 143 with their reverse number is 6.
theorem two_digit_numbers_summing_to_143 : 
  {n : ℕ | n / 10 + n % 10 = 13 ∧ n / 10 < 10 ∧ n / 10 > 0 ∧ n % 10 < 10}.card = 6 :=
sorry

end two_digit_numbers_summing_to_143_l113_113839


namespace avg_weight_section_B_l113_113403

theorem avg_weight_section_B 
  (W_B : ℝ) 
  (num_students_A : ℕ := 36) 
  (avg_weight_A : ℝ := 30) 
  (num_students_B : ℕ := 24) 
  (total_students : ℕ := 60) 
  (avg_weight_class : ℝ := 30) 
  (h1 : num_students_A * avg_weight_A + num_students_B * W_B = total_students * avg_weight_class) :
  W_B = 30 :=
sorry

end avg_weight_section_B_l113_113403


namespace mutual_exclusive_not_opposite_l113_113100

-- Define the sample space and events
def sample_space : set (set (ℕ)) := {{3, 0}, {2, 1}, {1, 2}, {0, 3}}

def at_least_two_white_balls : set (ℕ) := {2, 3}
def all_red_balls : set (ℕ) := {3}

-- Define the events
def mutually_exclusive (A B : set (ℕ)) : Prop :=
  ∀ a ∈ A, ∀ b ∈ B, a ≠ b

def not_opposite (A B : set (ℕ)) : Prop :=
  ∃ x ∈ sample_space, x ∉ A ∪ B

-- The proof problem
theorem mutual_exclusive_not_opposite :
  mutually_exclusive at_least_two_white_balls all_red_balls ∧ not_opposite at_least_two_white_balls all_red_balls :=
by {
  sorry
}

end mutual_exclusive_not_opposite_l113_113100


namespace solve_custom_inequality_l113_113530

def custom_op (a b : ℝ) : ℝ := a * b - a + b - 2

theorem solve_custom_inequality :
  ∀ x : ℝ, (3 ※ x < 2) → x = 1 :=
by
  intro x
  sorry

end solve_custom_inequality_l113_113530


namespace smallest_N_for_pairs_product_l113_113396

theorem smallest_N_for_pairs_product:
  ∃ N : ℕ, (∀ (a b : ℕ), 1 ≤ a → a ≤ 2016 → 1 ≤ b → b ≤ 2016 → (a * b ≤ N)) ∧
  ∀ M : ℕ, (∀ (a b : ℕ), 1 ≤ a → a ≤ 2016 → 1 ≤ b → b ≤ 2016 → (a * b ≤ M)) → N ≤ M :=
  ∃ N : ℕ, N = 1008 * 1009 := sorry

end smallest_N_for_pairs_product_l113_113396


namespace work_completion_time_l113_113774

theorem work_completion_time (a b c : ℝ) (ha : a = 16) (hc : c = 12) (habc : 1/16 + 1/b + 1/12 = 1/3.2) : b = 6 :=
by
  have h1 : 1/16 + 1/12 + 1/6 = 1/3.2, from sorry
  sorry

end work_completion_time_l113_113774


namespace categorize_numbers_l113_113849

def numbers := [6, -3, 2.4, -3/4, 0, -3.14, 2/9, 2, -7/2, -1.414, -17, 2/3]

def positive_numbers := [6, 2.4, 2/9, 2, 2/3]
def non_negative_integers := [6, 0, 2]
def integers := [6, -3, 0, 2, -17]
def negative_fractions := [-3/4, -3.14, -7/2, -1.414]

theorem categorize_numbers :
  ∀ x ∈ numbers,
    (x > 0 ↔ x ∈ positive_numbers) ∧
    (x ≥ 0 ∧ x % 1 = 0 ↔ x ∈ non_negative_integers) ∧
    (x % 1 = 0 ↔ x ∈ integers) ∧
    (x < 0 ∧ x % 1 ≠ 0 ↔ x ∈ negative_fractions) := by
  sorry

end categorize_numbers_l113_113849


namespace triangle_AC_length_l113_113953

-- Define the conditions in the problem
def AB : ℝ := real.sqrt 2
def angle_B : ℝ := 60 * real.pi / 180 -- converting degrees to radians
def area_S : ℝ := (real.sqrt 3 + 3) / 4

-- Define the lengths of sides and their relationships
def AC := real.sqrt 3

-- The main statement to be proven
theorem triangle_AC_length (AB_eq : AB = real.sqrt 2) (angle_B_eq : angle_B = 60 * real.pi / 180)
  (area_S_eq : area_S = (real.sqrt 3 + 3) / 4) : 
  AC = real.sqrt 3 :=
sorry

end triangle_AC_length_l113_113953


namespace person_B_days_l113_113747

theorem person_B_days (A_days : ℕ) (combined_work : ℚ) (x : ℕ) : 
  A_days = 30 → combined_work = (1 / 6) → 3 * (1 / 30 + 1 / x) = combined_work → x = 45 :=
by
  intros hA hCombined hEquation
  sorry

end person_B_days_l113_113747


namespace distribute_teachers_l113_113197

theorem distribute_teachers :
  let schools := {A, B, C, D}
  let teachers := 6
  let min_teachers_A := 2
  let min_teachers_B := 1
  let min_teachers_C := 1
  let min_teachers_D := 1
  ∃ (distribution : (ℕ × ℕ × ℕ × ℕ)),
    (distribution.1 + distribution.2 + distribution.3 + distribution.4 = teachers) ∧
    (distribution.1 ≥ min_teachers_A) ∧
    (distribution.2 ≥ min_teachers_B) ∧
    (distribution.3 ≥ min_teachers_C) ∧
    (distribution.4 ≥ min_teachers_D) ∧
    (∃ ways : ℕ, ways = 660) :=
sorry

end distribute_teachers_l113_113197


namespace cost_of_gravel_path_l113_113139

/-
A rectangular grassy plot 150 m by 95 m has a gravel path 4.5 m wide all round it on the inside.
Find the cost of gravelling the path at 90 paise per sq. metre.
-/

theorem cost_of_gravel_path
  (length plot_length : ℝ) (width plot_width : ℝ) (path_width : ℝ) (cost_per_sq_meter_paise : ℝ) :
  plot_length = 150 →
  plot_width = 95 →
  path_width = 4.5 →
  cost_per_sq_meter_paise = 90 →
  let full_length := plot_length + 2 * path_width,
      full_width := plot_width + 2 * path_width,
      plot_area := plot_length * plot_width,
      full_area := full_length * full_width,
      path_area := full_area - plot_area,
      cost_per_sq_meter := cost_per_sq_meter_paise / 100,
      total_cost := path_area * cost_per_sq_meter in
  total_cost = 2057.40 :=
begin
  intros,
  sorry
end

end cost_of_gravel_path_l113_113139


namespace count_repeating_decimals_in_range_l113_113527

theorem count_repeating_decimals_in_range :
  let has_repeating_decimal (n : ℕ) := 
    ∃ p, prime p ∧ ¬(p = 2 ∨ p = 5) ∧ (n + 1) % p = 0
  in nat.card {n // 1 ≤ n ∧ n ≤ 150 ∧ has_repeating_decimal n} = 135 :=
by
  let has_repeating_decimal (n : ℕ) :=
    ∃ p, prime p ∧ ¬(p = 2 ∨ p = 5) ∧ (n + 1) % p = 0
  sorry

end count_repeating_decimals_in_range_l113_113527


namespace angle_equality_l113_113640

noncomputable def quadrilateral (A B C D M : Type) : Prop :=
  parallelogram A B M D ∧ ∠C B M = ∠C D M 

theorem angle_equality (A B C D M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] 
  (h : quadrilateral A B C D M) : ∠A C D = ∠B C M :=
by 
  cases h with _ angle_eq,
  sorry

end angle_equality_l113_113640


namespace find_s12_l113_113191

-- Variables and Assumptions
variables (d : ℕ) (s24 s12 : ℕ) (t24 t12 : ℕ)
variable  (h1 : d = 200)
variable  (h2 : s24 = 50)
variable  (h3 : t12 = t24 + 6)
variable  (h4 : t24 = d / s24)

-- Statement of the problem
theorem find_s12 : s12 = 20 :=
by
  have t24_val : t24 = 4 := by rw [← h4, h1, h2]; norm_num
  have t12_val : t12 = 10 := by rw [h3, t24_val]; norm_num
  have s12_val : s12 = d / t12 := by rw [← h1]
  rw t12_val at s12_val
  norm_num at s12_val
  assumption

end find_s12_l113_113191


namespace standard_equation_of_parabola_line_NQ_passes_through_fixed_point_l113_113284

theorem standard_equation_of_parabola (k p : ℝ) (h₀ : p > 0) (h₁ : k = 1 / 2) (h₂ : ∃ M N : ℝ×ℝ, ∃ q : ℝ, 
  (M ≠ N ∧ 
  M.1 = q ∧ M.2 = 1/2 * (q + p/4) ∧ 
  N.1 = -q ∧ N.2 = 1/2 * (-q + p/4) ∧
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4 * 15
  )) 
  : ∃ (p : ℝ), p = 2 ∧ (∀ x y : ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) :=
begin
  sorry
end

theorem line_NQ_passes_through_fixed_point (p M N Q : ℝ×ℝ) (B : ℝ×ℝ := (1, -1)) (h₀ : p > 0) (h₁ : M.2 = 1/2 * M.1 + p/4) 
  (h₂ : N.2 = 1/2 * N.1 + p/4) (h₃ : M ≠ N) (h₄ : (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4*15) 
  (h₅ : M.2 ≠ 0 ∧ (B.1 - M.1) ∗ (Q.2 - M.2) = (B.2 - M.2) * (Q.1 - M.1)) 
  (h₆ : Q.2 = 1/2 * Q.1 + p/4) :
  (∀ x y : ℝ, x=1 ∧ y=-4) :=
begin
  sorry
end

end standard_equation_of_parabola_line_NQ_passes_through_fixed_point_l113_113284


namespace distance_from_cheese_to_tree_is_30_l113_113424

-- Given conditions
variables {dist_cheese_to_smaller_tree : ℝ}

-- The problem statement premise:
def equal_right_angled_triangles_formed : Prop :=
(dist_cheese_to_smaller_tree = 30) ∧ 
(forall (x y : ℝ), x ≠ y → x^2 + y^2 = 2 * (30^2))

-- The proof statement to show the distance is indeed 30 meters
theorem distance_from_cheese_to_tree_is_30 :
  equal_right_angled_triangles_formed → dist_cheese_to_smaller_tree = 30 :=
begin
  sorry
end

end distance_from_cheese_to_tree_is_30_l113_113424


namespace area_convex_quad_inequality_l113_113684

theorem area_convex_quad_inequality
  {A B C D : Type*}
  (is_convex : convex_quadrilateral A B C D)
  (AB BC AD DC : ℝ)
  (hAB : line_segment A B)
  (hBC : line_segment B C)
  (hAD : line_segment A D)
  (hDC : line_segment D C) :
  area_quadrilateral A B C D <= (1 / 2) * (AB * BC + AD * DC) := 
sorry

end area_convex_quad_inequality_l113_113684


namespace not_prime_abs_diff_l113_113643

theorem not_prime_abs_diff (a b : ℕ) (x y : ℕ)
  (h1 : a + b = x^2) (h2 : a * b = y^2) : ¬ Nat.Prime (|16 * a - 9 * b|) :=
sorry

end not_prime_abs_diff_l113_113643


namespace triangle_not_isosceles_l113_113587

variable {α : Type} [LinearOrder α]

def is_triangle (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def distinct (a b c : α) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_isosceles (a b c : α) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem triangle_not_isosceles {a b c : α} (h1 : is_triangle a b c) (h2 : distinct a b c) :
  ¬ is_isosceles a b c :=
by
  sorry

end triangle_not_isosceles_l113_113587


namespace exists_two_among_seven_l113_113680

theorem exists_two_among_seven (A : Fin 7 → ℝ) :
  ∃ (a b : Fin 7), a ≠ b ∧ (√3 * |A a - A b| ≤ |1 + A a * A b|) :=
begin
  sorry
end

end exists_two_among_seven_l113_113680


namespace cubic_inequality_solution_l113_113870

theorem cubic_inequality_solution (x : ℝ) : x^3 - 12 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (9 < x) :=
by sorry

end cubic_inequality_solution_l113_113870


namespace average_of_subsets_power_l113_113030

theorem average_of_subsets_power (n : ℕ) : ∃ X : Finset ℕ, 
  (X.card = n ∧ (∀ (S : Finset ℕ), S ⊆ X → 
  (S.sum / S.card : ℚ) ∈ {x : ℚ | ∃ m : ℕ, x = m^2 ∨ x = m^3 ∨ (∃ k > 3, x = m^k)}) 
  ∧ (∀ x ∈ X, x > 1)) :=
sorry

end average_of_subsets_power_l113_113030


namespace find_other_endpoint_l113_113060

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end find_other_endpoint_l113_113060


namespace max_trees_in_equilateral_triangles_l113_113779

-- Define the types of trees
inductive TreeType
| Apple
| Pear
| Plum
| Apricot
| Cherry
| Almond

open TreeType

-- Define vertices of equilateral triangles
structure TriangleVertex where
  t1 t2 t3 : TreeType
  h : t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3

-- Define the main theorem to be proven
theorem max_trees_in_equilateral_triangles : 
  ∃ (v1 v2 v3 v4 v5 v6 : TriangleVertex), by sorry := TreeType.Apple ∧
  by sorry := TreeType.Pear ∧ 
  by sorry := TreeType.Plum ∧
  by sorry := TreeType.Apricot ∧
  by sorry := TreeType.Cherry ∧
  by sorry := TreeType.Almond := 
by sorry

end max_trees_in_equilateral_triangles_l113_113779


namespace find_values_of_a_and_b_find_square_root_l113_113552

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end find_values_of_a_and_b_find_square_root_l113_113552


namespace find_number_l113_113786

theorem find_number (N : ℕ) (hN : (200 ≤ Nat.digits 10 N.length ∧ Nat.digits 10 N.length ≤ 200)) :
  (∃ a : ℕ, a ∈ {1, 2, 3} ∧ N = 125 * a * 10 ^ 197) :=
by
  sorry

end find_number_l113_113786


namespace lightbulb_stops_on_friday_l113_113446

theorem lightbulb_stops_on_friday
  (total_hours : ℕ) (daily_usage : ℕ) (start_day : ℕ) (stops_day : ℕ)
  (h_total_hours : total_hours = 24999)
  (h_daily_usage : daily_usage = 2)
  (h_start_day : start_day = 1) : 
  stops_day = 5 := by
  sorry

end lightbulb_stops_on_friday_l113_113446


namespace water_wasted_in_one_hour_l113_113710

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l113_113710


namespace wire_not_used_is_20_l113_113202

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l113_113202


namespace min_distance_MN_l113_113979

noncomputable def polar_eq_to_rect_eq (θ : ℝ) : ℝ := 
  4 / (Real.sin θ ^ 2)

def rect_eq (x y : ℝ) : Prop :=
  y^2 = 4 * x

def line1 (α t : ℝ) : (ℝ × ℝ) :=
  (1 + t * Real.cos α, t * Real.sin α)

def line2 (α t : ℝ) : (ℝ × ℝ) :=
  (1 - t * Real.sin α, t * Real.cos α)

theorem min_distance_MN (α : ℝ) : 
  let t_M := (2 * (Real.cos α) / (Real.sin α)^2) in
  let t_N := -(2 * (Real.sin α) / (Real.cos α)^2) in
  t_M ^ 2 + t_N ^ 2 = (4 * 2 ^ (1/2)) := 
begin
  sorry
end

end min_distance_MN_l113_113979


namespace smallest_period_abs_sin_l113_113875

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem smallest_period_abs_sin :
  is_periodic (λ x, f x) π :=
sorry

end smallest_period_abs_sin_l113_113875


namespace smallest_n_value_existence_l113_113153

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l113_113153


namespace parabola_is_x2_4y_coordinates_of_point_H_l113_113584

noncomputable def parabola := { p : ℝ // p > 0 }
def point := ℝ × ℝ
def line := ℝ → ℝ

-- Define the given conditions
variables (p : parabola) (D F P Q A B M H: point)

-- Given:
-- 1. The parabola is \( x^2 = 2py \).
def parabola_eq (p : parabola) (x y : ℝ) : Prop := x^2 = 2 * p.1 * y
-- 2. The tangent line \( l \) at \( P \).
def tangent_at_P (P : point) (l : line) : Prop := ∀ x, l x = (P.1 / p.1) * x - (P.1^2 / (2 * p.1))
-- 3. \( |FD| = 2 \).
def distance_FD (F D : point) : Prop := dist F D = 2
-- 4. \( \angle PFD = 60^\circ \).
def angle_PFD (P F D : point) : Prop := ∠ F P D = real.pi / 3 -- 60 degrees in radians
-- 5. Given that \( A \) and \( B \) on parabola \( C \), satisfying vector sum condition.
-- 6. Point \( M(2, 2) \)
def point_M : point := (2, 2)
-- 7. There exists a point \( H \) on parabola \( C \) such that the circle passing through \( A \), \( B \), \( H \) has the same tangent at \( H \) as the parabola.

-- Prove:
-- 1. The equation of the parabola \( C \) is \( x^2 = 4y \).
theorem parabola_is_x2_4y : 
  (∀ x y, parabola_eq ⟨2, by sorry⟩ x y) :=
begin
  sorry
end

-- 2. The coordinates of point \( H \) are \( (-2, 1) \).
theorem coordinates_of_point_H :
  H = (-2, 1) :=
begin
  sorry
end

end parabola_is_x2_4y_coordinates_of_point_H_l113_113584


namespace compute_series_sum_l113_113494

theorem compute_series_sum :
  ∑' n : ℕ in set.Ici 2, (3 * n^3 - 2 * n^2 - 2 * n + 3 : ℝ) / (n^6 - n^5 + n^3 - n^2 + n - 1) = 1 := 
sorry

end compute_series_sum_l113_113494


namespace bruce_total_payment_correct_l113_113827

theorem bruce_total_payment_correct : 
  let cost_grapes := 7 * 70
  let cost_mangoes := 9 * 55 in
  cost_grapes + cost_mangoes = 985 :=
by
  -- Definitions of costs based on the conditions
  let cost_grapes := 7 * 70
  let cost_mangoes := 9 * 55
  calc
    cost_grapes + cost_mangoes = 490 + 495      : by rfl
                          ... = 985              : by rfl

end bruce_total_payment_correct_l113_113827


namespace bread_slices_remaining_l113_113012

theorem bread_slices_remaining 
  (total_slices : ℕ)
  (third_eaten: ℕ)
  (slices_eaten_breakfast : total_slices / 3 = third_eaten)
  (slices_after_breakfast : total_slices - third_eaten = 8)
  (slices_used_lunch : 2)
  (slices_remaining : 8 - slices_used_lunch = 6) : 
  total_slices = 12 → third_eaten = 4 → slices_remaining = 6 := by 
  sorry

end bread_slices_remaining_l113_113012


namespace solve_for_x_l113_113944

theorem solve_for_x (x : Real) (h : 625 ^ -x + 25 ^ -(2 * x) + 5 ^ -(4 * x) = 11) :
  x = (Real.log 11 - Real.log 3) / (-4 * Real.log 5) :=
sorry

end solve_for_x_l113_113944


namespace coefficient_equals_25_implies_m_eq_neg5_l113_113265

theorem coefficient_equals_25_implies_m_eq_neg5 :
  (∃ (m : ℝ), ∀ (r : ℕ),
    (binom 5 r) * (-m)^r * x^((5 - 2 * r) / 2) = 25 -> m = -5) :=
by
  sorry

end coefficient_equals_25_implies_m_eq_neg5_l113_113265


namespace gain_percentage_l113_113459

-- Define the conditions as a Lean problem
theorem gain_percentage (C G : ℝ) (hC : (9 / 10) * C = 1) (hSP : (10 / 6) = (1 + G / 100) * C) : 
  G = 50 :=
by
-- Here, you would generally have the proof steps, but we add sorry to skip the proof for now.
sorry

end gain_percentage_l113_113459


namespace f_even_and_periodic_l113_113280

def f (x : ℝ) : ℝ := Real.sin (π * x + π / 2)

theorem f_even_and_periodic :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 2) = f x) := 
sorry

end f_even_and_periodic_l113_113280


namespace laborers_present_l113_113077

-- Define the total number of laborers
def total_laborers : ℕ := 26

-- Define the percentage of laborers that showed up
def percentage_present : ℚ := 38.5 / 100

-- Define the computed number of laborers that showed up
noncomputable def computed_present : ℚ := percentage_present * total_laborers

-- Define the number of laborers present as an integer, rounded to the nearest whole number
def number_present : ℕ := 10

-- The theorem stating that the number of laborers present is 10
theorem laborers_present : number_present = computed_present.round :=
by
  sorry

end laborers_present_l113_113077


namespace movie_book_difference_l113_113076

def num_movies : ℕ := 17
def num_books : ℕ := 11
def diff_movies_books : ℕ := num_movies - num_books

theorem movie_book_difference : diff_movies_books = 6 := by
  -- conditions
  let movies := num_movies
  let books := num_books
  -- test the difference
  have equation : movies - books = 6, from
    calc
      movies - books
          = 17 - 11   : by rw [num_movies, num_books]
      ... = 6         : by norm_num
  -- complete the proof
  exact equation

end movie_book_difference_l113_113076


namespace water_wasted_in_one_hour_l113_113711

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l113_113711


namespace min_value_of_expression_l113_113116

noncomputable def min_expression_value (a b c d : ℝ) : ℝ :=
  (a ^ 8) / ((a ^ 2 + b) * (a ^ 2 + c) * (a ^ 2 + d)) +
  (b ^ 8) / ((b ^ 2 + c) * (b ^ 2 + d) * (b ^ 2 + a)) +
  (c ^ 8) / ((c ^ 2 + d) * (c ^ 2 + a) * (c ^ 2 + b)) +
  (d ^ 8) / ((d ^ 2 + a) * (d ^ 2 + b) * (d ^ 2 + c))

theorem min_value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_expression_value a b c d = 1 / 2 :=
by
  -- Proof is omitted.
  sorry

end min_value_of_expression_l113_113116


namespace triangular_region_properties_l113_113813

def line1 (x : ℝ) := 2*x + 4
def line2 (x : ℝ) := -3*x + 9
def line3 (y : ℝ) := 2

def intersection1 : ℝ × ℝ := (-1, 2)
def intersection2 : ℝ × ℝ := (7/3, 2)
def intersection3 : ℝ × ℝ := (1, 6)

-- Using vertices of the triangle
def vertex1 : ℝ × ℝ := (-1, 2)
def vertex2 : ℝ × ℝ := (7/3, 2)
def vertex3 : ℝ × ℝ := (1, 6)

-- Function to calculate the area of the triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((p1.1 * (p2.2 - p3.2)) + (p2.1 * (p3.2 - p1.2)) + (p3.1 * (p1.2 - p2.2)))

-- The point P, equidistant from the two intersections on line3
def point_P : ℝ × ℝ := (2/3, 2)

-- Proof statement, area and coordinates of point P
theorem triangular_region_properties :
  triangle_area vertex1 vertex2 vertex3 = 6.67 ∧ point_P = (2/3, 2) :=
by sorry

end triangular_region_properties_l113_113813


namespace parallel_numeric_value_l113_113057

def letter_value (n : ℕ) : ℤ :=
  match n % 6 with
  | 0 => -1
  | 1 => 2
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -2
  | _ => 0 -- this case will never occur due to modulo 6
  end

def letter_position (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- default case for non-alphabet characters
  end

def word_value (word : String) : ℤ :=
  word.to_list.map (fun ch => letter_value (letter_position ch)).sum

theorem parallel_numeric_value :
  word_value "PARALLEL" = -2 :=
by
  sorry

end parallel_numeric_value_l113_113057


namespace div_fact_l113_113488

-- Conditions
def fact_10 : ℕ := 3628800
def fact_4 : ℕ := 4 * 3 * 2 * 1

-- Question and Correct Answer
theorem div_fact (h : fact_10 = 3628800) : fact_10 / fact_4 = 151200 :=
by
  sorry

end div_fact_l113_113488


namespace rotation_120_moves_shapes_l113_113134

def shape_config_initial : list string := ["rectangle", "smaller_circle", "square"]
def rotate_circle (config : list string) : list string := 
  [config.get! 1, config.get! 2, config.get! 0]  -- 120 degree clockwise rotation shifts

theorem rotation_120_moves_shapes :
  rotate_circle shape_config_initial = ["smaller_circle", "square", "rectangle"] :=
by
  unfold shape_config_initial rotate_circle
  sorry

end rotation_120_moves_shapes_l113_113134


namespace LaMar_eats_one_sixth_l113_113091

theorem LaMar_eats_one_sixth (T : ℚ) (M : ℚ) (L : ℚ)
  (hT : T = 1/2) (hM : M = 1/3) (total : T + M + L = 1) : 
  L = 1/6 :=
by
  have h₁ : T + M = 5/6,
  { rw [hT, hM],
    norm_num },
  have h₂ : 1 - (T + M) = 1 - (5/6),
  { rw h₁ },
  norm_num at h₂,
  simp [total, h₁] at h₂ ⊢,
  symmetry,
  exact h₂

end LaMar_eats_one_sixth_l113_113091


namespace find_value_of_a_l113_113267

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x^2 - 3 * a * Real.sin (π * x / 2)
else 0

theorem find_value_of_a (a : ℝ) : 
  (∀ x, f (-x) a = -f x a) ∧ f 3 a = 6 → a = 5 :=
by
  sorry

end find_value_of_a_l113_113267


namespace probability_divisible_l113_113737

noncomputable def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

def total_combinations (s : Finset ℕ) : ℕ := (s.card.choose 3)

def successful_outcomes (s : Finset ℕ) : ℕ := 
  s.subsets 3 |>.filter (fun s => 
    let x := s.min' (by simp [Finset.nonempty.subset, Finset.nonempty_of_mem]; tautology)
    all_b (s.erase x) (fun y => x ∣ y)).card

def probability (s : Finset ℕ) : ℚ := successful_outcomes s / total_combinations s

theorem probability_divisible (s : Finset ℕ) (h : s = number_set) : 
  probability s = 11 / 20 := by
  sorry

end probability_divisible_l113_113737


namespace locus_of_altitudes_l113_113539

-- Definitions related to the geometry of the problem
variables (A B C D H L N : Type*)

-- Given conditions as Lean variables
variable (ABC : Triangle A B C)
variable (D_on_perpendicular : Point_on_Perpendicular A  -- Condition D is on the perpendicular line through A 
                                                       (Triangle_Plane ABC)) 

variable (H_orthocenter : Orthocenter H ABC)  -- H is the orthocenter of triangle ABC
variable (L_altitude_foot : AltitudeFoot A B C L) -- L is the foot of the altitude from A onto BC
variable (N_orthocenter_DBC : Orthocenter N (Triangle D B C)) -- N is the orthocenter of triangle DBC

-- Statement about the locus of points N
theorem locus_of_altitudes : 
  Locus N_circle : Circle_with_diameter H L
    (orthocenter_locus D_on_perpendicular DABC) := sorry

end locus_of_altitudes_l113_113539


namespace BM_passes_through_fixed_point_l113_113938

noncomputable def ellipse_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

theorem BM_passes_through_fixed_point (N M B : ℝ × ℝ) (k : ℝ) :
  let ellipse_eq := λ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 in
  ellipse_eq B.1 B.2 →
  ellipse_eq M.1 M.2 →
  N = (4, 0) →
  ∃ x_0 : ℝ, x_0 = 1 ∧
  ∀ (x_k : ℝ), (y : ℝ), y = k * (x_k - x_0) → ellipse_eq x_k y → (B.1, B.2) = (x_0, 0) :=
sorry

end BM_passes_through_fixed_point_l113_113938


namespace geometric_sequence_k_value_l113_113335

theorem geometric_sequence_k_value
  (k : ℤ)
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = 3 * 2^n + k)
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1))
  (h3 : ∃ r, ∀ n, a (n + 1) = r * a n) : k = -3 :=
sorry

end geometric_sequence_k_value_l113_113335


namespace domain_of_f_l113_113097

noncomputable def f (x : ℝ) : ℝ := log 2 (log 3 (log 5 x))

theorem domain_of_f :
  {x : ℝ | 5 < x} = {x : ℝ | ∃ y, y = f(x)} :=
by
  sorry

end domain_of_f_l113_113097


namespace parabola_standard_equation_l113_113556

/-- Given that the directrix of a parabola coincides with the line on which the circles 
    x^2 + y^2 - 4 = 0 and x^2 + y^2 + y - 3 = 0 lie, the standard equation of the parabola 
    is x^2 = 4y.
-/
theorem parabola_standard_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 = 0 → x^2 + y^2 + y - 3 = 0 → y = -1) →
  ∀ p : ℝ, 4 * (p / 2) = 4 → x^2 = 4 * p * y :=
by
  sorry

end parabola_standard_equation_l113_113556


namespace blocks_for_tower_l113_113376

theorem blocks_for_tower (total_blocks : ℕ) (house_blocks : ℕ) (extra_blocks : ℕ) (tower_blocks : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : house_blocks = 20) 
  (h3 : extra_blocks = 30) 
  (h4 : tower_blocks = house_blocks + extra_blocks) : 
  tower_blocks = 50 :=
sorry

end blocks_for_tower_l113_113376


namespace inscribed_circle_divides_AP_l113_113627

def isosceles_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ (a b c : ℝ), a = b ∧ (a^2 + b^2 = c^2)

noncomputable def inscribed_circle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  (h : isosceles_right_triangle A B C) : Type :=
  {center : Type // ∃ (radius : ℝ), true}

noncomputable def point_on_segment (A B : Type) [metric_space A] [metric_space B] := A 

noncomputable def intersection_point (circle : Type) [metric_space circle] (segment : Type) [metric_space segment] := circle

def height (A B : Type) [metric_space A] [metric_space B] := A

theorem inscribed_circle_divides_AP (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  (h_tri : isosceles_right_triangle A B C) :
  ∃ r : ℝ, r = (13 + 8*sqrt 2)/41 := sorry

end inscribed_circle_divides_AP_l113_113627


namespace function_equiv_proof_l113_113156

def fA_R (x : ℝ) : ℝ := x - 1
def gA_N (x : ℕ) : ℕ := x - 1

def fB (x : ℝ) : ℝ := (x^2 - 4) / (x + 2)
def gB (x : ℝ) : ℝ := x - 2

def fC (x : ℝ) : ℝ := x
def gC (x : ℝ) : ℝ := (Real.sqrt x) ^ 2

def fD (x : ℝ) : ℝ := 2 * x - 1
def gD (x : ℝ) : ℝ := 2 * x - 1

theorem function_equiv_proof :
  (∀ x : ℝ, fA_R x ≠ gA_N x) ∧
  (∀ x : ℝ, fB x ≠ gB x) ∧
  (∀ x : ℝ, (x ≥ 0 → fC x = gC x) ∧ (x < 0 → fC x ≠ gC x)) ∧
  (∀ x : ℝ, fD x = gD x) :=
by
  sorry

end function_equiv_proof_l113_113156


namespace percentage_increase_correct_l113_113462

-- Define the constants involved
def final_value : ℝ := 550
def initial_value : ℝ := 499.99999999999994

-- The theorem to prove the percentage increase
theorem percentage_increase_correct :
  ((final_value - initial_value) / initial_value) * 100 ≈ 10 := 
by {
  -- Proof goes here
  sorry
}

end percentage_increase_correct_l113_113462


namespace probability_a_b_greater_than_5_l113_113024

-- Define the sets from which 'a' and 'b' will be selected.
def set_a : Finset ℕ := {1, 2, 3}
def set_b : Finset ℕ := {2, 3, 4}

-- Define the total number of possible pairs (a, b).
def total_pairs : ℕ := set_a.card * set_b.card

-- Define the pairs where a + b > 5
def favorable_pairs : Finset (ℕ × ℕ) :=
  (set_a.product set_b).filter (λ pair, pair.1 + pair.2 > 5)

-- Define the number of favorable pairs
def favorable_count : ℕ := favorable_pairs.card

-- Prove the probability that a + b > 5 is equal to 1/3
theorem probability_a_b_greater_than_5 : 
  (favorable_count : ℝ) / (total_pairs : ℝ) = 1 / 3 := by
  sorry

end probability_a_b_greater_than_5_l113_113024


namespace binomial_parameters_unique_l113_113235

noncomputable def verify_binomial_distribution_parameters (n p : ℕ) : Prop :=
  let X : ℕ → Prop := λ X, X ∼ binomial n p in
  (E X = 8) ∧ (D X = 1.6) → (n = 100) ∧ (p = 0.08)

theorem binomial_parameters_unique : ∃ n p : ℕ, verify_binomial_distribution_parameters n p :=
sorry

end binomial_parameters_unique_l113_113235


namespace find_integer_n_l113_113520

theorem find_integer_n : ∃ n : ℤ, 5 ≤ n ∧ n ≤ 10 ∧ (n % 6 = 12345 % 6) ∧ n = 9 :=
by
  use 9
  split
  case left => exact dec_trivial
  case right =>
    split
    case left => exact dec_trivial
    case right =>
      split
      case left => exact ModuloEq.symm
      case right => exact rfl

end find_integer_n_l113_113520


namespace proof_l113_113667

-- Define the propositions p and q
def p : Prop := slope_angle (line.mk 1 (-1) 1) = 135
def q : Prop := collinear (point.mk (-1) (-3)) (point.mk 1 1) (point.mk 2 2)

-- Define the final theorem to prove
theorem proof : ¬p ∧ ¬q :=
by sorry

end proof_l113_113667


namespace rajeev_share_of_profit_l113_113023

open Nat

theorem rajeev_share_of_profit (profit : ℕ) (ramesh_xyz_ratio1 ramesh_xyz_ratio2 xyz_rajeev_ratio1 xyz_rajeev_ratio2 : ℕ) (rajeev_ratio_part : ℕ) (total_parts : ℕ) (individual_part_value : ℕ) :
  profit = 36000 →
  ramesh_xyz_ratio1 = 5 →
  ramesh_xyz_ratio2 = 4 →
  xyz_rajeev_ratio1 = 8 →
  xyz_rajeev_ratio2 = 9 →
  rajeev_ratio_part = 9 →
  total_parts = ramesh_xyz_ratio1 * (xyz_rajeev_ratio1 / ramesh_xyz_ratio2) + xyz_rajeev_ratio1 + xyz_rajeev_ratio2 →
  individual_part_value = profit / total_parts →
  rajeev_ratio_part * individual_part_value = 12000 := 
sorry

end rajeev_share_of_profit_l113_113023


namespace ordered_quadruples_division_l113_113660

/-- 
Prove that the number of ordered quadruples (x1, x2, x3, x4) 
of positive odd integers that satisfy the sum being 100, divided by 100, is 208.25.
-/
theorem ordered_quadruples_division :
  (∃ n : ℕ, (∑ i in finset.range 4, (2*(n/2 + 1) - 1) = 100) ∧ (n / 100) = 208.25) := by
  sorry

end ordered_quadruples_division_l113_113660


namespace value_of_c_minus_a_l113_113603

variables (a b c : ℝ)

theorem value_of_c_minus_a (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 60) : (c - a) = 30 :=
by
  have h3 : a + b = 90 := by sorry
  have h4 : b + c = 120 := by sorry
  -- now we have the required form of the problem statement
  -- c - a = 120 - 90
  sorry

end value_of_c_minus_a_l113_113603


namespace exists_basis_of_eigenvectors_max_distinct_commuting_involutions_l113_113392

section InvolutionEigenvectors
open FiniteDimensional

variables {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V]

def is_involution (A : End K V) : Prop := A * A = 1

theorem exists_basis_of_eigenvectors [FiniteDimensional K V] (A : End K V) (hA : is_involution A) :
  ∃ (S : Basis (Fin (finrank K V)) K V), ∀ x : Fin (finrank K V), A (S x) = S x ∨ A (S x) = -S x :=
sorry
end InvolutionEigenvectors

section MaximalCommutingInvolutions
open FiniteDimensional

variables {K V : Type*} [Field K] [AddCommGroup V] [Module K V] [FiniteDimensional K V]

def is_involution (A : End K V) : Prop := A * A = 1

def pairwise_commute (S : Finset (End K V)) : Prop :=
  ∀ A B ∈ S, A * B = B * A

theorem max_distinct_commuting_involutions [FiniteDimensional K V] (hn : finrank K V = n) :
  ∃ (S : Finset (End K V)), S.card = 2 ^ n ∧ pairwise_commute S ∧ (∀ A ∈ S, is_involution A) :=
sorry
end MaximalCommutingInvolutions

end exists_basis_of_eigenvectors_max_distinct_commuting_involutions_l113_113392


namespace problem_statement_l113_113512

theorem problem_statement (pi : ℝ) (h : pi = 4 * Real.sin (52 * Real.pi / 180)) :
  (2 * pi * Real.sqrt (16 - pi ^ 2) - 8 * Real.sin (44 * Real.pi / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.sin (22 * Real.pi / 180)) ^ 2) = 8 * Real.sqrt 3 := 
  sorry

end problem_statement_l113_113512


namespace math_problem_l113_113945

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
noncomputable def y : ℝ := (Real.sqrt 5 - 1) / 2

theorem math_problem :
    x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := 
by
  sorry

end math_problem_l113_113945


namespace problem1_problem2_l113_113916

theorem problem1 (y x m : ℝ) (h1 : y^2 = 4 * x) (h2 : y = 2 * x + m) (h3 : ∃ A B : (ℝ × ℝ), |(fst A - fst B)^2 + (snd A - snd B)^2| = 3 * real.sqrt 5) : 
  m = -4 := by
  sorry

theorem problem2 (a : ℝ) (h1 : (∀ A B : (ℝ × ℝ), (y^2 = 4 * x) → (y = 2 * x - 4) → a ∈ {fst A, fst B}) ∧ 
  (let d := |2 * a - 4| / real.sqrt 5 in S_triangle (3 * real.sqrt 5) d = 9)) : 
  a = 5 ∨ a = -1 := by
  sorry

end problem1_problem2_l113_113916


namespace find_n_value_l113_113664

theorem find_n_value 
    (x y : Int)
    (h_x : x = 3)
    (h_y : y = -3) :
    let n := x - y^(x - y) in
    n = -726 := 
by
  sorry

end find_n_value_l113_113664


namespace ellen_painted_six_orchids_l113_113846

theorem ellen_painted_six_orchids :
  let lily_time := 5,
      rose_time := 7,
      orchid_time := 3,
      vine_time := 2,
      lilies := 17,
      roses := 10,
      vines := 20,
      total_time := 213,
      orchids := (total_time - (lilies * lily_time + roses * rose_time + vines * vine_time)) / orchid_time
  in orchids = 6 :=
by
  let lily_time := 5
  let rose_time := 7
  let orchid_time := 3
  let vine_time := 2
  let lilies := 17
  let roses := 10
  let vines := 20
  let total_time := 213
  let orchids := (total_time - (lilies * lily_time + roses * rose_time + vines * vine_time)) / orchid_time
  show orchids = 6, from sorry

end ellen_painted_six_orchids_l113_113846


namespace min_disks_needed_l113_113691

/-- 
  Sandhya must save 35 files onto disks, each with 1.44 MB space. 
  5 of the files take up 0.6 MB, 18 of the files take up 0.5 MB, 
  and the rest take up 0.3 MB. Files cannot be split across disks.
  Prove that the smallest number of disks needed to store all 35 files is 12.
--/
theorem min_disks_needed 
  (total_files : ℕ)
  (disk_capacity : ℝ)
  (file_sizes : ℕ → ℝ)
  (files_0_6_MB : ℕ)
  (files_0_5_MB : ℕ)
  (files_0_3_MB : ℕ)
  (remaining_files : ℕ)
  (storage_per_disk : ℝ)
  (smallest_disks_needed : ℕ) 
  (h1 : total_files = 35)
  (h2 : disk_capacity = 1.44)
  (h3 : file_sizes 0 = 0.6)
  (h4 : file_sizes 1 = 0.5)
  (h5 : file_sizes 2 = 0.3)
  (h6 : files_0_6_MB = 5)
  (h7 : files_0_5_MB = 18)
  (h8 : remaining_files = total_files - files_0_6_MB - files_0_5_MB)
  (h9 : remaining_files = 12)
  (h10 : storage_per_disk = file_sizes 0 * 2 + file_sizes 1 + file_sizes 2)
  (h11 : smallest_disks_needed = 12) :
  total_files = 35 ∧ disk_capacity = 1.44 ∧ storage_per_disk <= 1.44 ∧ smallest_disks_needed = 12 :=
by
  sorry

end min_disks_needed_l113_113691


namespace initial_distances_l113_113094

theorem initial_distances (x y : ℝ) 
  (h1: x^2 + y^2 = 400)
  (h2: (x - 6)^2 + (y - 8)^2 = 100) : 
  x = 12 ∧ y = 16 := 
by 
  sorry

end initial_distances_l113_113094


namespace neha_mother_age_l113_113673

variable (N M : ℕ)

theorem neha_mother_age (h1 : M - 12 = 4 * (N - 12)) (h2 : M + 12 = 2 * (N + 12)) : M = 60 := by
  sorry

end neha_mother_age_l113_113673


namespace part_I_part_II_l113_113915

-- Part (Ⅰ)
theorem part_I : 
  ∀ (M : ℝ × ℝ) (a : ℝ), 
  M = (1, 0) → 
  (∀ k : ℝ, ∀ A B : ℝ × ℝ, A = (x1, y1) → B = (x2, y2) → 
  y = k * (x - 1) → 
  y^2 = 12 * x → 
  (x1 + x2)/2 = 3 → y = ±(sqrt 3) * (x - 1)) := 
by sorry

-- Part (Ⅱ)
theorem part_II : 
  ∀ (a : ℝ) (A B A' : ℝ × ℝ), 
  a < 0 → 
  A = (x1, y1) → 
  B = (x2, y2) → 
  A' = (x1, -y1) → 
  y = k * (x - a) → 
  y^2 = 12 * x → 
  (A'B passes through (-a, 0)) := 
by sorry

end part_I_part_II_l113_113915


namespace mean_less_than_median_l113_113426

def h : Set ℕ := {1, 7, 18, 20, 29, 33}

noncomputable def mean (s : Set ℕ) : ℚ :=
  let sum := s.fold (λ (x : ℕ) (acc : ℕ), acc + x) 0
  sum / s.size

noncomputable def median (s : Set ℕ) : ℚ :=
  let sorted_list := s.toFinset.sort (λ a b => a < b)
  let n := sorted_list.length
  if n % 2 = 0 then
    (sorted_list.get! (n / 2 - 1) + sorted_list.get! (n / 2)) / 2
  else
    sorted_list.get! (n / 2)

theorem mean_less_than_median :
  mean h - median h = -1 := by
  sorry

end mean_less_than_median_l113_113426


namespace infinitely_many_even_positives_l113_113374

theorem infinitely_many_even_positives (k : ℤ) (t : ℤ) (prime_p : ℤ) (p : nat.prime prime_p): 
  (∃ k, 30 * t + 26 = k ∧ t > 0 ∧ k % 2 = 0) → ∃ t, p % t = 1 → composite (p^2 + k) :=
sorry

end infinitely_many_even_positives_l113_113374


namespace sequence_sum_l113_113142

noncomputable def b : ℕ → ℚ
| 1 => 2
| 2 => 3
| (n + 3) => (1/3) * b (n + 2) + (1/5) * b (n + 1)

theorem sequence_sum :
  ∑' n, b (n + 1) = 85 / 7 :=
sorry

end sequence_sum_l113_113142


namespace lap_length_l113_113093

theorem lap_length (I P : ℝ) (K : ℝ) 
  (h1 : 2 * I - 2 * P = 3 * K) 
  (h2 : 3 * I + 10 - 3 * P = 7 * K) : 
  K = 4 :=
by 
  -- Proof goes here
  sorry

end lap_length_l113_113093


namespace jasmine_added_amount_l113_113158

-- The initial conditions
constant initial_volume : ℝ := 80
constant initial_concentration : ℝ := 0.10
constant added_water : ℝ := 15
constant final_concentration : ℝ := 0.13

-- The amount of jasmine initially in the solution
noncomputable def initial_jasmine_volume : ℝ := initial_volume * initial_concentration

-- The final volume of the solution after adding jasmine and water
noncomputable def final_volume (jasmine_added : ℝ) : ℝ := initial_volume + jasmine_added + added_water

-- The final amount of jasmine in the solution
noncomputable def final_jasmine_volume (jasmine_added : ℝ) : ℝ := (final_volume jasmine_added) * final_concentration

-- The Lean statement we need to prove
theorem jasmine_added_amount (jasmine_added : ℝ) :
  initial_jasmine_volume + jasmine_added = final_jasmine_volume jasmine_added → jasmine_added = 5 := by
sorry

end jasmine_added_amount_l113_113158


namespace probability_of_different_colors_l113_113232

open_locale classical

-- Definitions of the problem
def total_balls : ℕ := 7
def white_balls : ℕ := 4
def red_balls : ℕ := 2
def yellow_balls : ℕ := 1
def total_combinations (n k : ℕ) : ℕ := nat.choose n k

-- Probability calculations
noncomputable def ways_to_draw_two_balls : ℕ := total_combinations total_balls 2
noncomputable def ways_to_draw_different_colors : ℕ :=
  total_combinations white_balls 1 * total_combinations red_balls 1 +
  total_combinations white_balls 1 * total_combinations yellow_balls 1 +
  total_combinations red_balls 1 * total_combinations yellow_balls 1

noncomputable def probability_different_colors : ℚ :=
  ways_to_draw_different_colors / ways_to_draw_two_balls

-- The statement to prove
theorem probability_of_different_colors :
  probability_different_colors = 2 / 3 :=
sorry

end probability_of_different_colors_l113_113232


namespace range_of_a_l113_113271

theorem range_of_a (a : ℝ) (h : ¬ ∃ t : ℝ, t^2 - a * t - a < 0) : -4 ≤ a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l113_113271


namespace machine_B_fewer_bottles_l113_113081

-- Definitions and the main theorem statement
def MachineA_caps_per_minute : ℕ := 12
def MachineC_additional_capacity : ℕ := 5
def total_bottles_in_10_minutes : ℕ := 370

theorem machine_B_fewer_bottles (B : ℕ) 
  (h1 : MachineA_caps_per_minute * 10 + 10 * B + 10 * (B + MachineC_additional_capacity) = total_bottles_in_10_minutes) :
  MachineA_caps_per_minute - B = 2 :=
by
  sorry

end machine_B_fewer_bottles_l113_113081


namespace calculate_expression_l113_113830

theorem calculate_expression : (-1 : ℝ) ^ 2 + | -real.sqrt 2 | + (real.pi - 3) ^ 0 - real.sqrt 4 = real.sqrt 2 :=
by
  sorry

end calculate_expression_l113_113830


namespace f_periodic_f2016_eq_sin_l113_113877

def f : ℕ → (ℝ → ℝ)
| 1       := λ x, Real.cos x
| n + 1 := λ x, derivative (f n x)

theorem f_periodic (n : ℕ) (x : ℝ) :
  f (4 + n) x = f n x :=
sorry

theorem f2016_eq_sin (x : ℝ) : f 2016 x = Real.sin x :=
by
  have h := f_periodic 2012 x
  rw [← h, f_periodic] -- Simplify using periodicity
  sorry

end f_periodic_f2016_eq_sin_l113_113877


namespace solution_set_of_absolute_inequality_l113_113268

variable {ℝ : Type*} [LinearOrderedField ℝ] {f : ℝ → ℝ}

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem solution_set_of_absolute_inequality
  (h_inc : is_increasing f)
  (hA : f 0 = -2)
  (hB : f 3 = 2) :
  { x : ℝ | |f (x + 1)| ≥ 2 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 } :=
by sorry

end solution_set_of_absolute_inequality_l113_113268


namespace number_of_real_solutions_l113_113524

def abs_val_eqn (x : ℝ) : ℝ :=
  x * abs (x - 1) - 4 * abs x + 3

theorem number_of_real_solutions :
  {x : ℝ | abs_val_eqn x = 0}.finite ∧
  finset.card {x : ℝ | abs_val_eqn x = 0}.to_finset = 2 :=
by sorry

end number_of_real_solutions_l113_113524


namespace minimum_expression_value_l113_113119

theorem minimum_expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_sum : a + b + c + d = 4) :
  (∃ x, x = (1/2) ∨
   (∃ y, y ≤ (∑ e in [a, b, c, d], e^8 / prod (λ f, if f = e then (e^2 + 1) else f^2 + e) [a, b, c, d]) ∧ y = x)) := sorry

end minimum_expression_value_l113_113119


namespace debby_candy_remaining_l113_113503

theorem debby_candy_remaining (initial_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 12) 
  (h2 : eaten_candy = 9) 
  (h3 : remaining_candy = initial_candy - eaten_candy) : 
  remaining_candy = 3 :=
by 
  rw [h1, h2] at h3
  exact h3.symm

end debby_candy_remaining_l113_113503


namespace value_of_m_l113_113900

-- Definitions of the conditions
def base6_num (m : ℕ) : ℕ := 2 + m * 6^2
def dec_num (d : ℕ) := d = 146

-- Theorem to prove
theorem value_of_m (m : ℕ) (h1 : base6_num m = 146) : m = 4 := 
sorry

end value_of_m_l113_113900


namespace count_valid_integers_l113_113296

def is_valid_integer (n : ℕ) : Prop :=
  200 <= n ∧ n <= 250 ∧
  let d2 := (n / 100) % 10,
      d1 := (n / 10) % 10,
      d0 := n % 10 in
  d2 = 2 ∧ 2 < d1 ∧ d1 < d0

theorem count_valid_integers : 
  (Finset.filter is_valid_integer (Finset.range (250 + 1))).card = 11 := 
by 
  sorry

end count_valid_integers_l113_113296


namespace natural_number_factors_of_m_l113_113659

theorem natural_number_factors_of_m :
  let m := 2^5 * 3^3 * 5^4 * 7^2 in
  (∀ n : ℕ, n | m → n > 0) → 
  (number_of_factors : ℕ := (((5 + 1) * (3 + 1) * (4 + 1) * (2 + 1)))) →
  number_of_factors = 360 :=
by
  intros
  let m := 2^5 * 3^3 * 5^4 * 7^2
  let number_of_factors := ((5 + 1) * (3 + 1) * (4 + 1) * (2 + 1))
  have : number_of_factors = 360 := sorry
  assumption

end natural_number_factors_of_m_l113_113659


namespace repeating_decimal_to_fraction_l113_113209

theorem repeating_decimal_to_fraction : (x : ℚ) (h : x = 0.3666) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l113_113209


namespace new_percentage_of_female_workers_l113_113401

theorem new_percentage_of_female_workers 
  (original_percentage_female : ℚ) (additional_male_workers : ℕ) (new_total_employees : ℕ)
  (original_percentage_female = 0.60) (additional_male_workers = 26) 
  (new_total_employees = 312) : 
  let original_total_employees := new_total_employees - additional_male_workers in
  let original_female_employees := original_percentage_female * original_total_employees in
  let new_female_percentage := (original_female_employees / new_total_employees) * 100 in
  new_female_percentage ≈ 54.81 := 
by
  sorry

end new_percentage_of_female_workers_l113_113401


namespace probability_real_power_four_l113_113687

noncomputable def S : Set ℚ := 
  {0, 1, 1/2, 3/2, 1/3, 2/3, 4/3, 5/3, 1/4, 3/4, 5/4, 7/4, 1/5, 2/5, 3/5, 4/5, 6/5, 7/5, 8/5, 9/5, 1/6, 5/6, 7/6, 11/6}

theorem probability_real_power_four : 
  let total_combinations := (S.product S).card
  let valid_combinations := {p | p ∈ S × S ∧ (∃ x y ∈ ℂ, x = cos (p.1 * π) ∧ y = sin (p.2 * π) ∧ 4 * x * y * (x^2 - y^2) = 0)}.card
  total_combinations = 576 → valid_combinations = 120 →
  valid_combinations / total_combinations = 5 / 24 := by sorry

end probability_real_power_four_l113_113687


namespace ratio_w_y_l113_113726

-- Define the necessary variables
variables (w x y z : ℚ)

-- Define the conditions as hypotheses
axiom h1 : w / x = 4 / 3
axiom h2 : y / z = 5 / 3
axiom h3 : z / x = 1 / 6

-- State the proof problem
theorem ratio_w_y : w / y = 24 / 5 :=
by sorry

end ratio_w_y_l113_113726


namespace slope_y_intercept_sum_l113_113226

theorem slope_y_intercept_sum 
  (m b : ℝ) 
  (h1 : (2 : ℝ) * m + b = -1) 
  (h2 : (5 : ℝ) * m + b = 2) : 
  m + b = -2 := 
sorry

end slope_y_intercept_sum_l113_113226


namespace count_valid_integers_l113_113298

def is_valid_integer (n : ℕ) : Prop :=
  200 <= n ∧ n <= 250 ∧
  let d2 := (n / 100) % 10,
      d1 := (n / 10) % 10,
      d0 := n % 10 in
  d2 = 2 ∧ 2 < d1 ∧ d1 < d0

theorem count_valid_integers : 
  (Finset.filter is_valid_integer (Finset.range (250 + 1))).card = 11 := 
by 
  sorry

end count_valid_integers_l113_113298


namespace sequence_term_condition_l113_113282

theorem sequence_term_condition (n : ℕ) : (n^2 - 8 * n + 15 = 3) ↔ (n = 2 ∨ n = 6) :=
by 
  sorry

end sequence_term_condition_l113_113282


namespace complex_exponential_sum_equiv_rtheta_l113_113829

theorem complex_exponential_sum_equiv_rtheta :
  15 * complex.exp (complex.I * real.pi / 7) + 15 * complex.exp (complex.I * 5 * real.pi / 7) =
  (30 * real.cos (3 * real.pi / 14) * real.cos (real.pi / 14)) * complex.exp (complex.I * 3 * real.pi / 7) :=
sorry

end complex_exponential_sum_equiv_rtheta_l113_113829


namespace c_eq_one_and_f_is_identity_l113_113352

noncomputable def f : ℝ → ℝ := sorry

axiom c_nonzero : ∃ c : ℝ, c ≠ 0 ∧ (∀ x : ℝ, f(x + 1) = f(x) + c) ∧ (∀ x : ℝ, f(x^2) = f(x)^2)

theorem c_eq_one_and_f_is_identity (c : ℝ) (h1 : ∀ x : ℝ, f(x + 1) = f(x) + c) (h2 : ∀ x : ℝ, f(x^2) = f(x)^2)
  : c = 1 ∧ (∀ x : ℝ, f x = x) :=
by
  sorry

end c_eq_one_and_f_is_identity_l113_113352


namespace hyperbola_equation_l113_113264

theorem hyperbola_equation
    (a b : ℝ)
    (ha : a ≠ 0)
    (hb : b ≠ 0)
    (focus : (2 : ℝ, 0 : ℝ))
    (c : ℝ)
    (hc : 2 = c)
    (asymptote_ratio : b / a = √3) :
    (\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1) = (x^2 - \frac{y^2}{3} = 1) := 
    sorry

end hyperbola_equation_l113_113264


namespace prob_2_pow_x_lt_2_l113_113247

theorem prob_2_pow_x_lt_2 (x : ℝ) (hx : 0 < x ∧ x < 4) : 
  (measure_theory.measure_prob {y | 0 < y ∧ y < 1}) / 
  (measure_theory.measure_prob {y | 0 < y ∧ y < 4}) = 1 / 4 := 
sorry

end prob_2_pow_x_lt_2_l113_113247


namespace real_part_commutes_with_conjugation_product_l113_113005

theorem real_part_commutes_with_conjugation_product (a b : ℂ) :
  (Complex.re (a * Complex.conj b)) = (Complex.re (Complex.conj a * b)) :=
sorry

end real_part_commutes_with_conjugation_product_l113_113005


namespace mod_product_example_l113_113189

theorem mod_product_example :
  (53 * 76 * 91) % 20 = 8 :=
by {
  have h1 : 53 % 20 = 13 := by norm_num,
  have h2 : 76 % 20 = 16 := by norm_num,
  have h3 : 91 % 20 = 11 := by norm_num,
  have h_prod : (13 * 16 * 11) % 20 = 8 := by norm_num,
  rw [←h1, ←h2, ←h3],
  exact h_prod,
}

end mod_product_example_l113_113189


namespace leak_empty_cistern_l113_113422

theorem leak_empty_cistern :
  (normal_fill_time leak_fill_time : ℝ)
  (h1 : normal_fill_time = 12)
  (h2 : leak_fill_time = 14)
  (L : ℝ)
  (h_leak : (1 / normal_fill_time) - L = 1 / leak_fill_time) :
  1 / L = 84 :=
by
  sorry

end leak_empty_cistern_l113_113422


namespace repeating_decimals_expr_as_fraction_l113_113177

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end repeating_decimals_expr_as_fraction_l113_113177


namespace function_properties_l113_113196

def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x > f y

noncomputable def f : ℝ → ℝ := λ x, 1 / (x^2)

theorem function_properties :
  is_even f ∧ is_decreasing f (Set.Ioi 0) := by
    sorry

end function_properties_l113_113196


namespace repeating_decimal_to_fraction_l113_113208

theorem repeating_decimal_to_fraction : (x : ℚ) (h : x = 0.3666) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l113_113208


namespace Q_at_1_eq_1_l113_113837

noncomputable def Q (x : ℚ) : ℚ := x^4 - 16*x^2 + 16

theorem Q_at_1_eq_1 : Q 1 = 1 := by
  sorry

end Q_at_1_eq_1_l113_113837


namespace sequence_term_formula_1_l113_113123

theorem sequence_term_formula_1 (n : ℕ) (h : n ≥ 1) :
  let S := λ n, n^2 + 1 in
  (S n - S (n-1)) = (2*n - 1) := by
  sorry

end sequence_term_formula_1_l113_113123


namespace jack_years_after_son_death_l113_113990

noncomputable def jackAdolescenceTime (L : Real) : Real := (1 / 6) * L
noncomputable def jackFacialHairTime (L : Real) : Real := (1 / 12) * L
noncomputable def jackMarriageTime (L : Real) : Real := (1 / 7) * L
noncomputable def jackSonBornTime (L : Real) (marriageTime : Real) : Real := marriageTime + 5
noncomputable def jackSonLifetime (L : Real) : Real := (1 / 2) * L
noncomputable def jackSonDeathTime (bornTime : Real) (sonLifetime : Real) : Real := bornTime + sonLifetime
noncomputable def yearsAfterSonDeath (L : Real) (sonDeathTime : Real) : Real := L - sonDeathTime

theorem jack_years_after_son_death : 
  yearsAfterSonDeath 84 
    (jackSonDeathTime (jackSonBornTime 84 (jackMarriageTime 84)) (jackSonLifetime 84)) = 4 :=
by
  sorry

end jack_years_after_son_death_l113_113990


namespace rectangle_area_l113_113807

theorem rectangle_area {A_s A_r : ℕ} (s l w : ℕ) (h1 : A_s = 36) (h2 : A_s = s * s)
  (h3 : w = s) (h4 : l = 3 * w) (h5 : A_r = w * l) : A_r = 108 :=
by
  sorry

end rectangle_area_l113_113807


namespace convert_yahs_to_bahs_l113_113943

noncomputable section

def bahs_to_rahs (bahs : ℕ) : ℕ := bahs * (36/24)
def rahs_to_bahs (rahs : ℕ) : ℕ := rahs * (24/36)
def rahs_to_yahs (rahs : ℕ) : ℕ := rahs * (18/12)
def yahs_to_rahs (yahs : ℕ) : ℕ := yahs * (12/18)
def yahs_to_bahs (yahs : ℕ) : ℕ := rahs_to_bahs (yahs_to_rahs yahs)

theorem convert_yahs_to_bahs :
  yahs_to_bahs 1500 = 667 :=
sorry

end convert_yahs_to_bahs_l113_113943


namespace trig_identity_l113_113439

theorem trig_identity : 4 * Real.sin (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l113_113439


namespace fraction_books_sold_l113_113453

variable (B : ℕ) (F : ℚ)
variable (h1 : 500 / 5 = B - 50)
variable (h2 : 100 = B - 50)
variable (h3 : F = (B - 50) / B)

theorem fraction_books_sold : 
  ∀ (B : ℕ) (F : ℚ), 500 / 5 = B - 50 → F = (B - 50) / B → F = 2 / 3 :=
by
  intros B F h1 h3
  have h2 := Nat.eq_of_mul_eq_mul_right (by norm_num) h1
  rw h2 at h3
  exact h3
--sorry

end fraction_books_sold_l113_113453


namespace sin_double_angle_identity_l113_113546

theorem sin_double_angle_identity (α : ℝ) (h : sin α + cos α = - (3 * sqrt 5) / 5) : 
  sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_identity_l113_113546


namespace find_other_endpoint_l113_113059

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end find_other_endpoint_l113_113059


namespace tilly_star_count_l113_113086

theorem tilly_star_count (E : ℕ) (h1 : 6 * E + E = 840) : E = 120 :=
by {
  -- This is to ensure it will build successfully.
  simp at h1,
  have h2: 7 * E = 840, 
  {exact h1,},
  have h3 : E = 120,
  {
    apply nat.eq_of_mul_eq_mul_right,
    norm_num,
    exact h2,
  },
  exact h3,
}

end tilly_star_count_l113_113086


namespace unique_coin_sums_count_l113_113132

def coin_values : List ℕ := [1, 1, 1, 5, 10, 10, 50]

theorem unique_coin_sums_count :
  (Finset.image (λ (pair : ℕ × ℕ), pair.1 + pair.2)
                ((Finset.universe (Fin $ List.length coin_values)).product (Finset.universe (Fin $ List.length coin_values)))).card = 8 :=
sorry

end unique_coin_sums_count_l113_113132


namespace sufficient_and_necessary_condition_l113_113811

theorem sufficient_and_necessary_condition (x : ℝ) :
  x^2 - 4 * x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 :=
sorry

end sufficient_and_necessary_condition_l113_113811


namespace train_passing_time_l113_113150

def train_length : ℝ := 360 -- meters
def platform_length : ℝ := 140 -- meters
def speed_kmh : ℝ := 45 -- km/hr
def total_distance : ℝ := train_length + platform_length
def speed_mps : ℝ := speed_kmh * (1000 / 3600)

theorem train_passing_time :
  total_distance / speed_mps = 40 :=
by
  sorry

end train_passing_time_l113_113150


namespace domain_of_f_l113_113217

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end domain_of_f_l113_113217


namespace line_AC_eq_line_altitude_B_AB_eq_l113_113559

open EuclideanGeometry

-- Given vertices of the triangle
def A := (4 : ℝ, 0 : ℝ)
def B := (6 : ℝ, 7 : ℝ)
def C := (0 : ℝ, 3 : ℝ)

-- Part 1: Equation of the line containing side AC
theorem line_AC_eq :
  ∃ (a b c : ℝ), (a = 3) ∧ (b = 4) ∧ (c = -12) ∧
  ∀ (x y : ℝ), (x, y) ∈ line (3, 4, -12) ↔ 3 * x + 4 * y - 12 = 0 :=
by
  sorry

-- Part 2: Equation of the line containing the altitude from B to side AB
theorem line_altitude_B_AB_eq :
  ∃ (a b c : ℝ), (a = 2) ∧ (b = 7) ∧ (c = -21) ∧
  ∀ (x y : ℝ), (x, y) ∈ line (2, 7, -21) ↔ 2 * x + 7 * y - 21 = 0 :=
by
  sorry

end line_AC_eq_line_altitude_B_AB_eq_l113_113559


namespace donuts_left_is_correct_l113_113490

def original_donuts := 2.5 * 12
def eaten_donuts := 0.10 * original_donuts
def snack_donuts := 4
def donuts_left_for_coworkers := original_donuts - eaten_donuts - snack_donuts

theorem donuts_left_is_correct :
  donuts_left_for_coworkers = 23 :=
by 
  sorry

end donuts_left_is_correct_l113_113490


namespace water_wasted_per_hour_l113_113709

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l113_113709


namespace david_initial_deposit_l113_113192

noncomputable def initialDeposit (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / (n : ℝ))^(n * t)

theorem david_initial_deposit :
  initialDeposit 5300 0.06 2 1 ≈ 4995.33 :=
by
  unfold initialDeposit
  have h : 5300 / (1 + 0.06 / 2)^(2 * 1) ≈ 4995.33 := by
    norm_num
  exact h

end david_initial_deposit_l113_113192


namespace gain_percentage_l113_113172

theorem gain_percentage (SP G : ℝ) (h_SP : SP = 180) (h_G : G = 30) : (G / (SP - G)) * 100 = 20 := by
  rw [h_SP, h_G]
  simp
  norm_num
  -- or use sorry to indicate skipping the proof
  -- sorry

end gain_percentage_l113_113172


namespace triangle_inequality_difference_l113_113336

theorem triangle_inequality_difference :
  ∀ (x : ℤ), (x + 8 > 3) → (x + 3 > 8) → (8 + 3 > x) →
  ( 10 - 6 = 4 ) :=
by sorry

end triangle_inequality_difference_l113_113336


namespace sacks_per_day_proof_l113_113292

-- Definitions based on the conditions in the problem
def totalUnripeOranges : ℕ := 1080
def daysOfHarvest : ℕ := 45

-- Mathematical statement to prove
theorem sacks_per_day_proof : totalUnripeOranges / daysOfHarvest = 24 :=
by sorry

end sacks_per_day_proof_l113_113292


namespace rectangle_area_l113_113628

theorem rectangle_area (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC : ℕ) (H1 : AB = 15) (H2 : AC = 17) (H3 : ∃ a b c d : A, MetricSpace.distance a b = AB ∧ MetricSpace.distance a c = AC ∧ (∠ABC = π / 2)) :
  let BC := Nat.sqrt (AC ^ 2 - AB ^ 2) in AB * BC = 120 :=
by
  sorry

end rectangle_area_l113_113628


namespace circle_center_l113_113051

-- Define the points A and B
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (8, 9)

-- Midpoint formula function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Prove that the midpoint of A and B is (5, 6)
theorem circle_center : midpoint A B = (5, 6) := by
  -- Here we would have the implementation of the proof, but it's replaced by sorry
  sorry

end circle_center_l113_113051


namespace meeting_success_probability_l113_113798

noncomputable def meeting_probability : ℝ := 
  let total_volume := 2^4
  let meeting_volume := total_volume / 4!
  let adjustment := 4 * ((1/4 * 1/2) / 3)
  let feasible_volume := meeting_volume - adjustment
  feasible_volume / total_volume

theorem meeting_success_probability :
  meeting_probability = 1 / 32 := by
  sorry

end meeting_success_probability_l113_113798


namespace sum_of_cubes_eleven_to_twentyfour_l113_113549

theorem sum_of_cubes_eleven_to_twentyfour:
  (∑ i in Finset.range 24 \ Finset.range 10, (i + 1)^3) = 86975 :=
by
  sorry

end sum_of_cubes_eleven_to_twentyfour_l113_113549


namespace linear_function_difference_l113_113658

-- Define the problem in Lean.
theorem linear_function_difference (g : ℕ → ℝ) (h : ∀ x y : ℕ, g x = 3 * x + g 0) (h_condition : g 4 - g 1 = 9) : g 10 - g 1 = 27 := 
by
  sorry -- Proof is omitted.

end linear_function_difference_l113_113658


namespace Jonathan_extra_calories_on_Saturday_l113_113994

variable (daily_intake : ℕ) (daily_burn : ℕ) (weekly_deficit : ℕ) (sat_intake : ℕ)

-- Conditions
def jonathan_intake_day := 2500
def jonathan_burn_day := 3000
def jonathan_weekly_deficit := 2500

-- Prove that Jonathan consumes 1000 extra calories on Saturday
theorem Jonathan_extra_calories_on_Saturday :
  sat_intake - daily_intake = 1000 :=
by
  -- Use the definitions from conditions
  let daily_intake := 2500
  let daily_burn := 3000
  let weekly_deficit := 2500
  -- Weekly total burn is 7 * daily_burn
  let total_weekly_burn := 7 * daily_burn
  -- Weekly intake including the extra on Saturday
  let total_weekly_intake := total_weekly_burn - weekly_deficit
  -- Intake for 6 days
  let intake_first_6_days := 6 * daily_intake
  -- Caloric intake on Saturday
  let sat_intake := total_weekly_intake - intake_first_6_days
  -- Extra calories on Saturday
  have h1 : sat_intake - daily_intake = 1000 := sorry
  exact h1


end Jonathan_extra_calories_on_Saturday_l113_113994


namespace orthogonal_projection_is_line_implies_perpendicular_plane_l113_113754

def geometric_figure_in_perpendicular_plane (P : plane) (G : set point) : Prop :=
  ∃ Q : plane, Q ⊥ P ∧ G ⊆ Q

theorem orthogonal_projection_is_line_implies_perpendicular_plane
  (P : plane) (G : set point) :
  (∃ L : line, orthogonal_projection(P, G) = L) →
  geometric_figure_in_perpendicular_plane P G :=
by
  sorry

end orthogonal_projection_is_line_implies_perpendicular_plane_l113_113754


namespace min_x_y_l113_113896

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_x_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 16 * y = x * y) :
  min_value x y = 25 :=
sorry

end min_x_y_l113_113896


namespace odd_integers_between_fractions_l113_113302

theorem odd_integers_between_fractions : 
  let lower := 23 / 6
  let upper := 53 / 3
  ∃ n : ℕ, (n = 7) ∧ (∀ x : ℕ, lower < ↑x ∧ ↑x < upper → x % 2 = 1) :=
begin
  sorry
end

end odd_integers_between_fractions_l113_113302


namespace exists_six_red_points_with_centroids_at_center_l113_113322

noncomputable def centroid (A B C : (ℤ × ℤ)) : (ℤ × ℤ) :=
  let (x1, y1), (x2, y2), (x3, y3) := A, B, C
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

def six_red_points_centroid_to_center (r_points : finset (ℤ × ℤ)) (A B C D E F : (ℤ × ℤ)) : Prop :=
  centroid A B C = (0, 0) ∧ centroid D E F = (0, 0) ∧ 
  A ∈ r_points ∧ B ∈ r_points ∧ C ∈ r_points ∧
  D ∈ r_points ∧ E ∈ r_points ∧ F ∈ r_points ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ D ≠ E ∧ E ≠ F ∧ D ≠ F

theorem exists_six_red_points_with_centroids_at_center :
  ∀ (H_lines V_lines : finset ℤ) (grid_size : ℤ),
  grid_size = 2015 →
  H_lines.card = 2017 →
  V_lines.card = 2017 →
  ∀ (r_points : finset (ℤ × ℤ)),
  (∀ x y, x ∈ H_lines → y ∈ V_lines → (x, y) ∈ r_points) →
  ∃ A B C D E F,
  six_red_points_centroid_to_center r_points A B C D E F :=
by {
  intros,
  -- Problem setup and proof steps go here
  sorry -- Proof to establish the theorem
}

end exists_six_red_points_with_centroids_at_center_l113_113322


namespace gcd_47_power5_1_l113_113185
-- Import the necessary Lean library

-- Mathematically equivalent proof problem in Lean 4
theorem gcd_47_power5_1 (a b : ℕ) (h1 : a = 47^5 + 1) (h2 : b = 47^5 + 47^3 + 1) :
  Nat.gcd a b = 1 :=
by
  sorry

end gcd_47_power5_1_l113_113185


namespace simplify_expression_l113_113848

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 1) (h₂ : a ≠ 1 / 2) :
    1 - 1 / (1 - a / (1 - a)) = -a / (1 - 2 * a) := by
  sorry

end simplify_expression_l113_113848


namespace ott_fraction_of_total_l113_113361

-- Define the conditions
def moe_given  : ℝ := 4
def moe_fraction : ℝ := 1 / 6
def loki_given : ℝ := 4
def loki_fraction : ℝ := 1 / 3
def nick_given : ℝ := 4
def nick_fraction : ℝ := 1 / 2
def total_original_money : ℝ := 72

-- Proof problem:
theorem ott_fraction_of_total :
  let moe_original := moe_given / moe_fraction
  let loki_original := loki_given / loki_fraction
  let nick_original := nick_given / nick_fraction
  let total_money_received_by_ott := moe_given + loki_given + nick_given
  moe_original + loki_original + nick_original = total_original_money →
  (total_money_received_by_ott / total_original_money) = 1 / 6 :=
by
  sorry

end ott_fraction_of_total_l113_113361


namespace equal_segments_inscribed_triangle_l113_113092

-- Definitions for geometry elements involved
variables {A B C M N : Point} {α β γ : ℝ}

-- Given conditions as definitions in Lean
def triangle_inscribed_in_circle (A B C : Point) : Prop := ∃ (O : Point) (r : ℝ), 
  Circle O r ∧ (A ∈ Circle O r ∧ B ∈ Circle O r ∧ C ∈ Circle O r)

def diameter_of_circle (A M : Point) (O : Point) (r : ℝ) : Prop := 
  ∃ (O : Point) (r : ℝ), Circle O r ∧ diameter O r (line_segment A M)

def perpendicular_segment (A N : Point) (BC : Line) : Prop := 
  ∃ (O : Point) (r : ℝ), Circle O r ∧ is_perpendicular (line_segment A N) BC

-- Main theorem to prove
theorem equal_segments_inscribed_triangle (A B C M N : Point) :
  triangle_inscribed_in_circle A B C →
  diameter_of_circle A M A B →
  perpendicular_segment A N (line_segment B C) →
  equal_lengths (line_segment B N) (line_segment C M) :=
by
  sorry

end equal_segments_inscribed_triangle_l113_113092


namespace complex_number_is_purely_imaginary_l113_113903

theorem complex_number_is_purely_imaginary (x : ℝ) : 
  (∃ z : ℂ, z = (2 + complex.i) / (x - complex.i) ∧ z.im ≠ 0 ∧ z.re = 0) → x = 1 / 2 :=
by
  sorry

end complex_number_is_purely_imaginary_l113_113903


namespace honey_jam_eating_time_l113_113675

theorem honey_jam_eating_time 
  (M B : ℕ) 
  (h1 : M = 8) 
  (h2 : B = 4) 
  (hcarlson_total_time : ∀ M B, 5 * M + 2 * B = 40 + 8)
  (hpooh_total_time : 3 * (10 - M) + 7 * (10 - B) = 40 + 8) :
  5 * M + 2 * B = 48 := 
by
  rw [h1, h2]
  trivial
  sorry

end honey_jam_eating_time_l113_113675


namespace sqrt_meaningful_real_l113_113316

theorem sqrt_meaningful_real (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) ↔ x ≤ 1 :=
by
  sorry

end sqrt_meaningful_real_l113_113316


namespace statement_1_statement_2_statement_3_statement_4_main_proof_l113_113377

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem statement_1 : ¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x := sorry

theorem statement_2 : ∃! x, f x - x = 0 := sorry

theorem statement_3 : ¬ ∃ k > 0, ∀ x > 0, f x > k * x := sorry

theorem statement_4 : ∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4 := sorry

theorem main_proof : (¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x) ∧ 
                     (∃! x, f x - x = 0) ∧ 
                     (¬ ∃ k > 0, ∀ x > 0, f x > k * x) ∧ 
                     (∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4) := 
by
  apply And.intro
  · exact statement_1
  · apply And.intro
    · exact statement_2
    · apply And.intro
      · exact statement_3
      · exact statement_4

end statement_1_statement_2_statement_3_statement_4_main_proof_l113_113377


namespace second_divisor_l113_113221

theorem second_divisor (x : ℕ) : (282 % 31 = 3) ∧ (282 % x = 3) → x = 9 :=
by
  sorry

end second_divisor_l113_113221


namespace communication_railway_conditions_l113_113369

theorem communication_railway_conditions (n : ℕ) :
  (∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → 
  ∃! (k : ℕ), k ∈ {1, 2, ..., n} ∧ (k = (i + 1) % n ∨ k = (i - 1) % n) ∧ i communicates with k) ↔ n = 4 :=
begin
  sorry
end

end communication_railway_conditions_l113_113369


namespace square_pyramid_frustum_volume_fraction_l113_113808

theorem square_pyramid_frustum_volume_fraction :
  ∀ (base_edge altitude : ℝ),
    base_edge = 40 →
    altitude = 20 →
    let V := (1/3) * (base_edge ^ 2) * altitude in
    let smaller_height := altitude / 3 in
    let smaller_volume := (1/27) * V in
    let frustum_volume := V - smaller_volume in
    (frustum_volume / V) = 26 / 27 :=
by
  intros base_edge altitude h_base h_alt V smaller_height smaller_volume frustum_volume
  rw [h_base, h_alt]
  let original_volume := (1/3) * (40 ^ 2) * 20
  let smaller_pyramid_volume := (1/27) * original_volume
  let frustum_volume := original_volume - smaller_pyramid_volume
  have h1 : V = original_volume := by rfl
  have h2 : smaller_volume = smaller_pyramid_volume := by rfl
  have h3 : frustum_volume = original_volume - smaller_pyramid_volume := by rfl
  have h_ratio : (frustum_volume / original_volume) = 26 / 27 := by sorry
  rw [h1, h2, h3]
  exact h_ratio
  sorry

end square_pyramid_frustum_volume_fraction_l113_113808


namespace speed_of_second_train_l113_113445

theorem speed_of_second_train 
  (length_train1 : ℝ)
  (speed_train1_kmph : ℝ)
  (crossing_time_seconds : ℝ)
  (length_train2 : ℝ) 
  (speed_train2_kmph : ℝ)
  (h1 : length_train1 = 270)
  (h2 : speed_train1_kmph = 120)
  (h3 : crossing_time_seconds = 9)
  (h4 : length_train2 = 230.04)
  (h5 : (speed_train1_kmph + speed_train2_kmph) × (5 / 18) = (length_train1 + length_train2) / crossing_time_seconds × (18 / 5)) :
  speed_train2_kmph = 880.08 :=
by {
  -- Skipping the proof steps
  sorry
}

end speed_of_second_train_l113_113445


namespace line_AC_eq_line_altitude_B_AB_eq_l113_113560

open EuclideanGeometry

-- Given vertices of the triangle
def A := (4 : ℝ, 0 : ℝ)
def B := (6 : ℝ, 7 : ℝ)
def C := (0 : ℝ, 3 : ℝ)

-- Part 1: Equation of the line containing side AC
theorem line_AC_eq :
  ∃ (a b c : ℝ), (a = 3) ∧ (b = 4) ∧ (c = -12) ∧
  ∀ (x y : ℝ), (x, y) ∈ line (3, 4, -12) ↔ 3 * x + 4 * y - 12 = 0 :=
by
  sorry

-- Part 2: Equation of the line containing the altitude from B to side AB
theorem line_altitude_B_AB_eq :
  ∃ (a b c : ℝ), (a = 2) ∧ (b = 7) ∧ (c = -21) ∧
  ∀ (x y : ℝ), (x, y) ∈ line (2, 7, -21) ↔ 2 * x + 7 * y - 21 = 0 :=
by
  sorry

end line_AC_eq_line_altitude_B_AB_eq_l113_113560


namespace all_odd_pos_ints_valid_k_l113_113866

-- Define the function d(n) which gives the number of positive divisors of n.
def d (n : ℕ) : ℕ := 
  n.divisors.count

-- Define the main theorem to capture the essence of the problem.
theorem all_odd_pos_ints_valid_k (k : ℕ) (h : k % 2 = 1) : 
  ∃ n : ℕ, 0 < n ∧ (d (n^2)) / (d n) = k :=
sorry

end all_odd_pos_ints_valid_k_l113_113866


namespace number_of_squares_in_figure_100_l113_113832

theorem number_of_squares_in_figure_100 :
  ∃ (a b c : ℤ), (c = 1) ∧ (a + b + c = 7) ∧ (4 * a + 2 * b + c = 19) ∧ (3 * 100^2 + 3 * 100 + 1 = 30301) :=
sorry

end number_of_squares_in_figure_100_l113_113832


namespace sufficient_but_not_necessary_condition_l113_113543

theorem sufficient_but_not_necessary_condition (α : ℝ) (h1 : α = π / 4) (h2 : sin (2 * α) = 1) : 
  (sin (2 * (π / 4)) = 1) ∧ (∃ k : ℤ, α = π/4 + k * π) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l113_113543


namespace angle_DAB_45_degrees_l113_113320

open EuclideanGeometry

-- Definitions for points, triangle, angles, and square.

variable {A B C D E : Point}

-- Conditions for Triangle ABC
def is_right_triangle (A B C : Point) :=
  ∃ (theta : ℝ), theta = 90 ∧ angle A C B = theta ∧ segment A B is_hypotenuse ABC

-- Conditions for Square BCDE
def is_square (B C D E : Point) :=
  segment B C = segment C D ∧ 
  segment C D = segment D E ∧ 
  segment D E = segment E B ∧ 
  angle B C D = 90 ∧ 
  angle C D E = 90 ∧ 
  angle D E B = 90 ∧ 
  angle E B C = 90

-- Assertion that angle DAB equals 45 degrees
theorem angle_DAB_45_degrees
  (h1: is_right_triangle A B C)
  (h2: is_square B C D E)
  (h3: angle B C A = 90) :
  angle D A B = 45 :=
sorry

end angle_DAB_45_degrees_l113_113320


namespace measure_of_angle_C_length_of_side_c_l113_113337

-- Definitions required for the conditions
variables {a b c : ℝ} {A B C : ℝ}
variables (ABC : Triangle a b c A B C)
variables (h0 : Triangle.measureOfAngle C = 2 * a * Real.cos C + b * Real.cos C + c * Real.cos B = 0)

-- Problem 1: Prove the measure of angle C
theorem measure_of_angle_C (h : Triangle.measureOfAngle C = 2 * a * Real.cos C + b * Real.cos C + c * Real.cos B = 0) :
  C = 2 * Real.pi / 3 :=
sorry

-- Problem 2: Given a = 2 and area = sqrt(3)/2, find the length of side c
theorem length_of_side_c (ha : a = 2) (S : Area ABC = sqrt(3) / 2) :
  c = sqrt 7 :=
sorry

end measure_of_angle_C_length_of_side_c_l113_113337


namespace correct_answer_is_D_l113_113399

theorem correct_answer_is_D (N_A : ℝ) :
  (∀ n : ℕ, (n = 1 → 0.1 * N_A < 0.1 * N_A) ∧
            (n = 2 → 2.4 / 24 * 2 * N_A ≠ 0.1 * N_A) ∧
            (n = 3 → 0.1 * N_A ≠ 0.2 * N_A) ∧
            (n = 4 → 0.1 * N_A + 0.1 * N_A = 0.2 * N_A)) →
  (∃ n, n = 4) :=
by
  intros h,
  existsi 4,
  trivial

end correct_answer_is_D_l113_113399


namespace private_pilot_course_cost_l113_113387

theorem private_pilot_course_cost :
  let flight_cost : ℕ := 950
  let ground_cost : ℕ := 325
  flight_cost = ground_cost + 625 → flight_cost + ground_cost = 1275 :=
by
  intros
  unfold flight_cost ground_cost
  sorry

end private_pilot_course_cost_l113_113387


namespace trapezoid_AD_length_l113_113744

open Real

def Trapezoid (ABCD : Type) := 
  (AB CD AD BD AC : ℝ) (BC : ℝ := 47) (CD : ℝ := 47) 
  (AB_parallel_CD AB_perpendicular_BD : Prop) 
  (intersection_O AC BD : Prop)
  (midpoint_P BD OP_Val : Prop := (BD / 2, 13))

noncomputable def length_AD (m n : ℕ) (A B C D : ℝ) (OP : ℝ := 13) : Prop :=
  m = 4 ∧ n = 172 ∧ AD = 4 * sqrt 172

theorem trapezoid_AD_length {ABCD : Type} 
  (h: Trapezoid ABCD)
  (AB_parallel_CD: h.AB AB_parallel_CD)
  (AB_perpendicular_BD: h.AD AB_perpendicular_BD)
  (intersection_O: h.intersection_O h.AC h.BD)
  (midpoint_P: h.midpoint_P h.BD h.OP_Val)
  (OP_13: h.OP_Val = 13) : 
length_AD 4 172 :=
sorry

end trapezoid_AD_length_l113_113744


namespace find_n_and_coefficients_l113_113705

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Define the conditions and proof goals
theorem find_n_and_coefficients (x : ℝ) :
  (∀ n : ℕ, binomial n 2 = binomial n 4 → n = 6) ∧
  (let n := 6 in
    (let a := (1 - 3 * x) ^ n in 
      ∀ (a : ℕ → ℤ),
        a 0 = 1 ∧
        a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 64 ∧
        a 1 + a 2 + a 3 + a 4 + a 5 + a 6 - a 0 = 63 ∧
        a 0 + a 2 + a 4 + a 6 = 2080 ∧
        abs (a 0) + abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) + abs (a 6) = 4096)) :=
begin
  sorry
end

end find_n_and_coefficients_l113_113705


namespace max_triangle_area_l113_113954

noncomputable def triangle_area (x : ℝ) (hx : 0 < x ∧ x < 2 * Real.pi / 3) : ℝ :=
  4 * Real.sqrt 3 * Real.sin x * Real.sin (2 * Real.pi / 3 - x)

theorem max_triangle_area :
  ∃ x : ℝ, 0 < x ∧ x < 2 * Real.pi / 3 ∧
    (∀ y : ℝ, 0 < y ∧ y < 2 * Real.pi / 3 → triangle_area y y.property ≤ triangle_area x x.property) ∧
    triangle_area x x.property = 3 * Real.sqrt 3 :=
begin
  use Real.pi / 3,
  split,
  { -- 0 < Real.pi / 3
    exact Real.pi_pos.div_two },
  split,
  { -- Real.pi / 3 < 2 * Real.pi / 3
    linarith [Real.pi_pos] },
  split,
  { -- ∀ y, (0 < y ∧ y < 2 * Real.pi / 3) → triangle_area y y.property ≤ triangle_area (Real.pi / 3) _
    intros y hy,
    have hxy : y ≤ Real.pi / 3,
    { sorry }, -- Proof of maximum
    rw [triangle_area, triangle_area],
    sorry -- Proof that triangle_area y ≤ triangle_area (Real.pi / 3)
  },
  -- triangle_area (Real.pi / 3) _ = 3 * Real.sqrt 3
  rw [triangle_area],
  have H : (Real.pi / 3 : ℝ).cos = 1 / 2 := Real.cos_pi_div_three,
  rw [Real.sin_sub, Real.cos_two_mul, H, Real.sin_pi_div_three],
  linarith
end

end max_triangle_area_l113_113954


namespace smallest_n_no_fancy_multiple_l113_113137

def is_fancy (n : ℕ) : Prop :=
  ∃ (a : Fin 100 → ℕ), n = ∑ i in Finset.univ, 2^(a i)

def has_100_ones (n : ℕ) : Prop :=
  (Nat.binaryDigits n).count 1 = 100

theorem smallest_n_no_fancy_multiple : 
  (∃ (n : ℕ), 
  (∀ m : ℕ, ¬ is_fancy (m * n)) ∧ 
  (∀ k : ℕ, (∀ m : ℕ, ¬ is_fancy (m * k)) → n ≤ k)) →
  n = 2^101 - 1 :=
sorry

end smallest_n_no_fancy_multiple_l113_113137


namespace fisher_needed_score_l113_113087

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score_l113_113087


namespace sides_of_polygons_l113_113723

-- Definitions of the conditions
def sides1 : Nat := n
def sides2 : Nat := n + 4
def sides3 : Nat := n + 12
def sides4 : Nat := n + 13

-- Formula for the number of diagonals in an n-sided polygon
def diagonals (n : Nat) : Nat := n * (n - 3) / 2

-- Equation based on the sum of diagonals condition
theorem sides_of_polygons (n : Nat) :
  (diagonals n + diagonals (n + 13)) = (diagonals (n + 4) + diagonals (n + 12)) →
  n = 3 := by
    sorry

end sides_of_polygons_l113_113723


namespace find_lambda_l113_113924

noncomputable def vec_a := (3, 4)
noncomputable def vec_b := (4, 3)
noncomputable def vec_c (λ : ℝ) := (3 * λ - 4, 4 * λ - 3)

theorem find_lambda (λ : ℝ) 
  (h_angle_eq : 
    (λ c, let dot_ab (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
              norm_a := Math.sqrt (vec_a.1 ^ 2 + vec_a.2 ^ 2)
              norm_b := Math.sqrt (vec_b.1 ^ 2 + vec_b.2 ^ 2)
              norm_c := Math.sqrt ((c  λ).1 ^ 2 + (c  λ).2 ^ 2) in
          (dot_ab (vec_c λ) vec_a) / (norm_c * norm_a) = (dot_ab (vec_c λ) vec_b) / (norm_c * norm_b)) vec_c) :
  λ = -1 :=
by {
  sorry
}

end find_lambda_l113_113924


namespace find_S2019_l113_113728

def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, a (n + 1) ^ 2 - 2 * a (n + 1) * a n + a (n + 1) - 2 * a n = 0)

theorem find_S2019 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_seq : sequence a) :
  S 2019 = 2 ^ 2019 - 1 :=
sorry

end find_S2019_l113_113728


namespace polar_coordinates_of_point_l113_113500

noncomputable def polarCoordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y >= 0 then Real.atan (y / x)
           else if x > 0 ∧ y < 0 then 2 * Real.pi - Real.atan (|y / x|)
           else if x < 0 then Real.pi + Real.atan (y / x)
           else if y > 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ)

theorem polar_coordinates_of_point :
  polarCoordinates 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_point_l113_113500


namespace helicopter_altitude_correct_l113_113475

noncomputable def helicopter_altitude : ℝ :=
  let a := 12 -- distance between Alice and Bob in miles
  let alpha := 45 * Real.pi / 180 -- Alice's angle of elevation in radians
  let beta := 30 * Real.pi / 180 -- Bob's angle of elevation in radians
  let h := a / (Real.cos alpha / Math.sqrt 2)
  h

theorem helicopter_altitude_correct : helicopter_altitude = 12 * Math.sqrt 2 := by
  sorry

end helicopter_altitude_correct_l113_113475


namespace count_two_digit_numbers_l113_113814

theorem count_two_digit_numbers :
  let count := finset.filter (λ n : ℕ, (n - (n / 10 + n % 10)) % 10 = 4) (finset.Ico 10 100)
  in count.card = 10 := 
by
  sorry

end count_two_digit_numbers_l113_113814


namespace larger_angle_at_330_l113_113828

def larger_angle_formed_by_clock_hands_at_330 : ℝ :=
  let hour_hand_position := 90 + (30 / 60) * 30
  let minute_hand_position := 30 * 6
  let smaller_angle := abs (minute_hand_position - hour_hand_position)
  let larger_angle := 360 - smaller_angle
  larger_angle

theorem larger_angle_at_330 : larger_angle_formed_by_clock_hands_at_330 = 285 :=
by
  -- Proof will be given here
  sorry

end larger_angle_at_330_l113_113828


namespace Banks_is_valid_l113_113181

-- Define the motion of a point on the wheel as a trochoid curve
structure Trochoid (R r : ℝ) (theta : ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (eq_x : x = R * theta - r * sin (theta))
  (eq_y : y = R - r * cos (theta))

-- Define the theorem that states Banks' observation is valid
theorem Banks_is_valid (R r : ℝ) (theta : ℝ) (h_R_positive : R > 0) (h_r_positive : r > 0):
  ∃ theta: ℝ, let t := Trochoid R r theta in t.x < 0 :=
sorry

end Banks_is_valid_l113_113181


namespace solve_for_x_in_complex_eq_l113_113694

theorem solve_for_x_in_complex_eq (x : ℂ) : 3 + 2 * complex.I * x = 4 - 5 * complex.I * x → x = -complex.I / 7 := by
  sorry

end solve_for_x_in_complex_eq_l113_113694


namespace max_islands_l113_113702

theorem max_islands (N : ℕ) (h1 : N ≥ 7)
  (h2 : ∀ (i j : ℕ), i ≠ j → i < N → j < N → ¬connected i j → ¬ bridge_exists i j)
  (h3 : ∀ i < N, ∃ B_i ⊆ {0 ... N-1}, |B_i| ≤ 5 ∧ (∀ j ∈ B_i, connected i j))
  (h4 : ∀ I ⊆ {0 ... N-1}, |I| = 7 → ∃ i j ∈ I, i ≠ j ∧ connected i j) :
  N ≤ 36 :=
sorry

end max_islands_l113_113702


namespace binary_to_decimal_101_eq_5_l113_113190

theorem binary_to_decimal_101_eq_5 : 
  let b0 := 1, b1 := 0, b2 := 1,
      p0 := 2^0, p1 := 2^1, p2 := 2^2,
      d := b0 * p0 + b1 * p1 + b2 * p2
  in d = 5 := by
  let b0 := 1;
  let b1 := 0;
  let b2 := 1;
  let p0 := 2^0;
  let p1 := 2^1;
  let p2 := 2^2;
  let d := b0 * p0 + b1 * p1 + b2 * p2;
  calc d
  = b0 * p0 + b1 * p1 + b2 * p2 : by rfl
  ... = 1 * 1 + 0 * 2 + 1 * 4 : by rfl
  ... = 1 + 0 + 4 : by rfl
  ... = 5 : by rfl

end binary_to_decimal_101_eq_5_l113_113190


namespace solve_ab_find_sqrt_l113_113555

variable (a b : ℝ)

-- Given Conditions
axiom h1 : real.cbrt (2 * b - 2 * a) = -2
axiom h2 : real.sqrt (4 * a + 3 * b) = 3

-- Goal: Prove that a = 3 and b = -1
theorem solve_ab : a = 3 ∧ b = -1 := by
  sorry

-- Given a = 3 and b = -1, find the square root of 5a - b
theorem find_sqrt : a = 3 ∧ b = -1 → real.sqrt (5 * a - b) = 4 ∨ real.sqrt (5 * a - b) = -4 := by
  sorry

end solve_ab_find_sqrt_l113_113555


namespace xyz_sum_eq_48_l113_113309

theorem xyz_sum_eq_48 (x y z : ℕ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
  sorry

end xyz_sum_eq_48_l113_113309


namespace wire_not_used_is_20_l113_113203

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l113_113203


namespace frequency_count_l113_113467

theorem frequency_count (N : ℕ) (f : ℝ) (F : ℝ) : N = 1000 → f = 0.6 → F = N * f → F = 600 :=
by
  intros hN hf hF
  rw [hN, hf] at hF
  calc
    1000 * 0.6 = 600 : sorry

end frequency_count_l113_113467


namespace mn_bound_l113_113416

theorem mn_bound (m : ℕ → ℝ) (F : ℕ → set (ℝ × ℕ)) (d : (ℝ × ℕ) → ℝ) (Co : (ℕ → set (ℝ × ℕ)) → set (ℝ × ℕ)) 
  (ε : set (ℝ × ℕ) → (ℝ × ℕ))
  (h1 : ∀ k ≥ 9, d(ε(Co(F k))) ≤ 3)
  : ∀ n ≥ 9, |m n| ≤ 3 * n - 11 :=
by
  sorry

end mn_bound_l113_113416


namespace scheduling_courses_l113_113305

-- Defining the problem in Lean
theorem scheduling_courses :
  let courses := 4
  let periods := 7
  let no_consec := true
  let free_period := 1
  in 
  (number_of_ways : Nat) = 624 :=
  sorry

end scheduling_courses_l113_113305


namespace interest_rate_B_C_is_correct_l113_113312

def principal : ℝ := 3500
def rate_A_B : ℝ := 10
def time : ℝ := 3
def gain_B : ℝ := 157.5

def interest_paid_B_A : ℝ := principal * rate_A_B * time / 100
def total_gain_B : ℝ := gain_B + interest_paid_B_A

def interest_rate_B_C : ℝ :=
  let interest_from_C := total_gain_B
  interest_from_C / (principal * time / 100)

theorem interest_rate_B_C_is_correct :
  interest_rate_B_C = 10 :=
by
  sorry

end interest_rate_B_C_is_correct_l113_113312


namespace max_value_m_n_squared_sum_l113_113263

theorem max_value_m_n_squared_sum (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m * n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_value_m_n_squared_sum_l113_113263


namespace equilateral_triangle_semicircle_perimeter_l113_113159

theorem equilateral_triangle_semicircle_perimeter (a : ℝ) (h : a = 1) :
  let r := a / 2 in
  let circumference := 2 * Real.pi * r in
  3 * (circumference / 2) = 3 * r * Real.pi :=
by
  sorry

end equilateral_triangle_semicircle_perimeter_l113_113159


namespace area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l113_113110

-- Define the side lengths of squares A, B, and C
def side_length_A (s : ℝ) : ℝ := s
def side_length_B (s : ℝ) : ℝ := 2 * s
def side_length_C (s : ℝ) : ℝ := 3.6 * s

-- Define the areas of squares A, B, and C
def area_A (s : ℝ) : ℝ := (side_length_A s) ^ 2
def area_B (s : ℝ) : ℝ := (side_length_B s) ^ 2
def area_C (s : ℝ) : ℝ := (side_length_C s) ^ 2

-- Define the sum of areas of squares A and B
def sum_area_A_B (s : ℝ) : ℝ := area_A s + area_B s

-- Prove that the area of square C is 159.2% greater than the sum of areas of squares A and B
theorem area_C_greater_than_sum_area_A_B_by_159_point_2_percent (s : ℝ) : 
  ((area_C s - sum_area_A_B s) / (sum_area_A_B s)) * 100 = 159.2 := 
sorry

end area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l113_113110


namespace customer_turned_in_two_nickels_l113_113824

-- Define the conditions in Lean 4
def number_of_dimes := 2
def total_coins := 11
def number_of_quarters := 7

-- Define the number of nickels to be proved
def number_of_nickels : ℕ :=
  total_coins - (number_of_dimes + number_of_quarters)

-- The theorem to be proved
theorem customer_turned_in_two_nickels : number_of_nickels = 2 := by
  -- state the problem in terms of condition
  have h1 : total_coins = number_of_dimes + number_of_quarters + number_of_nickels := by
    rw [number_of_nickels]
    linarith
  -- provide conclusion of proof
  sorry

end customer_turned_in_two_nickels_l113_113824


namespace function_properties_l113_113269

theorem function_properties {f : ℝ → ℝ}
  (h1 : ∀ x, f (2 * x - 2) = f (2 - 2 * x))
  (h2 : ∀ x, f (1 / 2 * x + 1) + f (3 - 1 / 2 * x) = 0) :
  (∀ x, f x = f (-x)) ∧ (f 2 = 0) ∧ (∀ x, f (x + 8) = f x) := 
by
  split
  { sorry }
  split
  { sorry }
  { sorry }

end function_properties_l113_113269


namespace integral_sqrt_plus_x_l113_113173

open Real

theorem integral_sqrt_plus_x :
  (∫ x in -1..1, (sqrt (1 - x^2) + x)) = (π / 2) :=
by
  -- We should use the provided condition to initially state and import half-area of a circle
  have h1 : (∫ x in -1..1, sqrt (1 - x^2)) = (π / 2), from sorry,
  -- Decompose the integral and sum up based on the fact that integral of x between -1 to 1 is 0.
  sorry

end integral_sqrt_plus_x_l113_113173


namespace expected_value_of_k_l113_113510

noncomputable def expected_turns (x : ℝ) : ℝ := (1 - x)⁻²

theorem expected_value_of_k : expected_turns 0.9 = 100 := by
  sorry

end expected_value_of_k_l113_113510


namespace total_toys_l113_113825

theorem total_toys (bill_toys hana_toys hash_toys: ℕ) 
  (hb: bill_toys = 60)
  (hh: hana_toys = (5 * bill_toys) / 6)
  (hs: hash_toys = (hana_toys / 2) + 9) :
  (bill_toys + hana_toys + hash_toys) = 144 :=
by
  sorry

end total_toys_l113_113825


namespace range_of_x_range_of_function_l113_113238

theorem range_of_x (x: Real) (h : sqrt 3 ≤ 3^x ∧ 3^x ≤ 9) : 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

theorem range_of_function (x: Real) (h : 1 / 2 ≤ x ∧ x ≤ 2) : -4 ≤ (Real.log2 x - 1) * (Real.log2 x + 3) ∧ (Real.log2 x - 1) * (Real.log2 x + 3) ≤ 0 :=
by
  sorry

end range_of_x_range_of_function_l113_113238


namespace line_circle_chord_shortest_l113_113888

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_circle_chord_shortest (m : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y m → m = -3 / 4) :=
sorry

end line_circle_chord_shortest_l113_113888


namespace nonempty_subsets_with_cond_l113_113187

open Finset

theorem nonempty_subsets_with_cond (S : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} →
  ((∀ T : Finset ℕ, T ⊆ S → T ≠ ∅ → (T.max' sorry + T.min' sorry)/2 ∈ T) → card ({ T : Finset ℕ | T ⊆ S ∧ T ≠ ∅ ∧ (T.max' sorry + T.min' sorry)/2 ∈ T }.to_finset) = 234) :=
by
  sorry

end nonempty_subsets_with_cond_l113_113187


namespace find_values_b_l113_113518

noncomputable def P (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 5*b - 4)*z + 2

theorem find_values_b (b : ℝ) :
  (∀ z : ℂ, z ∈ [root1, root2, root3, root4]  →
    P b z = 0) ∧
  ((root1 + root2 + root3 + root4) = 8) → 
  (b = 1 ∨ b = 2.5) :=
begin
  sorry
end

end find_values_b_l113_113518


namespace total_blankets_collected_l113_113228

/-- 
Freddie and his team are collecting blankets for three days to be donated to the Children Shelter Organization.
- There are 15 people on the team.
- On the first day, each of them gave 2 blankets.
- On the second day, they tripled the number they collected on the first day by asking door-to-door.
- On the last day, they set up boxes at schools and got a total of 22 blankets.

Prove that the total number of blankets collected for the three days is 142.
-/
theorem total_blankets_collected:
  let people := 15 in
  let blankets_per_person_first_day := 2 in
  let blankets_first_day := people * blankets_per_person_first_day in
  let blankets_second_day := blankets_first_day * 3 in
  let blankets_third_day := 22 in
  blankets_first_day + blankets_second_day + blankets_third_day = 142 :=
by
  sorry

end total_blankets_collected_l113_113228


namespace exists_sin_ge_two_l113_113678

theorem exists_sin_ge_two : ∃ x : ℝ, sin x ≥ 2 := 
sorry

end exists_sin_ge_two_l113_113678


namespace exchange_rate_change_2014_l113_113486

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_2014_l113_113486


namespace store_makes_profit_l113_113451

theorem store_makes_profit (m n : ℕ) (h1 : 40 * m) (h2 : 60 * n) (h3 : m > n) :
  10 * (m - n) > 0 := by
sorry

end store_makes_profit_l113_113451


namespace overlap_area_of_parallelogram_l113_113748

theorem overlap_area_of_parallelogram (w1 w2 : ℝ) (β : ℝ) (hβ : β = 30) (hw1 : w1 = 2) (hw2 : w2 = 1) : 
  (w1 * (w2 / Real.sin (β * Real.pi / 180))) = 4 :=
by
  sorry

end overlap_area_of_parallelogram_l113_113748


namespace gardening_project_total_cost_l113_113168

theorem gardening_project_total_cost :
  let 
    num_rose_bushes := 20
    cost_per_rose_bush := 150
    gardener_hourly_rate := 30
    gardener_hours_per_day := 5
    gardener_days := 4
    soil_volume := 100
    soil_cost_per_cubic_foot := 5

    cost_of_rose_bushes := num_rose_bushes * cost_per_rose_bush
    gardener_total_hours := gardener_hours_per_day * gardener_days
    cost_of_gardener := gardener_hourly_rate * gardener_total_hours
    cost_of_soil := soil_volume * soil_cost_per_cubic_foot

    total_cost := cost_of_rose_bushes + cost_of_gardener + cost_of_soil
  in
    total_cost = 4100 := 
  by
    intros
    simp [num_rose_bushes, cost_per_rose_bush, gardener_hourly_rate, gardener_hours_per_day, gardener_days, soil_volume, soil_cost_per_cubic_foot]
    rw [mul_comm num_rose_bushes, mul_comm gardener_total_hours, mul_comm soil_volume]
    sorry -- place for proof steps

end gardening_project_total_cost_l113_113168


namespace parallel_vectors_l113_113923

open Real

theorem parallel_vectors (m : ℝ) : 
  let OA : Real × Real := (0, 1),
      OB : Real × Real := (1, 3),
      OC : Real × Real := (m, m),
      AB : Real × Real := (1, 2),
      AC : Real × Real := (m, m-1)
  in  (AB.1 * AC.2 = AB.2 * AC.1) → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_l113_113923


namespace g_in_terms_of_f_l113_113715

def f (x : ℝ) : ℝ := 
if -3 ≤ x ∧ x ≤ 0 then -2 - x
else if 0 ≤ x ∧ x ≤ 2 then real.sqrt(4 - (x - 2)^2) - 2
else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
else 0 -- f is not defined outside [-3, 3], set it to 0 for simplicity

def g (x : ℝ) : ℝ := f(6 - x)

theorem g_in_terms_of_f : ∀ x : ℝ, g(x) = f(6 - x) := 
by 
  intro x
  refl

end g_in_terms_of_f_l113_113715


namespace find_f_value_l113_113876

def f : ℝ → ℝ :=
λ x, if x < 1 then f (x + 1) else 3^x

theorem find_f_value :
  f (-1 + log 3 5) = 5 :=
sorry

end find_f_value_l113_113876


namespace intersection_M_N_l113_113589

-- Define the conditions M and N
def M : Set ℝ := {x | log 2 (x-1) < 1} -- M = {x | log2(x-1) < 1}
def N : Set ℝ := {x | 1/4 < (1/2)^x ∧ (1/2)^x < 1} -- N = {x | 1/4 < (1/2)^x < 1}

-- Define the proof problem: proving M ∩ N = {x | 1 < x < 2}
theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l113_113589


namespace part1_part2_l113_113291

variable (a b : ℝ)
def A : ℝ := 2 * a * b - a
def B : ℝ := -a * b + 2 * a + b

theorem part1 : 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b := by
  sorry

theorem part2 : (∀ b : ℝ, 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b) -> a = 1 / 6 := by
  sorry

end part1_part2_l113_113291
