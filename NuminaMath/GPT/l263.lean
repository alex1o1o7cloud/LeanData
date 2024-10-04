import Mathlib

namespace monotone_f_a_range_l263_263458

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem monotone_f : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := sorry

theorem a_range : ∀ (a : ℝ), (∀ x ∈ set.Icc 1 3, f (a * x + x^2) > f (2 * x^2 + 4)) → a > 5 := sorry

end monotone_f_a_range_l263_263458


namespace find_m_n_l263_263084

theorem find_m_n (m n : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 1 → x^2 - m * x + n ≤ 0) → m = -4 ∧ n = -5 :=
by
  sorry

end find_m_n_l263_263084


namespace word_sum_problems_l263_263193

theorem word_sum_problems (J M O I : Fin 10) (h_distinct : J ≠ M ∧ J ≠ O ∧ J ≠ I ∧ M ≠ O ∧ M ≠ I ∧ O ≠ I) 
  (h_nonzero_J : J ≠ 0) (h_nonzero_I : I ≠ 0) :
  let JMO := 100 * J + 10 * M + O
  let IMO := 100 * I + 10 * M + O
  (JMO + JMO + JMO = IMO) → 
  (JMO = 150 ∧ IMO = 450) ∨ (JMO = 250 ∧ IMO = 750) :=
sorry

end word_sum_problems_l263_263193


namespace nominal_interest_rate_l263_263978

-- Define the conditions given in the problem
def effective_annual_rate : ℝ := 0.0609
def compounding_periods_per_year : ℝ := 2

-- State the goal: nominal interest rate per annum
theorem nominal_interest_rate : 
  ∃ i : ℝ, effective_annual_rate = (1 + i / compounding_periods_per_year)^ compounding_periods_per_year - 1 ∧ i ≈ 0.0598 := 
begin
  use 0.0598,
  split,
  { calc 
      (1 + 0.0598 / compounding_periods_per_year) ^ compounding_periods_per_year - 1
          = (1 + 0.0598 / 2) ^ 2 - 1 : by rw compounding_periods_per_year
      ... = (1 + 0.0299) ^ 2 - 1  : by rw [div_eq_mul_one_div, mul_one_div, mul_one, add_comm]
      ... = 1.0299 ^ 2 - 1 : by rw [← add_assoc]
      ... = 1.0609 - 1 : by sorry
      ... = 0.0609 : by ring },
  -- Completion of approximation step
  { sorry }
end

end nominal_interest_rate_l263_263978


namespace max_plus_shapes_l263_263348

def cover_square (x y : ℕ) : Prop :=
  3 * x + 5 * y = 49

theorem max_plus_shapes (x y : ℕ) (h1 : cover_square x y) (h2 : x ≥ 4) : y ≤ 5 :=
sorry

end max_plus_shapes_l263_263348


namespace sum_of_solutions_l263_263794

theorem sum_of_solutions :
  (∑ x in {x : ℝ | 2 * cos (2 * x) * (cos (2 * x) - cos (2000 * π^2 / x)) = cos (4 * x) - 1 ∧ 0 < x}, x) = 136 * π := by
  sorry

end sum_of_solutions_l263_263794


namespace smallest_value_of_y_l263_263305

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end smallest_value_of_y_l263_263305


namespace sample_is_subset_of_population_l263_263166

variables (Population Sample : Set ℕ) (population_size sample_size : ℕ)

def is_valid_sample (Population Sample : Set ℕ) (population_size : ℕ) (sample_size : ℕ) : Prop :=
  Population.card = population_size ∧ Sample.card = sample_size ∧ Sample ⊆ Population

theorem sample_is_subset_of_population :
  is_valid_sample Population Sample 70000 1000 → 
  Sample ⊆ Population := 
by
  intro h
  have h_subset : Sample ⊆ Population := h.2.2
  exact h_subset

end sample_is_subset_of_population_l263_263166


namespace number_of_integers_2017_satisfying_condition_l263_263039

theorem number_of_integers_2017_satisfying_condition :
  let S := {n ∈ Finset.range 2018 | (n - 2) * n * (n - 1) * (n - 7) % 1001 = 0} in
  S.card = 99 :=
by
  sorry

end number_of_integers_2017_satisfying_condition_l263_263039


namespace tetrahedron_inequality_l263_263900

variable {A B C D : Type*}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable (AB AC BC AD BD CD : ℝ)
variable (orthocenter : B)
variable (angle_BDC_right : ∀ (B D C : Type*), ∠ B D C = 90)

theorem tetrahedron_inequality
  (h1 : ∀ {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D], 
        ∠ B D C = 90)
  (h2 : ∀ {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D], 
        foot_perpendicular D (plane_to_ABC A B C) = orthocenter A B C):
  (AB + BC + CA) ^ 2 ≤ 6 * (AD ^ 2 + BD ^ 2 + CD ^ 2) ∧ 
  ∀ {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D],
  (AB + BC + CA) ^ 2 = 6 * (AD ^ 2 + BD ^ 2 + CD ^ 2) → equilateral_triangle A B C :=
sorry

end tetrahedron_inequality_l263_263900


namespace square_non_negative_is_universal_l263_263046

/-- The square of any real number is non-negative, which is a universal proposition. -/
theorem square_non_negative_is_universal : 
  ∀ x : ℝ, x^2 ≥ 0 :=
by
  sorry

end square_non_negative_is_universal_l263_263046


namespace shortest_distance_origin_to_line_l263_263705

-- Define the total number of items and ratios
def total_items : ℕ := 100
def ratio_first  : ℕ := 4
def ratio_second : ℕ := 3
def ratio_third  : ℕ := 2
def ratio_fourth : ℕ := 1

-- Define the number of items taken from each grade based on the given ratio
def items_first  : ℕ := (ratio_first  * total_items) / (ratio_first + ratio_second + ratio_third + ratio_fourth)
def items_third  : ℕ := (8 * ratio_third) / ratio_first
def items_fourth : ℕ := (8 * ratio_fourth) / ratio_first

-- Prove that the shortest distance from the origin to the line 2x + y + 8 = 0 is 8 * sqrt(5) / 5.
theorem shortest_distance_origin_to_line : 
  ∀ (a b : ℝ), a = items_third → b = items_fourth → 
          (2 * 0 + 1 * 0 + 8 : ℝ) / real.sqrt (2^2 + 1^2) = (8 * real.sqrt 5) / 5 :=
by 
  intros _ _ ha hb
  simp [ha, hb]
  norm_num
  sorry

end shortest_distance_origin_to_line_l263_263705


namespace part1_part2_l263_263464

noncomputable theory
open Real

def f (a x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1 (h : f 1 = λ x, -x^2 + x + 4) : 
  {x | f 1 x ≥ g x} = Icc (-1 : ℝ) ((sqrt 17 - 1) / 2) :=
by sorry

theorem part2 (h : ∀ x ∈ Icc (-1 : ℝ) 1, f a x ≥ g x) : 
  a ∈ Icc (-1 : ℝ) 1 :=
by sorry

end part1_part2_l263_263464


namespace inequality_solution_set_empty_range_l263_263873

theorem inequality_solution_set_empty_range (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_solution_set_empty_range_l263_263873


namespace rahim_books_l263_263592

theorem rahim_books (x : ℕ) 
  (h1 : 65 * (65 + x) = 2080 * 18.08695652173913)
  (h2 : 2080 / 18.08695652173913 ≈ 65 + x) : 
  x = 50 :=
begin
  sorry
end

end rahim_books_l263_263592


namespace students_in_first_bus_l263_263634

theorem students_in_first_bus (total_buses : ℕ) (avg_students_per_bus : ℕ) 
(avg_remaining_students : ℕ) (num_remaining_buses : ℕ) 
(h1 : total_buses = 6) 
(h2 : avg_students_per_bus = 28) 
(h3 : avg_remaining_students = 26) 
(h4 : num_remaining_buses = 5) :
  (total_buses * avg_students_per_bus - num_remaining_buses * avg_remaining_students = 38) :=
by
  sorry

end students_in_first_bus_l263_263634


namespace sin_pi_minus_alpha_l263_263048

theorem sin_pi_minus_alpha (
  hα1 : Real.pi / 2 < α,
  hα2 : α < Real.pi,
  h3 : 3 * Real.sin (2 * α) = 2 * Real.sin α
  ) :
  Real.sin (Real.pi - α) = 2 * Real.sqrt 2 / 3 :=
sorry

end sin_pi_minus_alpha_l263_263048


namespace Tian_Ji_wins_probability_l263_263908

structure Horse (name : String) :=
  (isTopTier  : Bool)
  (isMidTier  : Bool)
  (isBotTier  : Bool)

variable {A : Horse} {B : Horse} {C : Horse}
variable {a : Horse} {b : Horse} {c : Horse}

axiom Tian_Ji_top : a.isMidTier = True ∧ a.isTopTier = False
axiom Tian_Ji_mid : b.isBotTier = True ∧ b.isMidTier = False
axiom Tian_Ji_bot : c.isBotTier = True

theorem Tian_Ji_wins_probability : (∑ (x : Horse × Horse), if match x with
  | (A, C) | (B, C) | (B, A) | (C, B) | (C, A) => False
  | (_, _) => True by sorry 
/ 9 ) = 1 / 3 := by sorry

end Tian_Ji_wins_probability_l263_263908


namespace number_of_true_statements_l263_263875

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := 1/x
noncomputable def h (x : ℝ) : ℝ := 2 * Real.exp (Real.ln x)

lemma is_monotonically_increasing_F_on_interval : 
  ∀ x ∈ Ioo (- (1 / (2 : ℝ)^(1 / 3))) 0, deriv (λ x, f x - g x) x > 0 := 
sorry

lemma separation_line_f_g_exists :
  ∃ (k b : ℝ), (∀ x, f x ≥ k * x + b) ∧ (∀ x < 0, g x ≤ k * x + b) ∧ b = -4 := 
sorry

lemma separation_line_f_g_k_range :
  ∃ (k b : ℝ), (-4 < k ∧ k ≤ 0) ∧ (∀ x, f x ≥ k * x + b) ∧ (∀ x < 0, g x ≤ k * x + b) := 
sorry

lemma unique_separation_line_f_h :
  ∃! (k b : ℝ), (∀ x, f x ≥ k * x + b) ∧ (∀ x > 0, h x ≤ k * x + b) ∧ k = 2 * Real.sqrt 2 ∧ b = - Real.exp 1 := 
sorry

theorem number_of_true_statements : 
  (is_monotonically_increasing_F_on_interval ∧ separation_line_f_g_exists ∧ ¬ separation_line_f_g_k_range ∧ unique_separation_line_f_h) = 3 := 
sorry

end number_of_true_statements_l263_263875


namespace product_of_roots_l263_263730

theorem product_of_roots : 
  let p := (3 * x ^ 3 + 2 * x ^ 2 - 5 * x + 15) * (4 * x ^ 3 - 12 * x ^ 2 + 8 * x - 24) in
  ((∀ x, p = 0) → (∏ root in (roots_of_polynomial p), root) = -30) := 
by
  sorry

end product_of_roots_l263_263730


namespace no_solution_for_k_eq_2_l263_263402

theorem no_solution_for_k_eq_2 :
  ∀ m n : ℕ, m ≠ n → ¬ (lcm m n - gcd m n = 2 * (m - n)) :=
by
  sorry

end no_solution_for_k_eq_2_l263_263402


namespace number_of_5_tuples_is_odd_l263_263599

theorem number_of_5_tuples_is_odd : 
  ∃ (f : ℕ → ℕ → ℕ → ℕ → ℕ → Prop), 
  (∀ a b c d e : ℕ, f a b c d e ↔ a * b * c * d * e = 5 * (b * c * d * e + a * c * d * e + a * b * d * e + a * b * c * e + a * b * c * d)) ∧
  (∃ n : ℕ, n = 121) ∧ 
  (nat_odd n) :=
sorry

end number_of_5_tuples_is_odd_l263_263599


namespace sin_alpha_eq_sin_beta_l263_263157

theorem sin_alpha_eq_sin_beta (α β : Real) (k : Int) 
  (h_symmetry : α + β = 2 * k * Real.pi + Real.pi) : 
  Real.sin α = Real.sin β := 
by 
  sorry

end sin_alpha_eq_sin_beta_l263_263157


namespace exists_one_friend_l263_263006

variable {A : Type} [Fintype A] (friend_relation : A → A → Prop)

-- Definitions used in conditions
def friends (a : A) : Finset A := Finset.univ.filter (friend_relation a)

def unique_friends (a b : A) : Prop :=
  ∀ x, a ≠ b → (friends a ∩ friends b).card = 0 ∧ friends a.card = friends b.card → friends a ≠ friends b

-- The main theorem statement
theorem exists_one_friend (h1 : ∀ a b : A, a ≠ b → unique_friends a b) :
  ∃ a : A, (friends a).card = 1 :=
by {
  sorry -- proof omitted
}

end exists_one_friend_l263_263006


namespace cosine_decreasing_interval_l263_263018

theorem cosine_decreasing_interval : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ π / 2 → cos (2 * x₁) > cos (2 * x₂) := 
by
  intros x₁ x₂ h0 hx hpi
  sorry

end cosine_decreasing_interval_l263_263018


namespace factorization_count_l263_263183

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l263_263183


namespace pizza_party_l263_263337

theorem pizza_party (boys girls : ℕ) :
  (7 * boys + 3 * girls ≤ 59) ∧ (6 * boys + 2 * girls ≥ 49) ∧ (boys + girls ≤ 10) → 
  boys = 8 ∧ girls = 1 := 
by sorry

end pizza_party_l263_263337


namespace semi_circle_radius_l263_263276

theorem semi_circle_radius (P : ℝ) (r : ℝ) (π : ℝ) (h_perimeter : P = 113) (h_pi : π = Real.pi) :
  r = P / (π + 2) :=
sorry

end semi_circle_radius_l263_263276


namespace max_red_points_l263_263639

-- We start by defining the conditions as given in the problem

-- There are 100 points marked on a circle, which we will model as a finite set of 100 elements.
constant points : Finset ℕ
constant red blue : Finset ℕ

-- Hypothesize 100 points in total
axiom points_count : points.card = 100

-- These points are either red or blue, and together they partition the points set
axiom red_blue_partition : red ∪ blue = points
axiom disjoint_red_blue : Disjoint red blue

-- Each segment connects one red point to one blue point. This can be modeled as a function from
-- red points to blue points indicating the connections.
constant segments : red → blue

-- Ensure no two red points are connected to the same number of blue points
axiom unique_connections : ∀ (p1 p2 : red), p1 ≠ p2 → (segments p1 ≠ segments p2)

-- The goal is to prove the maximum number of red points
theorem max_red_points : red.card ≤ 50 :=
by sorry


end max_red_points_l263_263639


namespace non_congruent_triangles_with_perimeter_11_l263_263139

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263139


namespace pdf_of_random_point_A_in_square_l263_263583

noncomputable def pdf (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then π * x / 2
else if 1 ≤ x ∧ x < real.sqrt 2 then 
  π * x / 2 - 2 * x * real.arccos (1 / x)
else 0

theorem pdf_of_random_point_A_in_square (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ real.sqrt 2) :
  ∀ u v, 0 ≤ u → u ≤ 1 → 0 ≤ v → v ≤ 1 → 
  let ξ := real.sqrt (u^2 + v^2) in
  pdf ξ = 
  if 0 < ξ ∧ ξ < 1 then π * ξ / 2
  else if 1 ≤ ξ ∧ ξ < real.sqrt 2 then 
    π * ξ / 2 - 2 * ξ * real.arccos (1 / ξ)
  else 0 := 
sorry

end pdf_of_random_point_A_in_square_l263_263583


namespace ratio_paislee_to_calvin_l263_263000

theorem ratio_paislee_to_calvin (calvin_points paislee_points : ℕ) (h1 : calvin_points = 500) (h2 : paislee_points = 125) : paislee_points / calvin_points = 1 / 4 := by
  sorry

end ratio_paislee_to_calvin_l263_263000


namespace a_eq_2_sufficient_but_not_necessary_a_eq_2_not_necessary_l263_263814

def is_pure_imaginary (z : ℂ) : Prop := (z.re = 0) ∧ (z.im ≠ 0)

theorem a_eq_2_sufficient_but_not_necessary (a : ℝ) :
  (is_pure_imaginary ((a^2 - 4 : ℝ) + (a - 3 : ℂ) * complex.I) ∧ (a = 2)) :=
by
  sorry

theorem a_eq_2_not_necessary (a : ℝ) : 
  (¬a = 2) → (is_pure_imaginary ((a^2 - 4 : ℝ) + (a - 3 : ℂ) * complex.I)) → 
  (is_pure_imaginary ((a^2 - 4 : ℝ) + (a - 3 : ℂ) * complex.I)) :=
by
  sorry

end a_eq_2_sufficient_but_not_necessary_a_eq_2_not_necessary_l263_263814


namespace attainable_tables_count_l263_263056

theorem attainable_tables_count (m n : ℕ) (table : fin (2 * n) → fin (2 * m) → bool)
  (is_attainable : (fin (2 * n) → fin (2 * m) → bool) → Prop) :
  (m % 2 = 1 ∧ n % 2 = 1 → ∃ k, k = 2^(m + n - 2) ∧ is_attainable table) ∧
  (¬(m % 2 = 1 ∧ n % 2 = 1) → ∃ k, k = 2^(m + n - 1) ∧ is_attainable table) := 
sorry

end attainable_tables_count_l263_263056


namespace ab_eq_neg_one_l263_263219

variable (a b : ℝ)

-- Condition for the inequality (x >= 0) -> (0 ≤ x^4 - x^3 + ax + b ≤ (x^2 - 1)^2)
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → 
    0 ≤ x^4 - x^3 + a * x + b ∧ 
    x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2

-- Main statement to prove that assuming the condition, a * b = -1
theorem ab_eq_neg_one (h : condition a b) : a * b = -1 := 
  sorry

end ab_eq_neg_one_l263_263219


namespace sequence_a_general_formula_Tn_less_than_three_fourth_l263_263572

-- Definitions and assumptions
def sequence_a : ℕ → ℤ
| 1 := 1
| n := 3 * n - 2

def S (n : ℕ) := (n * (3 * n - 1)) / 2

def b (n : ℕ) := 3 / (2 * S n + 7 * n)

def T (n : ℕ) := (Finset.range n).sum (b ∘ Nat.succ)

-- Theorems to prove
theorem sequence_a_general_formula : ∀ n : ℕ, sequence_a n = 3 * n - 2 :=
begin
  sorry
end

theorem Tn_less_than_three_fourth : ∀ n : ℕ, T n < 3 / 4 :=
begin
  sorry
end

end sequence_a_general_formula_Tn_less_than_three_fourth_l263_263572


namespace integer_points_on_parabola_l263_263560

noncomputable def parabola (p : ℝ × ℝ) : Prop :=
  let focus := (0 : ℝ, 2)
  let directrix := line (0 : ℝ, 1) (-1 : ℝ, 0)
  (dist p focus) = (dist p (proj_on_line p directrix))

theorem integer_points_on_parabola :
  let points_on_q := {p : ℝ × ℝ | parabola p} in
  let integer_points_on_q := {p : ℝ × ℝ | p ∈ points_on_q ∧ ∃ x y : ℤ, p = (x, y)} in
  let valid_points := {p : ℝ × ℝ | p ∈ integer_points_on_q ∧ abs (5 * p.1 + 4 * p.2) ≤ 1200} in
  card valid_points = 617 :=
by
  sorry

end integer_points_on_parabola_l263_263560


namespace six_people_six_chairs_l263_263887

theorem six_people_six_chairs : 
  let n := 6 in 
  (Finset.univ.Perm n).card = n.factorial :=
by 
  have h : 6.factorial = 720 := rfl
  exact h

end six_people_six_chairs_l263_263887


namespace eval_frac_l263_263795

theorem eval_frac 
  (a b : ℚ)
  (h₀ : a = 7) 
  (h₁ : b = 2) :
  3 / (a + b) = 1 / 3 :=
by
  sorry

end eval_frac_l263_263795


namespace cut_triangle_to_form_20_sided_polygon_l263_263624

theorem cut_triangle_to_form_20_sided_polygon (T : Triangle) :
  ∃ (parts : List (Polygon)) (H1 : length parts = 2) (H2 : ∀ p ∈ parts, is_polygon p),
  (∃ (P : Polygon) (H : num_sides P = 20), ∀ part ∈ parts, part ⊆ P) :=
sorry

end cut_triangle_to_form_20_sided_polygon_l263_263624


namespace triangle_cosine_identity_l263_263825

theorem triangle_cosine_identity (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  real.cos A = 7 / 25 :=
by
  sorry

end triangle_cosine_identity_l263_263825


namespace pq_squared_over_mn_equals_2_sqrt_2_l263_263827

noncomputable def ellipse : Type := { p : ℝ × ℝ // (p.1^2) / 2 + (p.2^2) = 1 }

variables (F : ℝ × ℝ)
  (h_F : F = (-1, 0))
  (M N P Q : ellipse)
  -- slope of line (MN), parallel to (PQ)
  (k : ℝ)
  -- line (MN) and line (PQ) equations  
  (h_MN : ∃ k, ∃ b, ∀ (x : ℝ) (w : ℝ × ℝ), ((w.1 = x) → (w.2 = k * x + b)) ∧ (M.1 = w) ∧ (N.1 = w))
  (h_PQ : ∃ k, ∀ (x : ℝ) (z : ellipse), (z.1 = (0, 0) ∨ z.1 = x) ∧ ((z.2).1 = x) ∧ (k = k))
  -- points of intersection with ellipse
  (h_intersect_MN : ∀ p : ellipse, p = M ∨ p = N → ((p.1).1^2 / 2 + (p.1).2^2 = 1))
  (h_intersect_PQ : ∀ p : ellipse, p = P ∨ p = Q → ((p.1).1^2 / 2 + (p.1).2^2 = 1) ∧ p.1 = (0, 0))
  
-- To prove
theorem pq_squared_over_mn_equals_2_sqrt_2 :
  (|PQ|^2 / |MN| = 2 * real.sqrt 2) := by
sory

end pq_squared_over_mn_equals_2_sqrt_2_l263_263827


namespace total_amount_correct_l263_263693

noncomputable def total_amount : ℝ :=
  let nissin_noodles := 24 * 1.80 * 0.80
  let master_kong_tea := 6 * 1.70 * 0.80
  let shanlin_soup := 5 * 3.40
  let shuanghui_sausage := 3 * 11.20 * 0.90
  nissin_noodles + master_kong_tea + shanlin_soup + shuanghui_sausage

theorem total_amount_correct : total_amount = 89.96 := by
  sorry

end total_amount_correct_l263_263693


namespace number_divisibility_l263_263658

theorem number_divisibility (a b : ℕ) (x : ℕ) (h1 : a = 722425) (h2 : b = 335) (h3 : x = a + b):
  x % 30 = 0 :=
by
  have h4 : x = 722760 := by
    rw [h1, h2]
    exact rfl

  -- The proof would follow here
  sorry

end number_divisibility_l263_263658


namespace problem_1_problem_2_l263_263162

variable (A B C a b c : ℝ)
variable (triangle_ABC : a = b * Real.sin A)
variable (h1 : sqrt (3:ℝ) * b * Real.sin A = a * Real.cos B)

theorem problem_1 :
  B = π / 6 :=
  sorry

variable (h2 : b = 3)
variable (h3 : Real.sin C = sqrt (3:ℝ) * Real.sin A)

theorem problem_2 :
  a = 3 ∧ c = 3 * sqrt (3:ℝ) :=
  sorry

end problem_1_problem_2_l263_263162


namespace discount_rate_on_pony_jeans_l263_263416

-- Define the conditions as Lean definitions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.91
def total_discount_rate : ℝ := 22
def number_of_fox_pairs : ℕ := 3
def number_of_pony_pairs : ℕ := 2

-- Given definitions of the discount rates on Fox and Pony jeans
variable (F P : ℝ)

-- The system of equations based on the conditions
axiom sum_of_discount_rates : F + P = total_discount_rate
axiom savings_equation : 
  number_of_fox_pairs * (fox_price * F / 100) + number_of_pony_pairs * (pony_price * P / 100) = total_savings

-- The theorem to prove
theorem discount_rate_on_pony_jeans : P = 11 := by
  sorry

end discount_rate_on_pony_jeans_l263_263416


namespace Prod_tan_squared_expression_l263_263861

theorem Prod_tan_squared_expression : 
  (∏ i in finset.range (2022 - 6), (1 - tan (2^i * real.pi / 180)^2)) = 2^2016 → 
  ∃ a b : ℕ, squarefree a ∧ (a + b = 2018) ∧ (2^2016 = a^b) :=
begin
  intro h,
  use [2, 2016],
  split,
  { exact nat.squarefree_two },
  split,
  { refl },
  { exact h }
end

end Prod_tan_squared_expression_l263_263861


namespace remainder_mod_500_l263_263774

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l263_263774


namespace fewest_printers_l263_263317

theorem fewest_printers (cost1 cost2 : ℕ) (h1 : cost1 = 375) (h2 : cost2 = 150) : 
  ∃ (n : ℕ), n = 2 + 5 :=
by
  have lcm_375_150 : Nat.lcm cost1 cost2 = 750 := sorry
  have n1 : 750 / 375 = 2 := sorry
  have n2 : 750 / 150 = 5 := sorry
  exact ⟨7, rfl⟩

end fewest_printers_l263_263317


namespace second_order_arithmetic_sequence_term_15_l263_263965

theorem second_order_arithmetic_sequence_term_15 : 
  let a := [2, 3, 6, 11] in
  ∃ (an : ℕ → ℝ), 
    (∀ n, an n = (n^2 - 2*n + 3)) ∧ an 1 = 2 ∧ an 2 = 3 ∧ an 3 = 6 ∧ an 4 = 11 →
    an 15 = 198 :=
by 
  sorry

end second_order_arithmetic_sequence_term_15_l263_263965


namespace fraction_of_tomato_plants_in_second_garden_l263_263242

theorem fraction_of_tomato_plants_in_second_garden 
    (total_plants_first_garden : ℕ := 20)
    (percent_tomato_first_garden : ℚ := 10 / 100)
    (total_plants_second_garden : ℕ := 15)
    (percent_total_tomato_plants : ℚ := 20 / 100) :
    (15 : ℚ) * (1 / 3) = 5 :=
by
  sorry

end fraction_of_tomato_plants_in_second_garden_l263_263242


namespace max_value_f_on_interval_l263_263988

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f_on_interval :
    ∀ x ∈ Ioo 0 (Real.pi / 2), (f x) ≤ 3 * Real.sqrt 3 :=
sorry

end max_value_f_on_interval_l263_263988


namespace min_distance_on_bisector_l263_263833

-- Define our variables and the problem setup
variables {A X Y : Type*} [metric_space A] [metric_space X] [metric_space Y] -- defining the spaces

-- Definition of angle XAY
def is_angle (A X Y : Type*) [metric_space A] [metric_space X] [metric_space Y] : Prop := sorry

-- Definition of a circle within the angle
def exists_circle_in_angle (A X Y C : Type*) [metric_space A] [metric_space X] [metric_space Y] [metric_space C] : Prop := 
sorry

-- Now formulating the theorem per problem statement
theorem min_distance_on_bisector (A X Y C M : Type*) [metric_space A] [metric_space X] [metric_space Y] [metric_space C] [metric_space M]
  (h_angle: is_angle A X Y)
  (h_circle: exists_circle_in_angle A X Y C)
  (h_point_on_circle: M ∈ circle) :
  (∃ M : A, sum_of_distances_to_lines M AX AY = minimal) →
  lies_on_angle_bisector A X Y M :=
begin
  sorry
end

end min_distance_on_bisector_l263_263833


namespace second_order_arithmetic_sequence_term_15_l263_263962

theorem second_order_arithmetic_sequence_term_15 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 6) ∧ (a 4 = 11) ∧ 
  (∀ n, n ≥ 2 → a (n + 1) - a n = (a (n + 1) - a n)- (a n - a (n-1))) →
  (a 15 = 198) :=
by 
  intro a h,
  obtain ⟨h1, h2, h3, h4, h_pattern⟩ := h,
  sorry -- placeholder for the proof

end second_order_arithmetic_sequence_term_15_l263_263962


namespace range_of_absolute_difference_l263_263813

noncomputable def polynomial := fun (x : ℝ) (b c d : ℝ) => x^3 + b * x^2 + c * x + d

theorem range_of_absolute_difference (b c d : ℝ) (h1 : ∀ x < 0, 3 * x^2 + 2 * b * x + c ≥ 0)
                                   (h2 : ∀ x ∈ Icc 0 2, 3 * x^2 + 2 * b * x + c ≤ 0)
                                   (h3 : ∀ x, polynomial x b c d = 0 → x = α ∨ x = 2 ∨ x = β)
                                   (h4 : c = 0) (h5 : b ≤ -3) :
  3 ≤ |α - β| :=
sorry

end range_of_absolute_difference_l263_263813


namespace cube_painting_possible_min_purple_faces_l263_263659

/-- Part (a) of the problem: Is it possible to paint the cubes to form the desired structures? -/
theorem cube_painting_possible :
  ∃ (coloring : (ℤ × ℤ × ℤ) → Fin 3 → Prop),
    (∀ i j k ⟨vi, vj, vk⟩,
      (coloring (i, j, k) vi ↔ vi = 0 ∨ vi = 1 ∨ vi = 2) ∧
      (∀ face, ∃ i' j' k' vi' vj' vk', coloring (i', j', k') face)) :=
sorry

/-- Part (b) of the problem: Minimum number of purple faces needed. -/
theorem min_purple_faces (n : ℕ) (h : ∀ c : ℕ, n = 151) : Prop :=
sorry

end cube_painting_possible_min_purple_faces_l263_263659


namespace second_order_arithmetic_sequence_term_15_l263_263964

theorem second_order_arithmetic_sequence_term_15 : 
  let a := [2, 3, 6, 11] in
  ∃ (an : ℕ → ℝ), 
    (∀ n, an n = (n^2 - 2*n + 3)) ∧ an 1 = 2 ∧ an 2 = 3 ∧ an 3 = 6 ∧ an 4 = 11 →
    an 15 = 198 :=
by 
  sorry

end second_order_arithmetic_sequence_term_15_l263_263964


namespace unique_number_not_in_range_l263_263620

noncomputable def f (a b c d x : ℝ) := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range (a b c d : ℝ) 
    (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : f a b c d 19 = 19) 
    (H2 : f a b c d 97 = 97) 
    (H3 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
    ∃ y : ℝ, ∀ x : ℝ, f a b c d x ≠ y :=
begin
  use 58,
  intro x,
  sorry
end

end unique_number_not_in_range_l263_263620


namespace count_equilateral_triangles_in_grid_l263_263098

-- Define the possible triangle sizes and their corresponding counts
def count_upward_triangles := 10 + 6 + 3 + 1
def count_downward_triangles := 6 + 1
def count_sqrt3_triangles := 3 + 3
def count_sqrt7_triangles := 2

-- The main proof problem that the sum of all these is 35
theorem count_equilateral_triangles_in_grid : 
  count_upward_triangles + count_downward_triangles + count_sqrt3_triangles + count_sqrt7_triangles = 35 :=
by 
  -- We add the numbers directly, ensuring no sub-definitions assume additional solution knowledge
  have h1 : count_upward_triangles = 10 + 6 + 3 + 1 := rfl,
  have h2 : count_downward_triangles = 6 + 1 := rfl,
  have h3 : count_sqrt3_triangles = 3 + 3 := rfl,
  have h4 : count_sqrt7_triangles = 2 := rfl,
  calc
    count_upward_triangles + count_downward_triangles + count_sqrt3_triangles + count_sqrt7_triangles
    = 20 + 7 + 6 + 2 : by rw [h1, h2, h3, h4]
    ... = 35 : by norm_num

end count_equilateral_triangles_in_grid_l263_263098


namespace fraction_of_shaded_hexagons_l263_263187

-- Definitions
def total_hexagons : ℕ := 9
def shaded_hexagons : ℕ := 5

-- Theorem statement
theorem fraction_of_shaded_hexagons : 
  (shaded_hexagons: ℚ) / (total_hexagons : ℚ) = 5 / 9 := by
sorry

end fraction_of_shaded_hexagons_l263_263187


namespace intersection_of_A_and_B_l263_263439

def A : Set ℝ := { x | x^2 - 5 * x - 6 ≤ 0 }

def B : Set ℝ := { x | x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l263_263439


namespace max_score_possible_l263_263287

-- Definition of the problem conditions
def ordered_pairs (S : Type) := S × S

def valid_pairs (pairs : List (ordered_pairs ℤ)) : Prop :=
  ∀ k : ℤ, ¬((k, k) ∈ pairs ∧ (-k, -k) ∈ pairs)

def score (pairs : List (ordered_pairs ℤ)) (erased : Set ℤ) : ℕ :=
  pairs.countp (λ p, p.1 ∈ erased ∨ p.2 ∈ erased)

noncomputable def max_guaranteed_score : ℕ :=
  43

-- Theorem stating the maximum possible score
theorem max_score_possible (pairs : List (ordered_pairs ℤ))
  (h_valid : valid_pairs pairs) (erased : Set ℤ) (h_no_cancel : ∀ x ∈ erased, -x ∉ erased) :
  ∃ N, N = score pairs erased ∧ N ≤ max_guaranteed_score :=
sorry

end max_score_possible_l263_263287


namespace symmetry_center_of_g_l263_263570

open Real

noncomputable def g (x : ℝ) : ℝ := cos ((1 / 2) * x - π / 6)

def center_of_symmetry : Set (ℝ × ℝ) := { p | ∃ k : ℤ, p = (2 * k * π + 4 * π / 3, 0) }

theorem symmetry_center_of_g :
  (∃ p : ℝ × ℝ, p ∈ center_of_symmetry) :=
sorry

end symmetry_center_of_g_l263_263570


namespace fraction_covered_by_triangle_l263_263711

structure Point where
  x : ℤ
  y : ℤ

def area_of_triangle (A B C : Point) : ℚ :=
  (1/2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_grid (length width : ℤ) : ℚ :=
  (length * width : ℚ)

def fraction_of_grid_covered (A B C : Point) (length width : ℤ) : ℚ :=
  (area_of_triangle A B C) / (area_of_grid length width)

theorem fraction_covered_by_triangle :
  fraction_of_grid_covered ⟨2, 4⟩ ⟨7, 2⟩ ⟨6, 5⟩ 8 6 = 13 / 96 :=
by
  sorry

end fraction_covered_by_triangle_l263_263711


namespace area_of_figure_l263_263033

theorem area_of_figure :
  let D := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ 2 * (|p.1| - |p.2|) }
  (area D) = 2 * Real.pi - 4 :=
by
  sorry

end area_of_figure_l263_263033


namespace log_arithmetic_progression_l263_263247

variable (a b c P : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)
variable (h4 : 0 < P)
variable (h5 : a ≠ 1)
variable (h6 : b ≠ 1)
variable (h7 : c ≠ 1)
variable (h8 : (a * c) ^ real.log a b = c ^ 2)

theorem log_arithmetic_progression :
  real.log a P + real.log c P = 2 * real.log b P :=
sorry

end log_arithmetic_progression_l263_263247


namespace sum_of_candy_quantities_l263_263307

def is_solution (N : ℕ) : Prop :=
  N % 6 = 4 ∧ N % 8 = 5 ∧ N < 100

theorem sum_of_candy_quantities : (Finset.filter is_solution (Finset.range 100)).sum = 74 := by sorry

end sum_of_candy_quantities_l263_263307


namespace max_red_points_l263_263638

-- We start by defining the conditions as given in the problem

-- There are 100 points marked on a circle, which we will model as a finite set of 100 elements.
constant points : Finset ℕ
constant red blue : Finset ℕ

-- Hypothesize 100 points in total
axiom points_count : points.card = 100

-- These points are either red or blue, and together they partition the points set
axiom red_blue_partition : red ∪ blue = points
axiom disjoint_red_blue : Disjoint red blue

-- Each segment connects one red point to one blue point. This can be modeled as a function from
-- red points to blue points indicating the connections.
constant segments : red → blue

-- Ensure no two red points are connected to the same number of blue points
axiom unique_connections : ∀ (p1 p2 : red), p1 ≠ p2 → (segments p1 ≠ segments p2)

-- The goal is to prove the maximum number of red points
theorem max_red_points : red.card ≤ 50 :=
by sorry


end max_red_points_l263_263638


namespace find_f_1_2016_l263_263696

theorem find_f_1_2016 (f : ℝ → ℝ) 
  (h1 : f 0 = 0)
  (h2 : ∀ x ∈ Icc 0 1, f x + f (1 - x) = 1)
  (h3 : ∀ x ∈ Icc 0 1, f (x / 3) = (1 / 2) * f x)
  (h4 : ∀ x1 x2 ∈ Icc 0 1, x1 ≤ x2 → f x1 ≤ f x2) :
  f (1 / 2016) = 1 / 128 :=
sorry

end find_f_1_2016_l263_263696


namespace pyramid_volume_is_correct_l263_263924

noncomputable def volume_pyramid (A B C G : ℝ × ℝ × ℝ) : ℝ :=
  let base_area := 1 / 2 * (2 * 2) in
  let height := 2 in
  1 / 3 * base_area * height

theorem pyramid_volume_is_correct
  (A B C G : ℝ × ℝ × ℝ)
  (hA : A = (0, 0, 0))
  (hB : B = (2, 0, 0))
  (hC : C = (0, 2, 0))
  (hG : G = (0, 0, 2))
  (side_length : ℝ)
  (side_length_eq : side_length = 2) :
  volume_pyramid A B C G = 4 / 3 :=
by
  rw [hA, hB, hC, hG, side_length_eq]
  sorry

end pyramid_volume_is_correct_l263_263924


namespace point_min_dist_sum_l263_263400

noncomputable def find_min_point (A B C : Point) : Point :=
  let OL := dist_to_side O B C
  let OM := dist_to_side O A C
  let ON := dist_to_side O A B
  if (BC / AC = k) ∧ (AC / AB = k) ∧ (AB / BC = k) then
    let k := BC / OL
    {some point O such that
    OL^2 + OM^2 + ON^2 is minimized}
  else
    none

theorem point_min_dist_sum (A B C : Point) : 
  ∃ O : Point, inside_triangle O A B C ∧ 
    ∀ O_1 : Point, inside_triangle O_1 A B C →
      (dist_to_side O B C)^2 + (dist_to_side O A C)^2 + (dist_to_side O A B)^2 ≤
      (dist_to_side O_1 B C)^2 + (dist_to_side O_1 A C)^2 + (dist_to_side O_1 A B)^2 :=
sorry

end point_min_dist_sum_l263_263400


namespace least_odd_prime_factor_1234_power_10_plus_1_l263_263406

theorem least_odd_prime_factor_1234_power_10_plus_1 :
  ∀ p : ℕ, prime p ∧ p ∣ (1234 ^ 10 + 1) ∧ odd p → p = 61 := by
  sorry

end least_odd_prime_factor_1234_power_10_plus_1_l263_263406


namespace seeds_planted_on_wednesday_l263_263333

theorem seeds_planted_on_wednesday
  (total_seeds : ℕ) (seeds_thursday : ℕ) (seeds_wednesday : ℕ)
  (h_total : total_seeds = 22) (h_thursday : seeds_thursday = 2) :
  seeds_wednesday = 20 ↔ total_seeds - seeds_thursday = seeds_wednesday :=
by
  -- the proof would go here
  sorry

end seeds_planted_on_wednesday_l263_263333


namespace smallest_number_of_points_in_set_satisfying_symmetries_l263_263346

theorem smallest_number_of_points_in_set_satisfying_symmetries :
  ∃ (T : set (ℝ × ℝ)), (1, 4) ∈ T ∧
  (∀ (x y : ℝ), (x, y) ∈ T → (-x, -y) ∈ T) ∧
  (∀ (x y : ℝ), (x, y) ∈ T → (y, x) ∈ T) ∧
  (∀ (x y : ℝ), (x, y) ∈ T → (-y, x) ∈ T) ∧
  (∀ (T' : set (ℝ × ℝ)), (1, 4) ∈ T' ∧
  (∀ (x y : ℝ), (x, y) ∈ T' → (-x, -y) ∈ T') ∧
  (∀ (x y : ℝ), (x, y) ∈ T' → (y, x) ∈ T') ∧
  (∀ (x y : ℝ), (x, y) ∈ T' → (-y, x) ∈ T') →
  set.card T' ≥ 8) :=
begin
  sorry
end

end smallest_number_of_points_in_set_satisfying_symmetries_l263_263346


namespace trapezoid_height_l263_263976

-- Define the problem conditions
variables {a b : ℝ}
-- Define the height of the trapezoid
noncomputable def height_of_trapezoid (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

-- State the problem: proving the height formula given the diagonal and midline
theorem trapezoid_height (a b : ℝ) : height_of_trapezoid a b = real.sqrt (a^2 - b^2):=
by sorry

end trapezoid_height_l263_263976


namespace find_x_pow_3a_minus_b_l263_263053

variable (x : ℝ) (a b : ℝ)
theorem find_x_pow_3a_minus_b (h1 : x^a = 2) (h2 : x^b = 9) : x^(3 * a - b) = 8 / 9 :=
  sorry

end find_x_pow_3a_minus_b_l263_263053


namespace compare_angles_l263_263500

-- Define the acute triangle KLM with angle KLM = 68 degrees
def acute_triangle (K L M : Type) (angle_KLM : ℝ) : Prop :=
  angle_KLM = 68 ∧ ∀ α, α ∈ {α : ℝ | 0 < α ∧ α < 90}

-- Define the orthocenter V of the triangle KLM
def orthocenter (K L M V : Type) : Prop :=
  -- The orthocenter property here would normally be defined with respect to the perpendicular altitudes intersecting.

-- Define the foot of the altitude P from vertex K to side LM
def foot_of_altitude (K L M P : Type) (V : Type) : Prop :=
  -- The foot of the altitude property here would normally involve the perpendicular drop from K to LM intersecting at P.

-- Define the angle bisector property where the angle bisector of PVM is parallel to side KM
def angle_bisector_parallel (K L M V P : Type) : Prop :=
  -- This property involves defining the angle bisector of ∠PVM and showing it's parallel to KM.

-- The main theorem
theorem compare_angles (K L M V P : Type) (angle_KLM : ℝ) 
  (h₁ : acute_triangle K L M angle_KLM)
  (h₂ : orthocenter K L M V)
  (h₃ : foot_of_altitude K L M P V)
  (h₄ : angle_bisector_parallel K L M V P) :
  ∃ α : ℝ, ∃ β : ℝ, α = β :=
begin
  sorry
end

end compare_angles_l263_263500


namespace count_integers_abs_leq_4_l263_263851

theorem count_integers_abs_leq_4 : 
  let solution_set := {x : Int | |x - 3| ≤ 4}
  ∃ n : Nat, n = 9 ∧ (∀ x ∈ solution_set, x ∈ finset.range 9) := sorry

end count_integers_abs_leq_4_l263_263851


namespace roots_of_equation_l263_263393

theorem roots_of_equation (x : ℝ) : 
  (\frac{21}{x^2 - 9} - \frac{3}{x - 3} = 1) ↔ (x = 3 ∨ x = -7) := 
sorry

end roots_of_equation_l263_263393


namespace minimum_value_l263_263521

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l263_263521


namespace greatest_value_of_a_plus_b_l263_263478

-- Definition of the problem conditions
def is_pos_int (n : ℕ) := n > 0

-- Lean statement to prove the greatest possible value of a + b
theorem greatest_value_of_a_plus_b :
  ∃ a b : ℕ, is_pos_int a ∧ is_pos_int b ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 9) ∧ a + b = 100 :=
sorry  -- Proof omitted

end greatest_value_of_a_plus_b_l263_263478


namespace coefficient_linear_term_l263_263455

theorem coefficient_linear_term (x : ℝ) : 
  let eq := 5 * x - 2 = 3 * x ^ 2 in 
  let general_form := 3 * x ^ 2 - 5 * x + 2 = 0 in 
  general_form → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ c = 2) → b = -5 :=
by
  intros eq general_form h
  obtain ⟨a, b, c, ha, hb, hc⟩ := h
  exact hb

end coefficient_linear_term_l263_263455


namespace matrix_eigenvector_power_l263_263913

variable {𝕂 : Type*} [Field 𝕂]
variable (B : Matrix (Fin 2) (Fin 2) 𝕂)

theorem matrix_eigenvector_power (h : B.mul_vec ![3, -1] = (![12, -4] : Fin 2 → 𝕂)) :
    (B ^ 4).mul_vec ![3, -1] = (![768, -256] : Fin 2 → 𝕂) :=
by
  sorry

end matrix_eigenvector_power_l263_263913


namespace angle_AMH_l263_263586

-- Define the necessary geometrical points and properties
variables (A B C M L H : Type)

-- Assuming the properties of the given problem
def isosceles_right_triangle (A B C : Type) : Prop :=
∃ (AB BC : ℝ), ∠B = 90 ∧ AB = BC

def midpoint (M A B : Type) : Prop :=
∃ (AM MB : ℝ), AM = MB

def angle_bisector_intersects_circumcircle (A L : Type) (ABC : Type) : Prop :=
∃ (circ_ABC : Type), L ∈ circ_ABC

def perpendicular_foot (H L : Type) (AC : Type) : Prop :=
∃ (LH HA : ℝ), H ∈ AC ∧ LH ⊥ AC

-- The theorem stating the goal to be proved
theorem angle_AMH {A B C M L H : Type}
  (hABC : isosceles_right_triangle A B C)
  (hM : midpoint M A B)
  (hL : angle_bisector_intersects_circumcircle A L (triangle_ABC A B C))
  (hH : perpendicular_foot H L (line_AC A C)) :
∠AMH = 112.5 := sorry

end angle_AMH_l263_263586


namespace tv_station_ads_l263_263706

theorem tv_station_ads (n m : ℕ) :
  n > 1 → 
  ∃ (an : ℕ → ℕ), 
  (an 0 = m) ∧ 
  (∀ k, 1 ≤ k ∧ k < n → an k = an (k - 1) - (k + (1 / 8) * (an (k - 1) - k))) ∧
  an n = 0 →
  (n = 7 ∧ m = 49) :=
by
  intro h
  exists sorry
  sorry

-- The proof steps are omitted

end tv_station_ads_l263_263706


namespace triangle_count_with_perimeter_11_l263_263099

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263099


namespace equality_of_sets_l263_263589

theorem equality_of_sets 
  (a b c x y z : ℕ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : x^2 + y^2 = z^2)
  (h3 : |x - a| ≤ 1)
  (h4 : |y - b| ≤ 1) :
  {a, b} = {x, y} := sorry

end equality_of_sets_l263_263589


namespace angles_sum_eq_l263_263197

variables {a b c : ℝ} {A B C : ℝ}

theorem angles_sum_eq {a b c : ℝ} {A B C : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π)
  (h8 : (a + c - b) * (a + c + b) = 3 * a * c) :
  A + C = 2 * π / 3 :=
sorry

end angles_sum_eq_l263_263197


namespace disk_tangent_position_after_full_rotation_l263_263268

def clock_face_radius : ℝ := 30
def disk_radius : ℝ := 5
def initial_tangent_position : ℝ := 12 -- Representing 12 o'clock as starting position
def rotation_angle_per_full_rotation : ℝ := 60 -- 60 degrees anti-clockwise per full rotation of disk
def final_tangent_position : ℝ := 10 -- Representing 10 o'clock as final position

theorem disk_tangent_position_after_full_rotation :
  (clock_face_radius = 30) →
  (disk_radius = 5) →
  (initial_tangent_position = 12) →
  (rotation_angle_per_full_rotation = 60) →
  (∃ t tangent_position, 
    tangent_position = initial_tangent_position - rotation_angle_per_full_rotation / 30 / 360 * 12) →
  final_tangent_position = 10 :=
sorry

end disk_tangent_position_after_full_rotation_l263_263268


namespace remainder_mod_500_l263_263775

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l263_263775


namespace P_is_orthocenter_of_triangle_l263_263587

noncomputable def triangle (A B C : Type*) := 
  { P : Type* // (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧ (P ∈ line A C ∨ P ∈ line B C) }

noncomputable def is_incenter {A B C K : Type*} (P : Type*) (h : P ∈ circumcircle (triangle A B C) K) := 
  ∀ {A1 C1 : Type*}, P ∈ angle_bisector A B C K

theorem P_is_orthocenter_of_triangle
  {A B C A1 C1 K P : Type*} 
  (hA1 : A1 ∈ line B C) 
  (hC1 : C1 ∈ line A B)
  (hK : intersects (line A A1) (line C C1) K)
  (hCircumcircleAA1B : P ∈ circumcircle (triangle A A1 B) B)
  (hCircumcircleCC1B : P ∈ circumcircle (triangle C C1 B) B)
  (hIncenter : is_incenter P (triangle A K C))
  : is_orthocenter P (triangle A B C) :=
sorry

end P_is_orthocenter_of_triangle_l263_263587


namespace problem1_part1_problem1_part2_l263_263885

section Problem1

variables {A B C a b c : ℝ}

-- Problem 1, Part 1: Prove b = 2a
theorem problem1_part1 (h1 : sin (2 * A + B) = 2 * sin A * (1 - cos C)) : 
  ∃ (a b : ℝ), b = 2 * a := sorry

-- Problem 1, Part 2: Range of values for the given expression
theorem problem1_part2 (h_a : 0 < a) 
  (h1 : sin (2 * A + B) = 2 * sin A * (1 - cos C)) 
  (h2 : b = 2 * a) 
  (hABC : ∀ A B C : ℝ, 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 
      a^2 + b^2 > c^2 ∧ 
      a^2 + c^2 > b^2 ∧ 
      b^2 + c^2 > a^2) : 
  2 ≤ (3 * sin A^2 + sin B^2) / (2 * sin A * sin C) + cos B ∧ 
  (3 * sin A^2 + sin B^2) / (2 * sin A * sin C) + cos B < (7 * sqrt 3) / 6 := sorry

end Problem1

end problem1_part1_problem1_part2_l263_263885


namespace no_two_tuples_satisfy_eq_l263_263597

theorem no_two_tuples_satisfy_eq :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ ((x+1) * (x+2) * ... * (x+2014) = (y+1) * (y+2) * ... * (y+4028)) := 
sorry

end no_two_tuples_satisfy_eq_l263_263597


namespace greatest_number_of_subparts_l263_263899

-- Definitions based on conditions from a)
def language_of_wolves : Type := string -- A word in the language of wolves can be represented by a string with 'F' and 'P'

def is_subpart (Y X : language_of_wolves) : Prop := 
  ∃ sub : list nat, 
    let chars := X.data in 
    Y.data = sub.map (λ i, chars.nth_le i (by sorry))

-- Proof statement based on c)
theorem greatest_number_of_subparts (n : nat) (X : language_of_wolves) (h_length : X.length = n) : 
  ∃ k, k = 2^n - 1 :=
begin
  -- The actual proof would go here, replaced by 'sorry' as instructed.
  sorry
end

end greatest_number_of_subparts_l263_263899


namespace trapezoid_AD_length_l263_263648

-- Definitions for the problem setup
variables {A B C D O P : Type}
variables (f : A → B → C → D → Prop)
variables (g : A → D → C → D → Prop)
variables (h : A → C → D → B → Prop)

-- The main theorem we want to prove
theorem trapezoid_AD_length
  (ABCD_trapezoid : f A B C D)
  (BC_CD_same : ∀ {x y}, (g B C x y → y = 43) ∧ (g B C x y → x = 43))
  (AD_perpendicular_BD : ∀ {x y}, h A D x y → ∃ (p : P), p = O)
  (O_intersection_AC_BD : g A C O B)
  (P_midpoint_BD : ∃ (p : P), p = P ∧ ∀ (x y : B ∗ D), y = x / 2)
  (OP_length : ∃ (len : ℝ), len = 11) :
  let m := 4 in let n := 190 in m + n = 194 := sorry

end trapezoid_AD_length_l263_263648


namespace non_congruent_triangles_with_perimeter_11_l263_263112

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263112


namespace weighted_average_is_correct_l263_263754

-- Define the conditions
def Aang_fish_counts : List ℕ := [5, 7, 9]
def Aang_hours : List ℕ := [3, 4, 2]

def Sokka_fish_counts : List ℕ := [8, 5, 6]
def Sokka_hours : List ℕ := [4, 2, 3]

def Toph_fish_counts : List ℕ := [10, 12, 8]
def Toph_hours : List ℕ := [2, 3, 4]

def Zuko_fish_counts : List ℕ := [6, 7, 10]
def Zuko_hours : List ℕ := [3, 3, 4]

-- Helper functions to sum the elements of lists
def sum_list (l : List ℕ) : ℕ := l.foldl (λ sum x => sum + x) 0

-- Total fish caught and total hours spent by the group
def total_fish : ℕ :=
  sum_list Aang_fish_counts + sum_list Sokka_fish_counts + sum_list Toph_fish_counts + sum_list Zuko_fish_counts

def total_hours : ℕ :=
  sum_list Aang_hours + sum_list Sokka_hours + sum_list Toph_hours + sum_list Zuko_hours

-- The weighted average of fish caught per hour
def weighted_average : ℚ := total_fish / total_hours

-- Proof problem statement
theorem weighted_average_is_correct : weighted_average ≈ 2.51 := 
by 
  sorry

end weighted_average_is_correct_l263_263754


namespace factor_1024_into_three_factors_l263_263181

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l263_263181


namespace position_at_4_seconds_distance_traveled_by_4_seconds_l263_263430

noncomputable def velocity (t : ℝ) : ℝ := t^2 - 4 * t + 3

-- The position at t = 4 seconds
theorem position_at_4_seconds : ∫ x in 0..4, velocity x = 4 / 3 :=
sorry

-- The distance traveled by t = 4 seconds
theorem distance_traveled_by_4_seconds :
  |∫ x in 0..1, velocity x | + |∫ x in 1..3, velocity x | + |∫ x in 3..4, velocity x | = 4 :=
sorry

end position_at_4_seconds_distance_traveled_by_4_seconds_l263_263430


namespace integer_solutions_count_count_integer_solutions_l263_263848

theorem integer_solutions_count (x : ℤ) :
  (x ∈ (set_of (λ x : ℤ, |x - 3| ≤ 4))) ↔ x ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7} :=
by sorry

theorem count_integer_solutions :
  (finset.card (finset.filter (λ x, |x - 3| ≤ 4) (finset.range 10))) = 9 :=
by sorry

end integer_solutions_count_count_integer_solutions_l263_263848


namespace g_sum_eq_neg_one_l263_263871

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Main theorem to prove g(1) + g(-1) = -1 given the conditions
theorem g_sum_eq_neg_one
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end g_sum_eq_neg_one_l263_263871


namespace zeta1_zeta2_zeta3_pow5_sum_l263_263323

noncomputable def zeta1 := sorry
noncomputable def zeta2 := sorry
noncomputable def zeta3 := sorry

def condition1 : zeta1 + zeta2 + zeta3 = 2 := sorry
def condition2 : zeta1^2 + zeta2^2 + zeta3^2 = 5 := sorry
def condition3 : zeta1^3 + zeta2^3 + zeta3^3 = 14 := sorry

theorem zeta1_zeta2_zeta3_pow5_sum : zeta1^5 + zeta2^5 + zeta3^5 = 44 :=
by {
  -- using the conditions
  have cond1 := condition1,
  have cond2 := condition2,
  have cond3 := condition3,
  sorry
}

end zeta1_zeta2_zeta3_pow5_sum_l263_263323


namespace maximal_subsets_le_twice_n_l263_263057

-- Define necessary terms
def l (U : Finset (Vector ℝ 2)) : ℝ := (U.sum id).norm

-- Define maximal subset condition
def is_maximal_subset (V A : Finset (Vector ℝ 2)) : Prop :=
  ∀ B, B ⊆ A → B ≠ ∅ → l B ≥ l A

-- Total number of maximal subsets within finite vectors in 2D plane
theorem maximal_subsets_le_twice_n (V : Finset (Vector ℝ 2)) (hV : ∀ v ∈ V, v ≠ 0) :
  (Finset.filter (is_maximal_subset V) V.powerset).card ≤ 2 * V.card :=
sorry

end maximal_subsets_le_twice_n_l263_263057


namespace fair_contest_perfect_square_l263_263724

theorem fair_contest_perfect_square (n : ℕ) (h: 2 * n > 0) :
  ∃ k : ℕ, 
    let f : ℕ → ℕ := λ n, ((Nat.doubleFactorial (2 * n - 1)) ^ 2)
    in f n = k * k :=
sorry

end fair_contest_perfect_square_l263_263724


namespace determine_n_l263_263059

noncomputable def d (n : ℕ) : ℕ := 
  if n = 0 then 0 else (Finset.filter (λ k, n % k = 0) (Finset.range (n + 1))).card 

theorem determine_n (n : ℕ) (h : n ≥ 3) : 
  (d (n-1) + d n + d (n+1) ≤ 8) ↔ n = 3 ∨ n = 4 ∨ n = 6 := 
by sorry

end determine_n_l263_263059


namespace trigonometric_identity_l263_263682

theorem trigonometric_identity :
  sin (70 * real.pi / 180) * cos (20 * real.pi / 180) + cos (70 * real.pi / 180) * sin (20 * real.pi / 180) = 1 :=
by
  sorry

end trigonometric_identity_l263_263682


namespace minimal_degree_polynomial_l263_263960

theorem minimal_degree_polynomial (x : ℂ) (h1 : x^9 = 1) (h2 : x^3 ≠ 1) :
  ∃ p : polynomial ℂ, p.degree = 5 ∧ p.eval x = (1 + x)⁻¹ ∧ p = X^5 - X^4 + X^3 := 
by 
  sorry

end minimal_degree_polynomial_l263_263960


namespace length_A_l263_263214

def Point : Type := ℝ × ℝ

def A : Point := (0, 9)
def B : Point := (0, 12)
def C : Point := (2, 8)

def is_on_line (p : Point) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

-- Definitions to specify A' and B' on the line y = x
def A'_line_x : ℝ := 6
def A' : Point := (A'_line_x, A'_line_x)

def B'_line_x : ℝ := 4
def B' : Point := (B'_line_x, B'_line_x)

-- The length of a line segment given two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem length_A'B' : distance A' B' = 2 * real.sqrt 2 := by
  sorry

end length_A_l263_263214


namespace problem1_problem2_l263_263837

noncomputable def f (x : ℝ) : ℝ := log ((2 * x - 3) * (x - 1/2))

noncomputable def g (x a : ℝ) : ℝ := sqrt (-x^2 + 4 * a * x - 3 * a^2)

def domain_f : Set ℝ := {x : ℝ | (2 * x - 3) * (x - 1/2) > 0}
def domain_g (a : ℝ) : Set ℝ := {x : ℝ | -x^2 + 4 * a * x - 3 * a^2 >= 0}

theorem problem1 :
  domain_f ∩ ({x : ℝ | 1 ≤ x ∧ x ≤ 3}) = (Set.Ioo (3 / 2) 3 ∪ {3}) :=
sorry

theorem problem2 (a : ℝ) (h : domain_f ∩ ({x : ℝ | a ≤ x ∧ x ≤ 3 * a}) = {x : ℝ | a ≤ x ∧ x ≤ 3 * a}) :
  a ∈ (Set.Iio (1 / 6) ∪ Set.Ioi (3 / 2)) :=
sorry

end problem1_problem2_l263_263837


namespace cards_given_l263_263206

/-- 
   Jason had 13 Pokemon cards initially.
   Jason has 4 Pokemon cards left.
   Prove that the number of Pokemon cards Jason gave to his friends is 9.
-/
theorem cards_given (initial_cards remaining_cards cards_given : ℕ) 
  (h1 : initial_cards = 13) 
  (h2 : remaining_cards = 4) 
  (h3 : cards_given = initial_cards - remaining_cards) : 
  cards_given = 9 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cards_given_l263_263206


namespace find_b_l263_263007

noncomputable def b_value (p1 p2 : ℝ × ℝ) := (let v := (p2.1 - p1.1, p2.2 - p1.2) in
  (v.1 * (-2) / v.2))

theorem find_b :
  let p1 := (3 : ℝ, -1 : ℝ)
  let p2 := (-1 : ℝ, 4 : ℝ) in
  let b := b_value p1 p2 in
  b = 8 / 5 :=
by
  sorry

end find_b_l263_263007


namespace find_f_f_neg2_l263_263835

def f (x : ℝ) : ℝ := if x < 0 then 3 ^ x else 1 - real.sqrt x

theorem find_f_f_neg2 : f (f (-2)) = 2 / 3 := by
  sorry

end find_f_f_neg2_l263_263835


namespace triangle_count_with_perimeter_11_l263_263100

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263100


namespace largest_S_n_value_l263_263431

noncomputable def a_n (n : ℕ) : ℝ := 20 - 4 * n

noncomputable def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n (i + 1)

theorem largest_S_n_value :
  ∃ n : ℕ, (n = 4 ∨ n = 5) ∧ S_n n = max (S_n 4) (S_n 5) :=
begin
  sorry
end

end largest_S_n_value_l263_263431


namespace lattice_points_distance_5_count_l263_263195

def is_lattice_point (x y z : ℤ) : Prop :=
  x*x + y*y + z*z = 25

theorem lattice_points_distance_5_count :
  (∑ x in Finset.Icc (-5) 5, ∑ y in Finset.Icc (-5) 5, ∑ z in Finset.Icc (-5) 5,
    if is_lattice_point x y z then 1 else 0) = 12 :=
by
  sorry

end lattice_points_distance_5_count_l263_263195


namespace fraction_of_milk_in_first_cup_l263_263200

theorem fraction_of_milk_in_first_cup
  (V : ℝ)  -- Volume of each cup
  (h : 0 < V)  -- Volume must be positive
  (x : ℝ)  -- Fraction of milk in the first cup
  (h_ratio : (1 - x) * V + (1 / 5) * V = (3 / 7) * (x * V + (4 / 5) * V)) : 
  x = 3 / 5 :=
begin
  sorry
end

end fraction_of_milk_in_first_cup_l263_263200


namespace vector_dot_product_result_l263_263846

variable {α : Type*} [Field α]

structure Vector2 (α : Type*) :=
(x : α)
(y : α)

def vector_add (a b : Vector2 α) : Vector2 α :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_sub (a b : Vector2 α) : Vector2 α :=
  ⟨a.x - b.x, a.y - b.y⟩

def dot_product (a b : Vector2 α) : α :=
  a.x * b.x + a.y * b.y

variable (a b : Vector2 ℝ)

theorem vector_dot_product_result
  (h1 : vector_add a b = ⟨1, -3⟩)
  (h2 : vector_sub a b = ⟨3, 7⟩) :
  dot_product a b = -12 :=
by
  sorry

end vector_dot_product_result_l263_263846


namespace hyperbola_asymptotes_l263_263618

-- Define the condition for the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 9) = 1

-- Define the equations of the asymptotes
def is_asymptote (x y : ℝ) : Prop :=
  (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), hyperbola_eq x y → is_asymptote x y :=
begin
  sorry
end

end hyperbola_asymptotes_l263_263618


namespace max_value_even_function_1_2_l263_263485

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Given conditions
variables (f : ℝ → ℝ)
variable (h1 : even_function f)
variable (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → f x ≤ -2)

-- Prove the maximum value on [1, 2] is -2
theorem max_value_even_function_1_2 : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ -2) :=
sorry

end max_value_even_function_1_2_l263_263485


namespace athlete_performance_l263_263687

/-- Scores of the athlete in the training session. -/
def scores : List ℕ := [7, 5, 8, 9, 6, 6, 7, 7, 8, 7]

/-- Definition of the mode of the scores. -/
def mode (l : List ℕ) : ℕ := l.mode

/-- Definition of the median of the scores. -/
def median (l : List ℕ) : ℕ := l.median

/-- Definition of the mean of the scores. -/
def mean (l : List ℕ) : ℕ := l.sum / l.length

/-- Definition of the variance of the scores. -/
def variance (l : List ℕ) : ℚ :=
  let m := l.mean
  l.foldl (λ (acc : ℚ) (x : ℕ), acc + (x - m) ^ 2) 0 / l.length

/-- Theorem stating the properties of the scores of the athlete. -/
theorem athlete_performance :
  mode scores = 7 ∧
  median scores = 7 ∧
  mean scores = 7 ∧
  variance scores = 6 / 5 :=
by
  sorry

end athlete_performance_l263_263687


namespace probability_quadratic_real_roots_l263_263930

noncomputable def probability_real_roots (p : ℝ) : ℝ :=
  if (p ≥ 2) ∧ (p ≤ 5) then 
    1 
  else 
    0

theorem probability_quadratic_real_roots :
  (∫ (p : ℝ) in (Ιoo 0 5), probability_real_roots p) / (∫ (p : ℝ) in (Ιoo 0 5), 1) = 0.6 :=
by sorry

end probability_quadratic_real_roots_l263_263930


namespace angle_DBE_measure_l263_263691

noncomputable def circle_center_radius (O : Point) (r : ℝ) (ω : Circle) : Prop :=
  ω.center = O ∧ ω.radius = r

noncomputable def chord_length (B C : Point) (r : ℝ) (ω : Circle) : Prop :=
  ω.isChord B C ∧ dist B C = r

noncomputable def tangents_meet (A B C : Point) (ω : Circle) : Prop :=
  ω.isTangent B ∧ ω.isTangent C ∧ A = ω.tangentIntersection B C

noncomputable def ray_meets_past (A O D : Point) (ω : Circle) : Prop :=
  ω.isRay AO ∧ ω.rayMeetsPast O D

noncomputable def ray_meets_circle (A O E : Point) (AB : ℝ) : Prop :=
  circle_center_radius A AB (Circle.mkRadius A AB) ∧ (rayMeetingPoint O A E (Circle.mkRadius A AB))

theorem angle_DBE_measure (O B C A D E : Point) (r : ℝ) (ω : Circle) :
  circle_center_radius O r ω →
  chord_length B C r ω →
  tangents_meet A B C ω →
  ray_meets_past A O D ω →
  ray_meets_circle A O E (dist A B) →
  ∠DBE = 135 :=
by 
  sorry

end angle_DBE_measure_l263_263691


namespace number_of_arrangements_l263_263501

open Nat

-- Define the set of people as a finite type with 5 elements.
inductive Person : Type
| youngest : Person
| eldest : Person
| p3 : Person
| p4 : Person
| p5 : Person

-- Define a function to count valid arrangements.
def countValidArrangements :
    ∀ (first_pos last_pos : Person), 
    (first_pos ≠ Person.youngest → last_pos ≠ Person.eldest → Fin 120) 
| first_pos, last_pos, h1, h2 => 
    let remaining := [Person.youngest, Person.eldest, Person.p3, Person.p4, Person.p5].erase first_pos |>.erase last_pos
    (factorial 3) * 4 * 3

-- Theorem statement to prove the number of valid arrangements.
theorem number_of_arrangements : 
  countValidArrangements Person.youngest Person.p5 sorry sorry = 72 :=
by 
  sorry

end number_of_arrangements_l263_263501


namespace integral_value_eq_ln3_add8_l263_263482

theorem integral_value_eq_ln3_add8 (a : ℝ) (h : ∫ x in 1..a, (2 * x + 1 / x) = log 3 + 8) : a = 3 :=
sorry

end integral_value_eq_ln3_add8_l263_263482


namespace necessary_but_not_sufficient_l263_263744

theorem necessary_but_not_sufficient (x : ℝ) : 
  (x > 3) → ({x : ℝ | x < 2 ∨ x > 3}.nonempty) :=
begin
  sorry
end

end necessary_but_not_sufficient_l263_263744


namespace count_non_congruent_triangles_with_perimeter_11_l263_263118

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263118


namespace complex_power_sum_2013_l263_263254

noncomputable def complexPowerSum : ℂ :=
  let i := complex.I
  finset.sum (finset.range 2014) (λ n, i ^ n)

theorem complex_power_sum_2013 : complexPowerSum = 1 + complex.I :=
  sorry

end complex_power_sum_2013_l263_263254


namespace hexagon_area_ratio_l263_263510

theorem hexagon_area_ratio {s : ℝ} 
  (h_regular : regular_hexagon ABCDEF)
  (W_on_BC : W ∈ BC)
  (X_on_CD : X ∈ CD)
  (Y_on_EF : Y ∈ EF)
  (Z_on_FA : Z ∈ FA)
  (parallel_lines : parallel AB ZW ∧ parallel ZW YX ∧ parallel YX ED)
  (spacing : ∀ line : line, between s := (perp_height / 4)
  : ratio_area :=
  \left(1 - \frac{3\sqrt{3}}{8}\right)^2 :=
by
  sorry

end hexagon_area_ratio_l263_263510


namespace equal_number_of_acquaintances_maximum_number_of_subsets_l263_263320

open Set

variable {Person : Type}
variable (knows : Person → Person → Prop)

-- Conditions
axiom no_one_knows_all (S: Set Person) : ∀ x ∈ S, ∃ y ∈ S, ¬ knows x y
axiom three_at_least_two_not_knowing_each_other (S: Set Person) : ∀ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z → (¬ knows x y ∨ ¬ knows y z ∨ ¬ knows x z)
axiom one_person_knows_both (S: Set Person) : ∀ x y ∈ S, ¬ knows x y → ∃ z ∈ S, knows z x ∧ knows z y ∧ z ≠ x ∧ z ≠ y

-- Assumptions
axiom symmetry (x y : Person) : knows x y → knows y x
axiom reflexivity (x : Person) : knows x x

-- Proof problems
theorem equal_number_of_acquaintances (S : Set Person) (hS1 : ∀ x ∈ S, ∃ y ∈ S, ¬ knows x y)
    (hS2 : ∀ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z → (¬ knows x y ∨ ¬ knows y z ∨ ¬ knows x z))
    (hS3 : ∀ x y ∈ S, ¬ knows x y → ∃ z ∈ S, knows z x ∧ knows z y ∧ z ≠ x ∧ z ≠ y)
    (hSymm : ∀ x y : Person, knows x y → knows y x)
    (hRefl : ∀ x : Person, knows x x) :
  ∃ n : ℕ, ∀ x ∈ S, (count (λ y, knows x y) S) = n := sorry

theorem maximum_number_of_subsets :
  (∃ F : Finset (Set Person), (∀ S ∈ F, no_one_knows_all S ∧ three_at_least_two_not_knowing_each_other S ∧ one_person_knows_both S) ∧ F.card = 398) := sorry

end equal_number_of_acquaintances_maximum_number_of_subsets_l263_263320


namespace prod_n_1_to_30_eq_45927_l263_263001

theorem prod_n_1_to_30_eq_45927 : (∏ n in finset.range 30, (n + 5) / (n + 1)) = 45927 := by
  sorry

end prod_n_1_to_30_eq_45927_l263_263001


namespace candle_heights_equality_l263_263652

theorem candle_heights_equality (h₀ : ∀ x : ℝ, 
    (1 - x / 5) = 3 * (1 - x / 4)) : 
    ∃ x : ℝ, x = 40 / 11 :=
by
  have h₁ : 1 - (40 / 11) / 5 = 3 * (1 - (40 / 11) / 4), from h₀ (40 / 11),
  exact ⟨40 / 11, rfl⟩

end candle_heights_equality_l263_263652


namespace range_of_a_minimize_S_l263_263743

open Real

-- Problem 1: Prove the range of a 
theorem range_of_a (a : ℝ) : (∃ x ≠ 0, x^3 - 3*x^2 + (2 - a)*x = 0) ↔ a > -1 / 4 := sorry

-- Problem 2: Prove the minimizing value of a for the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 
  let α := sorry -- α is the root depending on a (to be determined from the context)
  let β := sorry -- β is the root depending on a (to be determined from the context)
  (1/4 * α^4 - α^3 + (1/2) * (2-a) * α^2) + (1/4 * β^4 - β^3 + (1/2) * (2-a) * β^2)

theorem minimize_S (a : ℝ) : a = 38 - 27 * sqrt 2 → S a = S (38 - 27 * sqrt 2) := sorry

end range_of_a_minimize_S_l263_263743


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263783

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263783


namespace milk_leftover_after_milkshakes_l263_263367

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l263_263367


namespace hannah_dogs_food_total_l263_263472

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l263_263472


namespace coeff_of_inv_x_in_expansion_l263_263190

theorem coeff_of_inv_x_in_expansion :
  let f := (x - 1/x)^5 in
  (∃ c : ℚ, f = c * (1/x) + ∑ i in finset.filter (λ j, j ≠ (1:ℚ/x)), i ∈ f) →
  ∑ i in finset.filter (λ j, j = (1:ℚ/x)), i ∈ f = -10 :=
sorry

end coeff_of_inv_x_in_expansion_l263_263190


namespace intersection_points_count_l263_263745

theorem intersection_points_count :
  ∃ P Q : Set (ℝ × ℝ), (∀ p ∈ P, p.1 ^ 2 + p.2 ^ 2 = 4) ∧ (∀ q ∈ Q, q.1 ^ 2 + q.2 ^ 2 / 9 = 1) ∧
    (P ∩ Q).card = 4 :=
by
  -- Definitions of the curves
  let C1 : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 4}
  let C2 : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 / 9 = 1}

  -- Count intersection points
  have h1 : (C1 ∩ C2).card = 4 := sorry

  exact ⟨C1, C2, by simp[C1], by simp[C2], h1⟩

end intersection_points_count_l263_263745


namespace max_siskins_on_poles_l263_263715

def max_siskins (n: ℕ) (occupied: ℕ → Prop) : ℕ :=
  if n ≤ 0 then 0
  else
    let k := n - 1
    if occupied k then k else max_siskins k occupied

theorem max_siskins_on_poles : 
  ∀ (n : ℕ) (occupied : ℕ → bool), (∀ i, (occupied i = true → i ≤ 24)) → 
  max_siskins 25 (λ i, occupied i) = 24 :=
begin
  intros n occupied h,
  sorry
end

end max_siskins_on_poles_l263_263715


namespace factorization_count_l263_263182

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l263_263182


namespace expression_evaluates_at_1_l263_263896

variable (x : ℚ)

def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

def substituted_expr (x : ℚ) : ℚ :=
  (original_expr (original_expr x) + 2) / (original_expr (original_expr x) - 3)

theorem expression_evaluates_at_1 :
  substituted_expr 1 = -1 / 9 :=
by
  sorry

end expression_evaluates_at_1_l263_263896


namespace train_passes_man_in_approximately_24_seconds_l263_263710

noncomputable def train_length : ℝ := 880 -- length of the train in meters
noncomputable def train_speed_kmph : ℝ := 120 -- speed of the train in km/h
noncomputable def man_speed_kmph : ℝ := 12 -- speed of the man in km/h

noncomputable def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph
noncomputable def relative_speed : ℝ := train_speed_mps + man_speed_mps

noncomputable def time_to_pass : ℝ := train_length / relative_speed

theorem train_passes_man_in_approximately_24_seconds :
  abs (time_to_pass - 24) < 1 :=
sorry

end train_passes_man_in_approximately_24_seconds_l263_263710


namespace circumcircles_intersect_on_BC_l263_263215

theorem circumcircles_intersect_on_BC (A B C M N O R : Point) 
  (h1 : triangle A B C)
  (h2 : acute_triangle A B C)
  (h3 : AB ≠ AC)
  (h4 : circle (B, C) ∩ line (A, B) = M)
  (h5 : circle (B, C) ∩ line (A, C) = N)
  (h6 : midpoint O B C)
  (h7 : angle_bisector_interior (A, B, C) (A, R))
  (h8 : angle_bisector_exterior (M, O, N) (O, R)) :
  ∃ E : Point, on_circumcircle (triangle B M R) E ∧ on_circumcircle (triangle C N R) E ∧ on_line (B, C) E :=
sorry

end circumcircles_intersect_on_BC_l263_263215


namespace solution_set_of_inequality_l263_263633

theorem solution_set_of_inequality (x : ℝ) :  (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
by
  sorry

end solution_set_of_inequality_l263_263633


namespace number_of_right_triangles_l263_263009

-- Definitions
variables (A B C D P Q : Type) [IsRectangle A B C D] [IsCircleCenteredAt A P Q]
variables (AP AQ : ℝ) [EqRadius AP AQ]

-- The theorem stating the number of right triangles formed is 4
theorem number_of_right_triangles : count_right_triangles ({A, P, B, C, Q, D} : set Type) = 4 :=
sorry

end number_of_right_triangles_l263_263009


namespace fish_population_estimation_l263_263334

-- Define the conditions
variables {totalCatches : ℕ} (c1 c2 c3 m1 m2 m3 markedFish : ℕ)
hypothesis catches_sum : c1 + c2 + c3 = totalCatches
hypothesis marked_sum : m1 + m2 + m3 = markedFish
hypothesis marked : markedFish = 19

-- Define the fishing data
def totalFish : ℕ := 50

-- Formula to find total fish in the lake
def lakeFish (totalCatches markedFish : ℕ) : ℕ := (totalFish * totalCatches) / markedFish

-- Theorem to be proved
theorem fish_population_estimation
(h1 : c1 = 67) (h2 : c2 = 94) (h3 : c3 = 43)
(h4 : m1 = 6) (h5 : m2 = 10) (h6 : m3 = 3)
: lakeFish totalCatches markedFish = 537 :=
by
  -- conditions provided
  have catches_sum : 67 + 94 + 43 = 204 := by norm_num
  have marked_sum : 6 + 10 + 3 = 19 := by norm_num
  unfold lakeFish
  norm_num
  sorry

end fish_population_estimation_l263_263334


namespace jessica_milk_problem_l263_263533

theorem jessica_milk_problem (gallons_owned : ℝ) (gallons_given : ℝ) : gallons_owned = 5 → gallons_given = 16 / 3 → gallons_owned - gallons_given = -(1 / 3) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- sorry

end jessica_milk_problem_l263_263533


namespace pos_sum_of_powers_l263_263093

theorem pos_sum_of_powers (a b c : ℝ) (n : ℕ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) : 
  a^n + b^n + c^n > 0 :=
sorry

end pos_sum_of_powers_l263_263093


namespace square_field_area_l263_263612

theorem square_field_area (speed time perimeter : ℕ) (h1 : speed = 20) (h2 : time = 4) (h3 : perimeter = speed * time) :
  ∃ s : ℕ, perimeter = 4 * s ∧ s * s = 400 :=
by
  -- All conditions and definitions are stated, proof is skipped using sorry
  sorry

end square_field_area_l263_263612


namespace astroid_arc_length_l263_263765

theorem astroid_arc_length (a : ℝ) (h_a : a > 0) :
  ∃ l : ℝ, (l = 6 * a) ∧ 
  ((a = 1 → l = 6) ∧ (a = 2/3 → l = 4)) := 
by
  sorry

end astroid_arc_length_l263_263765


namespace new_average_weight_l263_263613

theorem new_average_weight 
  (average_weight_19 : ℕ → ℝ)
  (weight_new_student : ℕ → ℝ)
  (new_student_count : ℕ)
  (old_student_count : ℕ)
  (h1 : average_weight_19 old_student_count = 15.0)
  (h2 : weight_new_student new_student_count = 11.0)
  : (average_weight_19 (old_student_count + new_student_count) = 14.8) :=
by
  sorry

end new_average_weight_l263_263613


namespace loss_percentage_is_30_l263_263973

theorem loss_percentage_is_30
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1900)
  (h2 : selling_price = 1330) :
  (cost_price - selling_price) / cost_price * 100 = 30 :=
by
  -- This is a placeholder for the actual proof
  sorry

end loss_percentage_is_30_l263_263973


namespace starting_lineup_count_l263_263239

theorem starting_lineup_count (total_players : ℕ) (goalie_choices : ℕ) (regular_players : ℕ) 
  (h1 : total_players = 18) (h2 : goalie_choices = 1) (h3 : regular_players = 10) : 
  let remaining_players := total_players - goalie_choices in
  let combination := Nat.factorial remaining_players / (Nat.factorial regular_players * Nat.factorial (remaining_players - regular_players)) in
  goalie_choices * combination = 349,864 :=
by
  sorry

end starting_lineup_count_l263_263239


namespace simplify_fraction_l263_263948

theorem simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by {
  -- Proof skipped
  sorry
}

end simplify_fraction_l263_263948


namespace city_of_Geometry_schools_count_l263_263026

theorem city_of_Geometry_schools_count:
  (∃ (n : ℕ), 
    (∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (median_score := (∀ x y, median_score x y := x ∪ y) ) 
    (∃ m : ℕ, Andrea_score = m ∧ 2 * n < m ∧ m < 43)) ∧
    (points := 4 * n ∧ n = 24) :=
by
  sorry

end city_of_Geometry_schools_count_l263_263026


namespace solve_system_l263_263843

noncomputable def solution (x y : ℝ) : Prop :=
  x^2 + y^2 = 13 ∧ xy = 6 ∧ x = 3 ∧ y = 2

theorem solve_system : ∃ x y : ℝ, solution x y :=
by {
  use [3, 2],
  simp [solution],
  split,
  { apply congr_arg,
    calc
      3^2 + 2^2 = 9 + 4 : by norm_num
      ... = 13 : by norm_num },
  split,
  { calc
      3 * 2 = 6 : by norm_num },
  refl
}

end solve_system_l263_263843


namespace marys_number_l263_263939

theorem marys_number :
  ∃ x : ℕ, 9 < x ∧ x < 100 ∧ (let y := 3 * x + 11 in ∃ a b : ℕ, y = 10 * a + b ∧ 71 ≤ 10 * b + a ≤ 75) ∧ x = 12 :=
begin
  sorry
end

end marys_number_l263_263939


namespace area_BCD_l263_263224

-- Defining the areas of triangles ABC, ACD, and ADB as x, y, and z, respectively.
variables (x y z : ℝ)

-- Conditions: A, B, C, D form a tetrahedron with mutually perpendicular edges.
-- The area of triangle BCD in terms of x, y, and z needs to be proven as sqrt(x^2 + y^2 + z^2).
theorem area_BCD (x y z : ℝ) : 
  let K := by exact sqrt(x^2 + y^2 + z^2)
  in K = sqrt(x^2 + y^2 + z^2) :=
by
  sorry

end area_BCD_l263_263224


namespace part_I_part_II_sum_l263_263551

-- Definition of arithmetic sequence and related conditions
def arithmetic_sequence (a_1 d n : ℕ) : ℕ := a_1 + (n - 1) * d
def S_n (a_1 d n : ℕ) : ℕ := n * a_1 + (n * (n - 1) / 2) * d

-- Define b_n as described above
def b_n (n : ℕ) : ℕ := Int.floor (Real.log10 (n : ℝ))

-- Prove part (I): specific term values of sequence b
theorem part_I (a_1 d b : ℕ) (a1_eq : a_1 = 1) (S7_eq : S_n a_1 d 7 = 28) :
  let a_n := arithmetic_sequence a_1 d in
  let b_n := λ n, Real.floor (Real.log10 (a_n n)) in
    b_n 1 = 0 ∧ b_n 11 = 1 ∧ b_n 101 = 2 :=
by sorry

-- Prove part (II): sum of the first 1000 terms of b_n
theorem part_II_sum (a_1 d : ℕ) (a1_eq : a_1 = 1) (S7_eq : S_n a_1 d 7 = 28) :
  let a_n := arithmetic_sequence a_1 d in
  let b_n := λ n, Real.floor (Real.log10 (n : ℝ)) in
    (Finset.sum (Finset.range 1000) b_n) = 1893 :=
by sorry

end part_I_part_II_sum_l263_263551


namespace earthquake_damage_in_usd_l263_263347

theorem earthquake_damage_in_usd :
  ∀ (damage_in_euros : ℝ) (exchange_rate_euro_to_usd : ℝ),
    damage_in_euros = 50000000 →
    exchange_rate_euro_to_usd = (3 / 2) →
    damage_in_euros * exchange_rate_euro_to_usd = 75000000 :=
by
  intros damage_in_euros exchange_rate_euro_to_usd
  assume h_damage : damage_in_euros = 50000000
  assume h_rate : exchange_rate_euro_to_usd = (3 / 2)
  rw [h_damage, h_rate]
  norm_num
  sorry

end earthquake_damage_in_usd_l263_263347


namespace base_8_representation_l263_263666

theorem base_8_representation :
  ∃ A B : ℕ, (A ≠ B ∧ 7^3 ≤ 777 ∧ 777 < 7^4 ∧ 
  (777 = A * 8^3 + B * 8^2 + B * 8^1 + A * 8^0) ∧ 
  A < 8 ∧ B < 8) :=
begin
  sorry

end base_8_representation_l263_263666


namespace fraction_of_satisfactory_grades_l263_263882

theorem fraction_of_satisfactory_grades (A B C D E F G : ℕ) (ha : A = 6) (hb : B = 3) (hc : C = 4) 
    (hd : D = 2) (he : E = 1) (hf : F = 7) (hg : G = 2) :
    (A + B + C + D + E) / (A + B + C + D + E + F + G) = 16 / 25 :=
by
  sorry

end fraction_of_satisfactory_grades_l263_263882


namespace sampling_method_is_systematic_l263_263642

-- Definitions based on the conditions
def num_classes := 12
def students_per_class := 50
def selected_student_number := 40

-- Main theorem
theorem sampling_method_is_systematic :
  ∀ (classes: fin num_classes → fin students_per_class), 
    (∀ (i: fin num_classes), classes i = ⟨selected_student_number, sorry⟩) →
      "Systematic sampling" = "Systematic sampling" :=
by
  intros classes classes_definition
  sorry

end sampling_method_is_systematic_l263_263642


namespace problem_statement_l263_263823

def f (x : ℝ) : ℝ := x^3 + x^2 + 2

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement : odd_function f → f (-2) = -14 := by
  intro h
  sorry

end problem_statement_l263_263823


namespace obtuse_triangle_iff_tan_product_lt_one_l263_263245

theorem obtuse_triangle_iff_tan_product_lt_one 
  (α β γ : ℕ) 
  (h_sum : α + β + γ = 180) 
  (h_obtuse : γ > 90) : 
  (α + β < 90 ↔ tan α * tan β < 1) := 
by
  sorry

end obtuse_triangle_iff_tan_product_lt_one_l263_263245


namespace Sn_formula_Tn_formula_l263_263064

def a (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def Sn (n : ℕ) : ℕ :=
  if n = 1 then 5 else (n * n + 3 * n + 1)

def bn (n : ℕ) : ℚ :=
  1 / (Sn n + 1 : ℚ)

def Tn (n : ℕ) : ℚ :=
  (nat.sum (finset.range n) (λ i, bn (i + 1)))

theorem Sn_formula (n : ℕ) : Sn n = n^2 + 3 * n + 1 :=
by
  sorry

theorem Tn_formula (n : ℕ) : Tn n = (n : ℚ) / (2 * n + 4) :=
by
  sorry

end Sn_formula_Tn_formula_l263_263064


namespace exists_unique_alpha_l263_263622

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  -x + real.sqrt ((x + a) * (x + b))

theorem exists_unique_alpha
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_diff : a ≠ b) :
  ∀ (s : ℝ), (0 < s ∧ s < 1) →
  ∃! (α : ℝ), (0 < α) ∧ (f α a b = real.cbrt ((a^s + b^s) / 2)) :=
begin
  sorry
end

end exists_unique_alpha_l263_263622


namespace QPO_area_eq_l263_263184

variables {A B C D P Q M N O : Point}
variables {k : ℝ} (ABCD : Parallelogram A B C D)
variables (DP : Line) (CQ : Line)

-- DP bisects BC at N and meets AB (extended) at P
-- CQ bisects AD at M and meets AB (extended) at Q
-- DP and CQ meet at O
-- Area of parallelogram ABCD is k
-- Need to prove area of triangle QPO is 9k/8

def bisects (l : Line) (p1 p2 p3 : Point) : Prop :=
  midpoint p1 p3 p2

def extended_at (l : Line) (p1 p2 : Point) : Prop :=
  ∃ x, point_on_line x l ∧ x = p1 ∨ x = p2

def area_parallelogram (p1 p2 p3 p4 : Point) (area : ℝ) : Prop :=
  parallelogram p1 p2 p3 p4 ∧ area_shape (quadrilateral p1 p2 p3 p4) = area

def area_triangle (p1 p2 p3 : Point) (area : ℝ) : Prop :=
  triangle p1 p2 p3 ∧ area_shape (triangle p1 p2 p3) = area

theorem QPO_area_eq (h1 : bisects DP B C N)
                    (h2 : extended_at DP P A B)
                    (h3 : bisects CQ A D M)
                    (h4 : extended_at CQ Q A B)
                    (h5 : point_on_line O DP)
                    (h6 : point_on_line O CQ)
                    (h7 : area_parallelogram A B C D k) :
                    ∃ area, area_triangle Q P O area ∧ area = (9 * k) / 8 :=
sorry

end QPO_area_eq_l263_263184


namespace remainder_of_exponentiation_is_correct_l263_263792

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l263_263792


namespace necessary_not_sufficient_condition_t_for_b_l263_263421

variable (x y : ℝ)

def condition_t : Prop := x ≤ 12 ∨ y ≤ 16
def condition_b : Prop := x + y ≤ 28 ∨ x * y ≤ 192

theorem necessary_not_sufficient_condition_t_for_b (h : condition_b x y) : condition_t x y ∧ ¬ (condition_t x y → condition_b x y) := by
  sorry

end necessary_not_sufficient_condition_t_for_b_l263_263421


namespace circle_equation_l263_263992

def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem circle_equation : ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ∧ (y = parabola x) ∧ (x = -1 ∨ x = 3 ∨ (x = 0 ∧ y = -3)) :=
by { sorry }

end circle_equation_l263_263992


namespace number_of_true_propositions_count_l263_263088

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

def converse (a b c : ℝ) (h : b^2 = a * c) : Prop :=
  ∃ r, r ≠ 0 ∧ b = a * r ∧ c = b * r

def inverse (a b c : ℝ) (h : ¬ (∃ r, r ≠ 0 ∧ b = a * r ∧ c = b * r)) : Prop :=
  ¬ (b^2 = a * c)

def contrapositive (a b c : ℝ) (h : b^2 ≠ a * c) : Prop :=
  ∃ r, r ≠ 0 ∧ b = a * r ∧ c = b * r

theorem number_of_true_propositions_count 
  (a b c : ℝ)
  (h_geom : geometric_sequence a b c)
  (h_conv : ¬ (converse a b c h_geom))
  (h_inv : ¬ (inverse a b c h_conv))
  (h_contr : contrapositive a b c (λ h, by contradiction)) :
  1 :=
sorry

end number_of_true_propositions_count_l263_263088


namespace coefficient_x8_in_expansion_l263_263014

theorem coefficient_x8_in_expansion :
  let f := (1:ℚ) - 3*x + 2*x^2 in
  ((f ^ 5).coeff 8 = -2520) :=
by
  sorry

end coefficient_x8_in_expansion_l263_263014


namespace fg_product_l263_263621

variable (x : ℝ)

def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
def g (x : ℝ) : ℝ := x + 3

theorem fg_product (hx : x ≠ -3) : f x * g x = x - 3 := 
  by
  sorry

end fg_product_l263_263621


namespace recipe_flour_amount_l263_263575

theorem recipe_flour_amount
  (cups_of_sugar : ℕ) (cups_of_salt : ℕ) (cups_of_flour_added : ℕ)
  (additional_cups_of_flour : ℕ)
  (h1 : cups_of_sugar = 2)
  (h2 : cups_of_salt = 80)
  (h3 : cups_of_flour_added = 7)
  (h4 : additional_cups_of_flour = cups_of_sugar + 1) :
  cups_of_flour_added + additional_cups_of_flour = 10 :=
by {
  sorry
}

end recipe_flour_amount_l263_263575


namespace intersection_A_B_l263_263073

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1 / (x^2 + 1) }
def B : Set ℝ := {x | 3 * x - 2 < 7}

theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := 
by
  sorry

end intersection_A_B_l263_263073


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263786

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263786


namespace tian_ji_wins_probability_l263_263907

-- Define the types for horses and their relative rankings
inductive horse
| king_top : horse
| king_middle : horse
| king_bottom : horse
| tian_top : horse
| tian_middle : horse
| tian_bottom : horse

open horse

-- Define the conditions based on the problem statement
def better_than : horse → horse → Prop
| tian_top king_middle := true
| tian_top king_top := false
| tian_middle king_bottom := true
| tian_middle king_middle := false
| tian_bottom king_bottom := false
| _ _ := false

-- Topic condition for probability
def is_win (tian_horse : horse) (king_horse : horse) : Prop :=
(tian_horse = tian_top ∧ (king_horse = king_middle ∨ king_horse = king_bottom)) ∨
(tian_horse = tian_middle ∧ king_horse = king_bottom)

-- The probability statement
def win_probability : ℚ := 1/3

-- Main theorem statement
theorem tian_ji_wins_probability :
  (∑ tian_horse king_horse,
     cond (is_win tian_horse king_horse) 1 0) / 9 = win_probability :=
begin
  -- Proof is omitted
  sorry
end

end tian_ji_wins_probability_l263_263907


namespace find_7c_plus_3d_l263_263740

-- Define the functions g and f as given in the problem.
def g (x : ℝ) : ℝ := 3 * x + 2
def f (c d : ℝ) (x : ℝ) : ℝ := c * x + d
noncomputable def finv (c d : ℝ) (x : ℝ) : ℝ := 3 * x + 7

-- State the conditions as hypotheses.
theorem find_7c_plus_3d (c d : ℝ) 
  (h1 : ∀ x : ℝ, g(x) = finv(c, d, x) - 5) 
  (h2 : ∀ x : ℝ, f(c, d, finv(c, d, x)) = x) :
  7 * c + 3 * d = -14 / 3 :=
sorry

end find_7c_plus_3d_l263_263740


namespace sin_supplementary_angle_l263_263444

theorem sin_supplementary_angle (α : ℝ) (h : Real.sin (π / 4 + α) = sqrt 3 / 2) :
  Real.sin (3 * π / 4 - α) = sqrt 3 / 2 :=
sorry

end sin_supplementary_angle_l263_263444


namespace minor_to_major_axis_ratio_l263_263552

theorem minor_to_major_axis_ratio (c a : ℝ) (F1 F2 P Q : ℝ) (e : ℝ) (theta : ℝ) 
    (h1 : P F2 = F1 F2) (h2 : 3 * P F1 = 4 * Q F2) 
    (h3 : P F1 = (2 * a - 2 * c)) (h4 : Q F2 = ((4 * c) * (cos theta))) :
    2 * (sqrt 6) / 7 :=
by
  sorry

end minor_to_major_axis_ratio_l263_263552


namespace k_league_teams_l263_263602

theorem k_league_teams (n : ℕ) (h : n*(n-1)/2 = 91) : n = 14 := sorry

end k_league_teams_l263_263602


namespace diana_total_earnings_l263_263396

-- Define the earnings in each month
def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def september_earnings : ℕ := 2 * august_earnings

-- State the theorem that the total earnings over the three months is $1500
theorem diana_total_earnings : july_earnings + august_earnings + september_earnings = 1500 :=
by
  have h1 : august_earnings = 3 * july_earnings := rfl
  have h2 : september_earnings = 2 * august_earnings := rfl
  sorry

end diana_total_earnings_l263_263396


namespace correct_statements_identification_l263_263717

-- Definitions based on given conditions
def syntheticMethodCauseToEffect := True
def syntheticMethodForward := True
def analyticMethodEffectToCause := True
def analyticMethodIndirect := False
def analyticMethodBackward := True

-- The main statement to be proved
theorem correct_statements_identification :
  (syntheticMethodCauseToEffect = True) ∧ 
  (syntheticMethodForward = True) ∧ 
  (analyticMethodEffectToCause = True) ∧ 
  (analyticMethodBackward = True) ∧ 
  (analyticMethodIndirect = False) :=
by
  sorry

end correct_statements_identification_l263_263717


namespace smallest_palindrome_l263_263005

def is_palindrome (s : String) : Bool :=
  s = s.reverse

def to_base (n b : ℕ) : String :=
  let rec aux (n : ℕ) (acc : String) :=
    if n = 0 then acc
    else aux (n / b) (to_string (n % b) ++ acc)
  aux n ""

def condition (n : ℕ) : Prop :=
  n > 15 ∧ is_palindrome (to_base n 2) ∧ is_palindrome (to_base n 4)

theorem smallest_palindrome : ∃ n : ℕ, condition n ∧ ∀ m : ℕ, condition m → n ≤ m :=
  sorry

end smallest_palindrome_l263_263005


namespace gcd_of_three_numbers_l263_263623

theorem gcd_of_three_numbers (a b c : ℕ) (h1: a = 4557) (h2: b = 1953) (h3: c = 5115) : 
    Nat.gcd a (Nat.gcd b c) = 93 :=
by
  rw [h1, h2, h3]
  -- Proof goes here
  sorry

end gcd_of_three_numbers_l263_263623


namespace count_valid_n_l263_263898

theorem count_valid_n :
  let is_valid_n (n : ℕ) := ∃ (x y : ℕ), (x ≠ n) ∧ (n = x^y) ∧ (1 ≤ n ∧ n ≤ 1000000)
  (finset.range 1000001).filter is_valid_n).card = 1111 :=
by
  let is_valid_n : ℕ → Prop := λ n, ∃ (x y : ℕ), (x ≠ n) ∧ (n = x^y) ∧ (1 ≤ n ∧ n ≤ 1000000)
  exact (finset.range 1000001).filter is_valid_n).card = 1111
  sorry

end count_valid_n_l263_263898


namespace median_duration_is_105_l263_263280

def durations : List ℕ := [45, 50, 55, 58, 65, 70, 82, 95, 100, 105, 120, 130, 135, 140, 150, 165, 185, 190, 195]

theorem median_duration_is_105 :
  List.median durations = 105 := by
  sorry

end median_duration_is_105_l263_263280


namespace parabola_equation_midpoint_coordinates_l263_263891

-- Problem 1
theorem parabola_equation (p : ℝ) (h : p > 0) (focus_on_line : (p / 2 - 2) = 0) :
  ∃ p, p = 4 ∧ C = λ (x y : ℝ), y^2 = 8 * x := 
sorry

-- Problem 2
theorem midpoint_coordinates (P Q : ℝ × ℝ) (hP : ∃ (x1 y1 : ℝ), P = (x1, y1) ∧ y1^2 = 2 * x1) 
  (hQ : ∃ (x2 y2 : ℝ), Q = (x2, y2) ∧ y2^2 = 2 * x2) 
  (symmetric_about_l : P.1 - P.2 - 2 = 0 ∧ Q.1 - Q.2 - 2 = 0 ∧ P.2 ≠ Q.2) :
  ∃ M : ℝ × ℝ, M = (1, -1) :=
sorry

end parabola_equation_midpoint_coordinates_l263_263891


namespace isosceles_right_triangle_l263_263883

theorem isosceles_right_triangle (A B C D E F : Type*) [RightTriangle A B C ∧ RightAngleAtC ∧
  MedianFromAIntersectsCircumcircleAtD ∧ ProjectionOfDOntoCBIn3to2Ratio] : IsoscelesRightTriangle A B C :=
sorry

end isosceles_right_triangle_l263_263883


namespace quadratic_equation_with_root_l263_263759

theorem quadratic_equation_with_root (b c : ℚ) (h : quadratic_eq_with_root b c (√5 - 3)) : 
  b = -6 ∧ c = -4 :=
sorry

end quadratic_equation_with_root_l263_263759


namespace mean_age_euler_family_l263_263966

theorem mean_age_euler_family :
  let ages := [6, 6, 9, 11, 13, 16]
  let total_children := 6
  let total_sum := 61
  (total_sum / total_children : ℝ) = (61 / 6 : ℝ) :=
by
  sorry

end mean_age_euler_family_l263_263966


namespace positive_difference_median_mode_l263_263012

def data : List ℕ := [20, 20, 21, 21, 21, 34, 34, 35, 35, 37, 39, 41, 43, 45, 47, 47]

def median (l : List ℕ) : ℕ := 35
def mode (l : List ℕ) : ℕ := 21

theorem positive_difference_median_mode (l : List ℕ) (h : l = data) : 
  abs (median l - mode l) = 14 := 
sorry

end positive_difference_median_mode_l263_263012


namespace solve_for_x_l263_263685

theorem solve_for_x :
  ∃ x : ℚ, (15 - 2 + 4 / x) / 2 * 8 = 77 ∧ x = 16 / 25 :=
by
  use 16 / 25
  split
  sorry

end solve_for_x_l263_263685


namespace question1_question2_l263_263442

open Real

variables (e1 e2 : ℝ)
variables (k : ℝ) (CB CD AB MN : ℝ)

def non_collinear (e1 e2 : ℝ) : Prop :=
¬ (e1 = 0 ∧ e2 = 0)

def vec_AB (e1 e2 : ℝ) (k : ℝ) : ℝ :=
2 * e1 + k * e2

def vec_CB (e1 e2 : ℝ) : ℝ :=
e1 + 3 * e2

def vec_CD (e1 e2 : ℝ) : ℝ :=
2 * e1 - e2

def collinear (AB BD : ℝ) : Prop :=
∃ λ : ℝ, AB = λ * BD

theorem question1 (e1 e2 : ℝ) (k : ℝ) :
  non_collinear e1 e2 →
  vec_AB e1 e2 k = 2 * e1 + k * e2 →
  collinear (2 * e1 + k * e2) (e1 - 4 * e2) →
  k = -8 :=
begin
  intros,
  sorry
end

def vec_MN (e1 e2 : ℝ) : ℝ :=
2 * e1 + 13 * e2

def decomposition (CB CD MN : ℝ) : Prop :=
∃ m n : ℝ, MN = m * CB + n * CD

theorem question2 (e1 e2 : ℝ) :
  decomposition (e1 + 3 * e2) (2 * e1 - e2) (2 * e1 + 13 * e2) →
  vec_MN e1 e2 = 4 * (e1 + 3 * e2) - (2 * e1 - e2) :=
begin
  intros,
  sorry
end

end question1_question2_l263_263442


namespace range_sin_add_cos_l263_263391

theorem range_sin_add_cos :
  ∀ x : ℝ, -real.sqrt 2 ≤ real.sin x + real.cos x ∧ real.sin x + real.cos x ≤ real.sqrt 2 :=
sorry

end range_sin_add_cos_l263_263391


namespace custom_op_eval_l263_263419

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 5 * a + 2 * b - 1

-- State the required proof problem
theorem custom_op_eval : custom_op (-4) 6 = -9 := 
by
  -- use sorry to skip the proof
  sorry

end custom_op_eval_l263_263419


namespace estimate_frequency_limit_estimate_white_balls_probability_of_picking_same_color_l263_263886

noncomputable def frequency_approaches_half (m n : ℕ) : Prop :=
  ∀ (ε : ℝ) (hε : 0 < ε), ∃ (N : ℕ), ∀ (n ≥ N), abs ((m / n : ℝ) - 0.5) < ε

def two_white_balls (total_balls white_balls : ℕ) : Prop :=
  total_balls = 4 ∧ white_balls = 2

def probability_same_color (white_balls black_balls : ℕ) : Prop :=
  white_balls = 2 ∧ black_balls = 2 →
  (∑ b1 in range 2, ∑ b2 in range 2, if b1 = b2 then 1 else 0) / (4 * (4 - 1)) = 0.5

axiom ball_picking_experiment (m n : ℕ)
  (data : list (ℕ × ℕ × ℝ)) 
  (data_cond : data = [
    (2048, 1061, 0.518),
    (4040, 2048, 0.5069),
    (10000, 4979, 0.4979),
    (12000, 6019, 0.5016),
    (24000, 12012, 0.5005)])
  : Prop := sorry

theorem estimate_frequency_limit (data : list (ℕ × ℕ × ℝ)) : frequency_approaches_half 1 1 :=
  sorry

theorem estimate_white_balls : two_white_balls 4 2 :=
  sorry

theorem probability_of_picking_same_color : probability_same_color 2 2 :=
  sorry

end estimate_frequency_limit_estimate_white_balls_probability_of_picking_same_color_l263_263886


namespace train_speed_proof_l263_263351

noncomputable def train_speed_kmph : ℝ := sorry

theorem train_speed_proof (length_train length_bridge time_crossing : ℝ) :
  length_train = 165 →
  length_bridge = 720 →
  time_crossing = 58.9952803775698 →
  train_speed_kmph = 54 :=
by
  intros h1 h2 h3
  let total_distance := length_train + length_bridge
  have h4 : total_distance = 885, from by rwa [h1, h2]
  let speed_mps := total_distance / time_crossing
  have h5 : speed_mps = 15, from by field_simp; linarith
  let speed_kmph := speed_mps * 3.6
  have h6 : speed_kmph = 54, from by field_simp; linarith
  exact h6

end train_speed_proof_l263_263351


namespace probability_white_balls_le_1_l263_263047

-- Definitions and conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

-- Combinatorial computations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculations based on the conditions
def total_combinations : ℕ := C total_balls selected_balls
def red_combinations : ℕ := C red_balls selected_balls
def white_combinations : ℕ := C white_balls 1 * C red_balls 2

-- Probability calculations
def P_xi_le_1 : ℚ :=
  (red_combinations / total_combinations : ℚ) +
  (white_combinations / total_combinations : ℚ)

-- Problem statement: Prove that the calculated probability is 4/5
theorem probability_white_balls_le_1 : P_xi_le_1 = 4 / 5 := 
  sorry

end probability_white_balls_le_1_l263_263047


namespace rectangle_extraction_l263_263818

theorem rectangle_extraction (m : ℤ) (h1 : m > 12) : 
  ∃ (x y : ℤ), x ≤ y ∧ x * y > m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_extraction_l263_263818


namespace f_solution_set_l263_263076

theorem f_solution_set (f : ℝ → ℝ) (f' : ∀ x, Deriv f x = f' x)
  (h_deriv : ∀ x : ℝ, f' x - f x < 1) (h_init : f 0 = 2022) :
  ∀ x, (f x + 1 > 2023 * Real.exp x) ↔ x < 0 :=
by
  sorry

end f_solution_set_l263_263076


namespace non_congruent_triangles_with_perimeter_11_l263_263132

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263132


namespace det_of_commuting_matrices_l263_263212

theorem det_of_commuting_matrices (n : ℕ) (hn : n ≥ 2) (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : A * A = -1) (hAB : A * B = B * A) : 
  0 ≤ B.det := 
sorry

end det_of_commuting_matrices_l263_263212


namespace bus_trip_duration_l263_263329

theorem bus_trip_duration : 
  ∀ (departure arrival : Nat), 
  departure = 463 →  -- 7:43 a.m. is 7 * 60 + 43 = 463 minutes from midnight
  arrival = 502 →   -- 8:22 a.m. is 8 * 60 + 22 = 502 minutes from midnight
  arrival - departure = 39 := 
by
  intros departure arrival h_dep h_arr
  rw [h_dep, h_arr]
  norm_num
  sorry

end bus_trip_duration_l263_263329


namespace first_cyclist_speed_l263_263294

theorem first_cyclist_speed (v₁ v₂ : ℕ) (c t : ℕ) 
  (h1 : v₂ = 8) 
  (h2 : c = 675) 
  (h3 : t = 45) 
  (h4 : v₁ * t + v₂ * t = c) : 
  v₁ = 7 :=
by {
  sorry
}

end first_cyclist_speed_l263_263294


namespace number_of_ways_to_lineup_five_people_l263_263506

noncomputable def numPermutations (people : List Char) (constraints : List (Char × Char)) : Nat :=
  List.factorial people.length / ∏ (c : Char × Char) in constraints, (match c.1 with
    | 'A' => (people.length - 1) -- A cannot be first
    | 'E' => (people.length - 1) -- E cannot be last
    | _ => people.length) 

theorem number_of_ways_to_lineup_five_people : 
  numPermutations ['A', 'B', 'C', 'D', 'E'] [('A', 'First-line'), ('E', 'Last-line')] = 96 := 
sorry

end number_of_ways_to_lineup_five_people_l263_263506


namespace k_zero_only_solution_l263_263751

noncomputable def polynomial_factorable (k : ℤ) : Prop :=
  ∃ (A B C D E F : ℤ), (A * D = 1) ∧ (B * E = 4) ∧ (A * E + B * D = k) ∧ (A * F + C * D = 1) ∧ (C * F = -k)

theorem k_zero_only_solution : ∀ k : ℤ, polynomial_factorable k ↔ k = 0 :=
by 
  sorry

end k_zero_only_solution_l263_263751


namespace find_k_l263_263152

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x ^ 2 + (k - 1) * x + 3

theorem find_k (k : ℝ) (h : ∀ x, f k x = f k (-x)) : k = 1 :=
by
  sorry

end find_k_l263_263152


namespace count_non_congruent_triangles_with_perimeter_11_l263_263119

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263119


namespace largest_divisor_of_visible_product_l263_263361

theorem largest_divisor_of_visible_product :
  ∀ (Q : ℕ), 
  (∃ (a b c d e f g h : ℕ), 
     {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
     (∃ (x : ℕ), x ∈ {a, b, c, d, e, f, g, h} ∧ Q = a * b * c * d * e * f * g * h / x)) → 
  192 ∣ Q :=
by
sorry

end largest_divisor_of_visible_product_l263_263361


namespace magic_triangle_max_S_l263_263171

-- Definitions for the conditions in the problem
def numbers := {16, 17, 18, 19, 20, 21}
def is_magic_triangle (a b c d e f : ℕ) (S : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b + c = S ∧
  c + d + e = S ∧
  e + f + a = S

-- Proof statement that given the conditions, the largest possible S is 57
theorem magic_triangle_max_S : ∃ (a b c d e f : ℕ), is_magic_triangle a b c d e f 57 :=
sorry

end magic_triangle_max_S_l263_263171


namespace part1_part2_l263_263428

namespace Problem

-- Defining given conditions
def isOnParabola (p x y : ℝ) : Prop := y ^ 2 = 2 * p * x

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def parabolicFocus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def directrixX (p : ℝ) : ℝ := -p / 2

def distanceToDirectrix (x p : ℝ) : ℝ :=
  Real.abs (x + p / 2)

def perp (k1 k2 : ℝ) : Prop := k1 * k2 = -1

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
 ( (x1 + x2) / 2, (y1 + y2) / 2)

-- Proof Statements
theorem part1 (m p : ℝ) : 
  isOnParabola p 1 m ∧ distance 1 m (p / 2) 0 = 2 → p = 2 ∧ m = 2 :=
by
  sorry

theorem part2 (y1 y2 : ℝ) :
  isOnParabola 2 (y1 ^ 2 / 4) y1 ∧ isOnParabola 2 (y2 ^ 2 / 4) y2 ∧
  perp
    ((y1 - 2) / ((y1 ^ 2 / 4) - 1))
    ((y2 - 2) / ((y2 ^ 2 / 4) - 1)) ∧ 
  distanceToDirectrix ((midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2).fst) 2 = 15 / 2
  → (midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2) = (13 / 2, 1) ∨ 
    (midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2) = (13 / 2, -3) :=
by
  sorry

end Problem

end part1_part2_l263_263428


namespace sum_of_altitudes_of_triangle_l263_263986

theorem sum_of_altitudes_of_triangle (a b c : ℝ) (h_line : ∀ x y, 8 * x + 10 * y = 80 → x = 10 ∨ y = 8) :
  (8 + 10 + 40/Real.sqrt 41) = 18 + 40/Real.sqrt 41 :=
by
  sorry

end sum_of_altitudes_of_triangle_l263_263986


namespace general_terms_l263_263295

-- Defining the sequences with initial conditions and recurrence relations
def a (n : ℕ) : ℕ := if n = 1 then 2 else 5 * a (n - 1) + 3 * b (n - 1) + 7
def b (n : ℕ) : ℕ := if n = 1 then 1 else 3 * a (n - 1) + 5 * b (n - 1)

-- Stating the theorem for the general terms of the sequences
theorem general_terms (n : ℕ) (hn : n > 0) :
  a n = 2 ^ (3 * n - 2) + 2 ^ (n + 1) - 4 ∧
  b n = 2 ^ (3 * n - 2) - 2 ^ (n + 1) + 3 :=
by
  -- Proof to be provided
  sorry

end general_terms_l263_263295


namespace geometric_sequence_logarithm_l263_263632

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem geometric_sequence_logarithm
  (a r : ℝ) (h_pos : 0 < a) (h_pos2 : 0 < r) (h4 : geom_seq a r 4 = 2) :
  f ((geom_seq a r 1)^3) + f ((geom_seq a r 2)^3) +
  f ((geom_seq a r 3)^3) + f ((geom_seq a r 4)^3) +
  f ((geom_seq a r 5)^3) + f ((geom_seq a r 6)^3) +
  f ((geom_seq a r 7)^3) = -12 :=
begin
  sorry
end

end geometric_sequence_logarithm_l263_263632


namespace yogurt_cases_l263_263694

theorem yogurt_cases (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_case : ℕ) :
  total_cups = 960 → cups_per_box = 6 → boxes_per_case = 8 → total_cups / cups_per_box / boxes_per_case = 20 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  congr
  norm_num
  sorry

end yogurt_cases_l263_263694


namespace positional_relationship_of_two_circles_l263_263155

noncomputable def circle {α : Type*} [metric_space α] 
(radius : ℝ) (center : α) : set α := 
{p | dist p center = radius}

-- Define the conditions
def r1 : ℝ := 4
def r2 : ℝ := 3
def dist_centers : ℝ := 5
def positional_relationship : String := "Intersecting"

-- A theorem to prove the positional relationship
theorem positional_relationship_of_two_circles (r1 r2 dist_centers : ℝ) :
  r1 = 4 → r2 = 3 → dist_centers = 5 → 
  1 < dist_centers ∧ dist_centers < r1 + r2 → positional_relationship = "Intersecting" :=
begin
  intros hr1 hr2 hdist hcond,
  sorry
end

end positional_relationship_of_two_circles_l263_263155


namespace find_n_smallest_n_l263_263037

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 180) : cos (n * π / 180) = cos (865 * π / 180) ↔ (n = 35 ∨ n = 145) :=
by
  sorry

theorem smallest_n : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ cos (n * π / 180) = cos (865 * π / 180) ∧ n = 35 :=
by
  use 35
  sorry

end find_n_smallest_n_l263_263037


namespace smallest_n_for_two_distinct_tuples_l263_263304

theorem smallest_n_for_two_distinct_tuples : ∃ (n : ℕ), n = 1729 ∧ 
  (∃ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ n = x1^3 + y1^3 ∧ n = x2^3 + y2^3 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2) := sorry

end smallest_n_for_two_distinct_tuples_l263_263304


namespace angle_between_a_b_is_90_degrees_l263_263554

noncomputable def angle_between_vectors 
  (a b : ℝ^3) 
  (ha : ∥a∥ = 2) 
  (hb : ∥b∥ = 3) 
  (hab : ∥a + b∥ = Real.sqrt 13) : ℝ := 90

theorem angle_between_a_b_is_90_degrees 
  (a b : ℝ^3) 
  (ha : ∥a∥ = 2) 
  (hb : ∥b∥ = 3) 
  (hab : ∥a + b∥ = Real.sqrt 13) : angle_between_vectors a b ha hb hab = 90 := 
by sorry

end angle_between_a_b_is_90_degrees_l263_263554


namespace sum_of_roots_eq_12_l263_263697

noncomputable def g : ℝ → ℝ := sorry -- placeholder for the actual function

-- Assuming symmetry condition for the function g
axiom symmetry_condition : ∀ x : ℝ, g(3 + x) = g(3 - x)

-- Assuming g(x) = 0 has exactly four distinct real roots
axiom four_distinct_roots : ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  g(a) = 0 ∧ g(b) = 0 ∧ g(c) = 0 ∧ g(d) = 0

-- The proof statement
theorem sum_of_roots_eq_12 :
  ∀ (roots : Finset ℝ), (∀ r ∈ roots, g(r) = 0) → roots.card = 4 → roots.sum id = 12 :=
sorry

end sum_of_roots_eq_12_l263_263697


namespace sandy_comic_books_l263_263951

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l263_263951


namespace triangle_count_with_perimeter_11_l263_263104

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263104


namespace digging_well_rate_correct_l263_263405

def volume_cylinder (r h : Float) : Float := Float.pi * r^2 * h

def rate_per_cubic_meter (cost volume : Float) : Float := cost / volume

theorem digging_well_rate_correct :
  ∀ (h diameter : Float) (total_cost : Float),
    h = 14 → diameter = 3 → total_cost = 1880.2432031734913 →
    rate_per_cubic_meter total_cost (volume_cylinder (diameter / 2) h) = 19 :=
by
  intros h diameter total_cost h_eq diameter_eq cost_eq
  rw [h_eq, diameter_eq, cost_eq]
  sorry

end digging_well_rate_correct_l263_263405


namespace product_of_spins_even_probability_l263_263756

-- Definitions based on the conditions
def first_spinner := [2, 3, 5, 7, 10]
def second_spinner := [6, 9, 11, 14]

-- The total number of outcomes
def total_outcomes : ℕ := first_spinner.length * second_spinner.length

-- Function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- The count of odd numbers in the first spinner
def odd_count_first_spinner : ℕ := (first_spinner.filter is_odd).length

-- The count of odd numbers in the second spinner
def odd_count_second_spinner : ℕ := (second_spinner.filter is_odd).length

-- The count of odd-odd pairings
def odd_odd_pairings : ℕ := odd_count_first_spinner * odd_count_second_spinner

-- The probability that the product is even
def probability_even_product : ℚ := 1 - (odd_odd_pairings / total_outcomes)

-- The theorem to prove
theorem product_of_spins_even_probability : probability_even_product = 7 / 10 := 
by sorry

end product_of_spins_even_probability_l263_263756


namespace solution_set_l263_263423

noncomputable def find_ab (a b : ℝ) : Prop :=
  ∃ θ : ℝ, sin θ + cos θ = a ∧ sin θ - cos θ = b ∧ sin θ * sin θ - cos θ * cos θ - sin θ = -b * b

theorem solution_set :
  {p : ℝ × ℝ | find_ab p.1 p.2} = 
  { (Real.sqrt 7 / 2, 1 / 2), (-Real.sqrt 7 / 2, 1 / 2), (1, -1), (-1, 1) } :=
sorry

end solution_set_l263_263423


namespace not_power_of_two_l263_263248

theorem not_power_of_two (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ¬ ∃ k : ℕ, (36 * m + n) * (m + 36 * n) = 2 ^ k :=
sorry

end not_power_of_two_l263_263248


namespace parallel_lines_condition_l263_263808

theorem parallel_lines_condition (a : ℝ) :
  (a = 1 → (ax + 2y - 1 = 0) ∧ (x + (a + 1)y + 4 = 0) are_parallel) ∧ 
  ((ax + 2y - 1 = 0) ∧ (x + (a + 1)y + 4 = 0) are_parallel → (a = 1 ∨ a = -2)) :=
by
  sorry

end parallel_lines_condition_l263_263808


namespace smallest_palindrome_l263_263004

def is_palindrome (s : String) : Bool :=
  s = s.reverse

def to_base (n b : ℕ) : String :=
  let rec aux (n : ℕ) (acc : String) :=
    if n = 0 then acc
    else aux (n / b) (to_string (n % b) ++ acc)
  aux n ""

def condition (n : ℕ) : Prop :=
  n > 15 ∧ is_palindrome (to_base n 2) ∧ is_palindrome (to_base n 4)

theorem smallest_palindrome : ∃ n : ℕ, condition n ∧ ∀ m : ℕ, condition m → n ≤ m :=
  sorry

end smallest_palindrome_l263_263004


namespace circle_area_pi_div_2_l263_263387

open Real EuclideanGeometry

variable (x y : ℝ)

def circleEquation : Prop := 3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

theorem circle_area_pi_div_2
  (h : circleEquation x y) : 
  ∃ (r : ℝ), r = sqrt 0.5 ∧ π * r * r = π / 2 :=
by
  sorry

end circle_area_pi_div_2_l263_263387


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263784

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263784


namespace system_of_linear_equations_m_l263_263092

theorem system_of_linear_equations_m (x y m : ℝ) :
  (2 * x + y = 1 + 2 * m) →
  (x + 2 * y = 2 - m) →
  (x + y > 0) →
  ((2 * m + 1) * x - 2 * m < 1) →
  (x > 1) →
  (-3 < m ∧ m < -1/2) ∧ (m = -2 ∨ m = -1) :=
by
  intros h1 h2 h3 h4 h5
  -- Placeholder for proof steps
  sorry

end system_of_linear_equations_m_l263_263092


namespace part_a_part_b_part_c_l263_263680

noncomputable theory

open_locale classical

variables (A B C D O F E G : Point)
variables (r : ℝ)
variables [Circle O r]
variables [Diameter O A B]
variables [Diameter O C D]

-- Definitions based on given conditions
def perpendicular_diameters : Prop := AB ⊥ CD
def midpoint (E : Point) (OD : Segment) : Prop := E is_the_midpoint_of OD
def chord_passing (AF : Segment) (E : Point) : Prop := E ∈ AF
def intersection_point (G : Point) (AB CF : Segment) : Prop := G ∈ AB ∧ G ∈ CF

-- Hypotheses
axiom h1 : perpendicular_diameters A B C D O
axiom h2 : midpoint E (OD O D)
axiom h3 : chord_passing (AF A F) E
axiom h4 : intersection_point G (AB A B) (CF C F)

-- Proof goals
theorem part_a : (length (AF A F)) = 2 * (length (BF B F)) :=
sorry

theorem part_b : (length (OB O B)) = 3 * (length (OG O G)) :=
sorry

theorem part_c : (length (CF C F)) = 3 * (length (DF D F)) :=
sorry

end part_a_part_b_part_c_l263_263680


namespace probability_both_good_probability_both_defective_probability_exact_one_good_l263_263935

noncomputable def machine_a_quality := 0.90
noncomputable def machine_b_quality := 0.80

axiom independence (A B : Prop) : P(A && B) = P(A) * P(B)
axiom complement (p : ℝ) : P(¬A) = 1 - P(A)

def good_quality_A := P (select good part from machine A) = machine_a_quality
def good_quality_B := P (select good part from machine B) = machine_b_quality

theorem probability_both_good :
  P(selecting a good part from machine A && selecting a good part from machine B) = 0.72 :=
sorry

theorem probability_both_defective :
  P(~selecting a good part from machine A && ~selecting a good part from machine B) = 0.02 :=
sorry

theorem probability_exact_one_good :
  P(selecting a good part from machine A && ~selecting a good part from machine B || ~selecting a good part from machine A && selecting a good part from machine B) = 0.26 :=
sorry

end probability_both_good_probability_both_defective_probability_exact_one_good_l263_263935


namespace valid_k_values_for_triangle_l263_263291

-- Given three lines defined by the following equations:
def l1 (x y : ℝ) : Prop := x - y = 0
def l2 (x y : ℝ) : Prop := x + y - 2 = 0
def l3 (k x y : ℝ) : Prop := 5 * x - k * y - 15 = 0

-- Prove that the set of k values such that these lines form a triangle is k ∈ ℝ \ {±5, -10}
theorem valid_k_values_for_triangle (k : ℝ) :
  (∃ x y : ℝ, l1 x y ∧ l2 x y ∧ l3 k x y) ↔ k ∈ set.univ \ ({5,-5,-10} : set ℝ) :=
by
  sorry

end valid_k_values_for_triangle_l263_263291


namespace num_students_in_section_A_l263_263285

def avg_weight (total_weight : ℕ) (total_students : ℕ) : ℕ :=
  total_weight / total_students

variables (x : ℕ) -- number of students in section A
variables (weight_A : ℕ := 40 * x) -- total weight of section A
variables (students_B : ℕ := 20)
variables (weight_B : ℕ := 20 * 35) -- total weight of section B
variables (total_weight : ℕ := weight_A + weight_B) -- total weight of the whole class
variables (total_students : ℕ := x + students_B) -- total number of students in the class
variables (avg_weight_class : ℕ := avg_weight total_weight total_students)

theorem num_students_in_section_A :
  avg_weight_class = 38 → x = 30 :=
by
-- The proof will go here
sorry

end num_students_in_section_A_l263_263285


namespace part1_part2_l263_263514

-- Define the parametric equations of the curve C
def curve_C (α : ℝ) : ℝ × ℝ := (5 * Real.cos α, Real.sin α)

-- Define the point P
def point_P := (3 * Real.sqrt 2, 0 : ℝ)

-- Define the parametric equations of the line l with slope angle 45 degrees through P
def line_l (t : ℝ) : ℝ × ℝ := (3 * Real.sqrt 2 + t * Real.cos (Real.pi / 4), t * Real.sin (Real.pi / 4))

open Real in

theorem part1 : ∀ α : ℝ, ∃ (x y : ℝ), (curve_C α = (x, y)) ∧ (x^2 / 25 + y^2 = 1) :=
  sorry

theorem part2 : ∃ t₁ t₂ : ℝ, (line_l t₁ ∈ (curve_C '' univ)) ∧ (line_l t₂ ∈ (curve_C '' univ)) ∧ (|t₁ * t₂| = 7 / 13) :=
  sorry

end part1_part2_l263_263514


namespace suitcase_weight_on_return_l263_263528

def initial_weight : ℝ := 5
def perfume_count : ℝ := 5
def perfume_weight_oz : ℝ := 1.2
def chocolate_weight_lb : ℝ := 4
def soap_count : ℝ := 2
def soap_weight_oz : ℝ := 5
def jam_count : ℝ := 2
def jam_weight_oz : ℝ := 8
def oz_per_lb : ℝ := 16

theorem suitcase_weight_on_return :
  initial_weight + (perfume_count * perfume_weight_oz / oz_per_lb) + chocolate_weight_lb +
  (soap_count * soap_weight_oz / oz_per_lb) + (jam_count * jam_weight_oz / oz_per_lb) = 11 := 
  by
  sorry

end suitcase_weight_on_return_l263_263528


namespace num_positive_integers_satisfying_condition_l263_263800

theorem num_positive_integers_satisfying_condition :
  {x : ℕ // 30 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 60}.card = 2 :=
by
  sorry

end num_positive_integers_satisfying_condition_l263_263800


namespace power_modulo_calculation_l263_263781

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l263_263781


namespace sum_of_k_is_S_l263_263376

noncomputable def base_neg4i := (Complex.mk (-4) 1)

noncomputable def expand_k (a0 a1 a2 a3 : ℤ) : ℤ :=
  let b2 := 15 - 8 * Complex.i
  let b3 := -52 + 47 * Complex.i
  let k := a3 * b3 + a2 * b2 + a1 * base_neg4i + a0
  if k.im = 0 then k.re else 0

noncomputable def sum_of_valid_k : ℤ :=
  let digits := finset.range 17
  (∑ a0 in digits, ∑ a1 in digits, ∑ a2 in digits, ∑ a3 in digits.filter (λ x, x ≠ 0), 
      if (47 * a3 - 8 * a2 + a1 = 0) then expand_k a0 a1 a2 a3 else 0)

theorem sum_of_k_is_S (S : ℤ) : 
  sum_of_valid_k = S :=
sorry

end sum_of_k_is_S_l263_263376


namespace boat_length_l263_263327

-- Define given conditions
def breadth := 2 -- meters
def sink_depth := 0.01 -- meters
def man_mass := 60 -- kg

-- Define physical constants
def g := 9.81 -- acceleration due to gravity in m/s^2
def water_density := 1000 -- density of water in kg/m^3

-- Main problem statement
theorem boat_length :
  Σ' (L : ℝ), 
  (L * breadth * sink_depth * water_density * g = man_mass * g) ∧ (L = 3) :=
begin
  sorry
end

end boat_length_l263_263327


namespace jenny_total_wins_l263_263532

theorem jenny_total_wins (mark_games_played : ℕ) (mark_wins : ℕ) (jill_multiplier : ℕ)
  (jill_win_percent : ℚ) (jenny_vs_mark_games : ℕ := 10) (mark_wins_out_of_10 : ℕ := 1) 
  (jill_games_played : ℕ := 2 * jenny_vs_mark_games) (jill_win_percent_value : ℚ := 0.75) :
  let jenny_wins_mark := jenny_vs_mark_games - mark_wins_out_of_10,
      jenny_wins_jill := jill_games_played - (jill_win_percent_value * jill_games_played).natAbs in
  jenny_wins_mark + jenny_wins_jill = 14 :=
by
  -- Definitions
  let jenny_vs_mark_games := 10
  let mark_wins_out_of_10 := 1
  let jenny_wins_mark := jenny_vs_mark_games - mark_wins_out_of_10
  let jill_games_played := 2 * jenny_vs_mark_games
  let jill_win_percent_value := 0.75
  let jill_wins := (jill_win_percent_value * jill_games_played).toNat
  let jenny_wins_jill := jill_games_played - jill_wins
  -- Calculation
  have jenny_wins_total := jenny_wins_mark + jenny_wins_jill
  -- Expected result
  show jenny_wins_total = 14, from
    sorry

end jenny_total_wins_l263_263532


namespace integer_solutions_count_count_integer_solutions_l263_263849

theorem integer_solutions_count (x : ℤ) :
  (x ∈ (set_of (λ x : ℤ, |x - 3| ≤ 4))) ↔ x ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7} :=
by sorry

theorem count_integer_solutions :
  (finset.card (finset.filter (λ x, |x - 3| ≤ 4) (finset.range 10))) = 9 :=
by sorry

end integer_solutions_count_count_integer_solutions_l263_263849


namespace power_modulo_calculation_l263_263782

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l263_263782


namespace sum_of_medians_squared_l263_263524

theorem sum_of_medians_squared
  (A B C D E F : Type u)
  (dist_AB : ℝ) (dist_BC : ℝ) (dist_CA : ℝ)
  (h1 : dist_AB = 6)
  (h2 : dist_BC = 10)
  (h3 : dist_CA = 14)
  (midpoint_D : dist_BC / 2 = 5)
  (midpoint_E : dist_CA / 2 = 7)
  (midpoint_F : dist_AB / 2 = 3) :
  let AD2 := 91,
      BE2 := 19,
      CF2 := 139
  in AD2 + BE2 + CF2 = 249 := by
  sorry

end sum_of_medians_squared_l263_263524


namespace find_range_of_m_l263_263420

noncomputable def range_of_m (m : ℝ) : Prop :=
  ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m))

theorem find_range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∨
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3)) ↔
  ¬((∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∧
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3))) →
  range_of_m m :=
sorry

end find_range_of_m_l263_263420


namespace general_rule_equation_l263_263241

theorem general_rule_equation (n : ℕ) (hn : n > 0) : (n + 1) / n + (n + 1) = (n + 2) + 1 / n :=
by
  sorry

end general_rule_equation_l263_263241


namespace formation_count_l263_263173

theorem formation_count :
  ∃ (n : ℕ), n = {
    card (({d | 3 ≤ d ∧ d ≤ 5}.product {m | 3 ≤ m ∧ m ≤ 6}.product {f | 1 ≤ f ∧ f ≤ 3}).filter (λ ⟨⟨d, m⟩, f⟩, d + m + f = 10)) :=
  8 :=
begin
  sorry,
end

end formation_count_l263_263173


namespace sandy_comic_books_l263_263952

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l263_263952


namespace part_one_part_two_l263_263561

namespace CMO1988

def highest_power_divides_binom (n k p : ℕ) (n_i s_i t_i : Fin (k+1) → ℕ) :=
  ∀ i : Fin (k+1), n_i i < p ∧ s_i i < p ∧ t_i i < p →
    s_i.prod (λ i, p ^ i) + t_i.prod (λ i, p ^ i) = n_i.prod (λ i, p ^ i) →
    ∑ i, s_i i + t_i i - n_i i = (p-1) * highest_power_p_divides_binom (choose n (s_i.prod (λ i, p ^ i)))

theorem part_one (s t n k p : ℕ) (n_i s_i t_i : Fin (k+1) → ℕ) (hp : Nat.Prime p)
  (h_eq_sum: ∑ i, n_i i < p ∧ s_i i < p ∧ t_i i < p) :
  (∑ i, s_i i + t_i i - n_i i)/(p - 1) = (highest_power_p_divides_binom (choose n (s_i.prod (λ i, p ^ i)))) :=
sorry

theorem part_two (n k p : ℕ) (n_i : Fin (k+1) → ℕ)
  (hp : Nat.Prime p)
  (h_eq_sum: ∑ i, n_i i < p) :
  ∃ s : Fin n → ℕ, ∀ i, (0 ≤ s_i i ≤ n_i i) ∧ (p ∤ choose n (s_i.prod (λ i, p ^ i))) ∧
  (∃ count : ℕ, count = (n_i.prod (λ i, n_i i + 1)) ∧ (∀ s, s ∈ Finset.range n → p ∤ choose n (s.prod (λ i, p ^ i))) :=
sorry

end CMO1988

end part_one_part_two_l263_263561


namespace walking_negative_west_is_east_l263_263989

theorem walking_negative_west_is_east:
  ∀ (d : ℕ), (d > 0) → 
  (let west := -d in 
  let east := d in 
  -d = -west → east = d) :=
  by {
    intros d hd,
    let west := -d,
    let east := d,
    sorry
  }

end walking_negative_west_is_east_l263_263989


namespace integer_solution_count_l263_263856

theorem integer_solution_count :
  (set.count {x : ℤ | abs (x - 3) ≤ 4}) = 9 :=
sorry

end integer_solution_count_l263_263856


namespace congruent_triangle_area_l263_263355

-- Define the area and prove the statement
variable {α : Type*} [LinearOrder α]

def triangle_area {A B C : α → α} (area_ABC : α) : α := 
  area_ABC / 25

theorem congruent_triangle_area (A B C : α → α) (lines : List (α → α))
  (h₁ : list.length lines = 3)
  (h₂ : divides_triangle ABC lines)
  (h₃ : congruent_triangles ABC lines) :
  let area_ABC := area_of_triangle A B C in
  area_of_congruent_triangle ABC lines = area_ABC / 25 := by
  sorry

end congruent_triangle_area_l263_263355


namespace find_BC_l263_263282

-- Define the geometric parameters given in the problem
variable (A B C D : Type) [Trapezoid A B C D]
variable (AB CD altitude area : ℝ)
variable (AB_val : AB = 13)
variable (CD_val : CD = 20)
variable (altitude_val : altitude = 8)
variable (area_val : area = 216)

-- The goal is to find the length of BC
theorem find_BC :
  (BC : ℝ) := 
  BC = 27 - Real.sqrt 210 := sorry

end find_BC_l263_263282


namespace remainder_of_power_mod_l263_263771

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l263_263771


namespace polynomial_expansion_l263_263862

theorem polynomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 = 5^4)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 = 1) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625 :=
by
  sorry

end polynomial_expansion_l263_263862


namespace find_c_l263_263971

-- Define the polynomial equation of a parabola
def parabola_equation (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define the vertex form of a parabola
def vertex_form (a x : ℝ) : ℝ :=
  a * (x + 1)^2 - 2

/-- Prove that given the vertex of the parabola at (-1, -2)
    and it passing through the point (-2, -1), the constant term c is -1 -/
theorem find_c (a b c : ℝ) :
  ∃ a = 1 ∧ ∃ b = 2 ∧ ∃ c = -1,
  (vertex_form a (-1) = -2) ∧ (parabola_equation a b c (-2) = -1) :=
sorry

end find_c_l263_263971


namespace find_AD_l263_263196

noncomputable def triangle_AD : ℝ → ℝ → ℝ
| AB, AC => let h := 8 in h

theorem find_AD (A B C D : Type) 
  (AB AC AD : ℝ)
  (BD CD h : ℝ)
  (ratio : BD / CD = 2 / 5)
  (hyp_AB : AB = 10)
  (hyp_AC : AC = 17)
  (hyp_AD : AD = h)
  (hyp_D : D = 0) -- assuming D is on the x-axis to indicate the foot of perpendicular
  : AD = 8 :=
by
  sorry

end find_AD_l263_263196


namespace variance_scaled_data_l263_263492

theorem variance_scaled_data (x : Fin 8 → ℝ) (σ² : ℝ) (h : σ² = 3) :
    let y := (λ i, 2 * x i)
    Var y = 12 :=
by
  sorry

end variance_scaled_data_l263_263492


namespace monotonic_decreasing_interval_l263_263990

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

theorem monotonic_decreasing_interval :
  ∃ a b, (0 ≤ a ∧ a ≤ b ∧ b ≤ 2 ∧ ∀ x ∈ Icc a b, f' x ≤ 0) :=
sorry

end monotonic_decreasing_interval_l263_263990


namespace non_congruent_triangles_with_perimeter_11_l263_263140

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263140


namespace Tian_Ji_wins_probability_l263_263909

structure Horse (name : String) :=
  (isTopTier  : Bool)
  (isMidTier  : Bool)
  (isBotTier  : Bool)

variable {A : Horse} {B : Horse} {C : Horse}
variable {a : Horse} {b : Horse} {c : Horse}

axiom Tian_Ji_top : a.isMidTier = True ∧ a.isTopTier = False
axiom Tian_Ji_mid : b.isBotTier = True ∧ b.isMidTier = False
axiom Tian_Ji_bot : c.isBotTier = True

theorem Tian_Ji_wins_probability : (∑ (x : Horse × Horse), if match x with
  | (A, C) | (B, C) | (B, A) | (C, B) | (C, A) => False
  | (_, _) => True by sorry 
/ 9 ) = 1 / 3 := by sorry

end Tian_Ji_wins_probability_l263_263909


namespace principal_amount_is_200_l263_263318

theorem principal_amount_is_200 
  (R : ℝ) (P : ℝ) 
  (SI_1 = (P * R * 10) / 100) 
  (SI_2 = (P * (R + 5) * 10) / 100) 
  (h : SI_2 - SI_1 = 100) : 
  P = 200 :=
by
  -- Proof goes here
  sorry

end principal_amount_is_200_l263_263318


namespace complex_plane_squares_areas_l263_263186

theorem complex_plane_squares_areas (z : ℂ) 
  (h1 : z^3 - z = i * (z^2 - z) ∨ z^3 - z = -i * (z^2 - z))
  (h2 : z^4 - z = i * (z^3 - z) ∨ z^4 - z = -i * (z^3 - z)) :
  ( ∃ A₁ A₂ : ℝ, (A₁ = 10 ∨ A₁ = 18) ∧ (A₂ = 10 ∨ A₂ = 18) ) := 
sorry

end complex_plane_squares_areas_l263_263186


namespace projection_is_correct_l263_263912

section ProjectionOntoPlane

variables {α : Type*} [LinearOrderedField α]

noncomputable def vector_1 : ℝ × ℝ × ℝ := (7, 4, -3)
noncomputable def projection_1 : ℝ × ℝ × ℝ := (1, 7, -5)
noncomputable def vector_2 : ℝ × ℝ × ℝ := (1, -4, 9)
noncomputable def correct_projection_2 : ℝ × ℝ × ℝ := (-167/49, -88/49, 369/49)

-- Plane Q passes through the origin and satisfies the given projection condition.
noncomputable def normal_vector : ℝ × ℝ × ℝ := (-2, 1, -2/3)

theorem projection_is_correct :
  ∀ (n : ℝ × ℝ × ℝ), 
    n = normal_vector →
    (projectOntoPlane n vector_1 = projection_1) →
    projectOntoPlane n vector_2 = correct_projection_2 :=
sorry

end ProjectionOntoPlane

end projection_is_correct_l263_263912


namespace hulk_strength_l263_263609

theorem hulk_strength:
    ∃ n: ℕ, (2^(n-1) > 1000) ∧ (∀ m: ℕ, (2^(m-1) > 1000 → n ≤ m)) := sorry

end hulk_strength_l263_263609


namespace intersection_eq_l263_263440

open Set

def A := { x : ℝ | 1 < x ∧ x ≤ 3 }
def B := { -2, 1, 2, 3 }

theorem intersection_eq : A ∩ B = {2, 3} := 
by sorry

end intersection_eq_l263_263440


namespace area_of_parallelogram_l263_263728

variables {V : Type*} [InnerProductSpace ℝ V]
variables {p q a b : V}

def vec_a : V := 3 • p - 2 • q
def vec_b : V := p + 5 • q
def norm_p : ℝ := 4
def norm_q : ℝ := 1 / 2
def angle_pq : ℝ := 5 * real.pi / 6

theorem area_of_parallelogram (h_p : ∥p∥ = norm_p) (h_q : ∥q∥ = norm_q) (h_angle : real.angle p q = angle_pq) :
  1/2 * (∥vec_a∥ * ∥vec_b∥ * real.sin (h_angle)) = 17 :=
sorry

end area_of_parallelogram_l263_263728


namespace common_difference_arithmetic_geometric_sequence_l263_263832

theorem common_difference_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1) :
  d = 0 :=
by
  sorry

end common_difference_arithmetic_geometric_sequence_l263_263832


namespace part1_part2_l263_263820

-- Define the conditions
def is_real_root (n : ℕ) (a : ℝ) : Prop :=
  a^3 + a / n = 1

-- Part 1: Proving a_{n+1} > a_n
theorem part1 (n : ℕ) (n_pos : 0 < n) (a_n a_nplus1 : ℝ) (h_n : is_real_root n a_n) (h_nplus1 : is_real_root (n + 1) a_nplus1) :
  a_nplus1 > a_n :=
sorry

-- Part 2: Proving ∑_{i=1}^n 1 / ((i+1)^2 * a_i) < a_n
theorem part2 (n : ℕ) (n_pos : 0 < n) (a : ℕ → ℝ) (h : ∀ i, 0 < i → i ≤ n → is_real_root i (a i)) :
  (∑ i in Finset.range n, 1 / (((i + 2)^2 : ℝ) * (a (i + 1)))) < a n :=
sorry

end part1_part2_l263_263820


namespace ratio_bc_cd_l263_263316

theorem ratio_bc_cd (a b c d e : Point) 
(de_eq_8 : distance d e = 8)
(ab_eq_5 : distance a b = 5)
(ac_eq_11 : distance a c = 11)
(ae_eq_21 : distance a e = 21) :
  distance b c / distance c d = 3 / 1 := 
sorry

end ratio_bc_cd_l263_263316


namespace Nelly_babysit_nights_l263_263240

theorem Nelly_babysit_nights
  (friends : ℕ)
  (pizza_cost : ℕ)
  (people_per_pizza : ℕ)
  (earnings_per_night : ℕ)
  (total_people : ℕ := 1 + friends)
  (pizzas_needed : ℕ := total_people / people_per_pizza)
  (total_cost : ℕ := pizzas_needed * pizza_cost)
  (nights_needed : ℕ := total_cost / earnings_per_night) :
  friends = 14 → pizza_cost = 12 → people_per_pizza = 3 → earnings_per_night = 4 → nights_needed = 15 :=
by
  intros h_friends h_pizza_cost h_people_per_pizza h_earnings_per_night
  simp [h_friends, h_pizza_cost, h_people_per_pizza, h_earnings_per_night]
  sorry

end Nelly_babysit_nights_l263_263240


namespace car_average_speed_l263_263279

theorem car_average_speed
  (d1 d2 t1 t2 : ℕ)
  (h1 : d1 = 85)
  (h2 : d2 = 45)
  (h3 : t1 = 1)
  (h4 : t2 = 1) :
  let total_distance := d1 + d2
  let total_time := t1 + t2
  (total_distance / total_time = 65) :=
by
  sorry

end car_average_speed_l263_263279


namespace students_operate_different_ids_l263_263959

-- Define the conditions in Lean 4

def students_ids := {1, 2, 3, 4, 5}
def computers_ids := {1, 2, 3, 4, 5}
def operates (i j : ℕ) := (i ∈ students_ids) ∧ (j ∈ computers_ids) ∧ (a : ℕ × ℕ → ℕ)

theorem students_operate_different_ids (a : ℕ × ℕ → ℕ):
  (a (1, 1) * a (2, 2) * a (3, 3) * a (4, 4) * a (5, 5) = 0) →
  (∃ i, i ∈ students_ids ∧ a (i, i) = 0) :=
by
  sorry

end students_operate_different_ids_l263_263959


namespace interval_monotonic_increase_g_l263_263646

theorem interval_monotonic_increase_g
  (x : ℝ)
  (h : x ∈ Set.Icc (-(π / 2)) (π / 2)) :
  let f := λ x, 1 - 2 * sqrt 3 * cos x ^ 2 - (sin x - cos x) ^ 2
  let g := λ x, f (x + π / 3)
  ∃ a b, Set.Icc a b = Set.Icc (-(5 * π / 12)) (π / 12) ∧
         (∀ x1 x2, x1 ∈ Set.Icc a b → x2 ∈ Set.Icc a b → x1 ≤ x2 → g x1 ≤ g x2) :=
begin
  sorry
end

end interval_monotonic_increase_g_l263_263646


namespace kim_status_update_time_l263_263538

theorem kim_status_update_time :
  ∃ (x : ℕ), 5 + 9 * x + 27 = 50 ∧ x = 2 :=
begin
  use 2,
  split,
  { norm_num },   -- This verifies 5 + 9 * 2 + 27 = 50
  { refl }        -- This verifies x = 2
end

end kim_status_update_time_l263_263538


namespace Q_diff_2023_2022_l263_263797

def Q (x : ℝ) : ℝ :=
  ∑ k in (Finset.range 10000).map Nat.cast, ⌊x / k⌋

theorem Q_diff_2023_2022 : Q 2023 - Q 2022 = 6 := 
  sorry

end Q_diff_2023_2022_l263_263797


namespace calculate_averages_l263_263677

-- Definitions for conditions
variables (N M : ℕ)
variables (X : ℕ → ℕ → ℕ) -- X(I, J) gives the grade of the I-th student in the J-th subject, 0 if not taken.

-- Definitions for averages
noncomputable def student_average (i : ℕ) : ℝ :=
  let total := (finset.range M).sum (λ j, X i j)
  let count := (finset.range M).filter (λ j, X i j ≠ 0).card
  if count = 0 then 0 else total / count

noncomputable def subject_average (j : ℕ) : ℝ :=
  let total := (finset.range N).sum (λ i, X i j)
  let count := (finset.range N).filter (λ i, X i j ≠ 0).card
  if count = 0 then 0 else total / count

-- Theorem stating that we can compute the averages
theorem calculate_averages :
  (∀ i : ℕ, i < N → ∃ avg : ℝ, avg = student_average N M X i) ∧
  (∀ j : ℕ, j < M → ∃ avg : ℝ, avg = subject_average N M X j) :=
by
  sorry

end calculate_averages_l263_263677


namespace sec_product_l263_263671

theorem sec_product : ∏ k in Finset.range 1 23, (sec (4 * k) * sec (4 * k)) = 2 ^ 22 ∧ 2 + 22 = 24 :=
by
  sorry

end sec_product_l263_263671


namespace factor_1024_into_three_factors_l263_263180

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l263_263180


namespace negation_of_existence_l263_263628

theorem negation_of_existence (h: ∃ x : ℝ, 0 < x ∧ (Real.log x + x - 1 ≤ 0)) :
  ¬ (∀ x : ℝ, 0 < x → ¬ (Real.log x + x - 1 ≤ 0)) :=
sorry

end negation_of_existence_l263_263628


namespace trig_solution_l263_263604

noncomputable def solve_trig_eq (t : ℝ) : Prop :=
  (tan t / (cos (5 * t))^2 - tan (5 * t) / (cos t)^2 = 0) ∧ (cos t ≠ 0) ∧ (cos (5 * t) ≠ 0)

theorem trig_solution (t : ℝ) (k n : ℤ) :
  solve_trig_eq t ↔ (t = (ofReal (π / 12 * (2 * k + 1))) ∨ t = (ofReal (π * n))) :=
by
  sorry

end trig_solution_l263_263604


namespace salary_distribution_possible_l263_263167

theorem salary_distribution_possible
    (total_workers : ℕ)
    (total_wage : ℝ)
    (regions : ℕ)
    (population : fin regions → ℕ)
    (wage_rate : fin regions → ℝ) :
    (total_workers = ∑ i, population i) →
    (total_wage = ∑ i, population i * wage_rate i) →
    (∀ i : fin regions, population i * wage_rate i / total_wage ≤ 0.11) →
    ((∑ i, if wage_rate i > 0 then population i else 0) = total_workers / 10) →
    ((∑ i, if wage_rate i > 0 then population i * wage_rate i else 0) / total_wage = 0.9) →
    true :=
begin
    sorry
end

end salary_distribution_possible_l263_263167


namespace election_total_votes_l263_263176

theorem election_total_votes (V : ℕ) (X_votes : ℕ) (Y_votes : ℕ) (invalid_votes : ℕ) (undecided_percentage : ℚ)
    (h1 : X_votes = 40 * V / 100)
    (h2 : Y_votes = X_votes + 3000)
    (h3 : V = X_votes + Y_votes - X_votes)
    (h4 : invalid_votes = 1000)
    (h5 : undecided_percentage = 2 / 100)
    : (V + invalid_votes + undecided_percentage * (V + invalid_votes)).toNat = 16320 := by
    sorry

end election_total_votes_l263_263176


namespace find_a_from_polynomial_expansion_l263_263451

theorem find_a_from_polynomial_expansion (a : ℝ) :
  let f := (x + a)^2 * (x - 1)^3 in
  ((f.expand : polynomial ℝ).coeff 4 = 1) → a = 2 :=
by
  sorry

end find_a_from_polynomial_expansion_l263_263451


namespace remainder_when_divided_by_x_minus_1_and_x_minus_3_l263_263392

noncomputable def polynomial_remainder (p : ℚ[X]) : Prop :=
  (p.eval 1 = 2) ∧ (p.eval 3 = -4) → 
  ∃ a b q : ℚ[X], (p = (X - 1) * (X - 3) * q + (a * X + b)) ∧ (a = -3) ∧ (b = 5)

theorem remainder_when_divided_by_x_minus_1_and_x_minus_3 (p : ℚ[X]) (h1 : p.eval 1 = 2) (h2 : p.eval 3 = -4) :
  polynomial_remainder p :=
begin
  sorry
end

end remainder_when_divided_by_x_minus_1_and_x_minus_3_l263_263392


namespace positive_integral_solution_l263_263029

theorem positive_integral_solution :
  ∃ (m : ℕ), 0 < m ∧
  (∑ k in finset.range m, (2 * k + 1) / ∑ k in finset.range m, (2 * (k + 1))) = (120 / 121) :=
sorry

end positive_integral_solution_l263_263029


namespace first_place_percentage_l263_263535

theorem first_place_percentage :
  let total_pot := 8 * 5 in
  let third_place := 4 in
  let remaining_after_third := total_pot - third_place in
  let second_place := third_place in
  let first_place := remaining_after_third - second_place in
  (first_place / total_pot) * 100 = 80 :=
by
  sorry

end first_place_percentage_l263_263535


namespace proof_problem_l263_263465

-- Define the function f and g
def f (x : ℝ) := (Real.exp x - 1) / (Real.exp x + 1)
def g (x : ℝ) := f (x - 1) + 1

-- Define the sequence a_n
def a (n : ℕ+) : ℝ :=
  (∑ k in Finset.range (2 * n).filter (λ x, x % 2 = 1), g (k / n))

-- Define S_n
def S (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i, a (⟨i + 1, Nat.succ_pos i⟩))

-- Define b_n
def b (n : ℕ+) (c : ℝ) : ℝ :=
  (2 * S n - n) / (n + c)

-- Define c_n
def c (n : ℕ+) : ℝ :=
  1 / (a n * a (⟨n + 1, Nat.succ_pos n⟩))

-- Define T_n
def T (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i, c (⟨i + 1, Nat.succ_pos i⟩))

-- The Lean statement to be proven
theorem proof_problem :
  (∀ n : ℕ+, a n = 2 * n - 1) ∧
  (∀ c : ℝ, (∀ n : ℕ+, b n c = (n * (4 * n - 1)) / ((2 * n - 1) * (n + c))) → c = -1 / 2) ∧
  (∀ n : ℕ+, T n > 18 / 57) := by
  sorry

end proof_problem_l263_263465


namespace chord_bisection_l263_263879

theorem chord_bisection {r : ℝ} (PQ RS : Set (ℝ × ℝ)) (O T P Q R S M : ℝ × ℝ)
  (radius_OP : dist O P = 6) (radius_OQ : dist O Q = 6)
  (radius_OR : dist O R = 6) (radius_OS : dist O S = 6) (radius_OT : dist O T = 6)
  (radius_OM : dist O M = 2 * Real.sqrt 13) 
  (PT_eq_8 : dist P T = 8) (TQ_eq_8 : dist T Q = 8)
  (sin_theta_eq_4_5 : Real.sin (Real.arcsin (8 / 10)) = 4 / 5) :
  4 * 5 = 20 :=
by
  sorry

end chord_bisection_l263_263879


namespace primes_p_plus_10_plus_14_l263_263490

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_p_plus_10_plus_14 (p : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (p + 10)) 
  (h3 : is_prime (p + 14)) 
  : p = 3 := sorry

end primes_p_plus_10_plus_14_l263_263490


namespace area_and_perimeter_difference_l263_263975
noncomputable theory

theorem area_and_perimeter_difference 
    (d_square : ℝ) (d_circle : ℝ) 
    (h1 : d_square = 10) 
    (h2 : d_circle = 10) :
    let s := d_square / Math.sqrt 2 in
    let r := d_circle / 2 in
    let area_square := s^2 in
    let area_circle := Real.pi * r^2 in
    let perimeter_square := 4 * s in
    let circumference_circle := 2 * Real.pi * r in
    (area_circle - area_square).round = 28.5 ∧ 
    (circumference_circle - perimeter_square).round = 3.1 := 
by 
  sorry

end area_and_perimeter_difference_l263_263975


namespace non_congruent_triangles_with_perimeter_11_l263_263131

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263131


namespace sum_of_characteristic_numbers_is_96_l263_263071

open Set

noncomputable def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

-- Define A₁, A₂, A₃ with the constraints given
def is_valid_partition (A1 A2 A3 : Set ℕ) : Prop :=
  A1 ∈ (powerset M) ∧ A2 ∈ (powerset M) ∧ A3 ∈ (powerset M) ∧
  A1.card = 5 ∧ A2.card = 5 ∧ A3.card = 5 ∧
  A1 ∪ A2 ∪ A3 = M ∧ ∀ (x : ℕ), x ∈ A1 → x ∈ A2 → x ∈ A3 → False

def characteristic_number (A : Set ℕ) : ℕ :=
  A.to_finset.max' + A.to_finset.min'

theorem sum_of_characteristic_numbers_is_96 :
  ∃ (A1 A2 A3 : Set ℕ),
    is_valid_partition A1 A2 A3 ∧
    characteristic_number A1 + characteristic_number A2 + characteristic_number A3 = 96 :=
sorry

end sum_of_characteristic_numbers_is_96_l263_263071


namespace circles_touching_externally_l263_263831

noncomputable def circle_touching_externally (r1 r2 d : ℝ) : Prop := r1 + r2 = d

theorem circles_touching_externally : 
  ∀ (r1 r2 d : ℝ), r1 = 6 ∧ r2 = 2 ∧ d = 8 → circle_touching_externally r1 r2 d := 
by 
  intros r1 r2 d h 
  cases h with hr1 h 
  cases h with hr2 hd 
  unfold circle_touching_externally 
  rw [hr1, hr2, hd]
  exact rfl

end circles_touching_externally_l263_263831


namespace non_congruent_triangles_with_perimeter_11_l263_263120

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263120


namespace length_EF_l263_263573

-- Defining the points and distances
def AB : ℝ := 12
def CD : ℝ := 13
def AC : ℝ := 5
def BD : ℝ := 5

-- EF is the segment to be calculated
noncomputable def EF : ℝ :=
  (1 / 3 : ℝ) * 5 * (sqrt 7)

theorem length_EF :
  EF = (5 / 3) * sqrt 7 :=
by
  sorry

end length_EF_l263_263573


namespace min_distance_circle_A_to_line_n_l263_263468

noncomputable def circle_A_parametric (θ : Real) : Real × Real :=
  ((1 + sqrt 2 * cos θ), (-1 + sqrt 2 * sin θ))

def polar_line_n (ρ θ : Real) : Prop :=
  ρ * cos (θ + π / 4) = 4 * sqrt 2

noncomputable def center_circle_A : Real × Real := (1, -1)

noncomputable def radius_circle_A : Real := sqrt 2

def standard_form_circle_A (x y : Real) : Prop :=
  (x - 1) ^ 2 + (y + 1) ^ 2 = 2

def distance_from_point_to_line (p : Real × Real) (a b c : Real) : Real :=
  (a * p.fst + b * p.snd + c).abs / sqrt (a ^ 2 + b ^ 2)

def line_n_standard_form : Prop :=
  ∃ a b c : Real, ∀ x y : Real, polar_line_n (sqrt (x ^ 2 + y ^ 2)) (atan2 y x) ↔ a * x + b * y + c = 0

theorem min_distance_circle_A_to_line_n :
  line_n_standard_form →
  ∃ d r : Real, d = distance_from_point_to_line center_circle_A 1 (-1) 4 ∧
                r = radius_circle_A ∧
                ∀ p : Real × Real, standard_form_circle_A p.fst p.snd →
                  min_distance_p_to_line_n (p : Real × Real) = abs (d - r) :=
sorry

end min_distance_circle_A_to_line_n_l263_263468


namespace range_of_a_l263_263151

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then a * 2^(x - 1) - 1/a else (a - 2) * x + 5/3

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f a x1 - f a x2) > 0) : 2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l263_263151


namespace constant_term_binomial_l263_263418

theorem constant_term_binomial (a : ℝ) (h : a > 0) :
  ∃ (r : ℕ), r = 8 ∧ ∃ b : ℝ, b = 5 ∧ (choose 10 8) * a ^ (10 - 8) = b → a = 1 / 3 := 
by
  sorry

end constant_term_binomial_l263_263418


namespace area_ratio_gt_two_ninths_l263_263161

variables {A B C P Q R : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

def divides_perimeter_eq (A B C : Type*) (P Q R : Type*) : Prop :=
-- Definition that P, Q, and R divide the perimeter into three equal parts
sorry

def is_on_side_AB (A B C P Q : Type*) : Prop :=
-- Definition that points P and Q are on side AB
sorry

theorem area_ratio_gt_two_ninths (A B C P Q R : Type*)
  (H1 : divides_perimeter_eq A B C P Q R)
  (H2 : is_on_side_AB A B C P Q) :
  -- Statement to prove that the area ratio is greater than 2/9
  (S_ΔPQR / S_ΔABC) > (2 / 9) :=
sorry

end area_ratio_gt_two_ninths_l263_263161


namespace complex_power_sum_2013_l263_263255

noncomputable def complexPowerSum : ℂ :=
  let i := complex.I
  finset.sum (finset.range 2014) (λ n, i ^ n)

theorem complex_power_sum_2013 : complexPowerSum = 1 + complex.I :=
  sorry

end complex_power_sum_2013_l263_263255


namespace main_theorem_l263_263816

-- Define sequences and conditions
def a_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := 
  a 3 = 5 ∧ a 5 = 9 ∧ (∀ n, a n = 2 * n - 1) ∧ d = 2

def b_sequence (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop := 
  S = λ n, (1 - b n) / 2 ∧ b = λ n, (1 / (3 ^ n))

def T_n (T : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  let c n := a n * b n in
  ∀ n, T n = ∑ i in range(n+1), (2 * i - 1) / (3 ^ i) → T n = 1 - (n + 1) / (3 ^ n)

-- Prove the main statement
theorem main_theorem : ∃ a b S T, 
  (∃ d, a_sequence a d) ∧ b_sequence b S ∧ T_n T a b :=
by
  sorry

end main_theorem_l263_263816


namespace cube_partition_exists_l263_263946

theorem cube_partition_exists : ∃ (n_0 : ℕ), (0 < n_0) ∧ (∀ (n : ℕ), n ≥ n_0 → ∃ k : ℕ, n = k) := sorry

end cube_partition_exists_l263_263946


namespace simplify_120_div_180_l263_263249

theorem simplify_120_div_180 : (120 : ℚ) / 180 = 2 / 3 :=
by sorry

end simplify_120_div_180_l263_263249


namespace josh_total_spent_l263_263210

theorem josh_total_spent :
  let film_cost := 5
  let book_cost := 4
  let cd_cost := 3
  let films_bought := 9
  let books_bought := 4
  let cds_bought := 6
  let total_cost := (film_cost * films_bought) + (book_cost * books_bought) + (cd_cost * cds_bought)
  in total_cost = 79 := by
    let film_cost := 5
    let book_cost := 4
    let cd_cost := 3
    let films_bought := 9
    let books_bought := 4
    let cds_bought := 6
    let total_cost := (film_cost * films_bought) + (book_cost * books_bought) + (cd_cost * cds_bought)
    show total_cost = 79
    sorry

end josh_total_spent_l263_263210


namespace intersection_volume_is_zero_l263_263665

-- Definitions of the regions
def region1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1

-- Main theorem stating the volume of their intersection
theorem intersection_volume_is_zero : 
  ∀ (x y z : ℝ), region1 x y z ∧ region2 x y z → (x = 0 ∧ y = 0 ∧ z = 2) := 
sorry

end intersection_volume_is_zero_l263_263665


namespace chocolates_per_small_box_l263_263698

theorem chocolates_per_small_box (total_chocolates small_boxes : ℕ) 
(h_total : total_chocolates = 504) (h_boxes : small_boxes = 18) : 
(total_chocolates / small_boxes = 28) :=
by
  rw [h_total, h_boxes]
  norm_num
  sorry

end chocolates_per_small_box_l263_263698


namespace log_8_y_eq_2_point_75_l263_263144

theorem log_8_y_eq_2_point_75 (y : ℝ) (h : log 8 y = 2.75) : y = 256 * (root 4 2) :=
sorry

end log_8_y_eq_2_point_75_l263_263144


namespace max_regions_inside_smallest_circle_min_regions_inside_smallest_circle_l263_263069

variable (n k m : ℕ)

theorem max_regions_inside_smallest_circle (n k m : ℕ) : 
  max_regions n k m = (k + 1) * (m + 1) * n := sorry

theorem min_regions_inside_smallest_circle (n k m : ℕ) : 
  min_regions n k m = (k + m + 1) + n - 1 := sorry

end max_regions_inside_smallest_circle_min_regions_inside_smallest_circle_l263_263069


namespace minimum_value_l263_263522

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l263_263522


namespace remainder_of_exponentiation_is_correct_l263_263790

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l263_263790


namespace non_congruent_triangles_with_perimeter_11_l263_263109

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263109


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263785

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263785


namespace old_record_was_300_points_l263_263205

theorem old_record_was_300_points :
  let touchdowns_per_game := 4
  let points_per_touchdown := 6
  let games_in_season := 15
  let conversions := 6
  let points_per_conversion := 2
  let points_beat := 72
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + conversions * points_per_conversion
  total_points - points_beat = 300 := 
by
  sorry

end old_record_was_300_points_l263_263205


namespace remainder_5_to_5_to_5_to_5_mod_1000_l263_263769

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l263_263769


namespace find_value_of_k_l263_263548

-- Definitions of the conditions
def onLine (P : ℝ × ℝ) (k : ℝ) : Prop := k > 0 ∧ P.1 * k + P.2 + 4 = 0

def isCircle (C : ℝ × ℝ × ℝ) : Prop := C = (0, 1, 1)  -- (center_x, center_y, radius)

def minAreaQuadrilateralEqualsTwo (k : ℝ) : Prop :=
  let center := (0, 1)
  let radius := 1
  ∃ P : ℝ × ℝ, onLine P k ∧
  (dist center P = radius + distanceFromPointToLine center k) ∧
  minimumAreaQuad P center = 2

noncomputable def distanceFromPointToLine (center : ℝ × ℝ) (k : ℝ) : ℝ :=
  abs (0 * k + 1 + 4) / (real.sqrt (k^2 + 1))

noncomputable def minimumAreaQuad (P : ℝ × ℝ) (center : ℝ × ℝ) : ℝ :=
  2 * (1 / 2 * radius * dist center P)

-- Theorem statement
theorem find_value_of_k : ∃ k > 0, minAreaQuadrilateralEqualsTwo k ∧ k = real.sqrt 21 / 2 :=
sorry

end find_value_of_k_l263_263548


namespace lollipops_remainder_l263_263967

theorem lollipops_remainder :
  let total_lollipops := 8362
  let lollipops_per_package := 12
  total_lollipops % lollipops_per_package = 10 :=
by
  let total_lollipops := 8362
  let lollipops_per_package := 12
  sorry

end lollipops_remainder_l263_263967


namespace non_congruent_triangles_with_perimeter_11_l263_263121

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263121


namespace number_of_true_statements_is_3_l263_263230

variable (S : Set ℤ)
variables (a b c d : ℤ)
variables [DecidableEq ℤ]

-- Condition 1: ∀ x, y ∈ S, xy ∈ S
def cond1 : Prop := ∀ x y, x ∈ S → y ∈ S → x * y ∈ S

-- Condition 2: ∀ x, y, z ∈ S, if x ≠ y, then xz ≠ yz
def cond2 : Prop := ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → x * z ≠ y * z

-- Statement ①: Among a, b, c, d, there must be one that is 0
def stmt1 : Prop := ∃ x ∈ {a, b, c, d}, x = 0

-- Statement ②: Among a, b, c, d, there must be one that is 1
def stmt2 : Prop := ∃ x ∈ {a, b, c, d}, x = 1

-- Statement ③: If x ∈ S and xy = 1, then y ∈ S
def stmt3 : Prop := ∀ x y, x ∈ S → x * y = 1 → y ∈ S

-- Statement ④: There exist distinct x, y, z ∈ S such that x^2 = y and y^2 = z
def stmt4 : Prop := ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x^2 = y ∧ y^2 = z

theorem number_of_true_statements_is_3
  (cond1 : cond1 S) 
  (cond2 : cond2 S) 
  (stmt1_false: ¬ stmt1 a b c d)
  (stmt2_true : stmt2 a b c d)
  (stmt3_true : stmt3 S)
  (stmt4_true : stmt4 S) : 
  3 = 2 := 
sorry

end number_of_true_statements_is_3_l263_263230


namespace pipe_B_fill_time_l263_263243

-- Define the rates and times
def rateA := 1 / 16
def combined_rate := 5 / 48
variable TB : ℝ -- time for pipe B to fill the tank
variable hfilled_in_12 : 12.000000000000002> 0 -- time taken to fill 5/4 of the tank

-- Main theorem to prove
theorem pipe_B_fill_time (h : (rateA + 1 / TB) = combined_rate) : TB = 24 :=
by
  sorry

end pipe_B_fill_time_l263_263243


namespace find_constants_l263_263217

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -1], ![2, -4]]
def I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem find_constants (x y : ℚ) (hx : x = 1 / 14) (hy : y = 1 / 14) : 
  N⁻¹ = x • N + y • I := by
  sorry

end find_constants_l263_263217


namespace projection_correct_l263_263768

variable (v1 : ℝ × ℝ × ℝ × ℝ := (4, -1, 5, 2))
variable (dir : ℝ × ℝ × ℝ × ℝ := (4, -2, 3, 1))
variable (proj_v1_on_dir : ℝ × ℝ × ℝ × ℝ := (14/3, -7/3, 10.5/3, 3.5/3))

theorem projection_correct :
  let dot_product (a b : ℝ × ℝ × ℝ × ℝ) :=
    a.1 * b.1 + a.2 * b.2 + a.3 * b.3 + a.4 * b.4
  let magnitude_squared (a : ℝ × ℝ × ℝ × ℝ) :=
    dot_product a a
  let projection (v d : ℝ × ℝ × ℝ × ℝ) :=
    (dot_product v d / magnitude_squared d) • d
  projection v1 dir = proj_v1_on_dir :=
by
  sorry

end projection_correct_l263_263768


namespace linear_equation_in_two_variables_l263_263667

/--
Prove that Equation C (3x - 1 = 2 - 5y) is a linear equation in two variables 
given the equations in conditions.
-/
theorem linear_equation_in_two_variables :
  ∀ (x y : ℝ),
  (2 * x + 3 = x - 5) →
  (x * y + y = 2) →
  (3 * x - 1 = 2 - 5 * y) →
  (2 * x + (3 / y) = 7) →
  ∃ (A B C : ℝ), A * x + B * y = C :=
by 
  sorry

end linear_equation_in_two_variables_l263_263667


namespace sum_of_zeros_g_eq_l263_263741

def f (x : ℝ) : ℝ := 
  if 1 ≤ x ∧ x ≤ 2 then 4 - 8 * |x - 1.5|
  else if x > 2 then 1 / 2 * f (x / 2)
  else 0 -- this is needed since we must return a value when both conditions are not met.

def g (x : ℝ) : ℝ := x * f x - 6

theorem sum_of_zeros_g_eq (n : ℕ) (hn : n > 0) :
  let zeros := {x : ℝ | 1 ≤ x ∧ x ≤ 2^n ∧ g x = 0}
  in ∃ sum_zeros, sum_zeros = (3 / 2) * (2^n - 1) :=
sorry

end sum_of_zeros_g_eq_l263_263741


namespace ratio_of_areas_l263_263362

-- Definition specifying the context of the original triangle
def original_triangle_side : ℕ := 12

-- Definition specifying the context of the smaller triangle
def smaller_triangle_side : ℕ := 6

-- Compute the area of an equilateral triangle given its side length
def triangle_area (s : ℕ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Compute the area of the original and smaller triangles
def area_large_triangle : ℝ := triangle_area original_triangle_side
def area_small_triangle : ℝ := triangle_area smaller_triangle_side

-- Compute the area of the isosceles trapezoid by subtraction
def area_trapezoid : ℝ := area_large_triangle - area_small_triangle

-- The target proof problem to show the ratio is 1/3
theorem ratio_of_areas : (area_small_triangle / area_trapezoid) = 1 / 3 := by
  sorry

end ratio_of_areas_l263_263362


namespace min_distance_ellipse_to_line_l263_263390

open Real

noncomputable def ellipse_point (θ : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 * cos θ, sqrt 2 * sin θ)

def line_distance (x y : ℝ) : ℝ :=
  abs (2 * x - y - 8) / sqrt (2^2 + (-1)^2)

theorem min_distance_ellipse_to_line :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧
  line_distance (sqrt 2 / 2 * cos θ) (sqrt 2 * sin θ) = 6 * sqrt 5 / 5 :=
  sorry

end min_distance_ellipse_to_line_l263_263390


namespace garden_length_to_width_ratio_l263_263625

theorem garden_length_to_width_ratio (area : ℕ) (width : ℕ) (h_area : area = 432) (h_width : width = 12) :
  ∃ length : ℕ, length = area / width ∧ (length / width = 3) := 
by
  sorry

end garden_length_to_width_ratio_l263_263625


namespace inequality_cannot_hold_l263_263145

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_cannot_hold (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) := 
by {
  sorry
}

end inequality_cannot_hold_l263_263145


namespace village_current_population_l263_263499

theorem village_current_population (initial_population : ℕ) (ten_percent_die : ℕ)
  (twenty_percent_leave : ℕ) : 
  initial_population = 4399 →
  ten_percent_die = initial_population / 10 →
  twenty_percent_leave = (initial_population - ten_percent_die) / 5 →
  (initial_population - ten_percent_die) - twenty_percent_leave = 3167 :=
sorry

end village_current_population_l263_263499


namespace exists_positive_integer_n_l263_263542

variable {R : Type*} [OrderedRing R]

/--
If P is a polynomial with real coefficients such that P(x) > 0 for all x ≥ 0,
then there exists a positive integer n such that (1 + x)^n * P(x) is a polynomial with nonnegative coefficients.
-/
theorem exists_positive_integer_n (P : Polynomial R) (hP : ∀ x : R, 0 ≤ x → 0 < P.eval x) :
  ∃ n : ℕ, ∀ x : R, 0 ≤ x → 0 ≤ (Polynomial.C (1 : R) + Polynomial.X)^n * P.eval x :=
sorry

end exists_positive_integer_n_l263_263542


namespace integer_solution_unique_l263_263556

theorem integer_solution_unique (x y : ℝ) (h : -1 < (y - x) / (x + y) ∧ (y - x) / (x + y) < 2) (hyx : ∃ n : ℤ, y = n * x) : y = x :=
by
  sorry

end integer_solution_unique_l263_263556


namespace calculate_mean_score_l263_263412

theorem calculate_mean_score (M SD : ℝ) 
  (h1 : M - 2 * SD = 60)
  (h2 : M + 3 * SD = 100) : 
  M = 76 :=
by
  sorry

end calculate_mean_score_l263_263412


namespace probability_one_red_one_yellow_l263_263165

def total_eggs : ℕ := 5
def yellow_eggs : ℕ := 2
def red_eggs : ℕ := 2
def purple_egg : ℕ := 1
def drawn_eggs : ℕ := 2

-- Define a function that computes the number of favorable outcomes
noncomputable def favorable_outcomes : ℕ := yellow_eggs * red_eggs

-- Total number of ways to choose 2 eggs from 5
noncomputable def total_outcomes : ℕ := (total_eggs.choose drawn_eggs)

-- Probability that exactly 1 red and 1 yellow egg are drawn
noncomputable def probability : ℚ := favorable_outcomes / total_outcomes.toRat

theorem probability_one_red_one_yellow :
  probability = 2/5 := by sorry

end probability_one_red_one_yellow_l263_263165


namespace area_of_region_S_l263_263191

def S (x y : ℝ) : Prop := (|x| + |y| - 1) * (x^2 + y^2 - 1) ≤ 0

def area_of_S : ℝ := π - 2

theorem area_of_region_S :
  (∫ (x y : ℝ), ite (S x y) 1 0) = area_of_S :=
sorry

end area_of_region_S_l263_263191


namespace maximize_prob_C_n_l263_263928

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def P (a b : ℕ) : ℕ × ℕ := (a, b)
def C_n (n : ℕ) : Set (ℕ × ℕ) :=
  {p | (p.1 + p.2 = n)}

theorem maximize_prob_C_n : ∀ (n : ℕ), (2 ≤ n ∧ n ≤ 5) → (n = 3 ∨ n = 4) :=
begin
  -- sorry
end

end maximize_prob_C_n_l263_263928


namespace sum_of_coords_of_four_points_l263_263288

noncomputable def four_points_sum_coords : ℤ :=
  let y1 := 13 + 5
  let y2 := 13 - 5
  let x1 := 7 + 12
  let x2 := 7 - 12
  ((x2 + y2) + (x2 + y1) + (x1 + y2) + (x1 + y1))

theorem sum_of_coords_of_four_points : four_points_sum_coords = 80 :=
  by
    sorry

end sum_of_coords_of_four_points_l263_263288


namespace sqrt_a_sqrt_a_l263_263370

theorem sqrt_a_sqrt_a (a : ℝ) (h1 : sqrt a = a ^ (1 / 2))
                      (h2 : ∀ m n : ℝ, a ^ m * a ^ n = a ^ (m + n))
                      (h3 : ∀ m n : ℝ, (a ^ m) ^ n = a ^ (m * n)) :
                      sqrt (a * sqrt a) = a ^ (3 / 4) := by
  sorry

end sqrt_a_sqrt_a_l263_263370


namespace find_k_unique_solution_l263_263767

theorem find_k_unique_solution :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (k - x)/8) → (3*x^2 + (8 - 3*k)*x = 0)) →
    k = 8 / 3 :=
by
  intros k h
  -- Using sorry here to skip the proof
  sorry

end find_k_unique_solution_l263_263767


namespace count_ways_line_up_l263_263508

theorem count_ways_line_up (persons : Finset ℕ) (youngest eldest : ℕ) :
  persons.card = 5 →
  youngest ∈ persons →
  eldest ∈ persons →
  (∃ seq : List ℕ, seq.length = 5 ∧ 
    ∀ (i : ℕ), i ∈ (List.finRange 5).erase 0 → seq.get ⟨i, sorry⟩ ≠ youngest ∧ 
    i ∈ (List.finRange 5).erase 4 → seq.get ⟨i, sorry⟩ ≠ eldest) →
  (persons \ {youngest, eldest}).card = 3 →
  4 * 4 * 3 * 2 * 1 = 96 :=
by
  sorry

end count_ways_line_up_l263_263508


namespace bounded_representations_l263_263227

theorem bounded_representations 
  (λ : ℝ) (hλ : λ > 1)
  (n : ℕ → ℕ) 
  (h_seq : ∀ k : ℕ, n (k + 1) / n k > λ) :
  ∃ c : ℕ, ∀ m : ℕ, ((∃ k j, m = n k + n j) → ((∃! k j, m = n k + n j) → false)) ∧ ((∃ r s, m = n r - n s) → ((∃! r s, m = n r - n s) → false)) := 
sorry

end bounded_representations_l263_263227


namespace area_between_polar_sine_curves_l263_263729

noncomputable def polar_area_between_curves : ℝ :=
  let r1 := λ φ : ℝ, 6 * Real.sin φ in
  let r2 := λ φ : ℝ, 4 * Real.sin φ in
  (1 / 2) * ∫ φ in - (Real.pi / 2) .. (Real.pi / 2), (r1 φ)^2 - (r2 φ)^2

theorem area_between_polar_sine_curves :
  polar_area_between_curves = 5 * Real.pi :=
sorry

end area_between_polar_sine_curves_l263_263729


namespace quadrilateral_diagonal_areas_relation_l263_263688

-- Defining the areas of the four triangles and the quadrilateral
variables (A B C D Q : ℝ)

-- Stating the property to be proven
theorem quadrilateral_diagonal_areas_relation 
  (H1 : Q = A + B + C + D) :
  A * B * C * D = ((A + B) * (B + C) * (C + D) * (D + A))^2 / Q^4 :=
by sorry

end quadrilateral_diagonal_areas_relation_l263_263688


namespace equation1_solution_equation2_solution_l263_263042

theorem equation1_solution (x : ℝ) : 4 * (2 * x - 1) ^ 2 = 36 ↔ x = 2 ∨ x = -1 :=
by sorry

theorem equation2_solution (x : ℝ) : (1 / 4) * (2 * x + 3) ^ 3 - 54 = 0 ↔ x = 3 / 2 :=
by sorry

end equation1_solution_equation2_solution_l263_263042


namespace alpha_sufficient_but_not_necessary_for_cos2alpha_zero_l263_263999

theorem alpha_sufficient_but_not_necessary_for_cos2alpha_zero :
  ∀ α : ℝ, (cos (2 * α) = 0) → (α = π / 4 → true) ∧ (¬ ∀ α = π / 4) := by
sorry

end alpha_sufficient_but_not_necessary_for_cos2alpha_zero_l263_263999


namespace smallest_palindromic_prime_is_1991_l263_263749

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def smallest_four_digit_palindromic_prime : ℕ :=
  if h : ∃ n, is_prime n ∧ is_palindrome n ∧ is_four_digit n
  then well_founded.min
    ⟨_, by {
      cases h with n hn,
      exact ⟨n, hn.right.right⟩,
    }⟩
    ⟨λ a b, a < b ⟩
    (by {
      intros a ha,
      apply classical.some_spec (⟨_, ha.1, ha.2.left⟩)
    })
  else 0

theorem smallest_palindromic_prime_is_1991 :
  smallest_four_digit_palindromic_prime = 1991 :=
by {
  sorry
}

end smallest_palindromic_prime_is_1991_l263_263749


namespace simplify_sum_powers_of_i_l263_263252

open Complex
open Finset

noncomputable def sum_powers_of_i : ℂ :=
∑ i in range (2014), (I ^ i)

theorem simplify_sum_powers_of_i :
  sum_powers_of_i = 1 + I :=
by
  -- Proof here
  sorry

end simplify_sum_powers_of_i_l263_263252


namespace not_possible_perimeter_l263_263653

theorem not_possible_perimeter (x : ℝ) (h1 : 6 < x) (h2 : x < 42) : 42 + x ≠ 87 :=
by
  intro h
  have h3 : 48 < 87 := by norm_num
  have h4 : 87 < 84 := by norm_num
  linarith

end not_possible_perimeter_l263_263653


namespace max_value_f_l263_263038

noncomputable def f (x : ℝ) : ℝ := x * (1 - x^2)

theorem max_value_f : ∃ x ∈ set.Icc (0:ℝ) (1:ℝ), f x = (2 * real.sqrt 3 / 9) :=
by {
  sorry
}

end max_value_f_l263_263038


namespace find_simple_interest_rate_l263_263479

variable (P : ℝ) (n : ℕ) (r_c : ℝ) (t : ℝ) (I_c : ℝ) (I_s : ℝ) (r_s : ℝ)

noncomputable def compound_interest_amount (P r_c : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r_c / n) ^ (n * t)

noncomputable def simple_interest_amount (P r_s : ℝ) (t : ℝ) : ℝ :=
  P * r_s * t

theorem find_simple_interest_rate
  (hP : P = 5000)
  (hr_c : r_c = 0.16)
  (hn : n = 2)
  (ht : t = 1)
  (hI_c : I_c = compound_interest_amount P r_c n t - P)
  (hI_s : I_s = I_c - 16)
  (hI_s_def : I_s = simple_interest_amount P r_s t) :
  r_s = 0.1632 := sorry

end find_simple_interest_rate_l263_263479


namespace drying_time_correct_l263_263904

theorem drying_time_correct :
  let short_haired_dog_drying_time := 10
  let full_haired_dog_drying_time := 2 * short_haired_dog_drying_time
  let num_short_haired_dogs := 6
  let num_full_haired_dogs := 9
  let total_short_haired_dogs_time := num_short_haired_dogs * short_haired_dog_drying_time
  let total_full_haired_dogs_time := num_full_haired_dogs * full_haired_dog_drying_time
  let total_drying_time_in_minutes := total_short_haired_dogs_time + total_full_haired_dogs_time
  let total_drying_time_in_hours := total_drying_time_in_minutes / 60
  total_drying_time_in_hours = 4 := 
by
  sorry

end drying_time_correct_l263_263904


namespace remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263787

-- Conditions
def λ (n : ℕ) : ℕ := n.gcd20p1.factorial5div
def M : ℕ := 5 ^ (5 ^ 5)

-- Theorem: Prove the remainder
theorem remainder_5_pow_5_pow_5_pow_5_mod_500 :
  M % 500 = 125 :=
by sorry

end remainder_5_pow_5_pow_5_pow_5_mod_500_l263_263787


namespace remainder_mod_500_l263_263776

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l263_263776


namespace max_additional_pens_l263_263672

-- Outline of the problem conditions
variable (initial_amount : ℕ)
variable (num_pens_bought : ℕ)
variable (amount_left : ℕ)
variable (cost_per_pen : ℕ)

-- Problem constants
constants h1 : initial_amount = 100
constants h2 : num_pens_bought = 3
constants h3 : amount_left = 61

-- Main theorem statement
theorem max_additional_pens : 
  (amount_left = initial_amount - num_pens_bought * cost_per_pen) →
  let cost_per_pen := (initial_amount - amount_left) / num_pens_bought in
  (amount_left / cost_per_pen) = 4 :=
sorry

end max_additional_pens_l263_263672


namespace solve_for_f_2012_l263_263080

noncomputable def f : ℝ → ℝ := sorry -- as the exact function definition isn't provided

variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (functional_eqn : ∀ x, f (x + 2) = f x + f 2)
variable (f_one : f 1 = 2)

theorem solve_for_f_2012 : f 2012 = 4024 :=
sorry

end solve_for_f_2012_l263_263080


namespace domain_proof_l263_263017

def domain_of_f (x : ℝ) : Prop := x > 0 ∧ x ≠ 1

theorem domain_proof (x : ℝ) :
  domain_of_f x ↔ ((0 < x ∧ x < 1) ∨ (1 < x ∧ x)) :=
by sorry

end domain_proof_l263_263017


namespace sum_powers_of_i_l263_263954

theorem sum_powers_of_i :
  (∑ k in Finset.range (2013), complex.I ^ k) = 1 :=
  sorry

end sum_powers_of_i_l263_263954


namespace right_triangle_satisfies_pythagorean_l263_263172

-- Definition of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove
theorem right_triangle_satisfies_pythagorean :
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_satisfies_pythagorean_l263_263172


namespace geom_seq_properties_l263_263267

theorem geom_seq_properties (a : ℕ → ℝ) (q : ℝ) (T : ℕ → ℝ) (h1 : a 1 > 1) 
  (h2 : a 2009 * a 2010 - 1 > 0) 
  (h3 : (a 2009 - 1) * (a 2010 - 1) < 0) 
  (h4 : ∀ n, a (n + 1) = a n * q)
  (h5 : ∀ n, T n = (list.range n).prod (λ i, a (i + 1))) :
  0 < q ∧ q < 1 ∧
  a 2009 * a 2011 < 1 ∧
  (∀ n, T n > 1 → n ≤ 4018) := 
by {
  sorry
}

end geom_seq_properties_l263_263267


namespace inequality_sum_l263_263228

theorem inequality_sum
  (x y z : ℝ)
  (h : abs (x * y * z) = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 := 
sorry

end inequality_sum_l263_263228


namespace problem_statement_l263_263544

theorem problem_statement
  (f : ℝ → ℝ)
  (h0 : ∀ x, 0 <= x → x <= 1 → 0 <= f x)
  (h1 : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
        (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) :
  ∀ (u v w : ℝ), 
    0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1 → 
    (w - v) / (w - u) * f u + (v - u) / (w - u) * f w ≤ f v + 2 :=
by
  intros u v w h
  sorry

end problem_statement_l263_263544


namespace proof_problem_l263_263731

noncomputable def problem_statement : Prop :=
  ((Real.log10 (Real.sqrt 27) + Real.log10 8 - 3 * Real.log10 (Real.sqrt 10)) / Real.log10 1.2) = 3 / 2

theorem proof_problem : problem_statement := 
by
  sorry

end proof_problem_l263_263731


namespace point_coordinates_are_minus1_3_l263_263150

-- Define the problem conditions
def isInSecondQuadrant (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ P.2 > 0
def distanceToXAxis (P : ℝ × ℝ) : ℝ := abs P.2
def distanceToYAxis (P : ℝ × ℝ) : ℝ := abs P.1

-- State that point P is in the second quadrant and has the given distances to the axes
variables (P : ℝ × ℝ)
hypothesis (h1 : isInSecondQuadrant P)
hypothesis (h2 : distanceToXAxis P = 3)
hypothesis (h3 : distanceToYAxis P = 1)

-- Prove that the coordinates of point P are (-1, 3)
theorem point_coordinates_are_minus1_3 : P = (-1, 3) :=
by
  sorry

end point_coordinates_are_minus1_3_l263_263150


namespace magnitude_B_value_of_b_l263_263435

variable {a b c A B C : ℝ}
variable {triangle_ABC : Triangle}
variable (acute_triangle : IsAcute triangle_ABC)
variable (opp_sides : Sides triangle_ABC a b c)
variable (opp_angles : Angles triangle_ABC A B C)
variable (cond1 : √3 * a = 2 * b * sin A)
variable (cond2 : a^2 + c^2 = 7)
variable (area_cond : Triangle.area triangle_ABC = √3)

-- The magnitude of angle B is π/3
theorem magnitude_B : B = π / 3 :=
sorry

-- The value of b is √3
theorem value_of_b : b = √3 :=
sorry

end magnitude_B_value_of_b_l263_263435


namespace tangent_line_at_2_intervals_of_monotonicity_l263_263456

noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 3

theorem tangent_line_at_2 :
  (∀ y : ℝ, 12 * 2 - y - 17 = 0) :=
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, (x < 0 ∨ x > 1) → differentiable_at ℝ f x ∧ deriv f x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → differentiable_at ℝ f x ∧ deriv f x < 0) :=
  sorry

end tangent_line_at_2_intervals_of_monotonicity_l263_263456


namespace count_pairs_sin_cos_lt_zero_l263_263441

def alpha_domain := {1, 2, 3, 4, 5}
def beta_domain := {1, 2, 3, 4, 5}

noncomputable def valid_pair_count : ℕ :=
  (∑ a in alpha_domain, ∑ b in beta_domain, if (Real.sin a * Real.cos b < 0) then 1 else 0)

theorem count_pairs_sin_cos_lt_zero : valid_pair_count = 13 :=
  sorry

end count_pairs_sin_cos_lt_zero_l263_263441


namespace simplify_complex_expression_l263_263600

theorem simplify_complex_expression :
  7 * (4 - 2 * complex.I) + 2 * complex.I * (7 - 3 * complex.I) = 34 :=
by
  sorry

end simplify_complex_expression_l263_263600


namespace find_f_2_f_prime_2_l263_263452

variable (f : ℝ → ℝ)
variable (hf : ∃ t : ℝ, t = 2 ∧ f t + 2 * (t - 2) = 2 * t + 3)

theorem find_f_2_f_prime_2 :
  let f_2 := f 2,
      f_prime_2 := (deriv f 2) in
  f_2 + f_prime_2 = 9 := by
  sorry

end find_f_2_f_prime_2_l263_263452


namespace non_similar_800_pointed_stars_l263_263762

theorem non_similar_800_pointed_stars : 
  let n := 800
  ∃ stars : ℕ, 
    (stars = 158) ∧ 
    (∀ (x : ℕ) (hx : x ∣ n && gcd x 800 > 1 && gcd (n-x) 800 > 1), 
    	x ≠ 1 ∧ x ≠ 799) ∧ 
    -- Condition: no three vertices are collinear
    (∀ (v1 v2 v3 : ℕ), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 → 
      ¬(collinear v1 v2 v3)) ∧ 
    -- Condition: all line segments intersect another at a point other than an endpoint
    (∀ (s1 s2 : segment), intersects s1 s2 → ¬endpoint_intersect s1 s2) ∧ 
    -- Condition: all angles at the vertices are congruent
    (∀ (v1 v2 : vertex), 
      same_angle v1 v2) ∧ 
    -- Condition: all line segments are congruent
    (∀ (s1 s2 : segment), 
      same_length s1 s2) ∧ 
    -- Condition: the path turns counterclockwise at an angle less than 180 degrees at each vertex
    (∀ (v : vertex), 
      counterclockwise_turn v < 180)
  := 
  sorry -- proof to be provided

end non_similar_800_pointed_stars_l263_263762


namespace problem_solution_l263_263819

noncomputable def f (x : ℝ) : ℝ := sorry -- We assume the existence of f with given properties.

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom functional_equation (x : ℝ) : f (x - 4) = -f (x)
axiom definition_0_2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f (x) = Real.log (x + 1) / Real.log 2

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

axiom proof_A : f 3 = 1
axiom proof_B : decreasing_on f (-6) (-2)
axiom proof_D (m : ℝ) (h : 0 < m ∧ m < 1) : 
    let roots := {x : ℝ | 0 ≤ x ∧ x ≤ 6 ∧ f x = m} in
    ∑ x in roots, x = 4

-- The equivalent Lean 4 theorem statement
theorem problem_solution : 
  (f 3 = 1) ∧ 
  (decreasing_on f (-6) (-2)) ∧ 
  (∀ m, 0 < m ∧ m < 1 → (let roots := {x : ℝ | 0 ≤ x ∧ x ≤ 6 ∧ f x = m} in ∑ x in roots, x = 4)) :=
  by 
  apply And.intro proof_A
  apply And.intro proof_B
  exact proof_D

end problem_solution_l263_263819


namespace num_initial_pairs_of_shoes_l263_263936

theorem num_initial_pairs_of_shoes (lost_shoes remaining_pairs : ℕ)
  (h1 : lost_shoes = 9)
  (h2 : remaining_pairs = 20) :
  (initial_pairs : ℕ) = 25 :=
sorry

end num_initial_pairs_of_shoes_l263_263936


namespace max_red_points_l263_263640

theorem max_red_points (n : ℕ) (h : n = 100)
  (colored : Fin n → Bool) -- True for red, False for blue
  (segments : Fin n × Fin n → Prop) -- (i, j) where colored i ≠ colored j and a segment exists
  (unique_red_connections : ∀ i j : Fin n, colored i = true → colored j = true → 
                            (∑ k : Fin n, if segments (i, k) then 1 else 0) ≠ 
                            (∑ k : Fin n, if segments (j, k) then 1 else 0)) :
  ∃ m : ℕ, m = 50 ∧ (∀ k : ℕ, m < k → ∃ i j : Fin n, colored i = true ∧ colored j = true ∧ 
                      (∑ l : Fin n, if segments (i, l) then 1 else 0) = 
                      (∑ l : Fin n, if segments (j, l) then 1 else 0)) :=
sorry

end max_red_points_l263_263640


namespace problem_range_of_function_l263_263020

theorem problem_range_of_function :
  ∃ (range : Set ℝ), range = Set.Icc (-1/8 : ℝ) 0 ∧
  ∀ x, x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6) →
    (2 * (sin x)^2 - 3 * (sin x) + 1) ∈ range :=
by
  sorry

end problem_range_of_function_l263_263020


namespace segment_midpoints_through_center_l263_263945

variable {P : Type} [AffineSpace P ℝ]

structure Parallelogram (A B C D O M N : P) : Prop :=
(parallelogram : ∃ l m n k : ℝ, 
  l + m = 1 ∧ n + k = 1 ∧
  O = affine_combination ℝ ∧ 
  affine_combination ℝ [A, C] = 2 • O ∧ 
  affine_combination ℝ [B, D] = 2 • O)
(is_midpoint_M : M = affine_combination ℝ [B, C] 1/2)
(is_midpoint_N : N = affine_combination ℝ [A, D] 1/2)

theorem segment_midpoints_through_center 
  {A B C D O M N : P} 
  (h : Parallelogram A B C D O M N) : 
  AffineSegment ℝ M N ∋ O := 
sorry

end segment_midpoints_through_center_l263_263945


namespace sum_of_bases_is_16_l263_263378

/-
  Given the fractions G_1 and G_2 in two different bases S_1 and S_2, we need to show 
  that the sum of these bases S_1 and S_2 in base ten is 16.
-/
theorem sum_of_bases_is_16 (S_1 S_2 G_1 G_2 : ℕ) :
  (G_1 = (4 * S_1 + 5) / (S_1^2 - 1)) →
  (G_2 = (5 * S_1 + 4) / (S_1^2 - 1)) →
  (G_1 = (S_2 + 4) / (S_2^2 - 1)) →
  (G_2 = (4 * S_2 + 1) / (S_2^2 - 1)) →
  S_1 + S_2 = 16 :=
by
  intros hG1_S1 hG2_S1 hG1_S2 hG2_S2
  sorry

end sum_of_bases_is_16_l263_263378


namespace sum_powers_of_i_l263_263256

def pow_i_cycle : ℕ → ℂ
| 0 => 1
| 1 => complex.I
| 2 => -1
| 3 => -complex.I
| (n + 4) => pow_i_cycle n

theorem sum_powers_of_i : (i_sum : ℂ) → (i_sum = ∑ n in finset.range 2014, pow_i_cycle n) ∧ i_sum = 1 + complex.I :=
by
  existsi ((∑ n in finset.range 2014, pow_i_cycle n) : ℂ)
  split
  · exact rfl
  · sorry

end sum_powers_of_i_l263_263256


namespace max_sum_of_vertex_products_l263_263357

theorem max_sum_of_vertex_products : 
  ∀ (a b c d e f : ℕ),
    ({a, b, c, d, e, f} = {7, 8, 9, 10, 11, 12}) →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f →
    (∀ s t u v w x : ℕ,  {s, t, u, v, w, x} = {a, b, c, d, e, f} → s + t = u + v = w + x = 19) →
    (a + b) * (c + d) * (e + f) = 6859 :=
by
  sorry

end max_sum_of_vertex_products_l263_263357


namespace final_height_of_tree_in_4_months_l263_263234

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end final_height_of_tree_in_4_months_l263_263234


namespace power_modulo_calculation_l263_263779

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l263_263779


namespace find_n_l263_263676

theorem find_n (n : ℕ) (h : sqrt (2 * n) = 64) : n = 2048 := by
  sorry

end find_n_l263_263676


namespace non_congruent_triangles_with_perimeter_11_l263_263134

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263134


namespace find_valid_n_l263_263760

def three_points_not_collinear (S : Finset (ℝ × ℝ)) : Prop :=
∀ (A B C : (ℝ × ℝ)), A ∈ S → B ∈ S → C ∈ S → A ≠ B → B ≠ C → A ≠ C → ¬ (∃ (a b c : ℝ), a ≠ 0 → b ≠ 0 → a*fst A + b*fst B + c = y)

def no_point_inside_circle_diameter (S : Finset (ℝ × ℝ)) : Prop :=
∀ (A B C : (ℝ × ℝ)), A ∈ S → B ∈ S → C ∈ S → A ≠ B → dist (0.5 * (A + B)) C ≥ dist ((A + B)/2))

theorem find_valid_n : ∀ (n : ℕ), n ≤ 3 →
  ∃ (S : Finset (ℝ × ℝ)), S.card = n ∧ three_points_not_collinear S ∧ no_point_inside_circle_diameter S ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by
  intros n hn
  cases n
  cases n
  cases n
  -- Each case should return the respective sets that satisfy the conditions 
  -- For simplicity, use sorry as the proof content is not needed
  sorry

end find_valid_n_l263_263760


namespace sample_size_is_15_l263_263690

variables (young middle_aged elderly sample_young : ℕ)
variables (sampling_ratio : ℚ)

-- Conditions given in the problem
def unit_conditions := young = 350 ∧ middle_aged = 250 ∧ elderly = 150 ∧ sample_young = 7

-- Stratified sampling ratio
def sampling_ratio_condition := sampling_ratio = (sample_young : ℚ) / (young : ℚ)

-- Proving the sample size given the conditions
theorem sample_size_is_15 (h : unit_conditions) (r : sampling_ratio_condition) :
  let sample_middle_aged := (sampling_ratio * middle_aged : ℚ).natAbs
  let sample_elderly := (sampling_ratio * elderly : ℚ).natAbs
  sample_young + sample_middle_aged + sample_elderly = 15 :=
by
  sorry

end sample_size_is_15_l263_263690


namespace varphi_value_l263_263424

theorem varphi_value (ω : ℝ) (ϕ : ℝ)
  (h1 : 0 < ω)
  (h2 : 0 < ϕ ∧ ϕ < π)
  (h3 : ∀ x, f x = sin (ω * x + ϕ) →
              (f (π / 4) = f (5 * π / 4))):
  ϕ = π / 4 := 
sorry

end varphi_value_l263_263424


namespace integer_solution_count_l263_263855

theorem integer_solution_count :
  (set.count {x : ℤ | abs (x - 3) ≤ 4}) = 9 :=
sorry

end integer_solution_count_l263_263855


namespace find_piglets_l263_263643

theorem find_piglets (chickens piglets goats sick_animals : ℕ) 
  (h1 : chickens = 26) 
  (h2 : goats = 34) 
  (h3 : sick_animals = 50) 
  (h4 : (chickens + piglets + goats) / 2 = sick_animals) : piglets = 40 := 
by
  sorry

end find_piglets_l263_263643


namespace x_intercept_is_8_over_3_l263_263349

open Real

noncomputable def x_intercept (p1 p2: Point) : Real :=
  let m := (p2.y - p1.y) / (p2.x - p1.x)
  let b := p1.y - m * p1.x
  -b / m

def Point := { x : Real, y : Real }

def P1 : Point := { x := 2, y := -2 }
def P2 : Point := { x := 6, y := 10 }

theorem x_intercept_is_8_over_3 :
  x_intercept P1 P2 = 8 / 3 :=
by
  sorry

end x_intercept_is_8_over_3_l263_263349


namespace remainder_of_S_mod_9_is_7_l263_263996

theorem remainder_of_S_mod_9_is_7 :
  let S := (Finset.range 28).sum (λ k, Nat.choose 27 k) in
  S % 9 = 7 :=
by
  let S := (Finset.range 28).sum (λ k, Nat.choose 27 k)
  sorry

end remainder_of_S_mod_9_is_7_l263_263996


namespace cyclic_six_points_cyclic_nine_points_l263_263921

variables {P : Type*} [EuclideanGeometry P]

-- Declare points in the Euclidean plane
variables {A B C M_A M_B M_C H_A H_B H_C A' B' C' H: P}

-- Given conditions
def conditions :=
  is_triangle A B C ∧
  midpoint M_A B C ∧ midpoint M_B C A ∧ midpoint M_C A B ∧
  foot H_A A B C ∧ foot H_B B A C ∧ foot H_C C A B ∧
  orthocenter H A B C ∧
  midpoint A' A H ∧ midpoint B' B H ∧ midpoint C' C H

-- The theorem statements
theorem cyclic_six_points (h : conditions) : are_concyclic M_A M_B M_C H_A H_B H_C :=
sorry

theorem cyclic_nine_points (h : conditions) : are_concyclic M_A M_B M_C H_A H_B H_C A' B' C' :=
sorry

end cyclic_six_points_cyclic_nine_points_l263_263921


namespace shiela_family_members_l263_263527

-- Define the conditions
variables (C : ℕ) (d : ℕ)
-- Define the required number of family members
def num_family_members (C d : ℕ) : ℕ := C / d

-- State the theorem
theorem shiela_family_members (h1 : C = 50) (h2 : d = 10) : num_family_members C d = 5 :=
by {
  -- Introduction of the variables into the context
  intros,
  -- Begin with a simple calculation proving the number of family members
  -- Here, we use the given definitions
  calc
  num_family_members C d = C / d : rfl
  ... = 50 / 10 : by rw [h1, h2]
  ... = 5 : by norm_num
}

end shiela_family_members_l263_263527


namespace moles_of_CaCl2_formed_l263_263407

theorem moles_of_CaCl2_formed 
  (hcl_moles : ℕ) (caco3_moles : ℕ) : 
  ((2 * caco3_moles ≤ hcl_moles) → (hcl_moles / 2 = caco3_moles) → (caco3_moles = 3)) → 
  ∃ (cacl2_moles : ℕ), cacl2_moles = 3 :=
by
  assume h_ca_co3_hcl : (2 * 3 ≤ 6) → (6 / 2 = 3) → (3 = 3),
  use 3,
  exact eq.refl 3,
  sorry

end moles_of_CaCl2_formed_l263_263407


namespace find_n_l263_263175

-- Define the parameters of the arithmetic sequence
def a1 : ℤ := 1
def d : ℤ := 3
def a_n : ℤ := 298

-- The general formula for the nth term in an arithmetic sequence
def an (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The theorem to prove that n equals 100 given the conditions
theorem find_n (n : ℕ) (h : an n = a_n) : n = 100 :=
by
  sorry

end find_n_l263_263175


namespace five_strips_area_covered_l263_263411

theorem five_strips_area_covered :
  (let length := 12
   let width := 1
   let num_strips := 5
   let total_area_without_overlap := num_strips * length * width
   let num_overlaps := (num_strips * (num_strips - 1)) / 2
   let overlap_area_per_intersection := 1  -- Since strips overlap perpendicularly and overlap area is 1x1 units²
   let total_overlap_area := num_overlaps * overlap_area_per_intersection
   let actual_area_covered := total_area_without_overlap - total_overlap_area
   in actual_area_covered) = 50 := sorry

end five_strips_area_covered_l263_263411


namespace CF_tangent_to_Γ_l263_263174

theorem CF_tangent_to_Γ {
  (A B C D E F : Point)
  (h₁ : Triangle A B C)
  (h₂ : Acute A B C)
  (h₃ : Perpendicular CD AB)
  (h₄ : Bisects_angle ABC CD E)
  (Γ : Circle ADE)
  (h₅ : Intersects_at Γ CD F)
  (h₆ : Angle A D F = 45) :
  Tangent CF Γ :=
sorry

end CF_tangent_to_Γ_l263_263174


namespace sqrt_expr_eval_l263_263669

theorem sqrt_expr_eval : 
  sqrt (16 - 8 * sqrt 3) + sqrt (16 + 8 * sqrt 3) = 4 :=
by sorry

end sqrt_expr_eval_l263_263669


namespace average_difference_l263_263969

theorem average_difference (t : ℚ) (ht : t = 4) :
  let m := (13 + 16 + 10 + 15 + 11) / 5
  let n := (16 + t + 3 + 13) / 4
  m - n = 4 :=
by
  sorry

end average_difference_l263_263969


namespace problem_example_l263_263568

theorem problem_example : 
  let A := {0, 1, 2, 4, 5, 7}
  let B := {1, 3, 6, 8, 9}
  let C := {3, 7, 8}
  (A ∩ B) ∪ C = {1, 3, 7, 8} :=
by 
  let A := {0, 1, 2, 4, 5, 7}
  let B := {1, 3, 6, 8, 9}
  let C := {3, 7, 8}
  sorry

end problem_example_l263_263568


namespace aₙ_term_Tₙ_formula_Tₙ_min_value_l263_263066

-- Definitions and conditions for a_n sequence
def a₁ := 1 / 2
def a_seq {n : ℕ+} (a_n : ℕ+ → ℚ) : Prop := (∀ n, a_n n > 0) ∧ (∀ n, is_arithmetic_sequence (S n + a_n n, S (n + 2) + a_n (n + 2), S (n + 1) + a_n (n + 1)))

-- Define the general term formula for the sequence (a_n)
theorem aₙ_term (a_n : ℕ+ → ℚ) : a_seq a_n → (∀ n, a_n n = (1 / 2) ^ n) :=
sorry

-- Definitions and conditions for b_n sequence
def bₙ (a_n : ℕ+ → ℚ) (n : ℕ) : ℚ := 3 * a_n n + (2 * n) - 7

-- Sum T_n for b_n
def Tₙ (a_n : ℕ+ → ℚ) (n : ℕ) : ℚ := ∑ i in finset.range n, bₙ a_n (i + 1)

-- Define the formula for T_n and minimum value evaluation
theorem Tₙ_formula (a_n : ℕ+ → ℚ) : a_seq a_n → (∀ n, Tₙ a_n n = n^2 - 6*n + 3 - 3 / (2 ^ (n - 1))) :=
sorry

theorem Tₙ_min_value (a_n : ℕ+ → ℚ) : a_seq a_n → Tₙ a_n 3 = -51 / 8 :=
sorry

end aₙ_term_Tₙ_formula_Tₙ_min_value_l263_263066


namespace solve_for_d_l263_263619

theorem solve_for_d : (∃ x d : ℝ, 3 * x + 8 = 4 ∧ d * x - 15 = -5) → d = -7.5 :=
begin
  -- The proof steps would go here, however, only the statement is required
  sorry
end

end solve_for_d_l263_263619


namespace maximize_profit_l263_263725

-- Necessary definitions based on conditions
def fixed_cost : ℝ := 1.5
def variable_cost_per_unit : ℝ := 380
def revenue (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then 500 - 2 * x
  else 370 + 2140 / x - 6250 / (x^2)

def profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then -2 * x^2 + 120 * x - 150
  else -10 * x - 6250 / x + 1990

-- Lean statement for proving the maximum profit
theorem maximize_profit : 
  (∀ x : ℝ, 0 < x → profit x ≤ 1490) ∧ profit 25 = 1490 :=
by
  sorry

end maximize_profit_l263_263725


namespace tan_neg_seven_pi_six_l263_263041

noncomputable def tan_neg (α : ℝ) : ℝ := -Real.tan α
noncomputable def tan_pi_plus_alpha (α : ℝ) : ℝ := Real.tan α
noncomputable def tan_pi_six : ℝ := Real.tan (Real.pi / 6)

theorem tan_neg_seven_pi_six : Real.tan (-7 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  have h1 : Real.tan (-7 * Real.pi / 6) = tan_neg (7 * Real.pi / 6), sorry
  have h2 : (7 * Real.pi / 6) = Real.pi + Real.pi / 6, sorry
  have h3 : Real.tan (Real.pi + Real.pi / 6) = tan_pi_plus_alpha (Real.pi / 6), sorry
  have h4 : tan_pi_six = Real.sqrt 3 / 3, sorry
  rw [h1, h2, h3, h4]
  sorry

end tan_neg_seven_pi_six_l263_263041


namespace imaginary_part_zero_iff_a_eq_neg1_l263_263807

theorem imaginary_part_zero_iff_a_eq_neg1 (a : ℝ) (h : (Complex.I * (a + Complex.I) + a - 1).im = 0) : 
  a = -1 :=
sorry

end imaginary_part_zero_iff_a_eq_neg1_l263_263807


namespace cos_seq_finite_implies_rational_l263_263562

theorem cos_seq_finite_implies_rational (x y : ℝ) 
  (h_seq_finite : set.finite {s : ℝ | ∃ n : ℕ, s = (Real.cos (n * Real.pi * x) + Real.cos (n * Real.pi * y))}) :
  x ∈ ℚ ∧ y ∈ ℚ :=
sorry

end cos_seq_finite_implies_rational_l263_263562


namespace solution_set_inequality_f_solution_range_a_l263_263839

-- Define the function f 
def f (x : ℝ) := |x + 1| + |x - 3|

-- Statement for question 1
theorem solution_set_inequality_f (x : ℝ) : f x < 6 ↔ -2 < x ∧ x < 4 :=
sorry

-- Statement for question 2
theorem solution_range_a (a : ℝ) (h : ∃ x : ℝ, f x = |a - 2|) : a ≥ 6 ∨ a ≤ -2 :=
sorry

end solution_set_inequality_f_solution_range_a_l263_263839


namespace candies_in_caramel_chews_l263_263207

theorem candies_in_caramel_chews (x : ℕ) (candies : ℕ) (choc_hearts : ℕ) (choc_kisses : ℕ) (fruit_jellies : ℕ) 
  (caramel_chews : ℕ) (h1 : candies = 500) (h2 : caramel_chews = 20 - (choc_hearts + choc_kisses + fruit_jellies))
  (h3 : choc_hearts = 6) (h4 : choc_kisses = 8) (h5 : fruit_jellies = 4) 
  (h6 : 6 * (x + 2) + 8 * x + 4 * (1.5 * x) + caramel_chews * x = candies) 
  (h7 : caramel_chews = 2) : 2 * x = 44 :=
by
  sorry

end candies_in_caramel_chews_l263_263207


namespace czechoslovak_inequality_l263_263720

-- Define the triangle and the points
structure Triangle (α : Type) [LinearOrderedRing α] :=
(A B C : α × α)

variables {α : Type} [LinearOrderedRing α]

-- Define the condition that O is on the segment AB but is not a vertex
def on_segment (O A B : α × α) : Prop :=
  ∃ x : α, 0 < x ∧ x < 1 ∧ O = (A.1 + x * (B.1 - A.1), A.2 + x * (B.2 - A.2))

-- Define the dot product for vectors
def dot (u v: α × α) : α := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem czechoslovak_inequality (T : Triangle α) (O : α × α) (hO : on_segment O T.A T.B) :
  dot O T.C * dot T.A T.B < dot T.A O * dot T.B T.C + dot T.B O * dot T.A T.C :=
sorry

end czechoslovak_inequality_l263_263720


namespace proof_x_eq_y_l263_263821

variable (x y z : ℝ)

theorem proof_x_eq_y (h1 : x = 6 - y) (h2 : z^2 = x * y - 9) : x = y := 
  sorry

end proof_x_eq_y_l263_263821


namespace hyperbola_condition_ellipse_with_foci_on_x_axis_condition_l263_263974

open Real

def curve (k : ℝ) := ∀ x y : ℝ, x^2 / (4 - k) + y^2 / (k - 1) = 1

theorem hyperbola_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 / (4 - k) + y^2 / (k - 1) = 1 ∧ 
  (superellipses_conditions : (4 - k) * (k - 1) < 0 → (k < 1 ∨ k > 4))) :=
sorry

theorem ellipse_with_foci_on_x_axis_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 / (4 - k) + y^2 / (k - 1) = 1 ∧
  (ellipse_foci_conditions : 4 - k > k - 1 ∧ k - 1 > 0 → 1 < k ∧ k < 5/2)) :=
sorry

end hyperbola_condition_ellipse_with_foci_on_x_axis_condition_l263_263974


namespace translated_circle_contains_lattice_point_l263_263332

theorem translated_circle_contains_lattice_point
  (r : ℝ) (h_r : 0 < r) (a1 a2 : ℝ) :
  ∃ (n : ℕ+), ∃ (m1 m2 : ℤ), (n:ℕ) • (a1, a2) - (m1, m2) = (x,y) ∧ (((m1 + r)^2 + (m2 + r)^2)) < r^2 :=
begin
  sorry
end

end translated_circle_contains_lattice_point_l263_263332


namespace sin_cos_identity_l263_263315

theorem sin_cos_identity (z : ℝ) :
  (∃ k : ℤ, z = (π / 18) * (6 * k + 1) ∨ z = (π / 18) * (6 * k - 1))
  ↔
  (1 - sin(z)^6 - cos(z)^6) / (1 - sin(z)^4 - cos(z)^4) = 2 * cos(3 * z)^2 :=
sorry

end sin_cos_identity_l263_263315


namespace solve_for_x_l263_263956

theorem solve_for_x (x : ℝ) (h : log 2 x + log 8 x = 5) : x = 2 ^ (15 / 4) :=
by
  sorry

end solve_for_x_l263_263956


namespace condition_swap_l263_263095

variable {p q : Prop}

theorem condition_swap (h : ¬ p → q) (nh : ¬ (¬ p ↔ q)) : (p → ¬ q) ∧ ¬ (¬ (p ↔ ¬ q)) :=
by
  sorry

end condition_swap_l263_263095


namespace find_a_l263_263836

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x - 1 else x + 1

theorem find_a (a : ℝ) (h : f a = f 1) : a = 0 ∨ a = 1 := 
  sorry

end find_a_l263_263836


namespace largest_binomial_coeff_and_rational_terms_l263_263826

theorem largest_binomial_coeff_and_rational_terms 
  (n : ℕ) 
  (h_sum_coeffs : 4^n - 2^n = 992) 
  (T : ℕ → ℝ → ℝ)
  (x : ℝ) :
  (∃ (r1 r2 : ℕ), T r1 x = 270 * x^(22/3) ∧ T r2 x = 90 * x^6)
  ∧
  (∃ (r3 r4 : ℕ), T r3 x = 243 * x^10 ∧ T r4 x = 90 * x^6)
:= 
  
sorry

end largest_binomial_coeff_and_rational_terms_l263_263826


namespace range_of_a_l263_263493

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ x^2 + (1 - a) * x + 3 - a > 0) ↔ a < 3 := 
sorry

end range_of_a_l263_263493


namespace initial_strawberries_l263_263581

-- Define the conditions
def strawberries_eaten : ℝ := 42.0
def strawberries_left : ℝ := 36.0

-- State the theorem
theorem initial_strawberries :
  strawberries_eaten + strawberries_left = 78 :=
by
  sorry

end initial_strawberries_l263_263581


namespace length_of_segment_AB_l263_263512

def line (t : Real) : Real × Real :=
  (1 + (1/2) * t, (Real.sqrt 3) / 2 * t)

def ellipse (θ : Real) : Real × Real :=
  (Real.cos θ, 2 * Real.sin θ)

def general_form (x y : Real) : Prop :=
  x^2 + y^2 / 4 = 1

theorem length_of_segment_AB :
  let A := line 0
  let B := line (-8/7)
  dist A B = 2 * Real.sqrt 7 / 7 :=
by
  sorry

end length_of_segment_AB_l263_263512


namespace max_artillery_range_max_distance_to_flying_object_l263_263331

-- (1) Prove the maximum range is 10 kilometers
theorem max_artillery_range (k : ℝ) (hk : k > 0) : 
  let y x := (λ k x : ℝ, k * x - (k^2 + 1) / 20 * x^2) in
  ∃ x : ℝ, y k x = 0 ∧ x ≤ 10 := 
by sorry

-- (2) Prove the maximum horizontal distance to the flying object is 6 kilometers at height 3.2 km
theorem max_distance_to_flying_object (k : ℝ) (hk : k > 0) : 
  let y a := (λ k a : ℝ, k * a - (k^2 + 1) / 20 * a^2) in
  let h := 3.2 in 
  ∃ a : ℝ, y k a = h ∧ a ≤ 6 := 
by sorry

end max_artillery_range_max_distance_to_flying_object_l263_263331


namespace sum_of_solutions_l263_263911

-- Define the system of equations as lean functions
def equation1 (x y : ℝ) : Prop := |x - 4| = |y - 10|
def equation2 (x y : ℝ) : Prop := |x - 10| = 3 * |y - 4|

-- Statement of the theorem
theorem sum_of_solutions : 
  ∃ (solutions : List (ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ), sol ∈ solutions → equation1 sol.1 sol.2 ∧ equation2 sol.1 sol.2) ∧ 
    (List.sum (solutions.map (fun sol => sol.1 + sol.2)) = 24) :=
  sorry

end sum_of_solutions_l263_263911


namespace find_prime_numbers_l263_263630

noncomputable def is_prime : ℕ → Prop := sorry

theorem find_prime_numbers :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
                 p * q * r = 5 * (p + q + r) ∧
                 {p, q, r} = {2, 5, 7} :=
by
  sorry

end find_prime_numbers_l263_263630


namespace jenny_total_wins_l263_263530

-- Definitions based on conditions
def games_mark : Nat := 10
def mark_wins : Nat := 1
def jill_wins_percent : Real := 0.75

-- Calculations based on definitions
def jenny_wins_mark : Nat := games_mark - mark_wins
def games_jill : Nat := 2 * games_mark
def jill_wins : Nat := floor (jill_wins_percent * games_jill).toNat -- convert from Real to Nat
def jenny_wins_jill : Nat := games_jill - jill_wins

-- Total wins
def total_wins : Nat := jenny_wins_mark + jenny_wins_jill

theorem jenny_total_wins : total_wins = 14 := by
  -- proof goes here
  sorry

end jenny_total_wins_l263_263530


namespace percentage_sum_l263_263605

theorem percentage_sum (A B C : ℕ) (x y : ℕ)
  (hA : A = 120) (hB : B = 110) (hC : C = 100)
  (hAx : A = C * (1 + x / 100))
  (hBy : B = C * (1 + y / 100)) : x + y = 30 := 
by
  sorry

end percentage_sum_l263_263605


namespace shanghai_masters_total_matches_l263_263757

theorem shanghai_masters_total_matches : 
  let players := 8
  let groups := 2
  let players_per_group := 4
  let round_robin_matches_per_group := (players_per_group * (players_per_group - 1)) / 2
  let round_robin_total_matches := round_robin_matches_per_group * groups
  let elimination_matches := 2 * (groups - 1)  -- semi-final matches
  let final_matches := 2  -- one final and one third-place match
  round_robin_total_matches + elimination_matches + final_matches = 16 :=
by
  sorry

end shanghai_masters_total_matches_l263_263757


namespace volume_of_pool_l263_263708

theorem volume_of_pool :
  let diameter := 60
  let radius := diameter / 2
  let height_shallow := 3
  let height_deep := 15
  let height_total := height_shallow + height_deep
  let volume_cylinder := π * radius^2 * height_total
  volume_cylinder / 2 = 8100 * π :=
by
  sorry

end volume_of_pool_l263_263708


namespace swimming_speed_in_still_water_l263_263342

-- Given conditions
def water_speed : ℝ := 4
def swim_time_against_current : ℝ := 2
def swim_distance_against_current : ℝ := 8

-- What we are trying to prove
theorem swimming_speed_in_still_water (v : ℝ) 
    (h1 : swim_distance_against_current = 8) 
    (h2 : swim_time_against_current = 2)
    (h3 : water_speed = 4) :
    v - water_speed = swim_distance_against_current / swim_time_against_current → v = 8 :=
by
  sorry

end swimming_speed_in_still_water_l263_263342


namespace non_congruent_triangles_with_perimeter_11_l263_263135

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263135


namespace roots_product_of_quadratic_equation_l263_263146

variables (a b : ℝ)

-- Given that a and b are roots of the quadratic equation x^2 - ax + b = 0
-- and given conditions that a + b = 5 and ab = 6,
-- prove that a * b = 6.
theorem roots_product_of_quadratic_equation 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) : 
  a * b = 6 := 
by 
 sorry

end roots_product_of_quadratic_equation_l263_263146


namespace required_run_rate_l263_263513

theorem required_run_rate 
  (run_rate_first_20 : ℝ := 4.2)
  (total_overs_first : ℕ := 20)
  (runs_scored_first_20 : ℝ := run_rate_first_20 * total_overs_first)
  (target_total_runs : ℝ := 250)
  (remaining_runs_needed : ℝ := target_total_runs - runs_scored_first_20)
  (remaining_overs : ℕ := 30) :
  (required_run_rate_remaining : ℝ := remaining_runs_needed / remaining_overs) = 5.53 := 
by
  sorry

end required_run_rate_l263_263513


namespace range_of_a_l263_263631

noncomputable def line1 (a : ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + a = 0}
noncomputable def line2 (a : ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + a^2 + 1 = 0}
noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 = 0}

def tangent (a : ℝ) : Prop := 
  -- Definition that checks the tangency condition
  ∃ p : ℝ × ℝ, line1 a p ∧ line2 a p ∧ circle p

theorem range_of_a :
  ∀ a : ℝ, (tangent a ↔ (-3 ≤ a ∧ a ≤ -real.sqrt 6) ∨ (real.sqrt 6 ≤ a ∧ a ≤ 7)) :=
by sorry

end range_of_a_l263_263631


namespace trajectory_of_center_of_moving_circle_l263_263340

noncomputable def circle_tangency_condition_1 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 1
noncomputable def circle_tangency_condition_2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 9

def ellipse_equation (x y : ℝ) : Prop := x ^ 2 / 4 + y ^ 2 / 3 = 1

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  circle_tangency_condition_1 x y ∧ circle_tangency_condition_2 x y →
  ellipse_equation x y := sorry

end trajectory_of_center_of_moving_circle_l263_263340


namespace domain_shift_l263_263870

theorem domain_shift (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x | -2 ≤ x ∧ x ≤ -1} →
  {x : ℝ | ∃ y : ℝ, x = y - 1 ∧ 1 ≤ y ∧ y ≤ 2} =
  {x : ℝ | ∃ y : ℝ, x = y + 2 ∧ -2 ≤ y ∧ y ≤ -1} :=
by
  sorry

end domain_shift_l263_263870


namespace prime_game_win_l263_263655

noncomputable def prime_game_strategy (P : List ℕ) : Bool :=
  -- Defines a function to encapsulate the game's strategy.
  sorry -- Detailed logic will be implemented here.

theorem prime_game_win :
  ∃ strategy : List ℕ, 
  (∀ (n : ℕ), n ∈ strategy → Prime n ∧ n ≤ 100) ∧
  (∀ (i j : ℕ), i < j → ((strategy[i].digits).last = (strategy[j].digits).head)) ∧
  (strategy.nodup) ∧
  (prime_game_strategy strategy = True) ∧
  strategy.length = 3 :=
by
  -- Proof to establish that there exists a strategy ensuring a win with exactly 3 primes.
  sorry

end prime_game_win_l263_263655


namespace prime_divides_ap_minus_b_l263_263213

theorem prime_divides_ap_minus_b 
  (p : ℕ) (hp : p > 3) (prime_p : Nat.Prime p) 
  (a b : ℕ) (hab : 1 + ∑ k in Finset.range p, (1 : ℚ) / (k+1) = a / b) :
  p^4 ∣ (a * p - b) := 
sorry

end prime_divides_ap_minus_b_l263_263213


namespace difference_second_largest_second_smallest_l263_263289

/-- Problem statement: Given three specific numbers, prove that the difference between the second largest and the second smallest is zero. -/
theorem difference_second_largest_second_smallest :
  let a := 10
  let b := 11
  let c := 12
  (∃ l : List Nat, l = [a, b, c] ∧ 
    l.nth_le (l.length - 2) (by sorry) = b ∧
    l.nth_le 1 (by sorry) = b ∧
    b - b = 0) :=
begin
  let a := 10,
  let b := 11,
  let c := 12,
  let l := [a, b, c],
  have h_length : l.length = 3 := by sorry,
  have h_second_largest : l.nth_le (l.length - 2) (by sorry) = b := by sorry,
  have h_second_smallest : l.nth_le 1 (by sorry) = b := by sorry,
  exact ⟨l, rfl, h_second_largest, h_second_smallest, by ring⟩,
end

end difference_second_largest_second_smallest_l263_263289


namespace egypt_free_tourists_l263_263321

theorem egypt_free_tourists (x : ℕ) :
  (13 + 4 * x = x + 100) → x = 29 :=
by {
  intros h,
  have h1 : 4 * x - x = 87, { linarith },
  have h2 : 3 * x = 87, { linarith },
  linarith,
  sorry -- Placeholder for the final steps of the proof
}

end egypt_free_tourists_l263_263321


namespace area_of_triangle_QRS_l263_263944

-- Define the points in 3D space
structure point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Define the Euclidean distance in 3D space
def dist (p1 p2 : point3D) : ℝ :=
real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define the angles in 3D space
def angle (p1 p2 p3 : point3D) : ℝ := sorry -- This is simplified, actual implementation is omitted

-- Define the area of triangle formed by given three points
def triangle_area (p1 p2 p3 : point3D) : ℝ :=
let a := dist p1 p2,
    b := dist p2 p3,
    c := dist p3 p1,
    s := (a + b + c) / 2 in
real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Given the points P, Q, R, S, T with mentioned properties
variables (P Q R S T : point3D)

-- Theorem to prove the area of triangle QRS
theorem area_of_triangle_QRS
  (hPQ : dist P Q = 3)
  (hQR : dist Q R = 3)
  (hRS : dist R S = 3)
  (hST : dist S T = 3)
  (hTP : dist T P = 3)
  (anglePQR : angle P Q R = real.pi / 3)
  (angleRST : angle R S T = real.pi / 3)
  (angleSTP : angle S T P = real.pi / 3)
  (plane_parallel : sorry) -- This parallel condition needs a formal expression
  : triangle_area Q R S = 9 * real.sqrt 3 / 4 :=
sorry

end area_of_triangle_QRS_l263_263944


namespace SufficientCondition_l263_263055

def PropositionP (a b c d : Prop) := a ≥ b → c > d
def PropositionQ (e f a b : Prop) := e ≤ f → a < b

theorem SufficientCondition (a b c d e f : Prop)
  (P: PropositionP a b c d)
  (¬Q: ¬PropositionQ e f a b) :
  c ≤ d → e ≤ f :=
by
  sorry

end SufficientCondition_l263_263055


namespace total_short_trees_after_planting_l263_263286

def initial_short_trees : ℕ := 31
def planted_short_trees : ℕ := 64

theorem total_short_trees_after_planting : initial_short_trees + planted_short_trees = 95 := by
  sorry

end total_short_trees_after_planting_l263_263286


namespace count_even_three_digit_numbers_l263_263657

theorem count_even_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  ∃ n : ℕ, 
    ∃ hundreds tens units : ℕ, 
      hundreds ∈ digits ∧ 
      tens ∈ digits ∧ 
      units ∈ even_digits ∧
      100*hundreds + 10*tens + units < 700 ∧
      100*hundreds + 10*tens + units < 1000 ∧
      n = 6 * 6 * 3 ∧
      n = 108 :=
sorry

end count_even_three_digit_numbers_l263_263657


namespace sulfuric_acid_moles_l263_263408

-- Definitions based on the conditions
def iron_moles := 2
def hydrogen_moles := 2

-- The reaction equation in the problem
def reaction (Fe H₂SO₄ : ℕ) : Prop :=
  Fe + H₂SO₄ = hydrogen_moles

-- Goal: prove the number of moles of sulfuric acid used is 2
theorem sulfuric_acid_moles (Fe : ℕ) (H₂SO₄ : ℕ) (h : reaction Fe H₂SO₄) :
  H₂SO₄ = 2 :=
sorry

end sulfuric_acid_moles_l263_263408


namespace formation_of_number_l263_263890

theorem formation_of_number 
  (x1 y1 x2 y2 : ℕ) 
  (x1_pos : x1 > 0) (y1_pos : y1 > 0) (x2_pos : x2 > 0) (y2_pos : y2 > 0) 
  (angle_OA_gt_45 : y1 > x1) (angle_OB_lt_45 : x2 < y2)
  (area_cond : x1 * y1 + 67 = x2 * y2) : 
  "1985" = to_digit_string x1 y1 x2 y2 := 
sorry

def to_digit_string (x1 y1 x2 y2 : ℕ) : String := 
  toString x1 ++ toString y1 ++ toString x2 ++ toString y2

end formation_of_number_l263_263890


namespace negation_of_exists_l263_263841
open Real

theorem negation_of_exists (p : ∃ x : ℝ, 4^x > x^4) : (¬ p) ↔ ∀ x : ℝ, 4^x ≤ x^4 :=
by
  sorry

end negation_of_exists_l263_263841


namespace largest_A_l263_263543

namespace EquivalentProofProblem

def F (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f (3 * x) ≥ f (f (2 * x)) + x

theorem largest_A (f : ℝ → ℝ) (hf : F f) (x : ℝ) (hx : x > 0) : 
  ∃ A, (∀ (f : ℝ → ℝ), F f → ∀ x, x > 0 → f x ≥ A * x) ∧ A = 1 / 2 :=
sorry

end EquivalentProofProblem

end largest_A_l263_263543


namespace maximum_value_of_f_l263_263453

theorem maximum_value_of_f :
  ∃ x : ℝ, f (x) = Real.exp 1 / 2 ∧ 
  (∀ y : ℝ, y ≠ x → f y ≤ f x)
  (f has_deriv_at f' x) [differentiable ℝ f'] :
  ∀ x : ℝ,
    (x * f'(x) + 2 * f(x) = 1 / x^2) ∧ 
    (f(1) = 1) → (∃ x : ℝ, f(x) ≤ Real.exp 1 / 2) :=
sorry

end maximum_value_of_f_l263_263453


namespace sin_identity_l263_263446

theorem sin_identity (α : ℝ) (h : Real.sin (π/4 + α) = √3 / 2) : Real.sin (3*π/4 - α) = √3 / 2 :=
  sorry

end sin_identity_l263_263446


namespace joshua_final_bottle_caps_l263_263536

def initial_bottle_caps : ℕ := 150
def bought_bottle_caps : ℕ := 23
def given_away_bottle_caps : ℕ := 37

theorem joshua_final_bottle_caps : (initial_bottle_caps + bought_bottle_caps - given_away_bottle_caps) = 136 := by
  sorry

end joshua_final_bottle_caps_l263_263536


namespace books_left_over_after_repacking_l263_263498

def initial_boxes : ℕ := 1430
def books_per_initial_box : ℕ := 42
def weight_per_book : ℕ := 200 -- in grams
def books_per_new_box : ℕ := 45
def max_weight_per_new_box : ℕ := 9000 -- in grams (9 kg)

def total_books : ℕ := initial_boxes * books_per_initial_box

theorem books_left_over_after_repacking :
  total_books % books_per_new_box = 30 :=
by
  -- Proof goes here
  sorry

end books_left_over_after_repacking_l263_263498


namespace non_congruent_triangles_with_perimeter_11_l263_263130

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263130


namespace range_of_f_l263_263051

noncomputable def f (c a b : ℝ) : ℝ := (c - a) * (c - b)

theorem range_of_f {a b c : ℝ} (h1 : a + b = 1 - c) (h2 : 0 ≤ c) (h3 : 0 ≤ a) (h4 : 0 ≤ b) :
  (∀ y : ℝ, y ∈ (set.range (λ c, f c a b)) ↔ (-1/8) ≤ y ∧ y ≤ 1) :=
sorry

end range_of_f_l263_263051


namespace correct_option_l263_263830

-- Define the odd function property and the condition
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = -f(x)
def function_condition (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 → (x1 - x2) * (f(x1) - f(x2)) > 0

-- The hypothesis that f is an odd function and satisfies the given condition
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_cond : function_condition f)

-- Theorem that needs to be proved, which is the correct option in the provided solution.
theorem correct_option : f 4 < f (-6) :=
sorry

end correct_option_l263_263830


namespace johns_allowance_is_3_45_l263_263673

noncomputable def johns_weekly_allowance (A : ℝ) : Prop :=
  -- Condition 1: John spent 3/5 of his allowance at the arcade
  let spent_at_arcade := (3/5) * A
  -- Remaining allowance
  let remaining_after_arcade := A - spent_at_arcade
  -- Condition 2: He spent 1/3 of the remaining allowance at the toy store
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  -- Condition 3: He spent his last $0.92 at the candy store
  let spent_at_candy_store := 0.92
  -- Remaining amount after the candy store expenditure should be 0
  remaining_after_toy_store = spent_at_candy_store

theorem johns_allowance_is_3_45 : johns_weekly_allowance 3.45 :=
sorry

end johns_allowance_is_3_45_l263_263673


namespace rachel_arrangements_count_l263_263591

-- Definitions
inductive Color | white | red | blue
inductive Plant | basil | aloe 

open Color
open Plant

def lamps : Color → fin 2
| white := 2
| red := 2
| blue := 2

-- The set of all configurations
def configurations := 
  { f : Plant → Color | true }

-- The proof problem
theorem rachel_arrangements_count : 
  finset.card configurations = 21 := 
sorry

end rachel_arrangements_count_l263_263591


namespace trapezoid_AD_length_l263_263647

-- Definitions for the problem setup
variables {A B C D O P : Type}
variables (f : A → B → C → D → Prop)
variables (g : A → D → C → D → Prop)
variables (h : A → C → D → B → Prop)

-- The main theorem we want to prove
theorem trapezoid_AD_length
  (ABCD_trapezoid : f A B C D)
  (BC_CD_same : ∀ {x y}, (g B C x y → y = 43) ∧ (g B C x y → x = 43))
  (AD_perpendicular_BD : ∀ {x y}, h A D x y → ∃ (p : P), p = O)
  (O_intersection_AC_BD : g A C O B)
  (P_midpoint_BD : ∃ (p : P), p = P ∧ ∀ (x y : B ∗ D), y = x / 2)
  (OP_length : ∃ (len : ℝ), len = 11) :
  let m := 4 in let n := 190 in m + n = 194 := sorry

end trapezoid_AD_length_l263_263647


namespace corrected_mean_l263_263274

theorem corrected_mean (initial_mean : ℝ) (n : ℕ) (incorrect1 correct1 incorrect2 correct2 incorrect3 correct3 : ℝ) :
  initial_mean = 45 → 
  n = 100 →
  incorrect1 = 35 → correct1 = 60 →
  incorrect2 = 25 → correct2 = 52 →
  incorrect3 = 40 → correct3 = 85 →
  let initial_sum := initial_mean * n in
  let total_error := (correct1 - incorrect1) + (correct2 - incorrect2) + (correct3 - incorrect3) in
  let corrected_sum := initial_sum + total_error in
  let corrected_mean := corrected_sum / n in
  corrected_mean = 45.97 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  let initial_sum := initial_mean * n
  let total_error := (correct1 - incorrect1) + (correct2 - incorrect2) + (correct3 - incorrect3)
  let corrected_sum := initial_sum + total_error
  let corrected_mean := corrected_sum / n
  sorry

end corrected_mean_l263_263274


namespace non_congruent_triangles_with_perimeter_11_l263_263111

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263111


namespace tan_75_deg_l263_263375

noncomputable def tan (x : Real) : Real := sin x / cos x

theorem tan_75_deg : tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  -- define the necessary trigonometric values
  let tan_45 := tan (45 * Real.pi / 180)
  have h_tan_45 : tan_45 = 1 := by
    sorry

  let tan_30 := tan (30 * Real.pi / 180)
  have h_tan_30 : tan_30 = 1 / Real.sqrt 3 := by
    sorry

  -- angle addition formula for tangent
  have h_tan_add : tan_75_deg = (tan_45 + tan_30) / (1 - tan_45 * tan_30) := by
    rw [tan_add]
    sorry
    
  -- simplification to get the final result
  sorry

end tan_75_deg_l263_263375


namespace remainder_of_2n_divided_by_11_l263_263675

theorem remainder_of_2n_divided_by_11
  (n k : ℤ)
  (h : n = 22 * k + 12) :
  (2 * n) % 11 = 2 :=
by
  -- This is where the proof would go
  sorry

end remainder_of_2n_divided_by_11_l263_263675


namespace number_of_arrangements_l263_263502

open Nat

-- Define the set of people as a finite type with 5 elements.
inductive Person : Type
| youngest : Person
| eldest : Person
| p3 : Person
| p4 : Person
| p5 : Person

-- Define a function to count valid arrangements.
def countValidArrangements :
    ∀ (first_pos last_pos : Person), 
    (first_pos ≠ Person.youngest → last_pos ≠ Person.eldest → Fin 120) 
| first_pos, last_pos, h1, h2 => 
    let remaining := [Person.youngest, Person.eldest, Person.p3, Person.p4, Person.p5].erase first_pos |>.erase last_pos
    (factorial 3) * 4 * 3

-- Theorem statement to prove the number of valid arrangements.
theorem number_of_arrangements : 
  countValidArrangements Person.youngest Person.p5 sorry sorry = 72 :=
by 
  sorry

end number_of_arrangements_l263_263502


namespace sum_first_five_terms_geometric_seq_l263_263063

theorem sum_first_five_terms_geometric_seq : 
  (∀ n : ℕ, a (n + 1) = 2 * a n) ∧ (a 1 = 1) → (a 1 + a 2 + a 3 + a 4 + a 5 = 31) :=
by
  sorry

end sum_first_five_terms_geometric_seq_l263_263063


namespace similar_triangle_perimeter_l263_263345

theorem similar_triangle_perimeter (leg1 leg2 new_leg : ℕ) 
  (h_leg1 : leg1 = 6) 
  (h_leg2 : leg2 = 8) 
  (h_new_leg : new_leg = 18) 
  (h_right_triangle : leg1^2 + leg2^2 = (nat.sqrt (leg1^2 + leg2^2))^2) :
  ∃ new_perimeter: ℕ, new_perimeter = 72 := 
by
  sorry

end similar_triangle_perimeter_l263_263345


namespace domain_of_expression_l263_263404

theorem domain_of_expression (x : ℝ) :
  (∃ f : ℝ → ℝ, 
    f = λ x, (sqrt (x-3)) / (sqrt (7-x) * (x-1)) ↔ 
    (3 ≤ x ∧ x < 7)
  ) :=
sorry

end domain_of_expression_l263_263404


namespace arithmetic_sequence_problem_l263_263894

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) 
  (a1 : a 1 = 3) 
  (d : ℕ := 2) 
  (h : ∀ n, a n = a 1 + (n - 1) * d) 
  (h_25 : a n = 25) : 
  n = 12 := 
by
  sorry

end arithmetic_sequence_problem_l263_263894


namespace center_of_symmetry_l263_263013

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.tan (-7 * x + (Real.pi / 3))

theorem center_of_symmetry : f (Real.pi / 21) = 0 :=
by
  -- Mathematical proof goes here, skipping with sorry.
  sorry

end center_of_symmetry_l263_263013


namespace no_infinite_prime_sequence_l263_263903

theorem no_infinite_prime_sequence :
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Nat.Prime (p n)) ∧ (∀ n, | p (n + 1) - 2 * p n | = 1) ∧ (∀ n, p n < p (n + 1)) :=
  sorry

end no_infinite_prime_sequence_l263_263903


namespace quadrilaterals_property_A_false_l263_263998

theorem quadrilaterals_property_A_false (Q A : Type → Prop) 
  (h : ¬ ∃ x, Q x ∧ A x) : ¬ ∀ x, Q x → A x :=
by
  sorry

end quadrilaterals_property_A_false_l263_263998


namespace find_lambda_l263_263844

noncomputable def vector_parallel (a b : ℝ × ℝ) : Prop :=
∃ (k : ℝ), (b.1 = k * a.1) ∧ (b.2 = k * a.2)

theorem find_lambda
  (a b c : ℝ × ℝ)
  (λ : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (1, 0))
  (h_c : c = (3, 4))
  (parallel_condition : vector_parallel (a.1 + λ * b.1, a.2 + λ * b.2) c) :
  λ =  1 / 2 :=
  sorry

end find_lambda_l263_263844


namespace suff_but_not_necessary_of_parallel_l263_263148

variable {α : Type} {l m : α → α → Prop}

-- Definition of being parallel to a plane
def parallel_to_plane (line : α → α → Prop) (plane : set (α → α → Prop)) : Prop :=
∀ p, plane p → ∀ x y, line x y → (¬ p x y ∧ ¬ p y x)

-- Statement of the problem
theorem suff_but_not_necessary_of_parallel 
  {a : set (α → α → Prop)}
  (h1 : a ≠ ∅)
  (h2 : parallel_to_plane m a)
  (l ≠ m)
  : (∀ x y, l x y → m x y) → (∀ x y, l x y → ∀ p, a p → p x y) :=
by
  sorry

end suff_but_not_necessary_of_parallel_l263_263148


namespace square_area_diagonal_100_l263_263264

-- We define the side length and area for a square where the diagonal is given as 100.
theorem square_area_diagonal_100 (s : ℝ) (A : ℝ) (h1 : 100 = s * real.sqrt 2) (h2 : A = s * s) : A = 5000 :=
sorry

end square_area_diagonal_100_l263_263264


namespace maximum_value_of_product_l263_263931

open Real

-- Definition for the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
def conditions : Prop :=
∀ i, (1 ≤ i ∧ i < 2016) → 9 * a i > 11 * (a (i + 1))^2

-- The statement to be proved
theorem maximum_value_of_product (h : conditions a) :
  ∃ (P : ℝ), P = (a 1 - (a 2)^2) * 
               (a 2 - (a 3)^2) * 
               ... * 
               (a 2015 - (a 2016)^2) * 
               (a 2016 - (a 1)^2) ∧ 
                 P ≤ (1 / 4)^2016 ∧ 
                 (∃ x, (∀ i, 1 ≤ i ∧ i ≤ 2016 → a i = x) ∧ 
                      x = 1 / 2 → P = (1 / 4)^2016) := 
sorry

end maximum_value_of_product_l263_263931


namespace count_real_root_quadratics_l263_263008

theorem count_real_root_quadratics : 
  (Finset.univ.filter (λ (b : ℕ × ℕ), b.1^2 - 4 * b.2 ≥ 0 ∧ b.1 ∈ {1, 2, 3, 4, 5, 6} ∧ b.2 ∈ {1, 2, 3, 4, 5, 6})).card = 19 := 
by
  sorry

end count_real_root_quadratics_l263_263008


namespace calc_dz_calc_d2z_calc_d3z_l263_263028

variables (x y dx dy : ℝ)

def z : ℝ := x^5 * y^3

-- Define the first differential dz
def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

-- Define the second differential d2z
def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

-- Define the third differential d3z
def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem calc_dz : (dz x y dx dy) = (5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) := 
by sorry

theorem calc_d2z : (d2z x y dx dy) = (20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) :=
by sorry

theorem calc_d3z : (d3z x y dx dy) = (60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end calc_dz_calc_d2z_calc_d3z_l263_263028


namespace concurrency_of_excircle_tangent_lines_l263_263526

theorem concurrency_of_excircle_tangent_lines
  (A B C D E F : Type)
  [triangle ABC : Triangle A B C]
  (tangent_points : excircle_tangent_points_triangle ABC D E F) :
  concurrent_lines (line_through_points A D) (line_through_points B E) (line_through_points C F) := by
  sorry

end concurrency_of_excircle_tangent_lines_l263_263526


namespace rate_percent_l263_263302

theorem rate_percent (SI P T: ℝ) (h₁: SI = 250) (h₂: P = 1500) (h₃: T = 5) : 
  ∃ R : ℝ, R = (SI * 100) / (P * T) := 
by
  use (250 * 100) / (1500 * 5)
  sorry

end rate_percent_l263_263302


namespace non_congruent_triangles_with_perimeter_11_l263_263136

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263136


namespace min_value_frac_l263_263520

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l263_263520


namespace base_h_equation_l263_263750

theorem base_h_equation (h : ℕ) :
  (3684_h + 4175_h = 1029_h) ↔ h = 9 :=
by
  sorry

end base_h_equation_l263_263750


namespace a_gt_b_l263_263916

theorem a_gt_b (x : ℝ) (hx : x < 0) : let a := log 2 + log 5 in let b := exp x in a > b :=
by
  let a := log 2 + log 5
  let b := exp x
  sorry

end a_gt_b_l263_263916


namespace rosa_initial_flowers_l263_263594

-- Definitions derived from conditions
def initial_flowers (total_flowers : ℕ) (given_flowers : ℕ) : ℕ :=
  total_flowers - given_flowers

-- The theorem stating the proof problem
theorem rosa_initial_flowers : initial_flowers 90 23 = 67 :=
by
  -- The proof goes here
  sorry

end rosa_initial_flowers_l263_263594


namespace log_identity_l263_263448

theorem log_identity
  (x : ℝ)
  (h1 : x < 1)
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^4) / Real.log 10 = 100) :
  (Real.log x / Real.log 10)^3 - Real.log (x^5) / Real.log 10 = -114 + Real.sqrt 104 := 
by
  sorry

end log_identity_l263_263448


namespace scenic_spots_arrangement_l263_263860

def arrangements : ℕ :=
  let C (n k : ℕ) := Nat.choose n k
  (C 5 3 * C (5 - 3) 1 * C (5 - 3 - 1) 1 + C 5 2 * C (5 - 2) 2 * C (5 - 2 - 2) 1) * 6

theorem scenic_spots_arrangement :
  arrangements = 150 :=
by
  sorry

end scenic_spots_arrangement_l263_263860


namespace median_length_is_sqrt2_l263_263884

noncomputable def length_of_median 
  (A B C : Type)
  [InnerProductSpace ℝ A] 
  [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] 
  (AB : ℝ) 
  (tanA tanB tanC : ℝ) 
  (h1 : AB = 2) 
  (h2 : (1 / tanA) + (1 / tanB) = 4 / tanC) 
  (median_length : ℝ) : Prop := 
  median_length = Real.sqrt(2)

theorem median_length_is_sqrt2 
  {A B C : Type} 
  [InnerProductSpace ℝ A] 
  [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] 
  {AB : ℝ} 
  {tanA tanB tanC : ℝ} 
  (h1 : AB = 2) 
  (h2 : (1 / tanA) + (1 / tanB) = 4 / tanC) : 
  length_of_median A B C AB tanA tanB tanC h1 h2 (Real.sqrt 2) :=
by 
  sorry

end median_length_is_sqrt2_l263_263884


namespace sum_of_solutions_l263_263926

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 
    x / 3 - 1
  else 
    -2 * x + 5

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | f x = 2}.to_finset, id x) = 3 / 2 := 
  sorry

end sum_of_solutions_l263_263926


namespace max_marks_set_l263_263350

theorem max_marks_set (M : ℝ) 
  (condition1 : 0.75 * M = 380) : M = 507 :=
by 
  have h1 : M = 380 / 0.75, from eq_div_of_mul_eq 380 0.75 condition1,
  have h2 : 380 / 0.75 = 506.67, from calc 380 / 0.75 = 506.67 : by norm_num,
  have h3 : 506.67 = 507, sorry,
  have h4 : M = 507, from eq.trans h1 (eq.trans h2 h3),
  exact h4

end max_marks_set_l263_263350


namespace num_correct_statements_is_2_l263_263218

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms
def sum_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := sum_terms n + a (n + 1)

-- Given conditions
variables (h1 : sum_terms a 5 < sum_terms a 6)
variables (h2 : sum_terms a 6 = sum_terms a 7 ∧ sum_terms a 7 > sum_terms a 8)

-- Statements about the sequence
def statements (a : ℕ → ℝ) (S : ℕ → ℝ) : list Prop :=
  [∀ n, a (n + 1) < a n,  -- Statement 1: {a_n} is a decreasing sequence
   a 7 = 0,              -- Statement 2: a_7 = 0
   sum_terms a 9 > sum_terms a 5,  -- Statement 3: S_9 > S_5
   ∀ n, sum_terms a n <= sum_terms a 6] -- Statement 4: S_6 and S_7 are maximum values of S_n

-- Proof problem: Number of correct statements is 2
theorem num_correct_statements_is_2 : (statements a S).count (λ p, p) = 2 :=
sorry

end num_correct_statements_is_2_l263_263218


namespace problem_statement_l263_263270

variable {f : ℝ → ℝ}

-- Assume the conditions provided in the problem statement.
def continuous_on_ℝ (f : ℝ → ℝ) : Prop := Continuous f
def condition_x_f_prime (f : ℝ → ℝ) (h : ℝ → ℝ) : Prop := ∀ x : ℝ, x * h x < 0

-- The main theorem statement based on the conditions and the correct answer.
theorem problem_statement (hf : continuous_on_ℝ f) (hf' : ∀ x : ℝ, x * (deriv f x) < 0) :
  f (-1) + f 1 < 2 * f 0 :=
sorry

end problem_statement_l263_263270


namespace diagonal_length_count_l263_263386

theorem diagonal_length_count :
  ∃ (x : ℕ) (h : (3 < x ∧ x < 22)), x = 18 := by
    sorry

end diagonal_length_count_l263_263386


namespace city_population_l263_263686

theorem city_population (P: ℝ) (h: 0.85 * P = 85000) : P = 100000 := 
by
  sorry

end city_population_l263_263686


namespace gcf_2550_7140_l263_263300

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_2550_7140 : gcf 2550 7140 = 510 := 
  by 
    sorry

end gcf_2550_7140_l263_263300


namespace moles_of_H2O_formed_l263_263032

def balanced_reaction (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  KOH = KI ∧ NH4I = NH3 ∧ KOH = H2O

theorem moles_of_H2O_formed (KOH NH4I : ℕ) (h : balanced_reaction KOH NH4I KOH NH4I KOH) :
  KOH = KOH :=
  by sorry

example : moles_of_H2O_formed 3 3 (by {simp [balanced_reaction], split; refl}) = 3 :=
  by exact rfl

end moles_of_H2O_formed_l263_263032


namespace square_of_second_arm_l263_263271

theorem square_of_second_arm (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
sorry

end square_of_second_arm_l263_263271


namespace length_of_AK_is_8_64_l263_263511

noncomputable def problem_statement : ℝ := 
  let BC := 25
  let BD := 20
  let BE := 7
  let CE := 24 -- calculated using the Pythagorean theorem in the solution (valid)
  let CD := BC - BD -- given in the problem BC = 25, BD = 20
  let DE := 30 -- diameter of the circle passing through D and E
  let AD := sqrt ((BD^2) + (CD^2))
  let AE := sqrt ((BE^2) + (CE^2))
  let AF := AE / 2
  let AP := CE -- AP is the extension of AH which is perpendicular to BC
  let AK := (AF * AP) / BC
  AK

theorem length_of_AK_is_8_64 : problem_statement = 8.64 := by
  sorry

end length_of_AK_is_8_64_l263_263511


namespace lana_needs_to_sell_more_muffins_l263_263540

/--
Lana aims to sell 20 muffins at the bake sale.
She sells 12 muffins in the morning.
She sells another 4 in the afternoon.
How many more muffins does Lana need to sell to hit her goal?
-/
theorem lana_needs_to_sell_more_muffins (goal morningSales afternoonSales : ℕ)
  (h_goal : goal = 20) (h_morning : morningSales = 12) (h_afternoon : afternoonSales = 4) :
  goal - (morningSales + afternoonSales) = 4 :=
by
  sorry

end lana_needs_to_sell_more_muffins_l263_263540


namespace roots_conjugates_a_b_zero_l263_263915

theorem roots_conjugates_a_b_zero (a b : ℝ) (hz : ∀ z : ℂ, z^2 + (6 + a * complex.I) * z + (15 + b * complex.I) = 0 → z.im = 0) :
  (a, b) = (0, 0) := 
by
  sorry

end roots_conjugates_a_b_zero_l263_263915


namespace digit_in_105th_place_of_7_over_26_l263_263143

theorem digit_in_105th_place_of_7_over_26 :
  let repeating_seq := "269230769"
  let repeat_length := 9
  let position := 105 % repeat_length
  (position = 3) → (repeating_seq.nth (position - 1) = '9') :=
by
  let repeating_seq := "269230769"
  let repeat_length := 9
  let position := 105 % repeat_length
  have h1 : position = 3 := by sorry
  have h2 : repeating_seq.nth (position - 1) = '9' := by sorry
  exact h2

end digit_in_105th_place_of_7_over_26_l263_263143


namespace medium_stores_to_select_l263_263878

-- Definitions based on conditions in a)
def total_stores := 1500
def ratio_large := 1
def ratio_medium := 5
def ratio_small := 9
def sample_size := 30
def medium_proportion := ratio_medium / (ratio_large + ratio_medium + ratio_small)

-- Main theorem to prove
theorem medium_stores_to_select : (sample_size * medium_proportion) = 10 :=
by sorry

end medium_stores_to_select_l263_263878


namespace shaded_region_area_correct_l263_263379

-- Definitions based on the problem conditions
structure RightAngledTriangle (A B C : Type) :=
  (AB BC : ℝ)
  (right_angle_at : Bool)

-- Geometric properties based on the problem statement
def triangle_ABC : RightAngledTriangle ℝ ℝ ℝ :=
  { AB := 10, BC := 7, right_angle_at := true }

def triangle_DEF : RightAngledTriangle ℝ ℝ ℝ :=
  { AB := 3, BC := 4, right_angle_at := true }

-- The combined structure representing the arrangement conditions
structure Arrangement :=
  (tri_ABC tri_DEF : RightAngledTriangle ℝ ℝ ℝ)
  (BC_DE_coincident : Bool)

def specific_arrangement : Arrangement :=
  { tri_ABC := triangle_ABC, tri_DEF := triangle_DEF, BC_DE_coincident := true }

-- Theorem: Proof problem with specified correct answer
theorem shaded_region_area_correct :
  ∀ (arr : Arrangement), arr = specific_arrangement → (35 - 6 = 29) :=
by
  intro arr
  intro h
  rw [h]
  exact eq.refl 29

end shaded_region_area_correct_l263_263379


namespace fractional_parts_sum_l263_263742

def fractional_part (x : ℚ) : ℚ := x - x.to_nat

theorem fractional_parts_sum : 
  fractional_part (2015 / 3) + fractional_part (315 / 4) + fractional_part (412 / 5) = 1.817 :=
by
  sorry

end fractional_parts_sum_l263_263742


namespace find_b_l263_263141

theorem find_b (x y z a b : ℝ) (h1 : x + y = 2) (h2 : xy - z^2 = a) (h3 : b = x + y + z) : b = 2 :=
by
  sorry

end find_b_l263_263141


namespace perfect_shuffle_restore_order_l263_263293

theorem perfect_shuffle_restore_order (n : ℕ) (hn : Nat.Prime (2 * n + 1)) :
  let original_order := List.range (2 * n + 1) in
  let shuffled_order := (original_order.filter (λ x, x % 2 = 0)) ++ 
                        (original_order.filter (λ x, x % 2 = 1)) in
  shuffled_order = original_order :=
sorry

end perfect_shuffle_restore_order_l263_263293


namespace magic_triangle_max_sum_l263_263497

/-- In a magic triangle, each of the six consecutive whole numbers 11 to 16 is placed in one of the circles. 
    The sum, S, of the three numbers on each side of the triangle is the same. One of the sides must contain 
    three consecutive numbers. Prove that the largest possible value for S is 41. -/
theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), 
  (a = 11 ∨ a = 12 ∨ a = 13 ∨ a = 14 ∨ a = 15 ∨ a = 16) ∧
  (b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16) ∧
  (c = 11 ∨ c = 12 ∨ c = 13 ∨ c = 14 ∨ c = 15 ∨ c = 16) ∧
  (d = 11 ∨ d = 12 ∨ d = 13 ∨ d = 14 ∨ d = 15 ∨ d = 16) ∧
  (e = 11 ∨ e = 12 ∨ e = 13 ∨ e = 14 ∨ e = 15 ∨ e = 16) ∧
  (f = 11 ∨ f = 12 ∨ f = 13 ∨ f = 14 ∨ f = 15 ∨ f = 16) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  (a + b + c = S) ∧ (c + d + e = S) ∧ (e + f + a = S) ∧
  (∃ k, a = k ∧ b = k+1 ∧ c = k+2 ∨ b = k ∧ c = k+1 ∧ d = k+2 ∨ c = k ∧ d = k+1 ∧ e = k+2 ∨ d = k ∧ e = k+1 ∧ f = k+2) →
  S = 41 :=
by
  sorry

end magic_triangle_max_sum_l263_263497


namespace citrus_yield_recovery_probability_l263_263024

theorem citrus_yield_recovery_probability :
  let p_first_year := [(1.0, 0.2), (0.9, 0.4), (0.8, 0.4)] in
  let p_second_year := [(1.5, 0.3), (1.25, 0.3), (1.0, 0.4)] in
  ((p_first_year[0].2 * p_second_year[2].2) + (p_first_year[1].2 * p_second_year[1].2) = 0.2) := 
by
  simp [p_first_year, p_second_year]
  sorry

end citrus_yield_recovery_probability_l263_263024


namespace truffles_more_than_caramels_l263_263373

-- Define the conditions
def chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def peanut_clusters := (64 * chocolates) / 100
def truffles := chocolates - (caramels + nougats + peanut_clusters)

-- Define the claim
theorem truffles_more_than_caramels : (truffles - caramels) = 6 := by
  sorry

end truffles_more_than_caramels_l263_263373


namespace tan_value_l263_263083

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (a_geom : ∀ m n : ℕ, a m / a n = a (m - n))
variable (b_arith : ∃ c d : ℝ, ∀ n : ℕ, b n = c + n * d)
variable (ha : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
variable (hb : b 1 + b 6 + b 11 = 7 * Real.pi)

theorem tan_value : Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end tan_value_l263_263083


namespace count_3_digit_product_36_is_21_l263_263474

-- Define the product of digits function
def digits_product (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 * d2 * d3

-- Define a predicate to check if a number is a 3-digit positive integer
def is_3_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the count of 3-digit integers with digits product equal to 36
def count_3_digit_with_product_36 : ℕ :=
  (Finset.range 1000).filter (λ n => is_3_digit n ∧ digits_product n = 36).card

theorem count_3_digit_product_36_is_21 :
  count_3_digit_with_product_36 = 21 :=
sorry

end count_3_digit_product_36_is_21_l263_263474


namespace candidates_scoring_between_100_and_120_admission_score_cutoff_l263_263480

noncomputable def normal_distribution (μ σ : ℝ) := sorry

constant P : ℝ → ℝ → ℝ → ℝ

theorem candidates_scoring_between_100_and_120 (X : ℝ → ℝ) (μ σ : ℝ) (total_candidates : ℕ)
  (h_dist : X = normal_distribution μ σ)
  (h_X : X = normal_distribution 90 100)
  (h1 : P 80 100 X = 0.6826)
  (h2 : P 60 120 X = 0.9974) :
  (0.6826 * 5000 ≈ 3413) :=
  sorry

theorem admission_score_cutoff (X : ℝ → ℝ) (μ σ : ℝ) (total_candidates top_candidates : ℕ)
  (h_dist : X = normal_distribution μ σ)
  (h_X : X = normal_distribution 90 100)
  (h3 : top_candidates = 114)
  (h4 : total_candidates = 5000) :
  (cutoff_score ≈ 290) :=
  sorry

end candidates_scoring_between_100_and_120_admission_score_cutoff_l263_263480


namespace smallest_x_l263_263863

theorem smallest_x (x y : ℕ) (h_pos: x > 0 ∧ y > 0) (h_eq: 8 / 10 = y / (186 + x)) : x = 4 :=
sorry

end smallest_x_l263_263863


namespace walkway_area_l263_263534

theorem walkway_area (flower_bed_width flower_bed_height : ℕ) (num_beds_per_row num_rows : ℕ) 
  (walkway_width : ℕ) : 
  (num_beds_per_row = 3) → (num_rows = 3) → (flower_bed_width = 5) → (flower_bed_height = 3) → (walkway_width = 2) → 
  (let total_width := num_beds_per_row * flower_bed_width + (num_beds_per_row + 1) * walkway_width,
       total_height := num_rows * flower_bed_height + (num_rows + 1) * walkway_width,
       total_garden_area := total_width * total_height,
       total_flower_bed_area := num_beds_per_row * num_rows * flower_bed_width * flower_bed_height,
       walkway_area := total_garden_area - total_flower_bed_area
  in walkway_area = 256) :=
by
  intros h1 h2 h3 h4 h5
  let total_width := 3 * 5 + (3 + 1) * 2
  let total_height := 3 * 3 + (3 + 1) * 2
  let total_garden_area := total_width * total_height
  let total_flower_bed_area := 3 * 3 * 5 * 3
  let walkway_area := total_garden_area - total_flower_bed_area
  have : walkway_area = 256 := by
    simp only [*, Nat.mul_add, Nat.add_mul, Nat.add_assoc, Nat.add_mul, Nat.mul_assoc]
  exact this

end walkway_area_l263_263534


namespace hyperbola_asymptote_l263_263979

theorem hyperbola_asymptote :
  (∀ x y : ℝ, (x^2 / 2 - y^2 = 1) → (y = ± (Real.sqrt 2 / 2) * x)) :=
sorry

end hyperbola_asymptote_l263_263979


namespace competition_score_l263_263169

theorem competition_score (x : ℕ) (h : x ≥ 15) : 10 * x - 5 * (20 - x) > 120 := by
  sorry

end competition_score_l263_263169


namespace remainder_mod_500_l263_263777

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l263_263777


namespace problem_l263_263838

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x : ℝ, f x a ≤ 2) : f (π / 6) a = -1 :=
by {
  sorry
}

end problem_l263_263838


namespace lizette_has_813_stamps_l263_263934

def minervas_stamps : ℕ := 688
def additional_stamps : ℕ := 125
def lizettes_stamps : ℕ := minervas_stamps + additional_stamps

theorem lizette_has_813_stamps : lizettes_stamps = 813 := by
  sorry

end lizette_has_813_stamps_l263_263934


namespace building_total_floors_l263_263399

def earl_final_floor (start : ℕ) : ℕ :=
  start + 5 - 2 + 7

theorem building_total_floors (start : ℕ) (current : ℕ) (remaining : ℕ) (total : ℕ) :
  earl_final_floor start = current →
  remaining = 9 →
  total = current + remaining →
  start = 1 →
  total = 20 := by
sorry

end building_total_floors_l263_263399


namespace problem_statement_l263_263766

noncomputable def numberOfFunctions (a k : ℝ) : ℕ :=
if k > 1 then 0 else 2

theorem problem_statement (f : ℝ → ℝ) (a k : ℝ) (h : k ≥ 0) :
  (∀ x y z : ℝ, f(xy + a) + f(xz + a) - f(x + a) * f(yz + a) ≥ k) ↔
  (k ≤ 1 → (f = (λ _, 1 + √(1 - k)) ∨ f = (λ _, 1 - √(1 - k)))
   ∧ k > 1 → f = (λ _, 0)) :=
by sorry

end problem_statement_l263_263766


namespace coeff_x18_x17_l263_263015

theorem coeff_x18_x17 (f : ℕ → ℕ) :
  f 18 = 0 ∧ f 17 = 3420 :=
by
  let f := λ n : ℕ, (Finset.natAntidiagonal 20).sum (λ p, if p.1 * 5 + p.2 * 7 = n then (Nat.choose 20 p.1) * (Nat.choose (20 - p.1) p.2) else 0)
  split
  {
    exact finset.sum_eq_zero (λ p hp, by {
      dsimp [f],
      cases p with a b,
      simp only [if_neg] at hp,
      have := finset.natAntidiagonal.mem_iff,
      rw this at hp,
      rintro rfl,
      exact hp.2 rfl })
  }
  {
    have : ∑ p in finset.natAntidiagonal 20, ite (p.1 * 5 + p.2 * 7 = 17) (nat.choose 20 p.1 * nat.choose (20 - p.1) p.2) 0 = 3420,
    { exact sorry },
    exact this
  }

end coeff_x18_x17_l263_263015


namespace non_congruent_triangles_with_perimeter_11_l263_263128

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263128


namespace oldest_child_age_l263_263265

theorem oldest_child_age (ages : Fin 7 → ℕ) (h_unique : Function.Injective ages)
  (h_avg : (∑ i, ages i) = 56) (h_diff : ∀ i : Fin 6, ages ⟨i + 1, Fin.is_lt_succ i⟩ = ages ⟨i, Fin.is_lt i⟩ + 1) :
  ages (Fin.last 7) = 11 :=
sorry

end oldest_child_age_l263_263265


namespace max_water_bottles_one_athlete_l263_263707

-- Define variables and key conditions
variable (total_bottles : Nat := 40)
variable (total_athletes : Nat := 25)
variable (at_least_one : ∀ i, i < total_athletes → Nat.succ i ≥ 1)

-- Define the problem as a theorem
theorem max_water_bottles_one_athlete (h_distribution : total_bottles = 40) :
  ∃ max_bottles, max_bottles = 16 :=
by
  sorry

end max_water_bottles_one_athlete_l263_263707


namespace fg_at_2_l263_263483

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2*x + 5

theorem fg_at_2 : f (g 2) = 729 := by
  sorry

end fg_at_2_l263_263483


namespace min_value_frac_l263_263519

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l263_263519


namespace jim_statue_original_cost_l263_263209

def original_cost (SP : ℝ) (profit_percentage : ℝ) : ℝ :=
  SP / (1 + profit_percentage)

theorem jim_statue_original_cost 
  (SP : ℝ) (profit_percentage : ℝ) (h_SP : SP = 670) (h_profit : profit_percentage = 0.25) :
  original_cost SP profit_percentage = 536 :=
by
  rw [h_SP, h_profit]
  -- This would leave the proof open, as one might need to show the actual calculation, but the important part is setting up the context correctly.
  sorry

end jim_statue_original_cost_l263_263209


namespace intersection_proportion_l263_263588

-- Definitions of the points and lines on the circle
variables {A B C D M : Type} [Incircle : C]

-- Lines AB and CD intersect at point M
variables (hM : M ∈ line A B) (hM' : M ∈ line C D)

-- The desired equality to be proven
theorem intersection_proportion
  (A B C D : Point)
  (M : Point)
  (hABC : collinear A B C)
  (hCDA : collinear C D A)
  (h_intersect : M ∈ (line A B) ∩ (line C D)) :
  (dist A C * dist A D) / dist A M = (dist B C * dist B D) / dist B M :=
sorry

end intersection_proportion_l263_263588


namespace problem_statement_l263_263371

noncomputable def solve_problem : ℝ :=
  let term1 := ((sqrt 2) - 1) ^ 0
  let term2 := (-27 : ℝ)^(1/3 : ℝ) -- Cube root can be written as taking the power of 1/3
  term1 + term2

theorem problem_statement : solve_problem = -2 := by
  sorry

end problem_statement_l263_263371


namespace pizza_slices_left_l263_263384

theorem pizza_slices_left (initial_slices : ℕ) (ate_by_dean : ℕ) (ate_by_frank : ℕ) (ate_by_sammy : ℕ) :
  initial_slices = 24 →
  ate_by_dean = 6 →
  ate_by_frank = 3 →
  ate_by_sammy = 4 →
  initial_slices - (ate_by_dean + ate_by_frank + ate_by_sammy) = 11 :=
by
  intros h_initial h_dean h_frank h_sammy
  rw [h_initial, h_dean, h_frank, h_sammy]
  norm_num
  exact sorry

end pizza_slices_left_l263_263384


namespace triangle_count_with_perimeter_11_l263_263101

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263101


namespace circumcenter_dot_product_proof_l263_263438

variables {A B C O : Type*}
variables [inner_product_space ℝ (A → ℝ)]
variables (B A C O : A → ℝ)
variables (BA BC AC BO : ℝ)

noncomputable def circumcenter_dot_product : Prop :=
  let BO_dot_AC := (BO : ℝ) in
  O = circumcenter A B C ∧
  (BA = 2 ∧ BC = 6) → 
  BO_dot_AC = 16

theorem circumcenter_dot_product_proof :
  circumcenter_dot_product A B C O BA BC AC BO :=
sorry

end circumcenter_dot_product_proof_l263_263438


namespace shortest_distance_segment_AB_l263_263584

theorem shortest_distance_segment_AB :
  ∃ (A B : ℝ × ℝ), A.2 = (8 / 15) * A.1 - 10 ∧ B.2 = B.1 ^ 2 ∧ 
  (∀ (C D : ℝ × ℝ), C.2 = (8 / 15) * C.1 - 10 ∧ D.2 = D.1 ^ 2 →
    dist C D ≥ dist A B) ∧ 
  dist A B = 2234 / 255 :=
begin
  sorry,
end

end shortest_distance_segment_AB_l263_263584


namespace hyperbola_eq_l263_263868

theorem hyperbola_eq (P : ℝ × ℝ) (asymptote : ℝ → ℝ) (hyperbola_eq : ℝ → ℝ → Prop) :
  P = (6, real.sqrt 3) →
  asymptote = (λ x : ℝ, ± (x / 3)) →
  (∀ x y : ℝ, hyperbola_eq x y ↔ ((x^2 / 9) - (y^2 / 1) = 1)) :=
by
  intros hP hA
  sorry

end hyperbola_eq_l263_263868


namespace sin_identity_l263_263445

theorem sin_identity (α : ℝ) (h : Real.sin (π/4 + α) = √3 / 2) : Real.sin (3*π/4 - α) = √3 / 2 :=
  sorry

end sin_identity_l263_263445


namespace final_height_of_tree_in_4_months_l263_263233

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end final_height_of_tree_in_4_months_l263_263233


namespace three_digit_integers_S_equal_S_plus_1_l263_263735

def S (n : ℕ) : ℕ :=
  n % 4 + n % 5 + n % 6 + n % 7 + n % 8

def Δ (n k : ℕ) : ℤ :=
  if n % k = k - 1 then -(k - 1) else 1

theorem three_digit_integers_S_equal_S_plus_1 :
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 }.card
  ( { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ (S(n) = S(n + 1)) }.card = 2 ) :=
sorry

end three_digit_integers_S_equal_S_plus_1_l263_263735


namespace fixed_point_PQ_l263_263094

open Real

-- Definitions of the points
def A : ℝ × ℝ := (-sqrt 5, 0)
def B : ℝ × ℝ := (sqrt 5, 0)
def M : ℝ × ℝ := (2, 0)

-- Condition that the incenter lies on x = 2
def incenter_on_x_eq_2 (C : ℝ × ℝ) : Prop :=
  C.1 = 2

-- Condition that vectors MP and MQ are orthogonal
def orthogonality_condition (P Q : ℝ × ℝ) : Prop :=
  let MP := (P.1 - M.1, P.2 - M.2) in
  let MQ := (Q.1 - M.1, Q.2 - M.2) in
  MP.1 * MQ.1 + MP.2 * MQ.2 = 0

-- The proof problem
theorem fixed_point_PQ (P Q : ℝ × ℝ) (C : ℝ × ℝ) :
  incenter_on_x_eq_2 C →
  orthogonality_condition P Q →
  ∃ (F : ℝ × ℝ), F = (10 / 3, 0) ∧
    (line_through P Q) F := sorry

end fixed_point_PQ_l263_263094


namespace fixed_line_of_midpoint_l263_263087

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end fixed_line_of_midpoint_l263_263087


namespace floor_log10_sum_equals_92_l263_263553

-- Definitions and conditions
def floor_log10 (x : ℝ) : ℤ := Int.floor (Real.log10 x)

-- Theorem stating that the sum of the greatest integer less than or equal to the base-10 logarithm from 1 to 100 equals 92
theorem floor_log10_sum_equals_92 : 
  (Finset.range 100).sum (λ n, floor_log10 (n + 1)) = 92 := 
by 
  sorry

end floor_log10_sum_equals_92_l263_263553


namespace circle_equation_proof_l263_263450

noncomputable def circle_equation (a r : ℝ) (x y : ℝ) := (x - a)^2 + y^2 = r^2

theorem circle_equation_proof (a r : ℝ) (h_a_pos : a > 0) (M : ℝ × ℝ) (line_dist : ℝ)
  (h_M : M = (0, real.sqrt 5))
  (h_dist : line_dist = abs (2 * a) / real.sqrt (2^2 + (-1)^2))
  (h_line_dist_val : line_dist = (4 * real.sqrt 5) / 5)
  (eq_a : a = 2)
  (eq_r : r = 3) :
  circle_equation a r = (λ x y, (x - 2)^2 + y^2 = 9) :=
  sorry

end circle_equation_proof_l263_263450


namespace constant_function_derivative_zero_l263_263981

-- Define the function and conditions
theorem constant_function_derivative_zero (f : ℝ → ℝ) (a b : ℝ) (h : a ≤ b)
  (h1 : ∃ M m, (∀ x ∈ set.Icc a b, f x ≤ M ∧ f x ≥ m) ∧ M = m) : ∀ x ∈ set.Icc a b, deriv f x = 0 :=
by
  sorry

end constant_function_derivative_zero_l263_263981


namespace average_age_new_students_l263_263968

theorem average_age_new_students (A : ℚ)
    (avg_original_age : ℚ := 48)
    (num_new_students : ℚ := 120)
    (new_avg_age : ℚ := 44)
    (total_students : ℚ := 160) :
    let num_original_students := total_students - num_new_students
    let total_age_original := num_original_students * avg_original_age
    let total_age_all := total_students * new_avg_age
    total_age_original + (num_new_students * A) = total_age_all → A = 42.67 := 
by
  intros
  sorry

end average_age_new_students_l263_263968


namespace milk_leftover_l263_263366

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l263_263366


namespace solve_floor_equation_l263_263603

theorem solve_floor_equation (x : ℚ) :
  (∃ m : ℤ, (m : ℚ) = floor ((9 * x - 4) / 6) ∧ (12 * x + 7) / 4 = m) ↔ 
  (x = -9/4 ∨ x = -23/12) :=
begin
  sorry
end

end solve_floor_equation_l263_263603


namespace track_completion_time_l263_263292

variable (r the_total_time: ℕ)
variable (runner_meet1_meet2_time runner_meet2_meet3_time runner_meet3_meet1_time: ℕ)

-- Conditions:
-- 1. Three runners with variable r moving along a circular track at equal constant speeds.
-- 2. When any two runners meet, they instantly turn around and start running in the opposite direction.
axiom (meet1_meet2_time : runner_meet1_meet2_time = 20)
axiom (meet2_meet3_time : runner_meet2_meet3_time = 30)
axiom (total_time_eq_2a_2b : the_total_time = 2 * runner_meet1_meet2_time + 2 * runner_meet2_meet3_time)

-- Goal: To prove that the total time for one runner to complete the entire track is 100 minutes.
theorem track_completion_time : the_total_time = 100 :=
    by
    rw [total_time_eq_2a_2b, meet1_meet2_time, meet2_meet3_time]
    sorry

end track_completion_time_l263_263292


namespace addition_neg3_plus_2_multiplication_neg3_times_2_l263_263372

theorem addition_neg3_plus_2 : -3 + 2 = -1 :=
  by
    sorry

theorem multiplication_neg3_times_2 : (-3) * 2 = -6 :=
  by
    sorry

end addition_neg3_plus_2_multiplication_neg3_times_2_l263_263372


namespace non_congruent_triangles_with_perimeter_11_l263_263122

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263122


namespace tan_ratio_triangle_area_l263_263517

theorem tan_ratio (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A) :
  Real.tan A / Real.tan B = -4 := by
  sorry

theorem triangle_area (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A)
  (h2 : c = 2) (h3 : Real.tan C = 3 / 4) :
  ∃ S : ℝ, S = 1 / 2 * b * c * Real.sin A ∧ S = 4 / 3 := by
  sorry

end tan_ratio_triangle_area_l263_263517


namespace odd_multiple_of_9_implies_multiple_of_3_l263_263679

theorem odd_multiple_of_9_implies_multiple_of_3 :
  ∀ (S : ℤ), (∀ (n : ℤ), 9 * n = S → ∃ (m : ℤ), 3 * m = S) ∧ (S % 2 ≠ 0) → (∃ (m : ℤ), 3 * m = S) :=
by
  sorry

end odd_multiple_of_9_implies_multiple_of_3_l263_263679


namespace maximum_area_of_triangle_l263_263433

variable {A B C : ℝ}  -- the angles of the triangle
variable {a b c : ℝ}  -- the sides opposite the angles A, B, C respectively

-- Define vectors based on the angles
def m : ℝ × ℝ := (Real.cos A, Real.sin A)
def n : ℝ × ℝ := (Real.cos B, Real.sin B)

-- Given dot product condition
variable (h_dot_product : m.1 * n.1 + m.2 * n.2 = Real.sqrt 3 * Real.sin B - Real.cos C)

-- Given side length
variable (h_side_a : a = 3)

-- The goal is to prove that the maximum area is 9√3/4
theorem maximum_area_of_triangle :
  (A = π / 3 ∨ A = 2 * π / 3) →
  (1 / 2) * b * c * Real.sin A ≤ 9 * Real.sqrt 3 / 4 :=
by
  sorry

end maximum_area_of_triangle_l263_263433


namespace xyz_squared_sum_l263_263491

theorem xyz_squared_sum (x y z : ℝ) 
  (h1 : x^2 + 4 * y^2 + 16 * z^2 = 48)
  (h2 : x * y + 4 * y * z + 2 * z * x = 24) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end xyz_squared_sum_l263_263491


namespace number_of_arrangements_l263_263503

open Nat

-- Define the set of people as a finite type with 5 elements.
inductive Person : Type
| youngest : Person
| eldest : Person
| p3 : Person
| p4 : Person
| p5 : Person

-- Define a function to count valid arrangements.
def countValidArrangements :
    ∀ (first_pos last_pos : Person), 
    (first_pos ≠ Person.youngest → last_pos ≠ Person.eldest → Fin 120) 
| first_pos, last_pos, h1, h2 => 
    let remaining := [Person.youngest, Person.eldest, Person.p3, Person.p4, Person.p5].erase first_pos |>.erase last_pos
    (factorial 3) * 4 * 3

-- Theorem statement to prove the number of valid arrangements.
theorem number_of_arrangements : 
  countValidArrangements Person.youngest Person.p5 sorry sorry = 72 :=
by 
  sorry

end number_of_arrangements_l263_263503


namespace weather_condition_l263_263364

theorem weather_condition (T : ℝ) (windy : Prop) (kites_will_fly : Prop) 
  (h1 : (T > 25 ∧ windy) → kites_will_fly) 
  (h2 : ¬ kites_will_fly) : T ≤ 25 ∨ ¬ windy :=
by 
  sorry

end weather_condition_l263_263364


namespace factor_expression_l263_263417

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) :=
by
  sorry

end factor_expression_l263_263417


namespace odd_square_imp_odd_l263_263598

theorem odd_square_imp_odd (n : ℤ) : odd (n^2) → odd n :=
sorry

end odd_square_imp_odd_l263_263598


namespace angle_of_inclination_l263_263403

theorem angle_of_inclination (θ : ℝ) (h_range : 0 ≤ θ ∧ θ < 180)
  (h_line : ∀ x y : ℝ, x + y - 1 = 0 → x = -y + 1) :
  θ = 135 :=
by 
  sorry

end angle_of_inclination_l263_263403


namespace problem_statement_l263_263845

noncomputable def f (k : ℝ) (a b : ℝ) : ℝ := 
  (2 * k) * (a * b) + 1

theorem problem_statement (a b : ℝ) (k x t : ℝ)
  (ha : |a| = 1) (hb : |b| = 1) 
  (h : |a + k * b| = sqrt(3) * |k * a - b|) (hk : k > 0) :
  f(k) = 4 * k / (k ^ 2 + 1) 
  ∧ ( ∀ t ∈ Icc (-2 : ℝ) (2 : ℝ), f(k) ≥ x^2 - 2 * t * x - (5/2)) ↔ (2 - sqrt 7 ≤ x ∧ x ≤ sqrt 7 - 2) :=
sorry

end problem_statement_l263_263845


namespace log_expression_l263_263681

theorem log_expression : log 2 + 2 * log 5 = 1 + log 5 := 
by
  sorry

end log_expression_l263_263681


namespace sequence_fifth_number_l263_263796

theorem sequence_fifth_number : (5^2 - 1) = 24 :=
by {
  sorry
}

end sequence_fifth_number_l263_263796


namespace area_DEF_twice_area_ABC_l263_263558

-- Define the problem conditions
variables {A B C P D E F : Point}
variable (circumcircle_ABC : Circle)
variable (equilateral_ABC : EquilateralTriangle ABC)
variable (P_on_circumcircle : P ∈ circumcircle_ABC)
variable (D_intersection : Collinear (Line PA) (Line BC) D)
variable (E_intersection : Collinear (Line PB) (Line CA) E)
variable (F_intersection : Collinear (Line PC) (Line AB) F)

-- Define the proof statement
theorem area_DEF_twice_area_ABC
  (equilateral_ABC : EquilateralTriangle ABC)
  (circle_ABC : Circumcircle ABC circumcircle_ABC)
  (P_on_circle : P ∈ circumcircle_ABC)
  (D_intersection : D ∈ (Line PA) ∩ (Line BC))
  (E_intersection : E ∈ (Line PB) ∩ (Line CA))
  (F_intersection : F ∈ (Line PC) ∩ (Line AB)) :
  area (triangle DEF) = 2 * area (triangle ABC) :=
sorry

end area_DEF_twice_area_ABC_l263_263558


namespace a_equals_bc_l263_263413

theorem a_equals_bc (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x y : ℝ, f x * g y = a * x * y + b * x + c * y + 1) → a = b * c :=
sorry

end a_equals_bc_l263_263413


namespace linear_increase_y_l263_263940

-- Progressively increase x and track y

theorem linear_increase_y (Δx Δy : ℝ) (x_increase : Δx = 4) (y_increase : Δy = 10) :
  12 * (Δy / Δx) = 30 := by
  sorry

end linear_increase_y_l263_263940


namespace second_order_arithmetic_sequence_term_15_l263_263963

theorem second_order_arithmetic_sequence_term_15 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 6) ∧ (a 4 = 11) ∧ 
  (∀ n, n ≥ 2 → a (n + 1) - a n = (a (n + 1) - a n)- (a n - a (n-1))) →
  (a 15 = 198) :=
by 
  intro a h,
  obtain ⟨h1, h2, h3, h4, h_pattern⟩ := h,
  sorry -- placeholder for the proof

end second_order_arithmetic_sequence_term_15_l263_263963


namespace count_non_congruent_triangles_with_perimeter_11_l263_263117

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263117


namespace triangle_count_with_perimeter_11_l263_263102

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263102


namespace max_value_f_l263_263541

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt(x * (100 - x)) + sqrt(x * (8 - x))

theorem max_value_f : 
  ∃ x₀ M, 0 ≤ x₀ ∧ x₀ ≤ 8 ∧ (∀ x, 0 ≤ x ∧ x ≤ 8 → f x ≤ f x₀) ∧ x₀ = 200 / 27 ∧ f x₀ = 12 * sqrt 6 :=
by
  sorry

end max_value_f_l263_263541


namespace max_elements_in_F_l263_263043

def D (x y : ℝ) : ℤ :=
  if h : x ≠ y then
    int.floor (real.log (abs (x - y)) / real.log 2) 
  else 
    0 -- when x = y, this case should not appear

def scale (F : set ℝ) (hF : F.nonempty) (x : ℝ) (hx : x ∈ F) : set ℤ :=
  {d | ∃ y ∈ F, x ≠ y ∧ D x y = d}

theorem max_elements_in_F {F : set ℝ} (hF : finite F) (k : ℕ) :
  (∀ x ∈ F, (scale F (finite.nonempty hF) x) .card ≤ k) → 
  F.card ≤ 2^k :=
sorry

end max_elements_in_F_l263_263043


namespace non_congruent_triangles_with_perimeter_11_l263_263125

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263125


namespace total_parallelepipeds_l263_263344

theorem total_parallelepipeds (m n k : ℕ) : 
  ∃ total : ℕ, total = (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8 :=
by
  use (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8
  sorry

end total_parallelepipeds_l263_263344


namespace count_good_numbers_less_1000_l263_263701

noncomputable def is_good_number (n : ℕ) : Prop :=
  ∀ d, n.digit d < 10 ∧ (d > 0 → n.digit d < 4) ∧ is_good_units (n % 10)

-- Helper function to check units place condition
def is_good_units (u : ℕ) : Prop :=
  u < 3

noncomputable def count_good_numbers (limit : ℕ) : ℕ :=
  (List.range limit).filter is_good_number |>.length

theorem count_good_numbers_less_1000 :
  count_good_numbers 1000 = 48 :=
by
  sorry

end count_good_numbers_less_1000_l263_263701


namespace length_QS_l263_263565

theorem length_QS : 
  ∀ (P Q R S : Type) 
  (right_angle_PQR : ∀ (a b c : Type), ∃ (θ : Type), θ = ∠PQR ∧ θ = 90) 
  (circle_with_diameter_QR_intersects_PR_at_S : ∀ (diameter : Type), diameter = QR ∧ S ∈ PR)
  (area_PQR : ∀ (P Q R : Type), ∃ (area : ℝ), area = 120)
  (PR : ℝ), 
  PR = 24 → 
  ∃ (QS : ℝ), QS = 10 := 
by 
  intros P Q R S right_angle_PQR circle_with_diameter_QR_intersects_PR_at_S area_PQR PR PR_value 
  use 10 
  sorry

end length_QS_l263_263565


namespace combined_population_percentage_l263_263194

theorem combined_population_percentage:
    let population_A := 10000
    let red_percentage_A := 0.60
    let female_percentage_A := 0.35
    let male_reds_A := population_A * red_percentage_A * (1 - female_percentage_A)
    
    let population_B := 15000
    let red_percentage_B := 0.45
    let female_percentage_B := 0.50
    let mutation_rate_B := 0.02
    let male_reds_B := population_B * red_percentage_B * (1 - female_percentage_B) * (1 - mutation_rate_B)
    
    let population_C := 20000
    let red_percentage_C := 0.70
    let female_percentage_C := 0.40
    let mutation_rate_C := 0.01
    let male_reds_C := population_C * red_percentage_C * (1 - female_percentage_C) * (1 - mutation_rate_C)
    
    let total_male_reds := male_reds_A + male_reds_B + male_reds_C
    let total_population := population_A + population_B + population_C
in
(total_male_reds / total_population) * 100 = 34.497 := 
begin
    sorry
end

end combined_population_percentage_l263_263194


namespace big_SUV_wash_ratio_l263_263670

-- Defining constants for time taken for various parts of the car
def time_windows : ℕ := 4
def time_body : ℕ := 7
def time_tires : ℕ := 4
def time_waxing : ℕ := 9

-- Time taken to wash one normal car
def time_normal_car : ℕ := time_windows + time_body + time_tires + time_waxing

-- Given total time William spent washing all vehicles
def total_time : ℕ := 96

-- Time taken for two normal cars
def time_two_normal_cars : ℕ := 2 * time_normal_car

-- Time taken for the big SUV
def time_big_SUV : ℕ := total_time - time_two_normal_cars

-- Ratio of time taken to wash the big SUV to the time taken to wash a normal car
def time_ratio : ℕ := time_big_SUV / time_normal_car

theorem big_SUV_wash_ratio : time_ratio = 2 := by
  sorry

end big_SUV_wash_ratio_l263_263670


namespace no_tangent_l263_263466

open Real

noncomputable theory

def line (m : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), (m + 2) * p.1 + (m - 1) * p.2 - 2 * m - 1 = 0

def circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 - 4 * p.1 + p.2 ^ 2 = 0

theorem no_tangent (m : ℝ) : ¬∃ (m : ℝ), ∀ p, line m p → circle p :=
sorry

end no_tangent_l263_263466


namespace tickets_total_l263_263709

theorem tickets_total (x y : ℕ) 
  (h1 : 12 * x + 8 * y = 3320)
  (h2 : y = x + 190) : 
  x + y = 370 :=
by
  sorry

end tickets_total_l263_263709


namespace count_non_congruent_triangles_with_perimeter_11_l263_263114

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263114


namespace integer_solution_count_l263_263854

theorem integer_solution_count :
  (set.count {x : ℤ | abs (x - 3) ≤ 4}) = 9 :=
sorry

end integer_solution_count_l263_263854


namespace determine_1000g_weight_l263_263312

-- Define the weights
def weights : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- Define the weight sets
def Group1 : List ℕ := [weights.get! 0, weights.get! 1]
def Group2 : List ℕ := [weights.get! 2, weights.get! 3]
def Group3 : List ℕ := [weights.get! 4]

-- Definition to choose the lighter group or determine equality
def lighterGroup (g1 g2 : List ℕ) : List ℕ :=
  if g1.sum = g2.sum then Group3 else if g1.sum < g2.sum then g1 else g2

-- Determine the 1000 g weight functionally
def identify1000gWeightUsing3Weighings : ℕ :=
  let firstWeighing := lighterGroup Group1 Group2
  if firstWeighing = Group3 then Group3.get! 0 else
  let remainingWeights := firstWeighing
  if remainingWeights.get! 0 = remainingWeights.get! 1 then Group3.get! 0
  else if remainingWeights.get! 0 < remainingWeights.get! 1 then remainingWeights.get! 0 else remainingWeights.get! 1

theorem determine_1000g_weight : identify1000gWeightUsing3Weighings = 1000 :=
sorry

end determine_1000g_weight_l263_263312


namespace three_digit_numbers_with_one_2_and_two_3s_l263_263476

theorem three_digit_numbers_with_one_2_and_two_3s :
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (nat.digit_frequencies n 10 2 = 1) ∧ (nat.digit_frequencies n 10 3 = 2)}.card = 3 :=
sorry

end three_digit_numbers_with_one_2_and_two_3s_l263_263476


namespace angle_conversion_l263_263027

/--
 Given an angle in degrees, express it in degrees, minutes, and seconds.
 Theorem: 20.23 degrees can be converted to 20 degrees, 13 minutes, and 48 seconds.
-/
theorem angle_conversion : (20.23:ℝ) = 20 + (13/60 : ℝ) + (48/3600 : ℝ) :=
by
  sorry

end angle_conversion_l263_263027


namespace complement_union_l263_263932

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 1 }
def C (s : Set ℝ) : Set ℝ := { x | ¬ s x }

theorem complement_union :
  C (A ∪ B) = { x | x ≤ -1 } :=
by {
  sorry
}

end complement_union_l263_263932


namespace fifty_gon_parallel_sides_l263_263637

theorem fifty_gon_parallel_sides :
  (∃ (L : Fin 50 → ℕ), (∀ i, 1 ≤ L i ∧ L i ≤ 50) ∧ 
  (Multiset.card (Multiset.map L Finset.univ.val) = 50) ∧
  (∀ i, abs (L i - L ((i + 25) % 50)) = 25)) →
  (∃ i j, i ≠ j ∧ L i = L j) :=
by
  sorry

end fifty_gon_parallel_sides_l263_263637


namespace math_problem_l263_263049

-- Definitions of the conditions
def condition (θ : ℝ) : Prop :=
  (2 * Real.cos (3 / 2 * Real.pi + θ) + Real.cos (Real.pi + θ)) / 
  (3 * Real.sin (Real.pi - θ) + 2 * Real.sin (5 / 2 * Real.pi + θ)) = 1 / 5

-- Definition of the first problem
def problem1 (θ : ℝ) (h : condition θ) : Prop :=
  Real.tan θ = 1

-- Definition of the second problem
def problem2 (θ : ℝ) (h1 : condition θ) (h2 : Real.tan θ = 1) : Prop :=
  Real.sin θ ^ 2 + 3 * Real.sin θ * Real.cos θ = 2

-- The Lean statement that includes both problems
theorem math_problem (θ : ℝ) (h1 : condition θ) : 
  problem1 θ h1 ∧ problem2 θ h1 (problem1 θ h1) :=
by
  sorry

end math_problem_l263_263049


namespace count_ways_line_up_l263_263509

theorem count_ways_line_up (persons : Finset ℕ) (youngest eldest : ℕ) :
  persons.card = 5 →
  youngest ∈ persons →
  eldest ∈ persons →
  (∃ seq : List ℕ, seq.length = 5 ∧ 
    ∀ (i : ℕ), i ∈ (List.finRange 5).erase 0 → seq.get ⟨i, sorry⟩ ≠ youngest ∧ 
    i ∈ (List.finRange 5).erase 4 → seq.get ⟨i, sorry⟩ ≠ eldest) →
  (persons \ {youngest, eldest}).card = 3 →
  4 * 4 * 3 * 2 * 1 = 96 :=
by
  sorry

end count_ways_line_up_l263_263509


namespace find_z_given_conditions_l263_263481

-- define the variable z and its conjugate \dot{z}
variables (z : ℂ) (conjugate_z : ℂ)

-- define the condition that \dot{z} is the conjugate of z
def is_conjugate (z conjugate_z : ℂ) : Prop := conjugate_z = conj z

-- define the condition \dot{z}(1-i) = 3+i
def satisfies_equation (conjugate_z : ℂ) : Prop := conjugate_z * (1 - I) = 3 + I

-- state the theorem we need to prove
theorem find_z_given_conditions (z : ℂ) (conjugate_z : ℂ) 
  (h1 : is_conjugate z conjugate_z)
  (h2 : satisfies_equation conjugate_z) :
  z = 1 - 2 * I :=
sorry

end find_z_given_conditions_l263_263481


namespace combination_sum_l263_263804

theorem combination_sum (n : ℕ)
  (h : (∑ k in Finset.range (n + 1), (3 ^ k) * Nat.choose n k) = 1024) :
  Nat.choose (n+1) 2 + Nat.choose (n+1) 3 = 35 :=
sorry

end combination_sum_l263_263804


namespace tree_height_after_4_months_l263_263235

noncomputable def tree_growth_rate := 50 -- growth in centimeters per two weeks
noncomputable def current_height_meters := 2 -- current height in meters
noncomputable def weeks_in_a_month := 4

def current_height_cm := current_height_meters * 100
def months := 4
def total_weeks := months * weeks_in_a_month
def growth_periods := total_weeks / 2
def total_growth := growth_periods * tree_growth_rate
def final_height := total_growth + current_height_cm

theorem tree_height_after_4_months :
  final_height = 600 :=
  by
    sorry

end tree_height_after_4_months_l263_263235


namespace max_n_T_n_less_than_7_l263_263277

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 1

def b_n (n : ℕ) : ℝ := (4 * n - 1) / (3 ^ (n - 1))

noncomputable def T_n (n : ℕ) : ℝ :=
  (15 / 2) - (4 * n + 5) / (2 * 3 ^ (n - 1))

theorem max_n_T_n_less_than_7 : ∀ n : ℕ, T_n n < 7 ↔ n ≤ 3 :=
by
  sorry

end max_n_T_n_less_than_7_l263_263277


namespace math_problem_l263_263179

noncomputable def canA_red_balls := 3
noncomputable def canA_black_balls := 4
noncomputable def canB_red_balls := 2
noncomputable def canB_black_balls := 3

noncomputable def prob_event_A := canA_red_balls / (canA_red_balls + canA_black_balls) -- P(A)
noncomputable def prob_event_B := 
  (canA_red_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls + 1) / (6) +
  (canA_black_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls) / (6) -- P(B)

theorem math_problem : 
  (prob_event_A = 3 / 7) ∧ 
  (prob_event_B = 17 / 42) ∧
  (¬ (prob_event_A * prob_event_B = (3 / 7) * (17 / 42))) ∧
  ((prob_event_A * (canB_red_balls + 1) / 6) / prob_event_A = 1 / 2) := by
  repeat { sorry }

end math_problem_l263_263179


namespace tetrahedron_volume_condition_l263_263298

noncomputable def volume_of_tetrahedron (a b c d : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * |(b - a) ⬝ (c - a) × (d - a)|

def is_on_same_face_of_cube (v w x y : ℝ × ℝ × ℝ) : Prop :=
  ∃ i, (i < 3) ∧ (v.1 - w.1 = 0 ∧ v.2 - w.2 = 0 ∧ v.3 - w.3 = 0) ∨
           (v.1 - x.1 = 0 ∧ v.2 - x.2 = 0 ∧ v.3 - x.3 = 0) ∨
           (v.1 - y.1 = 0 ∧ v.2 - y.2 = 0 ∧ v.3 - y.3 = 0)

theorem tetrahedron_volume_condition (a b c d : ℝ × ℝ × ℝ) 
  (h_cube : 
    a ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)} ∧ 
    b ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)} ∧
    c ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)} ∧ 
    d ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)}) : 
  volume_of_tetrahedron a b c d = (1/6) ↔ is_on_same_face_of_cube a b c d := 
by
  sorry

end tetrahedron_volume_condition_l263_263298


namespace real_values_of_k_l263_263385

theorem real_values_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 + (k + 1) * x + (k^2 - 3) = 0) ↔
  (frac (1 - 2 * real.sqrt 10) 3 ≤ k ∧ k ≤ frac (1 + 2 * real.sqrt 10) 3) :=
by
  sorry

end real_values_of_k_l263_263385


namespace finite_seq_sum_2009_l263_263401

theorem finite_seq_sum_2009 (n : ℕ) (a : ℕ → ℕ) (h : n ≥ 3) (h_sum : (∑ i in Finset.range n, a i) = 2009)
  (h_seq : ∀ i j : ℕ, i < j → a i = a (i + 1) - 1) :
  (n = 7 ∧ (a 0 = 284 ∧ a 1 = 285 ∧ a 2 = 286 ∧ a 3 = 287 ∧ a 4 = 288 ∧ a 5 = 289 ∧ a 6 = 290)) ∨
  (n = 14 ∧ (a 0 = 137 ∧ a 1 = 138 ∧ a 2 = 139 ∧ a 3 = 140 ∧ a 4 = 141 ∧ a 5 = 142 ∧ a 6 = 143 ∧
            a 7 = 144 ∧ a 8 = 145 ∧ a 9 = 146 ∧ a 10 = 147 ∧ a 11 = 148 ∧ a 12 = 149 ∧ a 13 = 150)) ∨
  (n = 41 ∧ (a 0 = 29 ∧ a 1 = 30 ∧ a 2 = 31 ∧ a 3 = 32 ∧ a 4 = 33 ∧ a 5 = 34 ∧ a 6 = 35 ∧ a 7 = 36 ∧
            a 8 = 37 ∧ a 9 = 38 ∧ a 10 = 39 ∧ a 11 = 40 ∧ a 12 = 41 ∧ a 13 = 42 ∧ a 14 = 43 ∧ a 15 = 44 ∧
            a 16 = 45 ∧ a 17 = 46 ∧ a 18 = 47 ∧ a 19 = 48 ∧ a 20 = 49 ∧ a 21 = 50 ∧ a 22 = 51 ∧ a 23 = 52 ∧
            a 24 = 53 ∧ a 25 = 54 ∧ a 26 = 55 ∧ a 27 = 56 ∧ a 28 = 57 ∧ a 29 = 58 ∧ a 30 = 59 ∧ a 31 = 60 ∧
            a 32 = 61 ∧ a 33 = 62 ∧ a 34 = 63 ∧ a 35 = 64 ∧ a 36 = 65 ∧ a 37 = 66 ∧ a 38 = 67 ∧ a 39 = 68 ∧
            a 40 = 69)) ∨
  (n = 49 ∧ (a 0 = 17 ∧ a 1 = 18 ∧ a 2 = 19 ∧ a 3 = 20 ∧ a 4 = 21 ∧ a 5 = 22 ∧ a 6 = 23 ∧ a 7 = 24 ∧
            a 8 = 25 ∧ a 9 = 26 ∧ a 10 = 27 ∧ a 11 = 28 ∧ a 12 = 29 ∧ a 13 = 30 ∧ a 14 = 31 ∧ a 15 = 32 ∧
            a 16 = 33 ∧ a 17 = 34 ∧ a 18 = 35 ∧ a 19 = 36 ∧ a 20 = 37 ∧ a 21 = 38 ∧ a 22 = 39 ∧ a 23 = 40 ∧
            a 24 = 41 ∧ a 25 = 42 ∧ a 26 = 43 ∧ a 27 = 44 ∧ a 28 = 45 ∧ a 29 = 46 ∧ a 30 = 47 ∧ a 31 = 48 ∧
            a 32 = 49 ∧ a 33 = 50 ∧ a 34 = 51 ∧ a 35 = 52 ∧ a 36 = 53 ∧ a 37 = 54 ∧ a 38 = 55 ∧ a 39 = 56 ∧
            a 40 = 57 ∧ a 41 = 58 ∧ a 42 = 59 ∧ a 43 = 60 ∧ a 44 = 61 ∧ a 45 = 62 ∧ a 46 = 63 ∧ a 47 = 64 ∧
            a 48 = 65)) :=
sorry

end finite_seq_sum_2009_l263_263401


namespace calculate_sin_C_calculate_area_l263_263876

variables {A B C : Type}
variables [T : triangle A B C] (a b c : ℝ) (cosB : ℝ)

-- Conditions
def b_value : b = 2 * real.sqrt 3 := sorry
def c_value : c = 3 := sorry
def cosB_value : cosB = -1 / 3 := sorry

-- Proof of sin C
theorem calculate_sin_C :
  ∃ (sinC : ℝ), sinC = real.sqrt 6 / 3 :=
sorry

-- Proof of area of triangle ABC
theorem calculate_area :
  ∃ (area : ℝ), area = real.sqrt 2 :=
sorry

end calculate_sin_C_calculate_area_l263_263876


namespace general_term_a_sum_first_n_c_l263_263204

-- Definitions based on the given conditions
def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, m > 0 → a (n + m) - a n = m * (a 2 - a 1)

def is_geometric_seq (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, m > 0 → b (n + m) = b n * (b 2 ^ m)

def a (n : ℕ) : ℤ := 2 * n - 1
def b (n : ℕ) : ℤ := 3 ^ (n - 1)
def c (n : ℕ) : ℤ := a n + b n

-- Assumptions based on the conditions
lemma b_seq_conditions : b 2 = 3 ∧ b 3 = 9 := by
  split
  · rfl -- b 2 = 3
  · rfl -- b 3 = 9

lemma a_b_relation : a 1 = b 1 ∧ a 14 = b 4 := by
  split
  · rfl -- a 1 = b 1
  · rfl -- a 14 = b 4

-- Proof statements
theorem general_term_a : ∀ n : ℕ, a n = 2 * n - 1 := by
  intro n
  rfl

theorem sum_first_n_c (n : ℕ) : (∑ i in Finset.range n, c (i + 1)) = n^2 + (3^n - 1) / 2 := by
  sorry

end general_term_a_sum_first_n_c_l263_263204


namespace part1_part2_l263_263815

variable {m n x1 x2 : ℝ}

theorem part1 (h : m * x^2 + n * x - (m + n) = 0) : 
  let Δ := n^2 + 4 * m * (m + n) in Δ ≥ 0 :=
sorry

theorem part2 (h : m * x^2 + x - (m + 1) = 0) (h1 : x1 * x2 > 1) :
  - (1 / 2) < m ∧ m < 0 :=
sorry

end part1_part2_l263_263815


namespace number_of_ways_to_lineup_five_people_l263_263505

noncomputable def numPermutations (people : List Char) (constraints : List (Char × Char)) : Nat :=
  List.factorial people.length / ∏ (c : Char × Char) in constraints, (match c.1 with
    | 'A' => (people.length - 1) -- A cannot be first
    | 'E' => (people.length - 1) -- E cannot be last
    | _ => people.length) 

theorem number_of_ways_to_lineup_five_people : 
  numPermutations ['A', 'B', 'C', 'D', 'E'] [('A', 'First-line'), ('E', 'Last-line')] = 96 := 
sorry

end number_of_ways_to_lineup_five_people_l263_263505


namespace total_production_in_march_l263_263398

-- Define the initial production, increase factor, and number of days in March as given conditions
def initial_production : ℕ := 7000
def increase_factor : ℕ := 3
def days_in_march : ℕ := 31

-- Define the problem statement
theorem total_production_in_march :
  (let additional_production_per_day := increase_factor * initial_production in
   let total_production_per_day := initial_production + additional_production_per_day in
   let total_production_in_march := total_production_per_day * days_in_march in
   total_production_in_march = 868000) :=
by sorry

end total_production_in_march_l263_263398


namespace problem1_problem2_problem3_l263_263719

-- Definition of given quantities and conditions
variables (a b x : ℝ) (α β : ℝ)

-- Given Conditions
@[simp] def cond1 := true
@[simp] def cond2 := true
@[simp] def cond3 := true
@[simp] def cond4 := true

-- First Question
theorem problem1 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    a * Real.sin α = b * Real.sin β := sorry

-- Second Question
theorem problem2 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    Real.sin β ≤ a / b := sorry

-- Third Question
theorem problem3 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    x = a * (1 - Real.cos α) + b * (1 - Real.cos β) := sorry

end problem1_problem2_problem3_l263_263719


namespace m_cannot_be_3_sin_A_l263_263160

-- Define the problem conditions
variables (a b c : ℝ) (m : ℝ)
def triangle_condition : Prop := a^2 + c^2 - b^2 = m * a * c

-- Proposition for part (I)
theorem m_cannot_be_3 (h : triangle_condition a b c 3) : false :=
by
  -- Proof omitted, filled with sorry
  sorry

-- Proposition for part (II)
theorem sin_A (h1 : triangle_condition a (2 * Real.sqrt 7) 4 (-1)) (h2 : b = 2 * Real.sqrt 7) (h3 : c = 4) : sin A = Real.sqrt 21 / 14 :=
by
  -- Proof omitted, filled with sorry
  sorry

end m_cannot_be_3_sin_A_l263_263160


namespace non_congruent_triangles_with_perimeter_11_l263_263106

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263106


namespace total_spending_is_140_l263_263961

-- Define definitions for each day's spending based on the conditions.
def monday_spending : ℕ := 6
def tuesday_spending : ℕ := 2 * monday_spending
def wednesday_spending : ℕ := 2 * (monday_spending + tuesday_spending)
def thursday_spending : ℕ := (monday_spending + tuesday_spending + wednesday_spending) / 3
def friday_spending : ℕ := thursday_spending - 4
def saturday_spending : ℕ := friday_spending + (friday_spending / 2)
def sunday_spending : ℕ := tuesday_spending + saturday_spending

-- The total spending for the week.
def total_spending : ℕ := 
  monday_spending + 
  tuesday_spending + 
  wednesday_spending + 
  thursday_spending + 
  friday_spending + 
  saturday_spending + 
  sunday_spending

-- The theorem to prove that the total spending is $140.
theorem total_spending_is_140 : total_spending = 140 := 
  by {
    -- Due to the problem's requirement, we skip the proof steps.
    sorry
  }

end total_spending_is_140_l263_263961


namespace area_of_inscribed_triangle_l263_263353

theorem area_of_inscribed_triangle (arc1 arc2 arc3 : ℝ) (h1 : arc1 = 4) (h2 : arc2 = 5) (h3 : arc3 = 7) :
  let circumference := arc1 + arc2 + arc3 in
  let radius := circumference / (2 * Real.pi) in
  let theta := 360 / (arc1 + arc2 + arc3) in
  let angle1 := (5 * theta + 7 * theta) in
  let angle2 := (4 * theta + 7 * theta) in
  let angle3 := (4 * theta + 5 * theta) in
  let a := radius in
  let b := radius in
  let area := (1 / 2) * a * b * (Real.sin (angle1 / 2 * Real.pi / 180) + Real.sin (angle2 / 2 * Real.pi / 180) + Real.sin (angle3 / 2 * Real.pi / 180)) in
  area = (16 / Real.pi ^ 2) * (Real.sqrt 2 + 1) :=
by 
  sorry

end area_of_inscribed_triangle_l263_263353


namespace count_non_congruent_triangles_with_perimeter_11_l263_263116

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263116


namespace find_a_l263_263231

open Function

noncomputable def slope_1 (a x_0 : ℝ) : ℝ :=
  (a * x_0 + a - 1) * Real.exp x_0

noncomputable def slope_2 (x_0 : ℝ) : ℝ :=
  (x_0 - 2) * Real.exp (-x_0)

theorem find_a (x_0 : ℝ) (a : ℝ)
  (h1 : x_0 ∈ Set.Icc 0 (3 / 2))
  (h2 : slope_1 a x_0 * slope_2 x_0 = -1) :
  1 ≤ a ∧ a ≤ 3 / 2 := sorry

end find_a_l263_263231


namespace find_OB_maximized_volume_l263_263970

-- Define the geometric setup and the given conditions
variables (P A B O H C : Type) [EuclideanSpace3D P A B O H C]

-- Definitions body
def isosceles_right_triangle {X Y Z : Type} (XYZ : Triangle X Y Z) : Prop :=
  XYZ.is_isosceles ∧ XYZ.right_angle_at_vertex X

def midpoint (M X Y : Type) : Prop := dist M X = dist M Y ∧ ∀ x, M = midpoint X Y

def perpendicular (X1 X2 X3 : Type) : Prop := PlaneAngle X1 X2 X3 = π/2

-- Given conditions as assumptions
variables (PA_length : dist P A = 4)
variables (mid_C_PA : midpoint C P A)
variables (perp_AB_OB : perpendicular A B O)
variables (perp_OH_PB : perpendicular O H P)
variables (B_inner_base : is_point_inside_base B O)
variables (O_center_base : is_center O)

-- Proof of volume maximization leads to specific length of OB
theorem find_OB_maximized_volume :
  maximized_volume (Tetrahedron O H P C) → dist O B = 2 * (sqrt 6) / 3 :=
  sorry

end find_OB_maximized_volume_l263_263970


namespace penny_identified_species_l263_263910

theorem penny_identified_species (sharks eels whales : ℕ) :
  sharks = 35 → eels = 15 → whales = 5 → sharks + eels + whales = 55 :=
by
  intros h_sharks h_eels h_whales
  rw [h_sharks, h_eels, h_whales]
  sorry

end penny_identified_species_l263_263910


namespace min_x_plus_3y_l263_263447

noncomputable def minimum_x_plus_3y (x y : ℝ) : ℝ :=
  if h : (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) then x + 3*y else 0

theorem min_x_plus_3y : ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ x + 3*y + x*y = 9) → x + 3*y = 6 :=
by
  intros x y h
  sorry

end min_x_plus_3y_l263_263447


namespace isosceles_triangle_range_of_expression_l263_263163

open Real

-- Given: In triangle ABC, sides opposite to angles A, B, C are a, b, c respectively
-- and satisfy a * cos B = b * cos A.
-- Prove:
-- 1. The triangle is isosceles (A = B).
-- 2. The range of sin (2A + π/6) - 2 * cos^2 B is (-3/2, 0).

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) (h₁ : a * cos B = b * cos A) :
  A = B :=
sorry

theorem range_of_expression (A : ℝ) (hA : 0 < A ∧ A < π / 2) (h_isosceles : A = B) :
  -3/2 < sin (2 * A + π / 6) - 2 * cos^2 B ∧ sin (2 * A + π / 6) - 2 * cos^2 B < 0 :=
sorry

end isosceles_triangle_range_of_expression_l263_263163


namespace school_ticket_purchase_l263_263608

theorem school_ticket_purchase :
  ∃ (x y : ℕ), x + y = 700 ∧ 60 * x + 10000 + 10000 + 80 * (y - 100) = 58000 ∧ x = 500 ∧ y = 200 :=
begin
  sorry
end

end school_ticket_purchase_l263_263608


namespace simplify_and_multiply_l263_263601

theorem simplify_and_multiply :
  let a := 3
  let b := 17
  let d1 := 504
  let d2 := 72
  let m := 5
  let n := 7
  let fraction1 := a / d1
  let fraction2 := b / d2
  ((fraction1 - (b * n / (d2 * n))) * (m / n)) = (-145 / 882) :=
by
  sorry

end simplify_and_multiply_l263_263601


namespace sin_supplementary_angle_l263_263443

theorem sin_supplementary_angle (α : ℝ) (h : Real.sin (π / 4 + α) = sqrt 3 / 2) :
  Real.sin (3 * π / 4 - α) = sqrt 3 / 2 :=
sorry

end sin_supplementary_angle_l263_263443


namespace trapezoid_ad_length_mn_l263_263649

open EuclideanGeometry

variables {A B C D O P : Point}
variables {m n : ℕ}

-- Given conditions
def is_trapezoid (A B C D : Point) : Prop := 
  A.y = B.y ∧ C.y = D.y ∧ B.x - A.x ≠ D.x - C.x

def length_eq (x y : ℕ) : Prop := 
  x = 43 ∧ y = 43

def perpendicular (A D B : Point) : Prop := 
  (A.x - D.x) * (D.x - B.x) + (A.y - D.y) * (D.y - B.y) = 0

def midpoint (P B D : Point) : Prop := 
  2 * P.x = B.x + D.x ∧ 2 * P.y = B.y + D.y

def inter_diag (A C B D O : Point) : Prop := 
  ∃ λ : ℝ, O = λ • A + (1 - λ) • C ∧  ∃ μ : ℝ, O = μ • B + (1 - μ) • D

def OP_length (O P : Point) (l : ℝ) : Prop := 
  dist O P = l

-- Prove the final tuple
theorem trapezoid_ad_length_mn (hT : is_trapezoid A B C D) (hL : length_eq (dist B C) (dist C D))
  (hP : perpendicular A D B) (hM : midpoint P B D) (hI : inter_diag A C B D O)
  (hO : OP_length O P 11) : 
  ∃ (m n : ℕ), dist A D = m * Real.sqrt n ∧ m + n = 194 := 
sorry

end trapezoid_ad_length_mn_l263_263649


namespace find_a_l263_263460

def f (a x : ℝ) : ℝ := a * x ^ 3 - 3 * x + 2016
def f_derivative (a x : ℝ) : ℝ := 3 * a * x ^ 2 - 3

theorem find_a (a : ℝ) : f_derivative a 1 = 0 → a = 1 :=
by
  assume h : f_derivative a 1 = 0
  show a = 1, from sorry

end find_a_l263_263460


namespace pizza_slices_left_l263_263383

theorem pizza_slices_left (initial_slices : ℕ) (ate_by_dean : ℕ) (ate_by_frank : ℕ) (ate_by_sammy : ℕ) :
  initial_slices = 24 →
  ate_by_dean = 6 →
  ate_by_frank = 3 →
  ate_by_sammy = 4 →
  initial_slices - (ate_by_dean + ate_by_frank + ate_by_sammy) = 11 :=
by
  intros h_initial h_dean h_frank h_sammy
  rw [h_initial, h_dean, h_frank, h_sammy]
  norm_num
  exact sorry

end pizza_slices_left_l263_263383


namespace non_congruent_triangles_with_perimeter_11_l263_263127

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263127


namespace last_two_nonzero_digits_of_80_fact_l263_263991

theorem last_two_nonzero_digits_of_80_fact :
  ∃ m : ℕ, (m = 52) ∧ (80! % 100 = m) :=
by
  sorry

end last_two_nonzero_digits_of_80_fact_l263_263991


namespace vector_simplification_l263_263955

variables {V : Type} [AddCommGroup V] [VectorSpace float V]

variable (A B C : V)

theorem vector_simplification :
  (B - A) - (C - A) + (C - B) = (0 : V) :=
sorry

end vector_simplification_l263_263955


namespace excluded_number_is_20_l263_263266

open Real

theorem excluded_number_is_20
  (nums : Fin 5 → ℝ) 
  (avg5 : (∑ i, nums i) / 5 = 12) 
  (excluded_num rest_nums : Fin 4 → ℝ)
  (avg4 : (∑ i, rest_nums i) / 4 = 10)
  (sum_relation : (∑ i, nums i) = (∑ i, rest_nums i) + excluded_num) :
  excluded_num = 20 :=
sorry

end excluded_number_is_20_l263_263266


namespace difference_between_extremes_l263_263016

-- Define the iterative average process
def iterative_average (s : List ℚ) : ℚ :=
  match s with
  | [] => 0
  | [x] => x
  | x :: y :: xs => iterative_average ((x + y) / 2 :: xs)

-- Define the sequences
def decreasing_seq : List ℚ := [6, 5, 4, 3, 2, 1]
def increasing_seq : List ℚ := [1, 2, 3, 4, 5, 6]

-- Define the final averages for both sequences
def final_average_decreasing := iterative_average decreasing_seq
def final_average_increasing := iterative_average increasing_seq

-- Prove the difference between the largest and smallest possible final averages
theorem difference_between_extremes :
  final_average_increasing - final_average_decreasing = 3.0625 := by
  sorry

end difference_between_extremes_l263_263016


namespace propositions_true_false_l263_263052

-- Definitions for the conditions
variables {m n : Line} {α β : Plane}
variable (f1 : α ⟂ β → m ∈ α → n ∈ β → m ⟂ n) -- f1 us a placeholder for falsey conditions
variable (f2 : m ⟂ α → n ⟂ β → m ∥ n → α ∥ β)
variable (f3 : α ∥ β → m ∈ α → m ∥ β)

-- Theorem to prove true and false propositions
theorem propositions_true_false :
  (∀ (α β : Plane) (m n : Line), (α ∥ β ∧ m ∈ α ∧ n ∈ β) → ¬(m ∥ n)) ∧ 
  (∀ (m n : Line) (α β : Plane), (m ⟂ α ∧ n ⟂ β ∧ m ∥ n) → (α ∥ β)) ∧ 
  (∀ (α β : Plane) (m : Line), (α ∥ β ∧ m ∈ α) → (m ∥ β)) :=
by
  split; try {split}; intros; sorry

end propositions_true_false_l263_263052


namespace pizza_slices_left_over_l263_263382

theorem pizza_slices_left_over :
  ∀ (total_pizzas : ℕ) (slices_per_pizza : ℕ) (dean_hawaiian_frac : ℚ) 
    (frank_hawaiian_slices : ℕ) (sammy_cheese_frac : ℚ)
    (total_slices_eaten : ℕ) (left_over_slices : ℕ),
  total_pizzas = 2 →
  slices_per_pizza = 12 →
  dean_hawaiian_frac = 1 / 2 →
  frank_hawaiian_slices = 3 →
  sammy_cheese_frac = 1 / 3 →
  total_slices_eaten = ((slices_per_pizza * dean_hawaiian_frac) + frank_hawaiian_slices).to_nat + (slices_per_pizza / 3) →
  left_over_slices = (total_pizzas * slices_per_pizza) - total_slices_eaten →
  left_over_slices = 11 :=
sorry

end pizza_slices_left_over_l263_263382


namespace flour_already_added_l263_263938

theorem flour_already_added (sugar flour salt additional_flour : ℕ) 
  (h1 : sugar = 9) 
  (h2 : flour = 14) 
  (h3 : salt = 40)
  (h4 : additional_flour = sugar + 1) : 
  flour - additional_flour = 4 :=
by
  sorry

end flour_already_added_l263_263938


namespace gas_volume_at_10_degrees_l263_263798

def volume_of_gas (V : ℕ) (T : ℕ) : Prop :=
  ∀ (T₁ T₂ : ℕ), T₂ = T₁ - 15 → T = 25 → V = 40 → V - 9 = 31

theorem gas_volume_at_10_degrees :
  volume_of_gas 31 10 :=
by
  unfold volume_of_gas
  intros T₁ T₂ hT2 hT hV
  have h : 3 * 5 = 15 := rfl
  rw [h] at hT2
  subst hT2
  rw hV
  exact rfl

end gas_volume_at_10_degrees_l263_263798


namespace find_eccentricity_of_ellipse_l263_263068

noncomputable def ellipse_eccentricity : ℝ :=
  let a := real.sqrt 1 in
  let b := real.sqrt (1 / 4) in
  real.sqrt (1 - (b^2 / a^2))

theorem find_eccentricity_of_ellipse :
  ∀ (a b : ℝ), (a > b) → (b > 0) → 
  let E := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} in
  let M := (2, 1) in M ∈ E →
  ( ∀ (λ : ℝ), λ > 0 → λ ≠ 1 →
    let A := (x1, y1), B := (x2, y2) in 
    ∀ (x1 y1 x2 y2 : ℝ), (y2 - y1) / (x2 - x1) = -1 / 2 ) →
  let e := real.sqrt (1 - (b^2 / a^2)) in
  e = real.sqrt 3 / 2 :=
by
  sorry

end find_eccentricity_of_ellipse_l263_263068


namespace sum_of_roots_l263_263664

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots :
  (quadratic_eq 1 (-6) 9) x → (quadratic_eq 1 (-6) 9) y → x ≠ y → x + y = 6 :=
by
  sorry

end sum_of_roots_l263_263664


namespace sufficient_but_not_necessary_condition_l263_263322

open Real

theorem sufficient_but_not_necessary_condition (k : ℤ) : 
  (∀ x, tan x = 1 ↔ ∃ k : ℤ, x = 2 * k * π + π / 4) → false :=
begin
  -- Define the necessary variables and conditions
  assume h,
  -- Proof goes here
  sorry
end

end sufficient_but_not_necessary_condition_l263_263322


namespace polynomial_complex_inequality_l263_263812

noncomputable def P (z : ℂ) (n : ℕ) (c : Fin n → ℝ) : ℂ :=
  (Finset.range n).sum (λ i, (c i : ℂ) * z ^ (n - (i+1)))

theorem polynomial_complex_inequality (n : ℕ) (c : Fin n → ℝ) (P_i_lt_1 : abs (P complex.I n c) < 1) :
  ∃ (a b : ℝ), P (a + b * complex.I) n c = 0 ∧ (a^2 + b^2 + 1)^2 < 4 * b^2 + 1 :=
sorry

end polynomial_complex_inequality_l263_263812


namespace symmetric_origin_l263_263917

noncomputable def z1 : ℂ := 2 - 3 * Complex.i
noncomputable def z2 : ℂ := -2 + 3 * Complex.i

theorem symmetric_origin (z1 z2 : ℂ) (h : z2 = -z1) : z2 = -2 + 3 * Complex.i :=
by
  have h1 : z1 = 2 - 3 * Complex.i := sorry
  have h2 : z2 = -z1     := sorry
  rw h1 at h
  exact h2
  /- sorry -/

end symmetric_origin_l263_263917


namespace inverse_function_log_base_3_l263_263764

theorem inverse_function_log_base_3 (x : ℝ) (hx : 0 < x) :
  (∀ y : ℝ, y = 3^x ↔ x = log 3 y) →
  ∃ f : ℝ → ℝ, ∀ y, y = f (3^x) ↔ x = log 3 y :=
by
  sorry

end inverse_function_log_base_3_l263_263764


namespace min_xy_sum_is_7_l263_263422

noncomputable def min_xy_sum (x y : ℝ) : ℝ := 
x + y

theorem min_xy_sum_is_7 (x y : ℝ) (h1 : x > 1) (h2 : y > 2) (h3 : (x - 1) * (y - 2) = 4) : 
  min_xy_sum x y = 7 := by 
  sorry

end min_xy_sum_is_7_l263_263422


namespace smallest_palindrome_base2_base4_l263_263002

-- Function to check if a number is a palindrome in a given base
def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n in digits = digits.reverse

theorem smallest_palindrome_base2_base4 (n : ℕ) (hn : n > 15) :
  is_palindrome n 2 ∧ is_palindrome n 4 → n = 85 :=
by sorry

end smallest_palindrome_base2_base4_l263_263002


namespace correct_propositions_l263_263834

def proposition_1 := ∃ α : ℝ, sin α * cos α = 1
def f (x : ℝ) := -2 * cos (7 * π / 2 - 2 * x)
def proposition_2 := ∀ x : ℝ, f (-x) = -f x
def g (x : ℝ) := 3 * sin (2 * x - 3 * π / 4)
def proposition_3 := ∀ x : ℝ, g (-3 * π / 8 + x) = g (-3 * π / 8 - x)
def h (x : ℝ) := cos (sin x)
def proposition_4 := ∀ y : ℝ, y = h x → y ∈ set.Icc 0 (cos 1)

theorem correct_propositions :
  (¬ proposition_1) ∧ proposition_2 ∧ proposition_3 ∧ ¬ proposition_4 :=
by sorry

end correct_propositions_l263_263834


namespace neg_sqrt_two_sq_l263_263732

theorem neg_sqrt_two_sq : (- Real.sqrt 2) ^ 2 = 2 := 
by
  sorry

end neg_sqrt_two_sq_l263_263732


namespace domain_and_range_l263_263459

-- Define the function
def f (x : ℝ) : ℝ := log (2 : ℝ) ((x - 1) / (x + 1))

-- Define the domain of f(x)
def domain_f : set ℝ := { x | (x - 1) / (x + 1) > 0 }

-- Define the sets A and B
def A : set ℝ := { x | x < -1 ∨ x > 1 }
def B (a : ℝ) : set ℝ := { x | (x - a) * (x - a - 2) < 0 }

-- The main statement to be proven
theorem domain_and_range (a : ℝ) :
  domain_f = A ∧ ((A ∩ B a = B a) → (a ≤ -3 ∨ a ≥ 1)) :=
by
  sorry

end domain_and_range_l263_263459


namespace propositions_l263_263684

variable (m n : Type) [Line m] [Line n]
variable (α β : Type) [Plane α] [Plane β]
variable [Parallel m α] [Parallel n β] [Perpendicular m α] [Perpendicular n β]
variable [Parallel α β] [Perpendicular α β]

theorem propositions (h₀ : Parallel m α ∧ Parallel n β ∧ Parallel α β →
                      ¬Parallel m n)
                    (h₁ : Perpendicular m α ∧ Perpendicular n β ∧ Perpendicular α β →
                      Perpendicular m n)
                    (h₂ : Perpendicular m α ∧ Parallel n β ∧ Parallel α β →
                      Perpendicular m n)
                    (h₃ : Parallel m α ∧ Perpendicular n β ∧ Perpendicular α β →
                      ¬Parallel m n):
  (¬Parallel m α ∨ ¬Parallel n β ∨ ¬Parallel α β ∨ Parallel m n) ∧
  (Perpendicular m α ∧ Perpendicular n β ∧ Perpendicular α β) ∧
  (Perpendicular m α ∧ Parallel n β ∧ Parallel α β) ∧
  (¬Parallel m α ∨ ¬Perpendicular n β ∨ ¬Perpendicular α β ∨ Parallel m n) := by
  sorry

end propositions_l263_263684


namespace find_p_l263_263982

noncomputable def p (x : ℝ) : ℝ := (9/5) * (x^2 - 4)

theorem find_p :
  ∃ (a : ℝ), (∀ x, p(x) = a * (x + 2) * (x - 2)) ∧ p(-3) = 9 :=
by
  use 9/5
  split
  { intro x
    rw p
    ring }
  { rw p
    norm_num }
  sorry

end find_p_l263_263982


namespace base7_multiplication_l263_263409

theorem base7_multiplication (a b : ℕ) (h₁ : a = 3 * 7^2 + 2 * 7^1 + 5) (h₂ : b = 3) : 
  let ab := (a * b) in
  nat_repr_ab7 3111 := nat_repr'_base ab 7 :=
begin
  sorry
end

end base7_multiplication_l263_263409


namespace percent_of_x_l263_263484

variable {x y z : ℝ}

-- Define the given conditions
def cond1 (z y : ℝ) : Prop := 0.45 * z = 0.9 * y
def cond2 (z x : ℝ) : Prop := z = 1.5 * x

-- State the theorem to prove
theorem percent_of_x (h1 : cond1 z y) (h2 : cond2 z x) : y = 0.75 * x :=
sorry

end percent_of_x_l263_263484


namespace parallel_lines_same_slope_l263_263395

theorem parallel_lines_same_slope (k : ℝ) : 
  (2*x + y + 1 = 0) ∧ (y = k*x + 3) → (k = -2) := 
by
  sorry

end parallel_lines_same_slope_l263_263395


namespace pizza_slices_left_over_l263_263381

theorem pizza_slices_left_over :
  ∀ (total_pizzas : ℕ) (slices_per_pizza : ℕ) (dean_hawaiian_frac : ℚ) 
    (frank_hawaiian_slices : ℕ) (sammy_cheese_frac : ℚ)
    (total_slices_eaten : ℕ) (left_over_slices : ℕ),
  total_pizzas = 2 →
  slices_per_pizza = 12 →
  dean_hawaiian_frac = 1 / 2 →
  frank_hawaiian_slices = 3 →
  sammy_cheese_frac = 1 / 3 →
  total_slices_eaten = ((slices_per_pizza * dean_hawaiian_frac) + frank_hawaiian_slices).to_nat + (slices_per_pizza / 3) →
  left_over_slices = (total_pizzas * slices_per_pizza) - total_slices_eaten →
  left_over_slices = 11 :=
sorry

end pizza_slices_left_over_l263_263381


namespace evaluate_expression_l263_263758

theorem evaluate_expression :
  (125^(1/3) * 8^(1/3) / 32^(-1/5) = 20) :=
by
  sorry

end evaluate_expression_l263_263758


namespace marbles_problem_l263_263025

def marbles_total : ℕ := 30
def prob_black_black : ℚ := 14 / 25
def prob_white_white : ℚ := 16 / 225

theorem marbles_problem (total_marbles : ℕ) (prob_bb prob_ww : ℚ) 
  (h_total : total_marbles = 30)
  (h_prob_bb : prob_bb = 14 / 25)
  (h_prob_ww : prob_ww = 16 / 225) :
  let m := 16
  let n := 225
  m.gcd n = 1 ∧ m + n = 241 :=
by {
  sorry
}

end marbles_problem_l263_263025


namespace decrypt_phone_number_l263_263296

theorem decrypt_phone_number
  (symbols : ℕ → Finset (Fin 4))
  (h_unique : ∀ i j, symbols i = symbols j → i = j)
  (h_segment : ∀ i j, ¬ Disjoint (symbols i) (symbols j) → |i - j| ≤ 2)
  (start_with_8 : symbols 8 = [-]) :
  (decode symbols 83859206147).head = 8 :=
sorry

end decrypt_phone_number_l263_263296


namespace length_of_each_piece_l263_263857

-- Definitions based on conditions
def total_length : ℝ := 42.5
def number_of_pieces : ℝ := 50

-- The statement that we need to prove
theorem length_of_each_piece (h1 : total_length = 42.5) (h2 : number_of_pieces = 50) : 
  total_length / number_of_pieces = 0.85 := 
by
  sorry

end length_of_each_piece_l263_263857


namespace Petya_entrance_solution_l263_263582

-- Define the entrances and positions
variable (A D B C : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  D = 4 ∧ (A ⟶ D = B ⟶ C ⟶ D)

-- Define the intended conclusion
def Petya_entrance_is_6 (A : ℕ) : Prop :=
  A = 6

theorem Petya_entrance_solution :
  ∃ A, (conditions A D B C) → (Petya_entrance_is_6 A) :=
by
  -- We skip the proof here, as only the statement structure is needed
  -- Proof would be constructed based on the solution steps outlined earlier
  sorry

end Petya_entrance_solution_l263_263582


namespace least_positive_int_factorial_5775_l263_263662

def prime_factors_5775 := [(5 : ℕ, 2), (3 : ℕ, 5), (7 : ℕ, 1)]

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def count_prime_factors (n k : ℕ) (p : ℕ) : ℕ :=
  if k = 0 then 0 else count_prime_factors n (k-1) p + nat.div (n - k + 1) p

def satisfies_factors (n : ℕ) :=
  ∀ (p m : ℕ), (p, m) ∈ prime_factors_5775 → count_prime_factors n n p ≥ m

theorem least_positive_int_factorial_5775 :
  ∃ (n : ℕ), satisfies_factors n ∧ ¬ ∃ (m : ℕ), satisfies_factors m ∧ m < n :=
begin
  sorry -- proof to be provided
end

end least_positive_int_factorial_5775_l263_263662


namespace trajectory_eq_ellipse_max_area_triangle_l263_263070

noncomputable def circle1 : set (ℝ × ℝ) := { p | (p.1 + 1)^2 + p.2^2 = 9 }
noncomputable def circle2 : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def trajectory : set (ℝ × ℝ) := { p | p.1^2 / 4 + p.2^2 / 3 = 1 ∧ p.1 ≠ 2 }
noncomputable def line (k : ℝ) : set (ℝ × ℝ) := { p | p.2 = k * p.1 - 2 }

theorem trajectory_eq_ellipse (P : set (ℝ × ℝ)) :
  (∀ p ∈ P, p ∈ circle1 → p ∈ trajectory ∧
            ∀ p ∈ P, p ∈ trajectory → p ∉ circle2) →
  trajectory = { p | p.1^2 / 4 + p.2^2 / 3 = 1 ∧ p.1 ≠ 2 } :=
sorry

theorem max_area_triangle (k : ℝ) :
  ∃ A B : ℝ × ℝ, A ∈ trajectory ∧ B ∈ trajectory ∧ A ∈ line k ∧ B ∈ line k ∧
  (let d := 2 / real.sqrt (1 + k^2) in
   let AB_dist := real.sqrt (1 + k^2) * real.sqrt ((16 * k) / (3 + 4 * k^2)) in
   let area := 1/2 * AB_dist * d in
   area = real.sqrt(3) ∧ k = real.sqrt(5)/2 ∨ k = -real.sqrt(5)/2) :=
sorry

end trajectory_eq_ellipse_max_area_triangle_l263_263070


namespace council_counts_l263_263170

theorem council_counts 
    (total_classes : ℕ := 20)
    (students_per_class : ℕ := 5)
    (total_students : ℕ := 100)
    (petya_class_council : ℕ × ℕ := (1, 4))  -- (boys, girls)
    (equal_boys_girls : 2 * 50 = total_students)  -- Equal number of boys and girls
    (more_girls_classes : ℕ := 15)
    (min_girls_each : ℕ := 3)
    (remaining_classes : ℕ := 4)
    (remaining_students : ℕ := 20)
    : (19, 1) = (19, 1) :=
by
    -- actual proof goes here
    sorry

end council_counts_l263_263170


namespace reflection_of_I_on_circumscribed_circle_l263_263564

open EuclideanGeometry -- Open the necessary part of the library

variables {ABC : Type*} [Triangle ABC] -- Define the type and properties of triangle ABC
variables {I D E : Point} -- Define the points I, D, and E
variables {AB AC BC : Real} -- Define the side lengths of the triangle
variables (BI : Line) -- Define the line BI

-- Assume the given conditions
axiom AB_eq_AC : AB = AC
axiom AB_ne_BC : AB ≠ BC
axiom I_center_of_incircle : is_incenter I ABC
axiom BI_cuts_AC_at_D : is_intersection BI AC D
axiom Perpendicular_to_AC_at_D : Perpendicular (Line_through D Perpendicular_to AC) AC
axiom Perpendicular_cuts_AI_at_E : is_intersection (Perpendicular D AC (Line_through D Perpendicular_to AC)) AI E

-- Define the reflection of I over AC
noncomputable def Reflection_I_over_AC : Point := reflection I AC

-- The main theorem to prove
theorem reflection_of_I_on_circumscribed_circle :
  OnCircumscribedCircle (Reflection_I_over_AC I AC) (Triangle BDE) :=
sorry

end reflection_of_I_on_circumscribed_circle_l263_263564


namespace abs_diff_U_l263_263515

variable (P Q R U V : Point)
variable (distance : Point → Point → ℝ)

-- Definitions of points
def P := (0, 10) : Point
def Q := (5, 0) : Point
def R := (10, 0) : Point
def V := (2, 0) : Point

-- Definition of line PR
def line_PR : Line := Line.mk (P, R)

-- Condition: VQ = 3 units
def VQ_distance := distance V Q = 3

-- Coordinates of U are determined by the intersection of line x = 2 with line_PR
def U : Point := let x := 2 in (x, -x + 10)  -- which is (2, 8)

-- The statement to prove
theorem abs_diff_U : |U.1 - U.2| = 6 :=
by sorry

end abs_diff_U_l263_263515


namespace female_officers_count_l263_263580
-- Lean 4 statement

theorem female_officers_count :
  ∃ (F : ℝ), (0.19 * F = 76) ∧ (152 / 2 = 76) ∧ (F = 400) :=
begin
  use 400,
  split,
  { linarith },
  split,
  { linarith },
  { refl }
end

end female_officers_count_l263_263580


namespace sum_of_odd_binomial_coeffs_number_of_rational_terms_l263_263829

-- Given n = 12 such that (2x - 1/∛x) ^ n has 13 terms.
def n : ℕ := 12

-- Proving the sum of the binomial coefficients of all odd terms in the expansion (2x - 1/∛x) ^ n is 2^11
theorem sum_of_odd_binomial_coeffs :
  ∑ k in (finset.range (n + 1)).filter (λ k, ¬(k % 2 = 0)), binomial n k = 2 ^ 11 := sorry

-- Proving there are a total of 5 rational terms in the expansion (2x - 1/∛x) ^ n
theorem number_of_rational_terms :
  ((finset.range (n + 1)).filter (λ k, is_integral (12 - 4 * k / 3))).card = 5 := sorry

end sum_of_odd_binomial_coeffs_number_of_rational_terms_l263_263829


namespace problem_equivalence_l263_263310

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≥ f y

theorem problem_equivalence :
  (even_function (λ x : ℝ, Real.cos (x / 2)) ∧ 
   monotonically_decreasing_on (λ x : ℝ, Real.cos (x / 2)) (set.Ioo 0 2)) :=
begin
  sorry
end

end problem_equivalence_l263_263310


namespace non_congruent_triangles_with_perimeter_11_l263_263108

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263108


namespace solution_set_l263_263220

open Set Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_two : f 2 = 0
axiom f_cond : ∀ x : ℝ, 0 < x → x * (deriv (deriv f) x) + f x < 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = Ioo (-2 : ℝ) 0 ∪ Ioo 0 2 :=
by
  sorry

end solution_set_l263_263220


namespace max_value_f_on_interval_l263_263626

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

theorem max_value_f_on_interval :
  ∀ x ∈ Icc (0 : ℝ) 1, f x ≤ 1 :=
by
  sorry -- proof to be filled in later

end max_value_f_on_interval_l263_263626


namespace daniel_paid_more_l263_263702

noncomputable def num_slices : ℕ := 10
noncomputable def plain_cost : ℕ := 10
noncomputable def truffle_extra_cost : ℕ := 5
noncomputable def total_cost : ℕ := plain_cost + truffle_extra_cost
noncomputable def cost_per_slice : ℝ := total_cost / num_slices

noncomputable def truffle_slices_cost : ℝ := 5 * cost_per_slice
noncomputable def plain_slices_cost : ℝ := 5 * cost_per_slice

noncomputable def daniel_cost : ℝ := 5 * cost_per_slice + 2 * cost_per_slice
noncomputable def carl_cost : ℝ := 3 * cost_per_slice

noncomputable def payment_difference : ℝ := daniel_cost - carl_cost

theorem daniel_paid_more : payment_difference = 6 :=
by 
  sorry

end daniel_paid_more_l263_263702


namespace reflection_line_eq_y_neg4_l263_263651

def point := ℝ × ℝ

constant P Q R P' Q' R' : point

axiom hP : P = (-3, 1)
axiom hQ : Q = (5, -2)
axiom hR : R = (2, 7)
axiom hP' : P' = (-3, -9)
axiom hQ' : Q' = (5, -8)
axiom hR' : R' = (2, -3)

theorem reflection_line_eq_y_neg4 :
  ∃ M : ℝ → point → point, 
    (∀ x, M (x, 1) = (x, -9)) ∧ 
    (∀ x, M (x, -2) = (x, -8)) ∧ 
    (∀ x, M (x, 7) = (x, -3)) ∧ 
    ∀ x, M (x, x) = (x, 2 * -4 - x) :=
  sorry

end reflection_line_eq_y_neg4_l263_263651


namespace is_isosceles_triangle_l263_263523

-- Definitions for triangle and sides
variables (A B C : ℝ) (a b c : ℝ)

-- Condition and required type of triangle
theorem is_isosceles_triangle (h : a * Real.cos C + c * Real.cos A = c) :
  ∠B = ∠C → a = c :=
sorry

end is_isosceles_triangle_l263_263523


namespace length_median_AD_eq_l263_263518

-- Define the points A, B, C
def A := (4 : ℝ, 1 : ℝ)
def B := (7 : ℝ, 5 : ℝ)
def C := (-4 : ℝ, 7 : ℝ)

-- Midpoint function (general definition)
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate midpoint D of side BC
def D := midpoint B C

-- Distance function (general definition)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Prove that the length of median AD is equal to 5/2 * sqrt(5)
theorem length_median_AD_eq : distance A D = (5 / 2 : ℝ) * real.sqrt 5 :=
by
  sorry

end length_median_AD_eq_l263_263518


namespace max_two_digit_number_divisible_by_23_l263_263354

theorem max_two_digit_number_divisible_by_23 :
  ∃ n : ℕ, 
    (n < 100) ∧ 
    (1000 ≤ n * 109) ∧ 
    (n * 109 < 10000) ∧ 
    (n % 23 = 0) ∧ 
    (n / 23 < 10) ∧ 
    (n = 69) :=
by {
  sorry
}

end max_two_digit_number_divisible_by_23_l263_263354


namespace triangular_formula_l263_263278

noncomputable def triangular_nums : ℕ → ℕ
| 0       := 0
| (k + 1) := triangular_nums k + (k + 1)

theorem triangular_formula (n : ℕ) : triangular_nums n = n * (n + 1) / 2 := 
sorry

end triangular_formula_l263_263278


namespace road_length_l263_263261

theorem road_length (n : ℕ) (d : ℕ) (trees : ℕ) (intervals : ℕ) (L : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 10) 
  (h3 : trees = 10) 
  (h4 : intervals = trees - 1) 
  (h5 : L = intervals * d) : 
  L = 90 :=
by
  sorry

end road_length_l263_263261


namespace train_cross_pole_time_l263_263326

noncomputable def L_train : ℝ := 300 -- Length of the train in meters
noncomputable def L_platform : ℝ := 870 -- Length of the platform in meters
noncomputable def t_platform : ℝ := 39 -- Time to cross the platform in seconds

theorem train_cross_pole_time
  (L_train : ℝ)
  (L_platform : ℝ)
  (t_platform : ℝ)
  (D : ℝ := L_train + L_platform)
  (v : ℝ := D / t_platform)
  (t_pole : ℝ := L_train / v) :
  t_pole = 10 :=
by sorry

end train_cross_pole_time_l263_263326


namespace line_equation_l263_263985

theorem line_equation (l : Line) (A B : Point) (x_intercept : ℝ)
  (h1 : x_intercept = 1)
  (hA : A = (-2:ℝ, -1:ℝ))
  (hB : B = (4:ℝ, 5:ℝ))
  (h_distances : distance A l = distance B l) :
  (l.equation = "x = 1" ∨ l.equation = "y = x - 1") := sorry

end line_equation_l263_263985


namespace part1_part2_l263_263429

namespace Problem

-- Defining given conditions
def isOnParabola (p x y : ℝ) : Prop := y ^ 2 = 2 * p * x

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def parabolicFocus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def directrixX (p : ℝ) : ℝ := -p / 2

def distanceToDirectrix (x p : ℝ) : ℝ :=
  Real.abs (x + p / 2)

def perp (k1 k2 : ℝ) : Prop := k1 * k2 = -1

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
 ( (x1 + x2) / 2, (y1 + y2) / 2)

-- Proof Statements
theorem part1 (m p : ℝ) : 
  isOnParabola p 1 m ∧ distance 1 m (p / 2) 0 = 2 → p = 2 ∧ m = 2 :=
by
  sorry

theorem part2 (y1 y2 : ℝ) :
  isOnParabola 2 (y1 ^ 2 / 4) y1 ∧ isOnParabola 2 (y2 ^ 2 / 4) y2 ∧
  perp
    ((y1 - 2) / ((y1 ^ 2 / 4) - 1))
    ((y2 - 2) / ((y2 ^ 2 / 4) - 1)) ∧ 
  distanceToDirectrix ((midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2).fst) 2 = 15 / 2
  → (midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2) = (13 / 2, 1) ∨ 
    (midpoint (y1 ^ 2 / 4) y1 (y2 ^ 2 / 4) y2) = (13 / 2, -3) :=
by
  sorry

end Problem

end part1_part2_l263_263429


namespace concentration_proof_l263_263363

-- Define the concentrations and conditions
def c_H_plus : ℝ := 5.0e-7
def pH_values_same : Prop := ∀ (NaHCO3 CH3COONa NaClO : ℝ), true -- Assume this condition to be true for simplicity
def acidity_order : ℝ -> ℝ -> ℝ -> Prop := 
  λ CH3COOH HClO HCO3 => CH3COOH > HClO ∧ HClO > HCO3

-- The proof problem statement
theorem concentration_proof (h : ℝ) (NaHCO3 CH3COONa NaClO CH3COOH HClO HCO3 : ℝ)
    (h_neutral : c_H_plus = h) 
    (h_pH_same : pH_values_same NaHCO3 CH3COONa NaClO)
    (h_acidity : acidity_order CH3COOH HClO HCO3) :
  c_H_plus = c_OH_minus ∧ CH3COONa > NaHCO3 ∧ NaHCO3 > NaClO :=
by
  -- Proof body goes here
  sorry

-- Example use
#check concentration_proof -- Ensure the theorem type-checks correctly within Lean.

end concentration_proof_l263_263363


namespace minimum_flights_per_city_l263_263168

theorem minimum_flights_per_city (n : ℕ) :
  (∀ (cities : Fin 100 → Fin n), 
     (∀ (i : Fin 100), ∃ k, k ≤ n ∧ 
     ∀ j ≠ i, ((∃ f1 f2, f1 ≠ f2 ∧ cities i f1 = j ∧ cities f1 f2 ≠ j) ∨ 
               (∃ f1 f2, f1 ≠ f2 ∧ cities f2 f1 = j ∧ cities f2 i ≠ j)))) →
     (Σ' (route : Fin 100 → Fin n), (∃ (x : Fin 100) (y : Fin 100), x ≠ y ∧ 
     (Σ' (k : Fin n), ∀ i, route x k ≠ y ∧ route k y ≠ x)) = 1000) → n ≥ 4 := 
begin
  sorry
end

end minimum_flights_per_city_l263_263168


namespace percent_decrease_square_area_l263_263881

theorem percent_decrease_square_area :
  let area_A := 50 * Real.sqrt 3,
      area_C := 18 * Real.sqrt 3,
      area_B := 50,
      side_square_original := Real.sqrt area_B
  in 19% = ((area_B - (0.9 * side_square_original)^2) / area_B) * 100 := by
  have side_square := Real.sqrt area_B
  have new_area := (0.9 * side_square)^2
  have percent_decrease := ((area_B - new_area) / area_B) * 100
  sorry

end percent_decrease_square_area_l263_263881


namespace parabola_min_a_l263_263260

variable (a b c : ℚ)

theorem parabola_min_a (h_vertex : ∃ a, ∃ b, ∃ c, ∀ x, y = a * (x - (3/5))^2 - (13/5) )
  (h_equation : ∀ x, y = a*x^2 + b*x + c)
  (h_pos_a : 0 < a)
  (h_cond : ∃ n : ℤ, 2*a + b + 3*c = n)
  : a = 45/19 := 
sorry

end parabola_min_a_l263_263260


namespace remainder_mod_500_l263_263773

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l263_263773


namespace area_of_smaller_circle_l263_263721

theorem area_of_smaller_circle (r R : ℝ) (PA AB : ℝ) 
  (h1 : R = 2 * r) (h2 : PA = 4) (h3 : AB = 4) :
  π * r^2 = 2 * π :=
by
  sorry

end area_of_smaller_circle_l263_263721


namespace complex_power_sum_2013_l263_263253

noncomputable def complexPowerSum : ℂ :=
  let i := complex.I
  finset.sum (finset.range 2014) (λ n, i ^ n)

theorem complex_power_sum_2013 : complexPowerSum = 1 + complex.I :=
  sorry

end complex_power_sum_2013_l263_263253


namespace sum_powers_of_i_l263_263257

def pow_i_cycle : ℕ → ℂ
| 0 => 1
| 1 => complex.I
| 2 => -1
| 3 => -complex.I
| (n + 4) => pow_i_cycle n

theorem sum_powers_of_i : (i_sum : ℂ) → (i_sum = ∑ n in finset.range 2014, pow_i_cycle n) ∧ i_sum = 1 + complex.I :=
by
  existsi ((∑ n in finset.range 2014, pow_i_cycle n) : ℂ)
  split
  · exact rfl
  · sorry

end sum_powers_of_i_l263_263257


namespace count_integers_abs_leq_4_l263_263853

theorem count_integers_abs_leq_4 : 
  let solution_set := {x : Int | |x - 3| ≤ 4}
  ∃ n : Nat, n = 9 ∧ (∀ x ∈ solution_set, x ∈ finset.range 9) := sorry

end count_integers_abs_leq_4_l263_263853


namespace luke_can_see_silvia_for_22_point_5_minutes_l263_263574

/--
Luke is initially 0.75 miles behind Silvia. Luke rollerblades at 10 mph and Silvia cycles 
at 6 mph. Luke can see Silvia until she is 0.75 miles behind him. Prove that Luke can see 
Silvia for a total of 22.5 minutes.
-/
theorem luke_can_see_silvia_for_22_point_5_minutes :
    let distance := (3 / 4 : ℝ)
    let luke_speed := (10 : ℝ)
    let silvia_speed := (6 : ℝ)
    let relative_speed := luke_speed - silvia_speed
    let time_to_reach := distance / relative_speed
    let total_time := 2 * time_to_reach * 60 
    total_time = 22.5 :=
by
    sorry

end luke_can_see_silvia_for_22_point_5_minutes_l263_263574


namespace unique_two_points_l263_263462

theorem unique_two_points (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f x = x^3 - 3 * x + a) :=
by
  let f := λ x : ℝ, x^3 - 3 * x + a
  have key_condition : a = 2 / real.sqrt 3 := sorry
  exact sorry

end unique_two_points_l263_263462


namespace comprehensive_score_correct_l263_263689

def comprehensive_score
  (study_score hygiene_score discipline_score participation_score : ℕ)
  (study_weight hygiene_weight discipline_weight participation_weight : ℚ) : ℚ :=
  study_score * study_weight +
  hygiene_score * hygiene_weight +
  discipline_score * discipline_weight +
  participation_score * participation_weight

theorem comprehensive_score_correct :
  let study_score := 80
  let hygiene_score := 90
  let discipline_score := 84
  let participation_score := 70
  let study_weight := 0.4
  let hygiene_weight := 0.25
  let discipline_weight := 0.25
  let participation_weight := 0.1
  comprehensive_score study_score hygiene_score discipline_score participation_score
                      study_weight hygiene_weight discipline_weight participation_weight
  = 82.5 :=
by 
  sorry

#eval comprehensive_score 80 90 84 70 0.4 0.25 0.25 0.1  -- output should be 82.5

end comprehensive_score_correct_l263_263689


namespace power_modulo_calculation_l263_263778

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l263_263778


namespace sufficient_not_necessary_condition_l263_263822

variable (x y : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 1 ∧ y > 1) → (x + y > 2 ∧ x * y > 1) ∧
  ¬((x + y > 2 ∧ x * y > 1) → (x > 1 ∧ y > 1)) :=
by
  sorry

end sufficient_not_necessary_condition_l263_263822


namespace product_equality_l263_263244

variables (a b c : ℝ)

def x := (a - b) / (a + b)
def y := (b - c) / (b + c)
def z := (c - a) / (c + a)

theorem product_equality (ha : a ≠ -b) (hb : b ≠ -c) (hc : c ≠ -a) :
  (1 + x a b) * (1 + y b c) * (1 + z c a) = (1 - x a b) * (1 - y b c) * (1 - z c a) :=
sorry

end product_equality_l263_263244


namespace smallest_multiple_of_15_with_digits_8_or_0_div_15_l263_263984

def smallest_multiple_of_15_with_digits_8_or_0 : ℕ :=
  8880

theorem smallest_multiple_of_15_with_digits_8_or_0_div_15 :
  smallest_multiple_of_15_with_digits_8_or_0 / 15 = 592 :=
by
  rw [smallest_multiple_of_15_with_digits_8_or_0]
  norm_num

end smallest_multiple_of_15_with_digits_8_or_0_div_15_l263_263984


namespace nth_derived_sequence_bound_l263_263432

noncomputable def initialSequence (n : ℕ) : ℕ → ℝ
| i => if 1 ≤ i ∧ i ≤ n then 1 / (i : ℝ) else 0

noncomputable def derivedSequence : List ℝ → List ℝ
| [] => []
| [a] => [a]
| a :: b :: l => (a + b) / 2 :: derivedSequence (b :: l)

def nthDerivedSequence (l : List ℝ) (k : ℕ) : List ℝ :=
  match k with
  | 0 => l
  | k + 1 => nthDerivedSequence (derivedSequence l) k

theorem nth_derived_sequence_bound (n : ℕ) :
  let seq := List.ofFn (initialSequence n)
  let x := (nthDerivedSequence seq (n - 1)).headD 0
  x < 2 / (n : ℝ) := by
  sorry

end nth_derived_sequence_bound_l263_263432


namespace g_neg2_l263_263555

def g (x : ℝ) (a b : ℝ) := a * x ^ 3 + b / x - 2

theorem g_neg2 (a b : ℝ) (h1 : g 2 a b = 2) : g (-2) a b = -6 :=
sorry

end g_neg2_l263_263555


namespace maximize_AD_in_triangle_l263_263199

theorem maximize_AD_in_triangle
  (a : ℝ)
  (triangle_ABC : triangle ABC)
  (AB : ℝ)
  (AC : ℝ)
  (B : ℝ)
  (C : ℝ)
  (D : ℝ)
  (is_equilateral_triangle_BCD : is_equilateral BCD)
  (h1 : AB = a)
  (h2 : AC = a) :
  angle BAC = 120 :=
sorry

end maximize_AD_in_triangle_l263_263199


namespace find_pink_highlighters_l263_263164

def yellow_highlighters : ℕ := 7
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 15

theorem find_pink_highlighters : (total_highlighters - (yellow_highlighters + blue_highlighters)) = 3 :=
by
  sorry

end find_pink_highlighters_l263_263164


namespace rebecca_tent_stakes_l263_263593

theorem rebecca_tent_stakes :
  ∃ T : ℕ, let drink_mix := 2 * T,
              bottles_water := T + 2,
              cans_food := T / 2 
          in T + drink_mix + bottles_water + (cans_food : ℕ) = 32 ∧ T = 6 :=
begin
  sorry
end

end rebecca_tent_stakes_l263_263593


namespace simplest_fraction_sum_l263_263275

theorem simplest_fraction_sum (a b : ℕ) (h : Rat.mkP 428125 1000000 = Rat.mkP a b) (h_coprime : Nat.coprime a b) : a + b = 457 := 
sorry

end simplest_fraction_sum_l263_263275


namespace line_through_P_and_D_divides_shape_equally_l263_263895

-- Definitions based on conditions
def unit_square_shape : ℕ := 9  -- The shape has a total area of 9 unit squares
def midpoint (a b : Point) : Point := { x := (a.x + b.x) / 2, y := (a.y + b.y) / 2 }
def point_A : Point := { x := 0, y := 0  }
def point_C : Point := { x := 2, y := 2 }
def point_E : Point := { x := 4, y := 0 }
def point_B : Point := midpoint point_A point_C
def point_D : Point := midpoint point_C point_E
def point_P : Point := ...

-- Problem statement
theorem line_through_P_and_D_divides_shape_equally (P D : Point) :
  (line P D).divides_shape_equality unit_square_shape :=
sorry

end line_through_P_and_D_divides_shape_equally_l263_263895


namespace math_problem_solution_l263_263427

noncomputable section

def parabola_condition (p m : ℝ) : Prop :=
  ∃ M : ℝ × ℝ, M = (1, m) ∧ (m^2 = 2 * p * 1)

def distance_to_focus_condition (p m : ℝ) : Prop :=
  ∃ F : ℝ × ℝ, F = (p / 2, 0) ∧ (sqrt ((1 - p / 2)^2 + m^2) = 2)

def perpendicular_condition (y1 y2 : ℝ) : Prop :=
  let k1 := (y1 - 2) / ((y1^2 / 2) - 1)
  let k2 := (y2 - 2) / ((y2^2 / 2) - 1)
  k1 * k2 = -1

def midpoint_condition (x₀ : ℝ) : Prop :=
  x₀ + 1 = 15 / 2

def find_p_m_and_D : Prop :=
  parabola_condition 2 2 ∧
  distance_to_focus_condition 2 2 ∧
  (∀ y1 y2 : ℝ, perpendicular_condition y1 y2 → 
    ∃ D : ℝ × ℝ, D = (13 / 2, 1) ∨ D = (13 / 2, -3))

theorem math_problem_solution : find_p_m_and_D := by
  sorry

end math_problem_solution_l263_263427


namespace hannah_dogs_food_total_l263_263473

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l263_263473


namespace milk_leftover_after_milkshakes_l263_263368

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end milk_leftover_after_milkshakes_l263_263368


namespace sine_sum_ge_one_l263_263563

theorem sine_sum_ge_one {n : ℕ} (hn : n ≥ 1) (x : Fin n → ℝ)
  (hx : ∀ j, 0 ≤ x j ∧ x j ≤ Real.pi)
  (odd_sum_cos : Odd ((Finset.univ : Finset (Fin n)).sum (λ j, (Real.cos (x j) + 1)))) :
  1 ≤ (Finset.univ : Finset (Fin n)).sum (λ j, Real.sin (x j)) :=
sorry

end sine_sum_ge_one_l263_263563


namespace locus_of_points_P_T_l263_263437

variable {P T : Type}
open_locale classical

noncomputable def is_equilateral_triangle (A B C : P) : Prop :=
∃ a : ℝ, a > 0 ∧ (dist A B = a ∧ dist B C = a ∧ dist C A = a)

theorem locus_of_points_P_T
  {A B C D E P T : P} (h_eq_tri : is_equilateral_triangle A B C)
  (h_line_l : ∃ l : P → Prop, l B)
  (h_perpendicular : ∀ (p : P), ∃ l' : P → Prop, l' ⟂ h_line_l ∧ l' p)
  (h_distinct : D ≠ E)
  (h_eq_tri_DEP : is_equilateral_triangle D E P)
  (h_eq_tri_DET : is_equilateral_triangle D E T) :
  ∃ (O : P) (r : ℝ), ∀ (X : P), dist B X = r ↔ (X = P ∨ X = T) :=
sorry

end locus_of_points_P_T_l263_263437


namespace part1_extreme_value_part2_range_of_a_l263_263461

-- Part (1): Extreme value when a = 0
theorem part1_extreme_value :
  ∀ (x : ℝ), 0 < x → 
  (∀ y : ℝ, 0 < y → (y ≠ x → (f y < f x))) ∧ f x = - 1 / (Real.exp 2)
:= 
sorry

-- Part (2): Range of a for f(x) ≥ 1
theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x ≥ 1) ↔ (a ≥ 1 / (Real.exp 2))
:= 
sorry

-- Definitions for f(x) in Part (1)
def f (x : ℝ) : ℝ := 
  (1 - Real.log x) / x

-- Definitions for f(x) with parameter a in Part (2)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp x + (1 - Real.log x) / x

end part1_extreme_value_part2_range_of_a_l263_263461


namespace cos_theta_correct_point_not_on_first_line_l263_263700

-- Define the two direction vectors
def direction_vector1 : ℝ × ℝ := (4, -1)
def direction_vector2 : ℝ × ℝ := (-2, 5)

-- Compute and define the cosine of the angle between the two lines
noncomputable def cos_theta : ℝ :=
  let dot_product := direction_vector1.1 * direction_vector2.1 + direction_vector1.2 * direction_vector2.2 in
  let norm1 := Real.sqrt (direction_vector1.1^2 + direction_vector1.2^2) in
  let norm2 := Real.sqrt (direction_vector2.1^2 + direction_vector2.2^2) in
  dot_product / (norm1 * norm2)

-- Assertion for the cosine of the angle
theorem cos_theta_correct : cos_theta = -13 / Real.sqrt 493 := by
  sorry

-- Define the parameterization of the first line
def first_line (s : ℝ) : ℝ × ℝ :=
  (2 + 4 * s, 1 - s)

-- Define the point to check
def point : ℝ × ℝ := (5, 0)

-- Assertion for the point not lying on the first line
theorem point_not_on_first_line : ¬ ∃ s : ℝ, first_line(s) = point := by
  sorry

end cos_theta_correct_point_not_on_first_line_l263_263700


namespace inequality_solution_set_l263_263074

theorem inequality_solution_set (a x : ℝ) (h : 4^a = 2^(a + 2)) :
  {x | 2^(2 * x + 1) > 2^(x - 1)} = {x | x > -2} :=
sorry

end inequality_solution_set_l263_263074


namespace find_valid_number_l263_263031

noncomputable def is_valid_number (n : ℕ) : Prop :=
  ∃ pairs : list (ℕ × ℕ), (∀ pair ∈ pairs, pair.1 < pair.2) ∧
  (∀ pair ∈ pairs, pair.2 - pair.1 = 545) ∧
  (∀ pair ∈ pairs, gcd n pair.1 = pair.1) ∧
  (∀ pair ∈ pairs, gcd n pair.2 = pair.2) ∧
  (n > 1)

theorem find_valid_number :
  ∀ n : ℕ, is_valid_number n ↔ n = 1094 :=
by
  sorry

end find_valid_number_l263_263031


namespace product_of_two_numbers_l263_263269

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h₁ : x - y = 8) 
  (h₂ : x^2 + y^2 = 160) 
  : x * y = 48 := 
sorry

end product_of_two_numbers_l263_263269


namespace product_of_areas_eq_576V_squared_l263_263061

-- Define the original dimensions and the volume
variables (a b c : ℝ)
def V : ℝ := a * b * c

-- Define the scaled dimensions
def scaled_a : ℝ := 2 * a
def scaled_b : ℝ := 3 * b
def scaled_c : ℝ := 4 * c

-- Define the areas
def bottom_area : ℝ := scaled_a * scaled_b
def side_area : ℝ := scaled_b * scaled_c
def front_area : ℝ := scaled_c * scaled_a

-- Theorem to prove product of areas equals 576V^2
theorem product_of_areas_eq_576V_squared (a b c : ℝ) :
  bottom_area a b c * side_area a b c * front_area a b c = 576 * (V a b c) ^ 2 :=
by

  sorry

end product_of_areas_eq_576V_squared_l263_263061


namespace required_CaO_for_CaOH2_l263_263763

def molar_ratio (x y : ℕ) : Prop := x = y

theorem required_CaO_for_CaOH2 : 
    ∀ (CaO H2O CaOH2: ℕ), (H2O = 2 ∧ CaOH2 = 2) → (CaOH2 = CaO) → (CaO = 2) :=
by
  intros CaO H2O CaOH2 h
  cases h with h1 h_ratio
  sorry

end required_CaO_for_CaOH2_l263_263763


namespace parallelogram_area_l263_263388

variables (p q : ℝ^3)
def a := p + 3 * q
def b := 3 * p - q

axiom norm_p : ‖p‖ = 3
axiom norm_q : ‖q‖ = 5
axiom angle_pq : real.angle (p, q) = 2 * real.pi / 3

theorem parallelogram_area : ‖(a × b)‖ = 75 * real.sqrt 3 :=
by sorry

end parallelogram_area_l263_263388


namespace triangle_inequality_l263_263065

theorem triangle_inequality (A B C : ℝ) :
  ∀ (a b c : ℝ), (a = 2 * Real.sin (A / 2) * Real.cos (A / 2)) ∧
                 (b = 2 * Real.sin (B / 2) * Real.cos (B / 2)) ∧
                 (c = Real.cos ((A + B) / 2)) ∧
                 (x = Real.sqrt (Real.tan (A / 2) * Real.tan (B / 2)))
                 → (Real.sqrt (a * b) / Real.sin (C / 2) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2)) := by {
  sorry
}

end triangle_inequality_l263_263065


namespace smallest_palindrome_base2_base4_l263_263003

-- Function to check if a number is a palindrome in a given base
def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n in digits = digits.reverse

theorem smallest_palindrome_base2_base4 (n : ℕ) (hn : n > 15) :
  is_palindrome n 2 ∧ is_palindrome n 4 → n = 85 :=
by sorry

end smallest_palindrome_base2_base4_l263_263003


namespace inequality_solution_set_l263_263079

variable (f : ℝ → ℝ)

theorem inequality_solution_set (h_deriv : ∀ x : ℝ, f' x - f x < 1)
  (h_initial : f 0 = 2022) :
  ∀ x : ℝ, f x + 1 > 2023 * Real.exp x ↔ x < 0 :=
by
  intro x
  sorry

end inequality_solution_set_l263_263079


namespace sum_remainder_of_consecutive_odds_l263_263303

theorem sum_remainder_of_consecutive_odds :
  (11075 + 11077 + 11079 + 11081 + 11083 + 11085 + 11087) % 14 = 7 :=
by
  -- Adding the proof here
  sorry

end sum_remainder_of_consecutive_odds_l263_263303


namespace total_money_is_correct_l263_263335

-- Define the values of different types of coins and the amount of each.
def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4
def cash : ℕ := 45

-- Define the total amount of money.
def total_money : ℕ :=
  (gold_count * gold_value) +
  (silver_count * silver_value) +
  (bronze_count * bronze_value) +
  (titanium_count * titanium_value) + cash

-- The proof statement
theorem total_money_is_correct : total_money = 1055 := by
  sorry

end total_money_is_correct_l263_263335


namespace range_of_x_l263_263897

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 2))) ↔ x > 2 :=
by
  sorry

end range_of_x_l263_263897


namespace non_congruent_triangles_with_perimeter_11_l263_263107

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263107


namespace compare_constants_l263_263806

noncomputable def a : ℝ := (1 / 2) ^ (1 / 3)
noncomputable def b : ℝ := log (1 / 2) (1 / 3)
noncomputable def c : ℝ := log (1 / 3) 2

theorem compare_constants : (c < a) ∧ (a < b) := 
by
  have ha : 0 < a := sorry
  have ha1 : a < 1 := sorry
  have hb : b > 1 := sorry
  have hc : c < 0 := sorry
  split
  show c < a
  from sorry
  show a < b
  from sorry

end compare_constants_l263_263806


namespace probability_no_aces_opposite_l263_263414

open Nat

-- Define the conditions
def players : ℕ := 4
def total_cards : ℕ := 32
def cards_per_player : ℕ := 32 / 4 -- 8 cards per player

-- Define the events
def event_A := choose 24 8 -- one player receives 8 of the 24 non-ace cards
def event_B := choose 20 8 -- another specified player receives 8 of the 20 remaining non-ace cards

-- Define the probability calculation
def conditional_probability := event_B.toRational / event_A.toRational

-- Final theorem stating that the conditional probability is equal to 130 / 759
theorem probability_no_aces_opposite : conditional_probability = 130 / 759 := sorry

end probability_no_aces_opposite_l263_263414


namespace A_coordinates_l263_263034

noncomputable def A_distance_equidistant : Prop :=
  ∃ x : ℝ, 
    (λ A B C : ℝ × ℝ × ℝ, 
      let AB := ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2)^0.5 in
      let AC := ((C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.3 - A.3)^2)^0.5 in
      AB = AC ∧ A.1 = x ∧ A.2 = 0 ∧ A.3 = 0)
    (x, 0, 0) (4, 6, 8) (2, 4, 6)

theorem A_coordinates :
  ∃ x, x = 15 ∧ A_distance_equidistant :=
by
  existsi (15 : ℝ)
  split
  · refl
  · sorry

end A_coordinates_l263_263034


namespace exists_univariate_polynomial_l263_263343

def polynomial_in_three_vars (P : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ,
  P x y z = P x y (x * y - z) ∧
  P x y z = P x (z * x - y) z ∧
  P x y z = P (y * z - x) y z

theorem exists_univariate_polynomial (P : ℝ → ℝ → ℝ → ℝ) (h : polynomial_in_three_vars P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x * y * z) :=
sorry

end exists_univariate_polynomial_l263_263343


namespace inverse_function_is_half_pow_l263_263273

def f (x : ℝ) : ℝ := (Real.log2 (1 / 2)) * (Real.log2 x)

theorem inverse_function_is_half_pow (y : ℝ) : f⁻¹ y = (1 / 2) ^ y :=
by
  sorry

end inverse_function_is_half_pow_l263_263273


namespace dan_initial_amount_l263_263738

theorem dan_initial_amount (left_amount : ℕ) (candy_cost : ℕ) : left_amount = 3 ∧ candy_cost = 2 → left_amount + candy_cost = 5 :=
by
  sorry

end dan_initial_amount_l263_263738


namespace terminal_side_angle_l263_263874

open Real

theorem terminal_side_angle (α : ℝ) (m n : ℝ) (h_line : n = 3 * m) (h_radius : m^2 + n^2 = 10) (h_sin : sin α < 0) (h_coincide : tan α = 3) : m - n = 2 :=
by
  sorry

end terminal_side_angle_l263_263874


namespace distinct_initial_values_finite_sequence_l263_263736

-- Definition of the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 6 * x

-- Definition of the sequence based on initial value x₀
def y₀ (x₀ : ℝ) := x₀
def y (n : ℕ) (x₀ : ℝ) : ℝ :=
  if n = 0 then y₀ x₀ else g (y (n - 1) x₀)

-- Theorem to prove the number of distinct initial values leading to a sequence with finite distinct values
theorem distinct_initial_values_finite_sequence : 
  { x₀ : ℝ | ∃ N : ℕ, ∀ m n ≥ N, y m x₀ = y n x₀ }.finite.card = 3 :=
sorry

end distinct_initial_values_finite_sequence_l263_263736


namespace inequality_always_true_l263_263668

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end inequality_always_true_l263_263668


namespace range_of_x_for_a_range_of_a_l263_263089

-- Define propositions p and q
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (I)
theorem range_of_x_for_a (a x : ℝ) (ha : a = 1) (hpq : prop_p a x ∧ prop_q x) : 2 < x ∧ x < 3 :=
by
  sorry

-- Part (II)
theorem range_of_a (p q : ℝ → Prop) (hpq : ∀ x : ℝ, ¬p x → ¬q x) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_for_a_range_of_a_l263_263089


namespace area_of_wrapping_paper_l263_263692

theorem area_of_wrapping_paper (l w h: ℝ) (l_pos: 0 < l) (w_pos: 0 < w) (h_pos: 0 < h) :
  ∃ s: ℝ, s = l + w ∧ s^2 = (l + w)^2 :=
by 
  sorry

end area_of_wrapping_paper_l263_263692


namespace ab_value_l263_263865

theorem ab_value (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b = 33) : a * b = 24 := 
by 
  sorry

end ab_value_l263_263865


namespace general_terms_a_b_sum_first_n_terms_lambda_range_l263_263075

noncomputable theory

def b_n (n : ℕ) : ℝ := 2^(n-1)

def a_n (n : ℕ) : ℝ := n / 2^(n-1)

def T_n (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, a_n (k+1))

theorem general_terms_a_b :
  ∀ n : ℕ, a_n n = n / 2^(n-1) ∧ b_n n = 2^(n-1) := sorry

theorem sum_first_n_terms :
  ∀ n : ℕ, T_n n = 4 - (2 + n) / 2^(n-1) := sorry

theorem lambda_range :
  ∀ n : ℕ, (-1) ^ n * (λ : ℝ) < T_n n ↔ λ ∈ set.Ioc (-1 : ℝ) 2 := sorry

end general_terms_a_b_sum_first_n_terms_lambda_range_l263_263075


namespace power_modulo_calculation_l263_263780

open Nat

theorem power_modulo_calculation :
  let λ500 := 100
  let λ100 := 20
  (5^5 : ℕ) ≡ 25 [MOD 100]
  (125^5 : ℕ) ≡ 125 [MOD 500]
  (5^{5^{5^5}} : ℕ) % 500 = 125 :=
by
  let λ500 := 100
  let λ100 := 20
  have h1 : (5^5 : ℕ) ≡ 25 [MOD 100] := by sorry
  have h2 : (125^5 : ℕ) ≡ 125 [MOD 500] := by sorry
  sorry

end power_modulo_calculation_l263_263780


namespace dirk_profit_is_362_l263_263752

-- Conditions from the problem
def sales_day1_typeA := 20
def sales_day1_typeB := 8
def sales_day2_typeA := 12
def sales_day2_typeB := 6
def sales_day3_typeA := 15
def sales_day3_typeB := 9

def cost_per_am_typeA := 30
def cost_per_am_typeB := 35
def sell_price_per_am_typeA := 40
def sell_price_per_am_typeB := 50

def faire_fee_percentage := 0.10
def stand_rental_fee := 150

-- Define total revenue calculation
def total_revenue : ℕ :=
  (sales_day1_typeA * sell_price_per_am_typeA + sales_day1_typeB * sell_price_per_am_typeB) +
  (sales_day2_typeA * sell_price_per_am_typeA + sales_day2_typeB * sell_price_per_am_typeB) +
  (sales_day3_typeA * sell_price_per_am_typeA + sales_day3_typeB * sell_price_per_am_typeB)

-- Define total cost calculation
def total_cost : ℕ :=
  (sales_day1_typeA * cost_per_am_typeA + sales_day1_typeB * cost_per_am_typeB) +
  (sales_day2_typeA * cost_per_am_typeA + sales_day2_typeB * cost_per_am_typeB) +
  (sales_day3_typeA * cost_per_am_typeA + sales_day3_typeB * cost_per_am_typeB)

-- Define faire fee calculation
def faire_fee : ℕ := (total_revenue * faire_fee_percentage).toNat

-- Define the function to calculate profit
def profit : ℕ :=
  total_revenue - total_cost - faire_fee - stand_rental_fee

-- The proof statement that Dirk's profit is $362
theorem dirk_profit_is_362 : profit = 362 :=
  by
    sorry

end dirk_profit_is_362_l263_263752


namespace min_value_x2_y2_z2_l263_263920

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 3 :=
sorry

end min_value_x2_y2_z2_l263_263920


namespace plums_picked_total_l263_263358

theorem plums_picked_total :
  let alyssa_first_hour := 17
  let jason_first_hour := 10
  let alyssa_second_hour := 2 * alyssa_first_hour
  let jason_second_hour := (3 / 2) * jason_first_hour
  let combined_third_hour := 38
  let dropped_third_hour := 6
  let kept_third_hour := combined_third_hour - dropped_third_hour
  in alyssa_first_hour + jason_first_hour + alyssa_second_hour + jason_second_hour + kept_third_hour = 108 :=
by
  let alyssa_first_hour := 17
  let jason_first_hour := 10
  let alyssa_second_hour := 2 * alyssa_first_hour
  let jason_second_hour := (3 / 2) * jason_first_hour
  let combined_third_hour := 38
  let dropped_third_hour := 6
  let kept_third_hour := combined_third_hour - dropped_third_hour
  have total := alyssa_first_hour + jason_first_hour + alyssa_second_hour + jason_second_hour + kept_third_hour
  calc total = (17 + 10) + (2 * 17 + (3 / 2) * 10) + (38 - 6) : by sorry
             ... = 27 + 49 + 32 : by sorry
             ... = 108 : by sorry

end plums_picked_total_l263_263358


namespace find_m_l263_263040

theorem find_m (m n : ℤ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 0 :=
sorry

end find_m_l263_263040


namespace P_Q_identity_implies_PPx_eq_QQx_no_real_solution_l263_263922

noncomputable theory

open Function

theorem P_Q_identity_implies_PPx_eq_QQx_no_real_solution
  (P Q : ℝ → ℝ)
  (polynomial_P : polynomial P)
  (polynomial_Q : polynomial Q)
  (H1 : ∀ x : ℝ, P (Q x) = Q (P x))
  (H2 : ∀ x : ℝ, P x ≠ Q x) :
  ∀ x : ℝ, P (P x) ≠ Q (Q x) :=
by
  sorry -- Proof omitted per instructions.

end P_Q_identity_implies_PPx_eq_QQx_no_real_solution_l263_263922


namespace inequality_proof_l263_263557

open scoped BigOperators

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 1 / 2) :
  (∑ i, (a i)^2 / (∑ i, a i)^2) ≥ (∑ i, (1 - a i)^2 / (∑ i, (1 - a i))^2) := 
by 
  sorry

end inequality_proof_l263_263557


namespace intersect_C2_C3_max_distance_C1_intersections_max_AB_value_l263_263185

-- Define the conditions
def curve_C1 (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)
def curve_C2_pol (θ : ℝ) : ℝ := 2 * Real.sin θ
def curve_C3_pol (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.cos θ

-- Cartesian forms of C2 and C3
def curve_C2_cart (x y : ℝ) : Prop := x^2 + y^2 = 2 * y
def curve_C3_cart (x y : ℝ) : Prop := x^2 + y^2 = 2 * Real.sqrt 3 * x

-- Problem part (I)
theorem intersect_C2_C3 : 
  {p : ℝ × ℝ // curve_C2_cart p.1 p.2 ∧ curve_C3_cart p.1 p.2} = 
  {⟨0, 0⟩, ⟨(Real.sqrt 3) / 2, 3 / 2⟩} := 
sorry

-- Problem part (II)
theorem max_distance_C1_intersections :
  ∀ (α : ℝ), 0 ≤ α ∧ α < Real.pi → 
  let A := (2 * Real.sin α, α) in
  let B := (2 * Real.sqrt 3 * Real.cos α, α) in
  ∥A.1 - B.1∥ = 4 * ∥Real.sin (α - Real.pi / 3)∥ :=
sorry

theorem max_AB_value : 
  let α := (5 * Real.pi) / 6 in 
  ∥ 2 * Real.sin α - 2 * Real.sqrt 3 * Real.cos α∥ = 4 :=
sorry

end intersect_C2_C3_max_distance_C1_intersections_max_AB_value_l263_263185


namespace find_RS_length_PQRS_l263_263889

noncomputable def RS_length (PQ QR PS: ℝ) (Angle_PSQ_congruent_Angle_PRQ : Prop)
  (Angle_PQR_congruent_Angle_QRS : Prop) : ℝ :=
if PQ = 7 ∧ QR = 9 ∧ PS = 5 ∧ Angle_PSQ_congruent_Angle_PRQ ∧ Angle_PQR_congruent_Angle_QRS then
  45 / 7
else
  0

theorem find_RS_length_PQRS :
  ∀ (PQ QR PS: ℝ) (Angle_PSQ_congruent_Angle_PRQ : Prop)
  (Angle_PQR_congruent_Angle_QRS : Prop),
  PQ = 7 ∧ QR = 9 ∧ PS = 5 ∧ Angle_PSQ_congruent_Angle_PRQ ∧ Angle_PQR_congruent_Angle_QRS →
  RS_length PQ QR PS Angle_PSQ_congruent_Angle_PRQ Angle_PQR_congruent_Angle_QRS = 45 / 7 :=
begin
  intros,
  -- Proof omitted
  sorry
end


end find_RS_length_PQRS_l263_263889


namespace correct_exponent_operation_l263_263311

open Real

theorem correct_exponent_operation (a : ℝ) : 
  (2 * a + 3 * a = 5 * a * a) ↔ False ∧ 
  ((a ^ 2) ^ 3 = a ^ 5) ↔ False ∧ 
  (a ^ 2 * a ^ 4 = a ^ 8) ↔ False ∧ 
  (a ^ 3 / a = a ^ 2) ↔ True :=
by
  sorry

end correct_exponent_operation_l263_263311


namespace dark_lord_squads_l263_263263

def total_weight : ℕ := 1200
def orcs_per_squad : ℕ := 8
def capacity_per_orc : ℕ := 15
def squads_needed (w n c : ℕ) : ℕ := w / (n * c)

theorem dark_lord_squads :
  squads_needed total_weight orcs_per_squad capacity_per_orc = 10 :=
by sorry

end dark_lord_squads_l263_263263


namespace mappings_count_A_to_B_l263_263072

open Finset

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 4}

theorem mappings_count_A_to_B : (card B) ^ (card A) = 4 :=
by
  -- This line will state that the proof is skipped for now.
  sorry

end mappings_count_A_to_B_l263_263072


namespace find_k_for_parallel_vectors_l263_263471

theorem find_k_for_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (9, k - 6)
  (1 * (k - 6) - 9 * k = 0) → k = -3 / 4 :=
by
  intros a b parallel_cond
  sorry

end find_k_for_parallel_vectors_l263_263471


namespace MarkBenchPressAmount_l263_263739

def DaveWeight : ℝ := 175
def DaveBenchPressMultiplier : ℝ := 3
def CraigBenchPressFraction : ℝ := 0.20
def MarkDeficitFromCraig : ℝ := 50

theorem MarkBenchPressAmount : 
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  MarkBenchPress = 55 := by
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  sorry

end MarkBenchPressAmount_l263_263739


namespace number_of_solutions_l263_263858

theorem number_of_solutions (a : ℝ) : 
  (a < (3 / 2) * real.cbrt 2) → (∃! x : ℝ, x^3 + 1 = a * x) ∧ 
  (a = (3 / 2) * real.cbrt 2) → (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂) ∧ 
  (a > (3 / 2) * real.cbrt 2) → (∃! x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₃^3 + 1 = a * x₃) :=
by 
  sorry

end number_of_solutions_l263_263858


namespace exists_right_angled_triangle_with_unique_digits_l263_263718

def is_unique_digit_set (sides : List ℕ) : Prop :=
  let digits := sides.bind (λ n => toDigits 10 n)
  (List.dedup digits).length = digits.length ∧
    ∀ digit, digit ∈ digits → digit < 10 ∧ digit ≥ 0

def is_right_angled (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem exists_right_angled_triangle_with_unique_digits :
  ∃ (a b c : ℕ), is_right_angled a b c ∧ is_unique_digit_set [a, b, c] :=
sorry

end exists_right_angled_triangle_with_unique_digits_l263_263718


namespace min_cost_garden_l263_263949

open_locale classical

def area_bottom_left : ℕ := 7 * 2
def area_top_left : ℕ := 5 * 5
def area_bottom_right : ℕ := 6 * 4
def area_middle_right : ℕ := 8 * 3
def area_top_right : ℕ := 8 * 3

def cost_sunflowers : ℝ := 0.75
def cost_tulips : ℝ := 1.25
def cost_orchids : ℝ := 1.75
def cost_roses : ℝ := 2
def cost_peonies : ℝ := 2.5

def total_cost := cost_peonies * area_bottom_left + cost_roses * area_top_right + 
                  cost_orchids * area_middle_right + cost_tulips * area_bottom_right +
                  cost_sunflowers * area_top_left

theorem min_cost_garden : total_cost = 173.75 := 
sorry

end min_cost_garden_l263_263949


namespace conic_section_eccentricity_l263_263828

theorem conic_section_eccentricity
  (m n p q : ℝ)
  (h1 : n > m)
  (h2 : m > 0)
  (h3 : p > 0)
  (h4 : q > 0)
  (shared_foci : ∃ F1 F2 : ℝ × ℝ, true) -- Placeholder for shared foci
  (common_point : ∃ M : ℝ × ℝ, M ∈ { (x, y) | mx^2 + ny^2 = 1 } ∧ M ∈ { (x, y) | px^2 - qy^2 = 1 }
                    ∧ ∃ F1 F2, angle_deg F1 M F2 = 90)
  (e_1 : ℝ)
  (h5 : e_1 = 3 / 4) :
  ∃ e_2, e_2 = 3 * Real.sqrt 2 / 2 :=
sorry

end conic_section_eccentricity_l263_263828


namespace teams_inequality_l263_263722

theorem teams_inequality (n : ℕ) (h : n ≥ 4) (attending : Finset (Fin 2n)) (teams : Finset (Finset (Fin 2n))) :
  (∀ couple in Finset.powersetLen 2 (attending), ∃! t ∈ teams, couple ⊆ t) →
  (∀ team ∈ teams, ∀ pair in Finset.powersetLen 2 team, ¬(pair ∈ Finset.powersetLen 2 (Finset.Ico n 2n))) →
  2 * n ≤ teams.card :=
sorry

end teams_inequality_l263_263722


namespace sin_double_angle_l263_263867

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2*θ) = 24/25 :=
by
  sorry

end sin_double_angle_l263_263867


namespace average_of_numbers_eq_x_l263_263281

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end average_of_numbers_eq_x_l263_263281


namespace limit_manipulation_l263_263153

variable {α : Type*} [RealNormedField α] [NormedSpace ℝ α]

theorem limit_manipulation (f : ℝ → α) (a b x₀ : ℝ) 
  (h_diff : ∀ x ∈ set.Ioo a b, differentiable_at ℝ f x)
  (h_x₀ : x₀ ∈ set.Ioo a b) :
  (tendsto (λ h, (f (x₀ + h) - f (x₀ - h)) / h) (nhds 0) (nhds (2 * (deriv f x₀)))) :=
begin
  sorry
end

end limit_manipulation_l263_263153


namespace Samia_walking_distance_l263_263595

-- Definitions and conditions
def average_biking_speed := 20 -- km per hour
def biking_fraction := 2 / 3
def average_walking_speed := 4 -- km per hour
def walking_fraction := 1 / 3
def total_time := 1 -- hour

-- Main theorem stating the question and the correct answer
theorem Samia_walking_distance : 
  (total_time = ((biking_fraction * d) / average_biking_speed + (walking_fraction * d) / average_walking_speed))
  → (walking_fraction * d = 2.9) :=
by
  sorry

end Samia_walking_distance_l263_263595


namespace du_chin_fraction_of_sales_l263_263023

theorem du_chin_fraction_of_sales :
  let pies := 200
  let price_per_pie := 20
  let remaining_money := 1600
  let total_sales := pies * price_per_pie
  let used_for_ingredients := total_sales - remaining_money
  let fraction_used_for_ingredients := used_for_ingredients / total_sales
  fraction_used_for_ingredients = (3 / 5) := by
    sorry

end du_chin_fraction_of_sales_l263_263023


namespace simplify_f_find_tan_alpha_l263_263824

variable (α x : ℝ)

-- Declaration of assumptions.
axiom angle_in_third_quadrant : π < α ∧ α < 3 * π / 2
def f (α x : ℝ) := 
  (sin (α - x / 2) * cos (3 * x / 2 + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-α - π))

-- Problem 1: Proving the simplified form of f(α)
theorem simplify_f (h : α ≠ 0) : f α x = -cos α := by
  sorry

-- Problem 2: Proving the value of tan(α)
theorem find_tan_alpha (h1 : f α x = 4/5) (h2 : π < α ∧ α < 3 * π / 2) : tan α = 3/4 := by
  sorry

end simplify_f_find_tan_alpha_l263_263824


namespace integer_solutions_count_count_integer_solutions_l263_263850

theorem integer_solutions_count (x : ℤ) :
  (x ∈ (set_of (λ x : ℤ, |x - 3| ≤ 4))) ↔ x ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7} :=
by sorry

theorem count_integer_solutions :
  (finset.card (finset.filter (λ x, |x - 3| ≤ 4) (finset.range 10))) = 9 :=
by sorry

end integer_solutions_count_count_integer_solutions_l263_263850


namespace exist_unique_rectangular_prism_Q_l263_263062

variable (a b c : ℝ) (h_lt : a < b ∧ b < c)
variable (x y z : ℝ) (hx_lt : x < y ∧ y < z ∧ z < a)

theorem exist_unique_rectangular_prism_Q :
  (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) ∧ (x < y ∧ y < z ∧ z < a) → 
  ∃! x y z, (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) :=
sorry

end exist_unique_rectangular_prism_Q_l263_263062


namespace triangle_count_with_perimeter_11_l263_263103

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263103


namespace find_a_l263_263156

variable {x y a : ℤ}

theorem find_a (h1 : 3 * x + y = 1 + 3 * a) (h2 : x + 3 * y = 1 - a) (h3 : x + y = 0) : a = -1 := 
sorry

end find_a_l263_263156


namespace meeting_point_distance_closer_A_l263_263654

/-
We will assume the conditions as hypotheses and the goal to prove that 
the meeting point is 31 miles closer to A than to B given the specified speeds and meeting time.
-/
theorem meeting_point_distance_closer_A (h : ℕ) : 
  (distance : ℝ) 
  (h_dist : distance = 100) 
  (speed_A : ℝ) (speed_A = 5) 
  (decrease_A : ℝ) (decrease_A = 0.4)
  (speed_B : ℝ) (speed_B = 4) 
  (increase_B : ℝ) (increase_B = 0.5)
  (meeting_hour : ℕ) (meeting_hour = 20) : 
  ∃ x : ℝ, x = 31 ∧ meeting_point distance speed_A decrease_A speed_B increase_B meeting_hour x := 
by
  sorry

end meeting_point_distance_closer_A_l263_263654


namespace find_S_n_expression_l263_263216

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  let rec a_aux : ℕ → ℝ
      | 0     := -1
      | (n+1) := S n * S (n+1)
    in a_aux n

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i+1)

theorem find_S_n_expression (n : ℕ) : S n = -(1 / n) :=
  sorry

end find_S_n_expression_l263_263216


namespace no_solution_l263_263159

-- Definitions following the given conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

def condition (x z : ℕ) : Prop :=
  is_prime x ∧
  is_prime z ∧
  let y := 2.134 * 10^x - 1 in
  Nat.floor y = y ∧
  is_prime (Nat.floor y) ∧
  z ∣ Nat.floor y ∧
  z < Real.log y ∧
  2.134 * 10^x < 21000

-- The final statement: There is no prime x that satisfies these conditions.
theorem no_solution : ∀ x z : ℕ, ¬ condition x z :=
by sorry

end no_solution_l263_263159


namespace count_newborns_l263_263495

theorem count_newborns 
  (prob_die_each_month : ℝ)
  (expected_survivors : ℝ)
  (prob_survival_each_month : ℝ := 1 - prob_die_each_month)
  (prob_survival_3_months : ℝ := prob_survival_each_month ^ 3)
  (N : ℝ := expected_survivors / prob_survival_3_months) : 
  prob_die_each_month = (1 / 10) → 
  expected_survivors = 510.3 → 
  N ≈ 700 :=
by
  sorry

end count_newborns_l263_263495


namespace equal_distance_IE_ID_l263_263211

noncomputable def triangle_abc : Type* := sorry
noncomputable def point (α : Type*) [nontrivial α] : Type* := sorry
noncomputable def segment_length_eq 
  {α : Type*} [linear_ordered_field α] (A B : point α) : α := sorry

theorem equal_distance_IE_ID
  (A B C D E I : point ℝ)
  (h_triangle_abc : triangle_abc)
  (h_angle_CAB : ∠A B C = 60)
  (hD_on_AC : D ∈ [A, C])
  (hE_on_AB : E ∈ [A, B])
  (h_angle_bisector_BD : angle_bisector B D (segment AC))
  (h_angle_bisector_CE : angle_bisector C E (segment AB))
  (hI_intersection : intersection_point (segment BD) (segment CE) = I) :
  segment_length_eq I D = segment_length_eq I E := sorry

end equal_distance_IE_ID_l263_263211


namespace average_time_within_storm_circle_l263_263712

-- Define the positions and speed of the truck and the storm center
def truck_position (t : ℝ) : ℝ × ℝ := (- 3 / 4 * t, 0)
def storm_center_position (t : ℝ) : ℝ × ℝ := (- t / 2, 130 - t / 2)
def storm_radius : ℝ := 60

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ := ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) ^ (1 / 2)

-- Define the condition that the truck is within the storm circle
def within_storm_circle (t : ℝ) : Prop :=
  distance (truck_position t) (storm_center_position t) ≤ storm_radius

-- The main theorem to prove
theorem average_time_within_storm_circle : 
  (∃ t1 t2 : ℝ, t1 < t2 ∧ 
    within_storm_circle t1 ∧ 
    within_storm_circle t2 ∧ 
    (∀ t, t1 ≤ t ∧ t ≤ t2 → within_storm_circle t) ∧ 
    1/2 * (t1 + t2) = 208) :=
sorry

end average_time_within_storm_circle_l263_263712


namespace remainder_of_exponentiation_is_correct_l263_263789

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l263_263789


namespace probability_not_fully_hearing_favorite_song_l263_263374

-- Define the conditions in Lean
def total_songs : Nat := 12
def time_increment : Nat := 20
def shortest_song : Nat := 20
def favorite_song_length : Nat := 240 -- 4 minutes in seconds

-- Calculate factorial function for permutations
noncomputable def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- The probability calculation problem
theorem probability_not_fully_hearing_favorite_song :
  let total_arrangements := factorial total_songs,
      non_favorable_scenarios := 3 * factorial 10
  in (total_arrangements - non_favorable_scenarios) / total_arrangements = 43 / 44 := sorry

end probability_not_fully_hearing_favorite_song_l263_263374


namespace ratio_out_of_state_to_in_state_l263_263733

/-
Given:
- total job applications Carly sent is 600
- job applications sent to companies in her state is 200

Prove:
- The ratio of job applications sent to companies in other states to the number sent to companies in her state is 2:1.
-/

def total_applications : ℕ := 600
def in_state_applications : ℕ := 200
def out_of_state_applications : ℕ := total_applications - in_state_applications

theorem ratio_out_of_state_to_in_state :
  (out_of_state_applications / in_state_applications) = 2 :=
by
  sorry

end ratio_out_of_state_to_in_state_l263_263733


namespace find_y_values_l263_263222

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = 0 ∨ y = 144 ∨ y = -24) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end find_y_values_l263_263222


namespace proof_inequality_l263_263810

noncomputable def proof_problem (x : ℝ) (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) : Prop :=
  let a := Real.log x
  let b := (1 / 2) ^ (Real.log x)
  let c := Real.exp (Real.log x)
  b > c ∧ c > a

theorem proof_inequality {x : ℝ} (Hx : x ∈ Set.Ioo (Real.exp (-1)) (1)) :
  proof_problem x Hx :=
sorry

end proof_inequality_l263_263810


namespace non_congruent_triangles_with_perimeter_11_l263_263126

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263126


namespace ratio_marcus_mona_l263_263937

variables {Marcus Mona Nicholas : ℕ}

-- Given conditions
def same_number_crackers := Marcus = Mona
def marcus_crackers := Marcus = 27

-- Prove that the ratio of the number of crackers Marcus has to the number of crackers Mona has is 1:1
theorem ratio_marcus_mona (h1 : same_number_crackers) (h2 : marcus_crackers) : 
    (Marcus / Mona) = 1 :=
by
    sorry

end ratio_marcus_mona_l263_263937


namespace remainder_5_to_5_to_5_to_5_mod_1000_l263_263770

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l263_263770


namespace reassess_routes_l263_263678

theorem reassess_routes (k : ℕ) :
  ∃ (routes : Fin (2*k + 1) → Fin (k + 1) → Fin (2*k + 1)), 
  (∀ i, card (routes i) = i) ∧ 
  (∀ j, card (routes (λ i, i+j)) = k + 1) ∧ 
  (∀ i j r, r ∈ routes i → route_complying r j)
  :=
sorry

end reassess_routes_l263_263678


namespace prob_last_is_one_correct_l263_263091

noncomputable def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ (∀m : ℕ, m ∣ n → m = 1 ∨ m = n)

structure SelectionProcess :=
  (A : Set ℕ)
  (selections_are_replaced : Bool := true)
  (halt_condition : ℕ × ℕ → Bool)
  (prob_last_is_1 : ℚ)

def selectionHaltCondition (x y : ℕ) : Bool :=
  isPrime (x + y)

def given_set : Set ℕ := {1, 2, 3, 4}

noncomputable def solution : SelectionProcess :=
  { A := given_set,
    selections_are_replaced := true,
    halt_condition := selectionHaltCondition,
    prob_last_is_1 := 3 / 44 }

theorem prob_last_is_one_correct :
  solution.prob_last_is_1 = 3 / 44 :=
sorry

end prob_last_is_one_correct_l263_263091


namespace largest_c_ineq_l263_263389

theorem largest_c_ineq (x : Fin 2018 → ℝ) :
  (∑ i in Finset.range 2016, x i * (x i + x (i + 1))) ≥ -((1008 : ℝ) / 2017) * x 2017 ^ 2 :=
sorry

end largest_c_ineq_l263_263389


namespace radius_of_circle_Q_sum_l263_263901

theorem radius_of_circle_Q_sum (AB AC BC : ℕ) (rP rQ m n k : ℤ) (dist_primes : ∀ (p q : ℤ), p ∣ k → q ∣ k → p ≠ q → Nat.Prime p ∧ Nat.Prime q) :
  AB = 120 → AC = 120 → BC = 72 → rP = 20 → 
  rQ = 65 - 5 * Real.sqrt 79 → 
  m = 65 → n = 5 → k = 79 →
  m + n * k = 460 :=
by
  intros hAB hAC hBC hrP hrQ hm hn hk
  exact
  have h1 : AB = 120 := hAB
  have h2 : AC = 120 := hAC
  have h3 : BC = 72 := hBC
  have hrP' : rP = 20 := hrP
  have hrQ' : rQ = 65 - 5 * Real.sqrt 79 := hrQ
  have hm' : m = 65 := hm
  have hn' : n = 5 := hn
  have hk' : k = 79 := hk
  show m + n * k = 460 from 
  by rw [hm', hn', hk']
  rfl

end radius_of_circle_Q_sum_l263_263901


namespace brett_blue_marbles_more_l263_263727

theorem brett_blue_marbles_more (r b : ℕ) (hr : r = 6) (hb : b = 5 * r) : b - r = 24 := by
  rw [hr, hb]
  norm_num
  sorry

end brett_blue_marbles_more_l263_263727


namespace exists_tangent_inequality_l263_263246

theorem exists_tangent_inequality {x : Fin 8 → ℝ} (h : Function.Injective x) :
  ∃ (i j : Fin 8), i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (Real.pi / 7) :=
by
  sorry

end exists_tangent_inequality_l263_263246


namespace max_min_value_l263_263550

noncomputable def R := {p : Real × Real | 
  ∃ (λ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
     ∃ (μ : ℝ, 0 ≤ μ ∧ μ ≤ 1 ∧ 
        ∃ (ν : ℝ, 0 ≤ ν ∧ ν ≤ 1 ∧ 
           t + μ + ν = 1 ∧ 
           p = (t * (4, 1) + μ * (-1, -6) + ν * (-3, 2))
        )
     )
  )
}

def f (p : ℝ × ℝ) := (4:ℝ) * p.1 - (3:ℝ) * p.2

theorem max_min_value :
  (∀ p ∈ R, f p ≤ 14) ∧ (∃ p ∈ R, f p = 14) ∧
  (∀ p ∈ R, f p ≥ -18) ∧ (∃ p ∈ R, f p = -18) := 
sorry

end max_min_value_l263_263550


namespace find_f_2007_l263_263737

theorem find_f_2007 (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  {x | ∃ n, f n = x} = {1, 2, ..., 2008} :=
by
  sorry

end find_f_2007_l263_263737


namespace sum_loose_numbers_is_correct_l263_263704

-- Definition of a "loose" number
def loose_number (n : ℕ) : Prop :=
  (∃ (d1 d2 d3 d4 d5 d6 : ℕ), list.nodup [d1, d2, d3, d4, d5, d6] ∧
                          list.all (list.pairwise (<) [d1, d2, d3, d4, d5, d6]) ∧
                          list.all (λ (a : ℕ), ∃ (b : ℕ), a = b ∨ b = a, [d1, d2, d3, d4, d5, d6]) ∧
                          list.all (λ (a : ℕ), (n % a = 0), [d1, d2, d3, d4, d5, d6]) ∧
                          (∀ (a b : ℕ), a < b -> b ∈ [d1, d2, d3, d4, d5, d6] -> a ∈ [d1, d2, d3, d4, d5, d6] -> b >= 2 * a))

-- 2 as a prime number constant
def _root_.nat.prime.two : nat.prime 2 := by norm_num

-- Calculating the sum of all loose numbers under 100
noncomputable def sum_loose_numbers_under_100 : ℕ :=
  (finset.range 100).filter (λ (n : ℕ), loose_number n).sum id

-- Theorem to prove that the sum is as expected
theorem sum_loose_numbers_is_correct :
  sum_loose_numbers_under_100 = 462 :=
by sorry

end sum_loose_numbers_is_correct_l263_263704


namespace angle_BAH_eq_angle_OAC_l263_263283

theorem angle_BAH_eq_angle_OAC
  (A B C O H : Type)
  [is_center_of_circumcircle : ∀ A B C : Type, O = circumcenter A B C] 
  (AH : A → B → C) -- altitude from A to B
  (AO : A → O) -- segment from A to O
  : ∠ BAH = ∠ OAC :=
sorry

end angle_BAH_eq_angle_OAC_l263_263283


namespace inlet_rate_correct_l263_263699

theorem inlet_rate_correct :
  (capacity : ℝ) → (rate_leak : ℝ) → (net_rate : ℝ) →
  capacity = 5040 → rate_leak = capacity / 6 → net_rate = capacity / 8 →
  (rate_inlet : ℝ) → rate_inlet - rate_leak = net_rate → rate_inlet = 1470 :=
by 
  intros capacity rate_leak net_rate hc hrl hnr rate_inlet hr
  rw [hc] at hrl hnr 
  have hrl' : rate_leak = 840 := by norm_num [hrl]
  have hnr' : net_rate = 630 := by norm_num [hnr]
  rw [hrl', hnr'] at hr
  norm_num at hr
  exact hr.symm

end inlet_rate_correct_l263_263699


namespace profit_percentage_l263_263487

variable (C S : ℝ)

theorem profit_percentage (h : 22 * C = 16 * S) : (S - C) / C * 100 = 37.5 := by
  have h1 : S = 22 * C / 16 := by linarith
  have h2 : (S - C) / C * 100 = ((22 * C / 16 - C) / C) * 100 := by rw [h1]
  have h3 : (22 * C / 16 - C) / C = (6 * C / 16) / C := by norm_num
  have h4 : (6 * C / 16) / C = 6 / 16 := by rw [div_div_eq_div_mul, mul_div_cancel_left C (by norm_num1)]
  have h5 : 6 / 16 * 100 = 37.5 := by norm_num
  exact (by rw [←h2, h3, h4, h5])

end profit_percentage_l263_263487


namespace residue_of_neg_1235_mod_29_l263_263748

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l263_263748


namespace max_dot_product_value_l263_263516

noncomputable def max_dot_product_BQ_CP (λ : ℝ) : ℝ :=
  - (3/5) * (λ - 2/3)^2 - 86/15

theorem max_dot_product_value :
  ∃ (λ : ℝ), 
    0 ≤ λ ∧ λ ≤ 1 ∧ max_dot_product_BQ_CP λ = -86/15 :=
by
  sorry

end max_dot_product_value_l263_263516


namespace sufficient_not_necessary_condition_l263_263436

variables {α : Type*} [linear_ordered_field α]

noncomputable def arithmetic_sequence (a1 q : α) (n : ℕ) : α :=
  a1 * q^(n - 1)

noncomputable def sum_sequence (a1 q : α) (n : ℕ) : α :=
  if q = 1 then n * a1
  else a1 * (1 - q^n) / (1 - q)

theorem sufficient_not_necessary_condition 
  (a1 q : α) (h1 : a1 > 0) :
  (q > 1 → (sum_sequence a1 q 3 + sum_sequence a1 q 5 > 2 * sum_sequence a1 q 4)) ∧
  ((sum_sequence a1 q 3 + sum_sequence a1 q 5 > 2 * sum_sequence a1 q 4) → q > 1) := 
sorry

end sufficient_not_necessary_condition_l263_263436


namespace complex_sum_l263_263050

theorem complex_sum : 
  ∀ (a b : ℝ), 
  (1 + 2 * complex.I⁻¹)^2 = a + b * complex.I → 
  a + b = -7 :=
by
  sorry

end complex_sum_l263_263050


namespace max_value_expr_l263_263019

theorem max_value_expr : ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) = 85 :=
by sorry

end max_value_expr_l263_263019


namespace jenny_total_wins_l263_263531

theorem jenny_total_wins (mark_games_played : ℕ) (mark_wins : ℕ) (jill_multiplier : ℕ)
  (jill_win_percent : ℚ) (jenny_vs_mark_games : ℕ := 10) (mark_wins_out_of_10 : ℕ := 1) 
  (jill_games_played : ℕ := 2 * jenny_vs_mark_games) (jill_win_percent_value : ℚ := 0.75) :
  let jenny_wins_mark := jenny_vs_mark_games - mark_wins_out_of_10,
      jenny_wins_jill := jill_games_played - (jill_win_percent_value * jill_games_played).natAbs in
  jenny_wins_mark + jenny_wins_jill = 14 :=
by
  -- Definitions
  let jenny_vs_mark_games := 10
  let mark_wins_out_of_10 := 1
  let jenny_wins_mark := jenny_vs_mark_games - mark_wins_out_of_10
  let jill_games_played := 2 * jenny_vs_mark_games
  let jill_win_percent_value := 0.75
  let jill_wins := (jill_win_percent_value * jill_games_played).toNat
  let jenny_wins_jill := jill_games_played - jill_wins
  -- Calculation
  have jenny_wins_total := jenny_wins_mark + jenny_wins_jill
  -- Expected result
  show jenny_wins_total = 14, from
    sorry

end jenny_total_wins_l263_263531


namespace nitrogen_L_shell_electrons_hydrazine_N2O4_reaction_hydrazine_combustion_heat_l263_263577

theorem nitrogen_L_shell_electrons : number_of_electrons_L_shell (atomic_number 7) = 5 := sorry

theorem hydrazine_N2O4_reaction (H1 : ΔH (N2 (g) + 2 * O2 (g) = N2O4 (l)) = -19.5)
                                (H2 : ΔH (N2H4 (l) + O2 (g) = N2 (g) + 2 * H2O (g)) = -534.2) :
  ΔH (2 * N2H4 (l) + N2O4 (l) = 3 * N2 (g) + 4 * H2O (g)) = -1048.9 :=
sorry

theorem hydrazine_combustion_heat (H2 : ΔH (N2H4 (l) + O2 (g) = N2 (g) + 2 * H2O (g)) = -534.2)
                                  (H3 : ΔH (H2O (l) = H2O (g)) = 44) :
    ΔH (N2H4 (l) + O2 (g) = N2 (g) + 2 * H2O (l)) = -622.2 :=
sorry

end nitrogen_L_shell_electrons_hydrazine_N2O4_reaction_hydrazine_combustion_heat_l263_263577


namespace determine_counterfeit_coin_one_weighing_l263_263313

theorem determine_counterfeit_coin_one_weighing :
  ∃ (coins : Fin 3 → ℝ), (∀ i : Fin 3, coins i > 0) ∧ (∃ (c : Fin 3), ∀ i : Fin 3, i ≠ c → coins i = w) ∧ (∀ c : Fin 3, ∃ i j k : Fin 3, {i, j, k}.all (λ x, x ≠ c) ∧ ((coins i = coins j ∧ coins i > coins k) ∨ (coins i ≠ coins j ∧ coins k = coins i ∨ coins k = coins j))) → 
  ∃ only_one_weighing_needed : ℕ, only_one_weighing_needed = 1 :=
begin
  sorry
end

end determine_counterfeit_coin_one_weighing_l263_263313


namespace julia_played_with_kids_on_monday_l263_263537

theorem julia_played_with_kids_on_monday (kids_tuesday : ℕ) (h_tuesday : kids_tuesday = 5) 
  (h_more_kids : kids_tuesday + 1 = 6) :
  ∃ kids_monday : ℕ, kids_monday = 6 :=
by {
  use 6,
  -- conditions
  exact h_more_kids,
}

end julia_played_with_kids_on_monday_l263_263537


namespace cube_edge_length_is_10_l263_263869

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℝ := 60
noncomputable def total_cost : ℝ := 32

theorem cube_edge_length_is_10 
  (cost_per_quart : ℝ := 3.20) 
  (coverage_per_quart : ℝ := 60) 
  (total_cost : ℝ := 32): 
  let quarts_needed := total_cost / cost_per_quart in
  let total_coverage := quarts_needed * coverage_per_quart in
  let total_surface_area := total_coverage in
  ∃ L : ℝ, 6 * L^2 = total_surface_area ∧ L = 10 :=
begin
  sorry
end

end cube_edge_length_is_10_l263_263869


namespace fourfold_composition_is_odd_l263_263221

variable (f : ℝ → ℝ)

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define the problem to prove that f(f(f(f(x)))) is odd
theorem fourfold_composition_is_odd (h : is_odd_function f) : 
  ∀ x, f (f (f (f x))) (-x) = -f (f (f (f x))) x :=
sorry

end fourfold_composition_is_odd_l263_263221


namespace range_of_y_l263_263746

-- Define the function y
def y (x : ℝ) : ℝ := - (Real.cos x)^2 + Real.sin x

-- Define the range of the function y
def in_range (b : ℝ) : Prop := b ∈ Set.Icc (-5 / 4) 1

-- State the theorem to prove
theorem range_of_y : ∀ b, ∃ x : ℝ, y x = b ↔ in_range b := by
  sorry

end range_of_y_l263_263746


namespace power_mod_remainder_l263_263793

theorem power_mod_remainder (a b c : ℕ) (h1 : 7^40 % 500 = 1) (h2 : 7^4 % 40 = 1) : (7^(7^25) % 500 = 43) :=
sorry

end power_mod_remainder_l263_263793


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l263_263086

noncomputable def f : ℝ → ℝ := λ x, 2 * (real.sqrt 3) * real.sin x * real.cos x + 2 * (real.cos x)^2 - 1

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

theorem max_min_values_of_f_on_interval :
  ∃ max min : ℝ, (∀ x ∈ Icc (-π / 6) (π / 4), min ≤ f x ∧ f x ≤ max) ∧ 
                    (∃ x ∈ Icc (-π / 6) (π / 4), f x = max) ∧ 
                    (∃ x ∈ Icc (-π / 6) (π / 4), f x = min) ∧ 
                    max = 2 ∧ 
                    min = -1 := 
sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l263_263086


namespace math_problem_proof_l263_263178

-- Definitions from conditions in part a)
def freq_100 : ℚ := 59 / 100
def freq_200 : ℚ := 0.58
def num_white_balls : ℕ := 12

-- Statement for equivalence proof
theorem math_problem_proof :
  (freq_100 = 0.59 ∧ 200 * freq_200 = 116) ∧
  (0.6 = Real.round (freq_100 + 0.64 + 0.58 + 0.59 + 0.60 + 0.601) / 6) ∧
  ((num_white_balls / 0.6) - num_white_balls = 8) :=
by
  sorry

end math_problem_proof_l263_263178


namespace find_a_from_slope_and_points_l263_263058

theorem find_a_from_slope_and_points (a : ℝ) :
  let k := Real.tan (Real.pi / 4) in
  k = 1 ∧ k = (a - 3) / (2 - (-1)) → a = 6 :=
by
  intro k h
  sorry

end find_a_from_slope_and_points_l263_263058


namespace find_integers_l263_263761

theorem find_integers (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ a : Fin n → Fin n, ∀ k : Fin (n + 1), (∑ i in Finset.range (k + 1).val, a i.val) % (k + 1).val = 0) :
  n = 1 ∨ n = 3 :=
sorry

end find_integers_l263_263761


namespace fraction_red_is_one_seventh_l263_263877

noncomputable def fraction_red (x : ℝ) : ℝ :=
  let blue := (2 / 3) * x
  let red := x - blue
  let new_blue := 3 * blue
  let new_total := new_blue + red
  red / new_total

theorem fraction_red_is_one_seventh (x : ℝ) (hx : x ≠ 0) : fraction_red x = 1 / 7 := by
  let blue := 2 / 3 * x
  let red := x - blue
  let new_blue := 3 * blue
  let new_total := new_blue + red
  have h1 : red = x / 3 := by sorry
  have h2 : new_total = 7 / 3 * x := by sorry
  have h3 : fraction_red x = (x / 3) / (7 / 3 * x) := by sorry
  calc
    (x / 3) / (7 / 3 * x) = 1 / 7 := by
      first
        extensionality, 
        ring

-- Proof is left as an exercise

end fraction_red_is_one_seventh_l263_263877


namespace yellow_marble_probability_correct_l263_263726

open Classical

variable (BagA_white BagA_black BagB_yellow BagB_blue BagB_green BagC_yellow BagC_blue : ℕ)

def probability_draw_second_yellow : ℚ :=
  let BagA_white := 4
  let BagA_black := 5
  let BagB_yellow := 5
  let BagB_blue := 3
  let BagB_green := 2
  let BagC_yellow := 2
  let BagC_blue := 5

  have total_BagA : ℕ := BagA_white + BagA_black
  have total_BagB : ℕ := BagB_yellow + BagB_blue + BagB_green
  have total_BagC : ℕ := BagC_yellow + BagC_blue
  have prob_white_BagA : ℚ := BagA_white / (BagA_white + BagA_black)
  have prob_black_BagA : ℚ := BagA_black / (BagA_white + BagA_black)
  have prob_yellow_BagB : ℚ := BagB_yellow / total_BagB
  have prob_yellow_BagC : ℚ := BagC_yellow / total_BagC

  (prob_white_BagA * prob_yellow_BagB) + (prob_black_BagA * prob_yellow_BagC)

theorem yellow_marble_probability_correct :
  probability_draw_second_yellow 4 5 5 3 2 2 5 = 8 / 21 :=
by
  let BagA_white := 4
  let BagA_black := 5
  let BagB_yellow := 5
  let BagB_blue := 3
  let BagB_green := 2
  let BagC_yellow := 2
  let BagC_blue := 5
  sorry

end yellow_marble_probability_correct_l263_263726


namespace iron_oxide_element_l263_263987

theorem iron_oxide_element (mass_percent : ℝ) (h : mass_percent = 70) : 
  ∃ element : string, element = "Fe" :=
by
  let molar_mass_fe := 55.85
  let molar_mass_o := 16.00
  let molar_mass_fe2o3 := (2 * molar_mass_fe) + (3 * molar_mass_o)
  let mass_percent_fe := (2 * molar_mass_fe) / molar_mass_fe2o3 * 100
  have h1 : mass_percent_fe ≈ 70 := sorry
  sorry

end iron_oxide_element_l263_263987


namespace gain_percent_is_100_l263_263674

variable {C S : ℝ}

-- Given conditions
axiom h1 : 50 * C = 25 * S
axiom h2 : S = 2 * C

-- Prove the gain percent is 100%
theorem gain_percent_is_100 (h1 : 50 * C = 25 * S) (h2 : S = 2 * C) : (S - C) / C * 100 = 100 :=
by
  sorry

end gain_percent_is_100_l263_263674


namespace minimize_ratio_l263_263596

noncomputable def circumsphere_radius (x : ℝ) : ℝ := (2 * x^2 + 1) / (4 * x)
noncomputable def insphere_radius (x : ℝ) : ℝ := x / (1 + sqrt (4 * x^2 + 1))
noncomputable def ratio (x : ℝ) : ℝ := (circumsphere_radius x) / (insphere_radius x)

theorem minimize_ratio : ∃ x : ℝ, x = sqrt ((1 + sqrt 2) / 2) ∧ ratio x = 1 + sqrt 2 :=
by
  use sqrt ((1 + sqrt 2) / 2)
  sorry

end minimize_ratio_l263_263596


namespace count_non_congruent_triangles_with_perimeter_11_l263_263113

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263113


namespace count_valid_n_l263_263045

theorem count_valid_n : 
  (finset.card {n : ℕ | 0 < n ∧ n < 36 ∧ (∃ m : ℕ, m > 0 ∧ n = (36 * m) / (m + 1) ∧ (36 * m) % (m + 1) = 0) } = 7) := 
by
  sorry

end count_valid_n_l263_263045


namespace factor_polynomial_l263_263802

theorem factor_polynomial (y : ℝ) : 3 * y ^ 2 - 75 = 3 * (y - 5) * (y + 5) :=
by
  sorry

end factor_polynomial_l263_263802


namespace find_a_l263_263997

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 1       := 1
| 2       := a
| (n + 1) := (2 * n + 1) * sequence a n - (n^2 - 1) * sequence a (n - 1)

def seq_property (a : ℕ) : Prop :=
∀ i j : ℕ, i < j → sequence a i ∣ sequence a j

theorem find_a : ∀ a : ℕ, a = 2 ∨ a = 4 ↔ seq_property a := sorry

end find_a_l263_263997


namespace value_domain_of_quadratic_function_l263_263636

-- Define the function and its interval
def quadraticFunction (x : ℝ) : ℝ := x^2 - 2 * x - 3
def interval (x : ℝ) : Prop := -1 ≤ x ∧ x < 2

-- State the theorem
theorem value_domain_of_quadratic_function :
  {y : ℝ | ∃ x : ℝ, interval x ∧ quadraticFunction x = y} = set.Icc (-4) 0 :=
by
  sorry

end value_domain_of_quadratic_function_l263_263636


namespace max_correct_answers_l263_263494

theorem max_correct_answers (a b c : ℕ) (n : ℕ := 60) (p_correct : ℤ := 5) (p_blank : ℤ := 0) (p_incorrect : ℤ := -2) (S : ℤ := 150) :
        a + b + c = n ∧ p_correct * a + p_blank * b + p_incorrect * c = S → a ≤ 38 :=
by
  sorry

end max_correct_answers_l263_263494


namespace projected_vector_unique_l263_263308

theorem projected_vector_unique (w : ℝ × ℝ) 
  (hw : ∀ (v : ℝ × ℝ), v.2 = 3 * v.1 + 1 → 
    (⟨v.1, 3 * v.1 + 1⟩ • w) / (w.1 ^ 2 + w.2 ^ 2) = ⟨-3, 1⟩ / 10) :
  (⟨-3, 1⟩ / 10) = ⟨-3/10, 1/10⟩ :=
sorry

end projected_vector_unique_l263_263308


namespace ab_geq_3_plus_cd_l263_263925

theorem ab_geq_3_plus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13) (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := 
sorry

end ab_geq_3_plus_cd_l263_263925


namespace car_distance_l263_263330

-- Define the conditions
def speed := 162  -- speed of the car in km/h
def time := 5     -- time taken in hours

-- Define the distance calculation
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

-- State the theorem
theorem car_distance : distance speed time = 810 := by
  -- Proof goes here
  sorry

end car_distance_l263_263330


namespace find_a_l263_263892

noncomputable def circle_eq : ℝ → ℝ → Prop :=
λ x y, x^2 + y^2 - 4*x - 8*y + 19 = 0

noncomputable def line_eq (a : ℝ) : ℝ → ℝ → Prop :=
λ x y, x + 2*y - a = 0

theorem find_a (a : ℝ) :
  (∃ x y, circle_eq x y ∧ line_eq a x y) →
  a = 10 :=
sorry

end find_a_l263_263892


namespace sum_of_ages_equal_to_grandpa_l263_263847

-- Conditions
def grandpa_age : Nat := 75
def grandchild_age_1 : Nat := 13
def grandchild_age_2 : Nat := 15
def grandchild_age_3 : Nat := 17

-- Main Statement
theorem sum_of_ages_equal_to_grandpa (t : Nat) :
  (grandchild_age_1 + t) + (grandchild_age_2 + t) + (grandchild_age_3 + t) = grandpa_age + t 
  ↔ t = 15 := 
by {
  sorry
}

end sum_of_ages_equal_to_grandpa_l263_263847


namespace relationship_between_a_and_b_l263_263082

-- Definitions based on the conditions
def point1_lies_on_line (a : ℝ) : Prop := a = (2/3 : ℝ) * (-1 : ℝ) - 3
def point2_lies_on_line (b : ℝ) : Prop := b = (2/3 : ℝ) * (1/2 : ℝ) - 3

-- The main theorem to prove the relationship between a and b
theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : point1_lies_on_line a)
  (h2 : point2_lies_on_line b) : a < b :=
by
  -- Skipping the actual proof. Including sorry to indicate it's not provided.
  sorry

end relationship_between_a_and_b_l263_263082


namespace total_toy_worth_l263_263714

theorem total_toy_worth : 
  (9 = 1 + 8) → 
  (∀ (t1 t2 t3 t4 t5 t6 t7 t8 t9: ℕ), t1 = 12 ∧ t2 = 5 ∧ t3 = 5 ∧ t4 = 5 ∧ t5 = 5 ∧ t6 = 5 ∧ t7 = 5 ∧ t8 = 5 ∧ t9 = 5 → 
  (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9) = 52) :=
begin
  sorry
end

end total_toy_worth_l263_263714


namespace greatest_integer_b_not_in_range_of_quadratic_l263_263661

theorem greatest_integer_b_not_in_range_of_quadratic :
  ∀ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ 5) ↔ (b^2 < 60) ∧ (b ≤ 7) := by
  sorry

end greatest_integer_b_not_in_range_of_quadratic_l263_263661


namespace terms_before_negative_seventeen_l263_263859

theorem terms_before_negative_seventeen :
  ∃ n : ℕ, (∀ m < n, a + m * d ≠ -17) ∧ (a + n * d = -17) → n - 1 = 17 :=
by
  sorry

def a : ℤ := 103
def d : ℤ := -7

end terms_before_negative_seventeen_l263_263859


namespace total_grey_area_l263_263617

theorem total_grey_area
    (wall_width wall_height : ℝ)
    (smaller_square_diagonal larger_square_diagonal : ℝ)
    (angle45 : ℝ)
    (smaller_square_count larger_square_count : ℝ)
    (total_area : ℝ)
    (h1 : wall_width = 16)
    (h2 : wall_height = 16)
    (h3 : smaller_square_count = 2)
    (h4 : angle45 = (Math.pi / 4))
    (h5 : smaller_square_diagonal = 8)
    (h6 : larger_square_count = 1)
    (h7 : total_area = 128) :
  total_area = smaller_square_count * (smaller_square_diagonal^2 / 2) + 
               larger_square_count * (smaller_square_diagonal^2 / 2 * 2) :=
by
  sorry

end total_grey_area_l263_263617


namespace problem1_problem2_problem3_l263_263683

-- Problem (1)
def setU := { x : ℤ | -5 ≤ x ∧ x ≤ 10 }
def setM := { x : ℤ | 0 ≤ x ∧ x ≤ 7 }
def setN := { x : ℤ | -2 ≤ x ∧ x < 4 }
def complement_U_N := { x : ℤ | -5 ≤ x ∧ x < -2 ∨ 4 ≤ x ∧ x ≤ 10 }
def expectedResult1 := { 4, 5, 6, 7 }

theorem problem1 : (complement_U_N ∩ setM) = expectedResult1 := sorry

-- Problem (2)
def universalSet := { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
def setA := { x : ℤ | x ∈ universalSet } -- Details of setA are not specified
def condition := { 2, 4, 6, 8 }
def complement_U_B := { x : ℤ | x ∈ universalSet ∧ x ∉ setB }
def setB := { 0, 1, 3, 5, 7, 9, 10 }

theorem problem2 : A ∩ complement_U_B = condition → B = setB := sorry

-- Problem (3)
def P (a : ℝ) := { x : ℝ | a * x^2 + 2 * a * x + 1 = 0 }
def valueOfA := 1
def singleElement := -1

theorem problem3 (a : ℝ): (∀ x y ∈ P a, x = y) → a = valueOfA ∧ P a = { singleElement } := sorry

end problem1_problem2_problem3_l263_263683


namespace perimeter_equilateral_triangle_l263_263188

theorem perimeter_equilateral_triangle :
  ∀ (a : ℝ), a = 600 → (3 * a) / 100 = 18 :=
by
  intros a ha
  rw ha
  sorry

end perimeter_equilateral_triangle_l263_263188


namespace problem_l263_263635

/-
A problem involving natural numbers a and b
where:
1. Their sum is 20000
2. One of them (b) is divisible by 5
3. Erasing the units digit of b gives the other number a

We want to prove their difference is 16358
-/

def nat_sum_and_difference (a b : ℕ) : Prop :=
  a + b = 20000 ∧
  b % 5 = 0 ∧
  (b % 10 = 0 ∧ b / 10 = a ∨ b % 10 = 5 ∧ (b - 5) / 10 = a)

theorem problem (a b : ℕ) (h : nat_sum_and_difference a b) : b - a = 16358 := 
  sorry

end problem_l263_263635


namespace number_of_valid_pairs_is_34_l263_263547

noncomputable def countValidPairs : Nat :=
  let primes : List Nat := [2, 3, 5, 7, 11, 13]
  let nonprimes : List Nat := [1, 4, 6, 8, 9, 10, 12, 14, 15]
  let countForN (n : Nat) : Nat :=
    match n with
    | 2 => Nat.choose 8 1
    | 3 => Nat.choose 7 2
    | 5 => Nat.choose 5 4
    | _ => 0
  primes.map countForN |>.sum

theorem number_of_valid_pairs_is_34 : countValidPairs = 34 :=
  sorry

end number_of_valid_pairs_is_34_l263_263547


namespace unique_functional_equation_solution_l263_263030

theorem unique_functional_equation_solution (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n) + f(f(n)) + f(f(f(n))) = 3 * n) : ∀ n : ℕ, f(n) = n :=
by 
  sorry

end unique_functional_equation_solution_l263_263030


namespace breadth_is_13_l263_263611

variable (b l : ℕ) (breadth : ℕ)

/-
We have the following conditions:
1. The area of the rectangular plot is 23 times its breadth.
2. The difference between the length and the breadth is 10 metres.
We need to prove that the breadth of the plot is 13 metres.
-/

theorem breadth_is_13
  (h1 : l * b = 23 * b)
  (h2 : l - b = 10) :
  b = 13 := 
sorry

end breadth_is_13_l263_263611


namespace solve_equation_l263_263957

theorem solve_equation (x : ℝ) : 
  (x + 1) / 6 = 4 / 3 - x ↔ x = 1 :=
sorry

end solve_equation_l263_263957


namespace differentiability_at_0_l263_263201

-- Definitions for the real and imaginary parts
def u (x y : ℝ) : ℝ := x^2 + y^2
def v (x y : ℝ) : ℝ := 0

-- Proving differentiability of the function f(z) = |z|^2 at z = 0 
theorem differentiability_at_0 :
  (∀ z : ℂ, z = 0 → (differentiable_at ℂ (λ (z : ℂ), abs z ^ 2) z)) ∧
  (complex.has_deriv_at (λ (z : ℂ), abs z ^ 2) 0 (0 : ℂ)) :=
by
  sorry

end differentiability_at_0_l263_263201


namespace areas_equal_l263_263579

-- Definitions based on the conditions
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def square_area (side : ℝ) : ℝ :=
  side^2

def triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

-- Assumptions and given conditions
variables {a b c : ℝ}
variables {P Q R : ℝ}

-- Point P is the intersection of LB and AC
-- Point Q is the intersection of AN and BC
-- Point R is the intersection of LB and AN

-- The statement to be proven
theorem areas_equal (h_triangle : right_triangle a b c)
  (h_P : some_condition_about_P)
  (h_Q : some_condition_about_Q)
  (h_R : some_condition_about_R) : 
  triangle_area a b = triangle_area a b := sorry

end areas_equal_l263_263579


namespace count_ways_line_up_l263_263507

theorem count_ways_line_up (persons : Finset ℕ) (youngest eldest : ℕ) :
  persons.card = 5 →
  youngest ∈ persons →
  eldest ∈ persons →
  (∃ seq : List ℕ, seq.length = 5 ∧ 
    ∀ (i : ℕ), i ∈ (List.finRange 5).erase 0 → seq.get ⟨i, sorry⟩ ≠ youngest ∧ 
    i ∈ (List.finRange 5).erase 4 → seq.get ⟨i, sorry⟩ ≠ eldest) →
  (persons \ {youngest, eldest}).card = 3 →
  4 * 4 * 3 * 2 * 1 = 96 :=
by
  sorry

end count_ways_line_up_l263_263507


namespace cost_of_500_sheets_is_10_dollars_l263_263995

-- Define the conditions as given in the problem.
def price_per_sheet_cents : ℕ := 2
def total_sheets : ℕ := 500
def cents_to_dollars (cents : ℕ) : ℝ := cents / 100

-- Define the statement we want to prove.
theorem cost_of_500_sheets_is_10_dollars : 
  let total_cost_cents := total_sheets * price_per_sheet_cents
  in 
  cents_to_dollars total_cost_cents = 10 :=
sorry

end cost_of_500_sheets_is_10_dollars_l263_263995


namespace logarithmic_expression_identity_l263_263324

theorem logarithmic_expression_identity :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - 5^(log 5 3) = -1 := by
sorry

end logarithmic_expression_identity_l263_263324


namespace zoo_adult_ticket_cost_l263_263610

theorem zoo_adult_ticket_cost
  (child_ticket_cost : ℕ := 4)
  (total_people : ℕ := 201)
  (children_count : ℕ := 161)
  (total_bill : ℕ := 964)
  (A : ℕ) :
  (total_people - children_count) * A + children_count * child_ticket_cost = total_bill →
  A = 8 :=
begin
  sorry
end

end zoo_adult_ticket_cost_l263_263610


namespace solveMatrixEquation_l263_263259

open Matrix

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, -1, 0], ![-2, 1, 1], ![2, -1, 4]]

noncomputable def B : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![5], ![0], ![15]]

noncomputable def X : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![2], ![1], ![3]]

theorem solveMatrixEquation : (A.mul X) = B := by
  sorry

end solveMatrixEquation_l263_263259


namespace exists_face_sum_greater_25_l263_263578

-- Define the problem conditions
def cube_edges := Finset.range 12  -- The set of edges labeled from 1 to 12

-- Define the sum of lengths on faces
def face_sums (f : ℕ → ℕ → ℕ) : Finset ℕ := 
  (Finset.range 6).filter (λ face, (f (2*face) + f (2*face + 1) + f (2*face + 2) + f (2*face + 3)) > 25)

-- Provide the critical theorem to be proved
theorem exists_face_sum_greater_25 (f : ℕ → ℕ) (hf : ∀ i ∈ (Finset.range 12), f i = i + 1) : 
  ∃ face ∈ (Finset.range 6), (f (2*face) + f (2*face + 1) + f (2*face + 2) + f (2*face + 3)) > 25 :=
by
  sorry

end exists_face_sum_greater_25_l263_263578


namespace intersecting_lines_midpoints_l263_263977

theorem intersecting_lines_midpoints 
  (A B C D M N : ℝ^2) 
  (h_convex: IsConvex {A, B, C, D})
  (h_perpendicular: ∠ A C B = 90 ∧ ∠ B D C = 90)
  (h_midpoints_M: M = (A + B) / 2)
  (h_midpoints_N: N = (A + D) / 2)
  (h_perpendicular_lines: ∀ P Q : ℝ^2, 
  (P = foot (line_through M N) C D → 
  Q = foot (line_through N M) B C → 
  line_through M P ∩ line_through N Q = some X) ) :
  ∃ X : ℝ^2, 
  point_line_intersection (line_through A C) (line_through M P) ∧ 
  point_line_intersection (line_through A C) (line_through N Q) :=
sorry

end intersecting_lines_midpoints_l263_263977


namespace ratio_of_edges_l263_263158

noncomputable def cube_volume (edge : ℝ) : ℝ := edge^3

theorem ratio_of_edges 
  {a b : ℝ} 
  (h : cube_volume a / cube_volume b = 27) : 
  a / b = 3 :=
by
  sorry

end ratio_of_edges_l263_263158


namespace smallest_n_value_l263_263223

theorem smallest_n_value :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ (i : ℕ), i < n → |x i| < 1) ∧ (∑ i in finset.range n, |x i| = 25 + |∑ i in finset.range n, x i|) → n = 26) :=
sorry

end smallest_n_value_l263_263223


namespace ellipse_equation_slope_PQ_constant_l263_263067

section ellipse

variables {a b x y : ℝ}

-- Given conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ((x^2 / a^2) + (y^2 / b^2) = 1)

def parabola_focus : (ℝ × ℝ) := (2, 0)

def point_A : (ℝ × ℝ) := (2, real.sqrt 2)

-- Question (1): Prove the equation of the ellipse is \( \frac{x^2}{8} + \frac{y^2}{4} = 1 \)
theorem ellipse_equation (a b : ℝ) :
  ellipse a b 2 (sqrt 2) → a^2 = 8 ∧ b^2 = 4 :=
sorry

-- Question (2): Prove the slope of line PQ is constant
theorem slope_PQ_constant (k x1 x2 y1 y2 : ℝ) :
  ellipse 2 (sqrt 2) x1 y1 → ellipse 2 (sqrt 2) x2 y2 →
  (k = y1/(x1-2) ∧ k = -y2/(x2-2)) →
  (1 + 2*k^2) * (x1 + x2) = 0 →
  slopes_PQ x1 y1 x2 y2 = (1/sqrt 2) := 
sorry

end ellipse

end ellipse_equation_slope_PQ_constant_l263_263067


namespace interval_of_increase_l263_263081

noncomputable def power_function (n : ℝ) (x : ℝ) : ℝ := x ^ n

theorem interval_of_increase (n : ℝ) (x : ℝ) (h1 : power_function n 4 = 2) :
  (0 < x) → ∃ f, f = power_function n x ∧ (f > 0) :=
begin
  sorry
end

end interval_of_increase_l263_263081


namespace ratio_of_sequences_is_5_over_4_l263_263394

-- Definitions of arithmetic sequences
def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Hypotheses
def sequence_1_sum : ℕ :=
  arithmetic_sum 5 5 16

def sequence_2_sum : ℕ :=
  arithmetic_sum 4 4 16

-- Main statement to be proven
theorem ratio_of_sequences_is_5_over_4 : sequence_1_sum / sequence_2_sum = 5 / 4 := sorry

end ratio_of_sequences_is_5_over_4_l263_263394


namespace count_non_congruent_triangles_with_perimeter_11_l263_263115

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l263_263115


namespace non_congruent_triangles_with_perimeter_11_l263_263133

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263133


namespace combined_selling_price_l263_263576

theorem combined_selling_price :
  let cost_price_A := 180
  let profit_percent_A := 0.15
  let cost_price_B := 220
  let profit_percent_B := 0.20
  let cost_price_C := 130
  let profit_percent_C := 0.25
  let selling_price_A := cost_price_A * (1 + profit_percent_A)
  let selling_price_B := cost_price_B * (1 + profit_percent_B)
  let selling_price_C := cost_price_C * (1 + profit_percent_C)
  selling_price_A + selling_price_B + selling_price_C = 633.50 := by
  sorry

end combined_selling_price_l263_263576


namespace part_a_part_b_part_b_answer_l263_263545

open Real

-- Define the form of the conditions for part (a)
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- The main theorem for Part (a)
theorem part_a (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h : ∃ᶠ n in at_top, floor (p * n) + floor (q * n) + floor (r * n) = n) :
  is_rational p ∧ is_rational q ∧ is_rational r :=
sorry

-- Define the condition for Part (b)
theorem part_b : ∃ c : ℕ, 
  ∀ a b : ℕ, ∃ᶠ n in at_top, 
    floor (n / a) + floor (n / b) + floor (c * n / 202) = n :=
sorry

-- The final answer for Part (b)
theorem part_b_answer : ∃ t : ℕ, t = 101 := 
begin
  -- t is defined as the number of positive integers c satisfying the condition
  use 101,
  sorry
end

end part_a_part_b_part_b_answer_l263_263545


namespace line_passes_through_fixed_point_l263_263994

-- Define the condition that represents the family of lines
def family_of_lines (k : ℝ) (x y : ℝ) : Prop := k * x + y + 2 * k + 1 = 0

-- Formulate the theorem stating that (-2, -1) always lies on the line
theorem line_passes_through_fixed_point (k : ℝ) : family_of_lines k (-2) (-1) :=
by
  -- Proof skipped with sorry.
  sorry

end line_passes_through_fixed_point_l263_263994


namespace set_intersection_eq_l263_263226

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- The proof statement
theorem set_intersection_eq :
  A ∩ B = A :=
sorry

end set_intersection_eq_l263_263226


namespace min_weights_correct_unique_composition_l263_263060

noncomputable def min_weights (n : ℕ) : ℕ :=
  ⌈log 3 (2 * n + 1)⌉

theorem min_weights_correct (n : ℕ) (hn : 0 < n) : 
  ∀ k : ℕ, (∀ m : ℕ, m ≤ n → ∃ (a : Fin k → ℕ), (∀ (x : Fin k), a x > 0) ∧ 
  (∀ w : ℕ, w ≤ n → ∃ b : Fin k → ℤ, (∀ (x : Fin k), b x ∈ {-1, 0, 1}) ∧ 
  w = ∑ i, b i * a i)) ↔ k ≥ min_weights n :=
sorry

theorem unique_composition (n : ℕ) : 
  (∃ m : ℕ, n = (3^m - 1) / 2) ↔ 
  ∃ (a : Fin (min_weights n) → ℕ), (∀ (x : Fin (min_weights n)), a x > 0) ∧ 
  (∀ (b1 b2 : Fin (min_weights n) → ℤ), 
    (∀ (x : Fin (min_weights n)), b1 x ∈ {-1, 0, 1}) ∧ 
    (∀ (x : Fin (min_weights n)), b2 x ∈ {-1, 0, 1}) ∧ 
    (∀ w : ℕ, w ≤ n → ∑ i, b1 i * a i = ∑ i, b2 i * a i) → 
    (∀ x, b1 x = b2 x)) :=
sorry

end min_weights_correct_unique_composition_l263_263060


namespace math_problem_solution_l263_263426

noncomputable section

def parabola_condition (p m : ℝ) : Prop :=
  ∃ M : ℝ × ℝ, M = (1, m) ∧ (m^2 = 2 * p * 1)

def distance_to_focus_condition (p m : ℝ) : Prop :=
  ∃ F : ℝ × ℝ, F = (p / 2, 0) ∧ (sqrt ((1 - p / 2)^2 + m^2) = 2)

def perpendicular_condition (y1 y2 : ℝ) : Prop :=
  let k1 := (y1 - 2) / ((y1^2 / 2) - 1)
  let k2 := (y2 - 2) / ((y2^2 / 2) - 1)
  k1 * k2 = -1

def midpoint_condition (x₀ : ℝ) : Prop :=
  x₀ + 1 = 15 / 2

def find_p_m_and_D : Prop :=
  parabola_condition 2 2 ∧
  distance_to_focus_condition 2 2 ∧
  (∀ y1 y2 : ℝ, perpendicular_condition y1 y2 → 
    ∃ D : ℝ × ℝ, D = (13 / 2, 1) ∨ D = (13 / 2, -3))

theorem math_problem_solution : find_p_m_and_D := by
  sorry

end math_problem_solution_l263_263426


namespace cards_given_l263_263947

-- Defining the conditions
def initial_cards : ℕ := 4
def final_cards : ℕ := 12

-- The theorem to be proved
theorem cards_given : final_cards - initial_cards = 8 := by
  -- Proof will go here
  sorry

end cards_given_l263_263947


namespace probability_negative_product_l263_263290

theorem probability_negative_product :
  let S := { -7, -3, 1, 5, 8 } in
  (∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) < 0) / 
  ∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c = 
  (3 : ℚ) / 5 :=
sorry

end probability_negative_product_l263_263290


namespace sandy_comic_books_l263_263953

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l263_263953


namespace arithmetic_geometric_sequence_l263_263817

noncomputable def a (n : ℕ) : ℝ := 2 ^ n
def b (n : ℕ) : ℝ := Real.log (a n) / Real.log 2
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / (b i * b (i + 1))

theorem arithmetic_geometric_sequence :
  (∀ n, a n > 0) ∧ a 3 = 8 ∧ (a 3 + 2 = (a 2 + a 4) / 2) →
  (∀ n, a n = 2 ^ n) ∧ (∀ n, T n = n / (n + 1)) :=
by
  sorry

end arithmetic_geometric_sequence_l263_263817


namespace triangle_count_with_perimeter_11_l263_263105

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l263_263105


namespace non_congruent_triangles_with_perimeter_11_l263_263129

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l263_263129


namespace jasmine_needs_additional_bottles_l263_263338

theorem jasmine_needs_additional_bottles :
  ∀ (medium_bottle_capacity giant_bottle_capacity filled_medium_bottles : ℕ),
    medium_bottle_capacity = 50 →
    giant_bottle_capacity = 750 →
    filled_medium_bottles = 3 →
    (giant_bottle_capacity / medium_bottle_capacity - filled_medium_bottles) = 12 :=
by
  intros medium_bottle_capacity giant_bottle_capacity filled_medium_bottles
  assume hmbc : medium_bottle_capacity = 50
  assume hgbc : giant_bottle_capacity = 750
  assume hfm : filled_medium_bottles = 3
  sorry

end jasmine_needs_additional_bottles_l263_263338


namespace all_terms_divisible_by_2005_l263_263360

noncomputable def arithmetic_progression_divisibility (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∃ a1 : ℕ, (∀ n : ℕ, a n = a1 + n * d) ∧ (∀ n : ℕ, 2005 ∣ (a n) * (a (n + 31)))

theorem all_terms_divisible_by_2005 (a : ℕ → ℕ) (d : ℕ) :
  (arithmetic_progression_divisibility a d) → (∀ n : ℕ, 2005 ∣ a n) :=
begin
  sorry
end

end all_terms_divisible_by_2005_l263_263360


namespace cyclic_quad_incenters_form_rectangle_l263_263477

-- Definitions used in the conditions
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def cyclic_quadrilateral (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
∃ (O : Type) [metric_space O], metric_space.is_cyclic_quad O A B C D

def incenter (Δ : Type) [triangle Δ] : Type := 
angle_bisectors.intersect (Δ.angle_bisectors)

-- Definition of a rectangle from its vertices
def is_rectangle (I₁ I₂ I₃ I₄ : Type) [metric_space I₁] [metric_space I₂] [metric_space I₃] [metric_space I₄] : Prop :=
(perpendicular I₁ I₃) ∧ (perpendicular I₂ I₄)

-- Define the rectangles formed by incenters 
theorem cyclic_quad_incenters_form_rectangle
  {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (h : cyclic_quadrilateral A B C D) :
  ∃ I₁ I₂ I₃ I₄, 
    I₁ = incenter (triangle ABC) ∧ 
    I₂ = incenter (triangle BCD) ∧ 
    I₃ = incenter (triangle CDA) ∧ 
    I₄ = incenter (triangle DAB) ∧ 
    is_rectangle I₁ I₂ I₃ I₄ :=
sorry

end cyclic_quad_incenters_form_rectangle_l263_263477


namespace large_marshmallows_are_eight_l263_263941

-- Definition for the total number of marshmallows
def total_marshmallows : ℕ := 18

-- Definition for the number of mini marshmallows
def mini_marshmallows : ℕ := 10

-- Definition for the number of large marshmallows
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

-- Theorem stating that the number of large marshmallows is 8
theorem large_marshmallows_are_eight : large_marshmallows = 8 := by
  sorry

end large_marshmallows_are_eight_l263_263941


namespace minimum_tiles_no_move_l263_263615

def tile (p: ℕ × ℕ) := 
  {p' : ℕ × ℕ // (p'.1 = p.1 + 1 ∨ p'.1 = p.1 - 1 ∨ p'.2 = p.2 + 1 ∨ p'.2 = p.2 - 1) ∧ p'.1 < 8 ∧ p'.2 < 8}

def no_mv_tile (tiles : finset (ℕ × ℕ)) := 
  ∀ p ∈ tiles, ∀ t, t ∈ tile p → t ∉ tiles

def min_tiles_8x8_table := 
  ∃ n, 0 < n ∧ no_mv_tile (finset.range n) ∧ ∀ m, (0 < m ∧ m < n) → ¬no_mv_tile (finset.range m)

theorem minimum_tiles_no_move : min_tiles_8x8_table → ∃ n, n = 28 := 
sorry

end minimum_tiles_no_move_l263_263615


namespace polynomial_expansion_l263_263154

theorem polynomial_expansion :
  let p := 5 * x^2 - 3 * x + 7 in
  let q := 9 - 4 * x in
  ∃ a b c d : ℝ, 
  (p * q = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = -29 :=
by sorry

end polynomial_expansion_l263_263154


namespace inverse_proposition_is_false_l263_263142

theorem inverse_proposition_is_false (a : ℤ) (h : a = 6) : ¬ (|a| = 6 → a = 6) :=
sorry

end inverse_proposition_is_false_l263_263142


namespace angle_CFD_right_l263_263585

noncomputable def midpoint {P : Type} [add_comm_group P] (a b : P) : P :=
1/2 • (a + b)

variables {A B C D E F : Type} [add_comm_group A] [vector_space ℝ A]
variables [add_comm_group B] [vector_space ℝ B]
variables [add_comm_group C] [vector_space ℝ C]
variables [add_comm_group D] [vector_space ℝ D]
variables [add_comm_group E] [vector_space ℝ E]
variables [add_comm_group F] [vector_space ℝ F]

variables {ABCD : Type} [parallelogram ABCD]
variables {AD : ℝ} [parallelogram AD]
variables {BF : ℝ} [parallelogram BF]

theorem angle_CFD_right (midpt_E : midpoint A B = E)
  (F_on_DE : F ∈ segment D E)
  (AD_eq_BF : AD = BF) :
  ∠ CFD = 90 :=
sorry

end angle_CFD_right_l263_263585


namespace solve_for_b_l263_263085

/-- 
Given the ellipse \( x^2 + \frac{y^2}{b^2 + 1} = 1 \) where \( b > 0 \),
and the eccentricity of the ellipse is \( \frac{\sqrt{10}}{10} \),
prove that \( b = \frac{1}{3} \).
-/
theorem solve_for_b (b : ℝ) (hb : b > 0) (heccentricity : b / (Real.sqrt (b^2 + 1)) = Real.sqrt 10 / 10) : 
  b = 1 / 3 :=
sorry

end solve_for_b_l263_263085


namespace find_some_number_l263_263306

theorem find_some_number : 
  let x := -5765435 in 
  7^8 - 6 / 2 + 9^3 + 3 + x = 95 :=
by
  let x := -5765435
  sorry

end find_some_number_l263_263306


namespace two_month_stay_62_days_l263_263325

-- Define the two-month period stated in the problem
structure TwoMonths where
  month1 : String
  month2 : String

-- List of long months
def long_months : List String :=
  ["January", "March", "May", "July", "August", "October", "December"]

-- Define the target pairs of months
def valid_months (m1 m2 : String) : TwoMonths :=
  (m1, m2)

-- Lean statement proving the two months of stay
theorem two_month_stay_62_days (m1 m2 : String) (h1 : m1 ∈ long_months) (h2 : m2 ∈ long_months) :
  (m1 = "July" ∧ m2 = "August") ∨ (m1 = "December" ∧ m2 = "January") := 
sorry

end two_month_stay_62_days_l263_263325


namespace problem_statement_l263_263929

noncomputable def f (ω ϕ x : ℝ) : ℝ := sin (ω * x + ϕ) + cos (ω * x + ϕ)

theorem problem_statement
  (ω : ℝ) (ϕ : ℝ)
  (hω : ω > 0)
  (hϕ : |ϕ| < (π / 2))
  (h_period : ∀ x, f ω ϕ x = f ω ϕ (x + π))
  (h_even : ∀ x, f ω ϕ (-x) = f ω ϕ x) :
  ∀ x, 0 < x ∧ x < (π / 2) → f ω ϕ x ≥ f ω ϕ (x + π / 2) :=
sorry

end problem_statement_l263_263929


namespace number_of_ways_to_lineup_five_people_l263_263504

noncomputable def numPermutations (people : List Char) (constraints : List (Char × Char)) : Nat :=
  List.factorial people.length / ∏ (c : Char × Char) in constraints, (match c.1 with
    | 'A' => (people.length - 1) -- A cannot be first
    | 'E' => (people.length - 1) -- E cannot be last
    | _ => people.length) 

theorem number_of_ways_to_lineup_five_people : 
  numPermutations ['A', 'B', 'C', 'D', 'E'] [('A', 'First-line'), ('E', 'Last-line')] = 96 := 
sorry

end number_of_ways_to_lineup_five_people_l263_263504


namespace minimum_number_of_gloves_needed_l263_263983
-- Import the necessary library

-- Problem conditions and statement
theorem minimum_number_of_gloves_needed (number_of_participants : Nat) (h : number_of_participants = 82) : 
  let gloves_per_participant := 2 in
  number_of_participants * gloves_per_participant = 164 :=
by
  -- Using the given condition
  rw [h]
  -- Simplifying the left-hand side
  simp [gloves_per_participant]
  -- Concluding the proof
  sorry

end minimum_number_of_gloves_needed_l263_263983


namespace monotonic_intervals_extreme_values_max_min_on_interval_l263_263457

def f (x : ℝ) : ℝ := 4 * x ^ 3 - 3 * x ^ 2 - 18 * x + 27

theorem monotonic_intervals_extreme_values : 
  (∀ x < -1, f x < f (-1)) ∧ 
  (∀ (x : ℝ), -1 < x ∧ x < (3/2) → f x < f (-1) ∧ f x > f (3/2)) ∧ 
  (∀ x > (3/2), f x > f (3/2)) ∧ 
  (f (-1) = 38) ∧ 
  (f (3 / 2) = 27 / 4) := 
sorry

theorem max_min_on_interval : 
  (∀ x ∈ set.Icc (0:ℝ) (3:ℝ), f x ≤ 54) ∧ 
  (∃ x ∈ set.Icc (0:ℝ) (3:ℝ), f x = 54) ∧ 
  (∀ x ∈ set.Icc (0:ℝ) (3:ℝ), f x ≥ 27 / 4) ∧ 
  (∃ x ∈ set.Icc (0:ℝ) (3:ℝ), f x = 27 / 4) := 
sorry

end monotonic_intervals_extreme_values_max_min_on_interval_l263_263457


namespace num_values_of_n_l263_263799

def f (n : ℤ) : ℤ := 2 * n^5 + 3 * n^4 + 5 * n^3 + 2 * n^2 + 3 * n + 6

theorem num_values_of_n : 
  (finset.card (finset.filter (λ n, f n % 7 = 0) (finset.Icc 2 100))) = 14 :=
by
  sorry

end num_values_of_n_l263_263799


namespace probability_not_exceeding_40_l263_263309

variable (P : ℝ → Prop)

def less_than_30_grams : Prop := P 0.3
def between_30_and_40_grams : Prop := P 0.5

theorem probability_not_exceeding_40 (h1 : less_than_30_grams P) (h2 : between_30_and_40_grams P) : P 0.8 :=
by
  sorry

end probability_not_exceeding_40_l263_263309


namespace impossible_to_place_19_bishops_l263_263202

theorem impossible_to_place_19_bishops :
  ∀ (board : matrix (fin 4) (fin 16) ℕ) (bishops : fin 19 → (fin 4) × (fin 16)),
  ¬ ∃ placement : fin 19 → (fin 4) × (fin 16),
    (∀ i j : fin 19, i ≠ j → (placement i).fst - (placement i).snd ≠ (placement j).fst - (placement j).snd) ∧
    (∀ i j : fin 19, i ≠ j → (placement i).fst + (placement i).snd ≠ (placement j).fst + (placement j).snd) := 
by sorry

end impossible_to_place_19_bishops_l263_263202


namespace tian_ji_wins_probability_l263_263906

-- Define the types for horses and their relative rankings
inductive horse
| king_top : horse
| king_middle : horse
| king_bottom : horse
| tian_top : horse
| tian_middle : horse
| tian_bottom : horse

open horse

-- Define the conditions based on the problem statement
def better_than : horse → horse → Prop
| tian_top king_middle := true
| tian_top king_top := false
| tian_middle king_bottom := true
| tian_middle king_middle := false
| tian_bottom king_bottom := false
| _ _ := false

-- Topic condition for probability
def is_win (tian_horse : horse) (king_horse : horse) : Prop :=
(tian_horse = tian_top ∧ (king_horse = king_middle ∨ king_horse = king_bottom)) ∨
(tian_horse = tian_middle ∧ king_horse = king_bottom)

-- The probability statement
def win_probability : ℚ := 1/3

-- Main theorem statement
theorem tian_ji_wins_probability :
  (∑ tian_horse king_horse,
     cond (is_win tian_horse king_horse) 1 0) / 9 = win_probability :=
begin
  -- Proof is omitted
  sorry
end

end tian_ji_wins_probability_l263_263906


namespace fourth_vertex_of_square_l263_263644

theorem fourth_vertex_of_square :
  ∃ (d : ℂ), set_of (λ x : ℂ, x ∈ ({2 + complex.i, -1 + 2 * complex.i, -2 - complex.i, d}).to_finset) =ᶠ[{2+complex.i, -1+2*complex.i, -2-complex.i, 1-2*complex.i}.to_finset] :=
sorry

end fourth_vertex_of_square_l263_263644


namespace probability_sum_divisible_by_3_l263_263284

theorem probability_sum_divisible_by_3 :
  let balls := {1, 3, 5, 7, 9}
  let all_combinations := Finset.powersetLen 3 (Finset.of_array balls)
  let favorable_combinations := all_combinations.filter (λ s, s.sum % 3 = 0)
  (favorable_combinations.card / all_combinations.card : ℚ) = 2 / 5 :=
by {
  sorry
}

end probability_sum_divisible_by_3_l263_263284


namespace problem_l263_263147

def f (u : ℝ) : ℝ := u^2 - 2

theorem problem : f 3 = 7 := 
by sorry

end problem_l263_263147


namespace f_solution_set_l263_263077

theorem f_solution_set (f : ℝ → ℝ) (f' : ∀ x, Deriv f x = f' x)
  (h_deriv : ∀ x : ℝ, f' x - f x < 1) (h_init : f 0 = 2022) :
  ∀ x, (f x + 1 > 2023 * Real.exp x) ↔ x < 0 :=
by
  sorry

end f_solution_set_l263_263077


namespace evaluate_f_f_neg_half_l263_263569

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 3 ^ x else Real.log x / Real.log 3

theorem evaluate_f_f_neg_half : f (f (-1 / 2)) = -1 / 2 := by
  sorry

end evaluate_f_f_neg_half_l263_263569


namespace best_starting_day_for_coupons_l263_263237

-- Definition of the days of the week as an enumeration
inductive Day
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open Day

-- Melanie has 8 coupons and uses them every 7 days, bakery is closed on Monday
def couponRedemption (start : Day) (n : Nat) : Day :=
  match start with
  | Monday    => match n % 7 with | 0 => Monday | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday
  | Tuesday   => match n % 7 with | 0 => Tuesday | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday
  | Wednesday => match n % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday
  | Thursday  => match n % 7 with | 0 => Thursday | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday
  | Friday    => match n % 7 with | 0 => Friday | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday
  | Saturday  => match n % 7 with | 0 => Saturday | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday
  | Sunday    => match n % 7 with | 0 => Sunday | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday

-- Prove starting on Sunday none of the redemptions falls on Monday
theorem best_starting_day_for_coupons : ∀ (n : Nat), n < 8 → couponRedemption Sunday n ≠ Monday :=
by
  sorry

end best_starting_day_for_coupons_l263_263237


namespace point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l263_263467

theorem point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb
  (x1 x2 : ℝ) : 
  (x1 * x2 / 4 = -1) ↔ ((x1 / 2) * (x2 / 2) = -1) :=
by sorry

end point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l263_263467


namespace find_angle_B_perimeter_range_vector_dot_product_l263_263198

-- Part 1: Prove that given the equation, angle B is pi/3
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h1 : a * sin A + a * sin C * cos B + b * sin C * cos A = b * sin B + c * sin A) : B = π / 3 :=
sorry

-- Part 2: Prove the range of the perimeter of triangle ABC when a = 2 and triangle is acute
theorem perimeter_range (A B C : ℝ) (a b c : ℝ) (h1 : a = 2) (h2 : π / 6 < A ∧ A < π / 2) (h3 : triangle_acute A B C) : 
∀ P, P ∈ Set.Ioo (3 + sqrt 3) (6 + 2 * sqrt 3) :=
sorry

-- Part 3: Prove the range of PA ⋅ PB given the circle conditions
theorem vector_dot_product (a b c R : ℝ) (A B C : ℝ) (O P : ℝ) (h1 : b^2 = a*c) (h2 : R = 2) (h3 : is_circumcenter O P) (h4 : P_is_on_circle O) : 
∀ dot_prod, dot_prod ∈ Set.Icc (-2) 6 :=
sorry

end find_angle_B_perimeter_range_vector_dot_product_l263_263198


namespace f_inequality_l263_263229

-- Given conditions on the function f
variable (f : ℚ → ℚ)
variable (h : ∀ (m n : ℚ), |f (m + n) - f m| ≤ n / m)

-- Formalizing the problem statement
theorem f_inequality (k : ℕ) (hk : 0 < k) :
  ∑ i in Finset.range k + 1, |f (2^k) - f (2^i)| ≤ k * (k - 1) / 2 :=
by sorry

end f_inequality_l263_263229


namespace y_coord_at_x_eq_10_l263_263192

theorem y_coord_at_x_eq_10
  (x1 y1 x2 y2 : ℝ)
  (hx1 : x1 = -2)
  (hy1 : y1 = -3)
  (hx2 : x2 = 4)
  (hy2 : y2 = 0)
  (m : ℝ)
  (hm : m = (y2 - y1) / (x2 - x1))
  (b : ℝ)
  (hb : b = y2 - m * x2)
  (x : ℝ)
  (hx : x = 10) :
  let y := m * x + b in y = 3 := by
  sorry

end y_coord_at_x_eq_10_l263_263192


namespace sequence_satisfies_n_squared_l263_263469

theorem sequence_satisfies_n_squared (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) :
  ∀ n, a n = n^2 :=
by
  -- sorry
  sorry

end sequence_satisfies_n_squared_l263_263469


namespace outlier_count_is_one_l263_263377

def data_set : List ℕ := [4, 21, 34, 34, 40, 42, 42, 44, 52, 59]
def Q1 := 34
def Q3 := 44
def IQR := Q3 - Q1
def lower_threshold := Q1 - 1.5 * IQR
def upper_threshold := Q3 + 1.5 * IQR

def is_outlier (x : ℕ) : Prop :=
  (x < lower_threshold) ∨ (x > upper_threshold)

def count_outliers (data : List ℕ) : ℕ :=
  data.countp is_outlier

theorem outlier_count_is_one : count_outliers data_set = 1 := by
  sorry

end outlier_count_is_one_l263_263377


namespace jenny_total_wins_l263_263529

-- Definitions based on conditions
def games_mark : Nat := 10
def mark_wins : Nat := 1
def jill_wins_percent : Real := 0.75

-- Calculations based on definitions
def jenny_wins_mark : Nat := games_mark - mark_wins
def games_jill : Nat := 2 * games_mark
def jill_wins : Nat := floor (jill_wins_percent * games_jill).toNat -- convert from Real to Nat
def jenny_wins_jill : Nat := games_jill - jill_wins

-- Total wins
def total_wins : Nat := jenny_wins_mark + jenny_wins_jill

theorem jenny_total_wins : total_wins = 14 := by
  -- proof goes here
  sorry

end jenny_total_wins_l263_263529


namespace bicycle_meets_scooter_l263_263415

noncomputable def speeds (v_A v_R v_M v_K : ℝ) : Prop :=
  let d_1 := 2 * (v_A + v_K)
  let d_2 := 4 * (v_A + v_M)
  let t := (10 / 3)
  (d_1 = t * (v_R + v_K)) ∧
  (d_2 = 5 * (v_R + v_M)) ∧
  (d_2 - d_1 = 6 * (v_M - v_K)) ∧
  (d_2 = 4 * d_1)

theorem bicycle_meets_scooter 
  (v_A v_R v_M v_K : ℝ) 
  (h: speeds v_A v_R v_M v_K) : 
  12 + (10 / 3) = 15 + (20 / 60) :=
begin
  sorry
end

end bicycle_meets_scooter_l263_263415


namespace locus_of_Q_max_area_of_triangle_OPQ_l263_263314

open Real

theorem locus_of_Q (x y : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x = 3 * x_0 ∧ y = 4 * y_0 →
  (x / 6)^2 + (y / 4)^2 = 1 :=
sorry

theorem max_area_of_triangle_OPQ (S : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x_0 > 0 ∧ y_0 > 0 →
  S <= sqrt 3 / 2 :=
sorry

end locus_of_Q_max_area_of_triangle_OPQ_l263_263314


namespace grade12_students_selected_l263_263645

theorem grade12_students_selected 
    (N : ℕ) (n10 : ℕ) (n12 : ℕ) (k : ℕ) 
    (h1 : N = 1200)
    (h2 : n10 = 240)
    (h3 : 3 * N / (k + 5 + 3) = n12)
    (h4 : k * N / (k + 5 + 3) = n10) :
    n12 = 360 := 
by sorry

end grade12_students_selected_l263_263645


namespace ninth_term_is_83_l263_263262

-- Definitions based on conditions
def a : ℕ := 3
def d : ℕ := 10
def arith_sequence (n : ℕ) : ℕ := a + n * d

-- Theorem to prove the 9th term is 83
theorem ninth_term_is_83 : arith_sequence 8 = 83 :=
by
  sorry

end ninth_term_is_83_l263_263262


namespace pen_ratio_l263_263753

theorem pen_ratio 
  (Dorothy_pens Julia_pens Robert_pens : ℕ)
  (pen_cost total_cost : ℚ)
  (h1 : Dorothy_pens = Julia_pens / 2)
  (h2 : Robert_pens = 4)
  (h3 : pen_cost = 1.5)
  (h4 : total_cost = 33)
  (h5 : total_cost / pen_cost = Dorothy_pens + Julia_pens + Robert_pens) :
  (Julia_pens / Robert_pens : ℚ) = 3 :=
  sorry

end pen_ratio_l263_263753


namespace mow_lawn_time_l263_263933

noncomputable def time_to_mow (lawn_length lawn_width: ℝ) 
(swat_width overlap width_conversion: ℝ) (speed: ℝ) : ℝ :=
(lawn_length * lawn_width) / (((swat_width - overlap) / width_conversion) * lawn_length * speed)

theorem mow_lawn_time : 
  time_to_mow 120 180 30 6 12 6000 = 1.8 := 
by
  -- Given:
  -- Lawn dimensions: 120 feet by 180 feet
  -- Mower swath: 30 inches with 6 inches overlap
  -- Walking speed: 6000 feet per hour
  -- Conversion factor: 12 inches = 1 foot
  sorry

end mow_lawn_time_l263_263933


namespace min_magnitude_is_sqrt2_l263_263096

noncomputable def vec_a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)
noncomputable def vec_b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
noncomputable def vec_sub (t : ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := vec_a t
  let (b1, b2, b3) := vec_b t
  (b1 - a1, b2 - a2, b3 - a3)

noncomputable def vec_magnitude (t : ℝ) : ℝ :=
  let (x, y, z) := vec_sub t
  real.sqrt (x^2 + y^2 + z^2)

theorem min_magnitude_is_sqrt2 : ∀ t : ℝ, ∃ t0 : ℝ, vec_magnitude t0 = real.sqrt 2 :=
by
  use 0
  simp [vec_a, vec_b, vec_sub, vec_magnitude]
  sorry

end min_magnitude_is_sqrt2_l263_263096


namespace min_area_of_triangle_PCD_l263_263434

noncomputable def min_area_PCD (s : ℝ) (SC : ℝ) (se_xy : ℝ → ℝ × ℝ × ℝ) : ℝ :=
  let CD := (sqrt 3 / 2) * s
  let min_area := (1/2) * CD * ((2 * sqrt 3) / 3)
  in 2 * sqrt 2

theorem min_area_of_triangle_PCD :
  (∀ (P ∈ line_segment SE), P ∈ { p | p = list.nil ∨ ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) • S + t • E = p }) →
  min_area_PCD (4 * sqrt 2) 2 (λ t, (2 * sqrt 3 / 3 * t, _, _)) = 2 * sqrt 2 :=
sorry

end min_area_of_triangle_PCD_l263_263434


namespace remainder_of_power_mod_l263_263772

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l263_263772


namespace shaded_area_l263_263189

theorem shaded_area (R : ℝ) (r : ℝ) (hR : R = 10) (hr : r = R / 2) : 
  π * R^2 - 2 * (π * r^2) = 50 * π :=
by
  sorry

end shaded_area_l263_263189


namespace remainder_of_exponentiation_is_correct_l263_263791

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l263_263791


namespace average_a_b_l263_263614

theorem average_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27) : (A + B) / 2 = 40 := 
by
  sorry

end average_a_b_l263_263614


namespace increasing_interval_log_function_l263_263272

noncomputable def log_function (x : ℝ) : ℝ := real.log (x^2 - 1)

theorem increasing_interval_log_function :
  ∀ x y : ℝ, 1 < x → 1 < y → x < y → log_function x < log_function y :=
by
  intros x y hx hy hxy
  sorry

end increasing_interval_log_function_l263_263272


namespace base_four_to_base_ten_of_20314_eq_568_l263_263299

-- Define what it means to convert a base-four number to base-ten
def base_four_to_base_ten (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (λ ⟨index, digit⟩ acc => acc + digit * 4^index) 0

-- Define the specific base-four number 20314_4 as a list of its digits
def num_20314_base_four : List ℕ := [2, 0, 3, 1, 4]

-- Theorem stating that the base-ten equivalent of 20314_4 is 568
theorem base_four_to_base_ten_of_20314_eq_568 : base_four_to_base_ten num_20314_base_four = 568 := sorry

end base_four_to_base_ten_of_20314_eq_568_l263_263299


namespace q_value_l263_263919

noncomputable def q (x : ℕ) (d e : ℤ) := x^2 + d * x + e

theorem q_value (d e : ℤ) (h1 : ∃ d e : ℤ, (λ x, x^2 + d * x + e) ∣ (λ x, x^4 + 8 * x^2 + 49))
                           (h2 : ∃ d e : ℤ, (λ x, x^2 + d * x + e) ∣ (λ x, 2 * x^4 + 5 * x^2 + 36 * x + 6)) :
  q 1 (-18) 49 = 32 :=
by
  unfold q
  simp
  sorry

end q_value_l263_263919


namespace A_pays_6_sum_of_fees_36_l263_263496

-- Definitions for the problem
def charge (hours : Nat) : Nat :=
  if hours <= 1 then 6 else 6 + 8 * (hours - 1)

def prob_parking_A (hours : Nat) : Rat :=
  match hours with
  | 1 => 1 - (1/3 + 1/4 + 1/6)
  | 2 => 1/3
  | 3 => 1/4
  | 4 => 1/6
  | _ => 0

def prob_parking_B (hours : Nat) : Rat :=
  match hours with
  | 1 => 1/2 - 1/4
  | 2, 3 => 1/4
  | 4 => 1/2
  | _ => 0

def prob_A_pays_6 : Rat :=
  prob_parking_A 1

def scenarios_A_B : List (Nat × Nat) :=
  [(6, 6), (6, 14), (6, 22), (6, 30),
   (14, 6), (14, 14), (14, 22), (14, 30),
   (22, 6), (22, 14), (22, 22), (22, 30),
   (30, 6), (30, 14), (30, 22), (30, 30)]

def valid_scenarios : List (Nat × Nat) :=
  scenarios_A_B.filter (fun (a_b : Nat × Nat) => a_b.1 + a_b.2 = 36)

def prob_sum_36 : Rat :=
  valid_scenarios.length / scenarios_A_B.length

-- Statements to be proved
theorem A_pays_6 : prob_A_pays_6 = 1/4 := by sorry

theorem sum_of_fees_36 : prob_sum_36 = 1/4 := by sorry

end A_pays_6_sum_of_fees_36_l263_263496


namespace difference_between_mean_and_median_l263_263177

def scores : List ℝ := [60, 75, 85, 90, 100]
def percentages : List ℝ := [0.15, 0.20, 0.25, 0.25, 0.15]

noncomputable def mean_score (scores : List ℝ) (percentages : List ℝ) : ℝ :=
  (List.zipWith (λ s p => s * p * 40) scores percentages).sum / 40

noncomputable def median_score : ℝ := scores.nthLe 2 (by simp [List.length_eq])   -- nthLe function gets the element assuming the list is sorted and 0-based index.

theorem difference_between_mean_and_median : 
  mean_score scores percentages - median_score = 2.25 :=
by
  sorry

end difference_between_mean_and_median_l263_263177


namespace sum_powers_of_i_l263_263258

def pow_i_cycle : ℕ → ℂ
| 0 => 1
| 1 => complex.I
| 2 => -1
| 3 => -complex.I
| (n + 4) => pow_i_cycle n

theorem sum_powers_of_i : (i_sum : ℂ) → (i_sum = ∑ n in finset.range 2014, pow_i_cycle n) ∧ i_sum = 1 + complex.I :=
by
  existsi ((∑ n in finset.range 2014, pow_i_cycle n) : ℂ)
  split
  · exact rfl
  · sorry

end sum_powers_of_i_l263_263258


namespace max_min_values_f_decreasing_interval_f_l263_263097

noncomputable def a : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := ((a.1 * (b x).1) + (a.2 * (b x).2)) + 2

theorem max_min_values_f (k : ℤ) :
  (∃ (x1 : ℝ), (x1 = 2 * k * Real.pi + Real.pi / 6) ∧ f x1 = 3) ∧
  (∃ (x2 : ℝ), (x2 = 2 * k * Real.pi - 5 * Real.pi / 6) ∧ f x2 = 1) := 
sorry

theorem decreasing_interval_f :
  ∀ x, (Real.pi / 6 ≤ x ∧ x ≤ 7 * Real.pi / 6) → (∀ y, f x ≥ f y → x ≤ y) := 
sorry

end max_min_values_f_decreasing_interval_f_l263_263097


namespace smallest_b_periodic_l263_263923

def f : ℝ → ℝ := sorry  -- The function f is arbitrary for now

def g (x : ℝ) : ℝ := f (2 * x / 5)

axiom f_periodic : ∀ x : ℝ, f (x + 10) = f x

theorem smallest_b_periodic (b : ℝ) (hb_pos : 0 < b) :
    (∀ x : ℝ, g (x - b) = g x) ↔ b = 25 :=
by
sorry

end smallest_b_periodic_l263_263923


namespace kendra_and_tony_keep_two_each_l263_263905

-- Define the conditions
def kendra_packs : Nat := 4
def tony_packs : Nat := 2
def pens_per_pack : Nat := 3
def pens_given_to_friends : Nat := 14

-- Define the total pens each has
def kendra_pens : Nat := kendra_packs * pens_per_pack
def tony_pens : Nat := tony_packs * pens_per_pack

-- Define the total pens
def total_pens : Nat := kendra_pens + tony_pens

-- Define the pens left after distribution
def pens_left : Nat := total_pens - pens_given_to_friends

-- Define the number of pens each keeps
def pens_each_kept : Nat := pens_left / 2

-- Prove the final statement
theorem kendra_and_tony_keep_two_each :
  pens_each_kept = 2 :=
by
  sorry

end kendra_and_tony_keep_two_each_l263_263905


namespace max_profit_at_60_l263_263352

/-- Definitions for the functional relationships -/
def ticket_price (x : ℤ) : ℤ :=
  if x <= 30 then 900
  else (1200 - 10 * x)

/-- Profit calculation based on the number of people -/
def profit (x : ℤ) : ℤ :=
  if x <= 30 then 900 * x - 15000
  else -10 * x * x + 1200 * x - 15000

/-- Proof that follows from the conditions -/
theorem max_profit_at_60 : ∃ (x : ℤ), 0 <= x ∧ x <= 75 ∧ profit x = 21000 :=
begin
  use 60,
  split, linarith,
  split, linarith,
  calc
    profit 60 = -10 * 60 * 60 + 1200 * 60 - 15000 : rfl
           ... = 21000 : by norm_num,
  sorry
end

end max_profit_at_60_l263_263352


namespace sin_product_inequality_triangle_l263_263525

theorem sin_product_inequality_triangle (A B C : ℝ) (hA_gt_0 : 0 < A) (hB_gt_0 : 0 < B) (hC_gt_0 : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π)
  (hSum : A + B + C = π) : 
  sin A * sin B * sin C ≤ 3 * sqrt 3 / 8 :=
sorry

end sin_product_inequality_triangle_l263_263525


namespace _l263_263656

noncomputable theorem find_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
(h1 : a * b = 5) (h2 : 2 * a * b / (a + b) = 5 / 3) : (a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1) :=
by sorry

end _l263_263656


namespace equation_holds_true_l263_263022

theorem equation_holds_true (a b : ℝ) (h₁ : a ≠ 0) (h₂ : 2 * b - a ≠ 0) :
  ((a + 2 * b) / a = b / (2 * b - a)) ↔ 
  (a = -b * (1 + Real.sqrt 17) / 2 ∨ a = -b * (1 - Real.sqrt 17) / 2) := 
sorry

end equation_holds_true_l263_263022


namespace domain_h_parity_h_h_pos_x_set_l263_263840

variable (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

def f (x : ℝ) : ℝ := log a (1 + x)
def g (x : ℝ) : ℝ := log a (1 - x)
def h (x : ℝ) : ℝ := f a x - g a x

theorem domain_h : ∀ x, (-1 < x ∧ x < 1) ↔ (∃ x, f a x ∈ ℝ ∧ g a x ∈ ℝ) := by
  sorry

theorem parity_h : ∀ x, h a (-x) = -h a x := by
  sorry

theorem h_pos_x_set : f a 3 = 2 → (∀ x, 0 < x ∧ x < 1 → h a x > 0) := by
  sorry

end domain_h_parity_h_h_pos_x_set_l263_263840


namespace inequality_solution_set_l263_263078

variable (f : ℝ → ℝ)

theorem inequality_solution_set (h_deriv : ∀ x : ℝ, f' x - f x < 1)
  (h_initial : f 0 = 2022) :
  ∀ x : ℝ, f x + 1 > 2023 * Real.exp x ↔ x < 0 :=
by
  intro x
  sorry

end inequality_solution_set_l263_263078


namespace max_red_points_l263_263641

theorem max_red_points (n : ℕ) (h : n = 100)
  (colored : Fin n → Bool) -- True for red, False for blue
  (segments : Fin n × Fin n → Prop) -- (i, j) where colored i ≠ colored j and a segment exists
  (unique_red_connections : ∀ i j : Fin n, colored i = true → colored j = true → 
                            (∑ k : Fin n, if segments (i, k) then 1 else 0) ≠ 
                            (∑ k : Fin n, if segments (j, k) then 1 else 0)) :
  ∃ m : ℕ, m = 50 ∧ (∀ k : ℕ, m < k → ∃ i j : Fin n, colored i = true ∧ colored j = true ∧ 
                      (∑ l : Fin n, if segments (i, l) then 1 else 0) = 
                      (∑ l : Fin n, if segments (j, l) then 1 else 0)) :=
sorry

end max_red_points_l263_263641


namespace trapezoid_ad_length_mn_l263_263650

open EuclideanGeometry

variables {A B C D O P : Point}
variables {m n : ℕ}

-- Given conditions
def is_trapezoid (A B C D : Point) : Prop := 
  A.y = B.y ∧ C.y = D.y ∧ B.x - A.x ≠ D.x - C.x

def length_eq (x y : ℕ) : Prop := 
  x = 43 ∧ y = 43

def perpendicular (A D B : Point) : Prop := 
  (A.x - D.x) * (D.x - B.x) + (A.y - D.y) * (D.y - B.y) = 0

def midpoint (P B D : Point) : Prop := 
  2 * P.x = B.x + D.x ∧ 2 * P.y = B.y + D.y

def inter_diag (A C B D O : Point) : Prop := 
  ∃ λ : ℝ, O = λ • A + (1 - λ) • C ∧  ∃ μ : ℝ, O = μ • B + (1 - μ) • D

def OP_length (O P : Point) (l : ℝ) : Prop := 
  dist O P = l

-- Prove the final tuple
theorem trapezoid_ad_length_mn (hT : is_trapezoid A B C D) (hL : length_eq (dist B C) (dist C D))
  (hP : perpendicular A D B) (hM : midpoint P B D) (hI : inter_diag A C B D O)
  (hO : OP_length O P 11) : 
  ∃ (m n : ℕ), dist A D = m * Real.sqrt n ∧ m + n = 194 := 
sorry

end trapezoid_ad_length_mn_l263_263650


namespace park_paths_total_length_l263_263880

def path_lengths (x : ℝ) : ℝ :=
  let straight_paths_1 := 10 * (30 + x)
  let straight_paths_2 := 6 * 60
  let circular_path := 150 * Real.pi
  straight_paths_1 + straight_paths_2 + circular_path

theorem park_paths_total_length : 
  ∀ (x : ℝ), x^2 + 60^2 = (x + 30)^2 → path_lengths 45 ≈ 1581.24 := by
  intro x h
  have h₁ : x = 45 := sorry
  rw [h₁, path_lengths]
  norm_num
  sorry

end park_paths_total_length_l263_263880


namespace find_eccentricity_l263_263011

section EllipseEccentricity
variable {a b c : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c^2 = a^2 - b^2)
variable (h4 : ∃ (AF AB BF : ℝ), AF = a - c ∧ AB = sqrt (a^2 + b^2) ∧ 3 * BF = 3 * a ∧ (AF * (3 * BF)) = AB^2)

theorem find_eccentricity : 
  let e := c / a in e = (sqrt 5 - 1) / 2 :=
by
  sorry

end find_eccentricity_l263_263011


namespace g_is_odd_function_l263_263902

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  sorry

end g_is_odd_function_l263_263902


namespace intersection_of_M_and_N_l263_263470

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}
def intersection_M_N : Set ℕ := {0, 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := by
  sorry

end intersection_of_M_and_N_l263_263470


namespace non_congruent_triangles_with_perimeter_11_l263_263137

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263137


namespace song_liking_count_l263_263734

theorem song_liking_count :
  let individuals := {chris, dana, eli, fran} in
  let songs := {s1, s2, s3, s4, s5} in
  -- Condition 1: No song is liked by all four
  (∀ s ∈ songs, ¬(chris ∈ s ∧ dana ∈ s ∧ eli ∈ s ∧ fran ∈ s)) →
  -- Condition 2: For each pair, there is at least one song liked by those two but disliked by others
  (∀ (i1 i2 : individuals) (h : i1 ≠ i2), ∃ s ∈ songs, (i1 ∈ s ∧ i2 ∈ s ∧ (∀ (i : individuals), (i = i1 ∨ i = i2) → i ∉ s))) →
  -- Condition 3: Exactly one song is liked by only one person
  (∃ s ∈ songs, ∃ i ∈ individuals, (∀ (j : individuals), j ≠ i → j ∉ s) ∧ (∀ t ∈ songs, t ≠ s → (∃ (k : individuals), (∀ y ∈ individuals, y ≠ k → y ∉ t) → False))) →
  -- Prove the total number of ways
  finset.card {config | satisfies_conditions config} = 4320 := 
by sorry

end song_liking_count_l263_263734


namespace lemonade_problem_l263_263232

theorem lemonade_problem (L S W : ℕ) (h1 : W = 4 * S) (h2 : S = 2 * L) (h3 : L = 3) : L + S + W = 24 :=
by
  sorry

end lemonade_problem_l263_263232


namespace simplify_sum_powers_of_i_l263_263251

open Complex
open Finset

noncomputable def sum_powers_of_i : ℂ :=
∑ i in range (2014), (I ^ i)

theorem simplify_sum_powers_of_i :
  sum_powers_of_i = 1 + I :=
by
  -- Proof here
  sorry

end simplify_sum_powers_of_i_l263_263251


namespace log_base_625_of_x_l263_263864

theorem log_base_625_of_x 
  (h: log 9 (x - 2) = 1 / 2) : log 625 x = 1 / 4 :=
sorry

end log_base_625_of_x_l263_263864


namespace paperclips_in_larger_box_l263_263328

theorem paperclips_in_larger_box (paperclips_per_24cm3 : ℕ) (volume_small_box volume_large_box : ℕ)
  (H_small : paperclips_per_24cm3 = 75) (H_volumes : volume_small_box = 24) (H_scale : volume_large_box = 60) :
  let paperclips_per_cm3 := paperclips_per_24cm3 / volume_small_box
  let expected_paperclips := (paperclips_per_cm3 * volume_large_box : ℝ).round
  expected_paperclips = 188 :=
by
  sorry

end paperclips_in_larger_box_l263_263328


namespace find_x_l263_263914

def oslash (a b : ℝ) : ℝ := (sqrt (3 * a + b))^3

theorem find_x (x : ℝ) (h : oslash 7 x = 125) : x = 4 := by
  sorry

end find_x_l263_263914


namespace find_p_q_r_s_l263_263549

def Q (x : ℝ) : ℝ := x^2 - 5 * x - 4

def interval_valid (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 12

def probability_condition (x : ℝ) : Prop :=
  ⌊ sqrt (Q x) ⌋ = sqrt (Q ⌊ x ⌋)

theorem find_p_q_r_s (p q r s : ℕ)
  (h1 : ∑ x in Icc 2 12, (x : ℝ) * (if probability_condition x then 1 else 0) / (12 - 2) = (sqrt p + sqrt q - r) / s)
  (h2 : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) 
  : p + q + r + s = 282 := 
sorry

end find_p_q_r_s_l263_263549


namespace largest_of_five_consecutive_divisible_by_three_l263_263410

theorem largest_of_five_consecutive_divisible_by_three (a b c d e : ℤ)
  (h1: 71 ≤ a ∧ a ≤ 99)
  (h2: a + 3 = b)
  (h3: a + 6 = c)
  (h4: a + 9 = d)
  (h5: a + 12 = e)
  (h6: ∀ n, n ∈ {a, b, c, d, e} → (n % 3 = 0)) :
  e = 84 :=
by
  sorry

end largest_of_five_consecutive_divisible_by_three_l263_263410


namespace coefficient_x2_binomial_largest_coefficient_binomial_l263_263809

theorem coefficient_x2_binomial (n : ℕ) (h : (Nat.choose n 3) = (Nat.choose n 7)) : 
  (n = 10) → 
  ((coeff (λ x => (sqrt x + 1 / (2 * (x^(1/4))))^n) 2) = 105/8) :=
by
  intros h₁
  rw [←h₁]
  sorry

theorem largest_coefficient_binomial (n : ℕ)
 (h₁ : (2 * (Nat.choose n 1 * (1 / 2))) = ((Nat.choose n 0) + (Nat.choose n 2 * (1 / 2)^2)) )
 (h₂ : n = 8) : 
 ∃ k, k ∈ finset.range (n + 1) ∧ 
   (C n 3 * (1 / 2)^(3)) = 7 * x^(5/2) ∧ 
   (C n 4 * (1 / 2)^(4)) = 7 * x^(7/4) :=
by
  intros
  use [3, 4]
  sorry

end coefficient_x2_binomial_largest_coefficient_binomial_l263_263809


namespace alloy_gold_percentage_l263_263339

theorem alloy_gold_percentage :
  ∀ (m1 m2 w1 w2 total_weight : ℝ),
    m1 = 0.60 →
    m2 = 0.40 →
    w1 = 6.2 →
    w2 = 6.2 →
    total_weight = 12.4 →
    ((m1 * w1 + m2 * w2) / total_weight) * 100 = 50 :=
by
  intros m1 m2 w1 w2 total_weight
  assume h1 h2 h3 h4 h5
  sorry

end alloy_gold_percentage_l263_263339


namespace probability_m_n_units_digit_1_l263_263341
open Set

def m_set : Set ℕ := {23, 27, 31, 35, 39}
def n_set : Set ℕ := {n | 2000 ≤ n ∧ n ≤ 2019}

def units_digit (x : ℕ) : ℕ := x % 10

theorem probability_m_n_units_digit_1 :
  (∑ m in m_set, ∑ n in n_set, if units_digit (m ^ n) = 1 then 1 else 0) /
  (|m_set| * |n_set|) = 3 / 10 :=
  sorry

end probability_m_n_units_digit_1_l263_263341


namespace train_crossing_time_l263_263475

theorem train_crossing_time :
  ∀ (train_length bridge_length : ℕ) (train_speed_kmph : ℝ) (conversion_factor : ℝ),
  train_length = 250 →
  bridge_length = 350 →
  train_speed_kmph = 50 →
  conversion_factor = 1000 / 3600 →
  let total_distance := (train_length + bridge_length : ℕ) in
  let train_speed_mps := train_speed_kmph * conversion_factor in
  let time_to_cross := total_distance / train_speed_mps in
  time_to_cross ≈ 43.20 := 
by
  intros train_length bridge_length train_speed_kmph conversion_factor
         train_length_def bridge_length_def train_speed_kmph_def conversion_factor_def
  let total_distance := (train_length + bridge_length : ℕ)
  let train_speed_mps := train_speed_kmph * conversion_factor
  let time_to_cross := total_distance / train_speed_mps
  sorry

end train_crossing_time_l263_263475


namespace elsa_cookie_time_l263_263044

variables (baking_time white_icing_time chocolate_icing_time total_time : ℕ)

def time_for_dough_and_cooling (baking_time white_icing_time chocolate_icing_time total_time : ℕ) : ℕ :=
  total_time - baking_time - white_icing_time - chocolate_icing_time

theorem elsa_cookie_time :
  baking_time = 15 →
  white_icing_time = 30 →
  chocolate_icing_time = 30 →
  total_time = 120 →
  time_for_dough_and_cooling baking_time white_icing_time chocolate_icing_time total_time = 45 :=
by {
  intros,
  rw [time_for_dough_and_cooling],
  simp,
  omega,
  sorry
}

end elsa_cookie_time_l263_263044


namespace bleachers_runs_l263_263980

theorem bleachers_runs (T : ℕ) (stairs_per_trip : ℕ) (calories_per_stair : ℕ) (calories_total : ℕ)
  (h1 : stairs_per_trip = 32)
  (h2 : calories_per_stair = 2)
  (h3 : calories_total = 5120) :
  T = calories_total / (stairs_per_trip * calories_per_stair * 2) :=
by
  rw [h1, h2, h3]
  have : stairs_per_trip * calories_per_stair * 2 = 128 := by norm_num
  rw this
  norm_num
  sorry

end bleachers_runs_l263_263980


namespace sum_of_roots_l263_263369

noncomputable def P (x : ℝ) : ℝ :=
  (x - 1)^2023 + 2*(x - 2)^2022 + 3*(x - 3)^2021 + ⋯ + 2022*(x - 2022)^2 + 2023*(x - 2023)

theorem sum_of_roots : 
  -- Let S be the sum of the 2023 roots of P(x)
  let S := (roots P).sum in
  S = 2021 := sorry

end sum_of_roots_l263_263369


namespace part_1_part_2_l263_263571

-- Condition definitions
def sequence_a : ℕ → ℕ
| 1 := 2
| (n+1) := 3 * sequence_a n + 2

def sequence_b (n : ℕ) : ℕ := log 3 (sequence_a n + 1)

-- Questions and Correct Ansers as Statements
theorem part_1 (n : ℕ) (h1 : n ≥ 2) : 
  ∃ r, sequence_a (n + 1) + 1 = r * (sequence_a n +1)
  :=
begin
  sorry
end

theorem part_2 (n : ℕ) (h2 : n ≥ 1) :
  ∑ i in finset.range n, (1 / sequence_b i * sequence_b (i + 1)) = n / (n + 1)
  :=
begin
  sorry
end

end part_1_part_2_l263_263571


namespace find_k_l263_263486

theorem find_k (k : ℝ) : (∃ x : ℝ, x^2 - 2*k*x + k^2 = 0) → (∃ k : ℝ, k = -1) :=
begin
  sorry
end

end find_k_l263_263486


namespace volume_of_sphere_l263_263703

-- Defining basic elements
variable (r : ℝ)  -- radius of the sphere
variable (d : ℝ)  -- distance from center to the plane
variable (C : ℝ)  -- radius of the circular section

-- Given conditions
axiom h1 : 2 * sqrt 5 / 2 = C
axiom h2 : d = 2

-- Question to prove: Volume of the sphere
theorem volume_of_sphere (C d r : ℝ) (h1 : 2 * sqrt 5 / 2 = C) (h2 : d = 2) (h3 : sqrt (C^2 + d^2) = r) :
  (4 / 3) * real.pi * r^3 = 36 * real.pi :=
sorry

end volume_of_sphere_l263_263703


namespace solve_for_x_l263_263866

theorem solve_for_x (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end solve_for_x_l263_263866


namespace surface_area_ratio_l263_263663

-- Definitions based on conditions
def side_length (s : ℝ) := s > 0
def A_cube (s : ℝ) := 6 * s ^ 2
def A_rect (s : ℝ) := 2 * (2 * s) * (3 * s) + 2 * (2 * s) * (4 * s) + 2 * (3 * s) * (4 * s)

-- Theorem statement proving the ratio
theorem surface_area_ratio (s : ℝ) (h : side_length s) : A_cube s / A_rect s = 3 / 26 :=
by
  sorry

end surface_area_ratio_l263_263663


namespace possible_values_product_xy_l263_263943

-- Define the points and the conditions for the congruent triangles
noncomputable def Point : Type := (ℝ × ℝ)

def congruent_triangles (A B C D E : Point) : Prop :=
  let dist (p1 p2 : Point) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist A B = dist A D ∧ 
  dist A C = dist A E ∧
  dist B C = dist D E

-- Define the property for the product of coordinates
def possible_product_of_coordinates (E : Point) : ℝ := E.1 * E.2

-- Main theorem to state the possible values of the product xy
theorem possible_values_product_xy (A B C D E : Point) 
  (h_congruent : congruent_triangles A B C D E) : 
  possible_product_of_coordinates E = 14 ∨ 
  possible_product_of_coordinates E = 18 ∨ 
  possible_product_of_coordinates E = 40 :=
sorry  -- Proof omitted

end possible_values_product_xy_l263_263943


namespace find_p_minus_q_l263_263842

theorem find_p_minus_q (x y p q : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 3 / (x * p) = 8) (h2 : 5 / (y * q) = 18)
  (hminX : ∀ x', x' ≠ 0 → 3 / (x' * 3) ≠ 1 / 8)
  (hminY : ∀ y', y' ≠ 0 → 5 / (y' * 5) ≠ 1 / 18) :
  p - q = 0 :=
sorry

end find_p_minus_q_l263_263842


namespace measure_of_central_angle_l263_263872

open_locale big_operators

theorem measure_of_central_angle (r L : ℝ) (θ : ℝ) 
  (h1 : L = r) 
  (h2 : L = r * θ) :
  θ = 1 :=
by {
  sorry
}

end measure_of_central_angle_l263_263872


namespace remainder_of_exponentiation_is_correct_l263_263788

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end remainder_of_exponentiation_is_correct_l263_263788


namespace non_congruent_triangles_with_perimeter_11_l263_263123

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263123


namespace quadratic_has_distinct_real_roots_find_k_l263_263090

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * k - 1
  let c := -k - 2
  let Δ := b^2 - 4 * a * c
  (Δ > 0) :=
by
  sorry

-- Part 2: Given the roots condition, find k
theorem find_k (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -(2 * k - 1))
  (h2 : x1 * x2 = -k - 2)
  (h3 : x1 + x2 - 4 * x1 * x2 = 1) : 
  k = -4 :=
by
  sorry

end quadratic_has_distinct_real_roots_find_k_l263_263090


namespace cd_total_l263_263539

theorem cd_total :
  ∀ (Kristine Dawn Mark Alice : ℕ),
  Dawn = 10 →
  Kristine = Dawn + 7 →
  Mark = 2 * Kristine →
  Alice = (Kristine + Mark) - 5 →
  (Dawn + Kristine + Mark + Alice) = 107 :=
by
  intros Kristine Dawn Mark Alice hDawn hKristine hMark hAlice
  rw [hDawn, hKristine, hMark, hAlice]
  sorry

end cd_total_l263_263539


namespace total_golf_balls_purchased_l263_263380

theorem total_golf_balls_purchased :
  let dozens_dan := 5
  let dozens_gus := 3
  let dozens_chris := 4 + 6 / 12
  let dozens_emily := 2
  let dozens_fred := 1
  let total_dozens := dozens_dan + dozens_gus + dozens_chris + dozens_emily + dozens_fred
  let golf_balls_per_dozen := 12
  let total_golf_balls := total_dozens * golf_balls_per_dozen
  total_golf_balls = 186 :=
by
  let dozens_dan := 5
  let dozens_gus := 3
  let dozens_chris := 4 + 6 / 12
  let dozens_emily := 2
  let dozens_fred := 1
  let total_dozens := dozens_dan + dozens_gus + dozens_chris + dozens_emily + dozens_fred
  let golf_balls_per_dozen := 12
  let total_golf_balls := total_dozens * golf_balls_per_dozen
  show total_golf_balls = 186 from sorry

end total_golf_balls_purchased_l263_263380


namespace sandy_comic_books_l263_263950

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l263_263950


namespace find_sale_in_fourth_month_l263_263336

noncomputable def sale_in_fourth_month (s1 s2 s3 s4 s5 s6 : ℝ) : ℝ :=
  let average := 6800
  let total_needed := average * 6
  total_needed - (s1 + s2 + s3 + s5 + s6)

theorem find_sale_in_fourth_month :
  let s1 := 6435
  let s2 := 6927
  let s3 := 6855
  let s5 := 6562
  let s6 := 6791
  sale_in_fourth_month s1 s2 s3 s4 s5 s6 = 7230 :=
by
  let s1 := 6435
  let s2 := 6927
  let s3 := 6855
  let s4 := 7230
  let s5 := 6562
  let s6 := 6791
  calc 
    sale_in_fourth_month s1 s2 s3 s4 s5 s6 
       = 40800 - (s1 + s2 + s3 + s5 + s6)
       = 40800 - (6435 + 6927 + 6855 + 6562 + 6791)
       = 40800 - 33570
       = 7230
  sorry

end find_sale_in_fourth_month_l263_263336


namespace non_congruent_triangles_with_perimeter_11_l263_263138

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l263_263138


namespace range_of_a_for_increasing_function_l263_263489

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x : ℝ, 1 + a * sin x ≥ 0) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  have h_min := h (-1)
  have h_max := h (1)
  sorry

end range_of_a_for_increasing_function_l263_263489


namespace tangent_line_eq_l263_263036

section TangentLine
variable {x y : ℝ}

-- Define the function f(x) = sqrt(2x - 4)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 4)

-- Define the point (4, f(4))
def point_of_tangency : ℝ × ℝ := (4, f 4)

-- Define the tangent line equation
noncomputable def tangent_line (x : ℝ) : ℝ := (1 / 2) * x - 1

-- The statement for the proof problem
theorem tangent_line_eq (x : ℝ) (h : point_of_tangency = (4, f 4)) : 
    tangent_line x = (1 / 2) * x - 1 :=
sorry

end TangentLine

end tangent_line_eq_l263_263036


namespace parity_difference_l263_263225

noncomputable def sum_of_simplified_numerators (n : ℕ) : ℕ := 
  (List.range n).map (λ k, if Nat.gcd k n = 1 then k else 0).sum

theorem parity_difference (n : ℕ) (h : n > 1) : 
  (sum_of_simplified_numerators n) % 2 ≠ (sum_of_simplified_numerators (2015 * n)) % 2 :=
sorry

end parity_difference_l263_263225


namespace min_nodes_hex_grid_l263_263716

-- Define what it means to be a node in the hexagonal grid
structure HexagonalGridNode where
  x y : Int

-- Define midpoint function for nodes
def midpoint (p1 p2 : HexagonalGridNode) : HexagonalGridNode :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

-- Define a function to check if a midpoint is also a node
def is_node (p : HexagonalGridNode) : Prop :=
  Int.even p.x ∧ Int.even p.y

-- The main theorem to be proved
theorem min_nodes_hex_grid {nodes : List HexagonalGridNode} :
  (∀ (p1 p2 : HexagonalGridNode), p1 ∈ nodes → p2 ∈ nodes → p1 ≠ p2 → is_node (midpoint p1 p2)) →
  nodes.length >= 9 :=
sorry

end min_nodes_hex_grid_l263_263716


namespace combustion_problem_l263_263297

noncomputable def thermochemical_eq1 : String := 
  "C6H5NO2 (liquid) + 6.25 O2 (gas) = 6 CO2 (gas) + 0.5 N2 (gas) + 2.5 H2O (liquid) + 3094.88 kJ"

noncomputable def thermochemical_eq2 : String := 
  "C6H5NH2 (liquid) + 7.75 O2 (gas) = 6 CO2 (gas) + 0.5 N2 (gas) + 3.5 H2O (liquid) + 3392.15 kJ"

noncomputable def thermochemical_eq3 : String := 
  "C2H5OH (liquid) + 3 O2 (gas) = 2 CO2 (gas) + 3 H2O (liquid) + 1370 kJ"

def mass_nitrobenzene (x : ℝ) : ℝ := 123 * x
def mass_aniline (y : ℝ) : ℝ := 93 * y

def mass_solution (x y : ℝ) : ℝ := 470 * x

def amount_ethanol (x y : ℝ) : ℝ := 7.54 * x - 2.02 * y

def enthalpy_eq (x y : ℝ) : ℝ := 
  13428.68 * x + 624.75 * (0.3 - x)

def nitrogen_eq (x y : ℝ) : ℝ := 
  0.5 * x + 0.5 * y

theorem combustion_problem (x y : ℝ) (h₁ : enthalpy_eq x y = 1467.4) (h₂ : nitrogen_eq x y = 0.15) :
  x ≈ 0.1 ∧ mass_solution x y = 47 :=
by
  sorry

end combustion_problem_l263_263297


namespace count_integers_abs_leq_4_l263_263852

theorem count_integers_abs_leq_4 : 
  let solution_set := {x : Int | |x - 3| ≤ 4}
  ∃ n : Nat, n = 9 ∧ (∀ x ∈ solution_set, x ∈ finset.range 9) := sorry

end count_integers_abs_leq_4_l263_263852


namespace ellipse_equation_and_lambda_range_l263_263010

-- Definitions based on conditions
variables {a b : ℝ} (x y : ℝ)
def ellipse (a b : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

variables (P Q : ℝ × ℝ) (l1 l2 : ℝ)
def lines_through_B (l1 l2 P Q : ℝ × ℝ) : Prop := 
  (l1 = 2) ∧ (P = (-5/3, -4/3))

-- Theorem statement for the proof problem
theorem ellipse_equation_and_lambda_range (x y : ℝ) (a b : ℝ) (P Q : ℝ × ℝ) (M : ℝ × ℝ) (l1 l2 : ℝ) :
  ellipse a b →
  lines_through_B l1 l2 P Q →
  (∃ (a b : ℝ), (a^2 = 5) ∧ (b^2 = 4) ∧ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (∃ (λ : ℝ), (4/5 < λ) ∧ (λ < 5/4)) :=
by
  sorry

end ellipse_equation_and_lambda_range_l263_263010


namespace intersection_eq_l263_263805

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def N : Set (ℝ × ℝ) := { (x, y) | (x^2 / 2) + y^2 = 1 }

theorem intersection_eq : {y | ∃ x, (M x ∧ N (x, y))} = [0, real.sqrt 2] :=
by
  sorry

end intersection_eq_l263_263805


namespace number_of_solutions_in_positive_integers_l263_263629

theorem number_of_solutions_in_positive_integers (x y : ℕ) (h1 : 3 * x + 4 * y = 806) : 
  ∃ n : ℕ, n = 67 := 
sorry

end number_of_solutions_in_positive_integers_l263_263629


namespace prob_equiv_l263_263425

noncomputable def a_n (n : ℕ) : ℕ := 3^n -- Since in the solution a_n = 3^n
def S (n : ℕ) : ℕ := (∑ i in Finset.range n, a_n i) -- Sum of first n terms of a_n
def b_n (n : ℕ) : ℕ := 2 * n + 1 -- Since in the solution b_n = log_3{3^(2n+1)}
def T (n : ℕ) : ℕ := ∑ i in Finset.range n, b_n i -- Sum of first n terms of b_n
def reciprocals_sum : ℕ → ℝ := λ n, (∑ i in Finset.range n, 1 / (T (i + 1) : ℝ))

theorem prob_equiv (n : ℕ) : reciprocals_sum n = (1 / 2 : ℝ) * (3 / 2 - 1 / (n + 1) - 1 / (n + 2)) := 
by 
  sorry -- The proof is omitted as per the instructions.

end prob_equiv_l263_263425


namespace value_of_a_plus_b_l263_263054

theorem value_of_a_plus_b (a b : ℝ) : (|a - 1| + (b + 3)^2 = 0) → (a + b = -2) :=
by
  sorry

end value_of_a_plus_b_l263_263054


namespace min_value_d1_d2_l263_263449

noncomputable def min_distance_sum : ℝ :=
  let d1 (u : ℝ) : ℝ := (1 / 5) * abs (3 * Real.cos u - 4 * Real.sin u - 10)
  let d2 (u : ℝ) : ℝ := 3 - Real.cos u
  let d_sum (u : ℝ) : ℝ := d1 u + d2 u
  ((5 - (4 * Real.sqrt 5 / 5)))

theorem min_value_d1_d2 :
  ∀ (P : ℝ × ℝ) (u : ℝ),
    P = (Real.cos u, Real.sin u) →
    (P.1 ^ 2 + P.2 ^ 2 = 1) →
    let d1 := (1 / 5) * abs (3 * P.1 - 4 * P.2 - 10)
    let d2 := 3 - P.1
    d1 + d2 ≥ (5 - (4 * Real.sqrt 5 / 5)) :=
by
  sorry

end min_value_d1_d2_l263_263449


namespace find_integer_with_properties_l263_263203

def is_sum_of_n_consecutive_integers (N k : ℕ) : Prop :=
  ∃ m : ℕ, N = k * m + (k * (k - 1)) / 2

def ways_to_write_as_consecutive_sums (N : ℕ) : ℕ :=
  (List.filter (λ k, is_sum_of_n_consecutive_integers N k) (List.range (N + 1))).length

theorem find_integer_with_properties :
  ∃ (N : ℕ),
    (is_sum_of_n_consecutive_integers N 1990) ∧
    (ways_to_write_as_consecutive_sums N = 1990) ∧
    (N = 5^10 * 199^180 / 2 ∨ N = 5^180 * 199^10 / 2) :=
sorry

end find_integer_with_properties_l263_263203


namespace repeating_decimal_fraction_denominator_minus_numerator_l263_263559

theorem repeating_decimal_fraction_denominator_minus_numerator
  (F : ℚ) (h : F = 925 / 999) :
  let reduced := F.num.gcd F.denom in
  F.num / reduced = 25 ∧ F.denom / reduced = 27 →
  (F.denom / reduced) - (F.num / reduced) = 2 := 
by
  intro reduced h1 h2
  sorry

end repeating_decimal_fraction_denominator_minus_numerator_l263_263559


namespace problem_l263_263463

noncomputable def f (x : ℝ) : ℝ := |x - 1|

def A : set ℝ := {x | -1 < x ∧ x < 1}

theorem problem (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) : f(a * b) > f(a) - f(b) := by
  sorry

end problem_l263_263463


namespace log_lt_zero_implies_x_lt_one_and_gt_zero_l263_263149

variable (x : ℝ)
variable (h1 : ∃ a : ℝ, log 10 x = a ∧ a < 0)

theorem log_lt_zero_implies_x_lt_one_and_gt_zero (h1 : ∃ a : ℝ, log 10 x = a ∧ a < 0) : 
  0 < x ∧ x < 1 :=
sorry

end log_lt_zero_implies_x_lt_one_and_gt_zero_l263_263149


namespace striped_nails_painted_l263_263208

theorem striped_nails_painted (total_nails purple_nails blue_nails : ℕ) (h_total : total_nails = 20)
    (h_purple : purple_nails = 6) (h_blue : blue_nails = 8)
    (h_diff_percent : |(blue_nails:ℚ) / total_nails * 100 - 
    ((total_nails - purple_nails - blue_nails):ℚ) / total_nails * 100| = 10) :
    (total_nails - purple_nails - blue_nails) = 6 := 
by 
  sorry

end striped_nails_painted_l263_263208


namespace barge_arrives_at_B_at_2pm_l263_263723

noncomputable def barge_arrival_time
  (constant_barge_speed : ℝ)
  (river_current_speed : ℝ)
  (distance_AB : ℝ)
  (time_depart_A : ℕ)
  (wait_time_B : ℝ)
  (time_return_A : ℝ) :
  ℝ := by
  sorry

theorem barge_arrives_at_B_at_2pm :
  ∀ (constant_barge_speed : ℝ), 
    (river_current_speed = 3) →
    (distance_AB = 60) →
    (time_depart_A = 9) →
    (wait_time_B = 2) →
    (time_return_A = 19 + 20 / 60) →
    barge_arrival_time constant_barge_speed river_current_speed distance_AB time_depart_A wait_time_B time_return_A = 14 := by
  sorry

end barge_arrives_at_B_at_2pm_l263_263723


namespace prob_A_and_B_is_37_over_900_l263_263566

-- Define the range of three-digit numbers
def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Define events A and B
def A (n : ℕ) : Prop := n % 3 = 0
def B (n : ℕ) : Prop := n % 8 = 0

-- Define the event A ∩ B
def A_and_B (n : ℕ) : Prop := A n ∧ B n

-- Define the probability calculation
noncomputable def probability_A_and_B : ℚ := 
  (finset.card (finset.filter A_and_B (finset.filter (λ n, n ∈ three_digit_numbers) (finset.range 1000)))) / 
  (finset.card (finset.filter (λ n, n ∈ three_digit_numbers) (finset.range 1000)))

-- The statement of the problem
theorem prob_A_and_B_is_37_over_900 : probability_A_and_B = 37 / 900 :=
  sorry

end prob_A_and_B_is_37_over_900_l263_263566


namespace eccentricity_difference_l263_263454

variables {a b m n : ℝ}

/-- 
Given:
- An ellipse C1: x²/a² + y²/b² = 1 with a > b > 0
- A hyperbola C2: x²/m² - y²/n² = 1 with m > 0, n > 0
- Both share the same foci F1 and F2 with F1 being the left focus
- Eccentricities e1 and e2 of curves C1 and C2 respectively
- Triangle P F1 F2 is isosceles with PF1 as the base
Prove: e2 - e1 = √2
-/
theorem eccentricity_difference
  (h_a : 0 < a) (h_b : 0 < b) (h_m : 0 < m) (h_n : 0 < n)
  (h_ab : b < a) 
  (h_intersect : ∃ (P : ℝ × ℝ), (P.1^2/a^2 + P.2^2/b^2 = 1) ∧ (P.1^2/m^2 - P.2^2/n^2 = 1)) :
  let e1 := (sqrt (a^2 - b^2)) / a,
      e2 := (sqrt (m^2 + n^2)) / m in
  (e2 - e1 = sqrt 2) :=
by
  sorry

end eccentricity_difference_l263_263454


namespace fraction_meaningful_l263_263488

theorem fraction_meaningful (x : ℝ) : (x ≠ 5) ↔ (x-5 ≠ 0) :=
by simp [sub_eq_zero]

end fraction_meaningful_l263_263488


namespace non_congruent_triangles_with_perimeter_11_l263_263124

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l263_263124


namespace expected_value_of_product_l263_263942

-- Probability definitions for the faces of the cube
structure CubeFace where
  a0 a1 a2 : ℚ
  h : a0 + a1 + a2 = 1

def fair_cube : CubeFace :=
{ a0 := 1 / 2, a1 := 1 / 3, a2 := 1 / 6,
  h := by norm_num }

-- Definition for the expected value calculation
def expected_value (cube : CubeFace) : ℚ :=
  let p0 := cube.a0
  let p1 := cube.a1
  let p2 := cube.a2
  0 * (p0 * p0 + p0 * p1 + p0 * p2 + p1 * p0 + p2 * p0) +
      1 * (p1 * p1) +
      2 * (p1 * p2 + p2 * p1) +
      4 * (p2 * p2)

-- The theorem to prove
theorem expected_value_of_product : expected_value fair_cube = 4 / 9 := by
  sorry

end expected_value_of_product_l263_263942


namespace proposition_B_l263_263918

variables {m n : Type} [linear_ordered_semiring m] [linear_ordered_semiring n]

-- Definitions for two different straight lines and planes
variables (m n : set α) (α β : set β)

-- Given conditions
variables (h1 : m ≠ n) (h2 : α ≠ β)
variables (h3 : m ⊆ α) (h4 : n ⊆ β)
variables (h5 : is_perpendicular m α) (h6 : is_parallel m n)
variables (h7 : is_parallel n β)

-- Prove statement
theorem proposition_B : is_perpendicular α β :=
  sorry

end proposition_B_l263_263918


namespace eq1_solutions_eq2_solutions_l263_263958

theorem eq1_solutions (x : ℝ) : x ^ 2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

theorem eq2_solutions (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + sqrt 5 ∨ x = 2 - sqrt 5 :=
by sorry

end eq1_solutions_eq2_solutions_l263_263958


namespace chromium_percentage_new_alloy_l263_263888

-- Define the weights and chromium percentages of the alloys
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 35
def chromium_percent_alloy1 : ℝ := 0.15
def chromium_percent_alloy2 : ℝ := 0.08

-- Define the theorem to calculate the chromium percentage of the new alloy
theorem chromium_percentage_new_alloy :
  ((weight_alloy1 * chromium_percent_alloy1 + weight_alloy2 * chromium_percent_alloy2)
  / (weight_alloy1 + weight_alloy2) * 100) = 10.1 :=
by
  sorry

end chromium_percentage_new_alloy_l263_263888


namespace contrapositive_proposition_l263_263972

theorem contrapositive_proposition (x a b : ℝ) : (x < 2 * a * b) → (x < a^2 + b^2) :=
sorry

end contrapositive_proposition_l263_263972


namespace find_u5_l263_263606

theorem find_u5 
  (u : ℕ → ℝ)
  (h_rec : ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n)
  (h_u3 : u 3 = 9)
  (h_u6 : u 6 = 243) : 
  u 5 = 69 :=
sorry

end find_u5_l263_263606


namespace milk_leftover_l263_263365

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end milk_leftover_l263_263365


namespace mrs_smith_strawberries_l263_263238

theorem mrs_smith_strawberries (girls : ℕ) (strawberries_per_girl : ℕ) 
                                (h1 : girls = 8) (h2 : strawberries_per_girl = 6) :
    girls * strawberries_per_girl = 48 := by
  sorry

end mrs_smith_strawberries_l263_263238


namespace determine_beta_l263_263803

theorem determine_beta (
  (β α : ℝ)
  (h1 : 0 < β)
  (h2 : β < α)
  (h3 : α < π / 2)
  (h4 : ∃ (P : ℝ × ℝ), P = (1, 4 * sqrt 3) ∧ 
                        P.1 = cos α * 7 ∧ P.2 = sin α * 7)
  (h5 : sin α * sin (π / 2 - β) + cos α * cos (π / 2 + β) = 3 * sqrt 3 / 14)
  ) : β = π / 3 :=
sorry

end determine_beta_l263_263803


namespace bottles_difference_l263_263397

noncomputable def Donald_drinks_bottles (P: ℕ): ℕ := 2 * P + 3
noncomputable def Paul_drinks_bottles: ℕ := 3
noncomputable def actual_Donald_bottles: ℕ := 9

theorem bottles_difference:
  actual_Donald_bottles - 2 * Paul_drinks_bottles = 3 :=
by 
  sorry

end bottles_difference_l263_263397


namespace thirty_seventh_digit_one_seventh_l263_263660

theorem thirty_seventh_digit_one_seventh : 
  let dec_repr := "142857"
  let digit_at (n : ℕ) (s : String) : Char := s.get ⟨n % s.length, sorry⟩
  digit_at 37 dec_repr = '1' :=
by
  sorry

end thirty_seventh_digit_one_seventh_l263_263660


namespace b_parallel_to_a_l263_263616

-- Define vectors $\overrightarrow{a}$ and $\overrightarrow{b}$
def a : ℝ × ℝ × ℝ := (1, 3, -2)
def b : ℝ × ℝ × ℝ := (-1/2, -3/2, 1)

-- Define what it means for two vectors to be parallel
def are_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, v1 = (k * v2.1, k * v2.2, k * v2.3)

-- State the theorem to be proven
theorem b_parallel_to_a : are_parallel b a := sorry

end b_parallel_to_a_l263_263616


namespace length_of_rectangle_from_conditions_l263_263993

-- Definitions as per the conditions
def side_of_square (P_rectangle : ℝ) (breadth : ℝ) : ℝ :=
  P_rectangle / 2 + breadth

def circumference_of_semicircle (side : ℝ) : ℝ :=
  (1 / 2) * 3.14 * side + side

noncomputable def length_of_rectangle (side : ℝ) : ℝ :=
  (2 * side - 12) / 2

-- Given conditions in Lean 4
theorem length_of_rectangle_from_conditions (h1 : ∀ (P_rectangle P_square : ℝ) (breadth : ℝ) (h1 : P_rectangle = 2 * P_square + 12),
    P_rectangle = 4 * side_of_square P_rectangle breadth)
 (h2 : circumference_of_semicircle (side_of_square (P_square - 2 * 6) 6) = 11.78) :
  length_of_rectangle (side_of_square (11.78 / 1.57) 6) = 3.16 :=
by
  sorry

end length_of_rectangle_from_conditions_l263_263993


namespace probability_at_most_2_heads_l263_263301

theorem probability_at_most_2_heads : 
  (let p_at_most_2_heads := 1 - (1 / 2) ^ 3 in p_at_most_2_heads = 7 / 8) := 
by
  let p_exactly_3_heads := (1 / 2) ^ 3
  have p_at_most_2_heads := 1 - p_exactly_3_heads
  show p_at_most_2_heads = 7 / 8
  sorry

end probability_at_most_2_heads_l263_263301


namespace simplify_sum_powers_of_i_l263_263250

open Complex
open Finset

noncomputable def sum_powers_of_i : ℂ :=
∑ i in range (2014), (I ^ i)

theorem simplify_sum_powers_of_i :
  sum_powers_of_i = 1 + I :=
by
  -- Proof here
  sorry

end simplify_sum_powers_of_i_l263_263250


namespace oe_perpendicular_to_cd_iff_ab_eq_ac_l263_263713

variables {A B C O D E : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] [Inhabited D] [Inhabited E]

-- Typeclass for triangle and points properties (circumcenter, midpoint, centroid)
class is_triangle (A B C : Type) := (triangle_prop : True)
class is_circumcenter (O : Type) (T : Type) := (circumcenter_prop : True)
class is_midpoint (D : Type) (A B : Type) := (midpoint_prop : True)
class is_centroid (E : Type) (A C D : Type) := (centroid_prop : True)

-- Introduce an arbitrary triangle ABC, with circumcenter O,
-- midpoint D of AB, and centroid E of triangle ACD.
variables (T : Type) [is_triangle A B C]
variables [is_circumcenter O T]
variables [is_midpoint D A B]
variables [is_centroid E A C D]

-- The theorem statement
theorem oe_perpendicular_to_cd_iff_ab_eq_ac : 
  (⊥(O E, C D)) ↔ (A B = A C) := 
sorry

end oe_perpendicular_to_cd_iff_ab_eq_ac_l263_263713


namespace line_intercepts_chord_inf_perpendicular_lines_l263_263893

-- Define the given circles
def C1 : Set (ℝ × ℝ) := {p | (p.1 + 3)^2 + (p.2 - 1)^2 = 4}
def C2 : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 - 5)^2 = 4}

-- Equation of line (1)
theorem line_intercepts_chord (A : ℝ × ℝ) (l : ℝ → ℝ) :
  A = (4, 0) →
  (∃ k : ℝ, (∀ x : ℝ, l(x) = k * (x - 4) + 0) ∨ l 0 = 0 ∨ (7 * x + 24 * l x - 28 = 0)) :=
sorry

-- Coordinates of points P (2)
theorem inf_perpendicular_lines (P : ℝ × ℝ) :
  (∃ P : ℝ × ℝ,
  (∃ k : ℝ, 
    (∀ x : ℝ, 
      (|1 + 3*k + k*P.1 - P.2| = |5*k + 4 - P.1 - k * P.2| ∧
      (P = (5/2, -1/2) ∨ P = (-3/2, 13/2))))) :=
sorry

end line_intercepts_chord_inf_perpendicular_lines_l263_263893


namespace g_neg3_g_3_l263_263927

def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 1
  else 4 * x - 2

theorem g_neg3 : g (-3) = -8 := by
  sorry

theorem g_3 : g (3) = 10 := by
  sorry

end g_neg3_g_3_l263_263927


namespace expression_divisible_by_25_l263_263590

theorem expression_divisible_by_25 (n : ℕ) : 
    (2^(n+2) * 3^n + 5 * n - 4) % 25 = 0 :=
by {
  sorry
}

end expression_divisible_by_25_l263_263590


namespace non_congruent_triangles_with_perimeter_11_l263_263110

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l263_263110


namespace distance_between_points_l263_263035

def point1 : ℝ × ℝ × ℝ := (3, 3, 3)
def point2 : ℝ × ℝ × ℝ := (0, 0, 0)

theorem distance_between_points :
  let dist := (λ (p1 p2 : ℝ × ℝ × ℝ), 
                (Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)))
  in dist point1 point2 = 3 * Real.sqrt 3 := by
  sorry

end distance_between_points_l263_263035


namespace dog_paws_ground_l263_263755

theorem dog_paws_ground (total_dogs : ℕ) (two_thirds_back_legs : ℕ) (remaining_dogs_four_legs : ℕ) (two_paws_per_back_leg_dog : ℕ) (four_paws_per_four_leg_dog : ℕ) :
  total_dogs = 24 →
  two_thirds_back_legs = 2 * total_dogs / 3 →
  remaining_dogs_four_legs = total_dogs - two_thirds_back_legs →
  two_paws_per_back_leg_dog = 2 →
  four_paws_per_four_leg_dog = 4 →
  (two_thirds_back_legs * two_paws_per_back_leg_dog + remaining_dogs_four_legs * four_paws_per_four_leg_dog) = 64 := 
by 
  sorry

end dog_paws_ground_l263_263755


namespace length_of_bridge_l263_263319

theorem length_of_bridge (L_train : ℕ) (v_km_hr : ℕ) (t : ℕ) 
  (h_L_train : L_train = 150)
  (h_v_km_hr : v_km_hr = 45)
  (h_t : t = 30) : 
  ∃ L_bridge : ℕ, L_bridge = 225 :=
by 
  sorry

end length_of_bridge_l263_263319


namespace tree_height_after_4_months_l263_263236

noncomputable def tree_growth_rate := 50 -- growth in centimeters per two weeks
noncomputable def current_height_meters := 2 -- current height in meters
noncomputable def weeks_in_a_month := 4

def current_height_cm := current_height_meters * 100
def months := 4
def total_weeks := months * weeks_in_a_month
def growth_periods := total_weeks / 2
def total_growth := growth_periods * tree_growth_rate
def final_height := total_growth + current_height_cm

theorem tree_height_after_4_months :
  final_height = 600 :=
  by
    sorry

end tree_height_after_4_months_l263_263236


namespace probability_of_divisible_by_11_five_digit_palindrome_l263_263695

def five_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 10001 * a + 1010 * b + 100 * c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_of_divisible_by_11_five_digit_palindrome :
  let total_palindromes := 9 * 10 * 10 in
  let valid_palindromes := (finset.range 10).sum (λ c, (finset.range 10).sum (λ b, (finset.range 9).filter
    (λ a, divisible_by_11 (10001 * (a + 1) + 1010 * b + 100 * c)).card)) in
  (valid_palindromes : ℚ) / total_palindromes = 1 / 20 :=
sorry

end probability_of_divisible_by_11_five_digit_palindrome_l263_263695


namespace find_B_and_distance_l263_263546

noncomputable def pointA : ℝ × ℝ := (2, 4)

noncomputable def pointB : ℝ × ℝ := (-(1 + Real.sqrt 385) / 8, (-(1 + Real.sqrt 385) / 8) ^ 2)

noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem find_B_and_distance :
  (pointA.snd = pointA.fst ^ 2) ∧
  (pointB.snd = (-(1 + Real.sqrt 385) / 8) ^ 2) ∧
  (distanceToOrigin pointB = Real.sqrt ((-(1 + Real.sqrt 385) / 8) ^ 2 + (-(1 + Real.sqrt 385) / 8) ^ 4)) :=
  sorry

end find_B_and_distance_l263_263546


namespace smallest_integer_n_conditions_l263_263021

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def digits_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem smallest_integer_n_conditions (n : ℕ) (hn : n % 10 = 5) (hn_square : is_square n)
    (hn_sqrt_sum : digits_sum (Nat.sqrt n) = 9) : n = 2025 :=
sorry

end smallest_integer_n_conditions_l263_263021


namespace residue_of_neg_1235_mod_29_l263_263747

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l263_263747


namespace class_6_1_students_l263_263607

noncomputable def number_of_students : ℕ :=
  let n := 30
  n

theorem class_6_1_students (n : ℕ) (t : ℕ) (h1 : (n + 1) * t = 527) (h2 : n % 5 = 0) : n = 30 :=
  by
  sorry

end class_6_1_students_l263_263607


namespace count_k_values_for_lcm_l263_263801

theorem count_k_values_for_lcm : 
  let k_values := {k : ℕ | ∃ a b : ℕ, k = 2^a * 3^b ∧ 0 ≤ a ∧ a ≤ 24 ∧ b = 24} in
  36^12 = Nat.lcm (Nat.lcm (6^6) (8^8)) (Nat.lcm (9^9) k) → k_values.card = 25 := by
  sorry

end count_k_values_for_lcm_l263_263801


namespace function_increasing_l263_263627

-- Define the function
def f (x : ℝ) := (x - 3) * Real.exp x

-- Define the derivative of the function
def f' (x : ℝ) := (x - 2) * Real.exp x

-- State the problem
theorem function_increasing (x : ℝ) (h : x > 2) : 
  (f x) > f 2 := 
sorry

end function_increasing_l263_263627


namespace solution_set_f_gt_4_l263_263567

noncomputable def f (x: ℝ) : ℝ :=
  max (1 - x) (2 ^ x)

theorem solution_set_f_gt_4 :
  {x : ℝ | f x > 4} = set.Iio (-3) ∪ set.Ioi 2 :=
by
  sorry

end solution_set_f_gt_4_l263_263567


namespace necessary_condition_to_contain_circle_in_parabola_l263_263811

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem necessary_condition_to_contain_circle_in_parabola (a : ℝ) : 
  (∀ x y, N x y a → M x y) ↔ a ≥ 5 / 4 := 
sorry

end necessary_condition_to_contain_circle_in_parabola_l263_263811


namespace statement_a_incorrect_l263_263359

-- Definitions of conditions
def statement_a (A B : Point) : Prop :=
  line_segment A B = distance A B

def statement_b (A B C : Point) : Prop :=
  line_segment A B = line_segment A C → distance A B = distance A C

def statement_c (A B : Point) : Prop :=
  length (line_segment A B) = distance A B

def statement_d (A B : Point) : Prop :=
  distance A B = shortest_length (all_lines A B)

-- Main theorem: Proof that statement A is incorrect
theorem statement_a_incorrect (A B : Point) :
  ¬ statement_a A B :=
sorry

end statement_a_incorrect_l263_263359


namespace sum_odd_lt_sum_even_l263_263356

theorem sum_odd_lt_sum_even (n : ℕ) (h₁ : n % 2020 = 0) :
  let divs := {d : ℕ | d ∣ n ∧ 1 ≤ d ∧ d < n}
  let sum_odd := divs.filter (λ d, d % 2 = 1).sum id
  let sum_even := divs.filter (λ d, d % 2 = 0).sum id
  sum_odd < sum_even :=
  sorry

end sum_odd_lt_sum_even_l263_263356
