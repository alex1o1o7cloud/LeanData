import Mathlib
import Mathlib.Algebra.Digitaroot
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Nat.Pow
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time.Calendar
import Mathlib.Geometry.Euclidean
import Mathlib.Probability
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.MetricSpace.Polish
import Mathlib.Trigonometry.Basic

namespace find_circle_center_l779_779024

theorem find_circle_center :
  ∀ x y : ℝ,
  (x^2 + 4*x + y^2 - 6*y = 20) →
  (x + 2, y - 3) = (-2, 3) := by
  sorry

end find_circle_center_l779_779024


namespace smallest_number_proof_l779_779995

-- Define the type of proof problem
def smallest_number_with_digit_sum_62_and_at_least_three_diff_digits : ℕ :=
  let n := 17999999 in
  (∑ d in n.digits 10, d) = 62 ∧ n.digits 10.to_finset.card ≥ 3 ∧ 
  ∀ m, (∑ d in m.digits 10, d) = 62 ∧ m.digits 10.to_finset.card ≥ 3 → m ≥ n
    
theorem smallest_number_proof :
  smallest_number_with_digit_sum_62_and_at_least_three_diff_digits = 17999999 :=
sorry

end smallest_number_proof_l779_779995


namespace girls_on_debate_team_l779_779233

def number_of_students (groups: ℕ) (group_size: ℕ) : ℕ :=
  groups * group_size

def total_students_debate_team : ℕ :=
  number_of_students 8 9

def number_of_boys : ℕ := 26

def number_of_girls : ℕ :=
  total_students_debate_team - number_of_boys

theorem girls_on_debate_team :
  number_of_girls = 46 :=
by
  sorry

end girls_on_debate_team_l779_779233


namespace total_ways_to_sit_in_5_seats_l779_779826

theorem total_ways_to_sit_in_5_seats (n := 5) (k := 3) : 
  (n.perm k) = 60 := 
by 
  -- _5P_3 (permutations of 3 from 5)
  sorry

end total_ways_to_sit_in_5_seats_l779_779826


namespace valid_domain_l779_779013

noncomputable def domain_of_sqrt_expression (x : ℝ) : Prop :=
  sqrt (x / (x - 1)) = sqrt (x / (x - 1))

theorem valid_domain :
  ∀ x : ℝ, domain_of_sqrt_expression x ↔ (0 ≤ x ∧ x ≠ 1) :=
by
  sorry

end valid_domain_l779_779013


namespace sqrt_nested_expression_l779_779275

theorem sqrt_nested_expression : 
  Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := 
by
  sorry

end sqrt_nested_expression_l779_779275


namespace count_ordered_triples_lcm_l779_779170

theorem count_ordered_triples_lcm (a b c : ℕ) :
  (Nat.lcm a b = 1000) ∧ (Nat.lcm b c = 2000) ∧ (Nat.lcm c a = 2000) →
  (∃! t : Finset (ℕ × ℕ × ℕ), t.card = 70) :=
begin
  sorry,
end

end count_ordered_triples_lcm_l779_779170


namespace exists_fixed_circle_with_constant_distance_ratio_l779_779626

/-- Given two bodies moving along two straight lines with constant and unequal speeds,
there exists a fixed circle such that the ratio of the distances from any point
on this circle to these bodies is constant and equal to the ratio of their speeds -/
theorem exists_fixed_circle_with_constant_distance_ratio
  (A B : ℝ → ℝ × ℝ × ℝ)
  (vA vB : ℝ)
  (h1 : ∀ t1 t2 : ℝ, A t1 ≠ B t2)
  (h2 : ∀ t : ℝ, A (t + 1) = A t + (vA, 0, 0))
  (h3 : ∀ t : ℝ, B (t + 1) = B t + (vB, 0, 0))
  (h4 : vA ≠ vB) :
  ∃ (C : ℝ × ℝ × ℝ), ∃ (r : ℝ), ∀ x : ℝ × ℝ × ℝ, dist x A = k * dist x B :=
sorry

end exists_fixed_circle_with_constant_distance_ratio_l779_779626


namespace area_bounded_by_parametric_eqs_l779_779276

-- declaring the given parametric equations and the area calculation
theorem area_bounded_by_parametric_eqs :
  let x := λ t : ℝ, (√2 : ℝ) * Real.cos t
  let y := λ t : ℝ, 4 * (√2 : ℝ) * Real.sin t
  let a := λ t₁ t₂, 8 * ∫ t in t₁..t₂, Real.sin t * (-Real.sin t)
  let b := ∫ t in (3 * Real.pi / 4 : ℝ).. (Real.pi / 4), -4 * (1 - Real.cos (2 * t) / 2)
  (-4 * b) = 2 * Real.pi - 4 :=
  sorry

end area_bounded_by_parametric_eqs_l779_779276


namespace area_of_abs_sum_eq_six_l779_779371

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l779_779371


namespace proof_problem_l779_779178

-- Define constants and parameters
def k : ℝ := real.sqrt (real.sqrt 2)

-- State the problem
theorem proof_problem (x : ℚ) (hx : 0 ≤ x) : 
  let A : ℤ := 2,
      B : ℤ := 2,
      C : ℤ := 2,
      a : ℤ := 1,
      b : ℤ := 2,
      c : ℤ := 2 in
  (| (A * (x^2) + B * x + C) / (a * (x^2) + b * x + c) - k | < | x - k |) :=
  sorry

end proof_problem_l779_779178


namespace shift_symmetric_minimum_m_value_l779_779977

theorem shift_symmetric_minimum_m_value :
  ∃ m > 0, (∀ x, sin (2 * (x - m) + π / 6) = sin (2 * (-x) + π / 6)) ∧ m = π / 3 :=
by
  sorry

end shift_symmetric_minimum_m_value_l779_779977


namespace value_of_f_cos_10_l779_779777

theorem value_of_f_cos_10 (f : ℝ → ℝ) (h1 : ∀ (x : ℝ), f (sin x) = cos (3 * x)) :
  f (cos (10 * (π / 180))) = -1 / 2 :=
by
  sorry

end value_of_f_cos_10_l779_779777


namespace smallest_positive_period_of_f_max_min_values_of_f_l779_779103

open Real

def f (x : ℝ) : ℝ := cos x * sin x - 1 / 2 * cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
by sorry

theorem max_min_values_of_f :
  (∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 ∧ f x ≥ -1 / 2) ∧
  (∃ c, 0 ≤ c ∧ c ≤ π / 2 ∧ f c = 1) ∧
  (∃ d, 0 ≤ d ∧ d ≤ π / 2 ∧ f d = -1 / 2) :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_l779_779103


namespace triangle_CD_length_l779_779151

theorem triangle_CD_length (A B C D : Type) [MetricSpace A] 
  (a b c : A)
  (h1 : ∠ b a c = 135)
  (h2 : dist a b = 4)
  (h3 : dist b c = 5)
  (h4 : LinePerpendicularToPoint a b at a intersects 
        LinePerpendicularToPoint b c at c at d) :
  dist c d = 4.5 * sqrt 2 :=
begin
  sorry
end

end triangle_CD_length_l779_779151


namespace M_intersection_N_l779_779839

noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 16}
noncomputable def N := {x : ℝ | x ≥ 1 / 3}

theorem M_intersection_N :
  (M ∩ N) = {x : ℝ | 1 / 3 ≤ x ∧ x < 16} := by
sorry

end M_intersection_N_l779_779839


namespace son_work_duration_l779_779682

-- Define the variables and conditions
def work_duration_man := 6  -- Days it takes for the man to complete the work alone
def work_duration_together := 3  -- Days it takes for the man and his son to complete the work together

-- Using the conditions to define the statement
theorem son_work_duration:
  let M := (1 : ℚ) / work_duration_man in  -- Man's work rate
  let combined_work_rate := (1 : ℚ) / work_duration_together in  -- Combined work rate
  let S := combined_work_rate - M in  -- Son's work rate
  1 / S = 6 :=  -- The time for the son to complete the work alone
by
  sorry

end son_work_duration_l779_779682


namespace sin_gt_cos_range_l779_779993

theorem sin_gt_cos_range : 
  {x ∈ set.Ioo (0:Real) (2 * Real.pi) | Real.sin x > Real.cos x} = set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := 
by
  sorry

end sin_gt_cos_range_l779_779993


namespace at_least_50_singers_l779_779491

def youth_summer_village (total people_not_working people_with_families max_subset : ℕ) : Prop :=
  total = 100 ∧ 
  people_not_working = 50 ∧ 
  people_with_families = 25 ∧ 
  max_subset = 50

theorem at_least_50_singers (S : ℕ) (h : youth_summer_village 100 50 25 50) : S ≥ 50 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end at_least_50_singers_l779_779491


namespace travel_with_decreasing_ticket_prices_l779_779853

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ∀ (i j : ℕ), i ≠ j → ℕ) : 
  cities.card = n ∧
  (∀ i j, i ≠ j → train_prices i j = train_prices j i) ∧
  (∀ i j k l, (i ≠ j ∧ k ≠ l ∧ (i ≠ k ∨ j ≠ l)) → train_prices i j ≠ train_prices k l) →
  ∃ (start : ℕ), ∃ (route : list (ℕ × ℕ)), 
  route.length = n - 1 ∧ 
  (∀ (m : ℕ), m < route.length - 1 → train_prices route.nth m route.nth (m+1) > train_prices route.nth (m+1) route.nth (m+2)) :=
by 
  sorry

end travel_with_decreasing_ticket_prices_l779_779853


namespace area_of_abs_sum_eq_six_l779_779372

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l779_779372


namespace range_of_a_l779_779432

def f (a x : ℝ) := (x - 2)^2 * abs (x - a)

def derivative_f (a x : ℝ) : ℝ :=
  if x < a 
  then (x - 2) * (-3 * x + 2 + 2 * a)
  else (x - 2) * (3 * x - 2 - 2 * a)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (2:ℝ) (4:ℝ), x * derivative_f a x ≥ 0) ↔ (a ≤ 2 ∨ a ≥ 5) :=
by sorry

end range_of_a_l779_779432


namespace domain_and_parity_range_of_a_l779_779787

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a

theorem domain_and_parity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x * g a x = f a (-x) * g a (-x)) ∧ (∀ x, -1 < x ∧ x < 1) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 1 + g a (1/4) < 1) :
  (a ∈ (Set.Ioo 0 1 ∪ Set.Ioi (3/2))) :=
sorry

end domain_and_parity_range_of_a_l779_779787


namespace good_points_on_graph_l779_779485

theorem good_points_on_graph :
  {p : ℤ × ℕ | ∃ x : ℤ, ∃ y : ℕ, y = (x - 90)^2 - 4907 ∧ x = p.1 ∧ y = p.2 ∧
                           x ∈ {444, -264, 2544, -2364} ∧ y ∈ {120409, 6017209}} =
  {[⟨444, 120409⟩, ⟨-264, 120409⟩, ⟨2544, 6017209⟩, ⟨-2364, 6017209⟩]} :=
sorry

end good_points_on_graph_l779_779485


namespace range_of_function_l779_779763

theorem range_of_function :
  set.range (λ x : ℝ, (3 * x + 4) / (x - 5)) = set.Ioo (-∞) 3 ∪ set.Ioo 3 ∞ :=
sorry

end range_of_function_l779_779763


namespace population_growth_l779_779728

noncomputable def population (x : ℕ) : ℝ := 1.3 * (1.01 ^ x)

theorem population_growth (x : ℕ) (hx : x > 0) : 
  ∃ y : ℝ, y = 1.3 * (1.01 ^ x) :=
begin
  use population x,
  simp [population],
end

end population_growth_l779_779728


namespace find_x_l779_779661

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 682.50) : x = 210 :=
by
  sorry

end find_x_l779_779661


namespace no_solution_x_l779_779936

theorem no_solution_x : ¬ ∃ x : ℝ, x * (x - 1) * (x - 2) + (100 - x) * (99 - x) * (98 - x) = 0 := 
sorry

end no_solution_x_l779_779936


namespace minimum_n_for_tournament_l779_779523

/-- Minimum number of people n such that for any possible outcome of table tennis games, 
    there exists an ordered four people group (a1, a2, a3, a4) where ai wins against aj
    for all 1 ≤ i < j ≤ 4 is n = 8. -/
theorem minimum_n_for_tournament (n : ℕ) (h : n ≥ 4) :
  (∃ (exists_ordered_four_group : Π(games : Fin n → Fin n → Prop), 
       ∃ (a1 a2 a3 a4 : Fin n), 
       (∀ (i j : Fin 4), i < j → games (a1 + i) (a1 + j))) →
  n = 8) :=
sorry

end minimum_n_for_tournament_l779_779523


namespace sequence_product_eq_3_div_5_l779_779700

noncomputable def b : ℕ → ℚ 
| 0       := 1 / 3
| (n + 1) := 1 + 2 * (b n - 1) ^ 3

theorem sequence_product_eq_3_div_5 : (∀ n, ∑' i, (b i) = b 0 * b 1 * b 2 * ... ) :=
begin
  sorry
end

end sequence_product_eq_3_div_5_l779_779700


namespace smallest_non_isosceles_triangles_l779_779698

theorem smallest_non_isosceles_triangles (n : ℕ) (h₁ : n = 2008)
  (triangulation : ∃ t, t ⊆ set.univ × set.univ.prod set.univ ∧ t.finite ∧ 
                    (∀ {x y z}, (x, y) ∈ t → (y, z) ∈ t → (x, z) ∈ t) ∧ 
                    cardinal.mk t = 2005) : 
  ∃ m, m = 5 ∧ m = 
  inf {k | ∃ triang : set (ℕ × ℕ × ℕ), triang ⊆ set.univ × set.univ.prod (set.univ.prod set.univ) ∧
        triang.finite ∧ 
        (∀ {x y z}, (x, y, z) ∈ triang → ¬is_isosceles x y z) ∧
        (∀ {x y z}, (x, y) ∈ t → (y, z) ∈ t → (x, z) ∈ t → (x, y, z) ∈ triang) ∧
        cardinal.mk triang ≤ k } :=
sorry

end smallest_non_isosceles_triangles_l779_779698


namespace build_wall_in_days_l779_779503

noncomputable def constant_k : ℝ := 20 * 6
def inverse_proportion (m : ℝ) (d : ℝ) : Prop := m * d = constant_k

theorem build_wall_in_days (d : ℝ) : inverse_proportion 30 d → d = 4.0 :=
by
  intros h
  have : d = constant_k / 30,
    sorry
  rw this,
  exact (by norm_num : 120 / 30 = 4.0)

end build_wall_in_days_l779_779503


namespace max_value_of_m_l779_779116

theorem max_value_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 > 0 → x < m) → m = -2 :=
by
  sorry

end max_value_of_m_l779_779116


namespace total_coins_l779_779547

def piles_of_quarters : Nat := 5
def piles_of_dimes : Nat := 5
def coins_per_pile : Nat := 3

theorem total_coins :
  (piles_of_quarters * coins_per_pile) + (piles_of_dimes * coins_per_pile) = 30 := by
  sorry

end total_coins_l779_779547


namespace function_characterization_l779_779357

/-- Define the set of positive integers -/
def ℕ+ := { n : ℕ // n > 0 }

/-- Define the set of functions from positive integers to positive integers -/
def pos_int_function := ℕ+ → ℕ+

/-- The main theorem statement -/
theorem function_characterization (f : pos_int_function) :
  (∀ p n : ℕ+, prime p → f(n) ^ (p : ℕ) % f(p) = n % f(p)) ↔
  (f = λ n, n) ∨ (∀ p, prime p → f(p) = 1 ∧ ∀ n, f(n) % 2 = n % 2) :=
sorry

end function_characterization_l779_779357


namespace exists_travel_route_l779_779857

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l779_779857


namespace meeting_point_l779_779559

/-
Points A, B, and C are positioned sequentially, with distances AB = a and BC = b.
A cyclist heads from A to C and a pedestrian heads from B to A.
Both reach their destinations at the same time.
Prove that they meet at a distance AD from A, where AD = a(a + b) / (2a + b).
-/
theorem meeting_point (a b : ℝ) : 
  let AD := (a * (a + b)) / (2 * a + b)
  in AD = (a * (a + b)) / (2 * a + b) :=
by
  -- Proof can be inserted here 
  sorry

end meeting_point_l779_779559


namespace quadratic_complete_square_l779_779016

theorem quadratic_complete_square (a b c : ℤ) (x : ℝ) :
  (a = -3) →
  (b = -4) →
  (c = 129) →
  (-3 * x^2 + 24 * x + 81 = a * (x + b)^2 + c) ∧ (a + b + c = 122) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have H1 : -3 * x^2 + 24 * x + 81 = -3 * (x^2 - 8 * x) + 81 by ring
  have H2 : -3 * (x^2 - 8 * x) = -3 * ((x - 4)^2 - 16) by ring
  have H3 : -3 * ((x - 4)^2 - 16) = -3 * (x - 4)^2 + 48 by ring
  have H4 : -3 * (x - 4)^2 + 48 + 81 = -3 * (x - 4)^2 + 129 by ring
  exact ⟨by rw [H1, H2, H3, H4], by ring⟩

end quadratic_complete_square_l779_779016


namespace total_marbles_l779_779184

theorem total_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l779_779184


namespace count_non_prime_numbers_l779_779006

open List Nat

/-- Sum of digits is 7 and digits are 1, 2, or 3 -/
def valid_digits (l : List ℕ) : Prop := all (fun x => x = 1 ∨ x = 2 ∨ x = 3) l ∧ sum l = 7

/-- Check if the number formed by the digits is prime -/
def digits_not_prime (l : List ℕ) : Prop := 
  let n := l.foldl (fun acc x => acc * 10 + x) 0
  ¬ prime n

/-- Determine the number of numbers composed exclusively of the digits 1, 2, 3 such that the sum 
of its digits equals 7 and none of its digits is zero that are not prime is 30 -/
theorem count_non_prime_numbers : 
  (card (filter digits_not_prime (sigma (fun l => perm l [1, 1, 1, 1, 1, 1, 1] ∨
                                                  perm l [1, 1, 1, 1, 1, 2] ∨
                                                  perm l [1, 1, 1, 1, 3] ∨
                                                  perm l [1, 1, 1, 2, 2] ∨
                                                  perm l [1, 1, 3, 2] ∨
                                                  perm l [1, 2, 2, 2] ∨
                                                  perm l [3, 3, 1] ∨
                                                  perm l [3, 2, 2])))) = 30 :=
  sorry

end count_non_prime_numbers_l779_779006


namespace striped_jerseys_count_l779_779520

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l779_779520


namespace range_of_m_l779_779059

variables {R : Type} [LinearOrder R] [IsDomain R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_increasing_on (f : R → R) (S : Set R) :=
  ∀ {x₁ x₂}, x₁ ∈ S → x₂ ∈ S → x₁ < x₂ → f x₁ < f x₂

noncomputable def f : R → R := sorry

axiom f_even : even_function f
axiom f_domain : ∀ x, x ∈ Set.univ
axiom f_condition : ∀ {x₁ x₂ : R}, x₁ ∈ Set.Ici 0 → x₂ ∈ Set.Ici 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem range_of_m (m : R) : f (m + 1) ≥ f 2 ↔ m ∈ (Set.Iic (-3) ∪ Set.Ici 1) :=
by
  sorry

end range_of_m_l779_779059


namespace additional_birds_correct_l779_779278

-- Assuming there were initially 231 birds in the tree
def initial_birds : ℕ := 231

-- Assuming there were 312 birds in total after more flew up to the tree
def total_birds : ℕ := 312

-- The number of birds that flew up to the tree
def additional_birds : ℕ := total_birds - initial_birds

-- Prove that the number of additional birds is 81
theorem additional_birds_correct : additional_birds = 81 :=
by
  have h1 : initial_birds = 231 := rfl
  have h2 : total_birds = 312 := rfl
  have h3 : additional_birds = total_birds - initial_birds := rfl
  rw [h1, h2, h3]
  norm_num

-- sorry is added to indicate the proof is omitted.

end additional_birds_correct_l779_779278


namespace max_t_l779_779411

-- Definitions
def f (x : ℝ) : ℝ := Real.log x - 2 * x
def g (x : ℝ) (t : ℝ) : ℝ := 1 / t * (-x^2 + 2 * x)

-- Statement of the theorem
theorem max_t (t : ℝ) : ∃ x_0 ∈ Set.Icc 1 Real.exp 1, f x_0 + x_0 ≥ g x_0 t → 
  t ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
  sorry

end max_t_l779_779411


namespace function_domain_l779_779946

noncomputable def sqrt_domain : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ 2 - x > 0 ∧ 2 - x ≠ 1}

theorem function_domain :
  sqrt_domain = {x | -1 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end function_domain_l779_779946


namespace sum_of_divisors_eq_n_l779_779712

theorem sum_of_divisors_eq_n (p : ℕ) (h_prime : Prime (2^p - 1)) :
  let n := 2^(p-1) * (2^p - 1) in
  (∑ d in (Finset.filter (λ d, d ≠ n) (Finset.divisors n)), d) = n :=
by
  sorry

end sum_of_divisors_eq_n_l779_779712


namespace maximum_area_of_triangle_l779_779493

-- Define the geometric and trigonometric context
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Conditions given
variables h1 : (a + b) * (real.sin A - real.sin B) = (c - b) * real.sin C
variable ha : a = 2
variable hpos₁ : 0 < A ∧ A < π
variable hpos₂ : 0 < B ∧ B < π
variable hpos₃ : 0 < C ∧ C < π
variable hsum : A + B + C = π

-- Formal statement
theorem maximum_area_of_triangle (h1: (a + b) * (real.sin A - real.sin B) = (c - b) * real.sin C)
(aeq : a = 2) (apos₁ : 0 < A ∧ A < π) (apos₂ : 0 < B ∧ B < π) (apos₃ : 0 < C ∧ C < π) 
(asum : A + B + C = π) :
  (1/2) * b * c * real.sin A ≤ sqrt 3 :=
sorry

end maximum_area_of_triangle_l779_779493


namespace general_formula_sum_inequality_l779_779899

noncomputable def T (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  list.prod (list.map a (list.range n))

variables {a : ℕ → ℝ}
variables (h1 : a 1 = 1) (h2 : a 2 = 2) (h3 : ∀ n, T a n * T a (n + 2) = 2 * (T a (n + 1))^2)

theorem general_formula (n : ℕ) : a n = 2^(n-1) :=
sorry

theorem sum_inequality (n : ℕ) : ∑ k in finset.range n, T a (2 * k + 1) / T a (2 * k + 2) < 2 / 3 :=
sorry

end general_formula_sum_inequality_l779_779899


namespace largest_consecutive_even_integer_l779_779236

theorem largest_consecutive_even_integer (n : ℕ) (h : 5 * n - 20 = 2 * 15 * 16 / 2) : n = 52 :=
sorry

end largest_consecutive_even_integer_l779_779236


namespace height_difference_percentage_l779_779644

variable {A B : ℝ} (h : A = B * 0.65)

theorem height_difference_percentage (h : A = B * 0.65) : (B - A) / A * 100 ≈ 53.85 :=
by
  -- proof goes here
  sorry

end height_difference_percentage_l779_779644


namespace sum_of_divisors_l779_779237

variable (i j k : ℕ) 

theorem sum_of_divisors (h : (Finset.range (i+1)).sum (λ n, 2^n) * (Finset.range (j+1)).sum (λ n, 3^n) * (Finset.range (k+1)).sum (λ n, 5^n) = 1800) :
  i + j + k = 8 :=
sorry

end sum_of_divisors_l779_779237


namespace red_red_pairs_l779_779479

theorem red_red_pairs (green_shirts red_shirts total_students total_pairs green_green_pairs : ℕ)
    (hg1 : green_shirts = 64)
    (hr1 : red_shirts = 68)
    (htotal : total_students = 132)
    (htotal_pairs : total_pairs = 66)
    (hgreen_green_pairs : green_green_pairs = 28) :
    (total_students = green_shirts + red_shirts) ∧
    (green_green_pairs ≤ total_pairs) ∧
    (∃ red_red_pairs, red_red_pairs = 30) :=
by
  sorry

end red_red_pairs_l779_779479


namespace rank_friends_proof_l779_779008

def David : Type := ℕ
def Emma : Type := ℕ
def Fiona : Type := ℕ

axiom E_neq_F : Emma ≠ Fiona
axiom F_neq_D : Fiona ≠ David
axiom D_neq_E : David ≠ Emma

axiom one_true : (Eldest : Prop) ∨ (Youngest : Prop) ∨ (Not_Eldest : Prop)
axiom one_true_exactly : ¬ ((Eldest ∧ Youngest) ∨ (Eldest ∧ Not_Eldest) ∨ (Youngest ∧ Not_Eldest))

noncomputable def ProblemStatement := 
(Emma > David ∧ Emma > Fiona → False) ∧ 
(Fiona < David ∧ Fiona < Emma ∧ (David > Emma ∧ David > Fiona)) ∧ 
(David ≠ Emma → David ≠ Fiona → Emma ≠ Fiona →
 rank_friends David Emma Fiona = "David, Emma, Fiona")

noncomputable def rank_friends (d e f : Type) : String := "David, Emma, Fiona"

theorem rank_friends_proof : ProblemStatement :=
sorry

end rank_friends_proof_l779_779008


namespace num_ways_books_distribution_l779_779298

def num_books_total := 8
def min_books_library := 1
def min_books_checked_out := 2

theorem num_ways_books_distribution : 
  (∃ in_library : ℕ, (min_books_library ≤ in_library ∧ in_library ≤ (num_books_total - min_books_checked_out)) ∧ 
                     (num_books_total - in_library) ≥ min_books_checked_out) ∧ 
  (finset.Icc min_books_library (num_books_total - min_books_checked_out)).card = 6 :=
begin
  sorry
end

end num_ways_books_distribution_l779_779298


namespace consecutive_integers_avg_l779_779617

theorem consecutive_integers_avg (n x : ℤ) (h_avg : (2*x + n - 1 : ℝ)/2 = 20.5) (h_10th : x + 9 = 25) :
  n = 10 :=
by
  sorry

end consecutive_integers_avg_l779_779617


namespace exists_parallelline_bisecting_area_l779_779567

theorem exists_parallelline_bisecting_area
  (A B C : Point)
  (l : Line)
  (triangle : Triangle A B C) :
  ∃ l₁ : Line, parallel l l₁ ∧ 
  bisects_area l₁ (triangle_area A B C) :=
  sorry

end exists_parallelline_bisecting_area_l779_779567


namespace solve_f_solve_c_solve_m_l779_779434

noncomputable def find_f (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  f = λ x, a * x ^ 3 + b * x ^ 2 - 3 * x

theorem solve_f (a b : ℝ) (f f' : ℝ → ℝ) (h : find_f a b f) (h_deriv : ∀ x, f' x = 3 * a * x ^ 2 + 2 * b * x - 3)
  (h_even : ∀ x, f' (-x) = f' x) (h_at_1 : f' 1 = 0) :
  f = (λ x, x ^ 3 - 3 * x) :=
by
  sorry

noncomputable def find_c (x1 x2 x : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ) : ℝ :=
  |g x1 - g x2|

theorem solve_c (f : ℝ → ℝ) (g : ℝ → ℝ) (x1 x2 : ℝ) (c : ℝ) (h_g : ∀ x, g x = (1 / 3) * f x - 6 * Real.log x)
  (h_in_interval : ∀ x ∈ Icc 1 2, ∀ y ∈ Icc 1 2, |g x - g y| ≤ c) :
  c = - (4 / 3) + 6 * Real.log 2 :=
by
  sorry

noncomputable def find_m (m : ℝ) (h : ℝ → ℝ) : Prop :=
  h = λ x, -3 * x ^ 4 + 8 * x ^ 3 + 3 * x ^ 2 - 12 * x

theorem solve_m (m : ℝ) (h : ℝ → ℝ) (k : ℝ) (h_deriv : ∀ x, h' x = -12 * x^3 + 24 * x^2 + 6 * x - 12)
  (h_tangent : ∀ x0, ∃ k, ∀ x, 2 * f x0 + (4 * x0 ^ 3 - 6 * x0) * (2 - x0))
  (h_sol : (m = 4) ∨ (m = (3 / 4) - 4 * Real.sqrt 2)) :
  m ∈ {4, (3 / 4) - 4 * Real.sqrt 2} :=
by
  sorry

end solve_f_solve_c_solve_m_l779_779434


namespace two_buttons_diff_size_color_l779_779352

variables (box : Type) 
variable [Finite box]
variables (Big Small White Black : box → Prop)

axiom big_ex : ∃ x, Big x
axiom small_ex : ∃ x, Small x
axiom white_ex : ∃ x, White x
axiom black_ex : ∃ x, Black x
axiom size : ∀ x, Big x ∨ Small x
axiom color : ∀ x, White x ∨ Black x

theorem two_buttons_diff_size_color : 
  ∃ x y, x ≠ y ∧ (Big x ∧ Small y ∨ Small x ∧ Big y) ∧ (White x ∧ Black y ∨ Black x ∧ White y) := 
by
  sorry

end two_buttons_diff_size_color_l779_779352


namespace balls_in_boxes_l779_779825

theorem balls_in_boxes : (∃ (balls boxes : ℕ), balls = 7 ∧ boxes = 3) →
  (number_of_ways_to_place := boxes ^ balls) = 2187 :=
by
  intro h
  cases h with balls boxes
  cases h_right with hballs hboxes
  have number_of_ways_to_place := boxes ^ balls
  sorry

end balls_in_boxes_l779_779825


namespace systematic_sampling_example_l779_779394

theorem systematic_sampling_example : 
  ∃ (init : ℕ), (1 ≤ init ∧ init ≤ 10) ∧ 
  {init, init + 10, init + 20, init + 30, init + 40} = {3, 13, 23, 33, 43} :=
begin
  sorry
end

end systematic_sampling_example_l779_779394


namespace radian_measure_of_minute_hand_rotation_l779_779638

theorem radian_measure_of_minute_hand_rotation :
  ∀ (t : ℝ), (t = 10) → (2 * π / 60 * t = -π/3) := by
  sorry

end radian_measure_of_minute_hand_rotation_l779_779638


namespace fourth_number_in_11th_row_l779_779737

-- Define the nth row using a function
def nth_row (n : ℕ) : ℕ → ℕ := λ k, 5 * (n - 1) + k

-- State the theorem
theorem fourth_number_in_11th_row : nth_row 11 4 = 54 := 
by sorry

end fourth_number_in_11th_row_l779_779737


namespace polynomial_P_xx_not_odd_degree_l779_779726

theorem polynomial_P_xx_not_odd_degree {R : Type*} [CommRing R] (P : R[X][Y]) (m n : ℕ)
  (hP_deg_x : ∀ (f : R[X]), C f ∈ P.support → degree f ≤ m)
  (hP_deg_y : ∀ (y : R), degree (P.eval y) ≤ n) 
  (hn_ge_m : n ≥ m) : ¬ odd (degree (P.eval₂ C id)) := 
sorry

end polynomial_P_xx_not_odd_degree_l779_779726


namespace find_k_value_l779_779677

theorem find_k_value :
  ∃ k : ℝ, (∃ (l : ℝ), (0, 5) ∈ l ∧ (7, k) ∈ l ∧ (25, 2) ∈ l) → k = 4.16 :=
by {
  use 4.16,
  intro l_exist,
  cases l_exist with l H,
  sorry
}

end find_k_value_l779_779677


namespace monotonic_increase_interval_l779_779128

noncomputable def f (a x : ℝ) : ℝ := log a (x^2 + 3 / 2 * x)

theorem monotonic_increase_interval 
  (a : ℝ) (hₐ : a > 1): ∀ x, x > 0 → ((∃ y, y ∈ Ioo 0 x ↔ f a y ≥ f a x) :=
by
  sorry

end monotonic_increase_interval_l779_779128


namespace show_positive_result_l779_779261

theorem show_positive_result :
  ∃ (x : ℤ), x = (-1)^2 ∧ x > 0 :=
by
  use 1
  split
  · calc (-1)^2 = 1 : by norm_num
  · exact zero_lt_one

end show_positive_result_l779_779261


namespace hypotenuse_length_l779_779595

theorem hypotenuse_length
  (x : ℝ) 
  (h_leg_relation : 3 * x - 3 > 0) -- to ensure the legs are positive
  (hypotenuse : ℝ)
  (area_eq : 1 / 2 * x * (3 * x - 3) = 84)
  (pythagorean : hypotenuse^2 = x^2 + (3 * x - 3)^2) :
  hypotenuse = Real.sqrt 505 :=
by 
  sorry

end hypotenuse_length_l779_779595


namespace sum_f_1_to_2010_l779_779401

open Real

def f (n : ℕ) : ℝ :=
  sin ((n * π / 2) + (π / 4))

theorem sum_f_1_to_2010 : (Finset.range 2010).sum (λ n, f (n + 1)) = 0 := by
  sorry

end sum_f_1_to_2010_l779_779401


namespace none_of_properties_hold_l779_779168

def at (a b : ℝ) : ℝ := a * b - 1

theorem none_of_properties_hold (x y z : ℝ) :
  ¬(x @ (y + z) = (x @ y) + (x @ z)) ∧
  ¬(x + (y @ z) = (x + y) @ (x + z)) ∧
  ¬(x @ (y @ z) = (x @ y) @ (x @ z)) :=
by
  sorry

end none_of_properties_hold_l779_779168


namespace hari_contribution_correct_l779_779921

-- Translate the conditions into definitions
def praveen_investment : ℝ := 3360
def praveen_duration : ℝ := 12
def hari_duration : ℝ := 7
def profit_ratio_praveen : ℝ := 2
def profit_ratio_hari : ℝ := 3

-- The target Hari's contribution that we need to prove
def hari_contribution : ℝ := 2160

-- Problem statement: prove Hari's contribution given the conditions
theorem hari_contribution_correct :
  (praveen_investment * praveen_duration) / (hari_contribution * hari_duration) = profit_ratio_praveen / profit_ratio_hari :=
by {
  -- The statement is set up to prove equality of the ratios as given in the problem
  sorry
}

end hari_contribution_correct_l779_779921


namespace green_shirts_l779_779042

theorem green_shirts (total_shirts blue_shirts : ℕ) (h1 : total_shirts = 23) (h2 : blue_shirts = 6) :
  total_shirts - blue_shirts = 17 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end green_shirts_l779_779042


namespace biased_coin_probability_l779_779666

-- Define the conditions and question
theorem biased_coin_probability :
  ∃ p : ℝ, p < 1/2 ∧ (3 * p * (1 - p)^2 = 1/2) ∧ (p ≈ 0.3177) :=
sorry

end biased_coin_probability_l779_779666


namespace Jake_has_8_peaches_l779_779160

variable (Jake Steven Jill : ℕ)

theorem Jake_has_8_peaches
  (h_steven_peaches : Steven = 15)
  (h_steven_jill : Steven = Jill + 14)
  (h_jake_steven : Jake = Steven - 7) :
  Jake = 8 := by
  sorry

end Jake_has_8_peaches_l779_779160


namespace unique_prime_B_l779_779741

theorem unique_prime_B : ∃! B : ℕ, B ∈ {1, 2, 3, 7, 9} ∧ Nat.Prime (303160 + B) :=
begin
  sorry
end

end unique_prime_B_l779_779741


namespace number_of_knick_knacks_l779_779161

-- Defining conditions as hypotheses
variables
  (num_hardcover_books : ℕ) (weight_hardcover_book : ℕ) (num_textbooks : ℕ)
  (weight_textbook : ℕ) (num_knick_knacks : ℕ) (weight_knick_knack : ℕ)
  (max_weight : ℕ) (exceed_weight : ℕ)

-- Initiate the given conditions
def conditions : Prop :=
  num_hardcover_books = 70 ∧ weight_hardcover_book = 1/2 ∧
  num_textbooks = 30 ∧ weight_textbook = 2 ∧
  weight_knick_knack = 6 ∧ max_weight = 80 ∧ exceed_weight = 33

-- The proposition to prove
theorem number_of_knick_knacks (h : conditions) : num_knick_knacks = 3 :=
  by
    sorry

end number_of_knick_knacks_l779_779161


namespace inradius_DEF_eq_l779_779704

variables (ABC : Triangle) (D E F : Point)
variable (r r' : ℝ) -- The inradii of triangles
variable (r'' : ℝ) -- The inradius of triangle DEF

-- Conditions given in the problem
variables (on_BC : Lies_On D BC) (on_CA : Lies_On E CA) (on_AB : Lies_On F AB)
variables (has_inradius_AEF : Has_Inradius (Triangle.mk A E F) r')
variables (has_inradius_BFD : Has_Inradius (Triangle.mk B F D) r')
variables (has_inradius_CDE : Has_Inradius (Triangle.mk C D E) r')
variable (has_inradius_ABC : Has_Inradius ABC r)

-- Conclusion we need to prove
theorem inradius_DEF_eq : Has_Inradius (Triangle.mk D E F) (r - r') := sorry

end inradius_DEF_eq_l779_779704


namespace count_three_digit_numbers_with_sum_9_l779_779149

theorem count_three_digit_numbers_with_sum_9 :
  let digits := {0, 1, 2, 3, 4, 5}
  let valid_numbers := { n // n ∈ {a | a ∈ digits × digits × digits} ∧
                                 let (h, t, u) := n in
                                  h ≠ t ∧ t ≠ u ∧ h ≠ u ∧ -- digits are distinct
                                  h ≠ 0 ∧                 -- first digit is not 0
                                  (h + t + u = 9) }       -- sum of digits is 9
  in
  Fintype.card valid_numbers = 12 :=
by sorry

end count_three_digit_numbers_with_sum_9_l779_779149


namespace f_1986_l779_779829

noncomputable def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 3 * f (a * b)
axiom f_1 : f 1 = 2

theorem f_1986 : f 1986 = 2 :=
by
  sorry

end f_1986_l779_779829


namespace find_other_number_l779_779464

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 12 n = 60) (h_hcf : Nat.gcd 12 n = 3) : n = 15 := by
  sorry

end find_other_number_l779_779464


namespace exists_positive_integral_solutions_l779_779749

theorem exists_positive_integral_solutions :
  ∃ (x : Fin 1985 → ℕ) (y z : ℕ), 
    (∀ i j, i ≠ j → x i ≠ x j) ∧ 
    (∀ i, 0 < x i) ∧ 
    (∑ i, (x i)^2 = y^3) ∧ 
    (∑ i, (x i)^3 = z^2) :=
by
  -- problem statement
  sorry

end exists_positive_integral_solutions_l779_779749


namespace final_hair_length_l779_779545

theorem final_hair_length (x y z : ℕ) (hx : x = 16) (hy : y = 11) (hz : z = 12) : 
  (x - y) + z = 17 :=
by
  sorry

end final_hair_length_l779_779545


namespace ned_washed_shirts_l779_779913

theorem ned_washed_shirts (short_sleeve long_sleeve not_washed: ℕ) (h1: short_sleeve = 9) (h2: long_sleeve = 21) (h3: not_washed = 1) : 
    (short_sleeve + long_sleeve - not_washed = 29) :=
by
  sorry

end ned_washed_shirts_l779_779913


namespace diagonal_difference_is_four_l779_779002

def original_matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]]

def modified_matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![![1, 2, 3, 4, 5],
    ![10, 9, 8, 7, 6],
    ![15, 14, 13, 12, 11],
    ![16, 17, 18, 19, 20],
    ![25, 24, 23, 22, 21]]

def main_diagonal_sum (mat : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  finset.univ.sum (λ i, mat i i)

def anti_diagonal_sum (mat : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  finset.univ.sum (λ i, mat i (Fin.rev i))

theorem diagonal_difference_is_four :
  |main_diagonal_sum modified_matrix - anti_diagonal_sum modified_matrix| = 4 :=
by
  sorry

end diagonal_difference_is_four_l779_779002


namespace intersection_empty_l779_779792

open Set

def A : Set ℤ := {0, 1, 2}

def B : Set ℝ := {x | (x + 1) * (x + 2) ≤ 0}

theorem intersection_empty : (A : Set ℝ) ∩ B = ∅ := by
  sorry

end intersection_empty_l779_779792


namespace negate_statement_six_l779_779085

theorem negate_statement_six
  (engineer_mathematics : Prop → Prop → Prop) -- Condition (1) & (2)
  (doctor_mathematics : Prop → Prop) -- Condition (3) & (4) & (5) & (6)
  (S1 : ∀ x, engineer_mathematics x True) -- All engineers are good at mathematics
  (S2 : ∃ x, engineer_mathematics x True) -- Some engineers are good at mathematics
  (S3 : ∀ x, doctor_mathematics x = False) -- No doctors are good at mathematics
  (S4 : ∀ x, doctor_mathematics x = False) -- All doctors are bad at mathematics
  (S5 : ∃ x, doctor_mathematics x = False) -- At least one doctor is bad at mathematics
  (S6 : ∀ x, doctor_mathematics x = True) -- All doctors are good at mathematics
  : S5 = ¬ S6 := sorry

end negate_statement_six_l779_779085


namespace antipov_inequality_l779_779789

theorem antipov_inequality (a b c : ℕ) 
  (h1 : ¬ (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) 
  (h2 : (ab + 1) ∣ (abc + 1)) : c ≥ b :=
sorry

end antipov_inequality_l779_779789


namespace find_xyz_l779_779068

variables (x y z s : ℝ)

theorem find_xyz (h₁ : (x + y + z) * (x * y + x * z + y * z) = 12)
    (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
    (hs : x + y + z = s) : xyz = -8 :=
by
  sorry

end find_xyz_l779_779068


namespace find_t_l779_779182

variable (t : ℝ)

def x := Real.exp (1 - 2 * t)
def y := 3 * Real.sin (2 * t - 2) + Real.cos (2 * t)

theorem find_t : x = y → Real.exp (1 - 2 * t) = 3 * Real.sin (2 * t - 2) + Real.cos (2 * t) := by
  intro h
  exact h

end find_t_l779_779182


namespace ratio_of_white_mice_l779_779849

theorem ratio_of_white_mice (white_mice brown_mice : ℕ) (h_white : white_mice = 14) (h_brown : brown_mice = 7) :
  (white_mice / (white_mice + brown_mice)) = 2 / 3 :=
by
  -- Introduce the proof context and assumptions
  have h_total : white_mice + brown_mice = 14 + 7, { rw [h_white, h_brown] },
  -- Simplify the total number of mice
  have total_mice : ℕ := white_mice + brown_mice,
  rw h_total at total_mice,
  -- Calculate the ratio and simplify
  have ratio : ℚ := white_mice / (white_mice + brown_mice),
  rw [h_white, h_brown] at ratio,
  norm_cast at ratio,
  norm_num at ratio,
  exact ratio

end ratio_of_white_mice_l779_779849


namespace income_expenditure_ratio_l779_779953

noncomputable def I : ℝ := 19000
noncomputable def S : ℝ := 3800
noncomputable def E : ℝ := I - S

theorem income_expenditure_ratio : (I / E) = 5 / 4 := by
  sorry

end income_expenditure_ratio_l779_779953


namespace side_equal_third_of_perimeter_l779_779598

theorem side_equal_third_of_perimeter
  {A B C H K M : Point} (r : ℝ) (s : ℝ) (S : ℝ)
  (triangle_area : S = r * s)
  (incircle_touch : K ∈ Line(B, C))
  (altitude_base : H ∈ Line(A, perpendicular_projection A (Line B C)))
  (midpoint : is_midpoint M B C)
  (symmetric : symmetric_about K H M)
  (semiperimeter : s = (dist A B + dist B C + dist A C) / 2) :
  dist B C = (dist A B + dist B C + dist A C) / 3 := 
sorry

end side_equal_third_of_perimeter_l779_779598


namespace sort_containers_in_operations_l779_779861

theorem sort_containers_in_operations (n : ℕ) :
  ∃ (operations : ℕ), operations ≤ 2 * n - 1 ∧
  (∀ (stackA stackB : list ℕ), 
    (∀ k, 1 ≤ k ∧ k ≤ n → list.nth stackA (k-1) = some k)
    ∨ (∀ j, 1 ≤ j ∧ j ≤ n → list.nth stackB (j-1) = some j)) :=
sorry

end sort_containers_in_operations_l779_779861


namespace ratio_of_divisors_l779_779895

-- Definition of N
def N : ℕ := 34 * 34 * 63 * 270

-- Definition of sum of odd divisors function
noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (λ d, ¬(d % 2 = 0)) (Finset.divisors n), d

-- Definition of sum of even divisors function
noncomputable def sum_of_even_divisors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d

-- Theorem statement
theorem ratio_of_divisors : (sum_of_odd_divisors N) / (sum_of_even_divisors N) = 1 / 14 := by
  sorry

end ratio_of_divisors_l779_779895


namespace box_height_l779_779240

theorem box_height (V L W : ℝ) (h₁ : V = 144) (h₂ : L = 12) (h₃ : W = 6) :
  ∃ H : ℝ, V = L * W * H ∧ H = 2 :=
by {
  use 2,
  split,
  { rw [h₁, h₂, h₃],
    norm_num, },
  { refl, },
}

end box_height_l779_779240


namespace initial_dimes_l779_779566

theorem initial_dimes (x : ℕ) (h1 : x + 7 = 16) : x = 9 := by
  sorry

end initial_dimes_l779_779566


namespace distinct_arrangements_seven_vertices_l779_779142

/-- In seven consecutive vertices of a regular 100-gon,
checkers of seven different colors are placed.
In one move, you are allowed to move any checker 10 positions clockwise to the 11th position if it is free.
The goal is to gather the checkers in the seven vertices following the initial vertices.
Prove that the number of distinct permutations of these checkers in these seven vertices is 7! -/
theorem distinct_arrangements_seven_vertices : 
  let positions := 7 in
  let checkers_colors := 7 in 
  (fact checkers_colors) = 5040 :=
by 
  -- Define key variables
  let positions := 7
  let checkers_colors := 7
  -- Use the factorial function to find the number of permutations
  have h : (fact checkers_colors) = 5040 := by sorry
  exact h

end distinct_arrangements_seven_vertices_l779_779142


namespace intersecting_chords_ratio_l779_779978

theorem intersecting_chords_ratio {XO YO WO ZO : ℝ} 
    (hXO : XO = 5) 
    (hWO : WO = 7) 
    (h_power_of_point : XO * YO = WO * ZO) : 
    ZO / YO = 5 / 7 :=
by
    rw [hXO, hWO] at h_power_of_point
    sorry

end intersecting_chords_ratio_l779_779978


namespace probability_defective_first_box_l779_779212

noncomputable def box1_pieces : ℕ := 5
noncomputable def box1_defective_pieces : ℕ := 2
noncomputable def box2_pieces : ℕ := 10
noncomputable def box2_defective_pieces : ℕ := 3
noncomputable def total_boxes : ℕ := 2

theorem probability_defective_first_box :
  let p_box1 := (1 : ℚ) / total_boxes
      p_def_given_box1 := box1_defective_pieces / box1_pieces
      p_box2 := (1 : ℚ) / total_boxes
      p_def_given_box2 := box2_defective_pieces / box2_pieces
      p_def := p_box1 * p_def_given_box1 + p_box2 * p_def_given_box2
      p_def_and_box1 := p_box1 * p_def_given_box1 in
  (p_def_and_box1 / p_def) = 4 / 7 :=
by
  sorry

end probability_defective_first_box_l779_779212


namespace min_abs_diff_l779_779119

theorem min_abs_diff (a b : ℕ) (h : a > 0 ∧ b > 0 ∧ ab - 8 * a + 7 * b = 571) : ∃ (m : ℕ), m = 29 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x * y - 8 * x + 7 * y = 571 → |x - y| ≥ m :=
by sorry

end min_abs_diff_l779_779119


namespace problem1_problem2_l779_779473

section TriangleIdentities

variables {R A B C a b c : ℝ}

-- Law of Sines in any triangle
def law_of_sines (hR: R ≠ 0) : a = 2 * R * real.sin A ∧ b = 2 * R * real.sin B ∧ c = 2 * R * real.sin C :=
⟨by { sorry },
 by { sorry },
 by { sorry }⟩

theorem problem1 (hR: R ≠ 0) (h1: a = 2 * R * real.sin A) (h2: b = 2 * R * real.sin B) (h3: c = 2 * R * real.sin C) :
  (a ^ 2 * real.sin (B - C) / (real.sin B * real.sin C)) +
  (b ^ 2 * real.sin (C - A) / (real.sin C * real.sin A)) +
  (c ^ 2 * real.sin (A - B) / (real.sin A * real.sin B)) = 0 :=
sorry

theorem problem2 (hR: R ≠ 0) (h1: a = 2 * R * real.sin A) (h2: b = 2 * R * real.sin B) (h3: c = 2 * R * real.sin C) :
  (a ^ 2 * real.sin (B - C) / (real.sin B + real.sin C)) +
  (b ^ 2 * real.sin (C - A) / (real.sin C + real.sin A)) +
  (c ^ 2 * real.sin (A - B) / (real.sin A + real.sin B)) = 0 :=
sorry

end TriangleIdentities

end problem1_problem2_l779_779473


namespace find_incorrect_option_l779_779821

-- Definitions based on the conditions given.
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (t : ℝ) : ℝ × ℝ := (-4, t)

-- Option A: Parallel vectors
def is_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Option B: Minimum magnitude of the difference of vectors
def min_magnitude_diff (v₁ v₂ : ℝ × ℝ) : ℝ :=
  real.sqrt (v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2

-- Option C: Equal magnitude sum and difference
def equal_mag_sum_diff (v₁ v₂ : ℝ × ℝ) : Prop :=
  real.sqrt (v₁.1 + v₂.1)^2 + (v₁.2 + v₂.2)^2 = real.sqrt (v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2

-- Option D: Obtuse angle between vectors
def obtuse_angle (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 < 0

-- The theorem to identify the incorrect statement
theorem find_incorrect_option (t : ℝ) :
  ¬ (
    (is_parallel vec_a (vec_b t) → t = -8) ∧
    (min_magnitude_diff vec_a (vec_b t) = 5) ∧
    (equal_mag_sum_diff vec_a (vec_b t) → t = 2) ∧
    (obtuse_angle vec_a (vec_b t) → t < 2)
  ) :=
sorry

end find_incorrect_option_l779_779821


namespace sum_of_areas_l779_779580

theorem sum_of_areas (r R : ℝ) (n : ℕ) (h1 : n = 10) (h2 : R = 2) (h3 : ∀ i, i < n ↔ congruent_disks_around_circle r R n i) :
  let a := 15
  let b := 5
  let c := 5
  a + b + c = 25 :=
sorry

end sum_of_areas_l779_779580


namespace round_trip_time_l779_779588

/-- The commuter rail between Scottsdale and Sherbourne is 200 km of track. One train makes a round trip in a certain amount of time. Harsha boards the train at the Forest Grove station, which is located one fifth of the track's length out from Scottsdale. It takes her 2 hours to get to Sherbourne. Prove that it takes the train 5 hours to make a round trip. -/
theorem round_trip_time (track_length : ℕ) (portion_forest_grove : ℚ) (time_forest_grove_to_sherbourne : ℚ)
  (h1 : track_length = 200)
  (h2 : portion_forest_grove = 1 / 5)
  (h3 : time_forest_grove_to_sherbourne = 2) :
  let speed := (track_length - (track_length * portion_forest_grove)) / time_forest_grove_to_sherbourne in
  let total_time := (track_length / speed) * 2 in
  total_time = 5 := 
by
  sorry

end round_trip_time_l779_779588


namespace compute_g_neg_101_l779_779576

variable (g : ℝ → ℝ)

def functional_eqn := ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
def g_neg_one := g (-1) = 3
def g_one := g (1) = 1

theorem compute_g_neg_101 (g : ℝ → ℝ)
  (H1 : functional_eqn g)
  (H2 : g_neg_one g)
  (H3 : g_one g) :
  g (-101) = 103 := 
by
  sorry

end compute_g_neg_101_l779_779576


namespace find_x_l779_779121

variable (x y : ℝ)

theorem find_x (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end find_x_l779_779121


namespace impossible_labeling_l779_779961

def labeling (n : ℕ) := Fin n → Fin 4

def satisfies_composite_square (f : labeling n) : Prop :=
∀ i j : Fin n, ∃ a b c d : Fin 4, set.eq (set.of_finset (i, j) { f (i + 1) (j), f (i) (j + 1), f (i + 1) (j + 1), f(i) (j) }) = {a,b,c,d}

def satisfies_row_column (f : labeling n) : Prop :=
∀ k : Fin n, ∃ a b c d : Fin 4,
  (∀ (i : Fin n), set.eq (set.of_finset { f k i }) a b c d) ∧
  (∀ (j : Fin n), set.eq (set.of_finset { f i k }) a b c d)

theorem impossible_labeling : ¬ ∃ f : labeling n, satisfies_composite_square f ∧ satisfies_row_column f :=
sorry

end impossible_labeling_l779_779961


namespace johns_total_amount_l779_779514

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l779_779514


namespace f_of_3_eq_11_l779_779801

theorem f_of_3_eq_11 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + 1 / x^2) : f 3 = 11 :=
by
  sorry

end f_of_3_eq_11_l779_779801


namespace region_area_correct_l779_779365

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l779_779365


namespace lucky_325th_is_52000_l779_779841

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_lucky (a : ℕ) : Prop :=
  digit_sum a = 7

def lucky_numbers := {a : ℕ | is_lucky a}

def a_seq (n : ℕ) : option ℕ :=
  list.nth (list.sort (≤) (lucky_numbers.to_list)) (n - 1)

theorem lucky_325th_is_52000 : a_seq 325 = some 52000 :=
sorry

end lucky_325th_is_52000_l779_779841


namespace num_combinations_of_4_choose_2_l779_779915

theorem num_combinations_of_4_choose_2 : finset.card (finset.powerset_len 2 (finset.range 4)) = 6 :=
by
  sorry

end num_combinations_of_4_choose_2_l779_779915


namespace odd_function_condition_l779_779601

def f (x : ℝ) (m n : ℝ) : ℝ :=
  x * |sin x + m| + n

theorem odd_function_condition (m n : ℝ) : (∀ x : ℝ, f (-x) m n = -f x m n) ↔ (m * n = 0) := by
  sorry

end odd_function_condition_l779_779601


namespace students_no_A_l779_779850

def total_students : Nat := 40
def students_A_chemistry : Nat := 10
def students_A_physics : Nat := 18
def students_A_both : Nat := 6

theorem students_no_A : (total_students - (students_A_chemistry + students_A_physics - students_A_both)) = 18 :=
by
  sorry

end students_no_A_l779_779850


namespace range_of_a_l779_779350

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l779_779350


namespace total_donation_correct_l779_779107

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l779_779107


namespace min_value_of_frac_l779_779805

theorem min_value_of_frac (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_cond : a + 2 * b = 1) :
  (∃ c : ℝ, c = (1 / a + 2 / b) ∧ ∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → (1 / x + 2 / y) ≥ c) :=
begin
  use 9,
  sorry
end

end min_value_of_frac_l779_779805


namespace other_tables_have_3_legs_l779_779671

-- Define the given conditions
variables (total_tables four_legged_tables : ℕ)
variables (total_legs legs_four_legged_tables : ℕ)

-- State the conditions as Lean definitions
def dealer_conditions :=
  total_tables = 36 ∧
  four_legged_tables = 16 ∧
  total_legs = 124 ∧
  legs_four_legged_tables = 4 * four_legged_tables

-- Main theorem to prove the number of legs on the other tables
theorem other_tables_have_3_legs (cond : dealer_conditions total_tables four_legged_tables total_legs legs_four_legged_tables) :
  let other_tables := total_tables - four_legged_tables in
  let other_legs := total_legs - legs_four_legged_tables in
  other_tables > 0 →
  other_legs % other_tables = 0 →
  other_legs / other_tables = 3 :=
sorry

end other_tables_have_3_legs_l779_779671


namespace percentage_error_in_calculated_area_l779_779267

theorem percentage_error_in_calculated_area
  (a : ℝ)
  (measured_side_length : ℝ := 1.025 * a) :
  (measured_side_length ^ 2 - a ^ 2) / (a ^ 2) * 100 = 5.0625 :=
by 
  sorry

end percentage_error_in_calculated_area_l779_779267


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779271

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779271


namespace polynomial_divisible_by_x_minus_2_l779_779748

theorem polynomial_divisible_by_x_minus_2 (k : ℝ) :
  (2 * (2 : ℝ)^3 - 8 * (2 : ℝ)^2 + k * (2 : ℝ) - 10 = 0) → 
  k = 13 :=
by 
  intro h
  sorry

end polynomial_divisible_by_x_minus_2_l779_779748


namespace f_cos_10_l779_779397

def f (x : ℝ) : ℝ := 2 * real.arcsin x + 1

theorem f_cos_10 :
  real.cos 10 ∈ [-1, 1] →
  real.sin⁻¹(real.cos 10) ∈ [-real.pi / 2, real.pi / 2] →
  f (real.cos 10) = 21 - 7 * real.pi := 
by
  sorry

end f_cos_10_l779_779397


namespace thabo_books_difference_l779_779215

-- Defining the number of books
variables {P F : ℕ}

-- Constants derived from the given problem
constant number_of_books : ℕ := 200
constant hardcover_nonfiction : ℕ := 35

-- Conditions derived from the given problem
def condition1 : Prop := P + F + hardcover_nonfiction = number_of_books
def condition2 : Prop := F = 2 * P
def condition3 : Prop := P > hardcover_nonfiction

-- The theorem we need to prove
theorem thabo_books_difference :
  condition1 ∧ condition2 ∧ condition3 → (P - hardcover_nonfiction = 20) :=
by
  sorry

end thabo_books_difference_l779_779215


namespace odd_function_f_expression_f_neg_l779_779903

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then x * (1 + x) else -x * (1 - x)

theorem odd_function_f (x : ℝ) (hx : x > 0) : f(-x) = -f(x) :=
by sorry

theorem expression_f_neg (x : ℝ) (hx : x < 0) : f(x) = x * (1 - x) :=
by sorry

end odd_function_f_expression_f_neg_l779_779903


namespace zero_count_l779_779347

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017^x + log 2017 x else
  if x < 0 then -(2017^(-x) + log 2017 (-x)) else 0

theorem zero_count (f : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_pos : ∀ x, x > 0 → f x = 2017^x + log 2017 x) :
  ∃! z ∈ set.Icc (-∞ : ℝ) (∞ : ℝ), f z = 0 ∧ set.Icc (-∞ : ℝ) (∞ : ℝ) = {z | z = 0} :=
sorry

end zero_count_l779_779347


namespace VictoriaGymSessionPlan_l779_779983

-- Define the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the gym schedule specification for Victoria
def VictoriaGymSessions (start_day : DayOfWeek) (end_day : DayOfWeek) : ℕ :=
  if start_day = DayOfWeek.Monday ∧ end_day = DayOfWeek.Friday then 3 else 0

-- The statement we want to prove
theorem VictoriaGymSessionPlan :
  VictoriaGymSessions DayOfWeek.Monday DayOfWeek.Friday = 3 :=
by
  sorry

end VictoriaGymSessionPlan_l779_779983


namespace probability_divisible_by_5_l779_779920

def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_5 :
  (finset.card ((finset.filter (λ (x : ℕ × ℕ × ℕ), 
    is_divisible_by_five (x.1 * x.2 + x.1 + x.3))
    (finset.product (finset.product (finset.range 28) (finset.range 28))
                    (finset.range 28))))
  : ℕ) =
  50 / 243 :=
sorry

end probability_divisible_by_5_l779_779920


namespace lcm_15_25_35_l779_779999

-- Define the three conditions: the numbers 15, 25, and 35
def a : Nat := 15
def b : Nat := 25
def c : Nat := 35

-- Define a function to compute the LCM
def lcm (m n : Nat) : Nat := Nat.lcm m n

-- Prove that lcm (a, (lcm (b, c))) = 525
theorem lcm_15_25_35 : lcm a (lcm b c) = 525 := by
  sorry

end lcm_15_25_35_l779_779999


namespace tan_half_angle_product_l779_779342

noncomputable def cos_half_angle (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (cos a + cos b) + 3 * (cos a * cos b - 1) = 0) : 
  ∃ u : ℝ, (u = sqrt ((-7 + sqrt 133) / 3) ∨ u = -sqrt ((-7 + sqrt 133) / 3) ∨ 
            u = sqrt ((-7 - sqrt 133) / 3) ∨ u = -sqrt ((-7 - sqrt 133) / 3)) ∧ 
            (u = tan (a / 2) * tan (b / 2)) :=
sorry

end tan_half_angle_product_l779_779342


namespace percentage_markup_l779_779959

theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 4800) (h₂ : cost_price = 3840) :
  (selling_price - cost_price) / cost_price * 100 = 25 :=
by
  sorry

end percentage_markup_l779_779959


namespace bill_before_tax_l779_779996

theorem bill_before_tax (T E : ℝ) (h1 : E = 2) (h2 : 3 * T + 5 * E = 12.70) : 2 * T + 3 * E = 7.80 :=
by
  sorry

end bill_before_tax_l779_779996


namespace sin_square_22_5_equals_l779_779654

theorem sin_square_22_5_equals :
  2 * sin (22.5 * real.degToRad) ^ 2 - 1 = - (real.sqrt 2 / 2) := 
sorry

end sin_square_22_5_equals_l779_779654


namespace area_of_closed_figure_l779_779582

theorem area_of_closed_figure :
  ∫ x in 1..Real.exp 1, x - (1/x) = (Real.exp 1 ^ 2 - 3) / 2 :=
by sorry

end area_of_closed_figure_l779_779582


namespace exists_chord_division_l779_779416

noncomputable theory

structure Annulus where
  O : Point  -- center of the circles
  r1 : ℝ    -- radius of the smaller circle
  r2 : ℝ    -- radius of the larger circle
  h_radii : r1 < r2

structure Chord (A1 A2 : Point) where
  lies_on_circle1 : dist A1 O = r1
  lies_on_circle2 : dist A2 O = r2

def chord_division (P : Point) (m n : ℝ) (ann : Annulus) : Prop :=
  ∃ (A1 A2 : Point), 
    Chord A1 A2 ∧ 
    P ∈ LineSegment A1 A2 ∧ 
    dist A1 P * n = dist P A2 * m

theorem exists_chord_division (P : Point) (m n : ℝ) (ann : Annulus) 
  (h_in_annulus : r1 < dist O P ∧ dist O P < r2): 
  ∃ (A1 A2 : Point), 
    chord_division P m n ann := sorry

end exists_chord_division_l779_779416


namespace length_ZD_in_triangle_XYZ_l779_779155

theorem length_ZD_in_triangle_XYZ :
  ∀ (X Y Z D : Type*)
  [HasDistance X Y (8 : Real)]
  [HasDistance Y Z (15 : Real)]
  [HasDistance X Z (17 : Real)]
  [AngleBisector Z D XY YZ],
  ∃ (ZD : Real), ZD = Real.sqrt 132897 / 23 :=
by
  sorry

end length_ZD_in_triangle_XYZ_l779_779155


namespace problem_I_problem_II_l779_779436

def f (x : ℝ) : ℝ := abs (x - 1)

theorem problem_I (x : ℝ) : f (2 * x) + f (x + 4) ≥ 8 ↔ x ≤ -10 / 3 ∨ x ≥ 2 := by
  sorry

variable {a b : ℝ}
theorem problem_II (ha : abs a < 1) (hb : abs b < 1) (h_neq : a ≠ 0) : 
  (abs (a * b - 1) / abs a) > abs ((b / a) - 1) :=
by
  sorry

end problem_I_problem_II_l779_779436


namespace tan_ineq_of_acuteangled_triangle_l779_779890

variables {A B C : ℝ} (α β γ : ℝ)

def is_AcuteAngledTriangle (α β γ : ℝ) (A B C : ℝ) : Prop :=
  α + β + γ = π ∧ 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2

noncomputable def perimeter (A B C : ℝ) : ℝ := A + B + C -- assuming A, B, C are side lengths of the triangle

noncomputable def inradius (A B C : ℝ) : ℝ := sorry -- inradius calculation to be filled in

theorem tan_ineq_of_acuteangled_triangle (α β γ : ℝ)
  (h : is_AcuteAngledTriangle α β γ A B C)
  (p : ℝ) (r : ℝ) (hp : perimeter A B C = p) (hr : inradius A B C = r) :
  (Real.tan α + Real.tan β + Real.tan γ) ≥ p / (2 * r) :=
sorry

end tan_ineq_of_acuteangled_triangle_l779_779890


namespace apples_per_slice_l779_779334

theorem apples_per_slice 
  (dozens_apples : ℕ)
  (apples_per_dozen : ℕ)
  (number_of_pies : ℕ)
  (pieces_per_pie : ℕ) :
  dozens_apples = 4 →
  apples_per_dozen = 12 →
  number_of_pies = 4 →
  pieces_per_pie = 6 →
  (dozens_apples * apples_per_dozen) / (number_of_pies * pieces_per_pie) = 2 :=
by
  intros h_dozen h_per_dozen h_pies h_pieces
  rw [h_dozen, h_per_dozen, h_pies, h_pieces]
  norm_num
  sorry

end apples_per_slice_l779_779334


namespace max_value_expression_l779_779781

open Real

theorem max_value_expression (k : ℝ) (h : k > 0) : 
  (sup { x | ∃ k > 0, x = (3*k^3 + 3*k) / ((3/2*k^2 + 14) * (14*k^2 + 3/2)) }) = (sqrt 21) / 175 
:= sorry

end max_value_expression_l779_779781


namespace function_continuity_l779_779165

variables {a b : ℝ} (f : ℝ → ℝ)

-- Conditions
variable (h1 : a < b)
variable (h2 : ∀ x y, a < x → x < y → y < b → ((x - a) * f x ≤ (y - a) * f y))
variable (h3 : ∀ x y, a < x → x < y → y < b → ((x - b) * f x ≤ (y - b) * f y))

-- To prove
theorem function_continuity (h1 : a < b) (h2 : ∀ x y, a < x → x < y → y < b → ((x - a) * f x ≤ (y - a) * f y)) (h3 : ∀ x y, a < x → x < y → y < b → ((x - b) * f x ≤ (y - b) * f y)) : 
  continuous_on f (set.Ioo a b) :=
sorry

end function_continuity_l779_779165


namespace simplified_sqrt_expression_l779_779987

theorem simplified_sqrt_expression : 
    let a := 5 - 3 * Real.sqrt 2
    let b := 5 + 3 * Real.sqrt 2
    Real.sqrt (a^2) + Real.sqrt (b^2) = 10 :=
by
  let a := 5 - 3 * Real.sqrt 2
  let b := 5 + 3 * Real.sqrt 2
  have ha : Real.sqrt (a^2) = abs a := Real.sqrt_sq (by norm_num : a ≥ 0)
  have hb : Real.sqrt (b^2) = abs b := Real.sqrt_sq (by norm_num : b ≥ 0)
  rw [ha, hb]
  rw [abs_of_nonneg, abs_of_nonneg]
  {
    norm_num,
    norm_num,
  }
  exact sorry

end simplified_sqrt_expression_l779_779987


namespace tangent_iff_ratio_l779_779548

variable {α : Type*} [EuclideanGeometry α]

def midpoint (a b m : Point) : Prop :=
  dist a m = dist b m

def tangent_circumcircle (a b c d n m : Point) : Prop :=
  (is_tangent a b c n) ∧ (midpoint a d m) ∧ (ray b m n)

theorem tangent_iff_ratio (a b c d n m : Point) (h_mid_ad : midpoint a d m) (h_ray_bm_ac : ray b m n) :
  (is_tangent a b c n) ↔ (dist b m / dist b n = (dist b c / dist b n)^2) :=
sorry

end tangent_iff_ratio_l779_779548


namespace min_value_f_when_a1_l779_779071

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |x - a|

theorem min_value_f_when_a1 : ∀ x : ℝ, f x 1 ≥ 3/4 :=
by sorry

end min_value_f_when_a1_l779_779071


namespace angle_HFG_is_60_l779_779889

/-- Definition of the problem Setup -/
structure circle_geometry :=
  (O : Point)
  (A B F : Point)
  (diameter : Line)
  (tangent_at_B : Line)
  (tangent_at_F : Line)
  (G H : Point)
  (angle_BAF : ℝ)

noncomputable def given_problem (geom : circle_geometry) : Prop :=
  (is_circle geom.O) ∧
  (is_diameter geom.A geom.B geom.diameter) ∧
  (on_circle geom.O geom.F) ∧
  (tangent_at_point geom.O geom.B geom.tangent_at_B) ∧
  (tangent_at_point geom.O geom.F geom.tangent_at_F) ∧
  (intersects geom.tangent_at_B geom.tangent_at_F geom.G) ∧
  (intersects geom.tangent_at_B (line_through geom.A geom.F) geom.H) ∧
  (geom.angle_BAF = 30)

theorem angle_HFG_is_60 (geom : circle_geometry) (h : given_problem geom) : ∠ HFG = 60 := 
  by
  sorry

end angle_HFG_is_60_l779_779889


namespace distinct_x_intercepts_l779_779112

theorem distinct_x_intercepts : 
  let f (x : ℝ) := ((x - 8) * (x^2 + 4*x + 3))
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
by
  sorry

end distinct_x_intercepts_l779_779112


namespace tan_alpha_value_l779_779046

theorem tan_alpha_value (α : ℝ) (h : Real.sin (α / 2) = 2 * Real.cos (α / 2)) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_value_l779_779046


namespace man_gets_dividend_l779_779299

/-- Given conditions -/
constant investment_amount : ℝ := 14400
constant face_value : ℝ := 100
constant premium_percentage : ℝ := 0.20
constant dividend_rate : ℝ := 0.05

/-- Calculate derived quantities -/
def cost_per_share := face_value * (1 + premium_percentage)
def number_of_shares := investment_amount / cost_per_share
def dividend_per_share := face_value * dividend_rate
def total_dividend := dividend_per_share * number_of_shares

/-- Prove the total dividend -/
theorem man_gets_dividend : total_dividend = 600 := by
{sqrry,  sorry}

end man_gets_dividend_l779_779299


namespace largest_shaded_area_l779_779007

def square_area (side: ℝ) : ℝ :=
  side * side

def circle_area (radius: ℝ) : ℝ :=
  Real.pi * radius * radius

-- Conditions
def side_length : ℝ := 3
def quarter_circle_radius : ℝ := 1

-- Shaded areas calculation
def shaded_area_A : ℝ :=
  square_area side_length - circle_area (side_length / 2) / 4

def shaded_area_B : ℝ :=
  square_area side_length - (4 * circle_area quarter_circle_radius / 4)

def shaded_area_C : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let radius := diagonal / 2
  square_area side_length - circle_area radius / 4

-- The proof problem statement
theorem largest_shaded_area :
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C :=
sorry

end largest_shaded_area_l779_779007


namespace repeated_number_divisibility_l779_779684

theorem repeated_number_divisibility (x : ℕ) (h : 1000 ≤ x ∧ x < 10000) :
  73 ∣ (10001 * x) ∧ 137 ∣ (10001 * x) :=
sorry

end repeated_number_divisibility_l779_779684


namespace roots_quadratic_identity_l779_779456

theorem roots_quadratic_identity 
  (a b c r s : ℝ)
  (h_root1 : a * r^2 + b * r + c = 0)
  (h_root2 : a * s^2 + b * s + c = 0)
  (h_distinct_roots : r ≠ s)
  : (1 / r^2) + (1 / s^2) = (b^2 - 2 * a * c) / c^2 := 
by
  sorry

end roots_quadratic_identity_l779_779456


namespace perpendicular_vectors_lambda_l779_779102

theorem perpendicular_vectors_lambda (λ : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, λ))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : λ = 1 / 2 :=
by
  sorry

end perpendicular_vectors_lambda_l779_779102


namespace rectangular_solid_edges_sum_l779_779696

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l779_779696


namespace kaleb_books_after_transaction_l779_779522

def initial_books : ℕ := 34
def sold_percentage : ℝ := 0.45
def sold_books : ℕ := (sold_percentage * initial_books).floor
def bought_books : ℕ := sold_books / 3

theorem kaleb_books_after_transaction :
  initial_books - sold_books + bought_books = 24 :=
sorry

end kaleb_books_after_transaction_l779_779522


namespace maximal_abelian_normal_subgroup_infinite_l779_779884

-- initial definition for the group G
def is_generated_by_nilpotent (G : Type) [group G] : Prop :=
  ∃ (S : set (subgroup G)), (∀ H ∈ S, H.normal) ∧ (Hølmo G _ = true) ∧ (∀ H ∈ S, is_nilpotent H)

def is_maximal_abelian_normal_subgroup (G : Type) [group G] (A : subgroup G) : Prop :=
  A.normal ∧ A.abelian ∧ ∀ (B : subgroup G), B.normal ∧ B.abelian → A ≤ B → A = B

theorem maximal_abelian_normal_subgroup_infinite
  (G : Type) [group G]
  (hG : infinite G)
  (hN : is_generated_by_nilpotent G) :
  ∀ (A : subgroup G), is_maximal_abelian_normal_subgroup A → infinite A :=
sorry

end maximal_abelian_normal_subgroup_infinite_l779_779884


namespace c_2018_eq_56_l779_779078
-- Import the full library

-- Define the sequences
def a_n : ℕ → ℤ
| 0       := 1
| (n + 1) := 3 * a_n n

def b_n : ℕ → ℤ
| 0       := -5
| (n + 1) := b_n n + 1

def c_n : ℕ → ℤ
| n :=
  let k := (n + 1) // 2018 in
  if 2 * k * k <= n then 3^(62 : ℤ) else b_n (n - k * (k + 1) // 2 - k - 1)

-- The problem statement to be proven with the exact conditions.
theorem c_2018_eq_56 : c_n 2018 = 56 := sorry

end c_2018_eq_56_l779_779078


namespace distinct_tetrahedrons_l779_779313

/-- 
There are six rods of different lengths, and it is known that no matter how they are ordered, 
they can form a tetrahedron. Prove that the number of distinct tetrahedrons (considering mirror images as indistinguishable) 
that can be formed with these rods is 30.
-/
theorem distinct_tetrahedrons (rods : Fin 6 → ℝ) (h_distinct : ∀ i j : Fin 6, i ≠ j → rods i ≠ rods j) 
    (h_form_tetrahedron : ∀ (perm : Fin 6 → Fin 6), ∃ tetrahedron, tetrahedron.sides = rods ∘ perm) :
    ∃ n : ℕ, n = 30 :=
by
  sorry

end distinct_tetrahedrons_l779_779313


namespace tom_filled_balloons_l779_779624

theorem tom_filled_balloons :
  ∀ (Tom Luke Anthony : ℕ), 
    (Tom = 3 * Luke) →
    (Luke = Anthony / 4) →
    (Anthony = 44) →
    (Tom = 33) :=
by
  intros Tom Luke Anthony hTom hLuke hAnthony
  sorry

end tom_filled_balloons_l779_779624


namespace region_area_correct_l779_779367

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l779_779367


namespace price_change_l779_779646

theorem price_change (p : ℝ) : 
  let p1 := p * (1 - 0.5) in
  let pf := p1 * (1 + 0.6) in
  ((pf - p) / p) * 100 = -20 :=
begin
  sorry
end

end price_change_l779_779646


namespace factorial_fraction_eq_seven_l779_779720

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l779_779720


namespace factorial_fraction_eq_seven_l779_779719

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l779_779719


namespace inverse_function_l779_779222

def f (x : ℝ) := x^2

theorem inverse_function (x : ℝ) (hx : x < -2) :
  (∃ y : ℝ, f y = x ∧ y < -2) ↔ (x > 4 ∧ ∃ y : ℝ, y = -sqrt x) :=
sorry

end inverse_function_l779_779222


namespace fourth_number_sorted_desc_is_0_6_l779_779967

-- Define the given numbers
def numbers : List ℚ := [1.7, 1/5, 1/5, 1, 3/5, 3/8, 1.4]

-- Define the decimal equivalents as rational numbers for clarity in ordering
def numbers_in_decimal : List ℚ := [1.7, 0.2, 0.2, 1.0, 0.6, 0.375, 1.4]

-- Statement: Prove that the fourth number when sorted from largest to smallest is 0.6
theorem fourth_number_sorted_desc_is_0_6 :
  (numbers_in_decimal.sort (≥)).nth 3 = some 0.6 :=
by
  sorry

end fourth_number_sorted_desc_is_0_6_l779_779967


namespace average_length_of_strings_l779_779556

theorem average_length_of_strings {l1 l2 l3 : ℝ} (h1 : l1 = 2) (h2 : l2 = 6) (h3 : l3 = 9) : 
  (l1 + l2 + l3) / 3 = 17 / 3 :=
by
  sorry

end average_length_of_strings_l779_779556


namespace abs_sum_neq_zero_iff_or_neq_zero_l779_779074

variable {x y : ℝ}

theorem abs_sum_neq_zero_iff_or_neq_zero (x y : ℝ) :
  (|x| + |y| ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end abs_sum_neq_zero_iff_or_neq_zero_l779_779074


namespace johns_work_end_time_l779_779018

noncomputable def work_hours_per_day := 9 -- John must work 9 hours each day.
noncomputable def lunch_break := 75 -- Lunch break is 1 hour and 15 minutes.
noncomputable def start_time := Time.mk 6 30 -- John starts working at 6:30 A.M.
noncomputable def lunch_time := Time.mk 11 30 -- John takes his lunch break at 11:30 A.M.

theorem johns_work_end_time (start_time lunch_time : Time)
  (work_hours_per_day : ℤ) (lunch_break : ℤ) : 
  johns_work_end_time = Time.mk 16 45 := 
by 
  sorry

end johns_work_end_time_l779_779018


namespace minimum_sum_of_labels_l779_779663

-- Define the conditions
def label (r c : ℕ) : ℚ := 1 / (r + c - 2)

-- Define the problem statement
theorem minimum_sum_of_labels :
  ∃ (r : ℕ → ℕ), (∀ i, 1 ≤ r i ∧ r i ≤ 9)
  ∧ (function.injective r)
  ∧ (∑ i in finset.range 9, label (r i) (i + 1)) = 1 := sorry

end minimum_sum_of_labels_l779_779663


namespace integer_count_expression_l779_779387

theorem integer_count_expression :
  ∃ (S : Finset ℕ), S.card = 34 ∧ S ⊆ Finset.Icc 1 50 ∧ 
  ∀ n ∈ S, (factorial (n^2 - 1)) % (factorial n ^ n) = 0 := 
by
  -- Define the set of integers from 1 to 50
  let I := Finset.Icc 1 50
  -- Define the condition that (n^2-1)! / (n!)^n is an integer 
  let condition := λ n: ℕ, (factorial (n^2 - 1)) % (factorial n ^ n) = 0
  -- Define set S as the subset of I fulfilling the condition
  let S := I.filter condition
  -- Prove that S has exactly 34 elements
  have hS : S.card = 34 := sorry
  use S, hS
  -- Show that S is a subset of I (already true by construction)
  rw ← Finset.subset_refl S
  -- Show that every element in S satisfies the condition
  intro n hn
  exact Finset.mem_filter.mp hn.right

end integer_count_expression_l779_779387


namespace intersection_eq_l779_779541

-- Define sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

-- The theorem to be proved
theorem intersection_eq : A ∩ B = {2} := 
by sorry

end intersection_eq_l779_779541


namespace smallest_n_rel_prime_to_300_l779_779986

theorem smallest_n_rel_prime_to_300 : ∃ n : ℕ, n > 1 ∧ Nat.gcd n 300 = 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → Nat.gcd m 300 ≠ 1 :=
by
  sorry

end smallest_n_rel_prime_to_300_l779_779986


namespace max_of_squares_leq_sum_of_diffs_l779_779201

theorem max_of_squares_leq_sum_of_diffs {n : ℕ} (a : ℕ → ℝ) (h : ∑ i in finset.range n, a i = 0) :
  max (finset.image (λ k, (a k) ^ 2) (finset.range n)) ≤ (n / 3) * ∑ i in finset.range (n - 1), (a i - a (i + 1)) ^ 2 :=
sorry

end max_of_squares_leq_sum_of_diffs_l779_779201


namespace count_years_with_at_most_two_digits_l779_779554

def is_valid_year (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9999 ∧ ∃ a b : ℕ, (∀ d ∈ (n.digits 10), d = a ∨ d = b) ∧ (a ≠ b → (a < 10 ∧ b < 10) ∧ (a = 0 ∨ 1 ≤ a)) ∧ (a = b → (1 ≤ a ∧ a < 10))

theorem count_years_with_at_most_two_digits : ∃ k, k = 927 ∧ k = (finset.range 10000).filter is_valid_year).card := sorry

end count_years_with_at_most_two_digits_l779_779554


namespace cost_price_of_one_toy_l779_779300

theorem cost_price_of_one_toy (C : ℝ) (h : 21 * C = 21000) : C = 1000 :=
by sorry

end cost_price_of_one_toy_l779_779300


namespace polynomial_identity_equals_neg_one_l779_779117

theorem polynomial_identity_equals_neg_one
  (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by
  intro h
  sorry

end polynomial_identity_equals_neg_one_l779_779117


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779273

-- Define the properties that need to be proven
variables (Q1 Q2 : Prop) (A1 A2 : Prop)

/-- Theorem to prove why joint purchases are popular despite risks -/
theorem joint_purchases_popular : Q1 → A1 :=
begin
  sorry -- proof not provided
end

/-- Theorem to prove why joint purchases are not popular among neighbors for groceries -/
theorem joint_purchases_unpopular_among_neighbors : Q2 → A2 :=
begin
  sorry -- proof not provided
end

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779273


namespace ball_falls_into_hole_within_six_bounces_l779_779785

structure Table :=
  (width : ℕ)
  (height : ℕ)
  (contains_hole : ℕ → ℕ → Prop)

def initial_positions := {A := (0, 0), B := (4, 5), C := (some_x, some_y)}

-- Condition: table dimensions and hole positions
def table : Table := 
  { width := 8, 
    height := 5, 
    contains_hole := λ x y, (x = 0 ∧ y = 0) ∨ (x = 8 ∧ y = 0) ∨ (x = 0 ∧ y = 5) ∨ (x = 8 ∧ y = 5) }

-- Function to simulate ball trajectory on the table with bounces
noncomputable def ball_trajectory (start_pos : ℕ × ℕ) (bounces : ℕ) : ℕ × ℕ :=
  sorry -- The simulation logic is not implemented here

-- Main Proof Statement
theorem ball_falls_into_hole_within_six_bounces
  (A B C : ℕ × ℕ) :
  (ball_trajectory A 6 = (0, 0) ∨ ball_trajectory A 6 = (8, 0) ∨ ball_trajectory A 6 = (0, 5) ∨ ball_trajectory A 6 = (8, 5)) ∨
  (ball_trajectory B 6 = (0, 0) ∨ ball_trajectory B 6 = (8, 0) ∨ ball_trajectory B 6 = (0, 5) ∨ ball_trajectory B 6 = (8, 5)) ∨
  (ball_trajectory C 6 = (0, 0) ∨ ball_trajectory C 6 = (8, 0) ∨ ball_trajectory C 6 = (0, 5) ∨ ball_trajectory C 6 = (8, 5)) :=
sorry

end ball_falls_into_hole_within_six_bounces_l779_779785


namespace factorial_expression_simplification_l779_779723

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l779_779723


namespace average_length_is_21_08_l779_779969

def lengths : List ℕ := [20, 21, 22]
def quantities : List ℕ := [23, 64, 32]

def total_length := List.sum (List.zipWith (· * ·) lengths quantities)
def total_quantity := List.sum quantities

def average_length := total_length / total_quantity

theorem average_length_is_21_08 :
  average_length = 2508 / 119 := by
  sorry

end average_length_is_21_08_l779_779969


namespace time_on_other_subjects_l779_779185

theorem time_on_other_subjects : 
  ∀ (total_time : ℕ) (math_percentage : ℝ) (science_percentage : ℝ),
  total_time = 150 →
  math_percentage = 0.30 →
  science_percentage = 0.40 →
  let time_on_math := math_percentage * total_time in
  let time_on_science := science_percentage * total_time in
  let time_on_other_subjects := total_time - (time_on_math + time_on_science) in
  time_on_other_subjects = 45 :=
by
  intros total_time math_percentage science_percentage h_total_time h_math_percentage h_science_percentage
  simp [h_total_time, h_math_percentage, h_science_percentage]
  let time_on_math := math_percentage * total_time
  let time_on_science := science_percentage * total_time
  have h_time_on_math : time_on_math = 45,
  have h_time_on_science : time_on_science = 60
  show total_time - (time_on_math + time_on_science) = 45
  symmetry
  sorry

end time_on_other_subjects_l779_779185


namespace sum_of_x_and_y_l779_779538

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_of_x_and_y_l779_779538


namespace units_digit_is_valid_l779_779258

theorem units_digit_is_valid (n : ℕ) : 
  (∃ k : ℕ, (k^3 % 10 = n)) → 
  (n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end units_digit_is_valid_l779_779258


namespace find_t_of_decreasing_function_l779_779455

theorem find_t_of_decreasing_function 
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_A : f 0 = 4)
  (h_B : f 3 = -2)
  (h_solution_set : ∀ x, |f (x + 1) - 1| < 3 ↔ -1 < x ∧ x < 2) :
  (1 : ℝ) = 1 :=
by
  sorry

end find_t_of_decreasing_function_l779_779455


namespace scheduling_methods_count_l779_779283

-- Define our conditions
def Days : Type := Fin 7
def Volunteers : Type := {A : Nat // A = 1} ∪ {B : Nat // B = 2} ∪ {C : Nat // C = 3} ∪ {D : Nat // D = 4}

-- Define the question and answer tuple
theorem scheduling_methods_count (h1 : ∀ (d1 d2 : Days), d1 ≠ d2) 
                                 (h2 : ∀ (v1 v2 : Volunteers), v1 ≠ v2) 
                                 (h3 : ∀ (d : Days) (v : Volunteers), d ∈ Days → v ∈ Volunteers → h1 d = v.1 → h2 v = d) 
                                 (A_d : Days)
                                 (B_d : Days)
                                 (h : A_d < B_d) : 
                                 @Finset.card (Equiv.Perm Volunteers) _ = 420 := 
begin 
  sorry 
end

end scheduling_methods_count_l779_779283


namespace area_contained_by_graph_l779_779364

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l779_779364


namespace solve_polynomial_l779_779251

theorem solve_polynomial : 
  ∀ x : ℝ, x^4 - x^2 - 2 = 0 ↔ x = real.sqrt 2 ∨ x = -real.sqrt 2 :=
by sorry

end solve_polynomial_l779_779251


namespace factorial_fraction_eq_seven_l779_779718

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l779_779718


namespace slices_per_pack_l779_779877

theorem slices_per_pack (sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_of_bread : ℕ) (total_slices : ℕ) 
  (h1 : sandwiches = 8) (h2 : slices_per_sandwich = 2) (h3 : packs_of_bread = 4) : 
  total_slices = 4 :=
by
  sorry

end slices_per_pack_l779_779877


namespace gain_per_year_is_200_l779_779641

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem gain_per_year_is_200 :
  let borrowed_principal := 5000
  let borrowing_rate := 4
  let borrowing_time := 2
  let lent_principal := 5000
  let lending_rate := 8
  let lending_time := 2

  let interest_paid := simple_interest borrowed_principal borrowing_rate borrowing_time
  let interest_earned := simple_interest lent_principal lending_rate lending_time

  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / 2

  gain_per_year = 200 := by
  sorry

end gain_per_year_is_200_l779_779641


namespace perimeter_triangle_sin_A_l779_779844

variables (a b c : ℝ)
variables (angle_A angle_B angle_C : ℝ)

-- Conditions from problem
def conditions : Prop := 
  a = 1 ∧ b = 2 ∧ cos angle_C = 3 / 4

-- Problem 1: Prove the perimeter of the triangle
theorem perimeter_triangle (h : conditions) : a + b + c = 3 + Real.sqrt 2 :=
by
  -- Extract conditions
  cases h with ha rest
  cases rest with hb hc
  -- Use the conditions
  sorry

-- Problem 2: Prove the value of sin A
theorem sin_A (h : conditions) : sin angle_A = Real.sqrt 14 / 8 :=
by
  -- Extract conditions
  cases h with ha rest
  cases rest with hb hc
  -- Use the conditions
  sorry

end perimeter_triangle_sin_A_l779_779844


namespace physicist_walked_2_5_miles_on_thursday_l779_779688

noncomputable def find_walking_distance (k : ℝ) := k / 8

theorem physicist_walked_2_5_miles_on_thursday :
  ∃ k : ℝ, (k = 4 * 5) ∧ (find_walking_distance k = 2.5) :=
by 
  let k := 4 * 5
  use k
  split
  . exact rfl
  . unfold find_walking_distance
    exact (div_eq_iff (by norm_num : (8 : ℝ) ≠ 0)).mpr rfl

end physicist_walked_2_5_miles_on_thursday_l779_779688


namespace verify_triangle_operation_l779_779211

def triangle (a b c : ℕ) : ℕ := a^2 + b^2 + c^2

theorem verify_triangle_operation : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end verify_triangle_operation_l779_779211


namespace imaginary_part_l779_779180

noncomputable def complex_modulus (z : ℂ) : ℝ := complex.abs z

theorem imaginary_part (z : ℂ) (hz : z = 3 + 4 * complex.I) : 
  (complex.im (z + complex_modulus z / z) = 16 / 5) :=
by
  sorry

end imaginary_part_l779_779180


namespace geese_in_marsh_l779_779618

theorem geese_in_marsh (D : ℝ) (hD : D = 37.0) (G : ℝ) (hG : G = D + 21) : G = 58.0 := 
by 
  sorry

end geese_in_marsh_l779_779618


namespace sequence_a_2015_l779_779605

theorem sequence_a_2015 (a : ℕ → ℝ) (h₀ : a 1 = 3)
    (h₁ : ∀ n, a (n + 1) = (a n - 1) / a n) : a 2015 = 2 / 3 :=
begin
    sorry
end

end sequence_a_2015_l779_779605


namespace original_average_l779_779943

theorem original_average (A : ℝ) (h : (10 * A = 70)) : A = 7 :=
sorry

end original_average_l779_779943


namespace sum_of_xy_is_1289_l779_779497

-- Define the variables and conditions
def internal_angle1 (x y : ℕ) : ℕ := 5 * x + 3 * y
def internal_angle2 (x y : ℕ) : ℕ := 3 * x + 20
def internal_angle3 (x y : ℕ) : ℕ := 10 * y + 30

-- Definition of the sum of angles of a triangle
def sum_of_angles (x y : ℕ) : ℕ := internal_angle1 x y + internal_angle2 x y + internal_angle3 x y

-- Define the theorem statement
theorem sum_of_xy_is_1289 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : sum_of_angles x y = 180) : x + y = 1289 :=
by sorry

end sum_of_xy_is_1289_l779_779497


namespace find_third_number_l779_779122

-- Definitions and conditions for the problem
def x : ℚ := 1.35
def third_number := 5
def proportion (a b c d : ℚ) := a * d = b * c 

-- Proposition to prove
theorem find_third_number : proportion 0.75 x third_number 9 := 
by
  -- It's advisable to split the proof steps here, but the proof itself is condensed.
  sorry

end find_third_number_l779_779122


namespace area_contained_by_graph_l779_779363

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l779_779363


namespace base_k_number_to_decimal_l779_779217

theorem base_k_number_to_decimal (k : ℕ) (h : 4 ≤ k) : 1 * k^2 + 3 * k + 2 = 30 ↔ k = 4 := by
  sorry

end base_k_number_to_decimal_l779_779217


namespace total_amount_from_grandparents_l779_779508

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l779_779508


namespace fair_coin_run_of_heads_before_tails_l779_779532

theorem fair_coin_run_of_heads_before_tails :
  let q := (6353 - 128) / 6353 in
  let m := 128 in
  let n := 6225 in
  nat.gcd m n = 1 ∧ q = m / n → (m + n = 6353) :=
by
  sorry

end fair_coin_run_of_heads_before_tails_l779_779532


namespace find_pictures_museum_l779_779992

-- Define the given conditions
def pictures_zoo : Nat := 24
def pictures_deleted : Nat := 14
def pictures_remaining : Nat := 22

-- Define the target: the number of pictures taken at the museum
def pictures_museum : Nat := 12

-- State the goal to be proved
theorem find_pictures_museum :
  pictures_zoo + pictures_museum - pictures_deleted = pictures_remaining :=
sorry

end find_pictures_museum_l779_779992


namespace range_of_f_l779_779760

def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  set_of (λ y, ∃ x, f(x) = y) = {y : ℝ | y ≠ 3} :=
by
  sorry

end range_of_f_l779_779760


namespace total_donation_correct_l779_779106

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l779_779106


namespace ratio_is_four_l779_779928

noncomputable def ratio_of_marbles_reina_kevin (Reina_Counters Reina_Marbles Kevin_Counters Kevin_Marbles Total_Counters_Marbles : ℕ) : ℕ :=
  if h : Reina_Counters = 3 * Kevin_Counters
     ∧ Kevin_Counters = 40
     ∧ Kevin_Marbles = 50
     ∧ Total_Counters_Marbles = 320
  then
    let Reina_Marbles := Total_Counters_Marbles - Reina_Counters in
    Reina_Marbles / Kevin_Marbles
  else
    0  -- default value when conditions are not met (should not be reached if conditions are correct)

theorem ratio_is_four :
  ratio_of_marbles_reina_kevin 120 200 40 50 320 = 4 :=
by
  sorry

end ratio_is_four_l779_779928


namespace prob_div_by_4_l779_779560

theorem prob_div_by_4 : 
  let S := {x : ℕ | 1 ≤ x ∧ x ≤ 100};
  let count := λ (s : Set ℕ), s.card;
  let prob := λ (s : Set ℕ), (count s) / (count S);
  let event := {t : ℕ × ℕ × ℕ | t.1 ∈ S ∧ t.2.1 ∈ S ∧ t.2.2 ∈ S ∧ (t.1 * t.2.1 * t.2.2 + t.1 * t.2.1 + t.1 * t.2.2 + t.1) % 4 = 0};

  prob event = 5 / 8 :=
sorry

end prob_div_by_4_l779_779560


namespace diagonal_perimeter_ratio_l779_779676

theorem diagonal_perimeter_ratio
    (b : ℝ)
    (h : b ≠ 0) -- To ensure the garden has non-zero side lengths
    (a : ℝ) (h1: a = 3 * b) 
    (d : ℝ) (h2: d = (Real.sqrt (b^2 + a^2)))
    (P : ℝ) (h3: P = 2 * a + 2 * b)
    (h4 : d = b * (Real.sqrt 10)) :
  d / P = (Real.sqrt 10) / 8 := by
    sorry

end diagonal_perimeter_ratio_l779_779676


namespace john_total_amount_l779_779510

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l779_779510


namespace eccentricity_proof_l779_779891

-- Definitions for the given problem conditions
def hyperbola_eq (a b : ℝ) : Prop := ∀ x y : ℝ, (a > 0 ∧ b > 0) → (x^2 / a^2 - y^2 / b^2 = 1)
def foci_distance (c : ℝ) : ℝ := 2 * c
def distance_condition (a c : ℝ) : Prop := a = 2 * c / 3
def eccentricity (a c : ℝ) : ℝ := c / a

-- The theorem we want to prove
theorem eccentricity_proof (a b c : ℝ) (h_hyperbola : hyperbola_eq a b) (h_distance_condition : distance_condition a c) :
    eccentricity a c = 3 / 2 := by
    sorry

end eccentricity_proof_l779_779891


namespace max_min_values_a_neg6_range_a_l779_779093

noncomputable def f (a : ℝ) (x : ℝ) := x^2 - x + a * Real.log x

theorem max_min_values_a_neg6 : 
  let a := -6 in 
  let f := f a in 
  ∃ x_max x_min, x_max ∈ Set.Icc (1:ℝ) 4 ∧ x_min ∈ Set.Icc (1:ℝ) 4 ∧ 
  (f x_max = Real.exp (12 - 12 * Real.log 2)) ∧ (f x_min = Real.exp (2 - 6 * Real.log 2)) :=
sorry

theorem range_a : 
  (∀ a x, a ≠ 0 → ((f a) x = x^2 - x + a * Real.log x) → 
   (∃ x_min x_max, x_min < x_max ∧ ∀ x, x_min ≤ x ∧ x ≤ x_max → f a x ∈ Set.Icc x_min x_max)) → 
  (0 < a ∧ a < (1 / 8 : ℝ)) := 
sorry

end max_min_values_a_neg6_range_a_l779_779093


namespace max_sin_cos_product_eq_9_over_2_l779_779029

theorem max_sin_cos_product_eq_9_over_2:
  ∀ (x y z : ℝ),
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 := 
sorry

end max_sin_cos_product_eq_9_over_2_l779_779029


namespace coefficient_f_x3_is_neg_448_coefficient_g_x2_is_4_l779_779402

noncomputable def coefficient_f_x3 (f : ℕ → ℕ → ℕ) (x : ℕ) : ℕ := 
  (list.range (x + 1)).sum (λ k, (-1 : ℕ) ^ k * f x k * (2 : ℕ) ^ k)

noncomputable def coefficient_g_x2 (g : ℕ → ℕ → ℕ) (x y : ℕ) : ℕ :=
  (list.range (x + 1)).sum (λ k, g x k * g y (2 - k))

theorem coefficient_f_x3_is_neg_448 : 
  let f := λ n k, nat.choose n k 
  in coefficient_f_x3 f 8 = 448 := 
sorry

theorem coefficient_g_x2_is_4 : 
  let g := λ n k, nat.choose n k 
  in coefficient_g_x2 g 9 8 = 4 := 
sorry

end coefficient_f_x3_is_neg_448_coefficient_g_x2_is_4_l779_779402


namespace pet_store_ratio_proof_l779_779687

def pet_store_animal_ratio : Prop :=
  ∃ (C D B F : ℕ), D = 6 ∧ C + D + B + F = 39 ∧ B = 2 * D ∧ F = 3 * D ∧ C / D = 1 / 2

theorem pet_store_ratio_proof : pet_store_animal_ratio :=
by {
  -- Declaration of variables and hypotheses
  let C := 3,
  let D := 6,
  let B := 12,
  let F := 18,
  have hD : D = 6 := by sorry,
  have htotal : C + D + B + F = 39 := by sorry,
  have hB : B = 2 * D := by sorry,
  have hF : F = 3 * D := by sorry,
  have hratio : C / D = 1 / 2 := by sorry,
  use [C, D, B, F, hD, htotal, hB, hF, hratio]
}

end pet_store_ratio_proof_l779_779687


namespace root_in_interval_l779_779327

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l779_779327


namespace average_speed_calculation_l779_779340

def average_speed (s1 s2 t1 t2 : ℕ) : ℕ :=
  (s1 * t1 + s2 * t2) / (t1 + t2)

theorem average_speed_calculation :
  average_speed 40 60 1 3 = 55 :=
by
  -- skipping the proof
  sorry

end average_speed_calculation_l779_779340


namespace negation_exists_l779_779225

theorem negation_exists {x : ℝ} (h : ∀ x, x > 0 → x^2 - x ≤ 0) : ∃ x, x > 0 ∧ x^2 - x > 0 :=
sorry

end negation_exists_l779_779225


namespace find_percentage_loss_l779_779321

theorem find_percentage_loss 
  (P : ℝ)
  (initial_marbles remaining_marbles : ℝ)
  (h1 : initial_marbles = 100)
  (h2 : remaining_marbles = 20)
  (h3 : (initial_marbles - initial_marbles * P / 100) / 2 = remaining_marbles) :
  P = 60 :=
by
  sorry

end find_percentage_loss_l779_779321


namespace value_of_expression_l779_779634

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l779_779634


namespace edges_remain_same_l779_779669

-- Define a cube 
structure Cube :=
(vertices : Fin 8) -- A cube has 8 vertices
(edges : Fin 12)    -- A cube has 12 edges
(faces : Fin 6)     -- A cube has 6 faces

-- Define the transformation, where each vertex touching three distinct faces is cut off
def transformedCube (c : Cube) : Prop :=
  ∀ (v : c.vertices), -- For each vertex
    ∃ (f : Fin 3),    -- There exists a triangular face created

-- Proposition that the number of edges remains the same after transformation
theorem edges_remain_same (c : Cube) (h : transformedCube c) : c.edges.val = 12 :=
by
  sorry

end edges_remain_same_l779_779669


namespace solve_for_a_l779_779770

noncomputable def perpendicular_tangent_line (a : ℝ) : Prop :=
  let P := (1 : ℝ, 2 : ℝ)
  let circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 1
  let point_to_line_distance := λ (x₀ y₀ a b c : ℝ), abs (a * x₀ + b * y₀ + c) / sqrt (a^2 + b^2)
  let tangent_line_at_point := λ k, k * (1 - 0) - 2 = k * 1 - 2 - k
  let perpendicular_condition := λ k, a = -1 / k ∨ k = 0 ∧ a = 0
  ∃ k, tangent_line_at_point k ∧ (point_to_line_distance 0 0 k (-1) 2 = 1) ∧ perpendicular_condition k

theorem solve_for_a :
  ∀ (a : ℝ), perpendicular_tangent_line a →
  a = 0 ∨ a = -4 / 3 :=
by
  sorry

end solve_for_a_l779_779770


namespace num_real_k_with_integer_roots_l779_779388

noncomputable def count_real_numbers_k_with_integer_roots : ℕ := 
  let possible_k_values := {k : ℝ | ∃ r s : ℤ, r + s = -k ∧ r * s = 4 * k}
  possible_k_values.to_finset.card

theorem num_real_k_with_integer_roots : count_real_numbers_k_with_integer_roots = 3 :=
  sorry

end num_real_k_with_integer_roots_l779_779388


namespace ribbon_length_ratio_l779_779594

theorem ribbon_length_ratio (original_length reduced_length : ℕ) (h1 : original_length = 55) (h2 : reduced_length = 35) : 
  (original_length / Nat.gcd original_length reduced_length) = 11 ∧
  (reduced_length / Nat.gcd original_length reduced_length) = 7 := 
  by
    sorry

end ribbon_length_ratio_l779_779594


namespace calculate_median_and_mean_l779_779133

noncomputable def scores : List ℝ := [16, 17, 25, 28, 29, 30, 32, 37]

def median (s : List ℝ) : ℝ :=
  let sorted := s.qsort (≤)
  let n := sorted.length
  if n % 2 = 1 then sorted[(n / 2) - 1]
  else (sorted[(n / 2) - 1] + sorted[n / 2]) / 2

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem calculate_median_and_mean : 
  median scores = 28.5 ∧ mean scores = 26.75 :=
by
  sorry

end calculate_median_and_mean_l779_779133


namespace tens_digit_less_than_5_probability_l779_779295

theorem tens_digit_less_than_5_probability 
  (n : ℕ) 
  (hn : 10000 ≤ n ∧ n ≤ 99999)
  (h_even : ∃ k, n % 10 = 2 * k ∧ k < 5) :
  (∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 1 / 2) :=
by
  sorry

end tens_digit_less_than_5_probability_l779_779295


namespace find_sum_of_all_possible_values_of_omega_l779_779667

noncomputable def sum_of_omegas (ω : ℂ) (h : ω^5 = 2) : ℂ :=
  (ℂ.sum (λ ω, ω^4 + ω^3 + ω^2 + ω + 1) {ω | ω^5 = 2})

theorem find_sum_of_all_possible_values_of_omega :
  ∑ ω in {ω | ω^5 = 2}, (ω^4 + ω^3 + ω^2 + ω + 1) = 5 :=
by
  sorry

end find_sum_of_all_possible_values_of_omega_l779_779667


namespace solve_triple_l779_779384

theorem solve_triple (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a * b + c = a^3) : 
  (b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1)) :=
by 
  sorry

end solve_triple_l779_779384


namespace least_possible_cost_of_garden_l779_779565

/-- Suppose we have a garden with specified regions and flower costs:
    - Two regions of 8 sq ft each
    - One region of 6 sq ft
    - One region of 25 sq ft
    - Sunflowers cost $1 each
    - Tulips cost $2 each
    - Orchids cost $2.50 each
    - Roses cost $3 each
    - Hydrangeas cost $4 each
    We need to verify that the least possible cost, in dollars, for planting the garden is $128. -/
theorem least_possible_cost_of_garden :
  let sunflowers_cost := 1
  let tulips_cost := 2
  let orchids_cost := 2.50
  let roses_cost := 3
  let hydrangeas_cost := 4
  let region1_area := 8
  let region2_area := 8
  let region3_area := 6
  let region4_area := 25
  in
  (region1_area * sunflowers_cost) + 
  (region2_area * sunflowers_cost) + 
  (region3_area * tulips_cost) + 
  (region4_area * hydrangeas_cost) = 128 := 
by
  sorry

end least_possible_cost_of_garden_l779_779565


namespace inequality_for_a_and_b_l779_779954

theorem inequality_for_a_and_b (a b : ℝ) : 
  (1 / 3 * a - b) ≤ 5 :=
sorry

end inequality_for_a_and_b_l779_779954


namespace Jake_has_more_peaches_than_Jill_l779_779873

variables (Jake Steven Jill : ℕ)
variable (h1 : Jake = Steven - 5)
variable (h2 : Steven = Jill + 18)
variable (h3 : Jill = 87)

theorem Jake_has_more_peaches_than_Jill (Jake Steven Jill : ℕ) (h1 : Jake = Steven - 5) (h2 : Steven = Jill + 18) (h3 : Jill = 87) :
  Jake - Jill = 13 :=
by
  sorry

end Jake_has_more_peaches_than_Jill_l779_779873


namespace avg_class_l779_779474

-- Problem definitions
def total_students : ℕ := 40
def num_students_95 : ℕ := 8
def num_students_0 : ℕ := 5
def num_students_70 : ℕ := 10
def avg_remaining_students : ℝ := 50

-- Assuming we have these marks
def marks_95 : ℝ := 95
def marks_0 : ℝ := 0
def marks_70 : ℝ := 70

-- We need to prove that the total average is 57.75 given the above conditions
theorem avg_class (h1 : total_students = 40)
                  (h2 : num_students_95 = 8)
                  (h3 : num_students_0 = 5)
                  (h4 : num_students_70 = 10)
                  (h5 : avg_remaining_students = 50)
                  (h6 : marks_95 = 95)
                  (h7 : marks_0 = 0)
                  (h8 : marks_70 = 70) :
                  (8 * 95 + 5 * 0 + 10 * 70 + 50 * (40 - (8 + 5 + 10))) / 40 = 57.75 :=
by sorry

end avg_class_l779_779474


namespace polynomial_value_l779_779635

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l779_779635


namespace license_plate_count_l779_779739

def num_license_plates : Nat :=
  26 * 10 * 36

theorem license_plate_count : num_license_plates = 9360 :=
by
  sorry

end license_plate_count_l779_779739


namespace nancy_more_money_l779_779191

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l779_779191


namespace optimal_worker_allocation_l779_779675

theorem optimal_worker_allocation :
  ∃ (x y : ℕ), x + y = 60 ∧ 3 * x * 4 = 6 * y ∧ x = 20 ∧ y = 40 := 
by
  existsi (20 : ℕ)
  existsi (40 : ℕ)
  split
  { exact nat.add_comm 20 40 }
  sorry

end optimal_worker_allocation_l779_779675


namespace exists_travel_route_l779_779855

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l779_779855


namespace find_phi_l779_779429

-- Define the function and its derivative
def f (x φ : ℝ) : ℝ := Real.sin (Real.sqrt 3 * x + φ)
def f' (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.cos (Real.sqrt 3 * x + φ)

-- Define g(x)
def g (x φ : ℝ) : ℝ := f x φ + f' x φ

-- State the main theorem
theorem find_phi 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (h_odd : ∀ x, g x φ = -g (-x) φ) : 
  φ = 2 * π / 3 :=
sorry

end find_phi_l779_779429


namespace sum_f_positive_l779_779778

def f (x : ℝ) : ℝ := x - Real.sin x

theorem sum_f_positive (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x1 + x3 > 0) :
  f x1 + f x2 + f x3 > 0 :=
sorry

end sum_f_positive_l779_779778


namespace costa_rica_points_are_three_l779_779581

def world_cup_group_stage : Type := sorry

variable (Group_E_points_table : world_cup_group_stage)

-- Conditions
axiom four_teams_in_each_group : ∀ (G: world_cup_group_stage), ∃ t1 t2 t3 t4, 
  G = (t1, t2, t3, t4)

axiom single_round_robin : ∀ (G: world_cup_group_stage), 
  ∀ t1 t2 t3 t4, G = (t1, t2, t3, t4) → 
  ∃ (matches: list (prod t1 t1)), 
  matches.length = 6 ∧ sorry -- each pair of teams plays exactly one match

axiom win_points : ∀ (team : Type), ∃ (points : ℕ),
  points = 3

axiom draw_points : ∀ (team : Type), ∃ (points : ℕ),
  points = 1

axiom lose_points : ∀ (team : Type), ∃ (points : ℕ),
  points = 0

axiom points_table_for_Group_E : ∀ (G: world_cup_group_stage), 
  ∀ t1 t2 t3 t4, G = (t1, t2, t3, t4) → 
  (find_points t1 = 6 ∧ find_points t2 = 4 ∧ find_points t3 = 4 ∧ find_points t4 = sorry)

axiom only_one_draw_in_Group_E : ∃ m, 
  sorry -- there exists only one match result in draw in the group.

-- Theorem: Points of Costa Rica based on conditions should be 3.
theorem costa_rica_points_are_three : 
  ∃ G, four_teams_in_each_group G ∧ single_round_robin G ∧ points_table_for_Group_E G →
  ∃ t1 t2 t3 t4, t4 = "Costa Rica" → find_points t4 = 3 := 
by {
  sorry 
}

end costa_rica_points_are_three_l779_779581


namespace license_plate_difference_l779_779735

theorem license_plate_difference :
  (26^3 * 10^4) - (26^4 * 10^3) = -281216000 :=
by
  sorry

end license_plate_difference_l779_779735


namespace total_payment_after_layoff_l779_779612

theorem total_payment_after_layoff
  (total_employees : ℕ)
  (salary_per_employee : ℕ)
  (fraction_laid_off : ℚ)
  (remaining_employees : ℕ)
  (total_payment : ℕ)
  (h1 : total_employees = 450)
  (h2 : salary_per_employee = 2000)
  (h3 : fraction_laid_off = 1 / 3)
  (h4 : remaining_employees = total_employees - (total_employees / 3))
  (h5 : total_payment = remaining_employees * salary_per_employee) :
  total_payment = 600000 :=
by 
  rw [h1, h2, h3] at h4,
  rw h4 at h5,
  norm_num at h5,
  exact h5

end total_payment_after_layoff_l779_779612


namespace number_of_non_zero_decimal_digits_l779_779349

-- Definition of the problem conditions
def fraction : ℚ := 90 / (3^2 * 2^5)

-- Statement of the proof problem
theorem number_of_non_zero_decimal_digits (n : ℕ) :
  n = 4 ↔ (fraction = 5 / 16 ∧ (5 / 16).to_decimal_string.right_of_point_non_zero_digits_count = n) :=
sorry

end number_of_non_zero_decimal_digits_l779_779349


namespace nancy_earns_more_l779_779188

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l779_779188


namespace coeff_of_expr_highest_degree_term_of_poly_l779_779218

-- Definition of the expression and polynomial
def expr := - (2 * a^3 * b * c^2) / 5
def poly := 3 * x^2 * y - 7 * x^4 * y^2 - x * y^4

-- Statement about the coefficient of the expression
theorem coeff_of_expr : coefficient expr = -2/5 := 
sorry

-- Statement about the highest degree term of the polynomial
theorem highest_degree_term_of_poly : highest_degree_term poly = -7 * x^4 * y^2 := 
sorry

end coeff_of_expr_highest_degree_term_of_poly_l779_779218


namespace compute_ratio_d_e_l779_779949

axiom quartic_eq_roots (a b c d e : ℝ) : (1, -1, 2, 3 : ℝ)

noncomputable def sum_products_three_at_a_time (a d : ℝ) : Prop := -5 = -d / a

noncomputable def product_of_roots (a e : ℝ) : Prop := -6 = e / a

theorem compute_ratio_d_e (a b c d e : ℝ) 
  (h1 : quartic_eq_roots a b c d e) 
  (h2 : sum_products_three_at_a_time a d) 
  (h3 : product_of_roots a e) : d / e = 5 / 6 :=
by
  sorry

end compute_ratio_d_e_l779_779949


namespace arithmetic_sequence_term_count_l779_779452

theorem arithmetic_sequence_term_count (a1 d an : ℤ) (h₀ : a1 = -6) (h₁ : d = 5) (h₂ : an = 59) :
  ∃ n : ℤ, an = a1 + (n - 1) * d ∧ n = 14 :=
by
  sorry

end arithmetic_sequence_term_count_l779_779452


namespace largest_square_area_l779_779757

-- Define the diameter of the circle
def diameter : ℝ := 8

-- Define the radius derived from the diameter
def radius : ℝ := diameter / 2

-- Define the area of the largest square inscribed in the circle
def area_of_largest_square_inscribed_in_circle (d : ℝ) : ℝ :=
  let a := (d / Math.sqrt 2) in
  a * a

-- Theorem stating the area of the largest square inscribed in a circle with a given diameter
theorem largest_square_area (d : ℝ) (h : d = 8) : area_of_largest_square_inscribed_in_circle d = 32 :=
by
  rw [h]
  sorry

end largest_square_area_l779_779757


namespace eldest_boy_age_l779_779583

-- Initial declarations based on conditions
def sister_has_age (s : ℕ) : Prop := s = 5
def youngest_boy_age_twice_sisters (b s : ℕ) : Prop := b = 2 * s
def boys_ages_proportional (ages : List ℕ) (x : ℕ) : Prop :=
  ages = [x, x + 2, x + 4, 2 * x - 3, 2 * x - 1, 2 * x + 1]
def average_age_18 (ages : List ℕ) : Prop :=
  List.length ages = 6 ∧ (List.sum ages) / 6 = 18

-- Main theorem statement
theorem eldest_boy_age :
  ∃ (x : ℕ), ∃ (ages : List ℕ), 
    sister_has_age 5 ∧ youngest_boy_age_twice_sisters (List.head ages) 5 ∧ boys_ages_proportional ages x ∧ average_age_18 ages ∧ List.last ages 21 := 
by
  sorry

end eldest_boy_age_l779_779583


namespace correct_train_process_l779_779257

-- Define each step involved in the train process
inductive Step
| buy_ticket
| wait_for_train
| check_ticket
| board_train
| repair_train

open Step

-- Define each condition as a list of steps
def process_a : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]
def process_b : List Step := [wait_for_train, buy_ticket, board_train, check_ticket]
def process_c : List Step := [buy_ticket, wait_for_train, board_train, check_ticket]
def process_d : List Step := [repair_train, buy_ticket, check_ticket, board_train]

-- Define the correct process
def correct_process : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]

-- The theorem to prove that process A is the correct representation
theorem correct_train_process : process_a = correct_process :=
by {
  sorry
}

end correct_train_process_l779_779257


namespace johns_total_amount_l779_779515

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l779_779515


namespace regular_price_of_tire_l779_779965

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 10 = 250) : x = 80 :=
sorry

end regular_price_of_tire_l779_779965


namespace line_passes_through_fixed_point_l779_779095

theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ (x y : ℝ), (m + 1) * x + (2 * m - 1) * y + m - 2 = 0 ∧ x = 1 ∧ y = -1 := 
by {
  use (1, -1),
  simp,
  ring,
  sorry,
}

end line_passes_through_fixed_point_l779_779095


namespace hyperbola_eccentricity_l779_779440

variables (a b c e : ℝ) (x y : ℝ)
variables (F1 F2 A B : ℝ × ℝ)

-- Definitions for hyperbola and points
def hyperbola := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1
def foci_distance := 2 * c
def angle_F1A60 := ∀ (θ : ℝ), θ = π / 3
def midpoint_A := ∀ (A B : ℝ × ℝ), A = (B.1 / 2, B.2 / 2)

-- Problem statement: Given the conditions, prove the eccentricity
theorem hyperbola_eccentricity : 
  (a > 0) → 
  (b > 0) → 
  (hyperbola x y) → 
  (F1 = (-c, 0)) → 
  (F2 = (c, 0)) → 
  (angle_F1A60 (atan2 (A.2 - F1.2) (A.1 - F1.1))) → 
  (midpoint_A A B) → 
  (e = c / a) → 
  e = 2 + sqrt 3 :=
by sorry

end hyperbola_eccentricity_l779_779440


namespace parallelogram_area_increase_l779_779341

theorem parallelogram_area_increase (b h : ℕ) :
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  (A2 - A1) * 100 / A1 = 300 :=
by
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  sorry

end parallelogram_area_increase_l779_779341


namespace exists_root_between_1_1_and_1_2_l779_779328

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_between_1_1_and_1_2 :
  ∃ x, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have pf1 : f 1.1 = -0.59 := by norm_num1
  have pf2 : f 1.2 = 0.84 := by norm_num1
  apply exists_between_of_sign_change pf1 pf2
  sorry

end exists_root_between_1_1_and_1_2_l779_779328


namespace find_a_range_of_f1_no_fixed_points_l779_779808

-- Define the quadratic function which is given as even
def f (a x : ℝ) : ℝ := (a + 1) * x ^ 2 + (a ^ 2 - 1) * x + 1

-- Define the condition that the function is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Prove that a == 1 given the quadratic function is even
theorem find_a (a : ℝ) (h : is_even (f a)) : a = 1 := by
  sorry

-- Given a = 1, define the new function
def f1 (x : ℝ) : ℝ := f 1 x

-- Define the range problem
theorem range_of_f1 (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) 2) : 
  1 ≤ f1 x ∧ f1 x ≤ 9 := by
  sorry

-- Define the fixed point problem
theorem no_fixed_points (x : ℝ) (h : f1 x = x) : false := by
  sorry

end find_a_range_of_f1_no_fixed_points_l779_779808


namespace find_m_values_l779_779022

theorem find_m_values :
  ∃ m : ℝ, (∀ (α β : ℝ), (3 * α^2 + m * α - 4 = 0 ∧ 3 * β^2 + m * β - 4 = 0) ∧ (α * β = -4 / 3) ∧ (α + β = -m / 3) ∧ (α * β = 2 * (α^3 + β^3))) ↔
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
sorry

end find_m_values_l779_779022


namespace total_money_paid_l779_779614

def total_employees := 450
def monthly_salary_per_employee := 2000
def fraction_remaining := (2 : ℝ) / 3

def remaining_employees := total_employees * fraction_remaining

theorem total_money_paid (h : remaining_employees = 300) :
  remaining_employees * monthly_salary_per_employee = 600000 := by
  -- Proof will be here
  sorry

end total_money_paid_l779_779614


namespace total_students_l779_779284

theorem total_students (N : ℕ)
    (h1 : (15 * 75) + (10 * 90) = N * 81) :
    N = 25 :=
by
  sorry

end total_students_l779_779284


namespace cube_edge_length_l779_779609

theorem cube_edge_length {e : ℝ} (h : 12 * e = 108) : e = 9 :=
by sorry

end cube_edge_length_l779_779609


namespace range_of_a_nonempty_intersection_range_of_a_subset_intersection_l779_779544

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) ≤ 0}

-- Define set B in terms of variable a
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Statement 1: Proving the range of a when A ∩ B ≠ ∅
theorem range_of_a_nonempty_intersection (a : ℝ) : (A ∩ B a ≠ ∅) → (-1 / 2 ≤ a ∧ a ≤ 2) :=
by
  sorry

-- Statement 2: Proving the range of a when A ∩ B = B
theorem range_of_a_subset_intersection (a : ℝ) : (A ∩ B a = B a) → (a ≥ 2 ∨ a ≤ -3) :=
by
  sorry

end range_of_a_nonempty_intersection_range_of_a_subset_intersection_l779_779544


namespace sequence_n_value_l779_779148

theorem sequence_n_value (n : ℤ) : (2 * n^2 - 3 = 125) → (n = 8) := 
by {
    sorry
}

end sequence_n_value_l779_779148


namespace abs_inequality_solution_l779_779034

theorem abs_inequality_solution {x : ℝ} (h : |x + 1| < 5) : -6 < x ∧ x < 4 :=
by
  sorry

end abs_inequality_solution_l779_779034


namespace num_true_props_is_one_l779_779319

-- Definitions for the propositions
def prop1 : Prop := ¬ (∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → xy ≠ 0)
def prop2 : Prop := ¬ ∀ a : Type, (a ∈ Square) → (a ∈ Rhombus)
def prop3 : Prop := ¬ ∀ a b c : ℝ, (a > b) → (αc^2 > bc^2)
def prop4 : Prop := ∀ m : ℝ, (m > 2) → (∀ x : ℝ, x^2 - 2x + m > 0)

-- Function to count true propositions
def num_true_props : ℕ :=
[ prop1, prop2, prop3, prop4 ].count (λ p, p)

-- Theorem stating that the count of true propositions is exactly 1
theorem num_true_props_is_one : num_true_props = 1 :=
by {
  sorry
}

end num_true_props_is_one_l779_779319


namespace number_of_factors_of_N_is_28_l779_779528

def N := 17^3 + 3 * 17^2 + 3 * 17 + 1

theorem number_of_factors_of_N_is_28 :
  ∃ k : ℕ, k = 28 ∧ (∀ d : ℕ, d ∣ N → (1 ≤ d) ∧ (N % d = 0 ∧ k = finset.card (finset.filter (λ x, x ∣ N) (finset.range (N+1))))) :=
sorry

end number_of_factors_of_N_is_28_l779_779528


namespace eval_f_at_neg_two_l779_779870

def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + (2/3) * x^3 + 3 * x^2 + 5 * x + 7 - 4 * x

theorem eval_f_at_neg_two : f (-2) = 1 := by
  sorry

end eval_f_at_neg_two_l779_779870


namespace area_of_ellipse_area_of_cartesian_leaf_l779_779360

section
  variable (a b : ℝ) (t : ℝ)

  -- Condition 1: Parametric equations of the ellipse
  def ellipse_x (t : ℝ) := a * cos t
  def ellipse_y (t : ℝ) := b * sin t

  -- Theorem 1: Area enclosed by the ellipse
  theorem area_of_ellipse : ∫ t in 0..2*real.pi, ellipse_x a b t * deriv (ellipse_y a b) t - ellipse_y a b t * deriv (ellipse_x a b) t = real.pi * a * b := by
    sorry
   
  -- Condition 2: Equation of the Cartesian leaf
  def cartesian_leaf (x y : ℝ) := x^3 + y^3 - 3 * a * x * y = 0

  variable (t : ℝ)

  -- Parametric form for the Cartesian leaf
  def leaf_x (a : ℝ) (t : ℝ) := 3 * a * t / (1 + t^3)
  def leaf_y (a : ℝ) (t : ℝ) := 3 * a * t^2 / (1 + t^3)

  -- Theorem 2: Area enclosed by the loop of the Cartesian leaf
  theorem area_of_cartesian_leaf :
    ∫ t in 0..real.inf, leaf_x a t * deriv (leaf_y a) t - leaf_y a t * deriv (leaf_x a) t = (3 / 4) * a^2 := by
    sorry
end

end area_of_ellipse_area_of_cartesian_leaf_l779_779360


namespace non_degenerate_triangle_l779_779049

noncomputable def positive_real (n : ℕ) (a : fin n → ℝ) : Prop :=
  ∀ i : fin n, 0 < a i

theorem non_degenerate_triangle (n : ℕ) (a : fin n → ℝ) :
  n ≥ 3 →
  positive_real n a →
  (∑ i, (a i) ^ 2) ^ 2 > (n - 1) * ∑ i, (a i) ^ 4 →
  ∀ (i j k : fin n), i ≠ j → j ≠ k → k ≠ i → 
  (a i < a j + a k ∧ a j < a k + a i ∧ a k < a i + a j) :=
begin
  intros hn hp hineq i j k hij hjk hki,
  sorry
end

end non_degenerate_triangle_l779_779049


namespace cricket_run_rate_l779_779643

noncomputable def requiredRunRate (target: ℕ) (initialRunRate: ℕ → ℤ) (runsInFirstOvers: ℕ) (remainingOvers: ℕ) : ℚ :=
  (target - (initialRunRate runsInFirstOvers)) / remainingOvers

theorem cricket_run_rate:
  let target := 252
  let initial_run_rate := (fun x => (32 : ℤ))
  let runs_in_first_overs := 10
  let remaining_overs := 40 in
  requiredRunRate target initial_run_rate runs_in_first_overs remaining_overs = 5.5 :=
by
  sorry

end cricket_run_rate_l779_779643


namespace torus_has_non_intersecting_paths_l779_779871

/-
  Define the problem: It is possible to construct nine non-intersecting paths 
  connecting each of the three houses to three wells on a toroidal planet.
-/

-- Define the surface types: Sphere and Torus
inductive SurfaceType
| sphere
| torus

-- Define the condition: it is impossible on a spherical surface
def impossible_on_sphere : Prop := 
  ∀ (H W : set Point), (card H = 3) → (card W = 3) → ¬ exists (paths : list (Path H W)), 
  (card paths = 9) ∧ (∀ p1 p2 ∈ paths, p1 ≠ p2 → non_intersecting p1 p2)

-- Define the problem statement on a toroidal surface
noncomputable def possible_on_torus : Prop :=
  ∃ (H W : set Point), (card H = 3) → (card W = 3) → exists (paths : list (Path H W)), 
  (card paths = 9) ∧ (∀ p1 p2 ∈ paths, p1 ≠ p2 → non_intersecting p1 p2)

-- The final theorem statement combining the conditions and question
theorem torus_has_non_intersecting_paths :
  impossible_on_sphere →
  possible_on_torus := 
sorry

end torus_has_non_intersecting_paths_l779_779871


namespace range_a_l779_779776

-- Define the conditions
variable (a : ℝ) (p q : Prop)
def condition1 : Prop := a > 0
def condition2 : Prop := (∀ x : ℝ, (f : ℝ → ℝ) (x) := a ^ x → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) -- p: y = a^x is monotonically increasing on ℝ
def condition3 : Prop := (∀ x : ℝ, a * x^2 - a * x + 1 > 0) -- q: ax^2 - ax + 1 > 0 for all x in ℝ
def p_false : Prop := ¬ condition2 a
def q_false : Prop := ¬ condition3 a
def either_p_or_q_true : Prop := condition2 a ∨ condition3 a

-- The statement to prove
theorem range_a (h1 : condition1 a) (h2 : p_false a) (h3 : q_false a) (h4 : either_p_or_q_true a) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
sorry

end range_a_l779_779776


namespace simplify_f_l779_779204

theorem simplify_f (α : ℝ) : 
  (sin (π - α) * cos (2 * π - α) * sin (-α + 3/2 * π)) / 
  (cos (-π - α) * cos (-α + 3/2 * π)) = -cos α :=
by 
  sorry

end simplify_f_l779_779204


namespace joan_seashells_correct_l779_779878

/-- Joan originally found 70 seashells -/
def joan_original_seashells : ℕ := 70

/-- Sam gave Joan 27 seashells -/
def seashells_given_by_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def joan_total_seashells : ℕ := joan_original_seashells + seashells_given_by_sam

theorem joan_seashells_correct : joan_total_seashells = 97 :=
by
  unfold joan_total_seashells
  unfold joan_original_seashells seashells_given_by_sam
  sorry

end joan_seashells_correct_l779_779878


namespace range_of_a_l779_779066

open Set

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : (A ∪ B a) = A ↔ a ∈ {a ∈ ℝ | a ≤ -4 ∨ (2 ≤ a ∧ a ≤ 3) ∨ a > 3} := 
by
  sorry

end range_of_a_l779_779066


namespace measure_angle_BAD_l779_779919

/-- 
Point D is on side AC of triangle ABC, ∠ABD = 15°, ∠DBC = 50°, and ∠ACB = 90°
Prove that the measure of angle ∠BAD is 25°.
-/
theorem measure_angle_BAD 
  (A B C D : Type) 
  (ABC : Triangle A B C) 
  (D_on_AC : Point D ∈ LineSegment A C) 
  (angle_ABD : Angle A B D = 15) 
  (angle_DBC : Angle D B C = 50) 
  (angle_ACB : Angle A C B = 90) : Angle B A D = 25 := 
by 
  sorry

end measure_angle_BAD_l779_779919


namespace sum_of_edges_l779_779693

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l779_779693


namespace travel_with_decreasing_ticket_prices_l779_779859

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ℕ × ℕ → ℕ)
  (distinct_prices : ∀ ⦃x y : ℕ × ℕ⦄, x ≠ y → train_prices x ≠ train_prices y)
  (symmetric_prices : ∀ x y : ℕ, train_prices (x, y) = train_prices (y, x)) :
  ∃ start_city : ℕ, ∃ route : List (ℕ × ℕ),
    length route = n-1 ∧ 
    (∀ i, i < route.length - 1 → train_prices (route.nth i).get_or_else (0,0) > train_prices (route.nth (i+1)).get_or_else (0,0)) :=
begin
  -- Proof goes here
  sorry
end

end travel_with_decreasing_ticket_prices_l779_779859


namespace inequality_proof_l779_779909

theorem inequality_proof
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 3) :
  (x + 3) / (y + z) + (y + 3) / (z + x) + (z + 3) / (x + y) + 3 ≥ 
  27 * ((sqrt x + sqrt y + sqrt z)^2 / (x + y + z)^3) := 
sorry

end inequality_proof_l779_779909


namespace determine_m_l779_779227

-- Constants and conditions
constant m : ℝ
constant α β : ℝ
axiom h_roots_eq : ∀ x, (x^2 - 4 * x + m = 0) ↔ (x - α) * (x - β) = 0
axiom h_ratio_3_1 : α = 3 * β

theorem determine_m (h_sum : α + β = 4) (h_prod : α * β = m) : m = 3 :=
by
  have h1 : (3 * β) + β = 4 := by rw [h_ratio_3_1, add_comm, add]
  have h2 : 4 * β = 4 := by { rw [← h1], linarith }
  have h3 : β = 1 := by { field_simp at h2, linarith }
  have h4 : α = 3 := by { rw [h_ratio_3_1, h3, mul_one] }
  have h5 : α * β = 3 * 1 := by { rw [h4, h3], ring }
  show m = 3, by { rw [← h_prod], exact h5 }

end determine_m_l779_779227


namespace travel_with_decreasing_ticket_prices_l779_779860

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ℕ × ℕ → ℕ)
  (distinct_prices : ∀ ⦃x y : ℕ × ℕ⦄, x ≠ y → train_prices x ≠ train_prices y)
  (symmetric_prices : ∀ x y : ℕ, train_prices (x, y) = train_prices (y, x)) :
  ∃ start_city : ℕ, ∃ route : List (ℕ × ℕ),
    length route = n-1 ∧ 
    (∀ i, i < route.length - 1 → train_prices (route.nth i).get_or_else (0,0) > train_prices (route.nth (i+1)).get_or_else (0,0)) :=
begin
  -- Proof goes here
  sorry
end

end travel_with_decreasing_ticket_prices_l779_779860


namespace intersection_point_l779_779012

noncomputable def line1 (x : ℚ) : ℚ := 3 * x
noncomputable def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_point : ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = -1/2 ∧ y = -3/2 :=
by
  -- skipping the actual proof steps
  sorry

end intersection_point_l779_779012


namespace exists_k_sum_diff_leq_one_l779_779651

theorem exists_k_sum_diff_leq_one 
  (n : ℕ) (h_n : n > 2) 
  (x : Fin n → ℝ)
  (h_abs_sum : abs (∑ i in Finset.range n, x i) > 1)
  (h_abs_x : ∀ i, abs (x i) ≤ 1) :
  ∃ k : ℕ, k < n ∧ abs (∑ i in Finset.range k, x i - ∑ i in Finset.Ico k n, x i) ≤ 1 :=
sorry

end exists_k_sum_diff_leq_one_l779_779651


namespace area_of_region_B_l779_779734

-- Given conditions
def region_B (z : ℂ) : Prop :=
  (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
  ∧
  (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
  0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1)

-- Theorem to be proved
theorem area_of_region_B : 
  (∫ z in {z : ℂ | region_B z}, 1) = 1875 - 312.5 * Real.pi :=
by
  sorry

end area_of_region_B_l779_779734


namespace cos_2theta_eq_one_l779_779067

theorem cos_2theta_eq_one 
  (θ : ℝ)
  (h : 2^(-5/2 + 3 * real.cos θ) + 1 = 2^(1/2 + 2 * real.cos θ)) :
  real.cos (2 * θ) = 1 :=
sorry

end cos_2theta_eq_one_l779_779067


namespace mean_height_correct_l779_779553

-- Define the heights of the players on the basketball team
def player_heights : List ℕ := [47, 49, 51, 53, 55, 55, 58, 60, 62, 63, 64, 65, 67, 71, 72, 73]

-- Calculate the total height
def total_height (heights : List ℕ) : ℕ := heights.foldl (+) 0

-- Calculate the mean height
def mean_height (heights : List ℕ) : Rat :=
  total_height heights / heights.length

theorem mean_height_correct : mean_height player_heights = 60.31 :=
by
  -- This is the proof that needs to be completed
  sorry

end mean_height_correct_l779_779553


namespace problem_1_problem_2_l779_779932

-- Problem 1 proof statement
theorem problem_1 (x : ℝ) (h : x = -1) : 
  (1 * (-x^2 + 5 * x) - (x - 3) - 4 * x) = 2 := by
  -- Placeholder for the proof
  sorry

-- Problem 2 proof statement
theorem problem_2 (m n : ℝ) (h_m : m = -1/2) (h_n : n = 1/3) : 
  (5 * (3 * m^2 * n - m * n^2) - (m * n^2 + 3 * m^2 * n)) = 4/3 := by
  -- Placeholder for the proof
  sorry

end problem_1_problem_2_l779_779932


namespace find_q_eq_51_l779_779158

theorem find_q_eq_51 (p q : ℝ) (h : (3:ℝ) * (1 + 4 * complex.I) * (1 - 4 * complex.I) = q * 1) : q = 51 :=
by sorry

end find_q_eq_51_l779_779158


namespace tangents_product_l779_779603

theorem tangents_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7) 
  (h2 : 2 * Real.sin (2 * (x - y)) = Real.sin (2 * x) * Real.sin (2 * y)) :
  Real.tan x * Real.tan y = -7/6 := 
sorry

end tangents_product_l779_779603


namespace sum_of_valid_as_l779_779842

theorem sum_of_valid_as :
  ∃ (a ∈ ℤ), (∀ x, (3*x - 1)/2 ≥ x + 1 ∧ 2*x - 4 < a) ∧
  (∃ y ∈ ℤ, (a*y - 3)/(y - 1) - 2 = (y - 5)/(1 - y)) ∧ a > 2 →
  ∑ a in {a | a ∈ ℤ ∧ (∀ x, (3*x - 1)/2 ≥ x + 1 ∧ 2*x - 4 < a) ∧ 
                   (∃ y ∈ ℤ, (a*y - 3)/(y - 1) - 2 = (y - 5)/(1 - y)) ∧ a > 2}, a = 7 :=
sorry

end sum_of_valid_as_l779_779842


namespace region_area_correct_l779_779368

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l779_779368


namespace range_of_g_l779_779746

noncomputable def g (x : ℝ) : ℝ :=
  (cos x)^3 + 7 * (cos x)^2 - 2 * cos x + 3 * (sin x)^2 - 12

theorem range_of_g :
  ∀ x : ℝ, cos x ≠ 1 → 5 ≤ g x ∧ g x < 9 :=
by
  intro x hx
  have h : sin x ^ 2 = 1 - cos x ^ 2 := by sorry -- Identity for sin^2
  rw [g, h]
  -- transformations and factoring steps would go here
  sorry -- rest of the proof

end range_of_g_l779_779746


namespace nancy_earns_more_l779_779189

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l779_779189


namespace constant_term_eq_70_l779_779219

theorem constant_term_eq_70 (n : ℕ) (h : ∑ i in range (2 * n + 1), (choose (2 * n) i) * (-1)^i * x^(2 * n - 2 * i) = 70) : n = 4 :=
by
  -- Proof omitted
  sorry

end constant_term_eq_70_l779_779219


namespace find_m_l779_779399

-- Define the vectors a and b and the condition for parallelicity
def a : ℝ × ℝ := (2, 1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)
def parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1

-- State the theorem with the given conditions and required proof goal
theorem find_m (m : ℝ) (h : parallel a (b m)) : m = 4 :=
by sorry  -- skipping proof

end find_m_l779_779399


namespace area_of_abs_sum_eq_six_l779_779370

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l779_779370


namespace tangent_circle_circumference_l779_779459

theorem tangent_circle_circumference (A B C : Type*)
  (r1 r2 : ℝ) (h_eq1 : ∠ABC = ∠BCA = ∠CAB = π / 3)
  (h_eq2 : (1 / 3) * 2 * π * r1 = 12) :
  2 * π * r2 = 27 :=
by sorry

end tangent_circle_circumference_l779_779459


namespace tan_alpha_half_l779_779396

-- Define the trigonometric condition
def trig_condition (α : ℝ) : Prop :=
  (sin α - 2 * cos α) / (sin α + cos α) = -1

-- State the theorem to prove
theorem tan_alpha_half (α : ℝ) (h : trig_condition α) : tan α = 1 / 2 :=
  sorry

end tan_alpha_half_l779_779396


namespace distinct_solutions_difference_l779_779531

theorem distinct_solutions_difference (p q : ℝ) (h1 : (x : ℝ) x^2 - 26 * x + 105 = 0 → x = p ∨ x = q) (h2 : p > q) : p - q = 16 :=
sorry

end distinct_solutions_difference_l779_779531


namespace sin_alpha_plus_beta_l779_779045

theorem sin_alpha_plus_beta (α β : ℝ) (hα_cos : cos (π / 4 - α) = 3 / 5) 
  (hβ_sin : sin (π / 4 + β) = 12 / 13) 
  (hα_range : π / 4 < α ∧ α < 3 * π / 4) 
  (hβ_range : 0 < β ∧ β < π / 4) : 
  sin (α + β) = 56 / 65 := 
sorry

end sin_alpha_plus_beta_l779_779045


namespace possible_third_side_lengths_l779_779125

theorem possible_third_side_lengths :
  {x : ℕ // 2 < x ∧ x < 14} = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13} := by
  sorry

end possible_third_side_lengths_l779_779125


namespace andrew_eggs_count_l779_779324

def cost_of_toast (num_toasts : ℕ) : ℕ :=
  num_toasts * 1

def cost_of_eggs (num_eggs : ℕ) : ℕ :=
  num_eggs * 3

def total_cost (num_toasts : ℕ) (num_eggs : ℕ) : ℕ :=
  cost_of_toast num_toasts + cost_of_eggs num_eggs

theorem andrew_eggs_count (E : ℕ) (H1 : total_cost 2 2 = 8)
                       (H2 : total_cost 1 E + 8 = 15) : E = 2 := by
  sorry

end andrew_eggs_count_l779_779324


namespace equation_of_plane_l779_779994

def A : ℝ × ℝ × ℝ := (1, 0, -6)
def B : ℝ × ℝ × ℝ := (-7, 2, 1)
def C : ℝ × ℝ × ℝ := (-9, 6, 1)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def BC : ℝ × ℝ × ℝ := vector_sub C B

def plane_eq (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : ℝ → ℝ → ℝ → ℝ :=
  λ x y z, n.1 * (x - p.1) + n.2 * (y - p.2) + n.3 * (z - p.3)

theorem equation_of_plane : ∀ x y z,
  plane_eq BC A x y z = 0 ↔ x - 2*y - 1 = 0 := by
  sorry

end equation_of_plane_l779_779994


namespace nancy_more_money_l779_779190

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l779_779190


namespace point_P_on_AC_l779_779868

-- Define the conditions
variable (A B C D E F G H P : Point)
variable [AffineSpace.Point Point Space]
variable {AB BC CD DA AC BD EF GH line: Segment}

-- Assume that ABCD is a space quadrilateral with the given point placements
axiom E_on_AB : On E AB
axiom F_on_BC : On F BC
axiom G_on_CD : On G CD
axiom H_on_DA : On H DA

-- Assume that EF and GH intersect at point P
axiom EF_contains_P : On P EF
axiom GH_contains_P : On P GH

-- Planes definition
def PlaneABC : Plane := Plane.mk A B C
def PlaneADC : Plane := Plane.mk A D C

-- Axiom stating the intersection of Plane ABC and Plane ADC is line AC
axiom Intersection_PlaneABC_PlaneADC : PlaneABC ∩ PlaneADC = AC

-- The proof goal is to show that P is on line AC
theorem point_P_on_AC : On P AC :=
sorry

end point_P_on_AC_l779_779868


namespace ellipse_equation_circle_through_fixed_points_l779_779056

-- Define the ellipse with given conditions
def is_ellipse (C : Type*) (origin : C) (on_x_axis : bool) 
  (left_vertex_A : C) (left_focus_F1 : C) : Prop :=
  let center := (0 : C, 0 : C) in 
  on_x_axis ∧ left_vertex_A = (-2 * √2, 0) ∧ left_focus_F1 = (-2, 0)

-- Define a point on the ellipse
def point_on_ellipse (C : Type*) (B : C) : Prop :=
  B = (2, √2)

-- Define the intersection of a line with the ellipse
def intersects_ellipse (C : Type*) (kx_line : C × C → C) (E F : C) : Prop :=
  ∃ x₀ y₀, kx_line (x₀, y₀) ∧ E = (x₀, y₀) ∧ F = (-x₀, -y₀)

-- Define the conditions for the ellipse
variables {C : Type*} 
variable [add_comm_group C] [module ℝ C]

-- Problem (1): Prove the equation of the ellipse
theorem ellipse_equation (center: C) (focus_F1: C) (point_B: C) (A: C) :
  is_ellipse C center true A focus_F1 →
  point_on_ellipse C point_B →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b ∧ (2 * a = 2 * √2 ∧ 2 * (a^2 - b^2) = 4) ∧ 
  (x^2) / (8) + (y^2) / (4) = 1)

-- Problem (2): Prove the circle with diameter MN passes through fixed points
theorem circle_through_fixed_points (kx_line : C × C → C) (A E F M N P1 P2 : C) :
  intersects_ellipse C kx_line E F →
  let M_intersection := ∃ y₀, y₀ = (2 * √2 * k) / (1 + √(1 + 2 * k^2)) in 
  let N_intersection := ∃ y₀, y₀ = (2 * √2 * k) / (1 - √(1 + 2 * k^2)) in 
  (0, -√2 / k) = P1 → (2, 0) = P2 → true

end ellipse_equation_circle_through_fixed_points_l779_779056


namespace smallest_pos_int_area_gt_2500_l779_779730

theorem smallest_pos_int_area_gt_2500 :
  ∃ n : ℕ, (0 < n) ∧ (2 * (abs ((n * (4 * n) + (n^2 - 4) * (6 * n^2 - 8) + (n^3 - 12 * n) * 2 - 2 * (n^2 - 4) - 4 * n * (n^3 - 12 * n) - (6 * n^2 - 8) * n))) / 2 > 2500) ∧ (∀ m : ℕ, 0 < m ∧ m < n -> 2 * (abs ((m * (4 * m) + (m^2 - 4) * (6 * m^2 - 8) + (m^3 - 12 * m) * 2 - 2 * (m^2 - 4) - 4 * m * (m^3 - 12 * m) - (6 * m^2 - 8) * m))) / 2 ≤ 2500) :=
  sorry

end smallest_pos_int_area_gt_2500_l779_779730


namespace solution_set_of_inequality_l779_779422

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_odd : odd_function f) (h_decreasing : decreasing f) :
  { a : ℝ | f (a^2) + f (2 * a) > 0 } = set.Ioo (-2) 0 :=
by
  sorry

end solution_set_of_inequality_l779_779422


namespace distinct_partition_bijection_odd_partition_l779_779345

-- A definition of what it means for a list to be a partition of n into distinct parts
def is_distinct_partition (n : ℕ) (l : List ℕ) : Prop :=
  l.sum = n ∧ l.nodup ∧ (∀ x ∈ l, x > 0)

-- A definition of what it means for a list to be a partition of n into odd parts
def is_odd_partition (n : ℕ) (l : List ℕ) : Prop :=
  l.sum = n ∧ (∀ x ∈ l, x % 2 = 1)

-- The theorem statement showing the bijection between these two types of partitions
theorem distinct_partition_bijection_odd_partition (n : ℕ) :
  ∃ (f : List ℕ → List ℕ) (g : List ℕ → List ℕ),
    (∀ (l : List ℕ), is_distinct_partition n l → is_odd_partition n (f l)) ∧
    (∀ (l : List ℕ), is_odd_partition n l → is_distinct_partition n (g l)) ∧
    (∀ (l : List ℕ), is_distinct_partition n l → g (f l) = l) ∧
    (∀ (l : List ℕ), is_odd_partition n l → f (g l) = l) :=
sorry

end distinct_partition_bijection_odd_partition_l779_779345


namespace sachin_younger_than_rahul_l779_779930

theorem sachin_younger_than_rahul :
  ∀ (sachin_age rahul_age : ℕ),
  (sachin_age / rahul_age = 6 / 9) →
  (sachin_age = 14) →
  (rahul_age - sachin_age = 7) :=
by
  sorry

end sachin_younger_than_rahul_l779_779930


namespace four_digit_numbers_condition_l779_779113

theorem four_digit_numbers_condition :
  ∃ (N : Nat), (1000 ≤ N ∧ N < 10000) ∧
               (∃ x a : Nat, N = 1000 * a + x ∧ x = 200 * a ∧ 1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end four_digit_numbers_condition_l779_779113


namespace value_of_d_l779_779130

theorem value_of_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (4 * y) / 20 + (3 * y) / d = 0.5 * y) : d = 10 :=
by
  sorry

end value_of_d_l779_779130


namespace odd_function_inequality_l779_779902

noncomputable def f (x : ℝ) : ℝ := sorry

theorem odd_function_inequality :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x > 0 → f(x) + x * (deriv f x) > 0) →
    -2 * f (-2) < -real.exp 1 * f (-real.exp 1) ∧ -real.exp 1 * f (-real.exp 1) < 3 * f 3 :=
by
  intro odd_f deriv_pos
  sorry

end odd_function_inequality_l779_779902


namespace rose_bushes_around_patio_l779_779463

theorem rose_bushes_around_patio : 
    let radius := 18
    let spacing := 1.5
    let pi_approx := 3.14159
    let circumference := 2 * pi * radius
    let num_bushes := circumference / spacing
    let approx_bushes := Real.ceil (num_bushes * pi_approx)
in approx_bushes = 75 := 
by
  sorry

end rose_bushes_around_patio_l779_779463


namespace triangles_in_circle_l779_779625

theorem triangles_in_circle : 
  ∀ (points : Finset ℕ), points.card = 12 →
  (∀ (p1 p2 p3 : ℕ), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  (¬ (∃ q1 q2 q3 : ℕ, q1 ≠ q2 → q2 ≠ q3 → q1 ≠ q3 → q1 ∈ points → q2 ∈ points → q3 ∈ points → 
  q1 ≠ p1 → q2 ≠ p2 → q3 ≠ p3 → 
  (q1, q2) and (q2, q3) and (q1, q3)) → False)) →
  sorry = 5775 := 
by sorry

end triangles_in_circle_l779_779625


namespace order_of_activities_l779_779607

noncomputable def fraction_liking_activity_dodgeball : ℚ := 8 / 24
noncomputable def fraction_liking_activity_barbecue : ℚ := 10 / 30
noncomputable def fraction_liking_activity_archery : ℚ := 9 / 18

theorem order_of_activities :
  (fraction_liking_activity_archery > fraction_liking_activity_dodgeball) ∧
  (fraction_liking_activity_archery > fraction_liking_activity_barbecue) ∧
  (fraction_liking_activity_dodgeball = fraction_liking_activity_barbecue) :=
by
  sorry

end order_of_activities_l779_779607


namespace problem_l779_779172

def a_n (n : ℕ) : ℝ := 2 * n + 1
def S (n : ℕ) : ℝ := n / 2 * (2 * 3 + (n - 1) * 2)
def b_n (n : ℕ) : ℝ := 1 / ((2 * n + 1) * (2 * n + 3))
def T (k : ℕ) : ℝ := (1 / 2) * (1 / 3 - 1 / (2 * k + 3))

theorem problem {k : ℕ} :
  a_n 5 = 11 ∧ S 10 = 120 ∧ T k = 8 / 57 → k = 8 :=
by
  -- Here goes the proof, but it's skipped.
  sorry

end problem_l779_779172


namespace rectangular_solid_edges_sum_l779_779695

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l779_779695


namespace total_value_of_gold_l779_779883

theorem total_value_of_gold (legacy_bars : ℕ) (aleena_bars : ℕ) (bar_value : ℕ) (total_gold_value : ℕ) 
  (h1 : legacy_bars = 12) 
  (h2 : aleena_bars = legacy_bars - 4)
  (h3 : bar_value = 3500) : 
  total_gold_value = (legacy_bars + aleena_bars) * bar_value := 
by 
  sorry

end total_value_of_gold_l779_779883


namespace simplify_radicals_l779_779570

theorem simplify_radicals : 
  sqrt (10 + 6 * sqrt 2) + sqrt (10 - 6 * sqrt 2) = 2 * sqrt 6 := 
  sorry

end simplify_radicals_l779_779570


namespace only_solution_is_neg_id_l779_779753

noncomputable def f : ℝ → ℝ := sorry

theorem only_solution_is_neg_id (f : ℝ → ℝ) (monotone_f : monotone f) (n0 : ℕ) (hn0 : n0 ≥ 0)
  (hfn0 : ∀ x : ℝ, (function.iterate f n0 x) = -x) : (∀ x : ℝ, f x = -x) :=
sorry

end only_solution_is_neg_id_l779_779753


namespace general_term_a_n_general_term_b_n_max_k_value_l779_779784

noncomputable def S_n (n : ℕ) : ℚ := (1 / 2) * n^2 + (11 / 2) * n

def a_n (n : ℕ) : ℚ := n + 5

def b : ℕ → ℚ
| 0     := 0
| 1     := 0
| 2     := 0
| 3     := 11
| (n+4) := 2 * b (n+2) - b (n)

def b_n (n : ℕ) : ℚ := 3 * n + 2

def c_n (n : ℕ) : ℚ := 3 / ((2 * a_n n - 11) * (2 * b_n n - 1))

noncomputable def T_n (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (2 * n + 1))

theorem general_term_a_n (n : ℕ) : a_n n = n + 5 :=
sorry

theorem general_term_b_n (n : ℕ) : b_n n = 3 * n + 2 :=
sorry

theorem max_k_value (k : ℕ) (hk: 18 < k) : ∀ n, T_n n > k / 57 :=
sorry

end general_term_a_n_general_term_b_n_max_k_value_l779_779784


namespace exists_largest_integer_in_range_l779_779378

theorem exists_largest_integer_in_range :
  ∃ x : ℤ, (1 / 4 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (2 / 3 : ℚ) ∧
           ∀ y : ℤ, (1 / 4 : ℚ) < (y / 7 : ℚ) ∧ (y / 7 : ℚ) < (2 / 3 : ℚ) → y ≤ x :=
begin
  use 4,
  split,
  { norm_num,
    linarith, },
  split,
  { norm_num,
    linarith, },
  intros y hy,
  cases hy with h₁ h₂,
  linarith,
end

end exists_largest_integer_in_range_l779_779378


namespace sum_mod_remainder_l779_779381

theorem sum_mod_remainder :
  (∑ i in Finset.range 10, 1001 + i) % 7 = 5 :=
by
  sorry

end sum_mod_remainder_l779_779381


namespace avg10_students_correct_l779_779458

-- Definitions for the conditions
def avg15_students : ℝ := 70
def num15_students : ℕ := 15
def num10_students : ℕ := 10
def num25_students : ℕ := num15_students + num10_students
def avg25_students : ℝ := 80

-- Total percentage calculation based on conditions
def total_perc25_students := num25_students * avg25_students
def total_perc15_students := num15_students * avg15_students

-- The average percent of the 10 students, based on the conditions and given average for 25 students.
theorem avg10_students_correct : 
  (total_perc25_students - total_perc15_students) / (num10_students : ℝ) = 95 := by
  sorry

end avg10_students_correct_l779_779458


namespace same_standard_deviation_l779_779134

-- Definitions of the data samples
def sampleA : List ℝ := [52, 54, 54, 56, 56, 56, 55, 55, 55, 55]
def sampleB : List ℝ := sampleA.map (λ x => x + 6)

-- Define the standard deviation function
def standard_deviation (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  let variance := (l.map (λ x => (x - mean)^2)).sum / l.length
  real.sqrt variance

-- The theorem we need to prove
theorem same_standard_deviation :
  standard_deviation sampleA = standard_deviation sampleB := 
sorry

end same_standard_deviation_l779_779134


namespace union_of_M_N_eq_set123_l779_779892

-- Define the sets M and N
def M (a : ℕ) : set ℕ := {3, 2^a}
def N (a b : ℕ) : set ℕ := {a, b}

-- Main theorem to state the problem and required to prove
theorem union_of_M_N_eq_set123 (a b: ℕ) (h: M a ∩ N a b = {2}) :
  M a ∪ N a b = {1, 2, 3} :=
begin
  sorry
end

end union_of_M_N_eq_set123_l779_779892


namespace factor_x_minus_1_l779_779076

-- Define the polynomials and the given condition
variables {R : Type*} [CommRing R]
variables (P Q R S : R[X])

-- Given condition: P(x^5) + x * Q(x^5) + x^2 * R(x^5) = (x^4 + x^3 + x^2 + x + 1) * S(x)
def given_condition : Prop := 
  ∀ x : R, P (x^5) + x * Q (x^5) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x

-- The statement we need to prove
theorem factor_x_minus_1 (h : given_condition P Q R S) : (X - C 1) ∣ P :=
sorry

end factor_x_minus_1_l779_779076


namespace remainder_t100_mod_7_l779_779955

theorem remainder_t100_mod_7 :
  ∀ T : ℕ → ℕ, (T 1 = 3) →
  (∀ n : ℕ, n > 1 → T n = 3 ^ (T (n - 1))) →
  (T 100 % 7 = 6) :=
by
  intro T h1 h2
  -- sorry to skip the actual proof
  sorry

end remainder_t100_mod_7_l779_779955


namespace triangle_area_zero_collinear_l779_779377

def u : ℝ³ := ⟨2, 3, 1⟩
def v : ℝ³ := ⟨8, 6, 4⟩
def w : ℝ³ := ⟨14, 9, 7⟩

theorem triangle_area_zero_collinear : 
  ∃ (a b c : ℝ³), a = u ∧ b = v ∧ c = w ∧ 
  (∃ k: ℝ, c - a = k • (b - a)) → 
  let T := {p : ℝ³ | ∃ (x y z : ℝ), x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
                            ∧ p = x • a + y • b + z • c} in 
  ∃ (O : ℝ), O = 0 ∧ ∀ (p : ℝ³), p ∈ T → p = a ∨ p = b ∨ p = c := sorry

end triangle_area_zero_collinear_l779_779377


namespace ice_melts_volume_decrease_l779_779289

theorem ice_melts_volume_decrease (V_w : ℝ) (h : V_w = 1) : 
  let V_i := V_w * (1 + 1 / 11) in 
  ((V_i - V_w) / V_i) = 1 / 12 :=
by
  sorry

end ice_melts_volume_decrease_l779_779289


namespace max_value_of_2sinx_max_value_of_2sinx_is_2_l779_779596

theorem max_value_of_2sinx : ∀ x : ℝ, 2 * (sin x) ≤ 2 ∧ 2 * (sin x) ≥ -2 := 
by
  sorry

theorem max_value_of_2sinx_is_2 : ∃ x : ℝ, 2 * (sin x) = 2 := 
by
  sorry

end max_value_of_2sinx_max_value_of_2sinx_is_2_l779_779596


namespace inequality_proof_l779_779406

theorem inequality_proof 
  (a b c d : ℝ) (n : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_n : 9 ≤ n) :
  a^n + b^n + c^n + d^n ≥ a^(n-9)*b^4*c^3*d^2 + b^(n-9)*c^4*d^3*a^2 + c^(n-9)*d^4*a^3*b^2 + d^(n-9)*a^4*b^3*c^2 :=
by
  sorry

end inequality_proof_l779_779406


namespace exists_travel_route_l779_779856

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l779_779856


namespace additional_time_due_to_leak_l779_779111

theorem additional_time_due_to_leak (fill_time_no_leak: ℝ) (leak_empty_time: ℝ) (fill_rate_no_leak: fill_time_no_leak ≠ 0):
  (fill_time_no_leak = 3) → 
  (leak_empty_time = 12) → 
  (1 / fill_time_no_leak - 1 / leak_empty_time ≠ 0) → 
  ((1 / fill_time_no_leak - 1 / leak_empty_time) / (1 / (1 / fill_time_no_leak - 1 / leak_empty_time)) - fill_time_no_leak = 1) := 
by
  intro h_fill h_leak h_effective_rate
  sorry

end additional_time_due_to_leak_l779_779111


namespace cyclic_quad_decomposition_l779_779562

theorem cyclic_quad_decomposition (n : ℕ) (h : n ≥ 4) (Q : Type) [quadrilateral Q] [cyclic Q] :
  ∃ (quads : list Q), quads.length = n ∧ ∀ (q ∈ quads), cyclic q :=
sorry

end cyclic_quad_decomposition_l779_779562


namespace f_monotonically_decreasing_interval_g_value_l779_779439

noncomputable def f (x : ℝ) : ℝ := 2 * Math.sin ( (Real.pi / 3) * x - (Real.pi / 6) )

-- (I) Prove the monotonically decreasing interval
theorem f_monotonically_decreasing_interval (k : ℤ) : 
  ∀ x : ℝ, 6 * k + 2 ≤ x ∧ x ≤ 6 * k + 5 ↔ 
  Real.deriv f x ≤ 0 := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * Math.sin ( (Real.pi / 3) * (x - 1/2) - (Real.pi / 6) )

-- (II) Prove the relationship for g(x₁ - x₂)
theorem g_value (x₁ x₂ : ℝ) (h1 : x₁ ∈ Set.Ioo (-3 : ℝ) (-2))
  (h2 : x₂ ∈ Set.Ioo (0 : ℝ) 1) (h3 : g x₁ + g x₂ = 0) : 
  g (x₁ - x₂) = Real.sqrt 3 :=
sorry

end f_monotonically_decreasing_interval_g_value_l779_779439


namespace rate_per_kg_grapes_l779_779822

/-- Define the conditions for the problem -/
def rate_per_kg_mangoes : ℕ := 55
def kg_grapes_purchased : ℕ := 3
def kg_mangoes_purchased : ℕ := 9
def total_paid : ℕ := 705

/-- The theorem statement to prove the rate per kg for grapes -/
theorem rate_per_kg_grapes (G : ℕ) :
  kg_grapes_purchased * G + kg_mangoes_purchased * rate_per_kg_mangoes = total_paid →
  G = 70 :=
by
  sorry -- Proof will go here

end rate_per_kg_grapes_l779_779822


namespace find_X_l779_779223

-- Definition: Set F
def F (X : ℝ) := {-4, -1, 0, 6, X}

-- Definition: Mean of the set F
def mean_F (X : ℝ) := (1 + X) / 5

-- Definition: Replaced set G with primes 2 and 3
def G (X : ℝ) := {2, 3, 0, 6, X}

-- Definition: Mean of the set G
def mean_G (X : ℝ) := (11 + X) / 5

-- Condition: The minimum percentage increase in the mean of set F is 100% 
-- if its two smallest elements are replaced with two different primes.
theorem find_X (X : ℝ) : (11 + X) / 5 ≥ 2 * (1 + X) / 5 ↔ X = 9 := by
  sorry

end find_X_l779_779223


namespace inequality_solution_l779_779766

theorem inequality_solution {x : ℝ} (hx_pos : x > -1) (hx_ne_zero : x ≠ 0) :
  (x^2 / (x + 1 - sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2) ↔ (x ∈ Ioo (-1 : ℝ) 0 ∪ Ioo 0 3) :=
sorry

end inequality_solution_l779_779766


namespace negatively_correlated_variables_l779_779262

theorem negatively_correlated_variables :
  (B: Prop) :=
begin
  let A := ¬negatively_correlated (weight_of_car, fuel_consumption_per_100_km),
  let B := negatively_correlated (weight_of_car, average_distance_per_liter_fuel),
  let C := ¬negatively_correlated (global_CO2_emissions, earth_temperature_increase),
  let D := ¬negatively_correlated (product_sales_revenue, advertising_expenditure),
  exact B,
sorry

end negatively_correlated_variables_l779_779262


namespace find_f_of_2_l779_779281

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^3 + x^2 else 0

theorem find_f_of_2 :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x < 0 → f x = x^3 + x^2) → f 2 = 4 :=
by
  intros h_odd h_def_neg
  sorry

end find_f_of_2_l779_779281


namespace not_kth_power_l779_779925

theorem not_kth_power (m k : ℕ) (hk : k > 1) : ¬ ∃ a : ℤ, m * (m + 1) = a^k :=
by
  sorry

end not_kth_power_l779_779925


namespace area_BCM_l779_779929

variable {α : Type*} [LinearOrderedField α]

structure RightTriangle (A B C : α) : Prop :=
(hypotenuse : α)
(leg : α)
(length_AB : A - B = hypotenuse)
(length_BC : B - C = leg)
(hypotenuse_length : hypotenuse = 6)
(leg_length : leg = 4.5)
(right_angle : true) -- For simplicity, just an assertion of the right angle

structure Midpoint (M A B : α) : Prop :=
(midpoint : M = (A + B) / 2)
(midpoint_exists : true) -- For simplicity, just an assertion that it exists

structure PointOnSegment (C A B : α) : Prop :=
(one_third_distance : C - A = (B - A) / 3)

noncomputable def area_of_bcm {A B C M : α}
  (h1 : RightTriangle A B C)
  (h2 : Midpoint M A B)
  (h3 : PointOnSegment C A B) : α :=
(1 / 2) * abs (B - C) * abs ((A + B) / 2 - C)

theorem area_BCM
  {A B C M : α}
  (h1 : RightTriangle A B C)
  (h2 : Midpoint M A B)
  (h3 : PointOnSegment C A B) :
  area_of_bcm h1 h2 h3 = 2.25 := by sorry

end area_BCM_l779_779929


namespace largest_product_sum_1976_l779_779743

theorem largest_product_sum_1976 (a : ℕ → ℕ) (h : ∑ i in finset.range 1976, a i = 1976) : 
  (∏ i in finset.range 1976, a i) ≤ 2 * 3 ^ 658 :=
sorry

end largest_product_sum_1976_l779_779743


namespace distance_between_skew_lines_l779_779465

theorem distance_between_skew_lines
  (a b : Line) -- lines a and b
  (θ α β m : ℝ) -- angles and length
  (hθ_pos : 0 < θ) (hθ_lt_pi : θ < π) -- conditions on θ
  (A' E : Point on a) (A F : Point on b) -- points on lines
  (h1 : angle E A' A = α) -- ∠ EA'A = α
  (h2 : angle A' A F = β) -- ∠ A'AF = β
  (h3 : distance A' A = m) -- A'A = m
  : distance_between_lines a b = 
      (m / sin θ) * sqrt (1 - cos θ ^ 2 - cos α ^ 2 - cos β ^ 2 - 2 * (cos α * cos β * cos θ)) := by
  sorry

end distance_between_skew_lines_l779_779465


namespace rectangular_eq_of_C_slope_of_l_l779_779147

noncomputable section

/-- Parametric equations for curve C -/
def parametric_eq (θ : ℝ) : ℝ × ℝ :=
⟨4 * Real.cos θ, 3 * Real.sin θ⟩

/-- Question 1: Prove that the rectangular coordinate equation of curve C is (x^2)/16 + (y^2)/9 = 1. -/
theorem rectangular_eq_of_C (x y θ : ℝ) (h₁ : x = 4 * Real.cos θ) (h₂ : y = 3 * Real.sin θ) : 
  x^2 / 16 + y^2 / 9 = 1 := 
sorry

/-- Line passing through point M(2, 2) with parametric equations -/
def line_through_M (t α : ℝ) : ℝ × ℝ :=
⟨2 + t * Real.cos α, 2 + t * Real.sin α⟩ 

/-- Question 2: Prove that the slope of line l passing M(2, 2) which intersects curve C at points A and B is -9/16 -/
theorem slope_of_l (t₁ t₂ α : ℝ) (t₁_t₂_sum_zero : (9 * Real.sin α + 36 * Real.cos α) = 0) :
  Real.tan α = -9 / 16 :=
sorry

end rectangular_eq_of_C_slope_of_l_l779_779147


namespace John_total_distance_l779_779879

-- Define the time and speeds for each segment
def segment1_time := 2
def segment1_speed := 45

def segment2_time := 3
def segment2_speed := 55

def segment3_time := 1.5
def segment3_speed := 60

def segment4_time := 2.5
def segment4_speed := 50

-- Define the distances for each segment
def segment1_distance := segment1_speed * segment1_time
def segment2_distance := segment2_speed * segment2_time
def segment3_distance := segment3_speed * segment3_time
def segment4_distance := segment4_speed * segment4_time

-- Define the total distance
def total_distance := segment1_distance + segment2_distance + segment3_distance + segment4_distance

-- Prove that the total distance is 470 miles
theorem John_total_distance : total_distance = 470 := by
  unfold total_distance
  unfold segment1_distance
  unfold segment2_distance
  unfold segment3_distance
  unfold segment4_distance
  unfold segment1_time
  unfold segment1_speed
  unfold segment2_time
  unfold segment2_speed
  unfold segment3_time
  unfold segment3_speed
  unfold segment4_time
  unfold segment4_speed
  sorry

end John_total_distance_l779_779879


namespace EF_parallel_BC_l779_779468

-- Define the geometric entities and conditions
variables {A B C I T J F S E : Point}

-- Assumptions
axiom h1 : incenter I (triangle A B C)
axiom h2 : intersects (line A I) (line B C) T
axiom h3 : touchpoint_excircle BC A J
axiom h4 : second_intersection (circumcircle A J T) (circumcircle A B C) F
axiom h5 : perpendicular (line I S) (line A T)
axiom h6 : meets (line I S) (line B C) S
axiom h7 : second_intersection (line A S) (circumcircle A B C) E

-- Goal
theorem EF_parallel_BC : parallel (line E F) (line B C) :=
sorry

end EF_parallel_BC_l779_779468


namespace area_of_abs_inequality_l779_779373

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l779_779373


namespace annual_interest_rate_l779_779301

theorem annual_interest_rate (r : ℝ): 
  (1000 * r * 4.861111111111111 + 1400 * r * 4.861111111111111 = 350) → 
  r = 0.03 :=
sorry

end annual_interest_rate_l779_779301


namespace logarithmic_inequality_l779_779804

open Real

theorem logarithmic_inequality
  (a b c : ℝ)
  (ha : 1 < a)
  (hb : 1 < b)
  (h : log a c * log b c = 4) :
  a * b ≥ c :=
begin
  sorry
end

end logarithmic_inequality_l779_779804


namespace largest_root_l779_779621

theorem largest_root (a b c : ℝ) 
  (h₁ : a + b + c = 3)
  (h₂ : a * b + a * c + b * c = -10)
  (h₃ : a * b * c = -18) :
  max a (max b c) = -1 + real.sqrt 7 := sorry

end largest_root_l779_779621


namespace ratio_PN_PR_l779_779917

variable {P Q R N L A : Type}
variable {m n : ℕ}
variable {triangle : Triangle P Q R}
variable {QN_eq_LR : length (NQ : Segment Q N) = length (LR : Segment L R)}
variable {A_intersects_QL_NR : Intersection QL NR = A}
variable {A_divides_QL_ratio_m_n : divides_ratio A QL m n}

theorem ratio_PN_PR (h1 : ∃ (N : Point P Q), ∃ (L : Point P R), length (Segment N Q) = length (Segment L R))
  (h2 : ∃ (A : Point Q L), A ∈ intersection (Segment Q L) (Segment N R) ∧ divides_ratio (Segment Q L) A m n) :
  (length (Segment P N) / length (Segment P R)) = (n / m) :=
by
  -- To be proved
  sorry

end ratio_PN_PR_l779_779917


namespace gum_boxes_l779_779296

theorem gum_boxes (c s t g : ℕ) (h1 : c = 2) (h2 : s = 5) (h3 : t = 9) (h4 : c + s + g = t) : g = 2 := by
  sorry

end gum_boxes_l779_779296


namespace rectangle_area_EFGH_l779_779865

open Real

noncomputable def point := ℝ × ℝ

noncomputable def E : point := (1, 1)
noncomputable def F : point := (101, 21)
noncomputable def H : point := (3, -9)

def length (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def area_rectangle (p1 p2 p3 : point) : ℝ :=
  let l1 := length p1 p2
  let l2 := length p1 p3
  l1 * l2

theorem rectangle_area_EFGH :
  area_rectangle E F H = 1040 := by
  sorry

end rectangle_area_EFGH_l779_779865


namespace expected_rolls_over_year_l779_779714

noncomputable def expected_rolls_per_day : ℚ :=
  let E := (7 / 8) * 1 + (1 / 8) * (1 + E)
  classical.some (Exists.intro E (by linarith))

noncomputable def expected_rolls_per_year : ℚ :=
  365 * expected_rolls_per_day

theorem expected_rolls_over_year : expected_rolls_per_year = 417.14 := 
by
  simp [expected_rolls_per_day, expected_rolls_per_year]
  -- further proof steps go here
  sorry

end expected_rolls_over_year_l779_779714


namespace total_raisins_l779_779880

theorem total_raisins (yellow raisins black raisins : ℝ) (h_yellow : yellow = 0.3) (h_black : black = 0.4) : yellow + black = 0.7 := 
by
  sorry

end total_raisins_l779_779880


namespace unique_real_solution_l779_779001

theorem unique_real_solution :
  ∀ (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) →
    (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) :=
by
  intro x y z w
  intros h
  have h1 : x = z + w + Real.sqrt (z * w * x) := h.1
  have h2 : y = w + x + Real.sqrt (w * x * y) := h.2.1
  have h3 : z = x + y + Real.sqrt (x * y * z) := h.2.2.1
  have h4 : w = y + z + Real.sqrt (y * z * w) := h.2.2.2
  sorry

end unique_real_solution_l779_779001


namespace distinct_L_shapes_l779_779322

-- Definitions of conditions
def num_convex_shapes : Nat := 10
def L_shapes_per_convex : Nat := 2
def corner_L_shapes : Nat := 4

-- Total number of distinct "L" shapes
def total_L_shapes : Nat :=
  num_convex_shapes * L_shapes_per_convex + corner_L_shapes

theorem distinct_L_shapes :
  total_L_shapes = 24 :=
by
  -- Proof is omitted
  sorry

end distinct_L_shapes_l779_779322


namespace find_a1_general_formula_sequence_a_l779_779543

noncomputable def sequence_a : ℕ → ℤ
| 1       := 1
| (n + 1) := 3 * 2^n - 2

def S (n : ℕ) : ℤ := (finset.range n).sum sequence_a

def T (n : ℕ) : ℤ := (finset.range n).sum S

axiom T_condition (n : ℕ) : T n = 2 * S n - n^2

theorem find_a1 :
  sequence_a 1 = 1 :=
sorry

theorem general_formula_sequence_a (n : ℕ) :
  sequence_a (n + 1) = 3 * 2^n - 2 :=
sorry

end find_a1_general_formula_sequence_a_l779_779543


namespace hats_in_box_total_l779_779970

theorem hats_in_box_total : 
  (∃ (n : ℕ), (∀ (r b y : ℕ), r + y = n - 2 ∧ r + b = n - 2 ∧ b + y = n - 2)) → (∃ n, n = 3) :=
by
  sorry

end hats_in_box_total_l779_779970


namespace bob_arrives_before_345_given_alice_after_bob_l779_779317

theorem bob_arrives_before_345_given_alice_after_bob :
  (∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 60) ∧ (0 ≤ y ∧ y ≤ 60) ∧ (y > x) → 
  (x < 45) / (y > x) = 9/16) := 
sorry

end bob_arrives_before_345_given_alice_after_bob_l779_779317


namespace gears_identical_flags_rotation_l779_779162

-- Definitions for our problem context
def initial_flags_at_top : Prop := true -- Assume both flags start at the top position.

def opposite_rotations_when_meshed (left_rotates right_rotates : ℝ) : Prop :=
  left_rotates = -right_rotates
 
-- The main problem statement regarding the flags reaching position (a)
theorem gears_identical_flags_rotation (left_rotation : ℝ) (right_rotation : ℝ) :
  initial_flags_at_top →
  opposite_rotations_when_meshed left_rotation right_rotation →
  right_rotation = -left_rotation :=
by {
  intro h1,  -- The flags start at the top position.
  intro h2,  -- The wheels rotate in opposite directions by the same angle.
  exact h2,  -- Hence the final position can be represented by option (a).
}

end gears_identical_flags_rotation_l779_779162


namespace part_one_part_two_l779_779435

def f (a x : ℝ) : ℝ := |a - 4 * x| + |2 * a + x|

theorem part_one (x : ℝ) : f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 / 5 := 
sorry

theorem part_two (a x : ℝ) : f a x + f a (-1 / x) ≥ 10 := 
sorry

end part_one_part_two_l779_779435


namespace maxwell_meets_brad_in_6_hours_l779_779551

noncomputable def T : ℝ :=
  54 / (4 + 6)

theorem maxwell_meets_brad_in_6_hours :
  (let T : ℝ := 54 / (4 + 6) in T) = 6 :=
by {
  -- Proof would go here
  dsimp only [T],
  norm_num,
  sorry
}

end maxwell_meets_brad_in_6_hours_l779_779551


namespace ratio_of_triangle_areas_l779_779304

theorem ratio_of_triangle_areas 
  (x y : ℝ) (n m : ℕ) 
  (h1: 0 < x) 
  (h2: 0 < y) 
  (h3: 0 < n) 
  (h4: 0 < m) :
  let area_A := (x + y) * real.sqrt (x^2 + y^2) / (4 * n * real.sqrt 2),
      area_B := (x + y) * real.sqrt (x^2 + y^2) / (4 * m * real.sqrt 2)
  in area_A / area_B = (m : ℝ) / n := by
  sorry

end ratio_of_triangle_areas_l779_779304


namespace proof_problem_l779_779064

noncomputable theory

open Real

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x < 3^x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- The Lean statement 
theorem proof_problem : ¬p ∧ q := 
by
  sorry

end proof_problem_l779_779064


namespace probability_distinct_digits_odd_units_l779_779392

-- Definition of conditions
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def allDigitsDistinct (n : ℕ) : Prop := 
  let digits := List.ofFn (λ i => n.digits (10^(i-1)) % 10) in
  digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.1 ≠ digits.4 ∧ digits.2 ≠ digits.3 ∧ digits.2 ≠ digits.4 ∧ digits.3 ≠ digits.4
def unitsDigitOdd (n : ℕ) : Prop := (n % 10) ∈ {1, 3, 5, 7, 9}

-- Main theorem statement
theorem probability_distinct_digits_odd_units : 
  let favorable_outcomes := 2240
  let total_outcomes := 9000
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  isFourDigit n ∧ allDigitsDistinct n ∧ unitsDigitOdd n → probability = 56 / 225 :=
by sorry

end probability_distinct_digits_odd_units_l779_779392


namespace compute_n_binom_l779_779413

-- Definitions based on conditions
def n : ℕ := sorry  -- Assume n is a positive integer defined elsewhere
def k : ℕ := 4

-- The binomial coefficient definition
def binom (n k : ℕ) : ℕ :=
  if h₁ : k ≤ n then
    (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))
  else 0

-- The theorem to prove
theorem compute_n_binom : n * binom k 3 = 4 * n :=
by
  sorry

end compute_n_binom_l779_779413


namespace needed_value_l779_779285

section ComplexExpressions

-- Definitions from the problem
variable (complex_set : Type)
variable (complex_expressions : Nat → complex_set → Prop)

-- Given conditions
axiom C : complex_set
axiom complexity_C : complex_expressions 1 C 

axiom f : complex_set → complex_set → complex_set
axiom complexity_f : forall (C1 C2 : complex_set) (c1 c2 : Nat), 
  complex_expressions c1 C1 → complex_expressions c2 C2 → complex_expressions (c1 + c2) (f C1 C2)

-- Define the generating function conditions related to the problem
axiom generating_function_condition : ∀ (a : ℕ → ℕ) (x : ℝ),
  (a 1 + a 2 * x + a 3 * x^2 + ∑ n in (Finset.range 100), ((a (n + 1)) * x^n)) = (7 / 4)

-- Define the desired constant
definition x_value := (2 * Real.sqrt 111) / 49

-- The main theorem to be proven
theorem needed_value : ∃ (x : ℝ), 
  generating_function_condition a x ∧ x = x_value :=
sorry

end ComplexExpressions

end needed_value_l779_779285


namespace rhombus_diagonals_not_equal_l779_779639

-- Define a rhombus and its properties
structure Rhombus (R : Type*) [AddCommGroup R] [Module ℝ R] :=
  (adjacent_sides_equal: ∀ (a b : R), a ≠ b → ∃ (c d : R), c = d)
  (diagonals_bisect: ∀ (a b : R), bisect a b)
  (diagonals_perpendicular: ∀ (a b : R), is_perpendicular a b)

-- Statement D: The diagonals are equal
def diagonals_equal (R : Type*) [AddCommGroup R] [Module ℝ R] (r : Rhombus R) : Prop := ∀ (a b : R), a = b

-- Proving statement D is incorrect for a rhombus
theorem rhombus_diagonals_not_equal (R : Type*) [AddCommGroup R] [Module ℝ R] (r : Rhombus R) : 
  ¬diagonals_equal R r :=
by 
  sorry

end rhombus_diagonals_not_equal_l779_779639


namespace num_true_statements_l779_779783

-- Define the sequence
def a_n (n : ℕ) (k : ℝ) : ℝ := n * k^n

-- Define conditions and statements
def statement_1 (k : ℝ) : Prop := 
  ∀ n : ℕ, 0 < n → (k = 1/2 → a_n n k > a_n (n + 1) k)

def statement_2 (k : ℝ) : Prop := 
  ∀ k, (1/2 < k ∧ k < 1) → ¬∃ max_n : ℕ, ∀ n : ℕ, 0 < n → a_n n k ≤ a_n max_n k

def statement_3 (k : ℝ) : Prop := 
  ∀ n : ℕ, 0 < n → (0 < k ∧ k < 1/2 → a_n n k > a_n (n + 1) k)

def statement_4 (k : ℝ) : Prop := 
  (k / (1 - k)).denom = 1 → ∃ n : ℕ, 0 < n ∧ a_n n k = a_n (n + 1) k

-- Proof goal: Verify the number of true statements
theorem num_true_statements (k : ℝ) : 
  0 < k → k < 1 → 
  (if statement_1 k then 1 else 0) + 
  (if statement_2 k then 1 else 0) + 
  (if statement_3 k then 1 else 0) + 
  (if statement_4 k then 1 else 0) = 2 :=
sorry

end num_true_statements_l779_779783


namespace rectangular_to_polar_l779_779738

theorem rectangular_to_polar (x y : ℝ) (hxy : x = 8 ∧ y = 8) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 8 * Real.sqrt 2 ∧ θ = Real.pi / 4 :=
by {
  use [8 * Real.sqrt 2, Real.pi / 4],
  split,
  { exact Real.sqrt_pos.2 (by linarith [show 8 ^ 2 + 8 ^ 2 > 0, by norm_num]) },
  split,
  { exact Real.pi_div_two_pos.le },
  split,
  { exact (Real.pi_pos).le.trans_lt (by norm_num [Real.pi]) },
  split,
  { exact (hxy.1 ▸ hxy.2 ▸ eq_of_sq_eq_sq (by norm_num [Real.sqrt_eq_rpow]) rfl).symm },
  { refl }
}

end rectangular_to_polar_l779_779738


namespace number_of_solutions_l779_779453

theorem number_of_solutions :
  ∃ (solutions : Finset (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ solutions ↔ (x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1)) ∧ 
  solutions.card = 2 :=
by
  sorry

end number_of_solutions_l779_779453


namespace sum_M_N_K_l779_779834

theorem sum_M_N_K (d K M N : ℤ) 
(h : ∀ x : ℤ, (x^2 + 3*x + 1) ∣ (x^4 - d*x^3 + M*x^2 + N*x + K)) :
  M + N + K = 5*K - 4*d - 11 := 
sorry

end sum_M_N_K_l779_779834


namespace toll_for_18_wheel_truck_l779_779270

-- Define the number of wheels and axles conditions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def number_of_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the toll calculation formula
def toll (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

-- Lean theorem statement asserting that the toll for the given truck is 6 dollars
theorem toll_for_18_wheel_truck : toll number_of_axles = 6 := by
  -- Skipping the actual proof using sorry
  sorry

end toll_for_18_wheel_truck_l779_779270


namespace valid_coloring_implication_l779_779048

theorem valid_coloring_implication (k h n : ℕ)
  (h_k_ge_2 : k ≥ 2)
  (h_h_ge_2 : h ≥ 2)
  (coloring : (ℕ × ℕ) → ℕ)
  (valid_coloring : ∀ (C : ℕ)
    (valid_C : ∀ (x : ℕ) (hx : x < n), x = C → true)
    (valid_rectangle_colors : ∀ (x1 y1 x2 y2 : ℕ)
      (hx1 : x1 < k * n) (hy1 : y1 < h * n)
      (hx2 : x2 < k * n) (hy2 : y2 < h * n),
      x1 ≠ x2 → y1 ≠ y2 →
      coloring (x1, y1) ≠ coloring (x1, y2) ∧
      coloring (x1, y1) ≠ coloring (x2, y1) ∧
      coloring (x1, y2) ≠ coloring (x2, y2) ∧
      coloring (x2, y1) ≠ coloring (x2, y2))
    (valid_row : ∀ y < h * n, set.count (λ x, coloring (x, y) = C) = k)
    (valid_column : ∀ x < k * n, set.count (λ y, coloring (x, y) = C) = h))
  : k * h ≤ n * (n + 1) :=
sorry

end valid_coloring_implication_l779_779048


namespace find_t_l779_779404

def f (x : ℝ) : ℝ := x^4 + Real.exp (|x|)

theorem find_t (t : ℝ) : (2 * f (Real.log t) - f (Real.log (1 / t)) ≤ f 2) ↔ (t ∈ Set.Icc (Real.exp (-2)) (Real.exp 2)) :=
sorry

end find_t_l779_779404


namespace option_A_correct_option_D_correct_l779_779819

-- Define the lines l1 and l2 in terms of the parameter a
def l1 (a : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), a * p.1 + 2 * p.2 + 3 * a = 0

def l2 (a : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), 3 * p.1 + (a - 1) * p.2 + 7 - a = 0

-- Proof that when a = 2/5, l1 is perpendicular to l2
theorem option_A_correct : 
  l1 (2/5) = l1 0 → l2 (2/5) = l2 0 → true := by sorry

-- Proof that when lines are parallel, distance between them is correct distance
theorem option_D_correct (a : ℝ) (h : ∀ p : ℝ × ℝ, l1 a p ↔ l2 a p) : 
  distance_between_lines (l1 a) (l2 a) = 5 / 13 * sqrt 13 := 
by sorry

end option_A_correct_option_D_correct_l779_779819


namespace tangent_value_of_k_k_range_l779_779910

noncomputable def f (x : Real) : Real := Real.exp (2 * x)
def g (k x : Real) : Real := k * x + 1

theorem tangent_value_of_k (k : Real) :
  (∃ t : Real, f t = g k t ∧ deriv f t = deriv (g k) t) → k = 2 :=
by
  sorry

theorem k_range (k : Real) (h : k > 0) :
  (∃ m : Real, m > 0 ∧ ∀ x : Real, 0 < x → x < m → |f x - g k x| > 2 * x) → 4 < k :=
by
  sorry

end tangent_value_of_k_k_range_l779_779910


namespace total_amount_from_grandparents_l779_779507

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l779_779507


namespace altitudes_lie_on_bisectors_l779_779948

theorem altitudes_lie_on_bisectors (A B C A₁ B₁ C₁ : Type*)
  [is_acute_angle_triangle ABC]
  [circumcircle_intersections A B C A₁ B₁ C₁]:
  altitudes_lie_on_bisectors A₁ B₁ C₁ A B C :=
by sorry

end altitudes_lie_on_bisectors_l779_779948


namespace table_legs_l779_779673

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end table_legs_l779_779673


namespace ellipse_problem_l779_779058

-- Define the conditions
def ellipse_conditions (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :=
  a > b ∧ a^2 - b^2 = c^2 ∧ c / a = (Real.sqrt 3) / 2 ∧ (a^2 / c - c = (Real.sqrt 3) / 3)

-- Define the problem statement
theorem ellipse_problem (a b c : ℝ) (h : ellipse_conditions a b c h_a h_b h_c) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a > b) ∧ (a^2 - b^2 = c^2) ∧ (c / a = (Real.sqrt 3) / 2) ∧ ((a^2 / c - c) = (Real.sqrt 3) / 3) → 
  (std_eq : ∀ x y, x^2 / 4 + y^2 = 1) ∧ 
  let right_focus := (Real.sqrt 3, 0),
  let line_eq : ℝ → ℝ := λ x, x - Real.sqrt 3,
  let intersection_points := {p : ℝ × ℝ | p.snd = line_eq p.fst ∧ p.fst^2 / 4 + p.snd^2 = 1},
  ∃ x1 x2, (x1, line_eq x1) ∈ intersection_points ∧ (x2, line_eq x2) ∈ intersection_points ∧
  dist (x1, line_eq x1) (x2, line_eq x2) = 8/5 
)) :=
sorry

end ellipse_problem_l779_779058


namespace alcohol_to_water_ratio_l779_779622

theorem alcohol_to_water_ratio
    (P_alcohol_percent Q_alcohol_percent R_alcohol_percent : ℝ)
    (V_P V_Q V_R : ℝ)
    (hP : P_alcohol_percent = 0.625)
    (hQ : Q_alcohol_percent = 0.875)
    (hR : R_alcohol_percent = 0.7)
    (hV_P : V_P = 4)
    (hV_Q : V_Q = 5)
    (hV_R : V_R = 3) :
    let total_alcohol := P_alcohol_percent * V_P + Q_alcohol_percent * V_Q + R_alcohol_percent * V_R,
        total_volume := V_P + V_Q + V_R,
        total_water := total_volume - total_alcohol,
        ratio := total_alcohol / total_water in
    ratio ≈ 3 :=
by
  sorry

end alcohol_to_water_ratio_l779_779622


namespace vertical_distance_from_top_to_bottom_l779_779291

-- Conditions
def ring_thickness : ℕ := 2
def largest_ring_diameter : ℕ := 18
def smallest_ring_diameter : ℕ := 4

-- Additional definitions based on the problem context
def count_rings : ℕ := (largest_ring_diameter - smallest_ring_diameter) / ring_thickness + 1
def inner_diameters_sum : ℕ := count_rings * (largest_ring_diameter - ring_thickness + smallest_ring_diameter) / 2
def vertical_distance : ℕ := inner_diameters_sum + 2 * ring_thickness

-- The problem statement to prove
theorem vertical_distance_from_top_to_bottom :
  vertical_distance = 76 := by
  sorry

end vertical_distance_from_top_to_bottom_l779_779291


namespace find_a_l779_779146

open Real

noncomputable def parametric_C1 (a φ : ℝ) : ℝ × ℝ :=
  (a + a * Real.cos φ, a * Real.sin φ)

def polar_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 / (1 + 3 * (Real.sin θ)^2)

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem find_a (a : ℝ) :
  (∀ φ, 0.5 * π ≤ φ ∧ φ ≤ 1.5 * π → parametric_C1 a φ ∈ set_of (λ (p : ℝ × ℝ), (p.1 - a)^2 + p.2^2 = a^2)) →
  (∀ θ ρ, polar_C2 ρ θ → ρ^2 = (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2) →
  ∀ A B : ℝ × ℝ, distance A B = 4 * Real.sqrt 2 / 3 → 
  ∃ a' : ℝ, a' = 1 :=
by  
  intro h1 h2 h3
  sorry

end find_a_l779_779146


namespace area_of_triangle_l779_779689

theorem area_of_triangle (x : ℝ) :
  let t1_area := 16
  let t2_area := 25
  let t3_area := 64
  let total_area_factor := t1_area + t2_area + t3_area
  let side_factor := 17 * 17
  ΔABC_area = side_factor * total_area_factor :=
by {
  -- Placeholder to complete the proof
  sorry
}

end area_of_triangle_l779_779689


namespace opposite_direction_of_vectors_l779_779104

theorem opposite_direction_of_vectors
  (x : ℝ)
  (a : ℝ × ℝ := (x, 1))
  (b : ℝ × ℝ := (4, x)) :
  (∃ k : ℝ, k ≠ 0 ∧ a = -k • b) → x = -2 := 
sorry

end opposite_direction_of_vectors_l779_779104


namespace number_of_subsets_with_exactly_one_isolated_element_l779_779525

def is_isolated_element (A : Finset ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def has_exactly_one_isolated_element (A : Finset ℤ) (B : Finset ℤ) : Prop :=
  ∃ k ∈ B, is_isolated_element B k ∧ ∀ m ∈ B, m ≠ k → ¬ is_isolated_element B m

theorem number_of_subsets_with_exactly_one_isolated_element :
  let A := ({1, 2, 3, 4, 5} : Finset ℤ) in
  (A.powerset.filter (has_exactly_one_isolated_element A)).card = 13 :=
by
  sorry

end number_of_subsets_with_exactly_one_isolated_element_l779_779525


namespace bus_stop_minutes_per_hour_l779_779751

/-- Given the average speed of a bus excluding stoppages is 60 km/hr
and including stoppages is 15 km/hr, prove that the bus stops for 45 minutes per hour. -/
theorem bus_stop_minutes_per_hour
  (speed_no_stops : ℝ := 60)
  (speed_with_stops : ℝ := 15) :
  ∃ t : ℝ, t = 45 :=
by
  sorry

end bus_stop_minutes_per_hour_l779_779751


namespace round_to_nearest_hundredth_l779_779729

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 28.737) : Real.round_to_nearest x 2 = 28.74 :=
by
  have h : Real.round_to_nearest 28.737 2 = 28.74 := 
    -- Insert reasoning specific to Lean methods that round the number
    sorry
  exact h

end round_to_nearest_hundredth_l779_779729


namespace compute_nested_f_l779_779740

-- Define the function f
def f (x y z : ℚ) : ℚ :=
  (x + y) / z

-- Define the condition that z ≠ 0
lemma z_ne_zero {z : ℚ} (hz : z ≠ 0) : z ≠ 0 := hz

-- The problem statement
theorem compute_nested_f :
  f (f 120 60 180) (f 4 2 6) (f 20 10 30) = 2 :=
begin
  sorry
end

end compute_nested_f_l779_779740


namespace lines_through_M_are_tangents_l779_779754

structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

def Line := Point × Point  -- A line represented by two points

def distance (a b : Point) : ℝ :=
  real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2)

-- Cylinder definition: axis line and radius
structure Cylinder :=
(axis : Line) (radius : ℝ)

-- Given conditions
variable (M : Point) (d : ℝ) (AB : Line)

-- Definition: Tangent lines to a cylinder around AB of radius d passing through M
def isTangentLine (c : Cylinder) (L : Line) : Prop :=
  let p1 := L.fst
  let p2 := L.snd
  p1 != p2 ∧ distance p1 p2 = c.radius ∧
  -- Both points must be on the surface of the cylinder
  (distance p1 AB.fst = distance p1 AB.snd) ∧
  (distance p2 AB.fst = distance p2 AB.snd)

theorem lines_through_M_are_tangents {M : Point} {d : ℝ} {AB : Line} :
  ∀ L : Line, (L.fst = M ∨ L.snd = M) ∧ distance (L.fst) (noncomputable.mkPoint AB.fst AB.snd) = d → 
  isTangentLine (Cylinder.mk AB d) L :=
sorry

end lines_through_M_are_tangents_l779_779754


namespace jack_walked_distance_l779_779159

def jack_walking_time: ℝ := 1.25
def jack_walking_rate: ℝ := 3.2
def jack_distance_walked: ℝ := 4

theorem jack_walked_distance:
  jack_walking_rate * jack_walking_time = jack_distance_walked :=
by
  sorry

end jack_walked_distance_l779_779159


namespace max_trig_expression_is_4_5_l779_779031

noncomputable def max_trig_expression : ℝ :=
  sup ((λ x y z, (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z)) '' 
    (set.univ : set (ℝ × ℝ × ℝ)))

theorem max_trig_expression_is_4_5 : max_trig_expression = 4.5 :=
sorry

end max_trig_expression_is_4_5_l779_779031


namespace min_marked_cells_max_marked_cells_l779_779846

/-- Define a 7x7 grid as a 2D array where each entry represents a cell being marked or unmarked. -/
def grid : Type := array 7 (array 7 bool)

/-- Condition 1: Ensure at least one cell is marked in each row. -/
def at_least_one_per_row (g : grid) : Prop :=
  ∀ (i : Fin 7), ∃ (j : Fin 7), g[i][j] = true

/-- Condition 2: Ensure at least one cell is marked in each column. -/
def at_least_one_per_column (g : grid) : Prop :=
  ∀ (j : Fin 7), ∃ (i : Fin 7), g[i][j] = true

/-- Condition 3: Ensure the number of marked cells in any row is odd. -/
def odd_cells_in_rows (g : grid) : Prop :=
  ∀ (i : Fin 7), (array.foldl (λ (acc : Nat) (b : bool), if b then acc + 1 else acc) 0 (g[i]) % 2 = 1)

/-- Condition 4: Ensure the number of marked cells in any column is divisible by 3. -/
def divisible_by_3_cells_in_columns (g : grid) : Prop :=
  ∀ (j : Fin 7), (array.foldl (λ (acc : Nat) (i : Fin 7), if g[i][j] then acc + 1 else acc) 0 (array.enumFromZeroTo 6) % 3 = 0)

/-- The minimum number of marked cells is 21. -/
theorem min_marked_cells : ∃ (g : grid), at_least_one_per_row g ∧ at_least_one_per_column g ∧ 
                            odd_cells_in_rows g ∧ divisible_by_3_cells_in_columns g ∧ 
                            (array.foldl (λ (acc : Nat) (i : Fin 7), acc + (array.foldl (λ (acc2 : Nat) (b : bool), if b then acc2 + 1 else acc2) 0 (g[i]))) 0 (array.enumFromZeroTo 6) = 21) :=
sorry

/-- The maximum number of marked cells is 39. -/
theorem max_marked_cells : ∃ (g : grid), at_least_one_per_row g ∧ at_least_one_per_column g ∧ 
                            odd_cells_in_rows g ∧ divisible_by_3_cells_in_columns g ∧ 
                            (array.foldl (λ (acc : Nat) (i : Fin 7), acc + (array.foldl (λ (acc2 : Nat) (b : bool), if b then acc2 + 1 else acc2) 0 (g[i]))) 0 (array.enumFromZeroTo 6) = 39) :=
sorry

end min_marked_cells_max_marked_cells_l779_779846


namespace angle_sum_eq_180_l779_779323

theorem angle_sum_eq_180 (A B C D E F G : ℝ) 
  (h1 : A + B + C + D + E + F = 360) : 
  A + B + C + D + E + F + G = 180 :=
by
  sorry

end angle_sum_eq_180_l779_779323


namespace polygon_three_sides_l779_779292

theorem polygon_three_sides (n : ℕ) (P : ℝ^3) (polygon : List (ℝ^3)) (h_convex : convex polygon) (h_not_plane : ∀ p ∈ polygon, p ≠ P) :
  Σ i : Fin n, angle (P, polygon[i], polygon[i+1]) = Σ i : Fin n, angle (P, polygon[i], P) →
  n = 3 :=
by
  sorry

end polygon_three_sides_l779_779292


namespace original_price_after_discount_l779_779124

theorem original_price_after_discount (a x : ℝ) (h : 0.7 * x = a) : x = (10 / 7) * a := 
sorry

end original_price_after_discount_l779_779124


namespace ratio_of_divisors_l779_779896

-- Definition of N
def N : ℕ := 34 * 34 * 63 * 270

-- Definition of sum of odd divisors function
noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (λ d, ¬(d % 2 = 0)) (Finset.divisors n), d

-- Definition of sum of even divisors function
noncomputable def sum_of_even_divisors (n : ℕ) : ℕ :=
  ∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d

-- Theorem statement
theorem ratio_of_divisors : (sum_of_odd_divisors N) / (sum_of_even_divisors N) = 1 / 14 := by
  sorry

end ratio_of_divisors_l779_779896


namespace xy_squared_value_l779_779457

variable {x y : ℝ}

theorem xy_squared_value :
  (y + 6 = (x - 3)^2) ∧ (x + 6 = (y - 3)^2) ∧ (x ≠ y) → (x^2 + y^2 = 25) := 
by
  sorry

end xy_squared_value_l779_779457


namespace evaluate_f_at_2_l779_779255

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end evaluate_f_at_2_l779_779255


namespace cylindrical_water_tower_height_l779_779294

noncomputable theory

-- Define the problem parameters
def r : ℝ := 1

-- Define the main theorem
theorem cylindrical_water_tower_height (h v : ℝ) 
  (h1 : 8 * v * π = π * (h - 3))
  (h2 : 10 * v * π = π * h - 2 * π) : h = 7 :=
sorry

end cylindrical_water_tower_height_l779_779294


namespace am_plus_an_eq_ab_l779_779447

noncomputable theory
open_locale classical

theorem am_plus_an_eq_ab (r a : ℝ) (h1 : 2a < r / 2) (h2 : ∀ M N : ℝ × ℝ,
  let AM := dist (M.1, 0) (r + a, 0),
      AN := dist (N.1, 0) (r + a, 0),
      MP := dist M (M.1, 0),
      NQ := dist N (N.1, 0) in
  (MP / AM = 1) ∧ (NQ / AN = 1)) :
  ∀ (AB : ℝ), AB = 2 * r → 
  (∃ M N : ℝ × ℝ, dist (M.1 + N.1, 0) (2 * (r + a), 0)) :=
begin
  sorry
end

end am_plus_an_eq_ab_l779_779447


namespace area_difference_of_square_and_rectangle_l779_779550

theorem area_difference_of_square_and_rectangle :
  ∀ (P : ℕ) (w : ℕ), P = 52 → w = 15 →
  let s := P / 4 in
  let A_square := s * s in
  let l := (P - 2 * w) / 2 in
  let A_rectangle := l * w in
  A_square - A_rectangle = 4 :=
by
  intros P w P_eq w_eq
  let s := P / 4
  let A_square := s * s
  let l := (P - 2 * w) / 2
  let A_rectangle := l * w
  have P_52 : P = 52 := P_eq
  have w_15 : w = 15 := w_eq
  rw [P_52, w_15] at *
  have s_def : s = 13 := rfl
  have l_def : l = 11 := by norm_num
  rw [s_def, l_def]
  have A_sq_def : A_square = 169 := by norm_num
  have A_rect_def : A_rectangle = 165 := by norm_num
  rw [A_sq_def, A_rect_def]
  norm_num

end area_difference_of_square_and_rectangle_l779_779550


namespace three_x_squared_y_squared_eq_588_l779_779533

theorem three_x_squared_y_squared_eq_588 (x y : ℤ) 
  (h : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 :=
sorry

end three_x_squared_y_squared_eq_588_l779_779533


namespace double_sum_evaluation_l779_779630

theorem double_sum_evaluation :
  ∑ i in Finset.range 100, ∑ j in Finset.range 100, (i + 1 + j + 1) = 1_010_000 :=
by
  sorry

end double_sum_evaluation_l779_779630


namespace age_of_other_man_l779_779584

variables (A M : ℝ)

theorem age_of_other_man 
  (avg_age_of_men : ℝ)
  (replaced_man_age : ℝ)
  (avg_age_of_women : ℝ)
  (total_age_6_men : 6 * avg_age_of_men = 6 * (avg_age_of_men + 3) - replaced_man_age - M + 2 * avg_age_of_women) :
  M = 44 :=
by
  sorry

end age_of_other_man_l779_779584


namespace problem_solution_l779_779901

variables {a b c : ℝ}

theorem problem_solution (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^3 * b^3 / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  a^3 * c^3 / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  b^3 * c^3 / ((b^3 - a^2 * c) * (c^3 - a^2 * b))) = 1 :=
sorry

end problem_solution_l779_779901


namespace remainder_3_pow_24_mod_7_l779_779985

theorem remainder_3_pow_24_mod_7 :
  (3 ^ 24) % 7 = 1 :=
by {
  -- Conditions
  have h1 : 7.prime := by { exact nat.prime_7 },
  have h2 : 3 % 7 ≠ 0 := by { norm_num },
  have h3 : 3^6 ≡ 1 [MOD 7] := by { exact pow_six_mod_seven },

  -- Proof of the theorem
  rw ← pow_mul,
  rw mul_comm,
  rw h3,
  norm_num,
  exact h3
 } -- sorry
end

end remainder_3_pow_24_mod_7_l779_779985


namespace equivalent_Ceva_l779_779649

/-- Equivalent form of Ceva's Theorem:
The lines AA1, BB1, and CC1 passing through the vertices of triangle ABC
are concurrent at point O if and only if the product of the sine ratios equals 1. -/
theorem equivalent_Ceva's_Theorem 
  {α β γ α' β' γ' : ℝ} (h1 : α = ∠ABB₁) (h2 : β = ∠BCC₁) (h3 : γ = ∠CAA₁)
  (h4 : α' = ∠ACC₁) (h5 : β' = ∠BAA₁) (h6 : γ' = ∠CBB₁) :
  (sin α / sin α') * (sin β / sin β') * (sin γ / sin γ') = 1 ↔ 
  concurrent (line AA₁) (line BB₁) (line CC₁) :=
sorry

end equivalent_Ceva_l779_779649


namespace find_m_l779_779418

theorem find_m (a : ℕ) (h1 : a > 0) (h2 : a < 28) :
  ∃ m, (∀ m', ((28 - a) * (29 - a) * (30 - a) * (31 - a) * (32 - a) * (33 - a) * (34 - a) * (35 - a)) = (nat.perm (35 - a) 8)) → m = 8 :=
sorry

end find_m_l779_779418


namespace additional_classes_per_grade_level_l779_779450

def students : Nat := 1590
def percentage_moving : Float := 0.40
def grade_levels : Nat := 3
def advanced_class_size : Nat := 20
def normal_class_size : Nat := 32

theorem additional_classes_per_grade_level :
  let moving_students := (percentage_moving * students.toFloat).toNat in
  let students_per_grade_level := moving_students / grade_levels in
  let remaining_students := students_per_grade_level - advanced_class_size in
  remaining_students / normal_class_size = 6 := 
sorry

end additional_classes_per_grade_level_l779_779450


namespace solve_quadratic_formula_solve_factoring_solve_factored_solution_l779_779208

noncomputable def quadratic_formula : (ℚ → ℚ → ℚ → Set ℚ) :=
  λ a b c, {x | x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)}

theorem solve_quadratic_formula :
  quadratic_formula 2 (-5) 1 = {x | x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4} :=
by
  sorry

theorem solve_factoring :
  ∀ x : ℚ, (2*x - 1)^2 - x^2 = 3*x^2 - 4*x + 1 :=
by
  intro x
  calc
    (2*x - 1)^2 - x^2
        = 4*x^2 - 4*x + 1 - x^2 : by ring
    ... = 3*x^2 - 4*x + 1       : by ring

theorem solve_factored_solution :
  ∀ x : ℚ, (3*x - 1) * (x - 1) = 0 ↔ x = 1/3 ∨ x = 1 :=
by
  intro x
  split
  . intro h
    cases’ eq_zero_or_eq_zero_of_mul_eq_zero h with h1 h2
    · right
      exact eq_zero_of_add_eq_zero h2
    · left
      exact eq_zero_of_add_eq_zero (eq.symm h2)
  . intro h
    cases h
    · exact mul_eq_zero_of_left (by linarith : 3 * x - 1 = 0)
    · exact mul_eq_zero_of_right (by linarith : x - 1 = 0)

end solve_quadratic_formula_solve_factoring_solve_factored_solution_l779_779208


namespace proof_count_valid_sets_l779_779395

/-- 
We need to count the number of sets of three distinct numbers from the set 
{5, 6, 7, 8, 9, 10, 11, 12, 13, 14} that include the number 10 and have a sum equal to 30. 
-/
noncomputable def count_valid_sets : ℕ :=
  let num_set := {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
  let target_set := {s : set ℕ | s ⊆ num_set ∧ 10 ∈ s ∧ s.sum = 30 ∧ s.card = 3}
  target_set.to_finset.card

theorem proof_count_valid_sets : count_valid_sets = 4 :=
sorry

end proof_count_valid_sets_l779_779395


namespace trapezoid_area_l779_779490

-- Define the conditions
variables {A B C D E F : Type*} [HT : Trapezoid A B C D]
variables [Midpoint E C D] [Perpendicular EF A B] [Length AB 5] [Length EF 4]

-- Define the theorem
theorem trapezoid_area (A B C D : Point) (E F : Point)
  (h1 : AD ∥ BC) (h2 : midpoint E C D) (h3 : perpendicular EF A B) (h4 : length AB = 5) (h5 : length EF = 4) :
  area (trapezoid A B C D) = 20 :=
sorry

end trapezoid_area_l779_779490


namespace factorial_expression_l779_779716

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l779_779716


namespace largest_alpha_exists_l779_779028

theorem largest_alpha_exists : 
  ∃ α, (∀ m n : ℕ, 0 < m → 0 < n → (m:ℝ) / (n:ℝ) < Real.sqrt 7 → α / (n^2:ℝ) ≤ 7 - (m^2:ℝ) / (n^2:ℝ)) ∧ α = 3 :=
by
  sorry

end largest_alpha_exists_l779_779028


namespace sum_of_reciprocals_bounds_l779_779561

theorem sum_of_reciprocals_bounds (n : ℕ) : 
  1/2 < ∑ k in finset.range (2 * n + 1) \ finset.range (n + 1), (1 : ℝ) / k ∧ 
  ∑ k in finset.range (2 * n + 1) \ finset.range (n + 1), (1 : ℝ) / k < 3/4 :=
by
  sorry

end sum_of_reciprocals_bounds_l779_779561


namespace coprime_in_five_consecutive_integers_l779_779266

theorem coprime_in_five_consecutive_integers (a : ℤ) :
  ∃ k ∈ {a, a+1, a+2, a+3, a+4}, ∀ b ∈ {a, a+1, a+2, a+3, a+4}, k ≠ b → is_coprime k b :=
by sorry

end coprime_in_five_consecutive_integers_l779_779266


namespace simplify_sqrt_tan_squared_to_trig_identity_l779_779280

-- Problem 1
theorem simplify_sqrt (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) :
  (sqrt ((1 + sin α) / (1 - sin α)) - sqrt ((1 - sin α) / (1 + sin α))) = -2 * tan α :=
sorry

-- Problem 2
theorem tan_squared_to_trig_identity (θ : ℝ) :
  (1 - (tan θ)^2) / (1 + (tan θ)^2) = (cos θ)^2 - (sin θ)^2 :=
sorry

end simplify_sqrt_tan_squared_to_trig_identity_l779_779280


namespace chemistry_more_than_physics_l779_779239

theorem chemistry_more_than_physics
  (M P C : ℕ)
  (h1 : M + P = 60)
  (h2 : (M + C) / 2 = 35) :
  ∃ x : ℕ, C = P + x ∧ x = 10 := 
by
  sorry

end chemistry_more_than_physics_l779_779239


namespace anderson_brown_swap_l779_779320

theorem anderson_brown_swap (d_A : ℝ) (d_B : ℝ) 
    (h_distance : d_A + d_B = 20)
    (anderson_walk_speed : 4) (anderson_cycle_speed : 10)
    (brown_walk_speed : 5) (brown_cycle_speed : 8) : 
    ∃ d_A d_B, 
        d_A = 60 / 7 ∧ d_B = 20 - d_A ∧
        (d_A / anderson_cycle_speed + (20 - d_A) / anderson_walk_speed 
         = d_B / brown_cycle_speed + (20 - d_B) / brown_walk_speed) := by
  sorry

end anderson_brown_swap_l779_779320


namespace cricket_average_increase_l779_779668

-- Define the conditions as variables
variables (innings_initial : ℕ) (average_initial : ℕ) (runs_next_innings : ℕ)
variables (runs_increase : ℕ)

-- Given conditions
def conditions := (innings_initial = 13) ∧ (average_initial = 22) ∧ (runs_next_innings = 92)

-- Target: Calculate the desired increase in average (runs_increase)
theorem cricket_average_increase (h : conditions innings_initial average_initial runs_next_innings) :
  runs_increase = 5 :=
  sorry

end cricket_average_increase_l779_779668


namespace problem1_problem2_l779_779540

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Statement 1: If a = 1 and p ∧ q is true, then the range of x is 2 < x < 3
theorem problem1 (x : ℝ) (h : 1 = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
sorry

-- Statement 2: If ¬p is a sufficient but not necessary condition for ¬q, then the range of a is 1 < a ≤ 2
theorem problem2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) (h3 : ¬ (∃ x, p x a) → ¬ (∃ x, q x)) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l779_779540


namespace polynomial_value_l779_779636

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l779_779636


namespace solve_problem_l779_779443

noncomputable def ellipseG : Set (ℝ × ℝ) :=
  {p | ∃ x y, (p = (x, y) ∧ (x ^ 2 / 3 + y ^ 2 / 2 = 1))}

theorem solve_problem
  (C1 : ∀ x y, (x - 1) ^ 2 + y ^ 2 = (7 * Real.sqrt 3 / 4) ^ 2)
  (C2 : ∀ x y, (x + 1) ^ 2 + y ^ 2 = (Real.sqrt 3 / 4) ^ 2)
  (internally_tangent_C1 : ∀ p : ℝ × ℝ, ellipseG p → ∃ r, ∀ x y, (x - 1) ^ 2 + y ^ 2 = (r + 7 * Real.sqrt 3 / 4) ^ 2)
  (externally_tangent_C2 : ∀ p : ℝ × ℝ, ellipseG p → ∃ r, ∀ x y, (x + 1) ^ 2 + y ^ 2 = (Real.sqrt 3 / 4 + r) ^ 2)
  (area_OPQ : ∀ (P Q : ℝ × ℝ), ellipseG P → ellipseG Q → ∃ S, 1 / 2 * (abs (P.1 * Q.2 - P.2 * Q.1)) = Real.sqrt 6 / 2)
  (max_dist_OM_PQ : ∀ (P Q : ℝ × ℝ), ellipseG P → ellipseG Q → 
                      let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in 
                      abs (real.sqrt (M.1 ^ 2 + M.2 ^ 2) - Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) ≤ Real.sqrt 6 / 2 - 2) :
  (∀ x y, ellipseG (x, y) → (x ^ 2 / 3 + y ^ 2 / 2 = 1)) ∧ 
  (∃ P Q : ℝ × ℝ, ellipseG P ∧ ellipseG Q ∧ let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in 
                     abs (real.sqrt (M.1 ^ 2 + M.2 ^ 2) - Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) = Real.sqrt 6 / 2 - 2) := sorry

end solve_problem_l779_779443


namespace region_area_correct_l779_779366

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l779_779366


namespace solve_trig_equation_l779_779036

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (cos (2 * x) ≠ 0) →
  (sin (2 * x) ≠ 0) →
  (tg(2 * x) * sin(2 * x) - 3 * √3 * ctg(2 * x) * cos(2 * x) = 0) →
  x = (π / 6) * (3 * n + 1) :=
by
  sorry

end solve_trig_equation_l779_779036


namespace exists_root_between_1_1_and_1_2_l779_779330

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_between_1_1_and_1_2 :
  ∃ x, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have pf1 : f 1.1 = -0.59 := by norm_num1
  have pf2 : f 1.2 = 0.84 := by norm_num1
  apply exists_between_of_sign_change pf1 pf2
  sorry

end exists_root_between_1_1_and_1_2_l779_779330


namespace factorial_expression_simplification_l779_779721

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l779_779721


namespace smallest_initial_number_wins_for_bernardo_l779_779331

theorem smallest_initial_number_wins_for_bernardo :
  ∃ N : ℕ, 0 ≤ N ∧ 27 * N + 360 ≥ 950 ∧ 27 * N + 360 ≤ 999 ∧ ∀ M : ℕ, 
  (0 ≤ M ∧ 27 * M + 360 ≥ 950 ∧ 27 * M + 360 ≤ 999 → N ≤ M) :=
begin
  sorry
end

end smallest_initial_number_wins_for_bernardo_l779_779331


namespace place_another_domino_l779_779193

theorem place_another_domino (board : matrix (fin 6) (fin 6) bool) (dominoes : list (fin 6 × fin 6 × fin 6 × fin 6)) :
  (∀ ⟨i, j⟩ ∈ dominoes, (board i j) = tt ∨ (board (i.fst) (i.snd)) = tt) ∧ (board.to_list.count tt = 14) →
  ∃ new_domino : fin 6 × fin 6 × fin 6 × fin 6, (board (new_domino.fst) (new_domino.snd) = ff ∧ board (new_domino.snd.fst) (new_domino.snd.snd) = ff) ∧
  is_adjacent (new_domino.fst, new_domino.snd) :=
sorry

end place_another_domino_l779_779193


namespace spherical_coordinates_neg_z_l779_779412

theorem spherical_coordinates_neg_z (x y z : ℝ) (h₀ : ρ = 5) (h₁ : θ = 3 * Real.pi / 4) (h₂ : φ = Real.pi / 3)
  (hx : x = ρ * Real.sin φ * Real.cos θ) 
  (hy : y = ρ * Real.sin φ * Real.sin θ) 
  (hz : z = ρ * Real.cos φ) : 
  (ρ, θ, π - φ) = (5, 3 * Real.pi / 4, 2 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_neg_z_l779_779412


namespace find_m_perpendicular_l779_779441

def vector_a : ℝ × ℝ := (-6, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (3, m)

theorem find_m_perpendicular (m : ℝ) (h : vector_a.1 * vector_b(m).1 + vector_a.2 * vector_b(m).2 = 0) : m = 9 :=
by
  -- proof goes here
  sorry

end find_m_perpendicular_l779_779441


namespace derivative_of_y_l779_779437

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem derivative_of_y (x : ℝ) (h : x ≠ 0) :
  Deriv (fun x => (Real.sin x) / x + Real.sqrt x + 2) x = 
  (x * Real.cos x - Real.sin x) / (x^2) + (1 / (2 * Real.sqrt x)) :=
by
  -- Explanation not needed, adding sorry for completeness.
  sorry

end derivative_of_y_l779_779437


namespace linear_function_unique_l779_779052

theorem linear_function_unique (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f(x) + f(y) - x * y| ≤ 1 / 4) :
  (∀ x : ℝ, f x = a * x + b) → a = 1 / 2 ∧ b = -1 / 8 :=
by
  sorry

end linear_function_unique_l779_779052


namespace geometric_sequence_inverse_sum_l779_779810

theorem geometric_sequence_inverse_sum (a : ℕ → ℝ) (n : ℕ) (h_geo : ∀ k, a (k + 1) = 2 * a k)
 (h_cond : a 3 - a 1 = 6) :
  (∑ i in Finset.range n, (1 / (a (i + 1) ^ 2))) = (1 / 3) * (1 - (1 / 4 ^ n)) :=
begin
  sorry
end

end geometric_sequence_inverse_sum_l779_779810


namespace balloons_count_l779_779318

theorem balloons_count :
  (Allan_balloons Jake_balloons : ℕ) (h1 : Allan_balloons = 2) (h2 : Jake_balloons = 4) :
  Allan_balloons + Jake_balloons = 6 := 
by 
  sorry

end balloons_count_l779_779318


namespace candy_per_bag_correct_l779_779772

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def friends : ℕ := 3
def bags : ℕ := 8

def remaining_candy (total candy_kept : ℕ) : ℕ := total - candy_kept
def candy_per_person (remaining people : ℕ) : ℕ := remaining / people
def candy_per_bag (per_person bags : ℕ) : ℕ := per_person / bags

theorem candy_per_bag_correct :
  candy_per_bag (candy_per_person (remaining_candy total_candy sister_candy) (friends + 1)) bags = 18 :=
by
  sorry

end candy_per_bag_correct_l779_779772


namespace apples_per_slice_is_two_l779_779338

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l779_779338


namespace simplify_expression_l779_779205

variable (a b : ℝ)

theorem simplify_expression :
  -2 * (a^3 - 3 * b^2) + 4 * (-b^2 + a^3) = 2 * a^3 + 2 * b^2 :=
by
  sorry

end simplify_expression_l779_779205


namespace problem1_problem2_l779_779658

-- Definitions for problem 1
def sin_330 : ℝ := -Math.sin (Math.pi / 6)
def five_log_expr : ℝ := 5 ^ (1 - Math.log 2 / Math.log 5)

-- Proof problem 1: Prove sin 330 + 5^{1 - log_5 2} = 2
theorem problem1 : sin_330 + five_log_expr = 2 := by
  sorry

-- Definitions for problem 2
def sqrt_expr1 : ℝ := Math.sqrt (4 - 2 * Math.sqrt 3)
def sqrt_expr2 : ℝ := 2 - Math.sqrt 3

-- Proof problem 2: Prove sqrt (4 - 2 sqrt 3) + 1/sqrt(7 + 4 sqrt 3) = 1
theorem problem2 : sqrt_expr1 + sqrt_expr2 = 1 := by
  sorry

end problem1_problem2_l779_779658


namespace train_pass_pole_time_l779_779702

/-- Given that a train traveling at a speed of 72 km/h takes 35 seconds to cross a stationary train that is 500 meters long, 
we want to prove that it takes the train 10 seconds to pass a pole. -/
theorem train_pass_pole_time 
  (train_speed_kmh : ℝ := 72) 
  (cross_time_stationary_train : ℝ := 35) 
  (stationary_train_length : ℝ := 500) 
  (pass_pole_time : ℝ := 10) :
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  let moving_train_length := train_speed_ms * cross_time_stationary_train - stationary_train_length in
  pass_pole_time = moving_train_length / train_speed_ms := by
  sorry

end train_pass_pole_time_l779_779702


namespace rounding_estimation_correct_l779_779477

theorem rounding_estimation_correct (a b d : ℕ)
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (a_round : ℕ) (b_round : ℕ) (d_round : ℕ)
  (h_round_a : a_round ≥ a) (h_round_b : b_round ≤ b) (h_round_d : d_round ≤ d) :
  (Real.sqrt (a_round / b_round) - Real.sqrt d_round) > (Real.sqrt (a / b) - Real.sqrt d) :=
by
  sorry

end rounding_estimation_correct_l779_779477


namespace sin2_cos2_line_l779_779771

theorem sin2_cos2_line (u : ℝ) : 
  let x := Real.sin u ^ 2
  let y := Real.cos u ^ 2
  in x + y = 1 := by
  sorry

end sin2_cos2_line_l779_779771


namespace triangle_problem_l779_779467

noncomputable theory

variables (A B C : Type) [MetricSpace A]

/-- In triangle ABC, given BC = 7, AB = 3, and sin C / sin B = 3/5, prove AC = 5 and angle A = 120 degrees. -/
theorem triangle_problem
  (BC AB AC : ℝ)
  (sin_C sin_B : ℝ)
  (hBC : BC = 7)
  (hAB : AB = 3)
  (h_sin_ratio : sin_C / sin_B = 3 / 5) :
  AC = 5 ∧ ∠A = 120 :=
begin
  -- skipping the actual proof steps
  sorry
end

end triangle_problem_l779_779467


namespace maximum_intersection_points_of_two_pentagons_l779_779981

theorem maximum_intersection_points_of_two_pentagons
  (P Q : Polygon)
  (hP : P.is_convex)
  (hQ : Q.is_convex)
  (hV_not_on_edges : ∀ v ∈ P.vertices, ∀ e ∈ Q.edges, v ∉ e)
  (hVQ_not_on_edges : ∀ v ∈ Q.vertices, ∀ e ∈ P.edges, v ∉ e) 
  (hPentagonP : P.sides = 5)
  (hPentagonQ : Q.sides = 5) :
  ∃ m, m ≤ 18 ∧ (∃ (intersections : ℕ), intersections = m) := 
sorry

end maximum_intersection_points_of_two_pentagons_l779_779981


namespace max_path_index_eq_l779_779604

open GraphTheory

-- Definitions based on the problem conditions and correct answer
namespace PathIndexProblem

-- Given connected graph G with independence number n > 1
variables {G : SimpleGraph} [Connected G] (n : ℕ) (hn : 1 < n)
variable (H : independence_number G = n)

-- Definition of path index for a graph
def path_index (G : SimpleGraph) : ℕ := sorry -- definition to be filled in

-- Theorem stating the maximum possible value of path index
theorem max_path_index_eq : path_index G = n - 1 := sorry

end PathIndexProblem

end max_path_index_eq_l779_779604


namespace range_of_a_l779_779234

theorem range_of_a (a : ℝ) : (5 - a > 1) → (a < 4) := 
by
  sorry

end range_of_a_l779_779234


namespace initial_mat_weavers_eq_4_l779_779937

theorem initial_mat_weavers_eq_4 :
  ∃ x : ℕ, (x * 4 = 4) ∧ (14 * 14 = 49) ∧ (x = 4) :=
sorry

end initial_mat_weavers_eq_4_l779_779937


namespace area_of_quadrilateral_rspy_l779_779152

open Real
open EuclideanGeometry

namespace TriangleArea

noncomputable def triangle_xyz (XYZ : Triangle ℝ) (XY XZ : ℝ) (a_xyz : ℝ) :=
XY = 40 ∧ XZ = 20 ∧ a_xyz = 160

def midpoints (XYZ : Triangle ℝ) (P Q : Point ℝ) : Prop :=
P = midpoint XYZ.X XYZ.Y ∧ Q = midpoint XYZ.X XYZ.Z

noncomputable def area_quadrilateral_rspy (XYZ : Triangle ℝ) (P Q R S Y : Point ℝ) (a_rspy : ℝ) : Prop :=
midpoints XYZ P Q ∧
∃ (bisector : Line ℝ), bisector ∈ angle_bisectors XYZ (XY, XZ) ∧
(R, S) = line_intersections bisector PQ YZ ∧
a_rspy = 120

theorem area_of_quadrilateral_rspy (XYZ : Triangle ℝ) (XY XZ : ℝ) (a_xyz a_rspy : ℝ) (P Q R S Y : Point ℝ) :
  triangle_xyz XYZ XY XZ a_xyz → area_quadrilateral_rspy XYZ P Q R S Y a_rspy :=
begin
  intro h,
  sorry
end

end TriangleArea

end area_of_quadrilateral_rspy_l779_779152


namespace floor_plus_ceil_eq_seven_l779_779014

theorem floor_plus_ceil_eq_seven (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end floor_plus_ceil_eq_seven_l779_779014


namespace maximize_triangles_l779_779483

-- Define the context and main problem conditions
theorem maximize_triangles (points : Fin 1989 → Point)
    (no_three_collinear : ∀ (p₁ p₂ p₃ : Point), p₁ ≠ p₂ → p₂ ≠ p₃ → p₁ ≠ p₃ → ¬ collinear p₁ p₂ p₃)
    (groups : Fin 30 → Fin 1989 → Prop)
    (different_group_sizes : ∀ (i j : Fin 30), i ≠ j → size (groups i) ≠ size (groups j)) :
  (∃ (n : Fin 30 → ℕ),  
    (∑ i, n i = 1989) ∧
    (∀ x : Fin 30, ∃ k ∈ {51, 52, 53, 54, 55, 56, 58, 59, 60, ..., 81}, n x = k)) :=
sorry

end maximize_triangles_l779_779483


namespace probability_tile_from_ANGLE_l779_779353

def letters_in_ALGEBRA : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A']
def letters_in_ANGLE : List Char := ['A', 'N', 'G', 'L', 'E']
def count_matching_letters (letters: List Char) (target: List Char) : Nat :=
  letters.foldr (fun l acc => if l ∈ target then acc + 1 else acc) 0

theorem probability_tile_from_ANGLE :
  (count_matching_letters letters_in_ALGEBRA letters_in_ANGLE : ℚ) / (letters_in_ALGEBRA.length : ℚ) = 5 / 7 :=
by
  sorry

end probability_tile_from_ANGLE_l779_779353


namespace more_ones_than_twos_in_first_billion_l779_779202

def digital_root (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

theorem more_ones_than_twos_in_first_billion : 
  ∃ (count_1 count_2 : ℕ), count_1 > count_2 ∧ 
  count_1 = (Finset.Icc 1 1000000000).filter (λ n, digital_root n = 1).card ∧ 
  count_2 = (Finset.Icc 1 1000000000).filter (λ n, digital_root n = 2).card :=
by
  sorry

end more_ones_than_twos_in_first_billion_l779_779202


namespace triangle_identity_l779_779843

theorem triangle_identity (A : ℝ) (a b c : ℝ) (B C : ℝ) (sin : ℝ → ℝ) (sqrt : ℝ → ℝ) :
  A = π / 3 →  -- Given angle A = 60 degrees, which is π/3 radians
  a = sqrt 13 →  -- Given length a = sqrt(13)
  (a / sin A) = (b / sin B) →  -- Sine Rule
  (a / sin A) = (c / sin C) →  -- Sine Rule
  (b / sin B) = (c / sin C) →  -- Sine Rule
  (a + b + c) / (sin A + sin B + sin C) = 2 * sqrt 3 * sqrt 13 / 9 :=
begin
  -- proof to be provided
  sorry,
end

end triangle_identity_l779_779843


namespace inequality_relation_l779_779798

theorem inequality_relation (a b c d : ℝ) 
  (hab : a > b) (hb : b > 0) 
  (hcd : c > d) (hd : d > 0) : 
  (a / d) > (b / c) :=
begin
  -- Insert proof here
  sorry
end

end inequality_relation_l779_779798


namespace exclude_stoppages_speed_l779_779354

-- Definitions based on conditions
def effective_speed : ℝ := 48 -- kmph
def stoppage_time : ℝ := 15.69 / 60 -- Converted from minutes to hours

-- Theorem statement
theorem exclude_stoppages_speed :
  ∃ S : ℝ, S ≈ 65 ∧ (effective_speed = S * (1 - stoppage_time)) :=
sorry

end exclude_stoppages_speed_l779_779354


namespace traveler_arrangements_l779_779206

theorem traveler_arrangements :
  let travelers := 6
  let rooms := 3
  ∃ (arrangements : Nat), arrangements = 240 := by
  sorry

end traveler_arrangements_l779_779206


namespace descending_order_of_numbers_l779_779009

theorem descending_order_of_numbers :
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  b > c ∧ c > a ∧ a > d :=
by
  let a := 62
  let b := 78
  let c := 64
  let d := 59
  sorry

end descending_order_of_numbers_l779_779009


namespace general_formula_sum_first_n_terms_l779_779051

open BigOperators

def geometric_sequence (a_3 : ℚ) (q : ℚ) : ℕ → ℚ
| 0       => 1 -- this is a placeholder since sequence usually start from 1
| (n + 1) => 1 * q ^ n

def sum_geometric_sequence (a_1 q : ℚ) (n : ℕ) : ℚ :=
  a_1 * (1 - q ^ n) / (1 - q)

theorem general_formula (a_3 : ℚ) (q : ℚ) (n : ℕ) (h_a3 : a_3 = 1 / 4) (h_q : q = -1 / 2) :
  geometric_sequence a_3 q (n + 1) = (-1 / 2) ^ n :=
by
  sorry

theorem sum_first_n_terms (a_1 q : ℚ) (n : ℕ) (h_a1 : a_1 = 1) (h_q : q = -1 / 2) :
  sum_geometric_sequence a_1 q n = 2 / 3 * (1 - (-1 / 2) ^ n) :=
by
  sorry

end general_formula_sum_first_n_terms_l779_779051


namespace math_proof_l779_779807

variable (f : ℝ → ℝ)

def is_odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def satisfies_period_2 : Prop :=
  ∀ x : ℝ, f (x - 1) = f (x + 1)

def decreasing_in_interval : Prop :=
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

noncomputable def proof_problem : Prop :=
  (f 1 = 0) ∧
  (∃ (zeros : set ℝ), zeros = {x | x ∈ Icc (-2) 2 ∧ f x = 0} ∧ zeros.card = 5) ∧
  (∀ x : ℝ, f (2014 + x) = -f (2014 - x))

theorem math_proof (h_odd : is_odd_function f) (h_period : satisfies_period_2 f) (h_decreasing : decreasing_in_interval f) : proof_problem f :=
sorry

end math_proof_l779_779807


namespace minimum_positive_period_tan_four_x_pi_div_three_l779_779224

def y (x : ℝ) := Real.tan (4 * x + Real.pi / 3)

theorem minimum_positive_period_tan_four_x_pi_div_three :
  ∀ T > 0, T = Real.pi / 4 → y (x + T) = y x :=
sorry

end minimum_positive_period_tan_four_x_pi_div_three_l779_779224


namespace remainder_when_x150_divided_by_x1_4_l779_779033

noncomputable def remainder_div_x150_by_x1_4 (x : ℝ) : ℝ :=
  x^150 % (x-1)^4

theorem remainder_when_x150_divided_by_x1_4 (x : ℝ) :
  remainder_div_x150_by_x1_4 x = -551300 * x^3 + 1665075 * x^2 - 1667400 * x + 562626 :=
by
  sorry

end remainder_when_x150_divided_by_x1_4_l779_779033


namespace least_number_to_add_l779_779256

theorem least_number_to_add (n : ℕ) (h : (1052 + n) % 37 = 0) : n = 19 := by
  sorry

end least_number_to_add_l779_779256


namespace return_forest_years_l779_779207

theorem return_forest_years (log_val : ℤ → ℝ)
    (total_land : ℝ)
    (initial_target : ℝ)
    (annual_increase : ℝ)
    (fraction_west : ℝ)
    (approx_log : log_val 124 = 8) :
    total_land = 9100 ∧ initial_target = 515 ∧ annual_increase = 0.12 ∧ fraction_west = 0.7 →
    (∃ x : ℕ, (1 + annual_increase)^x ≈ 2.4843) → 
    x = 9 := sorry

end return_forest_years_l779_779207


namespace qiantang_tide_facts_l779_779472

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def f_prime (A ω φ : ℝ) (x : ℝ) := A * ω * Real.cos (ω * x + φ)

theorem qiantang_tide_facts (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) (hφ : |φ| < Real.pi / 3)
  (h1 : f A ω φ (2 * Real.pi) = f_prime A ω φ (2 * Real.pi))
  (h2 : ∀ x, f_prime A ω φ x ≥ -4) :
  (f A ω φ (Real.pi / 3) = Real.sqrt 6 + Real.sqrt 2) ∧
  (Real.Even.fun (λ x, f_prime A ω φ (x - Real.pi / 4))) := 
  sorry

end qiantang_tide_facts_l779_779472


namespace angle_HKF_result_l779_779307

noncomputable def angle_HKF (r : ℝ) (KA : ℝ) (KN : ℝ) (KF : ℝ) (H_angle_obtuse : Bool) : ℝ :=
  let c := 7
  let d := sqrt(11) + sqrt(21)
  c * d / 28

theorem angle_HKF_result :
  (circle_radius : ℝ) = 2 →
  (KA : ℝ) = sqrt(11) - 1 →
  (KN : ℝ) = 2 →
  (NFH_obtuse : Bool) = True →
  ∃ (KF : ℝ), ∠HKF = arccos(7 * sqrt(11) + sqrt(21) / 28) :=
by
  let angle_HKF := arccos((7 * sqrt(11) + sqrt(21)) / 28)
  use 5
  sorry

end angle_HKF_result_l779_779307


namespace smallest_number_is_valid_l779_779555

/-- 
  Define the digits being used and their properties.
  Find the smallest number using all given digits exactly once without leading zero.
-/
def given_digits := [2, 4, 5, 1, 3, 7, 6, 0]

def is_valid_number (n : ℕ) : Prop :=
  n ∈ {10234567}

theorem smallest_number_is_valid : is_valid_number 10234567 := 
begin
  sorry
end

end smallest_number_is_valid_l779_779555


namespace division_remainder_l779_779136

theorem division_remainder 
  (R D Q : ℕ) 
  (h1 : D = 3 * Q)
  (h2 : D = 3 * R + 3)
  (h3 : 113 = D * Q + R) : R = 5 :=
sorry

end division_remainder_l779_779136


namespace complex_coord_l779_779127

variable (z : ℂ)

theorem complex_coord (z : ℂ) (h : (z - 2 * complex.I) * (1 + complex.I) = complex.I) : z = 1/2 + 5/2 * complex.I := by
  sorry

end complex_coord_l779_779127


namespace cotangent_tangent_identity_l779_779933

theorem cotangent_tangent_identity (x : ℝ) 
    (h : ∀ θ : ℝ, cot θ - 2 * cot (2 * θ) = tan θ) : 
    cot x + 2 * cot (2 * x) + 4 * cot (4 * x) + 8 * tan (8 * x) = cot x :=
by
  sorry

end cotangent_tangent_identity_l779_779933


namespace mom_twice_alex_l779_779914

-- Definitions based on the conditions
def alex_age_in_2010 : ℕ := 10
def mom_age_in_2010 : ℕ := 5 * alex_age_in_2010
def future_years_after_2010 (x : ℕ) : ℕ := 2010 + x

-- Defining the ages in the future year
def alex_age_future (x : ℕ) : ℕ := alex_age_in_2010 + x
def mom_age_future (x : ℕ) : ℕ := mom_age_in_2010 + x

-- The theorem to prove
theorem mom_twice_alex (x : ℕ) (h : mom_age_future x = 2 * alex_age_future x) : future_years_after_2010 x = 2040 :=
  by
  sorry

end mom_twice_alex_l779_779914


namespace decreasing_interval_f_l779_779589

noncomputable def f (x : ℝ) : ℝ := x^2 * real.log x

noncomputable def f' (x : ℝ) : ℝ := 2 * x * real.log x + x

theorem decreasing_interval_f :
  {x : ℝ | 0 < x ∧ f' x < 0} = {x : ℝ | 0 < x ∧ x < real.sqrt real.exp 1 / real.exp 1} :=
by
  sorry

end decreasing_interval_f_l779_779589


namespace plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l779_779706

theorem plan_Y_cheaper_than_X (x : ℕ) : 
  ∃ x, 2500 + 7 * x < 15 * x ∧ ∀ y, y < x → ¬ (2500 + 7 * y < 15 * y) := 
sorry

theorem plan_Z_cheaper_than_X (x : ℕ) : 
  ∃ x, 3000 + 6 * x < 15 * x ∧ ∀ y, y < x → ¬ (3000 + 6 * y < 15 * y) := 
sorry

end plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l779_779706


namespace average_value_correct_l779_779023

noncomputable def average_value (k z : ℝ) : ℝ :=
  (k + 2 * k * z + 4 * k * z + 8 * k * z + 16 * k * z) / 5

theorem average_value_correct (k z : ℝ) :
  average_value k z = (k * (1 + 30 * z)) / 5 := by
  sorry

end average_value_correct_l779_779023


namespace andrew_friends_brought_food_l779_779209

theorem andrew_friends_brought_food (slices_per_friend total_slices : ℕ) (h1 : slices_per_friend = 4) (h2 : total_slices = 16) :
  total_slices / slices_per_friend = 4 :=
by
  sorry

end andrew_friends_brought_food_l779_779209


namespace negation_example_l779_779226

theorem negation_example : ¬(∀ x : ℝ, x^2 + |x| ≥ 0) ↔ ∃ x : ℝ, x^2 + |x| < 0 :=
by
  sorry

end negation_example_l779_779226


namespace length_segment_AB_l779_779197

theorem length_segment_AB (A B : ℝ) (hA : A = -5) (hB : B = 2) : |A - B| = 7 :=
by
  sorry

end length_segment_AB_l779_779197


namespace prob_three_vertices_plane_contains_points_inside_l779_779243

theorem prob_three_vertices_plane_contains_points_inside (V : Set Point)
  (regular_tetrahedron : ∀ (S : Finset Point), S.card = 4 → RegularTetrahedron S) :
  let vertices : Finset Point := {v₁, v₂, v₃, v₄} in
  (∀ (S : Finset Point), S.card = 3 → S ⊆ vertices → no_points_inside (RegularTetrahedron.span S) (interior regular_tetrahedron)) →
  probability (plane_contains_points_inside regular_tetrahedron) = 0 :=
by
  sorry

end prob_three_vertices_plane_contains_points_inside_l779_779243


namespace student_test_ratio_l779_779312

theorem student_test_ratio :
  ∀ (total_questions correct_responses : ℕ),
  total_questions = 100 →
  correct_responses = 93 →
  (total_questions - correct_responses) / correct_responses = 7 / 93 :=
by
  intros total_questions correct_responses h_total_questions h_correct_responses
  sorry

end student_test_ratio_l779_779312


namespace find_p_l779_779269

variables (m n p : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x = y / 3 - 2 / 5

theorem find_p
  (h1 : line_equation m n)
  (h2 : line_equation (m + p) (n + 9)) :
  p = 3 :=
by
  sorry

end find_p_l779_779269


namespace problem_1_problem_2_l779_779092

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : f 0 a b c = -1)
  (h₂ : ∀ x : ℝ, f x a b c ≥ x - 1)
  (h₃ : ∀ x : ℝ, f (-1/2 + x) a b c = f (-1/2 - x) a b c) :
  (f x 1 1 (-1) = x^2 + x - 1) :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log (1/2) (f a a (-1) ^ x)

theorem problem_2 (a : ℝ) :
  (∃ a : ℝ, g (a) a < g (a - 1) a) ↔ (a < -2 ∨ a > 1) :=
sorry

end problem_1_problem_2_l779_779092


namespace dog_speed_correct_l779_779713

-- Definitions of the conditions
def football_field_length_yards : ℕ := 200
def total_football_fields : ℕ := 6
def yards_to_feet_conversion : ℕ := 3
def time_to_fetch_minutes : ℕ := 9

-- The goal is to find the dog's speed in feet per minute
def dog_speed_feet_per_minute : ℕ :=
  (total_football_fields * football_field_length_yards * yards_to_feet_conversion) / time_to_fetch_minutes

-- Statement for the proof
theorem dog_speed_correct : dog_speed_feet_per_minute = 400 := by
  sorry

end dog_speed_correct_l779_779713


namespace bisector_passes_through_fixed_point_l779_779711

noncomputable def Circle := { c : ℝ × ℝ // sqrt ((c.1)^2 + (c.2)^2) = R } -- definition of a circle with radius R

theorem bisector_passes_through_fixed_point
(p q o a b : ℝ × ℝ)
(h_circle : Circle o ∧ Circle a ∧ Circle b)
(h_diameter : a ≠ b ∧ dist a b = 2 * R)
(h_point_on_semi_circle : ∀ P, P ∈ semi_circle o a b)
(h_perpendicular : ∀ P, ∃ Q, Q ∈ extend_perpendicular PQ AB) :
  ∃ C, ∀ P, angle_bisector (angle o P Q) C :=
sorry

end bisector_passes_through_fixed_point_l779_779711


namespace fibonacci_divisibility_l779_779886

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility :
  3 ^ 2023 ∣ ∑ n in Finset.range (2023 - 1), (3 ^ (n + 2)) * fibonacci (2 * (n + 2)) :=
by sorry

end fibonacci_divisibility_l779_779886


namespace sum_of_perpendiculars_eq_l779_779303

-- Definition of the problem given the conditions
def equilateral_triangle_side_length : ℝ := 10

-- The main hypothesis is about a randomly chosen point inside the triangle
variable {P : Type} [Point_in_triangle : Point P]

-- The sum of the lengths of the perpendiculars from the point to the sides of the triangle
def sum_of_perpendiculars (P_inside : ∀ x : P, Point_in_triangle x) : ℝ :=
  ∑ (f : P → ℝ), f(P_inside) -- This function will depend on the specific implementation for the lengths of perpendiculars

-- The theorem we aim to prove
theorem sum_of_perpendiculars_eq :
  ∀ (P_inside : ∀ x : P, Point_in_triangle x), sum_of_perpendiculars P_inside = 5 * real.sqrt 3 :=
sorry

end sum_of_perpendiculars_eq_l779_779303


namespace range_g_f_comp_g_solve_inequality_l779_779786

/-- Given a function f(x) = 2 * x ^ 2 - 1 --/
def f (x : ℝ) : ℝ := 2 * x ^ 2 - 1

/-- Given a piecewise function g(x) --/
def g (x : ℝ) : ℝ := 
if x >= 0 then 2 * x - 1 else 2 - x

/-- Prove that the range of g(x) is [-1, +∞) --/
theorem range_g : 
  (∀ y : ℝ, ∃ x : ℝ, y = g x) ↔ ∃ y : ℝ, y >= -1 := sorry

/-- Prove that f(g(x)) is as defined --/
theorem f_comp_g (x : ℝ) : 
  f (g x) = 
    if x >= 0 then 8 * x ^ 2 - 8 * x + 1 
    else 2 * x ^ 2 - 8 * x + 7 := sorry

/-- Prove that the inequality f(x) > g(x) holds in the prescribed intervals --/
theorem solve_inequality : 
  {x : ℝ | f x > g x} 
  = {x : ℝ | x < -3 / 2} ∪ {x : ℝ | x > 1} := sorry

end range_g_f_comp_g_solve_inequality_l779_779786


namespace geometric_series_sum_l779_779424

theorem geometric_series_sum (a : ℝ) (q : ℝ) (a₁ : ℝ) 
  (h1 : a₁ = 1)
  (h2 : q = a - (3/2))
  (h3 : |q| < 1)
  (h4 : a = a₁ / (1 - q)) :
  a = 2 :=
sorry

end geometric_series_sum_l779_779424


namespace mass_percentage_of_O_in_CaCO3_l779_779252

-- Assuming the given conditions as definitions
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def formula_CaCO3 : (ℕ × ℝ) := (1, molar_mass_Ca) -- 1 atom of Calcium
def formula_CaCO3_C : (ℕ × ℝ) := (1, molar_mass_C) -- 1 atom of Carbon
def formula_CaCO3_O : (ℕ × ℝ) := (3, molar_mass_O) -- 3 atoms of Oxygen

-- Desired result
def mass_percentage_O_CaCO3 : ℝ := 47.95

-- The theorem statement to be proven
theorem mass_percentage_of_O_in_CaCO3 :
  let molar_mass_CaCO3 := formula_CaCO3.2 + formula_CaCO3_C.2 + (formula_CaCO3_O.1 * formula_CaCO3_O.2)
  let mass_percentage_O := (formula_CaCO3_O.1 * formula_CaCO3_O.2 / molar_mass_CaCO3) * 100
  mass_percentage_O = mass_percentage_O_CaCO3 :=
by
  sorry

end mass_percentage_of_O_in_CaCO3_l779_779252


namespace binom_60_12_has_digit_B_equal_7_l779_779138

def factorial : ℕ → ℕ 
| 0 := 1
| (n + 1) := (n + 1) * factorial n

def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_60_12_has_digit_B_equal_7 :
  ∃ B : ℕ, binom 60 12 = 1BCB529080B ∧ B = 7 :=
by
  sorry

#check binom_60_12_has_digit_B_equal_7

end binom_60_12_has_digit_B_equal_7_l779_779138


namespace xy_equals_twelve_l779_779120

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by
  sorry

end xy_equals_twelve_l779_779120


namespace cot_arccots_sum_correct_l779_779176

def polynomial : Polynomial ℂ :=
  X^10 - 2 * X^9 + 4 * X^8 - 8 * X^7 + 16 * X^6 - 32 * X^5 + 64 * X^4 - 128 * X^3 + 256 * X^2 - 512 * X + 1024

noncomputable def roots : List ℂ := (polynomial.roots).to_list

noncomputable def arccot (z : ℂ) : ℂ := Complex.cot z⁻¹

noncomputable def arccots_sum : ℂ := roots.map arccot |>.sum

noncomputable def cot_arccots_sum : ℂ := Complex.cot arccots_sum

theorem cot_arccots_sum_correct :
    cot_arccots_sum = 819 / 410 :=
by
  sorry

end cot_arccots_sum_correct_l779_779176


namespace mia_high_school_has_2000_students_l779_779186

variables (M Z : ℕ)

def mia_high_school_students : Prop :=
  M = 4 * Z ∧ M + Z = 2500

theorem mia_high_school_has_2000_students (h : mia_high_school_students M Z) : 
  M = 2000 := by
  sorry

end mia_high_school_has_2000_students_l779_779186


namespace g_one_eq_l779_779536

noncomputable def g (f : ℕ → ℝ) (x : ℝ) : ℝ :=
  -- Assuming the polynomial's leading coefficient is 1 and roots are reciprocals
  let roots := [1 / p, 1 / q, 1 / r, 1 / s] in
  (x - roots[0]) * (x - roots[1]) * (x - roots[2]) * (x - roots[3])

theorem g_one_eq (a b c d : ℝ) (h : a < b < c < d) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
(f : ℝ → ℝ) (hf : f = λ x, x^4 + a*x^3 + b*x^2 + c*x + d) :
  g f 1 = (1 + a + b + c + d) / d :=
by
  sorry

end g_one_eq_l779_779536


namespace incorrect_method_B_l779_779259

-- Definitions for conditions
def correct_method_A (P : α → Prop) (L : set α) : Prop :=
  ∀ x, (L x ↔ P x) 

def correct_method_B (P : α → Prop) (L : set α) : Prop :=
  (∀ x, L x → P x) ∧ (∃ x, ¬L x ∧ P x)

def correct_method_C (P : α → Prop) (L : set α) : Prop :=
  ∀ x, (P x ↔ L x)

def correct_method_D (P : α → Prop) (L : set α) : Prop :=
  ∀ x, (¬L x ↔ ¬P x)

def correct_method_E (P : α → Prop) (L : set α) : Prop :=
  ∀ x, (P x ↔ L x)

-- The theorem to be proven
theorem incorrect_method_B (P : α → Prop) (L : set α) :
  ¬ correct_method_B P L := sorry

end incorrect_method_B_l779_779259


namespace market_spending_l779_779552

theorem market_spending 
  (mildred spent: ℝ) 
  (candice spent: ℝ) 
  (joseph's actual : ℝ) 
  (total given: ℝ): mildred_spent = 25 ∧ candice_spent = 35 ∧ joseph_actual = 45 ∧ total_given = 150 → 
(remained: ℝ): remained = total_given - (mildred_spent + candice_spent + joseph_actual) :=
by
  intro h
  rcases h with ⟨hm, hc, hj, ht⟩
  rw [hm, hc, hj, ht]
  have eq1: (total_given - (mildred_spent + candice_spent + joseph_actual)) = (150 - (25 + 35 + 45)) := by rfl
  rw eq1
  linarith
  sorry -- To indicate the need for proof completion

end market_spending_l779_779552


namespace dot_product_sum_l779_779171

variables {u v w : ℝ → ℝ} -- Define our vectors as functions from ℝ to ℝ.

-- Given conditions
axiom norm_u : ∥u∥ = 2
axiom norm_v : ∥v∥ = 3
axiom norm_w : ∥w∥ = 6
axiom sum_zero : u + v + w = 0
axiom angle_120_uv : ∀ x, (x ∈ u.range ∪ v.range) → (2 ≤ x) ∧ (x ≤ 2)
axiom angle_120_uw : ∀ x, (x ∈ u.range ∪ w.range) → (2 ≤ x) ∧ (x ≤ 2)
axiom angle_120_vw : ∀ x, (x ∈ v.range ∪ w.range) → (2 ≤ x) ∧ (x ≤ 2)

theorem dot_product_sum : u.dot v + u.dot w + v.dot w = -18 := 
by
 { 
     -- Here the full proof would be written
    sorry 
}

end dot_product_sum_l779_779171


namespace vertex_of_parabola_l779_779606

theorem vertex_of_parabola (c d : ℝ) (h₁ : ∀ x, -x^2 + c*x + d ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) : 
  ∃ v : ℝ × ℝ, v = (3, 16) :=
by
  sorry

end vertex_of_parabola_l779_779606


namespace caramel_apple_cost_proof_l779_779187

-- Definition of the cost of the ice cream cone
def ice_cream_cone_cost : ℕ := 15

-- Definition that the apple costs 10 cents more than the ice cream cone
def caramel_apple_cost (x : ℕ) : ℕ := x + 10

-- Theorem stating the cost of the caramel apple given the conditions
theorem caramel_apple_cost_proof : caramel_apple_cost ice_cream_cone_cost = 25 :=
by
  unfold caramel_apple_cost
  rw ice_cream_cone_cost
  norm_num

end caramel_apple_cost_proof_l779_779187


namespace find_p_q_for_tangent_circles_l779_779908

theorem find_p_q_for_tangent_circles :
  let w1 := fun x y => x^2 + y^2 + 12*x + 10*y - 50 = 0
  let w2 := fun x y => x^2 + y^2 - 8*x + 10*y + 60 = 0
  let is_tangent (f g : ℝ → ℝ → Prop) := 
    ∃ x y r, f x y ∧ g x y ∧ (11 - r = real.sqrt ((x + 6)^2 + (y + 5)^2)) ∧ (r + 7 = real.sqrt ((x - 4)^2 + (y - 5)^2))
  let smallest_pos_a (λ : ℝ) := λ > 0 ∧ ∃ k, k = λ*λ ∧ gcd (numerator k) (denominator k) = 1
  in
    is_tangent w1 w2 →
    ∃ m : ℝ, smallest_pos_a m ∧ m*m = 77 / 100 ∧ numerator (m*m) + denominator (m*m) = 177 :=
sorry

end find_p_q_for_tangent_circles_l779_779908


namespace build_wall_in_days_l779_779502

noncomputable def constant_k : ℝ := 20 * 6
def inverse_proportion (m : ℝ) (d : ℝ) : Prop := m * d = constant_k

theorem build_wall_in_days (d : ℝ) : inverse_proportion 30 d → d = 4.0 :=
by
  intros h
  have : d = constant_k / 30,
    sorry
  rw this,
  exact (by norm_num : 120 / 30 = 4.0)

end build_wall_in_days_l779_779502


namespace inequality_holds_l779_779061

noncomputable def f (x : ℝ) : ℝ := (f' 1) / 2 * exp (2 * x - 2) + x ^ 2 - 2 * f 0 * x

axiom g : ℝ → ℝ
axiom g' : ℝ → ℝ

axiom g_inequality : ∀ x : ℝ, g' x + 2 * g x < 0

lemma f_two_eq_e_four : f 2 = Real.exp 4 := 
  sorry

theorem inequality_holds : g 2015 > f 2 * g 2017 :=
  sorry

end inequality_holds_l779_779061


namespace local_food_drive_correct_l779_779680

def local_food_drive_condition1 (R J x : ℕ) : Prop :=
  J = 2 * R + x

def local_food_drive_condition2 (J : ℕ) : Prop :=
  4 * J = 100

def local_food_drive_condition3 (R J : ℕ) : Prop :=
  R + J = 35

theorem local_food_drive_correct (R J x : ℕ)
  (h1 : local_food_drive_condition1 R J x)
  (h2 : local_food_drive_condition2 J)
  (h3 : local_food_drive_condition3 R J) :
  x = 5 :=
by
  sorry

end local_food_drive_correct_l779_779680


namespace line_intersection_difference_l779_779448

theorem line_intersection_difference :
  ∃ m n : ℤ, 
    (0 = 2 * m - 4) ∧ (n = -4) ∧ (m - n = 6) :=
by
  have h1 : 0 = 2 * m - 4,
  have h2 : n = -4,
  show m - n = 6,
  sorry

end line_intersection_difference_l779_779448


namespace remainder_T_2015_mod_10_l779_779038

-- Define the number of sequences with no more than two consecutive identical letters
noncomputable def T : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 6
| n + 1 => (T n + T (n - 1) + T (n - 2) + T (n - 3))  -- hypothetically following initial conditions pattern

theorem remainder_T_2015_mod_10 : T 2015 % 10 = 6 :=
by 
  sorry

end remainder_T_2015_mod_10_l779_779038


namespace Jaron_prize_points_l779_779874

def points_bunnies (bunnies: Nat) (points_per_bunny: Nat) : Nat :=
  bunnies * points_per_bunny

def points_snickers (snickers: Nat) (points_per_snicker: Nat) : Nat :=
  snickers * points_per_snicker

def total_points (bunny_points: Nat) (snicker_points: Nat) : Nat :=
  bunny_points + snicker_points

theorem Jaron_prize_points :
  let bunnies := 8
  let points_per_bunny := 100
  let snickers := 48
  let points_per_snicker := 25
  let bunny_points := points_bunnies bunnies points_per_bunny
  let snicker_points := points_snickers snickers points_per_snicker
  total_points bunny_points snicker_points = 2000 := 
by
  sorry

end Jaron_prize_points_l779_779874


namespace inequality_solution_set_l779_779359

open Real

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 3) > 5 / x + 11 / 6) ↔ x ∈ Ioo (-11 / 10) (-1) ∪ Ioo 0 3 :=
by
  sorry

end inequality_solution_set_l779_779359


namespace AP_perpendicular_OH_l779_779420

variables {A B C E F M N P O H : Type*}
variables [metric_space A] [inner_product_space ℝ A]
variables [metric_space B] [inner_product_space ℝ B]
variables [metric_space C] [inner_product_space ℝ C]
variables [metric_space E] [inner_product_space ℝ E]
variables [metric_space F] [inner_product_space ℝ F]
variables [metric_space M] [inner_product_space ℝ M]
variables [metric_space N] [inner_product_space ℝ N]
variables [metric_space P] [inner_product_space ℝ P]
variables [metric_space O] [inner_product_space ℝ O]
variables [metric_space H] [inner_product_space ℝ H]

-- Define midpoints
def midpoint (x y : Type*) [metric_space x] [metric_space y] : Type* := sorry

axiom E_def : E = midpoint A B
axiom F_def : F = midpoint A C

-- Define altitudes
def altitude (x y z : Type*) [metric_space x] [metric_space y] [metric_space z] : Type* := sorry

axiom CM_def : altitude C A B = M
axiom BN_def : altitude B A C = N

-- Define intersection
def intersection (x y z : Type*) [metric_space x] [metric_space y] [metric_space z] : Type* := sorry

axiom P_def : P = intersection (midpoint A B) (altitude C A B) (midpoint A C) (altitude B A C)

-- Circumcenter and orthocenter definitions
def circumcenter (x y z : Type*) [metric_space x] [metric_space y] [metric_space z] : Type* := sorry
def orthocenter (x y z : Type*) [metric_space x] [metric_space y] [metric_space z] : Type* := sorry

axiom O_def : O = circumcenter A B C
axiom H_def : H = orthocenter A B C

-- Perpendicularity definition
def perpendicular (x y : Type*) [metric_space x] [metric_space y] : Prop := sorry

-- The theorem to be proved
theorem AP_perpendicular_OH : perpendicular (line A P) (line O H) := sorry

end AP_perpendicular_OH_l779_779420


namespace circle_equation_tangent_to_lines_l779_779421

def tangent_to_lines (C : ℝ × ℝ → ℝ → Prop) : Prop :=
  ∃ (a r : ℝ), 
    (C (a, -a) r) ∧ 
    (abs (2 * a) / real.sqrt 2 = r) ∧ 
    (abs (2 * a - 4) / real.sqrt 2 = r)

def lies_on_line (p : ℝ × ℝ) (L : ℝ → ℝ) : Prop :=
  L p.1 = p.2

def equation_of_circle (C : ℝ × ℝ → ℝ → Prop) : Set (ℝ × ℝ) :=
  { (x, y) | ∀ a r, (C (a, -a) r) → (x - a)^2 + (y + a)^2 = r^2 }

theorem circle_equation_tangent_to_lines
  (C : ℝ × ℝ → ℝ → Prop)
  (h1 : tangent_to_lines C)
  (h2 : ∃ a, lies_on_line (a, -a) (λ p => -p)) :
  equation_of_circle C = { (x, y) | (x - 1)^2 + (y + 1)^2 = 2 } :=
by
  sorry

end circle_equation_tangent_to_lines_l779_779421


namespace correct_conclusions_correct_conclusions_independent_l779_779794

variables {Ω : Type*} {P : MeasureTheory.Measure Ω}
variables {A B : Set Ω}

-- Conditions
variable (PA_pos : 0 < P(A))
variable (PB_pos : 0 < P(B))

-- Statement of options B and C
theorem correct_conclusions (PB_given_A : MeasureTheory.condProb P B A + P (Set.compl B) = 1) :
  MeasureTheory.indep_sets {A} {B} P := 
by sorry

theorem correct_conclusions_independent (independent_A_B : MeasureTheory.indep_sets {A} {B} P) (PA : P(A) = 0.6) (PB : 0 < P(B)):
  MeasureTheory.condProb P A B = 0.6 := 
by sorry

end correct_conclusions_correct_conclusions_independent_l779_779794


namespace high_school_heralds_games_lost_percentage_l779_779964

theorem high_school_heralds_games_lost_percentage :
  ∀ (won lost : ℕ) (total_games : ℕ) (ratio_won_lost : ℚ),
    ratio_won_lost = 8 / 5 →
    total_games = won + lost →
    total_games = 52 →
    ∃ percentage_lost : ℚ, percentage_lost = (lost * 100 / total_games) ∧ percentage_lost ≈ 38 :=
by
  sorry

end high_school_heralds_games_lost_percentage_l779_779964


namespace smallest_n_value_l779_779904

theorem smallest_n_value (n : ℕ) (x : Fin n → ℝ) (h₀ : ∀ i, 0 ≤ x i)
  (h₁ : (∑ i, x i) = 1) (h₂ : (∑ i, (x i)^2) ≤ 1/50) : 50 ≤ n := by
  sorry

end smallest_n_value_l779_779904


namespace quotient_of_second_largest_and_second_smallest_is_one_l779_779971

theorem quotient_of_second_largest_and_second_smallest_is_one
  (a b c : ℕ)
  (h1 : a = 10) 
  (h2 : b = 11) 
  (h3 : c = 12) 
  (h4 : list.sort(≤) [a, b, c] = [10, 11, 12])
  :
  ((list.sort(≤) [a, b, c]).nth 1).get_or_else 0 / ((list.sort(≤) [a, b, c]).nth 1).get_or_else 0 = 1 :=
sorry

end quotient_of_second_largest_and_second_smallest_is_one_l779_779971


namespace find_value_of_expression_l779_779070

variable {a b c d x : ℝ}

-- Conditions
def opposites (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def abs_three (x : ℝ) : Prop := |x| = 3

-- Proof
theorem find_value_of_expression (h1 : opposites a b) (h2 : reciprocals c d) 
  (h3 : abs_three x) : ∃ res : ℝ, (res = 3 ∨ res = -3) ∧ res = 10 * a + 10 * b + c * d * x :=
by
  sorry

end find_value_of_expression_l779_779070


namespace distinct_prime_divisors_l779_779166

theorem distinct_prime_divisors (a : ℤ) (n : ℕ) (h₁ : a > 3) (h₂ : Odd a) (h₃ : n > 0) : 
  ∃ (p : Finset ℤ), p.card ≥ n + 1 ∧ ∀ q ∈ p, Prime q ∧ q ∣ (a ^ (2 ^ n) - 1) :=
sorry

end distinct_prime_divisors_l779_779166


namespace ellipse_standard_eq_line_eq_l779_779057

-- Define the conditions for part (I)
variables (a b : ℝ) (h : a > b) (hb : b > 0)
variables (h1 : 1 / a ^ 2 + 2 / (3 * b ^ 2) = 1)
variables (e : ℝ) (he : e = sqrt 6 / 3)
variables (c : ℝ) (hc : c = e * a)
variables (pass_point : (1 : ℝ, sqrt 6 / 3) ∈ {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1})

-- Prove the standard equation of the ellipse
theorem ellipse_standard_eq (ha : a ^ 2 - b ^ 2 = c ^ 2) : 
  (a = sqrt 3) ∧ (b = 1) ∧ (∀ x y, x ^ 2 / (sqrt 3) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 / 3 + y ^ 2 = 1) :=
sorry

-- Define the conditions for part (II)
variables (Q : ℝ × ℝ) (l : ℝ → ℝ) (hQ : Q = (0, 3 / 2))
variables (A : ℝ × ℝ) (hA : A = (0, -1))
variables (M N : ℝ × ℝ)
variables (intersectMN : M ∈ {p : ℝ × ℝ | p.1 ^ 2 / 3 + p.2 ^ 2 = 1} ∧ N ∈ {p : ℝ × ℝ | p.1 ^ 2 / 3 + p.2 ^ 2 = 1})
variables (l_point : ∀ (x : ℝ), (x, l x) = Q ∧ (x, l x) = M ∨ (x, l x) = N)
variables (equal_dist : dist A M = dist A N)

-- Prove the equation of the line l
theorem line_eq :
  (l = (λ x, sqrt 6 / 3 * x + 3 / 2) ∨ l = (λ x, - sqrt 6 / 3 * x + 3 / 2)) :=
sorry

end ellipse_standard_eq_line_eq_l779_779057


namespace exists_function_f_l779_779927

theorem exists_function_f :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n * n :=
by
  sorry

end exists_function_f_l779_779927


namespace equilateral_triangle_area_nearest_integer_l779_779496

-- Define the main problem
theorem equilateral_triangle_area_nearest_integer
  (A B C P : Point)
  (h_eq_triangle : EquilateralTriangle A B C)
  (h_interior : InteriorPoint P A B C)
  (PA PB PC : ℝ)
  (h_PA : PA = 7)
  (h_PB : PB = 6)
  (h_PC : PC = 5):
  nearest_integer_area (triangle_area A B C) = 43 := 
sorry

end equilateral_triangle_area_nearest_integer_l779_779496


namespace number_of_goats_l779_779476

theorem number_of_goats
  (total_animals : ℕ)
  (number_of_cows : ℕ)
  (number_of_sheep : ℕ)
  (h1 : total_animals = 200)
  (h2 : number_of_cows = 40)
  (h3 : number_of_sheep = 56) :
  total_animals - (number_of_cows + number_of_sheep) = 104 :=
by
  rw [h1, h2, h3]
  rfl

end number_of_goats_l779_779476


namespace complex_modulus_l779_779083

open Complex

def z : ℂ := (2 : ℂ) / (1 + I) + (1 - I) ^ 2

theorem complex_modulus :
  |z| = Real.sqrt 10 := by
  sorry

end complex_modulus_l779_779083


namespace max_x_plus_reciprocal_x_l779_779966

variables (n : ℕ) (x : ℝ)
open_locale big_operators

-- Assumptions
def sum_is_1501 {xs : fin n → ℝ} := ∑ i, xs i = 1501
def reciprocals_sum_is_1501 {xs : fin n → ℝ} := ∑ i, (1 / xs i) = 1501
def valid_numbers (n : ℕ) := n = 1500

-- The theorem statement
theorem max_x_plus_reciprocal_x 
  (xs : fin 1500 → ℝ) 
  (h_sum : sum_is_1501 xs) 
  (h_reciprocal_sum : reciprocals_sum_is_1501 xs) : 
  ∃ x, x ∈ set.range xs ∧ x + (1 / x) ≤ 5001 / 1501 :=
begin
  sorry
end

end max_x_plus_reciprocal_x_l779_779966


namespace polynomial_roots_sum_l779_779897

theorem polynomial_roots_sum {p q r : ℝ} (v : ℂ)
  (hroots : ∀ z : ℂ, (z = v - 2 * complex.I ∨ z = 2 * v + 2 ∨ z = v + 10 * complex.I) ↔ (z ^ 3 + p * z ^ 2 + q * z + r = 0)) :
  p + q + r = 6 := by
  sorry

end polynomial_roots_sum_l779_779897


namespace sequence_bound_l779_779885

noncomputable def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ k, a k - 2 * a (k + 1) + a (k + 2) ≥ 0

noncomputable def sum_property (a : ℕ → ℝ) : Prop :=
  ∀ k, (∑ j in Finset.range (k + 1), a j) ≤ 1

noncomputable def non_negativity_property (a : ℕ → ℝ) : Prop :=
  ∀ k, a k ≥ 0

theorem sequence_bound (a : ℕ → ℝ) :
  sequence_property a →
  sum_property a →
  non_negativity_property a →
  ∀ k, 0 ≤ a k - a (k + 1) ∧ a k - a (k + 1) < 2 / (k * k) :=
by
  intros h_seq h_sum h_nonneg k
  sorry

end sequence_bound_l779_779885


namespace person_Y_share_l779_779332

theorem person_Y_share (total_amount : ℝ) (r1 r2 r3 r4 r5 : ℝ) (ratio_Y : ℝ) 
  (h1 : total_amount = 1390) 
  (h2 : r1 = 13) 
  (h3 : r2 = 17)
  (h4 : r3 = 23) 
  (h5 : r4 = 29) 
  (h6 : r5 = 37) 
  (h7 : ratio_Y = 29): 
  (total_amount / (r1 + r2 + r3 + r4 + r5) * ratio_Y) = 338.72 :=
by
  sorry

end person_Y_share_l779_779332


namespace program_output_l779_779084

theorem program_output (x : ℤ) : 
  (if x < 0 then -1 else if x = 0 then 0 else 1) = 1 ↔ x = 3 :=
by
  sorry

end program_output_l779_779084


namespace ratio_of_divisors_l779_779894

-- Define the given number N
def N : ℕ := 34 * 34 * 63 * 270

-- Define the function to compute the sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, if d % 2 = 1 then d else 0

-- Define the function to compute the sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, if d % 2 = 0 then d else 0

-- Define the statement of the problem
theorem ratio_of_divisors : (sum_odd_divisors N) / (sum_even_divisors N) = 1 / 14 :=
by
  sorry

end ratio_of_divisors_l779_779894


namespace other_tables_have_3_legs_l779_779672

-- Define the given conditions
variables (total_tables four_legged_tables : ℕ)
variables (total_legs legs_four_legged_tables : ℕ)

-- State the conditions as Lean definitions
def dealer_conditions :=
  total_tables = 36 ∧
  four_legged_tables = 16 ∧
  total_legs = 124 ∧
  legs_four_legged_tables = 4 * four_legged_tables

-- Main theorem to prove the number of legs on the other tables
theorem other_tables_have_3_legs (cond : dealer_conditions total_tables four_legged_tables total_legs legs_four_legged_tables) :
  let other_tables := total_tables - four_legged_tables in
  let other_legs := total_legs - legs_four_legged_tables in
  other_tables > 0 →
  other_legs % other_tables = 0 →
  other_legs / other_tables = 3 :=
sorry

end other_tables_have_3_legs_l779_779672


namespace travel_with_decreasing_ticket_prices_l779_779858

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ℕ × ℕ → ℕ)
  (distinct_prices : ∀ ⦃x y : ℕ × ℕ⦄, x ≠ y → train_prices x ≠ train_prices y)
  (symmetric_prices : ∀ x y : ℕ, train_prices (x, y) = train_prices (y, x)) :
  ∃ start_city : ℕ, ∃ route : List (ℕ × ℕ),
    length route = n-1 ∧ 
    (∀ i, i < route.length - 1 → train_prices (route.nth i).get_or_else (0,0) > train_prices (route.nth (i+1)).get_or_else (0,0)) :=
begin
  -- Proof goes here
  sorry
end

end travel_with_decreasing_ticket_prices_l779_779858


namespace problem_l779_779818

theorem problem (a b : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 2 → ax^2 - bx + 2 < 0) : a + b = 4 :=
sorry

end problem_l779_779818


namespace no_positive_integer_solutions_l779_779348

theorem no_positive_integer_solutions (x y z : ℕ) (h_cond : x^2 + y^2 = 7 * z^2) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_positive_integer_solutions_l779_779348


namespace hyperbola_equation_ellipse_equation_x_axis_ellipse_equation_y_axis_l779_779011

theorem hyperbola_equation (λ : ℝ) (A : ℝ × ℝ) (Hx : λ = -1/4) (Hc : A = (2*Real.sqrt 3, -3)) :
  (λ * (x ^ 2 / 16 - y ^ 2 / 9) = 1) ↔ (y ^ 2 / (9/4) - x ^ 2 / 4 = 1) :=
sorry

theorem ellipse_equation_x_axis (t : ℝ) (P : ℝ × ℝ) (Hc : P = (2, -Real.sqrt 3)) :
  (t = 2) → ((x ^ 2 / 4 + y ^ 2 / 3 = t) ↔ (x ^ 2 / 8 + y ^ 2 / 6 = 1)) :=
sorry

theorem ellipse_equation_y_axis (λ : ℝ) (P : ℝ × ℝ) (Hc : P = (2, -Real.sqrt 3)) :
  (λ = 25/12) → ((y ^ 2 / 4 + x ^ 2 / 3 = λ) ↔ (y ^ 2 / (25/3) + x ^ 2 / (25/4) = 1)) :=
sorry

end hyperbola_equation_ellipse_equation_x_axis_ellipse_equation_y_axis_l779_779011


namespace length_of_parallel_segments_l779_779975

theorem length_of_parallel_segments (a b c : ℝ) 
    (h1 : ∀ (x : ℝ), ∃ (points : set (ℝ × ℝ)), 
    points ⊆ set.univ ∧ 
    (∀ p ∈ points, ∃ q r ∈ points, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
    parallel (segment p q) (segment (0,0) (a,b)) ∧ 
    parallel (segment q r) (segment (0,a) (c,0)) ∧ 
    parallel (segment r p) (segment (b,0) (0,c)) ∧ 
    length (segment p q) = x ∧ 
    length (segment q r) = x ∧ 
    length (segment r p) = x)) : 
    x = 2 * a * b * c / (a * b + a * c + b * c) :=
sorry

end length_of_parallel_segments_l779_779975


namespace problem_equivalent_l779_779557

def line1 := ∀ x y : ℝ, 3 * x + 4 * y - 2 = 0
def line2 := ∀ x y : ℝ, 2 * x + y + 2 = 0
def line3 := ∀ x y : ℝ, 3 * x - 2 * y + 4 = 0
def point_P (x y : ℝ) := 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 10 = 0
def orthogonal_line1 (x y : ℝ) := 4 * x - 3 * y = 0
def orthogonal_line2 (x y : ℝ) := x - 2 * y = 0
def is_right_angle_triangle (ℓ : ∀ x y : ℝ, Prop) : Prop :=
  ∀ x y : ℝ, ℓ x y → (line1 x y → False) ∨ (line2 x y → False)

theorem problem_equivalent :
  (∃ x y, point_P x y ∧ x = -2 ∧ y = 2) ∧
  (parallel_line (-2) 2) ∧
  (is_right_angle_triangle orthogonal_line1 ∨ is_right_angle_triangle orthogonal_line2) :=
by
  sorry

end problem_equivalent_l779_779557


namespace geom_series_sum_ratio_l779_779898

noncomputable def geom_series_sum (a1 q : ℚ) (n : ℕ) : ℚ :=
  a1 * (q^n - 1) / (q - 1)

theorem geom_series_sum_ratio (a1 a4 : ℚ) (q : ℚ)
  (h1 : 8 * a1 = a4)
  (h2 : q^3 = 8) :
  let S4 := geom_series_sum a1 q 4
  let S2 := geom_series_sum a1 q 2
  in S4 / S2 = 5 :=
by
  sorry

end geom_series_sum_ratio_l779_779898


namespace range_of_f_l779_779762

def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  set_of (λ y, ∃ x, f(x) = y) = {y : ℝ | y ≠ 3} :=
by
  sorry

end range_of_f_l779_779762


namespace profit_ratio_l779_779647

theorem profit_ratio (p_investment q_investment : ℝ) (h₁ : p_investment = 50000) (h₂ : q_investment = 66666.67) :
  (1 / q_investment) = (3 / 4 * 1 / p_investment) :=
by
  sorry

end profit_ratio_l779_779647


namespace gcd_lcm_product_eq_l779_779759

-- Define the numbers
def a : ℕ := 10
def b : ℕ := 15

-- Define the GCD and LCM
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Proposition that needs to be proved
theorem gcd_lcm_product_eq : gcd_ab * lcm_ab = 150 :=
  by
    -- Proof would go here
    sorry

end gcd_lcm_product_eq_l779_779759


namespace problem1_problem2_l779_779817

-- Problem 1: Prove the range of k for any real number x
theorem problem1 (k : ℝ) (x : ℝ) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  1 ≤ k ∧ k < 13 :=
sorry

-- Problem 2: Prove the range of k for any x in the interval (0, 1]
theorem problem2 (k : ℝ) (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  k > -1/2 :=
sorry

end problem1_problem2_l779_779817


namespace question_eq_answer_l779_779454

theorem question_eq_answer (n : ℝ) (h : 0.25 * 0.1 * n = 15) :
  0.1 * 0.25 * n = 15 :=
by
  sorry

end question_eq_answer_l779_779454


namespace monotonically_increasing_intervals_min_and_max_values_l779_779091

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.sin (2 * x + Real.pi / 4) + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, 
    -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi → 
    f (x + 1) ≥ f x := sorry

theorem min_and_max_values :
  ∃ min max, 
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≥ min ∧ f x ≤ max) ∧ 
    (min = 0) ∧ 
    (max = Real.sqrt 2 + 1) := sorry

end monotonically_increasing_intervals_min_and_max_values_l779_779091


namespace find_s_at_3_l779_779175

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := y^2 - (y + 12)

theorem find_s_at_3 : s 3 = -6 :=
by
  sorry

end find_s_at_3_l779_779175


namespace rachel_birthday_l779_779872

theorem rachel_birthday :
  ∀ (ticket_cost : ℝ) (number_of_tickets : ℕ), ticket_cost = 44 ∧ number_of_tickets = 7 → ticket_cost * number_of_tickets = 308 :=
by
  intros ticket_cost number_of_tickets
  intro h
  cases h
  rw [h_left, h_right]
  norm_num

end rachel_birthday_l779_779872


namespace function_range_l779_779989

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x - 2 * (Real.sin x) ^ 2

theorem function_range {x : ℝ} (hx : x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 2)) :
  Set.range (f x) = Set.Icc (-5 / 2 : ℝ) (-2) := 
sorry

end function_range_l779_779989


namespace line_equation_l779_779221

theorem line_equation (x y : ℝ) :
  (∃ (m b : ℝ), m = -2 ∧ b = 3 ∧ y = m * x + b) → 2 * x + y - 3 = 0 :=
by
  intro h
  obtain ⟨m, b, hm, hb, h_eq⟩ := h
  rw [hm, hb, h_eq]
  -- Steps to solve this particular equation go here, but proof is not provided as per instructions.
  sorry

end line_equation_l779_779221


namespace sector_area_proof_l779_779081

-- Define the given conditions of the problem
variables {r α : ℝ}
variables (L : ℝ := 10) (central_angle : ℝ := 3)

-- The formula for the perimeter of a sector
def perimeter_of_sector (r α : ℝ) := 2 * r + α * r

-- The formula for the area of a sector
def area_of_sector (r α : ℝ) := (1 / 2) * r^2 * α

-- Proof goal statement: 
-- Prove that the area of the sector is 6 given the conditions
theorem sector_area_proof : 
  (∃ r : ℝ, perimeter_of_sector r central_angle = L) → 
  area_of_sector r central_angle = 6 
:= sorry

end sector_area_proof_l779_779081


namespace men_build_wall_l779_779500

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end men_build_wall_l779_779500


namespace trigonometric_identity_l779_779264

theorem trigonometric_identity :
  (sin (Real.pi * 7 / 180) + sin (Real.pi * 8 / 180) * cos (Real.pi * 15 / 180)) /
  (cos (Real.pi * 7 / 180) - sin (Real.pi * 8 / 180) * sin (Real.pi * 15 / 180))
  = 2 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l779_779264


namespace part1_part2_l779_779814

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Problem (1) part:
theorem part1 (x : ℝ) (h : f (x + 1) - f x > 1) : 0 < x ∧ x < 1 :=
by
  sorry

-- Problem (2) part:
def g (x k : ℝ) : ℝ := f (2^x + 1) + k * x

theorem part2 (k : ℝ) : (∀ x : ℝ, g (-x) k = g x k) ↔ k = -1/2 :=
by
  sorry

end part1_part2_l779_779814


namespace length_of_cd_equals_five_l779_779869

-- Definitions for triangle and isosceles properties
structure Triangle :=
(base : ℝ)
(height : ℝ)

structure IsoscelesTriangle extends Triangle :=
(area : ℝ)
(altitude_from_vertex : ℝ)

-- Definition of problem specific terms
def triangle_abe := IsoscelesTriangle.mk ⟨10, 20⟩ 100 20

def is_trapezoid (area: ℝ) : Prop := 
  area = 75

def cuts_into_trapezoid (abe : IsoscelesTriangle) (cd : ℝ) (trapezoid_area : ℝ) : Prop :=
  let total_area := abe.area
  let smaller_triangle_area := total_area - trapezoid_area in
  smaller_triangle_area = (abe.area - trapezoid_area)

-- Main statement to prove
theorem length_of_cd_equals_five
  (abe : IsoscelesTriangle)
  (cd : ℝ)
  (h_cuts: cuts_into_trapezoid abe cd 75) : 
  cd = 5 :=
sorry

end length_of_cd_equals_five_l779_779869


namespace ellipse_area_144_l779_779756

-- Defining the conditions: ellipse equation and area of an ellipse
def ellipse_area (a b : ℝ) : ℝ := Real.pi * a * b

-- The given problem statement
theorem ellipse_area_144 :
    let a := 4
    let b := 3
    let area := 12 * Real.pi
    9 * (x : ℝ) ^ 2 + 16 * (y : ℝ) ^ 2 = 144 →
        area = ellipse_area a b :=
by
    intros a b area h
    unfold a b area ellipse_area
    sorry

end ellipse_area_144_l779_779756


namespace area_of_abs_inequality_l779_779375

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l779_779375


namespace shoebox_width_is_l779_779213

noncomputable def shoebox_width (height : ℕ) (block_side : ℕ) (uncovered_area : ℕ) : ℕ :=
  let block_area := block_side * block_side
  let total_area := block_area + uncovered_area
  total_area / height

theorem shoebox_width_is (height : ℕ) (block_side : ℕ) (uncovered_area : ℕ) (W : ℕ) :
  height = 4 →
  block_side = 4 →
  uncovered_area = 8 →
  W = 6 :=
by
  intros h_height h_block_side h_uncovered_area
  have h_total_area : total_area = 24 := by
    rw [h_block_side, h_uncovered_area]
    exact calc
      total_area = block_area + uncovered_area := rfl
             ... = ((4 : ℕ) * 4) + 8 := sorry
  -- Need to skip the proof
  sorry

end shoebox_width_is_l779_779213


namespace track_problem_l779_779699

theorem track_problem
  (inner_diameter : ℝ)
  (width_between_circles : ℝ)
  (inner_diameter = 200)
  (width_between_circles = 15) :
  let outer_diameter := inner_diameter + 2 * width_between_circles,
      inner_circumference := Real.pi * inner_diameter,
      outer_circumference := Real.pi * outer_diameter,
      circumference_difference := outer_circumference - inner_circumference,
      inner_radius := inner_diameter / 2,
      outer_radius := outer_diameter / 2,
      inner_area := Real.pi * inner_radius ^ 2,
      outer_area := Real.pi * outer_radius ^ 2,
      track_area := outer_area - inner_area in
  circumference_difference = 30 * Real.pi ∧ track_area = 12900 * Real.pi / 4 :=
by
  sorry


end track_problem_l779_779699


namespace find_e_of_x_l779_779129

noncomputable def x_plus_inv_x_eq_five (x : ℝ) : Prop :=
  x + (1 / x) = 5

theorem find_e_of_x (x : ℝ) (h : x_plus_inv_x_eq_five x) : 
  x^2 + (1 / x)^2 = 23 := sorry

end find_e_of_x_l779_779129


namespace math_problem_l779_779075

noncomputable theory

def question1 (z : ℂ) : Prop :=
  (z + 2 * complex.I).im = 0 ∧ ((1 - 2 * complex.I) * z).re = 0

def question2 (ω : ℂ) (z : ℂ) : Prop :=
  abs (ω - conj z) = 1 → abs ω = 2 * real.sqrt 5 - 1

theorem math_problem (z : ℂ) (ω : ℂ) :
  question1 z → question2 ω (4 - 2 * complex.I) :=
by sorry

end math_problem_l779_779075


namespace brick_fit_probability_l779_779620

theorem brick_fit_probability :
  ∃ p_num p_denom : ℕ, (a1 a2 a3 b1 b2 b3 : ℕ) →
  {a1, a2, a3} ⊆ {1, 2, ..., 500} ∧
  {b1, b2, b3} ⊆ {1, 2, ..., 500} \ {a1, a2, a3} ∧
  b1 + b2 + b3 > a1 + a2 + a3 ∧
  p_num / p_denom = 1 / 2 ∧
  p_num + p_denom = 3 :=
begin
  sorry
end

end brick_fit_probability_l779_779620


namespace total_profit_is_1184_l779_779290

variables
  (cost_chocolate : ℕ)
  (cost_potato_chips : ℕ)
  (cost_melon_seeds : ℕ)
  (x y : ℕ)

-- Conditions from the problem
def cost_prices : Prop :=
  cost_chocolate = 12 ∧ cost_potato_chips = 8 ∧ cost_melon_seeds = 6

def discounted_selling_prices : Prop :=
  10*x = 20 ∧ 6*x = 12 ∧ 5*x = 10

def daytime_sales_volumes : Prop :=
  y > 0 ∧ 2*y > 0 ∧ 2*y > 0

def promotion_sales_volumes : Prop :=
  y/2 > 0 ∧ y > 0 ∧ y > 0

def total_sales_volume : Prop :=
  250 < (y + 2*y + 2*y + y/2 + y + y) ∧ (y + 2*y + 2*y + y/2 + y + y) < 350

def revenue_potato_chips : Prop :=
  6*x * 2*y + 6*x * y * 4 / 5 = 1344

-- Proof goal
theorem total_profit_is_1184 :
  cost_prices ∧ discounted_selling_prices ∧ daytime_sales_volumes ∧ promotion_sales_volumes ∧ total_sales_volume ∧ revenue_potato_chips →
  let
    daytime_revenue_chocolate := 20 * 40
    promo_revenue_chocolate   := 20 * (20 * 4 / 5)
    total_revenue_chocolate   := daytime_revenue_chocolate + promo_revenue_chocolate

    daytime_revenue_potato    := 12 * 80
    promo_revenue_potato      := 12 * (40 * 4 / 5)
    total_revenue_potato      := daytime_revenue_potato + promo_revenue_potato

    daytime_revenue_melon     := 10 * 80
    promo_revenue_melon       := 10 * (40 * 4 / 5)
    total_revenue_melon       := daytime_revenue_melon + promo_revenue_melon

    total_cost_chocolate  := 12 * 40
    total_cost_potato     := 8 * 120
    total_cost_melon      := 6 * 120

    profit := total_revenue_chocolate + total_revenue_potato + total_revenue_melon -
              (total_cost_chocolate + total_cost_potato + total_cost_melon)
  in profit = 1184 :=
sorry

end total_profit_is_1184_l779_779290


namespace ab_product_l779_779828

theorem ab_product (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∃ x y : ℝ, ax + by = 8 ∧ (1/2) * (8/a) * (8/b) = 8) :
  a * b = 4 := sorry

end ab_product_l779_779828


namespace trapezoid_cd_length_l779_779245

noncomputable def proof_cd_length (AD BC CD : ℝ) (BD : ℝ) (angle_DBA angle_BDC : ℝ) (ratio_BC_AD : ℝ) : Prop :=
  AD > 0 ∧ BC > 0 ∧
  BD = 1 ∧
  angle_DBA = 23 ∧
  angle_BDC = 46 ∧
  ratio_BC_AD = 9 / 5 ∧
  AD / BC = 5 / 9 ∧
  CD = 4 / 5

theorem trapezoid_cd_length
  (AD BC CD : ℝ)
  (BD : ℝ := 1)
  (angle_DBA : ℝ := 23)
  (angle_BDC : ℝ := 46)
  (ratio_BC_AD : ℝ := 9 / 5)
  (h_conditions : proof_cd_length AD BC CD BD angle_DBA angle_BDC ratio_BC_AD) : CD = 4 / 5 :=
sorry

end trapezoid_cd_length_l779_779245


namespace volume_of_sphere_with_prism_vertices_on_surface_l779_779053

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem volume_of_sphere_with_prism_vertices_on_surface :
  ∀ (l w h : ℕ), l = 2 ∧ w = 1 ∧ h = 1 → volume_of_sphere (Real.sqrt (l^2 + w^2 + h^2) / 2) = Real.sqrt 6 * Real.pi :=
by
  intros l w h conditions
  cases conditions with l_cond rest
  cases rest with w_cond h_cond
  rw [l_cond, w_cond, h_cond]
  have : Real.sqrt (2^2 + 1^2 + 1^2) / 2 = Real.sqrt 6 / 2 := by sorry
  rw this
  exact sorry

end volume_of_sphere_with_prism_vertices_on_surface_l779_779053


namespace max_moves_to_remove_all_cars_l779_779916

-- Defining the grid dimensions and cardinal directions
def Grid := Fin 200 × Fin 200
inductive Direction | North | South | East | West

-- A unit move of the car in the specified direction.
def carMove : Direction → Grid → Option Grid
| Direction.North, (i, j) => if i = 0 then none else some (i.pred, j)
| Direction.South, (i, j) => if i = 199 then none else some (i.succ, j)
| Direction.East,  (i, j) => if j = 199 then none else some (i, j.succ)
| Direction.West,  (i, j) => if j = 0 then none else some (i, j.pred)

-- Maximum number of moves needed to remove all cars from the grid
noncomputable def maxMovesToRemoveCars : ℕ := 6014950

-- Theorem statement
theorem max_moves_to_remove_all_cars :
  ∀ (config : Grid → Direction), (∃ seq : list (Grid × Direction), 
    all_cars_removed seq config) → 
    moves seq ≤ maxMovesToRemoveCars := sorry

-- Auxiliary definitions
-- Check if all cars are removed based on a given move sequence
def all_cars_removed (seq : list (Grid × Direction)) 
  (init_config : Grid → Direction) : Prop := sorry

-- Count moves from a list of moves
noncomputable def moves (seq : list (Grid × Direction)) : ℕ := sorry

end max_moves_to_remove_all_cars_l779_779916


namespace triangle_area_l779_779246

theorem triangle_area (A B C : Type) [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C]
  (angle_BAC : angle A B C = 45) (BC_length : dist B C = 1) (AC_eq_2AB : dist A C = 2 * dist A B) :
  triangle_area A B C = 1 / (4 * real.sqrt 2) := 
sorry

end triangle_area_l779_779246


namespace math_problem_l779_779775

open Real

theorem math_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) :
  (a - 1) * (b - 1) = 1 ∧
  ¬(∀ x, x ≤ ab) ∧
  ∃ a', ∃ b', 1 / a' + 1 / b' = 1 ∧ (a' + 4 * b' = 9) ∧
  ∃ a'', ∃ b'', 1 / a'' + 1 / b'' = 1 ∧ (1 / a''^2 + 2 / b''^2 = 2 / 3) :=
by sorry

end math_problem_l779_779775


namespace safe_flight_prob_correct_l779_779664

def larger_cube_edge_length : ℝ := 3
def small_cube_edge_length : ℝ := 1

-- Function to calculate the volume of a cube given its edge length
def volume_of_cube (a : ℝ) : ℝ := a ^ 3

-- Definition of larger cube and smaller cube volumes
def larger_cube_volume : ℝ := volume_of_cube larger_cube_edge_length
def small_cube_volume : ℝ := volume_of_cube small_cube_edge_length

-- Safe flight probability
def safe_flight_probability : ℝ := small_cube_volume / larger_cube_volume

theorem safe_flight_prob_correct :
  safe_flight_probability = 1 / 27 :=
by
  unfold safe_flight_probability
  unfold small_cube_volume larger_cube_volume
  unfold volume_of_cube
  sorry

end safe_flight_prob_correct_l779_779664


namespace vector_length_problem_l779_779101

open Real EuclideanSpace

noncomputable def vec_a : EuclideanSpace ℝ (Fin 2) := ![3, -4]

noncomputable def vec_b : EuclideanSpace ℝ (Fin 2) :=
  let θ : ℝ := real.pi / 3
  let b_norm : ℝ := 2
  let b1 : ℝ := b_norm * cos(θ)
  let b2 : ℝ := b_norm * sin(θ)
  ![b1, b2]

theorem vector_length_problem :
  let vec_sum := vec_a + 2 • vec_b
  ‖vec_sum‖ = sqrt 61 := sorry

end vector_length_problem_l779_779101


namespace john_allowance_calculation_l779_779516

theorem john_allowance_calculation (initial_money final_money game_cost allowance: ℕ) 
(h_initial: initial_money = 5) 
(h_game_cost: game_cost = 2) 
(h_final: final_money = 29) 
(h_allowance: final_money = initial_money - game_cost + allowance) : 
  allowance = 26 :=
by
  sorry

end john_allowance_calculation_l779_779516


namespace probability_of_red_second_given_red_first_l779_779286

-- Define the conditions as per the problem.
def total_balls := 5
def red_balls := 3
def yellow_balls := 2
def first_draw_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
def both_draws_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of drawing a red ball in the second draw given the first was red.
def conditional_probability_red_second_given_first : ℚ :=
  both_draws_red / first_draw_red

-- The main statement to be proved.
theorem probability_of_red_second_given_red_first :
  conditional_probability_red_second_given_first = 1 / 2 :=
by
  sorry

end probability_of_red_second_given_red_first_l779_779286


namespace train_b_speed_l779_779628

noncomputable def relative_speed_in_km_per_hr (total_distance : ℕ) (time : ℕ) : ℕ :=
  (total_distance * 3600) / (time * 1000)

theorem train_b_speed
  (length_A : ℕ)
  (length_B : ℕ)
  (speed_A : ℕ)
  (cross_time : ℕ)
  (total_distance := length_A + length_B)
  (relative_speed := relative_speed_in_km_per_hr total_distance cross_time)
  (speed_B := relative_speed - speed_A) :
  length_A = 225 →
  length_B = 150 →
  speed_A = 54 →
  cross_time = 15 →
  speed_B = 36 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold relative_speed_in_km_per_hr
  simp
  sorry

end train_b_speed_l779_779628


namespace striped_jerseys_count_l779_779519

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l779_779519


namespace roots_eq_202_l779_779537

theorem roots_eq_202 (p q : ℝ) 
  (h1 : ∀ x : ℝ, ((x + p) * (x + q) * (x + 10) = 0 ↔ (x = -p ∨ x = -q ∨ x = -10)) ∧ 
       ∀ x : ℝ, ((x + 5) ^ 2 = 0 ↔ x = -5)) 
  (h2 : ∀ x : ℝ, ((x + 2 * p) * (x + 4) * (x + 8) = 0 ↔ (x = -2 * p ∨ x = -4 ∨ x = -8)) ∧ 
       ∀ x : ℝ, ((x + q) * (x + 10) = 0 ↔ (x = -q ∨ x = -10))) 
  (hpq : p = q) (neq_5 : q ≠ 5) (p_2 : p = 2):
  100 * p + q = 202 := sorry

end roots_eq_202_l779_779537


namespace cos_pi_minus_2alpha_l779_779795

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l779_779795


namespace krista_bank_exceeds_50_on_wednesday_l779_779164

theorem krista_bank_exceeds_50_on_wednesday :
  ∃ (n : ℕ), (5 * (2^n - 1)) / (2 - 1) > 5000 ∧ 
  (∃ (k : ℕ), k < 21 ∧ n = k) ∧ 
  (n + 1) % 7 = 3 :=
begin
  sorry
end

end krista_bank_exceeds_50_on_wednesday_l779_779164


namespace orthocenter_centroid_incenter_circumcenter_relation_l779_779527

theorem orthocenter_centroid_incenter_circumcenter_relation
  (M S J O : Type)
  [triangle M S J O] :
  (dist M J)^2 + 2 * (dist O J)^2 = 3 * (dist J S)^2 + 6 * (dist S O)^2 := 
sorry

end orthocenter_centroid_incenter_circumcenter_relation_l779_779527


namespace euro_comparison_l779_779940

theorem euro_comparison :
  let euro_to_dollar := 1.5
  let diana_dollars := 600
  let etienne_euros := 350
  let etienne_dollars := etienne_euros * euro_to_dollar
  let percentage_difference := ((diana_dollars - etienne_dollars) / diana_dollars) * 100
  in percentage_difference = 12.5 := by
  -- Proof omitted
  sorry

end euro_comparison_l779_779940


namespace sin_P_equal_4_over_5_l779_779140

theorem sin_P_equal_4_over_5 
  (P Q R : Type) 
  (PR PQ QR : Real) 
  (h1 : ∠Q = 90) 
  (h2 : 3 * Real.sin P = 4 * Real.cos P)
  (h3 : PQ * PQ + QR * QR = PR * PR) : 
  Real.sin P = 4 / 5 :=
by
  sorry

end sin_P_equal_4_over_5_l779_779140


namespace unique_19_tuple_l779_779380

theorem unique_19_tuple (a : Fin 19 → ℤ)
  (h : ∀ i, a i ^ 2 = ∑ j in Finset.univ.filter (≠ i), a j ^ 2) :
  ∃! t, ∀ i, a i = t :=
by {
  sorry
}

end unique_19_tuple_l779_779380


namespace sum_of_integers_ending_in_7_between_100_and_500_l779_779725

theorem sum_of_integers_ending_in_7_between_100_and_500 :
  let a := 107
  let d := 10
  let l := 497
  let n := 40
  let S_n := n * (a + l) / 2
  in S_n = 12080 :=
by
  let a := 107
  let d := 10
  let l := 497
  let n := 40
  let S_n := n * (a + l) / 2
  show S_n = 12080 from sorry

end sum_of_integers_ending_in_7_between_100_and_500_l779_779725


namespace hexagon_square_overlap_area_l779_779249

noncomputable def area_of_overlap (s : ℝ) : ℝ :=
  8 - 4 * real.sqrt 3

-- Statement of the theorem in Lean
theorem hexagon_square_overlap_area : 
  ∀ (s : ℝ), s = 2 → area_of_overlap s = 8 - 4 * real.sqrt 3 :=
by
  assume s hs,
  unfold area_of_overlap,
  rw hs,
  simp,
  sorry

end hexagon_square_overlap_area_l779_779249


namespace ms_emily_inheritance_l779_779882

theorem ms_emily_inheritance :
  ∃ (y : ℝ), 
    (0.25 * y + 0.15 * (y - 0.25 * y) = 19500) ∧
    (y = 53800) :=
by
  sorry

end ms_emily_inheritance_l779_779882


namespace jason_retirement_age_l779_779875

theorem jason_retirement_age :
  ∃ (age_at_retirement : ℕ),
    let initial_age := 18,
        years_to_chief := 8,
        perc_longer_chief_to_senior := 0.255,
        perc_shorter_senior_to_master := 0.125,
        perc_longer_master_to_command := 0.475,
        years_last_rank := 2.5,
        
        years_chief_to_senior := years_to_chief * (1 + perc_longer_chief_to_senior),
        years_senior_to_master := years_chief_to_senior * (1 - perc_shorter_senior_to_master),
        years_master_to_command := years_senior_to_master * (1 + perc_longer_master_to_command),

        total_years := years_to_chief + years_chief_to_senior + years_senior_to_master + 
                       years_master_to_command + years_last_rank,

        age_at_retirement := initial_age + total_years.to_nat
    in age_at_retirement = 60 :=
begin
  sorry
end

end jason_retirement_age_l779_779875


namespace count_solutions_eq_138_l779_779824

theorem count_solutions_eq_138 :
  let numerator := (λ (x : ℕ), ∏ i in (Finset.range 150).map (λ n, n + 1), (x - i))
  let denominator := (λ (x : ℕ), ∏ i in (Finset.range 150).map (λ n, n + 1), (x - i^2))
  (Finset.filter (λ x, numerator x = 0 ∧ denominator x ≠ 0) (Finset.range 150).map (λ n, n + 1)).card = 138 :=
by
  sorry

end count_solutions_eq_138_l779_779824


namespace angle_eq_of_quadrilateral_bisection_l779_779145

-- Define the quadrilateral and its points
variables {A B C D E G F : Point}

-- Hypotheses
hypothesis h1 : ConvexQuadrilateral A B C D
hypothesis h2 : Bisection (AC : Line) (∠BAD)
hypothesis h3 : E ∈ LineExtension CD
hypothesis h4 : IntersectionPoint (BE : Line) (AC : Line) G
hypothesis h5 : IntersectionPoint (LineExtension DG) (LineExtension CB) F

-- Goal
theorem angle_eq_of_quadrilateral_bisection 
  (h : ConvexQuadrilateral A B C D)
  (h₁ : Bisection (AC : Line) (∠BAD))
  (h₂ : E ∈ LineExtension CD)
  (h₃ : IntersectionPoint (BE : Line) (AC : Line) G)
  (h₄ : IntersectionPoint (LineExtension DG) (LineExtension CB) F) :
  ∠BAF = ∠DAE :=
sorry

end angle_eq_of_quadrilateral_bisection_l779_779145


namespace polynomial_perfect_square_trinomial_l779_779837

theorem polynomial_perfect_square_trinomial (k : ℝ) :
  (∀ x : ℝ, 4 * x^2 + 2 * k * x + 25 = (2 * x + 5) * (2 * x + 5)) → (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_trinomial_l779_779837


namespace length_of_A_l779_779526

structure Point (α : Type) :=
  (x y : α)

def slope {α : Type} [Field α] (A B : Point α) : α :=
  (B.y - A.y) / (B.x - A.x)

def line_equation {α : Type} [Field α] (A : Point α) (m : α) : α → α :=
  λ x, m * (x - A.x) + A.y

def perpendicular_slope {α : Type} [Field α] (m : α) : α :=
  -1 / m

def intersection {α : Type} [Field α] (m1 m2 b1 b2 : α) : α :=
  (b2 - b1) / (m1 - m2)

def point_on_line {α : Type} [Field α] (m : α) (x : α) (b : α) : Point α :=
  { x := x, y := m * x + b }

theorem length_of_A'B'_is_5_point4 : 
  let A : Point ℚ := { x := 2, y := 3 }
      B : Point ℚ := { x := 3, y := 8 }
      A' : Point ℚ := { x := 7 / 5, y := 14 / 5 }
      B' : Point ℚ := { x := 19 / 5, y := 38 / 5 }
      dist : Point ℚ → Point ℚ → ℚ := λ P Q, (real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)).toRat 
  in dist A' B' = 5.4 := sorry

end length_of_A_l779_779526


namespace find_reduced_price_l779_779697

noncomputable def reduced_price_per_kg 
  (total_spent : ℝ) (original_quantity : ℝ) (additional_quantity : ℝ) (price_reduction_rate : ℝ) : ℝ :=
  let original_price := total_spent / original_quantity
  let reduced_price := original_price * (1 - price_reduction_rate)
  reduced_price

theorem find_reduced_price 
  (total_spent : ℝ := 800)
  (original_quantity : ℝ := 20)
  (additional_quantity : ℝ := 5)
  (price_reduction_rate : ℝ := 0.15) :
  reduced_price_per_kg total_spent original_quantity additional_quantity price_reduction_rate = 34 :=
by
  sorry

end find_reduced_price_l779_779697


namespace min_length_segment_AB_l779_779484

open Set Real

-- Definitions for the problem
def C1 (θ : ℝ) : ℝ × ℝ := (3 + cos θ, 4 + sin θ)
def C2 : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 = 1

-- Lean statement for the mathematical proof problem
theorem min_length_segment_AB : 
  ∃ (A : ℝ × ℝ) (θ : ℝ) (B : ℝ × ℝ), A = C1 θ ∧ C2 B ∧ (dist A B = 3) :=
sorry

end min_length_segment_AB_l779_779484


namespace will_3_point_shots_l779_779848

theorem will_3_point_shots :
  ∃ x y : ℕ, 3 * x + 2 * y = 26 ∧ x + y = 11 ∧ x = 4 :=
by
  sorry

end will_3_point_shots_l779_779848


namespace expression_value_l779_779637

theorem expression_value (x : ℤ) (h : x = -2) : x ^ 2 + 6 * x - 8 = -16 := 
by 
  rw [h]
  sorry

end expression_value_l779_779637


namespace drum_Y_initial_filled_l779_779750

variable {C : ℝ} (Y_oil : ℝ)

-- Conditions
def drum_X_half_full : Prop := ∃ (C : ℝ), (0.5 * C < C)
def drum_Y_twice_capacity_X (Y : ℝ) : Prop := Y = 2 * C
def drum_Y_filled_after_pour (Y_oil : ℝ) : Prop := (Y_oil + 0.5 * C = 0.45 * (2 * C))

-- Proof that drum Y is initially filled to 0.4 of its capacity
theorem drum_Y_initial_filled : drum_X_half_full C ∧ drum_Y_twice_capacity_X (2 * C) ∧ drum_Y_filled_after_pour Y_oil → Y_oil = 0.4 * C :=
by
  sorry

end drum_Y_initial_filled_l779_779750


namespace distinct_arrangements_count_l779_779195

theorem distinct_arrangements_count :
  ∃ n : ℕ, n = 14 ∧
          ∀ grid : fin 11 → fin 11 → bool,
            (∀ i, (∑ j, if grid i j then 1 else 0) = 2) ∧
            (∀ j, (∑ i, if grid i j then 1 else 0) = 2) →
            n = 14 :=
sorry

end distinct_arrangements_count_l779_779195


namespace difference_max_min_f_l779_779430

noncomputable def f (x : ℝ) : ℝ := exp (sin x + cos x) - (1 / 2) * sin (2 * x)

theorem difference_max_min_f : 
  (∀ x : ℝ, f x ≥ exp (- sqrt 2) - (1 / 2)) ∧
  (∀ x : ℝ, f x ≤ exp (sqrt 2) - (1 / 2)) ∧
  (∃x y : ℝ, f x = exp (sqrt 2) - (1 / 2) ∧ f y = exp (- sqrt 2) - (1 / 2)) →
  (∃ x y : ℝ, f x - f y = exp (sqrt 2) - exp (- sqrt 2)) :=
sorry

end difference_max_min_f_l779_779430


namespace fraction_of_crop_to_longest_side_l779_779005

variable (base1 base2 side angle height areaTotal areaBelowMidsegment : ℝ)
variable (condition1 : base1 = 130)
variable (condition2 : base2 = 210)
variable (condition3 : side = 150)
variable (condition4 : angle = 70)
variable (condition5 : height = 150 * Real.sin (70 * Real.pi / 180))
variable (condition6 : areaTotal = 0.5 * (base1 + base2) * height)
variable (midsegment : ℝ := (base1 + base2) / 2)
variable (areaBelowMidsegment : ℝ := 0.5 * (midsegment + base2) * (height / 2))

theorem fraction_of_crop_to_longest_side
        (h1 : base1 = 130)
        (h2 : base2 = 210)
        (h3 : side = 150)
        (h4 : angle = 70)
        (h5 : height = 150 * Real.sin (70 * Real.pi / 180))
        (h6 : areaTotal = 0.5 * (base1 + base2) * height)
        (h7 : areaBelowMidsegment = 0.5 * (midsegment + base2) * (height / 2)) :
    areaBelowMidsegment / areaTotal ≈ 0.557 := by
  sorry

end fraction_of_crop_to_longest_side_l779_779005


namespace domain_f_l779_779026

def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 * (x^2 - 4) - 1)

theorem domain_f :
  ∀ x : ℝ, x ≠ -Real.sqrt (2 + Real.sqrt 5) → x ≠ Real.sqrt (2 + Real.sqrt 5) → 
  f x ≠ 0 :=
begin
  sorry
end

end domain_f_l779_779026


namespace find_f_2_l779_779592

noncomputable def f : ℝ → ℝ
| x := if x > 3 then Real.log2 (2 ^ x - 8) else f (x + 2)

theorem find_f_2 : f 2 = 3 :=
sorry

end find_f_2_l779_779592


namespace sum_of_B_reciprocals_p_plus_q_eq_51_l779_779169

def B : Set ℕ := { n | ∀ p, Prime p → p ∣ n → p = 3 ∨ p = 5 ∨ p = 7 }

noncomputable def sum_reciprocals_B : ℚ :=
  ∑' n in B, 1 / (n : ℚ)

theorem sum_of_B_reciprocals :
  ∑' n in B, (1 / (n : ℚ)) = 35 / 16 :=
by sorry -- Proof is omitted

theorem p_plus_q_eq_51 :
  ∑' n in B, (1 / (n : ℚ)) = 35 / 16 → 51 = 35 + 16 :=
by sorry -- Proof is omitted

end sum_of_B_reciprocals_p_plus_q_eq_51_l779_779169


namespace even_num_Z_tetrominoes_l779_779736

-- Definitions based on the conditions of the problem
def is_tiled_with_S_tetrominoes (P : Type) : Prop := sorry
def tiling_uses_S_Z_tetrominoes (P : Type) : Prop := sorry
def num_Z_tetrominoes (P : Type) : ℕ := sorry

-- The theorem statement
theorem even_num_Z_tetrominoes (P : Type) 
  (hTiledWithS : is_tiled_with_S_tetrominoes P) 
  (hTilingWithSZ : tiling_uses_S_Z_tetrominoes P) : num_Z_tetrominoes P % 2 = 0 :=
sorry

end even_num_Z_tetrominoes_l779_779736


namespace propositions_correctness_l779_779732

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def P : Prop := ∃ x : ℝ, x^2 - x - 1 > 0
def negP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

theorem propositions_correctness :
    (∀ a, a ∈ M → a ∈ N) = false ∧
    (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
    (∀ p q, ¬(p ∧ q) → ¬p ∧ ¬q) = false ∧ 
    (¬P ↔ negP) :=
by
  sorry

end propositions_correctness_l779_779732


namespace exists_k_with_sum_inequality_l779_779888

theorem exists_k_with_sum_inequality (n : ℕ) (a b : Fin n -> ℝ) :
  ∃ k : Fin n, 
    (∑ i, |a i - a k|) ≤ (∑ i, |b i - a k|) :=
sorry

end exists_k_with_sum_inequality_l779_779888


namespace sum_abs_roots_polynomial_l779_779768

theorem sum_abs_roots_polynomial :
  let f := λ x : ℝ, x^5 - 5*x^4 + 10*x^3 - 10*x^2 - 5*x - 1 in
  let roots := [real.sqrt 3.5, -real.sqrt 3.5, real.sqrt 1.5, -real.sqrt 1.5, 0, -0.1] in
  (roots.map real.abs).sum = 2 * real.sqrt 3.5 + 2 * real.sqrt 1.5 + 0.1 :=
by
  sorry

end sum_abs_roots_polynomial_l779_779768


namespace fraction_increase_by_50_percent_l779_779460

variable (x y : ℝ)
variable (h1 : 0 < y)

theorem fraction_increase_by_50_percent (h2 : 0.6 * x / 0.4 * y = 1.5 * x / y) : 
  1.5 * (x / y) = 1.5 * (x / y) :=
by
  sorry

end fraction_increase_by_50_percent_l779_779460


namespace trip_average_speed_l779_779847

theorem trip_average_speed 
  (d₁ d₂ : ℕ) (s₁ s₂ : ℕ) (total_distance : ℕ) 
  (d1_eq : d₁ = 30)
  (d2_eq : d₂ = 70)
  (s1_eq : s₁ = 60)
  (s2_eq : s₂ = 35)
  (total_distance_eq : total_distance = 100) :
  let t₁ := d₁ / s₁ in
  let t₂ := d₂ / s₂ in
  let total_time := t₁ + t₂ in
  let average_speed := total_distance / total_time in
  average_speed = 40 := 
by
  -- This is a statement template without providing the proof
  sorry

end trip_average_speed_l779_779847


namespace domain_of_f_l779_779086

noncomputable def f (x : ℝ) : ℝ :=
  (x - 4)^0 + Real.sqrt (2 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (1 < x ∧ x < 4) ∨ (4 < x) ↔
    ∃ y : ℝ, f y = f x :=
sorry

end domain_of_f_l779_779086


namespace john_total_amount_l779_779511

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l779_779511


namespace grid_diagonal_segments_l779_779414

theorem grid_diagonal_segments (m n : ℕ) (hm : m = 100) (hn : n = 101) :
    let d := m + n - gcd m n
    d = 200 := by
  sorry

end grid_diagonal_segments_l779_779414


namespace age_difference_l779_779608

theorem age_difference (d : ℕ) (h1 : 18 + (18 - d) + (18 - 2 * d) + (18 - 3 * d) = 48) : d = 4 :=
sorry

end age_difference_l779_779608


namespace unique_point_O_l779_779655

structure Point where
  x : ℝ
  y : ℝ

structure Square where
  A B C D : Point

def scaledVersion (scale : ℝ) (sq : Square) : Square :=
  { A := {x := sq.A.x * scale, y := sq.A.y * scale},
    B := {x := sq.B.x * scale, y := sq.B.y * scale},
    C := {x := sq.C.x * scale, y := sq.C.y * scale},
    D := {x := sq.D.x * scale, y := sq.D.y * scale} }

-- Definition of the unique overlap point O
theorem unique_point_O (s t : ℝ) (large_small : Square) (small_large : Square) :
  ∃! O : Point, 
    let large_map := large_small in
    let small_map := scaledVersion s small_large in
    O.x = (large_map.A.x + large_map.B.x + large_map.C.x + large_map.D.x) / 4 ∧
    O.y = (large_map.A.y + large_map.B.y + large_map.C.y + large_map.D.y) / 4 ∧
    O.x = (small_map.A.x + small_map.B.x + small_map.C.x + small_map.D.x) / 4 ∧
    O.y = (small_map.A.y + small_map.B.y + small_map.C.y + small_map.D.y) / 4 :=
begin
  sorry
end

end unique_point_O_l779_779655


namespace total_amount_from_grandparents_l779_779509

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l779_779509


namespace quadrilateral_similarity_l779_779050

def similar_quadrilateral (A B C D : ℝ) (A1 B1 C1 D1 A2 B2 C2 D2 : ℝ) : Prop :=
  let similarity_ratio := abs ((cot A + cot C) * (cot B + cot D) / 4)
  similar ABCD A2B2C2D2 similarity_ratio

theorem quadrilateral_similarity (A B C D : ℝ) (A1 B1 C1 D1 A2 B2 C2 D2 : ℝ)
  (h1 : convex_quadrilateral ABCD)
  (h2 : A1 = circumcenter BCD)
  (h3 : B1 = circumcenter CDA)
  (h4 : C1 = circumcenter DAB)
  (h5 : D1 = circumcenter ABC)
  (h6 : A2 = circumcenter B1C1D1)
  (h7 : B2 = circumcenter C1D1A1)
  (h8 : C2 = circumcenter D1A1B1)
  (h9 : D2 = circumcenter A1B1C1) :
  similar_quadrilateral A B C D A1 B1 C1 D1 A2 B2 C2 D2 :=
by
  sorry

end quadrilateral_similarity_l779_779050


namespace m_n_correct_l779_779426

noncomputable def m_n_sum : ℕ :=
let eq_coeffs : {a b c : ℤ // a = 40 ∧ b = 39 ∧ c = -1} := ⟨40, 39, -1, rfl, rfl, rfl⟩,
    arith_seq : ℕ → ℤ := λ m, 1 + (m - 1) * 2,
    geom_seq_sum : ℕ → ℤ := λ n, 1 - (-3)^n in
let roots_sum_recip := -eq_coeffs.val.2.1 / eq_coeffs.val.1 in
let roots_prod_recip := 1 / eq_coeffs.val.2.2 in
let m := ∃ m : ℕ, (arith_seq m : ℤ) = roots_sum_recip := 39 in
let n := ∃ n : ℕ, (geom_seq_sum n : ℤ) = -40 in
(m + n)

theorem m_n_correct : m_n_sum = 24 :=
sorry

end m_n_correct_l779_779426


namespace value_of_a_l779_779832

theorem value_of_a (a : ℕ) (h1 : a * 9^3 = 3 * 15^5) (h2 : a = 5^5) : a = 3125 := by
  sorry

end value_of_a_l779_779832


namespace range_of_a_l779_779080

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end range_of_a_l779_779080


namespace cannot_be_ten_from_20_after_33_moves_l779_779629

-- Define the conditions
def initial_number : ℤ := 20
def total_moves : ℤ := 33
def move_options (n : ℤ) : set ℤ := {n + 1, n - 1}
def target_number : ℤ := 10

-- Statement to be proved
theorem cannot_be_ten_from_20_after_33_moves :
  ¬ (target_number ∈ {m | ∃ (s : list ℤ), s.length = total_moves ∧ list.foldl (λ acc x, acc + x) initial_number s = m}) :=
sorry  -- Proof goes here

end cannot_be_ten_from_20_after_33_moves_l779_779629


namespace root_in_interval_l779_779326

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l779_779326


namespace no_integer_solutions_3a2_eq_b2_plus_1_l779_779358

theorem no_integer_solutions_3a2_eq_b2_plus_1 : 
  ¬ ∃ a b : ℤ, 3 * a^2 = b^2 + 1 :=
by
  intro h
  obtain ⟨a, b, hab⟩ := h
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l779_779358


namespace arrangements_of_five_students_l779_779385

theorem arrangements_of_five_students :
  let students := {A, B, C, D, E}
  A and B must be together
  C and D cannot be together
proves number of different arrangements = 24 :=
sorry

end arrangements_of_five_students_l779_779385


namespace cos_B_equivalence_l779_779131

theorem cos_B_equivalence (A B C : Angle) (a b c : ℝ)
  (h1 : sin B = 2 * sin A)
  (h2 : triangle_area ABC = a^2 * sin B)
  (h3 : a = side_opposite A)
  (h4 : b = side_opposite B)
  (h5 : c = side_opposite C) :
  cos B = 1 / 4 := 
by sorry

end cos_B_equivalence_l779_779131


namespace intersection_A_B_l779_779417

open Set

def A : Set ℤ := {x | log 2 x ≤ 1}
def B : Set ℤ := {x | x^2 - x - 2 ≤ 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by 
  sorry

end intersection_A_B_l779_779417


namespace code_for_rand_is_1236_l779_779645

def encode_range : list (char × nat) := [('r', 1), ('a', 2), ('n', 3), ('g', 4), ('e', 5)]
def encode_random : list (char × nat) := [('r', 1), ('a', 2), ('n', 3), ('d', 6), ('o', 7), ('m', 8)]

def encode (word : string) : string → option (list nat)
| "range" := some [1, 2, 3, 4, 5]
| "random" := some [1, 2, 3, 6, 7, 8]
| _ := none

theorem code_for_rand_is_1236 :
    encode "range" = some [1, 2, 3, 4, 5] ∧
    encode "random" = some [1, 2, 3, 6, 7, 8] →
    encode "rand" = some [1, 2, 3, 6] :=
sorry

end code_for_rand_is_1236_l779_779645


namespace game_winner_l779_779701

theorem game_winner (N : ℕ) : 
  (N = 1 → ∃ strategy : (Π turn : ℕ, string), 
    (strategy 0 = "X" ∧ (∀ t : ℕ, turn t = "X" → strategy t.succ ≠ "X") ∧ 
    (∃ t : ℕ, turn t ≠ "O" → strategy (t+1) = "O") 
    ∧ (∃ t1 t2 : ℕ, t1 ≠ t2 → strategy t1.succ ≠ strategy t2.succ))) 
  ∧ 
  (N > 1 → ∃ strategy : (Π turn : ℕ, string), 
    (strategy 1 ≠ "O" ∧ (∀ t : ℕ, turn t.succ = "X" → strategy t.succ.succ ∧ 
    (∃ t1 t2 : ℕ, (t1 ≠ t2 ∧ strategy t1.succ ≠ strategy t2.succ)) 
    ∧ (∃ t : ℕ, turn t = "O" ∧ strategy t.succ.succ ≠ "X")))) := 
sorry

end game_winner_l779_779701


namespace fractional_eq_repeated_root_l779_779591

theorem fractional_eq_repeated_root (x m : ℝ) (h : x = 3) : 
  (∃ r, r = x ∧ (r / (r - 3) + 1 = m / (r - 3)) ∧ (2 * r - 3 = m)) →
  m = 3 :=
by
  intro h1
  cases h1 with r hr
  rw hr.1 at hr
  sorry

end fractional_eq_repeated_root_l779_779591


namespace problem_l779_779063

noncomputable theory

open Real

def point (x y: ℝ) := (x, y)

def vector (p1 p2: (ℝ × ℝ)) := (p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v: (ℝ × ℝ)) := sqrt ((v.1)^2 + (v.2)^2)

def dot_product (v1 v2: (ℝ × ℝ)) := (v1.1 * v2.1 + v1.2 * v2.2)

def vector_parallel (v1 v2: (ℝ × ℝ)) := (v1.1 / v2.1) = (v1.2 / v2.2)

def vector_perpendicular (v1 v2: (ℝ × ℝ)) := dot_product v1 v2 = 0

def vector_acute (v1 v2: (ℝ × ℝ)) := dot_product v1 v2 > 0

theorem problem
  (m : ℝ)
  (A := point 1 2)
  (B := point 3 1)
  (C := point 4 (m + 1))
  (AB := vector A B)
  (BC := vector B C)
  (BA := vector B A):
  (magnitude AB = sqrt 5) ∧
  (vector_perpendicular AB BC → m ≠ -2) ∧
  (vector_parallel AB BC → m = -1/2) ∧
  (vector_acute BA BC → m > 2 ∧ m ≠ -1/2) := by
    sorry

end problem_l779_779063


namespace translate_function_l779_779087

def f : ℝ → ℝ := λ x, 2 * Real.sin (2 * x + 2 * Real.pi / 3)

theorem translate_function :
  (λ x, f (x - Real.pi / 6)) = (λ x, 2 * Real.sin (2 * x + Real.pi / 3)) :=
by
  sorry

end translate_function_l779_779087


namespace proposition_neg_p_and_q_false_l779_779838

theorem proposition_neg_p_and_q_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end proposition_neg_p_and_q_false_l779_779838


namespace imaginary_part_of_complex_num_l779_779803

def imaginary_unit : ℂ := Complex.I

noncomputable def complex_num : ℂ := 10 * imaginary_unit / (1 - 2 * imaginary_unit)

theorem imaginary_part_of_complex_num : complex_num.im = 2 := by
  sorry

end imaginary_part_of_complex_num_l779_779803


namespace exists_subset_with_property_l779_779017

theorem exists_subset_with_property :
  ∃ X : Set Int, ∀ n : Int, ∃ (a b : X), a + 2 * b = n ∧ ∀ (a' b' : X), (a + 2 * b = n ∧ a' + 2 * b' = n) → (a = a' ∧ b = b') :=
sorry

end exists_subset_with_property_l779_779017


namespace sum_first_5_terms_l779_779143

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

def S_5 (a1 d : ℝ) : ℝ :=
  (5 / 2) * (2 * a1 + 4 * d)

theorem sum_first_5_terms (a1 d : ℝ) (h : a_n a1 d 3 = 2) :
  S_5 a1 d = 10 :=
by
  sorry

end sum_first_5_terms_l779_779143


namespace Maggie_earnings_l779_779549

theorem Maggie_earnings :
  let family_commission := 7
  let neighbor_commission := 6
  let bonus_fixed := 10
  let bonus_threshold := 10
  let bonus_per_subscription := 1
  let monday_family := 4 + 1 
  let tuesday_neighbors := 2 + 2 * 2
  let wednesday_family := 3 + 1
  let total_family := monday_family + wednesday_family
  let total_neighbors := tuesday_neighbors
  let total_subscriptions := total_family + total_neighbors
  let bonus := if total_subscriptions > bonus_threshold then 
                 bonus_fixed + bonus_per_subscription * (total_subscriptions - bonus_threshold)
               else 0
  let total_earnings := total_family * family_commission + total_neighbors * neighbor_commission + bonus
  total_earnings = 114 := 
by {
  -- Placeholder for the proof. We assume this step will contain a verification of derived calculations.
  sorry
}

end Maggie_earnings_l779_779549


namespace length_of_AB_area_of_triangular_piece_area_of_five_sided_piece_area_of_hole_l779_779692

-- Define the rectangle and properties
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (intersection_at_center : Bool)
  (segments_equal_length : Bool)
  (segments_intersect_right_angles : Bool)

-- Define the data given in the problem
def PQRS : Rectangle := 
{ length := 30,
  width := 20,
  intersection_at_center := true,
  segments_equal_length := true,
  segments_intersect_right_angles := true }

-- Define the proof problems
theorem length_of_AB (r : Rectangle) (h : r = PQRS) : 
  let AB := r.width in AB = 20 :=
by 
  sorry

theorem area_of_triangular_piece (r : Rectangle) (h : r = PQRS) : 
  let triangular_area := (r.width * r.width) / 4 in 
  triangular_area = 100 :=
by 
  sorry

theorem area_of_five_sided_piece (r : Rectangle) (h : r = PQRS) : 
  let total_area := r.length * r.width,
      triangular_area := (r.width * r.width) / 4 in
  let remaining_area := total_area - 2 * triangular_area in 
  let five_sided_area := remaining_area / 2 in
  five_sided_area = 200 := 
by 
  sorry

theorem area_of_hole (r : Rectangle) (h : r = PQRS) : 
  let total_area := r.length * r.width,
      triangular_area := (r.width * r.width) / 4,
      square_with_hole_area := 8 * triangular_area in 
  let hole_area := square_with_hole_area - total_area in
  hole_area = 200 :=
by 
  sorry

end length_of_AB_area_of_triangular_piece_area_of_five_sided_piece_area_of_hole_l779_779692


namespace three_pow_gt_n_add_two_mul_two_pow_sub_one_l779_779924

theorem three_pow_gt_n_add_two_mul_two_pow_sub_one (n : ℕ) (hn1 : 2 < n) :
  3^n > (n+2) * 2^(n-1) := sorry

end three_pow_gt_n_add_two_mul_two_pow_sub_one_l779_779924


namespace area_contained_by_graph_l779_779362

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l779_779362


namespace max_sin_cos_product_eq_9_over_2_l779_779030

theorem max_sin_cos_product_eq_9_over_2:
  ∀ (x y z : ℝ),
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 := 
sorry

end max_sin_cos_product_eq_9_over_2_l779_779030


namespace semi_minor_axis_length_l779_779475

def length_of_semi_minor_axis (center : ℝ × ℝ) (focus : ℝ × ℝ) (endpoint_of_semi_major : ℝ × ℝ) : ℝ :=
  let c := abs (center.2 - focus.2)
  let a := abs (center.2 - endpoint_of_semi_major.2)
  real.sqrt (a ^ 2 - c ^ 2)

theorem semi_minor_axis_length :
  length_of_semi_minor_axis (2, -1) (2, -3) (2, 3) = 2 * real.sqrt 3 :=
by
  sorry

end semi_minor_axis_length_l779_779475


namespace total_players_l779_779660

-- Definitions for conditions
def K : Nat := 10
def KK : Nat := 30
def B : Nat := 5

-- Statement of the proof problem
theorem total_players : K + KK - B = 35 :=
by
  -- Proof not required, just providing the statement
  sorry

end total_players_l779_779660


namespace granola_bars_per_box_l779_779163

theorem granola_bars_per_box (num_kids : ℕ) (bars_per_kid : ℕ) (num_boxes : ℕ) : 
  num_kids = 30 → bars_per_kid = 2 → num_boxes = 5 → 
  (num_kids * bars_per_kid) / num_boxes = 12 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    (30 * 2) / 5 = 60 / 5 : by rw mul_comm
    ... = 12       : by norm_num

end granola_bars_per_box_l779_779163


namespace solve_for_x_l779_779998

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (sqrt_cond : sqrt(3 * x / 7) = x) : x = 3 / 7 :=
by
  sorry

end solve_for_x_l779_779998


namespace find_fk_range_x_l779_779419

variable (a b : ℝ → ℝ → ℝ) (k : ℝ) (t x : ℝ)

-- Assuming the conditions in the problem
axiom norm_a : ∥a∥ = 1 /-
  The norm of vector a is 1
-/
axiom norm_b : ∥b∥ = 1 /-
  The norm of vector b is 1
-/
axiom condition : ∥a + k • b∥ = real.sqrt 3 * ∥k • a - b∥ /-
  Given |a + k*b| = sqrt(3) * |k*a - b|
-/
axiom k_pos : 0 < k /-
  k is positive
-/

-- Definition of f(k)
noncomputable def f : ℝ := a • b

-- Theorem 1: f(k) = 4k / (k^2 + 1)
theorem find_fk (k : ℝ) (hk : k > 0)
  (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1)
  (h3 : ∥a + k • b∥ = real.sqrt 3 * ∥k • a - b∥) :
  f k = 4 * k / (k ^ 2 + 1) := sorry

-- Theorem 2: Range of x for any t in [-2, 2]
theorem range_x (k : ℝ) (hk : k > 0)
  (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1)
  (h3 : ∥a + k • b∥ = real.sqrt 3 * ∥k • a - b∥)
  (hfk : f k = 4 * k / (k ^ 2 + 1))
  (ht : ∀ t ∈ set.Icc (-2 : ℝ) 2, f k ≥ x ^ 2 - 2 * t * x - 5 / 2) :
  2 - real.sqrt 7 ≤ x ∧ x ≤ real.sqrt 7 - 2 := sorry

end find_fk_range_x_l779_779419


namespace find_k_l779_779840

theorem find_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2023) : k = 2022 := 
by {
  sorry
}

end find_k_l779_779840


namespace area_contained_by_graph_l779_779361

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l779_779361


namespace eq_exponents_l779_779391

theorem eq_exponents (m n : ℤ) : ((5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n) → (m = 0 ∧ n = 0) :=
by
  sorry

end eq_exponents_l779_779391


namespace log_problem_solution_l779_779831

noncomputable def solve_log_problem (x y : ℝ) : Prop :=
  abs (x - log y) = x + 2 * log y → x = 0 ∧ y = 1

theorem log_problem_solution (x y : ℝ) : solve_log_problem x y :=
sorry

end log_problem_solution_l779_779831


namespace element_is_hydrogen_l779_779379

def molar_mass_H : ℝ := 1.008
def molar_mass_O : ℝ := 16.00
def count_H : ℕ := 2
def count_O : ℕ := 1
def mass_percentage_given : ℝ := 11.11

def total_molar_mass : ℝ :=
  (count_H * molar_mass_H) + (count_O * molar_mass_O)

def mass_percentage_H : ℝ :=
  (count_H * molar_mass_H / total_molar_mass) * 100

theorem element_is_hydrogen : 
  mass_percentage_H ≈ mass_percentage_given :=
by
  sorry

end element_is_hydrogen_l779_779379


namespace pages_used_l779_779346

variable (n o c : ℕ)

theorem pages_used (h_n : n = 3) (h_o : o = 13) (h_c : c = 8) :
  (n + o) / c = 2 :=
  by
    sorry

end pages_used_l779_779346


namespace log_f_2_is_1_div_4_l779_779079

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem log_f_2_is_1_div_4 : log 4 (f 2) = 1 / 4 := by {
  sorry
}

end log_f_2_is_1_div_4_l779_779079


namespace sum_third_row_17x17_grid_l779_779863

-- Definition of the grid and its center placement
def spiral_grid (n : ℕ) : list (ℕ × ℕ) :=
  sorry

-- Define the integers from 1 to 289 placed in the 17 x 17 grid
def grid_integers : list ℕ :=
  list.range' 1 289  -- list.range' gives list from 1 to 289 inclusive

-- Sum of greatest and least number in the third row from the top
theorem sum_third_row_17x17_grid :
  let third_row := find_third_row 17 17 grid_integers in  -- third row from the top in the 17 x 17 grid
  let (least, greatest) := (third_row.least, third_row.greatest) in
  least + greatest = 528 :=
by
  -- Define the third row positions in a \(17 \times 17\) grid given the conditions
  sorry

end sum_third_row_17x17_grid_l779_779863


namespace collinear_vector_ratio_l779_779470

theorem collinear_vector_ratio
  (A B C D E : Type)
  [AddCommGroup A] [Module ℝ A]
  (AB AC BC BE BD : A)
  (lambda mu : ℝ)
  (h1 : BD = (2 / 3) • BC)
  (h2 : E ∈ segment ℝ A (0 : ℝ) 1)
  (h3 : BE = lambda • AB + mu • AC) :
  (lambda + 1) / mu = 1 / 2 :=
sorry

end collinear_vector_ratio_l779_779470


namespace natural_numbers_equal_l779_779912

theorem natural_numbers_equal (a b : ℕ) (h : ∀ n : ℕ, ¬ Nat.coprime (a + n) (b + n)) : a = b := by
  sorry

end natural_numbers_equal_l779_779912


namespace length_MN_l779_779488

noncomputable def parametricCircle (θ : ℝ) : ℝ × ℝ :=
  (3 + 2 * Real.cos θ, -1 + 2 * Real.sin θ)

noncomputable def polarLine (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * (Real.cos θ * Real.cos (θ - Real.pi / 4))

theorem length_MN : ∀ {C θ : ℝ} {l : ℝ} (M N : (ℝ × ℝ)),
    let circle_eq := (x - 3)^2 + (y + 1)^2 = 4
    let line_eq := x + y = 2
    x + y - 2 = 0 eventually
    intersection_in_circle (M N : ℝ × ℝ) : 
    (|M - N| = 4) :=
  sorry

end length_MN_l779_779488


namespace total_height_of_tower_l779_779705

theorem total_height_of_tower :
  let S₃₅ : ℕ := (35 * (35 + 1)) / 2
  let S₆₅ : ℕ := (65 * (65 + 1)) / 2
  S₃₅ + S₆₅ = 2775 :=
by
  let S₃₅ := (35 * (35 + 1)) / 2
  let S₆₅ := (65 * (65 + 1)) / 2
  sorry

end total_height_of_tower_l779_779705


namespace intersection_A_B_l779_779065

def set_A (x : ℝ) : Prop := 2 * x^2 + 5 * x - 3 ≤ 0

def set_B (x : ℝ) : Prop := -2 < x

theorem intersection_A_B :
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -2 < x ∧ x ≤ 1/2} := 
by {
  sorry
}

end intersection_A_B_l779_779065


namespace striped_jerseys_count_l779_779521

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l779_779521


namespace number_of_sets_A_that_satisfy_union_l779_779745

theorem number_of_sets_A_that_satisfy_union :
  ∃ (A : set ℕ), {1, 3} ∪ A = {1, 3, 5} ∧ finset.card 
    {A | {1, 3} ∪ A = {1, 3, 5}} = 4 :=
sorry

end number_of_sets_A_that_satisfy_union_l779_779745


namespace apples_per_slice_is_two_l779_779339

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l779_779339


namespace area_of_veranda_l779_779648

theorem area_of_veranda (room_length room_width veranda_width : ℕ) : 
  room_length = 19 → room_width = 12 → veranda_width = 2 → 
  let total_length := room_length + 2 * veranda_width
  let total_width := room_width + 2 * veranda_width
  let total_area := total_length * total_width
  let room_area := room_length * room_width
  total_area - room_area = 140 :=
by
  intros h1 h2 h3
  have total_length_eq : total_length = 23 := by
    rw [h1, h3]
  have total_width_eq : total_width = 16 := by
    rw [h2, h3]
  have total_area_eq : total_area = 368 := by
    rw [total_length_eq, total_width_eq]
    norm_num
  have room_area_eq : room_area = 228 := by
    rw [h1, h2]
    norm_num
  rw [total_area_eq, room_area_eq]
  norm_num
  done

end area_of_veranda_l779_779648


namespace intersection_A_B_l779_779398

-- Definitions of sets A and B based on the given conditions
def A : Set ℕ := {4, 5, 6, 7}
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- The theorem stating the proof problem
theorem intersection_A_B : A ∩ B = {4, 5} :=
by
  sorry

end intersection_A_B_l779_779398


namespace natural_numbers_equal_l779_779911

theorem natural_numbers_equal (a b : ℕ) (h : ∀ n : ℕ, ¬ Nat.coprime (a + n) (b + n)) : a = b := by
  sorry

end natural_numbers_equal_l779_779911


namespace find_m_l779_779100

-- Define the given vectors and the parallel condition
def vectors_parallel (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (3, 1)
  a.1 * b.2 = a.2 * b.1

-- Statement to be proved
theorem find_m (m : ℝ) : vectors_parallel m → m = 1 / 3 :=
by
  sorry

end find_m_l779_779100


namespace map_length_representation_l779_779194

variable (x : ℕ)

theorem map_length_representation :
  (12 : ℕ) * x = 17 * (72 : ℕ) / 12
:=
sorry

end map_length_representation_l779_779194


namespace factorial_expression_l779_779715

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l779_779715


namespace quotient_when_divided_by_44_is_3_l779_779685

/-
A number, when divided by 44, gives a certain quotient and 0 as remainder.
When dividing the same number by 30, the remainder is 18.
Prove that the quotient in the first division is 3.
-/

theorem quotient_when_divided_by_44_is_3 (N : ℕ) (Q : ℕ) (P : ℕ) 
  (h1 : N % 44 = 0)
  (h2 : N % 30 = 18) :
  N = 44 * Q →
  Q = 3 := 
by
  -- since no proof is required, we use sorry
  sorry

end quotient_when_divided_by_44_is_3_l779_779685


namespace angle_CED_gt_45_l779_779481

-- Definitions of the triangle, angle bisector, and altitude
variable (A B C D E : Type)
variables [Triangle A B C] [AngleBisector A D] [Altitude B E]

-- Statement of the theorem
theorem angle_CED_gt_45 (hABC_acute : IsAcuteTriangle A B C) 
    (hAD_bisector : IsAngleBisector A D)
    (hBE_altitude : IsAltitude B E) : ∠ CED > 45 :=
by
  sorry

end angle_CED_gt_45_l779_779481


namespace quadratic_sequence_exists_shortest_quadratic_sequence_0_to_1996_l779_779670

theorem quadratic_sequence_exists (h k : ℤ) : 
  ∃ (n : ℤ) (a : ℕ → ℤ), a 0 = h ∧ a n = k ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ ↑n → |a i - a (i - 1)| = i^2 := sorry

theorem shortest_quadratic_sequence_0_to_1996 : 
  ∃ (n : ℤ) (a : ℕ → ℤ), n = 20 ∧  
  a 0 = 0 ∧ a 20 = 1996 ∧ 
  ∀ (i : ℕ), 1 ≤ i → i ≤ 20 → |a i - a (i - 1)| = i^2 := sorry

end quadratic_sequence_exists_shortest_quadratic_sequence_0_to_1996_l779_779670


namespace monotonic_increasing_interval_l779_779089

noncomputable def f (x : ℝ) : ℝ :=
  cos^2 (π / 2 * x) + sqrt 3 * sin (π / 2 * x) * cos (π / 2 * x) - 2

theorem monotonic_increasing_interval :
  ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ),
  monotonic_increasing_on f (set.Icc x x) ↔ x ∈ set.Icc (-2/3) (1/3) :=
by
  sorry

end monotonic_increasing_interval_l779_779089


namespace arithmetic_sequence_sum_l779_779809

theorem arithmetic_sequence_sum:
  (∀ n : ℕ, ∃ (a : ℕ → ℕ), 
    (a 1, a 2 are the roots of x^2 - 3*x + 2 = 0) ∧ 
    (∀ n, a n = n)) →
  (∀ n, Σ (k : ℕ) in (range n), 1 / (a k * a (k + 1)) = n / (n + 1))
sorry

end arithmetic_sequence_sum_l779_779809


namespace graph_passes_through_fixed_point_l779_779952

theorem graph_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ x y : ℝ, (x, y) = (0, 0) ∧ y = log a (x + 1) := 
by 
  use 0, 0
  simp [log]
  sorry

end graph_passes_through_fixed_point_l779_779952


namespace sum_of_three_numbers_eq_zero_l779_779597

theorem sum_of_three_numbers_eq_zero (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : (a + b + c) / 3 = a + 20) (h3 : (a + b + c) / 3 = c - 10) (h4 : b = 10) : 
  a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l779_779597


namespace range_of_x_such_that_f_x_minus_2_l779_779173

variable (f : ℝ → ℝ)

def is_even (g : ℝ → ℝ) := ∀ x : ℝ, g x = g (-x)
def is_increasing_on_nonneg (g : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → x ≤ y → g x ≤ g y

theorem range_of_x_such_that_f_x_minus_2 :
  is_even f →
  is_increasing_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f (x - 2) ≤ 1} = set.Icc 0 4 :=
by
  intros h_even h_increasing h_value
  sorry

end range_of_x_such_that_f_x_minus_2_l779_779173


namespace find_m_l779_779461

def hyperbola_m (m : ℝ) : Prop :=
  let a := 1 in
  let b := Real.sqrt m in
  let focus := (3 : ℝ) in
  Real.sqrt (a ^ 2 + b ^ 2) = focus

theorem find_m (m : ℝ) (h : m = 8) : hyperbola_m m :=
  by
    rw [h]
    unfold hyperbola_m
    rfl

end find_m_l779_779461


namespace factorial_expression_l779_779717

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l779_779717


namespace triangle_area_triangle_perimeter_l779_779132

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) := 
  1/2 * b * c * (Real.sin A)

theorem triangle_area (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : A = Real.pi / 3) : 
  area_of_triangle A B C a b c = Real.sqrt 3 / 4 := 
  sorry

theorem triangle_perimeter (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : 4 * Real.cos B * Real.cos C - 1 = 0) 
  (h3 : b + c = 2)
  (h4 : a = 1) :
  a + b + c = 3 :=
  sorry

end triangle_area_triangle_perimeter_l779_779132


namespace minor_premise_statement_l779_779579

-- Defining the conditions as predicates
def square : Type := sorry -- placeholder for the type definition of square
def parallelogram : Type := sorry -- placeholder for the type definition of parallelogram
def opposite_sides_equal (x : parallelogram) : Prop := sorry -- placeholder for the property definition

-- The conditions given in the problem
axiom square_is_parallelogram : ∀ (s : square), parallelogram
axiom opposite_sides_of_parallelogram_equal : ∀ (p : parallelogram), opposite_sides_equal p

-- Predicate to determine the minor premise
def is_minor_premise (statement : Prop) : Prop := statement = (∀ (s : square), parallelogram)

-- The theorem we need to prove: Statement ① is the minor premise
theorem minor_premise_statement : is_minor_premise (∀ (s : square), parallelogram) :=
by
  unfold is_minor_premise,
  exact sorry

end minor_premise_statement_l779_779579


namespace swim_meet_capacity_l779_779610

theorem swim_meet_capacity:
  let cars := 2 in
  let vans := 3 in
  let people_per_car := 5 in
  let people_per_van := 3 in
  let max_people_per_car := 6 in
  let max_people_per_van := 8 in
  let actual_people := (cars * people_per_car) + (vans * people_per_van) in
  let max_capacity := (cars * max_people_per_car) + (vans * max_people_per_van) in
  max_capacity - actual_people = 17 :=
by
  sorry

end swim_meet_capacity_l779_779610


namespace tangent_line_fixed_point_zero_sum_greater_than_two_l779_779813

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  ln (abs (x - 1)) - a / x

theorem tangent_line_fixed_point (a : ℝ) :
(tangent_line : ∀ a : ℝ, passes_through (2, f 2 a) (4, 2)) sorry

theorem zero_sum_greater_than_two {a : ℝ} (h : a < 0) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) :
  x1 + x2 > 2 := sorry

end tangent_line_fixed_point_zero_sum_greater_than_two_l779_779813


namespace linear_function_expression_l779_779073

-- Define a linear function
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x : ℝ, f(x) = k * x + b

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop :=
  3 * f(1) - 2 * f(2) = -5

def condition2 (f : ℝ → ℝ) : Prop :=
  2 * f(0) - f(-1) = 1

-- Final proof statement
theorem linear_function_expression (f : ℝ → ℝ) :
  is_linear_function f →
  condition1 f →
  condition2 f →
  (∀ x : ℝ, f(x) = 3 * x - 2) :=
by
  intros h1 h2 h3
  sorry

end linear_function_expression_l779_779073


namespace Gabe_initial_seat_l779_779568

noncomputable theory

def initial_seating : Type := fin 7 -> fin 7

def Gabe_moves_to_end (seating : initial_seating) : Prop :=
  (∃ (g : fin 7), seating g = 1 ∨ seating g = 7)

def final_seating (initial : initial_seating) (g : fin 7) : initial_seating :=
  λ i, if i = g then 0 -- represent the empty seat when Gabe leaves
        else if seating i = g + 1 then g else -- Hal moves to Gabe's seat
        if seating i = 4 + 3 then 4 else -- Flo's new seat (if 3 seats right of original seat 4)
        if seating i = 1 + 5 then 1 else -- Dan's new seat (1 seat left of original seat 2)
        if seating i = seating 6 then seating 5 else -- Bea and Eva switch seats
        seating i -- Otherwise, they did not move (e.g., Cal)

theorem Gabe_initial_seat : ∃ g : fin 7, Gabe_moves_to_end (final_seating initial_seating g) :=
begin
  sorry,
end

end Gabe_initial_seat_l779_779568


namespace ratio_of_volume_to_surface_area_l779_779004

def volume_of_shape (num_cubes : ℕ) : ℕ :=
  -- Volume is simply the number of unit cubes
  num_cubes

def surface_area_of_shape : ℕ :=
  -- Surface area calculation given in the problem and solution
  12  -- edge cubes (4 cubes) with 3 exposed faces each
  + 16  -- side middle cubes (4 cubes) with 4 exposed faces each
  + 1  -- top face of the central cube in the bottom layer
  + 5  -- middle cube in the column with 5 exposed faces
  + 6  -- top cube in the column with all 6 faces exposed

theorem ratio_of_volume_to_surface_area
  (num_cubes : ℕ)
  (h1 : num_cubes = 9) :
  (volume_of_shape num_cubes : ℚ) / (surface_area_of_shape : ℚ) = 9 / 40 :=
by
  sorry

end ratio_of_volume_to_surface_area_l779_779004


namespace pieces_of_asian_art_l779_779110

theorem pieces_of_asian_art (total_pieces egyptian_pieces : ℕ) 
    (h_total : total_pieces = 992) (h_egyptian : egyptian_pieces = 527) :
    total_pieces - egyptian_pieces = 465 := by 
  rw [h_total, h_egyptian]
  norm_num
  sorry

end pieces_of_asian_art_l779_779110


namespace gumball_water_wednesday_l779_779105

variable (water_Mon_Thu_Sat : ℕ)
variable (water_Tue_Fri_Sun : ℕ)
variable (water_total : ℕ)
variable (water_Wed : ℕ)

theorem gumball_water_wednesday 
  (h1 : water_Mon_Thu_Sat = 9) 
  (h2 : water_Tue_Fri_Sun = 8) 
  (h3 : water_total = 60) 
  (h4 : 3 * water_Mon_Thu_Sat + 3 * water_Tue_Fri_Sun + water_Wed = water_total) : 
  water_Wed = 9 := 
by 
  sorry

end gumball_water_wednesday_l779_779105


namespace modulus_of_complex_sub_l779_779407

noncomputable def x : ℝ := -1
noncomputable def y : ℝ := Real.sqrt 2
noncomputable def i : ℂ := Complex.i

theorem modulus_of_complex_sub (x y : ℝ) (i : ℂ) (h : i^2 = -1) 
(hxy : (x + Real.sqrt 2 * i) / i = y + i) : 
|Complex.ofReal x - y * i| = Real.sqrt 3 :=
by
  sorry

end modulus_of_complex_sub_l779_779407


namespace find_m_plus_n_l779_779228
open Classical

noncomputable theory
open set function

def is_regular_octahedron (vertices : fin 8 → ℝ×ℝ×ℝ) := sorry -- placeholder for octahedron definition
def face_numbers := {x : fin 8 → ℕ // ∀ i, x i ∈ {1, 2, 3, 4, 5, 6, 7, 9}} -- Face numbers assigned

def consecutive (a b : ℕ) := (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) ∨ (abs (a - b) = 1)

def valid_assignment (faces : fin 8 → ℝ×ℝ×ℝ) (assignment : face_numbers) :=
  ∀ i j, consecutive (assignment.val i) (assignment.val j) → 
         ¬ (∃ edge, edge = (faces i, faces j))

def probability_no_consecutive_on_adjacent_faces (m n : ℕ) :=
  ∃ faces : fin 8 → ℝ×ℝ×ℝ, is_regular_octahedron faces ∧
  ∃ assign : face_numbers,
  valid_assignment faces assign ∧
  gcd m n = 1 ∧
  n ≠ 0 ∧
  m / (nat.factorial 7).cast ℝ = m / n -- 7! possible total assignments

theorem find_m_plus_n : 
  ∃ (m n : ℕ), probability_no_consecutive_on_adjacent_faces m n → ∃ k, k = m + n := sorry

end find_m_plus_n_l779_779228


namespace number_of_elements_in_A_l779_779534

-- The set A defined as per the conditions
def A : Set (ℝ × ℝ × ℝ) := { p | ∀ x : ℝ, p.1 + p.2 * Real.sin x + p.3 * Real.cos x = 0 }

-- The statement to prove that A contains exactly one element
theorem number_of_elements_in_A : Set.card A = 1 :=
sorry

end number_of_elements_in_A_l779_779534


namespace solve_for_y_l779_779769

theorem solve_for_y : ∃ y : ℝ, (2010 + y)^2 = y^2 ∧ y = -1005 :=
by
  sorry

end solve_for_y_l779_779769


namespace positive_diff_largest_third_largest_prime_factor_173459_l779_779632

theorem positive_diff_largest_third_largest_prime_factor_173459 :
  let x := 173459
  let pf := [13, 17, 5, 157]
  largest_prime_factor pf = 157 → third_largest_prime_factor pf = 13 → (p_diff : ℕ) = 144 :=
by
  sorry

end positive_diff_largest_third_largest_prime_factor_173459_l779_779632


namespace parabolic_triangle_area_l779_779315

theorem parabolic_triangle_area (n : ℕ) : 
  ∃ m a : ℕ, ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
  y1 = x1^2 ∧ y2 = x2^2 ∧ y3 = x3^2 ∧
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
  x1 = 0 ∧ y1 = 0 ∧
  x2 = a ∧ y2 = a^2 ∧
  x3 = a^2 ∧ y3 = a^4 ∧
  a = 2^(2*n + 1) + 1 ∧
  m % 2 = 1 ∧
  (2^n * m)^2 = 
  (1 / 2 * |x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1| : ℝ) :=
begin
  sorry
end

end parabolic_triangle_area_l779_779315


namespace range_of_p_l779_779181

noncomputable def a_n (n : ℕ) : ℝ := 4 + (-1/2)^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 4 * n + (2/3) * (1 - (-1/2)^n)

theorem range_of_p 
  (p : ℝ)
  (h : ∀ n : ℕ, 0 < n → 1 ≤ p * ((2/3) * (1 - (-1/2)^n)) ∧ p * ((2/3) * (1 - (-1/2)^n)) ≤ 3) :
  2 ≤ p ∧ p ≤ 3 :=
begin
  sorry
end

end range_of_p_l779_779181


namespace smallest_degree_measure_for_WYZ_l779_779900

def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100
def angle_WYZ : ℝ := angle_XYZ - angle_XYW

theorem smallest_degree_measure_for_WYZ : angle_WYZ = 30 :=
by
  sorry

end smallest_degree_measure_for_WYZ_l779_779900


namespace sin_sum_gt_cos_sum_of_non_obtuse_triangle_l779_779200

noncomputable def isNonObtuseTriangle (α β γ : ℝ) : Prop :=
α + β + γ = π ∧ α ≤ π / 2 ∧ β ≤ π / 2 ∧ γ ≤ π / 2

theorem sin_sum_gt_cos_sum_of_non_obtuse_triangle (α β γ : ℝ)
  (h : isNonObtuseTriangle α β γ) :
  sin α + sin β + sin γ > cos α + cos β + cos γ := sorry

end sin_sum_gt_cos_sum_of_non_obtuse_triangle_l779_779200


namespace find_a_l779_779415

noncomputable def regression_constant : ℝ :=
let x := [x1, x2, x3, x4, x5, x6],
    y := [y1, y2, y3, y4, y5, y6],
    n := 6 in
let x_sum := ∑ i in finset.range n, x[i] in
let y_sum := ∑ i in finset.range n, y[i] in
let x_bar := x_sum / n in
let y_bar := y_sum / n in
let a := y_bar - (1 / 4) * x_bar in
a

theorem find_a (x1 x2 x3 x4 x5 x6 y1 y2 y3 y4 y5 y6 : ℝ) 
  (hx_sum : x1 + x2 + x3 + x4 + x5 + x6 = 10)
  (hy_sum : y1 + y2 + y3 + y4 + y5 + y6 = 4) : 
  regression_constant = 1 / 4 :=
by
  -- Define mean values
  let x_mean := (x1 + x2 + x3 + x4 + x5 + x6) / 6
  let y_mean := (y1 + y2 + y3 + y4 + y5 + y6) / 6
  -- Substitute means into regression equation and solve for a
  have : y_mean = 1 / 4 * x_mean + regression_constant := sorry
  -- Prove that a = 1 / 4
  sorry

end find_a_l779_779415


namespace find_f_of_5_l779_779072

-- Define the function f and the condition given in the problem
def f (n : ℤ) : ℤ := 
  ∃ x : ℤ, n = 2 * x + 1 ∧ f(2 * x + 1) = x^2 - 2 * x

-- Define the theorem to prove
theorem find_f_of_5 : f 5 = 0 := 
sorry

end find_f_of_5_l779_779072


namespace find_p_l779_779097

noncomputable def parabola_focus (p : ℝ) := (p / 2, 0 : ℝ)

noncomputable def directrix (p : ℝ) := -p / 2

noncomputable def mid_point (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_eq (m : ℝ) (x₀ η₀ x : ℝ) := η₀ + m * (x - x₀)

axiom problem_conditions : ∃ (p : ℝ), 
  (∃ (A B : ℝ × ℝ), 
    (A.1 = -p / 2) ∧ (A.2 = line_eq (√3) 1 0 (-p / 2)) ∧
    (B.1, B.2) = (p / 2 + 2, (√3) / 2 * p + √3) ∧
    mid_point A B = (1, 0)) ∧
  p > 0

theorem find_p : ∃ (p : ℝ), problem_conditions ∧ p = 2 :=
sorry

end find_p_l779_779097


namespace sufficient_but_not_necessary_condition_l779_779408

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition :
  (a > 2 → a^2 > 2 * a)
  ∧ (¬(a^2 > 2 * a → a > 2)) := by
  sorry

end sufficient_but_not_necessary_condition_l779_779408


namespace exists_integers_m_n_for_inequalities_l779_779887

theorem exists_integers_m_n_for_inequalities (a b : ℝ) (h : a ≠ b) : ∃ (m n : ℤ), 
  (a * (m : ℝ) + b * (n : ℝ) < 0) ∧ (b * (m : ℝ) + a * (n : ℝ) > 0) :=
sorry

end exists_integers_m_n_for_inequalities_l779_779887


namespace coefficient_of_x9_in_P_l779_779025

-- Define the polynomial (P) and its expansion.
def P : ℤ[X] := (2 + 3 * X - 2 * X^2)^5

-- Define the target term with target power for which we want to find the coefficient
def target_power : ℕ := 9

-- Define the expected coefficient as calculated
def expected_coefficient : ℤ := 240

-- The theorem statement which we want to prove
theorem coefficient_of_x9_in_P : coeff P target_power = expected_coefficient :=
by
  sorry -- Proof omitted

end coefficient_of_x9_in_P_l779_779025


namespace number_of_ways_is_64_l779_779827

-- Definition of the problem conditions
def ways_to_sign_up (students groups : ℕ) : ℕ :=
  groups ^ students

-- Theorem statement asserting that for 3 students and 4 groups, the number of ways is 64
theorem number_of_ways_is_64 : ways_to_sign_up 3 4 = 64 :=
by sorry

end number_of_ways_is_64_l779_779827


namespace poly_coeff_sum_is_39_l779_779962

theorem poly_coeff_sum_is_39 (p q r s : ℝ) :
  (∀ x : ℂ, x^4 + (p*x^3 : ℂ) + (q*x^2 : ℂ) + (r*x : ℂ) + (s : ℂ) = 0 → x = 3*I ∨ x = -3*I ∨ x =  1 + 2*I ∨ x = 1 - 2*I) →
  (g : ℂ → ℂ): (∀ x : ℂ, g x = x^4 + p * x^3 + q * x^2 + r * x + s) →
  ((g 3*I = 0) ∧ (g (1+2*I) = 0)) →
  p + q + r + s = 39 :=
by
  sorry

end poly_coeff_sum_is_39_l779_779962


namespace find_number_of_pairs_l779_779881

variable (n : ℕ)
variable (prob_same_color : ℚ := 0.09090909090909091)
variable (total_shoes : ℕ := 12)
variable (pairs_of_shoes : ℕ)

-- The condition on the probability of selecting two shoes of the same color
def condition_probability : Prop :=
  (1 : ℚ) / ((2 * n - 1) : ℚ) = prob_same_color

-- The condition on the total number of shoes
def condition_total_shoes : Prop :=
  2 * n = total_shoes

-- The goal to prove that the number of pairs of shoes is 6 given the conditions
theorem find_number_of_pairs (h1 : condition_probability n) (h2 : condition_total_shoes n) : n = 6 :=
by
  sorry

end find_number_of_pairs_l779_779881


namespace frustum_volume_correct_l779_779683

noncomputable def volume_of_frustum 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) : ℝ :=
  let base_area_original := base_edge_original_pyramid ^ 2
  let volume_original := 1 / 3 * base_area_original * height_original_pyramid
  let similarity_ratio := base_edge_smaller_pyramid / base_edge_original_pyramid
  let volume_smaller := volume_original * (similarity_ratio ^ 3)
  volume_original - volume_smaller

theorem frustum_volume_correct 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) 
(h_orig_base_edge : base_edge_original_pyramid = 16) 
(h_orig_height : height_original_pyramid = 10) 
(h_smaller_base_edge : base_edge_smaller_pyramid = 8) 
(h_smaller_height : height_smaller_pyramid = 5) : 
  volume_of_frustum base_edge_original_pyramid height_original_pyramid base_edge_smaller_pyramid height_smaller_pyramid = 746.66 :=
by 
  sorry

end frustum_volume_correct_l779_779683


namespace problem1_problem2_l779_779659

-- Problem 1
theorem problem1 (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := 
by {
  sorry
}

-- Problem 2
theorem problem2 (a b m n s : ℤ) (h1 : a + b = 0) (h2 : m * n = 1) (h3 : |s| = 3) :
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := 
by {
  sorry
}

end problem1_problem2_l779_779659


namespace ratio_of_divisors_l779_779893

-- Define the given number N
def N : ℕ := 34 * 34 * 63 * 270

-- Define the function to compute the sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, if d % 2 = 1 then d else 0

-- Define the function to compute the sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, if d % 2 = 0 then d else 0

-- Define the statement of the problem
theorem ratio_of_divisors : (sum_odd_divisors N) / (sum_even_divisors N) = 1 / 14 :=
by
  sorry

end ratio_of_divisors_l779_779893


namespace find_angle_between_a_and_c_l779_779445

variables {ℝ} [normed_space ℝ (euclidean_space ℝ (fin 3))]

noncomputable def angle_between_vectors (a b c : euclidean_space ℝ (fin 3)) : ℝ :=
if h : ∃ m : ℝ, m ≠ 0 ∧ ∥a∥ = m ∧ ∥b∥ = m ∧ ∥c∥ = m ∧ a + b = (real.sqrt 3) • c
then real.arccos (real.sqrt 3 / 2)
else 0

theorem find_angle_between_a_and_c (a b c : euclidean_space ℝ (fin 3)) :
  (∥a∥ = ∥b∥ ∧ ∥b∥ = ∥c∥ ∧ ∥c∥ ≠ 0) ∧ (a + b = (real.sqrt 3) • c) →
  angle_between_vectors a b c = π / 6 :=
by
  intro h
  sorry

end find_angle_between_a_and_c_l779_779445


namespace time_3050_minutes_after_midnight_l779_779114

theorem time_3050_minutes_after_midnight :
  let midnight := datetime.mk 2015 1 1 0 0 0 0,
      mins_to_add := 3050,
      added_datetime := time_since midnight (minutes ∷ mins_to_add)
  in added_datetime = datetime.mk 2015 1 3 2 50 0 0 :=
sorry

end time_3050_minutes_after_midnight_l779_779114


namespace problem_statement_l779_779167

variable (a b c : ℝ)

theorem problem_statement 
  (h₀ : a * b + b * c + c * a > a + b + c) 
  (h₁ : a + b + c > 0) 
: a + b + c > 3 := 
sorry

end problem_statement_l779_779167


namespace line_equation_l779_779010

noncomputable def projection (w₁ w₂ : ℝ) := 
  (w₁ * 3 + w₂ * 4) / 25

theorem line_equation {w : ℝ × ℝ} :
  let w := ⟨w.1, w.2⟩;
  (proj := ⟨(9 * w.1 + 12 * w.2) / 25, (12 * w.1 + 16 * w.2) / 25⟩);
  proj = ⟨-9/5, -12/5⟩ →
  w.2 = -(3/4) * w.1 - 15/4 := 
by
  sorry

end line_equation_l779_779010


namespace area_of_abs_inequality_l779_779374

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l779_779374


namespace eggs_from_nancy_is_2_l779_779504

-- Defining the conditions as Lean variables
variables {eggs_from_gertrude eggs_from_blanche eggs_from_martha eggs_dropped eggs_left total_eggs : ℕ}

-- Assign values to known quantities
def eggs_from_gertrude := 4
def eggs_from_blanche := 3
def eggs_from_martha := 2
def eggs_dropped := 2
def eggs_left := 9

-- Define the number of eggs from Nancy
def eggs_from_nancy := total_eggs - (eggs_from_gertrude + eggs_from_blanche + eggs_from_martha)

-- The final proof problem: Prove that Trevor got 2 eggs from Nancy
theorem eggs_from_nancy_is_2 (total_eggs = eggs_left + eggs_dropped) :
  eggs_from_nancy = 2 :=
by
  sorry

end eggs_from_nancy_is_2_l779_779504


namespace isosceles_trapezoid_side_length_is_five_l779_779942

noncomputable def isosceles_trapezoid_side_length (b1 b2 area : ℝ) : ℝ :=
  let h := 2 * area / (b1 + b2)
  let base_diff_half := (b2 - b1) / 2
  Real.sqrt (h^2 + base_diff_half^2)
  
theorem isosceles_trapezoid_side_length_is_five :
  isosceles_trapezoid_side_length 6 12 36 = 5 := by
  sorry

end isosceles_trapezoid_side_length_is_five_l779_779942


namespace sequence_sixth_term_l779_779867

theorem sequence_sixth_term :
  ∀ (a : ℕ → ℚ),
  a 1 = 1 → 
  a 2 = 2 / 3 → 
  (∀ n ≥ 2, 1 / a (n - 1) + 1 / a (n + 1) = 2 / a n) → 
  a 6 = 2 / 7 := 
by {
  intros a ha1 ha2 hrec,
  sorry
}

end sequence_sixth_term_l779_779867


namespace find_t_l779_779790

noncomputable def given_conditions (m n : ℝ) : Prop :=
  (m ≠ 0) ∧ (n ≠ 0) ∧ (3 * m = 2 * n) ∧ (\<m, n\> = 1/2 * m * n) ∧ (n * (t * m + n) = 0)

theorem find_t (m n t : ℝ) (h : given_conditions m n) : t = -3 := 
sorry

end find_t_l779_779790


namespace find_integer_pairs_l779_779755

theorem find_integer_pairs (a b : ℤ) : 
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → d ∣ (a^n + b^n + 1)) → 
  (∃ k₁ k₂ : ℤ, ((a = 2 * k₁) ∧ (b = 2 * k₂ + 1)) ∨ ((a = 3 * k₁ + 1) ∧ (b = 3 * k₂ + 1))) :=
by
  sorry

end find_integer_pairs_l779_779755


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779272

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779272


namespace determine_x_l779_779590

theorem determine_x (x : ℝ) :
  let area_square_1 := (3 * x) ^ 2,
      area_square_2 := (7 * x) ^ 2,
      area_triangle := (1 / 2) * (3 * x) * (7 * x),
      total_area := area_square_1 + area_square_2 + area_triangle
  in total_area = 1360 → x = Real.sqrt (2720 / 119) :=
by
  intros h
  -- the proof steps would go here, but we will skip them
  sorry

end determine_x_l779_779590


namespace a50_value_l779_779054

theorem a50_value {a : ℕ → ℚ} {S : ℕ → ℚ} (hS: ∀ n, S n = ∑ i in finset.range (n+1), a i) 
  (h_initial: a 1 = 2)
  (h_recursive: ∀ n ≥ 2, a n = (3 * (S n) ^ 2) / (3 * S n - 2)) :
  a 50 = -99 / 23536802 :=
by 
  sorry

end a50_value_l779_779054


namespace carey_moved_chairs_l779_779727

theorem carey_moved_chairs
    (total_chairs : ℕ := 74)
    (pat_moved : ℕ := 29)
    (left_to_move : ℕ := 17) :
    (carey_moved : ℕ) :=
  carey_moved = total_chairs - pat_moved - left_to_move := sorry

end carey_moved_chairs_l779_779727


namespace verify_correct_algebraic_operation_l779_779260

theorem verify_correct_algebraic_operation (a b : ℝ) : 
  (3 * a^2 * b - 3 * b * a^2 = 0) :=
by
  -- Utilize the commutative property of multiplication
  have h_comm : a^2 * b = b * a^2 := by rw [mul_comm b (a^2)]
  rw [h_comm]
  sorry

end verify_correct_algebraic_operation_l779_779260


namespace tangent_lines_parallel_to_line_l779_779947

noncomputable def derivative_poly : ℝ -> ℝ := λ x, 3 * x^2 + 1

theorem tangent_lines_parallel_to_line (x y : ℝ) :
  ((y = x^3 + x) ∧ (derivative_poly x = 4)) ↔ ((4 * x - y - 2 = 0) ∨ (4 * x - y + 2 = 0)) :=
by
  sorry

end tangent_lines_parallel_to_line_l779_779947


namespace min_sum_of_dimensions_l779_779220

theorem min_sum_of_dimensions 
  (a b c : ℕ) 
  (h_pos : a > 0) 
  (h_pos_2 : b > 0) 
  (h_pos_3 : c > 0) 
  (h_even : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) 
  (h_vol : a * b * c = 1806) 
  : a + b + c = 56 :=
sorry

end min_sum_of_dimensions_l779_779220


namespace magician_and_assistant_agreement_l779_779681

-- Define the initial setup
def circle : Type := sorry -- assuming existence of a 'circle' type
def points_on_circle (n : ℕ) : set circle := sorry -- points on the circle
def arc (A B : circle) : Type := sorry -- assuming existence of 'arc' type

-- Define the erasure of a point
def erase_point (points : set circle) (A : circle) : set circle := points.erase A

-- Define the magician's goal
def magician_identifies_erased_point (initial_points : set circle) (erased_point : circle) : Prop := 
  ∃ (arc_AB : arc erased_point erased_point), erased_point ∈ initial_points ∧ arc_AB.length > sorry -- length of arc

-- Formalizing the main statement
theorem magician_and_assistant_agreement 
  (initial_points : set circle) 
  (h_points : initial_points.size = 2007) 
  (longest_arc : ∃ (A B : circle), (arc A B ∈ arcs (points_on_circle 2007)) ∧ (∀ (C D : circle), (arc C D ∈ arcs (points_on_circle 2007)) → arc A B.length ≥ arc C D.length))
  (erased_point : circle)
  (h_erase : erased_point ∈ initial_points) 
  (new_points : set circle := erase_point initial_points erased_point) 
  (h_new_points : new_points.size = 2006)
  (new_longest_arc : ∃ (C B : circle), (arc C B ∈ arcs new_points) ∧ (∀ (D E : circle), (arc D E ∈ arcs new_points) → arc C B.length ≥ arc D E.length))
  : magician_identifies_erased_point initial_points erased_point :=
sorry

end magician_and_assistant_agreement_l779_779681


namespace cost_of_child_ticket_is_4_l779_779976

def cost_of_child_ticket (cost_adult cost_total tickets_sold tickets_child receipts_total : ℕ) : ℕ :=
  let tickets_adult := tickets_sold - tickets_child
  let receipts_adult := tickets_adult * cost_adult
  let receipts_child := receipts_total - receipts_adult
  receipts_child / tickets_child

theorem cost_of_child_ticket_is_4 (cost_adult : ℕ) (cost_total : ℕ)
  (tickets_sold : ℕ) (tickets_child : ℕ) (receipts_total : ℕ) :
  cost_of_child_ticket 12 4 130 90 840 = 4 := by
  sorry

end cost_of_child_ticket_is_4_l779_779976


namespace sum_100th_layer_l779_779003

def f : ℕ → ℕ
| 0     := 0
| (n+1) := 3 * f n + 4

theorem sum_100th_layer :
  f 100 = 4 * (3^99 - 1) :=
by
  sorry

end sum_100th_layer_l779_779003


namespace correct_proposition_C_l779_779990

theorem correct_proposition_C (α β : Plane) (l m : Line) : 
  (¬(plane_perpendicular α β) → (¬ ∃ l: Line, line_in_plane l α ∧ line_perpendicular l β)) :=
begin
  sorry
end

end correct_proposition_C_l779_779990


namespace trigonometric_expr_simplification_l779_779640

-- Define the primary angles and trigonometric identities

variables (α : ℝ)

-- Double-angle identities
def identity1 := 1 - real.cos (2 * α) = 2 * real.sin α * real.sin α
def identity2 := real.sin (2 * α) = 2 * real.sin α * real.cos α
def identity3 := real.sin (4 * α) = 2 * real.sin (2 * α) * real.cos (2 * α)

theorem trigonometric_expr_simplification (α : ℝ) :
  (1 - real.cos (2 * α)) * real.cos (real.pi / 4 + α) / 
  (2 * real.sin (2 * α) * real.sin (2 * α) - real.sin (4 * α)) = 
  -(real.sqrt 2) / 4 * real.tan α :=
by
  sorry

end trigonometric_expr_simplification_l779_779640


namespace orthocenter_iff_angleDEF_is_90_l779_779862

open EuclideanGeometry

variables (A B C P Q D E F : Point)

-- Conditions
hypothesis (h_triangle_ABC : triangle A B C)
hypothesis (h_PQ_inside_ABC : inside_triangle P A B C ∧ inside_triangle Q A B C)
hypothesis (h_angles_equal_1 : ∠ACP = ∠BCQ)
hypothesis (h_angles_equal_2 : ∠CAP = ∠BAQ)
hypothesis (h_perpendiculars_drawn: ∀ P, meets (perp P (line BC)) D ∧ meets (perp P (line CA)) E ∧ meets (perp P (line AB)) F)

-- Question Reformulated as an if-and-only-if statement
theorem orthocenter_iff_angleDEF_is_90 : 
  (∠ DEF = 90) ↔ (orthocenter Q (triangle B D F)) :=
sorry

end orthocenter_iff_angleDEF_is_90_l779_779862


namespace find_x_l779_779774

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -1)
def c (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition that (a + 2b) is parallel to c
def parallel_condition (x : ℝ) : Prop :=
  let ab := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  ∃ k : ℝ, (c x = (k * ab.1, k * ab.2))

theorem find_x : ∃ x : ℝ, parallel_condition x ∧ x = -15 :=
by
  use -15
  unfold parallel_condition
  rw [a, b]
  existsi (-3 : ℝ) -- the scalar multiple
  dsimp
  split; refl

end find_x_l779_779774


namespace probability_only_one_product_probability_at_least_2_neither_l779_779577

-- Definitions based on given conditions
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6
def prob_both_AB : ℝ := prob_A * prob_B
def prob_neither_AB : ℝ := (1 - prob_A) * (1 - prob_B)

-- Question (1)
theorem probability_only_one_product (P A B: Type) [ProbSpace P] (hA: prob_A = 0.5) (hB: prob_B = 0.6) (h_ind_A_B : IndepEvents A B) :
  P (A ∪ B \ (A ∩ B)) = 0.5 := sorry

-- Question (2)
theorem probability_at_least_2_neither (P A B: Type) [ProbSpace P] (hA: prob_A = 0.5) (hB: prob_B = 0.6) (h_ind_A_B : IndepEvents A B) :
  let prob_neither := (1 - prob_A) * (1 - prob_B),
      prob_at_least_2 := 1 - (0.8 ^ 3 + 3 * 0.8 ^ 2 * 0.2) in
  prob_at_least_2 = 0.104 := sorry

end probability_only_one_product_probability_at_least_2_neither_l779_779577


namespace real_solution_count_l779_779758

theorem real_solution_count :
  ∃! x : ℝ, (x^2010 + 1) * (∑ i in range (1005), x^(2008 - 2*i) + 1) = 2010 * x^2009 :=
by
  sorry

end real_solution_count_l779_779758


namespace number_of_toothpicks_in_15th_stage_l779_779731

-- Define the function to compute the number of toothpicks at a given stage.
noncomputable def num_toothpicks (stage : ℕ) : ℕ :=
  if stage = 0 then 0 else 3 + ∑ i in finset.range stage, 2 + (i / 3)

theorem number_of_toothpicks_in_15th_stage : num_toothpicks 15 = 61 := by
  sorry

end number_of_toothpicks_in_15th_stage_l779_779731


namespace expand_polynomial_mul_l779_779355

variable {x : ℝ}

def a := 2 * x^17 - 4 * x^8 + x^(-3) + 2
def b := -3 * x^5

theorem expand_polynomial_mul : 
  (a * b) = -6 * x^22 + 12 * x^13 - 6 * x^5 - 3 * x^2 := 
by
  sorry

end expand_polynomial_mul_l779_779355


namespace john_total_amount_l779_779512

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l779_779512


namespace min_value_inequality_l779_779409

-- Given conditions
variables {a b c : ℝ} 
hypothesis a_pos : a > 0
hypothesis b_pos : b > 0
hypothesis c_pos : c > 0

-- Math proof problem statement
theorem min_value_inequality :
  inf {x : ℝ | ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  x = (a^2 + b^2 + c^2) / (ab + 2bc)} = (2 * Real.sqrt 5) / 5 :=
sorry

end min_value_inequality_l779_779409


namespace stopping_probability_l779_779241

/-- There are 4 labeled balls, drawn with replacement until both 'red' and 'flag' are drawn.
    The words are mapped to integers: 'wind' -> 1, 'exhibition' -> 2, 'red' -> 3, 'flag' -> 4.
    Given the 20 sets of random numbers, prove the stopping probability after the third draw is 3/20. -/

theorem stopping_probability :
  let sets := ["411", "231", "324", "412", "112", "443", "213", "144",
               "331", "123", "114", "142", "111", "344", "312", "334", 
               "223", "122", "113", "133"] in 
  let qual_sets := ["324", "144", "133"] in
  let total_sets := list.length sets in
  let num_qual_sets := list.length qual_sets in
  (num_qual_sets : ℚ) / (total_sets : ℚ) = 3 / 20 := sorry

end stopping_probability_l779_779241


namespace range_of_f_l779_779761

def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  set_of (λ y, ∃ x, f(x) = y) = {y : ℝ | y ≠ 3} :=
by
  sorry

end range_of_f_l779_779761


namespace total_students_at_competition_l779_779974

variable (K H N : ℕ)

theorem total_students_at_competition
  (H_eq : H = (3/5) * K)
  (N_eq : N = 2 * (K + H))
  (total_students : K + H + N = 240) :
  K + H + N = 240 :=
by
  sorry

end total_students_at_competition_l779_779974


namespace triangle_equilateral_l779_779157

variable {A B C A' B' C' : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
          [MetricSpace A'] [MetricSpace B'] [MetricSpace C']

-- Define the angle bisectors in both triangles
variable (angle_bisector_A : ∀ {a b c : A} {aa' : A'}, Bisector a b c aa')
variable (angle_bisector_B : ∀ {b a c : B} {bb' : B'}, Bisector b a c bb')
variable (angle_bisector_C : ∀ {c a b : C} {cc' : C'}, Bisector c a b cc')

-- Helper for defining triangle vertices
def AnglesEqual (x y : Type) [MetricSpace x] [MetricSpace y] : Prop := ∀ (a b : x) (c d : y), ∠a b = ∠c d

-- Conditions in the problem
variable (condition1 : ∀ {a b c : A} {a' b' c' : A'}, AnglesEqual a b c a' b' c')
variable (condition2 : ∀ {p q : B} {p' q' : B'}, ∠p q = ∠p' q')

-- Question: Is triangle ABC equilateral?
theorem triangle_equilateral : ∀ {a b c : A} {a' b' c' : A'},
  (angle_bisector_A a b c a') →
  (angle_bisector_B b a c b') →
  (angle_bisector_C c a b c') →
  (condition1 a b c a' b' c') →
  (condition2 a b c a' b' c') →
  EquilateralTriangle a b c :=
sorry

end triangle_equilateral_l779_779157


namespace number_of_valid_sequences_l779_779958

/--
The measures of the interior angles of a convex pentagon form an increasing arithmetic sequence.
Determine the number of such sequences possible if the pentagon is not equiangular, all of the angle
degree measures are positive integers less than 150 degrees, and the smallest angle is at least 60 degrees.
-/

theorem number_of_valid_sequences : ∃ n : ℕ, n = 5 ∧
  ∀ (x d : ℕ),
  x ≥ 60 ∧ x + 4 * d < 150 ∧ 5 * x + 10 * d = 540 ∧ (x + d ≠ x + 2 * d) := 
sorry

end number_of_valid_sequences_l779_779958


namespace max_trig_expression_is_4_5_l779_779032

noncomputable def max_trig_expression : ℝ :=
  sup ((λ x y z, (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z)) '' 
    (set.univ : set (ℝ × ℝ × ℝ)))

theorem max_trig_expression_is_4_5 : max_trig_expression = 4.5 :=
sorry

end max_trig_expression_is_4_5_l779_779032


namespace num_digits_satisfying_inequality_l779_779575

theorem num_digits_satisfying_inequality : 
  (∃ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (3.04 + d / 100.0 * 4.0) > 3.040) 
  ↔ 
  (∃ n : ℕ, n = 8) := by
sorry

end num_digits_satisfying_inequality_l779_779575


namespace cone_lateral_surface_area_l779_779425

theorem cone_lateral_surface_area (L r : ℝ) (hL : L = 5) (hr : r = 3) : 
  ∃ S : ℝ, S = π * r * L ∧ S = 15 * π :=
by {
  obtain rfl : L = 5 := hL,
  obtain rfl : r = 3 := hr,
  use (π * r * L),
  split,
  { 
    exact rfl,
  },
  {
    calc π * r * L = π * 3 * 5 : rfl
    ... = 15 * π : by ring,
  }
}

end cone_lateral_surface_area_l779_779425


namespace sin_of_angle_point_l779_779611

theorem sin_of_angle_point (t : ℝ) (ht : t > 0) : let θ := real.angle_arctan (4 * t / (3 * t)) in real.sin θ = 4 / 5 :=
by 
  let θ := real.angle_arctan (4 * t / (3 * t))
  have h_θ : θ = real.angle_arctan (4 / 3) := by rw [mul_div_cancel' (4 : ℝ) (ne_of_gt (mul_pos (mul_pos (norm_num.neg_pos.2 (by norm_num)) ht)))]
  rw [h_θ, real.sin_arctan_of_denom_pos (by norm_num : 3 > 0)]
  norm_num
  sorry

end sin_of_angle_point_l779_779611


namespace parabola_focus_l779_779027

theorem parabola_focus (x : ℝ) : 
  let parabola := (λ x, 9 * x^2 - 5)
  focus_of_parabola := (0, -179/36)
  in focus_of_parabola = (0, -179/36) :=
by 
  have h := parabola
  have f := (0, -179/36)
  sorry

end parabola_focus_l779_779027


namespace percentage_support_of_surveyed_population_l779_779690

-- Definitions based on the conditions
def men_percentage_support : ℝ := 0.70
def women_percentage_support : ℝ := 0.75
def men_surveyed : ℕ := 200
def women_surveyed : ℕ := 800

-- Proof statement
theorem percentage_support_of_surveyed_population : 
  ((men_percentage_support * men_surveyed + women_percentage_support * women_surveyed) / 
   (men_surveyed + women_surveyed) * 100) = 74 := 
by
  sorry

end percentage_support_of_surveyed_population_l779_779690


namespace max_d_value_l779_779077

theorem max_d_value : 
  ∃ (a b c : ℕ), prime a ∧ prime b ∧ prime c ∧ prime (a + b - c) ∧ prime (a + c - b) ∧ prime (b + c - a) ∧ prime (a + b + c) ∧ (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a < b ∧ b < c ∧ (d = a + b + c - (a + b - c) = 1594) := 
sorry

end max_d_value_l779_779077


namespace max_figures_in_grid_l779_779742

-- Definition of the grid size
def grid_size : ℕ := 9

-- Definition of the figure coverage
def figure_coverage : ℕ := 4

-- The total number of unit squares in the grid is 9 * 9 = 81
def total_unit_squares : ℕ := grid_size * grid_size

-- Each figure covers exactly 4 unit squares
def units_per_figure : ℕ := figure_coverage

-- The number of such 2x2 blocks that can be formed in 9x9 grid.
def maximal_figures_possible : ℕ := (grid_size / 2) * (grid_size / 2)

-- The main theorem to be proved
theorem max_figures_in_grid : 
  maximal_figures_possible = total_unit_squares / units_per_figure := by
  sorry

end max_figures_in_grid_l779_779742


namespace area_of_abs_inequality_l779_779376

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l779_779376


namespace number_of_non_empty_subsets_P_l779_779542

def P : Set ℝ := {x | ∫ t in 0..x, (3 * t^2 - 10 * t + 6) = 0 ∧ x > 0}

theorem number_of_non_empty_subsets_P : (∃ S : Finset (Fin n), S.card = 3) :=
by
  sorry

end number_of_non_empty_subsets_P_l779_779542


namespace simplify_expression_l779_779571

theorem simplify_expression : 
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = 1 / 120 := 
by 
  sorry

end simplify_expression_l779_779571


namespace prob_symmetric_l779_779780

variable {σ : ℝ} -- variance
variable {X : ℝ → ℝ} -- random variable X as a function

-- assuming X follows a normal distribution with mean 3 and variance σ^2
axiom normal_dist : (∀ t, X t) ∼ Normal 3 σ^2

-- given condition P(X > m) = 0.3
axiom prob_condition : Π m, P(X > m) = 0.3

-- proof that P(X > 6 - m) = 0.7
theorem prob_symmetric (m : ℝ) : P(X > 6 - m) = 0.7 := by
  sorry

end prob_symmetric_l779_779780


namespace marcy_drinks_in_250_minutes_l779_779183

-- Define a function to represent that Marcy takes n minutes to drink x liters of water.
def time_to_drink (minutes_per_sip : ℕ) (sip_volume_ml : ℕ) (total_volume_liters : ℕ) : ℕ :=
  let total_volume_ml := total_volume_liters * 1000
  let sips := total_volume_ml / sip_volume_ml
  sips * minutes_per_sip

theorem marcy_drinks_in_250_minutes :
  time_to_drink 5 40 2 = 250 :=
  by
    -- The function definition and its application will show this value holds.
    sorry

end marcy_drinks_in_250_minutes_l779_779183


namespace convex_ngon_obtuse_division_l779_779265

theorem convex_ngon_obtuse_division (n : ℕ) (h : n > 4) (P : convex_ngon n) : 
  ∃ T : list obtuse_triangle , disjoint_union T P ∧ length T = n := 
sorry

end convex_ngon_obtuse_division_l779_779265


namespace squirrel_rainy_days_l779_779600

theorem squirrel_rainy_days (s r : ℕ) (h1 : 20 * s + 12 * r = 112) (h2 : s + r = 8) : r = 6 :=
by {
  -- sorry to skip the proof
  sorry
}

end squirrel_rainy_days_l779_779600


namespace jack_apples_final_count_l779_779505

namespace JackApples

def initialApples := 150
def fractionSoldToJill := 0.30
def fractionSoldToJune := 0.20
def applesGivenToTeacher := 2

theorem jack_apples_final_count : 
  let applesAfterJill := initialApples - (initialApples * fractionSoldToJill).toNat
  let applesAfterJune := applesAfterJill - (applesAfterJill * fractionSoldToJune).toNat
  let finalApples := applesAfterJune - applesGivenToTeacher
  finalApples = 82 := by
  sorry

end JackApples

end jack_apples_final_count_l779_779505


namespace minimum_tables_l779_779316

-- Let's define the problem constants and variables.
variables (X Y Z A B C : ℕ)

-- Conditions: X is the total number of customers, Y is the total number of tables
-- Z is the number of customers who left, and A, B, C are the number of customers 
-- remaining at each of 3 tables respectively.
-- We are to prove that the minimum number of original tables is 3.

theorem minimum_tables (h1 : X = Z + A + B + C)
                      (h2 : Y >= 3) :
                      Y = 3 :=
begin
  sorry
end

end minimum_tables_l779_779316


namespace relay_team_orderings_l779_779517

theorem relay_team_orderings (Jordan Mike Friend1 Friend2 Friend3 : Type) :
  ∃ n : ℕ, n = 12 :=
by
  -- Define the team members
  let team : List Type := [Jordan, Mike, Friend1, Friend2, Friend3]
  
  -- Define the number of ways to choose the 4th and 5th runners
  let ways_choose_45 := 2
  
  -- Define the number of ways to order the first 3 runners
  let ways_order_123 := Nat.factorial 3
  
  -- Calculate the total number of ways
  let total_ways := ways_choose_45 * ways_order_123
  
  -- The total ways should be 12
  use total_ways
  have h : total_ways = 12
  sorry
  exact h

end relay_team_orderings_l779_779517


namespace participant_queue_process_ends_and_max_money_l779_779960

theorem participant_queue_process_ends_and_max_money (n : ℕ) : 
  ∃ k : ℕ, (k ≤ (2^n - n - 1)) ∧ (∃ i : ℕ, (1 ≤ i ∧ i ≤ n) → process_ends i) :=
sorry

-- Definitions
def process_ends (i : ℕ) : Prop :=
  ∀ C : ℕ → ℕ, ∀ j : ℕ, (C j = if j < i then C (j + 1) else C (j - i)) ∧ (j < i)

end participant_queue_process_ends_and_max_money_l779_779960


namespace sequence_nonzero_l779_779309

theorem sequence_nonzero : ∀ n : ℕ, n > 0 → 
  let a := (nat.recOn : ℕ → (ℕ → ℕ) → (ℕ → ℕ))
    (fun _ => 1) 
    (fun _ _ => 2) 
    (fun n r1 r2 => if (r1 * r2) % 2 = 0 then 5 * r2 - 3 * r1 else r2 - r1)
  in a n ≠ 0 :=
by
  sorry

end sequence_nonzero_l779_779309


namespace negation_exists_l779_779602

-- Definitions used in the conditions
def prop1 (x : ℝ) : Prop := x^2 ≥ 1
def neg_prop1 : Prop := ∃ x : ℝ, x^2 < 1

-- Statement to be proved
theorem negation_exists (h : ∀ x : ℝ, prop1 x) : neg_prop1 :=
by
  sorry

end negation_exists_l779_779602


namespace village_water_usage_l779_779135

theorem village_water_usage :
  let water_per_person_per_month := 20 in
  let people_per_2p_households := 2 in
  let number_of_2p_households := 7 in
  let people_per_5p_households := 5 in
  let number_of_5p_households := 3 in
  let total_water_available := 2500 in
  let total_water_need :=
    (number_of_2p_households * people_per_2p_households * water_per_person_per_month) +
    (number_of_5p_households * people_per_5p_households * water_per_person_per_month) in
  let number_of_months := total_water_available / total_water_need in
  number_of_months ≈ 4.31 := by
    sorry

end village_water_usage_l779_779135


namespace projection_of_a_on_b_l779_779446

def a : ℝ × ℝ × ℝ := (-2, 2, -1)
def b : ℝ × ℝ × ℝ := (0, 3, -4)

def dot_prod (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def proj (u v : ℝ × ℝ × ℝ) : ℝ :=
  (dot_prod u v) / (magnitude v)

theorem projection_of_a_on_b : proj a b = 2 := by
  sorry

end projection_of_a_on_b_l779_779446


namespace molecular_weight_AlPO4_correct_l779_779724

-- Noncomputable because we are working with specific numerical values.
noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_P : ℝ := 30.97
noncomputable def atomic_weight_O : ℝ := 16.00

noncomputable def molecular_weight_AlPO4 : ℝ := 
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

theorem molecular_weight_AlPO4_correct : molecular_weight_AlPO4 = 121.95 := by
  sorry

end molecular_weight_AlPO4_correct_l779_779724


namespace semicircles_area_increase_l779_779691

-- Definitions based on conditions
def side1 := 8
def side2 := 12
def radius_small := side1 / 2
def radius_large := side2 / 2

-- Expected correct answer
def expected_answer := 125

-- Main statement to be proven
theorem semicircles_area_increase :
  let area_circle_small := π * radius_small ^ 2
  let area_circle_large := π * radius_large ^ 2
  let ratio := area_circle_large / area_circle_small
  let percent_increase := (ratio - 1) * 100
  percent_increase = expected_answer :=
begin
  -- The logic steps in the solution are evident in the proof translation
  sorry
end

end semicircles_area_increase_l779_779691


namespace find_a_l779_779802

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

noncomputable def problem_statement (a : ℝ) : Prop :=
  is_pure_imaginary (a - (10 / (3 - (1 : ℂ).im * im)))

theorem find_a (a : ℝ) (h : problem_statement a) : a = 3 :=
sorry

end find_a_l779_779802


namespace sum_of_coefficients_l779_779779

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (2 * x + 1)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 →
  a₀ = 1 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 :=
by
  intros h_expand h_a₀
  sorry

end sum_of_coefficients_l779_779779


namespace largest_possible_integer_l779_779679

theorem largest_possible_integer (l : List ℕ) (h_length : l.length = 5) (h_pos : ∀ x ∈ l, x > 0)
    (h_occurs : l.count 8 = 2) (h_median : l.nth_le (l.length / 2) (by simp [h_length]) = 9)
    (h_mean : (l.sum : ℚ) / l.length = 10) : l.max = 15 := by
  sorry

end largest_possible_integer_l779_779679


namespace solve_equation_l779_779573

theorem solve_equation :
  ∀ (x : ℝ), (3 * real.sqrt x + 3 * x ^ (-1/2) = 8) →
  (x = ( (8 + real.sqrt 28) / 6 ) ^ 2 ∨ x = ( (8 - real.sqrt 28) / 6 ) ^ 2) :=
begin
  intros x h,
  sorry
end

end solve_equation_l779_779573


namespace actors_in_one_hour_l779_779141

theorem actors_in_one_hour (actors_per_set : ℕ) (minutes_per_set : ℕ) (total_minutes : ℕ) :
  actors_per_set = 5 → minutes_per_set = 15 → total_minutes = 60 →
  (total_minutes / minutes_per_set) * actors_per_set = 20 :=
by
  intros h1 h2 h3
  sorry

end actors_in_one_hour_l779_779141


namespace cards_difference_l779_779287

theorem cards_difference
  (H : ℕ)
  (F : ℕ)
  (B : ℕ)
  (hH : H = 200)
  (hF : F = 4 * H)
  (hTotal : B + F + H = 1750) :
  F - B = 50 :=
by
  sorry

end cards_difference_l779_779287


namespace h_neg_one_eq_neg_two_l779_779524

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := (Real.sqrt (f x))^3 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_neg_one_eq_neg_two : h (-1) = -2 := by
  sorry

end h_neg_one_eq_neg_two_l779_779524


namespace factor_correct_l779_779752

theorem factor_correct (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end factor_correct_l779_779752


namespace triangle_sine_cosine_l779_779845

theorem triangle_sine_cosine (a b A : ℝ) (B C : ℝ) (c : ℝ) 
  (ha : a = Real.sqrt 7) 
  (hb : b = 2) 
  (hA : A = 60 * Real.pi / 180) 
  (hsinB : Real.sin B = Real.sin B := by sorry)
  (hc : c = 3 := by sorry) :
  (Real.sin B = Real.sqrt 21 / 7) ∧ (c = 3) := 
sorry

end triangle_sine_cosine_l779_779845


namespace all_taps_fill_time_l779_779244

-- Define the rates as constants
def rate_tap1 : ℝ := 1 / 10
def rate_tap2 : ℝ := 1 / 15
def rate_tap3 : ℝ := 1 / 6

-- Define the combined rate function
def combined_rate (r1 r2 r3 : ℝ) : ℝ :=
  r1 + r2 + r3

-- Define the time to fill function
def time_to_fill (combined_rate : ℝ) : ℝ :=
  1 / combined_rate

-- Prove that the time taken by all taps together is 3 hours
theorem all_taps_fill_time :
  time_to_fill (combined_rate rate_tap1 rate_tap2 rate_tap3) = 3 :=
by
  sorry -- Proof to be provided

end all_taps_fill_time_l779_779244


namespace equation_of_perpendicular_line_l779_779678

theorem equation_of_perpendicular_line (c : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0 ∧ 2 * x + y - 5 = 0) → (x - 2 * y - 3 = 0) := 
by
  sorry

end equation_of_perpendicular_line_l779_779678


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779274

-- Define the properties that need to be proven
variables (Q1 Q2 : Prop) (A1 A2 : Prop)

/-- Theorem to prove why joint purchases are popular despite risks -/
theorem joint_purchases_popular : Q1 → A1 :=
begin
  sorry -- proof not provided
end

/-- Theorem to prove why joint purchases are not popular among neighbors for groceries -/
theorem joint_purchases_unpopular_among_neighbors : Q2 → A2 :=
begin
  sorry -- proof not provided
end

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l779_779274


namespace travel_with_decreasing_ticket_prices_l779_779852

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ∀ (i j : ℕ), i ≠ j → ℕ) : 
  cities.card = n ∧
  (∀ i j, i ≠ j → train_prices i j = train_prices j i) ∧
  (∀ i j k l, (i ≠ j ∧ k ≠ l ∧ (i ≠ k ∨ j ≠ l)) → train_prices i j ≠ train_prices k l) →
  ∃ (start : ℕ), ∃ (route : list (ℕ × ℕ)), 
  route.length = n - 1 ∧ 
  (∀ (m : ℕ), m < route.length - 1 → train_prices route.nth m route.nth (m+1) > train_prices route.nth (m+1) route.nth (m+2)) :=
by 
  sorry

end travel_with_decreasing_ticket_prices_l779_779852


namespace tangent_line_to_ellipse_l779_779015

theorem tangent_line_to_ellipse (k : ℝ) :
  (∀ x : ℝ, (x / 2 + 2 * (k * x + 2) ^ 2) = 2) →
  k^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l779_779015


namespace problem1_problem2_l779_779279

-- Problem 1: Proof statement in Lean 4.
theorem problem1 {α : Type*} [addGroup α] [module ℝ α] (a b : α) (α : submodule ℝ α) 
  (h₁ : inner_product_space.is_perp ℝ a b)
  (h₂ : inner_product_space.is_perp ℝ a (coe_submodule α)) :
  (b ∈ α) ∨ (linear_independent ℝ ![b] [α]) := sorry

-- Problem 2: Proof statement in Lean 4.
theorem problem2 {α : Type*} {β : Type*} [affine_space α β] 
  (a b c : β) (α : set β) (β : set β) 
  (h₁ : affine_independent ℝ ![a, b, c]) (h₂ : affine a α) (h₃ : affine b β) (h₄ : affine c β) :
  (affine_plane α β ∧ affine_plane α β) ∨ (α ∪ β ≠ ∅) := sorry

end problem1_problem2_l779_779279


namespace sum_of_edges_l779_779694

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l779_779694


namespace negation_of_tan_exists_l779_779442

theorem negation_of_tan_exists :
  (∃ x : ℝ, tan x = 1) ↔ ∀ x : ℝ, tan x ≠ 1 := sorry

end negation_of_tan_exists_l779_779442


namespace find_a_from_conditions_l779_779427

variables {a c e d : ℝ}
noncomputable def ellipse_eq := (x y : ℝ) ⟶ (x^2 / a^2 + y^2 / 2 = 1)

theorem find_a_from_conditions :
  (∀ x y : ℝ, ellipse_eq x y) ∧ 
  (a > real.sqrt 2) ∧
  (c = real.sqrt (a^2 - 2)) ∧
  (e = c / a) ∧
  (d = abs (- e * c) / real.sqrt (1 + e^2)) ∧
  (2 * d = c) →
  a = real.sqrt 3 :=
sorry

end find_a_from_conditions_l779_779427


namespace area_of_abs_sum_eq_six_l779_779369

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l779_779369


namespace sin_theta_l779_779179

open Real EuclideanGeometry

variables {a b c : ℝ^3}

noncomputable def theta : ℝ :=
  angle b c

theorem sin_theta (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : ¬ Colinear {a, b, c}) (h₄ : ((a ⨯ b) ⨯ c) = (1/2) * ‖b‖ * ‖c‖ * a) :
  sin (theta) = (√3) / 2 :=
sorry

end sin_theta_l779_779179


namespace union_sets_l779_779793

noncomputable theory

open Set

def P (a : ℝ) : Set ℝ := {3, Real.log2 a}
def Q (a b : ℝ) : Set ℝ := {a, b}
def IntersectionCondition (a b : ℝ) : Prop := P a ∩ Q a b = {0}

theorem union_sets (a b : ℝ) (h : IntersectionCondition a b) : P a ∪ Q a b = {3, 0, 1} :=
by
  sorry

end union_sets_l779_779793


namespace disjoint_sets_of_translations_l779_779907

open Set

/-- Define the given set S -/
def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1000000}

/-- Define a subset A of S with exactly 101 elements -/
variable (A : Set ℕ)
variable (hA : A ⊆ S)
variable (hA_card : A.card = 101)

/-- Define D as the set of all possible differences between elements of A -/
def D : Set ℤ := {d | ∃ x y ∈ A, d = x - y}

/-- The main theorem stating the existence of the desired elements -/
theorem disjoint_sets_of_translations : 
  ∃ (t : Fin 100 → ℕ), 
    (∀ i : Fin 100, t i ∈ S) ∧ 
    (∀ i j : Fin 100, i ≠ j → Disjoint (image (λ x, x + t i) A) (image (λ x, x + t j) A)) := 
by
  sorry

end disjoint_sets_of_translations_l779_779907


namespace expression_value_l779_779492

def α : ℝ := 60
def β : ℝ := 20
def AB : ℝ := 1

noncomputable def γ : ℝ := 180 - (α + β)

noncomputable def AC : ℝ := AB * (Real.sin γ / Real.sin β)
noncomputable def BC : ℝ := (Real.sin α / Real.sin γ) * AB

theorem expression_value : (1 / AC - BC) = 2 := by
  sorry

end expression_value_l779_779492


namespace min_n_for_distinct_scores_l779_779657

noncomputable def property_P (m : ℕ) (T : ℕ → ℕ → Prop) :=
  ∀ (players : Finset ℕ),
    players.card = m →
    (∃ i, ∀ j ∈ players, i ≠ j → T i j) ∧ (∃ i, ∀ j ∈ players, i ≠ j → T j i)

noncomputable def distinct_scores_property (n m : ℕ) :=
  ∀ (T : ℕ → ℕ → Prop),
    property_P m T →
    ∃ (scores : ℕ → ℕ), (∑ i in Finset.range n, scores i = n * (n - 1) / 2) ∧ ∀ i j, i ≠ j → scores i ≠ scores j

theorem min_n_for_distinct_scores (m : ℕ) (hm : m ≥ 4) :
  ∃ n, distinct_scores_property n m ∧ n = 2 * m - 3 :=
begin
  sorry
end

end min_n_for_distinct_scores_l779_779657


namespace eccentricity_of_ellipse_line_AB_tangent_to_circle_l779_779428

-- Definition of the ellipse
def ellipse (x y : ℝ) := x^2 + 2 * y^2 = 4

-- Definition stating point A is on the ellipse
def pointA_on_ellipse (A : ℝ × ℝ) := ellipse A.1 A.2

-- Definition of line y = 2
def line_y_eq_2 (B : ℝ × ℝ) := B.2 = 2

-- Definition stating OA is perpendicular to OB
def is_perpendicular (A B : ℝ × ℝ) := A.1 * B.1 + A.2 * B.2 = 0

-- Definition of the circle
def circle (x y : ℝ) := x^2 + y^2 = 2

-- Statement for the eccentricity of the ellipse
theorem eccentricity_of_ellipse :
  let a := 2
  let b := Real.sqrt 2
  let c := Real.sqrt (a^2 - b^2)
  c / a = Real.sqrt 2 / 2 := by
  sorry -- Proof omitted

-- Statement for the line AB tangent to the circle
theorem line_AB_tangent_to_circle (A B : ℝ × ℝ) 
  (hA : pointA_on_ellipse A) 
  (hB : line_y_eq_2 B) 
  (h_perp : is_perpendicular A B) :
  (∃ m c : ℝ, ∀ x, circle x (m * x + c) → false) := by
  sorry -- Proof omitted

end eccentricity_of_ellipse_line_AB_tangent_to_circle_l779_779428


namespace count_multiples_of_70_in_range_200_to_500_l779_779823

theorem count_multiples_of_70_in_range_200_to_500 : 
  ∃! count, count = 5 ∧ (∀ n, 200 ≤ n ∧ n ≤ 500 ∧ (n % 70 = 0) ↔ n = 210 ∨ n = 280 ∨ n = 350 ∨ n = 420 ∨ n = 490) :=
by
  sorry

end count_multiples_of_70_in_range_200_to_500_l779_779823


namespace point_on_curve_point_on_curve_point_on_curve_point_on_curve_l779_779351

theorem point_on_curve (x y : ℝ) (hx1 : x = 0) (hy1 : y = sqrt 2) :
  x^2 + y^2 - 3*x*y + 2 ≠ 0 := by sorry

theorem point_on_curve (x y : ℝ) (hx2 : x = sqrt 2) (hy2 : y = 0) :
  x^2 + y^2 - 3*x*y + 2 ≠ 0 := by sorry

theorem point_on_curve (x y : ℝ) (hx3 : x = -sqrt 2) (hy3 : y = sqrt 2) :
  x^2 + y^2 - 3*(x)*(y) + 2 ≠ 0 := by sorry

theorem point_on_curve (x y : ℝ) (hx4 : x = sqrt 2) (hy4 : y = sqrt 2) :
  x^2 + y^2 - 3*x*y + 2 = 0 := by sorry

end point_on_curve_point_on_curve_point_on_curve_point_on_curve_l779_779351


namespace integer_count_expression_l779_779386

theorem integer_count_expression :
  ∃ (S : Finset ℕ), S.card = 34 ∧ S ⊆ Finset.Icc 1 50 ∧ 
  ∀ n ∈ S, (factorial (n^2 - 1)) % (factorial n ^ n) = 0 := 
by
  -- Define the set of integers from 1 to 50
  let I := Finset.Icc 1 50
  -- Define the condition that (n^2-1)! / (n!)^n is an integer 
  let condition := λ n: ℕ, (factorial (n^2 - 1)) % (factorial n ^ n) = 0
  -- Define set S as the subset of I fulfilling the condition
  let S := I.filter condition
  -- Prove that S has exactly 34 elements
  have hS : S.card = 34 := sorry
  use S, hS
  -- Show that S is a subset of I (already true by construction)
  rw ← Finset.subset_refl S
  -- Show that every element in S satisfies the condition
  intro n hn
  exact Finset.mem_filter.mp hn.right

end integer_count_expression_l779_779386


namespace matrix_reconstruction_l779_779480

theorem matrix_reconstruction {m n : ℕ} (A : matrix (fin m) (fin n) ℤ) :
  (∀ r1 r2 c1 c2 : ℕ, r1 < m → r2 < m → c1 < n → c2 < n →
   A r1 c1 + A r2 c2 = A r1 c2 + A r2 c1) →
  (∃ S : finset (fin m × fin n), S.card ≥ m + n - 1 ∧
   ∀ i j : fin m, ∀ k l : fin n, (i, k) ∉ S → (j, l) ∉ S →
   ∃ A' : matrix (fin m) (fin n) ℤ, (∀ p q : finset (fin m × fin n),
    (p, q) ∈ S → A' p q = A p q) ∧
   (∀ r1 r2 c1 c2, r1 < m → r2 < m → c1 < n → c2 < n →
    A' r1 c1 + A' r2 c2 = A' r1 c2 + A' r2 c1)) :=
sorry

end matrix_reconstruction_l779_779480


namespace onur_biking_distance_l779_779198

-- Definitions based only on given conditions
def Onur_biking_distance_per_day (O : ℕ) := O
def Hanil_biking_distance_per_day (O : ℕ) := O + 40
def biking_days_per_week := 5
def total_distance_per_week := 2700

-- Mathematically equivalent proof problem
theorem onur_biking_distance (O : ℕ) (cond : 5 * (O + (O + 40)) = 2700) : O = 250 := by
  sorry

end onur_biking_distance_l779_779198


namespace evaluate_product_of_floors_and_ceilings_l779_779020

theorem evaluate_product_of_floors_and_ceilings :
  (⌊-4.6⌋ * ⌈4.4⌉) * (⌊-3.6⌋ * ⌈3.4⌉) * (⌊-2.6⌋ * ⌈2.4⌉) * (⌊-1.6⌋ * ⌈1.4⌉) * (⌊-0.6⌋ * ⌈0.4⌉) = -3600 := by
  sorry

end evaluate_product_of_floors_and_ceilings_l779_779020


namespace proposition_3_proposition_6_l779_779062

variables (l m : Line)
variables (α β γ : Plane)

-- Proposition 3
theorem proposition_3 (h1 : l || α) (h2 : l ∈ β) (h3 : α ∩ β = m) : l || m := 
sorry

-- Proposition 6
theorem proposition_6 (h4 : α || β) (h5 : α ∩ γ = l) (h6 : β ∩ γ = m) : l || m := 
sorry

end proposition_3_proposition_6_l779_779062


namespace choose_four_from_seven_l779_779931

theorem choose_four_from_seven : nat.choose 7 4 = 35 := 
by sorry

end choose_four_from_seven_l779_779931


namespace order_to_one_fruit_last_remaining_fruit_banana_impossible_to_make_nothing_remain_l779_779277

section magical_apple_tree

variable (bananas : Nat) (oranges : Nat) : Prop

-- Condition: On a magical apple tree, there are initially 15 bananas and 20 oranges
def initial_fruits (bananas : Nat) (oranges : Nat) : Prop :=
  bananas = 15 ∧ oranges = 20

variable (pick : (Nat × Nat) → (Nat × Nat)) : Prop

-- Condition: You are allowed to pick either one or two fruits at a time
def valid_pick (pick : (Nat × Nat)) : Prop :=
  (fst pick = 1 ∧ snd pick = 0) ∨ (fst pick = 0 ∧ snd pick = 1) ∨
  (fst pick = 2 ∨ snd pick = 2)

-- Condition: If you pick one fruit, an identical one will grow back
-- representing no change in the state
def pick_one_unchanged (fruits : Nat × Nat) (pick : Nat) : Prop :=
  (pick = 1 ∧ fruits.fst = fruits.fst) ∨ (pick = 2 ∧ fruits.snd = fruits.snd)

-- Condition: If you pick two identical fruits, an orange will grow back
def pick_two_identical (fruits : Nat × Nat) : ((Nat × Nat) → (Nat × Nat)) :=
  λ fruits, (fruits.fst - 2, fruits.snd + 1)

-- Condition: If you pick two different fruits, a banana will grow back
def pick_two_different (fruits : Nat × Nat) : ((Nat × Nat) → (Nat × Nat)) :=
  λ fruits, (fruits.fst + 1, fruits.snd - 1)

-- Theorem a: There is an order of picks such that exactly one fruit remains on the tree
theorem order_to_one_fruit :
  ∀ (bananas oranges : Nat), initial_fruits bananas oranges →
  ∃ picks : List (Nat × Nat), (∀ pick ∈ picks, valid_pick pick) ∧
  (bananas = 1 ∧ oranges = 0 ∨ bananas = 0 ∧ oranges = 1) :=
sorry

-- Theorem b: The last remaining fruit will be a banana
theorem last_remaining_fruit_banana :
  ∀ (bananas oranges : Nat) (picks : List (Nat × Nat)),
  initial_fruits bananas oranges → 
  (∀ pick ∈ picks, valid_pick pick) →
  (bananas = 1 ∧ oranges = 0) :=
sorry

-- Theorem c: It is impossible to pick the fruits such that nothing remains on the tree
theorem impossible_to_make_nothing_remain :
  ∀ (bananas oranges : Nat) (picks : List (Nat × Nat)),
  initial_fruits bananas oranges → 
  (∀ pick ∈ picks, valid_pick pick) →
  (bananas ≠ 0 ∨ oranges ≠ 0) :=
sorry

end magical_apple_tree

end order_to_one_fruit_last_remaining_fruit_banana_impossible_to_make_nothing_remain_l779_779277


namespace johns_total_amount_l779_779513

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l779_779513


namespace field_length_double_width_l779_779139

theorem field_length_double_width 
    (w l : ℝ) 
    (pond_area rectangle_area triangle1_area triangle2_area field_area : ℝ)
    (h_length_double_width : l = 2 * w)
    (h_rectangle_area : rectangle_area = 8 * 4)
    (h_triangle1_area : triangle1_area = (1/2) * 4 * 2)
    (h_triangle2_area : triangle2_area = (1/2) * 8 * 3)
    (h_pond_area : pond_area = rectangle_area + triangle1_area + triangle2_area)
    (h_pond_fraction_field : pond_area = (1/18) * field_area)
    (h_field_area : field_area = l * w) :
    l ≈ 41.5692 :=
by
  sorry

end field_length_double_width_l779_779139


namespace time_for_goods_train_to_pass_l779_779297

-- Definition of the conditions
def speed_of_girls_train_kmph : ℝ := 100
def speed_of_goods_train_kmph : ℝ := 235.973122150228
def length_of_goods_train_m : ℝ := 560

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed_kmph: ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Calculation of the relative speed in m/s
def relative_speed_mps : ℝ := 
  kmph_to_mps (speed_of_girls_train_kmph + speed_of_goods_train_kmph)

-- Expected time for the goods train to pass the girl
theorem time_for_goods_train_to_pass : 
  (length_of_goods_train_m / relative_speed_mps) = 6 := 
by
  -- We skip the proof as per instructions
  sorry

end time_for_goods_train_to_pass_l779_779297


namespace invitees_count_l779_779572

theorem invitees_count 
  (packages : ℕ) 
  (weight_per_package : ℕ) 
  (weight_per_burger : ℕ) 
  (total_people : ℕ)
  (H1 : packages = 4)
  (H2 : weight_per_package = 5)
  (H3 : weight_per_burger = 2)
  (H4 : total_people + 1 = (packages * weight_per_package) / weight_per_burger) :
  total_people = 9 := 
by
  sorry

end invitees_count_l779_779572


namespace range_of_x_l779_779044

variable (x y : ℝ)

theorem range_of_x (h1 : 2 * x - y = 4) (h2 : -2 < y ∧ y ≤ 3) :
  1 < x ∧ x ≤ 7 / 2 :=
  sorry

end range_of_x_l779_779044


namespace derivative_at_pi_div_4_l779_779088

def f (x : ℝ) : ℝ := cos x + 2 * sin x

theorem derivative_at_pi_div_4 : deriv f (π / 4) = (sqrt 2) / 2 :=
by
  sorry

end derivative_at_pi_div_4_l779_779088


namespace remainder_845307_div_6_l779_779253

theorem remainder_845307_div_6 :
  let n := 845307
  ∃ r : ℕ, n % 6 = r ∧ r = 3 :=
by
  let n := 845307
  have h_div_2 : ¬(n % 2 = 0) := by sorry
  have h_div_3 : n % 3 = 0 := by sorry
  exact ⟨3, by sorry, rfl⟩

end remainder_845307_div_6_l779_779253


namespace pos_perfect_squares_less_than_mult_36_l779_779115

theorem pos_perfect_squares_less_than_mult_36 :
  { n : ℕ // n > 0 ∧ (n * n) < 2000000 ∧ (n * n) % 36 = 0 }.card = 235 :=
by sorry

end pos_perfect_squares_less_than_mult_36_l779_779115


namespace total_donation_l779_779108

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l779_779108


namespace diagonals_intersect_at_single_point_l779_779410

-- Step d

universe u
variables {α : Type u} [EuclideanSpace α]

def convex_hexagon (hex : EuclideanSpace α → Prop) : Prop :=
  ∃ (A B C D E F : EuclideanSpace α),
  hex A ∧ hex B ∧ hex C ∧ hex D ∧ hex E ∧ hex F ∧
  convex (A :: B :: C :: D :: E :: F :: []) ∧
  area_divides AD BE CF

def area_divides (A B C D E F : EuclideanSpace α) (AD BE CF : EuclideanSpace α → EuclideanSpace α) : Prop :=
  ∀ {P Q R : EuclideanSpace α}, 
  split_by_diameter A D ∧ split_by_diameter B E ∧ split_by_diameter C F

-- Question to prove
theorem diagonals_intersect_at_single_point (hex : EuclideanSpace α → Prop) :
  convex_hexagon hex →
  ∃ P : EuclideanSpace α, 
  P ∈ AD ∧ P ∈ BE ∧ P ∈ CF :=
begin
  sorry
end

end diagonals_intersect_at_single_point_l779_779410


namespace sum_equals_one_l779_779535

def G (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 4
  else 3 * G (n - 1) - 2 * G (n - 2)

noncomputable def infinite_sum : ℕ :=
  ∑' n, 1 / G (2^n)

theorem sum_equals_one : infinite_sum = 1 := by
  sorry

end sum_equals_one_l779_779535


namespace length_ZD_angle_bisector_l779_779153

theorem length_ZD_angle_bisector (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 15) (hXZ : XZ = 17) (is_angle_bisector : True) : ∃ (ZD : ℝ), ZD = Real.sqrt 284.484375 :=
by {
  use Real.sqrt 284.484375,
  sorry
}

end length_ZD_angle_bisector_l779_779153


namespace locus_of_M_l779_779393

noncomputable def issue_problem (A : Point) (O : Point) (r : ℝ)
  (secant : Line) (B C M : Point)
  (tangentB : Line) (tangentC : Line) (D : Point) : 
  Prop :=
  secant.passes_through A ∧ 
  secant.intersects_circle (O, r) B ∧ 
  secant.intersects_circle (O, r) C ∧ 
  tangentB.is_tangent_to_circle_at B (O, r) ∧
  tangentC.is_tangent_to_circle_at C (O, r) ∧
  tangentB.intersects tangentC M ∧
  D.is_projection_of M A O ∧
  is_locus_of_point (M, D.outside_circle_segment O r)
  
theorem locus_of_M (A O : Point) (r : ℝ)
  (secant : Line) (B C M : Point)
  (tangentB : Line) (tangentC : Line) (D : Point) :
  issue_problem A O r secant B C M tangentB tangentC D :=
by
  sorry

end locus_of_M_l779_779393


namespace hcf_of_numbers_is_five_l779_779980

theorem hcf_of_numbers_is_five (a b x : ℕ) (ratio : a = 3 * x) (ratio_b : b = 4 * x)
  (lcm_ab : Nat.lcm a b = 60) (hcf_ab : Nat.gcd a b = 5) : Nat.gcd a b = 5 :=
by
  sorry

end hcf_of_numbers_is_five_l779_779980


namespace part_I_part_II_l779_779938

noncomputable def fuel_consumption (x : ℝ) : ℝ :=
  (1 / 128000) * x^3 - (3 / 80) * x + 8

def distance_A_to_B : ℝ := 100

def time_travelled (speed : ℝ) : ℝ :=
  distance_A_to_B / speed

noncomputable def total_fuel_consumed (speed : ℝ) : ℝ :=
  fuel_consumption speed * time_travelled speed

theorem part_I : total_fuel_consumed 40 = 17.5 :=
  by sorry

noncomputable def h (x : ℝ) : ℝ :=
  (1 / 1280) * x^2 + 800 / x - 15 / 4

theorem part_II : h 80 = 11.25 ∧ (∀ x > 0, x ≤ 120 → h x ≥ h 80) :=
  by sorry

end part_I_part_II_l779_779938


namespace diameter_length_2x_l779_779482

open EuclideanGeometry

noncomputable def Omega : Type := sorry

variables (Ω : Omega) (A B E D : Point) (x : ℝ)

-- Conditions
axiom AB_is_diameter : is_diameter Ω A B
axiom AD_is_tangent_at_A : tangent_to_circle Ω A D
axiom BC_is_tangent_at_B : tangent_to_circle Ω B C
axiom AD_length : length_segment A D = x
axiom BC_length : length_segment B C = 3 * x
axiom DE_bisects_angle : angle_bisector E D A B

-- Proof statement
theorem diameter_length_2x : length_segment A B = 2 * x :=
sorry

end diameter_length_2x_l779_779482


namespace sqrt_x_div_sqrt_y_l779_779021

theorem sqrt_x_div_sqrt_y
  (x y : ℚ)
  (h : (1/3)^2 + (1/4)^2 / (1/5)^2 + (1/6)^2 = 29 * x / (53 * y)) :
  real.sqrt x / real.sqrt y = 91 / 42 :=
by 
  sorry

end sqrt_x_div_sqrt_y_l779_779021


namespace true_supporters_of_rostov_l779_779196

theorem true_supporters_of_rostov
  (knights_liars_fraction : ℕ → ℕ)
  (rostov_support_yes : ℕ)
  (zenit_support_yes : ℕ)
  (lokomotiv_support_yes : ℕ)
  (cska_support_yes : ℕ)
  (h1 : knights_liars_fraction 100 = 10)
  (h2 : rostov_support_yes = 40)
  (h3 : zenit_support_yes = 30)
  (h4 : lokomotiv_support_yes = 50)
  (h5 : cska_support_yes = 0):
  rostov_support_yes - knights_liars_fraction 100 = 30 := 
sorry

end true_supporters_of_rostov_l779_779196


namespace sum_of_two_numbers_l779_779238

theorem sum_of_two_numbers (a b : ℕ) (m n : ℕ) 
  (h1 : a = 3 * m) (h2 : b = 3 * n) 
  (coprime_mn : Nat.gcd m n = 1) 
  (hcf_ab : Nat.gcd a b = 3) 
  (lcm_ab : Nat.lcm a b = 100) 
  (reciprocal_sum : (a⁻¹ + b⁻¹ : ℚ) = 0.3433333333333333) :
  a + b = 36 := 
  sorry

end sum_of_two_numbers_l779_779238


namespace xyz_inequality_l779_779923

theorem xyz_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z ≥ 1) :
    (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
by
  sorry

end xyz_inequality_l779_779923


namespace sum_of_squares_l779_779569

theorem sum_of_squares (R r r1 r2 r3 d d1 d2 d3 : ℝ) 
  (h1 : d^2 = R^2 - 2 * R * r)
  (h2 : d1^2 = R^2 + 2 * R * r1)
  (h3 : d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2) :
  d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2 :=
by
  sorry

end sum_of_squares_l779_779569


namespace factorial_expression_simplification_l779_779722

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l779_779722


namespace find_c_in_triangle_l779_779469

theorem find_c_in_triangle 
  (a b c : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a * b = 2) 
  (angle_C : ℝ) 
  (h3 : angle_C = real.pi / 3)  -- 60 degrees in radians
  : c = real.sqrt 19 := 
by 
  sorry

end find_c_in_triangle_l779_779469


namespace probability_problem_l779_779864

noncomputable def P (A B : Prop) : ℝ := sorry -- Placeholder for probability function

section 

variables (redA whiteA blackA : ℕ) (redB whiteB blackB : ℕ)
variables (totalA totalB : ℕ) -- Total balls in A and B
variables (A1 A2 A3 B : Prop) -- Events A1, A2, A3, and B

-- Conditions
def condition_boxA : Prop :=
  redA = 5 ∧ whiteA = 2 ∧ blackA = 3 ∧ totalA = redA + whiteA + blackA

def condition_boxB : Prop :=
  redB = 4 ∧ whiteB = 3 ∧ blackB = 3 ∧ totalB = redB + whiteB + blackB

-- Event definitions
def event_A1 : Prop := true -- Represents selecting a red ball from box A
def event_A2 : Prop := true -- Represents selecting a white ball from box A
def event_A3 : Prop := true -- Represents selecting a black ball from box A
def event_B : Prop  := true -- Represents selecting a red ball from box B

-- Proof problem
theorem probability_problem :
  condition_boxA →
  condition_boxB →
  P B A1 = 5 / 11 ∧ P B true = 9 / 22 :=
sorry
  
end

end probability_problem_l779_779864


namespace intersection_M_CR_N_l779_779098

open Set

def real_univ : Set ℝ := univ

def M : Set ℝ := { x | x^2 - 2 * x - 8 ≤ 0 }

def N : Set ℝ := { x | (log 2)^(1 - x) > 1 }

def CR_N : Set ℝ := -N

theorem intersection_M_CR_N :
  M ∩ CR_N = { x | -2 ≤ x ∧ x ≤ 1 } :=
by sorry

end intersection_M_CR_N_l779_779098


namespace cos_alpha_add_beta_div2_l779_779069

open Real 

theorem cos_alpha_add_beta_div2 (α β : ℝ) 
  (h_range : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h_cos1 : cos (π/4 + α) = 1/3)
  (h_cos2 : cos (π/4 - β/2) = sqrt 3 / 3) :
  cos (α + β/2) = 5 * sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_div2_l779_779069


namespace actual_plot_area_l779_779703

noncomputable def area_of_triangle_in_acres : Real :=
  let base_cm : Real := 8
  let height_cm : Real := 5
  let area_cm2 : Real := 0.5 * base_cm * height_cm
  let conversion_factor_cm2_to_km2 : Real := 25
  let area_km2 : Real := area_cm2 * conversion_factor_cm2_to_km2
  let conversion_factor_km2_to_acres : Real := 247.1
  area_km2 * conversion_factor_km2_to_acres

theorem actual_plot_area :
  area_of_triangle_in_acres = 123550 :=
by
  sorry

end actual_plot_area_l779_779703


namespace necessary_but_not_sufficient_condition_l779_779945

open Real

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 ≤ a ∧ a ≤ 4) → (a^2 - 4 * a < 0) := 
by
  sorry

end necessary_but_not_sufficient_condition_l779_779945


namespace number_of_zeros_l779_779656

noncomputable def f : ℝ → ℝ := sorry -- Define the function f(x) with given properties

-- Definitions based on conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x
def has_period_3 (f : ℝ → ℝ) := ∀ x, f (x + 3) = f x
def condition_at_2 (f : ℝ → ℝ) := f 2 = 0

-- Final statement to prove the number of zeros in the interval [-3, 3]
theorem number_of_zeros (f : ℝ → ℝ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : condition_at_2 f) :
  ∃ S : set ℝ, S = {x ∈ Icc (-3 : ℝ) 3 | f x = 0} ∧ S.card = 9 :=
sorry

end number_of_zeros_l779_779656


namespace wheel_resolutions_approx_l779_779963

noncomputable def number_of_resolutions_of_wheel
  (radius : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance / (2 * Real.pi * radius)

theorem wheel_resolutions_approx
  (r : ℝ) (d : ℝ) (h_r : r = 22.4) (h_d : d = 703.9999999999999) :
  number_of_resolutions_of_wheel r d ≈ 5 :=
by {
  have h₁ : number_of_resolutions_of_wheel r d = d / (2 * Real.pi * r),
  rw [number_of_resolutions_of_wheel],
  rw [h_r, h_d],
  calc (703.9999999999999 / (2 * 3.14159 * 22.4)) ≈ 5 := sorry
}

end wheel_resolutions_approx_l779_779963


namespace decimal_place_values_38_l779_779866

theorem decimal_place_values_38.82 : 
  let n := 38.82 in
  let first_8_place := "tenths" in
  let first_8_value := 8/10 in
  let second_8_place := "hundredths" in
  let second_8_value := 8/100 in
  (n = 38.82) →
  (first_8_place = "tenths") →
  (second_8_place = "hundredths") →
  (first_8_value = 8/10) →
  (second_8_value = 8/100) →
  (first_8_value + second_8_value = 8/10 + 8/100) :=
begin
  sorry
end

end decimal_place_values_38_l779_779866


namespace value_of_expression_l779_779633

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l779_779633


namespace smallest_m_exists_in_T_l779_779529

open Complex

def T : Set ℂ := {z | ∃ (x y : ℝ), z = x + y * complex.I ∧ (1/2 : ℝ) ≤ x ∧ x ≤ real.sqrt 2 / 2 }

theorem smallest_m_exists_in_T :
  ∃ (m : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (z : ℂ), z ∈ T ∧ z^n = 1) ∧ m = 12 :=
begin
  sorry
end

end smallest_m_exists_in_T_l779_779529


namespace xy_solutions_l779_779040

theorem xy_solutions : 
  ∀ (x y : ℕ), 0 < x → 0 < y →
  (xy ^ 2 + 7) ∣ (x^2 * y + x) →
  (x, y) = (7, 1) ∨ (x, y) = (14, 1) ∨ (x, y) = (35, 1) ∨ (x, y) = (7, 2) ∨ (∃ k : ℕ, x = 7 * k ∧ y = 7) :=
by
  sorry

end xy_solutions_l779_779040


namespace f_decreasing_intervals_f_range_l779_779812

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

-- Define function f
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define first proof statement
theorem f_decreasing_intervals (x : ℝ) (k : ℤ) :
  (∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3) ↔
  by sorry

-- Define second proof statement
theorem f_range (x : ℝ) :
  x ∈ Icc (-Real.pi / 4) (0 : ℝ) →
  f x ∈ Icc (-Real.sqrt 3 + 1) 2 :=
by sorry

end f_decreasing_intervals_f_range_l779_779812


namespace angle_q_measure_l779_779546

theorem angle_q_measure 
  (l k : Line) 
  (P R Q : Point) 
  (h_parallel_lk : l ∥ k) 
  (h_angle_P : m∠P = 100) 
  (h_angle_R : m∠R = 70) 
  : m∠Q = 170 := 
sorry

end angle_q_measure_l779_779546


namespace PA_PB_PC_gt_AB_AC_l779_779495

theorem PA_PB_PC_gt_AB_AC (A B C P : Point) (hABC : Triangle A B C) (hAngle : angle BAC = 120) (hPInside : inside_triangle P A B C) :
  PA + PB + PC > AB + AC := 
sorry

end PA_PB_PC_gt_AB_AC_l779_779495


namespace sum_of_squares_expr_l779_779333

theorem sum_of_squares_expr : 
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 := 
by
  sorry

end sum_of_squares_expr_l779_779333


namespace range_of_a_l779_779811

def f (a x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Ioo (1:ℝ) (⊤ : ℝ), deriv (f a) x ≥ 0) ∧ (∃ x ∈ Ioo 1 2, f a x = 0) →
  a ∈ set.Ioc (4 / 3) 3 :=
by
  intro h
  sorry

end range_of_a_l779_779811


namespace sum_100_consecutive_odd_numbers_l779_779035

theorem sum_100_consecutive_odd_numbers :
  let a1 := 1
  let n := 100
  let d := 2
  let an := a1 + (n - 1) * d
  n/2 * (a1 + an) = 10000 :=
by
  let a1 := 1
  let n := 100
  let d := 2
  let an := a1 + (n - 1) * d
  have h1 : an = a1 + (n - 1) * d := rfl
  have h2 : n / 2 * (a1 + an) = n / 2 * (1 + 199) := by
    rw [h1]
  have h3 : n / 2 * (1 + 199) = 50 * 200 := by
    norm_num
  have h4 : 50 * 200 = 10000 := by
    norm_num
  exact Eq.trans h2 (Eq.trans h3 h4)

end sum_100_consecutive_odd_numbers_l779_779035


namespace student_a_more_stable_l779_779574

-- Definition of variance comparison to determine stability
def more_stable_performance (sA sB : ℝ) : Prop :=
  sA ^ 2 < sB ^ 2

-- The conditions mentioned in the problem
constant avg_same : Prop
constant sA_squared : ℝ := 0.48
constant sB_squared : ℝ := 0.53

-- Statement to prove
theorem student_a_more_stable : more_stable_performance sA_squared sB_squared := 
by 
  sorry

end student_a_more_stable_l779_779574


namespace number_of_outfits_l779_779564

-- Definitions based on conditions a)
def trousers : ℕ := 5
def shirts : ℕ := 7
def jackets : ℕ := 3
def specific_trousers : ℕ := 2
def specific_jackets : ℕ := 2

-- Lean 4 theorem statement to prove the number of outfits
theorem number_of_outfits (trousers shirts jackets specific_trousers specific_jackets : ℕ) :
  (3 * jackets + specific_trousers * specific_jackets) * shirts = 91 :=
by
  sorry

end number_of_outfits_l779_779564


namespace range_of_m_l779_779816

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sin (2 * x)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + m

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x₀ ≥ g x₀ m) → m ≤ Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_m_l779_779816


namespace values_of_x_when_f_eq_1_l779_779389

def piecewise_func (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if -1 < x ∧ x < 2 then x^2
  else 2*x

theorem values_of_x_when_f_eq_1 (x : ℝ) : 
  piecewise_func x = 1 ↔ (x = 1 ∨ x = -1) := by
  sorry

end values_of_x_when_f_eq_1_l779_779389


namespace regression_line_mean_l779_779096

theorem regression_line_mean {(x y : Type)} [AddGroupₓ x] [AddGroupₓ y] [LinearOrder x] [LinearOrder y]
  (hx : ℝ) (hy : ℝ)
  (reg_eq : ∀ x, y = 1.5 * x - 15) :
  (∀ x, ∃ y, y = 1.5 * x - 15) → (∃ ᾰ, ᾰ = 1.5 * hx - 15) :=
by
  sorry

end regression_line_mean_l779_779096


namespace binomial_coefficient_x_neg3_l779_779047

theorem binomial_coefficient_x_neg3 (a : ℝ) (h : a = ∫ x in (1/e:ℝ)..e, (1 / x) dx):
  let b := (1 - a / (1:ℝ))
  ( ∃ C, (b ^ 5) = C * x^(-3)) ∧ C = -80 :=
by
  sorry

end binomial_coefficient_x_neg3_l779_779047


namespace terminating_decimal_count_l779_779039

def count_terminating_decimals (n: ℕ): ℕ :=
  (n / 17)

theorem terminating_decimal_count : count_terminating_decimals 493 = 29 := by
  sorry

end terminating_decimal_count_l779_779039


namespace dot_product_result_l779_779210

-- Define variables and the dot product operation
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- State the conditions as hypotheses
def condition1 : a ⬝ b = 5 :=
sorry

def condition2 : a ⬝ c = -7 :=
sorry

def condition3 : b ⬝ c = 3 :=
sorry

-- The theorem that needs to be proved
theorem dot_product_result : b ⬝ (10 • c - 3 • a) = 15 :=
by 
  have h1 : a ⬝ b = 5 := condition1
  have h2 : a ⬝ c = -7 := condition2
  have h3 : b ⬝ c = 3 := condition3
  calc
  b ⬝ (10 • c - 3 • a) = 10 * (b ⬝ c) + (-3) * (b ⬝ a) : by
      rw [dot_sub, dot_smul, dot_smul]; simp
  ... = 10 * 3 - 3 * 5 : by rw [h1, h3]
  ... = 15 : by ring

end dot_product_result_l779_779210


namespace ratio_YQ_ZT_l779_779199

-- Definition of points and their positions
variables (P Q R S T U V W X Y Z : Type*)
-- The lengths between adjacent points on line PW
variables (length_PQ length_QR length_RS length_ST length_TU length_UV length_VW : ℝ)
-- Given conditions
variables (h_lengths : length_PQ = 1 ∧ length_QR = 1 ∧ length_RS = 1 ∧ length_ST = 1 ∧ length_TU = 1 ∧ length_UV = 1 ∧ length_VW = 1)
variables (non_collinear : ¬ collinear [P, W, X])
variables (Y_on_XR : lies_on Y (segment X R))
variables (Z_on_XW : lies_on Z (segment X W))
variables (parallel_YQ_PX : parallel YQ PX)
variables (parallel_ZT_PX : parallel ZT PX)

-- The theorem to prove
theorem ratio_YQ_ZT : (ratio (length YQ) (length ZT)) = 7 / 6 :=
by
  sorry

end ratio_YQ_ZT_l779_779199


namespace apples_per_slice_l779_779336

theorem apples_per_slice 
  (dozens_apples : ℕ)
  (apples_per_dozen : ℕ)
  (number_of_pies : ℕ)
  (pieces_per_pie : ℕ) :
  dozens_apples = 4 →
  apples_per_dozen = 12 →
  number_of_pies = 4 →
  pieces_per_pie = 6 →
  (dozens_apples * apples_per_dozen) / (number_of_pies * pieces_per_pie) = 2 :=
by
  intros h_dozen h_per_dozen h_pies h_pieces
  rw [h_dozen, h_per_dozen, h_pies, h_pieces]
  norm_num
  sorry

end apples_per_slice_l779_779336


namespace mixed_sum_proof_l779_779383

def mixed_sum : ℚ :=
  3 + 1/3 + 4 + 1/2 + 5 + 1/5 + 6 + 1/6

def smallest_whole_number_greater_than_mixed_sum : ℤ :=
  Int.ceil (mixed_sum)

theorem mixed_sum_proof :
  smallest_whole_number_greater_than_mixed_sum = 20 := by
  sorry

end mixed_sum_proof_l779_779383


namespace aquarium_length_l779_779192

theorem aquarium_length {L : ℝ} (W H : ℝ) (final_volume : ℝ)
  (hW : W = 6) (hH : H = 3) (h_final_volume : final_volume = 54)
  (h_volume_relation : final_volume = 3 * (1/4 * L * W * H)) :
  L = 4 := by
  -- Mathematically translate the problem given conditions and resulting in L = 4.
  sorry

end aquarium_length_l779_779192


namespace min_value_fraction_l779_779799

theorem min_value_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b > 0) (h₃ : 2 * a + b = 1) : 
  ∃ x, x = 8 ∧ ∀ y, (y = (1 / a) + (2 / b)) → y ≥ x :=
sorry

end min_value_fraction_l779_779799


namespace find_a9_l779_779487

theorem find_a9 (a_1 a_2 : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n)
  (h2 : a 7 = 210)
  (h3 : a 1 = a_1)
  (h4 : a 2 = a_2) : 
  a 9 = 550 := by
  sorry

end find_a9_l779_779487


namespace striped_jerseys_count_l779_779518

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l779_779518


namespace ratio_length_to_breadth_l779_779593

-- Definitions of the given conditions
def length_landscape : ℕ := 120
def area_playground : ℕ := 1200
def ratio_playground_to_landscape : ℕ := 3

-- Property that the area of the playground is 1/3 of the area of the landscape
def total_area_landscape (area_playground : ℕ) (ratio_playground_to_landscape : ℕ) : ℕ :=
  area_playground * ratio_playground_to_landscape

-- Calculation that breadth of the landscape
def breadth_landscape (length_landscape total_area_landscape : ℕ) : ℕ :=
  total_area_landscape / length_landscape

-- The proof statement for the ratio of length to breadth
theorem ratio_length_to_breadth (length_landscape area_playground : ℕ) (ratio_playground_to_landscape : ℕ)
  (h1 : length_landscape = 120)
  (h2 : area_playground = 1200)
  (h3 : ratio_playground_to_landscape = 3)
  (h4 : total_area_landscape area_playground ratio_playground_to_landscape = 3600)
  (h5 : breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 30) :
  length_landscape / breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 4 :=
by
  sorry


end ratio_length_to_breadth_l779_779593


namespace max_expression_value_l779_779174

noncomputable def max_value : ℝ := 1.849

theorem max_expression_value
  (p q r : ℝ)
  (h_nonneg_p : 0 ≤ p)
  (h_nonneg_q : 0 ≤ q)
  (h_nonneg_r : 0 ≤ r)
  (h_sum : p + 2 * q + 3 * r = 1) :
  p + 2 * (Real.sqrt (p * q)) + 3 * (Real.cbrt (p * q * r)) ≤ max_value :=
sorry

end max_expression_value_l779_779174


namespace count_solutions_l779_779150

def consistent_products (a b c : ℕ) : Prop :=
  (14 * 4 * a = 14 * 6 * c) ∧ (56 = b * 2 * (c/2))

theorem count_solutions : 
  { (a, b, c) : ℕ × ℕ × ℕ | consistent_products a b c }.toFinset.card = 6 :=
sorry

end count_solutions_l779_779150


namespace sum_smallest_prime_factors_of_735_l779_779254

theorem sum_smallest_prime_factors_of_735 : ∃ p q : ℕ, p.prime ∧ q.prime ∧ p * q ≠ 735 ∧ p + q = 8 :=
by
  sorry

end sum_smallest_prime_factors_of_735_l779_779254


namespace taxpayer_annual_income_l779_779836

theorem taxpayer_annual_income (differential_savings : ℝ) (initial_rate new_rate : ℝ) (I : ℝ)
    (h_initial_rate : initial_rate = 0.40)
    (h_new_rate : new_rate = 0.33)
    (h_savings : differential_savings = 3150) :
  I = 45000 :=
by
  -- Assume the initial and new rates
  have h1 : initial_rate * I - new_rate * I = differential_savings, by sorry
  -- Simplified to: 0.07 * I = 3150
  have h2 : 0.07 * I = 3150, by sorry
  -- Solving for I gives: I = 45000
  have h3 : I = 3150 / 0.07, by sorry
  -- Since 3150 / 0.07 = 45000
  have h4 : I = 45000, by sorry
  exact h4

end taxpayer_annual_income_l779_779836


namespace cos_double_angle_through_point_1_2_l779_779082

theorem cos_double_angle_through_point_1_2 : 
  (∃ α : ℝ, ∃ r : ℝ, r = real.sqrt (1^2 + 2^2) ∧ cos α = 1 / r ∧ real.sqrt (1^2 + 2^2) = real.sqrt 5) → cos (2 * α) = -(3/5) :=
by
  sorry

end cos_double_angle_through_point_1_2_l779_779082


namespace opposite_of_pi_eq_neg_pi_l779_779263

theorem opposite_of_pi_eq_neg_pi (π : Real) (h : π = Real.pi) : -π = -Real.pi :=
by sorry

end opposite_of_pi_eq_neg_pi_l779_779263


namespace radius_of_semicircle_in_triangle_l779_779308

theorem radius_of_semicircle_in_triangle :
  ∀ (B C : ℝ) (A : ℝ) 
    (h : is_isosceles_triangle B C A [BC = 20, height AD = 12]), 
    semicircle_radius B C A h = 12 :=
by
  sorry

end radius_of_semicircle_in_triangle_l779_779308


namespace smallest_n_inequality_l779_779382

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    ∀ m : ℕ, m < n → ¬ (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) :=
by
  sorry

end smallest_n_inequality_l779_779382


namespace probability_five_correct_l779_779242

theorem probability_five_correct :
  let num_people := 6
  let num_total_permutations := (Nat.factorial num_people)
  let num_five_correct := 0
  P(exactly_five_correct num_people) = num_five_correct / num_total_permutations :=
begin
  -- Definitions
  sorry
end

end probability_five_correct_l779_779242


namespace equation_of_tangent_and_value_of_m_max_value_of_h_inequality_l779_779403

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (x : ℝ) (m : ℝ) := (1/2)*x^2 + m*x + (7/2)
noncomputable def l (x : ℝ) := x - 1
noncomputable def h (x : ℝ) (m : ℝ) := log (x + 1) - (x + m)

-- Conditions
axiom m_neg : ∀ m, m < 0

-- Proof 1: Equation of line l and m = -2
theorem equation_of_tangent_and_value_of_m (m : ℝ) : (∀ x, f 1 = l x) ∧ (∀ x, is_tangent_to_g : ℝ → Prop) :=
by
  sorry

-- Proof 2: Maximum value of h(x)
theorem max_value_of_h : ∀ x, h x (-2) ≤ 2 :=
by
  sorry

-- Proof 3: Prove \(f(a+b) - f(2a) < \frac{b-a}{2a}\)
theorem inequality (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) :=
by
  sorry

end equation_of_tangent_and_value_of_m_max_value_of_h_inequality_l779_779403


namespace points_on_sphere_l779_779773

theorem points_on_sphere (r : ℝ) (points : Fin₅ → ℝ³) (h_on_sphere : ∀ (i : Fin₅), dist (points i) (0, 0, 0) = r) :
  ∃ (i j : Fin₅), i ≠ j ∧ dist (points i) (points j) ≤ r * sqrt 2 := 
sorry

end points_on_sphere_l779_779773


namespace factorize_m_square_minus_4m_l779_779356

theorem factorize_m_square_minus_4m (m : ℝ) : m^2 - 4 * m = m * (m - 4) :=
by
  sorry

end factorize_m_square_minus_4m_l779_779356


namespace cosine_identity_l779_779118

-- Define the condition: sin (π / 3 - α) = 1 / 4
def condition (α : ℝ) : Prop := sin (π / 3 - α) = 1 / 4

-- Lean statement for proving the target value
theorem cosine_identity (α : ℝ) (h : condition α) : 
  cos (π / 3 + 2 * α) = -7 / 8 :=
by
  -- The proof goes here
  sorry

end cosine_identity_l779_779118


namespace find_slope_l779_779094

-- Define the context and conditions of the problem
def hyperbola (x y : ℝ) := x^2 - y^2 = 1
def vertex := (1 : ℝ, 0 : ℝ)
def line (k x : ℝ) := k * x - k

noncomputable def slope_of_line := (k : ℝ) : ℝ := 
  if ∃ P Q : ℝ × ℝ, (P.1 = k/(k-1) ∧ P.2 = k/(k-1) ∧ Q.1 = k/(k+1) ∧ Q.2 = -k/(k+1) ∧
    (P.1 - 1)^2 + (P.2 - 0)^2 = 4 * ((Q.1 - 1)^2 + Q.2^2)) 
  then k
  else 0

theorem find_slope : slope_of_line = 3 :=
sorry

end find_slope_l779_779094


namespace average_comparison_l779_779585

theorem average_comparison (x : Fin 100 → ℝ) (a b : ℝ)
  (h1 : (∑ i in Finset.range 40, x i) / 40 = a)
  (h2 : (∑ i in Finset.Ico 40 100, x i) / 60 = b) :
  (∑ i in Finset.range 100, x i) / 100 = (40*a + 60*b) / 100 :=
by
  sorry

end average_comparison_l779_779585


namespace total_spent_l779_779000

theorem total_spent :
  let price_a := 9.95
  let price_b := 12.50
  let price_c := 14.95
  let qty_a := 18
  let qty_b := 23
  let qty_c := 15
  let discount_b := 0.10
  let tax := 0.05
  let total_a := qty_a * price_a
  let total_b := qty_b * price_b
  let discounted_b := total_b * (1 - discount_b)
  let total_c := qty_c * price_c
  let subtotal := total_a + discounted_b + total_c
  let total_tax := subtotal * tax
  let total := subtotal + total_tax
  total = 695.21 :=
begin
  sorry
end

end total_spent_l779_779000


namespace sum_of_selected_elements_l779_779037

-- Define the matrix element formula
def a (i j : ℕ) : ℕ := 1 + 15 * (i - 1) + 3 * (j - 1)

theorem sum_of_selected_elements :
  ∃ σ : (Fin 5 → Fin 5), 
  (∑ i, a (i + 1) (σ i + 1)) = 185 :=
by
  sorry

end sum_of_selected_elements_l779_779037


namespace maria_made_cupcakes_l779_779390

-- Define initial parameters and conditions
def initial_cupcakes : ℕ := 19
def sold_cupcakes : ℕ := 5
def final_cupcakes : ℕ := 24

-- Define the unknown parameter to be proved
def more_cupcakes : ℕ := final_cupcakes - (initial_cupcakes - sold_cupcakes)

-- The theorem we want to prove
theorem maria_made_cupcakes : more_cupcakes = 10 := by
  have initial_cupcakes_minus_sold := initial_cupcakes - sold_cupcakes
  have more := final_cupcakes - initial_cupcakes_minus_sold
  rw [<-more_cupcakes] at more
  exact more

end maria_made_cupcakes_l779_779390


namespace tan_beta_l779_779797

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l779_779797


namespace table_legs_l779_779674

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end table_legs_l779_779674


namespace range_of_function_l779_779764

theorem range_of_function :
  set.range (λ x : ℝ, (3 * x + 4) / (x - 5)) = set.Ioo (-∞) 3 ∪ set.Ioo 3 ∞ :=
sorry

end range_of_function_l779_779764


namespace three_sleep_simultaneously_l779_779650

noncomputable def professors := Finset.range 5

def sleeping_times (p: professors) : Finset ℕ 
-- definition to be filled in, stating that p falls asleep twice.
:= sorry 

def moment_two_asleep (p q: professors) : ℕ 
-- definition to be filled in, stating that p and q are asleep together once.
:= sorry

theorem three_sleep_simultaneously :
  ∃ t : ℕ, ∃ p1 p2 p3 : professors, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (t ∈ sleeping_times p1) ∧
  (t ∈ sleeping_times p2) ∧
  (t ∈ sleeping_times p3) := by
  sorry

end three_sleep_simultaneously_l779_779650


namespace intersection_range_l779_779099

theorem intersection_range (m : ℝ) :
  let O1 := (0, 0)
      R1 := Real.sqrt m
      O2 := (-3, 4)
      R2 := 2
      distO1O2 := Real.sqrt ((-3)^2 + 4^2) in
  distO1O2 = 5 →
  9 < m ∧ m < 49 :=
by
  sorry

end intersection_range_l779_779099


namespace gina_charity_fraction_l779_779043

-- Definitions based on the given conditions
def original_money : ℝ := 400
def money_given_to_mom := original_money * (1/4)
def money_used_for_clothes := original_money * (1/8)
def money_kept := 170
def money_given_to_charity := original_money - money_kept - money_given_to_mom - money_used_for_clothes

-- The statement to prove that the fraction of her money given to charity is 1/5
theorem gina_charity_fraction : money_given_to_charity / original_money = 1/5 :=
by
  sorry

end gina_charity_fraction_l779_779043


namespace new_average_age_combined_l779_779203

theorem new_average_age_combined (nA : ℕ) (avgA : ℕ) (nB : ℕ) (avgB : ℕ) (moved_age : ℕ)
  (h1 : nA = 8) (h2 : avgA = 35) (h3 : nB = 5) (h4 : avgB = 30) (h5 : moved_age = 40) :
  (nA - 1 + nB + 1) * (avgA * nA - moved_age + avgB * nB + moved_age) / (nA + nB) = 33.08 :=
by
  sorry

end new_average_age_combined_l779_779203


namespace Adam_ate_more_than_Bill_l779_779619

-- Definitions
def Sierra_ate : ℕ := 12
def Bill_ate : ℕ := Sierra_ate / 2
def total_pies_eaten : ℕ := 27
def Sierra_and_Bill_ate : ℕ := Sierra_ate + Bill_ate
def Adam_ate : ℕ := total_pies_eaten - Sierra_and_Bill_ate
def Adam_more_than_Bill : ℕ := Adam_ate - Bill_ate

-- Statement to prove
theorem Adam_ate_more_than_Bill :
  Adam_more_than_Bill = 3 :=
by
  sorry

end Adam_ate_more_than_Bill_l779_779619


namespace eval_3f3_minus_f27_l779_779343

def f (x : ℝ) : ℝ := x^3 - 3 * Real.sqrt x

theorem eval_3f3_minus_f27 : 3 * f 3 - f 27 = -19602 := by
  sorry

end eval_3f3_minus_f27_l779_779343


namespace percentage_decrease_correct_l779_779229

-- Defining real numbers for original and new prices
def original_price : ℝ := 800
def new_price : ℝ := 608
def price_decrease := original_price - new_price

-- Defining percentage calculation
def percentage_decrease (dec orig : ℝ) := (dec / orig) * 100

-- The actual theorem statement we want to prove
theorem percentage_decrease_correct : percentage_decrease price_decrease original_price = 24 :=
by
  -- The proof is omitted
  sorry

end percentage_decrease_correct_l779_779229


namespace problem1_problem2_problem3_l779_779563

-- Problem 1
theorem problem1 : ∀ x ∈ ℝ, x^2 - 8 * x + 17 > 0 :=
by
  intros x
  sorry

-- Problem 2
theorem problem2 : ∀ x ∈ ℝ, ( (x + 2)^2 - (x - 3)^2 ≥ 0) → (x ≥ 1/2) :=
by
  intros x h
  sorry

-- Problem 3
theorem problem3 : ∃ n ∈ ℕ, 11 ∣ (6 * n^2 - 7) :=
by
  existsi (5 : ℕ)
  sorry

end problem1_problem2_problem3_l779_779563


namespace jacket_trouser_combinations_l779_779616

theorem jacket_trouser_combinations (jackets : ℕ) (trousers : ℕ) (h1 : jackets = 4) (h2 : trousers = 3) :
  jackets * trousers = 12 :=
by
  rw [h1, h2]
  exact Nat.mul_eq_one_iff.mp sorry

end jacket_trouser_combinations_l779_779616


namespace number_of_people_in_group_l779_779944

theorem number_of_people_in_group (n : ℕ) (h1 : 110 - 60 = 5 * n) : n = 10 :=
by 
  sorry

end number_of_people_in_group_l779_779944


namespace equal_functions_A_l779_779709

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^2 - 2*x - 1
def f₂ (t : ℝ) : ℝ := t^2 - 2*t - 1

-- Theorem stating that f₁ is equal to f₂
theorem equal_functions_A : ∀ x : ℝ, f₁ x = f₂ x :=
by
  intros x
  sorry

end equal_functions_A_l779_779709


namespace max_value_expression_l779_779782

open Real

theorem max_value_expression (k : ℝ) (h : k > 0) : 
  (sup { x | ∃ k > 0, x = (3*k^3 + 3*k) / ((3/2*k^2 + 14) * (14*k^2 + 3/2)) }) = (sqrt 21) / 175 
:= sorry

end max_value_expression_l779_779782


namespace arithmetic_seq_sum_l779_779055

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 3 = 9)
  (h3 : a 5 = 5) :
  S 9 / S 5 = 1 :=
by
  sorry

end arithmetic_seq_sum_l779_779055


namespace area_of_equilateral_triangle_l779_779247

noncomputable theory
open Classical

variable {P Q R M : Type}
variable [EquilateralTriangle P Q R] [InscribedCircleCenter P Q R M]

def inscribed_circle_area_eq (r : ℝ) : Prop :=
  real.pi * r * r = 9 * real.pi

def area_of_triangle_eq (A : ℝ) (B : ℝ) (H : ℝ) : Prop :=
  (A * B) / 2 = H

theorem area_of_equilateral_triangle (r : ℝ) (A : ℝ) (B : ℝ) (H : ℝ)
  (h1 : inscribed_circle_area_eq r)
  (h2 : r = 3)  -- From \pi r^2 = 9 \pi
  (h3 : A = 6 * real.sqrt 3)  -- Derived from the equilateral triangle properties
  (h4 : B = 9)                -- Derived from the equilateral triangle properties
  (h5 : area_of_triangle_eq (6 * real.sqrt 3) 9 27 * real.sqrt 3) :
  H = 27 * real.sqrt 3 :=
sorry

end area_of_equilateral_triangle_l779_779247


namespace fractional_part_rational_l779_779060

def smallest_prime_factor (n : ℤ) : ℤ :=
  sorry -- Implementation for choosing the smallest prime factor

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ (n : ℕ), n ≥ 2 → a (n+1) = smallest_prime_factor (a (n-1) + a n)

theorem fractional_part_rational 
  (a : ℕ → ℤ)
  (hseq : sequence a)
  (x : ℝ) : 
  (∃ (m n : ℤ), n ≠ 0 ∧ x = m / n) :=
  sorry -- Proof omitted

end fractional_part_rational_l779_779060


namespace petya_wins_if_and_only_if_m_ne_n_l779_779788

theorem petya_wins_if_and_only_if_m_ne_n 
  (m n : ℕ) 
  (game : ∀ m n : ℕ, Prop)
  (win_condition : (game m n ↔ m ≠ n)) : 
  Prop := 
by 
  sorry

end petya_wins_if_and_only_if_m_ne_n_l779_779788


namespace gretel_hansel_salary_difference_l779_779449

theorem gretel_hansel_salary_difference :
  let hansel_initial_salary := 30000
  let hansel_raise_percentage := 10
  let gretel_initial_salary := 30000
  let gretel_raise_percentage := 15
  let hansel_new_salary := hansel_initial_salary + (hansel_raise_percentage / 100 * hansel_initial_salary)
  let gretel_new_salary := gretel_initial_salary + (gretel_raise_percentage / 100 * gretel_initial_salary)
  gretel_new_salary - hansel_new_salary = 1500 := sorry

end gretel_hansel_salary_difference_l779_779449


namespace p_squared_plus_13_mod_n_eq_2_l779_779462

theorem p_squared_plus_13_mod_n_eq_2 (p : ℕ) (prime_p : Prime p) (h : p > 3) (n : ℕ) :
  (∃ (k : ℕ), p ^ 2 + 13 = k * n + 2) → n = 2 :=
by
  sorry

end p_squared_plus_13_mod_n_eq_2_l779_779462


namespace apples_per_slice_l779_779335

theorem apples_per_slice 
  (dozens_apples : ℕ)
  (apples_per_dozen : ℕ)
  (number_of_pies : ℕ)
  (pieces_per_pie : ℕ) :
  dozens_apples = 4 →
  apples_per_dozen = 12 →
  number_of_pies = 4 →
  pieces_per_pie = 6 →
  (dozens_apples * apples_per_dozen) / (number_of_pies * pieces_per_pie) = 2 :=
by
  intros h_dozen h_per_dozen h_pies h_pieces
  rw [h_dozen, h_per_dozen, h_pies, h_pieces]
  norm_num
  sorry

end apples_per_slice_l779_779335


namespace meeting_probability_l779_779288

namespace MeetingProbability

-- Define the problem conditions
def valid_time (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 2

def engineers_wait_condition (x y : ℝ) : Prop := ¬ (x > y + 1 ∨ y > x + 1)

def meeting_condition (x y z : ℝ) : Prop :=
  valid_time x ∧ valid_time y ∧ valid_time z ∧
  z > x ∧ z > y ∧ engineers_wait_condition x y

-- Define the probability of the event under the given conditions
theorem meeting_probability : 
  (∫ (z : ℝ) in 0..2, ∫ (y : ℝ) in 0..2, ∫ (x : ℝ) in 0..2,
    if meeting_condition x y z then 1 else 0) / 8 = 7 / 24 :=
by
  sorry

end MeetingProbability

end meeting_probability_l779_779288


namespace total_donation_l779_779109

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l779_779109


namespace travel_time_downstream_upstream_l779_779665

-- Definitions
def v : ℝ := 6 -- Speed of the boat in still water
def u : ℝ := 2 -- Speed of the stream

-- Conditions based on the given problem
axiom condition1 : 16 / (v + u) + 8 / (v - u) = 4
axiom condition2 : 12 / (v + u) + 10 / (v - u) = 4

-- Theorem to prove
theorem travel_time_downstream_upstream : 
  let downstream_speed := v + u,
      upstream_speed := v - u,
      downstream_distance := 24,
      upstream_distance := 24
  in (downstream_distance / downstream_speed) + (upstream_distance / upstream_speed) = 9 := 
by
  intros,
  exact sorry

end travel_time_downstream_upstream_l779_779665


namespace graph_of_g_is_E_l779_779951

def f (x : ℝ) : ℝ :=
  if h1 : -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if h2 : 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if h3 : 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def g (x : ℝ) : ℝ := f x + 2

theorem graph_of_g_is_E : true := sorry

end graph_of_g_is_E_l779_779951


namespace f_2015_2015_l779_779530

noncomputable def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d, d^3) |>.sum

def f (n : ℕ) : ℕ := sum_of_cubes_of_digits n

def f_k (k : ℕ) (n : ℕ) : ℕ :=
  if k = 1 then f n else f (f_k (k-1) n)

theorem f_2015_2015 : f_k 2015 2015 = 371 := sorry

end f_2015_2015_l779_779530


namespace joe_two_kinds_of_fruit_l779_779506

-- Definitions based on the conditions
def meals := ["breakfast", "lunch", "snack", "dinner"] -- 4 meals
def fruits := ["apple", "orange", "banana"] -- 3 kinds of fruits

-- Probability that Joe consumes the same fruit for all meals
noncomputable def prob_same_fruit := (1 / 3) ^ 4

-- Probability that Joe eats at least two different kinds of fruits
noncomputable def prob_at_least_two_kinds := 1 - 3 * prob_same_fruit

theorem joe_two_kinds_of_fruit :
  prob_at_least_two_kinds = 26 / 27 :=
by
  -- Proof omitted for this theorem
  sorry

end joe_two_kinds_of_fruit_l779_779506


namespace apples_per_slice_is_two_l779_779337

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l779_779337


namespace range_of_m_l779_779539

theorem range_of_m (x : ℝ) (m : ℝ) (hx : 0 < x ∧ x < π) 
  (h : Real.cot (x / 3) = m * Real.cot x): m > 3 ∨ m < 0 :=
sorry

end range_of_m_l779_779539


namespace removed_square_possible_positions_l779_779918

noncomputable def parallelepiped_net : Type :=
  { net : Finset (Fin 10) // ∀ {x}, x ∈ net → x < 10 }

def is_contiguous (squares : Finset (Fin 9)) : Prop :=
  sorry /- A property that defines contiguity of the squares -/

def valid_positions_for_removal (net : parallelepiped_net) (pos : Fin 10) : Prop :=
  (net.val.erase pos).card = 9 ∧ is_contiguous (net.val.erase pos)

theorem removed_square_possible_positions :
  ∃ positions : Finset (Fin 10), positions.card = 5 ∧
  ∀ pos ∈ positions, valid_positions_for_removal parallelepiped_net pos := by
  sorry

end removed_square_possible_positions_l779_779918


namespace distance_between_centers_l779_779979

theorem distance_between_centers (R : ℝ) (O₁ O₂ B C : Type)
  [metric_space O₁] [metric_space O₂] [metric_space B] [metric_space C]
  (r_equal : ∀ (P : Type), metric_space P → P = R)
  (intersect : ∃ (B : Type), metric_space B)
  (O₁B_line : ∃ (C : Type), metric_space C)
  (perpendicular : ∃ (O₁ : Type) (O₂ : Type) (C : Type), O₂ ⟂ O₁O₂)
  : dist O₁ O₂ = R * sqrt 3 := 
sorry

end distance_between_centers_l779_779979


namespace game_points_product_l779_779707

def g (n : Nat) : Nat :=
  if n % 10 == 0 then 8
  else if n % 2 == 0 then 3
  else 0

def allie_rolls : List Nat := [5, 4, 1, 2]
def betty_rolls : List Nat := [10, 3, 3, 2]

def compute_points (rolls : List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + g n) 0

def allie_points : Nat := compute_points allie_rolls
def betty_points : Nat := compute_points betty_rolls

def total_product : Nat := allie_points * betty_points

theorem game_points_product : total_product = 66 := by
  -- Allie's points
  have allie_points_computation : allie_points = 6 := by
    simp [compute_points, g, allie_rolls]
  -- Betty's points
  have betty_points_computation : betty_points = 11 := by
    simp [compute_points, g, betty_rolls]
  -- Final product
  have product_computation : total_product = 66 := by
    simp [total_product, allie_points_computation, betty_points_computation]
  -- Conclusion
  exact product_computation

end game_points_product_l779_779707


namespace greatest_integer_solution_l779_779984

theorem greatest_integer_solution :
  ∃ n : ℤ, (n^2 - 17 * n + 72 ≤ 0) ∧ (∀ m : ℤ, (m^2 - 17 * m + 72 ≤ 0) → m ≤ n) ∧ n = 9 :=
sorry

end greatest_integer_solution_l779_779984


namespace distance_center_to_chords_l779_779248

/-- 
  Given a circle with center O and diameter AB, with chords of lengths 12 and 16 drawn through the ends of the diameter, and these chords intersecting on the circumference. Let d1 and d2 be the distances from the center O to these chords. 
  Prove that d1 = 6 and d2 = 8.
--/
theorem distance_center_to_chords (O A B M P Q : ℝ) (ch1 ch2 d1 d2 : ℝ)
  (h1 : ch1 = 12) (h2 : ch2 = 16)
  (h3 : d1 = 6) (h4 : d2 = 8)
  (h5 : perpendicular (line O P) (line (chord A M)))
  (h6 : perpendicular (line O Q) (line (chord B M)))
  (h7 : midpoint P (chord A M))
  (h8 : midpoint Q (chord B M)) :
  d1 = 6 ∧ d2 = 8 := sorry

end distance_center_to_chords_l779_779248


namespace profit_share_difference_l779_779642

theorem profit_share_difference (P : ℝ) (rx ry : ℝ) (hx : rx = 1/2) (hy : ry = 1/3) (hP : P = 800) :
  (let total_parts := (3 + 2 : ℕ) in
   let part_value := P / total_parts in
   let x_share := 3 * part_value in
   let y_share := 2 * part_value in
   x_share - y_share = 160) := 
sorry

end profit_share_difference_l779_779642


namespace tan_beta_l779_779796

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l779_779796


namespace shorter_side_of_rectangular_room_l779_779305

theorem shorter_side_of_rectangular_room 
  (a b : ℕ) 
  (h1 : 2 * a + 2 * b = 52) 
  (h2 : a * b = 168) : 
  min a b = 12 := 
  sorry

end shorter_side_of_rectangular_room_l779_779305


namespace arithmetic_mean_increase_l779_779311

theorem arithmetic_mean_increase :
  ∀ (a : ℕ → ℝ), (∑ i in finset.range 15, a i + (2 * (i + 1))) / 15 = (∑ i in finset.range 15, a i) / 15 + 16 :=
by
  sorry

end arithmetic_mean_increase_l779_779311


namespace sum_of_solutions_l779_779767

theorem sum_of_solutions (x : ℝ) (h : ∀ x, (x ≠ 1) ∧ (x ≠ -1) → ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) )) : 
  (∀ x, (x ≠ 1) ∧ (x ≠ -1) → -15 * x / (x^2 - 1) = 3 * x / (x+1) - 9 / (x-1)) → (x = ( -1 + Real.sqrt 13 ) / 2 ∨ x = ( -1 - Real.sqrt 13 ) / 2) → (x + ( -x ) = -1) :=
by
  sorry

end sum_of_solutions_l779_779767


namespace grape_juice_percent_correct_l779_779833

-- Define the initial mixture and its properties
def initial_mixture_volume : ℝ := 30
def initial_mixture_grape_juice_percent : ℝ := 0.10
def added_pure_grape_juice_volume : ℝ := 10

-- Define the resulting mixture's properties
def final_mixture_volume : ℝ := initial_mixture_volume + added_pure_grape_juice_volume
def initial_grape_juice_volume : ℝ := initial_mixture_volume * initial_mixture_grape_juice_percent
def total_grape_juice_volume : ℝ := initial_grape_juice_volume + added_pure_grape_juice_volume
def grape_juice_percent_in_final_mixture : ℝ := (total_grape_juice_volume / final_mixture_volume) * 100

-- Theorem to be proved
theorem grape_juice_percent_correct : grape_juice_percent_in_final_mixture = 32.5 :=
by
  sorry

end grape_juice_percent_correct_l779_779833


namespace range_of_m_l779_779800

theorem range_of_m (a b m : ℝ) (h1 : 2 * b = 2 * a + b) (h2 : b * b = a * a * b) (h3 : 0 < Real.log b / Real.log m) (h4 : Real.log b / Real.log m < 1) : m > 8 :=
sorry

end range_of_m_l779_779800


namespace weight_difference_l779_779214

theorem weight_difference (brown black white grey : ℕ) 
  (h_brown : brown = 4)
  (h_white : white = 2 * brown)
  (h_grey : grey = black - 2)
  (avg_weight : (brown + black + white + grey) / 4 = 5): 
  (black - brown) = 1 := by
  sorry

end weight_difference_l779_779214


namespace total_payment_after_layoff_l779_779613

theorem total_payment_after_layoff
  (total_employees : ℕ)
  (salary_per_employee : ℕ)
  (fraction_laid_off : ℚ)
  (remaining_employees : ℕ)
  (total_payment : ℕ)
  (h1 : total_employees = 450)
  (h2 : salary_per_employee = 2000)
  (h3 : fraction_laid_off = 1 / 3)
  (h4 : remaining_employees = total_employees - (total_employees / 3))
  (h5 : total_payment = remaining_employees * salary_per_employee) :
  total_payment = 600000 :=
by 
  rw [h1, h2, h3] at h4,
  rw h4 at h5,
  norm_num at h5,
  exact h5

end total_payment_after_layoff_l779_779613


namespace range_of_a_iff_l779_779282

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x| + |x - 1| ≤ a → a ≥ 1

theorem range_of_a_iff (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) ↔ (a ≥ 1) :=
by sorry

end range_of_a_iff_l779_779282


namespace winning_candidate_votes_l779_779972

def received_votes (W V : ℕ) := W = (65.21739130434783 / 100) * V

def total_votes (V W : ℕ) := V = W + 3000 + 5000

theorem winning_candidate_votes (W : ℕ) (V : ℕ) 
  (H1 : received_votes W V) 
  (H2 : total_votes V W) : 
  W ≈ 15000 :=
by
  sorry

end winning_candidate_votes_l779_779972


namespace sin_75_mul_sin_15_eq_one_fourth_l779_779747

theorem sin_75_mul_sin_15_eq_one_fourth :
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by 
  -- stating the equivalent trigonometric identities with radians
  have cofunction_identity : ∀ θ, Real.sin (Real.pi / 2 - θ) = Real.cos θ := sorry,
  have double_angle_identity : ∀ θ, Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ := sorry,
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry,
  -- using the cofunction identity to rewrite sin(75°)
  have h1 : Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = Real.cos (15 * Real.pi / 180) * Real.sin (15 * Real.pi / 180),
  from congrArg (λ x, x * Real.sin (15 * Real.pi / 180)) (cofunction_identity (15 * Real.pi / 180)),
  -- using the double angle identity to simplify cos(15°) * sin(15°)
  rw [h1, double_angle_identity (15 * Real.pi / 180)],
  -- applying sin(30°) = 1/2
  rw [Real.sin (30 * Real.pi / 180), sin_30],
  norm_num,
  done

end sin_75_mul_sin_15_eq_one_fourth_l779_779747


namespace cylindrical_cube_radius_l779_779293

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def volume_cylinder (R h : ℝ) : ℝ := Real.pi * R^2 * h

theorem cylindrical_cube_radius :
  ∀ (R : ℝ), 
  let r := 8.999999999999998,
      h := 6.75 in
  volume_sphere r = volume_cylinder R h → R = 12 :=
by
  intro R,
  simp [volume_sphere, volume_cylinder],
  sorry

end cylindrical_cube_radius_l779_779293


namespace find_a9_l779_779486

variable {α : Type*}

-- Definitions for the arithmetic sequence conditions
variable (a : ℕ → α) [Add α] [DecidableEq α] [OfNat α 16] [OfNat α 1]

-- Stating the conditions given in the problem
def condition_1 : Prop := a 5 + a 7 = (16 : α)
def condition_2 : Prop := a 3 = (1 : α)

-- Define the statement to prove
theorem find_a9 (h1 : condition_1 a) (h2 : condition_2 a) : a 9 = (15 : α) := by
  sorry

end find_a9_l779_779486


namespace largest_difference_of_prime_pairs_eq_148_l779_779820

theorem largest_difference_of_prime_pairs_eq_148 : 
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p + q = 154 ∧ (abs (p - q) = 148) := 
by
  sorry

end largest_difference_of_prime_pairs_eq_148_l779_779820


namespace g_symmetric_about_3_over_2_l779_779733

def g (x : ℝ) : ℝ := |(floor x : ℤ)| - |(floor (2 - x) : ℤ)| + x

theorem g_symmetric_about_3_over_2 : ∀ x : ℝ, g x = g (3 - x) := by
  sorry

end g_symmetric_about_3_over_2_l779_779733


namespace suff_and_not_nec_condition_not_nec_condition_suff_but_not_nec_condition_l779_779653

noncomputable def pi := Real.pi

theorem suff_and_not_nec_condition (k : ℤ) (α : ℝ) :
  (α = pi / 6 + k * pi) -> (cos (2 * α) = 1 / 2) :=
by
  sorry

theorem not_nec_condition (α : ℝ) :
  (cos (2 * α) = 1 / 2) -> ∃ k : ℤ, α = pi / 6 + k * pi :=
by
  sorry

theorem suff_but_not_nec_condition (α: ℝ) (k: ℤ):
  (α = pi / 6 + k * pi) ↔ (cos (2 * α) = 1 / 2) ∧ ¬ (∀ α, (cos (2 * α) = 1 / 2) -> α = pi / 6 + k * pi) :=
by
  split
  case mp => intros h; exact ⟨suff_and_not_nec_condition k α h, not_nec_condition h⟩
  case mpr => sorry

end suff_and_not_nec_condition_not_nec_condition_suff_but_not_nec_condition_l779_779653


namespace root_in_interval_l779_779325

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l779_779325


namespace max_int_solution_of_inequality_system_l779_779957

theorem max_int_solution_of_inequality_system :
  ∃ (x : ℤ), (∀ (y : ℤ), (3 * y - 1 < y + 1) ∧ (2 * (2 * y - 1) ≤ 5 * y + 1) → y ≤ x) ∧
             (3 * x - 1 < x + 1) ∧ (2 * (2 * x - 1) ≤ 5 * x + 1) ∧
             x = 0 :=
by
  sorry

end max_int_solution_of_inequality_system_l779_779957


namespace sum_of_special_indices_l779_779344

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2020 / 2021 ∧ ∀ k, k ≥ 1 → 
    ∃ m n : ℕ, nat.coprime m n ∧ a k = m / n ∧ a (k+1) = (m + 18) / (n + 19)

noncomputable def sum_of_indices_property (a : ℕ → ℚ) (n : ℕ) : Prop :=
  ∃ t : ℕ, a n = t / (t + 1)

theorem sum_of_special_indices {a : ℕ → ℚ}
  (h : sequence a) :
  (finset.univ.filter (λ n, sum_of_indices_property a n)).sum (λ x, x) = 59 :=
sorry

end sum_of_special_indices_l779_779344


namespace max_value_x2_plus_2xy_l779_779939

open Real

theorem max_value_x2_plus_2xy (x y : ℝ) (h : x + y = 5) : 
  ∃ (M : ℝ), (M = x^2 + 2 * x * y) ∧ (∀ z w : ℝ, z + w = 5 → z^2 + 2 * z * w ≤ M) :=
by
  sorry

end max_value_x2_plus_2xy_l779_779939


namespace systematic_sampling_41st_number_l779_779137

theorem systematic_sampling_41st_number :
  ∀ (students : List Nat) (sample_size : Nat) (num_parts : Nat) (group_size : Nat) (first_selected : Nat) (k : Nat),
    (students = List.range' 1 1000 + 1) →
    (sample_size = 50) →
    (num_parts = 50) →
    (group_size = 20) →
    (first_selected = 10) →
    (k = 41) →
    (num_parts * group_size = 1000) →
    (k > 0) →
    (k ≤ 50) →
    students[((k - 1) * group_size + first_selected) - 1] = 810 :=
by
  intros students sample_size num_parts group_size first_selected k
  intro h_students
  intro h_sample_size
  intro h_num_parts
  intro h_group_size
  intro h_first_selected
  intro h_k
  intro h_correct_partition
  intro h_k_positive
  intro h_k_range
  sorry

end systematic_sampling_41st_number_l779_779137


namespace C_01_Polish_l779_779652

open Set Filter Metric

variables {α : Type} [TopologicalSpace α] [MetricSpace α]

def C_01 : Type := { f : α → ℝ // Continuous f }

instance : MetricSpace C_01 :=
{ dist := λ f g, supr (λ x, |(f x : ℝ) - (g x : ℝ)|),
  dist_self := sorry,
  eq_of_dist_eq_zero := sorry,
  dist_comm := sorry,
  dist_triangle := sorry }

theorem C_01_Polish : continuous f → ∃ (d : MetricSpace α), PolishSpace α :=
begin
  sorry
end

end C_01_Polish_l779_779652


namespace determine_m_l779_779950

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ Real.pi then 2 * Real.sin x else |Real.cos x|

theorem determine_m (m : ℝ) :
  (∃ g, g = λ x : ℝ, f x - m ∧ ∃ l, 0 ≤ l ∧ l ≤ 2 * Real.pi ∧ g l = 0 ∧
       (∃ a b c d, (a ≠ b ∧ b ≠ c ∧ c ≠ d) ∧ g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0)) ↔
  m ∈ set.Ioo 0 1 :=
by
  sorry

end determine_m_l779_779950


namespace range_of_function_l779_779231

-- Define the function
def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 3)

-- The proof statement
theorem range_of_function : set.Ioo (-1 : ℝ) (1 / 3 : ℝ) = {y : ℝ | ∃ x : ℝ, y = f x} :=
sorry

end range_of_function_l779_779231


namespace circumradius_right_triangle_l779_779997

theorem circumradius_right_triangle (a b c : ℝ) (h₁ : a = 10) (h₂ : b = 8) (h₃ : c = 6) (h₄ : c^2 + b^2 = a^2) :
  ∃ R : ℝ, R = a / 2 ∧ R = 5 :=
by {
  use (a / 2),
  split,
  { refl },
  { rw h₁, norm_num }
}

end circumradius_right_triangle_l779_779997


namespace find_perimeter_l779_779956

noncomputable def perimeter_of_triangle_def (DP DE PE: ℝ) (r: ℝ) : ℝ :=
  let s := (DE + 2 * (DP + PE)) / 2;
  2 * s

theorem find_perimeter (r DP PE DE: ℝ) (h1: r = 30) (h2: DP = 26) (h3: PE = 32) (DE := DP + 26 + PE + 32):
  perimeter_of_triangle_def DP DE PE r ≈ 442.28 :=
by
  sorry

end find_perimeter_l779_779956


namespace tan_alpha_eq_two_then_reciprocal_sin_two_alpha_l779_779400

theorem tan_alpha_eq_two_then_reciprocal_sin_two_alpha :
  ∀ (α : ℝ), tan α = 2 → 1 / sin (2 * α) = 5 / 4 :=
by
  intros α h
  sorry

end tan_alpha_eq_two_then_reciprocal_sin_two_alpha_l779_779400


namespace exists_root_between_1_1_and_1_2_l779_779329

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_between_1_1_and_1_2 :
  ∃ x, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have pf1 : f 1.1 = -0.59 := by norm_num1
  have pf2 : f 1.2 = 0.84 := by norm_num1
  apply exists_between_of_sign_change pf1 pf2
  sorry

end exists_root_between_1_1_and_1_2_l779_779329


namespace no_reordered_power_of_2_l779_779499

theorem no_reordered_power_of_2 :
  ¬∃ a b : ℕ, a ≠ b ∧ 
    (∀ digit ∈ (nat.digits 10 (2^a)), digit ≠ 0) ∧
    (multiset.sort (nat.digits 10 (2^a)) = multiset.sort (nat.digits 10 (2^b))) :=
sorry

end no_reordered_power_of_2_l779_779499


namespace sum_S100_l779_779905

noncomputable def a : ℕ → ℤ
| 0     := 0
| (n+1) := (1 + (-1)^n) * a n + (-2)^n

def S (n : ℕ) : ℤ := ∑ i in finset.range n, a i

theorem sum_S100 : S 100 = (2 - 2^101) / 3 :=
by
  sorry

end sum_S100_l779_779905


namespace distance_from_M_to_x_axis_is_3_l779_779558

-- Define point M and the line equation
def M : ℝ × ℝ := (-2, k)
def line_eq (x : ℝ) : ℝ := 2 * x + 1

-- Condition that point M is on the line y = 2x + 1
axiom point_on_line : (M.2 = line_eq M.1)

-- Prove that the distance d from M to the x-axis is 3
theorem distance_from_M_to_x_axis_is_3 (k : ℝ) (h : k = -3) : |M.2| = 3 := by
  sorry

end distance_from_M_to_x_axis_is_3_l779_779558


namespace largest_integer_among_four_l779_779041

theorem largest_integer_among_four 
  (p q r s : ℤ)
  (h1 : p + q + r = 210)
  (h2 : p + q + s = 230)
  (h3 : p + r + s = 250)
  (h4 : q + r + s = 270) :
  max (max p q) (max r s) = 110 :=
by
  sorry

end largest_integer_among_four_l779_779041


namespace average_side_length_and_perimeter_l779_779216

theorem average_side_length_and_perimeter (A1 A2 A3 : ℝ)
  (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 121) :
  (sqrt A1 + sqrt A2 + sqrt A3) / 3 = 8 ∧ (4 * sqrt A1 + 4 * sqrt A2 + 4 * sqrt A3) / 3 = 32 := by
  sorry

end average_side_length_and_perimeter_l779_779216


namespace solve_for_x_l779_779934

theorem solve_for_x (x : ℝ) (h : (3 / 4) - (1 / 2) = 1 / x) : x = 4 :=
sorry

end solve_for_x_l779_779934


namespace num_unique_sums_l779_779451

theorem num_unique_sums : 
  let s := {2, 5, 8, 11, 14, 17, 20, 23}
  let sums := {x | ∃ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a + b + c}
  set.count sums = 16 :=
sorry

end num_unique_sums_l779_779451


namespace profit_margin_increase_l779_779230

theorem profit_margin_increase
  (P S : ℝ)           -- purchase price is P, selling price is S
  (r : ℝ)             -- original profit margin
  (decreased_price : 0.92 * P) -- purchase price decreased by 8%
  (new_margin : (r + 10)) :    -- profit margin increased by 10%

  ( ((S - P) / P) * 100 = r ) →
  ( ((S - (0.92 * P)) / (0.92 * P)) * 100 = r + 10 ) →
  r = 15 :=

sorry

end profit_margin_increase_l779_779230


namespace pinwheel_area_l779_779631

-- Define the pinwheel area problem in Lean 4
theorem pinwheel_area (i : ℕ) (b : ℕ) : 
  let scaled_factor := 4
  let num_kites := 4
  Pick_area_formula : ℕ := i + b / 2 - 1
  let original_area := (num_kites * Pick_area_formula) / scaled_factor
  (scaled_factor = 4) →
  (num_kites = 4) →
  (i = 0) →
  (b = 5) →
  (original_area = 6) := 
begin
  intros,
  -- Annotate the given conditions
  rw [Pick_area_formula, scaled_factor, num_kites, i, b],
  -- Pick's Theorem calculation
  have area_one_kite: Pick_area_formula = i + b / 2 - 1, {
    rw [Pick_area_formula],
    exact rfl,
  },
  calc
    original_area
      = (num_kites * Pick_area_formula) / scaled_factor : by rw [original_area]
  ... = (4 * (0 + 5 / 2 - 1)) / 4 : by rw [num_kites, i, b, area_one_kite]
  ... = 6 : by norm_num,
  exact rfl,
end

end pinwheel_area_l779_779631


namespace Jeff_Jogging_Extra_Friday_l779_779876

theorem Jeff_Jogging_Extra_Friday :
  let planned_daily_minutes := 60
  let days_in_week := 5
  let planned_weekly_minutes := days_in_week * planned_daily_minutes
  let thursday_cut_short := 20
  let actual_weekly_minutes := 290
  let thursday_run := planned_daily_minutes - thursday_cut_short
  let other_four_days_minutes := actual_weekly_minutes - thursday_run
  let mondays_to_wednesdays_run := 3 * planned_daily_minutes
  let friday_run := other_four_days_minutes - mondays_to_wednesdays_run
  let extra_run_on_friday := friday_run - planned_daily_minutes
  extra_run_on_friday = 10 := by trivial

end Jeff_Jogging_Extra_Friday_l779_779876


namespace hexagon_problem_l779_779306

theorem hexagon_problem (r x: ℝ) (A1 A2 A3 A4 A5 A6 : ℂ) (h_hexagon_regular: 
  (dist A1 A2 = dist A2 A3) ∧
  (dist A2 A3 = dist A3 A4) ∧
  (dist A3 A4 = dist A4 A5) ∧
  (dist A4 A5 = dist A5 A6) ∧
  (dist A5 A6 = dist A6 A1) ∧ 
  (abs (A1 - A2) = r * (complex.exp (2*π*complex.I/6))) ∧
  (abs (A2 - A3) = r * (complex.exp (4*π*complex.I/6))) ∧
  (abs (A3 - A4) = r * (complex.exp (6*π*complex.I/6))) ∧
  (abs (A4 - A5) = r * (complex.exp (8*π*complex.I/6))) ∧
  (abs (A5 - A6) = r * (complex.exp (10*π*complex.I/6))) ∧
  (abs (A6 - A1) = r * (complex.exp (12*π*complex.I/6)))
) :
(x = (r * real.sqrt 3)/3) ∧
(area_hexagon A1 A2 A3 A4 A5 A6 = (3 * real.sqrt 3 * r^2) / 2) ∧ 
(area_equilateral_triangle A1 A3 A5  = (3 * r^2 * real.sqrt 3) / 4) ∧ 
(area_new_hexagon r x = (r^2 * real.sqrt 3) / 6) ∧ 
(circumradius_ratio r (r/3) = 1/3)
:= sorry

end hexagon_problem_l779_779306


namespace problem_statement_l779_779405

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x else f (x + 1)

theorem problem_statement : f (-4/3) + f (4/3) = 4 := by
  sorry

end problem_statement_l779_779405


namespace frustum_has_only_two_parallel_surfaces_l779_779708

-- Definitions for the geometric bodies in terms of their properties
structure Pyramid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 0

structure Prism where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

structure Frustum where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 2

structure Cuboid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

-- The main theorem stating that the Frustum is the one with exactly two parallel surfaces.
theorem frustum_has_only_two_parallel_surfaces (pyramid : Pyramid) (prism : Prism) (frustum : Frustum) (cuboid : Cuboid) :
  frustum.parallel_surfaces = 2 ∧
  pyramid.parallel_surfaces ≠ 2 ∧
  prism.parallel_surfaces ≠ 2 ∧
  cuboid.parallel_surfaces ≠ 2 :=
by
  sorry

end frustum_has_only_two_parallel_surfaces_l779_779708


namespace value_of_x_l779_779988

theorem value_of_x (x : ℝ) (h : (10 - x)^2 = x^2 + 4) : x = 24 / 5 :=
by
  sorry

end value_of_x_l779_779988


namespace mean_of_data_is_5_l779_779806

theorem mean_of_data_is_5 (h : s^2 = (1 / 4) * ((3.2 - x)^2 + (5.7 - x)^2 + (4.3 - x)^2 + (6.8 - x)^2))
  : x = 5 := 
sorry

end mean_of_data_is_5_l779_779806


namespace roots_of_cubic_eq_l779_779830

theorem roots_of_cubic_eq (r s t p q : ℝ) (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) 
(h3 : r * s * t = r) : r^2 + s^2 + t^2 = p^2 - 2 * q := 
by 
  sorry

end roots_of_cubic_eq_l779_779830


namespace prime_roll_probability_l779_779941

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

-- Consider a list of numbers from 1 to 8, representing the faces of the die
def die_faces := [1, 2, 3, 4, 5, 6, 7, 8]

-- Filter the prime numbers from the die faces
def primes_on_die : List ℕ := die_faces.filter is_prime

-- Calculate the probability as a rational number
noncomputable def probability_prime_roll : ℚ :=
  primes_on_die.length / die_faces.length

-- Theorem stating the desired probability is 1/2
theorem prime_roll_probability : probability_prime_roll = 1 / 2 := by
  sorry

end prime_roll_probability_l779_779941


namespace tan_angle_sum_pyramids_l779_779982

theorem tan_angle_sum_pyramids
  (a R : ℝ)
  (α β : ℝ)
  (base_side_length a)
  (sphere_radius R)
  (base_angle_1 α)
  (base_angle_2 β)
  (pyramids_same_base : ∀ (P Q : TrianglePyramid), P.base = Q.base ∧ P.sphere = Q.sphere)
  (pyramids_inscribed_in_sphere : ∀ (P Q : TrianglePyramid), inscribed_in_sphere P Q R)
  (equal_base_side_length : ∀ (P : TrianglePyramid), P.base_side_length = a)
  (angles_between_faces_base : ∀ (P Q : TrianglePyramid), angle_between_face_base P Q α β) :
  tan (α + β) = - (4 * Real.sqrt 3 * R) / (3 * a) := sorry

end tan_angle_sum_pyramids_l779_779982


namespace range_of_function_l779_779765

theorem range_of_function :
  set.range (λ x : ℝ, (3 * x + 4) / (x - 5)) = set.Ioo (-∞) 3 ∪ set.Ioo 3 ∞ :=
sorry

end range_of_function_l779_779765


namespace maximize_sector_area_l779_779250

variable (r : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
def perimeter_condition := l + 2 * r = 20
def area_formula := S = 1 / 2 * l * r
def central_angle := l / r

-- Defining the problem statement:
theorem maximize_sector_area (h1 : perimeter_condition) (h2 : area_formula) :
  central_angle = 2 :=
sorry

end maximize_sector_area_l779_779250


namespace rectangular_to_polar_l779_779423

theorem rectangular_to_polar {x y : ℝ} (h1 : x = -2) (h2 : y = -2 * real.sqrt 3) :
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = 4 * real.pi / 3 ∧ (ρ * real.cos θ = x ∧ ρ * real.sin θ = y) :=
by
  use 4, (4 * real.pi / 3)
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end rectangular_to_polar_l779_779423


namespace min_points_for_lines_l779_779478

noncomputable theory

-- Define the sets and conditions given in the problem
def point_set := Set ℝ → Set ℝ

def line (P : point_set) (L : Set ℝ) := ∃ x y : ℝ, x ≠ y ∧ 
  Set.P = {p | p ∈ s ∧ p ∈ L}

-- Declare the main theorem stating the minimum number of points
theorem min_points_for_lines (L : Fin 6 (Set ℝ)) (h : ∀ i, ∃!(S⒮:point_set) S⒮=S.i) :
  ∃ (S : Finset ℝ) (hS : S.card ≤ 7), 
  ∀ (i : Fin 6), ∃! T : Finset ℝ, T ⊆ S ∧ T.card = 3 ∧ T⒮T=i :=
begin
  sorry
end

end min_points_for_lines_l779_779478


namespace sequence_general_formula_l779_779489

theorem sequence_general_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1 / 2) 
  (h_rec : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) / (a (n + 1) + 3)) 
  (n : ℕ) : 
  a (n + 1) = 3 / (n + 6) :=
by
  sorry

end sequence_general_formula_l779_779489


namespace train_speed_l779_779314
-- Definitions based on conditions
def train_length : ℕ := 100
def tunnel_length : ℕ := 1700
def time_to_pass_tunnel : ℝ := 1.5 / 60

-- The statement to be proven: The speed of the train in km/hr
theorem train_speed :
  let total_distance := (train_length + tunnel_length) / 1000 in 
  let speed := total_distance / time_to_pass_tunnel in 
  speed = 72 := 
by
  sorry

end train_speed_l779_779314


namespace false_statement_two_right_angled_triangles_congruent_l779_779991

theorem false_statement_two_right_angled_triangles_congruent :
  ¬ ∀ (T1 T2 : Triangle), (T1.angle = 90 ∧ T2.angle = 90) →
  (exists (a b c d : ℝ), T1.a = a ∧ T1.b = b ∧ T2.c = c ∧ T2.d = d ∧ ((a = c ∧ b = d) ∨ (a = d ∧ b = c))) →
  Congruent T1 T2 :=  
sorry

end false_statement_two_right_angled_triangles_congruent_l779_779991


namespace max_cardinality_of_S_l779_779310

noncomputable def S : Set ℕ := sorry

theorem max_cardinality_of_S : 
  (∀ x ∈ S, (S.erase x).sum % (S.erase x).card = 0) → 1 ∈ S → 1801 ∈ S → S.card ≤ 37 :=
by
  introv hMean h1 h1801
  sorry

end max_cardinality_of_S_l779_779310


namespace painted_cube_problem_l779_779686

theorem painted_cube_problem :
  let large_edge := 10
  let small_edge := 1
  (∃ (cubes_with_3_faces cubes_with_2_faces cubes_with_1_face cubes_with_no_faces : ℕ),
    cubes_with_3_faces = 8 ∧
    cubes_with_2_faces = 96 ∧
    cubes_with_1_face = 384 ∧
    cubes_with_no_faces = 512) :=
begin
  let large_edge := 10,
  let small_edge := 1,
  have cubes_with_3_faces := 8,
  have cubes_with_2_faces := 96,
  have cubes_with_1_face := 384,
  have cubes_with_no_faces := 512,
  use [cubes_with_3_faces, cubes_with_2_faces, cubes_with_1_face, cubes_with_no_faces],
  simp,
  sorry
end

end painted_cube_problem_l779_779686


namespace max_a_condition_l779_779835

theorem max_a_condition (x : ℝ) : (x^2 - 2 * x - 3 > 0) → (x < -1) :=
  begin
    -- We solve the inequality x^2 - 2x - 3 > 0, which factors to (x - 3)(x + 1) > 0
    -- The roots of the equation x^2 - 2x - 3 = 0 are x = 3 and x = -1
    -- The intervals where the inequality is satisfied are x > 3 or x < -1
    -- Because x^2 - 2x - 3 > 0 is a necessary but not sufficient condition for x < a, the maximum value of a must be -1
    sorry
  end

end max_a_condition_l779_779835


namespace length_ZD_angle_bisector_l779_779154

theorem length_ZD_angle_bisector (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 15) (hXZ : XZ = 17) (is_angle_bisector : True) : ∃ (ZD : ℝ), ZD = Real.sqrt 284.484375 :=
by {
  use Real.sqrt 284.484375,
  sorry
}

end length_ZD_angle_bisector_l779_779154


namespace smallest_positive_period_intervals_of_monotonic_decrease_area_of_triangle_abc_l779_779433

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * Real.cos x ^ 2 - Real.sqrt 3 

theorem smallest_positive_period (x : ℝ) : 
  y = f (-3 * x) + 1 → has_period y (π / 3) :=
sorry

theorem intervals_of_monotonic_decrease (x k : ℝ) (h : k ∈ ℤ) : 
  y = f (-3 * x) + 1 → ( (1 / 3) * k * π - π / 36 ≤ x ∧ x ≤ (1 / 3) * k * π + 5 * π / 36 ) :=
sorry

theorem area_of_triangle_abc (A a b c : ℝ) (A_acute : 0 < A ∧ A < π / 2)
  (ha : a = 7) (hbc : b + c = 13) (hbcsum : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14)
  (hAf : f (A / 2 - π / 6) = Real.sqrt 3) :
  area_of_triangle ABC = 10 * Real.sqrt 3 :=
sorry

end smallest_positive_period_intervals_of_monotonic_decrease_area_of_triangle_abc_l779_779433


namespace number_of_paths_l779_779662

-- Conditions
def start_point : (ℤ × ℤ) := (-5, -2)
def end_point : (ℤ × ℤ) := (3, 8)
def total_steps := 20
def step_increase := 1
def rect_left := 0
def rect_right := 5
def rect_bottom := 3
def rect_top := 5

-- Theorem statement
theorem number_of_paths : 
  ∃ n : ℕ, n = 15840 ∧ 
    (∀ path : list (ℤ × ℤ), 
      path.head = start_point ∧ 
      path.last = end_point ∧ 
      path.length = total_steps ∧
      (∀ point ∈ path, 
        (point.1 = path.head.1 + step_increase * (count_steps_x path)) ∧ 
        (point.2 = path.head.2 + step_increase * (count_steps_y path))) ∧ 
      (∀ point ∈ path, 
        ¬ (rect_left ≤ point.1 ∧ point.1 ≤ rect_right ∧ rect_bottom ≤ point.2 ∧ point.2 ≤ rect_top))
    → n = 15840) :=
sorry

-- Auxiliary functions
def count_steps_x (path: list (ℤ × ℤ)) : ℕ := path.countp (λ p, p.1 ≠ start_point.1)
def count_steps_y (path: list (ℤ × ℤ)) : ℕ := path.countp (λ p, p.2 ≠ start_point.2)

end number_of_paths_l779_779662


namespace total_money_paid_l779_779615

def total_employees := 450
def monthly_salary_per_employee := 2000
def fraction_remaining := (2 : ℝ) / 3

def remaining_employees := total_employees * fraction_remaining

theorem total_money_paid (h : remaining_employees = 300) :
  remaining_employees * monthly_salary_per_employee = 600000 := by
  -- Proof will be here
  sorry

end total_money_paid_l779_779615


namespace expression_value_l779_779268

-- Definition for greatest integer less than or equal to x
def greatest_int_le (x : ℝ) : ℤ := ⌊x⌋

-- Define the expression and the result
theorem expression_value :
    greatest_int_le 6.5 * greatest_int_le (2 / 3) + 
    greatest_int_le 2 * 7.2 + 
    greatest_int_le 8.4 - 6.6 = 15.8 := 
by
    sorry

end expression_value_l779_779268


namespace function_intersects_line_x2_once_l779_779815

-- Define the function and its domain
variable (f : ℝ → ℝ)
variable (a b : ℝ)
variable (h_domain : ∀ x, x ∈ set.Icc a b → f x ∈ set.Icc a b)

-- State the theorem
theorem function_intersects_line_x2_once : (a = -2) → (b = 3) → ∃! y, f 2 = y :=
by
  intros ha hb
  sorry

end function_intersects_line_x2_once_l779_779815


namespace puppies_per_cage_l779_779302

/-
Theorem: If a pet store had 56 puppies, sold 24 of them, and placed the remaining puppies into 8 cages, then each cage contains 4 puppies.
-/

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (cages : ℕ)
  (remaining_puppies : ℕ)
  (puppies_per_cage : ℕ) :
  initial_puppies = 56 →
  sold_puppies = 24 →
  cages = 8 →
  remaining_puppies = initial_puppies - sold_puppies →
  puppies_per_cage = remaining_puppies / cages →
  puppies_per_cage = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end puppies_per_cage_l779_779302


namespace convex_polygon_circumscribed_triangle_l779_779926

theorem convex_polygon_circumscribed_triangle (P : polygon) (h_convex : P.is_convex) (h_not_parallelogram : ¬ P.is_parallelogram) : 
  ∃ (s1 s2 s3 : side), 
  P.has_side s1 ∧ P.has_side s2 ∧ P.has_side s3 ∧ 
  are_non_adjacent s1 s2 ∧ are_non_adjacent s2 s3 ∧ are_non_adjacent s3 s1 ∧
  extended_intersect_outside P s1 s2 ∧ extended_intersect_outside P s2 s3 ∧ extended_intersect_outside P s3 s1 :=
begin
  sorry
end

end convex_polygon_circumscribed_triangle_l779_779926


namespace divide_weights_into_equal_sums_l779_779851

theorem divide_weights_into_equal_sums (weights : Fin 2009 → ℕ)
    (h1 : ∀ i : Fin 2009, weights i ≤ 1000)
    (h2 : ∀ i : Fin 2008, abs (weights i - weights (i+1)) = 1)
    (h3 : (Finset.univ : Finset (Fin 2009)).sum weights % 2 = 0) :
    ∃ (s1 s2 : Finset (Fin 2009)), s1 ∪ s2 = Finset.univ ∧ s1 ∩ s2 = ∅ ∧ 
    (s1.sum weights = s2.sum weights) :=
sorry

end divide_weights_into_equal_sums_l779_779851


namespace solve_polynomial_l779_779935

theorem solve_polynomial :
  ∃ x : ℂ, (x^6 - 64 = 0) ∧
  (x = 2 ∨ x = -2 ∨
   x = 1 + complex.I * complex.sqrt 3 ∨
   x = 1 - complex.I * complex.sqrt 3 ∨
   x = -1 + complex.I * complex.sqrt 3 ∨
   x = -1 - complex.I * complex.sqrt 3) :=
by sorry

end solve_polynomial_l779_779935


namespace max_and_min_values_l779_779744

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem max_and_min_values (a : ℝ) :
  let interval := set.Icc (0 : ℝ) (2 : ℝ)
  let f_val (x : ℝ) := f x a
  ∃ m M, (∀ x ∈ interval, f_val x ≤ M) ∧ (∀ x ∈ interval, m ≤ f_val x) ∧
         ((a < 0 ∧ M = 3 - 4 * a ∧ m = -1) ∨
          (0 ≤ a ∧ a < 1 ∧ M = 3 - 4 * a ∧ m = -1 - a^2) ∨
          (1 ≤ a ∧ a < 2 ∧ M = -1 ∧ m = -1 - a^2) ∨
          (a ≥ 2 ∧ M = -1 ∧ m = 3 - 4 * a)) :=
by
  sorry

end max_and_min_values_l779_779744


namespace min_period_sin_2x_l779_779599

def is_periodic (f : ℝ → ℝ) (T : ℝ) := ∀ x : ℝ, f (x + T) = f x

theorem min_period_sin_2x : ∃ T > 0, is_periodic (λ x, Real.sin (2 * x)) T ∧ (∀ T' > 0, is_periodic (λ x, Real.sin (2 * x)) T' → T ≤ T') ∧ T = Real.pi :=
by
  sorry

end min_period_sin_2x_l779_779599


namespace prove_inequality_l779_779090

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - x
noncomputable def h (a x : ℝ) : ℝ := a * f(x) + g(x)

variables (a : ℝ) (h_a : 0 < a ∧ a < 1)
let x1 := - Real.sqrt (1 - a)
let x2 := Real.sqrt (1 - a)

theorem prove_inequality : 2 * h a x2 - x1 > 0 :=
sorry

end prove_inequality_l779_779090


namespace triangle_area_l779_779494

-- Define the triangle ABC with given conditions
structure Triangle (α β γ : Type) :=
(a b c : α)
(angle_bisector_length : β)
(area : γ)

-- Define the conditions as contained in the problem
def given_triangle : Triangle ℝ ℝ ℝ :=
{ a := 30,
  b := 70,
  c := _,
  angle_bisector_length := 21,
  area := 525 * real.sqrt 3 }

-- State the theorem to be proved
theorem triangle_area (T : Triangle ℝ ℝ ℝ) (h : T = given_triangle) :
  T.area = 525 * real.sqrt 3 :=
by
  intro,
  exact h ▸ rfl

end triangle_area_l779_779494


namespace highest_score_l779_779586

theorem highest_score (H L : ℕ) (avg total46 total44 runs46 runs44 : ℕ)
  (h1 : H - L = 150)
  (h2 : avg = 61)
  (h3 : total46 = 46)
  (h4 : runs46 = avg * total46)
  (h5 : runs46 = 2806)
  (h6 : total44 = 44)
  (h7 : runs44 = 58 * total44)
  (h8 : runs44 = 2552)
  (h9 : runs46 - runs44 = H + L) :
  H = 202 := by
  sorry

end highest_score_l779_779586


namespace coefficient_of_x2_term_l779_779126

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x2_term :
  let n := 12
  let expr := (x - x⁻¹)^n
  (∃ r : ℕ, r = 9 ∧ (12 - (4*r) / 3 = 2)) →
  (binomialCoeff 12 9) = 220 :=
by
  intros
  sorry

end coefficient_of_x2_term_l779_779126


namespace length_ZD_in_triangle_XYZ_l779_779156

theorem length_ZD_in_triangle_XYZ :
  ∀ (X Y Z D : Type*)
  [HasDistance X Y (8 : Real)]
  [HasDistance Y Z (15 : Real)]
  [HasDistance X Z (17 : Real)]
  [AngleBisector Z D XY YZ],
  ∃ (ZD : Real), ZD = Real.sqrt 132897 / 23 :=
by
  sorry

end length_ZD_in_triangle_XYZ_l779_779156


namespace third_candidate_votes_l779_779973

theorem third_candidate_votes
  (total_votes : ℝ)
  (votes_for_two_candidates : ℝ)
  (winning_percentage : ℝ)
  (H1 : votes_for_two_candidates = 4636 + 11628)
  (H2 : winning_percentage = 67.21387283236994 / 100)
  (H3 : total_votes = votes_for_two_candidates / (1 - winning_percentage)) :
  (total_votes - votes_for_two_candidates) = 33336 :=
by
  sorry

end third_candidate_votes_l779_779973


namespace country_x_tax_l779_779471

theorem country_x_tax (X I T : ℝ) 
  (h₁ : I = 50000)
  (h₂ : T = 8000)
  (h₃ : T = 0.15 * X + 0.20 * (I - X)) :
  X = 40000 :=
by
  subst h₁
  subst h₂
  rw [h₃]
  linarith

end country_x_tax_l779_779471


namespace men_build_wall_l779_779501

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end men_build_wall_l779_779501


namespace equal_area_condition_l779_779144

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2)

theorem equal_area_condition : 2 * θ = (Real.tan θ) * (Real.tan (2 * θ)) :=
by {
  sorry
}

end equal_area_condition_l779_779144


namespace find_a_max_value_l779_779438

theorem find_a_max_value :
  ∃ a : ℝ, 0 < a ∧ ∀ x ∈ Icc (0 : ℝ) (π / 2), (a * cos (2 * x + π / 3) + 3 ≤ 4) ∧ (∃ x ∈ Icc (0 : ℝ) (π / 2), a * cos (2 * x + π / 3) + 3 = 4) :=
sorry

end find_a_max_value_l779_779438


namespace PF_eq_PG_l779_779177

noncomputable theory
open_locale classical

variables {A B C P D E F G : Type}

-- Definitions of points, segments and angles assuming they are classical points in geometry.
variables [euclidean_geometry A B C] [point P] [point D] [point E] [point F] [point G]
variables [segment A B] [segment A C]
variables [90_degree_angle A B P] [90_degree_angle A C P]
variables [equal_length_segment_segment D B P] [equal_length_segment_segment C P E]
variables [perpendicular_segment_segment D F A B] [perpendicular_segment_segment E G A C]

theorem PF_eq_PG
  (h1 : inside_angle P ∠ BAC)
  (h2 : angle_90_deg ∠ ABP)
  (h3 : angle_90_deg ∠ ACP)
  (h4 : point_on_segment D B A)
  (h5 : point_on_segment E C A)
  (h6 : equal_length B P B D)
  (h7 : equal_length C P C E)
  (h8 : point_on_segment F A C)
  (h9 : point_on_segment G A B)
  (h10 : perpendicular D F A B)
  (h11 : perpendicular E G A C) :
  length P F = length P G :=
sorry

end PF_eq_PG_l779_779177


namespace travel_with_decreasing_ticket_prices_l779_779854

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ∀ (i j : ℕ), i ≠ j → ℕ) : 
  cities.card = n ∧
  (∀ i j, i ≠ j → train_prices i j = train_prices j i) ∧
  (∀ i j k l, (i ≠ j ∧ k ≠ l ∧ (i ≠ k ∨ j ≠ l)) → train_prices i j ≠ train_prices k l) →
  ∃ (start : ℕ), ∃ (route : list (ℕ × ℕ)), 
  route.length = n - 1 ∧ 
  (∀ (m : ℕ), m < route.length - 1 → train_prices route.nth m route.nth (m+1) > train_prices route.nth (m+1) route.nth (m+2)) :=
by 
  sorry

end travel_with_decreasing_ticket_prices_l779_779854


namespace min_sum_exponents_powers_of_2_eq_22_l779_779123

theorem min_sum_exponents_powers_of_2_eq_22 :
  ∃ (s : Finset ℕ), (∑ x in s, 2^x = 800) ∧ 2 ≤ s.card ∧ (s.sum id = 22) :=
by
  sorry

end min_sum_exponents_powers_of_2_eq_22_l779_779123


namespace car_second_hour_speed_l779_779235

theorem car_second_hour_speed (s1 s2 : ℕ) (h1 : s1 = 100) (avg : (s1 + s2) / 2 = 80) : s2 = 60 :=
by
  sorry

end car_second_hour_speed_l779_779235


namespace apollonian_circle_correct_l779_779444

noncomputable def apollonian_circle 
  (A B : ℝ × ℝ) (k : ℝ) (k_ne_one : k ≠ 1) : set (ℝ × ℝ) :=
  let a := (B.1 - A.1) / 2 in
  let C : ℝ × ℝ := ( (- (1 + k^2) / (1 - k^2)) * a, 0 ) in
  let r := abs (2 * k * a / (1 - k^2)) in
  {M | (M.1 - C.1)^2 + (M.2 - C.2)^2 = r^2}

theorem apollonian_circle_correct 
  (A B : ℝ × ℝ) (k : ℝ) (M : ℝ × ℝ) (h : (A, B).1 = (- (B, A)).1) (k_ne_one : k ≠ 1) : 
  (dist M A / dist M B = k) ↔ M ∈ (apollonian_circle A B k k_ne_one) :=
sorry

end apollonian_circle_correct_l779_779444


namespace power_function_passes_through_point_l779_779466

theorem power_function_passes_through_point {a : ℝ} (h : 2^a = sqrt 2) : a = 1 / 2 :=
sorry

end power_function_passes_through_point_l779_779466


namespace no_such_decomposition_l779_779498

theorem no_such_decomposition : 
  ¬(∃ (S : Finset (Finset ℕ)), 
    S.card = 11 ∧ 
    (∀ s ∈ S, ∃ a b c, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c = a + b ∧ s = {a, b, c}) ∧ 
    (⋃₀ S) = Finset.range 34 ∧
    (∀ s s' ∈ S, s ≠ s' → s ∩ s' = ∅)) := 
sorry

end no_such_decomposition_l779_779498


namespace value_range_of_a_l779_779922

theorem value_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ↔ (-2 < a ∧ a < 2) →
  (∀ x : ℝ, (3 - 2*a) ^ x > 0) ↔ (a < 1) →
  (a ≤ -2 ∨ 1 ≤ a ∧ a < 2) :=
sorry

end value_range_of_a_l779_779922


namespace baseball_team_groups_l779_779232

theorem baseball_team_groups
  (new_players : ℕ) 
  (returning_players : ℕ)
  (players_per_group : ℕ)
  (total_players : ℕ := new_players + returning_players) :
  new_players = 48 → 
  returning_players = 6 → 
  players_per_group = 6 → 
  total_players / players_per_group = 9 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end baseball_team_groups_l779_779232


namespace line_of_intersecting_circles_l779_779627

theorem line_of_intersecting_circles
  (A B : ℝ × ℝ)
  (hAB1 : A.1^2 + A.2^2 + 4 * A.1 - 4 * A.2 = 0)
  (hAB2 : B.1^2 + B.2^2 + 4 * B.1 - 4 * B.2 = 0)
  (hAB3 : A.1^2 + A.2^2 + 2 * A.1 - 12 = 0)
  (hAB4 : B.1^2 + B.2^2 + 2 * B.1 - 12 = 0) :
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0 ∧
                  a = 1 ∧ b = -2 ∧ c = 6 :=
sorry

end line_of_intersecting_circles_l779_779627


namespace ratio_of_segements_square_of_others_l779_779710

theorem ratio_of_segements_square_of_others
  {A B C D E : Type}
  [a : B ∈ line_segment (C, extension_of (B))]
  (h1 : is_right_triangle A B C)
  (h2 : altitude A D (hypotenuse B C))
  (h3 : angle E A B = angle B A D)
  : BD / CD = (AE^2) / (EC^2) := by
  sorry

end ratio_of_segements_square_of_others_l779_779710


namespace emily_tv_episodes_watched_l779_779019

-- Definitions based on the conditions
def flight_duration_minutes : ℕ := 10 * 60
def episode_duration_minutes : ℕ := 25
def sleep_duration_minutes : ℕ := 4.5 * 60
def movie_duration_minutes : ℕ := 1 * 60 + 45
def num_movies : ℕ := 2
def remaining_flight_time : ℕ := 45

-- The theorem to prove
theorem emily_tv_episodes_watched : 
  ((flight_duration_minutes - (sleep_duration_minutes + (num_movies * movie_duration_minutes) + remaining_flight_time)) / episode_duration_minutes) = 3 :=
  sorry

end emily_tv_episodes_watched_l779_779019


namespace binomial_variance_red_ball_variance_correct_l779_779578

-- Assuming we have a binomial distribution X ~ B(3, 2/3), we need to prove that its variance is 2/3.
theorem binomial_variance (n : ℕ) (p : ℚ) (X : ℕ → ℚ) (hX : X ~ binomial n p) : 
  X.variance = n * p * (1 - p) := by
  sorry

-- Given our specific problem
noncomputable def red_ball_experiment : Prop :=
  let n := 3
  let p := 2 / 3
  let X := binomial n p
  X.variance = 2 / 3

-- The theorem to prove it
theorem red_ball_variance_correct : red_ball_experiment := by
  sorry

end binomial_variance_red_ball_variance_correct_l779_779578


namespace range_of_a_l779_779791

open Real

theorem range_of_a (a : ℝ)
  (p : ∀ x ∈ Icc (0 : ℝ) 3, a ≥ -x^2 + 2*x - 2/3)
  (q : ∃ x : ℝ, x^2 + 4*x + a = 0) :
  1/3 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l779_779791


namespace perpendiculars_concurrent_l779_779906

theorem perpendiculars_concurrent 
  {A B C D E F : Type} [ConvexHexagon A B C D E F]
  (h_AB_BC : distance A B = distance B C)
  (h_CD_DE : distance C D = distance D E)
  (h_EF_FA : distance E F = distance F A) :
  concurrent (perpendicular_to (line_through F B) A)
             (perpendicular_to (line_through B D) C)
             (perpendicular_to (line_through D F) E)
  :=
sorry

end perpendiculars_concurrent_l779_779906


namespace product_of_bc_l779_779968

theorem product_of_bc
  (b c : Int)
  (h1 : ∀ r, r^2 - r - 1 = 0 → r^5 - b * r - c = 0) :
  b * c = 15 :=
by
  -- We start the proof assuming the conditions
  sorry

end product_of_bc_l779_779968


namespace center_after_reflection_l779_779587

theorem center_after_reflection (x₀ y₀ : ℝ) (h₁ : x₀ = 3) (h₂ : y₀ = -7) 
  : let (x', y') := (-y₀, -x₀) in x' = 7 ∧ y' = -3 :=
by
  sorry

end center_after_reflection_l779_779587


namespace max_tables_possible_l779_779623

/-- 
Definitions:
- wood_per_tabletop: 1 m^3 of wood yields 20 tabletops
- wood_per_table_leg: 1 m^3 of wood yields 400 table legs
- table_requirements: Each table needs 1 tabletop and 4 table legs
- total_available_wood: 12 m^3 of wood available

Statement:
Prove that given these conditions, the maximum number of tables that can be made is 200.
-/
def wood_per_tabletop : ℝ := 1 / 20
def wood_per_table_leg : ℝ := 1 / 400
def legs_per_table : ℝ := 4
def available_wood : ℝ := 12

theorem max_tables_possible : 
  ∀ (x : ℝ), (wood_per_tabletop * x + wood_per_table_leg * legs_per_table * x ≤ available_wood) → x ≤ 200 :=
sorry

end max_tables_possible_l779_779623


namespace tangent_line_eq_range_of_a_inequality_proof_l779_779431

-- Define the function f
def f (a x : ℝ) : ℝ := a / Real.exp x - x + 1

-- Part 1: Prove the equation of the tangent line to the curve y = f(x) at the point (0, f(0)) for a=1
theorem tangent_line_eq (x y : ℝ) (a : ℝ) (ha : a = 1) (hx : x = 0) (hy : y = f a x) :
  2 * x + y - 2 = 0 := 
sorry

-- Part 2: Prove the range of values of a such that f(x) < 0 for all x in (0, +∞)
theorem range_of_a (a : ℝ) (h : ∀ x > 0, f a x < 0) : a ≤ -1 := 
sorry

-- Part 3: Prove that for any x in (0, +∞), (2 / exp x) - 2 < (1 / 2) * x^2 - x
theorem inequality_proof (x : ℝ) (hx : 0 < x) : 
  (2 / Real.exp x) - 2 < (1 / 2) * x^2 - x := 
sorry

end tangent_line_eq_range_of_a_inequality_proof_l779_779431
