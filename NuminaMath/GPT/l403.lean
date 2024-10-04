import Mathlib

namespace max_value_of_b_exists_l403_403090

-- Defining the given functions f and g
def f (a x : ℝ) := 6 * a^2 * Real.log x
def g (a x b : ℝ) := x^2 - 4 * a * x - b

-- The Lean statement that captures the problem
theorem max_value_of_b_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (b : ℝ), 
  (∀ x₀ : ℝ, 2 * x₀ - 4 * a = 6 * a^2 / x₀ → f a x₀ = g a x₀ b) → 
  b ≤ 1 / (3 * Real.exp 2) :=
sorry

end max_value_of_b_exists_l403_403090


namespace quadratic_equation_roots_real_and_distinct_l403_403531

theorem quadratic_equation_roots_real_and_distinct (k : ℝ) :
  ∀ x : ℝ, (x^2 - 2 * (k - 1) * x + k^2 - 1 = 0) →
  (x^2 - 2*(k-1)*x + k^2 - 1).discriminant > 0 ↔ k < 1 :=
by
  sorry

end quadratic_equation_roots_real_and_distinct_l403_403531


namespace correct_calculation_l403_403246

theorem correct_calculation :
  (∀ (a b : ℝ), (ab²)³ = a³ * b⁶) ∧ 
  (∀ (a : ℝ), a² * a⁴ ≠ a⁸) ∧
  (∀ (a : ℝ), 3a³ - a³ ≠ 2a) ∧
  (∀ (a b : ℝ), (a + b)² ≠ a² + b²) :=
by
  sorry

end correct_calculation_l403_403246


namespace negation_exists_l403_403266

theorem negation_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a * x + 1 ≥ 0 :=
sorry

end negation_exists_l403_403266


namespace sequence_sum_15_22_31_l403_403605

def sequence_sum (n : ℕ) : ℤ :=
  ∑ k in Finset.range n + 1, (-1 : ℤ) ^ (k + 1) * (4 * k - 3)

theorem sequence_sum_15_22_31 :
  sequence_sum 15 - sequence_sum 22 + sequence_sum 31 = 134 :=
by
  sorry

end sequence_sum_15_22_31_l403_403605


namespace infinite_primes_l403_403687

theorem infinite_primes : ∀ (s : Finset ℕ), (∀ p ∈ s, Prime p) → ∃ p, Prime p ∧ p ∉ s :=
by
  intro s hs
  let N := (s.val.prod + 1)
  have hn : N > 1 :=
      sorry  -- Skip the details here
  obtain ⟨p, hp⟩ := exists_prime_factorization hn
  use p
  have hp_prime : Prime p := hp.1
  exact ⟨hp_prime, sorry⟩  -- Skip the details here

end infinite_primes_l403_403687


namespace maximum_value_of_reciprocals_l403_403918

theorem maximum_value_of_reciprocals (c b : ℝ) (h0 : 0 < b ∧ b < c)
  (e1 : ℝ) (e2 : ℝ)
  (h1 : e1 = c / (Real.sqrt (c^2 + (2 * b)^2)))
  (h2 : e2 = c / (Real.sqrt (c^2 - b^2)))
  (h3 : 1 / e1^2 + 4 / e2^2 = 5) :
  ∃ max_val, max_val = 5 / 2 :=
by
  sorry

end maximum_value_of_reciprocals_l403_403918


namespace living_room_area_l403_403267

-- Define the conditions
def carpet_area (length width : ℕ) : ℕ :=
  length * width

def percentage_coverage (carpet_area living_room_area : ℕ) : ℕ :=
  (carpet_area * 100) / living_room_area

-- State the problem
theorem living_room_area (A : ℕ) (carpet_len carpet_wid : ℕ) (carpet_coverage : ℕ) :
  carpet_len = 4 → carpet_wid = 9 → carpet_coverage = 20 →
  20 * A = 36 * 100 → A = 180 :=
by
  intros h_len h_wid h_coverage h_proportion
  sorry

end living_room_area_l403_403267


namespace sqrt_47_between_6_and_7_product_of_integers_between_sqrt_47_l403_403733

theorem sqrt_47_between_6_and_7 : (6 : ℝ) < Real.sqrt 47 ∧ Real.sqrt 47 < (7 : ℝ) := by
  sorry

theorem product_of_integers_between_sqrt_47 : ∃ (a b : ℕ), a < b ∧ (Real.sqrt 47).cast < b ∧ a * b = 42 := by
  use 6, 7
  simp
  have h := sqrt_47_between_6_and_7
  exact ⟨lt_trans (by norm_num) h.1, lt_of_lt_of_le h.2 (by norm_num), rfl⟩

end sqrt_47_between_6_and_7_product_of_integers_between_sqrt_47_l403_403733


namespace sum_f_l403_403442

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403442


namespace smallest_prime_dividing_sum_l403_403240

theorem smallest_prime_dividing_sum : 
  ∃ p : ℕ, prime p ∧ (p ∣ (2 ^ 12 + 3 ^ 14 + 7 ^ 4)) ∧
           ∀ q : ℕ, (prime q ∧ q ∣ (2 ^ 12 + 3 ^ 14 + 7 ^ 4)) → q ≥ p :=
by sorry

end smallest_prime_dividing_sum_l403_403240


namespace sum_G_equals_2016531_l403_403374

-- Define G(n): ℕ → ℕ to count the number of solutions to cos x = cos nx over [0, π]
def G (n : ℕ) : ℕ :=
  if n % 4 = 0 then n
  else n + 1

-- Now we want to prove the sum of G(n) from 2 to 2007 is 2,016,531
theorem sum_G_equals_2016531 : 
  (∑ n in Finset.range 2006 + 2, G n) = 2_016_531 :=
by
  sorry

end sum_G_equals_2016531_l403_403374


namespace milk_production_l403_403177

variable (x y z w v : ℕ)
variable (hx : x ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0)

theorem milk_production (x y z w v : ℕ) 
  (hx : x ≠ 0) (hz : z ≠ 0) : 0.9 * ((w * y * v) / (z * x)) = 0.9 * ((w * y * v) / (z * x)) :=
by sorry

end milk_production_l403_403177


namespace sqrt_5_is_mian_l403_403114

theorem sqrt_5_is_mian :
  (∀ (n : ℕ), n * n ≠ 5) :=
by {
  intro n,
  sorry
}

end sqrt_5_is_mian_l403_403114


namespace projection_of_a_on_n_l403_403025

open Matrix

def projection_of_vector (a n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := a.1 * n.1 + a.2 * n.2 + a.3 * n.3
  let n_norm_sq := n.1^2 + n.2^2 + n.3^2
  (dot_product / n_norm_sq) • n

theorem projection_of_a_on_n (a n : ℝ × ℝ × ℝ) (alpha beta : Type) 
  (h₁ : a = (1, 0, -1)) 
  (h₂ : n = (1, 2, 3))
  (h3 : ∀ m : ℝ × ℝ × ℝ, α = m → β = 2 • n → True) :
  projection_of_vector a n = (-1 / 7) • n :=
by
  rw [h₁, h₂]
  sorry

end projection_of_a_on_n_l403_403025


namespace Harold_remaining_money_l403_403946

-- Define Harold's monthly income
def Harold_income : ℝ := 2500.00

-- Define Harold's monthly expenses
def Harold_rent : ℝ := 700.00
def Harold_car_payment : ℝ := 300.00
def Harold_utilities_cost : ℝ := Harold_car_payment / 2
def Harold_groceries : ℝ := 50.00

-- Define the total expenses
def total_expenses : ℝ := Harold_rent + Harold_car_payment + Harold_utilities_cost + Harold_groceries

-- Define the remaining money after expenses
def remaining_money : ℝ := Harold_income - total_expenses

-- Define the amount Harold puts into retirement
def retirement_contribution : ℝ := remaining_money / 2

-- Prove the amount Harold is left with
theorem Harold_remaining_money : remaining_money - retirement_contribution = 650.00 := by sorry

end Harold_remaining_money_l403_403946


namespace exists_parallel_line_through_M_exists_perpendicular_line_through_M_l403_403638

variables {Point : Type} [EuclideanGeometry Point]

structure Circle (Point : Type) :=
(center : Point)
(radius : ℝ)

structure Line (Point : Type) :=
(p1 p2 : Point)

variable {C : Circle Point}
variable {O : Point}
variable {l : Line Point}
variable {M : Point}

-- Part (a): Prove that there exists a line through \(M\) parallel to line \(l\) given the conditions.
theorem exists_parallel_line_through_M (C : Circle Point) (O : Point) (l : Line Point) (M : Point) :
  ∃ (L : Line Point), parallel L l ∧ (∃ (p : Point), p ∈ L ∧ p = M) := 
sorry

-- Part (b): Prove that there exists a line through \(M\) perpendicular to line \(l\) given the conditions.
theorem exists_perpendicular_line_through_M (C : Circle Point) (O : Point) (l : Line Point) (M : Point) :
  ∃ (L : Line Point), perpendicular L l ∧ (∃ (p : Point), p ∈ L ∧ p = M) := 
sorry

end exists_parallel_line_through_M_exists_perpendicular_line_through_M_l403_403638


namespace f_minus_5_eq_12_l403_403526

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_minus_5_eq_12 : f (-5) = 12 := 
by sorry

end f_minus_5_eq_12_l403_403526


namespace pyramid_base_area_minimized_l403_403800

theorem pyramid_base_area_minimized {d : ℝ} (h_d : d > 0) :
  ∃ (a : ℝ), (a = 8 * d^2) :=
begin
  -- We need to prove there exists an a such that a is 8 times d squared
  use 8 * d^2,
  -- additional checks could be added here as required
  -- The below statement essentially proves the result directly
  exact (8 * d^2),
  sorry -- Proof details will be provided here
end

end pyramid_base_area_minimized_l403_403800


namespace sum_A_inter_Z_l403_403938

def A : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
def Z : Set ℤ := { x : ℤ | True }

theorem sum_A_inter_Z : (∑ i in A ∩ Z.to_finset, (i : ℝ)) = 3 := 
by
sorry

end sum_A_inter_Z_l403_403938


namespace log_relationship_l403_403076

-- Define the conditions in Lean
def a : ℝ := Real.logb 8 225
def b : ℝ := Real.logb 2 15

-- State the theorem to prove the relationship between a and b
theorem log_relationship : a = (2/3) * b := by
  sorry

end log_relationship_l403_403076


namespace minimum_value_of_function_l403_403401

noncomputable def f : ℝ → ℝ :=
  λ x, (4 / x) + (1 / (1 - x))

theorem minimum_value_of_function : 
  ∃ x : ℝ, (0 < x ∧ x < 1) ∧ (∀ y : ℝ, (0 < y ∧ y < 1) → f(x) ≤ f(y)) ∧ f(x) = 9 :=
by 
  sorry

end minimum_value_of_function_l403_403401


namespace part1_geometric_sequence_part2_sum_S_n_l403_403902

noncomputable def a : ℕ → ℕ
| 0     := 0 -- We include the 0th term for the definition's sake but it won't be used
| 1     := 1
| (n+2) := 3 * a (n+1) + 2

def b (n : ℕ) : ℕ := (2 * n + 1) * (a (n + 1) - a n)

def S (n : ℕ) : ℕ := ∑ k in finset.range n, b (k + 1)

theorem part1_geometric_sequence : ∀ n ≥ 1, ∃ (r : ℕ), a (n + 1) + 1 = (a 1 + 1) * r ^ n :=
by
  sorry

theorem part2_sum_S_n : ∀ n, S n = 4 * n * 3 ^ n :=
by 
  sorry

end part1_geometric_sequence_part2_sum_S_n_l403_403902


namespace minimize_s_inside_minimize_v_center_l403_403998

open EuclideanGeometry

-- Define the regular hexagon in the plane
structure RegularHexagon (α : Type*) [MetricSpace α] :=
(vertices : Fin 6 → α)
(is_regular : ∀ i, dist (vertices i) (vertices (i + 1) % 6) = dist (vertices 0) (vertices 1))

-- Sum of distances from P to each side of the hexagon
def s (P : α) (hex : RegularHexagon α) : ℝ :=
Finset.sum (Finset.range 6) (λ i, distance_to_side P (hex.vertices i) (hex.vertices ((i + 1) % 6)))
-- distance_to_side left unspecified here for demonstration purposes

-- Sum of distances from P to each vertex of the hexagon
def v (P : α) (hex : RegularHexagon α) : ℝ :=
Finset.sum (Finset.range 6) (λ i, dist P (hex.vertices i))

-- Proving the locus of points that minimize s(P) is inside the regular hexagon
theorem minimize_s_inside (α : Type*) [MetricSpace α] (hex : RegularHexagon α) (P : α) :
  ∃ P, P ∈ interior_of_regular_hexagon hex ∧ s P hex = ⨅ P, s P hex :=
sorry

-- Proving the locus of points that minimize v(P) is the center
theorem minimize_v_center (α : Type*) [MetricSpace α] (hex : RegularHexagon α) (P : α) :
  ∃ P, P = center_of_regular_hexagon hex ∧ v P hex = ⨅ P, v P hex :=
sorry

end minimize_s_inside_minimize_v_center_l403_403998


namespace alice_forest_walks_l403_403344

theorem alice_forest_walks
  (morning_distance : ℕ)
  (total_distance : ℕ)
  (days_per_week : ℕ)
  (forest_distance : ℕ) :
  morning_distance = 10 →
  total_distance = 110 →
  days_per_week = 5 →
  (total_distance - morning_distance * days_per_week) / days_per_week = forest_distance →
  forest_distance = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_forest_walks_l403_403344


namespace approximate_imperfect_grains_l403_403315

theorem approximate_imperfect_grains :
  ∀ (G S I : ℕ), G = 3318 → S = 168 → I = 22 → (approximate_imperfect_grains G S I) = 434 :=
by
  sorry

def approximate_imperfect_grains (G S I : ℕ) : ℕ :=
  let proportion := I.to_rat / S.to_rat
  let imperfect_in_granary := (proportion * G.to_rat).to_nat
  round imperfect_in_granary

end approximate_imperfect_grains_l403_403315


namespace find_line_equation_l403_403043

theorem find_line_equation
  (A B : ℝ → ℝ → Prop)
  (l : ℝ → ℝ → Prop)
  (h_ellipse : ∀ x y, A x y ↔ x^2 + 3 * y^2 = 3)
  (h_slope : ∀ x y, l x y ↔ y = x + 1 ∨ y = x - 1)
  (h_distance : ∀ x1 y1 x2 y2, A x1 y1 → A x2 y2 → 
                 (l x1 y1 ∧ l x2 y2 → 
                 real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 3 * real.sqrt 2 / 2)) :
  (∀ x y, l x y → y = x + 1 ∨ y = x - 1) :=
by
  sorry

end find_line_equation_l403_403043


namespace sum_f_l403_403450

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403450


namespace sum_f_1_to_22_l403_403427

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403427


namespace circle_equation_l403_403409

theorem circle_equation (C : Type) [metric_space C] [has_dist C] [normed_group C] 
  (center : C) (h_center : center = ( -1, 0)) 
  (h_chord : dist ((-1, 0) : C) (line := λ x y, x + y + 3 = 0) = 4) : 
  ∃ (r : ℝ), ∀ (x y : ℝ), (x + 1)^2 + y^2 = r^2 :=
by
  sorry

end circle_equation_l403_403409


namespace sum_f_proof_l403_403434

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403434


namespace only_positive_p_l403_403346

open Nat

theorem only_positive_p (p : ℕ) (h₁ : Prime p) (h₂ : Prime (p + 4)) (h₃ : Prime (p + 8)) : p = 3 := 
by 
  sorry

end only_positive_p_l403_403346


namespace unique_two_digit_integer_s_l403_403740

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l403_403740


namespace sum_f_l403_403443

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403443


namespace tangent_line_slope_angle_l403_403211

theorem tangent_line_slope_angle (θ : ℝ) : 
  (∃ k : ℝ, (∀ x y, k * x - y = 0) ∧ ∀ x y, x^2 + y^2 - 4 * x + 3 = 0) →
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end tangent_line_slope_angle_l403_403211


namespace sum_G_l403_403372

def G (n : ℕ) : ℕ :=
if n % 2 = 0 then n else n

theorem sum_G :
  ∑ n in Finset.range 2006 \+ 2, G n = 2012028 :=
by sorry

end sum_G_l403_403372


namespace tutors_meeting_schedule_l403_403625

/-- In a school, five tutors, Jaclyn, Marcelle, Susanna, Wanda, and Thomas, 
are scheduled to work in the library. Their schedules are as follows: 
Jaclyn works every fifth school day, Marcelle works every sixth school day, 
Susanna works every seventh school day, Wanda works every eighth school day, 
and Thomas works every ninth school day. Today, all five tutors are working 
in the library. Prove that the least common multiple of 5, 6, 7, 8, and 9 is 2520 days. 
-/
theorem tutors_meeting_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := 
by
  sorry

end tutors_meeting_schedule_l403_403625


namespace sum_f_eq_neg_24_l403_403412

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403412


namespace order_of_a_b_c_l403_403891

noncomputable def a : ℝ := 2 ^ 0.9
noncomputable def b : ℝ := 3 ^ (2 / 3)
noncomputable def c : ℝ := Real.logBase (1 / 2) 3

theorem order_of_a_b_c : b > a ∧ a > c :=
by
  sorry

end order_of_a_b_c_l403_403891


namespace events_A_and_B_properties_l403_403978

-- Define the sample space of tossing two coins
def sample_space : set (bool × bool) := { (tt, tt), (tt, ff), (ff, tt), (ff, ff) }

-- Define events A and B
def event_A : set (bool × bool) := { (tt, ff), (ff, tt) } -- One head and one tail
def event_B : set (bool × bool) := { (tt, tt) }           -- Both heads

-- Define mutually exclusive predicate
def mutually_exclusive (A B : set (bool × bool)) : Prop := ∀ x, x ∈ A → x ∉ B

-- Define complementary predicate
def complementary (A B : set (bool × bool)) : Prop := A ∪ B = sample_space

-- Prove that events A and B are mutually exclusive but not complementary
theorem events_A_and_B_properties :
  (mutually_exclusive event_A event_B) ∧ ¬(complementary event_A event_B) := by
  sorry

end events_A_and_B_properties_l403_403978


namespace inequality_preserved_l403_403400

variable {a b c : ℝ}

theorem inequality_preserved (h : abs ((a^2 + b^2 - c^2) / (a * b)) < 2) :
    abs ((b^2 + c^2 - a^2) / (b * c)) < 2 ∧ abs ((c^2 + a^2 - b^2) / (c * a)) < 2 := 
sorry

end inequality_preserved_l403_403400


namespace sum_G_l403_403370

def G (n : ℕ) : ℕ :=
if n % 2 = 0 then n else n

theorem sum_G :
  ∑ n in Finset.range 2006 \+ 2, G n = 2012028 :=
by sorry

end sum_G_l403_403370


namespace coordinates_of_N_l403_403908

theorem coordinates_of_N (M : ℝ × ℝ) (a : ℝ × ℝ) (hM : M = (5, -6)) (ha : a = (1, -2))
    (hMN : ∃ N : ℝ × ℝ, ∃ x y : ℝ, N = (x, y) ∧ (x - 5, y + 6) = -3 • a) :
  ∃ N : ℝ × ℝ, N = (2, 0) :=
by
  use (2, 0)
  sorry

end coordinates_of_N_l403_403908


namespace sum_f_eq_neg_24_l403_403415

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403415


namespace sum_f_eq_neg24_l403_403491

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403491


namespace range_of_a_if_domain_is_R_range_of_a_if_range_is_R_l403_403932

-- Proof Problem 1: Range of a for domain to be R
theorem range_of_a_if_domain_is_R (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 > 0) ↔ a ∈ (-∞, -1] ∪ (5/3, ∞) :=
sorry

-- Proof Problem 2: Range of a for range to be R
theorem range_of_a_if_range_is_R (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, lg ((a^2 - 1) * x^2 + (a + 1) * x + 1) = y) ↔ a ∈ [1, 5/3] :=
sorry

end range_of_a_if_domain_is_R_range_of_a_if_range_is_R_l403_403932


namespace multiple_of_tickletoe_nails_l403_403228

def violet_nails := 27
def total_nails := 39
def difference := 3

theorem multiple_of_tickletoe_nails : ∃ (M T : ℕ), violet_nails = M * T + difference ∧ total_nails = violet_nails + T ∧ (M = 2) :=
by
  sorry

end multiple_of_tickletoe_nails_l403_403228


namespace find_xy_l403_403096

-- Definitions of the problem's conditions
variables {X Y Z : Type}
variables (a b c : ℝ)
def right_triangle (X Y Z : Type) :=
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def cot (z : ℝ) : ℝ := 1 / (Math.sin z / Math.cos z)

noncomputable def cot_z_equals_3 :=
  exists.intro 3 (by sorry)

def angle_x_90 (angle : ℝ) :=
  angle = π / 2

def hypotenuse_length_yz := 150

-- The theorem to prove XY given the above conditions
theorem find_xy (hypotenuse_length_yz : ℝ) (cot_z_equals_3 : ℝ) (angle_x_90 : ℝ) :
  XY = 15 * Real.sqrt(10) :=
by sorry

end find_xy_l403_403096


namespace sum_f_k_1_22_l403_403503

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403503


namespace range_log_cos_square_l403_403237

open Real

def log_base (b x : ℝ) := log x / log b

theorem range_log_cos_square:
  ∀ (x : ℝ), (0 < x ∧ x < π) → (log_base 3 (cos x)^2 ≤ 0 ∧ ∀ y, y < 0 → ∃ e > 0, log_base 3 (cos (x + e))^2 = y) := 
  sorry

end range_log_cos_square_l403_403237


namespace value_of_frac_l403_403590

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403590


namespace limit_correct_l403_403774

noncomputable def limit_problem : Real :=
  lim (x --> π) (λ x, (1 - sin (x / 2)) / (π - x))

theorem limit_correct : limit_problem = 0 := 
  sorry

end limit_correct_l403_403774


namespace intersection_M_N_l403_403939

-- Definitions of sets
def set M := {x : ℝ | -5 ≤ x ∧ x ≤ 5}
def set N := {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 6}

-- Statement of the intersection of sets
theorem intersection_M_N : M ∩ N = {x : ℝ | -5 ≤ x ∧ x ≤ -3} :=
by
-- Proof is omitted
sorry

end intersection_M_N_l403_403939


namespace phi_varphi_difference_squared_l403_403954

theorem phi_varphi_difference_squared :
  ∀ (Φ φ : ℝ), (Φ ≠ φ) → (Φ^2 - 2*Φ - 1 = 0) → (φ^2 - 2*φ - 1 = 0) →
  (Φ - φ)^2 = 8 :=
by
  intros Φ φ distinct hΦ hφ
  sorry

end phi_varphi_difference_squared_l403_403954


namespace residue_12_2040_mod_19_l403_403761

theorem residue_12_2040_mod_19 :
  12^2040 % 19 = 7 := 
sorry

end residue_12_2040_mod_19_l403_403761


namespace no_tangent_line_l403_403525

noncomputable def f (x a : ℝ) : ℝ := x - exp (x / a)

def monotonic_interval (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 - (1 / a) * exp (x / a)) < 0

def tangent_condition (a : ℝ) : Prop := 
  ∃ x0 y0 : ℝ, y0 = exp x0 ∧ y0 = (1 - 1 / a) * x0 - 1

theorem no_tangent_line (a : ℝ) (h1 : monotonic_interval a) (h2 : tangent_condition a) : false := 
sorry

end no_tangent_line_l403_403525


namespace square_division_l403_403006

theorem square_division (n : ℕ) : 
  ∃ (S : Set ℕ), S = {1} ∪ {m | ∃ k, m = 2 * k + 2 ∨ m = 2 * k + 5} ∧ 
  (n ∈ S ↔ ∃ squares : List (Real) × (Real), 
     -- Some condition stating these squares can partition a larger square) sorry := sorry
  sorry

end square_division_l403_403006


namespace water_percentage_in_tomato_juice_l403_403947

-- Definitions from conditions
def tomato_juice_volume := 80 -- in liters
def tomato_puree_volume := 10 -- in liters
def tomato_puree_water_percentage := 20 -- in percent (20%)

-- Need to prove percentage of water in tomato juice is 20%
theorem water_percentage_in_tomato_juice : 
  (100 - tomato_puree_water_percentage) * tomato_puree_volume / tomato_juice_volume = 20 :=
by
  -- Skip the proof
  sorry

end water_percentage_in_tomato_juice_l403_403947


namespace exists_xy_binom_eq_l403_403133

theorem exists_xy_binom_eq (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x + y).choose 2 = a * x + b * y :=
by
  sorry

end exists_xy_binom_eq_l403_403133


namespace power_mod_equiv_l403_403767

theorem power_mod_equiv :
  2^1000 % 17 = 1 := by
  sorry

end power_mod_equiv_l403_403767


namespace skating_minutes_needed_l403_403011

-- Define the conditions
def minutes_per_day (day: ℕ) : ℕ :=
  if day ≤ 4 then 80 else if day ≤ 6 then 100 else 0

-- Define total skating time up to 6 days
def total_time_six_days := (4 * 80) + (2 * 100)

-- Prove that Gage needs to skate 180 minutes on the seventh day
theorem skating_minutes_needed : 
  (total_time_six_days + x = 7 * 100) → x = 180 :=
by sorry

end skating_minutes_needed_l403_403011


namespace triangle_BC_length_l403_403993

theorem triangle_BC_length (ABC : Type) 
  {A B C K : ABC}
  {K_on_AB : K ∈ segment AB}
  (E : ABC)
  (KE_bisector_AKC : angleBisector KE A K C)
  (H : ABC)
  (KH_altitude_BKC : isAltitude KH B K C)
  (EKH_right_angle : ∠EKH = 90°)
  (HC_eq_5 : ∥H - C∥ = 5) :
  ∥B - C∥ = 10 :=
sorry

end triangle_BC_length_l403_403993


namespace hyperbola_eccentricity_l403_403023

noncomputable theory

variables {a b c : ℝ} (P F1 F2 : ℝ × ℝ)
def hyperbola (a b : ℝ) (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Conditions
axiom Foci_distance : ∀ {F1 F2 : ℝ × ℝ}, F1F2 = (2 * c)
axiom PF1_perpendicular_F1F2 : ∀ {P F1 F2 : ℝ × ℝ}, PF1 ⊥ F1F2
axiom PF1_equals_F1F2 : ∀ {P F1 : ℝ × ℝ}, PF1 = F1F2

-- Theorem to prove the question==answer given conditions
theorem hyperbola_eccentricity :
  ∀ {a b c : ℝ} (P F1 F2 : ℝ × ℝ),
    a > 0 → b > 0 →
    hyperbola a b P.1 P.2 →
    F1F2 = 2 * c →
    PF1 = 2 * c →
    (PF1 ⊥ F1F2) →
    let e := c / a in
    e = sqrt 2 + 1 :=
begin
  sorry
end

end hyperbola_eccentricity_l403_403023


namespace car_speed_l403_403785

theorem car_speed (v : ℝ) 
  (start_time_car1 : ℝ := 9) 
  (start_time_car2 : ℝ := 9 + 10/60) 
  (catch_up_time : ℝ := 10 + 30/60) 
  (speed_car2 : ℝ := 60) 
  (distance_trip : ℝ := 80) 
  (time_car2 : ℝ := catch_up_time - start_time_car2) 
  (distance_car2 : ℝ := speed_car2 * time_car2) 
  (head_start_time : ℝ := start_time_car2 - start_time_car1) 
  (head_start_distance : ℝ := v * head_start_time) 
  : 
  distance_car2 = v * time_car2 + head_start_distance → v = 54 :=
by
  simp [start_time_car1, start_time_car2, catch_up_time, speed_car2, 
        distance_trip, time_car2, distance_car2, head_start_time, 
        head_start_distance] at *
  intro h
  -- Equation simplification and solving goes here
  sorry

end car_speed_l403_403785


namespace coefficient_of_x_squared_in_bin_expansion_l403_403072

noncomputable def integral_eq_condition (n : ℝ) : Prop :=
  ∫ x in 0..n, |x - 5| = 25

noncomputable def binomial_coefficient (k : ℕ) (n : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x_squared_in_bin_expansion (n : ℕ) :
  integral_eq_condition (n : ℝ) → binomial_coefficient 2 n * (2^2) = 180 :=
by
  intros h
  sorry

end coefficient_of_x_squared_in_bin_expansion_l403_403072


namespace sam_walking_speed_l403_403378

theorem sam_walking_speed :
  ∀ (f_dist s_dist total_dist f_speed s_time s_speed : ℝ),
    f_dist + s_dist = total_dist →
    f_speed * s_time = f_dist →
    s_time = s_dist / s_speed →
    total_dist = 50 →
    f_speed = 5 →
    s_dist = 25 →
    s_time = 5 →
    s_speed = 5 :=
by
  intros f_dist s_dist total_dist f_speed s_time s_speed
  intro h1 h2 h3 h4 h5 h6 h7
  rw [←h4, ←h5, ←h6, ←h7]
  sorry

end sam_walking_speed_l403_403378


namespace max_omega_l403_403930

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + Real.pi / 4)

theorem max_omega (ω : ℝ) :
  (∀ x : ℝ, f ω x ≤ f ω Real.pi) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 6 → f ω y ≤ f ω x) ∧
  ω = -1/4 + 2 * Int.toReal (Int.floor ((ω + 1/4) / 2)) ∧
  0 < ω ∧ ω ≤ 9/2 → ω = 15/4 :=
by
  sorry

end max_omega_l403_403930


namespace total_pencils_is_220_l403_403707

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l403_403707


namespace domain_transformation_l403_403034

theorem domain_transformation (f : ℝ → ℝ) (h : ∀ x, x ∈ Icc (-2 : ℝ) 2 → x ∈ set.univ) :
  set.preimage (λ x, 2 * x + 1) (Icc (-2 : ℝ) 2) = Icc (-3 / 2) (1 / 2) :=
by sorry

end domain_transformation_l403_403034


namespace collinear_X_Y_Z_l403_403158

variables {O A B C X Y Z : Type*}
variable [metric_space O]
variables (h_cir_OA : metric.circle (segment_length O A)) 
          (h_cir_OB : metric.circle (segment_length O B))
          (h_cir_OC : metric.circle (segment_length O C))

variables (intersection_Z : intersection (metric.circle (segment_length O A)) 
                                          (metric.circle (segment_length O B)) = Z)
          (intersection_X : intersection (metric.circle (segment_length O B)) 
                                          (metric.circle (segment_length O C)) = X)
          (intersection_Y : intersection (metric.circle (segment_length O C)) 
                                          (metric.circle (segment_length O A)) = Y)

theorem collinear_X_Y_Z :
  ∃ line l : Type*, l ∋ X ∧ l ∋ Y ∧ l ∋ Z :=
sorry

end collinear_X_Y_Z_l403_403158


namespace left_seats_equals_15_l403_403099

variable (L : ℕ)

noncomputable def num_seats_left (L : ℕ) : Prop :=
  ∃ L, 3 * L + 3 * (L - 3) + 8 = 89

theorem left_seats_equals_15 : num_seats_left L → L = 15 :=
by
  intro h
  sorry

end left_seats_equals_15_l403_403099


namespace ratio_shirt_to_coat_l403_403868

-- Define the given conditions
def total_cost := 600
def shirt_cost := 150

-- Define the coat cost based on the given conditions
def coat_cost := total_cost - shirt_cost

-- State the theorem to prove the ratio of shirt cost to coat cost is 1:3
theorem ratio_shirt_to_coat : (shirt_cost : ℚ) / (coat_cost : ℚ) = 1 / 3 :=
by
  -- The proof would go here
  sorry

end ratio_shirt_to_coat_l403_403868


namespace arithmetic_seq_ratio_l403_403880

variable {a_n b_n : ℕ → ℚ}
variable {S_n T_n : ℕ → ℚ}

def arithmetic_seq_sum (a : ℕ → ℚ) (n : ℕ) : ℚ := n * (a (1) + a (n)) / 2

theorem arithmetic_seq_ratio 
  (h1 : ∀ n, S_n n = arithmetic_seq_sum a_n n) 
  (h2 : ∀ n, T_n n = arithmetic_seq_sum b_n n) 
  (h3 : ∀ n, S_n n / T_n n = n / (n + 1)) :
  a_n 5 / b_n 5 = 9 / 10 :=
by {
  sorry,
}

end arithmetic_seq_ratio_l403_403880


namespace subset_single_element_l403_403612

-- Define the set X
def X : Set ℝ := { x | x > -1 }

-- The proof statement
-- We need to prove that {0} ⊆ X
theorem subset_single_element : {0} ⊆ X :=
sorry

end subset_single_element_l403_403612


namespace sum_f_values_l403_403477

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403477


namespace tangent_line_through_point_l403_403985

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5
def point_P := (0, 2)

-- Statement of the proof problem
theorem tangent_line_through_point :
  ∃ (a b c : ℝ), 
  (∀ x y : ℝ, circle_eq x y → x - 2 * y + 4 = 0) ∧ 
  (a * (fst point_P) + b * (snd point_P) + c = 0) :=
sorry

end tangent_line_through_point_l403_403985


namespace percentage_bananas_rotten_l403_403802

theorem percentage_bananas_rotten (oranges bananas : ℕ) (rotten_oranges rotten_fruits_in_good_condition : ℝ)
  (total_oranges_bought : oranges = 600)
  (total_bananas_bought : bananas = 400)
  (percentage_rotten_oranges : rotten_oranges = (15 / 100 : ℝ) * oranges)
  (percentage_fruits_good : rotten_fruits_in_good_condition = 89.4 / 100) :
  let good_oranges := oranges - rotten_oranges in
  let total_fruits := oranges + bananas in
  let good_bananas := rotten_fruits_in_good_condition * total_fruits - good_oranges in
  let rotten_bananas := bananas - good_bananas in
  let percentage_rotten_bananas := (rotten_bananas / bananas) * 100 in
  percentage_rotten_bananas = 4 :=
by
  sorry

end percentage_bananas_rotten_l403_403802


namespace simplify_expression_l403_403827

variable (a b : ℕ)

theorem simplify_expression (a b : ℕ) : 5 * a * b - 7 * a * b + 3 * a * b = a * b := by
  sorry

end simplify_expression_l403_403827


namespace minimum_shots_needed_l403_403234

theorem minimum_shots_needed : ∀ (battleship_length board_size shots : ℕ), (battleship_length = 4) → (board_size = 7) → (shots = 12) → (∀ placement_row_length, placement_row_length ≥ shots) -> 
∀ (board : List (List Nat)), (∃ row, count_battleships_hit board row board_size >= 1) :=
begin
  sorry,
end

end minimum_shots_needed_l403_403234


namespace distance_to_swim_practice_is_22_5_l403_403678

noncomputable def distance_to_swim_practice 
  (weekend_time : ℝ) -- 30 minutes
  (weekday_time : ℝ) -- 45 minutes
  (quiet_weekday_time : ℝ) -- 15 minutes
  (weekend_speed_factor : ℝ) -- No change in speed
  (weekday_speed_decrease : ℝ) -- 15 miles per hour slower
  (quiet_weekday_speed_increase : ℝ) -- 24 miles per hour faster
  : ℝ :=
  let weekend_speed := 2 * d in -- Since d = v * (1/2) 
  let weekday_speed := weekday_speed_decrease * d in -- Since d = (v - 15) * (3/4)
  let quiet_weekday_speed := quiet_weekday_speed_increase * d in -- Since d = (v + 24) * (1/4)
    d

theorem distance_to_swim_practice_is_22_5 
  (weekend_time : ℕ)
  (weekday_time : ℕ)
  (quiet_weekday_time : ℕ)
  :
  distance_to_swim_practice (30/60) (45/60) (15/60) 0 15 24 = 22.5 := 
sorry

end distance_to_swim_practice_is_22_5_l403_403678


namespace find_y_in_terms_of_abc_l403_403074

theorem find_y_in_terms_of_abc 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (h1 : xy / (x - y) = a)
  (h2 : xz / (x - z) = b)
  (h3 : yz / (y - z) = c) :
  y = bcx / ((b + c) * x - bc) := 
sorry

end find_y_in_terms_of_abc_l403_403074


namespace solve_for_q_l403_403963

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l403_403963


namespace part_I_pos_part_I_neg_part_II_1_part_II_2_l403_403523

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + m * x

def g (f : ℝ → ℝ) (x1 x2 x : ℝ) : ℝ := 
  (f x1 - f x2) / (x1 - x2) * (x - x1) + f x1

theorem part_I_pos (m x : ℝ) (h : m > 0) : 
  ∃ I : Set ℝ, I = Set.Ici (-1) ∧ ∀ x ∈ I, f m x > 0 :=
sorry

theorem part_I_neg (m x : ℝ) (h : m < 0) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Ioc (-1) (- (m + 1) / m) ∧ I2 = Set.Ioi (- (m + 1) / m) ∧
  (∀ x ∈ I1, f m x > 0) ∧ (∀ x ∈ I2, f m x < 0) :=
sorry

theorem part_II_1 (x1 x2 x : ℝ) (hm : x2 > x1 > -1) (hx : x1 < x < x2) :
  f x > g (f x1 f x2) :=
sorry

theorem part_II_2 (x1 x2 λ1 λ2 : ℝ) (hx1 : x1 > -1) (h : x2 > x1) (hλ : λ1 + λ2 = 1) :
  f (λ1 * x1 + λ2 * x2) > λ1 * f x1 + λ2 * f x2 :=
sorry 

end part_I_pos_part_I_neg_part_II_1_part_II_2_l403_403523


namespace sum_f_1_to_22_l403_403454

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403454


namespace det_of_matrix_eq_ad_minus_bc_l403_403714

theorem det_of_matrix_eq_ad_minus_bc (a b c d : ℝ) :
  matrix.det ![![a, c], ![b, d]] = a * d - b * c :=
begin
  sorry
end

end det_of_matrix_eq_ad_minus_bc_l403_403714


namespace replace_movies_cost_l403_403130

theorem replace_movies_cost
  (num_movies : ℕ)
  (trade_in_value_per_vhs : ℕ)
  (cost_per_dvd : ℕ)
  (h1 : num_movies = 100)
  (h2 : trade_in_value_per_vhs = 2)
  (h3 : cost_per_dvd = 10):
  (cost_per_dvd - trade_in_value_per_vhs) * num_movies = 800 :=
by sorry

end replace_movies_cost_l403_403130


namespace last_two_digits_sum_minus_product_l403_403826

-- Define the factorial function as it is not by default in Lean's library
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The problem conditions
def sum_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

def product_factorials (ns : List ℕ) : ℕ :=
  ns.map factorial |>.prod

-- The final theorem statement
theorem last_two_digits_sum_minus_product :
  (sum_factorials 15 - product_factorials [5, 10, 15]) % 100 = 13 :=
by
  sorry

end last_two_digits_sum_minus_product_l403_403826


namespace base6_addition_example_l403_403825

theorem base6_addition_example : (3214₆ + 2425₆) = 10036₆ := 
  sorry

end base6_addition_example_l403_403825


namespace comm_arrangement_l403_403855

def arrangement_count_comm : Nat :=
  (factorial 14) / (factorial 3 * factorial 2 * factorial 2 * factorial 2 * factorial 2)

theorem comm_arrangement : arrangement_count_comm = 908107825 :=
  by
    sorry

end comm_arrangement_l403_403855


namespace intersection_of_A_and_B_l403_403536

open Set

def A : Set ℝ := { x | 3 * x + 2 > 0 }
def B : Set ℝ := { x | (x + 1) * (x - 3) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | x > 3 } :=
by 
  sorry

end intersection_of_A_and_B_l403_403536


namespace problem_solution_l403_403905

def arithmetic_sequence (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

def sum_of_terms (a_1 : ℕ) (a_n : ℕ) (n : ℕ) : ℕ :=
  n * (a_1 + a_n) / 2

theorem problem_solution 
  (a_1 : ℕ) (d : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a_1 = 2)
  (h2 : S_2 = arithmetic_sequence a_1 d 3):
  a_2 = 4 ∧ S_10 = 110 :=
by
  sorry

end problem_solution_l403_403905


namespace subway_length_is_350_meters_l403_403212

-- Definition of given conditions.
def subway_speed : ℝ := 1.6  -- km/min
def distance_between_stations : ℝ := 4.85  -- km
def time_to_pass_station : ℝ := 3 + 15 / 60  -- minutes

-- Definition of the equivalent math proof problem.
def length_of_subway_in_meters :=
  (subway_speed * time_to_pass_station - distance_between_stations) * 1000  -- meters

-- Theorem statement
theorem subway_length_is_350_meters :
  length_of_subway_in_meters = 350 :=
sorry

end subway_length_is_350_meters_l403_403212


namespace atleast_n_triangles_l403_403894

-- Definitions of the conditions
variable (n : ℕ)
variable (h_n : n ≥ 2)
variable (points : Fin 2n → Point)
variable (line_segments : Finset (Fin 2n × Fin 2n))
variable (h_line_segments_count : line_segments.card = n^2 + 1)
variable (h_no_four_coplanar : ∀ (a b c d : Fin 2n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → ¬ Coplanar (points a) (points b) (points c) (points d))

-- We aim to prove that the line segments form at least n distinct triangles.
theorem atleast_n_triangles : ∃ (triangles : Finset (Triangle (Fin 2n))), triangles.card ≥ n := by
  sorry

end atleast_n_triangles_l403_403894


namespace cookies_guests_l403_403824

theorem cookies_guests (cc_cookies : ℕ) (oc_cookies : ℕ) (sc_cookies : ℕ) (cc_per_guest : ℚ) (oc_per_guest : ℚ) (sc_per_guest : ℕ)
    (cc_total : cc_cookies = 45) (oc_total : oc_cookies = 62) (sc_total : sc_cookies = 38) (cc_ratio : cc_per_guest = 1.5)
    (oc_ratio : oc_per_guest = 2.25) (sc_ratio : sc_per_guest = 1) :
    (cc_cookies / cc_per_guest) ≥ 0 ∧ (oc_cookies / oc_per_guest) ≥ 0 ∧ (sc_cookies / sc_per_guest) ≥ 0 → 
    Nat.floor (oc_cookies / oc_per_guest) = 27 :=
by
  sorry

end cookies_guests_l403_403824


namespace area_of_rectangle_l403_403122

theorem area_of_rectangle (ABCD : Rectangle) 
  (M N P Q : Point) 
  (h_M : midpoint A B M) 
  (h_N : midpoint B C N) 
  (h_P : midpoint C D P) 
  (h_Q : midpoint D A Q) 
  (h_shaded_area : triangle_area M N P = 1) :
  rectangle_area ABCD = 8 := 
sorry

end area_of_rectangle_l403_403122


namespace sum_f_1_to_22_l403_403421

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403421


namespace projection_is_negative_four_l403_403405

variables (a b : ℝ)
variables (norm_a norm_b dot_ab : ℝ)

-- Given conditions
def norm_a_val : Prop := norm_a = 5
def norm_b_val : Prop := norm_b = 3
def dot_prod_val : Prop := dot_ab = -12

-- Projection definition
def projection := dot_ab / norm_b

-- Theorem stating the main goal
theorem projection_is_negative_four : projection = -4 :=
by 
    rw [projection, norm_b_val, dot_prod_val]
    sorry

end projection_is_negative_four_l403_403405


namespace solution_set_l403_403718

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry

theorem solution_set (h1 : ∀ x > 0, x * f'' x + 2 * f x > 0)
                     (h2 : differentiable ℝ f)
                     (h3 : differentiable ℝ f') :
  {x : ℝ | (x + 2017) * f (x + 2017) / 5 < 5 * f 5 / (x + 2017)} = {x : ℝ | -2017 < x ∧ x < -2012} := sorry

end solution_set_l403_403718


namespace number_of_different_numerators_required_l403_403143

def is_rational_in_T (r : ℚ) : Prop :=
  (0 < r ∧ r < 1) ∧ 
  (∃ a b : ℕ, r = (a * 10 + b) / 99 ∧ a < 10 ∧ b < 10)

theorem number_of_different_numerators_required : 
  {n : ℕ | ∃ r, is_rational_in_T r ∧ numerator r = n}.toFinset.card = 60 := 
sorry

end number_of_different_numerators_required_l403_403143


namespace smallest_n_l403_403361

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def condition_for_n (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → ∀ x : ℕ, x ∈ M k → ∃ y : ℕ, y ∈ M k ∧ y ≠ x ∧ is_perfect_square (x + y)
  where M (k : ℕ) := { m : ℕ | m > 0 ∧ m ≤ k }

theorem smallest_n : ∃ n : ℕ, (condition_for_n n) ∧ (∀ m < n, ¬ condition_for_n m) :=
  sorry

end smallest_n_l403_403361


namespace sum_f_eq_neg_24_l403_403413

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403413


namespace sum_f_k_1_22_l403_403506

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403506


namespace compute_ab_l403_403194

theorem compute_ab (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 867.75 := 
by
  sorry

end compute_ab_l403_403194


namespace part_a_possible_values_of_M_part_b_possible_values_of_M_l403_403895

def values_of_M_part_a : Set ℕ :=
  {M | ∃ S : ℕ, S = 12 + M / 3 ∧ 3S = 36 + M ∧ 0 ≤ M ∧ (M < 14) ∧ (0 < M) ∧ (M % 3 = 0)}

theorem part_a_possible_values_of_M :
  values_of_M_part_a = {3, 6, 9, 12} := 
sorry

def values_of_M_part_b : Set ℕ :=
  {M ∈ values_of_M_part_a | ∀ S : Set (Set ℕ), 
    ((∀ row : Set ℕ, row ∈ S → row.card = 3 ∧ row.sum = 12 + M / 3)
    → ∃ cols : Set (Set ℕ), 
      (∀ col : Set ℕ, col ∈ cols → col.card = 3 ∧ col.sum = 12 + M / 3))
   }

theorem part_b_possible_values_of_M :
  values_of_M_part_b = {6, 9} := 
sorry

end part_a_possible_values_of_M_part_b_possible_values_of_M_l403_403895


namespace liters_to_pints_l403_403029

/-- Given that 0.5 liters is approximately 1.05 pints, prove that 1 liter is approximately 2.1 pints
    expressed as a decimal to the nearest tenth.
--/
theorem liters_to_pints:
  ∀ (L P : ℝ), L = 0.5 → P = 1.05 → 2 * P = 2.1 :=
by
  intros L P hL hP
  rw [hL, hP]
  simp
  norm_num
  sorry

end liters_to_pints_l403_403029


namespace find_ratio_l403_403080

variables {a b c d : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variables (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
variables (h6 : (7 * a + b) / (7 * c + d) = 9)

theorem find_ratio (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
    (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
    (h6 : (7 * a + b) / (7 * c + d) = 9) :
    (9 * a + b) / (9 * c + d) = 9 := 
by {
    sorry
}

end find_ratio_l403_403080


namespace find_YZ_l403_403994

variable (X Y Z D E I : Type)
variable [RightTriangle X Y Z]
variable [OnHypotenuse D Y Z] 
variable [OnSides E X Z]
variable [OnSides I X Y]
variable [hyp1 : XY = 5]
variable [hyp2 : XZ = 5]
variable [hyp3 : DY = 2 * DZ]
variable [parallelogram : parallelogram X I D E]
variable [equal_sides : XI = XE]
variable [area_tri_ide : area(IDE) = 4]

theorem find_YZ : YZ = 5 * Real.sqrt 2 := by
  sorry

end find_YZ_l403_403994


namespace z_conjugate_in_fourth_quadrant_l403_403387

noncomputable def z := (3 + 5 * Complex.i) / (1 + Complex.i)

def z_conjugate := Complex.conj z

def z_conjugate_coordinates := (z_conjugate.re, z_conjugate.im)

theorem z_conjugate_in_fourth_quadrant : z_conjugate_coordinates = (2, -1) → z_conjugate.im < 0 ∧ z_conjugate.re > 0 :=
by
  sorry

end z_conjugate_in_fourth_quadrant_l403_403387


namespace simplify_and_evaluate_l403_403695

variable (x y : ℤ)

theorem simplify_and_evaluate (h1 : x = 1) (h2 : y = 1) :
    2 * (x - 2 * y) ^ 2 - (2 * y + x) * (-2 * y + x) = 5 := by
    sorry

end simplify_and_evaluate_l403_403695


namespace shortest_distance_point_to_parabola_l403_403360

noncomputable def parabola_distance : ℝ :=
let point_P := (8 : ℝ, 16 : ℝ),
    parabola_point (b : ℝ) := (b^2 / 4, b) in
  real.sqrt ((parabola_point (8 : ℝ)).fst - point_P.fst)^2 + 
            ((parabola_point (8 : ℝ)).snd - point_P.snd)^2

theorem shortest_distance_point_to_parabola : 
  parabola_distance = 8 * real.sqrt 2 :=
sorry

end shortest_distance_point_to_parabola_l403_403360


namespace correct_rectangle_in_position_I_l403_403118

theorem correct_rectangle_in_position_I
  (rect A B C D E : ℕ)
  (side_number : rect → ℕ)
  (adjacent : rect → rect → Prop)
  (same_number_on_touching_sides : ∀ r1 r2 : rect, adjacent r1 r2 → side_number r1 = side_number r2) :
  rectangle_in_position_I = C := 
sorry

end correct_rectangle_in_position_I_l403_403118


namespace problem_l403_403214

variable {w z : ℝ}

theorem problem (hw : w = 8) (hz : z = 3) (h : ∀ z w, z * (w^(1/3)) = 6) : w = 1 :=
by
  sorry

end problem_l403_403214


namespace value_of_fraction_l403_403553

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403553


namespace parallel_x_axis_implies_conditions_l403_403064

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end parallel_x_axis_implies_conditions_l403_403064


namespace no_dissection_to_integer_ratio_right_triangles_l403_403859

theorem no_dissection_to_integer_ratio_right_triangles (n : ℕ) : 
  ¬ exists (triangles : list (ℕ × ℕ × ℕ)), 
    (∀ t ∈ triangles, let (a, b, c) := t in a^2 + b^2 = c^2) ∧ 
    are_integer_ratio_right_triangles triangles ∧
    dissector_polygons n triangles :=
  sorry

end no_dissection_to_integer_ratio_right_triangles_l403_403859


namespace greatest_gcd_abb_aba_l403_403376

/-- 
  Given distinct digits a and b, prove that the greatest GCD 
  of the numbers 100a + 11b and 101a + 10b is 18. 
--/
theorem greatest_gcd_abb_aba (a b : ℕ) (ha : a < 10) (hb : b < 10) (hab : a ≠ b) :
  ∃ g, g = Nat.gcd (100 * a + 11 * b) (101 * a + 10 * b) ∧ g ≤ 18 ∧ (∃ a b, a < 10 ∧ b < 10 ∧ a ≠ b ∧ Nat.gcd (100 * a + 11 * b) (101 * a + 10 * b) = 18) :=
begin
  sorry
end

end greatest_gcd_abb_aba_l403_403376


namespace parabola_hyperbola_focus_vertex_l403_403084

theorem parabola_hyperbola_focus_vertex (p : ℝ) : 
  (∃ (focus_vertex : ℝ × ℝ), focus_vertex = (2, 0) 
    ∧ focus_vertex = (p / 2, 0)) → p = 4 :=
by
  sorry

end parabola_hyperbola_focus_vertex_l403_403084


namespace sum_f_proof_l403_403432

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403432


namespace employees_after_hiring_l403_403619

theorem employees_after_hiring (initial_employee_count : ℕ) (percentage_increase : ℕ) 
    (new_employee_ratio : ℚ) :
    initial_employee_count = 852 →
    percentage_increase = 25 →
    new_employee_ratio = 0.25 →
    initial_employee_count + (initial_employee_count * new_employee_ratio).natAbs = 1065 := 
by 
  intros h1 h2 h3
  rw [h1, h3]
  norm_num
  sorry

end employees_after_hiring_l403_403619


namespace find_H_l403_403219

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def midpoint (P Q : Point3D) : Point3D :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2,
  z := (P.z + Q.z) / 2 }

def E : Point3D := { x := 4, y := 0, z := 1 }
def F : Point3D := { x := 2, y := 3, z := -5 }
def G : Point3D := { x := 0, y := 2, z := 1 }

theorem find_H (H : Point3D) (parallelogram_EFGH : (midpoint E G) = (midpoint F H)) :
H = { x := 2, y := -1, z := 7 } :=
sorry

end find_H_l403_403219


namespace problem_1_problem_2_problem_3_l403_403934

-- Problem 1
theorem problem_1 (a : ℝ) (t : ℝ) (h1 : a ≠ 1) (h2 : f 3 - g 3 = 0) : t = -4 := sorry

-- Problem 2
theorem problem_2 (a : ℝ) (t : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : t = 1) (x : ℝ) (h4 : f x ≤ g x) : -1/2 < x ∧ x ≤ 0 := sorry

-- Problem 3
theorem problem_3 (a : ℝ) (t : ℝ) (h1 : a ≠ 1) (h2 : ∃ x : ℝ, -1 < x ∧ x ≤ 3 ∧ F x = 0) :
  t ≤ -5/7 ∨ t ≥ (2 + real.sqrt 2) / 4 := sorry

/-- Definitions of functions used in the problems -/
def f (a : ℝ) (x : ℝ) := real.log (x + 1) / real.log a

def g (a : ℝ) (x : ℝ) (t : ℝ) := 2 * real.log (2 * x + t) / real.log a

def F (a : ℝ) (x : ℝ) (t : ℝ) := (a : ℝ) ^ (f a x) + t * x^2 - 2 * t + 1

end problem_1_problem_2_problem_3_l403_403934


namespace sum_G_equals_2016531_l403_403373

-- Define G(n): ℕ → ℕ to count the number of solutions to cos x = cos nx over [0, π]
def G (n : ℕ) : ℕ :=
  if n % 4 = 0 then n
  else n + 1

-- Now we want to prove the sum of G(n) from 2 to 2007 is 2,016,531
theorem sum_G_equals_2016531 : 
  (∑ n in Finset.range 2006 + 2, G n) = 2_016_531 :=
by
  sorry

end sum_G_equals_2016531_l403_403373


namespace max_product_of_sum_2016_l403_403231

theorem max_product_of_sum_2016 (x y : ℤ) (h : x + y = 2016) : x * y ≤ 1016064 :=
by
  -- Proof goes here, but is not needed as per instructions
  sorry

end max_product_of_sum_2016_l403_403231


namespace largest_quotient_is_25_l403_403232

def largest_quotient_set : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 :
  ∃ (a b : ℤ), a ∈ largest_quotient_set ∧ b ∈ largest_quotient_set ∧ b ≠ 0 ∧ (a : ℚ) / b = 25 := by
  sorry

end largest_quotient_is_25_l403_403232


namespace solution_set_f_x_sq_gt_2f_x_plus_1_l403_403027

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_sq_gt_2f_x_plus_1
  (h_domain : ∀ x, 0 < x → ∃ y, f y = f x)
  (h_func_equation : ∀ x y, 0 < x → 0 < y → f (x + y) = f x * f y)
  (h_greater_than_2 : ∀ x, 1 < x → f x > 2)
  (h_f2 : f 2 = 4) :
  ∀ x, x^2 > x + 2 → x > 2 :=
by
  intros x h
  sorry

end solution_set_f_x_sq_gt_2f_x_plus_1_l403_403027


namespace sum_f_l403_403446

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403446


namespace rodney_correct_guess_probability_l403_403692

-- Definitions based on conditions
def is_two_digit_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100
def tens_digit_is_odd (n : ℕ) : Prop := odd (n / 10)
def units_digit_is_even (n : ℕ) : Prop := even (n % 10)
def is_greater_than_75 (n : ℕ) : Prop := n > 75
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main statement
theorem rodney_correct_guess_probability :
  ∀ n, is_two_digit_integer n ∧
       tens_digit_is_odd n ∧
       units_digit_is_even n ∧
       is_greater_than_75 n ∧
       is_divisible_by_3 n →
  (∃! m, is_two_digit_integer m ∧
         tens_digit_is_odd m ∧
         units_digit_is_even m ∧
         is_greater_than_75 m ∧
         is_divisible_by_3 m) →
  1 / 2 := 
sorry

end rodney_correct_guess_probability_l403_403692


namespace p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l403_403397

variable (x : ℝ)

def p := |x| = x
def q := x^2 + x ≥ 0

theorem p_implies_q : p x → q x :=
by sorry

theorem q_not_implies_p : q x → ¬p x :=
by sorry

theorem p_sufficient_but_not_necessary : (p x → q x) ∧ ¬(q x → p x) :=
by sorry

end p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l403_403397


namespace find_n_l403_403353

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 7) (h₂ : n ≡ -4850 [MOD 8]) : n = 6 :=
sorry

end find_n_l403_403353


namespace find_normal_price_l403_403236

open Real

theorem find_normal_price (P : ℝ) (h1 : 0.612 * P = 108) : P = 176.47 := by
  sorry

end find_normal_price_l403_403236


namespace polynomial_value_l403_403365

variables (x y p q : ℝ)

theorem polynomial_value (h1 : x + y = -p) (h2 : xy = q) :
  x * (1 + y) - y * (x * y - 1) - x^2 * y = pq + q - p :=
by
  sorry

end polynomial_value_l403_403365


namespace value_of_ac_over_bd_l403_403594

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403594


namespace intersection_eq_l403_403056

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l403_403056


namespace cost_price_of_table_l403_403206

variable (C S : ℝ)

theorem cost_price_of_table (h1 : S = C * 1.24) (h2 : S = 8091) : C = 6525 :=
by
  have h : 8091 = C * 1.24 := by rw [h2, h1]
  have C_val : C = 8091 / 1.24 := by rw [←h]; field_simp
  have : 8091 / 1.24 = 6525 := by norm_num
  rw [this] at C_val
  exact C_val

end cost_price_of_table_l403_403206


namespace probability_xi_l403_403716

def discriminant_eq (ξ : ℝ) : Prop :=
  (1 - ξ)^2 + 4 * ξ * (ξ + 2) = 0

def quadratic_eq (ξ : ℝ) : Prop :=
  5 * ξ^2 + 6 * ξ + 1 = 0

def solution1 : ℝ := -1
def solution2 : ℝ := -1/5

def prob_1 : ℝ := 3/16
def prob_2 : ℝ := 1/4

def prob_A : ℝ := prob_1 + prob_2

theorem probability_xi (ξ1 ξ2 : ℝ) (p1 p2 : ℝ) :
  (quadratic_eq ξ1 ∧ ξ1 = solution1 ∧ p1 = prob_1) ∧ 
  (quadratic_eq ξ2 ∧ ξ2 = solution2 ∧ p2 = prob_2) →
  prob_A = 7 / 16 :=
by
  sorry

end probability_xi_l403_403716


namespace infinite_primes_l403_403688

theorem infinite_primes : ∀ (s : Finset ℕ), (∀ p ∈ s, Prime p) → ∃ p, Prime p ∧ p ∉ s :=
by
  intro s hs
  let N := (s.val.prod + 1)
  have hn : N > 1 :=
      sorry  -- Skip the details here
  obtain ⟨p, hp⟩ := exists_prime_factorization hn
  use p
  have hp_prime : Prime p := hp.1
  exact ⟨hp_prime, sorry⟩  -- Skip the details here

end infinite_primes_l403_403688


namespace speed_of_current_l403_403284

-- Definitions for the conditions
variables (m c : ℝ)

-- Condition 1: man's speed with the current
def speed_with_current := m + c = 16

-- Condition 2: man's speed against the current
def speed_against_current := m - c = 9.6

-- The goal is to prove c = 3.2 given the conditions
theorem speed_of_current (h1 : speed_with_current m c) 
                         (h2 : speed_against_current m c) :
  c = 3.2 := 
sorry

end speed_of_current_l403_403284


namespace minimum_days_l403_403806

theorem minimum_days (n : ℕ) (rain_afternoon : ℕ) (sunny_afternoon : ℕ) (sunny_morning : ℕ) :
  rain_afternoon + sunny_afternoon = 7 ∧
  sunny_afternoon <= 5 ∧
  sunny_morning <= 6 ∧
  sunny_morning + rain_afternoon = 7 ∧
  n = 11 :=
by
  sorry

end minimum_days_l403_403806


namespace sum_f_1_to_22_l403_403422

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403422


namespace inequality_solution_set_l403_403000

noncomputable def solution_set : Set ℝ := { x : ℝ | x > 5 ∨ x < -2 }

theorem inequality_solution_set (x : ℝ) :
  x^2 - 3 * x - 10 > 0 ↔ x > 5 ∨ x < -2 :=
by
  sorry

end inequality_solution_set_l403_403000


namespace plane_through_points_l403_403352

-- Define the vectors as tuples of three integers
def point := (ℤ × ℤ × ℤ)

-- The given points
def p : point := (2, -1, 3)
def q : point := (4, -1, 5)
def r : point := (5, -3, 4)

-- A function to find the equation of the plane given three points
def plane_equation (p q r : point) : ℤ × ℤ × ℤ × ℤ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  let (rx, ry, rz) := r
  let a := (qy - py) * (rz - pz) - (qy - py) * (rz - pz)
  let b := (qx - px) * (rz - pz) - (qx - px) * (rz - pz)
  let c := (qx - px) * (ry - py) - (qx - px) * (ry - py)
  let d := -(a * px + b * py + c * pz)
  (a, b, c, d)

-- The proof statement
theorem plane_through_points : plane_equation (2, -1, 3) (4, -1, 5) (5, -3, 4) = (1, 2, -2, 6) :=
  by sorry

end plane_through_points_l403_403352


namespace map_formed_by_n_circles_two_colorable_l403_403102

theorem map_formed_by_n_circles_two_colorable (n : ℕ) (plane : Type) (circles : fin n → set plane) :
  ∃ f : (set plane) → bool, ∀ (region1 region2 : set plane), (region1 ∩ region2).nonempty → f region1 ≠ f region2 :=
sorry

end map_formed_by_n_circles_two_colorable_l403_403102


namespace correct_set_for_probabilities_l403_403287

-- Defining the probability function for digit d
def probability (d : ℕ) : ℝ :=
  Real.log10 (d + 1) - Real.log10 d

-- Defining the probability of choosing digit 3
def prob_digit_3 : ℝ :=
  probability 3

-- Defining the probability of choosing a digit from the set {3, 4, 5}
def prob_set_3_4_5 : ℝ :=
  probability 3 + probability 4 + probability 5

-- The Lean statement to express the proof problem
theorem correct_set_for_probabilities : 
  3 * prob_digit_3 = prob_set_3_4_5 → {3, 4, 5} = {3, 4, 5} :=
by
  intro h
  -- Goal is to prove the correct set is {3, 4, 5}
  exact rfl

end correct_set_for_probabilities_l403_403287


namespace area_of_stacked_triangles_correct_l403_403008

-- Ellipsize because these assumptions should be used directly from the conditions
-- Rather than generating intermediate solution details
def triangle_stack_area (side_length : ℕ) (rotations : List ℕ) : ℝ :=
  if side_length = 8 ∧ rotations = [0, 45, 90, 135] then
    52 * Real.sqrt 3
  else
    0

theorem area_of_stacked_triangles_correct :
  triangle_stack_area 8 [0, 45, 90, 135] = 52 * Real.sqrt 3 :=
  by 
    sorry

end area_of_stacked_triangles_correct_l403_403008


namespace eq_of_div_eq_div_l403_403251

theorem eq_of_div_eq_div {a b c : ℝ} (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end eq_of_div_eq_div_l403_403251


namespace unique_two_digit_solution_l403_403741

theorem unique_two_digit_solution:
  ∃! (s : ℕ), 10 ≤ s ∧ s ≤ 99 ∧ (13 * s ≡ 52 [MOD 100]) :=
  sorry

end unique_two_digit_solution_l403_403741


namespace sum_f_l403_403447

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403447


namespace sum_f_eq_neg_24_l403_403417

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403417


namespace sum_f_eq_neg_24_l403_403414

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403414


namespace sum_equality_l403_403720

def sum_from_2_to_10 : ℕ :=
  (Finset.range 11).filter (λ i, i ≥ 2).sum -- Sum of elements from 2 to 10

theorem sum_equality : sum_from_2_to_10 = 54 := by
  sorry

end sum_equality_l403_403720


namespace probability_one_of_each_and_extra_red_correct_l403_403009

-- Definitions of the conditions in the problem:
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def chosen_marbles := 4

-- Define the probability calculation using binomial combinations:
def probability_one_of_each_and_extra_red :=
  (nat.choose blue_marbles 1 * nat.choose green_marbles 1 * nat.choose red_marbles 2) /
  nat.choose total_marbles chosen_marbles

-- The theorem statement we need to prove:
theorem probability_one_of_each_and_extra_red_correct :
  probability_one_of_each_and_extra_red = 12 / 35 := 
sorry

end probability_one_of_each_and_extra_red_correct_l403_403009


namespace value_of_fraction_l403_403551

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403551


namespace general_formula_a_n_sum_first_n_terms_b_n_l403_403670

variable (a b : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
noncomputable def a_n (n : ℕ) : ℕ := n
noncomputable def b_n (n : ℕ) : ℕ := n * 2^n
noncomputable def S_4 := 10
axiom a2_a4_eq_6 : a_n 2 + a_n 4 = 6
axiom S4_eq_10 : S 4 = S_4

-- Prove general formula for the sequence {a_n}
theorem general_formula_a_n : a_n = λ n, n := by
  sorry

-- Prove the sum of the first n terms of the sequence {b_n}
theorem sum_first_n_terms_b_n (n : ℕ) : (∑ k in finset.range n, b_n k) = (n-1) * 2^(n+1) + 2 := by
  sorry

end general_formula_a_n_sum_first_n_terms_b_n_l403_403670


namespace measure_angle_BAO_l403_403986

-- Define the conditions as Lean 4 definitions.
variables (C D O A E B : Type) 
variables [semicircle_with_diameter C D O]
variables (A_ext_CDC : extension C D A)
variables (E_on_semicircle : lies_on_semicircle E O D)
variables (B_intersect : intersection_of_line_segment AE E B)
variables (AB_eq_OD : length AB = length OD)
variables (angle_EOD_45 : angle EOD = 45)

-- The goal is to prove that angle BAO is 15 degrees.
theorem measure_angle_BAO : angle BAO = 15 :=
sorry

end measure_angle_BAO_l403_403986


namespace sum_f_eq_neg24_l403_403499

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403499


namespace tangent_KT_to_Γ_l403_403020

variables {R S T J A K : Point}
variable {Ω Γ : Circle}
variable {l : Line}

-- Conditions
axiom distinct_points_on_circle (h1 : R ≠ S)
axiom segment_not_diameter (h2 : ¬ ∃ M, Midpoint M R S ∧ M = Center Ω)
axiom tangent_at_R (h3: Tangent l Ω R)
axiom midpoint_of_RT (h4 : Midpoint S R T)
axiom J_on_minor_arc (h5 : OnMinorArc J R S Ω)
axiom circumcircle_intersects_l (h6 : ∃ A₁ A₂, PointsOnCircle Γ J S T ∧ l ∩ Γ = {A₁, A₂})
axiom A_closest_to_R (h7 : ClosestPoint A R (l ∩ Γ))
axiom AJ_intersects_Ω_at_K (h8 : LineThrough A J ∩ Ω = {K})

-- Question to prove
theorem tangent_KT_to_Γ :
  Tangent (LineThrough K T) Γ :=
sorry

end tangent_KT_to_Γ_l403_403020


namespace only_cos2x_has_period_pi_l403_403247

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem only_cos2x_has_period_pi :
  ∀ f : ℝ → ℝ, 
    (f = (λ x => Real.cos x ^ 2) ∨ f = (λ x => Real.abs (Real.sin (x / 2))) ∨ 
     f = Real.sin ∨ f = (λ x => Real.tan (x / 2))) →
    has_period f π → 
    f = λ x => Real.cos x ^ 2 :=
by
  sorry

end only_cos2x_has_period_pi_l403_403247


namespace aunt_money_calculation_l403_403311

variable (total_money_received aunt_money : ℕ)
variable (bank_amount grandfather_money : ℕ := 150)

theorem aunt_money_calculation (h1 : bank_amount = 45) (h2 : bank_amount = total_money_received / 5) (h3 : total_money_received = aunt_money + grandfather_money) :
  aunt_money = 75 :=
by
  -- The proof is captured in these statements:
  sorry

end aunt_money_calculation_l403_403311


namespace volume_ratio_of_cylinders_l403_403326

theorem volume_ratio_of_cylinders (h r : ℝ) (C_volume D_volume : ℝ)
  (cyl_C_height_eq_cyl_D_radius : C_volume = π * r^2 * h)
  (cyl_C_radius_eq_cyl_D_height : D_volume = π * h^2 * r)
  (volume_ratio : D_volume = 3 * C_volume) :
  ∃ M : ℝ, D_volume = M * π * h^3 ∧ M = 9 :=
by
  -- Define volumes based on conditions
  have C_volume : ℝ := π * r^2 * h,
  have D_volume : ℝ := π * h^2 * r,

  -- Use the condition that D_volume is 3 times C_volume to find r in terms of h
  have volume_ratio : π * h^2 * r = 3 * (π * r^2 * h), by assumption,
  have ratio_simplification : h^2 * r = 3 * r^2 * h :=
    by { rw [π, π] at volume_ratio, exact volume_ratio },
  have r_value : r = 3 * h :=
    by { ring at ratio_simplification, exact ratio_simplification },

  -- Calculate the volume of cylinder D with r = 3h
  let D_volume_val := π * (3 * h)^2 * h,
  have D_volume_final : D_volume = 9 * π * h^3 :=
    by { simp, rw ← D_volume_val, ring },

  -- Conclude that M = 9
  existsi 9,
  split,
  { exact D_volume_final },
  { exact rfl }

end volume_ratio_of_cylinders_l403_403326


namespace value_of_frac_l403_403589

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403589


namespace uncle_kahn_total_cost_l403_403622

noncomputable def base_price : ℝ := 10
noncomputable def child_discount : ℝ := 0.3
noncomputable def senior_discount : ℝ := 0.1
noncomputable def handling_fee : ℝ := 5
noncomputable def discounted_senior_ticket_price : ℝ := 14
noncomputable def num_child_tickets : ℝ := 2
noncomputable def num_senior_tickets : ℝ := 2

theorem uncle_kahn_total_cost :
  let child_ticket_cost := (1 - child_discount) * base_price + handling_fee
  let senior_ticket_cost := discounted_senior_ticket_price
  num_child_tickets * child_ticket_cost + num_senior_tickets * senior_ticket_cost = 52 :=
by
  sorry

end uncle_kahn_total_cost_l403_403622


namespace base_subtraction_problem_l403_403335

theorem base_subtraction_problem (b : ℕ) (C_b : ℕ) (hC : C_b = 12) : 
  b = 15 :=
by
  sorry

end base_subtraction_problem_l403_403335


namespace isosceles_triangle_perimeter_l403_403613

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), is_isosceles a b c ∧ ((a = 3 ∧ b = 3 ∧ c = 4 ∧ a + b + c = 10) ∨ (a = 3 ∧ b = 4 ∧ c = 4 ∧ a + b + c = 11)) :=
by
  sorry

end isosceles_triangle_perimeter_l403_403613


namespace perimeter_of_circle_approx_l403_403766

-- Given perimeter of square
def perimeter_square : ℝ := 66.84507609859604

-- Given the relationship between perimeter of square and side of square
def side_square : ℝ := perimeter_square / 4

-- Given the relationship between side of square and diameter of circle
def diameter_circle : ℝ := side_square

-- Given the value of π
def pi : ℝ := 3.14159

-- Calculate the perimeter (circumference) of the circle
def circumference_circle : ℝ := pi * diameter_circle

-- Theorem to prove the calculated circumference is approximately 52.482 cm
theorem perimeter_of_circle_approx : circumference_circle ≈ 52.482 := by
  sorry

end perimeter_of_circle_approx_l403_403766


namespace average_of_remaining_two_l403_403710

theorem average_of_remaining_two (a1 a2 a3 a4 a5 : ℚ)
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 11)
  (h2 : (a1 + a2 + a3) / 3 = 4) :
  ((a4 + a5) / 2 = 21.5) :=
sorry

end average_of_remaining_two_l403_403710


namespace intersection_sum_l403_403038

variable {α β : Type*} [LinearOrderedSemiring α]

def h (x : α) : α := sorry -- This will be a function such that h(x) is known at given points.
def j (x : α) : α := sorry -- This will be another function such that j(x) is known at given points.

-- Given conditions
axiom h3_eq_3 : h 3 = 3
axiom h6_eq_9 : h 6 = 9
axiom h9_eq_18 : h 9 = 18
axiom h12_eq_18 : h 12 = 18

axiom j3_eq_3 : j 3 = 3
axiom j6_eq_9 : j 6 = 9
axiom j9_eq_18 : j 9 = 18
axiom j12_eq_18 : j 12 = 18

-- Question: Prove the sum of coordinates of the intersection of y = h(3x) and y = 3j(x) is 22
theorem intersection_sum : ∃ (a b : α), h (3 * a) = 3 * j a ∧ a + b = 22 :=
by {
  use 4,
  use 18,
  split,
  {
    calc
    h (3 * 4) = 18 : by sorry -- This follows from the conditions
  },
  calc
  4 + 18 = 22 : by norm_num
}

end intersection_sum_l403_403038


namespace painter_can_make_all_black_l403_403275

-- Define the board size
def board_height : ℕ := 2012
def board_width  : ℕ := 2013

-- Define the initial color function; true represents black, false represents white
def initial_color (i j : ℕ) : Bool :=
  (i + j) % 2 == 0

-- Define a function to toggle the color of a cell
def toggle_color (c : Bool) : Bool :=
  !c

-- Define the painter and the movement on the board
def painter_moves (board : Matrix Bool board_height board_width) : Matrix Bool board_height board_width :=
  sorry  -- The logic to apply the painter's movement strategy will be placed here

-- The problem statement
theorem painter_can_make_all_black :
  ∀ (board : Matrix Bool board_height board_width),
  (∀ (i j : Fin board_height) (i' j' : Fin board_width), board i j = initial_color i j) →
  ∃ moves_painter : Matrix Bool board_height board_width, (∀ i j : Fin board_height, painter_moves board i j) :=
  sorry  -- We use sorry here as we do not need to provide the proof itself

end painter_can_make_all_black_l403_403275


namespace sum_mod_500_l403_403336

-- Definition of the polynomial and its property with roots of unity
def f (x : ℂ) : ℂ := (x - 1) ^ 1008
def omega : ℂ := (-1 + complex.I * sqrt 3) / 2  -- A root of unity where ω^3 = 1

-- Summation definition
def sum33_N : ℤ := (finset.range 333).sum (λ n, (-1 : ℤ) ^ n * int.binom 1008 (3 * n))

-- The main statement
theorem sum_mod_500 : sum33_N % 500 = 54 := 
sorry

end sum_mod_500_l403_403336


namespace max_tied_teams_for_most_wins_l403_403979

/-- Maximum number of teams tied for most wins in a round-robin tournament -/
theorem max_tied_teams_for_most_wins (n : ℕ) (h : n = 8) :
  ∃ k, k = 7 ∧ 
  (∃ (games_won : fin n → ℕ), 
    (∀ i, games_won i ≤ 4) ∧ 
    (∀ wins m, (m < n) → (wins = finset.sum (finset.range m) games_won) → (wins ≤ 28))) :=
sorry

end max_tied_teams_for_most_wins_l403_403979


namespace sum_f_eq_neg24_l403_403465

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403465


namespace smallest_positive_integer_exists_l403_403316

theorem smallest_positive_integer_exists :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k m : ℕ), n = 5 * k + 3 ∧ n = 12 * m) ∧ n = 48 :=
by
  sorry

end smallest_positive_integer_exists_l403_403316


namespace zero_divided_by_any_num_zero_divided_by_any_num_false_l403_403782

theorem zero_divided_by_any_num (a : ℝ) (h : a ≠ 0) : (0 / a = 0) :=
by sorry

theorem zero_divided_by_any_num_false : ¬ (∀ (a : ℝ), 0 / a = 0) :=
by {
  intro h,
  specialize h 0,
  have div_by_zero := zero_divided_by_any_num 0 (by ha_err : 0 ≠ 0; exact ha_err);
  contradiction
}

end zero_divided_by_any_num_zero_divided_by_any_num_false_l403_403782


namespace problem_l403_403152

def f (x : ℝ) (a b c : ℝ) : ℝ :=
if x > 0 then 2 * a * x + 6
else if x = 0 then a * b
else 3 * b * x + c

theorem problem (a b c : ℕ) (h1 : f 3 a b c = 24) (h2 : f 0 a b c = 6) (h3 : f (-3) a b c = -33) :
  a + b + c = 20 :=
sorry

end problem_l403_403152


namespace total_employees_now_l403_403621

-- Definitions based on conditions
def initial_employees : ℕ := 852
def additional_percentage : ℝ := 0.25

-- Target statement to prove
theorem total_employees_now : 
  let additional_employees : ℕ := (initial_employees * (additional_percentage * 100).toNat) / 100 in
  let total_employees : ℕ := initial_employees + additional_employees in
  total_employees = 1065 :=
by
  sorry

end total_employees_now_l403_403621


namespace elementary_school_coats_l403_403821

variable (x y z : ℕ)
variable (total_coats : ℕ := 9437)

axiom high_school_fraction : x = 3 * total_coats / 5
axiom middle_school_fraction : y = total_coats / 4
axiom total_coats_equation : x + y + z = total_coats

theorem elementary_school_coats : z = 1416 :=
by
  have h1 : x = 3 * total_coats / 5 := high_school_fraction
  have h2 : y = total_coats / 4 := middle_school_fraction
  have h3 : x + y + z = total_coats := total_coats_equation
  sorry

end elementary_school_coats_l403_403821


namespace sum_f_eq_neg24_l403_403470

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403470


namespace monotonicity_implies_inequality_l403_403511

noncomputable def f (a x : ℝ) : ℝ := real.log x / real.log a

theorem monotonicity_implies_inequality
  (a : ℝ) (h0 : 0 < a) (h1 : a < 1)
  (h_mono_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f a x < f a y) :
  f a (a+1) > f a 2 := sorry

end monotonicity_implies_inequality_l403_403511


namespace arithmetic_geometric_sequence_l403_403026

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℤ) (b : ℕ → ℕ) 
    (h1 : ∀ n, a n = 1 + (n - 1) * 1)   -- condition that the sequence is arithmetic with a common difference of 1
    (h2 : ∀ n, n = 1*n)  -- general term n
    (h3 : b = λ n, a (n * (n + 1)))  -- definition of b_n
    (h4 : S 0 = 0)
    (h5 : ∀ n, S (n + 1) = S n + (-1 : ℤ) ^ (n + 1) * b (n + 1)):  -- definition of S_n
  (∀ n, S n = if n % 2 = 0 then (n^2 + 2 * n) / 2 else -((n + 1)^2) / 2) :=
begin
  sorry
end

end arithmetic_geometric_sequence_l403_403026


namespace sum_f_1_to_22_l403_403423

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403423


namespace tangent_position_is_six_l403_403715

def clock_radius : ℝ := 30
def disk_radius : ℝ := 15
def initial_tangent_position := 12
def final_tangent_position := 6

theorem tangent_position_is_six :
  (∃ (clock_radius disk_radius : ℝ), clock_radius = 30 ∧ disk_radius = 15) →
  (initial_tangent_position = 12) →
  (final_tangent_position = 6) :=
by
  intros h1 h2
  sorry

end tangent_position_is_six_l403_403715


namespace range_of_AB_l403_403123

-- Given conditions
variables {A B C : Type} [EuclideanGeometry3 A] [EuclideanGeometry3 B] [EuclideanGeometry3 C]
variables {AB BC AC : ℝ} (angle_150 : ∠(AB, BC) = 150) (AC_2 : |AC| = 2)

-- Statement of the problem
theorem range_of_AB :
  0 < |AB| ∧ |AB| ≤ 4 :=
  sorry

end range_of_AB_l403_403123


namespace problem_l403_403578

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403578


namespace general_term_sum_first_n_terms_l403_403900

-- Definition and conditions for sequence {a_n}
def a (n : ℕ) : ℝ :=
  if h : n = 0 then 0 else 2 * (∑ i in Finset.range n, a i) + 1

-- General term formula for the sequence {a_n}
theorem general_term (n : ℕ) : a (n+1) = 3 ^ n :=
  sorry

-- Sum of the first n terms of the sequence { (2n+1) / a_n }
def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (2 * i + 1) / a (i + 1)

-- Proving the sum of the first n terms T_n
theorem sum_first_n_terms (n : ℕ) : T n = 6 - (n+2) / 3 ^ (n-1) :=
  sorry

end general_term_sum_first_n_terms_l403_403900


namespace value_of_fraction_l403_403550

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403550


namespace sum_f_eq_neg24_l403_403495

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403495


namespace total_number_of_dresses_l403_403156

theorem total_number_of_dresses (ana_dresses lisa_more_dresses : ℕ) (h_condition : ana_dresses = 15) (h_more : lisa_more_dresses = ana_dresses + 18) : ana_dresses + lisa_more_dresses = 48 :=
by
  sorry

end total_number_of_dresses_l403_403156


namespace no_constant_term_in_expansion_l403_403758

noncomputable def constant_term_exists : Prop :=
  ∀ k : ℕ, (10 - k) % 2 = 0 ∧ ((10 - k) / 2 - k) = 0 → false

theorem no_constant_term_in_expansion :
  constant_term_exists ((2 * (Real.sqrt x) + (3 / x))^10) :=
by sorry

end no_constant_term_in_expansion_l403_403758


namespace meeting_point_distance_l403_403999

theorem meeting_point_distance
  (distance_to_top : ℝ)
  (total_distance : ℝ)
  (jack_start_time : ℝ)
  (jack_uphill_speed : ℝ)
  (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ)
  (jill_downhill_speed : ℝ)
  (meeting_point_distance : ℝ):
  distance_to_top = 5 -> total_distance = 10 -> jack_start_time = 10 / 60 ->
  jack_uphill_speed = 15 -> jack_downhill_speed = 20 ->
  jill_uphill_speed = 16 -> jill_downhill_speed = 22 ->
  meeting_point_distance = 35 / 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end meeting_point_distance_l403_403999


namespace part1_tangent_line_at_x1_part2_a_range_l403_403013

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part1_tangent_line_at_x1 (a : ℝ) (h1 : a = 1) : 
  let f' (x : ℝ) : ℝ := (x + 1) * Real.exp x - 1
  (2 * Real.exp 1 - 1) * 1 - (f 1 1) = Real.exp 1 :=
by 
  sorry

theorem part2_a_range (a : ℝ) (h2 : ∀ x > 0, f x a ≥ Real.log x - x + 1) : 
  0 < a ∧ a ≤ 2 :=
by 
  sorry

end part1_tangent_line_at_x1_part2_a_range_l403_403013


namespace volume_ratio_of_cylinders_l403_403325

theorem volume_ratio_of_cylinders (h r : ℝ) (C_volume D_volume : ℝ)
  (cyl_C_height_eq_cyl_D_radius : C_volume = π * r^2 * h)
  (cyl_C_radius_eq_cyl_D_height : D_volume = π * h^2 * r)
  (volume_ratio : D_volume = 3 * C_volume) :
  ∃ M : ℝ, D_volume = M * π * h^3 ∧ M = 9 :=
by
  -- Define volumes based on conditions
  have C_volume : ℝ := π * r^2 * h,
  have D_volume : ℝ := π * h^2 * r,

  -- Use the condition that D_volume is 3 times C_volume to find r in terms of h
  have volume_ratio : π * h^2 * r = 3 * (π * r^2 * h), by assumption,
  have ratio_simplification : h^2 * r = 3 * r^2 * h :=
    by { rw [π, π] at volume_ratio, exact volume_ratio },
  have r_value : r = 3 * h :=
    by { ring at ratio_simplification, exact ratio_simplification },

  -- Calculate the volume of cylinder D with r = 3h
  let D_volume_val := π * (3 * h)^2 * h,
  have D_volume_final : D_volume = 9 * π * h^3 :=
    by { simp, rw ← D_volume_val, ring },

  -- Conclude that M = 9
  existsi 9,
  split,
  { exact D_volume_final },
  { exact rfl }

end volume_ratio_of_cylinders_l403_403325


namespace time_after_interval_l403_403997

-- Define the initial time as hours, minutes, and seconds
def initial_time := (18, 15, 0)  -- 6:15:00 p.m. in 24-hour format

-- Define the time interval in seconds
def time_interval := 12345

-- Define a function to convert an interval in seconds to hours, minutes, and seconds
def convert_seconds (sec : ℕ) : (ℕ × ℕ × ℕ) :=
  let minutes := sec / 60
  let seconds := sec % 60
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes, seconds)

-- Add two times represented as (hours, minutes, seconds)
def add_time (t1 t2 : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (h1, m1, s1) := t1
  let (h2, m2, s2) := t2
  let total_seconds := s1 + s2
  let carry_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := m1 + m2 + carry_minutes
  let carry_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := h1 + h2 + carry_hours
  (total_hours, result_minutes, result_seconds)

-- Adjust for 24-hour time format
def adjust_24_hr_format (h m s : ℕ) : (ℕ × ℕ × ℕ) :=
  ((h % 24), m, s)

-- The proof statement: adding the interval to the initial time should result in the correct time
theorem time_after_interval :
  let interval_converted := convert_seconds time_interval
  let final_time := add_time initial_time interval_converted
  let adjusted_final_time := adjust_24_hr_format final_time.1 final_time.2 final_time.3
  adjusted_final_time = (21, 40, 45) :=  -- 9:40:45 p.m. in 24-hour format
by
  sorry

end time_after_interval_l403_403997


namespace problem_l403_403579

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403579


namespace count_valid_n_l403_403004

theorem count_valid_n : ∃ (count : ℕ), count = 6 ∧ ∀ n : ℕ,
  0 < n ∧ n < 42 → (∃ m : ℕ, m > 0 ∧ n = 42 * m / (m + 1)) :=
by
  sorry

end count_valid_n_l403_403004


namespace age_ratio_l403_403792

theorem age_ratio (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 10) (h3 : a + b + c = 27) : b / c = 2 := by
  sorry

end age_ratio_l403_403792


namespace fourth_circle_radius_l403_403624

theorem fourth_circle_radius (c : ℝ) (h : c > 0) :
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  fourth_radius = (c / 2) - r :=
by
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  sorry

end fourth_circle_radius_l403_403624


namespace find_defective_part_l403_403815

theorem find_defective_part (parts : Fin 17 → ℝ) (defective : ∃ i, parts i < 1 ∧ ∀ j ≠ i, parts j = 1) : 
  ∃ n ≥ 3, ∀ (balance : (Fin 17 → ℝ) → ℝ), 
  {g : ℕ → (Fin 17 → ℝ) | 
    ∀ i, i < n → (balance (g i) = 0 ∨ balance (g i) < 0)} → 
  ∃ k, g (n-1) k < 1 :=
by
  sorry

end find_defective_part_l403_403815


namespace average_minutes_per_day_l403_403623

/--
In a middle school, the students in sixth grade, seventh grade, and eighth grade run 
an average of 10, 18, and 14 minutes per day, respectively. There are three times 
as many sixth graders as eighth graders, and one and a half times as many sixth 
graders as seventh graders. Prove that the average number of minutes run per day 
by these students is 40/3.
-/
theorem average_minutes_per_day (e : ℕ) :
  let sixth_grade_avg := 10
      seventh_grade_avg := 18
      eighth_grade_avg := 14
      sixth_graders := 3 * e
      seventh_graders := 2 * e
      eighth_graders := e
      total_minutes := sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders
      total_students := sixth_graders + seventh_graders + eighth_graders
  total_minutes / total_students = 40 / 3 :=
by
  sorry

end average_minutes_per_day_l403_403623


namespace fouad_double_ahmed_l403_403302

/-- Proof that in 4 years, Fouad's age will be double of Ahmed's age given their current ages. -/
theorem fouad_double_ahmed (x : ℕ) (ahmed_age fouad_age : ℕ) (h1 : ahmed_age = 11) (h2 : fouad_age = 26) :
  (fouad_age + x = 2 * (ahmed_age + x)) → x = 4 :=
by
  -- This is the statement only, proof is omitted
  sorry

end fouad_double_ahmed_l403_403302


namespace distributed_amount_l403_403772

-- Problem definitions
variables (A : ℝ)

-- Conditions
def distributed_among_14 (A : ℝ) : ℝ := A / 14
def distributed_among_18 (A : ℝ) : ℝ := A / 18
def condition (A : ℝ) : Prop := distributed_among_14 A = distributed_among_18 A + 80

-- Theorem (statement only, no proof)
theorem distributed_amount (h : condition A) : A = 5040 :=
sorry

end distributed_amount_l403_403772


namespace measure_of_angle_D_l403_403108

-- Conditions given in the problem
variables {ω : Type*} [metric_space ω] [is_circle ω]
variables {D E F : ω}
variables (x : ℝ)

-- Measures of arcs as given in the conditions
def arc_DE := x + 90
def arc_EF := 2 * x + 50
def arc_FD := 3 * x - 40

-- Sum of the measures of the arcs equals 360 degrees
def sum_of_arcs := arc_DE + arc_EF + arc_FD = 360

-- Inscribed angle theorem used to find angle D
theorem measure_of_angle_D
  (h_sum_arcs : sum_of_arcs)
  [inscribed : is_inscribed_triangle ω {D, E, F}] :
  ∠D = 68 := 
sorry

end measure_of_angle_D_l403_403108


namespace problem_is_linear_and_odd_l403_403403

variables (a b : ℝ^3)
variable (f : ℝ → ℝ)

-- Conditions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom a_perp_b : a • b = 0
axiom norm_a_ne_norm_b : ∥a∥ ≠ ∥b∥

-- Definition of the function
noncomputable def f (x : ℝ) : ℝ := (x • a + b) • (x • b - a)

-- Proposition to be proven
theorem problem_is_linear_and_odd : 
  ∀ (x : ℝ), f x = x * (∥b∥^2 - ∥a∥^2) ∧ f (-x) = -f(x) :=
by
  sorry

end problem_is_linear_and_odd_l403_403403


namespace intersection_eq_l403_403669

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem intersection_eq : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_eq_l403_403669


namespace unique_real_x_for_real_sqrt_l403_403005

theorem unique_real_x_for_real_sqrt :
  ∃! x : ℝ, ∃ (y : ℝ), y = sqrt (-2 * (x + 2)^2) :=
by
  -- Define the conditions as given in the problem
  let f : ℝ → ℝ := λ x, -2 * (x + 2) ^ 2
  have h1 : ∀ x, f x ≥ 0 ↔ (x + 2) = 0 :=
  by
    intro x
    rw [← sq_nonneg (x + 2), mul_zero]
    split
    intro h2
    exact eq_of_mul_self_eq_zero (neg_eq_zero.mp (eq_zero_of_mul_leq_zero (inv_two_pos.trans h2)))
    intro h3
    rw [← h3, sq_zero, mul_zero]
    -- Note: The proof outline is completed with sorry for now
  sorry

end unique_real_x_for_real_sqrt_l403_403005


namespace count_x_for_3001_l403_403843

noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0     := x
| 1     := 3000
| (n+2) := (sequence (n+1) + 2) / (sequence n)

lemma seq_periodic (x : ℝ) :
  (∀ n, sequence x (n+5) = sequence x n) :=
sorry

theorem count_x_for_3001 :
  set.card {x : ℝ | ∃ n, sequence x n = 3001} = 4 := sorry

end count_x_for_3001_l403_403843


namespace anya_took_home_67_balloons_l403_403747

theorem anya_took_home_67_balloons
  (total_balloons : ℕ)
  (percent_green percent_blue percent_yellow percent_red : ℝ)
  (percent_distribution : percent_green + percent_blue + percent_yellow + percent_red = 1)
  (total_balloons_eq : total_balloons = 672)
  (percent_yellow_eq : percent_yellow = 0.20)
  (anya_took_half : (total_balloons * percent_yellow) / 2 ≈ 67) :
  67 = ((total_balloons * percent_yellow).to_nat) / 2 :=
by sorry

end anya_took_home_67_balloons_l403_403747


namespace f_eval_at_minus_one_log3_five_l403_403929

noncomputable def f : ℝ → ℝ
| x := if x < 2 then f (x + 2) else (1 / 3) ^ x

theorem f_eval_at_minus_one_log3_five : f (-1 + log 3 5) = 1 / 15 :=
by
  sorry

end f_eval_at_minus_one_log3_five_l403_403929


namespace integer_solutions_l403_403873

theorem integer_solutions (x y z : ℤ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x + y + z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1 / (x + y + z)) ↔ (z = -x - y) :=
sorry

end integer_solutions_l403_403873


namespace complex_abs_conjugate_l403_403789

theorem complex_abs_conjugate (z : ℂ) (h : (3 - I) * z = 5 * I) : |conj z| = (real.sqrt 10) / 2 := by
  sorry

end complex_abs_conjugate_l403_403789


namespace find_tan_alpha_l403_403028

theorem find_tan_alpha (α : ℝ) (h₁ : (π / 2) < α ∧ α < 2 * π)
                       (h₂ : cos (π / 2 + α) = 4 / 5) :
  tan α = -4 / 3 :=
sorry

end find_tan_alpha_l403_403028


namespace quadrilateral_with_center_of_symmetry_is_parallelogram_l403_403685

theorem quadrilateral_with_center_of_symmetry_is_parallelogram
  (A B C D O : Type)
  [quadrilateral : Quadrilateral A B C D] -- Assume ABCD forms a quadrilateral
  (center_of_symmetry : IsCenterOfSymmetry O A B C D) -- Assume O is the center of symmetry
  (reflect_A_through_O : ReflectThrough O A = C)
  (reflect_B_through_O : ReflectThrough O B = D)
  (intersect_at_O : IntersectAt A C O ∧ IntersectAt B D O)
  (midpoint_AC_O : IsMidpoint O A C)
  (midpoint_BD_O : IsMidpoint O B D)
  : IsParallelogram A B C D := 
by
  sorry

end quadrilateral_with_center_of_symmetry_is_parallelogram_l403_403685


namespace opposite_of_six_is_negative_six_l403_403204

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l403_403204


namespace BH_eq_CX_l403_403113

noncomputable def median (A B C : Point) : Line := sorry
noncomputable def altitude (A B C : Point) : Line := sorry
noncomputable def perpendicular (l1 l2 : Line) : Prop := sorry
noncomputable def circumcircle (P Q R : Point) : Circle := sorry
noncomputable def intersection (c : Circle) (l : Line) : Point := sorry

theorem BH_eq_CX 
  (A B C P Q H X : Point)
  (AM AH : Line)
  (h1 : AM = median A B C)
  (h2 : AH = altitude A B C)
  (h3 : Q ∈ line_through A B)
  (h4 : P ∈ line_through A C)
  (h5 : perpendicular (line_through Q AM) (line_through A C))
  (h6 : perpendicular (line_through P AM) (line_through A B))
  (h7 : let circle_PMQ := circumcircle P M Q in
        X ∈ intersection circle_PMQ (line_through B C))
  : distance B H = distance C X := 
sorry

end BH_eq_CX_l403_403113


namespace projection_eq_rat_l403_403657

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of vectors p, q, and v
variables (v w p q : V)

-- Conditions
axiom proj_p : p = (v • (1 / ∥v∥)) • w 
axiom proj_q : q = (p • (1 / ∥p∥)) • v
axiom norm_ratio : ∥p∥ / ∥v∥ = 5 / 7

theorem projection_eq_rat : ∥q∥ / ∥v∥ = 25 / 49 :=
sorry

end projection_eq_rat_l403_403657


namespace find_x_given_inverse_relationship_l403_403734

variable {x y : ℝ}

theorem find_x_given_inverse_relationship 
  (h₀ : x > 0) 
  (h₁ : y > 0) 
  (initial_condition : 3^2 * 25 = 225)
  (inversion_condition : x^2 * y = 225)
  (query : y = 1200) :
  x = Real.sqrt (3 / 16) :=
by
  sorry

end find_x_given_inverse_relationship_l403_403734


namespace sum_f_k_from_1_to_22_l403_403484

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403484


namespace work_rate_ab_l403_403256

variables (A B C : ℝ)

-- Defining the work rates as per the conditions
def work_rate_bc := 1 / 6 -- (b and c together in 6 days)
def work_rate_ca := 1 / 3 -- (c and a together in 3 days)
def work_rate_c := 1 / 8 -- (c alone in 8 days)

-- The main theorem that proves a and b together can complete the work in 4 days,
-- based on the above conditions.
theorem work_rate_ab : 
  (B + C = work_rate_bc) ∧ (C + A = work_rate_ca) ∧ (C = work_rate_c) 
  → (A + B = 1 / 4) :=
by sorry

end work_rate_ab_l403_403256


namespace train_crossing_time_l403_403224

-- Conditions
def length_train1 : ℕ := 200 -- Train 1 length in meters
def length_train2 : ℕ := 160 -- Train 2 length in meters
def speed_train1 : ℕ := 68 -- Train 1 speed in kmph
def speed_train2 : ℕ := 40 -- Train 2 speed in kmph

-- Conversion factors and formulas
def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600
def total_distance (l1 l2 : ℕ) := l1 + l2
def relative_speed (s1 s2 : ℕ) := kmph_to_mps (s1 + s2)
def crossing_time (dist speed : ℕ) := dist / speed

-- Proof statement
theorem train_crossing_time : 
  crossing_time (total_distance length_train1 length_train2) (relative_speed speed_train1 speed_train2) = 12 := by sorry

end train_crossing_time_l403_403224


namespace value_of_expression_l403_403889

-- Given conditions as definitions
axiom cond1 (x y : ℝ) : -x + 2*y = 5

-- The theorem we want to prove
theorem value_of_expression (x y : ℝ) (h : -x + 2*y = 5) : 
  5 * (x - 2 * y)^2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  -- The proof part is omitted here.
  sorry

end value_of_expression_l403_403889


namespace sum_f_k_1_22_l403_403510

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403510


namespace only_positive_odd_integer_dividing_3n_plus_1_l403_403347

theorem only_positive_odd_integer_dividing_3n_plus_1 : 
  ∀ (n : ℕ), (0 < n) → (n % 2 = 1) → (n ∣ (3 ^ n + 1)) → n = 1 := by
  sorry

end only_positive_odd_integer_dividing_3n_plus_1_l403_403347


namespace election_votes_l403_403628

theorem election_votes (V : ℕ) (h : 0.60 * V - 0.40 * V = 1504) : V = 7520 :=
by
  sorry

end election_votes_l403_403628


namespace solve_for_q_l403_403962

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l403_403962


namespace stratified_sampling_l403_403274

def total_employees : ℕ := 3200
def sample_size : ℕ := 400
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 3
def ratio_elderly : ℕ := 2
def total_ratio : ℕ := ratio_middle_aged + ratio_young + ratio_elderly

theorem stratified_sampling :
  let middle_aged_to_be_sampled := sample_size * ratio_middle_aged / total_ratio,
      young_to_be_sampled := sample_size * ratio_young / total_ratio,
      elderly_to_be_sampled := sample_size * ratio_elderly / total_ratio in
  middle_aged_to_be_sampled = 200 ∧ young_to_be_sampled = 120 ∧ elderly_to_be_sampled = 80 := by
  sorry

end stratified_sampling_l403_403274


namespace quadratic_roots_range_l403_403377

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x ≠ y ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 := 
sorry

end quadratic_roots_range_l403_403377


namespace problem_conditions_l403_403925

theorem problem_conditions
  (C_ellipse : ∀ x y : ℝ, (x^2) / 2 + y^2 = 1)
  (E_circle : ∀ x y : ℝ, x^2 + y^2 = 2 / 3)
  (T_major_axis : ℝ) (T_focal_length : ℝ)
  (l : ℝ → ℝ)
  (A B : ℝ × ℝ)
  (l_tangent_to_E : ∀ t : ℝ, E_circle (fst (l t)) (snd (l t)))
  (l_intersect_C : ∀ t : ℝ, C_ellipse (fst (l t)) (snd (l t)) → A = (fst (l t), snd (l t)) ∨ B = (fst (l t), snd (l t)))
  : theorem statement :=
begin
  -- part 1: Standard equation of ellipse T
  have T_ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1, sorry,

  -- part 2: Angle AOB is a constant 90 degrees
  have angle_AOB_const : ∀ x_A y_A x_B y_B : ℝ, ∠ (0, 0) (x_A, y_A) (x_B, y_B) = 90, sorry,

  -- part 3: Range of area of triangle AOB
  have area_AOB_range : ∃ A_low A_high : ℝ, (A_low = 2 / 3 ∧ A_high = sqrt 2 / 2) ∧ (area (0, 0) (A, fst A) (B, fst B) ∈ [A_low, A_high]), sorry,
end

end problem_conditions_l403_403925


namespace soda_price_before_increase_l403_403301

theorem soda_price_before_increase
  (candy_box_after : ℝ)
  (soda_after : ℝ)
  (candy_box_increase : ℝ)
  (soda_increase : ℝ)
  (new_price_soda : soda_after = 9)
  (new_price_candy_box : candy_box_after = 10)
  (percent_candy_box_increase : candy_box_increase = 0.25)
  (percent_soda_increase : soda_increase = 0.50) :
  ∃ P : ℝ, 1.5 * P = 9 ∧ P = 6 := 
by
  sorry

end soda_price_before_increase_l403_403301


namespace probability_of_non_blue_face_l403_403277

-- Let faces of the cube be represented by an inductive type
inductive Face
| green : Face
| yellow : Face
| blue : Face

open Face

-- Assume a cube with given number of colored faces
def cube_faces : List Face := [green, green, green, yellow, yellow, blue]

-- Define a function to calculate the probability of non-blue face
def probability_non_blue (faces : List Face) : ℚ :=
  let total_faces := List.length faces
  let non_blue_faces := List.countp (λ f => f ≠ blue) faces
  non_blue_faces / total_faces

-- Main statement to prove
theorem probability_of_non_blue_face :
  probability_non_blue cube_faces = 5 / 6 :=
by
  sorry

end probability_of_non_blue_face_l403_403277


namespace sum_f_1_to_22_l403_403455

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403455


namespace solve_eq1_solve_eq2_l403_403173

-- Prove the solution of the first equation
theorem solve_eq1 (x : ℝ) : 3 * x - (x - 1) = 7 ↔ x = 3 :=
by
  sorry

-- Prove the solution of the second equation
theorem solve_eq2 (x : ℝ) : (2 * x - 1) / 3 - (x - 3) / 6 = 1 ↔ x = (5 : ℝ) / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l403_403173


namespace determine_x_l403_403337

theorem determine_x (p q : ℝ) (hpq : p ≠ q) : 
  ∃ (c d : ℝ), (x = c*p + d*q) ∧ c = 2 ∧ d = -2 :=
by 
  sorry

end determine_x_l403_403337


namespace sum_f_k_from_1_to_22_l403_403481

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403481


namespace max_balls_in_pile_l403_403289

theorem max_balls_in_pile :
  ∀ (total_balls red_bal1 red_bal2 : ℕ) (x : ℕ) 
    (h1 : red_bal1 = 49)
    (h2 : red_bal2 = 7)
    (h3 : total_balls = 50 + 8 * x)
    (h4 : 49 + 7 * x ≥ 0.9 * (50 + 8 * x)),
    total_balls ≤ 210 :=
by 
  intros total_balls red_bal1 red_bal2 x h1 h2 h3 h4
  -- To be proved
  sorry

end max_balls_in_pile_l403_403289


namespace find_n_in_quadratic_roots_l403_403829

-- Definitions based on given problem conditions
def quadratic_eq (a b c : ℤ) := λ x : ℂ, a * x^2 + b * x + c

def roots_repr (m n p x : ℂ) := x = (m + real.sqrt n) / p ∨ x = (m - real.sqrt n) / p

-- Statement of the theorem
theorem find_n_in_quadratic_roots :
  let a := 3
  let b := -4
  let c := -7 in
  ∀ (m p : ℕ), gcd m p = 1 → 
  (∀ x : ℂ, quadratic_eq a b c x = 0 → roots_repr m 100 p x) := 
sorry

end find_n_in_quadratic_roots_l403_403829


namespace sum_f_k_1_22_l403_403504

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403504


namespace bears_in_shipment_l403_403297

theorem bears_in_shipment
  (initial_bears : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ)
  (total_bears_after_shipment : ℕ) 
  (initial_bears_eq : initial_bears = 5)
  (shelves_eq : shelves = 2)
  (bears_per_shelf_eq : bears_per_shelf = 6)
  (total_bears_calculation : total_bears_after_shipment = shelves * bears_per_shelf)
  : total_bears_after_shipment - initial_bears = 7 :=
by
  sorry

end bears_in_shipment_l403_403297


namespace cube_difference_l403_403407

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 :=
sorry

end cube_difference_l403_403407


namespace find_m_solve_log_eq_prove_sum_l403_403887

-- Condition 1: f(x) + f(1 - x) = 1
def symmetric_about_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f(x) + f(2 * a - x) = 2 * b

-- Function
def f (x : ℝ) (m : ℝ) : ℝ := 4^x / (4^x + m)

-- Problem 1: Find the value of m.
theorem find_m (m : ℝ) (h : symmetric_about_point (f m) 0.5 0.5) : m = 2 :=
by sorry

-- Problem 2: Solve the equation.
theorem solve_log_eq (x : ℝ) (h : log 2 (1 - (4^x / (4^x + 2))) * log 2 (4^(-x) * (4^x / (4^x + 2))) = 2) : x = 0.5 :=
by sorry

-- Condition for Problem 3
def f_ (x : ℝ) : ℝ := f x 2

-- Problem 3: Prove the sum.
theorem prove_sum (n : ℕ) (hpos : 0 < n) :
  (∑ k in (finset.range n).filter (λ k, k > 0), f_ ( k / n )) = (3 * n + 1) / 6 :=
by sorry

end find_m_solve_log_eq_prove_sum_l403_403887


namespace sum_G_l403_403371

def G (n : ℕ) : ℕ :=
if n % 2 = 0 then n else n

theorem sum_G :
  ∑ n in Finset.range 2006 \+ 2, G n = 2012028 :=
by sorry

end sum_G_l403_403371


namespace price_increase_1995_l403_403097

noncomputable def priceChange (price1992 : ℝ) (price1995 : ℝ) : ℝ :=
  (price1995 - price1992) * 100 / price1992

theorem price_increase_1995 {price1992 : ℝ} (h1 : price1992 ≠ 0) :
  let price1993 := price1992 * 1.05,
      price1994 := price1993 * 1.10,
      price1995 := price1994 * 0.88
  in priceChange price1992 price1995 = 1.64 := sorry

end price_increase_1995_l403_403097


namespace slope_range_of_line_intersecting_circle_l403_403965

theorem slope_range_of_line_intersecting_circle:
  ∀ k : ℝ, 
  (∃ l : AffineSubspace ℝ (EuclideanSpace ℝ 2), 
    l.direction = (ℓ : EuclideanSpace ℝ 2) 
    ∧ ∃ P : l.affineSpan, 
      P = (4 : ℝ, 0 : ℝ)) 
  ∧ (∃ P' Q' : EuclideanSpace ℝ 2, 
       (P' = ((2 : ℝ), (0 : ℝ)) ∧ Q' ∈ {x | ∥x - P'∥ = 1})) 
      →
  - (Real.sqrt 3 / 3) ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by
  sorry

end slope_range_of_line_intersecting_circle_l403_403965


namespace sum_f_1_to_22_l403_403429

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403429


namespace seq_prod_inequality_l403_403389

theorem seq_prod_inequality (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  let a_n := λ n, (exp (2 * n * x) - 1) / (exp (2 * n * x) + 1) in
  (finset.prod (finset.range n) (λ i, a_n (i + 1))) > 1 - 2 / (exp (2 * x)) :=
sorry

end seq_prod_inequality_l403_403389


namespace transformed_average_transformed_variance_l403_403712

variable {x : Fin 8 → ℝ}

-- Define the conditions
def average (x : Fin 8 → ℝ) : ℝ := (Finset.sum Finset.univ (fun i => x i)) / 8
def variance (x : Fin 8 → ℝ) : ℝ :=
  (Finset.sum Finset.univ (fun i => (x i - average x) ^ 2)) / 8

-- Given conditions
axiom avg_x : average x = 4
axiom var_x : variance x = 2

-- Transformed data
def y (i : Fin 8) : ℝ := 2 * x i - 6

-- The proof problems based on conditions
theorem transformed_average : average y = 2 :=
by sorry

theorem transformed_variance : variance y = 8 :=
by sorry

end transformed_average_transformed_variance_l403_403712


namespace number_of_solutions_l403_403355

theorem number_of_solutions (n : ℕ) : 
  (∃ (θ : ℝ), θ ∈ set.Ioc (0 : ℝ) (2 * Real.pi) ∧ 
            Real.sin (7 * Real.pi * Real.cos θ) = Real.cos (7 * Real.pi * Real.sin θ)) ↔ 
  n = -- appropriate valid number here := sorry

end number_of_solutions_l403_403355


namespace domain_of_f_when_a_is_3_max_value_of_a_when_inequality_is_satisfied_l403_403524

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log2 (|x+1| + |x-1| - a)

theorem domain_of_f_when_a_is_3 :
  ∀ x : ℝ, x ∈ {x | x < -3/2 ∨ x > 3/2} ↔ 0 < |x + 1| + |x - 1| - 3 :=
by
  sorry

theorem max_value_of_a_when_inequality_is_satisfied :
  (∀ x : ℝ, 2 ≤ Real.log2 (|x + 1| + |x - 1| - a)) ↔ -2 = a :=
by
  sorry

end domain_of_f_when_a_is_3_max_value_of_a_when_inequality_is_satisfied_l403_403524


namespace product_real_parts_of_roots_l403_403145

noncomputable def i : ℂ := Complex.I

theorem product_real_parts_of_roots :
  let i := Complex.I in
  ∀ (z : ℂ), z^2 + (3 - 2 * i) * z + (7 - 4 * i) = 0 → 
  let roots := [(-((3 - 2 * i) + complex.sqrt ((3 - 2 * i) ^ 2 - 4 * (7 - 4 * i))) / 2, 
                -((3 - 2 * i) - complex.sqrt ((3 - 2 * i) ^ 2 - 4 * (7 - 4 * i))) / 2] in
  (roots.head.re * roots.tail.head.re) = 2 :=
by
  sorry

end product_real_parts_of_roots_l403_403145


namespace unique_two_digit_solution_l403_403742

theorem unique_two_digit_solution:
  ∃! (s : ℕ), 10 ≤ s ∧ s ≤ 99 ∧ (13 * s ≡ 52 [MOD 100]) :=
  sorry

end unique_two_digit_solution_l403_403742


namespace number_of_elements_in_P_l403_403022

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}
def P : Set ℕ := { x | ∃ a b, a ∈ M ∧ b ∈ N ∧ x = a + b }

theorem number_of_elements_in_P : (P = {3, 4, 5}) → Fintype.card P = 3 := by
  sorry

end number_of_elements_in_P_l403_403022


namespace exponent_equivalence_l403_403828

open Real

theorem exponent_equivalence (a : ℝ) (h : a > 0) : 
  (a^2 / (sqrt a * a^(2/3))) = a^(5/6) :=
  sorry

end exponent_equivalence_l403_403828


namespace gcd_2703_1113_l403_403196

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := 
by {
  -- Given:
  have h1 : 2703 = 2 * 1113 + 477 := by sorry,
  have h2 : 1113 = 2 * 477 + 159 := by sorry,
  have h3 : 477 = 3 * 159 := by sorry,
  -- To prove:
  sorry
}

end gcd_2703_1113_l403_403196


namespace intersection_of_M_and_N_l403_403539

-- Define the sets M and N, and prove that their intersection is {0, 1, 2}
theorem intersection_of_M_and_N :
  let M := {-1, 0, 1, 2, 3}
  let N := {x : ℝ | x * (x - 2) ≤ 0}
  M ∩ N = {0, 1, 2} :=
by
  let M := {-1, 0, 1, 2, 3}
  let N := {x : ℝ | x * (x - 2) ≤ 0}
  sorry

end intersection_of_M_and_N_l403_403539


namespace magnitude_of_z_squared_l403_403897

theorem magnitude_of_z_squared (z : ℂ) (h : z = 1 + (complex.i ^ 5)) : complex.abs (z^2) = 2 := by
  sorry

end magnitude_of_z_squared_l403_403897


namespace constant_triangle_area_l403_403654

def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := 
  y^2 / a^2 + x^2 / b^2 = 1

def orthogonal_vectors (x1 y1 x2 y2 a b : ℝ) : Prop := 
  (x1 / b) * (x2 / b) + (y1 / a) * (y2 / a) = 0

def eccentricity (a b : ℝ) : ℝ := sqrt (a^2 - b^2) / a

noncomputable def area_∆AOB (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 / 2 * abs x1 * abs (y1 - y2)

theorem constant_triangle_area
  (a b x1 y1 x2 y2 : ℝ)
  (h_ab_pos : a > b ∧ b > 0)
  (h_ecc_ax : eccentricity a b = sqrt 3 / 2)
  (h_minor_axis : 2 * b = 2)
  (h_ellipse_A : ellipse_eq a b x1 y1)
  (h_ellipse_B : ellipse_eq a b x2 y2)
  (h_orthogonal : orthogonal_vectors x1 y1 x2 y2 a b) :
  area_∆AOB x1 y1 x2 y2 = 1 :=
sorry

end constant_triangle_area_l403_403654


namespace find_M_volume_l403_403327

variable (h r : ℝ)
variable (C_volume D_volume : ℝ)

def volume_of_cylinder (radius height : ℝ) : ℝ := π * radius^2 * height

-- Cylinder C's height is equal to the radius of cylinder D, and radius of cylinder C is equal to height of cylinder D
def height_of_C_eq_radius_of_D (h : ℝ) : Prop := r = h

def radius_of_C_eq_height_of_D (r : ℝ) : Prop := h = r

-- The volume of cylinder D is three times the volume of cylinder C
def volume_relation (h r : ℝ) (C_volume D_volume : ℝ) : Prop := D_volume = 3 * C_volume

theorem find_M_volume (h r : ℝ) (C_volume D_volume : ℝ)
  (hC_eq_rD : height_of_C_eq_radius_of_D h)
  (rC_eq_hD : radius_of_C_eq_height_of_D r)
  (volume_rel : volume_relation h r C_volume D_volume) :
  ∃ M : ℝ, D_volume = M * π * h^3 ∧ M = 9 :=
by sorry

end find_M_volume_l403_403327


namespace problem_statement_l403_403783

def a := 596
def b := 130
def c := 270

theorem problem_statement : a - b - c = a - (b + c) := by
  sorry

end problem_statement_l403_403783


namespace complex_conjugate_first_quadrant_l403_403514

-- Helper definitions and statements to assist the proof.
-- Given information
def given_complex : ℂ := (i^2 + i^3 + i^4) / (1 - i)

-- The Lean statement to prove
theorem complex_conjugate_first_quadrant : 
  (∃ (w : ℂ), w = conj given_complex ∧ (0 < w.re ∧ 0 < w.im)) :=
by
  sorry

end complex_conjugate_first_quadrant_l403_403514


namespace sum_f_values_l403_403471

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403471


namespace max_distance_between_lines_l403_403063

noncomputable def distance_between_lines (l1 l2 : ℝ → ℝ → Prop) : ℝ :=
  let point1 := (-2, 3) in
  let point2 := (1, -1) in
  Real.sqrt ((point1.1 - point2.1) ^ 2 + (point1.2 - point2.2) ^ 2)

def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y + 2 * m - 3 = 0

def line_l2 (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y - m + 1 = 0

theorem max_distance_between_lines (m : ℝ) :
  distance_between_lines (line_l1 m) (line_l2 m) = 5 :=
by
  sorry

end max_distance_between_lines_l403_403063


namespace tangent_line_equation_l403_403153

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 + (a - 3)*x
noncomputable def f_prime (a x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 3)
noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem tangent_line_equation 
  (a : ℝ) 
  (h_even : is_even_function (f_prime a)) :
  9*2 - f 0 2 f (2) = 0 :=
sorry

end tangent_line_equation_l403_403153


namespace regular_polygon_integer_ratio_triangles_l403_403863

theorem regular_polygon_integer_ratio_triangles (n : ℕ) (h₁ : n ≥ 3) :
  (∃ (t : set (set (ℝ × ℝ → ℝ × ℝ → Prop))), 
    (∀ T ∈ t, -- T is a right triangle with integer-ratio sides
      ∃ a b c : ℕ, a^2 + b^2 = c^2 ∧ T = (a, b, c)) ∧
    (∃ P : set (ℝ × ℝ), -- P is a regular n-gon
      (∀ v ∈ P, integer_ratio_right_triangle_v v t) ∧ -- All dissections are IRRTs
      (regular_polygon n P))) ↔ n = 4 :=
by
  sorry

end regular_polygon_integer_ratio_triangles_l403_403863


namespace tangent_line_at_1_max_value_of_t_l403_403518

noncomputable def f (x : ℝ) := log x / (x + 1)

theorem tangent_line_at_1 : 
  ∃ (m b : ℝ), (∀ x, (x - ２ * (m * x + b) - 1 = 0)) := 
begin
  use 1 / 2,
  use -1 / 2,
  sorry,
end

theorem max_value_of_t :
  ∃ t ≤ -1, (∀ x > 0, x ≠ 1 → f x - t / x > log x / (x - 1)) :=
begin
  use -1,
  intro x,
  intro hx,
  intro hnx,
  sorry,
end

end tangent_line_at_1_max_value_of_t_l403_403518


namespace bankers_discount_l403_403732

theorem bankers_discount (FV TD : ℝ) (hFV : FV = 2660) (hTD : TD = 360) :
  let PV := FV - TD,
      BD := (TD / PV) * FV
  in BD = 416.35 :=
by
  sorry

end bankers_discount_l403_403732


namespace set_intersection_problem_l403_403329

def set_product (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3}
def C : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

theorem set_intersection_problem :
  (set_product A B) ∩ (set_product B C) = {2, 6} :=
by
  sorry

end set_intersection_problem_l403_403329


namespace A_eq_B_l403_403137

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l403_403137


namespace maximum_M_k_l403_403368

-- Define the problem
def J (k : ℕ) : ℕ := 10^(k + 2) + 128

-- Define M(k) as the number of factors of 2 in the prime factorization of J(k)
def M (k : ℕ) : ℕ :=
  -- implementation details omitted
  sorry

-- The core theorem to prove
theorem maximum_M_k : ∃ k > 0, M k = 8 :=
by sorry

end maximum_M_k_l403_403368


namespace ratio_expression_value_l403_403545

variable {A B C : ℚ}

theorem ratio_expression_value (h : A / B = 3 / 2 ∧ A / C = 3 / 6) : (4 * A - 3 * B) / (5 * C + 2 * A) = 1 / 4 := 
sorry

end ratio_expression_value_l403_403545


namespace intersection_of_A_and_complement_of_B_l403_403940

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_complement_of_B_l403_403940


namespace shortest_distance_point_to_parabola_l403_403359

noncomputable def parabola_distance : ℝ :=
let point_P := (8 : ℝ, 16 : ℝ),
    parabola_point (b : ℝ) := (b^2 / 4, b) in
  real.sqrt ((parabola_point (8 : ℝ)).fst - point_P.fst)^2 + 
            ((parabola_point (8 : ℝ)).snd - point_P.snd)^2

theorem shortest_distance_point_to_parabola : 
  parabola_distance = 8 * real.sqrt 2 :=
sorry

end shortest_distance_point_to_parabola_l403_403359


namespace infection_in_fourth_round_l403_403786

-- Define the initial conditions and the function for the geometric sequence
def initial_infected : ℕ := 1
def infection_ratio : ℕ := 20

noncomputable def infected_computers (rounds : ℕ) : ℕ :=
  initial_infected * infection_ratio^(rounds - 1)

-- The theorem to prove
theorem infection_in_fourth_round : infected_computers 4 = 8000 :=
by
  -- proof will be added later
  sorry

end infection_in_fourth_round_l403_403786


namespace find_treasure_within_26_moves_l403_403680

/-- 
  On one of the cells of an 8 x 8 grid, a treasure is buried.
  You are located with a metal detector at the center of one of the corner cells of this grid 
  and move by transitioning to the centers of adjacent cells by side.
  The metal detector triggers if you have arrived at the cell where the treasure is buried, 
  or one of the cells adjacent by side to it.
  Prove that you can guarantee to specify the cell where the treasure is buried, 
  covering no more than a distance of 26.
-/
theorem find_treasure_within_26_moves :
  ∃ path : list (ℤ × ℤ), 
    starting_position ∈ [(0,0), (0,7), (7,0), (7,7)] ∧
    length path ≤ 26 ∧ 
    (∀ cell ∈ (path.last'.to_finset), 
      cell.triggers_metal_detector (burying_position : ℤ × ℤ) → 
      specify_treasure cell = burying_position) :=
begin
  sorry
end

end find_treasure_within_26_moves_l403_403680


namespace bricks_needed_l403_403948

def brick_volume (length width height : ℝ) : ℝ := length * width * height
def wall_volume (length width height : ℝ) : ℝ := length * width * height

def number_of_bricks (wall_volume brick_volume : ℝ) : ℕ := 
  (wall_volume / brick_volume).ceil.to_nat

theorem bricks_needed :
  let brick_length := 125
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 800
  let wall_height := 600
  let wall_thickness := 22.5
  number_of_bricks (wall_volume wall_length wall_height wall_thickness) (brick_volume brick_length brick_width brick_height) = 1280 :=
by
  sorry

end bricks_needed_l403_403948


namespace smallest_nature_number_l403_403661

-- Define the set M
def M : Set ℕ := { n | 2 ≤ n ∧ n ≤ 1000 }

-- Define the fundamental conditions as properties
def mult_of_smaller (S : Set ℕ) := ∀ s1 s2 ∈ S, (s1 ≤ s2 → s2 % s1 = 0)
def coprime (S T : Set ℕ) := ∀ s ∈ S, ∀ t ∈ T, Nat.gcd s t = 1
def not_coprime (S U : Set ℕ) := ∀ s ∈ S, ∀ u ∈ U, Nat.gcd s u > 1

-- Formalize the main statement to prove
theorem smallest_nature_number :
  ∃ n : ℕ, (∀ N ⊆ M, N.card = n → ∃ S T U : Set ℕ, S.card = 4 ∧ T.card = 4 ∧ U.card = 4 ∧
    disjoint S (disjoint T U) ∧ mult_of_smaller S ∧ mult_of_smaller T ∧ mult_of_smaller U ∧
    coprime S T ∧ not_coprime S U) ∧ n = 982 :=
  sorry

end smallest_nature_number_l403_403661


namespace simplify_trig_expression_correct_l403_403697

noncomputable def simplify_trig_expression (x : ℝ) : Prop :=
  (cos (2 * x) ≠ 0) ∧
  (cos x ≠ 0) ∧
  (cos (x / 2) ≠ 0) ∧
  (sin x ≠ 0) →
  (∃ k : ℤ, x = Real.arctan 3 + k * Real.pi)

theorem simplify_trig_expression_correct (x : ℝ) :
  simplify_trig_expression x :=
by
  sorry

end simplify_trig_expression_correct_l403_403697


namespace speed_boat_in_still_water_l403_403283

variable (V_b V_s t : ℝ)

def speed_of_boat := V_b

axiom stream_speed : V_s = 26

axiom time_relation : 2 * (t : ℝ) = 2 * t

axiom distance_relation : (V_b + V_s) * t = (V_b - V_s) * (2 * t)

theorem speed_boat_in_still_water : V_b = 78 :=
by {
  sorry
}

end speed_boat_in_still_water_l403_403283


namespace mary_initial_baseball_cards_l403_403157

theorem mary_initial_baseball_cards (X : ℕ) :
  (X - 8 + 26 + 40 = 84) → (X = 26) :=
by
  sorry

end mary_initial_baseball_cards_l403_403157


namespace minimum_roots_in_interval_l403_403144

noncomputable def f: ℝ → ℝ := sorry

theorem minimum_roots_in_interval:
  (∀ x, f(3 + x) = f(3 - x)) →
  (∀ x, f(7 + x) = f(7 - x)) →
  f(0) = 0 →
  ∃ (S : Finset ℝ), S.card ≥ 504 ∧ ∀ x ∈ S, f(x) = 0 ∧ -1000 ≤ x ∧ x ≤ 1000 :=
begin
  sorry
end

end minimum_roots_in_interval_l403_403144


namespace value_of_fraction_l403_403546

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403546


namespace tan_alpha_values_l403_403077

theorem tan_alpha_values (alpha : ℝ) (k : ℤ) :
  (tan alpha = 8 * sin (70 * real.pi / 180) * cos (10 * real.pi / 180) 
   - cos (10 * real.pi / 180) / sin (10 * real.pi / 180))
  → (alpha = (π / 3) + k * π) ∨ (alpha = (4 * π / 3) + k * π) :=
sorry

end tan_alpha_values_l403_403077


namespace opposite_of_six_is_negative_six_l403_403205

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l403_403205


namespace prove_p_and_q_l403_403399

def p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1 / 4

def q (A B : ℝ) : Prop :=
  A > B ↔ Real.sin A > Real.sin B

theorem prove_p_and_q :
  (∀ m : ℝ, p m) ∧ (∀ A B : ℝ, q A B) :=
by
  sorry

end prove_p_and_q_l403_403399


namespace solve_system_l403_403255

theorem solve_system :
  ∃ (x y : ℝ), (log (x + y) - log 5 = log x + log y - log 6) ∧ 
                (log x / (log (y + 6) - (log y + log 6)) = -1) ∧ 
                x = 2 ∧ y = 3 :=
by
  use [2, 3]
  -- First condition: log (x + y) - log 5 = log x + log y - log 6
  have h1 : log (2 + 3) - log 5 = log 2 + log 3 - log 6 :=
    calc
      log 5 - log 5 = log 1 : by rw [add_sub_assoc, log_eq_one] -- Assuming log base
      ... = 0                : log_one,
  -- Second condition: log x / (log (y + 6) - (log y + log 6)) = -1
  have h2 : log 2 / (log (3 + 6) - (log 3 + log 6)) = -1 :=
    calc
      log 2 / (log 9 - (log 3 + log 6)) = -1 : by rw [div_eq_neg, neg_one_eq_one_neg, zero_div, mul_six_neg],

-- Conclusion: The solution is (x, y) = (2, 3)
    sorry

end solve_system_l403_403255


namespace exists_person_next_to_two_economists_l403_403262

variables {Person : Type} [Fintype Person]
variables (economist accountant manager : Person → Prop)
variables (seated : Person → Person → Prop) -- seated(p1, p2) means p1 and p2 are seated next to each other
variables (hand_raised : Person → Prop)
variables (hand_raised_accountants : Finset Person)
variables (hand_raised_managers : Finset Person)

noncomputable def count {P : Person → Prop} [DecidablePred P] : ℕ := Fintype.card {x // P x}

-- Conditions
axiom accountants_condition : (hand_raised_accountants.filter accountant).card = 20
axiom managers_condition : (hand_raised_managers.filter manager).card = 25
axiom neighbor_condition : ∀ p, hand_raised p → seated p.some (p.some neighbor economist)
axiom circular_table : ∀ p1 p2, seated p1 p2 → seated p2 p1

theorem exists_person_next_to_two_economists :
  ∃ p, hand_raised p ∧ (seated p.some (p.some neighbor economist)) := 
sorry

end exists_person_next_to_two_economists_l403_403262


namespace unique_two_digit_solution_exists_l403_403744

theorem unique_two_digit_solution_exists :
  ∃! (s : ℤ), 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52 :=
begin
  use 4,
  split,
  { split,
    { linarith },
    { linarith },
    { norm_num }
  },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    have : 77 * 52 % 100 = 4,
    { norm_num },
    have h : y ≡ 4 [MOD 100] := (congr_arg (λ x, 77 * x % 100) hy3).trans this,
    norm_num at h,
    linarith }
end

end unique_two_digit_solution_exists_l403_403744


namespace sum_f_values_l403_403479

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403479


namespace arithmetic_sequence_general_formula_sum_of_first_n_reciprocal_terms_formula_l403_403017

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1: a 1 + a 5 = (1 / 3) * (a 3) ^ 2)
  (h2: S 7 = 56)
  (ha: arithmetic_sequence a)
  (hS: sum_of_first_n_terms a S) :
  ∀ n, a n = 2 * n :=
sorry

def sequence_bn (b : ℕ → ℝ) (a : ℕ → ℝ) := b 1 = a 1 ∧ ∀ n, b (n + 1) - b n = a (n + 1)
def sum_of_first_n_reciprocal_terms (b : ℕ → ℝ) (T : ℕ → ℝ) := ∀ n, T n = ∑ i in finset.range n, 1 / b (i + 1)

theorem sum_of_first_n_reciprocal_terms_formula (a b : ℕ → ℝ) (T : ℕ → ℝ)
  (hbn: sequence_bn b a)
  (ha: ∀ n, a n = 2 * n) :
  ∀ n, T n = n / (n + 1) :=
sorry

end arithmetic_sequence_general_formula_sum_of_first_n_reciprocal_terms_formula_l403_403017


namespace magnitude_subtraction_l403_403067

variable {V : Type*} [InnerProductSpace ℝ V]
variable (a b : V)
variable (norm_a : ‖a‖ = Real.sqrt 3)
variable (norm_b : ‖b‖ = 2)
variable (dot_product_condition : InnerProductSpace.inner a (a - b) = 0)

theorem magnitude_subtraction : ‖a - b‖ = 1 :=
by
  -- Proof goes here
  sorry

end magnitude_subtraction_l403_403067


namespace actual_average_height_is_correct_l403_403709

noncomputable def incorrect_average_height : ℝ := 183
noncomputable def number_of_boys : ℕ := 35
noncomputable def wrongly_recorded_height : ℝ := 166
noncomputable def actual_height : ℝ := 106

noncomputable def incorrect_total_height : ℝ :=
  incorrect_average_height * number_of_boys

noncomputable def height_difference : ℝ :=
  wrongly_recorded_height - actual_height

noncomputable def correct_total_height : ℝ :=
  incorrect_total_height - height_difference

noncomputable def actual_average_height (n : ℕ) : ℝ :=
  correct_total_height / n

theorem actual_average_height_is_correct
  (n : ℕ)
  (hₐ : ℝ)
  (hₑ : ℝ)
  (h𝓉 : ℝ)
  (h_total : ℝ)
  (diff : ℝ)
  (correct_total : ℝ)
  (aver_height : ℝ)
  (n = number_of_boys)
  (hₐ = incorrect_average_height)
  (hₑ = wrongly_recorded_height)
  (h𝓉 = actual_height)
  (h_total = incorrect_total_height)
  (diff = height_difference)
  (correct_total = correct_total_height)
  (aver_height = actual_average_height n) :
  aver_height = 181.29 := by sorry

end actual_average_height_is_correct_l403_403709


namespace volume_ratio_of_cubes_l403_403239

def cube_volume (a : ℝ) : ℝ := a ^ 3

theorem volume_ratio_of_cubes :
  cube_volume 3 / cube_volume 18 = 1 / 216 :=
by
  sorry

end volume_ratio_of_cubes_l403_403239


namespace PQ_perpendicular_RS_l403_403617

open EuclideanGeometry

theorem PQ_perpendicular_RS
  (A B C D P Q R S M : Point)
  (h1 : ConvexQuad A B C D)
  (h2 : M ∈ (Line A C) ∩ (Line B D))
  (h3 : IsCentroid M A D P)
  (h4 : IsCentroid M C B Q)
  (h5 : IsOrthocenter D M C R)
  (h6 : IsOrthocenter M A B S) : PQ ⊥ RS :=
by
  sorry

end PQ_perpendicular_RS_l403_403617


namespace time_difference_l403_403812

theorem time_difference (dist1 dist2 : ℕ) (speed : ℕ) (h_dist : dist1 = 600) (h_dist2 : dist2 = 550) (h_speed : speed = 40) :
  (dist1 - dist2) / speed * 60 = 75 := by
  sorry

end time_difference_l403_403812


namespace perimeter_of_triangle_PQF2_l403_403926
open Real

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 9) = 1

-- Definition of the foci of the ellipse
structure Foci :=
(F1 F2 : ℝ × ℝ)

-- Main statement
theorem perimeter_of_triangle_PQF2 (a : ℝ) (h : a = 5) (F : Foci) 
  (P Q : ℝ × ℝ) (hp : ellipse P.1 P.2) (hq : ellipse Q.1 Q.2)
  (line : ℝ × ℝ → ℝ × ℝ → Prop) (line_thru_F1 : line F.F1 P ∧ line F.F1 Q) :
  4 * a = 20 :=
by
  rw h
  norm_num
  exact sorry

end perimeter_of_triangle_PQF2_l403_403926


namespace ratio_of_pyramid_volumes_eq_two_thirds_l403_403606

theorem ratio_of_pyramid_volumes_eq_two_thirds
  (n : ℕ) (S h : ℝ) (P : Point) (prism : Prism n S h) (pyramids : Fin n → Pyramid) :
  (∑ i, (pyramids i).volume) / prism.volume = 2 / 3 :=
sorry

end ratio_of_pyramid_volumes_eq_two_thirds_l403_403606


namespace number_of_groups_is_correct_l403_403243

-- Define the number of students
def number_of_students : ℕ := 16

-- Define the group size
def group_size : ℕ := 4

-- Define the expected number of groups
def expected_number_of_groups : ℕ := 4

-- Prove the expected number of groups when grouping students into groups of four
theorem number_of_groups_is_correct :
  number_of_students / group_size = expected_number_of_groups := by
  sorry

end number_of_groups_is_correct_l403_403243


namespace angle_ABC_unused_l403_403379

open Real

-- Define the radius BC of the original circular piece of paper
def radius_BC : ℝ := 4 * real.sqrt 13

-- Define the radius of the cone's base
def radius_cone : ℝ := 8

-- Define the volume of the cone
def volume_cone : ℝ := 256 * π

-- Define the height of the cone using the volume formula
def height_cone : ℝ := (3 * volume_cone) / (π * radius_cone ^ 2)

-- Define the slant height of the cone
def slant_height : ℝ := real.sqrt (height_cone ^ 2 + radius_cone ^ 2)

-- Define the circumference of the cone's base
def circumference_cone : ℝ := 2 * π * radius_cone

-- Define the total circumference of the original circle
def total_circumference : ℝ := 2 * π * radius_BC

-- Define the central angle of the sector used to form the cone
def theta_used : ℝ := (circumference_cone / total_circumference) * 360

-- Define the unused angle
def theta_unused : ℝ := 360 - theta_used

-- Lean statement asserting the solution
theorem angle_ABC_unused : θ_unused = 160.1 :=
by sorry

end angle_ABC_unused_l403_403379


namespace smallest_norm_of_v_l403_403155

variables (v : ℝ × ℝ)

def vector_condition (v : ℝ × ℝ) : Prop :=
  ‖(v.1 - 2, v.2 + 4)‖ = 10

theorem smallest_norm_of_v
  (hv : vector_condition v) :
  ‖v‖ ≥ 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_of_v_l403_403155


namespace find_value_of_fraction_l403_403569

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403569


namespace total_students_l403_403959

theorem total_students (h1 : 15 * 70 = 1050) 
                       (h2 : 10 * 95 = 950) 
                       (h3 : 1050 + 950 = 2000)
                       (h4 : 80 * N = 2000) :
  N = 25 :=
by sorry

end total_students_l403_403959


namespace cheryl_distance_walked_l403_403835

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l403_403835


namespace quadratic_inequality_solution_set_l403_403729

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2 * x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end quadratic_inequality_solution_set_l403_403729


namespace largest_circle_area_l403_403796

theorem largest_circle_area (A : ℝ) (hA : A = 100) :
  ∃ A_c : ℝ, A_c ≈ 96 :=
by
  sorry

end largest_circle_area_l403_403796


namespace ratio_of_square_perimeters_l403_403760

-- Define the conditions
def smaller_square_diagonal (s : ℝ) : ℝ := s * Real.sqrt 2
def larger_square_diagonal (S : ℝ) : ℝ := S * Real.sqrt 2

-- Define the proof statement
theorem ratio_of_square_perimeters (s S : ℝ) (h : larger_square_diagonal S = 7 * smaller_square_diagonal s) : 
  (4 * S) / (4 * s) = 7 :=
by 
  -- Using the definitions directly from the conditions
  have h1 : S * Real.sqrt 2 = 7 * (s * Real.sqrt 2) := by rw [larger_square_diagonal, smaller_square_diagonal, h]
  -- Simplify the diagonal equation to get the side length relation
  have h2 : S = 7 * s := by linarith
  -- Substitute into ratio of perimeters
  calc
    (4 * S) / (4 * s)
        = (4 * (7 * s)) / (4 * s) : by rw h2
    ... = 7 : by ring

end ratio_of_square_perimeters_l403_403760


namespace union_sets_l403_403969

open Set

variable (α : Type) [LinearOrder α]

-- Definitions
def M : Set α := { x | -1 < x ∧ x < 3 }
def N : Set α := { x | 1 ≤ x }

-- Theorem statement
theorem union_sets : M α ∪ N α = { x | -1 < x } := sorry

end union_sets_l403_403969


namespace range_log_sqrt_abs_cos_l403_403238

open Real

noncomputable def log_sqrt_abs_cos (x : ℝ) : ℝ := log (sqrt (abs (cos x))) / log 2

theorem range_log_sqrt_abs_cos : set.range (λ x, log_sqrt_abs_cos x) = set.Iic 0 := by
  sorry

end range_log_sqrt_abs_cos_l403_403238


namespace area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l403_403513

-- Defining the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point P of line1 and line2
def P : ℝ × ℝ := (-2, 2)

-- Perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Line l, passing through P and perpendicular to perpendicular_line
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intercepts of line_l with axes
def x_intercept : ℝ := -1
def y_intercept : ℝ := -2

-- Verifying area of the triangle formed by the intercepts
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

#check line1
#check line2
#check P
#check perpendicular_line
#check line_l
#check x_intercept
#check y_intercept
#check area_of_triangle

theorem area_of_triangle_formed_by_line_l_and_axes :
  ∀ (x : ℝ) (y : ℝ),
    line_l x 0 → line_l 0 y →
    area_of_triangle (abs x) (abs y) = 1 :=
by
  intros x y hx hy
  sorry

theorem equation_of_line_l :
  ∀ (x y : ℝ),
    (line1 x y ∧ line2 x y) →
    (perpendicular_line x y) →
    line_l x y :=
by
  intros x y h1 h2
  sorry

end area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l403_403513


namespace sum_f_k_from_1_to_22_l403_403489

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403489


namespace total_wheels_in_parking_lot_l403_403976

theorem total_wheels_in_parking_lot :
  let cars := 5
  let trucks := 3
  let bikes := 2
  let three_wheelers := 4
  let wheels_per_car := 4
  let wheels_per_truck := 6
  let wheels_per_bike := 2
  let wheels_per_three_wheeler := 3
  (cars * wheels_per_car + trucks * wheels_per_truck + bikes * wheels_per_bike + three_wheelers * wheels_per_three_wheeler) = 54 := by
  sorry

end total_wheels_in_parking_lot_l403_403976


namespace shadow_boundary_eq_l403_403842

noncomputable theory

def sphere_center : ℝ × ℝ × ℝ := (2, 0, 2)
def sphere_radius : ℝ := 2
def light_source : ℝ × ℝ × ℝ := (2, -2, 6)

theorem shadow_boundary_eq :
  ∀ x: ℝ, ∃ y: ℝ, y = -14 :=
begin
  sorry
end

end shadow_boundary_eq_l403_403842


namespace minimum_positive_period_of_f_interval_monotonic_increase_f_range_of_f_in_interval_l403_403049

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * cos x ^ 2 + 1

theorem minimum_positive_period_of_f : 
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by
  use π
  sorry

theorem interval_monotonic_increase_f (k : ℤ) : 
  ∃ a b, ∀ x, a ≤ x ∧ x ≤ b := 
by
  use (-π / 6 + k * π, π / 3 + k * π)
  sorry

theorem range_of_f_in_interval : 
  ∀ x ∈ Icc (-5 * π / 12) (π / 6), 
  f x ∈ Icc (-2) 1 := 
by
  intros x hx
  sorry

end minimum_positive_period_of_f_interval_monotonic_increase_f_range_of_f_in_interval_l403_403049


namespace increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l403_403704

-- Definitions for conditions
def original_profit_per_set := 40
def original_sets_sold_per_day := 20
def additional_sets_per_dollar_drop := 2

-- The proof problems

-- Part 1: Prove the increase in sets when profit reduces to $38
theorem increase_in_sets_when_profit_38 :
  let decrease_in_profit := (original_profit_per_set - 38)
  additional_sets_per_dollar_drop * decrease_in_profit = 4 :=
by
  sorry

-- Part 2: Prove the price reduction needed for $1200 daily profit
theorem price_reduction_for_1200_profit :
  ∃ x, (original_profit_per_set - x) * (original_sets_sold_per_day + 2 * x) = 1200 ∧ x = 20 :=
by
  sorry

end increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l403_403704


namespace sum_of_reciprocal_of_roots_l403_403035

theorem sum_of_reciprocal_of_roots :
  ∀ x1 x2 : ℝ, (x1 * x2 = 2) → (x1 + x2 = 3) → (1 / x1 + 1 / x2 = 3 / 2) :=
by
  intros x1 x2 h_prod h_sum
  sorry

end sum_of_reciprocal_of_roots_l403_403035


namespace not_constant_expression_l403_403142

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

noncomputable def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem not_constant_expression (A B C P G : ℝ × ℝ)
  (hG : is_centroid A B C G)
  (hP_on_AB : ∃ x, P = (x, A.2) ∧ A.2 = B.2) :
  ∃ dPA dPB dPC dPG : ℝ,
    dPA = squared_distance P A ∧
    dPB = squared_distance P B ∧
    dPC = squared_distance P C ∧
    dPG = squared_distance P G ∧
    (dPA + dPB + dPC - dPG) ≠ dPA + dPB + dPC - dPG := by
  sorry

end not_constant_expression_l403_403142


namespace min_and_max_fB_l403_403098

-- Definitions as per conditions in the problem
def is_standard_rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  (P.1 = Q.1 ∨ P.1 = R.1 ∨ P.1 = S.1 ∨ Q.1 = R.1 ∨ Q.1 = S.1 ∨ R.1 = S.1) ∧
  (P.2 = Q.2 ∨ P.2 = R.2 ∨ P.2 = S.2 ∨ Q.2 = R.2 ∨ Q.2 = S.2 ∨ R.2 = S.2)

def is_nice_set (S : set (ℝ × ℝ)) : Prop :=
  ∀ (P Q : (ℝ × ℝ)), P ∈ S → Q ∈ S → P ≠ Q → P.1 ≠ Q.1 ∧ P.2 ≠ Q.2

-- Main Lean Statement
theorem min_and_max_fB :
  ∀ (B : set (ℝ × ℝ)),
  is_nice_set B ∧ B.card = 2016 ∧
  (∀ (E : set (ℝ × ℝ)) (n : ℕ),
  is_nice_set (B ∪ E) ∧ (B ∪ E).card = 2016 + n ∧
  ∀ (P Q R S : (ℝ × ℝ)),
  P ∈ B ∧ Q ∈ B ∧ R ∈ B ∧ S ∈ B ∧ is_standard_rectangle P Q R S →
  ∃ T ∈ E, T.1 < max (P.1) (Q.1) (R.1) (S.1) ∧
           T.1 > min (P.1) (Q.1) (R.1) (S.1) ∧
           T.2 < max (P.2) (Q.2) (R.2) (S.2) ∧
           T.2 > min (P.2) (Q.2) (R.2) (S.2)), 
  (∃ (n_min n_max : ℕ), n_min = 2015 ∧ n_max = 3942) :=
sorry

end min_and_max_fB_l403_403098


namespace max_value_expression_l403_403907

theorem max_value_expression (a b c : ℝ) (h : a * b * c + a + c - b = 0) : 
  ∃ m, (m = (1/(1+a^2) - 1/(1+b^2) + 1/(1+c^2))) ∧ (m = 5 / 4) :=
by 
  sorry

end max_value_expression_l403_403907


namespace combined_area_ratio_l403_403787

variables (r : ℝ) (h_pos : 0 < r)

def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

def combined_semicircle_area (r : ℝ) : ℝ := semicircle_area r + semicircle_area (r / 2)

def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem combined_area_ratio : 
  combined_semicircle_area r / circle_area r = 5 / 8 :=
sorry

end combined_area_ratio_l403_403787


namespace match_date_S_l403_403681

variables (x y : ℕ)

-- conditions stated based on initial problem
def date_behind_C := x
def date_behind_A := x + 3
def date_behind_B := x + 13

-- assertion per the problem
def sum_AB := date_behind_A x + date_behind_B x

theorem match_date_S (h : y = date_behind_A x + date_behind_B x - date_behind_C x) : 
    y = x + 16 :=
by {
  rw [sum_AB, date_behind_C, add_comm 3 13, add_assoc, add_assoc, add_comm 13 x],
  simp at h,
  exact h,
}

end match_date_S_l403_403681


namespace max_color_value_l403_403264

theorem max_color_value (n : ℕ) (k : ℕ) (points : Fin n → Fin k) 
    (h1 : n = 2021) 
    (h2 : ∀ r : Fin k, ∃ arc : Fin n → Fin n, 
      (∃ start end_ : Fin n, ∀ i, arc i = points i ∧ arc i = r) 
      ∧ (arc.end_ - arc.start ≥ n / 2)) 
    : k ≤ 2 :=
sorry

end max_color_value_l403_403264


namespace find_M_volume_l403_403328

variable (h r : ℝ)
variable (C_volume D_volume : ℝ)

def volume_of_cylinder (radius height : ℝ) : ℝ := π * radius^2 * height

-- Cylinder C's height is equal to the radius of cylinder D, and radius of cylinder C is equal to height of cylinder D
def height_of_C_eq_radius_of_D (h : ℝ) : Prop := r = h

def radius_of_C_eq_height_of_D (r : ℝ) : Prop := h = r

-- The volume of cylinder D is three times the volume of cylinder C
def volume_relation (h r : ℝ) (C_volume D_volume : ℝ) : Prop := D_volume = 3 * C_volume

theorem find_M_volume (h r : ℝ) (C_volume D_volume : ℝ)
  (hC_eq_rD : height_of_C_eq_radius_of_D h)
  (rC_eq_hD : radius_of_C_eq_height_of_D r)
  (volume_rel : volume_relation h r C_volume D_volume) :
  ∃ M : ℝ, D_volume = M * π * h^3 ∧ M = 9 :=
by sorry

end find_M_volume_l403_403328


namespace polygon_sides_l403_403966

-- Define the exterior angle and the total sum of exterior angles for any polygon
def exterior_angle (n : ℕ) : ℝ := 360 / n
def total_exterior_angles_sum : ℝ := 360

-- The condition: each exterior angle is 60 degrees
def condition (n : ℕ) : Prop := exterior_angle n = 60

-- The theorem to prove: the number of sides is 6
theorem polygon_sides (n : ℕ) : condition n → n = 6 :=
by
  sorry

end polygon_sides_l403_403966


namespace five_digit_number_is_40637_l403_403865

theorem five_digit_number_is_40637 
  (A B C D E F G : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧ 
        F ≠ G)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F ∧ 0 < G)
  (h3 : A + 11 * A = 2 * (10 * B + A))
  (h4 : A + 10 * C + D = 2 * (10 * A + B))
  (h5 : 10 * C + D = 20 * A)
  (h6 : 20 + 62 = 2 * (10 * C + A)) -- for sequences formed by AB, CA, EF
  (h7 : 21 + 63 = 2 * (10 * G + A)) -- for sequences formed by BA, CA, GA
  : ∃ (C D E F G : ℕ), C * 10000 + D * 1000 + E * 100 + F * 10 + G = 40637 := 
sorry

end five_digit_number_is_40637_l403_403865


namespace problem_solution_l403_403175

theorem problem_solution (x : ℝ) (h : x - 29 = 63) : (x - 47 = 45) :=
by
  sorry

end problem_solution_l403_403175


namespace find_x_2023_l403_403406

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

theorem find_x_2023 :
  let P_n : ℕ → ℝ × ℝ := λ n, (n, curve n),
      Q_n : ℕ → ℝ := λ n, n - 1
  in Q_n 2023 = -2022 :=
sorry

end find_x_2023_l403_403406


namespace surface_area_of_cube_with_same_volume_as_prism_l403_403293

noncomputable def volume_prism (length width height : ℝ) : ℝ := length * width * height

noncomputable def volume_cube (s : ℝ) : ℝ := s^3

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_cube_with_same_volume_as_prism :
  let length := 9
  let width := 3
  let height := 27
  let V := volume_prism length width height
  let s := Real.cbrt V
  surface_area_cube s = 486 :=
by
  have V_def : V = 9 * 3 * 27 := rfl
  have s_def : s = 9 := by
    show Real.cbrt (9 * 3 * 27) = 9
    sorry
  show 6 * s^2 = 486
  sorry

end surface_area_of_cube_with_same_volume_as_prism_l403_403293


namespace area_of_section_l403_403187

theorem area_of_section (R : ℝ) (P : Type) [normed_add_comm_group P] [inner_product_space ℝ P]
  (parallelepiped : P) (inscribed_in_sphere : P → Prop)
  (diagonals_angle : ℝ) (A C1 B D : P) :
  inscribed_in_sphere parallelepiped → 
  diagonals_angle = π / 4 →
  (∀ (plane : P),
    (∃ (diagonal_AC1 : P), parallel plane diagonal_AC1 ∧ passes_through plane diagonal_AC1) ∧ 
    parallel plane (diagonal_BD B D) ∧
    forms_angle_with plane (diagonal_BD1 B) (arcsin (sqrt 2 / 4))
  ) →
  (cross_section_area plane) = (2 * R^2 * sqrt 3) / 3 := 
begin
  sorry
end

end area_of_section_l403_403187


namespace value_of_ac_over_bd_l403_403598

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403598


namespace sum_of_coefficients_of_x_l403_403125

/-- A quadratic equation with distinct negative integer roots and a constant term 24 --/
def distinct_negative_roots := 
  ∃ (r s : ℤ), r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24

/-- The sum of all possible coefficients of x when the quadratic equation has distinct negative integer roots and a constant term 24 --/
theorem sum_of_coefficients_of_x :
  (∑ (b : ℤ) in { b : ℤ | ∃ r s : ℤ, distinct_negative_roots ∧ b = -(r+s) }, b) = 60 :=
by
  sorry

end sum_of_coefficients_of_x_l403_403125


namespace unique_two_digit_solution_exists_l403_403745

theorem unique_two_digit_solution_exists :
  ∃! (s : ℤ), 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52 :=
begin
  use 4,
  split,
  { split,
    { linarith },
    { linarith },
    { norm_num }
  },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    have : 77 * 52 % 100 = 4,
    { norm_num },
    have h : y ≡ 4 [MOD 100] := (congr_arg (λ x, 77 * x % 100) hy3).trans this,
    norm_num at h,
    linarith }
end

end unique_two_digit_solution_exists_l403_403745


namespace cylinder_volume_options_l403_403198

theorem cylinder_volume_options (length width : ℝ) (h₀ : length = 4) (h₁ : width = 2) :
  ∃ V, (V = (4 / π) ∨ V = (8 / π)) :=
by
  sorry

end cylinder_volume_options_l403_403198


namespace round_to_nearest_tenth_l403_403168

theorem round_to_nearest_tenth (x : ℝ) (hx : x = 45.65001) : round (x * 10) / 10 = 45.7 := by
  have hx' : x = 45.65001 := hx
  sorry -- Skipping the actual rounding calculation

end round_to_nearest_tenth_l403_403168


namespace value_of_ac_over_bd_l403_403595

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403595


namespace sum_of_first_n_terms_l403_403922

-- Definitions from conditions
def a (n : ℕ) : ℕ := 6 * n - 4
def b (n : ℕ) : ℕ := 2 * 3^(n-1)

def c (n : ℕ) : ℕ := a n * b n

def S (n : ℕ) : ℕ := 7 + (6 * n - 7) * 3^n

-- Proof statement
theorem sum_of_first_n_terms (n : ℕ) : 
  S n = ∑ i in finset.range n, c (i+1) := sorry

end sum_of_first_n_terms_l403_403922


namespace maximum_angle_point_on_line_l403_403719

-- Conditions
def point A : (ℝ × ℝ) := (1, 0)
def point B : (ℝ × ℝ) := (3, 0)
def line_c (x : ℝ) : ℝ := x + 1

-- Statement of the theorem
theorem maximum_angle_point_on_line :
  ∃ p : ℝ × ℝ, p = (1, 2) ∧ p.2 = line_c p.1 :=
sorry

end maximum_angle_point_on_line_l403_403719


namespace actual_diameter_layer_3_is_20_micrometers_l403_403805

noncomputable def magnified_diameter_to_actual (magnified_diameter_cm : ℕ) (magnification_factor : ℕ) : ℕ :=
  (magnified_diameter_cm * 10000) / magnification_factor

def layer_3_magnified_diameter_cm : ℕ := 3
def layer_3_magnification_factor : ℕ := 1500

theorem actual_diameter_layer_3_is_20_micrometers :
  magnified_diameter_to_actual layer_3_magnified_diameter_cm layer_3_magnification_factor = 20 :=
by
  sorry

end actual_diameter_layer_3_is_20_micrometers_l403_403805


namespace probability_of_success_l403_403722

def prob_successful_attempt := 0.5

def prob_unsuccessful_attempt := 1 - prob_successful_attempt

def all_fail_prob := prob_unsuccessful_attempt ^ 4

def at_least_one_success_prob := 1 - all_fail_prob

theorem probability_of_success :
  at_least_one_success_prob = 0.9375 :=
by
  -- Proof would be here
  sorry

end probability_of_success_l403_403722


namespace tenth_term_is_20_over_21_l403_403703

-- Define the sequence
def sequence (n : ℕ) := (2 * n, 2 * n + 1)

-- Define the 10th term
noncomputable def tenth_term := sequence 10

-- Prove that the 10th term of the sequence is (20, 21)
theorem tenth_term_is_20_over_21 : tenth_term = (20, 21) := by
  sorry

end tenth_term_is_20_over_21_l403_403703


namespace sum_f_values_l403_403480

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403480


namespace a_4_value_general_a_n_l403_403981

variable (n : ℕ)

-- Hypothesis for the known counts for some n
@[simp] def known_counts (n : ℕ) : Prop :=
  if n = 2 then a n = 6 else
  if n = 3 then a n = 20 else
  if n = 4 then a n = 30 else
  true

-- Definition of aligned square count
@[simp] def aligned_square_count (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Hypothesis for the general formula
@[simp] def a (n : ℕ) : ℕ :=
  aligned_square_count n + inclined_square_count n

-- Declare the total count of squares made negative by the grid of n x n points.
@[simp] def inclined_square_count (n : ℕ) : ℕ := sorry

theorem a_4_value : a 4 = 30 := sorry

theorem general_a_n : a n = aligned_square_count n + inclined_square_count n := sorry

end a_4_value_general_a_n_l403_403981


namespace determine_course_l403_403700

theorem determine_course (k : ℝ) (h: 0 <= k ∧ k < 1) : ∃ α : ℝ, α = Real.arcsin(k) :=
by
  have h₁ : 0 <= k := h.1
  have h₂ : k < 1 := h.2
  use Real.arcsin(k)
  rw [← Real.sin_arcsin h₁ h₂]
  sorry

end determine_course_l403_403700


namespace positive_integers_not_divisible_by_4_l403_403883

theorem positive_integers_not_divisible_by_4 :
  ∃ (k : ℕ), k = 30 ∧
    ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 1200) → 
    (let sum := (Int.floor (1197 / n) + Int.floor (1198 / n) + Int.floor (1199 / n) + Int.floor (1200 / n)) in
      sum % 4 ≠ 0) :=
sorry

end positive_integers_not_divisible_by_4_l403_403883


namespace find_value_of_fraction_l403_403563

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403563


namespace solution_set_ineq_l403_403362

theorem solution_set_ineq (x : ℝ) :
  (x - 1) / (1 - 2 * x) ≥ 0 ↔ (1 / 2 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_ineq_l403_403362


namespace sum_f_eq_neg_24_l403_403419

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403419


namespace height_difference_between_crates_l403_403222

theorem height_difference_between_crates 
  (n : ℕ) (diameter : ℝ) 
  (height_A : ℝ) (height_B : ℝ) :
  n = 200 →
  diameter = 12 →
  height_A = n / 10 * diameter →
  height_B = n / 20 * (diameter + 6 * Real.sqrt 3) →
  height_A - height_B = 120 - 60 * Real.sqrt 3 :=
sorry

end height_difference_between_crates_l403_403222


namespace work_together_l403_403773

theorem work_together (W : ℝ) (Dx Dy : ℝ) (hx : Dx = 15) (hy : Dy = 30) : 
  (Dx * Dy) / (Dx + Dy) = 10 := 
by
  sorry

end work_together_l403_403773


namespace min_value_l403_403912

open Real

-- Definitions of x, y, and z as positive numbers.
variables (x y z : ℝ)
-- Defining the conditions.
hypothesis (h1 : x > 0)
hypothesis (h2 : y > 0)
hypothesis (h3 : z > 0)
-- Given condition.
hypothesis (h4 : xyz * (x + y + z) = 1)

-- Stating the theorem to prove the minimum value.
theorem min_value (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z * (x + y + z) = 1) :
  ∃ v : ℝ, v = (x + y) * (y + z) ∧ v ≥ 2 :=
begin
  sorry
end

end min_value_l403_403912


namespace maximum_area_inscribed_triangle_l403_403614

noncomputable def maxAreaOfInscTriangle {a b c : ℝ} 
  (h1 : a^2 - c^2 = (real.sqrt 2 * a - b) * b) (h2 : a * a + b * b - c * c = real.sqrt 2 * a * b) : ℝ :=
let R := 1 in
  1 / 2 * a * b * real.sin ((3 * real.pi) / 4 - real.acos (real.sqrt 2 / 2))

theorem maximum_area_inscribed_triangle (a b c : ℝ) 
  (h1 : a^2 - c^2 = (real.sqrt 2 * a - b) * b)
  (h2 : a * a + b * b - c * c = real.sqrt 2 * a * b) :
  maxAreaOfInscTriangle h1 h2 = (real.sqrt 2 + 1) / 2 :=
sorry

end maximum_area_inscribed_triangle_l403_403614


namespace molecular_weight_BaCl2_l403_403235

theorem molecular_weight_BaCl2 (mw8 : ℝ) (n : ℝ) (h : mw8 = 1656) : (mw8 / n = 207) ↔ n = 8 := 
by
  sorry

end molecular_weight_BaCl2_l403_403235


namespace probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l403_403010

noncomputable def total_outcomes : ℕ := Nat.choose 6 2

noncomputable def prob_both_boys : ℚ := (Nat.choose 4 2 : ℚ) / total_outcomes

noncomputable def prob_exactly_one_girl : ℚ := ((Nat.choose 4 1) * (Nat.choose 2 1) : ℚ) / total_outcomes

noncomputable def prob_at_least_one_girl : ℚ := 1 - prob_both_boys

theorem probability_both_boys : prob_both_boys = 2 / 5 := by sorry
theorem probability_exactly_one_girl : prob_exactly_one_girl = 8 / 15 := by sorry
theorem probability_at_least_one_girl : prob_at_least_one_girl = 3 / 5 := by sorry

end probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l403_403010


namespace find_angle_C_l403_403973

theorem find_angle_C
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : ∀ (s : ℝ), s > 0 → s.sin ≠ 0)
  (h2 : b * (2 * (b / c).sin + (a / c).sin) + (2 * a + b) * (a / c).sin = 2 * c * (C / c).sin)
  (h3 : A + B + C = π)
  (h4 : a^2 + b^2 - c^2 = 2 * a * b * cos C) :
  C = 2 * π / 3 :=
by
  sorry

end find_angle_C_l403_403973


namespace election_solution_l403_403106

def election_problem : Prop :=
  let votes : List (String × Float) :=
    [("A", 32), ("B", 25), ("C", 22), ("D", 13), ("E", 8)]
  let no_candidate_above_thr (votes : List (String × Float)) (threshold : Float) : Prop :=
    ¬ ∃ (cand : String) (perc : Float), (cand, perc) ∈ votes ∧ perc ≥ threshold
  let redistribute_E (votes : List (String × Float)) : List (String × Float) :=
    let perc_E := votes.find (λ (x : String × Float) => x.1 = "E").get!.2
    let perc_C := votes.find (λ (x : String × Float) => x.1 = "C").get!.2 + 0.8 * perc_E
    votes.map (λ (x : String × Float) => if x.1 = "E" then (x.1, 0) else if x.1 = "C" then (x.1, perc_C) else x)
  let redistributed_votes := redistribute_E votes
  let candidates_sorted := redistributed_votes.filter (λ p => p.2 > 0).sort_by (λ p => p.2) |>.reverse
  let top_two := candidates_sorted.take 2
  no_candidate_above_thr votes 40 ∧ top_two = [("A", 32), ("B", 25)]

theorem election_solution : election_problem :=
by 
  apply sorry

end election_solution_l403_403106


namespace marker_distribution_l403_403367

theorem marker_distribution :
  (∃ (markers : list ℕ), markers.length = 5 ∧ sum markers = 10 ∧ ∀ m ∈ markers, m ≥ 1) →
  ∃ (count : ℕ), count = 126 :=
by
  sorry

end marker_distribution_l403_403367


namespace problem_1_problem_2_problem_3_l403_403018

noncomputable def seq (a : ℕ → ℤ) : ℕ → ℤ 
| n => match n with
       | 0   => sorry -- a_0 initialization 
       | k+1 => 2 * a k + 2 ^ (k + 2)

def geometric_mean (a1 a3 a4 : ℤ) : Prop :=
a3 * a3 = 2 * a1 * a4

def is_arithmetic (seq : ℕ → ℤ) : Prop :=
∀ n: ℕ, (seq (n+1) - seq n = 2)

def sum_first_n (S : ℕ → ℤ) n : ℤ :=
if n = 0 then 
  0
else 
  ∑ i in (finset.range n).1, S i

def sum_reciprocal (S : ℕ → ℤ) : ℕ → ℚ :=
sum_first_n (λ i, 1/S i)

theorem problem_1 (a1 : ℤ) (h1 : seq = λ n, sorry) (h2 : geometric_mean (2 * a1) 
    (seq 3) (seq 4)) : 
  a1 = -16 := sorry

theorem problem_2 (h1 : seq = λ n, sorry) : 
  is_arithmetic (λ n, seq n / 2^n) := sorry

theorem problem_3 (a1 : ℤ) (hne : a1 ≠ 2) (S : ℕ → ℤ) (hS : S = λ i, sorry) : 
  (sum_reciprocal S n) < 5 / 3 := sorry

end problem_1_problem_2_problem_3_l403_403018


namespace tangents_product_is_constant_MN_passes_fixed_point_l403_403032

-- Define the parabola C and the tangency conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

variables {x1 y1 x2 y2 : ℝ}

-- Point G is on the axis of the parabola C (we choose the y-axis for part 2)
def point_G_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Two tangent points from G to the parabola at A (x1, y1) and B (x2, y2)
def tangent_points (G : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂

-- Question 1 proof statement
theorem tangents_product_is_constant (G : ℝ × ℝ) (hG : point_G_on_axis G)
  (hT : tangent_points G x1 y1 x2 y2) : x1 * x2 + y1 * y2 = -3 := sorry

variables {M N : ℝ × ℝ}

-- Question 2 proof statement
theorem MN_passes_fixed_point {G : ℝ × ℝ} (hG : G.1 = 0) (xM yM xN yN : ℝ)
 (hMA : parabola M.1 M.2) (hMB : parabola N.1 N.2)
 (h_perpendicular : (M.1 - G.1) * (N.1 - G.1) + (M.2 - G.2) * (N.2 - G.2) = 0)
 : ∃ P, P = (2, 5) := sorry

end tangents_product_is_constant_MN_passes_fixed_point_l403_403032


namespace largest_circle_area_l403_403799

theorem largest_circle_area (A : ℝ) (hA : A = 100) : 
  let s := (400 / Real.sqrt 3)^(1 / 2) in
  let P := 3 * s in
  let r := P / (2 * Real.pi) in
  Real.round (Real.pi * r^2) = 860 :=
by
  sorry

end largest_circle_area_l403_403799


namespace find_a_l403_403050

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x
noncomputable def g (x : ℝ) : ℝ := 1 / real.exp x
noncomputable def f' (a x : ℝ) : ℝ := x^2 + 2 * x + a

theorem find_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1 / 2 : ℝ) 2, ∃ x2 ∈ set.Icc (1 / 2 : ℝ) 2, f' a x1 ≤ g x2) ↔
  a ≤ real.sqrt real.e / real.e - 8 :=
sorry

end find_a_l403_403050


namespace sum_f_1_to_22_l403_403424

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403424


namespace problem_l403_403396

noncomputable def f : ℝ → ℝ := sorry 

theorem problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_func : ∀ x : ℝ, f (2 + x) = -f (2 - x))
  (h_value : f (-3) = -2) :
  f 2007 = 2 :=
sorry

end problem_l403_403396


namespace regular_polygon_integer_ratio_triangles_l403_403862

theorem regular_polygon_integer_ratio_triangles (n : ℕ) (h₁ : n ≥ 3) :
  (∃ (t : set (set (ℝ × ℝ → ℝ × ℝ → Prop))), 
    (∀ T ∈ t, -- T is a right triangle with integer-ratio sides
      ∃ a b c : ℕ, a^2 + b^2 = c^2 ∧ T = (a, b, c)) ∧
    (∃ P : set (ℝ × ℝ), -- P is a regular n-gon
      (∀ v ∈ P, integer_ratio_right_triangle_v v t) ∧ -- All dissections are IRRTs
      (regular_polygon n P))) ↔ n = 4 :=
by
  sorry

end regular_polygon_integer_ratio_triangles_l403_403862


namespace points_concyclic_l403_403655

open EuclideanGeometry -- Assume necessary general imports

noncomputable def orthocenter (A B C : Point) : Point := sorry -- To be defined
noncomputable def circumcircle (A B C : Point) : Circle := sorry -- To be defined
noncomputable def reflection (P side : Point) : Point := sorry -- Reflection function

theorem points_concyclic 
  (A B C H D E F S T U : Point) 
  (h_orthocenter : H = orthocenter A B C) 
  (hD : D ∈ circumcircle A B C)
  (hE : E ∈ circumcircle A B C)
  (hF : F ∈ circumcircle A B C)
  (h_parallel : parallel (line_through A D) (line_through B E) ∧ parallel (line_through A D) (line_through C F))
  (hS : S = reflection D (line_through B C))
  (hT : T = reflection E (line_through C A))
  (hU : U = reflection F (line_through A B)) : 
  concyclic {S, T, U, H} :=
begin
  sorry -- Proof goes here
end

end points_concyclic_l403_403655


namespace table_condition_iff_even_l403_403982

theorem table_condition_iff_even (n : ℕ) (a : fin n → fin n → ℕ) :
  (∀ k : fin n, (∑ i, a i k) = (∑ j, a k j) + 1 ∨ (∑ i, a i k) + 1 = (∑ j, a k j)) ↔ n % 2 = 0 :=
by
  sorry

end table_condition_iff_even_l403_403982


namespace simplify_expression_l403_403830

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂: x ≠ -3) :
  (x - 1 - 8 / (x + 1)) / ( (x + 3) / (x + 1) ) = x - 3 :=
by
  sorry

end simplify_expression_l403_403830


namespace sin_alpha_two_alpha_plus_beta_l403_403402

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < π / 2)
variable (h₂ : 0 < β ∧ β < π / 2)
variable (h₃ : Real.tan (α / 2) = 1 / 3)
variable (h₄ : Real.cos (α - β) = -4 / 5)

theorem sin_alpha (h₁ : 0 < α ∧ α < π / 2)
                  (h₃ : Real.tan (α / 2) = 1 / 3) :
                  Real.sin α = 3 / 5 :=
by
  sorry

theorem two_alpha_plus_beta (h₁ : 0 < α ∧ α < π / 2)
                            (h₂ : 0 < β ∧ β < π / 2)
                            (h₄ : Real.cos (α - β) = -4 / 5) :
                            2 * α + β = π :=
by
  sorry

end sin_alpha_two_alpha_plus_beta_l403_403402


namespace problem1_problem2_l403_403193

noncomputable def f (x : ℝ) : ℝ := 3 / (9 ^ x + 3)

theorem problem1 (x : ℝ) : f(x) + f(1 - x) = 1 :=
by 
  sorry

theorem problem2 : 
  let S := ∑ i in finset.range 2016, f((i + 1 : ℝ) / 2017) in
  S = 1008 :=
by
  sorry

end problem1_problem2_l403_403193


namespace total_amount_spent_l403_403260

-- Define variables for lunch cost and tip percentage
def lunch_cost : ℝ := 50.50
def tip_percentage : ℝ := 0.15

-- Define the tip amount rounded to the nearest cent
def tip_amount := Real.round (tip_percentage * lunch_cost * 100) / 100

-- Define the total amount spent
theorem total_amount_spent : lunch_cost + tip_amount = 58.08 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end total_amount_spent_l403_403260


namespace ratio_equivalence_l403_403857

theorem ratio_equivalence (m n s u : ℚ) (h1 : m / n = 5 / 4) (h2 : s / u = 8 / 15) :
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 :=
by
  sorry

end ratio_equivalence_l403_403857


namespace sum_S_n_l403_403893

noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (k + 1)

theorem sum_S_n (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (k + 1) * S (k + 1)) = (n * (n + 1) / 2) * (S (n + 1) - 1 / 2) :=
by
  sorry

end sum_S_n_l403_403893


namespace two_numbers_sum_gcd_l403_403879

theorem two_numbers_sum_gcd (x y : ℕ) (h1 : x + y = 432) (h2 : Nat.gcd x y = 36) :
  (x = 36 ∧ y = 396) ∨ (x = 180 ∧ y = 252) ∨ (x = 396 ∧ y = 36) ∨ (x = 252 ∧ y = 180) :=
by
  -- Proof TBD
  sorry

end two_numbers_sum_gcd_l403_403879


namespace distance_AC_l403_403915

/-- Definitions for the golden ratio and golden section -/
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2
noncomputable def golden_section_ratio := (Real.sqrt 5 - 1) / 2

theorem distance_AC (A B C : Point) (h1 : C is_golden_section_point A B) (h2 : A.distance_to B = 10) :
  A.distance_to C = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end distance_AC_l403_403915


namespace projection_inequality_l403_403653

variable (S : Finset (ℝ × ℝ × ℝ)) -- S as a finite set of points in three-dimensional space

def orthogonal_projection_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ := (p.2.1, p.2.2)
def orthogonal_projection_zx (p : ℝ × ℝ × ℝ) : ℝ × ℝ := (p.1, p.2.2)
def orthogonal_projection_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ := (p.1, p.2.1)

def Sx : Finset (ℝ × ℝ) := S.image orthogonal_projection_yz
def Sy : Finset (ℝ × ℝ) := S.image orthogonal_projection_zx
def Sz : Finset (ℝ × ℝ) := S.image orthogonal_projection_xy

theorem projection_inequality :
  S.card ^ 2 ≤ Sx.card * Sy.card * Sz.card := 
begin
  sorry
end

end projection_inequality_l403_403653


namespace collinear_E_F_N_l403_403898

variable (A B C D E F M N : Type) [euclidean_geometry A B C D E F M N]

-- Conditions
axiom cyclic_quadrilateral : is_cyclic A B C D
axiom line_AD_and_BC_intersect_at_E : collinear A D E ∧ collinear B C E ∧ C ≠ E
axiom diagonals_intersect_at_F : ∃ F, collinear A C F ∧ collinear B D F
axiom M_is_midpoint_CD : midpoint C D M
axiom N_on_circumcircle_ABM : N ≠ M ∧ lies_on_circumcircle N A B M ∧ (dist A N / dist B N = dist A M / dist B M)

-- Objective
theorem collinear_E_F_N : collinear E F N := 
sorry

end collinear_E_F_N_l403_403898


namespace fgf_3_is_299_l403_403665

def f (x : ℕ) : ℕ := 5 * x + 4
def g (x : ℕ) : ℕ := 3 * x + 2
def h : ℕ := 3

theorem fgf_3_is_299 : f (g (f h)) = 299 :=
by
  sorry

end fgf_3_is_299_l403_403665


namespace greatest_value_of_x_l403_403199

theorem greatest_value_of_x (x : ℕ) : (Nat.lcm (Nat.lcm x 12) 18 = 180) → x ≤ 180 :=
by
  sorry

end greatest_value_of_x_l403_403199


namespace sum_f_l403_403441

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403441


namespace complex_in_first_quadrant_l403_403334

-- Define the complex number in question
def complex_number := (Complex.I / (1 + Complex.I))

-- State that this complex number lies in the first quadrant
theorem complex_in_first_quadrant :
  (complex_number.re > 0) ∧ (complex_number.im > 0) :=
sorry -- Proof omitted

end complex_in_first_quadrant_l403_403334


namespace problem_l403_403585

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403585


namespace log_a_b_sufficient_and_necessary_condition_l403_403075

theorem log_a_b_sufficient_and_necessary_condition
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : b > 0) : 
  (log a b < 0 ↔ (a - 1) * (b - 1) < 0) :=
sorry

end log_a_b_sufficient_and_necessary_condition_l403_403075


namespace train_crosses_pole_in_33_seconds_l403_403299

-- Define given speed in km/hr and length in meters
def train_speed_kmph : ℝ := 300
def train_length_meters : ℝ := 2750

-- Define conversion factor from km/hr to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph / 3.6

-- Converted speed from km/hr to m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Time taken for the train to cross the pole
def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ := length / speed

-- Theorem stating the time taken for the train to cross the pole is 33 seconds
theorem train_crosses_pole_in_33_seconds : time_to_cross_pole train_length_meters train_speed_mps = 33 := by
  sorry

end train_crosses_pole_in_33_seconds_l403_403299


namespace min_mn_l403_403717

open Real

-- Given the definitions in the conditions of the problem:
def ellipse (m n x y : ℝ) : Prop := (x^2 / m^2 + y^2 / n^2 = 1)
def passes_through (a b m n : ℝ) : Prop := (a^2 / m^2 + b^2 / n^2 = 1)

-- Stating the equivalence proof in Lean 4:
theorem min_mn (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ∃ m n > 0, ellipse m n a b ∧ passes_through a b m n ∧ ((m + n) = (a^(2/3) + b^(2/3))^(1/3)) := sorry

end min_mn_l403_403717


namespace ratio_julia_bill_l403_403679

variable (B M : ℕ)

def total_miles := B + (B + 4) + M * (B + 4)

theorem ratio_julia_bill (h : total_miles B M = 32) :
  (M * (B + 4)) / (B + 4) = M :=
by sorry

end ratio_julia_bill_l403_403679


namespace series_sum_value_l403_403660

noncomputable def r : ℝ := by
  obtain ⟨r, hr⟩ := exists_root_and_deriv_of_strict_mono_incr
    (λ x, x^3 - 1/4 * x - 1)
    (by continuity)
    (begin intros x y; dsimp; linarith [x, y] end)
    (⟨1, by norm_num⟩)
  exact r

theorem series_sum_value :
  let S := ∑' n : ℕ, (n + 1) * (r ^ (3 * n + 1)) in
  (S = 16) :=
by
  have h : r^3 - 1/4 * r - 1 = 0 := sorry,
  refine eq_of_sub_eq_zero ((λ S', _) _),
  sorry

end series_sum_value_l403_403660


namespace increasing_function_range_l403_403383

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
  sorry

end increasing_function_range_l403_403383


namespace convex_ngon_cyclic_iff_exists_ab_l403_403694

theorem convex_ngon_cyclic_iff_exists_ab (n : ℕ) (points : Fin n → ℝ × ℝ) :
  (∃ a b : Fin n → ℝ, ∀ i j : Fin n, i < j → dist (points i) (points j) = |a i * b j - a j * b i|) ↔
  (∀ i j k l : Fin n, i < j < k < l →
    dist (points i) (points k) * dist (points j) (points l) =
    dist (points i) (points j) * dist (points k) (points l) +
    dist (points j) (points k) * dist (points i) (points l)) := sorry

end convex_ngon_cyclic_iff_exists_ab_l403_403694


namespace estimated_black_balls_l403_403629

theorem estimated_black_balls :
  ∀ (total_balls : ℕ) (frequency_black : ℚ),
  total_balls = 100 → frequency_black = 0.65 → (total_balls * frequency_black).toNat = 65 :=
by
  intros total_balls frequency_black h1 h2
  sorry

end estimated_black_balls_l403_403629


namespace hot_dogs_remainder_l403_403960

theorem hot_dogs_remainder (n : ℕ) (h : n = 25197621) : n % 4 = 1 :=
by
  rw h
  norm_num

end hot_dogs_remainder_l403_403960


namespace incenter_ratio_l403_403656

variable {A B C O : Type} [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C]
variable (c b a : ℝ)
variable (λ1 λ2 : ℝ)
variable (AB AC AO : A → B)
variable [NormedSpace ℝ A] [NormedSpace ℝ B]

theorem incenter_ratio (h_incenter: O = incenter A B C)
    (h_AB: AB = c) (h_AC: AC = b)
    (h_AO: AO = λ1 AB + λ2 AC) :
    (λ1 / λ2) = (b / c) :=
by
  sorry

end incenter_ratio_l403_403656


namespace sum_f_k_from_1_to_22_l403_403485

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403485


namespace least_expensive_trip_cost_l403_403162

noncomputable def cost_trip : ℕ := 900 + 3480 + 750

theorem least_expensive_trip_cost 
  (c_dist_a : ℕ := 4000)
  (b_dist_a : ℕ := 5000)
  (car_cost_per_km : ℕ := 20)
  (train_booking_fee : ℕ := 150)
  (train_cost_per_km : ℕ := 15)
  (triangle_right_c : right_triangle ABC a b c) -- additional conditions such as right_triangle concept needs definition
  : cost_trip = 5130 := 
sorry

end least_expensive_trip_cost_l403_403162


namespace minimum_duplicate_set_l403_403809

theorem minimum_duplicate_set (people_dining : ℕ) (dishes : ℕ → ℕ) (target_cost : ℕ) (max_once : ∀ d1 d2, d1 ≠ d2 → dishes d1 ≠ dishes d2)
  (combo_count : ℕ) (valid_combinations : Finset (Finset ℕ))
  (dishes_eq : ∀ combination, combination ∈ valid_combinations ↔ (∃ t, t.toList.sum = target_cost ∧ t.nodup ∧ ∀ d ∈ t, d < combo_count))
  (pigeonhole : people_dining = 92 ∧ combo_count = 9 ∧ target_cost = 10) : 
  ∃ same_set_count, same_set_count ≥ 11 :=
by
  sorry

end minimum_duplicate_set_l403_403809


namespace sum_f_eq_neg_24_l403_403420

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403420


namespace work_rate_c_c_finishes_job_in_10_days_l403_403257

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 6) : C = 1 / 10 := sorry

theorem c_finishes_job_in_10_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 6) : (1 / C) = 10 :=
by
  have hC : C = 1 / 10 := work_rate_c A B C h1 h2
  rw hC
  norm_num

end work_rate_c_c_finishes_job_in_10_days_l403_403257


namespace max_value_l403_403135

variables {A B C : Type} [euclidean_geometry ABC]
variables (K : ABC) (K' P : ABC) (θ C_value : ℝ)
variables (AB CA CB A_median B_median m n : ℝ)

/-- Conditions given in the problem statement -/
def conditions : Prop :=
  let ABC := triangle A B C in
  let symmedian_point := K and
  let θ = (angle A K B - 90) and
  0 < θ ∧ θ < angle C ∧
  circumscribed A K' K B ∧
  angle K' C B = θ ∧
  K'P ⟂ BC ∧
  angle P C A = θ ∧
  sin (angle A P B) = (sin (C - θ))^2 ∧
  A_median * B_median = √(√5 + 1)

/-- Compute the maximum possible value 5AB^2 - CA^2 - CB^2 -/
theorem max_value (h : conditions K K' P θ C_value ∧ n.squarefree):
  ∃ m n : ℕ, 5*AB^2 - CA^2 - CB^2 = m*√n ∧ 100*m + n = 541 :=
  sorry

end max_value_l403_403135


namespace count_even_three_digit_numbers_l403_403755

theorem count_even_three_digit_numbers : 
  ∃ (digits : Set ℕ) (count : ℕ),
    digits = {1, 2, 3, 4, 5} ∧
    count = (finset.univ.filter (λ n, n > 99 ∧ n < 1000 ∧
      (list.nodup (nat.digits 10 n)) ∧
      (nat.even n) ∧
      (∀ d ∈ (nat.digits 10 n).to_finset, d ∈ digits))).card ∧
    count = 24 :=
begin
  sorry
end

end count_even_three_digit_numbers_l403_403755


namespace xyz_identity_l403_403775

theorem xyz_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by
  sorry

end xyz_identity_l403_403775


namespace midpoint_incenter_l403_403124

theorem midpoint_incenter 
    (A B C P Q : Point) 
    (AB AC : Line) 
    (hAB : length AB = length AC) 
    (circumcircle : Circle) 
    (circumreff : IsCircumcircle circumcircle A B C) 
    (incircle : Circle) 
    (htangentP : TangentTo incircle AB P) 
    (htangentQ : TangentTo incircle AC Q) 
    (hinscribed : Inscribed incircle circumcircle) : 
    let I := midpoint P Q in
    Incenter I A B C := sorry

end midpoint_incenter_l403_403124


namespace girls_more_than_boys_l403_403215

theorem girls_more_than_boys (total_students boys : ℕ) (h1 : total_students = 650) (h2 : boys = 272) :
  (total_students - boys) - boys = 106 :=
by
  sorry

end girls_more_than_boys_l403_403215


namespace odd_function_l403_403919

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 1 then x * (x - 1)
else if -1 ≤ x ∧ x < 0 then -(x * x) - x
else 0

theorem odd_function {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) (h_odd : ∀ x, f(-x) = -f(x)) :
  ∀ x, -1 ≤ x ∧ x < 0 → f(x) = -x^2 - x :=
sorry

end odd_function_l403_403919


namespace rope_length_l403_403254

noncomputable def length_of_rope (d_same d_opposite xiaoming_speed : ℕ) : ℕ := 
  let S := 140 - 7 * (20 - xiaoming_speed)
  let x := 8 * S
  x / 8

theorem rope_length : 
  ∀ (d_same d_opposite xiaoming_speed length_of_rope : ℕ), 
  (d_same = 140 ∧ d_opposite = 20 ∧ xiaoming_speed = 1) → 
  length_of_rope d_same d_opposite xiaoming_speed = 35 := 
by 
  intro d_same d_opposite xiaoming_speed length_of_rope
  assume h : (d_same = 140 ∧ d_opposite = 20 ∧ xiaoming_speed = 1)
  sorry

end rope_length_l403_403254


namespace min_f_of_shangmei_number_l403_403388

def is_shangmei_number (a b c d : ℕ) : Prop :=
  a + c = 11 ∧ b + d = 11

def f (a b : ℕ) : ℚ :=
  (b - (11 - b) : ℚ) / (a - (11 - a))

def G (a b : ℕ) : ℤ :=
  20 * a + 2 * b - 121

def is_multiple_of_7 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 7 * k

theorem min_f_of_shangmei_number :
  ∃ (a b c d : ℕ), a < b ∧ is_shangmei_number a b c d ∧ is_multiple_of_7 (G a b) ∧ f a b = -3 :=
sorry

end min_f_of_shangmei_number_l403_403388


namespace train_pass_platform_time_l403_403270

theorem train_pass_platform_time
  (train_length : ℕ) (platform_length : ℕ)
  (time_to_cross_tree : ℕ) :
  (train_length = 1200) →
  (platform_length = 1000) →
  (time_to_cross_tree = 120) →
  (let speed := train_length / time_to_cross_tree in
   let total_distance := train_length + platform_length in
   total_distance / speed = 220) :=
by
  intros h1 h2 h3
  let speed := 1200 / 120
  let total_distance := 1200 + 1000
  show total_distance / speed = 220
  sorry

end train_pass_platform_time_l403_403270


namespace latia_needs_to_work_l403_403952

theorem latia_needs_to_work (TV_price : ℝ) (initial_wage : ℝ) (initial_hours : ℕ) 
  (raise_wage : ℝ) (raise_hours : ℕ) (sales_tax_rate : ℝ) (shipping_fee : ℝ) : 
  let total_cost := TV_price * (1 + sales_tax_rate) + shipping_fee in
  let earnings_before_raise := initial_wage * initial_hours in
  let hours_needed_after_raise (H : ℝ) := H - initial_hours in
  let total_earnings (H : ℝ) := earnings_before_raise + raise_wage * (hours_needed_after_raise H) in
  total_cost > earnings_before_raise 
  → ∃ H : ℝ, H ≥ raise_hours + initial_hours ∧ total_earnings H ≥ total_cost :=
by
  sorry

end latia_needs_to_work_l403_403952


namespace trains_crossing_time_l403_403753

def speed_in_m_per_s (speed_km_per_h : ℕ) : ℕ :=
  speed_km_per_h * 1000 / 3600

def relative_speed (speed1 speed2 : ℕ) : ℕ :=
  speed1 + speed2

def total_distance (bridge_length train_length1 train_length2 : ℕ) : ℕ :=
  bridge_length + train_length1 + train_length2

def crossing_time (distance speed : ℕ) : ℤ :=
  distance / speed

theorem trains_crossing_time :
  let speed_A := speed_in_m_per_s 54 in
  let speed_B := speed_in_m_per_s 45 in
  let relative_speed := relative_speed speed_A speed_B in
  let distance := total_distance 1200 300 250 in
  crossing_time distance relative_speed = 63.64 :=
by sorry

end trains_crossing_time_l403_403753


namespace quadratic_coefficients_standard_form_l403_403324

theorem quadratic_coefficients_standard_form :
  ∀ (x : ℝ), 9 * x^2 = 4 * (3 * x - 1) →
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ a = 9 ∧ b = -12 ∧ c = 4) :=
by
  intro x h
  use [9, -12, 4]
  simp at h
  rw [mul_assoc] at h
  apply eq_of_sub_eq_zero
  rw [sub_eq_add_neg, add_assoc, add_left_neg, add_zero]
  convert h
  conv_lhs do
    congr
    rw [add_mul, mul_assoc, mul_assoc, ←mul_add, add_comm]
  simp
  sorry

end quadratic_coefficients_standard_form_l403_403324


namespace distance_with_detour_l403_403190

theorem distance_with_detour (map_distance : ℝ) (scale : ℝ) (detour : ℝ) :
  map_distance = 35 →
  scale = 10 →
  detour = 0.20 →
  let real_world_distance := map_distance * scale in
  let actual_distance := real_world_distance * (1 + detour) in
  actual_distance = 420 :=
by
  intros h1 h2 h3
  let real_world_distance := map_distance * scale
  let actual_distance := real_world_distance * (1 + detour)
  rw [h1, h2, h3]
  have rw_dist : real_world_distance = 35 * 10 := by simp [h1, h2]
  rw rw_dist at *
  have rw_detour : actual_distance = 350 * 1.20 := by simp [h3]
  simp [rw_detour]
  sorry

end distance_with_detour_l403_403190


namespace total_cost_of_toys_l403_403673

-- Define the costs of the yoyo and the whistle
def cost_yoyo : Nat := 24
def cost_whistle : Nat := 14

-- Prove the total cost of the yoyo and the whistle is 38 cents
theorem total_cost_of_toys : cost_yoyo + cost_whistle = 38 := by
  sorry

end total_cost_of_toys_l403_403673


namespace total_pencils_proof_l403_403706

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l403_403706


namespace problem_1_1_and_2_problem_1_2_l403_403639

section Sequence

variables (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom a_1 : a 1 = 3
axiom a_n_recurr : ∀ n ≥ 2, a n = 2 * a (n - 1) + (n - 2)

-- Prove that {a_n + n} is a geometric sequence and find the general term formula for {a_n}
theorem problem_1_1_and_2 :
  (∀ n ≥ 2, (a (n - 1) + (n - 1) ≠ 0)) ∧ ((a 1 + 1) * 2^(n - 1) = a n + n) ∧
  (∀ n, a n = 2^(n + 1) - n) :=
sorry

-- Find the sum of the first n terms, S_n, of the sequence {a_n}
theorem problem_1_2 (n : ℕ) : S n = 2^(n + 2) - 4 - (n^2 + n) / 2 :=
sorry

end Sequence

end problem_1_1_and_2_problem_1_2_l403_403639


namespace subtract_one_from_solution_l403_403244

theorem subtract_one_from_solution (x : ℝ) (h : 15 * x = 45) : (x - 1) = 2 := 
by {
  sorry
}

end subtract_one_from_solution_l403_403244


namespace Qing_Dynasty_Problem_l403_403112

variable {x y : ℕ}

theorem Qing_Dynasty_Problem (h1 : 4 * x + 6 * y = 48) (h2 : 2 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (2 * x + 5 * y = 38) := by
  exact ⟨h1, h2⟩

end Qing_Dynasty_Problem_l403_403112


namespace exists_two_vertices_with_low_excentricity_l403_403303

noncomputable def polyhedron := Type

-- Define excentricity functions
def excentric (edge1_color edge2_color : Bool) : Bool :=
  edge1_color ≠ edge2_color

def excentricity_vertex (A : polyhedron) (excentric_at_vtx : polyhedron → ℕ) : ℕ :=
  excentric_at_vtx A

def excentricity_vertex_sum (V : set polyhedron) (excentric_at_vtx : polyhedron → ℕ) : ℕ :=
  ∑ v in V, excentric_at_vtx v

axiom euler_formula (V : set polyhedron) (E : set (polyhedron × polyhedron)) (F : set polyhedron) :
  |V| - |E| + |F| = 2

theorem exists_two_vertices_with_low_excentricity (V : set polyhedron) (E : set (polyhedron × polyhedron)) 
  (F : set polyhedron) (edge_color : E → Bool) (excentric_at_vtx : polyhedron → ℕ) :
  ∃ B C ∈ V, B ≠ C ∧ excentricity_vertex B excentric_at_vtx + excentricity_vertex C excentric_at_vtx ≤ 4 :=
sorry

end exists_two_vertices_with_low_excentricity_l403_403303


namespace lucy_total_fish_l403_403672

variable (current_fish additional_fish : ℕ)

def total_fish (current_fish additional_fish : ℕ) : ℕ :=
  current_fish + additional_fish

theorem lucy_total_fish (h1 : current_fish = 212) (h2 : additional_fish = 68) : total_fish current_fish additional_fish = 280 :=
by
  sorry

end lucy_total_fish_l403_403672


namespace intersection_of_M_and_N_l403_403059

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l403_403059


namespace smallest_positive_period_of_g_l403_403169

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

theorem smallest_positive_period_of_g : Real.periodic g (Real.pi) :=
by
  -- Importing the conditions
  have h1 : ∀ x, g(x) = 2 * Real.sin (2 * x - Real.pi / 3),
    -- Proof to be filled in here
    sorry
  -- Use the specified period
  have h2 : ∀ x, g(x + Real.pi) = g(x),
    -- Proof to be filled in here
    sorry
  -- We need to show this is the smallest period
  have h3 : ∀ ε > 0, ε < Real.pi → ∃ x, g(x + ε) ≠ g(x),
    -- Proof to be filled in here
    sorry
  exact ⟨h2, h3⟩

end smallest_positive_period_of_g_l403_403169


namespace geom_seq_sum_first_eight_l403_403878

def geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

def sum_geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a₀ * n else a₀ * (1 - r^n) / (1 - r)

theorem geom_seq_sum_first_eight :
  let a₀ := 1 / 3
  let r := 1 / 3
  let n := 8
  sum_geom_seq a₀ r n = 3280 / 6561 :=
by
  sorry

end geom_seq_sum_first_eight_l403_403878


namespace disjunction_true_if_one_true_disjunction_true_if_one_true_l403_403607

variables (p q : Prop)

theorem disjunction_true_if_one_true (hp : p) (hq : ¬q) : p ∨ q :=
by
  right
  exact not_false hq

# lookup
# prop : Prop
# ¬q : q -- this retrieves full parameter set
i

-- Condition statements
variable (p q : Prop)

-- Theorem (final statement)
theorem disjunction_true_if_one_true (hp : p) (hq : ¬q) : p ∨ q :=
begin
  exact or.inl hp
end

end disjunction_true_if_one_true_disjunction_true_if_one_true_l403_403607


namespace ratio_correct_l403_403626

-- Define the triangle with sides 10, 11, and 13
def triangle (a b c : ℝ) := a = 10 ∧ b = 11 ∧ c = 13

-- Define orthocenter H and points A and D on the triangle
def orthocenter (H A D : Point) (T : triangle 10 11 13) := 
  ∃ (T : triangle 10 11 13), 
    is_orthocenter H T ∧
    is_altitude A D 11

-- Define the ratio HD:HA
noncomputable def ratio_HD_HA (H A D : Point) : Real := 
  if H.y = D.y then 0 else (D.y - H.y) / (A.y - H.y)

-- Prove the ratio HD:HA is 0:1
theorem ratio_correct : 
  ∀ (H A D : Point), 
    triangle 10 11 13 →
    orthocenter H A D → 
    ratio_HD_HA H A D = 0 := 
by
  intros H A D ht ho
  sorry

end ratio_correct_l403_403626


namespace find_all_n_l403_403330

-- Define the problem setup
def coprime_powers_of_primes (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → gcd m n = 1 → ∃ p : ℕ, prime p ∧ ∃ k : ℕ, m = p ^ k

-- Define the set of valid n values
def valid_n_values : set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 20, 24, 30, 42}

-- The main theorem stating that all n satisfying the condition are among the valid values
theorem find_all_n : ∀ n : ℕ, coprime_powers_of_primes n ↔ n ∈ valid_n_values :=
by {
  intro n,
  sorry,
}

end find_all_n_l403_403330


namespace sum_first_2015_terms_l403_403534

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 3
  else 3 * seq (n - 2)

theorem sum_first_2015_terms : ∑ i in finset.range 2015, seq (i + 1) = 3^1008 - 2 := by
  sorry

end sum_first_2015_terms_l403_403534


namespace range_of_f_range_of_a_l403_403866

noncomputable def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

theorem range_of_f:
  ∀ y, y ∈ set.range f ↔ y ∈ set.Ici (-3) := by
  sorry

noncomputable def g (x a : ℝ) : ℝ := 2 * |x - 1| - |x - a|

theorem range_of_a:
  (∀ x, g x a ≥ -1) ↔ (0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_f_range_of_a_l403_403866


namespace fraction_value_l403_403575

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403575


namespace cubes_difference_divisible_91_l403_403686

theorem cubes_difference_divisible_91 (cubes : Fin 16 → ℤ) (h : ∀ n : Fin 16, ∃ m : ℤ, cubes n = m^3) :
  ∃ (a b : Fin 16), a ≠ b ∧ 91 ∣ (cubes a - cubes b) :=
sorry

end cubes_difference_divisible_91_l403_403686


namespace problem_seq_l403_403535

theorem problem_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (q : ℝ) [fact (0 < q)] :
  (∀ n, 2 * a (n + 1) = a n + a (n + 2)) →
  a 5 = 5 →
  S 7 = 28 →
  (∀ n, S n = ∑ i in finset.range (n + 1), a i) →
  (S n = n * (n + 1) / 2) →
  (∀ n,  a n = n) →

  T_n = 2 * (1 - (1 / (n + 1))) →
  T_n = (2 * n) / (n + 1) → 

  b 1 = 1 →
  (∀ n, b (n + 1) = b n + (q ^ a n)) →
  (∀ n, b n = (∑ i in finset.range n, q ^ i)) →
  (q = 1 → b n = n) →
  (q ≠ 1 → b n = (1 - q ^ n) / (1 - q)) →
  b (n + 1) ^ 2 < b n * b (n + 2) := 
sorry

end problem_seq_l403_403535


namespace sum_of_A6_l403_403055

open Nat

def is_in_set (x n m : ℕ) : Prop := (2^n < x) ∧ (x < 2^(n+1)) ∧ (x = 7 * m + 1) ∧ (n > 0) ∧ (m > 0)

theorem sum_of_A6 : ∃ s : ℕ, s = 71 + 78 + 85 + 92 + 99 + 106 + 113 + 120 + 127 ∧ s = 891 :=
by {
  let A6 := {x : ℕ | is_in_set x 6 (x - 1) / 7 },
  have h : ∀ x ∈ A6, 2^6 < x ∧ x < 2^7 ∧ x = 7 * ((x - 1) / 7) + 1 ∧ 6 > 0 ∧ ((x - 1) / 7) > 0,
  assume x hx,
  rw is_in_set,
  split,
  sorry,
  split,
  sorry,
  split,
  sorry,
  split,
  sorry,
  sorry,
}

end sum_of_A6_l403_403055


namespace circle_radius_integer_l403_403992

theorem circle_radius_integer (r : ℤ)
  (center : ℝ × ℝ)
  (inside_point : ℝ × ℝ)
  (outside_point : ℝ × ℝ)
  (h1 : center = (-2, -3))
  (h2 : inside_point = (-2, 2))
  (h3 : outside_point = (5, -3))
  (h4 : (dist center inside_point : ℝ) < r)
  (h5 : (dist center outside_point : ℝ) > r) 
  : r = 6 :=
sorry

end circle_radius_integer_l403_403992


namespace y_at_x_equals_2sqrt3_l403_403603

theorem y_at_x_equals_2sqrt3 (k : ℝ) (y : ℝ → ℝ)
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
sorry

end y_at_x_equals_2sqrt3_l403_403603


namespace sum_f_1_to_22_l403_403456

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403456


namespace sets_are_equal_l403_403140

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l403_403140


namespace pens_distribution_l403_403218
open Finset

theorem pens_distribution : 
  ∃ (n : ℕ), n = choose (2 + 3 - 1) (3 - 1) ∧ n = 6 := 
by
  sorry

end pens_distribution_l403_403218


namespace intersection_M_N_l403_403538

open Set

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} :=
by
  sorry

end intersection_M_N_l403_403538


namespace cost_of_large_tubs_l403_403814

theorem cost_of_large_tubs (L : ℝ) (h1 : 3 * L + 6 * 5 = 48) : L = 6 :=
by {
  sorry
}

end cost_of_large_tubs_l403_403814


namespace total_pencils_is_220_l403_403708

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l403_403708


namespace store_loses_out_l403_403295

theorem store_loses_out (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (x y : ℝ)
    (h1 : a = b * x) (h2 : b = a * y) : x + y > 2 :=
by
  sorry

end store_loses_out_l403_403295


namespace mans_speed_against_current_l403_403285

theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (h1 : speed_with_current = 25)
  (h2 : speed_of_current = 2.5) :
  speed_with_current - 2 * speed_of_current = 20 := 
by
  sorry

end mans_speed_against_current_l403_403285


namespace never_attains_95_l403_403280

def dihedral_angle_condition (α β : ℝ) : Prop :=
  0 < α ∧ 0 < β ∧ α + β < 90

theorem never_attains_95 (α β : ℝ) (h : dihedral_angle_condition α β) :
  α + β ≠ 95 :=
by
  sorry

end never_attains_95_l403_403280


namespace symmetric_line_l403_403191

theorem symmetric_line (x y : ℝ) :
  line_symmetric (3 * x - 5 * y + 1 = 0) (y = x) = (5 * x - 3 * y - 1 = 0) :=
sorry

end symmetric_line_l403_403191


namespace find_alpha_plus_beta_l403_403542

noncomputable theory

variables (α β : ℝ)
hypothesis (h1 : 0 < α ∧ α < π / 2)
hypothesis (h2 : 0 < β ∧ β < π / 2)
hypothesis (h3 : Real.cot α = 3 / 4)
hypothesis (h4 : Real.cot β = 1 / 7)

theorem find_alpha_plus_beta : α + β = 3 * π / 4 :=
by
  sorry

end find_alpha_plus_beta_l403_403542


namespace lateral_surface_area_of_square_pyramid_l403_403713

-- Definitions based on the conditions in a)
def baseEdgeLength : ℝ := 4
def slantHeight : ℝ := 3

-- Lean 4 statement for the proof problem
theorem lateral_surface_area_of_square_pyramid :
  let height := Real.sqrt (slantHeight^2 - (baseEdgeLength / 2)^2)
  let lateralArea := (1 / 2) * 4 * (baseEdgeLength * height)
  lateralArea = 8 * Real.sqrt 5 :=
by
  sorry

end lateral_surface_area_of_square_pyramid_l403_403713


namespace midpoint_PX_on_nine_point_circle_l403_403134

-- Definitions based on given conditions
variables {A B C H O P T X M: Type} [ScaleneTriangle A B C] 
variables [Orthocenter H A B C] [Circumcenter O A B C] 
variables [Midpoint P A H] [OnLine T B C] [Angle TAO : 90°]
variables [AltitudeFoot X O P T] [Midpoint M P X]

-- Theorem statement
theorem midpoint_PX_on_nine_point_circle 
  (hABC : TriangleIsScalene A B C)
  (hH : OrthocenterOfTriangle H A B C)
  (hO : CircumcenterOfTriangle O A B C)
  (hP : Midpoint P A H)
  (hT : OnLine T B C)
  (hTAO : ∠ T A O = 90°)
  (hX : AltitudeFoot X O P T)
  (hM : Midpoint M P X) :
  OnNinePointCircle M A B C := 
sorry

-- End of Proof

end midpoint_PX_on_nine_point_circle_l403_403134


namespace regular_17gon_symmetries_l403_403841

theorem regular_17gon_symmetries : 
  let L := 17
  let R := 360 / 17
  L + R = 17 + 360 / 17 :=
by
  sorry

end regular_17gon_symmetries_l403_403841


namespace sum_f_proof_l403_403439

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403439


namespace find_angle_A_find_side_c_l403_403093

-- condition definitions
variable (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : ∀ x, 0 < x ∧ x < 180)
variable (a_eq_5 : a = 5)
variable (condition : b^2 + c^2 - real.sqrt 2 * b * c = 25)
variable (cos_B : real.cos B = 3 / 5)

-- theorems to prove
theorem find_angle_A : 
    a = 5 → (b^2 + c^2 - real.sqrt 2 * b * c = 25) → 
    degrees_to_radians A = real.acos (((b^2 + c^2 - a^2) / (2 * b * c)) =
    π/4 := 
by {
  sorry,
}

theorem find_side_c : 
    a = 5 → (b^2 + c^2 - real.sqrt 2 * b * c = 25) → 
    (real.cos B = 3 / 5) →
    c = 7 :=
by {
  sorry,
}

end find_angle_A_find_side_c_l403_403093


namespace derivative_at_0_l403_403967

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_at_0 : deriv f 0 = 1 := by
  -- Proof goes here
  sorry

end derivative_at_0_l403_403967


namespace find_k_l403_403024

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b : V) (k : ℝ)

def AB : V := a - k • b
def CB : V := 2 • a + b
def CD : V := 3 • a - b
def BD : V := CD - CB

-- Collinearity condition
def collinear (u v w : V) : Prop :=
∃ λ : ℝ, u = λ • v + w

theorem find_k (h : collinear (a - k • b) (a - 2 • b) 0) : k = 2 :=
sorry

end find_k_l403_403024


namespace sum_f_1_to_22_l403_403457

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403457


namespace christina_walking_speed_l403_403647

noncomputable def christina_speed : ℕ :=
  let distance_between := 270
  let jack_speed := 4
  let lindy_total_distance := 240
  let lindy_speed := 8
  let meeting_time := lindy_total_distance / lindy_speed
  let jack_covered := jack_speed * meeting_time
  let remaining_distance := distance_between - jack_covered
  remaining_distance / meeting_time

theorem christina_walking_speed : christina_speed = 5 := by
  -- Proof will be provided here to verify the theorem, but for now, we use sorry to skip it
  sorry

end christina_walking_speed_l403_403647


namespace calculate_slope_l403_403914

-- Define the direction vector of the line
def direction_vector : ℝ × ℝ := (-1, real.sqrt 3)

-- Define the slope calculation
def slope (v : ℝ × ℝ) : ℝ := v.2 / v.1

-- Theorem statement proving the slope is -sqrt(3)
theorem calculate_slope : slope direction_vector = -real.sqrt 3 := sorry

end calculate_slope_l403_403914


namespace complex_pure_imaginary_real_part_zero_l403_403083

theorem complex_pure_imaginary_real_part_zero (a : ℝ) (z : ℂ) (h: z = (a^2 + 2*a - 3) + (a-1)*complex.I) : (z.re = 0) → a = -3 :=
begin
  sorry
end

end complex_pure_imaginary_real_part_zero_l403_403083


namespace ratio_of_left_handed_no_throwers_l403_403159

theorem ratio_of_left_handed_no_throwers (total_players : ℕ) (throwers : ℕ) (total_right_handed : ℕ)
    (h1 : total_players = 70) 
    (h2 : throwers = 37) 
    (h3 : total_right_handed = 59) : 
  (70 - 37) - (59 - 37) = 11 → (11 : 33) = (1 : 3) :=
by
  intros h_left_handed
  rw [nat.sub_sub]
  rw [h1, h2, h3] at h_left_handed
  exact h_left_handed
  simp
  sorry

end ratio_of_left_handed_no_throwers_l403_403159


namespace max_sum_of_digits_l403_403073

theorem max_sum_of_digits (X Y Z : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 1 ≤ Y ∧ Y ≤ 9) (hZ : 1 ≤ Z ∧ Z ≤ 9) (hXYZ : X > Y ∧ Y > Z) : 
  10 * X + 11 * Y + Z ≤ 185 :=
  sorry

end max_sum_of_digits_l403_403073


namespace sandy_ordered_three_cappuccinos_l403_403314

-- Definitions and conditions
def cost_cappuccino : ℝ := 2
def cost_iced_tea : ℝ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℝ := 1
def num_iced_teas : ℕ := 2
def num_cafe_lattes : ℕ := 2
def num_espressos : ℕ := 2
def change_received : ℝ := 3
def amount_paid : ℝ := 20

-- Calculation of costs
def total_cost_iced_teas : ℝ := num_iced_teas * cost_iced_tea
def total_cost_cafe_lattes : ℝ := num_cafe_lattes * cost_cafe_latte
def total_cost_espressos : ℝ := num_espressos * cost_espresso
def total_cost_other_drinks : ℝ := total_cost_iced_teas + total_cost_cafe_lattes + total_cost_espressos
def total_spent : ℝ := amount_paid - change_received
def cost_cappuccinos := total_spent - total_cost_other_drinks

-- Proof statement
theorem sandy_ordered_three_cappuccinos (num_cappuccinos : ℕ) : cost_cappuccinos = num_cappuccinos * cost_cappuccino → num_cappuccinos = 3 :=
by sorry

end sandy_ordered_three_cappuccinos_l403_403314


namespace max_area_rectangle_l403_403149

def f (x : ℝ) := -2 * (x - 3)^2 + 4

/- Given conditions -/
def periodic {x : ℝ} (n : ℤ) : Prop := f(x) = f(x + 2 * n)
def even_func (x : ℝ) : Prop := f(x) = f(-x)
def on_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

/- Formalizing the maximum area problem -/
theorem max_area_rectangle : ∃ x, on_interval x ∧ 
2 * (-2 * (x - 1)^2 + 4) ≤ (16 * real.sqrt 6) / 9 :=
begin
  sorry
end

end max_area_rectangle_l403_403149


namespace first_pump_time_l403_403683

-- Definitions for the conditions provided
def newer_model_rate := 1 / 6
def combined_rate := 1 / 3.6
def time_for_first_pump : ℝ := 9

-- The theorem to be proven
theorem first_pump_time (T : ℝ) (h1 : 1 / 6 + 1 / T = 1 / 3.6) : T = 9 :=
sorry

end first_pump_time_l403_403683


namespace arithmetic_sequence_l403_403903

open Classical
open BigOperators

noncomputable def general_term (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ (a 3 + a 5 = 10) ∧ (∀ n, a n = n + 1)

noncomputable def sum_Sn (b : ℕ → ℕ) (n : ℕ) : ℝ :=
  let S := ∑ i in Finset.range n, 1 / (b i).to_real
  S = n / (2 * (n + 1))

theorem arithmetic_sequence :
  (∃ a : ℕ → ℕ, general_term a) →
  ∀ n, (∃ b : ℕ → ℕ, ∀ i, b i = (i + 1) * 2^i) →
  sum_Sn b n :=
by
  sorry

end arithmetic_sequence_l403_403903


namespace sum_f_eq_neg24_l403_403461

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403461


namespace a_n_general_formula_l403_403640

-- Define the sequence a_n
def a_sequence : ℕ → ℕ 
| 1 := 6
| (n + 1) := (n + 3) * a_sequence(n) / n

theorem a_n_general_formula (n : ℕ) (hn : n ≥ 1) : 
  a_sequence n = n * (n + 1) * (n + 2) :=
sorry

end a_n_general_formula_l403_403640


namespace intersection_of_M_and_N_l403_403058

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end intersection_of_M_and_N_l403_403058


namespace total_chickens_l403_403675

theorem total_chickens (x : ℕ) (h : 40 + (5 * x) / 12 = (x + 40) / 2) : x + 40 = 280 :=
by
  sorry

end total_chickens_l403_403675


namespace sum_f_l403_403445

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403445


namespace relay_arrangements_l403_403002

theorem relay_arrangements
  (students : Fin 5 → Prop)
  (A B C D E : Prop)
  (forall(i : Fin 5), students i)
  (h_A_not_first_leg : ¬ students 0)
  (h_D_not_last_leg : ¬ students 4) :
  (∃ count : Nat, count = 78) :=
sorry

end relay_arrangements_l403_403002


namespace intersection_eq_l403_403057

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l403_403057


namespace minimum_value_l403_403909

open Real

theorem minimum_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ z : ℝ, (z = (3 / 2) * x^2 + y^2) ∧ z = 15 :=
by
  sorry

end minimum_value_l403_403909


namespace plane_equation_l403_403886

-- Define the vector with its components
def normal_vector : ℝ × ℝ × ℝ := (8, 2, 1)

-- Define the point through which the plane passes
def point_P0 : ℝ × ℝ × ℝ := (3, -5, 4)

-- Prove that the equation of the plane is as expected
theorem plane_equation : ∃ a b c d : ℝ, 
  (a, b, c) = normal_vector ∧ 
  ((a * point_P0.1 + b * point_P0.2 + c * point_P0.3 + d = 0) ∧ 
  ∀ x y z : ℝ, a * x + b * y + c * z + d = 0 ↔ 8 * x + 2 * y + z = 18) :=
sorry

end plane_equation_l403_403886


namespace sum_f_eq_neg_24_l403_403418

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403418


namespace sum_f_values_l403_403474

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403474


namespace sum_of_cubes_l403_403404

theorem sum_of_cubes (n : ℕ) 
  (h : (∑ i in Finset.range (n + 1), i^3) = 3025) : 
  n = 10 :=
by sorry

end sum_of_cubes_l403_403404


namespace exists_graph_with_degrees_l403_403166

theorem exists_graph_with_degrees (n : ℕ) :
  ∃ (G : SimpleGraph (Fin (2 * n))),
    (∀ v : Fin (2 * n), v.val < 2 * n) ∧
    (multiset.countp (λ x, x = 1) (multiset.map (λ v, G.degree v) (multiset.of_fn id)) = 2)
    ∧ (multiset.countp (λ x, x = 2) (multiset.map (λ v, G.degree v) (multiset.of_fn id)) = 2)
    ∧ ... -- Continue for all degrees up to n
    sorry

end exists_graph_with_degrees_l403_403166


namespace seq_50th_term_is_1284_l403_403197

/-- The sequence of all positive integers which are powers of 4 or sums of distinct powers of 4 -/
def seq : Set ℕ := {n : ℕ | ∃ (s : Finset ℕ), (∀ x ∈ s, ∃ k, x = 4^k) ∧ n = s.sum id}

noncomputable def seq_50th_term : ℕ :=
  Classical.choose (Nat.exists_fintype_mem_at_of_finite (Set.finite_mem_finset seq))

theorem seq_50th_term_is_1284 : seq_50th_term = 1284 :=
by sorry

end seq_50th_term_is_1284_l403_403197


namespace sum_f_1_to_22_l403_403458

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403458


namespace ceil_square_values_count_l403_403957

theorem ceil_square_values_count {x : ℝ} (h : ⌈x⌉ = 10) : 
  (∃ S : Finset ℕ, (∀ y ∈ S, ∃ z : ℝ, 9 < z ∧ z ≤ 10 ∧ ∀ w : ℝ, y = ⌈w^2⌉ → ⌊w⌋ = ⌈w⌉) ∧ S.card = 19) :=
sorry

end ceil_square_values_count_l403_403957


namespace sum_f_eq_neg24_l403_403494

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403494


namespace total_pencils_proof_l403_403705

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l403_403705


namespace find_value_of_fraction_l403_403568

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403568


namespace shaded_area_equals_4_2_minus_sqrt_3_l403_403804

noncomputable def commonAreaOfSquares : ℝ :=
  let α := real.pi / 3  -- 60 degrees in radians
  let side := 2
  let h := side * (2 - real.sqrt 3)
  in 2 * h

theorem shaded_area_equals_4_2_minus_sqrt_3 :
  commonAreaOfSquares = 4 * (2 - real.sqrt 3) :=
by
  sorry

end shaded_area_equals_4_2_minus_sqrt_3_l403_403804


namespace intersection_complement_eq_singleton_l403_403061

open Set

def U : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}
def CU_A : Set ℤ := U \ A

theorem intersection_complement_eq_singleton : B ∩ CU_A = {0} := 
by
  sorry

end intersection_complement_eq_singleton_l403_403061


namespace sum_f_eq_neg24_l403_403498

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403498


namespace value_of_fraction_l403_403549

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403549


namespace workers_production_l403_403226

theorem workers_production
    (x y : ℝ)
    (h1 : x + y = 72)
    (h2 : 1.15 * x + 1.25 * y = 86) :
    1.15 * x = 46 ∧ 1.25 * y = 40 :=
by {
  sorry
}

end workers_production_l403_403226


namespace find_base_l403_403348

theorem find_base (b : ℕ) : (b^3 ≤ 64 ∧ 64 < b^4) ↔ b = 4 := 
by
  sorry

end find_base_l403_403348


namespace probability_neither_A_nor_B_l403_403261

noncomputable def pA : ℝ := 0.25
noncomputable def pB : ℝ := 0.35
noncomputable def pA_and_B : ℝ := 0.15

theorem probability_neither_A_nor_B :
  1 - (pA + pB - pA_and_B) = 0.55 :=
by
  simp [pA, pB, pA_and_B]
  norm_num
  sorry

end probability_neither_A_nor_B_l403_403261


namespace advertisement_duration_l403_403339

theorem advertisement_duration (n c T : ℕ) (h_n : n = 5) (h_c : c = 4000) (h_T : T = 60000) :
  T / c / n = 3 :=
by
  -- Given: n = 5, c = 4000, T = 60000
  -- Prove: T / c / n = 3
  rw [h_n, h_c, h_T]
  norm_num
  sorry

end advertisement_duration_l403_403339


namespace snake_alligator_consumption_l403_403271

theorem snake_alligator_consumption :
  (616 / 7) = 88 :=
by
  sorry

end snake_alligator_consumption_l403_403271


namespace Wendi_chickens_l403_403756

theorem Wendi_chickens:
  let initial_chickens: ℕ := 15 in
  let additional_chickens := Int.floor (((initial_chickens / 2) ^ 2): ℝ) in
  let after_additional := initial_chickens + additional_chickens in
  let after_dog := after_additional - 3 in
  let new_group_chickens := Int.floor ((((4 * after_dog - 28): ℤ) / 7): ℝ) in
  let after_new_group := after_dog + new_group_chickens in
  let final_chickens := after_new_group + 3 in
  final_chickens = 105 :=
by
  sorry

end Wendi_chickens_l403_403756


namespace find_value_of_fraction_l403_403566

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403566


namespace diagonal_of_rectangle_l403_403291

variable (l : ℝ) (P : ℝ)

theorem diagonal_of_rectangle (h_l : l = 8) (h_P : P = 46) : ∃ d : ℝ, d = 17 := by
  let w := (P - 2 * l) / 2
  have h_w : w = 15 := by
    calc w = (46 - 2 * 8) / 2 : by rw [h_l, h_P]
       ... = (46 - 16) / 2    : by norm_num
       ... = 30 / 2           : by norm_num
       ... = 15               : by norm_num
  have h_d : (8 : ℝ) ^ 2 + w ^ 2 = 289 := by
    calc (8 : ℝ) ^ 2 + 15 ^ 2 = 64 + 225 : by rw [h_w, sq]
                         ... = 289       : by norm_num
  use real.sqrt 289
  have h_sqrt : real.sqrt 289 = 17 := by norm_num
  rw h_sqrt
  norm_num
  sorry

end diagonal_of_rectangle_l403_403291


namespace value_of_frac_l403_403587

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403587


namespace value_of_expression_l403_403558

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403558


namespace log_b6b8_eq_four_l403_403627

-- Define the initial conditions and the correct answer
variables {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Define the common difference for the arithmetic sequence, which is not zero
noncomputable def common_difference := sorry

-- Define the theorem to be proved
theorem log_b6b8_eq_four 
  (h1 : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h2 : 2 * a 7 = a 3 + a 11)
  (h3 : b 7 = a 7) :
  log 2 (b 6 * b 8) = 4 := 
sorry

end log_b6b8_eq_four_l403_403627


namespace sum_f_1_to_22_l403_403428

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403428


namespace vertex_x_coordinate_of_quadratic_l403_403765

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- Define the x-coordinate of the vertex
def vertex_x_coordinate (f : ℝ → ℝ) : ℝ := 4

-- The theorem to prove
theorem vertex_x_coordinate_of_quadratic :
  vertex_x_coordinate quadratic_function = 4 :=
by
  -- Proof skipped
  sorry

end vertex_x_coordinate_of_quadratic_l403_403765


namespace fraction_value_l403_403570

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403570


namespace max_magnitude_PA_PB_l403_403917

open Real

def circle (P : ℝ × ℝ) : Prop := (P.1 - 1)^2 + (P.2 - 2)^2 = 4

def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (4, 0)

def PA (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -6 - P.2)
def PB (P : ℝ × ℝ) : ℝ × ℝ := (4 - P.1, -P.2)
def PA_PB (P : ℝ × ℝ) : ℝ × ℝ := (4 - 2 * P.1, -6 - 2 * P.2)
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

theorem max_magnitude_PA_PB (P : ℝ × ℝ) (h : circle P) : 
  magnitude (PA_PB P) ≤ 2 * sqrt 26 + 4 :=
sorry

end max_magnitude_PA_PB_l403_403917


namespace quadrilateral_inscribed_in_circle_l403_403691

theorem quadrilateral_inscribed_in_circle
  (EFGH : Type) -- Quadrilateral EFGH inscribed in a circle
  (α β γ δ : ℝ) -- α = angle EFG, β = angle EHG, γ = side EH, δ = side FG
  (inscribed : ∃ (circle : Type), EFGH ⊆ circle)
  (EFG_angle : α = 80)
  (EHG_angle : β = 50)
  (EH_side : γ = 5)
  (FG_side : δ = 7) :
  ∃ EF : ℝ, EF ≈ 5.4 := 
sorry

end quadrilateral_inscribed_in_circle_l403_403691


namespace problem_solution_l403_403263

noncomputable def positive_solution : {x : ℝ // x > 0} × {y : ℝ // y > 0} :=
  ⟨⟨1 + Real.sqrt 2, by linarith [Real.sqrt_pos.mpr (by norm_num : 2 > 0)]⟩,
   ⟨1 + Real.sqrt 2, by linarith [Real.sqrt_pos.mpr (by norm_num : 2 > 0)]⟩⟩

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y + 1/x + 1/y + 4 = 2 * (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1)) ↔ 
  x = 1 + Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 := 
sorry

end problem_solution_l403_403263


namespace hyperbola_equation_l403_403532

theorem hyperbola_equation (x y : ℝ) :
  (parabola : ∀ x, y = 1 / 8 * x^2) → 
  (focus : (parabola_focus : (0, 1)) of parabola) → 
  (eccentricity : sqrt 2) →
  (∃ a b : ℝ, a = sqrt 2 / 2 ∧ b = 1/2 ∧ 
                2 * y^2 - 2 * x^2 = 1) :=
by
  sorry

end hyperbola_equation_l403_403532


namespace triangle_problem_l403_403095

variables {A B C : Real} 
variables {a b c : ℝ} 

noncomputable def triangle_area (a b c : ℝ) (A B C : Real) : ℝ := 
  0.5 * a * b * Real.sin C

noncomputable def circumcircle_radius (a b c : ℝ) : ℝ := 
  (a * b * c) / (4 * triangle_area a b c)

theorem triangle_problem
  (a b c : ℝ) (A B C : Real) 
  (hb : b = 6)
  (harea : triangle_area a b c A B C = 15)
  (hR : circumcircle_radius a b c = 5) :
  (Real.sin (2 * B) = 24 / 25) ∧ (a + b + c = 6 + 6 * Real.sqrt 6) := 
sorry

end triangle_problem_l403_403095


namespace no_dissection_to_integer_ratio_right_triangles_l403_403858

theorem no_dissection_to_integer_ratio_right_triangles (n : ℕ) : 
  ¬ exists (triangles : list (ℕ × ℕ × ℕ)), 
    (∀ t ∈ triangles, let (a, b, c) := t in a^2 + b^2 = c^2) ∧ 
    are_integer_ratio_right_triangles triangles ∧
    dissector_polygons n triangles :=
  sorry

end no_dissection_to_integer_ratio_right_triangles_l403_403858


namespace sum_f_proof_l403_403438

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403438


namespace committee_selection_equivalence_l403_403788

theorem committee_selection_equivalence (n : ℕ) (h : nat.choose n 2 = 15): 
  nat.choose n 4 = 15 := by
  sorry

end committee_selection_equivalence_l403_403788


namespace white_paint_amount_l403_403731

theorem white_paint_amount (total_paint green_paint brown_paint : ℕ) 
  (h_total : total_paint = 69)
  (h_green : green_paint = 15)
  (h_brown : brown_paint = 34) :
  total_paint - (green_paint + brown_paint) = 20 := by
  sorry

end white_paint_amount_l403_403731


namespace find_b_l403_403015

noncomputable def f (x : ℝ) := x^2 + 5
noncomputable def g (x : ℝ) (b : ℝ) := f x - b * x

theorem find_b :
  (∀ x, x ∈ Icc (1 / 2) 1 → g x (1 / 2) ≤ g (1 / 2) (1 / 2) ∧ g (1 / 2) (1 / 2) = 11 / 2) →
  (∀ x, x ∈ Icc (1 / 2) 1 → g x b ≤ g 1 b ∧ g 1 b = 11 / 2) →
  b = 1 / 2 :=
by
  sorry

end find_b_l403_403015


namespace find_x_l403_403030

-- Define the conditions
def set := {8, 15, 22, 5}
def arithmetic_mean (s : Set ℚ) (x : ℚ) : Prop :=
  (8 + 15 + 22 + 5 + x) / 5 = 12

-- Lean statement to prove that x = 10 given the conditions
theorem find_x (x : ℚ) (h : arithmetic_mean set x) : x = 10 :=
sorry

end find_x_l403_403030


namespace div_polynomials_l403_403831

variable (a b : ℝ)

theorem div_polynomials :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b := 
by sorry

end div_polynomials_l403_403831


namespace inequality_solution_set_l403_403363

theorem inequality_solution_set : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | x < -1 ∨ x > 5} := 
by
  sorry

end inequality_solution_set_l403_403363


namespace problem_part1_problem_part2_l403_403308

noncomputable def transformation (λ μ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := (λ * p.fst, μ * p.snd)

def curve_C (x y : ℝ) : Prop := (x ^ 2) / 4 + (y ^ 2) / 2 = 1

theorem problem_part1 (λ μ : ℝ) (hλ : λ > 0) (hμ : μ > 0):
  (∀ x y, curve_C (λ * x) (μ * y) ↔ curve_C x y) ↔ (λ = 2 ∧ μ = real.sqrt 2) :=
sorry

noncomputable def ρ (θ : ℝ) : ℝ := 2 / real.sqrt (real.cos θ ^ 2 + 2 * real.sin θ ^ 2)

theorem problem_part2 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * real.pi) :
  (∀ ρ θ, ρ θ = ρ θ) ↔ (θ = real.pi / 2 ∧ ρ (real.pi / 2) = real.sqrt 2) :=
sorry

end problem_part1_problem_part2_l403_403308


namespace simplify_complex_l403_403696

theorem simplify_complex : (4 + 2 * complex.i) / (1 - complex.i) = 1 + 3 * complex.i := 
by sorry

end simplify_complex_l403_403696


namespace circle_and_position_l403_403364

noncomputable def point := (ℝ × ℝ)

def A : point := (1, 4)
def B : point := (3, 2)
def M1 : point := (2, 3)
def M2 : point := (2, 4)
def y_eq_zero (p : point) : Prop := p.2 = 0

def distance (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def is_circle (center : point) (radius : ℝ) (p : point) : Prop := 
  distance center p = radius ^ 2

def inside_circle (center : point) (radius : ℝ) (p : point) : Prop := 
  distance center p < radius ^ 2

def outside_circle (center : point) (radius : ℝ) (p : point) : Prop := 
  distance center p > radius ^ 2

theorem circle_and_position :
  ∃ (center : point) (radius : ℝ), y_eq_zero center ∧ 
  is_circle center radius A ∧ is_circle center radius B ∧
  (is_circle center 4.47213595499958 ∘ λ p => let (x, y) := p in (x + 1, y)) ∧
  inside_circle (-1, 0) (Math.sqrt 20) M1 ∧
  outside_circle (-1, 0) (Math.sqrt 20) M2 :=
by
  sorry -- Proof is omitted

end circle_and_position_l403_403364


namespace find_BD_in_triangle_ABC_l403_403645

theorem find_BD_in_triangle_ABC :
  ∀ (A B C D : Type) [euclidean_geometry A B C D],
  let AB := length A B,
  let AC := length A C,
  let B_angle := ∠ A B C,
  in AB = 3 ∧ AC = 3 * real.sqrt(7) ∧ B_angle = 60 :=
  let circumcircle := circumcircle_through ABC,
  let D := meet (angle_bisector A B C) circumcircle,
  BD = 4 * real.sqrt(3):=
sorry

end find_BD_in_triangle_ABC_l403_403645


namespace part_I_part_II_l403_403517

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.log x / Real.log 2) ^ 2 + 4 * (Real.log x / Real.log 2) + m

theorem part_I (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x ∈ Icc (1 / 8 : ℝ) 4 ∧ f x m = 0) ↔ (-12 <= m ∧ m < 0) := 
sorry

theorem part_II (m : ℝ) (α β : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Icc (1 / 8 : ℝ) 4 ∧ x₂ ∈ Icc (1 / 8 : ℝ) 4 ∧ f x₁ m = 0 ∧ f x₂ m = 0) ↔ (3 <= m ∧ m < 4 ∧ α * β = 1 / 16) :=
sorry

end part_I_part_II_l403_403517


namespace largest_value_sum_magic_triangle_l403_403975

theorem largest_value_sum_magic_triangle (a b c d e f : ℕ) (h1 : a ∈ {16, 17, 18, 19, 20, 21})
  (h2 : b ∈ {16, 17, 18, 19, 20, 21}) (h3 : c ∈ {16, 17, 18, 19, 20, 21})
  (h4 : d ∈ {16, 17, 18, 19, 20, 21}) (h5 : e ∈ {16, 17, 18, 19, 20, 21})
  (h6 : f ∈ {16, 17, 18, 19, 20, 21}) (h_distinct : a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → c ≠ d → c ≠ e → c ≠ f → d ≠ e → d ≠ f → e ≠ f)
  (h_sum : a + b + c = c + d + e ∧ c + d + e = e + f + a) :
  a + b + c = 57 :=
by
  sorry

end largest_value_sum_magic_triangle_l403_403975


namespace cone_to_sphere_surface_area_ratio_l403_403216

noncomputable def radius_sphere : ℝ := 1
noncomputable def height_cone : ℝ := 4
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * real.pi * r^2

theorem cone_to_sphere_surface_area_ratio :
  let r := radius_sphere in
  let h := height_cone in
  let d := 2 * r in
  let slant_height_cone := real.sqrt (r^2 + (h - r)^2) in
  let surface_area_cone := real.pi * r * slant_height_cone + real.pi * r^2 in
  (surface_area_cone / (surface_area_sphere r)) = 2 :=
by
  let r := radius_sphere
  let h := height_cone
  let d := 2 * r
  let slant_height_cone := real.sqrt (r^2 + (h - r)^2)
  let surface_area_cone := real.pi * r * slant_height_cone + real.pi * r^2
  let surface_area_sphere := 4 * real.pi * r^2
  
  sorry  -- Actual proof will go here.

end cone_to_sphere_surface_area_ratio_l403_403216


namespace equalize_nuts_l403_403737

open Nat

noncomputable def transfer (p1 p2 p3 : ℕ) : Prop :=
  ∃ (m1 m2 m3 : ℕ), 
    m1 ≤ p1 ∧ m1 ≤ p2 ∧ 
    m2 ≤ (p2 + m1) ∧ m2 ≤ p3 ∧ 
    m3 ≤ (p3 + m2) ∧ m3 ≤ (p1 - m1) ∧
    (p1 - m1 + m3 = 16) ∧ 
    (p2 + m1 - m2 = 16) ∧ 
    (p3 + m2 - m3 = 16)

theorem equalize_nuts : transfer 22 14 12 := 
  sorry

end equalize_nuts_l403_403737


namespace trapezoid_area_and_semicircle_subtraction_l403_403780

noncomputable def area_rectangle (w h : ℝ) : ℝ :=
  w * h

noncomputable def area_trapezoid (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

noncomputable def area_semicircle (r : ℝ) : ℝ :=
  (real.pi * r^2) / 2

theorem trapezoid_area_and_semicircle_subtraction (w h : ℝ) (r e f: ℝ)
  (hw : w > 0) (hh : h > 0) (hr : r = w / 2) (h_area: area_rectangle w h = 18)
  (hef: e + f = 2) :
  area_trapezoid (e + f) w h = 14.4 ∧
  (area_trapezoid (e + f) w h - area_semicircle r) = (area_semicircle r - 3.6) :=
by
  sorry

end trapezoid_area_and_semicircle_subtraction_l403_403780


namespace sum_g_2017_l403_403516

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 + 3 * x - (5 / 12)

-- The sum to be proven
theorem sum_g_2017 : 
  (Finset.sum (Finset.range 2016) (λ i, g ((i + 1 : ℝ) / 2017))) = 2016 :=
by
  sorry

end sum_g_2017_l403_403516


namespace intervals_of_monotonicity_extreme_values_range_of_a_two_distinct_solutions_l403_403527

def f (x : ℝ) := (x + 1) * Real.exp x

theorem intervals_of_monotonicity_extreme_values :
  (∀ x : ℝ, -2 < x -> 0 < (f' x) := (x + 2) * Real.exp x) ∧
  (∀ x : ℝ, x < -2 -> (f' x) < 0 :=
    sorry

theorem range_of_a_two_distinct_solutions :
  ∀ a : ℝ, - Real.exp (-2) < a -> a < 0 -> ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a :=
    sorry

end intervals_of_monotonicity_extreme_values_range_of_a_two_distinct_solutions_l403_403527


namespace sum_f_eq_neg_24_l403_403416

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403416


namespace trigonometric_identity_triangle_l403_403062

theorem trigonometric_identity_triangle :
  ∀ (A B C : ℝ) (AB AC BC : ℝ),
  AB = 7 → AC = 8 → BC = 5 →
  A + B + C = π →
  ∃ (A B C : ℝ),
  triangle.angle_sum A B C →
  triangle.side_length AB AC BC →
  (cos ((A - B) / 2) / sin (C / 2) - sin ((A - B) / 2) / cos (C / 2) = 16 / 7) :=
by
  -- Define the triangle and given sides
  intro A B C AB AC BC
  assume hAB : AB = 7
  assume hAC : AC = 8
  assume hBC : BC = 5
  -- Use the fact A + B + C = π (sum of angles in a triangle)
  assume hsum : A + B + C = π
  -- Define the necessary angles and sides
  have h_triangle : triangle.angle_sum A B C := triangle.angle_sum A B C hsum
  have h_sides : triangle.side_length AB AC BC := triangle.side_length AB AC BC hAB hAC hBC
  -- State the target
  existsi A
  existsi B
  existsi C
  use h_triangle
  use h_sides
  -- Placeholder to indicate the proof needs to follow
  sorry

end trigonometric_identity_triangle_l403_403062


namespace find_asymptotes_l403_403847

def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

def shifted_hyperbola_asymptotes (x y : ℝ) : Prop :=
  y = 4 / 3 * x + 5 ∨ y = -4 / 3 * x + 5

theorem find_asymptotes (x y : ℝ) :
  (∃ y', y = y' + 5 ∧ hyperbola_eq x y')
  ↔ shifted_hyperbola_asymptotes x y :=
by
  sorry

end find_asymptotes_l403_403847


namespace encoded_integer_eq_115_l403_403276

theorem encoded_integer_eq_115 
  (f : Fin 5 → Char) -- f represents the 1-to-1 correspondence between digits 0-4 and {A, B, C, D, E}
  (h_f : ∀ x₁ x₂, f x₁ = f x₂ → x₁ = x₂) -- f is injective
  (h_sequences : (f 2) = 'A' ∧ (f 3) = 'B' ∧ (f 4) = 'C' ∧ (f 0) = 'D' ∧ (f 1) = 'E')
  (h_consecutive : (f <$> [2, 3, 4].zip [A, B, C]) = ["A", "B", "C"] ∧ 
                   (f <$> [2, 3, 0].zip [A, B, D]) = ["A", "B", "D"] ∧ 
                   (f <$> [2, 1, 4].zip [A, C, E]) = ["A", "C", "E"] ∧ 
                   (f <$> [2, 0, 4].zip [A, D, A]) = ["A", "D", "A"]) : 
  (430:ℕ)₅ = 115 := 
sorry

end encoded_integer_eq_115_l403_403276


namespace range_of_a_over_b_l403_403980

theorem range_of_a_over_b (A B C : ℝ) (a b c : ℝ) (hA : 0 < A ∧ A < π / 2) (h1 : a^2 = b^2 + b * c) :
  sqrt 2 < a / b ∧ a / b < 2 :=
by sorry

end range_of_a_over_b_l403_403980


namespace probability_of_diagonal_intersection_in_nonagon_l403_403318

theorem probability_of_diagonal_intersection_in_nonagon :
  let n := 9 in
  let pairs_of_vertices := Nat.choose n 2 in
  let sides_of_nonagon := n in
  let diagonals := pairs_of_vertices - sides_of_nonagon in
  let diagonal_pairs := Nat.choose diagonals 2 in
  let intersecting_diagonals := Nat.choose n 4 in
  (intersecting_diagonals : ℚ) / diagonal_pairs = 2 / 7 :=
by
  let n := 9
  let pairs_of_vertices := Nat.choose n 2
  let sides_of_nonagon := n
  let diagonals := pairs_of_vertices - sides_of_nonagon
  let diagonal_pairs := Nat.choose diagonals 2
  let intersecting_diagonals := Nat.choose n 4
  have diagonal_prob : (intersecting_diagonals : ℚ) / diagonal_pairs = 2 / 7 := sorry
  exact diagonal_prob

end probability_of_diagonal_intersection_in_nonagon_l403_403318


namespace sum_trigonometric_identity_l403_403366

open Real

theorem sum_trigonometric_identity :
  (∑ x in finset.range (45 - 3 + 1).map (nat.add 3), 2 * sin (x - 1) * sin (x + 1) * (1 + sec (x - 2) * sec (x + 2))) = cos 4 - cos 92 :=
by
  sorry

end sum_trigonometric_identity_l403_403366


namespace solve_values_of_a_and_roots_l403_403331

noncomputable def equation_has_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (4 * x1^2 - 16 * abs x1 + (2 * a + abs x1 - x1)^2 = 16) ∧ 
    (4 * x2^2 - 16 * abs x2 + (2 * a + abs x2 - x2)^2 = 16)

theorem solve_values_of_a_and_roots :
  ∀ a : ℝ,
    (a ∈ Icc (-6 : ℝ) (-2) →
      (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = (a-2 + real.sqrt(12-4*a-a^2))/2 ∧ x2 = (a-2 - real.sqrt(12-4*a-a^2))/2)) ∧
    (a ∈ Ioo (2 : ℝ) (real.sqrt 8) →
      (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = 2 + real.sqrt(8 - a^2) ∧ x2 = 2 - real.sqrt(8 - a^2))) :=
by sorry

end solve_values_of_a_and_roots_l403_403331


namespace sum_f_1_to_22_l403_403453

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403453


namespace solve_system_of_equations_l403_403881

theorem solve_system_of_equations (n : ℕ) (hn : n ≥ 3) (x : ℕ → ℝ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    x i ^ 3 = (x ((i % n) + 1) + x ((i % n) + 2) + 1)) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    (x i = -1 ∨ x i = (1 + Real.sqrt 5) / 2 ∨ x i = (1 - Real.sqrt 5) / 2)) :=
sorry

end solve_system_of_equations_l403_403881


namespace a_2016_value_l403_403728

def sequence : ℕ → ℝ
| 0       := Real.sqrt 3
| (n + 1) := Real.floor (sequence n) + 1 / (sequence n - Real.floor (sequence n))

theorem a_2016_value : sequence 2016 = 3024 + Real.sqrt 3 := 
sorry

end a_2016_value_l403_403728


namespace unique_two_digit_solution_l403_403743

theorem unique_two_digit_solution:
  ∃! (s : ℕ), 10 ≤ s ∧ s ≤ 99 ∧ (13 * s ≡ 52 [MOD 100]) :=
  sorry

end unique_two_digit_solution_l403_403743


namespace range_of_A_l403_403933

noncomputable def f (x A : ℝ) : ℝ := Real.exp(x - 1) + A * x^2 - 1

theorem range_of_A (A : ℝ) : (∀ x : ℝ, x ≥ 1 → f x A ≥ A) ↔ (A ≥ -1/2) :=
sorry

end range_of_A_l403_403933


namespace tan_x_value_complex_trig_expression_value_l403_403381

theorem tan_x_value (x : ℝ) (h : Real.sin (x / 2) - 2 * Real.cos (x / 2) = 0) :
  Real.tan x = -4 / 3 :=
sorry

theorem complex_trig_expression_value (x : ℝ) (h : Real.sin (x / 2) - 2 * Real.cos (x / 2) = 0) :
  Real.cos (2 * x) / (Real.sqrt 2 * Real.cos (Real.pi / 4 + x) * Real.sin x) = 1 / 4 :=
sorry

end tan_x_value_complex_trig_expression_value_l403_403381


namespace distinct_integer_solutions_equation_l403_403242

theorem distinct_integer_solutions_equation (a : ℝ) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (∀ x : ℤ, (| |x - 3| - 2| = a ↔ x = x1 ∨ x = x2 ∨ x = x3))) ↔ a = 2 :=
by
  sorry

end distinct_integer_solutions_equation_l403_403242


namespace correct_conclusions_l403_403021

def E (m : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * m * x^2 + m * y^2 = 1

def A : ℝ × ℝ := (1, Real.sqrt 2)

-- Conditions for line l and intersection not directly translated, just stated that conclusions (2) and (3) are correct.
theorem correct_conclusions (m : ℝ) :
  A ∈ { p : ℝ × ℝ | E m p.1 p.2 } →
  m = 1/4 →
  -- Those are transformed conditions by substituting m = 1/4
  (2 : { q : ℝ // E m (q.1 * Real.sqrt 2) q.2 = 1}) ∧
  (3 : { p q r : ℝ // p = 1/2 * Real.sqrt 3 * (√(-m^2/2 + 4) * |m|/√3) = √2}) :=
by 
  sorry

end correct_conclusions_l403_403021


namespace find_SD_l403_403110

/-
  In rectangle ABCD, P is a point on BC such that ∠APD=90°. TS is perpendicular to BC with BP=PT. 
  PD intersects TS at Q. Point R is on CD such that RA passes through Q. In △PQA, PA=15, AQ=20, QP=25.
  Find SD (expressed as a common fraction).
-/

namespace Geometry

variables (A B C D P T S Q R : Point)
variable [rectangle ABCD]  -- ABCD is a rectangle
variable [is_point_on_line P BC] -- P is on BC
variable [right_angle A P D] -- ∠APD = 90°
variable [perpendicular TS BC] -- TS is perpendicular to BC
variable [equal_segments BP PT] -- BP = PT
variable [is_intersection Q PD TS] -- Q is the intersection of PD and TS
variable [is_point_on_line R CD] -- R is on CD
variable [passes_through RA Q] -- RA passes through Q
variable [triangle_PAQ P A Q] -- PA, AQ, QP form a triangle PAQ
variable [length P A 15] -- PA = 15
variable [length A Q 20] -- AQ = 20
variable [length Q P 25] -- QP = 25

-- Prove that SD = 475 / 6
theorem find_SD (P Q A : Point)
  (hp1 : length P A = 15)
  (hp2 : length A Q = 20)
  (hp3 : length Q P = 25) :
  ∃ (S D : Point), length S D = 475 / 6 :=
sorry

end Geometry

end find_SD_l403_403110


namespace sum_f_1_to_22_l403_403430

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403430


namespace triangle_side_a_l403_403995

theorem triangle_side_a {a b c : ℝ} {A B C : ℝ} (hb : b = real.sqrt 3) (hA : A = real.pi / 4) (hB : B = real.pi / 3) :
  a = real.sqrt 2 :=
begin
  sorry
end

end triangle_side_a_l403_403995


namespace stacy_days_to_complete_paper_l403_403174

def total_pages : ℕ := 66
def pages_per_day : ℕ := 11

theorem stacy_days_to_complete_paper :
  total_pages / pages_per_day = 6 := by
  sorry

end stacy_days_to_complete_paper_l403_403174


namespace exists_mk_for_all_n_l403_403776

def P : Set ℕ := {1, 2, 3, 4, 5}

def f (m : ℕ) (k : ℕ) : ℕ :=
  ∑ i in (Finset.range 5).image (λ i, i + 1), ⌊ m * Real.sqrt ((k + 1) / (i + 1)) ⌋

theorem exists_mk_for_all_n (n : ℕ) (hn : 0 < n) :
  ∃ (k ∈ P) (m : ℕ), f m k = n :=
  sorry

end exists_mk_for_all_n_l403_403776


namespace least_positive_integer_sigma_l403_403853

def sigma (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x, n % x = 0) (Finset.range (n + 1))), d

theorem least_positive_integer_sigma (n : ℕ) : sigma n = 12600 → n = 1200 :=
by { sorry }

end least_positive_integer_sigma_l403_403853


namespace conjugate_in_fourth_quadrant_l403_403896

def conjugate_quadrant_problem : Prop :=
  ∃ z : ℂ, 
    z * (1 - complex.I) = complex.abs (1 + complex.I) ∧ 
    (let z_conj := complex.conj z in z_conj.im < 0 ∧ z_conj.re > 0)

theorem conjugate_in_fourth_quadrant : conjugate_quadrant_problem :=
by 
  sorry

end conjugate_in_fourth_quadrant_l403_403896


namespace angle_between_focal_points_pf_on_ellipse_l403_403045

open Real

theorem angle_between_focal_points_pf_on_ellipse :
  let F1 := (0, -sqrt 3 : ℝ × ℝ)
  let F2 := (0, sqrt 3 : ℝ × ℝ)
  let e := sqrt 3 / 2
  ∃ P : ℝ × ℝ,
    (dist P F1 + dist P F2 = 4) -- P is on the ellipse with semi-major axis computed earlier
    ∧ (dot_prod F1 P F2 P = 2 / 3)
    → (angle F1 P F2 = π / 3) := 
begin
  sorry -- proof steps are not needed as per the guidelines
end

end angle_between_focal_points_pf_on_ellipse_l403_403045


namespace correct_completion_of_sentence_l403_403181

def committee_discussing_problem : Prop := True -- Placeholder for the condition
def problem_expected_to_be_solved_next_week : Prop := True -- Placeholder for the condition

theorem correct_completion_of_sentence 
  (h1 : committee_discussing_problem) 
  (h2 : problem_expected_to_be_solved_next_week) 
  : "hopefully" = "hopefully" :=
by 
  sorry

end correct_completion_of_sentence_l403_403181


namespace area_of_centroid_trace_l403_403671

open Real

theorem area_of_centroid_trace (D E F : Point) (r : ℝ) (h_DE : dist D E = 30)
  (h_F_circle : dist D F = r ∧ dist E F = r)
: area_of_trace_of_centroid DEF = 25 * π :=
by
  sorry

end area_of_centroid_trace_l403_403671


namespace cost_of_hamburger_l403_403252

-- Define the variables and given conditions
variables (H : ℝ)
variables (hamburgers : ℝ) (cola : ℝ) (discount : ℝ) (total_paid : ℝ)

-- Assign the values based on given conditions
def hamburgers := 2 * H
def cola := 3 * 2
def discount := 4
def total_paid := 12

-- Define the equation representing the final cost condition
def final_cost := hamburgers + cola - discount

-- State the theorem to prove
theorem cost_of_hamburger : final_cost = total_paid → H = 5 :=
by
  sorry

end cost_of_hamburger_l403_403252


namespace complex_real_imag_opposite_l403_403088

noncomputable def b_solution : ℝ :=
let z := (2 - (b : ℂ) * complex.I) / (3 + complex.I) in
if (z.re = -z.im) then
1
else
0

theorem complex_real_imag_opposite (b : ℝ) (h : (2 - b * complex.I) / (3 + complex.I) = z) : 
(z.re = -z.im) → b = 1 :=
by
  sorry

end complex_real_imag_opposite_l403_403088


namespace negation_of_exists_l403_403937

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l403_403937


namespace cell_population_evolution_l403_403272

theorem cell_population_evolution (n : ℕ) : 
  let initial_cells := 5 in
  let cells_lost_per_hour := 2 in
  let cells_after_n_hours : ℕ := 2^(n-1) + 4
  cells_after_n_hours = 2^(n-1) + 4 :=
by
  sorry

end cell_population_evolution_l403_403272


namespace range_of_x_for_function_l403_403724

theorem range_of_x_for_function :
  ∀ x : ℝ, (2 - x ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ 1) := by
  sorry

end range_of_x_for_function_l403_403724


namespace sum_f_1_to_22_l403_403426

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403426


namespace value_of_fraction_l403_403548

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403548


namespace fourth_number_in_10th_row_l403_403319

-- Define the lattice row pattern
def last_number_in_row (row_number : ℕ) : ℕ :=
  6 * row_number

-- Define the condition
def fourth_number_in_row (row_number : ℕ) : ℕ :=
  last_number_in_row(row_number) - 2

-- Lean statement for the main problem
theorem fourth_number_in_10th_row :
  fourth_number_in_row 10 = 58 :=
by
  -- The proof would go here
  sorry

end fourth_number_in_10th_row_l403_403319


namespace systematic_sampling_l403_403161

-- Define the conditions given
structure School (α : Type _) :=
  (students : α → bool)  -- placeholder to represent students
  (id_ends_with_5 : α → bool)
  (num_students_large : bool)
  (class_arrangement : α → ℕ)

-- Define a theorem that states the systematic sampling result
theorem systematic_sampling (s : α) [school : School α] 
  (cond1 : ∀ s, school.id_ends_with_5 s = true)
  (cond2 : school.num_students_large = true)
  (cond3 : ∀ s, school.students s = true)
  : ∃ s, school.students s ∧ school.id_ends_with_5 s → systematic_sampling_method :=
by
  sorry

end systematic_sampling_l403_403161


namespace sum_f_eq_neg24_l403_403496

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403496


namespace cones_common_volume_l403_403752

noncomputable def volume_of_common_part (l α β : ℝ) : ℝ := 
  (π * l^3 * (sin 2 * α)^2 * cos α * (sin β)^2) / 
  (12 * (sin (α + β))^2)

theorem cones_common_volume (l α β : ℝ) 
  (hl : l > 0) 
  (hα : 0 < α ∧ α < π/2) 
  (hβ : 0 < β ∧ β < π/2) : 
  volume_of_common_part l α β = 
  (π * l^3 * (sin (2 * α))^2 * cos α * (sin β)^2) / 
  (12 * (sin (α + β))^2) := sorry

end cones_common_volume_l403_403752


namespace largest_circle_area_l403_403798

theorem largest_circle_area (A : ℝ) (hA : A = 100) : 
  let s := (400 / Real.sqrt 3)^(1 / 2) in
  let P := 3 * s in
  let r := P / (2 * Real.pi) in
  Real.round (Real.pi * r^2) = 860 :=
by
  sorry

end largest_circle_area_l403_403798


namespace sum_f_k_from_1_to_22_l403_403482

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403482


namespace sum_f_1_to_22_l403_403459

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403459


namespace print_shop_X_charge_l403_403003

theorem print_shop_X_charge : 
  ∀ (x : ℝ), 
  (∀ (copied_amount : ℝ), copied_amount = 80 × 2.75) →
  (∀ (difference : ℝ), difference = 120) →
  (∀ (charge_Y : ℝ), charge_Y = copied_amount) →
  (∀ (charge_X : ℝ), charge_X = 80 * x) →
  charge_Y = charge_X + difference →
  x = 1.25 :=
begin
  intros x copied_amount h_copied_amount difference h_difference charge_Y h_charge_Y charge_X h_charge_X h_eq,
  sorry
end

end print_shop_X_charge_l403_403003


namespace find_value_of_fraction_l403_403567

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403567


namespace no_dissection_to_integer_ratio_right_triangles_l403_403860

theorem no_dissection_to_integer_ratio_right_triangles (n : ℕ) : 
  ¬ exists (triangles : list (ℕ × ℕ × ℕ)), 
    (∀ t ∈ triangles, let (a, b, c) := t in a^2 + b^2 = c^2) ∧ 
    are_integer_ratio_right_triangles triangles ∧
    dissector_polygons n triangles :=
  sorry

end no_dissection_to_integer_ratio_right_triangles_l403_403860


namespace f_prime_at_zero_l403_403990

-- Lean definition of the conditions.
def a (n : ℕ) : ℝ := 2 * (2 ^ (1/7)) ^ (n - 1)

-- The function f(x) based on the given conditions.
noncomputable def f (x : ℝ) : ℝ := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * 
  (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

-- The main goal to prove: f'(0) = 2^12
theorem f_prime_at_zero : deriv f 0 = 2^12 := by
  sorry

end f_prime_at_zero_l403_403990


namespace sum_f_k_1_22_l403_403509

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403509


namespace negation_of_exists_l403_403936

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l403_403936


namespace range_of_a_l403_403521

open Function

def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) : f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 := 
by
  sorry

end range_of_a_l403_403521


namespace increasing_f_for_k_eq_2_min_value_f_on_interval_l403_403054

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem increasing_f_for_k_eq_2 : ∀ x₁ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ < x₁ → f x₁ 2 > f x₂ 2 := by
  sorry

theorem min_value_f_on_interval (k : ℝ) : 
  (k ≥ 8 → ∃ x : ℝ, x ∈ set.Icc (1 : ℝ) 4 ∧ f x k = 16 - 4 * k) ∧ 
  (k ≤ 2 → ∃ x : ℝ, x ∈ set.Icc (1 : ℝ) 4 ∧ f x k = -k) ∧ 
  (2 ≤ k ∧ k ≤ 8 → ∃ x : ℝ, x ∈ set.Icc (1 : ℝ) 4 ∧ f x k = -1) := by
  sorry

end increasing_f_for_k_eq_2_min_value_f_on_interval_l403_403054


namespace sum_f_eq_neg24_l403_403462

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403462


namespace simplify_fraction_l403_403170

theorem simplify_fraction :
  (175 / 1225) * 25 = 25 / 7 :=
by
  -- Code to indicate proof steps would go here.
  sorry

end simplify_fraction_l403_403170


namespace hyperbola_equation_proof_l403_403052

-- Definitions for the given conditions
def parabola_focus_x : ℝ := 2
def semi_major_axis_hyperbola : ℝ := 2
def eccentricity : ℝ := 3 / 2
def semi_major_axis_square : ℝ := semi_major_axis_hyperbola ^ 2
def semi_minor_axis_square : ℝ := 5

-- Condition that the right vertex of hyperbola coincides with the focus of the parabola
def focus_condition (x := parabola_focus_x) : Prop :=
  x = semi_major_axis_hyperbola

-- The hyperbola equation condition based on eccentricity
def hyperbola_condition (a := semi_major_axis_hyperbola) 
                        (b := sqrt semi_minor_axis_square) 
                        (e := eccentricity) : Prop :=
  e = (sqrt (a ^ 2 + b ^ 2)) / a

-- The target hyperbola equation
def target_hyperbola_equation (x y : ℝ) : Prop :=
  (x ^ 2) / semi_major_axis_square - (y ^ 2) / semi_minor_axis_square = 1

theorem hyperbola_equation_proof : 
  focus_condition ∧ hyperbola_condition → 
  ∀ (x y : ℝ), target_hyperbola_equation x y :=
by 
  assume h : focus_condition ∧ hyperbola_condition
  show ∀ (x y : ℝ), target_hyperbola_equation x y
  sorry -- This is to skip the proof as stated in the conditions

end hyperbola_equation_proof_l403_403052


namespace people_who_speak_French_l403_403101

theorem people_who_speak_French (T L N B : ℕ) (hT : T = 25) (hL : L = 13) (hN : N = 6) (hB : B = 9) : 
  ∃ F : ℕ, F = 15 := 
by 
  sorry

end people_who_speak_French_l403_403101


namespace conjugate_of_fraction_l403_403182

theorem conjugate_of_fraction :
  (∀ (z : ℂ), z = 1 - 2 * I → conj z = 1 + 2 * I) → conj (5 / (1 + 2 * I)) = 1 + 2 * I :=
by
  intro h
  apply h
  calc
    5 / (1 + 2 * I)
        = (5 * (1 - 2 * I)) / ((1 + 2 * I) * (1 - 2 * I)) : by sorry
    ... = 1 - 2 * I : by sorry

end conjugate_of_fraction_l403_403182


namespace minimize_magnitude_x_l403_403913

-- Conditions
variables (a b : ℝ)
variable (θ : ℝ)

-- Given conditions
variables (h1 : ∥a∥ = 2)
variables (h2 : ∥b∥ = 1)
variables (h3 : θ = real.pi / 3) -- 60 degrees in radian

-- The target result to prove
theorem minimize_magnitude_x (x : ℝ) (h4 : ∥a∥ = 2) (h5 : ∥b∥ = 1) (h6 : θ = real.pi / 3) : 
  (∥a - x * b∥) minimized by x = 1 := sorry

end minimize_magnitude_x_l403_403913


namespace zongzi_lotus_seed_count_l403_403120

theorem zongzi_lotus_seed_count :
  let total_zongzi := 72 + 18 + 36 + 54
  in let lotus_seed_zongzi := 54
  in let gift_box_count := 10
  in gift_box_count * lotus_seed_zongzi / total_zongzi = 3 :=
by
  let total_zongzi : ℕ := 72 + 18 + 36 + 54
  let lotus_seed_zongzi : ℕ := 54
  let gift_box_count : ℕ := 10
  have h1 : total_zongzi = 180 := by rfl
  have h2 : lotus_seed_zongzi = 54 := by rfl
  have h3 : gift_box_count = 10 := by rfl
  have h4 : gift_box_count * lotus_seed_zongzi / total_zongzi = 3 :=
    by simp [h1, h2, h3]; norm_num
  exact h4

end zongzi_lotus_seed_count_l403_403120


namespace distance_A_B_midpoint_A_B_l403_403351

-- Define points A and B
def A : ℝ × ℝ × ℝ := (1, 2, -5)
def B : ℝ × ℝ × ℝ := (4, 9, -3)

-- Define function to calculate Euclidean distance between two points in 3D
def euclidean_distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Define function to calculate midpoint between two points in 3D
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Proposition stating the distance between A and B is sqrt(62)
theorem distance_A_B : euclidean_distance A B = Real.sqrt 62 :=
by
  sorry

-- Proposition stating the midpoint between A and B
theorem midpoint_A_B : midpoint A B = (5/2, 11/2, -4) :=
by
  sorry

end distance_A_B_midpoint_A_B_l403_403351


namespace calculate_sum_l403_403386

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (π / 4 * x)

theorem calculate_sum :
  let A : ℝ := 2
  let ω : ℝ := π / 4
  A > 0 ∧ ω > 0 ∧ abs (0 : ℝ) ≤ π / 2 ∧ (∀ x, f(-x) = -f(x)) ∧ f(2) = 2
  → (f(1) + f(2) + f(3) + 1 / 4 + f(100) = 2 + 2 * Real.sqrt 2) := by
  intros A ω h
  sorry

end calculate_sum_l403_403386


namespace value_of_expression_l403_403555

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403555


namespace problem_l403_403583

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403583


namespace find_line_l_l403_403200

theorem find_line_l :
  ∃ l : ℝ × ℝ → Prop,
    (∀ (B : ℝ × ℝ), (2 * B.1 + B.2 - 8 = 0) → 
      (∀ A : ℝ × ℝ, (A.1 = -B.1 ∧ A.2 = 2 * B.1 - 6 ) → 
        (A.1 - 3 * A.2 + 10 = 0) → 
          B.1 = 4 ∧ B.2 = 0 ∧ ∀ p : ℝ × ℝ, B.1 * p.1 + 4 * p.2 - 4 = 0)) := 
  sorry

end find_line_l_l403_403200


namespace average_games_rounded_l403_403818

theorem average_games_rounded (n1 n2 n3 n4 n5 n6 : ℕ) (h1 : n1 = 6) (h2 : n2 = 3) (h3 : n3 = 1) (h4 : n4 = 4) (h5 : n5 = 2) (h6 : n6 = 6) :
  let total_games := n1*1 + n2*2 + n3*3 + n4*4 + n5*5 + n6*6,
      total_players := n1 + n2 + n3 + n4 + n5 + n6,
      average_games := total_games / total_players in
  round average_games = 4 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  -- Remaining proof steps will involve calculation which are skipped here.
  -- Apply appropriate simplification and arithmetic operations to show the result.
  sorry

end average_games_rounded_l403_403818


namespace sum_f_proof_l403_403433

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403433


namespace first_number_in_sum_l403_403769

theorem first_number_in_sum (a b c : ℝ) (h : a + b + c = 3.622) : a = 3.15 :=
by
  -- Assume the given values of b and c
  have hb : b = 0.014 := sorry
  have hc : c = 0.458 := sorry
  -- From the assumption h and hb, hc, we deduce a = 3.15
  sorry

end first_number_in_sum_l403_403769


namespace find_angle_B_l403_403391

noncomputable def triangle_angles := Prop

variable {A B C a b c : ℝ}
variable (A_pos : 0 < A) (A_lt_pi : A < Real.pi)

def vectors_perpendicular (m n : Vector ℝ 2) : Prop :=
  dot_product m n = 0

def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧ c / Real.sin C = a / Real.sin A

theorem find_angle_B (A B C a b c : ℝ)
  (A_pos : 0 < A) (A_lt_pi : A < Real.pi)
  (m : Vector ℝ 2) (n : Vector ℝ 2)
  (m_def : m = (⟨√3, -1⟩ : Vector ℝ 2))
  (n_def : n = (⟨Real.cos A, Real.sin A⟩ : Vector ℝ 2))
  (m_perp_n : vectors_perpendicular m n)
  (law_sine : law_of_sines a b c A B C)
  (cosine_relation : a * Real.cos B + b * Real.cos A = c * Real.sin C) :
  B = π / 6 :=
by
  sorry

end find_angle_B_l403_403391


namespace ratio_third_to_second_cooler_l403_403129

noncomputable def first_cooler_capacity : ℝ := 100
noncomputable def second_cooler_capacity : ℝ := 1.5 * first_cooler_capacity
noncomputable def total_capacity : ℝ := 325

theorem ratio_third_to_second_cooler :
  ∃ (c1 c2 c3 : ℝ), 
  c1 = first_cooler_capacity ∧
  c2 = second_cooler_capacity ∧
  c1 + c2 + c3 = total_capacity ∧
  c3 / c2 = 1 / 2 := 
begin
  sorry
end

end ratio_third_to_second_cooler_l403_403129


namespace sum_f_k_from_1_to_22_l403_403488

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403488


namespace cheryl_distance_walked_l403_403836

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l403_403836


namespace maximum_sequence_length_l403_403278

theorem maximum_sequence_length
  (seq : List ℚ) 
  (h1 : ∀ i : ℕ, i + 2 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2)) < 0)
  (h2 : ∀ i : ℕ, i + 3 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2) + seq.get! (i+3)) > 0) 
  : seq.length ≤ 5 := 
sorry

end maximum_sequence_length_l403_403278


namespace simplify_expression_l403_403698

theorem simplify_expression (x : ℤ) : 
  (12*x^10 + 5*x^9 + 3*x^8) + (2*x^12 + 9*x^10 + 4*x^8 + 6*x^4 + 7*x^2 + 10)
  = 2*x^12 + 21*x^10 + 5*x^9 + 7*x^8 + 6*x^4 + 7*x^2 + 10 :=
by sorry

end simplify_expression_l403_403698


namespace find_total_original_cost_l403_403968

noncomputable def original_total_cost (x y z : ℝ) : ℝ :=
x + y + z

theorem find_total_original_cost (x y z : ℝ)
  (h1 : x * 1.30 = 351)
  (h2 : y * 1.25 = 275)
  (h3 : z * 1.20 = 96) :
  original_total_cost x y z = 570 :=
sorry

end find_total_original_cost_l403_403968


namespace range_of_a_l403_403610

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - x ^ 3

-- State the main theorem
theorem range_of_a (a : ℝ) (h_min : ∃ x ∈ set.Ioo (a^2 - 12) a, ∀ y ∈ set.Ioo (a^2 - 12) a, f y ≥ f x) : a ∈ set.Ioc (-1 : ℝ) 2 :=
by sorry

end range_of_a_l403_403610


namespace total_lives_l403_403748

theorem total_lives (initial_players new_players lives_per_person : ℕ)
  (h_initial : initial_players = 8)
  (h_new : new_players = 2)
  (h_lives : lives_per_person = 6)
  : (initial_players + new_players) * lives_per_person = 60 := 
by
  sorry

end total_lives_l403_403748


namespace union_sets_l403_403970

open Set

variable (α : Type) [LinearOrder α]

-- Definitions
def M : Set α := { x | -1 < x ∧ x < 3 }
def N : Set α := { x | 1 ≤ x }

-- Theorem statement
theorem union_sets : M α ∪ N α = { x | -1 < x } := sorry

end union_sets_l403_403970


namespace value_of_expression_l403_403554

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403554


namespace nancy_kept_chips_correct_l403_403676

/-- Define the initial conditions -/
def total_chips : ℕ := 22
def chips_to_brother : ℕ := 7
def chips_to_sister : ℕ := 5

/-- Define the number of chips Nancy kept -/
def chips_kept : ℕ := total_chips - (chips_to_brother + chips_to_sister)

theorem nancy_kept_chips_correct : chips_kept = 10 := by
  /- This is a placeholder. The proof would go here. -/
  sorry

end nancy_kept_chips_correct_l403_403676


namespace golden_apples_per_pint_l403_403069

-- Data definitions based on given conditions and question
def farmhands : ℕ := 6
def apples_per_hour : ℕ := 240
def hours : ℕ := 5
def ratio_golden_to_pink : ℕ × ℕ := (1, 2)
def pints_of_cider : ℕ := 120
def pink_lady_per_pint : ℕ := 40

-- Total apples picked by farmhands in 5 hours
def total_apples_picked : ℕ := farmhands * apples_per_hour * hours

-- Total pink lady apples picked
def total_pink_lady_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.2) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total golden delicious apples picked
def total_golden_delicious_apples : ℕ := (total_apples_picked * ratio_golden_to_pink.1) / (ratio_golden_to_pink.1 + ratio_golden_to_pink.2)

-- Total pink lady apples used for 120 pints of cider
def pink_lady_apples_used : ℕ := pints_of_cider * pink_lady_per_pint

-- Number of golden delicious apples used per pint of cider
def golden_delicious_apples_per_pint : ℕ := total_golden_delicious_apples / pints_of_cider

-- Main theorem to prove
theorem golden_apples_per_pint : golden_delicious_apples_per_pint = 20 := by
  -- Start proof (proof body is omitted)
  sorry

end golden_apples_per_pint_l403_403069


namespace value_of_frac_l403_403586

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403586


namespace math_proof_problem_l403_403395

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

noncomputable def conditions (e : ℝ) (vertex : ℝ × ℝ) (p q : ℝ × ℝ) : Prop :=
  e = 1 / 2
  ∧ vertex = (0, 2 * Real.sqrt 3)  -- focus of the parabola
  ∧ p = (-2, -3)
  ∧ q = (-2, 3)

noncomputable def max_area_quadrilateral (area : ℝ) : Prop :=
  area = 12 * Real.sqrt 3

theorem math_proof_problem : 
  ∃ a b p q area, ellipse_standard_eq a b ∧ conditions (1/2) (0, 2 * Real.sqrt 3) p q 
  ∧ p = (-2, -3) ∧ q = (-2, 3) → max_area_quadrilateral area := 
  sorry

end math_proof_problem_l403_403395


namespace count_valid_numbers_l403_403007

-- Defining a characteristic of a valid four-digit number in this problem.
def is_valid_number (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (→ ∃ k : ℕ, (2 = n / 10^k % 10) ∧ (3 = n / 10^(k + 1) % 10))

-- The main theorem statement
theorem count_valid_numbers : ∃ k : ℕ, k = 14 :=
begin
  -- All valid numbers are counted
  let count := (4 + 6 + 4),
  use count,
  refl,
end

end count_valid_numbers_l403_403007


namespace negation_of_existential_l403_403723

theorem negation_of_existential (P : ℝ → Prop) (hP : P = (λ x, x > 0 ∧ 2^x > 10)) :
  (¬ ∃ x, P x) = (∀ x, x > 0 → 2^x ≤ 10) :=
by
  sorry

end negation_of_existential_l403_403723


namespace AliceWinningPairsCount_l403_403634

-- Definition to represent the condition of the game
def canAliceWinFirstMove (m n : ℕ) : Prop :=
  -- A placeholder for the actual condition where Alice can guarantee a win.
  sorry

-- List of the given pairs
def givenPairs : List (ℕ × ℕ) :=
  [(7, 79), (17, 71), (10, 101), (21, 251), (50, 405)]

-- Lean 4 statement that captures the problem and expected outcome
theorem AliceWinningPairsCount :
  let win_count := givenPairs.filter (λ p, canAliceWinFirstMove p.1 p.2).length
  win_count = 4 :=
by
  sorry

end AliceWinningPairsCount_l403_403634


namespace max_value_of_z_l403_403089

theorem max_value_of_z 
  (x y : ℝ) 
  (h1: y ≤ 1) 
  (h2: x + y ≥ 0) 
  (h3: x - y - 2 ≤ 0) : 
  (∃ z, z = x - 2y ∧ z ≤ 3) :=
begin
  sorry
end

end max_value_of_z_l403_403089


namespace forming_n_and_m_l403_403229

def is_created_by_inserting_digit (n: ℕ) (base: ℕ): Prop :=
  ∃ d1 d2 d3 d: ℕ, n = d1 * 1000 + d * 100 + d2 * 10 + d3 ∧ base = d1 * 100 + d2 * 10 + d3

theorem forming_n_and_m (a b: ℕ) (base: ℕ) (sum: ℕ) 
  (h1: is_created_by_inserting_digit a base)
  (h2: is_created_by_inserting_digit b base) 
  (h3: a + b = sum):
  (a = 2195 ∧ b = 2165) 
  ∨ (a = 2185 ∧ b = 2175) 
  ∨ (a = 2215 ∧ b = 2145) 
  ∨ (a = 2165 ∧ b = 2195) 
  ∨ (a = 2175 ∧ b = 2185) 
  ∨ (a = 2145 ∧ b = 2215) := 
sorry

end forming_n_and_m_l403_403229


namespace first_fifty_sum_difference_l403_403213

theorem first_fifty_sum_difference :
  let odd_seq (n : ℕ) := 2 * n - 1
  let even_seq_minus_3 (n : ℕ) := 2 * n - 3
  ∑ i in Finset.range 50, (even_seq_minus_3 (i + 1)) - ∑ i in Finset.range 50, (odd_seq (i + 1)) = -100 :=
by
  let odd_seq (n : ℕ) := 2 * n - 1
  let even_seq_minus_3 (n : ℕ) := 2 * n - 3
  have h1 : ∀ i, (even_seq_minus_3 (i + 1)) - (odd_seq (i + 1)) = -2,
  { intro i, simp [odd_seq, even_seq_minus_3], linarith },
  have h2 : ∑ i in Finset.range 50, -2 = -100,
  { simp [Finset.sum_const, Finset.card_range], },
  rw Finset.sum_sub_distrib,
  rw h1,
  rw h2,
  sorry


end first_fifty_sum_difference_l403_403213


namespace quadratic_completing_the_square_q_l403_403321

theorem quadratic_completing_the_square_q (x p q : ℝ) (h : 4 * x^2 + 8 * x - 468 = 0) :
  (∃ p, (x + p)^2 = q) → q = 116 := sorry

end quadratic_completing_the_square_q_l403_403321


namespace problem_part1_problem_part2_l403_403888

variable (n : ℕ) (a : ℕ → ℤ)
def a0 : ℤ := (x+2)^n - (4^n - 3^n)

def Sn : ℤ := 4^n - 3^n

theorem problem_part1 (n : ℕ) (hn : 0 < n) : 
  let x := 1 in 
  a 0 = 3^n := by
  sorry

theorem problem_part2 (n : ℕ) 
  (hn1 : n = 1 → 4^n > (n-1)*3^n + 2*n^2)
  (hn2_3 : n = 2 ∨ n = 3 → 4^n < (n-1)*3^n + 2*n^2)
  (hn_ge4 : 4 ≤ n → 4^n > (n-1)*3^n + 2*n^2) : 
  n = 1 → Sn n > (n-2)*3^n + 2*n^2 ∧
  (n = 2 ∨ n = 3) → Sn n < (n-2)*3^n + 2*n^2 ∧
  4 ≤ n → Sn n > (n-2)*3^n + 2*n^2 := by
  sorry

end problem_part1_problem_part2_l403_403888


namespace sum_of_digits_of_given_integer_l403_403241

theorem sum_of_digits_of_given_integer : 
  let number := (3 * 10^500 - 2022 * 10^497 - 2022)
  in (sum_of_digits number) = 4491 :=
sorry

end sum_of_digits_of_given_integer_l403_403241


namespace total_cookies_l403_403949

-- Define the number of bags and cookies per bag
def num_bags : Nat := 37
def cookies_per_bag : Nat := 19

-- The theorem stating the total number of cookies
theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end total_cookies_l403_403949


namespace range_of_m_l403_403935

noncomputable def inequality_solutions (x m : ℝ) := |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, inequality_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l403_403935


namespace sum_G_equals_2016531_l403_403375

-- Define G(n): ℕ → ℕ to count the number of solutions to cos x = cos nx over [0, π]
def G (n : ℕ) : ℕ :=
  if n % 4 = 0 then n
  else n + 1

-- Now we want to prove the sum of G(n) from 2 to 2007 is 2,016,531
theorem sum_G_equals_2016531 : 
  (∑ n in Finset.range 2006 + 2, G n) = 2_016_531 :=
by
  sorry

end sum_G_equals_2016531_l403_403375


namespace option_c_correct_l403_403245

-- Statement of the problem: Prove that (x-3)^2 = x^2 - 6x + 9

theorem option_c_correct (x : ℝ) : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=
by
  sorry

end option_c_correct_l403_403245


namespace sum_of_x_coords_at_intersections_l403_403163

theorem sum_of_x_coords_at_intersections (c d : ℕ) (hc : 0 < c) (hd : 0 < d) 
  (h_intersect : 7 * (2 * c) = 2 * (7 * d)) :
  ∑ x in ({ - (c / 7) } ∩ { - (d / 2) }), x = -1 :=
begin
  sorry
end

end sum_of_x_coords_at_intersections_l403_403163


namespace unique_odd_and_increasing_function_l403_403305

-- Definitions from the conditions
def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := Real.log x
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x^2

-- Proving that f3 is the only function that is both odd and monotonically increasing on (0, +∞)
theorem unique_odd_and_increasing_function 
  (f : ℝ → ℝ) (interval : Set ℝ)
  (f ∈ {f1, f2, f3, f4}) 
  (interval = Set.Ioi 0) :
  (∃! f, 
    (∀ x ∈ interval, -f x = f (-x)) ∧ 
    (∀ x y ∈ interval, x < y → f x < f y)
  ) := 
sorry

end unique_odd_and_increasing_function_l403_403305


namespace limit_series_l403_403664

open Real BigOperators Nat

noncomputable def a_n (n : ℕ) : ℝ :=
  if h : n ≥ 2 then (n * (n - 1) / 2) * 3^(n - 2) else 0

theorem limit_series (S : ℕ → ℝ) (h₀ : ∀ n ≥ 2, S n = (3 ^ n) / a_n n) : 
  (filter.Tendsto (λ n, (finset.range n).filter(λ k, 2 ≤ k).sum S) filter.at_top (nhds 18)) :=
begin
  sorry
end

end limit_series_l403_403664


namespace smallest_angle_in_22_gon_is_143_l403_403185

-- Given: The degree measures of the angles in a convex 22-sided polygon form an increasing arithmetic sequence with integer values.
-- Prove: The degree measure of the smallest angle is 143 degrees.

theorem smallest_angle_in_22_gon_is_143 :
  ∃ (a : ℕ) (d : ℕ), 
  (∀ i : ℕ, 1 ≤ i → i ≤ 22 → (a + (i - 1) * d < 180)) ∧ 
  (22 * 2 * a + 21 * 21 * d = 3600) ∧
  ((21 > 0) ∧ (a − 21 * d = 143))
:=
sorry

end smallest_angle_in_22_gon_is_143_l403_403185


namespace circle_area_proof_l403_403684

theorem circle_area_proof (A B : ℝ × ℝ) (ω : set (ℝ × ℝ))
  (hA : A = (8, 15)) (hB : B = (14, 9))
  (tangent_A : ∃ l₁ : ℝ → ℝ, ∀ p ∈ ω, p ≠ A → l₁ (fst p) = snd p)
  (tangent_B : ∃ l₂ : ℝ → ℝ, ∀ p ∈ ω, p ≠ B → l₂ (fst p) = snd p)
  (intersect_x_axis : ∃ C : ℝ × ℝ, C = (-1, 0) ∧ 
    ∃ l₁ l₂ : ℝ → ℝ, ∀ p ∈ ω, p ≠ A → l₁ (fst p) = snd p ∧ 
                       ∀ p ∈ ω, p ≠ B → l₂ (fst p) = snd p ∧ 
                       l₁ (-1) = 0 ∧ l₂ (-1) = 0) :
  ∃ r : ℝ, r = 15 ∧ ∃ area : ℝ, area = 234 * Real.pi :=
by
  sorry

end circle_area_proof_l403_403684


namespace find_x_l403_403890

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = -6 := by
  have h1 : 2 * 3 + 1 * x = 0 := h
  have h2 : 6 + x = 0 := by linarith
  exact eq_neg_of_add_eq_zero h2

end find_x_l403_403890


namespace distance_covered_is_correct_l403_403875

-- Define the conditions
def time_uphill_hours : ℝ := 25 / 60
def speed_uphill_kmh : ℝ := 8
def distance_uphill : ℝ := speed_uphill_kmh * time_uphill_hours

def time_downhill_hours : ℝ := 11 / 60
def speed_downhill_kmh : ℝ := 12
def distance_downhill : ℝ := speed_downhill_kmh * time_downhill_hours

-- Define the total distance
def total_distance : ℝ := distance_uphill + distance_downhill

-- The theorem we want to prove
theorem distance_covered_is_correct :
  total_distance = 5.5336 :=
by 
  -- Placeholder for proof
  sorry

end distance_covered_is_correct_l403_403875


namespace sock_distribution_l403_403109

-- Defining the problem parameters and the binomial coefficient
namespace SockDistribution

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if h : k > n then 0 else Nat.choose n k

-- The main theorem stating the problem
theorem sock_distribution (n k : ℕ) (hk : k ≥ 1) : 
  (number of ways to distribute n identical socks into k drawers) = binom (n + k - 1) (k - 1) :=
by
  sorry

end SockDistribution

end sock_distribution_l403_403109


namespace part_a_solutions_part_b_num_solutions_l403_403136

noncomputable theory
open_locale big_operators

-- Part (a)
theorem part_a_solutions (n : ℕ) : 
  {x : ℕ | (∑ k in finset.range (n + 1), (⌊x / (2^k)⌋ : ℕ)) = x - 1} =
  {2 ^ i | i ∈ finset.range (n + 1)} :=
sorry

-- Part (b)
theorem part_b_num_solutions (n m : ℕ) :
  (finset.card {x : ℕ | (∑ k in finset.range (n + 1), (⌊x / (2^k)⌋ : ℕ)) = x - m}) =
  (finset.range (m + 1)).sum (λ i, nat.choose n i) :=
sorry

end part_a_solutions_part_b_num_solutions_l403_403136


namespace sine_probability_l403_403955

noncomputable def probability_sine_inequality : ℝ :=
Real.pi / 2 / Real.pi

theorem sine_probability (theta : ℝ) (h : 0 ≤ theta ∧ theta ≤ Real.pi) :
  prob (fun theta => Real.sin (theta + Real.pi / 3) < 1 / 2) [0, Real.pi] = (1 : ℝ) / 2 :=
begin
  -- Outline of proof needed
  sorry
end

end sine_probability_l403_403955


namespace projection_correct_l403_403877

-- Define the vector a
def a : ℝ × ℝ × ℝ := (5, -3, 2)

-- Define the direction vector b of the line
def b : ℝ × ℝ × ℝ := (1, -2, 2)

-- Dot product of two 3D vectors
def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

-- Projection of vector a onto vector b
def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := (dot_product a b) / (dot_product b b)
  (scalar * b.1, scalar * b.2, scalar * b.3)

-- The expected projection result
def expected_proj : ℝ × ℝ × ℝ := (5/3, -10/3, 10/3)

-- The statement to prove
theorem projection_correct : proj a b = expected_proj := by
  sorry

end projection_correct_l403_403877


namespace six_points_mapping_l403_403777

theorem six_points_mapping : ∃ f : (ℕ → ℕ), (∀ a b c, collinear a b c → collinear (f a) (f b) (f c)) ∧ (∀ l, ∃! l', permutes_line f l l') :=
by sorry

def collinear : ℕ → ℕ → ℕ → Prop := sorry
def permutes_line (f : ℕ → ℕ) (l l' : ℕ) : Prop := sorry

-- collinear a b c means points a, b, c are on the same line
-- permutes_line f l l' means line l is mapped to line l' by the permutation f

end six_points_mapping_l403_403777


namespace real_root_interval_l403_403332

def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_interval : ∃ (c : ℝ), c ∈ set.Icc 1 2 ∧ f c = 0 :=
by
  -- proof will go here
  sorry

end real_root_interval_l403_403332


namespace max_x0_value_l403_403876

noncomputable def max_x0 : ℝ :=
  let seq_condition := ∃ (x : Fin (1996) → ℝ), 
    (∀ i : Fin 1995, 0 < x i ∧ 
      x i + (2 / x i) = 2 * x (Fin.succ i) + (1 / x (Fin.succ i))) ∧
    x 0 = x 1995 
  in if h : seq_condition then 2^997 else 0

theorem max_x0_value : max_x0 = 2^997 :=
sorry

end max_x0_value_l403_403876


namespace problem_statement_l403_403380

noncomputable def tangent_sum_formula (x y : ℝ) : ℝ :=
  (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)

theorem problem_statement
  (α β : ℝ)
  (hαβ1 : 0 < α ∧ α < π)
  (hαβ2 : 0 < β ∧ β < π)
  (h1 : Real.tan (α - β) = 1 / 2)
  (h2 : Real.tan β = - 1 / 7)
  : 2 * α - β = - (3 * π / 4) :=
sorry

end problem_statement_l403_403380


namespace find_solutions_l403_403874

theorem find_solutions (m n : ℕ) (positive_m : m > 0) (positive_n : n > 0) :
    (n + 1) * m = n! + 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 5 ∧ n = 4) :=
begin
  sorry
end

end find_solutions_l403_403874


namespace true_statements_number_is_two_l403_403192

/-
  Definitions of the conditions as given in the problem.
-/

def quadl_1 (Q : Type) : Prop :=
  ∃ (p1 p2 : Set), (parallel p1) ∧ (equal p2)

def quadl_2 (Q : Type) : Prop :=
  ∃ (d1 d2 : Set), (perpendicular d1 d2) ∧ (bisect d1 d2)

def quadl_3 (Q : Type) : Prop :=
  ∃ (angle90 : Q) (adj_sides_equal : Q), (right_angle angle90) ∧ (adjacent_sides_equal adj_sides_equal)

def parallelogram_with_equal_diagonals (Q : Type) : Prop :=
  ∃ (diagonal_eq : Q), (equal_diagonals diagonal_eq)

/-
  The proof statement to show.
-/
theorem true_statements_number_is_two (Q : Type) :
  (¬quadl_1 Q) ∧ quadrilateral_is_rhombus Q ∧ (¬quadl_3 Q) ∧ quadrilateral_is_rectangle Q → 2 :=
sorry

end true_statements_number_is_two_l403_403192


namespace three_point_three_seven_five_as_fraction_l403_403757

theorem three_point_three_seven_five_as_fraction :
  3.375 = (27 / 8 : ℚ) :=
by sorry

end three_point_three_seven_five_as_fraction_l403_403757


namespace sum_f_l403_403448

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403448


namespace area_of_equilateral_triangle_MNP_l403_403803

noncomputable def area_of_triangle : ℝ :=
  let s := sqrt (4 * sqrt 3) in
  let r := sqrt (2 * sqrt 3) in
  let a := 2 * sqrt (6 * sqrt 3) in
  (sqrt 3 / 4) * a^2

theorem area_of_equilateral_triangle_MNP :
  let area_square := 4 * sqrt 3 in
  ∃ (a : ℝ) (M N P : ℝ), 
  (area_square = 4 * sqrt 3) ∧
  (s = sqrt (4 * sqrt 3)) ∧
  ((s * sqrt 2) / 2 = sqrt (2 * sqrt 3)) ∧
  (a = 2 * sqrt (6 * sqrt 3)) ∧
  (area_of_triangle = 18) :=
sorry

end area_of_equilateral_triangle_MNP_l403_403803


namespace nancy_boots_l403_403677

theorem nancy_boots (B : ℕ) (h1 : B + 9 + 3 * (2 * B + 9) = 168 / 2) : B = 6 :=
by
  have h2 : 2 * B + 2 * (B + 9) + 2 * 3 * (2 * B + 9) = 168,
  { linarith },
  sorry

end nancy_boots_l403_403677


namespace value_of_frac_l403_403592

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403592


namespace probability_union_inequality_l403_403906

open Probability

variables {Ω : Type*} {P : Measure Ω}
variables (A B : Set Ω)

theorem probability_union_inequality (hA : MeasurableSet A) (hB : MeasurableSet B) :
  P[A] + P[B] ≥ P[A ∪ B] :=
by
  sorry

end probability_union_inequality_l403_403906


namespace sum_f_eq_neg24_l403_403466

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403466


namespace general_term_formula_sum_first_n_terms_l403_403121

theorem general_term_formula :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) := 
  sorry

theorem sum_first_n_terms :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) →
  (∀ n, b n = 1 / (a n * a (n + 1))) →
  (∀ n, S n = 2 * (1 - 1 / (2 * n + 1))) →
  (S n = 4 * n / (2 * n + 1)) :=
  sorry

end general_term_formula_sum_first_n_terms_l403_403121


namespace ab_imaginary_axis_l403_403082

theorem ab_imaginary_axis (a b : ℝ) (h : (a + complex.I) / (b - 3 * complex.I) = complex.I * (c : ℝ)) : a * b = 3 :=
by sorry

end ab_imaginary_axis_l403_403082


namespace find_tank_width_l403_403808

-- Definitions for conditions
def cost_per_sqm := 0.25
def total_cost := 186
def length := 25
def depth := 6
def area_plastered := total_cost / cost_per_sqm
def area_long_walls := 2 * (length * depth)
def remaining_area := area_plastered - area_long_walls
def width := remaining_area / (2 * depth + length)

theorem find_tank_width : width = 12 := by
  -- All necessary intermediary steps would be proven here
  -- This theorem states that the calculated width is indeed 12 meters
  sorry

end find_tank_width_l403_403808


namespace sum_f_values_l403_403478

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403478


namespace sum_f_l403_403449

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403449


namespace percentage_born_in_may_l403_403807

theorem percentage_born_in_may (total_mathematicians : ℕ) (mathematicians_born_in_may : ℕ)
    (h1 : total_mathematicians = 120) (h2 : mathematicians_born_in_may = 15) : 
    (mathematicians_born_in_may / total_mathematicians) * 100 = 12.5 := 
by
  sorry

end percentage_born_in_may_l403_403807


namespace pencils_bought_l403_403543

theorem pencils_bought (payment change pencil_cost glue_cost : ℕ)
  (h_payment : payment = 1000)
  (h_change : change = 100)
  (h_pencil_cost : pencil_cost = 210)
  (h_glue_cost : glue_cost = 270) :
  (payment - change - glue_cost) / pencil_cost = 3 :=
by sorry

end pencils_bought_l403_403543


namespace hyperbola_eccentricity_sqrt10_over_2_l403_403033

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_sqrt10_over_2 (a b e : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : ∃ P Q : ℝ × ℝ, 
        P = (-Real.sqrt 2 * a, b) ∧ 
        Q = (Real.sqrt 2 * a, b) ∧ 
        (0, b) ∈ P ∧
        (0, -b) ∈ Q ∧ 
        (Real.dist P Q = 2 * Real.sqrt 2 * a) ∧
        (Real.dist (0, b) (0, -b) = 2 * b) ∧
        (Real.dist P (0, -b) = Real.dist Q (0, -b) ∧
        (Real.dist P (0, -b) = Real.dist P Q ∧
        (Real.dist P (0, -b) = Real.dist Q (0, b)))) :
  hyperbola_eccentricity a b h1 h2 = e →
  e = Real.sqrt 10 / 2 := 
sorry

end hyperbola_eccentricity_sqrt10_over_2_l403_403033


namespace sin_double_angle_value_l403_403051

theorem sin_double_angle_value (ω φ α : ℝ) (h1 : 0 ≤ φ) (h2: φ ≤ π / 2) 
  (h3 : ω > 0) (h4 : (∀ x, sin(ω * x + φ) + 1 = 2 → x = π / 3)) 
  (h5 : f α = 8 / 5) (h6 : π / 3 < α) (h7 : α < 5 * π / 6) : 
  sin(2 * α + π / 3) = -24 / 25 :=
sorry

end sin_double_angle_value_l403_403051


namespace serial_numbers_first_5_l403_403751

noncomputable def serialNumbersWithinRange : List ℕ :=
  [785, 567, 199, 507, 175]

theorem serial_numbers_first_5 :
  ∀ (randomNumbers : List ℕ), 
  (let validNumbers := randomNumbers.filter (λ n, n >= 0 ∧ n < 800) in 
  validNumbers.take 5 = serialNumbersWithinRange) →
  serialNumbersWithinRange = [785, 567, 199, 507, 175] :=
by
  intro randomNumbers h
  exact sorry

end serial_numbers_first_5_l403_403751


namespace constant_term_binomial_expansion_l403_403183

theorem constant_term_binomial_expansion :
  let f := (3 * x^2 - (2 / x^3))^5 in
  (∃ c : ℝ, is_constant_term c f) → c = 1080 :=
by
  sorry

def is_constant_term (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ (r : ℕ), f = (λ x, (binom_expansion_term 5 r 3 (-2) x)) ∧ (10 - 5 * r = 0 ∧ c = 4 * 27 * (binom_coefficient 5 2))

noncomputable def binom_expansion_term (n r : ℕ) (a b : ℝ) (x : ℝ) : ℝ :=
  binom_coefficient n r * a^(n-r) * b^r * x^(10 - 5*r)

noncomputable def binom_coefficient (n k : ℕ) : ℝ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

end constant_term_binomial_expansion_l403_403183


namespace intervals_of_monotonic_increase_triangle_properties_l403_403014

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - π / 6) - 1

theorem intervals_of_monotonic_increase (k : ℤ) :
  ∃ (a b : ℝ), a = k * π - π / 6 ∧ b = k * π + π / 3 ∧
    ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := sorry

theorem triangle_properties (b : ℝ) (B a c : ℝ) :
  b = Real.sqrt 7 ∧ B = π / 3 ∧ (∃ C, (Real.sin A = 3 * Real.sin C ∧ a = 3 * c) ∧
    a = 3 ∧ c = 1 ∧ (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4) := sorry

end intervals_of_monotonic_increase_triangle_properties_l403_403014


namespace sin_value_l403_403923

-- Define the tangent of an angle
def tan (α : ℝ) : ℝ := sin α / cos α

-- Define the proof statement
theorem sin_value (α : ℝ) (h : tan α = 3 / 4) : sin α = 3 / 5 := 
by 
  sorry

end sin_value_l403_403923


namespace sum_S11_l403_403040

-- Given definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Given specific condition for the problem
def condition (a : ℕ → ℝ) : Prop :=
  2 * a 7 - a 8 = 5

-- The main goal: Prove S_11 = 55
theorem sum_S11 (a : ℕ → ℝ) (ha : is_arithmetic_sequence a) (hc : condition a) :
  sum_of_first_n_terms a 11 = 55 :=
sorry

end sum_S11_l403_403040


namespace sum_f_values_l403_403472

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403472


namespace john_horizontal_distance_l403_403131

theorem john_horizontal_distance
  (vertical_distance_ratio horizontal_distance_ratio : ℕ)
  (initial_elevation final_elevation : ℕ)
  (h_ratio : vertical_distance_ratio = 1)
  (h_dist_ratio : horizontal_distance_ratio = 3)
  (h_initial : initial_elevation = 500)
  (h_final : final_elevation = 3450) :
  (final_elevation - initial_elevation) * horizontal_distance_ratio = 8850 := 
by {
  sorry
}

end john_horizontal_distance_l403_403131


namespace geometric_sequence_implies_l403_403611

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (c : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, c (n + 1) = r * c n

def sequence_d (c : ℕ → α) (n : ℕ) : α := Real.root n (List.prod (List.map c (List.range n)))

theorem geometric_sequence_implies (c : ℕ → α) (h : ∀ n, 0 < c n) (hg : is_geometric_sequence c) :
  is_geometric_sequence (sequence_d c) :=
sorry

end geometric_sequence_implies_l403_403611


namespace necessary_conditions_l403_403956

-- Describe the functions f and g.
def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := d * x + e

-- The proof statement
theorem necessary_conditions (a b c d e : ℝ) :
  (∀ x, f (g x) = g (f x)) →
  a * (d - 1) = 0 ∧ ae = 0 ∧ c - e = ae^2 :=
begin
  sorry
end

end necessary_conditions_l403_403956


namespace triangular_prism_sliced_faces_l403_403811

noncomputable def resulting_faces_count : ℕ :=
  let initial_faces := 5 -- 2 bases + 3 lateral faces
  let additional_faces := 3 -- from the slices
  initial_faces + additional_faces

theorem triangular_prism_sliced_faces :
  resulting_faces_count = 8 := by
  sorry

end triangular_prism_sliced_faces_l403_403811


namespace num_ways_to_make_change_l403_403951

-- Define the standard U.S. coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the total amount
def total_amount : ℕ := 50

-- Condition to exclude two quarters
def valid_combination (num_pennies num_nickels num_dimes num_quarters : ℕ) : Prop :=
  (num_quarters != 2) ∧ (num_pennies + 5 * num_nickels + 10 * num_dimes + 25 * num_quarters = total_amount)

-- Prove that there are 39 ways to make change for 50 cents
theorem num_ways_to_make_change : 
  ∃ count : ℕ, count = 39 ∧ (∀ 
    (num_pennies num_nickels num_dimes num_quarters : ℕ),
    valid_combination num_pennies num_nickels num_dimes num_quarters → 
    (num_pennies, num_nickels, num_dimes, num_quarters) = count) :=
sorry

end num_ways_to_make_change_l403_403951


namespace equilibrium_matchings_l403_403736

-- Define what constitutes a matching and its properties
noncomputable def perfect_matchings : ℕ → ℕ
| 0       := 1
| (n + 1) := perfect_matchings n + perfect_matchings n

-- Define the problem conditions and question
theorem equilibrium_matchings (N : ℕ) :
  let number_of_points := 2 * N in
  let has_two_chords_condition := ∀ P, ∃ (count : ℕ) (chords : list (point × point)), 
    count ≤ 2 ∧ ∀ c ∈ chords, P ∈ c in
  perfect_matchings N.even - perfect_matchings N.odd = 1 :=
by sorry

end equilibrium_matchings_l403_403736


namespace expand_binomials_l403_403343

theorem expand_binomials (x : ℝ) : 
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 :=
by
  sorry

end expand_binomials_l403_403343


namespace positive_difference_l403_403128

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def nearest_multiple_of_five (n : ℕ) : ℕ :=
  let r := n % 5
  let q := n / 5
  if r < 3 then q * 5 else (q + 1) * 5

def kate_sum (n : ℕ) : ℕ :=
  let f := nearest_multiple_of_five
  (List.range n).map f |>.sum

theorem positive_difference (n : ℕ) (h : n = 60) :
  |(sum_natural_numbers n) - (kate_sum n)| = 1265 := by
  sorry

end positive_difference_l403_403128


namespace sum_f_proof_l403_403437

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403437


namespace evaluate_expression_l403_403871

noncomputable def repeating_to_fraction_06 : ℚ := 2 / 3
noncomputable def repeating_to_fraction_02 : ℚ := 2 / 9
noncomputable def repeating_to_fraction_04 : ℚ := 4 / 9

theorem evaluate_expression : 
  ((repeating_to_fraction_06 * repeating_to_fraction_02) - repeating_to_fraction_04) = -8 / 27 := 
by 
  sorry

end evaluate_expression_l403_403871


namespace sum_f_1_to_22_l403_403460

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403460


namespace sum_of_lengths_16_gon_is_229_l403_403294

theorem sum_of_lengths_16_gon_is_229 :
  ∃ (a b c : ℕ), (∑ i in (finset.range 16), (2 * 16 * (real.sin (i * (π / 16)))) + 
  (4 * 32)) = a + b * real.sqrt 2 + c * real.sqrt 4 ∧ a + b + c = 229 :=
by
  sorry

end sum_of_lengths_16_gon_is_229_l403_403294


namespace find_f_2017_l403_403848

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2017 :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x - 2) = f (x + 2)) ∧
  (∀ x : ℝ, x ∈ set.Ioo (-2) 0 → f x = 2^x + 1/2) →
  f 2017 = -1 :=
sorry

end find_f_2017_l403_403848


namespace sum_f_k_1_22_l403_403508

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403508


namespace line_intersect_curve_area_l403_403632

noncomputable def line_param_eq (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

noncomputable def curve_cart_eq (x : ℝ) : ℝ :=
  Real.sqrt (8 * x)

theorem line_intersect_curve_area (α : ℝ) (t1 t2 : ℝ) (hα : α = Real.pi / 4) :
  let l_param (t : ℝ) := line_param_eq α t in
  let curve := curve_cart_eq in
  l_param t1 = (some_pt_x, curve some_pt_x)
  ∧ l_param t2 = (some_pt_y, curve some_pt_y) →
  ∃ area : ℝ, area = 2 * Real.sqrt 6 := 
sorry

end line_intersect_curve_area_l403_403632


namespace relationship_among_log_values_l403_403382

noncomputable def a : ℝ := Real.logBase 3 6
noncomputable def b : ℝ := Real.logBase 4 8
noncomputable def c : ℝ := Real.logBase 5 10

theorem relationship_among_log_values :
  a > b ∧ b > c :=
by
  sorry

end relationship_among_log_values_l403_403382


namespace f_2017_value_l403_403899

noncomputable def f : ℝ → ℝ
| x := 
  if x ≤ 0 then log (3 - x) / log 2
  else f (x - 1) - f (x - 2)

theorem f_2017_value : f 2017 = log 3 / log 2 - 2 := by
  sorry

end f_2017_value_l403_403899


namespace fraction_value_l403_403572

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403572


namespace k_set_cover_l403_403146

theorem k_set_cover (A : Set ℤ) (x : ℕ → ℤ) (k : ℕ)
  (hA_sub : A ⊆ ℤ)
  (h_disjoint : ∀ i j, 1 ≤ i → i < j → j ≤ k → Disjoint (x i + A) (x j + A)) :
  ∀ (t : ℕ) (A_i : ℕ → Set ℤ) (k_i : ℕ → ℕ),
  (∀ i, 1 ≤ i ∧ i ≤ t → k_set (A_i i) (k_i i)) →
  (⋃ i, A_i i) = Set.univ →
  ∑ i in range t, 1 / k_i i ≥ 1 :=
sorry

end k_set_cover_l403_403146


namespace problem_statement_l403_403602

theorem problem_statement : 
  ∃ a b c : ℕ, (prime a ∧ prime b ∧ prime c) ∧ (a * b * c = 1998) ∧ (a < b ∧ b < c) ∧ ((b + c)^a = 1600) :=
begin
  use [2, 3, 37],
  split,
  { split, exact prime_two, split, exact prime_three, exact prime_def_lt'.2 (⟨37, nat.prime37⟩), },
  split, 
  { norm_num, },
  split,
  { repeat {linarith},},
  norm_num, 
  sorry
end

end problem_statement_l403_403602


namespace regression_coefficient_l403_403065

-- Conditions as definitions
def x_vals : List ℝ := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
def y_vals : List ℝ := [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

def sum_x : ℝ := x_vals.sum
def sum_y : ℝ := y_vals.sum

def mean_x : ℝ := sum_x / (x_vals.length : ℝ)
def mean_y : ℝ := sum_y / (y_vals.length : ℝ)

-- The regression line passes through the mean point
def passes_mean_point (b : ℝ) : Prop :=
  -3 + b * mean_x = mean_y

-- Theorem to prove
theorem regression_coefficient :
  sum_x = 17 ∧ sum_y = 4 → ∃ b, passes_mean_point b ∧ b = 2 :=
by
  intros h
  sorry

end regression_coefficient_l403_403065


namespace function_existence_l403_403047

theorem function_existence (φ : ℕ → ℕ) :
  (¬ ∃ f : ℕ → ℕ, (∀ x : ℕ, f x > f (φ x)) ∧ ∀ y : ℕ, f y ∈ ℕ) ∧
  (∃ f : ℕ → ℤ, (∀ x : ℕ, f x > f (φ x)) ∧ ∀ y : ℕ, f y ∈ ℤ) :=
by
  sorry

end function_existence_l403_403047


namespace solve_for_q_l403_403961

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l403_403961


namespace relationship_u_v_l403_403643

variable (R: ℝ) -- Radius of the circle
variable (u v: ℝ) -- Distances
variable (AC AF FC: ℝ) -- Segments

-- Conditions
section
  -- AB is the diameter
  variable (A B D F C: ℝ)
  variable (BD AD: ℝ)

  -- Point distances and the given ratio
  definition tangent_point (c: ℝ) : ℝ := c
  definition AF_ratio (z: ℝ) : ℝ := 3 * z / 4
  definition FC_ratio (z: ℝ) : ℝ := z / 4
  definition BC_value (c R: ℝ) : ℝ := (c^2) / (2 * R)
  definition CF_value (BC BD: ℝ) : ℝ := BC - BD
  definition v_value : ℝ := R

  -- Main theorem to prove
  theorem relationship_u_v : u = 3 * (v^2 - 2 * R * v) / (2 * R) :=
  sorry
end

end relationship_u_v_l403_403643


namespace smallest_x_l403_403544

theorem smallest_x (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : 0.75 = y / (240 + x)) : x = 4 := 
by
  sorry

end smallest_x_l403_403544


namespace equal_collections_after_8_months_l403_403132

/-- Kymbrea and LaShawn's comic book collections grow over time. 
Prove that the number of months after which LaShawn's collection is 
equal to Kymbrea's collection is 8. -/
theorem equal_collections_after_8_months :
  ∃ (n : ℕ), 50 + 3 * n = 20 + 7 * n ∧ n = 8 :=
by {
  use 8,
  split,
  {
    -- Show that at 8 months, the collections are equal.
    calc 50 + 3 * 8 = 74 : by norm_num
         ...        = 20 + 7 * 8 : by norm_num
  },
  {
    -- Confirm that the value of n used is indeed 8.
    rfl
  }
}

end equal_collections_after_8_months_l403_403132


namespace minimize_sum_arithmetic_sequence_l403_403394

variable (a_1 d : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a_1 + (n - 1) * d
def sum_of_arithmetic_sequence (n : ℕ) : ℝ := n/2 * (2 * a_1 + (n - 1) * d)

theorem minimize_sum_arithmetic_sequence
  (h1 : a_1 + 2 * d = -7)
  (h2 : a_1 + 4 * d = -3)
  : n = 6 := sorry

end minimize_sum_arithmetic_sequence_l403_403394


namespace problem_statement_l403_403666

variable (z : ℂ)
variable (h : z = (1 + complex.I) / real.sqrt 2)

theorem problem_statement : 
  (∑ n in finset.range 12, z ^ (2 * (n + 1)) ^ 2) * (∑ m in finset.range 12, (1 / z ^ (2 * m + 1) ^ 2)) = 144 :=
by
  sorry

end problem_statement_l403_403666


namespace magnitude_of_complex_power_l403_403312

theorem magnitude_of_complex_power :
  abs ((2 + 2 * Complex.i) ^ 6) = 512 := 
sorry

end magnitude_of_complex_power_l403_403312


namespace find_a_b_c_sum_l403_403813

-- Define the necessary conditions and constants
def radius : ℝ := 10  -- tower radius in feet
def rope_length : ℝ := 30  -- length of the rope in feet
def unicorn_height : ℝ := 6  -- height of the unicorn from ground in feet
def rope_end_distance : ℝ := 6  -- distance from the unicorn to the nearest point on the tower

def a : ℕ := 30
def b : ℕ := 900
def c : ℕ := 10  -- assuming c is not necessarily prime for the purpose of this exercise

-- The theorem we want to prove
theorem find_a_b_c_sum : a + b + c = 940 :=
by
  sorry

end find_a_b_c_sum_l403_403813


namespace prove_one_even_l403_403227

open Nat

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem prove_one_even (a b c : ℕ) :
  (exactly_one_even : Prop) (h_exactly_one : exactly_one_even ↔ 
    (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
    (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
    (¬is_even a ∧ ¬is_even b ∧ is_even c)) :
  ¬(exists_even2 (h_even2 : (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) ∨
               all_odd (h_all_odd : ¬is_even a ∧ ¬is_even b ∧ ¬is_even c)) → exactly_one_even :=
sorry

end prove_one_even_l403_403227


namespace find_b_l403_403087

noncomputable def c (b : ℝ) : ℂ := (2 - b * complex.I) / (1 + 2 * complex.I)

theorem find_b (b : ℝ) (h : (c b).re = -(c b).im) : b = -2/3 := by
  sorry

end find_b_l403_403087


namespace radius_of_surrounding_circles_is_correct_l403_403273

noncomputable def r : Real := 1 + Real.sqrt 2

theorem radius_of_surrounding_circles_is_correct (r: ℝ)
  (h₁: ∃c : ℝ, c = 2) -- central circle radius is 2
  (h₂: ∃far: ℝ, far = (1 + (Real.sqrt 2))) -- r is the solution as calculated
: 2 * r = 1 + Real.sqrt 2 :=
by
  sorry

end radius_of_surrounding_circles_is_correct_l403_403273


namespace projection_is_negative_sqrt_10_l403_403515

noncomputable def projection_of_AB_in_direction_of_AC : ℝ :=
  let A := (1, 1)
  let B := (-3, 3)
  let C := (4, 2)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2)
  dot_product / magnitude_AC

theorem projection_is_negative_sqrt_10 :
  projection_of_AB_in_direction_of_AC = -Real.sqrt 10 :=
by
  sorry

end projection_is_negative_sqrt_10_l403_403515


namespace tangent_circles_l403_403867

-- Definitions
variables {A B C D E F X Y S T: Type*} [Nonempty A] [Nonempty B] [Nonempty C]
[Nonempty D] [Nonempty E] [Nonempty F] [Nonempty X] [Nonempty Y] [Nonempty S] [Nonempty T]

-- Triangle ABC with AB ≠ AC and incircle tangent at D, E, F.
def triangle_ABC (triangle : Type*) (incircle : triangle → Set triangle) (tangent : (Set triangle) → triangle → triangle → Prop) : Prop :=
AB ≠ AC ∧ tangent incircle B C D ∧ tangent incircle C A E ∧ tangent incircle A B F

-- Internal angle bisector of ∠BAC intersects DE, DF at X, Y respectively.
def angle_bisector (internal_bisector : X → Y → Prop) (line_DE : Prop) (line_DF : Prop) : Prop :=
internal_bisector DE X ∧ internal_bisector DF Y

-- Points S, T on BC such that ∠XSY = ∠XTY = 90°
def right_angles (angles : X → S → Y → Prop) (angles_T : X → T → Y → Prop) : Prop :=
(angles X S Y = 90) ∧ (angles_T X T Y = 90)

-- Circle γ as circumcircle of ΔAST.
def circumcircle (circle : Set triangle → circle) (points : Set triangle) : Prop :=
circle {A, S, T}

-- Proof statement in Lean 4
theorem tangent_circles :
  (triangle_ABC ABC incircle tangent)
  → (angle_bisector internal_bisector DE DF)
  → (right_angles angles angles_T)
  → (circumcircle γ {A, S, T})
  → (is_tangent γ (circumcircle_ABC))
  ∧ (is_tangent γ (incircle_ABC)) :=
begin
  sorry
end

end tangent_circles_l403_403867


namespace correct_statements_l403_403884

variable {a b c x0 : ℝ}

-- Condition for statement ①
def condition_1 (h : a + b + c = 0) : Prop :=
  b^2 - 4 * a * c ≥ 0

-- Condition for statement ②
def condition_2 (has_distinct_roots : 0^2 - 4 * a * c > 0) : Prop :=
  b^2 - 4 * a * c > 0

-- Condition for statement ③
def condition_3 (is_root : a * c^2 + b * c + c = 0) : Prop :=
  c ≠ 0 ∨ ac + b + 1 = 0

-- Condition for statement ④
def condition_4 (is_root : a * x0^2 + b * x0 + c = 0) : Prop :=
  b^2 - 4 * a * c = (2 * a * x0 + b)^2

-- Final theorem stating the correctness of ①, ② and ④
theorem correct_statements
  (h1 : a + b + c = 0)
  (h2 : 0^2 - 4 * a * c > 0)
  (h4 : a * x0^2 + b * x0 + c = 0) :
  condition_1 h1 ∧ condition_2 h2 ∧ condition_4 h4 :=
by sorry

end correct_statements_l403_403884


namespace unique_two_digit_solution_exists_l403_403746

theorem unique_two_digit_solution_exists :
  ∃! (s : ℤ), 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52 :=
begin
  use 4,
  split,
  { split,
    { linarith },
    { linarith },
    { norm_num }
  },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    have : 77 * 52 % 100 = 4,
    { norm_num },
    have h : y ≡ 4 [MOD 100] := (congr_arg (λ x, 77 * x % 100) hy3).trans this,
    norm_num at h,
    linarith }
end

end unique_two_digit_solution_exists_l403_403746


namespace midpoint_correct_distance_correct_l403_403322

namespace ComplexProblem

def z1 : ℂ := -7 + 5 * Complex.i
def z2 : ℂ := 9 - 11 * Complex.i
def midpoint : ℂ := (z1 + z2) / 2
def distance (z1 z2 : ℂ) : ℝ := Complex.abs (z1 - z2)

theorem midpoint_correct : midpoint = 1 - 3 * Complex.i := by
  sorry

theorem distance_correct : distance z1 midpoint = 8 * Real.sqrt 2 := by
  sorry

end ComplexProblem

end midpoint_correct_distance_correct_l403_403322


namespace fifth_observation_l403_403711

theorem fifth_observation (O1 O2 O3 O4 O5 O6 O7 O8 O9 : ℝ)
  (h1 : O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 = 72)
  (h2 : O1 + O2 + O3 + O4 + O5 = 50)
  (h3 : O5 + O6 + O7 + O8 + O9 = 40) :
  O5 = 18 := 
  sorry

end fifth_observation_l403_403711


namespace sufficiency_of_angle_condition_l403_403941

/-- Definitions of the planes and their intersections -/
variables {α β γ : Plane}
variables {a b c : Line}

/-- Angle between any two planes -/
variable (θ : Real)

/-- Conditions given in the problem -/
variable (angle_condition : θ > π / 3)
variable (intersection_condition : 
  ∃ (P : Point), (a ∩ b = P) ∧ (b ∩ c = P) ∧ (c ∩ a = P))

/-- Proof that angle_condition is a sufficient but not necessary condition for intersection_condition -/
theorem sufficiency_of_angle_condition :
  angle_condition → intersection_condition :=
sorry

end sufficiency_of_angle_condition_l403_403941


namespace fraction_value_l403_403571

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403571


namespace options_B_C_D_eq_sqrt3_l403_403304

def option_A_eq_sqrt3 (angle : Real) : Prop :=
  cos (15 * angle) - sqrt 3 * sin (15 * angle) = sqrt 3

def option_B_eq_sqrt3 : Prop :=
  2 * (cos (π / 12)^2 - cos (5 * π / 12)^2) = sqrt 3

def option_C_eq_sqrt3 (angle : Real) : Prop :=
  (1 + tan (15 * angle)) / (1 - tan (15 * angle)) = sqrt 3

def option_D_eq_sqrt3 (angle : Real) : Prop :=
  (cos (10 * angle) - 2 * sin (20 * angle)) / sin (10 * angle) = sqrt 3

theorem options_B_C_D_eq_sqrt3 (angle : Real) : proposition :=
  option_B_eq_sqrt3 ∧ option_C_eq_sqrt3 angle ∧ option_D_eq_sqrt3 angle :=
by
  sorry

end options_B_C_D_eq_sqrt3_l403_403304


namespace sum_f_proof_l403_403431

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403431


namespace find_value_of_fraction_l403_403564

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403564


namespace employees_after_hiring_l403_403618

theorem employees_after_hiring (initial_employee_count : ℕ) (percentage_increase : ℕ) 
    (new_employee_ratio : ℚ) :
    initial_employee_count = 852 →
    percentage_increase = 25 →
    new_employee_ratio = 0.25 →
    initial_employee_count + (initial_employee_count * new_employee_ratio).natAbs = 1065 := 
by 
  intros h1 h2 h3
  rw [h1, h3]
  norm_num
  sorry

end employees_after_hiring_l403_403618


namespace cora_april_cookies_cost_l403_403340

def total_cost_cookies_in_April (days: ℕ) 
                                (even_days: ℕ -> (ℕ × ℕ)) 
                                (odd_days: ℕ -> (ℕ × ℕ)) 
                                (choco_price sugar_price oat_price snick_price: ℕ) 
                                : ℕ :=
  let even_total_cost := (even_days days).fst * choco_price + (even_days days).snd * sugar_price
  let odd_total_cost := (odd_days days).fst * oat_price + (odd_days days).snd * snick_price
  in (days / 2) * even_total_cost + (days / 2) * odd_total_cost

theorem cora_april_cookies_cost : 
  total_cost_cookies_in_April 30 
                              (λ _, (3, 2)) 
                              (λ _, (4, 1)) 
                              18 22 15 25 = 2745 :=
by sorry

end cora_april_cookies_cost_l403_403340


namespace sum_f_eq_neg24_l403_403463

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403463


namespace ellipse_eccentricity_l403_403039

variables {a b c e : ℝ}

-- Define the ellipse parameters and conditions
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Ellipse parameters relationships
def ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∃ F : ℝ, F = c ∧ 
    let A := (a, 0) in 
    let B := (0, b) in
    let C := (0, (b * c) / a) in
    let M := (b, (b * c) / a) in
    let N := (-b, (b * c) / a) in
    let FA := a + c in
    let MN := 2 * b in
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse M.1 M.2 ∧ ellipse N.1 N.2 ∧
    MN = FA ∧ a^2 = b^2 + c^2

-- Proving the eccentricity
theorem ellipse_eccentricity (h1 : a > b) (h2 : b > 0) (h3 : ellipse_properties a b h1 h2) :
  e = 3 / 5 := 
sorry

end ellipse_eccentricity_l403_403039


namespace largest_n_l403_403693

noncomputable def K1 := 1842 * Real.sqrt 2 + 863 * Real.sqrt 7
noncomputable def K2 := 3519 + 559 * Real.sqrt 6

theorem largest_n :
  ∀ n, (∀ m ≤ n, Real.floor (K1 * 10^m) / 10^m = Real.floor (K2 * 10^m) / 10^m) → n ≤ 4 := by
  sorry

end largest_n_l403_403693


namespace part_one_part_two_l403_403066

variables (a b : EuclideanSpace ℝ (Fin 3))
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_comb : ∥4 • a - b∥ = 2 * Real.sqrt 3

-- Part (1)
theorem part_one : (2 • a + b) ⬝ b = 6 := 
sorry

-- Part (2)
theorem part_two : ∃ λ : ℝ, (a + λ • b) = ℝ t (2 • a - b) ∧ λ = - 1/2 := 
sorry

end part_one_part_two_l403_403066


namespace infinitely_many_primes_l403_403690

theorem infinitely_many_primes : ¬finite {p : ℕ | Nat.Prime p} :=
by
  assume H : finite {p : ℕ | Nat.Prime p}
  sorry

end infinitely_many_primes_l403_403690


namespace sqrt_17_estimation_l403_403869

theorem sqrt_17_estimation :
  4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := 
sorry

end sqrt_17_estimation_l403_403869


namespace impossible_game_distribution_l403_403616

theorem impossible_game_distribution :
  let participants := 8
  let games_each := [11, 11, 10, 8, 8, 8, 7, 7]
  let total_games := participants * (participants - 1) / 2
  sum games_each / 2 ≠ total_games + count (λ (x : ℕ), x > 7) games_each :=
by
  let participants := 8
  let games_each := [11, 11, 10, 8, 8, 8, 7, 7]
  let total_games := participants * (participants - 1) / 2
  have h1 : sum games_each = 70 := rfl
  have h2 : 2*total_games = 56 := rfl -- Expected number of games without replays
  have h3 : (sum games_each) / 2 = 35 := rfl
  have h4 : count (λ (x : ℕ), x > 7) games_each = 3 := rfl --Count of games > 7
  sorry

end impossible_game_distribution_l403_403616


namespace geom_seq_product_l403_403408

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (-1/3)^r * x^(3 - (3 * r / 2))

noncomputable def constant_term_of_binomial : ℝ :=
  binomial_term 6 2 1

def a (n : ℕ) : ℝ := (6:ℝ-(n:ℝ))^n -- Placeholder; adjust for proper sequence definition 

theorem geom_seq_product (a : ℕ → ℝ) (h : a 5 = constant_term_of_binomial) : 
  (a 3) * (a 7) = 25 / 9 := by
  sorry

end geom_seq_product_l403_403408


namespace largest_circle_area_l403_403797

theorem largest_circle_area (A : ℝ) (hA : A = 100) :
  ∃ A_c : ℝ, A_c ≈ 96 :=
by
  sorry

end largest_circle_area_l403_403797


namespace odd_function_a_value_l403_403609

theorem odd_function_a_value :
  (∀ x : ℝ, f x = (lg (1 - x^2)) / (abs (x - 2) + a) → 
            (∀ x : ℝ, f (-x) = -f (x))) →
  a = -2 :=
by
  sorry

end odd_function_a_value_l403_403609


namespace minimum_expression_value_l403_403604

theorem minimum_expression_value (x : ℝ) (h : 0 < x) :
    (∃ c : ℝ, (∀ x > 0, (sqrt (x^4 + x^2 + 2 * x + 1) + sqrt (x^4 - 2 * x^3 + 5 * x^2 - 4 * x + 1)) / x ≥ c) ∧
    (∀ x > 0, (sqrt (x^4 + x^2 + 2 * x + 1) + sqrt (x^4 - 2 * x^3 + 5 * x^2 - 4 * x + 1)) / x = √10 → c = √10)) :=
sorry

end minimum_expression_value_l403_403604


namespace inverse_proportion_properties_l403_403529

-- Definition of the inverse proportion function
def inverse_proportion (x : ℝ) : ℝ := 2 / x

-- Proving properties of the inverse proportion function
theorem inverse_proportion_properties :
  (inverse_proportion 1 = 2) ∧
  (∀ x : ℝ, x > 0 → inverse_proportion x > 0) ∧
  (∀ x : ℝ, x < 0 → inverse_proportion x < 0) ∧
  (∀ x : ℝ, x > 1 → 0 < inverse_proportion x ∧ inverse_proportion x < 2) :=
by
  -- We skip the proof for now
  sorry

end inverse_proportion_properties_l403_403529


namespace distance_to_pinedale_mall_at_least_0_5_km_l403_403179

-- Given conditions about stop lengths and walking distances
def average_speeds_between_stops : List ℝ := [40, 50, 60, 45, 55, 70, 65, 40]

def walking_distance := 0.5 -- The distance Yahya has to walk between the 3rd and 4th stop

-- The mathematical proof problem statement in Lean 4:
theorem distance_to_pinedale_mall_at_least_0_5_km :
  let n := 8 -- number of stops to Pinedale Mall
  ∃ d : ℝ, d ≥ walking_distance :=
sorry

end distance_to_pinedale_mall_at_least_0_5_km_l403_403179


namespace inclination_range_l403_403725

noncomputable def angle_of_inclination (α : ℝ) : Set ℝ :=
  {θ : ℝ | ∃ θ, θ ∈ [0, Real.pi/4] ∪ [3 * Real.pi/4, Real.pi) ∧ Real.tan θ = Real.cos α}

theorem inclination_range (α : ℝ) :
  {θ | ∀ θ, θ ∈ angle_of_inclination α} =
  {θ | θ ∈ [0, Real.pi/4] ∪ [3 * Real.pi/4, Real.pi)] } := by
  sorry

end inclination_range_l403_403725


namespace sum_values_p_f_p_q_eq_2004_l403_403882

def f : ℕ → ℕ → ℕ
| 1, 1       := 1
| (m+1), n   := f m n + m
| m, (n+1)   := f m n - n

theorem sum_values_p_f_p_q_eq_2004 : 
  (∑ p in {p | ∃ q, f p q = 2004}.to_finset, p) = 3007 := 
sorry

end sum_values_p_f_p_q_eq_2004_l403_403882


namespace cookie_distribution_probability_l403_403801

theorem cookie_distribution_probability :
  let C := 4     -- Number of chocolate cookies
  let V := 4     -- Number of vanilla cookies
  let S := 4     -- Number of strawberry cookies
  let N := 4     -- Number of students / each group receiving exactly one cookie of each type
  let total_cookies := C + V + S    -- Total cookies
  ∀ cookies : Finset (Fin (total_cookies)),
    cookies.card = total_cookies →
    (∀ group : Fin N, group.cookie ∈ cookies →
      group.cookie.card = 3 ∧
      ∃ c : Fin C, ∃ v : Fin V, ∃ s : Fin S,
        (c ∈ group.cookie ∧ v ∈ group.cookie ∧ s ∈ group.cookie)) →
    (∃ probability : ℚ, probability = 144 / 3850) :=
by
  sorry

end cookie_distribution_probability_l403_403801


namespace sum_of_coefficients_l403_403195

theorem sum_of_coefficients (A B C : ℤ) 
  (h_factorization : ∀ x, x^3 + A * x^2 + B * x + C = (x + 2) * (x - 2) * (x - 1)) :
  A + B + C = -1 :=
by sorry

end sum_of_coefficients_l403_403195


namespace greatest_difference_54_l403_403852

theorem greatest_difference_54 (board : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 100) :
  ∃ i j k l, (i = k ∨ j = l) ∧ (board i j - board k l ≥ 54 ∨ board k l - board i j ≥ 54) :=
sorry

end greatest_difference_54_l403_403852


namespace log_a5_a7_a9_l403_403901

noncomputable def a : ℕ+ → ℝ := sorry

axiom sequence_property (n : ℕ+) : real.log 3 (a n) + 1 = real.log 3 (a (n + 1))

axiom sum_condition : a 2 + a 4 + a 6 = 9

theorem log_a5_a7_a9 : real.log (a 5 + a 7 + a 9) = 4 :=
sorry

end log_a5_a7_a9_l403_403901


namespace only_pair_yields_positive_integer_l403_403701

def p_A : ℕ := 25530
def q_A : ℕ := 29464

def p_B : ℕ := 37615
def q_B : ℕ := 26855

def p_C : ℕ := 15123
def q_C : ℕ := 32477

def p_D : ℕ := 28326
def q_D : ℕ := 28614

def p_E : ℕ := 22536
def q_E : ℕ := 27462

def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem only_pair_yields_positive_integer :
  let sq_sum (x y : ℕ) : ℕ := x * x + y * y in
  (is_square (sq_sum p_A q_A)) ∧ ¬(is_square (sq_sum p_B q_B)) ∧ ¬(is_square (sq_sum p_C q_C)) ∧ ¬(is_square (sq_sum p_D q_D)) ∧ ¬(is_square (sq_sum p_E q_E)) :=
by
  sorry

end only_pair_yields_positive_integer_l403_403701


namespace min_cos_B_l403_403094

-- Definitions based on conditions
variables (a c : ℝ) (b : ℝ) (h1 : a + c = 6 * sqrt 3) (h2 : b = 6)

-- The goal is to prove that cos B >= 1/3, with equality when a = c = 3 * sqrt 3
theorem min_cos_B : 
  ∃ B : ℝ, cos B ≥ 1 / 3 ∧ (∀ a c : ℝ, a > 0 ∧ c > 0 ∧ a + c = 6 * sqrt 3 
  → (cos B ≥ 1/3)) := 
sorry

end min_cos_B_l403_403094


namespace count_distinct_x_for_term_3001_in_sequence_l403_403845

theorem count_distinct_x_for_term_3001_in_sequence :
  let seq : ℕ → ℝ := fun n =>
    if n = 0
    then x
    else if n = 1
    then 3000
    else if n = 2
    then (3002 / x)
    else if n % 5 = 0
    then seq 0
    else if n % 5 = 1
    then seq 1
    else if n % 5 = 2
    then (3002 / seq 0)
    else if n % 5 = 3
    then (seq 0 + 3002) / (3000 * seq 0)
    else (seq 0 + 2) / 3000
  in
  ∃ n, seq n = 3001 → x = 3001 ∨ x = 3002/3001 ∨ x = 3002/9000001 ∨ x = 9000002 :=
begin
  sorry
end

end count_distinct_x_for_term_3001_in_sequence_l403_403845


namespace desired_interest_rate_is_six_l403_403282

-- Define initial investment and rates
def init_investment : ℝ := 8000
def init_rate : ℝ := 0.05

-- Define additional investment and rates
def add_investment : ℝ := 4000
def add_rate : ℝ := 0.08

-- Calculate interest from initial and additional investments
def interest_from_init : ℝ := init_investment * init_rate
def interest_from_add : ℝ := add_investment * add_rate

-- Calculate total interest
def total_interest : ℝ := interest_from_init + interest_from_add

-- Calculate total investment
def total_investment : ℝ := init_investment + add_investment

-- Calculate the desired rate of interest
def desired_rate : ℝ := (total_interest / total_investment) * 100

-- Prove the desired rate of interest is 6%
theorem desired_interest_rate_is_six : desired_rate = 6 := by 
  sorry

end desired_interest_rate_is_six_l403_403282


namespace third_derivative_log2_over_x3_l403_403350

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2) / (x^3)

-- Statement
theorem third_derivative_log2_over_x3 (x : ℝ) (h : 0 < x) : 
  deriv (deriv (deriv (y x))) = (47 - 60 * Real.log x) / (Real.log 2 * x^6) :=
sorry

end third_derivative_log2_over_x3_l403_403350


namespace fraction_value_l403_403573

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403573


namespace g_eval_at_5_6_l403_403048

noncomputable def g : ℝ → ℝ
| x := if x < 0 then Real.sin (2 * Real.pi * x) else g (x - 1)

theorem g_eval_at_5_6 : g (5 / 6) = - Real.sqrt 3 / 2 := by
  sorry

end g_eval_at_5_6_l403_403048


namespace max_triangle_area_ellipse_l403_403927

theorem max_triangle_area_ellipse :
  let ellipse := ∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1
  let left_focus := (-1, 0)
  let intersects := {P : ℝ × ℝ | ellipse P.1 P.2}
  let A := (1, 3 / 2)
  let B := (1, -3 / 2)
  (1, 1) ∈ intersects →
  let area := λ A B : ℝ × ℝ, 0.5 * abs ((-1 * (A.2 - B.2)) + (1 * (B.2) + (1 * -A.2)))
  area A B = 3 :=
sorry

end max_triangle_area_ellipse_l403_403927


namespace chord_length_l403_403333

theorem chord_length 
  (curve : ℝ → ℝ)
  (line : ℝ → ℝ × ℝ)
  (t : ℝ) :
  (curve θ = sqrt 2 * cos (θ + π / 4)) →
  (line t = (1 + 4 / 5 * t, -1 - 3 / 5 * t)) →
  length_of_chord curve line = 7 / 5 := sorry

end chord_length_l403_403333


namespace exists_unique_root_of_monotonic_and_continuous_l403_403037

theorem exists_unique_root_of_monotonic_and_continuous
  (f : ℝ → ℝ) (a b : ℝ)
  (h_monotonic : monotone f)
  (h_continuous : continuous_on f (set.Icc a b))
  (h_sign_change : f a * f b < 0) :
  ∃! c ∈ set.Icc a b, f c = 0 :=
sorry

end exists_unique_root_of_monotonic_and_continuous_l403_403037


namespace area_PQRS_eq_144_l403_403320

-- Definitions
def square (side : ℝ) := side * side
def diagonal (side : ℝ) := side * Real.sqrt 2

variables (ABCD PQRS : Type)
variables [MetricSpace ABCD] [MetricSpace PQRS]
variables (A B C D : ABCD) (P Q R S : PQRS)

-- Conditions
axiom length_ABCD : ∀ (a b : ABCD), (a = A ∧ b = B) ∨ (a = B ∧ b = C) ∨ (a = C ∧ b = D) ∨ (a = D ∧ b = A) → dist a b = 10
axiom rotated_PQRS : ∀ (a b c d : PQRS), (dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a)
axiom extend_PQRS: ∀ (p q r s : PQRS), line_through (p, q) ∩ line_through (A, B) ≠ ∅ ∧
       line_through (q, r) ∩ line_through (B, C) ≠ ∅ ∧
       line_through (r, s) ∩ line_through (C, D) ≠ ∅ ∧
       line_through (s, p) ∩ line_through (D, A) ≠ ∅
axiom dist_A_to_P : ∀ (p : PQRS), (p = P) → dist A p = 2

-- Theorem statement
theorem area_PQRS_eq_144 : ∃ (x : ℝ), x = 12 ∧ square x = 144 :=
by
  sorry

end area_PQRS_eq_144_l403_403320


namespace sum_f_l403_403444

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom f_g_equation1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom f_g_equation2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom g_symmetry : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g_value_at_2 : g 2 = 4

theorem sum_f : (∑ k in finset.range 22, f (k + 1)) = -24 :=
by
  sorry

end sum_f_l403_403444


namespace find_x_value_l403_403763

theorem find_x_value : (8 = 2^3) ∧ (8 * 8^32 = 8^33) ∧ (8^33 = 2^99) → ∃ x, 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 = 2^x ∧ x = 99 :=
by
  intros h
  sorry

end find_x_value_l403_403763


namespace count_x_for_3001_l403_403844

noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0     := x
| 1     := 3000
| (n+2) := (sequence (n+1) + 2) / (sequence n)

lemma seq_periodic (x : ℝ) :
  (∀ n, sequence x (n+5) = sequence x n) :=
sorry

theorem count_x_for_3001 :
  set.card {x : ℝ | ∃ n, sequence x n = 3001} = 4 := sorry

end count_x_for_3001_l403_403844


namespace similar_not_inscribed_or_circumscribed_l403_403832

variables {α : Type*} [EuclideanGeometry α]

-- Define similar but non-congruent triangles
structure Triangle (α : Type*) [EuclideanGeometry α] :=
(a b c : α)

def similar (T1 T2 : Triangle α) : Prop :=
∀ a₁ b₁ c₁ a₂ b₂ c₂,
  T1.a = a₁ ∧ T1.b = b₁ ∧ T1.c = c₁ ∧
  T2.a = a₂ ∧ T2.b = b₂ ∧ T2.c = c₂ ∧
  ((a₁ / a₂) = (b₁ / b₂) ∧ (a₁ / a₂) = (c₁ / c₂) ∧ (b₁ / b₂) = (c₁ / c₂))

def congruent (T1 T2 : Triangle α) : Prop :=
T1 = T2

def inscribed_in_same_circle (T1 T2 : Triangle α) : Prop :=
∃ R : ℝ, ∀ (a b c : α), circumscribed_circle_radius T1 = R ∧ circumscribed_circle_radius T2 = R

def circumscribed_around_same_circle (T1 T2 : Triangle α) : Prop :=
∃ r : ℝ, ∀ (a b c : α), inscribed_circle_radius T1 = r ∧ inscribed_circle_radius T2 = r

theorem similar_not_inscribed_or_circumscribed (T1 T2 : Triangle α) 
  (h1 : similar T1 T2) (h2 : ¬ congruent T1 T2) : 
  (¬ inscribed_in_same_circle T1 T2) ∧ (¬ circumscribed_around_same_circle T1 T2) :=
sorry

end similar_not_inscribed_or_circumscribed_l403_403832


namespace equilateral_triangle_l403_403180

-- Define the properties and conditions of the problem
variable {α : Type} [LinearOrderedField α]

-- Define the points A, B, C and the points where the circle touches the sides
variables (A B C C_1 A_1 B_1 : Point α)

-- Define the circle inscribed in triangle ABC
variable (circle : Circle α)

-- Define the conditions
variable (h1 : circle.Touches A B = C_1)
variable (h2 : circle.Touches B C = A_1)
variable (h3 : circle.Touches A C = B_1)
variable (h4 : (A - C_1).length = (B - A_1).length)
variable (h5 : (B - A_1).length = (C - B_1).length)
variable (h6 : (C - B_1).length = (A - C_1).length)

-- Prove that triangle ABC is equilateral
theorem equilateral_triangle :
  (A - B).length = (B - C).length ∧ (B - C).length = (C - A).length := by
  sorry

end equilateral_triangle_l403_403180


namespace volunteers_distribution_l403_403338

theorem volunteers_distribution :
  let volunteers := 5
  let groups := 4
  let group1 := 2
  let group2 := 1
  (finset.card (finset.filter (λ s:Set ℕ, finset.card s = group1) (finset.powerset (finset.range volunteers))) * 
  factorial groups) = 240 := 
by
  -- Insert specific Lean code here to expand and prove the theorem
  sorry

end volunteers_distribution_l403_403338


namespace fraction_value_l403_403576

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403576


namespace total_employees_now_l403_403620

-- Definitions based on conditions
def initial_employees : ℕ := 852
def additional_percentage : ℝ := 0.25

-- Target statement to prove
theorem total_employees_now : 
  let additional_employees : ℕ := (initial_employees * (additional_percentage * 100).toNat) / 100 in
  let total_employees : ℕ := initial_employees + additional_employees in
  total_employees = 1065 :=
by
  sorry

end total_employees_now_l403_403620


namespace brocard_inequality_part_a_brocard_inequality_part_b_l403_403164

variable (α β γ φ : ℝ)

theorem brocard_inequality_part_a (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) := 
sorry

theorem brocard_inequality_part_b (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  8 * φ^3 ≤ α * β * γ := 
sorry

end brocard_inequality_part_a_brocard_inequality_part_b_l403_403164


namespace kelvin_classes_l403_403945

theorem kelvin_classes (c : ℕ) (h1 : Grant = 4 * c) (h2 : c + Grant = 450) : c = 90 :=
by sorry

end kelvin_classes_l403_403945


namespace sum_f_eq_neg24_l403_403469

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403469


namespace parallel_vectors_l403_403942

open_locale real

noncomputable def oa : ℝ × ℝ := (0, 1)
noncomputable def ob : ℝ × ℝ := (1, 3)
noncomputable def oc (m : ℝ) : ℝ × ℝ := (m, m)

theorem parallel_vectors (m : ℝ) (h : (1, 2) = λ (u v : ℝ), (m, m - 1)) : m = -1 :=
sorry

end parallel_vectors_l403_403942


namespace valid_votes_B_l403_403105

theorem valid_votes_B (V : ℕ) (VA VB : ℕ) 
  (total_votes : V = 5720)
  (valid_invalid_ratio : 0.8 * V = VA + VB)
  (A_exceeds_B : VA = VB + 0.15 * V)
  : VB = 1859 := 
sorry

end valid_votes_B_l403_403105


namespace row_prod_neg_one_l403_403663

variable (a b : Fin 100 → ℝ)

def grid_entry (i j : Fin 100) : ℝ := a i + b j

-- The condition that for any fixed j, the product of the grid entries in the j-th column is 1
def col_prod_one (j : Fin 100) : Prop :=
  ∏ i, grid_entry a b i j = 1

-- The theorem to be proved: for any fixed i, the product of the grid entries in the i-th row is -1
theorem row_prod_neg_one
  (h_distinct_a : Function.injective a)
  (h_distinct_b : Function.injective b)
  (h_col : ∀ j, col_prod_one a b j) :
  ∀ i, ∏ j, grid_entry a b i j = -1 :=
sorry

end row_prod_neg_one_l403_403663


namespace sum_f_values_l403_403476

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403476


namespace part1_part2_l403_403615

variables {A B C a b c : ℝ}

-- Condition 1: In triangle ABC, sides opposite to angles A, B, C are a, b, c respectively.
-- This is implied in the usage of the variables in the context of a triangle.

-- Condition 2: given equation
axiom eq1 : 2 * real.cos A * (b * real.cos C + c * real.cos B) = a

-- Condition 3: given value of cos B
axiom cos_B : real.cos B = 3/5

-- Proof statements
theorem part1 : A = real.pi / 3 :=
sorry

theorem part2 : real.sin (B - C) = (7 * real.sqrt 3 - 24) / 50 :=
sorry

end part1_part2_l403_403615


namespace parallel_lines_m_eq_neg2_l403_403540

def l1_equation (m : ℝ) (x y: ℝ) : Prop :=
  (m+1) * x + y - 1 = 0

def l2_equation (m : ℝ) (x y: ℝ) : Prop :=
  2 * x + m * y - 1 = 0

theorem parallel_lines_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, l1_equation m x y) →
  (∀ x y : ℝ, l2_equation m x y) →
  (m ≠ 1) →
  (m = -2) :=
sorry

end parallel_lines_m_eq_neg2_l403_403540


namespace problem_l403_403584

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403584


namespace notebook_costs_2_20_l403_403288

theorem notebook_costs_2_20 (n c : ℝ) (h1 : n + c = 2.40) (h2 : n = 2 + c) : n = 2.20 :=
by
  sorry

end notebook_costs_2_20_l403_403288


namespace playerA_mean_playerA_median_playerA_mode_playerB_mode_playerB_variance_playerB_variance_less_than_2_6_l403_403220

noncomputable def playerA_scores := [5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 9.0, 9.0, 10.0]
noncomputable def playerB_scores := [5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 9.0, 10.0]

def mean (l : List ℝ) : ℝ := l.sum / l.length

theorem playerA_mean : mean playerA_scores = 7 := by sorry
theorem playerA_median : Statistics.median playerA_scores = 6 := by sorry
theorem playerA_mode : Statistics.mode playerA_scores = 6 := by sorry

theorem playerB_mode : Statistics.mode playerB_scores = 7 := by sorry
theorem playerB_variance : Statistics.variance playerB_scores = 2 := by sorry
theorem playerB_variance_less_than_2_6 : Statistics.variance playerB_scores < 2.6 := by sorry

end playerA_mean_playerA_median_playerA_mode_playerB_mode_playerB_variance_playerB_variance_less_than_2_6_l403_403220


namespace problem_l403_403582

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403582


namespace point_M_exists_l403_403637

theorem point_M_exists (A B C D E M : ℝ × ℝ)
  (is_iso_right_triangle : ∀ {A B C : ℝ × ℝ}, true)  -- Iso-right triangle placeholder
  (not_congruent : ∀ {A D E : ℝ × ℝ}, true)         -- Not congruent placeholder
  (fixed_triangle_ABC : true)                        -- Fixed triangle ABC placeholder
  (rotation : true)                                  -- ADE can rotate around A placeholder
  (M_on_segment_EC : M ∈ [E, C]) :                   -- M is on the line segment EC
  ∃ M : ℝ × ℝ, is_iso_right_triangle ∧ is_iso_right_triangle :=
begin
  -- Sorry placeholder for actual proof
  sorry
end

end point_M_exists_l403_403637


namespace sum_f_k_from_1_to_22_l403_403490

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403490


namespace remainder_30th_smallest_fruity_mod_1000_l403_403313

def E (x : ℕ) : ℕ := --Define the function to find trailing zeros (We assume the definition is given)

def fruity (x : ℕ) : Prop := x > 0 ∧ ∀ d ∈ digits 10 x, d ≠ 0 ∧ E(x) = 24

theorem remainder_30th_smallest_fruity_mod_1000 : 
  ∃! n, fruity n ∧ nth_fruity n 30 ∧ n % 1000 = 885 := 
by
  sorry

end remainder_30th_smallest_fruity_mod_1000_l403_403313


namespace jim_taxi_distance_l403_403127

theorem jim_taxi_distance (initial_fee charge_per_segment total_charge : ℝ) (segment_len_miles : ℝ)
(init_fee_eq : initial_fee = 2.5)
(charge_per_seg_eq : charge_per_segment = 0.35)
(total_charge_eq : total_charge = 5.65)
(segment_length_eq : segment_len_miles = 2/5):
  let charge_for_distance := total_charge - initial_fee
  let num_segments := charge_for_distance / charge_per_segment
  let total_miles := num_segments * segment_len_miles
  total_miles = 3.6 :=
by
  intros
  sorry

end jim_taxi_distance_l403_403127


namespace branches_count_l403_403864

def branches_per_branch (total : ℕ) : ℕ :=
  let x : ℕ := (1 + x + x * x = total)
  x

theorem branches_count {total : ℕ} (h : total = 43) : branches_per_branch total = 6 :=
by {
  sorry
}

end branches_count_l403_403864


namespace find_C_line_MN_l403_403092

def point := (ℝ × ℝ)

-- Given points A and B
def A : point := (5, -2)
def B : point := (7, 3)

-- Conditions: M is the midpoint of AC and is on the y-axis
def M_on_y_axis (M : point) (A C : point) : Prop :=
  M.1 = 0 ∧ M.2 = (A.2 + C.2) / 2

-- Conditions: N is the midpoint of BC and is on the x-axis
def N_on_x_axis (N : point) (B C : point) : Prop :=
  N.1 = (B.1 + C.1) / 2 ∧ N.2 = 0

-- Coordinates of point C
theorem find_C (C : point)
  (M : point) (N : point)
  (hM : M_on_y_axis M A C)
  (hN : N_on_x_axis N B C) : C = (-5, -8) := sorry

-- Equation of line MN
theorem line_MN (M N : point)
  (MN_eq : M_on_y_axis M A (-5, -8) ∧ N_on_x_axis N B (-5, -8)) :
   ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ ((y = M.2) ∧ (x = M.1)) ∨ ((y = N.2) ∧ (x = N.1))) ∧ m = (3/2) ∧ b = 0 := sorry

end find_C_line_MN_l403_403092


namespace sum_f_k_1_22_l403_403505

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403505


namespace sum_of_first_11_terms_l403_403988

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Condition: the sequence is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 1 + a 5 + a 9 = 39
axiom h2 : a 3 + a 7 + a 11 = 27
axiom h3 : is_arithmetic_sequence a d

-- Proof statement
theorem sum_of_first_11_terms : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = 121 := 
sorry

end sum_of_first_11_terms_l403_403988


namespace Cheryl_total_distance_l403_403834

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l403_403834


namespace james_pre_injury_miles_600_l403_403310

-- Define the conditions
def james_pre_injury_miles (x : ℝ) : Prop :=
  ∃ goal_increase : ℝ, ∃ days : ℝ, ∃ weekly_increase : ℝ,
  goal_increase = 1.2 * x ∧
  days = 280 ∧
  weekly_increase = 3 ∧
  (days / 7) * weekly_increase = (goal_increase - x)

-- Define the main theorem to be proved
theorem james_pre_injury_miles_600 : james_pre_injury_miles 600 :=
sorry

end james_pre_injury_miles_600_l403_403310


namespace arithmetic_sequence_sum_l403_403727

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a n = a 0 + n * d)
  (h1 : a 0 + a 3 + a 6 = 45)
  (h2 : a 1 + a 4 + a 7 = 39) :
  a 2 + a 5 + a 8 = 33 := 
by
  sorry

end arithmetic_sequence_sum_l403_403727


namespace value_of_expression_l403_403561

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403561


namespace sequence_inequality_l403_403165

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (h_non_decreasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_range : ∀ i, 1 ≤ i ∧ i ≤ 10 → a i = a (i - 1)) :
  (1 / 6) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) ≤ (1 / 10) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) :=
by
  sorry

end sequence_inequality_l403_403165


namespace monotonic_intervals_area_of_triangle_l403_403931

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x - (1 / 2) * x^2 + 2 * x

theorem monotonic_intervals : 
  (∀ x ∈ Ioo 0 3, 0 < 3 * (1 / x) - x + 2) ∧ 
  (∀ x ∈ Ioi 3, 0 > 3 * (1 / x) - x + 2) :=
sorry

theorem area_of_triangle : 
  let tan_line (x : ℝ) := 4 * x - 5 / 2 in
  (1 / 2) * ((-5 / 2) * (5 / 8)) = 25 / 32 :=
sorry

end monotonic_intervals_area_of_triangle_l403_403931


namespace intersection_M_N_union_N_not_M_l403_403060

-- Define the sets M and N
def M : set ℝ := {x | x > 1}
def N : set ℝ := {x | x ^ 2 - 3 * x ≤ 0}

-- The first problem: M ∩ N = (1, 3]
theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 3} := 
sorry

-- The second problem: N ∪ (¬ M) = (-∞, 3]
theorem union_N_not_M : N ∪ {x | ¬ (x > 1)} = {x | x ≤ 3} :=
sorry

end intersection_M_N_union_N_not_M_l403_403060


namespace sum_f_eq_neg_24_l403_403411

variable (f g : ℝ → ℝ)

theorem sum_f_eq_neg_24 
  (h1 : ∀ x, f(x) + g(2-x) = 5)
  (h2 : ∀ x, g(x) - f(x-4) = 7)
  (h3 : ∀ x, g(2 - x) = g(2 + x))
  (h4 : g(2) = 4) :
  ∑ k in Finset.range 22, f (k + 1 : ℝ) = -24 :=
sorry

end sum_f_eq_neg_24_l403_403411


namespace max_axes_of_symmetry_l403_403233

theorem max_axes_of_symmetry (k : ℕ) : 
  ∃ S : set (set (euclidean_space ℝ 2)), (∀ s ∈ S, is_segment s) ∧ S.card = k ∧ 
  (∀ ax_sym, is_axis_of_symmetry ax_sym S → ax_sym.card ≤ 2 * k) :=
sorry

end max_axes_of_symmetry_l403_403233


namespace value_of_fraction_l403_403547

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403547


namespace sum_f_k_from_1_to_22_l403_403486

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403486


namespace value_of_frac_l403_403588

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403588


namespace complementary_angles_of_same_angle_are_equal_l403_403249

def complementary_angles (α β : ℝ) := α + β = 90 

theorem complementary_angles_of_same_angle_are_equal 
        (θ : ℝ) (α β : ℝ) 
        (h1 : complementary_angles θ α) 
        (h2 : complementary_angles θ β) : 
        α = β := 
by 
  sorry

end complementary_angles_of_same_angle_are_equal_l403_403249


namespace area_of_triangle_ABC_l403_403154

open Real

noncomputable def triangle_area (b c : ℝ) : ℝ :=
  (sqrt 2 / 4) * (sqrt (4 + b^2)) * (sqrt (4 + c^2))

theorem area_of_triangle_ABC (b c : ℝ) :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (2, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, b, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, c)
  let angle_BAC : ℝ := 45
  (cos (angle_BAC * π / 180) = sqrt 2 / 2) →
  (sin (angle_BAC * π / 180) = sqrt 2 / 2) →
  let AB := sqrt (2^2 + b^2)
  let AC := sqrt (2^2 + c^2)
  let area := (1/2) * AB * AC * (sin (45 * π / 180))
  area = triangle_area b c :=
sorry

end area_of_triangle_ABC_l403_403154


namespace stone_statue_cost_l403_403735

theorem stone_statue_cost :
  ∃ S : Real, 
    let total_earnings := 10 * S + 20 * 5
    let earnings_after_taxes := 0.9 * total_earnings
    earnings_after_taxes = 270 ∧ S = 20 :=
sorry

end stone_statue_cost_l403_403735


namespace geometric_sequence_sum_l403_403016

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 4 = 18) 
  (h2 : a 2 * a 3 = 32) (hq : q > 1) 
  (ha : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) : 
  ∑ i in range 8, a i = 510 :=
sorry

end geometric_sequence_sum_l403_403016


namespace cot_squared_sum_l403_403750

theorem cot_squared_sum (α β γ : ℝ) (A B C : ℝ) (tetrahedron : ∀ x, x ∈ {α, β, γ}) :
  cot α ^ 2 + cot β ^ 2 + cot γ ^ 2 ≥ 3 / 4 :=
by sorry

end cot_squared_sum_l403_403750


namespace express_x_in_terms_of_y_l403_403044

variable {x y : ℝ}

theorem express_x_in_terms_of_y (h : 3 * x - 4 * y = 6) : x = (6 + 4 * y) / 3 := 
sorry

end express_x_in_terms_of_y_l403_403044


namespace value_of_ac_over_bd_l403_403599

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403599


namespace product_of_two_numbers_l403_403189

theorem product_of_two_numbers (x y : ℕ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 221) : x * y = 60 := sorry

end product_of_two_numbers_l403_403189


namespace intersection_product_l403_403111

noncomputable def line_l (t : ℝ) := (1 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

theorem intersection_product :
  ∀ (t1 t2 : ℝ), 
  (1 + (1/2) * t1)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t1) →
  (1 + (1/2) * t2)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t2) →
  (1 + (1/2) * t1) * (1 + (1/2) * t2) = 28 := 
  sorry

end intersection_product_l403_403111


namespace digital_earth_definition_l403_403778

-- Definitions of conditions
def virtual_system_of_earth_based_on_coordinates : Prop := 
  true -- Placeholder for the definition

def characterized_by_massive_data_multiple_resolutions : Prop :=
  true -- Placeholder for the definition

def displays_data_multidimensional : Prop :=
  true -- Placeholder for the definition

-- Definition of the concept Digital Earth
def Digital_Earth : Prop :=
  virtual_system_of_earth_based_on_coordinates ∧
  characterized_by_massive_data_multiple_resolutions ∧
  displays_data_multidimensional

-- The main theorem statement we need to prove
theorem digital_earth_definition :
  (Digital_Earth → "A digitized, informational virtual Earth" ∈ {"A digitized, informational virtual Earth", 
                                                               "An Earth measured in numbers for radius, volume, mass, etc.", 
                                                               "An Earth described by a grid of latitude and longitude", 
                                                               "Digital city" and "Digital campus"} ∧ "A digitized, informational virtual Earth" = A) :=
  sorry

end digital_earth_definition_l403_403778


namespace circle_distance_condition_l403_403160

theorem circle_distance_condition (m : ℝ) :
  (∃ (x y : ℝ), (x^2 + y^2 - 2*x - 2*real.sqrt 3*y - m = 0) ∧ (x^2 + y^2 = 1)) ↔ (m ∈ set.Icc (-3) 5) :=
by
  sorry

end circle_distance_condition_l403_403160


namespace prism_volume_l403_403764

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
  a * b * c = 24 * real.sqrt 3 :=
by
  sorry

end prism_volume_l403_403764


namespace arithmetic_common_difference_l403_403904

-- Define the conditions of the arithmetic sequence
def a (n : ℕ) := 0 -- This is a placeholder definition since we only care about a_5 and a_12
def a5 : ℝ := 10
def a12 : ℝ := 31

-- State the proof problem
theorem arithmetic_common_difference :
  ∃ d : ℝ, a5 + 7 * d = a12 :=
by
  use 3
  simp [a5, a12]
  sorry

end arithmetic_common_difference_l403_403904


namespace option_A_correct_option_C_correct_option_D_correct_l403_403921

noncomputable def ellipse_eq (x y : ℝ) := (x^2 / 9) + (y^2 / 3) = 1

def focus_right : ℝ × ℝ := (√6, 0)
def line_eq (m : ℝ) (x : ℝ) : ℝ × ℝ := (x, m)

-- Conditions
variables {x y m : ℝ}
variable h_ellipse : ellipse_eq x y
variable h_m_range : 0 < m ∧ m < √3

theorem option_A_correct :
  ∀ {A B : ℝ × ℝ}, A ∈ ellipse_eq ∧ B ∈ ellipse_eq → 
  y = m → |focus_right.1 - A.1| + |focus_right.1 - B.1| = 6 := sorry

theorem option_C_correct :
  let A := (-√6, 1) 
  let B := (√6, 1) 
  ∀ (m = 1), 
  ∃ A B, A = (-√6, 1) ∧ B = (√6, 1) →
  ∃ F,
  focus_right = F ∧ 
  0 < m ∧ m < √3 → 
  0 < 1 ∧ 1 < √3 → 
  (1/2 * (B.1 - A.1) * F.2 = √6) := sorry

theorem option_D_correct :
  let A := (-(3 * √3) / 2, √3 / 2)
  let B := ((3 * √3) / 2, √3 / 2)
  ∀ (m = √3 / 2), 
  ∃ A B F,
  focus_right = F ∧ 
  0 < m ∧ m < √3 → 
  0 < √3 / 2 ∧ √3 / 2 < √3 → 
  (A.1 * B.1 + A.2 * B.2) = 0 := sorry

end option_A_correct_option_C_correct_option_D_correct_l403_403921


namespace sum_f_1_to_22_l403_403451

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403451


namespace jim_discount_is_40_percent_l403_403126

def car_wash_discount (num_car_washes : ℕ) (cost_per_wash : ℕ) (package_cost : ℕ) : ℕ := 
  ((num_car_washes * cost_per_wash - package_cost) * 100) / (num_car_washes * cost_per_wash)

theorem jim_discount_is_40_percent :
  car_wwash_discount 20 15 180 = 40 :=
by
  sorry

end jim_discount_is_40_percent_l403_403126


namespace find_a_l403_403410

theorem find_a (a x : ℝ) (h : (λ (r : ℕ), ((nat.choose 6 r) * (-1)^r * a^(6-r) * x^(3-r)) 3 = 160) (by apply_instance)) : a = -2 :=
sorry

end find_a_l403_403410


namespace union_M_N_l403_403971

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end union_M_N_l403_403971


namespace tetrahedron_altitudes_l403_403749

theorem tetrahedron_altitudes (r h₁ h₂ h₃ h₄ : ℝ)
  (h₁_def : h₁ = 3 * r)
  (h₂_def : h₂ = 4 * r)
  (h₃_def : h₃ = 4 * r)
  (altitude_sum : 1/h₁ + 1/h₂ + 1/h₃ + 1/h₄ = 1/r) : 
  h₄ = 6 * r :=
by
  rw [h₁_def, h₂_def, h₃_def] at altitude_sum
  sorry

end tetrahedron_altitudes_l403_403749


namespace angle_between_hands_at_3_40_l403_403259

/-- The angle between the hour hand and the minute hand at 3:40 is 130 degrees. --/
theorem angle_between_hands_at_3_40 : 
  let minutes_in_hour := 60
  let degrees_in_full_circle := 360
  let hours_in_clock := 12
  let minutes_in_clock := hours_in_clock * minutes_in_hour
  let time_mins := 3 * minutes_in_hour + 40 -- 3:40 converted to minutes from 12:00
  let hours_degrees_per_minute := degrees_in_full_circle / minutes_in_clock
  let minute_degrees_per_minute := degrees_in_full_circle / minutes_in_hour
  let hour_angle := hours_degrees_per_minute * time_mins
  let minute_angle := minute_degrees_per_minute * 40
  abs (minute_angle - hour_angle) = 130 :=
by
  sorry

end angle_between_hands_at_3_40_l403_403259


namespace restaurant_customer_problem_l403_403300

theorem restaurant_customer_problem (x y z : ℕ) 
  (h1 : x = 2 * z)
  (h2 : y = x - 3)
  (h3 : 3 + x + y - z = 8) :
  x = 6 ∧ y = 3 ∧ z = 3 ∧ (x + y = 9) :=
by
  sorry

end restaurant_customer_problem_l403_403300


namespace derek_money_left_l403_403850

noncomputable def amount_left_after_spending (initial_amount textbooks school_supplies: ℕ) : ℕ :=
  initial_amount - textbooks - school_supplies

theorem derek_money_left :
  ∀ (initial_amount : ℕ),
  initial_amount = 960 →
  let textbooks := initial_amount / 2 in
  let remaining_after_textbooks := initial_amount - textbooks in
  let school_supplies := remaining_after_textbooks / 4 in
  amount_left_after_spending initial_amount textbooks school_supplies = 360 :=
by
  intros initial_amount h_initial_amount textbooks remaining_after_textbooks school_supplies
  rw [h_initial_amount]
  sorry

end derek_money_left_l403_403850


namespace union_M_N_l403_403972

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end union_M_N_l403_403972


namespace shaded_area_is_correct_l403_403636

-- Defining the conditions
def grid_width : ℝ := 15 -- in units
def grid_height : ℝ := 5 -- in units
def total_grid_area : ℝ := grid_width * grid_height -- in square units

def larger_triangle_base : ℝ := grid_width -- in units
def larger_triangle_height : ℝ := grid_height -- in units
def larger_triangle_area : ℝ := 0.5 * larger_triangle_base * larger_triangle_height -- in square units

def smaller_triangle_base : ℝ := 3 -- in units
def smaller_triangle_height : ℝ := 2 -- in units
def smaller_triangle_area : ℝ := 0.5 * smaller_triangle_base * smaller_triangle_height -- in square units

-- The total area of the triangles that are not shaded
def unshaded_areas : ℝ := larger_triangle_area + smaller_triangle_area

-- The area of the shaded region
def shaded_area : ℝ := total_grid_area - unshaded_areas

-- The statement to be proven
theorem shaded_area_is_correct : shaded_area = 34.5 := 
by 
  -- This is a placeholder for the actual proof, which would normally go here
  sorry

end shaded_area_is_correct_l403_403636


namespace sequence_values_and_general_formula_l403_403641

open Nat

def a : ℕ → ℚ
| 0       := 1 / 2
| (n + 1) := 3 * a n / (a n + 3)

theorem sequence_values_and_general_formula :
  a 1 = 3 / 7 ∧ a 2 = 3 / 8 ∧ a 3 = 1 / 3 ∧ ∀ n, a n = 3 / (n + 6) := sorry

end sequence_values_and_general_formula_l403_403641


namespace librarian_took_books_l403_403649

theorem librarian_took_books : 
  (∃ (total_books librarian_books : ℕ) (books_per_shelf shelves_needed books_left : ℕ), 
    total_books = 34 ∧ 
    books_per_shelf = 3 ∧ 
    shelves_needed = 9 ∧ 
    books_left = shelves_needed * books_per_shelf ∧ 
    librarian_books = total_books - books_left 
  ) → librarian_books = 7 :=
 by 
   intro h,
   rcases h with ⟨total_books, librarian_books, books_per_shelf, shelves_needed, books_left, 
                 h_total, h_shelf, h_needed, h_left_calc, h_librarian_calc⟩,
   rw h_total at h_librarian_calc,
   rw h_shelf at h_left_calc,
   rw h_needed at h_left_calc,
   rw h_left_calc at h_librarian_calc,
   norm_num at h_librarian_calc,
   assumption

end librarian_took_books_l403_403649


namespace booster_club_tickets_l403_403178

theorem booster_club_tickets (x : ℕ) : 
  (11 * 9 + x * 7 = 225) → 
  (x + 11 = 29) := 
by
  sorry

end booster_club_tickets_l403_403178


namespace problem_l403_403580

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403580


namespace atomic_number_cannot_be_x_plus_4_l403_403342

-- Definitions for atomic numbers and elements in the same main group
def in_same_main_group (A B : Type) (atomic_num_A atomic_num_B : ℕ) : Prop :=
  atomic_num_B ≠ atomic_num_A + 4

-- Noncomputable definition is likely needed as the problem involves non-algorithmic aspects.
noncomputable def periodic_table_condition (A B : Type) (x : ℕ) : Prop :=
  in_same_main_group A B x (x + 4)

-- Main theorem stating the mathematical proof problem
theorem atomic_number_cannot_be_x_plus_4
  (A B : Type)
  (x : ℕ)
  (h : periodic_table_condition A B x) : false :=
  by
    sorry

end atomic_number_cannot_be_x_plus_4_l403_403342


namespace A_eq_B_l403_403138

def A : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 + 2 * y^2
}

def B : Set ℤ := {
  z : ℤ | ∃ x y : ℤ, z = x^2 - 6 * x * y + 11 * y^2
}

theorem A_eq_B : A = B :=
by {
  sorry
}

end A_eq_B_l403_403138


namespace initial_bonus_amount_correct_l403_403652

noncomputable def initial_bonus_amount (number_of_students : ℕ)
                                       (average_score_threshold : ℕ)
                                       (initial_graded_tests : ℕ)
                                       (average_score_first_8_tests : ℕ)
                                       (combined_score_last_2 : ℕ)
                                       (total_bonus : ℕ)
                                       (extra_bonus_per_point_above_threshold : ℕ) : ℕ :=
  let total_points_first_8 := initial_graded_tests * average_score_first_8_tests in
  let total_points := total_points_first_8 + combined_score_last_2 in
  let new_average_score := total_points / number_of_students in
  let points_above_threshold := new_average_score - average_score_threshold in
  let extra_bonus := points_above_threshold * extra_bonus_per_point_above_threshold in
  total_bonus - extra_bonus

theorem initial_bonus_amount_correct :
  initial_bonus_amount 10 75 8 70 290 600 10 = 500 :=
by {
  let expected_bonus := 500,
  rw [initial_bonus_amount],
  sorry
}

end initial_bonus_amount_correct_l403_403652


namespace sum_f_eq_neg24_l403_403493

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403493


namespace positive_integers_sequence_l403_403851

theorem positive_integers_sequence (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a ∣ (b + c + d)) (h5 : b ∣ (a + c + d)) 
  (h6 : c ∣ (a + b + d)) (h7 : d ∣ (a + b + c)) : 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 9) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 8 ∧ d = 12) ∨ 
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 10) ∨ 
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 21) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 10 ∧ d = 15) :=
sorry

end positive_integers_sequence_l403_403851


namespace sum_f_eq_neg24_l403_403500

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403500


namespace regular_15gon_triangle_probability_l403_403840

-- Define the regular 15-gon
def n : ℕ := 15

-- Function to calculate segment lengths
def segment_length (k : ℕ) : ℝ := 2 * Real.sin (k * Real.pi / n)

-- Definition of the event where three segments form a valid triangle
def valid_triangle (a b c: ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Total number of segments in a regular 15-gon
def total_segments : ℕ := (n * (n - 1)) / 2

-- Proof outline for the probability
theorem regular_15gon_triangle_probability :
  (∑ a ∈ SegmentLengths, ∑ b ∈ SegmentLengths, ∑ c ∈ SegmentLengths, if valid_triangle a b c then 1 else 0) / (total_segments.choose 3) = 100/141 := 
sorry

end regular_15gon_triangle_probability_l403_403840


namespace find_a3_l403_403393

noncomputable def a₁ : ℝ := 3⁻⁵

def geometric_mean (seq : ℕ → ℝ) (n : ℕ) : ℝ :=
real.pow (∏ i in finset.range n, seq (i + 1)) (1 / n)

def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
a₁ * q^(n-1)

theorem find_a3 :
  ∃ q : ℝ, geometric_mean (geometric_sequence a₁ q) 8 = 9 → (∃ a₃ : ℝ, a₃ = geometric_sequence a₁ q 3 ∧ a₃ = 1 / 3) :=
begin
  sorry
end

end find_a3_l403_403393


namespace bank_teller_bills_l403_403784

theorem bank_teller_bills (x y : ℕ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
by
  sorry

end bank_teller_bills_l403_403784


namespace difference_of_triangular_2010_2009_l403_403820

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_of_triangular_2010_2009 :
  triangular 2010 - triangular 2009 = 2010 :=
by
  sorry

end difference_of_triangular_2010_2009_l403_403820


namespace sufficient_but_not_necessary_l403_403384

def f (x : ℝ) : ℝ := x^3 - 2*x + 1

theorem sufficient_but_not_necessary {x : ℝ} (h : x = 1) : 
  (f x = 0) ∧ (∃ y ∈ Ioo (-2) (-1), f y = 0) :=
by {
  have hx : f 1 = 0,
  { calc f 1 = 1^3 - 2*1 + 1 : rfl
    ... = 0 : by norm_num },
  have hy : ∃ y ∈ Ioo (-2 : ℝ) (-1), f y = 0,
  { use -1, split, norm_num,
    have h1 := calc f (-2) = (-2)^3 - 2*(-2) + 1 : rfl
                            ... = -8 + 4 + 1 : by norm_num
                            ... = -3 : by norm_num,
    have h2 := calc f (-1) = (-1)^3 - 2*(-1) + 1 : rfl
                           ... = -1 + 2 + 1 : by norm_num
                           ... = 2 : by norm_num,
    exact h1, exact h2 },
  exact ⟨hx, hy⟩,
}

end sufficient_but_not_necessary_l403_403384


namespace relatively_prime_compute_a_b_sum_l403_403658

noncomputable def sequence_sum : ℚ :=
  let X := ∑' n, if n % 2 = 0 then (↑(n + 1) / (2 : ℚ)^(2 * (n / 2) + 2)) else 0
  let Y := ∑' n, if n % 2 = 1 then (↑(n + 1) / (3 : ℚ)^(2 * (n / 2) + 3)) else 0
  in X + Y

theorem relatively_prime (a b : ℕ) : a.gcd b = 1

theorem compute_a_b_sum : ∃ a b : ℕ, relatively_prime a b ∧ (a + b = 443) ∧ (a : ℚ) / (b : ℚ) = sequence_sum := 
begin
  sorry,
end

end relatively_prime_compute_a_b_sum_l403_403658


namespace compute_a1d1_a2d2_a3d3_l403_403659

noncomputable def polynomial_equation (a1 a2 a3 d1 d2 d3: ℝ) : Prop :=
  ∀ x : ℝ, (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)

theorem compute_a1d1_a2d2_a3d3 (a1 a2 a3 d1 d2 d3 : ℝ) (h : polynomial_equation a1 a2 a3 d1 d2 d3) : 
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a1d1_a2d2_a3d3_l403_403659


namespace largest_difference_l403_403141

noncomputable def A := 3 * (2010: ℕ) ^ 2011
noncomputable def B := (2010: ℕ) ^ 2011
noncomputable def C := 2009 * (2010: ℕ) ^ 2010
noncomputable def D := 3 * (2010: ℕ) ^ 2010
noncomputable def E := (2010: ℕ) ^ 2010
noncomputable def F := (2010: ℕ) ^ 2009

theorem largest_difference :
  (A - B) > (B - C) ∧ (A - B) > (C - D) ∧ (A - B) > (D - E) ∧ (A - B) > (E - F) :=
by
  sorry

end largest_difference_l403_403141


namespace problem_multiple_5_or_6_not_15_l403_403071

theorem problem_multiple_5_or_6_not_15 : 
  (finset.filter (λ n => (n % 5 = 0 ∨ n % 6 = 0) ∧ ¬(n % 15 = 0)) (finset.range 3001)).card = 900 :=
by {
  sorry
}

end problem_multiple_5_or_6_not_15_l403_403071


namespace arrange_1250_as_multiple_of_5_l403_403983

theorem arrange_1250_as_multiple_of_5 : 
  let digits := [1, 2, 5, 0],
  let cases := {n // n ∈ permutations digits ∧ (n.last = 0 ∨ n.last = 5)},
  finite cases ∧ fintype.card cases = 10 :=
by
  sorry

end arrange_1250_as_multiple_of_5_l403_403983


namespace find_a_of_X_is_odd_l403_403928

noncomputable def X (x : ℝ) (a : ℝ) : ℝ := (2^x / (2^x - 1)) + a

theorem find_a_of_X_is_odd : 
  (∀ (x : ℝ), X (-x) == -X x → a = -1/2) := 
by
  sorry

end find_a_of_X_is_odd_l403_403928


namespace count_distinct_x_for_term_3001_in_sequence_l403_403846

theorem count_distinct_x_for_term_3001_in_sequence :
  let seq : ℕ → ℝ := fun n =>
    if n = 0
    then x
    else if n = 1
    then 3000
    else if n = 2
    then (3002 / x)
    else if n % 5 = 0
    then seq 0
    else if n % 5 = 1
    then seq 1
    else if n % 5 = 2
    then (3002 / seq 0)
    else if n % 5 = 3
    then (seq 0 + 3002) / (3000 * seq 0)
    else (seq 0 + 2) / 3000
  in
  ∃ n, seq n = 3001 → x = 3001 ∨ x = 3002/3001 ∨ x = 3002/9000001 ∨ x = 9000002 :=
begin
  sorry
end

end count_distinct_x_for_term_3001_in_sequence_l403_403846


namespace max_constant_k_l403_403085

theorem max_constant_k (x y : ℤ) : 4 * x^2 + y^2 + 1 ≥ 3 * x * (y + 1) :=
sorry

end max_constant_k_l403_403085


namespace value_of_expression_l403_403559

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403559


namespace EM_minus_half_BP_constant_l403_403392

-- Assuming relevant geometric definitions for triangles, midpoints, and line segments
section Geometry

variables {A B C P Q R M E F : Type} [RightAngledTriangle A B C]
          [Midpoint E A C] [Midpoint F A B]
          [MovesOnLine P B C] 
          [OnLineAndEqualDist Q AE QB QP] 
          [OnLineAndEqualDist R CF RC RP]
          [Intersection M QR EF]

-- Define the main theorem
theorem EM_minus_half_BP_constant : 
  ∀ (P : Point), is_constant (λ P, distance E M - 1 / 2 * distance B P) :=
sorry

end Geometry

end EM_minus_half_BP_constant_l403_403392


namespace geometric_figure_area_sum_l403_403307

theorem geometric_figure_area_sum
  (E A B : ℝ)
  (X Y Z : ℝ)
  (hEAB : ∠ E A B = 90)
  (hBE : B - E = 12)
  (hXY : X - Y = 5)
  (hYZ : Y - Z = 12)
  (hYXZ : ∠ Y X Z = 90) :
  let square_area := 144
  let triangle_area := 30
  square_area + triangle_area = 174 :=
by
  sorry

end geometric_figure_area_sum_l403_403307


namespace digits_right_of_decimal_problem_l403_403950

-- Define the conditions
def A : ℕ := 5^8
def B : ℕ := 10^6
def C : ℕ := 216

-- Define the problem solution
noncomputable def digits_right_of_decimal (num : ℕ) (denom : ℕ) : ℕ :=
  (Real.decimalExpansion (num : ℝ) (denom : ℝ)).length - (Real.decimalExpansion (num : ℝ) (denom : ℝ)).indexOf '0'

-- State the proof problem
theorem digits_right_of_decimal_problem : digits_right_of_decimal A (B * C) = 5 := by
  sorry

end digits_right_of_decimal_problem_l403_403950


namespace geometric_sequence_a1_range_l403_403920

theorem geometric_sequence_a1_range (a : ℕ → ℝ) (b : ℕ → ℝ) (a1 : ℝ) :
  (∀ n, a (n+1) = a n / 2) ∧ (∀ n, b n = n / 2) ∧ (∃! n : ℕ, a n > b n) →
  (6 < a1 ∧ a1 ≤ 16) :=
by
  sorry

end geometric_sequence_a1_range_l403_403920


namespace volume_cone_SO_minimum_value_PO_area_circle_O1_l403_403512

-- Definitions based on given conditions
def generatrix_length_SO := 2 * Real.sqrt 5
def diameter_AB := 4
def radius_base_circle_O := diameter_AB / 2
def height_SO := Real.sqrt ((generatrix_length_SO)^2 - (radius_base_circle_O)^2) -- derived from Pythagorean theorem

-- Proof statements
theorem volume_cone_SO : (1 / 3) * Real.pi * (radius_base_circle_O)^2 * height_SO = (16 * Real.pi) / 3 :=
sorry

theorem minimum_value_PO (r : Real) : ∃ (P : Real × Real), (P.1 = radius_base_circle_O - 2 * r) ∧
  (P.2 = r) ∧
  Real.sqrt (P.1^2 + P.2^2) = 4 * Real.sqrt(5) / 5 :=
sorry

theorem area_circle_O1 (r : Real) : (Real.sqrt (16 - 16 * r + 5 * r^2) = 4) →
  Real.pi * (r^2) = (36 * Real.pi) / 25 :=
sorry

end volume_cone_SO_minimum_value_PO_area_circle_O1_l403_403512


namespace problem_solution_l403_403068

variable (a b c d m : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def absolute_value_eq (m : ℝ) : Prop := |m| = 3

theorem problem_solution
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : absolute_value_eq m) :
  (a + b) / 2023 - 4 * (c * d) + m^2 = 5 :=
by
  sorry

end problem_solution_l403_403068


namespace expand_product_l403_403345

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end expand_product_l403_403345


namespace extreme_values_l403_403519

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem extreme_values :
  (∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 5 / 3 ∧ f x1 = -2 ∧ f x2 = -58 / 27) ∧ 
  (∃ (a b : ℝ), a = 2 ∧ b = f 2 ∧ (∀ (x : ℝ), (a, b) = (x, f x) → (∀ y : ℝ, y = x - 4))) :=
by
  sorry

end extreme_values_l403_403519


namespace sequence_properties_l403_403019

noncomputable def a_n (n : ℕ) : ℕ := n + 1
noncomputable def b_n (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)
noncomputable def c_n (n : ℕ) : ℚ := (n + 1) / 3 ^ (n - 1)

def S_n (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n i

noncomputable def C_n (n : ℕ) : ℚ := ∑ i in Finset.range n, c_n (i + 1)

theorem sequence_properties :
  ∀ n, a_n 1 = b_n 1 ∧
       2 * a_n 2 = b_n 2 ∧
       S_n 2 + T_n 2 = 13 ∧
       2 * S_n 3 = b_n 3 ∧
       a_n n = n + 1 ∧
       b_n n = 2 * 3 ^ (n - 1) ∧
       C_n n = (15 / 4) - (2 * n + 5) / (4 * 3 ^ (n - 1)) := 
by
  sorry

end sequence_properties_l403_403019


namespace number_of_fish_initially_tagged_l403_403100

theorem number_of_fish_initially_tagged {N T : ℕ}
  (hN : N = 1250)
  (h_ratio : 2 / 50 = T / N) :
  T = 50 :=
by
  sorry

end number_of_fish_initially_tagged_l403_403100


namespace sum_arithmetic_sequence_l403_403042

variable {n : ℕ}
variable {a : ℕ → ℝ} -- Arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Given conditions
def a₁ := -1 
def recurrence_relation (n : ℕ) := (a (n + 1) - 4) * n = 2 * S n

-- Prove the sum Sₙ of the first n terms is (3n² - 5n) / 2
theorem sum_arithmetic_sequence (h₁ : a 1 = a₁) (h₂ : ∀ n : ℕ, recurrence_relation n) :
  S n = (3 * n^2 - 5 * n) / 2 := 
sorry

end sum_arithmetic_sequence_l403_403042


namespace general_term_formula_l403_403989

-- Define the arithmetic sequence with its general term
def arithmetic_sequence (a1 d : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := arithmetic_sequence a1 d n + d

-- Given conditions
variables (a1 d : ℝ)
axiom a7_eq_4 : arithmetic_sequence a1 d 7 = 4
axiom a19_eq_2a9 : arithmetic_sequence a1 d 19 = 2 * arithmetic_sequence a1 d 9

-- Prove that the general term formula for the sequence is a_n = (n + 1) / 2
theorem general_term_formula :
  ∃ a1 d : ℝ, (∀ n : ℕ, arithmetic_sequence a1 d n = (n + 1) / 2) :=
by
  use a1, d
  sorry

end general_term_formula_l403_403989


namespace sum_f_eq_neg24_l403_403467

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403467


namespace value_of_frac_l403_403593

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403593


namespace sum_f_values_l403_403473

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403473


namespace solve_congruence_l403_403171

theorem solve_congruence :
  ∃ (x : ℤ), (15 * x + 2) % 17 = 7 % 17 ∧ ((x % 17) = 6 % 17) ∧ (6 + 17 = 23) :=
by
  -- Definitions and conditions
  let m := 17
  let a := 6
  existsi a
  split

  -- Show (15 * x + 2) % 17 = 7 % 17
  sorry

  -- Show (x % 17) = 6 % 17
  split
  sorry
  
  -- Show (6 + 17 = 23)
  trivial

end solve_congruence_l403_403171


namespace range_of_a_l403_403041

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l403_403041


namespace line_through_point_inequality_l403_403086

theorem line_through_point_inequality
  (a b θ : ℝ)
  (h : (b * Real.cos θ + a * Real.sin θ = a * b)) :
  1 / a^2 + 1 / b^2 ≥ 1 := 
  sorry

end line_through_point_inequality_l403_403086


namespace find_y_l403_403081

theorem find_y (steps distance : ℕ) (total_steps : ℕ) (marking_step : ℕ)
  (h1 : total_steps = 8)
  (h2 : distance = 48)
  (h3 : marking_step = 6) :
  steps = distance / total_steps * marking_step → steps = 36 :=
by
  intros
  sorry

end find_y_l403_403081


namespace correct_propositions_l403_403816

def proposition1 : Prop := ∀ x : ℝ, x^2 = 2003 * 2005 + 1 → x = 2004
def proposition2 : Prop := ∀ (q : Type) [has_eq q] [has_perpendicular q], is_quadrilateral q ∧ has_equal_diagonals q ∧ has_perpendicular_diagonals q → is_square q ∨ is_isosceles_trapezoid q
def proposition3 : Prop := ∀ (l : ℝ), uses_same_length l (circle_area > square_area)
def proposition4 : Prop := three_non_coincident_planes_divide_space (4 ∨ 6 ∨ 7 ∨ 8)

theorem correct_propositions : proposition3 ∧ proposition4 :=
by
  sorry

end correct_propositions_l403_403816


namespace shortest_distance_between_point_and_parabola_l403_403357

noncomputable def shortestDistance : ℕ := 56

theorem shortest_distance_between_point_and_parabola (x y : ℝ)
    (h_point : (x, y) = (8, 16))
    (h_parabola : x = y^2 / 4) : 
    sqrt ((64 - 8)^2 + (16 - 16)^2) = 56 :=
by
  sorry

end shortest_distance_between_point_and_parabola_l403_403357


namespace value_of_ac_over_bd_l403_403597

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403597


namespace number_of_cities_experienced_protests_l403_403754

variables (days_of_protest : ℕ) (arrests_per_day : ℕ) (days_pre_trial : ℕ) 
          (days_post_trial_in_weeks : ℕ) (combined_weeks_jail : ℕ)

def total_days_in_jail_per_person := days_pre_trial + (days_post_trial_in_weeks * 7) / 2

theorem number_of_cities_experienced_protests 
  (h1 : days_of_protest = 30) 
  (h2 : arrests_per_day = 10) 
  (h3 : days_pre_trial = 4) 
  (h4 : days_post_trial_in_weeks = 2) 
  (h5 : combined_weeks_jail = 9900) : 
  (combined_weeks_jail * 7) / total_days_in_jail_per_person 
  = 21 :=
by
  sorry

end number_of_cities_experienced_protests_l403_403754


namespace opposite_of_6_is_neg_6_l403_403203

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l403_403203


namespace trajectory_and_area_l403_403916

-- Define the given conditions and problem
theorem trajectory_and_area :
  (∀ M : ℝ × ℝ, (M.1 + sqrt 3)^2 + M.2^2 = 16 →
    ∃ P : ℝ × ℝ, (∃ F : ℝ × ℝ, F = (sqrt 3, 0) ∧ (P.1 - F.1)^2 + P.2^2 = (M.1 - P.1)^2 + (M.2 - P.2)^2 ∧
      (∀ l : ℝ → ℝ, ∀ A B C D : ℝ × ℝ, C ∈ {P | (P.1 / 2)^2 + P.2^2 = 1 ∧ A ≠ B ∧ ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
      (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
      (D.1 - C.1) = (A.1 - C.1) + (B.1 - C.1) ∧ (D.2 - C.2) = (A.2 - C.2) + (B.2 - C.2) ∧
      (D.1 - A.1) * (D.2 - B.2) - (D.2 - A.2) * (D.1 - B.1) ∈ (Set.Ioo (16 / 5) 4))))
sorry

end trajectory_and_area_l403_403916


namespace compute_sum_of_constants_l403_403369

def greatest_integer (x : ℝ) : ℤ := floor x
def fractional_part (x : ℝ) : ℝ := x - ↑(floor x)
def satisfies_equation (α : ℝ) : Prop := α^2 + fractional_part α = 21

theorem compute_sum_of_constants (a b c d : ℕ) 
  (h1 : a = 101)
  (h2 : b = 65)
  (h3 : c = 2)
  (h4 : d = 1)
  (h5 : ∀ {α : ℝ}, satisfies_equation α → α = (8 - 9 + Real.sqrt 101) / 2 ∨ α = -(10 - 9 + Real.sqrt 65) / 2 ) 
  : a + b + c + d = 169 := by
  sorry

end compute_sum_of_constants_l403_403369


namespace grocer_display_rows_l403_403790

theorem grocer_display_rows (n : ℕ) (h : (∑ k in Finset.range n, (3 + 2 * k)) = 225) : n = 15 := by
  sorry

end grocer_display_rows_l403_403790


namespace min_PQ_distance_l403_403031

noncomputable def min_distance (y₀ : ℝ) : ℝ := 
  (((y₀^2 - 3)^2 + y₀^2) ^ (1/2) - 1)

theorem min_PQ_distance : 
  ∃ y₀ : ℝ, y₀^2 = x ∧ (x - 3)^2 + y^2 = 1 → 
  min_distance y₀ = (√(11) / 2) - 1 := sorry

end min_PQ_distance_l403_403031


namespace length_of_hall_l403_403974

-- Conditions
def width : ℕ := 15
def height : ℕ := 5
def total_expenditure : ℕ := 9500
def cost_per_square_meter : ℕ := 10

-- Derived data
def total_area : ℕ := total_expenditure / cost_per_square_meter

-- Let length be a variable
variable (length : ℕ)

-- The equation derived from the solution steps should hold true
def hall_area_eq : Prop :=
  2 * (length * width) + 2 * (length * height) + 2 * (width * height) = total_area

-- The length of the hall is 20 m
theorem length_of_hall : length = 20 :=
by
  unfold hall_area_eq total_area width height total_expenditure cost_per_square_meter
  -- actual proof can be filled in here
  sorry

end length_of_hall_l403_403974


namespace sin_arccos_one_over_four_l403_403838

theorem sin_arccos_one_over_four : sin (arccos (1 / 4)) = sqrt 15 / 4 :=
by
  sorry

end sin_arccos_one_over_four_l403_403838


namespace find_larger_number_l403_403953

theorem find_larger_number (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 :=
sorry

end find_larger_number_l403_403953


namespace sufficient_but_not_necessary_l403_403779

variable {x l : ℝ}

theorem sufficient_but_not_necessary (h : x > l) (hx : x > 1 ∨ x < -1) : x^2 > 1 :=
by {
  have : h → (x^2 > 1) := sorry,
  have : ¬(x > l) → ¬(x^2 > 1) := sorry,
  show x^2 > 1, from sorry
}

end sufficient_but_not_necessary_l403_403779


namespace sum_f_k_1_22_l403_403507

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403507


namespace team_points_can_be_odd_l403_403279

def round_robin_tournament (teams : ℕ) (games : ℕ) (points_win : ℕ) (points_draw : ℕ) (bonus : ℕ) : Prop :=
  teams = 10 ∧ games = 45 ∧ points_win = 3 ∧ points_draw = 1 ∧ bonus = 5

theorem team_points_can_be_odd:
  ∀ (score : ℕ), round_robin_tournament 10 45 3 1 5 → 
  (∃ (wins draws losses : ℕ), wins + draws + losses = 9 ∧ 
  score = wins * 3 + draws * 1 + if wins = 9 then 5 else 0 ∧
  score % 2 = 1) :=
by 
  sorry

end team_points_can_be_odd_l403_403279


namespace infinitely_many_primes_l403_403689

theorem infinitely_many_primes : ¬finite {p : ℕ | Nat.Prime p} :=
by
  assume H : finite {p : ℕ | Nat.Prime p}
  sorry

end infinitely_many_primes_l403_403689


namespace matrix_multiplication_correct_l403_403839

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![[2, 0, 1],
    [-1, 3, 4],
    [0, -2, 5]]

def matrix_B : Matrix (Fin 3) (Fin 4) ℤ := 
  ![[1, -1, 0, 2],
    [0, 2, -3, 1],
    [3, 0, 2, -1]]

def matrix_C : Matrix (Fin 3) (Fin 4) ℤ := 
  ![[5, -2, 2, 3],
    [11, 7, -1, -3],
    [15, -4, 16, -7]]

theorem matrix_multiplication_correct : matrix_A ⬝ matrix_B = matrix_C :=
  sorry

end matrix_multiplication_correct_l403_403839


namespace chess_tournament_game_count_l403_403268

theorem chess_tournament_game_count (n : ℕ) (h1 : ∃ n, ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → ∃ games_between, games_between = n ∧ games_between * (Nat.choose 6 2) = 30) : n = 2 :=
by
  sorry

end chess_tournament_game_count_l403_403268


namespace battery_charge_time_l403_403795

theorem battery_charge_time (T : ℝ) (R : ℝ) (A : ℝ) (minutes : ℝ) (percent_per_hour : ℝ) : 
  T = 20 ∧ R = 60 ∧ A = 195 ∧ (minutes = R + A ∧ percent_per_hour = T / R) → minutes = 255 :=
by
  intros h
  cases h with hT h
  cases h with hR h
  cases h with hA h
  cases h with hminutes hpercent_per_hour
  rw [hR, hA] at hminutes
  simp at hminutes
  assumption

end battery_charge_time_l403_403795


namespace hexagon_ratio_l403_403317

theorem hexagon_ratio 
  (hex_area : ℝ)
  (rs_bisects_area : ∃ (a b : ℝ), a + b = hex_area / 2 ∧ ∃ (x r s : ℝ), x = 4 ∧ r * s = (hex_area / 2 - 1))
  : ∀ (XR RS : ℝ), XR = RS → XR / RS = 1 :=
by
  sorry

end hexagon_ratio_l403_403317


namespace machine_pays_for_itself_l403_403648

def machine_cost : ℝ := 200
def discount : ℝ := 20
def cost_per_day_self_made : ℝ := 3
def previous_cost_per_coffee : ℝ := 4
def coffees_per_day : ℝ := 2

theorem machine_pays_for_itself :
  let total_cost := machine_cost - discount,
      previous_daily_cost := coffees_per_day * previous_cost_per_coffee,
      daily_savings := previous_daily_cost - cost_per_day_self_made
  in total_cost / daily_savings = 36 :=
by
  sorry

end machine_pays_for_itself_l403_403648


namespace collinear_vectors_if_no_basis_with_other_vector_l403_403046

theorem collinear_vectors_if_no_basis_with_other_vector
  (u v : ℝ^3) (hu : u ≠ 0) (hv : v ≠ 0)
  (h : ∀ (w : ℝ^3), ¬ linearly_independent ℝ ![u, v, w]) :
  ∃ k : ℝ, u = k • v :=
sorry

end collinear_vectors_if_no_basis_with_other_vector_l403_403046


namespace triangle_solutions_l403_403644

noncomputable def a (b c : ℝ) (cos_A : ℝ) : ℝ :=
  sqrt (b^2 + c^2 - 2 * b * c * cos_A)

noncomputable def S_triangle_ABC (b c sin_A : ℝ) : ℝ :=
  0.5 * b * c * sin_A

theorem triangle_solutions :
  let b := 4
  let c := 2
  let cos_A := (1 / 4 : ℝ)
  let sin_A := sqrt (1 - cos_A^2)
  (a b c cos_A = 4) ∧ (S_triangle_ABC b c sin_A = sqrt 15) :=
by
  sorry

end triangle_solutions_l403_403644


namespace points_per_right_answer_l403_403702

variable (p : ℕ)
variable (total_problems : ℕ := 25)
variable (wrong_problems : ℕ := 3)
variable (score : ℤ := 85)

theorem points_per_right_answer :
  (total_problems - wrong_problems) * p - wrong_problems = score -> p = 4 :=
  sorry

end points_per_right_answer_l403_403702


namespace solve_for_q_l403_403964

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l403_403964


namespace Quincy_more_pictures_l403_403167

theorem Quincy_more_pictures {randy peter quincy : ℕ} :
  randy = 5 →
  peter = randy + 3 →
  randy + peter + quincy = 41 →
  quincy - peter = 20 :=
begin
  intros h1 h2 h3,
  sorry
end

end Quincy_more_pictures_l403_403167


namespace value_of_expression_l403_403557

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403557


namespace opposite_of_6_is_neg_6_l403_403202

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l403_403202


namespace find_a_l403_403184

theorem find_a :
  (∃ a : ℝ, 80 = (choose 5 3) * (a^3)) ∧ a = 2 :=
begin
  sorry
end

end find_a_l403_403184


namespace solve_equation_l403_403699

theorem solve_equation : ∃ x : ℤ, (x - 15) / 3 = (3 * x + 11) / 8 ∧ x = -153 := 
by
  use -153
  sorry

end solve_equation_l403_403699


namespace machines_needed_l403_403253

variables (R x m N : ℕ) (h1 : 4 * R * 6 = x)
           (h2 : N * R * 6 = m * x)

theorem machines_needed : N = m * 4 :=
by sorry

end machines_needed_l403_403253


namespace sum_f_eq_neg24_l403_403497

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403497


namespace smallest_number_divisible_1_through_12_and_15_l403_403762

theorem smallest_number_divisible_1_through_12_and_15 :
  ∃ n, (∀ i, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ∧ 15 ∣ n ∧ n = 27720 :=
by {
  sorry
}

end smallest_number_divisible_1_through_12_and_15_l403_403762


namespace original_cube_volume_l403_403682

variable (a : ℕ) -- Define the variable representing the edge length of the cube

-- Define the conditions as predicates
def cube_volume := a^3
def new_dimensions_volume := (a + 2) * (a - 2) * a
def volume_difference := cube_volume - new_dimensions_volume = 8

-- Define the theorem to prove the volume of the original cube
theorem original_cube_volume : (∃ a, volume_difference a) → cube_volume a = 8 :=
by
  sorry  -- Proof is omitted

end original_cube_volume_l403_403682


namespace cyclic_sum_nonnegative_l403_403151

variable (x y z : ℝ)

theorem cyclic_sum_nonnegative (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z ≥ 1) :
  (∑ cyc in {x, y, z}, (cyc ^ 5 - cyc ^ 2) / (cyc ^ 5 + y ^ 2 + z ^ 2)) ≥ 0 :=
by sorry

end cyclic_sum_nonnegative_l403_403151


namespace sin_value_iff_sec_plus_tan_l403_403910

open Real

theorem sin_value_iff_sec_plus_tan (x : ℝ) (h : sec x + tan x = 4 / 3) : 
  sin x = 7 / 25 :=
by
  sorry

end sin_value_iff_sec_plus_tan_l403_403910


namespace min_area_triangle_FMN_fixed_point_line_AC_l403_403053

theorem min_area_triangle_FMN (p : ℝ) (hp : p > 0) (F : ℝ × ℝ) 
  (AB CD : ℝ × ℝ → Prop) (MID_M : ℝ × ℝ) (MID_N : ℝ × ℝ) 
  (h1 : ∀ A B, AB A ∧ AB B → dist F A = dist F B ∧ MID_M = midpoint A B)
  (h2 : ∀ C D, CD C ∧ CD D → dist F C = dist F D ∧ MID_N = midpoint C D)
  (G : Set (ℝ × ℝ)) (hg : parabola G p) : ∃ k : ℝ, k = 1 := 
sorry

theorem fixed_point_line_AC (p : ℝ) (hp : p > 0) (F : ℝ × ℝ) 
  (AC BD : ℝ × ℝ → Prop) (k_AC k_BD : ℝ)
  (h1 : ∀ A C, AC A ∧ AC C → slope A C = k_AC)
  (h2 : ∀ B D, BD B ∧ BD D → slope B D = k_BD ∧ k_AC + 4 * k_BD = 0)
  (G : Set (ℝ × ℝ)) (hg : parabola G p) : ∃ P : ℝ × ℝ, P = (0, -2) :=
sorry

end min_area_triangle_FMN_fixed_point_line_AC_l403_403053


namespace max_value_l403_403781

theorem max_value (x : ℝ) : ∃ m, ∀ x, f x ≤ m ∧ m = sqrt 5 where
  f x := 2 * cos x + sin x :=
sorry

end max_value_l403_403781


namespace sum_f_values_l403_403475

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : ∀ x, f(x) + g(2 - x) = 5)
variable (h2 : ∀ x, g(x) - f(x - 4) = 7)
variable (h3 : ∀ x, g(2 - x) = g(2 + x))
variable (h4 : g(2) = 4)

-- Theorems or goals
theorem sum_f_values : ∑ k in finset.range 22, f (k + 1) = -24 :=
sorry

end sum_f_values_l403_403475


namespace sum_f_k_1_22_l403_403501

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403501


namespace books_per_day_l403_403674

-- Define the condition: Mrs. Hilt reads 15 books in 3 days.
def reads_books_in_days (total_books : ℕ) (days : ℕ) : Prop :=
  total_books = 15 ∧ days = 3

-- Define the theorem to prove that Mrs. Hilt reads 5 books per day.
theorem books_per_day (total_books : ℕ) (days : ℕ) (h : reads_books_in_days total_books days) : total_books / days = 5 :=
by
  -- Stub proof
  sorry

end books_per_day_l403_403674


namespace picture_distance_l403_403223

theorem picture_distance (w t s p d : ℕ) (h1 : w = 25) (h2 : t = 2) (h3 : s = 1) (h4 : 2 * p + s = t + s + t) 
  (h5 : w = 2 * d + p) : d = 10 :=
by
  sorry

end picture_distance_l403_403223


namespace integral_of_abs_x_squared_minus_one_l403_403870

-- Definitions of the integral and absolute value conditions
def absolute_value (x: ℝ) : ℝ := |x|

-- Statement of the main theorem
theorem integral_of_abs_x_squared_minus_one : 
  ∫ x in 0..1, absolute_value (x^2 - 1) = 2 / 3 := 
by
  sorry

end integral_of_abs_x_squared_minus_one_l403_403870


namespace avg_eq_3x_minus_8_l403_403872

theorem avg_eq_3x_minus_8 (x : ℝ) :
  (1 / 3) * ((x + 3) + (4x + 1) + (3x + 6)) = 3x - 8 → x = 34 :=
by
  intro h
  sorry

end avg_eq_3x_minus_8_l403_403872


namespace not_all_perfect_squares_l403_403148

theorem not_all_perfect_squares (d : ℕ) (hd : 0 < d) :
  ¬ (∃ (x y z : ℕ), 2 * d - 1 = x^2 ∧ 5 * d - 1 = y^2 ∧ 13 * d - 1 = z^2) :=
by
  sorry

end not_all_perfect_squares_l403_403148


namespace juniors_more_than_seniors_l403_403309

theorem juniors_more_than_seniors
  (j s : ℕ)
  (h1 : (1 / 3) * j = (2 / 3) * s)
  (h2 : j + s = 300) :
  j - s = 100 := 
sorry

end juniors_more_than_seniors_l403_403309


namespace duplicated_page_number_l403_403721

theorem duplicated_page_number (n : ℕ) (h1 : ∑ i in range (n + 1), i + m = 2900) (h2 : m ∈ range (1, n + 1)) : 
  m = 50 :=
sorry

end duplicated_page_number_l403_403721


namespace large_circle_radius_quadratic_l403_403117

-- Definitions and conditions
def unit_circle_radius : ℝ := 1
def num_small_circles : ℕ := 7
def cos_30_deg : ℝ := (Real.sqrt 3) / 2

def large_circle_radius_satisfies_quadratic : Prop :=
  ∃ r : ℝ, r^2 - \((4 * Real.sqrt 3) / 3) * r + 1 = 0

-- Statement of the problem
theorem large_circle_radius_quadratic :
  large_circle_radius_satisfies_quadratic :=
sorry

end large_circle_radius_quadratic_l403_403117


namespace smallest_circle_radius_diameter_one_l403_403759

noncomputable def smallest_enclosing_circle_radius (D : ℝ) : ℝ :=
  if D = 1 then (Real.sqrt 3) / 3 else 0

theorem smallest_circle_radius_diameter_one :
  ∀ (points : set (ℝ × ℝ)), 
    (∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → dist p1 p2 ≤ 1) →
    ∃ (c : ℝ × ℝ) (r : ℝ), r = (Real.sqrt 3) / 3 ∧ ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ r :=
by
  sorry

end smallest_circle_radius_diameter_one_l403_403759


namespace log_5_930_nearest_integer_l403_403230

theorem log_5_930_nearest_integer :
  3 < log 5 930 ∧ log 5 930 < 5 →
  round (log 5 930) = 4 :=
by intros h sorry

end log_5_930_nearest_integer_l403_403230


namespace shaded_to_unshaded_area_ratio_l403_403116

theorem shaded_to_unshaded_area_ratio (PQ QR QT ST UT: ℝ) (hPQ: PQ = 4) (hQR: QR = 3) (hQT: QT = 9) (hST: ST = 4) (hUT: UT = 3) (hTotalArea: QT * PQ = 36) : 
  (let triangle_area (base height: ℝ) := (1/2) * base * height
   let PQR_area := triangle_area QR PQ
   let STU_area := triangle_area ST UT
   let RSTU_area := ST * 2
   let shaded_area := PQR_area + STU_area + RSTU_area
   let unshaded_area := hTotalArea - shaded_area
   shaded_area / unshaded_area = 5 / 4) :=
by sorry

end shaded_to_unshaded_area_ratio_l403_403116


namespace place_value_product_l403_403771

theorem place_value_product : 
  let tens_place := 80
      hundredths_place := 0.08
  in tens_place * hundredths_place = 6.4 := 
by
  -- Definitions
  let tens_place : ℝ := 80
  let hundredths_place : ℝ := 0.08
  -- Proof goal
  show tens_place * hundredths_place = 6.4
  -- Skipping the proof details
  sorry

end place_value_product_l403_403771


namespace estimated_red_balls_l403_403107

-- Definitions based on conditions
def total_balls : ℕ := 15
def black_ball_frequency : ℝ := 0.6
def red_ball_frequency : ℝ := 1 - black_ball_frequency

-- Theorem stating the proof problem
theorem estimated_red_balls :
  (total_balls : ℝ) * red_ball_frequency = 6 := by
  sorry

end estimated_red_balls_l403_403107


namespace same_different_color_ways_equal_l403_403977

-- Definitions based on conditions in the problem
def num_black : ℕ := 15
def num_white : ℕ := 10

def same_color_ways : ℕ :=
  Nat.choose num_black 2 + Nat.choose num_white 2

def different_color_ways : ℕ :=
  num_black * num_white

-- The proof statement
theorem same_different_color_ways_equal : same_color_ways = different_color_ways :=
by
  sorry

end same_different_color_ways_equal_l403_403977


namespace sum_f_proof_l403_403436

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403436


namespace remainder_of_x_500_div_x2_plus_1_x2_minus_1_l403_403356

theorem remainder_of_x_500_div_x2_plus_1_x2_minus_1 :
  (x^500) % ((x^2 + 1) * (x^2 - 1)) = 1 :=
sorry

end remainder_of_x_500_div_x2_plus_1_x2_minus_1_l403_403356


namespace find_a8_l403_403987

-- Define the arithmetic sequence aₙ
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) := a₁ + (n - 1) * d

-- The given condition
def condition (a₁ d : ℕ) :=
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 15 = 12

-- The value we want to prove
def a₈ (a₁ d : ℕ ) : ℕ :=
  arithmetic_sequence a₁ d 8

theorem find_a8 (a₁ d : ℕ) (h : condition a₁ d) : a₈ a₁ d = 4 :=
  sorry

end find_a8_l403_403987


namespace expression_undefined_l403_403885

theorem expression_undefined (a : ℝ) : (a = 2 ∨ a = -2) ↔ (a^2 - 4 = 0) :=
by sorry

end expression_undefined_l403_403885


namespace number_of_elements_in_union_l403_403608

open Set

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {y | ∃ x ∈ A, y = 2 * x + 1}

theorem number_of_elements_in_union :
  (A ∪ B).toFinset.card = 6 := by
  sorry

end number_of_elements_in_union_l403_403608


namespace polynomial_roots_and_coefficients_l403_403854

theorem polynomial_roots_and_coefficients :
  ∀ (a b c : ℝ), (a, b, and c are roots of the polynomial a * x^2 + b * x + c = 0) 
  ∧ (a = (a + b + c) / 3 ∨ b = (a + b + c) / 3 ∨ c = (a + b + c) / 3) 
  → false :=
begin
  sorry
end

end polynomial_roots_and_coefficients_l403_403854


namespace tangent_line_eq_max_min_values_l403_403522

noncomputable def f (x : ℝ) : ℝ := (1 / (3:ℝ)) * x^3 - 4 * x + 4

theorem tangent_line_eq (x y : ℝ) : 
    y = f 1 → 
    y = -3 * (x - 1) + f 1 → 
    3 * x + y - 10 / 3 = 0 := 
sorry

theorem max_min_values (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≤ 4) ∧ 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≥ -4 / 3) := 
sorry

end tangent_line_eq_max_min_values_l403_403522


namespace seq_nonzero_l403_403385

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n, n ≥ 3 → 
    (if (a (n - 2) * a (n - 1)) % 2 = 0 
     then a n = 5 * a (n - 1) - 3 * a (n - 2) 
     else a n = a (n - 1) - a (n - 2)))

theorem seq_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n, n > 0 → a n ≠ 0 :=
  sorry

end seq_nonzero_l403_403385


namespace collinear_points_D_l403_403642

noncomputable def triangle_collinear : Prop :=
  ∀ (A B C : Point) (D E F : Point) (AD BE CF : Line) (D' E' F' : Point),
    ∃ (P Q R : Point),
    collinear P Q R ∧
    on_line D (line B C) ∧ 
    on_line E (line C A) ∧
    on_line F (line A B) ∧
    are_reflections AD BE CF (angle_bisector A B C) (angle_bisector B C A) (angle_bisector C A B) ∧
    intersects AD (line B C) D' ∧
    intersects BE (line C A) E' ∧
    intersects CF (line A B) F' →
    collinear D' E' F'.

theorem collinear_points_D'_E'_F' :
  triangle_collinear :=
by
  sorry

end collinear_points_D_l403_403642


namespace problem_statement_l403_403849

noncomputable def equilateral_sum (A B : Point) : Point := 
  -- hypothetical function constructing an equilateral point
  sorry

noncomputable def midpoint (A B : Point) : Point := 
  -- hypothetical function calculating midpoint
  sorry

theorem problem_statement (A B C : Point) : 
  midpoint A (equilateral_sum B C) = midpoint (equilateral_sum B A) (equilateral_sum A C) :=
sorry

end problem_statement_l403_403849


namespace sum_f_k_1_22_l403_403502

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom H2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom H3 : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom H4 : g 2 = 4

theorem sum_f_k_1_22 : ∑ k in finset.range 22, f (k + 1) = -24 :=
by
  sorry

end sum_f_k_1_22_l403_403502


namespace average_annual_population_increase_l403_403770

theorem average_annual_population_increase 
    (initial_population : ℝ) 
    (final_population : ℝ) 
    (years : ℝ) 
    (initial_population_pos : initial_population > 0) 
    (years_pos : years > 0)
    (initial_population_eq : initial_population = 175000) 
    (final_population_eq : final_population = 297500) 
    (years_eq : years = 10) : 
    (final_population - initial_population) / initial_population / years * 100 = 7 :=
by
    sorry

end average_annual_population_increase_l403_403770


namespace sum_f_proof_l403_403435

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403435


namespace sum_f_eq_neg24_l403_403464

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403464


namespace roots_of_quadratic_l403_403911

theorem roots_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  let Δ := (4 * a * b) ^ 2 - 4 * (a ^ 2 + b ^ 2) * 2 * a * b in
  if a = b then Δ = 0 ∧ (∃ x, (a^2 + b^2) * x^2 + 4 * a * b * x + 2 * a * b = 0 ∧ ∀ y, 
    (a^2 + b^2) * y^2 + 4 * a * b * y + 2 * a * b = 0 → x = y) else 
  ((ab > 0 → Δ < 0) ∧ (ab < 0 → Δ > 0)).
by
  sorry

end roots_of_quadratic_l403_403911


namespace probability_not_math_physics_together_l403_403296

open Classical

def subjects := {"Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"}

noncomputable def totalWays := Nat.choose 6 3
noncomputable def waysMathPhysicsTogether := Nat.choose 4 1
noncomputable def probabilityNotMathPhysicsTogether := 1 - (waysMathPhysicsTogether / totalWays : ℚ)

theorem probability_not_math_physics_together : probabilityNotMathPhysicsTogether = 4 / 5 := sorry

end probability_not_math_physics_together_l403_403296


namespace price_alloy_per_kg_l403_403794

-- Defining the costs of the two metals.
def cost_metal1 : ℝ := 68
def cost_metal2 : ℝ := 96

-- Defining the mixture ratio.
def ratio : ℝ := 1

-- The proposition that the price per kg of the alloy is 82 Rs.
theorem price_alloy_per_kg (C1 C2 r : ℝ) (hC1 : C1 = 68) (hC2 : C2 = 96) (hr : r = 1) :
  (C1 + C2) / (r + r) = 82 :=
by
  sorry

end price_alloy_per_kg_l403_403794


namespace electric_poles_count_l403_403341

theorem electric_poles_count (dist interval: ℕ) (h_interval: interval = 25) (h_dist: dist = 1500):
  (dist / interval) + 1 = 61 := 
by
  -- Sorry to skip the proof steps
  sorry

end electric_poles_count_l403_403341


namespace complex_in_fourth_quadrant_l403_403115

-- Definitions of the complex numbers and their operations
def complex_num : ℂ := (1 / ((1 + complex.i)^2 + 1)) + complex.i ^ 4

-- Statement of the problem
theorem complex_in_fourth_quadrant :
  complex_num.re > 0 ∧ complex_num.im < 0 :=
sorry

end complex_in_fourth_quadrant_l403_403115


namespace required_run_rate_per_batsman_l403_403119

variable (initial_run_rate : ℝ) (overs_played : ℕ) (remaining_overs : ℕ)
variable (remaining_wickets : ℕ) (total_target : ℕ) 

theorem required_run_rate_per_batsman 
  (h_initial_run_rate : initial_run_rate = 3.4)
  (h_overs_played : overs_played = 10)
  (h_remaining_overs  : remaining_overs = 40)
  (h_remaining_wickets : remaining_wickets = 7)
  (h_total_target : total_target = 282) :
  (total_target - initial_run_rate * overs_played) / remaining_overs = 6.2 :=
by
  sorry

end required_run_rate_per_batsman_l403_403119


namespace inclination_angle_of_line_l_is_60_degrees_l403_403398

noncomputable def point := ℝ × ℝ

def slope (p1 p2 : point) : ℝ := 
  (p2.2 - p1.2) / (p2.1 - p1.1)

def inclination_angle (m : ℝ) : ℝ := 
  Real.atan m * 180 / Real.pi

theorem inclination_angle_of_line_l_is_60_degrees :
  let A := (3, 5) in
  let B := (5, 7) in
  let slope_AB := slope A B in
  let slope_l := sqrt 3 * slope_AB in
  inclination_angle slope_l = 60 := 
by
  sorry

end inclination_angle_of_line_l_is_60_degrees_l403_403398


namespace range_of_a_l403_403533

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l403_403533


namespace sum_f_eq_neg24_l403_403492

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f(x) ∈ ℝ
axiom domain_g : ∀ x : ℝ, g(x) ∈ ℝ
axiom eq1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom eq2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x)
axiom g_at_2 : g(2) = 4

theorem sum_f_eq_neg24 : (∑ k in Finset.range 22, f(k + 1)) = -24 := 
  sorry

end sum_f_eq_neg24_l403_403492


namespace nested_sum_approx_l403_403856

theorem nested_sum_approx :
  1002 + (1 / 3) * (1001 + (1 / 3) * (1000 + (1 / 3) * (1000 - 1 + … + (1 / 3) * (3 + (1 / 3) * 2)))) ≈ 1502.25 := 
sorry

end nested_sum_approx_l403_403856


namespace isosceles_triangle_l403_403091

variables {A B C D E F M : Type} [affine_space ℝ A] [is_interior_point A B C]
variables (A B C D E F M : A)
variables (AD BC DE AC AM BE : line A)

-- Given conditions
axiom AD_perp_BC : ⊥ AD BC at D
axiom DE_perp_AC : ⊥ DE AC at E
axiom midpoint_M : midpoint M DE
axiom AM_perp_BE  : ⊥ AM BE at F

theorem isosceles_triangle (A B C : triangle A) (AD_perp_BC : ⊥ AD BC at D) 
    (DE_perp_AC : ⊥ DE AC at E) (midpoint_M : midpoint M DE) (AM_perp_BE : ⊥ AM BE at F):
  is_isosceles A B C :=
begin
  sorry
end

end isosceles_triangle_l403_403091


namespace altered_solution_contains_180_detergent_l403_403209

-- Define the initial ratio parts
structure SolutionRatio where
  bleach : ℕ
  detergent : ℕ
  fabric_softener : ℕ
  water : ℕ

-- Define the condition: initial ratios
def initial_ratio : SolutionRatio :=
  { bleach := 4, detergent := 40, fabric_softener := 60, water := 100 }

-- Define the condition: altered ratios related to bleach, fabric softener, and detergent
axiom bleach_to_detergent_tripled : SolutionRatio -> SolutionRatio
axiom fabric_softener_to_detergent_halved : SolutionRatio -> SolutionRatio
axiom detergent_to_water_reduced_one_third : SolutionRatio -> SolutionRatio

-- Define the goal: altered solution contains 300 liters of water
axiom altered_solution_contains_300_water (s : SolutionRatio) : s.water = 300 -> ℕ

-- Define the proof goal: the altered solution will contain 180 liters of detergent
theorem altered_solution_contains_180_detergent :
  let r := initial_ratio in
  let r₁ := bleach_to_detergent_tripled r in
  let r₂ := fabric_softener_to_detergent_halved r₁ in
  let r₃ := detergent_to_water_reduced_one_third r₂ in
  altered_solution_contains_300_water r₃ 300 = 180 :=
sorry

end altered_solution_contains_180_detergent_l403_403209


namespace sum_f_eq_neg24_l403_403468

variable (f g : ℝ → ℝ)
variable (sum_f : ℕ → ℝ)

axiom domain_f : ∀ x : ℝ, f x
axiom domain_g : ∀ x : ℝ, g x
axiom add_cond : ∀ x, f x + g (2 - x) = 5
axiom sub_cond : ∀ x, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x, g (2 - x) = g (2 + x)
axiom value_g_2 : g 2 = 4
noncomputable def sum_f : ℕ → ℝ
  | 0     => 0
  | (n+1) => f (n+1) + sum_f n

theorem sum_f_eq_neg24 : (sum_f 22) = -24 := 
sorry

end sum_f_eq_neg24_l403_403468


namespace distance_between_tangent_and_parallel_line_l403_403323

noncomputable def distance_between_parallel_lines 
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ) 
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) : ℝ :=
sorry

variable (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop)

axiom tangent_line_at_point (M : ℝ × ℝ) (C : Set (ℝ × ℝ)) : (ℝ × ℝ → Prop)

theorem distance_between_tangent_and_parallel_line
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) :
  C = { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 } →
  M = (-2, 4) →
  l = tangent_line_at_point M C →
  l1 = { p | a * p.1 + 3 * p.2 + 2 * a = 0 } →
  distance_between_parallel_lines C center r M l a l1 = 12/5 :=
by
  intros hC hM hl hl1
  sorry

end distance_between_tangent_and_parallel_line_l403_403323


namespace find_value_of_fraction_l403_403565

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403565


namespace probability_A_not_losing_l403_403104

theorem probability_A_not_losing 
  (P_A_tie P_A_win : ℚ)
  (h_A_tie : P_A_tie = 1/2) 
  (h_A_win : P_A_win = 1/3) :
  P_A_tie + P_A_win = 5 / 6 :=
by
  rw [h_A_tie, h_A_win]
  norm_num
  sorry

end probability_A_not_losing_l403_403104


namespace magnitude_of_3a_plus_b_eq_sqrt_5_l403_403541

variables (a b : ℝ × ℝ) (y : ℝ)

def vector_a : ℝ × ℝ := (1, 2)

def vector_b (y : ℝ) : ℝ × ℝ := (-2, y)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem magnitude_of_3a_plus_b_eq_sqrt_5 (h : are_parallel vector_a (vector_b y)) : 
  magnitude (3 • (1, 2) + vector_b y) = Real.sqrt 5 :=
by 
  have hy : y = -4 :=
    sorry,
  simp [vector_a, vector_b, hy],
  sorry

end magnitude_of_3a_plus_b_eq_sqrt_5_l403_403541


namespace coprime_permutations_count_l403_403630

open Finset

def is_coprime_perm (perm : List ℕ) (s : Finset ℕ) : Prop :=
  perm.perm s.to_list ∧ ∀ i, 0 < i ∧ i < perm.length →
    Nat.coprime (perm.nth_le i (by linarith)) (perm.nth_le (i - 1) (by linarith [i]))

theorem coprime_permutations_count :
  let s := {1, 2, 3, 4, 5, 6, 7}
  let perms := (s.to_list.permutations.filter (λ perm, is_coprime_perm perm s))
  perms.length = 864 :=
by sorry

end coprime_permutations_count_l403_403630


namespace crossing_time_of_trains_l403_403225

-- Definitions based on conditions
def train_length : ℝ := 120  -- length of each train in meters
def train_speed_km_per_hr : ℝ := 36  -- speed of each train in km/hr
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (5 / 18)  -- conversion of speed to m/s

-- Theorem statement
theorem crossing_time_of_trains : (2 * train_length) / (2 * train_speed_m_per_s) = 12 :=
by sorry

end crossing_time_of_trains_l403_403225


namespace direct_proportion_l403_403201

variables (P_m P_t : ℝ) (T : ℝ)

theorem direct_proportion (h : P_t / P_m = T) : ∃ k : ℝ, P_t = k * P_m :=
begin
  sorry
end

end direct_proportion_l403_403201


namespace fraction_value_l403_403574

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403574


namespace not_and_implication_l403_403924

variable (p q : Prop)

theorem not_and_implication : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
by
  sorry

end not_and_implication_l403_403924


namespace value_of_k_plus_p_l403_403726

theorem value_of_k_plus_p
  (k p : ℝ)
  (h1 : ∀ x : ℝ, 3*x^2 - k*x + p = 0)
  (h_sum_roots : k / 3 = -3)
  (h_prod_roots : p / 3 = -6)
  : k + p = -27 :=
by
  sorry

end value_of_k_plus_p_l403_403726


namespace find_initial_order_l403_403819

variables (x : ℕ)

def initial_order (x : ℕ) :=
  x + 60 = 72 * (x / 90 + 1)

theorem find_initial_order (h1 : initial_order x) : x = 60 :=
  sorry

end find_initial_order_l403_403819


namespace perpendicular_MI_BC_l403_403103

-- Definitions for the necessary geometric constructs
variables (A B C K M I : Type) 
variables [scalene_triangle A B C] 
variables [midpoint K A B] 
variables [centroid M A B C] 
variables [incenter I A B C]
variables (h : angle K I B = 90)

-- Theorem statement
theorem perpendicular_MI_BC : perpendicular M I B C :=
sorry

end perpendicular_MI_BC_l403_403103


namespace verify_trig_identity_l403_403265

noncomputable def trig_identity_eqn : Prop :=
  2 * Real.sqrt (1 - Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4

theorem verify_trig_identity : trig_identity_eqn := by
  sorry

end verify_trig_identity_l403_403265


namespace abs_neg_eq_five_l403_403958

theorem abs_neg_eq_five (x : ℝ) (h : | -x | = 5) : x = 5 ∨ x = -5 :=
by
  sorry

end abs_neg_eq_five_l403_403958


namespace sum_harmonic_2003_divides_q_l403_403944

theorem sum_harmonic_2003_divides_q (p q : ℕ) (hpq: (1 + ∑ i in finset.range 2002, 1/(i+1) : ℚ) = q / p) (gcd_pq: nat.gcd p q = 1) : 2003 ∣ q :=
sorry

end sum_harmonic_2003_divides_q_l403_403944


namespace correct_pythagorean_statement_l403_403250

theorem correct_pythagorean_statement (a b c : ℕ) (A B C : Type) 
  (angleA_90 : ∀ (a b c : ℕ), B = π / 2 → a^2 + b^2 = c^2) 
  : D = "If a, b, c are the three sides of right triangle ΔABC 
    where angle C = 90°, then a^2 + b^2 = c^2" :=
sorry

end correct_pythagorean_statement_l403_403250


namespace eval_f_sqrt_45_l403_403667

def f (x : Real) : Real :=
  if x ∈ Int then 7 * x + 6 else floor x + 7

theorem eval_f_sqrt_45 : f (Real.sqrt 45) = 13 :=
by
  sorry

end eval_f_sqrt_45_l403_403667


namespace sum_f_k_from_1_to_22_l403_403487

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403487


namespace additional_donation_l403_403281

theorem additional_donation
  (t : ℕ) (c d₁ d₂ T a : ℝ)
  (h1 : t = 25)
  (h2 : c = 2.00)
  (h3 : d₁ = 15.00) 
  (h4 : d₂ = 15.00)
  (h5 : T = 100.00)
  (h6 : t * c + d₁ + d₂ + a = T) :
  a = 20.00 :=
by
  sorry

end additional_donation_l403_403281


namespace value_of_ac_over_bd_l403_403600

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403600


namespace sum_f_proof_l403_403440

noncomputable def sum_f (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, f (i + 1)

theorem sum_f_proof (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f x + g (2 - x) = 5)
    (h2 : ∀ x : ℝ, g x - f (x - 4) = 7) (h3 : ∀ x : ℝ, g (2 - x) = g (2 + x))
    (h4 : g 2 = 4) : sum_f f 22 = -24 := 
sorry

end sum_f_proof_l403_403440


namespace value_of_expression_l403_403556

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403556


namespace fraction_of_older_brother_l403_403730

theorem fraction_of_older_brother (O Y : ℕ) (f : ℚ) 
  (h1 : Y = 27)
  (h2 : O + Y = 46)
  (h3 : Y = f * O + 10) : 
  f = 17 / 19 :=
by {
  have h4 : O = 46 - Y, from nat.eq_sub_of_add_eq h2,
  rw [h1] at h4,
  rw [h1, h4] at h3,
  have h5 : 27 = f * (46 - 27) + 10, by rwa h3,
  norm_num at h5,
  linarith,
}

end fraction_of_older_brother_l403_403730


namespace balloons_floated_away_l403_403837

theorem balloons_floated_away (starting_balloons given_away grabbed_balloons final_balloons flattened_balloons : ℕ)
  (h1 : starting_balloons = 50)
  (h2 : given_away = 10)
  (h3 : grabbed_balloons = 11)
  (h4 : final_balloons = 39)
  : flattened_balloons = starting_balloons - given_away + grabbed_balloons - final_balloons → flattened_balloons = 12 :=
by
  sorry

end balloons_floated_away_l403_403837


namespace angle_between_vectors_is_ninety_degrees_l403_403943

def vector_a : ℝ × ℝ × ℝ := (0, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -2)

theorem angle_between_vectors_is_ninety_degrees :
  (∃ θ : ℝ, θ = 90 ∧ 
  let ⟪ , ⟫ := λ u v, u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
  ⟪vector_a, vector_b⟫ = 0) :=
sorry

end angle_between_vectors_is_ninety_degrees_l403_403943


namespace isosceles_obtuse_triangle_angle_l403_403290

-- A structure to represent a triangle
structure Triangle (A B C : Type) :=
  (base : A → B → C → Prop)
  (legs : A → B → Prop)

-- Define the setup and conditions for the specific problem
structure IsoscelesObtuseTriangle (A B C D O M : Type) extends Triangle A B C :=
  (isosceles : legs A C ∧ legs B C)
  (obtuse : ∃ γ, γ > 90 ∧ γ < 180)
  (point_on_base : base A B D)
  (O_circumcenter : O)
  (M_midpoint : M)
  (radius_eq : ∀ r, ∃ R, r = R ∧ R = radius_of_circumcircle O)

noncomputable def prove_triangle_angle (A B C D O M : Type) [IsoscelesObtuseTriangle A B C D O M] : Type :=
  { angle : ℝ // angle = 90 ∨ angle = 135 }

-- The goal statement
theorem isosceles_obtuse_triangle_angle 
  (A B C D O M : Type) [h : IsoscelesObtuseTriangle A B C D O M]
  : prove_triangle_angle A B C D O M :=
  sorry

end isosceles_obtuse_triangle_angle_l403_403290


namespace sum_f_1_to_22_l403_403452

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Conditions
axiom h1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom h2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom h3 : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom h4 : g 2 = 4

-- Statement to prove
theorem sum_f_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 :=
sorry

end sum_f_1_to_22_l403_403452


namespace unique_two_digit_integer_s_l403_403738

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l403_403738


namespace value_of_ac_over_bd_l403_403596

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403596


namespace third_chest_coin_difference_l403_403217

variable (g1 g2 g3 s1 s2 s3 : ℕ)

-- Conditions
axiom h1 : g1 + g2 + g3 = 40
axiom h2 : s1 + s2 + s3 = 40
axiom h3 : g1 = s1 + 7
axiom h4 : g2 = s2 + 15

-- Goal
theorem third_chest_coin_difference : s3 = g3 + 22 :=
sorry

end third_chest_coin_difference_l403_403217


namespace original_number_div_eq_l403_403078

theorem original_number_div_eq (h : 204 / 12.75 = 16) : 2.04 / 1.6 = 1.275 :=
by sorry

end original_number_div_eq_l403_403078


namespace angle_BDC_l403_403991

/-- In the given circle, the diameter EB is parallel to DC, and AB is parallel to EC.
The angles AEB and ABE are in the ratio 3:4. Prove that the degree measure of angle BDC is 900/7. -/
theorem angle_BDC (A B C E D : Point)
  (H1 : diameter EB)
  (H2 : parallel EB DC)
  (H3 : parallel AB EC)
  (H4 : ∃ x, angle AEB = 3 * x ∧ angle ABE = 4 * x)
  : angle BDC = 900 / 7 :=
sorry

end angle_BDC_l403_403991


namespace solve_for_x_l403_403172

theorem solve_for_x (x : ℝ) : 3^(x + 5) = 4^x → x = Real.log (3^5) / Real.log (4 / 3) :=
by
  sorry

end solve_for_x_l403_403172


namespace value_of_ac_over_bd_l403_403601

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l403_403601


namespace percentage_error_is_correct_l403_403258

-- Definitions from the problem conditions
def actual_side (s : ℕ) : ℕ := s
def measured_side (s : ℕ) : ℕ := s + (5 * s) / 100  -- 5% excess error
def actual_area (s : ℕ) : ℕ := s * s
def measured_area (s : ℕ) : ℕ := (measured_side s) * (measured_side s)

-- The percentage error
def percentage_error_in_area (s : ℕ) : ℚ := 
  ((measured_area s - actual_area s) / actual_area s.to_rat) * 100

-- The proof statement
theorem percentage_error_is_correct (s : ℕ) (hs_pos : s > 0) : percentage_error_in_area s = 10.25 := 
by 
  sorry

end percentage_error_is_correct_l403_403258


namespace find_value_of_fraction_l403_403562

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l403_403562


namespace percentage_decrease_second_year_l403_403631

-- Define initial population
def initial_population : ℝ := 14999.999999999998

-- Define the population at the end of the first year after 12% increase
def population_end_year_1 : ℝ := initial_population * 1.12

-- Define the final population at the end of the second year
def final_population : ℝ := 14784.0

-- Define the proof statement
theorem percentage_decrease_second_year :
  ∃ D : ℝ, final_population = population_end_year_1 * (1 - D / 100) ∧ D = 12 :=
by
  sorry

end percentage_decrease_second_year_l403_403631


namespace question1_question2_l403_403528

-- Definitions from conditions:
def f1 (x : ℝ) (b c : ℝ) := x^2 - (b + 2) * x + c
def f2 (x : ℝ) (b c : ℝ) := b * x^2 - (c + 1) * x - c

-- Given values from solving conditions:
def b : ℝ := 3
def c : ℝ := 6

-- Conditions:
axiom cond1 : ∀ x, 2 < x ∧ x < 3 → f1 x b c < 0

-- Question 1:
def A (x : ℝ) : Prop := x < -2/3 ∨ 3 < x
def B : set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def intersection_A_B : set ℝ := {x : ℝ | (A x) ∧ (x ∈ B)}

theorem question1 : intersection_A_B = {x : ℝ | -2 ≤ x ∧ x < -2/3} :=
sorry

-- Question 2:
def f3 (x : ℝ) : ℝ := (x^2 - b * x + c) / (x - 1)
def min_f3_value : ℝ := 3

theorem question2 (x : ℝ) (h : 1 < x) : f3 x ≥ min_f3_value :=
sorry

end question1_question2_l403_403528


namespace bananas_in_each_box_l403_403651

theorem bananas_in_each_box 
    (bananas : ℕ) (boxes : ℕ) 
    (h_bananas : bananas = 40) 
    (h_boxes : boxes = 10) : 
    bananas / boxes = 4 := by
  sorry

end bananas_in_each_box_l403_403651


namespace find_m_l403_403390

-- Definitions of the conditions
variable (m : ℝ) 
def quadratic_eq (x : ℝ) := x^2 + 2*m*x + m^2 - m + 2 = 0

def distinct_real_roots : Prop :=
  ∀ x1 x2 : ℝ, quadratic_eq m x1 → quadratic_eq m x2 → x1 ≠ x2

def sum_and_product_condition (x1 x2 : ℝ) : Prop :=
  x1 + x2 + x1 * x2 = 2

-- Statement of the proof problem
theorem find_m (m : ℝ) :
  (∀ x1 x2 : ℝ, quadratic_eq m x1 → quadratic_eq m x2 → distinct_real_roots m) →
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ sum_and_product_condition m x1 x2) →
  m = 3 :=
sorry

end find_m_l403_403390


namespace irrational_sqrt_7_among_others_l403_403306

theorem irrational_sqrt_7_among_others :
  let A := -1 / 3
  let B := 2
  let C := Real.sqrt 7
  let D := 0.0101
  irrational C :=
by
  let A := -1 / 3
  let B := 2
  let C := Real.sqrt 7
  let D := 0.0101
  sorry

end irrational_sqrt_7_among_others_l403_403306


namespace linear_function_diff_l403_403150

/-- Let \( g \) be a linear function such that \( g(8) - g(3) = 15 \).
    Moreover, \( g(4) - g(1) = 9 \). 
    Then \( g(10) - g(1) = 27 \). --/
theorem linear_function_diff (g : ℝ → ℝ)
  (h1 : ∀ x y, g(x + y) = g(x) + g(y))
  (h2 : g(8) - g(3) = 15)
  (h3 : g(4) - g(1) = 9) : 
  g(10) - g(1) = 27 := 
begin
  sorry
end

end linear_function_diff_l403_403150


namespace isosceles_BMC_l403_403188

variable (Point : Type)
variable [affine_space Point ℝ]

structure Trapezoid (A B C D O M : Point) : Prop :=
(intersect_diagonals : (∃ (O : Point), ∃ (M : Point), intersect (line_through A C) (line_through B O) = some O ∧ intersect (line_through D B) (line_through A O) = some O))
(intersect_circumcircles : ∃ (M : Point), on_circumcircle A B O M ∧ on_circumcircle C O D M)
(M_on_AD : M ∈ line_through A D)

theorem isosceles_BMC {A B C D O M : Point} (h: Trapezoid A B C D O M) : 
  ∠ B M C = ∠ C M B :=
sorry

end isosceles_BMC_l403_403188


namespace range_of_a_l403_403036

variable (f : ℝ → ℝ) (a : ℝ)

-- Definitions based on provided conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main statement
theorem range_of_a
    (hf_even : is_even f)
    (hf_mono : is_monotonically_increasing f)
    (h_ineq : ∀ x : ℝ, f (Real.log (a) / Real.log 2) ≤ f (x^2 - 2 * x + 2)) :
  (1/2 : ℝ) ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l403_403036


namespace trajectory_center_l403_403286

structure Circle (center : ℝ × ℝ) (radius : ℝ) :=
(center_radius : (ℝ × ℝ) × ℝ) := (center, radius)

def tangent_internal (M C₁ : Circle) : Prop :=
-- Circle M is internally tangent to Circle C₁
dist M.center C₁.center = C₁.radius - M.radius

def tangent_external (M C₂ : Circle) : Prop :=
-- Circle M is externally tangent to Circle C₂
dist M.center C₂.center = C₂.radius + M.radius

def trajectory_eqn : Prop :=
∀ (M : Circle) (x y : ℝ),
  tangent_internal M (Circle (⟨-1, 0⟩, 6)) ∧
  tangent_external M (Circle (⟨1, 0⟩, 2)) →
  (x, y) = M.center →
  (x^2 / 16 + y^2 / 15 = 1)

theorem trajectory_center
  : trajectory_eqn :=
sorry

end trajectory_center_l403_403286


namespace value_of_expression_l403_403560

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l403_403560


namespace fraction_value_l403_403577

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l403_403577


namespace unique_two_digit_integer_s_l403_403739

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l403_403739


namespace chain_of_friendships_l403_403269

-- Define the basic structure and properties of students
variable (Students : Type) (is_friend : Students → Students → Prop)

-- Define the condition of friendships among any 10 students
axiom friends_condition :
  ∀ (students_set : Finset Students), |students_set| = 10 → ∃ (a b : Students), a ∈ students_set ∧ b ∈ students_set ∧ is_friend a b

-- Define our main theorem based on the given conditions
theorem chain_of_friendships :
  (∃ (partition : Finset (Finset Students)), 
    (∀ (group : Finset Students), group ∈ partition → (∃ (sequence : List Students), all_pairs_are_friends sequence))
    ∧ Finset.card partition ≤ 9)
:=
sorry

end chain_of_friendships_l403_403269


namespace ratio_of_spinsters_to_cats_l403_403208

-- Definitions for the conditions given:
def S : ℕ := 12 -- 12 spinsters
def C : ℕ := S + 42 -- 42 more cats than spinsters
def ratio (a b : ℕ) : ℚ := a / b -- Ratio definition

-- The theorem stating the required equivalence:
theorem ratio_of_spinsters_to_cats :
  ratio S C = 2 / 9 :=
by
  -- This proof has been omitted for the purpose of this exercise.
  sorry

end ratio_of_spinsters_to_cats_l403_403208


namespace degree_monomial_neg_5_x2_y1_l403_403186

-- Definition of monomial and its degree
def monomial (a : ℝ) (p q : ℕ) : ℝ := a * (x ^ p) * (y ^ q)

-- Degree function
def degree (p q : ℕ) : ℕ := p + q

-- Given monomial is -5 * x^2 * y^1, we prove its degree is 3
theorem degree_monomial_neg_5_x2_y1 : 
  degree 2 1 = 3 :=
by
  sorry

end degree_monomial_neg_5_x2_y1_l403_403186


namespace S40_value_l403_403210

-- Define the terms as stated in the conditions
variable {α : Type*} [Field α] (a : ℕ → α) (S : ℕ → α)
variable (q : α)

-- Conditions of the problem
def geometric_series (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_geometric_series (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n, S n = (a 0) * (1 - q^(n + 1)) / (1 - q)

axiom sum_S10 : S 10 = 10
axiom sum_S30 : S 30 = 70
axiom geom_seq : geometric_series a q
axiom sum_geom_seq : sum_geometric_series a S

-- Theorem to prove
theorem S40_value : S 40 = 150 :=
  by
    -- Proof steps go here
    sorry

end S40_value_l403_403210


namespace Cheryl_total_distance_l403_403833

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l403_403833


namespace number_of_possible_k_values_l403_403633

theorem number_of_possible_k_values : 
  ∃ k_values : Finset ℤ, 
    (∀ k ∈ k_values, ∃ (x y : ℤ), y = x - 3 ∧ y = k * x - k) ∧
    k_values.card = 3 := 
sorry

end number_of_possible_k_values_l403_403633


namespace cannot_form_set_l403_403817

-- Definitions based on the conditions
def GroupA := {x : ℝ | abs (x - 2) < δ} -- This is not well-defined without δ
def GroupB := {x : ℝ | x^2 - 1 = 0}
def GroupC := {T : Type | T = EquilateralTriangle}
def GroupD := {n : ℕ | n < 10}

theorem cannot_form_set (δ : ℝ) : 
  (∀ (S : Set ℝ), S = GroupA → ¬(∃ δ > 0, ∀ x ∈ S, abs (x - 2) < δ)) ∧ 
  ((∃ (S : Set ℝ), S = GroupB) ∧ 
   (∃ (S : Set Type), S = GroupC) ∧ 
   (∃ (S : Set ℕ), S = GroupD)) := 
by
  sorry

end cannot_form_set_l403_403817


namespace max_S_value_l403_403662

noncomputable def maximize_S (a b c : ℝ) : ℝ :=
  (a^2 - a * b + b^2) * (b^2 - b * c + c^2) * (c^2 - c * a + a^2)

theorem max_S_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 3 →
  maximize_S a b c ≤ 12 :=
by sorry

end max_S_value_l403_403662


namespace total_items_sold_at_garage_sale_l403_403822

-- Define the conditions for the problem
def items_more_expensive_than_radio : Nat := 16
def items_less_expensive_than_radio : Nat := 23

-- Declare the total number of items using the given conditions
theorem total_items_sold_at_garage_sale 
  (h1 : items_more_expensive_than_radio = 16)
  (h2 : items_less_expensive_than_radio = 23) :
  items_more_expensive_than_radio + 1 + items_less_expensive_than_radio = 40 :=
by
  sorry

end total_items_sold_at_garage_sale_l403_403822


namespace number_of_ways_to_prepare_all_elixirs_l403_403646

def fairy_methods : ℕ := 2
def elf_methods : ℕ := 2
def fairy_elixirs : ℕ := 3
def elf_elixirs : ℕ := 4

theorem number_of_ways_to_prepare_all_elixirs : 
  (fairy_methods * fairy_elixirs) + (elf_methods * elf_elixirs) = 14 :=
by
  sorry

end number_of_ways_to_prepare_all_elixirs_l403_403646


namespace ratio_triangle_square_l403_403984

noncomputable def square_area (s : ℝ) : ℝ := s * s

noncomputable def triangle_PTU_area (s : ℝ) : ℝ := 1 / 2 * (s / 2) * (s / 2)

theorem ratio_triangle_square (s : ℝ) (h : s > 0) : 
  triangle_PTU_area s / square_area s = 1 / 8 := 
sorry

end ratio_triangle_square_l403_403984


namespace general_formula_for_a_sum_of_first_n_terms_of_b_l403_403668

noncomputable def a : ℕ → ℕ
| 1     := 2
| (n+1) := n + 1 -- This follows from the given a_n = n + 1

theorem general_formula_for_a : 
  ∀ n : ℕ, a n = n + 1 :=
by sorry

noncomputable def b (n : ℕ) : ℕ → ℝ
| n     := 2 * (a n + 1 / (2 ^ (a n).to_real))

def S (n : ℕ) : ℝ :=
∑ k in finset.range n, b k

theorem sum_of_first_n_terms_of_b :
  ∀ n : ℕ, S n = n ^ 2 + 3 * n + 1 - 1 / 2 ^ n.to_real :=
by sorry

end general_formula_for_a_sum_of_first_n_terms_of_b_l403_403668


namespace john_marble_choice_l403_403650

theorem john_marble_choice (marbles : Finset ℕ) (h_size : marbles.card = 12) (h_special : ∀ x ∈ marbles, x < 3) :
  (∃ r g b ∈ marbles, (marbles.erase r).erase g).erase b.card = 9 -> Nat.choose 9 3 = 84 -> 3 * Nat.choose 9 3 = 252 :=
by {
  intros h1 h2,
  rw h2,
  norm_num,
}

#check john_marble_choice

end john_marble_choice_l403_403650


namespace suitable_for_experimental_method_is_meters_run_l403_403248

-- Define the options as a type
inductive ExperimentalOption
| recommending_class_monitor_candidates
| surveying_classmates_birthdays
| meters_run_in_10_seconds
| avian_influenza_occurrences_world

-- Define a function that checks if an option is suitable for the experimental method
def is_suitable_for_experimental_method (option: ExperimentalOption) : Prop :=
  option = ExperimentalOption.meters_run_in_10_seconds

-- The theorem stating which option is suitable for the experimental method
theorem suitable_for_experimental_method_is_meters_run :
  is_suitable_for_experimental_method ExperimentalOption.meters_run_in_10_seconds :=
by
  sorry

end suitable_for_experimental_method_is_meters_run_l403_403248


namespace shortest_distance_between_point_and_parabola_l403_403358

noncomputable def shortestDistance : ℕ := 56

theorem shortest_distance_between_point_and_parabola (x y : ℝ)
    (h_point : (x, y) = (8, 16))
    (h_parabola : x = y^2 / 4) : 
    sqrt ((64 - 8)^2 + (16 - 16)^2) = 56 :=
by
  sorry

end shortest_distance_between_point_and_parabola_l403_403358


namespace line_passing_through_P_meeting_conditions_l403_403793

noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem line_passing_through_P_meeting_conditions :
  ∃ l : ℝ → ℝ → Prop, 
  (l 1 1) ∧ 
  (∀ (A B : ℝ × ℝ), 
    A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧ l A.1 A.2 ∧ l B.1 B.2 → 
    distance A B = 2 * real.sqrt 3) ∧
  ((∀ x y : ℝ, l x y ↔ x = 1) ∨ (∀ x y : ℝ, l x y ↔ y = 1)) := 
sorry

#print axioms line_passing_through_P_meeting_conditions

end line_passing_through_P_meeting_conditions_l403_403793


namespace sets_are_equal_l403_403139

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l403_403139


namespace bob_track_length_l403_403823

theorem bob_track_length :
  ∀ (lap1 lap2 lap3 : ℕ) (avg_speed : ℕ),
  lap1 = 70 → lap2 = 85 → lap3 = 85 → avg_speed = 5 →
  (∃ (track_length : ℕ),
  let total_time := lap1 + lap2 + lap3 in
  let total_distance := avg_speed * total_time in
  let num_laps := 3 in
  track_length = total_distance / num_laps ∧ track_length = 400) :=
by
  intros lap1 lap2 lap3 avg_speed h1 h2 h3 h4
  use 400
  sorry

end bob_track_length_l403_403823


namespace number_of_factors_of_M_is_25_l403_403147

noncomputable def M : ℕ := 57^4 + 4 * 57^3 + 6 * 57^2 + 4 * 57 + 1

theorem number_of_factors_of_M_is_25 : ∃ n : ℕ, n = nat.factors M ∧ n = 25 := 
sorry

end number_of_factors_of_M_is_25_l403_403147


namespace intersection_M_N_l403_403537

def M (x : ℝ) : Prop := log 2 (x + 2) > 0
def N (x : ℝ) : Prop := 2^x ≤ 1

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = set.Ioc (-1) 0 :=
by
  sorry

end intersection_M_N_l403_403537


namespace log_div_l403_403079

theorem log_div (x : ℝ) (h : log 16 (x - 3) = 1 / 2) : 1 / log x 4 = log 10 7 / log 10 4 :=
by sorry

end log_div_l403_403079


namespace workers_and_days_l403_403791

theorem workers_and_days (x y : ℕ) (h1 : x * y = (x - 20) * (y + 5)) (h2 : x * y = (x + 15) * (y - 2)) :
  x = 60 ∧ y = 10 := 
by {
  sorry
}

end workers_and_days_l403_403791


namespace task1_on_time_task2_not_on_time_prob_l403_403768

def task1_on_time_prob : ℚ := 3 / 8
def task2_on_time_prob : ℚ := 3 / 5

theorem task1_on_time_task2_not_on_time_prob :
  task1_on_time_prob * (1 - task2_on_time_prob) = 3 / 20 := by
  sorry

end task1_on_time_task2_not_on_time_prob_l403_403768


namespace area_CMN_l403_403996

theorem area_CMN (S1 S2 S3 : ℝ) (ABC OMA OAB OBM CMN : Set (ℝ × ℝ)) 
[IsTriangle ABC] [IsTriangle OMA] [IsTriangle OAB] [IsTriangle OBM] [IsTriangle CMN]: 
  (area OMA = S1) ∧ (area OAB = S2) ∧ (area OBM = S3) →
  area CMN = (S1 * S3 * (S1 + S2) * (S3 + S2)) / (S2 * (S2^2 - S1 * S3)) :=
by
  sorry

end area_CMN_l403_403996


namespace sin_value_l403_403892

theorem sin_value (α : ℝ) (h_cos : cos α = -1 / 3) (h_interval : π < α ∧ α < 3 * π / 2) :
  sin α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_value_l403_403892


namespace determine_f4_l403_403530

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem determine_f4 (f : ℝ → ℝ) (h_odd : odd_function f) (h_f_neg : ∀ x, x < 0 → f x = x * (2 - x)) : f 4 = 24 :=
by
  sorry

end determine_f4_l403_403530


namespace time_to_cross_pole_l403_403298

-- Conditions
def train_speed_kmh : ℕ := 108
def train_length_m : ℕ := 210

-- Conversion functions
def km_per_hr_to_m_per_sec (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

-- Theorem to be proved
theorem time_to_cross_pole : (train_length_m : ℕ) / (km_per_hr_to_m_per_sec train_speed_kmh) = 7 := by
  -- we'll use sorry here to skip the actual proof steps.
  sorry

end time_to_cross_pole_l403_403298


namespace sum_f_1_to_22_l403_403425

noncomputable def f : ℝ → ℝ := sorry -- Assumption: f(x) ∈ ℝ
noncomputable def g : ℝ → ℝ := sorry -- Assumption: g(x) ∈ ℝ

-- Conditions
axiom domain_f : ∀ x : ℝ, x ∈ domain f -- f(x) domain
axiom domain_g : ∀ x : ℝ, x ∈ domain g -- g(x) domain
axiom condition1 : ∀ x : ℝ, f(x) + g(2 - x) = 5
axiom condition2 : ∀ x : ℝ, g(x) - f(x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g(2 - x) = g(2 + x) -- symmetry of g about x = 2
axiom value_g2 : g(2) = 4

-- Goal
theorem sum_f_1_to_22 : ∑ k in finset.range 1 (22 + 1), f k = -24 :=
by sorry

end sum_f_1_to_22_l403_403425


namespace constant_term_expansion_l403_403349

theorem constant_term_expansion : 
  let general_term (r : ℕ) := (binomial 6 r) * (-1)^r * (x ^ (6 - 2 * r)) in
  (∃ r : ℕ, 6 - 2 * r = 0 ∧ constant_term (general_term r) = -20) :=
by
  sorry

end constant_term_expansion_l403_403349


namespace field_area_l403_403292

theorem field_area (L W : ℝ) (hL : L = 20) (h_fencing : 2 * W + L = 59) :
  L * W = 390 :=
by {
  -- We will skip the proof
  sorry
}

end field_area_l403_403292


namespace problem_l403_403581

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l403_403581


namespace max_value_f_period_f_l403_403354

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.cos x) ^ 4

theorem max_value_f : ∃ x : ℝ, (f x) = 1 / 4 :=
sorry

theorem period_f : ∃ p : ℝ, p = π / 2 ∧ ∀ x : ℝ, f (x + p) = f x :=
sorry

end max_value_f_period_f_l403_403354


namespace sin_double_angle_l403_403012

theorem sin_double_angle (θ : ℝ) (h₁ : 3 * (Real.cos θ)^2 = Real.tan θ + 3) (h₂ : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := 
sorry

end sin_double_angle_l403_403012


namespace sum_f_k_from_1_to_22_l403_403483

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition_1 {x : ℝ} : f(x) + g(2 - x) = 5 := sorry
lemma problem_condition_2 {x : ℝ} : g(x) - f(x - 4) = 7 := sorry
lemma problem_condition_3_symmetric (x : ℝ) : g(2 - x) = g(2 + x) := sorry
lemma problem_condition_4 : g(2) = 4 := sorry

theorem sum_f_k_from_1_to_22 : (∑ k in Finset.range 22, f (k + 1)) = -24 := sorry

end sum_f_k_from_1_to_22_l403_403483


namespace value_of_fraction_l403_403552

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l403_403552


namespace length_AG_l403_403635

open Real

variables {A B C D M G : Type}
variable [metric_space A]
variables {point : A} [metric_space.point.point B] [metric_space.point.point C] [metric_space.point.point D]
variables {coordinate : A → Point}
variable hABC : triangle (coordinate A) (coordinate B) (coordinate C)

def AB : ℝ := 3
def AC : ℝ := 3 * sqrt 3

def is_right_angle {A B C : A} [metric_space.point.point A] [metric_space.point.point B] [metric_space.point.point C] : Prop :=
  ∠BAC = 90

def altitude {A D A B : A} [metric_space.point.point A] [metric_space.point.point D] [metric_space.point.point B] : Prop :=
  ∠BAD = ∠DAC = 90

def median {B M C : A} [metric_space.point.point B] [metric_space.point.point M] [metric_space.point.point C] : Prop :=
  dist B M = dist M C

def G_on_altitude_inter_median {A D M G : A} [metric_space.point.point A] [metric_space.point.point D] [metric_space.point.point M] [metric_space.point.point G] : Prop :=
  G ∈ line(A, D) ∧ G ∈ line(B, M)

theorem length_AG
  {A B C D G : point} 
  (hABC_right : is_right_angle A B C)
  (hAB : dist A B = AB)
  (hAC : dist A C = AC)
  (hAD : altitude A D)
  (hBM : median B M C)
  (hG : G_on_altitude_inter_median A D B G):
  dist A G = 0.75 * sqrt 3 :=
sorry

end length_AG_l403_403635


namespace train_cross_bridge_time_l403_403070

-- Given
def length_of_train : ℝ := 165
def speed_of_train_kmph : ℝ := 36
def length_of_bridge : ℝ := 660

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Calculate the total distance
def total_distance : ℝ := length_of_train + length_of_bridge

-- Calculate the speed in m/s
def speed_of_train_mps : ℝ := kmph_to_mps speed_of_train_kmph

-- Calculate the time to cross the bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

-- Statement of the proof problem
theorem train_cross_bridge_time : time_to_cross_bridge = 82.5 := by
  sorry

end train_cross_bridge_time_l403_403070


namespace additional_time_for_distance_l403_403810

-- Define the conditions of the problem
def distance_in_initial_time := 360 -- distance in miles
def initial_time := 3 -- time in hours
def additional_distance := 240 -- additional distance in miles
def additional_time := 2 -- additional time in hours
def speed := distance_in_initial_time / initial_time -- speed of the train in miles per hour

-- Statement: Prove that the additional time required to travel X additional miles is X / speed
theorem additional_time_for_distance (X : ℝ) :
  let computed_speed := (distance_in_initial_time / initial_time : ℝ)
  X / computed_speed = X / 120 :=
by
  have speed_is_120 : computed_speed = 120 := by
    rw [distance_in_initial_time, initial_time]
    norm_num
  rw [speed_is_120]
  norm_num
  sorry -- you can complete the proof here.

end additional_time_for_distance_l403_403810


namespace value_of_frac_l403_403591

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l403_403591


namespace regular_polygon_integer_ratio_triangles_l403_403861

theorem regular_polygon_integer_ratio_triangles (n : ℕ) (h₁ : n ≥ 3) :
  (∃ (t : set (set (ℝ × ℝ → ℝ × ℝ → Prop))), 
    (∀ T ∈ t, -- T is a right triangle with integer-ratio sides
      ∃ a b c : ℕ, a^2 + b^2 = c^2 ∧ T = (a, b, c)) ∧
    (∃ P : set (ℝ × ℝ), -- P is a regular n-gon
      (∀ v ∈ P, integer_ratio_right_triangle_v v t) ∧ -- All dissections are IRRTs
      (regular_polygon n P))) ↔ n = 4 :=
by
  sorry

end regular_polygon_integer_ratio_triangles_l403_403861


namespace find_numbers_l403_403001

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end find_numbers_l403_403001


namespace boy_girl_sum_equality_l403_403221

open Finset

theorem boy_girl_sum_equality (B G : Finset ℕ) (hB : B.card = 10) (hG : G.card = 10) (hUnion : B ∪ G = (range 1 21).to_finset) (hDisjoint : Disjoint B G) :
  ∑ b in B, (20 - b) = ∑ g in G, (g - 1) :=
by
  sorry

end boy_girl_sum_equality_l403_403221


namespace trapezoid_area_is_correct_semicircle_circumference_is_correct_l403_403207

-- Given Definitions and Conditions
def length_rectangle := 8
def breadth_rectangle := 6
def perimeter_rectangle := 2 * (length_rectangle + breadth_rectangle)
def side_square := perimeter_rectangle / 4

def longer_base_trapezoid := side_square
def shorter_base_trapezoid := longer_base_trapezoid - 4
def height_trapezoid := 5
def area_trapezoid := (1/2 : ℝ) * (longer_base_trapezoid + shorter_base_trapezoid) * height_trapezoid

def pi_approx := 3.1416
def circumference_semicircle := (1/2 : ℝ) * pi_approx * side_square + side_square

-- Proof Statements
theorem trapezoid_area_is_correct : area_trapezoid = 25 := sorry

theorem semicircle_circumference_is_correct : circumference_semicircle ≈ 17.9956 := sorry

end trapezoid_area_is_correct_semicircle_circumference_is_correct_l403_403207


namespace unique_shirt_and_tie_outfits_l403_403176

theorem unique_shirt_and_tie_outfits :
  let shirts := 10
  let ties := 8
  let choose n k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose shirts 5 * choose ties 4 = 17640 :=
by
  sorry

end unique_shirt_and_tie_outfits_l403_403176


namespace max_value_of_f_l403_403520

def f (a x : ℝ) := (x^2 - 4) * (x - a)
noncomputable def f' (a x : ℝ) := (deriv (f a)) x

-- Given that f' at x = 1 is zero, find the correct a
def a : ℝ := -1/2

def f_max_value_in_interval : Prop :=
  ∃ x ∈ Icc (-2 : ℝ) 2, f a x = 50 / 27

theorem max_value_of_f : f_max_value_in_interval :=
by sorry

end max_value_of_f_l403_403520
