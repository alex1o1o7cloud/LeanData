import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.FactorialRing
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCD
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.ProbabilityTheory
import Mathlib.Algebra.Trig
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Statistics.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Rewrite
import Mathlib.Topology.Sequences
import Real

namespace fraction_identity_l685_685294

theorem fraction_identity (N F : ℝ) (hN : N = 8) (h : 0.5 * N = F * N + 2) : F = 1 / 4 :=
by {
  -- proof will go here
  sorry
}

end fraction_identity_l685_685294


namespace boxes_A_B_cost_condition_boxes_B_profit_condition_l685_685705

/-
Part 1: Prove the number of brand A boxes is 60 and number of brand B boxes is 40 given the cost condition.
-/
theorem boxes_A_B_cost_condition (x : ℕ) (y : ℕ) :
  80 * x + 130 * y = 10000 ∧ x + y = 100 → x = 60 ∧ y = 40 :=
by sorry

/-
Part 2: Prove the number of brand B boxes should be at least 54 given the profit condition.
-/
theorem boxes_B_profit_condition (y : ℕ) :
  40 * (100 - y) + 70 * y ≥ 5600 → y ≥ 54 :=
by sorry

end boxes_A_B_cost_condition_boxes_B_profit_condition_l685_685705


namespace order_f_values_l685_685471

def f : ℝ → ℝ := sorry

-- Conditions:
axiom f_even : ∀ x : ℝ, f(x) = f(-x)
axiom f_periodic : ∀ x : ℝ, f(x) = f(x + 2)
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = x
axiom f_increasing : ∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y ≤ 1) → f(x) < f(y)

-- Prove the ordering
theorem order_f_values : f(0.5) < f(2.5) ∧ f(2.5) < f(3.5) :=
by
    -- Complete the proof based on the provided conditions
    sorry

end order_f_values_l685_685471


namespace mike_unbroken_seashells_l685_685553

-- Define the conditions from the problem
def totalSeashells : ℕ := 6
def brokenSeashells : ℕ := 4
def unbrokenSeashells : ℕ := totalSeashells - brokenSeashells

-- Statement to prove
theorem mike_unbroken_seashells : unbrokenSeashells = 2 := by
  sorry

end mike_unbroken_seashells_l685_685553


namespace solve_cos_arcsin_fraction_equivalence_l685_685006

noncomputable def cos_arcsin_fraction_equivalence : Prop :=
  \(\cos \left( \arcsin \frac{3}{5} \right) = \frac{4}{5}\)

theorem solve_cos_arcsin_fraction_equivalence : cos_arcsin_fraction_equivalence :=
by
  sorry

end solve_cos_arcsin_fraction_equivalence_l685_685006


namespace overall_percent_decrease_l685_685167

theorem overall_percent_decrease (trouser_price_italy : ℝ) (jacket_price_italy : ℝ) 
(trouser_price_uk : ℝ) (trouser_discount_uk : ℝ) (jacket_price_uk : ℝ) 
(jacket_discount_uk : ℝ) (exchange_rate : ℝ) 
(h1 : trouser_price_italy = 200) (h2 : jacket_price_italy = 150) 
(h3 : trouser_price_uk = 150) (h4 : trouser_discount_uk = 0.20) 
(h5 : jacket_price_uk = 120) (h6 : jacket_discount_uk = 0.30) 
(h7 : exchange_rate = 0.85) : 
((trouser_price_italy + jacket_price_italy) - 
 ((trouser_price_uk * (1 - trouser_discount_uk) / exchange_rate) + 
 (jacket_price_uk * (1 - jacket_discount_uk) / exchange_rate))) / 
 (trouser_price_italy + jacket_price_italy) * 100 = 31.43 := 
by 
  sorry

end overall_percent_decrease_l685_685167


namespace root_values_l685_685937

noncomputable def a : ℂ := 1.4656
noncomputable def b : ℂ := -0.2328 + 0.7926 * complex.I
noncomputable def c : ℂ := -0.2328 - 0.7926 * complex.I

theorem root_values :
  (a + b + c = 1) ∧
  (a * b + a * c + b * c = 0) ∧
  (a * b * c = -1) :=
by {
  sorry
}

end root_values_l685_685937


namespace max_area_cyclic_quadrilateral_l685_685200

theorem max_area_cyclic_quadrilateral 
  (a b c d : ℝ) (S : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  ∀ (Q : Type) [quadrilateral Q] (has_sides : Q.has_sides a b c d), 
  (Q.inscribed_in_circle → Q.area = S) → 
  (∀ (R : Type) [quadrilateral R] (has_sidesR : R.has_sides a b c d), R.area ≤ S) := 
sorry

end max_area_cyclic_quadrilateral_l685_685200


namespace total_time_correct_l685_685257

-- Define the lengths of the trains
def length_train_A : ℝ := 110
def length_train_B : ℝ := 150

-- Define the speeds of the trains in km/hr and conversion to m/s
def speed_train_A_kmh : ℝ := 72
def speed_train_B_kmh : ℝ := 90
def speed_train_A : ℝ := speed_train_A_kmh * 1000 / 3600
def speed_train_B : ℝ := speed_train_B_kmh * 1000 / 3600

-- Define the lengths of the bridges
def length_bridge_A : ℝ := 132
def length_bridge_B : ℝ := 200

-- Define the total distances to be traveled by the trains
def total_distance_train_A : ℝ := length_train_A + length_bridge_A
def total_distance_train_B : ℝ := length_train_B + length_bridge_B

-- Define the time it takes for each train to cross their respective bridges
def time_train_A : ℝ := total_distance_train_A / speed_train_A
def time_train_B : ℝ := total_distance_train_B / speed_train_B

-- Define the total time for both trains
def total_time : ℝ := time_train_A + time_train_B

-- Theorem stating the total time is 26.1 seconds
theorem total_time_correct : total_time = 26.1 := by
  sorry

end total_time_correct_l685_685257


namespace foundation_cost_calculation_l685_685926

section FoundationCost

-- Define the constants given in the conditions
def length : ℝ := 100
def width : ℝ := 100
def height : ℝ := 0.5
def density : ℝ := 150  -- in pounds per cubic foot
def cost_per_pound : ℝ := 0.02
def number_of_houses : ℕ := 3

-- Define the problem using these conditions
theorem foundation_cost_calculation :
  let volume := length * width * height in
  let weight := volume * density in
  let cost_one_house := weight * cost_per_pound in
  let total_cost := cost_one_house * (number_of_houses:ℝ) in
  total_cost = 45000 := 
by {
  -- The proof goes here
  sorry
}

end FoundationCost

end foundation_cost_calculation_l685_685926


namespace trajectory_is_ellipse_fixed_point_exists_l685_685414

-- Define the circle A
def circleA (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 15 = 0

-- Define the fixed point B
def B : ℝ × ℝ := (1, 0)

-- Define the ellipse C which is the trajectory of point N
def ellipseC (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the required fixed point R
def R : ℝ × ℝ := (4, 0)

-- The first proof statement about the ellipse C
theorem trajectory_is_ellipse : ∀ (N : ℝ × ℝ), point_on_perpendicular_bisector N → ellipseC N.1 N.2 :=
sorry

-- The second proof statement about the fixed point on the x-axis
theorem fixed_point_exists : 
  ∀ (P Q : ℝ × ℝ) (k : ℝ), intersects_line_and_ellipse P Q k → ∠O R P = ∠O R Q := 
sorry

end trajectory_is_ellipse_fixed_point_exists_l685_685414


namespace max_val_f_in_interval_l685_685118

theorem max_val_f_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 2 ∧ (∀ y : ℝ, 0 < y → y < 2 → f y ≤ f x) ∧ f x = 5 :=
begin
  let f : ℝ → ℝ := λ x, 1 + sqrt (24 * x - 9 * x^2),
  use 4 / 3, -- critical point found in the solution
  split, norm_num, -- 0 < 4/3
  split, norm_num, -- 4/3 < 2
  split,
  { intros y hy1 hy2,
    sorry, -- Proof of f(y) ≤ f(4/3) 
  },
  { -- Proof that f(4/3) = 5
    calc f (4 / 3) = 1 + sqrt (24 * (4 / 3) - 9 * (4 / 3)^2) : rfl
              ... = 1 + sqrt (16) : by norm_num
              ... = 5 : by norm_num
  }
end

end max_val_f_in_interval_l685_685118


namespace min_sum_of_4x6_table_with_distinct_sums_l685_685515

theorem min_sum_of_4x6_table_with_distinct_sums : 
  ∃ (table : Fin 4 → Fin 6 → ℕ), 
  (∀ i j, table i j > 0) ∧ 
  (let row_sums := λ i, (Finset.univ.sum (λ j, table i j)),
       col_sums := λ j, (Finset.univ.sum (λ i, table i j)) in
   ∀ i1 i2, i1 ≠ i2 → row_sums i1 ≠ row_sums i2 ∧
   ∀ j1 j2, j1 ≠ j2 → col_sums j1 ≠ col_sums j2) ∧
  (Finset.univ.sum (λ i, Finset.univ.sum (λ j, table i j)) = 43) :=
sorry

end min_sum_of_4x6_table_with_distinct_sums_l685_685515


namespace inequality_positive_numbers_l685_685561

theorem inequality_positive_numbers (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := 
sorry

end inequality_positive_numbers_l685_685561


namespace range_of_c_l685_685421

-- Define the conditions
variables (c : ℝ)
def positive_c := c > 0
def not_one := c ≠ 1

-- Define statements p and q
def p := ∀ x > 0, (differentiable ℝ (λ x, log c x) ∧ (∀ x > 0, deriv (λ x, log c x) x > 0))
def q := ∀ x < 1/2, deriv (λ x, x^2 - 2*c*x + 1) x < 0

-- Define the goal theorem
theorem range_of_c (hc : positive_c c) (hn : not_one c) : 
  (¬ (p c ∧ q c) ∧ (p c ∨ q c)) ↔ (1/2 ≤ c ∧ c < 1) := sorry

end range_of_c_l685_685421


namespace final_result_is_110_l685_685730

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtracted_value : ℕ := 142

def final_result : ℕ := (chosen_number * multiplier) - subtracted_value

theorem final_result_is_110 : final_result = 110 := by
  sorry

end final_result_is_110_l685_685730


namespace range_of_f_l685_685019

def and_op (a b : ℝ) : ℝ :=
if a <= b then a else b

def f (x : ℝ) : ℝ :=
and_op (Real.sin x) (Real.cos x)

theorem range_of_f :
  set.range f = set.Icc (-1) (Real.sqrt 2 / 2) :=
sorry

end range_of_f_l685_685019


namespace coins_ratio_l685_685761

-- Conditions
def initial_coins : Nat := 125
def gift_coins : Nat := 35
def sold_coins : Nat := 80

-- Total coins after receiving the gift
def total_coins := initial_coins + gift_coins

-- Statement to prove the ratio simplifies to 1:2
theorem coins_ratio : (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end coins_ratio_l685_685761


namespace find_d_square_plus_5d_l685_685345

theorem find_d_square_plus_5d (a b c d : ℤ) (h₁: a^2 + 2 * a = 65) (h₂: b^2 + 3 * b = 125) (h₃: c^2 + 4 * c = 205) (h₄: d = 5 + 6) :
  d^2 + 5 * d = 176 :=
by
  rw [h₄]
  sorry

end find_d_square_plus_5d_l685_685345


namespace intersection_A_B_intersection_A_complementB_l685_685184

-- Definitions of the sets A and B
def setA : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x | x < -2 ∨ x > 4 }

-- Proof problem 1: A ∩ B = { x | -5 ≤ x < -2 }
theorem intersection_A_B:
  setA ∩ setB = { x : ℝ | -5 ≤ x ∧ x < -2 } :=
sorry

-- Definition of the complement of B
def complB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Proof problem 2: A ∩ (complB) = { x | -2 ≤ x ≤ 3 }
theorem intersection_A_complementB:
  setA ∩ complB = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
sorry

end intersection_A_B_intersection_A_complementB_l685_685184


namespace num_toys_purchased_min_selling_price_l685_685566

variable (x m : ℕ)

-- Given conditions
axiom cond1 : 1500 / x + 5 = 3500 / (2 * x)
axiom cond2 : 150 * m - 5000 >= 1150

-- Required proof
theorem num_toys_purchased : x = 50 :=
by
  sorry

theorem min_selling_price : m >= 41 :=
by
  sorry

end num_toys_purchased_min_selling_price_l685_685566


namespace greatest_fourth_term_l685_685243

theorem greatest_fourth_term (a d : ℕ) (h1 : a > 0) (h2 : d > 0) 
  (h3 : 5 * a + 10 * d = 50) (h4 : a + 2 * d = 10) : 
  a + 3 * d = 14 :=
by {
  -- We introduced the given constraints and now need a proof
  sorry
}

end greatest_fourth_term_l685_685243


namespace lean_proof_problem_l685_685850

theorem lean_proof_problem (a : ℕ) (h₀ : 0 < a) (n : ℕ) (h₁ : 0 < n) :
  ∃ (a_n : ℕ), 
    let r := (Real.sqrt (a + 1) + Real.sqrt a) 
    in (r^(2 * n) + r^(-2 * n) = 4 * a_n + 2) ∧ (r^n = Real.sqrt (a_n + 1) + Real.sqrt a_n) :=
sorry

end lean_proof_problem_l685_685850


namespace brian_read_75_chapters_l685_685328

def book_chapters : Type :=
  {chapters : ℕ // chapters > 0}

def books_read : list book_chapters :=
 [
  ⟨20, by decide⟩,
  ⟨15, by decide⟩,
  ⟨15, by decide⟩,
  let sum_prev_books := 20 + 15 + 15 in
  ⟨sum_prev_books / 2, by decide⟩
]

theorem brian_read_75_chapters (books : list book_chapters) : 
  books.sum (λ b, b.chapters) = 75 :=
by 
  have h : books = 
    [ 
      ⟨20, by decide⟩, 
      ⟨15, by decide⟩, 
      ⟨15, by decide⟩, 
      let sum_prev_books := 20 + 15 + 15 in 
      ⟨sum_prev_books / 2, by decide⟩ 
    ] := 
    by reflexivity
  rw h
  unfold list.sum
  simp
  sorry

end brian_read_75_chapters_l685_685328


namespace ratio_area_PUTS_QUR_l685_685523

-- Define the triangle PQR with given side lengths
structure Triangle :=
(P Q R : Point)
(PQ PR QR : ℝ)
(PQ_length : dist P Q = 12)
(PR_length : dist P R = 16)
(QR_length : dist Q R = 20)

-- Define the midpoints S and T and the intersection point U
structure MidpointsAndCentroid :=
(S T U : Point)
(S_midpoint : midpoint P Q S)
(T_midpoint : midpoint P R T)
(U_centroid : centroid P Q R U)

-- Define the Lean theorem to prove the ratio of areas
theorem ratio_area_PUTS_QUR (Δ : Triangle) (M : MidpointsAndCentroid)
    (h_PQ_length : Δ.PQ_length) (h_PR_length : Δ.PR_length) (h_QR_length : Δ.QR_length)
    (h_S_midpoint : M.S_midpoint) (h_T_midpoint : M.T_midpoint) (h_U_centroid : M.U_centroid) :
    area (quadrilateral Δ.P Δ.U Δ.T Δ.S) = area (triangle Δ.Q Δ.U Δ.R) :=
by
    sorry

end ratio_area_PUTS_QUR_l685_685523


namespace scientific_notation_of_384_000_000_l685_685495

theorem scientific_notation_of_384_000_000 :
  384000000 = 3.84 * 10^8 :=
sorry

end scientific_notation_of_384_000_000_l685_685495


namespace convex_polygon_diagonals_perimeter_bound_l685_685067

theorem convex_polygon_diagonals_perimeter_bound
  (n : ℕ) (d p : ℝ)
  (h1 : n ≥ 3)
  (h2 : 0 < p)
  (h3 : 0 < d)
  (convex_n_gon : is_convex_n_gon n)
  (sum_diagonals_eq_d : sum_of_diagonals convex_n_gon = d)
  (perimeter_eq_p : perimeter convex_n_gon = p)
  : (n : ℝ) - 3 < (2 * d) / p ∧ (2 * d) / p < (⌊(n : ℝ) / 2⌋) * (⌊(n : ℝ + 1) / 2⌋) - 2 := 
sorry

end convex_polygon_diagonals_perimeter_bound_l685_685067


namespace total_chapters_brian_read_l685_685330

theorem total_chapters_brian_read :
  let chapters_first_two_books := 2 * 15 in
  let chapters_third_book := 20 in
  let total_first_three_books := chapters_first_two_books + chapters_third_book in
  let chapters_last_book := total_first_three_books / 2 in
  let total_chapters := chapters_first_two_books + chapters_third_book + chapters_last_book in
  total_chapters = 75 := by
  sorry

end total_chapters_brian_read_l685_685330


namespace sequence_relation_l685_685073

theorem sequence_relation :
  ∀ (n : ℕ), (a : ℕ → ℝ) (b : ℕ → ℝ),
    a 1 = 1 →
    (∀ n, a (n + 1) = a n / (3 * a n + 1)) →
    (∀ n, b n = 1 / a n) →
    b n = 3 * n - 2 :=
by
  intros n a b h1 h2 h3
  sorry

end sequence_relation_l685_685073


namespace top_quality_soccer_balls_l685_685703

theorem top_quality_soccer_balls (N : ℕ) (f : ℝ) (hN : N = 10000) (hf : f = 0.975) : N * f = 9750 := by
  sorry

end top_quality_soccer_balls_l685_685703


namespace find_x_value_l685_685876

theorem find_x_value
    (x : ℝ)
    (a : ℝ × ℝ := (-1, 2))
    (b : ℝ × ℝ := (1, x))
    (h : dot_product a (a + 2*b) = 0) :
    x = -3/4 :=
begin
    sorry
end

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
    v.1 * w.1 + v.2 * w.2

end find_x_value_l685_685876


namespace isosceles_triangle_area_l685_685015

theorem isosceles_triangle_area (p x : ℝ) 
  (h1 : 2 * p = 6 * x) 
  (h2 : 0 < p) 
  (h3 : 0 < x) :
  (1 / 2) * (2 * x) * (Real.sqrt (8 * p^2 / 9)) = (Real.sqrt 8 * p^2) / 3 :=
by
  sorry

end isosceles_triangle_area_l685_685015


namespace probability_of_selecting_one_defective_l685_685085

theorem probability_of_selecting_one_defective :
  (choose 2 6) = 15 → (choose 1 4) * (choose 1 2) = 8 →
  2/15 = 8/15 := 
by 
  intros h1 h2
  sorry

end probability_of_selecting_one_defective_l685_685085


namespace number_of_customers_l685_685531

theorem number_of_customers 
  (total_cartons : ℕ) 
  (damaged_cartons : ℕ) 
  (accepted_cartons : ℕ) 
  (customers : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : damaged_cartons = 60)
  (h3 : accepted_cartons = 160)
  (h_eq_per_customer : (total_cartons / customers) - damaged_cartons = accepted_cartons / customers) :
  customers = 4 :=
sorry

end number_of_customers_l685_685531


namespace find_greatest_m_l685_685347

def greatest_m : ℝ := 3 + 2 * Real.sqrt 2

theorem find_greatest_m (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hu_pos : 0 < u) (hxy : x ≥ y) :
  ∃ (m : ℝ), m = greatest_m ∧ m ≤ (x : ℝ) / (y : ℝ) :=
sorry

end find_greatest_m_l685_685347


namespace min_AB_CD_is_2_l685_685519

variable (AB CD BD AC CI DX : ℝ)
variable (triangle_inequality : ∀ {a b c : ℝ}, a + b > c → a + c > b → b + c > a)
variable (x_on_AB : ∃ X : ℝ, X > 0 → X < AB)
variable (BD_bisects_CX : ∃ M : ℝ, M = 1/2 * (CI + CX))
variable (AC_bisects_DX : ∃ N : ℝ, N = 1/2 * (DI + DX))

theorem min_AB_CD_is_2 : (forall (AB CD BD AC CI DX) (triangle_inequality: ∀ {a b c : ℝ}, a + b > c → a + c > b → b + c > a) (x_on_AB : ∃ X : ℝ, X > 0 → X < AB) (BD_bisects_CX: ∃ M : ℝ, M = 1/2 * (CI + CX)) (AC_bisects_DX : ∃ N : ℝ, N = 1/2 * (DI + DX)) , (AB / CD) >= 2 := sorry

end min_AB_CD_is_2_l685_685519


namespace distribute_tickets_l685_685354

-- Define the conditions of the problem
variable (T : Fin 5) -- 5 tickets labeled from 0 to 4
variable (P : Fin 4) -- 4 people labeled from 0 to 3
-- Each person must receive at least one ticket and if they get more than one, the tickets must be consecutive.

-- Define the problem statement as a theorem
theorem distribute_tickets :
  -- define the condition each person gets at least one ticket and tickets for each person are consecutive
  let condition := ∀ (f : T → P), 
    (∀ p : P, ∃ (t1 t2 t3 t4 t5: Fin 5), 
      (t1 < 4) ∧ (t2 < 4) ∧ (t3 < 4) ∧ (t4 < 4) ∧ (t5 < 4)) -- simplified condition placeholder
  -- define the goal to prove
  (∃ (f : T → P), condition f) → 
  (number of ways to distribute tickets satisfying the condition) = 96 := 
sorry

end distribute_tickets_l685_685354


namespace small_mold_radius_l685_685715

-- Define the parameters of the problem
def large_bowl_radius : ℝ := 2
def num_small_molds : ℝ := 64

-- Define the volume of a hemisphere formula
def hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

-- Given conditions
def large_bowl_volume : ℝ := hemisphere_volume large_bowl_radius
def total_small_molds_volume (r : ℝ) : ℝ := num_small_molds * hemisphere_volume r

-- The statement to prove
theorem small_mold_radius :
  ∃ r : ℝ, total_small_molds_volume r = large_bowl_volume ∧ r = 1 / 2 := by
  sorry

end small_mold_radius_l685_685715


namespace solution_set_l685_685811

variable (f : ℝ → ℝ)

-- Odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
axiom h_odd : is_odd_function f
axiom h_slope : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → (f x2 - f x1) / (x2 - x1) < 3
axiom h_f3 : f 3 = 9

-- Define the problem
theorem solution_set (x : ℝ) : f x - 3 * x < 0 ↔ x ∈ Ioo (-3) 0 ∪ Ioo 3 ∞ := by
  sorry

end solution_set_l685_685811


namespace probability_of_picking_red_ball_l685_685249

theorem probability_of_picking_red_ball (w r : ℕ) 
  (h1 : r > w) 
  (h2 : r < 2 * w) 
  (h3 : 2 * w + 3 * r = 60) : 
  r / (w + r) = 7 / 11 :=
sorry

end probability_of_picking_red_ball_l685_685249


namespace cube_ends_in_432_l685_685046

theorem cube_ends_in_432 :
  ∃ n : ℕ, (n^3 % 1000 = 432) ∧ (∀ m : ℕ, (m^3 % 1000 = 432 → m ≥ n)) :=
by
  use 138
  split
  all_goals { sorry }

end cube_ends_in_432_l685_685046


namespace stadium_length_in_feet_l685_685610

-- Assume the length of the stadium is 80 yards
def stadium_length_yards := 80

-- Assume the conversion factor is 3 feet per yard
def conversion_factor := 3

-- The length in feet is the product of the length in yards and the conversion factor
def length_in_feet := stadium_length_yards * conversion_factor

-- We want to prove that this length in feet is 240 feet
theorem stadium_length_in_feet : length_in_feet = 240 := by
  -- Definitions and conditions are directly restated here; the proof is sketched as 'sorry'
  sorry

end stadium_length_in_feet_l685_685610


namespace part1_part2_distribution_part2_expectation_l685_685856

noncomputable def num_defective : ℕ := 2
noncomputable def num_non_defective : ℕ := 3
noncomputable def total_products : ℕ := num_defective + num_non_defective
noncomputable def cost_per_test : ℕ := 100

def prob_first_defective_second_nondefective : ℚ :=
  (num_defective * num_non_defective : ℚ) / (total_products * (total_products - 1))

def prob_cost_200 : ℚ := (1 * 2 : ℚ) / (total_products * (total_products - 1))
def prob_cost_300 : ℚ :=
  (1 * 3 + 2 * 1 * 3 * 1 : ℚ) / (total_products * (total_products - 1) * (total_products - 2))
def prob_cost_400 : ℚ := 1 - prob_cost_200 - prob_cost_300

def expected_cost : ℚ := 
  200 * prob_cost_200 + 300 * prob_cost_300 + 400 * prob_cost_400

theorem part1 :
  prob_first_defective_second_nondefective = 3 / 10 :=
sorry

theorem part2_distribution :
  prob_cost_200 = 1 / 10 ∧ prob_cost_300 = 3 / 10 ∧ prob_cost_400 = 3 / 5 :=
sorry

theorem part2_expectation :
  expected_cost = 350 :=
sorry

end part1_part2_distribution_part2_expectation_l685_685856


namespace student_distribution_l685_685355

open Finset Nat

theorem student_distribution :
  let n := 7
  count_combine (choose n 2 + choose n 3) 2 = 112 :=
by
  sorry

end student_distribution_l685_685355


namespace rate_of_walking_l685_685322

theorem rate_of_walking (v : ℝ) : 
  (160 = (v + 8) * 16) → v = 2 :=
by {
  intros h,
  -- Proof steps would go here
  sorry
}

end rate_of_walking_l685_685322


namespace count_birches_in_forest_l685_685136

theorem count_birches_in_forest:
  ∀ (t p_s p_p : ℕ), t = 4000 → p_s = 10 → p_p = 13 →
  let n_s := (p_s * t) / 100 in
  let n_p := (p_p * t) / 100 in
  let n_o := n_s + n_p in 
  let n_b := t - (n_s + n_p + n_o) in 
  n_b = 2160 :=
by 
  intros t p_s p_p ht hps hpp
  let n_s := (p_s * t) / 100 
  let n_p := (p_p * t) / 100 
  let n_o := n_s + n_p 
  let n_b := t - (n_s + n_p + n_o) 
  exact sorry

end count_birches_in_forest_l685_685136


namespace jerry_weighted_mean_l685_685165

noncomputable def weighted_mean (aunt uncle sister cousin friend1 friend2 friend3 friend4 friend5 : ℝ)
    (eur_to_usd gbp_to_usd cad_to_usd : ℝ) (family_weight friends_weight : ℝ) : ℝ :=
  let uncle_usd := uncle * eur_to_usd
  let friend3_usd := friend3 * eur_to_usd
  let friend4_usd := friend4 * gbp_to_usd
  let cousin_usd := cousin * cad_to_usd
  let family_sum := aunt + uncle_usd + sister + cousin_usd
  let friends_sum := friend1 + friend2 + friend3_usd + friend4_usd + friend5
  family_sum * family_weight + friends_sum * friends_weight

theorem jerry_weighted_mean : 
  weighted_mean 9.73 9.43 7.25 20.37 22.16 23.51 18.72 15.53 22.84 
               1.20 1.38 0.82 0.40 0.60 = 85.4442 := 
sorry

end jerry_weighted_mean_l685_685165


namespace problem_statement_l685_685447

noncomputable def f (ϕ x : ℝ) := sin (2 * x + ϕ)

theorem problem_statement (ϕ : ℝ) (k: ℤ) (c: ℝ)
    (hϕ: -π < ϕ ∧ ϕ < 0)
    (h_symmetry: ∃ x : ℝ, x = π / 8 ∧ (f ϕ x = 1 ∨ f ϕ x = -1)) :
  (ϕ = - (3 * π / 4)) ∧
  (∀ x, kπ + π/8 ≤ x ∧ x ≤ kπ + 5 * π/8 → strict_mono_incr_on (λ x, f ϕ x)) ∧
  ¬ ∃ x, f ϕ x = (5 * x + c) / 2 := sorry

end problem_statement_l685_685447


namespace Z1_mul_Z2_real_iff_l685_685933

-- Definitions based on conditions from the problem statement
variables {a b c d : ℝ}
def Z1 := a + b * complex.i
def Z2 := c + d * complex.i

-- Lean statement for the proof problem
theorem Z1_mul_Z2_real_iff :
  (Z1 * Z2).im = 0 ↔ (a * d + b * c = 0) :=
by
sorrysteps.

end Z1_mul_Z2_real_iff_l685_685933


namespace solve_for_m_l685_685897

theorem solve_for_m :
  (m : ℝ) → (h : tan (π / 12) * cos (5 * π / 12) = sin (5 * π / 12) - m * sin (π / 12)) → 
  m = 2 * sqrt 3 := 
sorry

end solve_for_m_l685_685897


namespace min_value_alpha_beta_gamma_l685_685831

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma_l685_685831


namespace intersection_of_sets_l685_685844

def A := {1, 6, 8, 10}
def B := {2, 4, 8, 10}

theorem intersection_of_sets : A ∩ B = {8, 10} :=
by
  sorry

end intersection_of_sets_l685_685844


namespace evaluate_polynomial_at_2_l685_685672

noncomputable def polynomial_value (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 8

theorem evaluate_polynomial_at_2 : polynomial_value 2 = 6 :=
by 
  calc
    polynomial_value 2 = 2^3 - 4 * 2^2 + 3 * 2 + 8 : rfl
                     ... = 8 - 16 + 6 + 8 : by norm_num
                     ... = 6 : by ring

end evaluate_polynomial_at_2_l685_685672


namespace min_safe_combinations_l685_685994

theorem min_safe_combinations (n m : ℕ) (correct_combinations : Finset (ℕ × ℕ × ℕ)) :
  (∀ x y z, correct_combinations ∈ (Finset.range 8) × (Finset.range 8) × (Finset.range 8)) ∧
  (∀ (a b c a' b' c' : ℕ), ((a = a' ∧ b = b') ∨ (a = a' ∧ c = c') ∨ (b = b' ∧ c = c')) → 
    (a, b, c) ∈ correct_combinations → (a', b', c') ∈ correct_combinations) →
  correct_combinations.card = 32 :=
by
  sorry

end min_safe_combinations_l685_685994


namespace find_parabola_point_l685_685307

theorem find_parabola_point :
  ∃ (x y : ℝ), 
  (0 < x ∧ 0 < y) ∧
  (√(x^2 + (y - 2)^2) = 150) ∧
  y = 148 ∧
  x = 2 * √296 :=
by 
  sorry

end find_parabola_point_l685_685307


namespace power_function_nature_l685_685231

def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_nature:
  (f 3 = Real.sqrt 3) ∧
  (¬ (∀ x, f (-x) = f x)) ∧
  (¬ (∀ x, f (-x) = -f x)) ∧
  (∀ x, 0 < x → 0 < f x) := 
by
  sorry

end power_function_nature_l685_685231


namespace speed_of_stream_l685_685700

theorem speed_of_stream (b s : ℝ) 
  (H1 : b + s = 10)
  (H2 : b - s = 4) : 
  s = 3 :=
sorry

end speed_of_stream_l685_685700


namespace simpler_transformation_l685_685636

theorem simpler_transformation :
  ∀ (x y : ℝ), (2 * x - y = 7) ↔ (y = 2 * x - 7) :=
by
  intros x y
  split
  { -- direction 1: if 2 * x - y = 7, then y = 2 * x - 7
    intro h
    rw [←h],
    linarith
  }
  { -- direction 2: if y = 2 * x - 7, then 2 * x - y = 7
    intro h
    rw h,
    linarith
  }

end simpler_transformation_l685_685636


namespace smallest_n_exists_l685_685774

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end smallest_n_exists_l685_685774


namespace sum_of_two_digit_numbers_with_squares_ending_in_04_l685_685264

theorem sum_of_two_digit_numbers_with_squares_ending_in_04 :
  ∑ n in Finset.filter (λ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (n^2 % 100 = 4)) (Finset.range 100) = 326 := by
  sorry

end sum_of_two_digit_numbers_with_squares_ending_in_04_l685_685264


namespace average_TV_sets_in_shops_l685_685285

def shop_a := 20
def shop_b := 30
def shop_c := 60
def shop_d := 80
def shop_e := 50
def total_shops := 5

theorem average_TV_sets_in_shops : (shop_a + shop_b + shop_c + shop_d + shop_e) / total_shops = 48 :=
by
  have h1 : shop_a + shop_b + shop_c + shop_d + shop_e = 240
  { sorry }
  have h2 : 240 / total_shops = 48
  { sorry }
  exact Eq.trans (congrArg (fun x => x / total_shops) h1) h2

end average_TV_sets_in_shops_l685_685285


namespace age_of_15th_student_l685_685682

theorem age_of_15th_student (avg_age_15 avg_age_3 avg_age_11 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_3 : avg_age_3 = 14) 
  (h_avg_11 : avg_age_11 = 16) : 
  ∃ x : ℕ, x = 7 := 
by
  sorry

end age_of_15th_student_l685_685682


namespace complete_the_square_l685_685647

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l685_685647


namespace determine_a_b_l685_685438

theorem determine_a_b (a b : ℝ) :
  (∀ x, y = x^2 + a * x + b) ∧ (∀ t, t = 0 → 3 * t - (t^2 + a * t + b) + 1 = 0) →
  a = 3 ∧ b = 1 :=
by
  sorry

end determine_a_b_l685_685438


namespace quadrilateral_max_angle_l685_685760

-- assumptions
variable (square : Type)
variable (diagonal : square → Prop)
variable (folded_sides : square → Prop → Prop)

-- Given a square paper folded along its diagonal:

theorem quadrilateral_max_angle (s : square) (h1 : diagonal s) (h2 : folded_sides s h1) :
  ∃ α : ℝ, α = 112.5 ∧ (∀ β : ℝ, β = α → β ≤ 112.5) := sorry

end quadrilateral_max_angle_l685_685760


namespace negation_proof_l685_685228

theorem negation_proof (a b : ℝ) (h : a^2 + b^2 = 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end negation_proof_l685_685228


namespace compound_weight_distribution_l685_685768

theorem compound_weight_distribution :
  let total_weight := 500
  let ratio_A := 2
  let ratio_B := 10
  let ratio_C := 5
  let total_parts := ratio_A + ratio_B + ratio_C
  let weight_per_part := (total_weight : ℝ) / total_parts
in 
  (ratio_A * weight_per_part = 58.82) ∧ 
  (ratio_B * weight_per_part = 294.12) ∧ 
  (ratio_C * weight_per_part = 147.06) :=
by
  sorry

end compound_weight_distribution_l685_685768


namespace units_digit_of_sequence_sum_l685_685266

theorem units_digit_of_sequence_sum : 
  let s := List.sum [1! + 1, 2! + 2, 3! + 3, 4! + 4, 5! + 5, 6! + 6, 7! + 7, 8! + 8, 9! + 9] in
  (s % 10) = 8 :=
by
  sorry

end units_digit_of_sequence_sum_l685_685266


namespace eraser_difference_l685_685109

theorem eraser_difference
  (hanna_erasers rachel_erasers tanya_erasers tanya_red_erasers : ℕ)
  (h1 : hanna_erasers = 2 * rachel_erasers)
  (h2 : rachel_erasers = tanya_red_erasers)
  (h3 : tanya_erasers = 20)
  (h4 : tanya_red_erasers = tanya_erasers / 2)
  (h5 : hanna_erasers = 4) :
  rachel_erasers - (tanya_red_erasers / 2) = 5 :=
sorry

end eraser_difference_l685_685109


namespace fraction_phone_numbers_9_ending_even_l685_685748

def isValidPhoneNumber (n : Nat) : Bool :=
  n / 10^6 != 0 && n / 10^6 != 1 && n / 10^6 != 2

def isValidEndEven (n : Nat) : Bool :=
  let lastDigit := n % 10
  lastDigit == 0 || lastDigit == 2 || lastDigit == 4 || lastDigit == 6 || lastDigit == 8

def countValidPhoneNumbers : Nat :=
  7 * 10^6

def countValidStarting9EndingEven : Nat :=
  5 * 10^5

theorem fraction_phone_numbers_9_ending_even :
  (countValidStarting9EndingEven : ℚ) / (countValidPhoneNumbers : ℚ) = 1 / 14 :=
by 
  sorry

end fraction_phone_numbers_9_ending_even_l685_685748


namespace eccentricity_bound_l685_685815

theorem eccentricity_bound (e : ℝ) : 
  (∃ F : ℝ × ℝ, F = (4, 0)) ∧ 
  (∃ l : ℝ → Prop, ∀ x, l x ↔ x = -1) ∧
  (∃ A B : ℝ × ℝ, ∃ P : ℝ × ℝ, P.1^2 + P.2^2 < ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) →
  e < sqrt (2 - sqrt 2) :=
by
  sorry

end eccentricity_bound_l685_685815


namespace luke_initial_money_l685_685551

def initial_amount (X : ℤ) : Prop :=
  let spent := 11
  let received := 21
  let current_amount := 58
  X - spent + received = current_amount

theorem luke_initial_money : ∃ (X : ℤ), initial_amount X ∧ X = 48 :=
by
  sorry

end luke_initial_money_l685_685551


namespace hexagonal_star_ratio_l685_685706

noncomputable def circle_radius : ℝ := 3
def circle_area : ℝ := Real.pi * circle_radius^2
def hexagonal_star_area : ℝ := 40.5 * Real.sqrt 3
def ratio_of_areas : ℝ := hexagonal_star_area / circle_area

theorem hexagonal_star_ratio :
  (circle_radius = 3) →
  (circle_area = Real.pi * circle_radius^2) →
  (hexagonal_star_area = 40.5 * Real.sqrt 3) →
  (ratio_of_areas = 4.5 * Real.sqrt 3 / Real.pi) :=
by
  intros h1 h2 h3
  unfold ratio_of_areas
  rw [h2, h3]
  -- skipping the actual proof steps
  sorry

end hexagonal_star_ratio_l685_685706


namespace red_subsequence_2009th_number_l685_685501

def is_red (n : ℕ) : Prop :=
  -- Define the rule for numbers to be colored red according to the specified sequence
  sorry

theorem red_subsequence_2009th_number :
  ∃ n, is_red n ∧ n = 3953 ∧ 
    carddinal {x : ℕ | is_red x ∧ x ≤ 3953} = 2009 := sorry

end red_subsequence_2009th_number_l685_685501


namespace max_mixed_gender_groups_l685_685616

theorem max_mixed_gender_groups (b g : ℕ) (h_b : b = 31) (h_g : g = 32) : 
  ∃ max_groups, max_groups = min (b / 2) (g / 3) :=
by
  use 10
  sorry

end max_mixed_gender_groups_l685_685616


namespace original_proposition_false_implies_negation_true_l685_685102

-- Define the original proposition and its negation
def original_proposition (x y : ℝ) : Prop := (x + y > 0) → (x > 0 ∧ y > 0)
def negation (x y : ℝ) : Prop := ¬ original_proposition x y

-- Theorem statement
theorem original_proposition_false_implies_negation_true (x y : ℝ) : ¬ original_proposition x y → negation x y :=
by
  -- Since ¬ original_proposition x y implies the negation is true
  intro h
  exact h

end original_proposition_false_implies_negation_true_l685_685102


namespace original_cards_proof_l685_685568

-- Definitions corresponding to the conditions
def sasha_added := 48
def karen_fraction := 1 / 6
def total_after_karen := 83
def karen_cards (added : ℕ) (fraction : ℝ) : ℕ :=
  (added * fraction).to_nat
def original_cards (total : ℕ) (removed : ℕ) : ℕ :=
  total + removed - sasha_added

-- Prove that the original number of cards is 75
theorem original_cards_proof : original_cards total_after_karen (karen_cards sasha_added karen_fraction) = 75 := by
  -- Proof can be filled later
  sorry

end original_cards_proof_l685_685568


namespace completing_the_square_solution_correct_l685_685638

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l685_685638


namespace lcm_b_c_l685_685849

noncomputable def lcm (m n : ℕ) : ℕ := 
  (m * n) / Nat.gcd m n

theorem lcm_b_c {a b c : ℕ} (h1 : lcm a b = 60) (h2 : lcm a c = 270) : lcm b c = 540 :=
  sorry

end lcm_b_c_l685_685849


namespace mean_evaluation_l685_685031

variable (x a b : ℝ)

def arithmeticMean (x a b : ℝ) : ℝ :=
  ( ((x + a + b) / (x + b)) + ((x - a - b) / (x - b)) ) / 2

theorem mean_evaluation :
  x ≠ b → x ≠ -b → arithmeticMean x a b = 1 - ab / (x^2 - b^2) := 
by
  intro h1 h2
  sorry

end mean_evaluation_l685_685031


namespace meeting_streetlight_l685_685321

-- Let's define the parameters and conditions from the given problem.
def total_streetlights : ℕ := 400
def alla_start : ℕ := 1
def boris_start : ℕ := 400
def alla_at_55th : ℕ := 55
def boris_at_321st : ℕ := 321

-- Define the theorem to prove the meeting point is the 163rd streetlight.
theorem meeting_streetlight : 
  let alla_intervals := alla_at_55th - alla_start in
  let boris_intervals := boris_start - boris_at_321st in
  let total_intervals := total_streetlights - 1 in
  let relative_intervals := alla_intervals + boris_intervals in
  let scale_factor := (total_intervals / relative_intervals) in
  let alla_meeting_position := alla_start + scale_factor * alla_intervals in
  alla_meeting_position = 163 := 
by
  sorry

end meeting_streetlight_l685_685321


namespace hyperbola_line_intersection_l685_685101

theorem hyperbola_line_intersection (a b : ℝ) (h_b : b ≠ 0) :
  let e := Real.sqrt (1 + (b ^ 2) / (a ^ 2)) in
  (∃ x y : ℝ, y = 2 * x ∧ x^2 / a^2 - y^2 / b^2 = 1) ↔ (1 < e ∧ e ≤ Real.sqrt 5) :=
by
  let e := Real.sqrt (1 + (b ^ 2) / (a ^ 2))
  sorry

end hyperbola_line_intersection_l685_685101


namespace max_log_sum_is_two_l685_685977

noncomputable def max_log_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = 40) : ℝ :=
  real.log10 x + real.log10 y

theorem max_log_sum_is_two : 
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), (4 * x + y = 40) ∧ (max_log_sum x y hx hy = 2) :=
begin
  sorry
end

end max_log_sum_is_two_l685_685977


namespace log_calculation_l685_685332

variable (log5_25 : ℝ)
variable (log2_64 : ℝ)
variable (ln1 : ℝ)

axiom log5_25_ax : log5_25 = 2
axiom log2_64_ax : log2_64 = 6
axiom ln1_ax : ln1 = 0

theorem log_calculation : 2 * log5_25 + 3 * log2_64 - 81 * ln1 = 22 :=
by
  rw [log5_25_ax, log2_64_ax, ln1_ax]
  sorry

end log_calculation_l685_685332


namespace product_of_m_and_u_l685_685542

noncomputable def g : ℝ → ℝ := sorry

axiom g_conditions : (∀ x y : ℝ, g (x^2 - y^2) = (x - y) * ((g x) ^ 3 + (g y) ^ 3)) ∧ (g 1 = 1)

def m : ℕ := sorry
def u : ℝ := sorry

theorem product_of_m_and_u : m * u = 3 :=
by 
  -- all conditions about 'g' are assumed as axioms and not directly included in the proof steps
  exact sorry

end product_of_m_and_u_l685_685542


namespace robin_piano_highest_before_lowest_l685_685205

def probability_reach_highest_from_middle_C : ℚ :=
  let p_k (k : ℕ) (p_prev : ℚ) (p_next : ℚ) : ℚ := (1/2 : ℚ) * p_prev + (1/2 : ℚ) * p_next
  let p_1 := 0
  let p_88 := 1
  let A := -1/87
  let B := 1/87
  A + B * 40

theorem robin_piano_highest_before_lowest :
  probability_reach_highest_from_middle_C = 13 / 29 :=
by
  sorry

end robin_piano_highest_before_lowest_l685_685205


namespace min_packs_needed_l685_685208

-- Define pack sizes
def pack_sizes : List ℕ := [6, 12, 24, 30]

-- Define the total number of cans needed
def total_cans : ℕ := 150

-- Define the minimum number of packs needed to buy exactly 150 cans of soda
theorem min_packs_needed : ∃ packs : List ℕ, (∀ p ∈ packs, p ∈ pack_sizes) ∧ List.sum packs = total_cans ∧ packs.length = 5 := by
  sorry

end min_packs_needed_l685_685208


namespace max_value_b_le_e_cubed_l685_685848

noncomputable def f (x a b : ℝ) : ℝ := (1 / 2) * x^2 - a * x + b * Real.log x

theorem max_value_b_le_e_cubed (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, (x > 0 → f x a b < 0) → (f x a b has a maximum at some x0) → b ≤ Real.exp 3) :=
sorry

end max_value_b_le_e_cubed_l685_685848


namespace isosceles_trapezoid_base_difference_l685_685505

theorem isosceles_trapezoid_base_difference (A B C D: Point) (AB CD: Line) (AD BC: Segment)
    (isosceles_trapezoid: IsoscelesTrapezoid A B C D AB CD)
    (angle_BAD_60: ∠A B AD = 60) :
  length AB = length AD - length BC :=
by
  sorry

end isosceles_trapezoid_base_difference_l685_685505


namespace sphere_volume_correct_l685_685729

noncomputable def sphere_volume (BC : ℝ) (angle_BAC : ℝ) (distance_O_to_plane : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (sqrt ((BC / (2 * Real.sin angle_BAC)) ^ 2 + distance_O_to_plane ^ 2)) ^ 3

theorem sphere_volume_correct : sphere_volume 3 (Real.pi / 6) 4 = (500 * Real.pi) / 3 := sorry

end sphere_volume_correct_l685_685729


namespace smallest_positive_integer_n_l685_685771

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), (n > 0) ∧ 
  (∑ k in finset.range (n + 1), real.logb 3 (1 + 1 / 3^(3^k))) ≥ 1 + real.logb 3 (3000 / 3001) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (∑ k in finset.range (m + 1), real.logb 3 (1 + 1 / 3^(3^k))) < 1 + real.logb 3 (3000 / 3001)) := 
sorry

end smallest_positive_integer_n_l685_685771


namespace find_j_l685_685222

noncomputable def j_value {j k : ℝ} : Prop :=
  ∃ (b d : ℝ), 
  (d = -2/3 * b) ∧ 
  (b ≠ 0) ∧
  (∀ (b d : ℝ), 
    b ≠ 0 → 
    d = -2/3 * b → 
    let roots := [b, -b/3, b/3, -b] in
    (roots.prod id = 256) ∧ 
    (roots.sum id = 0) →
    j = -40)

theorem find_j {j k : ℝ} (h : j_value) : j = -40 := 
by { obtain ⟨b, d, h1, h2, h3⟩ := h, exact h3 b d h2 h1 }

end find_j_l685_685222


namespace sarees_with_6_shirts_l685_685998

-- Define the prices of sarees, shirts and the equation parameters
variables (S T : ℕ) (X : ℕ)

-- Define the conditions as hypotheses
def condition1 := 2 * S + 4 * T = 1600
def condition2 := 12 * T = 2400
def condition3 := X * S + 6 * T = 1600

-- Define the theorem to prove X = 1 under these conditions
theorem sarees_with_6_shirts
  (h1 : condition1 S T)
  (h2 : condition2 T)
  (h3 : condition3 S T X) : 
  X = 1 :=
sorry

end sarees_with_6_shirts_l685_685998


namespace cos_arcsin_l685_685008

theorem cos_arcsin (h₀ : real.sqrt (5^2 - 3^2) = 4) : real.cos (real.arcsin (3 / 5)) = 4 / 5 :=
by 
  sorry

end cos_arcsin_l685_685008


namespace trips_needed_l685_685754

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l685_685754


namespace rooms_equation_l685_685914

theorem rooms_equation (x : ℕ) (h₁ : ∃ n, n = 6 * (x - 1)) (h₂ : ∃ m, m = 5 * x + 4) :
  6 * (x - 1) = 5 * x + 4 :=
sorry

end rooms_equation_l685_685914


namespace numeral_at_position_150_l685_685271

noncomputable def repeating_sequence : list ℕ := [5, 6, 5, 2, 1, 7, 3, 9, 1, 3, 0, 4, 3, 4, 7, 8, 2, 6, 0, 8, 6, 9, 5, 6]

theorem numeral_at_position_150 : 
  (repeating_sequence.nth ((150 % 23) - 1)) = some 3 :=
by
  -- nit: specifying exact position in repeating sequence
  have position := 150 % 23 - 1
  show repeating_sequence.nth position = some 3
  sorry

end numeral_at_position_150_l685_685271


namespace factor_difference_of_squares_l685_685375

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l685_685375


namespace jimmy_hostel_stay_days_l685_685364

-- Definitions based on the conditions
def nightly_hostel_charge : ℕ := 15
def nightly_cabin_charge_per_person : ℕ := 15
def total_lodging_expense : ℕ := 75
def days_in_cabin : ℕ := 2

-- The proof statement
theorem jimmy_hostel_stay_days : 
    ∃ x : ℕ, (nightly_hostel_charge * x + nightly_cabin_charge_per_person * days_in_cabin = total_lodging_expense) ∧ x = 3 := by
    sorry

end jimmy_hostel_stay_days_l685_685364


namespace jill_marathon_time_l685_685160

-- Define the conditions
def distance : ℝ := 43
def jack_time : ℝ := 4.5
def speed_ratio : ℝ := 0.9555555555555555

-- Given that Jack's running speed and the speed ratio, prove that Jill's marathon time is 4.3 hours
theorem jill_marathon_time : 
  let jack_speed := distance / jack_time in
  let jill_speed := jack_speed / speed_ratio in
  (distance / jill_speed) = 4.3 :=
by
  let jack_speed := distance / jack_time
  let jill_speed := jack_speed / speed_ratio
  have h1 : jack_speed = 9.555555555555555, by sorry
  have h2 : jill_speed = 10, by sorry
  show distance / jill_speed = 4.3, by sorry

end jill_marathon_time_l685_685160


namespace constant_term_in_binomial_expansion_l685_685520

theorem constant_term_in_binomial_expansion 
  (a b : ℕ) (n : ℕ)
  (sum_of_coefficients : (1 + 1)^n = 4)
  (A B : ℕ)
  (sum_A_B : A + B = 72) 
  (A_value : A = 4) :
  (b^2 = 9) :=
by sorry

end constant_term_in_binomial_expansion_l685_685520


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l685_685585

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l685_685585


namespace sale_in_first_month_is_5420_l685_685712

-- Definitions of the sales in months 2 to 6
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month4 : ℕ := 6350
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470

-- Definition of the average sale goal
def average_sale_goal : ℕ := 6100

-- Calculating the total needed sales to achieve the average sale goal
def total_required_sales := 6 * average_sale_goal

-- Known sales for months 2 to 6
def known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6

-- Definition of the sale in the first month
def sale_month1 := total_required_sales - known_sales

-- The proof statement that the sale in the first month is 5420
theorem sale_in_first_month_is_5420 : sale_month1 = 5420 := by
  sorry

end sale_in_first_month_is_5420_l685_685712


namespace problem_statement_l685_685812

def manhattan_dist (A B : ℝ × ℝ) : ℝ :=
  |B.1 - A.1| + |B.2 - A.2|

theorem problem_statement (A B C : ℝ × ℝ) :
  (C.1 ≥ A.1 ∧ C.1 ≤ B.1 ∧ C.2 ≥ A.2 ∧ C.2 ≤ B.2 → manhattan_dist A C + manhattan_dist C B = manhattan_dist A B) ∧
  (angle ABC C = π / 2 → ¬(manhattan_dist A C)^2 + (manhattan_dist C B)^2 = (manhattan_dist A B)^2) ∧
  ((¬(manhattan_dist A C + manhattan_dist C B > manhattan_dist A B))) :=
sorry

end problem_statement_l685_685812


namespace min_value_frac_l685_685068

open Real

theorem min_value_frac (a b : ℝ) (h1 : a + b = 1/2) (h2 : a > 0) (h3 : b > 0) :
    (4 / a + 1 / b) = 18 :=
sorry

end min_value_frac_l685_685068


namespace no_triangle_pairs_l685_685781

theorem no_triangle_pairs (S : Set α) (n m : ℕ) (h_card : S.card = n) (h_condition : 4 * m ≤ n^2) :
  ∃ P ⊆ (S × S), P.card = m ∧ ∀ {a b c}, (a ∈ S ∧ b ∈ S ∧ c ∈ S) → (a, b) ∈ P → (b, c) ∈ P → (c, a) ∈ P → false :=
sorry

end no_triangle_pairs_l685_685781


namespace domain_of_inverse_l685_685089

def f (x : ℝ) : ℝ := 3^x + 5

theorem domain_of_inverse :
  (∃ x : ℝ, (f x = y)) ↔ y ∈ Set.Ioi 5 :=
sorry

end domain_of_inverse_l685_685089


namespace num_valid_sequences_length_15_l685_685116
    
    def valid_sequence (n : ℕ) : ℕ :=
    if n = 3 then 1
    else if n = 4 then 1
    else if n = 5 then 1
    else if n = 6 then 2
    else valid_sequence(n-4) + 2 * valid_sequence(n-5) + valid_sequence(n-6)
    
    theorem num_valid_sequences_length_15 : valid_sequence 15 = 21 :=
    sorry
    
end num_valid_sequences_length_15_l685_685116


namespace probability_no_shaded_square_l685_685695

/-
  Given:
  A 2 by 2001 rectangle consists of unit squares.
  The middle unit square of each row is shaded.
  
  Prove:
  The probability that a randomly chosen rectangle from the figure does not include a shaded square is 1001 / 2001.
-/
theorem probability_no_shaded_square (rect_width : ℕ) (rect_height : ℕ) (mid_shaded : ℕ) :
  rect_width = 2001 → rect_height = 2 → mid_shaded = 1001 →
  let n := ((rect_width + 1) * rect_width) / 2 in
  let m := mid_shaded * (rect_width - mid_shaded) in
  let total_no_shaded_rectangles := 1 - (m / n : ℚ) in
  total_no_shaded_rectangles = 1001 / 2001 :=
by
  intros h_width h_height h_mid_shaded
  let n := ((2001 + 1) * 2001) / 2
  let m := 1001 * (2001 - 1001)
  let total_no_shaded_rectangles : ℚ := 1 - (m / n)
  show total_no_shaded_rectangles = 1001 / 2001
  sorry

end probability_no_shaded_square_l685_685695


namespace apples_left_after_transactions_l685_685526

variables (initial_apples : ℕ) (sell_to_jill_percent : ℕ) (sell_to_june_percent : ℕ)
          (donate_to_charity : ℕ) (give_to_teacher : ℕ)

theorem apples_left_after_transactions :
  initial_apples = 150 →
  sell_to_jill_percent = 20 →
  sell_to_june_percent = 30 →
  donate_to_charity = 10 →
  give_to_teacher = 1 →
  let
    apples_after_jill := initial_apples - (sell_to_jill_percent * initial_apples / 100),
    apples_after_june := apples_after_jill - (sell_to_june_percent * apples_after_jill / 100),
    apples_after_donation := apples_after_june - donate_to_charity,
    apples_left := apples_after_donation - give_to_teacher
  in
  apples_left = 73 :=
by
  intros
  sorry

end apples_left_after_transactions_l685_685526


namespace ab_value_l685_685104

-- Define sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b = 0}

-- The proof statement: Given A = B, prove ab = 0.104
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 :=
by
  sorry

end ab_value_l685_685104


namespace simplify_expression_l685_685564

theorem simplify_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  ((a^3 - a^2 * b) / (a^2 * b) - (a^2 * b - b^3) / (a * b - b^2) - (a * b) / (a^2 - b^2)) = 
  (-3 * a) / (a^2 - b^2) := 
by
  sorry

end simplify_expression_l685_685564


namespace cauchy_problem_solution_l685_685969

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = (x^2) / 2 + (x^3) / 6 + (x^4) / 12 + (x^5) / 20 + x + 1

theorem cauchy_problem_solution (y : ℝ → ℝ) (x : ℝ) 
  (h1: ∀ x, (deriv^[2] y) x = 1 + x + x^2 + x^3)
  (h2: y 0 = 1)
  (h3: deriv y 0 = 1) : 
  solution y x := 
by
  -- Proof Steps
  sorry

end cauchy_problem_solution_l685_685969


namespace range_of_a_l685_685452

theorem range_of_a (a : ℝ) :
  (a + 1)^2 > (3 - 2 * a)^2 ↔ (2 / 3) < a ∧ a < 4 :=
sorry

end range_of_a_l685_685452


namespace coefficient_x3_in_binomial_expansion_l685_685434

theorem coefficient_x3_in_binomial_expansion : 
  let f := (2 * x + sqrt x) ^ 5 in 
  (coeff (expand f) 3) = 10 :=
by 
  sorry

end coefficient_x3_in_binomial_expansion_l685_685434


namespace range_f_l685_685246

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x + 1)

theorem range_f : (Set.range f) = Set.univ := by
  sorry

end range_f_l685_685246


namespace banana_production_total_l685_685696

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l685_685696


namespace prime_numbers_solution_l685_685787

theorem prime_numbers_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : Nat.Prime (p + q)) (h2 : Nat.Prime (p^2 + q^2 - q)) : p = 3 ∧ q = 2 :=
by
  sorry

end prime_numbers_solution_l685_685787


namespace factor_difference_of_squares_l685_685372

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l685_685372


namespace equilateral_triangles_count_l685_685990

theorem equilateral_triangles_count :
  let k_values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] in
  let eq1 := λ k, ∀ x, y = k in
  let eq2 := λ k, ∀ x, y = 2 * x + 3 * k in
  let eq3 := λ k, ∀ x, y = -2 * x + 3 * k in
  (∃ triangles_count : ℕ,
    triangles_count = 654) := 
begin
  sorry
end

end equilateral_triangles_count_l685_685990


namespace sum_of_divisors_l685_685179

theorem sum_of_divisors (p : ℕ) (n : ℕ) (h1 : n = 2^(p-1) * (2^p - 1)) (h2 : Nat.prime (2^p - 1)) :
  (∑ d in Nat.divisors n, if d < n then d else 0) = n :=
sorry

end sum_of_divisors_l685_685179


namespace smallest_n_exists_l685_685773

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end smallest_n_exists_l685_685773


namespace steiner_line_l685_685211

-- Definitions for the geometric entities in the problem
variable {ABC : Triangle} -- A triangle ABC
variable {P : Point} -- A point P

-- Definitions for reflections of P over the sides of the triangle
def reflection (P : Point) (line : Line) : Point := sorry -- Placeholder for reflection definition

def PA' := reflection P ABC.BC
def PB' := reflection P ABC.AC
def PC' := reflection P ABC.AB

-- The main theorem statement
theorem steiner_line
  (collinear : PointsAreCollinear PA' PB' PC') :
  OnCircumcircle ABC P ↔ PointsAreCollinear PA' PB' PC' :=
sorry

end steiner_line_l685_685211


namespace arithmetic_sequence_sum_l685_685482

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end arithmetic_sequence_sum_l685_685482


namespace vasya_turn_off_lights_l685_685296

theorem vasya_turn_off_lights (k : ℕ) 
  (initially_off : ∀ i j t : ℕ, i < 100 → j < 100 → t < 100 → bulb_state (i, j, t) = 0) 
  (peter_presses : ∀ i j : ℕ, i < 100 → j < 100 → ∃ n : ℕ, n ≤ 1 ∧ bulb_state (i, j, 0) + 100 * n = k) :
  ∃ S : set (ℕ × ℕ), 
    ∀ (i j : ℕ), (i, j) ∈ S → i < 100 ∧ j < 100 ∧ ∀ t, bulb_state (i, j, t) = 1 ∧ 
    S.card ≤ k / 100 → ∀ (i j t : ℕ), i < 100 → j < 100 → t < 100 → bulb_state (i, j, t) = 0 :=
by
  sorry

end vasya_turn_off_lights_l685_685296


namespace number_of_zeros_of_fx_is_2_l685_685427

theorem number_of_zeros_of_fx_is_2 {a x₁ x₂ : ℝ}
  (h1 : f x = (1 : ℝ)/3 * x^3 - (3 : ℝ)/2 * a * x^2 + 2 * a * x - (2 : ℝ)/3)
  (h2 : ∃ x₁ x₂, x₁ ≠ x₂ ∧ (∀ y, f' y = 0 → y = x₁ ∨ y = x₂))
  (h3 : x₂ = 2 * x₁) :
  number_of_zeros f = 2 := sorry

where f (x : ℝ) := (1 : ℝ)/3 * x^3 - (3 : ℝ)/2 * a * x^2 + 2 * a * x - (2 : ℝ)/3

and f' (x : ℝ) := x^2 - 3 * a * x + 2 * a

end number_of_zeros_of_fx_is_2_l685_685427


namespace main_theorem_l685_685686
open Classical
noncomputable section

def f (q : ℚ) : ℤ :=
if h : q.denom = 1 then q.num - 3
else ⌈q⌉ - 3 + f (1 / (⌈q⌉ - q))

theorem main_theorem (a b : ℚ) (ha : 1 < a) (hb : 1 < b) (h : 1 / a + 1 / b = 1) :
  f a + f b = -2 := sorry

end main_theorem_l685_685686


namespace journey_distance_l685_685486

-- Define the given conditions.
def person_time : ℝ := 10
def first_half_speed : ℝ := 21
def second_half_speed : ℝ := 24

-- Define the total journey distance, and prove that it equals 224 km given the conditions.
theorem journey_distance :
  (∃ D : ℝ, 
    let D1 := D / 2 in
    let D2 := D / 2 in
    let T1 := D1 / first_half_speed in
    let T2 := D2 / second_half_speed in
    T1 + T2 = person_time) →
  ∃ D : ℝ, D = 224 :=
by
  intro h
  cases' h with D hD
  use 224
  sorry

end journey_distance_l685_685486


namespace edward_books_bought_l685_685030

def money_spent : ℕ := 6
def cost_per_book : ℕ := 3

theorem edward_books_bought : money_spent / cost_per_book = 2 :=
by
  sorry

end edward_books_bought_l685_685030


namespace number_of_house_numbers_l685_685362

def two_digit_primes : List ℕ := [ 
  11, 13, 17, 19, 23, 29, 
  31, 37, 41, 43, 47, 53, 59 
]

theorem number_of_house_numbers : 
  ((WXYZ : ℕ) → 
    ∀ (W X Y Z : ℕ), 
    (W ≠ 0) ∧ 
    (X ≠ 0) ∧ 
    (Y ≠ 0) ∧ 
    (Z ≠ 0) ∧ 
    (W * 10 + X ∈ two_digit_primes) ∧ 
    (Y * 10 + Z ∈ two_digit_primes) ∧ 
    ((W * 10 + X) ≠ (Y * 10 + Z))) → 
  card {a : ℕ | exists W X Y Z, 
          (WXYZ == W * 1000 + X * 100 + Y * 10 + Z) ∧ 
          (W * 10 + X ∈ two_digit_primes) ∧ 
          (Y * 10 + Z ∈ two_digit_primes) ∧ 
          ((W * 10 + X) ≠ (Y * 10 + Z))} = 156 :=
sorry

end number_of_house_numbers_l685_685362


namespace students_neither_football_nor_cricket_l685_685557

def total_students : ℕ := 450
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def both_players : ℕ := 100

theorem students_neither_football_nor_cricket : 
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_football_nor_cricket_l685_685557


namespace factor_x_squared_minus_sixtyfour_l685_685376

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l685_685376


namespace additional_men_needed_l685_685463

-- Definitions of the forces exerted
def force_grandpa : ℝ := F
def force_grandma : ℝ := 3 / 4 * force_grandpa
def force_granddaughter : ℝ := (3 / 4) ^ 2 * force_grandpa
def force_zhuchka : ℝ := (3 / 4) ^ 3 * force_grandpa
def force_cat : ℝ := (3 / 4) ^ 4 * force_grandpa

-- Combined force of grandma, granddaughter, Zhuchka, and the cat
def combined_force : ℝ := force_grandma + force_granddaughter + force_zhuchka + force_cat

-- Total force including Grandpa
def total_force : ℝ := force_grandpa + combined_force

-- Required additional force
def additional_force : ℝ := F * (1293 / 256 - 1)

-- Proving the number of additional men required is 4
theorem additional_men_needed : 
  ∃ n : ℕ, n = 4 ∧ (1:ℝ) + n = 1293 / 256 := by
  sorry

end additional_men_needed_l685_685463


namespace trigonometric_identity_l685_685904

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := 
by {
  sorry
}

end trigonometric_identity_l685_685904


namespace max_swaps_to_transform_l685_685832

variable (n : ℕ) (hn : 2 ≤ n)
variable (A B : matrix (fin n) (fin n) ℕ)
variable (hA : ∀ i j, A i j ∈ finset.range (n * n) + 1)
variable (hB : ∀ i j, B i j ∈ finset.range (n * n) + 1)
variable (h_eq : finset.univ.image (λ (i : fin n) × (j : fin n), A i.1 i.2) = finset.univ.image (λ (i : fin n) × (j : fin n), B i.1 i.2))

theorem max_swaps_to_transform (hA hB h_eq) : 
  ∃ m, (∀ A B, (∀ i j, A i j ∈ finset.range (n * n + 1)) → 
               (∀ i j, B i j ∈ finset.range (n * n + 1)) → 
               (finset.univ.image (λ (i : fin n) × (j : fin n), A i.1 i.2) = finset.univ.image (λ (i : fin n) × (j : fin n), B i.1 i.2)) → 
               transformable_by_swaps A B m) ∧ m = n^2 := 
sorry

end max_swaps_to_transform_l685_685832


namespace area_of_triangle_calculation_l685_685260

variable {X Y Z M : Type}
variable (YZ XZ YM: ℝ)
variable (triangle_XYZ : Triangle X Y Z)
variable (point_M : ∃ M : Type, is_on_line_segment Y Z M)
variable (XM_is_altitude : is_altitude X M Y Z)

def area_of_triangle_XYZ : ℝ := 20 * real.sqrt 26

theorem area_of_triangle_calculation (h_XZ : XZ = 15) (h_YM : YM = 9) (h_YZ : YZ = 20) : 
  area_of_triangle_XYZ YZ XZ YM = 20 * real.sqrt 26 :=
by
  sorry

end area_of_triangle_calculation_l685_685260


namespace unique_x2_range_of_a_l685_685091

noncomputable def f (x : ℝ) (k a : ℝ) : ℝ :=
if x >= 0
then k*x + k*(1 - a^2)
else x^2 + (a^2 - 4*a)*x + (3 - a)^2

theorem unique_x2 (k a : ℝ) (x1 : ℝ) (hx1 : x1 ≠ 0) (hx2 : ∃ x2 : ℝ, x2 ≠ 0 ∧ x2 ≠ x1 ∧ f x2 k a = f x1 k a) :
f 0 k a = k*(1 - a^2) →
0 ≤ a ∧ a < 1 →
k = (3 - a)^2 / (1 - a^2) :=
sorry

variable (a : ℝ)

theorem range_of_a :
0 ≤ a ∧ a < 1 ↔ a^2 - 4*a ≤ 0 :=
sorry

end unique_x2_range_of_a_l685_685091


namespace sin_intersections_sum_l685_685099

noncomputable theory

def sin_intersection (x m : ℝ) : ℝ := 4 * Real.sin (2 * x + π / 6) - m

theorem sin_intersections_sum (m x1 x2 x3 : ℝ) (h_domain : x1 ∈ Set.Icc 0 (7 * π / 6) ∧
                                                       x2 ∈ Set.Icc 0 (7 * π / 6) ∧
                                                       x3 ∈ Set.Icc 0 (7 * π / 6) ∧
                                                       x1 < x2 ∧ x2 < x3) 
  (h_eq1 : sin_intersection x1 m = 0)
  (h_eq2 : sin_intersection x2 m = 0)
  (h_eq3 : sin_intersection x3 m = 0)
  (h_symmetry1 : x1 + x2 = π / 3)
  (h_symmetry2 : x2 + x3 = 4 * π / 3) :
  x1 + 2 * x2 + x3 = 5 * π / 3 :=
by
  sorry

end sin_intersections_sum_l685_685099


namespace triangle_inequality_side_len_l685_685412

theorem triangle_inequality_side_len (x : ℝ) : x = 8 → ¬ (2 < x ∧ x < 8) :=
by
  intro h
  rw h
  exact not_and_of_not_right _ (lt_irrefl 8)

#eval triangle_inequality_side_len 8 rfl

end triangle_inequality_side_len_l685_685412


namespace transformed_incircle_tangent_to_circumcircle_l685_685916

variables {α : Type*}
variables {A B C : α} [metric_space α] [normed_group α] [normed_space ℝ α]
variables (circumcenter incircle_center : α)
variables (circumradius inradius : ℝ)

theorem transformed_incircle_tangent_to_circumcircle
  (h_right_angle : ∃ (A B C : α), ∠C = π / 2)
  (homothety_center : C)
  (homothety_ratio : 2)
  (circumcenter_correct : circumcenter = midpoint A B)
  (incircle_center_correct : incircle_center = (inradius, inradius))
  (inradius_correct : inradius = dist incircle_center circumcenter - circumradius) :
  dist circumcenter (2 • incircle_center) = circumradius + 2 * inradius :=
sorry

end transformed_incircle_tangent_to_circumcircle_l685_685916


namespace calculate_perimeter_of_t_shape_l685_685982

def area_of_squares (n : ℕ) (a_total : ℝ) : ℝ := 
  a_total / n

def side_length_of_squares (a : ℝ) : ℝ := 
  real.sqrt a

def perimeter_t_shape (side_len : ℝ) : ℝ :=
  7 * side_len

theorem calculate_perimeter_of_t_shape (a_total : ℝ) (n : ℕ) (side_len : ℝ) : 
  a_total = 160 → n = 6 → 
  side_len = real.sqrt (a_total / n) →
  perimeter_t_shape side_len = 36.12 :=
by
  intros a_total_eq n_eq side_len_eq
  sorry

end calculate_perimeter_of_t_shape_l685_685982


namespace average_of_remaining_numbers_l685_685589

theorem average_of_remaining_numbers
    (T : ℕ)
    (H_avg : T / 12 = 90)
    (H_remove_sum : 82 + 95 = 177) :
    ((T - 177) / 10 = 90.3) := by
  sorry

end average_of_remaining_numbers_l685_685589


namespace diff_implies_continuous_l685_685223

def differentiable_imp_continuous (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀

-- Problem statement: if f is differentiable at x₀, then it is continuous at x₀.
theorem diff_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) : differentiable_imp_continuous f x₀ :=
by
  sorry

end diff_implies_continuous_l685_685223


namespace bound_on_weights_l685_685417

-- Definitions
variable (G : Type) [GraphG : Graph G]
variable (V : G → Type) [FiniteG : Fintype V]
variable (w : V → ℝ)
variable (n : ℕ)
variable (root : V)

-- Conditions
axiom initial_node_weight : w root = 0
axiom graph_structure :
  ∀ u : V, u ≠ root → ∃ v : V, G.edge u v ∧ v ≠ u
axiom weight_condition :
  ∀ u : V, u ≠ root → w u ≤ ∑ v in G.neighbors u | w v / (G.degree u) + 1

-- Proof statement
theorem bound_on_weights (u : V) : w u ≤ (n^2 : ℝ) := sorry

end bound_on_weights_l685_685417


namespace binomial_coefficient_plus_ten_l685_685340

theorem binomial_coefficient_plus_ten : (nat.choose 12 10) + 10 = 76 := by
  sorry

end binomial_coefficient_plus_ten_l685_685340


namespace austin_total_fruit_l685_685349

theorem austin_total_fruit (dallas_apples : ℕ) (dallas_pears : ℕ) (austin_apples_more : ℕ) (austin_pears_less : ℕ) 
  (h1 : dallas_apples = 14) 
  (h2 : dallas_pears = 9)
  (h3 : austin_apples_more = 6)
  (h4 : austin_pears_less = 5) : 
  let austin_apples := dallas_apples + austin_apples_more in
  let austin_pears := dallas_pears - austin_pears_less in
  austin_apples + austin_pears = 24 := 
by
  sorry

end austin_total_fruit_l685_685349


namespace adam_earnings_correct_l685_685319

def total_earnings (lawns_mowed lawns_to_mow : ℕ) (lawn_pay : ℕ)
                   (cars_washed cars_to_wash : ℕ) (car_pay_euros : ℕ) (euro_to_dollar : ℝ)
                   (dogs_walked dogs_to_walk : ℕ) (dog_pay_pesos : ℕ) (peso_to_dollar : ℝ) : ℝ :=
  let lawn_earnings := lawns_mowed * lawn_pay
  let car_earnings := (cars_washed * car_pay_euros : ℝ) * euro_to_dollar
  let dog_earnings := (dogs_walked * dog_pay_pesos : ℝ) * peso_to_dollar
  lawn_earnings + car_earnings + dog_earnings

theorem adam_earnings_correct :
  total_earnings 4 12 9 4 6 10 1.1 3 4 50 0.05 = 87.5 :=
by
  sorry

end adam_earnings_correct_l685_685319


namespace height_from_B_to_BC_eq_intercepts_equal_eq_line_l685_685693

-- For question (1)
theorem height_from_B_to_BC_eq {A B C : Point}
  (hA : A = (0, 5))
  (hB : B = (1, -2))
  (hC : C = (-6, 4)) :
  ∃ (a b c : ℝ), a = 7 ∧ b = -6 ∧ c = 30 ∧ (∀ x y : ℝ, y = (-7 / 6) * x + (59 / 6) → a * x + b * y + c = 0) := 
sorry

-- For question (2)
theorem intercepts_equal_eq_line {a : ℝ}
  (h_eq : ∀ x y : ℝ, (a-1) * x + y - 2 - a = 0) 
  (h_intercept : ∀ x y : ℝ, (x = 0 → y = 2 + a) ∧ (y = 0 → x = (2 + a) / (a - 1))) :
  ((a = -2 ∨ a = 2) → ∃ (b c : ℝ), (b = 1 ∧ c = -4 ∧ (∀ x y : ℝ, x + y + c = 0)) ∨ (b = 3 ∧ c = 0 ∧ (∀ x y : ℝ, b * x - y + c = 0))) := 
sorry

end height_from_B_to_BC_eq_intercepts_equal_eq_line_l685_685693


namespace find_f_neg_2016_l685_685822

variable (a b k : ℝ)
variable (h₀ : a ≠ 0 ∧ b ≠ 0)
variable (h₁ : f (2016) = k)
where
  f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg_2016 (a b k : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : f a b (2016) = k) :
  f a b (-2016) = 2 - k := by
  sorry

end find_f_neg_2016_l685_685822


namespace find_a_intersecting_line_circle_l685_685854

theorem find_a_intersecting_line_circle
  {A B O : Type*}
  [normed_group A] [inner_product_space ℝ A] [normed_group B] [inner_product_space ℝ B]
  (h1 : ∀ x y : ℝ, (x, y) ∈ { (x, y) | x + y = a })
  (h2 : ∀ x y : ℝ, (x, y) ∈ { (x, y) | x^2 + y^2 = 4 })
  (h3 : ∀ A B O : A, |A + B| = |A - B|) : 
  a = 2 ∨ a = -2 :=
sorry

end find_a_intersecting_line_circle_l685_685854


namespace average_of_21_numbers_l685_685590

theorem average_of_21_numbers (n₁ n₂ : ℕ) (a b c : ℕ)
  (h₁ : n₁ = 11 * 48) -- Sum of the first 11 numbers
  (h₂ : n₂ = 11 * 41) -- Sum of the last 11 numbers
  (h₃ : c = 55) -- The 11th number
  : (n₁ + n₂ - c) / 21 = 44 := -- Average of all 21 numbers
by
  sorry

end average_of_21_numbers_l685_685590


namespace cos_eight_arccos_one_fourth_l685_685763

theorem cos_eight_arccos_one_fourth :
  cos (8 * arccos (1 / 4)) = 18593 / 32768 := by
sorry

end cos_eight_arccos_one_fourth_l685_685763


namespace johns_original_number_l685_685927

def switch_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem johns_original_number :
  ∃ x : ℕ, (10 ≤ x ∧ x < 100) ∧ (∃ y : ℕ, y = 5 * x + 13 ∧ 82 ≤ switch_digits y ∧ switch_digits y ≤ 86 ∧ x = 11) :=
by
  sorry

end johns_original_number_l685_685927


namespace min_value_g_squared_2f_min_value_f_squared_2g_l685_685458

variable {x : ℝ}
variable {a b c : ℝ}
variable (h_non_zero : a ≠ 0)

def f (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_2f :
  (∃ (x : ℝ), (g x)^2 + 2 * (f x) = -7) :=
sorry

theorem min_value_f_squared_2g :
  (∃ (x : ℝ), (f x)^2 + 2 * (g x) = 5) :=
sorry

end min_value_g_squared_2f_min_value_f_squared_2g_l685_685458


namespace monotonic_intervals_a_le_0_monotonic_intervals_a_gt_0_distinct_real_roots_gt_two_l685_685449

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (1 / 2) * a * x^2 + (2 * a - 1) * x
noncomputable def h (x a : ℝ) : ℝ := f x - g x a

theorem monotonic_intervals_a_le_0 (a : ℝ) (h1 : a ≤ 0) :
  ∀ x > 0, monotone (λ x, h x a) :=
sorry

theorem monotonic_intervals_a_gt_0 (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Ioi (0 : ℝ), 0 < x ∧ x < 1 / a → monotone (λ x, h x a)) ∧
  (∀ x ∈ Ioi (0 : ℝ), x > 1 / a → antitone (λ x, h x a)) :=
sorry

noncomputable def phi (x : ℝ) : ℝ := f x - a * x

theorem distinct_real_roots_gt_two (x1 x2 a : ℝ) (h1 : x1 ≠ x2)
  (h2 : (f x1 - a * x1 = 0) ∧ (f x2 - a * x2 = 0)) (h3 : x1 * x2 > Real.exp 2) :
  Real.log x1 + Real.log x2 > 2 :=
sorry

end monotonic_intervals_a_le_0_monotonic_intervals_a_gt_0_distinct_real_roots_gt_two_l685_685449


namespace circle_through_point_l685_685041

noncomputable def circle_eq (a b r : ℝ) := (λ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2)

theorem circle_through_point (a r : ℝ) (h1 : r = abs(-a - 1) / real.sqrt 2)
(h2 : (2 - a)^2 + (-1 + 2 * a)^2 = r^2) :
  circle_eq 1 (-2) (real.sqrt 2) = (λ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2) :=
sorry

end circle_through_point_l685_685041


namespace cos_arcsin_proof_l685_685003

noncomputable def cos_arcsin : ℝ :=
  let θ := arcsin (3 / 5) in
  cos θ

theorem cos_arcsin_proof : cos_arcsin = 4 / 5 := 
by
  sorry

end cos_arcsin_proof_l685_685003


namespace isosceles_triangle_height_ratio_l685_685684

theorem isosceles_triangle_height_ratio (b1 h1 b2 h2 : ℝ) 
  (A1 : ℝ := 1/2 * b1 * h1) (A2 : ℝ := 1/2 * b2 * h2)
  (area_ratio : A1 / A2 = 16 / 49)
  (similar : b1 / b2 = h1 / h2) : 
  h1 / h2 = 4 / 7 := 
by {
  sorry
}

end isosceles_triangle_height_ratio_l685_685684


namespace binomial_expansion_coeff_x10_sub_x5_eq_251_l685_685117

open BigOperators Polynomial

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coeff_x10_sub_x5_eq_251 :
  ∀ (a : Fin 11 → ℤ), (fun (x : ℤ) =>
    x^10 - x^5 - (a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
                  a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + 
                  a 5 * (x - 1)^5 + a 6 * (x - 1)^6 + 
                  a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
                  a 9 * (x - 1)^9 + a 10 * (x - 1)^10)) = 0 → 
  a 5 = 251 := 
by 
  sorry

end binomial_expansion_coeff_x10_sub_x5_eq_251_l685_685117


namespace derivative_sin_div_x_l685_685385

theorem derivative_sin_div_x (x : ℝ) (h : x ≠ 0) : 
  deriv (λ x : ℝ, sin x / x) x = (x * cos x - sin x) / x^2 :=
by
  sorry

end derivative_sin_div_x_l685_685385


namespace simplify_cos_sum_l685_685575

theorem simplify_cos_sum : 
  let ω : ℂ := complex.exp (2 * real.pi * complex.I / 17) in
  (ω ^ 17 = 1) → (complex.abs ω = 1) → 
  (cos (2 * real.pi / 17) + cos (6 * real.pi / 17) + cos (8 * real.pi / 17) = (real.sqrt 13 - 1) / 4) :=
by {
  intros ω hω1 hω2,
  sorry
}

end simplify_cos_sum_l685_685575


namespace triangle_properties_l685_685494

def angle_A (a b : ℝ) : ℝ := real.arcsin (a * real.sin (60 * real.pi / 180) / b)
def angle_B : ℝ := 60
def angle_C (A : ℝ) : ℝ := 180 - A - 60
def side_c (a c_A C : ℝ) : ℝ := (a * real.sin (C * real.pi / 180)) / (real.sin (c_A * real.pi / 180))

theorem triangle_properties :
  let a : ℝ := real.sqrt 2,
      b : ℝ := real.sqrt 3,
      A : ℝ := 45,
      C : ℝ := 75,
      c : ℝ := (real.sqrt 2 + real.sqrt 6) / 2
  in angle_A a b = 45 ∧ angle_C (angle_A a b) = 75 ∧ side_c a (angle_A a b) (angle_C (angle_A a b)) = c :=
by
  let a := real.sqrt 2
  let b := real.sqrt 3
  let A := 45
  let C := 75
  let c := (real.sqrt 2 + real.sqrt 6) / 2
  have hA : angle_A a b = A := sorry
  have hC : angle_C (angle_A a b) = C := sorry
  have hc : side_c a (angle_A a b) (angle_C (angle_A a b)) = c := sorry
  exact ⟨hA, hC, hc⟩

end triangle_properties_l685_685494


namespace train_length_l685_685733

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : L = v * 36)
  (h2 : L + 25 = v * 39) :
  L = 300 :=
by
  sorry

end train_length_l685_685733


namespace value_of_f_pi_over_12_l685_685403

noncomputable def f (α : ℝ) : ℝ :=
  (cos (-α) * sin (π + α)) / cos (3 * π + α) +
  (sin (-2 * π - α) * sin (α + π / 2)) / cos (3 * π / 2 - α)

theorem value_of_f_pi_over_12 :
  f (π / 12) = (sqrt 2 + sqrt 6) / 2 :=
sorry

end value_of_f_pi_over_12_l685_685403


namespace min_value_u_l685_685404

theorem min_value_u (x y : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0)
  (h₂ : 2 * x + y = 6) : 
  ∀u, u = 4 * x ^ 2 + 3 * x * y + y ^ 2 - 6 * x - 3 * y -> 
  u ≥ 27 / 2 := sorry

end min_value_u_l685_685404


namespace origin_on_circle_diameter_AB_equations_of_line_and_circle_given_point_P_l685_685868

-- Definitions of the given geometrical objects and conditions
def parabola (x y : ℝ) := y^2 = 2 * x
def line_through (x₀ y₀ k : ℝ) (x y : ℝ) := y = k * (x - x₀)

-- Prove that origin lies on the circle with diameter AB for any line intersecting parabola at A and B passing through (2, 0)
theorem origin_on_circle_diameter_AB :
  ∀ k : ℝ, let x1 := 2, y1 := 2 in -- simplified for vertical line case
  ∀ A B : ℝ × ℝ, 
  A = (x1, y1) → B = (x1, -y1) →
    origin ∈ circle (A, B) :=
begin
  sorry
end

-- If circle passes through P(4, -2), determine equations of line l and circle M
theorem equations_of_line_and_circle_given_point_P :
  ∀ P : ℝ × ℝ, P = (4, -2) →
  ∃ k1 k2 : ℝ,
  ∃ (l₁ l₂ c₁ c₂ : string), 
  l₁ = "y = -2x + 4" ∧ l₂ = "y = x - 2" ∧ 
  c₁ = "(x - 9/4)^2 + (y + 1/2)^2 = 85/16" ∧ 
  c₂ = "(x - 3)^2 + (y - 1)^2 = 10" :=
begin
  sorry
end

end origin_on_circle_diameter_AB_equations_of_line_and_circle_given_point_P_l685_685868


namespace misha_scored_48_in_second_attempt_l685_685956

theorem misha_scored_48_in_second_attempt (P1 P2 P3 : ℕ)
  (h1 : P2 = 2 * P1)
  (h2 : P3 = (3 / 2) * P2)
  (h3 : 24 ≤ P1)
  (h4 : (3 / 2) * 2 * P1 = 72) : P2 = 48 :=
by sorry

end misha_scored_48_in_second_attempt_l685_685956


namespace reflection_across_x_axis_l685_685149

theorem reflection_across_x_axis (x y : ℝ) : 
  (x, -y) = (-2, -4) ↔ (x, y) = (-2, 4) :=
by
  sorry

end reflection_across_x_axis_l685_685149


namespace minimum_real_roots_l685_685541

theorem minimum_real_roots (g : Polynomial ℝ) (roots : Fin 2011 → ℂ)
  (h_degree : g.degree = 2011)
  (h_roots : ∀ i, g.is_root (roots i))
  (h_distinct_absolute_values : (Finset.image (λ i, |roots i|) Finset.univ).card = 1010) :
  ∃ real_roots : Finset ℝ, (∀ x ∈ real_roots, g.is_root x) ∧ real_roots.card ≥ 9 :=
sorry

end minimum_real_roots_l685_685541


namespace coeff_sum_squared_difference_l685_685847

theorem coeff_sum_squared_difference 
  (a : Fin 2018 → ℚ)
  (h : (x - √3) ^ 2017 = ∑ i in Finset.range 2018, a i * x ^ (2017 - i)) :
  (∑ i in Finset.range 1009, a (2 * i)) ^ 2 - (∑ i in Finset.range 1009, a (2 * i + 1)) ^ 2 = -2 ^ 2017 :=
by
  sorry

end coeff_sum_squared_difference_l685_685847


namespace worker_total_amount_l685_685743

-- Definitions of the conditions
def pay_per_day := 20
def deduction_per_idle_day := 3
def total_days := 60
def idle_days := 40
def worked_days := total_days - idle_days
def earnings := worked_days * pay_per_day
def deductions := idle_days * deduction_per_idle_day

-- Statement of the problem
theorem worker_total_amount : earnings - deductions = 280 := by
  sorry

end worker_total_amount_l685_685743


namespace joe_bought_7_juices_l685_685528

noncomputable def cost_oranges := 3 * 4.5
noncomputable def cost_honey := 3 * 5
noncomputable def cost_plants := (4 / 2) * 18  -- since plantes are 2 for $18
noncomputable def total_cost := 68
noncomputable def cost_known_items := cost_oranges + cost_honey + cost_plants
noncomputable def cost_juices := total_cost - cost_known_items

theorem joe_bought_7_juices : (cost_juices / 0.5) = 7 := by
  sorry

end joe_bought_7_juices_l685_685528


namespace find_interest_rates_l685_685662

-- Define the principal amount
def P : ℝ := 800

-- Define the total interest after 4 years
def total_interest : ℝ := 192

-- State the relation for R1, R2, R3, R4 under simple interest conditions
def increasing_interest_pattern (R1 R2 R3 R4 : ℝ) : Prop :=
  R1 < R2 ∧ R2 < R3 ∧ R3 < R4 ∧
  total_interest / P = 0.24 ∧
  24 = R1 + R2 + R3 + R4

theorem find_interest_rates (R1 R2 R3 R4 : ℝ) :
  increasing_interest_pattern R1 R2 R3 R4 → 24 = R1 + R2 + R3 + R4 ∧ R1 < R2 ∧ R2 < R3 ∧ R3 < R4 :=
by 
  intros h
  exact ⟨h.4.2, h.1, h.2, h.3⟩

end find_interest_rates_l685_685662


namespace communication_scenarios_l685_685365

theorem communication_scenarios
  (nA : ℕ) (nB : ℕ) (hA : nA = 10) (hB : nB = 20) : 
  (∃ scenarios : ℕ, scenarios = 2 ^ (nA * nB)) :=
by
  use 2 ^ (10 * 20)
  sorry

end communication_scenarios_l685_685365


namespace find_a_if_equal_parts_complex_l685_685130

def equal_parts_complex (z : ℂ) : Prop :=
  z.re = z.im

theorem find_a_if_equal_parts_complex :
  ∀ (a : ℝ), equal_parts_complex ((1 + a * complex.I) * complex.I) → a = -1 :=
by
  intro a h
  sorry

end find_a_if_equal_parts_complex_l685_685130


namespace vectorFieldB_exists_l685_685337

-- Define vector field A for Case (b)
def vectorFieldA (x y z : ℝ) : ℝ × ℝ × ℝ := 
  ( y * Real.exp(x*x),
    2 * y * z,
    - (2 * x * z * y * Real.exp(x*x) + z * z) )

-- Define divergence operation
noncomputable def div {f : ℝ × ℝ × ℝ → ℝ} (df_dx df_dy df_dz : ℝ × ℝ × ℝ → ℝ) : ℝ × ℝ × ℝ → ℝ := 
  λ p, df_dx p + df_dy p + df_dz p

-- Define the divergence of A
noncomputable def divA (x y z : ℝ) : ℝ :=
  div 
    (λ p, 2 * p.1 * p.2 * Real.exp(p.1 * p.1)) 
    (λ p, 2 * p.3)
    (λ p, 2 * p.1 * p.2 * Real.exp(p.1 * p.1) + 2 * p.3)
    (x, y, z)

-- Define the vector potential B for Case (b)
noncomputable def vectorPotentialB (x y z : ℝ) : ℝ × ℝ × ℝ :=
  ( y * z * z,
    - y * z * Real.exp(x * x),
    0 )

-- The goal is to prove that divA == 0 and find vector potential B
theorem vectorFieldB_exists (x y z : ℝ) : divA x y z = 0 ∧ 
  ∃ (φ : ℝ × ℝ × ℝ → ℝ), vectorFieldA x y z = ((∂ / ∂y) (∂ / ∂z) φ, (∂ / ∂z) (∂ / ∂x) φ,
  (∂ / ∂x) (∂ / ∂y) φ) + vectorPotentialB x y z :=
  sorry

end vectorFieldB_exists_l685_685337


namespace regular_decagon_equilateral_triangles_count_l685_685875

theorem regular_decagon_equilateral_triangles_count :
  let vertices := (fin 10) -- representing vertices set {A_1, ..., A_10} as fin 10
  in 
  let pairs := (vertices.pairwise id) -- pairs of vertices  
  in
  let unique_pairs := (pairs.card.Choose 2) * 2 -- each pair determining 2 triangles
  in
  unique_pairs = 90 := 
sorry

end regular_decagon_equilateral_triangles_count_l685_685875


namespace PropositionC_true_l685_685277

-- Definitions of the propositions
def PropositionA : Prop := ∀ (l1 l2 t : Line), (Intersects t l1) ∧ (Intersects t l2) 
  → CorrespondingAnglesEqual l1 l2 t

def PropositionB : Prop := ∀ (a b : ℝ), (a^2 = b^2) → (a = b)

def PropositionC : Prop := ∀ (l : Line) (P : Point), ¬(OnLine P l) 
  → ∃! (m : Line), (Parallel m l) ∧ (OnLine P m)

def PropositionD : Prop := ∀ (l1 l2 t : Line), (Perpendicular l1 t) ∧ (Perpendicular l2 t) 
  → (Parallel l1 l2)

-- The theorem that Proposition C is true
theorem PropositionC_true : PropositionC := by
  sorry

end PropositionC_true_l685_685277


namespace student_distribution_l685_685356

open Finset Nat

theorem student_distribution :
  let n := 7
  count_combine (choose n 2 + choose n 3) 2 = 112 :=
by
  sorry

end student_distribution_l685_685356


namespace coefficient_of_x3_in_expansion_is_60_l685_685948

theorem coefficient_of_x3_in_expansion_is_60 : 
  let A := (x - 2 / sqrt x) ^ 6;
  (∃ c, c * x ^ 3 = A → c) = 60 :=
by
  sorry

end coefficient_of_x3_in_expansion_is_60_l685_685948


namespace hyperbola_passes_focus_of_parabola_l685_685511

noncomputable def find_p (p : ℝ) : Prop :=
  (p > 0) ∧ (∃ (x y : ℝ), (x^2 / 4 - y^2 = 1) ∧ (y^2 = 2 * p * x)) ∧ (p = 4)

theorem hyperbola_passes_focus_of_parabola : find_p 4 :=
by
  split
  sorry  -- Proof that p > 0
  split
  sorry  -- Proof of specific (x, y) coordinates
  refl  -- Proof that p = 4

end hyperbola_passes_focus_of_parabola_l685_685511


namespace equivalent_proof_to_original_problem_l685_685546

variables {O : Point} {a b c d e f p q r : ℝ}
variables (A B C : Point) 

def pass_plane_through_fixed_point_and_intersections 
(O A B C : Point) (a b c d e f p q r : ℝ) :=
A = (d, 0, 0) ∧ B = (0, e, 0) ∧ C = (0, 0, f) ∧ 
(p, q, r) is the center of sphere passing through O, A, B, and C

theorem equivalent_proof_to_original_problem 
(pass_plane_through_fixed_point_and_intersections O A B C a b c d e f p q r) :
\frac{a}{p} + \frac{b}{q} + \frac{c}{r} = 2 := sorry

end equivalent_proof_to_original_problem_l685_685546


namespace puppy_eats_times_per_day_l685_685624

theorem puppy_eats_times_per_day 
  (let number_of_puppies := 4)
  (let number_of_dogs := 3)
  (let dog_food_per_meal := 4)
  (let dog_meals_per_day := 3)
  (let total_daily_food := 108)
  (let puppy_meal_food (dog_food_per_meal: ℝ) := dog_food_per_meal / 2)
  (let dog_daily_food (dog_food_per_meal dog_meals_per_day: ℝ) := dog_food_per_meal * dog_meals_per_day)
  (let total_dog_food (number_of_dogs: ℝ) (dog_daily_food: ℝ) := number_of_dogs * dog_daily_food)
  (let total_puppy_food (number_of_puppies: ℝ) (puppy_meal_food: ℝ) (x: ℝ) := number_of_puppies * puppy_meal_food * x) 
  (let equation_correct (dog_food: ℝ) (puppy_food: ℝ) := dog_food + puppy_food = total_daily_food) :
  (let x := (total_daily_food - total_dog_food number_of_dogs (dog_daily_food dog_food_per_meal dog_meals_per_day)) / (number_of_puppies * puppy_meal_food dog_food_per_meal)) :
  x = 9 := by
  sorry

end puppy_eats_times_per_day_l685_685624


namespace condition_for_equation_l685_685895

theorem condition_for_equation (a b c d : ℝ) 
  (h : (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2)) : 
  a = c ∨ a^2 + d + 2 * b = 0 :=
by
  sorry

end condition_for_equation_l685_685895


namespace shaded_region_area_l685_685726

theorem shaded_region_area (EF EH: ℝ) (H_circ_center: ℙ) (F_on_circ: ℙ) 
  (EF_val: EF = 5) (EH_val: EH = 6) (radius: ℝ)
  (circ_radius: radius = Real.sqrt(EF^2 + EH^2)) :
  (1/4) * π * radius ^ 2 - (EF * EH) = (61 * π) / 4 - 30 :=
by
  sorry

end shaded_region_area_l685_685726


namespace range_of_p_slope_of_AB_l685_685229

-- Definitions for conditions
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 1

-- Condition for the parabola and the circle
def vertex_on_circle (p : ℝ) := (∃ x y, parabola p x y ∧ circle x y) ∧ ∀ x y, parabola p x y → x ≠ 0 → ¬ circle x y

-- Theorem 1: Range of p
theorem range_of_p (p : ℝ) (h : p > 0) (h_vertex : vertex_on_circle p) : p ≥ 1 :=
sorry

-- Theorem 2: Slope of AB is non-zero constant
theorem slope_of_AB (p x0 y0 k xA yA xB yB : ℝ) 
  (h_p_gt_0 : p > 0)
  (h_y0_gt_0 : y0 > 0)
  (h_M_on_C1 : parabola p x0 y0)
  (h_A_on_C1 : parabola p xA yA)
  (h_B_on_C1 : parabola p xB yB)
  (h_slope_MA_MB : k ≠ 0 ∧ yA = 2*p/k - y0 ∧ yB= -2*p/k - y0 ∧ xA = 2*p/(k^2) - 2*y0/k + x0 ∧ xB = 2*p/(k^2) + 2*y0/k + x0) :
  (yB - yA) / (xB - xA) = -p / y0 :=
sorry

end range_of_p_slope_of_AB_l685_685229


namespace complex_fraction_simplification_l685_685692

theorem complex_fraction_simplification :
  (2 - (1:ℂ)) / (1 + (2:ℂ) * complex.I) = - complex.I :=
by
  sorry

end complex_fraction_simplification_l685_685692


namespace evaluate_expression_l685_685966

theorem evaluate_expression (m n : ℤ) (hm : m = 2) (hn : n = -3) : (m + n) ^ 2 - 2 * m * (m + n) = 5 := by
  -- Proof skipped
  sorry

end evaluate_expression_l685_685966


namespace f_log2_9_l685_685081

def f (x : ℝ) : ℝ := sorry

theorem f_log2_9 : 
  (∀ x, f (x + 1) = 1 / f x) → 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = 2^x) → 
  f (Real.log 9 / Real.log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end f_log2_9_l685_685081


namespace curve_eq_and_max_distance_l685_685150

theorem curve_eq_and_max_distance
  (x y θ : ℝ)
  (h1 : x = sqrt 3 * Real.cos θ)
  (h2 : y = Real.sin θ)
  (h_line : ∀ ρ, ρ * Real.cos (θ - π / 4) = 2 * sqrt 2 → x = ρ * cos θ ∧ y = ρ * sin θ) :
  (x^2 / 3 + y^2 = 1) ∧
  (x + y - 4 = 0) ∧
  (∀ P : ℝ, 
    P = (sqrt 3 * Real.cos θ, Real.sin θ) → 
    let d := abs (sqrt 3 * Real.cos θ + Real.sin θ - 4) / sqrt 2 in 
    ∃ k : ℤ, d = 3 * sqrt 2) :=
  sorry

end curve_eq_and_max_distance_l685_685150


namespace linear_relationship_max_profit_l685_685315

section cherries
variables (x : ℝ) (y : ℝ) (W : ℝ)
-- Given conditions:
-- cost price
def cost_price : ℝ := 15
-- selling 250 kg at 20 yuan/kg
def sell_20kg := (20 : ℝ)
def sales_20kg := 250
-- profit 2000 yuan when selling at 25 yuan/kg
def sell_25kg := (25 : ℝ)
def profit_2000 := 2000
-- Linear relationship y = kx + b
def y_eq := (-10 : ℝ) * x + 450
-- Profit function W(x)
def profit (x : ℝ) := (x - cost_price) * (y_eq)

-- Proof statements
theorem linear_relationship :
  ∀ (x : ℝ), y_eq = -10 * x + 450 := sorry

theorem max_profit :
  (∃ (x : ℝ), x ≤ 28 ∧ W = profit 28) ∧
  W = 2210 := sorry
end cherries

end linear_relationship_max_profit_l685_685315


namespace isosceles_triangle_base_l685_685323

theorem isosceles_triangle_base (h_perimeter : 2 * 1.5 + x = 3.74) : x = 0.74 :=
by
  sorry

end isosceles_triangle_base_l685_685323


namespace area_ratio_of_extended_equilateral_triangle_l685_685931

theorem area_ratio_of_extended_equilateral_triangle :
  ∀ (D E F E' F' D' : Type) [has_size D E F] [has_distance EE' FF' DD'] [is_equilateral D E F],
    -- Conditions:
    (distance E E' = 2 * distance D E) ∧
    (distance F F' = 2 * distance E F) ∧
    (distance D D' = 2 * distance F D) →
    -- Conclusion:
    area_ratio (triangle D' E' F') (triangle D E F) = 9 :=
by sorry

end area_ratio_of_extended_equilateral_triangle_l685_685931


namespace angle_e1_minus_sqrt3_e2_e2_l685_685635

open Real

variables {e1 e2 : ℝ^3}

def is_unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

def orthogonal (v1 v2 : ℝ^3) : Prop :=
  dot_product v1 v2 = 0

noncomputable def vector_angle_cosine (v1 v2 : ℝ^3) : ℝ :=
  (dot_product v1 v2) / (∥v1∥ * ∥v2∥)

theorem angle_e1_minus_sqrt3_e2_e2 :
  is_unit_vector e1 → is_unit_vector e2 → orthogonal e1 e2 →
  let v := e1 - sqrt 3 * e2 in
  let θ := acos (vector_angle_cosine v e2) in
  θ = 150 * π / 180 :=
by
  intros h1 h2 h3 v θ
  sorry

end angle_e1_minus_sqrt3_e2_e2_l685_685635


namespace find_pairs_l685_685036

theorem find_pairs (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2) := 
by
  sorry

end find_pairs_l685_685036


namespace train_length_l685_685734

theorem train_length (T : ℕ) (S : ℕ) (conversion_factor : ℚ) (L : ℕ) 
  (hT : T = 16)
  (hS : S = 108)
  (hconv : conversion_factor = 5 / 18)
  (hL : L = 480) :
  L = ((S * conversion_factor : ℚ) * T : ℚ) :=
sorry

end train_length_l685_685734


namespace abs_pi_minus_abs_pi_minus_10_l685_685013

theorem abs_pi_minus_abs_pi_minus_10 (h1 : Real.pi < 10) (h2 : 2 * Real.pi < 10) : 
  |Real.pi - |Real.pi - 10|| = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_minus_abs_pi_minus_10_l685_685013


namespace sum_of_coefficients_after_shift_l685_685273

-- Define the original quadratic function
def f(x : ℝ) := 3 * x^2 + 2 * x + 1

-- Define the function shifted five units to the right
def g(x : ℝ) := f(x - 5)

theorem sum_of_coefficients_after_shift:
  let a := 3
  let b := -28
  let c := 66
  a + b + c = 41 :=
by
  sorry

end sum_of_coefficients_after_shift_l685_685273


namespace coprime_reachable_area_l685_685497

noncomputable def reachable_area (v_p v_d t : ℝ) : ℝ :=
  4 * (∫ x in 0..(v_p * t), real.pi * (v_d * (t - x / v_p))^2)

theorem coprime_reachable_area (v_p v_d t : ℝ) (m n : ℕ) (h_coprime : nat.coprime m n) :
  reachable_area v_p v_d t = m / n :=
sorry

-- Example instantiation:
example : coprime_reachable_area 40 10 (15 / 60) 1 1 (by norm_num : nat.coprime 1 1) :=
sorry

end coprime_reachable_area_l685_685497


namespace find_x_l685_685862

noncomputable def f (x : ℝ) : ℝ := Math.cos x * Math.cos (x - Real.pi / 3)

theorem find_x (k : ℤ) : {x | f x < 1 / 4} = ⋃ k, (Set.Ioo (k * Real.pi - 7 * Real.pi / 12) (k * Real.pi - Real.pi / 12)) :=
by
  sorry

end find_x_l685_685862


namespace division_of_cookies_l685_685280

theorem division_of_cookies (n p : Nat) (h1 : n = 24) (h2 : p = 6) : n / p = 4 :=
by sorry

end division_of_cookies_l685_685280


namespace eleventh_term_of_sequence_l685_685029

def inversely_proportional_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = c

theorem eleventh_term_of_sequence :
  ∃ a : ℕ → ℝ,
    (a 1 = 3) ∧
    (a 2 = 6) ∧
    inversely_proportional_sequence a 18 ∧
    a 11 = 3 :=
by
  sorry

end eleventh_term_of_sequence_l685_685029


namespace derivative_sin_over_x_l685_685387

noncomputable def differentiate_sin_over_x (x : ℝ) (x ≠ 0) : ℝ := derivative (λ x, sin x / x) x

theorem derivative_sin_over_x (x : ℝ) (h : x ≠ 0) :
  differentiate_sin_over_x x h = (x * cos x - sin x) / x^2 :=
sorry

end derivative_sin_over_x_l685_685387


namespace quadratic_real_roots_l685_685418

theorem quadratic_real_roots (a b c : ℝ) (h : a * c < 0) : 
  ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y :=
by
  sorry

end quadratic_real_roots_l685_685418


namespace sequence_general_formula_and_max_value_l685_685442

theorem sequence_general_formula_and_max_value
  (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x, (2 * a * x + b) = -2 * x + 7)
  (S : ℕ → ℝ)
  (h3 : ∀ n : ℕ, S n = -(n : ℝ)^2 + 7 * (n : ℝ)) :
  (∀ n : ℕ, a_n n = -2 * (n : ℝ) + 8) ∧ (S 3 = 12 ∧ S 4 = 12) :=
begin
  sorry
end

end sequence_general_formula_and_max_value_l685_685442


namespace range_of_m_l685_685825

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (1 / y) = 1) (h2 : x + 2 * y > m^2 + 2 * m) : -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l685_685825


namespace measure_angle_EDF_l685_685997

theorem measure_angle_EDF (P D E F : Point) (h1 : is_circumcenter P D E F)
  (h2 : ∠DP E = 110) (h3 : ∠EP F = 150) : ∠ED F = 50 :=
by sorry

end measure_angle_EDF_l685_685997


namespace probability_two_planes_in_3_minutes_probability_fewer_than_two_planes_in_3_minutes_probability_at_least_two_planes_in_3_minutes_l685_685214

noncomputable def poisson_pmf (λ k : ℝ) (t k : ℕ) : ℝ :=
  ((λ * t)^k * Real.exp(-λ * t)) / Real.factorial k

def λ : ℝ := 4
def t : ℕ := 3

theorem probability_two_planes_in_3_minutes :
  poisson_pmf λ t 2 = 72 * Real.exp(-12) := by
  sorry

theorem probability_fewer_than_two_planes_in_3_minutes :
  (poisson_pmf λ t 0 + poisson_pmf λ t 1) = 13 * Real.exp(-12) := by
  sorry

theorem probability_at_least_two_planes_in_3_minutes :
  (1 - (poisson_pmf λ t 0 + poisson_pmf λ t 1)) = 1 - 13 * Real.exp(-12) := by
  sorry

end probability_two_planes_in_3_minutes_probability_fewer_than_two_planes_in_3_minutes_probability_at_least_two_planes_in_3_minutes_l685_685214


namespace exists_multiple_with_digits_0_or_1_l685_685963

theorem exists_multiple_with_digits_0_or_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k % n = 0) ∧ (∀ digit ∈ k.digits 10, digit = 0 ∨ digit = 1) ∧ (k.digits 10).length ≤ n :=
sorry

end exists_multiple_with_digits_0_or_1_l685_685963


namespace smallest_solution_of_quadratic_l685_685670

theorem smallest_solution_of_quadratic :
  ∃ x : ℝ, 6 * x^2 - 29 * x + 35 = 0 ∧ x = 7 / 3 :=
sorry

end smallest_solution_of_quadratic_l685_685670


namespace completing_the_square_solution_l685_685658

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l685_685658


namespace intersection_vertices_of_regular_octagon_l685_685456

noncomputable def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}

def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem intersection_vertices_of_regular_octagon (a : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_A a ∧ p ∈ set_B) ↔ (a = Real.sqrt 2 ∨ a = 2 + Real.sqrt 2) :=
  sorry

end intersection_vertices_of_regular_octagon_l685_685456


namespace stratified_sampling_correct_l685_685906

-- Definitions as per conditions
def total_students : ℕ := 36 + 18
def sample_size : ℕ := 9
def number_of_female_students : ℕ := 18
def number_of_male_students : ℕ := 36
def female_ratio : ℚ := number_of_female_students / total_students 

-- Lean statement to prove the problem
theorem stratified_sampling_correct : 
  (female_ratio * sample_size).to_nat = 3 :=
by
  -- Proof omitted
  sorry

end stratified_sampling_correct_l685_685906


namespace pq_div_bq_l685_685691

def Circle (Q : Type) := ∃ (O : Q) (r : ℝ), ∀ (A B C D : Q), 
  dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r ∧ 
  ∠ A O B = 90 ∧ ∠ C O D = 90

variables {Q : Type} [MetricSpace Q]

/-- PQ divided by BQ is sqrt(2) / 2 given the geometric constraints. -/
theorem pq_div_bq (O P A B C D : Q) (r : ℝ) (h: Circle Q)
  (h_perpendicular1: ∠ A O B = 90)
  (h_perpendicular2: ∠ C O D = 90)
  (h_P: ∠ Q P C = 45) 
  (h_extends: P lies_on (line_extend B Q)) :
  PQ / BQ = sqrt(2) / 2 :=
sorry

end pq_div_bq_l685_685691


namespace evaluate_complex_expression_l685_685032

theorem evaluate_complex_expression : (let i := Complex.I in i * (1 - 2 * i)) = (2 + Complex.I) := by
  sorry

end evaluate_complex_expression_l685_685032


namespace range_of_fx_l685_685445

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / (Real.sin x * Real.cos x + 1)

theorem range_of_fx : Set.range f = Set.Icc (-1 : ℝ) 1 := 
sorry

end range_of_fx_l685_685445


namespace missed_questions_l685_685191

theorem missed_questions (F M : ℕ) (h1 : M = 5 * F) (h2 : M + F = 216) : M = 180 :=
by
  sorry

end missed_questions_l685_685191


namespace cartesian_line_and_max_value_l685_685076

noncomputable def ellipse : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ y^2 / 16 + x^2 / 4 = 1}

def polarLine (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 3) = 3

def parametricEllipse (φ : ℝ) : ℝ × ℝ :=
  (2 * cos φ, 4 * sin φ)

theorem cartesian_line_and_max_value :
  (∀ ρ θ, polarLine ρ θ → ∃ x y, (ρ * cos θ, ρ * sin θ) = (x, y) ∧ sqrt 3 * x + y - 6 = 0) ∧
  (∀ φ, parametricEllipse φ ∈ ellipse) ∧
  (∃ M : ℝ × ℝ, M ∈ ellipse → ∀ x y, M = (x, y) → (|2 * sqrt 3 * x + y - 1| ≤ 9)) :=
by
  sorry

end cartesian_line_and_max_value_l685_685076


namespace true_propositions_l685_685234

noncomputable def sphere_radius : ℝ := 4
noncomputable def chord_AB_length : ℝ := 2 * real.sqrt 7
noncomputable def chord_CD_length : ℝ := 4 * real.sqrt 3
noncomputable def midpoint_M_distance : ℝ := 3
noncomputable def midpoint_N_distance : ℝ := 2
noncomputable def max_MN_distance : ℝ := 5
noncomputable def min_MN_distance : ℝ := 1

theorem true_propositions 
  (r : ℝ := sphere_radius)
  (AB : ℝ := chord_AB_length)
  (CD : ℝ := chord_CD_length)
  (OM : ℝ := midpoint_M_distance)
  (ON : ℝ := midpoint_N_distance)
  (maxMN : ℝ := max_MN_distance)
  (minMN : ℝ := min_MN_distance)
  : ① maxMN = 5 ∧ ③ minMN = 1 ∧ ④ (AB ∩ CD ≠ ∅) := sorry

end true_propositions_l685_685234


namespace smaller_molds_radius_l685_685713

noncomputable def large_radius : ℝ := 2
def num_molds : ℕ := 64

noncomputable def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * real.pi * (r ^ 3)

theorem smaller_molds_radius : ∃ r : ℝ, (num_molds * volume_hemisphere r = volume_hemisphere large_radius) ∧ (r = 1 / 2) :=
by
  use (1 / 2)
  sorry

end smaller_molds_radius_l685_685713


namespace equal_12_mn_P_2n_Q_m_l685_685583

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l685_685583


namespace element_with_mass_percent_61_54_in_Al2CO33_is_oxygen_l685_685802

-- Defining the atomic masses of the elements
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_C : ℝ := 12.01
def atomic_mass_O : ℝ := 16.00

-- Defining the numbers of each element in Al2(CO3)3
def num_Al : ℝ := 2
def num_C : ℝ := 3
def num_O : ℝ := 9

-- Defining the total molar mass of Al2(CO3)3
def molar_mass_Al2CO33 : ℝ := num_Al * atomic_mass_Al + num_C * atomic_mass_C + num_O * atomic_mass_O

-- Calculating the mass percentages
def mass_percent_Al : ℝ := (num_Al * atomic_mass_Al) / molar_mass_Al2CO33 * 100
def mass_percent_C : ℝ := (num_C * atomic_mass_C) / molar_mass_Al2CO33 * 100
def mass_percent_O : ℝ := (num_O * atomic_mass_O) / molar_mass_Al2CO33 * 100

-- Stating the problem as a theorem
theorem element_with_mass_percent_61_54_in_Al2CO33_is_oxygen :
  mass_percent_O = 61.54 :=
by sorry

end element_with_mass_percent_61_54_in_Al2CO33_is_oxygen_l685_685802


namespace product_of_common_divisors_180_45_l685_685044

theorem product_of_common_divisors_180_45 : 
  (∏ d in ({d : ℤ | d ∣ 180 ∧ d ∣ 45}), d) = 65181640625 := 
by
  sorry

end product_of_common_divisors_180_45_l685_685044


namespace findABInverse_l685_685537

noncomputable def transformCircleToEllipse (A : Matrix (fin 2) (fin 2) ℝ) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ P : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 = 1) →
  ((A ⬝ (λ x : ℝ × ℝ, ![x.1, x.2])) P) = (λ Q : ℝ × ℝ, (Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)) P

theorem findABInverse (A : Matrix (fin 2) (fin 2) ℝ) (a b : ℝ) :
  transformCircleToEllipse A a b → 
  (a = 2 ∧ b = 1 / 2 ∧ A⁻¹ = (Matrix.vecCons (λ x, ![1 / 2, 0]) (Matrix.vecCons (λ x, ![0, 2])))) :=
by
  -- proof goes here
  sorry

end findABInverse_l685_685537


namespace problem1_valid_problem2_valid_l685_685766

noncomputable def problem1_expr : ℝ :=
  real.sqrt (1/6) * real.sqrt 96 / real.sqrt 6

noncomputable def problem1_answer : ℝ :=
  (2 * real.sqrt 6) / 3

noncomputable def problem2_expr : ℝ :=
  real.sqrt 80 - real.sqrt 8 - real.sqrt 45 + 4 * real.sqrt (1/2)

noncomputable def problem2_answer : ℝ :=
  real.sqrt 5

theorem problem1_valid : problem1_expr = problem1_answer := 
by sorry

theorem problem2_valid : problem2_expr = problem2_answer := 
by sorry

end problem1_valid_problem2_valid_l685_685766


namespace solve_problem_l685_685055

variables (α β : ℝ)
def vec_a := (Real.cos α, 3)
def vec_b := (-4, Real.sin α)

theorem solve_problem (h1 : 0 < α ∧ α < π/2 ∧ π/2 < β ∧ β < π)
  (h2 : vec_a α ⬝ vec_b α = 0)
  (h3 : Real.cos (β - α) = sqrt 2 / 10) :
  tan α = 4 / 3 ∧ sin α = 4 / 5 ∧ sin β = sqrt 2 / 2 :=
sorry

end solve_problem_l685_685055


namespace sequences_24_length_l685_685886

noncomputable def g : ℕ → ℕ
| 0 := 0
| 1 := 0
| 2 := 0
| 3 := 1
| 4 := 1
| 5 := 2
| n := if h : n ≥ 6 then g (n-4) + 2 * g (n-5) + 2 * g (n-6) else 0

theorem sequences_24_length :
  g 24 = 100 :=
by
  sorry

end sequences_24_length_l685_685886


namespace main_problem_l685_685000

noncomputable def termSeq1 (n : ℕ) : ℝ :=
  ∏ k in Finset.range (n + 1), 1 + 15 / (1 + k)

noncomputable def termSeq2 (m : ℕ) : ℝ :=
  ∏ j in Finset.range (m + 1), 1 + 21 / (1 + j)

theorem main_problem :
  (termSeq1 21) / (termSeq2 15) = 1 :=
sorry

end main_problem_l685_685000


namespace total_banana_produce_correct_l685_685699

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l685_685699


namespace spiral_strip_winds_twice_l685_685298

noncomputable def spiral_strip_length (circumference height: ℝ) (wraps: ℕ) : ℝ :=
  let width := circumference
  let new_height := height * wraps
  real.sqrt (width ^ 2 + new_height ^ 2)

theorem spiral_strip_winds_twice (circumference height : ℝ) (wraps : ℕ) :
  circumference = 16 →
  height = 8 →
  wraps = 2 →
  spiral_strip_length circumference height wraps = 16 * real.sqrt 2 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  dsimp [spiral_strip_length],
  rw [mul_assoc, pow_two, add_comm (8 * 2)^2, mul_add, <- pow_two (16:ℝ), mul_assoc, pow_two (8:ℝ)],
  rw [pow_two (16:ℝ), pow_two (8:ℝ), mul_pow, ← add_mul],
  norm_num,
end

end spiral_strip_winds_twice_l685_685298


namespace nylon_needed_is_192_l685_685163

-- Define the required lengths for the collars
def nylon_needed_for_dog_collar : ℕ := 18
def nylon_needed_for_cat_collar : ℕ := 10

-- Define the number of collars needed
def number_of_dog_collars : ℕ := 9
def number_of_cat_collars : ℕ := 3

-- Define the total nylon needed
def total_nylon_needed : ℕ :=
  (nylon_needed_for_dog_collar * number_of_dog_collars) + (nylon_needed_for_cat_collar * number_of_cat_collars)

-- State the theorem we need to prove
theorem nylon_needed_is_192 : total_nylon_needed = 192 := 
  by
    -- Simplification to match the complete statement for completeness
    sorry

end nylon_needed_is_192_l685_685163


namespace small_mold_radius_l685_685716

-- Define the parameters of the problem
def large_bowl_radius : ℝ := 2
def num_small_molds : ℝ := 64

-- Define the volume of a hemisphere formula
def hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

-- Given conditions
def large_bowl_volume : ℝ := hemisphere_volume large_bowl_radius
def total_small_molds_volume (r : ℝ) : ℝ := num_small_molds * hemisphere_volume r

-- The statement to prove
theorem small_mold_radius :
  ∃ r : ℝ, total_small_molds_volume r = large_bowl_volume ∧ r = 1 / 2 := by
  sorry

end small_mold_radius_l685_685716


namespace coconut_transport_l685_685758

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l685_685758


namespace flagpole_break_height_l685_685302

theorem flagpole_break_height (total_height break_point distance_from_base : ℝ) 
(h_total : total_height = 6) 
(h_distance : distance_from_base = 2) 
(h_equation : (distance_from_base^2 + (total_height - break_point)^2) = break_point^2) :
  break_point = 3 := 
sorry

end flagpole_break_height_l685_685302


namespace monotonicity_f_min_g_value_final_inequality_l685_685863

def f (a b : ℝ) (x : ℝ) : ℝ := a * x * Real.log x + b * x
def g (x : ℝ) : ℝ := x + 1 / Real.exp (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x ∈ (0, 1 : ℝ), f a (-a) x < 0 → a > 0 → a * Real.log x < 0) ∧ 
  (∀ x ∈ (1, +∞ : ℝ), f a (-a) x > 0 → a > 0 → a * Real.log x > 0) ∧
  (∀ x ∈ (0, 1 : ℝ), f a (-a) x > 0 → a < 0 → a * Real.log x > 0) ∧
  (∀ x ∈ (1, +∞ : ℝ), f a (-a) x < 0 → a < 0 → a * Real.log x < 0) := 
sorry

theorem min_g_value :
  ∀ x ∈ (0, +∞ : ℝ), g x ≥ 2 := 
sorry

theorem final_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x ∈ (0, +∞ : ℝ), (f a (-a) x) / a + 2 / (x * Real.exp (x - 1) + 1) ≥ 1 - x :=
  sorry

end monotonicity_f_min_g_value_final_inequality_l685_685863


namespace altitude_on_hypotenuse_l685_685498

theorem altitude_on_hypotenuse (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) (c : ℝ) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  ∃ h : ℝ, h = (a * b) / c ∧ h = 60 / 13 :=
by
  use (5 * 12) / 13
  -- proof that (60 / 13) is indeed the altitude will be done by verifying calculations
  sorry

end altitude_on_hypotenuse_l685_685498


namespace intersection_of_A_and_B_l685_685846

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end intersection_of_A_and_B_l685_685846


namespace trigonometric_identity_l685_685821

variable (α : Real)
hypothesis (h : Real.sin (-α) = sqrt 5 / 3)

theorem trigonometric_identity : Real.cos (Real.pi / 2 + α) = sqrt 5 / 3 :=
by
  sorry

end trigonometric_identity_l685_685821


namespace treasure_value_l685_685808

theorem treasure_value
    (fonzie_paid : ℕ) (auntbee_paid : ℕ) (lapis_paid : ℕ)
    (lapis_share : ℚ) (lapis_received : ℕ) (total_value : ℚ)
    (h1 : fonzie_paid = 7000) 
    (h2 : auntbee_paid = 8000) 
    (h3 : lapis_paid = 9000) 
    (h4 : fonzie_paid + auntbee_paid + lapis_paid = 24000) 
    (h5 : lapis_share = lapis_paid / (fonzie_paid + auntbee_paid + lapis_paid)) 
    (h6 : lapis_received = 337500) 
    (h7 : lapis_share * total_value = lapis_received) :
  total_value = 1125000 := by
  sorry

end treasure_value_l685_685808


namespace no_integer_n_gt_1_satisfies_inequality_l685_685361

open Int

theorem no_integer_n_gt_1_satisfies_inequality :
  ∀ (n : ℤ), n > 1 → ¬ (⌊(Real.sqrt (↑n - 2) + 2 * Real.sqrt (↑n + 2))⌋ < ⌊Real.sqrt (9 * (↑n : ℝ) + 6)⌋) :=
by
  intros n hn
  sorry

end no_integer_n_gt_1_satisfies_inequality_l685_685361


namespace max_spheres_tangent_l685_685064

-- Each sphere touches all the other n-1 spheres, and no three spheres touch at the same point.
def spheres_tangent_condition (n : ℕ) : Prop :=
  ∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k → (touches i j ∧ touches i k ∧ touches j k)

axiom touches : Π {n : ℕ}, (fin n) → (fin n) → Prop

theorem max_spheres_tangent (n : ℕ) (h1 : 0 < n) (h2 : spheres_tangent_condition n) : n ≤ 5 := 
  sorry

end max_spheres_tangent_l685_685064


namespace missing_number_in_sequence_l685_685390

theorem missing_number_in_sequence :
  ∀ (a1 a2 a3 a4 a6 : ℕ),
  a1 = 2 → 
  a2 = 6 → 
  a3 = 12 → 
  a4 = 20 → 
  a6 = 42 → 
  (a2 - a1) = 4 → 
  (a3 - a2) = 6 →
  (a4 - a3) = 8 → 
  let a5 := a4 + 10 in
  a5 = 30 :=
by
  intros a1 a2 a3 a4 a6 ha1 ha2 ha3 ha4 ha6 diff12 diff23 diff34
  let a5 := a4 + 10
  sorry

end missing_number_in_sequence_l685_685390


namespace prob_union_l685_685128

-- Define the probabilities of events A and B
def P (A : Prop) : ℝ

-- Setting conditions
axiom mutually_exclusive (A B : Prop) : A → ¬ B
axiom prob_A : P A = 0.6
axiom prob_B : P B = 0.3

-- Define probabilistic union for exclusive events
theorem prob_union (A B : Prop) (h1 : mutually_exclusive A B) (h2 : P A = 0.6) (h3 : P B = 0.3) : P (A ∪ B) = 0.9 :=
by
  sorry

end prob_union_l685_685128


namespace unique_solution_max_a_l685_685466

theorem unique_solution_max_a :
  (∃! (x y : ℝ) (a : ℝ), (y = 1 - real.sqrt x) ∧ (a - 2 * (a - y)^2 = real.sqrt x)) → a = 2 :=
sorry

end unique_solution_max_a_l685_685466


namespace initial_brownies_l685_685957

theorem initial_brownies (B : ℕ) (eaten_by_father : ℕ) (eaten_by_mooney : ℕ) (new_brownies : ℕ) (total_brownies : ℕ) :
  eaten_by_father = 8 →
  eaten_by_mooney = 4 →
  new_brownies = 24 →
  total_brownies = 36 →
  (B - (eaten_by_father + eaten_by_mooney) + new_brownies = total_brownies) →
  B = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end initial_brownies_l685_685957


namespace range_of_x_l685_685097

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x : {x : ℝ | f (x^2) < f (3 * x - 2)} = {x : ℝ | 1 < x ∧ x < 2} 
  where 
  f (x : ℝ) : ℝ := Real.exp x + x^3 := by
  sorry

end range_of_x_l685_685097


namespace tangent_line_eq_range_of_a_l685_685444

-- Part 1: Equation of the tangent line
theorem tangent_line_eq (f : ℝ → ℝ) (a : ℝ) (x : ℝ) (h₀ : f x = (x + 1) * Real.exp x) :
  (∀ x, deriv f x = (x + 2) * Real.exp x) → 
  f 0 = 1 → 
  deriv f 0 = 2 → 
  tangent_eq_line f (0, 1) 2 :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (x : ℝ) 
  (h₀ : f x = (x + a) * Real.exp x)
  (ineq : ∀ x, f x ≥ (1/6) * x ^ 3 - x - 2) :
  a ≥ -2 :=
sorry

end tangent_line_eq_range_of_a_l685_685444


namespace largest_product_is_168_l685_685022

open Set

noncomputable def largest_product_from_set (s : Set ℤ) (n : ℕ) (result : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x y z : ℤ), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
  x * y * z ≤ a * b * c ∧ a * b * c = result

theorem largest_product_is_168 :
  largest_product_from_set {-4, -3, 1, 3, 7, 8} 3 168 :=
sorry

end largest_product_is_168_l685_685022


namespace solve_cos_arcsin_fraction_equivalence_l685_685005

noncomputable def cos_arcsin_fraction_equivalence : Prop :=
  \(\cos \left( \arcsin \frac{3}{5} \right) = \frac{4}{5}\)

theorem solve_cos_arcsin_fraction_equivalence : cos_arcsin_fraction_equivalence :=
by
  sorry

end solve_cos_arcsin_fraction_equivalence_l685_685005


namespace triangle_angle_sum_acute_l685_685605

theorem triangle_angle_sum_acute (x : ℝ) (h1 : 60 + 70 + x = 180) (h2 : x ≠ 60 ∧ x ≠ 70) :
  x = 50 ∧ (60 < 90 ∧ 70 < 90 ∧ x < 90) := by
  sorry

end triangle_angle_sum_acute_l685_685605


namespace determinant_calculation_l685_685792

variable {R : Type*} [CommRing R]

def matrix_example (a b c : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![1, a, b], ![1, a + b, b + c], ![1, a, a + c]]

theorem determinant_calculation (a b c : R) :
  (matrix_example a b c).det = ab + b^2 + bc :=
by sorry

end determinant_calculation_l685_685792


namespace cos_arcsin_l685_685010

theorem cos_arcsin (h₀ : real.sqrt (5^2 - 3^2) = 4) : real.cos (real.arcsin (3 / 5)) = 4 / 5 :=
by 
  sorry

end cos_arcsin_l685_685010


namespace solution_set_inequality_l685_685047

open Set

theorem solution_set_inequality (x : ℝ) :
  (2 ^ (x^2 - 5 * x + 5) > 1 / 2) ↔ x < 2 ∨ x > 3 :=
by
  -- By recognizing that 2^x is increasing and simplifying
  have h : (2 ^ (x^2 - 5 * x + 5)) > 1 / 2 ↔ (x^2 - 5 * x + 5) > -1 := by sorry
  -- Simplifying the inequality further
  have h1 : (x^2 - 5 * x + 5) > -1 ↔ (x^2 - 5 * x + 6) > 0 := by sorry
  -- Solving for the quadratic inequality
  exact sorry

end solution_set_inequality_l685_685047


namespace negation_of_proposition_l685_685182

theorem negation_of_proposition (p : ∀ (x : ℝ), x^2 + 1 > 0) :
  ∃ (x : ℝ), x^2 + 1 ≤ 0 ↔ ¬ (∀ (x : ℝ), x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l685_685182


namespace a_n_general_term_b_n_general_term_l685_685780

noncomputable def seq_a (n : ℕ) : ℕ :=
  2 * n - 1

theorem a_n_general_term (n : ℕ) (Sn : ℕ → ℕ) (S_property : ∀ n : ℕ, 4 * Sn n = (seq_a n) ^ 2 + 2 * seq_a n + 1) :
  seq_a n = 2 * n - 1 :=
sorry

noncomputable def geom_seq (q : ℕ) (n : ℕ) : ℕ :=
  q ^ (n - 1)

theorem b_n_general_term (n m q : ℕ) (a1 am am3 : ℕ) (b_property : ∀ n : ℕ, geom_seq q n = q ^ (n - 1))
  (a_property : ∀ n : ℕ, seq_a n = 2 * n - 1)
  (b1_condition : geom_seq q 1 = seq_a 1) (bm_condition : geom_seq q m = seq_a m)
  (bm1_condition : geom_seq q (m + 1) = seq_a (m + 3)) :
  q = 3 ∨ q = 7 ∧ (∀ n : ℕ, geom_seq q n = 3 ^ (n - 1) ∨ geom_seq q n = 7 ^ (n - 1)) :=
sorry

end a_n_general_term_b_n_general_term_l685_685780


namespace jameson_total_medals_l685_685162

-- Define the number of track, swimming, and badminton medals
def track_medals := 5
def swimming_medals := 2 * track_medals
def badminton_medals := 5

-- Define the total number of medals
def total_medals := track_medals + swimming_medals + badminton_medals

-- Theorem statement
theorem jameson_total_medals : total_medals = 20 := 
by
  sorry

end jameson_total_medals_l685_685162


namespace count_birches_in_forest_l685_685135

theorem count_birches_in_forest:
  ∀ (t p_s p_p : ℕ), t = 4000 → p_s = 10 → p_p = 13 →
  let n_s := (p_s * t) / 100 in
  let n_p := (p_p * t) / 100 in
  let n_o := n_s + n_p in 
  let n_b := t - (n_s + n_p + n_o) in 
  n_b = 2160 :=
by 
  intros t p_s p_p ht hps hpp
  let n_s := (p_s * t) / 100 
  let n_p := (p_p * t) / 100 
  let n_o := n_s + n_p 
  let n_b := t - (n_s + n_p + n_o) 
  exact sorry

end count_birches_in_forest_l685_685135


namespace expected_heads_after_three_tosses_l685_685922

theorem expected_heads_after_three_tosses :
  let n := 64 in
  let p := (1/2 : ℝ) + (1/4) + (1/8) in
  n * p = 56 :=
by
  sorry

end expected_heads_after_three_tosses_l685_685922


namespace bisector_angle_v_l685_685934

def vector := ℝ × ℝ × ℝ

def a : vector := (2, 3, 1)
def b : vector := (4, -2, 2)
def unit_vector (v : vector) : Prop :=
  (v.fst^2 + v.snd^2 + v.snd.snd^2) = 1

noncomputable def v : vector := 
  (4 / Real.sqrt 120, -10 / Real.sqrt 120, 2 / Real.sqrt 120)

theorem bisector_angle_v (v : vector) (h : unit_vector v) :
  b = (sqrt 14 / 2) • (a + (sqrt 14) • v) :=
sorry

end bisector_angle_v_l685_685934


namespace count_birches_in_forest_l685_685134

theorem count_birches_in_forest:
  ∀ (t p_s p_p : ℕ), t = 4000 → p_s = 10 → p_p = 13 →
  let n_s := (p_s * t) / 100 in
  let n_p := (p_p * t) / 100 in
  let n_o := n_s + n_p in 
  let n_b := t - (n_s + n_p + n_o) in 
  n_b = 2160 :=
by 
  intros t p_s p_p ht hps hpp
  let n_s := (p_s * t) / 100 
  let n_p := (p_p * t) / 100 
  let n_o := n_s + n_p 
  let n_b := t - (n_s + n_p + n_o) 
  exact sorry

end count_birches_in_forest_l685_685134


namespace set_C_cannot_form_triangle_l685_685676

-- Define the sets of line segments as lists of real numbers
def set_A : List ℝ := [3, 4, 5]
def set_B : List ℝ := [real.sqrt 3, 1, 2]
def set_C : List ℝ := [3, 6, 3]
def set_D : List ℝ := [1.5, 2.5, 3]

-- Function to check if three given segments can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- The main theorem stating which set cannot form a triangle
theorem set_C_cannot_form_triangle :
  ¬ can_form_triangle (set_C.nthLe 0 (by simp)) (set_C.nthLe 1 (by simp)) (set_C.nthLe 2 (by simp)) :=
  by sorry

end set_C_cannot_form_triangle_l685_685676


namespace find_perimeter_square3_l685_685683

-- Define the conditions: perimeter of first and second square
def perimeter_square1 := 60
def perimeter_square2 := 48

-- Calculate side lengths based on the perimeter
def side_length_square1 := perimeter_square1 / 4
def side_length_square2 := perimeter_square2 / 4

-- Calculate areas of the two squares
def area_square1 := side_length_square1 * side_length_square1
def area_square2 := side_length_square2 * side_length_square2

-- Calculate the area of the third square
def area_square3 := area_square1 - area_square2

-- Calculate the side length of the third square
def side_length_square3 := Nat.sqrt area_square3

-- Define the perimeter of the third square
def perimeter_square3 := 4 * side_length_square3

/-- Theorem: The perimeter of the third square is 36 cm -/
theorem find_perimeter_square3 : perimeter_square3 = 36 := by
  sorry

end find_perimeter_square3_l685_685683


namespace common_measure_exists_l685_685799

-- Define the conditions that the lengths of the segments are given.
def segment_length1 : ℝ := 1/5
def segment_length2 : ℝ := 1/3

-- State that the common measure should be 1/15.
def common_measure : ℝ := 1/15

-- Main theorem to prove that 1/15 is the common measure for the given segments
theorem common_measure_exists (h1: segment_length1 = 1/5) (h2: segment_length2 = 1/3) : 
  ∃ (m : ℝ), m = common_measure ∧ segment_length1 % m = 0 ∧ segment_length2 % m = 0 := by
  sorry

end common_measure_exists_l685_685799


namespace hyperbola_equation_l685_685855

theorem hyperbola_equation {a b c : ℝ} (h1 : c = 5) (h2 : 2 * a = b) 
  (h3 : c^2 = a^2 + b^2) : 
  (∃ (a b : ℝ), a^2 = 5 ∧ b^2 = 20 ∧ 
    (∀ x y : ℝ, x^2 / 5 - y^2 / 20 = 1)) :=
begin
  use [sqrt 5, sqrt 20],
  split,
  { field_simp,
    exact (sqrt_sq (5)).symm, },
  split,
  { field_simp,
    exact (sqrt_sq (20)).symm, },
  { intros x y,
    have h : sqrt 5^2 = 5 := sqrt_sq (5),
    have h' : sqrt 20^2 = 20 := sqrt_sq (20),
    rw [h, h'],
    exact rfl, }
end

end hyperbola_equation_l685_685855


namespace solve_cos_arcsin_fraction_equivalence_l685_685007

noncomputable def cos_arcsin_fraction_equivalence : Prop :=
  \(\cos \left( \arcsin \frac{3}{5} \right) = \frac{4}{5}\)

theorem solve_cos_arcsin_fraction_equivalence : cos_arcsin_fraction_equivalence :=
by
  sorry

end solve_cos_arcsin_fraction_equivalence_l685_685007


namespace rose_bought_flowers_l685_685565

theorem rose_bought_flowers (F : ℕ) (h1 : ∃ (daisies tulips sunflowers : ℕ), daisies = 2 ∧ sunflowers = 4 ∧ 
  tulips = (3 / 5) * (F - 2) ∧ sunflowers = (2 / 5) * (F - 2)) : F = 12 :=
sorry

end rose_bought_flowers_l685_685565


namespace find_d_l685_685890

theorem find_d (d : ℤ) :
  (∀ x : ℤ, 6 * x^3 + 19 * x^2 + d * x - 15 = 0) ->
  d = -32 :=
by
  sorry

end find_d_l685_685890


namespace expression_value_l685_685620

theorem expression_value : (5 - 2) / (2 + 1) = 1 := by
  sorry

end expression_value_l685_685620


namespace lines_concurrent_or_parallel_l685_685180

-- Definitions for geometric entities
variables {Point : Type} [affine_space Point] (A B C D P Q R S O : Point)

-- Convex quadrilateral
variable (hConvex : convex_quad A B C D)

-- Points on the sides of the quadrilateral
variable (hP : on_segment A B P)
variable (hQ : on_segment B C Q)
variable (hR : on_segment C D R)
variable (hS : on_segment D A S)

-- Intersection point O of PR and QS
variable (hIntersection : intersect PR QS O)

-- Incircles conditions for each of the quadrilaterals
variable (hIncircles : has_incircle (A P O S) ∧ has_incircle (B Q O P) ∧ has_incircle (C R O Q) ∧ has_incircle (D S O R))

-- Theorem statement
theorem lines_concurrent_or_parallel :
  concurrent_or_parallel A C P Q R S :=
sorry

end lines_concurrent_or_parallel_l685_685180


namespace incenter_circumcenter_collinear_l685_685878

theorem incenter_circumcenter_collinear 
  (A B C K : Point) 
  (O I : Point)
  (O1 O2 O3 : Circle) 
  (h1 : K ∈ O1 ∧ K ∈ O2 ∧ K ∈ O3)
  (h2 : O1.radius = O2.radius ∧ O2.radius = O3.radius ∧ O1.radius = O3.radius)
  (h3 : O1.tangent_to_side A B ∧ O1.tangent_to_side A C)
  (h4 : O2.tangent_to_side B A ∧ O2.tangent_to_side B C)
  (h5 : O3.tangent_to_side C A ∧ O3.tangent_to_side C B)
  (h6 : O.circumcenter_of_triangle A B C)
  (h7 : I.incenter_of_triangle A B C) :
  collinear I K O :=
sorry

end incenter_circumcenter_collinear_l685_685878


namespace min_value_l685_685450

-- Given conditions
variables (a b : ℝ)
variable (h_a : a ≥ 1)
variable (h_b : b ≥ 1)
variable (h_hyperbola : ∀ x y, (x^2 / a^2 - y^2 / b^2 = 1))
variable (h_eccentricity : ∀ c, c / a = 2)

-- Proving the minimum value of the given expression
theorem min_value (h_a : a ≥ 1) (h_b : b ≥ 1) (h_eccentricity : ∀ c, c / a = 2) :
  ∀ t : ℝ, t = (b^2 + 1) / (sqrt 3 * a) → t ≥ 4 * sqrt 3 / 3 :=
sorry

end min_value_l685_685450


namespace completing_the_square_l685_685646

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l685_685646


namespace exponent_question_l685_685473

theorem exponent_question (x y : ℝ) (h : x + 2 * y = 3) : 2^x * 4^y = 8 :=
by
  sorry

end exponent_question_l685_685473


namespace find_pairs_of_natural_numbers_l685_685038

theorem find_pairs_of_natural_numbers (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔ (m, n) = (1, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (3, 2) :=
by
  sorry

end find_pairs_of_natural_numbers_l685_685038


namespace range_of_a_l685_685949

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x - 3 else x^2

theorem range_of_a (a : ℝ) : f a > 1 ↔ a ∈ Set.Ioo 1 ⊤ ∪ Set.Iio (-2) :=
by
  sorry

end range_of_a_l685_685949


namespace exponent_equality_l685_685893

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l685_685893


namespace angle_AOC_l685_685617

theorem angle_AOC :
  ∀ (A B O C D P : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace O] [MetricSpace C] [MetricSpace D] [MetricSpace P]
  (AB_cd : ℝ) (OD_c_eq : OD = DP) (angle_APC_eq : ∠APC = 18) (circle_O_AB_diameter : is_diameter AB O)
  (length_AB : distance A B = 16),
  ∠AOC = 54 := 
by
  sorry

end angle_AOC_l685_685617


namespace euler_totient_divides_l685_685810

noncomputable def phi (n : ℕ) : ℕ :=
if n = 0 then 0 else n * (Finset.univ.filter (λ m, Nat.coprime m n)).card / n

theorem euler_totient_divides (
  positive_integers_le_100 : Finset ℕ := Finset.filter (λ n, n ≤ 100) (Finset.range 101)) :
  positive_integers_le_100.filter (λ n, n % phi n = 0)).card = 19 :=
sorry

end euler_totient_divides_l685_685810


namespace find_positive_integer_pairs_l685_685798

theorem find_positive_integer_pairs :
  ∀ (x n : ℕ), x > 0 → n > 0 → 3 * 2^x + 4 = n^2 → (x = 2 ∧ n = 4) ∨ (x = 5 ∧ n = 10) ∨ (x = 6 ∧ n = 14) :=
by {
  intros x n x_pos n_pos h,
  sorry
}

end find_positive_integer_pairs_l685_685798


namespace rectangle_area_conditions_l685_685833

variables {A B C D P Q : ℝ}

-- Given a rectangle ABCD with an area equal to 2
def rectangle_area (a b : ℝ) : Prop := a * b = 2

-- Point P lies on side CD
def point_on_cd (P : ℝ) (c d : ℝ) : Prop := P = d

-- Incircle of triangle PAB touches side AB at Q
def incircle_touches_ab (P A B Q : ℝ) : Prop := -- need refined geometrical definitions

-- To Prove 1: AB ≥ 2BC
def ab_ge_two_bc (a b : ℝ) : Prop := a ≥ 2 * b

-- To Prove 2: AQ * BQ = 1
def aq_times_bq_eq_one (A Q B : ℝ) : Prop := -- need relationship definition

theorem rectangle_area_conditions 
  (a b : ℝ) (h_area : rectangle_area a b)
  (h_P_cd : point_on_cd P C D)
  (h_Q_incircle : incircle_touches_ab P A B Q) : (ab_ge_two_bc a b) ∧ (aq_times_bq_eq_one A Q B) :=
by 
  sorry

end rectangle_area_conditions_l685_685833


namespace max_gcd_value_l685_685324

-- Define the condition and the greatest common divisor
def g : ℕ → ℕ := λ n, Nat.gcd (17 * n + 4) (10 * n + 3)

-- Statement of the theorem
theorem max_gcd_value : ∃ n : ℕ, n > 0 → g n = 11 :=
sorry

end max_gcd_value_l685_685324


namespace find_AX_l685_685381

variables (A B C X : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space X]
variables (AB AX BX CX : ℝ)
variables (angle_bisector_ACB : C → A → X → Prop)

noncomputable def lengths := sorry

def given_conditions : Prop :=
  CX = 25 ∧
  BX = 35 ∧
  AB = 65 ∧
  angle_bisector_ACB C A B

theorem find_AX (h : given_conditions) : AX = 30 :=
  sorry

end find_AX_l685_685381


namespace vasya_has_more_fanta_l685_685195

-- Definitions based on the conditions:
def initial_fanta_vasya (a : ℝ) : ℝ := a
def initial_fanta_petya (a : ℝ) : ℝ := 1.1 * a
def remaining_fanta_vasya (a : ℝ) : ℝ := a * 0.98
def remaining_fanta_petya (a : ℝ) : ℝ := 1.1 * a * 0.89

-- The theorem to prove Vasya has more Fanta left than Petya.
theorem vasya_has_more_fanta (a : ℝ) (h : 0 < a) : remaining_fanta_vasya a > remaining_fanta_petya a := by
  sorry

end vasya_has_more_fanta_l685_685195


namespace sin_minus_cos_eq_pm_sqrt_b_l685_685124

open Real

/-- If θ is an acute angle such that cos(2θ) = b, then sin(θ) - cos(θ) = ±√b. -/
theorem sin_minus_cos_eq_pm_sqrt_b (θ b : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hcos2θ : cos (2 * θ) = b) :
  sin θ - cos θ = sqrt b ∨ sin θ - cos θ = -sqrt b :=
sorry

end sin_minus_cos_eq_pm_sqrt_b_l685_685124


namespace cos_arcsin_proof_l685_685002

noncomputable def cos_arcsin : ℝ :=
  let θ := arcsin (3 / 5) in
  cos θ

theorem cos_arcsin_proof : cos_arcsin = 4 / 5 := 
by
  sorry

end cos_arcsin_proof_l685_685002


namespace minimum_value_of_expression_l685_685406

noncomputable def min_value_expression (x y : ℝ) : ℝ := 
  (x + 1)^2 / (x + 2) + 3 / (x + 2) + y^2 / (y + 1)

theorem minimum_value_of_expression :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → x + y = 2 → min_value_expression x y = 14 / 5 :=
by
  sorry

end minimum_value_of_expression_l685_685406


namespace reflection_sum_l685_685602

theorem reflection_sum (m b : ℝ) :
    (∃ (m b : ℝ), is_reflection (2, 3) (8, 6) (m, b)) → m + b = 12.5 :=
sorry

end reflection_sum_l685_685602


namespace non_similar_120_pointed_stars_count_l685_685020

def euler_totient (n : ℕ) : ℕ :=
  (List.range' 1 n).filter (Nat.coprime n).length

def valid_steps_for_120 (m : ℕ) : Prop :=
  Nat.coprime m 120 ∧ ¬ (3 ∣ m)

theorem non_similar_120_pointed_stars_count : 
  ∃ k, k = 15 ∧ (card ((Finset.filter valid_steps_for_120 (Finset.range 120)) / ~) = k)
/- Proof skipped -/
sorry

end non_similar_120_pointed_stars_count_l685_685020


namespace sum_cos_powers_of_i_l685_685794

theorem sum_cos_powers_of_i :
  ∑ n in Finset.range 21, (complex.I ^ n * real.cos (90 * n * real.pi / 180)) = 11 := 
sorry

end sum_cos_powers_of_i_l685_685794


namespace find_secant_slope_l685_685805

noncomputable def secant_slope (f : ℝ → ℝ) (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem find_secant_slope (Δx : ℝ) (hΔx : Δx = 0.5) :
  secant_slope (λ x, x / (1 - x)) 2 (-2) (2 + Δx) (-2 + (2.5 / (1 - 2.5))) = 2 / 3 :=
by
  unfold secant_slope
  rw [hΔx]
  norm_num
  sorry

end find_secant_slope_l685_685805


namespace find_length_CD_m_plus_n_l685_685510

noncomputable def lengthAB : ℝ := 7
noncomputable def lengthBD : ℝ := 11
noncomputable def lengthBC : ℝ := 9

axiom angle_BAD_ADC : Prop
axiom angle_ABD_BCD : Prop

theorem find_length_CD_m_plus_n :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 67) :=
sorry  -- Proof would be provided here

end find_length_CD_m_plus_n_l685_685510


namespace max_blue_points_l685_685623

-- Define the types for spheres and colors
datatype Sphere : Type
datatype Color : Type
def red : Color := sorry
def green : Color := sorry

-- Define that we have 2016 spheres in total
def number_of_spheres : Nat := 2016

-- Define a function to count the blue points
def count_blue_points (r g : Nat) : Nat := r * g

-- State the main theorem
theorem max_blue_points (r : Nat) (h1 : r + (number_of_spheres - r) = number_of_spheres) 
  (h2 : r ≤ number_of_spheres / 2 ∧ r > 0): 
  count_blue_points r (number_of_spheres - r) ≤ 1016064 := 
sorry

end max_blue_points_l685_685623


namespace length_MF_l685_685304

-- Definitions and assumptions for the problem
variables (p a b : ℝ)
variables (P Q R S M F : ℝ × ℝ)

-- The parabola y^2 = 4px
def parabola (y x : ℝ) : Prop := y^2 = 4 * p * x

-- Focus of the parabola is at (p, 0)
def focus : ℝ × ℝ := (p, 0)

-- Directrix of the parabola x = -p
def directrix (x : ℝ) : Prop := x = -p

-- Distance conditions |PF| = a and |QF| = b
def distance_to_focus_P : Prop := dist P F = a
def distance_to_focus_Q : Prop := dist Q F = b

-- Intersection of the line through the focus and parabola at P and Q
def line_through_focus (l : ℝ × ℝ → Prop) : Prop :=
  l focus ∧ ∃ Q, l Q ∧ parabola Q.2 Q.1

-- Perpendiculars to the directrix
def perpendicular_to_directrix_P (R : ℝ × ℝ) : Prop :=
  R.1 = -p ∧ R.2 = P.2

def perpendicular_to_directrix_Q (S : ℝ × ℝ) : Prop :=
  S.1 = -p ∧ S.2 = Q.2

-- Midpoint M of RS
def midpoint_of_RS (M : ℝ × ℝ) : Prop :=
  M = ((R.1 + S.1) / 2, (R.2 + S.2) / 2)

-- Final proof statement
theorem length_MF :
  ∀ (line : ℝ × ℝ → Prop),
  line_through_focus p line →
  ∀ P Q R S : ℝ × ℝ,
  parabola p P ∧ parabola p Q →
  distance_to_focus_P p P F a →
  distance_to_focus_Q p Q F b →
  perpendicular_to_directrix_P p P R →
  perpendicular_to_directrix_Q p Q S →
  midpoint_of_RS R S M →
  dist M F = real.sqrt (a * b) :=
sorry

end length_MF_l685_685304


namespace find_abc_l685_685615

/-- Define the repeating decimal 0.abababab as a rational number -/
def repeating_decimal_ab (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

/-- Define the repeating decimal 0.abcabcabc as a rational number -/
def repeating_decimal_abc (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / 999

/-- Prove that for digits a, b, and c satisfying the equation involving their repeating decimal forms 
   summing to 35/37, the three-digit number abc is 530. -/
theorem find_abc :
  ∃ (a b c : ℕ), a ∈ finset.range 10 ∧ b ∈ finset.range 10 ∧ c ∈ finset.range 10 ∧
  repeating_decimal_ab a b + repeating_decimal_abc a b c = 35 / 37 ∧
  (100 * a + 10 * b + c = 530) :=
sorry

end find_abc_l685_685615


namespace removed_element_is_1677_l685_685103

def set_of_first_n_integers (n : ℕ) : set ℕ :=
  { m | 1 ≤ m ∧ m ≤ n }

def sum_of_set (s : set ℕ) : ℕ :=
  set.sum s id

theorem removed_element_is_1677 : 
  let M := set_of_first_n_integers 2017 in
  let S := sum_of_set M in
  S = 2035153 →
  ∃ x ∈ M, ∃ k : ℕ, S - x = k ^ 2 ∧ x = 1677 :=
by
  intros M S hS
  simp only [set_of_first_n_integers, sum_of_set] at hS
  sorry

end removed_element_is_1677_l685_685103


namespace nina_money_l685_685556

theorem nina_money (W M : ℕ) (h1 : 6 * W = M) (h2 : 8 * (W - 2) = M) : M = 48 :=
by
  sorry

end nina_money_l685_685556


namespace certain_power_of_prime_p_l685_685487

theorem certain_power_of_prime_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (x : ℕ) :
  Nat.divisors_count (p^x * q^9) = 50 ↔ x = 4 :=
by
  sorry

end certain_power_of_prime_p_l685_685487


namespace lower_percentage_increase_l685_685123

theorem lower_percentage_increase (E P : ℝ) (h1 : 1.26 * E = 693) (h2 : (1 + P) * E = 660) : P = 0.2 := by
  sorry

end lower_percentage_increase_l685_685123


namespace remainder_of_sum_div_8_l685_685765

theorem remainder_of_sum_div_8 :
  let a := 2356789
  let b := 211
  (a + b) % 8 = 0 := 
by 
  sorry

end remainder_of_sum_div_8_l685_685765


namespace smallest_constant_for_triangle_sides_l685_685392

theorem smallest_constant_for_triangle_sides (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_condition : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ N, (∀ a b c, (a + b > c ∧ b + c > a ∧ c + a > b) → (a^2 + b^2) / (a * b) < N) ∧ N = 2 := by
  sorry

end smallest_constant_for_triangle_sides_l685_685392


namespace sequence_geometric_sum_of_first_n_terms_l685_685409

noncomputable def sequence (n : ℕ) : ℕ := sorry

theorem sequence_geometric (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 3 * n) :
    ∀ n, ∃ b, a n + 3 = b * 2 ^ n :=
sorry

theorem sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 3 * n)
    (h2 : ∀ n, a n = 3 * 2^n - 3) :
    ∀ n, T n = 6 + (3 * n - 3) * 2^(n + 1) - 3 * (n * (n + 1)) / 2 :=
sorry

end sequence_geometric_sum_of_first_n_terms_l685_685409


namespace pappus_theorem_l685_685946

variables (A1 A3 A5 A2 A4 A6 : Type*) [affine_space ℝ A1] [affine_space ℝ A3] [affine_space ℝ A5]
[affine_space ℝ A2] [affine_space ℝ A4] [affine_space ℝ A6]
(line1 : affine_subspace ℝ ℝ) (line2 : affine_subspace ℝ ℝ)

-- Conditions: Points A1, A3, A5 are on one line, and A2, A4, A6 are on another line.
axiom A1A3A5_collinear : ∀ (A1 A3 A5 : ℝ), A1 ∈ line1 ∧ A3 ∈ line1 ∧ A5 ∈ line1
axiom A2A4A6_collinear : ∀ (A2 A4 A6 : ℝ), A2 ∈ line2 ∧ A4 ∈ line2 ∧ A6 ∈ line2

-- Pappus' Theorem: The points of intersection are collinear.
theorem pappus_theorem (A1A2 : affine_subspace ℝ ℝ) (A4A5 : affine_subspace ℝ ℝ)
(A2A3 : affine_subspace ℝ ℝ) (A5A6 : affine_subspace ℝ ℝ)
(A3A4 : affine_subspace ℝ ℝ) (A6A1 : affine_subspace ℝ ℝ) :
∃ (L K N : Type*) [_inst_1 : affine_space ℝ L] [_inst_2 : affine_space ℝ K] [_inst_3 : affine_space ℝ N], 
collinear ℝ
  (λ (i : fin 3), 
    if i = 0 then affine_intersect A1A2 A4A5 
    else if i = 1 then affine_intersect A2A3 A5A6 
    else affine_intersect A3A4 A6A1) :=
sorry

end pappus_theorem_l685_685946


namespace range_of_a_l685_685823

variable (a : ℝ)

def p : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ + 1 = 0

def q : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - 2 * a * x + a^2 + 1 ≥ 1

theorem range_of_a : ¬(p a ∨ q a) → -2 < a ∧ a < 0 := by
  sorry

end range_of_a_l685_685823


namespace beta_value_l685_685784

theorem beta_value (α β : ℝ) (h1 : cos α = 1 / 7)
  (h2 : sin α * cos β - cos α * sin β = 3 * real.sqrt 3 / 14)
  (h3 : 0 < β ∧ β < α ∧ α < π / 2) :
  β = π / 3 :=
sorry

end beta_value_l685_685784


namespace find_amount_older_brother_gave_l685_685555

-- Definitions
def mother_gave_older_brother : ℕ := 2800
def mother_gave_younger_brother : ℕ := 1500

def transaction_condition (x : ℕ) : Prop :=
  mother_gave_younger_brother + x = mother_gave_older_brother - x - 360

-- Theorem
theorem find_amount_older_brother_gave :
  ∃ x : ℕ, transaction_condition x := 
begin
  use 470, -- this is the value we found in the solution (470)
  unfold transaction_condition, -- reveals the condition
  sorry -- skipping the proof steps
end

end find_amount_older_brother_gave_l685_685555


namespace insert_words_proof_l685_685157

theorem insert_words_proof :
  ∃ (w1 w2 w3 : String) (w4 w5 : String) (w6 : String),
    w1 = "ВРАЧ" ∧ -- 4 letters for B
    w2 = "МАМА" ∧ -- 4 letters for M
    w3 = "ГОРА" ∧ -- 4 letters for Γ
    w4 = "ЯГОДА" ∧ -- 5 letters for ЯГ
    w5 = "ШКОЛА" ∧ -- 5 letters for ШК
    w6 = "БОРОДА" -- 6 letters for БОР
:= by
  exists ["ВРАЧ", "МАМА", "ГОРА", "ЯГОДА", "ШКОЛА", "БОРОДА"];
  repeat {split}; done;
  sorry

end insert_words_proof_l685_685157


namespace must_be_divisor_of_a_l685_685539

variable (a b c d : ℕ)

-- Conditions as hypotheses
hypothesis h1 : Nat.gcd a b = 24
hypothesis h2 : Nat.gcd b c = 36
hypothesis h3 : Nat.gcd c d = 54
hypothesis h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100

-- Target conclusion
theorem must_be_divisor_of_a : 13 ∣ a :=
by
  sorry

end must_be_divisor_of_a_l685_685539


namespace merchant_total_gross_profit_l685_685305

noncomputable def total_gross_profit : ℝ :=
let 
  purchase_price_jacket := 60,
  purchase_price_jeans := 45,
  purchase_price_shirt := 30,
  markup_jacket := 25 / 100,
  markup_jeans := 30 / 100,
  markup_shirt := 15 / 100,
  discount_jacket := 20 / 100,
  discount_jeans := 10 / 100,
  discount_shirt := 5 / 100,

  selling_price_jacket := purchase_price_jacket * (1 + markup_jacket),
  selling_price_jeans := purchase_price_jeans * (1 + markup_jeans),
  selling_price_shirt := purchase_price_shirt * (1 + markup_shirt),

  final_price_jacket := selling_price_jacket * (1 - discount_jacket),
  final_price_jeans := selling_price_jeans * (1 - discount_jeans),
  final_price_shirt := selling_price_shirt * (1 - discount_shirt),

  profit_jacket := final_price_jacket - purchase_price_jacket,
  profit_jeans := final_price_jeans - purchase_price_jeans,
  profit_shirt := final_price_shirt - purchase_price_shirt,
  total_profit := profit_jacket + profit_jeans + profit_shirt
in
total_profit

theorem merchant_total_gross_profit : total_gross_profit = 10.43 := sorry

end merchant_total_gross_profit_l685_685305


namespace minimal_moves_to_sort_l685_685248

theorem minimal_moves_to_sort (n : ℕ) (nums : list ℕ) (h_len : nums.length = 2 * n) :
  (∃ f : list ℕ → list ℕ, (∀ xs ys, perm xs ys → perm (f xs) ys ∧ (∀ i, f xs i ≤ f xs (i + 1))) ∧ length (f nums) = length nums) →
  (∃ k : ℕ, k ≤ n) :=
sorry

end minimal_moves_to_sort_l685_685248


namespace average_chemistry_mathematics_l685_685619

noncomputable def marks (P C M B : ℝ) : Prop := 
  P + C + M + B = (P + B) + 180 ∧ P = 1.20 * B

theorem average_chemistry_mathematics 
  (P C M B : ℝ) (h : marks P C M B) : (C + M) / 2 = 90 :=
by
  sorry

end average_chemistry_mathematics_l685_685619


namespace projection_of_scaled_vector_l685_685170

-- Define the vectors and given conditions
variables {v w : ℝ^3} -- Treat v and w as vectors in ℝ^3

def proj (w v : ℝ^3) : ℝ^3 := (v.dot w / w.dot w) • w

theorem projection_of_scaled_vector (proj_v_w : proj w v = ⟨2, -1, 4⟩) : 
  proj w (3 • v) = ⟨6, -3, 12⟩ :=
by sorry

end projection_of_scaled_vector_l685_685170


namespace sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l685_685423

variable (α : ℝ)

-- Given conditions
def α_condition (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) : Prop := 
  true

-- Prove the first part: sin(π / 6 + α) = (3 + 4 * real.sqrt 3) / 10
theorem sin_pi_over_6_plus_α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.sin (π / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

-- Prove the second part: cos(π / 3 + 2 * α) = -(7 + 24 * real.sqrt 3) / 50
theorem cos_pi_over_3_plus_2α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.cos (π / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 :=
by
  sorry

end sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l685_685423


namespace factor_difference_of_squares_l685_685371

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l685_685371


namespace sweets_ratio_l685_685253

theorem sweets_ratio (number_orange_sweets : ℕ) (number_grape_sweets : ℕ) (max_sweets_per_tray : ℕ)
  (h1 : number_orange_sweets = 36) (h2 : number_grape_sweets = 44) (h3 : max_sweets_per_tray = 4) :
  (number_orange_sweets / max_sweets_per_tray) / (number_grape_sweets / max_sweets_per_tray) = 9 / 11 :=
by
  sorry

end sweets_ratio_l685_685253


namespace solve_for_f_8_l685_685060

noncomputable def f (x : ℝ) : ℝ := (Real.logb 2 x)

theorem solve_for_f_8 {x : ℝ} (h : f (x^3) = Real.logb 2 x) : f 8 = 1 :=
by
sorry

end solve_for_f_8_l685_685060


namespace arithmetic_sequence_sum_l685_685483

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end arithmetic_sequence_sum_l685_685483


namespace exists_right_triangle_same_color_l685_685258

open Set

noncomputable theory

-- Define an equilateral triangle with vertices A, B, and C.
variables {A B C : Point}

-- Define that every point on the edges of the equilateral triangle 
-- is colored either red or blue.
variable (color : Point → Prop)
variable red blue : Prop

axiom color_disjoint : ∀ p : Point, ¬ (color p = red ∧ color p = blue)
axiom color_complete : ∀ p : Point, color p = red ∨ color p = blue

-- Given statement
theorem exists_right_triangle_same_color
    (H : is_equilateral_triangle A B C) :
    ∃ P Q R, (P ∈ segment A B) ∧ (Q ∈ segment B C) ∧ (R ∈ segment C A) ∧ 
             (is_right_triangle P Q R) ∧ 
             (color P = color Q ∧ color Q = color R) := 
sorry

end exists_right_triangle_same_color_l685_685258


namespace first_question_second_question_l685_685072

noncomputable variable {a_n : ℕ → ℕ} 

def recurrence_relation (p : ℕ) (a : ℕ → ℕ) : ℕ → ℕ
| 0   := a 0
| (n+1) := if a n ≤ p then 2 * a n else 2 * a n - 6

def sequence_1 := recurrence_relation 90 (λ n, if n = 0 then 3 else a_n n)
def sequence_2 := recurrence_relation 18 (λ n, if n = 0 then a_n 1 else a_n n)

-- Theorem for first question:
theorem first_question (a_2_eq : a_n 1 = 6) : 
  a_n 0 = 3 ∧ 
  a_n 1 = 6 ∧ 
  a_n 2 = 12 ∧ 
  a_n 3 = 24 ∧ 
  a_n 4 = 48 ∧ 
  a_n 5 = 96 ∧ 
  a_n 6 = 186 := 
sorry

-- Theorem for second question:
theorem second_question (h : ∃ n, a_n n % 3 = 0) :
  ∀ n, a_n n % 3 = 0 := 
sorry

end first_question_second_question_l685_685072


namespace find_angle_C_find_area_l685_685132

-- Define the triangle and given conditions
variables {A B C : ℝ} -- The angles of the triangle
variables {a b c : ℝ} -- The sides opposite to angles A, B, and C respectively

-- Given condition for part (I)
axiom sin_cos_condition : sin ((π / 3) - C) + cos (C - (π / 6)) = sqrt 3 / 2

-- Problem (I)
theorem find_angle_C : C = π / 3 :=
by
  -- Since we are not providing the proof, we skip it with "sorry".
  sorry

-- Given conditions for part (II)
axiom sides_conditions : c = 2 * sqrt 3 ∧ sin A = 2 * sin B

-- Problem (II)
theorem find_area : (1 / 2) * a * b * sin C = 2 * sqrt 3 :=
by
  -- Since we are not providing the proof, we skip it with "sorry".
  sorry

end find_angle_C_find_area_l685_685132


namespace computer_sequences_l685_685465

theorem computer_sequences : 
  ∃ (S : Finset (List Char)), S.card = 180 ∧ 
  ∀ s ∈ S, 
    s.head = 'M' ∧ 
    s.length = 4 ∧ 
    'B' ∉ s ∧ 
    s.nodup :=
begin
  sorry
end

end computer_sequences_l685_685465


namespace acute_angle_lights_at_9_35_20_pm_l685_685749

noncomputable def angle_hour_hand (hours minutes seconds : ℕ) : ℝ :=
  30.0 * hours + 0.5 * minutes + (1.0 / 120) * seconds

noncomputable def angle_minute_hand (minutes seconds : ℕ) : ℝ :=
  6.0 * minutes + 0.1 * seconds

noncomputable def angle_between_hands (hour_angle minute_angle : ℝ) : ℝ :=
  abs (hour_angle - minute_angle)

noncomputable def lights_within_acute_angle (angle : ℝ) : ℕ :=
  ⌊angle / 6.0⌋

theorem acute_angle_lights_at_9_35_20_pm :
  lights_within_acute_angle 
    (angle_between_hands 
      (angle_hour_hand 9 35 20) 
      (angle_minute_hand 35 20)) 
  = 12 := 
by 
  sorry

end acute_angle_lights_at_9_35_20_pm_l685_685749


namespace ten_percent_of_number_l685_685299

theorem ten_percent_of_number (x : ℝ)
  (h : x - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 3.325 :=
sorry

end ten_percent_of_number_l685_685299


namespace exists_point_with_given_mean_distance_mean_distance_fixed_at_half_l685_685562

variables (n : ℕ) (x : fin n → ℝ)

noncomputable def mean_distance (c : ℝ) : ℝ :=
  (1 / n) * (finset.univ.sum (λ i, |c - x i|))

theorem exists_point_with_given_mean_distance :
  ∃ c ∈ set.Icc 0 1, mean_distance n x c = (1 / 2) :=
begin
  sorry
end

theorem mean_distance_fixed_at_half :
  ¬ ∃ k ≠ (1 / 2), ∃ c ∈ set.Icc 0 1, mean_distance n x c = k :=
begin
  sorry
end

end exists_point_with_given_mean_distance_mean_distance_fixed_at_half_l685_685562


namespace bus_problem_l685_685295

theorem bus_problem : ∀ before_stop after_stop : ℕ, before_stop = 41 → after_stop = 18 → before_stop - after_stop = 23 :=
by
  intros before_stop after_stop h_before h_after
  sorry

end bus_problem_l685_685295


namespace find_lambda_perpendicular_l685_685884

def vector (α β : Type) [Add α] [Mul α] [Add β] [Mul β] := Prod α β

def dot_product {α : Type} [Add α] [Mul α] (v w : vector α α) : α :=
v.fst * w.fst + v.snd * w.snd

def perpendicular (α β : Type) [Add α] [Mul α] [Add β] [Mul β] (v w : vector α α) : Prop :=
dot_product v w = 0

theorem find_lambda_perpendicular :
  let α := (1, -3) in
  let β := (4, -2) in
  perpendicular ℝ ℝ (λ x : ℝ, (x * α.fst + β.fst, x * α.snd + β.snd)) α 
  := sorry

end find_lambda_perpendicular_l685_685884


namespace no_integer_solution_for_conditions_l685_685475

theorem no_integer_solution_for_conditions :
  ¬∃ (x : ℤ), 
    (18 + x = 2 * (5 + x)) ∧
    (18 + x = 3 * (2 + x)) ∧
    ((18 + x) + (5 + x) + (2 + x) = 50) :=
by
  sorry

end no_integer_solution_for_conditions_l685_685475


namespace count_two_digit_integers_l685_685114

theorem count_two_digit_integers (n : ℕ) :
  (∃ n : ℕ, 10 ≤ 7 * n + 5 ∧ 7 * n + 5 < 100) ↔ 13 := 
sorry

end count_two_digit_integers_l685_685114


namespace Nara_height_is_1_69_l685_685206

-- Definitions of the conditions
def SangheonHeight : ℝ := 1.56
def ChihoHeight : ℝ := SangheonHeight - 0.14
def NaraHeight : ℝ := ChihoHeight + 0.27

-- The statement to prove
theorem Nara_height_is_1_69 : NaraHeight = 1.69 :=
by {
  sorry
}

end Nara_height_is_1_69_l685_685206


namespace pages_per_chapter_l685_685053

theorem pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ)
  (h1 : total_pages = 555)
  (h2 : num_chapters = 5) :
  (total_pages / num_chapters) = 111 :=
by
  rw [h1, h2]
  sorry

end pages_per_chapter_l685_685053


namespace sum_of_adjacent_cells_multiple_of_4_l685_685951

theorem sum_of_adjacent_cells_multiple_of_4 :
  ∃ (i j : ℕ) (a b : ℕ) (H₁ : i < 22) (H₂ : j < 22),
    let grid (i j : ℕ) : ℕ := -- define the function for grid indexing
      ((i * 22) + j + 1 : ℕ)
    ∃ (i1 j1 : ℕ) (H₁₁ : i1 = i ∨ i1 = i + 1 ∨ i1 = i - 1)
                   (H₁₂ : j1 = j ∨ j1 = j + 1 ∨ j1 = j - 1),
      a = grid i j ∧ b = grid i1 j1 ∧ (a + b) % 4 = 0 := sorry

end sum_of_adjacent_cells_multiple_of_4_l685_685951


namespace area_ratio_S2_l685_685168

def S1' (x y : ℝ) : Prop :=
  log 10 (3 + x^2 + y^2) ≤ 1 + log 10 (x + 2 * y)

def S2' (x y : ℝ) : Prop :=
  log 10 (4 + x^2 + y^2) ≤ 2 + log 10 (x + 2 * y)

theorem area_ratio_S2'_to_S1' : 
  (let area_S1' := π * 122 in 
   let area_S2' := π * 12804 in
   area_S2' / area_S1' = 105) :=
sorry

end area_ratio_S2_l685_685168


namespace amy_earnings_l685_685740

theorem amy_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (tips : ℝ) (wage_per_hour : hourly_wage = 2) (hours_worked_equals : hours_worked = 7) (tips_equals : tips = 9) : 
  hourly_wage * hours_worked + tips = 23 := by
  rw [wage_per_hour, hours_worked_equals, tips_equals]
  norm_num
  sorry

end amy_earnings_l685_685740


namespace initial_population_of_town_l685_685230

theorem initial_population_of_town 
  (final_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (initial_population : ℝ) 
  (h : final_population = initial_population * (1 + growth_rate) ^ years) : 
  initial_population = 297500 / (1 + 0.07) ^ 10 :=
by
  sorry

end initial_population_of_town_l685_685230


namespace ratio_rectangle_to_semicircles_area_l685_685708

theorem ratio_rectangle_to_semicircles_area (AB AD : ℝ) (h1 : AB = 40) (h2 : AD / AB = 3 / 2) : 
  (AB * AD) / (2 * (π * (AB / 2)^2)) = 6 / π :=
by
  -- here we process the proof
  sorry

end ratio_rectangle_to_semicircles_area_l685_685708


namespace inequality_solution_l685_685786

theorem inequality_solution (x : ℝ) :
  (x ^ 2 - 9) / (x ^ 3 - 1) > 0 ↔ x ∈ Set.Ioo (-(3 : ℝ)) ∞ ∪ Set.Ioo (-(3 : ℝ)) 1 ∪ Set.Ioo 3 ∞ :=
by
  sorry

end inequality_solution_l685_685786


namespace shortest_altitude_of_right_triangle_l685_685235

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l685_685235


namespace partitions_distinct_eq_partitions_odd_l685_685964

noncomputable def partitions_distinct (n : ℕ) : Finset (Finset ℕ) :=
  {s : Finset ℕ | s.sum = n ∧ s.card = s.toList.eraseDups.length}

noncomputable def partitions_odd (n : ℕ) : Finset (Finset ℕ) :=
  {s : Finset ℕ | s.sum = n ∧ (∀ x ∈ s, x % 2 = 1)}

theorem partitions_distinct_eq_partitions_odd (n : ℕ) :
  (partitions_distinct n).card = (partitions_odd n).card :=
sorry

end partitions_distinct_eq_partitions_odd_l685_685964


namespace problem1_problem2_l685_685334

def E1 : ℝ := 0.027^(-1/3) - (1/7)^(-2) + (2 + 7/9)^(1/2) - (real.sqrt 2 - 1)^0
def E2 : ℝ := (1/2) * real.log10 25 + real.log10 2 - real.log10 (real.sqrt 0.1) - (real.log9 2) * (real.log3 2)

theorem problem1 : E1 = -45 := by
  sorry

theorem problem2 : E2 = -1/2 := by
  sorry

end problem1_problem2_l685_685334


namespace clothing_factory_exceeded_tasks_l685_685707

theorem clothing_factory_exceeded_tasks :
  let first_half := (2 : ℚ) / 3
  let second_half := (3 : ℚ) / 5
  first_half + second_half - 1 = (4 : ℚ) / 15 :=
by
  sorry

end clothing_factory_exceeded_tasks_l685_685707


namespace function_range_log3_l685_685621

theorem function_range_log3 : 
  let f : ℝ → ℝ := λ x, log 3 (x^2 - 2 * x + 10) in
  set.range f = set.Ici 2 :=
by
  sorry

end function_range_log3_l685_685621


namespace infinite_non_negative_real_numbers_l685_685172

theorem infinite_non_negative_real_numbers
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2)^2)
  (x_0 : ℝ)
  (h_x0_nonneg : x_0 ≥ 0)
  (x_seq : ℕ → ℝ)
  (h_x0 : x_seq 0 = x_0)
  (h_seq : ∀ n, x_seq (n + 1) = f (x_seq n)) :
  ∃ᶠ x_0 in 𝓝[Ici 0] 0, {x | set.finite {x_seq n | n ∈ ℕ }} :=
sorry

end infinite_non_negative_real_numbers_l685_685172


namespace completing_the_square_solution_correct_l685_685640

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l685_685640


namespace mass_percentage_Br_in_AlBr3_is_89_89_l685_685043

-- Definitions based on conditions
def molar_mass_Al : ℝ := 26.98
def molar_mass_Br : ℝ := 79.90
def num_Br_atoms_in_AlBr3 : ℕ := 3

-- Calculate the molar mass of AlBr3 using the conditions
def molar_mass_AlBr3 : ℝ := molar_mass_Al + num_Br_atoms_in_AlBr3 * molar_mass_Br

-- Calculate the mass percentage of Br in AlBr3
def mass_percentage_Br_in_AlBr3 : ℝ := (num_Br_atoms_in_AlBr3 * molar_mass_Br / molar_mass_AlBr3) * 100

-- Proof problem statement
theorem mass_percentage_Br_in_AlBr3_is_89_89 :
  mass_percentage_Br_in_AlBr3 = 89.89 := by
  sorry

end mass_percentage_Br_in_AlBr3_is_89_89_l685_685043


namespace parabola_focus_l685_685869

/-- Given the parabola y^2 = 2 * p * x (p > 0) with focus F, let M (2, y0) be a point on the parabola 
such that |MO| = |MF|, where O is the origin. Determine p = 8. -/
theorem parabola_focus (p : ℝ) (y0 : ℝ) (hp : 0 < p)
    (hM_on_parabola : y0^2 = 2 * p * 2)
    (h_eq_dist : dist ⟨2, y0⟩ ⟨0, 0⟩ = dist ⟨2, y0⟩ ⟨p/2, 0⟩) :
    p = 8 :=
sorry

end parabola_focus_l685_685869


namespace simplify_sqrt_expression_l685_685764

variable (q : ℝ)

noncomputable def simplify_expression := (sqrt (45 * q) * sqrt (15 * q) * sqrt (10 * q))

theorem simplify_sqrt_expression (hq : 0 < q) : 
  simplify_expression q = 30 * q * sqrt (15 * q) :=
sorry

end simplify_sqrt_expression_l685_685764


namespace simplify_trig_expression_l685_685284

theorem simplify_trig_expression (α : ℝ) : 
  (cos(2 * α) ^ 4 - 6 * (cos(2 * α) ^ 2) * (sin(2 * α) ^ 2) + sin(2 * α) ^ 4) = cos(8 * α) := 
by 
  sorry

end simplify_trig_expression_l685_685284


namespace roots_equal_iff_m_eq_neg_half_l685_685912

theorem roots_equal_iff_m_eq_neg_half (x m : ℝ) :
  (∀ x, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) ∧
  discriminant (1 : ℝ) (-1 : ℝ) (-(m * m + m)) = 0 ↔ m = -1/2 :=
by sorry

end roots_equal_iff_m_eq_neg_half_l685_685912


namespace height_of_zions_house_l685_685281

-- Definitions given in the problem
variable (total_area : ℝ) (base : ℝ)
variable (num_houses : ℕ)

-- The total area of the triangular houses is 1200 cm²
def total_area_triangular_houses := total_area = 1200

-- The base measurement of Zion's house is 40 cm
def base_measurement := base = 40

-- The number of similar shaped houses is 3
def number_of_houses := num_houses = 3

-- Define the height of Zion's house (which we are going to prove)
def height : ℝ := 20

-- Proof statement: Given the conditions, the height of Zion's house is 20 cm
theorem height_of_zions_house 
  (h_total_area : total_area_triangular_houses total_area)
  (h_base : base_measurement base)
  (h_num_houses : number_of_houses num_houses) : 
  base * height / 2 * num_houses = total_area := 
sorry

end height_of_zions_house_l685_685281


namespace birches_count_l685_685142

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l685_685142


namespace evan_tax_deduction_per_hour_in_cents_l685_685368

theorem evan_tax_deduction_per_hour_in_cents :
  ∀ (hourly_wage_dollars : ℕ), hourly_wage_dollars = 25 →
  ∀ (tax_rate : ℚ), tax_rate = 2.4 / 100 →
  (hourly_wage_dollars * 100 * tax_rate) = 60 :=
by
  intros hourly_wage_dollars h_wage_eq tax_rate t_rate_eq
  have h_wage_in_cents : hourly_wage_dollars * 100 = 2500 :=
    by rw [h_wage_eq]; norm_num
  have h_tax : (2500 * tax_rate) = 60 := by rw [t_rate_eq]; norm_num
  rw [←h_wage_in_cents] at h_tax
  exact h_tax

end evan_tax_deduction_per_hour_in_cents_l685_685368


namespace simplify_expr1_simplify_expr2_l685_685577

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l685_685577


namespace min_value_of_F_on_negative_half_l685_685430

variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def F (x : ℝ) := a * f x + b * g x + 2

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_value_of_F_on_negative_half
  (h_f : is_odd f) (h_g : is_odd g)
  (max_F_positive_half : ∃ x, x > 0 ∧ F f g a b x = 5) :
  ∃ x, x < 0 ∧ F f g a b x = -3 :=
by {
  sorry
}

end min_value_of_F_on_negative_half_l685_685430


namespace sum_and_product_of_divisors_l685_685607

variable (a : ℕ) (p : ℕ) [Fact (Nat.prime p)]

theorem sum_and_product_of_divisors
  (h_prime_power : a = p ^ 102)
  (h_divisors : (Nat.divisors a).length = 103) :
  (∑ d in Nat.divisors a, d = (a * (a ^ (1 / 102)) - 1) / ((a ^ (1 / 102)) - 1)) ∧
  (∏ d in Nat.divisors a, d = (a ^ 103) ^ (1 / 2)) := sorry

end sum_and_product_of_divisors_l685_685607


namespace exists_same_color_ratios_l685_685738

-- Definition of coloring function.
def coloring : ℕ → Fin 2 := sorry

-- Definition of the problem: there exist A, B, C such that A : C = C : B,
-- and A, B, C are of same color.
theorem exists_same_color_ratios :
  ∃ A B C : ℕ, coloring A = coloring B ∧ coloring B = coloring C ∧ 
  (A : ℚ) / C = (C : ℚ) / B := 
sorry

end exists_same_color_ratios_l685_685738


namespace rational_squares_of_numbers_l685_685586
-- First, import the necessary libraries and modules.

-- Define the main theorem.
theorem rational_squares_of_numbers :
  ∀ (S : Finset ℝ), S.card = 10 →
  (∀ (a b ∈ S), a ≠ b → (∃ q : ℚ, a + b = q ∨ a * b = q)) →
  (∀ (x ∈ S), ∃ q : ℚ, x^2 = q) :=
begin
  -- Placeholder for the proof.
  sorry
end

end rational_squares_of_numbers_l685_685586


namespace smallest_square_eq_121_l685_685393

theorem smallest_square_eq_121 :
  ∃ (n : ℕ), (∃ (k : ℕ), n = k^2) ∧ 100 ≤ n ∧ n < 1000 ∧ (n % 10 ≠ 0) ∧ 
  (∃ (m : ℕ), (n / 100) = m^2) ∧ 
  ∀ (n' : ℕ), (∃ (k' : ℕ), n' = k'^2) ∧ 100 ≤ n' ∧ n' < 1000 ∧ (n' % 10 ≠ 0) ∧ 
  (∃ (m' : ℕ), (n' / 100) = m'^2) → n ≤ n' :=
begin
  sorry
end

end smallest_square_eq_121_l685_685393


namespace increase_percentage_when_selfcheckout_broken_l685_685980

-- The problem conditions as variable definitions and declarations
def normal_complaints : ℕ := 120
def short_staffed_increase : ℚ := 1 / 3
def short_staffed_complaints : ℕ := normal_complaints + (normal_complaints / 3)
def total_complaints_three_days : ℕ := 576
def days : ℕ := 3
def both_conditions_complaints : ℕ := total_complaints_three_days / days

-- The theorem that we need to prove
theorem increase_percentage_when_selfcheckout_broken : 
  (both_conditions_complaints - short_staffed_complaints) * 100 / short_staffed_complaints = 20 := 
by
  -- This line sets up that the conclusion is true
  sorry

end increase_percentage_when_selfcheckout_broken_l685_685980


namespace triangle_congruence_case1_triangle_congruence_case2_l685_685563

-- Define the first version of the problem
theorem triangle_congruence_case1 {A B C D A' B' C' D' : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace A'] [MetricSpace B'] [MetricSpace C'] [MetricSpace D']
  (AB_eq : dist A B = dist A' B')
  (BC_eq : dist B C = dist B' C')
  (AD_eq : dist A D = dist A' D') :
  congruent {A B C} {A' B' C'} :=
sorry

-- Define the second version of the problem
theorem triangle_congruence_case2 {A B C D A' B' C' D' : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace A'] [MetricSpace B'] [MetricSpace C'] [MetricSpace D']
  (AB_eq : dist A B = dist A' B')
  (AC_eq : dist A C = dist A' C')
  (AD_eq : dist A D = dist A' D') :
  congruent {A B C} {A' B' C'} :=
sorry

end triangle_congruence_case1_triangle_congruence_case2_l685_685563


namespace geom_seq_a5_l685_685244

noncomputable def S3 (a1 q : ℚ) : ℚ := a1 + a1 * q^2
noncomputable def a (a1 q : ℚ) (n : ℕ) : ℚ := a1 * q^(n - 1)

theorem geom_seq_a5 (a1 q : ℚ) (hS3 : S3 a1 q = 5 * a1) (ha7 : a a1 q 7 = 2) :
  a a1 q 5 = 1 / 2 :=
by
  sorry

end geom_seq_a5_l685_685244


namespace max_principals_in_10_years_max_principals_is_4_l685_685983

theorem max_principals_in_10_years (serve_term : ℕ) (total_years : ℕ) (no_overlap : Prop) : ℕ :=
  sorry

def maximum_principals (serve_term : ℕ) (total_years : ℕ) (no_overlap : Prop) : ℕ :=
  if total_years / serve_term >= 4 then 4 else total_years / serve_term

theorem max_principals_is_4 : maximum_principals 4 10 (λ p1 p2 : ℕ, p1 ≠ p2) = 4 := 
  sorry

end max_principals_in_10_years_max_principals_is_4_l685_685983


namespace total_accessories_correct_l685_685921

-- Definitions
def dresses_first_period := 10 * 4
def dresses_second_period := 3 * 5
def total_dresses := dresses_first_period + dresses_second_period
def accessories_per_dress := 3 + 2 + 1
def total_accessories := total_dresses * accessories_per_dress

-- Theorem statement
theorem total_accessories_correct : total_accessories = 330 := by
  sorry

end total_accessories_correct_l685_685921


namespace circle_center_polar_coords_l685_685858

theorem circle_center_polar_coords :
  (∃ θ : ℝ, (x = 1 + cos θ) ∧ (y = 1 + sin θ)) →
  (∃ r θ, (r, θ) = (sqrt 2, π / 4)) :=
by
  sorry

end circle_center_polar_coords_l685_685858


namespace coefficient_x3_in_binomial_expansion_l685_685435

theorem coefficient_x3_in_binomial_expansion : 
  let f := (2 * x + sqrt x) ^ 5 in 
  (coeff (expand f) 3) = 10 :=
by 
  sorry

end coefficient_x3_in_binomial_expansion_l685_685435


namespace united_airlines_discount_l685_685767

theorem united_airlines_discount :
  ∀ (delta_price original_price_u discount_delta discount_u saved_amount cheapest_price: ℝ),
    delta_price = 850 →
    original_price_u = 1100 →
    discount_delta = 0.20 →
    saved_amount = 90 →
    cheapest_price = delta_price * (1 - discount_delta) - saved_amount →
    discount_u = (original_price_u - cheapest_price) / original_price_u →
    discount_u = 0.4636363636 :=
by
  intros delta_price original_price_u discount_delta discount_u saved_amount cheapest_price δeq ueq deq saeq cpeq dueq
  -- Placeholder for the actual proof steps
  sorry

end united_airlines_discount_l685_685767


namespace limes_left_correct_l685_685954

variables (Mike_picked : ℝ) (Alyssa_ate : ℝ)

noncomputable def limes_left (Mike_picked Alyssa_ate : ℝ) : ℝ := Mike_picked - Alyssa_ate

theorem limes_left_correct : 
  Mike_picked = 32.0 → Alyssa_ate = 25.0 → 
  limes_left Mike_picked Alyssa_ate = 7.0 := 
by
  intros hMike hAlyssa
  unfold limes_left
  rw [hMike, hAlyssa]
  norm_num
  exact rfl

end limes_left_correct_l685_685954


namespace mod_exponent_problem_l685_685669

theorem mod_exponent_problem : (11 ^ 2023) % 100 = 31 := by
  sorry

end mod_exponent_problem_l685_685669


namespace rum_consumed_earlier_day_l685_685961

-- Defining our assumptions as Lean statements
def sally_gave_rum : ℕ := 10  -- Sally gave 10 oz of rum
def can_consume_maximum (x : ℕ) : ℕ := 3 * x  -- Maximum consumption is 3 times x
def rum_after_pancakes (given_rum : ℕ) : ℕ := given_rum + 8  -- 10 oz on pancakes + 8 oz remaining

-- The proof statement
theorem rum_consumed_earlier_day :
  (∀ (given_rum : ℕ), given_rum = sally_gave_rum → can_consume_maximum sally_gave_rum = 30 ∧
  rum_after_pancakes given_rum = 18) →
  ∃ earlier_rum : ℕ, earlier_rum = 30 - 18 :=
by
  intros h
  have hc : can_consume_maximum sally_gave_rum = 30 := by exact h 10 rfl.left
  have hr : rum_after_pancakes sally_gave_rum = 18 := by exact h 10 rfl.right
  use 12
  exact Eq.refl 12

end rum_consumed_earlier_day_l685_685961


namespace flea_angle_rational_l685_685685

theorem flea_angle_rational (α : ℝ) (l₁ l₂ : set ℝ) (flea_jump : ℕ → ℝ) :
  -- Conditions
  (∀ n, flea_jump n ∈ l₁ ∪ l₂) ∧ 
  (∀ n, flea_jump (n + 1) ≠ flea_jump n) ∧ 
  (∃ n, flea_jump n = flea_jump 0) →
  -- Conclusion
  ∃ q : ℚ, α = q * (180 : ℝ) :=
sorry

end flea_angle_rational_l685_685685


namespace triangle_right_if_sin_squared_l685_685522

theorem triangle_right_if_sin_squared :
  ∀ (A B C : ℝ), sin A ^ 2 = sin B ^ 2 + sin C ^ 2 → angle_is_right A B C :=
by
  sorry

end triangle_right_if_sin_squared_l685_685522


namespace work_completion_time_l685_685701

theorem work_completion_time (hA : nat := 12) (hB : nat := 14) (hC : nat := 16) : 
  let total_rate := (1 / hA.toReal) + (1 / hB.toReal) + (1 / hC.toReal)
  in (1 / total_rate).toReal ≈ 4.6027 :=
by
  let rate_A := (1 : ℚ) / hA
  let rate_B := (1 : ℚ) / hB
  let rate_C := (1 : ℚ) / hC
  let combined_rate := rate_A + rate_B + rate_C
  let combined_time := (1 : ℚ) / combined_rate
  have : (combined_time.toReal - 4.6027).abs < 0.0001 := sorry
  exact real.eq_of_abs_sub_lt (combined_time.toReal) 4.6027 this

end work_completion_time_l685_685701


namespace intersection_of_A_and_B_l685_685845

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end intersection_of_A_and_B_l685_685845


namespace hundredth_number_is_524_l685_685745


open Nat

-- Define the set as described
def exp_triple_set := { n : ℕ | ∃ (x y z : ℕ), x < y ∧ y < z ∧ n = 2^x + 2^y + 2^z }

-- The theorem proving the 100th element in the ordered set is 524
theorem hundredth_number_is_524 : 
  finset.val (finset.sort (≤) (finset.filter (λ n, n ∈ exp_triple_set) (finset.range 9999))) 99 = 524 := 
begin
  sorry
end

end hundredth_number_is_524_l685_685745


namespace find_a_if_circle_l685_685426

noncomputable def curve_eq (a x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

def is_circle_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, curve_eq a x y = 0 → (∃ k : ℝ, curve_eq a x y = k * (x^2 + y^2))

theorem find_a_if_circle :
  (∀ a : ℝ, is_circle_condition a → a = -1) :=
by
  sorry

end find_a_if_circle_l685_685426


namespace quiz_competition_l685_685204

open Set

-- Define the set of participants
def participants : Set String := {"Rita", "Sam", "Tom", "Victor", "Wendy", "Zara"}

-- Define a function to count valid podium outcomes
noncomputable def countPodiumOutcomes (p : Set String) : Nat :=
  let first_place := p
  let second_place := first_place \ {first | first ∈ p}
  let third_place := second_place \ {second | second ∈ p ∨ second = "Rita"}
  first_place.card * second_place.card * third_place.card

-- Ensure the count matches the expected number of outcomes
theorem quiz_competition : countPodiumOutcomes participants = 120 := by sorry

end quiz_competition_l685_685204


namespace find_smallest_n_l685_685872

-- Sequence definition
def a : ℕ → ℝ
| 0     := 9
| (n+1) := (4 - a n) / 3

-- Sum of first n terms of sequence a
def S : ℕ → ℝ 
| 0     := a 0
| (n+1) := S n + a (n+1)

-- Prove that the smallest positive integer n satisfying the inequality is 7
theorem find_smallest_n : 
  ∃ (n : ℕ), 0 < n ∧ | S n - n - 6 | < 1/125 ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬ | S m - m - 6 | < 1/125 :=
sorry

end find_smallest_n_l685_685872


namespace smallest_c_no_9_in_range_l685_685262

/-- 
  If 9 is not in the range of the quadratic function y = x^2 + cx + 18,
  the smallest integer value of c is -5.
-/
theorem smallest_c_no_9_in_range : 
  ∀ (c : ℤ), (∀ x : ℝ, x^2 + (c : ℝ) * x + 18 ≠ 9) → c = -5 :=
begin
  sorry
end

end smallest_c_no_9_in_range_l685_685262


namespace mean_age_is_10_l685_685587

def ages : List ℤ := [7, 7, 7, 14, 15]

theorem mean_age_is_10 : (List.sum ages : ℤ) / (ages.length : ℤ) = 10 := by
-- sorry placeholder for the actual proof
sorry

end mean_age_is_10_l685_685587


namespace find_CD_l685_685918

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given points and vectors
variables (A B C D : V)
variables (AB DB CA CB CD : V)

-- Condition
def condition := AB = 4 • DB

-- Definition of corresponding vectors
def AB_def : AB = CB - CA := sorry
def DB_def : DB = CB - CD := sorry

-- Proof statement
theorem find_CD (h : condition) (hab : AB_def) (hdb : DB_def) :
  CD = 1/4 • CA + 3/4 • CB := sorry

end find_CD_l685_685918


namespace arithmetic_sequence_general_formula_l685_685503

noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 3

theorem arithmetic_sequence_general_formula
    (a : ℕ → ℤ)
    (h1 : (a 2 + a 6) / 2 = 5)
    (h2 : (a 3 + a 7) / 2 = 7) :
  arithmetic_sequence a :=
by
  sorry

end arithmetic_sequence_general_formula_l685_685503


namespace number_of_math_books_l685_685259

theorem number_of_math_books (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end number_of_math_books_l685_685259


namespace original_cards_proof_l685_685567

-- Definitions corresponding to the conditions
def sasha_added := 48
def karen_fraction := 1 / 6
def total_after_karen := 83
def karen_cards (added : ℕ) (fraction : ℝ) : ℕ :=
  (added * fraction).to_nat
def original_cards (total : ℕ) (removed : ℕ) : ℕ :=
  total + removed - sasha_added

-- Prove that the original number of cards is 75
theorem original_cards_proof : original_cards total_after_karen (karen_cards sasha_added karen_fraction) = 75 := by
  -- Proof can be filled later
  sorry

end original_cards_proof_l685_685567


namespace trapezoid_area_l685_685744

-- Define the problem context
variables (b : ℝ) (theta : ℝ)
def is_isosceles (a c : ℝ) := 
  a = c

def is_inscribed (b : ℝ) := 
  ∀ (r : ℝ), r > 0

theorem trapezoid_area
  (h1 : is_isosceles 18 18)
  (h2 : is_inscribed 18)
  (h3 : \arccos(0.6) = theta)
  : trapezoid_area 18 theta = 101.25 :=
sorry

end trapezoid_area_l685_685744


namespace root_in_interval_k_eq_2_l685_685095

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem root_in_interval_k_eq_2
  (k : ℤ)
  (h1 : 0 < f 2)
  (h2 : Real.log 2 + 2 * 2 - 5 < 0)
  (h3 : Real.log 3 + 2 * 3 - 5 > 0) 
  (h4 : f (k : ℝ) * f (k + 1 : ℝ) < 0) :
  k = 2 := 
sorry

end root_in_interval_k_eq_2_l685_685095


namespace curve_length_eq_four_l685_685801

noncomputable def length_of_curve : ℝ :=
  ∫ θ in 0..π, sqrt ((-sin θ)^2 + (1 + cos θ)^2)

theorem curve_length_eq_four : length_of_curve = 4 :=
by
  sorry

end curve_length_eq_four_l685_685801


namespace circumcircle_diameter_l685_685155

theorem circumcircle_diameter (a b c : ℝ) (A B C : ℝ)
  (h1 : a = 2)
  (h2 : b^2 + c^2 = a^2 + b * c)
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h4 : sin A ≠ 0) : 2 * (a / (Real.sin A)) = (4 * Real.sqrt 3) / 3 := 
sorry

end circumcircle_diameter_l685_685155


namespace expression_divisible_by_19_l685_685560

theorem expression_divisible_by_19 (n : ℕ) (h : n > 0) : 
  19 ∣ (5^(2*n - 1) + 3^(n - 2) * 2^(n - 1)) := 
by 
  sorry

end expression_divisible_by_19_l685_685560


namespace smallest_integer_cube_root_condition_l685_685941

-- Mathematically equivalent problem rewritten in Lean 4
theorem smallest_integer_cube_root_condition :
  ∃ m r, m ∈ ℕ ∧ r ∈ ℝ ∧ r > 0 ∧ r < 1 / 10000 ∧ m = (58 + r)^3 :=
begin
  sorry
end

end smallest_integer_cube_root_condition_l685_685941


namespace size_of_angle_A_area_of_Triangle_ABC_l685_685133

def sides_and_angles_in_Triangle_ABC (a b c A B C : ℝ) : Prop := 
  B ≠ 0 ∧ 2 * b - c = a * cos C

theorem size_of_angle_A 
  (a b c A B C : ℝ)
  (h1 : sides_and_angles_in_Triangle_ABC a b c A B C)
  (h2 : B ≠ 0)
  (h3 : (2 * b - c) * cos A = a * cos C)
: A = π / 3 :=
sorry

theorem area_of_Triangle_ABC 
  (a b c A : ℝ)
  (h1 : a = 2)
  (h2 : b + c = 4)
  (h3 : sides_and_angles_in_Triangle_ABC a b c A)
  : real.sqrt 3 :=
sorry

end size_of_angle_A_area_of_Triangle_ABC_l685_685133


namespace area_of_three_sectors_l685_685628

-- The mathematical properties and conditions
def radius := 15
def angle := 60
def num_sectors := 3

-- The area of one sector of a circle given its radius and central angle
def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * π * r^2

-- The total area of num_sectors sectors
def total_area (r : ℝ) (θ : ℝ) (n : ℕ) : ℝ :=
  n * sector_area r θ

-- Statement of the theorem
theorem area_of_three_sectors : total_area radius angle num_sectors = 112.5 * π := 
by
  sorry

end area_of_three_sectors_l685_685628


namespace radius_of_circumscribed_sphere_of_cube_l685_685986

theorem radius_of_circumscribed_sphere_of_cube (a : ℝ) (ha : a = 1) : 
  ∃ r : ℝ, r = (1 / 2) * Real.sqrt (a^2 + a^2 + a^2) ∧ r = (Real.sqrt 3) / 2 := 
by 
  use (1 / 2) * Real.sqrt 3
  split
  · rw [ha, ←real.sqrt_mul, mul_self_sqrt]
    ring
    norm_num
  · norm_num [←real.sqrt_mul, mul_self_sqrt, Real.sqrt]

end radius_of_circumscribed_sphere_of_cube_l685_685986


namespace inequality_proof_l685_685478

theorem inequality_proof (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) : b < a :=
by
  sorry

end inequality_proof_l685_685478


namespace exists_regular_dodecagon_diagonal_sum_l685_685360

-- Defining the problem conditions
def is_regular_dodecagon (P : Polygon) : Prop :=
  P.sides = 12 ∧ P.is_regular ∧ P.is_inscribed_in_circle

def diagonal_lengths (P : Polygon) (R : ℝ) : Prop :=
  let A1A7 := P.diagonal_length 1 7
  let A1A3 := P.diagonal_length 1 3
  let A1A11 := P.diagonal_length 1 11
  A1A7 = 2 * R ∧ A1A3 = R ∧ A1A11 = R

-- Problem statement to prove
theorem exists_regular_dodecagon_diagonal_sum :
  ∃ (P : Polygon) (R : ℝ), is_regular_dodecagon P ∧ diagonal_lengths P R ∧
  P.diagonal_length 1 7 = P.diagonal_length 1 3 + P.diagonal_length 1 11 := 
by
  sorry

end exists_regular_dodecagon_diagonal_sum_l685_685360


namespace abs_pi_minus_abs_pi_minus_10_l685_685014

theorem abs_pi_minus_abs_pi_minus_10 (h1 : Real.pi < 10) (h2 : 2 * Real.pi < 10) : 
  |Real.pi - |Real.pi - 10|| = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_minus_abs_pi_minus_10_l685_685014


namespace max_ab_l685_685119

/- Define the lines in terms of their equations and the condition for perpendicularity -/
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x + (2 * a - 4) * y + 1 = 0
def line2 (b : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * b * x + y - 2 = 0

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_perp : perpendicular ((-2) / (2*a - 4)) (-2*b)) :
  ab ≤ 1/2 :=
begin
  sorry
end

end max_ab_l685_685119


namespace max_min_sum_l685_685604

noncomputable def y (x : ℝ) : ℝ := x^3 - 3*x
noncomputable def y' (x : ℝ) : ℝ := deriv y x

theorem max_min_sum : 
  let m := y 1, let n := y (-1) in m + n = 0 := 
by {
  let m := y 1,
  let n := y (-1),
  have h₁ : m = 2,
  { unfold y, simp },
  have h₂ : n = -2,
  { unfold y, simp },
  calc 
    m + n = 2 + (-2) : by rw [h₁, h₂]
         ... = 0 : by ring
}

end max_min_sum_l685_685604


namespace smallest_hot_dog_packages_l685_685278

theorem smallest_hot_dog_packages :
  ∃ n m : ℕ, 5 * n = 7 * m ∧ n = 7 :=
by
  use 7, 5
  sorry

end smallest_hot_dog_packages_l685_685278


namespace geometric_number_difference_l685_685777

theorem geometric_number_difference :
  ∃ (a b c d e f g h : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h) ∧
    (a, b, c, d ∈ Finset.range(1, 10)) ∧
    (e, f, g, h ∈ Finset.range(1, 10)) ∧
    ((b = a * r) ∧ (c = a * r^2) ∧ (d = a * r^3)) ∧
    ((f = e * s) ∧ (g = e * s^2) ∧ (h = e * s^3)) ∧
    ((to_nat a * 1000 + to_nat b * 100 + to_nat c * 10 + to_nat d) - 
     (to_nat e * 1000 + to_nat f * 100 + to_nat g * 10 + to_nat h) = 7173) := 
sorry

end geometric_number_difference_l685_685777


namespace initial_cows_proof_l685_685709

-- Definitions based on our conditions
def initial_cows (C : ℕ) := C
def days_full_food := 50
def cows_run_away := 200 -- Approximation of 199.99999999999994 according to problem context
def food_after_10_days (F : ℝ) := 4 * F / 5
def remaining_food_enough (C : ℕ) (F : ℝ) := food_after_10_days F / (50 * (C - cows_run_away)) = F / 50

-- Theorem statement proving the initial number of cows is 200
theorem initial_cows_proof (F : ℝ) (C : ℕ) (h₁ : remaining_food_enough C F) : initial_cows C = 200 := 
sorry

end initial_cows_proof_l685_685709


namespace next_term_of_geometric_sequence_l685_685665

theorem next_term_of_geometric_sequence (x : ℝ) (hx : x ≠ 0) : 
  let a₁ := 2
  let a₂ := 6 * x
  let a₃ := 18 * x^2
  let a₄ := 54 * x^3
  let r := a₂ / a₁
  let next_term := a₄ * r
  in next_term = 162 * x^4 :=
by
  -- Proof goes here
  sorry

end next_term_of_geometric_sequence_l685_685665


namespace contrapositive_proof_l685_685960

-- Defining the necessary variables and the hypothesis
variables (a b : ℝ)

theorem contrapositive_proof (h : a^2 - b^2 + 2 * a - 4 * b - 3 ≠ 0) : a - b ≠ 1 :=
sorry

end contrapositive_proof_l685_685960


namespace lemonade_cost_l685_685338

theorem lemonade_cost (amount_given change_received : ℕ) (h1 : amount_given = 75) (h2 : change_received = 17) : (amount_given - change_received = 58) :=
by {
  rewrite [h1, h2],
  exact rfl,
}

end lemonade_cost_l685_685338


namespace license_plate_configurations_l685_685727

theorem license_plate_configurations :
  (3 * 10^4 = 30000) :=
by
  sorry

end license_plate_configurations_l685_685727


namespace no_real_values_satisfy_log_eq_l685_685481

theorem no_real_values_satisfy_log_eq (x : ℝ) 
  (h1 : x + 5 > 0) 
  (h2 : x - 2 > 0)
  (h3 : x^2 - 3x - 10 > 0) : 
  ¬ (log (x + 5) + log (x - 2) = log (x^2 - 3x - 10)) :=
by sorry

end no_real_values_satisfy_log_eq_l685_685481


namespace sequence_general_term_l685_685521

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ k ≥ 1, a (k + 1) = 2 * a k) : a n = 2 ^ (n - 1) :=
sorry

end sequence_general_term_l685_685521


namespace red_peaches_count_l685_685625

theorem red_peaches_count (yellow_peaches green_peaches total_peaches : Nat) 
  (h_yellow : yellow_peaches = 15) 
  (h_green : green_peaches = 8) 
  (h_total : total_peaches = 30) : 
  (total_peaches - (yellow_peaches + green_peaches)) = 7 := 
  by
  rw [h_yellow, h_green, h_total]
  calc
    30 - (15 + 8) = 30 - 23 : by rfl
    ... = 7 : by rfl
    sorry

end red_peaches_count_l685_685625


namespace final_statue_weight_l685_685286

-- Conditions
def original_weight : ℝ := 180
def first_week_cut_percentage : ℝ := 28
def second_week_cut_percentage : ℝ := 18
def third_week_cut_percentage : ℝ := 20

-- Definitions based on conditions
def weight_after_first_week : ℝ := original_weight * (1 - first_week_cut_percentage / 100)
def weight_after_second_week : ℝ := weight_after_first_week * (1 - second_week_cut_percentage / 100)
def weight_after_third_week : ℝ := weight_after_second_week * (1 - third_week_cut_percentage / 100)

-- Proof statement
theorem final_statue_weight :
  weight_after_third_week = 85.0176 :=
by
  sorry

end final_statue_weight_l685_685286


namespace shortest_altitude_right_triangle_l685_685238

theorem shortest_altitude_right_triangle (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2) :
  let area := 0.5 * a * b in
  let altitude := 2 * area / c in
  altitude = 7.2 :=
by
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  rw [h1, h2, h3] at *,
  let area := 0.5 * 9 * 12,
  let altitude := 2 * area / 15,
  have : area = 54, by norm_num,
  rw this at *,
  have : altitude = 7.2, by norm_num,
  exact this

end shortest_altitude_right_triangle_l685_685238


namespace quadrilateral_area_l685_685627

/-- 
Prove that the area of the quadrilateral formed by three coplanar squares with sides of 4, 6, 
and 8 arranged side-by-side with one side on line CD, and a segment connecting the bottom 
left corner of the smallest square to the upper right corner of the largest square, is 56/3.
-/
theorem quadrilateral_area (side1 side2 side3 : ℕ) (h1 : side1 = 4) (h2 : side2 = 6) (h3 : side3 = 8) :
  let total_length := side1 + side2 + side3,
      base1 := side1,
      base2 := side1 + side2,
      height_ratio := (side3 : ℚ) / total_length,
      height1 := base1 * height_ratio,
      height2 := base2 * height_ratio,
      length_between_bases := side2,
      area := (height1 + height2) / 2 * length_between_bases
  in
  area = 56 / 3 := 
by
  sorry

end quadrilateral_area_l685_685627


namespace complex_division_problem_l685_685857

theorem complex_division_problem :
  let z1 := 1 - complex.i
  let z2 := 1 + complex.i
  (z1 * z2) / complex.i = -2 * complex.i :=
by 
  let z1 := (1 : ℂ) - complex.i
  let z2 := (1 : ℂ) + complex.i
  sorry

end complex_division_problem_l685_685857


namespace sufficiency_but_not_necessary_l685_685451

theorem sufficiency_but_not_necessary (x y : ℝ) : |x| + |y| ≤ 1 → x^2 + y^2 ≤ 1 ∧ ¬(x^2 + y^2 ≤ 1 → |x| + |y| ≤ 1) :=
by
  sorry

end sufficiency_but_not_necessary_l685_685451


namespace sin_period_l685_685352

theorem sin_period (b : ℝ) (h : b = 3) : ∃ T : ℝ, T = 2 * Real.pi / b ∧ (∀ x : ℝ, sin (b * (x + T)) = sin (b * x)) :=
by
  use 2 * Real.pi / b
  split
  . rfl
  . intro x
    simp [h, mul_add, Real.sin_add, mul_comm]
    sorry

end sin_period_l685_685352


namespace graph_shift_l685_685601

theorem graph_shift (f g : ℝ → ℝ) 
  (h1 : ∀ x, f x = √3 * Real.sin x + Real.cos x)
  (h2 : ∀ x, g x = Real.sin x - √3 * Real.cos x) : 
  ∃ (φ : ℝ), φ = π / 2 ∧ ∀ x, g x = f (x - φ) :=
by
  sorry

end graph_shift_l685_685601


namespace triangle_is_right_triangle_l685_685965

noncomputable def sin_squared_sum_eq_cos_squared_sum (A B C : ℝ) : Prop :=
  (sin A)^2 + (sin B)^2 + (sin C)^2 = 2 * ((cos A)^2 + (cos B)^2 + (cos C)^2)

theorem triangle_is_right_triangle {A B C : ℝ} (h : sin_squared_sum_eq_cos_squared_sum A B C) :
  A + B + C = π → (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
sorry

end triangle_is_right_triangle_l685_685965


namespace sum_of_exponents_of_powers_of_two_1990_l685_685826

theorem sum_of_exponents_of_powers_of_two_1990 :
  ∃ (α : List ℕ), 
    (∑ i in α, 2^i = 1990) ∧ 
    List.Nodup α ∧ 
    (∑ i in α, i = 43) :=
by
  sorry

end sum_of_exponents_of_powers_of_two_1990_l685_685826


namespace least_possible_number_of_coins_in_jar_l685_685272

theorem least_possible_number_of_coins_in_jar (n : ℕ) : 
  (n % 7 = 3) → (n % 4 = 1) → (n % 6 = 5) → n = 17 :=
by
  sorry

end least_possible_number_of_coins_in_jar_l685_685272


namespace angle_between_a_b_is_45_degrees_l685_685078

variables {ι : Type*} [inner_product_space ℝ ι]

noncomputable def e1 : ι := sorry
noncomputable def e2 : ι := sorry

-- Conditions
axiom e1_unit : ∥e1∥ = 1
axiom e2_unit : ∥e2∥ = 1
axiom e1_e2_orthogonal : ⟪e1, e2⟫ = 0

def a : ι := 3 • e1 - e2
def b : ι := 2 • e1 + e2

-- Proof Statement
theorem angle_between_a_b_is_45_degrees : real.angle_of a b = real.pi / 4 := sorry

end angle_between_a_b_is_45_degrees_l685_685078


namespace find_divisor_l685_685306

theorem find_divisor (x y : ℝ) (hx : x = 0.42857142857142855) (hneq : x ≠ 0) (h : sqrt (3 * x / y) = x) : y = 7 :=
sorry

end find_divisor_l685_685306


namespace find_n_l685_685535

def C_n (n : ℕ) : ℝ :=
  512 * (1 - (1/2)^n)

def D_n (n : ℕ) : ℝ :=
  (2048 / 3) * (1 - (-1/2)^n)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C_n n = D_n n ↔ n = 4 := by
  sorry

end find_n_l685_685535


namespace solve_system_l685_685974

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + 2 * y + 3 * z = 3) ∧
    (3 * x + y + 2 * z = 7) ∧
    (2 * x + 3 * y + z = 2) ∧
    x = 2 ∧ y = -1 ∧ z = 1 :=
by
  use 2, -1, 1
  split
  { linarith }
  split
  { linarith }
  split
  { linarith }
  split
  { refl }
  split
  { refl }
  { refl }

end solve_system_l685_685974


namespace lucas_age_correct_l685_685188

variable (Noah_age : ℕ) (Mia_age : ℕ) (Lucas_age : ℕ)

-- Conditions
axiom h1 : Noah_age = 12
axiom h2 : Mia_age = Noah_age + 5
axiom h3 : Lucas_age = Mia_age - 6

-- Goal
theorem lucas_age_correct : Lucas_age = 11 := by
  sorry

end lucas_age_correct_l685_685188


namespace f_value_at_2011_5_l685_685350

noncomputable def f : ℝ → ℝ := sorry  -- We define the function f later.

theorem f_value_at_2011_5 : f 2011.5 = -0.5 :=
by
  -- Define the conditions in the problem statement
  have h1 : ∀ x : ℝ, f (-x) = -f x, from sorry,
  have h2 : ∀ x : ℝ, f (x + 2) = f x, from sorry,
  have h3 : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = x, from sorry,
  
  -- Use these conditions to prove the required statement
  sorry

end f_value_at_2011_5_l685_685350


namespace area_percentage_l685_685489

theorem area_percentage (D_S D_R : ℝ) (h : D_R = 0.8 * D_S) : 
  let R_S := D_S / 2
  let R_R := D_R / 2
  let A_S := π * R_S^2
  let A_R := π * R_R^2
  (A_R / A_S) * 100 = 64 := 
by
  sorry

end area_percentage_l685_685489


namespace correct_inequality_l685_685402

-- Define the conditions
variables (a b : ℝ)
variable (h : a > 1 ∧ 1 > b ∧ b > 0)

-- State the theorem to prove
theorem correct_inequality (h : a > 1 ∧ 1 > b ∧ b > 0) : 
  (1 / Real.log a) > (1 / Real.log b) :=
sorry

end correct_inequality_l685_685402


namespace adam_coins_value_l685_685736

def totalValueOfCollection (num_coins: ℕ) 
                            (num_rare: ℕ) 
                            (value_5_rare: ℕ) 
                            (value_diff: ℕ)
                            (equal_number: Prop) : ℕ :=
  if equal_number then
    let value_rare := value_5_rare / 5 in
    let value_common := value_rare - value_diff in
    let total_rare_value := num_rare * value_rare in
    let total_common_value := (num_coins - num_rare) * value_common in
    total_rare_value + total_common_value
  else 0

theorem adam_coins_value : 
    totalValueOfCollection 20 10 15 1 (10 = 10) = 50 := 
by   
    -- conditions & mathematical proof steps
    sorry

end adam_coins_value_l685_685736


namespace product_fraction_series_l685_685033

theorem product_fraction_series : 
  (∏ n in Finset.range (100 - 3 + 1), (1 - 1 / (n + 3))) = 1 / 50 := 
by
  sorry

end product_fraction_series_l685_685033


namespace factorial_equation_l685_685026

theorem factorial_equation (N : ℕ) (h : 6! * 11! = 20 * N!) : N = 12 :=
sorry

end factorial_equation_l685_685026


namespace interest_payment_frequency_l685_685219

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ)
  (h1 : i = 0.10) (h2 : EAR = 0.1025) :
  (1 + i / n)^n = 1 + EAR → n = 2 :=
by
  intros
  sorry

end interest_payment_frequency_l685_685219


namespace distribute_students_l685_685357

theorem distribute_students:
  (∃ f : Fin 7 → Fin 2, (∀ i, f i = 0 ∨ f i = 1) ∧ 
  (∑ i, if f i = 0 then 1 else 0) ≥ 2 ∧ 
  (∑ i, if f i = 1 then 1 else 0) ≥ 2) → 
  (∃ d_choices: ℕ, d_choices = (Nat.choose 7 2 + Nat.choose 7 3 + Nat.choose 7 4 + Nat.choose 7 5) ∧ d_choices = 112) := 
by
  sorry

end distribute_students_l685_685357


namespace contacts_after_8_levels_correct_l685_685728

-- Define the number of contacts at each level
def contacts_at_level (n : ℕ) : ℕ :=
  2^(n-1)

-- Define the total number of contacts after 8 levels
def total_contacts_after_8_levels : ℕ :=
  (List.sum (List.map contacts_at_level [2, 3, 4, 5, 6, 7, 8]))

-- Prove the total number of contacts is 254
theorem contacts_after_8_levels_correct : total_contacts_after_8_levels = 254 :=
by
  unfold total_contacts_after_8_levels contacts_at_level
  norm_num
  unfold List.foldr List.map List.sum
  norm_num
  sorry

end contacts_after_8_levels_correct_l685_685728


namespace derivative_sin_div_x_l685_685384

theorem derivative_sin_div_x (x : ℝ) (h : x ≠ 0) : 
  deriv (λ x : ℝ, sin x / x) x = (x * cos x - sin x) / x^2 :=
by
  sorry

end derivative_sin_div_x_l685_685384


namespace algebraic_identity_specific_case_l685_685968

theorem algebraic_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2 * a * b :=
by sorry

theorem specific_case : 2021^2 - 2021 * 4034 + 2017^2 = 16 :=
by sorry

end algebraic_identity_specific_case_l685_685968


namespace equation_one_solution_equation_two_solution_l685_685971

theorem equation_one_solution (x : ℝ) (h : 2 * (2 - x) - 5 * (2 - x) = 9) : x = 5 :=
sorry

theorem equation_two_solution (x : ℝ) (h : x / 3 - (3 * x - 1) / 6 = 1) : x = -5 :=
sorry

end equation_one_solution_equation_two_solution_l685_685971


namespace find_h_zero_l685_685342

-- Given conditions as definitions
def h (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d  -- Monic quartic polynomial
variables (a b c d : ℝ)

axiom h_neg2 : h (-2) = -4
axiom h_1 : h 1 = -1
axiom h_3 : h 3 = -9
axiom h_5 : h 5 = -25

-- Statement of the problem
theorem find_h_zero : h 0 = -30 :=
by
  sorry

end find_h_zero_l685_685342


namespace sqrt_expression_simplification_l685_685793

noncomputable def sqrt_expr (x : ℝ) := real.sqrt (x + real.sqrt (x + real.sqrt x))

theorem sqrt_expression_simplification (x : ℝ) (hx : 0 ≤ x) :
  sqrt_expr x = real.sqrt (x + real.sqrt (x + real.sqrt x)) := sorry

end sqrt_expression_simplification_l685_685793


namespace area_of_region_bounded_by_graph_l685_685383

theorem area_of_region_bounded_by_graph :
  (∫ x in 0..1, (2*x + 1 - 1)) = 6 :=
by
  -- Proof here
  sorry

end area_of_region_bounded_by_graph_l685_685383


namespace part1_part2_l685_685457

-- Given conditions
def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x - Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * ((vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2) - 1
def h (x : ℝ) (a : ℝ) : ℝ := f x - a

-- Part (1) statement
theorem part1 (m : ℝ) : 
  (∀ (x ∈ Set.Icc 0 Real.pi), (f x)^2 - (m + 3) * (f x) + 3 * m = 0 → 
  (∃ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ f a = f b ∧ f b = f c ∧ f c = f d)) → 
  m = Real.sqrt 3 ∨ m = - Real.sqrt 3 :=
sorry

-- Part (2) statement
theorem part2 (a : ℝ) (n : ℤ) :
  (∃! (x : ℝ) (y : ℝ), 0 ≤ x ∧ x < n * Real.pi ∧ h x a = 0 → h y a = 0 → x ≠ y) →
  (a = 3 ∧ n = 1011 ∨ a = Real.sqrt 3 ∨ a = - Real.sqrt 3 ∧ n = 2023) :=
sorry

end part1_part2_l685_685457


namespace total_quarters_l685_685962

def Sara_initial_quarters : Nat := 21
def quarters_given_by_dad : Nat := 49

theorem total_quarters : Sara_initial_quarters + quarters_given_by_dad = 70 := 
by
  sorry

end total_quarters_l685_685962


namespace proof_problem_l685_685106

def U := {1, 2, 3, 4, 6}
def P := {1, 3, 4}
def Q := {1, 2, 3, 6}

theorem proof_problem : (P ∩ Q = {1, 3}) ∧ (U.card = 5) := by
  sorry

end proof_problem_l685_685106


namespace part1_part2_part3_l685_685818

-- Given conditions
def Gamma (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)
def is_vertex (B : ℝ × ℝ) : Prop := B = (0, 1)
def distance_eq (v1 v2 v3 v4 : ℝ × ℝ) : Prop := 
  let dist := λ u v, Real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2)
  dist v1 v2 = dist v3 v4
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1
def distance_from_origin (l : ℝ → ℝ) (d : ℝ) : Prop := 
  let p := l 0
  Real.abs p / (Real.sqrt (1 + (l 1)^2)) = d

-- Part 1
theorem part1 (m : ℝ) (P := point_P m) :
  (m ≤ -Real.sqrt 2) →
  is_vertex (0, 1) →
  distance_eq (0, 1) F1 P F1 →
  m = -1 - Real.sqrt 2 := sorry

-- Part 2
theorem part2 (l : ℝ → ℝ) :
  (∃ m, m ≤ -Real.sqrt 2 ∧ distance_from_origin l (4 * Real.sqrt 15 / 15)) →
  (∃ θ, dot_product (√2 * Real.cos θ - 1, Real.sin θ) (√2 * Real.cos θ + 1, Real.sin θ) = 1 / 3) →
  (l 1 = 1/3 ∧ l 0 = 4 * Real.sqrt 6 / 9) := sorry

-- Part 3
theorem part3 (m : ℝ) :
  m < -Real.sqrt 2 →
  ∃ l : ℝ → ℝ,
  ∃ (A B : ℝ × ℝ), 
    (Gamma A.1 A.2 ∧ Gamma B.1 B.2 ∧ A.2 > 0 ∧ B.2 > 0 ∧ l A.1 = A.2 ∧ l B.1 = B.2) →
    (parallel (f1_f2_diff (A.1 - F1.1, A.2 - F1.2)) (B.1 - F2.1, B.2 - F2.2)) := sorry


end part1_part2_part3_l685_685818


namespace intersection_A_B_l685_685183

-- Defining sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l685_685183


namespace probability_x_greater_8y_l685_685197

theorem probability_x_greater_8y :
  let rectangle := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3014 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3015}
  let area_rectangle := (3014 : ℝ) * 3015
  let f : set (ℝ × ℝ) := {p | p.1 > 8 * p.2}
  let rectangle_triangle := {p | p.2 < p.1 / 8 ∧ p.1 ≤ 3014}
  let area_triangle := 1 / 2 * 3014 * 376.75
  (area_triangle / area_rectangle = 7535 / 120600) :=
by
  sorry

end probability_x_greater_8y_l685_685197


namespace a_n_formula_b_n_formula_c_n_monotonically_increasing_λ_range_l685_685835

-- Given conditions
def S_n (n : ℕ) := 2 * a_n n - 2
def b_n_seq_condition (n : ℕ) := (1 / (a_n n)) = (b_1 / (2 + 1 : ℝ)) -
  (b_2 / ((2 ^ 2) + 1)) + (b_3 / ((2 ^ 3) + 1)) + ... + ((-1)^(n + 1)) * (b_n / ((2 ^ n) + 1))

-- Problem 1
theorem a_n_formula (n : ℕ) (h : n > 0) : (a_n n = 2^n) := 
sorry

-- Problem 2
theorem b_n_formula (n : ℕ) (h : n > 0) : 
  b_n n = if n = 1 then 3 / 2 else (-1)^n * (1 / (2^n) + 1) := 
sorry

-- Problem 3
theorem c_n_monotonically_increasing_λ_range (λ : ℝ) :
  (-128 / 35 < λ) ∧ (λ < 32 / 19) :=
sorry

end a_n_formula_b_n_formula_c_n_monotonically_increasing_λ_range_l685_685835


namespace range_of_f_l685_685232

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x)

theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2 ∧ f x ≤ 1) :=
sorry

end range_of_f_l685_685232


namespace perfect_squares_example_l685_685694

def isPerfectSquare (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

theorem perfect_squares_example :
  let a := 10430
  let b := 3970
  let c := 2114
  let d := 386
  isPerfectSquare (a + b) ∧
  isPerfectSquare (a + c) ∧
  isPerfectSquare (a + d) ∧
  isPerfectSquare (b + c) ∧
  isPerfectSquare (b + d) ∧
  isPerfectSquare (c + d) ∧
  isPerfectSquare (a + b + c + d) :=
by
  -- Proof steps go here
  sorry

end perfect_squares_example_l685_685694


namespace max_remainder_division_by_9_l685_685233

theorem max_remainder_division_by_9 : ∀ (r : ℕ), r < 9 → r ≤ 8 :=
by sorry

end max_remainder_division_by_9_l685_685233


namespace max_of_f_l685_685859

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x ^ 2

theorem max_of_f 
  (k : ℝ) 
  (hk : k ∈ Set.Ioc (1 / 2) 1) :
  (∃ x ∈ Set.Icc (0 : ℝ) k, ∀ y ∈ Set.Icc (0 : ℝ) k, f y k ≤ f x k) ∧ 
  ∀ x ∈ Set.Icc (0 : ℝ) k, f x k ≤ f k k :=
begin
  sorry
end

end max_of_f_l685_685859


namespace find_s_l685_685477

theorem find_s (s t : ℚ) (h1 : 8 * s + 6 * t = 120) (h2 : s = t - 3) : s = 51 / 7 := by
  sorry

end find_s_l685_685477


namespace vector_inequality_l685_685415

noncomputable
def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1*v.1 + v.2*v.2)

theorem vector_inequality
  (a b c d : ℝ × ℝ)
  (h : a + b + c + d = (0, 0)) :
  vector_magnitude a + vector_magnitude b + vector_magnitude c + vector_magnitude d
  ≥ vector_magnitude (a + d) + vector_magnitude (b + d) + vector_magnitude (c + d) :=
by sorry

end vector_inequality_l685_685415


namespace smallest_n_cube_ends_with_2016_l685_685480

theorem smallest_n_cube_ends_with_2016 : ∃ n : ℕ, (n^3 % 10000 = 2016) ∧ (∀ m : ℕ, (m^3 % 10000 = 2016) → n ≤ m) :=
sorry

end smallest_n_cube_ends_with_2016_l685_685480


namespace isosceles_triangle_l685_685070
   
   theorem isosceles_triangle (a b c : ℝ) 
         (h_eqn: (a + c) * 1^2 - 2 * b * 1 - a + c = 0) : 
         c = b :=
   by
   simp at h_eqn,
   sorry
   
end isosceles_triangle_l685_685070


namespace product_of_roots_l685_685474

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ^ 2 - 2 * x1 = 2) (h2 : x2 ^ 2 - 2 * x2 = 2) (hne : x1 ≠ x2) :
  x1 * x2 = -2 := 
sorry

end product_of_roots_l685_685474


namespace next_term_of_geometric_sequence_l685_685666

theorem next_term_of_geometric_sequence (x : ℝ) (hx : x ≠ 0) : 
  let a₁ := 2
  let a₂ := 6 * x
  let a₃ := 18 * x^2
  let a₄ := 54 * x^3
  let r := a₂ / a₁
  let next_term := a₄ * r
  in next_term = 162 * x^4 :=
by
  -- Proof goes here
  sorry

end next_term_of_geometric_sequence_l685_685666


namespace smallest_sum_of_table_l685_685517

open Nat

theorem smallest_sum_of_table :
  ∀ (table : Matrix (Fin 4) (Fin 6) ℕ), 
    (∀ i j, i ≠ j → row_sum table i ≠ row_sum table j ∧ col_sum table i ≠ col_sum table j) → 
    ∑ i j, table i j = 43 := 
sorry

end smallest_sum_of_table_l685_685517


namespace trips_needed_l685_685755

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l685_685755


namespace determine_stability_metric_l685_685279

-- Definitions for the problem
def ExamScores (n : ℕ) := Vector ℝ n

-- Conditions: Xiao Liang took 6 exams
def xiao_liang_scores : ExamScores 6 := Vector.nil

-- Definitions for Variance and Standard Deviation
def Variance : Type := ℝ -- Placeholder type
def StandardDeviation : Type := ℝ -- Placeholder type

-- Question: What metric is needed for stability?
inductive StabilityMetric
| Variance
| StandardDeviation

-- Translation of the Problem to a Lean Statement
theorem determine_stability_metric (scores : ExamScores 6) : 
  StabilityMetric = StabilityMetric.Variance ∨ StabilityMetric = StabilityMetric.StandardDeviation :=
sorry

end determine_stability_metric_l685_685279


namespace probability_one_special_opera_l685_685221

noncomputable def probability_of_selecting_one_of_two_special_operas : ℚ :=
  let n := 5 in
  let r := 2 in
  let c_n_r := Nat.choose n r in
  let c_2_1 := Nat.choose 2 1 in
  let c_3_1 := Nat.choose 3 1 in
  (c_2_1 * c_3_1) / c_n_r

theorem probability_one_special_opera :
  probability_of_selecting_one_of_two_special_operas = 3 / 5 := 
by
  sorry

end probability_one_special_opera_l685_685221


namespace MayDayPromotion_l685_685614

variable (x : ℝ)
def discountRate := 0.1
def originalPrice := x
def sellingPrice := originalPrice - discountRate * originalPrice

theorem MayDayPromotion (x : ℝ) :
  sellingPrice x = 0.9 * x :=
by
  sorry

end MayDayPromotion_l685_685614


namespace number_of_true_propositions_l685_685088

/-- Given propositions about the parallelism and perpendicularity of lines and planes. -/
def p1 : Prop := ∀ (l₁ l₂ l₃ : Line), parallel l₁ l₃ ∧ parallel l₂ l₃ → parallel l₁ l₂
def p2 : Prop := ∀ (l₁ l₂ : Line) (P : Plane), parallel l₁ P ∧ parallel l₂ P → parallel l₁ l₂
def p3 : Prop := ∀ (l₁ l₂ l₃ : Line), perp l₁ l₃ ∧ perp l₂ l₃ → parallel l₁ l₂
def p4 : Prop := ∀ (l₁ l₂ : Line) (P : Plane), perp l₁ P ∧ perp l₂ P → parallel l₁ l₂

theorem number_of_true_propositions : 
  (∃ (n : ℕ), n = 2 ∧ (list.count (λ p, p = true) [p1, p2, p3, p4] = n)) :=
sorry

end number_of_true_propositions_l685_685088


namespace lines_parallel_iff_l685_685840

def line1 (a : ℝ) : set (ℝ × ℝ) :=
  {p | let (x, y) := p in a * x + (a + 2) * y + 2 = 0}

def line2 (a : ℝ) : set (ℝ × ℝ) :=
  {p | let (x, y) := p in x + a * y + 1 = 0}

theorem lines_parallel_iff (a : ℝ) :
  (∀ x y, line1 a (x, y) → line2 a (x, y) → a = -1) ∨
  (∀ x y, line1 a (x, y) → ¬line2 a (x, y) → true) :=
by
  sorry

end lines_parallel_iff_l685_685840


namespace pipeB_faster_than_pipeA_l685_685558

-- Define the rates of the pipes using the given conditions
def rateA := 1 / 56
def combined_rate := 1 / 7

-- Define the rate of Pipe B
def rateB := combined_rate - rateA

-- Define the theorem to prove
theorem pipeB_faster_than_pipeA (rateA rateB : ℝ) (h_rateA : rateA = 1 / 56) (h_combined_rate : combined_rate = 1 / 7) : rateB / rateA = 7 :=
by
  -- Given conditions
  have h_rateB : rateB = combined_rate - rateA, from sorry,
  rw [h_rateA, h_combined_rate] at h_rateB,
  linarith
  
-- Here we assume that the given rateB and calculations make sense within the context.

end pipeB_faster_than_pipeA_l685_685558


namespace greatest_possible_percentage_l685_685942

theorem greatest_possible_percentage :
  ∀ (A B C D S R : ℝ), 
    A = 0.4 → 
    C = 0.6 → 
    (B ≤ 1 ∧ B ≥ 0) → 
    (D ≤ 1 ∧ D ≥ 0) → 
    ∃ (p : ℝ), p = 0.4 ∧ p = min A (min B (min C D)) :=
by {
    intros A B C D S R hA hC hB hD,
    use 0.4,
    split,
    { assumption },
    { rw [hA, hC],
      have hB1 : B ≤ 1 := hB.left,
      have hB0 : B ≥ 0 := hB.right,
      have hD1 : D ≤ 1 := hD.left,
      have hD0 : D ≥ 0 := hD.right,
      apply min_left,
      apply min_left,
      linarith }
}

end greatest_possible_percentage_l685_685942


namespace true_universal_quantifier_l685_685276

theorem true_universal_quantifier :
  ∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1) := by
  sorry

end true_universal_quantifier_l685_685276


namespace abs_pi_minus_abs_pi_minus_9_l685_685770

theorem abs_pi_minus_abs_pi_minus_9 : ∀ (π : ℝ), \pi ≈ 3.14 -> |π - |π - 9|| = 9 - 2 * π :=
by
  intro π hπ
  sorry

end abs_pi_minus_abs_pi_minus_9_l685_685770


namespace foundation_cost_l685_685923

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end foundation_cost_l685_685923


namespace ratio_square_dodecagon_l685_685598

theorem ratio_square_dodecagon 
  (x : ℝ) 
  (regular_dodecagon : ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → vertex_location : ℝ²) 
  (square_vertices : ℝ² → Prop) 
  (h1 : ∀ v ∈ square_vertices, ∃ θ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ vertex_location θ = v) 
  (area_dodecagon : ℝ) 
  (area_square : ℝ) 
  (h2 : area_dodecagon = 3 * x^2 * (2 + Real.sqrt 3)) 
  (h3 : area_square = 2 * x^2 * (2 + Real.sqrt 3)) :
  (area_square / area_dodecagon = 2 / 3) :=
by 
  sorry

end ratio_square_dodecagon_l685_685598


namespace factorial_expression_eq_l685_685333

theorem factorial_expression_eq :
  (7! - 6 * 6! - 2 * 6!) = -720 := by
  sorry

end factorial_expression_eq_l685_685333


namespace equation_of_hyperbola_abscissa_of_B_l685_685226
noncomputable theory

-- Definitions of the hyperbola and its characteristics
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Given conditions
variables (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
def left_vertex (A : ℝ × ℝ) : Prop := A = (-a, 0)
def right_focus (F : ℝ × ℝ) : Prop := F = (c, 0)
def point_on_hyperbola (B : ℝ × ℝ) : Prop := hyperbola a b B.1 B.2
def perpendicular (BF AF : ℝ × ℝ) : Prop := BF.1 * AF.1 + BF.2 * AF.2 = 0
def distance_AF_BF (A F B : ℝ × ℝ) : Prop := dist A F = 2 * dist B F
def area_triangle (A B F : ℝ × ℝ) : Prop := 0.5 * abs (A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2) + F.1 * (A.2 - B.2)) = 25 / 4

-- Theorem statements
theorem equation_of_hyperbola :
  ∃ a b, a > 0 ∧ b > 0 ∧ (∃ x y, hyperbola a b x y) ∧ left_vertex (-a, 0) ∧ right_focus (3/2 * a, 0) ∧
  point_on_hyperbola (3/2 * a, b^2 / a) ∧ perpendicular 
  ((3/2 * a, b^2 / a), (3/2 * a, 0)) ((-a, 0), (3/2 * a, 0)) ∧ 
  distance_AF_BF (-a, 0) (3/2 * a, 0) (3/2 * a, b^2 / a) ∧ 
  area_triangle (-a, 0) (3/2 * a, b^2 / a) (3/2 * a, 0) → 
  a = 2 ∧ b = √5 :=
sorry

theorem abscissa_of_B :
  ∃ x₀ > 2, ∃ y₀ > 0,
  point_on_hyperbola (x₀, y₀) ∧ tan (3 * arctan (y₀ / (x₀ + 2))) = -(y₀ / (x₀ - 3)) ∧
  x₀ = (29 + 5 * sqrt 17) / 8 :=
sorry

end equation_of_hyperbola_abscissa_of_B_l685_685226


namespace area_triangle_CDE_eq_8_l685_685603

-- Define the isosceles right triangle properties
variables {A B C D E : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]
variables {s : A} (area_ABC : A) (s_eq : s = 4) (hypotenuse_AB : A) (AD DE EB : A)
variables (area_CDE : A)

-- Assume the conditions given in the problem
axiom is_isosceles_right_triangle : s = sqrt (2 * area_ABC)
axiom area_ABC_eq : area_ABC = 8
axiom hypotenuse_AB_eq : hypotenuse_AB = s * sqrt 2
axiom points_EQ : AD = 2 * sqrt 2 ∧ DE = 2 * sqrt 2 ∧ EB = 2 * sqrt 2
axiom legs_EQ : CD = s ∧ CE = s

-- Prove the area equivalence
theorem area_triangle_CDE_eq_8 : (1 / 2) * CD * CE = 8 :=
by
  sorry

end area_triangle_CDE_eq_8_l685_685603


namespace construct_pentagon_from_midpoints_l685_685405

open Function

theorem construct_pentagon_from_midpoints
  (O₁ O₂ O₃ O₄ O₅ : Point) : 
  (∃ P₁ P₂ P₃ P₄ P₅ : Point, 
    midpoint P₁ P₂ = O₁ ∧
    midpoint P₂ P₃ = O₂ ∧
    midpoint P₃ P₄ = O₃ ∧
    midpoint P₄ P₅ = O₄ ∧
    midpoint P₅ P₁ = O₅) :=
sorry

end construct_pentagon_from_midpoints_l685_685405


namespace hyperbola_equation_l685_685025

theorem hyperbola_equation :
  ∃ (a b : ℝ), (∃ c : ℝ, c = 3 ∧ a = sqrt 5 ∧ b = sqrt (c^2 - a^2) ∧ b = 2) ∧ 
  (∀ (x y : ℝ), y^2 / a^2 - x^2 / b^2 = 1 ↔ (x, y) = (4, -5)) ∧ 
  (a^2 = 5 ∧ b^2 = 4) ∧ 
  ((y : ℝ) y^2 / 5 - (x : ℝ) x^2 / 4 = 1) :=
sorry

end hyperbola_equation_l685_685025


namespace well_defined_log_expression_l685_685353

noncomputable def log_base (b x : ℝ) : ℝ := 
  if b > 0 ∧ b ≠ 1 ∧ x > 0 then real.log x / real.log b else 0

theorem well_defined_log_expression (y : ℝ) : 
  (∃ x1 x2 x3, 
    x1 = log_base 2008 y ∧ 
    x2 = log_base 2005 x1 ∧ 
    x3 = log_base 2010 x2 ∧ 
    x3 > 0
  ) ↔ y > 2008 ^ 2005 :=
by
  sorry

end well_defined_log_expression_l685_685353


namespace symmetry_center_coords_min_value_a_l685_685058

noncomputable theory

/-- 
  Given a function f(x) = cos^2 x - sqrt(3)/2 sin 2x - 1/2,
  prove that the coordinates of the symmetry center of the graph of f(x) is (pi/12 + k * pi/2, 0).
-/
theorem symmetry_center_coords :
  ∀ k : ℤ, 
  ∃ x : ℝ, f x = cos(2 * x + π / 3) ∧ x = π / 12 + k * π / 2 :=
sorry

/-- 
  In triangle ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
  Given f(A) + 1 = 0 and b + c = 2, prove that the minimum value of a is 1.
-/
theorem min_value_a (A a b c : ℝ) (hA : 0 < A ∧ A < π) (h1 : f A + 1 = 0) (h2 : b + c = 2) :
  a ≥ 1 :=
sorry

end symmetry_center_coords_min_value_a_l685_685058


namespace find_b_c_l685_685988

open Matrix

def line : ℝ × ℝ × ℝ :=
  let a := (-5, 0, 1)
  let b := (1, 4, 3)
  (b.1 - a.1, b.2 - a.2, b.3 - a.3)

theorem find_b_c : 
  (∃ b c : ℝ, 
    (line = (6, 4, 2)) ∧ 
    (2, b, c) = (2, θ.1 * (b / 3), θ.2 * (c / 3)) 
  ) :=
sorry

end find_b_c_l685_685988


namespace complete_square_l685_685656

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l685_685656


namespace problem_l685_685864

def f (x : ℝ) : ℝ := x * (Real.exp x - (1 / Real.exp x))

theorem problem (x1 x2 : ℝ) (h : f x1 < f x2) : x1^2 < x2^2 :=
sorry

end problem_l685_685864


namespace projection_of_2a_minus_b_on_a_l685_685460

-- Definitions of the vectors
variable {V : Type*} [inner_product_space ℝ V]

def a : V := sorry  -- The vector a
def b : V := sorry  -- The vector b

-- Conditions
axiom angle_ab : real_inner a b = ∥a∥ * ∥b∥ * real.cos (2 / 3 * real.pi)
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 4

-- Statement of the problem
theorem projection_of_2a_minus_b_on_a :
  (real_inner (2 • a - b) a) / ∥a∥ = 6 := 
sorry

end projection_of_2a_minus_b_on_a_l685_685460


namespace experimental_pi_value_l685_685146

theorem experimental_pi_value 
  (side_length : ℝ) (n m : ℕ) 
  (h1 : side_length = 3) 
  (h2 : ∃ (points : fin n → (ℝ × ℝ)), ∀ i, points i ∈ { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ side_length ∧ 0 ≤ p.2 ∧ p.2 ≤ side_length })
  (h3 : ∃ (points' : fin m → (ℝ × ℝ)), ∀ j, points' j ∈ { p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 < 1^2 ∨ (p.1 - side_length)^2 + (p.2 - 0)^2 < 1^2 ∨ (p.1 - 0)^2 + (p.2 - side_length)^2 < 1^2 ∨ (p.1 - side_length)^2 + (p.2 - side_length)^2 < 1^2 })
  : (9 * (m : ℝ)) / (n : ℝ) = real.pi :=
sorry

end experimental_pi_value_l685_685146


namespace find_b_plus_m_l685_685446

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x + 1) / Real.log a + b 

variable (a b m : ℝ)
-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : f a b m = 3

theorem find_b_plus_m : b + m = 3 :=
sorry

end find_b_plus_m_l685_685446


namespace infinitely_many_numbers_with_888_l685_685887

theorem infinitely_many_numbers_with_888 (k : ℕ) :
  ∃ (n : ℕ), (n = 81 * k^2 - 2 * k) ∧  
  (∃ (b : ℕ), (b = (floor (sqrt ↑n *10^3 )) % 1000) ∧ b = 888 ) :=
by
  sorry

end infinitely_many_numbers_with_888_l685_685887


namespace root_values_l685_685938

noncomputable def a : ℂ := 1.4656
noncomputable def b : ℂ := -0.2328 + 0.7926 * complex.I
noncomputable def c : ℂ := -0.2328 - 0.7926 * complex.I

theorem root_values :
  (a + b + c = 1) ∧
  (a * b + a * c + b * c = 0) ∧
  (a * b * c = -1) :=
by {
  sorry
}

end root_values_l685_685938


namespace birch_trees_count_l685_685137

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l685_685137


namespace vector_magnitude_example_l685_685882

def vector := (ℝ × ℝ)

def magnitude (v : vector) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vector_add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_magnitude_example : magnitude (vector_add (1, 2) (2, 2)) = 5 := by
  sorry

end vector_magnitude_example_l685_685882


namespace smallest_abundant_not_multiple_of_8_l685_685391

def is_abundant(n : ℕ) : Prop :=
  (nat.divisors n).sum > 2 * n

def is_not_multiple_of_8(n : ℕ) : Prop :=
  ¬ (n % 8 = 0)

theorem smallest_abundant_not_multiple_of_8 : ∃ n, is_abundant n ∧ is_not_multiple_of_8 n ∧ (∀ m, is_abundant m ∧ is_not_multiple_of_8 m → n ≤ m) :=
begin
  use 18,
  split,
  { show is_abundant 18, sorry },
  split,
  { show is_not_multiple_of_8 18, sorry },
  { intros m h,
    sorry }
end

end smallest_abundant_not_multiple_of_8_l685_685391


namespace find_b_l685_685152

theorem find_b (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = 3^n + b)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1))
  (h_geometric : ∃ r, ∀ n ≥ 1, a n = a 1 * r^(n-1)) : b = -1 := 
sorry

end find_b_l685_685152


namespace general_term_l685_685084

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S2 : S 2 = 4
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by
  sorry

end general_term_l685_685084


namespace feb_29_is_sunday_l685_685476

noncomputable def is_leap_year : Prop := sorry

def day_of_week := ℕ

def Wednesday : day_of_week := 3   -- Assuming days of the week are numbered 0 to 6 with Sunday as 0
def Sunday : day_of_week := 0

axiom feb_11_is_wednesday (Y : Type) [is_leap_year] : day_of_week := Wednesday

theorem feb_29_is_sunday (Y : Type) [is_leap_year] (h : feb_11_is_wednesday Y) : day_of_week :=
  (h + 18) % 7

#eval feb_29_is_sunday Type sorry (by exact lhs) -- Expected result: Sunday

end feb_29_is_sunday_l685_685476


namespace complete_the_square_l685_685648

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l685_685648


namespace num_distinct_values_S_l685_685056

def i : ℂ := Complex.I

def S (n k : ℤ) : ℂ := i^(n + k) + i^(-(n + k))

theorem num_distinct_values_S : (finset.card (finset.image (λ (nk : ℤ × ℤ), S nk.1 nk.2) ((finset.univ : finset ℤ).product (finset.of_list [1, 2])))) = 3 :=
by {
  sorry
}

end num_distinct_values_S_l685_685056


namespace abs_pi_abs_pi_minus_10_l685_685011

def pi := Real.pi

theorem abs_pi_abs_pi_minus_10 (h : pi < 10) : |pi - |pi - 10|| = 10 - 2 * pi := by
  sorry

end abs_pi_abs_pi_minus_10_l685_685011


namespace length_of_tunnel_length_of_tunnel_km_final_tunnel_length_l685_685735

def train_length : ℝ := 100 -- length in meters
def train_speed : ℝ := 72 -- speed in km/hr
def time_minutes : ℝ := 3 -- time in minutes

def train_speed_ms (v : ℝ) : ℝ := v * (1000 / 3600) -- function to convert speed to m/s
def time_seconds (t : ℝ) : ℝ := t * 60 -- function to convert time to seconds

def distance_travelled (v : ℝ) (t : ℝ) : ℝ := v * t -- distance = speed * time

theorem length_of_tunnel (L_train : ℝ) (v_train : ℝ) (t_min : ℝ) : 
  distance_travelled (train_speed_ms v_train) (time_seconds t_min) - L_train = 3500 :=
sorry

theorem length_of_tunnel_km (L_tunnel_m : ℝ) : L_tunnel_m / 1000 = 3.5 :=
sorry

theorem final_tunnel_length : 
  length_of_tunnel 100 72 3 / 1000 = 3.5 :=
sorry

end length_of_tunnel_length_of_tunnel_km_final_tunnel_length_l685_685735


namespace planet_X_periods_l685_685301

theorem planet_X_periods (N : ℕ) (h1 : N = 100000) :
  ∃ n m : ℕ, n * m = N ∧ 1 ≤ n ∧ 1 ≤ m ∧ ∀ a b : ℕ, a * b = N → ((a = n ∧ b = m) ∨ (a = m ∧ b = n)) :=
sorry

end planet_X_periods_l685_685301


namespace events_B_and_C_are_independent_l685_685631

theorem events_B_and_C_are_independent :
  let outcomes := { (i, j) | i, j ∈ fin 6 }
  let A := { (x, y) ∈ outcomes | x + y = 6 }
  let B := { (x, y) ∈ outcomes | x % 2 = 1 }
  let C := { (x, y) ∈ outcomes | x = y }
  let P := λ s : set (ℕ × ℕ), (s.card : ℚ) / ((outcomes.card : ℚ))
  P (B ∩ C) = P B * P C := sorry

end events_B_and_C_are_independent_l685_685631


namespace scientific_notation_of_42000_l685_685193

theorem scientific_notation_of_42000 : 42000 = 4.2 * 10^4 := 
by 
  sorry

end scientific_notation_of_42000_l685_685193


namespace expression_evaluation_l685_685268

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l685_685268


namespace XG_XH_calculation_l685_685721

variables (O A B C D E X Y F G H : Type)
variables [circle O]
variables [inscribed_pentagon A B C D E O]
variables [point_on_line X D B] [point_on_line Y B D]
variables (AX : line A X) [parallel_line_through Y AE F] (CX : line C X) [parallel_line_through F CD G] (DG : line D G)
variables [another_point_on_circle_except H D O DG]

def XG_mult_XH : ℝ :=
  let BD := line_length B D in
  let DX := (1 : ℝ) / (5 : ℝ) * BD in
  let BY := (1 : ℝ) / (4 : ℝ) * BD in
  let length_AX_G_CX_F_X := sorry in
  -- Assuming power of point theorem and properties of similar triangles to set the final answer
  length_AX_G_CX_F_X

theorem XG_XH_calculation :
  let BD := sorry in
  let XG := sorry in
  let XH := sorry in
  XG * XH = (51 : ℝ) / (7 : ℝ) :=
sorry

end XG_XH_calculation_l685_685721


namespace assembly_line_arrangements_l685_685741

noncomputable def numberOfValidArrangements : Nat :=
  let independentTasks := 3 -- S, IS, WiI
  let dependentTasks := 2   -- A-W, P-AC
  (independentTasks + dependentTasks)! -- total number of ways to arrange

theorem assembly_line_arrangements :
  let A := "Axles"
  let W := "Wheels"
  let P := "Powertrain"
  let AC := "AirConditioning"
  let Wi := "Windshield"
  let I := "InstrumentPanel"
  let S := "SteeringWheel"
  let IS := "InteriorSeating"
  -- Conditions:
  -- 1. A before W
  -- 2. P before AC
  -- 3. Wi and I together
 :
  numberOfValidArrangements = 120 :=
  by 
  sorry

end assembly_line_arrangements_l685_685741


namespace smallest_n_for_purple_candy_l685_685028

theorem smallest_n_for_purple_candy :
  ∃ (n : ℕ), (∃ (r g b : ℕ), 10 * r = 18 * g ∧ 18 * g = 24 * b ∧ 24 * b = 25 * n) ∧ n = 15 :=
by
  use 15
  obtain ⟨r, (18 * g), hb⟩ : (∃ r g b, 10 * r = 18 * g ∧ 18 * g = 24 * b) := sorry
  exact ⟨r, 18 * g, hb, by sorry⟩

end smallest_n_for_purple_candy_l685_685028


namespace log_multiplication_example_l685_685283

theorem log_multiplication_example : (Real.logBase 2 9) * (Real.logBase 3 4) = 4 := 
sorry

end log_multiplication_example_l685_685283


namespace problem_statement_l685_685468

theorem problem_statement (a x : ℝ) (h_linear_eq : (a + 4) * x ^ |a + 3| + 8 = 0) : a^2 + a - 1 = 1 :=
sorry

end problem_statement_l685_685468


namespace value_of_expression_l685_685936

open Polynomial

theorem value_of_expression (a b : ℚ) (h1 : (3 : ℚ) * a ^ 2 + 9 * a - 21 = 0) (h2 : (3 : ℚ) * b ^ 2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (2 * b - 2) = -4 :=
by sorry

end value_of_expression_l685_685936


namespace parallelogram_divides_congruent_polygons_l685_685779

theorem parallelogram_divides_congruent_polygons :
  ∃ m n : ℤ, m + n = 6 ∧ 
    let line := λ (x : ℝ), (m : ℝ) * x / (n : ℝ) in
    let p1 := (12, 50 : ℝ) in
    let p2 := (12, 120 : ℝ) in
    let p3 := (30, 160 : ℝ) in
    let p4 := (30, 90 : ℝ) in
    let v1 := (12, 50 : ℝ) in
    let v2 := (12, 50 + (30*m)/(12*n) : ℝ) in
    let v3 := (30, 160 - (30*m)/(12*n) : ℝ) in
    let v4 := (30, 90 : ℝ) in
    line (v1.1) = v1.2 ∧ line (v2.1) = v2.2 ∧ 
    line (v3.1) = v3.2 ∧ line (v4.1) = v4.2 ∧ 
    (arc_dist p1 p3) = (arc_dist p2 p4) :=
by 
sory

end parallelogram_divides_congruent_polygons_l685_685779


namespace hh_of_2_eq_91265_l685_685173

def h (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - x + 1

theorem hh_of_2_eq_91265 : h (h 2) = 91265 := by
  sorry

end hh_of_2_eq_91265_l685_685173


namespace intersection_eq_l685_685874

-- Define the conditions M and N based on the problem
def M : Set ℝ := { x | log 10 (x - 1) < 0 }
def N : Set ℝ := { x | 2 * x ^ 2 - 3 * x ≤ 0 }

-- The proof problem statement
theorem intersection_eq : M ∩ N = { x : ℝ | 1 < x ∧ x ≤ 3 / 2 } :=
sorry

end intersection_eq_l685_685874


namespace variance_after_scaling_l685_685247

variable {s^2 : ℝ}  -- Assume the original variance s^2 is a real number
variable {a : ℝ}    -- Assume a is a real number.

theorem variance_after_scaling {s^2 : ℝ} (a : ℝ) (s : ℝ) (h : s^2 = s * s):
  (a * 2) * (a * 2) * s^2 = 4 * s^2 :=
by
  sorry

end variance_after_scaling_l685_685247


namespace problems_per_page_l685_685953

theorem problems_per_page (total_problems : ℕ) (percent_solved : ℝ) (pages_left : ℕ)
  (h_total : total_problems = 550)
  (h_percent : percent_solved = 0.65)
  (h_pages : pages_left = 3) :
  (total_problems - Nat.ceil (percent_solved * total_problems)) / pages_left = 64 := by
  sorry

end problems_per_page_l685_685953


namespace ellipse_C2_equation_slopes_sum_constant_tangent_circles_l685_685185

-- Condition 1: parabola C1 with directrix and focus
def parabola_C1 : Prop :=
  ∃ F1 F2 : ℝ × ℝ, ⦃(y, x) : ℝ × ℝ | y^2 = 8 * x⦄ ∧ 
  -- F1 intersects the x-axis
  F1.2 = 0 ∧ 
  -- F2 is the focus
  (∃ d : ℝ, d > 0 ∧ (λ (x y : ℝ), y² = 8 * x) = F2)

-- Condition 2: ellipse C2 with foci and eccentricity
def ellipse_C2 : Prop :=
  ∃ (F1 F2 : ℝ × ℝ) (e : ℝ),
  F1 = (-2, 0) ∧ F2 = (2, 0) ∧ e = sqrt 2 / 2 ∧ 
  (∃ (a b c : ℝ), a > b > 0 ∧ c > 0 ∧ c = 2 ∧ c / a = sqrt 2 / 2 ∧ a² = b² + c²)

-- Condition 3: point N
def point_N : Prop := (0, -2)

-- Condition 4: line l through P intersects C2
def line_l_through_P : Prop :=
  ∀ (P A B : ℝ × ℝ), P = (1,2) ∧
  (A, B ∈ (λ (x y : ℝ), x² / 8 + y² / 4 = 1)) ∧ 
  A ≠ (0, -2) ∧ B ≠ (0, -2) 

-- Condition 5: slopes of lines NA and NB
def slopes_NA_NB (k1 k2 : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), 

-- Problem 1: Equation of ellipse C2
theorem ellipse_C2_equation (h1 : parabola_C1) (h2 : ellipse_C2) : 
  ∃ (C2_eq : ℝ → ℝ → Prop), C2_eq = (λ (x y : ℝ), x² / 8 + y² / 4 = 1) :=
sorry

-- Problem 2: Prove k1 + k2 constant
theorem slopes_sum_constant (h2 : ellipse_C2) (h3 : point_N) (h4 : line_l_through_P) (h5 : slopes_NA_NB) :
  ∀ k1 k2 : ℝ, k1 + k2 = 4 :=
sorry

-- Problem 3: Existence of fixed circle such that B's circle is always tangent
theorem tangent_circles (h1 : parabola_C1) (h2 : ellipse_C2) : 
  ∃ (circle_M_eq : ℝ → ℝ → Prop), circle_M_eq = (λ (x y : ℝ), (x-2)² + y² = 32) :=
sorry

end ellipse_C2_equation_slopes_sum_constant_tangent_circles_l685_685185


namespace gordonia_population_ratio_l685_685629

theorem gordonia_population_ratio (total_population : ℕ) (lake_bright_population : ℕ) (toadon_percentage : ℚ) (gordonia_population : ℚ) :
  total_population = 80000 →
  lake_bright_population = 16000 →
  toadon_percentage = 0.60 →
  let toadon_population := toadon_percentage * gordonia_population in
  let total_population_calc := gordonia_population + toadon_population + ↑lake_bright_population in
  total_population_calc = ↑total_population →
  gordonia_population / ↑total_population = 1 / 2 :=
begin
  intros h_total h_lake h_percentage, 
  simp only, 
  intros h_total_calc,
  sorry
end

end gordonia_population_ratio_l685_685629


namespace sum_first_seven_terms_l685_685484

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 a2 a3 a4 a5 a6 a7 : ℝ)

-- Assuming a is an arithmetic sequence with common difference d
axiom (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)

-- Given conditions
axiom (h_cond : a 3 + a 4 + a 5 = 12)

open Nat

theorem sum_first_seven_terms : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
  sorry

end sum_first_seven_terms_l685_685484


namespace value_of_a_if_p_and_q_are_false_l685_685841

variable (a : ℝ)
def p : Prop := ∃ x ∈ Icc 0 1, x^2 - 2*x - 2 + a > 0
def q : Prop := ∀ x : ℝ, x^2 - 2*x - a ≠ 0

theorem value_of_a_if_p_and_q_are_false (hp : ¬ p) (hq : ¬ q) : a ∈ Icc (-1 : ℝ) 2 :=
by
  -- Step 1: Negation of p
  have hnp : ∀ x ∈ Icc (0 : ℝ) 1, x^2 - 2*x - 2 + a ≤ 0 := by
    intro x hx
    exact sorry
  
  -- Step 2: Negation of q
  have hnq : ∃ x : ℝ, x^2 - 2*x - a = 0 := by
    exact sorry
  
  -- Step 3: Combining the results
  exact sorry

end value_of_a_if_p_and_q_are_false_l685_685841


namespace calculate_m_l685_685892

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l685_685892


namespace ellipse_equation_value_of_m_l685_685851

noncomputable def ellipse_foci_radius (a b : ℝ) (h : a^2 - b^2 = 1) : Prop :=
  ∃ F : ℝ, ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_equation (F_eq : F = (1, 0)) 
  (symmetry : ∃ a b : ℝ, a > b > 0 ∧ a^2 - b^2 = 1)
  (tangent : ∃ l : ℝ, x - 2 * sqrt(2) * y + 2 = 0):
  ellipse_foci_radius _ _ :=
sorry

theorem value_of_m (symmetry : ∃ a b : ℝ, a > b > 0 ∧ a^2 - b^2 = 1)
  (F_eq : F = (1, 0)) 
  (intersection : ∃ x1 y1 x2 y2 : ℝ, 
   y1 = x1 + m ∧ x1^2 + 2*y1^2 = 2 ∧ y2 = x2 + m ∧ x2^2 + 2*y2^2 = 2)
  (P_on_ellipse : ∃ P : ℝ × ℝ, P = (x1 + x2, y1 + y2) ∧ x1 + x2 = -4/3 * m ∧ y1 + y2 = 2/3 * m):
  m = ± (sqrt(3) / 2) :=
sorry

end ellipse_equation_value_of_m_l685_685851


namespace deductive_reasoning_not_always_correct_l685_685507

theorem deductive_reasoning_not_always_correct (P: Prop) (Q: Prop) 
    (h1: (P → Q) → (P → Q)) :
    (¬ (∀ P Q : Prop, (P → Q) → Q → Q)) :=
sorry

end deductive_reasoning_not_always_correct_l685_685507


namespace circumcenter_locus_l685_685737

variables {α : Type*} [LinearOrderedField α]

-- Definitions of the given conditions
def line (p q : α × α) : set (α × α) := 
  { r | ∃ t : α, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2)) }

def point_on_line (p r : α × α) (e : set (α × α)) : Prop :=
  r ∈ e

def segment_translated (p q : α × α) (e : set (α × α)) : Prop :=
  ∀ (k : α × α), (point_on_line k k e) → (∃ l : α × α, (segment_center p k l) ∈ e)

def segment_center (p q r : α × α) : α × α :=
  ((p.1 + q.1 + r.1) / 3, (p.2 + q.2 + r.2) / 3)

def reflection (p q : α × α) : α × α :=
  (2*p.1 - q.1, 2*p.2 - q.2)

-- Definition of the goal
theorem circumcenter_locus
  (e : set (α × α)) (J K L : α × α)
  (HJ : point_on_line J J e) (HK : point_on_line J K e) (Htranslate : segment_translated J K e) :
  let M := reflection J K in
  let f := line (segment_center J K M) (segment_center J K M) in
  ∀ C : α × α, (C ∈ f → C ≠ J → 
  ∃ (O : α × α), circumcircle J K L ∧ O ∈ segment_center J K L) := 
sorry

end circumcenter_locus_l685_685737


namespace inscribed_semicircle_radius_l685_685148

theorem inscribed_semicircle_radius (DE EF : ℝ) (angleF_right : ∠ F = π / 2) 
  (hDE : DE = 15) (hEF : EF = 8) :
  ∃ r : ℝ, r = 3 :=
by
  sorry

end inscribed_semicircle_radius_l685_685148


namespace exterior_angle_measure_l685_685433

theorem exterior_angle_measure (sum_interior_angles : ℝ) (h : sum_interior_angles = 1260) :
  ∃ (n : ℕ) (d : ℝ), (n - 2) * 180 = sum_interior_angles ∧ d = 360 / n ∧ d = 40 := 
by
  sorry

end exterior_angle_measure_l685_685433


namespace two_digit_integers_remainder_5_when_divided_by_7_count_l685_685112

theorem two_digit_integers_remainder_5_when_divided_by_7_count :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 7 = 5}.to_finset.card = 13 := by
sorry

end two_digit_integers_remainder_5_when_divided_by_7_count_l685_685112


namespace equal_tangent_lengths_l685_685593

noncomputable theory

open_locale classical

section TangentCircles

variables {R : Type*} [euclidean_space R] 
variables (O₀ O₁ : R) (r₀ r₁ : ℝ) (O₀T₀ O₁T₁ : R)
variables (A₀ B₀ A₁ B₁ : R)

-- Definitions for the circles and tangency points
def circle_β (O₀ O₁ : R) (r₀ r₁ : ℝ) : Prop :=
  dist O₀ O₁ > r₀ + r₁

def tangents_inter (O₀ O₁ A₀ B₀ A₁ B₁ : R) : Prop :=
  dist O₀ A₀ = dist O₀ B₀ ∧ dist O₁ A₁ = dist O₁ B₁

def is_tangent (O₀ O₁ A₀ B₀ A₁ B₁ : R) : Prop :=
tangents_inter O₀ O₁ A₀ B₀ A₁ B₁ ∧ circle_β O₀ O₁ (dist O₀ A₀) (dist O₀ B₀)

-- Statement to prove
theorem equal_tangent_lengths {O₀ O₁ A₀ B₀ A₁ B₁ : R} (h_tangent : is_tangent O₀ O₁ A₀ B₀ A₁ B₁) :
   dist A₀ B₀ = dist A₁ B₁ :=
sorry

end TangentCircles

end equal_tangent_lengths_l685_685593


namespace negation_of_diagonals_equal_l685_685608

-- Define a rectangle type and a function for the diagonals being equal
structure Rectangle :=
  (a b c d : ℝ) -- Assuming rectangle sides

-- Assume a function that checks if the diagonals of a given rectangle are equal
def diagonals_are_equal (r : Rectangle) : Prop :=
  sorry -- The actual function definition is omitted for this context

-- The proof problem
theorem negation_of_diagonals_equal :
  ¬ (∀ r : Rectangle, diagonals_are_equal r) ↔ (∃ r : Rectangle, ¬ diagonals_are_equal r) :=
by
  sorry

end negation_of_diagonals_equal_l685_685608


namespace hernandez_state_tax_l685_685288

theorem hernandez_state_tax 
    (res_months : ℕ) (total_months : ℕ) 
    (taxable_income : ℝ) (tax_rate : ℝ) 
    (prorated_income : ℝ) (state_tax : ℝ) 
    (h1 : res_months = 9) 
    (h2 : total_months = 12) 
    (h3 : taxable_income = 42500) 
    (h4 : tax_rate = 0.04) 
    (h5 : prorated_income = taxable_income * (res_months / total_months)) 
    (h6 : state_tax = prorated_income * tax_rate) : 
    state_tax = 1275 := 
by 
  -- this is where the proof would go
  sorry

end hernandez_state_tax_l685_685288


namespace least_altered_entries_make_sums_different_l685_685245

theorem least_altered_entries_make_sums_different :
  let original_matrix := ![![4, 9, 2], ![8, 1, 6], ![3, 5, 7]]
  (∀ (r1 r2 r3 c1 c2 c3 : ℕ),
     r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧
     c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧
     (∃ altered_matrix : Matrix (Fin 3) (Fin 3) ℕ, 
       ∀ (i j : Fin 3), altered_matrix i j ≠ original_matrix i j →
       ∃ altered_entries : Fin 3 → Fin 3 → ℕ,
       (∑ k, altered_entries i k ≠ ∑ k, (original_matrix i k : ℕ)) ∧
       (∑ k, altered_entries k j ≠ ∑ k, (original_matrix k j : ℕ)) ∧
       (alters_total := ∑ i j, if altered_matrix i j ≠ original_matrix i j then 1 else 0)
         alters_total = 4)) := sorry

end least_altered_entries_make_sums_different_l685_685245


namespace sum_of_zeros_of_fg_l685_685090

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
if x >= 0 then 2^(x-2) - 1
else x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 2 * x

-- Define the composite function f[g(x)]
def fg (x : ℝ) : ℝ := f (g x)

-- The theorem to prove the sum of all zeros of fg(x)
theorem sum_of_zeros_of_fg : 
  (∀ x ∈ ({1 + Real.sqrt 3, 1 - Real.sqrt 3} : set ℝ), fg x = 0) ∧
  (∀ x ∉ ({1 + Real.sqrt 3, 1 - Real.sqrt 3} : set ℝ), fg x ≠ 0) →
  finset.sum (finset.insert (1 + Real.sqrt 3) (finset.singleton (1 - Real.sqrt 3))) id = 2 :=
by
  sorry

end sum_of_zeros_of_fg_l685_685090


namespace fermat_numbers_divides_l685_685202

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_numbers_divides (n : ℕ) : F(n) ∣ 2^(F(n)) - 2 := 
sorry

end fermat_numbers_divides_l685_685202


namespace complement_correct_l685_685105

universe u

-- We define sets A and B
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

-- Define the complement of B with respect to A
def complement (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- The theorem we need to prove
theorem complement_correct : complement A B = {2, 4} := 
  sorry

end complement_correct_l685_685105


namespace problem_part1_problem_part2_l685_685108

open Real

def vecOA (λ α : ℝ) : ℝ × ℝ := (λ * sin α, λ * cos α)
def vecOB (β : ℝ) : ℝ × ℝ := (cos β, sin β)
def angleBetweenVectors (λ α β : ℝ) : ℝ := 
  let dot_product := (λ * sin α) * (cos β) + (λ * cos α) * (sin β)
  let magnitude_OA := sqrt ((λ * sin α)^2 + (λ * cos α)^2)
  let magnitude_OB := sqrt ((cos β)^2 + (sin β)^2)
  acos (dot_product / (magnitude_OA * magnitude_OB))

def distanceAB (λ α β : ℝ) : ℝ :=
  let vec_a := vecOA λ α
  let vec_b := vecOB β
  sqrt ((vec_b.1 - vec_a.1)^2 + (vec_b.2 - vec_a.2)^2)

theorem problem_part1 (λ α β : ℝ) (h1 : λ < 0) (h2 : α + β = 5 * π / 6) : 
  angleBetweenVectors λ α β = 2 * π / 3 := 
sorry

theorem problem_part2 (λ α β : ℝ) (h1 : -2 ≤ λ) (h2 : λ ≤ 2) (h3 : α + β = 5 * π / 6) : 
  ∃ l u, l = sqrt 3 / 2 ∧ u = sqrt 7 ∧ (sqrt 3 / 2 ≤ distanceAB λ α β ∧ distanceAB λ α β ≤ sqrt 7) := 
sorry

end problem_part1_problem_part2_l685_685108


namespace cone_volume_l685_685492

theorem cone_volume (r l h V: ℝ) (h1: 15 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2: 2 * Real.pi * r = (1 / 3) * Real.pi * l) :
  (V = (1 / 3) * Real.pi * r^2 * h) → h = Real.sqrt (l^2 - r^2) → l = 6 * r → r = Real.sqrt (15 / 7) → 
  V = (25 * Real.sqrt 3 / 7) * Real.pi :=
sorry

end cone_volume_l685_685492


namespace sum_of_numbers_with_reverse_base8_base9_eq_28_l685_685807

-- Assume definitions for the equivalence of a number's base 8 and base 9 representations are reversed
def is_reverse_base8_base9 (n : ℕ) : Prop :=
  let base8_digits := n.digits 8
  let base9_digits := n.digits 9
  base8_digits.reverse == base9_digits

-- Main theorem with the conditions mentioned translated
theorem sum_of_numbers_with_reverse_base8_base9_eq_28 :
    (∑ n in (Finset.filter is_reverse_base8_base9 (Finset.range 1000)), id n) = 28 := 
  sorry

end sum_of_numbers_with_reverse_base8_base9_eq_28_l685_685807


namespace distinct_numbers_by_multiplying_elements_l685_685111

def distinct_products_count (s : Finset ℕ) : ℕ := 
  (s.powerset.filter (λ t, 2 ≤ t.card)).image (λ t, t.prod id).card

theorem distinct_numbers_by_multiplying_elements : 
  distinct_products_count (Finset.insert 1 {3, 7, 9, 13}) = 11 :=
by
  sorry

end distinct_numbers_by_multiplying_elements_l685_685111


namespace juggler_balls_division_l685_685027

theorem juggler_balls_division (num_jugglers : ℕ) (total_balls : ℕ) (h1 : num_jugglers = 378) (h2 : total_balls = 2268) :
  total_balls / num_jugglers = 6 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num)
  rw [Nat.mul_comm, nat.add6_eq_id 2161] -- error assuming "~norm_num" here
  sorry

end juggler_balls_division_l685_685027


namespace meetings_percentage_l685_685928

-- Define all the conditions given in the problem
def first_meeting := 60 -- duration of first meeting in minutes
def second_meeting := 2 * first_meeting -- duration of second meeting in minutes
def third_meeting := first_meeting / 2 -- duration of third meeting in minutes
def total_meeting_time := first_meeting + second_meeting + third_meeting -- total meeting time
def total_workday := 10 * 60 -- total workday time in minutes

-- Statement to prove that the percentage of workday spent in meetings is 35%
def percent_meetings : Prop := (total_meeting_time / total_workday) * 100 = 35

theorem meetings_percentage :
  percent_meetings :=
by
  sorry

end meetings_percentage_l685_685928


namespace isosceles_triangle_exists_l685_685573

theorem isosceles_triangle_exists (n : Nat) (h : n ≥ 3) :
  ∀ (vertices : Finset (Fin (2 * n - 1))), vertices.card = n → 
  ∃ (a b c : Fin (2 * n - 1)), a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (dist a c = dist b c) :=
by
  sorry

end isosceles_triangle_exists_l685_685573


namespace variance_of_transformed_data_l685_685836

variable {n : ℕ}
variable (x : Fin n → ℝ)

-- Variance function definition
noncomputable def variance (x : Fin n → ℝ) : ℝ :=
  let μ := (∑ i, x i) / (n : ℝ)
  (∑ i, (x i - μ) ^ 2) / (n : ℝ)

theorem variance_of_transformed_data {n : ℕ} (x : Fin n → ℝ) (h : variance x = 1) : 
  variance (λ i, 3 * x i + 2) = 9 :=
by 
  sorry

end variance_of_transformed_data_l685_685836


namespace find_a_from_slope_monotonicity_of_f_inequality_holds_range_a_l685_685092

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 * Real.exp x

-- Condition (I): Given that the derivative of f at x=1 equals 6e, prove a = 2
theorem find_a_from_slope (a : ℝ) :
  (λ x, (f a x).1).deriv 1 = 6 * Real.exp 1 → a = 2 := 
sorry

-- Condition (II): Discuss the monotonicity of the function f for a ≠ 0
theorem monotonicity_of_f (a : ℝ) : 
  a ≠ 0 → 
  ((∀ x: ℝ, x < -2 ∨ x > 0 → (λ x, (f a x).1).deriv x > 0) ∧
   (∀ x: ℝ, -2 < x ∧ x < 0 → (λ x, (f a x).1).deriv x < 0)) ∨
  ((∀ x: ℝ, x < -2 ∨ x > 0 → (λ x, (f a x).1).deriv x < 0) ∧
   (∀ x: ℝ, -2 < x ∧ x < 0 → (λ x, (f a x).1).deriv x > 0)) := 
sorry

-- Condition (III): Given the inequality holds, find the range of a
theorem inequality_holds_range_a :
  (∀ x: ℝ, x ≤ 0 → f a x + x * Real.exp x + 1 ≥ Real.exp x) → 
  a ∈ Set.Ici (-1 / 2) := 
sorry

end find_a_from_slope_monotonicity_of_f_inequality_holds_range_a_l685_685092


namespace center_on_circumcircle_of_triangle_APB_l685_685597

-- Define points and functions related to the trapezoid and circles
variables {A B C D P O : Type} [IsoscelesTrapezoid A B C D AB_leg : bool]

-- Define the condition that O is the circumcenter of trapezoid ABCD
def is_circumcenter (O : Type) (trapezoid : Type) : Prop := sorry

-- Define the condition that P is the intersection point of diagonals AC and BD in trapezoid ABCD
def intersection_of_diagonals (A B C D P : Type) : Prop := sorry

-- Define what it means for a point to lie on the circumcircle of a triangle
def lies_on_circumcircle (O : Type) (triangle : Type) : Prop := sorry

-- Proof statement
theorem center_on_circumcircle_of_triangle_APB :
  IsIsoscelesTrapezoid A B C D →
  intersection_of_diagonals A B C D P →
  is_circumcenter O (IsoscelesTrapezoid A B C D) →
  lies_on_circumcircle O (triangle A P B) :=
by
  sorry -- Proof

end center_on_circumcircle_of_triangle_APB_l685_685597


namespace trips_needed_l685_685756

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l685_685756


namespace average_score_of_whole_class_l685_685287

theorem average_score_of_whole_class 
  (n : ℕ)
  (scores : Fin n → ℝ)
  (h_n : n = 20)
  (h_2_students : ∃ i₁ i₂ : Fin n, scores i₁ = 100 ∧ scores i₂ = 100 ∧ i₁ ≠ i₂)
  (h_3_students : ∃ i₃ i₄ i₅ : Fin n, scores i₃ = 0 ∧ scores i₄ = 0 ∧ scores i₅ = 0 ∧ i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₄ ≠ i₅)
  (h_rest : ∃ (remaining_indices : Fin n → Prop) 
            (h_card : (remaining_indices ∧ λ i, i ≠ i₁ ∧ i ≠ i₂ ∧ i ≠ i₃ ∧ i ≠ i₄ ∧ i ≠ i₅).card = 15) 
            (h_avg : (∑ i in Finset.filter remaining_indices Finset.univ, scores i) / 15 = 40)
  : (∑ i, scores i) / 20 = 40 :=
by
  sorry

end average_score_of_whole_class_l685_685287


namespace initial_best_method_method_a_not_best_after_additional_l685_685331

-- Definitions for initial grades and additional grades
def initial_grades : List ℕ := [4, 1, 2, 5, 2]
def additional_grades : List ℕ := [5, 5]

-- Function to calculate the arithmetic mean rounded to the nearest whole number
def rounded_arithmetic_mean (grades : List ℕ) : ℕ :=
  let mean := (grades.sum : ℚ) / (grades.length : ℚ) 
  Int.natAbs ⌊mean + 0.5⌋

-- Function to calculate the median
def median (grades : List ℕ) : ℕ :=
  let sorted := grades.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

-- Prove that initially, Method A is the best
theorem initial_best_method : rounded_arithmetic_mean initial_grades = 3 ∧ median initial_grades = 2 ∧ 3 > 2 := by
  sorry

-- Prove that after adding two grades of 5, Method A is not the best
theorem method_a_not_best_after_additional :
  let new_grades := initial_grades ++ additional_grades
  rounded_arithmetic_mean new_grades = 3 ∧ median new_grades = 4 ∧ 3 < 4 := by
  sorry

end initial_best_method_method_a_not_best_after_additional_l685_685331


namespace additional_men_needed_l685_685464

-- Definitions of the forces exerted
def force_grandpa : ℝ := F
def force_grandma : ℝ := 3 / 4 * force_grandpa
def force_granddaughter : ℝ := (3 / 4) ^ 2 * force_grandpa
def force_zhuchka : ℝ := (3 / 4) ^ 3 * force_grandpa
def force_cat : ℝ := (3 / 4) ^ 4 * force_grandpa

-- Combined force of grandma, granddaughter, Zhuchka, and the cat
def combined_force : ℝ := force_grandma + force_granddaughter + force_zhuchka + force_cat

-- Total force including Grandpa
def total_force : ℝ := force_grandpa + combined_force

-- Required additional force
def additional_force : ℝ := F * (1293 / 256 - 1)

-- Proving the number of additional men required is 4
theorem additional_men_needed : 
  ∃ n : ℕ, n = 4 ∧ (1:ℝ) + n = 1293 / 256 := by
  sorry

end additional_men_needed_l685_685464


namespace value_of_expression_l685_685819

theorem value_of_expression (x y z : ℝ) (h : (x * y * z) / (|x * y * z|) = 1) :
  (|x| / x + y / |y| + |z| / z) = 3 ∨ (|x| / x + y / |y| + |z| / z) = -1 :=
sorry

end value_of_expression_l685_685819


namespace fraction_numerator_l685_685596

theorem fraction_numerator (x : ℤ) (h₁ : 2 * x + 11 ≠ 0) (h₂ : (x : ℚ) / (2 * x + 11) = 3 / 4) : x = -33 / 2 :=
by
  sorry

end fraction_numerator_l685_685596


namespace log_equation_solutions_irrational_l685_685804

theorem log_equation_solutions_irrational :
  ∀ a : ℝ, (log 5 (2 * a^2 - 20 * a) = 3) →
  (∃ (x y : ℝ), (x = 5 + 2.5 * Real.sqrt 14) ∧ (y = 5 - 2.5 * Real.sqrt 14) ∧ (a = x ∨ a = y)) :=
by
  sorry

end log_equation_solutions_irrational_l685_685804


namespace log_inequality_solution_set_l685_685972

noncomputable def solve_log_inequality (x : ℝ) : Prop :=
  log (x^12 + 3*x^10 + 5*x^8 + 3*x^6 + 1) / log 2 < 1 + log (x^4 + 1) / log 2

theorem log_inequality_solution_set :
  {x : ℝ | solve_log_inequality x} =
  {x : ℝ | x > -sqrt ((sqrt 5 - 1) / 2) ∧ x < sqrt ((sqrt 5 - 1) / 2)} :=
sorry

end log_inequality_solution_set_l685_685972


namespace smaller_molds_radius_l685_685714

noncomputable def large_radius : ℝ := 2
def num_molds : ℕ := 64

noncomputable def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * real.pi * (r ^ 3)

theorem smaller_molds_radius : ∃ r : ℝ, (num_molds * volume_hemisphere r = volume_hemisphere large_radius) ∧ (r = 1 / 2) :=
by
  use (1 / 2)
  sorry

end smaller_molds_radius_l685_685714


namespace congruent_figures_l685_685255

theorem congruent_figures (F₁ F₂ : Set ℝ) (third_figure : Set ℝ) 
  (are_symmetrical : ∀ x ∈ F₁, refl_point x ∈ F₂ ∧ refl_point x' ∈ F₂) 
  (A B C : PSet ℝ) 
  (A_in_F₁ : A ∈ F₁)
  (B_in_F₁ : B ∈ F₁)
  (C_in_F₁ : C ∈ F₁)
  (A' B' C' : PSet ℝ) 
  (A'_symm : A' = reflect A first_axis)
  (B'_symm : B' = reflect B first_axis)
  (C'_symm : C' = reflect C first_axis) 
  (A'' B'' C'' : PSet ℝ) 
  (A''_symm : A'' = reflect A second_axis)
  (B''_symm : B'' = reflect B second_axis)
  (C''_symm : C'' = reflect C second_axis)
  : F₁ = F₂ := sorry

end congruent_figures_l685_685255


namespace min_degree_polynomial_with_roots_l685_685719

def is_rational (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

/-- The roots given are all the numbers \(1+\sqrt{2}, 2+\sqrt{4}, \dots, 1000+\sqrt{1002}\). -/
def given_roots : List ℚ := List.map (λ n, (n + Real.sqrt(n + 2) : ℚ)) (List.range (1000 + 1))

/-- The roots occurring in pairs of radical conjugates unless they are whole numbers -/
def split_conjugates (roots : List ℚ) : List (ℚ × ℚ) :=
roots.bind (λ n, if Real.sqrt(n + 2) ^ 2 = n + 2 then [(n, 0)] else [(n, -Real.sqrt(n + 2))])

theorem min_degree_polynomial_with_roots :
  ∃ p : Polynomial ℚ, (∀ r ∈ given_roots, Polynomial.eval r p = 0) ∧ p.degree = 1970 :=
by
  sorry

end min_degree_polynomial_with_roots_l685_685719


namespace log_defined_if_y_gt_d_l685_685024

theorem log_defined_if_y_gt_d (y : ℝ) : (∃ d : ℝ, d = 2401 ∧ y > d) → ∃ L : ℝ, 
  L = log 10 (log 9 (log 8 (log 7 (y^2)))) := 
by
sory

end log_defined_if_y_gt_d_l685_685024


namespace min_sum_of_4x6_table_with_distinct_sums_l685_685516

theorem min_sum_of_4x6_table_with_distinct_sums : 
  ∃ (table : Fin 4 → Fin 6 → ℕ), 
  (∀ i j, table i j > 0) ∧ 
  (let row_sums := λ i, (Finset.univ.sum (λ j, table i j)),
       col_sums := λ j, (Finset.univ.sum (λ i, table i j)) in
   ∀ i1 i2, i1 ≠ i2 → row_sums i1 ≠ row_sums i2 ∧
   ∀ j1 j2, j1 ≠ j2 → col_sums j1 ≠ col_sums j2) ∧
  (Finset.univ.sum (λ i, Finset.univ.sum (λ j, table i j)) = 43) :=
sorry

end min_sum_of_4x6_table_with_distinct_sums_l685_685516


namespace transformed_function_is_sin_l685_685853

noncomputable def f (A : ℝ) (ϕ x : ℝ) : ℝ := A * Real.cos(2 * x + ϕ)

theorem transformed_function_is_sin {A : ℝ} (hA : A > 0) {ϕ : ℝ} (hϕ : |ϕ| < Real.pi) 
  (h_odd : ∀ x, f A ϕ (-x) = -f A ϕ x) (h_value : f A ϕ (3 * Real.pi / 4) = -1) : 
  ∀ x, f A ϕ (x / 2) = Real.sin x := 
sorry

end transformed_function_is_sin_l685_685853


namespace first_number_positive_l685_685156

-- Define the initial condition
def initial_pair : ℕ × ℕ := (1, 1)

-- Define the allowable transformations
def transform1 (x y : ℕ) : Prop :=
(x, y - 1) = initial_pair ∨ (x + y, y + 1) = initial_pair

def transform2 (x y : ℕ) : Prop :=
(x, x * y) = initial_pair ∨ (1 / x, y) = initial_pair

-- Define discriminant function
def discriminant (a b : ℕ) : ℤ := b ^ 2 - 4 * a

-- Define the invariants maintained by the transformations
def invariant (a b : ℕ) : Prop :=
discriminant a b < 0

-- Statement to be proven
theorem first_number_positive :
(∀ (a b : ℕ), invariant a b → a > 0) :=
by
  sorry

end first_number_positive_l685_685156


namespace sequence_eventually_periodic_l685_685069

-- Define P(n) as the largest prime factor of n
def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).erase_dup'.last'.get_or_else 1

-- Recursive definition of the sequence a_n
def a_seq : ℕ → ℕ → ℕ
| 1, a1 := a1
| (n+1), a1 := let an := a_seq n a1 in
               if (Nat.factors an).length > 1 
               then an - largest_prime_factor an
               else an + largest_prime_factor an

-- Define the property of eventual periodicity
def eventually_periodic (f : ℕ → ℕ) (N T : ℕ) : Prop :=
  ∀ n, n ≥ N → f (n + T) = f n

-- Main theorem statement: the sequence {a_n} defined as above is eventually periodic.
theorem sequence_eventually_periodic (a1 : ℕ) (h : a1 ≥ 2) :
  ∃ (N T : ℕ), eventually_periodic (a_seq a1) N T :=
by
  sorry

end sequence_eventually_periodic_l685_685069


namespace picnic_men_count_l685_685680

variables 
  (M W A C : ℕ)
  (h1 : M + W + C = 200) 
  (h2 : M = W + 20)
  (h3 : A = C + 20)
  (h4 : A = M + W)

theorem picnic_men_count : M = 65 :=
by
  sorry

end picnic_men_count_l685_685680


namespace completing_the_square_l685_685642

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l685_685642


namespace number_of_four_digit_numbers_with_two_identical_digits_l685_685612

-- Define the conditions
def starts_with_nine (n : ℕ) : Prop := n / 1000 = 9
def has_exactly_two_identical_digits (n : ℕ) : Prop := 
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d2) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d2 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d1) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d2 ∧ n % 10 = d1)

-- Define the proof problem
theorem number_of_four_digit_numbers_with_two_identical_digits : 
  ∃ n, starts_with_nine n ∧ has_exactly_two_identical_digits n ∧ n = 432 := 
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l685_685612


namespace shortest_distance_between_points_is_line_segment_l685_685677

theorem shortest_distance_between_points_is_line_segment (a b c : Type) [plane a] [perpendicular b a] [perpendicular b c] [geometry b] :
  (∀ (P Q : Type) (hPQ : P ≠ Q), segment P Q = shortest_path P Q) := sorry

end shortest_distance_between_points_is_line_segment_l685_685677


namespace intersection_complement_B_l685_685186

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 3 * x < 0 }
def B : Set ℝ := { x | abs x > 2 }

-- Complement of B
def complement_B : Set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

-- Final statement to prove the intersection equals the given set
theorem intersection_complement_B :
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 2 } := 
by 
  -- Proof omitted
  sorry

end intersection_complement_B_l685_685186


namespace proof1_proof2_monotonically_increasing_interval_l685_685461

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 0.5 * Real.cos (2 * x)

theorem proof1 : ∀ x : ℝ, f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

theorem proof2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → -0.5 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
∃ lb ub : ℝ, lb = Real.pi / 6 + k * Real.pi ∧ ub = 2 * Real.pi / 3 + k * Real.pi ∧ ∀ x : ℝ, lb ≤ x ∧ x ≤ ub → f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

end proof1_proof2_monotonically_increasing_interval_l685_685461


namespace AC_tangent_to_circumcircle_CEG_l685_685905

open Triangle EuclideanGeometry Circle

noncomputable def problem := 
  ∀ (A B C D E F G : Point) (ABC_circum : Circle),
  (IsoscelesTriangle ABC A B C) →
  (OnArc ABC_circum.overarc_B C D) →
  (OnArc ABC_circum.overarc_C B E) →
  (LineIntersects AD BC F) →
  (Circle G : Circle) (OnCircumcircleOfTriangle G D E F) →
  (LineIntersects AE G G) →
  TangentToCircumcircle A C G C E G

theorem AC_tangent_to_circumcircle_CEG :
  problem :=
begin
  intros,
  sorry,
end

end AC_tangent_to_circumcircle_CEG_l685_685905


namespace derivative_of_y_l685_685040

noncomputable def y (x : ℝ) : ℝ :=
  (cos x) / ((sin x)^2) - 2 * (cos x) - 3 * log (tan (x / 2))

theorem derivative_of_y (x : ℝ) : deriv y x = - (2 + 3 * (sin x)^2) / (sin x)^3 :=
by
  sorry

end derivative_of_y_l685_685040


namespace range_of_m_solutions_l685_685023

noncomputable theory

open Real

theorem range_of_m_solutions (m : ℝ) :
  (∃ x : ℝ, |x + m| + |x - 1| ≤ 3) ↔ (-4 ≤ m ∧ m ≤ 2) :=
sorry

end range_of_m_solutions_l685_685023


namespace value_of_m_l685_685865

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 + m * x + 1) : is_even_function f → m = 0 := by
  sorry

end value_of_m_l685_685865


namespace problem1_l685_685688

theorem problem1 :
  (Real.sqrt (3/2)) * (Real.sqrt (21/4)) / (Real.sqrt (7/2)) = 3/2 :=
sorry

end problem1_l685_685688


namespace min_surface_area_sphere_l685_685834

-- Define the conditions
def ab_eq_one (a b : ℝ) := a * b = 1
def abc_eq_four (a b c : ℝ) := a * b * c = 4
def c_eq_four (a b : ℝ) := 4
def space_diagonal_eq_diameter (a b c r : ℝ) := 2 * r = Real.sqrt (a^2 + b^2 + c^2)

-- The main statement to prove
theorem min_surface_area_sphere (a b c r : ℝ) (h_ab: ab_eq_one a b) (h_abc: abc_eq_four a b c) (h_space_diag: space_diagonal_eq_diameter a b c r) :
  (c = 4) → (∀a, ∀b, a * b = 1 → 4 * π * r^2 = 18 * π)
  :=
begin
  intro hc,
  sorry
end

end min_surface_area_sphere_l685_685834


namespace sum_bn_correct_l685_685420

noncomputable def arithmetic_seq (a1 d : ℕ) : ℕ → ℕ
| n => a1 + (n - 1) * d

noncomputable def geometric_seq (g1 q : ℕ) : ℕ → ℕ
| n => g1 * q ^ (n - 1)

noncomputable def sequence_b (a1 a4 b1 b4 : ℕ) (n : ℕ) : ℕ :=
  let an := arithmetic_seq a1 ((a4 - a1) / 3) n
  let gn := geometric_seq (b1 - a1) ((b4 - a4) / (b1 - a1)).pow(1/3) n
  an + gn

noncomputable def sum_of_first_n_terms (b1 b4 : ℕ) (n : ℕ) : ℕ :=
  let seq_b := sequence_b 3 12 b1 b4 n
  (3 * n * (n + 1)) / 2 + 2^n - 1

theorem sum_bn_correct : ∀ n, sum_of_first_n_terms 4 20 n = (3 * n * (n + 1)) / 2 + 2^n - 1 :=
by
  intro n
  sorry

end sum_bn_correct_l685_685420


namespace sqrt_27_eq_3_sqrt_3_l685_685574

theorem sqrt_27_eq_3_sqrt_3 : Real.sqrt 27 = 3 * Real.sqrt 3 :=
by
  sorry

end sqrt_27_eq_3_sqrt_3_l685_685574


namespace lcm_16_35_l685_685388

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end lcm_16_35_l685_685388


namespace num_water_proof_l685_685499

-- Definitions based on the conditions
variable (T : ℕ) -- total number of students
variable (p : ℚ) -- proportion of students who chose juice
variable (q : ℚ) -- proportion of students who chose water
variable (n_j : ℕ) -- number of students who chose juice
variable (n_w : ℕ) -- number of students who chose water

-- Given conditions
def prop_juice := p = 0.40
def prop_water := q = 0.30
def num_juice := n_j = 120
def total_students := 0.40 * T = 120

-- The statement we need to prove
theorem num_water_proof : prop_juice → prop_water → num_juice → total_students → (0.30 * T = n_w) → n_w = 90 :=
by 
  intros prop_juice prop_water num_juice total_students water_eq
  sorry

end num_water_proof_l685_685499


namespace birch_trees_count_l685_685139

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l685_685139


namespace shea_final_height_l685_685571

def original_height_elton : ℝ := 60
def current_height_elton : ℝ := 75
def growth_percentage_elton : ℝ := 0.25
def growth_ratio_shea : ℝ := 2 / 5

def shea_grew_by := growth_ratio_shea * (growth_percentage_elton * original_height_elton)
def original_height_shea := original_height_elton

theorem shea_final_height : original_height_shea + shea_grew_by = 66 :=
by
  have elton_grew_by : ℝ := growth_percentage_elton * original_height_elton
  have shea_grew : ℝ := growth_ratio_shea * elton_grew_by
  have original_height : ℝ := original_height_elton
  show original_height + shea_grew = 66
  sorry

end shea_final_height_l685_685571


namespace find_a_and_b_l685_685443

noncomputable def f (x a b : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) :
  let f' (x : ℝ) :=  x^2 - 2 * x + a in
  (3 : ℝ) = f' 0 ∧ -2 = f 0 a b :=
by
  -- Proof omitted
  sorry

end find_a_and_b_l685_685443


namespace correct_representation_l685_685217

-- Definitions of conditions in Lean
def option_A : Prop := (-6 + 4 = 4)
def option_B : Prop := (+5 = -(−5))
def option_C : Prop := (+3.2 = -9)
def option_D : Prop := (+6 = -(−7))

-- The proposition that Option B is correct
theorem correct_representation : option_B :=
by { sorry }

end correct_representation_l685_685217


namespace find_car_costs_optimize_purchasing_plan_minimum_cost_l685_685678

theorem find_car_costs (x y : ℝ) (h1 : 3 * x + y = 85) (h2 : 2 * x + 4 * y = 140) :
    x = 20 ∧ y = 25 :=
by
  sorry

theorem optimize_purchasing_plan (m : ℕ) (h_total : m + (15 - m) = 15) (h_constraint : m ≤ 2 * (15 - m)) :
    m = 10 :=
by
  sorry

theorem minimum_cost (w : ℝ) (h_cost_expr : ∀ (m : ℕ), w = 20 * m + 25 * (15 - m)) (m := 10) :
    w = 325 :=
by
  sorry

end find_car_costs_optimize_purchasing_plan_minimum_cost_l685_685678


namespace shortest_altitude_right_triangle_l685_685239

theorem shortest_altitude_right_triangle (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2) :
  let area := 0.5 * a * b in
  let altitude := 2 * area / c in
  altitude = 7.2 :=
by
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  rw [h1, h2, h3] at *,
  let area := 0.5 * 9 * 12,
  let altitude := 2 * area / 15,
  have : area = 54, by norm_num,
  rw this at *,
  have : altitude = 7.2, by norm_num,
  exact this

end shortest_altitude_right_triangle_l685_685239


namespace max_m_value_l685_685066

def Circle (C : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
  {p | ((p.1 - C.1) ^ 2 + (p.2 - C.2) ^ 2 = r ^ 2)}

def perp_dot (A B P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1, P.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  (AP.1 * BP.1 + AP.2 * BP.2 = 0)

theorem max_m_value :
  ∃ P : ℝ × ℝ, P ∈ Circle (3, 4) 1 → perp_dot (-m, 0) (m, 0) P → m ≤ 6 :=
by
  sorry

end max_m_value_l685_685066


namespace sin_cos_value_l685_685776

noncomputable def sin_cos_identity : ℝ :=
  (real.sin (43 * real.pi / 180) * real.cos (13 * real.pi / 180) -
   real.sin (13 * real.pi / 180) * real.cos (43 * real.pi / 180))

theorem sin_cos_value : sin_cos_identity = 1 / 2 := by
  sorry

end sin_cos_value_l685_685776


namespace constant_k_for_linear_function_l685_685989

theorem constant_k_for_linear_function (k : ℝ) (h : ∀ (x : ℝ), y = x^(k-1) + 2 → y = a * x + b) : k = 2 :=
sorry

end constant_k_for_linear_function_l685_685989


namespace sequence_inequality_l685_685346

theorem sequence_inequality {n : ℕ} 
  (a b : Fin n → ℝ) 
  (h_ab : ∀ i, 1 ≤ a i ∧ a i ≤ 2 ∧ 1 ≤ b i ∧ b i ≤ 2)
  (h_sum_sq : ∑ i, (a i) ^ 2 = ∑ i, (b i) ^ 2) :
  ∑ i, (a i) ^ 3 / (b i) ≤ (17 / 10) * ∑ i, (a i) ^ 2 :=
sorry

end sequence_inequality_l685_685346


namespace number_of_x_satisfying_condition_l685_685290

theorem number_of_x_satisfying_condition :
  ∃ n : ℕ, n = 2857 ∧ 
  (∀ x : ℕ, (1 ≤ x ∧ x < 10000 → (2^x - x^2) % 7 = 0) ↔ ∃ m : ℕ, x = 21 * m + k ∧ k ∈ {2, 4, 5, 6, 11, 13, 14, 15, 16, 18, 19, 20, 9998}) :=
sorry

end number_of_x_satisfying_condition_l685_685290


namespace sequence_properties_l685_685419

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

def S_n (n : ℕ) : ℕ := (n * (2 * (n - 1) + 1)) / 2

def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

theorem sequence_properties :
  (a_4 = 7) →
  (S_4 = 16) →
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, T_n n = n / (2 * n + 1)) :=
by
  sorry

end sequence_properties_l685_685419


namespace locus_of_P_l685_685177

-- Define the geometrical constructs and properties.
variables {A B C P : Point} -- Vertices of the triangle and point P.
variables {A1 B1 C1 : Point} -- Projections of P.
variables [is_acute_triangle A B C] -- The triangle ABC is acute and scalene.
variables (projA : is_projection P B C A1) -- A1 is the projection of P onto BC.
variables (projB : is_projection P C A B1) -- B1 is the projection of P onto CA.
variables (projC : is_projection P A B C1) -- C1 is the projection of P onto AB.

-- Define the concurrency condition and angle sum condition.
variables (concurrent : concur AA1 BB1 CC1) -- AA1, BB1, and CC1 are concurrent.
variables (angle_sum : angle P A B + angle P B C + angle P C A = 90) -- Angle sum condition.

-- State the problem: proving P as incenter, circumcenter, or orthocenter.
theorem locus_of_P :
  concurrent → angle_sum →
  (P = incenter A B C ∨ P = circumcenter A B C ∨ P = orthocenter A B C) :=
sorry -- Proof to be provided later.

end locus_of_P_l685_685177


namespace inscribed_trapezoid_inradii_relation_l685_685985

theorem inscribed_trapezoid_inradii_relation
  (A B C D O : Type) 
  (r1 r2 r3 r4 : ℝ) 
  (AD BC : A → B → C → D → Prop)
  (inscribed : ∀ (A B C D : Type), A → B → C → D → Prop)
  (diagonals_intersect : ∀ (A B C D O : Type), A → B → C → D → O → Prop)
  (inradii_relation : (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4)) :
  Prop :=
begin
  sorry
end

end inscribed_trapezoid_inradii_relation_l685_685985


namespace point_in_quadrant_l685_685127

theorem point_in_quadrant (m : ℝ) (h : m < 0) : 
  -m + 1 > 0 ∧ -1 < 0 :=
begin
  sorry
end

end point_in_quadrant_l685_685127


namespace BD_bisects_AC_l685_685083

variables {O P A B C D : Type*}
variables (ABCD : CyclicQuadrilateral O A B C D)
variables (is_tangent : TangentToCircle O A P ∧ TangentToCircle O C P)
variables (not_on_extension : ¬ Collinear P D B)
variables (tangent_condition : PA ^ 2 = PB * PD)
variables (midpoint_AC : LineSegmentBisects AC BD)

theorem BD_bisects_AC (ABCD : CyclicQuadrilateral O A B C D)
  (is_tangent : TangentToCircle O A P ∧ TangentToCircle O C P)
  (not_on_extension : ¬ Collinear P D B)
  (tangent_condition : PA ^ 2 = PB * PD) : LineSegmentBisects AC BD :=
sorry

end BD_bisects_AC_l685_685083


namespace problem1_problem2_problem3_problem4_l685_685336

-- Proof problem 1
theorem problem1 : 4 - (-8) + (-6) = 6 :=
by sorry

-- Proof problem 2
theorem problem2 : 17 + 4 * (-5) - (1 / 2) * (1 / 2) = -3.25 :=
by sorry

-- Proof problem 3
theorem problem3 : -2^3 - ((1 / 3) - (5 / 6) + (3 / 4)) / (-1 / 24) = -2 :=
by sorry

-- Proof problem 4
theorem problem4 : |(sqrt 3) - 2| - sqrt((-2)^2) - sqrt(3) (64) = -4 - sqrt 3 :=
by sorry

end problem1_problem2_problem3_problem4_l685_685336


namespace tan_angle_QDE_l685_685196

theorem tan_angle_QDE (Q E F D : Type) 
  (DE : ℝ) (EF : ℝ) (FD : ℝ)
  (hDE : DE = 15)
  (hEF : EF = 16)
  (hFD : FD = 17)
  (hQ_inside : ∃ (θ : ℝ), θ = ∠QDE ∧ θ = ∠QEF ∧ θ = ∠QFD) :
  tan (∠QDE) = 168 / 385 :=
by
  let θ := ∠QDE
  have hθ_congruent : θ = ∠QEF ∧ θ = ∠QFD := by sorry
  -- assuming the angle congruence and the side lengths.
  let a := length (DQ)
  let b := length (EQ)
  let c := length (FQ)

  -- using given conditions and provided solution steps.
  sorry

end tan_angle_QDE_l685_685196


namespace barbie_bruno_trips_l685_685751

theorem barbie_bruno_trips (coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : 
  coconuts = 144 → barbie_capacity = 4 → bruno_capacity = 8 → (coconuts / (barbie_capacity + bruno_capacity) = 12) :=
by 
  intros h_coconuts h_barbie h_bruno
  rw [h_coconuts, h_barbie, h_bruno]
  norm_num
  sorry

end barbie_bruno_trips_l685_685751


namespace explicit_expression_l685_685129

variable {α : Type*} [LinearOrder α] {f : α → α}

/-- Given that the function satisfies a specific condition, prove the function's explicit expression. -/
theorem explicit_expression (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : 
  ∀ x, f x = 3 * x + 2 :=
by
  sorry

end explicit_expression_l685_685129


namespace trigonometric_identity_l685_685945

theorem trigonometric_identity (x y z a : ℝ) (h1 : (sin x + sin y + sin z) / sin (x + y + z) = a)
  (h2 : (cos x + cos y + cos z) / cos (x + y + z) = a) : 
  cos (x + y) + cos (y + z) + cos (z + x) = a :=
by
  sorry

end trigonometric_identity_l685_685945


namespace matrix_cubic_eqn_l685_685536

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, -2], ![2, 3]]

noncomputable def p : ℝ := -9
noncomputable def q : ℝ := 14

theorem matrix_cubic_eqn : A * A * A = p * A + q * (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end matrix_cubic_eqn_l685_685536


namespace sue_travel_time_correct_l685_685212

-- Define the flight and layover times as constants
def NO_to_ATL_flight_hours : ℕ := 2
def ATL_layover_hours : ℕ := 4
def ATL_to_CHI_flight_hours : ℕ := 5
def CHI_time_diff_hours : ℤ := -1
def CHI_layover_hours : ℕ := 3
def CHI_to_NY_flight_hours : ℕ := 3
def NY_time_diff_hours : ℤ := 1
def NY_layover_hours : ℕ := 16
def NY_to_DEN_flight_hours : ℕ := 6
def DEN_time_diff_hours : ℤ := -2
def DEN_layover_hours : ℕ := 5
def DEN_to_SF_flight_hours : ℕ := 4
def SF_time_diff_hours : ℤ := -1

-- Total time calculation including flights, layovers, and time zone changes
def total_travel_time_hours : ℕ :=
  NO_to_ATL_flight_hours +
  ATL_layover_hours +
  (ATL_to_CHI_flight_hours + CHI_time_diff_hours).toNat +  -- Handle time difference (ensure non-negative)
  CHI_layover_hours +
  (CHI_to_NY_flight_hours + NY_time_diff_hours).toNat +
  NY_layover_hours +
  (NY_to_DEN_flight_hours + DEN_time_diff_hours).toNat +
  DEN_layover_hours +
  (DEN_to_SF_flight_hours + SF_time_diff_hours).toNat

-- Statement to prove in Lean:
theorem sue_travel_time_correct : total_travel_time_hours = 45 :=
by {
  -- Skipping proof details since only the statement is required
  sorry
}

end sue_travel_time_correct_l685_685212


namespace factor_difference_of_squares_l685_685370

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l685_685370


namespace find_angle_MLP_l685_685215

variables {A B C : ℝ}
variables (AB AC BC OA OB : set ℝ)
variables (O : point) (M K L N P : point)
axioms 
  (angle_ABC_acute : ∀ {A B C : ℝ}, is_acute (angle A B C))
  (O_circumcenter : circumcenter ABC O)
  (M_minor_arc_AB : M ∈ minor_arc AB)
  (K_on_AB : K ∈ (line_through M OA ∩ AB))
  (L_on_AC : L ∈ (line_through M OA ∩ AC))
  (N_on_AB : N ∈ (line_through M OB ∩ AB))
  (P_on_BC : P ∈ (line_through M OB ∩ BC))
  (MN_eq_KL : distance M N = distance K L)
  
theorem find_angle_MLP (A B C : ℝ) :
  angle M L P = angle A B C :=
sorry

end find_angle_MLP_l685_685215


namespace intersecting_circle_area_ratio_l685_685313

theorem intersecting_circle_area_ratio (R : ℝ) (r : ℝ) 
  (h : R^2 = (1/2 * R)^2 + r^2) :
  (π * r^2) / (4 * π * R^2) = 3/16 := 
by
  have h' : r^2 = 3/4 * R^2 :=
    by rw [←h, sq, (1/2 : ℝ), mul_assoc, sq, mul_comm]; ring
  rw [h', pi_mul, div_mul_eq_mul_div, ←mul_assoc, mul_div, div_self pi_ne_zero, mul_comm, div_mul_btwn_all_eq]
  exact rfl

end intersecting_circle_area_ratio_l685_685313


namespace purely_imaginary_iff_eq_zero_compute_expression_when_m_eq_2_l685_685339

-- Part (I) : Proof that z is purely imaginary iff m = 0 with given conditions.
theorem purely_imaginary_iff_eq_zero (m : ℝ) :
  (∃ z : ℂ, z = m * (m - 1) + (m - 1) * complex.I ∧ z.re = 0) ↔ m = 0 :=
sorry

-- Part (II) : Proof of the simplified expression when m = 2.
theorem compute_expression_when_m_eq_2 :
  let z := (2 : ℂ) + complex.I
  in complex.conj z - (z / (1 + complex.I)) = (1 / 2) - (1 / 2) * complex.I :=
sorry

end purely_imaginary_iff_eq_zero_compute_expression_when_m_eq_2_l685_685339


namespace complete_the_square_l685_685649

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l685_685649


namespace max_square_plots_l685_685710
-- Lean 4 statement for the equivalent math problem

theorem max_square_plots (w l f s : ℕ) (h₁ : w = 40) (h₂ : l = 60) 
                         (h₃ : f = 2400) (h₄ : s ≠ 0) (h₅ : 2400 - 100 * s ≤ 2400)
                         (h₆ : w % s = 0) (h₇ : l % s = 0) :
  (w * l) / (s * s) = 6 :=
by {
  sorry
}

end max_square_plots_l685_685710


namespace solve_trig_equation_l685_685970

noncomputable def trigonometric_equation_solution (x : ℝ) : Prop :=
  ∃ k : ℤ, (x = (↑k * Real.pi / 1010) ∧ ¬ ( (1010 : ℤ) ∣ k)) ∨ (x = (Real.pi + 4 * ↑k * Real.pi) / 4040)

theorem solve_trig_equation (x : ℝ) (h : (∑ k in Finset.range 1010, Real.sin ((2 * k + 1 : ℕ) * x))
                                 = (∑ k in Finset.range 1010, Real.cos ((2 * k + 1 : ℕ) * x))) :
  trigonometric_equation_solution x :=
by
  sorry

end solve_trig_equation_l685_685970


namespace ratio_proof_l685_685900

variable (x y z : ℝ)
variable (h1 : y / z = 1 / 2)
variable (h2 : z / x = 2 / 3)
variable (h3 : x / y = 3 / 1)

theorem ratio_proof : (x / (y * z)) / (y / (z * x)) = 4 / 1 := 
  sorry

end ratio_proof_l685_685900


namespace max_blue_balls_l685_685907

theorem max_blue_balls (r b : ℕ) (h1 : r + b = 72) (h2 : ∃ p : ℕ, p.prime ∧ r = b + p) : b ≤ 35 :=
sorry

end max_blue_balls_l685_685907


namespace range_of_a_l685_685493

open Real

theorem range_of_a :
  (∃ x > 0, 2 ^ x * (x - a) < 1) ↔ a > -1 :=
by
  sorry

end range_of_a_l685_685493


namespace problem1_problem2_l685_685837

-- Definitions based on given conditions
variable {P A B C D E F H: Type}
variable (isRhombus : Rhombus ABCD) -- assuming Rhombus defined elsewhere
variable (PA_perp_ABCD : Perp PA (Plane ABCD)) -- assuming Perp and Plane defined elsewhere
variable (angle_ABC_60 : angle ABC = 60) -- assuming a way to represent angles
variable (E_mid_BC : Midpoint E BC) -- assuming Midpoint defined elsewhere
variable (F_mid_PC : Midpoint F PC)
variable (H_on_PD : On H PD) -- assuming On defined elsewhere
variable (tan_EH_PAD_max : max_tangent_angle EH (Plane PAD) = (sqrt 6) / 2)
-- Proving Problem 1: AE ⊥ PD
theorem problem1 : Perp AE PD := sorry
-- Proving Problem 2: Find the cosine value of the dihedral angle E-ADF
theorem problem2 : cosine_dihedral_angle E A D F = (sqrt 15) / 5  := sorry

end problem1_problem2_l685_685837


namespace barbie_bruno_trips_l685_685753

theorem barbie_bruno_trips (coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : 
  coconuts = 144 → barbie_capacity = 4 → bruno_capacity = 8 → (coconuts / (barbie_capacity + bruno_capacity) = 12) :=
by 
  intros h_coconuts h_barbie h_bruno
  rw [h_coconuts, h_barbie, h_bruno]
  norm_num
  sorry

end barbie_bruno_trips_l685_685753


namespace friend_P_distance_l685_685632

theorem friend_P_distance (v t : ℝ) (hv : v > 0)
  (distance_trail : 22 = (1.20 * v * t) + (v * t))
  (h_t : t = 22 / (2.20 * v)) : 
  (1.20 * v * t = 12) :=
by
  sorry

end friend_P_distance_l685_685632


namespace concentration_after_5500_evaporates_l685_685316

noncomputable def concentration_after_evaporation 
  (V₀ Vₑ : ℝ) (C₀ : ℝ) : ℝ := 
  let sodium_chloride := C₀ * V₀
  let remaining_volume := V₀ - Vₑ
  100 * sodium_chloride / remaining_volume

theorem concentration_after_5500_evaporates 
  : concentration_after_evaporation 10000 5500 0.05 = 11.11 := 
by
  -- Formalize the calculations as we have derived
  -- sorry is used to skip the proof
  sorry

end concentration_after_5500_evaporates_l685_685316


namespace rook_polynomial_of_chessboard_l685_685065

-- Define the specific chessboard configuration
def chessboard_config : List (List (Option Nat)) :=
[
  [none, none, none, some 1],
  [none, none, some 1, some 1],
  [some 1, some 1, some 8, none],
  [some 1, some 1, none, none],
  [some 1, none, none, none]
]

-- Define the rook polynomial calculation and the expected value
theorem rook_polynomial_of_chessboard :
  computeRookPolynomial(chessboard_config) = (1 + 10 * t + 25 * t^2 + 24 * t^3 + 6 * t^4) :=
  sorry

end rook_polynomial_of_chessboard_l685_685065


namespace angle_GDA_is_135_l685_685514

-- Definitions of conditions
def is_isosceles_right_triangle (C D E : Point) : Prop :=
  ∠CDE = 45 ∧ ∠DEC = 45 ∧ ∠ECD = 90

def is_square (A B C D : Point) : Prop :=
  ∀ {a b c d : Point}, is_square_ABCDE_true [A, B, C, D]

-- The main theorem
theorem angle_GDA_is_135
  {A B C D E F G : Point} 
  (h1 : is_isosceles_right_triangle C D E)
  (h2 : is_square A B C D)
  (h3 : is_square D E F G) :
  ∠GDA = 135 :=
sorry

end angle_GDA_is_135_l685_685514


namespace construction_teams_l685_685704

theorem construction_teams (
  work_planned_days : ℕ,
  work_done_first_team_days : ℕ,
  early_completion_days : ℕ,
  combined_days_worked : ℕ,
  work_done_fraction : ℚ,
  remaining_work_fraction : ℚ,
  work_done_combined : ℚ
) : 
  work_planned_days = 30 →
  work_done_first_team_days = 10 →
  early_completion_days = 8 →
  combined_days_worked = 12 →
  work_done_fraction = 1 / 3 →
  remaining_work_fraction = 2 / 3 →
  work_done_combined = 1 →
  let x := (12 * 15) / 4 in
  x = 45 ∧
  let combined_rate := (1 / 30) + (1 / 45) in
  let total_days_for_both := 1 / combined_rate in
  total_days_for_both = 18 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  let x := 45
  have hx : x = 45 := by sorry
  have combined_rate := (1 / 30) + (1 / 45)
  have total_days_for_both := 1 / combined_rate
  have htb : total_days_for_both = 18 := by sorry
  exact ⟨hx, htb⟩

end construction_teams_l685_685704


namespace marble_arrangement_l685_685320

def arrange_marbles : Prop :=
  ∃ N : ℕ, 
  let m := 5 in
  let num_ways := (Nat.choose (5 + 8) 5) in
  (num_ways % 1000 = 287)

theorem marble_arrangement (h₁ : 4 = 4) (h₂ : 3 = 3) : arrange_marbles := 
by
  sorry

end marble_arrangement_l685_685320


namespace rook_paths_on_chessboard_l685_685782

/-- Definition of the Catalan number, C(n) --/
def catalan (n : ℕ) : ℕ :=
  if n = 0 then 1 else (2 * (2 * n - 1) * catalan (n - 1)) / (n + 1)

theorem rook_paths_on_chessboard (n : ℕ) : ℕ :=
  catalan (n - 1)

end rook_paths_on_chessboard_l685_685782


namespace smallest_sum_of_table_l685_685518

open Nat

theorem smallest_sum_of_table :
  ∀ (table : Matrix (Fin 4) (Fin 6) ℕ), 
    (∀ i j, i ≠ j → row_sum table i ≠ row_sum table j ∧ col_sum table i ≠ col_sum table j) → 
    ∑ i j, table i j = 43 := 
sorry

end smallest_sum_of_table_l685_685518


namespace max_value_of_f_l685_685389

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ x, (f x = 1 / exp 1) ∧ (∀ y, f y ≤ f x) :=
by
  sorry

end max_value_of_f_l685_685389


namespace vector_magnitude_b_l685_685881

variable (a : ℝ × ℝ × ℝ := (-1, 2, -3))
variable (b : ℝ × ℝ × ℝ := (-4, -1, 2))

theorem vector_magnitude_b : 
  real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2) = real.sqrt 21 :=
by
  sorry

end vector_magnitude_b_l685_685881


namespace f_m_eq_five_l685_685861

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five_l685_685861


namespace variance_of_data_set_l685_685087

theorem variance_of_data_set :
  let data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]
  let n := data.length
  let mean := data.sum / n
  let squared_diffs := data.map (λ x => (x - mean)^2)
  let variance := squared_diffs.sum / n
  variance = 0.02 := by
    sorry

end variance_of_data_set_l685_685087


namespace compound_interest_calculation_l685_685724

-- Define the variables used in the problem
def principal : ℝ := 8000
def annual_rate : ℝ := 0.05
def compound_frequency : ℕ := 1
def final_amount : ℝ := 9261
def years : ℝ := 3

-- Statement we need to prove
theorem compound_interest_calculation :
  final_amount = principal * (1 + annual_rate / compound_frequency) ^ (compound_frequency * years) :=
by 
  sorry

end compound_interest_calculation_l685_685724


namespace triangle_area_change_l685_685154

theorem triangle_area_change 
  (PQ PR QR : ℝ) 
  (hPQ : PQ = 8) 
  (hPR : PR = 15) 
  (hQR : QR = 17) 
  (PQ' : ℝ) (hPQ' : PQ' = 2 * PQ) 
  (PR' : ℝ) (hPR' : PR' = 1.5 * PR) 
  (QR' : ℝ) (hQR' : QR' = QR) :
  let area_original := (let s := (PQ + PR + QR) / 2 in
    real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))),
      area_new := (let s' := (PQ' + PR' + QR') / 2 in
    real.sqrt (s' * (s' - PQ') * (s' - PR') * (s' - QR')))
  in (area_new > 2 * area_original) ∧ (area_new < 3 * area_original) := sorry

end triangle_area_change_l685_685154


namespace tim_nickels_count_l685_685252

-- Definitions of the conditions
def dimes_value (n : ℕ) := n * 0.10
def half_dollars_value (n : ℕ) := n * 0.50
def total_value (dimes_shined dimes_tip half_dollars_tip : ℕ) :=
  dimes_value dimes_shined + dimes_value dimes_tip + half_dollars_value half_dollars_tip

-- Main theorem statement
theorem tim_nickels_count (nickels_shined dimes_shined dimes_tip half_dollars_tip : ℕ)
  (h₁ : dimes_shined = 13)
  (h₂ : dimes_tip = 7)
  (h₃ : half_dollars_tip = 9)
  (h₄ : total_value dimes_shined dimes_tip half_dollars_tip + nickels_shined * 0.05 = 6.65) :
  nickels_shined = 3 :=
sorry

end tim_nickels_count_l685_685252


namespace coefficient_x3_in_expansion_l685_685436

theorem coefficient_x3_in_expansion :
  (∃ c : ℝ, c * x^3 = coeff_of_x3 (2 * x + sqrt x) ^ 5) ∧ c = 10 :=
sorry

end coefficient_x3_in_expansion_l685_685436


namespace range_of_n_l685_685976

theorem range_of_n (m n : ℝ) (h : (m^2 - 2 * m)^2 + 4 * m^2 - 8 * m + 6 - n = 0) : n ≥ 3 :=
sorry

end range_of_n_l685_685976


namespace comparison_of_exponential_values_l685_685057

theorem comparison_of_exponential_values : 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  c < a ∧ a < b := 
by 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  sorry

end comparison_of_exponential_values_l685_685057


namespace barbie_bruno_trips_l685_685752

theorem barbie_bruno_trips (coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : 
  coconuts = 144 → barbie_capacity = 4 → bruno_capacity = 8 → (coconuts / (barbie_capacity + bruno_capacity) = 12) :=
by 
  intros h_coconuts h_barbie h_bruno
  rw [h_coconuts, h_barbie, h_bruno]
  norm_num
  sorry

end barbie_bruno_trips_l685_685752


namespace find_x0_l685_685094

noncomputable def f : ℝ → ℝ := λ x, 13 - 8 * x + real.sqrt 2 * x^2

noncomputable def f_prime : ℝ → ℝ := λ x, -8 + 2 * real.sqrt 2 * x

theorem find_x0 :
  ∃ x0 : ℝ, f_prime x0 = 4 ↔ x0 = 3 * real.sqrt 2 := by
  sorry

end find_x0_l685_685094


namespace parallel_lines_condition_l685_685275

theorem parallel_lines_condition (a : ℝ) :
  (a = 3 / 2) ↔ (∀ x y : ℝ, (x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → (a = 3 / 2)) :=
sorry

end parallel_lines_condition_l685_685275


namespace chebyshevs_inequality_generalized_l685_685291

theorem chebyshevs_inequality_generalized 
  {n : ℕ} {a b : ℕ → ℝ} {m t r : ℝ}
  (ha_pos : ∀ i, 0 < a i)
  (hb_pos : ∀ i, 0 < b i)
  (ha_order : (∀ i j, i ≤ j → a i ≥ a j) ∨ (∀ i j, i ≤ j → a i ≤ a j))
  (hb_order : (∀ i j, i ≤ j → b i ≥ b j) ∨ (∀ i j, i ≤ j → b i ≤ b j))
  (hn : 2 ≤ n)
  (hr_pos : 0 < r)
  (ht_pos : 0 < t)
  (hmt_pos : 0 < m - t) :
  ∑ i in Finset.range n, a i * b i 
  ≥ (1 / (m - t)) * ∑ i in Finset.range n, (r * ∑ j in Finset.range n, a j - t * a i) * b i :=
by
  sorry

end chebyshevs_inequality_generalized_l685_685291


namespace vector_expression_simplification_l685_685050

variables (A B C D : Point)

def vector_sub (u v : Vector) : Vector := u + (-v)

theorem vector_expression_simplification :
  vector_sub ((B - A) + (C - B) - (C - D) + 2 * (A - D)) = (A - D) :=
by
  sorry

end vector_expression_simplification_l685_685050


namespace rectangle_area_constant_k_l685_685308

theorem rectangle_area_constant_k 
  (length width diagonal : ℝ) 
  (h_ratio : length = 5/4 * width) 
  (h_diagonal : diagonal = 13) : 
  ∃ k : ℝ, k = 20 / 41 :=
by 
  let x := 1 in
  have h_diag_eq : 13 ^ 2 = (5 * x)^2 + (4 * x)^2, from sorry,
  have h_area : (5 * x) * (4 * x) = 20 * x^2, from sorry,
  show ∃ k : ℝ, k = 20 / 41, from
    have h_k : 20 * x^2 = k * (13^2), from sorry,
    have k_val : k = 20 / 41, from sorry,
    ⟨k, k_val⟩

end rectangle_area_constant_k_l685_685308


namespace cost_price_as_percentage_l685_685595

theorem cost_price_as_percentage (SP CP : ℝ) 
  (profit_percentage : ℝ := 4.166666666666666) 
  (P : ℝ := SP - CP)
  (profit_eq : P = (profit_percentage / 100) * SP) :
  CP = (95.83333333333334 / 100) * SP := 
by
  sorry

end cost_price_as_percentage_l685_685595


namespace constant_term_expansion_l685_685216

theorem constant_term_expansion :
  (∀r, ∀n : ℕ, n = 10 ∧ 5 - (5 * r) / 2 = 0 → (binomial n r) * (2^r) = 180) :=
begin
  intros r n,
  assume h1 : n = 10,
  assume h2 : 5 - (5 * r) / 2 = 0,
  sorry  -- Proof not required
end

end constant_term_expansion_l685_685216


namespace arithmetic_sequence_cosine_ratio_l685_685075

theorem arithmetic_sequence_cosine_ratio :
  (∀ n : ℤ, a_ n = a_8 + (n - 8) * (a_9 - a_8))
  → a_8 = 8
  → a_9 = 8 + real.pi / 3
  → (cos (a_5) + cos (a_7)) / cos (a_6) = 1 :=
by
  intro h_a_seq ha8 ha9
  sorry

end arithmetic_sequence_cosine_ratio_l685_685075


namespace find_sample_size_l685_685314

def sports_team (total: Nat) (soccer: Nat) (basketball: Nat) (table_tennis: Nat) : Prop :=
  total = soccer + basketball + table_tennis

def valid_sample_size (total: Nat) (n: Nat) :=
  (n > 0) ∧ (total % n == 0) ∧ (n % 6 == 0)

def systematic_sampling_interval (total: Nat) (n: Nat): Nat :=
  total / n

theorem find_sample_size :
  ∀ (total soccer basketball table_tennis: Nat),
  sports_team total soccer basketball table_tennis →
  total = 36 →
  soccer = 18 →
  basketball = 12 →
  table_tennis = 6 →
  (∃ n, valid_sample_size total n ∧ valid_sample_size (total - 1) (n + 1)) →
  ∃ n, n = 6 := by
  sorry

end find_sample_size_l685_685314


namespace largest_number_composed_of_3_and_2_summing_to_11_l685_685664

theorem largest_number_composed_of_3_and_2_summing_to_11 (n : ℕ) : 
  (∃ l : List ℕ, (∀ d ∈ l, d = 3 ∨ d = 2) ∧ l.sum = 11 ∧ l.join_digits = n) → n = 32222 :=
by
  sorry

end largest_number_composed_of_3_and_2_summing_to_11_l685_685664


namespace carbon_neutrality_l685_685687

theorem carbon_neutrality (a b : ℝ) (t : ℕ) (ha : a > 0)
  (h1 : S = a * b ^ t)
  (h2 : a * b ^ 7 = 4 * a / 5)
  (h3 : a / 4 = S) :
  t = 42 := 
sorry

end carbon_neutrality_l685_685687


namespace parametric_curve_intersects_l685_685746

noncomputable def curve_crosses_itself : Prop :=
  let t1 := Real.sqrt 11
  let t2 := -Real.sqrt 11
  let x (t : ℝ) := t^3 - t + 1
  let y (t : ℝ) := t^3 - 11*t + 11
  (x t1 = 10 * Real.sqrt 11 + 1) ∧ (y t1 = 11) ∧
  (x t2 = 10 * Real.sqrt 11 + 1) ∧ (y t2 = 11)

theorem parametric_curve_intersects : curve_crosses_itself :=
by
  sorry

end parametric_curve_intersects_l685_685746


namespace smallest_positive_integer_n_l685_685772

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), (n > 0) ∧ 
  (∑ k in finset.range (n + 1), real.logb 3 (1 + 1 / 3^(3^k))) ≥ 1 + real.logb 3 (3000 / 3001) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (∑ k in finset.range (m + 1), real.logb 3 (1 + 1 / 3^(3^k))) < 1 + real.logb 3 (3000 / 3001)) := 
sorry

end smallest_positive_integer_n_l685_685772


namespace original_number_of_cards_l685_685569

theorem original_number_of_cards (X : ℕ) :
  (∀ (sasha_added karen_took_out remaining_cards : ℕ),
    sasha_added = 48 →
    karen_took_out = sasha_added / 6 →
    remaining_cards = X + sasha_added - karen_took_out →
    remaining_cards = 83) →
  X = 43 :=
by
  assume h
  have sasha_added := 48
  have karen_took_out := sasha_added / 6
  have remaining_cards := X + sasha_added - karen_took_out
  have eq1 : sasha_added = 48 := rfl
  have eq2 : karen_took_out = sasha_added / 6 := rfl
  have eq3 : remaining_cards = X + 48 - 8 := by
    simp [karen_took_out]
    norm_num
    rfl
  have eq4 : remaining_cards = 83 := h 48 karen_took_out remaining_cards eq1 eq2 eq3
  have eq5 : X + 40 = 83 := by
    simp [eq3] at eq4
    exact eq4
  have eq6 : X = 83 - 40 := by
    simp [eq5]
    norm_num
  have eq7 : X = 43 := by
    simp [eq6]
    norm_num
  exact eq7

#check original_number_of_cards

end original_number_of_cards_l685_685569


namespace find_base_k_l685_685827

theorem find_base_k (k : ℕ) (h1 : 1 + 3 * k + 2 * k^2 = 30) : k = 4 :=
by sorry

end find_base_k_l685_685827


namespace neg_p_l685_685996

-- Let's define the original proposition p
def p : Prop := ∃ x : ℝ, x ≥ 2 ∧ x^2 - 2 * x - 2 > 0

-- Now, we state the problem in Lean as requiring the proof of the negation of p
theorem neg_p : ¬p ↔ ∀ x : ℝ, x ≥ 2 → x^2 - 2 * x - 2 ≤ 0 :=
by
  sorry

end neg_p_l685_685996


namespace arrangement_proof_l685_685731

def num_arrangements : ℕ :=
  let teacher_positions := 3  -- Teacher has 3 positions to stand (not ends)
  let girl_unit_positions := 3!  -- Arrange the unit of girls and two boys
  let girl_switch := 2  -- Girls within the unit can switch places
  teacher_positions * girl_unit_positions * girl_switch

theorem arrangement_proof : num_arrangements = 24 := by
  sorry

end arrangement_proof_l685_685731


namespace derivative_sin_over_x_l685_685386

noncomputable def differentiate_sin_over_x (x : ℝ) (x ≠ 0) : ℝ := derivative (λ x, sin x / x) x

theorem derivative_sin_over_x (x : ℝ) (h : x ≠ 0) :
  differentiate_sin_over_x x h = (x * cos x - sin x) / x^2 :=
sorry

end derivative_sin_over_x_l685_685386


namespace trapezoid_perimeter_l685_685343

def Base1 : ℕ := 10
def Base2 : ℕ := 14
def Side1 : ℕ := 9
def Side2 : ℕ := 9
def Perimeter : ℕ := Base1 + Base2 + Side1 + Side2

theorem trapezoid_perimeter : Perimeter = 42 := 
by
  unfold Perimeter Base1 Base2 Side1 Side2
  simp
  sorry

end trapezoid_perimeter_l685_685343


namespace max_rectangle_area_l685_685739

theorem max_rectangle_area (P : ℝ) (hP : 0 < P) : 
  ∃ (x y : ℝ), (2*x + 2*y = P) ∧ (x * y = P ^ 2 / 16) :=
by
  sorry

end max_rectangle_area_l685_685739


namespace find_pairs_of_natural_numbers_l685_685037

theorem find_pairs_of_natural_numbers (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔ (m, n) = (1, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (3, 2) :=
by
  sorry

end find_pairs_of_natural_numbers_l685_685037


namespace cosine_angle_l685_685459

open Real EuclideanSpace

variables (u v : ℝ^3)
variables (φ : ℝ)

theorem cosine_angle (h1 : ‖u‖ = 5) (h2 : ‖v‖ = 7) (h3 : ‖u + v‖ = 9) : 
  cos (∠ (u, v)) = 1 / 10 := by 
sorry

end cosine_angle_l685_685459


namespace foundation_cost_l685_685924

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end foundation_cost_l685_685924


namespace arithmetic_sequence_vertex_l685_685120

variables {a b c d : ℤ}
variable f : ℤ → ℤ

def is_arithmetic_sequence (a b c d : ℤ) : Prop :=
  b = a + (d - a) / 3 ∧ c = a + 2 * (d - a) / 3

def f_vertex (f : ℤ → ℤ) (a d : ℤ) : Prop :=
  ∀ x, f x = x^2 - 2 * x ∧ d = f a

theorem arithmetic_sequence_vertex (h1 : is_arithmetic_sequence a b c d)
 (h2 : f_vertex f a d) : b + c = 0 :=
sorry

end arithmetic_sequence_vertex_l685_685120


namespace range_fraction_proof_l685_685077

noncomputable def range_fraction (a b : ℝ) : ℝ :=
a^2 / (2 + b)

theorem range_fraction_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∀ y : ℝ, (y = range_fraction a b) → y ∈ Ioo 0 (1/2) :=
by
  sorry

end range_fraction_proof_l685_685077


namespace marks_age_more_than_thrice_aarons_l685_685630

theorem marks_age_more_than_thrice_aarons :
  ∃ (A : ℕ)(X : ℕ), 28 = A + 17 ∧ 25 = 3 * (A - 3) + X ∧ 32 = 2 * (A + 4) + 2 ∧ X = 1 :=
by
  sorry

end marks_age_more_than_thrice_aarons_l685_685630


namespace average_shift_l685_685425

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem average_shift (h : (∑ i, x i) / n = 2) : (∑ i, x i + 2) / n = 4 :=
by
  sorry

end average_shift_l685_685425


namespace factor_difference_of_squares_l685_685373

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l685_685373


namespace limit_Sn_div_an_l685_685544

noncomputable def a_n (n : ℕ) : ℝ := 9^n
noncomputable def S_n (n : ℕ) : ℝ := 4 * 9^(n-1) * (1 - (1 / 10)^n)

theorem limit_Sn_div_an : 
  tendsto (λ n, (S_n n) / (a_n n)) atTop (𝓝 (4 / 9)) :=
by
  sorry

end limit_Sn_div_an_l685_685544


namespace determine_exponents_l685_685711

def geometric_sequence_second_term_product
(first_term : ℕ) (last_term : ℕ) (rational_common_ratio : Prop) (x y : ℕ) : Prop :=
first_term = 32^16 ∧ last_term = 625^30 ∧ rational_common_ratio →
(exists sequence, ∀ a b ∈ sequence, 
  a * b = 16^x * 625^y ∧ x = 214 ∧ y = 69)

theorem determine_exponents :
  geometric_sequence_second_term_product 32^16 625^30 (∃ a ∈ ℚ) x y :=
sorry

end determine_exponents_l685_685711


namespace three_digit_number_uniq_l685_685673

theorem three_digit_number_uniq (n : ℕ) (h : 100 ≤ n ∧ n < 1000)
  (hundreds_digit : n / 100 = 5) (units_digit : n % 10 = 3)
  (div_by_9 : n % 9 = 0) : n = 513 :=
sorry

end three_digit_number_uniq_l685_685673


namespace f_neg_seven_eq_neg_four_l685_685540

noncomputable def f : ℝ → ℝ := λ x, if x > 0 then Real.log (x + 9) / Real.log 2 else -Real.log (-x + 9) / Real.log 2

theorem f_neg_seven_eq_neg_four : f (-7) = -4 := by
  -- We assert that f is odd and provide the function definition for positive x.
  have h1 : ∀ x : ℝ, f (-x) = -f(x) := by sorry
  -- Now, we state that for positive x, f is defined as given.
  have h2 : ∀ x : ℝ, x > 0 → f(x) = Real.log (x + 9) / Real.log 2 := by sorry
  -- Using the two conditions to prove f(-7) = -4.
  sorry

end f_neg_seven_eq_neg_four_l685_685540


namespace no_sum_of_squares_of_8n_plus_7_l685_685198

theorem no_sum_of_squares_of_8n_plus_7 (n : ℕ) : ∀ x y z : ℤ, x^2 + y^2 + z^2 ≠ 8 * n + 7 :=
by
  sorry

end no_sum_of_squares_of_8n_plus_7_l685_685198


namespace line_passing_through_midpoints_l685_685929

variables {A B C D P O : Type} [EuclideanGeometry]

/-- Define the quadrilateral ABCD --/
variables (ABCD : Quadrilateral A B C D)
variables (AB_parallel_CD : Parallel (Line.mk A B) (Line.mk C D))
variables (AB_gt_CD : length (Segment.mk A B) > length (Segment.mk C D))
variables (O : Point) (P : Point)

-- Define the intersections
variables (O_eq_inter_AC_BD : O = intersection (Line.mk A C) (Line.mk B D))
variables (P_eq_inter_AD_BC : P = intersection (Line.mk A D) (Line.mk B C))

theorem line_passing_through_midpoints :
  passes_through_midpoints (Line.mk O P) (Segment.mk A B) (Segment.mk C D) :=
sorry

end line_passing_through_midpoints_l685_685929


namespace abs_pi_abs_pi_minus_10_l685_685012

def pi := Real.pi

theorem abs_pi_abs_pi_minus_10 (h : pi < 10) : |pi - |pi - 10|| = 10 - 2 * pi := by
  sorry

end abs_pi_abs_pi_minus_10_l685_685012


namespace quad_eq_sum_ab_l685_685871

theorem quad_eq_sum_ab {a b : ℝ} (h1 : a < 0)
  (h2 : ∀ x : ℝ, (x = -1 / 2 ∨ x = 1 / 3) ↔ ax^2 + bx + 2 = 0) :
  a + b = -14 :=
by
  sorry

end quad_eq_sum_ab_l685_685871


namespace x_pow_n_sub_inv_x_pow_n_value_l685_685547

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π)
variable {x : ℂ} (h3 : x - x⁻¹ = 2 * complex.I * complex.sin θ)
variable {n : ℕ} (h4 : 0 < n)

theorem x_pow_n_sub_inv_x_pow_n_value :
  x^n - x⁻¹^n = 2 * complex.I * complex.sin (n * θ) :=
sorry

end x_pow_n_sub_inv_x_pow_n_value_l685_685547


namespace factor_difference_of_squares_l685_685374

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l685_685374


namespace jack_bill_age_difference_l685_685599

def jack_bill_ages_and_difference (a b : ℕ) :=
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  (a + b = 2) ∧ (7 * a - 29 * b = 14) → jack_age - bill_age = 18

theorem jack_bill_age_difference (a b : ℕ) (h₀ : a + b = 2) (h₁ : 7 * a - 29 * b = 14) : 
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  jack_age - bill_age = 18 :=
by {
  sorry
}

end jack_bill_age_difference_l685_685599


namespace selection_methods_count_l685_685816

theorem selection_methods_count : 
  (nat.choose 5 4) * (nat.choose 4 2 / 2) * (2) = 30 := 
by
  -- proof goes here
  sorry

end selection_methods_count_l685_685816


namespace total_interest_l685_685750

def P : ℝ := 1000
def r : ℝ := 0.1
def n : ℕ := 3

theorem total_interest : (P * (1 + r)^n) - P = 331 := by
  sorry

end total_interest_l685_685750


namespace simplify_and_evaluate_l685_685967

variable (x y : ℤ)

noncomputable def given_expr := (x + y) ^ 2 - 3 * x * (x + y) + (x + 2 * y) * (x - 2 * y)

theorem simplify_and_evaluate : given_expr 1 (-1) = -3 :=
by
  -- The proof is to be completed here
  sorry

end simplify_and_evaluate_l685_685967


namespace isabella_euros_l685_685525

theorem isabella_euros (d : ℝ) : 
  (5 / 8) * d - 80 = 2 * d → d = 58 :=
by
  sorry

end isabella_euros_l685_685525


namespace area_correct_l685_685992

-- Define the conditions provided in the problem
def width (w : ℝ) := True
def length (l : ℝ) := True
def perimeter (p : ℝ) := True

-- Add the conditions about the playground
axiom length_exceeds_width_by : ∃ l w, l = 3 * w + 30
axiom perimeter_is_given : ∃ l w, 2 * (l + w) = 730

-- Define the area of the playground and state the theorem
noncomputable def area_of_playground : ℝ := 83.75 * 281.25

theorem area_correct :
  (∃ l w, l = 3 * w + 30 ∧ 2 * (l + w) = 730) →
  area_of_playground = 23554.6875 :=
by
  sorry

end area_correct_l685_685992


namespace chips_heavier_than_juice_l685_685622

theorem chips_heavier_than_juice :
  (∀ b j : ℕ, 2 * b = 800 ∧ 5 * b + 4 * j = 2200) → (∃ d : ℕ, d = 350) :=
by {
  intro h,
  cases h with b j hj,
  existsi b - j,
  sorry
}

end chips_heavier_than_juice_l685_685622


namespace find_2a_minus_b_plus_c_l685_685224

theorem find_2a_minus_b_plus_c :
  (∀ x y : ℝ, y = 2*x^2 - 3*x + 4 → (1, 3) = (x, y)) →
  ∃ a b c : ℤ, a = 2 ∧ b = -3 ∧ c = 4 ∧ 2*a - b + c = 11 :=
by
  intro h
  existsi [2, -3, 4]
  simp
  sorry

end find_2a_minus_b_plus_c_l685_685224


namespace max_n_of_sequence_l685_685454

theorem max_n_of_sequence (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, sqrt (a (n + 1)) - sqrt (a n) = 1) : max (λ n, a n < 32) = 5 := 
sorry

end max_n_of_sequence_l685_685454


namespace expression_value_l685_685270

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l685_685270


namespace hyperbola_center_l685_685303

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 )

theorem hyperbola_center (f1 f2 : ℝ × ℝ) (c : ℝ × ℝ) 
  (h1 : f1 = (-3, 2)) (h2 : f2 = (7, -6)) (h3 : c = midpoint f1 f2) : 
  c = (2, -2) :=
by
  sorry

end hyperbola_center_l685_685303


namespace largest_of_three_numbers_l685_685991

noncomputable def hcf (a b : ℕ) := sorry -- define hcf
noncomputable def lcm (a b : ℕ) := sorry -- define lcm

theorem largest_of_three_numbers (a b c : ℕ) (hcf_23 : hcf (hcf a b) c = 23) 
  (lcm_factors : lcm (lcm a b) c = 23 * 13 * 19 * 17) :
  a = 23 * 19 ∨ b = 23 * 19 ∨ c = 23 * 19 :=
sorry -- proof omitted

end largest_of_three_numbers_l685_685991


namespace principal_is_869_65_l685_685394

noncomputable def calculate_principal : ℝ := 13 / 0.01496

theorem principal_is_869_65 : calculate_principal ≈ 869.65 :=
by
  -- Definition of interest conditions
  let compound_interest_amount_2_years (P : ℝ) : ℝ := P * 1.10 * (1 + 0.06)^2
  let simple_interest_amount_2_years (P : ℝ) : ℝ := P * 0.22
  let interest_difference_eq_13 (P : ℝ) : Prop := compound_interest_amount_2_years P - P - simple_interest_amount_2_years P = 13

  -- Proving principal amount computation
  calc
    calculate_principal = 13 / 0.01496             : by rfl
                   ... ≈ 869.65                    : by norm_num1

end principal_is_869_65_l685_685394


namespace multiple_of_Mel_weight_l685_685762

/-- Given that Brenda weighs 10 pounds more than a certain multiple of Mel's weight,
    and given that Brenda weighs 220 pounds and Mel's weight is 70 pounds,
    show that the multiple is 3. -/
theorem multiple_of_Mel_weight 
    (Brenda_weight Mel_weight certain_multiple : ℝ) 
    (h1 : Brenda_weight = Mel_weight * certain_multiple + 10)
    (h2 : Brenda_weight = 220)
    (h3 : Mel_weight = 70) :
  certain_multiple = 3 :=
by 
  sorry

end multiple_of_Mel_weight_l685_685762


namespace geometric_sum_formula_l685_685407

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  sorry

theorem geometric_sum_formula (a : ℕ → ℕ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_a1_a2 : a 0 + a 1 = 3)
  (h_a1_a2_a3 : a 0 * a 1 * a 2 = 8) :
  geometric_sequence_sum n = 2^n - 1 :=
sorry

end geometric_sum_formula_l685_685407


namespace distance_from_polar_to_circle_center_l685_685915

noncomputable def polarToCartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem distance_from_polar_to_circle_center :
  let p := polarToCartesian 2 (Real.pi / 3) in
  let c := (1, 0) in
  ∃ (d : ℝ), d = Real.sqrt 3 ∧
             d = Real.sqrt ((p.1 - c.1)^2 + (p.2 - c.2)^2) :=
by
  simp
  sorry

end distance_from_polar_to_circle_center_l685_685915


namespace min_dwarfs_to_prevent_snow_white_sitting_l685_685580

/-- Minimum number of dwarfs required to ensure Snow White cannot sit with two empty chairs
beside her, given a circular table with 30 chairs. -/
theorem min_dwarfs_to_prevent_snow_white_sitting : 
  ∃ (d : ℕ), d = 10 ∧ 
  (∀ (arrangement : Fin 30 → bool), 
   (∀ n, arrangement n = ff → arrangement (n + 1) % 30 = tt ∨ arrangement (n - 1) % 30 = tt))
:=
sorry

end min_dwarfs_to_prevent_snow_white_sitting_l685_685580


namespace problem_statement_l685_685441

noncomputable def f (a b : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 0 then Real.sin (x + a) else Real.cos (x + b)

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

theorem problem_statement : 
  let a := Real.pi / 3
  let b := Real.pi / 6 in
  let f := f a b in
  is_even_function f →
  (Real.sin a = Real.cos b ∧ 
   ∀ x, x ≤ 0 → Real.sin (x + a) = Real.cos (-x + b)) :=
by
  sorry

end problem_statement_l685_685441


namespace A_works_15_days_l685_685702

noncomputable def total_work : ℝ := 1  -- Define total work as some unit work
noncomputable def B_time : ℝ := 4.5 -- B can complete the work in 4.5 days
noncomputable def B_work_rate : ℝ := total_work / B_time -- work rate of B
noncomputable def A_time : ℝ := 15 -- The time A alone can complete the work
noncomputable def A_work_rate : ℝ := total_work / A_time -- work rate of A

theorem A_works_15_days : 
  let work_by_A_in_5_days := 5 * A_work_rate,
      work_by_B_in_3_days := 3 * B_work_rate
  in work_by_A_in_5_days + work_by_B_in_3_days = total_work := 
by
  let work_by_A_in_5_days := 5 * A_work_rate
  let work_by_B_in_3_days := 3 * B_work_rate
  have work_by_A_and_B := work_by_A_in_5_days + work_by_B_in_3_days
  show work_by_A_and_B = total_work
  sorry

end A_works_15_days_l685_685702


namespace complete_the_square_l685_685651

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l685_685651


namespace find_ab_l685_685538

noncomputable def poly (x a b : ℝ) := x^4 + a * x^3 - 5 * x^2 + b * x - 6

theorem find_ab (a b : ℝ) (h : poly 2 a b = 0) : (a = 0 ∧ b = 4) :=
by
  sorry

end find_ab_l685_685538


namespace trapezoid_inscribed_circle_radius_trapezoid_circumscribed_circle_radius_l685_685592

-- Definitions of the trapezoid properties
variables {AB BC CD AD r R : ℝ}

-- Conditions
def is_isosceles_trapezoid :=
  BC = 4 ∧ AD = 16 ∧ AB = CD ∧ AB = 10

-- Goals to prove
def inscribed_circle_radius (h : is_isosceles_trapezoid) : Prop :=
  r = 4

def circumscribed_circle_radius (h : is_isosceles_trapezoid) : Prop :=
  R = (5 * Real.sqrt 41) / 4

-- Lean statements combining conditions and goals
theorem trapezoid_inscribed_circle_radius : is_isosceles_trapezoid → inscribed_circle_radius :=
  sorry

theorem trapezoid_circumscribed_circle_radius : is_isosceles_trapezoid → circumscribed_circle_radius :=
  sorry

end trapezoid_inscribed_circle_radius_trapezoid_circumscribed_circle_radius_l685_685592


namespace chocolate_distribution_a_chocolate_distribution_b_l685_685292

open Real

-- Problem (a):
theorem chocolate_distribution_a (m n : ℕ) (h_m : m = 9) (h_cond : ∀ k, k | n → k | m):
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18} :=
by
  sorry

-- Problem (b):
theorem chocolate_distribution_b (m n : ℕ) (h_m_n : m < n → (n - m) | m) :
  ∀ m n, ∃ k, (k | m ∨ k = n - m ∧ n - k * (m / k) = 0) :=
by
  sorry

end chocolate_distribution_a_chocolate_distribution_b_l685_685292


namespace fill_squares_exists_l685_685797

theorem fill_squares_exists : ∃ (a b c d e : ℕ), 
  (a ∈ {1, 2, 3, 5, 6}) ∧ (b ∈ {1, 2, 3, 5, 6}) ∧ (c ∈ {1, 2, 3, 5, 6}) ∧ 
  (d ∈ {1, 2, 3, 5, 6}) ∧ (e ∈ {1, 2, 3, 5, 6}) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ 
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ 
  (c ≠ d) ∧ (c ≠ e) ∧ 
  (d ≠ e) ∧ 
  ((a + b - c) * d / e = 4) := 
sorry

end fill_squares_exists_l685_685797


namespace ellipse_standard_equation_line_intersection_l685_685839

-- Given problem (1)
theorem ellipse_standard_equation
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (e : ℝ) (h3 : e = (real.sqrt 2) / 2)
  (h4 : (1 : ℝ, real.sqrt 2 / 2) ∈ set_of (λ p : ℝ × ℝ, (p.1^2 / a^2 + p.2^2 / b^2 = 1))) :
  (a^2 = 2 ∧ b^2 = 1) ∧ ∀ x y, x^2 / 2 + y^2 = 1 := 
sorry 

-- Given problem (2)
theorem line_intersection
  (k : ℝ)
  (M N : ℝ × ℝ)
  (h1 : M ∈ set_of (λ p : ℝ × ℝ, p.2 = k * (p.1 + 1)))
  (h2 : N ∈ set_of (λ p : ℝ × ℝ, p.2 = k * (p.1 + 1)))
  (h3 : M ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1))
  (h4 : N ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1))
  (F2 : ℝ × ℝ) (h5 : F2 = (1, 0)) -- Coordinates of F_2
  (h6 : abs (real.sqrt ((M.1 - F2.1 + N.1 - F2.1)^2 + (M.2 + N.2)^2)) = 2 * real.sqrt 26 / 3) :
  k = 1 ∨ k = -1 :=
sorry 

end ellipse_standard_equation_line_intersection_l685_685839


namespace part_I_part_II_part_III_l685_685107

noncomputable def sequences (a b : ℕ+ → ℚ) : Prop := 
  ∀ n : ℕ+, b (n+1) * a n + b n * a (n+1) = (-2)^n + 1

def b_n (n : ℕ+) : ℚ := (3 + (-1:ℚ)^(n-1)) / 2

def S_n (a : ℕ+ → ℚ) (n : ℕ) : ℚ := ∑ i in Finset.range n, a (i+1)

theorem part_I (a : ℕ+ → ℚ) (h_seq : sequences a b_n) (h_a1 : a 1 = 2) :
  a 2 = -3 / 2 ∧ a 3 = 8 :=
sorry

def c_n (a : ℕ+ → ℚ) (n : ℕ+) : ℚ := a (2*n+1) - a (2*n-1)

theorem part_II (a : ℕ+ → ℚ) (h_seq : sequences a b_n) :
  geometric_sequence (c_n a) :=
sorry

theorem part_III (a : ℕ+ → ℚ) (h_seq : sequences a b_n) (h_sums : ∀ n, S_n a n) :
  ∀ n : ℕ+, (Finset.range (2*n)).sum (λ k, S_n a (k+1) / a (k+1)) ≤ n - (1/3 : ℚ) :=
sorry

end part_I_part_II_part_III_l685_685107


namespace total_banana_produce_correct_l685_685698

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l685_685698


namespace tan_theta_eq_neg3_then_expr_eq_5_div_2_l685_685401

theorem tan_theta_eq_neg3_then_expr_eq_5_div_2
  (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5 / 2 := 
sorry

end tan_theta_eq_neg3_then_expr_eq_5_div_2_l685_685401


namespace perimeter_of_shaded_figure_l685_685348

/-- Define the context of the problem: square and right triangle setup -/
structure Square (A B C D: Point) where
  side_length : ℝ
  perimeter : ℝ
  h_side : 4 * side_length = perimeter

structure Triangle (B F C : Point) where
  side_length : ℝ
  angle : ℝ
  h_right_angle : angle = 90

/-- Conditions: perimeter of square and right triangle properties -/
def conditions : Prop :=
  let s := Square.mk A B C D 16 64 (by norm_num)
  let t := Triangle.mk B F C 16 90 (by norm_num)
  true

/-- Main theorem to prove: the perimeter of the shaded figure ABFCDE -/
theorem perimeter_of_shaded_figure (h : conditions) : ∃ (s : Square A B C D) (t : Triangle B F C),
    s.perimeter = 64 →
    t.angle = 90 →
    let ABFCDE_perimeter := 5 * s.side_length
    ABFCDE_perimeter = 80 :=
  sorry

end perimeter_of_shaded_figure_l685_685348


namespace matrix_fourth_power_l685_685001

noncomputable def matrix_to_the_power : Matrix (Fin 2) (Fin 2) ℝ := 
  ![
    ![√3, 1],
    ![-1, √3]
  ]

noncomputable def result_matrix : Matrix (Fin 2) (Fin 2) ℝ := 
  ![
    ![-8, 8 * √3],
    ![-8 * √3, -8]
  ]

theorem matrix_fourth_power :
  (matrix_to_the_power ^ 4) = result_matrix :=
sorry

end matrix_fourth_power_l685_685001


namespace complete_square_l685_685654

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l685_685654


namespace sum_first_16_terms_b_seq_eq_160_sum_first_16_terms_a_seq_eq_72_l685_685074

-- Define the sequence a_n = n
def a_seq : ℕ → ℕ := λ n, n

-- Define the sequence b_n = a_{n+1} + (-1)^n * a_n
def b_seq (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, a (n + 1) + (-1:ℤ) ^ n * a n

theorem sum_first_16_terms_b_seq_eq_160 :
  (∑ n in Finset.range 16, b_seq a_seq n) = 160 := by
  sorry

theorem sum_first_16_terms_a_seq_eq_72 :
  (∑ n in Finset.range 16, a_seq n) = 72 := by
  sorry

end sum_first_16_terms_b_seq_eq_160_sum_first_16_terms_a_seq_eq_72_l685_685074


namespace num_unique_pairs_example_l685_685051

noncomputable def num_unique_pairs : Nat :=
  let some_number := 15
  let pairs := { (a, b) : Nat × Nat | a^2 - b^2 = some_number ∧ a ≥ b }
  pairs.count

theorem num_unique_pairs_example (some_number : Nat) :
  some_number = 15 →
  ∃! n : Nat, num_unique_pairs = 2 :=
by
  sorry

end num_unique_pairs_example_l685_685051


namespace square_area_inscribed_parabola_l685_685975

/-- Given a square ABCD inscribed in the region bounded by the parabola
    y = x^2 - 8x + 12 and the x-axis, proving the area of the square ABCD is 24 - 8\sqrt{5}. -/
theorem square_area_inscribed_parabola : 
  ∃ (x y : ℝ), y = x^2 - 8 * x + 12 ∧
                let t := -1 + Real.sqrt 5 in
                let side_length := 2 * t in
                y = -2 * t ∧
                side_length^2 = 24 - 8 * Real.sqrt 5 := 
by sorry

end square_area_inscribed_parabola_l685_685975


namespace no_standing_pairs_probability_l685_685979

-- Definition for calculating the number of ways no two adjacent people form a standing pair
def b : ℕ → ℕ
| 0        := 1
| 1        := 2
| 2        := 3
| 3        := 6
| (n + 4) := b (n + 3) + b (n + 2)

-- Probability that no two adjacent people form a standing pair for 10 people
def probability_no_standing_pairs : ℕ := (b 10) / 2^10

-- Theorem for the given probability
theorem no_standing_pairs_probability : probability_no_standing_pairs = 31 / 128 := by
  sorry

end no_standing_pairs_probability_l685_685979


namespace intersection_of_sets_l685_685843

def A := {1, 6, 8, 10}
def B := {2, 4, 8, 10}

theorem intersection_of_sets : A ∩ B = {8, 10} :=
by
  sorry

end intersection_of_sets_l685_685843


namespace polynomial_expansion_sum_is_21_l685_685889

theorem polynomial_expansion_sum_is_21 :
  ∃ (A B C D : ℤ), (∀ (x : ℤ), (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) ∧
  A + B + C + D = 21 :=
by
  sorry

end polynomial_expansion_sum_is_21_l685_685889


namespace Martinez_family_combined_height_l685_685189

def combined_height : ℕ :=
  let h_C := 5 in
  let h_M := h_C + 2 in
  let h_W := h_C - 1 in
  let h_S := h_C + 3 in
  h_C + h_M + h_W + h_S

theorem Martinez_family_combined_height :
  combined_height = 24 := by
  unfold combined_height
  rfl

end Martinez_family_combined_height_l685_685189


namespace solve_for_z_l685_685896

theorem solve_for_z (x y : ℝ) (z : ℝ) (h : 2 / x - 1 / y = 3 / z) : 
  z = (2 * y - x) / 3 :=
by
  sorry

end solve_for_z_l685_685896


namespace line_integral_value_l685_685335

noncomputable def line_integral_parametric :=
  let x := λ t : ℝ, 2 * Real.cos t
  let y := λ t : ℝ, 2 * Real.sin t
  let z := λ t : ℝ, 1 - 2 * Real.cos t - 2 * Real.sin t
  ∫ t in 0..(Real.pi / 2), ((y t) / 3 * (-2 * Real.sin t) - 3 * (x t) * 2 * Real.cos t + (x t) * (2 * Real.sin t - 2 * Real.cos t))

theorem line_integral_value :
  line_integral_parametric = 2 - (13 * Real.pi) / 3 :=
  sorry

end line_integral_value_l685_685335


namespace expression_evaluation_l685_685267

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l685_685267


namespace bacteria_reaches_final_in_24_hours_l685_685591

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 200

-- Define the final number of bacteria
def final_bacteria : ℕ := 16200

-- Define the tripling period in hours
def tripling_period : ℕ := 6

-- Define the tripling factor
def tripling_factor : ℕ := 3

-- Define the number of hours needed to reach final number of bacteria
def hours_to_reach_final_bacteria : ℕ := 24

-- Define a function that models the number of bacteria after t hours
def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * tripling_factor^((t / tripling_period))

-- Main statement of the problem: prove that the number of bacteria is 16200 after 24 hours
theorem bacteria_reaches_final_in_24_hours :
  bacteria_after hours_to_reach_final_bacteria = final_bacteria :=
sorry

end bacteria_reaches_final_in_24_hours_l685_685591


namespace final_answer_is_15_l685_685382

-- We will translate the conditions from the problem into definitions and then formulate the theorem

-- Define the product of 10 and 12
def product : ℕ := 10 * 12

-- Define the result of dividing this product by 2
def divided_result : ℕ := product / 2

-- Define one-fourth of the divided result
def one_fourth : ℚ := (1/4 : ℚ) * divided_result

-- The theorem statement that verifies the final answer
theorem final_answer_is_15 : one_fourth = 15 := by
  sorry

end final_answer_is_15_l685_685382


namespace seventh_term_ratio_l685_685943

-- Conditions
variables {a b d e : ℚ}

-- Define the sums of the first n terms of the sequences
def S_n (n : ℕ) : ℚ := n / 2 * (2 * a + (n - 1) * d)
def T_n (n : ℕ) : ℚ := n / 2 * (2 * b + (n - 1) * e)

-- Define the given ratio condition
def given_ratio_condition (n : ℕ) : Prop := S_n n / T_n n = (7 * n + 3) / (4 * n + 30)

-- Define the 7th terms of the sequences
def a_7 : ℚ := a + 6 * d
def b_7 : ℚ := b + 6 * e

-- The proof statement
theorem seventh_term_ratio : (∀ n, given_ratio_condition n) → (a_7 / b_7 = 17 / 33) :=
by 
  intros h
  sorry

end seventh_term_ratio_l685_685943


namespace tallest_tree_height_l685_685250
-- Import the necessary library

-- Define the conditions
def tallest_tree : ℝ := sorry --  We'll use a placeholder for now to show that it's something we need to determine.
def middle_tree := (2 / 3) * tallest_tree
def shortest_tree := 50 -- given in feet
def middle_tree_given := 2 * shortest_tree

-- State the theorem to be proven
theorem tallest_tree_height :
  middle_tree = middle_tree_given → tallest_tree = 150 := 
by
  -- Place the proof here
  sorry

end tallest_tree_height_l685_685250


namespace necessary_but_not_sufficient_for_log_gt_zero_l685_685171

theorem necessary_but_not_sufficient_for_log_gt_zero {a b : ℝ} (hb : b ≠ 0) : 
  (∀ (h : lg(a - b) > 0), a > b) ∧ ¬ (∀ (h : a > b), lg(a - b) > 0) := 
begin
  sorry
end

end necessary_but_not_sufficient_for_log_gt_zero_l685_685171


namespace boys_from_school_A_not_studying_science_l685_685496

theorem boys_from_school_A_not_studying_science
  (total_boys : ℕ)
  (perc_school_A : ℝ)
  (perc_school_A_study_science : ℝ)
  (approx_total_boys : total_boys ≈ 350) :
  perc_school_A = 0.20 →
  perc_school_A_study_science = 0.30 →
  let boys_from_school_A := (perc_school_A * total_boys).toInt in
  let boys_from_school_A_study_science := (perc_school_A_study_science * boys_from_school_A).toInt in
  boys_from_school_A - boys_from_school_A_study_science = 49 :=
begin
  intros h1 h2,
  let boys_from_school_A := (0.20 * total_boys).toInt,
  let boys_from_school_A_study_science := (0.30 * boys_from_school_A).toInt,
  have h3 : boys_from_school_A = 70 := sorry,
  have h4 : boys_from_school_A_study_science = 21 := sorry,
  show boys_from_school_A - boys_from_school_A_study_science = 49,
  rw [h3, h4],
  simp,
end

end boys_from_school_A_not_studying_science_l685_685496


namespace quadrilateral_area_l685_685725

noncomputable def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  in area_triangle A B C + area_triangle A C D

theorem quadrilateral_area :
  area_of_quadrilateral (1, 2) (1, 1) (4, 1) (2009, 2010) = 2012021.5 :=
by
  sorry

end quadrilateral_area_l685_685725


namespace smallest_n_with_347_sequence_l685_685984

theorem smallest_n_with_347_sequence : ∃ (n : ℕ), (∀ (m : ℕ),
    Nat.coprime m n ∧ m < n →
    let d := (m : ℚ) / n
    let decimal_digits := (d.toRealDigits 10).snd
    3 ∈ decimal_digits ∧ 4 ∈ decimal_digits ∧ 7 ∈ decimal_digits ∧ 
        ∀ i j k, i < j ∧ j < k ∧ decimal_digits.get? i = some 3 ∧
                 decimal_digits.get? j = some 4 ∧
                 decimal_digits.get? k = some 7) ∧ n = 1000 :=
begin
  -- Proof starts here
  sorry
end

end smallest_n_with_347_sequence_l685_685984


namespace range_of_a_l685_685867

noncomputable def f (a x : ℝ) : ℝ := log a ((x - 2 * a) / (x + 2 * a))

theorem range_of_a (a s t : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x, x ∈ set.Icc s t → f a x ∈ set.Icc (log a (t - a)) (log a (s - a))) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l685_685867


namespace equal_12_mn_P_2n_Q_m_l685_685582

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l685_685582


namespace sixtieth_pair_is_5_7_l685_685410

def pair_by_sum (n m : ℕ) : ℕ :=
  n + m

theorem sixtieth_pair_is_5_7
  (pairs : ℕ → ℕ × ℕ)
  (h_pattern : ∀ n, let (a, b) := pairs n in pair_by_sum a b = nat.find_greatest (λ k, ((k * (k + 1)) / 2) < n) + 2)
  (h_conditions : (pairs 60) = (5,7)) :
  pairs 60 = (5, 7) :=
sorry

end sixtieth_pair_is_5_7_l685_685410


namespace area_of_path_l685_685309

theorem area_of_path
  (length_field : ℕ)
  (width_field : ℕ)
  (width_path : ℕ)
  (cost_per_sqm : ℕ)
  (total_cost : ℕ) :
  length_field = 95 →
  width_field = 55 →
  width_path = 25 / 10 →  -- Converted 2.5 m to fraction
  cost_per_sqm = 2 →
  total_cost = 1550 →
  let length_total := length_field + 2 * width_path in
  let width_total := width_field + 2 * width_path in
  let area_with_path := length_total * width_total in
  let area_field := length_field * width_field in
  let area_path := area_with_path - area_field in
  area_path = 775 :=
begin
  sorry  -- Placeholder for the proof
end

end area_of_path_l685_685309


namespace range_of_a_for_monotonically_decreasing_function_l685_685860

theorem range_of_a_for_monotonically_decreasing_function :
  ∀ (a : ℝ),
  (∀ (x y : ℝ), x ≤ y → f x ≥ f y) ↔
  (a ∈ set.Icc (1/2 : ℝ) (3/4 : ℝ)) :=
by
  let f : ℝ → ℝ := λ x,
    if x ≤ 1/2 then x^2 - 2 * a * x + 1
    else log a (x + 1/2) + 1/2
  sorry

end range_of_a_for_monotonically_decreasing_function_l685_685860


namespace minimum_value_expression_l685_685543

theorem minimum_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  (x^2 + 6 * x * y + 9 * y^2 + 3/2 * z^2) ≥ 102 :=
sorry

end minimum_value_expression_l685_685543


namespace angle_K_is_72_l685_685153

variables {J K L M : ℝ}

/-- Given that $JKLM$ is a trapezoid with parallel sides $\overline{JK}$ and $\overline{LM}$,
and given $\angle J = 3\angle M$, $\angle L = 2\angle K$, $\angle J + \angle K = 180^\circ$,
and $\angle L + \angle M = 180^\circ$, prove that $\angle K = 72^\circ$. -/
theorem angle_K_is_72 {J K L M : ℝ}
  (h1 : J = 3 * M)
  (h2 : L = 2 * K)
  (h3 : J + K = 180)
  (h4 : L + M = 180) :
  K = 72 :=
by
  sorry

end angle_K_is_72_l685_685153


namespace smallest_positive_c_exists_l685_685806

theorem smallest_positive_c_exists : 
  ∃ (c : ℝ), c = 1 / 3 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z →
  (real.cbrt (x * y * z) + c * |x - y + z|) ≥ (x + y + z) / 3 :=
sorry

end smallest_positive_c_exists_l685_685806


namespace tangent_parallel_at_point_l685_685903

theorem tangent_parallel_at_point :
  ∃ (x y : ℝ), (y = exp (-x)) ∧ ((- I.exp x) = -2) ∧ (x = -Real.log 2) ∧ (y = 2) := by
sorry

end tangent_parallel_at_point_l685_685903


namespace max_equilateral_triangle_area_in_rectangle_l685_685618

theorem max_equilateral_triangle_area_in_rectangle 
  (rectangle : Type)
  (P Q R S : rectangle)
  (side_PQ : ℝ) (side_PS : ℝ)
  (hPQ : side_PQ = 12) (hPS : side_PS = 13) :
  ∃ (a b c : ℕ), b ≠ 0 ∧ ¬ ∃ p, p^2 ∣ b ∧ 
  max_area_triangle = a * sqrt b - c ∧ 
  max_area_triangle = 48 * sqrt 3 - 0 :=
sorry

end max_equilateral_triangle_area_in_rectangle_l685_685618


namespace triangle_hypotenuse_and_area_l685_685901

theorem triangle_hypotenuse_and_area 
  (A B C D : Type) 
  (CD : ℝ) 
  (angle_A : ℝ) 
  (hypotenuse_AC : ℝ) 
  (area_ABC : ℝ) 
  (h1 : CD = 1) 
  (h2 : angle_A = 45) : 
  hypotenuse_AC = Real.sqrt 2 
  ∧ 
  area_ABC = 1 / 2 := 
by
  sorry

end triangle_hypotenuse_and_area_l685_685901


namespace range_of_m_l685_685491

theorem range_of_m (x y m : ℝ) 
  (h1 : x - 2 * y = 1) 
  (h2 : 2 * x + y = 4 * m) 
  (h3 : x + 3 * y < 6) : 
  m < 7 / 4 := 
sorry

end range_of_m_l685_685491


namespace number_of_imaginary_numbers_l685_685054

theorem number_of_imaginary_numbers : 
  let S := {0, 1, 2, 3, 4, 5, 6}
  in (S.product (S.erase 0)).card = 36 :=
by
  sorry

end number_of_imaginary_numbers_l685_685054


namespace length_of_curve_l685_685613

theorem length_of_curve (y : ℝ) (h_y : -1 ≤ y ∧ y ≤ 1) :
  ∫ v in -2..2, sqrt (1 + (v^2 / 4)) = 4 * (sqrt 2 + log (1 + sqrt 2)) :=
sorry

end length_of_curve_l685_685613


namespace parabola_directrix_standard_eq_l685_685241

theorem parabola_directrix_standard_eq (p : ℝ) (h : p = 2) :
  ∀ y x : ℝ, (x = -1) → (y^2 = 4 * x) :=
by
  sorry

end parabola_directrix_standard_eq_l685_685241


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l685_685584

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l685_685584


namespace equidistant_line_l685_685042

noncomputable def line1 : ℝ → ℝ → Prop := λ x y, x - 2*y + 4 = 0
noncomputable def line2 : ℝ → ℝ → Prop := λ x y, 2*x - y - 1 = 0
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (4, 0)

theorem equidistant_line (x y : ℝ) : 
  (line1 2 3 ∧ line2 2 3 ∧ 
   (line1 x y ∨ line2 x y) ∧ 
    (eq A 0 4 ∧ eq B 4 0)) 
  → (x + y - 5 = 0 ∨ x = 2) := 
sorry

end equidistant_line_l685_685042


namespace sum_of_squares_of_roots_l685_685775

theorem sum_of_squares_of_roots :
  ∀ x : ℝ, (2 * x * real.sqrt x - 9 * x + 5 * real.sqrt x - 3 = 0 ∧ x ≥ 0) →
  ∃ r s t : ℝ, r^2 + s^2 + t^2 = 15.25 :=
begin
  sorry
end

end sum_of_squares_of_roots_l685_685775


namespace proof_problem_l685_685431

variable (X : ℝ)
variable (μ σ : ℝ)
variable (hX : X ~ Normal μ σ)
variable (P2_6 : μ = 4 → σ = 1 → 0.9544 = ∫ x, indicator (Ioc (μ - 2 * σ) (μ + 2 * σ)) x * pdf (Normal μ σ) x)
variable (P3_5 : μ = 4 → σ = 1 → 0.6826 = ∫ x, indicator (Ioc (μ - σ) (μ + σ)) x * pdf (Normal μ σ) x)
variable (P5_6 : μ = 4 → σ = 1 → 0.1359 = ∫ x, indicator (Ioo 5 6) x * pdf (Normal μ σ) x)

theorem proof_problem : P5_6 := by
  sorry

end proof_problem_l685_685431


namespace unoccupied_volume_is_correct_l685_685790

-- Definition of the tank volume
def tank_volume : ℝ := 12^3

-- Definition of the volume of water
def water_volume : ℝ := (1/3) * tank_volume

-- Definition of the volume of one ice cube
def ice_cube_volume : ℝ := 1.5^3

-- Definition of the total volume of ice cubes
def total_ice_volume : ℝ := 15 * ice_cube_volume

-- Definition of the total volume occupied by water and ice
def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Definition of the unoccupied volume in the tank
def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- The statement to be proved
theorem unoccupied_volume_is_correct : unoccupied_volume = 1101.375 := by
  sorry

end unoccupied_volume_is_correct_l685_685790


namespace probability_of_two_digit_number_in_set_l685_685720

theorem probability_of_two_digit_number_in_set :
  (∃ (S : set ℕ), S = {n | 30 ≤ n ∧ n ≤ 1000} ∧ 
    (∃ (t : ℕ), t = 70) ∧ 
    (∃ (u : ℕ), u = 971) ∧ 
    ∀ (p : ℚ), p = (70 : ℕ) / (971 : ℕ) → p = (70 / 971 : ℚ)) := 
begin
  sorry
end

end probability_of_two_digit_number_in_set_l685_685720


namespace minimum_A2_minus_B2_l685_685181

noncomputable def A (x y z : ℝ) : ℝ := 
  Real.sqrt (x + 6) + Real.sqrt (y + 7) + Real.sqrt (z + 12)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 2) + Real.sqrt (y + 3) + Real.sqrt (z + 5)

theorem minimum_A2_minus_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z)^2 - (B x y z)^2 = 49.25 := 
by 
  sorry 

end minimum_A2_minus_B2_l685_685181


namespace banana_production_total_l685_685697

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l685_685697


namespace find_f_2010_l685_685829

noncomputable def f (a b α β x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2010 (a b α β : ℝ) (h : f a b α β 2009 = 3) : f a b α β 2010 = -3 :=
  sorry

end find_f_2010_l685_685829


namespace perpendicular_lines_and_slope_l685_685600

theorem perpendicular_lines_and_slope (b : ℝ) : (x + 3 * y + 4 = 0) ∧ (b * x + 3 * y + 6 = 0) → b = -9 :=
by
  sorry

end perpendicular_lines_and_slope_l685_685600


namespace n_minus_m_eq_200_l685_685993

-- Define the parameters
variable (m n x : ℝ)

-- State the conditions
def condition1 : Prop := m ≤ 8 * x - 1 ∧ 8 * x - 1 ≤ n 
def condition2 : Prop := (n + 1)/8 - (m + 1)/8 = 25

-- State the theorem to prove
theorem n_minus_m_eq_200 (h1 : condition1 m n x) (h2 : condition2 m n) : n - m = 200 := 
by 
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end n_minus_m_eq_200_l685_685993


namespace intersection_complement_l685_685455

def R := Set ℝ

def A : R := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B : R := { x : ℝ | x < 1 }

theorem intersection_complement :
  A ∩ (Set.univ \ B) = { x : ℝ | 1 ≤ x ∧ x ≤ 2 } :=
  sorry

end intersection_complement_l685_685455


namespace shortest_altitude_of_right_triangle_l685_685237

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l685_685237


namespace sum_of_decimals_as_fraction_l685_685796

theorem sum_of_decimals_as_fraction :
  let x := (0 : ℝ) + 1 / 3;
  let y := (0 : ℝ) + 2 / 3;
  let z := (0 : ℝ) + 2 / 5;
  x + y + z = 7 / 5 :=
by
  let x := (0 : ℝ) + 1 / 3
  let y := (0 : ℝ) + 2 / 3
  let z := (0 : ℝ) + 2 / 5
  show x + y + z = 7 / 5
  sorry

end sum_of_decimals_as_fraction_l685_685796


namespace original_number_of_cards_l685_685570

theorem original_number_of_cards (X : ℕ) :
  (∀ (sasha_added karen_took_out remaining_cards : ℕ),
    sasha_added = 48 →
    karen_took_out = sasha_added / 6 →
    remaining_cards = X + sasha_added - karen_took_out →
    remaining_cards = 83) →
  X = 43 :=
by
  assume h
  have sasha_added := 48
  have karen_took_out := sasha_added / 6
  have remaining_cards := X + sasha_added - karen_took_out
  have eq1 : sasha_added = 48 := rfl
  have eq2 : karen_took_out = sasha_added / 6 := rfl
  have eq3 : remaining_cards = X + 48 - 8 := by
    simp [karen_took_out]
    norm_num
    rfl
  have eq4 : remaining_cards = 83 := h 48 karen_took_out remaining_cards eq1 eq2 eq3
  have eq5 : X + 40 = 83 := by
    simp [eq3] at eq4
    exact eq4
  have eq6 : X = 83 - 40 := by
    simp [eq5]
    norm_num
  have eq7 : X = 43 := by
    simp [eq6]
    norm_num
  exact eq7

#check original_number_of_cards

end original_number_of_cards_l685_685570


namespace percentage_of_liquid_X_in_solution_A_l685_685550

theorem percentage_of_liquid_X_in_solution_A (P : ℝ) :
  (0.018 * 700 / 1200 + P * 500 / 1200) = 0.0166 → P = 0.01464 :=
by 
  sorry

end percentage_of_liquid_X_in_solution_A_l685_685550


namespace Ben_remaining_money_l685_685326

-- Define Ben's initial bonus amount
def initialBonus : ℝ := 1496

-- Define the fractions for each allocation
def kitchenFraction : ℝ := 1 / 22
def renovationFraction : ℝ := 2 / 5
def holidayFraction : ℝ := 1 / 4
def charityFraction : ℝ := 1 / 6
def giftFraction : ℝ := 1 / 8

-- Calculate the total allocation
def totalAllocation : ℝ := 
  (kitchenFraction + renovationFraction + holidayFraction + charityFraction + giftFraction) * initialBonus

-- Define the remaining amount
def remainingAmount : ℝ := initialBonus - totalAllocation

-- Assertion: Ben's remaining amount
theorem Ben_remaining_money : abs (remainingAmount - 19.27) < 0.01 := 
begin
  sorry
end

end Ben_remaining_money_l685_685326


namespace square_minus_self_divisible_by_2_l685_685201

theorem square_minus_self_divisible_by_2 (a : ℕ) : 2 ∣ (a^2 - a) :=
by sorry

end square_minus_self_divisible_by_2_l685_685201


namespace part_a_part_b_l685_685125

section TriangleProofs

variables {α β γ : ℝ} {a b c : ℝ} (λ : ℝ)
variables [h0 : ∀ (α β γ : ℝ), γ ≠ 0]
variables [λ_def : λ = (Real.cot α + Real.cot β) / Real.cot γ]

theorem part_a (h0 : γ ≠ 0) (λ_def : λ = (Real.cot α + Real.cot β) / Real.cot γ) :
  a^2 + b^2 = (1 + 2 / λ) * c^2 :=
sorry

theorem part_b (h1 : γ ≠ 0) (h_lambda : λ = 2) (λ_def : λ = (Real.cot α + Real.cot β) / Real.cot γ) :
  γ ≤ (π / 3) :=
sorry

end TriangleProofs

end part_a_part_b_l685_685125


namespace quadratic_common_root_inverse_other_roots_l685_685534

variables (p q r s : ℝ)
variables (hq : q ≠ -1) (hs : s ≠ -1)

theorem quadratic_common_root_inverse_other_roots :
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 + p * a + q = 0) ∧ (a * b = 1) ∧ (b^2 + r * b + s = 0)) ↔ 
  (p * r = (q + 1) * (s + 1) ∧ p * (q + 1) * s = r * (s + 1) * q) :=
sorry

end quadratic_common_root_inverse_other_roots_l685_685534


namespace ants_left_correct_l685_685311

-- Defining the initial number of ants.
def initial_ants : ℕ := 6300

-- Defining the ratios for each step.
def first_step_ratio : ℝ := 0.7
def second_step_ratio : ℝ := 0.4
def third_step_ratio : ℝ := 0.25

-- Calculating the number of ants left after each step.
def first_step (ants : ℕ) : ℕ :=
  ants - (first_step_ratio * ants).toInt

def second_step (ants : ℕ) : ℕ :=
  ants - (second_step_ratio * ants).toInt

def third_step (ants : ℕ) : ℕ :=
  ants - (third_step_ratio * ants).toInt

-- Combining the steps to find the number of ants left after the third step.
def ants_left_after_three_steps : ℕ :=
  let after_first := first_step initial_ants
  let after_second := second_step after_first
  third_step after_second

-- Statement of the theorem.
theorem ants_left_correct : ants_left_after_three_steps = 851 := 
  sorry

end ants_left_correct_l685_685311


namespace completing_the_square_solution_l685_685657

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l685_685657


namespace systematic_sampling_l685_685732

theorem systematic_sampling(
  (N : ℕ) (sample_size : ℕ) (first_part_range : Finset ℕ) (r : ℕ)
  (hN : N = 1000)
  (hsample : sample_size = 50)
  (hfirst_part : first_part_range = Finset.range (0 + 20 + 1))
  (hr : r = 15)
) : r + 10 * (N / sample_size) = 215 :=
by
  -- Provided constants and calculations
  have f : ℕ := N / sample_size,
  sorry

end systematic_sampling_l685_685732


namespace max_value_l685_685175

noncomputable def maxN (x y z : ℝ) := 3 * x * z + 5 * y * z + 8 * x * y

theorem max_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 4 * x^2 + 9 * y^2 + 16 * z^2 = 144) :
  let N := maxN x y z in N + x + y + z = 319 := by
  sorry

end max_value_l685_685175


namespace algorithm_can_contain_any_combination_l685_685626

-- Definitions for the types of logical structures
inductive LogicalStructure
| sequence
| conditional
| loop

-- The statement to prove: An algorithm can contain any combination of these logical structures.
theorem algorithm_can_contain_any_combination :
  ∃ (seq cond loop : bool), 
    (seq = true ∨ seq = false) ∧
    (cond = true ∨ cond = false) ∧
    (loop = true ∨ loop = false) := 
sorry

end algorithm_can_contain_any_combination_l685_685626


namespace product_of_six_numbers_l685_685194

theorem product_of_six_numbers (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x^3 * y^2 = 108) : 
  x * y * (x * y) * (x^2 * y) * (x^3 * y^2) * (x^5 * y^3) = 136048896 := 
by
  sorry

end product_of_six_numbers_l685_685194


namespace find_ellipse_eccentricity_l685_685220

noncomputable def ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * c = 2 * real.sqrt(a^2 - b^2)) : ℝ :=
    real.sqrt(4 - 2 * real.sqrt(3))

theorem find_ellipse_eccentricity (a b c : ℝ) (x y : ℝ) (F₁ F₂ M : ℝ × ℝ) :
  (a > b) -> 
  (b > 0) -> 
  (2 * c = 2 * real.sqrt(a^2 - b^2)) -> 
  (line (λ x, real.sqrt(3) * (x + c)) F₁ a b M) -> 
  (γ b > γ b / 2) -> 
  (eccentricity = real.sqrt(3) - 1) := 
by sorry

end find_ellipse_eccentricity_l685_685220


namespace smallest_value_of_y_undefined_l685_685263

-- Noncomputable means we expect some computations at some point that are not easy to guarantee as computable in Lean
noncomputable def find_smallest_y : Real :=
  let quadratic_formula (a b c : Real) : Real :=
    let discriminant := b * b - 4 * a * c in
    let sqrt_discriminant := Real.sqrt discriminant in
    let y1 := (-b + sqrt_discriminant) / (2 * a) in
    let y2 := (-b - sqrt_discriminant) / (2 * a) in
    if y1 < y2 then y1 else y2
  quadratic_formula 9 (-56) 7

theorem smallest_value_of_y_undefined :
  ∃ y : Real, y = find_smallest_y ∧ y ≈ 0.128 :=
by
  exists find_smallest_y
  unfold find_smallest_y
  sorry

end smallest_value_of_y_undefined_l685_685263


namespace product_b1_b13_l685_685838

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) := ∀ n m k : ℕ, m > 0 → k > 0 → a (n + m) - a n = a (n + k) - a (n + k - m)

-- Conditions for the geometric sequence
def is_geometric_seq (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

-- Given conditions
def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (a 3 - (a 7 ^ 2) / 2 + a 11 = 0) ∧ (b 7 = a 7)

theorem product_b1_b13 
  (ha : is_arithmetic_seq a)
  (hb : is_geometric_seq b)
  (h : conditions a b) :
  b 1 * b 13 = 16 :=
sorry

end product_b1_b13_l685_685838


namespace PA_squared_plus_PB_squared_geq_2r_squared_l685_685218

-- Define the centers and conditions
variables {O1 O2 A B P : Point}
variable {r : ℝ}
variable (k1 k2 : Circle)

-- Assuming necessary conditions
def conditions : Prop :=
  distance O1 O2 = r ∧
  on_circle A k1 ∧ on_circle B k1 ∧
  symmetric A B (line_through O1 O2) ∧
  on_circle P k2

theorem PA_squared_plus_PB_squared_geq_2r_squared
  (h : conditions) :
  distance_squared P A + distance_squared P B ≥ 2 * r^2 :=
sorry

end PA_squared_plus_PB_squared_geq_2r_squared_l685_685218


namespace find_y_coordinate_l685_685917

theorem find_y_coordinate (m n : ℝ) 
  (h₁ : m = 2 * n + 5) 
  (h₂ : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := 
sorry

end find_y_coordinate_l685_685917


namespace proof_n_value_l685_685581

theorem proof_n_value (n : ℕ) (h : (9^n) * (9^n) * (9^n) * (9^n) * (9^n) = 81^5) : n = 2 :=
by
  sorry

end proof_n_value_l685_685581


namespace consecutiveWhiteBallsProb_l685_685147

-- Definitions based on conditions
def totalBalls : ℕ := 9
def whiteBalls : ℕ := 5
def blackBalls : ℕ := 4
def firstDrawWhiteProb : ℚ := whiteBalls / totalBalls
def secondDrawWhiteProb : ℚ := (whiteBalls - 1) / (totalBalls - 1)

-- Proving that the probability of drawing two white balls consecutively is 5/18
theorem consecutiveWhiteBallsProb : firstDrawWhiteProb * secondDrawWhiteProb = 5 / 18 := by
  sorry

end consecutiveWhiteBallsProb_l685_685147


namespace hexagon_area_l685_685778

theorem hexagon_area :
  ∀ (s : ℝ), 
    s = 3 → 
    let area_triangle := (sqrt 3 / 4) * s^2 in 
    let area_hexagon := 2 * area_triangle in 
    area_hexagon = (9 * sqrt 3) / 2 :=
by
  intros s h
  let area_triangle := (sqrt 3 / 4) * s^2
  let area_hexagon := 2 * area_triangle
  rw h
  sorry

end hexagon_area_l685_685778


namespace part1_part2_l685_685873

section Part1

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = 2 * a n + 4

def general_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2^n - 4

theorem part1 (a : ℕ → ℤ) (h : sequence a) : general_formula a :=
  sorry

end Part1

section Part2

def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  Int.ofNat ((a (n + 1) + 4) ^ (a n + 4))

def S (a : ℕ → ℤ) (b : (ℕ → ℤ) → ℕ → ℤ) (n : ℕ) : ℤ :=
  -(∑ i in Finset.range n, (i + 1) * (2 ^ i))

theorem part2 (a : ℕ → ℤ) (ha : general_formula a) (n : ℕ) : 
  S a (b a) n = -n * 2^(n+1) := 
  sorry

end Part2

end part1_part2_l685_685873


namespace coefficient_x3_in_expansion_l685_685437

theorem coefficient_x3_in_expansion :
  (∃ c : ℝ, c * x^3 = coeff_of_x3 (2 * x + sqrt x) ^ 5) ∧ c = 10 :=
sorry

end coefficient_x3_in_expansion_l685_685437


namespace product_of_geometric_sequence_l685_685158

theorem product_of_geometric_sequence (x y z : ℝ) 
  (h_seq : ∃ r, x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) : 
  1 * x * y * z * 4 = 32 :=
by
  sorry

end product_of_geometric_sequence_l685_685158


namespace coefficient_of_x2_in_polynomial_sum_l685_685913

theorem coefficient_of_x2_in_polynomial_sum :
  let f (x : ℕ) := (1 + x) + (1 + x)^2 + (1 + x)^3 + ... + (1 + x)^11
  (x : ℕ) := f 2 = 220 := sorry

end coefficient_of_x2_in_polynomial_sum_l685_685913


namespace angle_difference_l685_685747

-- Given data and assumptions
variables {K : Type*} [Field K]
variables {P A B Q : K}
variables [Circle : Set K]
variables {angle : K → K → K → K}

-- Definitions corresponding to given conditions
-- Points P, A, and B lie on the circle
def on_circle (P A B : K) : Prop := P ∈ Circle ∧ A ∈ Circle ∧ B ∈ Circle

-- Angle conditions
def angle_90 (P A Q : K) : Prop := angle P A Q = 90
def equal_length (PQ BQ : K) : Prop := PQ = BQ

-- Proof statement
theorem angle_difference (P A B Q : K) 
(on_circle P A B) (angle_90 P A Q) (equal_length PQ BQ) : 
angle A Q B - angle P Q A = (arc A B) :=
sorry

end angle_difference_l685_685747


namespace find_values_of_x_and_y_l685_685899

-- Define the conditions
def first_condition (x : ℝ) : Prop := 0.75 / x = 5 / 7
def second_condition (y : ℝ) : Prop := y / 19 = 11 / 3

-- Define the main theorem to prove
theorem find_values_of_x_and_y (x y : ℝ) (h1 : first_condition x) (h2 : second_condition y) :
  x = 1.05 ∧ y = 209 / 3 := 
by 
  sorry

end find_values_of_x_and_y_l685_685899


namespace omega_and_phi_min_max_f_l685_685440

-- Given the function f(x) = cos²(ωx + φ) - 1/2
def f (ω φ x : ℝ) : ℝ := (Real.cos (ω * x + φ))^2 - 1/2

-- Conditions for ω and φ
axiom omega_pos (ω : ℝ) : ω > 0
axiom phi_bound (φ : ℝ) : 0 < φ ∧ φ < π / 2
axiom f_period (ω φ : ℝ) : (∀ x : ℝ, f (ω φ (x + π)) = f ω φ x)
axiom f_at_point (ω φ : ℝ) : f ω φ (π / 8) = 1 / 4

-- Prove ω = 1 and φ = π / 24
theorem omega_and_phi (ω φ : ℝ) 
  (h1 : omega_pos ω)
  (h2 : phi_bound φ)
  (h3 : f_period ω φ)
  (h4 : f_at_point ω φ) :
  ω = 1 ∧ φ = π / 24 :=
sorry

-- Prove minimum and maximum values of f in the interval [π / 24, 13π / 24]
theorem min_max_f (ω φ : ℝ) 
  (h5 : ω = 1)
  (h6 : φ = π / 24) :
  (∀ x : ℝ, π / 24 ≤ x ∧ x ≤ 13 * π / 24 → 
    -1 / 2 ≤ f ω φ x ∧ f ω φ x ≤ (√(3) / 4)) :=
sorry

end omega_and_phi_min_max_f_l685_685440


namespace tangent_lines_l685_685866

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def P : ℝ × ℝ := (1, -2)

theorem tangent_lines (x₀ : ℝ) (h₀ : f(1) = -2) :
  let y := f(x₀) in
  let slope := 3 * x₀^2 - 3 in
  if x₀ = 1 then y = -2 ∧ slope = 0 ∧ (x = (1 : ℝ) → y = -2)
  else if x₀ = -1/2 then y = (-1/2)^3 - 3 * (-1/2) ∧ slope = 3 * (-1/2)^2 - 3 ∧ (x = x₀ → y = -9/4 * x + 1/4)
  else false := 
sorry

end tangent_lines_l685_685866


namespace rook_reaches_right_total_rook_reaches_right_seven_moves_l685_685310

-- Definition of the conditions for the problem
def rook_ways_total (n : Nat) :=
  2 ^ (n - 2)

def rook_ways_in_moves (n k : Nat) :=
  Nat.choose (n - 2) (k - 1)

-- Proof problem statements
theorem rook_reaches_right_total : rook_ways_total 30 = 2 ^ 28 := 
by sorry

theorem rook_reaches_right_seven_moves : rook_ways_in_moves 30 7 = Nat.choose 28 6 := 
by sorry

end rook_reaches_right_total_rook_reaches_right_seven_moves_l685_685310


namespace simplify_expr1_simplify_expr2_l685_685578

theorem simplify_expr1 (a b : ℤ) : 2 * a - (4 * a + 5 * b) + 2 * (3 * a - 4 * b) = 4 * a - 13 * b :=
by sorry

theorem simplify_expr2 (x y : ℤ) : 5 * x^2 - 2 * (3 * y^2 - 5 * x^2) + (-4 * y^2 + 7 * x * y) = 15 * x^2 - 10 * y^2 + 7 * x * y :=
by sorry

end simplify_expr1_simplify_expr2_l685_685578


namespace wet_surface_area_correct_l685_685300

def cistern_length := 15
def cistern_width := 10
def cistern_height := 8
def water_depth := 4.75

def bottom_area := cistern_length * cistern_width
def longer_side_area := 2 * (cistern_length * water_depth)
def shorter_side_area := 2 * (cistern_width * water_depth)

def wet_surface_area := bottom_area + longer_side_area + shorter_side_area

theorem wet_surface_area_correct : wet_surface_area = 387.5 := 
by 
  have h1 : bottom_area = 150 := by sorry
  have h2 : longer_side_area = 142.5 := by sorry
  have h3 : shorter_side_area = 95 := by sorry
  have h4 : wet_surface_area = bottom_area + longer_side_area + shorter_side_area := by refl
  rw [h1, h2, h3]
  norm_num

end wet_surface_area_correct_l685_685300


namespace original_price_l685_685529

theorem original_price (P : ℝ) (final_price_with_tax : ℝ) (tax_rate : ℝ) 
(d1 d2 d3 : ℝ) (fixed_final_price : ℝ) :
  final_price_with_tax = 17 ∧ tax_rate = 0.08 ∧ d1 = 0.25 ∧ d2 = 0.25 ∧ d3 = 0.15 ∧
  fixed_final_price = 15.74 →
  let final_price := P * (1 - d1) * (1 - d2) * (1 - d3) in
  fixed_final_price = final_price / (1 + tax_rate) →
  P ≈ 32.91 :=
begin
  -- proof goes here
  sorry
end

end original_price_l685_685529


namespace find_n_l685_685999

theorem find_n (n : ℕ) (h1 : n > 0) (h2 : ∏ d in (finset.filter (λ k, n % k = 0) (finset.range (n+1))), d = 256) : n = 16 := 
sorry

end find_n_l685_685999


namespace meeting_point_one_third_l685_685885

theorem meeting_point_one_third :
  let Harry : ℝ × ℝ := (8, 3)
  let Sandy : ℝ × ℝ := (14, 9)
  let t : ℝ := 1 / 3
  let Δ := (Sandy.1 - Harry.1, Sandy.2 - Harry.2)
  let MeetingPoint := (Harry.1 + t * Δ.1, Harry.2 + t * Δ.2)
  MeetingPoint = (10, 5) :=
by
  -- Define Harry and Sandy positions
  let Harry := (8 : ℝ, 3 : ℝ)
  let Sandy := (14 : ℝ, 9 : ℝ)
  -- Define the proportion t = 1 / 3
  let t := 1 / 3
  -- Calculate the difference in coordinates
  let Δ := (Sandy.1 - Harry.1, Sandy.2 - Harry.2)
  -- Compute the meeting point
  let MeetingPoint := (Harry.1 + t * Δ.1, Harry.2 + t * Δ.2)
  -- Assert and prove the meeting point
  have : MeetingPoint = (10, 5) := by sorry
  assumption

end meeting_point_one_third_l685_685885


namespace a_and_b_complete_work_l685_685297

def work_days (a_days : ℕ) (b_days : ℕ) (a_worked_alone : ℕ) : ℕ :=
  let a_rate := 1 / a_days
  let b_rate := 1 / b_days
  let a_work_done := a_worked_alone * a_rate
  let remaining_work := 1 - a_work_done
  let combined_rate := a_rate + b_rate
  remaining_work / combined_rate

theorem a_and_b_complete_work (a_days : ℕ) (b_days : ℕ) (a_worked_alone : ℕ) : work_days a_days b_days a_worked_alone = 3 :=
by
  -- proof omitted
  sorry

#eval a_and_b_complete_work 12 6 3 -- Expected 3

end a_and_b_complete_work_l685_685297


namespace tangent_circle_intersects_origin_at_x_axis_l685_685559

noncomputable def parabola (p : ℝ) (hp : 0 < p) : set (ℝ × ℝ) :=
  {P | ∃ y, P = (y^2 / (2 * p), y)}

noncomputable def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

noncomputable def is_tangent (C P F : ℝ × ℝ) : Prop :=
  dist C P = dist C F

theorem tangent_circle_intersects_origin_at_x_axis
  (p : ℝ) (hp : 0 < p)
  (P : ℝ × ℝ) (hP : P ∈ parabola p hp)
  (F : ℝ × ℝ) (hF : F = focus p)
  (Q : ℝ × ℝ) (hQc : Q.2 = 0):
  is_tangent Q (0,0) F → Q = (0,0) :=
by
  sorry

end tangent_circle_intersects_origin_at_x_axis_l685_685559


namespace foundation_cost_calculation_l685_685925

section FoundationCost

-- Define the constants given in the conditions
def length : ℝ := 100
def width : ℝ := 100
def height : ℝ := 0.5
def density : ℝ := 150  -- in pounds per cubic foot
def cost_per_pound : ℝ := 0.02
def number_of_houses : ℕ := 3

-- Define the problem using these conditions
theorem foundation_cost_calculation :
  let volume := length * width * height in
  let weight := volume * density in
  let cost_one_house := weight * cost_per_pound in
  let total_cost := cost_one_house * (number_of_houses:ℝ) in
  total_cost = 45000 := 
by {
  -- The proof goes here
  sorry
}

end FoundationCost

end foundation_cost_calculation_l685_685925


namespace part1_part2_l685_685820

variables (λ θ : ℝ)

def a := (Real.cos (λ * θ), Real.cos ((10 - λ) * θ)) 
def b := (Real.sin ((10 - λ) * θ), Real.sin (λ * θ))

theorem part1 : (a λ θ).fst ^ 2 + (a λ θ).snd ^ 2 + (b λ θ).fst ^ 2 + (b λ θ).snd ^ 2 = 2 := sorry

theorem part2 (θ := Real.pi / 20) : 
  let a := (Real.cos (λ * θ), Real.cos ((10 - λ) * θ))
  let b := (Real.sin ((10 - λ) * θ), Real.sin (λ * θ))
  (a.fst * b.fst + a.snd * b.snd = 0) := sorry

end part1_part2_l685_685820


namespace ceil_neg_3_6_eq_neg_3_l685_685791

-- Define the ceiling function as finding the smallest integer 
-- greater than or equal to the input number
def ceil (x : ℝ) : ℤ :=
  let z := Int.floor (x)
  if x == z.to_real then z else z + 1

-- Statement: The ceiling of -3.6 is -3
theorem ceil_neg_3_6_eq_neg_3 : ceil (-3.6) = -3 :=
by
  sorry

end ceil_neg_3_6_eq_neg_3_l685_685791


namespace complete_square_l685_685652

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l685_685652


namespace shortest_altitude_of_right_triangle_l685_685131

theorem shortest_altitude_of_right_triangle (a b c : ℕ) (h₀ : a = 30) (h₁ : b = 40) (h₂ : c = 50)
  (hc : a^2 + b^2 = c^2) : ∃ h : ℕ, 2 * 600 = c * h ∧ h = 24 :=
by
  have h_triangle : 30^2 + 40^2 = 50^2 := by { norm_num }
  have h_area : 600 = 30 * 40 / 2 := by { norm_num }
  use 24
  split
  · 
    calc 2 * 600 = 1200 : by norm_num
         ... = 50 * 24 : by norm_num
  ·
    norm_num
  sorry

end shortest_altitude_of_right_triangle_l685_685131


namespace isosceles_triangle_l685_685071
   
   theorem isosceles_triangle (a b c : ℝ) 
         (h_eqn: (a + c) * 1^2 - 2 * b * 1 - a + c = 0) : 
         c = b :=
   by
   simp at h_eqn,
   sorry
   
end isosceles_triangle_l685_685071


namespace elvis_album_songs_l685_685367

theorem elvis_album_songs :
  let studio_time := 5 * 60 in
  let time_per_song := 15 + 12 in
  let edit_time := 30 in
  let remaining_time := studio_time - edit_time in
  let number_of_songs := remaining_time / time_per_song in
  number_of_songs = 10 :=
by 
  let studio_time := 5 * 60
  let time_per_song := 15 + 12
  let edit_time := 30
  let remaining_time := studio_time - edit_time
  let number_of_songs := remaining_time / time_per_song
  show number_of_songs = 10
  from sorry

end elvis_album_songs_l685_685367


namespace coconut_transport_l685_685759

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l685_685759


namespace endpoints_on_one_sphere_l685_685879

-- Definitions for points, spheres, and chords
noncomputable section

variable (A B C D E F P : Point)
variable (S1 S2 S3 : Sphere)
variable (Chord1 : Chord S1) (Chord2 : Chord S2) (Chord3 : Chord S3)

-- Conditions as per the problem statement
axiom AB_intersects_P : P ∈ Chord1 ∧ P ∈ AB
axiom CD_intersects_P : P ∈ Chord2 ∧ P ∈ CD
axiom EF_intersects_P : P ∈ Chord3 ∧ P ∈ EF

axiom AB_on_S1 : AB ⊆ S1
axiom CD_on_S2 : CD ⊆ S2
axiom EF_on_S3 : EF ⊆ S3

axiom common_chord_P : ∀ S : Sphere, P ∈ S ∧ C ∈ S → S = S1 ∨ S = S2 ∨ S = S3

-- Theorem statement
theorem endpoints_on_one_sphere :
  ∃ S : Sphere, A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ F ∈ S := 
sorry

end endpoints_on_one_sphere_l685_685879


namespace factorization_l685_685379

theorem factorization (a b : ℤ) (h1 : a * b = -12) (h2 : 2 * b + a = 5) : a - b = -7 :=
by 
  sorry

end factorization_l685_685379


namespace expression_value_l685_685269

theorem expression_value : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end expression_value_l685_685269


namespace find_n_l685_685016

-- Define the sequence based on the given conditions
def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n % 3 = 0 then 2 + sequence (n / 3)
  else 1 / sequence (n - 1)

-- Prove that t_n = 5 / 116 implies n = 13
theorem find_n (n : ℕ) (h : sequence n = 5 / 116) : n = 13 := by 
  sorry

end find_n_l685_685016


namespace monotonic_decreasing_interval_l685_685606

def f (x : ℝ) : ℝ := x - x^2 - x

theorem monotonic_decreasing_interval :
  { x : ℝ | f' x < 0 } = set.Icc (-1/3 : ℝ) (1 : ℝ) := by
  sorry

end monotonic_decreasing_interval_l685_685606


namespace solve_system_equations_l685_685209

theorem solve_system_equations :
  ∀ (x y : ℝ),
  (y + sqrt (y - 3 * x) + 3 * x = 12 ∧ y^2 + y - 3 * x - 9 * x^2 = 144) ↔
  (x = -24 ∧ y = 72) ∨ (x = -4 / 3 ∧ y = 12) := by
  sorry

end solve_system_equations_l685_685209


namespace exponent_equality_l685_685894

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l685_685894


namespace constant_S13_l685_685594

theorem constant_S13 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a n = a 1 + (n - 1) * d) 
(h_sum : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
(h_constant : ∀ a1 d, (a 2 + a 8 + a 11 = 3 * a1 + 18 * d)) : (S 13 = 91 * d) :=
by
  sorry

end constant_S13_l685_685594


namespace solve_log_equation_l685_685679

theorem solve_log_equation (x : ℝ) (h : 0 < x ∧ x ≠ 1) :
  (log x (sqrt 2) - (log x (sqrt 2))^2 = log 3 27 - log x (2 * x)) ↔ (x = real.sqrt 2 ∨ x = real.sqrt (real.sqrt 2)) :=
by
  sorry

end solve_log_equation_l685_685679


namespace complete_square_l685_685655

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l685_685655


namespace Zoe_drank_bottles_l685_685282

variable (Initial Final Bought : ℕ)
variable (Drank : ℕ)

axiom initial_bottles : Initial = 42
axiom final_bottles : Final = 47
axiom bought_bottles : Bought = 30

theorem Zoe_drank_bottles : Initial - Drank + Bought = Final → Drank = 25 := by
  intros h
  rw [initial_bottles, final_bottles, bought_bottles] at h
  simp at h
  exact h.symm
  sorry

end Zoe_drank_bottles_l685_685282


namespace isosceles_triangle_base_function_l685_685413

theorem isosceles_triangle_base_function (x : ℝ) (hx : 5 < x ∧ x < 10) :
  ∃ y : ℝ, y = 20 - 2 * x := 
by
  sorry

end isosceles_triangle_base_function_l685_685413


namespace find_polar_coords_of_A_and_B_find_max_PA2_PB2_l685_685870

-- coordinates for the points in polar coordinates
def A_polar := (2 * Real.sqrt 3, -Real.pi / 6)
def B_polar := (2 * Real.sqrt 3, 5 * Real.pi / 6)

-- Parametric equations part
theorem find_polar_coords_of_A_and_B :
  ∃ (θA ρA θB ρB : ℝ),
  (θA, ρA) = A_polar ∧ (θB, ρB) = B_polar := by
  sorry

-- Curve-related part
def C2 (ρ θ : ℝ) : Prop :=
  ρ = 6 / Real.sqrt (9 - 3 * (Real.sin θ)^2)

theorem find_max_PA2_PB2 :
  ∀ θ : ℝ, 
  let P := (2 * Real.cos θ, Real.sqrt 6 * Real.sin θ) in
  let PA := (P.1 - 3, P.2 + Real.sqrt 3) in
  let PB := (P.1 + 3, P.2 - Real.sqrt 3) in
  (PA.1 ^ 2 + PA.2 ^ 2) + (PB.1 ^ 2 + PB.2 ^ 2) ≤ 36 := by
  sorry

end find_polar_coords_of_A_and_B_find_max_PA2_PB2_l685_685870


namespace fraction_simplification_l685_685439

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2 * d * e) / (d^2 + f^2 - e^2 + 3 * d * f) = (d + e - f) / (d + f - e) :=
sorry

end fraction_simplification_l685_685439


namespace total_cost_of_pets_l685_685634

theorem total_cost_of_pets 
  (num_puppies num_kittens num_parakeets : ℕ)
  (cost_parakeet cost_puppy cost_kitten : ℕ)
  (h1 : num_puppies = 2)
  (h2 : num_kittens = 2)
  (h3 : num_parakeets = 3)
  (h4 : cost_parakeet = 10)
  (h5 : cost_puppy = 3 * cost_parakeet)
  (h6 : cost_kitten = 2 * cost_parakeet) : 
  num_puppies * cost_puppy + num_kittens * cost_kitten + num_parakeets * cost_parakeet = 130 :=
by
  sorry

end total_cost_of_pets_l685_685634


namespace minimize_ω_l685_685363

def inv_prop_y1 : ℕ → ℝ := λ x, 48 / (x + 1)
def dir_prop_y2 : ℕ → ℝ := λ x, 3 * x + 1

def ω (x : ℕ) : ℝ := inv_prop_y1 x + dir_prop_y2 x

theorem minimize_ω : (∀ x : ℕ, ω x = (48 / (x + 1)) + (3 * x + 1)) ∧ (ω 3 = 22) :=
by
  sorry

end minimize_ω_l685_685363


namespace range_of_a_l685_685824

variable (a x : ℝ)

def p : Prop := -4 < x - a ∧ x - a < 4
def q : Prop := (x - 1) * (2 - x) > 0
def neg_p_sufficient_for_neg_q : Prop := ¬p → ¬q

theorem range_of_a : 
  neg_p_sufficient_for_neg_q a x → (-2 ≤ a ∧ a ≤ 5) := 
by
  sorry

end range_of_a_l685_685824


namespace cut_into_5_triangles_and_form_square_l685_685192

-- Define a type for figures on a graph paper
inductive Figure : Type
| depicted (graph_paper : Bool) : Figure

-- Assume a specific figure exists
def specific_figure : Figure := Figure.depicted true

-- Main statement: Prove that the specific figure can be cut into 5 triangles and assembled into a square
theorem cut_into_5_triangles_and_form_square (f : Figure) :
  f = specific_figure → ∃ (triangles : List Triangle), (triangles.length = 5 ∧ assemble_to_square triangles) :=
by
  intro h
  sorry

end cut_into_5_triangles_and_form_square_l685_685192


namespace factorize_expression_l685_685380

variable (a : ℝ)

theorem factorize_expression : a^3 - 2 * a^2 = a^2 * (a - 2) :=
by
  sorry

end factorize_expression_l685_685380


namespace four_points_partition_l685_685052

theorem four_points_partition (A B C D : Point) : 
  ∃ (G1 G2 : set Point), 
  G1 ∪ G2 = {A, B, C, D} ∧ 
  G1 ∩ G2 = ∅ ∧ 
  ¬∃ l : Line, G1 ⊆ { p | p ≠ side l } ∧ G2 ⊆ { p | p ≠ side l } :=
by
  sorry

end four_points_partition_l685_685052


namespace ratio_unavailable_l685_685145

noncomputable def builder_time_unknowable: Prop :=
  ∃ (homes_shops_ratio: ℕ → ℕ → ℕ), 
  ∀ (homes: ℕ) (shops: ℕ) (builder1_time: ℕ) (builder2_time: ℕ),
  homes = 100 ∧ shops ∈ {4, 5} ∧ builder2_time = 15 ∧
  (20 2’s are used in numbering 100 homes) →
  ¬ ∃ (ratio: ℤ), ratio = builder2_time / builder1_time

theorem ratio_unavailable : builder_time_unknowable :=
begin
  sorry
end

end ratio_unavailable_l685_685145


namespace all_participants_score_same_l685_685910

theorem all_participants_score_same (P : Type) [linear_order P] 
  (score : P → ℕ) 
  (defeated : P → set P)
  (h1 : ∀ p q : P, p ≠ q → (score p = score q ∨ score p ≠ score q))
  (h2 : ∀ p q : P, p ∈ defeated q → score q = 0)
  (coeff : P → ℕ := λ p, ∑ q in defeated p, score q)
  (h3 : ∀ p q : P, coeff p = coeff q)
  (h4 : ∃ p q r : P, p ≠ q ∧ q ≠ r ∧ r ≠ p) :
  ∀ p q : P, score p = score q :=
begin
  sorry
end

end all_participants_score_same_l685_685910


namespace line_perpendicular_to_Ax_By_C_l685_685126

theorem line_perpendicular_to_Ax_By_C (A B C x y : ℝ) :
  (∃ ℓ : (ℝ × ℝ) → Prop, 
    (∀ p, ℓ p → p = (x, y)) ∧ (∀ p, ℓ p → p = (x, y)) ∧ (∃ p₀, ℓ p₀) ∧ (B*(p.1 - x) - A*(p.2 - y) = 0)) :=
sorry

end line_perpendicular_to_Ax_By_C_l685_685126


namespace birches_count_l685_685140

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l685_685140


namespace brian_read_75_chapters_l685_685327

def book_chapters : Type :=
  {chapters : ℕ // chapters > 0}

def books_read : list book_chapters :=
 [
  ⟨20, by decide⟩,
  ⟨15, by decide⟩,
  ⟨15, by decide⟩,
  let sum_prev_books := 20 + 15 + 15 in
  ⟨sum_prev_books / 2, by decide⟩
]

theorem brian_read_75_chapters (books : list book_chapters) : 
  books.sum (λ b, b.chapters) = 75 :=
by 
  have h : books = 
    [ 
      ⟨20, by decide⟩, 
      ⟨15, by decide⟩, 
      ⟨15, by decide⟩, 
      let sum_prev_books := 20 + 15 + 15 in 
      ⟨sum_prev_books / 2, by decide⟩ 
    ] := 
    by reflexivity
  rw h
  unfold list.sum
  simp
  sorry

end brian_read_75_chapters_l685_685327


namespace greatest_possible_S_l685_685169

theorem greatest_possible_S {S : ℝ} 
  (h : ∀ (L : list ℝ), (∀ x ∈ L, 0 < x ∧ x ≤ 1) → L.sum = S →
     ∃ (A B : list ℝ), A ⊆ L ∧ B ⊆ L ∧ A ∩ B = [] ∧
       (A.sum ≤ 1 ∧ B.sum ≤ 5 ∨ B.sum ≤ 1 ∧ A.sum ≤ 5)) :
  S = 5.5 := 
sorry

end greatest_possible_S_l685_685169


namespace avg_age_9_proof_l685_685506

-- Definitions of the given conditions
def total_persons := 16
def avg_age_all := 15
def total_age_all := total_persons * avg_age_all -- 240
def persons_5 := 5
def avg_age_5 := 14
def total_age_5 := persons_5 * avg_age_5 -- 70
def age_15th_person := 26
def persons_9 := 9

-- The theorem to prove the average age of the remaining 9 persons
theorem avg_age_9_proof : 
  total_age_all - total_age_5 - age_15th_person = persons_9 * 16 :=
by
  sorry

end avg_age_9_proof_l685_685506


namespace maximum_possible_value_of_expression_l685_685151

theorem maximum_possible_value_of_expression :
  ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) ∧
  (b = 0 ∨ b = 1 ∨ b = 3 ∨ b = 4) ∧
  (c = 0 ∨ c = 1 ∨ c = 3 ∨ c = 4) ∧
  (d = 0 ∨ d = 1 ∨ d = 3 ∨ d = 4) ∧
  ¬ (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) →
  (c * a^b + d ≤ 196) :=
by sorry

end maximum_possible_value_of_expression_l685_685151


namespace smallest_angle_of_triangle_ratio_l685_685213

theorem smallest_angle_of_triangle_ratio (a b c : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : a + b + c = 12) :
  let x := 180 / 12
  in 3 * x = 45 :=
by
  sorry

end smallest_angle_of_triangle_ratio_l685_685213


namespace convert_degrees_to_seconds_l685_685689

noncomputable def degrees_to_seconds (d : ℝ) : ℝ :=
  d * 60 * 60

theorem convert_degrees_to_seconds (d : ℝ) (h : d = 1.45) : degrees_to_seconds d = 5220 :=
by {
  rw [h],
  simp [degrees_to_seconds],
  linarith,
}

end convert_degrees_to_seconds_l685_685689


namespace distance_between_parallel_lines_l685_685880

theorem distance_between_parallel_lines 
  (m : ℝ)
  (l1_eq : ∀ (x y : ℝ), 2 * x + 3 * m * y - m + 2 = 0)
  (l2_eq : ∀ (x y : ℝ), m * x + 6 * y - 4 = 0)
  (parallel : ∀ (x1 y1 x2 y2 : ℝ), 2 * x1 + 3 * m * y1 - m + 2 = 0 ∧ m * x2 + 6 * y2 - 4 = 0 → m ≠ 0 ∧ 2 / (3 * m) = m / 6) :
  ∀ d, d = |(-2) - 0| / Real.sqrt (1^2 + 3^2) → d = Real.sqrt 10 / 5 :=
by {
  sorry
}

end distance_between_parallel_lines_l685_685880


namespace parallel_vectors_perpendicular_vectors_magnitude_vectors_projection_vectors_l685_685883

def vector_a (λ : ℝ) : ℝ × ℝ := (5, λ)
def vector_b (λ : ℝ) : ℝ × ℝ := (λ - 2, 3)

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1
def are_perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2)
  (scalar * b.1, scalar * b.2)

theorem parallel_vectors (λ : ℝ) :
  are_parallel (vector_a λ) (vector_b λ) → (λ = -3 ∨ λ = 5) :=
sorry

theorem perpendicular_vectors (λ : ℝ) :
  are_perpendicular (vector_a λ) (vector_b λ) → λ ≠ 4 / 5 :=
sorry

theorem magnitude_vectors (λ : ℝ) :
  λ = 1 → vector_magnitude (vector_a λ + vector_b λ) = 4 * Real.sqrt 2 :=
sorry

theorem projection_vectors (λ : ℝ) :
  λ = 2 → vector_projection (vector_a λ) (vector_b λ) = (0, 2) :=
sorry

end parallel_vectors_perpendicular_vectors_magnitude_vectors_projection_vectors_l685_685883


namespace shift_quadratic_five_units_right_l685_685674

theorem shift_quadratic_five_units_right :
  let f := λ x : ℝ, 3*x^2 - x + 7
  let g := λ x : ℝ, 3*(x - 5)^2 - (x - 5) + 7
  a + b + c from (g x = ax^2 + bx + c) in calculate (3 - 31 + 87 = 59) :=
sorry

end shift_quadratic_five_units_right_l685_685674


namespace spherical_to_cartesian_correspondence_l685_685723

noncomputable def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

noncomputable def cartesian_to_spherical (x y z : ℝ) (ρ : ℝ) : ℝ × ℝ :=
  (Real.arctan2 y x, Real.arccos (z / ρ))

noncomputable def coordinates_transformation (ρ θ φ : ℝ) : (ℝ × ℝ) :=
let (x, y, z) := spherical_to_cartesian ρ θ φ in
let θ' := 2 * Real.pi - θ in
let φ' := Real.pi - φ in
(θ', φ')

theorem spherical_to_cartesian_correspondence :
  coordinates_transformation 3 (9 * Real.pi / 7) (Real.pi / 3) = (5 * Real.pi / 7, 2 * Real.pi / 3) :=
by {
  sorry
}

end spherical_to_cartesian_correspondence_l685_685723


namespace sum_prime_factors_57_l685_685265

theorem sum_prime_factors_57 : (∑ x in ({3, 19}: finset ℕ), x) = 22 :=
by
  sorry

end sum_prime_factors_57_l685_685265


namespace total_distance_walked_is_20_l685_685722

-- Let's define the problem statement in Lean 4.

theorem total_distance_walked_is_20 :
  ∃ (t1 t2 t3 : ℝ), 2 * t1 + t2 + t3 = 5 ∧
  (4 * t1 + 3 * t2) * 2 = 20 ∧
  (4 * t1 + 6 * t3) * 2 = 20 :=
begin
  sorry,
end

end total_distance_walked_is_20_l685_685722


namespace a_range_l685_685400

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem a_range (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ a ∈ Set.Ico (1/7 : ℝ) (1/3 : ℝ) :=
by
  sorry

end a_range_l685_685400


namespace find_C_D_l685_685049

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem find_C_D (C D : ℂ) (h₁ : 1 + C + D = 0) (h₂ : 1 + C * ω + D = 0) (h₃ : 1 + C * ω^2 + D = 0) :
  C = 0 ∧ D = -1 :=
by {
  -- Define necessary constants
  have hω : ω^3 = 1 := by {
    rw [← Complex.exp_nat_mul, mul_assoc, ← div_eq_mul_inv, div_mul_cancel],
    norm_num,
    exact two_ne_zero,
  },
  -- Solve using hypotheses
  add_three_eqs,
  exact sorry,
}

end find_C_D_l685_685049


namespace coconut_transport_l685_685757

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l685_685757


namespace part1_part2_l685_685467

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x ^ 2 + b) / (Real.sqrt (x ^ 2 + 1))

-- Prove the first part: ∀ a, if the minimum value of f is 3, then b ≥ 3
theorem part1 (a b : ℝ) (h : ∀ x : ℝ, f a b x ≥ 3) : b ≥ 3 :=
begin
  -- The proof goes here
  sorry 
end

-- Prove the second part: ∀ b ≥ 3, there exists a such that the minimum value of f is 3
theorem part2 (b : ℝ) (hb : b ≥ 3) : ∃ a : ℝ, ∀ x : ℝ, f a b x ≥ 3 ∧ (f a b x = 3 ↔ x = 0) :=
begin
  -- For given b, we find a such that the minimum value of f is 3
  use (b - Real.sqrt (b^2 - 9)) / 2,
  -- The proof goes here
  sorry 
end

end part1_part2_l685_685467


namespace sqrt_div_simplification_l685_685795

theorem sqrt_div_simplification (x y : ℝ) (h : (1/3)^2 + (1/4)^2 / (1/5)^2 + (1/6)^2 = 25 * x / (61 * y)) :
  sqrt x / sqrt y = 5 / 2 := 
sorry

end sqrt_div_simplification_l685_685795


namespace sample_size_l685_685508

theorem sample_size (questionnaires_distributed : ℕ) (sample_size_number : ℕ)
  (h1 : questionnaires_distributed = 30000)
  (h2 : sample_size_number = questionnaires_distributed) :
  sample_size_number = 30000 :=
by
  rw [← h1, h2]
  sorry

end sample_size_l685_685508


namespace no_point_in_common_l685_685902

theorem no_point_in_common (b : ℝ) :
  (∀ (x y : ℝ), y = 2 * x + b → (x^2 / 4) + y^2 ≠ 1) ↔ (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) :=
by
  sorry

end no_point_in_common_l685_685902


namespace proof_problem_l685_685671

def x : ℝ := 0.80 * 1750
def y : ℝ := 0.35 * 3000
def z : ℝ := 0.60 * 4500
def w : ℝ := 0.40 * 2800
def a : ℝ := z * w
def b : ℝ := x + y

theorem proof_problem : a - b = 3021550 := by
  sorry

end proof_problem_l685_685671


namespace muffin_sum_l685_685530

theorem muffin_sum (N : ℕ) : 
  (N % 13 = 3) → 
  (N % 8 = 5) → 
  (N < 120) → 
  (N = 16 ∨ N = 81 ∨ N = 107) → 
  (16 + 81 + 107 = 204) := 
by sorry

end muffin_sum_l685_685530


namespace correct_sampling_method_is_D_l685_685675

def is_simple_random_sample (method : String) : Prop :=
  method = "drawing lots method to select 3 out of 10 products for quality inspection"

theorem correct_sampling_method_is_D : 
  is_simple_random_sample "drawing lots method to select 3 out of 10 products for quality inspection" :=
sorry

end correct_sampling_method_is_D_l685_685675


namespace tenth_root_of_unity_condition_polynomial_with_smallest_roots_find_polynomial_equation_l685_685063

noncomputable def omega : ℂ := Complex.exp (Complex.I * Real.pi / 5)

theorem tenth_root_of_unity_condition : omega ^ 10 = 1 := sorry

theorem polynomial_with_smallest_roots :
  Polynomial.degree 
    ((Polynomial.X - omega) * (Polynomial.X - omega^3) * (Polynomial.X - omega^7) * (Polynomial.X - omega^9)) 
    = 4 := sorry

theorem find_polynomial_equation : 
  (Polynomial.X - omega) * (Polynomial.X - omega^3) * (Polynomial.X - omega^7) * (Polynomial.X - omega^9) 
  = Polynomial.X^4 - Polynomial.X^3 + Polynomial.X^2 - Polynomial.X + 1 := sorry

end tenth_root_of_unity_condition_polynomial_with_smallest_roots_find_polynomial_equation_l685_685063


namespace complex_quadrant_l685_685079

open Complex

theorem complex_quadrant (z : ℂ) (h : (1 + 2 * Complex.i) * z = 4 + 3 * Complex.i) : 
    z.re > 0 ∧ z.im < 0 :=
sorry

end complex_quadrant_l685_685079


namespace min_dwarfs_needed_l685_685144

theorem min_dwarfs_needed : ∀ (x y z : ℕ), 
  (1 <= x ∧ x <= 6) ∧ 
  (4 <= z ∧ z <= 9) ∧ 
  (x, y, z ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (z = x + 3) ∧ 
  (2 * x + y) % 3 = 0 → 
  ∃ d : ℕ, d = 19 :=
by
  intros x y z
  intro h
  sorry

end min_dwarfs_needed_l685_685144


namespace negation_of_universal_statement_l685_685609

variable (x : ℝ)

-- Original statement to be negated:
def original_statement := ∀ x ∈ set.Icc (0 : ℝ) 2, x^2 - 2 * x ≤ 0

-- Negation of the original statement:
def negated_statement := ∃ x ∈ set.Icc (0 : ℝ) 2, x^2 - 2 * x > 0

theorem negation_of_universal_statement :
  (¬ original_statement ≃ negated_statement) :=
sorry

end negation_of_universal_statement_l685_685609


namespace tv_cost_solution_l685_685289

theorem tv_cost_solution (M T : ℝ) 
  (h1 : 2 * M + T = 7000)
  (h2 : M + 2 * T = 9800) : 
  T = 4200 :=
by
  sorry

end tv_cost_solution_l685_685289


namespace sequence_is_linear_l685_685408

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom pos_terms (n : ℕ) : a (n + 1) > 0
axiom sum_of_terms (n : ℕ) : S (n + 1) = ∑ i in Finset.range (n + 1), a (i + 1)
axiom arithmetic_sequence (n : ℕ) : 2 * S (n + 1) = a (n + 1) + (a (n + 1))^2

-- Prove that a_n = n
theorem sequence_is_linear (n : ℕ) : a (n + 1) = n + 1 := 
  sorry

end sequence_is_linear_l685_685408


namespace range_of_a_l685_685950

theorem range_of_a
  (a : ℝ)
  (f : ℝ → ℝ)
  (hf : ∀ x, f (2^x) = x^2 - 2*a*x + a^2 - 1)
  (range_cond : ∀ x ∈ set.Icc (2^(a-1)) (2^(a^2 - 2*a + 2)), f x ∈ set.Icc (-1) 0)
  : a ∈ set.Icc ((3 - Real.sqrt 5) / 2) 1 ∪ set.Icc 2 ((3 + Real.sqrt 5) / 2) :=
sorry

end range_of_a_l685_685950


namespace subtract_045_from_3425_l685_685663

theorem subtract_045_from_3425 : 34.25 - 0.45 = 33.8 :=
by sorry

end subtract_045_from_3425_l685_685663


namespace problem_statement_l685_685504

noncomputable def total_questions_attempted 
  (marks_correct : ℕ) (marks_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) : ℕ :=
  let wrong_answers := marks_correct * correct_answers - total_marks
  in correct_answers + wrong_answers

theorem problem_statement :
  total_questions_attempted 4 1 130 38 = 60 :=
by
  unfold total_questions_attempted
  sorry

end problem_statement_l685_685504


namespace sqrt_neg4_sq_eq_4_l685_685274

theorem sqrt_neg4_sq_eq_4 : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := by
  sorry

end sqrt_neg4_sq_eq_4_l685_685274


namespace f_plus_g_eq_l685_685429

variables {R : Type*} [CommRing R]

-- Define the odd function f
def f (x : R) : R := sorry

-- Define the even function g
def g (x : R) : R := sorry

-- Define that f is odd and g is even
axiom f_odd (x : R) : f (-x) = -f x
axiom g_even (x : R) : g (-x) = g x

-- Define the given equation
axiom f_minus_g_eq (x : R) : f x - g x = x ^ 2 + 9 * x + 12

-- Statement of the goal
theorem f_plus_g_eq (x : R) : f x + g x = -x ^ 2 + 9 * x - 12 := by
  sorry

end f_plus_g_eq_l685_685429


namespace minute_hand_hour_hand_overlap_l685_685920

theorem minute_hand_hour_hand_overlap :
  ∃ x : ℕ, 0 < x ∧ 0.5 * x + 60 + 0.5 * 10 = 6 * x ∧ x = 11 := by
  sorry

end minute_hand_hour_hand_overlap_l685_685920


namespace find_k_l685_685344

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 2
def g (x k : ℝ) : ℝ := x^3 - 2 * x^2 + k * x - 10

theorem find_k :
  (f 5) - (g 5 9.4) = 0 :=
by
  calc
    f 5 = 5 * 5^2 - 3 * 5 + 2 : by sorry
    g 5 9.4 = 5^3 - 2 * 5^2 + 9.4 * 5 - 10 : by sorry
    112 - (65 + 5 * 9.4) = 0 : by sorry

end find_k_l685_685344


namespace factor_x_squared_minus_sixtyfour_l685_685377

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l685_685377


namespace find_n_l685_685898

theorem find_n (a b : ℤ) (h1 : a ≡ 18 [ZMOD 42]) (h2 : b ≡ 73 [ZMOD 42]) :
  ∃ n : ℤ, 100 ≤ n ∧ n < 142 ∧ a - b ≡ n [ZMOD 42] ∧ n = 113 :=
by
  use 113
  split
  · exact le_of_eq rfl
  · split
    · exact lt_of_le_of_lt le_of_eq rfl (by norm_num)
    · split
      · sorry
      · rfl

end find_n_l685_685898


namespace proof_problem_l685_685187

section MathProof

variable (x : ℝ)

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | log 0.5 x ≥ -1}
def B : Set ℝ := {x | |x| > 1}
def C_U_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem proof_problem : A ∩ C_U_B = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end MathProof

end proof_problem_l685_685187


namespace general_term_and_sum_min_value_ns_l685_685100

variable (y : ℝ → ℝ) (x : ℝ) (n : ℕ) (a : ℕ → ℝ) (Sn : ℕ → ℝ) (nS : ℕ → ℝ)

-- Define the given function
def f : ℝ → ℝ := λ x, x^2 - 2 * x - 11

-- Define the sequence {a_n}
def a_seq (n : ℕ) : ℝ := 2 * (n : ℝ) - 15

-- Define the sum of the first n terms, S_n
def S (n : ℕ) : ℝ := (n : ℝ)^2 - 14 * (n : ℝ)

-- Define the product nS_n
def nS (n : ℕ) : ℝ := (n : ℝ) * (S n)

-- Prove the general term formula and the sum of the first n terms
theorem general_term_and_sum :
  a_seq n = 2 * (n : ℝ) - 15 ∧ S n = (n : ℝ)^2 - 14 * (n : ℝ) :=
by
  sorry

-- Prove the minimum value of nS_n
theorem min_value_ns :
  ∃ (n_min : ℕ), nS n_min = -405 :=
by
  exists 9
  sorry

end general_term_and_sum_min_value_ns_l685_685100


namespace AD_DB_ratio_l685_685325

theorem AD_DB_ratio (AC BC : ℝ) (h1 : AC = BC) (h2 : AC = 1) (h3 : BC = 1) 
  (h4 : ∃ D E F, true) -- Placeholder for existence of points D, E, F to form the arc sector
  (h5 : ∀ area1 area2 : ℝ, area1 = area2 → area1 = (π * (2 / sqrt π)^2 / 8))
  : (2 * sqrt π : ℝ) / (π - 2) = (sqrt (2 * π) + 2 : ℝ) / (π - 2) :=
by
  sorry

end AD_DB_ratio_l685_685325


namespace ball_distribution_l685_685930

theorem ball_distribution (N a b : ℕ) (h1 : N = 6912) (h2 : N = 100 * a + b) (h3 : a < 100) (h4 : b < 100) : a + b = 81 :=
by
  sorry

end ball_distribution_l685_685930


namespace kem_hourly_wage_l685_685572

theorem kem_hourly_wage (shem_total_earnings: ℝ) (shem_hours_worked: ℝ) (ratio: ℝ)
  (h1: shem_total_earnings = 80)
  (h2: shem_hours_worked = 8)
  (h3: ratio = 2.5) :
  (shem_total_earnings / shem_hours_worked) / ratio = 4 :=
by 
  sorry

end kem_hourly_wage_l685_685572


namespace completing_the_square_solution_correct_l685_685641

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l685_685641


namespace tony_pool_filling_time_l685_685527

theorem tony_pool_filling_time
  (J S T : ℝ)
  (hJ : J = 1 / 30)
  (hS : S = 1 / 45)
  (hCombined : J + S + T = 1 / 15) :
  T = 1 / 90 :=
by
  -- the setup for proof would be here
  sorry

end tony_pool_filling_time_l685_685527


namespace incorrect_inequality_l685_685472

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by
  sorry

end incorrect_inequality_l685_685472


namespace portrait_in_silver_box_l685_685396

-- Definitions for the first trial
def gold_box_1 : Prop := false
def gold_box_2 : Prop := true
def silver_box_1 : Prop := true
def silver_box_2 : Prop := false
def lead_box_1 : Prop := false
def lead_box_2 : Prop := true

-- Definitions for the second trial
def gold_box_3 : Prop := false
def gold_box_4 : Prop := true
def silver_box_3 : Prop := true
def silver_box_4 : Prop := false
def lead_box_3 : Prop := false
def lead_box_4 : Prop := true

-- The main theorem statement
theorem portrait_in_silver_box
  (gold_b1 : gold_box_1 = false)
  (gold_b2 : gold_box_2 = true)
  (silver_b1 : silver_box_1 = true)
  (silver_b2 : silver_box_2 = false)
  (lead_b1 : lead_box_1 = false)
  (lead_b2 : lead_box_2 = true)
  (gold_b3 : gold_box_3 = false)
  (gold_b4 : gold_box_4 = true)
  (silver_b3 : silver_box_3 = true)
  (silver_b4 : silver_box_4 = false)
  (lead_b3 : lead_box_3 = false)
  (lead_b4 : lead_box_4 = true) : 
  (silver_box_1 ∧ ¬lead_box_2) ∧ (silver_box_3 ∧ ¬lead_box_4) :=
sorry

end portrait_in_silver_box_l685_685396


namespace jason_combination_count_l685_685164

theorem jason_combination_count :
  let digits := {1, 2, 3, 4, 5, 6}
  let odds := {1, 3, 5}
  let evens := {2, 4, 6}
  let valid_combinations := {combo : list ℕ // (∀ (i : ℕ) (h : i < 4), 
                                                (combo.nth i ∈ odds ↔ combo.nth (i + 1) ∈ evens) ∧ 
                                                (combo.nth i ∈ evens ↔ combo.nth (i + 1) ∈ odds)) ∧
                                                combo.all (λ x, x ∈ digits)}
  valid_combinations.card = 486 :=
sorry

end jason_combination_count_l685_685164


namespace exponent_tower_divisibility_l685_685207

theorem exponent_tower_divisibility (h1 h2 : ℕ) (Hh1 : h1 ≥ 3) (Hh2 : h2 ≥ 3) : 
  (2 ^ (5 ^ (2 ^ (5 ^ h1))) + 4 ^ (5 ^ (4 ^ (5 ^ h2)))) % 2008 = 0 := by
  sorry

end exponent_tower_divisibility_l685_685207


namespace option_A_option_C_l685_685086

variable {α : Type*} [LinearOrderedField α]

def star (a b : α) : α := a ^ (b + 1)

theorem option_A (a b : α) (ha : 0 < a) (hb : 0 < b) : star a b = a ^ (b + 1) := 
by simp [star]

theorem option_C (a b : α) (n : ℕ) (ha : 0 < a) (hb : 0 < b) : star a (b ^ n) = a ^ (b ^ n + 1) :=
by simp [star]

-- Additional declarations to skip proof
#check star
#check option_A
#check option_C

end option_A_option_C_l685_685086


namespace harold_number_l685_685366

def skips (n : ℕ) (k : ℕ) : ℕ := 4 * n - k - 1

theorem harold_number :
  ∃ n, (4 (4 (4 (4 (4 (4 n - 1) - 1) - 1) - 1) - 1) - 1) = 239 :=
sorry

end harold_number_l685_685366


namespace positive_t_value_l685_685398

theorem positive_t_value (t : ℝ) (h : abs (-7 + t * Complex.i) = 15) : t = 4 * Real.sqrt 11 := 
sorry

end positive_t_value_l685_685398


namespace completing_the_square_solution_l685_685661

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l685_685661


namespace geometric_sequence_sum_l685_685935

theorem geometric_sequence_sum :
  (∃ (q > 1) (a : ℕ → ℝ) (h1: a 2016 = 1/2) (h2: a 2017 = 3/2), 
    (∀ n, a (n + 1) = q * a n) → a 2018 + a 2019 = 18) :=
by
  sorry

end geometric_sequence_sum_l685_685935


namespace sum_b_n_l685_685549

-- Define the sequence a_n
noncomputable def a : ℕ → ℚ
| 0     := 1
| (n+1) := 4 / (4 - a n)

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℚ := a (2 * n + 1) / a (2 * n)

-- Define the sum T_n
noncomputable def T (n : ℕ) : ℚ := ∑ i in range n, b i

-- The statement to prove
theorem sum_b_n (n : ℕ) : T n = n + 1/2 * (1 - 1/(2 * n + 1)) :=
sorry

end sum_b_n_l685_685549


namespace polynomial_non_divisible_l685_685199

theorem polynomial_non_divisible (A : ℤ) (n m : ℕ) :
  ¬ ∃ P : Polynomial ℚ, 2 * X^(2 * m) + (Polynomial.C ↑A) * X^m + 3 = (3 * X^(2 * n) + (Polynomial.C ↑A) * X^n + 2) * P := sorry

end polynomial_non_divisible_l685_685199


namespace simplify_polynomial_l685_685576

theorem simplify_polynomial : 
  (12 * X^10 - 3 * X^9 + 8 * X^8 - 5 * X^7) - (2 * X^10 + 2 * X^9 - X^8 + X^7 + 4 * X^4 + 6 * X^2 + 9) 
  = 10 * X^10 - 5 * X^9 + 9 * X^8 - 6 * X^7 - 4 * X^4 - 6 * X^2 - 9 :=
by
  sorry

end simplify_polynomial_l685_685576


namespace min_val_alpha_beta_l685_685470

theorem min_val_alpha_beta (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (alpha : ℝ) (beta : ℝ)
  (h_alpha : alpha = a + 1/a) (h_beta : beta = b + 1/b) : alpha + beta ≥ 4 :=
begin
  sorry
end

end min_val_alpha_beta_l685_685470


namespace maximize_net_profit_l685_685981

noncomputable def p (t : ℝ) : ℝ :=
  if 2 ≤ t ∧ t < 10 then 1200 - 10 * (10 - t) ^ 2
  else if 10 ≤ t ∧ t ≤ 20 then 1200
  else 0

noncomputable def Q (t : ℝ) : ℝ :=
  (6 * p t - 3360) / t - 360

theorem maximize_net_profit :
  ∃ t : ℝ, 2 ≤ t ∧ t ≤ 20 ∧ (∀ x : ℝ, 2 ≤ x ∧ x ≤ 20 → Q x ≤ Q 6) ∧ Q 6 = 120 :=
begin
  use 6,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { intro x,
    intro h,
    sorry },
  { exact dec_trivial }
end

end maximize_net_profit_l685_685981


namespace angle_C_is_pi_over_3_perimeter_of_triangle_ABC_l685_685524

-- Define the problem and conditions.
variables {A B C : ℝ} (a b c : ℝ)
hypothesis (h1 : a * Real.cos B + b * Real.cos A = c / (2 * Real.cos C))
hypothesis (h2 : c = 6)
hypothesis (h3 : exists h : ℝ, h = 2 * Real.sqrt 3)

-- Prove the first part: Given the equation, we find that angle C is π/3.
theorem angle_C_is_pi_over_3 : C = Real.pi / 3 := by sorry

-- Prove the second part: Given specific conditions, calculate the perimeter.
theorem perimeter_of_triangle_ABC : a + b + c = 6 * Real.sqrt 3 + 6 := by sorry

end angle_C_is_pi_over_3_perimeter_of_triangle_ABC_l685_685524


namespace max_value_sqrt_sum_l685_685809

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  ∃ y, y = sqrt (49 + x) + sqrt (49 - x) ∧ y ≤ 14 :=
by
  sorry

end max_value_sqrt_sum_l685_685809


namespace area_of_equilateral_triangle_with_given_altitude_eq_expected_l685_685588

-- Define the altitude of the triangle
def altitude : ℝ := Real.sqrt 8

-- Define the required area of the triangle
def expected_area : ℝ := 8 * Real.sqrt 3

-- Prove that the area of the equilateral triangle with the given altitude is as expected
theorem area_of_equilateral_triangle_with_given_altitude_eq_expected :
  ∃ (A : ℝ), altitude = Real.sqrt 8 ∧ A = (∃ (BC : ℝ), BC = 2 * (altitude / Real.sqrt 3) ∧ A = (1/2) * BC * altitude) :=
sorry

end area_of_equilateral_triangle_with_given_altitude_eq_expected_l685_685588


namespace floor_T_is_179_l685_685545

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry
noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry

axiom pqrs_pos (p q r s : ℝ) : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s
axiom eq1 (p q : ℝ) : p^2 + q^2 = 4016
axiom eq2 (r s : ℝ) : r^2 + s^2 = 4016
axiom eq3 (p r : ℝ) : p * r = 2000
axiom eq4 (q s : ℝ) : q * s = 2000

theorem floor_T_is_179
  (h1 : pqrs_pos p q r s)
  (h2 : eq1 p q)
  (h3 : eq2 r s)
  (h4 : eq3 p r)
  (h5 : eq4 q s) :
  (⌊p + q + r + s⌋ : ℤ) = 179 := 
sorry

end floor_T_is_179_l685_685545


namespace data_properties_l685_685995

-- Defining the list of numbers
def data : List ℝ := [-5, 3, 2, -3, 3]

-- Defining the mean function
noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Defining the mode function
noncomputable def mode (l : List ℝ) : ℝ :=
  l.mode

-- Defining the median function
noncomputable def median (l : List ℝ) : ℝ :=
  l.median

-- Defining the variance function
noncomputable def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ)^2)).sum / l.length

-- Define the theorem to be proven
theorem data_properties :
  mean data = 0 ∧
  mode data = 3 ∧
  median data = 2 ∧
  variance data = 11.2 :=
by
  sorry

end data_properties_l685_685995


namespace f_zero_f_odd_a_range_l685_685939

variable {ℝ : Type*} [OrderedAddCommGroup ℝ] [TopologicalSpace ℝ] [TopologicalAddGroup ℝ]

noncomputable def f (x : ℝ) : ℝ := sorry

-- Condition: f satisfies the Cauchy functional equation
axiom f_eqn : ∀ x y : ℝ, f(x + y) = f(x) + f(y)

-- Condition: f is increasing
axiom f_increasing : ∀ x y : ℝ, x < y → f(x) < f(y)

-- Condition: f(1) = 1
@[simp] axiom f_one : f(1) = 1

-- Condition: f(2a) > f(a-1) + 2
axiom f_2a : ∀ a : ℝ, f(2 * a) > f(a - 1) + 2

-- Proof for f(0) = 0
theorem f_zero : f(0) = 0 := sorry

-- Proof that f(x) is an odd function
theorem f_odd (x : ℝ) : f(-x) = -f(x) := sorry

-- Find the range of values for a
theorem a_range (a : ℝ) : a > 1 ↔ f(2 * a) > f(a - 1) + 2 := sorry

end f_zero_f_odd_a_range_l685_685939


namespace simplify_expression_l685_685690

theorem simplify_expression :
  2^(-1) - real.sqrt 3 * real.tan (real.pi / 3) + (real.pi - 2011)^0 + abs (-1 / 2) = -1 :=
by
  -- Using given conditions:
  have h1 : 2^(-1) = 1 / 2, from sorry,
  have h2 : real.tan (real.pi / 3) = real.sqrt 3, from sorry,
  have h3 : (real.pi - 2011)^0 = 1, from sorry,
  have h4 : abs (-1 / 2) = 1 / 2, from sorry,
  -- Applying these conditions should help complete the proof:
  sorry

end simplify_expression_l685_685690


namespace total_chapters_brian_read_l685_685329

theorem total_chapters_brian_read :
  let chapters_first_two_books := 2 * 15 in
  let chapters_third_book := 20 in
  let total_first_three_books := chapters_first_two_books + chapters_third_book in
  let chapters_last_book := total_first_three_books / 2 in
  let total_chapters := chapters_first_two_books + chapters_third_book + chapters_last_book in
  total_chapters = 75 := by
  sorry

end total_chapters_brian_read_l685_685329


namespace dave_did_not_wash_16_shirts_l685_685018

theorem dave_did_not_wash_16_shirts
  (short_sleeve_shirts : ℕ)
  (long_sleeve_shirts : ℕ)
  (shirts_washed : ℕ)
  (total_shirts := short_sleeve_shirts + long_sleeve_shirts)
  (shirts_not_washed := total_shirts - shirts_washed) :
  short_sleeve_shirts = 9 →
  long_sleeve_shirts = 27 →
  shirts_washed = 20 →
  shirts_not_washed = 16 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end dave_did_not_wash_16_shirts_l685_685018


namespace completing_the_square_l685_685645

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l685_685645


namespace domain_of_f_symmetry_of_f_range_f_greater_zero_l685_685096

open Real

noncomputable def f (a x : ℝ) : ℝ := log a (1 - 2 * x) - log a (1 + 2 * x)

theorem domain_of_f (a : ℝ) (ha : 0 < a) (h₁ : a ≠ 1) : 
  {x : ℝ | x ∈ set.Ioo (-1/2 : ℝ) (1/2 : ℝ)} = 
  {x : ℝ | x ∈ domain_of (f a x)} :=
sorry

theorem symmetry_of_f (a : ℝ) (ha : 0 < a) (h₁ : a ≠ 1) :
  ∀ x, f a (-x) = - (f a x) :=
sorry

theorem range_f_greater_zero (a : ℝ) (ha : 0 < a) (h₁ : a ≠ 1) :
  if h₂ : 1 < a then
    {x : ℝ | (-1/2 : ℝ) < x ∧ x < 0 ∧ f a x > 0} = 
    {x : ℝ | x ∈ set.Ioo (-1/2 : ℝ) 0}
  else
    {x : ℝ | 0 < x ∧ x < (1/2 : ℝ) ∧ f a x > 0} = 
    {x : ℝ | x ∈ set.Ioo 0 (1/2 : ℝ)} :=
sorry

end domain_of_f_symmetry_of_f_range_f_greater_zero_l685_685096


namespace quadrilateral_perimeter_l685_685017

def is_cyclic_quadrilateral (A B C D : Point) : Bool := sorry

def angle (A B C : Point) : ℝ := sorry

def distance (A B : Point) : ℝ := sorry

noncomputable def perimeter (A B C D : Point) : ℝ := 
  distance A B + distance B C + distance C D + distance D A

theorem quadrilateral_perimeter 
  (A B C D E F : Point) 
  (H1: is_cyclic_quadrilateral A B C D = true)
  (H2: angle D A B = 60)
  (H3: distance B C = 1)
  (H4: distance C D = 1)
  (H5: ∑ p in [A B E, E B C], perimeter p ∈ ℤ)
  (H6: ∑ p in [C D F, F D C], perimeter p ∈ ℤ):
  perimeter A B C D = 38 / 7 := 
sorry

end quadrilateral_perimeter_l685_685017


namespace beta_minus_alpha_eq_pi_over_four_l685_685462

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables (α β : ℝ)
variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) := dot_product v1 v2 = 0

def a_vec := vector (sqrt 2 * cos α) (sqrt 2 * sin α)
def b_vec := vector (2 * cos β) (2 * sin β)
def diff_vec := vector ((2 * cos β) - (sqrt 2 * cos α)) ((2 * sin β) - (sqrt 2 * sin α))

theorem beta_minus_alpha_eq_pi_over_four
  (h1 : a = a_vec)
  (h2 : b = b_vec)
  (h3 : (π / 6) ≤ α ∧ α < (π / 2) ∧ (π / 2) < β ∧ β ≤ (5 * π / 6))
  (h4 : perpendicular a (b - a)) :
  β - α = (π / 4) :=
sorry

end beta_minus_alpha_eq_pi_over_four_l685_685462


namespace range_of_h_l685_685021

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x ^ 2)

theorem range_of_h (a b : ℝ) (h_range : set.Ioo a b = {x | 0 < h x ∧ h x ≤ 1}) : a + b = 1 :=
by
  have a_zero : a = 0 :=
    sorry
  have b_one : b = 1 :=
    sorry
  rw [a_zero, b_one]
  norm_num

end range_of_h_l685_685021


namespace completing_the_square_solution_l685_685659

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l685_685659


namespace infinite_composite_of_form_2_pow_n_minus_1_l685_685959

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem infinite_composite_of_form_2_pow_n_minus_1 :
  ∀ (n : ℕ), is_odd n ∧ is_composite n → ∃ m, 2^m - 1 = 2^n - 1 ∧ is_composite (2^n - 1) ∧ m = n :=
by
  assume n h,
  sorry

end infinite_composite_of_form_2_pow_n_minus_1_l685_685959


namespace rita_butterfly_hours_l685_685509

theorem rita_butterfly_hours (total_hours : ℕ) (backstroke_hours : ℕ) (breaststroke_hours : ℕ) 
    (monthly_practice_hours : ℕ) (months : ℕ): total_hours = 1500 ∧ backstroke_hours = 50 ∧ breaststroke_hours = 9 ∧ 
    monthly_practice_hours = 220 ∧ months = 6 → 
  let hours_needed_for_butterfly := total_hours - (backstroke_hours + breaststroke_hours + monthly_practice_hours * months) in
    hours_needed_for_butterfly = 121 :=
by
  intros h
  cases h with h1 rest
  cases rest with h2 rest
  cases rest with h3 rest
  cases rest with h4 h5
  sorry

end rita_butterfly_hours_l685_685509


namespace mural_lunch_break_duration_l685_685718

variable (a t L : ℝ)

theorem mural_lunch_break_duration
  (h1 : (8 - L) * (a + t) = 0.6)
  (h2 : (6.5 - L) * t = 0.3)
  (h3 : (11 - L) * a = 0.1) :
  L = 40 :=
by
  sorry

end mural_lunch_break_duration_l685_685718


namespace find_a_range_l685_685842

variable (a : ℝ)

def p : Prop :=
  let line_dist_to_center := abs (1 + a) / Real.sqrt 2
  line_dist_to_center < Real.sqrt 2

def q : Prop :=
  ∀ (x : ℝ), exp x - a > 1

theorem find_a_range (hp : p a) (hnq : ¬q a) : a ∈ Ioo (-1) 1 := by
  sorry

end find_a_range_l685_685842


namespace cubic_polynomial_roots_l685_685034

variables (a b c : ℚ)

theorem cubic_polynomial_roots (a b c : ℚ) :
  (c = 0 → ∃ x y z : ℚ, (x = 0 ∧ y = 1 ∧ z = -2) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) ∧
  (c ≠ 0 → ∃ x y z : ℚ, (x = 1 ∧ y = -1 ∧ z = -1) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) :=
by
  sorry

end cubic_polynomial_roots_l685_685034


namespace minutes_until_midnight_l685_685159

/-
  Problem: It has just turned 22:22. How many minutes are there until midnight?
  Ensure using condition and answer without steps.
-/

theorem minutes_until_midnight (h : time = mk_time 22 22) : minutes_until_midnight h = 98 := by
  sorry

end minutes_until_midnight_l685_685159


namespace completing_the_square_solution_correct_l685_685637

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l685_685637


namespace union_of_sets_l685_685548

open Set

theorem union_of_sets (a : ℝ) :
  let A := {1, 3}
  let B := {a + 2, 5}
  A ∩ B = {3} → A ∪ B = {1, 3, 5} :=
by
  intros h
  -- We include sorry here to skip the proof.
  sorry

end union_of_sets_l685_685548


namespace sin_25alpha_form_l685_685919

theorem sin_25alpha_form (α : ℝ) (h : Real.sin α = 3 / 5) : ∃ (n : ℤ), Real.sin (25 * α) = n / 5^25 ∧ ∃ k : ℤ, n = 3^25 * k + 1 ∨ n = -(3^25 * k + 1) :=
begin
  sorry
end

end sin_25alpha_form_l685_685919


namespace a_107_result_l685_685121

noncomputable def a_sequence (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then 1 / (1 - x)
  else if n = 2 then  1 / (1 - (1 / (1 - x)))
  else a_sequence_aux (n-1)

noncomputable def a_sequence_aux (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then 1 / (1 - x)
  else 1 / (1 - a_sequence (n-1) x)

theorem a_107_result (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  a_sequence 107 x = (x - 1) / x :=
  sorry

end a_107_result_l685_685121


namespace completing_the_square_solution_l685_685660

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l685_685660


namespace dot_product_eq_neg23_l685_685800

-- Definitions of the vectors based on given conditions.
def u : ℝ × ℝ × ℝ := (4, -3, 0)
def v : ℝ × ℝ × ℝ := (-2, 5, 8)

-- Statement to prove that the dot product of u and v is -23.
theorem dot_product_eq_neg23 : u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = -23 := by
  sorry

end dot_product_eq_neg23_l685_685800


namespace birch_trees_count_l685_685138

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l685_685138


namespace completing_the_square_l685_685644

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l685_685644


namespace inequality_for_m_l685_685082

def f (x m : ℝ) : ℝ := sqrt 6 * sin (x / 2 + π / 6) - m

theorem inequality_for_m (m : ℝ) : (∀ x, -5 * π / 6 ≤ x ∧ x ≤ π / 6 → f x m ≤ 0) → m ≥ sqrt 3 :=
by
  sorry

end inequality_for_m_l685_685082


namespace line_through_point_parallel_to_line_l685_685987

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end line_through_point_parallel_to_line_l685_685987


namespace most_appropriate_sampling_l685_685312

def total_students := 126 + 280 + 95
def adjusted_total_students := 126 - 1 + 280 + 95
def required_sample_size := 100

def elementary_proportion (total : Nat) (sample : Nat) : Nat := (sample * 126) / total
def middle_proportion (total : Nat) (sample : Nat) : Nat := (sample * 280) / total
def high_proportion (total : Nat) (sample : Nat) : Nat := (sample * 95) / total

theorem most_appropriate_sampling :
  required_sample_size = elementary_proportion adjusted_total_students required_sample_size + 
                         middle_proportion adjusted_total_students required_sample_size + 
                         high_proportion adjusted_total_students required_sample_size :=
by
  sorry

end most_appropriate_sampling_l685_685312


namespace f_val_4_over_3_l685_685059

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then cos (Real.pi * x) else f (x - 1) + 1

theorem f_val_4_over_3 : f (4 / 3) = 3 / 2 := sorry

end f_val_4_over_3_l685_685059


namespace tom_needs_495_boxes_l685_685254

-- Define the conditions
def total_chocolate_bars : ℕ := 3465
def chocolate_bars_per_box : ℕ := 7

-- Define the proof statement
theorem tom_needs_495_boxes : total_chocolate_bars / chocolate_bars_per_box = 495 :=
by
  sorry

end tom_needs_495_boxes_l685_685254


namespace construction_of_triangle_from_polygon_centers_l685_685783

noncomputable def triangle_properties (n : ℕ) (X Y Z A B C : ℝ) : Prop :=
  ∃ (X Y Z A B C : ℝ), 
    (∃ α : ℝ, α = 2 * real.pi / n ∧ 
    (rotation(X, α) = Y ∧ rotation(Y, α) = Z ∧ rotation(Z, α) = X)) ∧
    (((n = 3) ∧ (are_vertices_of_regular_triangle X Y Z) → 
    ∃∞ (A B C : ℝ), is_unique_triangle ABC XYZ) ∧ 
    ((n ≥ 4) → (unique_triangle ABC XYZ)))

def prove_triangle_properties (n : ℕ) (X Y Z A B C: ℝ) :=
  triangle_properties n X Y Z A B C 

theorem construction_of_triangle_from_polygon_centers:
  ∀ (n : ℕ) (X Y Z A B C : ℝ),
  (∀ (n : ℕ) (X Y Z A B C : ℝ),
   (prove_triangle_properties n X Y Z A B C)) :=
by sorry

end construction_of_triangle_from_polygon_centers_l685_685783


namespace point_pairs_distance_leq_bound_l685_685062

theorem point_pairs_distance_leq_bound (n : ℕ) (points : Fin n → ℝ × ℝ) :
  let point_pairs_with_distance_1 := 
    { (i, j) | i < j ∧ dist (points i) (points j) = 1 } in
  point_pairs_with_distance_1.card ≤ (2 / Real.sqrt 7) * n ^ (3 / 2) :=
by
  sorry

end point_pairs_distance_leq_bound_l685_685062


namespace william_max_moves_l685_685952

-- Define the possible moves
def moveA (S : ℕ) : ℕ := 2 * S + 1
def moveB (S : ℕ) : ℕ := 4 * S + 3

-- Define the starting condition
def initial_value : ℕ := 1

-- Define the maximum value threshold
def max_value : ℕ := 2^100

-- Define the conditions for optimal play and the question
theorem william_max_moves (optimal_play : Π (current_value moves : ℕ) (is_mark_turn : Bool), 
  (current_value < max_value) → current_value = initial_value →
  (moves : Mark, William optimal_play), (maximum_moves  optimal_play : )  : 
  moves Mark > moves William: ) : 
  (maximum_plays_william < maximum  William := 
sorry

end william_max_moves_l685_685952


namespace sum_first_seven_terms_l685_685485

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 a2 a3 a4 a5 a6 a7 : ℝ)

-- Assuming a is an arithmetic sequence with common difference d
axiom (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)

-- Given conditions
axiom (h_cond : a 3 + a 4 + a 5 = 12)

open Nat

theorem sum_first_seven_terms : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
  sorry

end sum_first_seven_terms_l685_685485


namespace f_derivative_at_1_intervals_of_monotonicity_l685_685093

def f (x : ℝ) := x^3 - 3 * x^2 + 10
def f' (x : ℝ) := 3 * x^2 - 6 * x

theorem f_derivative_at_1 : f' 1 = -3 := by
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x < 0 → f' x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, x > 2 → f' x > 0) := by
  sorry

end f_derivative_at_1_intervals_of_monotonicity_l685_685093


namespace minimum_value_of_MP_MQ_is_14_div_5_l685_685080

noncomputable def minimum_value_MP_MQ (O P Q M : Point) (hO : O = (0, 0))
  (hP : P ∈ {p : Point | p.1^2 + p.2^2 = 4})
  (hQ : Q ∈ {q : Point | q.1^2 + q.2^2 = 4})
  (hAngle : angle O P Q = 2 * Real.pi / 3)
  (hM : M ∈ {m : Point | 3 * m.1 + 4 * m.2 = 12}) : ℝ :=
|vec_M_vec_P_plus_vec_M_vec_Q : 
  (distance M P + distance M Q = 2 * distance (M / 2) O) :=
  sorry

/-- Proof that given the conditions, the minimum value of |vec_MP + vec_MQ| is 14/5. -/
theorem minimum_value_of_MP_MQ_is_14_div_5 (O P Q M : Point) (hO : O = (0,0)) 
  (hP: P ∈ {p : Point | p.1^2 + p.2^2 = 4}) 
  (hQ : Q ∈ {q : Point | q.1^2 + q.2^2 = 4}) 
  (hAngle : angle O P Q = 2 * Real.pi / 3) 
  (hM : M ∈ {m : Point | 3 * m.1 + 4 * m.2 = 12}) :
  minimum_value_MP_MQ O P Q M hO hP hQ hAngle hM = 14 / 5 :=
sorry

end minimum_value_of_MP_MQ_is_14_div_5_l685_685080


namespace absolute_value_is_four_l685_685488

-- Given condition: the absolute value of a number equals 4
theorem absolute_value_is_four (x : ℝ) : abs x = 4 → (x = 4 ∨ x = -4) :=
by
  sorry

end absolute_value_is_four_l685_685488


namespace surface_eq_l685_685048

theorem surface_eq (f : ℝ → ℝ → ℝ → ℝ) (x y z : ℝ) :
  (∀ x y z, (f x y z = (dx / (yz)) = (dy / (zx)) = (dz / (xy)))) →
  (x = 0 → y^2 + z^2 = 1) →
  (y^2 + z^2 = 1 + 2 * x^2) :=
by
  intros h_eq h_circle
  sorry

end surface_eq_l685_685048


namespace point_on_curve_l685_685803

theorem point_on_curve (θ : ℝ) :
  (sin (2 * θ) = -3 / 4 ∧ cos θ + sin θ = 1 / 2) →
  (∃ θ : ℝ, (sin (2 * θ) = -3 / 4 ∧ cos θ + sin θ = 1 / 2)) :=
sorry

end point_on_curve_l685_685803


namespace product_of_positive_solutions_product_of_all_positive_integral_values_l685_685045

theorem product_of_positive_solutions (n : ℕ) : 
  (∃ p, prime p ∧ (n^2 - 34 * n + 300 = p)) → (n = 14 ∨ n = 20) :=
by
  sorry

theorem product_of_all_positive_integral_values :
  let n1 := 14
  let n2 := 20
  n1 * n2 = 298 :=
by
  sorry

end product_of_positive_solutions_product_of_all_positive_integral_values_l685_685045


namespace min_value_reciprocal_sum_is_four_l685_685432

noncomputable def min_value_reciprocal_sum (x y : ℝ) (h : Real.ln (x + y) = 0) : ℝ :=
  min (1 / x + 1 / y)

theorem min_value_reciprocal_sum_is_four (x y : ℝ) (h : Real.ln (x + y) = 0) :
  min_value_reciprocal_sum x y h = 4 :=
sorry

end min_value_reciprocal_sum_is_four_l685_685432


namespace simplify_expr_range_values_a_values_satisfy_equation_l685_685203

-- For Problem 1
theorem simplify_expr (a : ℝ) (h1 : 3 ≤ a ∧ a ≤ 7) : 
  (sqrt ((3 - a) ^ 2) + sqrt ((a - 7) ^ 2) = 4) :=
sorry

-- For Problem 2
theorem range_values_a (a : ℝ) : 
  (sqrt ((a - 1) ^ 2) + sqrt ((a - 6) ^ 2) = 5) ↔ (1 ≤ a ∧ a ≤ 6) :=
sorry

-- For Problem 3
theorem values_satisfy_equation (a : ℝ) : 
  (sqrt ((a + 1) ^ 2) + sqrt ((a - 3) ^ 2) = 6) ↔ (a = -2 ∨ a = 4) :=
sorry

end simplify_expr_range_values_a_values_satisfy_equation_l685_685203


namespace excluded_values_of_m_l685_685479

theorem excluded_values_of_m (m : ℝ) : 
  m ∈ Set.Union (λ x : ℝ, { x, x^2 + 3*x }) ↔ (m ≠ 0 ∧ m ≠ -2) :=
by 
  -- The actual proof steps go here
  sorry

end excluded_values_of_m_l685_685479


namespace shortest_altitude_of_right_triangle_l685_685236

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l685_685236


namespace increasing_range_a_extremum_value_at_3_l685_685098

-- Statement for (Ⅰ)
theorem increasing_range_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → 3*x^2 - 2*a*x + 3 ≥ 0) ↔ (a ≤ 0) :=
sorry

-- Statement for (Ⅱ)
theorem extremum_value_at_3 (a : ℝ) (h : 3*(3:ℝ)^2 - 2*a*3 - 3 = 0) : 
  f (3:ℝ) = (3^3 - a*3^2 + 3*3) :=
sorry

-- Definition of the function f(x)
def f (x : ℝ) := x^3 - a*x^2 + 3*x

end increasing_range_a_extremum_value_at_3_l685_685098


namespace prob_6_lt_X_lt_7_l685_685817

noncomputable theory

-- Define normal distributions and the properties involved
def normal_dist (μ σ : ℝ) := sorry -- Placeholder for an actual normal distribution definition

variable (X : ℝ → Prop) (μ σ : ℝ)

-- Given conditions
axiom normal_properties_1 : ∀ (Z : ℝ → Prop) (μ σ : ℝ),
  ( ∀ z, Z z ↔ normal_dist μ σ z ) →
  ( ∀ z, z ∈ set.Ioo (μ - σ) (μ + σ) → P(z) = 0.6826 ) →
  true

axiom normal_properties_2 : ∀ (Z : ℝ → Prop) (μ σ : ℝ),
  ( ∀ z, Z z ↔ normal_dist μ σ z ) →
  ( ∀ z, z ∈ set.Ioo (μ - 2*σ) (μ + 2*σ) → P(z) = 0.9544 ) →
  true

-- Define specific instance for X
def X_normal_dist : Prop := normal_dist 5 1

-- The proof problem in Lean 4
theorem prob_6_lt_X_lt_7 : 
  ( ∀ x, x ∈ set.Ioo (5 - 1) (5 + 1) → P(x) = 0.6826 ) →
  ( ∀ x, x ∈ set.Ioo (5 - 2*1) (5 + 2*1) → P(x) = 0.9544 ) →
  P(6 < X ∧ X < 7) = 0.1359 :=
by
  intros
  sorry

end prob_6_lt_X_lt_7_l685_685817


namespace birches_count_l685_685141

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end birches_count_l685_685141


namespace cos_arcsin_l685_685009

theorem cos_arcsin (h₀ : real.sqrt (5^2 - 3^2) = 4) : real.cos (real.arcsin (3 / 5)) = 4 / 5 :=
by 
  sorry

end cos_arcsin_l685_685009


namespace sin_alpha_plus_beta_tan_alpha_minus_beta_l685_685424

variable {α β : ℝ}

-- Define the conditions
def sin_alpha : ℝ := 4 / 5
def cos_beta : ℝ := 12 / 13
def acute_angles : Prop := 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2

-- Prove sin(α + β) == 63 / 65 given the conditions
theorem sin_alpha_plus_beta :
  acute_angles →
  sin α = sin_alpha →
  cos β = cos_beta →
  sin (α + β) = 63 / 65 := by
  sorry

-- Prove tan(α - β) == 33 / 56 given the conditions
theorem tan_alpha_minus_beta :
  acute_angles →
  sin α = sin_alpha →
  cos β = cos_beta →
  tan (α - β) = 33 / 56 := by
  sorry

end sin_alpha_plus_beta_tan_alpha_minus_beta_l685_685424


namespace even_partition_dominoes_l685_685978

theorem even_partition_dominoes 
  (m n a b : ℕ) 
  (h_pos_a : 0 < a) 
  (h_leq_ab : a ≤ b) 
  (h_not_div_by_4 : ¬ (4 ∣ (m * n))) :
  Even (number_of_ways_partition m n a b) := 
sorry

end even_partition_dominoes_l685_685978


namespace color_identifiable_set_size_l685_685502

def g (n t : ℕ) : ℕ := ⌈n / t⌉

theorem color_identifiable_set_size (n t : ℕ) :
  ∃ S : set (set ℕ), 
    (∀ team ∈ S, ∃ colors ⊆ {1, ..., n}, colors.card ≤ t) ∧ 
    (∀ S' ⊆ S, ∀ (s1 s2 ∈ S'), s1 ≠ s2 → disjoint s1 s2) ∧
    S.card = g(n, t) :=
sorry

end color_identifiable_set_size_l685_685502


namespace count_two_digit_integers_l685_685115

theorem count_two_digit_integers (n : ℕ) :
  (∃ n : ℕ, 10 ≤ 7 * n + 5 ∧ 7 * n + 5 < 100) ↔ 13 := 
sorry

end count_two_digit_integers_l685_685115


namespace max_value_range_of_t_l685_685397

theorem max_value_range_of_t (t x : ℝ) (h : t ≤ x ∧ x ≤ t + 2) 
: ∃ y : ℝ, y = -x^2 + 6 * x - 7 ∧ y = -(t - 3)^2 + 2 ↔ t ≥ 3 := 
by {
    sorry
}

end max_value_range_of_t_l685_685397


namespace minimize_x_2y_l685_685947

noncomputable def minimum_value_x_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 / (x + 2) + 3 / (y + 2) = 1) : ℝ :=
  x + 2 * y

theorem minimize_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / (x + 2) + 3 / (y + 2) = 1) :
  minimum_value_x_2y x y hx hy h = 3 + 6 * Real.sqrt 2 :=
sorry

end minimize_x_2y_l685_685947


namespace number_of_friends_l685_685532

-- Define the conditions
def kendra_packs : ℕ := 7
def tony_packs : ℕ := 5
def pens_per_kendra_pack : ℕ := 4
def pens_per_tony_pack : ℕ := 6
def pens_kendra_keep : ℕ := 3
def pens_tony_keep : ℕ := 3

-- Define the theorem to be proved
theorem number_of_friends 
  (packs_k : ℕ := kendra_packs)
  (packs_t : ℕ := tony_packs)
  (pens_per_pack_k : ℕ := pens_per_kendra_pack)
  (pens_per_pack_t : ℕ := pens_per_tony_pack)
  (kept_k : ℕ := pens_kendra_keep)
  (kept_t : ℕ := pens_tony_keep) :
  packs_k * pens_per_pack_k + packs_t * pens_per_pack_t - (kept_k + kept_t) = 52 :=
by
  sorry

end number_of_friends_l685_685532


namespace zongzi_unit_prices_max_type_A_zongzi_l685_685789

theorem zongzi_unit_prices (x : ℝ) : 
  (800 / x - 1200 / (2 * x) = 50) → 
  (x = 4 ∧ 2 * x = 8) :=
by
  intro h
  sorry

theorem max_type_A_zongzi (m : ℕ) : 
  (m ≤ 200) → 
  (8 * m + 4 * (200 - m) ≤ 1150) → 
  (m ≤ 87) :=
by
  intros h1 h2
  sorry

end zongzi_unit_prices_max_type_A_zongzi_l685_685789


namespace intercepts_sum_l685_685717

def line_eq (x y : ℝ) : Prop := y - 3 = -3 * (x - 6)

def x_intercept : ℝ :=
  let y := 0 in
  let x := (y - 3 + 18) / -3 in
  x

def y_intercept : ℝ :=
  let x := 0 in
  let y := -3 * (x - 6) + 3 in
  y

def sum_of_intercepts : ℝ :=
  x_intercept + y_intercept

theorem intercepts_sum :
  sum_of_intercepts = 28 :=
sorry

end intercepts_sum_l685_685717


namespace next_term_geometric_sequence_l685_685668

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end next_term_geometric_sequence_l685_685668


namespace max_value_of_M_l685_685395

def J_k (k : Nat) : Nat :=
  8 * 10^(k + 1) + 32

def M (k : Nat) : Nat :=
  Nat.findGreatestPowerDivisor J_k k 2

theorem max_value_of_M :
  ∀ k > 0, M k = 5 :=
by
  sorry

end max_value_of_M_l685_685395


namespace complete_the_square_l685_685650

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l685_685650


namespace shortest_altitude_right_triangle_l685_685240

theorem shortest_altitude_right_triangle (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2) :
  let area := 0.5 * a * b in
  let altitude := 2 * area / c in
  altitude = 7.2 :=
by
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  rw [h1, h2, h3] at *,
  let area := 0.5 * 9 * 12,
  let altitude := 2 * area / 15,
  have : area = 54, by norm_num,
  rw this at *,
  have : altitude = 7.2, by norm_num,
  exact this

end shortest_altitude_right_triangle_l685_685240


namespace next_term_geometric_sequence_l685_685667

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end next_term_geometric_sequence_l685_685667


namespace sandy_savings_last_year_l685_685533

theorem sandy_savings_last_year (S : ℝ) (P : ℝ) 
(h1 : P / 100 * S = x)
(h2 : 1.10 * S = y)
(h3 : 0.10 * y = 0.11 * S)
(h4 : 0.11 * S = 1.8333333333333331 * x) :
P = 6 := by
  -- proof goes here
  sorry

end sandy_savings_last_year_l685_685533


namespace find_g_pi_over_4_l685_685061

def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * (sin x) ^ 2

def g (x : ℝ) : ℝ := f (x - π / 12) + sqrt 3 / 2

theorem find_g_pi_over_4 : g (π / 4) = sqrt 3 / 2 := by
  sorry

end find_g_pi_over_4_l685_685061


namespace population_difference_is_16_l685_685909

def total_birds : ℕ := 250

def pigeons_percent : ℕ := 30
def sparrows_percent : ℕ := 25
def crows_percent : ℕ := 20
def swans_percent : ℕ := 15
def parrots_percent : ℕ := 10

def black_pigeons_percent : ℕ := 60
def white_pigeons_percent : ℕ := 40
def black_male_pigeons_percent : ℕ := 20
def white_female_pigeons_percent : ℕ := 50

def female_sparrows_percent : ℕ := 60
def male_sparrows_percent : ℕ := 40

def female_crows_percent : ℕ := 30
def male_crows_percent : ℕ := 70

def male_parrots_percent : ℕ := 65
def female_parrots_percent : ℕ := 35

noncomputable
def black_male_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (black_pigeons_percent * (black_male_pigeons_percent / 100)) / 100
noncomputable
def white_female_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (white_pigeons_percent * (white_female_pigeons_percent / 100)) / 100
noncomputable
def male_sparrows : ℕ := (sparrows_percent * total_birds / 100) * (male_sparrows_percent / 100)
noncomputable
def female_crows : ℕ := (crows_percent * total_birds / 100) * (female_crows_percent / 100)
noncomputable
def male_parrots : ℕ := (parrots_percent * total_birds / 100) * (male_parrots_percent / 100)

noncomputable
def max_population : ℕ := max (max (max (max black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots
noncomputable
def min_population : ℕ := min (min (min (min black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots

noncomputable
def population_difference : ℕ := max_population - min_population

theorem population_difference_is_16 : population_difference = 16 :=
sorry

end population_difference_is_16_l685_685909


namespace trajectory_midpoint_area_l685_685908

-- Definitions for the given conditions
def Cube : Type := ℝ × ℝ × ℝ
def P_in_base (P : Cube) : Prop := P.2 = 0  -- P is in the base ABCD (z = 0)
def Q_on_edge (Q : Cube) : Prop := Q = (0, 0, Q.3) ∧ Q.3 ∈ Set.Icc (0, 2)  -- Q is on the edge AA₁
def PQ_distance (P Q : Cube) : Prop := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2 = 4  -- |PQ| = 2

-- Function to get the midpoint of PQ
def midpoint (P Q : Cube) : Cube := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

-- Trajectory of M
def midpoint_trajectory (A : Cube) (r : ℝ) : Set Cube := 
  {M | ∃ P Q, P_in_base P ∧ Q_on_edge Q ∧ PQ_distance P Q ∧ midpoint P Q = M}

-- The figure formed by the trajectory of the midpoint M is one-eighth of a spherical surface with radius 1 centered at A.
noncomputable def trajectory_area : ℝ := (1/8) * 4 * Real.pi * (1^2)

-- Theorem to prove the given statement
theorem trajectory_midpoint_area (A : Cube) (r : ℝ) (P Q : Cube) 
  (hP : P_in_base P) 
  (hQ : Q_on_edge Q) 
  (hPQ : PQ_distance P Q) 
  (hA : A = (0, 0, 0)) 
  (hr : r = 1): 
  trajectory_area = Real.pi / 2 :=
by 
  sorry

end trajectory_midpoint_area_l685_685908


namespace speaker_ages_l685_685210

theorem speaker_ages (x y : ℕ) : 
  (x > 1 ∧ y > 1 ∧ y > x ∧ 1961 % x = 0 ∧ 1961 % y = 0) -> {x, y} = {37, 53} :=
by
  sorry

end speaker_ages_l685_685210


namespace factor_x_squared_minus_sixtyfour_l685_685378

theorem factor_x_squared_minus_sixtyfour (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_sixtyfour_l685_685378


namespace even_intersections_same_side_odd_intersections_opposite_side_no_intersections_same_side_tangencies_parity_tangencies_parity2_l685_685633

variables {A B : ℝ × ℝ} {axis : ℝ → Prop} (h1 : ¬ axis A) (h2 : ¬ axis B)
variable (curve_intersects : ℕ) 
variable (tangencies : ℕ)

-- We will state the equivalences separately

theorem even_intersections_same_side (h_even : even curve_intersects) :
  (axis A ↔ axis B) :=
sorry

theorem odd_intersections_opposite_side (h_odd : odd curve_intersects) :
  (axis A ↔ ¬axis B) :=
sorry

theorem no_intersections_same_side (h_zero : curve_intersects = 0) :
  (axis A ↔ axis B) :=
sorry

theorem tangencies_parity (h_tangents_even : even tangencies) :
  even curve_intersects → even (curve_intersects + tangencies) :=
sorry

theorem tangencies_parity2 (h_tangents_odd : odd tangencies) :
  odd curve_intersects → odd (curve_intersects + tangencies) :=
sorry

end even_intersections_same_side_odd_intersections_opposite_side_no_intersections_same_side_tangencies_parity_tangencies_parity2_l685_685633


namespace poly_coeff_difference_l685_685888

theorem poly_coeff_difference :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (2 + x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
  a = 16 →
  1 = a - a_1 + a_2 - a_3 + a_4 →
  a_2 - a_1 + a_4 - a_3 = -15 :=
by
  intros a a_1 a_2 a_3 a_4 h_poly h_a h_eq
  sorry

end poly_coeff_difference_l685_685888


namespace total_surface_area_correct_l685_685579

def six_cubes_surface_area : ℕ :=
  let cube_edge := 1
  let cubes := 6
  let initial_surface_area := 6 * cubes -- six faces per cube, total initial surface area
  let hidden_faces := 10 -- determined by counting connections
  initial_surface_area - hidden_faces

theorem total_surface_area_correct : six_cubes_surface_area = 26 := by
  sorry

end total_surface_area_correct_l685_685579


namespace entries_multiples_of_73_l685_685318

def triangular_array (a : ℕ → ℕ → ℕ) : Prop := ∀ n k, a n k = 2^(n-1) * (n + 2 * k - 2)

def odd_integers_first_row (a : ℕ → ℕ → ℕ) : Prop := ∀ k, a 1 k = 2 * k - 1 ∧ k ≤ 51

theorem entries_multiples_of_73 :
  ∃ a : ℕ → ℕ → ℕ, triangular_array a ∧ odd_integers_first_row a ∧ 
  (finset.card (finset.filter (λ nk, 73 ∣ (a nk.1 nk.2)) (finset.univ : finset (ℕ × ℕ))) = 15) :=
begin
  -- Proof goes here
  sorry
end

end entries_multiples_of_73_l685_685318


namespace sum_of_digits_M_is_21_l685_685611

-- Define the integer M
def M : ℕ := 2^86 * 3^50 * 5^36

-- The main theorem stated as the problem requires
theorem sum_of_digits_M_is_21 : (∑ digit in M.digits, digit) = 21 := 
sorry

end sum_of_digits_M_is_21_l685_685611


namespace sum_of_prism_features_l685_685161

theorem sum_of_prism_features : (12 + 8 + 6 = 26) := by
  sorry

end sum_of_prism_features_l685_685161


namespace largest_prime_difference_l685_685242

theorem largest_prime_difference (h : ∀ n, n > 7 → (∃ p q, p + q = n ∧ p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q)) :
  ∃ p q, p + q = 134 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ ∀ r s, r + s = 134 ∧ Nat.Prime r ∧ Nat.Prime s ∧ r ≠ s → |p - q| ≥ |r - s| :=
sorry

end largest_prime_difference_l685_685242


namespace ticket_price_values_l685_685317

theorem ticket_price_values : 
  ∃ (x_values : Finset ℕ), 
    (∀ x ∈ x_values, x ∣ 60 ∧ x ∣ 80) ∧ 
    x_values.card = 6 :=
by
  sorry

end ticket_price_values_l685_685317


namespace sum_of_squares_series_l685_685788

theorem sum_of_squares_series :
  (∑ k in Finset.range 13, (2*k + 1)^2 - (2*k - 1)^2) = 1196 :=
by
  sorry

end sum_of_squares_series_l685_685788


namespace charge_difference_is_51_l685_685814

-- Define the charges and calculations for print shop X
def print_shop_x_cost (n : ℕ) : ℝ :=
  if n ≤ 50 then n * 1.20 else 50 * 1.20 + (n - 50) * 0.90

-- Define the charges and calculations for print shop Y
def print_shop_y_cost (n : ℕ) : ℝ :=
  10 + n * 1.70

-- Define the difference in charges for 70 copies
def charge_difference : ℝ :=
  print_shop_y_cost 70 - print_shop_x_cost 70

-- The proof statement
theorem charge_difference_is_51 : charge_difference = 51 :=
by
  sorry

end charge_difference_is_51_l685_685814


namespace solution_set_condition_l685_685852

open Set

variable {f : ℝ → ℝ}

-- Conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def inequality_condition (f : ℝ → ℝ) := ∀ {x1 x2 : ℝ}, x1 ≠ x2 → x1 * f(x1) + x2 * f(x2) > x1 * f(x2) + x2 * f(x1)

-- Problem Statement
theorem solution_set_condition
  (h1 : odd_function (λ x, f x))
  (h2 : inequality_condition (λ x, f x)) :
  ∀ x ∈ Ioi (1 : ℝ), f (2 - x) < 0 :=
begin
  -- proof will go here
  sorry
end

end solution_set_condition_l685_685852


namespace two_digit_integers_remainder_5_when_divided_by_7_count_l685_685113

theorem two_digit_integers_remainder_5_when_divided_by_7_count :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 7 = 5}.to_finset.card = 13 := by
sorry

end two_digit_integers_remainder_5_when_divided_by_7_count_l685_685113


namespace polar_equation_C1_area_triangle_AOB_l685_685512

-- Define the parametric equation for C1.
def parametric_curve_C1 (alpha : ℝ) : ℝ × ℝ := 
  (2 + 2 * Real.cos alpha,
   2 * Real.sin alpha)

-- Proof Problem (1): Prove the polar equation of C1.
theorem polar_equation_C1 :
  ∀ (alpha θ : ℝ), 
    let (x, y) := parametric_curve_C1 alpha in 
    x^2 - 4*x + y^2 = 0 → θ = Real.arctan (y / x) → 
    ∃ ρ, ρ = 4 * Real.cos θ := 
  sorry

-- Polar equation for C2.
def polar_equation_C2 (θ : ℝ) : ℝ := 
  √3 / Real.sin θ

-- Proof Problem (2): Prove the area of triangle AOB.
theorem area_triangle_AOB :
  ∀ (α θ : ℝ), 
    let (x, y) := parametric_curve_C1 α in 
    (θ = Real.arctan (y / x) ∧ 4 * Real.cos θ = √3 / Real.sin θ) → 
    let ρA := 2 * Real.sqrt 3
    let ρB := 2
    let angle_AOB := Real.pi / 2
    ∃ S, S = ρA * ρB * Real.sin angle_AOB / 2 :=
  sorry

end polar_equation_C1_area_triangle_AOB_l685_685512


namespace fraction_product_l685_685261

theorem fraction_product :
  (5 / 8) * (7 / 9) * (11 / 13) * (3 / 5) * (17 / 19) * (8 / 15) = 14280 / 1107000 :=
by sorry

end fraction_product_l685_685261


namespace exists_circle_with_n_lattice_points_l685_685830

noncomputable def lattice_points_in_circle (r : ℝ) (a b : ℤ) : ℕ :=
  finset.card $ finset.filter (λ p : ℤ × ℤ, (p.1 - a)^2 + (p.2 - b)^2 < r^2)
    (finset.product (finset.range (2 * r.to_nat + 1)) (finset.range (2 * r.to_nat + 1)))

theorem exists_circle_with_n_lattice_points (n : ℕ) :
  ∃ (r : ℝ) (a b : ℤ), lattice_points_in_circle r a b = n :=
sorry

end exists_circle_with_n_lattice_points_l685_685830


namespace solve_system_of_equations_l685_685973

theorem solve_system_of_equations (x y : ℤ) (h1 : x + y = 8) (h2 : x - 3 * y = 4) : x = 7 ∧ y = 1 :=
by {
    -- Proof would go here
    sorry
}

end solve_system_of_equations_l685_685973


namespace angle_BCD_is_30_degrees_l685_685225

theorem angle_BCD_is_30_degrees
  (A B C D H M : Type)
  [Point A] [Point B] [Point C] [Point D] [Point H] [Point M]
  (triangle_ABC : Triangle A B C)
  (foot_perpendicular_from_A : IsFootPerpendicular A H B C)
  (H_eq_median_BM : distance A H = distance B M)
  (D_on_extension_AB : lies_on_extension D B A)
  (BD_eq_AB : distance B D = distance A B)
  (BM_is_median : is_median B M A C) :
  angle B C D = 30 :=
sorry

end angle_BCD_is_30_degrees_l685_685225


namespace length_of_AB_l685_685176

-- Definition of the points A and B on the parabola
variables {x1 y1 x2 y2 : ℝ}

def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 2

-- Conditions
def on_parabola_A : y1 = parabola x1 := sorry
def on_parabola_B : y2 = parabola x2 := sorry
def midpoint_is_origin : (0 : ℝ) = (x1 + x2) / 2 ∧ (0 : ℝ) = (y1 + y2) / 2 := sorry

-- Theorem statement
theorem length_of_AB : sqrt ((x1 - x2)^2 + (y1 - y2)^2) = sqrt 34 :=
  sorry

end length_of_AB_l685_685176


namespace solve_for_a_l685_685469

theorem solve_for_a (a : ℝ) (h : 2 * a + (1 - 4 * a) = 0) : a = 1 / 2 :=
sorry

end solve_for_a_l685_685469


namespace only_powers_of_2_satisfy_condition_l685_685785

theorem only_powers_of_2_satisfy_condition:
  ∀ (n : ℕ), n ≥ 2 →
  (∃ (x : ℕ → ℕ), 
    ∀ (i j : ℕ), 
      0 < i ∧ i < n → 0 < j ∧ j < n → i ≠ j ∧ (n ∣ (2 * i + j)) → x i < x j) ↔
      ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
by
  sorry

end only_powers_of_2_satisfy_condition_l685_685785


namespace fourth_root_sum_of_square_roots_eq_l685_685958

theorem fourth_root_sum_of_square_roots_eq :
  (1 + Real.sqrt 2 + Real.sqrt 3) = 
    Real.sqrt (Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608) ^ 4 :=
by
  sorry

end fourth_root_sum_of_square_roots_eq_l685_685958


namespace calculate_m_l685_685891

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l685_685891


namespace volume_ratio_eq_25_over_192_l685_685828

noncomputable def volume_ratio_separated_by_plane (a : ℝ) : ℝ :=
  let coords A := (0, 0, 0)
  let coords A1 := (0, 0, a)
  let coords X := (0, 0, a / 8)
  let coords C := (a, a, 0)
  let coords C1 := (a, a, a)
  let coords Y := (a, a, a / 8)
  let coords M_AD := (0, a / 2, 0)
  -- Calculation on closed follow with correct placements leading to ratio
  let ratio := (25 / 192)
  ratio

theorem volume_ratio_eq_25_over_192 (a : ℝ) : volume_ratio_separated_by_plane a = 25 / 192 :=
sorry

end volume_ratio_eq_25_over_192_l685_685828


namespace max_perimeter_isosceles_triangle_l685_685399

theorem max_perimeter_isosceles_triangle
    (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (triangle : Triangle A B C) 
    (angle_C : Angle A C B) 
    (CA_length1 : dist C A = 1) 
    (CB_length1 : dist C B = 1) :
    ∃ triangle', (is_isosceles_triangle triangle' ∧ triangle'.perimeter = 2) :=
sorry

end max_perimeter_isosceles_triangle_l685_685399


namespace find_a_l685_685422

noncomputable def f (x : Real) (a : Real) : Real :=
if h : 0 < x ∧ x < 2 then (Real.log x - a * x) 
else 
if h' : -2 < x ∧ x < 0 then sorry
else 
   sorry

theorem find_a (a : Real) : (∀ x : Real, f x a = - f (-x) a) → (∀ x: Real, (0 < x ∧ x < 2) → f x a = Real.log x - a * x) → a > (1 / 2) → (∀ x: Real, (-2 < x ∧ x < 0) → f x a ≥ 1) → a = 1 := 
sorry

end find_a_l685_685422


namespace dot_product_subtract_eq_eight_l685_685877

theorem dot_product_subtract_eq_eight (a b : ℝ × ℝ) (ha : a = (-1, 2)) (hb : b = (1, -1)) : ((a.1 - b.1, a.2 - b.2) : ℝ × ℝ) ⋅ a = 8 :=
by
  simp [ha, hb]
  sorry

end dot_product_subtract_eq_eight_l685_685877


namespace abs_diff_gt_two_l685_685944

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 1010).sum (λ k, 1 / (x - 2 * k))

noncomputable def g (x : ℝ) : ℝ :=
  (Finset.range 1010).sum (λ k, 1 / (x - (2 * k + 1)))

theorem abs_diff_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < 2018) (hx3 : ¬∃ n : ℤ, x = n) :
  |f x - g x| > 2 :=
by
  sorry

end abs_diff_gt_two_l685_685944


namespace complete_square_l685_685653

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l685_685653


namespace equal_number_of_boys_and_girls_l685_685500

theorem equal_number_of_boys_and_girls
  (m d M D : ℝ)
  (hm : m ≠ 0)
  (hd : d ≠ 0)
  (avg1 : M / m ≠ D / d)
  (avg2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d :=
by
  sorry

end equal_number_of_boys_and_girls_l685_685500


namespace distribute_students_l685_685358

theorem distribute_students:
  (∃ f : Fin 7 → Fin 2, (∀ i, f i = 0 ∨ f i = 1) ∧ 
  (∑ i, if f i = 0 then 1 else 0) ≥ 2 ∧ 
  (∑ i, if f i = 1 then 1 else 0) ≥ 2) → 
  (∃ d_choices: ℕ, d_choices = (Nat.choose 7 2 + Nat.choose 7 3 + Nat.choose 7 4 + Nat.choose 7 5) ∧ d_choices = 112) := 
by
  sorry

end distribute_students_l685_685358


namespace average_of_numbers_between_6_and_36_divisible_by_7_l685_685681

noncomputable def average_of_divisibles_by_seven : ℕ :=
  let numbers := [7, 14, 21, 28, 35]
  let sum := numbers.sum
  let count := numbers.length
  sum / count

theorem average_of_numbers_between_6_and_36_divisible_by_7 : average_of_divisibles_by_seven = 21 :=
by
  sorry

end average_of_numbers_between_6_and_36_divisible_by_7_l685_685681


namespace minnie_slower_than_penny_by_65_minutes_l685_685955

def minnie_speed_flat : ℝ := 20 -- kph
def minnie_speed_downhill : ℝ := 30 -- kph
def minnie_speed_uphill : ℝ := 5 -- kph

def penny_speed_flat : ℝ := 30 -- kph
def penny_speed_downhill : ℝ := 40 -- kph
def penny_speed_uphill : ℝ := 10 -- kph

def distance_AB_uphill : ℝ := 10 -- km
def distance_BC_downhill : ℝ := 15 -- km
def distance_CA_flat : ℝ := 20 -- km

noncomputable def minnie_time := 
  (distance_AB_uphill / minnie_speed_uphill) + 
  (distance_BC_downhill / minnie_speed_downhill) + 
  (distance_CA_flat / minnie_speed_flat)

noncomputable def penny_time := 
  (distance_CA_flat / penny_speed_flat) + 
  (distance_BC_downhill / penny_speed_uphill) + 
  (distance_AB_uphill / penny_speed_downhill)

noncomputable def minnie_time_minutes := minnie_time * 60 -- to minutes
noncomputable def penny_time_minutes := penny_time * 60 -- to minutes

def time_difference := minnie_time_minutes - penny_time_minutes

theorem minnie_slower_than_penny_by_65_minutes : time_difference = 65 := by
  sorry

end minnie_slower_than_penny_by_65_minutes_l685_685955


namespace function_values_order_l685_685428

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

theorem function_values_order (f : ℝ → ℝ) 
  (even_f : is_even_function f)
  (mono_f : is_monotonically_decreasing_on_interval (λ x, f (x - 2)) 0 2) :
  f 0 < f (-1) < f 2 :=
sorry

end function_values_order_l685_685428


namespace g_at_negative_two_l685_685940

-- Function definition
def g (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 2*x^3 - 5*x^2 - x + 8

-- Theorem statement
theorem g_at_negative_two : g (-2) = -186 :=
by
  -- Proof will go here, but it is skipped with sorry
  sorry

end g_at_negative_two_l685_685940


namespace sequence_sum_relation_Sn_expression_an_expression_l685_685453

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 2 else 3 * n^2 - 3 * n + 2

noncomputable def S (n : ℕ) : ℕ :=
∑ i in Finset.range n, a (i + 1)

theorem sequence_sum_relation (n : ℕ) :
  (n^2 + 1) * a (n + 1) = (2 * n + 1) * S n + n^4 + 2 * n^3 + 3 * n^2 + 2 * n + 2 :=
by {
  sorry
}

theorem Sn_expression (n : ℕ) : S n = n * (n^2 + 1) :=
by {
  sorry
}

theorem an_expression (n : ℕ) : a n = 3 * n^2 - 3 * n + 2 :=
by {
  sorry
}

end sequence_sum_relation_Sn_expression_an_expression_l685_685453


namespace distinct_values_of_T_l685_685174

noncomputable def j : ℂ := Complex.exp (Complex.I * Real.pi / 3)

def T (n : ℤ) : ℂ := j^n + j^(-n)

theorem distinct_values_of_T : 
  {t : ℂ | ∃ n : ℤ, T n = t}.finite ∧ {t : ℂ | ∃ n : ℤ, T n = t}.card = 4 := 
sorry

end distinct_values_of_T_l685_685174


namespace mike_scored_212_l685_685554

variable {M : ℕ}

def passing_marks (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

def mike_marks (passing_marks shortfall : ℕ) : ℕ := passing_marks - shortfall

theorem mike_scored_212 (max_marks : ℕ) (shortfall : ℕ)
  (h1 : max_marks = 790)
  (h2 : shortfall = 25)
  (h3 : M = mike_marks (passing_marks max_marks) shortfall) : 
  M = 212 := 
by 
  sorry

end mike_scored_212_l685_685554


namespace maria_initial_carrots_l685_685552

theorem maria_initial_carrots (C : ℕ) (h : C - 11 + 15 = 52) : C = 48 :=
by
  sorry

end maria_initial_carrots_l685_685552


namespace minimum_distinct_numbers_l685_685251

theorem minimum_distinct_numbers {x : ℝ} (hx : ∀ i j, i ≠ j → x i ≠ x j) (h_not_int : ∀ i, ∃ n : ℤ, x i = n + fract(x)) :
  ∃ distinct_count : ℕ, distinct_count = 4 :=
by
  sorry

end minimum_distinct_numbers_l685_685251


namespace area_QRS_eq_2a_l685_685932

-- Let PQRS be a pyramid with a square base QRS
-- Assume QRS has side length, we name it s, and the height of the pyramid is also s
variable (s : ℝ)

-- The areas of triangles PQR, PQS, and PRS are denoted by a, b, and c respectively
variables (a b c : ℝ)

-- Define the areas a, b, c in terms of s
def area_PQR : ℝ := 1 / 2 * s ^ 2
def area_PQS : ℝ := 1 / 2 * s ^ 2
def area_PRS : ℝ := 1 / 2 * s ^ 2

-- Given these areas
axiom ha : area_PQR s = a
axiom hb : area_PQS s = b
axiom hc : area_PRS s = c

-- We need to prove that the area of the square QRS is 2a
theorem area_QRS_eq_2a : 2 * a = area_PQR s + area_PQS s + area_PRS s := by
  sorry

end area_QRS_eq_2a_l685_685932


namespace game_winning_strategy_l685_685256

def player_has_winning_strategy (n : ℕ) (p : ℕ → Prop) : Prop :=
∃ k c : ℕ, k ≥ 0 ∧ c ≥ 0 ∧ c < 2^k ∧ (p n ↔ n = 2^(2*k + 1) + 2*c)

theorem game_winning_strategy (n : ℕ) (h : n > 1) :
  player_has_winning_strategy n (λ n, player.has_strategy 2) ∨
  player_has_winning_strategy n (λ n, player.has_strategy 1) := by
  sorry

end game_winning_strategy_l685_685256


namespace completing_the_square_solution_correct_l685_685639

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l685_685639


namespace convert_to_scientific_notation_l685_685190

theorem convert_to_scientific_notation (N : ℕ) (h : 2184300000 = 2184.3 * 10^6) : 
    (2184300000 : ℝ) = 2.1843 * 10^7 :=
by 
  sorry

end convert_to_scientific_notation_l685_685190


namespace min_steps_to_top_and_back_l685_685742
open Nat

theorem min_steps_to_top_and_back (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ n : ℕ, (∀ k : ℕ, k ≥ n → (∃ x y : ℕ, k = x * a - y * b)) ∧ n = a + b - gcd a b :=
begin
  sorry
end

end min_steps_to_top_and_back_l685_685742


namespace triangle_inequality_side_len_l685_685411

theorem triangle_inequality_side_len (x : ℝ) : x = 8 → ¬ (2 < x ∧ x < 8) :=
by
  intro h
  rw h
  exact not_and_of_not_right _ (lt_irrefl 8)

#eval triangle_inequality_side_len 8 rfl

end triangle_inequality_side_len_l685_685411


namespace cos_arcsin_proof_l685_685004

noncomputable def cos_arcsin : ℝ :=
  let θ := arcsin (3 / 5) in
  cos θ

theorem cos_arcsin_proof : cos_arcsin = 4 / 5 := 
by
  sorry

end cos_arcsin_proof_l685_685004


namespace smallest_M_le_max_l685_685416

noncomputable def M (n : ℕ) (a : ℝ) : ℝ :=
  if h : a < 1 then 1 / a else n / (a + n - 1)

theorem smallest_M_le_max (n : ℕ) (a : ℝ) (hn : n ≥ 2) (ha : 0 < a)
  (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) (hx_prod : (∏ i, x i) = 1) :
  (∑ i, 1 / (a + (∑ j, x j) - x i)) ≤ M n a :=
sorry

end smallest_M_le_max_l685_685416


namespace tangent_line_difference_l685_685490

noncomputable def curve (x : ℝ) (b : ℝ) : ℝ := x * Real.log x + b
noncomputable def line (x : ℝ) (a : ℝ) : ℝ := x + a

theorem tangent_line_difference (a b : ℝ) (h_tangent : ∃ (x₀ : ℝ), 
  line x₀ a = curve x₀ b ∧ 
  ∀ x₀, Deriv (curve x₀ b) = 1) : b - a = 1 :=
sorry

end tangent_line_difference_l685_685490


namespace inclination_angle_of_line_proof_l685_685227

noncomputable def inclination_angle_of_line (a b c : ℝ) : ℝ :=
if b = 0 then 0 else real.atan2 b (-a)

theorem inclination_angle_of_line_proof : 
  inclination_angle_of_line 1 1 (-√3) = 3 * real.pi / 4 :=
by sorry

end inclination_angle_of_line_proof_l685_685227


namespace binom_20_17_l685_685769

theorem binom_20_17 : nat.choose 20 17 = 1140 := by
  sorry

end binom_20_17_l685_685769


namespace find_pairs_l685_685035

theorem find_pairs (m n : ℕ) :
  (m + 1) % n = 0 ∧ (n^2 - n + 1) % m = 0 ↔
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2) := 
by
  sorry

end find_pairs_l685_685035


namespace john_ahead_of_steve_l685_685166

def distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem john_ahead_of_steve :
  ∀ (initial_distance : ℝ) (john_speed : ℝ) (steve_speed : ℝ) (time : ℝ),
    initial_distance = 16 →
    john_speed = 4.2 →
    steve_speed = 3.7 →
    time = 36 →
    (distance_covered john_speed time - initial_distance) - (distance_covered steve_speed time) = 2 :=
by
  intros initial_distance john_speed steve_speed time h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  linarith [mul_pos (by norm_num) (by norm_num : 4.2 * 36 > 0),
            mul_pos (by norm_num) (by norm_num : 3.7 * 36 > 0)]
  simp only [mul_sub, sub_add, add_sub_assoc, mul_assoc]
  norm_num
  linarith
  done

end john_ahead_of_steve_l685_685166


namespace train_cross_time_approx_l685_685110

noncomputable def length_of_train : ℝ := 100
noncomputable def speed_of_train_km_hr : ℝ := 80
noncomputable def length_of_bridge : ℝ := 142
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_cross_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 :=
by
  sorry

end train_cross_time_approx_l685_685110


namespace solve_abs_eq_l685_685122

theorem solve_abs_eq (x : ℝ) (h : |x - 1| = 2 * x) : x = 1 / 3 :=
by
  sorry

end solve_abs_eq_l685_685122


namespace area_of_regular_hexagon_l685_685178

noncomputable def area_of_hexagon (ABCDEF : Hexagon) (J K L : Points) 
  (hJKL : is_midpoint_of(J, AB) ∧ is_midpoint_of(K, CD) ∧ is_midpoint_of(L, EF)) 
  (area_JKL : area(⟨J, K, L⟩) = 144) : Prop :=
  area ABCDEF = 384

theorem area_of_regular_hexagon 
  {ABCDEF : Hexagon} {J K L : Point} 
  (hJKL : is_midpoint_of(J, AB) ∧ is_midpoint_of(K, CD) ∧ is_midpoint_of(L, EF)) 
  (area_JKL : area(⟨J, K, L⟩) = 144) :
  area(ABCDEF) = 384 :=
sorry

end area_of_regular_hexagon_l685_685178


namespace unique_solution_largest_a_l685_685039

theorem unique_solution_largest_a (a x : ℝ) (h_eq : (| (a * x^2 - a * x - 12 * a + x^2 + x + 12) / (a * x + 3 * a - x - 3) - a |) * |4 * a - 3 * x - 19| = 0) :
    ∃ (a_max : ℝ), a ≤ a_max ∧ ( ∀ (a' : ℝ), a' > a_max → ¬((|(a' * x^2 - a' * x - 12 * a' + x^2 + x + 12) / (a' * x + 3 * a' - x - 3) - a' |) * |4 * a' - 3 * x - 19| = 0) ) ∧ a_max = 7 :=
sorry

end unique_solution_largest_a_l685_685039


namespace expand_expression_l685_685369

variable (x y z : ℝ)

theorem expand_expression :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := 
  sorry

end expand_expression_l685_685369


namespace find_x_coordinate_of_Q_l685_685513

noncomputable def P : ℝ × ℝ := (3/5, 4/5)
def O : ℝ × ℝ := (0, 0)

def Q_in_third_quadrant (Q : ℝ × ℝ) : Prop := Q.1 < 0 ∧ Q.2 < 0
def angle_POQ : ℝ := 3 * π / 4
def length_OQ (Q : ℝ × ℝ) : ℝ := real.sqrt(Q.1^2 + Q.2^2)

theorem find_x_coordinate_of_Q (Q : ℝ × ℝ) 
  (hq : Q_in_third_quadrant Q) 
  (hq_length : length_OQ Q = 1)
  (hangle : angle_POQ = 3 * π / 4)
  (hP1 : P.1 = 3 / 5) 
  (hP2 : P.2 = 4 / 5) : 
  Q.1 = -7 * real.sqrt 2 / 10 := 
sorry

end find_x_coordinate_of_Q_l685_685513


namespace quadrilateral_parallelogram_l685_685293

theorem quadrilateral_parallelogram (A B C D E : Type) [Quad : Quadrilateral A B C D]
  (h1 : divides_diag_eq_area A C B D E)
  (h2 : divides_diag_eq_area B D A C E) :
  parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l685_685293


namespace problem_1_problem_2_problem_3_l685_685448

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

def F (a b x : ℝ) : ℝ :=
if x > 0 then f a b x else -f a b x

theorem problem_1 (a b : ℝ) (hx : f a b (-1) = 0) (hrange : ∀ x, f a b x ≥ 0) :
  ∀ x, F a b x = if x > 0 then (x + 1)^2 else -(x + 1)^2 :=
sorry

theorem problem_2 (a b k : ℝ) (hx : f a b (-1) = 0) (hrange : ∀ x, f a b x ≥ 0) 
  (hmono : ∀ x, -2 ≤ x ∧ x ≤ 2 → (∃ c, ∀ y, c > 0 ∧ y = f a b c - k * c)) :
  k ≥ 6 ∨ k ≤ -2 :=
sorry

theorem problem_3 (a m n : ℝ) (hmn : m * n < 0) (hmn_sum : m + n > 0) (ha : a > 0) 
  (heven : ∀ x, f a 0 x = f a 0 (-x)) :
  F a 0 m + F a 0 n > 0 :=
sorry

end problem_1_problem_2_problem_3_l685_685448


namespace completing_the_square_l685_685643

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l685_685643


namespace max_runs_35_overs_l685_685143

theorem max_runs_35_overs (
    h_legal_deliveries_per_over: ∀ over : ℕ, over < 35 → 6,
    h_max_runs_per_delivery : ∀ delivery : ℕ, delivery < 6 → 6,
    h_max_consecutive_boundaries : 3
) : ∃ max_runs : ℕ, max_runs = 735 :=
by 
    sorry

end max_runs_35_overs_l685_685143


namespace compute_fraction_mul_l685_685341

theorem compute_fraction_mul :
  (1 / 3) ^ 2 * (1 / 8) = 1 / 72 :=
by
  sorry

end compute_fraction_mul_l685_685341


namespace graph_of_conic_section_is_straight_lines_l685_685351

variable {x y : ℝ}

theorem graph_of_conic_section_is_straight_lines:
  (x^2 - 9 * y^2 = 0) ↔ (x = 3 * y ∨ x = -3 * y) := by
  sorry

end graph_of_conic_section_is_straight_lines_l685_685351


namespace tan_condition_oblique_triangle_l685_685911

theorem tan_condition_oblique_triangle (A : ℝ) (hA : A > π / 4) (tan_A_gt_one : tan A > 1) : Prop :=
(A > π / 4) ↔ (tan A > 1) := 
sorry

end tan_condition_oblique_triangle_l685_685911


namespace solve_eqn_l685_685813

theorem solve_eqn (n : ℕ) : 
  (∀ (x y : ℝ), (x + y)^n = x^n + y^n ↔ 
  (n = 1 ∧ (x, y) ∈ set.univ) ∨
  (n % 2 = 0 ∧ (x = 0 ∨ y = 0)) ∨
  (n % 2 = 1 ∧ (x = 0 ∨ y = 0 ∨ x + y = 0))) 
:= sorry

end solve_eqn_l685_685813


namespace equal_number_of_various_square_sizes_l685_685359

theorem equal_number_of_various_square_sizes (n : ℕ) (h_pos : 0 < n) :
  ∃ (sizes : list ℕ), (∀ s ∈ sizes, s ≤ n) ∧ 
                      (∀ a b ∈ sizes, a = b → a = b) ∧ 
                      (∃ k, ∀ s ∈ sizes, sizes.count s = k) := 
sorry

end equal_number_of_various_square_sizes_l685_685359
