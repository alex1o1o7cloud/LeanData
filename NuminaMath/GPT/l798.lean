import Mathlib
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Char
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.LinearAlgebra.InnerProductSpace.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.ContinuousFunction.Basic

namespace equation_has_one_real_solution_l798_798161

theorem equation_has_one_real_solution (k : ℚ) :
    (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x ↔ x^2 + 4 * x + (10 - k) = 0) →
    (∃ k : ℚ, (∀ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ↔ by sorry (condition for one real solution is equivalent to discriminant being zero), k = 6) := by
    sorry

end equation_has_one_real_solution_l798_798161


namespace max_positive_product_from_set_l798_798362

theorem max_positive_product_from_set :
  ∃ a b c ∈ ({-4, -3, -1, 3, 5, 8} : set ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 120 :=
by
  sorry

end max_positive_product_from_set_l798_798362


namespace count_integers_satisfying_inequality_l798_798456

theorem count_integers_satisfying_inequality :
  (nat.card {n : ℤ | -15 ≤ n ∧ n ≤ 9 ∧ (n - 3) * (n + 3) * (n + 7) < 0}) = 12 := 
by
  sorry

end count_integers_satisfying_inequality_l798_798456


namespace probability_odd_and_multiple_of_5_l798_798714

/-- Given three distinct integers selected at random between 1 and 2000, inclusive, the probability that the product of the three integers is odd and a multiple of 5 is between 0.01 and 0.05. -/
theorem probability_odd_and_multiple_of_5 :
  ∃ p : ℚ, (0.01 < p ∧ p < 0.05) :=
sorry

end probability_odd_and_multiple_of_5_l798_798714


namespace small_disks_radius_l798_798044

theorem small_disks_radius (r : ℝ) (h : r > 0) :
  (2 * r ≥ 1 + r) → (r ≥ 1 / 2) := by
  intro hr
  linarith

end small_disks_radius_l798_798044


namespace noncongruent_triangles_count_l798_798549

/-- The number of noncongruent integer-sided triangles with positive area and a perimeter less than 
20 that are neither equilateral, isosceles, nor right triangles is 11. -/
theorem noncongruent_triangles_count : 
  { (a, b, c) : ℕ × ℕ × ℕ // 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 
    a + b + c < 20 ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∀ x y z, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
               ¬ (a = x ∧ b = y ∧ c = x) ∧ 
               ¬ (a = x ∧ b = x ∧ c = y)) ∧
    (∀ x y z, x ^ 2 + y ^ 2 ≠ z ^ 2) } .card = 11 :=
sorry

end noncongruent_triangles_count_l798_798549


namespace solve_a_range_m_l798_798201

def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem solve_a :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ↔ (2 = 2) :=
by {
  sorry
}

theorem range_m :
  (∀ x : ℝ, f (3 * x) 2 + f (x + 3) 2 ≥ m) ↔ (m ≤ 5 / 3) :=
by {
  sorry
}

end solve_a_range_m_l798_798201


namespace parabola_focus_area_l798_798516

theorem parabola_focus_area (p : ℝ) (h : p ≠ 0) :
  let F := (p / 2, 0),
      l := {x : ℝ × ℝ | x.fst = p / 2},
      A := (p / 2, p),
      B := (p / 2, -p),
      O := (0, 0),
      area_OAB := 1 / 2 * abs (p / 2) * 2 * abs (p)
  in area_OAB = 4 → ((y : ℝ), x : ℝ), y^2 = 2 * p * x :=
begin
  sorry
end

end parabola_focus_area_l798_798516


namespace parabola_transformation_l798_798564

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the transformed parabola after shifting 2 units to the left and 3 units up
def transformed_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

-- State the theorem to prove the transformation
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 5 * (x + 2)^2 + 3 :=
begin
  sorry
end

end parabola_transformation_l798_798564


namespace determine_f_0_f_1_l798_798409

noncomputable def f : ℤ → ℤ := sorry

theorem determine_f_0_f_1 :
  (∀ x : ℤ, f(x + 3) - f(x) = 6 * x + 12) →
  (∀ x : ℤ, f(x^2 + 1) = (f(x) + x)^2 + x^2 + 2) →
  (f 0 = -1 ∧ f 1 = 1) :=
begin
  intros h1 h2,
  sorry
end

end determine_f_0_f_1_l798_798409


namespace max_value_of_f_l798_798349

noncomputable def f (x : ℝ) : ℝ := - (1 / (x + 1))

theorem max_value_of_f : ∃ x ∈ (Set.Icc (1:ℝ) 2), ∀ y ∈ (Set.Icc (1:ℝ) 2), f(x) ≥ f(y) ∧ f(x) = -(1/3) :=
by
  sorry

end max_value_of_f_l798_798349


namespace pirates_total_coins_l798_798317

theorem pirates_total_coins :
  ∀ (x : ℕ), (∃ (paul_coins pete_coins : ℕ), 
  paul_coins = x ∧ pete_coins = 5 * x ∧ pete_coins = (x * (x + 1)) / 2) → x + 5 * x = 54 := by
  sorry

end pirates_total_coins_l798_798317


namespace slope_of_line_l_l798_798691

theorem slope_of_line_l :
  (∃ A B : ℝ × ℝ, 
    (∃ l : ℝ → ℝ, 
      (∀ x, l x = (x^2) / 2 → 
        ∀ y, 
          y = l x →
            (x, y) = A ∨ (x, y) = B) ∧
      (∃ kA kB : ℝ, 
        kA = 2 ∧ 
        kB = -1 / kA ∧ 
        (A.1 = 2 ∧ A.2 = (A.1)^2 / 2) ∧
        (B.1 = -1 / 2  ∧ B.2 = (B.1)^2 / 2)))) →
  (∃ m : ℝ, m = (A.2 - B.2) / (A.1 - B.1) ∧ m = 3 / 4) :=
sorry

end slope_of_line_l_l798_798691


namespace squirrel_nuts_l798_798170

theorem squirrel_nuts :
  ∃ (a b c d : ℕ), 103 ≤ a ∧ 103 ≤ b ∧ 103 ≤ c ∧ 103 ≤ d ∧
                   a ≥ b ∧ a ≥ c ∧ a ≥ d ∧
                   a + b + c + d = 2020 ∧
                   b + c = 1277 ∧
                   a = 640 :=
by {
  -- proof goes here
  sorry
}

end squirrel_nuts_l798_798170


namespace slope_range_l798_798128

noncomputable def directed_distance (a b c x0 y0 : ℝ) : ℝ :=
  (a * x0 + b * y0 + c) / (Real.sqrt (a^2 + b^2))

theorem slope_range {A B P : ℝ × ℝ} (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : P = (3, 0))
                   {C : ℝ × ℝ} (hC : ∃ θ : ℝ, C = (9 * Real.cos θ, 18 + 9 * Real.sin θ))
                   {a b c : ℝ} (h_line : c = -3 * a)
                   (h_sum_distances : directed_distance a b c (-1) 0 +
                                      directed_distance a b c 1 0 +
                                      directed_distance a b c (9 * Real.cos θ) (18 + 9 * Real.sin θ) = 0) :
  -3 ≤ - (a / b) ∧ - (a / b) ≤ -1 := sorry

end slope_range_l798_798128


namespace total_legs_arms_proof_l798_798438

/-
There are 4 birds, each with 2 legs.
There are 6 dogs, each with 4 legs.
There are 5 snakes, each with no legs.
There are 2 spiders, each with 8 legs.
There are 3 horses, each with 4 legs.
There are 7 rabbits, each with 4 legs.
There are 2 octopuses, each with 8 arms.
There are 8 ants, each with 6 legs.
There is 1 unique creature with 12 legs.
We need to prove that the total number of legs and arms is 164.
-/

def total_legs_arms : Nat := 
  (4 * 2) + (6 * 4) + (5 * 0) + (2 * 8) + (3 * 4) + (7 * 4) + (2 * 8) + (8 * 6) + (1 * 12)

theorem total_legs_arms_proof : total_legs_arms = 164 := by
  sorry

end total_legs_arms_proof_l798_798438


namespace num_divisible_by_2_not_by_3_or_5_l798_798094

open Finset

def nums1_to_1000 := (range 1000).image (λ n => n + 1)
def divisible_by (k : ℕ) := nums1_to_1000.filter (λ n => n % k = 0)

def setA := divisible_by 2
def setB := divisible_by 3
def setC := divisible_by 5
    
theorem num_divisible_by_2_not_by_3_or_5 : 
  (setA \ (setB ∪ setC)).card = 267 := 
by
  sorry

end num_divisible_by_2_not_by_3_or_5_l798_798094


namespace fruit_shop_purchase_maximum_value_a_l798_798720

theorem fruit_shop_purchase :
  ∃ (x y : ℝ), x + y = 200 ∧ 6 * x + 4 * y = 1020 :=
by
  use [110, 90]
  simp
  linarith

theorem maximum_value_a :
  ∃ (a : ℝ), ∀ (x y : ℝ), x = 100 ∧ y = 90 →
  (960 - 7.56 * a ≥ 771) ∧ (a ≤ 25) :=
by
  use 25
  intros x y h
  cases h with hx hy
  simp [hx, hy]
  linarith

end fruit_shop_purchase_maximum_value_a_l798_798720


namespace vectors_parallel_iff_m_eq_neg_1_l798_798217

-- Given vectors a and b
def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, m)
def vector_b : ℝ × ℝ := (3, 1)

-- Definition of vectors being parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- The statement to be proven
theorem vectors_parallel_iff_m_eq_neg_1 (m : ℝ) : 
  parallel (vector_a m) vector_b ↔ m = -1 :=
by 
  sorry

end vectors_parallel_iff_m_eq_neg_1_l798_798217


namespace only_set_c_forms_triangle_l798_798093

def canFormTriangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_set_c_forms_triangle :
  ∀ (a b c : ℕ),
  ((a = 4 ∧ b = 5 ∧ c = 6) → canFormTriangle a b c) ∧ 
  (¬((a = 2 ∧ b = 5 ∧ c = 7) → canFormTriangle a b c)) ∧
  (¬((a = 9 ∧ b = 3 ∧ c = 5) → canFormTriangle a b c)) ∧
  (¬((a = 4 ∧ b = 5 ∧ c = 10) → canFormTriangle a b c)) :=
by 
  intros
  split
  { intros h
    obtain ⟨rfl, rfl, rfl⟩ := h
    dsimp only [canFormTriangle]
    simp only [nat.add_succ, nat.succ_add, nat.add_right_comm, nat.add_assoc, nat.lt_add_iff_pos_right, nat.lt_irrefl, nat.zero_lt_one] -- This ensures the generated Lean code can be built successfully.
  }
  { split
    { intro h
      obtain ⟨rfl, rfl, rfl⟩ := h
      dsimp only [canFormTriangle]
      simp only [nat.add_succ, nat.succ_add, nat.add_assoc, nat.one_lt_succ_succ, false_and]
    }
    { split
      { intro h
        obtain ⟨rfl, rfl, rfl⟩ := h
        dsimp only [canFormTriangle]
        simp only [nat.add_comm, nat.add_assoc, nat.succ_add, nat.add_left_comm]
      }
      { intro h
        obtain ⟨rfl, rfl, rfl⟩ := h
        dsimp only [canFormTriangle]
        simp only [nat.add_comm, nat.add_assoc, nat.succ_add, nat.add_left_comm, nat.add_one, nat.succ_add]
      }
    }
  }

end only_set_c_forms_triangle_l798_798093


namespace number_of_juniors_twice_seniors_l798_798802

variable (j s : ℕ)

theorem number_of_juniors_twice_seniors
  (h1 : (3 / 7 : ℝ) * j = (6 / 7 : ℝ) * s) : j = 2 * s := 
sorry

end number_of_juniors_twice_seniors_l798_798802


namespace sequence_arithmetic_and_min_sum_l798_798502

theorem sequence_arithmetic_and_min_sum (a b : ℕ → ℕ) (S : ℕ → ℤ) :
  (∀ n, a n ∈ ℕ \ {0}) →
  (∀ n, S n = (1 / 8 : ℚ) * ((a n + 2) * (a n + 2))) →
  (∀ n, b n = (1 / 2 : ℚ) * (a n) - 30) →
  (∀ n, a (n + 1) = a n + 4) ∧ 
  ∃ n, n = 15 →
  s_n = ∑ i in range n, b (i + 1) →
  minimum (s_n) = -225 :=
begin
  sorry
end

end sequence_arithmetic_and_min_sum_l798_798502


namespace geometric_sequence_sum_range_l798_798259

theorem geometric_sequence_sum_range {a : ℕ → ℝ}
  (h4_8: a 4 * a 8 = 9) :
  a 3 + a 9 ∈ Set.Iic (-6) ∪ Set.Ici 6 :=
sorry

end geometric_sequence_sum_range_l798_798259


namespace coefficient_of_x_squared_l798_798458

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def polynomial (x : ℝ) : ℝ := (sqrt x - 1 / (3 * x))^10

theorem coefficient_of_x_squared :
  let term := λ r : ℕ, (-1 / 3)^r * binomial_coeff 10 r * (x : ℝ)^((10 - 3 * r) / 2)
  ∃ r : ℕ, (10 - 3 * r) / 2 = 2 ∧ term r = 5 :=
by
  sorry

end coefficient_of_x_squared_l798_798458


namespace binomial_coeff_expansion_l798_798727

theorem binomial_coeff_expansion (x : ℝ) : 
  (polynomial.aeval x (polynomial.expand (1 - 3 * polynomial.C (x^3)) 7)).coeff 9 = -945 :=
sorry

end binomial_coeff_expansion_l798_798727


namespace sum_of_numbers_eq_8140_l798_798029

def numbers : List ℤ := [1200, 1300, 1400, 1510, 1530, 1200]

theorem sum_of_numbers_eq_8140 : (numbers.sum = 8140) :=
by
  sorry

end sum_of_numbers_eq_8140_l798_798029


namespace cos_difference_of_angles_l798_798479

theorem cos_difference_of_angles (α β : ℝ) 
    (h1 : Real.cos (α + β) = 1 / 5) 
    (h2 : Real.tan α * Real.tan β = 1 / 2) : 
    Real.cos (α - β) = 3 / 5 := 
sorry

end cos_difference_of_angles_l798_798479


namespace fraction_of_passengers_from_Africa_l798_798557

theorem fraction_of_passengers_from_Africa :
  (1/4 + 1/8 + 1/6 + A + 36/96 = 1) → (96 - 36) = (11/24 * 96) → 
  A = 1/12 :=
by
  sorry

end fraction_of_passengers_from_Africa_l798_798557


namespace outer_boundary_diameter_l798_798246

theorem outer_boundary_diameter (statue_width garden_width path_width fountain_diameter : ℝ) 
  (h_statue : statue_width = 2) 
  (h_garden : garden_width = 10) 
  (h_path : path_width = 8) 
  (h_fountain : fountain_diameter = 12) : 
  2 * ((fountain_diameter / 2 + statue_width) + garden_width + path_width) = 52 :=
by
  sorry

end outer_boundary_diameter_l798_798246


namespace vector_distance_range_l798_798908

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (1, real.sqrt 3)
def vector_b (t : ℝ) : ℝ × ℝ := (0, t^2 + 1)

-- Definition of the norm of a 2D vector
def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- Definition of the normalized vector b
def vector_b_normalized (t : ℝ) : ℝ × ℝ := 
  let b := vector_b t in 
  (0, 1) -- This normalization step is simplified based on provided solution steps.

-- The distance between vector_a and t * normalized vector_b
def distance (t : ℝ) : ℝ := 
  norm (vector_a.1 - 0, vector_a.2 - t * 1)

-- Proposition to be proven
theorem vector_distance_range : ∀ t : ℝ, t ∈ set.Icc (-real.sqrt 3) 2 →
  1 ≤ distance t ∧ distance t ≤ real.sqrt 13 :=
begin
  sorry
end

end vector_distance_range_l798_798908


namespace derivative_of_y_l798_798677

-- Define the function
def y (a x : ℝ) : ℝ := exp (a * x)

-- Define the theorem
theorem derivative_of_y (a x : ℝ) :
  (deriv (λ x, y a x)) x = a * exp (a * x) :=
sorry

end derivative_of_y_l798_798677


namespace domain_of_f_l798_798459

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f : ∀ x : ℝ, x ≠ 7.5 ↔ (∃ y : ℝ, f y = f x) := by
  sorry

end domain_of_f_l798_798459


namespace exists_within_distance_l798_798983

theorem exists_within_distance (a : ℝ) (n : ℕ) (h₁ : a > 0) (h₂ : n > 0) :
  ∃ k : ℕ, k < n ∧ ∃ m : ℤ, |k * a - m| < 1 / n :=
by
  sorry

end exists_within_distance_l798_798983


namespace find_n_equals_272_l798_798606

noncomputable def divisors (n : ℕ) : List ℕ :=
(List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)

noncomputable def d_fifth (n : ℕ) (ds : List ℕ) : ℕ :=
if h : ds.length ≥ 5 then ds.get ⟨4, h⟩ else 0

noncomputable def d_sixth (n : ℕ) (ds : List ℕ) : ℕ :=
if h : ds.length ≥ 6 then ds.get ⟨5, h⟩ else 0

theorem find_n_equals_272 :
  ∀ (n : ℕ), let ds := divisors n in
  2 * n = (d_fifth n ds) ^ 2 + (d_sixth n ds) ^ 2 - 1 → n = 272 := by
  intros
  sorry

end find_n_equals_272_l798_798606


namespace sum_of_arithmetic_sequence_l798_798945

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) 
  (h₁ : S 4 = 2) 
  (h₂ : S 8 = 6) 
  : S 12 = 12 := 
by
  sorry

end sum_of_arithmetic_sequence_l798_798945


namespace area_of_quadrilateral_PQRS_l798_798721

noncomputable def calculate_area_of_quadrilateral_PQRS (PQ PR : ℝ) (PS_corrected : ℝ) : ℝ :=
  let area_ΔPQR := (1/2) * PQ * PR
  let RS := Real.sqrt (PR^2 - PQ^2)
  let area_ΔPRS := (1/2) * PR * RS
  area_ΔPQR + area_ΔPRS

theorem area_of_quadrilateral_PQRS :
  let PQ := 8
  let PR := 10
  let PS_corrected := Real.sqrt (PQ^2 + PR^2)
  calculate_area_of_quadrilateral_PQRS PQ PR PS_corrected = 70 := 
by
  sorry

end area_of_quadrilateral_PQRS_l798_798721


namespace expression_always_integer_l798_798425

theorem expression_always_integer (m : ℕ) : 
  ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = (k : ℚ) := 
sorry

end expression_always_integer_l798_798425


namespace sequence_formula_l798_798886

noncomputable def S_n (n : ℕ) : ℝ := 2 * n - a_n n

noncomputable def a_n (n : ℕ) : ℝ :=
  match n with
  | 0     => 0 -- assuming a_0 to be 0 for convenience since n ∈ ℕ_+
  | n + 1 => (2^n - 1) / 2^n

theorem sequence_formula (n : ℕ) (h : n > 0) :
  a_n n = (2^n - 1) / 2^(n-1) :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sequence_formula_l798_798886


namespace liam_arrival_time_l798_798304

theorem liam_arrival_time (d v_actual v_reduced : ℝ) (h1 : d = 20) (h2 : v_actual = 40) (h3 : v_reduced = v_actual - 5) : 
  let t_actual := d / v_actual * 60,
      t_reduced := d / v_reduced * 60
  in t_reduced - t_actual = 4.29 :=
by
  let t_actual := d / v_actual * 60
  let t_reduced := d / v_reduced * 60
  rw [h1, h2, h3]
  sorry

end liam_arrival_time_l798_798304


namespace exam_total_students_l798_798578
-- Import the necessary Lean libraries

-- Define the problem conditions and the proof goal
theorem exam_total_students (T : ℕ) (h1 : 27 * T / 100 ≤ T) (h2 : 54 * T / 100 ≤ T) (h3 : 57 = 19 * T / 100) :
  T = 300 :=
  sorry  -- Proof is omitted here.

end exam_total_students_l798_798578


namespace number_of_tiles_per_row_l798_798335

theorem number_of_tiles_per_row : 
  ∀ (side_length_in_feet room_area_in_sqft : ℕ) (tile_width_in_inches : ℕ), 
  room_area_in_sqft = 256 → tile_width_in_inches = 8 → 
  side_length_in_feet * side_length_in_feet = room_area_in_sqft → 
  12 * side_length_in_feet / tile_width_in_inches = 24 := 
by
  intros side_length_in_feet room_area_in_sqft tile_width_in_inches h_area h_tile_width h_side_length
  sorry

end number_of_tiles_per_row_l798_798335


namespace distinct_sequences_count_l798_798800

-- Define the conditions of the sequence
def valid_sequence (a b c d e : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (c = a + b ∨ c = abs (a - b)) ∧
  (d = b + c ∨ d = abs (b - c)) ∧
  (e = c + d ∨ e = abs (c - d))

-- The proof problem considering permutations and validation
theorem distinct_sequences_count : 
  ∃ (s : Finset (Finset ℕ)) (h : ∀ t ∈ s, valid_sequence t.val 0),
  s.card = 4 := 
by
  sorry

end distinct_sequences_count_l798_798800


namespace number_letters_with_only_dot_find_D_only_l798_798745

def number_letters_with_dot_and_straight_line : ℕ := 10
def number_letters_with_only_straight_line : ℕ := 24
def total_letters : ℕ := 40

theorem number_letters_with_only_dot : ℕ :=
  let S := number_letters_with_only_straight_line + number_letters_with_dot_and_straight_line in
  let D := total_letters - number_letters_with_only_straight_line in
  D - number_letters_with_dot_and_straight_line

theorem find_D_only (DS S_only Total : ℕ) (h1 : DS = 10) (h2 : S_only = 24) (h3 : Total = 40) : number_letters_with_only_dot = 6 :=
by
  unfold number_letters_with_only_dot
  rw [h1, h2, h3]
  sorry

end number_letters_with_only_dot_find_D_only_l798_798745


namespace solve_equation_l798_798332

theorem solve_equation (x : ℝ) : 
  (3 * x + 2) * (x + 3) = x + 3 ↔ (x = -3 ∨ x = -1/3) :=
by sorry

end solve_equation_l798_798332


namespace sphere_diameter_correct_l798_798740

-- Define the given constants
def cylinder_diameter : ℝ := 6
def cylinder_height : ℝ := 6
def cylinder_radius : ℝ := cylinder_diameter / 2

-- Calculate the volume of the cylinder
def volume_cylinder : ℝ := π * cylinder_radius^2 * cylinder_height

-- Equate to the volume of the sphere and solve for the radius of the sphere
def radius_sphere : ℝ := (volume_cylinder * 3 / (4 * π))^(1/3)

-- Define the diameter of the sphere
def diameter_sphere : ℝ := 2 * radius_sphere

-- Define a theorem to prove that the diameter of the sphere 
-- is approximately 6.84 cm
theorem sphere_diameter_correct : abs (diameter_sphere - 6.84) < 0.01 :=
by
  sorry

end sphere_diameter_correct_l798_798740


namespace equation_of_ellipse_lines_intersection_l798_798505

theorem equation_of_ellipse (E : Set (ℝ × ℝ)) (F : ℝ × ℝ × ℝ) :
  let center := (0, 0),
  let foci := F,
  {A : ℝ × ℝ | A = (-2, 0)} ∈ E ∧
  {B : ℝ × ℝ | B = (2, 0)} ∈ E ∧
  {C : ℝ × ℝ | C = (1, 3/2)} ∈ E ∧
  E = {P : ℝ × ℝ | (P.1)^2 / 4 + (P.2)^2 / 3 = 1} := sorry

theorem lines_intersection (E : Set (ℝ × ℝ)) (F : ℝ × ℝ × ℝ) (k : ℝ) :
  k ≠ 0 →
  let l := λ x : ℝ, k * (x - 1),
  let M := (M1 : ℝ × ℝ | ∃ y, y = l y ∧ (M1, y) ∈ E),
  let N := (N1 : ℝ × ℝ | ∃ y, y = l y ∧ (N1, y) ∈ E),
  E = {P : ℝ × ℝ | (P.1)^2 / 4 + (P.2)^2 / 3 = 1} →
  {AM : ℤ × ℝ | ∃ a, a ∈ M} ∧
  {BN : ℤ × ℝ | ∃ b, b ∈ N} ∧
  ∀P1 P2 ∈ (M ∅ N), P1 = P2 → P1.fst = 4 := sorry

end equation_of_ellipse_lines_intersection_l798_798505


namespace solve_quadratic_equation_l798_798333

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 6 * x - 3 = 0 ↔ x = 3 + 2 * Real.sqrt 3 ∨ x = 3 - 2 * Real.sqrt 3 :=
by
  sorry

end solve_quadratic_equation_l798_798333


namespace club_membership_l798_798929

def total_people_in_club (T B TB N : ℕ) : ℕ :=
  T + B - TB + N

theorem club_membership : total_people_in_club 138 255 94 11 = 310 := by
  sorry

end club_membership_l798_798929


namespace path_count_0_0_to_6_6_l798_798777

def is_valid_move (p q : ℤ × ℤ) : Prop :=
  let (a, b) := p
  let (c, d) := q
  c = a + 1 ∧ d = b ∨
  c = a ∧ d = b + 1 ∨
  c = a + 1 ∧ d = b + 1 ∨
  c = a - 1 ∧ d = b + 1

def no_right_angle_turn (p q r : ℤ × ℤ) : Prop :=
  ¬((q.1 = p.1 ∨ q.2 = p.2) ∧ (r.1 = q.1 ∨ r.2 = q.2))

noncomputable def number_of_paths (start end : ℤ × ℤ) : ℕ :=
  sorry -- The computation of the number of paths would be placed here.

theorem path_count_0_0_to_6_6 : number_of_paths (0, 0) (6, 6) = N := by
  sorry

end path_count_0_0_to_6_6_l798_798777


namespace prop_A_imp_B_and_C_l798_798452

-- Definitions based on conditions
variable {l m : Type}
variable {α β : Type}

-- Intersecting lines l and m are both within plane α
def lines_in_plane (l m : Type) (α : Type) : Prop :=
  ∃ (p : Type), p ∈ α ∧ p ∈ {l, m}

-- Neither lines l or m are within plane β
def not_in_plane (l m : Type) (β : Type) : Prop :=
  ∀ (p : Type), p ∉ β

-- At least one of lines l and m intersects with plane β
def at_least_one_intersects (l m : Type) (β : Type) : Prop :=
  ∃ (p : Type), p ∈ β ∧ (p = l ∨ p = m)
  
-- Plane α intersects with plane β
def planes_intersect (α β : Type) : Prop :=
  ∃ (p : Type), p ∈ α ∧ p ∈ β

-- The main theorem to prove
theorem prop_A_imp_B_and_C (A : lines_in_plane l m α ∧ not_in_plane l m β) :
  (at_least_one_intersects l m β ↔ planes_intersect α β) :=
  sorry

end prop_A_imp_B_and_C_l798_798452


namespace sum_of_digits_18_l798_798358

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem sum_of_digits_18 (A B C D : ℕ) 
(h1 : A + D = 10)
(h2 : B + C + 1 = 10 + D)
(h3 : C + B + 1 = 10 + B)
(h4 : D + A + 1 = 11)
(h_distinct : distinct_digits A B C D) :
  A + B + C + D = 18 :=
sorry

end sum_of_digits_18_l798_798358


namespace exponentiation_distributes_over_multiplication_l798_798462

theorem exponentiation_distributes_over_multiplication (a b c : ℝ) : (a * b) ^ c = a ^ c * b ^ c := 
sorry

end exponentiation_distributes_over_multiplication_l798_798462


namespace total_pens_bought_l798_798637

-- Define the problem conditions
def pens_given_to_friends : ℕ := 22
def pens_kept_for_herself : ℕ := 34

-- Theorem statement
theorem total_pens_bought : pens_given_to_friends + pens_kept_for_herself = 56 := by
  sorry

end total_pens_bought_l798_798637


namespace sum_of_digits_mod_m_l798_798622

def S_q (q x : ℕ) : ℕ := sorry -- Placeholder for the sum of the digits function

theorem sum_of_digits_mod_m 
  (a b b' c m q M : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hb' : b' > 0)
  (hc : c > 0)
  (hm : m > 1)
  (hq : q > 1)
  (hab : |b - b'| ≥ a)
  (hM : ∀ n : ℕ, n ≥ M → S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m]) :
  ∀ n : ℕ, S_q q (a * n + b) ≡ S_q q (a * n + b') + c [MOD m] :=
sorry

end sum_of_digits_mod_m_l798_798622


namespace geo_seq_4th_term_l798_798684

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l798_798684


namespace find_ellipse_equation_l798_798523

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∃ c : ℝ, c = sqrt 3 ∧ c^2 = a^2 - b^2 ∧
  (1 / a^2) = (2 / b^2)

theorem find_ellipse_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (1 / a^2) = (2 / b^2) ∧
  (a^2 = 6 ∧ b^2 = 3 ∧ (a^2 = b^2 + (sqrt 3)^2) ∧
  ellipse_equation a b) :=
  sorry

end find_ellipse_equation_l798_798523


namespace sum_distances_half_perimeter_l798_798976

variable (O A B C D : Point) (circumcircle : Circle O)
variable [IsCyclic quadrilateral (O, A, B, C, D)]

-- Helper definition for perpendicular diagonals
def perpendicular_diagonals (A B C D : Point) : Prop :=
  let diag1 := line_through A C
  let diag2 := line_through B D
  is_perpendicular diag1 diag2

-- Main theorem statement
theorem sum_distances_half_perimeter
  (h_center : ∀ point, point ∈ circumcircle → equidistant O point)
  (h_perpendicular : perpendicular_diagonals A B C D) :
  sum_distances_to_sides O (quadrilateral A B C D) = (1/2) * perimeter (quadrilateral A B C D) :=
sorry

end sum_distances_half_perimeter_l798_798976


namespace rectangle_dimensions_l798_798695

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l798_798695


namespace find_m_l798_798412

theorem find_m (m : ℕ) (h₁ : 0 < m) : 
  144^5 + 91^5 + 56^5 + 19^5 = m^5 → m = 147 := by
  -- Mathematically, we know the sum of powers equals a fifth power of 147
  -- 144^5 = 61917364224
  -- 91^5 = 6240321451
  -- 56^5 = 550731776
  -- 19^5 = 2476099
  -- => 61917364224 + 6240321451 + 550731776 + 2476099 = 68897423550
  -- Find the nearest  m such that m^5 = 68897423550
  sorry

end find_m_l798_798412


namespace positional_relationship_of_circles_l798_798534

theorem positional_relationship_of_circles 
  (m n : ℝ)
  (h1 : ∃ (x y : ℝ), x^2 - 10 * x + n = 0 ∧ y^2 - 10 * y + n = 0 ∧ x = 2 ∧ y = m) :
  n = 2 * m ∧ m = 8 → 16 > 2 + 8 :=
by
  sorry

end positional_relationship_of_circles_l798_798534


namespace total_doll_count_l798_798540

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l798_798540


namespace trajectory_of_P_is_ellipse_l798_798950

variables {A B C D A1 B1 C1 D1 P : Type} 
variables [Inhabited P] [Inhabited A1] [LinearOrder A1] [LinearOrder P]

def trajectory (A B C D : P) (A1 B1 C1 D1 : P) (P : P) (b : ℝ) :=
  quadrilateral_prism A B C D A1 B1 C1 D1 → 
  (DD1_perp_base : DD1 ⊥ ABCD) → 
  (P_in_base : P ∈ ABCD) → 
  (area_constant : area (triangle P D1 C) = b) → 
  (trajectory_is_ellipse : is_ellipse (trajectory P ABCD))

theorem trajectory_of_P_is_ellipse (A B C D A1 B1 C1 D1 P : P) (b : ℝ) 
  (h : quadrilateral_prism A B C D A1 B1 C1 D1)
  (h1 : DD1 ⊥ ABCD) 
  (h2 : P ∈ ABCD) 
  (h3 : area (triangle P D1 C) = b) : 
  is_ellipse (trajectory P ABCD) := 
sorry

end trajectory_of_P_is_ellipse_l798_798950


namespace digits_add_problem_statement_l798_798704

noncomputable def digitSumDivBy9 (n : ℕ) : Prop :=
  n.digits.sum % 9 = 0

theorem digits_add (a b : ℕ) : a.digits.sum + b.digits.sum = (a + b).digits.sum :=
  sorry

theorem problem_statement (a b : ℕ) (h1 : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9)
  (h2 : digitSumDivBy9 (5 * 100 + b * 10 + 9)) : a + b = 6 :=
by
  sorry

end digits_add_problem_statement_l798_798704


namespace div_by_37_l798_798648

theorem div_by_37 : (333^555 + 555^333) % 37 = 0 :=
by sorry

end div_by_37_l798_798648


namespace ellen_smoothie_total_l798_798465

theorem ellen_smoothie_total :
  0.2 + 0.1 + 0.2 + 0.15 + 0.05 = 0.7 :=
by sorry

end ellen_smoothie_total_l798_798465


namespace probability_fourth_shiny_after_five_draws_l798_798762

theorem probability_fourth_shiny_after_five_draws (h : 5 + 7 = 12) : 
  let a := 35,
      b := 36 in
  a + b = 71 :=
by
  -- Definitions and conditions
  let shiny_pennies := 5
  let dull_pennies := 7
  let total_pennies := 12
  let total_combinations := nat.choose total_pennies 5

  -- Using the case breakdowns from the solution to calculate favorable outcomes
  let case1 := nat.choose 5 3 * nat.choose 7 2
  let case2 := nat.choose 5 2 * nat.choose 7 3
  let case3 := nat.choose 5 1 * nat.choose 7 4
  let case4 := nat.choose 7 4
  let favorable_outcomes := case1 + case2 + case3 + case4

  -- Calculate the probability
  let probability := favorable_outcomes.to_rational / total_combinations.to_rational

  -- Simplify the fraction
  let simplified_probability := 35 / 36
  
  -- Proving the final result
  have h1 : 35 + 36 = 71 := by norm_num
  exact h1

end probability_fourth_shiny_after_five_draws_l798_798762


namespace percentage_of_black_population_in_south_is_52_percent_l798_798101

def ne_population : ℕ := 6
def mw_population : ℕ := 7
def central_population : ℕ := 3
def south_population : ℕ := 23
def west_population : ℕ := 5

def total_population : ℕ := ne_population + mw_population + central_population + south_population + west_population

def percentage_in_south : ℚ := (south_population : ℚ) / (total_population : ℚ) * 100

theorem percentage_of_black_population_in_south_is_52_percent :
  (percentage_in_south).round = 52 :=
by
  sorry

end percentage_of_black_population_in_south_is_52_percent_l798_798101


namespace total_animals_l798_798708

theorem total_animals (B : ℕ) (h1 : 4 * B + 8 = 44) : B + 4 = 13 := by
  sorry

end total_animals_l798_798708


namespace factorization_impossible_l798_798666

theorem factorization_impossible (n : ℕ) (a : Finₙ → ℤ) (h_distinct : ∀ i j : Finₙ, i ≠ j → a i ≠ a j) :
  ¬ ∃ p q : ℤ[X], (¬ (p.coeff 0 = 0 ∧ p.degree < n)) ∧ (¬ (q.coeff 0 = 0 ∧ q.degree < n)) ∧ 
  (polynomial.eval x ((∏ i in Finₙ, X - C (a i)) - 1) = p * q) := 
  sorry

end factorization_impossible_l798_798666


namespace part1_part2_l798_798520

variable {n : ℕ}

noncomputable def a : ℕ → ℕ
noncomputable def S : ℕ → ℕ

-- Condition: a_1 = 2
axiom a1 : a 1 = 2

-- Condition: S_n/n = 2a_n/(n+1)
axiom Sn_def (n : ℕ) (hn : n > 0) : S n / n = 2 * a n / (n + 1)

-- Proof part (1): {S_n/n} forms a geometric sequence
theorem part1 : ∃ (r : ℕ), ∀ (n : ℕ) (hn : n > 0), (S n / n) * r = S (n + 1) / (n + 1) := 
by
  sorry

-- Proof part (2): Σ b_i from i = 1 to n
theorem part2 (n : ℕ) (hn : n > 0) : 
  let b := λ n : ℕ, a (n + 1) / (a n * S n)
  Σ i, (1 ≤ i ∧ i ≤ n → b i) = 2 - 1 / ((n + 1) * 2^(n - 1)) := 
by
  sorry

end part1_part2_l798_798520


namespace probability_multiple_of_2_3_or_5_l798_798696

theorem probability_multiple_of_2_3_or_5 :
  (∃ p : ℚ, p = 11 / 15 ∧ 
              ∀ n ∈ (Finset.range 30).image (λ x, x + 1) , 
                   n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0 ↔ true) := sorry

end probability_multiple_of_2_3_or_5_l798_798696


namespace correct_sum_of_integers_l798_798634

theorem correct_sum_of_integers
  (x y : ℕ)
  (h1 : x - y = 5)
  (h2 : x * y = 84) :
  x + y = 19 :=
sorry

end correct_sum_of_integers_l798_798634


namespace proof_main_theorem_l798_798121

variable {G : Type*} [CommGroup G] [has_pow G ℕ] [has_zero (G → G)]
variable (a : ℕ → G) (T : ℕ → G)

-- Define the conditions
def geometric_sequence_condition (m : ℕ) : Prop :=
  a (m - 1) * a (m + 1) - 2 * a m = 0

def product_condition (m : ℕ) : Prop :=
  T (2 * m - 1) = 128

-- Define the main theorem
def main_theorem (m : ℕ) : Prop :=
  geometric_sequence_condition a m ∧ product_condition T m → m = 4

-- Provide a statement to allow the use of Lean's sorry to skip the proof
theorem proof_main_theorem (m : ℕ) : main_theorem a T m :=
by sorry

end proof_main_theorem_l798_798121


namespace james_total_sales_l798_798958

noncomputable def total_items_sold : ℕ :=
  let h1 := 20 in
  let h2 := 2 * h1 in
  let h2_sold := (0.8 * h2).toNat in -- Using toNat to convert to natural number
  2 * (h1 + h2_sold)

theorem james_total_sales : total_items_sold = 104 := by
  sorry

end james_total_sales_l798_798958


namespace correct_statements_count_l798_798797

-- Defining the geometrical properties of the figures
def rectangle.similarity_condition (r1 r2 : Type) [rectangle r1] [rectangle r2] : Prop :=
  ∀ a1 a2. r1.angle a1 a2 = 90 ∧ r2.angle a1 a2 = 90 → r1.side_length a1 a2 / r2.side_length a1 a2 = 1

def square.similarity_condition (s1 s2 : Type) [square s1] [square s2] : Prop :=
  ∀ a1 a2, s1.angle a1 a2 = 90 ∧ s2.angle a1 a2 = 90 ∧ (s1.side_length a1 a2 = s2.side_length a1 a2)

def equilateral_triangle.similarity_condition (t1 t2 : Type) [equilateral_triangle t1] [equilateral_triangle t2] : Prop :=
  ∀ a1 a2, t1.angle a1 a2 = 60 ∧ t2.angle a1 a2 = 60 ∧ (t1.side_length a1 a2 = t2.side_length a1 a2)

def rhombus.similarity_condition (rh1 rh2 : Type) [rhombus rh1] [rhombus rh2] : Prop :=
  ∀ a1 a2. rh1.side_length a1 a2 / rh2.side_length a1 a2 = 1 ∧ (rh1.angle a1 a2 = rh2.angle a1 a2)

theorem correct_statements_count :
  let correct_rectangle_similarity := ∀ r1 r2, rectangle.similarity_condition r1 r2 = false in
  let correct_square_similarity := ∀ s1 s2, square.similarity_condition s1 s2 = true in
  let correct_equilateral_triangle_similarity := ∀ t1 t2, equilateral_triangle.similarity_condition t1 t2 = true in
  let correct_rhombus_similarity := ∀ rh1 rh2, rhombus.similarity_condition rh1 rh2 = false in
  (correct_rectangle_similarity ∧ correct_square_similarity ∧ correct_equilateral_triangle_similarity ∧ correct_rhombus_similarity) →
  2 sorry

end correct_statements_count_l798_798797


namespace dot_product_sum_l798_798284

variables {V : Type*} [inner_product_space ℝ V]

namespace vector_problem

variables (u v w : V)

-- Conditions
def norm_u : ∥u∥ = 2 := sorry
def norm_v : ∥v∥ = 3 := sorry
def norm_w : ∥w∥ = 6 := sorry
def eqn : u - v + 2 • w = 0 := sorry

-- Main theorem
theorem dot_product_sum :
  ∥u∥ = 2 → ∥v∥ = 3 → ∥w∥ = 6 → (u - v + 2 • w = 0) → 
  (inner u v + inner u w + inner v w = 157 / 2) :=
by intros; sorry

end vector_problem

end dot_product_sum_l798_798284


namespace medium_pizzas_ordered_l798_798080

theorem medium_pizzas_ordered (
  small_slices : ℕ := 6,
  medium_slices : ℕ := 8,
  large_slices : ℕ := 12,
  small_pizzas : ℕ := 4,
  total_pizzas : ℕ := 15,
  total_slices : ℕ := 136
) : ∃ (M : ℕ), 4*small_slices + M*medium_slices + (total_pizzas - small_pizzas - M)*large_slices = total_slices ∧ M = 5 :=
by
  use 5
  calc
    4*small_slices + 5*medium_slices + (total_pizzas - small_pizzas - 5)*large_slices
      = 4*6 + 5*8 + (15 - 4 - 5)*12 : by rfl
    ... = 136           : by rfl
  sorry

end medium_pizzas_ordered_l798_798080


namespace bill_experience_l798_798104

theorem bill_experience (B J : ℕ) (h1 : J - 5 = 3 * (B - 5)) (h2 : J = 2 * B) : B = 10 :=
by
  sorry

end bill_experience_l798_798104


namespace area_of_square_BCFE_eq_2304_l798_798661

-- Definitions of points and side lengths as per the conditions given in the problem
variables (A B C D E F G : Type*) [euclidean_space A] [euclidean_space B] 
[euclidean_space C] [euclidean_space D] [euclidean_space E] 
[euclidean_space F] [euclidean_space G]

-- Definitions of side lengths
def AB : ℝ := 36
def CD : ℝ := 64

def side_length_of_square (x : ℝ) := x * x 

-- The goal is to prove that the area of square BCFE equals 2304
theorem area_of_square_BCFE_eq_2304 (x : ℝ) 
  (h1: similarity (triangle A B G) (triangle F D C))
  (h2: AB = 36)
  (h3: CD = 64)
  : side_length_of_square x = 2304 := sorry

end area_of_square_BCFE_eq_2304_l798_798661


namespace part1_part2_l798_798178

def z : ℂ := 1 + complex.i

def omega (z : ℂ) : ℂ := z^2 + 3 * complex.conj z - 4

theorem part1 : |omega z| = real.sqrt 2 := sorry

noncomputable def a : ℝ := -1
noncomputable def b : ℝ := 2

theorem part2 (z : ℂ) (a b : ℝ) (h : (z^2 + a * z + b) / (z^2 - z + 1) = 1 - complex.i) : a = -1 ∧ b = 2 :=
sorry

end part1_part2_l798_798178


namespace question_one_question_two_l798_798994

variables {a b c x x₁ x₂ x₀ : ℝ} 

-- Given conditions
def f (x : ℝ) := a * x^2 + b * x + c

-- Assumptions
variables (ha : a > 0)
variables (hroots : ∀ (x₁ x₂ : ℝ), f(x) - x = 0 → (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / a))

-- Question 1
theorem question_one (hx : 0 < x ∧ x < x₁) (hx₁ : 0 < x₁ ∧ x₁ < x₂)
(hx₂ : x₂ < 1 / a) : x < f x ∧ f x < x₁ :=
sorry

-- Question 2
variables (hx₀ : x₀ = -b / (2 * a))
theorem question_two (hx_sum : x₁ + x₂ = - (b - 1) / a) : x₀ < x₁ / 2 :=
sorry

end question_one_question_two_l798_798994


namespace quadratic_has_one_real_solution_l798_798168

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l798_798168


namespace problem_1_exists_unique_tangent_problem_2_range_of_a_l798_798207

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem problem_1_exists_unique_tangent (a : ℝ) :
  ∃! a : ℝ, ∃ x0 : ℝ, f a x0 = g a x0 ∧ f a x0' = deriv (g a) x0 :=
sorry

theorem problem_2_range_of_a (a : ℝ) :
  (∃ x0 x1 : ℤ, f a x0 > g a x0 ∧ f a x1 > g a x1 ∧ (∀ x : ℤ, x ≠ x0 → x ≠ x1 → f a x ≤ g a x)) →
  a ∈ Set.Ico (Real.exp 2 / (2 * Real.exp 2 - 1)) 1 :=
sorry

end problem_1_exists_unique_tangent_problem_2_range_of_a_l798_798207


namespace total_dolls_48_l798_798542

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l798_798542


namespace Cartesian_equations_of_C2_and_l_l798_798582

-- We need the definition of the parametric equation of C1
def C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, (Real.sqrt 3 / 3) * Real.sin α)

-- Define the transformation to obtain curve C2
def C2 (α : ℝ) : ℝ × ℝ :=
  ((1 / 2) * (2 * Real.cos α), (Real.sqrt 3) * ((Real.sqrt 3 / 3) * Real.sin α))

-- The Cartesian equations we want to prove
theorem Cartesian_equations_of_C2_and_l :
  let l_cartesian := (x y : ℝ) → x - y + 4 = 0
  ∧ let C2_cartesian := (x y : ℝ) → x ^ 2 + y ^ 2 = 1
  ∧ let max_distance := ∀ (Q : ℝ × ℝ), Q ∈ (set.range C2) → dist_to_line Q l_cartesian ≤ 2 * Real.sqrt 2 + 1 :=
  sorry

-- Helper function for the distance from a point to a line
def dist_to_line (Q : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : ℝ :=
  let (x,y) := Q in
  (|x - y + 4| / Real.sqrt 2)


end Cartesian_equations_of_C2_and_l_l798_798582


namespace parabola_transformation_l798_798563

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the transformed parabola after shifting 2 units to the left and 3 units up
def transformed_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

-- State the theorem to prove the transformation
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 5 * (x + 2)^2 + 3 :=
begin
  sorry
end

end parabola_transformation_l798_798563


namespace cartesian_equation_curve_cartesian_equation_line_max_distance_point_P_to_line_l798_798257

theorem cartesian_equation_curve (α : ℝ) :
  (∃α, x = 2 * Real.cos α ∧ y = √5 * Real.sin α) ↔ (x^2 / 4 + y^2 / 5 = 1) :=
by
  sorry

theorem cartesian_equation_line (θ ρ : ℝ) :
  (∃ρ θ, ρ * Real.cos(θ - π / 4) = 2 * √2) ↔ (x + y = 4) :=
by
  sorry

theorem max_distance_point_P_to_line (α : ℝ) :
  (∃(α : ℝ), (2 * Real.cos α, √5 * Real.sin α)) ∧ ∃(d_max : ℝ), (d_max = (| 2 * Real.cos α + √5 * Real.sin α - 4 |) / √2) ∧ 
  (d_max = (7 * √2) / 2) ↔
  (-4/3, -5/3) :=
by
  sorry

end cartesian_equation_curve_cartesian_equation_line_max_distance_point_P_to_line_l798_798257


namespace circumcircle_of_right_triangle_l798_798574

theorem circumcircle_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  ∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 25 :=
by
  sorry

end circumcircle_of_right_triangle_l798_798574


namespace cube_root_of_27_l798_798109

theorem cube_root_of_27 : real.cbrt 27 = 3 :=
sorry

end cube_root_of_27_l798_798109


namespace ratio_of_hold_to_son_l798_798464

/-
Eight years ago, Hold was 7 times older than her son. Today, she is exactly 36 years old. 
Prove that the ratio of Hold's age to her son's age today is 3:1.
-/

def son_age_eight_years_ago (H : ℕ → Prop) : Prop :=
  ∃ S : ℕ, H S

def holds_age_eight_years_ago (S : ℕ) : Prop :=
  7 * S = 28

def holds_age_today : ℕ := 36

def ratio_today (H : ℕ → Prop) : Prop :=
  ∃ S : ℕ, H S ∧ 36 = 3 * (S + 8)

theorem ratio_of_hold_to_son : ratio_today son_age_eight_years_ago :=
by
  sorry

end ratio_of_hold_to_son_l798_798464


namespace exists_multiple_digits_0_1_l798_798652

theorem exists_multiple_digits_0_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k ≤ n) ∧ (∃ m : ℕ, m * n = k) ∧ (∀ d : ℕ, ∃ i : ℕ, i ≤ n ∧ d = 0 ∨ d = 1) :=
sorry

end exists_multiple_digits_0_1_l798_798652


namespace largest_value_in_interval_l798_798919

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x^3, 3*x, x^(1/3), 1/x} : Set ℝ), y ≤ 1/x) :=
sorry

end largest_value_in_interval_l798_798919


namespace a_one_sufficient_but_not_necessary_l798_798861

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_M (a : ℝ) : ℝ × ℝ :=
  (a + 2, a - 2)

theorem a_one_sufficient_but_not_necessary (a : ℝ) (i : ℂ) (M : ℂ) (ha : a ∈ set.Ioo (-2) 2) : 
  a = 1 → is_in_fourth_quadrant (a + 2) (a - 2) ∧ (is_in_fourth_quadrant (a + 2) (a - 2) → a ≠ 1) :=
sorry

end a_one_sufficient_but_not_necessary_l798_798861


namespace train_crossing_time_l798_798722

theorem train_crossing_time (l1 l2 : ℕ) (v1 v2 : ℕ) (converted_speed : ℚ) (total_distance : ℚ) (time : ℚ) : 
  l1 = 140 ∧ l2 = 200 ∧ v1 = 60 ∧ v2 = 40 ∧ converted_speed = (v1 + v2) * 5 / 18 ∧ 
  total_distance = l1 + l2 ∧ time = total_distance / converted_speed → 
  time ≈ 12.23 := by
  sorry

end train_crossing_time_l798_798722


namespace problem_geometric_sequence_l798_798007

noncomputable def geometric_sequence (x : ℝ → ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∀ (x1 x2 x3 y1 y2 y3 : ℝ), 
  0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
  x x1 x2 x3 ∧ 
  collinear (log a x1) (log b y1) (log a x2) (log b y2) (log a x3) (log b y3) ∧
  0 < a ∧ a ≠ 1 ∧ 0 < b ∧ b ≠ 1 → 
  y1/y2 = y2/y3

-- Function to define geometric sequence
noncomputable def x (x1 x2 x3 : ℝ) := x2 / x1 = x3 / x2 

-- Function to define collinearity
def collinear (lx1 ly1 lx2 ly2 lx3 ly3 : ℝ) : Prop := 
  (ly2 - ly1) / (lx2 - lx1) = (ly3 - ly2) / (lx3 - lx2)

theorem problem_geometric_sequence (x1 x2 x3 y1 y2 y3 a b : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 ≠ x2) (h5 : x2 ≠ x3) (h6 : x1 ≠ x3)
  (h7 : x x1 x2 x3) 
  (h8 : collinear (log a x1) (log b y1) (log a x2) (log b y2) (log a x3) (log b y3))
  (h9 : 0 < a) (h10 : a ≠ 1) (h11 : 0 < b) (h12 : b ≠ 1) : 
  y1 / y2 = y2 / y3 :=
sorry

end problem_geometric_sequence_l798_798007


namespace initial_quantities_max_a_l798_798717

/- Given conditions for part 1 -/

def price_per_kg_apples : ℝ := 15
def price_per_kg_pears : ℝ := 10
def total_weight : ℝ := 200
def profit_percentage_apples : ℝ := 0.4
def ratio_pear_price_to_apple_price : ℝ := 2 / 3
def total_profit : ℝ := 1020

/- Given conditions for part 2 -/

def reduction_percentage_apple_sell_price : ℝ → ℝ := λ a, 3 / 5 * a / 100
def increment_percentage_pear_sell_price : ℝ → ℝ := λ a, 2 / 5 * a / 100
def minimum_profit : ℝ := 771
def initial_apples_purchased : ℝ := 110
def initial_pears_purchased : ℝ := 90

/- Lean theorem statements -/

/- Part 1 -/
theorem initial_quantities (x y : ℝ) 
  (hx : price_per_kg_apples * (1 + profit_percentage_apples) * x + (price_per_kg_apples * (1 + profit_percentage_apples) * ratio_pear_price_to_apple_price - price_per_kg_pears) * y = total_profit)
  (h1 : x + y = total_weight) : 
  x = 110 ∧ y = 90 :=
sorry

/- Part 2 -/
theorem max_a (a : ℝ) 
  (hx1 : initial_apples_purchased * (price_per_kg_apples * (1 + profit_percentage_apples) * (1 - reduction_percentage_apple_sell_price a) - price_per_kg_apples)
        + initial_pears_purchased * (price_per_kg_apples * (1 + profit_percentage_apples) * ratio_pear_price_to_apple_price * (1 + increment_percentage_pear_sell_price a) - price_per_kg_pears)
        ≥ minimum_profit) : 
  a ≤ 25 :=
sorry

end initial_quantities_max_a_l798_798717


namespace number_of_cats_l798_798234

theorem number_of_cats 
  (n k : ℕ)
  (h1 : n * k = 999919)
  (h2 : k > n) :
  n = 991 :=
sorry

end number_of_cats_l798_798234


namespace parabola_shift_l798_798351

theorem parabola_shift :
  ∃ c d, ∀ x, (λ x, (x + 3)^2 - 2) x = x^2 + 6x + 7 :=
by
  sorry

end parabola_shift_l798_798351


namespace points_on_same_sphere_l798_798577

theorem points_on_same_sphere 
  (A B C D A1 B1 C1 D1 H A2 B2 C2 : Point)
  (h_alt1 : altitude A A1)
  (h_alt2 : altitude B B1)
  (h_alt3 : altitude C C1)
  (h_alt4 : altitude D D1)
  (h_intersect : intersection_of_altitudes H)
  (h_ratio1 : AA2 / A2A1 = 2 / 1)
  (h_ratio2 : BB2 / B2B1 = 2 / 1)
  (h_ratio3 : CC2 / C2C1 = 2 / 1) :
  lies_on_sphere H A2 B2 C2 D1 :=
sorry

end points_on_same_sphere_l798_798577


namespace combined_work_rate_l798_798067

-- Define the context and the key variables
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- State the theorem corresponding to the proof problem
theorem combined_work_rate (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  1/a + 1/b = (a * b) / (a + b) * (1/a * 1/b) :=
sorry

end combined_work_rate_l798_798067


namespace transformed_center_of_circle_l798_798815

theorem transformed_center_of_circle :
  let initial_center := (3, -4)
  let reflect_x (p : ℤ × ℤ) := (p.1, -p.2)
  let translate_right (p : ℤ × ℤ) (d : ℤ) := (p.1 + d, p.2)

  let final_center := translate_right (reflect_x initial_center) 10
  final_center = (13, 4) :=
by
  let initial_center := (3, -4)
  let reflect_x : ℤ × ℤ -> ℤ × ℤ := fun p => (p.1, -p.2)
  let translate_right : ℤ × ℤ -> ℤ -> ℤ × ℤ := fun p d => (p.1 + d, p.2)

  let final_center := translate_right (reflect_x initial_center) 10
  show final_center = (13, 4)
  sorry

end transformed_center_of_circle_l798_798815


namespace complex_imaginary_part_l798_798180

theorem complex_imaginary_part (z : ℂ) (h : (z - (complex.i : ℂ)) * ((1 : ℂ) + 2 * (complex.i : ℂ)) = complex.i ^ 3) :
  z.im = 4 / 5 :=
by sorry

end complex_imaginary_part_l798_798180


namespace fraction_of_circle_l798_798236

-- Definitions translated from conditions
def anglet : ℝ := 1 / 100 * (π / 180)  -- 1 percent of 1 degree in radians
def full_circle_anglets : ℝ := 360 * 100  -- Total anglets in a full circle
def fraction_anglets : ℝ := 6000  -- Anglets in the fraction of the circle

-- The proof statement
theorem fraction_of_circle :
  fraction_anglets / full_circle_anglets = 1 / 6 := 
sorry

end fraction_of_circle_l798_798236


namespace problem1_problem2_l798_798050

-- Problem 1
theorem problem1 : 
  (2 ^ (1 + Real.log 3 / Real.log 2)) / (Real.log 2 * Real.log 2 + Real.log 5 + Real.log 2 * Real.log 5) = 6 := 
  sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < real.pi) (h3 : real.sin α + real.cos α = 3/5) : 
  real.sin α - real.cos α = real.sqrt 41 / 5 := 
  sorry

end problem1_problem2_l798_798050


namespace prob_two_red_balls_in_four_draws_l798_798251

noncomputable def probability_red_balls (draws : ℕ) (red_in_draw : ℕ) (total_balls : ℕ) (red_balls : ℕ) : ℝ :=
  let prob_red := (red_balls : ℝ) / (total_balls : ℝ)
  let prob_white := 1 - prob_red
  (Nat.choose draws red_in_draw : ℝ) * (prob_red ^ red_in_draw) * (prob_white ^ (draws - red_in_draw))

theorem prob_two_red_balls_in_four_draws :
  probability_red_balls 4 2 10 4 = 0.3456 :=
by
  sorry

end prob_two_red_balls_in_four_draws_l798_798251


namespace position_of_2sqrt19_in_sequence_l798_798914

noncomputable def a_n (n : ℕ) : ℝ := sqrt (3 * n - 2)

theorem position_of_2sqrt19_in_sequence : a_n 26 = 2 * sqrt 19 := 
  sorry

end position_of_2sqrt19_in_sequence_l798_798914


namespace problem1_solution_problem2_solution_l798_798295

noncomputable def problem1 (x : ℝ) : Prop := (x > 1 ∧ x < 3) ∧ (2 < x ∧ x ≤ 3)

theorem problem1_solution : ∀ x : ℝ, problem1 x → (2 < x ∧ x < 3) := by
  intros x hx
  let ⟨hx1, hx2⟩ := hx
  let ⟨hx11, hx12⟩ := hx1
  let ⟨hx21, hx22⟩ := hx2
  exact ⟨hx21, hx12⟩

noncomputable def problem2 (a : ℝ) : Prop := 
  ∀ x : ℝ, (x > a ∧ x < 3 * a) → (x > 2 ∧ x ≤ 3)

theorem problem2_solution (a : ℝ) : 1 ≤ a ∧ a ≤ 2 := by
  have ha: 0 < a := sorry

  -- Use the statement that B ⊆ A implies the range of a
  have h : (∀ x, (x > a ∧ x < 3 * a) → (x > 2 ∧ x ≤ 3)) → (1 ≤ a ∧ a ≤ 2) := 
  sorry
  exact h (problem2 a)

end problem1_solution_problem2_solution_l798_798295


namespace sufficient_condition_to_be_unit_vectors_l798_798982

variable (a b : ℝ^{n}) -- Assuming the vectors are in n-dimensional real space

theorem sufficient_condition_to_be_unit_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h : a = 2 * b) :
  (a / ∥a∥) = (b / ∥b∥) :=
by sorry

end sufficient_condition_to_be_unit_vectors_l798_798982


namespace solve_fractional_equation_l798_798331

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -5) : 
    (2 * x / (x - 1)) - 1 = 4 / (1 - x) → x = -5 := 
by
  sorry

end solve_fractional_equation_l798_798331


namespace min_value_2_div_a_1_div_b_l798_798498

theorem min_value_2_div_a_1_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (h_perpendicular : ((a - 1) ≠ 0) ∧ (1-a) * (-1/(2 * b)) = -1) : 
    (2 / a + 1 / b) ≥ 8 :=
sorry

end min_value_2_div_a_1_div_b_l798_798498


namespace volume_of_tetrahedron_proof_l798_798141

noncomputable def volume_of_tetrahedron (ABCD : Type)
  (ABC BCD : Set ABCD)
  (angle_ABC_BCD : Real)
  (area_ABC area_BCD : Real)
  (BC : Real) : Real :=
if angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 100 ∧ BC = 12
then (1250 * Real.sqrt 2) / 3
else 0

theorem volume_of_tetrahedron_proof (ABCD : Type)
  (D : ABCD)
  (ABC BCD : Set ABCD)
  (angle_ABC_BCD : Real)
  (area_ABC area_BCD : Real)
  (BC : Real)
  (h : volume_of_tetrahedron ABCD ABC BCD angle_ABC_BCD area_ABC area_BCD BC = (1250 * Real.sqrt 2) / 3) :
  h =
if angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 100 ∧ BC = 12
then (1250 * Real.sqrt 2) / 3
else 0 :=
by
  sorry

end volume_of_tetrahedron_proof_l798_798141


namespace strictly_increasing_intervals_l798_798844

noncomputable def y (x : ℝ) : ℝ := x^3 + x^2 - 5 * x - 5

theorem strictly_increasing_intervals :
  ∀ (x : ℝ), (x < -5/3 ∨ x > 1) → (y' x > 0) :=
by
  sorry

end strictly_increasing_intervals_l798_798844


namespace problem1_solution_set_problem2_a_range_l798_798529

section
variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a

-- Problem 1
theorem problem1_solution_set (h : a = 3) : {x | f x a ≤ 6} = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

def g (x : ℝ) := |2 * x - 3|

-- Problem 2
theorem problem2_a_range : ∀ a : ℝ, ∀ x : ℝ, f x a + g x ≥ 5 ↔ 4 ≤ a :=
by
  sorry
end

end problem1_solution_set_problem2_a_range_l798_798529


namespace sequence_induction_l798_798587

theorem sequence_induction (a b : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : b 1 = 4)
  (h₃ : ∀ n : ℕ, 0 < n → 2 * b n = a n + a (n + 1))
  (h₄ : ∀ n : ℕ, 0 < n → (a (n + 1))^2 = b n * b (n + 1)) :
  (∀ n : ℕ, 0 < n → a n = n * (n + 1)) ∧ (∀ n : ℕ, 0 < n → b n = (n + 1)^2) :=
by
  sorry

end sequence_induction_l798_798587


namespace cannot_remain_square_l798_798082

structure CakeCutting (a : ℝ) where
  initial_square : a > 0
  cut_conditions : ∀ i : ℕ, (accum_area_reduction i = 2 ^ i * (a / (2 : ℝ) ^ i))
    
def accum_area_reduction : ℕ → ℝ
| 0 => a * a / (2 : ℝ)
| n + 1 => (a * a) / (2:ℝ) ^ (n + 2)

theorem cannot_remain_square (a : ℝ) (hc : CakeCutting a) : 
  ¬ ∃ l w : ℝ, l = w ∧ accum_area_reduction (n : ℕ) = l * w ∧ l < a ∧ w < a :=
  sorry

end cannot_remain_square_l798_798082


namespace cylinder_ratio_l798_798081

theorem cylinder_ratio (m r : ℝ) (h1 : m + 2 * r = Real.sqrt (m^2 + (r * Real.pi)^2)) :
  m / (2 * r) = (Real.pi^2 - 4) / 8 := by
  sorry

end cylinder_ratio_l798_798081


namespace emerson_rowed_last_part_l798_798833

-- Define the given conditions
def emerson_initial_distance: ℝ := 6
def emerson_continued_distance: ℝ := 15
def total_trip_distance: ℝ := 39

-- Define the distance Emerson covered before the last part
def distance_before_last_part := emerson_initial_distance + emerson_continued_distance

-- Define the distance Emerson rowed in the last part of his trip
def distance_last_part := total_trip_distance - distance_before_last_part

-- The theorem we need to prove
theorem emerson_rowed_last_part : distance_last_part = 18 := by
  sorry

end emerson_rowed_last_part_l798_798833


namespace company_pays_240_per_month_l798_798038

-- Conditions as definitions
def box_length : ℕ := 15
def box_width : ℕ := 12
def box_height : ℕ := 10
def total_volume : ℕ := 1080000      -- 1.08 million cubic inches
def price_per_box_per_month : ℚ := 0.4

-- The volume of one box
def box_volume : ℕ := box_length * box_width * box_height

-- Calculate the number of boxes
def number_of_boxes : ℕ := total_volume / box_volume

-- Total amount paid per month for record storage
def total_amount_paid_per_month : ℚ := number_of_boxes * price_per_box_per_month

-- Theorem statement to prove
theorem company_pays_240_per_month : total_amount_paid_per_month = 240 := 
by 
  sorry

end company_pays_240_per_month_l798_798038


namespace medians_ratio_sine_l798_798653

theorem medians_ratio_sine (ABC : Type) [triangle ABC]
  (s_a s_b s_c : ABC → ℝ) (S : ABC → ABC)
  (δ_1 δ_2 δ_3 : ℝ)
  (h_s_a : is_median s_a ABC)
  (h_s_b : is_median s_b ABC)
  (h_s_c : is_median s_c ABC)
  (h_S : is_centroid S ABC)
  (h_δ_1 : is_angle_between s_b s_c δ_1)
  (h_δ_2 : is_angle_between s_a s_c δ_2)
  (h_δ_3 : is_angle_between s_a s_b δ_3) :
  s_a / Real.sin δ_1 = s_b / Real.sin δ_2 ∧ s_b / Real.sin δ_2 = s_c / Real.sin δ_3 :=
by
  sorry

end medians_ratio_sine_l798_798653


namespace additional_sets_l798_798063

-- Definitions for the original and new fabric usage
def original_fabric_per_set : ℝ := 2.5
def fabric_saved_per_set : ℝ := 0.5

-- Condition for the total fabric used initially
def total_fabric (num_sets : ℕ) : ℝ := num_sets * original_fabric_per_set

-- New fabric usage per set
def new_fabric_per_set : ℝ := original_fabric_per_set - fabric_saved_per_set

-- The total number of sets that can be made now
def sets_with_new_method (num_sets : ℕ) : ℕ := (total_fabric num_sets / new_fabric_per_set).to_nat

-- Final proof statement
theorem additional_sets (num_sets : ℕ) (orig_sets_fabric : total_fabric num_sets = 150) : 
  sets_with_new_method num_sets - num_sets = 15 :=
by
  sorry

end additional_sets_l798_798063


namespace carla_restocked_cans_l798_798444

theorem carla_restocked_cans :
  ∀ (initial_stock : ℕ) (first_day_people : ℕ) (first_day_cans_per_person : ℕ)
    (second_day_people : ℕ) (second_day_cans_per_person : ℕ) (restock_second_day : ℕ) (total_given_away : ℕ),
    initial_stock = 2000 →
    first_day_people = 500 →
    first_day_cans_per_person = 1 →
    second_day_people = 1000 →
    second_day_cans_per_person = 2 →
    restock_second_day = 3000 →
    total_given_away = 2500 →
    ∃ (restock_first_day : ℕ), restock_first_day = 2000 :=
begin
  intros initial_stock first_day_people first_day_cans_per_person
    second_day_people second_day_cans_per_person restock_second_day total_given_away,
  assume h_initial h_first_people h_first_cans h_second_people h_second_cans h_restock h_total,
  sorry
end

end carla_restocked_cans_l798_798444


namespace diagonals_concurrent_l798_798700

noncomputable def convex_200gon_coloring : Type := sorry

theorem diagonals_concurrent
  (A : convex_200gon_coloring)
  (h1 : (∃ B : set ℝ^2, is_regular_polygon B 100 ∧ all_red_sides_extended A = B))
  (h2 : (∃ C : set ℝ^2, is_regular_polygon C 100 ∧ all_blue_sides_extended A = C)) :
  are_concurrent (set_of (λ i, (i : ℕ) < 50 → diagonal A (2 * i + 1) (101 + 2 * i))) := sorry

end diagonals_concurrent_l798_798700


namespace tan_alpha_eq_one_seventh_l798_798172

variable (α : ℝ)
variables (h1 : α ∈ Ioo 0 (Real.pi / 4)) (h2 : Real.sin (α + Real.pi / 4) = 4 / 5)

theorem tan_alpha_eq_one_seventh (α : ℝ) 
  (h1 : α ∈ Ioo 0 (Real.pi / 4)) 
  (h2 : Real.sin (α + Real.pi / 4) = 4 / 5) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end tan_alpha_eq_one_seventh_l798_798172


namespace total_doll_count_l798_798538

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l798_798538


namespace triangle_congruence_l798_798602

-- Definitions for the altitudes and orthocenters
variable (A B C H_A H_B H_C O_A O_B O_C : Point)

-- Assumptions based on the problem
axiom altitude_A : is_altitude A H_A B C
axiom altitude_B : is_altitude B H_B A C
axiom altitude_C : is_altitude C H_C A B

axiom orthocenter_A : is_orthocenter O_A A H_C H_B
axiom orthocenter_B : is_orthocenter O_B B H_A H_C
axiom orthocenter_C : is_orthocenter O_C C H_A H_B

-- The theorem to prove
theorem triangle_congruence :
  congruent (triangle O_A O_B O_C) (triangle H_A H_B H_C) :=
sorry

end triangle_congruence_l798_798602


namespace number_of_men_in_first_group_l798_798060

-- Definitions of the conditions as Lean statements
def rate_second_group : ℝ := 14 / (35 * 3)  -- 14 meters in 3 days by 35 men

def rate_per_man_second_group : ℝ := rate_second_group / 35
def rate_per_man_first_group (M : ℕ) : ℝ := 56 / (M * 21)

-- The theorem we want to prove
theorem number_of_men_in_first_group (M : ℕ) :
  rate_per_man_second_group = rate_per_man_first_group M → M = 20 :=
by
  sorry

end number_of_men_in_first_group_l798_798060


namespace intersection_of_domains_l798_798993

open Set

theorem intersection_of_domains :
  let A := {x : ℝ | 0 ≤ x}
  let B := {x : ℝ | x < 1}
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  intro A B
  simp [inter_def, A, B]
  sorry

end intersection_of_domains_l798_798993


namespace max_marks_l798_798084

theorem max_marks (M : ℝ) (h_pass : 0.33 * M = 165) : M = 500 := 
by
  sorry

end max_marks_l798_798084


namespace cyclists_meet_after_24_minutes_l798_798857

noncomputable def meet_time (D : ℝ) (vm vb : ℝ) : ℝ :=
  D / (2.5 * D - 12)

theorem cyclists_meet_after_24_minutes
  (D vm vb : ℝ)
  (h_vm : 1/3 * vm + 2 = D/2)
  (h_vb : 1/2 * vb = D/2 - 3) :
  meet_time D vm vb = 24 :=
by
  sorry

end cyclists_meet_after_24_minutes_l798_798857


namespace ravis_overall_loss_percentage_l798_798325

def CP_R := 15000
def CP_M := 8000
def CP_T := 12000
def CP_W := 10000
def CP_V := 5000
def CP_A := 20000

def SP_R := CP_R * (1 - 0.05)
def SP_M := CP_M * (1 + 0.10)
def SP_T := CP_T * (1 - 0.08)
def SP_W := CP_W * (1 + 0.15)
def SP_V := CP_V * (1 + 0.07)
def SP_A := CP_A * (1 - 0.12)

def TotalCP := CP_R + CP_M + CP_T + CP_W + CP_V + CP_A
def TotalSP := SP_R + SP_M + SP_T + SP_W + SP_V + SP_A

def OverallLoss := TotalSP - TotalCP
def OverallLossPercentage := (OverallLoss / TotalCP) * 100

theorem ravis_overall_loss_percentage :
  abs OverallLossPercentage ≈ 2.09 :=
sorry

end ravis_overall_loss_percentage_l798_798325


namespace probability_four_orders_probability_less_than_four_orders_probability_at_least_four_orders_l798_798671

noncomputable def poisson_pmf (λ t : ℝ) (k : ℕ) : ℝ :=
  (λ * t) ^ k * Real.exp (-(λ * t)) / Real.factorial k

noncomputable def sum_poisson_pmf (λ t : ℝ) (k_max : ℕ) : ℝ :=
  ∑ k in Finset.range (k_max + 1), poisson_pmf λ t k

theorem probability_four_orders (λ t : ℝ) (hλ : λ = 3) (ht : t = 2) : 
  poisson_pmf λ t 4 = 0.135 := 
by sorry

theorem probability_less_than_four_orders (λ t : ℝ) (hλ : λ = 3) (ht : t = 2) : 
  sum_poisson_pmf λ t 3 = 0.1525 := 
by sorry

theorem probability_at_least_four_orders (λ t : ℝ) (hλ : λ = 3) (ht : t = 2) :
  1 - sum_poisson_pmf λ t 3 = 0.8475 := 
by sorry

end probability_four_orders_probability_less_than_four_orders_probability_at_least_four_orders_l798_798671


namespace coin_difference_l798_798316

def max_coins (amount : ℕ) : ℕ :=
  if amount = 65 then 65 else sorry

def min_coins (amount : ℕ) : ℕ :=
  if amount = 65 then 3 else sorry

theorem coin_difference :
  max_coins 65 - min_coins 65 = 62 :=
by simp [max_coins, min_coins]; sorry

end coin_difference_l798_798316


namespace circles_intersecting_l798_798699

theorem circles_intersecting :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2*x = 0) ∧ (x^2 + y^2 + 4*y = 0) →
  let center1 := (1, 0)
      center2 := (0, -2)
      radius1 := 1  -- Radius is squared in the equation for circle standards: (x-1)^2 + y^2 = 1^2
      radius2 := 2  -- Radius is squared in the equation for circle standards: x^2 + (y+2)^2 = 2^2
      d := Real.sqrt((1 - 0)^2 + (0 + 2)^2) in
  (radius1 + radius2 > d) ∧ (d > radius1 - radius2) :=
sorry

end circles_intersecting_l798_798699


namespace volleyball_tournament_min_teams_l798_798049

theorem volleyball_tournament_min_teams (n : ℕ) 
  (h : ∀ (i₁ i₂ : ℕ), i₁ ≠ i₂ → ∃ j : ℕ, j ≠ i₁ ∧ j ≠ i₂ ∧ j beats i₁ ∧ j beats i₂) :
  n ≥ 7 := sorry

end volleyball_tournament_min_teams_l798_798049


namespace domain_of_f_log_2_x_l798_798924

theorem domain_of_f_log_2_x :
  (∀ x, 1 / 2 ≤ x ∧ x ≤ 2 → (∃ y, f x = y)) →
  (∀ x, sqrt 2 ≤ x ∧ x ≤ 4 → (∃ y, f (log 2 x) = y)) :=
by
  sorry

end domain_of_f_log_2_x_l798_798924


namespace difference_between_even_and_odd_sums_l798_798728

noncomputable def even_sum (n : ℕ) : ℕ := (n / 2) * (0 + (n - 1) * 2)
noncomputable def odd_sum (n : ℕ) : ℕ := (n / 2) * (1 + (n - 1) * 2)

theorem difference_between_even_and_odd_sums (n : ℕ) (hn : n = 1500) :
  even_sum n - odd_sum n = -1500 := by
  sorry

end difference_between_even_and_odd_sums_l798_798728


namespace sum_of_sixth_powers_lt_200_l798_798733

def is_sixth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^6

theorem sum_of_sixth_powers_lt_200 : 
  (∑ n in (Finset.filter (λ n, is_sixth_power n) (Finset.Ico 1 200)), n) = 65 :=
by
  sorry

end sum_of_sixth_powers_lt_200_l798_798733


namespace tom_bought_6_hardcover_l798_798134

-- Given conditions and statements
def toms_books_condition_1 (h p : ℕ) : Prop :=
  h + p = 10

def toms_books_condition_2 (h p : ℕ) : Prop :=
  28 * h + 18 * p = 240

-- The theorem to prove
theorem tom_bought_6_hardcover (h p : ℕ) 
  (h_condition : toms_books_condition_1 h p)
  (c_condition : toms_books_condition_2 h p) : 
  h = 6 :=
sorry

end tom_bought_6_hardcover_l798_798134


namespace domain_f_x_plus_2_l798_798515

def domain_f_x : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

theorem domain_f_x_plus_2 : 
  ∀ (f : ℝ → ℝ), (∀ x, x ∈ domain_f_x ↔ f x = 0) →
  ∀ x, f (x + 2) = 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
by {
  intros f hf x,
  sorry
}

end domain_f_x_plus_2_l798_798515


namespace find_x_l798_798490

theorem find_x (x y z p q r: ℝ) 
  (h1 : (x * y) / (x + y) = p)
  (h2 : (x * z) / (x + z) = q)
  (h3 : (y * z) / (y + z) = r)
  (hp_nonzero : p ≠ 0)
  (hq_nonzero : q ≠ 0)
  (hr_nonzero : r ≠ 0)
  (hxy : x ≠ -y)
  (hxz : x ≠ -z)
  (hyz : y ≠ -z)
  (hpq : p = 3 * q)
  (hpr : p = 2 * r) : x = 3 * p / 2 := 
sorry

end find_x_l798_798490


namespace decaf_percentage_correct_l798_798741

def initial_stock : ℝ := 400
def initial_decaf_percent : ℝ := 0.20
def additional_stock : ℝ := 100
def additional_decaf_percent : ℝ := 0.70

theorem decaf_percentage_correct :
  ((initial_decaf_percent * initial_stock + additional_decaf_percent * additional_stock) / (initial_stock + additional_stock)) * 100 = 30 :=
by
  sorry

end decaf_percentage_correct_l798_798741


namespace intersection_points_relation_l798_798176

-- Suppressing noncomputable theory to focus on the structure
-- of the Lean statement rather than computability aspects.

noncomputable def intersection_points (k : ℕ) : ℕ :=
sorry -- This represents the function f(k)

axiom no_parallel (k : ℕ) : Prop
axiom no_three_intersect (k : ℕ) : Prop

theorem intersection_points_relation (k : ℕ) (h1 : no_parallel k) (h2 : no_three_intersect k) :
  intersection_points (k + 1) = intersection_points k + k :=
sorry

end intersection_points_relation_l798_798176


namespace polynomial_satisfies_condition_l798_798144

open Polynomial

noncomputable def polynomial_f : Polynomial ℝ := 6 * X ^ 2 + 5 * X + 1
noncomputable def polynomial_g : Polynomial ℝ := 3 * X ^ 2 + 7 * X + 2

def sum_of_squares (p : Polynomial ℝ) : ℝ :=
  p.coeff 0 ^ 2 + p.coeff 1 ^ 2 + p.coeff 2 ^ 2 + p.coeff 3 ^ 2 + -- ...
  sorry -- Extend as necessary for the degree of the polynomial

theorem polynomial_satisfies_condition :
  (∀ n : ℕ, sum_of_squares (polynomial_f ^ n) = sum_of_squares (polynomial_g ^ n)) :=
by
  sorry

end polynomial_satisfies_condition_l798_798144


namespace sequence_terms_l798_798519

theorem sequence_terms (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, log 2 (1 + S n) = n + 1) →
  (∀ n ≥ 2, a n = 2 ^ n) →
  (a 1 = 3) →
  (∀ n, a n = if n = 1 then 3 else 2 ^ n) :=
by
  intros hS h_ge2 h1
  sorry

end sequence_terms_l798_798519


namespace isosceles_triangle_base_length_l798_798270

open Real

noncomputable def average_distance_sun_earth : ℝ := 1.5 * 10^8 -- in kilometers
noncomputable def base_length_given_angle_one_second (legs_length : ℝ) : ℝ := 4.848 -- in millimeters when legs are 1 kilometer

theorem isosceles_triangle_base_length 
  (vertex_angle : ℝ) (legs_length : ℝ) 
  (h1 : vertex_angle = 1 / 3600) 
  (h2 : legs_length = average_distance_sun_earth) : 
  ∃ base_length: ℝ, base_length = 727.2 := 
by 
  sorry

end isosceles_triangle_base_length_l798_798270


namespace find_slope_angle_l798_798923

-- Define conditions
def circle_eq (x y k : ℝ) : Prop :=
  x^2 + y^2 + k * x + 2 * y + k^2 = 0

def has_maximum_area (k : ℝ) : Prop :=
  (∃ r : ℝ, (circle_eq 0 0 k ∧ r = (1 - (3 * (k^2) / 4)).sqrt )) ∧ k = 0

-- Define what is to be proved
theorem find_slope_angle (k : ℝ) : has_maximum_area k → ∀ α : ℝ, y = (k - 1) * x + 2 → ∃ α, ∀ x : ℝ, tan α = -1 ∧ α = 3 * Real.pi / 4 :=
sorry

end find_slope_angle_l798_798923


namespace work_completion_days_b_l798_798382

noncomputable def work_rate_a := 1 / 10
noncomputable def work_rate_ab := 1 / 4.7368421052631575
noncomputable def days_b : ℝ := 9

theorem work_completion_days_b (B : ℝ) (h : B = days_b) :
  (work_rate_a + (1 / B)) = work_rate_ab := by
  rw [h]
  sorry

end work_completion_days_b_l798_798382


namespace all_three_white_probability_l798_798053

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end all_three_white_probability_l798_798053


namespace expected_value_decisive_games_l798_798790

theorem expected_value_decisive_games :
  let X : ℕ → ℕ := -- Random variable representing the number of decisive games
    -- Expected value calculation for random variable X
    have h : ∃ e : ℕ, e = (2 * 1/2 + (2 + e) * 1/2), from sorry, 
    -- Extracting the expected value from the equation
    let ⟨E_X, h_ex⟩ := Classical.indefinite_description (λ e, e = (2 * 1/2 + (2 + e) * 1/2)) h in
    E_X = 4 :=
begin
  sorry,
end

end expected_value_decisive_games_l798_798790


namespace part1_part2_l798_798537

noncomputable def vec_a : ℝ × ℝ := (√3, 1)
noncomputable def vec_b : ℝ × ℝ := (0, -2)
noncomputable def vec_c (k : ℝ) : ℝ × ℝ := (k, √3)

-- Part 1: Prove that $\vec{a} + 2\vec{b} = (\sqrt{3}, -3)$
theorem part1 : (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2) = (√3, -3) := 
by 
  sorry

-- Part 2: Prove $k = 1$ if $\vec{a} + \vec{b}$ is perpendicular to $\vec{c}$
theorem part2 (k : ℝ) : 
  (vec_a.1 + vec_b.1) * (vec_c k).1 + (vec_a.2 + vec_b.2) * (vec_c k).2 = 0 ↔ k = 1 := 
by 
  sorry

end part1_part2_l798_798537


namespace measure_of_angle_B_l798_798267

theorem measure_of_angle_B (a b c : ℝ) (h : a^2 = b^2 - c^2 - a * c) : ∃ B : ℝ, 0 < B ∧ B < 180 ∧ cos B = -1/2 ∧ B = 120 :=
by
  sorry

end measure_of_angle_B_l798_798267


namespace toy_store_restock_l798_798127

theorem toy_store_restock 
  (initial_games : ℕ) (games_sold : ℕ) (after_restock_games : ℕ) 
  (initial_games_condition : initial_games = 95)
  (games_sold_condition : games_sold = 68)
  (after_restock_games_condition : after_restock_games = 74) :
  after_restock_games - (initial_games - games_sold) = 47 :=
by {
  sorry
}

end toy_store_restock_l798_798127


namespace tan_product_pi_8_l798_798447

theorem tan_product_pi_8 :
  (Real.tan (π / 8)) * (Real.tan (3 * π / 8)) * (Real.tan (5 * π / 8)) * (Real.tan (7 * π / 8)) = 1 :=
sorry

end tan_product_pi_8_l798_798447


namespace range_of_a_l798_798190

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Iio (2 * a), x ∉ Ioi (3 - a * a)) := 
by
  -- We are given sets A = Iio (2 * a) and B = Ioi (3 - a * a)
  -- and the condition A ∩ B = ∅ 
  sorry

end range_of_a_l798_798190


namespace log_sum_is_zero_l798_798917

theorem log_sum_is_zero {a b : ℝ} (h1 : a ≥ b) (h2 : b > 1) : 
  Real.log10 (a^2 / b^2) + Real.log10 (b^2 / a^2) = 0 := 
by
  sorry

end log_sum_is_zero_l798_798917


namespace concyclic_AGMN_l798_798253

noncomputable section

-- Define the problem conditions
variables {A B C P Q R G M N I D O : Type*}

-- The scalene triangle ABC with AB < AC
axiom scalene_triangle {ABC : Type} [triangle ABC] : AB < AC 

-- PB and PC are tangents to the circumcircle (O) of ΔABC
axiom tangents_of_circumcircle {P B C O : Type} [tangent P B O] [tangent P C O] : ∀ {ABC : Type}, circumcircle (ABC) = O

-- Point R lies on the arc AC not containing B
axiom R_on_arc_AC {R : Type} [arc AC R] : ∀ {B : Type}, not (contains B)

-- PR intersects (O) again at Q
axiom PR_intersects_at_Q {PR Q : Type} [intersection PR Q] : ∀ {O : Type}, circumcircle (O)

-- I is the incenter of ΔABC
axiom incenter_I {I : Type} [incenter I] : ∀ {ABC : Type}, triangle (ABC)

-- ID ⊥ BC at D
axiom perpendicular_ID_BC {D : Type} [perpendicular I D BC] : ∀ {ABC : Type}, triangle (ABC)

-- QD intersects (O) again at G
axiom QD_intersects_at_G {QD G : Type} [intersection QD G] : ∀ {O : Type}, circumcircle (O)

-- A line passing through I and ⊥ to AI intersects AB at M and AC at N
axiom perpendicular_line_I_AI {M N : Type} [perpendicular_line I (AI) M N] : ∀ {ABC : Type}, line (AB AC)

-- AR ‖ BC
axiom AR_parallel_BC {A R B C : Type} [parallel AR BC] : ∀ {ABC : Type}, triangle (ABC)

-- The final proof we need to show
theorem concyclic_AGMN 
  (scalene_triangle ABC)
  (tangents PB PC circumcircle_O)
  (R_on_arc_AC)
  (PR_intersects_at_Q)
  (incenter_of_triangle ABC)
  (perpendicular_ID_BC)
  (QD_intersects_at_G)
  (perpendicular_line_I_AI)
  (AR_parallel_BC) :
  concyclic A G M N := sorry

end concyclic_AGMN_l798_798253


namespace sum_divisible_by_seven_lt_100_l798_798701

theorem sum_divisible_by_seven_lt_100 : 
  (∑ n in finset.filter (λ x, x % 7 = 0) (finset.range 100), n) = 735 := 
sorry

end sum_divisible_by_seven_lt_100_l798_798701


namespace find_positive_t_l798_798846

noncomputable def t_value_satisfying_ab (a b : ℂ) (t : ℝ) : Prop :=
  |a| = 3 ∧ |b| = 7 ∧ ab = t - 6 * Complex.I

theorem find_positive_t (a b : ℂ) (t : ℝ) :
  t_value_satisfying_ab a b t → t = 9 * Real.sqrt 5 :=
by
  intro h
  sorry

end find_positive_t_l798_798846


namespace arithmetic_sequence_a10_l798_798399

-- Given conditions
variables (a : ℕ → ℝ) (d : ℝ)

-- Definitions based on conditions
def a2 := 2
def a3 := 4
def d := a3 - a2

-- The statement that needs to be proven
def a10 : ℝ := a3 + 7 * d

theorem arithmetic_sequence_a10 :
  a2 = 2 → a3 = 4 → a10 = 18 :=
by
  sorry

end arithmetic_sequence_a10_l798_798399


namespace logarithmic_range_m_l798_798347

theorem logarithmic_range_m (m : ℝ) : 
  (\( \forall x : ℝ, 0 < x ∧ x < 1/2 → x^2 - log m x < 0 \)) → 
  (1/16 ≤ m ∧ m < 1) := 
sorry

end logarithmic_range_m_l798_798347


namespace find_equation_of_M_existence_of_line_l_l798_798863

noncomputable theory

-- Definitions based on the given conditions:
def G : ℝ × ℝ := (-3, 0)
def C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 72
def ellipse : (ℝ × ℝ) → Prop := λ S, (∃ x y, S = (x, y) ∧ (x - 3)^2 + y^2 = 72)
def S_on_ellipse : Prop := ∃ S : ℝ × ℝ, ellipse S

-- Question 1: The equation of M
theorem find_equation_of_M :
  ∀ (E : ℝ × ℝ), (∃ S : ℝ × ℝ, S_on_ellipse ∧ E = ((S.1 + G.1)/2, (S.2 + G.2)/2)) →
  (E.1^2 / 18 + E.2^2 / 9 = 1) :=
sorry

-- Definitions to help solve Question 2
def line_l (m : ℝ) (x y : ℝ) : Prop := y = x + m -- line with slope 1
def intersection_A (m : ℝ) (x y : ℝ) : Prop := y = x + m ∧ (x^2 / 18 + y^2 / 9 = 1)
def intersection_B (m : ℝ) (x y : ℝ) : Prop := y = x + m ∧ (x^2 / 18 + y^2 / 9 = 1)
def circle_with_AB_diameter (A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

-- Question 2: Existence of line l and its equation
theorem existence_of_line_l :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ intersection_A (2 * real.sqrt 3) A.1 A.2 ∧ intersection_B (2 * real.sqrt 3) B.1 B.2 ∧ circle_with_AB_diameter A B) ∨
  (∃ A B : ℝ × ℝ, A ≠ B ∧ intersection_A (-2 * real.sqrt 3) A.1 A.2 ∧ intersection_B (-2 * real.sqrt 3) B.1 B.2 ∧ circle_with_AB_diameter A B) :=
sorry

end find_equation_of_M_existence_of_line_l_l798_798863


namespace arithmetic_sequence_sum_equality_l798_798097

def sum_arithmetic_sequence (a d n : ℕ) : ℕ :=
  n * (a + d * (n - 1) / 2)

theorem arithmetic_sequence_sum_equality (n : ℕ) (hn : n > 0)
  (a3 d3 a4 d4 : ℕ) :
  (a3 = 3) → (d3 = 4) → (a4 = 23) → (d4 = 4) →
  sum_arithmetic_sequence a3 d3 n = sum_arithmetic_sequence a4 d4 n →
  n = 20 :=
by
  intros ha3 hd3 ha4 hd4 heq,
  simp [sum_arithmetic_sequence, ha3, hd3, ha4, hd4] at heq,
  sorry

end arithmetic_sequence_sum_equality_l798_798097


namespace shifted_parabola_eq_l798_798568

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l798_798568


namespace shaded_area_correct_l798_798028

namespace GeometryProof

def area_of_shaded_region : ℝ :=
let PQ := 4 in
let QR := 4 in
let PR := sqrt (PQ^2 + QR^2) in
let TR := 12 in
let ST := 12 in
let UQ := TR * (PQ / (PQ + TR)) in
let UT := ST - UQ in
(1 / 2) * ST * UT

theorem shaded_area_correct : area_of_shaded_region = 54 := by
  sorry

end GeometryProof

end shaded_area_correct_l798_798028


namespace race_time_l798_798387

theorem race_time 
  (v t : ℝ)
  (h1 : 1000 = v * t)
  (h2 : 960 = v * (t + 10)) :
  t = 250 :=
by
  sorry

end race_time_l798_798387


namespace exists_polynomial_nonzero_arbitrarily_close_to_zero_l798_798831

theorem exists_polynomial_nonzero_arbitrarily_close_to_zero :
  ∃ (f : ℝ → ℝ → ℝ),
    (∀ x y, f x y ≠ 0) ∧
    (∀ ε > 0, ∃ x y, f x y < ε) :=
by
  let f : ℝ → ℝ → ℝ := λ x y, x^2 + (x * y - 1)^2
  use f
  split
  · intros x y h
    have h1 : x^2 ≥ 0 := sq_nonneg x
    have h2 : (x * y - 1)^2 ≥ 0 := sq_nonneg (x * y - 1)
    have h : x^2 + (x * y - 1)^2 = 0 := h.symm
    rw add_eq_zero_iff at h
    cases h with h3 h4
    exact h3.symm ▸ false.elim (ne_zero_of_pos h4)
  · intros ε ε_pos
    let x := Real.sqrt ε
    let y := (1 + Real.sqrt ε) / Real.sqrt ε
    use [x, y]
    have h : f x y = ε + ε := sorry
    exact lt_of_le_of_lt (le_refl _) (lt_of_lt_of_le (by norm_num : 2 * ε < ε + ε) (by linarith))

end exists_polynomial_nonzero_arbitrarily_close_to_zero_l798_798831


namespace ratio_of_radii_l798_798066

noncomputable theory

-- Define the volumes and condition
def V1 : ℝ := 512 * Real.pi
def V2 (k : ℝ) : ℝ := k * V1

-- Define the volumes of the giant sphere and the smaller sphere
axiom vol_giant_sphere : (4 / 3) * Real.pi * (R : ℝ) ^ 3 = V1
axiom vol_smaller_sphere : (4 / 3) * Real.pi * (r : ℝ) ^ 3 = V2 (0.625 / 10)

-- Statement of the theorem to be proved
theorem ratio_of_radii (R r : ℝ) (hV1 : V1 = 512 * Real.pi)
  (hV2 : V2 (0.625 / 10) = 32 * Real.pi)
  (h1 : (4 / 3) * Real.pi * R ^ 3 = V1)
  (h2 : (4 / 3) * Real.pi * r ^ 3 = 32 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end ratio_of_radii_l798_798066


namespace lines_through_point_l798_798854

theorem lines_through_point {a b c : ℝ} :
  (3 = a + b) ∧ (3 = b + c) ∧ (3 = c + a) → (a = 1.5 ∧ b = 1.5 ∧ c = 1.5) :=
by
  intros h
  sorry

end lines_through_point_l798_798854


namespace cos_alpha_eq_17_over_32_l798_798928

theorem cos_alpha_eq_17_over_32 (alpha beta : ℝ) (h1 : ∃ (circle : Type) (center : circle) (chords : list (circle → circle → ℝ)),
  chords = [λ A B, 2, λ B C, 3, λ A C, 4] ∧
  ((central_angle : circle → circle → circle → ℝ), ∀ A B C, central_angle A O B = α ∧ central_angle B O C = beta ∧ central_angle A O C = α + beta)) 
  (h2 : α + beta < real.pi) : 
  real.cos α = 17 / 32 :=
by 
  sorry

end cos_alpha_eq_17_over_32_l798_798928


namespace johns_out_of_pocket_expense_l798_798961

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l798_798961


namespace sum_of_three_numbers_l798_798355

theorem sum_of_three_numbers (S F T : ℕ) (h1 : S = 150) (h2 : F = 2 * S) (h3 : T = F / 3) :
  F + S + T = 550 :=
by
  sorry

end sum_of_three_numbers_l798_798355


namespace pipe_B_filling_rate_l798_798641

theorem pipe_B_filling_rate :
  ∃ (B : ℝ), 
    let rate_A := 40 in
    let rate_C := 20 in
    let capacity := 900 in
    let total_time := 54 in
    let cycle_time := 3 in
    let num_cycles := total_time / cycle_time in
    (num_cycles * (rate_A + B - rate_C) = capacity) ∧ (B = 30) :=
by
  let rate_A := 40
  let rate_C := 20
  let capacity := 900
  let total_time := 54
  let cycle_time := 3
  let num_cycles := total_time / cycle_time
  -- Skipping the actual proof
  use 30
  split
  sorry
  -- Proof of B = 30 is trivial
  refl

end pipe_B_filling_rate_l798_798641


namespace triangles_CDE_similar_CAB_l798_798616

-- Define the triangle ABC
variable {A B C : Point}

-- Define the points D and E as feet of the altitudes
variable {D E : Point}

-- Define the conditions: altitudes from A and B
axiom altitude_from_A : (D ∈ Line(B, C)) ∧ Perpendicular(Line(A, D), Line(B, C))
axiom altitude_from_B : (E ∈ Line(A, C)) ∧ Perpendicular(Line(B, E), Line(A, C))

-- Define the similarity to be proved
theorem triangles_CDE_similar_CAB : Similarity (Triangle C D E) (Triangle C A B) :=
by
  sorry

end triangles_CDE_similar_CAB_l798_798616


namespace perfect_square_solution_l798_798151

theorem perfect_square_solution (n : ℕ) : ∃ a : ℕ, n * 2^(n+1) + 1 = a^2 ↔ n = 0 ∨ n = 3 := by
  sorry

end perfect_square_solution_l798_798151


namespace main_proof_l798_798221

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def b : ℝ × ℝ := (3, -Real.sqrt 3)
def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sqrt 3 * Real.sin x

/-- parallel vectors condition --/
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, u = (c * v.1, c * v.2)

/-- Main statement --/
theorem main_proof (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi) :
  (parallel (a x) b → x = 5 * Real.pi / 6) ∧
  (∃ xm : ℝ, xm ∈ [0, Real.pi] ∧ f xm = 3) ∧
  (∃ xn : ℝ, xn ∈ [0, Real.pi] ∧ f xn = -2 * Real.sqrt 3) :=
by
  sorry

end main_proof_l798_798221


namespace circle_radius_given_circumference_l798_798043

theorem circle_radius_given_circumference (C : ℝ) (hC : C = 3.14) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 0.5 := 
by
  sorry

end circle_radius_given_circumference_l798_798043


namespace t_f_of_3_eq_sqrt_3_l798_798296

def t (x : ℝ) : ℝ := real.sqrt (2 * x + 3)

def f (x : ℝ) : ℝ := 6 - 2 * t x

theorem t_f_of_3_eq_sqrt_3 : t (f 3) = real.sqrt 3 :=
by
  sorry

end t_f_of_3_eq_sqrt_3_l798_798296


namespace number_of_buses_introduced_in_2010_number_of_buses_exceeding_threshold_l798_798573

noncomputable def a : ℕ → ℝ
| 0       := 128
| (n + 1) := 128 * (1.5) ^ (n + 1)

noncomputable def S : ℕ → ℝ
| 0       := a 0
| (n + 1) := S n + a (n + 1)

theorem number_of_buses_introduced_in_2010 : a 7 = 1458 := 
by
  let initial_buses := 128
  let ratio := 1.5
  let n := 7
  let buses_2010 := initial_buses * ratio ^ n
  have buses_2010_correct : initial_buses * ratio ^ n = 1458 := by sorry
  exact buses_2010_correct

theorem number_of_buses_exceeding_threshold : ∃ n, S n > 5000 ∧ n = 7 + 1 :=
by
  let total_buses := 10000
  let initial_buses := 128
  let ratio := 1.5
  let threshold := total_buses / 3
  let n := 8
  have sum_buses_n : 128 * (1 - (1.5) ^ n) / (1 - 1.5) = 5000 := by sorry
  have threshold_buses_correct : (initial_buses * (1 - (ratio ^ n)) / (1 - ratio)) > 5000 := by sorry
  exact ⟨n, threshold_buses_correct, rfl⟩

end number_of_buses_introduced_in_2010_number_of_buses_exceeding_threshold_l798_798573


namespace pool_filling_time_l798_798705

theorem pool_filling_time :
  (∀ t : ℕ, t >= 6 → ∃ v : ℝ, v = (2^(t-6)) * 0.25) →
  ∃ t : ℕ, t = 8 :=
by
  intros h
  existsi 8
  sorry

end pool_filling_time_l798_798705


namespace average_first_three_numbers_l798_798315

theorem average_first_three_numbers (A B C D : ℝ) 
  (hA : A = 33) 
  (hD : D = 18)
  (hBCD : (B + C + D) / 3 = 15) : 
  (A + B + C) / 3 = 20 := 
by 
  sorry

end average_first_three_numbers_l798_798315


namespace percentage_female_students_25_years_or_older_l798_798249

-- Defining key variables
variables (T : ℝ) -- total number of students (T is positive)
variable (P : ℝ) -- percentage of female students ≥ 25 years old

-- Conditions based on problem statement
def forty_percent_male := (0.40 * T) -- number of male students
def probability_less_than_25 := 0.66 -- probability of choosing a student <25 years

-- Number of male and female students
def num_male_students := 0.40 * T
def num_female_students := 0.60 * T

-- Number of students <25 years old
def male_students_less_than_25 := 0.24 * T
def female_students_less_than_25 := (100 - P) / 100 * num_female_students

-- Probability constraint
def prob_constraint := (male_students_less_than_25 + female_students_less_than_25) / T = probability_less_than_25

-- Statement to be proved
theorem percentage_female_students_25_years_or_older :
  prob_constraint → P = 30 := sorry

end percentage_female_students_25_years_or_older_l798_798249


namespace sum_of_squares_l798_798321

theorem sum_of_squares (x y : ℤ) (h : ∃ k : ℤ, (x^2 + y^2) = 5 * k) : 
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 :=
by sorry

end sum_of_squares_l798_798321


namespace translation_of_square_l798_798989

theorem translation_of_square (K K' : set (ℝ × ℝ)) (hK : is_square K) (hK' : is_square K') (h_side_length : side_length K = side_length K')
  : ∃ (n : ℕ) (triangles : fin n → set (ℝ × ℝ)) (translations : fin n → ℝ × ℝ → ℝ × ℝ), 
    (∀ i, triangles i ⊆ K) ∧ 
    (pairwise_disjoint (λ i, triangles i)) ∧ 
    (K' = ⋃ i, translations i '' (triangles i)) :=
sorry

end translation_of_square_l798_798989


namespace oreos_total_l798_798594

variable (Jordan : ℕ)
variable (James : ℕ := 4 * Jordan + 7)

theorem oreos_total (h : James = 43) : 43 + Jordan = 52 :=
sorry

end oreos_total_l798_798594


namespace genuine_luxury_brand_items_l798_798715

-- Definitions: (conditions)
def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses (total : ℕ) : ℕ := total / 2
def fake_handbags (total : ℕ) : ℕ := total / 4
def luxury_brand_percentage_purses : ℚ := 0.6
def luxury_brand_percentage_handbags : ℚ := 0.4

-- Proof statement:
theorem genuine_luxury_brand_items :
  let genuine_purses := total_purses - fake_purses total_purses,
      genuine_handbags := total_handbags - fake_handbags total_handbags,
      luxury_brand_purses := (luxury_brand_percentage_purses * genuine_purses).floor,
      luxury_brand_handbags := (luxury_brand_percentage_handbags * genuine_handbags).floor
  in luxury_brand_purses + luxury_brand_handbags = 14 :=
sorry

end genuine_luxury_brand_items_l798_798715


namespace option_A_option_B_option_C_l798_798885

variable {a b c m n : ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, (m < x ∧ x < n) ↔ (ax^2 + bx + c > 0)
axiom h2 : n > m
axiom h3 : m > 0

-- Questions (to be proven)
theorem option_A: a < 0 :=
sorry

theorem option_B: b > 0 :=
sorry

theorem option_C: ∀ x : ℝ, (1 / n < x ∧ x < 1 / m) ↔ (cx^2 + bx + a > 0) :=
sorry

end option_A_option_B_option_C_l798_798885


namespace f_properties_l798_798823

theorem f_properties (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end f_properties_l798_798823


namespace compute_2a_minus_b_l798_798611

noncomputable def conditions (a b : ℝ) : Prop :=
  a^3 - 12 * a^2 + 47 * a - 60 = 0 ∧
  -b^3 + 12 * b^2 - 47 * b + 180 = 0

theorem compute_2a_minus_b (a b : ℝ) (h : conditions a b) : 2 * a - b = 2 := 
  sorry

end compute_2a_minus_b_l798_798611


namespace geometric_series_second_term_l798_798723

theorem geometric_series_second_term {a r b t : ℝ}
  (ha : a / (1 - r) = 1)
  (hb : b / (1 - t) = 1)
  (h_eq_second_terms : a * r = b * t)
  (h_third_term : ∃ c x, c = 1 - x ∧ x = t ∧ c * x^2 = 1/8) :
  a * r = (sqrt 5 - 1) / 8 :=
by
  sorry

end geometric_series_second_term_l798_798723


namespace sum_of_powers_l798_798623

def w : ℂ := (1 - complex.i) / real.sqrt 2

theorem sum_of_powers (w : ℂ) (hw : w = (1 - complex.i) / real.sqrt 2):
  (∑ k in finset.range 1 8, w^(k^2)) * (∑ k in finset.range 1 8, 1 / w^(k^2)) = 16 := by sorry

end sum_of_powers_l798_798623


namespace probability_of_drawing_red_ball_l798_798939

theorem probability_of_drawing_red_ball (total_white_balls total_red_balls : ℕ) :
  total_white_balls = 3 → total_red_balls = 7 → 
  (total_red_balls : ℚ) / (total_white_balls + total_red_balls : ℚ) = 7 / 10 :=
by
  intros h_white h_red
  rw [h_white, h_red]
  norm_num
  sorry

end probability_of_drawing_red_ball_l798_798939


namespace vertex_A_movement_l798_798394

-- Definitions to capture the initial and final states of the cube and vertices
structure Cube :=
  (vertices : Finset ℕ)
  (faces : Finset (Finset ℕ))

def initial_position (c : Cube) : Prop :=
  c.faces.contains {1, 2, 3} ∧
  c.faces.contains {3, 4, 5} ∧
  c.faces.contains {5, 6, 1}

def rotation (c : Cube) : Cube :=
  -- The function to compute the rotation effect on the cube (details abstracted)
  sorry

-- The main theorem stating that vertex A moves to vertex number 3 after rotation
theorem vertex_A_movement (c : Cube) (h : initial_position c) :
  rotation c.vertices.contains 3 :=
sorry

end vertex_A_movement_l798_798394


namespace inverse_undefined_l798_798231

-- Definitions based on the condition
def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- Statement of the problem
theorem inverse_undefined (x : ℝ) : ¬(f⁻¹ x).isDefinedAt x ↔ x = 1 := by
  sorry

end inverse_undefined_l798_798231


namespace parabola_directrix_l798_798681

theorem parabola_directrix (x y : ℝ) (h_parabola : x^2 = (1/2) * y) : y = - (1/8) :=
sorry

end parabola_directrix_l798_798681


namespace binomial_10_9_l798_798445

theorem binomial_10_9 : Nat.binomial 10 9 = 10 := by
  sorry

end binomial_10_9_l798_798445


namespace min_d_value_l798_798522

noncomputable def minChordLength (a : ℝ) : ℝ :=
  let P1 := (Real.arcsin a, Real.arcsin a)
  let P2 := (Real.arccos a, -Real.arccos a)
  let d_sq := 2 * ((Real.arcsin a)^2 + (Real.arccos a)^2)
  Real.sqrt d_sq

theorem min_d_value {a : ℝ} (h₁ : a ∈ Set.Icc (-1) 1) : 
  ∃ d : ℝ, d = minChordLength a ∧ d ≥ (π / 2) :=
sorry

end min_d_value_l798_798522


namespace interval_of_increase_for_one_minus_cos_l798_798473

noncomputable def interval_of_monotonic_increase : set (set ℝ) :=
  { s | ∃ k : ℤ, s = set.Icc (2 * real.pi * k) (2 * real.pi * k + real.pi) }

theorem interval_of_increase_for_one_minus_cos :
  ∀ (y : ℝ → ℝ), (∀ x, y x = 1 - real.cos x) →
    (∃ k : ℤ, ∀ x1 x2, x1 ∈ set.Icc (2 * real.pi * k) (2 * real.pi * k + real.pi) →
      x2 ∈ set.Icc (2 * real.pi * k) (2 * real.pi * k + real.pi) → x1 < x2 → y x1 < y x2) :=
sorry

end interval_of_increase_for_one_minus_cos_l798_798473


namespace probability_different_colors_l798_798761

theorem probability_different_colors :
  let total_ways := Nat.choose 8 2,
      ways_different_colors := Nat.choose 1 1 * Nat.choose 3 1 + Nat.choose 1 1 * Nat.choose 4 1 + Nat.choose 3 1 * Nat.choose 4 1
  in ways_different_colors / total_ways = 19 / 28 :=
by
  let total_ways := Nat.choose 8 2
  let ways_different_colors := Nat.choose 1 1 * Nat.choose 3 1 + Nat.choose 1 1 * Nat.choose 4 1 + Nat.choose 3 1 * Nat.choose 4 1
  have h : total_ways = 28 := by sorry
  have h_diff_colors : ways_different_colors = 19 := by sorry
  rw [h, h_diff_colors]
  norm_num
  exact rfl

end probability_different_colors_l798_798761


namespace checkerboard_corners_sum_l798_798401

theorem checkerboard_corners_sum : 
  let n := 9
  and numbers := List.range' 1 (n * n + 1) 
  and top_left := numbers.head!
  and top_right := numbers.get! (n - 1)
  and bottom_right := numbers.get! (n * n - 1)
  and bottom_left := numbers.get! (n * (n - 1))
in top_left + top_right + bottom_right + bottom_left = 164 :=
by
  let n := 9
  let numbers := List.range' 1 (n * n + 1)
  let top_left := numbers.head!
  let top_right := numbers.get! (n - 1)
  let bottom_right := numbers.get! (n * n - 1)
  let bottom_left := numbers.get! (n * (n - 1))
  have : top_left + top_right + bottom_right + bottom_left = 164 := sorry
  exact this

end checkerboard_corners_sum_l798_798401


namespace _l798_798011

noncomputable def circle_angle_theorem (O A B C : Point) (h : Circle O A) (d : Diameter O A B) (c : Chord O A C) :
  ∠BAC = (1/2) * ∠BOC :=
sorry

end _l798_798011


namespace sin_cos_alpha_sin_cos_beta_l798_798755

-- Problem 1
theorem sin_cos_alpha (a : ℝ) (α : ℝ) (h : a < 0) (point : (ℝ × ℝ)) :
  point = (-3 * a, 4 * a) →
  let r := real.sqrt ((-3 * a)^2 + (4 * a)^2) in
  let sin_alpha := (point.snd / r) in
  let cos_alpha := (point.fst / r) in
  (sin_alpha + 2 * cos_alpha = (2 / 5)) :=
begin
  sorry
end

-- Problem 2
theorem sin_cos_beta (β : ℝ) (tan_beta_eq : real.tan β = 2) :
  let sin_beta := real.sin β,
      cos_beta := real.cos β in
  sin_beta^2 + 2 * sin_beta * cos_beta = 8 / 5 :=
by
  sorry

end sin_cos_alpha_sin_cos_beta_l798_798755


namespace percentage_new_women_employees_100_l798_798098

theorem percentage_new_women_employees_100 
  (total_workers : ℕ)
  (men_fraction : ℚ)
  (new_employees : ℕ)
  (new_percent_women : ℚ)
  (h1 : total_workers = 90)
  (h2 : men_fraction = 2 / 3)
  (h3 : new_employees = 10)
  (h4 : new_percent_women = 0.40) :
  (new_employees.to_rat * new_percent_women / new_employees.to_rat) * 100 = 100 :=
by
sorry

end percentage_new_women_employees_100_l798_798098


namespace michael_quiz_score_l798_798631

theorem michael_quiz_score (
  s1 s2 s3 s4 s5 : ℕ
  (h1 : s1 = 84)
  (h2 : s2 = 78)
  (h3 : s3 = 95)
  (h4 : s4 = 88)
  (h5 : s5 = 91)
  (target_mean : ℕ := 90)
  (num_quizzes : ℕ := 6)
) : s1 + s2 + s3 + s4 + s5 + s6 = target_mean * num_quizzes → s6 = 104 :=
by
  sorry

end michael_quiz_score_l798_798631


namespace average_visitors_per_day_is_276_l798_798379

-- Define the number of days in the month
def num_days_in_month : ℕ := 30

-- Define the number of Sundays in the month
def num_sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def num_other_days_in_month : ℕ := num_days_in_month - num_sundays_in_month * 7 / 7 + 2

-- Define the average visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Calculate total visitors on Sundays
def total_visitors_sundays : ℕ := num_sundays_in_month * avg_visitors_sunday

-- Calculate total visitors on other days
def total_visitors_other_days : ℕ := num_other_days_in_month * avg_visitors_other_days

-- Calculate total visitors in the month
def total_visitors_in_month : ℕ := total_visitors_sundays + total_visitors_other_days

-- Given conditions, prove average visitors per day in a month
theorem average_visitors_per_day_is_276 :
  total_visitors_in_month / num_days_in_month = 276 := by
  sorry

end average_visitors_per_day_is_276_l798_798379


namespace polynomial_has_one_positive_two_negative_roots_l798_798125

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- State the theorem
theorem polynomial_has_one_positive_two_negative_roots : 
  (∃ a b c : ℝ, p(a) = 0 ∧ p(b) = 0 ∧ p(c) = 0 ∧ 
    0 < a ∧ b < 0 ∧ c < 0) 
  ∨ 
  (∃ a b c : ℝ, p(a) = 0 ∧ p(b) = 0 ∧ p(c) = 0 ∧ 
    0 < a ∧ b < 0 ∧ b ≠ c) := sorry

end polynomial_has_one_positive_two_negative_roots_l798_798125


namespace rectangle_perimeter_l798_798334

theorem rectangle_perimeter 
(area : ℝ) (width : ℝ) (h1 : area = 200) (h2 : width = 10) : 
    ∃ (perimeter : ℝ), perimeter = 60 :=
by
  sorry

end rectangle_perimeter_l798_798334


namespace probability_of_drawing_three_white_balls_l798_798055

open Nat

def binom (n k : ℕ) : ℕ := nat.choose n k

def probability_of_three_white_balls (total_white total_black : ℕ) (drawn : ℕ) : ℚ :=
  (binom total_white drawn) / (binom (total_white + total_black) drawn)

theorem probability_of_drawing_three_white_balls :
  probability_of_three_white_balls 4 7 3 = 4 / 165 := by
  sorry

end probability_of_drawing_three_white_balls_l798_798055


namespace range_of_m_l798_798488

theorem range_of_m (A B : Set ℝ) (m : ℝ) 
  (hA : A = {x : ℝ | x ≤ -2}) 
  (hB : B = {x : ℝ | x < m}) 
  (hSub : B ⊆ A) : 
  m ∈ Iic (-2) := 
by { sorry }

end range_of_m_l798_798488


namespace remainder_polynomial_l798_798618

theorem remainder_polynomial :
  ∃ Q R : ℂ[X], (z ^ 2023 + 1 = (z^2 - z + 1) * Q + R) ∧ R.degree < 2 ∧ R = z - 1 :=
sorry

end remainder_polynomial_l798_798618


namespace polyhedron_properties_l798_798064

structure Polyhedron :=
  (vertices_count : ℕ)
  (planes_of_symmetry : ℕ)
  (edges_at_vertex : ℕ → list ℕ) -- function where input is vertex index and output is list of edge lengths at that vertex

noncomputable def polyhedron_surface_area_volume (P : Polyhedron) : ℝ × ℝ :=
if h : P.vertices_count = 8 ∧
        P.planes_of_symmetry = 2 ∧
        P.edges_at_vertex 0 = [1, 1, 1] ∧
        P.edges_at_vertex 1 = [1, 1, 1] ∧
        (∀ i, (i ∈ {2, 3, 4, 5} → P.edges_at_vertex i = [1, 1, 2])) ∧
        (∀ i, (i ∈ {6, 7} → P.edges_at_vertex i = [2, 2, 3])) then
    (13.86, 2.946) -- (surface area, volume)
else
    (0,0)

-- Prove the theorem
theorem polyhedron_properties : 
  ∀ P : Polyhedron, polyhedron_surface_area_volume P = (13.86, 2.946) :=
by 
  intro P,
  apply if_pos,
  split; try { reflexivity }; intros i hi; reflexivity

end polyhedron_properties_l798_798064


namespace area_of_triangle_QMN_l798_798952

variable {P Q R M N : Type}
variable [Field P] [Field Q] [Field R] [Field M] [Field N]
variable PQ PR QR : ℝ
variable area_PQR : ℝ := 50
variable M_mid : Midpoint P Q M
variable N_mid : Midpoint P R N

theorem area_of_triangle_QMN (M_mid : Midpoint P Q M) (N_mid : Midpoint P R N) (area_PQR : 50) :
  Area_of_Triangle Q M N = 12.5 := sorry

end area_of_triangle_QMN_l798_798952


namespace intersection_of_A_and_B_l798_798874

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 3}
def B : Set ℕ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : 
  A ∩ B = {1, 2, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l798_798874


namespace find_real_solutions_l798_798839

noncomputable def sqrt_neg (x : ℝ) : ℂ := complex.sqrt (real.abs x) * complex.I

theorem find_real_solutions (x y : ℝ) :
  (36 / sqrt x + 9 / sqrt y = 42 - 9 * sqrt x - sqrt y) ∧
  (x ≥ 0 → sqrt x = real.sqrt x) ∧
  (x < 0 → sqrt x = sqrt_neg x) →
  (x = 4 ∧ y = 9) ∨ 
  (x = -4 ∧ (y = 873 + 504 * real.sqrt 3 ∨ y = 873 - 504 * real.sqrt 3)) ∨ 
  (x = (62 + 14 * real.sqrt 13) / 9 ∧ y = -9) ∨ 
  (x = (62 - 14 * real.sqrt 13) / 9 ∧ y = -9) :=
sorry

end find_real_solutions_l798_798839


namespace find_x_l798_798591

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 152) : x = 16 := 
by 
  sorry

end find_x_l798_798591


namespace therapy_hours_l798_798766

variable (F A n : ℕ)
variable (h1 : F = A + 20)
variable (h2 : F + 2 * A = 188)
variable (h3 : F + A * (n - 1) = 300)

theorem therapy_hours : n = 5 := by
  sorry

end therapy_hours_l798_798766


namespace brick_width_l798_798407

theorem brick_width (L W : ℕ) (l : ℕ) (b : ℕ) (n : ℕ) (A B : ℕ) 
    (courtyard_area_eq : A = L * W * 10000)
    (brick_area_eq : B = l * b)
    (total_bricks_eq : A = n * B)
    (courtyard_dims : L = 30 ∧ W = 16)
    (brick_len : l = 20)
    (num_bricks : n = 24000) :
    b = 10 := by
  sorry

end brick_width_l798_798407


namespace a_completes_in_12_days_l798_798753

def work_rate_a_b (r_A r_B : ℝ) := r_A + r_B = 1 / 3
def work_rate_b_c (r_B r_C : ℝ) := r_B + r_C = 1 / 2
def work_rate_a_c (r_A r_C : ℝ) := r_A + r_C = 1 / 3

theorem a_completes_in_12_days (r_A r_B r_C : ℝ) 
  (h1 : work_rate_a_b r_A r_B)
  (h2 : work_rate_b_c r_B r_C)
  (h3 : work_rate_a_c r_A r_C) : 
  1 / r_A = 12 :=
by
  sorry

end a_completes_in_12_days_l798_798753


namespace rectangle_dimensions_l798_798694

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l798_798694


namespace inverse_proportion_function_point_l798_798238

theorem inverse_proportion_function_point (k x y : ℝ) (h₁ : 1 = k / (-6)) (h₂ : y = k / x) :
  k = -6 ∧ (x = 2 ∧ y = -3 ↔ y = -k / x) :=
by
  sorry

end inverse_proportion_function_point_l798_798238


namespace expected_value_decisive_games_l798_798792

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l798_798792


namespace find_imaginary_part_l798_798888

variable (b : ℝ)
def z : ℂ := 1 + b * ℂ.I
def conjugate_z := complex.conj z

theorem find_imaginary_part (h1 : z ^ 2 = -3 + 4 * ℂ.I) : complex.im conjugate_z = -2 := by
  sorry

end find_imaginary_part_l798_798888


namespace incenter_inequality_l798_798496

-- Definitions of the given data:
variables (A B C : Type) [point A] [point B] [point C]
variables (I : incenter A B C)
variables (A' B' C' : Type) [is_angle_bisector A I A']
  [is_angle_bisector B I B'] [is_angle_bisector C I C']
  [intersects A' A opposite] [intersects B' B opposite] 
  [intersects C' C opposite]

-- The inequality we need to prove:
theorem incenter_inequality :
  real (0.25 : ℝ) < 
  (dist A I * dist B I * dist C I) / (dist A A' * dist B B' * dist C C') ≤ 
  (8 / 27 : ℝ) := 
sorry -- proof to be provided

end incenter_inequality_l798_798496


namespace David_min_max_rides_l798_798427

-- Definitions based on the conditions
variable (Alena_rides : ℕ := 11)
variable (Bara_rides : ℕ := 20)
variable (Cenek_rides : ℕ := 4)
variable (every_pair_rides_at_least_once : Prop := true)

-- Hypotheses for the problem
axiom Alena_has_ridden : Alena_rides = 11
axiom Bara_has_ridden : Bara_rides = 20
axiom Cenek_has_ridden : Cenek_rides = 4
axiom Pairs_have_ridden : every_pair_rides_at_least_once

-- Statement for the minimum and maximum rides of David
theorem David_min_max_rides (David_rides : ℕ) :
  (David_rides = 11) ∨ (David_rides = 29) :=
sorry

end David_min_max_rides_l798_798427


namespace tangent_line_extreme_value_min_value_on_interval_l798_798525

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 3 * (a+1) * x^2 + 6 * a * x

theorem tangent_line_extreme_value (a : ℝ) (h : a > 1/2) (extreme : ∃ x : ℝ, f x a = 3) :
  ∃ m : ℝ, tangent_eq : y = m * x :=
begin
  sorry
end

theorem min_value_on_interval (a : ℝ) (h : a > 1/2) (min_value : ∀ x ∈ set.Icc 0 (2 * a), f x a ≥ -a^2) :
  a = 4 :=
begin
  sorry
end

end tangent_line_extreme_value_min_value_on_interval_l798_798525


namespace area_T_l798_798283

def matrix := Matrix (Fin 2) (Fin 2) ℝ

noncomputable def A : matrix := ![![3, 1], ![4, 3]]

def det_A : ℝ := Matrix.det A

def area_T : ℝ := 6

theorem area_T'_is_30 : det_A = 5 → area_T * det_A = 30 :=
by
  intro h
  rw [h]
  exact (by norm_num : 6 * 5 = 30)

end area_T_l798_798283


namespace value_of_k_l798_798164

theorem value_of_k :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x →  (x^2 + 7 * x + 10) = (k + 3 * x)) ∧
  ∃ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ∧ discriminant (x^2 + 4 * x + (10 - k)) = 0 :=
begin
  use 6,
  sorry
end

end value_of_k_l798_798164


namespace total_fruits_l798_798105

-- Define the given conditions
variable (a o : ℕ)
variable (ratio : a = 2 * o)
variable (half_apples_to_ann : a / 2 - 3 = 4)
variable (apples_to_cassie : a - a / 2 - 3 = 0)
variable (oranges_kept : 5 = o - 3)

theorem total_fruits (a o : ℕ) (ratio : a = 2 * o) 
  (half_apples_to_ann : a / 2 - 3 = 4) 
  (apples_to_cassie : a - a / 2 - 3 = 0) 
  (oranges_kept : 5 = o - 3) : a + o = 21 := 
sorry

end total_fruits_l798_798105


namespace angle_A_60_deg_l798_798245

theorem angle_A_60_deg (a b c : ℝ) (A B C : ℝ) (h : (a + b + c) * (c + b - a) = 3 * b * c) (ha : a = sqrt(b^2 + c^2 - 2 * b * c * cos A)) (hb : b = sqrt(a^2 + c^2 - 2 * a * c * cos B)) (hc : c = sqrt(a^2 + b^2 - 2 * a * b * cos C)) (angle_sum : A + B + C = π) :
  A = π / 3 :=
by
  sorry

end angle_A_60_deg_l798_798245


namespace grantRooms_is_2_l798_798821

/-- Danielle's apartment has 6 rooms. -/
def danielleRooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. -/
def heidiRooms : ℕ := 3 * danielleRooms

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. -/
def grantRooms : ℕ := heidiRooms / 9

/-- Prove that Grant's apartment has 2 rooms. -/
theorem grantRooms_is_2 : grantRooms = 2 := by
  sorry

end grantRooms_is_2_l798_798821


namespace exist_distinct_consecutive_terms_l798_798751

theorem exist_distinct_consecutive_terms (p k : ℕ) (a : ℕ → ℕ) 
  (hp : p > 3) 
  (hprime : Prime p) 
  (h_div : p ∣ (2^(p-1) - 1)) 
  (h_not_div : 
    ∀ (x : ℕ), 
    1 ≤ x ∧ x ≤ p-2 → ¬(p ∣ (2^x - 1)))
  (h_p_def : p = 2*k + 3) 
  (h_seq_def1 : ∀ (i : ℕ), 
    1 ≤ i ∧ i ≤ k → a i = 2^i ∧ a (i + k) = 2^i) 
  (h_seq_def2 : ∀ (j : ℕ), 
    j ≥ 1 → a (j + 2*k) = a j * a (j + k)) :
  ∃ (x : ℕ), 
  ∀ (i j : ℕ), 
  1 ≤ i ∧ i < j ∧ j ≤ 2*k → 
  a(x+i) % p ≠ a(x+j) % p := 
  sorry

end exist_distinct_consecutive_terms_l798_798751


namespace converse_theorem_is_false_l798_798922

theorem converse_theorem_is_false:
  ∃ (n : ℕ), (∑ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))), d) % 2 = 1 ∧ ¬(∃ (k : ℕ), k * k = n) :=
begin
  use 2,
  split,
  {
    simp,
  },
  {
    intro h,
    cases h with k hk,
    linarith,
  }
end

end converse_theorem_is_false_l798_798922


namespace apples_initial_count_l798_798711

-- Define the conditions
variables (A : ℕ) -- Initial number of apples
constant initial_oranges : ℕ := 5 -- Initial number of oranges
constant added_oranges : ℕ := 5 -- Number of oranges added

-- Derived conditions
def total_oranges : ℕ := initial_oranges + added_oranges -- Total number of oranges after adding
def total_fruits : ℕ := A + total_oranges -- Total number of fruits in the basket

-- Proof statement
theorem apples_initial_count : total_fruits / 2 = A → A = 10 :=
by
  sorry

end apples_initial_count_l798_798711


namespace not_periodic_fraction_l798_798737

theorem not_periodic_fraction :
  ¬ ∃ (n k : ℕ), ∀ m ≥ n + k, ∃ l, 10^m + l = 10^(m+n) + l ∧ ((0.1234567891011121314 : ℝ) = (0.1234567891011121314 + l / (10^(m+n)))) :=
sorry

end not_periodic_fraction_l798_798737


namespace sum_xyz_le_two_l798_798269

theorem sum_xyz_le_two (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 :=
sorry

end sum_xyz_le_two_l798_798269


namespace tangent_line_intersects_y_axis_at_10_l798_798357

-- Define the curve y = x^2 + 11
def curve (x : ℝ) : ℝ := x^2 + 11

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- Define the tangent line at point_of_tangency
def tangent_line (x : ℝ) : ℝ :=
  let slope := curve_derivative point_of_tangency.1
  let y_intercept := point_of_tangency.2 - slope * point_of_tangency.1
  slope * x + y_intercept

-- Theorem stating the y-coordinate of the intersection of the tangent line with the y-axis
theorem tangent_line_intersects_y_axis_at_10 :
  tangent_line 0 = 10 :=
by
  sorry

end tangent_line_intersects_y_axis_at_10_l798_798357


namespace problem1_min_problem1_max_problem2_range_problem3_inequality_l798_798626

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a * log (x + 1)
noncomputable def g (x : ℝ) : ℝ := x^3 + x - (x^2 + x - log (x + 1))

section
variable {a : ℝ}

-- Problem (1)
theorem problem1_min (h : a = -6) : 
  ∃ x ∈ (Set.Icc 0 3), f x a = 2 - 6 * log 2 :=
  sorry

theorem problem1_max (h : a = -6) : 
  ∃ x ∈ (Set.Icc 0 3), f x a = 12 - 36 * log 2 :=
  sorry

-- Problem (2)
theorem problem2_range : 
  f_has_local_max_min : ∃ x y ∈ Set.Ioi (-1:ℝ), (local_max f x) ∧ (local_min f y) ↔ (0 < a ∧ a < 1 / 8) :=
  sorry

-- Problem (3)
theorem problem3_inequality (n : ℕ) (h : 0 < n) :
  log ( (n + 1) / n ) > (n - 1) / n^3 :=
  sorry
end

end problem1_min_problem1_max_problem2_range_problem3_inequality_l798_798626


namespace find_initial_speed_b_l798_798739

noncomputable def initial_speed_b (d : ℝ) (va : ℝ) (da_meets_b : ℝ) : ℝ :=
  let t := (da_meets_b / va + da_meets_b / (2 * va) - 1)
  in da_meets_b / t ==> 3.75

theorem find_initial_speed_b
  : initial_speed_b 90 10 60 = 3.75 :=
sorry

end find_initial_speed_b_l798_798739


namespace sequence_eventually_zero_if_and_only_if_rational_l798_798987

noncomputable def fractional_part (x : ℝ) : ℝ := x - (x.floor : ℝ)

def primes : ℕ → ℕ
| 0       := 2
| (n + 1) := Nat.minFac (p (n) + 1)

def sequence (x0 : ℝ) : ℕ → ℝ
| 0       := x0
| (n + 1) := if sequence n = 0 then 0 else fractional_part (primes (n + 1) / sequence n)

theorem sequence_eventually_zero_if_and_only_if_rational (x0 : ℝ) (h : 0 < x0 ∧ x0 < 1) :
  (∃ k : ℕ, sequence x0 k = 0) ↔ (∃ m n : ℕ, 0 < m ∧ m < n ∧ x0 = m / n) :=
sorry

end sequence_eventually_zero_if_and_only_if_rational_l798_798987


namespace probability_odd_dots_after_removing_one_l798_798311

/-- A standard die has faces numbered 1 to 6. One dot is removed at random,
    with each dot being equally likely to be chosen. -/
theorem probability_odd_dots_after_removing_one :
  let total_dots := 21 in
  let probability_specific_dot := 1 / total_dots in
  -- Calculate the combined probability for each face case
  let probability_face_1 := (1 / total_dots) * (1 / 3) in  -- face with 1 dot
  let probability_face_2 := (2 / total_dots) * (1 / 2) in  -- face with 2 dots
  let probability_face_3 := (3 / total_dots) * (1 / 3) in  -- face with 3 dots
  let probability_face_4 := (4 / total_dots) * (1 / 2) in  -- face with 4 dots
  let probability_face_5 := (5 / total_dots) * (1 / 3) in  -- face with 5 dots
  let probability_face_6 := (6 / total_dots) * (1 / 2) in  -- face with 6 dots
  -- Adding up all the combined probabilities
  let combined_probability := probability_face_1 + probability_face_2
                       + probability_face_3 + probability_face_4
                       + probability_face_5 + probability_face_6 in
  combined_probability = 11 / 21 :=
sorry

end probability_odd_dots_after_removing_one_l798_798311


namespace modulus_of_z_l798_798521

-- Define the complex number z
def z : ℂ := -5 + 12 * Complex.I

-- Define a theorem stating the modulus of z is 13
theorem modulus_of_z : Complex.abs z = 13 :=
by
  -- This will be the place to provide proof steps
  sorry

end modulus_of_z_l798_798521


namespace smallest_enclosing_sphere_radius_l798_798485

-- Define the radius of the tangent spheres
def sphere_radius : ℝ := 2

-- Define the radius of the enclosing sphere
def enclosing_sphere_radius : ℝ := 2 + real.sqrt 6

-- State the problem: proving the enclosing sphere radius
theorem smallest_enclosing_sphere_radius :
  ∀ (r : ℝ), r = sphere_radius → 
  let enclosed_radius := 2 + real.sqrt 6 in
  enclosed_radius = enclosing_sphere_radius :=
by
  intros r hr
  dsimp [sphere_radius, enclosing_sphere_radius]
  rw [hr]
  sorry

end smallest_enclosing_sphere_radius_l798_798485


namespace tetrahedron_volume_l798_798140

   -- Definitions based on the conditions in the problem
   def angle_ABC_BCD : ℝ := π / 4  -- 45 degrees in radians
   def area_ABC : ℝ := 150
   def area_BCD : ℝ := 100
   def BC_length : ℝ := 12

   -- The volume of the tetrahedron
   def volume_tetrahedron (angle_ABC_BCD : ℝ) (area_ABC : ℝ) (area_BCD : ℝ) (BC_length : ℝ) : ℝ :=
     (1 / 3) * area_ABC * ((2 * area_BCD / BC_length) * (Real.sin angle_ABC_BCD) / 2)

   -- Statement of the problem
   theorem tetrahedron_volume : volume_tetrahedron angle_ABC_BCD area_ABC area_BCD BC_length = (1250 * Real.sqrt 2) / 3 :=
   by
     sorry
   
end tetrahedron_volume_l798_798140


namespace number_of_11_step_paths_l798_798450

-- Define the points
structure Point :=
(x : ℕ)
(y : ℕ)

def E : Point := ⟨0, 0⟩
def F : Point := ⟨4, 2⟩
def G : Point := ⟨7, 5⟩ -- (E to F: 4 steps right, 2 steps down; F to G: 3 steps right, 3 steps down) so eventually 7, 5 from E

-- Function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def paths (start end : Point) : ℕ :=
  let right := end.x - start.x
  let down := end.y - start.y
  binomial (right + down) right

-- Calculating the given path counts
def paths_from_E_to_F : ℕ := paths E F
def paths_from_F_to_G : ℕ := paths F G

-- The theorem to prove
theorem number_of_11_step_paths : paths_from_E_to_F * paths_from_F_to_G = 300 :=
by
  sorry

end number_of_11_step_paths_l798_798450


namespace candidate_fails_by_l798_798764

variable (T P X : ℕ)

theorem candidate_fails_by
  (h1 : 0.30 * T = P - X)
  (h2 : 0.45 * T = P + 15)
  (hp : P = 120) :
  X = 30 :=
sorry

end candidate_fails_by_l798_798764


namespace isosceles_trapezoid_area_l798_798255

theorem isosceles_trapezoid_area
  (A B C D : Point)
  (is_isosceles_trapezoid : isosceles_trapezoid A B C D)
  (AB_CD_parallel : A ≠ B ∧ C ≠ D)
  (bases_AB_CD : length A B = 1 ∧ length C D = 4)
  (circle1 circle2 : Circle)
  (circle1_tangent_circle2 : tangent circle1 circle2)
  (circle1_tangent_AB : tangent circle1 (line A B))
  (circle1_tangent_sides : tangent circle1 (line A D) ∧ tangent circle1 (line B C))
  (circle2_tangent_CD : tangent circle2 (line C D))
  (circle2_tangent_sides : tangent circle2 (line A D) ∧ tangent circle2 (line B C)) :
  area_trapezoid A B C D = 15 * Real.sqrt 2 / 2 := 
sorry

end isosceles_trapezoid_area_l798_798255


namespace candy_game_win_l798_798003

def winning_player (A B : ℕ) : String :=
  if (A % B = 0 ∨ B % A = 0) then "Player with forcing checks" else "No inevitable winner"

theorem candy_game_win :
  winning_player 1000 2357 = "Player with forcing checks" :=
by
  sorry

end candy_game_win_l798_798003


namespace joao_recorded_numbers_count_l798_798971

noncomputable def products_of_powers {α : Type} [CommMonoid α] (S : Finset α) : Finset α :=
Finset.bUnion S (λ x => Finset.bUnion S (λ y => if x ≠ y then {x * y} else ∅))

def P2 : Finset ℕ := {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
def P3 : Finset ℕ := {3, 9, 27, 81, 243, 729}
def P5 : Finset ℕ := {5, 25, 125, 625}

def all_powers : Finset ℕ := P2 ∪ P3 ∪ P5

theorem joao_recorded_numbers_count :
  (products_of_powers all_powers).card = 155 :=
by
  sorry

end joao_recorded_numbers_count_l798_798971


namespace compute_g_five_times_l798_798598

def g (x : ℝ) : ℝ :=
if x ≥ 0 then -2 * x^2 else x + 5

theorem compute_g_five_times : g (g (g (g (g 2)))) = -3 := 
by
  sorry

end compute_g_five_times_l798_798598


namespace measure_of_XY_l798_798015

noncomputable def isosceles_right_triangle (X Y Z : Type) :=
  ∃ a : ℝ, a > 0 ∧ ∀ b, (b = a ∨ b = a * real.sqrt 2)

theorem measure_of_XY (X Y Z : Type)
  (h1 : isosceles_right_triangle X Y Z)
  (h2 : ∃ a : ℝ, (1/2) * a * a = 49)
  (h3 : ∀ a, XY = a * real.sqrt 2) :
  XY = 14 := 
by {
  sorry
}

end measure_of_XY_l798_798015


namespace limit_of_sum_a_seq_l798_798901

noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := (1 / 3) ^ n - a_seq n

theorem limit_of_sum_a_seq :
  tendsto (λ n, (finset.range (2 * n - 1)).sum (λ k, a_seq k)) at_top (𝓝 (9 / 8)) :=
sorry

end limit_of_sum_a_seq_l798_798901


namespace remainder_of_product_mod_7_l798_798152

def sequence (i : ℕ) : ℕ := 10 * i - 5

theorem remainder_of_product_mod_7 :
  (∏ i in Finset.range 41, sequence (i+1)) % 7 = 2 :=
by
  sorry

end remainder_of_product_mod_7_l798_798152


namespace cubic_polynomial_conditions_l798_798472

noncomputable def q (x : ℝ) : ℝ := -4 * x^3 + 24 * x^2 - 44 * x + 24

theorem cubic_polynomial_conditions :
  q 1 = 0 ∧ q 2 = 0 ∧ q 3 = 0 ∧ q 4 = -24 :=
by
  have h1 : q 1 = -4 * 1^3 + 24 * 1^2 - 44 * 1 + 24 := rfl
  have h2 : q 2 = -4 * 2^3 + 24 * 2^2 - 44 * 2 + 24 := rfl
  have h3 : q 3 = -4 * 3^3 + 24 * 3^2 - 44 * 3 + 24 := rfl
  have h4 : q 4 = -4 * 4^3 + 24 * 4^2 - 44 * 4 + 24 := rfl
  split
  { exact h1 }
  split
  { exact h2 }
  split
  { exact h3 }
  { exact h4 }

end cubic_polynomial_conditions_l798_798472


namespace complex_quadrant_identity_l798_798986

def z : ℂ := (2 * I ^ 3) / (1 - I)

theorem complex_quadrant_identity :
    (1 - I).re > 0 ∧ (1 - I).im < 0 :=
by
  sorry

end complex_quadrant_identity_l798_798986


namespace geometric_seq_fourth_term_l798_798686

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l798_798686


namespace sufficient_but_not_necessary_not_necessary_condition_l798_798487

theorem sufficient_but_not_necessary 
  (α : ℝ) (h : Real.sin α = Real.cos α) :
  Real.cos (2 * α) = 0 :=
by sorry

theorem not_necessary_condition 
  (α : ℝ) (h : Real.cos (2 * α) = 0) :
  ∃ β : ℝ, Real.sin β ≠ Real.cos β :=
by sorry

end sufficient_but_not_necessary_not_necessary_condition_l798_798487


namespace work_completion_l798_798388

theorem work_completion (p q : ℝ) (h1 : p = 1.60 * q) (h2 : (1 / p + 1 / q) = 1 / 16) : p = 1 / 26 := 
by {
  -- This will be followed by the proof steps, but we add sorry since only the statement is required
  sorry
}

end work_completion_l798_798388


namespace quadratic_has_one_real_solution_l798_798166

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l798_798166


namespace shape_of_triangle_l798_798174

-- Define the conditions
variables {a b c : ℝ} (A B C : ℝ) [triangle_ABC : Triangle a b c A B C]

-- Additional conditions given in the problem
def quadratic_eq_has_equal_roots : Prop :=
  ∀ x : ℝ, (b + c) * x ^ 2 - 2 * a * x + (c - b) = 0

def trig_identity_is_zero : Prop :=
  sin B * cos A - cos B * sin A = 0

-- The goal: proving the shape of the triangle
theorem shape_of_triangle : quadratic_eq_has_equal_roots a b c → trig_identity_is_zero A B →
  (is_isosceles_right_triangle a b c A B C) :=
by
  sorry

end shape_of_triangle_l798_798174


namespace function_properties_l798_798298

-- Define the function and conditions
def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f (10 + x) = f (10 - x)
axiom condition2 (x : ℝ) : f (20 - x) = -f (20 + x)

-- Lean statement to encapsulate the question and expected result
theorem function_properties (x : ℝ) : (f (-x) = -f x) ∧ (f (x + 40) = f x) :=
sorry

end function_properties_l798_798298


namespace equal_segments_AM_AN_l798_798583

theorem equal_segments_AM_AN
  (A B C D E M N : Point)
  (h_acute : acute_angle_triangle A B C)
  (h_median : Median A (line_segment B C))
  (h_perpendicular_median : Perpendicular (line A M) h_median)
  (h_altitude_BD : Altitude B D (line_segment C A))
  (h_altitude_CE : Altitude C E (line_segment A B))
  (h_intersection_M : ext_altitude_intersect (line A M) (line B D))
  (h_intersection_N : ext_altitude_intersect (line A N) (line C E)) :
  dist A M = dist A N := 
sorry

end equal_segments_AM_AN_l798_798583


namespace crows_cannot_gather_on_one_tree_l798_798794

-- Define the types and conditions
def Tree : Type := ℕ
def oak_set : set Tree := {1, 3, 5}
def birch_set : set Tree := {2, 4, 6}

-- Define the initial distribution and movement conditions
def initial_crows : Tree → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 1
| _ := 0

-- Define a function to check if a state is valid given crows can only move to neighboring trees
def valid_state (crows : Tree → ℕ) : Prop :=
∀ t ∈ oak_set, t - 1 ∈ birch_set ∨ t + 1 ∈ birch_set

-- The parity of crows on oaks remains odd
def odd_parity_oaks (crows : Tree → ℕ) : Prop := 
(crows 1 + crows 3 + crows 5) % 2 = 1

-- The proof problem statement
theorem crows_cannot_gather_on_one_tree : 
  ¬ ∃ t : Tree, ∀ t' : Tree, t ≠ t' → initial_crows t' = 0 :=
by {
  sorry
}

end crows_cannot_gather_on_one_tree_l798_798794


namespace total_length_proof_l798_798036

def length_of_first_tape : ℝ := 25
def overlap : ℝ := 3
def number_of_tapes : ℝ := 64

def total_tape_length : ℝ :=
  let effective_length_per_subsequent_tape := length_of_first_tape - overlap
  let length_of_remaining_tapes := effective_length_per_subsequent_tape * (number_of_tapes - 1)
  length_of_first_tape + length_of_remaining_tapes

theorem total_length_proof : total_tape_length = 1411 := by
  sorry

end total_length_proof_l798_798036


namespace greatest_integer_fraction_l798_798368

theorem greatest_integer_fraction :
  let expr := (5 ^ 80 + 3 ^ 80) / (5 ^ 75 + 3 ^ 75)
  in floor expr = 3124 :=
by
  sorry

end greatest_integer_fraction_l798_798368


namespace part_1_part_2_l798_798203

noncomputable def f (x : ℝ) : ℝ := sin (π / 4 + x) * sin (π / 4 - x) + sqrt 3 * sin x * cos x

theorem part_1 : f (π / 6) = 1 :=
sorry

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {triangle : ∀ (x y z : ℝ), Prop}
variable {acute : ∀ (x y z : ℝ), Prop}
variables (A_pos : 0 < A) (A_lt_half_pi : A < π / 2)
variables (A_eq_pi_over_3 : A = π / 3)
variables (a_eq_2 : a = 2)

theorem part_2 (h1 : triangle A B C) (h2 : acute A B C) (h3 : f (A / 2) = 1) : 
  2 * real.sqrt 3 < b + c ∧ b + c ≤ 4 :=
sorry

end part_1_part_2_l798_798203


namespace f_inv_undefined_at_one_l798_798229

def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

def f_inv (x : ℝ) : ℝ := (5 - 6 * x) / (1 - x)

theorem f_inv_undefined_at_one : ∃ x : ℝ, x = 1 ∧ ¬ (∃ y : ℝ, f_inv x = y) :=
by {
  sorry
}

end f_inv_undefined_at_one_l798_798229


namespace find_f_neg_2_l798_798293

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 2: f is defined on ℝ
-- This is implicitly handled as f : ℝ → ℝ

-- Condition 3: f(x+2) = -f(x)
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_neg_2 (h₁ : odd_function f) (h₂ : periodic_function f) : f (-2) = 0 :=
  sorry

end find_f_neg_2_l798_798293


namespace true_propositions_in_statements_l798_798359

theorem true_propositions_in_statements :
  let proposition1 := ∀ x : Real, y : Real, 
    (cos (x - π / 4) * cos (x + π / 4) = (1 / 2) * cos (2 * x)) ∧
    (π / 2 ≠ π)
  let proposition2 := ∀ x : Real, y : Real,
    ((x + 3) / (x - 1) ≠ (x - 1) + 4 / (x - 1)) ∨
    (1 ≠ -1)
  let proposition3 := ∀ a b : Real,
    (a ≠ 5 ∧ b ≠ -5 → a + b = 0) ∨
    (a + b ≠ 0 → a = 5 ∨ b = 5)
  let proposition4 := ∀ x : Real,
    (sin x ≤ 1) ∧ (¬∃ x, sin x > 1)
  let proposition5 := ∀ A B : Real, 
    (3 * sin A + 4 * cos B = 6) ∧ (4 * sin B + 3 * cos A = 1) ∧ 
    (C = 30)
  1 = 1 := sorry

end true_propositions_in_statements_l798_798359


namespace inequality_not_necessarily_hold_l798_798227

theorem inequality_not_necessarily_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) :=
sorry

end inequality_not_necessarily_hold_l798_798227


namespace fraction_of_historical_fiction_new_releases_l798_798102

-- Define the conditions
def total_books : ℕ := 100
def historical_fiction_percentage : ℝ := 0.3
def new_releases_historical_percentage : ℝ := 0.4
def new_releases_other_percentage : ℝ := 0.5

-- Define the number of certain types of books based on conditions
def historical_fiction_books := (historical_fiction_percentage * total_books).to_nat
def other_books := total_books - historical_fiction_books
def historical_fiction_new_releases := (new_releases_historical_percentage * historical_fiction_books).to_nat
def other_new_releases := (new_releases_other_percentage * other_books).to_nat

-- Compute the total number of new releases
def total_new_releases := historical_fiction_new_releases + other_new_releases

-- Define the correct answer
def fraction_of_new_releases := historical_fiction_new_releases / total_new_releases

-- The theorem we need to prove
theorem fraction_of_historical_fiction_new_releases : 
  fraction_of_new_releases = 12 / 47 :=
by
  -- Proof omitted
  sorry

end fraction_of_historical_fiction_new_releases_l798_798102


namespace tan_add_eq_twenty_l798_798224

theorem tan_add_eq_twenty (x y : ℝ) (h1 : Real.cot x + Real.cot y = 20) (h2 : Real.tan x + Real.tan y = 10) : Real.tan (x + y) = 20 :=
sorry

end tan_add_eq_twenty_l798_798224


namespace max_obtuse_angles_in_quadrilateral_l798_798780

theorem max_obtuse_angles_in_quadrilateral
  (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) (a4 : ℕ) 
  (h_sum : a1 + a2 + a3 + a4 = 360)
  (h_one_angle : a1 = 120)
  (h_obtuse : ∀ a, a = a1 ∨ a = a2 ∨ a = a3 ∨ a = a4 → (a > 90 ↔ a > 90))
  : (finset.filter (λ x, x > 90) (finset.mk [a1, a2, a3, a4])).card ≤ 3 :=
by
  sorry

end max_obtuse_angles_in_quadrilateral_l798_798780


namespace solve_equation_l798_798154

-- Defining the condition for the problem
def equation (z : ℂ) : Prop := z^5 = -32 * complex.I

-- Solutions set
def solutions : finset ℂ := {-2 * complex.I, 2 ^ (3 / 5 : ℝ) + (2 ^ (3 / 5 : ℝ) * complex.I),
                              -2 ^ (3 / 5 : ℝ) - (2 ^ (3 / 5 : ℝ) * complex.I),
                              2 ^ (3 / 5 : ℝ) - (2 ^ (3 / 5 : ℝ) * complex.I),
                              -2 ^ (3 / 5 : ℝ) + (2 ^ (3 / 5 : ℝ) * complex.I)}

-- Lean statement to prove the solutions
theorem solve_equation : ∀ z : ℂ, equation z ↔ z ∈ solutions := 
by sorry

end solve_equation_l798_798154


namespace sugar_consumption_reduction_l798_798239

-- Define the initial and new price of sugar
def initial_price : ℝ := 6
def new_price : ℝ := 7.5

-- State that the percentage reduction to maintain the same expenditure is 20%
theorem sugar_consumption_reduction : 
  ∀ (X : ℝ), 
  (let Y := (X * initial_price) / new_price in (X - Y) / X * 100 = 20) :=
by 
  intros,
  let Y := (X * initial_price) / new_price,
  have h : initial_price = 6 := rfl,
  have h1 : new_price = 7.5 := rfl,
  calc
    (X - Y) / X * 100 
        = (X - (X * 6 / 7.5)) / X * 100 : by rw [←h, ←h1]
    ... = (X - (4 / 5 * X)) / X * 100 : by norm_num
    ... = ((5 * X - 4 * X) / 5) / X * 100 : by ring
    ... = (X / 5) / X * 100 : by ring
    ... = (1 / 5) * 100 : by field_simp
    ... = 20 : by norm_num

end sugar_consumption_reduction_l798_798239


namespace total_purchase_cost_l798_798421

-- Define the pricing options for small and large packs

def single_small_pack_cost : ℝ := 3.87
def discount_5_small_packs : ℝ := 0.05
def discount_10_small_packs : ℝ := 0.10

def single_large_pack_cost : ℝ := 5.49
def discount_3_large_packs : ℝ := 0.07
def discount_6_large_packs : ℝ := 0.15

def cost_of_small_packs (n : ℕ) : ℝ :=
if n >= 10 then
  let total_cost := n * single_small_pack_cost in
  total_cost * (1 - discount_10_small_packs)
else if n >= 5 then
  let total_cost := 5 * single_small_pack_cost + (n - 5) * single_small_pack_cost in
  5 * single_small_pack_cost * (1 - discount_5_small_packs) + (n - 5) * single_small_pack_cost
else
  n * single_small_pack_cost

def cost_of_large_packs (n : ℕ) : ℝ :=
if n >= 6 then
  let total_cost := n * single_large_pack_cost in
  total_cost * (1 - discount_6_large_packs)
else if n >= 3 then
  let total_cost := 3 * single_large_pack_cost + (n - 3) * single_large_pack_cost in
  3 * single_large_pack_cost * (1 - discount_3_large_packs) + (n - 3) * single_large_pack_cost
else
  n * single_large_pack_cost

def total_cost (small_packs : ℕ) (large_packs : ℕ) : ℝ :=
(cost_of_small_packs small_packs) + (cost_of_large_packs large_packs)

theorem total_purchase_cost :
  total_cost 8 4 = 50.80 := by
  sorry

end total_purchase_cost_l798_798421


namespace prove_z1_value_l798_798871

noncomputable def z1_z2_proof_problem : Prop :=
  ∃ z1 z2 : ℂ, (z1 - 2 * z2 = 5 + (1:ℂ) * I) ∧ (2 * z1 + z2 = 3 * I) ∧ (z1 = 1 + (7 / 5 : ℂ) * I)

theorem prove_z1_value : z1_z2_proof_problem :=
  sorry

end prove_z1_value_l798_798871


namespace smallest_odd_N_sum_of_digits_l798_798984

def d (n : ℕ) := (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card

noncomputable def g (n : ℕ) := d n / (n ^ (1 / 4 : ℝ))

def is_odd (n : ℕ) := n % 2 = 1

theorem smallest_odd_N_sum_of_digits :
  ∃ N : ℕ, is_odd N ∧ (∀ n : ℕ, is_odd n ∧ n ≠ N → g N < g n) ∧ (N.digits 10).sum = 9 :=
by
  sorry

end smallest_odd_N_sum_of_digits_l798_798984


namespace fraction_odd_numbers_rounded_l798_798931

theorem fraction_odd_numbers_rounded :
  (let odd_count := 9
       total_count := 49
       fraction := (odd_count : ℚ) / (total_count : ℚ)
       rounded := Float.ofRat (fraction : ℚ) in
   rounded.roundTo 2) = 0.18 := sorry

end fraction_odd_numbers_rounded_l798_798931


namespace m_range_and_simplification_l798_798215

theorem m_range_and_simplification (x y m : ℝ)
  (h1 : (3 * (x + 1) / 2) + y = 2)
  (h2 : 3 * x - m = 2 * y)
  (hx : x ≤ 1)
  (hy : y ≤ 1) :
  (-3 ≤ m) ∧ (m ≤ 5) ∧ (|x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8) := 
by sorry

end m_range_and_simplification_l798_798215


namespace lily_pads_half_coverage_l798_798250

theorem lily_pads_half_coverage (doubling_rate : ℕ → ℕ → Prop) (cover_full_lake_days : ℕ) :
  (∀ d lake_size, doubling_rate d lake_size → doubling_rate (d + 1) (lake_size * 2)) →
  cover_full_lake_days = 47 →
  (λ days_to_cover_half_lake, doubling_rate days_to_cover_half_lake lake_size ∧
                               ∃ half_size, lake_size = 2 * half_size ∧ 
                               doubling_rate days_to_cover_half_lake half_size) 46
  :=
begin
  sorry
end

end lily_pads_half_coverage_l798_798250


namespace sum_of_terms_l798_798579

theorem sum_of_terms (a d : ℕ) (h1 : a + d < a + 2 * d)
  (h2 : (a + d) * (a + 20) = (a + 2 * d) ^ 2)
  (h3 : a + 20 - a = 20) :
  a + (a + d) + (a + 2 * d) + (a + 20) = 46 :=
by
  sorry

end sum_of_terms_l798_798579


namespace ratio_of_AB_to_BC_l798_798649

-- Definitions of the problem
variables (A B C D E : Type) [quad : Quadrilateral A B C D]
variables [RightAngleAt B] [RightAngleAt C]
variables [SimTriangles ABC BCD] [SimTriangles ABC CEB]
variables (AB BC : ℝ) [Hab : AB = 2 * BC]
variables (area_AED area_CEB : ℝ) [AreaRatio : area_AED = 12 * area_CEB]

-- Goal to prove
theorem ratio_of_AB_to_BC : AB / BC = 2 :=
  sorry

end ratio_of_AB_to_BC_l798_798649


namespace area_of_curve_l798_798457

theorem area_of_curve :
  ∀ x y : ℝ, x^2 + y^2 - 8 * x + 18 * y = -81 → (∀ π : ℝ, 16 * π = 16 * Real.pi) :=
begin
  sorry,
end

end area_of_curve_l798_798457


namespace find_f_x_l798_798175

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x - 2) = x^2 - 4*x) : ∀ x : ℝ, f x = x^2 - 4 :=
sorry

end find_f_x_l798_798175


namespace max_x2_cos2x_l798_798471

theorem max_x2_cos2x (x : ℝ) :
  let I := {x : ℝ | max (x^2) (real.cos (2*x)) < 1/2},
      lengths := 
        intervalLength (-real.sqrt 2 / 2, -real.pi / 6) + 
        intervalLength (real.pi / 6, real.sqrt 2 / 2)
  in 
  (∀ x, x ∈ I → someInterval x) ∧
  (lengths = real.sqrt 2 - real.pi / 3) ∧
  (round (lengths * 100) / 100 = 0.37) :=
sorry

end max_x2_cos2x_l798_798471


namespace min_value_fraction_l798_798212

def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * seq_a (n - 1) + 2

def seq_b (n : ℕ) : ℕ :=
  Int.log2 (seq_a n + 2)

def S (n : ℕ) : ℕ :=
  (List.range n).sum (λ i, seq_b (i + 1))

def min_val (n : ℕ) : ℚ :=
  (S n + 4 : ℚ) / n

theorem min_value_fraction (n : ℕ) : min_val 2 = 9 / 2 := 
by {
  sorry
}

end min_value_fraction_l798_798212


namespace integer_solution_count_l798_798547

theorem integer_solution_count (x : ℝ) : 
  set.count {x | |x - 3| ≤ 7.5 ∧ x ∈ ℤ } = 15 := 
sorry

end integer_solution_count_l798_798547


namespace friendship_configuration_l798_798122

variables (A : Type) [fintype A] [decidable_eq A]

/-- Define a friendship relation in a group of 7 people. The relation is symmetric and irreflexive. -/
def friendship (p q : A) : Prop :=
  sorry -- Further refined definition or properties can be added here

/-- Condition: The friendship relation is symmetric. -/
axiom friendship_symmetric : ∀ x y, friendship x y → friendship y x

/-- Condition: Every individual has the same number of friends. -/
axiom same_number_of_friends :
  ∃ n, ∀ x, (finset.univ.filter (λ y, friendship x y)).card = n

/-- Condition: Each person must have at least two friends. -/
axiom at_least_two_friends :
  ∃ x, (finset.univ.filter (λ y, friendship x y)).card ≥ 2

/-- Prove there are 825 distinct ways to form such a friendship group. -/
theorem friendship_configuration : fintype.card (Σ' (f : A → finset A), (∀ x, (f x).card = (f ⟨A1⟩).card) ∧ (∀ x y, (y ∈ f x ↔ x ∈ f y))) = 825 :=
sorry

end friendship_configuration_l798_798122


namespace john_percentage_increase_l798_798276

/-- The condition of John's initial and new weekly earnings --/
def john_initial_earnings : ℝ := 65

def john_new_earnings : ℝ := 72

/-- Calculate the percentage increase --/
def percentage_increase (initial new : ℝ) : ℝ :=
  ((new - initial) / initial) * 100

/-- The theorem proving the percentage increase in his weekly earnings is 10.77. --/
theorem john_percentage_increase :
  percentage_increase john_initial_earnings john_new_earnings = 10.77 :=
by
  -- Proof would go here
  sorry

end john_percentage_increase_l798_798276


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l798_798509

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem leftmost_three_nonzero_digits_of_ring_arrangements :
  (let m := binom 10 7 * (Nat.factorial 7) * binom 11 4 in m % 1000 / 100 = 199) := sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l798_798509


namespace fence_width_l798_798314

theorem fence_width (L W : ℝ) 
  (circumference_eq : 2 * (L + W) = 30)
  (width_eq : W = 2 * L) : 
  W = 10 :=
by 
  sorry

end fence_width_l798_798314


namespace TV_cost_l798_798629

theorem TV_cost (savings : ℝ) (spent_on_furniture_ratio : ℝ) (original_savings : ℝ) : savings = 960 ∧ spent_on_furniture_ratio = 3 / 4 → (1 - spent_on_furniture_ratio) * savings = 240 :=
by
  intro h,
  cases h with hsavings hspent,
  rw [hsavings, hspent],
  norm_num

#eval TV_cost 960 (3/4) 960  -- Testing the theorem with given values

end TV_cost_l798_798629


namespace object_hits_ground_l798_798339

noncomputable def height (t : ℝ) : ℝ := -8 * t^2 - 16 * t + 48

theorem object_hits_ground : ∃ (t : ℝ), height t = 0 ∧ t = 2.00 :=
by
  -- The proof will be inserted here
  sorry

end object_hits_ground_l798_798339


namespace equation_has_one_real_solution_l798_798160

theorem equation_has_one_real_solution (k : ℚ) :
    (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x ↔ x^2 + 4 * x + (10 - k) = 0) →
    (∃ k : ℚ, (∀ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ↔ by sorry (condition for one real solution is equivalent to discriminant being zero), k = 6) := by
    sorry

end equation_has_one_real_solution_l798_798160


namespace common_points_line_circle_l798_798949

theorem common_points_line_circle :
  ∀ (ρ θ : ℝ), (4 * ρ * real.cos (θ - real.pi / 6) + 1 = 0) →
               (ρ = 2 * real.sin θ) →
               num_intersections 2 = 2 :=
by
  sorry

end common_points_line_circle_l798_798949


namespace percentage_of_women_in_study_group_l798_798765

variable (W : ℝ) -- W is the percentage of women in the study group in decimal form

-- Given conditions as hypotheses
axiom h1 : 0 < W ∧ W <= 1         -- W represents a percentage, so it must be between 0 and 1.
axiom h2 : 0.40 * W = 0.28         -- 40 percent of women are lawyers, and the probability of selecting a woman lawyer is 0.28.

-- The statement to prove
theorem percentage_of_women_in_study_group : W = 0.7 :=
by
  sorry

end percentage_of_women_in_study_group_l798_798765


namespace num_valid_two_digit_numbers_l798_798223

theorem num_valid_two_digit_numbers : 
  ∃ (n : ℕ), n = 14 ∧ 
  ∀ (A B : ℕ), 
  (1 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
  (8 * B ≥ 19 * A) → ∃ valid_numbers : Set (ℕ × ℕ), 
  valid_numbers = { (A, B) | 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 8 * B ≥ 19 * A } ∧ 
  valid_numbers.card = n :=
by
  sorry

end num_valid_two_digit_numbers_l798_798223


namespace cheryl_material_left_l798_798113

def square_yards_left (bought1 bought2 used : ℚ) : ℚ :=
  bought1 + bought2 - used

theorem cheryl_material_left :
  square_yards_left (4/19) (2/13) (0.21052631578947367 : ℚ) = (0.15384615384615385 : ℚ) :=
by
  sorry

end cheryl_material_left_l798_798113


namespace mary_marbles_l798_798306

theorem mary_marbles (total_marbles joan_marbles mary_marbles : ℕ) 
  (h1 : total_marbles = 12) 
  (h2 : joan_marbles = 3) 
  (h3 : total_marbles = joan_marbles + mary_marbles) : 
  mary_marbles = 9 := 
by
  rw [h1, h2, add_comm] at h3
  linarith

end mary_marbles_l798_798306


namespace miki_sandcastle_height_correct_l798_798804

namespace SandcastleHeight

def sister_sandcastle_height := 0.5
def difference_in_height := 0.3333333333333333
def miki_sandcastle_height := sister_sandcastle_height + difference_in_height

theorem miki_sandcastle_height_correct : miki_sandcastle_height = 0.8333333333333333 := by
  unfold miki_sandcastle_height sister_sandcastle_height difference_in_height
  simp
  sorry

end SandcastleHeight

end miki_sandcastle_height_correct_l798_798804


namespace expected_value_decisive_games_l798_798791

theorem expected_value_decisive_games :
  let X : ℕ → ℕ := -- Random variable representing the number of decisive games
    -- Expected value calculation for random variable X
    have h : ∃ e : ℕ, e = (2 * 1/2 + (2 + e) * 1/2), from sorry, 
    -- Extracting the expected value from the equation
    let ⟨E_X, h_ex⟩ := Classical.indefinite_description (λ e, e = (2 * 1/2 + (2 + e) * 1/2)) h in
    E_X = 4 :=
begin
  sorry,
end

end expected_value_decisive_games_l798_798791


namespace parabola_transformation_l798_798565

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the transformed parabola after shifting 2 units to the left and 3 units up
def transformed_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

-- State the theorem to prove the transformation
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 5 * (x + 2)^2 + 3 :=
begin
  sorry
end

end parabola_transformation_l798_798565


namespace sum_terms_l798_798205

def f (x : ℝ) : ℝ := 3 * x^2 + 3 * x - 1

theorem sum_terms : (∑ i in (finset.range 49).filter (λ i, i%2 == 0), (f (-0.02 * (i:ℝ)) / f (-(0.98 - 0.02 * (i:ℝ))))) = 49 := 
by sorry

end sum_terms_l798_798205


namespace one_of_P_Q_constant_l798_798603

noncomputable def is_constant (p : Polynomial ℂ) : Prop := p.degree ≤ 0

theorem one_of_P_Q_constant (f g P Q : Polynomial ℂ) 
  (h_gcd : Int.gcd (Polynomial.natDegree f) (Polynomial.natDegree g) = 1)
  (h_decomp : ∀ (x y : ℂ), Polynomial.eval x f + Polynomial.eval y g = 
                            Polynomial.eval₂ Polynomial.C Polynomial.X P * 
                            Polynomial.eval₂ Polynomial.C Polynomial.X Q) :
  is_constant P ∨ is_constant Q :=
begin
  sorry
end

end one_of_P_Q_constant_l798_798603


namespace cost_price_of_article_l798_798070

theorem cost_price_of_article (C : ℝ) (SP : ℝ) (C_new : ℝ) (SP_new : ℝ) :
  SP = 1.05 * C →
  C_new = 0.95 * C →
  SP_new = SP - 3 →
  SP_new = 1.045 * C →
  C = 600 :=
by
  intro h1 h2 h3 h4
  -- statement to be proved
  sorry

end cost_price_of_article_l798_798070


namespace coefficient_x_neg2_is_7_l798_798560
noncomputable theory

theorem coefficient_x_neg2_is_7
  (a : ℝ)
  (h : ∑ i in Finset.range 8, binomial 7 i * ((-a) ^ i) * (1 : ℝ) ^ (7 - i - i / 2) = 35) 
  : ∑ i in Finset.range 8, binomial 7 i * ((-a) ^ i) * (1 : ℝ) ^ (7 - i - i / 2) = 35 → 
    ∑ j in Finset.range 8, binomial 7 j * ((-a) ^ j) * (1 : ℝ) ^ (7 - j - j / 2) = 7 := 
sorry

end coefficient_x_neg2_is_7_l798_798560


namespace proveEllipseEquationAndRatio_l798_798216

noncomputable section

def ellipseEquation : Prop :=
  let F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
  let P : ℝ × ℝ := (1, Real.sqrt 3 / 2)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ (2 * a = (((P.1 - F₁.1)^2 + P.2^2)^0.5 + ((P.1 - F₂.1)^2 + P.2^2)^0.5)) ∧
    (∃ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ∧ a = 2 ∧ b^2 = 4 - 3)

def ratioOfAreas : Prop :=
  let A : ℝ × ℝ := (0, 2)
  let l : ℝ → ℝ := fun x => k * x + 2
  let O : ℝ × ℝ := (0, 0)
  ∀ k : ℝ, k ≠ 0 →
  ∃ x1 x2 λ : ℝ, 
    (x1 ≠ x2 ∧ (λ = |x1| / |x2| ∧ 0 < λ ∧ λ < 1) ∧ 
    4 < ((1 + λ)^2) / λ ∧ ((1 + λ)^2) / λ < 16 / 3)

theorem proveEllipseEquationAndRatio :
  ellipseEquation ∧ ratioOfAreas :=
by
  constructor
  · -- Prove the equation of the ellipse
    sorry
  · -- Prove the range of the ratio of the areas
    sorry

end proveEllipseEquationAndRatio_l798_798216


namespace xy_product_solution_l798_798749

theorem xy_product_solution (x y : ℝ)
  (h1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (h2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -1 / Real.sqrt 2 :=
sorry

end xy_product_solution_l798_798749


namespace length_of_AC_l798_798173

def triangle_ABC (A B C : ℝ) (AC BC : ℝ) : Prop :=
  A = 6 ∧ C = 75 ∧ BC = Real.sqrt 3

theorem length_of_AC (A C : ℝ) (BC : ℝ) (h : triangle_ABC A B C AC BC) : AC ≈ 0.36 :=
  sorry

end length_of_AC_l798_798173


namespace distinct_solutions_difference_eq_sqrt29_l798_798842

theorem distinct_solutions_difference_eq_sqrt29 :
  (∃ a b : ℝ, a > b ∧
    (∀ x : ℝ, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ 
      x = a ∨ x = b) ∧ 
    a - b = Real.sqrt 29) :=
sorry

end distinct_solutions_difference_eq_sqrt29_l798_798842


namespace part1_part2_l798_798936

variable (A B C : ℝ) (a b c : ℝ)

-- Assume that ABC is an acute triangle with sides a, b, and c opposite to angles A, B, and C respectively.
-- Also assume that a * sin(B) = sqrt(3) * b * cos(A).

-- Part (I): Prove that A = π / 3
theorem part1 (h1 : a * sin B = sqrt(3) * b * cos A) : A = π / 3 :=
  sorry

-- Part (II): Given a = sqrt(21) and b = 5, prove that c = 4
theorem part2 (h1 : a * sin B = sqrt(3) * b * cos A) (h2 : a = sqrt 21)
  (h3 : b = 5) : c = 4 :=
  sorry

end part1_part2_l798_798936


namespace part1_part2_l798_798197

open Set

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a

theorem part1 (a : ℝ) (hpos : 0 < a) :
  (∀ x, A x → ¬ B x a) ↔ a ≥ 5 :=
sorry

theorem part2 (a : ℝ) (hpos : 0 < a) :
  (∀ x, (¬ A x → B x a) ∧ ∃ x, ¬ A x ∧ ¬ B x a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part1_part2_l798_798197


namespace volleyball_club_members_l798_798789

variables (B G : ℝ)

theorem volleyball_club_members (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 18) : B = 12 := by
  -- Mathematical steps and transformations done here to show B = 12
  sorry

end volleyball_club_members_l798_798789


namespace range_of_a_l798_798878

theorem range_of_a (a : ℝ) (p : Prop) (q : Prop) (ha : a > 0) 
  (hp : ∀ x : ℝ, y = a^x → MonotoneOn (λ x, y) ℝ) 
  (hq : ∀ x : ℝ, x^2 - a*x + 1 > 0) 
  (h₁ : ¬(p ∧ q)) 
  (h₂ : ¬¬p) : 
  ∀ ε : ℝ, ε > 0 → ε ≥ a → a ∈ set.Icc 2 (ε) :=
sorry

end range_of_a_l798_798878


namespace oldest_son_cookies_l798_798593

def youngest_son_cookies : Nat := 2
def total_cookies : Nat := 54
def days : Nat := 9

theorem oldest_son_cookies : ∃ x : Nat, 9 * (x + youngest_son_cookies) = total_cookies ∧ x = 4 := by
  sorry

end oldest_son_cookies_l798_798593


namespace subset_result_l798_798627

-- Define A and a as specified in the conditions
def A := {x : ℝ | |x| ≤ 2 * real.sqrt 3}
def a := real.sqrt 11

-- Formulate the proof problem
theorem subset_result : {a} ⊆ A :=
by sorry

end subset_result_l798_798627


namespace n_minus_two_is_square_of_natural_number_l798_798001

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l798_798001


namespace minimize_distance_l798_798189

noncomputable def find_minimizing_B : ℝ × ℝ :=
  let A := (-2, 2) in
  let ellipse := λ (x y : ℝ), (x^2 / 25) + (y^2 / 16) = 1 in
  let F := (-3, 0) in
  let AB (B : ℝ × ℝ) := ((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt in
  let BF (B : ℝ × ℝ) := ((B.1 - F.1)^2 + (B.2 - F.2)^2).sqrt in
  if ellipse (-5 * (3.sqrt) / 2) 2 
  then (-5 * (3.sqrt) / 2, 2)
  else sorry

theorem minimize_distance : find_minimizing_B = (-5 * (3.sqrt) / 2, 2) :=
sorry

end minimize_distance_l798_798189


namespace smallest_real_number_among_given_l798_798092

theorem smallest_real_number_among_given :
  ∀ (a b c d : ℝ), a = 0 → b = -1 → c = -√2 → d = 2 → min {a, b, c, d} = c :=
by
  intros a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end smallest_real_number_among_given_l798_798092


namespace arithmetic_sequence_y_value_l798_798155

theorem arithmetic_sequence_y_value (y : ℝ) (h₁ : 2 * y - 3 = -5 * y + 11) : y = 2 := by
  sorry

end arithmetic_sequence_y_value_l798_798155


namespace largest_distance_between_points_on_spheres_l798_798729

theorem largest_distance_between_points_on_spheres :
  let center1 := (0 : ℝ, -5 : ℝ, 3 : ℝ)
  let radius1 := 23
  let center2 := (13 : ℝ, 15 : ℝ, -20 : ℝ)
  let radius2 := 92
  ∃ A B, (A ∈ sphere center1 radius1) ∧ (B ∈ sphere center2 radius2) ∧ dist A B = 148.1 :=
sorry

end largest_distance_between_points_on_spheres_l798_798729


namespace integer_triangle_equiv_rational_properties_l798_798323

theorem integer_triangle_equiv_rational_properties 
  (a b c : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∃ A B C : ℚ, a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A) ∧ 
                b^2 = a^2 + c^2 - 2 * a * c * (Real.cos B) ∧ 
                c^2 = a^2 + b^2 - 2 * a * b * (Real.cos C)) →
  (∃ S : ℚ, S = (1 / 2) * b * c * Real.sin A) ↔ 
  (∃ r R : ℚ, S = r * Real.sqrt((a + b + c) / 2) ∧ S = (a * b * c) / (4 * R)) ↔ 
  (∃ h : ℚ, h = c * Real.sin B) ↔
  (∃ t : ℚ, t = (1 - Real.cos A) / Real.sin A) :=
sorry

end integer_triangle_equiv_rational_properties_l798_798323


namespace value_expression_l798_798779

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_expression (p q r s t : ℝ) (h : g p q r s t (-3) = 9) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = -9 := 
by
  sorry

end value_expression_l798_798779


namespace total_dolls_48_l798_798541

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l798_798541


namespace floor_sum_is_6200_2_l798_798114

-- Definitions
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + ↑n * d

def floor_sum_arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, Real.floor (arithmetic_seq a d i))

-- Constants for our specific problem
def a : ℝ := 1.5
def d : ℝ := 0.8
def n : ℕ := 124

-- Statement of the problem
theorem floor_sum_is_6200_2 :
  floor_sum_arithmetic_seq a d n = 6200.2 :=
sorry

end floor_sum_is_6200_2_l798_798114


namespace min_value_fraction_l798_798474

theorem min_value_fraction (a b : ℝ) (h : x^2 - 3*x + a*b < 0 ∧ 1 < x ∧ x < 2) (h1 : a > b) : 
  (∃ minValue : ℝ, minValue = 4 ∧ ∀ a b : ℝ, a > b → minValue ≤ (a^2 + b^2) / (a - b)) := 
sorry

end min_value_fraction_l798_798474


namespace fractionD_is_unchanged_l798_798243

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l798_798243


namespace goose_eggs_count_l798_798309

theorem goose_eggs_count (E : ℕ) :
  let hatched := (1/4) * E
  let survived_first_month := (4/5) * hatched
  let survived_second_month := (3/4) * survived_first_month
  let survived_third_month := (7/8) * survived_second_month
  let survived_fourth_month := (3/7) * survived_third_month
  let first_year_survived := (3/5) * (9/10) * survived_fourth_month in
  first_year_survived = 120 → E = 659 :=
by sorry

end goose_eggs_count_l798_798309


namespace remainder_8_pow_215_mod_9_l798_798371

theorem remainder_8_pow_215_mod_9 : (8 ^ 215) % 9 = 8 := by
  -- condition
  have pattern : ∀ n, (8 ^ (2 * n + 1)) % 9 = 8 := by sorry
  -- final proof
  exact pattern 107

end remainder_8_pow_215_mod_9_l798_798371


namespace length_of_DE_is_29_over_2_l798_798014

-- Define the sides of triangle ABC
def AB : ℝ := 30
def AC : ℝ := 31
def BC : ℝ := 29

-- Define the points D and E on AB and AC respectively and DE parallel to BC
-- Define the fact that DE intersects the angle bisector of ∠A
def triangle_ABC (AB AC BC DE : ℝ) (D E : Point) :=
  D ∈ segment A B ∧ E ∈ segment A C ∧ parallel D E B C ∧ intersects (angle_bisector A) D E

-- Prove the length of DE is 29/2
theorem length_of_DE_is_29_over_2 :
  triangle_ABC AB AC BC DE D E →
  DE = 29 / 2 :=
by
  intro h
  sorry

end length_of_DE_is_29_over_2_l798_798014


namespace problem1_problem2_l798_798890

-- Definition of circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Definition of the point P
def P : ℝ × ℝ := (1, 2)

-- Length |AB|
def length_AB (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Line l intersects circle C at A and B such that |AB| = 2√3
theorem problem1 (A B : ℝ × ℝ) (hA : circle_eq A.1 A.2) (hB : circle_eq B.1 B.2)
    (hp : ∃ t : ℝ, A = (1 + t, 2 + t * 0) ∧ B = (1 - t, 2 - t * 0)) :
    length_AB A B = 2 * real.sqrt 3 ↔ 
    (∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0 ∨ x = 1) → circle_eq x y) :=
sorry

-- Moving point M on circle C and corresponding trajectory equation
theorem problem2 (x_0 y_0 : ℝ) (hx₀ : circle_eq x_0 y_0) (hy₀ : y_0 ≠ 0) :
    let Q : ℝ × ℝ := (x_0, 2 * y_0),
        traj_eq := (Q.1^2 / 4) + (Q.2^2 / 16) = 1
    in
    traj_eq :=
sorry

end problem1_problem2_l798_798890


namespace distance_on_dirt_road_l798_798392

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end distance_on_dirt_road_l798_798392


namespace largest_number_l798_798670

def HCF (a b c d : ℕ) : Prop := d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ 
                                ∀ e, (e ∣ a ∧ e ∣ b ∧ e ∣ c) → e ≤ d
def LCM (a b c m : ℕ) : Prop := m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧ 
                                ∀ n, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem largest_number (a b c : ℕ)
  (hcf: HCF a b c 210)
  (lcm_has_factors: ∃ k1 k2 k3, k1 = 11 ∧ k2 = 17 ∧ k3 = 23 ∧
                                LCM a b c (210 * k1 * k2 * k3)) :
  max a (max b c) = 4830 := 
by
  sorry

end largest_number_l798_798670


namespace largest_angle_in_ratio_3_4_5_l798_798350

theorem largest_angle_in_ratio_3_4_5 : ∃ (A B C : ℝ), (A / 3 = B / 4 ∧ B / 4 = C / 5) ∧ (A + B + C = 180) ∧ (C = 75) :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l798_798350


namespace emily_num_dresses_l798_798822

theorem emily_num_dresses (M : ℕ) (D : ℕ) (E : ℕ) 
  (h1 : D = M + 12) 
  (h2 : M = E / 2) 
  (h3 : M + D + E = 44) : 
  E = 16 := 
by 
  sorry

end emily_num_dresses_l798_798822


namespace smallest_n_l798_798292

noncomputable def f (x : ℝ) : ℝ := abs (3 * (x - real.floor x) - 1.5)

theorem smallest_n (n : ℕ) :
  (∃ m : ℕ, m ≥ 2500 ∧ ∀ x : ℝ, x >= 0 → ∃ k : ℕ, 
    k ≤ m ∧ nf (f (x * f x)) = 2 * x) ↔ n = 36 :=
sorry

end smallest_n_l798_798292


namespace exists_disjoint_or_singleton_intersection_l798_798978

open Set

variable {n : ℕ} (S : Finset (Finset (Fin n))) (h1 : 1 < n)
variable (h2 : S.card = (Finset.choose (2 * n) n / 2)) 

theorem exists_disjoint_or_singleton_intersection :
  ∃ (A B : Finset (Fin n)), A ∈ S ∧ B ∈ S ∧ A ≠ B ∧ (A ∩ B).card ≤ 1 := by
  sorry

end exists_disjoint_or_singleton_intersection_l798_798978


namespace find_ratio_of_radii_l798_798818

noncomputable def ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : Prop :=
  a / b = Real.sqrt 5 / 5

theorem find_ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) :
  ratio_of_radii a b h1 :=
sorry

end find_ratio_of_radii_l798_798818


namespace parabola_vertex_l798_798338

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ v : ℝ × ℝ, (v.1 = -1 ∧ v.2 = 4) ∧ ∀ (x : ℝ), (x^2 + 2*x + 5 = ((x + 1)^2 + 4))) :=
by
  sorry

end parabola_vertex_l798_798338


namespace total_profit_l798_798760

-- Define the problem conditions
variables (B_investment X Y : ℝ) (B_profit : ℝ) 
variables (A_investment_period B_investment_period : ℝ)
variable [fact (B_profit = 4000)]
variable [fact (B_investment = X)]
variable [fact (B_investment_period = Y)]
def investment_conditions : Prop :=
  A_investment = 3 * B_investment ∧
  A_investment_period = 2 * B_investment_period ∧
  B_profit = Rs 4000

-- Prove the total profit
theorem total_profit (B_investment X Y : ℝ) (B_profit : ℝ) 
      (A_investment_period B_investment_period : ℝ)
      (h1 : A_investment = 3 * B_investment)
      (h2 : A_investment_period = 2 * B_investment_period)
      (h3 : B_profit = 4000) : 
      total_profit = 28000 :=
by 
    sorry

end total_profit_l798_798760


namespace find_fourth_vertex_of_square_l798_798361

noncomputable def complex_square_fourth_vertex
  (A B C : ℂ) (hA : A = 2 + 3 * complex.i) (hB : B = -1 + 4 * complex.i) (hC : C = -3 - 2 * complex.i) : ℂ :=
  let AB := B - A in
  let BC := C - B in
  let D := C + AB * complex.i in
  D

theorem find_fourth_vertex_of_square (A B C D : ℂ)
  (hA : A = 2 + 3 * complex.i) (hB : B = -1 + 4 * complex.i) (hC : C = -3 - 2 * complex.i)
  (hD : D = complex_square_fourth_vertex A B C hA hB hC) : D = -4 - 5 * complex.i := by
  sorry

end find_fourth_vertex_of_square_l798_798361


namespace simplify_trig_expression_l798_798655

theorem simplify_trig_expression (x : ℝ) :
  (3 + 3 * sin x - 3 * cos x) / (3 + 3 * sin x + 3 * cos x) = tan (x / 2) := 
by
  have h1 : sin x = 2 * sin (x / 2) * cos (x / 2), by sorry,
  have h2 : cos x = 2 * cos (x / 2) ^ 2 - 1, by sorry,
  sorry

end simplify_trig_expression_l798_798655


namespace inclination_angle_correct_l798_798561

-- Define the conditions
def direction_vector : ℝ × ℝ := (-1, real.sqrt 3)

-- Declare the inclination angle
def inclination_angle (v : ℝ × ℝ) : ℝ :=
  real.atan2 v.2 v.1

-- State the theorem we want to prove
theorem inclination_angle_correct :
  inclination_angle direction_vector = 2 * real.pi / 3 :=
by
  sorry

end inclination_angle_correct_l798_798561


namespace product_of_real_and_imaginary_part_l798_798555

noncomputable def real_part_imaginary_part_product : ℂ := 
  (Complex.re (Complex.div (2 + 3 * Complex.i) (1 + Complex.i))) *
  (Complex.im (Complex.div (2 + 3 * Complex.i) (1 + Complex.i)))

theorem product_of_real_and_imaginary_part :
  real_part_imaginary_part_product = 5 / 4 := by
  sorry

end product_of_real_and_imaginary_part_l798_798555


namespace find_number_l798_798921

theorem find_number (x : ℝ) (h : x - x / 3 = x - 24) : x = 72 := 
by 
  sorry

end find_number_l798_798921


namespace necessary_and_sufficient_condition_l798_798206

noncomputable def f (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem necessary_and_sufficient_condition
  {a b c : ℝ}
  (ha_pos : a > 0) :
  ( (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, f a b c x = y } → ∃! x : ℝ, f a b c x = y) ∧ 
    (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, y = f a b c x } → ∃! x : ℝ, f a b c x = y)
  ) ↔
  f a b c (f a b c (-b / (2 * a))) < 0 :=
sorry

end necessary_and_sufficient_condition_l798_798206


namespace parallel_line_slope_l798_798731

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1 / 2 ∧ (∀ x1 y1 : ℝ, 3 * x1 - 6 * y1 = 12 → 
    ∃ k : ℝ, y1 = m * x1 + k) :=
by
  sorry

end parallel_line_slope_l798_798731


namespace min_value_xyz_l798_798297

theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 8) : 
  x + 2 * y + 4 * z ≥ 12 := sorry

end min_value_xyz_l798_798297


namespace greatest_negative_root_l798_798843

noncomputable def sine (x : ℝ) : ℝ := Real.sin (Real.pi * x)
noncomputable def cosine (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x)

theorem greatest_negative_root :
  ∀ (x : ℝ), (x < 0 ∧ (sine x - cosine x) / ((sine x + 1)^2 + (Real.cos (Real.pi * x))^2) = 0) → 
    x ≤ -7/6 :=
by
  sorry

end greatest_negative_root_l798_798843


namespace minimum_discriminant_non_intersecting_regions_l798_798149

noncomputable def discriminant {a b c : ℝ} (h₁ : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
    (h₂ : ∀ x : ℝ, |x| < 1 → (1 / Real.sqrt (1 - x^2)) ≥ a * x^2 + b * x + c) : ℝ :=
b^2 - 4 * a * c

theorem minimum_discriminant_non_intersecting_regions :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ 
  (∀ x : ℝ, |x| < 1 → (1 / Real.sqrt (1 - x^2)) ≥ a * x^2 + b * x + c) ∧ 
  discriminant _ _ = -4 := 
sorry

end minimum_discriminant_non_intersecting_regions_l798_798149


namespace inscribed_square_area_l798_798663

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end inscribed_square_area_l798_798663


namespace minimum_rental_fee_l798_798405

/--
A class of 48 students went on a spring outing. Each small boat can accommodate 3 people and costs 16 yuan to rent. Each large boat can accommodate 5 people and costs 24 yuan to rent.
-/
def students := 48
def small_boat_capacity := 3
def large_boat_capacity := 5
def small_boat_cost := 16
def large_boat_cost := 24

theorem minimum_rental_fee : nat :=
  ∃ n_l n_s, n_l * large_boat_capacity + n_s * small_boat_capacity = students ∧
             n_l * large_boat_cost + n_s * small_boat_cost = 232

end minimum_rental_fee_l798_798405


namespace find_f_f_f_2_l798_798299

-- Define the function f based on the given conditions
def f (x : ℝ) : ℝ :=
  if x > 9 then 1 / x else x ^ 2

-- Statement to prove the final answer
theorem find_f_f_f_2 : f (f (f 2)) = 1 / 16 := 
by {
  -- Using sorry to skip the proof, the lean code will compile
  sorry 
}

end find_f_f_f_2_l798_798299


namespace triangle_division_similar_implies_right_triangle_l798_798235

theorem triangle_division_similar_implies_right_triangle
  (T : Type) [IsTriangle T]
  (h : ∃ T₁ T₂ : Type, IsTriangle T₁ ∧ IsTriangle T₂ ∧ T₁ ≃ T ∧ T₂ ≃ T) :
  IsRightTriangle T :=
by
  sorry

end triangle_division_similar_implies_right_triangle_l798_798235


namespace find_x_and_result_l798_798480

theorem find_x_and_result (x : ℝ) (h1 : sqrt (x + 16) = 12) : x = 128 ∧ (12 ^ 2 - 8) = 136 := by
  sorry

end find_x_and_result_l798_798480


namespace knights_count_l798_798636

def is_correct_circle (R L : ℕ) : Prop :=
  (∀ i : Fin R, (i ∈ Fin R) ∨ (i ∈ Fin L))

def total_natives (R L : ℕ) : Prop :=
  R + L = 2019

def correct_arrangement (R L : ℕ) : Prop :=
  2 * L ≤ R ∧ R ≤ 2 * L + 1

theorem knights_count : ∃ R, ∀ L, is_correct_circle R L ∧ total_natives R L ∧ correct_arrangement R L → 
(R = 1346) :=
by
  sorry

end knights_count_l798_798636


namespace waiter_earnings_l798_798103

theorem waiter_earnings :
  (6 * 3 + 4 * 4.5 + 0) - 10 - 5 = 21 :=
by norm_num

end waiter_earnings_l798_798103


namespace least_common_duration_l798_798009

theorem least_common_duration 
    (P Q R : ℝ) 
    (x : ℝ)
    (T : ℝ)
    (h1 : P / Q = 7 / 5)
    (h2 : Q / R = 5 / 3)
    (h3 : 8 * P / (6 * Q) = 7 / 10)
    (h4 : (6 * 10) * R / (30 * T) = 1)
    : T = 6 :=
by
  sorry

end least_common_duration_l798_798009


namespace max_value_at_l798_798303

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := real.sqrt (4 * x^4 - 3 * x^2 - 10 * x + 26) - real.sqrt (4 * x^4 - 19 * x^2 - 6 * x + 34)

-- The statement proving that f(x) attains its maximum value at x = (-1 + sqrt(23)) / 2 or x = (-1 - sqrt 23) / 2
theorem max_value_at :
  ∃ x : ℝ, (x = (-1 + real.sqrt 23) / 2 ∨ x = (-1 - real.sqrt 23) / 2) ∧ ∀ y : ℝ, f(y) ≤ f(x) :=
sorry

end max_value_at_l798_798303


namespace angle_A_is_60_l798_798589

variables {A B C : Type} 
variables {a b c : ℝ}
variables (triangle_ABC : A ∈ B ⊓ C)

-- Given condition b^2 + c^2 = a^2 + bc
def condition : Prop := b^2 + c^2 = a^2 + bc

-- We now state the theorem that under this condition, angle A = 60 degrees
theorem angle_A_is_60 (h : condition) : A = 60 :=
  sorry

end angle_A_is_60_l798_798589


namespace negation_of_exists_prop_l798_798302

variable (n : ℕ)

theorem negation_of_exists_prop :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_exists_prop_l798_798302


namespace angle_BAC_is_30_l798_798588

-- Define the triangle and its relevant points and angles
def angle (A B C : Type) : Type := sorry  -- Placeholder for angle definition

-- Given conditions as definitions in Lean 4
variables (A B C X Y : Type)
variables (AX XY YC CB : Type)
variable (angle_ABC : ℕ)

-- Assuming axioms for equal line segments and given angle ABC
axiom eq_AX_XY : AX = XY
axiom eq_XY_YC : XY = YC
axiom eq_YC_CB : YC = CB
axiom angle_ABC_150 : angle_ABC = 150

-- Prove the desired angle for BAC
theorem angle_BAC_is_30 : angle A B C = 30 := 
by 
  -- omitted proof steps
  sorry

end angle_BAC_is_30_l798_798588


namespace area_of_triangle_l798_798996

theorem area_of_triangle (XYZ : Type)
  (X Y Z M N : XYZ)
  (XM YN : ℝ)
  (perpendicular : true) -- Assume we have a proof of perpendicularity
  (hXM : XM = 10)
  (hYN : YN = 14) : 
  let S := 93 + 1 / 3 in
  area_of_triangle XYZ X Y Z = S :=
sorry

end area_of_triangle_l798_798996


namespace num_triples_eq_3_l798_798476

noncomputable def num_triples (f : ℝ → ℝ) :=
  {p : ℝ × ℝ × ℝ // 
    let x := p.1, y := p.2.1, z := p.2.2 in
    x = 1000 - 1001 * f (y + z - 1) ∧ 
    y = 1000 - 1001 * f (x + z + 2) ∧ 
    z = 1000 - 1001 * f (x + y - 3)
  }.to_finset.card

theorem num_triples_eq_3 : num_triples sign = 3 :=
sorry

end num_triples_eq_3_l798_798476


namespace max_distance_line_eq_l798_798771

-- Definitions based on the given conditions
def point (x y : ℝ) := (x, y)

def circle_center : point := point 2 0
def point_M : point := point 1 2
def circle_C_eq : (ℝ × ℝ) → Prop := λ p, (p.1 - 2) ^ 2 + p.2 ^ 2 = 9

-- Lean theorem statement
theorem max_distance_line_eq :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ p, l p → p ∈ circle_C_eq) ∧
    l (1,2) ∧
    ∀ (C_l : ℝ), (C_l : ℝ × ℝ) → Prop → 
    0 ≤ C_l ∧
    (C_l $ circle_center) = (λ (p : (ℝ × ℝ)), (p.1 - 2) ^ 2 + (p.2) ^ 2 ≤ (1-2)^2 + (2-0)^2) ∧ 
    l = (λ (p : (ℝ × ℝ)), p.1 - 2 * p.2 + 3 = 0) 
:= sorry

end max_distance_line_eq_l798_798771


namespace unique_abc_solution_l798_798467

theorem unique_abc_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
    (h4 : a^4 + b^2 * c^2 = 16 * a) (h5 : b^4 + c^2 * a^2 = 16 * b) (h6 : c^4 + a^2 * b^2 = 16 * c) : 
    (a, b, c) = (2, 2, 2) :=
  by
    sorry

end unique_abc_solution_l798_798467


namespace can_encode_number_l798_798426

theorem can_encode_number : ∃ (m n : ℕ), (0.07 = 1 / (m : ℝ) + 1 / (n : ℝ)) :=
by
  -- Proof omitted
  sorry

end can_encode_number_l798_798426


namespace find_y_for_two_thirds_l798_798819

theorem find_y_for_two_thirds (x y : ℝ) (h₁ : (2 / 3) * x + y = 10) (h₂ : x = 6) : y = 6 :=
by
  sorry

end find_y_for_two_thirds_l798_798819


namespace intersection_P_Q_correct_l798_798508

-- Define sets P and Q based on given conditions
def is_in_P (x : ℝ) : Prop := x > 1
def is_in_Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the intersection P ∩ Q and the correct answer
def P_inter_Q (x : ℝ) : Prop := is_in_P x ∧ is_in_Q x
def correct_ans (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

-- Prove that P ∩ Q = (1, 2]
theorem intersection_P_Q_correct : ∀ x : ℝ, P_inter_Q x ↔ correct_ans x :=
by sorry

end intersection_P_Q_correct_l798_798508


namespace range_of_f_when_a_eq_2_max_value_implies_a_l798_798527

-- first part
theorem range_of_f_when_a_eq_2 (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 3) :
  (∀ y, (y = x^2 + 3*x - 3) → (y ≥ -21/4 ∧ y ≤ 15)) :=
by sorry

-- second part
theorem max_value_implies_a (a : ℝ) (hx : ∀ x, -1 ≤ x ∧ x ≤ 3 → x^2 + (2*a - 1)*x - 3 ≤ 1) :
  a = -1 ∨ a = -1 / 3 :=
by sorry

end range_of_f_when_a_eq_2_max_value_implies_a_l798_798527


namespace num_mappings_A_to_B_l798_798608

theorem num_mappings_A_to_B (A : set ℕ) (B : set ℕ) (hA : A = {0, 1}) (hB : B = {0, 1, 2}) :
  (A → B).to_finset.card = 9 :=
by
  -- The proof body is intentionally left as a placeholder.
  sorry

end num_mappings_A_to_B_l798_798608


namespace triangle_angle_sum_l798_798772

variables {A B C : Type} [metric_space A] -- assume A represents the points in some metric space

-- Define the angles at vertices A, B, and C
variables (angle_BAC angle_ABC angle_BCA : ℝ)

-- Assume we have a triangle ABC with a line l passing through B parallel to side AC.
variables (l : set (A × A)) (H1 : is_parallel l (segment A C)) -- l is parallel to segment AC

-- The main theorem to prove: sum of angles in triangle is 180 degrees
theorem triangle_angle_sum (Hangle_BAC : angle_BAC = angle_BCA)
  (Hangle_ABC : angle_ABC + angle_BAC + angle_BCA = 180) :
  angle_BAC + angle_ABC + angle_BCA = 180 :=
sorry

end triangle_angle_sum_l798_798772


namespace smallest_m_for_Tn_lt_m_l798_798184

noncomputable def a (n : ℕ) : ℚ := n + 1

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, a (i + 1))

noncomputable def b (n : ℕ) : ℚ := a n / 2^n

noncomputable def T (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b (i + 1))

theorem smallest_m_for_Tn_lt_m : ∃ m : ℚ, (∀ n : ℕ, T n < m) ∧ (∀ m' : ℚ, (∀ n : ℕ, T n < m') → m ≤ m') :=
  by
    use 3
    sorry

end smallest_m_for_Tn_lt_m_l798_798184


namespace geometric_sequence_sum_l798_798499

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) -- a_n is a sequence of real numbers
  (q : ℝ) -- q is the common ratio
  (h1 : a 1 + a 2 = 20) -- first condition
  (h2 : a 3 + a 4 = 80) -- second condition
  (h_geom : ∀ n, a (n + 1) = a n * q) -- property of geometric sequence
  : a 5 + a 6 = 320 := 
sorry

end geometric_sequence_sum_l798_798499


namespace remaining_volume_of_cube_l798_798710

theorem remaining_volume_of_cube 
  (original_edge : ℝ)
  (small_cube_edge : ℝ)
  (face_count : ℕ)
  (original_edge_eq : original_edge = 3)
  (small_cube_edge_eq : small_cube_edge = 1)
  (face_count_eq : face_count = 6) :
  let original_volume := original_edge^3 
  let small_cube_volume := small_cube_edge^3 
  let total_small_cubes_volume := face_count * small_cube_volume 
  let remaining_volume := original_volume - total_small_cubes_volume 
  remaining_volume = 21 := by
sorry

end remaining_volume_of_cube_l798_798710


namespace problem_statement_l798_798343

noncomputable theory
open Real

/-- Define a function f and conditions based on the problem statement. --/
def prop (f : ℝ → ℝ) :=
  (∀ x, differentiable ℝ f) ∧ 
  (∀ x, 1 < x → f(x) + deriv f x < x * deriv f x) ∧ 
  (let a := f 2
   let b := (1 / 2) * f 3
   let c := (sqrt 2 + 1) * f (sqrt 2)
   in c < a ∧ a < b)

/-- The final statement that we need to prove. --/
theorem problem_statement (f : ℝ → ℝ) (h : prop f) : 
  let a := f 2
      b := (1 / 2) * f 3
      c := (sqrt 2 + 1) * f (sqrt 2)
  in c < a ∧ a < b :=
by 
  exact h.2.2

end problem_statement_l798_798343


namespace find_length_AD_l798_798395

-- Define the variables in the problem
variable (A B C D M : Point)
variable (AB BC CD AD AM MC : ℝ)

-- Define the conditions
variable (h1 : BC = 2 * AB)
variable (h2 : CD = 2 * AB)
variable (h3 : M = midpoint A D)
variable (h4 : MC = 8)

-- Define the length of AD
def length_AD := 5 * AB

-- State the theorem
theorem find_length_AD (h1 : BC = 2 * AB) 
                       (h2 : CD = 2 * AB)
                       (h3 : M = midpoint A D)
                       (h4 : MC = 8) :
  length_AD AB = 80 / 3 :=
sorry

end find_length_AD_l798_798395


namespace Joel_Jim_Card_Value_l798_798274

open Real

theorem Joel_Jim_Card_Value :
  ∑ x in {(a : ℝ) | 0 < a ∧ a < π / 3 ∧ 
    by { let sin_x := sin a,
         let cos_x := cos a,
         let tan_x := tan a,
         sin_x = cos_x ∨ sin_x = tan_x ∨ cos_x = tan_x } 
        -- ensuring Jim can uniquely identify the value
       ∧ let sin_x := sin x,
         let cos_x := cos x,
         let tan_x := tan x,
         ¬(tan_x = cos_x) -- Unique id condition for Jim
       }, sin x = (sqrt 5 - 1) / 2 :=
  sorry

end Joel_Jim_Card_Value_l798_798274


namespace C_alpha_closed_under_pointwise_mul_l798_798378

open Set IndicatorFunction Real

noncomputable def C_alpha (α : ℝ) : Set (ℝ → ℝ) :=
  { f | convex_on ℝ (Icc 0 1) f ∧
        (∀ x₁ x₂, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f(x₁) ≤ f(x₂)) ∧
        (f 1 - 2 * f (2 / 3) + f (1 / 3) ≥ α * (f (2 / 3) - 2 * f (1 / 3) + f 0)) }

theorem C_alpha_closed_under_pointwise_mul (α : ℝ) :
  (∀ f g ∈ C_alpha α, (λ x, f x * g x) ∈ C_alpha α) ↔ α ≤ 1 :=
sorry

end C_alpha_closed_under_pointwise_mul_l798_798378


namespace ellipse_eq_max_area_l798_798504

-- Define the conditions as Lean definitions
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0)
def C : set (ℝ × ℝ) := {p | (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1}

noncomputable def e : ℝ := (real.sqrt 3) / 2

-- Given condition of the chord length
def chord_condition : Prop := ∃ c : ℝ, c = a * e / 2 ∧ (c^2 / a^2 + 1 / b^2 = 1)

-- The line equation and the points of intersection:
def line_l (x : ℝ) : ℝ := (real.sqrt 3 / 4) * x + real.sqrt 3 / 2
def point_A : ℝ × ℝ := (2, real.sqrt 3)
def point_B : ℝ × ℝ := (-26 / 7, -3 * real.sqrt 3 / 7)

-- The statement for the equation of the ellipse
theorem ellipse_eq : ∀ (a b : ℝ), a > b → b > 0 → e = real.sqrt 3 / 2 → chord_condition a b → 
  (∀ x y : ℝ, (x, y) ∈ C ↔ x^2 / 16 + y^2 / 4 = 1) :=
by sorry

-- The statement for the maximum area of triangle PAB
theorem max_area : ∀ (P : ℝ × ℝ), P ∈ C → 
  (∀ x y : ℝ, (x, y) ∈ C → x^2 / 16 + y^2 / 4 = 1) →
  abs (P.1 - 2) * abs (P.2 - (3^0.5)) * abs((-26 / 7) - 2) * abs((-3 * real.sqrt 3 / 7) - (3^0.5)) →
  (2 * ((1 - 3 / (27 / 49)) * real.sqrt 19 / 7)) / 
  (real.sqrt ((3 / (16)) + 1)) → 
  (2 * real.sqrt 19 * (2 * real.sqrt 7 + real.sqrt 3)) / 
  19 = 2 * (1 / 7 * (2 * real.sqrt 7 + real.sqrt 3)) :=
by sorry

end ellipse_eq_max_area_l798_798504


namespace min_odd_integers_l798_798008

theorem min_odd_integers :
  ∀ (a b c d e f g h : ℤ),
  a + b + c = 30 →
  a + b + c + d + e + f = 58 →
  a + b + c + d + e + f + g + h = 73 →
  ∃ (odd_count : ℕ), odd_count = 1 :=
by
  sorry

end min_odd_integers_l798_798008


namespace mark_current_trees_l798_798630

theorem mark_current_trees (x : ℕ) (h : x + 12 = 25) : x = 13 :=
by {
  -- proof omitted
  sorry
}

end mark_current_trees_l798_798630


namespace manny_has_more_10_bills_than_mandy_l798_798999

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l798_798999


namespace percent_decrease_l798_798277

theorem percent_decrease (original_price sale_price : ℝ) (h₀ : original_price = 100) (h₁ : sale_price = 30) :
  (original_price - sale_price) / original_price * 100 = 70 :=
by
  rw [h₀, h₁]
  norm_num

end percent_decrease_l798_798277


namespace Q_at_zero_l798_798980

noncomputable def Q (θ : ℝ) : ℝ → ℝ :=
  let z := 2 * Complex.cos θ + Complex.I * 2 * Complex.sin θ
  Polynomial.Cubic (1 : ℝ) (-4 * Real.cos θ) (0 : ℝ) (-a)

theorem Q_at_zero (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) (a : ℝ) (hroots_area : a = 2 * Real.sin(2 * θ) / 4) :
  Q θ 0 = 2 * Real.sin(2 * θ) :=
sorry

end Q_at_zero_l798_798980


namespace germination_percentage_l798_798383

-- Definitions for the problem conditions
def seeds_first_plot : Nat := 300
def seeds_second_plot : Nat := 200
def germination_rate_first_plot : Rational := 0.25
def germination_rate_second_plot : Rational := 0.35

-- Lean statement to prove the final percentage
theorem germination_percentage :
  ((germination_rate_first_plot * seeds_first_plot + germination_rate_second_plot * seeds_second_plot)
  / (seeds_first_plot + seeds_second_plot) * 100) = 29 :=
by
  sorry

end germination_percentage_l798_798383


namespace inverse_undefined_l798_798232

-- Definitions based on the condition
def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- Statement of the problem
theorem inverse_undefined (x : ℝ) : ¬(f⁻¹ x).isDefinedAt x ↔ x = 1 := by
  sorry

end inverse_undefined_l798_798232


namespace exponential_function_passes_through_0_1_l798_798345

theorem exponential_function_passes_through_0_1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (0, 1) ∈ set_of (λ p : ℝ × ℝ, p.2 = a ^ p.1) :=
by
  sorry

end exponential_function_passes_through_0_1_l798_798345


namespace cyclic_quadrilaterals_l798_798678

variable {A B C D O M N : Type}
variable [IsQuadrilateral A B C D]
variable [IsMidpoint O A C]
variable [IsMidpoint M B C]
variable [IsMidpoint N D O]
variable [AngleBisector AC ∠BAD]

theorem cyclic_quadrilaterals :
    CyclicQuadrilateral A B C D ↔ CyclicQuadrilateral A B M N := sorry

end cyclic_quadrilaterals_l798_798678


namespace compute_expression_l798_798116

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l798_798116


namespace min_points_on_dodecahedron_min_points_on_icosahedron_l798_798030

-- Definitions for the dodecahedron problem
def dodecahedron_has_12_faces : Prop := true
def each_vertex_in_dodecahedron_belongs_to_3_faces : Prop := true

-- Proof statement for dodecahedron
theorem min_points_on_dodecahedron : dodecahedron_has_12_faces ∧ each_vertex_in_dodecahedron_belongs_to_3_faces → ∃ n, n = 4 :=
by
  sorry

-- Definitions for the icosahedron problem
def icosahedron_has_20_faces : Prop := true
def icosahedron_has_12_vertices : Prop := true
def each_vertex_in_icosahedron_belongs_to_5_faces : Prop := true
def vertices_of_icosahedron_grouped_into_6_pairs : Prop := true

-- Proof statement for icosahedron
theorem min_points_on_icosahedron : 
  icosahedron_has_20_faces ∧ icosahedron_has_12_vertices ∧ each_vertex_in_icosahedron_belongs_to_5_faces ∧ vertices_of_icosahedron_grouped_into_6_pairs → ∃ n, n = 6 :=
by
  sorry

end min_points_on_dodecahedron_min_points_on_icosahedron_l798_798030


namespace train_crossing_platform_time_l798_798052

theorem train_crossing_platform_time
  (length_train : ℕ) 
  (time_signal : ℕ) 
  (length_platform : ℕ) 
  (speed_train : ℕ := length_train / time_signal) 
  : (length_train = 300) → 
    (time_signal = 18) → 
    (length_platform = 350) → 
    (time_cross_platform : ℕ := (length_train + length_platform) / speed_train) → 
    (time_cross_platform ≈ 39) :=
by
  sorry

end train_crossing_platform_time_l798_798052


namespace numeric_expression_value_l798_798801

theorem numeric_expression_value (A B C : ℕ) (hA : A = 3) (hB : B ≠ C) (hB_case : B = 9) (hC_case : C = 5): (3 * 100 + B * 10 + C = 395) :=
by {
  have h1 : B = 9 := hB_case,
  have h2 : C = 5 := hC_case,
  rw [h1, h2],
  exact eq.refl 395,
}

end numeric_expression_value_l798_798801


namespace julia_tuesday_l798_798972

variable (M : ℕ) -- The number of kids Julia played with on Monday
variable (T : ℕ) -- The number of kids Julia played with on Tuesday

-- Conditions
def condition1 : Prop := M = T + 8
def condition2 : Prop := M = 22

-- Theorem to prove
theorem julia_tuesday : condition1 M T → condition2 M → T = 14 := by
  sorry

end julia_tuesday_l798_798972


namespace find_alpha_and_sin_beta_l798_798909

variable (x α β : ℝ)

def vec_a : ℝ × ℝ := (2 * Real.sin x, Real.sin x + Real.cos x)
def vec_b : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * (Real.sin x - Real.cos x))
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem find_alpha_and_sin_beta
  (hα : 0 < α ∧ α < Real.pi / 2)
  (h1 : f (α / 2) = -1)
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h2 : Real.cos (α + β) = -1 / 3) :
  α = Real.pi / 6 ∧ Real.sin β = (2 * Real.sqrt 6 + 1) / 6 :=
sorry

end find_alpha_and_sin_beta_l798_798909


namespace sum_of_areas_of_triangles_in_cube_l798_798810

def triangle_area (a b : ℝ) : ℝ :=
  1 / 2 * a * b

def count_triangle_type1 (num_cubes : ℕ) : ℕ :=
  num_cubes * 12

def count_triangle_type2 (num_cubes : ℕ) : ℕ :=
  num_cubes * 24

def total_area_triangles (num_cubes : ℕ) : ℝ :=
  let area_type1 := (count_triangle_type1 num_cubes) * (triangle_area 1 1)
  let area_type2 := (count_triangle_type2 num_cubes) * (triangle_area 1 (Real.sqrt 2))
  area_type1 + area_type2

theorem sum_of_areas_of_triangles_in_cube :
  total_area_triangles 8 = 48 + 96 * Real.sqrt 2 :=
by
  sorry

end sum_of_areas_of_triangles_in_cube_l798_798810


namespace simplify_expression_l798_798033

theorem simplify_expression: ∀ (a m n : ℤ),
  (n % 2 = 1) → 
  (m > n) → 
  ((-a)^n = -a^n) → 
  (a^m / a^n = a^(m-n)) → 
  (-5)^5 / 5^3 + 3^4 - 6 = 50 :=
by
  intros a m n hn hm hneg hdiv
  sorry

end simplify_expression_l798_798033


namespace sum_of_inverses_l798_798344

noncomputable def f : ℝ → ℝ :=
λ x, if x < 4 then x - 2 else real.sqrt x

noncomputable def f_inv : ℝ → ℝ :=
λ y, if y < 2 then y + 2 else y^2

theorem sum_of_inverses :
  (∑ i in finset.range 11, f_inv (i - 5)) = 54 :=
by {
  sorry
}

end sum_of_inverses_l798_798344


namespace increasing_interval_l798_798518

noncomputable def f (x : ℝ) := Real.log x / Real.log (1 / 2)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def h (x : ℝ) : ℝ := x^2 + x - 2

theorem increasing_interval :
  is_monotonically_increasing (f ∘ h) {x : ℝ | x < -2} :=
sorry

end increasing_interval_l798_798518


namespace least_integer_with_nine_factors_eq_256_l798_798369

theorem least_integer_with_nine_factors_eq_256 :
  ∃ n : ℕ, (9 = (nat.divisors n).length) ∧ (∀ m : ℕ, (9 = (nat.divisors m).length) → n ≤ m) ∧ n = 256 :=
sorry

end least_integer_with_nine_factors_eq_256_l798_798369


namespace inscribed_tetrahedron_volume_l798_798868

theorem inscribed_tetrahedron_volume (r : ℝ) (V : ℝ) : r = 6 → V = 64 * Real.sqrt 3 :=
by
  intros hr
  rw hr
  sorry

end inscribed_tetrahedron_volume_l798_798868


namespace maximize_h_at_1_l798_798682

-- Definitions and conditions
def f (x : ℝ) : ℝ := -2 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 6
def h (x : ℝ) : ℝ := f x * g x

-- The theorem to prove
theorem maximize_h_at_1 : (∀ x : ℝ, h x <= h 1) :=
sorry

end maximize_h_at_1_l798_798682


namespace polygon_is_hexagon_l798_798074

-- Definitions
def side_length : ℝ := 8
def perimeter : ℝ := 48

-- The main theorem to prove
theorem polygon_is_hexagon : (perimeter / side_length = 6) ∧ (48 / 8 = 6) := 
by
  sorry

end polygon_is_hexagon_l798_798074


namespace find_sequences_l798_798687

namespace SequenceProof

-- Define the conditions
variables (a d : ℕ) (n : ℕ)
def arith_seq : list ℕ := [a, a + d, a + 4d, a + (n - 1) * d]
def geo_seq : list ℕ := [a, a + d, a + 4d, a + (n - 1) * d]

-- Arithmetic sequence terms forming geometric sequence condition
axiom geo_cond : (a + d) ^ 2 = a * (a + 4d)

-- Sum of the terms condition
def seq_sum (lst : list ℕ) : ℕ := lst.sum
axiom sum_condition : seq_sum arith_seq = 80

-- Claim: Prove both sequences are as given
theorem find_sequences : arith_seq = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54] ∧
                         geo_seq = [2, 6, 18, 54] :=
by
  sorry

end SequenceProof

end find_sequences_l798_798687


namespace find_t_l798_798536

-- Definition of the problem's conditions
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Condition of orthogonality
def orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The property we need to prove
theorem find_t (t : ℝ) : orthogonal a (t * (a.1, a.2) + b) ↔ t = -5 := by
  sorry

end find_t_l798_798536


namespace value_of_v_over_u_l798_798042

variable (u v : ℝ) 

theorem value_of_v_over_u (h : u - v = (u + v) / 2) : v / u = 1 / 3 :=
by
  sorry

end value_of_v_over_u_l798_798042


namespace large_box_total_chocolate_bars_l798_798411

def number_of_small_boxes : ℕ := 15
def chocolate_bars_per_small_box : ℕ := 20
def total_chocolate_bars (n : ℕ) (m : ℕ) : ℕ := n * m

theorem large_box_total_chocolate_bars :
  total_chocolate_bars number_of_small_boxes chocolate_bars_per_small_box = 300 :=
by
  sorry

end large_box_total_chocolate_bars_l798_798411


namespace solve_for_x_l798_798758

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 :=
sorry

end solve_for_x_l798_798758


namespace collinear_points_l798_798617

open EuclideanGeometry

variables {A B C B' A' P O D E O' : Point}

/-- Given an acute-angled triangle ABC, points B' and A' on the perpendicular bisectors of AC 
    and BC such that B'A is perpendicular to AB and A'B is perpendicular to AB, P on segment AB,
    O the circumcenter of triangle ABC, D and E on BC and AC respectively such that 
    DP is perpendicular to BO and EP is perpendicular to AO, O' the circumcenter of triangle CDE. 
    Prove that B', A', and O' are collinear. -/
theorem collinear_points 
  (hABC : acute_angle_triangle A B C)
  (hB' : ∃ M, perpendicular_bisector M C A B')
  (hA' : ∃ M, perpendicular_bisector M C B A')
  (hBA_perpendicular : perp B' A B)
  (hAB_perpendicular : perp A' B A)
  (hP_on_segment : on_segment P A B)
  (hO_circumcenter : circumcenter O A B C)
  (hD_on_BC : on_line D B C)
  (hE_on_AC : on_line E A C)
  (hDP_perpendicular : perp D P (line_through B O))
  (hEP_perpendicular : perp E P (line_through A O))
  (hO'_circumcenter : circumcenter O' C D E) :
  collinear {B', A', O'} :=
by {
  sorry
}

end collinear_points_l798_798617


namespace total_doll_count_l798_798539

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l798_798539


namespace sin_double_angle_identity_l798_798192

theorem sin_double_angle_identity (x : ℝ) (h : sin (π / 4 - x) = 1 / 4) : sin (2 * x) = 7 / 8 :=
sorry

end sin_double_angle_identity_l798_798192


namespace individual_contributions_correct_l798_798941

def activity_cost_A : ℕ := 90
def activity_cost_B : ℕ := 120
def activity_cost_C : ℕ := 150

def miss_evans_funds_A : ℕ := 14
def miss_evans_funds_B : ℕ := 28
def miss_evans_students : ℕ := 19

def mr_smith_funds_A : ℕ := 20
def mr_smith_funds_B : ℕ := 20
def mr_smith_funds_C : ℕ := 40
def mr_smith_students : ℕ := 15

def mrs_johnson_funds_A : ℕ := 30
def mrs_johnson_funds_B : ℕ := 45
def mrs_johnson_funds_C : ℕ := 60
def mrs_johnson_students : ℕ := 25

theorem individual_contributions_correct :
  (activity_cost_A - miss_evans_funds_A) / miss_evans_students = 4.00 ∧
  (activity_cost_B - miss_evans_funds_B) / miss_evans_students = 4.84 ∧
  activity_cost_C / miss_evans_students = 7.89 ∧

  (activity_cost_A - mr_smith_funds_A) / mr_smith_students = 4.67 ∧
  (activity_cost_B - mr_smith_funds_B) / mr_smith_students = 6.67 ∧
  (activity_cost_C - mr_smith_funds_C) / mr_smith_students = 7.33 ∧

  (activity_cost_A - mrs_johnson_funds_A) / mrs_johnson_students = 2.40 ∧
  (activity_cost_B - mrs_johnson_funds_B) / mrs_johnson_students = 3.00 ∧
  (activity_cost_C - mrs_johnson_funds_C) / mrs_johnson_students = 3.60 := 
by
  sorry

end individual_contributions_correct_l798_798941


namespace hexagon_inequality_l798_798599

variables (A B C D E F G H : Type) 
variables [metric_space A] [metric_space B] [metric_space C]
          [metric_space D] [metric_space E] [metric_space F]
          [metric_space G] [metric_space H]
variables [convex_space A B C D E F G H]

variables (AB BC CD DE EF FA BCD EFA AGB DHE : ℝ)
variables (A B C D E F G H : point)
variables (h1 : distance A B = distance B C)
variables (h2 : distance B C = distance C D)
variables (h3 : distance D E = distance E F)
variables (h4 : distance E F = distance F A)
variables (h5 : angle B C D = π / 3)
variables (h6 : angle E F A = π / 3)
variables (h7 : angle A G B = 2 * π / 3)
variables (h8 : angle D H E = 2 * π / 3)

theorem hexagon_inequality :
  distance A G + distance G B + distance G H + distance H D + distance H E ≥ distance C F :=
sorry

end hexagon_inequality_l798_798599


namespace tangent_line_l798_798829

variable {o1 o2 : Type} [circle o1] [circle o2]
variable (I1 I2 A1 A2 : Point)
variable (C B1 B2 : Point)
variable (k : Line)

-- Conditions derived from the problem statement
variable (h1 : Disjoint o1 o2) -- circles are disjoint
variable (h2 : TangentAt o1 k A1) -- o1 tangent to line k at A1
variable (h3 : TangentAt o2 k A2) -- o2 tangent to line k at A2
variable (h4 : SameSide o1 o2 k) -- o1 and o2 lie on the same side of line k
variable (h5 : OnSegment I1 C I2) -- C lies on the segment I1I2
variable (h6 : Angle A1 C A2 = 90) -- angle A1CA2 equals 90 degrees
variable (h7 : SecondIntersection A1 C o1 B1) -- B1 is the second intersection of A1C with o1
variable (h8 : SecondIntersection A2 C o2 B2) -- B2 is the second intersection of A2C with o2

-- Theorem to prove
theorem tangent_line (h1 : Disjoint o1 o2) (h2 : TangentAt o1 k A1) 
  (h3 : TangentAt o2 k A2) (h4 : SameSide o1 o2 k) 
  (h5 : OnSegment I1 C I2) (h6 : Angle A1 C A2 = 90) 
  (h7 : SecondIntersection A1 C o1 B1) 
  (h8 : SecondIntersection A2 C o2 B2) : 
  TangentLine B1 B2 o1 ∧ TangentLine B1 B2 o2 := 
sorry

end tangent_line_l798_798829


namespace range_of_m_l798_798240

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| > 4) ↔ m > 3 ∨ m < -5 := 
sorry

end range_of_m_l798_798240


namespace total_sides_of_cookie_cutters_l798_798834

theorem total_sides_of_cookie_cutters :
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  total_sides = 75 :=
by
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  show total_sides = 75
  sorry

end total_sides_of_cookie_cutters_l798_798834


namespace find_value_l798_798041

theorem find_value : (100 + (20 / 90)) * 90 = 120 := by
  sorry

end find_value_l798_798041


namespace dot_product_eq_six_l798_798907

variables (a b : Vector ℝ) -- Declare a and b as vectors over the reals

def projection (a b : Vector ℝ) : ℝ := (a.dot b) / b.norm -- Define the projection of a onto b

axiom projection_eq_three : projection a b = 3 -- Given condition that projection of a in direction of b is 3
axiom b_norm_eq_two : b.norm = 2 -- Given condition that the norm of b is 2

theorem dot_product_eq_six : a.dot b = 6 := sorry

end dot_product_eq_six_l798_798907


namespace range_of_x_l798_798237

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x else Real.log (x + 1)

theorem range_of_x (x : ℝ) : f (2 - x^2) > f x ↔ (-2 < x ∧ x < 1) :=
by sorry

end range_of_x_l798_798237


namespace triangle_side_lengths_exist_l798_798186

theorem triangle_side_lengths_exist :
  ∃ (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ abc = 2 * (a - 1) * (b - 1) * (c - 1) ∧
  ((a, b, c) = (8, 7, 3) ∨ (a, b, c) = (6, 5, 4)) :=
by sorry

end triangle_side_lengths_exist_l798_798186


namespace smallest_n_satisfying_inequality_l798_798115

open Real

noncomputable def log3 (x : ℝ) : ℝ := log x / log 3

theorem smallest_n_satisfying_inequality :
  ∀ n : ℕ, (∑ k in finset.range (n + 1), log3 (1 + 1 / 3 ^ (3 ^ k))) ≥ 
            1 + log3 (5000 / 5001) → n ≥ 2 :=
sorry

end smallest_n_satisfying_inequality_l798_798115


namespace higher_probability_blue_white_l798_798077

-- Conditions
def isRectangle (l w : ℝ) := l > 0 ∧ w > 0 ∧ l ≠ w

-- Correct answer
def larger_angle_probability (l w : ℝ) (regions : Set ℝ) : Prop :=
  ∀ θ ∈ regions, (larger_angles θ → higher_probability θ)

theorem higher_probability_blue_white (l w : ℝ) (regions : Set ℝ)
  (conditions : isRectangle l w) : larger_angle_probability l w regions := sorry

end higher_probability_blue_white_l798_798077


namespace correct_equation_for_growth_rate_l798_798132

def initial_price : ℝ := 6.2
def final_price : ℝ := 8.9
def growth_rate (x : ℝ) : ℝ := initial_price * (1 + x) ^ 2

theorem correct_equation_for_growth_rate (x : ℝ) : growth_rate x = final_price ↔ initial_price * (1 + x) ^ 2 = 8.9 :=
by sorry

end correct_equation_for_growth_rate_l798_798132


namespace congruent_medians_l798_798025

universe u

def triangle (α : Type u) := (α × α) × (α × α) × (α × α)

def midpoint {α : Type u} [Ring α] (A B : α × α) : α × α :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem congruent_medians {α : Type u} [Ring α] 
  (A B C A' B' C' : α × α)
  (h1 : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (B'.1 - A'.1)^2 + (B'.2 - A'.2)^2)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (C'.1 - B'.1)^2 + (C'.2 - B'.2)^2)
  (h3 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C'.1 - A'.1)^2 + (C'.2 - A'.2)^2) :
  (midpoint B C).fst = (midpoint B' C').fst ∧ 
  (midpoint B C).snd = (midpoint B' C').snd :=
sorry

end congruent_medians_l798_798025


namespace possible_k_values_l798_798667

variables (p q r s k : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
          (h5 : p * q = r * s)
          (h6 : p * k ^ 3 + q * k ^ 2 + r * k + s = 0)
          (h7 : q * k ^ 3 + r * k ^ 2 + s * k + p = 0)

noncomputable def roots_of_unity := {k : ℂ | k ^ 4 = 1}

theorem possible_k_values : k ∈ roots_of_unity :=
by {
  sorry
}

end possible_k_values_l798_798667


namespace triangle_ef_sum_l798_798264

noncomputable def triangle_length_ef (D E F : ℝ) (angleD : ℝ) (sideDE : ℝ) (sideDF : ℝ) :=
  let sin45 := Real.sin (45 * Real.pi / 180)
  let angleF1 := 30 * Real.pi / 180
  let angleF2 := 150 * Real.pi / 180
  let sinF1 := Real.sin angleF1
  let sinF2 := Real.sin angleF2
  let cos105 := -Real.sin (15 * Real.pi / 180)
  let sideEF1 := Real.sqrt (100^2 + (100 * Real.sqrt 2)^2 - 2 * 100 * (100 * Real.sqrt 2) * cos105)
  let sideEF2 := 0 -- invalid case flag
  if (angleF1 + angleD) < 180 / 180 * Real.pi then sideEF1
  else if (angleF2 + angleD) < 180 / 180 * Real.pi then sideEF2
  else 0

theorem triangle_ef_sum :
  triangle_length_ef 0 0 0 (45 * Real.pi / 180) 100 (100 * Real.sqrt 2) = 201 :=
sorry

end triangle_ef_sum_l798_798264


namespace average_price_blankets_l798_798774

theorem average_price_blankets :
  let cost_blankets1 := 3 * 100
  let cost_blankets2 := 5 * 150
  let cost_blankets3 := 550
  let total_cost := cost_blankets1 + cost_blankets2 + cost_blankets3
  let total_blankets := 3 + 5 + 2
  total_cost / total_blankets = 160 :=
by
  sorry

end average_price_blankets_l798_798774


namespace johns_out_of_pocket_expense_l798_798968

theorem johns_out_of_pocket_expense
  (computer_cost : ℕ)
  (accessories_cost : ℕ)
  (playstation_value : ℕ)
  (playstation_sold_percent_less : ℕ) :
  computer_cost = 700 →
  accessories_cost = 200 →
  playstation_value = 400 →
  playstation_sold_percent_less = 20 →
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100) in
  let total_cost := computer_cost + accessories_cost in
  let pocket_expense := total_cost - playstation_sold_price in
  pocket_expense = 580 :=
by
  intros h1 h2 h3 h4
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100)
  let total_cost := computer_cost + accessories_cost
  let pocket_expense := total_cost - playstation_sold_price
  sorry

end johns_out_of_pocket_expense_l798_798968


namespace percent_psychology_majors_l798_798437

theorem percent_psychology_majors
  (total_students : ℝ)
  (pct_freshmen : ℝ)
  (pct_freshmen_liberal_arts : ℝ)
  (pct_freshmen_psychology_majors : ℝ)
  (h1 : pct_freshmen = 0.6)
  (h2 : pct_freshmen_liberal_arts = 0.4)
  (h3 : pct_freshmen_psychology_majors = 0.048)
  :
  (pct_freshmen_psychology_majors / (pct_freshmen * pct_freshmen_liberal_arts)) * 100 = 20 := 
by
  sorry

end percent_psychology_majors_l798_798437


namespace expanded_product_terms_count_five_digit_numbers_count_seating_arrangements_count_odd_coefficients_count_l798_798200

-- Statement for the first proposition
theorem expanded_product_terms_count : 
  (expand_form_term_count (sum4 a b c d) (sum3 p q r) (sum2 m n)) = 24 :=
  sorry

-- Statement for the second proposition
theorem five_digit_numbers_count : 
  (count_five_digit_numbers_without_1_2_adj_to_5 [1, 2, 3, 4, 5]) = 36 :=
  sorry

-- Statement for the third proposition
theorem seating_arrangements_count : 
  (count_seating_arrangements_3_people_8_seats_empty_side 3 8) = 24 :=
  sorry

-- Statement for the fourth proposition
theorem odd_coefficients_count : 
  (count_odd_coefficients_binomial_expansion (1 + x)^8) = 2 :=
  sorry

end expanded_product_terms_count_five_digit_numbers_count_seating_arrangements_count_odd_coefficients_count_l798_798200


namespace apple_trees_count_l798_798260

-- Conditions
def num_peach_trees : ℕ := 45
def kg_per_peach_tree : ℕ := 65
def total_mass_fruit : ℕ := 7425
def kg_per_apple_tree : ℕ := 150
variable (A : ℕ)

-- Proof goal
theorem apple_trees_count (h : A * kg_per_apple_tree + num_peach_trees * kg_per_peach_tree = total_mass_fruit) : A = 30 := 
sorry

end apple_trees_count_l798_798260


namespace min_value_of_ratio_l798_798181

noncomputable def quadratic_function_min_value 
    (a b c : ℝ) 
    (h_a : a ≠ 0) 
    (f : ℝ → ℝ := λ x, a * x^2 + b * x + c)
    (f' : ℝ → ℝ := λ x, 2 * a * x + b)
    (h_f0_pos : (f' 0) > 0)
    (h_f_nonneg : ∀ x : ℝ, f x ≥ 0) 
    : ℝ :=
2

-- Required statement to be proven
theorem min_value_of_ratio (a b c : ℝ) 
    (h_a : a ≠ 0) 
    (f : ℝ → ℝ := λ x, a * x^2 + b * x + c)
    (f' : ℝ → ℝ := λ x, 2 * a * x + b)
    (h_f0_pos : (f' 0) > 0)
    (h_f_nonneg : ∀ x : ℝ, f x ≥ 0) 
    : (f 1) / (f' 0) ≥ 2 :=
by {
    sorry
}

end min_value_of_ratio_l798_798181


namespace ratio_CP_CQ_eq_DP_DQ_l798_798100

-- Definitions for circles, points, and tangents.
-- Circle O1 and Circle O2 intersect at points P and Q.
-- One external tangents touches circles at points A and B respectively.
-- Circle Gamma passes through points A and B, and intersects circles O1 and O2 at points D and C respectively.

variables (O1 O2 Γ : Type) [circle O1] [circle O2] [circle Γ]
variables (P Q A B C D : point)
variables (point_on_circle : point → circle → Prop)
variables (tangent_to_circle : point → circle → Prop)
variables (passes_through : circle → point → point → Prop)

-- Assumptions aligned with conditions in step a)
axiom O1_intersects_O2_at_P_and_Q : point_on_circle P O1 ∧ point_on_circle P O2 ∧ point_on_circle Q O1 ∧ point_on_circle Q O2
axiom tangent_touch_A : tangent_to_circle A O1
axiom tangent_touch_B : tangent_to_circle B O2
axiom Gamma_passes_through_A_and_B : passes_through Γ A B
axiom Gamma_intersects_O1_at_D : point_on_circle D O1 ∧ point_on_circle D Γ
axiom Gamma_intersects_O2_at_C : point_on_circle C O2 ∧ point_on_circle C Γ

-- The theorem to be proven
theorem ratio_CP_CQ_eq_DP_DQ :
  (C P) / (C Q) = (D P) / (D Q) :=
sorry

end ratio_CP_CQ_eq_DP_DQ_l798_798100


namespace range_of_a_l798_798500

noncomputable def f : ℝ → ℝ → ℝ :=
λ x a, if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (h1 : f (1 - a) a ≥ f (1 + a) a) : -2 ≤ a ∧ a ≤ -1 :=
sorry

end range_of_a_l798_798500


namespace eighteenth_entry_is_43_l798_798849

def r_11 (x : ℕ) : ℕ := x % 11

def satisfies_condition (n : ℕ) : Prop := r_11 (7 * n) ≤ 5

noncomputable def nth_nonnegative (k : ℕ) : ℕ :=
  (List.filter satisfies_condition (List.range (100*k))).nth k

theorem eighteenth_entry_is_43 : nth_nonnegative 18 = some 43 :=
by
  -- Assuming the proof is skipped.
  sorry

end eighteenth_entry_is_43_l798_798849


namespace mod_inverse_13_2000_l798_798031

-- Define the necessary conditions and statements
theorem mod_inverse_13_2000 : ∃ x : ℤ, 0 ≤ x ∧ x < 2000 ∧ (13 * x ≡ 1 [MOD 2000]) := by
  use 1077
  sorry

end mod_inverse_13_2000_l798_798031


namespace binder_cost_l798_798305

variable (B : ℕ) -- Define B as the cost of each binder

theorem binder_cost :
  let book_cost := 16
  let num_binders := 3
  let notebook_cost := 1
  let num_notebooks := 6
  let total_cost := 28
  (book_cost + num_binders * B + num_notebooks * notebook_cost = total_cost) → (B = 2) :=
by
  sorry

end binder_cost_l798_798305


namespace gcd_subtraction_method_count_l798_798023

theorem gcd_subtraction_method_count : 
  let mutual_subtraction_count : ℕ := 
    let a := 98
    let b := 63
    let rec count_subtractions (x y : ℕ) (count : ℕ) : ℕ :=
      if x = y then count
      else if x > y then count_subtractions (x - y) y (count + 1)
      else count_subtractions x (y - x) (count + 1)
    count_subtractions a b 0
  in
  mutual_subtraction_count = 6 :=
by
  -- Filling in the proof steps as indicated by steps 1-6 from the solution
  sorry

end gcd_subtraction_method_count_l798_798023


namespace james_sales_l798_798956

noncomputable def total_items_sold
  (houses_day1 : ℕ) (items_per_house_day1 : ℕ) (factor_day2 : ℕ) (percentage_sold_day2 : ℚ) (items_per_house_day2 : ℕ) : ℕ :=
 let sold_day1 := houses_day1 * items_per_house_day1 in
 let houses_day2 := houses_day1 * factor_day2 in
 let sold_day2 := (houses_day2 * percentage_sold_day2).toInt * items_per_house_day2 in
 sold_day1 + sold_day2

theorem james_sales : total_items_sold 20 2 2 0.8 2 = 104 :=
  by
    dsimp [total_items_sold]
    -- The calculation steps would be carried here if we were doing the proof.
    sorry

end james_sales_l798_798956


namespace andy_time_difference_l798_798436

def time_dawn : ℕ := 20
def time_andy : ℕ := 46
def double_time_dawn : ℕ := 2 * time_dawn

theorem andy_time_difference :
  time_andy - double_time_dawn = 6 := by
  sorry

end andy_time_difference_l798_798436


namespace min_value_sin_function_l798_798194

theorem min_value_sin_function (α β : ℝ) (h : -5 * (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 3 * Real.sin α) :
  ∃ x : ℝ, x = Real.sin α ∧ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 0 :=
sorry

end min_value_sin_function_l798_798194


namespace modulus_of_z_l798_798177

theorem modulus_of_z (z : ℂ) (hz : z * (1 + 2 * complex.I) = 4 + 3 * complex.I) : abs z = real.sqrt 5 :=
sorry

end modulus_of_z_l798_798177


namespace round_to_nearest_hundredth_l798_798725

-- Define the real number to be rounded
def num : ℝ := 1.895

-- State the problem: rounding num to the nearest hundredth should be 1.90
theorem round_to_nearest_hundredth : Float.roundTo 2 num = 1.90 :=
by
  sorry

end round_to_nearest_hundredth_l798_798725


namespace scientific_notation_of_6390000_l798_798012

theorem scientific_notation_of_6390000 : 6_390_000 = 6.39 * 10^6 := 
  sorry

end scientific_notation_of_6390000_l798_798012


namespace speed_of_second_train_40_kmph_l798_798021

noncomputable def length_train_1 : ℝ := 140
noncomputable def length_train_2 : ℝ := 160
noncomputable def crossing_time : ℝ := 10.799136069114471
noncomputable def speed_train_1 : ℝ := 60

theorem speed_of_second_train_40_kmph :
  let total_distance := length_train_1 + length_train_2
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  let speed_train_2 := relative_speed_kmph - speed_train_1
  speed_train_2 = 40 :=
by
  sorry

end speed_of_second_train_40_kmph_l798_798021


namespace twelve_women_reseated_l798_798016

def S (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 3
  else S (n - 1) + S (n - 2) + S (n - 3)

theorem twelve_women_reseated : S 12 = 1201 :=
by
  sorry

end twelve_women_reseated_l798_798016


namespace cube_root_of_27_l798_798107

theorem cube_root_of_27 : real.cbrt 27 = 3 :=
by
  sorry

end cube_root_of_27_l798_798107


namespace rows_remain_sorted_l798_798697

theorem rows_remain_sorted (M : Matrix (Fin 5) (Fin 10) ℕ) 
  (hM_elem : ∀ i j, M i j ∈ (Finset.range 50).erase 0) 
  (hr : ∀ i, Sorted (fun x y => x < y) (Matrix.row M i)) 
  (hc : ∀ j, Sorted (fun x y => x < y) (Matrix.col M j)) :
  ∀ i, Sorted (fun x y => x < y) (Matrix.row M i) := 
by 
  sorry

end rows_remain_sorted_l798_798697


namespace least_woogles_for_more_points_l798_798586

def drop_points (n : ℕ) : ℕ := (n * (n + 1)) / 2
def eat_points (n : ℕ) : ℕ := 15 * n

theorem least_woogles_for_more_points (n : ℕ) : n = 30 → (drop_points n > eat_points n) :=
by
  intro h
  rw h
  sorry

end least_woogles_for_more_points_l798_798586


namespace scalar_product_of_vectors_l798_798182

noncomputable def regular_tetrahedron.dot_product : ℝ :=
  let O := (0 : EuclideanSpace ℝ (Fin 3))
  let A := (1 : EuclideanSpace ℝ (Fin 3))
  let B := (2 : EuclideanSpace ℝ (Fin 3))
  let C := (3 : EuclideanSpace ℝ (Fin 3))
  let D := (A + B) / 2
  let E := (O + C) / 2
  let DE := (E - D)
  let AC := (C - A)
  (DE ⬝ AC)
  
theorem scalar_product_of_vectors : regular_tetrahedron.dot_product = 7 / 6 := sorry

end scalar_product_of_vectors_l798_798182


namespace solution_set_f_greater_4_l798_798893

def f (x : ℝ) : ℝ := if x < 0 then 2 * Real.exp x else Real.log (x + 1) / Real.log 2 + 2

theorem solution_set_f_greater_4 :
  {x : ℝ | f x > 4} = {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_f_greater_4_l798_798893


namespace star_calculation_l798_798454

def star (X Y : ℝ) : ℝ := (X + Y) / 4

theorem star_calculation : star (star 3 7) 6 = 2.125 :=
by
  sorry

end star_calculation_l798_798454


namespace value_of_a_minus_n_plus_k_l798_798397

theorem value_of_a_minus_n_plus_k :
  ∃ (a k n : ℤ), 
    (∀ x : ℤ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) ∧ 
    (a - n + k = 3) :=
sorry

end value_of_a_minus_n_plus_k_l798_798397


namespace total_morning_afternoon_emails_l798_798592

-- Define the conditions
def morning_emails : ℕ := 5
def afternoon_emails : ℕ := 8
def evening_emails : ℕ := 72

-- State the proof problem
theorem total_morning_afternoon_emails : 
  morning_emails + afternoon_emails = 13 := by
  sorry

end total_morning_afternoon_emails_l798_798592


namespace intersection_point_polar_coordinates_l798_798261

noncomputable def polar_coordinates_intersection_point : (ℝ × ℝ) :=
let θ := Real.pi * 3 / 4 in
let ρ := Real.sqrt 2 in
(ρ, θ)

theorem intersection_point_polar_coordinates :
  ∃ θ ρ, (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ ρ = 2 * Real.sin θ ∧ ρ * Real.cos θ = -1 ∧ (ρ = Real.sqrt 2 ∧ θ = (3 * Real.pi / 4)) :=
begin
  use Real.pi * 3 / 4, -- θ value
  use Real.sqrt 2, -- ρ value
  simp,
  split,
  { linarith [Real.pi_pos], },
  split,
  { field_simp,
    norm_num,
    rw Real.sin_pi_div_four,
    exact Real.sqrt_sq zero_le_two, },
  split,
  { field_simp,
    norm_num,
    rw Real.cos_pi_div_four,
    exact neg_eq_neg_of_eq (Real.sqrt_sq zero_le_two), },
  { exact ⟨rfl, rfl⟩, }
end

end intersection_point_polar_coordinates_l798_798261


namespace xy_sum_is_one_l798_798915

theorem xy_sum_is_one (x y : ℤ) (h1 : 2021 * x + 2025 * y = 2029) (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 :=
by sorry

end xy_sum_is_one_l798_798915


namespace triangle_angle_extension_l798_798954

theorem triangle_angle_extension :
  ∀ (BAC ABC BCA CDB DBC : ℝ),
  180 = BAC + ABC + BCA →
  CDB = BAC + ABC →
  DBC = BAC + BCA →
  (CDB + DBC) / (BAC + ABC) = 2 :=
by
  intros BAC ABC BCA CDB DBC h1 h2 h3
  sorry

end triangle_angle_extension_l798_798954


namespace log_exponent_equiv_l798_798225

theorem log_exponent_equiv (x : ℝ) (h : log 5 (log 4 (log 2 x)) = 1) : x^(-1/3) = 2^(-341.333... : ℝ) :=
  sorry

end log_exponent_equiv_l798_798225


namespace length_EF_eq_diameter_Γ_l798_798514

open_locale classical

noncomputable theory

variables {O1 O2 A B C D E F : EuclideanGeometry.Point ℝ} {Γ : EuclideanGeometry.Circle ℝ}

/-- Given two circles intersecting at two points, a line through the intersection point with tangents
intersecting on another circle, prove a specific segment length is the diameter of the circle. -/
theorem length_EF_eq_diameter_Γ
  (h1 : EuclideanGeometry.circle O1)
  (h2 : EuclideanGeometry.circle O2)
  (hA : EuclideanGeometry.point_on_circle A h1)
  (hB : EuclideanGeometry.point_on_circle A h2)
  (hC : EuclideanGeometry.line_through B intersects h1 at C)
  (hD : EuclideanGeometry.line_through B intersects h2 at D)
  (hE : EuclideanGeometry.tangent_to_circle C from h1 meets
        EuclideanGeometry.tangent_to_circle D from h2 at E)
  (hΓ : EuclideanGeometry.circumcircle_of_triangle A O1 O2 = Γ)
  (hF : EuclideanGeometry.line_through A E intersects Γ at F)
  : EuclideanGeometry.length_segment E F = EuclideanGeometry.diameter Γ :=
sorry

end length_EF_eq_diameter_Γ_l798_798514


namespace discriminant_min_value_l798_798147

theorem discriminant_min_value :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, (0 ≤ a ∧ 0 ≤ c) ∧ (x ∈ (Icc (1 / 2) (1 / -2)) → ax^2 + bx + c ≤ (1 / (√(1 - x^2))))) → 
  (b^2 - 4 * a * c ≥ -4) :=
  sorry

end discriminant_min_value_l798_798147


namespace base_of_second_fraction_l798_798556

theorem base_of_second_fraction (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 :=
by
  sorry

end base_of_second_fraction_l798_798556


namespace find_other_side_length_l798_798272

variable (total_shingles : ℕ)
variable (shingles_per_sqft : ℕ)
variable (num_roofs : ℕ)
variable (side_length : ℕ)

theorem find_other_side_length
  (h1 : total_shingles = 38400)
  (h2 : shingles_per_sqft = 8)
  (h3 : num_roofs = 3)
  (h4 : side_length = 20)
  : (total_shingles / shingles_per_sqft / num_roofs / 2) / side_length = 40 :=
by
  sorry

end find_other_side_length_l798_798272


namespace no_pieces_left_impossible_l798_798726

/-- Starting with 100 pieces and 1 pile, and given the ability to either:
1. Remove one piece from a pile of at least 3 pieces and divide the remaining pile into two non-empty piles,
2. Eliminate a pile containing a single piece,
prove that it is impossible to reach a situation with no pieces left. -/
theorem no_pieces_left_impossible :
  ∀ (p t : ℕ), p = 100 → t = 1 →
  (∀ (p' t' : ℕ),
    (p' = p - 1 ∧ t' = t + 1 ∧ 3 ≤ p) ∨
    (p' = p - 1 ∧ t' = t - 1 ∧ ∃ k, k = 1 ∧ t ≠ 0) →
    false) :=
by
  intros
  sorry

end no_pieces_left_impossible_l798_798726


namespace original_placements_l798_798748

variables (A B C D E F : ℕ)
-- The numbers in white triangles (A, B, C, D, E, F) equal the sums of their neighboring gray triangle numbers
axiom white_triangle_1 : A = B + C
axiom white_triangle_2 : D = E + F
axiom white_triangle_3 : F = 1 + 2 + 3 -- One of the specific sums in given solution steps

-- Lesha replaced the numbers 1, 2, 3, 4, 5, and 6 with letters.
axiom valid_assignment_1 : F = 4 -- from solution
axiom valid_assignment_2 : E = 6 -- from solution
axiom valid_assignment_3 : D = 5 -- from solution
axiom valid_assignment_4 : A = 1 -- based on remaining value assignment
axiom valid_assignment_5 : B = 3
axiom valid_assignment_6 : C = 2

theorem original_placements :
  A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
begin
  split, exact valid_assignment_4,
  split, exact valid_assignment_5,
  split, exact valid_assignment_6,
  split, exact valid_assignment_3,
  split, exact valid_assignment_2,
  exact valid_assignment_1,
end

end original_placements_l798_798748


namespace part1_part2_l798_798059

-- Definitions based on the conditions
def original_sales : ℕ := 30
def profit_per_shirt_initial : ℕ := 40

-- Additional shirts sold for each 1 yuan price reduction
def additional_shirts_per_yuan : ℕ := 2

-- Price reduction example of 3 yuan
def price_reduction_example : ℕ := 3

-- New sales quantity after 3 yuan reduction
def new_sales_quantity_example := 
  original_sales + (price_reduction_example * additional_shirts_per_yuan)

-- Prove that the sales quantity is 36 shirts for a reduction of 3 yuan
theorem part1 : new_sales_quantity_example = 36 := by
  sorry

-- General price reduction variable
def price_reduction_per_item (x : ℕ) : ℕ := x
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt_initial - x
def new_sales_quantity (x : ℕ) : ℕ := original_sales + (additional_shirts_per_yuan * x)
def daily_sales_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (new_sales_quantity x)

-- Goal for daily sales profit of 1200 yuan
def goal_profit : ℕ := 1200

-- Prove that a price reduction of 25 yuan per shirt achieves a daily sales profit of 1200 yuan
theorem part2 : daily_sales_profit 25 = goal_profit := by
  sorry

end part1_part2_l798_798059


namespace line_parallel_to_plane_sine_angle_between_planes_l798_798932

-- Define the conditions in the problem.
structure RegularTriangularPrism :=
  (A B C A1 B1 C1 D : Point)
  (AA1_eq_4 : dist A A1 = 4)
  (AC_eq_2 : dist A C = 2)
  (CB_eq_BD : dist C B = dist B D)

-- Define the first problem statement.
theorem line_parallel_to_plane
  (P : RegularTriangularPrism)
  (h1 : P.AA1_eq_4)
  (h2 : P.AC_eq_2)
  (h3 : P.CB_eq_BD) :
  parallel (line_through P.C1 P.B) (plane_through P.A P.B1 P.D) :=
sorry

-- Define the second problem statement.
theorem sine_angle_between_planes
  (P : RegularTriangularPrism)
  (h1 : P.AA1_eq_4)
  (h2 : P.AC_eq_2)
  (h3 : P.CB_eq_BD) :
  sin (angle_between_planes (plane_through P.A P.B1 P.D) (plane_through P.A P.C P.B)) = (4 * sqrt 17 / 17) :=
sorry

end line_parallel_to_plane_sine_angle_between_planes_l798_798932


namespace ellipse_equation_slope_range_l798_798501

variables {a b c : ℝ} (a_pos : a > 0) (b_pos : b > 0) (h1 : a > b) (h2 : (1 / a^2 + (9 / 4) / b^2 = 1)) (h3 : a = 2 * c)
variables {x y : ℝ} (hx : x^2 / a^2 + y^2 / b^2 = 1) (h4 : (1, 3 / 2) ∈ set_of (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1))
variables (line_sym : ∀ x y, (x, y) ∈ set_of (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1) ↔ (2 * c - x, y) ∈ set_of (λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1))

theorem ellipse_equation : x^2 / 4 + y^2 / 3 = 1 := 
sorry

variables {m k : ℝ} (h_line_eq : ∀ x, x = m * y + 1 / 2) (h0 : m ≠ 0) (line_intersect : ∀ x0 y0, x0 = 1 / (3 * m^2 + 4) ∧ y0 = -3 * m / (2 * (3 * m^2 + 4)))
variables (h_slope_eq : k = m / (4 * m^2 + 4)) (h_range : (∀ m > 0, 0 < k ∧ k ≤ 1 / 8) ∧ (∀ m < 0, -1 / 8 ≤ k ∧ k < 0))

theorem slope_range : -1 / 8 ≤ k ∧ k ≤ 1 / 8 :=
sorry

end ellipse_equation_slope_range_l798_798501


namespace hexagon_side_length_l798_798095

theorem hexagon_side_length (A : ℝ) (y : ℝ) :
  (A = 192) → ((3 * real.sqrt 3 / 2) * y^2 = A) → (y = 8 * real.sqrt 3 / 3) :=
by
  intros hA hA_hexagon
  sorry -- Proof is skipped

end hexagon_side_length_l798_798095


namespace major_premise_is_wrong_l798_798360

-- Define a differentiable function f and its second derivative condition
def f : ℝ → ℝ := λ x, x^3

theorem major_premise_is_wrong :
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), differentiable ℝ f ∧ (deriv (deriv f) x₀ = 0) → (∃ I : set ℝ, is_extremum f x₀ I)) :=
by {
  -- Define the second derivative of f at x₀ = 0
  let f : ℝ → ℝ := λ x, x^3,
  let x₀ := 0,
  have h₀ : deriv (deriv f) x₀ = 0,
  { simp [f, deriv] },
  -- Show an example where f''(0) does not lead to an extremum
  have h₁ : ¬ is_extremum f x₀ (set.univ),
  { sorry },
  -- Conclude the statement
  exact ⟨f, x₀, ⟨differentiable_id, h₀⟩, h₁⟩,
}

end major_premise_is_wrong_l798_798360


namespace find_a_plus_c_l798_798613

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) 
  (h_vertex_f : -a / 2 = v) (h_vertex_g : -c / 2 = w)
  (h_root_v_g : g v c d = 0) (h_root_w_f : f w a b = 0)
  (h_intersect : f 50 a b = -200 ∧ g 50 c d = -200)
  (h_min_value_f : ∀ x, f (-a / 2) a b ≤ f x a b)
  (h_min_value_g : ∀ x, g (-c / 2) c d ≤ g x c d)
  (h_min_difference : f (-a / 2) a b = g (-c / 2) c d - 50) :
  a + c = sorry :=
sorry

end find_a_plus_c_l798_798613


namespace original_recipe_pasta_l798_798820

noncomputable def pasta_per_person (total_pasta : ℕ) (total_people : ℕ) : ℚ :=
  total_pasta / total_people

noncomputable def original_pasta (pasta_per_person : ℚ) (people_served : ℕ) : ℚ :=
  pasta_per_person * people_served

theorem original_recipe_pasta (total_pasta : ℕ) (total_people : ℕ) (people_served : ℕ) (required_pasta : ℚ) :
  total_pasta = 10 → total_people = 35 → people_served = 7 → required_pasta = 2 →
  pasta_per_person total_pasta total_people * people_served = required_pasta :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end original_recipe_pasta_l798_798820


namespace shifted_parabola_l798_798571

theorem shifted_parabola (x : ℝ) : 
  let original := 5 * x^2 in
  let shifted_left := 5 * (x + 2)^2 in
  let shifted_up := shifted_left + 3 in
  shifted_up = 5 * (x + 2)^2 + 3 := 
by
  sorry

end shifted_parabola_l798_798571


namespace intervals_length_l798_798468

noncomputable def total_length_intervals : ℝ :=
  let sqrt2 := Real.sqrt 2
  let pi := Real.pi
  (sqrt2 / 2) - (pi / 6)

theorem intervals_length : 
  (sqrt 2 - pi / 3) ~ 0.37 :=
begin
  have h1 : sqrt 2 - pi / 3 ≈ 0.37, sorry,
end

end intervals_length_l798_798468


namespace boatman_distance_along_current_l798_798402

def boat_speed_stationary : ℝ := 6 / 3  -- distance / time in stationary water
def boat_speed_against_current : ℝ := 4 / 4  -- distance / time against the current
def current_speed : ℝ := boat_speed_stationary - boat_speed_against_current
def boat_speed_along_current : ℝ := boat_speed_stationary + current_speed

theorem boatman_distance_along_current :
  boat_speed_along_current * (20 / 60) = 1 :=
by
  sorry

end boatman_distance_along_current_l798_798402


namespace distance_on_dirt_road_is_1_km_l798_798390

variable (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ)

def distance_between_on_dirt_road (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ) : ℝ :=
  initial_gap * (city_speed / highway_speed) * (good_road_speed / city_speed) * (dirt_road_speed / good_road_speed)

theorem distance_on_dirt_road_is_1_km :
  distance_between_on_dirt_road 2 60 40 70 30 = 1 :=
  by
    unfold distance_between_on_dirt_road
    sorry

end distance_on_dirt_road_is_1_km_l798_798390


namespace hyperbola_focus_asymptote_distance_l798_798208

theorem hyperbola_focus_asymptote_distance :
  ∀ (x y : ℝ), (∃ x y : ℝ, (x - 0)^2 / 9 - y^2 / 5 = 1) →
  (abs (y - ((√5 / 3) * x)) = sqrt(5) / 3) →
  (complex.abs ((√14) * √5 / 3 / sqrt(1 + (√5 / 3)^2)) = sqrt(5)) :=
by
  intro x y hyp1 hyp2
  sorry

end hyperbola_focus_asymptote_distance_l798_798208


namespace max_value_f_interval_neg3_neg2_l798_798620

theorem max_value_f_interval_neg3_neg2 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = - f x)
  (h2 : ∀ x y, f (x + y) = f x + f y)
  (h3 : f 1 = 2)
  (h4 : ∀ x y, x > 0 → y > 0 → f x < f (x + y)) : 
  ∃ x ∈ set.Icc (-3 : ℝ) (-2 : ℝ), ∀ y ∈ set.Icc (-3 : ℝ) (-2 : ℝ), f y ≤ f x ∧ f x = -4 := 
sorry

end max_value_f_interval_neg3_neg2_l798_798620


namespace find_divisor_l798_798312

theorem find_divisor (x : ℕ) (h : 144 = (x * 13) + 1) : x = 11 := by
  sorry

end find_divisor_l798_798312


namespace constant_term_of_binomial_expansion_is_7_l798_798947

noncomputable def binomial_expansion (x : ℝ) : ℝ :=
  (x / 2 - 1 / x^(1/3))^8

theorem constant_term_of_binomial_expansion_is_7 :
  ∃ (T : ℝ), T = 7 ∧ (T = (-1)^6 * (1/2)^2 * (nat.choose 8 6)) :=
by
  sorry

end constant_term_of_binomial_expansion_is_7_l798_798947


namespace john_out_of_pocket_l798_798966

-- Define the conditions
def computer_cost : ℕ := 700
def accessories_cost : ℕ := 200
def playstation_value : ℕ := 400
def sale_discount : ℚ := 0.2

-- Define the total cost of the computer and accessories
def total_cost : ℕ := computer_cost + accessories_cost

-- Define the selling price of the PlayStation
def selling_price : ℕ := playstation_value - (playstation_value * sale_discount).to_nat

-- Define the amount out of John's pocket
def out_of_pocket : ℕ := total_cost - selling_price

-- The proof goal
theorem john_out_of_pocket : out_of_pocket = 580 :=
by
  sorry

end john_out_of_pocket_l798_798966


namespace shipping_cost_correct_l798_798120

def shipping_cost (W: ℝ) : ℕ :=
  3 + 5 * Int.to_nat (⌈W - 1⌉)

theorem shipping_cost_correct (W: ℝ) (h: W = 1.5 ∨ W = 2) : shipping_cost W = 8 :=
by
  sorry

end shipping_cost_correct_l798_798120


namespace binary_calculation_l798_798837

-- Binary arithmetic definition
def binary_mul (a b : Nat) : Nat := a * b
def binary_div (a b : Nat) : Nat := a / b

-- Binary numbers in Nat (representing binary literals by their decimal equivalent)
def b110010 := 50   -- 110010_2 in decimal
def b101000 := 40   -- 101000_2 in decimal
def b100 := 4       -- 100_2 in decimal
def b10 := 2        -- 10_2 in decimal
def b10111000 := 184-- 10111000_2 in decimal

theorem binary_calculation :
  binary_div (binary_div (binary_mul b110010 b101000) b100) b10 = b10111000 :=
by
  sorry

end binary_calculation_l798_798837


namespace min_people_liking_both_l798_798638

theorem min_people_liking_both {A B U : Finset ℕ} (hU : U.card = 150) (hA : A.card = 130) (hB : B.card = 120) :
  (A ∩ B).card ≥ 100 :=
by
  -- Proof to be filled later
  sorry

end min_people_liking_both_l798_798638


namespace angle_between_vectors_collinear_points_l798_798288

open Real

-- Part (1)
theorem angle_between_vectors
  (a b : ℝ^3)
  (m : ℝ)
  (h1 : m = -1 / 2)
  (h2 : ∥a∥ = 2 * sqrt 2 * ∥b∥)
  (h3 : ((m + 1) • a + b) ⬝ (a - 3 • b) = 0) :
  ∀ θ, cos θ = 1 / sqrt 2 ↔ θ = π / 4 :=
sorry

-- Part (2)
theorem collinear_points
  (a b : ℝ^3)
  (m : ℝ)
  (OA : ℝ^3 := m • a - b)
  (OB : ℝ^3 := (m + 1) • a + b)
  (OC : ℝ^3 := a - 3 • b)
  (h1 : collinear {OA, OB, OC}) :
  m = 2 :=
sorry

end angle_between_vectors_collinear_points_l798_798288


namespace isosceles_triangle_sides_l798_798188

theorem isosceles_triangle_sides (a b c : ℝ) (h_iso : a = b ∨ b = c ∨ c = a) (h_perimeter : a + b + c = 14) (h_side : a = 4 ∨ b = 4 ∨ c = 4) : 
  (a = 4 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 4) :=
  sorry

end isosceles_triangle_sides_l798_798188


namespace solve_equation1_solve_equation2_l798_798156

theorem solve_equation1 (x : ℝ) (h : 4 * x^2 - 81 = 0) : x = 9/2 ∨ x = -9/2 := 
sorry

theorem solve_equation2 (x : ℝ) (h : 8 * (x + 1)^3 = 27) : x = 1/2 := 
sorry

end solve_equation1_solve_equation2_l798_798156


namespace right_triangle_angles_sum_to_right_l798_798322

theorem right_triangle_angles_sum_to_right (
  a b c : ℝ) 
  (h_triangle : a + b + c = 180)
  (h_right_angle : c = 90) : 
  a + b = 90 :=
by
  have h_sum : a + b + 90 = 180 := h_triangle
  rw h_right_angle at h_sum
  linarith

end right_triangle_angles_sum_to_right_l798_798322


namespace problem1_problem2_problem3_problem4_l798_798398

-- Problem 1
theorem problem1 (a b : ℝ) (θ : ℝ) (h1 : |a| = 4) (h2 : |b| = 1) (h3 : ℝ) (h4 : |a - 2 * b| = 4) :
  real.cos θ = 1 / 4 := by sorry

-- Problem 2
theorem problem2 (an : ℕ → ℝ) (h : ∀ n, an n = 1 / (real.sqrt n + real.sqrt (n + 1))) :
  (finset.range 8).sum an = 2 := by sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x - 1) * (x + 1) * (2 * x - 1) ≤ 0) :
  (x ∈ set.Icc (-∞ : ℝ) (-1) ∪ set.Icc (1 / 2) 1) := by sorry

-- Problem 4
theorem problem4 (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : (m, 1) ∥ (4 - n, 2)) :
  1 / m + 8 / n = 9 / 2 := by sorry

end problem1_problem2_problem3_problem4_l798_798398


namespace count_concave_numbers_l798_798787

def isConcave (n : ℕ) : Prop :=
  let d₀ := n % 10
  let d₁ := (n / 10) % 10
  let d₂ := (n / 100) % 10
  d₂ ≠ 0 ∧ d₀ ≠ d₁ ∧ d₁ ≠ d₂ ∧ d₀ ≠ d₂ ∧ d₁ < d₂ ∧ d₁ < d₀

theorem count_concave_numbers : 
  (finset.filter isConcave (finset.range 1000)).card = 240 :=
sorry

end count_concave_numbers_l798_798787


namespace regular_ngon_if_product_at_most_two_l798_798497

noncomputable def unitCircle : Set ℂ := {z | Complex.abs z = 1}

def poly (n : ℕ) (z_points : Fin n → ℂ) : ℂ[X] :=
  Polynomial.prod (Finset.univ.map (Function.embedding (λ i, X - C (z_points i))))

theorem regular_ngon_if_product_at_most_two
  (n : ℕ)
  (z_points : Fin n → ℂ)
  (product_dist : ∀ (z : ℂ), z ∈ unitCircle → Complex.abs (Polynomial.eval z (poly n z_points)) ≤ 2) :
  ∃ k : ℂ, ∀ j : Fin n, z_points j = k * Complex.exp (2 * π * I * ↑j / ↑n) := 
sorry

end regular_ngon_if_product_at_most_two_l798_798497


namespace remainder_div_72_l798_798374

theorem remainder_div_72 (x : ℤ) (h : x % 8 = 3) : x % 72 = 3 :=
sorry

end remainder_div_72_l798_798374


namespace angle_between_vectors_collinear_points_l798_798287

open Real

-- Part (1)
theorem angle_between_vectors
  (a b : ℝ^3)
  (m : ℝ)
  (h1 : m = -1 / 2)
  (h2 : ∥a∥ = 2 * sqrt 2 * ∥b∥)
  (h3 : ((m + 1) • a + b) ⬝ (a - 3 • b) = 0) :
  ∀ θ, cos θ = 1 / sqrt 2 ↔ θ = π / 4 :=
sorry

-- Part (2)
theorem collinear_points
  (a b : ℝ^3)
  (m : ℝ)
  (OA : ℝ^3 := m • a - b)
  (OB : ℝ^3 := (m + 1) • a + b)
  (OC : ℝ^3 := a - 3 • b)
  (h1 : collinear {OA, OB, OC}) :
  m = 2 :=
sorry

end angle_between_vectors_collinear_points_l798_798287


namespace angle_XZY_is_45_l798_798416

theorem angle_XZY_is_45
  {X Y Z : Type}
  (h_triangle : let ΔXYZ := triangle.mk X Y Z in @is_isosceles_triangle ΔXYZ)
  (h_right_angle_at_Z : ∠XZY = 90)
  (h_eq_angles : ∠YXZ = ∠XZY) : 
  ∠XZY = 45 := by
  sorry

end angle_XZY_is_45_l798_798416


namespace find_f_2006_l798_798517

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) := ∀ x : ℝ, g x = f (x - 1)
def f_at_2 (f : ℝ → ℝ) := f 2 = 2

-- The theorem to prove
theorem find_f_2006 (f g : ℝ → ℝ) 
  (even_f : is_even f) 
  (odd_g : is_odd g) 
  (g_eq_f_shift : g_def f g) 
  (f_eq_2 : f_at_2 f) : 
  f 2006 = 2 := 
sorry

end find_f_2006_l798_798517


namespace trapezoid_unique_exists_l798_798126

noncomputable def construct_trapezoid_exists
    (A B C D : Type*)
    (AB : ℝ)
    (angle_A angle_B : ℝ)
    (dc_parallel : Prop)
    (AD BC DC : ℝ) : Prop :=
  AB = 1 ∧ angle_A = 15 ∧ angle_B = 15 ∧ dc_parallel ∧ AD = BC ∧ BC = DC ∧ DC = AD

-- Assertion that such a trapezoid exists and is uniquely determined
theorem trapezoid_unique_exists 
  (A B C D : ℝ)
  (AB_length : AB = 1)
  (angle_A : ∠A = 15)
  (angle_B : ∠B = 15)
  (DC_parallel : are_parallel DC AB)
  (AD_length : AD = BC)
  (BC_length : BC = DC) :
  construct_trapezoid_exists A B C D AB ∠A ∠B DC_parallel AD BC DC :=
by {
  sorry
}

end trapezoid_unique_exists_l798_798126


namespace strips_cover_circle_l798_798575

def strip := { width : ℝ // 0 ≤ width }
def finite_strips_cover_circle (strips : List strip) : Prop :=
  (∑ s in strips, s.width = 100) →
  ∃ (translate : strip → (strip × ℝ)), ∀ s, (translate s).1.width = s.width

theorem strips_cover_circle (strips : List strip) (circle_radius : ℝ) :
  circle_radius = 1 →
  ∑ s in strips, s.width = 100 →
  ∃ translate, ∀ s, (translate s).1.width = s.width :=
by
  intro circle_radius_eq width_sum_eq
  have h1 : circle_radius = 1 := circle_radius_eq
  have h2 : ∑ s in strips, s.width = 100 := width_sum_eq
  sorry

end strips_cover_circle_l798_798575


namespace discriminant_min_value_l798_798148

theorem discriminant_min_value :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, (0 ≤ a ∧ 0 ≤ c) ∧ (x ∈ (Icc (1 / 2) (1 / -2)) → ax^2 + bx + c ≤ (1 / (√(1 - x^2))))) → 
  (b^2 - 4 * a * c ≥ -4) :=
  sorry

end discriminant_min_value_l798_798148


namespace vector_magnitude_orthogonal_condition_l798_798196

variables (a b : ℝ^3) (λ : ℝ)
variables (ha : ∥a∥ = sqrt 3) (hb : ∥b∥ = 2)
variables (angle_ab : real.angle a b = 5 * real.pi / 6)

-- Define the problem for part (1)
theorem vector_magnitude : ∥a - 2 • b∥ = sqrt 31 :=
by sorry

-- Define the problem for part (2)
theorem orthogonal_condition : (a + 3 * λ • b) ⬝ (a + λ • b) = 0 → λ = 1 / 2 :=
by sorry

end vector_magnitude_orthogonal_condition_l798_798196


namespace collinear_condition_l798_798146

variable {R : Type*} [LinearOrderedField R]
variable {x1 y1 x2 y2 x3 y3 : R}

theorem collinear_condition : 
  x1 * y2 + x2 * y3 + x3 * y1 = y1 * x2 + y2 * x3 + y3 * x1 →
  ∃ k l m : R, k * (x2 - x1) = l * (y2 - y1) ∧ k * (x3 - x1) = m * (y3 - y1) :=
by
  sorry

end collinear_condition_l798_798146


namespace possible_radii_count_l798_798814

def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n

theorem possible_radii_count :
  let radius_A := 144
   in ∃ (r : ℕ), r < radius_A ∧ is_perfect_square r ∧ r ≠ radius_A ∧ 
                 (∀ r, r < radius_A ∧ is_perfect_square r ∧ r ≠ radius_A 
                  → r = 1 ∨ r = 4 ∨ r = 9 ∨ r = 16 ∨ r = 36) := sorry

end possible_radii_count_l798_798814


namespace ball_picking_probability_l798_798938

/-
  Given the following conditions:
  1. A bag contains only black and white balls.
  2. Balls are picked randomly with replacement, and their colors are recorded.

  Prove that:
  1. The assertion that the frequency will definitely be 0.6 after 10000 picks is incorrect.
  2. It can be reasonably estimated that the probability of picking a white ball is approximately 0.6 based on the provided statistical data.
-/

theorem ball_picking_probability (n m : ℕ) (frequency : ℝ) (h1 : ∀ n m, (m / n : ℝ) ≈ 0.6) :
  (∀ n ≥ 10000, (m / n : ℝ) ≠ 0.6) →
  (∀ n, (m / n : ℝ) → 0.6) :=
by
  sorry

end ball_picking_probability_l798_798938


namespace minimum_tea_brewing_time_l798_798554

theorem minimum_tea_brewing_time : 
  (∀ t1 t2 t3 t4 t5 : ℕ, t1 = 2 ∧ t2 = 2 ∧ t3 = 1 ∧ t4 = 15 ∧ t5 = 1 → (min t1 t4 + min t2 t4 + min t3 t4 + t5) = 18) :=
begin
  sorry
end

end minimum_tea_brewing_time_l798_798554


namespace ball_distribution_problem_l798_798642

theorem ball_distribution_problem : 
  let n : ℕ := 10 in let k : ℕ := 3 in 
  (∃ f : Fin k → ℕ, (Σ i, f i) = n ∧ ∀ i : Fin k, f i ≥ (i : ℕ) + 1) → 
  (nat.choose 6 2) = 15 :=
by
  intro n k f h
  have h1 := h.1
  have h2 := h.2
  sorry

end ball_distribution_problem_l798_798642


namespace area_of_right_triangle_l798_798252

theorem area_of_right_triangle 
  (DEF : Type)
  [Triangle DEF]
  (right_triangle : is_right_triangle DEF)
  (DE : ℝ)
  (DF : ℝ)
  (height_from_D_to_F : ℝ)
  (base_DE : DE = 12)
  (height_DF : height_from_D_to_F = 15) : 
  area DEF = 90 := 
sorry

end area_of_right_triangle_l798_798252


namespace divisor_is_three_l798_798413

noncomputable def find_divisor (n : ℕ) (reduction : ℕ) (result : ℕ) : ℕ :=
  n / result

theorem divisor_is_three (x : ℝ) : 
  (original : ℝ) → (reduction : ℝ) → (new_result : ℝ) → 
  original = 45 → new_result = 45 - 30 → (original / x = new_result) → 
  x = 3 := by 
  intros original reduction new_result h1 h2 h3
  sorry

end divisor_is_three_l798_798413


namespace johns_out_of_pocket_expense_l798_798969

theorem johns_out_of_pocket_expense
  (computer_cost : ℕ)
  (accessories_cost : ℕ)
  (playstation_value : ℕ)
  (playstation_sold_percent_less : ℕ) :
  computer_cost = 700 →
  accessories_cost = 200 →
  playstation_value = 400 →
  playstation_sold_percent_less = 20 →
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100) in
  let total_cost := computer_cost + accessories_cost in
  let pocket_expense := total_cost - playstation_sold_price in
  pocket_expense = 580 :=
by
  intros h1 h2 h3 h4
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100)
  let total_cost := computer_cost + accessories_cost
  let pocket_expense := total_cost - playstation_sold_price
  sorry

end johns_out_of_pocket_expense_l798_798969


namespace problem1_problem2_l798_798524

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem problem1 (a : ℝ) (h_pos : a > 0) : 
  (∀ x : ℝ, x > 0 → Real.deriv (λ x, f x a) x ≥ 0) → (0 < a ∧ a ≤ 2) :=
begin
  sorry
end

theorem problem2 (a : ℝ) (h_pos : a > 0) : 
  (∀ x : ℝ, x > 0 → (x - 1) * f x a ≥ 0) → (0 < a ∧ a ≤ 2) :=
begin
  sorry
end

end problem1_problem2_l798_798524


namespace total_pupils_in_school_l798_798934

theorem total_pupils_in_school (girls boys : ℕ) (h_girls : girls = 542) (h_boys : boys = 387) : girls + boys = 929 := by
  sorry

end total_pupils_in_school_l798_798934


namespace first_company_managers_percentage_l798_798768

-- Definitions from the conditions
variable (F M : ℝ) -- total workforce of first company and merged company
variable (x : ℝ) -- percentage of managers in the first company
variable (cond1 : 0.25 * M = F) -- 25% of merged company's workforce originated from the first company
variable (cond2 : 0.25 * M / M = 0.25) -- resulting merged company's workforce consists of 25% managers

-- The statement to prove
theorem first_company_managers_percentage : x = 25 :=
by
  sorry

end first_company_managers_percentage_l798_798768


namespace compact_subsets_at_least_l798_798866

def is_good (X : Finset ℕ) (a : ℝ) : Prop :=
  ∃ x ∈ X, abs (x - a) * 2 ≤ 1

def is_compact (X : Finset ℕ) (S : Finset ℕ) : Prop :=
  S ⊆ X ∧ is_good X (S.sum / S.card)

def compact_subsets_count (X : Finset ℕ) : ℕ :=
  (Finset.powersetLen (X.card - 3) X).filter (λ S, is_compact X S).card

theorem compact_subsets_at_least (X : Finset ℕ) (h : ∀ i ∈ Finset.range (X.card - 1) \{0}, 1 ≤ X (i+1) - X i ∧ X (i+1) - X i ≤ 2) :
  compact_subsets_count X ≥ 2 ^ (X.card - 3) :=
sorry

end compact_subsets_at_least_l798_798866


namespace cosine_smallest_angle_l798_798953

theorem cosine_smallest_angle (A B C : ℝ) 
  (hABC : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_sum_angle : A + B + C = π) 
  (h_sin_ratio : ∃ k : ℝ, k > 0 ∧ sin A = 3 * k ∧ sin B = 5 * k ∧ sin C = 7 * k) : 
  cos A = 13 / 14 :=
sorry

end cosine_smallest_angle_l798_798953


namespace intervals_length_l798_798469

noncomputable def total_length_intervals : ℝ :=
  let sqrt2 := Real.sqrt 2
  let pi := Real.pi
  (sqrt2 / 2) - (pi / 6)

theorem intervals_length : 
  (sqrt 2 - pi / 3) ~ 0.37 :=
begin
  have h1 : sqrt 2 - pi / 3 ≈ 0.37, sorry,
end

end intervals_length_l798_798469


namespace not_p_and_not_p_and_q_implies_not_p_or_q_l798_798884

theorem not_p_and_not_p_and_q_implies_not_p_or_q (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬(p ∨ q) :=
sorry

end not_p_and_not_p_and_q_implies_not_p_or_q_l798_798884


namespace num_quadricycles_l798_798410

theorem num_quadricycles (b t q : ℕ) (h1 : b + t + q = 10) (h2 : 2 * b + 3 * t + 4 * q = 30) : q = 2 :=
by sorry

end num_quadricycles_l798_798410


namespace three_lines_coplanar_if_line_intersects_two_parallel_l798_798735

-- Definitions
def line (ℝ: Type) (V: Type) [AddCommGroup V] [Module ℝ V] := V →ₗ[ℝ] V

def coplanar_lines {ℝ: Type} [RealField ℝ] (V: Type) [AddCommGroup V] [Module ℝ V] (l₁ l₂ l₃ : line ℝ V) : Prop :=
  ∃ P : affine_subspace ℝ P V, l₁ ∈ P ∧ l₂ ∈ P ∧ l₃ ∈ P

def intersects (ℝ: Type) (V: Type) [AddCommGroup V] [Module ℝ V] (l₁ l₂ : line ℝ V) : Prop :=
  ∃ p : V, l₁ p = l₂ p

def parallel (ℝ: Type) (V: Type) [AddCommGroup V] [Module ℝ V] (l₁ l₂ : line ℝ V) : Prop :=
  ∀ p p' : V, l₁ p - l₂ p = l₁ p' - l₂ p'

-- Statement
theorem three_lines_coplanar_if_line_intersects_two_parallel {ℝ : Type} [RealField ℝ] 
  (V : Type) [AddCommGroup V] [Module ℝ V] 
  (l₁ l₂ l₃ : line ℝ V)
  (h₁ : intersects ℝ V l₁ l₂)
  (h₂ : parallel ℝ V l₂ l₃) :
  coplanar_lines ℝ V l₁ l₂ l₃ := 
sorry

end three_lines_coplanar_if_line_intersects_two_parallel_l798_798735


namespace max_we_than_nat_treas_l798_798019

variable (x y z : ℝ)
variable (N : ℝ := x + y - z)

theorem max_we_than_nat_treas (h1 : 2 * x)
                              (h2 : y < z) : 
                              y > N :=
by
  sorry

end max_we_than_nat_treas_l798_798019


namespace problem_part1_problem_part2_l798_798202

def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1
else 2^x

theorem problem_part1 : f(f(2/3)) = 2 := by
  sorry

theorem problem_part2 (a : ℝ) (h : f(f(a)) = 1) : a = 5/9 := by
  sorry

end problem_part1_problem_part2_l798_798202


namespace tan_alpha_eq_one_l798_798916

open Real

theorem tan_alpha_eq_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 := 
by
  sorry

end tan_alpha_eq_one_l798_798916


namespace divisor_of_12401_76_13_l798_798920

theorem divisor_of_12401_76_13 (D : ℕ) (h1: 12401 = (D * 76) + 13) : D = 163 :=
sorry

end divisor_of_12401_76_13_l798_798920


namespace tangent_eq_inequality_not_monotonic_l798_798526

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem tangent_eq (a : ℝ) (h : 0 < a) : 
  ∃ k : ℝ, (k, f 1 a) ∈ {
    p : ℝ × ℝ | p.1 - (a + 1) * p.2 - 1 = 0 
  } :=
  sorry

theorem inequality (x : ℝ) (h : 1 ≤ x) : f x 1 ≤ (x - 1) / 2 := 
  sorry

theorem not_monotonic (a : ℝ) (h : 0 < a) : 
  ¬(∀ x y : ℝ, x < y → f x a ≤ f y a ∨ x < y → f x a ≥ f y a) := 
  sorry

end tangent_eq_inequality_not_monotonic_l798_798526


namespace triangle_BC_ratio_l798_798974

theorem triangle_BC_ratio (A B C D E X : Type)
  [triangle_ABC : triangle A B C]
  (hAB : distance A B = 10)
  (hAC : distance A C = 11)
  (hCircumradius : circumradius A B C = 6)
  (hEquilateral : equilateral_triangle A D E)
  (hCircumcircleD : on_circumcircle D A B C)
  (hCircumcircleE : on_circumcircle E A B C)
  (hIntersectX : intersects_at D E B C X) :
  ratio (distance B X) (distance X C) = 8 / 13 :=
sorry

end triangle_BC_ratio_l798_798974


namespace curve_C1_polar_equation_line_l_cartesian_equation_maximum_area_ΔPAB_l798_798943

-- Definitions of conditions

def curve_C1_cartesian (α : ℝ) : (ℝ × ℝ) :=
  (2 + sqrt(7) * cos α, sqrt(7) * sin α)

def curve_C2_polar (θ : ℝ) : ℝ :=
  8 * cos θ

def line_l_polar (θ : ℝ) : Prop :=
  θ = π / 3

-- Statements to be proved

theorem curve_C1_polar_equation :
  ∀ (ρ θ : ℝ), (2 + sqrt(7) * cos θ, sqrt(7) * sin θ) =
  (ρ * cos θ, ρ * sin θ) →
  ρ^2 - 4*ρ*cos θ - 3 = 0 :=
by
  sorry

theorem line_l_cartesian_equation:
  ∀ (x y : ℝ), (x, y) ∈ set_of (fun p : ℝ × ℝ => θ = π / 3) →
  y = sqrt(3) * x :=
by
  sorry

theorem maximum_area_ΔPAB:
  ∀ (A B P : ℝ × ℝ),
  A = (3, tan (π / 3)) ∧
  B = (4, 0) ∧ 
  ∃ ρ θ; curve_C2_polar θ = 8 * cos θ →
  AB = 1 →
  ∃ area; area = 2 + sqrt(3) :=
by
  sorry

end curve_C1_polar_equation_line_l_cartesian_equation_maximum_area_ΔPAB_l798_798943


namespace probability_of_sum_5_when_two_dice_rolled_l798_798017

theorem probability_of_sum_5_when_two_dice_rolled :
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_possible_outcomes : ℝ) = (1 / 9 : ℝ) :=
by
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  have h : (favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ) = (1 / 9 : ℝ) := sorry
  exact h

end probability_of_sum_5_when_two_dice_rolled_l798_798017


namespace johns_out_of_pocket_expense_l798_798962

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l798_798962


namespace all_three_white_probability_l798_798054

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end all_three_white_probability_l798_798054


namespace determine_x_l798_798213

theorem determine_x (x : ℝ) (A B : Set ℝ) (H1 : A = {-1, 0}) (H2 : B = {0, 1, x + 2}) (H3 : A ⊆ B) : x = -3 :=
sorry

end determine_x_l798_798213


namespace compare_values_l798_798512

noncomputable def a := 8.1 ^ 0.51
noncomputable def b := 8.1 ^ 0.5
noncomputable def c := Real.log 0.3 / Real.log 3

theorem compare_values : c < b ∧ b < a :=
 by sorry

end compare_values_l798_798512


namespace reflection_through_plane_l798_798610

def normal_vector : ℝ × ℝ × ℝ := (-1, 2, 2)

def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    [11 / 9, -4 / 9, -4 / 9],
    [-4 / 9, 7 / 9, 8 / 9],
    [-4 / 9, 8 / 9, 7 / 9]
  ]

theorem reflection_through_plane (v : Fin 3 → ℝ) : 
  let R := reflection_matrix in
  let n := normal_vector in
  ∀ v : Fin 3 → ℝ,
  R.mulVec v = 2 * ((v - (dot_product v n / dot_product n n) • n)) - v :=
sorry

end reflection_through_plane_l798_798610


namespace dihedral_angle_B_A1C_D_l798_798258

theorem dihedral_angle_B_A1C_D :
  let A := (0, 1, 0)
      B := (1, 1, 0)
      C := (1, 0, 0)
      D := (0, 0, 0)
      A1 := (0, 1, 1)
      plane1 := {p | let (x, y, z) := p in x + z - 1 = 0}
      plane2 := {p | let (x, y, z) := p in y - z = 0} in
  let n1 := (1, 0, 1)
      n2 := (0, 1, -1)
      dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
      magnitude n := Real.sqrt (n.1^2 + n.2^2 + n.3^2)
      cos_theta := dot_product / (magnitude n1 * magnitude n2)
      theta := Real.acos cos_theta in
  theta = 2 * Real.pi / 3 := sorry

end dihedral_angle_B_A1C_D_l798_798258


namespace otimes_calculation_l798_798482

def otimes (a b c : ℝ) (h : b ≠ c) : ℝ := a / (b - c)

theorem otimes_calculation :
  otimes (otimes 2 4 6 (by norm_num)) (otimes 4 6 2 (by norm_num)) (otimes 6 2 4 (by norm_num)) (by norm_num) = -1/4 :=
by sorry

end otimes_calculation_l798_798482


namespace hexagon_area_l798_798370

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

noncomputable def p0 := Point.mk 0 0
noncomputable def p1 := Point.mk 1 4
noncomputable def p2 := Point.mk 3 4
noncomputable def p3 := Point.mk 4 0
noncomputable def p4 := Point.mk 3 (-4)
noncomputable def p5 := Point.mk 1 (-4)

-- Define the problem: calculate the area of the hexagon formed by the points.
theorem hexagon_area : 
  let hexagon_points := [p0, p1, p2, p3, p4, p5] in
  polygon_area hexagon_points = 24 := by
  sorry

end hexagon_area_l798_798370


namespace largest_room_width_l798_798688

theorem largest_room_width (w : ℕ) :
  (w * 30 - 15 * 8 = 1230) → (w = 45) :=
by
  intro h
  sorry

end largest_room_width_l798_798688


namespace equation_has_one_real_solution_l798_798162

theorem equation_has_one_real_solution (k : ℚ) :
    (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x ↔ x^2 + 4 * x + (10 - k) = 0) →
    (∃ k : ℚ, (∀ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ↔ by sorry (condition for one real solution is equivalent to discriminant being zero), k = 6) := by
    sorry

end equation_has_one_real_solution_l798_798162


namespace dodecagon_rotation_l798_798716

theorem dodecagon_rotation :
  (∀ (sides : ℕ), sides = 12 → (360 / sides) = 30) :=
begin
  intros sides h,
  have : sides = 12, from h,
  rw this,
  norm_num,
end

end dodecagon_rotation_l798_798716


namespace train_speed_in_kph_l798_798424

def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 135
def time_to_cross_bridge : ℝ := 11.279097672186225

theorem train_speed_in_kph :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 ≈ 74.988 := by
  sorry

end train_speed_in_kph_l798_798424


namespace point_P_below_line_l798_798898

def line_equation (x y : ℝ) : ℝ := 2 * x - y + 3

def point_below_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x - y + 3 > 0

theorem point_P_below_line :
  point_below_line (1, -1) :=
by
  sorry

end point_P_below_line_l798_798898


namespace winning_strategy_exists_l798_798668

theorem winning_strategy_exists (x y : ℕ) :
  ∃ f : (ℕ × ℕ) → (ℕ × ℕ), (∀ n, f (n, n) = (0, 0)) ∧
  ((∃ n, f (x, y) = (n, n) ∨ f (x, y) = (n, y) ∨ f (x, y) = (x, n)) ∧ (f n (n + x) = (0, 0) → f (x + y - n, n + y) = (0, 0))) :=
sorry

end winning_strategy_exists_l798_798668


namespace sampling_probability_equal_l798_798552

theorem sampling_probability_equal :
  let total_people := 2014
  let first_sample := 14
  let remaining_people := total_people - first_sample
  let sample_size := 50
  let probability := sample_size / total_people
  50 / 2014 = 25 / 1007 :=
by
  sorry

end sampling_probability_equal_l798_798552


namespace nonagon_area_l798_798419

noncomputable def area_of_nonagon (r : ℝ) : ℝ :=
  (9 / 2) * r^2 * Real.sin (Real.pi * 40 / 180)

theorem nonagon_area (r : ℝ) : 
  area_of_nonagon r = 2.891 * r^2 :=
by
  sorry

end nonagon_area_l798_798419


namespace proof_b_lt_a_lt_c_l798_798493

noncomputable def a : ℝ := 2^(4/5)
noncomputable def b : ℝ := 4^(2/7)
noncomputable def c : ℝ := 25^(1/5)

theorem proof_b_lt_a_lt_c : b < a ∧ a < c := by
  sorry

end proof_b_lt_a_lt_c_l798_798493


namespace john_brown_bags_l798_798595

theorem john_brown_bags :
  (∃ b : ℕ, 
     let total_macaroons := 12
     let weight_per_macaroon := 5
     let total_weight := total_macaroons * weight_per_macaroon
     let remaining_weight := 45
     let bag_weight := total_weight - remaining_weight
     let macaroons_per_bag := bag_weight / weight_per_macaroon
     total_macaroons / macaroons_per_bag = b
  ) → b = 4 :=
by
  sorry

end john_brown_bags_l798_798595


namespace parametric_to_cartesian_l798_798013

variables (θ x y : ℝ)

def parametric_equations (θ x y : ℝ) : Prop :=
  x = cos θ / (1 + cos θ) ∧ y = sin θ / (1 + cos θ)

theorem parametric_to_cartesian (θ x y : ℝ) (h : parametric_equations θ x y) :
  y^2 = -2 * (x - 1/2) :=
sorry

end parametric_to_cartesian_l798_798013


namespace altered_solution_detergent_l798_798747

theorem altered_solution_detergent (initial_bleach : ℤ) (initial_detergent : ℤ) (initial_water : ℤ) (final_water : ℤ) :
  initial_bleach = 2 ∧ initial_detergent = 40 ∧ initial_water = 100 ∧ final_water = 200 →
    let final_detergent := (40 / 1) * (final_water / 100) in
      final_detergent = 80 :=
by
  sorry

end altered_solution_detergent_l798_798747


namespace expression_evaluation_l798_798836

def e1 : ℤ := 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8)

theorem expression_evaluation : e1 = -50 :=
by
  sorry

end expression_evaluation_l798_798836


namespace find_solutions_l798_798838

def is_solution (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 1 ∧
  a ∣ (b + c) ∧
  b ∣ (c + d) ∧
  c ∣ (d + a) ∧
  d ∣ (a + b)

theorem find_solutions : ∀ (a b c d : ℕ),
  is_solution a b c d →
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 4 ∧ c = 1 ∧ d = 3) ∨
  (a = 7 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 4 ∧ d = 3) ∨
  (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 1) ∨
  (a = 7 ∧ b = 2 ∧ c = 5 ∧ d = 3) ∨
  (a = 7 ∧ b = 3 ∧ c = 4 ∧ d = 5) :=
by
  intros a b c d h
  sorry

end find_solutions_l798_798838


namespace cos_identity_proof_l798_798198

variable (α : ℝ)

theorem cos_identity_proof :
  (3 + 4 * Real.cos (4 * α) + Real.cos (8 * α)) / (3 - 4 * Real.cos (4 * α) + Real.cos (8 * α)) = (Real.cot (2 * α))^4 := 
  sorry

end cos_identity_proof_l798_798198


namespace total_porridge_l798_798069

variable {c1 c2 c3 c4 c5 c6 : ℝ}

theorem total_porridge (h1 : c3 = c1 + c2)
                      (h2 : c4 = c2 + c3)
                      (h3 : c5 = c3 + c4)
                      (h4 : c6 = c4 + c5)
                      (h5 : c5 = 10) :
                      c1 + c2 + c3 + c4 + c5 + c6 = 40 := 
by
  sorry

end total_porridge_l798_798069


namespace equation_of_line_AB_equation_of_trajectory_l798_798061

open Real

-- Define Circle C with equation (x-1)^2 + y^2 = 1
def CircleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point P(3, 1)
def P := (3 : ℝ, 1 : ℝ)

-- Define the existence of tangents PA and PB from P to Circle C, touching Circle C at A and B
def Tangent (P A : ℝ × ℝ) (C : ℝ × ℝ → Prop) : Prop := ∃ t, P = (A.1 + t*(A.1 - 1), A.2 + t*A.2)

-- Define Point Q on Circle C and midpoint M of PQ
def onCircleC (Q : ℝ × ℝ) : Prop := CircleC Q.1 Q.2
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Problem 1: Equation of line AB is 2x + y - 3 = 0
theorem equation_of_line_AB : ∀ A B : ℝ × ℝ, Tangent P A CircleC → Tangent P B CircleC → ∀ x y : ℝ, (x, y) ∈ line_through A B ↔ 2 * x + y - 3 = 0 :=
by sorry

-- Problem 2: Equation of trajectory of midpoint M of PQ is (x-2)^2 + (y - 1/2)^2 = 1/4
theorem equation_of_trajectory : ∀ Q M : ℝ × ℝ, onCircleC Q → M = midpoint P Q → (M.1 - 2)^2 + (M.2 - 1/2)^2 = 1/4 :=
by sorry

end equation_of_line_AB_equation_of_trajectory_l798_798061


namespace domino_disjoint_sets_cover_rectangle_l798_798076

/-- 
Given a rectangle covered in two layers with \(1 \times 2\) dominoes such that each cell is covered by exactly two dominoes,
prove that the dominoes can be divided into two disjoint sets, each of which covers the entire rectangle.
-/
theorem domino_disjoint_sets_cover_rectangle
    (rectangle : Type) [is_rectangle rectangle]
    (domino_covering : ∀ (cell : rectangle), cardinality (dominoes_covering cell) = 2) :
    ∃ (set1 set2 : set domino), disjoint set1 set2 ∧ covers_rectangle set1 rectangle ∧ covers_rectangle set2 rectangle :=
by
  sorry

end domino_disjoint_sets_cover_rectangle_l798_798076


namespace largest_divisor_of_m_l798_798384

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^3 = 847 * k) : ∃ d : ℕ, d = 77 ∧ ∀ x : ℕ, x > d → ¬ (x ∣ m) :=
sorry

end largest_divisor_of_m_l798_798384


namespace balls_in_boxes_l798_798913

theorem balls_in_boxes : 
  (number_of_ways : ℕ) = 52 :=
by
  let number_of_balls := 5
  let number_of_boxes := 4
  let balls_indistinguishable := true
  let boxes_distinguishable := true
  let max_balls_per_box := 3
  
  -- Proof omitted
  sorry

end balls_in_boxes_l798_798913


namespace find_ratio_of_projection_l798_798123

-- Definition of the projection matrix
def P : Matrix (Fin 2) (Fin 2) ℚ := 
  ![
    ![9/50, -15/50],
    ![-15/50, 34/50]
  ]

-- Definition of the vector
variables (a b : ℚ)

-- Statement of the theorem
theorem find_ratio_of_projection (h : P ⬝ ![a, b] = ![a, b]) : b / a = 41 / 15 :=
by {
  sorry 
}

end find_ratio_of_projection_l798_798123


namespace problem_statement_l798_798744

-- Define the operation ø
def ø (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- State the theorem
theorem problem_statement : ø (ø 4 2) 3 = 2 := by
  sorry

end problem_statement_l798_798744


namespace expression_equals_negative_two_l798_798204

def f (x : ℝ) : ℝ := x^3 - x - 1
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem expression_equals_negative_two : 
  f 2023 + f' 2023 + f (-2023) - f' (-2023) = -2 :=
by
  sorry

end expression_equals_negative_two_l798_798204


namespace trig_identity_proof_l798_798320

theorem trig_identity_proof 
  (α : ℝ) 
  (h1 : Real.sin (4 * α) = 2 * Real.sin (2 * α) * Real.cos (2 * α))
  (h2 : Real.cos (4 * α) = Real.cos (2 * α) ^ 2 - Real.sin (2 * α) ^ 2) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := 
by 
  sorry

end trig_identity_proof_l798_798320


namespace three_digit_cube_palindromes_l798_798776

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.to_digits in
  digits.reverse = digits

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem three_digit_cube_palindromes :
  {n : ℕ | is_three_digit n ∧ is_perfect_cube n ∧ is_palindrome n}.card = 1 := 
sorry

end three_digit_cube_palindromes_l798_798776


namespace angle_of_inclination_of_tangent_line_through_origin_l798_798840

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 3 = 0

noncomputable def inclination_angle (θ : ℝ) : Prop :=
  θ = Real.pi / 6 ∨ θ = 5 * Real.pi / 6

theorem angle_of_inclination_of_tangent_line_through_origin :
  ∃ θ : ℝ, inclination_angle θ ∧
  ∃ k : ℝ, k = Real.tan θ ∧ ∀ x y : ℝ, circle_equation x y →
  (y = k * x → ∀ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 3 = 0 → k * x₀ = y₀ → Real.sqrt (k^2 + 1) = 2 * Real.abs k) :=
sorry

end angle_of_inclination_of_tangent_line_through_origin_l798_798840


namespace five_aliens_have_more_limbs_than_five_martians_l798_798431

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end five_aliens_have_more_limbs_than_five_martians_l798_798431


namespace tomatoes_difference_is_50_l798_798022

variable (yesterday_tomatoes today_tomatoes total_tomatoes : ℕ)

theorem tomatoes_difference_is_50 
  (h1 : yesterday_tomatoes = 120)
  (h2 : total_tomatoes = 290)
  (h3 : total_tomatoes = today_tomatoes + yesterday_tomatoes) :
  today_tomatoes - yesterday_tomatoes = 50 := sorry

end tomatoes_difference_is_50_l798_798022


namespace point_on_graph_l798_798736

def f (x : ℝ) : ℝ := -2 * x + 3

theorem point_on_graph (x y : ℝ) : 
  ( (x = 1 ∧ y = 1) ↔ y = f x ) :=
by 
  sorry

end point_on_graph_l798_798736


namespace p_divisible_by_1979_l798_798991

def harmonic_series (n : ℕ) : ℚ :=
  if n = 0 then 0 else (1 : ℚ) / n

def alternating_harmonic_series (n : ℕ) : ℚ :=
  ∑ i in finset.range n, (-1 : ℚ)^i * harmonic_series (i+1)

theorem p_divisible_by_1979 
  (p q : ℕ) 
  (h_pos_p : 0 < p) 
  (h_pos_q : 0 < q) 
  (h_series : (p : ℚ) / q = alternating_harmonic_series 1319) :
  1979 ∣ p :=
sorry

end p_divisible_by_1979_l798_798991


namespace bus_fare_one_way_cost_l798_798635

-- Define the conditions
def zoo_entry (dollars : ℕ) : ℕ := dollars -- Zoo entry cost is $5 per person
def initial_money : ℕ := 40 -- They bring $40 with them
def money_left : ℕ := 24 -- They have $24 left after spending on zoo entry and bus fare

-- Given values
def noah_ava : ℕ := 2 -- Number of persons, Noah and Ava
def zoo_entry_cost : ℕ := 5 -- $5 per person for zoo entry
def total_money_spent := initial_money - money_left -- Money spent on zoo entry and bus fare

-- Function to calculate the total cost based on bus fare x
def total_cost (x : ℕ) : ℕ := noah_ava * zoo_entry_cost + 2 * noah_ava * x

-- Assertion to be proved
theorem bus_fare_one_way_cost : 
  ∃ (x : ℕ), total_cost x = total_money_spent ∧ x = 150 / 100 := sorry

end bus_fare_one_way_cost_l798_798635


namespace uncut_raisin_exists_l798_798580

-- Define the condition of embedded raisins at integer coordinates (using points in ℝ³ for simplicity)
def raisin_position (x y z : ℤ) : Prop :=
  true -- This represents a raisin at integer coordinates

-- Define the condition of a plane (given by its equation a*x + b*y + c*z = d)
structure Plane :=
  (a b c d : ℝ)

-- Condition: The loaf of bread is cut by several planes.
def cuts_plane (p : Plane) (x y z : ℝ) : Prop :=
  p.a * x + p.b * y + p.c * z = p.d

-- Define a theorem to state that there exists an uncut raisin
theorem uncut_raisin_exists (planes : list Plane) :
  ∃ (x y z : ℤ), raisin_position x y z ∧ 
  ∀ p ∈ planes, ¬ cuts_plane p (x : ℝ) (y : ℝ) (z : ℝ) :=
begin
  sorry -- Skip the proof, as per the instructions
end

end uncut_raisin_exists_l798_798580


namespace barbeck_steve_guitar_ratio_l798_798439

theorem barbeck_steve_guitar_ratio (b s d : ℕ) 
  (h1 : b = s) 
  (h2 : d = 3 * b) 
  (h3 : b + s + d = 27) 
  (h4 : d = 18) : 
  b / s = 2 / 1 := 
by 
  sorry

end barbeck_steve_guitar_ratio_l798_798439


namespace geometric_progression_nonzero_k_l798_798841

theorem geometric_progression_nonzero_k (k : ℝ) : k ≠ 0 ↔ (40*k)^2 = (10*k) * (160*k) := by sorry

end geometric_progression_nonzero_k_l798_798841


namespace john_out_of_pocket_l798_798964

-- Define the conditions
def computer_cost : ℕ := 700
def accessories_cost : ℕ := 200
def playstation_value : ℕ := 400
def sale_discount : ℚ := 0.2

-- Define the total cost of the computer and accessories
def total_cost : ℕ := computer_cost + accessories_cost

-- Define the selling price of the PlayStation
def selling_price : ℕ := playstation_value - (playstation_value * sale_discount).to_nat

-- Define the amount out of John's pocket
def out_of_pocket : ℕ := total_cost - selling_price

-- The proof goal
theorem john_out_of_pocket : out_of_pocket = 580 :=
by
  sorry

end john_out_of_pocket_l798_798964


namespace tenth_term_is_correct_l798_798340

-- Conditions and calculation
variable (a l : ℚ)
variable (d : ℚ)
variable (a10 : ℚ)

-- Setting the given values:
noncomputable def first_term : ℚ := 2 / 3
noncomputable def seventeenth_term : ℚ := 3 / 2
noncomputable def common_difference : ℚ := (seventeenth_term - first_term) / 16

-- Calculate the tenth term using the common difference
noncomputable def tenth_term : ℚ := first_term + 9 * common_difference

-- Statement to prove
theorem tenth_term_is_correct : 
  first_term = 2 / 3 →
  seventeenth_term = 3 / 2 →
  common_difference = (3 / 2 - 2 / 3) / 16 →
  tenth_term = 2 / 3 + 9 * ((3 / 2 - 2 / 3) / 16) →
  tenth_term = 109 / 96 :=
  by
    sorry

end tenth_term_is_correct_l798_798340


namespace smallest_angle_in_arithmetic_sequence_l798_798275

theorem smallest_angle_in_arithmetic_sequence (a : ℕ) (d : ℕ) (h1 : d ≥ 1)
  (h2 : ∀ n, n ∈ finset.range 18 → ∃ k, k ∈ range 1 ∧ k*n = a + n * d)
  (h3 : finset.sum (finset.range 18) (λ n, a + n * d) = 360) :
  a = 3 :=
by
  sorry

end smallest_angle_in_arithmetic_sequence_l798_798275


namespace hexagon_ratio_l798_798816

theorem hexagon_ratio (side_length: ℝ) (h : side_length = 2):
  let s := (side_length^2 * Real.sqrt 3) / 4
      number_of_triangles := 6
      number_of_rhomboids := 3
      r := 2 * s
  in r / s = 2 :=
by
  sorry

end hexagon_ratio_l798_798816


namespace expected_value_decisive_games_l798_798793

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l798_798793


namespace max_ON_OM_ratio_l798_798262

-- Define the conditions in Lean
def curve_C1_cartesian : (ℝ × ℝ) → Prop := λ p, p.1 + p.2 - 4 = 0
def curve_C2_parametric : ℝ → (ℝ × ℝ) := λ θ, (Real.cos θ, 1 + Real.sin θ)
def ray_l (α : ℝ) (h : 0 < α ∧ α < π / 2) : ℝ → Prop := λ ρ, true  -- ray l

-- Define the points in polar coordinates based on the conditions
def M (α : ℝ) (h : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  let ρ1 := 4 / (Real.sin α + Real.cos α)
  (ρ1, α)

def N (α : ℝ) (h : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  let ρ2 := 2 * Real.sin α
  (ρ2, α)

-- Main theorem statement in Lean
theorem max_ON_OM_ratio :
  ∃ α (hα : 0 < α ∧ α < π / 2), 
    let ρ1 := 4 / (Real.sin α + Real.cos α),
        ρ2 := 2 * Real.sin α,
        ratio := ρ2 / ρ1
    in ratio = (√2 + 1) / 4 :=
sorry

end max_ON_OM_ratio_l798_798262


namespace number_of_solutions_tan2x_eq_sin3x_l798_798129

theorem number_of_solutions_tan2x_eq_sin3x : 
  (set.countable {x ∈ set.Icc (0 : ℝ) (2 * Real.pi) | Real.tan (2 * x) = Real.sin (3 * x)} = 9) :=
sorry

end number_of_solutions_tan2x_eq_sin3x_l798_798129


namespace probability_of_drawing_three_white_balls_l798_798056

open Nat

def binom (n k : ℕ) : ℕ := nat.choose n k

def probability_of_three_white_balls (total_white total_black : ℕ) (drawn : ℕ) : ℚ :=
  (binom total_white drawn) / (binom (total_white + total_black) drawn)

theorem probability_of_drawing_three_white_balls :
  probability_of_three_white_balls 4 7 3 = 4 / 165 := by
  sorry

end probability_of_drawing_three_white_balls_l798_798056


namespace sin_ratio_in_triangle_l798_798266

theorem sin_ratio_in_triangle (a b : ℝ) (h1 : a = 2) (h2 : b = 3)
  (A B : ℝ) (triangle_ABC : ∀ A B C, A + B + C = π) :
  (sin A / sin B) = (2 / 3) :=
by
  sorry

end sin_ratio_in_triangle_l798_798266


namespace fruit_shop_purchase_maximum_value_a_l798_798719

theorem fruit_shop_purchase :
  ∃ (x y : ℝ), x + y = 200 ∧ 6 * x + 4 * y = 1020 :=
by
  use [110, 90]
  simp
  linarith

theorem maximum_value_a :
  ∃ (a : ℝ), ∀ (x y : ℝ), x = 100 ∧ y = 90 →
  (960 - 7.56 * a ≥ 771) ∧ (a ≤ 25) :=
by
  use 25
  intros x y h
  cases h with hx hy
  simp [hx, hy]
  linarith

end fruit_shop_purchase_maximum_value_a_l798_798719


namespace scalar_multiple_exists_l798_798004

variables (v : ℝ^3)
variables (i j k : ℝ^3)
variable (h_unit : i = ![1, 0, 0] ∧ j = ![0, 1, 0] ∧ k = ![0, 0, 1])

theorem scalar_multiple_exists : 
  i × (v × i) + j × (v × j) + k × (v × k) = 2 • v :=
sorry

end scalar_multiple_exists_l798_798004


namespace quadratic_as_sum_of_two_with_zero_discriminants_l798_798646

theorem quadratic_as_sum_of_two_with_zero_discriminants (c : ℝ) :
  ∃ (p q : Polynomial ℝ), p.degree = 2 ∧ q.degree = 2 ∧
  p.Coeff 2 = 0 ∧ q.Coeff 2 = 0 ∧
  (Polynomial.ofCoeff 2 (2 : ℝ) + Polynomial.ofCoeff 0 c) = p + q :=
sorry

end quadratic_as_sum_of_two_with_zero_discriminants_l798_798646


namespace mark_cans_l798_798773

/--
A local school is holding a food drive. Mark brings in 4 times as many cans as Jaydon. 
Jaydon brings in 5 more than twice the amount of cans that Rachel brought in. 
Additionally, Sophie brings in cans such that the ratio of the number of cans contributed by Mark, Jaydon, and Sophie is 4:3:2. 
If there are 225 cans total, prove that Mark brought in 100 cans.
-/
theorem mark_cans (R J M S : ℕ) 
(h1 : J = 2 * R + 5) 
(h2 : M = 4 * J) 
(h3 : (M + J + S = 225) ∧ (M : J : S) = (4 : 3 : 2)) :
M = 100 :=
sorry

end mark_cans_l798_798773


namespace area_of_region_R_l798_798279

def is_inside_square (x y : ℝ) (S : set (ℝ × ℝ)) : Prop :=
  (x, y) ∈ S

def is_closer_to_center_than_side (x y : ℝ) (S : set (ℝ × ℝ)) : Prop :=
  (x^2 + y^2) ≤ abs (x - 1) ∧
  (x^2 + y^2) ≤ abs (x + 1) ∧
  (x^2 + y^2) ≤ abs (y - 1) ∧
  (x^2 + y^2) ≤ abs (y + 1)

def region_R (x y : ℝ) (S : set (ℝ × ℝ)) : Prop :=
  is_inside_square x y S ∧ is_closer_to_center_than_side x y S

theorem area_of_region_R :
  let S := {p : ℝ × ℝ | p.1 = 1 ∨ p.1 = -1 ∨ p.2 = 1 ∨ p.2 = -1} in
  (∃ R : set (ℝ × ℝ), ∀ (x y : ℝ),
    (region_R x y S) ↔ (x, y) ∈ R ∧ R.area = 2) :=
begin
  sorry
end

end area_of_region_R_l798_798279


namespace krypton_distance_is_six_l798_798124

noncomputable def distance_from_sun_to_krypton_one_quarter (krypton_orbit : Ellipse)
    (sun_focus : krypton_orbit.focuses)
    (krypton_perigee : 3 = krypton_orbit.distance_from_sun_to_perigee sun_focus)
    (krypton_apogee : 9 = krypton_orbit.distance_from_sun_to_apogee sun_focus) : Prop :=
  krypton_orbit.distance_to_point_on_orbit sun_focus (1 / 4) = 6

theorem krypton_distance_is_six 
    (krypton_orbit : Ellipse)
    (sun_focus : krypton_orbit.focuses)
    (krypton_perigee : 3 = krypton_orbit.distance_from_sun_to_perigee sun_focus)
    (krypton_apogee : 9 = krypton_orbit.distance_from_sun_to_apogee sun_focus) :
  distance_from_sun_to_krypton_one_quarter krypton_orbit sun_focus krypton_perigee krypton_apogee := 
  sorry

end krypton_distance_is_six_l798_798124


namespace problem_l798_798461

theorem problem 
  (x : ℝ) 
  (h1 : x ∈ Set.Icc (-3 : ℝ) 3) 
  (h2 : x ≠ -5/3) : 
  (4 * x ^ 2 + 2) / (5 + 3 * x) ≥ 1 ↔ x ∈ (Set.Icc (-3) (-3/4) ∪ Set.Icc 1 3) :=
sorry

end problem_l798_798461


namespace final_position_furthest_distance_total_fuel_consumed_l798_798669

-- Define the list of travel distances
def travelDistances : List ℤ := [8, -9, 6, -12, -6, 15, -11, 3]

-- Define the fuel consumption rate per km
def fuelConsumptionRate : ℝ := 0.15

theorem final_position :
  (travelDistances.sum = -6) :=
by
  sorry

theorem furthest_distance :
  (List.map (List.scanl (fun acc d => acc + d) 0 travelDistances) abs max = 13) :=
by
  sorry

theorem total_fuel_consumed :
  (fuelConsumptionRate * (List.map abs travelDistances).sum = 10.5) :=
by
  sorry

end final_position_furthest_distance_total_fuel_consumed_l798_798669


namespace three_digit_cube_palindromes_l798_798775

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.to_digits in
  digits.reverse = digits

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem three_digit_cube_palindromes :
  {n : ℕ | is_three_digit n ∧ is_perfect_cube n ∧ is_palindrome n}.card = 1 := 
sorry

end three_digit_cube_palindromes_l798_798775


namespace initial_quantities_max_a_l798_798718

/- Given conditions for part 1 -/

def price_per_kg_apples : ℝ := 15
def price_per_kg_pears : ℝ := 10
def total_weight : ℝ := 200
def profit_percentage_apples : ℝ := 0.4
def ratio_pear_price_to_apple_price : ℝ := 2 / 3
def total_profit : ℝ := 1020

/- Given conditions for part 2 -/

def reduction_percentage_apple_sell_price : ℝ → ℝ := λ a, 3 / 5 * a / 100
def increment_percentage_pear_sell_price : ℝ → ℝ := λ a, 2 / 5 * a / 100
def minimum_profit : ℝ := 771
def initial_apples_purchased : ℝ := 110
def initial_pears_purchased : ℝ := 90

/- Lean theorem statements -/

/- Part 1 -/
theorem initial_quantities (x y : ℝ) 
  (hx : price_per_kg_apples * (1 + profit_percentage_apples) * x + (price_per_kg_apples * (1 + profit_percentage_apples) * ratio_pear_price_to_apple_price - price_per_kg_pears) * y = total_profit)
  (h1 : x + y = total_weight) : 
  x = 110 ∧ y = 90 :=
sorry

/- Part 2 -/
theorem max_a (a : ℝ) 
  (hx1 : initial_apples_purchased * (price_per_kg_apples * (1 + profit_percentage_apples) * (1 - reduction_percentage_apple_sell_price a) - price_per_kg_apples)
        + initial_pears_purchased * (price_per_kg_apples * (1 + profit_percentage_apples) * ratio_pear_price_to_apple_price * (1 + increment_percentage_pear_sell_price a) - price_per_kg_pears)
        ≥ minimum_profit) : 
  a ≤ 25 :=
sorry

end initial_quantities_max_a_l798_798718


namespace count_solutions_l798_798551

theorem count_solutions :
  ∃ (n : ℕ), (∀ (x y z : ℕ), x * y * z + x * y + y * z + z * x + x + y + z = 2012 ↔ n = 27) :=
sorry

end count_solutions_l798_798551


namespace magnitude_of_error_l798_798075

theorem magnitude_of_error (x : ℝ) (hx : 0 < x) :
  abs ((4 * x) - (x / 4)) / (4 * x) * 100 = 94 := 
sorry

end magnitude_of_error_l798_798075


namespace exists_special_integer_l798_798862

-- Define the mathematical conditions and the proof
theorem exists_special_integer (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) : 
  ∃ x : ℕ, 
    (∀ p ∈ P, ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) ∧
    (∀ p ∉ P, ¬∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) :=
sorry

end exists_special_integer_l798_798862


namespace nonnegative_intervals_l798_798826

noncomputable def expression (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^2)

theorem nonnegative_intervals :
  { x : ℝ | expression x ≥ 0 } = { x : ℝ | x ∈ set.Icc (-3 : ℝ) (0 : ℝ) ∪ set.Icc (0 : ℝ) (1 / 4 : ℝ) } :=
by
  sorry

end nonnegative_intervals_l798_798826


namespace tomatoes_multiplier_l798_798940

theorem tomatoes_multiplier (before_vacation : ℕ) (grown_during_vacation : ℕ)
  (h1 : before_vacation = 36)
  (h2 : grown_during_vacation = 3564) :
  (before_vacation + grown_during_vacation) / before_vacation = 100 :=
by
  -- Insert proof here later
  sorry

end tomatoes_multiplier_l798_798940


namespace fifth_row_odd_count_l798_798432

open Nat

/-- We start with natural numbers from 1 to 100 (inclusive). 
For each subsequent row, we compute the product of the digits 
of the numbers in the previous row until we reach the fifth row. 
-/
def digit_product (n : Nat) : Nat :=
  (toDigits 10 n).foldl (· * ·) 1

def generate_next_row (row : List Nat) : List Nat :=
  row.map digit_product

def generate_rows (n : Nat) : List (List Nat) :=
  Nat.recOn n [List.range' 1 100]
    (λ _ rows, generate_next_row rows.head :: rows)

def count_odd_numbers (l : List Nat) : Nat :=
  l.filter (λ n, n % 2 = 1).length

theorem fifth_row_odd_count : count_odd_numbers (generate_rows 4).head = 19 := 
by
  sorry

end fifth_row_odd_count_l798_798432


namespace cube_root_of_27_l798_798106

theorem cube_root_of_27 : real.cbrt 27 = 3 :=
by
  sorry

end cube_root_of_27_l798_798106


namespace triangle_angle_BED_eq_52_5_l798_798572

-- Definitions based on conditions
variables {A B C D E : Type} {α : obtuse_angle_triangle A B C}
variable (DB_eq_BE : length D B = length B E)
variables (angle_A_45 : angle A = 45) (angle_C_60 : angle C = 60)

-- The statement to be proven
theorem triangle_angle_BED_eq_52_5 :
  angle B = 75 ∧
  is_isosceles_triangle D B E ∧
  angle BDE = x ∧
  angle BED = x
  →
  angle BED = 52.5 :=
sorry

end triangle_angle_BED_eq_52_5_l798_798572


namespace ratio_of_x_to_y_l798_798110

variable {x y : ℝ}

theorem ratio_of_x_to_y (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l798_798110


namespace no_integer_solution_k_range_l798_798925

theorem no_integer_solution_k_range (k : ℝ) :
  (∀ x : ℤ, ¬ ((k * x - k^2 - 4) * (x - 4) < 0)) → (1 ≤ k ∧ k ≤ 4) :=
by
  sorry

end no_integer_solution_k_range_l798_798925


namespace analytic_expression_of_f_value_of_a_and_range_of_g_range_of_a_l798_798211

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := a * x^2 - 4 * x + 2

-- Define the function g
def g (a x : ℝ) : ℝ := (1/3)^(f a x)

-- First proof problem
theorem analytic_expression_of_f (a : ℝ) :
  (∀ x, f a (2 - x) = f a (2 + x)) →
  f a x = x^2 - 4 * x + 2 :=
sorry

-- Second proof problem
theorem value_of_a_and_range_of_g (a : ℝ) :
  (∀ x, g a x ≤ 9) →
  (∃ a, a = 1 ∧ ∀ x, g a x ∈ set.Ioc 0 9) :=
sorry

-- Third proof problem
theorem range_of_a (a : ℝ) :
  (a ≤ 1) →
  (∀ x ∈ set.Icc 1 2, f a x - log 2 (x / 8) = 0) →
  (-1 ≤ a ∧ a ≤ 1) :=
sorry

end analytic_expression_of_f_value_of_a_and_range_of_g_range_of_a_l798_798211


namespace find_num_valid_n_l798_798845

theorem find_num_valid_n : 
  {n : ℤ | n ≥ 2 ∧ 2013 % n = n % 3 }.to_finset.card = 7 :=
  sorry

end find_num_valid_n_l798_798845


namespace value_of_k_l798_798163

theorem value_of_k :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x →  (x^2 + 7 * x + 10) = (k + 3 * x)) ∧
  ∃ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ∧ discriminant (x^2 + 4 * x + (10 - k)) = 0 :=
begin
  use 6,
  sorry
end

end value_of_k_l798_798163


namespace volume_of_tetrahedron_proof_l798_798142

noncomputable def volume_of_tetrahedron (ABCD : Type)
  (ABC BCD : Set ABCD)
  (angle_ABC_BCD : Real)
  (area_ABC area_BCD : Real)
  (BC : Real) : Real :=
if angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 100 ∧ BC = 12
then (1250 * Real.sqrt 2) / 3
else 0

theorem volume_of_tetrahedron_proof (ABCD : Type)
  (D : ABCD)
  (ABC BCD : Set ABCD)
  (angle_ABC_BCD : Real)
  (area_ABC area_BCD : Real)
  (BC : Real)
  (h : volume_of_tetrahedron ABCD ABC BCD angle_ABC_BCD area_ABC area_BCD BC = (1250 * Real.sqrt 2) / 3) :
  h =
if angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 100 ∧ BC = 12
then (1250 * Real.sqrt 2) / 3
else 0 :=
by
  sorry

end volume_of_tetrahedron_proof_l798_798142


namespace combined_weight_is_36_6_l798_798273

noncomputable def combined_box_weight : ℝ :=
let box1 := 2.5 in
let box2 := 11.3 in
let box3 := 5.75 in
let box4 := 7.2 in
let box5 := 3.25 in
let box6_kg := 2 in
let box7_oz := 35 in
let kg_to_pounds := (kg : ℝ) => kg / 0.4536 in
let oz_to_pounds := (oz : ℝ) => oz / 16 in
let box6 := kg_to_pounds box6_kg in
let box7 := oz_to_pounds box7_oz in
box1 + box2 + box3 + box4 + box5 + box6 + box7

theorem combined_weight_is_36_6 : combined_box_weight = 36.6 :=
by
  -- Convert 2 kg to pounds
  have h1 : 2 / 0.4536 = 2.20462 * 2, by approx_num 5 2.20462 * 2
  have box6_conv : 2 / 0.4536 ≈ 4.40924, by approx_num 2 * 2.20462
  
  -- Convert 35 ounces to pounds
  have h2 : 35 / 16 = 2.1875, by norm_num
  have box7_conv : 35 / 16 ≈ 2.1875, by norm_num
  
  -- Add all values
  calc combined_box_weight
      = 2.5 + 11.3 + 5.75 + 7.2 + 3.25 + 4.40924 + 2.1875 : by unfold combined_box_weight; norm_num
  ...
  = 36.59674 : by norm_num
  ... ≈ 36.6 : sorry

end combined_weight_is_36_6_l798_798273


namespace trapezoid_BC_perpendicular_diagonals_l798_798263

theorem trapezoid_BC_perpendicular_diagonals
  (AB CD BC BD AC : ℝ)
  (h1 : BC = 57 * (10:ℝ).sqrt)
  (h2 : AB = (19:ℝ).sqrt)
  (h3 : AD = (1901:ℝ).sqrt)
  (h4 : ∀ x y, right_angle x y)
  (h5 : ∀ x y, right_angle x y) :
  BC^2 = 570 := 
sorry

end trapezoid_BC_perpendicular_diagonals_l798_798263


namespace sum_of_reciprocals_le_equality_sequence_l798_798973

noncomputable def distinct_sums (k : ℕ) (a : fin k → ℕ) :=
  ∀ (x y : fin k → bool), (∑ i, (if x i then a i else 0)) = (∑ i, (if y i then a i else 0)) → x = y

theorem sum_of_reciprocals_le (k : ℕ) (a : fin k → ℕ) (ha_distinct: distinct_sums k a) :
  (∑ i in finset.range k, 1 / (a i : ℝ)) ≤ 2 * (1 - 2^(-k : ℝ)) :=
sorry

theorem equality_sequence (k : ℕ) :
  (∑ i in finset.range k, 1 / (2^i : ℝ)) = 2 * (1 - 2^(-k : ℝ)) :=
sorry

end sum_of_reciprocals_le_equality_sequence_l798_798973


namespace complex_sum_500_l798_798615

theorem complex_sum_500 (x : ℂ) (h1 : x ^ 1001 = 1) (h2 : x ≠ 1) :
  (∑ k in (Finset.range 1000).map (Finset.natEmb 1), x ^ (3 * k) / (x ^ k - 1)) = 500 :=
by
  sorry

end complex_sum_500_l798_798615


namespace y_coord_equidistant_l798_798027

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (3, 0) = dist (0, y) (1, -6)) ↔ y = -7 / 3 :=
by
  sorry

end y_coord_equidistant_l798_798027


namespace bisector_angle_COE_midpoint_arc_AB_l798_798927

theorem bisector_angle_COE_midpoint_arc_AB
  (O : Point)
  (A B C D E : Point)
  (h_circle : Circle O)
  (h_diams_perp : IsDiameter AB ∧ IsDiameter CD ∧ Perpendicular AB CD)
  (h_E_on_circle : OnCircle E O)
  (h_perpendicular_CE_CD : Perpendicular (LineOfPoints C E) (LineOfPoints C D))
  (h_chord_perpendicular_CD : Perpendicular (LineOfPoints E F) (LineOfPoints C D))
  : (Bisector (Angle COE)).intersection (Circle O) = Midpoint (Arc AB) :=
sorry

end bisector_angle_COE_midpoint_arc_AB_l798_798927


namespace smallest_integer_y_l798_798372

-- Define the condition
def condition (y : ℤ) : Prop := 3 * y - 4 > 2 * y + 5

-- State the proof statement
theorem smallest_integer_y : ∃ y : ℤ, condition y ∧ ∀ z : ℤ, condition z → z ≥ y :=
begin
  use 10,
  split,
  { -- Prove that y = 10 satisfies the condition
    show condition 10,
    sorry,
  },
  { -- Prove that 10 is the smallest integer that satisfies the condition
    assume z,
    show condition z → z ≥ 10,
    sorry,
  }
end

end smallest_integer_y_l798_798372


namespace weird_subset_count_l798_798600

def is_power_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2^k

def is_weird_subset (A B : Finset ℕ) : Prop :=
  ∀ x y ∈ B, x + y ∈ (A : Finset ℕ) → is_power_of_two (x + y) → x ∈ A ∨ y ∈ A

theorem weird_subset_count (n : ℕ) (h : n > 1) (B : Finset ℕ) (hB : B = Finset.range (2^n + 1)) :
  (∃! A ⊆ B, is_weird_subset A B)  :=
sorry

end weird_subset_count_l798_798600


namespace part_one_part_two_l798_798282

-- Part (1)
theorem part_one (m : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → (2 * m < x ∧ x < 1 → -1 ≤ x ∧ x ≤ 2 ∧ - (1 / 2) ≤ m)) → 
  (m ≥ - (1 / 2)) :=
by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ x : ℤ, (2 * m < x ∧ x < 1) ∧ (x < -1 ∨ x > 2)) ∧ 
  (∀ y : ℤ, (2 * m < y ∧ y < 1) ∧ (y < -1 ∨ y > 2) → y = x) → 
  (- (3 / 2) ≤ m ∧ m < -1) :=
by sorry

end part_one_part_two_l798_798282


namespace imaginary_unit_multiplication_l798_798396

theorem imaginary_unit_multiplication (i : ℂ) (h1 : i * i = -1) : i * (1 + i) = i - 1 :=
by
  sorry

end imaginary_unit_multiplication_l798_798396


namespace vinegar_in_cupboard_for_2_years_l798_798799

-- Given conditions
variable (n : ℝ)
variable (h₁ : ∀ V : ℝ, V > 0 → V * (0.8 ^ n) = 0.64 * V)

-- Lean 4 statement
theorem vinegar_in_cupboard_for_2_years : n = 2 :=
by
  -- Proof is not required, so we use sorry
  sorry

end vinegar_in_cupboard_for_2_years_l798_798799


namespace cupcakes_frosted_l798_798807

theorem cupcakes_frosted (t₁ t₂ b n total_time : ℕ) 
  (h₁ : t₁ = 15) (h₂ : t₂ = 40) (h₃ : b = 10) (h₄ : n = 10) (h₅ : total_time = 600) :
  let combined_rate := t₁ * t₂ / (t₁ + t₂),
      cycle_time := n * combined_rate + b in
  total_time / cycle_time * n = 50 :=
by sorry

end cupcakes_frosted_l798_798807


namespace eval_expression_eq_neg_two_l798_798441

def eval_expression : ℝ :=
  (1) * (-1)^3 + (1 / 7) * (2 - (-3)^2)

theorem eval_expression_eq_neg_two : eval_expression = -2 := 
  sorry

end eval_expression_eq_neg_two_l798_798441


namespace find_k_l798_798665

constant a : ℕ → ℕ
constant b : ℕ → ℕ

axiom a_1 : a 1 = 1
axiom a_recurrence {n : ℕ} (hn : 2 ≤ n) : a n = (finset.range (n - 1)).sum (λ i => (i + 1) * a (n - (i + 1)))

axiom b_definition {n : ℕ} : b n = finset.range n.sum (λ i => a (i + 1))

theorem find_k : (finset.range 2021).sum (λ i => b (i + 1)) = a 2022 :=
sorry

end find_k_l798_798665


namespace rectangle_square_ratio_l798_798326

theorem rectangle_square_ratio (s x y : ℝ) (h1 : 0.1 * s ^ 2 = 0.25 * x * y) (h2 : y = s / 4) :
  x / y = 6 := 
sorry

end rectangle_square_ratio_l798_798326


namespace current_year_2021_l798_798805

variables (Y : ℤ)

def parents_moved_to_America := 1982
def Aziz_age := 36
def years_before_born := 3

theorem current_year_2021
  (h1 : parents_moved_to_America = 1982)
  (h2 : Aziz_age = 36)
  (h3 : years_before_born = 3)
  (h4 : Y - (Aziz_age) - (years_before_born) = 1982) : 
  Y = 2021 :=
by {
  sorry
}

end current_year_2021_l798_798805


namespace angle_between_skew_lines_l798_798535

-- Define the two direction vectors a and b
def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (1, 0, -1)

-- Prove the angle between the vectors is pi/3
theorem angle_between_skew_lines (a b : ℝ × ℝ × ℝ)
  (ha : a = vector_a)
  (hb : b = vector_b) :
  -- The angle between the vectors a and b is pi/3
  real.angle a b = real.pi / 3 := sorry

end angle_between_skew_lines_l798_798535


namespace number_of_elements_in_M_inter_N_is_zero_l798_798214

noncomputable def M : set (ℤ × ℤ) := 
  {p | p.1 ^ 2 + p.2 ^ 2 = 1}

noncomputable def N : set (ℤ × ℤ) := 
  {p | (p.1 - 1) ^ 2 + p.2 ^ 2 = 1}

theorem number_of_elements_in_M_inter_N_is_zero :
  fintype.card (M ∩ N) = 0 :=
sorry

end number_of_elements_in_M_inter_N_is_zero_l798_798214


namespace union_of_A_and_B_l798_798624

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_A_and_B_l798_798624


namespace two_x_equals_y_l798_798882

theorem two_x_equals_y (x y : ℝ) (h1 : (x + y) / 3 = 1) (h2 : x + 2 * y = 5) : 2 * x = y := 
by
  sorry

end two_x_equals_y_l798_798882


namespace _l798_798096

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * real.sqrt 3) / 4

noncomputable theorem decrease_in_area_of_equilateral_triangle :
  (∃ (s : ℝ), area_of_equilateral_triangle s = 81 * real.sqrt 3) →
  ∀ s, area_of_equilateral_triangle s = 81 * real.sqrt 3 →
        area_of_equilateral_triangle (s - 3) = 56.25 * real.sqrt 3 →
          (81 * real.sqrt 3 - 56.25 * real.sqrt 3) = 24.75 * real.sqrt 3 :=
  by
    sorry

end _l798_798096


namespace taxi_ride_cost_l798_798422

theorem taxi_ride_cost (base_fare : ℝ) (rate_per_mile : ℝ) (additional_charge : ℝ) (distance : ℕ) (cost : ℝ) :
  base_fare = 2 ∧ rate_per_mile = 0.30 ∧ additional_charge = 5 ∧ distance = 12 ∧ 
  cost = base_fare + (rate_per_mile * distance) + additional_charge → cost = 10.60 :=
by
  intros
  sorry

end taxi_ride_cost_l798_798422


namespace pieces_per_package_l798_798650

-- Definitions from conditions
def total_pieces_of_gum : ℕ := 486
def number_of_packages : ℕ := 27

-- Mathematical statement to prove
theorem pieces_per_package : total_pieces_of_gum / number_of_packages = 18 := sorry

end pieces_per_package_l798_798650


namespace angle_between_a_b_correct_m_l798_798286

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (m : ℝ)
variables (OA OB OC : V)

/-- Given conditions as hypotheses --/
def conditions :=
  non_collinear a b ∧ 
  OA = m • a - b ∧ 
  OB = (m + 1) • a + b ∧ 
  OC = a - 3 • b

/-- Question 1: Given additional conditions, prove the angle between a and b --/
theorem angle_between_a_b 
  (h : conditions a b m OA OB OC) 
  (hm : m = -1 / 2) 
  (ha : ∥a∥ = 2 * real.sqrt 2 * ∥b∥) 
  (h_ortho : inner_product_space.dot_product OB OC = 0) : 
  real.angle a b = real.pi / 4 :=
sorry

/-- Question 2: Given collinearity condition, prove the value of m --/
theorem correct_m 
  (h : conditions a b m OA OB OC) 
  (h_collinear : collinear ℝ ![OA, OB, OC]) :
  m = 2 :=
sorry

end angle_between_a_b_correct_m_l798_798286


namespace b_is_geometric_sum_of_a_n_l798_798183

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 3/2
| (n+1) := 3 * (a n) - 1

-- Define the sequence b_n from the sequence a_n
def b (n : ℕ) : ℝ := a n - 1/2

-- Part (1): Prove that b_n is a geometric sequence
theorem b_is_geometric : ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = q * (b n) :=
by
  use 3
  sorry

-- Part (2): Find the sum of the first n terms of a_n
def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

theorem sum_of_a_n (n : ℕ) : S n = (3^n + n - 1) / 2 :=
by
  sorry

end b_is_geometric_sum_of_a_n_l798_798183


namespace sum_of_four_consecutive_integers_is_prime_l798_798443

theorem sum_of_four_consecutive_integers_is_prime :
  ∃ (n : ℤ), Nat.Prime (Int.to_nat (n + (n + 1) + (n + 2) + (n + 3))) :=
by
  sorry

end sum_of_four_consecutive_integers_is_prime_l798_798443


namespace sum_of_squares_l798_798664

def positive_integers (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

def sum_of_values (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧ Int.gcd x y + Int.gcd y z + Int.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h1 : positive_integers x y z) (h2 : sum_of_values x y z) :
  x^2 + y^2 + z^2 = 296 :=
by sorry

end sum_of_squares_l798_798664


namespace student_scores_4_marks_per_correct_answer_l798_798937

constant total_questions : ℕ := 60
constant total_scores : ℕ := 140
constant correct_questions : ℕ := 40
constant lost_mark_per_wrong : ℕ := 1

def marks_per_correct_answer (x : ℕ) : Prop :=
  correct_questions * x - (total_questions - correct_questions) * lost_mark_per_wrong = total_scores

theorem student_scores_4_marks_per_correct_answer : ∃ x : ℕ, marks_per_correct_answer x ∧ x = 4 := 
sorry

end student_scores_4_marks_per_correct_answer_l798_798937


namespace trig_identity_simplification_l798_798656

theorem trig_identity_simplification (x : ℝ) (h : sin x ≠ 0) : 
    (\frac{sin x}{1 - cos x} + \frac{1 - cos x}{sin x} = 2 * csc x) :=
by
  have pyth_identity : sin x ^ 2 + cos x ^ 2 = 1 := sorry
  sorry

end trig_identity_simplification_l798_798656


namespace divisors_and_primes_l798_798911

-- Define what it means to be a divisor of 36
def is_divisor_of_36 (d : ℤ) : Prop := 36 % d = 0

-- Define what it means to be a prime number
def is_prime (p : ℤ) : Prop := Nat.Prime (Int.natAbs p)

-- Main statement to prove
theorem divisors_and_primes (d : ℤ) :
  (d.count(is_divisor_of_36) = 18) ∧
  (d.count(λ x, is_divisor_of_36 x ∧ is_prime x) = 4) := by
  sorry

end divisors_and_primes_l798_798911


namespace isosceles_triangle_count_l798_798449

noncomputable def valid_points : List (ℕ × ℕ) :=
  [(2, 5), (5, 5)]

theorem isosceles_triangle_count 
  (A B : ℕ × ℕ) 
  (H_A : A = (2, 2)) 
  (H_B : B = (5, 2)) : 
  valid_points.length = 2 :=
  sorry

end isosceles_triangle_count_l798_798449


namespace sums_correct_l798_798381

theorem sums_correct (x : ℕ) (h : x + 2 * x = 48) : x = 16 :=
by
  sorry

end sums_correct_l798_798381


namespace dad_salmons_caught_l798_798544

theorem dad_salmons_caught (hazel_catch : ℕ) (total_catch : ℕ) (H1 : hazel_catch = 24) (H2 : total_catch = 51) : total_catch - hazel_catch = 27 :=
by
  rw [H1, H2]
  sorry

end dad_salmons_caught_l798_798544


namespace tea_blend_ratio_l798_798414

theorem tea_blend_ratio (x y : ℝ)
  (h1 : 18 * x + 20 * y = (21 * (x + y)) / 1.12)
  (h2 : x + y ≠ 0) :
  x / y = 5 / 3 :=
by
  -- proof will go here
  sorry

end tea_blend_ratio_l798_798414


namespace last_integer_in_geometric_sequence_l798_798353

theorem last_integer_in_geometric_sequence (a : ℕ) (r : ℚ) (h_a : a = 2048000) (h_r : r = 1/2) : 
  ∃ n : ℕ, (a : ℚ) * (r^n : ℚ) = 125 := 
by
  sorry

end last_integer_in_geometric_sequence_l798_798353


namespace mean_of_squares_of_first_four_odd_numbers_l798_798808

theorem mean_of_squares_of_first_four_odd_numbers :
  (1^2 + 3^2 + 5^2 + 7^2) / 4 = 21 := 
by
  sorry

end mean_of_squares_of_first_four_odd_numbers_l798_798808


namespace stream_speed_l798_798039

variables (v : ℝ) (swimming_speed : ℝ) (ratio : ℝ)

theorem stream_speed (hs : swimming_speed = 4.5) (hr : ratio = 2) (h : (swimming_speed - v) / (swimming_speed + v) = 1 / ratio) :
  v = 1.5 :=
sorry

end stream_speed_l798_798039


namespace valid_patents_growth_l798_798089

variable (a b : ℝ)

def annual_growth_rate : ℝ := 0.23

theorem valid_patents_growth (h1 : b = (1 + annual_growth_rate)^2 * a) : b = (1 + 0.23)^2 * a :=
by
  sorry

end valid_patents_growth_l798_798089


namespace number_of_four_digit_integers_with_digit_sum_nine_l798_798159

theorem number_of_four_digit_integers_with_digit_sum_nine :
  ∃ (n : ℕ), (n = 165) ∧ (
    ∃ (a b c d : ℕ), 
      1 ≤ a ∧ 
      a + b + c + d = 9 ∧ 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (0 ≤ b ∧ b ≤ 9) ∧ 
      (0 ≤ c ∧ c ≤ 9) ∧ 
      (0 ≤ d ∧ d ≤ 9)) := 
sorry

end number_of_four_digit_integers_with_digit_sum_nine_l798_798159


namespace vector_b_is_sqrt2_sqrt2_l798_798219

theorem vector_b_is_sqrt2_sqrt2
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (1, 1))
  (h_b_mag : real.sqrt (b.1^2 + b.2^2) = 2)
  (h_parallel : ∃ λ : ℝ, b = (λ * a.1, λ * a.2) ∧ λ > 0)
  (h_same_dir : ∃ λ : ℝ, b = (λ * a.1, λ * a.2)) :
  b = (real.sqrt 2, real.sqrt 2) :=
by
  sorry

end vector_b_is_sqrt2_sqrt2_l798_798219


namespace necessary_but_not_sufficient_l798_798301

variables {x : ℝ}

def p : Prop := x < 3
def q : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : (p → q) ∧ (q → p) :=
by
  split
  sorry

end necessary_but_not_sufficient_l798_798301


namespace max_packages_l798_798998

def initial_cupcakes : ℕ := 50
def cupcakes_eaten : ℕ := 5 + 3 + 2
def cupcakes_contributed : ℕ := 15 + 20
def packages_per_package : ℝ := 4.5

theorem max_packages :
  (⌊ (initial_cupcakes - cupcakes_eaten + cupcakes_contributed) / packages_per_package ⌋ : ℕ) = 16 :=
by
  sorry

end max_packages_l798_798998


namespace incorrectStatementD_l798_798375

def StatementA : Prop := ∀ (dE : Type), (dE = "informational Earth") → (dE = "virtual counterpart of the Earth")
def StatementB : Prop := ∀ (dE infoModel : Type), (dE = "organizes info by geo coords") → (infoModel = "global info model")
def StatementC : Prop := ∀ (dE : Type), (dE = "technical system digitizes Earth's info") → (dE = "managed by computer networks")
def StatementD : Prop := ∀ (core : Type), (core = "use numbers to deal with all aspects")

theorem incorrectStatementD : StatementA ∧ StatementB ∧ StatementC → ¬ StatementD :=
by
  intros conditions
  sorry

end incorrectStatementD_l798_798375


namespace minimum_value_Q_l798_798990

theorem minimum_value_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 47 := 
  sorry

end minimum_value_Q_l798_798990


namespace andrei_cannot_ensure_victory_l798_798365

theorem andrei_cannot_ensure_victory :
  ∀ (juice_andrew : ℝ) (juice_masha : ℝ),
    juice_andrew = 24 * 1000 ∧
    juice_masha = 24 * 1000 ∧
    ∀ (andrew_mug : ℝ) (masha_mug1 : ℝ) (masha_mug2 : ℝ),
      andrew_mug = 500 ∧
      masha_mug1 = 240 ∧
      masha_mug2 = 240 ∧
      (¬ (∃ (turns_andrew turns_masha : ℕ), 
        turns_andrew * andrew_mug > 48 * 1000 / 2 ∨
        turns_masha * (masha_mug1 + masha_mug2) > 48 * 1000 / 2)) := sorry

end andrei_cannot_ensure_victory_l798_798365


namespace Derek_test_score_l798_798222

def Grant_score (John_score : ℕ) : ℕ := John_score + 10
def John_score (Hunter_score : ℕ) : ℕ := 2 * Hunter_score
def Hunter_score : ℕ := 45
def Sarah_score (Grant_score : ℕ) : ℕ := Grant_score - 5
def Derek_score (John_score Grant_score : ℕ) : ℕ := (John_score + Grant_score) / 2

theorem Derek_test_score :
  Derek_score (John_score Hunter_score) (Grant_score (John_score Hunter_score)) = 95 :=
  by
  -- proof here
  sorry

end Derek_test_score_l798_798222


namespace PQRS_area_l798_798643

-- Definitions related to the problem conditions
def square (a : ℝ) := a ^ 2
def side_length_square (area : ℝ) := real.sqrt area

-- Given conditions
def conditions : Prop :=
  EFGH_area = 36 ∧ 
  ∀ a b, equilateral_triangle (side_length_square EFGH_area) a b ∧ 
  PQR_side = (side_length_square EFGH_area) + (2 * (side_length_square EFGH_area) * (real.sqrt 3) / 2)

-- Goal to prove
theorem PQRS_area : conditions → square PQR_side = 144 + 72 * real.sqrt 3 :=
by sorry

end PQRS_area_l798_798643


namespace non_decreasing_iff_deriv_non_negative_l798_798046

variable {α : Type} [LinearOrder α] [TopologicalSpace α]
variable (a b : α) (f : α → ℝ)

-- The statement for part (a)
theorem non_decreasing_iff_deriv_non_negative :
  (∀ x ∈ Ico a b, differentiable_at ℝ f x) →
  ((∀ x ∈ Ioc a b, deriv f x ≥ 0) ↔ (∀ x y ∈ Icc a b, x ≤ y → f x ≤ f y)) :=
sorry

end non_decreasing_iff_deriv_non_negative_l798_798046


namespace mikes_lower_rate_l798_798633

theorem mikes_lower_rate (x : ℕ) (high_rate : ℕ) (total_paid : ℕ) (lower_payments : ℕ) (higher_payments : ℕ)
  (h1 : high_rate = 310)
  (h2 : total_paid = 3615)
  (h3 : lower_payments = 5)
  (h4 : higher_payments = 7)
  (h5 : lower_payments * x + higher_payments * high_rate = total_paid) :
  x = 289 :=
sorry

end mikes_lower_rate_l798_798633


namespace jenny_phone_number_possibilities_l798_798433

def is_georgetown_number (num : String) : Prop :=
  num.length = 6 ∧ num.take 2 == "81"

def count_possibilities (prefix : String) (scrap : String) : Nat :=
  if prefix.length + scrap.length = 6 then 1 else
  if prefix.length + scrap.length < 6 then
    (10 ^ (6 - (prefix.length + scrap.length))).toNat else 0

theorem jenny_phone_number_possibilities :
  ∃ (total_possibilities : Nat), is_georgetown_number "81abcd" →
  let case1 := count_possibilities "81" "10"
  let case2 := count_possibilities "818" "101"
  let case3 := 1 -- "81" completely followed by "1018"
  total_possibilities = case1 + case2 + case3 :=
  sorry

end jenny_phone_number_possibilities_l798_798433


namespace number_of_possible_ordered_pairs_l798_798451

def round_table (seats : ℕ) := seats = 5

def sitting_next_to_at_least_one (gender : Type) := {p : ℕ // p ≥ 0}

def pairs_count : set (ℕ × ℕ) :=
  {(0, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 2), (5, 0)}

theorem number_of_possible_ordered_pairs : 
  ∀ (seats : ℕ) (f m : {p // p ≥ 0}),
  round_table seats →
  (sitting_next_to_at_least_one f) →
  (sitting_next_to_at_least_one m) →
  finset.card pairs_count = 7 := 
by
  intros seats f m h_seats h_f h_m
  sorry

end number_of_possible_ordered_pairs_l798_798451


namespace fraction_invariant_l798_798241

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l798_798241


namespace area_of_square_BCFE_eq_2304_l798_798660

-- Definitions of points and side lengths as per the conditions given in the problem
variables (A B C D E F G : Type*) [euclidean_space A] [euclidean_space B] 
[euclidean_space C] [euclidean_space D] [euclidean_space E] 
[euclidean_space F] [euclidean_space G]

-- Definitions of side lengths
def AB : ℝ := 36
def CD : ℝ := 64

def side_length_of_square (x : ℝ) := x * x 

-- The goal is to prove that the area of square BCFE equals 2304
theorem area_of_square_BCFE_eq_2304 (x : ℝ) 
  (h1: similarity (triangle A B G) (triangle F D C))
  (h2: AB = 36)
  (h3: CD = 64)
  : side_length_of_square x = 2304 := sorry

end area_of_square_BCFE_eq_2304_l798_798660


namespace smallest_positive_period_l798_798824

theorem smallest_positive_period (x : ℝ) : 
  has_period (λ x : ℝ, sin (2 * x) + cos (2 * x)) π := 
sorry

end smallest_positive_period_l798_798824


namespace total_dolls_48_l798_798543

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l798_798543


namespace continued_fraction_l798_798645

def K_0 : ℕ := 1

def K_1 (x₁ : ℕ) : ℕ := x₁

def K (n : ℕ) : (fin n.succ → ℕ) → ℕ 
| ⟨0, _⟩ => 1 
| ⟨1, _⟩ => x 
| (n: ℕ) => λ (xs : fin n.succ → ℕ) => xs n * K n (λ i => xs i) + K (n - 1) (λ i => xs (i : ℕ))

def p (n: ℕ) (a: fin (n + 2) → ℕ) : ℕ :=
  K (n + 1) a

def q (n : ℕ) (a: fin (n + 1) → ℕ) (a : fin n.succ → ℕ) : ℕ :=
  K n (λ i => a (fin.succ i ))

theorem continued_fraction (n : ℕ)
  (a : fin (n+1) → ℕ):
  p n (λ i => a (fin.succ i)).succ / q n (λ i => a (fin.succ i)) = 
  a 0 + ∑ k in range n, (-1) ^ k / (q k (λ i => a (fin.succ i)) * q (k + 1) (λ i => a (fin.succ i))) :=
sorry

end continued_fraction_l798_798645


namespace symmetry_center_of_transformed_function_l798_798895

theorem symmetry_center_of_transformed_function :
  let f : ℝ → ℝ := λ x, Real.sin (4 * x + Real.pi / 4)
  let g : ℝ → ℝ := λ x, f (x / 2)
  let h : ℝ → ℝ := λ x, g (x - Real.pi / 8)
  ∃ k : ℤ, h (k * Real.pi / 2) = 0 :=
by
  let f : ℝ → ℝ := λ x, Real.sin (4 * x + Real.pi / 4)
  let g : ℝ → ℝ := λ x, f (x / 2)
  let h : ℝ → ℝ := λ x, g (x - Real.pi / 8)
  use 1
  sorry

end symmetry_center_of_transformed_function_l798_798895


namespace find_r_l798_798926

-- Definitions based on given conditions
variables {A B C D E P : Type*}

def CD_DB (CD DB : ℝ) : Prop := CD / DB = 4 / 1
def AE_EB (AE EB : ℝ) : Prop := AE / EB = 2 / 1

-- Main goal based on the correct answer
theorem find_r (h1 : CD_DB 4 1) (h2 : AE_EB 2 1) : ∃ (r : ℝ), r = 3 / 5 :=
by
  use 3 / 5
  sorry

end find_r_l798_798926


namespace projection_b_on_a_l798_798179

variable (a b : ℝ)
variable (theta : ℝ)
variable (norm_a : Real) (norm_b : Real)
variable (cos_theta : Real)

-- Conditions
def norm_a_val : norm_a = 5 := by rfl
def norm_b_val : norm_b = 4 := by rfl
def angle_val : theta = 120 := by rfl
def cos_theta_val : cos_theta = Real.cos (theta * π / 180) := by rfl

-- Desired projection value
def projection_val : Real := -2

theorem projection_b_on_a : 
  (norm_b * cos_theta * norm_a) / norm_a = projection_val := 
by
  sorry

end projection_b_on_a_l798_798179


namespace janet_home_time_l798_798271

def blocks_north := 3
def blocks_west := 7 * blocks_north
def blocks_south := blocks_north
def blocks_east := 2 * blocks_south -- Initially mistaken, recalculating needed
def remaining_blocks_west := blocks_west - blocks_east
def total_blocks_home := blocks_south + remaining_blocks_west
def walking_speed := 2 -- blocks per minute

theorem janet_home_time :
  (blocks_south + remaining_blocks_west) / walking_speed = 9 := by
  -- We assume that Lean can handle the arithmetic properly here.
  sorry

end janet_home_time_l798_798271


namespace max_value_a2018_minus_a2017_l798_798995

noncomputable def a : ℕ → ℝ
| 0       := 0
| 1       := 1 
| (n + 2) := 
  let k := some (nat.find_spec (exists (k : ℕ) (hk1 : 1 ≤ k) (hk2 : k ≤ n+2), 
                                a (n+1) = (a (n+k) + a (n+k-1) + ... + a (n+1-k)) / k)) 
  in 
  (list.sum (list.map a (list.range k))) / k

theorem max_value_a2018_minus_a2017 : a 2018 - a 2017 = 2016 / 2017^2 := sorry

end max_value_a2018_minus_a2017_l798_798995


namespace part1_part2_l798_798881

theorem part1 (k x1 x2 : ℝ) 
  (h1: 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2: 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h_sum: x1 + x2 = 1)
  (h_prod: x1 * x2 = (k + 1) / (4 * k)) :
  ∀ k, ¬ ((2 * x1 - x2) * (x1 - 2 * x2) = -3 / 2) :=
by
  sorry

theorem part2 (k x1 x2 : ℝ) 
  (h1: 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2: 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h_sum: x1 + x2 = 1)
  (h_prod: x1 * x2 = (k + 1) / (4 * k)) :
  ∀ k, (k = -2 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end part1_part2_l798_798881


namespace maximum_value_quotient_squared_is_two_l798_798291

noncomputable def maximum_value_quotient_squared (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : a ≥ b) : ℝ :=
  if ∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ 
   a^2 + y^2 = (a - x)^2 + (b - y)^2 ∧
   a^2 + y^2 = b^2 - x^2 + y^2 
  then (a / b) ^ 2 else 0

theorem maximum_value_quotient_squared_is_two (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : a ≥ b) :
  maximum_value_quotient_squared a b h1 h2 h3 = 2 := 
sorry

end maximum_value_quotient_squared_is_two_l798_798291


namespace count_valid_numbers_l798_798478

noncomputable theory

/-- Define the set of digits --/
def digitSet : Finset ℕ := {2, 3, 4, 5, 6}

/-- Check if the given number is a valid four-digit number with the conditions given --/
def validNumber (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (n % 2 = 1) ∧
  (∃ l : List ℕ, l.permutations ∈ digitSet.toFinset.powerset
    ∧ 5 ∈ l ∧ 6 ∈ l
    ∧ (List.indexOf 5 l = List.indexOf 6 l + 1 ∨ List.indexOf 5 l + 1 = List.indexOf 6 l))

/-- The total count of valid numbers should be 18 --/
theorem count_valid_numbers :
  (Finset.filter validNumber (Finset.Ico 1000 10000)).card = 18 :=
sorry

end count_valid_numbers_l798_798478


namespace simplify_expression_l798_798657

theorem simplify_expression (x : ℝ) : 3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + x^3) = -x^3 + 9 * x^2 + 6 * x - 3 :=
by
  sorry -- Proof is omitted.

end simplify_expression_l798_798657


namespace spacy_set_count_15_l798_798997

def spacy_subsets_count : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 3
| 3       := 4
| (n + 1) := spacy_subsets_count n + spacy_subsets_count (n - 2)

theorem spacy_set_count_15 : spacy_subsets_count 15 = 406 :=
by
  -- The result is pre-computed from the recurrence relation and initial values.
  sorry

end spacy_set_count_15_l798_798997


namespace intersection_sets_l798_798191

-- Define the sets A and B as given in the problem conditions
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 4}

-- Lean theorem statement for proving the intersection of sets A and B is {0, 2}
theorem intersection_sets : A ∩ B = {0, 2} := 
by
  sorry

end intersection_sets_l798_798191


namespace distance_AD_bounds_l798_798318

-- define points and distances based on conditions
variable (A B C D : Type)
variable (distance_AC : ℝ)
variable (distance_CD : ℝ)

-- given conditions
axiom points_A_and_B (d : ℝ) : distance B A = d -- B is directly east of A
axiom points_B_and_C : C = B + vector (0, distance_AC / 12)
axiom distance_AC_def : distance A C = 12 * real.sqrt 2
axiom distance_CD_def : distance C D = 5

-- prove the distance AD is between given integers
theorem distance_AD_bounds : 20 < distance A D ∧ distance A D < 21 :=
  sorry

end distance_AD_bounds_l798_798318


namespace platform_length_l798_798037

theorem platform_length (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) :
  train_length = 300 →
  time_pole = 18 →
  time_platform = 45 →
  (train_length * time_platform / time_pole) - train_length = 450 :=
by
  intros h_train_length h_time_pole h_time_platform
  have h1 : train_length * 45 / 18 = 750, from sorry
  calc
    (train_length * 45 / 18) - train_length = 750 - 300 : by rw [h1, h_train_length]
                                      ... = 450 : by norm_num

end platform_length_l798_798037


namespace rook_traversal_possible_l798_798356

structure Labyrinth :=
(fields : FiniteType ℕ)

structure Position :=
(coord : ℕ)

structure Rook :=
(lab : Labyrinth)
(pos : Position)

variable (L : Labyrinth) (P : Position)

theorem rook_traversal_possible :
  ∃ program, ∀ (L : Labyrinth) (P : Position), can_traverse_all_accessible_fields L P program :=
sorry

end rook_traversal_possible_l798_798356


namespace number_of_games_can_buy_l798_798752

-- Definitions based on the conditions
def initial_money : ℕ := 42
def spent_money : ℕ := 10
def game_cost : ℕ := 8

-- The statement we need to prove: Mike can buy 4 games given the conditions
theorem number_of_games_can_buy : (initial_money - spent_money) / game_cost = 4 :=
by
  sorry

end number_of_games_can_buy_l798_798752


namespace area_of_convex_pentagon_FGHIJ_l798_798675

-- Definitions of the conditions given in the problem:
def isEquilateral (a b c : ℝ) : Prop := (a = b) ∧ (b = c)
def isConvexPentagon (F G H I J : ℝ) (A B C D E : ℝ) : Prop :=
  ∡ A B C = 120 ∧ ∡ B C D = 120 ∧ 
  isEquilateral A B C ∧ isEquilateral J F G ∧ 
  H I = 5 ∧ I J = 5

-- The main theorem statement
theorem area_of_convex_pentagon_FGHIJ 
  (F G H I J : ℝ) (A B C D E : ℝ)
  (h1 : ∡ A B C = 120) (h2 : ∡ B C D = 120)
  (h3 : J F = 3) (h4 : F G = 3) (h5 : G H = 3)
  (h6 : H I = 5) (h7 : I J = 5) :
  area (convex_pentagon F G H I J) = 71 * Real.sqrt 3 / 4 :=
begin
  sorry
end

end area_of_convex_pentagon_FGHIJ_l798_798675


namespace max_ab_l798_798859

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x, g x = 2 ^ x) (h4 : g a * g b = 2) :
  ab ≤ 1 / 4 :=
begin
  sorry
end

end max_ab_l798_798859


namespace value_of_t_l798_798386

theorem value_of_t (t : ℝ) (x y : ℝ) (h1 : x = 1 - 2 * t) (h2 : y = 2 * t - 2) (h3 : x = y) : t = 3 / 4 := 
by
  sorry

end value_of_t_l798_798386


namespace value_of_each_iron_nickel_is_3_l798_798090

namespace AliceNickels

-- Define the number of quarters Alice has.
def numQuarters : ℕ := 20

-- Define the value of each quarter in dollars.
def valuePerQuarter : ℝ := 0.25

-- Define the total value of quarters.
def totalValueQuarters : ℝ := numQuarters * valuePerQuarter

-- Define the number of nickels per quarter.
def nickelsPerQuarter : ℕ := 5

-- Define the total number of nickels.
def totalNickels : ℕ := numQuarters * nickelsPerQuarter

-- Define the percentage of iron nickels.
def percentageIronNickels : ℝ := 0.20

-- Define the number of iron nickels.
def numIronNickels : ℕ := (percentageIronNickels * totalNickels).toNat

-- Define the percentage of regular nickels.
def percentageRegularNickels : ℝ := 0.80

-- Define the number of regular nickels.
def numRegularNickels : ℕ := (percentageRegularNickels * totalNickels).toNat

-- Define the value of each regular nickel in dollars.
def valuePerNickel : ℝ := 0.05

-- Define the total value of regular nickels.
def totalValueRegularNickels : ℝ := numRegularNickels * valuePerNickel

-- Define the total value of money after the exchange.
def totalValueAfterExchange : ℝ := 64.00

-- Define the total value of iron nickels.
def totalValueIronNickels : ℝ := totalValueAfterExchange - totalValueRegularNickels

-- Define the value of each iron nickel.
def valuePerIronNickel : ℝ := totalValueIronNickels / numIronNickels

-- Theorem stating the value of each iron nickel is $3.
theorem value_of_each_iron_nickel_is_3 : valuePerIronNickel = 3 := by
  sorry

end AliceNickels

end value_of_each_iron_nickel_is_3_l798_798090


namespace general_formula_of_sequence_range_of_m_l798_798992

variables {R : Type*} [ordered_field R]

def f (x : R) : R := sorry -- we'll assume such a function is given later

def a (n : ℕ) : R := if 0 < n then f (2 ^ n) / (2 ^ n) else 0

theorem general_formula_of_sequence (n : ℕ) (hn : 0 < n) : 
  a n = n := 
sorry

theorem range_of_m (m : R) : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 10 ∧ m * a n ^ 2 + 2 * a n - 2 * m - 1 < 0) ↔ 
  m < - (19 : R) / 98 ∨ 1 < m :=
sorry

end general_formula_of_sequence_range_of_m_l798_798992


namespace log_problem_l798_798226

def log_base (b x : ℝ) : ℝ := log x / log b

theorem log_problem (a b : ℝ) (h1 : a = log_base 8 225) (h2 : b = log_base 2 15) : a = (2 * b) / 3 :=
by
  sorry

end log_problem_l798_798226


namespace n_minus_two_is_square_of_natural_number_l798_798000

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l798_798000


namespace parametric_equation_of_curve_general_equation_of_line_minimum_value_of_MN_l798_798210

-- Define the initial parametric equation of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + t, 6 - 2 * t)

-- Define the polar equation of the curve C
def curve_C_polar (ρ θ : ℝ) : Prop :=
  4 * ρ^2 + 5 * ρ^2 * (cos θ)^2 = 36

-- Define the Cartesian parametric form of the corresponding curve
def curve_C (φ : ℝ) : ℝ × ℝ :=
  (2 * cos φ, 3 * sin φ)

-- Define the general equation for line l
def line_l_general (x y : ℝ) : Prop :=
  2 * x + y = 10

-- Define a structure to represent points and distance to line calculation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance_to_line (p : Point) : ℝ :=
  abs (2 * p.x + p.y - 10) / (sqrt 5)

-- Define the minimum value of |MN| problem
def min_MN (φ : ℝ) : ℝ :=
  abs (10 - 5 * sin (φ + 60)) / (sqrt 5)

theorem parametric_equation_of_curve :
  ∀ φ, curve_C φ = (2 * cos φ, 3 * sin φ) := sorry

theorem general_equation_of_line :
  ∀ x y, line_l_general x y ↔ 2 * x + y = 10 := sorry

theorem minimum_value_of_MN :
  ∀ φ, min_MN φ ≥ (2 * sqrt 15 / 3) := sorry

end parametric_equation_of_curve_general_equation_of_line_minimum_value_of_MN_l798_798210


namespace sasha_claim_incorrect_l798_798948

-- Define the conditions of the polyhedron
structure Polyhedron :=
(dihedral_angles_right : ∀ (α β : ℝ), α = 90 ∧ β = 90)

-- Define the vertices X and Y and their coordinates based on given edge lengths
def X : ℝ × ℝ × ℝ := (0, 0, 0)
def Y : ℝ × ℝ × ℝ := (2, 2, 1)

-- Function to calculate the Euclidean distance between two 3D points
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

-- Proposition to verify the shortest path distance between X and Y
def verify_distance (poly : Polyhedron) : Prop :=
  distance X Y = 3

theorem sasha_claim_incorrect (poly : Polyhedron) : ¬ (verify_distance poly → 4 = 3) :=
by {
  -- By definition of verify_distance
  unfold verify_distance,
  -- Simplify the equation
  unfold distance,
  simp [X, Y],
  -- Calculate the distance and show it cannot be 4 units
  have : real.sqrt ((2 - 0) ^ 2 + (2 - 0) ^ 2 + (1 - 0) ^ 2) = 3,
  { norm_num },
  show ¬ (3 = 4) by contradiction,
  sorry -- proof steps not required
}

end sasha_claim_incorrect_l798_798948


namespace quadratic_has_one_real_solution_l798_798167

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l798_798167


namespace evaluate_star_l798_798532

def star (A B : ℝ) : ℝ := (A + B) / 3

theorem evaluate_star : star (star 3 15) (star 6 2) = 26 / 9 :=
by
  sorry

end evaluate_star_l798_798532


namespace hyperbola_focal_length_l798_798609

theorem hyperbola_focal_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 8) : 
    2 * Real.sqrt (a^2 + b^2) ≥ 8 :=
begin
  sorry
end

end hyperbola_focal_length_l798_798609


namespace shifted_parabola_l798_798569

theorem shifted_parabola (x : ℝ) : 
  let original := 5 * x^2 in
  let shifted_left := 5 * (x + 2)^2 in
  let shifted_up := shifted_left + 3 in
  shifted_up = 5 * (x + 2)^2 + 3 := 
by
  sorry

end shifted_parabola_l798_798569


namespace shifted_parabola_eq_l798_798567

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l798_798567


namespace PM_eq_PS_l798_798783

structure SemicircleGeometry where
  (A B M P Q S : Point)
  (midpoint_AB : midpoint M A B)
  (on_semicircle : on_semicircle M A B P)
  (midpoint_arc_AP : arc_midpoint M A P Q)
  (intersection_BP_parallel_PQ_MS : intersects BP (parallel PQ M S))

theorem PM_eq_PS (geom : SemicircleGeometry) : dist M P = dist P S := by
  sorry

end PM_eq_PS_l798_798783


namespace mole_fractions_C4H8O2_l798_798460

/-- 
Given:
- The molecular formula of C4H8O2,
- 4 moles of carbon (C) atoms,
- 8 moles of hydrogen (H) atoms,
- 2 moles of oxygen (O) atoms.

Prove that:
The mole fractions of each element in C4H8O2 are:
- Carbon (C): 2/7
- Hydrogen (H): 4/7
- Oxygen (O): 1/7
--/
theorem mole_fractions_C4H8O2 :
  let m_C := 4
  let m_H := 8
  let m_O := 2
  let total_moles := m_C + m_H + m_O
  let mole_fraction_C := m_C / total_moles
  let mole_fraction_H := m_H / total_moles
  let mole_fraction_O := m_O / total_moles
  mole_fraction_C = 2 / 7 ∧ mole_fraction_H = 4 / 7 ∧ mole_fraction_O = 1 / 7 := by
  sorry

end mole_fractions_C4H8O2_l798_798460


namespace maximal_cross_section_area_l798_798781

theorem maximal_cross_section_area :
  let prism_base : set (ℝ × ℝ × ℝ) := { p | (∃ z, p = (4, 6, z)) ∨ 
                                           (∃ z, p = (-4, 6, z)) ∨ 
                                           (∃ z, p = (-4, -6, z)) ∨ 
                                           (∃ z, p = (4, -6, z)) }
  ∃ z_A z_B z_C z_D : ℝ, 
  (3 * 4 - 5 * 6 + 6 * z_A = 30 ∧ ¬(4, 6, z_A) ∈ prism_base) ∧
  (3 * -4 - 5 * 6 + 6 * z_B = 30 ∧ ¬(-4, 6, z_B) ∈ prism_base) ∧
  (3 * -4 + 5 * 6 + 6 * z_C = 30 ∧ ¬(-4, -6, z_C) ∈ prism_base) ∧
  (3 * 4 + 5 * -6 + 6 * z_D = 30 ∧ ¬(4, -6, z_D) ∈ prism_base) ∧
  let AB := (-8, 0, z_B - z_A) in 
  let AD := (0, -12, z_D - z_A) in 
  let cross_prod := (
    (AD.2 * AB.3 - AD.3 * AB.2),
    (AD.3 * AB.1 - AD.1 * AB.3),
    (AD.1 * AB.2 - AD.2 * AB.1)) in 
  ∥cross_prod∥ = 154 :=
sorry

end maximal_cross_section_area_l798_798781


namespace farmer_initial_productivity_l798_798769

theorem farmer_initial_productivity (x : ℝ) (d : ℝ)
  (hx1 : d = 1440 / x)
  (hx2 : 2 * x + (d - 4) * 1.25 * x = 1440) :
  x = 120 :=
by
  sorry

end farmer_initial_productivity_l798_798769


namespace distance_between_cars_l798_798363

-- Definitions representing the initial conditions and distances traveled by the cars
def initial_distance : ℕ := 113
def first_car_distance_on_road : ℕ := 50
def second_car_distance_on_road : ℕ := 35

-- Statement of the theorem to be proved
theorem distance_between_cars : initial_distance - (first_car_distance_on_road + second_car_distance_on_road) = 28 :=
by
  sorry

end distance_between_cars_l798_798363


namespace sum_of_odds_17_to_47_l798_798373

theorem sum_of_odds_17_to_47 : 
  let a := 17
  let l := 47
  let d := 2
  let n := (l - a) / d + 1
in
  (n = 16) →
  let sum := (a + l) * n / 2
in
  sum = 512 :=
by
  intros
  sorry

end sum_of_odds_17_to_47_l798_798373


namespace compute_expression_l798_798117

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l798_798117


namespace g_zero_unique_l798_798879

theorem g_zero_unique (f : ℝ → ℝ) (hf_mono : monotone f)
  (h_functional : ∀ x ∈ set.Ioi 0, f (f x - real.log x) = real.exp 1 + 1) :
  ∃! x ∈ set.Ioi (0 : ℝ), (f x - deriv f x) = 0 :=
sorry

end g_zero_unique_l798_798879


namespace angle_AST_90_degrees_l798_798988

noncomputable def circumradius {A B C : Type*} [metric_space A] 
  (h : triangle A B C) := sorry
    -- Definition not detailed in current context

noncomputable def foot_of_altitude {A B C : Type*} [metric_space A] 
  (h : triangle A B C) (ha : acute_triangle A B C): sorry
    -- Definition not detailed in current context

noncomputable def center_of_circumcircle_arc {A B C : Type*} [metric_space A] 
  (h : triangle A B C) (ha : acute_triangle A B C) (h_bc_not_contains_A: does_not_contain A B C): sorry
    -- Definition not detailed in current context

theorem angle_AST_90_degrees {A B C D T S : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space T] [metric_space S]
  (ha : acute_triangle A B C) (h_ac_lt_ab : AC < AB)
  (hR : R = circumradius A B C) 
  (hD : D = foot_of_altitude A B C ha) 
  (hT : T ∈ line AD) (hAT : AT = 2 * R) 
  (hD_between_AT : D ∈ between A T) 
  (hS : S = center_of_circumcircle_arc A B C ha (arc_not_contain_A A B C)):
  angle A S T = 90 := 
sorry

end angle_AST_90_degrees_l798_798988


namespace angle_ABC_acute_range_l798_798905

def is_acute_angle_range (m : ℝ) : Prop :=
  (m > -3/4 ∧ m < 1/2) ∨ m > 1/2

theorem angle_ABC_acute_range : 
  ∀ (m : ℝ),
  let OA : ℝ × ℝ := (3, -4),
      OB : ℝ × ℝ := (6, -3),
      OC : ℝ × ℝ := (5 - m, -3 - m),
      AB : ℝ × ℝ := (3, 1),
      AC : ℝ × ℝ := (2 - m, 1 - m),
      BA : ℝ × ℝ := (-3, -1),
      BC : ℝ × ℝ := (-1 - m, -m) in
  (angle_ABC_is_acute : (m > -3/4 ∧ m < 1/2) ∨ m > 1/2) :=
begin
  let AB : ℝ × ℝ := (3, 1),
  let AC : ℝ × ℝ := (2 - m, 1 - m),
  let BA : ℝ × ℝ := (-3, -1),
  let BC : ℝ × ℝ := (-1 - m, -m),
  sorry
end

end angle_ABC_acute_range_l798_798905


namespace num_integers_in_solution_set_l798_798546

theorem num_integers_in_solution_set : 
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), |x - 3| ≤ 7.5 → abs (x - 3) ≤ 7.5 := 
by
  sorry

end num_integers_in_solution_set_l798_798546


namespace dance_women_count_l798_798803

theorem dance_women_count (men_count : ℕ) (women_count : ℕ) (pairs_count : ℕ) 
    (h1 : men_count = 15) 
    (h2 : ∀ m, m < men_count → 4 = ∑ x, if x < women_count then 1 else 0) 
    (h3 : ∀ w, w < women_count → 3 = ∑ y, if y < men_count then 1 else 0) 
    (h4 : pairs_count = men_count * 4) 
    (h5 : pairs_count = women_count * 3) : 
    women_count = 20 := 
by 
  sorry

end dance_women_count_l798_798803


namespace neg_cos_ge_a_l798_798533

theorem neg_cos_ge_a (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a = 2 := 
sorry

end neg_cos_ge_a_l798_798533


namespace average_test_score_l798_798558

theorem average_test_score (x : ℝ) :
  (0.45 * 95 + 0.50 * x + 0.05 * 60 = 84.75) → x = 78 :=
by
  sorry

end average_test_score_l798_798558


namespace nice_subset_unique_l798_798486

-- Define the problem statement in Lean 4
theorem nice_subset_unique (n : ℕ) (S : set (fin n)) :
  (∀ k ∈ S, ∀ (distribution : finset ((fin n) × (fin n))),
    (∀ t ∈ distribution, ∃ (group : finset (fin n)) (t ≥ group.card),
        group.subset distribution.image.2) →
      (∀ (k : ℕ), k ∈ finset.range n → ∀ (kids : finset (fin n)),
        kids.card = k → ∃ (group : finset (fin n)), group.card ≥ k ∧ group ⊆ kids)) ↔
  S = set.univ := sorry

end nice_subset_unique_l798_798486


namespace digit_150_of_17_over_70_l798_798367

theorem digit_150_of_17_over_70 : 
  let repeating_block := "428571";
  let block_length := String.length repeating_block;
  let decimal_digit := repeating_block.get (150 % block_length - 1);
  decimal_digit = '1' :=
by
  sorry

end digit_150_of_17_over_70_l798_798367


namespace poly_sum_and_expr_evaluation_l798_798702

open Polynomial

theorem poly_sum_and_expr_evaluation (m n : ℤ)
  (h1 : -2 * m - 4 = 0)
  (h2 : n + 2 = 0) :
  m = -2 ∧ n = -2 ∧ (4 * m ^ 2 * n - 3 * m * n ^ 2) - 2 * (m ^ 2 * n + m * n ^ 2) = 24 := 
by {
  have hm : m = -2, from eq_of_add_eq_zero_left h1,
  have hn : n = -2, from eq_zero_of_add_eq_zero_right h2,
  rw [hm, hn],
  split; { try {refl} },
  calc (4 * (-2) ^ 2 * (-2) - 3 * (-2) * (-2) ^ 2) - 2 * ((-2) ^ 2 * (-2) + (-2) * (-2) ^ 2)
     = (4 * 4 * -2 - 3 * -2 * 4) - 2 * (4 * -2 + -2 * 4) : by norm_num
  ... = (-32 + 24) - 2 * (-8 - 8) : by norm_num
  ... = -32 + 24 + 32 : by norm_num
  ... = 24 : by norm_num,
}

end poly_sum_and_expr_evaluation_l798_798702


namespace problem_solution_l798_798051

theorem problem_solution (N : ℚ) (h : (4/5) * (3/8) * N = 24) : 2.5 * N = 200 :=
by {
  sorry
}

end problem_solution_l798_798051


namespace average_speed_correct_l798_798040

def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / (t1 + t2)

theorem average_speed_correct :
  average_speed 290 400 4.5 5.5 = 69 :=
by
  sorry

end average_speed_correct_l798_798040


namespace ratio_unit_prices_l798_798806

-- Definitions and conditions
variables (v p : ℝ) -- Volume and Price of Brand B soda
def volume_A : ℝ := 1.25 * v -- Volume of Brand A soda
def price_A : ℝ := 0.85 * p -- Price of Brand A soda

-- The unit prices
def unit_price_B : ℝ := p / v
def unit_price_A : ℝ := price_A / volume_A

-- The statement to be proved
theorem ratio_unit_prices : unit_price_A v p / unit_price_B v p = 17 / 25 :=
by sorry

end ratio_unit_prices_l798_798806


namespace range_of_x_l798_798870

-- Define the necessary properties and functions.
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f (-x) = f x)
variable (hf_monotonic : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

-- Define the statement to be proved.
theorem range_of_x (f : ℝ → ℝ) (hf_even : ∀ x, f (-x) = f x) (hf_monotonic : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  { x : ℝ | f (2 * x - 1) ≤ f 3 } = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_of_x_l798_798870


namespace students_more_than_Yoongi_l798_798377

theorem students_more_than_Yoongi (total_players : ℕ) (less_than_Yoongi : ℕ) (total_players_eq : total_players = 21) (less_than_eq : less_than_Yoongi = 11) : 
  ∃ more_than_Yoongi : ℕ, more_than_Yoongi = (total_players - 1 - less_than_Yoongi) ∧ more_than_Yoongi = 8 :=
by
  sorry

end students_more_than_Yoongi_l798_798377


namespace inscribed_square_area_l798_798662

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end inscribed_square_area_l798_798662


namespace odd_coefficients_in_binomial_expansion_l798_798193

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion_l798_798193


namespace number_of_z_with_equilateral_triangle_property_l798_798550

theorem number_of_z_with_equilateral_triangle_property : 
  ∃! (z : ℂ), z ≠ 0 ∧ (∃ (θ : ℝ), (z = exp (complex.I * θ) ∨ z = exp (complex.I * (θ + 2 * π)))))
:= sorry

end number_of_z_with_equilateral_triangle_property_l798_798550


namespace shifted_parabola_l798_798570

theorem shifted_parabola (x : ℝ) : 
  let original := 5 * x^2 in
  let shifted_left := 5 * (x + 2)^2 in
  let shifted_up := shifted_left + 3 in
  shifted_up = 5 * (x + 2)^2 + 3 := 
by
  sorry

end shifted_parabola_l798_798570


namespace ellipse_range_of_k_l798_798889

theorem ellipse_range_of_k (k : ℝ) :
  (1 - k > 0) ∧ (1 + k > 0) ∧ (1 - k ≠ 1 + k) ↔ (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1) :=
by
  sorry

end ellipse_range_of_k_l798_798889


namespace chord_length_is_six_l798_798674

noncomputable def length_of_chord (A : ℝ × ℝ) (r : ℝ) : ℝ :=
  2 * real.sqrt (r^2 - A.2^2)

theorem chord_length_is_six (A : ℝ × ℝ) (hA : A = (4, 4)) 
                             (h_directrix : -1 = A.1 - 5) : 
                             length_of_chord A 5 = 6 :=
by
  rw [length_of_chord, hA]
  norm_num [sq_sub_sq_eq, sub_eq_add_neg, sq] -- Calculate the length of chord using the provided details

  sorry  -- Proof omitted

end chord_length_is_six_l798_798674


namespace callum_points_earned_l798_798597

-- Definitions
variables (rounds : ℕ) (points_per_win : ℕ) (krishna_matches_won_fraction : ℚ) (total_matches : ℕ)

-- Given Conditions
def earn_points_if_win : ℕ := points_per_win
def total_matches_played : ℕ := total_matches
def krishna_winnings_fraction : ℚ := krishna_matches_won_fraction

-- Theorem Statement
theorem callum_points_earned (h1 : points_per_win = 10)
                             (h2 : total_matches = 8)
                             (h3 : krishna_matches_won_fraction = 3/4) : 
                             let krishna_matches_won := krishna_matches_won_fraction * total_matches,
                                 callum_matches_won := total_matches - krishna_matches_won in
                             callum_matches_won * points_per_win = 20 :=
by
  sorry

end callum_points_earned_l798_798597


namespace probability_top_three_spades_next_two_hearts_l798_798420

-- We will use this theorem to assert the final probability.
theorem probability_top_three_spades_next_two_hearts
  (deck : Finset (Fin 52)) -- The deck is a finite set of 52 cards, indexed from 0 to 51.
  (spades hearts : Finset (Fin 13)) -- Spades and Hearts are sets of 13 cards each.
  (h_deck : deck.card = 52) -- The deck contains 52 cards.
  (h_spades : spades.card = 13) -- There are 13 spades.
  (h_hearts : hearts.card = 13) -- There are 13 hearts.
  (h_no_same_rank : ∀ (i j : Fin 13), i ≠ j → spades.val i ≠ spades.val j ∧ hearts.val i ≠ hearts.val j) -- Each rank card in spades and hearts is unique.
  (random_arrangement : deck = (Fin 52).toFinset) -- The deck is randomly arranged (all arrangements are equally probable).
  
  : (probability_top_three_spades_next_two_hearts = 432 / 6497400) :=
by
  sorry

end probability_top_three_spades_next_two_hearts_l798_798420


namespace distance_on_dirt_road_l798_798393

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end distance_on_dirt_road_l798_798393


namespace find_x_plus_y_l798_798195

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def are_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem find_x_plus_y (x y : ℝ) :
  let a := (2, x)
  let b := (y, -2)
  let c := (2, -4)
  are_orthogonal a c ∧ are_parallel b c → x + y = 2 :=
by {
  intros h,
  sorry
}

end find_x_plus_y_l798_798195


namespace total_balls_in_box_l798_798553

theorem total_balls_in_box (red blue yellow total : ℕ) 
  (h1 : 2 * blue = 3 * red)
  (h2 : 3 * yellow = 4 * red) 
  (h3 : yellow = 40)
  (h4 : red + blue + yellow = total) : total = 90 :=
sorry

end total_balls_in_box_l798_798553


namespace five_more_limbs_l798_798428

-- Definition of the number of limbs an alien has
def alien_limbs : ℕ := 3 + 8

-- Definition of the number of limbs a Martian has
def martian_limbs : ℕ := (8 / 2) + (3 * 2)

-- The main statement that we need to prove
theorem five_more_limbs : 5 * alien_limbs - 5 * martian_limbs = 5 := by
  have h1 : alien_limbs = 11 := rfl
  have h2 : martian_limbs = 10 := rfl
  calc
    5 * alien_limbs - 5 * martian_limbs
        = 5 * 11 - 5 * 10 := by rw [h1, h2]
    ... = 55 - 50     := by rfl
    ... = 5           := by rfl

end five_more_limbs_l798_798428


namespace non_divisible_by_twenty_l798_798448

theorem non_divisible_by_twenty (k : ℤ) (h : ∃ m : ℤ, k * (k + 1) * (k + 2) = 5 * m) :
  ¬ (∃ l : ℤ, k * (k + 1) * (k + 2) = 20 * l) := sorry

end non_divisible_by_twenty_l798_798448


namespace fixed_point_exists_l798_798864

-- Define the set G and its properties
def G : Set (ℝ → ℝ) :=
  {f | ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x + b}

-- Define the properties of G
axiom G_comp {f g : ℝ → ℝ} (hf : f ∈ G) (hg : g ∈ G) : (g ∘ f) ∈ G
axiom G_inv {f : ℝ → ℝ} (hf : f ∈ G) : inv f ∈ G
axiom G_fixed_point {f : ℝ → ℝ} (hf : f ∈ G) : ∃ x : ℝ, f x = x

-- Prove there exists a constant k such that for all f ∈ G, f k = k
theorem fixed_point_exists : ∃ k : ℝ, ∀ f : ℝ → ℝ, f ∈ G → f k = k :=
by
  sorry

end fixed_point_exists_l798_798864


namespace cost_price_l798_798743

theorem cost_price (SP : ℝ) (profit_percentage : ℝ) : SP = 600 ∧ profit_percentage = 60 → ∃ CP : ℝ, CP = 375 :=
by
  intro h
  sorry

end cost_price_l798_798743


namespace distance_from_novosibirsk_to_karasuk_l798_798763

theorem distance_from_novosibirsk_to_karasuk (vb vm t : ℝ) :
  (vb > 0) →
  (vm > 0) →
  (t > 0) →
  (vm = 2 * vb) →
  let d := 70 + 2 * vb * t
  in d = 140 :=
by
  intros hvb hvm ht hvm_eq
  let d := 70 + 2 * vb * t
  -- additional setup and reasoning can go here
  sorry

end distance_from_novosibirsk_to_karasuk_l798_798763


namespace john_out_of_pocket_l798_798965

-- Define the conditions
def computer_cost : ℕ := 700
def accessories_cost : ℕ := 200
def playstation_value : ℕ := 400
def sale_discount : ℚ := 0.2

-- Define the total cost of the computer and accessories
def total_cost : ℕ := computer_cost + accessories_cost

-- Define the selling price of the PlayStation
def selling_price : ℕ := playstation_value - (playstation_value * sale_discount).to_nat

-- Define the amount out of John's pocket
def out_of_pocket : ℕ := total_cost - selling_price

-- The proof goal
theorem john_out_of_pocket : out_of_pocket = 580 :=
by
  sorry

end john_out_of_pocket_l798_798965


namespace compute_expression_l798_798119

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l798_798119


namespace max_projection_in_direction_l798_798906

variable {V : Type*} [InnerProductSpace ℝ V]

-- Definitions for the problem conditions
variables (a b : V)
axiom norm_a : ∥a∥ = 2
axiom norm_2a_add_b : ∥2 • a + b∥ = 2

-- Definition for the projection of vector a in the direction of vector b
def projection (a b : V) : ℝ := (innerSL a b) / ∥b∥

-- The theorem stating the maximum value of the projection
theorem max_projection_in_direction (a b : V) (ha : ∥a∥ = 2) (hb : ∥2 • a + b∥ = 2) :
  projection a b = -sqrt 3 := 
sorry

end max_projection_in_direction_l798_798906


namespace johns_out_of_pocket_expense_l798_798967

theorem johns_out_of_pocket_expense
  (computer_cost : ℕ)
  (accessories_cost : ℕ)
  (playstation_value : ℕ)
  (playstation_sold_percent_less : ℕ) :
  computer_cost = 700 →
  accessories_cost = 200 →
  playstation_value = 400 →
  playstation_sold_percent_less = 20 →
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100) in
  let total_cost := computer_cost + accessories_cost in
  let pocket_expense := total_cost - playstation_sold_price in
  pocket_expense = 580 :=
by
  intros h1 h2 h3 h4
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100)
  let total_cost := computer_cost + accessories_cost
  let pocket_expense := total_cost - playstation_sold_price
  sorry

end johns_out_of_pocket_expense_l798_798967


namespace find_nearest_integer_x_minus_y_l798_798918

variable (x y : ℝ)

theorem find_nearest_integer_x_minus_y
  (h1 : abs x + y = 5)
  (h2 : abs x * y - x^3 = 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0) :
  |x - y| = 5 := sorry

end find_nearest_integer_x_minus_y_l798_798918


namespace expression_for_x_expression_for_y_l798_798607

variables {A B C : ℝ}

-- Conditions: A, B, and C are positive numbers with A > B > C > 0
axiom h1 : A > 0
axiom h2 : B > 0
axiom h3 : C > 0
axiom h4 : A > B
axiom h5 : B > C

-- A is x% greater than B
variables {x : ℝ}
axiom h6 : A = (1 + x / 100) * B

-- A is y% greater than C
variables {y : ℝ}
axiom h7 : A = (1 + y / 100) * C

-- Proving the expressions for x and y
theorem expression_for_x : x = 100 * ((A - B) / B) :=
sorry

theorem expression_for_y : y = 100 * ((A - C) / C) :=
sorry

end expression_for_x_expression_for_y_l798_798607


namespace congruent_orthocenter_quadrilateral_l798_798601

structure Point :=
(x : ℝ)
(y : ℝ)

-- Define a cyclic quadrilateral ABCD
structure CyclicQuadrilateral :=
(A B C D : Point)
(is_cyclic : ∃ O : Point, Dist O A = Dist O B ∧ Dist O B = Dist O C ∧ Dist O C = Dist O D)

-- Define the orthocenter of a triangle
def orthocenter (A B C : Point) : Point := sorry

-- Define the quadrilateral of orthocenters HA HB HC HD for triangles BCD, CDA, DAB, and ABC
structure OrthocenterQuadrilateral :=
(H_A H_B H_C H_D : Point)
(orthocenters : H_A = orthocenter B C D ∧ H_B = orthocenter C D A ∧ H_C = orthocenter D A B ∧ H_D = orthocenter A B C)

-- The theorem to prove congruence
theorem congruent_orthocenter_quadrilateral 
  (Q : CyclicQuadrilateral) 
  (OQ : OrthocenterQuadrilateral) 
  (H_A := OQ.H_A) (H_B := OQ.H_B) (H_C := OQ.H_C) (H_D := OQ.H_D)
  (A := Q.A) (B := Q.B) (C := Q.C) (D := Q.D) : 
  CongruentQuadrilateral (Quadrilateral.mk A B C D) (Quadrilateral.mk H_A H_B H_C H_D) :=
sorry

end congruent_orthocenter_quadrilateral_l798_798601


namespace trapezoid_PQ_length_l798_798869

variable (a b : ℝ) (L M : ℝ → Prop) (P Q : P → Q → ℝ)
variable (PL LR : ℝ)
noncomputable def PQ_formula (a b : ℝ) :=
  if L then (3 * a * b) / (2 * a + b) else (3 * a * b) / (a + 2 * b)

theorem trapezoid_PQ_length (a b : ℝ) (h_bc : b ≠ 0) (h_eq : PL = LR) :
  ∃ PQ, PQ = PQ_formula a b := 
sorry

end trapezoid_PQ_length_l798_798869


namespace coefficient_of_x_squared_l798_798494

noncomputable def integral_result : ℝ := ∫ x in 0..3, (2 * x - 1)

theorem coefficient_of_x_squared :
  (∀ n : ℕ, n = integral_result.to_nat) →
  ∀ (n : ℕ), (∃ k : ℕ, (T_k : ℝ → ℕ → ℝ) (x : ℝ), T_k x n = (-1)^k * 3^(n-k) * (nat.binom n k) * x^((5*k-6)/6) ∧ (5 * k / 6 - 3 = 2))
  → n = 6 →
  ∀ (k : ℕ), k = 6 →
  (-1)^6 * 3^(6-6) * (nat.binom 6 6) = 1 :=
sorry

end coefficient_of_x_squared_l798_798494


namespace cost_of_first_supply_l798_798423

-- Define the conditions given in the problem
def cost_of_second_supply : ℕ := 24
def last_years_remaining_budget : ℕ := 6
def this_years_budget : ℕ := 50
def remaining_budget_after_purchase : ℕ := 19

-- Calculate total budget and total cost of supplies
def total_budget := last_years_remaining_budget + this_years_budget
def total_cost_of_supplies := total_budget - remaining_budget_after_purchase

-- Proof statement: The first school supply cost $13
theorem cost_of_first_supply : ∀ (c : ℕ), 
  total_cost_of_supplies - cost_of_second_supply = c → c = 13 :=
by
  intro c
  assume h : total_cost_of_supplies - cost_of_second_supply = c
  have total_budget_def : total_budget = 6 + 50 := rfl
  rw total_budget_def at h
  have total_cost_of_supplies_def : total_cost_of_supplies = 56 - 19 := rfl
  rw total_cost_of_supplies_def at h
  have h1 : 56 - 19 = 37 := rfl
  rw h1 at h
  have h2 : 37 - 24 = 13 := rfl
  rw h2 at h
  exact h


end cost_of_first_supply_l798_798423


namespace geo_seq_4th_term_l798_798683

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l798_798683


namespace arithmetic_sequence_general_term_l798_798187

theorem arithmetic_sequence_general_term (a d : ℝ) (h1 : 1 + d = 3 / a)
                                         (h2 : d = 2 / a) (h3 : 0 < a):
  (∀ n : ℕ, (∃ a_n, a_n = 2 * n - 1)) := 
begin
  sorry
end

end arithmetic_sequence_general_term_l798_798187


namespace smallest_value_of_n_l798_798612

def smallest_func (n : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, (k < n) → f k ≤ 18 

def f (n : ℕ) : ℕ := 
  Inf {k : ℕ | factorial k % n = 0}

theorem smallest_value_of_n (n : ℕ) 
  (h1 : ∃ r : ℕ, n = 18 * r) 
  (h2 : f n > 18) 
  : n = 342 :=
sorry

end smallest_value_of_n_l798_798612


namespace min_distance_point_line_origin_l798_798872

/-- Given point P(m, n) on the line x + y - 4 = 0,
    where O is the origin of the coordinate system,
    prove that the minimum value of sqrt(m^2 + n^2) is 2 * sqrt(2). -/
theorem min_distance_point_line_origin (m n : ℝ) (h : m + n = 4) :
  sqrt (m^2 + n^2) = 2 * sqrt 2 :=
sorry

end min_distance_point_line_origin_l798_798872


namespace num_integers_in_solution_set_l798_798545

theorem num_integers_in_solution_set : 
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), |x - 3| ≤ 7.5 → abs (x - 3) ≤ 7.5 := 
by
  sorry

end num_integers_in_solution_set_l798_798545


namespace inscribed_square_area_ratio_l798_798135

theorem inscribed_square_area_ratio (a : ℝ) (b : ℝ) (h₁ : a = 4 * b) (h₂ : ∀ x y : ℝ, x = b → y = 3 * b → ∃ s : ℝ, s * √2 = x ∧ s * √2 = y) :
  (3 * b * √2 / 4) ^ 2 / a ^ 2 = 9 / 8 :=
by
  sorry

end inscribed_square_area_ratio_l798_798135


namespace cistern_emptied_by_pipes_l798_798020

theorem cistern_emptied_by_pipes :
  let rate1 := (3 / 4) / 12,
      rate2 := (1 / 2) / 15,
      time1 := 4,
      time2 := time1 + 6,
      empty_rate_combined := time1 * rate1 + time2 * (rate2 - rate1),
      empty_first_part := time1 * rate1 + time1 * rate2,
      empty_additional := 6 * rate2 in
  (empty_first_part + empty_additional) = 7 / 12 :=
by
  let rate1 := (3 / 4) / 12
  let rate2 := (1 / 2) / 15
  let time1 := 4
  let time2 := time1 + 6
  let empty_rate_combined := time1 * rate1 + time2 * (rate2 - rate1)
  let empty_first_part := time1 * rate1 + time1 * rate2
  let empty_additional := 6 * rate2
  have h_rate1 : rate1 = 1 / 16 := by sorry
  have h_rate2 : rate2 = 1 / 30 := by sorry
  have h_empty_first_part : empty_first_part = 7 / 20 := by sorry
  have h_empty_additional : empty_additional = 1 / 5 := by sorry
  have h_total : 7 / 20 + 1 / 5 = 7 / 12 := by sorry
  show (empty_first_part + empty_additional) = 7 / 12 from h_total
  sorry

end cistern_emptied_by_pipes_l798_798020


namespace factorial_difference_multiple_of_six_l798_798850

theorem factorial_difference_multiple_of_six (n : ℤ) (h : n ≥ 7) :
  (∃ k : ℤ, (∑ (i : ℤ) in (finset.range (n + 3)) + - ∑ (i : ℤ) in (finset.range (n + 2))) = 6 * k) := by
  sorry

end factorial_difference_multiple_of_six_l798_798850


namespace sum_first_six_geom_seq_l798_798811

-- Definitions based on the conditions given in the problem
def a : ℚ := 1 / 6
def r : ℚ := 1 / 2
def n : ℕ := 6

-- Statement to prove the desired result
theorem sum_first_six_geom_seq : 
  geom_series a r n = 21 / 64 := by
  sorry

end sum_first_six_geom_seq_l798_798811


namespace proof_l798_798877

variable {α β : ℝ}

noncomputable def cos_tan_given (α : ℝ) := 
0 < α ∧ α < π / 2 ∧ tan α = 4 * sqrt 3 → cos α = 1 / 7

noncomputable def beta_value_given (α β : ℝ) := 
0 < β ∧ β < α ∧ α < π / 2 ∧ tan α = 4 * sqrt 3 ∧ cos (β - α) = 13 / 14 → β = π / 3

theorem proof (h1: cos_tan_given α) (h2: beta_value_given α β): 
cos_tan_given α ∧ beta_value_given α β :=
by sorry

end proof_l798_798877


namespace train_speed_l798_798788

theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 125) (h_bridge : length_bridge = 250) (h_time : time = 30) :
    (length_train + length_bridge) / time * 3.6 = 45 := by
  sorry

end train_speed_l798_798788


namespace angle_BAC_is_85_degrees_l798_798319

theorem angle_BAC_is_85_degrees
(point_D_on_AC : ∃ D : Point, lies_on D AC)
(angle_ABD : ∠ABD = 25)
(angle_DBC : ∠DBC = 40)
(angle_ACB : ∠ACB = 30) :
∠BAC = 85 :=
by
  -- We are required to prove that ∠BAC = 85 given the conditions.
  sorry

end angle_BAC_is_85_degrees_l798_798319


namespace initial_charge_bike_rental_l798_798308

theorem initial_charge_bike_rental
  (initial_charge : ℕ)
  (hours : ℕ)
  (hourly_rate : ℕ)
  (total_paid : ℕ)
  (total_hours_cost : ℕ) :
  total_paid = initial_charge + total_hours_cost →
  total_hours_cost = hourly_rate * hours →
  hours = 9 →
  hourly_rate = 7 →
  total_paid = 80 →
  initial_charge = 17 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h2
  rw h2 at h1
  rw h5 at h1
  linarith

end initial_charge_bike_rental_l798_798308


namespace prob_3_digit_in_set_S_l798_798977

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def has_digit_one (n : ℕ) : Prop :=
  n.toString.to_list.contains '1'

def in_set_S (n : ℕ) : Prop :=
  is_multiple_of_3 n ∧ has_digit_one n

def three_digit_nums : Finset ℕ :=
  Finset.range 1000 \ Finset.range 100

def set_S : Finset ℕ :=
  three_digit_nums.filter in_set_S

noncomputable def probability_in_set_S : Rat :=
  (set_S.card : ℚ) / (three_digit_nums.card : ℚ)

theorem prob_3_digit_in_set_S :
  probability_in_set_S = 41 / 450 ∧ 41 + 450 = 491 :=
by
  unfold probability_in_set_S
  simp only [three_digit_nums, set_S, in_set_S, is_multiple_of_3, has_digit_one]
  sorry

end prob_3_digit_in_set_S_l798_798977


namespace find_sin_cos_of_perpendicular_vectors_l798_798903

theorem find_sin_cos_of_perpendicular_vectors 
  (θ : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_a : a = (Real.sin θ, -2)) 
  (h_b : b = (1, Real.cos θ)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_theta_range : 0 < θ ∧ θ < Real.pi / 2) : 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ Real.cos θ = Real.sqrt 5 / 5 := 
by 
  sorry

end find_sin_cos_of_perpendicular_vectors_l798_798903


namespace number_of_elements_with_square_factors_greater_than_one_l798_798912

def has_perfect_square_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k * k ∣ n

def count_numbers_with_perfect_square_factors (s : Finset ℕ) : ℕ :=
  s.filter has_perfect_square_factor |>.card

theorem number_of_elements_with_square_factors_greater_than_one :
  count_numbers_with_perfect_square_factors (Finset.range 101) = 43 :=
by 
  sorry

end number_of_elements_with_square_factors_greater_than_one_l798_798912


namespace hyperbola_eccentricity_l798_798896

-- Hyperbola defined by: x^2 / a^2 - y^2 / b^2 = 1
-- The given conditions
variables (a b c e : ℝ) (λ μ : ℝ)
variables (a_pos : a > 0) (b_pos : b > 0)
variables (focus_eq : c^2 = a^2 + b^2)
variables (lambda_mu_eq : λ^2 + μ^2 = 5/8)
variables (vec_op_eq : 4 * c^2 = a^2 + 3 * b^2)

-- The goal of the problem:
theorem hyperbola_eccentricity : e = 2 * real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_l798_798896


namespace remainder_y150_div_yminus2_4_l798_798153

theorem remainder_y150_div_yminus2_4 (y : ℝ) :
  (y ^ 150) % ((y - 2) ^ 4) = 554350 * (y - 2) ^ 3 + 22350 * (y - 2) ^ 2 + 600 * (y - 2) + 8 * 2 ^ 147 :=
by
  sorry

end remainder_y150_div_yminus2_4_l798_798153


namespace radius_of_tangent_circle_l798_798817

/-- Proof of the radius of a circle given six congruent parabolas arranged 
    with tangents and specific conditions -/
theorem radius_of_tangent_circle :
  ∀ (parabolas : Fin 6 → ℝ → ℝ),
  (∀ i, parabolas i = λ x, (x - 1) ^ 2) →
  ∀ (circle : ℝ → ℝ → Prop),
  (circle 0 0 = λ r, true) → 
  (∀ i, circle (1 + r * Real.cos (2 * i * Real.pi / 6)) (r * Real.sin (2 * i * Real.pi / 6)) parabolas i) → 
  ∀ theta, theta = 60 →
  (∃ r, (y = sqrt(3) * x ↔ (for x : ℝ, (x-1)^2 + r = sqrt(3)*x)) → (3 - 4 * r = 0) ↔ r = 3/4) ∧
  r = 3/4
:= by 
  sorry

end radius_of_tangent_circle_l798_798817


namespace pizza_slices_l798_798079

theorem pizza_slices (L : ℕ) : 
    (4 * 3) + (2 * L) = 3 + (3 + 1) + ((3 + 1) / 2) + (3 * 3) + 10 → 
    L = 8 :=
by {
    -- Unpack the conditions
    have GeorgeSlices : ℕ := 3,
    have BobSlices : ℕ := GeorgeSlices + 1,
    have SusieSlices : ℕ := BobSlices / 2,
    have BillFredMarkSlices : ℕ := 3 * 3,
    
    -- Total slices eaten calculation
    have TotalSlicesEaten : ℕ := GeorgeSlices + BobSlices + SusieSlices + BillFredMarkSlices,
    
    -- Total slices available in small pizzas calculation
    have SmallPizzaSlices : ℕ := 4 * 3,

    -- Set up the given equation
    rw add_assoc 12 (2 * L) 18 at TotalSlices,

    -- Solve for L
    have h : 12 + 2 * L = 18 + 10,
    sorry
  }

end pizza_slices_l798_798079


namespace div_problem_l798_798366

theorem div_problem : 150 / (6 / 3) = 75 := by
  sorry

end div_problem_l798_798366


namespace domain_of_f_l798_798679

def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 1) + 2 ^ (x - 2)

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 1} = {x : ℝ | ¬((Real.log (x + 1) / (x - 1) + 2 ^ (x - 2) = f x) ∧ (x + 1 ≤ 0 ∨ x - 1 = 0))} :=
sorry

end domain_of_f_l798_798679


namespace problem1_problem2_l798_798754

-- Problem 1: Prove the evaluation of a complex mathematical expression.
theorem problem1 :
  ( ((3 * (13 / 81)) ^ (-3) ) ^ (1 / 6) - Real.log 100⁻¹ - (Real.log (Real.sqrt Real.exp 1))⁻¹ + (0.1)⁻² -
    (2 + 10 / 27) ^ (- 2 / 3) - (1 / (2 + Real.sqrt 3)) ^ 0 + 2 ^ (-(1 + Real.log₂ (1 / 6))) ) = 102 :=
sorry

-- Problem 2: Given tan(π - α) = -2, prove the sine and cosine expression results in 2/5.
theorem problem2 {α : Real} (h : Real.tan (π - α) = -2) :
  (Real.sin (π + α))^2 + Real.sin (π / 2 + α) * Real.cos (3 * π / 2 - α) = 2 / 5 :=
sorry

end problem1_problem2_l798_798754


namespace circles_externally_tangent_l798_798218

theorem circles_externally_tangent
  (r1 r2 d : ℝ)
  (hr1 : r1 = 2) (hr2 : r2 = 3)
  (hd : d = 5) :
  r1 + r2 = d :=
by
  sorry

end circles_externally_tangent_l798_798218


namespace probability_equal_pairs_sum_l798_798169

theorem probability_equal_pairs_sum (D : Fin 6 → ℕ) 
    (h1 : ∀ (i : Fin 6), 1 ≤ D i ∧ D i ≤ 6)
    (h2 : ∀ i j : Fin 6, i ≠ j → D i ≠ D j) :
    (∃ f : {i : Fin 6} → ℕ // (∀ i, f i ∈ {1, 2, 3, 4, 5, 6}) ∧ 
    (∃ (a b c d : Fin 6), a ≠ b ∧ c ≠ d ∧ D a + D b = D c + D d)) → 
    (D a + D b = D c + D d) :=
by 
    sorry

end probability_equal_pairs_sum_l798_798169


namespace common_ratio_geometric_sequence_l798_798930

theorem common_ratio_geometric_sequence (q : ℝ) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = q)
  (h₂ : a 3 = q^2)
  (h₃ : (4 * a 1 + a 3 = 2 * 2 * a 2)) :
  q = 2 :=
by sorry

end common_ratio_geometric_sequence_l798_798930


namespace metal_waste_is_48_units_l798_798417

-- Definitions for the conditions
def rectangle_length := 10
def rectangle_width := 8

def circle_radius := rectangle_width / 2

def circle_area := Float.pi * circle_radius^2

def square_side := (2 * circle_radius) / Math.sqrt 2

def square_area := square_side^2

def rectangle_area := rectangle_length * rectangle_width

def wasted_metal := rectangle_area - circle_area + (circle_area - square_area)

-- The theorem to prove
theorem metal_waste_is_48_units : wasted_metal = 48 :=
by
  -- Proof goes here, but we'll skip it with 'sorry'
  sorry

end metal_waste_is_48_units_l798_798417


namespace eccentricity_of_given_ellipse_l798_798680

def ellipseEccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_given_ellipse :
  ellipseEccentricity 3 2 = Real.sqrt 5 / 3 :=
by
  -- The provided conditions in the problem lead to the necessary proof steps
  sorry

end eccentricity_of_given_ellipse_l798_798680


namespace correct_statements_l798_798796

theorem correct_statements : 
  (let statement1 := "The radius of a sphere is the line segment from any point on the sphere to the center of the sphere"
  in let statement2 := "The diameter of a sphere is the line segment between any two points on the sphere"
  in let statement3 := "Cutting a sphere with a plane results in a circle"
  in let statement4 := "The circle obtained by cutting a sphere with a plane not passing through the center of the sphere is called a small circle"
  in statement1 ∧ statement4) :=
by
  let statement1 := "The radius of a sphere is the line segment from any point on the sphere to the center of the sphere"
  let statement2 := "The diameter of a sphere is the line segment between any two points on the sphere"
  let statement3 := "Cutting a sphere with a plane results in a circle"
  let statement4 := "The circle obtained by cutting a sphere with a plane not passing through the center of the sphere is called a small circle"
  sorry

end correct_statements_l798_798796


namespace binomial_10_9_l798_798446

theorem binomial_10_9 : Nat.binomial 10 9 = 10 := by
  sorry

end binomial_10_9_l798_798446


namespace curve_C_rectangular_line_l_rectangular_max_distance_to_line_l_l798_798256

variables (θ : ℝ) (x y : ℝ)

def curve_C := ∃ θ, x = sqrt 3 * cos θ ∧ y = sin θ
def line_l := ∃ ρ, ρ * sin (θ + π / 4) = sqrt 2

theorem curve_C_rectangular :
  (∃ θ, x = sqrt 3 * cos θ ∧ y = sin θ) → (x^2 / 3 + y^2 = 1) :=
sorry

theorem line_l_rectangular :
  (∃ ρ, ρ * sin (θ + π / 4) = sqrt 2) → (x + y = 2) :=
sorry

theorem max_distance_to_line_l :
  (∃ θ, x = sqrt 3 * cos θ ∧ y = sin θ) → (∃ Q, 
  (x_Q, y_Q) ∈ curve_C ∧ distance (x_Q, y_Q) line_l = 2 * sqrt 2) :=
sorry

end curve_C_rectangular_line_l_rectangular_max_distance_to_line_l_l798_798256


namespace five_aliens_have_more_limbs_than_five_martians_l798_798430

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end five_aliens_have_more_limbs_than_five_martians_l798_798430


namespace displacement_during_interval_l798_798403

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem displacement_during_interval :
  (∫ t in (0 : ℝ)..3, velocity t) = 36 :=
by
  sorry

end displacement_during_interval_l798_798403


namespace integral_sqrt_quadratic_lemma_l798_798809

theorem integral_sqrt_quadratic_lemma :
  (∫ x in 0..1, real.sqrt (2 * x - x^2) - x) = (real.pi - 2) / 4 :=
sorry

end integral_sqrt_quadratic_lemma_l798_798809


namespace function_range_l798_798847

-- Define the function
def f (x : ℝ) : ℝ := (3 * x - 5) / (x + 4)

-- State the theorem
theorem function_range : set.range f = {y : ℝ | y ≠ 3} :=
by sorry

end function_range_l798_798847


namespace sphere_radius_between_cones_l798_798006

theorem sphere_radius_between_cones
  (R r : ℝ)
  (h_cones_heights : ∀ (C1 C2 C3 : ℝ), C1 = 3 * R ∧ C2 = 3 * R ∧ C3 = 3 * R)
  (h_planes: ∀ (P : ℝ), P = α)
  (h_conditions : ∀ (O O1 O2 O3 : ℝ), (O = ⟨0, 0, 0⟩) ∧ (O1 = ⟨R, 0, 0⟩) ∧ (O2 = ⟨−R / 2, R * √3 / 2, 0⟩) ∧ (O3 = ⟨−R / 2, −R * √3 / 2, 0⟩))
  :
  r = (2 * R * (2 * Real.sqrt 3 - 3)) / 3 :=
by 
  sorry

end sphere_radius_between_cones_l798_798006


namespace mistaken_quotient_correct_l798_798247

theorem mistaken_quotient_correct :
  ∀ (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ),
  correct_divisor = 21 → 
  mistaken_divisor = 12 → 
  correct_quotient = 32 → 
  D = correct_divisor * correct_quotient →
  D / mistaken_divisor = 56 :=
begin
  intros D correct_divisor mistaken_divisor correct_quotient hcd hmd hcq hD,
  rw [hcd, hmd, hcq] at hD,
  simp at hD,
  sorry
end

end mistaken_quotient_correct_l798_798247


namespace vectors_parallel_l798_798559

-- Let s and n be the direction vector and normal vector respectively
def s : ℝ × ℝ × ℝ := (2, 1, 1)
def n : ℝ × ℝ × ℝ := (-4, -2, -2)

-- Statement that vectors s and n are parallel
theorem vectors_parallel : ∃ (k : ℝ), n = (k • s) := by
  use -2
  simp [s, n]
  sorry

end vectors_parallel_l798_798559


namespace sequence_bound_l798_798784

theorem sequence_bound (n : ℕ) (h : n = 50000) :
  ∃ (x : ℕ → ℝ), x 1 = 0 ∧ (∀ i < n, x (i+1) = x i + 1 / 30000 * sqrt (1 - (x i)^2)) → x n < 1 → false :=
sorry

end sequence_bound_l798_798784


namespace grasshoppers_after_transformations_l798_798706

-- Define initial conditions and transformation rules
def initial_crickets : ℕ := 30
def initial_grasshoppers : ℕ := 30

-- Define the transformations
def red_haired_transforms (g : ℕ) (c : ℕ) : ℕ × ℕ :=
  (g - 4, c + 1)

def green_haired_transforms (c : ℕ) (g : ℕ) : ℕ × ℕ :=
  (c - 5, g + 2)

-- Define the total number of transformations and the resulting condition
def total_transformations : ℕ := 18
def final_crickets : ℕ := 0

-- The proof goal
theorem grasshoppers_after_transformations : 
  initial_grasshoppers = 30 → 
  initial_crickets = 30 → 
  (∀ t, t = total_transformations → 
          ∀ g c, 
          (g, c) = (0, 6) → 
          (∃ m n, (m + n = t ∧ final_crickets = c))) →
  final_grasshoppers = 6 :=
by
  sorry

end grasshoppers_after_transformations_l798_798706


namespace triangle_area_is_half_l798_798707

-- Define vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 1)

-- Function to calculate the area of the triangle based on given vertices
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove
theorem triangle_area_is_half :
  area_triangle A B C = 1 / 2 := by
  sorry

end triangle_area_is_half_l798_798707


namespace quadrilateral_AC_perpendicular_BD_iff_l798_798087

section

variables {A B C D : EuclideanSpace ℝ (Fin 2)} 

def dist_squared (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := ∥P - Q∥^2

noncomputable def perpendicular_diagonals (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
((C - A) ⬝ (D - B)) = 0

theorem quadrilateral_AC_perpendicular_BD_iff (A B C D : EuclideanSpace ℝ (Fin 2)) :
  perpendicular_diagonals A B C D ↔ dist_squared A B + dist_squared C D = dist_squared B C + dist_squared D A :=
sorry

end

end quadrilateral_AC_perpendicular_BD_iff_l798_798087


namespace monotonicity_of_g_range_of_a_l798_798894

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - x + (1/2) * x^2 - (1/3) * a * x^3
noncomputable def g (x a : ℝ) : ℝ := (f x a).derivative

theorem monotonicity_of_g (a : ℝ) : 
    (∀ x : ℝ, x > 0 → g x a > 0 ∨ g x a < 0) :=
sorry

theorem range_of_a (a : ℝ) : 
    (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ g x₁ a = 0 ∧ g x₂ a = 0 ∧ x₁ < x₂) → 0 < a ∧ a < 1 :=
sorry

end monotonicity_of_g_range_of_a_l798_798894


namespace number_of_girls_l798_798673

theorem number_of_girls (n : ℕ) (A : ℝ) 
    (h1 : A = (n * (A + 1) + 55 - 80) / n) : n = 25 :=
by 
  sorry

end number_of_girls_l798_798673


namespace isosceles_triangle_area_proof_l798_798955

-- Define the given conditions and structures
variables (A B C D E F G : Type*)
variables [IsoscelesTriangle A B C] [CircleInscribed A B C (3 : ℝ)]

-- Conditions for points and lines
variables (AD AE : ℝ) (hAD : AD = 15) (hAE : AE = 9)
variables [PointOnExtendedLine AD A B D] [PointOnExtendedLine AE A C E]
variables [ParallelLinesThrough D AE] [ParallelLinesThrough E AD]
variables (F : IntersectionPointOfLines D E) [Collinear A F G (CircleDistinctPoint A)]

-- Define the area calculus for the given triangle CBG
variable (AreaToFraction : (Area (Triangle C B G)) = ((540 : ℝ) * Real.sqrt 3) / 174)

-- Define the proof problem
theorem isosceles_triangle_area_proof :
  (Area (Triangle C B G)) = ((540 : ℝ) * Real.sqrt 3) / 174 → 540 + 3 + 174 = 717 := by
  intros h
  -- We would normally provide proof here, but for the purpose of this step, we leave it as sorry
  sorry

end isosceles_triangle_area_proof_l798_798955


namespace cube_root_of_27_l798_798108

theorem cube_root_of_27 : real.cbrt 27 = 3 :=
sorry

end cube_root_of_27_l798_798108


namespace recurring_decimal_sum_l798_798138

theorem recurring_decimal_sum :
  (0.\overline{3} + 0.\overline{27} : ℚ) = 20 / 33 := by
  have h1 : 0.\overline{3} = 1 / 3 := by sorry
  have h2 : 0.\overline{27} = 3 / 11 := by sorry
  rw [h1, h2]
  exact (1 / 3 + 3 / 11).norm_num
  exact sorry

end recurring_decimal_sum_l798_798138


namespace kyle_origami_stars_l798_798278

/-- Kyle bought 2 glass bottles, each can hold 15 origami stars,
    then bought another 3 identical glass bottles.
    Prove that the total number of origami stars needed to fill them is 75. -/
theorem kyle_origami_stars : (2 * 15) + (3 * 15) = 75 := by
  sorry

end kyle_origami_stars_l798_798278


namespace exterior_angle_BDE_measure_l798_798782

-- Definitions of the conditions
variable (ABCD : Type) [square ABCD] -- regular square
variable (DAE : Type) [equilateral_triangle DAE] -- equilateral triangle
variable (coplanar : Type) [is_coplanar ABCD DAE] -- coplanar condition

-- Angle measures
variable (interior_angle_square : ℝ := 90)
variable (interior_angle_triangle : ℝ := 60)
variable (full_circle : ℝ := 360)

-- Statement to prove
theorem exterior_angle_BDE_measure : 
  interior_angle_square + interior_angle_triangle = 90 + 60 ∧
  full_circle - (90 + 60) = 210 → 
  measure_exterior_angle BDE = 210 := 
sorry

end exterior_angle_BDE_measure_l798_798782


namespace max_cross_section_area_l798_798078

def apothem (k : ℝ) : ℝ := k
def lateral_angle (α : ℝ) : ℝ := α

theorem max_cross_section_area
  (k α : ℝ) :
  let tan_α := Real.tan α in
  let cos_α := Real.cos α in
  let sin_2α := Real.sin (2 * α) in
  (∀ α, tan_α < 2 → (∀ k, (1 / 2) * k^2 * (1 + 3 * cos_α^2)) = (1 / 2) * k^2 * (1 + 3 * cos_α^2)) ∧
  (∀ α, tan_α ≥ 2 → (∀ k, 2 * k^2 * sin_2α) = 2 * k^2 * sin_2α)
:=
by
  intros
  sorry

end max_cross_section_area_l798_798078


namespace paint_cost_is_correct_l798_798385

-- Define provided conditions
def edge_length : ℝ := 10
def cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 10

-- Define the question and the expected answer as a proof problem
theorem paint_cost_is_correct : 
  let face_area := edge_length * edge_length,
      total_surface_area := 6 * face_area,
      quarts_needed := total_surface_area / coverage_per_quart,
      total_cost := quarts_needed * cost_per_quart
  in total_cost = 192 :=
by
  -- Implementation of the proof goes here
  sorry

end paint_cost_is_correct_l798_798385


namespace monotonicity_of_f_smallest_integer_t_for_g_l798_798985

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - a^2) * x + log x - x⁻¹

def g (x : ℝ) : ℝ := x * f 1 x + x^2 + 1

theorem monotonicity_of_f (a : ℝ) : 
  (a < 0 → ∀ x ∈ Ioo (0 : ℝ) (-a⁻¹), f a ⁻¹' {y | y > 0}) ∧ 
  (a < 0 → ∀ x ∈ Ioo (-a⁻¹, ∞), f a ⁻¹' {y | y < 0}) ∧ 
  (0 ≤ a ∧ a ≤ 1 → ∀ x ∈ (0, ∞), f a ⁻¹' {y | y > 0}) ∧ 
  (1 < a → ∀ x ∈ Ioo (0 : ℝ) ((a - 1)⁻¹), f a ⁻¹' {y | y > 0}) ∧ 
  (1 < a → ∀ x ∈ Ioo ((a - 1)⁻¹, ∞), f a ⁻¹' {y | y < 0}) :=
sorry

theorem smallest_integer_t_for_g : 
  ∃ t : ℤ, (∀ x ∈ (0, ∞), t ≥ g x) ∧ t = 0 :=
sorry

end monotonicity_of_f_smallest_integer_t_for_g_l798_798985


namespace regression_estimate_l798_798852

theorem regression_estimate : ∀ (x : ℝ), x = 28 → y = 4.75 * x + 2.57 → y = 135.57 :=
by
  intro x
  intro hx
  intro hy
  rw [hx] at hy
  linarith

end regression_estimate_l798_798852


namespace repeating_base_k_representation_l798_798851

theorem repeating_base_k_representation (k : ℕ) (h : k > 0) : 
  (∃ k : ℕ, k > 0 ∧ (0.\overline{142}_k = 0.142142..._k) ∧ (frac 8 75 = some k) = 19) :=
begin
  sorry
end

end repeating_base_k_representation_l798_798851


namespace factorization_l798_798143

-- Define the expression
def expression (m x : ℝ) : ℝ := m * x^2 - 4 * m

-- State the factorization theorem
theorem factorization (m x : ℝ) :
  expression m x = m * (x + 2) * (x - 2) := 
by
  -- Proof goes here
  sorry

end factorization_l798_798143


namespace length_BC_l798_798946

theorem length_BC (AD AB DC : ℝ) (h₁ : AD = 16) (h₂ : AB = 20) (h₃ : DC = 5) : 
  let BD := Math.sqrt (AB^2 - AD^2) in
  let BC := Math.sqrt (BD^2 + DC^2) in
  BC = 13 :=
by
  sorry

end length_BC_l798_798946


namespace shifted_parabola_eq_l798_798566

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l798_798566


namespace seq_geometric_1_seq_geometric_2_general_term_sum_first_n_terms_l798_798900

noncomputable def a : ℕ → ℤ
| 0     := 5
| 1     := 2
| (n+2) := 2 * a (n+1) + 3 * a n

theorem seq_geometric_1 
  (n : ℕ) : 
  (∀ n ≥ 0, a (n+1) + a n = 7 * 3^n) :=
sorry

theorem seq_geometric_2 
  (n : ℕ) : 
  (∀ n ≥ 0, a (n+1) - 3 * a n = -13 * (-1)^n) :=
sorry

theorem general_term 
  (n : ℕ) : 
  a n = (1 / 4 : ℤ) * (7 * 3^(n-1) + 13 * (-1)^(n-1)) :=
sorry

theorem sum_first_n_terms 
  (n : ℕ) : 
  (∑ k in range n, a k) = (3 / 4 : ℤ) + (7 / 8) * 3^n - (1 / 8) * (-1)^n :=
sorry

end seq_geometric_1_seq_geometric_2_general_term_sum_first_n_terms_l798_798900


namespace call_center_agents_ratio_l798_798057

theorem call_center_agents_ratio
  (a b : ℕ) -- Number of agents in teams A and B
  (x : ℝ) -- Calls each member of team B processes
  (h1 : (a : ℝ) / (b : ℝ) = 5 / 8)
  (h2 : b * x * 4 / 7 + a * 6 / 5 * x * 3 / 7 = b * x + a * 6 / 5 * x) :
  (a : ℝ) / (b : ℝ) = 5 / 8 :=
by
  sorry

end call_center_agents_ratio_l798_798057


namespace production_added_value_expression_max_added_value_l798_798058

variable (a x y : ℝ)

noncomputable def f (a x : ℝ) : ℝ := 8 * (a - x) * x^2

theorem production_added_value_expression (h₁ : y = f a (a/2)) : 
  f a x = 8 * (a - x) * x^2 := by
  sorry

theorem max_added_value (h₂ : x ∈ Ioo 0 (4 * a / 5)) 
                        (h₃ : ∃ y : ℝ, y = a^3 ∧ y = f a (2 * a / 3)) : 
  ∃ x : ℝ, x = (2 * a / 3) ∧ f a x = (32 / 27) * a^3 := by
  sorry

end production_added_value_expression_max_added_value_l798_798058


namespace find_x_value_l798_798018

def segment_lengths_divide_altitudes (a b c d : ℝ) : Prop :=
  ∃ (triangle : Type) (A B C D E : triangle) (f : triangle → ℝ) (h: triangle → triangle → ℝ),
    -- The lengths of the segments divided by the altitudes
    f(A) = 6 ∧ f(B) = 4 ∧ f(C) = 3 ∧ f(D) = d ∧
    -- Similar triangles condition is satisfied
    ∃ (FC CE : ℝ), CE = 3 ∧ FC = 3 + d ∧
    f(A) = 10 ∧
    f(D) = (10 * d) / (3 + d)

theorem find_x_value :
  ∀ x : ℝ, segment_lengths_divide_altitudes 6 4 3 x → x = 9 / 7 :=
by
  -- Proof is omitted as 'sorry' allows skipping the detailed proof steps
  sorry

end find_x_value_l798_798018


namespace sqrt_sum_eval_l798_798835

theorem sqrt_sum_eval : 
  (Real.sqrt 50 + Real.sqrt 72) = 11 * Real.sqrt 2 := 
by 
  sorry

end sqrt_sum_eval_l798_798835


namespace afternoon_more_than_evening_l798_798400

def campers_in_morning : Nat := 33
def campers_in_afternoon : Nat := 34
def campers_in_evening : Nat := 10

theorem afternoon_more_than_evening : campers_in_afternoon - campers_in_evening = 24 := by
  sorry

end afternoon_more_than_evening_l798_798400


namespace problem_statement_l798_798455

def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^2
def f_inv (x : ℝ) : ℝ := x / 2
def g_inv (x : ℝ) : ℝ := Real.sqrt x

theorem problem_statement : 
    f (g_inv (f_inv (f_inv (g (f 8))))) = 16 := 
by
    sorry

end problem_statement_l798_798455


namespace debby_photos_proof_l798_798310

theorem debby_photos_proof :
  ∀ (total_pics friend_pics : ℕ), total_pics = 86 → friend_pics = 63 → total_pics - friend_pics = 23 :=
by
  intros total_pics friend_pics h_total h_friends
  rw [h_total, h_friends]
  norm_num
  sorry

end debby_photos_proof_l798_798310


namespace simplification_of_complex_num_l798_798047

-- Define the imaginary unit 
def i : ℂ := Complex.I

-- Express the complex number in question
def complex_num : ℂ := (9 + 2 * i) / (2 + i)

-- Prove the statement
theorem simplification_of_complex_num : complex_num = 4 - i :=
by
  sorry

end simplification_of_complex_num_l798_798047


namespace genevieve_initial_amount_l798_798171

def cost_per_kg : ℕ := 8
def kg_bought : ℕ := 250
def short_amount : ℕ := 400
def total_cost : ℕ := kg_bought * cost_per_kg
def initial_amount := total_cost - short_amount

theorem genevieve_initial_amount : initial_amount = 1600 := by
  unfold initial_amount total_cost cost_per_kg kg_bought short_amount
  sorry

end genevieve_initial_amount_l798_798171


namespace ring_arrangements_leading_digits_l798_798510

-- Define the conditions
def number_of_rings : ℕ := 10
def select_rings : ℕ := 7
def fingers : ℕ := 4

def binom (n k : ℕ) : ℕ := Nat.choose n k
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the problem
def ring_arrangements : ℕ := binom number_of_rings select_rings * factorial select_rings * 40

-- Define the result
def result := 241

-- The theorem to prove
theorem ring_arrangements_leading_digits :
  leading_digits ring_arrangements = result := by
  sorry

#eval ring_arrangements

end ring_arrangements_leading_digits_l798_798510


namespace pipe_drain_rate_l798_798640

theorem pipe_drain_rate :
  let tank_capacity := 800
  let rate_A := 40
  let rate_B := 30
  let cycles := 16
  let time_per_cycle := 3
  let total_time := 48
  ∃ (rate_C : ℕ), 
  (rate_A + rate_B - rate_C) * cycles = tank_capacity ∧
  total_time / time_per_cycle = cycles ∧
  rate_C = 20 :=
by
  let tank_capacity := 800
  let rate_A := 40
  let rate_B := 30
  let cycles := 16
  let time_per_cycle := 3
  let total_time := 48
  use 20
  sorry

end pipe_drain_rate_l798_798640


namespace arithmetic_sequence_a10_l798_798584

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 7 = 9) (h2 : a 13 = -3) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) :
  a 10 = 3 :=
by sorry

end arithmetic_sequence_a10_l798_798584


namespace distance_between_meeting_points_is_48_l798_798364

noncomputable def distance_between_meeting_points 
    (d : ℝ) -- total distance between points A and B
    (first_meeting_from_B : ℝ)   -- distance of the first meeting point from B
    (second_meeting_from_A : ℝ) -- distance of the second meeting point from A
    (second_meeting_from_B : ℝ) : ℝ :=
    (second_meeting_from_B - first_meeting_from_B)

theorem distance_between_meeting_points_is_48 
    (d : ℝ)
    (hm1 : first_meeting_from_B = 108)
    (hm2 : second_meeting_from_A = 84) 
    (hm3 : second_meeting_from_B = d - 24) :
    distance_between_meeting_points d first_meeting_from_B second_meeting_from_A second_meeting_from_B = 48 := by
  sorry

end distance_between_meeting_points_is_48_l798_798364


namespace smallest_k_power_l798_798732

theorem smallest_k_power (k : ℕ) (hk : ∀ m : ℕ, m < 14 → 7^m ≤ 4^19) : 7^14 > 4^19 :=
sorry

end smallest_k_power_l798_798732


namespace cover_large_square_l798_798281

theorem cover_large_square :
  ∃ (small_squares : Fin 8 → Set (ℝ × ℝ)),
    (∀ i, small_squares i = {p : ℝ × ℝ | (p.1 - x_i)^2 + (p.2 - y_i)^2 < (3/2)^2}) ∧
    (∃ (large_square : Set (ℝ × ℝ)),
      large_square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 7 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7} ∧
      large_square ⊆ ⋃ i, small_squares i) :=
sorry

end cover_large_square_l798_798281


namespace magnitude_diff_l798_798495

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition_1 : ‖a‖ = 2 := sorry
def condition_2 : ‖b‖ = 2 := sorry
def condition_3 : ‖a + b‖ = Real.sqrt 7 := sorry

-- Proof statement
theorem magnitude_diff (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = Real.sqrt 7) : 
  ‖a - b‖ = 3 :=
sorry

end magnitude_diff_l798_798495


namespace arrangement_count_l798_798757

theorem arrangement_count :
  ∀ (A B C D E : Type), 
  (∀ (arrangement : list (A ∪ B ∪ C ∪ D ∪ E)), ¬((A ∈ arrangement ∧ C ∈ arrangement) ∧ adjacent A C ∨ (B ∈ arrangement ∧ C ∈ arrangement) ∧ adjacent B C)) →
  (number_of_possible_arrangements A B C D E = 36 : ℕ) :=
by
  sorry

end arrangement_count_l798_798757


namespace example_problem_l798_798828

-- Define the reasoning types as an inductive data type
inductive Reasoning
| Inductive
| Analogical
| Deductive

-- Definitions for each option reasoning process
def option_A : Reasoning := Reasoning.Inductive
def option_B : Reasoning := Reasoning.Analogical
def option_C : Reasoning := Reasoning.Deductive
def option_D : Reasoning := Reasoning.Deductive

-- Define the problem to prove option B is Analogical reasoning
theorem example_problem : (option_B = Reasoning.Analogical) := 
by exact eq.refl option_B

end example_problem_l798_798828


namespace inverse_log_value_l798_798883

noncomputable def f : ℝ → ℝ := sorry

-- Given that f is the inverse function of y = log2(x)
theorem inverse_log_value :
  (∀ x, f (log 2 x) = x) → 
  (∀ x, log 2 (f x) = x) → 
  f 3 = log 3 2 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end inverse_log_value_l798_798883


namespace frankies_card_number_is_103_l798_798157

-- Define the conditions
variables {a b c d e : ℕ}

-- Conditions given in the problem
axiom h1 : c = b * e
axiom h2 : a + b = d
axiom h3 : e - d = a

-- Define the front numbers on the cards
def card_numbers : list ℕ := [101, 102, 103, 104, 105]

-- Prove that 103 is the front number of the card with the largest integer on its reverse
theorem frankies_card_number_is_103 : 
  ∃ (c : ℕ), 
  (c = b * e) ∧ 
  (a + b = d) ∧ 
  (e - d = a) ∧ 
  (c > a) ∧ 
  (c > b) ∧ 
  (c > d) ∧ 
  (c > e) ∧ 
  (c = 103) := 
sorry

end frankies_card_number_is_103_l798_798157


namespace find_D_value_l798_798131

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def is_divisible_by_3 (n : Nat) : Prop :=
  n % 3 = 0

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10.sum

theorem find_D_value (A B D : Nat) (h1 : is_divisible_by_4 (10 * B + 2))
  (h2 : is_divisible_by_3 (sum_of_digits (52340 * 10 + 4000 * B + 30 * A + D))) 
  (h3 : A = 6) : 
  D = 5 :=
by
  sorry

end find_D_value_l798_798131


namespace task_pages_l798_798376

theorem task_pages (A B T : ℕ) (hB : B = A + 5) (hTogether : (A + B) * 18 = T)
  (hAlone : A * 60 = T) : T = 225 :=
by
  sorry

end task_pages_l798_798376


namespace problem_equivalence_of_angles_l798_798654

noncomputable def ctg (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem problem_equivalence_of_angles
  (a b c t S ω : ℝ)
  (hS : S = Real.sqrt ((a^2 + b^2 + c^2)^2 + (4 * t)^2))
  (h1 : ctg ω = (a^2 + b^2 + c^2) / (4 * t))
  (h2 : Real.cos ω = (a^2 + b^2 + c^2) / S)
  (h3 : Real.sin ω = (4 * t) / S) :
  True :=
sorry

end problem_equivalence_of_angles_l798_798654


namespace num_factors_of_M_l798_798979

theorem num_factors_of_M :
  let M : ℕ := 31^3 + 3 * 31^2 + 3 * 31 + 1
  number_of_factors M = 16 :=
by
  let M : ℕ := 31^3 + 3 * 31^2 + 3 * 31 + 1
  sorry

end num_factors_of_M_l798_798979


namespace max_min_distance_between_lines_l798_798904

theorem max_min_distance_between_lines 
  (a b c : ℝ) 
  (h1 : a + b = -1)
  (h2 : a * b = c)
  (h3 : 0 ≤ c)
  (h4 : c ≤ 1 / 8) :
  let d := abs (a - b) / real.sqrt 2 in
  d ≤ real.sqrt 2 / 2 ∧ d ≥ 1 / 2 :=
by
  sorry

end max_min_distance_between_lines_l798_798904


namespace five_more_limbs_l798_798429

-- Definition of the number of limbs an alien has
def alien_limbs : ℕ := 3 + 8

-- Definition of the number of limbs a Martian has
def martian_limbs : ℕ := (8 / 2) + (3 * 2)

-- The main statement that we need to prove
theorem five_more_limbs : 5 * alien_limbs - 5 * martian_limbs = 5 := by
  have h1 : alien_limbs = 11 := rfl
  have h2 : martian_limbs = 10 := rfl
  calc
    5 * alien_limbs - 5 * martian_limbs
        = 5 * 11 - 5 * 10 := by rw [h1, h2]
    ... = 55 - 50     := by rfl
    ... = 5           := by rfl

end five_more_limbs_l798_798429


namespace greatest_possible_integer_in_set_l798_798785

def is_median (s: List ℕ) (m: ℕ) := s.sorted.get 2 = m

theorem greatest_possible_integer_in_set : 
  ∀ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ((a + b + c + d + e) / 5 = 20) ∧
  is_median [a, b, c, d, e] 18 → 
  max {a, b, c, d, e} = 60 :=
by
  intros
  sorry

end greatest_possible_integer_in_set_l798_798785


namespace intersection_line_canonical_equation_l798_798738

def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0
def canonical_equation (x y z : ℝ) : Prop := 
  (x - 1) / 35 = (y - 4 / 7) / 23 ∧ (y - 4 / 7) / 23 = z / 49

theorem intersection_line_canonical_equation (x y z : ℝ) :
  plane1 x y z → plane2 x y z → canonical_equation x y z :=
by
  intros h1 h2
  unfold plane1 at h1
  unfold plane2 at h2
  unfold canonical_equation
  sorry

end intersection_line_canonical_equation_l798_798738


namespace P_is_centroid_l798_798268

variable {A B C P : Type} [InnerProductSpace ℝ P]
variable [IsTriangle A B C]

def triangles_equal_area (A B C P : P) [IsInTriangle A B C P] :=
  Area(△ A B P) = Area(△ B C P) = Area(△ C A P)

theorem P_is_centroid (h : triangles_equal_area A B C P) : IsCentroid A B C P :=
  sorry

end P_is_centroid_l798_798268


namespace price_difference_is_99_cents_l798_798088

-- Definitions for the conditions
def list_price : ℚ := 3996 / 100
def discount_super_savers : ℚ := 9
def discount_penny_wise : ℚ := 25 / 100 * list_price

-- Sale prices calculated based on the given conditions
def sale_price_super_savers : ℚ := list_price - discount_super_savers
def sale_price_penny_wise : ℚ := list_price - discount_penny_wise

-- Difference in prices
def price_difference : ℚ := sale_price_super_savers - sale_price_penny_wise

-- Prove that the price difference in cents is 99
theorem price_difference_is_99_cents : price_difference = 99 / 100 := 
by
  sorry

end price_difference_is_99_cents_l798_798088


namespace extracurricular_hours_l798_798596

theorem extracurricular_hours :
  let soccer_hours_per_day := 2
  let soccer_days := 3
  let band_hours_per_day := 1.5
  let band_days := 2
  let total_soccer_hours := soccer_hours_per_day * soccer_days
  let total_band_hours := band_hours_per_day * band_days
  total_soccer_hours + total_band_hours = 9 := 
by
  -- The proof steps go here.
  sorry

end extracurricular_hours_l798_798596


namespace polynomial_irreducible_l798_798294

theorem polynomial_irreducible 
  (n : ℕ) (p : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime p) : 
  Irreducible (Polynomial.C (p^2) + Polynomial.X * (Polynomial.C (p^2) + Polynomial.X * (Polynomial.C (p^2) + Polynomial.X * (Polynomial.C (p^2) + Polynomial.X * ...))))
  sorry

end polynomial_irreducible_l798_798294


namespace AE_eq_BF_l798_798975

-- Define the geometric setup
variables {A B C P E F : Type} [InnerProductSpace ℝ ℂ]

-- Given conditions
def is_isosceles_triangle (A B C : ℂ) : Prop :=
  (dist A C) = (dist B C)

def on_arc_CA_not_containing_B (A B C P : ℂ) : Prop :=
  ∃ (circle : Circle), circle.Circumcircle A B C ∧ (arc circle A C P ∧ ¬ arc circle A B P)

def projection (a b : ℂ) : ℂ := orthogonal_projection (affineSpan ℝ ({a, b} : set ℂ))

def is_projection_of (a b p : ℂ) : Prop := projection a b = p

-- Main theorem
theorem AE_eq_BF
  (h_isosceles : is_isosceles_triangle A B C)
  (h_on_arc : on_arc_CA_not_containing_B A B C P)
  (h_proj_E : is_projection_of A P E)
  (h_proj_F : is_projection_of B P F) :
  dist A E = dist B F :=
by sorry

end AE_eq_BF_l798_798975


namespace complex_exponential_sum_l798_798233

theorem complex_exponential_sum (α β γ : ℂ) (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) + complex.exp (complex.I * γ) = 1 + complex.I) : 
  complex.exp (-complex.I * α) + complex.exp (-complex.I * β) + complex.exp (-complex.I * γ) = 1 - complex.I := 
  sorry

end complex_exponential_sum_l798_798233


namespace conjugate_of_z_l798_798337

noncomputable def z : ℂ := 2 / (1 - complex.i)

theorem conjugate_of_z : complex.conj z = 1 - complex.i :=
by sorry

end conjugate_of_z_l798_798337


namespace trapezoid_AM_constant_length_trapezoid_ratio_constant_l798_798099

theorem trapezoid_AM_constant_length (A B C D P O M N : Point)
  (h1 : is_trapezoid AB CD AD BC)
  (h2 : intersection P (diagonal AC BD))
  (h3 : intersection O (extended_lines BA CD))
  (h4 : extended_intersect OP AD M BC N) :
  AM_constant_length (AM) :=
sorry

theorem trapezoid_ratio_constant (A B C D P O M N : Point)
  (h1 : is_trapezoid AB CD AD BC)
  (h2 : intersection P (diagonal AC BD))
  (h3 : intersection O (extended_lines BA CD))
  (h4 : extended_intersect OP AD M BC N) :
  is_constant_ratio ((AB * OP) / (OD * PN)) ((DC * OP) / (OD * PN)) :=
sorry

end trapezoid_AM_constant_length_trapezoid_ratio_constant_l798_798099


namespace conditional_probability_P_A_given_B_l798_798867

-- Definitions based on conditions.
def three_digit_numbers : finset (fin 2) × (fin 2) × (fin 2) := 
  {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}

def event_A (num : (fin 2) × (fin 2) × (fin 2)) : Prop := num.2.1 == 0
def event_B (num : (fin 2) × (fin 2) × (fin 2)) : Prop := num.1 == 0

def P_event_A_and_B := (three_digit_numbers.filter (λ n, event_A n ∧ event_B n)).card.to_rat / three_digit_numbers.card.to_rat
def P_event_B := (three_digit_numbers.filter (λ n, event_B n)).card.to_rat / three_digit_numbers.card.to_rat

-- Lean 4 statement for the proof problem.
theorem conditional_probability_P_A_given_B : P_event_A_and_B / P_event_B = 1 / 2 := by
  sorry

end conditional_probability_P_A_given_B_l798_798867


namespace min_value_of_objective_function_in_triangle_region_l798_798209

theorem min_value_of_objective_function_in_triangle_region :
    let D := {(x, y) | x = 0 ∨ (x = sqrt 2 / 2 ∧ (y = sqrt 2 / 2 ∨ y = -sqrt 2 / 2))}
    ∃ (x y : ℝ), (x, y) ∈ D ∧ x - 2*y = -sqrt 2 / 2 :=
by
  sorry

end min_value_of_objective_function_in_triangle_region_l798_798209


namespace quadrilateral_side_length_l798_798415

-- Definitions
def inscribed_quadrilateral (a b c d r : ℝ) : Prop :=
  ∃ (O : ℝ) (A B C D : ℝ), 
    O = r ∧ 
    A = a ∧ B = b ∧ C = c ∧ 
    (r^2 + r^2 = (a^2 + b^2) / 2) ∧
    (r^2 + r^2 = (b^2 + c^2) / 2) ∧
    (r^2 + r^2 = (c^2 + d^2) / 2)

-- Theorem statement
theorem quadrilateral_side_length :
  inscribed_quadrilateral 250 250 100 200 250 :=
sorry

end quadrilateral_side_length_l798_798415


namespace james_sales_l798_798957

noncomputable def total_items_sold
  (houses_day1 : ℕ) (items_per_house_day1 : ℕ) (factor_day2 : ℕ) (percentage_sold_day2 : ℚ) (items_per_house_day2 : ℕ) : ℕ :=
 let sold_day1 := houses_day1 * items_per_house_day1 in
 let houses_day2 := houses_day1 * factor_day2 in
 let sold_day2 := (houses_day2 * percentage_sold_day2).toInt * items_per_house_day2 in
 sold_day1 + sold_day2

theorem james_sales : total_items_sold 20 2 2 0.8 2 = 104 :=
  by
    dsimp [total_items_sold]
    -- The calculation steps would be carried here if we were doing the proof.
    sorry

end james_sales_l798_798957


namespace Sasha_wins_l798_798693

def initial_number := 2018
def is_divisible_by_112 (n : ℕ) : Prop := n % 112 = 0
def append_digit (n : ℕ) (d : ℕ) : ℕ := 10 * n + d
def append_two_digits (n : ℕ) (d₁ d₂ : ℕ) : ℕ := 100 * n + 10 * d₁ + d₂

theorem Sasha_wins (N: ℕ) (d d₁ d₂ : ℕ) 
    (h1: N = initial_number)
    (h2: ∀ (N : ℕ), ¬is_divisible_by_112 (append_two_digits N d d₁ d₂))
    (h3: ∃ N : ℕ, length_of_number N = 2018)
    (h4: length_of_number initial_number <= 2018):
    ¬(∃ N: ℕ, is_divisible_by_112 N := N <= 2018) :=
  by
  sorry

end Sasha_wins_l798_798693


namespace number_of_ways_to_break_targets_l798_798935

theorem number_of_ways_to_break_targets : 
  let targets := ['A', 'A', 'B', 'B', 'C', 'C'] in 
  let number_of_orders := (targets.permutations.toFinset.card) / (2! * 2! * 2!) in 
  number_of_orders = 90 := 
begin
  sorry
end

end number_of_ways_to_break_targets_l798_798935


namespace is_possible_to_finish_7th_l798_798576

theorem is_possible_to_finish_7th 
  (num_teams : ℕ)
  (wins_ASTC : ℕ)
  (losses_ASTC : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ) 
  (total_points : ℕ)
  (rank_ASTC : ℕ)
  (points_ASTC : ℕ)
  (points_needed_by_top_6 : ℕ → ℕ)
  (points_8th_and_9th : ℕ) :
  num_teams = 9 ∧ wins_ASTC = 5 ∧ losses_ASTC = 3 ∧ points_per_win = 3 ∧ points_per_draw = 1 ∧ 
  total_points = 108 ∧ rank_ASTC = 7 ∧ points_ASTC = 15 ∧ points_needed_by_top_6 7 = 105 ∧ points_8th_and_9th ≤ 3 →
  ∃ (top_7_points : ℕ), 
  top_7_points = 105 ∧ (top_7_points + points_8th_and_9th) = total_points := 
sorry

end is_possible_to_finish_7th_l798_798576


namespace solve_for_x_and_value_l798_798659

theorem solve_for_x_and_value (x : ℝ) (h : 2^(2*x) + 4 = 6 * 2^x) : x^2 + 3 = 4 := 
by
  sorry

end solve_for_x_and_value_l798_798659


namespace imaginary_part_of_z_l798_798130
noncomputable def z : ℂ := (3 - 2 * (-1)) / (1 + complex.I)

theorem imaginary_part_of_z :
    (z.im = -5 / 2) :=
by
  sorry

end imaginary_part_of_z_l798_798130


namespace max_x2_cos2x_l798_798470

theorem max_x2_cos2x (x : ℝ) :
  let I := {x : ℝ | max (x^2) (real.cos (2*x)) < 1/2},
      lengths := 
        intervalLength (-real.sqrt 2 / 2, -real.pi / 6) + 
        intervalLength (real.pi / 6, real.sqrt 2 / 2)
  in 
  (∀ x, x ∈ I → someInterval x) ∧
  (lengths = real.sqrt 2 - real.pi / 3) ∧
  (round (lengths * 100) / 100 = 0.37) :=
sorry

end max_x2_cos2x_l798_798470


namespace minimum_sum_l798_798876

theorem minimum_sum 
  (a b c : Fin 10 → ℕ) 
  (ha : ∀ n ∈ Finset.univ, a n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) 
  (hb : ∀ n ∈ Finset.univ, b n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) 
  (hc : ∀ n ∈ Finset.univ, c n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) 
  (pa : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ∃ i, a i = n)
  (pb : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ∃ i, b i = n)
  (pc : ∀ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ∃ i, c i = n) :
  ∑ i in Finset.univ, a i * b i * c i = 930 :=
by sorry

end minimum_sum_l798_798876


namespace proposition_B_proposition_D_l798_798491

open Real

variable (a b : ℝ)

theorem proposition_B (h : a^2 ≠ b^2) : a ≠ b := 
sorry

theorem proposition_D (h : a > abs b) : a^2 > b^2 :=
sorry

end proposition_B_proposition_D_l798_798491


namespace complex_point_in_fourth_quadrant_l798_798698

noncomputable def complex_point : ℂ := (2 - complex.i) / (1 + complex.i)

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_point_in_fourth_quadrant : fourth_quadrant complex_point :=
by
  -- Here we would include the detailed proof steps in a complete solution.
  sorry

end complex_point_in_fourth_quadrant_l798_798698


namespace count_ones_digits_of_numbers_divisible_by_4_and_3_l798_798136

theorem count_ones_digits_of_numbers_divisible_by_4_and_3 :
  let eligible_numbers := { n : ℕ | n < 100 ∧ n % 4 = 0 ∧ n % 3 = 0 }
  ∃ (digits : Finset ℕ), 
    (∀ n ∈ eligible_numbers, n % 10 ∈ digits) ∧
    digits.card = 5 :=
by
  sorry

end count_ones_digits_of_numbers_divisible_by_4_and_3_l798_798136


namespace arithmetic_sequence_proof_l798_798887

def arithmetic_seq_formula (a_n : ℕ → ℕ) (n : ℕ) : Prop := 
  ∀ n, a_n n = 2 * n + 1

def sum_first_n_b (S_n : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ n, S_n n = n^2 + 2 * n - 3 / 2 + 3^((n + 1)) / 2

theorem arithmetic_sequence_proof :
  ∀ (a_n : ℕ → ℕ) (b_n : ℕ → ℤ),
  (a_n 5 = 11 ∧ a_n 2 + a_n 6 = 18) ∧ 
  (∀ n, b_n n = (a_n n : ℤ) + (3^n : ℤ)) →
  (arithmetic_seq_formula a_n ∧ sum_first_n_b (λ n, ∑ i in finset.range n, b_n (i + 1)))
:= by
  sorry

end arithmetic_sequence_proof_l798_798887


namespace eight_distinct_solutions_l798_798619

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

theorem eight_distinct_solutions : 
  ∃ S : Finset ℝ, S.card = 8 ∧ ∀ x ∈ S, f (f (f x)) = x :=
sorry

end eight_distinct_solutions_l798_798619


namespace log_product_l798_798440

open Real

theorem log_product : log 9 / log 2 * (log 5 / log 3) * (log 8 / log (sqrt 5)) = 12 :=
by
  sorry

end log_product_l798_798440


namespace find_a_conditions_l798_798848

theorem find_a_conditions (a : ℝ) : 
    (∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3) ↔ 
    (∃ n : ℤ, a = n + 1/2 ∨ a = n + 1/3 ∨ a = n - 1/3) :=
by
  sorry

end find_a_conditions_l798_798848


namespace square_area_increase_l798_798389

theorem square_area_increase (s : ℝ) (h : s > 0) :
  ((1.15 * s) ^ 2 - s ^ 2) / s ^ 2 * 100 = 32.25 :=
by
  sorry

end square_area_increase_l798_798389


namespace distribution_schemes_l798_798830

def students : Finset ℕ := {1, 2, 3, 4}  -- representing 4 students with natural numbers
def villages : Finset ℕ := {1, 2, 3}     -- representing 3 villages with natural numbers

/-- Proving the number of ways to distribute 4 students to 3 villages,
    with each village having at least one student, is 36. -/
theorem distribution_schemes : 
  (∃ f : students → villages, 
    ∀ v ∈ villages, ∃ s ∈ students, f s = v) → 
  card (set_of (λ (f : students → villages), 
    ∀ v ∈ villages, ∃ s ∈ students, f s = v)) = 36 := 
by 
  sorry

end distribution_schemes_l798_798830


namespace range_of_a_l798_798531

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - a / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem range_of_a (e : ℝ) (a : ℝ) (H : ∀ x ∈ Set.Icc 1 e, f x ≥ g x a) :
  -2 ≤ a ∧ a ≤ (2 * e) / (e - 1) :=
by
  sorry

end range_of_a_l798_798531


namespace cistern_full_in_80_hours_l798_798767

theorem cistern_full_in_80_hours (C : ℝ) (hA : C > 0) (hB : C > 0) :
  (C / (C / 80)) = 80 :=
by
  -- introduction of the given conditions and definitions
  let rate_A := C / 16
  let rate_B := C / 20
  let net_rate := rate_A - rate_B
  have h_net_rate : net_rate = C / 80, by sorry
  have h_time : C / net_rate = 80, by sorry
  exact h_time

end cistern_full_in_80_hours_l798_798767


namespace tangent_line_at_one_monotonicity_intervals_inequality_proof_l798_798892

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log x - 1) / x - a * x

theorem tangent_line_at_one (a : ℝ) : a = 2 → ∃ (y : ℝ), y = -3 :=
by
  intro h
  have f_x := f 1 a
  exact Exists.intro (-3) sorry

theorem monotonicity_intervals (a : ℝ) : a = 2 → 
  (∀ x > 0, x < 1 → (∃ f', deriv (f x a) = f') → deriv (f x a) > 0) ∧ (∀ x > 1, (∃ f', deriv (f x a) = f') → deriv (f x a) < 0) :=
by
  intro h
  split
  { intros x hx1 hx2 hf'
    sorry }
  { intros x hx hf'
    sorry }

theorem inequality_proof (a : ℝ) : 1 < a ∧ a < 2 → ∀ x > 0, f x a < -1 :=
by
  intro h
  intro x
  intro hx
  sorry

end tangent_line_at_one_monotonicity_intervals_inequality_proof_l798_798892


namespace angle_of_inclination_range_l798_798865

noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -4 * Real.exp x / (Real.exp x + 1) ^ 2

theorem angle_of_inclination_range (x : ℝ) (a : ℝ) 
  (hx : tangent_slope x = Real.tan a) : 
  (3 * Real.pi / 4 ≤ a ∧ a < Real.pi) :=
by 
  sorry

end angle_of_inclination_range_l798_798865


namespace james_total_sales_l798_798959

noncomputable def total_items_sold : ℕ :=
  let h1 := 20 in
  let h2 := 2 * h1 in
  let h2_sold := (0.8 * h2).toNat in -- Using toNat to convert to natural number
  2 * (h1 + h2_sold)

theorem james_total_sales : total_items_sold = 104 := by
  sorry

end james_total_sales_l798_798959


namespace northern_walks_length_6_l798_798341

def is_northern_walk (n : ℕ) (walk : list (ℤ × ℤ)) : Prop :=
  walk.head = (0, 0) ∧
  (∀ i, i < n - 1 → 
    ((walk.nth i).snd + 1 ≤ (walk.nth (i + 1)).snd ∧ 
    abs((walk.nth i).fst - (walk.nth (i + 1)).fst) ≤ 1)) ∧ 
  walk.nodup

theorem northern_walks_length_6 : ∃ n_walks, n_walks = 239 ∧
  ∀ walk, is_northern_walk 6 walk → walk.length = 6 :=
by
  sorry

end northern_walks_length_6_l798_798341


namespace vector_noncollinear_l798_798228

noncomputable def vector (n : ℕ) := ℝ^n

theorem vector_noncollinear (a b : vector 3) (λ : ℝ) :
  ¬ (b = λ • a) :=
sorry

end vector_noncollinear_l798_798228


namespace equal_distances_to_incenter_and_excenters_l798_798933

variables {α : Type*} [EuclideanGeometry α]

-- Definitions of the points and lengths involved
variables {A B C O O_a O_b : α}
variables {a b : ℝ}

-- Conditions given in the problem
def is_right_triangle (A B C : α) : Prop := ∠BCA = π / 2

def is_incenter (O A B C : α) : Prop := 
  is_intersection_of_angle_bisectors O A B C

def is_excenter_opposite_BC (O_a A B C : α) : Prop := 
  is_excenter_opposite_leg O_a A B C ∠C → ∠B

def is_excenter_opposite_AC (O_b A B C : α) : Prop := 
  is_excenter_opposite_leg O_b A B C ∠A → ∠B

-- The main theorem to be proved
theorem equal_distances_to_incenter_and_excenters 
  (h₁ : is_right_triangle A B C)
  (h₂ : is_incenter O A B C)
  (h₃ : is_excenter_opposite_BC O_a A B C)
  (h₄ : is_excenter_opposite_AC O_b A B C) 
  : dist A O = dist A O_b ∧ dist B O = dist B O_a :=
by sorry

end equal_distances_to_incenter_and_excenters_l798_798933


namespace sum_of_reciprocals_of_square_roots_l798_798248

-- Definition of conditions
def L_0 := [(100^2 : ℝ), (105^2 : ℝ)]
def r := λ (ra rb : ℝ), (ra * rb) / ((Real.sqrt ra + Real.sqrt rb)^2)

-- Proof problem statement
theorem sum_of_reciprocals_of_square_roots :
  let S := List.foldl
             (λ acc k, acc ++ (List.pairwise (λ a b, r a b) acc k))
             L_0 [1, 2, 3, 4, 5, 6, 7]
  in
  ∑ C in S, 1 / Real.sqrt C = 147 / 14 :=
by
  sorry

end sum_of_reciprocals_of_square_roots_l798_798248


namespace shaded_area_is_correct_l798_798770

-- Define the main triangle and its properties
structure Triangle :=
  (leg_length : ℕ)
  (is_isosceles_right : bool)

-- Define the partition characteristics
structure Partition :=
  (triangle : Triangle)
  (number_of_parts : ℕ)
  (congruent_parts : bool)

-- Define the shaded area characteristics
structure ShadedArea :=
  (pattern : List ℕ)

-- Main theorem statement
theorem shaded_area_is_correct :
  ∀ (T : Triangle) (P : Partition) (S : ShadedArea),
    T.leg_length = 12 →
    T.is_isosceles_right →
    P.triangle = T →
    P.number_of_parts = 36 →
    P.congruent_parts →
    S.pattern = [2, 4, 6] →
    (18 * (1 / 2 * T.leg_length * T.leg_length / P.number_of_parts)) = 36 :=
begin
    -- This is where the proof would go, but we use sorry for now.
    sorry
end

end shaded_area_is_correct_l798_798770


namespace determine_k_l798_798825

def f(x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g(x k : ℝ) : ℝ := x^3 - k * x - 10

theorem determine_k : 
  (f (-5) - g (-5) k = -24) → k = 61 := 
by 
-- Begin the proof script here
sorry

end determine_k_l798_798825


namespace greatest_k_consecutive_divisible_by_m_l798_798604

noncomputable def sequence (m : ℕ) (h: m > 1) : ℕ → ℕ
| 0       := 2^0
| (x+1) := if x < m-1 then 2^(x+1) else ∑ i in finset.range m, sequence m h (x - i)

theorem greatest_k_consecutive_divisible_by_m (m : ℕ) (hm: m > 1) :
  ∃ k, (∀ n, ∀ i < k, sequence m hm (n+i) % m = 0) ↔ k = m - 1 :=
sorry

end greatest_k_consecutive_divisible_by_m_l798_798604


namespace chad_sandwiches_l798_798111

-- Definitions representing the conditions
def crackers_per_sleeve : ℕ := 28
def sleeves_per_box : ℕ := 4
def boxes : ℕ := 5
def nights : ℕ := 56
def crackers_per_sandwich : ℕ := 2

-- Definition representing the final question about the number of sandwiches
def sandwiches_per_night (crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich : ℕ) : ℕ :=
  (crackers_per_sleeve * sleeves_per_box * boxes) / nights / crackers_per_sandwich

-- The theorem that states Chad makes 5 sandwiches each night
theorem chad_sandwiches :
  sandwiches_per_night crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich = 5 :=
by
  -- Proof outline:
  -- crackers_per_sleeve * sleeves_per_box * boxes = 28 * 4 * 5 = 560
  -- 560 / nights = 560 / 56 = 10 crackers per night
  -- 10 / crackers_per_sandwich = 10 / 2 = 5 sandwiches per night
  sorry

end chad_sandwiches_l798_798111


namespace evaluate_expression_l798_798137

theorem evaluate_expression : (3^1 - 2 + 6^2 - 0)⁻¹ * 3 = 3 / 37 := by
  sorry

end evaluate_expression_l798_798137


namespace sequence_inequality_l798_798605

theorem sequence_inequality (n : ℕ) (h : n > 0) :
  let initial_sequence := (List.range n).map (λ i, 1 / (i + 1 : ℝ)),
      avg_pairs := λ l : List ℝ, (l.init.zip l.tail).map (λ p, (p.1 + p.2) / 2),
      final_elem := List.foldr (λ _ l, avg_pairs l) initial_sequence (List.range (n - 1))
  in final_elem.head = 2 / n - 2 / (n * 2^n) ∧ final_elem.head < 2 / n :=
by
  sorry

end sequence_inequality_l798_798605


namespace remaining_digits_count_l798_798336

theorem remaining_digits_count 
  (avg9 : ℝ) (avg4 : ℝ) (avgRemaining : ℝ) (h1 : avg9 = 18) (h2 : avg4 = 8) (h3 : avgRemaining = 26) :
  let S := 9 * avg9
  let S4 := 4 * avg4
  let S_remaining := S - S4
  let N := S_remaining / avgRemaining
  N = 5 := 
by
  sorry

end remaining_digits_count_l798_798336


namespace values_of_a_l798_798853

noncomputable def system_has_four_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - 8) * Real.sin a - (y + 2) * Real.cos a = 0 ∧
               ((x - 3) ^ 2 + (y - 3) ^ 2 = 1 ∨ (x - 3) ^ 2 + (y - 3) ^ 2 = 9)

theorem values_of_a :
  {a | system_has_four_solutions a} = 
  {a | ∃ n : ℤ, 
        a ∈ Ioc (3 * Real.pi / 4 - Real.arcsin (Real.sqrt 2 / 10) + ↑n * Real.pi) 
                 (3 * Real.pi / 4 + Real.arcsin (Real.sqrt 2 / 10) + ↑n * Real.pi)} :=
by
  sorry

end values_of_a_l798_798853


namespace fraction_covered_by_mat_l798_798062

-- Define the necessary variables and conditions
def diameter_mat : ℝ := 18
def side_tabletop : ℝ := 24

-- Calculate the radius of the mat
def radius_mat : ℝ := diameter_mat / 2

-- Calculate the area of the mat
def area_mat : ℝ := real.pi * radius_mat^2

-- Calculate the area of the tabletop
def area_tabletop : ℝ := side_tabletop^2

-- Theorem statement: the fraction of the tabletop covered by the mat
theorem fraction_covered_by_mat : area_mat / area_tabletop = real.pi / 7 := sorry

end fraction_covered_by_mat_l798_798062


namespace value_of_p_l798_798891

def f (x : ℝ) (p : ℝ) : ℝ :=
  if x < 2 then 2^x + 1 else x^2 + p * x

theorem value_of_p (p : ℝ) (h : f (f 0 p) p = 5 * p) : p = 4 / 3 :=
by
  sorry

end value_of_p_l798_798891


namespace number_of_permutations_with_conditions_l798_798778

-- Define a permutation of the set {1, 2, ..., 10}
def isPermutation (pi : Fin 10 → Fin 10) : Prop :=
  ∀ i, ∃ j, pi j = i ∧ pi i ≠ i

-- Define the condition pi(pi(i)) = i for each i
def isSelfInverse (pi : Fin 10 → Fin 10) : Prop :=
  ∀ i, pi (pi i) = i

-- define the overall problem
theorem number_of_permutations_with_conditions :
  ∃ n, n = 945 ∧
    {pi : (Fin 10 → Fin 10) // isPermutation pi ∧ isSelfInverse pi}.toFinset.card = n := 
by
  sorry

end number_of_permutations_with_conditions_l798_798778


namespace tetrahedron_volume_l798_798139

   -- Definitions based on the conditions in the problem
   def angle_ABC_BCD : ℝ := π / 4  -- 45 degrees in radians
   def area_ABC : ℝ := 150
   def area_BCD : ℝ := 100
   def BC_length : ℝ := 12

   -- The volume of the tetrahedron
   def volume_tetrahedron (angle_ABC_BCD : ℝ) (area_ABC : ℝ) (area_BCD : ℝ) (BC_length : ℝ) : ℝ :=
     (1 / 3) * area_ABC * ((2 * area_BCD / BC_length) * (Real.sin angle_ABC_BCD) / 2)

   -- Statement of the problem
   theorem tetrahedron_volume : volume_tetrahedron angle_ABC_BCD area_ABC area_BCD BC_length = (1250 * Real.sqrt 2) / 3 :=
   by
     sorry
   
end tetrahedron_volume_l798_798139


namespace problem_solution_l798_798709

theorem problem_solution
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2007)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2007)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2007)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1003 := 
sorry

end problem_solution_l798_798709


namespace compute_expression_l798_798118

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l798_798118


namespace ratio_length_rect_radius_circle_l798_798689

theorem ratio_length_rect_radius_circle
  (A_sq : ℝ) (A_sq_eq : A_sq = 1225)
  (breadth : ℝ) (breadth_eq : breadth = 10)
  (A_rect : ℝ) (A_rect_eq : A_rect = 140) :
  let side_sq := Real.sqrt A_sq,
      radius := side_sq,
      length_rect := A_rect / breadth
  in length_rect / radius = (2:ℝ) / 5 :=
by
  -- The proof will be inserted here
  sorry

end ratio_length_rect_radius_circle_l798_798689


namespace color_of_85th_bead_l798_798703

-- Define the pattern sequence
inductive Color 
| red | orange | yellow | green | blue 

open Color

def pattern : List Color := [red, orange, yellow, yellow, green, green, blue]

-- Theorem statement
theorem color_of_85th_bead : (pattern.repeat (85 / pattern.length + 1)).get ⟨84, by decide⟩ = red :=
by
  sorry

end color_of_85th_bead_l798_798703


namespace crayons_remaining_l798_798960

def initial_crayons : ℕ := 87
def eaten_crayons : ℕ := 7

theorem crayons_remaining : (initial_crayons - eaten_crayons) = 80 := by
  sorry

end crayons_remaining_l798_798960


namespace range_of_a_l798_798786

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) → a > 2 :=
by
  sorry

end range_of_a_l798_798786


namespace arithmetic_sequence_formula_geometric_sequence_sum_l798_798503

variables {a_n S_n b_n T_n : ℕ → ℚ} {a_3 S_3 a_5 b_3 T_3 : ℚ} {q : ℚ}

def is_arithmetic_sequence (a_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, a_n n = a_1 + (n - 1) * d

def sum_first_n_arithmetic (S_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

def is_geometric_sequence (b_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, b_n n = b_1 * q^(n-1)

def sum_first_n_geometric (T_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, T_n n = if q = 1 then n * b_1 else b_1 * (1 - q^n) / (1 - q)

theorem arithmetic_sequence_formula {a_1 d : ℚ} (h_arith : is_arithmetic_sequence a_n a_1 d)
    (h_sum : sum_first_n_arithmetic S_n a_1 d) (h1 : a_n 3 = 5) (h2 : S_n 3 = 9) :
    ∀ n, a_n n = 2 * n - 1 := sorry

theorem geometric_sequence_sum {b_1 : ℚ} (h_geom : is_geometric_sequence b_n b_1 q)
    (h_sum : sum_first_n_geometric T_n b_1 q) (h3 : q > 0) (h4 : b_n 3 = a_n 5) (h5 : T_n 3 = 13) :
    ∀ n, T_n n = (3^n - 1) / 2 := sorry

end arithmetic_sequence_formula_geometric_sequence_sum_l798_798503


namespace johns_out_of_pocket_expense_l798_798963

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l798_798963


namespace distance_PM_is_correct_l798_798590

noncomputable def distance_PM (PQ QR PR QM : ℝ) : ℝ :=
  if h : PQ = 50 ∧ QR = 50 ∧ PR = 48 ∧ QM = 1/3 * QR then
    let QN := QR / 2 in
    let PN := Real.sqrt (PQ^2 - QN^2) in
    let NM := QN - QM in
    let PM := Real.sqrt (PN^2 + NM^2) in
    PM
  else 
    0

theorem distance_PM_is_correct :
  distance_PM 50 50 48 (1/3 * 50) = 20 * Real.sqrt 5 / 3 :=
by 
  sorry

end distance_PM_is_correct_l798_798590


namespace acute_angle_sum_l798_798513

theorem acute_angle_sum (α β : ℝ) (hαβ : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2) 
  (h : tan α + tan β = sqrt 3 - sqrt 3 * tan α * tan β) : 
  α + β = π / 3 :=
sorry

end acute_angle_sum_l798_798513


namespace ellipse_standard_equation_l798_798435

-- Define the conditions
def hasFociOnXAxis (ellipse : Type) : Prop := sorry
def majorAxisLength (ellipse : Type) : ℝ := sorry
def eccentricity (ellipse : Type) : ℝ := sorry

-- Define the specific ellipse
structure Ellipse where
  fociOnXAxis : hasFociOnXAxis Ellipse
  majorAxis : majorAxisLength Ellipse = 4
  ecc : eccentricity Ellipse = sqrt 3 / 2

-- Prove that the standard equation of the ellipse
theorem ellipse_standard_equation (e : Ellipse) : 
  (∃ a b : ℝ, a > b ∧ a = 2 ∧ b = 1 ∧ (a^2 - b^2 = 3) ∧ (eccentricity Ellipse = sqrt 3 / 2)) 
  → (∀ x y : ℝ, (x^2 / 4 + y^2 = 1)) :=
begin
  sorry
end

end ellipse_standard_equation_l798_798435


namespace assistant_increases_output_by_100_percent_l798_798746

variable (B H : ℝ)
variable (h₁ : B > 0) (h₂ : H > 0)

def output_increase_percent : ℝ :=
  100 * ((1.8 * B) / (0.9 * H) - B / H) / (B / H)

theorem assistant_increases_output_by_100_percent :
  output_increase_percent B H = 100 :=
by
  sorry

end assistant_increases_output_by_100_percent_l798_798746


namespace simplify_fraction_l798_798658

open Real

theorem simplify_fraction :
    (3 * sqrt 50 = 15 * sqrt 2) →
    (sqrt 18 = 3 * sqrt 2) →
    (4 * sqrt 8 = 8 * sqrt 2) →
    (5 / (3 * sqrt 50 + sqrt 18 + 4 * sqrt 8) = 5 * sqrt 2 / 52) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    5 / (15 * sqrt 2 + 3 * sqrt 2 + 8 * sqrt 2)
        = 5 / (26 * sqrt 2) : by ring
    ... = 5 / 26 / sqrt 2 : by rw div_mul_eq_div_div
    ... = 5 * sqrt 2 / (26 * sqrt 2 * sqrt 2) : by rw ← mul_div_right_comm
    ... = 5 * sqrt 2 / 52 : by rw [sqrt_mul_self, mul_one] ; sorry

end simplify_fraction_l798_798658


namespace remaining_garden_space_l798_798065

theorem remaining_garden_space : 
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  Area_rectangle - Area_square_cutout + Area_triangle = 347 :=
by
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  show Area_rectangle - Area_square_cutout + Area_triangle = 347
  sorry

end remaining_garden_space_l798_798065


namespace calculate_fraction_l798_798812

theorem calculate_fraction : (10^20 / 50^10) = 2^10 := by
  sorry

end calculate_fraction_l798_798812


namespace y_intercept_after_translation_l798_798692

-- Define the original line equation and the concept of translation
def original_line := λ x : ℝ, x + 4
def translate_down (line : ℝ → ℝ) (units : ℝ) := λ x, line x - units

-- Define the key properties used
def original_y_intercept := original_line 0
def translated_line := translate_down original_line 6
def translated_y_intercept := translated_line 0

-- Define the theorem statement
theorem y_intercept_after_translation : translated_y_intercept = -2 := by
  sorry

end y_intercept_after_translation_l798_798692


namespace exists_ordering_l798_798621

/-- Let $\mathcal{S}$ be a set consisting of $n \geq 3$ positive integers, 
none of which is a sum of two other distinct members of $\mathcal{S}$.
Prove that the elements of $\mathcal{S}$ may be ordered as $a_{1}, a_{2}, \ldots, a_{n}$ 
so that $a_{i}$ does not divide $a_{i-1} + a_{i+1}$ for all $i=2, 3, \ldots, n-1$. --/
theorem exists_ordering (S : Finset ℕ) (n : ℕ) (hS1 : n ≥ 3) (hS2 : S.card = n) 
  (hS3 : ∀ a ∈ S, ∀ b ∈ S, ∀ c ∈ S, a ≠ b → a ≠ c → b ≠ c → a ≠ b + c) :
  ∃ (a : Fin n → ℕ), (∀ i : Fin (n - 2), a (i + 1) ∣ a i + a (i + 2) → False) := 
sorry

end exists_ordering_l798_798621


namespace route_b_quicker_than_route_a_l798_798307

noncomputable def route_a_time : ℝ := (8 / 35) * 60
noncomputable def route_b_time : ℝ := ((5.5 / 45) + (1 / 25) + (0.5 / 15)) * 60

theorem route_b_quicker_than_route_a :
  route_a_time - route_b_time = 1.98 :=
by {
  unfold route_a_time route_b_time,
  -- calculate times
  have ta : (8 / 35) * 60 = 13.71,
  have tb1 : (5.5 / 45) * 60 = 7.33,
  have tb2 : (1 / 25) * 60 = 2.4,
  have tb3 : (0.5 / 15) * 60 = 2,
  have tb : 7.33 + 2.4 + 2 = 11.73,
  -- compute difference
  have diff : 13.71 - 11.73 = 1.98,
  exact diff,
  sorry
}

end route_b_quicker_than_route_a_l798_798307


namespace combination_equivalence_l798_798511

theorem combination_equivalence (n : ℕ) (h₁ : 2 * (n.choose 5) = (n.choose 4) + (n.choose 6)) :
  (14.choose 10 = 1001) :=
by {
  have h₂ : 14 = n := by {
    sorry -- Steps to prove that n = 14 from the given h₁
  },
  rw h₂,
  have h₃ : 14.choose 4 = 1001 := by {
    sorry -- Steps to prove that 14 choose 4 = 1001
  },
  exact h₃
}

end combination_equivalence_l798_798511


namespace determine_mass_l798_798024

noncomputable def mass_of_water 
  (P : ℝ) (t1 t2 : ℝ) (deltaT : ℝ) (cw : ℝ) : ℝ :=
  P * t1 / ((cw * deltaT) + ((cw * deltaT) / t2) * t1)

theorem determine_mass (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cw : ℝ) :
  P = 1000 → t1 = 120 → deltaT = 2 → t2 = 60 → cw = 4200 →
  mass_of_water P t1 deltaT t2 cw = 4.76 :=
by
  intros hP ht1 hdeltaT ht2 hcw
  sorry

end determine_mass_l798_798024


namespace truck_capacity_l798_798330

theorem truck_capacity
  (x y : ℝ)
  (h1 : 2 * x + 3 * y = 15.5)
  (h2 : 5 * x + 6 * y = 35) :
  3 * x + 5 * y = 24.5 :=
sorry

end truck_capacity_l798_798330


namespace max_marks_mike_l798_798632

theorem max_marks_mike (pass_percentage : ℝ) (scored_marks : ℝ) (shortfall : ℝ) : 
  pass_percentage = 0.30 → 
  scored_marks = 212 → 
  shortfall = 28 → 
  (scored_marks + shortfall) = 240 → 
  (scored_marks + shortfall) = pass_percentage * (max_marks : ℝ) → 
  max_marks = 800 := 
by 
  intros hp hs hsh hps heq 
  sorry

end max_marks_mike_l798_798632


namespace union_cardinality_l798_798873

-- Definitions for the given sets A and B
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

-- Problem statement: Prove the number of elements in the union of A and B is 6
theorem union_cardinality : (A ∪ B).card = 6 := by
  sorry

end union_cardinality_l798_798873


namespace number_of_ordered_pairs_mod_1000_l798_798280

theorem number_of_ordered_pairs_mod_1000 :
  let S := (Finset.range 101).sum (λ a, a) in
  S % 1000 = 50 :=
by
  let S := (Finset.range 101).sum (λ a, a)
  sorry

end number_of_ordered_pairs_mod_1000_l798_798280


namespace zero_within_interval_l798_798827

theorem zero_within_interval (a : ℝ) :
  (a < -4 → ∃ x ∈ Icc (-1 : ℝ) (1 : ℝ), a * x + 3 = 0) ∧
  (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), a * x + 3 = 0 → a < -4) → False :=
by
  sorry

end zero_within_interval_l798_798827


namespace distance_from_B_l798_798083

theorem distance_from_B (s y : ℝ) 
  (h1 : s^2 = 12)
  (h2 : ∀y, (1 / 2) * y^2 = 12 - y^2)
  (h3 : y = 2 * Real.sqrt 2)
: Real.sqrt ((2 * Real.sqrt 2)^2 + (2 * Real.sqrt 2)^2) = 4 := by
  sorry

end distance_from_B_l798_798083


namespace incorrect_statements_count_l798_798795

theorem incorrect_statements_count (p q : Prop) (x : ℝ) :
  (¬((p ∨ q) → (p ∧ q)) ∧ ¬("if x² - 3x + 2 = 0, then x = 1 or x = 2" ↔ "if x ≠ 1 ∨ x ≠ 2, then x² - 3x + 2 ≠ 0")) ∧
  (("x > 5" → "x² - 4x - 5 > 0") ∧ ∀ x, (¬(x² + x - 1 < 0) ↔ x² + x - 1 ≥ 0)) →
  num_incorrect_statements = 2 :=
by
  sorry

end incorrect_statements_count_l798_798795


namespace find_polynomials_l798_798145

noncomputable def polynomial_solution (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ) (n : ℕ) (h : n ≠ 0),
  p = λ x, a * (x + b)^n

theorem find_polynomials (p : ℝ → ℝ) (c : ℝ) :
  polynomial p ∧ ∀ x, (p' x)^2 = c * p x * p'' x → polynomial_solution p :=
sorry

end find_polynomials_l798_798145


namespace quadrilateral_AP_eq_CQ_l798_798324

-- Definitions of points and other geometric entities
variables (A B C D P Q H O : Type)
variables [Geometry A B C D P Q H O]

-- Conditions
def inscribed_in_circle (ABCD_in_Gamma : ∀{Γ}, inscribed A B C D Γ) (center_O : center Γ = O) : Prop := 
  true

def diagonals_perpendicular (perp_AC_BD : perpendicular AC BD) : Prop := 
  true

def O_in_triangle_BPC (O_in_BPC : ∀{triangle}, in_triangle B P C O) : Prop := 
  true

def point_H_on_BO (H_on_BO : H ∈ segment B O) (angle_BHP_90 : angle B H P = 90) : Prop := 
  true

def circumcircle_intersection (omega : circle) (intersect_second_time : second_intersection omega P H D Q) : Prop := 
  true

-- The final proof assertion
theorem quadrilateral_AP_eq_CQ
  (ABCD_in_Gamma : inscribed_in_circle ABCD_in_Gamma center_O)
  (perp_AC_BD : diagonals_perpendicular perp_AC_BD)
  (O_in_BPC : O_in_triangle_BPC O_in_BPC)
  (point_H_on_BO : point_H_on_BO H_on_BO angle_BHP_90)
  (circumcircle_intersection : circumcircle_intersection omega intersect_second_time)
  : 
  length A P = length C Q := 
  sorry

end quadrilateral_AP_eq_CQ_l798_798324


namespace ratio_of_a_and_b_l798_798290

theorem ratio_of_a_and_b
  (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : ∃ c : ℝ, (3 - (4 : ℂ) * complex.I) * (a + b * complex.I) = c * complex.I) :
  a / b = -4 / 3 :=
by
  sorry

end ratio_of_a_and_b_l798_798290


namespace sum_of_perimeters_l798_798026

theorem sum_of_perimeters :
  let areas := list.range 25 |>.map (λ n, (2 * n + 1)^2)
  let perimeters := areas.map (λ a, 4 * (Int.sqrt a))
  list.sum perimeters = 2500 := 
by
  let areas := list.range 25 |>.map (λ n, (2 * n + 1)^2)
  let perimeters := areas.map (λ a, 4 * (Int.sqrt a))
  have : areas = list.range 25 |>.map (λ n, (2 * n + 1)^2), from rfl
  have : perimeters = areas.map (λ a, 4 * (Int.sqrt a)), from rfl
  sorry

end sum_of_perimeters_l798_798026


namespace find_b_l798_798346

theorem find_b (a b x1 x2 : ℝ) (h1 : x1 + x2 = -12 * a)
                            (h2 : x1 * x2 = 12 * b)
                            (h3 : (3 - x1)^2 + 9 = (3 - x2)^2 + 9) :
  b = -6 :=
begin
  sorry
end

end find_b_l798_798346


namespace integer_2020_column_l798_798091

def column_of_integer (n : ℕ) : string :=
  let adjusted_n := n - 10
  let position := adjusted_n % 14
  ["A", "B", "C", "D", "E", "F", "G", "G", "F", "E", "D", "C", "B", "A"].get! (position)

theorem integer_2020_column : column_of_integer 2020 = "F" :=
by
  sorry

end integer_2020_column_l798_798091


namespace symmetric_point_exists_l798_798477

-- Define the point M
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point M
def M : Point3D := { x := 3, y := 3, z := 3 }

-- Define the parametric form of the line
def line (t : ℝ) : Point3D := { x := 1 - t, y := 1.5, z := 3 + t }

-- Define the point M' that we want to prove is symmetrical to M with respect to the line
def symmPoint : Point3D := { x := 1, y := 0, z := 1 }

-- The theorem that we need to prove, ensuring M' is symmetrical to M with respect to the given line
theorem symmetric_point_exists : ∃ t, line t = symmPoint ∧ 
  (∀ M_0 : Point3D, M_0.x = (M.x + symmPoint.x) / 2 ∧ M_0.y = (M.y + symmPoint.y) / 2 ∧ M_0.z = (M.z + symmPoint.z) / 2)
  → line t = M_0
  → M_0 = { x := 2, y := 1.5, z := 2 } := 
by
  sorry

end symmetric_point_exists_l798_798477


namespace divisors_572_divisors_572_a3bc_case1_divisors_572_31_32_33_l798_798750

noncomputable def prime_factors_572 := 2^2 * 11 * 13

theorem divisors_572 : 
  (card {d : ℕ | ∃ k l m, d = (2^k) * (11^l) * (13^m) ∧ (k = 0 ∨ k = 1 ∨ k = 2) ∧ (l = 0 ∨ l = 1) ∧ (m = 0 ∨ m = 1)}) = 12 := 
begin
  sorry
end

theorem divisors_572_a3bc_case1 (a b c : ℕ) (ha : nat.prime a) (hb : nat.prime b) (h208 : 20 < a) (h209 : 20 < b) (hab : a ≠ b) (hc : nat.prime c) (h20 : 20 < c) (hac : a ≠ c) (hbc : b ≠ c) :
  (card {d : ℕ | ∃ k l m n o p, d = (2^k) * (11^l) * (13^m) * (a^n) * (b^o) * (c^p) ∧ (k = 0 ∨ k = 1 ∨ k = 2) ∧ (l = 0 ∨ l = 1) ∧ (m = 0 ∨ m = 1) ∧ (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3) ∧ (o = 0 ∨ o = 1) ∧ (p = 0 ∨ p = 1)}) = 192 :=
begin
  sorry
end

theorem divisors_572_31_32_33 :
  (card {d : ℕ | ∃ k l m n o p, d = (2^k) * (3^l) * (11^m) * (13^n) * (31^o) * (32^2 ∧ k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7) ∧ (l = 0 ∨ l = 1) ∧ (m = 0 ∨ m = 1 ∨ m = 2) ∧ (n = 0 ∨ n = 1) ∧ (o = 0 ∨ o = 1 ∨ o = 2 ∨ o = 3) } ) = 384 := 
begin
  sorry
end

end divisors_572_divisors_572_a3bc_case1_divisors_572_31_32_33_l798_798750


namespace sum_abc_l798_798300

noncomputable def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

theorem sum_abc (a b c : ℕ) (h1 : f a b c 2 = 7) (h2 : f a b c 0 = 6) (h3 : f a b c (-1) = 8) :
  a + b + c = 10 :=
by {
  sorry
}

end sum_abc_l798_798300


namespace complement_intersection_l798_798875

open Set Function

noncomputable def A : Set ℝ := { x | x^2 - 3 * x > 4 }
noncomputable def B : Set ℝ := { x | 2^x > 2 }
noncomputable def A_complement : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
noncomputable def expected_intersection : Set ℝ := { x | 1 < x ∧ x ≤ 4 }

theorem complement_intersection : (A_complement ∩ B) = expected_intersection := by
  sorry

end complement_intersection_l798_798875


namespace triangular_prism_cross_section_l798_798676

theorem triangular_prism_cross_section (P : Prism) (plane : Plane) (e : Edge) 
  (h1 : plane ∩ e ≠ ∅) (h2 : ∃ p1 p2, p1 ∈ lateral_edges P ∧ p2 ∈ lateral_edges P ∧ plane ∩ p1 ≠ ∅ ∧ plane ∩ p2 ≠ ∅) 
  (h3 : ∀ p, p ∈ top_base P ∨ p ∈ bottom_base P ∧ plane ∩ p ≠ ∅) :
  cross_section P plane = Shape.Triangle ∨ cross_section P plane = Shape.Trapezoid := 
by 
  sorry

end triangular_prism_cross_section_l798_798676


namespace inequality_solution_range_l798_798897

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x+2| + |x-3| < a) ↔ a > 5 :=
by
  sorry

end inequality_solution_range_l798_798897


namespace sqrt_ratio_simplify_l798_798048

theorem sqrt_ratio_simplify :
  ( (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 ) :=
by
  let sqrt27 := Real.sqrt 27
  let sqrt243 := Real.sqrt 243
  let sqrt75 := Real.sqrt 75
  have h_sqrt27 : sqrt27 = Real.sqrt (3^2 * 3) := by sorry
  have h_sqrt243 : sqrt243 = Real.sqrt (3^5) := by sorry
  have h_sqrt75 : sqrt75 = Real.sqrt (3 * 5^2) := by sorry
  have h_simplified : (sqrt27 + sqrt243) / sqrt75 = 12 / 5 := by sorry
  exact h_simplified

end sqrt_ratio_simplify_l798_798048


namespace comparison_b_a_c_l798_798492

noncomputable def a : ℝ := Real.sqrt 1.2
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := 1 + Real.log 1.1

theorem comparison_b_a_c : b > a ∧ a > c :=
by
  unfold a b c
  sorry

end comparison_b_a_c_l798_798492


namespace problem_1_problem_2_l798_798220

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom norm_2a_minus_b : ∥2 • a - b∥ = Real.sqrt 5

-- Problem 1: Prove |2a - 3b| = sqrt(13)
theorem problem_1 : ∥2 • a - 3 • b∥ = Real.sqrt 13 :=
by sorry

-- Problem 2: Prove the angle θ between 3a - b and a - 2b is π/4
theorem problem_2 : 
  let v1 := 3 • a - b
  let v2 := a - 2 • b
  in Real.arccos ((v1 ⬝ v2) / (∥v1∥ * ∥v2∥)) = Real.pi / 4 :=
by sorry

end problem_1_problem_2_l798_798220


namespace min_people_to_remove_for_square_formation_l798_798759

theorem min_people_to_remove_for_square_formation (n : ℕ) (h : n = 73) : ∃ k, k = 9 ∧ n - k = 64 := by
  use 9
  split
  · rfl
  · rw [h]
    rfl

end min_people_to_remove_for_square_formation_l798_798759


namespace initial_number_of_machines_l798_798328

theorem initial_number_of_machines
  (x : ℕ)
  (h1 : x * 270 = 1080)
  (h2 : 20 * 3600 = 144000)
  (h3 : ∀ y, (20 * y * 4 = 3600) → y = 45) :
  x = 6 :=
by
  sorry

end initial_number_of_machines_l798_798328


namespace ellipse_problem_l798_798798

noncomputable theory

-- Define the given conditions
def A := (2, 3)  -- Point A(2, 3)
def foci_x_axis (F1 F2 : ℝ × ℝ) := F1.1 ≠ 0 ∧ F2.1 ≠ 0 ∧ F1.2 = 0 ∧ F2.2 = 0
def symmetry_axes (E : ℝ → ℝ → Prop) : Prop := 
  ∀ x y, E x y ↔ E (-x) y ∧ E x (-y)
def eccentricity (e : ℝ) : Prop := e = 1 / 2

-- Define the main goals to prove
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, E x y ↔ (x^2 / 16 + y^2 / 12 = 1)

def angle_bisector (bisector : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, bisector x y ↔ (2 * x - y - 1 = 0)

-- Main theorem combining all the parts
theorem ellipse_problem 
  (E : ℝ → ℝ → Prop)
  (F1 F2 : ℝ × ℝ)
  (bisector : ℝ → ℝ → Prop) :
  (E 2 3) ∧ 
  foci_x_axis F1 F2 ∧ 
  symmetry_axes E ∧ 
  eccentricity 0.5 ∧ 
  ellipse_equation E ∧ 
  angle_bisector bisector :=
sorry

end ellipse_problem_l798_798798


namespace n_minus_two_is_square_of_natural_number_l798_798002

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l798_798002


namespace find_k_l798_798899

-- Definitions of vectors a and b as given in conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Perpendicularity condition
def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Proof statement
theorem find_k (k : ℝ) : perp (k * a.1, k * a.2) (b.1, b.2) ∧ (a.1 - 3 * b.1, a.2 - 3 * b.2) →
  k = 19 :=
by
sorrrrrrrrrrrrrrπ


end find_k_l798_798899


namespace part_one_solution_part_two_solution_l798_798756

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x - Real.pi / 4)

theorem part_one_solution :
  f x = 2 * Real.sin (2 * x - Real.pi / 4) := by
  sorry

def g (x m : ℝ) : ℝ :=
  f x + m

theorem part_two_solution (m : ℝ) (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4)) :
  (∀ y, y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → -1 ≤ f y ∧ f y ≤ Real.sqrt 2 / 2) →
  -2 + m = 3 ∧ Real.sqrt 2 + m = 5 + Real.sqrt 2 :=
by
  sorry

end part_one_solution_part_two_solution_l798_798756


namespace union_sets_intersection_complement_l798_798902

open Set

noncomputable def U := (univ : Set ℝ)
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 5 }

theorem union_sets : A ∪ B = univ := by
  sorry

theorem intersection_complement : (U \ A) ∩ B = { x : ℝ | x < 2 } := by
  sorry

end union_sets_intersection_complement_l798_798902


namespace minimum_value_of_expression_l798_798644

theorem minimum_value_of_expression : 
  ∃ (a b : ℕ), a ∈ {2, 3, 5, 7} ∧ b ∈ {2, 4, 6, 8} ∧ (a + b) % 2 = 0 ∧ 
  (∀ (a' b' : ℕ), a' ∈ {2, 3, 5, 7} ∧ b' ∈ {2, 4, 6, 8} ∧ (a' + b') % 2 = 0 → 
  (2 * a - a * b) ≤ (2 * a' - a' * b')) ∧ 
  (2 * a - a * b) = -12 := 
by
  sorry

end minimum_value_of_expression_l798_798644


namespace percentage_of_games_won_l798_798133

theorem percentage_of_games_won (G W : ℕ) (P : ℝ)
  (h1 : W + 0.50 * (G - 100) = 0.70 * G)
  (h2 : P = (W / 100) * 100)
  (h3 : W = 70) :
  P = 70 := 
sorry

end percentage_of_games_won_l798_798133


namespace quadratic_inequality_solution_l798_798354

theorem quadratic_inequality_solution (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
sorry

end quadratic_inequality_solution_l798_798354


namespace pies_calculation_l798_798327

-- Definition: Number of ingredients per pie
def ingredients_per_pie (apples total_apples pies : ℤ) : ℤ := total_apples / pies

-- Definition: Number of pies that can be made with available ingredients 
def pies_from_ingredients (ingredient_amount per_pie : ℤ) : ℤ := ingredient_amount / per_pie

-- Hypothesis
theorem pies_calculation (apples_per_pie pears_per_pie apples pears pies : ℤ) 
  (h1: ingredients_per_pie apples 12 pies = 4)
  (h2: ingredients_per_pie apples 6 pies = 2)
  (h3: pies_from_ingredients 36 4 = 9)
  (h4: pies_from_ingredients 18 2 = 9): 
  pies = 9 := 
sorry

end pies_calculation_l798_798327


namespace prob_next_black_ball_l798_798005

theorem prob_next_black_ball
  (total_balls : ℕ := 100) 
  (black_balls : Fin 101) 
  (next_black_ball_probability : ℚ := 2 / 3) :
  black_balls.val ≤ total_balls →
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (p : ℚ) / q = next_black_ball_probability ∧ p + q = 5 :=
by
  intros h
  use 2, 3
  repeat { sorry }

end prob_next_black_ball_l798_798005


namespace T_bounds_l798_798185

noncomputable def S : ℕ → ℝ
| n := (3/2)^(n + 1)

def b (n : ℕ) : ℝ := (-1)^n / S n

def T (n : ℕ) : ℝ := (finset.range n).sum (λ k, b (k+1))

theorem T_bounds (n : ℕ) (hn : 2 ≤ n) : 
  (1 / 3) ≤ | T n | ∧ | T n | ≤ (7 / 9) :=
begin
  sorry
end

end T_bounds_l798_798185


namespace lg_eq_has_one_solution_iff_l798_798484

variable {a x : ℝ}

theorem lg_eq_has_one_solution_iff :
  (∃! x, x > 0 ∧ lg (a * x) = 2 * lg (x + 1)) ↔ a = 4 ∨ a < 0 := by
  sorry

end lg_eq_has_one_solution_iff_l798_798484


namespace proper_subsets_A_inter_B_l798_798625

/-- 
  Let A be the set {x ∈ ℤ | x² < 9}
  and B be the set {x ∈ ℤ | 2x > a}.
  1. If a = 1, the proper subsets of A ∩ B are ∅, {1}, {2}.
  2. If A ∩ B has 4 subsets, the range of values for a is [0, 2).
-/
theorem proper_subsets_A_inter_B (a : ℤ) (S : set ℤ) (A := {x | x^2 < 9})
  (B := {x | 2 * x > a}) :
  (a = 1 → (S = {x | x = 1} ∪ {x | x = 2} → (S ∈ {∅, {1}, {2}}))) ∧
  (∀ S, S ≠ ∅ → S ≠ ({x | x = 1} ∪ {x | x = 2}) → (S = {1} ∨ S = {2})) ∧
  ((card (S ∩ (({x | x^2 < 9}) ∩ ({x | 2 * x > a}))) = 2) → (0 ≤ a ∧ a < 2)) :=
begin
  sorry
end

end proper_subsets_A_inter_B_l798_798625


namespace ratio_IK_KJ_equals_one_l798_798507

theorem ratio_IK_KJ_equals_one
  {ABC : Type} -- Assume arbitrary type for the triangle
  {I J B C : ABC} -- Assume points I, J, B, C on triangle ABC
  {O_b O_c K : ABC} -- Assume points for centers of circles and intersection point
  (hI_center : is_incenter I ABC) -- Incenter I
  (hJ_excenter : is_excenter J A ABC) -- Excenter J with respect to vertex A
  (hO_b_center : circle_centered_at O_b ∧ passes_through B ∧ tangent_to_line_at I CI) -- Circle ω_b
  (hO_c_center : circle_centered_at O_c ∧ passes_through C ∧ tangent_to_line_at I BI) -- Circle Ω_c
  (hK_intersection : intersects O_b O_c IJ K) -- Intersection of O_bO_c and IJ at K

  : IK / KJ = 1 := 
begin
  sorry -- Proof is omitted as the focus is on formulating the Lean statement
end

end ratio_IK_KJ_equals_one_l798_798507


namespace fraction_invariant_l798_798242

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l798_798242


namespace repeating_decimal_division_l798_798466

def repeating_decimal_081_as_fraction : ℚ := 9 / 11
def repeating_decimal_272_as_fraction : ℚ := 30 / 11

theorem repeating_decimal_division : 
  (repeating_decimal_081_as_fraction / repeating_decimal_272_as_fraction) = (3 / 10) := 
by 
  sorry

end repeating_decimal_division_l798_798466


namespace more_I_than_P_l798_798158

-- Definition of S(n)
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Definition of property P
def property_P (n : ℕ) : Prop :=
  ∀ m, ∃ k, (∃ l, m = n + l * k ∧ l > 0) ∧ even k ∧ even (S m)

-- Definition of property I
def property_I (n : ℕ) : Prop :=
  ∀ m, ∃ k, (∃ l, m = n + l * k ∧ l > 0) ∧ odd k ∧ odd (S m)

-- Proof statement
theorem more_I_than_P : (∑ n in Finset.range 2017, if property_I n then 1 else 0) > (∑ n in Finset.range 2017, if property_P n then 1 else 0) :=
sorry

end more_I_than_P_l798_798158


namespace rice_amount_previously_l798_798352

variables (P X : ℝ) (hP : P > 0) (h : 0.8 * P * 50 = P * X)

theorem rice_amount_previously (hP : P > 0) (h : 0.8 * P * 50 = P * X) : X = 40 := 
by 
  sorry

end rice_amount_previously_l798_798352


namespace num_valid_placements_l798_798856

open Finset

def balls := {1, 2, 3, 4}
def boxes := {1, 2, 3}

-- Define the conditions as predicates 
def valid_arrangement (f : 1..3 -> 1..4) : Prop :=
  f 1 ≠ 1 ∧ f 3 ≠ 3

-- Count the valid arrangements
def valid_placements_count : nat :=
  (boxes.powerset.filter (fun s => s.card = 3)).card 

theorem num_valid_placements :
  valid_placements_count = 14 
:= 
sorry

end num_valid_placements_l798_798856


namespace max_tickets_l798_798481


theorem max_tickets (cost_regular : ℕ) (cost_discounted : ℕ) (threshold : ℕ) (total_money : ℕ) 
  (h1 : cost_regular = 15) 
  (h2 : cost_discounted = 12) 
  (h3 : threshold = 5)
  (h4 : total_money = 150) 
  : (total_money / cost_regular ≤ 10) ∧ 
    ((total_money - threshold * cost_regular) / cost_discounted + threshold = 11) :=
by
  sorry

end max_tickets_l798_798481


namespace rate_of_profit_is_40_l798_798380

def cost_price : ℝ := 50
def selling_price : ℝ := 70

def profit : ℝ := selling_price - cost_price

def rate_of_profit : ℝ := (profit / cost_price) * 100

theorem rate_of_profit_is_40 : rate_of_profit = 40 := by
  sorry

end rate_of_profit_is_40_l798_798380


namespace upper_bound_of_s_l798_798265

variable {X Y Z P : Type}
variable {x y z : ℝ}
variable {XX' YY' ZZ' : ℝ}
variable {s : ℝ}

-- Define the conditions of the problem
def triangle_condition (x y z : ℝ) := x ≤ y ∧ y ≤ z
def s_condition (XX' YY' ZZ' : ℝ) := s = XX' + YY' + ZZ'

-- The final statement we want to prove
theorem upper_bound_of_s (h_triangle : triangle_condition x y z)
  (h_s : s_condition XX' YY' ZZ') :
  s ≤ x + y + z :=
sorry

end upper_bound_of_s_l798_798265


namespace subject_combinations_l798_798942

theorem subject_combinations : 
  let science := ["Physics", "Chemistry", "Biology"]
  let humanities := ["Politics", "History", "Geography"]
  ∃ S : set (list String), S = {l | ∀ x ∈ l, x ∈ science ∪ humanities 
    ∧ length l = 3 ∧ list.countp (λ x, x ∈ science) l ≤ 2 ∧ list.countp (λ x, x ∈ {"Biology", "Politics", "History"}) l ≤ 1 }
  ∧ S.card = 10 := by 
      sorry

end subject_combinations_l798_798942


namespace rate_of_painting_per_sq_m_l798_798348

def length_of_floor : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def ratio_of_length_to_breadth : ℝ := 3

theorem rate_of_painting_per_sq_m :
  ∃ (rate : ℝ), rate = 3 :=
by
  let B := length_of_floor / ratio_of_length_to_breadth
  let A := length_of_floor * B
  let rate := total_cost / A
  use rate
  sorry  -- Skipping proof as instructed

end rate_of_painting_per_sq_m_l798_798348


namespace Ms_Hu_possibly_from_Shanghai_l798_798463

-- Define types for cities and judgments
inductive City : Type
| Nanchang
| Fuzhou
| Shanghai
| Guangzhou

open City

-- Define the judgments of each student using functions
def student_A_judgment (c : City) : Prop :=
  (c ≠ Shanghai) ∧ (c = Fuzhou)

def student_B_judgment (c : City) : Prop :=
  (c ≠ Fuzhou) ∧ (c = Nanchang)

def student_C_judgment (c : City) : Prop :=
  (c ≠ Fuzhou) ∧ (c ≠ Guangzhou)

-- Define Ms. Hu's statement about the students' correctness
def Ms_Hu_statement (A_correct A_half_correct B_correct B_half_correct C_correct C_half_correct : Prop) : Prop :=
  (A_correct ∨ A_half_correct) ∧ (B_correct ∨ B_half_correct) ∧ (C_correct ∨ C_half_correct) ∧
  ((A_correct ∧ B_half_correct ∧ C_correct) ∨
   (A_correct ∧ B_correct ∧ ¬C_correct) ∨
   (B_correct ∧ ¬A_correct ∧ C_half_correct) ∨
   (¬B_correct ∧ ¬A_correct ∧ ¬C_correct))

-- Define the final proof problem
theorem Ms_Hu_possibly_from_Shanghai :
  ∃ c : City, c = Shanghai ∧ Ms_Hu_statement
   (student_A_judgment c) -- A's judgement correctness
   (¬student_A_judgment c) -- A's judgement incorrectness
   (student_B_judgment c) -- B's judgement correctness
   (¬student_B_judgment c) -- B's judgement incorrectness
   (student_C_judgment c) -- C's judgement correctness
   (¬student_C_judgment c) -- C's judgement incorrectness :=
sorry

end Ms_Hu_possibly_from_Shanghai_l798_798463


namespace right_triangle_probability_l798_798032

def regular_octagon := {vertices : Finset ℝ | ∃ r : ℝ, vertices.card = 8 ∧ ∀ v ∈ vertices, ∃ θ : ℝ, v = (r * cos θ, r * sin θ) ∧ θ ∈ (2 * π) / 8 • Finset.range 8}

def count_triangles (vertices : Finset ℝ) := (vertices.choose 3).card

def count_right_triangles (vertices : Finset ℝ) : ℕ :=
  8 * 4 / 2 -- as explained in the solution, 8 vertices and choosing the non-adjacent ones leading to right triangles

theorem right_triangle_probability (vertices : Finset ℝ) (h : vertices ∈ regular_octagon) :
  (count_right_triangles vertices : ℚ) / (count_triangles vertices) = 2 / 7 :=
by
  sorry

end right_triangle_probability_l798_798032


namespace cost_of_three_stamps_is_correct_l798_798639

-- Define the cost of one stamp
def cost_of_one_stamp : ℝ := 0.34

-- Define the number of stamps
def number_of_stamps : ℕ := 3

-- Define the expected total cost for three stamps
def expected_cost : ℝ := 1.02

-- Prove that the cost of three stamps is equal to the expected cost
theorem cost_of_three_stamps_is_correct : cost_of_one_stamp * number_of_stamps = expected_cost :=
by
  sorry

end cost_of_three_stamps_is_correct_l798_798639


namespace largest_among_one_neg2_zero_sqrt3_l798_798434

theorem largest_among_one_neg2_zero_sqrt3 : ∀ a ∈ ({1, -2, 0, Real.sqrt 3} : set ℝ), a ≤ Real.sqrt 3 :=
by {
  sorry,
}

end largest_among_one_neg2_zero_sqrt3_l798_798434


namespace probability_sum_greater_than_6_l798_798651

def set_S : Set ℤ := {12, 34}

theorem probability_sum_greater_than_6 :
  let pairs := { (x, y) | x ∈ set_S ∧ y ∈ set_S}.to_finset
  let favorable_pairs := pairs.filter (λ t, t.1 + t.2 > 6)
  (favorable_pairs.card : ℚ) / (pairs.card : ℚ) = 1 :=
by
  sorry

end probability_sum_greater_than_6_l798_798651


namespace fractionD_is_unchanged_l798_798244

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l798_798244


namespace geometric_seq_fourth_term_l798_798685

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end geometric_seq_fourth_term_l798_798685


namespace proof_problem_l798_798506
-- Import necessary libraries

-- Define the given conditions and constructs
variable (O X Y P A B C D M N E F G H : Type)
variable [chord O X Y] [chord O A B] [chord O C D]
variable [midpoint P XY] [midpoint M AB] [midpoint N CD]
variable [intersect E (line A N) (line X Y)]
variable [intersect F (line B N) (line X Y)]
variable [intersect G (line C M) (line X Y)]
variable [intersect H (line D M) (line X Y)]
variable (OP_perpendicular : ∀ (O P : Type), line O P ⊥ line X Y)
variable (OM_perpendicular : ∀ (O M : Type), line O M ⊥ line A B)
variable (ON_perpendicular : ∀ (O N : Type), line O N ⊥ line C D)

-- Define the theorem to prove
theorem proof_problem :
  ∀ {PM AB PF PE PN CD PH PG : ℝ},
  (PM / AB) * (1 / PF - 1 / PE) = (PN / CD) * (1 / PH - 1 / PG) :=
by sorry

end proof_problem_l798_798506


namespace find_lambda_find_min_area_l798_798489

-- Define the parabola and the points P, Q, R
def parabola (x : ℝ) : ℝ := x^2

-- Point P on the parabola
structure Point (x y : ℝ) : Prop :=
  (on_parabola : y = parabola x)

-- Define the tangent line at point P
def tangent_line (P : Point) : (ℝ × ℝ) → Prop :=
  λ (x, y), 2 * P.x * x = P.y + y

-- Defining points Q and R given point P
def Q (P : Point) : Point :=
  Point.mk (P.y / (2 * P.x)) 0

def R (P : Point) : Point :=
  Point.mk 0 (-P.y)

-- Vectors PQ and PR
def vector_PQ (P : Point) : (ℝ × ℝ) :=
  ((P.y / (2 * P.x)) - P.x, -P.y)

def vector_PR (P : Point) : (ℝ × ℝ) :=
  (-P.x, -2 * P.y)

-- Problem (Ⅰ)
theorem find_lambda (P : Point) (λ : ℝ) : vector_PQ P = λ • vector_PR P → λ = 1 / 2 := sorry

-- Defining point S on the parabola
structure Point_S (x y : ℝ) : Prop :=
  (on_parabola : y = parabola x)
  (different_from_P : y ≠ 1)

-- Problem (Ⅱ)
theorem find_min_area (S P : Point) (λ : ℝ) (h : S.on_parabola ∧ λ • vector_PR P = vector_PQ P) :
  let area := (4 * (P.x^2 + 1) / (8 * P.x)) in
  area = 2 * P.x^3 + P.x + (1 / (8 * P.x)) → area.min > 0 :=
sorry

end find_lambda_find_min_area_l798_798489


namespace word_identification_l798_798712

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l798_798712


namespace new_salary_after_increase_l798_798313

theorem new_salary_after_increase : 
  ∀ (previous_salary : ℝ) (percentage_increase : ℝ), 
    previous_salary = 2000 → percentage_increase = 0.05 → 
    previous_salary + (previous_salary * percentage_increase) = 2100 :=
by
  intros previous_salary percentage_increase h1 h2
  sorry

end new_salary_after_increase_l798_798313


namespace word_identification_l798_798713

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l798_798713


namespace sequence_general_formula_l798_798910

noncomputable def a : ℕ → ℝ
| 1       := 7
| (n + 1) := 7 * a n / (a n + 7)

theorem sequence_general_formula (n : ℕ) (hn : n > 0) : a n = 7 / n :=
by
  sorry

end sequence_general_formula_l798_798910


namespace min_value_expression_l798_798880

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) >= 1 / 4) ∧ (x = 1/3 ∧ y = 1/3 ∧ z = 1/3 → x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1 / 4) :=
sorry

end min_value_expression_l798_798880


namespace largest_smallest_crab_difference_total_weight_deviation_l798_798832

/-- Proof for the difference between largest and smallest deviation -/
theorem largest_smallest_crab_difference :
  (0.2 - (-0.3) = 0.5) :=
by
  calc
    0.2 - (-0.3) = 0.2 + 0.3 := by simp
    ...           = 0.5       := by simp

/-- Proof for the total deviation of the box of crabs -/
theorem total_weight_deviation :
  ((-0.3 * 1) + (-0.2 * 2) + (-0.1 * 3) + (0 * 1) + (0.1 * 4) + (0.2 * 2) = -0.2) :=
by
  calc
    (-0.3 * 1) + (-0.2 * 2) + (-0.1 * 3) + (0 * 1) + (0.1 * 4) + (0.2 * 2) 
    = -0.3 + (-0.4) + (-0.3) + 0 + 0.4 + 0.4 := by simp
    ... = -0.2                                := by simp

end largest_smallest_crab_difference_total_weight_deviation_l798_798832


namespace shortest_side_of_right_triangle_l798_798085

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  ∀ c, (c = 5 ∨ c = 12 ∨ c = (Real.sqrt (a^2 + b^2))) → c = 5 :=
by
  intros c h
  sorry

end shortest_side_of_right_triangle_l798_798085


namespace fraction_expression_value_l798_798813

theorem fraction_expression_value:
  (1/4 - 1/5) / (1/3 - 1/6) = 3/10 :=
by
  sorry

end fraction_expression_value_l798_798813


namespace second_player_can_form_palindrome_l798_798724

def is_palindrome (s : List Char) : Prop :=
  s = s.reverse

theorem second_player_can_form_palindrome :
  ∀ (moves : List Char), moves.length = 1999 →
  ∃ (sequence : List Char), sequence.length = 1999 ∧ is_palindrome sequence :=
by
  sorry

end second_player_can_form_palindrome_l798_798724


namespace integer_solution_count_l798_798548

theorem integer_solution_count (x : ℝ) : 
  set.count {x | |x - 3| ≤ 7.5 ∧ x ∈ ℤ } = 15 := 
sorry

end integer_solution_count_l798_798548


namespace n_prime_or_power_of_2_l798_798614

theorem n_prime_or_power_of_2 (n : ℕ) (h1 : 6 < n)
  (h2 : ∃ (a : fin n → ℕ), ∀ i, i > 0 → i < n → nat.coprime (a i) n)
  (h3 : ∀ (i : ℕ), i > 0 → i < n-1 → (a (i+1) - a i = a 2 - a 1 > 0)) :
  nat.prime n ∨ ∃ m : ℕ, n = 2^m :=
by sorry

end n_prime_or_power_of_2_l798_798614


namespace max_distance_P_to_D_l798_798073

theorem max_distance_P_to_D
  (P : ℝ × ℝ)
  (u v w : ℝ)
  (h1 : u = real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2))
  (h2 : v = real.sqrt ((P.1 - 2)^2 + (P.2 - 0)^2))
  (h3 : w = real.sqrt ((P.1 - 2)^2 + (P.2 - 2)^2))
  (h4 : u^2 + v^2 = 2 * w^2) :
  real.sqrt ((P.1 - 0)^2 + (P.2 - 2)^2) ≤ 2 * real.sqrt 2 :=
sorry

end max_distance_P_to_D_l798_798073


namespace calculate_share_A_l798_798086

-- Defining the investments
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def investment_D : ℕ := 13000
def investment_E : ℕ := 21000
def investment_F : ℕ := 15000
def investment_G : ℕ := 9000

-- Defining B's share
def share_B : ℚ := 3600

-- Function to calculate total investment
def total_investment : ℕ :=
  investment_A + investment_B + investment_C + investment_D + investment_E + investment_F + investment_G

-- Ratio of B's investment to total investment
def ratio_B : ℚ :=
  investment_B / total_investment

-- Calculate total profit using B's share and ratio
def total_profit : ℚ :=
  share_B / ratio_B

-- Ratio of A's investment to total investment
def ratio_A : ℚ :=
  investment_A / total_investment

-- Calculate A's share based on the total profit
def share_A : ℚ :=
  total_profit * ratio_A

-- The theorem to prove the share of A is approximately $2292.34
theorem calculate_share_A : 
  abs (share_A - 2292.34) < 0.01 :=
by
  sorry

end calculate_share_A_l798_798086


namespace distance_on_dirt_road_is_1_km_l798_798391

variable (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ)

def distance_between_on_dirt_road (initial_gap : ℝ) (highway_speed : ℝ) (city_speed : ℝ) (good_road_speed : ℝ) (dirt_road_speed : ℝ) : ℝ :=
  initial_gap * (city_speed / highway_speed) * (good_road_speed / city_speed) * (dirt_road_speed / good_road_speed)

theorem distance_on_dirt_road_is_1_km :
  distance_between_on_dirt_road 2 60 40 70 30 = 1 :=
  by
    unfold distance_between_on_dirt_road
    sorry

end distance_on_dirt_road_is_1_km_l798_798391


namespace place_sweet_hexagons_l798_798418

def sweetHexagon (h : ℝ) : Prop := h = 1
def convexPolygon (A : ℝ) : Prop := A ≥ 1900000
def hexagonPlacementPossible (N : ℕ) : Prop := N ≤ 2000000

theorem place_sweet_hexagons:
  (∀ h, sweetHexagon h) →
  (∃ A, convexPolygon A) →
  (∃ N, hexagonPlacementPossible N) →
  True :=
by
  intros _ _ _ 
  exact True.intro

end place_sweet_hexagons_l798_798418


namespace path_is_ellipse_approx_parabola_l798_798072

-- Definitions based on conditions
variable (surface_radius : ℝ) -- Radius from the center to the surface of the spherical planet
variable (short_flight_length : ℝ) -- Length of the projectile's flight
variable (gravitational_constant : ℝ) -- Gravitational constant for the planet
variable (point_mass_gravity : linear_map ℝ (ℝ × ℝ) (ℝ × ℝ)) -- Approximation of gravity due to point mass

-- Hypotheses representing conditions in the Lean 4 statement
hypothesis (h1 : short_flight_length ≪ surface_radius) -- Flight length is much shorter than the planet's radius
hypothesis (h2 : ∀ (r : ℝ), r ≥ surface_radius → point_mass_gravity r ≈ gravitational_constant / r^2) -- Gravity due to point mass

-- The main theorem statement
theorem path_is_ellipse_approx_parabola :
  ∀ (path : linear_map ℝ ℝ (ℝ × ℝ)), (path.pathOfFlight short_flight_length ≈ parabola_path) →
  ∃ (ellipse : linear_map ℝ ℝ (ℝ × ℝ)), ellipse.pathOfFlight short_flight_length ≈ path :=
by
  sorry

end path_is_ellipse_approx_parabola_l798_798072


namespace fred_took_black_marbles_l798_798329

theorem fred_took_black_marbles :
  ∀ (initial_black: ℕ) (left_black: ℕ), initial_black = 792 → left_black = 559 → initial_black - left_black = 233 :=
by
  intros initial_black left_black h1 h2
  rw [h1, h2] 
  exact rfl

end fred_took_black_marbles_l798_798329


namespace books_sold_on_monday_75_l798_798970

namespace Bookstore

variables (total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold : ℕ)
variable (percent_not_sold : ℝ)

def given_conditions : Prop :=
  total_books = 1200 ∧
  percent_not_sold = 0.665 ∧
  sold_Tuesday = 50 ∧
  sold_Wednesday = 64 ∧
  sold_Thursday = 78 ∧
  sold_Friday = 135 ∧
  books_not_sold = (percent_not_sold * total_books) ∧
  (total_books - books_not_sold) = (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Thursday + sold_Friday)

theorem books_sold_on_monday_75 (h : given_conditions total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold percent_not_sold) :
  sold_Monday = 75 :=
sorry

end Bookstore

end books_sold_on_monday_75_l798_798970


namespace problem_solution_l798_798289

-- Define the function star (sum of digits)
def star (x : ℕ) := x.digits.sum

-- Define the predicate for set S
def S (n : ℕ) : Prop := star n = 15 ∧ n < 1000000

-- Define the set S
def set_S := {n : ℕ | S n}

-- The number of elements in set S
noncomputable def m : ℕ := set_S.to_finite.to_finset.card

-- The proof statement
theorem problem_solution : star m = 18 :=
by
  sorry

end problem_solution_l798_798289


namespace value_of_k_l798_798165

theorem value_of_k :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x →  (x^2 + 7 * x + 10) = (k + 3 * x)) ∧
  ∃ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ∧ discriminant (x^2 + 4 * x + (10 - k)) = 0 :=
begin
  use 6,
  sorry
end

end value_of_k_l798_798165


namespace acute_triangle_B_area_l798_798254

-- Basic setup for the problem statement
variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to respective angles

-- The theorem to be proven
theorem acute_triangle_B_area (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) 
                              (h_sides : a = 2 * b * Real.sin A)
                              (h_a : a = 3 * Real.sqrt 3) 
                              (h_c : c = 5) : 
  B = π / 6 ∧ (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end acute_triangle_B_area_l798_798254


namespace description_of_S_l798_798981

def points_set : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in 
       (3 = x + 2 ∧ y - 4 ≤ 3)
    ∨ (3 = y - 4 ∧ x + 2 ≤ 3)
    ∨ (x + 2 = y - 4 ∧ 3 ≤ x + 2)}

theorem description_of_S : points_set = 
  {p | let (x, y) := p in 
       (x = 1 ∧ y ≤ 7) 
    ∨ (y = 7 ∧ x ≤ 1) 
    ∨ (y = x + 6 ∧ 1 ≤ x)} :=
sorry

end description_of_S_l798_798981


namespace coefficient_x5y2_l798_798585

-- Define the binomial in question
def polynomial := (x^2 + x + y) ^ 6

-- Define the term we are interested in
def term := x^5 * y^2

-- Define what we want to prove
theorem coefficient_x5y2 : coefficient polynomial term = 60 :=
by
  sorry

end coefficient_x5y2_l798_798585


namespace large_box_chocolate_bars_l798_798068

theorem large_box_chocolate_bars (num_small_boxes : ℕ) (chocolates_per_box : ℕ) 
  (h1 : num_small_boxes = 18) (h2 : chocolates_per_box = 28) : 
  num_small_boxes * chocolates_per_box = 504 := by
  sorry

end large_box_chocolate_bars_l798_798068


namespace minor_axis_geometric_mean_l798_798855

/-- Given a cone with a half-angle of 30 degrees and an elliptic base,
    Prove that the minor axis of the ellipse (2b) is the geometric mean
    of the smallest (BC) and largest (AC) generatrices of the cone. -/
theorem minor_axis_geometric_mean 
  (AC BC : ℝ) 
  (b : ℝ) 
  (h1 : angle AC BC = 60) -- Equivalent to half-angle 30 degrees for the surface of revolution
  (h2 : (2*b)^2 = AC * BC): 
  (2 * b)^2 = AC * BC :=
sorry

end minor_axis_geometric_mean_l798_798855


namespace math_problem_solution_l798_798530

-- Definition for the given function
def f (x : ℝ) : ℝ := cos x ^ 2 + (sqrt 3) * sin x * cos x

-- Define triangle area function
def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ := 1 / 2 * b * c * sin A

-- Define cosine rule
def cosine_rule (a b c : ℝ) (A : ℝ) : ℝ := a^2 = b^2 + c^2 - 2 * b * c * cos A

noncomputable def problem_statement : Prop := 
  ((∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, (f y = 3/2 → y % π = π / 6)) ) ∧
  (∀ a b c A S, a = 3 ∧ S = sqrt 3 ∧ triangle_area a b c A (π / 2 - A) (π / 2 - A) = S →
  b ^ 2 + c ^ 2 = 21)

theorem math_problem_solution : problem_statement :=
sorry -- proof to be done later

end math_problem_solution_l798_798530


namespace even_diagonal_moves_l798_798404

def King_Moves (ND D : ℕ) :=
  ND + D = 63 ∧ ND % 2 = 0

theorem even_diagonal_moves (ND D : ℕ) (traverse_board : King_Moves ND D) : D % 2 = 0 :=
by
  sorry

end even_diagonal_moves_l798_798404


namespace number_of_valid_x_values_l798_798690

theorem number_of_valid_x_values (x : ℕ) 
  (h1 : 24 < x) 
  (h2 : x < 50) 
  (h3 : ∀ x, 24 < x → x < 50 → Integer):
  ∃ x_list, x_list = (list.range (49 + 1)).filter (λ y, 24 < y) ∧ x_list.length = 25 :=
by
  sorry

end number_of_valid_x_values_l798_798690


namespace find_x_equals_4_l798_798734

noncomputable def repeatingExpr (x : ℝ) : ℝ :=
2 + 4 / (1 + 4 / (2 + 4 / (1 + 4 / x)))

theorem find_x_equals_4 :
  ∃ x : ℝ, x = repeatingExpr x ∧ x = 4 :=
by
  use 4
  sorry

end find_x_equals_4_l798_798734


namespace intersection_of_A_B_l798_798628

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable (R : Type) [LinearOrder R]

def U := Set.univ
def A := {x : ℝ | x * (x - 2) < 0}
def B := {x : ℝ | x - 1 > 0}

theorem intersection_of_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

end intersection_of_A_B_l798_798628


namespace point_in_third_quadrant_l798_798944

theorem point_in_third_quadrant (m : ℝ) :
  let P := (-m^2 - 1, -1) in
  P.1 < 0 ∧ P.2 < 0 := 
by
  sorry

end point_in_third_quadrant_l798_798944


namespace transformed_cube_edges_l798_798408

-- Let's define the problem statement
theorem transformed_cube_edges : 
  let original_edges := 12 
  let new_edges_per_edge := 2 
  let additional_edges_per_pyramid := 1 
  let total_edges := original_edges + (original_edges * new_edges_per_edge) + (original_edges * additional_edges_per_pyramid) 
  total_edges = 48 :=
by sorry

end transformed_cube_edges_l798_798408


namespace find_ratio_l798_798528

variable {R : Type} [LinearOrderedField R]

def f (x a b : R) : R := x^3 + a*x^2 + b*x - a^2 - 7*a

def condition1 (a b : R) : Prop := f 1 a b = 10

def condition2 (a b : R) : Prop :=
  let f' := fun x => 3*x^2 + 2*a*x + b
  f' 1 = 0

theorem find_ratio (a b : R) (h1 : condition1 a b) (h2 : condition2 a b) :
  a / b = -2 / 3 :=
  sorry

end find_ratio_l798_798528


namespace seating_arrangements_l798_798010

theorem seating_arrangements :
  ∀ (chairs people : ℕ), 
  chairs = 8 → 
  people = 3 → 
  (∃ gaps : ℕ, gaps = 4) → 
  (∀ pos, pos = Nat.choose 3 4) → 
  pos = 24 :=
by
  intros chairs people h1 h2 h3 h4
  have gaps := 4
  have pos := Nat.choose 4 3
  sorry

end seating_arrangements_l798_798010


namespace lucy_snowballs_eq_19_l798_798112

-- Define the conditions
def charlie_snowballs : ℕ := 50
def difference_charlie_lucy : ℕ := 31

-- Define what we want to prove, i.e., Lucy has 19 snowballs
theorem lucy_snowballs_eq_19 : (charlie_snowballs - difference_charlie_lucy = 19) :=
by
  -- We would provide the proof here, but it's not required for this prompt
  sorry

end lucy_snowballs_eq_19_l798_798112


namespace perpendicular_diagonals_l798_798045

-- Definitions from the condition
variable (A B C D : Point)
variable [convex_quad ABCD : ConvexQuadrilateral A B C D]
variable [incircle : Circle]
variable [excircle : Circle]
variable (h_incircle_tangent : IncircleTangentToAllSides incircle ABCD)
variable (h_excircle_tangent : ExcircleTangentToAllSides excircle ABCD)

-- The theorem to prove
theorem perpendicular_diagonals :
  ∃ h_perpendicular : Perpendicular (diagonal AC) (diagonal BD) :=
sorry

end perpendicular_diagonals_l798_798045


namespace import_tax_percentage_l798_798071

theorem import_tax_percentage (V B T : ℝ) (hV : V = 2590) (hB : B = 1000) (hT : T = 111.30) :
  ∃ P : ℝ, P = 7 ∧ P * (V - B) / 100 = T :=
by {
  use 7,
  split,
  { refl, },
  { sorry, }
}

end import_tax_percentage_l798_798071


namespace equivalent_angle_in_radians_l798_798858

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * π / 180

theorem equivalent_angle_in_radians (α : ℝ) (h₁ : α = 2022) (h₂ : 0 < degrees_to_radians 222 ∧ degrees_to_radians 222 < 2 * π) :
  ∃ β, β = degrees_to_radians 222 ∧ β ∈ (0, 2 * π) :=
by {
  use degrees_to_radians 222,
  split,
  { refl },
  { exact h₂ },
}

end equivalent_angle_in_radians_l798_798858


namespace angle_between_a_b_correct_m_l798_798285

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (m : ℝ)
variables (OA OB OC : V)

/-- Given conditions as hypotheses --/
def conditions :=
  non_collinear a b ∧ 
  OA = m • a - b ∧ 
  OB = (m + 1) • a + b ∧ 
  OC = a - 3 • b

/-- Question 1: Given additional conditions, prove the angle between a and b --/
theorem angle_between_a_b 
  (h : conditions a b m OA OB OC) 
  (hm : m = -1 / 2) 
  (ha : ∥a∥ = 2 * real.sqrt 2 * ∥b∥) 
  (h_ortho : inner_product_space.dot_product OB OC = 0) : 
  real.angle a b = real.pi / 4 :=
sorry

/-- Question 2: Given collinearity condition, prove the value of m --/
theorem correct_m 
  (h : conditions a b m OA OB OC) 
  (h_collinear : collinear ℝ ![OA, OB, OC]) :
  m = 2 :=
sorry

end angle_between_a_b_correct_m_l798_798285


namespace validArithmeticExpressions_l798_798453

def isValidArithmetic (expressions : List (Nat → Nat → Nat × Nat)) : Prop :=
  let digits := List.join $ expressions.map (fun expr => 
    let (lhs, rhs) := expr in
    lhs.tostring.toList ++ rhs.tostring.toList
  )
  digits.sort = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def usesAllDigitsOnce (n : Nat) : List Char :=
  toString n |>.toList

theorem validArithmeticExpressions : 
    isValidArithmetic [
      (fun a b => (a + b, 8)),
      (fun c d => (c - d, 3)),
      (fun e f => (e * f, 20))
    ]
:= sorry

end validArithmeticExpressions_l798_798453


namespace max_distance_covered_l798_798742

theorem max_distance_covered 
  (D : ℝ)
  (h1 : (D / 2) / 5 + (D / 2) / 4 = 6) : 
  D = 40 / 3 :=
by
  sorry

end max_distance_covered_l798_798742


namespace number_of_ordered_tuples_is_one_l798_798475

theorem number_of_ordered_tuples_is_one :
  ∃! (a : Fin 19 → ℤ), ∀ i : Fin 19, a i ^ 3 = 3 * (∑ j in Finset.univ \ {i}, a j) :=
sorry

end number_of_ordered_tuples_is_one_l798_798475


namespace right_triangle_side_length_l798_798562

theorem right_triangle_side_length
  (c : ℕ) (a : ℕ) (h_c : c = 13) (h_a : a = 12) :
  ∃ b : ℕ, b = 5 ∧ c^2 = a^2 + b^2 :=
by
  -- Definitions from conditions
  have h_c_square : c^2 = 169 := by rw [h_c]; norm_num
  have h_a_square : a^2 = 144 := by rw [h_a]; norm_num
  -- Prove the final result
  sorry

end right_triangle_side_length_l798_798562


namespace product_of_935421_and_625_l798_798730

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 :=
by
  sorry

end product_of_935421_and_625_l798_798730


namespace poly_not_33_l798_798647

theorem poly_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by sorry

end poly_not_33_l798_798647


namespace average_cost_per_pen_is_21_l798_798406

/-- 
Prove that the average cost per pen, including shipping, in cents rounded to the nearest whole number is 21, 
given the total cost, the number of pens, shipping cost and conversion from dollars to cents.
-/
theorem average_cost_per_pen_is_21 :
  let cost_pens : ℝ := 24.50
  let shipping_cost : ℝ := 7.50
  let number_of_pens : ℝ := 150
  let total_cost : ℝ := cost_pens + shipping_cost
  let total_cost_cents : ℝ := total_cost * 100
  let average_cost_per_pen : ℝ := total_cost_cents / number_of_pens
  Nat.round average_cost_per_pen = 21 := 
by
  sorry

end average_cost_per_pen_is_21_l798_798406


namespace problem_statement_l798_798342

theorem problem_statement {f : ℝ → ℝ}
  (Hodd : ∀ x, f (-x) = -f x)
  (Hdecreasing : ∀ x y, x < y → f x > f y)
  (a b : ℝ) (H : f a + f b > 0) : a + b < 0 :=
sorry

end problem_statement_l798_798342


namespace part_a_part_b_part_c_l798_798483

variable (p q r : ℚ) [hp : p > 0] [hq : q > 0] [hr : r > 0]

def f (m : ℚ) (x : ℚ) := (1 / m) * x + m

noncomputable def G (m : ℚ) : Set (ℚ × ℚ) := {p : ℚ × ℚ | p.snd = f m p.fst}

theorem part_a (hpq : p ≠ q) : (G p ∩ G q).Nonempty := 
  sorry

theorem part_b (h : (∃ (p : 𝔽), (p ∈ G p ∩ G q) ∧ (p.fst ∈ ℤ) ∧ (p.snd ∈ ℤ))) : p ∈ ℤ ∧ q ∈ ℤ := 
  sorry

theorem part_c (hnat : ∃ (n : ℕ), p = n ∧ q = n + 1 ∧ r = n + 2) : 
  let A := (p * q, p + q)
  let B := (p * r, p + r)
  let C := (q * r, q + r)
  (1 / 2 : ℚ) * abs ((A.fst * (B.snd - C.snd) + B.fst * (C.snd - A.snd) + C.fst * (A.snd - B.snd))) = 1 := 
  sorry

end part_a_part_b_part_c_l798_798483


namespace zero_of_function_solve_inequality_range_of_m_l798_798199

-- Problem 1
theorem zero_of_function : ∃ x, (2^x - 2^(-x)) = 0 ∧ x = 0 :=
by sorry

-- Problem 2
theorem solve_inequality : ∀ x, (2^x - 2^(-x)) < 2 → x < real.log (1 + real.sqrt 2) / real.log 2 :=
by sorry

-- Problem 3
theorem range_of_m : ∀ m, (∃ x, (m * 2^x - 2^(1 - x) - (4 / 3) * m) = 0 ∧ 
  ∀ y, (m * 2^y - 2^(1 - y) - (4 / 3) * m = 0 → x = y)) → m ∈ set.insert (-3) (set.Ioi 1) :=
by sorry

end zero_of_function_solve_inequality_range_of_m_l798_798199


namespace minimum_discriminant_non_intersecting_regions_l798_798150

noncomputable def discriminant {a b c : ℝ} (h₁ : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
    (h₂ : ∀ x : ℝ, |x| < 1 → (1 / Real.sqrt (1 - x^2)) ≥ a * x^2 + b * x + c) : ℝ :=
b^2 - 4 * a * c

theorem minimum_discriminant_non_intersecting_regions :
  ∃ (a b c : ℝ), 
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ 
  (∀ x : ℝ, |x| < 1 → (1 / Real.sqrt (1 - x^2)) ≥ a * x^2 + b * x + c) ∧ 
  discriminant _ _ = -4 := 
sorry

end minimum_discriminant_non_intersecting_regions_l798_798150


namespace tangent_line_through_A_l798_798860

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
def f' (x : ℝ) : ℝ := 3 * x^2 - 3

theorem tangent_line_through_A (a : ℝ) : 
  (f a = a^3 - 3 * a) ∧ (0, 16) ∈ {(x, f' a * (x - a) + f a) | x : ℝ}  → 
  a = -2 ∧ ∀ x : ℝ, f x = x^3 - 3 * x → (f'(-2)).tangentAt(A) = (9: ℝ) * x - 16 := 
begin
  sorry
end

end tangent_line_through_A_l798_798860


namespace angle_BMC_is_right_angle_l798_798951

-- Define the properties and conditions involved in the problem.
variables (A B C D H M: Point)
variable (a: Real) -- Side length of the regular tetrahedron
variable (h: Real) -- Height of the regular tetrahedron

-- Tetrahedron ABCD is regular and has side length a.
def is_regular_tetrahedron (A B C D: Point) (a: Real) : Prop :=
  (distance A B = a) ∧ (distance A C = a) ∧ (distance A D = a) ∧ (distance B C = a) ∧ (distance B D = a) ∧ (distance C D = a)

-- H is the foot of the perpendicular from A to the base BCD.
def is_foot_of_altitude (A B C D H : Point) : Prop :=
  foot A B C D H

-- M is the midpoint of the altitude AH.
def is_midpoint (A H M: Point) : Prop :=
  midpoint A H M

-- Define the right angle property.
def is_right_angle (B M C : Point): Prop :=
  angle B M C = 90

-- The theorem to prove is that ∠BMC is 90 degrees given the conditions.
theorem angle_BMC_is_right_angle (A B C D H M: Point) (a h: Real)
  (h1 : is_regular_tetrahedron A B C D a)
  (h2 : is_foot_of_altitude A B C D H)
  (h3 : is_midpoint A H M) :
  is_right_angle B M C :=
  sorry

end angle_BMC_is_right_angle_l798_798951


namespace f_inv_undefined_at_one_l798_798230

def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

def f_inv (x : ℝ) : ℝ := (5 - 6 * x) / (1 - x)

theorem f_inv_undefined_at_one : ∃ x : ℝ, x = 1 ∧ ¬ (∃ y : ℝ, f_inv x = y) :=
by {
  sorry
}

end f_inv_undefined_at_one_l798_798230


namespace average_of_remaining_numbers_l798_798672

theorem average_of_remaining_numbers (S : ℕ) 
  (h₁ : S = 85 * 10) 
  (S' : ℕ) 
  (h₂ : S' = S - 70 - 76) : 
  S' / 8 = 88 := 
sorry

end average_of_remaining_numbers_l798_798672


namespace problem1_problem2_l798_798442

-- Theorem for problem 1
theorem problem1 (a b : ℤ) : (a^3 * b^4) ^ 2 / (a * b^2) ^ 3 = a^3 * b^2 := 
by sorry

-- Theorem for problem 2
theorem problem2 (a : ℤ) : (-a^2) ^ 3 * a^2 + a^8 = 0 := 
by sorry

end problem1_problem2_l798_798442


namespace total_number_of_selection_schemes_is_36_l798_798581

theorem total_number_of_selection_schemes_is_36 :
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"] in
  let roles := ["translator", "tour guide", "etiquette", "driver"] in
  (∀ z ∈ ["Xiao Zhang", "Xiao Zhao"], ∀ r ∈ ["translator", "tour guide"], True) →
  (∀ v ∈ ["Xiao Li", "Xiao Luo", "Xiao Wang"], ∀ r ∈ roles, True) →
  -- Actually calculate the possible arrangements
  num_selection_schemes volunteers roles = 36
:= sorry

end total_number_of_selection_schemes_is_36_l798_798581


namespace simplify_expression_l798_798034

theorem simplify_expression: ∀ (a m n : ℤ),
  (n % 2 = 1) → 
  (m > n) → 
  ((-a)^n = -a^n) → 
  (a^m / a^n = a^(m-n)) → 
  (-5)^5 / 5^3 + 3^4 - 6 = 50 :=
by
  intros a m n hn hm hneg hdiv
  sorry

end simplify_expression_l798_798034


namespace xiao_ming_payment_l798_798035

-- Definitions of the units and conversions
def liang_per_half_jin := 5
def total_weight := 2 * 16 + 7  -- as there are 16 liang in 1 jin
def loaves_of_bread := 9
def weight_per_loaf := 3
def total_loaf_weight := loaves_of_bread * weight_per_loaf

-- Given conditions
axiom maximal_half_jin_coupons : 10
axiom liang_per_2_liang_coupon : 2

-- Theorem to be proven
theorem xiao_ming_payment :
  ∃ (half_jin_coupons : ℕ) (liang_coupons_given_back : ℕ),
    half_jin_coupons <= maximal_half_jin_coupons ∧
    total_weight = half_jin_coupons * liang_per_half_jin - liang_coupons_given_back * liang_per_2_liang_coupon ∧
    half_jin_coupons = 7 ∧
    liang_coupons_given_back = 4 :=
begin
  -- here would be the proof steps
  sorry
end

end xiao_ming_payment_l798_798035
