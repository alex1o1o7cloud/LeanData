import Mathlib
import Mathlib.Algebra.ArithmeticSeries
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Group
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Continuity
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Derangements.Basic
import Mathlib.Combinatorics.SimpleGraph.Cliques
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Prob.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Algebra.Order.Basic
import Mathlib.Topology.LocalExtr.Set
import analysis.special_functions.exp_log
import data.real.sqrt

namespace trailing_zeroes_500_factorial_l628_628493

def numberOfTrailingZeros (n : Nat) : Nat :=
  let countDivisibleBy (k : Nat) : Nat :=
    if k > n then 0
    else n / k + countDivisibleBy (k * 5)
  countDivisibleBy 5

theorem trailing_zeroes_500_factorial : numberOfTrailingZeros 500 = 124 :=
  by
    sorry

end trailing_zeroes_500_factorial_l628_628493


namespace domain_of_f_l628_628279

def f (x : ℝ) : ℝ := 1 / ((x - 3) + 2 * (x - 6))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = (set.Ioo float.neg_inf 5 ∪ set.Ioo 5 float.inf) :=
by
  sorry

end domain_of_f_l628_628279


namespace part1_part2_part3_l628_628479

def f (x : ℝ) := x ^ 2
def g (x t : ℝ) := -x^2 + 2 * |x| + t
def h (x : ℝ) := 2^x - 2^(-x)

theorem part1 (m : ℝ) (h1 : m = 1) : ∀ x, (m^2 + m - 1) * x^(-2 * m^2 + m + 3) = x^2 := 
by simp [h1]

theorem part2 (x x1 x2 t : ℝ) (h1 : x ∈ Icc 1 2) (h2 : x1 ∈ Icc 1 2) (h3 : x2 ∈ Icc 1 2)
  (hx1 : f x ≤ f x1) (hx2 : g x t ≤ g x2 t) (h4 : f x1 = g x2 t) : t = 3 := 
by sorry

theorem part3 (x : ℝ) (λ : ℝ) (h1 : x ∈ Icc 1 2) (h2 : 2^x * h (2 * x) + λ * h x ≥ 0) :
  λ ≥ -5 := 
by sorry

end part1_part2_part3_l628_628479


namespace infinite_superset_of_infinite_subset_l628_628511

theorem infinite_superset_of_infinite_subset {A B : Set ℕ} (h_subset : B ⊆ A) (h_infinite : Infinite B) : Infinite A := 
sorry

end infinite_superset_of_infinite_subset_l628_628511


namespace min_packs_needed_l628_628227

theorem min_packs_needed
  (packs : List ℕ)
  (h_packs : packs = [8, 18, 30])
  (total_cans : ℕ)
  (h_total_cans : total_cans = 144)
  : ∃ n : ℕ, n = 6 ∧ ∀ combination : List ℕ, combination ∈ list.permutations_with_replacement packs n → combination.sum = total_cans := 
sorry

end min_packs_needed_l628_628227


namespace probability_not_grade_5_l628_628804

theorem probability_not_grade_5 :
  let A1 := 0.3
  let A2 := 0.4
  let A3 := 0.2
  let A4 := 0.1
  (A1 + A2 + A3 + A4 = 1) → (1 - A1 = 0.7) := by
  intros A1_def A2_def A3_def A4_def h
  sorry

end probability_not_grade_5_l628_628804


namespace product_of_invertibles_mod_120_l628_628147

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628147


namespace product_of_invertibles_mod_120_l628_628127

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628127


namespace limit_a_n_to_a_l628_628215

open Real

-- Definition of the sequence
def a_n (n : ℕ) : ℝ := (2 * n - 5) / (3 * n + 1)

-- Definition of the limit value
def a : ℝ := 2 / 3

-- The proof statement
theorem limit_a_n_to_a : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (a_n n - a) < ε := by
  sorry

end limit_a_n_to_a_l628_628215


namespace count_four_digit_numbers_divisible_by_17_and_end_in_17_l628_628486

theorem count_four_digit_numbers_divisible_by_17_and_end_in_17 :
  ∃ S : Finset ℕ, S.card = 5 ∧ ∀ n ∈ S, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0 ∧ n % 100 = 17 :=
by
  sorry

end count_four_digit_numbers_divisible_by_17_and_end_in_17_l628_628486


namespace find_special_four_digit_square_l628_628402

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l628_628402


namespace similar_triangles_sides_proportion_l628_628317

theorem similar_triangles_sides_proportion
  (A B C D E F : Type)
  [is_similar : is_similar (triangle A B C) (triangle D E F)]
  (BA_length : ℝ) (BC_length : ℝ) (ED_length : ℝ)
  (hBA : BA_length = 6) (hBC : BC_length = 10) (hED : ED_length = 4) :
  ∃ (EF_length : ℝ), EF_length ≈ 6.7 :=
by
  have h : EF_length = (BC_length / BA_length) * ED_length := sorry
  use (10 / 6) * 4
  have approx : (10 / 6) * 4 ≈ 6.67 := sorry
  exact approx

end similar_triangles_sides_proportion_l628_628317


namespace num_solutions_g_g_eq_5_l628_628624

def g : ℤ → ℤ
-- The definition of function g should be implicit based on given conditions.
-- We do not actually define it here, because it is supposed to be known and complex.

-- The given conditions from the problem
axiom g_eq_5_neg3 : g(-3) = 5
axiom g_eq_5_1 : g(1) = 5
axiom g_eq_5_5 : g(5) = 5
axiom g_eq_1_neg1 : g(-1) = 1

-- The problem requires us to find how many values of x satisfy g(g(x)) = 5
theorem num_solutions_g_g_eq_5 : 
  (∃ (x : ℤ), g (g x) = 5) ↔ 4 := sorry 

end num_solutions_g_g_eq_5_l628_628624


namespace original_number_of_men_l628_628700

theorem original_number_of_men (x : ℕ) (h1 : x * 10 = (x - 5) * 12) : x = 30 :=
by
  sorry

end original_number_of_men_l628_628700


namespace average_weight_increase_l628_628901

theorem average_weight_increase (n : ℕ) (w_old w_new : ℝ) (h1 : n = 9) (h2 : w_old = 65) (h3 : w_new = 87.5) :
  (w_new - w_old) / n = 2.5 :=
by
  rw [h1, h2, h3]
  norm_num

end average_weight_increase_l628_628901


namespace dartboard_partitions_count_l628_628363

theorem dartboard_partitions_count : 
  ∃ (l : Multiset ℕ), l.sum = 6 ∧ l.card ≤ 5 ∧ l.sort = l ∧ Multiset.card (Multiset.Powerset l) = 10 :=
sorry

end dartboard_partitions_count_l628_628363


namespace value_of_f_l628_628816

noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / (sin (π / 2 + α) * sin (-π - α))

theorem value_of_f (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : cos (α + π / 3) = 3 / 5) :
  f α = (4 * real.sqrt 3 - 3) / 10 :=
by
  sorry

end value_of_f_l628_628816


namespace line_l_problem_l628_628475

theorem line_l_problem (a : ℝ) :
  let line_l := (a^2 + a + 1) * x - y + 1 = 0 in
  (a = -1 → (∀ x y, (x + y = 0) → line_l → (-1 : ℝ) = 1)) ∧
  ((∀ x y, (x - y = 0) → (a^2 + a + 1 = 1) → (a * (a + 1) = 0 → (a = 0 ∨ a = -1))) = false) ∧
  (line_l ∧ (∀ x, x = 0 → 1 = 1) = true) ∧
  (a = 0 → let intercepts := (1 * x - y + 1 = 0) in
    (∀ x, x = -1) ∧ (∀ y, y = 1) = (false)) :=
sorry

end line_l_problem_l628_628475


namespace problem_I_problem_II_problem_III_l628_628857

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2 * x + f x a
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + a * x^2 + (a^2 + 2) * x
noncomputable def u (x : ℝ) : ℝ := (x - 1) / (2 * x)

-- Proof for (I)
theorem problem_I (a : ℝ) : a = -2 * Real.exp 1 ↔ ∃ x : ℝ, g x a = 0 := sorry

-- Proof for (II)
theorem problem_II {a : ℝ} :
  (a = 0 → ∀ x, 0 < x → h x a > 0) ∧
  (a > 0 → ∀ x, 0 < x → h x a > 0) ∧
  (a < 0 →
    (a > -Real.sqrt 2 →
        (∀ x, 0 < x ∧ x < -a/2 ∨ -1/a < x → h x a > 0) ∧
        (∀ x, -a/2 < x ∧ x < -1/a → h x a < 0)) ∧
    (a < -Real.sqrt 2 →
        (∀ x, 0 < x ∧ x < -1/a ∨ -a/2 < x → h x a > 0) ∧
        (∀ x, -1/a < x ∧ x < -a/2 → h x a < 0)) ∧
    (a = -Real.sqrt 2 → ∀ x, 0 < x → h x a > 0)) := sorry

-- Proof for (III)
theorem problem_III : ∃ a P, a = 1/2 ∧ P = (1 : ℝ, 0) ∧
  ((f P.1 a = u P.1) ∧
   (∀ x, f x a = u x → f' x a = u' x)) := sorry

end problem_I_problem_II_problem_III_l628_628857


namespace angleEqualityInConvexQuad_l628_628537

-- Define a structure for a convex quadrilateral
structure ConvexQuadrilateral (A B C D K M : Type) :=
  (AB_eq_CD : A ≠ B → A ≠ D → B ≠ C → C ≠ D → AB = CD)
  (K_on_AB : K ∈ segment A B)
  (M_on_CD : M ∈ segment C D)
  (AM_eq_KC : distance A M = distance K C)
  (BM_eq_KD : distance B M = distance K D)

-- Define a function to check the angle equality condition
def angleEqualityCondition (A B C D K M : Type) [ConvexQuadrilateral A B C D K M] : Prop :=
  angle (line A B) (line K M) = angle (line K M) (line C D)

-- Main theorem stating the angle equality
theorem angleEqualityInConvexQuad (A B C D K M : Type) [q: ConvexQuadrilateral A B C D K M] :
  angleEqualityCondition A B C D K M :=
by 
  sorry

end angleEqualityInConvexQuad_l628_628537


namespace fourth_group_students_l628_628259

theorem fourth_group_students (total_students group1 group2 group3 group4 : ℕ)
  (h_total : total_students = 24)
  (h_group1 : group1 = 5)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 7)
  (h_groups_sum : group1 + group2 + group3 + group4 = total_students) :
  group4 = 4 :=
by
  -- Proof will go here
  sorry

end fourth_group_students_l628_628259


namespace sides_of_regular_polygon_with_20_diagonals_l628_628036

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628036


namespace range_of_a_l628_628828

-- Definitions of conditions
def is_odd_function {A : Type} [AddGroup A] (f : A → A) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing {A : Type} [LinearOrderedAddCommGroup A] (f : A → A) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Main statement
theorem range_of_a 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_monotone_dec : is_monotonically_decreasing f)
  (h_domain : ∀ x, -7 < x ∧ x < 7 → -7 < f x ∧ f x < 7)
  (h_cond : ∀ a, f (1 - a) + f (2 * a - 5) < 0): 
  ∀ a, 4 < a → a < 6 :=
sorry

end range_of_a_l628_628828


namespace square_field_area_l628_628235

noncomputable def area_of_square_field(speed_kph : ℝ) (time_hrs : ℝ) : ℝ :=
  let speed_mps := (speed_kph * 1000) / 3600
  let distance := speed_mps * (time_hrs * 3600)
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

theorem square_field_area 
  (speed_kph : ℝ := 2.4)
  (time_hrs : ℝ := 3.0004166666666667) :
  area_of_square_field speed_kph time_hrs = 25939764.41 := 
by 
  -- This is a placeholder for the proof. 
  sorry

end square_field_area_l628_628235


namespace solution_set_ineq_l628_628836

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then x * (x - 1) else -(-x * (-x - 1))

theorem solution_set_ineq (m : ℝ) (h₁ : f (1 - m) = -(f(-(1 - m)))) (h₂ : f (1 - m^2) = -(f(-(1 - m^2)))) :
  f (1 - m) + f (1 - m^2) < 0 ↔ 0 ≤ m ∧ m < 1 := 
by {
  sorry
}

end solution_set_ineq_l628_628836


namespace a_b_positional_relationship_l628_628815

-- Definitions for parallel and non-parallel relations
def parallel (x y : Type) : Prop := sorry
def not_parallel (x y : Type) : Prop := sorry

-- Positional relationship determination
def positional_relation (x y : Type) : Type := sorry

-- a is parallel to c
axiom a_parallel_c (a c : Type) : parallel a c

-- b is not parallel to c
axiom b_not_parallel_c (b c : Type) : not_parallel b c

-- Define the positional relationship between a and b
theorem a_b_positional_relationship (a b c : Type) :
  (parallel a c) → (not_parallel b c) → (positional_relation a b = "intersecting or skew") :=
by
  intros h1 h2
  sorry

end a_b_positional_relationship_l628_628815


namespace balls_in_boxes_l628_628019

theorem balls_in_boxes :
  ∃ (ways : ℕ), ways = 42 ∧ ∀ (balls boxes : ℕ), balls = 6 → boxes = 2 → 
  ways = (choose 6 6) + (choose 6 5) + (choose 6 4) + (choose 6 3) := 
by
  sorry

end balls_in_boxes_l628_628019


namespace measure_of_arc_BD_l628_628618

-- Definitions for conditions
def diameter (A B M : Type) : Prop := sorry -- Placeholder definition for diameter
def chord (C D M : Type) : Prop := sorry -- Placeholder definition for chord intersecting at point M
def angle_measure (A B C : Type) (angle_deg: ℝ) : Prop := sorry -- Placeholder for angle measure
def arc_measure (C B : Type) (arc_deg: ℝ) : Prop := sorry -- Placeholder for arc measure

-- Main theorem to prove
theorem measure_of_arc_BD
  (A B C D M : Type)
  (h_diameter : diameter A B M)
  (h_chord : chord C D M)
  (h_angle_CMB : angle_measure C M B 73)
  (h_arc_BC : arc_measure B C 110) :
  ∃ (arc_BD : ℝ), arc_BD = 144 :=
by
  sorry

end measure_of_arc_BD_l628_628618


namespace ratio_of_largest_roots_l628_628955

noncomputable def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
noncomputable def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

theorem ratio_of_largest_roots :
  let x1 := sorry in -- let x1 be the largest root of f
  let x2 := sorry in -- let x2 be the largest root of g
  x2 = 2 * x1 :=
by sorry

end ratio_of_largest_roots_l628_628955


namespace product_of_all_valid_c_values_is_1307674368000_l628_628421

-- Define the condition for a quadratic equation to have two real roots
def has_two_real_roots (a b c : ℤ) : Prop := (b^2 - 4 * a * c > 0)

-- Define valid values of c based on the problem condition
def valid_c (c : ℤ) : Prop := 10 * (c^2) + 25 * c + c = 0 ∧ has_two_real_roots 10 25 c

noncomputable def product_of_valid_c : ℤ :=
  ∏ k in (Finset.range 16).filter (λ k, 1 ≤ k ∧ has_two_real_roots 10 25 k), k

-- Theorem statement
theorem product_of_all_valid_c_values_is_1307674368000 :
  product_of_valid_c = 1307674368000 :=
by 
  sorry

end product_of_all_valid_c_values_is_1307674368000_l628_628421


namespace gcd_pow_sub_one_l628_628280

theorem gcd_pow_sub_one (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2000 - 1) : Nat.gcd m n = 2^24 - 1 := 
by
  sorry

end gcd_pow_sub_one_l628_628280


namespace exists_x_mean_absolute_deviation_eq_half_l628_628584

theorem exists_x_mean_absolute_deviation_eq_half 
  {n : ℕ} (hn : 0 < n) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  ∃ t ∈ Icc (0 : ℝ) 1, (1 / (n : ℝ)) * (Finset.univ.sum (λ i, |t - x i|)) = 1 / 2 :=
begin
  sorry
end

end exists_x_mean_absolute_deviation_eq_half_l628_628584


namespace combined_work_time_l628_628308

noncomputable def work_time_first_worker : ℤ := 5
noncomputable def work_time_second_worker : ℤ := 4

theorem combined_work_time :
  (1 / (1 / work_time_first_worker + 1 / work_time_second_worker)) = 20 / 9 :=
by
  unfold work_time_first_worker work_time_second_worker
  -- The detailed reasoning and computation would go here
  sorry

end combined_work_time_l628_628308


namespace savings_per_bagel_in_cents_l628_628608

def cost_of_one_bagel : ℝ := 2.25
def cost_of_dozen_bagels : ℝ := 24.0
def number_of_bagels_in_dozen : ℕ := 12

theorem savings_per_bagel_in_cents :
  let total_cost_individual := number_of_bagels_in_dozen * cost_of_one_bagel in
  let total_savings_dollar := total_cost_individual - cost_of_dozen_bagels in
  let savings_per_bagel_dollar := total_savings_dollar / number_of_bagels_in_dozen in
  let savings_per_bagel_cents := savings_per_bagel_dollar * 100 in
  savings_per_bagel_cents = 25 :=
begin
  sorry
end

end savings_per_bagel_in_cents_l628_628608


namespace eugene_payment_correct_l628_628913

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l628_628913


namespace price_reduction_to_achieve_desired_profit_l628_628896

-- Defining the conditions
def initial_selling_price : ℝ := 40
def initial_sales_quantity : ℕ := 20
def additional_sales_quantity_per_yuan : ℕ := 2
def cost_price : ℝ := 24
def desired_profit : ℝ := 330

-- Define the algebraic expressions for actual selling price and sales quantity
def actual_selling_price (x : ℝ) : ℝ := initial_selling_price - x
def sales_quantity (x : ℕ) : ℕ := initial_sales_quantity + additional_sales_quantity_per_yuan * x

-- Define the profit equation
def daily_profit (x : ℝ) : ℝ := (actual_selling_price x - cost_price) * (sales_quantity x)

-- State the proof problem
theorem price_reduction_to_achieve_desired_profit : ∃ x : ℝ, daily_profit x = desired_profit ∧ x = 5 := by
  sorry

end price_reduction_to_achieve_desired_profit_l628_628896


namespace number_of_non_congruent_triangles_with_perimeter_20_l628_628488

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_20_l628_628488


namespace square_of_cube_of_third_smallest_prime_l628_628683

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l628_628683


namespace reconstruct_triangle_from_circumcircle_l628_628530

noncomputable theory

open Locale Classical
open Triangle Circles

theorem reconstruct_triangle_from_circumcircle
  (A B C : Point) 
  (triangle_ABC : Triangle A B C)
  (hA : Acute A)
  (hB : Acute B)
  (hC : Acute C)
  (circumcircle : Circle)
  (angle_bisector_C : Line)
  (altitude_A : Line)
  (altitude_B : Line)
  (intersect_angle_bisector_C_points : Intersection angle_bisector_C circumcircle)
  (intersect_altitude_A_points : Intersection altitude_A circumcircle)
  (intersect_altitude_B_points : Intersection altitude_B circumcircle ) :
  ∃ (A B C : Point), Triangle A B C :=
by
  sorry

end reconstruct_triangle_from_circumcircle_l628_628530


namespace sequence_general_formula_sequence_comparison_l628_628447

open Real

theorem sequence_general_formula (t : ℝ) (h1 : t ≠ 1) (h2 : t ≠ -1) :
  (∀ n : ℕ, n > 0 → a_n = (2 * (t^n - 1)) / n - 1) := sorry

theorem sequence_comparison (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) (h3 : t ≠ -1) :
  (∀ n : ℕ, n > 0 → a_{n+1} > a_n) := sorry

end sequence_general_formula_sequence_comparison_l628_628447


namespace problem_equivalent_final_answer_l628_628075

noncomputable def a := 12
noncomputable def b := 27
noncomputable def c := 6

theorem problem_equivalent :
  2 * Real.sqrt 3 + (2 / Real.sqrt 3) + 3 * Real.sqrt 2 + (3 / Real.sqrt 2) = (a * Real.sqrt 3 + b * Real.sqrt 2) / c :=
  sorry

theorem final_answer :
  a + b + c = 45 :=
  by
    unfold a b c
    simp
    done

end problem_equivalent_final_answer_l628_628075


namespace magic_square_vector_sum_zero_l628_628917

-- Define a magic square and the required properties
structure MagicSquare (n : ℕ) :=
  (grid : Fin n → Fin n → ℕ)
  (unique : ∀ i j k l : Fin n, grid i j = grid k l → i = k ∧ j = l)
  (range : ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ n ^ 2)
  (row_sum_eq : ∀ i : Fin n, ∑ j : Fin n, grid i j = (n * (n^2 + 1)) / 2)
  (col_sum_eq : ∀ j : Fin n, ∑ i : Fin n, grid i j = (n * (n^2 + 1)) / 2)

-- Statement of theorem to be proven
theorem magic_square_vector_sum_zero {n : ℕ} (M : MagicSquare n) :
  ∑ i j : Fin n, ∑ k l : Fin n, if M.grid i j > M.grid k l then -1 else if M.grid i j < M.grid k l then 1 else 0 = 0 :=
  sorry

end magic_square_vector_sum_zero_l628_628917


namespace perfect_square_pattern_l628_628399

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l628_628399


namespace product_invertibles_mod_120_l628_628158

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628158


namespace mumblian_possible_words_l628_628976

theorem mumblian_possible_words (alphabet_size : ℕ) (max_word_length : ℕ)
  (h_alphabet_size : alphabet_size = 5) (h_max_word_length : max_word_length = 3) : 
  ∑ i in finset.range (max_word_length + 1), alphabet_size ^ i - 1 = 155 :=
by
  sorry

end mumblian_possible_words_l628_628976


namespace quadratic_positive_range_l628_628446

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) ↔ ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) := 
by {
  sorry
}

end quadratic_positive_range_l628_628446


namespace angle_RPQ_108_degrees_l628_628093

theorem angle_RPQ_108_degrees
  (P Q R S : Type)
  [incidence_geometry P Q R S]
  (bisects_angle : QP bisects (angle SQR))
  (PQ_eq_PR : PQ = PR)
  (angle_RSQ : ∠RSQ = 2 * y)
  (angle_RPQ : ∠RPQ = 3 * y) :
  ∠RPQ = 108 := sorry

end angle_RPQ_108_degrees_l628_628093


namespace distance_to_lake_l628_628746

theorem distance_to_lake (d : ℝ) :
  ¬ (d ≥ 10) → ¬ (d ≤ 9) → d ≠ 7 → d ∈ Set.Ioo 9 10 :=
by
  intros h1 h2 h3
  sorry

end distance_to_lake_l628_628746


namespace function_range_real_l628_628293

theorem function_range_real (f : ℝ → ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ↔ f = λ x, -4 * x + 5 :=
by {
  sorry
}

end function_range_real_l628_628293


namespace fraction_of_draws_is_two_ninths_l628_628905

-- Define the fraction of games that Ben wins and Tom wins
def BenWins : ℚ := 4 / 9
def TomWins : ℚ := 1 / 3

-- Definition of the fraction of games ending in a draw
def fraction_of_draws (BenWins TomWins : ℚ) : ℚ :=
  1 - (BenWins + TomWins)

-- The theorem to be proved
theorem fraction_of_draws_is_two_ninths : fraction_of_draws BenWins TomWins = 2 / 9 :=
by
  sorry

end fraction_of_draws_is_two_ninths_l628_628905


namespace fish_still_water_speed_l628_628305

theorem fish_still_water_speed :
  (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 35) 
  (h2 : downstream_speed = 55) :
  (upstream_speed + downstream_speed) / 2 = 45 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end fish_still_water_speed_l628_628305


namespace part1_part2_l628_628767

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + Real.log x

-- Define the conditions
def condition1 (a : ℝ) : Prop := deriv (f 1 a) = 3 - 2 * a
def condition1_perpendicular (a : ℝ) : Prop := (3 - 2 * a) * (1 / 2) = -1

def g (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x + x - a + 3 / x
def g_deriv (x : ℝ) (a : ℝ) : ℝ := 2 / x + 1 - 3 / x^2
def condition2 (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Ioc (0 : ℝ) Real.exp → g x a ≥ 0

-- Prove that a == 5 / 2 satisfies condition 1
theorem part1 : ∀ a : ℝ, condition1_perpendicular a → a = 5 / 2 := by
  intro a
  assume h : condition1_perpendicular a
  sorry

-- Prove that the range of a that satisfies condition 2 is (-∞, 4]
theorem part2 : ∀ a : ℝ, condition2 a ↔ a ≤ 4 := by
  intro a
  sorry

end part1_part2_l628_628767


namespace regular_polygon_sides_l628_628052

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628052


namespace part1_part2_l628_628813

variable (α : ℝ)
variable (tan_α : ℝ) (sin_α : ℝ) (cos_α : ℝ)
variable (h_tanα : tan_α = 2)
variable (h_sin_α : sin_α = 2 * cos_α)

-- Define α in the third quadrant
axiom h_quadrant : π < α ∧ α < 3 * π / 2

theorem part1 (h : tan α = 2) : 
    (3 * sin α + 2 * cos α) / (sin α - cos α) = 8 :=
by 
  sorry

theorem part2 (h : tan α = 2) (h3_third_quadrant : π < α ∧ α < 3 * π / 2) : 
    cos α = - (real.sqrt 5) / 5 :=
by 
  sorry

end part1_part2_l628_628813


namespace prove_d_minus_r_eq_1_l628_628503

theorem prove_d_minus_r_eq_1 
  (d r : ℕ) 
  (h_d1 : d > 1)
  (h1 : 1122 % d = r)
  (h2 : 1540 % d = r)
  (h3 : 2455 % d = r) :
  d - r = 1 :=
by sorry

end prove_d_minus_r_eq_1_l628_628503


namespace cyclist_speed_north_l628_628267

theorem cyclist_speed_north (
  (v_south : ℝ) (t : ℝ) (d : ℝ),
  v_south = 40 → 
  t = 0.7142857142857143 → 
  d = 50
) : 
  let v_north := d / t - v_south in v_north = 30
:= by
  intros v_south t d h1 h2 h3
  sorry

end cyclist_speed_north_l628_628267


namespace triangle_APB_area_l628_628733

noncomputable
def area_triangle_apb : ℚ := 75 / 4

theorem triangle_APB_area
  (side_length : ℚ)
  (h1 : side_length = 10)
  (P : ℝ × ℝ)
  (A B C D G : ℝ × ℝ)
  (h2 : dist P A = dist P B ∧ dist P B = dist P C)
  (h3 : is_midpoint G (0, 10) (10, 10))
  (h4 : dist C P = dist C D ∧ (C.2 = 10 ∨ C.2 = 0))
  (h5 : ∃ E : ℝ × ℝ, collinear {A, B, E} ∧ is_right_angle (C, E, P)) :
  let area := (1/2) * side_length * (15/4) in area = area_triangle_apb :=
by
  sorry

end triangle_APB_area_l628_628733


namespace count_integers_satisfy_inequality_l628_628487

theorem count_integers_satisfy_inequality :
  {x : ℤ | (x + 5)^2 ≤ 4}.to_finset.card = 5 := 
sorry

end count_integers_satisfy_inequality_l628_628487


namespace exists_natural_number_with_digit_sum_1990_l628_628544

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem exists_natural_number_with_digit_sum_1990 :
  ∃ m : ℕ, sum_of_digits m = 1990 ∧ sum_of_digits (m ^ 2) = 1990 ^ 2 :=
begin
  sorry
end

end exists_natural_number_with_digit_sum_1990_l628_628544


namespace hyperbola_perimeter_l628_628442

/--
Let F₁ and F₂ be the left and right foci of a hyperbola, respectively. 
Given:
1. A chord AB on the left branch of the hyperbola passing through F₁ with a length of 5.
2. 2a = 8.

Prove: The perimeter of triangle ABF₂ is 26.
-/
theorem hyperbola_perimeter (F₁ F₂ A B : Point)
    (h_hyperbola : ∀ P, |P - F₂| - |P - F₁| = 8)
    (h_chord : |A - B| = 5)
    (h_foci_distance : 2 * a = 8) :
    |A - F₂| + |B - F₂| + |A - B| = 26 := 
  sorry

end hyperbola_perimeter_l628_628442


namespace required_percentage_to_pass_l628_628341

-- Definitions based on conditions
def obtained_marks : ℕ := 175
def failed_by : ℕ := 56
def max_marks : ℕ := 700
def pass_marks : ℕ := obtained_marks + failed_by

-- Theorem stating the required percentage to pass
theorem required_percentage_to_pass : 
  (pass_marks : ℚ) / max_marks * 100 = 33 := 
by 
  sorry

end required_percentage_to_pass_l628_628341


namespace johns_total_pay_l628_628106

-- Define the given conditions
def lastYearBonus : ℝ := 10000
def CAGR : ℝ := 0.05
def numYears : ℕ := 1
def projectsCompleted : ℕ := 8
def bonusPerProject : ℝ := 2000
def thisYearSalary : ℝ := 200000

-- Define the calculation for the first part of the bonus using the CAGR formula
def firstPartBonus (presentValue : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  presentValue * (1 + growthRate)^years

-- Define the calculation for the second part of the bonus
def secondPartBonus (numProjects : ℕ) (bonusPerProject : ℝ) : ℝ :=
  numProjects * bonusPerProject

-- Define the total pay calculation
def totalPay (salary : ℝ) (bonus1 : ℝ) (bonus2 : ℝ) : ℝ :=
  salary + bonus1 + bonus2

-- The proof statement, given the conditions, prove the total pay is $226,500
theorem johns_total_pay : totalPay thisYearSalary (firstPartBonus lastYearBonus CAGR numYears) (secondPartBonus projectsCompleted bonusPerProject) = 226500 := 
by
  -- insert proof here
  sorry

end johns_total_pay_l628_628106


namespace polar_equation_of_C_slope_of_line_l_l628_628097

-- Define the parametric equations of curve C
def parametric_curve_C (theta: ℝ) : ℝ × ℝ := (sqrt 5 * cos theta, 3 + sqrt 5 * sin theta)

-- Define the polar equation of curve C
def polar_eq_curve_C (rho theta: ℝ) : Prop := rho^2 - 6 * rho * sin theta + 4 = 0

-- Prove the polar equation of curve C
theorem polar_equation_of_C (theta: ℝ) :
  ∃ rho, polar_eq_curve_C rho theta :=
sorry

-- Define the parametric equations of line l
def parametric_line_l (t alpha: ℝ) : ℝ × ℝ := (t * cos alpha, t * sin alpha)

-- Define the slope condition on the line l
def slope_condition (alpha: ℝ) : Prop := tan alpha = sqrt (7 / 2) / 2 ∨ tan alpha = -sqrt (7 / 2) / 2

-- Prove the slope of the line l given the distance condition
theorem slope_of_line_l (alpha: ℝ) (h: 2 ≤ sqrt (7 / 2)) :
  slope_condition alpha :=
sorry

end polar_equation_of_C_slope_of_line_l_l628_628097


namespace square_of_cube_of_third_smallest_prime_l628_628681

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l628_628681


namespace rhombus_area_calculation_l628_628615

def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_calculation (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 18) : rhombus_area d1 d2 = 270 :=
by
  rw [h1, h2]
  -- Here we should demonstrate that the actual calculation is correct, but since the proof is not required, we use sorry.
  sorry

end rhombus_area_calculation_l628_628615


namespace sequence_a1000_l628_628522

theorem sequence_a1000 :
  ∃ (a : ℕ → ℤ),
    a 1 = 1001 ∧
    a 2 = 1004 ∧
    (∀ n ≥ 1, a n + a (n + 1) - 2 * a (n + 2) = 2) ∧
    a 1000 = 1004 :=
begin
  sorry
end

end sequence_a1000_l628_628522


namespace simplify_expression_value_of_A_plus_3B_l628_628709

-- Part 1: Proof of the simplified expression
theorem simplify_expression (x y : ℝ) :
  (3 * x ^ 2 - 2 * x * y + 5 * y ^ 2) - 2 * (x ^ 2 - x * y - 2 * y ^ 2) = x ^ 2 + 9 * y ^ 2 :=
by
  sorry

-- Part 2: Proof of A + 3B given the conditions
theorem value_of_A_plus_3B (x y : ℝ) (h : x + 2 * y = 6) :
  let A := - x - 2 * y - 1
      B := x + 2 * y + 2
  in A + 3 * B = 17 :=
by
  sorry

end simplify_expression_value_of_A_plus_3B_l628_628709


namespace sphere_volume_in_cone_with_diameter_18_base_angle_90_l628_628333

-- Define the radius of the sphere
def radius_of_sphere_in_cone (D : ℝ) (angle : ℝ) : ℝ :=
  let a := D / 2 in
  a * (1 - 1 / sqrt 2)

-- Define the volume of the sphere given a radius
def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- The conditions are:
-- Diameter D = 18 inches
-- Vertex angle = 90 degrees
-- Prove the volume of the sphere equals the given expression

theorem sphere_volume_in_cone_with_diameter_18_base_angle_90 :
  volume_of_sphere (radius_of_sphere_in_cone 18 (Real.pi / 2)) = 
  (4 / 3) * Real.pi * (9 - 9 / sqrt 2)^3 := by
  sorry

end sphere_volume_in_cone_with_diameter_18_base_angle_90_l628_628333


namespace candy_bar_cost_l628_628379

/-
Conditions:
- Dan has $3 (irrelevant for this specific proof, so we omit it)
- For a total of $4, he bought 2 candy bars.
- Each candy bar costs the same amount of money.
-/

theorem candy_bar_cost : ∀ (total_cost num_bars : ℕ), num_bars = 2 → total_cost = 4 → ∃ cost_per_bar, cost_per_bar = total_cost / num_bars :=
by {
    assume (total_cost num_bars : ℕ) (h1 : num_bars = 2) (h2 : total_cost = 4),
    use 2,
    rw [← h1, ← h2],
    exact eq.refl 2,
    sorry
}

end candy_bar_cost_l628_628379


namespace mixed_oil_rate_l628_628882

/-- Given quantities and prices of three types of oils, any combination
that satisfies the volume and price conditions will achieve a final mixture rate of Rs. 65 per litre. -/
theorem mixed_oil_rate (x y z : ℝ) : 
  12.5 * 55 + 7.75 * 70 + 3.25 * 82 = 1496.5 ∧ 12.5 + 7.75 + 3.25 = 23.5 →
  x + y + z = 23.5 ∧ 55 * x + 70 * y + 82 * z = 65 * 23.5 →
  true :=
by
  intros h1 h2
  sorry

end mixed_oil_rate_l628_628882


namespace scientific_notation_of_1300000_l628_628710

theorem scientific_notation_of_1300000 : 1300000 = 1.3 * 10^6 :=
by
  sorry

end scientific_notation_of_1300000_l628_628710


namespace sarah_speeding_tickets_l628_628366

def total_tickets (mark_speeding sarah_speeding mark_parking sarah_parking : ℕ) : ℕ :=
  mark_speeding + mark_parking + sarah_speeding + sarah_parking

theorem sarah_speeding_tickets :
  ∃ (sarah_speeding : ℕ), 
    let mark_speeding := sarah_speeding in
    let mark_parking := 8 in
    let sarah_parking := 4 in
    (2 * mark_parking + 2 * mark_speeding = 24) ∧
    mark_parking = 2 * sarah_parking ∧
    mark_speeding = sarah_speeding ∧
    total_tickets mark_speeding sarah_speeding mark_parking sarah_parking = 24 ∧
    sarah_speeding = 6 := sorry

end sarah_speeding_tickets_l628_628366


namespace line_properties_l628_628469

theorem line_properties (a : ℝ) :
  -- Condition of the problem
  let l := (a^2 + a + 1) * x - y + 1 = 0,

  -- Correct Answer (A): Line l is perpendicular to x + y = 0 when a = -1
  (a = -1 → (let k1 := (a^2 + a + 1)
             let k2 := -1
             (k1 * k2 = -1))) ∧
  
  -- Correct Answer (C): Line l passes through the point (0, 1) for any a
  (l 0 1 = 0) :=
  
by sorry

end line_properties_l628_628469


namespace tetrahedron_inscribable_in_sphere_l628_628780

theorem tetrahedron_inscribable_in_sphere 
  (T : Type) [tetrahedron T]
  (h : ∀ F : face T, ∃ r : ℝ, inscribed_circle F r ∧ r = 1) :
  ∃ R : ℝ, inscribed_sphere T R ∧ R = 3 / (2 * Real.sqrt 2) :=
begin
  sorry
end

end tetrahedron_inscribable_in_sphere_l628_628780


namespace average_possible_values_of_x_l628_628501

theorem average_possible_values_of_x :
  (∀ x : ℝ, (sqrt (2 * x^2 + 1) = sqrt 25) → 
    (((sqrt 12) + (-sqrt 12)) / 2) = 0) :=
by
  intro x h
  sorry

end average_possible_values_of_x_l628_628501


namespace product_of_invertibles_mod_120_l628_628152

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628152


namespace abc_equality_l628_628065

theorem abc_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                      (h : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equality_l628_628065


namespace value_of_expression_l628_628832

-- Definitions based on conditions
variables {θ : ℝ}
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (Real.sin θ, Real.cos θ)
def parallel (x y : ℝ × ℝ) : Prop := ∃ k : ℝ, x = (k * y.1, k * y.2)

-- Main theorem statement
theorem value_of_expression
  (h1 : parallel a b) : 2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11 / 5 :=
sorry

end value_of_expression_l628_628832


namespace range_of_gauss_f_l628_628430

def f (x : ℝ) : ℝ := (1/2) * x^2 - 3 * x + 4

def gauss (x : ℝ) : ℤ := Int.floor x

theorem range_of_gauss_f :
  (set.range (λ x : ℝ, gauss (f x))) = {-1, 0, 1} :=
by
  sorry

end range_of_gauss_f_l628_628430


namespace ratio_area_rectangle_triangle_l628_628284

noncomputable def area_rectangle (L W : ℝ) : ℝ :=
  L * W

noncomputable def area_triangle (L W : ℝ) : ℝ :=
  (1 / 2) * L * W

theorem ratio_area_rectangle_triangle (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  area_rectangle L W / area_triangle L W = 2 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end ratio_area_rectangle_triangle_l628_628284


namespace tan_sum_inequality_l628_628592

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end tan_sum_inequality_l628_628592


namespace assigned_values_are_zero_l628_628109

noncomputable def is_similar {α : Type*} [metric_space α] (P Q : finset α) : Prop :=
∃ φ : affine_map α α, φ.is_linear_embedding ∧ φ '' P = Q

variable {point : Type*} [add_torsor ℝ point]

def assigned_value (f : point → ℝ) (P : finset point) : Prop :=
∀ (Q : finset point), is_similar P Q → Q.sum f = 0

theorem assigned_values_are_zero (P : finset point) (f : point → ℝ) (h : assigned_value f P) :
  ∀ (K : point), f K = 0 :=
by sorry

end assigned_values_are_zero_l628_628109


namespace square_of_cube_of_third_smallest_prime_l628_628674

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628674


namespace evaluate_nested_fraction_l628_628396

-- We start by defining the complex nested fraction
def nested_fraction : Rat :=
  1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))

-- We assert that the value of the nested fraction is 8/21 
theorem evaluate_nested_fraction : nested_fraction = 8 / 21 := by
  sorry

end evaluate_nested_fraction_l628_628396


namespace problem_solution_l628_628089

open Real

noncomputable def curve_C1_parametric_x (t : ℝ) : ℝ := -1 + sqrt 2 * t
noncomputable def curve_C1_parametric_y (t : ℝ) : ℝ := 1 + sqrt 2 * t

noncomputable def curve_C1_cartesian : ∀ x y : ℝ, (x = curve_C1_parametric_x t) ∧ (y = curve_C1_parametric_y t) → x - y + 2 = 0 := 
sorry

noncomputable def curve_C2_cartesian (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 4 * ρ * sin θ + 1 = 0

noncomputable def curve_C3_polar (ρ θ : ℝ) : Prop :=
  θ = π / 4 ∧ ρ ∈ ℝ

noncomputable def curve_C3_intersect_C2_1 (ρ: ℝ) : Prop :=
  ρ^2 - 3 * sqrt 2 * ρ + 1 = 0

noncomputable def value_OA_OB (ρ1 ρ2 : ℝ) : Prop :=
  ρ1 + ρ2 = 3 * sqrt 2 ∧ ρ1 * ρ2 = 1 → (1 / abs ρ1) + (1 / abs ρ2) = 3 * sqrt 2

theorem problem_solution (t ρ1 ρ2: ℝ) (θ : ℝ) :
  (∀ t:ℝ, -1 + sqrt 2 * t = 1 + sqrt 2 * t) ∧ 
  (∀ x y : ℝ, ((x = -1 + sqrt 2 * t) ∧ (y = 1 + sqrt 2 * t)) → (x - y + 2 = 0)) ∧
  ((ρ² - 2 * ρ * cos θ - 4 * ρ * sin θ + 1 = 0) → curve_C2_cartesian (ρ * cos θ) (ρ * sin θ)) ∧
  (curve_C3_polar ρ θ → (θ = π / 4 ∧ curve_C3_intersect_C2_1 ρ))  ∧ 
  ((ρ1 + ρ2 = 3 * sqrt 2) ∧ (ρ1 * ρ2 = 1) → (1 / abs ρ1) + (1 / abs ρ2) = 3 * sqrt 2) :=
sorry

end problem_solution_l628_628089


namespace distance_between_vertices_l628_628386

-- Define the equations of the parabolas
def eq1 (x : ℝ) : ℝ := x^2 - 4*x + 5
def eq2 (x : ℝ) : ℝ := x^2 + 2*x + 4

-- Define the vertices
def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (-1, 3)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

-- Proof statement
theorem distance_between_vertices : distance vertex1 vertex2 = Real.sqrt 13 :=
by
  sorry

end distance_between_vertices_l628_628386


namespace product_coprime_mod_120_l628_628186

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628186


namespace square_area_proof_l628_628315

variables (A B C D P Q R S : ℝ²)
variables (h1 : ∃ s : ℝ, A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s))
variables (h2 : P = ((D.1 + C.1)/2, (D.2 + C.2)/2))
variables (h3 : Q = ((A.1 + D.1)/2, (A.2 + D.2)/2))
variables (h4 : quadrilateral_area Q B C P = 15)

noncomputable def area_of_square_ABCD : ℝ :=
  if ∃ s : ℝ, A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s)
  then s ^ 2
  else 0

theorem square_area_proof :
  ∃ (s : ℝ), h1 ∧ h2 ∧ h3 ∧ h4 → area_of_square_ABCD A B C D = 24 := by
  sorry

end square_area_proof_l628_628315


namespace interval_where_g_is_decreasing_l628_628856

def g (x : ℝ) : ℝ := Real.cos (2 * x - (Real.pi / 6))

theorem interval_where_g_is_decreasing :
  ∀ x, x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12) ↔ x ∈ Set.Icc 0 Real.pi ∧ StrictAnti (g x) (x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) :=
by
  sorry

end interval_where_g_is_decreasing_l628_628856


namespace sin_480_eq_sqrt3_div_2_l628_628777

theorem sin_480_eq_sqrt3_div_2 :
  real.sin (480 * real.pi / 180) = real.cos (30 * real.pi / 180) := 
by 
have h1 : 480 = 360 + 120, from rfl,
have h2 : real.sin (480 * real.pi / 180) = real.sin ((360 + 120) * real.pi / 180), by rw h1,
have h3 : real.sin ((360 + 120) * real.pi / 180) = real.sin (120 * real.pi / 180), by { rw [add_mul, add_mul, real.sin_add]; simp },
have h4 : real.sin (120 * real.pi / 180) = real.sin (90 * real.pi / 180 + 30 * real.pi / 180), from rfl,
have h5 : real.sin (90 * real.pi / 180 + 30 * real.pi / 180) = real.cos (30 * real.pi / 180), by { rw real.sin_add_pi_div_two },
exact eq.trans (eq.trans (eq.trans h2 h3) h4) h5

end sin_480_eq_sqrt3_div_2_l628_628777


namespace JamesFlowers_l628_628548

noncomputable def numberOfFlowersJamesPlantedInADay (F : ℝ) := 0.5 * (F + 0.15 * F)

theorem JamesFlowers (F : ℝ) (H₁ : 6 * F + (F + 0.15 * F) = 315) : numberOfFlowersJamesPlantedInADay F = 25.3:=
by
  sorry

end JamesFlowers_l628_628548


namespace product_pq_l628_628252

theorem product_pq {p q : ℝ} 
  (h1 : ∃ A B C : ℝ × ℝ, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (A.2 = p * A.1 - A.1 ^ 2) ∧ (B.2 = p * B.1 - B.1 ^ 2) ∧ (C.2 = p * C.1 - C.1 ^ 2) ∧ (A.1 * A.2 = q) ∧ (B.1 * B.2 = q) ∧ (C.1 * C.2 = q))
  (h2 : let A B C := classical.some h1, let A' := classical.some_spec h1 in let B' := A' \.some_spec.left in let C' := A' \.some_spec.right \.some_spec in 
          (A.1 - B.1)^2 + (B.1 - C.1)^2 + (C.1 - A.1)^2 + (A.2 - B.2)^2 + (B.2 - C.2)^2 + (C.2 - A.2)^2 = 324)
  (h3 : let A B C := classical.some h1, let A' := classical.some_spec h1 in let B' := A' \.some_spec.left in let C' := A' \.some_spec.right \.some_spec in
          dist ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) (0, 0) = 2) 
: p * q = 42 :=
sorry

end product_pq_l628_628252


namespace average_time_heard_is_41_4_l628_628898

theorem average_time_heard_is_41_4 :
  let d := 90 in
  let p1 := 0.25 in
  let p2 := 0.15 in
  let r := 1 - p1 - p2 in
  let p3 := 0.4 * r in
  let p4 := 0.6 * r in
  let total_minutes := p1 * d * 100 + p2 * 0 * 100 + p3 * 45 * 100 + p4 * 22.5 * 100 in
  let average_time_heard := total_minutes / 100 in
  average_time_heard = 41.4 :=
by
  sorry

end average_time_heard_is_41_4_l628_628898


namespace ordered_triples_count_l628_628792

theorem ordered_triples_count :
  (finset.card (finset.filter (λ (xyz : ℕ × ℕ × ℕ), 
    let (x, y, z) := xyz in x <= y ∧ y <= z ∧ x + y + z <= 100) 
    ((finset.range 101).product (finset.range 101).product (finset.range 101)))) = 30787 :=
sorry

end ordered_triples_count_l628_628792


namespace product_of_invertibles_mod_120_l628_628123

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628123


namespace find_200_digit_number_l628_628716

theorem find_200_digit_number : ∃ c ∈ {1, 2, 3}, ∃ (N : ℕ), N = 132 * c * 10^197 ∧ (N < 10^200 ∧ ∀ (N' : ℕ), remove_leading_and_third_digit N = N' → N = 44 * N') :=
by sorry

noncomputable def remove_leading_and_third_digit (N : ℕ) : ℕ := sorry

end find_200_digit_number_l628_628716


namespace total_investment_100000_l628_628971

variable (x : ℝ) (r_x r_y : ℝ) (interest_difference : ℝ) (total_investment : ℝ)

def investment_in_fund_Y (x : ℝ) (r_x r_y : ℝ) (interest_difference : ℝ) : ℝ :=
  interest_difference / r_y + r_x * x / r_y

theorem total_investment_100000 :
  x = 42000 → r_x = 0.23 → r_y = 0.17 → interest_difference = 200 →
  let y := investment_in_fund_Y x r_x r_y interest_difference in
  total_investment = x + y →
  total_investment = 100000 :=
by
  intros h1 h2 h3 h4 hy
  sorry

end total_investment_100000_l628_628971


namespace tax_percentage_excess_l628_628521

theorem tax_percentage_excess (income tax total_income : ℕ) (f_tax : ℕ → ℕ) (e_tax : ℕ → ℕ) 
  (h1 : ∀ x, f_tax x = x * 12 / 100) (h2 : income = 56000) (h3 : tax = 8000) (h4: e_tax = λ x, x * 100 / 16000) :
  e_tax (tax - f_tax 40000) = 20 :=
by
  sorry

end tax_percentage_excess_l628_628521


namespace sequence_a_n_l628_628256

theorem sequence_a_n (a : ℕ → ℕ) (h_condition : ∀ n, (finset.range n).sum (λ i, (1/(2^(i+1))) * a (i+1)) = 2*n + 1) :
  (∀ n, a n = if n = 1 then 6 else 2^(n+1)) :=
begin
  sorry
end

end sequence_a_n_l628_628256


namespace parabola_constant_unique_l628_628779

theorem parabola_constant_unique (b c : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 20) → y = x^2 + b * x + c) →
  (∀ x y : ℝ, (x = -2 ∧ y = -4) → y = x^2 + b * x + c) →
  c = 4 :=
by
    sorry

end parabola_constant_unique_l628_628779


namespace quadratic_no_real_roots_range_l628_628890

theorem quadratic_no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l628_628890


namespace question_1_question_2_question_3_range_of_a_l628_628438

-- Definitionally relevant conditions
def sqrt_c (c : ℝ) := sqrt c
def f (x : ℝ) (a : ℝ) := real.sqrt (2 ^ x)

-- Question 1
theorem question_1 (a : ℝ) (x : ℝ) (h : a = 1) : f x a > 1 ↔ 0 < x ∧ x < 1 := 
sorry -- Proof is omitted

-- Question 2
theorem question_2 (a : ℝ) (x : ℝ) : 
  (∃! x, f x a + real.log x ^ 2 = 0) ↔ (a = 0 ∨ a = - 1 / 4) :=
sorry -- Proof is omitted

-- Question 3
theorem question_3_range_of_a (a : ℝ) (h : a > 0) : 
  (∀ t ∈ set.Icc (1/2 : ℝ) 1, 
    abs (real.bsupr t (λ x : ℝ, ∃ x ∈ set.Icc t (t + 1), f x a) - 
         real.binfr t (λ x : ℝ, ∃ x ∈ set.Icc t (t + 1), f x a)) ≤ 1) ↔ 
  a ≥ 2 / 3 :=
sorry -- Proof is omitted

end question_1_question_2_question_3_range_of_a_l628_628438


namespace inequality_l628_628569

theorem inequality (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) 
(h_sum : ∑ i in Finset.range n, (i + 1) • a i = n * (n + 1) / 2) : 
  ∑ i in Finset.range n, 1 / (1 + (n - 1) * a i ^ i) ≥ 1 :=
by
  sorry

end inequality_l628_628569


namespace inequality_relations_l628_628504

variable {R : Type} [OrderedAddCommGroup R]
variables (x y z : R)

theorem inequality_relations (h1 : x - y > x + z) (h2 : x + y < y + z) : y < -z ∧ x < z :=
by
  sorry

end inequality_relations_l628_628504


namespace regular_polygon_sides_l628_628055

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628055


namespace product_pq_l628_628253

theorem product_pq {p q : ℝ} 
  (h1 : ∃ A B C : ℝ × ℝ, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (A.2 = p * A.1 - A.1 ^ 2) ∧ (B.2 = p * B.1 - B.1 ^ 2) ∧ (C.2 = p * C.1 - C.1 ^ 2) ∧ (A.1 * A.2 = q) ∧ (B.1 * B.2 = q) ∧ (C.1 * C.2 = q))
  (h2 : let A B C := classical.some h1, let A' := classical.some_spec h1 in let B' := A' \.some_spec.left in let C' := A' \.some_spec.right \.some_spec in 
          (A.1 - B.1)^2 + (B.1 - C.1)^2 + (C.1 - A.1)^2 + (A.2 - B.2)^2 + (B.2 - C.2)^2 + (C.2 - A.2)^2 = 324)
  (h3 : let A B C := classical.some h1, let A' := classical.some_spec h1 in let B' := A' \.some_spec.left in let C' := A' \.some_spec.right \.some_spec in
          dist ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) (0, 0) = 2) 
: p * q = 42 :=
sorry

end product_pq_l628_628253


namespace binary_arithmetic_correct_l628_628764

theorem binary_arithmetic_correct :
  (2^3 + 2^2 + 2^0) + (2^2 + 2^1 + 2^0) - (2^3 + 2^2 + 2^1) + (2^3 + 2^0) + (2^3 + 2^1) = 2^4 + 2^3 + 2^0 :=
by sorry

end binary_arithmetic_correct_l628_628764


namespace find_m_l628_628457

-- Definitions from conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1
def major_axis_twice_minor_axis (a b : ℝ) : Prop := a = 2 * b

-- Main statement
theorem find_m (m : ℝ) (h1 : ellipse_eq 0 0 m) (h2 : 0 < m) (h3 : 0 < m ∧ m < 1) :
  m = 1 / 4 :=
by
  sorry

end find_m_l628_628457


namespace find_f_2023_l628_628354

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2023 :
  (∀ x : ℝ, f (x + 2) = 3 - real.sqrt (6 * f x - f x * f x)) →
  (∀ x : ℝ, f x = f (-x)) →
  f 2023 = 3 - (3 / 2) * real.sqrt 2 :=
sorry

end find_f_2023_l628_628354


namespace percentage_excess_votes_l628_628918

theorem percentage_excess_votes :
  ∀ (total_votes : ℕ) (invalid_percentage : ℕ) (B_votes : ℕ),
  total_votes = 8720 →
  invalid_percentage = 20 →
  B_votes = 2834 →
  let valid_votes := total_votes * (100 - invalid_percentage) / 100 in
  let A_votes := valid_votes - B_votes in
  (A_votes - B_votes) * 100 / B_votes = 46.13 :=
by
  intros total_votes invalid_percentage B_votes h1 h2 h3
  let valid_votes := total_votes * (100 - invalid_percentage) / 100
  let A_votes := valid_votes - B_votes
  have : (A_votes - B_votes) * 100 / B_votes = 46.13 := sorry
  exact this

end percentage_excess_votes_l628_628918


namespace find_sides_from_diagonals_l628_628061

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628061


namespace problem_statement_l628_628837

-- Defining f as an odd function and given specific form for x > 0
def f : ℝ → ℝ := λ x, if x > 0 then x^3 - x else -(x^3 - x)

axiom odd_f : ∀ x : ℝ, f (-x) = - f x

-- The problem statement: Prove f(-2) = -6
theorem problem_statement : f (-2) = -6 := 
by
  sorry

end problem_statement_l628_628837


namespace end_of_month_books_count_l628_628329

theorem end_of_month_books_count:
  ∀ (initial_books : ℝ) (loaned_out_books : ℝ) (return_rate : ℝ)
    (rounded_loaned_out_books : ℝ) (returned_books : ℝ)
    (not_returned_books : ℝ) (end_of_month_books : ℝ),
    initial_books = 75 →
    loaned_out_books = 60.00000000000001 →
    return_rate = 0.65 →
    rounded_loaned_out_books = 60 →
    returned_books = return_rate * rounded_loaned_out_books →
    not_returned_books = rounded_loaned_out_books - returned_books →
    end_of_month_books = initial_books - not_returned_books →
    end_of_month_books = 54 :=
by
  intros initial_books loaned_out_books return_rate
         rounded_loaned_out_books returned_books
         not_returned_books end_of_month_books
  intros h_initial_books h_loaned_out_books h_return_rate
         h_rounded_loaned_out_books h_returned_books
         h_not_returned_books h_end_of_month_books
  sorry

end end_of_month_books_count_l628_628329


namespace smallest_integer_y_l628_628661

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end smallest_integer_y_l628_628661


namespace fred_earned_money_l628_628108

theorem fred_earned_money (initial_money final_money earned_money : ℤ) 
  (h_initial : initial_money = 23) 
  (h_final : final_money = 86) :
  earned_money = final_money - initial_money :=
by
  have h_earned := final_money - initial_money
  rw [h_initial, h_final] at h_earned
  exact h_earned

end fred_earned_money_l628_628108


namespace polygon_sides_from_diagonals_l628_628029

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628029


namespace balls_in_boxes_l628_628018

theorem balls_in_boxes :
  ∃ (ways : ℕ), ways = 42 ∧ ∀ (balls boxes : ℕ), balls = 6 → boxes = 2 → 
  ways = (choose 6 6) + (choose 6 5) + (choose 6 4) + (choose 6 3) := 
by
  sorry

end balls_in_boxes_l628_628018


namespace more_sons_than_daughters_prob_l628_628970

noncomputable def binom (n k : ℕ) : ℚ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

def prob_sons (k : ℕ) : ℚ := binom 8 k * (0.4^k) * (0.6^(8-k))

def prob_more_sons_than_daughters : ℚ :=
  prob_sons 5 + prob_sons 6 + prob_sons 7 + prob_sons 8

theorem more_sons_than_daughters_prob : prob_more_sons_than_daughters = 0.1752 := by
  sorry

end more_sons_than_daughters_prob_l628_628970


namespace find_m_find_A_inter_CUB_l628_628200

-- Definitions of sets A and B given m
def A (m : ℤ) : Set ℤ := {-4, 2 * m - 1, m ^ 2}
def B (m : ℤ) : Set ℤ := {9, m - 5, 1 - m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- First part: Prove that m = -3
theorem find_m (m : ℤ) : A m ∩ B m = {9} → m = -3 := sorry

-- Condition that m = -3 is true
def m_val : ℤ := -3

-- Second part: Prove A ∩ C_U B = {-4, -7}
theorem find_A_inter_CUB: A m_val ∩ (U \ B m_val) = {-4, -7} := sorry

end find_m_find_A_inter_CUB_l628_628200


namespace least_m_such_that_log_exceeds_6_l628_628954

def f(m : ℕ) : ℕ := 2^(2^(2^2)) % m

theorem least_m_such_that_log_exceeds_6 : ∃ (m : ℕ), (∀ n : ℕ, f(n) ≤ 10^6 → n ≤ 5) ∧ f(m) > 10^6 ∧ m = 5 :=
by
  sorry

end least_m_such_that_log_exceeds_6_l628_628954


namespace cube_painting_problem_l628_628737

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l628_628737


namespace find_segments_AD_DC_l628_628103

-- Define the points and triangle property
variables (A B C D : Type) [metric_space A]
variables (AB AC BD : line_segment A)
variables (BC : line_segment B C)
variables (AB_eq_2 : length AB = 2)
variables (AC_eq_4 : length AC = 4)
variables (angle_ABD_eq_angle_C : angle ABD = angle ACB)

-- Define the property to be proved
theorem find_segments_AD_DC (A B C D : Type) [metric_space A]
  (AB AC BD : line_segment A)
  (BC : line_segment B C)
  (AB_eq_2 : length AB = 2)
  (AC_eq_4 : length AC = 4)
  (angle_ABD_eq_angle_C : angle ABD = angle ACB) :
  length (AD : line_segment A D) = 1 ∧ length (DC : line_segment D C) = 3 :=
sorry

end find_segments_AD_DC_l628_628103


namespace area_triangle_PZM_l628_628924

theorem area_triangle_PZM :
  let L M N O P Q Y Z R G : Point := sorry -- Locations not needed precisely for the example
  let hexagon_equiangular : EquiangularHexagon L M N O P Q := sorry
  let square_LMXY : Square L M X Y (sqrt 50) := sorry
  let square_QPRG : Square Q P R G (sqrt 8) := sorry
  let triangle_MYZ : IsoscelesTriangle M Y Z MY := sorry ∧ MY = MZ
  let QO_eq_MP : QO = MP := sorry
  PZM_area P Z M = 5 :=
by
  sorry  -- Proof body not required

end area_triangle_PZM_l628_628924


namespace product_of_invertible_integers_mod_120_l628_628144

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628144


namespace custom_operations_correctness_l628_628651

theorem custom_operations_correctness : 
  (∀ a b : ℚ, a ∇ b = a + b - 1) → 
  (∀ a b : ℚ, a ⊙ b = a * b - a^2) → 
  (-2 : ℚ) ⊙ (8 ∇ (-3)) = -12 :=
by
  sorry

/-- We could define the operations in Lean -/
def ∇ (a b : ℚ) : ℚ := a + b - 1
def ⊙ (a b : ℚ) : ℚ := a * b - a ^ 2

end custom_operations_correctness_l628_628651


namespace oil_production_correct_rapeseed_requirements_correct_l628_628630

-- Definitions based on conditions
def oil_yield_rate : ℝ := 0.38
def rapeseed_amount_1 : ℝ := 500
def produced_oil_expected : ℝ := 190
def produced_oil_actual := rapeseed_amount_1 * oil_yield_rate

def required_oil_amount : ℝ := 380
def required_rapeseed_expected : ℝ := 1000
def required_rapeseed_actual := required_oil_amount / oil_yield_rate

-- Theorem statements based on the questions and expected answers
theorem oil_production_correct :
  produced_oil_actual = produced_oil_expected :=
by
  -- The actual proof would go here, we're using sorry to assume it
  sorry

theorem rapeseed_requirements_correct :
  required_rapeseed_actual = required_rapeseed_expected :=
by
  -- The actual proof would go here, we're using sorry to assume it
  sorry

end oil_production_correct_rapeseed_requirements_correct_l628_628630


namespace expression_identity_l628_628760

theorem expression_identity (x : ℝ) (n : ℕ) : 
  (1 - x) * (∑ k in Finset.range (n + 1), x ^ k) = 1 - x ^ (n + 1) :=
by
  sorry

end expression_identity_l628_628760


namespace shortest_distance_between_curve_and_point_l628_628926

noncomputable def shortestDist : ℝ :=
  let line := { p : Point | p.x - p.y = 1 }
  let circle_center : Point := ⟨-2, 1⟩
  let circle_radius : ℝ := 1
  let distance := λ p₁ p₂ : Point, 
    (| p₁.x - p₂.x | + | p₁.y - p₂.y |) / sqrt(2)
  distance circle_center line - circle_radius

theorem shortest_distance_between_curve_and_point 
  (C : Set Point) (M : Set Point) 
  (hC : ∀ (p : Point), p ∈ C → p.x - p.y - 1 = 0)
  (hM : ∀ (φ : ℝ), (Point.mk (-2 + cos φ) (1 + sin φ)) ∈ M) :
  shortest_distance C M = 2 * sqrt 2 - 1 :=
sorry

end shortest_distance_between_curve_and_point_l628_628926


namespace total_carpets_l628_628261

theorem total_carpets 
(house1 : ℕ) 
(house2 : ℕ) 
(house3 : ℕ) 
(house4 : ℕ) 
(h1 : house1 = 12) 
(h2 : house2 = 20) 
(h3 : house3 = 10) 
(h4 : house4 = 2 * house3) : 
  house1 + house2 + house3 + house4 = 62 := 
by 
  -- The proof will be inserted here
  sorry

end total_carpets_l628_628261


namespace smallest_m_l628_628750

-- Definitions of lengths and properties of the pieces
variable {lengths : Fin 21 → ℝ} 
variable (h_all_pos : ∀ i, lengths i > 0)
variable (h_total_length : (Finset.univ : Finset (Fin 21)).sum lengths = 21)
variable (h_max_factor : ∀ i j, max (lengths i) (lengths j) ≤ 3 * min (lengths i) (lengths j))

-- Proof statement
theorem smallest_m (m : ℝ) (hm : ∀ i j, max (lengths i) (lengths j) ≤ m * min (lengths i) (lengths j)) : 
  m ≥ 1 := 
sorry

end smallest_m_l628_628750


namespace vec_v_satisfies_conditions_l628_628425

open Real

noncomputable def vec_v : ℝ × ℝ :=
  (-85/187, 3060/187)

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := ((u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) in
  (scalar * v.1, scalar * v.2)

theorem vec_v_satisfies_conditions :
  proj vec_v (3, 1) = (45 / 10, 15 / 10) ∧
  proj vec_v (1, 4) = (65 / 17, 260 / 17) :=
sorry

end vec_v_satisfies_conditions_l628_628425


namespace ratio_of_areas_l628_628904
open Set

variables {A B C D E F P Q : Point}

-- Define regular hexagon
def regular_hexagon (A B C D E F : Point) : Prop :=
  -- define the property of regular hexagon (e.g., equal sides, equal angles)
  sorry

-- Define midpoints
def midpoint (P X Y : Point) : Prop := 
  -- define the property of midpoint
  sorry

-- Areas of quadrilaterals
def area (S : Set Point) : ℝ := sorry

-- Points A, B, C, D, E, F form a regular hexagon
axiom hexagon : regular_hexagon A B C D E F

-- Points P and Q are midpoints of BC and EF respectively
axiom mp_BC : midpoint P B C
axiom mp_EF : midpoint Q E F

-- The final theorem to prove
theorem ratio_of_areas : area ({A, B, P, Q} : Set Point) / area ({C, D, P, Q} : Set Point) = 1 := 
  sorry

end ratio_of_areas_l628_628904


namespace solution_set_of_inequality_l628_628068

def f (x : ℝ) : ℝ := if x > 0 then 2^x - 4 else - (2^(-x) - 4)

theorem solution_set_of_inequality (x : ℝ) :
  x * f(x + 1) < 0 ↔ (0 < x ∧ x < 1) ∨ (-3 < x ∧ x < -1) :=
by
  sorry

end solution_set_of_inequality_l628_628068


namespace square_of_cube_of_third_smallest_prime_l628_628671

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628671


namespace regular_polygon_sides_l628_628053

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628053


namespace root_count_eq_20_l628_628015

def f : ℝ → ℝ := λ x, |x| - 1

def f_iterate : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := λ x, f (f_iterate n x)

theorem root_count_eq_20 : (finset.univ.filter (λ x, f_iterate 10 x + (1/2) = 0)).card = 20 :=
by sorry

end root_count_eq_20_l628_628015


namespace smallest_m_exists_l628_628110

theorem smallest_m_exists (n : ℕ) (h : n ≥ 2) : 
  ∃ w : Fin (2 * n + 1) → ℝ, ∀ a : Fin (n - 1) → ℕ,
  (∀ i : Fin (n - 1), a i = 1 ∨ a i = -1) → 
  ∃ s : Fin n → ℝ, ∀ i : Fin (n - 1), 
  (a i = 1 → s (i + 1) > s i) ∧ 
  (a i = -1 → s (i + 1) < s i) → 
  ∃ subsequence : Fin (n - 1) → Fin (2 * n + 1), 
  ∀ i j : Fin n, i < j → subsequence i < subsequence j ∧
  (a i = 1 → w (subsequence j) > w (subsequence i)) ∧
  (a i = -1 → w (subsequence j) < w (subsequence i)) := sorry

end smallest_m_exists_l628_628110


namespace calculate_lower_profit_percentage_l628_628353

theorem calculate_lower_profit_percentage 
  (CP : ℕ) 
  (profitAt18Percent : ℕ) 
  (additionalProfit : ℕ)
  (hCP : CP = 800) 
  (hProfitAt18Percent : profitAt18Percent = 144) 
  (hAdditionalProfit : additionalProfit = 72) 
  (hProfitRelation : profitAt18Percent = additionalProfit + ((9 * CP) / 100)) :
  9 = ((9 * CP) / 100) :=
by
  sorry

end calculate_lower_profit_percentage_l628_628353


namespace roots_cubic_sum_of_cubes_l628_628506

theorem roots_cubic_sum_of_cubes (r s p q : ℝ) (h_roots : ∀ x, (x^3 - p * x^2 + q * x - p = 0) → (x = r ∨ x = s)) :
  r + s = p ∧ r * s = p → r^3 + s^3 = p^3 :=
by
  intros h_eq
  cases h_eq with h_sum h_prod
  have h1 : r^3 + s^3 = (r + s) * (r^2 - r * s + s^2) + 3 * r * s * (r + s),
  calc
    (r + s) * ((r + s)^2 - 3 * r * s) + 3 * r * s * (r + s)
    _ = p * (p^2 - 3 * p) + 3 * p^2
    _ = p^3 - 3 * p^2 + 3 * p^2
    _ = p^3
  sorry

end roots_cubic_sum_of_cubes_l628_628506


namespace inverse_square_variation_l628_628706

variable (x y : ℝ)

theorem inverse_square_variation (h1 : x = 1) (h2 : y = 3) (h3 : y = 2) : x = 2.25 :=
by
  sorry

end inverse_square_variation_l628_628706


namespace product_of_invertibles_mod_120_l628_628124

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628124


namespace twice_total_credits_l628_628356

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end twice_total_credits_l628_628356


namespace integer_mod_condition_l628_628429

theorem integer_mod_condition
  (n : ℕ)
  (x : ℕ → ℝ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (i : ℝ) ≤ x i ∧ x i ≤ 2 * i)
  (h2 : (∑ i in finset.range n, (x (i + 1))^2) / (∑ i in finset.range n, (i + 1) * x (i + 1))^2 = 27 / (4 * n * (n + 1) * (2 * n + 1))) :
  n % 9 = 0 ∨ n % 9 = 4 ∨ n % 9 = 8 :=
sorry

end integer_mod_condition_l628_628429


namespace multiplicatively_perfect_count_2_to_30_l628_628728

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

def multiplicatively_perfect (n : ℕ) : Prop :=
  n > 1 ∧ List.prod (proper_divisors n) = n

theorem multiplicatively_perfect_count_2_to_30 : 
  (List.filter multiplicatively_perfect [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]).length = 9 :=
by
  sorry

end multiplicatively_perfect_count_2_to_30_l628_628728


namespace obtuse_triangle_k_values_l628_628636

theorem obtuse_triangle_k_values :
  let valid_k_values := 
    List.range' 5 6 ++         -- Range [5, 6, 7, 8, 9, 10]
    List.range' 22 8 in        -- Range [22, 23, 24, 25, 26, 27, 28, 29]
  valid_k_values.count = 14 :=
by
  let values_with_17_as_max := List.range' 5 6   -- [5, 6, 7, 8, 9, 10]
  let values_with_k_as_max := List.range' 22 8  -- [22, 23, 24, 25, 26, 27, 28, 29]
  let all_valid_values := values_with_17_as_max ++ values_with_k_as_max
  have count_values : all_valid_values.count = 14,
    from sorry
  exact count_values

end obtuse_triangle_k_values_l628_628636


namespace sovereign_states_upper_bound_l628_628230

theorem sovereign_states_upper_bound (n : ℕ) (k : ℕ) : 
  (∃ (lines : ℕ) (border_stop_moving : Prop) (countries_disappear : Prop)
     (create_un : Prop) (total_countries : ℕ),
        (lines = n)
        ∧ (border_stop_moving = true)
        ∧ (countries_disappear = true)
        ∧ (create_un = true)
        ∧ (total_countries = k)) 
  → k ≤ (n^3 + 5*n) / 6 + 1 := 
sorry

end sovereign_states_upper_bound_l628_628230


namespace non_congruent_triangles_perimeter_18_l628_628010

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628010


namespace AB_perp_CM_l628_628531

-- Definitions of the points and lines in the triangle geometry setup
variables {A B C F P Q M : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited F] [Inhabited P] [Inhabited Q] [Inhabited M]

-- Assume A, B, and C form an acute-angled triangle
variable (triangle_ABC : ∀ (A B C : Type), Prop)

-- CF is the angle bisector of ∠ACB intersecting AB at F
variable (angle_bisector_CF : ∀ (A B C F : Type), Prop)

-- Perpendiculars dropped from F to sides BC and CA intersecting at P and Q respectively
variable (perpendicular_FP_BC : ∀ (F P B C : Type), Prop)
variable (perpendicular_FQ_CA : ∀ (F Q C A : Type), Prop)

-- M is the intersection point of AP and BQ
variable (intersection_M : ∀ (A P B Q M : Type), Prop)

-- The goal is to prove AB is perpendicular to CM
theorem AB_perp_CM (h1 : triangle_ABC A B C)
                   (h2 : angle_bisector_CF A B C F)
                   (h3 : perpendicular_FP_BC F P B C)
                   (h4 : perpendicular_FQ_CA F Q C A)
                   (h5 : intersection_M A P B Q M) :
  ∃ T, AB ⊥ CM := sorry

end AB_perp_CM_l628_628531


namespace sides_of_regular_polygon_with_20_diagonals_l628_628033

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628033


namespace polyhedron_volume_eq_l628_628590

-- Definitions related to the polyhedron and the inscribed sphere.
variables (T : Type) [polyhedron T]
variables (r : ℝ) (S : ℝ) 

-- Assumptions that a sphere is inscribed in the polyhedron
axiom inscribed_sphere (T : Type) [polyhedron T] (r : ℝ) (S : ℝ) : 
  ∃ (O : Type) (is_center : is_center O), 
  (forall (face : T), distance (O, face) = r)  ∧ 
  (sum (faces T)) = S

-- Proposition stating the volume of the polyhedron
theorem polyhedron_volume_eq (T : Type) [polyhedron T] (r : ℝ) (S : ℝ) (V_T : ℝ) 
  (h : inscribed_sphere T r S):
  V_T = (1 / 3) * r * S :=
sorry

end polyhedron_volume_eq_l628_628590


namespace part1_part2_part3_l628_628467

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := Real.exp x - e * x

def g (a x : ℝ) : ℝ := 2 * a * x + a

theorem part1 : ∀ x : ℝ, f x ≥ 0 := sorry

theorem part2 (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ = g a x₀) → (a ≥ 0 ∨ a < -e / 2) := sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, x < -1 → f x ≥ g a x) →
  a ≥ -e / 2 := sorry

end part1_part2_part3_l628_628467


namespace white_black_arrangements_l628_628801

theorem white_black_arrangements (W B : ℕ) (hW : W = 5) (hB : B = 10) :
  ∃ n, n = (nat.choose (B) (W)) ∧ n = 252 :=
by
  sorry

end white_black_arrangements_l628_628801


namespace line_through_fixed_point_l628_628502

-- Define the arithmetic sequence condition
def arithmetic_sequence (k b : ℝ) : Prop :=
  k + b = -2

-- Define the line passing through a fixed point
def line_passes_through (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ (x = 1 ∧ y = -2)

-- The theorem stating the main problem
theorem line_through_fixed_point (k b : ℝ) (h : arithmetic_sequence k b) : line_passes_through k b :=
  sorry

end line_through_fixed_point_l628_628502


namespace exists_equilateral_triangle_on_parallel_lines_l628_628769

theorem exists_equilateral_triangle_on_parallel_lines
  (a b c : set (ℝ × ℝ))
  (ha : ∀ (x : ℝ), ∃ y : ℝ, (x, y) ∈ a)
  (hb : ∀ (x : ℝ), ∃ y : ℝ, (x, y) ∈ b)
  (hc : ∀ (x : ℝ), ∃ y : ℝ, (x, y) ∈ c)
  (h_parallel : ∀ (p1 p2 : ℝ × ℝ), p1 ∈ a → p2 ∈ c → p1.2 < p2.2 ↔ p1 ∈ b) :
  ∃ A B C : ℝ × ℝ, 
    A ∈ b ∧ B ∈ a ∧ C ∈ c ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B :=
by
  sorry

end exists_equilateral_triangle_on_parallel_lines_l628_628769


namespace claudia_has_three_25_cent_coins_l628_628374

def number_of_coins (x y z : ℕ) := x + y + z = 15
def number_of_combinations (x y : ℕ) := 4 * x + 3 * y = 51

theorem claudia_has_three_25_cent_coins (x y z : ℕ) 
  (h1: number_of_coins x y z) 
  (h2: number_of_combinations x y): 
  z = 3 := 
by 
sorry

end claudia_has_three_25_cent_coins_l628_628374


namespace no_natural_number_divisible_by_1998_with_sum_of_digits_less_than_27_l628_628931

open Nat

-- Definition of the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Main problem statement
theorem no_natural_number_divisible_by_1998_with_sum_of_digits_less_than_27 :
  ∀ n : ℕ, (1998 ∣ n) → sum_of_digits n < 27 → false :=
by
  sorry

end no_natural_number_divisible_by_1998_with_sum_of_digits_less_than_27_l628_628931


namespace average_of_solutions_l628_628445

theorem average_of_solutions (a b c : ℝ) (h_eq : a = 3 ∧ b = 4 ∧ c = -5) :
  let sum := -b / a in
  let avg := sum / 2 in
  avg = -2 / 3 := 
by
  cases h_eq with ha hbc
  cases hbc with hb hc
  have sum := -b / a
  have avg := sum / 2
  rw [ha, hb, hc] at *
  simp at *
  sorry

end average_of_solutions_l628_628445


namespace prod_coprime_mod_l628_628174

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628174


namespace product_invertibles_mod_120_l628_628162

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628162


namespace average_velocity_l628_628541

noncomputable def h (t : ℝ) : ℝ := t^2 + 1

theorem average_velocity (Δt : ℝ) (hpos : 0 < Δt) : 
  let v := (h(1 + Δt) - h(1)) / Δt in v = 2 + Δt :=
by
  sorry

end average_velocity_l628_628541


namespace problem_solution_l628_628450

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x_0 : ℝ, x_0^2 + (a-1)*x_0 + 1 < 0

theorem problem_solution (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) :
  -1 ≤ a ∧ a ≤ 1 ∨ a > 3 :=
sorry

end problem_solution_l628_628450


namespace cos_product_identity_l628_628588

theorem cos_product_identity (n : ℕ) (hn : 0 < n) :
  let α := (real.pi : ℝ) / (1 + 2^n) in
  (∏ i in finset.range n, real.cos (2^i * α)) = 1 / (2^n) :=
by
  let α := (real.pi : ℝ) / (1 + 2^n)
  sorry

end cos_product_identity_l628_628588


namespace problem_solution_l628_628498

theorem problem_solution
  (x : ℝ)
  (h : 4 - x^2 ≤ 0) : x ≤ -2 ∨ x ≥ 2 :=
begin
  sorry
end

end problem_solution_l628_628498


namespace fractional_part_near_integer_l628_628114

theorem fractional_part_near_integer (a : ℝ) (n : ℕ) (h_pos : 0 < a) (h_n : 1 < n) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ |(k * a - ⌊k * a⌋) - (⌊k * a⌋ + 1) | ≤ 1 / n :=
sorry

end fractional_part_near_integer_l628_628114


namespace polygon_sides_from_diagonals_l628_628027

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628027


namespace bounded_area_eq_e_l628_628757

theorem bounded_area_eq_e :
  let f (x : ℝ) := x ^ (1 / Real.log x) in
  ∃ e : ℝ, e = Real.exp 1 ∧
  ((∫ x in 2..3, f x) - (∫ x in 2..3, (0 : ℝ))) = e :=
by
  let f (x : ℝ) := x ^ (1 / Real.log x)
  have h1 : (∫ x in 2..3, f x) = Real.exp 1 * 1 := sorry
  have h2 : (∫ x in 2..3, (0 : ℝ)) = 0 := sorry
  refine ⟨Real.exp 1, rfl, _⟩
  rw [h1, h2, sub_zero]
  exact Real.mul_one (Real.exp 1)

end bounded_area_eq_e_l628_628757


namespace exists_prime_and_positive_integer_l628_628776

theorem exists_prime_and_positive_integer (a : ℕ) (h : a = 9) : 
  ∃ (p : ℕ) (hp : Nat.Prime p) (b : ℕ) (hb : b ≥ 2), (a^p - a) / p = b^2 := 
  by
  sorry

end exists_prime_and_positive_integer_l628_628776


namespace egg_weight_probability_l628_628809

theorem egg_weight_probability : 
  let P_lt_30 := 0.3
  let P_30_40 := 0.5
  P_lt_30 + P_30_40 ≤ 1 → (1 - (P_lt_30 + P_30_40) = 0.2) := by
  intro h
  sorry

end egg_weight_probability_l628_628809


namespace words_are_not_synonyms_l628_628095

-- Define the difference in number of letters M and O
def diff (word : String) : Int :=
  word.toList.count (· = 'M') - word.toList.count (· = 'O')

theorem words_are_not_synonyms : diff "OMM" ≠ diff "MOO" :=
by
  -- Calculate differences and show they are not equal
  have h1: diff "OMM" = 1 := by sorry
  have h2: diff "MOO" = -1 := by sorry
  show 1 ≠ -1 from Nat.one_ne_neg_one
  exact sorry -- Placeholder for actual proof verification

end words_are_not_synonyms_l628_628095


namespace a_2019_eq_l628_628811

noncomputable def a : ℕ → ℝ
| 1       := real.sqrt 3
| (n + 1) := let an := a n in 
             real.floor an + 1 / (an - real.floor an)

theorem a_2019_eq : a 2019 = 3027 + real.sqrt 3 := 
by
  sorry

end a_2019_eq_l628_628811


namespace eval_f_g_at_4_l628_628566

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem eval_f_g_at_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  sorry

end eval_f_g_at_4_l628_628566


namespace ladder_width_l628_628725

theorem ladder_width (l x : ℝ) (h1 : cos (real.pi / 3) = 1/2) (h2 : cos (real.pi / 4) = real.sqrt 2 / 2)
  (h3 : 2 * x = y) (h4 : y = 2 * x) : w = l :=
by
  sorry

end ladder_width_l628_628725


namespace sam_investment_interest_rate_l628_628981

theorem sam_investment_interest_rate :
  ∃ r : ℝ, 
  let A1 := 10000 * (1 + r)^3 in
  let A2 := 3 * A1 * 1.15 in
  A2 = 59616 ∧ r = 0.2 :=
by {
  use 0.2,
  let A1 := 10000 * (1 + 0.2)^3,
  let A2 := 3 * A1 * 1.15,
  have h1 : A1 = 10000 * 1.728,
  { sorry },
  have h2 : A2 = 3 * (10000 * 1.728) * 1.15,
  { sorry },
  have h3 : A2 = 59616,
  { sorry },
  exact ⟨rfl, rfl⟩
}

end sam_investment_interest_rate_l628_628981


namespace range_of_m_l628_628885

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : 0 < x) (hy : 0 < y) 
                    (h : (2 * y - x / exp 1) * (log x - log y) - y / m ≤ 0) : 
  0 < m ∧ m ≤ 1 := 
sorry

end range_of_m_l628_628885


namespace largest_minimal_arc_length_l628_628322

variable {α : Type}
variables (a : Fin 7 → ℝ)
variable (A : ℝ)

-- Sum of the seven arcs must be 360 degrees
axiom H1 : ∑ i, a i = 360

-- Sum of any two adjacent arcs does not exceed 103 degrees
axiom H2 : ∀ i : Fin 7, a i + a ((i + 1) % 7) ≤ 103

-- Each of the seven arcs is at least A degrees
def valid_arcs : Prop :=
  ∀ i : Fin 7, a i ≥ A

-- Claim: The largest possible value of A is 51 degrees
theorem largest_minimal_arc_length : A = 51 :=
sorry

end largest_minimal_arc_length_l628_628322


namespace total_pastries_sum_l628_628576

   theorem total_pastries_sum :
     let lola_mini_cupcakes := 13
     let lola_pop_tarts := 10
     let lola_blueberry_pies := 8
     let lola_chocolate_eclairs := 6

     let lulu_mini_cupcakes := 16
     let lulu_pop_tarts := 12
     let lulu_blueberry_pies := 14
     let lulu_chocolate_eclairs := 9

     let lila_mini_cupcakes := 22
     let lila_pop_tarts := 15
     let lila_blueberry_pies := 10
     let lila_chocolate_eclairs := 12

     lola_mini_cupcakes + lulu_mini_cupcakes + lila_mini_cupcakes +
     lola_pop_tarts + lulu_pop_tarts + lila_pop_tarts +
     lola_blueberry_pies + lulu_blueberry_pies + lila_blueberry_pies +
     lola_chocolate_eclairs + lulu_chocolate_eclairs + lila_chocolate_eclairs = 147 :=
   by
     sorry
   
end total_pastries_sum_l628_628576


namespace collinear_points_in_cyclic_quadrilateral_l628_628193

theorem collinear_points_in_cyclic_quadrilateral
  (ω : Circle)
  (A B C D P Q R E : Point)
  (h_circle : Circle.ContainingPoints ω [A, B, C, D])
  (h_P_on_AC_extension : PointOnLineExtension P C A)
  (h_PB_tangent : TangentToCircle P B ω)
  (h_PD_tangent : TangentToCircle P D ω)
  (h_C_tangent_intersects_PD_at_Q : TangentIntersectionAt C P D Q ω)
  (h_C_tangent_intersects_AD_at_R : TangentIntersectionAt C P D R AD)
  (h_Q_second_intersection_AQ : SecondIntersection A Q ω E) :
  Collinear [B, E, R] :=
sorry

end collinear_points_in_cyclic_quadrilateral_l628_628193


namespace find_b_for_integer_a_l628_628563

theorem find_b_for_integer_a (a : ℤ) (b : ℝ) (h1 : 0 ≤ b) (h2 : b < 1) (h3 : (a:ℝ)^2 = 2 * b * (a + b)) :
  b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 :=
sorry

end find_b_for_integer_a_l628_628563


namespace cost_of_tree_planting_l628_628983

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l628_628983


namespace no_real_roots_A_l628_628694

-- Definitions for each quadratic equation
def A : ℝ → ℝ := λ x => x^2 + x + 3
def B : ℝ → ℝ := λ x => x^2 + 2*x + 1
def C : ℝ → ℝ := λ x => x^2 - 2
def D : ℝ → ℝ := λ x => x^2 - 2*x - 3

-- Definitions for the discriminants of each equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Proving that equation A has no real roots
theorem no_real_roots_A : discriminant 1 1 3 < 0 :=
by
  show (1:ℝ)^2 - 4*1*3 < 0
  rfl

end no_real_roots_A_l628_628694


namespace zayne_total_revenue_l628_628299

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l628_628299


namespace angle_between_vectors_l628_628483

variable {V : Type*} [InnerProductSpace ℝ V]

variables (a b : V)
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 3) (h : (a - b) ⬝ a = 7)

theorem angle_between_vectors : ∠ a b = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l628_628483


namespace kara_uses_28_cups_of_sugar_l628_628555

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l628_628555


namespace sum_of_A_elements_l628_628623

def floor (x : ℝ) : ℤ := Int.floor x

def g (x : ℝ) : ℤ := floor x + floor (2 * x)

def A : Set ℤ := {y | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ y = g x}

theorem sum_of_A_elements : (0 + 1 + 3) ∈ A ∧ (∀ y ∈ A, y = 0 ∨ y = 1 ∨ y = 3) → ∑ (y ∈ A), y = 4 := 
by 
  sorry

end sum_of_A_elements_l628_628623


namespace probability_avg_greater_than_5_l628_628074

theorem probability_avg_greater_than_5 :
  ∑ s in (finset.powerset (finset.range 5)).filter (λ s, s.card = 2) ,
    ∑ t in (finset.range 5).filter (λ t, t ∉ s), 
      (t / 3 > 5) = (2 / 5) :=
begin 
  sorry,
end

end probability_avg_greater_than_5_l628_628074


namespace line_tangent_to_circle_chord_length_l628_628823

noncomputable def circle_eq := ∀ (x y : ℝ), x^2 + y^2 + 2*x - 2*y + 1 = 0

noncomputable def line_eq (b : ℝ) := ∀ (x y : ℝ), y = x + b

theorem line_tangent_to_circle (b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq b x y) → b = 2 + Real.sqrt 2 ∨ b = 2 - Real.sqrt 2 :=
by sorry

theorem chord_length (b : ℝ) :
  (b = 1) → (∃ x₁ x₂ y₁ y₂ : ℝ, line_eq b x₁ y₁ ∧ line_eq b x₂ y₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ x₁ ≠ x₂) →
  Real.sqrt ((1 + Real.sqrt 2) - (1 - Real.sqrt 2)) = Real.sqrt 2 :=
by sorry

end line_tangent_to_circle_chord_length_l628_628823


namespace car_trip_problem_l628_628965

theorem car_trip_problem (a b c : ℕ) (x : ℕ) 
(h1 : 1 ≤ a) 
(h2 : a + b + c ≤ 9)
(h3 : 100 * b + 10 * c + a - 100 * a - 10 * b - c = 60 * x) 
: a^2 + b^2 + c^2 = 14 := 
by
  sorry

end car_trip_problem_l628_628965


namespace square_of_cube_of_third_smallest_prime_l628_628679

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l628_628679


namespace product_of_invertibles_mod_120_l628_628134

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628134


namespace area_enclosed_by_line_and_curve_l628_628605

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..2, (4 * x - x ^ 3)

theorem area_enclosed_by_line_and_curve :
  enclosed_area = 4 :=
by
  sorry

end area_enclosed_by_line_and_curve_l628_628605


namespace complete_square_formula_D_l628_628294

-- Definitions of polynomial multiplications
def poly_A (a b : ℝ) : ℝ := (a - b) * (a + b)
def poly_B (a b : ℝ) : ℝ := -((a + b) * (b - a))
def poly_C (a b : ℝ) : ℝ := (a + b) * (b - a)
def poly_D (a b : ℝ) : ℝ := (a - b) * (b - a)

theorem complete_square_formula_D (a b : ℝ) : 
  poly_D a b = -(a - b)*(a - b) :=
by sorry

end complete_square_formula_D_l628_628294


namespace integral_3x_plus_sin_x_l628_628758

theorem integral_3x_plus_sin_x :
  ∫ x in (0 : ℝ)..(π / 2), (3 * x + Real.sin x) = (3 / 8) * π^2 + 1 :=
by
  sorry

end integral_3x_plus_sin_x_l628_628758


namespace compare_Sn_Tn_l628_628380

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (k + 1) / ((2 * n - 2 * (k + 1) + 1) * (2 * n - (k + 1) + 1))

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 1 / (k + 1)

theorem compare_Sn_Tn (n : ℕ) : T_n n > S_n n := 
  sorry

end compare_Sn_Tn_l628_628380


namespace sum_of_digits_l628_628064

variables {a b c d : ℕ}

theorem sum_of_digits (h1 : ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
                      (h2 : c + a = 10)
                      (h3 : b + c = 9)
                      (h4 : a + d = 10) :
  a + b + c + d = 18 :=
sorry

end sum_of_digits_l628_628064


namespace quadratic_distinct_zeros_range_l628_628069

theorem quadratic_distinct_zeros_range (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - (k+1)*x1 + k + 4 = 0 ∧ x2^2 - (k+1)*x2 + k + 4 = 0)
  ↔ k ∈ (Set.Iio (-3) ∪ Set.Ioi 5) :=
by
  sorry

end quadratic_distinct_zeros_range_l628_628069


namespace count_inequalities_l628_628241

def expr1 : Prop := 3 < 5
def expr2 (x : ℝ) : Prop := x > 0
def expr3 (x : ℝ) : Prop := 2 * x ≠ 3
def expr4 (a : ℝ) : Prop := a = 3
def expr5 (a : ℝ) : ℝ := 2 * a + 1
def expr6 (x : ℝ) : Prop := (1 - x) / 5 > 1

theorem count_inequalities : ∀ x a : ℝ, 4 = (if expr1 then 1 else 0) + (if expr2 x then 1 else 0) + (if expr3 x then 1 else 0) + (if expr6 x then 1 else 0) := by
  intros x a
  simp [expr1, expr2, expr3, expr4, expr5, expr6]
  sorry

end count_inequalities_l628_628241


namespace square_of_cube_of_third_smallest_prime_l628_628687

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l628_628687


namespace proof_problem_l628_628880

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

def condition_1 := ∥a∥ = 1
def condition_2 := ∥b∥ = 1
def condition_3 := ∥a + b∥ = real.sqrt 3
def condition_4 := c = 2 • a + 3 • b + 4 • (a ×ₗ b)

theorem proof_problem (h1 : condition_1 a)
                      (h2 : condition_2 b)
                      (h3 : condition_3 a b)
                      (h4 : condition_4 a b c) :
  inner_product_space.inner a c = 7 / 2 := sorry

end proof_problem_l628_628880


namespace largest_integer_value_neg_quadratic_l628_628414

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end largest_integer_value_neg_quadratic_l628_628414


namespace prod_coprime_mod_l628_628181

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628181


namespace symmetry_of_transformed_graphs_l628_628860

variable (f : ℝ → ℝ)

theorem symmetry_of_transformed_graphs :
  (∀ x, f x = f (-x)) → (∀ x, f (1 + x) = f (1 - x)) :=
by
  intro h_symmetry
  intro x
  sorry

end symmetry_of_transformed_graphs_l628_628860


namespace jim_cousin_money_l628_628937

theorem jim_cousin_money (jim_money : ℕ) (cheeseburger_cost : ℕ) (cheeseburgers_ordered : ℕ) 
    (milkshake_cost : ℕ) (milkshakes_ordered : ℕ) (cheesefries_cost : ℕ) 
    (spent_percentage : ℚ) (spent_money : ℕ) :
  jim_money = 20 →
  cheeseburger_cost = 3 →
  cheeseburgers_ordered = 2 →
  milkshake_cost = 5 →
  milkshakes_ordered = 2 →
  cheesefries_cost = 8 →
  spent_percentage = 0.8 →
  spent_money = 24 →
  jim_money + (spent_money / spent_percentage) - jim_money = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  sorry

end jim_cousin_money_l628_628937


namespace marbles_per_boy_l628_628079

theorem marbles_per_boy (total_marbles boys : ℕ) (h_total : total_marbles = 80) (h_boys : boys = 8) :
  total_marbles / boys = 10 :=
by
  rw [h_total, h_boys]
  norm_num

end marbles_per_boy_l628_628079


namespace find_triplets_l628_628789

theorem find_triplets (m n k : ℕ) (pos_m : 0 < m) (pos_n : 0 < n) (pos_k : 0 < k) : 
  (k^m ∣ m^n - 1) ∧ (k^n ∣ n^m - 1) ↔ (k = 1) ∨ (m = 1 ∧ n = 1) :=
by
  sorry

end find_triplets_l628_628789


namespace sequence_and_sum_l628_628575

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sequence_and_sum
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_first_n_terms a S)
  (cond : a 2 + a 8 = 15 - a 5) :
  S 9 = 45 :=
sorry

end sequence_and_sum_l628_628575


namespace train_crosses_pole_in_2point4_seconds_l628_628543

noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (speed_kmh * (5/18))

theorem train_crosses_pole_in_2point4_seconds :
  time_to_cross 120 180 = 2.4 := by
  sorry

end train_crosses_pole_in_2point4_seconds_l628_628543


namespace prod_coprime_mod_l628_628175

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628175


namespace constant_and_middle_term_l628_628385

open scoped BigOperators

-- Define the expression
def expr (x : ℚ) : ℚ := (2 * x^2 - 1 / x)

-- Constants for the binomial expansion
noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.choose n k

-- The 6th power expansion of the given expression
def expand_expr := ∑ r in finset.range 7, (-1)^r * (2^(6-r)) * binomial_coeff 6 r * (λ x : ℚ, x^(12-3*r))

/-- 
    Prove that the constant term in the binomial expansion of (2x^2 - 1/x)^6 is 60
    and the middle term is -160x^3.
-/
theorem constant_and_middle_term :
    ∃ (x : ℚ), (expr x)^6 = 60 ∧ (expr x)^3 = -160 * x^3 :=
by
  sorry

end constant_and_middle_term_l628_628385


namespace number_of_n_not_divisible_by_4_l628_628805

theorem number_of_n_not_divisible_by_4 : 
  (∑ n in finset.range 1000, if (⌊996 / n⌋ + ⌊997 / n⌋ + ⌊998 / n⌋) % 4 ≠ 0 then 1 else 0) = 20 := 
by sorry

end number_of_n_not_divisible_by_4_l628_628805


namespace non_congruent_triangles_perimeter_18_l628_628008

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628008


namespace division_problem_l628_628719

theorem division_problem (n : ℕ) (h : n / 6 = 209) : n = 1254 := 
sorry

end division_problem_l628_628719


namespace p_sufficient_not_necessary_for_q_l628_628564

variables {x : ℝ}

def p : Prop := x^2 - x - 20 > 0
def q : Prop := |x| - 2 > 0

theorem p_sufficient_not_necessary_for_q : (p → q) ∧ ¬(q → p) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l628_628564


namespace div_gcd_iff_div_ab_gcd_mul_l628_628948

variable (a b n c : ℕ)
variables (h₀ : a ≠ 0) (d : ℕ)
variable (hd : d = Nat.gcd a b)

theorem div_gcd_iff_div_ab : (n ∣ a ∧ n ∣ b) ↔ n ∣ d :=
by
  sorry

theorem gcd_mul (h₁ : c > 0) : Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by
  sorry

end div_gcd_iff_div_ab_gcd_mul_l628_628948


namespace square_of_cube_of_third_smallest_prime_l628_628677

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628677


namespace length_of_plot_l628_628246

-- We introduce the definitions based on the conditions
def breadth (b : ℝ) := b
def length (b : ℝ) := b + 26
def cost_per_meter := 26.50
def total_cost := 5300
def perimeter (b : ℝ) := 2 * (b + 26) + 2 * b

-- Lean proof problem statement
theorem length_of_plot (b : ℝ) (P : ℝ) (l := length b) 
    (h1 : P = 2 * l + 2 * breadth b)
    (h2 : cost_per_meter * P = total_cost) : l = 63 := 
    by
    sorry

end length_of_plot_l628_628246


namespace vertical_angles_always_equal_l628_628296

theorem vertical_angles_always_equal (a b : ℝ) (h : a = b) : 
  (∀ θ1 θ2, θ1 + θ2 = 180 ∧ θ1 = a ∧ θ2 = b → θ1 = θ2) :=
by 
  intro θ1 θ2 
  intro h 
  sorry

end vertical_angles_always_equal_l628_628296


namespace vector_dot_product_midpoint_l628_628518

open EuclideanGeometry

theorem vector_dot_product_midpoint 
  {A B C M P : Point} 
  (h_midpoint : Midpoint M B C) 
  (h_AM : dist A M = 3) 
  (h_AP_PM : vector_represent A P = 2 • vector_represent P M) : 
  vector_represent P A • (vector_represent P B + vector_represent P C) = -4 := 
sorry

end vector_dot_product_midpoint_l628_628518


namespace product_invertibles_mod_120_l628_628161

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628161


namespace percentage_of_burpees_is_10_l628_628394

-- Definitions for each exercise count
def jumping_jacks : ℕ := 25
def pushups : ℕ := 15
def situps : ℕ := 30
def burpees : ℕ := 10
def lunges : ℕ := 20

-- Total number of exercises
def total_exercises : ℕ := jumping_jacks + pushups + situps + burpees + lunges

-- The proof statement
theorem percentage_of_burpees_is_10 :
  (burpees * 100) / total_exercises = 10 :=
by
  sorry

end percentage_of_burpees_is_10_l628_628394


namespace probability_of_x_gt_2y_l628_628978

noncomputable def probability_x_gt_2y : ℚ :=
  let area_triangle := (2008 * 2008) / 4
  let area_rectangle := 2008 * 2009
  in area_triangle / area_rectangle

theorem probability_of_x_gt_2y :
  probability_x_gt_2y = 502 / 2009 :=
by
  sorry

end probability_of_x_gt_2y_l628_628978


namespace common_tangent_sum_eq_one_l628_628887

theorem common_tangent_sum_eq_one (a b m : ℝ) 
  (h1 : ∀ (x : ℝ), a * Real.cos x = x ^ 2 + b * x + 1) 
  (h2 : ∀ (x : ℝ), -a * Real.sin x = 2 * x + b) 
  (h3 : f(0) = g(0))
  (h4 : f'(0) = g'(0)) :
  a + b = 1 :=
by 
  sorry

end common_tangent_sum_eq_one_l628_628887


namespace product_sum_abcd_e_l628_628283

-- Define the individual numbers
def a : ℕ := 12
def b : ℕ := 25
def c : ℕ := 52
def d : ℕ := 21
def e : ℕ := 32

-- Define the sum of the numbers a, b, c, and d
def sum_abcd : ℕ := a + b + c + d

-- Prove that multiplying the sum by e equals 3520
theorem product_sum_abcd_e : sum_abcd * e = 3520 := by
  sorry

end product_sum_abcd_e_l628_628283


namespace inclination_angle_l628_628879

theorem inclination_angle (α : ℝ) (h : -real.pi / 2 < α ∧ α < 0) :
  let slope := Real.tan α in
  ∃ θ : ℝ, Real.tan θ = slope ∧ θ = real.pi + α :=
by
  sorry

end inclination_angle_l628_628879


namespace product_of_invertibles_mod_120_l628_628165

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628165


namespace spent_on_books_l628_628876

theorem spent_on_books (allowance games_fraction snacks_fraction toys_fraction : ℝ)
  (h_allowance : allowance = 50)
  (h_games : games_fraction = 1/4)
  (h_snacks : snacks_fraction = 1/5)
  (h_toys : toys_fraction = 2/5) :
  allowance - (allowance * games_fraction + allowance * snacks_fraction + allowance * toys_fraction) = 7.5 :=
by
  sorry

end spent_on_books_l628_628876


namespace alley_width_theorem_l628_628081

noncomputable def width_of_alley (a k h : ℝ) (h₁ : k = a / 2) (h₂ : h = a * (Real.sqrt 2) / 2) : ℝ :=
  Real.sqrt ((a * (Real.sqrt 2) / 2)^2 + (a / 2)^2)

theorem alley_width_theorem (a k h w : ℝ)
  (h₁ : k = a / 2)
  (h₂ : h = a * (Real.sqrt 2) / 2)
  (h₃ : w = width_of_alley a k h h₁ h₂) :
  w = (Real.sqrt 3) * a / 2 :=
by
  sorry

end alley_width_theorem_l628_628081


namespace product_coprime_mod_120_l628_628188

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628188


namespace product_of_invertible_integers_mod_120_l628_628137

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628137


namespace change_in_expression_l628_628076

variables (x b : ℝ) (hb : 0 < b)

theorem change_in_expression : (b * x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 :=
by sorry

end change_in_expression_l628_628076


namespace smallest_integer_y_l628_628662

theorem smallest_integer_y : ∃ y : ℤ, (8:ℚ) / 11 < y / 17 ∧ ∀ z : ℤ, ((8:ℚ) / 11 < z / 17 → y ≤ z) :=
by
  sorry

end smallest_integer_y_l628_628662


namespace arithmetic_sequence_sum_l628_628536

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a4 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end arithmetic_sequence_sum_l628_628536


namespace product_of_invertibles_mod_120_l628_628133

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628133


namespace find_f_neg_one_l628_628318

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * tan x + 3

theorem find_f_neg_one (a b : ℝ) (h : f a b 1 = 1) : f a b (-1) = 5 :=
by
  sorry

end find_f_neg_one_l628_628318


namespace quadrilateral_ABCD_AD_length_l628_628218

noncomputable def triangle (a b c d: ℝ) := a^2 + b^2 = c^2

theorem quadrilateral_ABCD_AD_length
  (AB BC CD : ℝ) (B C : ℝ)
  (H_AB : AB = 5)
  (H_BC : BC = 7)
  (H_CD : CD = 15)
  (H_B_obtuse : B > 90 ∧ B < 180)
  (H_C_obtuse : C > 90 ∧ C < 180)
  (H_sin_cos : sin C = -(cos B) ∧ sin C = 4/5) :
  ∃ AD, AD ≈ 27.16 :=
sorry

end quadrilateral_ABCD_AD_length_l628_628218


namespace non_congruent_triangles_with_perimeter_18_l628_628003

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l628_628003


namespace max_min_values_on_ellipse_l628_628499

theorem max_min_values_on_ellipse :
  (P : ℝ × ℝ) (h : P.1 ^ 2 / 4 + P.2 ^ 2 = 1) →
  ∃ x_max x_min : ℝ, 
  (x_max = sqrt 5 ∧ x_min = -sqrt 5) ∧
  ∀ (x y : ℝ), (x, y) = P → y = x → x = x_max ∨ x = x_min :=
by
  sorry

end max_min_values_on_ellipse_l628_628499


namespace total_potatoes_sold_is_322kg_l628_628337

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end total_potatoes_sold_is_322kg_l628_628337


namespace transformed_graph_passes_point_l628_628514

theorem transformed_graph_passes_point (f : ℝ → ℝ) 
  (h₁ : f 1 = 3) :
  f (-1) + 1 = 4 :=
by
  sorry

end transformed_graph_passes_point_l628_628514


namespace complex_power_identity_l628_628369

theorem complex_power_identity (z : ℂ) (i : ℂ) 
  (h1 : z = (1 + i) / Real.sqrt 2) 
  (h2 : z^2 = i) : 
  z^100 = -1 := 
  sorry

end complex_power_identity_l628_628369


namespace product_coprime_mod_120_l628_628185

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628185


namespace product_of_invertibles_mod_120_l628_628132

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628132


namespace total_potato_weight_l628_628336

theorem total_potato_weight (bags_morning : ℕ) (bags_afternoon : ℕ) (weight_per_bag : ℕ) :
  bags_morning = 29 → 
  bags_afternoon = 17 → 
  weight_per_bag = 7 → 
  (bags_morning + bags_afternoon) * weight_per_bag = 322 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num 
  sorry

end total_potato_weight_l628_628336


namespace sum_of_other_endpoint_coordinates_l628_628254

theorem sum_of_other_endpoint_coordinates :
  ∃ (x y: ℤ), (8 + x) / 2 = 6 ∧ y / 2 = -10 ∧ x + y = -16 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l628_628254


namespace FractionBoundsFibonacci_l628_628376

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem FractionBoundsFibonacci (a b n : ℕ) (h1 : 0 < n)
    (h2 : (a:ℚ)/(b:ℚ) < (fibonacci n:ℚ)/(fibonacci (n-1):ℚ) ∨ (a:ℚ)/(b:ℚ) < (fibonacci (n+1):ℚ)/(fibonacci n:ℚ))
    (h3 : (a:ℚ)/(b:ℚ) > (fibonacci n:ℚ)/(fibonacci (n-1):ℚ) ∨ (a:ℚ)/(b:ℚ) > (fibonacci (n+1):ℚ)/(fibonacci n:ℚ))
    (hab : 0 < a) (hb : 0 < b) :
    b ≥ fibonacci (n + 1) :=
begin
  sorry
end

end FractionBoundsFibonacci_l628_628376


namespace cost_of_each_item_l628_628646

theorem cost_of_each_item 
  (x y z : ℝ) 
  (h1 : 3 * x + 5 * y + z = 32)
  (h2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 :=
by 
  sorry

end cost_of_each_item_l628_628646


namespace total_travel_time_l628_628641

-- Define the conditions
def total_distance : ℝ := 200
def fraction_driven_before_lunch : ℝ := 1 / 4
def driving_time_before_lunch : ℝ := 1
def lunch_time : ℝ := 1

-- Calculate the distance driven before lunch
def distance_before_lunch := total_distance * fraction_driven_before_lunch

-- Calculate the speed
def speed := distance_before_lunch / driving_time_before_lunch

-- Calculate the remaining distance
def remaining_distance := total_distance - distance_before_lunch

-- Calculate the time to drive the remaining distance
def driving_time_after_lunch := remaining_distance / speed

-- The total time (driving before lunch + lunch time + driving after lunch)
def total_time := driving_time_before_lunch + lunch_time + driving_time_after_lunch

-- Prove that the total time taken is 5 hours
theorem total_travel_time : total_time = 5 := by
  -- Insert necessary reasoning here
  sorry

end total_travel_time_l628_628641


namespace annie_jacob_ratio_l628_628578

theorem annie_jacob_ratio :
  ∃ (a j : ℕ), ∃ (m : ℕ), (m = 2 * a) ∧ (j = 90) ∧ (m = 60) ∧ (a / j = 1 / 3) :=
by
  sorry

end annie_jacob_ratio_l628_628578


namespace toby_steps_on_tuesday_l628_628264

theorem toby_steps_on_tuesday:
  ∀ (total_steps_week sunday_steps monday_steps wed_steps thurs_steps steps_needed_fri_and_sat steps_remaining : ℕ),
    total_steps_week = 63000 →
    sunday_steps = 9400 →
    monday_steps = 9100 →
    wed_steps = 9200 →
    thurs_steps = 8900 →
    steps_needed_fri_and_sat = 18100 →
    steps_remaining = total_steps_week - (sunday_steps + monday_steps + wed_steps + thurs_steps + steps_needed_fri_and_sat) →
    steps_remaining = 8300 :=
begin
  -- proof steps here
  sorry
end

end toby_steps_on_tuesday_l628_628264


namespace smallest_debt_resolveable_in_cows_and_sheep_l628_628649

noncomputable def cow_value : ℕ := 500
noncomputable def sheep_value : ℕ := 350

theorem smallest_debt_resolveable_in_cows_and_sheep : ∃ (d : ℕ), d > 0 ∧ (∃ (c s : ℤ), d = cow_value * c + sheep_value * s) ∧ d = 50 :=
begin
  sorry
end

end smallest_debt_resolveable_in_cows_and_sheep_l628_628649


namespace original_triangle_area_l628_628619

theorem original_triangle_area (A_new : ℝ) (r : ℝ) (A_original : ℝ) 
  (h1 : r = 3) 
  (h2 : A_new = 54) 
  (h3 : A_new = r^2 * A_original) : 
  A_original = 6 := 
by 
  sorry

end original_triangle_area_l628_628619


namespace rent_percentage_l628_628703

-- Conditions
variables (X : ℝ) -- Elaine's earnings last year
def earnings_last_year := X
def earnings_on_rent_last_year := 0.20 * X
def earnings_this_year := 1.35 * X
def earnings_on_rent_this_year := 0.30 * earnings_this_year

-- Question (to prove)
theorem rent_percentage :
  (earnings_on_rent_this_year X) / (earnings_on_rent_last_year X) * 100 = 202.5 :=
by
  sorry

end rent_percentage_l628_628703


namespace carmen_pets_l628_628372

theorem carmen_pets (initial_cats initial_dogs cats_given_1 cats_given_2 cats_given_3 : ℕ) :
  initial_cats = 48 →
  initial_dogs = 36 →
  cats_given_1 = 6 →
  cats_given_2 = 12 →
  cats_given_3 = 8 →
  initial_cats - (cats_given_1 + cats_given_2 + cats_given_3) = 22 →
  initial_dogs = 36 →
  22 - initial_dogs = -14 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end carmen_pets_l628_628372


namespace monotonicity_f_when_a_half_range_of_b_l628_628855

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x + (1 - a) / x - 1
def g (x : ℝ) (b : ℝ) : ℝ := x ^ 2 - 2 * b * x + 4

theorem monotonicity_f_when_a_half : ∀ x : ℝ, 0 < x → f x (1 / 2) ≤ 0 := 
sorry

theorem range_of_b (b : ℝ) : 
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 2 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 (1 / 4) ≥ g x2 b) ↔ 
  b ∈ Ici (17 / 8) := 
sorry

end monotonicity_f_when_a_half_range_of_b_l628_628855


namespace factor_theorem_l628_628507

-- Define the polynomial function f(x)
def f (k : ℚ) (x : ℚ) : ℚ := k * x^3 + 27 * x^2 - k * x + 55

-- State the theorem to find the value of k such that x+5 is a factor of f(x)
theorem factor_theorem (k : ℚ) : f k (-5) = 0 ↔ k = 73 / 12 :=
by sorry

end factor_theorem_l628_628507


namespace find_a_of_equal_coeffs_l628_628538

theorem find_a_of_equal_coeffs (a : ℝ) (h : a ≠ 0)
  (h1 : (∃ c1 c2 : ℝ, (2 + a * x)^5 = c1 * x ∧ (2 + a * x)^5 = c2 * x^2 ∧ c1 = c2)): a = 1 :=
begin
  sorry
end

end find_a_of_equal_coeffs_l628_628538


namespace product_of_invertibles_mod_120_l628_628119

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628119


namespace range_of_a_l628_628821

def f (x : ℝ) : ℝ := x - sin x

theorem range_of_a 
  (a : ℝ)
  (h_domain_ineq1 : -1 < a - 2)
  (h_domain_ineq2 : a - 2 < 1)
  (h_domain_ineq3 : -1 < 4 - a^2)
  (h_domain_ineq4 : 4 - a^2 < 1)
  (h_ineq : f (a-2) + f (4 - a^2) < 0) :
  2 < a ∧ a < sqrt 5 :=
sorry

end range_of_a_l628_628821


namespace perpendicular_when_a_is_neg1_passes_through_origin_l628_628472

noncomputable theory

def line_l (a : ℝ) : (ℝ × ℝ) → Prop := λ p, (a^2 + a + 1) * p.1 - p.2 + 1 = 0
def line_x_plus_y : (ℝ × ℝ) → Prop := λ p, p.1 + p.2 = 0

def passes_through (l : (ℝ × ℝ) → Prop) (p : ℝ × ℝ) : Prop := l p

theorem perpendicular_when_a_is_neg1 
  (a : ℝ) (p : ℝ × ℝ) : 
  a = -1 → (∀ p, line_l a p → line_x_plus_y p → false) :=
by sorry

theorem passes_through_origin
  (a : ℝ) :
  passes_through (line_l a) (0, 1) :=
by sorry

end perpendicular_when_a_is_neg1_passes_through_origin_l628_628472


namespace percentage_half_day_students_l628_628753

theorem percentage_half_day_students
  (total_students : ℕ)
  (full_day_students : ℕ)
  (h_total : total_students = 80)
  (h_full_day : full_day_students = 60) :
  ((total_students - full_day_students) / total_students : ℚ) * 100 = 25 := 
by
  sorry

end percentage_half_day_students_l628_628753


namespace eugene_total_payment_l628_628911

-- Define the initial costs of items
def cost_tshirt := 20
def cost_pants := 80
def cost_shoes := 150

-- Define the quantities
def quantity_tshirt := 4
def quantity_pants := 3
def quantity_shoes := 2

-- Define the discount rate
def discount_rate := 0.10

-- Define the total pre-discount cost
def pre_discount_cost :=
  (cost_tshirt * quantity_tshirt) +
  (cost_pants * quantity_pants) +
  (cost_shoes * quantity_shoes)

-- Define the discount amount
def discount_amount := discount_rate * pre_discount_cost

-- Define the post-discount cost
def post_discount_cost := pre_discount_cost - discount_amount

-- Theorem statement
theorem eugene_total_payment : post_discount_cost = 558 := by
  sorry

end eugene_total_payment_l628_628911


namespace ratio_of_students_l628_628529

-- Define the conditions
def total_students : Nat := 800
def students_spaghetti : Nat := 320
def students_fettuccine : Nat := 160

-- The proof problem
theorem ratio_of_students (h1 : students_spaghetti = 320) (h2 : students_fettuccine = 160) :
  students_spaghetti / students_fettuccine = 2 := by
  sorry

end ratio_of_students_l628_628529


namespace pentagon_area_eq_l628_628820

variables (A B C D E : Point)
variables (triangle_area : Triangle → ℝ)
variables (S : ℝ)
variables (hABC : triangle_area (Triangle.mk A B C) = S)
variables (hBCD : triangle_area (Triangle.mk B C D) = S)
variables (hCDE : triangle_area (Triangle.mk C D E) = S)
variables (hDEA : triangle_area (Triangle.mk D E A) = S)
variables (hEAB : triangle_area (Triangle.mk E A B) = S)

theorem pentagon_area_eq : 
  pentagon_area (Pentagon.mk A B C D E) = (5 + Real.sqrt 5) * S / 2 :=
sorry

end pentagon_area_eq_l628_628820


namespace find_200_digit_number_l628_628713

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end find_200_digit_number_l628_628713


namespace largest_possible_gcd_l628_628258

theorem largest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) : ∃ d, Nat.gcd a b = d ∧ d = 17 :=
sorry

end largest_possible_gcd_l628_628258


namespace minimum_wire_length_l628_628869

/-
  Grapevines are planted with the following conditions:
  - There are 20 stakes, each 2 meters long.
  - Stakes are placed 5 meters apart in a row.
  - Tops of the stakes are connected with wire.
  - End stakes are braced 1 meter away from the stakes.

  We aim to prove that the minimum length of wire required to set up one 
  row of grapevines is equal to 95 + 2 * sqrt(5) meters.
-/

theorem minimum_wire_length :
  ∃ (n : ℝ), n = 95 + 2 * Real.sqrt 5 :=
begin
  use 95 + 2 * Real.sqrt 5,
  sorry
end

end minimum_wire_length_l628_628869


namespace perpendicular_when_a_is_neg1_passes_through_origin_l628_628474

noncomputable theory

def line_l (a : ℝ) : (ℝ × ℝ) → Prop := λ p, (a^2 + a + 1) * p.1 - p.2 + 1 = 0
def line_x_plus_y : (ℝ × ℝ) → Prop := λ p, p.1 + p.2 = 0

def passes_through (l : (ℝ × ℝ) → Prop) (p : ℝ × ℝ) : Prop := l p

theorem perpendicular_when_a_is_neg1 
  (a : ℝ) (p : ℝ × ℝ) : 
  a = -1 → (∀ p, line_l a p → line_x_plus_y p → false) :=
by sorry

theorem passes_through_origin
  (a : ℝ) :
  passes_through (line_l a) (0, 1) :=
by sorry

end perpendicular_when_a_is_neg1_passes_through_origin_l628_628474


namespace BMWs_sold_l628_628325

def total_cars_sold : ℕ := 300
def percentage_ford : ℕ := 20
def percentage_nissan : ℕ := 25
def percentage_chevrolet : ℕ := 30

theorem BMWs_sold (total_cars_sold = 300) 
                    (percentage_ford = 20) 
                    (percentage_nissan = 25) 
                    (percentage_chevrolet = 30) : 
                    (number_BMWs_sold total_cars_sold percentage_ford percentage_nissan percentage_chevrolet) = 75 := 
by 
  sorry

end BMWs_sold_l628_628325


namespace quadratic_inequality_l628_628851

theorem quadratic_inequality
  (a x₁ x₂ : ℝ)
  (h₀ : 0 < a)
  (h₁ : a < 3)
  (hx : x₁ < x₂)
  (hneq : x₁ + x₂ ≠ 1 - a)
  : (let f := λ x, a * x^2 + 2 * a * x + 4 in f x₁ < f x₂) :=
sorry

end quadratic_inequality_l628_628851


namespace product_of_invertibles_mod_120_l628_628146

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628146


namespace product_of_roots_l628_628784

open Real

theorem product_of_roots : (cbrt 8) * (root 6 64) = 4 := by 
  sorry

end product_of_roots_l628_628784


namespace product_of_invertibles_mod_120_l628_628164

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628164


namespace geometric_sequence_formula_l628_628899

theorem geometric_sequence_formula (a : ℕ → ℝ) (h₀ : ∀ n, a n > 0) 
(h₁ : a 1 = 2) (h₂ : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) : 
∀ n, a n = 2^(↑(n + 1) / 2) :=
begin
  sorry
end

end geometric_sequence_formula_l628_628899


namespace square_of_cube_of_third_smallest_prime_l628_628684

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l628_628684


namespace area_of_quadrilateral_l628_628595

def quadrilateral_area (a b : ℝ) : ℝ :=
  2 * real.sqrt a + (3/2) * real.sqrt b

theorem area_of_quadrilateral (EF FG EH HG EG : ℝ) 
  (h1 : EF ^ 2 + FG ^ 2 = EG ^ 2)
  (h2 : EH ^ 2 + HG ^ 2 = EG ^ 2)
  (h3 : EG = 4)
  (h4 : EF ≠ EH ∨ FG ≠ HG)
  (h5 : EF < 4) (h6 : FG < 4) (h7 : EH < 4) (h8 : HG < 4) : 
  quadrilateral_area (3) (7) = 2 * real.sqrt 3 + (3/2) * real.sqrt 7 := 
sorry

end area_of_quadrilateral_l628_628595


namespace square_of_cube_of_third_smallest_prime_l628_628676

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628676


namespace jean_business_hours_l628_628515

-- Definitions of the conditions
def weekday_hours : ℕ := 10 - 16 -- from 4 pm to 10 pm
def weekend_hours : ℕ := 10 - 18 -- from 6 pm to 10 pm
def weekdays : ℕ := 5 -- Monday through Friday
def weekends : ℕ := 2 -- Saturday and Sunday

-- Total weekly hours
def total_weekly_hours : ℕ :=
  (weekday_hours * weekdays) + (weekend_hours * weekends)

-- Proof statement
theorem jean_business_hours : total_weekly_hours = 38 :=
by
  sorry

end jean_business_hours_l628_628515


namespace measure_of_angle_MTN_l628_628752

-- Problem conditions
variables {A B C M N T : Type}
variables [Field.real A] [Field.real B] [Field.real C] [Field.real M] [Field.real N] [Field.real T]
variables (eqA : EquilateralTriangle A B C)
variables (hAB : CircleRollingAlongBase A B C)
variables (hM : Intersects AC M)
variables (hN : Intersects BC N)
variables (hT : TangentPointOnAB T)

-- Theorem statement
theorem measure_of_angle_MTN {α : RealAngle α} (hMTN : angle α M T N) :
  α = 60 :=
sorry 

end measure_of_angle_MTN_l628_628752


namespace interest_rate_calculation_l628_628886

-- Define the conditions and the main statement
theorem interest_rate_calculation
  (r : ℝ)
  (h1 : 10000 * (1 + r / 100)^18 = 40000)
  (h2 : 70 / r = 9) : r ≈ 7.78 := by sorry

end interest_rate_calculation_l628_628886


namespace buyer_observed_price_l628_628721

-- Define the conditions
def cost_of_item : ℝ := 18
def profit_percentage : ℝ := 0.20
def commission_percentage : ℝ := 0.20

-- Calculate the price that the buyer observes online
noncomputable def final_price : ℝ :=
  let profit := cost_of_item * profit_percentage in
  let price_set_by_distributor := cost_of_item + profit in
  let commission := price_set_by_distributor * commission_percentage in
  price_set_by_distributor + commission

-- Statement of the theorem
theorem buyer_observed_price : final_price = 25.92 :=
  sorry

end buyer_observed_price_l628_628721


namespace rays_total_grocery_bill_l628_628597

-- Conditions
def hamburger_meat_cost : ℝ := 5.0
def crackers_cost : ℝ := 3.50
def frozen_veg_cost_per_bag : ℝ := 2.0
def frozen_veg_bags : ℕ := 4
def cheese_cost : ℝ := 3.50
def discount_rate : ℝ := 0.10

-- Total cost before discount
def total_cost_before_discount : ℝ :=
  hamburger_meat_cost + crackers_cost + (frozen_veg_cost_per_bag * frozen_veg_bags) + cheese_cost

-- Discount amount
def discount_amount : ℝ := discount_rate * total_cost_before_discount

-- Total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Theorem: Ray's total grocery bill
theorem rays_total_grocery_bill : total_cost_after_discount = 18.0 :=
  by
    sorry

end rays_total_grocery_bill_l628_628597


namespace part_a_part_b_part_c_part_d_part_e_l628_628571

noncomputable section

open ProbabilityTheory

variables {Ω : Type*} {σ : MeasurableSpace Ω} (μ : Measure Ω)

def sigma (k : ℕ) (x : Ω) (S : ℕ → Ω) : ℕ∞ :=
  if h : k = 1 then inf {n | S n = x}
  else inf {n | n > sigma (k-1) x S ∧ S n = x}

theorem part_a (n : ℕ) :
  μ {ω | sigma 1 0 (λ i, S i) ω = 2*n} = 2^(-2*n+1) * n^(-1) * nat.choose (2*n-2) (n-1) :=
sorry

theorem part_b (n : ℕ) :
  μ {ω | sigma 1 0 (λ i, S i) ω > 2 * n} = 2^(-2 * n) * nat.choose (2 * n) n :=
sorry

theorem part_c :
  μ {ω | sigma 1 0 (λ i, S i) ω < ⊤} = 1 :=
sorry

theorem part_d :
  ∑' n, 2 * n * (2^(-2*n+1) * n^(-1) * nat.choose (2*n-2) (n-1)) = ⊤ :=
sorry

theorem part_e :
  i.i.d {sigma 1 0 (λ i, S i), (λ ω, sigma 2 0 (λ i, S i) ω - sigma 1 0 (λ i, S i) ω), ...} :=
sorry

end part_a_part_b_part_c_part_d_part_e_l628_628571


namespace sum_first_six_terms_l628_628574

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Define the existence of a geometric sequence with given properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given Condition: a_3 = 2a_4 = 2
def cond1 (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ a 4 = 1

-- Define the sum of the first n terms of the sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

-- We need to prove that under these conditions, S_6 = 63/4
theorem sum_first_six_terms 
  (hq : q = 1 / 2) 
  (ha : is_geometric_sequence a q) 
  (hcond1 : cond1 a) 
  (hS : geometric_sum a q S) : 
  S 6 = 63 / 4 := 
sorry

end sum_first_six_terms_l628_628574


namespace paint_proof_l628_628373

theorem paint_proof (L : ℝ) : 
  let l := 2
  let w := 2
  let h := 4
  let prism_surface_area := 2 * (l * w + l * h + w * h)
  let r := (10 / π).sqrt
  let sphere_volume := (4/3) * π * r^3
  let given_volume := L * sqrt 12 / sqrt π
  in prism_surface_area = 40 → sphere_volume = given_volume → L = (40 * sqrt 2) / 3 :=
by
  -- skipping the proof
  sorry

end paint_proof_l628_628373


namespace polynomial_divisible_m_l628_628383

theorem polynomial_divisible_m (m : ℤ) : (∀ x, (4 * x^2 - 6 * x + m) % (x - 3) = 0) → m = -18 :=
by
  assume h : ∀ x, (4 * x^2 - 6 * x + m) % (x - 3) = 0
  let f := λ x, 4 * x^2 - 6 * x + m
  have : f 3 = 0, from by simp [h]
  simp [f] at this
  sorry

end polynomial_divisible_m_l628_628383


namespace trapezium_second_side_length_l628_628407

theorem trapezium_second_side_length (a b h : ℕ) (Area : ℕ) 
  (h_area : Area = (1 / 2 : ℚ) * (a + b) * h)
  (ha : a = 20) (hh : h = 12) (hA : Area = 228) : b = 18 := by
  sorry

end trapezium_second_side_length_l628_628407


namespace tangent_line_slope_positive_l628_628349

theorem tangent_line_slope_positive : ∀ x : ℝ, (differentiable_at ℝ (λ x, Real.exp x) x) ∧ (deriv (λ x, Real.exp x) x > 0) :=
by
  intros x
  split
  { apply differentiable_at_exp }
  { apply deriv_exp_pos }
  sorry

end tangent_line_slope_positive_l628_628349


namespace area_figure_ABCD_l628_628269

theorem area_figure_ABCD :
  let r := 10
  let angle_ABC := 90 * (π / 180) -- Convert degrees to radians
  let angle_BCD := 45 * (π / 180) -- Convert degrees to radians
  let area_ABC := (angle_ABC / (2 * π)) * (π * r^2)
  let area_BCD := (angle_BCD / (2 * π)) * (π * r^2)
  area_ABC + area_BCD = 37.5 * π := 
by {
  -- Conditions
  let r := 10
  let angle_ABC := 90 * (π / 180)
  let angle_BCD := 45 * (π / 180)
  let area_ABC := (angle_ABC / (2 * π)) * (π * r^2)
  let area_BCD := (angle_BCD / (2 * π)) * (π * r^2)
  
  -- Calculate
  have h1 : area_ABC = 25 * π := by sorry
  have h2 : area_BCD = 12.5 * π := by sorry
  calc
    area_ABC + area_BCD = 25 * π + 12.5 * π : by rw [h1, h2]
    ... = 37.5 * π : by ring
}

end area_figure_ABCD_l628_628269


namespace unique_monic_polynomial_divisibility_l628_628771

-- Define the conditions and the problem statement in Lean 4
theorem unique_monic_polynomial_divisibility (f : ℤ[X]) (h_monic : f.monic) (h_nonconstant : ¬ is_constant f) :
  (∃ M : ℕ, ∀ n : ℕ, n ≥ M → f.eval n ∣ f.eval (2^n) - 2^(f.eval n)) →
  f = polynomial.C 1 * polynomial.X :=
by
  sorry

end unique_monic_polynomial_divisibility_l628_628771


namespace polygon_diagonals_l628_628046

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628046


namespace area_CMN_eq_sqrt2_div_9_l628_628535

-- Define the rectangle ABCD with given side lengths
def is_rectangle (A B C D: ℝ × ℝ) (AB BC: ℝ) :=
  A.1 + AB = B.1 ∧ A.2 = B.2 ∧
  B.1 = C.1 ∧ B.2 + BC = C.2 ∧
  C.1 - AB = D.1 ∧ C.2 = D.2 ∧
  D.1 = A.1 ∧ D.2 - BC = A.2

-- Define the isosceles triangle CMN with specified CM
def is_isosceles_triangle (C M N: ℝ × ℝ) (CM: ℝ) :=
  dist C M = CM ∧ dist C N = CM

-- Define the points A, B, C, D, M, and N.
def A := (0, 0) : ℝ × ℝ
def B := (2, 0) : ℝ × ℝ
def C := (2, 1) : ℝ × ℝ
def D := (0, 1) : ℝ × ℝ
def M := (1, 0) : ℝ × ℝ  -- assuming M is at (x, 0) and we do not predefine x
def N := (2, 1 / 3) : ℝ × ℝ  -- assuming N is at (2, y) and we do not predefine y

noncomputable def area_CM_NN (C M N: ℝ × ℝ) :=
  1 / 2 * abs ((C.1 - N.1) * (C.2 - M.2))

-- Prove the area of triangle CMN is sqrt(2) / 9 given the conditions.
theorem area_CMN_eq_sqrt2_div_9 :
  is_rectangle A B C D 2 1 ∧
  is_isosceles_triangle C M N (2 * Real.sqrt(2) / 3) →
  area_CM_NN C M N = Real.sqrt(2) / 9 :=
sorry

end area_CMN_eq_sqrt2_div_9_l628_628535


namespace sides_of_regular_polygon_with_20_diagonals_l628_628035

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628035


namespace find_barycenter_with_integer_coordinates_l628_628765

theorem find_barycenter_with_integer_coordinates :
  ∃ (A B C : ℤ × ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (1 ≤ A.1 ∧ A.1 ≤ 37) ∧ (1 ≤ B.1 ∧ B.1 ≤ 37) ∧ (1 ≤ C.1 ∧ C.1 ≤ 37) ∧ (1 ≤ A.2 ∧ A.2 ≤ 37) ∧ (1 ≤ B.2 ∧ B.2 ≤ 37) ∧ (1 ≤ C.2 ∧ C.2 ≤ 37) ∧ (1 ≤ A.3 ∧ A.3 ≤ 37) ∧ (1 ≤ B.3 ∧ B.3 ≤ 37) ∧ (1 ≤ C.3 ∧ C.3 ≤ 37) ∧ ((A.1 + B.1 + C.1) % 3 = 0) ∧ ((A.2 + B.2 + C.2) % 3 = 0) ∧ ((A.3 + B.3 + C.3) % 3 = 0) :=
begin
  sorry
end

end find_barycenter_with_integer_coordinates_l628_628765


namespace arithmetic_progression_sum_l628_628892

-- Define the sum of the first 15 terms of the arithmetic progression
theorem arithmetic_progression_sum (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 16) :
  (15 / 2) * (2 * a + 14 * d) = 120 := by
  sorry

end arithmetic_progression_sum_l628_628892


namespace least_n_sum_interesting_l628_628272

def is_interesting (x : ℝ) : Prop :=
  irrational x ∧ 0 < x ∧ x < 1 ∧ (exists d : ℕ, d < 10 ∧ (∀ i : ℕ, 0 ≤ i ∧ i < 4 → floor ((x * 10^(i + 1)) % 10) = d))

def can_be_written_as_sum_of_interesting (t : ℝ) (n : ℕ) : Prop :=
  ∃ (s : finset ℝ), s.card = n ∧ (∀ x ∈ s, is_interesting x) ∧ s.sum id = t

theorem least_n_sum_interesting :
  (∀ t : ℝ, 0 < t ∧ t < 1 → can_be_written_as_sum_of_interesting t 1112) ∧
  (∀ n : ℕ, (∀ t : ℝ, 0 < t ∧ t < 1 → can_be_written_as_sum_of_interesting t n) → 1112 ≤ n) :=
by
  sorry

end least_n_sum_interesting_l628_628272


namespace range_of_a_l628_628865

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 1 > 0) ↔ (-2 < a ∧ a < 2) :=
sorry

end range_of_a_l628_628865


namespace product_of_invertibles_mod_120_l628_628148

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628148


namespace product_of_invertibles_mod_120_l628_628131

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628131


namespace regular_polygon_sides_l628_628056

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628056


namespace bald_eagle_pairs_l628_628645

theorem bald_eagle_pairs (n_1963 : ℕ) (increase : ℕ) (h1 : n_1963 = 417) (h2 : increase = 6649) :
  (n_1963 + increase = 7066) :=
by
  sorry

end bald_eagle_pairs_l628_628645


namespace probability_one_boy_one_girl_l628_628312

def outcomes : Set (Prod Bool Bool) := {(true, true), (true, false), (false, true), (false, false)}

def equally_probable_outcomes : outcomes → ℝ
  | (true, true) => 1/4
  | (true, false) => 1/4
  | (false, true) => 1/4
  | (false, false) => 1/4

def favorable_outcomes : Set (Prod Bool Bool) := {(true, false), (false, true)}

theorem probability_one_boy_one_girl :
  ∑ outcome in favor_outcomes, equally_probable_outcomes outcome = 1/2 :=
sorry

end probability_one_boy_one_girl_l628_628312


namespace total_triangles_l628_628873

theorem total_triangles (small_triangles : ℕ)
    (triangles_4_small : ℕ)
    (triangles_9_small : ℕ)
    (triangles_16_small : ℕ)
    (number_small_triangles : small_triangles = 20)
    (number_triangles_4_small : triangles_4_small = 5)
    (number_triangles_9_small : triangles_9_small = 1)
    (number_triangles_16_small : triangles_16_small = 1) :
    small_triangles + triangles_4_small + triangles_9_small + triangles_16_small = 27 := 
by 
    -- proof omitted
    sorry

end total_triangles_l628_628873


namespace no_real_roots_equationD_l628_628696

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end no_real_roots_equationD_l628_628696


namespace annie_miles_l628_628579

theorem annie_miles (x : ℝ) :
  2.50 + (0.25 * 42) = 2.50 + 5.00 + (0.25 * x) → x = 22 :=
by
  sorry

end annie_miles_l628_628579


namespace area_of_BEDC_l628_628087

-- Definitions for the conditions provided in the problem
variables {A B C D E : Point}
variable [parallelogram ABCD : parallelogram A B C D]
variable (height_BE : segment B E) (side_AD : line A D)
variable (BE : ℝ) (area_ABCD : ℝ) (AE : ℝ) (ED : ℝ)

-- Conditions directly from the problem
def height_from_B := BE = 6
def area_of_ABCD := area_ABCD = 72
def AE_length := AE = 3
def ED_length := ED = 5

-- The target statement that BEDC's area is 63 cm²
theorem area_of_BEDC : height_from_B BE → area_of_ABCD area_ABCD → AE_length AE → ED_length ED → 
  ∃ (area_BEDC : ℝ), area_BEDC = 63 :=
by
  sorry

end area_of_BEDC_l628_628087


namespace area_of_quadrilateral_l628_628596

def quadrilateral_area (a b : ℝ) : ℝ :=
  2 * real.sqrt a + (3/2) * real.sqrt b

theorem area_of_quadrilateral (EF FG EH HG EG : ℝ) 
  (h1 : EF ^ 2 + FG ^ 2 = EG ^ 2)
  (h2 : EH ^ 2 + HG ^ 2 = EG ^ 2)
  (h3 : EG = 4)
  (h4 : EF ≠ EH ∨ FG ≠ HG)
  (h5 : EF < 4) (h6 : FG < 4) (h7 : EH < 4) (h8 : HG < 4) : 
  quadrilateral_area (3) (7) = 2 * real.sqrt 3 + (3/2) * real.sqrt 7 := 
sorry

end area_of_quadrilateral_l628_628596


namespace determine_correct_selling_price_l628_628731

/-- The cost price of the product is 12 yuan per piece. -/
def cost_price : ℝ := 12

/-- 200 pieces can be sold per day at a selling price of 20 yuan per piece. -/
def initial_selling_price : ℝ := 20
def initial_quantity : ℝ := 200

/-- For every one yuan increase in unit price, 20 fewer pieces are sold per day,
    and for every one yuan decrease, 20 more pieces are sold per day. -/
def quantity_sold (selling_price : ℝ) : ℝ :=
  initial_quantity - 20 * (selling_price - initial_selling_price)

/-- Define the profit equation to express daily profit based on selling price. -/
def daily_profit (selling_price : ℝ) : ℝ :=
  (selling_price - cost_price) * quantity_sold(selling_price)

/-- To achieve a daily profit of 1540 yuan, the selling price per piece
    should be either 19 yuan or 23 yuan. -/
theorem determine_correct_selling_price :
  ∃ x : ℝ, daily_profit x = 1540 ∧ (x = 19 ∨ x = 23) := by
  sorry

end determine_correct_selling_price_l628_628731


namespace correct_mapping_l628_628432

open Set

-- Definitions of the sets A and B
def A := {0, 1, 2, 4}
def B := {1/2, 0, 1, 2, 6, 8}

-- Definitions of the functions
def f1 (x : ℕ) := x^3 - 1
def f2 (x : ℕ) := (x - 1)^2
def f3 (x : ℕ) := 2^(x - 1)
def f4 (x : ℕ) := 2 * x

-- The theorem to be proved
theorem correct_mapping : ( ∀ x ∈ A, f3 x ∈ B ) ∧ ¬( ∃ f ∈ ({f1, f2, f4} : Set (ℕ → ℕ)), ∀ x ∈ A, f x ∈ B ) :=
by
  sorry

end correct_mapping_l628_628432


namespace men_l628_628319

namespace WagesProblem

def men_women_boys_equivalence (man woman boy : ℕ) : Prop :=
  9 * man = woman ∧ woman = 7 * boy

def total_earnings (man woman boy earnings : ℕ) : Prop :=
  (9 * man + woman + woman) = earnings ∧ earnings = 216

theorem men's_wages (man woman boy : ℕ) (h1 : men_women_boys_equivalence man woman boy) (h2 : total_earnings man woman 7 216) : 9 * man = 72 :=
sorry

end WagesProblem

end men_l628_628319


namespace trigonometric_propositions_l628_628748

theorem trigonometric_propositions :
  let prop1 := ∀ (k : ℤ), ∀ x, (sin (k * Real.pi - x) = sin x) ∨ (sin (k * Real.pi - x) = -sin x)
  let prop2 := ∀ x, (tan (Real.pi - x) = 2) → (cos x)^2 = 1 / 5
  let prop3 := (∀ x, (y = tan (2 * x + Real.pi / 6)) → (2 * x + Real.pi / 6 ≠ Real.pi / 3)) → False
  let prop4 := (∀ x, (y = cos (2 * x + Real.pi / 3)) → ¬ x = -2 * Real.pi / 3) → False
  prop1 ∧ prop2 ∧ prop3 ∧ prop4 := sorry

end trigonometric_propositions_l628_628748


namespace sin_alpha_half_l628_628435

theorem sin_alpha_half {α : ℝ} (h₁ : sin (α + π / 3) = -1 / 2) (h₂ : 2 * π / 3 < α ∧ α < π) : sin α = 1 / 2 :=
sorry

end sin_alpha_half_l628_628435


namespace find_n_l628_628701

theorem find_n (n : ℕ) (h1 : let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7 in
                          a = 24)
               (h2 : let b := 2 * n in
                          a^2 - b^2 = 0) :
  n = 12 :=
by
  sorry

end find_n_l628_628701


namespace sum_seq_eq_max_k_val_l628_628572

def geom_seq (n : ℕ) : ℝ :=
  2^n - 1

def a_n (n : ℕ) : ℝ :=
  2^(n-1)

def b_n (n : ℕ) : ℝ :=
  2 * n

def sum_first_n_terms (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (a_n k) * (b_n k)

theorem sum_seq_eq (n : ℕ) : sum_first_n_terms n = ((n-1) * 2^(n+1) + 2) :=
by
  sorry

theorem max_k_val : ∀ n : ℕ, (∏ i in finset.range n, (1 + b_n i) / (b_n i)) ≥ (3 * real.sqrt 2 / 4) * real.sqrt (n + 1) :=
by
  sorry

end sum_seq_eq_max_k_val_l628_628572


namespace find_sides_from_diagonals_l628_628062

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628062


namespace min_S2_value_l628_628802

theorem min_S2_value (a : Fin 351 → {x // x ∈ {1, 2, 3, 4}}) 
  (h1 : (∑ i : Fin 351, a i.1) = 513) 
  (h2 : (∑ i : Fin 351, (a i.1)^4) = 4745) : 
  (∃ m : Nat, m = 905 ∧ ∀ S2 : Nat, 
    (∀ b : Fin 351 → {x // x ∈ {1, 2, 3, 4}},
      (∑ i : Fin 351, b i.1) = 513 ∧ 
      (∑ i : Fin 351, (b i.1)^4) = 4745 → S2 = (∑ i : Fin 351, (b i.1)^2)) → 
    S2 ≥ m)
  ∧ (∑ i : Fin 351, (a i.1)^2 = 905) :=
by
  sorry

end min_S2_value_l628_628802


namespace minimal_value_f_l628_628628

noncomputable def f : ℝ → ℝ := λ x, x * Real.exp x

theorem minimal_value_f : ∃ x, f x = -(1 / Real.exp 1) ∧ ∀ y, f y ≥ f x :=
begin
  sorry
end

end minimal_value_f_l628_628628


namespace correct_propositions_l628_628956

theorem correct_propositions : 
  ∀ (a b : Int), 
    (∀ n : Int, a + 5 * b = 2 * n → a - 3 * b = 2 * k for some k : Int) ∧
    (∀ n : Int, a + b = 3 * n → ∃ m p : Int, a = 3 * m ∧ b = 3 * p) ∧
    (∀ p : Int, prime (a + b) → ¬prime (a - b)) ∧
    (c = a + b ≠ 0 → (a^3 - b^3) / (a^3 + c^3) = (a - b) / (a + c))
  → 3 :=
by
  sorry

end correct_propositions_l628_628956


namespace bridge_length_correct_l628_628705

def train_length : ℝ := 180
def train_speed_kmh : ℝ := 60
def crossing_time : ℝ := 25

def km_to_m (km : ℝ) : ℝ := km * 1000
def hr_to_s (hr : ℝ) : ℝ := hr * 3600

def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
def total_distance : ℝ := train_speed_ms * crossing_time
def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 236.75 := by
  sorry

end bridge_length_correct_l628_628705


namespace determine_c_l628_628626

theorem determine_c (a b c : ℕ) (h₁ : (a - 3) ^ 2 + (b - 2) ^ 2 = 0) (h₂ : 1 < c < 5) : c = 4 :=
by sorry

end determine_c_l628_628626


namespace john_distance_first_second_l628_628546

variables (d : ℝ)

-- Conditions
def john_time_to_run_100m : ℝ := 13
def james_time_to_run_100m : ℝ := 11
def james_first_10m_time : ℝ := 2
def john_speed_after_first_second (d : ℝ) : ℝ := (100 - d) / 12
def james_top_speed : ℝ := 10
def john_top_speed : ℝ := james_top_speed - 2

-- Main statement to prove
theorem john_distance_first_second (h1 : john_speed_after_first_second d = john_top_speed) : d = 4 :=
by sorry

end john_distance_first_second_l628_628546


namespace sandy_total_money_l628_628221

def half_dollar_value := 0.5
def quarter_value := 0.25
def dime_value := 0.1
def nickel_value := 0.05
def dollar_value := 1.0

def monday_total := 12 * half_dollar_value + 5 * quarter_value + 10 * dime_value
def tuesday_total := 8 * half_dollar_value + 15 * quarter_value + 5 * dime_value
def wednesday_total := 3 * dollar_value + 4 * half_dollar_value + 10 * quarter_value + 7 * nickel_value
def thursday_total := 5 * dollar_value + 6 * half_dollar_value + 8 * quarter_value + 5 * dime_value + 12 * nickel_value
def friday_total := 2 * dollar_value + 7 * half_dollar_value + 20 * nickel_value + 25 * dime_value

def total_amount := monday_total + tuesday_total + wednesday_total + thursday_total + friday_total

theorem sandy_total_money : total_amount = 44.45 := by
  sorry

end sandy_total_money_l628_628221


namespace textile_firm_sales_value_l628_628343

noncomputable def aggregate_sales_value (number_of_looms : ℕ) (manufacturing_expenses : ℕ) (establishment_charges : ℕ) (decrease_in_profit : ℕ) : ℕ :=
let total_expenses := manufacturing_expenses + establishment_charges
let profit_for_one_loom := decrease_in_profit
let sales_value_per_loom := profit_for_one_loom * number_of_looms
sales_value_per_loom

theorem textile_firm_sales_value :
  aggregate_sales_value 70 150000 75000 5000 = 350000 :=
by
  let number_of_looms := 70
  let manufacturing_expenses := 150000
  let establishment_charges := 75000
  let decrease_in_profit := 5000
  let total_expenses := manufacturing_expenses + establishment_charges
  let sales_value_per_loom := decrease_in_profit * number_of_looms
  show sales_value_per_loom = 350000 from sorry

end textile_firm_sales_value_l628_628343


namespace proof1_proof2_l628_628833

-- Given assumptions
variables {α : ℝ}
axiom h1 : sin α = 4 / 5
axiom h2 : α ∈ Ioo (π / 2) π

-- Prove sin (α - π / 6) = (4 * sqrt 3 + 3) / 10
theorem proof1 : sin (α - π / 6) = (4 * Real.sqrt 3 + 3) / 10 :=
by
  sorry

-- Prove tan (2 * α) = 24 / 7
theorem proof2 : tan (2 * α) = 24 / 7 :=
by
  sorry

end proof1_proof2_l628_628833


namespace product_of_invertibles_mod_120_l628_628136

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628136


namespace perimeter_of_figure_l628_628793

theorem perimeter_of_figure (x y : ℝ) (p : Set (ℝ × ℝ)) 
  (h1 : ∀ (x y : ℝ), (|x + y| + |x - y| = 8) ↔ ((x, y) ∈ p)) :
  (perimeter p) = 16 * Real.sqrt 2 := 
sorry

end perimeter_of_figure_l628_628793


namespace arc_length_parametric_l628_628707

open Real Interval

noncomputable def arc_length (f_x f_y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in Set.Icc t1 t2, sqrt ((deriv f_x t)^2 + (deriv f_y t)^2)

theorem arc_length_parametric :
  arc_length
    (λ t => 2.5 * (t - sin t))
    (λ t => 2.5 * (1 - cos t))
    (π / 2) π = 5 * sqrt 2 :=
by
  sorry

end arc_length_parametric_l628_628707


namespace deer_weight_calculation_l628_628547

def hunting_times_per_month := 6
def hunting_season_duration_months := 3  -- 1 quarter of the year
def deers_per_hunt := 2
def weight_kept_per_year := 10800  -- pounds
def weight_ratio_kept := 1 / 2

theorem deer_weight_calculation :
  let total_hunts := hunting_times_per_month * hunting_season_duration_months in
  let total_deers := total_hunts * deers_per_hunt in
  let total_weight := weight_kept_per_year / weight_ratio_kept in
  ∀ w, total_weight = total_deers * w → w = 600 := 
by
  intros total_hunts total_deers total_weight w h
  have h1 : total_hunts = 6 * 3 := by rfl
  have h2 : total_deers = total_hunts * 2 := by rfl
  have h3 : total_weight = 10800 / (1 / 2) := by rfl
  sorry  -- Proof steps go here

end deer_weight_calculation_l628_628547


namespace flowers_per_bouquet_l628_628210

noncomputable def num_flowers_per_bouquet (total_flowers wilted_flowers bouquets : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / bouquets

theorem flowers_per_bouquet : num_flowers_per_bouquet 53 18 5 = 7 := by
  sorry

end flowers_per_bouquet_l628_628210


namespace hyperbola_focus_distance_l628_628957

variables {m : ℝ}
def hyperbola (x y : ℝ) : Prop := (x^2) / 9 - (y^2) / m = 1

variables {F1 F2 : ℝ × ℝ} (F_eq : F1 = (-4, 0) ∨ F2 = (-4, 0))

theorem hyperbola_focus_distance 
  (x y : ℝ)
  (P : x^2 / 9 - y^2 / m = 1)
  (orthogonal : let PF1 := ((x-F1.1)^2 + (y-F1.2)^2)^0.5 in 
                     let PF2 := ((x-F2.1)^2 + (y-F2.2)^2)^0.5 in
                     PF1 * PF2 = 0)
  (directrix_pass : F_eq) :
  ∃ PF1 PF2 : ℝ, PF1 * PF2 = 14 := by
  sorry

end hyperbola_focus_distance_l628_628957


namespace part1_sin_C_part1_side_b_part2_cos_2A_plus_pi_over_3_l628_628929

noncomputable theory
open Classical

variables (A B C : ℝ)
variables (a b c : ℝ)
variables (cos_A : ℝ)

-- Given conditions
def triangle_config (a b c : ℝ) (cos_A : ℝ) : Prop :=
  a = 2 ∧ c = sqrt 2 ∧ cos_A = -sqrt 2 / 4

-- Part 1: Prove sin C
def sin_C (A B C : ℝ) : ℝ := sorry

theorem part1_sin_C (a c cos_A : ℝ) (h : triangle_config a b c cos_A) : sin_C A B C = sqrt 7 / 4 :=
sorry

-- Part 1: Prove side b
def side_b (A B C : ℝ) : ℝ := sorry

theorem part1_side_b (a c cos_A : ℝ) (h : triangle_config a b c cos_A) : side_b A B C = 1 :=
sorry

-- Part 2: Prove cos(2A + pi/3)
def cos_2A_plus_pi_over_3 (A B C : ℝ) : ℝ := sorry

theorem part2_cos_2A_plus_pi_over_3 (a c cos_A : ℝ) (h : triangle_config a b c cos_A) : cos_2A_plus_pi_over_3 A B C = (-3 + sqrt 21) / 8 :=
sorry

end part1_sin_C_part1_side_b_part2_cos_2A_plus_pi_over_3_l628_628929


namespace all_are_knights_l628_628262

-- Definitions for inhabitants as either knights or knaves
inductive Inhabitant
| Knight : Inhabitant
| Knave : Inhabitant

open Inhabitant

-- Functions that determine if an inhabitant is a knight or a knave
def is_knight (x : Inhabitant) : Prop :=
  x = Knight

def is_knave (x : Inhabitant) : Prop :=
  x = Knave

-- Given conditions
axiom A : Inhabitant
axiom B : Inhabitant
axiom C : Inhabitant

axiom statement_A : is_knight A → is_knight B
axiom statement_B : is_knight B → (is_knight A → is_knight C)

-- The proof goal
theorem all_are_knights : is_knight A ∧ is_knight B ∧ is_knight C := by
  sorry

end all_are_knights_l628_628262


namespace polygon_diagonals_l628_628047

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628047


namespace greatest_real_part_of_z6_l628_628973

def z_A : ℂ := -3
def z_B : ℂ := -2 * Real.sqrt 3 + 2 * Complex.i
def z_C : ℂ := -3 * Real.sqrt 3 + 3 * Complex.i
def z_D : ℂ := -2 + 3 * Real.sqrt 3 * Complex.i
def z_E : ℂ := -3 * Complex.i

theorem greatest_real_part_of_z6 :
  ∀ (z : ℂ), (z = z_A ∨ z = z_B ∨ z = z_C ∨ z = z_D ∨ z = z_E) →
  Complex.re (z^6) ≤ 729 :=
sorry

end greatest_real_part_of_z6_l628_628973


namespace at_least_one_prob_better_option_l628_628232

-- Definitions based on the conditions in a)

def player_A_prelim := 1 / 2
def player_B_prelim := 1 / 3
def player_C_prelim := 1 / 2

def final_round := 1 / 3

def prelim_prob_A := player_A_prelim * final_round
def prelim_prob_B := player_B_prelim * final_round
def prelim_prob_C := player_C_prelim * final_round

def prob_none := (1 - prelim_prob_A) * (1 - prelim_prob_B) * (1 - prelim_prob_C)

def prob_at_least_one := 1 - prob_none

-- Question 1 statement

theorem at_least_one_prob :
  prob_at_least_one = 31 / 81 :=
sorry

-- Definitions based on the reward options in the conditions

def option_1_lottery_prob := 1 / 3
def option_1_reward := 600
def option_1_expected_value := 600 * 3 * (1 / 3)

def option_2_prelim_reward := 100
def option_2_final_reward := 400

-- Expected values calculation for Option 2

def option_2_expected_value :=
  (300 * (1 / 6) + 600 * (5 / 12) + 900 * (1 / 3) + 1200 * (1 / 12))

-- Question 2 statement

theorem better_option :
  option_1_expected_value < option_2_expected_value :=
sorry

end at_least_one_prob_better_option_l628_628232


namespace find_AC_l628_628895

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]

def length (a b : Type) [MetricSpace a] [MetricSpace b] (x : a) (y : b) : ℝ := sorry

def angle (a b c : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] (x : a) (y : b) (z : c) : ℝ := sorry

noncomputable def area_of_triangle (a b c : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] (x : a) (y : b) (z : c) : ℝ := sorry

theorem find_AC (ABC : Type) [MetricSpace ABC] (BC_length : length B C = 1) (angle_B : angle A B C = π / 3)
  (area_ABC : area_of_triangle A B C = sqrt 3) : length A C = sqrt 13 :=
sorry

end find_AC_l628_628895


namespace students_taking_German_but_not_Spanish_l628_628206

theorem students_taking_German_but_not_Spanish :
  ∃ (x y : ℕ), x + y + 2 = 30 ∧ 3 * (x + 2) = y + 2 ∧ (y - 2) = 20 :=
by
  have x := 6
  have y := 22
  exact ⟨x, y, by linarith, by linarith, by linarith⟩

end students_taking_German_but_not_Spanish_l628_628206


namespace two_digit_numbers_34_and_17_l628_628878

theorem two_digit_numbers_34_and_17 : 
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    a = 2 * b ∧ 
    (∀ d ∈ [digits a], d ∉ [digits b]) ∧ 
    (∃ s d, b = s ∧ (10 * (a / 10) + a % 10 = s + s) ∧ (10 * (a / 10) - a % 10 = d)) :=
by {
  sorry
}

end two_digit_numbers_34_and_17_l628_628878


namespace typing_pages_l628_628512

theorem typing_pages (typists : ℕ) (pages min : ℕ) 
  (h_typists_can_type_two_pages_in_two_minutes : typists * 2 / min = pages / min) 
  (h_10_typists_type_25_pages_in_5_minutes : 10 * 25 / 5 = pages / min) :
  pages / min = 2 := 
sorry

end typing_pages_l628_628512


namespace arithmetic_sequence_sum_l628_628091

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℚ), 
    (∀ n, a (n+1) = a n + 1/2) →
    (a 1 + a 3 + a 99 = 60) →
    (∑ n in finset.range (100), a (n + 1) = 145) :=
begin
  sorry
end

end arithmetic_sequence_sum_l628_628091


namespace square_of_cube_of_third_smallest_prime_l628_628678

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628678


namespace find_a_l628_628202

noncomputable def normal_distribution (mean variance : ℝ) : Type :=
sorry

variables (ξ : Type) [normal_distribution 3 7 ξ]

def probability_greater (x : ℝ) : ℝ := sorry
def probability_less (x : ℝ) : ℝ := sorry

theorem find_a (a : ℝ) (h1 : probability_greater(ξ) (a + 2) = probability_less(ξ) (a - 2)) : a = 3 :=
sorry

end find_a_l628_628202


namespace triangle_problem_l628_628842

noncomputable theory
open Real

/--
In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
and (2b - c) * cos A = a * cos C, then
1. angle A = pi / 3
2. If D is a point on side BC such that BD = 2 * DC and AD = 2, the maximum area of triangle ABC = (3 * sqrt 3) / 2
-/
theorem triangle_problem 
  (a b c : ℝ)
  (A B C : ℝ)
  (D : ℝ)
  (BD DC : ℝ)
  (AD : ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (sqrt : ℝ → ℝ)
  (h1 : a / sin A = b / sin B = c / sin C)
  (h2 : (2 * b - c) * cos A = a * cos C)
  (h3 : BD = 2 * DC)
  (h4 : AD = 2) : 
  (A = π / 3) ∧ (∀ t : ℝ, t = D → (1 / 2) * b * c * (sin A) ≤ (3 * sqrt 3) / 2) :=
by
  sorry

end triangle_problem_l628_628842


namespace eugene_payment_correct_l628_628914

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l628_628914


namespace dave_paid_more_l628_628330

noncomputable def pizza_cost : ℕ := 12
noncomputable def olive_cost : ℕ := 3
noncomputable def mushroom_cost : ℕ := 4

def total_slices : ℕ := 12
def olive_slices : ℕ := total_slices / 4
def mushroom_slices : ℕ := total_slices / 2

def dave_slices : ℕ := mushroom_slices + 2
def doug_slices : ℕ := total_slices - dave_slices

def base_slice_cost : ℕ := pizza_cost / total_slices
def olive_slice_cost : ℕ := olive_cost / olive_slices
def mushroom_slice_cost : ℕ := mushroom_cost / mushroom_slices

noncomputable def dave_payment : ℕ :=
  dave_slices - 2 * base_slice_cost + 2 * (base_slice_cost + olive_slice_cost) + mushroom_slices * (base_slice_cost + mushroom_slice_cost)

noncomputable def doug_payment : ℕ :=
  doug_slices * base_slice_cost

noncomputable def payment_difference : ℕ :=
  dave_payment - doug_payment

theorem dave_paid_more : payment_difference = 10 := by
  sorry -- Proof goes here

end dave_paid_more_l628_628330


namespace sqrt_mult_sqrt_sq_l628_628689

theorem sqrt_mult_sqrt_sq (h1 : (\sqrt{49 \times \sqrt{25}})^2 = 245) : 
    (\sqrt{49 * \sqrt{25}})^2 = 245 :=
by sorry

end sqrt_mult_sqrt_sq_l628_628689


namespace range_of_x_l628_628827

noncomputable def f : ℝ → ℝ := sorry -- Define f with provided properties

theorem range_of_x (f_even : ∀ x, f x = f (-x))
  (f_monotonic_decreasing : ∀ x y, 0 ≤ x ∧ x ≤ y → f y ≤ f x)
  (h_f_neg2_zero : f (-2) = 0) :
  {x : ℝ | x * (f (x - 1)) > 0} = (Ioo (-∞) (-1)) ∪ (Ioo 0 3) :=
sorry

end range_of_x_l628_628827


namespace gold_coin_multiple_l628_628751

theorem gold_coin_multiple (x y k : ℕ) (h₁ : x + y = 16) (h₂ : x ≠ y) (h₃ : x^2 - y^2 = k * (x - y)) : k = 16 :=
sorry

end gold_coin_multiple_l628_628751


namespace gcd_of_powers_one_l628_628194

theorem gcd_of_powers_one (a b m n : ℤ) (h_coprime : Int.gcd a b = 1) (h_gt : a > b) :
  Int.gcd (a^m - b^m) (a^n - b^n) = a^(Int.gcd m n) - b^(Int.gcd m n) :=
sorry

end gcd_of_powers_one_l628_628194


namespace option_d_correct_l628_628290

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l628_628290


namespace find_special_four_digit_square_l628_628401

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l628_628401


namespace non_congruent_triangles_with_perimeter_18_l628_628001

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l628_628001


namespace exercise_l628_628238

theorem exercise (x y z : ℝ)
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : 1/x + 1/y + 1/z = 3/5) : x^2 + y^2 + z^2 = 488.4 :=
sorry

end exercise_l628_628238


namespace minimum_value_is_1_l628_628657

def minimum_value_expression (x y : ℝ) : ℝ :=
  x^2 + y^2 - 8*x + 6*y + 26

theorem minimum_value_is_1 (x y : ℝ) (h : x ≥ 4) : 
  minimum_value_expression x y ≥ 1 :=
by {
  sorry
}

end minimum_value_is_1_l628_628657


namespace ratio_consequent_l628_628526

theorem ratio_consequent (a b x : ℕ) (h_ratio : a = 4) (h_b : b = 6) (h_x : x = 30) :
  (a : ℚ) / b = x / 45 := 
by 
  -- add here the necessary proof steps 
  sorry

end ratio_consequent_l628_628526


namespace factorial_divisibility_l628_628198

def binary_ones_count (n : ℕ) : ℕ :=
  n.to_digits 2 |>.foldl (λ acc d => acc + if d == 1 then 1 else 0) 0

theorem factorial_divisibility (n : ℕ) (h₁ : 0 < n) (h₂ : binary_ones_count n = 1995) : 2^(n - 1995) ∣ n! :=
by {
  sorry
}

end factorial_divisibility_l628_628198


namespace star_operation_example_l628_628328

-- Define the operation ☆
def star (a b : ℚ) : ℚ := a - b + 1

-- The theorem to prove
theorem star_operation_example : star (star 2 3) 2 = -1 := by
  sorry

end star_operation_example_l628_628328


namespace count_two_digit_primes_with_digit_sum_eight_l628_628427

def isPrime (n : ℕ) : Prop := sorry  -- Assume a pre-existing definition or use a built-in one
def sumDigits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2

theorem count_two_digit_primes_with_digit_sum_eight :
  (Finset.filter (λ x, isPrime x ∧ sumDigits x = 8) (Finset.Icc 10 99)).card = 3 := 
by
  sorry

end count_two_digit_primes_with_digit_sum_eight_l628_628427


namespace hexagon_sides_equal_l628_628951

-- Define the basic structure of an acute-angled triangle
structure AcuteAngledTriangle :=
(A B C : Point)
(A1 B1 C1 : Point)
(altitudes : Segment)
(A_altitude : altitude (Segment A A1))
(B_altitude : altitude (Segment B B1))
(C_altitude : altitude (Segment C C1))

-- Define the centers of the incircles of the triangles
structure IncircleCenters :=
(OA OB OC : Point)
(OA_center : incenter (Triangle A B1 C1) = OA)
(OB_center : incenter (Triangle B C1 A1) = OB)
(OC_center : incenter (Triangle C A1 B1) = OC)

-- Points of tangency of the incircle
structure TangencyPoints :=
(TA TB TC : Point)
(TA_tangent : tangent Point B C TA)
(TB_tangent : tangent Point C A TB)
(TC_tangent : tangent Point A B TC)

-- Define the theorem
theorem hexagon_sides_equal (T : AcuteAngledTriangle) (I : IncircleCenters) (TP : TangencyPoints) :
  side_length (Segment TP.TA I.OC) = side_length (Segment TP.TB I.OA) ∧
  side_length (Segment TP.TB I.OA) = side_length (Segment TP.TC I.OB) ∧
  side_length (Segment TP.TC I.OB) = side_length (Segment TP.TA I.OC) ∧
  side_length (Segment TP.TA I.OC) = side_length (Segment TP.TB I.OA) :=
sorry

end hexagon_sides_equal_l628_628951


namespace solve_expression_l628_628602

theorem solve_expression :
  ( (12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45) ^ 2) = 90.493 := 
by 
  sorry

end solve_expression_l628_628602


namespace non_congruent_triangles_perimeter_18_l628_628005

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628005


namespace sum_of_integers_is_28_l628_628508

theorem sum_of_integers_is_28 (m n p q : ℕ) (hmnpq_diff : m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q)
  (hm_pos : 0 < m) (hn_pos : 0 < n) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_prod : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 :=
by
  sorry

end sum_of_integers_is_28_l628_628508


namespace circumcircle_tangent_to_line_AC_l628_628560

variables {A B C T M R S P : Type}

-- Given conditions
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]
variables [nonempty A] [nonempty B] [nonempty C]
variables (triangle_acute_A : A) (triangle_acute_B : B) (triangle_acute_C : C)
variables (circumcircle : ∀ (A B C : Type), Type)
variables (tangent_from_A : A) (intersects_BC_at_T : T)
variables (midpoint_M : M) (reflection_R : R) (parallelogram_SABT : S)
variables (point_line_SB : P) (MP_parallel_AB : Prop)
variables (P_on_Γ : Prop)

-- Proving statement
theorem circumcircle_tangent_to_line_AC :
  ∀ {A B C T M R S P : Type},
    A = triangle_acute_A ∧
    B = triangle_acute_B ∧
    C = triangle_acute_C ∧
    ∀ (Γ : Type), Γ = circumcircle A B C → 
    ∀ (tangent : Type), tangent = tangent_from_A → 
    ∀ (T : Type), T = intersects_BC_at_T →
    ∀ (M : Type), M = midpoint_M →
    ∀ (R : Type), R = reflection_R →
    ∀ (S : Type), S = parallelogram_SABT →
    ∀ (P : Type), (P = point_line_SB ∧ MP_parallel_AB) →
    P_on_Γ →
    let circumcircle_STR := circumcircle S T R,
    tangent_to_line circumcircle_STR AC :=
  sorry

end circumcircle_tangent_to_line_AC_l628_628560


namespace area_sum_proof_l628_628561

noncomputable def area_sum : ℝ :=
  let S := 1 -- Side length of the unit square
  let area_one := (1/2) * (1/2) * (1/2) -- Area of the triangle BQ1P1
  (1/2) * ∑' i in (Set.Ici 1 : Set ℕ), area_one / (4^(i-1))

theorem area_sum_proof : 
  area_sum = 1 / 6 := 
by
  sorry

end area_sum_proof_l628_628561


namespace lateral_surface_area_truncated_cone_l628_628255

theorem lateral_surface_area_truncated_cone :
  let r := 1
  let R := 4
  let h := 4
  let l := Real.sqrt ((R - r)^2 + h^2)
  let S := Real.pi * (r + R) * l
  S = 25 * Real.pi :=
by
  sorry

end lateral_surface_area_truncated_cone_l628_628255


namespace sampling_method_is_systematic_sampling_l628_628722

-- Definitions based on the problem's conditions
def produces_products (factory : Type) : Prop := sorry
def uses_conveyor_belt (factory : Type) : Prop := sorry
def takes_item_every_5_minutes (inspector : Type) : Prop := sorry

-- Lean 4 statement to prove the question equals the answer given the conditions
theorem sampling_method_is_systematic_sampling
  (factory : Type)
  (inspector : Type)
  (h1 : produces_products factory)
  (h2 : uses_conveyor_belt factory)
  (h3 : takes_item_every_5_minutes inspector) :
  systematic_sampling_method := 
sorry

end sampling_method_is_systematic_sampling_l628_628722


namespace negation_equiv_l628_628891

theorem negation_equiv (p : Prop) : 
  (p = (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) → 
  (¬ p = (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by
  sorry

end negation_equiv_l628_628891


namespace distance_between_lines_l628_628774

-- Definitions and Hypotheses
def line1 (x y : ℝ) := 2 * x + 3 * y - 5 = 0
def line2 (x y : ℝ) := 2 * x + 3 * y - 2 = 0

-- The theorem stating the distance between two given parallel lines
theorem distance_between_lines (d : ℝ) :
  let a := 2
  let b := 3
  let c1 := -5
  let c2 := -2
  d = abs (c1 - c2) / sqrt (a^2 + b^2) →
  d = (3 * sqrt 13) / 13 :=
begin
  intros ha,
  sorry
end

end distance_between_lines_l628_628774


namespace inequality_solution_l628_628303

noncomputable def log_base_0_3 (x : ℝ) := log x / log 0.3

lemma log_base_0_3_behavior (x y : ℝ) (h1 : 0.3 < 1) (h2 : x = abs (y - 2)) : 
  log_base_0_3 x < 0 ↔ y < 1 ∨ y > 3 :=
sorry 

lemma quadratic_behavior (x : ℝ) : x^2 - 4 * x < 0 ↔ 0 < x ∧ x < 4 :=
sorry

theorem inequality_solution (x : ℝ) :
  (log_base_0_3 (abs (x - 2)) / (x^2 - 4 * x) < 0) ↔ 
  (x < 0 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 4) :=
sorry

end inequality_solution_l628_628303


namespace square_of_cube_of_third_smallest_prime_l628_628685

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l628_628685


namespace common_difference_arithmetic_sequence_l628_628083

variable (n d : ℝ) (a : ℝ := 7 - 2 * d) (an : ℝ := 37) (Sn : ℝ := 198)

theorem common_difference_arithmetic_sequence :
  7 + (n - 3) * d = 37 ∧ 
  396 = n * (44 - 2 * d) ∧
  Sn = n / 2 * (a + an) →
  (∃ d : ℝ, 7 + (n - 3) * d = 37 ∧ 396 = n * (44 - 2 * d)) :=
by
  sorry

end common_difference_arithmetic_sequence_l628_628083


namespace non_congruent_triangles_perimeter_18_l628_628012

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628012


namespace arithmetic_sequence_lemma_l628_628826

theorem arithmetic_sequence_lemma (a : ℕ → ℝ) (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0)
  (h_condition : a 3 + a 11 = 22) : a 7 = 11 :=
sorry

end arithmetic_sequence_lemma_l628_628826


namespace angle_between_lines_l628_628234

theorem angle_between_lines (x y : ℝ) (l₁ : x - y = 0) (l₂ : x + y - 1 = 0) :
  angle_between l₁ l₂ = Real.pi / 2 :=
sorry

end angle_between_lines_l628_628234


namespace konstantin_step_problem_l628_628557

theorem konstantin_step_problem:
  ∀ (a : ℝ) (d : ℕ → ℝ),
    (∀ (n : ℕ), n < 300 → d n ≤ a) →
    (∑ i in Finset.range 300, d i) = 18000 →
    (∀ (i j k : ℕ), i < 300 → j < 300 → k < 300 → i ≠ j → j ≠ k → i ≠ k → d i + d j > d k) →
    60 ≤ a ∧ a < 18000 / 299 :=
begin
  sorry
end

end konstantin_step_problem_l628_628557


namespace total_earnings_correct_l628_628301

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l628_628301


namespace smallest_integer_y_l628_628660

theorem smallest_integer_y : ∃ (y : ℕ), (\frac{8}{11} < \frac{y}{17}) ∧ y = 13 :=
by
  sorry

end smallest_integer_y_l628_628660


namespace find_k_l628_628945

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b : V) (k : ℝ)
variables (A B C D : V)

-- Conditions
def non_collinear : Prop := ¬Collinear ℝ {a, b}
def AB_eq : Prop := B - A = 2 • a + k • b
def CB_eq : Prop := C - B = a + 3 • b
def CD_eq : Prop := D - C = 2 • a - b
def collinear_ABD : Prop := Collinear ℝ {A, B, D}

-- The statement to be proved
theorem find_k
  (h1 : non_collinear a b)
  (h2 : AB_eq a b k A B)
  (h3 : CB_eq a b C B)
  (h4 : CD_eq a b C D)
  (h5 : collinear_ABD A B D) :
  k = -8 :=
sorry

end find_k_l628_628945


namespace bus_stops_for_4_minutes_per_hour_l628_628397

theorem bus_stops_for_4_minutes_per_hour
  (V_excluding_stoppages V_including_stoppages : ℝ)
  (h1 : V_excluding_stoppages = 90)
  (h2 : V_including_stoppages = 84) :
  (60 * (V_excluding_stoppages - V_including_stoppages)) / V_excluding_stoppages = 4 :=
by
  sorry

end bus_stops_for_4_minutes_per_hour_l628_628397


namespace length_of_AG_l628_628921

/-- Given a right-angled triangle ABC with ∠A = 90°, AB = 3 cm, AC = 3√5 cm. 
    E is the midpoint of BC, AD is the altitude from A to BC, and G is the point
    where AD intersects the median from B to E, then the length of AG is 3√10 / 2 cm. -/
theorem length_of_AG {A B C D E G : Point}
  (h_triangle : IsRightTriangle A B C)
  (h_angle : ∠A = 90°)
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 3 * Real.sqrt 5)
  (h_E : Midpoint B C E)
  (h_AD_perp_BC : Perpendicular A D B C)
  (h_G : MedianIntersection A D B E G) :
  dist A G = 3 * (Real.sqrt 10) / 2 :=
  sorry

end length_of_AG_l628_628921


namespace find_fx2_l628_628838

theorem find_fx2 (f : ℝ → ℝ) (x : ℝ) (h : f (x - 1) = x ^ 2) : f (x ^ 2) = (x ^ 2 + 1) ^ 2 := by
  sorry

end find_fx2_l628_628838


namespace non_congruent_triangles_perimeter_18_l628_628006

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628006


namespace number_of_possible_values_for_A_l628_628220

def is_rounding_to_1300 (A : ℕ) : Prop :=
  A >= 5 ∧ A ≤ 9

theorem number_of_possible_values_for_A : 
  (finset.filter is_rounding_to_1300 (finset.range 10)).card = 5 :=
by
  sorry

end number_of_possible_values_for_A_l628_628220


namespace find_least_prime_y_l628_628527

def is_prime (n : ℕ) : Prop := nat.prime n

theorem find_least_prime_y :
  ∃ (y : ℕ), y + x = 90 ∧ is_prime y ∧ is_prime x ∧ ∀ (z : ℕ), z + u = 90 → is_prime z → is_prime u → y ≤ z :=
by
  sorry

end find_least_prime_y_l628_628527


namespace dartboard_partitions_count_l628_628364

theorem dartboard_partitions_count : 
  ∃ (l : Multiset ℕ), l.sum = 6 ∧ l.card ≤ 5 ∧ l.sort = l ∧ Multiset.card (Multiset.Powerset l) = 10 :=
sorry

end dartboard_partitions_count_l628_628364


namespace cylinder_sphere_ratio_is_3_2_l628_628845

noncomputable def cylinder_sphere_surface_ratio (r : ℝ) : ℝ :=
  let cylinder_surface_area := 2 * Real.pi * r^2 + 2 * r * Real.pi * (2 * r)
  let sphere_surface_area := 4 * Real.pi * r^2
  cylinder_surface_area / sphere_surface_area

theorem cylinder_sphere_ratio_is_3_2 (r : ℝ) (h : r > 0) :
  cylinder_sphere_surface_ratio r = 3 / 2 :=
by
  sorry

end cylinder_sphere_ratio_is_3_2_l628_628845


namespace hyperbola_eccentricity_l628_628862

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    let C := (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1,
        P := (x y : ℝ) => (x - b)^2 + y^2 = a^2,
        asymptote := (x y : ℝ) => b * x - a * y = 0,
        M N : ℝ × ℝ := sorry--intersection points of P and asymptote,
        angle_MPN := sorry-- ∠MPN = 90°
    in eccentricity(C) = √2 := sorry


end hyperbola_eccentricity_l628_628862


namespace inradius_orthic_inequality_l628_628741

noncomputable theory

variables (A B C : Type) [triangle A B C] (R r p : ℝ)
class orthic_triangle (A B C : Type) :=
  (circumradius : ℝ) 
  (inradius : ℝ) 
  (orthic_inradius : ℝ)

theorem inradius_orthic_inequality [orthic_triangle A B C] :
  ∀ (R r p : ℝ), 
  R = orthic_triangle.circumradius →
  r = orthic_triangle.inradius →
  p = orthic_triangle.orthic_inradius →
  (p / R) ≤ 1 - (1 / 3) * (1 + (r / R))^2 := 
sorry

end inradius_orthic_inequality_l628_628741


namespace product_of_invertibles_mod_120_l628_628150

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628150


namespace value_of_2_times_4_l628_628803

def sequence_sum (x : ℤ) : ℤ :=
2 * x + (2 * x - 1) + (2 * x - 2) + ... + 2 + 1

theorem value_of_2_times_4 : sequence_sum 2 * sequence_sum 4 = 360 := by
  sorry

end value_of_2_times_4_l628_628803


namespace find_value_of_p_l628_628242

theorem find_value_of_p (p : ℝ) :
  (∀ x y, (x = 0 ∧ y = -2) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 1/2 ∧ y = 0) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 2 ∧ y = 0) → y = p*x^2 + 5*x + p) →
  p = -2 :=
by
  sorry

end find_value_of_p_l628_628242


namespace find_special_four_digit_square_l628_628403

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l628_628403


namespace sum_or_element_divisible_by_n_l628_628528

theorem sum_or_element_divisible_by_n (n : ℕ) (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≤ n) :
  ∃ i j, i ≤ j ∧ j ≤ n ∧ (∑ m in Finset.range (i+1), a m.succ % n) % n = 0 :=
by sorry

end sum_or_element_divisible_by_n_l628_628528


namespace function_passes_through_point_l628_628854

variable (a : ℝ) (x : ℝ)
noncomputable def f : ℝ := 3 * Real.logBase a (4 * x - 7) + 2

theorem function_passes_through_point 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  : f a 2 = 2 := 
by 
  sorry

end function_passes_through_point_l628_628854


namespace product_invertibles_mod_120_l628_628159

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628159


namespace equation1_unique_solutions_equation2_unique_solutions_l628_628228

noncomputable def solve_equation1 : ℝ → Prop :=
fun x => x ^ 2 - 4 * x + 1 = 0

noncomputable def solve_equation2 : ℝ → Prop :=
fun x => 2 * x ^ 2 - 3 * x + 1 = 0

theorem equation1_unique_solutions :
  ∀ x, solve_equation1 x ↔ (x = 2 + Real.sqrt 3) ∨ (x = 2 - Real.sqrt 3) := by
  sorry

theorem equation2_unique_solutions :
  ∀ x, solve_equation2 x ↔ (x = 1) ∨ (x = 1 / 2) := by
  sorry

end equation1_unique_solutions_equation2_unique_solutions_l628_628228


namespace ramu_profit_percent_l628_628310

theorem ramu_profit_percent
  (cost_of_car : ℕ)
  (cost_of_repairs : ℕ)
  (selling_price : ℕ)
  (total_cost : ℕ := cost_of_car + cost_of_repairs)
  (profit : ℕ := selling_price - total_cost)
  (profit_percent : ℚ := ((profit : ℚ) / total_cost) * 100)
  (h1 : cost_of_car = 42000)
  (h2 : cost_of_repairs = 15000)
  (h3 : selling_price = 64900) :
  profit_percent = 13.86 :=
by
  sorry

end ramu_profit_percent_l628_628310


namespace smallest_c_value_l628_628949

-- Definition of the problem
def satisfies_equation (c : ℝ) : Prop :=
  (3 * c + 4) * (c - 2) = 7 * c + 6

-- The goal is to prove that the smallest c satisfying the equation is (9 - sqrt(249)) / 6
theorem smallest_c_value : ∃ c : ℝ, satisfies_equation c ∧ c = (9 - real.sqrt 249) / 6 := by
  sorry

end smallest_c_value_l628_628949


namespace tan_theta_eq_sqrt_3_of_f_maximum_l628_628620

theorem tan_theta_eq_sqrt_3_of_f_maximum (θ : ℝ) 
  (h : ∀ x : ℝ, 3 * Real.sin (x + (Real.pi / 6)) ≤ 3 * Real.sin (θ + (Real.pi / 6))) : 
  Real.tan θ = Real.sqrt 3 :=
sorry

end tan_theta_eq_sqrt_3_of_f_maximum_l628_628620


namespace desired_depth_proof_l628_628320

-- Definitions based on the conditions in Step a)
def initial_men : ℕ := 9
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def extra_men : ℕ := 11
def total_men : ℕ := initial_men + extra_men
def new_hours : ℕ := 6

-- Total man-hours for initial setup
def initial_man_hours (days : ℕ) : ℕ := initial_men * initial_hours * days

-- Total man-hours for new setup to achieve desired depth
def new_man_hours (desired_depth : ℕ) (days : ℕ) : ℕ := total_men * new_hours * days

-- Proportional relationship between initial setup and desired depth
theorem desired_depth_proof (days : ℕ) (desired_depth : ℕ) :
  initial_man_hours days / initial_depth = new_man_hours desired_depth days / desired_depth → desired_depth = 18 :=
by
  sorry

end desired_depth_proof_l628_628320


namespace sneakers_original_price_l628_628326

noncomputable theory

def original_price (P : ℝ) : Prop :=
    ∃ (P : ℝ), ((P - 10) * 0.9 = 99) ∧ P = 120

theorem sneakers_original_price : original_price 120 :=
by
  sorry

end sneakers_original_price_l628_628326


namespace num_valid_x_is_14_l628_628113

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_three_digit (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

def num_valid_x : ℕ :=
  (Finset.range 1000).filter (λ x, is_three_digit x ∧ digit_sum (digit_sum x) = 5).card

theorem num_valid_x_is_14 : num_valid_x = 14 := sorry

end num_valid_x_is_14_l628_628113


namespace part1_part2_l628_628761

variable (a b : ℝ)

theorem part1 : ((-a)^2 * (a^2)^2 / a^3) = a^3 := sorry

theorem part2 : (a + b) * (a - b) - (a - b)^2 = 2 * a * b - 2 * b^2 := sorry

end part1_part2_l628_628761


namespace geometric_series_r_l628_628243

theorem geometric_series_r (a r : ℝ) 
  (h1 : a * (1 + r + r^2 + r^3 + r^4 + · · ·) = 20)
  (h2 : a * (r + r^3 + r^5 + · · ·) = 8) :
  r = 2 / 3 :=
sorry

end geometric_series_r_l628_628243


namespace quadratic_inequality_empty_solution_set_l628_628070

theorem quadratic_inequality_empty_solution_set (a b c : ℝ) (hₐ : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0 → False) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
by 
  sorry

end quadratic_inequality_empty_solution_set_l628_628070


namespace ratio_height_to_base_rectangle_l628_628925

def square_side_length : ℝ := 4
def E_midpoint (x : ℝ) : Prop := x = 2
def F_midpoint (y : ℝ) : Prop := y = 2
def AG_perpendicular_BF : Prop := true

theorem ratio_height_to_base_rectangle :
  ∃ XY YZ : ℝ,
  XY = 8 * Real.sqrt 5 / 5 ∧
  YZ = 2 * Real.sqrt 5 ∧
  XY / YZ = 0.8 :=
by
  use (8 * Real.sqrt 5 / 5)
  use (2 * Real.sqrt 5)
  split
  { simp }
  { field_simp [Real.sqrt_ne_zero.2 (show 5 ≠ 0 from by norm_num), division_def] }
  sorry


end ratio_height_to_base_rectangle_l628_628925


namespace angle_FEA_eq_60_l628_628345

variable {Point : Type} [InnerProductSpace ℝ Point]

def midpoint (A B : Point) : Point := (A + B) / 2

variables (A B C D E F : Point)
variable (FAE FEA : ℝ)
variable h_midpoint : D = midpoint B C
variable h_BE_eq_2AD : dist B E = 2 * dist A D 
variable h_FA_meets_BE_AD : ∃ F, line(A, D) ∩ line(B, E) = {F}
variable h_FAE_eq_60 : ∠ F A E = 60

-- The goal is to prove that ∠FEA = 60°
theorem angle_FEA_eq_60 : ∠ F E A = 60 := by 
  sorry

end angle_FEA_eq_60_l628_628345


namespace mangoes_in_shop_l628_628966

-- Define the conditions
def ratio_mango_to_apple := 10 / 3
def apples := 36

-- Problem statement to prove
theorem mangoes_in_shop : ∃ (m : ℕ), m = 120 ∧ m = apples * ratio_mango_to_apple :=
by
  sorry

end mangoes_in_shop_l628_628966


namespace monotonic_decreasing_min_value_l628_628464

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem monotonic_decreasing (a : ℝ) : (∀ x ∈ Icc 2 3, (1/x - a) ≤ 0) ↔ a ≥ 1/2 :=
begin
  sorry,
end

theorem min_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ Icc 1 2, f x a = 
  if a < Real.log 2 then -a else Real.log 2 - 2 * a) :=
begin
  sorry,
end

end monotonic_decreasing_min_value_l628_628464


namespace proof_by_contradiction_l628_628988

theorem proof_by_contradiction (P : Prop) (h : P) : (¬ P → false) :=
by 
  intro np 
  exact (np h)

end proof_by_contradiction_l628_628988


namespace evaluate_sqrt_sum_l628_628999

theorem evaluate_sqrt_sum : (Real.sqrt 1 + Real.sqrt 9) = 4 := by
  sorry

end evaluate_sqrt_sum_l628_628999


namespace min_value_proof_l628_628417

noncomputable def min_value_expression : ℝ :=
  let expr : ℝ → ℝ := λ x, (Real.sin x ^ 8 + Real.cos x ^ 8 + 1) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 1)
  let min_value := (2/15 : ℝ)
  min_value

theorem min_value_proof : 
  ∃ x : ℝ, let expr : ℝ → ℝ := λ x, (Real.sin x ^ 8 + Real.cos x ^ 8 + 1) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 1) 
  ∧ expr x = (2 / 15 : ℝ) := sorry

end min_value_proof_l628_628417


namespace square_of_cube_of_third_smallest_prime_l628_628664

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628664


namespace minimum_radius_third_sphere_l628_628351

-- Definitions for the problem
def height_cone := 4
def base_radius_cone := 3
def cos_alpha := 4 / 5
def radius_identical_sphere := 4 / 3
def cos_beta := 1 -- since beta is maximized

-- Define the required minimum radius for the third sphere based on the given conditions
theorem minimum_radius_third_sphere :
  ∃ x : ℝ, x = 27 / 35 ∧
    (height_cone = 4) ∧ 
    (base_radius_cone = 3) ∧ 
    (cos_alpha = 4 / 5) ∧ 
    (radius_identical_sphere = 4 / 3) ∧ 
    (cos_beta = 1) :=
sorry

end minimum_radius_third_sphere_l628_628351


namespace square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l628_628708

-- Define the problem conditions.
def square_grid (n : Nat) : Prop := true
def rectangle_grid (m n : Nat) : Prop := true

-- Define the grid size for square and rectangle.
def square_grid_21 := square_grid 21
def rectangle_grid_20_21 := rectangle_grid 20 21

-- Define the proof problem to find maximum moves.
theorem square_grid_21_max_moves : ∃ m : Nat, m = 3 :=
  sorry

theorem rectangle_grid_20_21_max_moves : ∃ m : Nat, m = 4 :=
  sorry

end square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l628_628708


namespace percentage_increase_in_output_per_hour_l628_628105

theorem percentage_increase_in_output_per_hour:
  ∀ (X Y : ℝ),
  let original_output_per_hour := X / Y,
      A_output := (1.8 * X) / (0.9 * Y),
      B_output := (2.2 * X) / (0.85 * Y),
      C_output := (2.5 * X) / (0.8 * Y),
      percentage_increase (new_output original_output : ℝ) :=
        ((new_output / original_output) - 1) * 100 in
  percentage_increase A_output original_output_per_hour = 100 ∧
  percentage_increase B_output original_output_per_hour = 158.82 ∧
  percentage_increase C_output original_output_per_hour = 212.5 :=
by
  intros X Y
  let original_output_per_hour := X / Y
  let A_output := (1.8 * X) / (0.9 * Y)
  let B_output := (2.2 * X) / (0.85 * Y)
  let C_output := (2.5 * X) / (0.8 * Y)
  let percentage_increase := 
        λ new_output original_output: ℝ, ((new_output / original_output) - 1) * 100
  split
  case left =>
    exact sorry
  case right =>
    split
    case left =>
      exact sorry
    case right =>
      exact sorry

end percentage_increase_in_output_per_hour_l628_628105


namespace product_of_invertible_integers_mod_120_l628_628143

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628143


namespace square_of_cube_of_third_smallest_prime_l628_628665

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628665


namespace triangle_area_l628_628735

theorem triangle_area :
  let A := (1, 3)
  let B := (7, 1)
  let C := (5, 6)
  let area := 0.5 * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))
  abs area = 13.0 :=
by
  let A := (1, 3)
  let B := (7, 1)
  let C := (5, 6)
  let area := 0.5 * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))
  have h : abs area = 13.0 := sorry
  exact h

end triangle_area_l628_628735


namespace colorable_into_four_regions_l628_628919

-- Define the parameters and the graph structure
variables {V : Type*} [fintype V]
variables (G : simple_graph V) [decidable_rel G.adj]

theorem colorable_into_four_regions (h1 : G.connected)
  (h2 : ∀ (C : list V), G.is_cycle_of_length (2 * n + 1).succ C → ∃ u v, ¬G.adj u v) :
  ∃ (f : V → fin 4), G.proper_coloring f := sorry

end colorable_into_four_regions_l628_628919


namespace geometric_ratio_l628_628441

noncomputable def S (n : ℕ) : ℝ := sorry  -- Let's assume S is a function that returns the sum of the first n terms of the geometric sequence.

-- Conditions
axiom S_10_eq_S_5 : S 10 = 2 * S 5

-- Definition to be proved
theorem geometric_ratio :
  (S 5 + S 10 + S 15) / (S 10 - S 5) = -9 / 2 :=
sorry

end geometric_ratio_l628_628441


namespace no_real_roots_equationD_l628_628695

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end no_real_roots_equationD_l628_628695


namespace largest_digit_B_divisible_by_3_l628_628927

-- Define the six-digit number form and the known digits sum.
def isIntegerDivisibleBy3 (b : ℕ) : Prop :=
  b < 10 ∧ (b + 30) % 3 = 0

-- The main theorem: Find the largest digit B such that the number 4B5,894 is divisible by 3.
theorem largest_digit_B_divisible_by_3 : ∃ (B : ℕ), isIntegerDivisibleBy3 B ∧ ∀ (b' : ℕ), isIntegerDivisibleBy3 b' → b' ≤ B := by
  -- Notice the existential and universal quantifiers involved in finding the largest B.
  sorry

end largest_digit_B_divisible_by_3_l628_628927


namespace rhombus_area_l628_628655

theorem rhombus_area (s : ℝ) (d1 d2 : ℝ) (h1 : s = Real.sqrt 145) (h2 : abs (d1 - d2) = 10) : 
  (1/2) * d1 * d2 = 100 :=
sorry

end rhombus_area_l628_628655


namespace sum_of_all_valid_three_digit_numbers_l628_628798

-- Define the set of valid digits
def valid_digits : Finset ℕ := {1, 2, 3, 4, 6, 7, 8, 9}

-- Define a three-digit number using valid digits
def is_valid_number (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n in
  n >= 100 ∧ n < 1000 ∧
  (digits.get 2 (by simp [Nat.digits_len 10 n])) ∈ valid_digits ∧ -- hundreds place
  (digits.get 1 (by simp [Nat.digits_len 10 n])) ∈ valid_digits ∧ -- tens place
  (digits.get 0 (by simp [Nat.digits_len 10 n])) ∈ valid_digits   -- units place

-- Define the sum of all valid three-digit numbers
def sum_of_valid_numbers : ℕ :=
  (Finset.filter is_valid_number (Finset.Icc 100 999)).sum id

theorem sum_of_all_valid_three_digit_numbers :
  sum_of_valid_numbers = 284160 :=
sorry -- Proof omitted

end sum_of_all_valid_three_digit_numbers_l628_628798


namespace last_digit_8_pow_19_l628_628581

theorem last_digit_8_pow_19 : Nat.digits 10 (8^19) % 10 = 2 := 
  by
    -- Define the powers of 2 and their remainders when divided by 10
    have h1 : (2^1) % 10 = 2 := rfl
    have h2 : (2^2) % 10 = 4 := rfl
    have h3 : (2^3) % 10 = 8 := rfl
    have h4 : (2^4) % 10 = 6 := rfl

    -- Define the cycle and use it to deduce the result for 2^57
    have cycle : ∀ (n : Nat), (2^n) % 10 = [2, 4, 8, 6][n % 4] := sorry

    -- Knowing 8 = 2^3, we deduce 8^19 = 2^(3*19) = 2^57
    have h19 := Nat.pow_mod 2 57 10
    have h57 := cycle 57

    -- Using the cycle length (4) to determine the last digit
    have remainder := Nat.mod_eq_of_lt (57 % 4)
    show (2^57) % 10 = 2 from sorry

end last_digit_8_pow_19_l628_628581


namespace angle_C1D_B1B_is_90_l628_628920

-- Define the basic structure and conditions of the problem
structure Cuboid :=
( A B C D A1 B1 C1 D1 : Point )
( angle_BAB1 : ℝ )
( angle_BAB1_val : angle_BAB1 = 30 )

variable { cuboid : Cuboid }

-- Define the theorem to prove the angle between C1D and B1B is 90 degrees
theorem angle_C1D_B1B_is_90 (cuboid : Cuboid) : angle (cuboid.C1 − cuboid.D) (cuboid.B1 − cuboid.B) = 90 :=
by sorry

end angle_C1D_B1B_is_90_l628_628920


namespace calculate_new_concentration_l628_628307

def new_concentration : ℚ :=
  let V1 := 2   -- volume of first vessel in liters
  let V2 := 6   -- volume of second vessel in liters
  let C1 := 0.20  -- concentration of alcohol in first vessel
  let C2 := 0.55  -- concentration of alcohol in second vessel
  let Vt := 8 -- total final volume
  let alcohol1 := C1 * V1  -- alcohol in the first vessel in liters
  let alcohol2 := C2 * V2  -- alcohol in the second vessel in liters
  let total_alcohol := alcohol1 + alcohol2 -- total alcohol in liters
  (total_alcohol / Vt) * 100 

theorem calculate_new_concentration :
  new_concentration = 46.25 := 
by
  -- inserting the value of new_concentration
  unfold new_concentration
  -- performing the necessary calculations
  have h1 : nat.cast (2) = (2:ℚ) := rfl
  have h2 : nat.cast (6) = (6:ℚ) := rfl
  have h3 : (0.20:ℚ) * 2 = (0.4:ℚ) := by norm_num
  have h4 : (0.55:ℚ) * 6 = (3.3:ℚ) := by norm_num
  have h5 : (0.4:ℚ) + 3.3 = 3.7 := by norm_num
  have h6 : (3.7:ℚ) / 8 = (0.4625:ℚ) := by norm_num
  rw [h3, h4, h5, h6]
  norm_num
  sorry

end calculate_new_concentration_l628_628307


namespace value_of_expression_l628_628436

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
  sorry

end value_of_expression_l628_628436


namespace real_part_of_z_is_zero_l628_628632

-- Define the complex number z
def z : ℂ := (1 + complex.i)^2

-- State the goal: the real part of z is 0
theorem real_part_of_z_is_zero : z.re = 0 :=
by
  sorry

end real_part_of_z_is_zero_l628_628632


namespace range_of_a_l628_628460

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 0 then -x^2 - 2*x + a else f (x - 1) a

def has_three_distinct_zeros (a : ℝ) : Prop :=
  let f_x := λ x, if x < 0 then -x^2 - 2*x + a else f (x - 1) a
  let y := λ x, f_x x - x
  ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ y x1 = 0 ∧ y x2 = 0 ∧ y x3 = 0

theorem range_of_a (h : ∀ a, has_three_distinct_zeros a) : 
  ∀ a, a ∈ set.Ici (-1) := 
sorry

end range_of_a_l628_628460


namespace leadership_structure_ways_l628_628350

/-!
# Leadership Selection in a Tribe

Given a tribe consisting of 15 members, we prove that the number of different ways to choose:
1. One chief,
2. Three supporting chiefs,
3. Two inferior officers for each supporting chief,
is 19320300.
-/

open Finset

noncomputable def chooseLeadershipWays (total_members : ℕ) : ℕ :=
  total_members * 
  (total_members - 1) * 
  (total_members - 2) * 
  choose (total_members - 4) 2 * 
  choose (total_members - 6) 2 * 
  choose (total_members - 8) 2

theorem leadership_structure_ways (total_members : ℕ)
  (h : total_members = 15) : chooseLeadershipWays total_members = 19320300 :=
by
  rw [chooseLeadershipWays, h]
  norm_num
  sorry

end leadership_structure_ways_l628_628350


namespace perfect_square_pattern_l628_628398

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l628_628398


namespace find_m_l628_628455

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) (h1 : ∀ n, S n = n^2)
  (h2 : S m = (a m + a (m + 1)) / 2)
  (h3 : ∀ n > 1, a n = S n - S (n - 1))
  (h4 : a 1 = 1) :
  m = 2 :=
sorry

end find_m_l628_628455


namespace alexa_weight_proof_l628_628743

variable (totalWeight katerinaWeight alexaWeight : ℕ)

def weight_relation (totalWeight katerinaWeight alexaWeight : ℕ) : Prop :=
  totalWeight = katerinaWeight + alexaWeight

theorem alexa_weight_proof (h1 : totalWeight = 95) (h2 : katerinaWeight = 49) : alexaWeight = 46 :=
by
  have h : alexaWeight = totalWeight - katerinaWeight := by
    sorry
  rw [h1, h2] at h
  exact h

end alexa_weight_proof_l628_628743


namespace f_decreasing_on_interval_l628_628213

open Set Filter

-- Define the function f and the interval (1, +∞)
def f (x : ℝ) : ℝ := 2*x / (x - 1)

-- The theorem we want to prove
theorem f_decreasing_on_interval (x : ℝ) (h : 1 < x) : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 > f x2 :=
by 
  admit

end f_decreasing_on_interval_l628_628213


namespace product_of_invertibles_mod_120_l628_628121

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628121


namespace list_price_unique_l628_628744

theorem list_price_unique (x : ℝ) 
  (Alice_sell_price : x - 10) (Alice_commission : 0.10 * (x - 10)) 
  (Bob_sell_price : x - 20) (Bob_commission : 0.20 * (x - 20)) 
  (commission_equal : 0.10 * (x - 10) = 0.20 * (x - 20)) : 
  x = 30 :=
by sorry

end list_price_unique_l628_628744


namespace ellipse_h_k_a_b_sum_15_l628_628568

noncomputable def center_of_ellipse (F1 F2 : ℝ × ℝ) : ℝ × ℝ := 
((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def semi_major_axis (sum_distances : ℝ) : ℝ := sum_distances / 2

noncomputable def semi_minor_axis (a c : ℝ) : ℝ := real.sqrt (a ^ 2 - c ^ 2)

theorem ellipse_h_k_a_b_sum_15 :
  let F1 := (1, 2 : ℝ)
      F2 := (7, 2 : ℝ)
      sum_distances := 10
      a := semi_major_axis sum_distances
      c := distance F1 F2 / 2
      b := semi_minor_axis a c
      (h, k) := center_of_ellipse F1 F2
  in h + k + a + b = 15 :=
by
  sorry

end ellipse_h_k_a_b_sum_15_l628_628568


namespace problem_B_solution_l628_628118

notation "^" => pow

theorem problem_B_solution (m := 16^1000) : (m / 8) = 2^3997 :=
by
  sorry

end problem_B_solution_l628_628118


namespace find_constants_positive_f_l628_628858

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.exp 2)) * (Real.exp x) - Real.log x

theorem find_constants :
  let a := 1 / (Real.exp 2)
  let b := 1
  f(1) = (1 / Real.exp 1) ∧
  (a * Real.exp 1 - b = 1 / Real.exp 1 - 1) :=
  by
  sorry

theorem positive_f (x : ℝ) (h : 0 < x) : f x > 0 :=
  by
  sorry

end find_constants_positive_f_l628_628858


namespace rhombus_area_calculation_l628_628614

def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_calculation (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 18) : rhombus_area d1 d2 = 270 :=
by
  rw [h1, h2]
  -- Here we should demonstrate that the actual calculation is correct, but since the proof is not required, we use sorry.
  sorry

end rhombus_area_calculation_l628_628614


namespace dress_hem_length_in_feet_l628_628934

def stitch_length_in_inches : ℚ := 1 / 4
def stitches_per_minute : ℕ := 24
def time_in_minutes : ℕ := 6

theorem dress_hem_length_in_feet :
  (stitch_length_in_inches * (stitches_per_minute * time_in_minutes)) / 12 = 3 :=
by
  sorry

end dress_hem_length_in_feet_l628_628934


namespace positive_integers_count_l628_628872

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  0 < n

-- Define the condition that (n + 9)(n - 4)(n - 13) < 0
def satisfies_inequality (n : ℤ) : Prop :=
  (n + 9) * (n - 4) * (n - 13) < 0

-- The main theorem stating the number of positive integers satisfying the inequality is 8
theorem positive_integers_count :
  { n : ℕ | is_positive_integer n ∧ satisfies_inequality n }.toFinset.card = 8 :=
  sorry

end positive_integers_count_l628_628872


namespace line_properties_l628_628471

theorem line_properties (a : ℝ) :
  -- Condition of the problem
  let l := (a^2 + a + 1) * x - y + 1 = 0,

  -- Correct Answer (A): Line l is perpendicular to x + y = 0 when a = -1
  (a = -1 → (let k1 := (a^2 + a + 1)
             let k2 := -1
             (k1 * k2 = -1))) ∧
  
  -- Correct Answer (C): Line l passes through the point (0, 1) for any a
  (l 0 1 = 0) :=
  
by sorry

end line_properties_l628_628471


namespace find_b_values_l628_628778

-- Define the point coordinates
def point_x : ℕ := 3
def point_y : ℕ := 10

-- Prove the values of b for given linear equations
theorem find_b_values (b1 b2 b3 b4 : ℝ) :
  point_y = point_x + b1 ∧ point_y = 3 * point_x + b2 ∧ point_y = -(1 / 3) * point_x + b3 ∧ point_y = -(1 / 2) * point_x + b4 :=
by {
  split;
  { sorry },
  split;
  { sorry },
  split;
  { sorry },
  { sorry }
}

end find_b_values_l628_628778


namespace gumballs_remaining_l628_628745

theorem gumballs_remaining (Alicia_gumballs : ℕ)
    (Pedro_additional_pct : ℚ)
    (Pedro_take_out_pct : ℚ)
    (hAlicia  : Alicia_gumballs = 20)
    (hPedro_additional_pct : Pedro_additional_pct = 1.5)
    (hPedro_take_out_pct : Pedro_take_out_pct = 0.55) :
    let Pedro_gumballs := Alicia_gumballs + (Pedro_additional_pct * Alicia_gumballs).to_nat in
    let Total_gumballs := Alicia_gumballs + Pedro_gumballs in
    let Taken_gumballs := (Pedro_take_out_pct * Total_gumballs).to_nat in
    Total_gumballs - Taken_gumballs = 32 := 
begin
  sorry
end

end gumballs_remaining_l628_628745


namespace polygon_sides_l628_628492

theorem polygon_sides :
  ∃ (n : ℕ), (n * (n - 3) / 2) = n + 33 ∧ n = 11 :=
by
  sorry

end polygon_sides_l628_628492


namespace limit_a_n_to_a_l628_628214

open Real

-- Definition of the sequence
def a_n (n : ℕ) : ℝ := (2 * n - 5) / (3 * n + 1)

-- Definition of the limit value
def a : ℝ := 2 / 3

-- The proof statement
theorem limit_a_n_to_a : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (a_n n - a) < ε := by
  sorry

end limit_a_n_to_a_l628_628214


namespace find_angle_sum_l628_628115

theorem find_angle_sum (c d : ℝ) (hc : 0 < c ∧ c < π/2) (hd : 0 < d ∧ d < π/2)
    (h1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
    (h2 : 4 * Real.sin (2 * c) = 3 * Real.cos (2 * d)) :
    2 * c + 3 * d = π / 2 :=
by
  sorry

end find_angle_sum_l628_628115


namespace collinear_endpoints_minimize_distance_l628_628500

open Real

-- Definitions and problem statements
variables {a b : ℝ → ℝ → ℝ} -- Representing vectors in R^2 or R^3

-- Conditions for Problem 1
axiom non_collinear_non_zero (a b : ℝ → ℝ → ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ k : ℝ, a = k • b → False

-- Problem 1: Collinearity
theorem collinear_endpoints {t : ℝ} (a b : ℝ → ℝ → ℝ) 
  (h : non_collinear_non_zero a b) :
  ∀ t : ℝ, endpoints_collinear a (t • b) (1/3 • (a + b)) ↔ t = 1/2 := 
sorry

-- Conditions for Problem 2
axiom equal_magnitude_angle (a b : ℝ → ℝ → ℝ) :
  (∃ k : ℝ, ∥a∥ = k ∧ ∥b∥ = k ∧ ∃ theta : ℝ, theta = π / 3)

-- Problem 2: Minimizing distance
theorem minimize_distance {t : ℝ} (a b : ℝ → ℝ → ℝ)
  (h : equal_magnitude_angle a b) :
  ∀ t : ℝ, minimizer (∥a - t • b∥) ↔ t = 1/2 := 
sorry

end collinear_endpoints_minimize_distance_l628_628500


namespace non_congruent_triangles_perimeter_18_l628_628009

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628009


namespace total_points_scored_l628_628209

theorem total_points_scored :
  ∀ (Noa_points Phillip_points : ℕ),
  Noa_points = 30 →
  Phillip_points = 2 * Noa_points →
  Noa_points + Phillip_points = 90 :=
by
  intros Noa_points Phillip_points Noa_points_def Phillip_points_def
  rw [Noa_points_def, Phillip_points_def]
  exact rfl

end total_points_scored_l628_628209


namespace common_tangents_of_circles_l628_628239

theorem common_tangents_of_circles :
  let C1 := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1}
  let C2 := {p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 4) ^ 2 = 16}
  ∃ tangents : ℕ, tangents = 3 := 
by
  let C1 := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1}; assume hC1
  let C2 := {p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 4) ^ 2 = 16}; assume hC2
  use 3
  sorry

end common_tangents_of_circles_l628_628239


namespace roots_squared_sum_diff_l628_628589

theorem roots_squared_sum_diff (x₁ x₂ p q : ℝ) 
  (h1 : x₁^2 + p * x₁ + q = 0) 
  (h2 : x₂^2 + p * x₂ + q = 0) : 
  x₁^2 + x₂^2 = p^2 - 2 * q ∧ x₁^2 - x₂^2 = p * real.sqrt (p^2 - 4 * q) ∨ x₁^2 - x₂^2 = -p * real.sqrt (p^2 - 4 * q) := 
by
  sorry

end roots_squared_sum_diff_l628_628589


namespace odd_period_function_subtraction_l628_628881

theorem odd_period_function_subtraction 
  (f : ℝ → ℝ) 
  (hf_odd : ∀ x : ℝ, f(-x) = -f(x)) 
  (hf_periodic : ∀ x : ℝ, f(x + 5) = f(x))
  (hf1 : f(1) = 1) 
  (hf2 : f(2) = 2) :
  f(3) - f(4) = -1 :=
sorry

end odd_period_function_subtraction_l628_628881


namespace eugene_total_cost_l628_628906

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l628_628906


namespace K_3_18_12_eq_17_div_3_l628_628806

def K (x y z : ℝ) : ℝ :=
  (x / y) + (y / z) + (z / x)

theorem K_3_18_12_eq_17_div_3 : K 3 18 12 = 17 / 3 := by
  sorry

end K_3_18_12_eq_17_div_3_l628_628806


namespace larger_number_is_23_l628_628653

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l628_628653


namespace product_of_invertibles_mod_120_l628_628125

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628125


namespace roots_bounds_if_and_only_if_conditions_l628_628766

theorem roots_bounds_if_and_only_if_conditions (a b c : ℝ) (h : a > 0) (x1 x2 : ℝ) (hr : ∀ {x : ℝ}, a * x^2 + b * x + c = 0 → x = x1 ∨ x = x2) :
  (|x1| ≤ 1 ∧ |x2| ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) :=
sorry

end roots_bounds_if_and_only_if_conditions_l628_628766


namespace correct_calculation_l628_628292

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l628_628292


namespace concyclic_points_l628_628192

open Real EuclideanGeometry

-- Definitions based on conditions
variables {A B C D S E F : Point}
variable (circle : Circle)

-- Points A, B, C, and D lie on the circle in that order
def points_on_circle_in_order (A B C D : Point) (circle : Circle) : Prop :=
  ∀ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (circle.contains A at_angle a) ∧
  (circle.contains B at_angle b) ∧
  (circle.contains C at_angle c) ∧
  (circle.contains D at_angle d) ∧
  a < b ∧ b < c ∧ c < d

-- S is the midpoint of the arc AB that does not contain C and D
def is_midpoint_of_arc_not_containing (S A B C D : Point) (circle : Circle) : Prop :=
  let arc := circle.minor_arc A B in
  arc.contains S ∧ ¬ arc.contains C ∧ ¬ arc.contains D

-- Lines (SD) and (SC) intersect (AB) at E and F respectively
def line_intersects (line1 line2 : Set Point) (point : Point) : Prop :=
  point ∈ line1 ∧ point ∈ line2

-- Main problem statement
theorem concyclic_points
  (H1 : points_on_circle_in_order A B C D circle)
  (H2 : is_midpoint_of_arc_not_containing S A B C D circle)
  (H3 : line_intersects (line_through S D) (line_through A B) E)
  (H4 : line_intersects (line_through S C) (line_through A B) F) :
  concyclic_points C D E F :=
sorry

end concyclic_points_l628_628192


namespace midpoint_translation_l628_628223

theorem midpoint_translation (x1 y1 x2 y2 tx ty mx my : ℤ) 
  (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 5) (hy2 : y2 = -7)
  (htx : tx = 3) (hty : ty = -4)
  (hmx : mx = (x1 + x2) / 2 + tx) (hmy : my = (y1 + y2) / 2 + ty) : 
  mx = 6 ∧ my = -6 :=
by
  sorry

end midpoint_translation_l628_628223


namespace pipe_B_fill_time_l628_628268

theorem pipe_B_fill_time
  (rate_A : ℝ)
  (rate_B : ℝ)
  (t : ℝ)
  (h_rate_A : rate_A = 2 / 75)
  (h_rate_B : rate_B = 1 / t)
  (h_fill_total : 9 * (rate_A + rate_B) + 21 * rate_A = 1) :
  t = 45 := 
sorry

end pipe_B_fill_time_l628_628268


namespace university_pays_per_box_l628_628285

def box_dimensions : ℝ × ℝ × ℝ := (20, 20, 15)

def total_volume_needed : ℝ := 3.06 * 10^6

def total_cost : ℝ := 663

def volume_of_box : ℝ := let (length, width, height) := box_dimensions in length * width * height

def number_of_boxes : ℝ := total_volume_needed / volume_of_box

def cost_per_box : ℝ := total_cost / number_of_boxes

theorem university_pays_per_box : cost_per_box = 1.30 := by
  sorry

end university_pays_per_box_l628_628285


namespace single_discount_equiv_l628_628324

theorem single_discount_equiv (P : ℝ) (d1 d2 : ℝ) (final_price : ℝ) : 
  P = 1200 ∧ d1 = 0.15 ∧ d2 = 0.10 ∧ final_price = P * (1 - d1) * (1 - d2) → 
  ∃ deq : ℝ, final_price = P * (1 - deq) ∧ deq = 0.235 :=
by
  intros h
  rcases h with ⟨hP, hd1, hd2, hfp⟩
  use 0.235
  split
  · exact hfp
  · sorry

end single_discount_equiv_l628_628324


namespace bob_works_40_hours_per_week_l628_628756

variable (h : ℕ) -- defining the number of hours Bob works per week

-- Defining the necessary conditions
variables 
  (raise : ℕ → ℝ := λ h, 0.50 * h) -- Bob gets a raise of $0.50/hour
  (housing_benefit_reduction : ℝ := 60 / 4) -- Bob's housing benefit reduction per week
  (weekly_extra_earning : ℝ := 5) -- Bob earns $5 more per week after changes

-- The proof problem statement
theorem bob_works_40_hours_per_week (raise_amt : raise h - housing_benefit_reduction = weekly_extra_earning) : h = 40 :=
sorry

end bob_works_40_hours_per_week_l628_628756


namespace radius_of_X_l628_628986

-- Define the types and sizes of the coins
constant CoinLabel : Type
constant C3 : CoinLabel
constant C2 : CoinLabel
constant X : CoinLabel

-- Define the radius function
def radius : CoinLabel → ℝ
| C3 => 3
| C2 => 2
| X  => 0.605  -- Hypothesis for the proof

-- Define the touching relationship between coins
constant touches : CoinLabel → CoinLabel → Prop

-- Define the conditions
axiom condition1 : touches C3 C3
axiom condition2 : touches C2 C3
axiom condition3 : touches X C3
axiom condition4 : touches X C2

-- Prove that the radius of coin X is 0.605 cm
theorem radius_of_X : radius X = 0.605 := sorry

end radius_of_X_l628_628986


namespace prod_coprime_mod_l628_628176

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628176


namespace michael_truck_meet_once_l628_628968

-- Mathematical definitions directly from the problem
def michael_speed : ℝ := 6 -- feet per second
def truck_speed : ℝ := 12 -- feet per second
def distance_between_benches : ℝ := 180 -- feet
def truck_stop_time : ℝ := 40 -- seconds
def initial_truck_distance : ℝ := 180 -- feet

theorem michael_truck_meet_once :
  ∃ t : ℝ, michael_speed * t = truck_speed * t - initial_truck_distance - ∑ k in range (floor (t / (distance_between_benches / truck_speed + truck_stop_time))), (k * truck_speed * truck_stop_time) = 1 := 
sorry

end michael_truck_meet_once_l628_628968


namespace obtuse_triangle_k_values_l628_628635

noncomputable def number_of_valid_k (a b k : ℕ) : ℕ :=
  if a > b then number_of_valid_k b a k else
    -- Case 1: 'b' as the longest side
    (if b ≤ a + k ∧ b * b > a * a + k * k then 1 else 0) +
    -- Case 2: 'k' as the longest side
    if k > 0 then number_of_valid_k a k b else 0

theorem obtuse_triangle_k_values : ∃ k_vals : Finset ℕ, 14 = k_vals.card ∧
  ∀ k ∈ k_vals, 4 < k ∧ k < 30 ∧ 
    ((17 < 13 + k ∧ 17^2 > 13^2 + k^2) ∨ (13 + 17 > k ∧ k^2 > 13^2 + 17^2)) := by
  sorry

end obtuse_triangle_k_values_l628_628635


namespace sequence_sum_l628_628768

theorem sequence_sum (a b : ℕ) (h1 : ∏ i in range (a - 4 + 1), (i + 3) / (i + 2) = 12)
(h2 : b = a - 1) : a + b = 71 := by
  sorry

end sequence_sum_l628_628768


namespace largest_integer_m_l628_628411

theorem largest_integer_m (m : ℤ) : (m^2 - 11 * m + 28 < 0) → m = 6 :=
begin
  sorry
end

end largest_integer_m_l628_628411


namespace total_boundary_length_is_60_nearest_tenth_l628_628204

noncomputable def boundary_length_of_bolded_figure : ℝ :=
  let side_length := real.sqrt 64
  let segment_length := side_length / 4
  let arc_length_sides := 3 * 4 * real.pi
  let arc_length_corners := 2 * real.pi
  let straight_segments_length := 8 * segment_length
  let total_length := straight_segments_length + arc_length_sides + arc_length_corners
  total_length

theorem total_boundary_length_is_60_nearest_tenth : boundary_length_of_bolded_figure ≈ 60.0 :=
by
  sorry

end total_boundary_length_is_60_nearest_tenth_l628_628204


namespace square_of_cube_of_third_smallest_prime_l628_628675

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_smallest_prime (n : ℕ) : ℕ :=
  (List.filter is_prime (List.range (n * n))).nth (n - 1).getD 0

-- The third smallest prime number
def third_smallest_prime : ℕ := nth_smallest_prime 3

-- The cube of a number
def cube (x : ℕ) : ℕ := x * x * x

-- The square of a number
def square (x : ℕ) : ℕ := x * x

theorem square_of_cube_of_third_smallest_prime : square (cube third_smallest_prime) = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628675


namespace pradeep_maximum_marks_l628_628586

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.20 * M = 185) : M = 925 :=
by
  sorry

end pradeep_maximum_marks_l628_628586


namespace error_in_step_one_l628_628298

theorem error_in_step_one : 
  ∃ a b c d : ℝ, 
    (a * (x + 1) - b = c * (x - 2)) = (3 * (x + 1) - 6 = 2 * (x - 2)) → 
    a ≠ 3 ∨ b ≠ 6 ∨ c ≠ 2 := 
by
  sorry

end error_in_step_one_l628_628298


namespace rhombus_area_correct_l628_628616

-- Definitions and conditions
def diagonal1 : ℝ := 30
def diagonal2 : ℝ := 18
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- Prove the area of the rhombus
theorem rhombus_area_correct : area_of_rhombus diagonal1 diagonal2 = 270 := by
  sorry

end rhombus_area_correct_l628_628616


namespace range_of_a_l628_628439

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 > 0
def q (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 > 0
def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x : ℝ, p x → q x ∧ ¬ (q x → p x)

-- The range of values for the positive real number a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, p x) →
  (∃ x : ℝ, q x a) →
  sufficient_but_not_necessary p (λ x, q x a) →
  0 < a ∧ a ≤ 3 :=
begin
  sorry
end

end range_of_a_l628_628439


namespace jeff_costume_payment_l628_628933

theorem jeff_costume_payment (last_year_cost : ℝ) (costume_increase_percent : ℝ) (deposit_percent : ℝ) :
  last_year_cost = 250 → costume_increase_percent = 0.4 → deposit_percent = 0.1 → 
  (let this_year_cost := last_year_cost * (1 + costume_increase_percent) in
  let deposit := this_year_cost * deposit_percent in
  this_year_cost - deposit = 315) :=
begin
  intros,
  sorry
end

end jeff_costume_payment_l628_628933


namespace composite_rate_proof_l628_628278

noncomputable def composite_rate (P A : ℝ) (T : ℕ) (X Y Z : ℝ) (R : ℝ) : Prop :=
  let factor := (1 + X / 100) * (1 + Y / 100) * (1 + Z / 100)
  1.375 = factor ∧ (A = P * (1 + R / 100) ^ T)

theorem composite_rate_proof :
  composite_rate 4000 5500 3 X Y Z 11.1 :=
by sorry

end composite_rate_proof_l628_628278


namespace no_six_digit_numbers_exists_l628_628389

theorem no_six_digit_numbers_exists :
  ¬(∃ (N : Fin 6 → Fin 720), ∀ (a b c : Fin 6), a ≠ b → a ≠ c → b ≠ c →
  (∃ (i : Fin 6), N i == 720)) := sorry

end no_six_digit_numbers_exists_l628_628389


namespace bobik_total_distance_l628_628598

noncomputable def total_distance_bobik_runs
  (vS : ℝ) (vV : ℝ) (d : ℝ) (vB : ℝ) : ℝ :=
  let approach_speed := vS + vV
  let time_to_meet := d / approach_speed
  let distance_ran_by_bobik := vB * time_to_meet
  distance_ran_by_bobik

theorem bobik_total_distance
  (vS : ℝ) (vV : ℝ) (d : ℝ) (vB : ℝ)
  (hs : vS = 4) (hv : vV = 3) (hd : d = 21) (hb : vB = 11) :
  total_distance_bobik_runs vS vV d vB = 33 :=
by {
  rw [hs, hv, hd, hb],
  exact sorry -- This is where the proof would be constructed
}

end bobik_total_distance_l628_628598


namespace num_zeros_f_f_x_l628_628463

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a*x + 1 else Real.log x / Real.log 3

def g (a : ℝ) (x : ℝ) : ℝ :=
  f a (f a x) + 1

theorem num_zeros_f_f_x (a : ℝ) :
  if a > 0 then (∃ x₁ x₂ x₃ x₄ : ℝ, g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧ g a x₄ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₄) 
  else if a < 0 then (∃ x : ℝ, g a x = 0) 
  else true :=
by
  sorry

end num_zeros_f_f_x_l628_628463


namespace quadratic_no_real_roots_l628_628698

theorem quadratic_no_real_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 5) : 
  b^2 - 4 * a * c < 0 :=
by
  rw [h1, h2, h3]
  calc
    (3 : ℝ)^2 - 4 * 1 * 5 = 9 - 20 := by norm_num
    ... = -11 := by norm_num
    ... < 0 := by norm_num

end quadratic_no_real_roots_l628_628698


namespace problem_statement_l628_628533

noncomputable def AD : ℝ := 150 * real.sqrt 2
def greatest_integer_less_than_AD : ℤ := int.floor (150 * real.sqrt 2)

theorem problem_statement : greatest_integer_less_than_AD = 212 := 
by {
  -- Sorry placeholder for actual proof steps
  sorry
}

end problem_statement_l628_628533


namespace sufficient_not_necessary_condition_l628_628775

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, (x^2 - 2 * x < 0 → 0 < x ∧ x < 4)) ∧ (∃ x : ℝ, (0 < x ∧ x < 4) ∧ ¬ (x^2 - 2 * x < 0)) :=
by
  sorry

end sufficient_not_necessary_condition_l628_628775


namespace eisenstein_criterion_irreducible_l628_628712

theorem eisenstein_criterion_irreducible 
  (p : ℕ) [fact (nat.prime p)] 
  (n : ℕ) 
  (a : fin (n) → ℤ) 
  (a₀ : ℤ)
  (h₀ : int.gcd a₀ p = p)
  (h₁ : ∀ i, p ∣ a i) 
  (h₂ : ¬ (p^2 ∣ a₀)) :
  ¬ ∃ g h : polynomial ℤ, 
      g.degree < polynomial.degree (polynomial.X^n + (fin.cons a₀ a) (polynomial.X ^ n))
    ∧ h.degree < polynomial.degree (polynomial.X^n + (fin.cons a₀ a) (polynomial.X ^ n))
    ∧ (polynomial.X^n + (fin.cons a₀ a) (polynomial.X ^ n)) = g * h := sorry

end eisenstein_criterion_irreducible_l628_628712


namespace num_assignment_schemes_l628_628224

-- Define the conditions
def total_volunteers : ℕ := 5
def selected_volunteers : ℕ := 3
def tasks : ℕ := 3

-- Define person A
inductive Volunteer
| A | B | C | D | E

def is_translation (v : Volunteer) : Prop := v = Volunteer.A

-- Proof statement
theorem num_assignment_schemes :
  ∑(n : ℕ) in { if is_translation v then 0 else 1 | v : Volunteer } = 48 :=
sorry

end num_assignment_schemes_l628_628224


namespace polygon_sides_from_diagonals_l628_628031

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628031


namespace terminal_side_in_fourth_quadrant_l628_628063

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ < 0) : 
  (θ : Mathematics.Angle).quadrant = Mathematics.Angle.Quadrant.fourth :=
sorry

end terminal_side_in_fourth_quadrant_l628_628063


namespace area_of_EFGH_l628_628594

-- Defining constants and variables
variables (E F G H : Type) [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace H]
variables (EFG EHG : Triangle) (EF FH HG GH EG : ℝ)

-- Conditions
def right_angles_at_F_and_H := right_angle FH ∧ right_angle HG
def EG_length := EG = 4
def distinct_integer_lengths := ∃ a b : ℕ, (a ≠ b) ∧ (EF = a ∨ FH = a ∨ HG = a ∨ GH = a) ∧ (EF = b ∨ FH = b ∨ HG = b ∨ GH = b)

-- Hypotheses based on the conditions
hypothesis (h1 : right_angles_at_F_and_H)
hypothesis (h2 : EG_length)
hypothesis (h3 : distinct_integer_lengths)

-- Goal
noncomputable def area_of_quadrilateral_EFGH : ℝ := 4 * sqrt 3
theorem area_of_EFGH : (EFG.area + EHG.area = area_of_quadrilateral_EFGH) := sorry

end area_of_EFGH_l628_628594


namespace incorrect_equation_l628_628340

theorem incorrect_equation (x : ℕ) (h : x + 2 * (12 - x) = 20) : 2 * (12 - x) - 20 ≠ x :=
by 
  sorry

end incorrect_equation_l628_628340


namespace problem_statement_l628_628449

theorem problem_statement (n : ℕ) (h : ∀ (a b : ℕ), ¬ (n ∣ (2^a * 3^b + 1))) :
  ∀ (c d : ℕ), ¬ (n ∣ (2^c + 3^d)) := by
  sorry

end problem_statement_l628_628449


namespace find_length_of_c_find_area_of_triangle_l628_628916

-- Definitions for the conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Conditions
def acute_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∠A > 0 ∧ ∠B > 0 ∧ ∠C > 0 ∧ ∠A + ∠B + ∠C = π

-- Part 1: Prove that c = 4
theorem find_length_of_c {a c : ℝ} (h1 : a = 2) (h2 : 2 * Real.sin A = Real.sin C) (h_triangle : acute_triangle a b c A B C) :
  c = 4 := 
by sorry

-- Part 2: Prove that area of the triangle is sqrt(15)
theorem find_area_of_triangle {a c : ℝ} (h1 : a = 2) (h2 : 2 * Real.sin A = Real.sin C) (h3 : Real.cos C = 1 / 4) (h_triangle : acute_triangle a b c A B C) :
  let b := 4 in 
  (1 / 2) * a * b * Real.sin C = sqrt(15) :=
by sorry

end find_length_of_c_find_area_of_triangle_l628_628916


namespace divisible_digit_B_l628_628277

-- Define the digit type as natural numbers within the range 0 to 9.
def digit := {n : ℕ // n <= 9}

-- Define what it means for a number to be even.
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be divisible by 3.
def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define our problem in Lean as properties of the digit B.
theorem divisible_digit_B (B : digit) (h_even : even B.1) (h_div_by_3 : divisible_by_3 (14 + B.1)) : B.1 = 4 :=
sorry

end divisible_digit_B_l628_628277


namespace profit_calculation_l628_628723

open Nat

-- Define the conditions 
def cost_of_actors : Nat := 1200 
def number_of_people : Nat := 50
def cost_per_person_food : Nat := 3
def sale_price : Nat := 10000

-- Define the derived costs
def total_food_cost : Nat := number_of_people * cost_per_person_food
def total_combined_cost : Nat := cost_of_actors + total_food_cost
def equipment_rental_cost : Nat := 2 * total_combined_cost
def total_cost : Nat := cost_of_actors + total_food_cost + equipment_rental_cost
def expected_profit : Nat := 5950 

-- Define the profit calculation
def profit : Nat := sale_price - total_cost 

-- The theorem to be proved
theorem profit_calculation : profit = expected_profit := by
  -- Proof is omitted
  sorry

end profit_calculation_l628_628723


namespace arrange_coprime_1_to_8_l628_628358
open BigOperators

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def adjacent_coprime (l : List ℕ) : Prop :=
  match l with
  | []       => True
  | [_]      => True
  | x :: xs  => List.pairwise is_coprime (x :: xs)

theorem arrange_coprime_1_to_8 : 
  ∃ l : List ℕ, l.perm (List.range 1 9) ∧ adjacent_coprime l ∧ l.length = 8 ∧ (List.number_of_permutations l) = 1728 := sorry

end arrange_coprime_1_to_8_l628_628358


namespace ellipse_eqn_k1_k2_constant_l628_628534

-- Definition of the ellipse with conditions
def ellipse (x y : ℝ) (a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Given constants
variable (a b : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (ab_order : a > b)

-- Given the focal length
axiom focal_length : 2 * sqrt (a^2 - b^2) = 2

-- Right focus of the ellipse
def right_focus : ℝ × ℝ := (√ (a^2 - b^2), 0)

-- Line l passing through the right focus F and perpendicular to the x-axis
def line_l (y : ℝ) := right_focus a b = (√ (a^2 - b^2), y)

-- Area of quadrilateral ACBD
axiom area_ACBD : ∀ (A B C D : ℝ × ℝ), 6 = 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Proving the standard equation of the ellipse
theorem ellipse_eqn : ∀ x y, ellipse x y a b → (a = 2) ∧ (b = sqrt 3) ∧ (ellipse x y 2 (sqrt 3)) := sorry

-- Line passes through fixed point P(1,0) when k2 = 3k1
axiom line_fixed_point : ∀ k1 k2 : ℝ, k2 = 3 * k1 → ∃ p : ℝ × ℝ, p = (1, 0)

-- Prove k1/k2 is constant when passing through the right focus
theorem k1_k2_constant : ∀ k1 k2 : ℝ, (line_l a b k1) ∧ (line_l a b k2) → (k1 / k2 = 1 / 3) := sorry

end ellipse_eqn_k1_k2_constant_l628_628534


namespace sides_of_regular_polygon_with_20_diagonals_l628_628034

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628034


namespace range_of_k_l628_628481

theorem range_of_k (k : ℝ) : 
  (∀ x, x ∈ {x | -3 ≤ x ∧ x ≤ 2} ∩ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1} ↔ x ∈ {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}) →
   -1 ≤ k ∧ k ≤ 1 / 2 :=
by sorry

end range_of_k_l628_628481


namespace tangent_line_intersecting_lines_l628_628829

variable (x y : ℝ)

-- Definition of the circle
def circle_C : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Definition of the point
def point_A : Prop := x = 1 ∧ y = 0

-- (I) Prove that if l is tangent to circle C and passes through A, l is 3x - 4y - 3 = 0
theorem tangent_line (l : ℝ → ℝ) (h : ∀ x, l x = k * (x - 1)) :
  (∀ {x y}, circle_C x y → 3 * x - 4 * y - 3 = 0) :=
by
  sorry

-- (II) Prove that the maximum area of triangle CPQ intersecting circle C is 2, and l's equations are y = 7x - 7 or y = x - 1
theorem intersecting_lines (k : ℝ) :
  (∃ x y, circle_C x y ∧ point_A x y) →
  (∃ k : ℝ, k = 7 ∨ k = 1) :=
by
  sorry

end tangent_line_intersecting_lines_l628_628829


namespace min_value_frac_l628_628453

theorem min_value_frac (x y: ℝ) (hx: x > 0) (hy: y > 0) (h: 2 * x + y = 2) :
  ∃ m : ℝ, m = 8 ∧ (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 
  (frac_one_over_x_squared_plus_four_over_y_squared x y) ≥ m) := sorry


noncomputable def frac_one_over_x_squared_plus_four_over_y_squared (x y : ℝ) : ℝ :=
  (1/(x^2) + 4/(y^2))

end min_value_frac_l628_628453


namespace weavers_in_first_group_l628_628231

theorem weavers_in_first_group :
  (∃ W : ℕ, (W * 4 = 4) ∧ (12 * 12 = 36) ∧ (4 / (W * 4) = 36 / (12 * 12))) -> (W = 4) :=
by
  sorry

end weavers_in_first_group_l628_628231


namespace complementary_angles_difference_l628_628627

-- Given that the measures of two complementary angles are in the ratio 4:1,
-- we want to prove that the positive difference between the measures of the two angles is 54 degrees.

theorem complementary_angles_difference (x : ℝ) (h_complementary : 4 * x + x = 90) : 
  abs (4 * x - x) = 54 :=
by
  sorry

end complementary_angles_difference_l628_628627


namespace color_polygon_2003_l628_628375

theorem color_polygon_2003 : ∃ T : ℕ, 
  (∀ (n : ℕ), n = 2003 → T = 2^n + 2) ∧
  (T = 2^2003 + 2) :=
by
  use 2^2003 + 2
  split
  { intros n hn
    rw [hn]
    refl }
  { refl }

end color_polygon_2003_l628_628375


namespace find_pq_l628_628250

-- Given definitions based on conditions
def parabola (p x : ℝ) : ℝ := p * x - x^2
def hyperbola (x y q : ℝ) : Prop := x * y = q
def centroid_distance (p : ℝ) : ℝ := abs (p / 3)

theorem find_pq (p q x1 x2 x3 y1 y2 y3 : ℝ) 
    (h1 : ∀ x, parabola p x = y1 ∨ parabola p x = y2 ∨ parabola p x = y3) 
    (h2 : hyperbola x1 y1 q ∧ hyperbola x2 y2 q ∧ hyperbola x3 y3 q)
    (h3 : centroid_distance p = 2)
    (h4 : (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + (y3 - y1)^2 = 324): 
    p * q = 42 := 
sorry

end find_pq_l628_628250


namespace correct_sampling_statements_l628_628263

-- Conditions
def population : Type := List Athlete
def athlete := {a : Athlete // a ∈ population}
def sample_size := 100

-- Assuming we have a sampling method
def stratified_sampling (lst : List Athlete) : List (List Athlete) :=
  sorry -- Placeholder definition

-- Statements to be proved
def correct_statements : Prop :=
  (sample_size = 100) ∧
  (∃ (sample : List Athlete), stratified_sampling sample = [sample]) ∧
  (∀ (a : Athlete), a ∈ population → a ∈ sample)

-- Main theorem
theorem correct_sampling_statements :
  correct_statements :=
sorry

end correct_sampling_statements_l628_628263


namespace range_of_f_l628_628452

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(2 * k)

theorem range_of_f {k : ℝ} (h : k ≥ 1) : 
  set.range (λ (x : ℝ), f x k) ∩ set.Ici 2 = set.Ici 4 := 
by
  sorry

end range_of_f_l628_628452


namespace product_of_invertibles_mod_120_l628_628151

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628151


namespace max_remainder_2015_largest_possible_remainder_2015_l628_628942

theorem max_remainder_2015 (m : ℕ) (n : ℕ) (h1 : m > 0) (h2 : m < 2015) (h3 : 2015 % m = n) : n ≤ m - 1 :=
by sorry

theorem largest_possible_remainder_2015 : ∃ m n, m > 0 ∧ m < 2015 ∧ 2015 % m = n ∧ n = 1007 :=
by {
  use 1008,
  use 1007,
  split,
  exact dec_trivial,
  split,
  linarith,
  split,
  exact dec_trivial,
  refl,
}

end max_remainder_2015_largest_possible_remainder_2015_l628_628942


namespace interval_lengths_l628_628382

def floor (x : ℝ) : ℤ := int.floor x

def fractional_part (x : ℝ) : ℝ := x - int.floor x

def f (x : ℝ) : ℝ := (int.floor x : ℝ) * (x - (int.floor x : ℝ))

def g (x : ℝ) : ℝ := 2*x - (int.floor x : ℝ) - 2

theorem interval_lengths :
  let d1 := (λ I : set ℝ, I.sup_sub - I.inf_sub) (set_of (λ x : ℝ, f x > g x))
  let d2 := (λ I : set ℝ, I.sup_sub - I.inf_sub) (set_of (λ x : ℝ, f x = g x))
  let d3 := (λ I : set ℝ, I.sup_sub - I.inf_sub) (set_of (λ x : ℝ, f x < g x))
  (0 ≤ d1 ∧ 0 ≤ d2 ∧ 0 ≤ d3 ∧ d1 + d2 + d3 = 2012) ∧ (d1 = 2 ∧ d2 = 1 ∧ d3 = 2009) := by {
  sorry
}

end interval_lengths_l628_628382


namespace jenny_spent_fraction_l628_628550

theorem jenny_spent_fraction
  (x : ℝ) -- The original amount of money Jenny had
  (h_half_x : 1/2 * x = 21) -- Half of the original amount is $21
  (h_left_money : x - 24 = 24) -- Jenny had $24 left after spending
  : (x - 24) / x = 3 / 7 := sorry

end jenny_spent_fraction_l628_628550


namespace finns_total_cost_l628_628781

theorem finns_total_cost :
  let cost_paper_clip := 1.85 in
  let num_paper_clips_eldora := 15 in
  let num_index_cards_eldora := 7 in
  let total_cost_eldora := 55.40 in
  let num_paper_clips_finn := 12 in
  let num_index_cards_finn := 10 in
  let cost_index_card := (total_cost_eldora - num_paper_clips_eldora * cost_paper_clip) / num_index_cards_eldora in
  let cost_paper_clips_finn := num_paper_clips_finn * cost_paper_clip in
  let cost_index_cards_finn := num_index_cards_finn * cost_index_card in
  let total_cost_finn := cost_paper_clips_finn + cost_index_cards_finn in
  total_cost_finn = 61.70 :=
by
   sorry

end finns_total_cost_l628_628781


namespace increasing_interval_l628_628247

noncomputable def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2

theorem increasing_interval :
  is_monotonic_increasing y π (2 * π) :=
by
  -- Proof would go here
  sorry

end increasing_interval_l628_628247


namespace product_of_invertibles_mod_120_l628_628149

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628149


namespace largest_integer_value_neg_quadratic_l628_628413

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end largest_integer_value_neg_quadratic_l628_628413


namespace product_of_sums_l628_628658

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

theorem product_of_sums (h1 : ∀ n, 20 ≤ n ∧ n ≤ 30 → is_prime n ∨ is_composite n)
  (h2 : ∑ n in {n | 20 ≤ n ∧ n ≤ 30 ∧ is_prime n}.to_finset, n = 52)
  (h3 : ∑ n in {n | 20 ≤ n ∧ n ≤ 30 ∧ is_composite n}.to_finset, n = 213):
  52 * 213 = 11076 := by
  sorry

end product_of_sums_l628_628658


namespace rhombus_area_correct_l628_628617

-- Definitions and conditions
def diagonal1 : ℝ := 30
def diagonal2 : ℝ := 18
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- Prove the area of the rhombus
theorem rhombus_area_correct : area_of_rhombus diagonal1 diagonal2 = 270 := by
  sorry

end rhombus_area_correct_l628_628617


namespace students_picking_uniforms_l628_628426

theorem students_picking_uniforms :
  (∑ s in finset.powerset_len 2 (finset.range 5), card (derangements {x | ¬(x ∈ s)})) = 20 := by {
  sorry
}

end students_picking_uniforms_l628_628426


namespace sum_counts_equal_l628_628644

noncomputable theory

def weights := (Fin 200) → ℕ -- Mapping from indices to weights
def left_pan (w: weights) := (Fin 100) → Fin 200 -- Elements in the left pan
def right_pan (w: weights) := (Fin 100) → Fin 200 -- Elements in the right pan

def is_balanced (w: weights) (l r: Fin 200 → Bool) : Prop :=
  (∑ i, ite (l i) (w i) 0) = (∑ i, ite (r i) (w i) 0)

def count_opposite_lighter (w: weights) (side: Fin 200 → Bool) (i: Fin 200) : ℕ :=
  (∑ j, ite ((¬ side j) ∧ (w j < w i)) 1 0)

theorem sum_counts_equal (w: weights) (l r: Fin 200 → Bool) (h_balanced : is_balanced w l r) : 
  (∑ i, ite (l i) (count_opposite_lighter w r i) 0) = 
  (∑ i, ite (r i) (count_opposite_lighter w l i) 0) := 
sorry

end sum_counts_equal_l628_628644


namespace median_salary_is_25000_l628_628850

-- Defining the data for the problem
def num_ceo := 1
def num_svp := 4
def num_mgr := 12
def num_leader := 8
def num_assistant := 38

def salary_ceo := 140000
def salary_svp := 95000
def salary_mgr := 80000
def salary_leader := 55000
def salary_assistant := 25000

-- Total number of employees
def total_employees := num_ceo + num_svp + num_mgr + num_leader + num_assistant

-- Defining the position of the median
def median_position := (total_employees + 1) / 2

-- Proving the median salary
theorem median_salary_is_25000 : median_position <= total_employees ∧ median_position ≥ 1 → 
  salary_assistant = 25000 := by
suffices h : salary_assistant = 25000 by
  sorry

end median_salary_is_25000_l628_628850


namespace number_of_female_students_in_school_l628_628730

-- Let n be the total number of students in the school
def total_students := 1600

-- The size of the sample
def sample_size := 200

-- Let x be the number of sampled girls
variable (x : ℕ)

-- Let the number of sampled boys be x + 10
def sampled_boys := x + 10

-- Sample conditions
def sample_condition := x + sampled_boys x = sample_size

-- The ratio of the sample to the total population
def sample_ratio := 1 / 8

-- Condition for the number of female students in the school
def female_students := 95 * 8

-- The proof statement
theorem number_of_female_students_in_school : 
  (let x : ℕ := 95 in sample_condition x ∧ sample_ratio = sample_size / total_students) → female_students = 760 := 
by
  intros h
  sorry

end number_of_female_students_in_school_l628_628730


namespace obtuse_triangle_k_values_l628_628637

theorem obtuse_triangle_k_values :
  let valid_k_values := 
    List.range' 5 6 ++         -- Range [5, 6, 7, 8, 9, 10]
    List.range' 22 8 in        -- Range [22, 23, 24, 25, 26, 27, 28, 29]
  valid_k_values.count = 14 :=
by
  let values_with_17_as_max := List.range' 5 6   -- [5, 6, 7, 8, 9, 10]
  let values_with_k_as_max := List.range' 22 8  -- [22, 23, 24, 25, 26, 27, 28, 29]
  let all_valid_values := values_with_17_as_max ++ values_with_k_as_max
  have count_values : all_valid_values.count = 14,
    from sorry
  exact count_values

end obtuse_triangle_k_values_l628_628637


namespace number_of_elements_in_A_l628_628225

open Set

variables (A B C : Set α)
variables (a b c : ℕ)
variables (h_a : a = 3 * b)
variables (h_intersection_AB : (A ∩ B).card = 1200)
variables (h_union_ABC : (A ∪ B ∪ C).card = 4200)
variables (h_intersection_AC_not_B : (A ∩ C).card = 300)

theorem number_of_elements_in_A : a = 3825 :=
  sorry

end number_of_elements_in_A_l628_628225


namespace sum_of_thetas_l628_628638

noncomputable def theta (k : ℕ) : ℝ := (54 + 72 * k) % 360

theorem sum_of_thetas : (theta 0 + theta 1 + theta 2 + theta 3 + theta 4) = 990 :=
by
  -- proof goes here
  sorry

end sum_of_thetas_l628_628638


namespace measure_diagonal_of_brick_l628_628967

def RectangularParallelepiped (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def DiagonalMeasurementPossible (a b c : ℝ) : Prop :=
  ∃ d : ℝ, d = (a^2 + b^2 + c^2)^(1/2)

theorem measure_diagonal_of_brick (a b c : ℝ) 
  (h : RectangularParallelepiped a b c) : DiagonalMeasurementPossible a b c :=
by
  sorry

end measure_diagonal_of_brick_l628_628967


namespace savings_per_bagel_in_cents_l628_628609

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_per_bagel_in_cents_l628_628609


namespace find_angle_A_max_area_triangle_ABC_l628_628843

-- Definitions for the given conditions
variables (a b c A B C : ℝ) (ABC : Type) [triangle ABC]
variables (D : point) (BC : line)
variable  h1 : sideoptoanglesABC ABC a b c

-- Condition: \( (2b - c) \cos A = a \cos C \)
variable  h2 : (2 * b - c) * cos A = a * cos C
-- Condition: \( D \) is on side \( BC \) such that \( BD = 2DC \)
variable  h3 : on_side D BC ∧ BD = 2 * DC
-- Condition: \( AD = 2 \)
variable  h4 : length (line A D) = 2

-- Statement for the first proof
theorem find_angle_A (h1 h2 : h1) : A = π / 3 := 
sorry

-- Statement for the second proof
theorem max_area_triangle_ABC (h1 h2 h3 h4 : h1 h2 h3 h4) : area_triangle_ABC ABC = (3 * sqrt 3) / 2 := 
sorry

end find_angle_A_max_area_triangle_ABC_l628_628843


namespace candidates_scoring_between_60_and_100_l628_628897

noncomputable def normal_dist (x : ℝ) (μ σ : ℝ) : ℝ :=
  (1 / (σ * real.sqrt (2 * real.pi))) * real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2))

theorem candidates_scoring_between_60_and_100 :
  let μ := 80
  let σ := 10
  let lower_bound := μ - 2 * σ
  let upper_bound := μ + 2 * σ
  normal_dist 60 μ σ = 0.9544 ∧ normal_dist 100 μ σ = 0.9544
  := by
  sorry

end candidates_scoring_between_60_and_100_l628_628897


namespace parabola_open_upwards_l628_628852

theorem parabola_open_upwards 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0)
  (h4 : a > 0)
  (h5 : c < 0) : 
  ∀ x ∈ Ioo 0 1, (λ x : ℝ, a * x^2 + b * x + c) x < 0 :=
by 
  sorry

end parabola_open_upwards_l628_628852


namespace find_delta_l628_628505

-- Define the simultaneous equations condition
def exists_solution (x y z : ℝ) : Prop :=
  x - y - z = -1 ∧ y - x - z = -2 ∧ z - x - y = -4

-- Define β as a specific remainder
def beta (α : ℕ) : ℕ := 5 -- assuming the equivalent simplified result from conditions

-- Define γ as the remainder of the specified polynomial expression modulo 3
def gamma : ℕ := 2 -- determined from calculation

-- Define δ based on the polynomial's property
theorem find_delta (α : ℕ) (x y z : ℝ) (a b : ℝ) (δ : ℝ) :
  exists_solution x y z ∧
  beta α = 5 ∧
  gamma = 2 ∧
  (1 + γ + γ^2 + (-7) = 0) ∧
  (1 * γ * γ^2 * (-7) = δ) → δ = -56 :=
by
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h7 h8,
  sorry

end find_delta_l628_628505


namespace semi_minor_axis_l628_628902

theorem semi_minor_axis (a c : ℝ) (h_a : a = 5) (h_c : c = 2) : 
  ∃ b : ℝ, b = Real.sqrt (a^2 - c^2) ∧ b = Real.sqrt 21 :=
by
  use Real.sqrt 21
  sorry

end semi_minor_axis_l628_628902


namespace find_interest_rate_l628_628331

-- Define the conditions given in the problem
def A : ℝ := 4893
def P : ℝ := 4079.325870140444
def n : ℝ := 1
def t : ℝ := 3

-- State the problem as a Lean 4 theorem
theorem find_interest_rate :
  ∃ r : ℝ, (P * (1 + r / n) ^ (n * t) = A) ∧ (r ≈ 0.0633 * 100) :=
by
  -- The proof will be completed here
  sorry

end find_interest_rate_l628_628331


namespace origin_symmetric_cosine_l628_628622

variable (A ω : ℝ) (k : ℤ)

def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def cos_func (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.cos (ω * x + φ)

theorem origin_symmetric_cosine (φ : ℝ) :
  symmetric_about_origin (cos_func A ω φ) ↔ ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 :=
  sorry

end origin_symmetric_cosine_l628_628622


namespace savings_per_bagel_in_cents_l628_628607

def cost_of_one_bagel : ℝ := 2.25
def cost_of_dozen_bagels : ℝ := 24.0
def number_of_bagels_in_dozen : ℕ := 12

theorem savings_per_bagel_in_cents :
  let total_cost_individual := number_of_bagels_in_dozen * cost_of_one_bagel in
  let total_savings_dollar := total_cost_individual - cost_of_dozen_bagels in
  let savings_per_bagel_dollar := total_savings_dollar / number_of_bagels_in_dozen in
  let savings_per_bagel_cents := savings_per_bagel_dollar * 100 in
  savings_per_bagel_cents = 25 :=
begin
  sorry
end

end savings_per_bagel_in_cents_l628_628607


namespace product_of_invertible_integers_mod_120_l628_628142

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628142


namespace min_value_3x_4y_l628_628509

noncomputable def minValue (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem min_value_3x_4y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 3 * y = 5 * x * y) : 
  minValue x y ≥ 5 :=
sorry

end min_value_3x_4y_l628_628509


namespace broken_line_intersection_l628_628440

variables {n : ℕ}
variables {A : Fin n → Type*} {B : Fin n → Type*}
variables (A B : Fin n → Type*)

-- Closed spatial broken line intersected by a plane at points B
theorem broken_line_intersection (h: ∀ i : Fin n, ∃ (Bi: B i), true) :
  (Finset.prod (Finset.range n) (λ i : Fin n,
    let B₁ := classical.some (h i)
    let B₂ := classical.some (h ((i + 1) % n))
    in (dist A[i] B₁ / dist B₁ A[(i + 1) % n])
  ) = 1) :=
sorry

end broken_line_intersection_l628_628440


namespace square_product_area_perimeter_l628_628582

open Real

-- Define the positions of the points P, Q, R, S on the grid
def P : ℝ × ℝ := (1, 5)
def Q : ℝ × ℝ := (5, 6)
def R : ℝ × ℝ := (6, 2)
def S : ℝ × ℝ := (2, 1)

-- Function to calculate the Euclidean distance between two points
def distance (a b : ℝ × ℝ) : ℝ :=
  sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- The side length of the square PQRS (using P and Q as example, same for others)
def side_length : ℝ := distance P Q

-- The area of the square
def area : ℝ := side_length ^ 2

-- The perimeter of the square
def perimeter : ℝ := 4 * side_length

-- The product of the area and the perimeter
def product : ℝ := area * perimeter

-- Statement to prove
theorem square_product_area_perimeter :
  product = 68 * sqrt 17 := sorry

end square_product_area_perimeter_l628_628582


namespace base_7_to_base_10_equivalent_l628_628378

theorem base_7_to_base_10_equivalent :
  let n := 3216
  let base := 7
  n.toNatBase base = 1140 :=
by sorry

end base_7_to_base_10_equivalent_l628_628378


namespace closest_point_on_line_l628_628794

def closestPoint (P Q : ℝ × ℝ) (L : ℝ → ℝ → Prop) :=
  L Q.1 Q.2 ∧ ∀ (R : ℝ × ℝ), L R.1 R.2 → dist P Q ≤ dist P R

theorem closest_point_on_line :
  let line := λ x y, y = (x + 2) / 3
  let point := (5, -1) : ℝ × ℝ
  let closest := (4, 2) : ℝ × ℝ
  closestPoint point closest line :=
by
  -- proof goes here
  sorry

end closest_point_on_line_l628_628794


namespace angle_between_asymptotes_l628_628468

noncomputable def hyperbola_asymptotes_angle
  (a b : ℝ)
  (eccentricity : ℝ)
  (h_eq : x^2/a^2 - y^2/b^2 = 1)
  (h_ecc : eccentricity = √2) :
  ℝ :=
π / 2

theorem angle_between_asymptotes
  (a b : ℝ)
  (h_eq : x^2/a^2 - y^2/b^2 = 1)
  (h_ecc : (a^2 + b^2)/ a^2 = 2)
  : hyperbola_asymptotes_angle a b (√2) h_eq h_ecc = π / 2 :=
by sorry

end angle_between_asymptotes_l628_628468


namespace find_angle_CED_l628_628112

-- Given conditions
variables (O A B E C D : Type)
variables [circle_center O] [diameter O A B] [on_circle E] 
variables [tangent_at B O C] [tangent_at A O D] [line_segment A E D]
variables [angle BAE : Real]

-- Given angle measurement
axiom angle_BAE_is_53 : angle BAE = 53

-- Target to prove
theorem find_angle_CED : ∀ (angle CED : Real), angle CED = 37 :=
begin
  sorry  -- Proof goes here
end

end find_angle_CED_l628_628112


namespace complex_fraction_simplification_l628_628370

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : (2 : ℂ) / (1 + i)^2 = i :=
by 
-- this will be filled when proving the theorem in Lean
sorry

end complex_fraction_simplification_l628_628370


namespace non_congruent_integer_triangles_with_perimeter_20_l628_628491

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_20 (a b c : ℕ) : Prop :=
  a + b + c = 20

def distinct_triples (T : Set (ℕ × ℕ × ℕ)) (a b c : ℕ) : Prop :=
  ∀ x ∈ T, x ≠ (a, b, c)

theorem non_congruent_integer_triangles_with_perimeter_20 :
  ∃ (T : Set (ℕ × ℕ × ℕ)), (∀ (a b c : ℕ), (a, b, c) ∈ T → is_triangle a b c ∧ perimeter_20 a b c) ∧ 
  (∀ (a b c : ℕ), is_triangle a b c → perimeter_20 a b c → (a, b, c) ∈ T) ∧ 
  (∀ (x y : (ℕ × ℕ × ℕ)), x ∈ T → y ∈ T → x ≠ y) ∧ 
  T.card = 11 :=
by
  sorry

end non_congruent_integer_triangles_with_perimeter_20_l628_628491


namespace product_invertibles_mod_120_l628_628163

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628163


namespace divide_real_set_l628_628587

theorem divide_real_set (S : Finset ℝ) :
  ∃ G1 G2 : Finset ℝ, S = G1 ∪ G2 ∧ G1 ∩ G2 = ∅ ∧
  (∀ a b ∈ G1, ∀ k : ℤ, a ≠ b → |a - b| ≠ 3 ^ k) ∧
  (∀ a b ∈ G2, ∀ k : ℤ, a ≠ b → |a - b| ≠ 3 ^ k) :=
by sorry

end divide_real_set_l628_628587


namespace larger_number_is_23_l628_628652

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end larger_number_is_23_l628_628652


namespace cos_inequality_le_mn_squared_l628_628217

theorem cos_inequality_le_mn_squared (A B : ℝ) (m n : ℕ) :
  abs ((cos (m * A) * cos (n * B) - cos (n * A) * cos (m * B)) / (cos A - cos B)) ≤ abs (m^2 - n^2) :=
by
  sorry

end cos_inequality_le_mn_squared_l628_628217


namespace all_lines_intersect_at_single_point_l628_628273

-- Define the finite set of lines within a 2-dimensional space.
variables {L : set (set (ℝ × ℝ))}
variable h_fin : L.finite

-- Define the condition that no two lines are parallel.
variable h_no_parallel : ∀ l₁ l₂ ∈ L, l₁ ≠ l₂ → ¬ parallel l₁ l₂

-- Define the condition that at each point of intersection, at least three lines pass through it.
variable h_three_lines_at_intersection : ∀ P ∈ ⋃ l ∈ L, l, ∃ L : finset (set (ℝ × ℝ)), L ⊆ L ∧ P ∈ ⋂₀ L ∧ 3 ≤ L.card

-- The theorem to be proven.
theorem all_lines_intersect_at_single_point : ∃ P, ∀ l ∈ L, P ∈ l :=
by
  sorry

end all_lines_intersect_at_single_point_l628_628273


namespace regular_polygon_num_sides_l628_628043

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628043


namespace prove_solutions_l628_628800

noncomputable def solution1 (x : ℝ) : Prop :=
  3 * x^2 + 6 = abs (-25 + x)

theorem prove_solutions :
  solution1 ( (-1 + Real.sqrt 229) / 6 ) ∧ solution1 ( (-1 - Real.sqrt 229) / 6 ) :=
by
  sorry

end prove_solutions_l628_628800


namespace perfect_square_of_polynomial_l628_628023

theorem perfect_square_of_polynomial (k : ℝ) (h : ∃ (p : ℝ), ∀ x : ℝ, x^2 + 6*x + k^2 = (x + p)^2) : k = 3 ∨ k = -3 := 
sorry

end perfect_square_of_polynomial_l628_628023


namespace r_add_s_correct_l628_628249

noncomputable def r_and_s_sum : ℕ :=
  let b := \frac{r}{s} in
  if (r, s : ℕ), (coprime r s ∧ ∑ (y : ℝ) in (λ y, (⌊y⌋ * (y - ⌊y⌋)) = b * y^2), (y : ℝ) = 360) then
    r + s
  else
    0

theorem r_add_s_correct : r_and_s_sum = 861 :=
sorry

end r_add_s_correct_l628_628249


namespace set_P_equals_set_interval_l628_628570

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x <= 1 ∨ x >= 3}
def P : Set ℝ := {x | x ∈ A ∧ ¬ (x ∈ A ∧ x ∈ B)}

theorem set_P_equals_set_interval :
  P = {x | 1 < x ∧ x < 3} :=
sorry

end set_P_equals_set_interval_l628_628570


namespace coeff_of_x_l628_628773

-- Defining the initial expression
def expr : ℚ[X] := 2 * (X - 5) + 5 * (8 - 3 * X^2 + 6 * X) - 9 * (3 * X - 2)

-- Statement to assert that the coefficient of x in the simplified expression is 5
theorem coeff_of_x : (expr.coeff 1) = 5 := by
  sorry

end coeff_of_x_l628_628773


namespace geoboard_pentagon_area_l628_628772

def points : List (ℝ × ℝ) := [(2,1), (1,4), (4,5), (7,2), (5,0)]

noncomputable def shoelace_area (points : List (ℝ × ℝ)) : ℝ :=
  let n := points.length
  let cyclic_points := points ++ [points.head!]
  let sum1 := ∑ i in Finset.range n, (cyclic_points[i].1 * cyclic_points[i+1].2)
  let sum2 := ∑ i in Finset.range n, (cyclic_points[i].2 * cyclic_points[i+1].1)
  0.5 * |sum1 - sum2|

theorem geoboard_pentagon_area : shoelace_area points = 18 :=
  by
  -- The proof would be provided here
  sorry

end geoboard_pentagon_area_l628_628772


namespace product_of_invertible_integers_mod_120_l628_628140

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628140


namespace polygon_sides_from_diagonals_l628_628028

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628028


namespace cube_sum_l628_628884

theorem cube_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end cube_sum_l628_628884


namespace product_of_invertibles_mod_120_l628_628168

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628168


namespace square_of_negative_eq_square_l628_628759

theorem square_of_negative_eq_square (a : ℝ) : (-a)^2 = a^2 :=
sorry

end square_of_negative_eq_square_l628_628759


namespace log_exponential_eq_l628_628026

theorem log_exponential_eq (x : ℝ) : 
  (real.log (x^2 - 5 * x + 8) = 2) → 
  (x ≈ 4.874 ∨ x ≈ 0.1255) :=
sorry

end log_exponential_eq_l628_628026


namespace min_value_f_zero_a_range_a_geq_one_range_a_fraction_l628_628466

-- (1) Minimum value of f(x) on the interval [1/2, 1] when a = 0
theorem min_value_f_zero_a :
  ∃ x ∈ (Set.Icc (1/2: ℝ) 1), (x * Exp.exp (2*x) - Real.log x) = (Real.exp 1 / 2 + Real.log 2) := sorry

-- (2) Range of a such that f(x) ≥ 1 for all x > 0
theorem range_a_geq_one (a: ℝ) :
  (∀ x > 0, x * Exp.exp (2*x) - Real.log x - a * x ≥ 1) → a ≤ 2 := sorry

-- (3) Range of a such that f(1/x) - 1 ≥ ∃ holds for all x > 0
theorem range_a_fraction (a: ℝ) :
  (∀ x > 0, (1/x * Exp.exp (2/x) - Real.log (1/x) - a / x - 1) ≥ (1/x * Exp.exp (2/x)) + ((1/(Exp.exp 1 - 1) + 1/x) / Exp.exp (x/Exp.exp 1))) →
  a ≤ -1 - Real.exp 1 / ((Real.exp 1 - 1) * Exp.exp (1/Real.exp 1)) := sorry

end min_value_f_zero_a_range_a_geq_one_range_a_fraction_l628_628466


namespace cylindrical_coordinates_cone_shape_l628_628532

def cylindrical_coordinates := Type

def shape_description (r θ z : ℝ) : Prop :=
θ = 2 * z

theorem cylindrical_coordinates_cone_shape (r θ z : ℝ) :
  shape_description r θ z → θ = 2 * z → Prop := sorry

end cylindrical_coordinates_cone_shape_l628_628532


namespace tray_height_l628_628732

noncomputable def height_of_tray : ℝ :=
  let side_length := 120
  let cut_distance := 4 * Real.sqrt 2
  let angle := 45 * (Real.pi / 180)
  -- Define the function that calculates height based on given conditions
  
  sorry

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  side_length = 120 ∧ cut_distance = 4 * Real.sqrt 2 ∧ angle = 45 * (Real.pi / 180) →
  height_of_tray = 4 * Real.sqrt 2 :=
by
  intros
  unfold height_of_tray
  sorry

end tray_height_l628_628732


namespace work_days_of_a_l628_628309

variable (da wa wb wc : ℕ)
variable (hcp : 3 * wc = 5 * wa)
variable (hbw : 4 * wc = 5 * wb)
variable (hwc : wc = 100)
variable (hear : 60 * da + 9 * 80 + 4 * 100 = 1480)

theorem work_days_of_a : da = 6 :=
by
  sorry

end work_days_of_a_l628_628309


namespace find_x_l628_628025

-- Definitions of the conditions
def eq1 (x y z : ℕ) : Prop := x + y + z = 25
def eq2 (y z : ℕ) : Prop := y + z = 14

-- Statement of the mathematically equivalent proof problem
theorem find_x (x y z : ℕ) (h1 : eq1 x y z) (h2 : eq2 y z) : x = 11 :=
by {
  -- This is where the proof would go, but we can omit it for now:
  sorry
}

end find_x_l628_628025


namespace max_parts_with_4_lines_l628_628391

-- Define L(n) according to the given conditions
def L : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 4
| (n+3) := L (n + 2) + (n + 3)

-- The theorem to prove the maximum number of parts is 11 when n = 4
theorem max_parts_with_4_lines : L 4 = 11 := by
  sorry

end max_parts_with_4_lines_l628_628391


namespace non_congruent_integer_triangles_with_perimeter_20_l628_628490

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_20 (a b c : ℕ) : Prop :=
  a + b + c = 20

def distinct_triples (T : Set (ℕ × ℕ × ℕ)) (a b c : ℕ) : Prop :=
  ∀ x ∈ T, x ≠ (a, b, c)

theorem non_congruent_integer_triangles_with_perimeter_20 :
  ∃ (T : Set (ℕ × ℕ × ℕ)), (∀ (a b c : ℕ), (a, b, c) ∈ T → is_triangle a b c ∧ perimeter_20 a b c) ∧ 
  (∀ (a b c : ℕ), is_triangle a b c → perimeter_20 a b c → (a, b, c) ∈ T) ∧ 
  (∀ (x y : (ℕ × ℕ × ℕ)), x ∈ T → y ∈ T → x ≠ y) ∧ 
  T.card = 11 :=
by
  sorry

end non_congruent_integer_triangles_with_perimeter_20_l628_628490


namespace non_congruent_triangles_perimeter_18_l628_628013

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628013


namespace product_invertibles_mod_120_l628_628157

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628157


namespace bank_deposit_exceeds_1000_on_saturday_l628_628558

theorem bank_deposit_exceeds_1000_on_saturday:
  ∃ n: ℕ, (2 * (3^n - 1) / 2 > 1000) ∧ ((n + 1) % 7 = 0) := by
  sorry

end bank_deposit_exceeds_1000_on_saturday_l628_628558


namespace average_minutes_per_player_is_2_l628_628211

def total_player_footage := 130 + 145 + 85 + 60 + 180
def total_additional_content := 120 + 90 + 30
def pause_transition_time := 15 * (5 + 3) -- 5 players + game footage + interviews + opening/closing scenes - 1
def total_film_time := total_player_footage + total_additional_content + pause_transition_time
def number_of_players := 5
def average_seconds_per_player := total_player_footage / number_of_players
def average_minutes_per_player := average_seconds_per_player / 60

theorem average_minutes_per_player_is_2 :
  average_minutes_per_player = 2 := by
  -- Proof goes here.
  sorry

end average_minutes_per_player_is_2_l628_628211


namespace sugar_cups_used_l628_628553

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l628_628553


namespace ratio_of_areas_l628_628266

-- Define the regular octagon and the properties of tangency 
def regular_octagon := {s: ℝ // s = 1} -- We assume a side length 1 as in the solution

-- Define the radii of the circles based on given tangency properties
def radius_first_circle (s : ℝ) : ℝ := s / (2 * real.tan (real.pi / 8))
def radius_second_circle (s : ℝ) : ℝ := s / (2 * real.tan (real.pi / 8))

-- Define the areas of the circles
def area_circle (r : ℝ) : ℝ := real.pi * r^2

-- Prove the ratio of the areas is 1
theorem ratio_of_areas (s : ℝ) (h : s = 1) : 
  (area_circle (radius_second_circle s)) / (area_circle (radius_first_circle s)) = 1 :=
by
  sorry

end ratio_of_areas_l628_628266


namespace eugene_total_cost_l628_628907

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l628_628907


namespace product_eq_1519000000_div_6561_l628_628808

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end product_eq_1519000000_div_6561_l628_628808


namespace sufficient_condition_for_one_positive_and_one_negative_root_l628_628639

theorem sufficient_condition_for_one_positive_and_one_negative_root (a : ℝ) (h₀ : a ≠ 0) :
  a < -1 ↔ (∃ x y : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ (a * y^2 + 2 * y + 1 = 0) ∧ x > 0 ∧ y < 0) :=
by {
  sorry
}

end sufficient_condition_for_one_positive_and_one_negative_root_l628_628639


namespace simplified_product_sequence_l628_628599

theorem simplified_product_sequence : 
  (∏ k in (Finset.range 503).map (Finset.natEmb (+) 1), (5*k + 5) / (5*k)) = 504 :=
by
  sorry

end simplified_product_sequence_l628_628599


namespace modulus_of_complex_number_l628_628066

noncomputable def pure_imaginary (z : ℂ) : Prop :=
z.re = 0

theorem modulus_of_complex_number (m : ℝ) (H : pure_imaginary (1 + complex.I * m) ∧ pure_imaginary ((1 + complex.I * m) * (3 + complex.I) * complex.I)):
  complex.abs ((m + 3 * complex.I) / (1 - complex.I)) = 3 := 
sorry

end modulus_of_complex_number_l628_628066


namespace false_dist_half_radius_l628_628199

variable {A B : Type} [MetricSpace A] [MetricSpace B]
variable {P Q : A} {p q d : ℝ}
variable (h1 : q = p / 2) (h2 : dist P Q = d)

theorem false_dist_half_radius :
  ¬(dist P Q = p / 2) :=
begin
  -- proof would go here
  sorry,
end

end false_dist_half_radius_l628_628199


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l628_628964

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l628_628964


namespace ratio_cookies_to_pie_l628_628711

def num_surveyed_students : ℕ := 800
def num_students_preferred_cookies : ℕ := 280
def num_students_preferred_pie : ℕ := 160

theorem ratio_cookies_to_pie : num_students_preferred_cookies / num_students_preferred_pie = 7 / 4 := by
  sorry

end ratio_cookies_to_pie_l628_628711


namespace sum_of_segments_altitude_proportionality_l628_628631

-- Define the necessary geometrical objects and their properties
noncomputable theory

variables {A B C M A1 B1 C1 : Point}
variables {R r : ℝ}

-- Assuming the orthocenter of triangle ABC is M, and A1, B1, C1 are the feet of the altitudes from A, B, and C respectively.

def orthocenter_of (A B C : Point) (M : Point) : Prop :=
  is_orthocenter A B C M

def feet_of_altitudes (A B C A1 B1 C1 : Point) : Prop :=
  are_feet_of_altitudes A B C A1 B1 C1

-- Prove that AM + BM + CM = 2(R + r)
theorem sum_of_segments (h₁ : orthocenter_of A B C M) (h₂ : feet_of_altitudes A B C A1 B1 C1) :
  dist A M + dist B M + dist C M = 2 * (R + r) :=
sorry

-- Prove that AM * CM / B1M = BM * CM / A1M = AM * BM / C1M = const
theorem altitude_proportionality (h₁ : orthocenter_of A B C M) (h₂ : feet_of_altitudes A B C A1 B1 C1) (A_moves_on_circumcircle : Point → Prop) :
   (dist A M * dist C M) / dist B1 M = 
   (dist B M * dist C M) / dist A1 M = 
   (dist A M * dist B M) / dist C1 M :=
sorry

end sum_of_segments_altitude_proportionality_l628_628631


namespace not_p_suff_not_q_l628_628212

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q_l628_628212


namespace remainder_zero_when_x_divided_by_y_l628_628692

theorem remainder_zero_when_x_divided_by_y :
  ∀ (x y : ℝ), 
    0 < x ∧ 0 < y ∧ x / y = 6.12 ∧ y = 49.99999999999996 → 
      x % y = 0 := by
  sorry

end remainder_zero_when_x_divided_by_y_l628_628692


namespace non_congruent_triangles_perimeter_18_l628_628007

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628007


namespace problem1_problem2_l628_628510

-- Proof problem related to (1)
theorem problem1 (x : ℝ) (h : abs (x^2 - 1) < 3) : -2 < x ∧ x < 2 := sorry

-- Proof problem related to (2)
theorem problem2 (x a : ℝ) (h : a > 0) (h1 : x^2 + a < abs ((a + 1) * x)) :
  (0 < a ∧ a < 1 → (x ∈ set.Ioo (-1) (-a) ∪ set.Ioo a 1)) ∧
  (a = 1 → false) ∧
  (a > 1 → (x ∈ set.Ioo (-a) (-1) ∪ set.Ioo 1 a)) := sorry

end problem1_problem2_l628_628510


namespace find_sin_alpha_and_beta_l628_628451

theorem find_sin_alpha_and_beta (h1 : 0 < α) (h2 : α < π / 2) (h3 : π / 2 < β) (h4 : β < π) 
    (h5 : tan (α / 2) = 1 / 3) (h6 : cos (β - α) = -sqrt 2 / 10) 
    : sin α = 3 / 5 ∧ β = 3 * π / 4 := 
by { sorry }

end find_sin_alpha_and_beta_l628_628451


namespace probability_avg_greater_than_5_l628_628073

theorem probability_avg_greater_than_5 :
  ∑ s in (finset.powerset (finset.range 5)).filter (λ s, s.card = 2) ,
    ∑ t in (finset.range 5).filter (λ t, t ∉ s), 
      (t / 3 > 5) = (2 / 5) :=
begin 
  sorry,
end

end probability_avg_greater_than_5_l628_628073


namespace hemisphere_surface_area_and_volume_l628_628994

-- Definitions and conditions
def base_area (r : ℝ) : ℝ := π * r^2
def curved_surface_area (r : ℝ) : ℝ := 2 * π * r^2
def total_surface_area (r : ℝ) : ℝ := base_area r + curved_surface_area r
def volume (r : ℝ) : ℝ := (2 / 3) * π * r^3

-- Given condition
axiom base_area_given : base_area 15 = 225 * π

-- Theorem statement to prove total surface area and volume 
theorem hemisphere_surface_area_and_volume {r : ℝ} (h : base_area r = 225 * π) :
  total_surface_area r = 675 * π ∧ volume r = 2250 * π := by
  sorry

end hemisphere_surface_area_and_volume_l628_628994


namespace perfect_square_pattern_l628_628400

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l628_628400


namespace complementary_sets_count_l628_628785

-- Define the conditions
def shapes := ["circle", "square", "triangle", "rectangle"]
def colors := ["red", "blue", "green", "yellow"]
def shades := ["light", "medium", "dark", "very dark"]

def deck := {c : string × string × string | c.1 ∈ shapes ∧ c.2 ∈ colors ∧ c.3 ∈ shades}

def complementary (s : set (string × string × string)) :=
  (∀ (x y z : (string × string × string)), 
    x ∈ s ∧ y ∈ s ∧ z ∈ s →
    (x.1 = y.1 ∧ y.1 = z.1 ∨ x.1 ≠ y.1 ∧ y.1 ≠ z.1 ∧ z.1 ≠ x.1) ∧
    (x.2 = y.2 ∧ y.2 = z.2 ∨ x.2 ≠ y.2 ∧ y.2 ≠ z.2 ∧ z.2 ≠ x.2) ∧
    (x.3 = y.3 ∧ y.3 = z.3 ∨ x.3 ≠ y.3 ∧ y.3 ≠ z.3 ∧ z.3 ≠ x.3))

noncomputable def number_of_complementary_sets : ℕ :=
  sorry  -- Placeholder for the actual computation method

theorem complementary_sets_count : number_of_complementary_sets = 30016 :=
  sorry  -- The proof of this theorem

end complementary_sets_count_l628_628785


namespace polynomial_value_l628_628893
variable {x y : ℝ}
theorem polynomial_value (h : 3 * x^2 + 4 * y + 9 = 8) : 9 * x^2 + 12 * y + 8 = 5 :=
by
   sorry

end polynomial_value_l628_628893


namespace find_n_l628_628355

theorem find_n 
  (a : ℝ := 9 / 15)
  (S1 : ℝ := 15 / (1 - a))
  (b : ℝ := (9 + n) / 15)
  (S2 : ℝ := 3 * S1)
  (hS1 : S1 = 37.5)
  (hS2 : S2 = 112.5)
  (hb : b = 13 / 15)
  (hn : 13 = 9 + n) : 
  n = 4 :=
by
  sorry

end find_n_l628_628355


namespace loan_amount_calculation_l628_628742

theorem loan_amount_calculation
  (annual_interest : ℝ) (interest_rate : ℝ) (time : ℝ) (loan_amount : ℝ)
  (h1 : annual_interest = 810)
  (h2 : interest_rate = 0.09)
  (h3 : time = 1)
  (h4 : loan_amount = annual_interest / (interest_rate * time)) :
  loan_amount = 9000 := by
sorry

end loan_amount_calculation_l628_628742


namespace speed_of_current_l628_628718

theorem speed_of_current (v_b v_c v_d : ℝ) (hd : v_d = 15) 
  (hvd1 : v_b + v_c = v_d) (hvd2 : v_b - v_c = 12) :
  v_c = 1.5 :=
by sorry

end speed_of_current_l628_628718


namespace largest_integer_m_l628_628412

theorem largest_integer_m (m : ℤ) : (m^2 - 11 * m + 28 < 0) → m = 6 :=
begin
  sorry
end

end largest_integer_m_l628_628412


namespace eugene_total_payment_l628_628909

-- Define the initial costs of items
def cost_tshirt := 20
def cost_pants := 80
def cost_shoes := 150

-- Define the quantities
def quantity_tshirt := 4
def quantity_pants := 3
def quantity_shoes := 2

-- Define the discount rate
def discount_rate := 0.10

-- Define the total pre-discount cost
def pre_discount_cost :=
  (cost_tshirt * quantity_tshirt) +
  (cost_pants * quantity_pants) +
  (cost_shoes * quantity_shoes)

-- Define the discount amount
def discount_amount := discount_rate * pre_discount_cost

-- Define the post-discount cost
def post_discount_cost := pre_discount_cost - discount_amount

-- Theorem statement
theorem eugene_total_payment : post_discount_cost = 558 := by
  sorry

end eugene_total_payment_l628_628909


namespace area_ratio_inequality_l628_628237

theorem area_ratio_inequality 
  (S S1 : ℝ) 
  (a b c a1 b1 c1 k : ℝ) 
  (h1 : ∀ (a b c : ℝ), Triangle.is_not_obtuse a b c → Triangle.area a b c = S) 
  (h2 : ∀ (a1 b1 c1 : ℝ), Triangle.area a1 b1 c1 = S1) 
  (h3 : max (a1 / a) (max (b1 / b) (c1 / c)) = k) 
  : S1 ≤ k^2 * S := by 
  sorry

end area_ratio_inequality_l628_628237


namespace chord_length_eq_three_quarters_l628_628377

theorem chord_length_eq_three_quarters
  (a : ℝ) 
  (A B C D F : EuclideanSpace ℝ 2)
  (triangle_ABC : triangle A B C)
  (equilateral_triangle : triangle.is_equilateral triangle_ABC)
  (AD_altitude : is_altitude A D B C)
  (D_midpoint : is_midpoint D B C)
  (AF_chord_intersection_circle : intersects_circle_diameter (circle (A + D) (dist A D / 2)) A F)
  (AF_on_AB : lies_on F A B) :
  dist A F = (3 / 4) * a := 
sorry

end chord_length_eq_three_quarters_l628_628377


namespace sum_symmetry_l628_628117

def f (x : ℝ) : ℝ :=
  x^2 * (1 - x)^2

theorem sum_symmetry :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 :=
by
  sorry

end sum_symmetry_l628_628117


namespace square_of_cube_of_third_smallest_prime_l628_628688

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l628_628688


namespace find_200_digit_number_l628_628715

theorem find_200_digit_number : ∃ c ∈ {1, 2, 3}, ∃ (N : ℕ), N = 132 * c * 10^197 ∧ (N < 10^200 ∧ ∀ (N' : ℕ), remove_leading_and_third_digit N = N' → N = 44 * N') :=
by sorry

noncomputable def remove_leading_and_third_digit (N : ℕ) : ℕ := sorry

end find_200_digit_number_l628_628715


namespace product_of_invertibles_mod_120_l628_628135

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628135


namespace shaded_region_area_is_correct_l628_628923

noncomputable def area_of_shaded_region : ℝ :=
  let R := 6 -- radius of the larger circle
  let r := R / 2 -- radius of each smaller circle
  let area_large_circle := Real.pi * R^2
  let area_two_small_circles := 2 * Real.pi * r^2
  area_large_circle - area_two_small_circles

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 18 * Real.pi :=
sorry

end shaded_region_area_is_correct_l628_628923


namespace polygon_diagonals_l628_628050

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628050


namespace coefficient_x3_product_eq_neg4_l628_628094

theorem coefficient_x3_product_eq_neg4 :
  let P := (1 - 2 * x^2)
  let Q := (1 + x)^4
  (P * Q).coeff 3 = -4 := by
  sorry

end coefficient_x3_product_eq_neg4_l628_628094


namespace expected_ascending_pairs_l628_628444

open Classical

variable (n : ℕ) (hn : n > 1)

noncomputable def permutation_of_n := {σ : Fin n → Fin n // ∀ i, i ∈ σ}

def ascending_pair_count (σ : {σ : Fin n → Fin n // ∀ i, i ∈ σ}) : ℕ :=
  ∑ i j, if i < j ∧ σ.val i < σ.val j then 1 else 0

def E_X (n : ℕ) [fact (n > 1)] : ℝ :=
  ∑ σ in finset.univ, (ascending_pair_count σ) / (n.factorial)

theorem expected_ascending_pairs (n : ℕ) (hn : n > 1) : E_X n = (n * (n - 1) : ℝ) / 4 := by
  sorry

end expected_ascending_pairs_l628_628444


namespace mark_total_theater_spending_l628_628577

def week1_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week2_cost : ℝ := (2.5 * 6 - 0.1 * (2.5 * 6)) + 3
def week3_cost : ℝ := 4 * 4 + 3
def week4_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week5_cost : ℝ := (2 * (3.5 * 6 - 0.1 * (3.5 * 6))) + 6
def week6_cost : ℝ := 2 * 7 + 3

def total_cost : ℝ := week1_cost + week2_cost + week3_cost + week4_cost + week5_cost + week6_cost

theorem mark_total_theater_spending : total_cost = 126.30 := sorry

end mark_total_theater_spending_l628_628577


namespace determine_f_4_l628_628116

theorem determine_f_4 (f g : ℝ → ℝ)
  (h1 : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y)
  (h2 : ∀ x : ℝ, g x = 2 * x) :
  f 4 = 32 :=
sorry

end determine_f_4_l628_628116


namespace general_formula_for_sequence_l628_628990

noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def S_n (n : ℕ) : ℕ := sorry

theorem general_formula_for_sequence {n : ℕ} (hn: n > 0)
  (h1: ∀ n, a_n n > 0)
  (h2: ∀ n, 4 * S_n n = (a_n n)^2 + 2 * (a_n n))
  : a_n n = 2 * n := sorry

end general_formula_for_sequence_l628_628990


namespace available_spaces_l628_628727

noncomputable def numberOfBenches : ℕ := 50
noncomputable def capacityPerBench : ℕ := 4
noncomputable def peopleSeated : ℕ := 80

theorem available_spaces :
  let totalCapacity := numberOfBenches * capacityPerBench;
  let availableSpaces := totalCapacity - peopleSeated;
  availableSpaces = 120 := by
    sorry

end available_spaces_l628_628727


namespace center_locus_of_moving_circle_range_of_slope_of_line_l628_628248

-- Definition of the circles F1 and F2
def F1 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 1}
def F2 : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 9}

-- Center locus of the moving circle E
theorem center_locus_of_moving_circle :
  ∃ E_center : set (ℝ × ℝ), E_center = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} ∧
  (∀ p : ℝ × ℝ, p ∈ E_center → ∃ r : ℝ, r > 0 ∧
  dist p (center F1) = r + 1 ∧ dist p (center F2) = 3 - r) := 
sorry

-- Range of the slope of line l
theorem range_of_slope_of_line :
  ∀ (k : ℝ), k ≠ 0 →
  (∃ A B : ℝ × ℝ, B.2 ≠ 0 ∧
  BF2_perpendicular_HF2 (1, 0) (0, (9 - 4*k^2) / (12 * k)) ∧
  angle_MOA_geq_angle_MAO (2, 0) (k, 2*k)) →
  (-sqrt 6 / 4 ≤ k ∧  k < 0) ∨ (0 < k ∧ k ≤ sqrt 6 / 4) :=
sorry

end center_locus_of_moving_circle_range_of_slope_of_line_l628_628248


namespace centers_of_rectangles_l628_628824

-- Conditions and definitions from the problem
def triangle_ABC (A B C : Point) : Prop :=
  ∃ α : ℝ, α > 0 ∧ ∃ β : ℝ, β > 0 ∧ (0, α) ∈ line (A, C) ∧ (α, β) ∈ line (A, B) ∧ (0, β) ∈ line (B, C)

def point_O (B H : Point) : Point := midpoint B H

def point_M (A C : Point) : Point := midpoint A C

def lies_on_AC (Q P A C : Point) : Prop :=
  (Q ∈ line (A, C)) ∧ (P ∈ line (A, C))

def lies_on_AB_BC (R S A B C : Point) : Prop :=
  (R ∈ line (A, B)) ∧ (S ∈ line (B, C))

-- The proof statement
theorem centers_of_rectangles (A B C Q P R S O M : Point) (h_triangle : triangle_ABC A B C) 
(h_O : O = point_O B (altitude A C)) (h_M : M = point_M A C) 
(h_QP_on_AC : lies_on_AC Q P A C) (h_RS_on_AB_BC : lies_on_AB_BC R S A B C) :
  centers_of_PQRS Q P R S = (line_segment O M) \ {O, M} :=
  sorry

end centers_of_rectangles_l628_628824


namespace polygon_diagonals_l628_628049

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628049


namespace product_of_invertibles_mod_120_l628_628171

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628171


namespace product_of_invertibles_mod_120_l628_628172

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628172


namespace least_possible_value_of_one_integer_l628_628704

theorem least_possible_value_of_one_integer (
  A B C D E F : ℤ
) (h1 : (A + B + C + D + E + F) / 6 = 63)
  (h2 : A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100 ∧ E ≤ 100 ∧ F ≤ 100)
  (h3 : (A + B + C) / 3 = 65) : 
  ∃ D E F, (D + E + F) = 183 ∧ min D (min E F) = 83 := sorry

end least_possible_value_of_one_integer_l628_628704


namespace healthy_half_exists_l628_628740

noncomputable def apple_sphere (radius : ℝ) := 
  { P : ℝ × ℝ × ℝ // P.1 ^ 2 + P.2 ^ 2 + P.3 ^ 2 = radius ^ 2 }

noncomputable def point_A := (5, 0, 0) 
noncomputable def point_B := (x, y, z)  -- Assume coordinates of B
-- not necessarily on a straight line from A
noncomputable def point_C := (-5, 0, 0) 

-- Given conditions
def radius := 5
def path_length_AB := 9.9

-- Assuming path_bound condition for AB based on distance formula
def path_bound := (5 - x) ^ 2 + y ^ 2 + z ^ 2 < path_length_AB ^ 2

-- The perpendicular bisector plane:
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def perpendicular_bisector_plane (A B O : ℝ × ℝ × ℝ) :=
  { P : ℝ × ℝ × ℝ // 
    (P.1 - midpoint A B.1) * (O.1 - midpoint A B.1) + 
    (P.2 - midpoint A B.2) * (O.2 - midpoint A B.2) + 
    (P.3 - midpoint A B.3) * (O.3 - midpoint A B.3) = 0 }

-- The theorem we need to prove
theorem healthy_half_exists : 
  ∃ hemisphere : ℝ × ℝ × ℝ,
    (hemisphere ≠ point_B) ∧
    (∀ P ∈ apple_sphere radius, 
      if perpendicular_bisector_plane point_B point_C (0,0,0) P then 
        P ≠ (x, y, z))
  :=
sorry

end healthy_half_exists_l628_628740


namespace correct_statements_l628_628749

theorem correct_statements : 
  (∀ x : ℝ, ∃ y : ℝ, y^3 = x) ∧ (∀ x : ℝ, x ∈ ℝ ↔ (∃ p : ℝ, p = x)) := 
by
  -- statement 2: Every real number has a cube root
  have h2 : ∀ x : ℝ, ∃ y : ℝ, y^3 = x := by
    sorry
  -- statement 4: All real numbers correspond one-to-one with points on the number line
  have h4 : ∀ x : ℝ, x ∈ ℝ ↔ (∃ p : ℝ, p = x) := by
    sorry
  exact And.intro h2 h4

end correct_statements_l628_628749


namespace train_length_l628_628344

theorem train_length (v_kmh : ℕ) (t_s : ℝ) (v_m_s : ℝ) (L : ℝ) : 
  v_kmh = 144 → 
  t_s = 2.5 → 
  v_m_s = (v_kmh * 1000 / 3600) → 
  L = (v_m_s * t_s) → 
  L = 100 :=
by
  intros
  rw [← a, ← b, ← c, ← d]
  sorry

end train_length_l628_628344


namespace square_of_cube_of_third_smallest_prime_l628_628666

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628666


namespace integral_equality_l628_628313

noncomputable def calcIntegral : ℝ :=
  ∫ x in 0..2, (x + 1)^2 * Real.log(x + 1)^2

theorem integral_equality :
  calcIntegral = 9 * Real.log 3 ^ 2 - 6 * Real.log 3 + 79 / 27 :=
by
  sorry

end integral_equality_l628_628313


namespace below_sea_level_is_negative_l628_628495
-- Lean 4 statement


theorem below_sea_level_is_negative 
  (above_sea_pos : ∀ x : ℝ, x > 0 → x = x)
  (below_sea_neg : ∀ x : ℝ, x < 0 → x = x) : 
  (-25 = -25) :=
by 
  -- here we are supposed to provide the proof but we are skipping it with sorry
  sorry

end below_sea_level_is_negative_l628_628495


namespace sin_alpha_value_l628_628812

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end sin_alpha_value_l628_628812


namespace angle_BPC_equal_112_point_5_l628_628922

theorem angle_BPC_equal_112_point_5
    (A B C D E P Q : Type*)
    (h_sq : square A B C D)
    (side6 : side_length A B = 6)
    (isosceles_abe : is_isosceles_triangle A B E)
    (ab_eq_ae : A B = A E)
    (angle_bae_45 : angle B A E = 45)
    (intersect_bp : intersects B E A C P)
    (on_bc_q : on_line_segment B C Q)
    (perpendicular_pq_bc : is_perpendicular P Q B C)
    (pq_y : seg_length P Q = y) :
  angle B P C = 112.5 :=
sorry

end angle_BPC_equal_112_point_5_l628_628922


namespace return_trip_amount_l628_628979

noncomputable def gasoline_expense : ℝ := 8
noncomputable def lunch_expense : ℝ := 15.65
noncomputable def gift_expense_per_person : ℝ := 5
noncomputable def grandma_gift_per_person : ℝ := 10
noncomputable def initial_amount : ℝ := 50

theorem return_trip_amount : 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  initial_amount - total_expense + total_money_gifted = 36.35 :=
by 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  sorry

end return_trip_amount_l628_628979


namespace product_invertibles_mod_120_l628_628155

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628155


namespace eighth_term_of_geometric_sequence_l628_628656

def geometric_sequence_term (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_geometric_sequence : 
  geometric_sequence_term 12 (1 / 3) 8 = 4 / 729 :=
by 
  sorry

end eighth_term_of_geometric_sequence_l628_628656


namespace find_d_in_polynomial_l628_628195

theorem find_d_in_polynomial 
  (a b c d : ℤ) 
  (x1 x2 x3 x4 : ℤ)
  (roots_neg : x1 < 0 ∧ x2 < 0 ∧ x3 < 0 ∧ x4 < 0)
  (h_poly : ∀ x, 
    (x + x1) * (x + x2) * (x + x3) * (x + x4) = 
    x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_sum_eq : a + b + c + d = 2009) :
  d = (x1 * x2 * x3 * x4) :=
by
  sorry

end find_d_in_polynomial_l628_628195


namespace cube_painting_problem_l628_628736

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l628_628736


namespace Jimmy_earns_229_l628_628939

-- Definitions based on conditions from the problem
def number_of_type_A : ℕ := 5
def number_of_type_B : ℕ := 4
def number_of_type_C : ℕ := 3

def value_of_type_A : ℕ := 20
def value_of_type_B : ℕ := 30
def value_of_type_C : ℕ := 40

def discount_type_A : ℕ := 7
def discount_type_B : ℕ := 10
def discount_type_C : ℕ := 12

-- Calculation of the total amount Jimmy will earn
def total_earnings : ℕ :=
  let price_A := value_of_type_A - discount_type_A
  let price_B := value_of_type_B - discount_type_B
  let price_C := value_of_type_C - discount_type_C
  (number_of_type_A * price_A) +
  (number_of_type_B * price_B) +
  (number_of_type_C * price_C)

-- The statement to be proved
theorem Jimmy_earns_229 : total_earnings = 229 :=
by
  -- Proof omitted
  sorry

end Jimmy_earns_229_l628_628939


namespace savings_per_bagel_in_cents_l628_628610

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_per_bagel_in_cents_l628_628610


namespace pupils_like_pizza_l628_628643

theorem pupils_like_pizza (total_pupils : ℕ) (B : ℕ) (PB : ℕ) (h1 : total_pupils = 200) (h2 : B = 115) (h3 : PB = 40) :
  ∃ P : ℕ, P = 125 :=
by
  use (total_pupils - (B - PB))
  rw [h1, h2, h3]
  norm_num 
  sorry

end pupils_like_pizza_l628_628643


namespace smallest_integer_y_l628_628659

theorem smallest_integer_y : ∃ (y : ℕ), (\frac{8}{11} < \frac{y}{17}) ∧ y = 13 :=
by
  sorry

end smallest_integer_y_l628_628659


namespace length_of_AD_l628_628088

variables {P Q R S A B C D : Type} [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define all necessary points and set up conditions as variables and definitions
variable (rectangle_PQRS : is_rectangle P Q R S)
variable (PQ_length : dist P Q = 10)
variable (QR_length : dist Q R = 8)
variable (A_on_PQ : A ∈ line_segment P Q)
variable (B_on_PQ : B ∈ line_segment P Q)
variable (C_on_QR : C ∈ line_segment Q R)
variable (D_on_QR : D ∈ line_segment Q R)
variable (AP_length : dist A P = 3)
variable (BQ_length : dist B Q = 3)
variable (DP_equals_PC : dist D P = dist P C)
variable (triangle_PCD_isosceles_right : isosceles_right_triangle P C D)

-- Define the goal to prove the length of segment AD
theorem length_of_AD : dist A D = 3 + 4 * real.sqrt 2 := 
sorry

end length_of_AD_l628_628088


namespace sasha_tree_planting_cost_l628_628985

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l628_628985


namespace cubic_polynomial_value_at_3_and_neg3_l628_628953

variable (Q : ℝ → ℝ)
variable (a b c d m : ℝ)
variable (h1 : Q 1 = 5 * m)
variable (h0 : Q 0 = 2 * m)
variable (h_1 : Q (-1) = 6 * m)
variable (hQ : ∀ x, Q x = a * x^3 + b * x^2 + c * x + d)

theorem cubic_polynomial_value_at_3_and_neg3 :
  Q 3 + Q (-3) = 67 * m := by
  -- sorry is used to skip the proof
  sorry

end cubic_polynomial_value_at_3_and_neg3_l628_628953


namespace volume_of_pyramid_l628_628611

variable (α β r : ℝ)

-- Occurring conditions
variable (acute_alpha : 0 < α ∧ α < π / 2)
variable (rhombus_base : true) -- Placeholder for more complex condition on base later
variable (dihedral_angle : true) -- Placeholder for more complex condition on dihedral angle later
variable (inscribed_sphere_radius : r > 0)

theorem volume_of_pyramid (h1 : acute_alpha) (h2 : rhombus_base) (h3 : dihedral_angle) (h4 : inscribed_sphere_radius) :
  volume = (4 * r^3 * tan β) / (3 * sin α * tan^3 (β / 2)) :=
sorry

end volume_of_pyramid_l628_628611


namespace probability_of_prime_is_one_third_l628_628311

noncomputable def probability_prime_between_1_and_30 : ℚ :=
  let primes : ℕ → bool := λ n, n ∈ [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] in
  let favorable_outcomes : ℕ := (List.range' 1 30).filter primes |>.length in
  let total_outcomes : ℕ := 30 in
  favorable_outcomes / total_outcomes

theorem probability_of_prime_is_one_third :
  probability_prime_between_1_and_30 = 1 / 3 :=
by
  sorry

end probability_of_prime_is_one_third_l628_628311


namespace y_value_increase_l628_628207

theorem y_value_increase (initial_x initial_y x_increase : ℕ) (hx : initial_x = 1) (hy : initial_y = 1) (h_step : ∀ (x y : ℕ), x + 2 → y + 5) : initial_y + (x_increase / 2) * 5 = 21 :=
by
  have h_factor : x_increase = 8 := rfl
  have h_step_adj : (x_increase / 2) = 4 := rfl
  calc 
    initial_y + (x_increase / 2) * 5
      = 1 + 4 * 5 := by rfl
  ... = 1 + 20 := by rfl
  ... = 21 := by rfl
  done

end y_value_increase_l628_628207


namespace transformed_complex_result_l628_628387

open Complex

def initial_complex : ℂ := -6 - 2 * I

def rotation (θ : ℝ) : ℂ := Real.cos θ + Real.sin θ * I

def dilation (factor : ℝ) : ℂ := factor

def transformed_complex : ℂ := initial_complex * (dilation 3 * rotation (Real.pi / 6))

theorem transformed_complex_result :
  transformed_complex = -3 - 9 * Real.sqrt 3 - 12 * I :=
sorry

end transformed_complex_result_l628_628387


namespace cube_problem_l628_628739

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l628_628739


namespace proof_true_props_l628_628818

variable (m n : Line)
variable (α β γ : Plane)

axiom non_coincident_lines : m ≠ n
axiom non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

def prop1 := ∀ m α β, (m ⊥ α ∧ m ⊥ β) → (α ∥ β)
def prop2 := ∀ α β γ, (α ⊥ γ ∧ β ⊥ γ) → (α ∥ β)
def prop3 := ∀ m n α β, (m ∥ n ∧ angle_between m α = angle_between n β) → (α ∥ β)
def prop4 := ∀ m n α β, 
  (skew m n ∧ m ⊆ α ∧ m ∥ β ∧ n ⊆ β ∧ n ∥ α) → (α ∥ β)

theorem proof_true_props (h1 : prop1) (h4 : prop4) (h2 : ¬proposition2) (h3 : ¬proposition3) :
  (h1 ∧ h4 ∧ h2 ∧ h3) = ([h1, h4]) :=
sorry

end proof_true_props_l628_628818


namespace regular_polygon_num_sides_l628_628042

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628042


namespace total_potato_weight_l628_628335

theorem total_potato_weight (bags_morning : ℕ) (bags_afternoon : ℕ) (weight_per_bag : ℕ) :
  bags_morning = 29 → 
  bags_afternoon = 17 → 
  weight_per_bag = 7 → 
  (bags_morning + bags_afternoon) * weight_per_bag = 322 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  norm_num 
  sorry

end total_potato_weight_l628_628335


namespace minimum_distance_l628_628458

open real

noncomputable def pointA := (0 : ℝ, 1 : ℝ)
noncomputable def pointB := (1 : ℝ, 0 : ℝ)
noncomputable def pointM := (0 : ℝ, -3 : ℝ)

def line_eq (x y : ℝ) : Prop := y = -x + 1

def dist_point_line (x1 y1 x2 y2 a b c : ℝ) : ℝ :=
  abs(a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

theorem minimum_distance : dist_point_line 0 (-3) 0 1 1 1 (-1) = 2 * sqrt 2 :=
by
  -- Calculate the distance
  sorry

end minimum_distance_l628_628458


namespace pond_volume_extraction_l628_628085

/--
  Let length (l), width (w), and depth (h) be dimensions of a pond.
  Given:
  l = 20,
  w = 10,
  h = 5,
  Prove that the volume of the soil extracted from the pond is 1000 cubic meters.
-/
theorem pond_volume_extraction (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
  by
    sorry

end pond_volume_extraction_l628_628085


namespace necessary_but_not_sufficient_l628_628525

variable {α : Type*}
variables (A B P : α)
variable (PA PB : α → ℝ)
variable (k : ℝ)

def propositionA : Prop := PA P + PB P = k
def propositionB : Prop := ∃ e > 0, (PA P + PB P = k) ∧ true

theorem necessary_but_not_sufficient : propositionA A B P PA PB k → propositionB A B P PA PB k :=
sorry

end necessary_but_not_sufficient_l628_628525


namespace number_of_non_congruent_triangles_with_perimeter_20_l628_628489

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_20_l628_628489


namespace midpoint_distance_ratio_l628_628567

theorem midpoint_distance_ratio (p q r : ℝ) (P Q R : ℝ × ℝ × ℝ)
  (h1 : midpoint Q R = (p, 0, 0))
  (h2 : midpoint P R = (0, q, 0))
  (h3 : midpoint P Q = (0, 0, r)) :
  (dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 := by
  sorry

end midpoint_distance_ratio_l628_628567


namespace dice_sum_probability_l628_628807

def four_dice_probability_sum_to_remain_die : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 4 * 120
  favorable_outcomes / total_outcomes

theorem dice_sum_probability : four_dice_probability_sum_to_remain_die = 10 / 27 :=
  sorry

end dice_sum_probability_l628_628807


namespace circle_area_l628_628236

/-!

# Problem: Prove that the area of the circle defined by the equation \( x^2 + y^2 - 2x + 4y + 1 = 0 \) is \( 4\pi \).
-/

theorem circle_area : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 = 0) →
  ∃ (A : ℝ), A = 4 * Real.pi := 
by
  sorry

end circle_area_l628_628236


namespace example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l628_628583

-- Define what it means to be a three-digit number using only two distinct digits
def two_digit_natural (d1 d2 : ℕ) (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = d1 ∨ d = d2) ∧ 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem example_of_four_three_digit_numbers_sum_2012_two_digits_exists :
  ∃ a b c d : ℕ, 
    two_digit_natural 3 5 a ∧
    two_digit_natural 3 5 b ∧
    two_digit_natural 3 5 c ∧
    two_digit_natural 3 5 d ∧
    a + b + c + d = 2012 :=
by
  sorry

end example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l628_628583


namespace necessary_but_not_sufficient_cond_l628_628947

noncomputable
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_cond (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (hseq : geometric_sequence a a1 q)
  (hpos : a1 > 0) :
  (q < 0 ↔ (∀ n : ℕ, a (2 * n + 1) + a (2 * n + 2) < 0)) :=
sorry

end necessary_but_not_sufficient_cond_l628_628947


namespace product_of_invertibles_mod_120_l628_628169

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628169


namespace num_ways_to_sum_1800_l628_628874

-- Define the conditions as provided
def sum_condition : Nat → Nat → Prop := λ x y, 2 * x + 3 * y = 1800
def term_condition : Nat → Nat → Prop := λ x y, x + y = 600

-- The number of ways to write 1800 as the sum of twos and threes with exactly 600 terms
theorem num_ways_to_sum_1800 : 
  (∃ (x y : Nat), sum_condition x y ∧ term_condition x y) ∧ 
  (∀ x y, sum_condition x y ∧ term_condition x y → (x = 3 * (y / 3) ∧ 0 ≤ y / 3 ∧ y / 3 ≤ 300)) → 
  ∃ (n : Nat), 301 = n :=
by
  sorry

end num_ways_to_sum_1800_l628_628874


namespace regular_polygon_num_sides_l628_628041

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628041


namespace second_investment_rate_l628_628648

theorem second_investment_rate (P : ℝ) (r₁ t : ℝ) (I_diff : ℝ) (P900 : P = 900) (r1_4_percent : r₁ = 0.04) (t7 : t = 7) (I_years : I_diff = 31.50) :
∃ r₂ : ℝ, 900 * (r₂ / 100) * 7 - 900 * 0.04 * 7 = 31.50 → r₂ = 4.5 := 
by
  sorry

end second_investment_rate_l628_628648


namespace find_sides_from_diagonals_l628_628059

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628059


namespace product_of_invertibles_mod_120_l628_628128

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628128


namespace trigonometric_values_30_45_60_90_l628_628276

noncomputable def angle_30 : Real := Real.angle (π / 6)
noncomputable def angle_45 : Real := Real.angle (π / 4)
noncomputable def angle_60 : Real := Real.angle (π / 3)
noncomputable def angle_90 : Real := Real.angle (π / 2)

theorem trigonometric_values_30_45_60_90 :
  sin angle_30 = 1/2 ∧ cos angle_30 = sqrt 3 / 2 ∧ tan angle_30 = sqrt 3 / 3 ∧
  sin angle_45 = sqrt 2 / 2 ∧ cos angle_45 = sqrt 2 / 2 ∧ tan angle_45 = 1 ∧
  sin angle_60 = sqrt 3 / 2 ∧ cos angle_60 = 1/2 ∧ tan angle_60 = sqrt 3 ∧
  sin angle_90 = 1 ∧ cos angle_90 = 0 := 
by {
  -- Proof skipped for brevity
  sorry
}

end trigonometric_values_30_45_60_90_l628_628276


namespace jim_cousin_money_l628_628935

theorem jim_cousin_money :
  (∀ (cheeseburger_cost milkshake_cost cheese_fries_cost jim_money : ℝ)
    (spent_fraction : ℝ),
    cheeseburger_cost = 3 →
    milkshake_cost = 5 →
    cheese_fries_cost = 8 →
    spent_fraction = 0.8 →
    jim_money = 20 →
    -- Calculating the total cost of their meal
    let total_meal_cost := 2 * (cheeseburger_cost + milkshake_cost) + cheese_fries_cost in
    -- Calculating the total initial money
    let total_initial_money := total_meal_cost / spent_fraction in
    -- Finding Jim's cousin's money
    let cousin_money := total_initial_money - jim_money in
    cousin_money = 10) :=
begin
  sorry
end

end jim_cousin_money_l628_628935


namespace arcs_midpoints_perpendicular_on_circle_l628_628830

noncomputable def midpoint_of_arc (A B O : Point) : Point := sorry

theorem arcs_midpoints_perpendicular_on_circle 
  (A B C D O A₁ B₁ C₁ D₁ : Point)
  (h_circle : Circle O (distance O A) = {A, B, C, D}) 
  (h_A1 : A₁ = midpoint_of_arc A B O)
  (h_B1 : B₁ = midpoint_of_arc B C O) 
  (h_C1 : C₁ = midpoint_of_arc C D O) 
  (h_D1 : D₁ = midpoint_of_arc D A O) :
  ∠ A₁ O C₁ = 90 ∧ ∠ B₁ O D₁ = 90 :=
sorry

end arcs_midpoints_perpendicular_on_circle_l628_628830


namespace find_sides_from_diagonals_l628_628058

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628058


namespace line_l_problem_l628_628476

theorem line_l_problem (a : ℝ) :
  let line_l := (a^2 + a + 1) * x - y + 1 = 0 in
  (a = -1 → (∀ x y, (x + y = 0) → line_l → (-1 : ℝ) = 1)) ∧
  ((∀ x y, (x - y = 0) → (a^2 + a + 1 = 1) → (a * (a + 1) = 0 → (a = 0 ∨ a = -1))) = false) ∧
  (line_l ∧ (∀ x, x = 0 → 1 = 1) = true) ∧
  (a = 0 → let intercepts := (1 * x - y + 1 = 0) in
    (∀ x, x = -1) ∧ (∀ y, y = 1) = (false)) :=
sorry

end line_l_problem_l628_628476


namespace stratified_leader_selection_probability_of_mixed_leaders_l628_628604

theorem stratified_leader_selection :
  let num_first_grade := 150
  let num_second_grade := 100
  let total_leaders := 5
  let leaders_first_grade := (total_leaders * num_first_grade) / (num_first_grade + num_second_grade)
  let leaders_second_grade := (total_leaders * num_second_grade) / (num_first_grade + num_second_grade)
  leaders_first_grade = 3 ∧ leaders_second_grade = 2 :=
by
  sorry

theorem probability_of_mixed_leaders :
  let num_first_grade_leaders := 3
  let num_second_grade_leaders := 2
  let total_leaders := 5
  let total_ways := 10
  let favorable_ways := 6
  (favorable_ways / total_ways) = (3 / 5) :=
by
  sorry

end stratified_leader_selection_probability_of_mixed_leaders_l628_628604


namespace optimal_optimism_coefficient_l628_628368

theorem optimal_optimism_coefficient (a b : ℝ) (x : ℝ) (h_b_gt_a : b > a) (h_x : 0 < x ∧ x < 1) 
  (h_c : ∀ (c : ℝ), c = a + x * (b - a) → (c - a) * (c - a) = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end optimal_optimism_coefficient_l628_628368


namespace yard_fraction_occupied_by_flowerbeds_l628_628332

noncomputable def rectangular_yard_area (length width : ℕ) : ℕ :=
  length * width

noncomputable def triangle_area (leg_length : ℕ) : ℕ :=
  2 * (1 / 2 * leg_length ^ 2)

theorem yard_fraction_occupied_by_flowerbeds :
  let length := 30
  let width := 7
  let parallel_side_short := 20
  let parallel_side_long := 30
  let flowerbed_leg := 7
  rectangular_yard_area length width ≠ 0 ∧
  triangle_area flowerbed_leg * 2 = 49 →
  (triangle_area flowerbed_leg * 2) / rectangular_yard_area length width = 7 / 30 :=
sorry

end yard_fraction_occupied_by_flowerbeds_l628_628332


namespace circle_cartesian_eq_circle_polar_eq_length_PQ_l628_628096

-- Conditions
def circle_parametric_eq (φ : ℝ) : ℝ × ℝ := 
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_polar_eq (ρ θ : ℝ) : Prop := 
  2 * ρ * Real.sin (θ + Real.pi / 3) = 6 * Real.sqrt 3

def ray_theta : ℝ := Real.pi / 6

-- Proof Goals
theorem circle_cartesian_eq : ∀ φ, let (x, y) := circle_parametric_eq φ in 
  (x - 1)^2 + y^2 = 1 :=
sorry

theorem circle_polar_eq : ∀ θ ρ, 
  (∃ φ, let (x, y) := circle_parametric_eq φ in x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) 
  → ρ = 2 * Real.cos θ :=
sorry

theorem length_PQ : ∀ θ, θ = ray_theta → 
  let ρ1 := (sqrt 3 : ℝ) in
  let ρ2 := (3 * sqrt 3 : ℝ) in
  |ρ1 - ρ2| = 2 * sqrt 3 :=
sorry

end circle_cartesian_eq_circle_polar_eq_length_PQ_l628_628096


namespace total_distance_l628_628306

/--
A man completes a journey in 30 hours. He travels the first half of the journey at the rate of 20 km/hr and 
the second half at the rate of 10 km/hr. Prove that the total journey is 400 km.
-/
theorem total_distance (D : ℝ) (h : D / 40 + D / 20 = 30) :
  D = 400 :=
sorry

end total_distance_l628_628306


namespace average_weight_increase_l628_628996

theorem average_weight_increase 
  (w_old : ℝ) (w_new : ℝ) (n : ℕ) 
  (h1 : w_old = 65) 
  (h2 : w_new = 93) 
  (h3 : n = 8) : 
  (w_new - w_old) / n = 3.5 := 
by 
  sorry

end average_weight_increase_l628_628996


namespace large_gears_workers_l628_628524

theorem large_gears_workers (total_workers large_gears_per_worker small_gears_per_worker: ℕ)
  (pair_large_gears pair_small_gears: ℕ) :
  total_workers = 34 →
  large_gears_per_worker = 20 →
  small_gears_per_worker = 15 →
  pair_large_gears = 3 →
  pair_small_gears = 2 →
  ∃ x, (2 * large_gears_per_worker * x = 3 * small_gears_per_worker * (total_workers - x)) ∧ x = 18 :=
by
  intros h1 h2 h3 h4 h5
  use 18
  split
  . rw [h1, h2, h3, h4, h5]
    -- Simplifying the original equation (2 * 20 * x = 3 * 15 * (34 - x))
    calc 2 * 20 * 18 = 40 * 18 : by ring
       ... = 720 : by norm_num
       ... = 45 * 34 : by norm_num
       ... = 3 * 15 * 34 : by ring
       ... = 3 * 15 * (34 - 18) : by norm_num

   -- Confirming x = 18
   exact rfl

end large_gears_workers_l628_628524


namespace jim_cousin_money_l628_628938

theorem jim_cousin_money (jim_money : ℕ) (cheeseburger_cost : ℕ) (cheeseburgers_ordered : ℕ) 
    (milkshake_cost : ℕ) (milkshakes_ordered : ℕ) (cheesefries_cost : ℕ) 
    (spent_percentage : ℚ) (spent_money : ℕ) :
  jim_money = 20 →
  cheeseburger_cost = 3 →
  cheeseburgers_ordered = 2 →
  milkshake_cost = 5 →
  milkshakes_ordered = 2 →
  cheesefries_cost = 8 →
  spent_percentage = 0.8 →
  spent_money = 24 →
  jim_money + (spent_money / spent_percentage) - jim_money = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  sorry

end jim_cousin_money_l628_628938


namespace find_a21_l628_628846

def seq_a (n : ℕ) : ℝ := sorry  -- This should define the sequence a_n
def seq_b (n : ℕ) : ℝ := sorry  -- This should define the sequence b_n

theorem find_a21 (h1 : seq_a 1 = 2)
  (h2 : ∀ n, seq_b n = seq_a (n + 1) / seq_a n)
  (h3 : ∀ n m, seq_b n = seq_b m * r^(n - m)) 
  (h4 : seq_b 10 * seq_b 11 = 2) :
  seq_a 21 = 2 ^ 11 :=
sorry

end find_a21_l628_628846


namespace jim_cousin_money_l628_628936

theorem jim_cousin_money :
  (∀ (cheeseburger_cost milkshake_cost cheese_fries_cost jim_money : ℝ)
    (spent_fraction : ℝ),
    cheeseburger_cost = 3 →
    milkshake_cost = 5 →
    cheese_fries_cost = 8 →
    spent_fraction = 0.8 →
    jim_money = 20 →
    -- Calculating the total cost of their meal
    let total_meal_cost := 2 * (cheeseburger_cost + milkshake_cost) + cheese_fries_cost in
    -- Calculating the total initial money
    let total_initial_money := total_meal_cost / spent_fraction in
    -- Finding Jim's cousin's money
    let cousin_money := total_initial_money - jim_money in
    cousin_money = 10) :=
begin
  sorry
end

end jim_cousin_money_l628_628936


namespace count_odd_three_digit_numbers_l628_628270

theorem count_odd_three_digit_numbers : 
  let digits := {1, 2, 3, 4}
  in (∃ n : ℕ, 100 ≤ n ∧ n < 400 ∧ n % 2 = 1 ∧ (∀ d ∈ finset.digits n, d ∈ digits) ∧ finset.digits n ⊆ digits) →
  finset.sum (finset.powerset digits) (λ s, if ((100 <= s ∧ s < 400 ∧ s % 2 = 1)) then 1 else 0) = 24 :=
by
  sorry

end count_odd_three_digit_numbers_l628_628270


namespace cubic_polynomial_evaluation_l628_628022

theorem cubic_polynomial_evaluation (f : ℝ → ℝ)
  (h_monic : ∀ a b c d : ℝ, f = λ x, x^3 + a * x^2 + b * x + c ∧ d = 1)
  (h_f_neg1 : f (-1) = 1)
  (h_f_2 : f 2 = -2)
  (h_f_neg3 : f (-3) = 3) : f 1 = -9 :=
  sorry

end cubic_polynomial_evaluation_l628_628022


namespace arithmetic_seq_a10_l628_628831

variable (a : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (d : ℚ := 1)

def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem arithmetic_seq_a10 (h_seq : is_arithmetic_seq a d)
                          (h_sum : sum_first_n_terms a S)
                          (h_condition : S 8 = 4 * S 4) :
  a 10 = 19/2 := 
sorry

end arithmetic_seq_a10_l628_628831


namespace problem1_problem2_l628_628482

open Real

-- Define vectors a and b
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-2, -4)

-- Define vector c as a function of k
def c (k : ℝ) : ℝ × ℝ := (3 - 2 * k, 2 - 4 * k)

-- Problem 1: Find k when b is perpendicular to c
theorem problem1 (k : ℝ) : b.1 * (c k).1 + b.2 * (c k).2 = 0 → k = 7 / 10 := by
  sorry

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Norm of a vector
def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem 2: Find the cosine of the angle between a and b
theorem problem2 : 
  cos_angle := dot_product a b / (norm a * norm b)  
  cos_angle = -7 * sqrt 65 / 65 := 
by
  sorry

end problem1_problem2_l628_628482


namespace a_20_value_l628_628098

def a : ℕ → ℚ 
| 0     := 4 / 5
| (n+1) := if a n < (1/2) then 2 * a n else 2 * a n - 1

theorem a_20_value : a 20 = 2 / 5 :=
by sorry

end a_20_value_l628_628098


namespace find_B_plus_10_B_inv_l628_628889

variable {ι : Type} [DecidableEq ι] [Fintype ι]
variable {α : Type} [Field α]

noncomputable def matrix_equation (B : Matrix ι ι α) (I : Matrix ι ι α) : Prop :=
  invertible B ∧ (B - 3 • I) ⬝ (B - 5 • I) = 0

theorem find_B_plus_10_B_inv (B : Matrix ι ι α) (I : Matrix ι ι α) (h : matrix_equation B I) :
  B + 10 • B⁻¹ = 8 • I :=
sorry

end find_B_plus_10_B_inv_l628_628889


namespace geometric_sequence_product_property_l628_628082

theorem geometric_sequence_product_property
  (b : ℕ → ℝ) 
  (h : b 9 = 1)
  (n : ℕ) 
  (hn : n < 17) : 
  b 1 * b 2 * ... * b n = b 1 * b 2 * ... * b (17 - n) := 
sorry

end geometric_sequence_product_property_l628_628082


namespace non_congruent_triangles_with_perimeter_18_l628_628002

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l628_628002


namespace product_of_invertible_integers_mod_120_l628_628139

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628139


namespace product_coprime_mod_120_l628_628184

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628184


namespace original_average_age_l628_628606

-- Define the conditions
def avg_age_new_students := 32
def total_students_after := 160
def decrease_in_avg_age := 4
def new_students := 120
def old_students := total_students_after - new_students

-- Hypothesis expression for the total age equation
def equation (A : ℝ) : ℝ :=
  40 * A + 120 * avg_age_new_students

-- Concluding the average age before joining the new students
theorem original_average_age :
  ∃ A : ℝ, equation A = total_students_after * (A - decrease_in_avg_age) ∧ A = 4480 / 120 :=
by
  -- We state that 40 * A + 120 * avg_age_new_students must equal to 160 * (A - 4)
  use 4480 / 120
  split
  { 
    -- Rewrite the problem into simplified equations
    admit 
  }
  {
    -- Conclude the correct average age
    sorry 
  }

end original_average_age_l628_628606


namespace square_of_cube_of_third_smallest_prime_l628_628669

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628669


namespace log_relationship_l628_628437

theorem log_relationship :
  (a b c : ℝ) (h₁ : a = Real.log 2 / Real.log 3) (h₂ : b = Real.log (1/3) / Real.log 2) (h₃ : c = Real.sqrt 2) :
  b < a ∧ a < c :=
by {
  sorry
}

end log_relationship_l628_628437


namespace count_big_boxes_l628_628754

theorem count_big_boxes (B : ℕ) (h : 7 * B + 4 * 9 = 71) : B = 5 :=
sorry

end count_big_boxes_l628_628754


namespace triangulate_colored_odd_polygon_l628_628260

open Classical

theorem triangulate_colored_odd_polygon
  {n : ℕ} (h_odd : Odd n) (h_coloring : ∀ (vertices : Fin n → ℕ) (i : Fin n), vertices (i + 1) % n ≠ vertices i) :
  ∃ (triangulation : List (Fin n × Fin n)), 
    (∀ (d : Fin n × Fin n), d ∈ triangulation → vertices d.1 ≠ vertices d.2) ∧ 
    (∀ (d₁ d₂ : Fin n × Fin n), d₁ ∈ triangulation → d₂ ∈ triangulation → non_intersecting d₁ d₂) :=
by
  sorry

end triangulate_colored_odd_polygon_l628_628260


namespace largest_distance_between_points_on_spheres_l628_628281

-- Defining the centers and the radii of the spheres
def sphere_center1 : ℝ × ℝ × ℝ := (3, -5, 10)
def sphere_radius1 : ℝ := 25
def sphere_center2 : ℝ × ℝ × ℝ := (-7, 15, -20)
def sphere_radius2 : ℝ := 95

-- Defining the distance formula in 3D
def distance (point1 point2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2 + (point1.3 - point2.3)^2)

-- Calculating the distance between the centers of the two spheres
def distance_between_centers := distance sphere_center1 sphere_center2

-- Stating the proof problem
theorem largest_distance_between_points_on_spheres :
  (sphere_radius1 + distance_between_centers + sphere_radius2) = 120 + 10 * Real.sqrt 14 :=
sorry

end largest_distance_between_points_on_spheres_l628_628281


namespace evaluate_f_i_l628_628395

noncomputable def f (x : ℂ) : ℂ :=
  (x^5 + 2 * x^3 + x) / (x + 1)

theorem evaluate_f_i : f (Complex.I) = 0 := 
  sorry

end evaluate_f_i_l628_628395


namespace range_of_m_l628_628959

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end range_of_m_l628_628959


namespace product_invertibles_mod_120_l628_628160

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628160


namespace non_congruent_triangles_with_perimeter_18_l628_628004

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l628_628004


namespace common_points_conic_sections_l628_628424

theorem common_points_conic_sections :
  ∀ (x y : ℝ), 
    (2 * x^2 + 3 * x * y - 2 * y^2 - 6 * x + 3 * y = 0 ∧
    3 * x^2 + 7 * x * y + 2 * y^2 - 7 * x + y - 6 = 0) ↔ 
    ( (x, y) = (-1,2) ∨ (x, y) = (1,1) ∨ (x, y) = (0, 3/2) ∨ (x, y) = (3,0) ∨ (x, y) = (4, -1/2) ∨ (x, y) = (5,-1) )
  :=
begin
  sorry
end

end common_points_conic_sections_l628_628424


namespace jump_rope_sum_l628_628552

theorem jump_rope_sum : 
  let Jung_min_rate := 256 / 4,
      Jimin_three_minutes := 111,
      Jung_min_three_minutes := Jung_min_rate * 3
  in
  Jung_min_three_minutes + Jimin_three_minutes = 303 := 
by {
  let Jung_min_rate := 256 / 4,
  let Jimin_three_minutes := 111,
  let Jung_min_three_minutes := Jung_min_rate * 3,
  sorry
}

end jump_rope_sum_l628_628552


namespace obtuse_triangle_k_values_l628_628634

noncomputable def number_of_valid_k (a b k : ℕ) : ℕ :=
  if a > b then number_of_valid_k b a k else
    -- Case 1: 'b' as the longest side
    (if b ≤ a + k ∧ b * b > a * a + k * k then 1 else 0) +
    -- Case 2: 'k' as the longest side
    if k > 0 then number_of_valid_k a k b else 0

theorem obtuse_triangle_k_values : ∃ k_vals : Finset ℕ, 14 = k_vals.card ∧
  ∀ k ∈ k_vals, 4 < k ∧ k < 30 ∧ 
    ((17 < 13 + k ∧ 17^2 > 13^2 + k^2) ∨ (13 + 17 > k ∧ k^2 > 13^2 + 17^2)) := by
  sorry

end obtuse_triangle_k_values_l628_628634


namespace four_digit_perfect_square_l628_628405

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l628_628405


namespace max_trading_cards_l628_628932

theorem max_trading_cards (h : 10 ≥ 1.25 * nat):
  nat ≤ 8 :=
sorry

end max_trading_cards_l628_628932


namespace num_ways_choose_two_people_from_ten_l628_628086

theorem num_ways_choose_two_people_from_ten :
  ∃ (n : ℕ), n = 10 * 9 :=
by
  use 90
  sorry

end num_ways_choose_two_people_from_ten_l628_628086


namespace henry_distance_l628_628870

noncomputable def distance_travelled (steps : ℕ) : ℝ :=
if steps = 0 then 0 else
  let b_n : ℕ → ℝ := λ n, if n = 0 then 8 / 3 else (b_n (n - 1) - 8 / 3) + 2 / 3 * (4 - (b_n (n - 1) - 8 / 3))
  in b_n steps

noncomputable def total_distance (steps : ℕ) : ℝ :=
(distance_travelled steps) .sum

theorem henry_distance (d : ℝ) (total_travelled : ℝ) :
  d = 4 →
  (∀(steps : ℕ), total_travelled ≤ 6) →
  total_travelled = 6 →
  total_distance 3 = 1.2 := sorry

end henry_distance_l628_628870


namespace triangle_problem_l628_628841

noncomputable theory
open Real

/--
In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
and (2b - c) * cos A = a * cos C, then
1. angle A = pi / 3
2. If D is a point on side BC such that BD = 2 * DC and AD = 2, the maximum area of triangle ABC = (3 * sqrt 3) / 2
-/
theorem triangle_problem 
  (a b c : ℝ)
  (A B C : ℝ)
  (D : ℝ)
  (BD DC : ℝ)
  (AD : ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (sqrt : ℝ → ℝ)
  (h1 : a / sin A = b / sin B = c / sin C)
  (h2 : (2 * b - c) * cos A = a * cos C)
  (h3 : BD = 2 * DC)
  (h4 : AD = 2) : 
  (A = π / 3) ∧ (∀ t : ℝ, t = D → (1 / 2) * b * c * (sin A) ≤ (3 * sqrt 3) / 2) :=
by
  sorry

end triangle_problem_l628_628841


namespace correct_calculation_l628_628291

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l628_628291


namespace polynomial_non_zero_coefficients_l628_628565

theorem polynomial_non_zero_coefficients 
  (Q : Polynomial ℝ) 
  (hQ : Q ≠ 0) 
  (n : ℕ) 
  (hn : n > 0) : 
  (Q * (X-1)^n).coeffs.len ≥ n + 1 := 
sorry

end polynomial_non_zero_coefficients_l628_628565


namespace total_surface_area_of_cubes_structure_l628_628392

open Nat

/-- 
  There are eight cubes with volumes 1, 8, 27, 64, 125, 216, 512, and 729 cubic units.
  The first six cubes are stacked vertically in decreasing order of volume.
  The seventh cube (with side length 6 units) is placed adjacent to the fifth cube (with side length 5 units).
  The eighth cube (with side length 8 units) is placed on top of the stack.
  Prove that the total surface area of this structure is 1266 square units.
-/
theorem total_surface_area_of_cubes_structure (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ)
  (h1 : v1 = 1) (h2 : v2 = 8) (h3 : v3 = 27) (h4 : v4 = 64) (h5 : v5 = 125)
  (h6 : v6 = 216) (h7 : v7 = 512) (h8 : v8 = 729) :
  let s1 := 1
  let s2 := 2
  let s3 := 3
  let s4 := 4
  let s5 := 5
  let s6 := 6
  let s7 := 8
  let s8 := 9
  let sa1 := 6 * s1^2
  let sa2 := 6 * s2^2 - 4 * s2^2
  let sa3 := 6 * s3^2 - 4 * s3^2
  let sa4 := 6 * s4^2 - 4 * s4^2
  let sa5 := 6 * s5^2 - 4 * s5^2
  let sa6 := 6 * s6^2 - 4 * s6^2
  let sa7 := 6 * s7^2 - 4 * s7^2
  let sa8 := 6 * s8^2 in
  (sa1 + sa2 + sa3 + sa4 + sa5 + sa6 + sa7 + sa8 - (4 * s4^2) = 1266) :=
by
  sorry

end total_surface_area_of_cubes_structure_l628_628392


namespace sweet_potatoes_left_l628_628691

def total_sweet_potatoes : ℝ := 52.5
def amount_per_person : ℝ := 5

theorem sweet_potatoes_left (total_sweet_potatoes = 52.5) (amount_per_person = 5) :
  total_sweet_potatoes - amount_per_person * (total_sweet_potatoes / amount_per_person).floor = 2.5 :=
by
  sorry

end sweet_potatoes_left_l628_628691


namespace sqrt_meaningful_iff_l628_628517

theorem sqrt_meaningful_iff (x : ℝ) : (3 - x ≥ 0) ↔ (x ≤ 3) := by
  sorry

end sqrt_meaningful_iff_l628_628517


namespace product_of_common_divisors_180_18_l628_628420

theorem product_of_common_divisors_180_18 : 
  (∏ d in ({1, -1, 2, -2, 3, -3, 6, -6, 9, -9, 18, -18} : Finset Int), d) = 3294172 := 
sorry

end product_of_common_divisors_180_18_l628_628420


namespace minimum_t_value_l628_628859

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_t_value :
  (∃ t : ℝ, (∀ x1 x2 ∈ set.Icc (-3 : ℝ) 2, |f x1 - f x2| ≤ t) ∧
  ∀ t' : ℝ, (∀ x1 x2 ∈ set.Icc (-3 : ℝ) 2, |f x1 - f x2| ≤ t') → t ≤ t') :=
by
  let t := 20
  use t
  split
  sorry
  intro t' h
  have h1: ∃ x1 x2 ∈ set.Icc (-3 : ℝ) 2, |f x1 - f x2| = t := sorry
  apply le_of_forall_sub_le
  intros ε hε
  specialize h1
  cases h1 with x1 hx1
  cases hx1 with hx1_range hx1_val
  specialize h x1_left x1_right hx1_range(hx1_range_right)
  linarith

end minimum_t_value_l628_628859


namespace chocolate_cost_in_promotion_l628_628365

/-!
Bernie buys two chocolates every week at a local store, where one chocolate costs $3.
In a different store with a promotion, each chocolate costs some amount and Bernie would save $6 
in three weeks if he bought his chocolates there. Prove that the cost of one chocolate 
in the store with the promotion is $2.
-/

theorem chocolate_cost_in_promotion {n p_local savings : ℕ} (weeks : ℕ) (p_promo : ℕ)
  (h_n : n = 2)
  (h_local : p_local = 3)
  (h_savings : savings = 6)
  (h_weeks : weeks = 3)
  (h_promo : p_promo = (p_local * n * weeks - savings) / (n * weeks)) :
  p_promo = 2 :=
by {
  -- Proof would go here
  sorry
}

end chocolate_cost_in_promotion_l628_628365


namespace go_state_space_complexity_vs_universe_atoms_l628_628346

noncomputable def M : ℝ := 3^361
noncomputable def N : ℝ := 10^80
noncomputable def lg3_approx : ℝ := 0.48

theorem go_state_space_complexity_vs_universe_atoms :
  (M / N) ≈ 10^93 :=
by
  have H1 : 3 = 10^lg3_approx, by sorry
  have H2 : M ≈ (10^lg3_approx)^361, by sorry
  have H3 : (10^lg3_approx)^361 ≈ 10^173, by sorry
  have H4 : N ≈ 10^80, by sorry
  have H5 : (10^173 / 10^80) ≈ 10^93, by sorry
  sorry

end go_state_space_complexity_vs_universe_atoms_l628_628346


namespace trig_values_l628_628819

variables (α : ℝ)
noncomputable def f (α : ℝ) : ℝ :=
  (2 * sin (Real.pi + α) + cos (2 * Real.pi + α)) / (cos (α - Real.pi / 2) + sin (Real.pi / 2 + α))

theorem trig_values (h1 : sin α = -3/5) (h2 : π < α ∧ α < 3 * π / 2) :
  cos α = -4/5 ∧ tan α = 3/4 ∧ f α = -2/7 :=
by
  sorry

end trig_values_l628_628819


namespace solve_for_b_l628_628244

theorem solve_for_b :
  (∀ (x y : ℝ), 4 * y - 3 * x + 2 = 0) →
  (∀ (x y : ℝ), 2 * y + b * x - 1 = 0) →
  (∃ b : ℝ, b = 8 / 3) := 
by
  sorry

end solve_for_b_l628_628244


namespace deck_return_initial_order_l628_628952

theorem deck_return_initial_order 
  (K N : ℕ) (h1 : 1 ≤ K) (h2 : K ≤ N) :
  ∃ n : ℕ, n ≤ 4 * (N^2) / (K^2) ∧ 
  (∀ (deck_initial : list ℕ), 
  (perm_repeat (λ deck, (perm_reverse (take K deck) ++ drop K deck)) n deck_initial) = deck_initial) :=
sorry

end deck_return_initial_order_l628_628952


namespace trapezoid_smallest_angle_l628_628734

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end trapezoid_smallest_angle_l628_628734


namespace min_value_sin_tan_l628_628520

variables {A B C a b c : ℝ}

-- Given conditions
def given_conditions : Prop := (a^2 + b^2 + 4 * Real.sqrt 2 = c^2) ∧ (a * b = 4)

-- The goal to prove
theorem min_value_sin_tan : given_conditions → 
  ∃ m, m = (3 * Real.sqrt 2) / 2 + 2 ∧ 
  ∀ x, x = (Real.sin C / (Real.tan A ^ 2 * Real.sin (2 * B))) → x ≥ m :=
by sorry

end min_value_sin_tan_l628_628520


namespace nina_total_running_distance_l628_628208

theorem nina_total_running_distance :
  let r1 := 0.08333333333333333
  let r2 := 0.08333333333333333
  let r3 := 0.6666666666666666
  r1 + r2 + r3 = 0.8333333333333333 :=
by
  rfl -- sorry to skip the proof since rfl suffice in such a case

end nina_total_running_distance_l628_628208


namespace f_2010_2011_l628_628835

def f : ℕ+ × ℕ+ → ℕ+
| ⟨m, 1⟩ := 2 ^ (m - 1)
| ⟨m, n + 1⟩ := f (⟨m, n⟩) + 2

theorem f_2010_2011 : f (⟨2010, 2011⟩) = 2 ^ 2010 + 4020 :=
sorry

end f_2010_2011_l628_628835


namespace find_angle_A_max_area_triangle_ABC_l628_628844

-- Definitions for the given conditions
variables (a b c A B C : ℝ) (ABC : Type) [triangle ABC]
variables (D : point) (BC : line)
variable  h1 : sideoptoanglesABC ABC a b c

-- Condition: \( (2b - c) \cos A = a \cos C \)
variable  h2 : (2 * b - c) * cos A = a * cos C
-- Condition: \( D \) is on side \( BC \) such that \( BD = 2DC \)
variable  h3 : on_side D BC ∧ BD = 2 * DC
-- Condition: \( AD = 2 \)
variable  h4 : length (line A D) = 2

-- Statement for the first proof
theorem find_angle_A (h1 h2 : h1) : A = π / 3 := 
sorry

-- Statement for the second proof
theorem max_area_triangle_ABC (h1 h2 h3 h4 : h1 h2 h3 h4) : area_triangle_ABC ABC = (3 * sqrt 3) / 2 := 
sorry

end find_angle_A_max_area_triangle_ABC_l628_628844


namespace simplify_expression_l628_628226

variable {x y z : ℝ}

theorem simplify_expression (h : x^2 - y^2 ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ := by
  sorry

end simplify_expression_l628_628226


namespace fan_shaped_field_area_l628_628316

-- Defining the perimeter, diameter, and radius of the circle
def arc_length : ℝ := 30
def diameter : ℝ := 16
def radius : ℝ := diameter / 2

-- Defining the formula for the area of a sector
def sector_area (r l : ℝ) : ℝ := (1 / 2) * r * l

-- Stating the problem
theorem fan_shaped_field_area : 
  sector_area radius arc_length = 120 :=
by
  -- This is where you would provide a proof.
  sorry

end fan_shaped_field_area_l628_628316


namespace sum_valid_three_digit_numbers_l628_628796

def digits_without_0_or_5 : Set ℕ := {1, 2, 3, 4, 6, 7, 8, 9}

def is_valid_digit (d : ℕ) : Prop := d ∈ digits_without_0_or_5

def is_valid_three_digit (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧ is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

theorem sum_valid_three_digit_numbers :
  (∑ n in Finset.filter is_valid_three_digit (Finset.range 1000), n) = 284160 :=
  by sorry

end sum_valid_three_digit_numbers_l628_628796


namespace geometric_sequence_sum_is_five_eighths_l628_628822

noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) : ℝ :=
  if q = 1 then 4 * a₁ else a₁ * (1 - q^4) / (1 - q)

theorem geometric_sequence_sum_is_five_eighths
  (a₁ q : ℝ)
  (h₀ : q ≠ 1)
  (h₁ : a₁ * (a₁ * q) * (a₁ * q^2) = -1 / 8)
  (h₂ : 2 * (a₁ * q^2) = a₁ * q + a₁ * q^2) :
  geometric_sequence_sum a₁ q = 5 / 8 := by
sorry

end geometric_sequence_sum_is_five_eighths_l628_628822


namespace total_earnings_correct_l628_628302

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l628_628302


namespace friends_total_candies_l628_628992

noncomputable def total_candies (T S J C V B : ℕ) : ℕ :=
  T + S + J + C + V + B

theorem friends_total_candies :
  let T := 22
  let S := 16
  let J := T / 2
  let C := 2 * S
  let V := J + S
  let B := (T + C) / 2 + 9
  total_candies T S J C V B = 144 := by
  sorry

end friends_total_candies_l628_628992


namespace rearrangement_inequality_l628_628868

theorem rearrangement_inequality 
{n : ℕ} {a b : Fin n → ℝ}
(h_a : ∀ i j : Fin n, i ≤ j → a i ≤ a j)
(h_b : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
(1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, a i * b (⟨n - 1 - i.1, sorry⟩ : Fin n))) ≤
((1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, a i))) * 
((1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, b i))) ∧
((1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, a i))) * 
((1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, b i))) ≤
(1 / (n : ℝ)) * (Finset.sum (Finset.univ) (λ i, a i * b i)) :=
sorry

end rearrangement_inequality_l628_628868


namespace biased_coin_heads_probability_l628_628287

theorem biased_coin_heads_probability :
  ∃ h : ℚ, 0 < h ∧ h < 1 ∧
  (∃ k : ℚ, k = 21 * (1 - h) / (35 * h) ∧ k = h) ∧
  let heads_prob := (35 * (3 / 8)^4 * (5 / 8)^3) in
  ∃ i j : ℤ, (i : ℚ) / j = heads_prob ∧ int.gcd i j = 1 ∧ i + j = 647 :=
by {
  sorry -- proof goes here
}

end biased_coin_heads_probability_l628_628287


namespace mean_temperature_correct_l628_628233

-- Define the condition (temperatures)
def temperatures : List Int :=
  [-6, -3, -3, -4, 2, 4, 1]

-- Define the total number of days
def num_days : ℕ := 7

-- Define the expected mean temperature
def expected_mean : Rat := (-6 : Int) / (7 : Int)

-- State the theorem that we need to prove
theorem mean_temperature_correct :
  (temperatures.sum : Rat) / (num_days : Rat) = expected_mean := 
by
  sorry

end mean_temperature_correct_l628_628233


namespace tan_alpha_quad4_l628_628456

theorem tan_alpha_quad4 (α : ℝ) (h1 : π / 2 + α = arcsin (4 / 5)) (h2 : cos α > 0) (h3 : sin α < 0) :
  tan α = -3 / 4 := by
  sorry

end tan_alpha_quad4_l628_628456


namespace find_m_value_l628_628099

theorem find_m_value
  (m : ℝ) 
  (dA1 dA2 dA3 : ℝ) 
  (B1 B2 B3 : ℝ) 
  (dist : ℝ)
  (geom_condition : dist = real.sqrt ((m - B1)^2 + (dA2 - B2)^2 + (dA3 - B3)^2))
  (distance_condition : dist = 2 * real.sqrt 2 ) : m = 1 :=
by
  sorry

end find_m_value_l628_628099


namespace polynomial_eval_l628_628943

noncomputable def P (x : ℝ) : ℝ := ∑ i in (finRange n), (b i) * x ^ i

theorem polynomial_eval 
  (n : ℕ) (b : fin (n + 1) → ℤ) 
  (h1 : ∀ i, 0 ≤ (b i) ∧ (b i) < 4) 
  (h2 : P(ℝ.sqrt 5) = 31 + 26 * ℝ.sqrt 5) 
  : P 3 = 46 := sorry

end polynomial_eval_l628_628943


namespace product_of_invertibles_mod_120_l628_628122

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628122


namespace no_infinite_arithmetic_progression_divisible_l628_628545

-- Definitions based on the given condition
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

def product_divisible_by_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
(a n * a (n+1) * a (n+2) * a (n+3) * a (n+4) * a (n+5) * a (n+6) * a (n+7) * a (n+8) * a (n+9)) %
(a n + a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6) + a (n+7) + a (n+8) + a (n+9)) = 0

-- Final statement to be proven
theorem no_infinite_arithmetic_progression_divisible :
  ¬ ∃ (a : ℕ → ℕ), is_arithmetic_progression a ∧ ∀ n : ℕ, product_divisible_by_sum a n := 
sorry

end no_infinite_arithmetic_progression_divisible_l628_628545


namespace a_plus_2b_eq_21_l628_628883

-- Definitions and conditions based on the problem statement
def a_log_250_2_plus_b_log_250_5_eq_3 (a b : ℤ) : Prop :=
  a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3

-- The theorem that needs to be proved
theorem a_plus_2b_eq_21 (a b : ℤ) (h : a_log_250_2_plus_b_log_250_5_eq_3 a b) : a + 2 * b = 21 := 
  sorry

end a_plus_2b_eq_21_l628_628883


namespace net_change_in_onions_l628_628222

-- Definitions for the given conditions
def onions_added_by_sara : ℝ := 4.5
def onions_taken_by_sally : ℝ := 5.25
def onions_added_by_fred : ℝ := 9.75

-- Statement of the problem to be proved
theorem net_change_in_onions : 
  onions_added_by_sara - onions_taken_by_sally + onions_added_by_fred = 9 := 
by
  sorry -- hint that proof is required

end net_change_in_onions_l628_628222


namespace ratio_of_A_to_B_is_4_l628_628980

noncomputable def A_share : ℝ := 360
noncomputable def B_share : ℝ := 90
noncomputable def ratio_A_B : ℝ := A_share / B_share

theorem ratio_of_A_to_B_is_4 : ratio_A_B = 4 :=
by
  -- This is the proof that we are skipping
  sorry

end ratio_of_A_to_B_is_4_l628_628980


namespace theater_tickets_cost_l628_628361

-- Define the context and conditions
noncomputable def price_of_adult_ticket : ℝ := 3.50
def price_of_child_ticket : ℝ := price_of_adult_ticket / 2
def cost_for_5_adults_and_4_children : ℝ := 5 * price_of_adult_ticket + 4 * price_of_child_ticket
def cost_for_8_adults_and_6_children : ℝ := 8 * price_of_adult_ticket + 6 * price_of_child_ticket

-- Theorem statement
theorem theater_tickets_cost :
  (cost_for_5_adults_and_4_children = 24.50) →
  (cost_for_8_adults_and_6_children = 38.50) :=
by
  sorry -- Proof goes here

end theater_tickets_cost_l628_628361


namespace distance_after_one_hour_l628_628998

-- Definitions representing the problem's conditions
def initial_distance : ℕ := 20
def speed_athos : ℕ := 4
def speed_aramis : ℕ := 5

-- The goal is to prove that the possible distances after one hour are among the specified values
theorem distance_after_one_hour :
  ∃ d : ℕ, d = 11 ∨ d = 29 ∨ d = 21 ∨ d = 19 :=
sorry -- proof not required as per the instructions

end distance_after_one_hour_l628_628998


namespace magnitude_of_a_l628_628434

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ :=
Real.sqrt (u.1^2 + u.2^2)

theorem magnitude_of_a (λ : ℝ) :
  let a := vec (-2) λ,
      b := vec 3 1,
      sum := vec (a.1 + b.1) (a.2 + b.2)
  in dot_product sum b = 0 →
      magnitude a = 2 * Real.sqrt 5 :=
by
  intro h
  sorry

end magnitude_of_a_l628_628434


namespace zayne_total_revenue_l628_628300

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l628_628300


namespace unpainted_unit_cubes_l628_628717

-- Definition of the problem conditions
def is_unit_cube (n : ℕ) : Prop :=
  n = 4 * 4 * 4

def is_painted_face : ℕ → Prop
| 2 := true
| _ := false

-- Central part painted 2x2 region
def is_painted (face : ℕ) (x y z : ℕ) : Prop :=
  (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 2) ∧ is_painted_face face

-- Main statement
theorem unpainted_unit_cubes :
  ∀ (n : ℕ). is_unit_cube n →
  ∑ (x : ℕ) (hx : x ∈ finset.range 4),
    ∑ (y : ℕ) (hy : y ∈ finset.range 4),
    ∑ (z : ℕ) (hz : z ∈ finset.range 4),
    (¬ (is_painted 1 x y z ∨ is_painted 2 x y z ∨ is_painted 3 x y z ∨
        is_painted 4 x y z ∨ is_painted 5 x y z ∨ is_painted 6 x y z)) = 40 :=
by
  sorry

end unpainted_unit_cubes_l628_628717


namespace range_of_a_l628_628462

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x

theorem range_of_a {a : ℝ} :
  (∀ x ∈ set.Ioo (-(1:ℝ)/2) 0, deriv (λ x, f x a) x ≤ 0) ↔ (a ∈ set.Ici (3/4)) :=
by
  sorry

end range_of_a_l628_628462


namespace number_of_buses_l628_628974

theorem number_of_buses (total_students : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) (buses : ℕ)
  (h1 : total_students = 375)
  (h2 : students_per_bus = 53)
  (h3 : students_in_cars = 4)
  (h4 : buses = (total_students - students_in_cars + students_per_bus - 1) / students_per_bus) :
  buses = 8 := by
  -- We will demonstrate that the number of buses indeed equals 8 under the given conditions.
  sorry

end number_of_buses_l628_628974


namespace tile_difference_8th_9th_tile_difference_9th_10th_l628_628334

theorem tile_difference_8th_9th : (9^2 - 8^2) = 17 := 
by {
  calc
    9^2 - 8^2 = 81 - 64 : by norm_num
            ... = 17    : by norm_num
}

theorem tile_difference_9th_10th : (10^2 - 9^2) = 19 := 
by {
  calc
    10^2 - 9^2 = 100 - 81 : by norm_num
             ... = 19    : by norm_num
}

end tile_difference_8th_9th_tile_difference_9th_10th_l628_628334


namespace probability_even_product_l628_628989

-- Define spinner A and spinner C
def SpinnerA : List ℕ := [1, 2, 3, 4]
def SpinnerC : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define even and odd number sets for Spinner A and Spinner C
def evenNumbersA : List ℕ := [2, 4]
def oddNumbersA : List ℕ := [1, 3]

def evenNumbersC : List ℕ := [2, 4, 6]
def oddNumbersC : List ℕ := [1, 3, 5]

-- Define a function to check if a product is even
def isEven (n : ℕ) : Bool := n % 2 == 0

-- Probability calculation
def evenProductProbability : ℚ :=
  let totalOutcomes := (SpinnerA.length * SpinnerC.length)
  let evenA_outcomes := (evenNumbersA.length * SpinnerC.length)
  let oddA_evenC_outcomes := (oddNumbersA.length * evenNumbersC.length)
  (evenA_outcomes + oddA_evenC_outcomes) / totalOutcomes

theorem probability_even_product :
  evenProductProbability = 3 / 4 :=
by
  sorry

end probability_even_product_l628_628989


namespace sugar_cups_used_l628_628554

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l628_628554


namespace length_AE_l628_628900

noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def pointE : ℝ × ℝ := (4, 2)

theorem length_AE : length (0, 4) pointE = 5 * real.sqrt 13 / 4 :=
by {
  sorry
}

end length_AE_l628_628900


namespace find_x_l628_628286

variables (a b c d x y : ℚ)

noncomputable def modified_fraction (a b x y : ℚ) := (a + x) / (b + y)

theorem find_x (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : modified_fraction a b x y = c / d) :
  x = (b * c - a * d + y * c) / d :=
by
  sorry

end find_x_l628_628286


namespace find_sides_from_diagonals_l628_628057

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628057


namespace billion_conversion_correct_l628_628297

theorem billion_conversion_correct (n : ℕ) (h : n = 640080000) : n / 10^9 = 0.64008 := by
  rw h
  norm_num
  sorry

end billion_conversion_correct_l628_628297


namespace prod_coprime_mod_l628_628173

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628173


namespace eugene_payment_correct_l628_628912

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l628_628912


namespace arithmetic_sequence_a4_eq_1_l628_628443

theorem arithmetic_sequence_a4_eq_1 
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 ^ 2 + 2 * a 2 * a 6 + a 6 ^ 2 - 4 = 0) : 
  a 4 = 1 :=
sorry

end arithmetic_sequence_a4_eq_1_l628_628443


namespace eugene_total_cost_l628_628908

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l628_628908


namespace triangle_area_is_correct_l628_628625

noncomputable def isosceles_right_triangle_area (leg_length : ℝ) : ℝ :=
  if h : leg_length > 0 then
    (1 / 2) * leg_length * leg_length
  else 0

theorem triangle_area_is_correct :
  isosceles_right_triangle_area 3 = 4.5 :=
by
  unfold isosceles_right_triangle_area
  rw if_pos (by norm_num)
  norm_num
  sorry

end triangle_area_is_correct_l628_628625


namespace chase_travel_time_l628_628371

-- Definitions of speeds
def chase_speed (C : ℝ) := C
def cameron_speed (C : ℝ) := 2 * C
def danielle_speed (C : ℝ) := 6 * (cameron_speed C)

-- Time taken by Danielle to cover distance
def time_taken_by_danielle (C : ℝ) := 30  
def distance_travelled (C : ℝ) := (time_taken_by_danielle C) * (danielle_speed C)  -- 180C

-- Speeds on specific stretches
def cameron_bike_speed (C : ℝ) := 0.75 * (cameron_speed C)
def chase_scooter_speed (C : ℝ) := 1.25 * (chase_speed C)

-- Prove the time Chase takes to travel the same distance D
theorem chase_travel_time (C : ℝ) : 
  (distance_travelled C) / (chase_speed C) = 180 := sorry

end chase_travel_time_l628_628371


namespace find_a_l628_628381

def star (a b : ℕ) : ℕ := 3 * a - b ^ 2

theorem find_a (a : ℕ) (b : ℕ) (h : star a b = 14) : a = 10 :=
by sorry

end find_a_l628_628381


namespace product_of_invertibles_mod_120_l628_628166

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628166


namespace no_burial_needed_for_survivors_l628_628603

def isSurvivor (p : Person) : Bool := sorry
def isBuried (p : Person) : Bool := sorry
variable (p : Person) (accident : Bool)

theorem no_burial_needed_for_survivors (h : accident = true) (hsurvive : isSurvivor p = true) : isBuried p = false :=
sorry

end no_burial_needed_for_survivors_l628_628603


namespace total_potatoes_sold_is_322kg_l628_628338

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end total_potatoes_sold_is_322kg_l628_628338


namespace parabola_equation_l628_628863

noncomputable def hyperbola : set (ℝ × ℝ) :=
  { p | let (x, y) := p in (x^2 / 4) - (y^2 / 5) = 1 }

def center : (ℝ × ℝ) := (0, 0)

def right_focus : (ℝ × ℝ) := (3, 0)

def parabola_formula (p : ℝ) :=
  λ x y : ℝ, y^2 = 4 * p * x

theorem parabola_equation :
  parabola_formula 3 12 :=
sorry

end parabola_equation_l628_628863


namespace cat_and_mouse_position_after_365_moves_l628_628539

-- Define the cycle length for cat and mouse
def cycle_length_cat := 6
def cycle_length_mouse := 12

-- Define the positions within the 6-cycle for the cat
inductive CatPosition
| topLeft : CatPosition
| topMiddle : CatPosition
| topRight : CatPosition
| bottomRight : CatPosition
| bottomMiddle : CatPosition
| bottomLeft : CatPosition

-- Define the positions within the 12-cycle for the mouse
inductive MousePosition
| topMiddle : MousePosition
| topLeft : MousePosition
| upperLeft : MousePosition
| upperLeftMid : MousePosition
| upperMiddleLeft : MousePosition
| upperMiddle : MousePosition
| upperMiddleRight : MousePosition
| upperRightMid : MousePosition
| upperRight : MousePosition
| lowerRight : MousePosition
| bottomRight : MousePosition
| bottomMiddle : MousePosition

-- Define the positions for the cat after a given number of moves
def cat_position (moves : ℕ) : CatPosition :=
  let remainder := moves % cycle_length_cat in
  match remainder with
  | 0 => CatPosition.topLeft
  | 1 => CatPosition.topMiddle
  | 2 => CatPosition.topRight
  | 3 => CatPosition.bottomRight
  | 4 => CatPosition.bottomMiddle
  | 5 => CatPosition.bottomLeft
  | _ => CatPosition.topLeft -- to handle impossible case due to modulo operation
  end

-- Define the positions for the mouse after a given number of moves
def mouse_position (moves : ℕ) : MousePosition :=
  let remainder := moves % cycle_length_mouse in
  match remainder with
  | 0 => MousePosition.topMiddle
  | 1 => MousePosition.topLeft
  | 2 => MousePosition.upperLeft
  | 3 => MousePosition.upperLeftMid
  | 4 => MousePosition.upperMiddleLeft
  | 5 => MousePosition.bottomRight
  | 6 => MousePosition.upperMiddle
  | 7 => MousePosition.upperMiddleRight
  | 8 => MousePosition.upperRightMid
  | 9 => MousePosition.upperRight
  | 10 => MousePosition.lowerRight
  | 11 => MousePosition.bottomMiddle
  | _ => MousePosition.topMiddle -- to handle impossible case due to modulo operation
  end

-- The proof problem statement
theorem cat_and_mouse_position_after_365_moves :
  cat_position 365 = CatPosition.bottomMiddle ∧ mouse_position 365 = MousePosition.bottomRight :=
by
  -- Proof with 'sorry' to indicate unfinished proof
  sorry

end cat_and_mouse_position_after_365_moves_l628_628539


namespace line_l_problem_l628_628477

theorem line_l_problem (a : ℝ) :
  let line_l := (a^2 + a + 1) * x - y + 1 = 0 in
  (a = -1 → (∀ x y, (x + y = 0) → line_l → (-1 : ℝ) = 1)) ∧
  ((∀ x y, (x - y = 0) → (a^2 + a + 1 = 1) → (a * (a + 1) = 0 → (a = 0 ∨ a = -1))) = false) ∧
  (line_l ∧ (∀ x, x = 0 → 1 = 1) = true) ∧
  (a = 0 → let intercepts := (1 * x - y + 1 = 0) in
    (∀ x, x = -1) ∧ (∀ y, y = 1) = (false)) :=
sorry

end line_l_problem_l628_628477


namespace product_of_invertibles_mod_120_l628_628126

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628126


namespace smallest_positive_period_minimum_value_in_interval_l628_628465

namespace MySolution

def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ p > 0, ∀ x, f (x + p) = f x :=
sorry

theorem minimum_value_in_interval :
  ∃ a ∈ Set.Icc 0 (Real.pi / 2), f a = -1 / 2 :=
sorry

end MySolution

end smallest_positive_period_minimum_value_in_interval_l628_628465


namespace product_coprime_mod_120_l628_628190

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628190


namespace symmetry_center_cos2theta_sintheta_costheta_eq_neg1_l628_628516

theorem symmetry_center_cos2theta_sintheta_costheta_eq_neg1 (θ : ℝ) 
  (h : sin θ + 2 * cos θ = 0) : cos (2 * θ) + sin θ * cos θ = -1 := 
sorry

end symmetry_center_cos2theta_sintheta_costheta_eq_neg1_l628_628516


namespace polygon_diagonals_l628_628048

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628048


namespace percent_own_cats_l628_628915

theorem percent_own_cats (total_students cats_students : ℕ) (h1 : total_students = 500) (h2 : cats_students = 90) :
  cats_students * 100 / total_students = 18 :=
by
  rw [h1, h2]
  norm_num
  sorry

end percent_own_cats_l628_628915


namespace isosceles_triangle_leg_length_l628_628448

theorem isosceles_triangle_leg_length :
  ∀ (x : ℝ), x ^ 2 - 9 * x + 20 = 0 → x = 5 → 
  ∀ (a b c : ℝ), (a = x → b = x → c = 8) → (a + b > c ∧ a + c > b ∧ b + c > a) → 
  (a = 5 ∧ b = 5 ∧ c = 8) :=
begin
  sorry
end

end isosceles_triangle_leg_length_l628_628448


namespace infinite_primes_dividing_polynomial_l628_628562

open Polynomial Int

theorem infinite_primes_dividing_polynomial (P : ℤ[X]) (h_non_constant : ¬ constantCoeff P = 0) : 
  ∃ (primes : ℕ → Prop), set.infinite { p | prime p ∧ ∃ n : ℤ, p ∣ eval n P } :=
sorry

end infinite_primes_dividing_polynomial_l628_628562


namespace apple_juice_consumption_l628_628993

theorem apple_juice_consumption :
  let apples_total := 6.0 -- million tons
  let export_percentage := 0.30
  let juice_percentage := 0.40
  let remainder := apples_total * (1 - export_percentage)
  let juice := remainder * juice_percentage
  (Float.round(juice * 10) / 10) = 1.7 :=
by
  let apples_total := 6.0
  let export_percentage := 0.30
  let juice_percentage := 0.40
  let remainder := apples_total * (1 - export_percentage)
  let juice := remainder * juice_percentage
  show (Float.round(juice * 10) / 10) = 1.7
  sorry

end apple_juice_consumption_l628_628993


namespace d_is_greatest_l628_628950

variable (p : ℝ)

def a := p - 1
def b := p + 2
def c := p - 3
def d := p + 4

theorem d_is_greatest : d > b ∧ d > a ∧ d > c := 
by sorry

end d_is_greatest_l628_628950


namespace find_tents_l628_628848

theorem find_tents (x y : ℕ) (hx : x + y = 600) (hy : 1700 * x + 1300 * y = 940000) : x = 400 ∧ y = 200 :=
by
  sorry

end find_tents_l628_628848


namespace greatest_percentage_increase_l628_628903

def pop1970_F := 30000
def pop1980_F := 45000
def pop1970_G := 60000
def pop1980_G := 75000
def pop1970_H := 40000
def pop1970_I := 20000
def pop1980_combined_H := 70000
def pop1970_J := 90000
def pop1980_J := 120000

def percentage_increase (pop1970 pop1980 : ℕ) : ℚ :=
  ((pop1980 - pop1970 : ℚ) / pop1970) * 100

theorem greatest_percentage_increase :
  ∀ (city : ℕ), (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_G pop1980_G) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase (pop1970_H + pop1970_I) pop1980_combined_H) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_J pop1980_J) := by 
  sorry

end greatest_percentage_increase_l628_628903


namespace evaluate_expression_correct_l628_628783
noncomputable def evaluate_expression : ℝ := sorry

theorem evaluate_expression_correct :
  2^(Real.logb 2 (1 / 4)) - (8 / 27)^(-2 / 3) + Real.log10 (1 / 100) + (Real.sqrt 2 - 1)^(Real.log10 1) = -3 :=
sorry

end evaluate_expression_correct_l628_628783


namespace find_NCB_angle_l628_628102

-- Define the given angles in the triangle and the required internal point N
def triangle_angles (A B C N : Type) [IsTriangle A B C]
  (angle_ABC : ℕ) (angle_ACB : ℕ)
  (angle_NBC : ℕ) (angle_NAB : ℕ) : Prop :=
  angle_ABC = 40 ∧ angle_ACB = 20 ∧ angle_NBC = 30 ∧ angle_NAB = 20
  
-- Main theorem stating our proof goal
theorem find_NCB_angle (A B C N : Type) [IsTriangle A B C]
  (h : triangle_angles A B C N 40 20 30 20) : 
  ∃ (angle_NCB : ℕ), angle_NCB = 10 :=
sorry

end find_NCB_angle_l628_628102


namespace log_geometric_seq_is_arithmetic_l628_628591

theorem log_geometric_seq_is_arithmetic (a q k : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < q) (h3 : 1 < k): 
  ∃ d, ∀ m, (m ≤ n) → (log k (a * q ^ (m + 1)) - log k (a * q ^ m) = log k q) :=
sorry

end log_geometric_seq_is_arithmetic_l628_628591


namespace num_proper_subsets_A_l628_628480

theorem num_proper_subsets_A : 
  let A := {x : ℤ | ((x + 1) / (x - 3) : ℝ) ≤ 0}
  ∃ S : finset ℤ, (S = A) ∧ S.card = 4 → 2^S.card - 1 = 15 := 
by
  let A := {x : ℤ | ((x + 1) / (x - 3) : ℝ) ≤ 0}
  use finset.filter (λ x, x ∈ A) (finset.Icc (-1) 2)
  sorry

end num_proper_subsets_A_l628_628480


namespace product_of_invertible_integers_mod_120_l628_628141

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628141


namespace log_sum_to_one_l628_628020

theorem log_sum_to_one (a b : ℝ) (h1 : 4 ^ a = 24) (h2 : 6 ^ b = 24) : (1 / a) + (1 / b) = 1 := by
  sorry

end log_sum_to_one_l628_628020


namespace product_coprime_mod_120_l628_628182

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628182


namespace distinct_values_of_g_l628_628196

def floor (r : ℝ) : ℤ := int.floor r

def g (x : ℝ) : ℤ :=
  (Finset.range 10).sum (λ k : ℕ, floor ((k + 3) * x) - (k + 3) * floor x)

theorem distinct_values_of_g : ∀ (x : ℝ), x ≥ 0 → g(x) = 45 :=
by
  intros x hx
  sorry

end distinct_values_of_g_l628_628196


namespace total_cable_cost_l628_628367

theorem total_cable_cost (ew_streets : ℕ) (ew_length : ℕ) (ns_streets : ℕ) (ns_length : ℕ)
    (cable_per_mile : ℕ) (cost_per_mile : ℕ) (total_length := ew_streets * ew_length + ns_streets * ns_length)
    (total_cable := total_length * cable_per_mile) :
    ew_streets = 18 →
    ew_length = 2 →
    ns_streets = 10 →
    ns_length = 4 →
    cable_per_mile = 5 →
    cost_per_mile = 2000 →
    total_cable * cost_per_mile = 760000 := by
  intros
  rw [total_length, total_cable]
  -- Arithmetic calculations to be filled
  sorry

end total_cable_cost_l628_628367


namespace simplify_f_value_f_second_quadrant_l628_628817

noncomputable def f (α : ℝ) : ℝ :=
  (cos (π - α) * cos ((3 * π / 2) + α)) / sin (α - π)

-- First part: prove the simplified form
theorem simplify_f (α : ℝ) : f α = cos α :=
by
  -- Simplification proof comes here
  sorry

-- Second part: given conditions, prove the specific value of f(α)
theorem value_f_second_quadrant (α : ℝ) (h1 : α > π / 2 ∧ α < π) (h2 : cos (α - π/2) = 3 / 5) :
  f α = -4 / 5 :=
by
  -- Proof using given conditions comes here
  sorry

end simplify_f_value_f_second_quadrant_l628_628817


namespace square_of_cube_of_third_smallest_prime_l628_628680

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l628_628680


namespace social_logistics_turnover_scientific_notation_l628_628078

noncomputable def total_social_logistics_turnover_2022 : ℝ := 347.6 * (10 ^ 12)

theorem social_logistics_turnover_scientific_notation :
  total_social_logistics_turnover_2022 = 3.476 * (10 ^ 14) :=
by
  sorry

end social_logistics_turnover_scientific_notation_l628_628078


namespace cone_height_l628_628720

theorem cone_height (h : ℝ) (r : ℝ) 
  (volume_eq : (1/3) * π * r^2 * h = 19683 * π) 
  (isosceles_right_triangle : h = r) : 
  h = 39.0 :=
by
  -- The proof will go here
  sorry

end cone_height_l628_628720


namespace john_new_earnings_after_raise_l628_628107

-- Definition of original earnings and raise percentage
def original_earnings : ℝ := 50
def raise_percentage : ℝ := 0.50

-- Calculate raise amount and new earnings after raise
def raise_amount : ℝ := raise_percentage * original_earnings
def new_earnings : ℝ := original_earnings + raise_amount

-- Math proof problem: Prove new earnings after raise equals $75
theorem john_new_earnings_after_raise : new_earnings = 75 := by
  sorry

end john_new_earnings_after_raise_l628_628107


namespace total_length_of_segments_in_new_figure_l628_628347

-- Defining the given conditions.
def left_side := 10
def top_side := 3
def right_side := 8
def segments_removed_from_bottom := [2, 1, 2] -- List of removed segments from the bottom.

-- This is the theorem statement that confirms the total length of the new figure's sides.
theorem total_length_of_segments_in_new_figure :
  (left_side + top_side + right_side) = 21 :=
by
  -- This is where the proof would be written.
  sorry

end total_length_of_segments_in_new_figure_l628_628347


namespace square_of_cube_of_third_smallest_prime_l628_628667

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628667


namespace triangle_area_leq_half_rectangle_area_l628_628642

theorem triangle_area_leq_half_rectangle_area (A B C D K L M : ℝ × ℝ)
  (h_rect : convex_hull ℝ {A, B, C, D} = rectangle ℝ A B C D)
  (h_tri : K ∈ convex_hull ℝ {A, B, C, D} ∧ L ∈ convex_hull ℝ {A, B, C, D} ∧ M ∈ convex_hull ℝ {A, B, C, D}) :
  area (triangle K L M) ≤ 1/2 * area (rectangle ℝ A B C D) :=
sorry

end triangle_area_leq_half_rectangle_area_l628_628642


namespace number_of_intersections_l628_628245

-- Define the four logarithmic functions
def f1 (x : ℝ) : ℝ := Real.logBase 3 x
def f2 (x : ℝ) : ℝ := (Real.logBase 3 3) / (Real.logBase 3 x) -- equivalent to 1 / log_3(x)
def f3 (x : ℝ) : ℝ := -(Real.logBase 3 x)
def f4 (x : ℝ) : ℝ := -(Real.logBase 3 3) / (Real.logBase 3 x) -- equivalent to -1 / log_3(x)

-- Define a condition for intersection of graphs
def is_intersection (x : ℝ) (y : ℝ) : Prop := 
  (y = f1 x ∨ y = f2 x ∨ y = f3 x ∨ y = f4 x) 

-- Define the problem statement
theorem number_of_intersections : 
  ∃! p : ℝ × ℝ, 0 < p.1 ∧ (is_intersection p.1 p.2) := sorry

end number_of_intersections_l628_628245


namespace student_correct_answers_l628_628342

theorem student_correct_answers (C I : ℕ) (h₁ : C + I = 100) (h₂ : C - 2 * I = 61) : C = 87 :=
sorry

end student_correct_answers_l628_628342


namespace product_of_invertible_integers_mod_120_l628_628145

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628145


namespace shaded_ratio_l628_628991

theorem shaded_ratio (full_rectangles half_rectangles : ℕ) (n m : ℕ) (rectangle_area shaded_area total_area : ℝ)
  (h1 : n = 4) (h2 : m = 5) (h3 : rectangle_area = n * m) 
  (h4 : full_rectangles = 3) (h5 : half_rectangles = 4)
  (h6 : shaded_area = full_rectangles * 1 + 0.5 * half_rectangles * 1)
  (h7 : total_area = rectangle_area) :
  shaded_area / total_area = 1 / 4 := by
  sorry

end shaded_ratio_l628_628991


namespace product_invertibles_mod_120_l628_628156

theorem product_invertibles_mod_120 :
  let n := 120
  let m : ℕ := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  m % n = 1 := 
by
  let n := 120
  let m := ∏ i in (Finset.filter (λ x, Nat.coprime x n) (Finset.range n)), i
  sorry

end product_invertibles_mod_120_l628_628156


namespace inequality_solution_l628_628229

theorem inequality_solution (x : ℝ) :
  (x^2 - 4 * x - 11 > 0) ∧ (-3 * x^2 - 5 * x + 2 ≠ 0) →
  (x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo 6 ∞ ∪ Set.Ioo (-2) (2 - Real.sqrt 15)) :=
sorry

end inequality_solution_l628_628229


namespace limit_tangent_three_x_l628_628314

open Real

theorem limit_tangent_three_x :
  (Real.limit (fun x => (Real.tan (3 * x)) / (Real.tan x)) (Real.pi / 2)) = (1 / 3) :=
by
  sorry

end limit_tangent_three_x_l628_628314


namespace sum_of_all_valid_three_digit_numbers_l628_628799

-- Define the set of valid digits
def valid_digits : Finset ℕ := {1, 2, 3, 4, 6, 7, 8, 9}

-- Define a three-digit number using valid digits
def is_valid_number (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n in
  n >= 100 ∧ n < 1000 ∧
  (digits.get 2 (by simp [Nat.digits_len 10 n])) ∈ valid_digits ∧ -- hundreds place
  (digits.get 1 (by simp [Nat.digits_len 10 n])) ∈ valid_digits ∧ -- tens place
  (digits.get 0 (by simp [Nat.digits_len 10 n])) ∈ valid_digits   -- units place

-- Define the sum of all valid three-digit numbers
def sum_of_valid_numbers : ℕ :=
  (Finset.filter is_valid_number (Finset.Icc 100 999)).sum id

theorem sum_of_all_valid_three_digit_numbers :
  sum_of_valid_numbers = 284160 :=
sorry -- Proof omitted

end sum_of_all_valid_three_digit_numbers_l628_628799


namespace trigonometric_system_solution_l628_628699

theorem trigonometric_system_solution:
  {x y : ℝ} (
    sin_x: Real.sin x + Real.cos y = 0) 
    (sin_squared_cos_squared: Real.sin x ^ 2 + Real.cos y ^ 2 = 0.5):
    (∃ k n : ℤ, x = ((-1:ℤ)^((k:ℤ)+1) : ℝ) * Real.pi / 6 + (k : ℝ) * Real.pi ∧
                 y = (±(Real.pi / 3)) + (2:ℝ) * (n : ℝ) * Real.pi) ∨
    (∃ k n : ℤ, x = ((-1:ℤ) ^ (k:ℤ) : ℝ) * Real.pi / 6 + (k : ℝ) * Real.pi ∧
                 y = (±(2:ℝ) * Real.pi / 3) + (2:ℝ) * (n : ℝ) * Real.pi) :=
sorry

end trigonometric_system_solution_l628_628699


namespace find_200_digit_number_l628_628714

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end find_200_digit_number_l628_628714


namespace car_circuit_velocity_solution_l628_628327

theorem car_circuit_velocity_solution
    (v_s v_p v_d : ℕ)
    (h1 : v_s < v_p)
    (h2 : v_p < v_d)
    (h3 : s = d)
    (h4 : s + p + d = 600)
    (h5 : (d : ℚ) / v_s + (p : ℚ) / v_p + (d : ℚ) / v_d = 50) :
    (v_s = 7 ∧ v_p = 12 ∧ v_d = 42) ∨
    (v_s = 8 ∧ v_p = 12 ∧ v_d = 24) ∨
    (v_s = 9 ∧ v_p = 12 ∧ v_d = 18) ∨
    (v_s = 10 ∧ v_p = 12 ∧ v_d = 15) :=
by
  sorry

end car_circuit_velocity_solution_l628_628327


namespace determine_forest_width_l628_628724

def forest_width (W : ℕ) : Prop :=
  let total_trees : ℕ := 4 * W * 600
  let trees_per_logger_per_month : ℕ := 6 * 30
  let total_cut_trees : ℕ := 8 * trees_per_logger_per_month * 10
  total_trees = total_cut_trees

theorem determine_forest_width : forest_width 6 :=
by
  unfold forest_width
  simp
  sorry

end determine_forest_width_l628_628724


namespace employees_drive_more_l628_628323

theorem employees_drive_more {total_employees drivers_ratio non_drivers_ratio : ℝ}
  (h_total : total_employees = 200) 
  (h_drivers : drivers_ratio = 0.60)
  (h_non_drivers : non_drivers_ratio = 0.5) :
  let drivers := drivers_ratio * total_employees in
  let non_drivers := total_employees - drivers in
  let public_transport := non_drivers_ratio * non_drivers in
  drivers - public_transport = 80 := 
by 
  rw [h_total, h_drivers, h_non_drivers] 
  let drivers := 0.60 * 200 
  let non_drivers := 200 - drivers 
  let public_transport := 0.5 * non_drivers 
  suffices drivers - public_transport = 80
  sorry

end employees_drive_more_l628_628323


namespace parallel_midline_angle_bisector_l628_628265

noncomputable def Triangle (A B C : Type) := ∃ inscribed_in_circle: ∀ X Y Z : Type, True

noncomputable def midpoint (X Y : Type) : Type := sorry

noncomputable def perpendicular (X Y Z : Type) : Prop := sorry

noncomputable def is_midpoint (M X Y : Type) : Prop := sorry

noncomputable def parallel (X Y : Type) : Prop := sorry

theorem parallel_midline_angle_bisector
    (A B C D K M1 : Type)
    (H1 : Triangle A B C)
    (H2 : D = midpoint (arc A B C) )
    (H3 : perpendicular D K A C)
    (H4 : is_midpoint M1 B C) :
    parallel (segment K M1) (angle_bisector A) :=
sorry

end parallel_midline_angle_bisector_l628_628265


namespace tournament_games_l628_628523

theorem tournament_games (n : ℕ) (h : n = 512) :
  let defeats_per_non_winner := 3,
      total_non_winners := n - 1,
      max_defeats_winner := 2,
      total_games := (total_non_winners * defeats_per_non_winner) + max_defeats_winner
  in total_games = 1535 :=
by
  sorry

end tournament_games_l628_628523


namespace circumcircles_intersect_at_one_point_l628_628321

theorem circumcircles_intersect_at_one_point
  (ABC : Triangle) [IsCircumscribed ABC] [Acute ABC]
  (A1 B1 C1 : Point)
  (hA1 : SmallArcPoint ABC B C A1)
  (hB1 : SmallArcPoint ABC A C B1)
  (hC1 : SmallArcPoint ABC A B C1)
  (A2 B2 C2 : Point)
  (hA2 : Orthocenter (Triangle.mk B A1 C))
  (hB2 : Orthocenter (Triangle.mk A B1 C))
  (hC2 : Orthocenter (Triangle.mk A C1 B)) :
  ∃ M : Point, EquivCircle (Triangle.mk B A2 C) M ∧ EquivCircle (Triangle.mk A B2 C) M ∧ EquivCircle (Triangle.mk A C2 B) M :=
sorry

end circumcircles_intersect_at_one_point_l628_628321


namespace prod_coprime_mod_l628_628177

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628177


namespace imaginary_part_zero_l628_628428

-- Definition for the complex number z
def z (b : ℝ) : ℂ := 2 + b * complex.I

-- Condition that b is a real number and not equal to zero
variable {b : ℝ} (hb : b ≠ 0)

-- The statement to prove
theorem imaginary_part_zero (hb : b ≠ 0) : complex.im (z b * complex.conj (z b)) = 0 := 
sorry

end imaginary_part_zero_l628_628428


namespace distinct_collections_of_letters_l628_628975

noncomputable def choose_ways_vowels := 4
noncomputable def choose_ways_consonants := 25

theorem distinct_collections_of_letters :
  let total_collections := choose_ways_vowels * choose_ways_consonants
  in total_collections = 100 :=
by
  sorry

end distinct_collections_of_letters_l628_628975


namespace Gina_house_units_digit_l628_628485

theorem Gina_house_units_digit (n : ℕ) (h1 : 10 ≤ n ∧ n < 100) (h2 : (n % 5 = 0 + n % 2 = 1 + n % 9 = 0 + ((n / 10) = 4 ∨ (n % 10) = 4)) = 3) : n % 10 = 5 :=
sorry

end Gina_house_units_digit_l628_628485


namespace sphere_radius_l628_628613

-- The surface area of a sphere is 64 * π cm²
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_radius (r : ℝ) (h : surface_area r = 64 * Real.pi) : r = 4 := by
  -- Proof is omitted; hence we use 'sorry'
  sorry

end sphere_radius_l628_628613


namespace solve_equation_l628_628690

theorem solve_equation :
  (3 * x - 6 = abs (-21 + 8 - 3)) → x = 22 / 3 :=
by
  intro h
  sorry

end solve_equation_l628_628690


namespace largest_t_value_maximum_t_value_l628_628416

noncomputable def largest_t : ℚ :=
  (5 : ℚ) / 2

theorem largest_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ (5 : ℚ) / 2 :=
sorry

theorem maximum_t_value (t : ℚ) :
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  (5 : ℚ) / 2 = largest_t :=
sorry

end largest_t_value_maximum_t_value_l628_628416


namespace range_of_g_l628_628422

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + (Real.pi / 4) * (Real.arcsin (x / 3)) 
    - (Real.arcsin (x / 3))^2 + (Real.pi^2 / 16) * (x^2 + 2 * x + 3)

theorem range_of_g : 
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ y ∈ (Set.Icc (Real.pi^2 / 4) (15 * Real.pi^2 / 16 + Real.pi / 4 * Real.arcsin 1)) :=
by
  sorry

end range_of_g_l628_628422


namespace carly_practice_backstroke_days_per_week_l628_628762

theorem carly_practice_backstroke_days_per_week 
  (butterfly_hours_per_day : ℕ) 
  (butterfly_days_per_week : ℕ) 
  (backstroke_hours_per_day : ℕ) 
  (total_hours_per_month : ℕ)
  (weeks_per_month : ℕ)
  (d : ℕ)
  (h1 : butterfly_hours_per_day = 3)
  (h2 : butterfly_days_per_week = 4)
  (h3 : backstroke_hours_per_day = 2)
  (h4 : total_hours_per_month = 96)
  (h5 : weeks_per_month = 4)
  (h6 : total_hours_per_month - (butterfly_hours_per_day * butterfly_days_per_week * weeks_per_month) = backstroke_hours_per_day * d * weeks_per_month) :
  d = 6 := by
  sorry

end carly_practice_backstroke_days_per_week_l628_628762


namespace max_value_of_f_value_of_tan_theta_l628_628853

def f (x : ℝ) := 2 * sin x * cos x + cos (2 * x)

theorem max_value_of_f :
  ∃ k : ℤ, f (k * π + π / 8) = sqrt 2 :=
sorry

theorem value_of_tan_theta
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (h_f : f (θ + π / 8) = sqrt 2 / 3) :
  tan θ = sqrt 2 / 2 :=
sorry

end max_value_of_f_value_of_tan_theta_l628_628853


namespace four_digit_perfect_square_l628_628406

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l628_628406


namespace twice_total_credits_l628_628357

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end twice_total_credits_l628_628357


namespace minimum_value_in_interval_l628_628621

open Set Real

noncomputable def f (x : ℝ) : ℝ := (exp x) / x

theorem minimum_value_in_interval :
  let interval := Icc (1/2 : ℝ) 2 in
  ∃ x ∈ interval, f x = inf (f '' interval) ∧ f x = exp 1 :=
by
  let interval := Icc (1/2 : ℝ) 2
  have fx_def : ∀ x, f x = exp x / x := by simp [f]
  use 1
  simp [interval, f]
  sorry

end minimum_value_in_interval_l628_628621


namespace new_students_count_l628_628995

theorem new_students_count (O N : ℕ) (avg_class_age avg_new_students_age avg_decrease original_strength : ℕ)
  (h1 : avg_class_age = 40)
  (h2 : avg_new_students_age = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 8)
  (total_age_class : ℕ := avg_class_age * original_strength)
  (new_avg_age : ℕ := avg_class_age - avg_decrease)
  (total_age_new_students : ℕ := avg_new_students_age * N)
  (total_students : ℕ := original_strength + N)
  (new_total_age : ℕ := total_age_class + total_age_new_students)
  (new_avg_class_age : ℕ := new_total_age / total_students)
  (h5 : new_avg_class_age = new_avg_age) : N = 8 :=
by
  sorry

end new_students_count_l628_628995


namespace smallest_solution_eqn_l628_628795

theorem smallest_solution_eqn :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 24) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 24) → y ≥ x) ∧ x = 6 * Real.sqrt 2 :=
by
  use 6 * Real.sqrt 2
  split
  {
    sorry
  }
  split
  {
    sorry
  }
  {
    refl
  }

end smallest_solution_eqn_l628_628795


namespace product_of_invertibles_mod_120_l628_628154

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628154


namespace cos_sum_zero_sin_sum_zero_l628_628788

theorem cos_sum_zero (x : ℝ) : 
  cos x + cos (x + 2 * π / 3) + cos (x + 4 * π / 3) = 0 :=
  sorry

theorem sin_sum_zero (x : ℝ) : 
  sin x + sin (x + 2 * π / 3) + sin (x + 4 * π / 3) = 0 :=
  sorry

end cos_sum_zero_sin_sum_zero_l628_628788


namespace product_of_invertibles_mod_120_l628_628129

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628129


namespace probability_one_each_of_four_l628_628877

theorem probability_one_each_of_four :
  let total_items := 6 + 8 + 7 + 3,
      choose_four := (Nat.choose total_items 4),
      ways_one_each := (6 * 8 * 7 * 3)
  in choose_four ≠ 0 ∧
     (ways_one_each / choose_four = 72 / 91) := 
by
  let total_items := 24
  have total_items_def : 6 + 8 + 7 + 3 = total_items := by decide
  let choose_four := Nat.choose total_items 4
  let ways_one_each := 6 * 8 * 7 * 3
  have choose_four_def : choose_four = 12650 := by decide
  have ways_one_each_def : ways_one_each = 1008 := by decide
  have nonzero_choose_four : choose_four ≠ 0 := by decide
  have prob_eq : (ways_one_each : ℚ) / choose_four = 72 / 91 := by
    calc (ways_one_each : ℚ) / choose_four = 1008 / 12650 : by decide
                                   ...        = 72 / 91    : by decide
  exact And.intro nonzero_choose_four prob_eq

end probability_one_each_of_four_l628_628877


namespace intersection_ratios_l628_628092

noncomputable def ratio_divided_by_intersection (λ₁₂ λ₂₃ λ₃₄ λ₄₁ : ℝ) : Prop :=
  let A₁₂_O_over_O_A₃₄ := (λ₂₃ * λ₁₂ + 1) / (1 + λ₁₂)
  let A₂₃_O_over_O_A₄₁ := (λ₃₄ * λ₂₃ + 1) / (λ₂₃ + 1)
  A₁₂_O_over_O_A₃₄ = A₁₂_O_over_O_A₃₄ ∧ A₂₃_O_over_O_A₄₁ = A₂₃_O_over_O_A₄₁

theorem intersection_ratios (λ₁₂ λ₂₃ λ₃₄ λ₄₁ : ℝ) :
  ratio_divided_by_intersection λ₁₂ λ₂₃ λ₃₄ λ₄₁ :=
  by 
    sorry

end intersection_ratios_l628_628092


namespace regular_polygon_num_sides_l628_628040

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628040


namespace triangle_area_median_increase_l628_628100

noncomputable def triangle_change_area_and_median (P Q R : Type)
  (PQ PR QR : ℝ) (PQ' QR' PR' : ℝ) (area original_median changed_median : ℝ) : Prop :=
  PQ = 15 ∧ PR = 9 ∧ QR = 12 ∧
  PQ' = 1.5 * PQ ∧ QR' = 2 * QR ∧ PR' = PR ∧
  exists (a1 a2 m1 m2 : ℝ), 
    a1 = sqrt ((18) * (18 - 15) * (18 - 9) * (18 - 12)) ∧
    a2 = sqrt ((27.75) * (27.75 - 22.5) * (27.75 - 24) * (27.75 - 9)) ∧
    area = a2 ∧ a2 > 2 * a1 ∧ original_median = m1 ∧ changed_median = m2 ∧ m1 ≠ m2

theorem triangle_area_median_increase (P Q R : Type) (PQ PR QR : ℝ) (PQ' QR' PR' : ℝ) (area original_median changed_median : ℝ) :
  triangle_change_area_and_median P Q R PQ PR QR PQ' QR' PR' area original_median changed_median :=
by
  -- conditions
  have hPQ : PQ = 15 := sorry,
  have hPR : PR = 9 := sorry,
  have hQR : QR = 12 := sorry,
  have hPQ' : PQ' = 1.5 * PQ := sorry,
  have hQR' : QR' = 2 * QR := sorry,
  have hPR' : PR' = PR := sorry,

  -- exists
  existsi 54, -- a1 calculated as area of original triangle
  existsi 108, -- a2 as the new area
  existsi some_original_median, -- original median length
  existsi some_changed_median, -- changed median length

  split; assumption

end triangle_area_median_increase_l628_628100


namespace initial_total_fish_l628_628219

def total_days (weeks : ℕ) : ℕ := weeks * 7
def fish_added (rate : ℕ) (days : ℕ) : ℕ := rate * days
def initial_fish (final_count : ℕ) (added : ℕ) : ℕ := final_count - added

theorem initial_total_fish {final_goldfish final_koi rate_goldfish rate_koi days init_goldfish init_koi : ℕ}
    (h_final_goldfish : final_goldfish = 200)
    (h_final_koi : final_koi = 227)
    (h_rate_goldfish : rate_goldfish = 5)
    (h_rate_koi : rate_koi = 2)
    (h_days : days = total_days 3)
    (h_init_goldfish : init_goldfish = initial_fish final_goldfish (fish_added rate_goldfish days))
    (h_init_koi : init_koi = initial_fish final_koi (fish_added rate_koi days)) :
    init_goldfish + init_koi = 280 :=
by
    sorry -- skipping the proof

end initial_total_fish_l628_628219


namespace final_coordinates_l628_628763

-- Definitions for the given conditions
def initial_point : ℝ × ℝ := (-2, 6)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

-- The final proof statement
theorem final_coordinates :
  let S_reflected := reflect_x_axis initial_point
  let S_translated := translate_up S_reflected 10
  S_translated = (-2, 4) :=
by
  sorry

end final_coordinates_l628_628763


namespace cube_problem_l628_628738

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l628_628738


namespace greatest_distance_between_squares_l628_628339

-- Definitions of the conditions
def inner_square_perimeter : ℝ := 16
def outer_square_perimeter : ℝ := 36

-- Definition of the side lengths based on perimeters
def inner_square_side_length : ℝ := inner_square_perimeter / 4
def outer_square_side_length : ℝ := outer_square_perimeter / 4

-- Definition of the diagonals of the squares
def inner_square_diagonal : ℝ := inner_square_side_length * Real.sqrt 2
def outer_square_diagonal : ℝ := outer_square_side_length * Real.sqrt 2

-- The maximum distance between a vertex of the inner and outer squares
def max_distance (inner_diag outer_diag : ℝ) : ℝ :=
  (outer_diag / 2) - (inner_diag / 2)

-- Statement to prove
theorem greatest_distance_between_squares :
  max_distance inner_square_diagonal outer_square_diagonal = 2.5 * Real.sqrt 2 :=
by
  sorry

end greatest_distance_between_squares_l628_628339


namespace value_of_f_10_l628_628021

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem value_of_f_10 : f 10 = 107 := by
  sorry

end value_of_f_10_l628_628021


namespace four_digit_perfect_square_l628_628404

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l628_628404


namespace smallest_integer_y_l628_628663

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y ≤ 29) : y ≥ -7 := by
  have h1 : -3 * y ≤ 22 := by linarith
  have h2 : y ≥ -22 / 3 := by 
    linarith
  have h3 : -22 / 3 ≤ -7 := by norm_num
  linarith

end smallest_integer_y_l628_628663


namespace bowling_ball_weight_l628_628393

theorem bowling_ball_weight :
  (∃ (b c : ℝ), 8 * b = 4 * c ∧ 2 * c = 64) → ∃ b : ℝ, b = 16 :=
by
  sorry

end bowling_ball_weight_l628_628393


namespace find_constants_l628_628419

theorem find_constants (c d : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
     (r^3 + c*r^2 + 17*r + 10 = 0) ∧ (s^3 + c*s^2 + 17*s + 10 = 0) ∧
     (r^3 + d*r^2 + 22*r + 14 = 0) ∧ (s^3 + d*s^2 + 22*s + 14 = 0)) →
  (c = 8 ∧ d = 9) :=
by
  sorry

end find_constants_l628_628419


namespace sequence_identity_l628_628866

section
  variable {a b : ℕ → ℕ}

  -- Define sequences a_n and b_n based on the given conditions
  def a_seq : ℕ → ℕ
  | 0       => 1
  | (n + 1) => 5 * a_seq n + 7 * b_seq n

  def b_seq : ℕ → ℕ
  | 0       => 1
  | (n + 1) => 7 * a_seq n + 10 * b_seq n

  -- The main theorem statement to be proved
  theorem sequence_identity (m n : ℕ) :
    a_seq (m + n) + b_seq (m + n) = a_seq m * a_seq n + b_seq m * b_seq n :=
  sorry
end

end sequence_identity_l628_628866


namespace profit_percentage_on_cost_price_l628_628352

theorem profit_percentage_on_cost_price (CP MP SP : ℝ)
    (h1 : CP = 100)
    (h2 : MP = 131.58)
    (h3 : SP = 0.95 * MP) :
    ((SP - CP) / CP) * 100 = 25 :=
by
  -- Sorry to skip the proof
  sorry

end profit_percentage_on_cost_price_l628_628352


namespace circle_statements_correct_l628_628585

theorem circle_statements_correct
  (x₀ y₀ : ℝ)
  (h : x₀^2 + y₀^2 - 8*x₀ - 6*y₀ + 21 = 0) :
  (∀ x₀ y₀, ∀ k : ℝ, k = y₀ / (x₀ - 3) → ∃ k', ∀ k, k' ≤ k ∧ k > k') ∧
  (∃ b : ℝ, 11 - 2 * Real.sqrt 5 ≤ 2 * x₀ + y₀ ∧ 2 * x₀ + y₀ ≤ 11 + 2 * Real.sqrt 5) := 
begin
  sorry
end

end circle_statements_correct_l628_628585


namespace product_of_invertibles_mod_120_l628_628167

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628167


namespace polygon_sides_from_diagonals_l628_628032

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628032


namespace sides_of_regular_polygon_with_20_diagonals_l628_628037

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628037


namespace length_segment_AB_l628_628867

def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 - 3 = 0}
def circle2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.2 - 3 = 0}
def intersection_points := {p : ℝ × ℝ | p ∈ circle1 ∧ p ∈ circle2}

theorem length_segment_AB : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  (∥A - B∥ = 2 * real.sqrt (7 - 2)) :=
sorry

end length_segment_AB_l628_628867


namespace sum_three_digit_numbers_l628_628814

theorem sum_three_digit_numbers : 
  ∀ (A B C : ℕ), 
  A ≠ B → A ≠ C → B ≠ C → 
  0 < A ∧ A < 10 → 
  0 < B ∧ B < 10 → 
  0 < C ∧ C < 10 → 
  let num1 := 10765 
  let num2 := 100 * A + 10 * B + 4 
  let num3 := 10 * C + B 
  let total := num1 + num2 + num3
  Nat.digits 10 total = [_, _, _, _, _] := 
sorry

end sum_three_digit_numbers_l628_628814


namespace good_numbers_product_sum_digits_not_equal_l628_628962

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l628_628962


namespace prod_coprime_mod_l628_628179

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628179


namespace final_price_for_staff_l628_628304

variable (d : ℝ)

-- Conditions
def initial_discount : ℝ := 0.25 * d
def price_after_initial_discount : ℝ := d - initial_discount
def staff_discount : ℝ := 0.20 * price_after_initial_discount

-- Final question: What is the final price a staff member has to pay in terms of d?
theorem final_price_for_staff : price_after_initial_discount - staff_discount = 0.60 * d := by
  calc
    price_after_initial_discount - staff_discount
    = (d - initial_discount) - (0.20 * (d - initial_discount)) : by rfl
    ... = (d - 0.25 * d) - (0.20 * (d - 0.25 * d)) : by sorry
    ... = (0.75 * d) - (0.20 * 0.75 * d) : by sorry
    ... = (0.75 * d) - (0.15 * d) : by sorry
    ... = 0.60 * d : by sorry

end final_price_for_staff_l628_628304


namespace photograph_perimeter_l628_628729

-- Definitions of the conditions
def photograph_is_rectangular : Prop := True
def one_inch_border_area (w l m : ℕ) : Prop := (w + 2) * (l + 2) = m
def three_inch_border_area (w l m : ℕ) : Prop := (w + 6) * (l + 6) = m + 52

-- Lean statement of the problem
theorem photograph_perimeter (w l m : ℕ) 
  (h1 : photograph_is_rectangular)
  (h2 : one_inch_border_area w l m)
  (h3 : three_inch_border_area w l m) : 
  2 * (w + l) = 10 := 
by 
  sorry

end photograph_perimeter_l628_628729


namespace least_subtracted_divisible_by_5_l628_628702

theorem least_subtracted_divisible_by_5 :
  ∃ n : ℕ, (568219 - n) % 5 = 0 ∧ n ≤ 4 ∧ (∀ m : ℕ, m < 4 → (568219 - m) % 5 ≠ 0) :=
sorry

end least_subtracted_divisible_by_5_l628_628702


namespace kay_exercise_time_l628_628940

variable (A W : ℕ)
variable (exercise_total : A + W = 250) 
variable (ratio_condition : A * 2 = 3 * W)

theorem kay_exercise_time :
  A = 150 ∧ W = 100 :=
by
  sorry

end kay_exercise_time_l628_628940


namespace problem1_problem2_l628_628861

-- Definition of f and g for first proof problem
def f1 (x : ℝ) := Real.cos x + Real.sin x
def g1 (x : ℝ) := f1 x * f1 (x + Real.pi / 2)

-- Statement of the first proof problem
theorem problem1 (x : ℝ) : g1 x = Real.cos (2 * x) :=
by
  -- Proof is omitted.
  sorry

-- Definitions for the second proof problem
def f2 (x : ℝ) := abs (Real.sin x) + Real.cos x
def g2 (x : ℝ) := f2 x * f2 (x + Real.pi / 2)

-- Statement of the second proof problem
theorem problem2 : ∀ x1 x2 : ℝ, (∀ x : ℝ, g2 x1 ≤ g2 x ∧ g2 x ≤ g2 x2) →
   |x1 - x2| ≥ Real.pi * 3 / 4 :=
by
  -- Proof is omitted.
  sorry

end problem1_problem2_l628_628861


namespace crayons_given_correct_l628_628977

def crayons_lost : ℕ := 161
def additional_crayons : ℕ := 410
def crayons_given (lost : ℕ) (additional : ℕ) : ℕ := lost + additional

theorem crayons_given_correct : crayons_given crayons_lost additional_crayons = 571 :=
by
  sorry

end crayons_given_correct_l628_628977


namespace square_of_cube_of_third_smallest_prime_l628_628670

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628670


namespace correct_statement_l628_628295

-- Definitions
def algebraic_expression := ∀ {x : α}, x ∈ algebraic_expression α
def monomial := ∀ {x : α}, monomial x
def degree (m : α) : ℕ := degree m

-- Statements
def statement_A := ∀ e, algebraic_expression e → monomial e
def statement_B := ∀ m, monomial m → algebraic_expression m
def statement_C := degree (1 : ℝ) = 0
def statement_D := degree ((-π^2 : ℝ) * (2 : ℝ) * (2 : ℝ)) = 6

-- Proof Problem
theorem correct_statement : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D := by
  sorry

end correct_statement_l628_628295


namespace solid_is_triangular_prism_l628_628888

-- Given conditions as definitions
def front_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the front view is an isosceles triangle
  sorry

def left_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the left view is an isosceles triangle
  sorry

def top_view_is_circle (solid : Type) : Prop := 
   -- Define the property that the top view is a circle
  sorry

-- Define the property of being a triangular prism
def is_triangular_prism (solid : Type) : Prop :=
  -- Define the property that the solid is a triangular prism
  sorry

-- The main theorem: proving that given the conditions, the solid could be a triangular prism
theorem solid_is_triangular_prism (solid : Type) :
  front_view_is_isosceles_triangle solid ∧ 
  left_view_is_isosceles_triangle solid ∧ 
  top_view_is_circle solid →
  is_triangular_prism solid :=
sorry

end solid_is_triangular_prism_l628_628888


namespace line_properties_l628_628470

theorem line_properties (a : ℝ) :
  -- Condition of the problem
  let l := (a^2 + a + 1) * x - y + 1 = 0,

  -- Correct Answer (A): Line l is perpendicular to x + y = 0 when a = -1
  (a = -1 → (let k1 := (a^2 + a + 1)
             let k2 := -1
             (k1 * k2 = -1))) ∧
  
  -- Correct Answer (C): Line l passes through the point (0, 1) for any a
  (l 0 1 = 0) :=
  
by sorry

end line_properties_l628_628470


namespace arithmetic_sequence_geo_ratio_l628_628454

theorem arithmetic_sequence_geo_ratio
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_seq : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_geo : (S 2) ^ 2 = S 1 * S 4) :
  (a_n 2 + a_n 3) / a_n 1 = 8 :=
by sorry

end arithmetic_sequence_geo_ratio_l628_628454


namespace area_ratio_parallelogram_to_triangle_l628_628647

variables {A B C D R E : Type*}
variables (s_AB s_AD : ℝ)

-- Given AR = 2/3 AB and AE = 1/3 AD
axiom AR_proportion : s_AB > 0 → s_AB * (2/3) = s_AB
axiom AE_proportion : s_AD > 0 → s_AD * (1/3) = s_AD

-- Given the relationship, we need to prove
theorem area_ratio_parallelogram_to_triangle (hAB : s_AB > 0) (hAD : s_AD > 0) :
  ∃ (S_ABCD S_ARE : ℝ), S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_parallelogram_to_triangle_l628_628647


namespace problem_triangle_dot_product_l628_628077

noncomputable def vec (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

theorem problem_triangle_dot_product :
  ∀ (A B C M : ℝ × ℝ),
    B = (0, 0) →
    A = (0, 1) →
    C = (1, 0) →
    M.1 / 2 = M.1 / 2 - (A.1 - M.1) / 2 →
    vec B C = (1, 0) →
    vec A B = (0, 1) →
    vec C M = (1, -M.1) →
    vec C A = (1, 1 - M.1) →
    (vec C M).1 * (vec C A).1 + (vec C M).2 * (vec C A).2 = 3 :=
begin
  -- We skip the proof for now
  sorry
end

end problem_triangle_dot_product_l628_628077


namespace fraction_of_red_knights_magical_l628_628080

theorem fraction_of_red_knights_magical
  (total_knights : ℕ)
  (fraction_red : ℚ)
  (fraction_magical : ℚ)
  (fraction_magical_red_relative : ℚ)
  (h_fraction_red : fraction_red = 3 / 8)
  (h_fraction_magical : fraction_magical = 1 / 5)
  (h_fraction_magical_red_relative : fraction_magical_red_relative = 3) :
  let red_knights := total_knights * fraction_red
      blue_knights := total_knights - red_knights
      magical_knights := total_knights * fraction_magical
      fraction_magical_red := magical_knights / (red_knights + blue_knights) * fraction_magical_red_relative in
  fraction_magical_red = 12 / 35 := sorry

end fraction_of_red_knights_magical_l628_628080


namespace function_properties_l628_628388

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem function_properties :
  (∃ x : ℝ, f x = -1) = false ∧ 
  (∃ x_0 : ℝ, -1 < x_0 ∧ x_0 < 0 ∧ deriv f x_0 = 0) ∧ 
  (∀ x : ℝ, -3 < x → f x > -1 / 2) ∧ 
  (∃ x_0 : ℝ, -3 < x_0 ∧ ∀ x : ℝ, -3 < x → f x_0 ≤ f x) :=
by
  sorry

end function_properties_l628_628388


namespace polygon_diagonals_l628_628045

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l628_628045


namespace part_a_part_b_exists_x_part_b_continuity_l628_628240

-- Defining the concentration function
noncomputable def concentration (P : MeasureTheory.ProbabilityMeasure ℝ) (X : ℝ → ℝ) (l : ℝ) : ℝ :=
  ⨆ x : ℝ, P { ω | x < X ω ∧ X ω ≤ x + l }

-- Assumptions that X and Y are independent random variables
variables {P : MeasureTheory.ProbabilityMeasure ℝ} {X Y : ℝ → ℝ}
variable (independent : MeasureTheory.Indep (P.comp (MeasureTheory.map (λ p, (X p, Y p)) P)) ⊤)

theorem part_a (l : ℝ) (hl : 0 ≤ l) :
  concentration P (λ ω, X ω + Y ω) l ≤ min (concentration P X l) (concentration P Y l) := by
  sorry

theorem part_b_exists_x (l : ℝ) (hl : 0 ≤ l) :
  ∃ (x_l_star : ℝ), concentration P X l = P { ω | x_l_star < X ω ∧  X ω ≤ x_l_star + l } := by
  sorry

theorem part_b_continuity :
  (∀ x, continuous_at (λω, X ω) x) ↔ concentration P X 0 = 0 := by
  sorry

end part_a_part_b_exists_x_part_b_continuity_l628_628240


namespace total_distance_kolya_travelled_l628_628941

theorem total_distance_kolya_travelled :
  ∃ (t t1 t2 : ℝ), 
    t + t1 + t2 = 5 ∧ 
    4 * t + 3 * t1 + 6 * t2 = 20 :=
begin
  sorry
end

end total_distance_kolya_travelled_l628_628941


namespace eugene_total_payment_l628_628910

-- Define the initial costs of items
def cost_tshirt := 20
def cost_pants := 80
def cost_shoes := 150

-- Define the quantities
def quantity_tshirt := 4
def quantity_pants := 3
def quantity_shoes := 2

-- Define the discount rate
def discount_rate := 0.10

-- Define the total pre-discount cost
def pre_discount_cost :=
  (cost_tshirt * quantity_tshirt) +
  (cost_pants * quantity_pants) +
  (cost_shoes * quantity_shoes)

-- Define the discount amount
def discount_amount := discount_rate * pre_discount_cost

-- Define the post-discount cost
def post_discount_cost := pre_discount_cost - discount_amount

-- Theorem statement
theorem eugene_total_payment : post_discount_cost = 558 := by
  sorry

end eugene_total_payment_l628_628910


namespace smallest_n_for_divisibility_l628_628958

noncomputable def geometric_sequence (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem smallest_n_for_divisibility (h₁ : ∀ n : ℕ, geometric_sequence (1/2 : ℚ) 60 n = (1/2 : ℚ) * 60^(n-1))
    (h₂ : (60 : ℚ) * (1 / 2) = 30)
    (n : ℕ) :
  (∃ n : ℕ, n ≥ 1 ∧ (geometric_sequence (1/2 : ℚ) 60 n) ≥ 10^6) ↔ n = 7 :=
by
  sorry

end smallest_n_for_divisibility_l628_628958


namespace part_a_not_necessarily_rational_part_b_sum_of_squares_rat_l628_628654

variable (a b : ℝ)
variable (h_a_pos : 0 < a)
variable (h_b_pos : 0 < b)
variable (h_sum_rat : (a + b) ∈ ℚ)
variable (h_cubes_rat : (a^3 + b^3) ∈ ℚ)

-- Part (a): Proving that \( a \) and \( b \) are not necessarily rational numbers
theorem part_a_not_necessarily_rational : ¬(a ∈ ℚ ∧ b ∈ ℚ) :=
  sorry

-- Part (b): Proving the sum of their squares is a rational number
theorem part_b_sum_of_squares_rat : (a^2 + b^2) ∈ ℚ :=
  by {
    have h_ab_rat : (a * b) ∈ ℚ,
    { -- Proof that a * b is rational
      sorry },
    show (a^2 + b^2) ∈ ℚ,
    { -- Proof that sum of squares is rational
      calc
        a^2 + b^2 = (a + b)^2 - 2 * (a * b) : by nlinarith
                ... ∈ ℚ                    : by { apply add_subgroup_closure; assumption }
    }
  }

end part_a_not_necessarily_rational_part_b_sum_of_squares_rat_l628_628654


namespace non_congruent_triangles_with_perimeter_18_l628_628000

theorem non_congruent_triangles_with_perimeter_18 : ∃ (S : set (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 18 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  ∧ (S.card = 8) := sorry

end non_congruent_triangles_with_perimeter_18_l628_628000


namespace product_of_invertible_integers_mod_120_l628_628138

theorem product_of_invertible_integers_mod_120:
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x 120) (Finset.range 120) id
  in m % 120 = 1 := by 
  sorry

end product_of_invertible_integers_mod_120_l628_628138


namespace product_of_invertibles_mod_120_l628_628130

open Nat

theorem product_of_invertibles_mod_120 :
  let n := 5! in
  let invertibles := { i | i < n ∧ gcd i n = 1 } in
  let m := ∏ i in invertibles, i in
  m % n = 1 := by
  let n := 5!
  let invertibles := { i | i < n ∧ gcd i n = 1 }
  let m := ∏ i in invertibles, i
  show m % n = 1
  sorry

end product_of_invertibles_mod_120_l628_628130


namespace total_pears_l628_628747

theorem total_pears (Alyssa_picked Nancy_picked : ℕ) (h₁ : Alyssa_picked = 42) (h₂ : Nancy_picked = 17) : Alyssa_picked + Nancy_picked = 59 :=
by
  sorry

end total_pears_l628_628747


namespace part_I_part_II_l628_628201

def f (x : ℝ) : ℝ := abs (2 * x - 7) + 1

def g (x : ℝ) : ℝ := abs (2 * x - 7) - 2 * abs (x - 1) + 1

theorem part_I :
  {x : ℝ | f x ≤ x} = {x : ℝ | (8 / 3) ≤ x ∧ x ≤ 6} := sorry

theorem part_II (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 := sorry

end part_I_part_II_l628_628201


namespace inequality_xy_gt_xz_l628_628839

theorem inequality_xy_gt_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : 
  x * y > x * z := 
by
  sorry  -- Proof is not required as per the instructions

end inequality_xy_gt_xz_l628_628839


namespace hypotenuse_is_correct_l628_628282

noncomputable def hypotenuse_of_right_triangle (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_is_correct :
  hypotenuse_of_right_triangle 140 210 = 70 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_is_correct_l628_628282


namespace number_of_digits_l628_628871

-- Lean 4 statement for the problem
theorem number_of_digits (n : ℕ) : 
  (∑ k in finset.range n, (k + 1) * 9 * 10^k) = (1 / 9 : ℚ) * ((9 * n - 1) * 10^n + 1) := 
sorry

end number_of_digits_l628_628871


namespace good_numbers_product_sum_digits_not_equal_l628_628961

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l628_628961


namespace product_of_invertibles_mod_120_l628_628153

theorem product_of_invertibles_mod_120 :
  let n := 120
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k n) (Finset.range n)), k
  m % n = 1 :=
by
  sorry

end product_of_invertibles_mod_120_l628_628153


namespace fixed_deposit_interest_rate_l628_628600

theorem fixed_deposit_interest_rate :
  ∃ r : ℝ, r = 7.75 ∧
  (∃ loanAmount : ℝ, loanAmount = 15000) ∧
  (∃ depositAmount : ℝ, depositAmount = 10000) ∧
  (∃ dailyInterest : ℝ, dailyInterest * 7 = 180.83) ∧
  (∃ monthlyInterest : ℝ, monthlyInterest = dailyInterest * 30) ∧
  (monthlyInterest / depositAmount * 100 = r)
  :=
begin
  sorry
end

end fixed_deposit_interest_rate_l628_628600


namespace angle_bisector_passes_midpoint_l628_628930

/-- Given the incircle of triangle ABC touching AC at D, BD intersects the incircle at E.
    Points F and G on the incircle are such that FE is parallel to BC and GE is parallel to AB.
    I1 and I2 are the incenters of triangles DEF and DEG, respectively.
    Prove that the angle bisector of ∠GDF passes through the midpoint of the line segment I1I2. -/
theorem angle_bisector_passes_midpoint
  (A B C D E F G I1 I2 : Point)
  (h_incircle_AC : touches_incircle AC D)
  (h_BD_intersects_incircle : intersects BD E)
  (h_FE_parallel_BC : parallel FE BC)
  (h_GE_parallel_AB : parallel GE AB)
  (h_I1_incenter_DEF : incenter I1 DEF)
  (h_I2_incenter_DEG : incenter I2 DEG) :
  passes_through_midpoint (angle_bisector G D F) (midpoint I1 I2) :=
sorry

end angle_bisector_passes_midpoint_l628_628930


namespace polygon_sides_from_diagonals_l628_628030

theorem polygon_sides_from_diagonals (n : ℕ) (h : 20 = n * (n - 3) / 2) : n = 8 :=
sorry

end polygon_sides_from_diagonals_l628_628030


namespace find_other_root_l628_628840

theorem find_other_root (m : ℝ) : is_root (λ x : ℝ, x^2 + m * x - 4) (-1) → -1 * 4 = -4 → is_root (λ x : ℝ, x^2 + m * x - 4) 4 :=
  by
    intro h_root
    dsimp [is_root] at h_root
    rw [polynomial.eval, polynomial.eval₂] at h_root
    sorry

end find_other_root_l628_628840


namespace balls_in_boxes_l628_628017

theorem balls_in_boxes (n : ℕ) (h : n = 6) : 
  (number_of_ways_to_distribute_in_boxes 6 2) = 32 :=
sorry

def number_of_ways_to_distribute_in_boxes (ball_count box_count : ℕ) : ℕ :=
if box_count = 2 then
  let ways_case_1 := nat.choose ball_count 6 in
  let ways_case_2 := nat.choose ball_count 5 in
  let ways_case_3 := nat.choose ball_count 4 in
  let ways_case_4 := (nat.choose ball_count 3) / 2 in
  ways_case_1 + ways_case_2 + ways_case_3 + ways_case_4
else 0 -- This simplistic placeholder assumes box_count other than 2 are not interesting

end balls_in_boxes_l628_628017


namespace square_of_cube_of_third_smallest_prime_l628_628682

theorem square_of_cube_of_third_smallest_prime :
  let p := 5 in ((p ^ 3) ^ 2) = 15625 :=
by
  let p := 5
  sorry

end square_of_cube_of_third_smallest_prime_l628_628682


namespace perpendicular_when_a_is_neg1_passes_through_origin_l628_628473

noncomputable theory

def line_l (a : ℝ) : (ℝ × ℝ) → Prop := λ p, (a^2 + a + 1) * p.1 - p.2 + 1 = 0
def line_x_plus_y : (ℝ × ℝ) → Prop := λ p, p.1 + p.2 = 0

def passes_through (l : (ℝ × ℝ) → Prop) (p : ℝ × ℝ) : Prop := l p

theorem perpendicular_when_a_is_neg1 
  (a : ℝ) (p : ℝ × ℝ) : 
  a = -1 → (∀ p, line_l a p → line_x_plus_y p → false) :=
by sorry

theorem passes_through_origin
  (a : ℝ) :
  passes_through (line_l a) (0, 1) :=
by sorry

end perpendicular_when_a_is_neg1_passes_through_origin_l628_628473


namespace tests_to_confirm_infected_l628_628875

theorem tests_to_confirm_infected:
  ∀ (n : ℕ), n = 16 → (∃ infected : ℕ, infected = 1) → 
  ∀ (method : ℕ → ℕ), (method = (λ n, if n > 1 then method (n / 2 + n % 2) else 1)) →
  method n = 4 := 
by
  intros n h_n infected h_infected method h_method
  -- Pseudo proof that sets up the scenario and asserts the result.
  sorry

end tests_to_confirm_infected_l628_628875


namespace quadratic_no_real_roots_l628_628697

theorem quadratic_no_real_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 5) : 
  b^2 - 4 * a * c < 0 :=
by
  rw [h1, h2, h3]
  calc
    (3 : ℝ)^2 - 4 * 1 * 5 = 9 - 20 := by norm_num
    ... = -11 := by norm_num
    ... < 0 := by norm_num

end quadratic_no_real_roots_l628_628697


namespace max_PM_plus_PN_l628_628478

noncomputable def curve1_cartesian_eq (x y : ℝ) : Prop :=
    x^2 / 4 + y^2 / 3 = 1

noncomputable def curve2_cartesian_eq (x y : ℝ) : Prop :=
    x^2 + y^2 = 4

noncomputable def distance (P M : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)

theorem max_PM_plus_PN (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (h1 : curve1_cartesian_eq M.1 M.2)
  (h2 : curve1_cartesian_eq N.1 N.2)
  (h3 : curve2_cartesian_eq P.1 P.2)
  (hM : M = (2, real.sqrt 3))
  (hN : N = (2, -real.sqrt 3)) :
  ∃ α : ℝ, distance P M + distance P N = 2 * real.sqrt 7 :=
begin
  sorry
end

end max_PM_plus_PN_l628_628478


namespace john_spent_on_candy_l628_628360

theorem john_spent_on_candy (M : ℝ) (C : ℝ) (h1 : M = 120) 
  (h2 : ∑ i in {1/2, 1/3, 1/10}, i = 1/2 + 1/3 + 1/10)
  (h3 : 1 - (1/2 + 1/3 + 1/10) = 1/15) 
  : C = M * (1/15) → C = 8 :=
by
  sorry

end john_spent_on_candy_l628_628360


namespace nth_equation_l628_628864

theorem nth_equation (n : ℕ) (hn : 0 < n) : 
  (∑ i in (range n).map (λ k, 1 / (2 * k + 1 : ℕ)) - ∑ i in (range n).map (λ k, 1 / (2 * (k + 1) : ℕ))) = 
  (∑ i in (range n).map (λ k, 1 / (n + k + 1 : ℕ))) :=
sorry

end nth_equation_l628_628864


namespace power_sum_l628_628494

theorem power_sum (a b c : ℝ) (h1 : a + b + c = 1)
                  (h2 : a^2 + b^2 + c^2 = 3)
                  (h3 : a^3 + b^3 + c^3 = 4)
                  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 :=
  sorry

end power_sum_l628_628494


namespace largest_lambda_inequality_l628_628791

theorem largest_lambda_inequality :
  (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + b^2 + c^2 + d^2 ≥ ab + (3 / 2) * bc + cd) ∧
  (∀ λ : ℝ, (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + b^2 + c^2 + d^2 ≥ ab + λ * bc + cd) → λ ≤ 3 / 2) :=
by
  sorry

end largest_lambda_inequality_l628_628791


namespace part_a_part_b_l628_628111

-- Step d: Lean statements for the proof problems
theorem part_a (p : ℕ) (hp : Nat.Prime p) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 :=
by {
  sorry
}

theorem part_b (p : ℕ) (hp : Nat.Prime p) : (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 ∧ a % p ≠ 0 ∧ b % p ≠ 0) ↔ p ≠ 3 :=
by {
  sorry
}

end part_a_part_b_l628_628111


namespace arithmetic_sequence_common_difference_l628_628825

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 13) 
  (h2 : (5 * (a 1 + a 5)) / 2 = 35) 
  (h_arithmetic_sequence : ∀ n, a (n+1) = a n + d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l628_628825


namespace point_outside_circle_l628_628928

theorem point_outside_circle :
  let center := (-2, -3 : ℝ × ℝ)
  let radius : ℝ := 6
  let point_inside := (-2, 2 : ℝ × ℝ)
  let point_outside := (-2, -10 : ℝ × ℝ)
  dist center point_outside > radius :=
by
  sorry

end point_outside_circle_l628_628928


namespace prime_p_geq_7_div_240_l628_628216

theorem prime_p_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (hge7 : p ≥ 7) : 240 ∣ p^4 - 1 := 
sorry

end prime_p_geq_7_div_240_l628_628216


namespace log2_inequality_l628_628612

theorem log2_inequality (x : ℝ) : (x < 1) → (0 < x → (log x / log 2 < 0) ∧ ¬ (x < 1 → log x / log 2 < 0)) :=
by
  sorry

end log2_inequality_l628_628612


namespace cost_of_tree_planting_l628_628982

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l628_628982


namespace probability_avg_gt_five_l628_628072

theorem probability_avg_gt_five (S : set ℕ) (hS : S = {1, 3, 5, 7, 9}) : 
  (∃ (A : set ℕ), A ⊆ S ∧ A.card = 2 ∧ ((erase S A).1.sum / 3 > 5 → 
  (A.card.choose 2) / S.card.choose 2 = 2 / 5)) := 
sorry

end probability_avg_gt_five_l628_628072


namespace regular_polygon_num_sides_l628_628044

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628044


namespace probability_of_letter_in_mathematics_l628_628024

theorem probability_of_letter_in_mathematics :
  let distinct_letters_in_mathematics := 8
  let total_letters_in_alphabet := 26
  distinct_letters_in_mathematics.to_rat / total_letters_in_alphabet.to_rat = 4 / 13 :=
by
  sorry

end probability_of_letter_in_mathematics_l628_628024


namespace two_x_plus_14_congruent_5_l628_628497

theorem two_x_plus_14_congruent_5 
    (x : ℤ)
    (h : 3 * x + 8 ≡ 3 [MOD 17]) :
    2 * x + 14 ≡ 5 [MOD 17] :=
by 
    sorry

end two_x_plus_14_congruent_5_l628_628497


namespace square_of_cube_of_third_smallest_prime_l628_628686

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end square_of_cube_of_third_smallest_prime_l628_628686


namespace non_congruent_triangles_perimeter_18_l628_628011

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end non_congruent_triangles_perimeter_18_l628_628011


namespace parabola_intersects_y_axis_at_0_4_l628_628847

theorem parabola_intersects_y_axis_at_0_4 : 
  (∀ k, (0 = -(4:ℕ)^2 + (k + 1) * 4 - k) → (y : ℝ) = -x^2 + (k+1) * x - k ∧ x = 0 → y = (0, -4)) :=
by
  sorry

end parabola_intersects_y_axis_at_0_4_l628_628847


namespace sides_of_regular_polygon_with_20_diagonals_l628_628038

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l628_628038


namespace find_b_l628_628274

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
by {
  -- Proof will be filled in here
  sorry
}

end find_b_l628_628274


namespace slope_angle_of_line_l628_628849

theorem slope_angle_of_line (alpha : ℝ) :
  (∃ t : ℝ, (1 + t * cos alpha - 2)^2 + (t * sin alpha)^2 = 4) →
  ((∀ t1 t2 : ℝ, (t1 + t2 = 2 * cos alpha ∧ t1 * t2 = -3) → 
    (abs (t1 - t2) = sqrt 14)) →
  (alpha = π / 4 ∨ alpha = 3 * π / 4)) :=
by
  sorry

end slope_angle_of_line_l628_628849


namespace range_of_p_l628_628960

noncomputable def a_n (n : ℕ) : ℝ := 4 + (-1 / 2)^(n - 1)

noncomputable def S_n (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, a_n (k + 1))

theorem range_of_p (p : ℝ) (h : ∀ n : ℕ, 0 < n → 1 ≤ p * (S_n n - 4 * n) ∧ p * (S_n n - 4 * n) ≤ 3) :
  2 ≤ p ∧ p ≤ 3 :=
sorry

end range_of_p_l628_628960


namespace jason_total_amount_l628_628549

def shorts_price : ℝ := 14.28
def jacket_price : ℝ := 4.74
def shoes_price : ℝ := 25.95
def socks_price : ℝ := 6.80
def tshirts_price : ℝ := 18.36
def hat_price : ℝ := 12.50
def swimsuit_price : ℝ := 22.95
def sunglasses_price : ℝ := 45.60
def wristbands_price : ℝ := 9.80

def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount)

def total_discounted_price : ℝ := 
  (discounted_price shorts_price discount1) + 
  (discounted_price jacket_price discount1) + 
  (discounted_price hat_price discount1) + 
  (discounted_price shoes_price discount2) + 
  (discounted_price socks_price discount2) + 
  (discounted_price tshirts_price discount2) + 
  (discounted_price swimsuit_price discount2) + 
  (discounted_price sunglasses_price discount2) + 
  (discounted_price wristbands_price discount2)

def total_with_tax : ℝ := total_discounted_price + (total_discounted_price * sales_tax_rate)

theorem jason_total_amount : total_with_tax = 153.07 := by
  sorry

end jason_total_amount_l628_628549


namespace max_distance_polar_points_l628_628540

/-- In the polar coordinate system, P is a moving point on the curve 
    ρ = 12 * sin(θ), and Q is a moving point on the curve 
    ρ = 12 * cos(θ - π / 6). 

    Prove that the maximum value of the distance PQ is 18. 
--/
theorem max_distance_polar_points : 
  ∀ (θ : ℝ), let P := (12 * sin θ, 12 * sin θ) in
             let Q := (12 * (cos θ * (cos (π / 6)) + sin θ * (sin (π / 6))), 
                       12 * (cos θ * (cos (π / 6)) + sin θ * (sin (π / 6)))) in
             dist P Q ≤ 18 :=
begin
  sorry,
end

end max_distance_polar_points_l628_628540


namespace crickets_over_15_weeks_l628_628431

def totalCricketsEaten (weeks : Nat) (crickets_per_week_90F : Nat) (crickets_per_week_100F : Nat) (p_90F : Rat) : Nat :=
  let weeks_90F := (p_90F * weeks).toNat
  let weeks_100F := weeks - weeks_90F
  let crickets_90F := weeks_90F * crickets_per_week_90F
  let crickets_100F := weeks_100F * crickets_per_week_100F
  crickets_90F + crickets_100F

theorem crickets_over_15_weeks :
  totalCricketsEaten 15 4 8 (4/5) = 72 :=
by
  sorry

end crickets_over_15_weeks_l628_628431


namespace wire_cut_pieces_l628_628014

theorem wire_cut_pieces (length_wire : ℝ) (length_piece : ℝ) (num_pieces : ℕ)
  (h1 : length_wire = 27.9) (h2 : length_piece = 3.1) :
  (length_wire / length_piece) = num_pieces → num_pieces = 9 :=
by
  intros
  have h3 : 27.9 / 3.1 = 9 := by norm_num
  rw [h1, h2] at h3
  exact h3

end wire_cut_pieces_l628_628014


namespace nearest_integer_sqrt5_sqrt3_pow4_l628_628409

theorem nearest_integer_sqrt5_sqrt3_pow4 : 
  let x := (Real.sqrt 5 + Real.sqrt 3) ^ 4 
  in Int.nearest x = 248 :=
by
  sorry

end nearest_integer_sqrt5_sqrt3_pow4_l628_628409


namespace geometric_sum_of_bn_l628_628946

theorem geometric_sum_of_bn (n : ℕ) (n ≥ 1) :
  let a : ℕ → ℕ := λ n, 2^n,
      b : ℕ → ℕ := λ n, (2 * n - 1) * a n,
      S : ℕ → ℕ := λ n, ∑ i in finset.range n, b (i + 1) in
  S n = 6 - (3 - 2 * n) * 2 ^ (n + 1) :=
by
  sorry

end geometric_sum_of_bn_l628_628946


namespace find_point_A_z_l628_628790

-- Define points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define distance function for 3D points
def dist (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define the points A, B, and C
def A (z : ℝ) : Point3D := { x := 0, y := 0, z := z }
def B : Point3D := { x := 6, y := -7, z := 1 }
def C : Point3D := { x := -1, y := 2, z := 5 }

-- The main theorem to prove
theorem find_point_A_z (z : ℝ) : 
  dist (A z) B = dist (A z) C → z = -7 := by
  sorry

end find_point_A_z_l628_628790


namespace prod_coprime_mod_l628_628180

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628180


namespace prod_coprime_mod_l628_628178

open Nat

def euler_phi (n : ℕ) : ℕ := (List.range n).filter (λ k, gcd n k = 1).length

theorem prod_coprime_mod (n : ℕ) (h : 1 < euler_phi n ∧ euler_phi n = 32) 
  (m : ℕ) : (∏ k in (List.range n).filter (λ k, gcd n k = 1), k) % n = 1 :=
by
  sorry

end prod_coprime_mod_l628_628178


namespace lara_harvest_raspberries_l628_628559

-- Define measurements of the garden
def length : ℕ := 10
def width : ℕ := 7

-- Define planting and harvesting constants
def plants_per_sq_ft : ℕ := 5
def raspberries_per_plant : ℕ := 12

-- Calculate expected number of raspberries
theorem lara_harvest_raspberries :  length * width * plants_per_sq_ft * raspberries_per_plant = 4200 := 
by sorry

end lara_harvest_raspberries_l628_628559


namespace find_angle_A_find_perimeter_range_l628_628519

-- Part 1
theorem find_angle_A {a b c A B C : ℝ} (h1 : a / b * Real.cos C + c / (2 * b) = 1) (h2 : A ≠ C) : A = Real.pi / 3 :=
by sorry

-- Part 2
theorem find_perimeter_range {b c B C : ℝ} (h: 1 / b * Real.cos C + c / (2 * b) = 1) : 
  let A := Real.pi / 3 in
  let a := 1 in
  1 + (2 / Real.sqrt 3) * (Real.sin B + Real.sin C) ∈ set.Icc (2 : ℝ) (3) :=
by sorry

end find_angle_A_find_perimeter_range_l628_628519


namespace option_d_correct_l628_628289

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l628_628289


namespace raspberry_carton_ounces_l628_628755

-- Definitions
def blueberry_cost_per_carton : ℝ := 5.00
def blueberry_ounces_per_carton : ℝ := 6.00
def raspberry_cost_per_carton : ℝ := 3.00
def num_batches : ℝ := 4.00
def ounces_per_batch : ℝ := 12.00
def savings_by_using_raspberries : ℝ := 22.00

-- Total ounces needed
def total_ounces_needed : ℝ := num_batches * ounces_per_batch

-- Cost calculations
def cost_of_blueberries : ℝ := (total_ounces_needed / blueberry_ounces_per_carton) * blueberry_cost_per_carton
def cost_of_raspberries : ℝ := cost_of_blueberries - savings_by_using_raspberries

-- Proof statement
theorem raspberry_carton_ounces :
  ∃ x : ℝ, (total_ounces_needed / x) * raspberry_cost_per_carton = cost_of_raspberries ∧ x = 8 := 
begin
  sorry
end

end raspberry_carton_ounces_l628_628755


namespace scheduling_plans_l628_628359

theorem scheduling_plans : 
  ∃ (A B C : list ℕ), 
  ∀ (schedule : list (list ℕ)), 
  (length schedule = 3) ∧ 
  (∀ person, length person = 2) ∧ 
  (∀ day, ¬ (day = 1 ∧ A.contains day)) ∧ 
  (∀ day, ¬ (day = 6 ∧ B.contains day)) ∧ 
  (schedule = [A, B, C] → 42) :=
sorry

end scheduling_plans_l628_628359


namespace find_tangent_points_l628_628408

noncomputable def f (x : ℝ) := x^3 + x - 2

def derivative_at (x : ℝ) := 3 * x^2 + 1

theorem find_tangent_points :
  { P : ℝ × ℝ // (P.1 = -1 ∧ P.2 = f P.1) ∨ (P.1 = 1 ∧ P.2 = f P.1) } ∧
  derivative_at (-1) = 4 ∧ derivative_at 1 = 4 :=
by
  have h1 := derivative_at (-1)
  have h2 := derivative_at 1
  have h3 : f (-1) = -4 := by simp [f]
  have h4 : f (1) = 0 := by simp [f]
  use [(-1, -4), (1, 0)]
  repeat { split <|> assumption <|> simp [derivative_at, f, (3 : ℝ)] }
  sorry

end find_tangent_points_l628_628408


namespace slope_positive_range_of_a_l628_628461

section Problem1

variables {x y : ℝ}
def f (x : ℝ) : ℝ := x + sin x

theorem slope_positive (x1 x2 : ℝ) (h : x1 < x2) :
  (f x2 - f x1) / (x2 - x1) > 0 :=
begin
  -- Proof goes here
  sorry
end

end Problem1

section Problem2

variables {x a : ℝ}
def g (x a : ℝ) : ℝ := a * x * cos x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) (pi / 2), f x ≥ g x a) ↔ a ≤ 2 :=
begin
  -- Proof goes here
  sorry
end

end Problem2

end slope_positive_range_of_a_l628_628461


namespace largest_three_digit_number_divisible_by_digits_and_difference_l628_628415

theorem largest_three_digit_number_divisible_by_digits_and_difference :
  ∃ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧
    (∀ d : ℕ, d ∈ (Int.natAbs <$> n.digits 10) ∧ d ≠ 0 → n % d = 0) ∧
    (let digits := Int.natAbs <$> n.digits 10 in
     (digits.nth 1).isSome ∧ (digits.nth 0).isSome ∧ 
     let diff := (digits.nth 1).iget - (digits.nth 0).iget in 
     n % (Int.natAbs diff) = 0) ∧
    n = 864 :=
sorry

end largest_three_digit_number_divisible_by_digits_and_difference_l628_628415


namespace find_pq_l628_628251

-- Given definitions based on conditions
def parabola (p x : ℝ) : ℝ := p * x - x^2
def hyperbola (x y q : ℝ) : Prop := x * y = q
def centroid_distance (p : ℝ) : ℝ := abs (p / 3)

theorem find_pq (p q x1 x2 x3 y1 y2 y3 : ℝ) 
    (h1 : ∀ x, parabola p x = y1 ∨ parabola p x = y2 ∨ parabola p x = y3) 
    (h2 : hyperbola x1 y1 q ∧ hyperbola x2 y2 q ∧ hyperbola x3 y3 q)
    (h3 : centroid_distance p = 2)
    (h4 : (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + (y3 - y1)^2 = 324): 
    p * q = 42 := 
sorry

end find_pq_l628_628251


namespace square_of_cube_of_third_smallest_prime_l628_628672

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628672


namespace Oliver_ferris_wheel_rides_l628_628362

theorem Oliver_ferris_wheel_rides :
  ∃ (F : ℕ), (4 * 7 + F * 7 = 63) ∧ (F = 5) :=
by
  sorry

end Oliver_ferris_wheel_rides_l628_628362


namespace product_of_invertibles_mod_120_l628_628170

theorem product_of_invertibles_mod_120 :
  let m := ∏ k in (Finset.filter (λ k, Nat.coprime k 120) (Finset.range 120)), k
  in m % 120 = 1 := by
  sorry

end product_of_invertibles_mod_120_l628_628170


namespace nancy_picked_correct_l628_628972

def good_carrots : ℕ := 71
def bad_carrots : ℕ := 14
def mother_picked : ℕ := 47
def total_carrots : ℕ := good_carrots + bad_carrots
def nancy_picked : ℕ := total_carrots - mother_picked

theorem nancy_picked_correct : nancy_picked = 38 := 
by 
  unfold total_carrots
  unfold nancy_picked
  sorry

end nancy_picked_correct_l628_628972


namespace regular_polygon_sides_l628_628051

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628051


namespace correct_calculation_l628_628693

-- Definition of the expressions in the problem
def exprA (a : ℝ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def exprB (x y : ℝ) : Prop := ((-3 * x^2 * y)^2 / (x * y) = 9 * x^5 * y^3)
def exprC (b : ℝ) : Prop := (2 * b^2)^3 = 8 * b^6
def exprD (x : ℝ) : Prop := (2 * x * 3 * x^5 = 6 * x^5)

-- The proof problem
theorem correct_calculation (a x y b : ℝ) : exprC b ∧ ¬ exprA a ∧ ¬ exprB x y ∧ ¬ exprD x :=
by {
  sorry
}

end correct_calculation_l628_628693


namespace infinitely_many_good_and_bad_terms_l628_628770

def sequence (n : ℕ) : ℕ := 2^n + 2^(n / 2)

def good_term (n : ℕ) : Prop := ∃ i j : ℕ, i ≠ j ∧ sequence n = sequence i + sequence j 

def bad_term (n : ℕ) : Prop := ¬ good_term n

theorem infinitely_many_good_and_bad_terms :
  (∀ m : ℕ, ∃ n : ℕ, n > m ∧ good_term n) ∧
  (∀ m : ℕ, ∃ n : ℕ, n > m ∧ bad_term n) :=
by
  sorry

end infinitely_many_good_and_bad_terms_l628_628770


namespace four_digit_special_count_l628_628384

theorem four_digit_special_count :
  let S := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ (∃ a b c d, n = a * 1000 + b * 100 + c * 10 + d ∧ (list.nodup [a, b, c, d]) ∧ 0 < a ∧ (d = 0 ∨ d = 5) ∧ list.max [a, b, c, d] = 7)} in
  S.card = 212 :=
by {
  sorry
}

end four_digit_special_count_l628_628384


namespace pascal_triangle_fifth_num_and_factorial_l628_628542

noncomputable def fifth_num_pascal_factorial : ℕ :=
let fifth_num := Nat.choose 15 4 in
let factorial := Nat.factorial fifth_num in
factorial

theorem pascal_triangle_fifth_num_and_factorial :
  Nat.choose 15 4 = 1365 ∧ Nat.factorial (Nat.choose 15 4) = 1365! :=
by
  apply And.intro
  · show Nat.choose 15 4 = 1365
    sorry
  · show Nat.factorial (Nat.choose 15 4) = 1365!
    sorry

end pascal_triangle_fifth_num_and_factorial_l628_628542


namespace ratio_A_B_l628_628629

-- Given conditions as definitions
def P_both : ℕ := 500  -- Number of people who purchased both books A and B

def P_only_B : ℕ := P_both / 2  -- Number of people who purchased only book B

def P_only_A : ℕ := 1000  -- Number of people who purchased only book A

-- Total number of people who purchased books
def P_A : ℕ := P_only_A + P_both  -- Total number of people who purchased book A

def P_B : ℕ := P_only_B + P_both  -- Total number of people who purchased book B

-- The ratio of people who purchased book A to book B
theorem ratio_A_B : P_A / P_B = 2 :=
by
  sorry

end ratio_A_B_l628_628629


namespace expand_expression_l628_628786

theorem expand_expression (x : ℝ) :
  (2 * x + 3) * (4 * x - 5) = 8 * x^2 + 2 * x - 15 :=
by
  sorry

end expand_expression_l628_628786


namespace pq_range_l628_628090

noncomputable def circle (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2

noncomputable def is_on_x_axis (A: ℝ) : Prop := True

noncomputable def is_tangent (A x y : ℝ) : Prop := 
  ∃ P Q : ℝ × ℝ, x^2 + (P.snd - 3)^2 = 2 
                   ∧ y^2 + (Q.snd - 3)^2 = 2

theorem pq_range 
  (x y : ℝ)
  (hx: circle x y)
  (A: ℝ)
  (hA: is_on_x_axis A)
  (hT: is_tangent A x y):
  ∃ l1 l2 : ℝ, l1 = 2 * Real.sqrt 14 / 3 ∧ l2 = 2 * Real.sqrt 2 ∧ (P.snd - Q.snd)^2 + (P.fst - Q.fst)^2 = l1 → l1 ∈ Ico 2 * Real.sqrt 14 / 3, 2 * Real.sqrt 2) :=
sorry 

end pq_range_l628_628090


namespace find_roots_l628_628423

theorem find_roots (x : ℝ) :
  (2^(2 * x) - 5 * 3^x + 29 = 0) → 
  (x = 2 ∨ (x ≈ 5.55)) := sorry

end find_roots_l628_628423


namespace kara_uses_28_cups_of_sugar_l628_628556

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l628_628556


namespace solid_is_cone_l628_628067

-- Definitions for the conditions
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

def is_isosceles_triangle (shape : Type) : Prop := sorry
def is_circle (shape : Type) : Prop := sorry

-- Define the solid based on the given conditions
noncomputable def my_solid : Solid := {
  front_view := sorry,
  side_view := sorry,
  top_view := sorry
}

-- Prove that the solid is a cone given the provided conditions
theorem solid_is_cone (s : Solid) : 
  is_isosceles_triangle s.front_view → 
  is_isosceles_triangle s.side_view → 
  is_circle s.top_view → 
  s = my_solid :=
by
  sorry

end solid_is_cone_l628_628067


namespace area_of_EFGH_l628_628593

-- Defining constants and variables
variables (E F G H : Type) [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace H]
variables (EFG EHG : Triangle) (EF FH HG GH EG : ℝ)

-- Conditions
def right_angles_at_F_and_H := right_angle FH ∧ right_angle HG
def EG_length := EG = 4
def distinct_integer_lengths := ∃ a b : ℕ, (a ≠ b) ∧ (EF = a ∨ FH = a ∨ HG = a ∨ GH = a) ∧ (EF = b ∨ FH = b ∨ HG = b ∨ GH = b)

-- Hypotheses based on the conditions
hypothesis (h1 : right_angles_at_F_and_H)
hypothesis (h2 : EG_length)
hypothesis (h3 : distinct_integer_lengths)

-- Goal
noncomputable def area_of_quadrilateral_EFGH : ℝ := 4 * sqrt 3
theorem area_of_EFGH : (EFG.area + EHG.area = area_of_quadrilateral_EFGH) := sorry

end area_of_EFGH_l628_628593


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l628_628963

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l628_628963


namespace monotonicity_decreasing_interval_l628_628275

noncomputable def function_x_pow_1_over_x (x : ℝ) : ℝ := 
  x^(1/x)

theorem monotonicity_decreasing_interval :
  ∀ (x : ℝ), (0 < x) → 
  (∃ a b : ℝ, (e < a ∧ b = +∞ ∧ (a < x → (function_x_pow_1_over_x x) < (function_x_pow_1_over_x (e)) ∧ ∀ (x : ℝ), (e < x) → (e < x) ∧ function_x_pow_1_over_x (x) < 0))) :=
by
  sorry

end monotonicity_decreasing_interval_l628_628275


namespace james_music_listening_hours_l628_628104

theorem james_music_listening_hours (BPM : ℕ) (beats_per_week : ℕ) (hours_per_day : ℕ) 
  (h1 : BPM = 200) 
  (h2 : beats_per_week = 168000) 
  (h3 : hours_per_day * 200 * 60 * 7 = beats_per_week) : 
  hours_per_day = 2 := 
by
  sorry

end james_music_listening_hours_l628_628104


namespace sum_valid_three_digit_numbers_l628_628797

def digits_without_0_or_5 : Set ℕ := {1, 2, 3, 4, 6, 7, 8, 9}

def is_valid_digit (d : ℕ) : Prop := d ∈ digits_without_0_or_5

def is_valid_three_digit (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧ is_valid_digit hundreds ∧ is_valid_digit tens ∧ is_valid_digit units

theorem sum_valid_three_digit_numbers :
  (∑ n in Finset.filter is_valid_three_digit (Finset.range 1000), n) = 284160 :=
  by sorry

end sum_valid_three_digit_numbers_l628_628797


namespace sasha_tree_planting_cost_l628_628984

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l628_628984


namespace log3_10_in_terms_of_a_and_b_l628_628433

noncomputable def log3_4_eq_a (a : ℝ) : Prop := log 3 4 = a
noncomputable def log3_5_eq_b (b : ℝ) : Prop := log 3 5 = b
noncomputable def log3_10_eq_half_a_plus_b (a b : ℝ) : Prop := log 3 10 = (1 / 2) * a + b

theorem log3_10_in_terms_of_a_and_b (a b : ℝ)
  (h1 : log3_4_eq_a a)
  (h2 : log3_5_eq_b b) :
  log3_10_eq_half_a_plus_b a b := by
  sorry

end log3_10_in_terms_of_a_and_b_l628_628433


namespace ratio_tuesday_to_monday_l628_628580

variable (time_monday : ℚ) (time_wednesday : ℚ) (time_thursday : ℚ) (time_friday : ℚ) (total_time : ℚ)
variable (time_tuesday : ℚ)

-- Given conditions in the problem:
def condition_monday := time_monday = 3 / 4
def condition_wednesday := time_wednesday = 2 / 3
def condition_thursday := time_thursday = 5 / 6
def condition_friday := time_friday = 75 / 60
def condition_total := total_time = 4
def condition_tuesday := time_tuesday = total_time - (time_monday + time_wednesday + time_thursday + time_friday)

-- The statement to prove:
theorem ratio_tuesday_to_monday :
  condition_monday →
  condition_wednesday →
  condition_thursday →
  condition_friday →
  condition_total →
  condition_tuesday →
  (time_tuesday / time_monday) = 2 / 3 :=
by
  intro h_monday h_wednesday h_thursday h_friday h_total h_tuesday
  rw [condition_monday, condition_wednesday, condition_thursday, condition_friday, condition_total, condition_tuesday] at *
  norm_num
  sorry

end ratio_tuesday_to_monday_l628_628580


namespace number_of_days_counted_l628_628271

-- Define Sophie receiving 20 oranges per day
def Oranges_per_day_Sophie : ℕ := 20

-- Define Hannah receiving 40 grapes per day
def Grapes_per_day_Hannah : ℕ := 40

-- Define the total number of fruits given to both girls per day
def Fruits_per_day := Oranges_per_day_Sophie + Grapes_per_day_Hannah

-- Define the total number of fruits Vicente counted over some days
def Total_fruits (d : ℕ) := Fruits_per_day * d

-- Provide the given total number of fruits counted
def Total_fruits_given : ℕ := 1800

-- The theorem to determine the number of days counted
theorem number_of_days_counted : ∃ d : ℕ, Total_fruits d = Total_fruits_given ∧ d = 30 :=
by
  use 30
  dsimp [Total_fruits, Fruits_per_day, Oranges_per_day_Sophie, Grapes_per_day_Hannah, Total_fruits_given]
  split
  . -- Show Total_fruits 30 = 1800
    calc Total_fruits 30
         = Fruits_per_day * 30 : by rfl
     ... = (Oranges_per_day_Sophie + Grapes_per_day_Hannah) * 30 : by rfl
     ... = (20 + 40) * 30 : by rfl
     ... = 60 * 30 : by rfl
     ... = 1800 : by norm_num
  . -- Show d = 30
    rfl

end number_of_days_counted_l628_628271


namespace ai_bi_ci_gt_bc_kl_l628_628969

open_locale classical

variables {A B C I N M K L : Type*}
variables [Triangle ABC] [Incenter A B C I] [Midpoint N A B] [Midpoint M A C]

theorem ai_bi_ci_gt_bc_kl 
  (I_incenter: Incenter A B C I)
  (N_mid_AB: Midpoint N A B)
  (M_mid_AC: Midpoint M A C)
  (K_inter_BI_MN: Inter (BI) (MN) K)
  (L_inter_CI_MN: Inter (CI) (MN) L) :
  dist A I + dist B I + dist C I > dist B C + dist K L :=
sorry

end ai_bi_ci_gt_bc_kl_l628_628969


namespace sum_eight_numbers_l628_628513

-- Define the conditions as hypotheses
def avg (numbers : List ℝ) := (numbers.sum) / (numbers.length)

theorem sum_eight_numbers (numbers : List ℝ) (h_avg : avg numbers = 5.6) (h_len : numbers.length = 8) : numbers.sum = 44.8 := by
  have h1 : avg numbers = (numbers.sum) / 8 := by
    simp [avg, h_len]
  rw [h1, h_avg] at h1
  sorry

end sum_eight_numbers_l628_628513


namespace largest_n_fact_product_of_four_consecutive_integers_l628_628410

theorem largest_n_fact_product_of_four_consecutive_integers :
  ∀ (n : ℕ), (∃ x : ℕ, n.factorial = x * (x + 1) * (x + 2) * (x + 3)) → n ≤ 6 :=
by
  sorry

end largest_n_fact_product_of_four_consecutive_integers_l628_628410


namespace mail_per_house_l628_628726

theorem mail_per_house (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : 
  total_mail / total_houses = 6 := 
by 
  sorry

end mail_per_house_l628_628726


namespace always_possible_to_rotate_disks_l628_628650

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end always_possible_to_rotate_disks_l628_628650


namespace maximize_profit_l628_628288

structure MarketConditions where
  cost_price : ℕ
  initial_selling_price : ℕ
  initial_units_sold : ℕ
  price_increase_effect : ℕ

noncomputable def profit_maximizing_price (conds : MarketConditions) : ℕ :=
  let x_max := -(200) / (2 * (-20))
  conds.initial_selling_price + x_max

-- Conditions for the given problem
def conds : MarketConditions :=
{ cost_price := 80,
  initial_selling_price := 90,
  initial_units_sold := 400,
  price_increase_effect := 20 }

-- The theorem we need to prove: the price that maximizes profit.
theorem maximize_profit : profit_maximizing_price conds = 95 := by
  sorry

end maximize_profit_l628_628288


namespace same_terminal_side_l628_628348

theorem same_terminal_side : ∃ k : ℤ, 36 + k * 360 = -324 :=
by
  use -1
  linarith

end same_terminal_side_l628_628348


namespace shift_sin_left_l628_628987

-- Definition of the sin function
def f (x : ℝ) : ℝ := Real.sin x

-- Shifted function g
def g (x : ℝ) : ℝ := Real.sin (x + π / 6)

-- Proof statement
theorem shift_sin_left (x : ℝ) : g x = Real.sin (x + π / 6) :=
by sorry

end shift_sin_left_l628_628987


namespace square_of_cube_of_third_smallest_prime_l628_628673

theorem square_of_cube_of_third_smallest_prime :
  let p := nat.prime 5
  let cube := p ^ 3
  let square := cube ^ 2
  square = 15625 :=
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628673


namespace triangle_ratio_l628_628101

/-- In triangle ABC, with ∠B = 45 degrees and ∠C = 30 degrees, 
and circles constructed on the medians BM and CN as diameters, 
intersecting at points P and Q, and the chord PQ intersecting side BC at point D, 
the ratio of segments BD to DC is 1 / √3. -/
theorem triangle_ratio (A B C M N P Q D : Point) 
  (BM_median : median B M A C) (CN_median : median C N A B)
  (angle_B : ∠ B A C = 45) (angle_C : ∠ C B A = 30)
  (circles_intersect : intersects (circle_on_diameter BM) (circle_on_diameter CN) P Q)
  (PQ_intersects_BC : chord_intersects_side PQ B C D) :
  ratio (segment_length B D) (segment_length D C) = 1 / Real.sqrt 3 :=
begin
  sorry
end

end triangle_ratio_l628_628101


namespace simplify_expression_l628_628459

-- Define the given expressions.
def expr1 (x : ℝ) : ℝ := (3 * x^2 - 4 * x + 1) / ((x - 1) * (x + 3))
def expr2 (x : ℝ) : ℝ := (6 * x - 5) / ((x - 1) * (x + 3))

-- State the theorem with the conditions provided.
theorem simplify_expression (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ -3) : 
  expr1 x - expr2 x = 3 :=
sorry

end simplify_expression_l628_628459


namespace pascal_triangle_no_such_row_l628_628390

noncomputable def no_special_row_in_pascals_triangle : Prop :=
  ∀ (n : ℕ), ∀ (i j k l : ℕ),
    i ≠ j → j ≠ k → k ≠ l → l ≠ i →
    i ≠ k → j ≠ l →
    binomial n i ≠ binomial n j →
    binomial n j ≠ binomial n k →
    binomial n k ≠ binomial n l →
    binomial n l ≠ binomial n i →
    (¬ (binomial n j = 2 * binomial n i ∧ binomial n l = 2 * binomial n k))

-- Include a declaration for the theorem statement
theorem pascal_triangle_no_such_row : no_special_row_in_pascals_triangle :=
  sorry

end pascal_triangle_no_such_row_l628_628390


namespace omit_monomials_preserve_root_l628_628573

theorem omit_monomials_preserve_root (P : Polynomial ℝ) (a_n a_0 : ℝ) (a_n_ne_zero : a_n ≠ 0) (a_0_ne_zero : a_0 ≠ 0) :
  (∃ x, P.eval x = 0) →
  (∃ seq : List (Polynomial ℝ), seq ≠ [] ∧ seq.contains (Polynomial.C a_0) ∧ ∀ Q ∈ seq, ∃ x, Q.eval x = 0 ∧
    ∃ R, seq.qnext Q = some R ∧ ∃ x, R.eval x = 0)
:= sorry

end omit_monomials_preserve_root_l628_628573


namespace roots_equation_l628_628633

-- Definitions of α and β based on the given conditions
variables (α β : ℝ)
-- Assumptions based on Vieta's formulas
def sum_roots (h : α + β = 1 / 2) : Prop := α + β = 1 / 2
def product_roots (h : α * β = -2) : Prop := α * β = -2

-- The necessary proof statement
theorem roots_equation (h1 : sum_roots α β (1 / 2)) (h2 : product_roots α β (-2)) :
  α^2 + α * β + β^2 = 9 / 4 :=
by
  sorry

end roots_equation_l628_628633


namespace min_val_abs_diff_l628_628834

theorem min_val_abs_diff (a1 a2 : ℝ) (h : (3 / (3 + 2 * sin a1)) + (2 / (4 - sin (2 * a2))) = 1) :
    ∃ m ∈ ℤ, ∃ k ∈ ℤ, |4 * Real.pi - a1 + a2| = (| (4 - 2 * k + m) * Real.pi - (3 * Real.pi / 4) |) := by
  sorry

end min_val_abs_diff_l628_628834


namespace regular_polygon_num_sides_l628_628039

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l628_628039


namespace standard_parabola_equation_l628_628257

-- Lean 4 statement
theorem standard_parabola_equation 
  (h₁ : ∃ (vertex : ℝ × ℝ), vertex = (0, -1)) 
  (h₂ : ∃ (focus : ℝ × ℝ), focus = (0, 1))
  (h₃ : ∃ (distance : ℝ), distance = 2) :
  ∃ (equation : ℝ → ℝ → Prop), equation = (λ x y, x^2 = 4 * y) := 
by
  sorry

end standard_parabola_equation_l628_628257


namespace sum_of_s_r_l628_628191

-- Define the domains and ranges as Sets
def r_domain : Set ℤ := {-2, -1, 0, 1, 2, 3}
def r_range : Set ℤ := {-3, -1, 0, 1, 3, 5}
def s_domain : Set ℤ := {-3, -1, 1, 3, 5}

-- Define the function r, noting only its range (not the exact mapping)
def r : ℤ → Set ℤ := λ x => if x ∈ r_domain then r_range else ∅

-- Define the function s
def s : ℤ → ℤ := λ x => x * x

-- Define the proof statement
theorem sum_of_s_r (hr : ∀ x ∈ r_domain, r x = r_range) : 
  (∑ y in ({-3, -1, 1, 3, 5} : Set ℤ), s y) = 45 :=
by
  sorry

end sum_of_s_r_l628_628191


namespace find_volume_from_vessel_c_l628_628997

noncomputable def concentration_vessel_a : ℝ := 0.45
noncomputable def concentration_vessel_b : ℝ := 0.30
noncomputable def concentration_vessel_c : ℝ := 0.10
noncomputable def volume_vessel_a : ℝ := 4
noncomputable def volume_vessel_b : ℝ := 5
noncomputable def resultant_concentration : ℝ := 0.26

theorem find_volume_from_vessel_c (x : ℝ) : 
    concentration_vessel_a * volume_vessel_a + concentration_vessel_b * volume_vessel_b + concentration_vessel_c * x = 
    resultant_concentration * (volume_vessel_a + volume_vessel_b + x) → 
    x = 6 :=
by
  sorry

end find_volume_from_vessel_c_l628_628997


namespace maxDet_l628_628944

open Matrix

def v : Vector ℝ 3 := ⟨[4, 2, -2], sorry⟩
def w : Vector ℝ 3 := ⟨[2, 0, 6], sorry⟩
def u (u_unit : ∥u∥ = 1) : Vector ℝ 3 := sorry

noncomputable def largestDeterminant (u_unit : ∥u u_unit∥ = 1) : ℝ :=
  ∥(v × w)∥

theorem maxDet : ∃ u_unit, largestDeterminant (by auto [u, is_unit u_unit]) = √944 :=
sorry

end maxDet_l628_628944


namespace area_of_shape_ADBFCEA_l628_628810

theorem area_of_shape_ADBFCEA 
  (a : ℝ) (AB D C A B F E : ℝ)
  (diam_AB : AB = 2 * D)
  (semicircle_AEC : ∀ x, x ∈ interval_integral a 0 A → dist x D = A)
  (semicircle_CFB : ∀ x, x ∈ interval_integral C B E → dist x F = B )
  (DC : dist D C = a)
  (right_angle_ADC : ∠ADC = 90)
  (right_angle_CDB : ∠CDB = 90) : 
  area_of_shape AD B F C E A = π * (a / 4)^2 := 
sorry

end area_of_shape_ADBFCEA_l628_628810


namespace find_sin_C_and_area_l628_628894

variable {a b c : ℝ}
variable {A B C : ℝ}

/-- 
Conditions:
In ΔABC, a, b, c are the sides opposite to angles A, B, C respectively.
a^2 + c^2 - √3ac = b^2
3a = 2b
-/
def triangle_conditions (a b c A B C : ℝ) : Prop :=
  a^2 + c^2 - sqrt 3 * a * c = b^2 ∧ 3 * a = 2 * b

/-- 
Part (I): Find the value of sin C
-/
def sin_C (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) : Prop :=
  sin C = (sqrt 3 + 2 * sqrt 2) / 6

/-- 
Part (II): If b = 6, find the area of ΔABC
-/
def area_of_triangle (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) (hb : b = 6) : Prop :=
  let area := 1 / 2 * a * b * sin C in
  area = 2 * sqrt 3 + 4 * sqrt 2

/-- 
Combining both parts
-/
theorem find_sin_C_and_area (a b c A B C : ℝ)
  (h : triangle_conditions a b c A B C) (hb : b = 6) :
  sin_C a b c A B C h ∧ area_of_triangle a b c A B C h hb :=
by
  sorry

end find_sin_C_and_area_l628_628894


namespace square_of_cube_of_third_smallest_prime_l628_628668

theorem square_of_cube_of_third_smallest_prime : 
  let p := 5 in (p ^ 3) ^ 2 = 15625 := 
by
  sorry

end square_of_cube_of_third_smallest_prime_l628_628668


namespace balls_in_boxes_l628_628016

theorem balls_in_boxes (n : ℕ) (h : n = 6) : 
  (number_of_ways_to_distribute_in_boxes 6 2) = 32 :=
sorry

def number_of_ways_to_distribute_in_boxes (ball_count box_count : ℕ) : ℕ :=
if box_count = 2 then
  let ways_case_1 := nat.choose ball_count 6 in
  let ways_case_2 := nat.choose ball_count 5 in
  let ways_case_3 := nat.choose ball_count 4 in
  let ways_case_4 := (nat.choose ball_count 3) / 2 in
  ways_case_1 + ways_case_2 + ways_case_3 + ways_case_4
else 0 -- This simplistic placeholder assumes box_count other than 2 are not interesting

end balls_in_boxes_l628_628016


namespace product_of_invertibles_mod_120_l628_628120

theorem product_of_invertibles_mod_120 :
  let N := 5!
  let coprime_numbers := { n | n < N ∧ Nat.gcd n N = 1 }
  let m := ∏ i in coprime_numbers, i
  m % N = 119 := by
  sorry

end product_of_invertibles_mod_120_l628_628120


namespace find_TR_l628_628084

-- Definitions
variables (PQ RS PR TR PT : ℝ) 
variables (P Q R S T : Type) 
variables [convex_quad PQRS]
variables (perpendicularPQ : ⊥ PQ RS)
variables (perpendicularRS : ⊥ RS PQ)
variables (sideRS : RS = 72)
variables (sidePQ : PQ = 30)
variables (intersectT : T ∈ line_through(Q) ∩ perpendicular(line_through(Q), PS) ∩ line_through(PR))
variables (linePT : PT = 40)

-- Theorem
theorem find_TR : TR = 50 :=
by
  sorry

end find_TR_l628_628084


namespace probability_avg_gt_five_l628_628071

theorem probability_avg_gt_five (S : set ℕ) (hS : S = {1, 3, 5, 7, 9}) : 
  (∃ (A : set ℕ), A ⊆ S ∧ A.card = 2 ∧ ((erase S A).1.sum / 3 > 5 → 
  (A.card.choose 2) / S.card.choose 2 = 2 / 5)) := 
sorry

end probability_avg_gt_five_l628_628071


namespace find_three_xsq_ysq_l628_628787

theorem find_three_xsq_ysq (x y : ℤ) (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 :=
sorry

end find_three_xsq_ysq_l628_628787


namespace regular_polygon_sides_l628_628054

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : (n * (n - 3)) / 2 = 20 → n = 8 :=
by
  -- The proof goes here
  sorry

end regular_polygon_sides_l628_628054


namespace product_coprime_mod_120_l628_628189

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628189


namespace euler_polyhedron_problem_l628_628782

theorem euler_polyhedron_problem
  (V E F : ℕ)
  (t h T H : ℕ)
  (euler_formula : V - E + F = 2)
  (faces_count : F = 30)
  (tri_hex_faces : t + h = 30)
  (edges_equation : E = (3 * t + 6 * h) / 2)
  (vertices_equation1 : V = (3 * t) / T)
  (vertices_equation2 : V = (6 * h) / H)
  (T_val : T = 1)
  (H_val : H = 2)
  (t_val : t = 10)
  (h_val : h = 20)
  (edges_val : E = 75)
  (vertices_val : V = 60) :
  100 * H + 10 * T + V = 270 :=
by
  sorry

end euler_polyhedron_problem_l628_628782


namespace problem_solution_1_problem_solution_2_l628_628640

def Sn (n : ℕ) := n * (n + 2)

def a_n (n : ℕ) := 2 * n + 1

def b_n (n : ℕ) := 2 ^ (n - 1)

def c_n (n : ℕ) := if n % 2 = 1 then 2 / Sn n else b_n n

def T_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i => c_n (i + 1))

theorem problem_solution_1 : 
  ∀ (n : ℕ), a_n n = 2 * n + 1 ∧ b_n n = 2 ^ (n - 1) := 
  by sorry

theorem problem_solution_2 (n : ℕ) : 
  T_n (2 * n) = (2 * n) / (2 * n + 1) + (2 / 3) * (4 ^ n - 1) := 
  by sorry

end problem_solution_1_problem_solution_2_l628_628640


namespace find_sides_from_diagonals_l628_628060

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l628_628060


namespace bell_ringing_count_l628_628205

-- Axiomatically define the conditions.
axiom rings_per_class : ℕ -- The number of times a bell rings per class (once at start, once at end)
axiom break_duration : ℕ -- The break duration after each class in minutes
axiom total_classes : ℕ -- The total number of classes on Monday
axiom current_class : ℕ -- The current class Madison is attending 
axiom times_bell_rings_per_class : ℕ -- The number of times the bell rings for each class including break

-- Define the specifics
def rings_per_class := 2 
def break_duration := 15
def total_classes := 5
def current_class := total_classes -- Since Madison is currently in Music class, the current class is the 5th one
def times_bell_rings_per_class := 1 -- The bell has rung once to indicate start of Music class

-- Define a theorem to prove that the bell has rung 9 times by now
theorem bell_ringing_count : 
  (4 * rings_per_class) + times_bell_rings_per_class = 9 := 
sorry

end bell_ringing_count_l628_628205


namespace moles_of_naoh_combined_l628_628418

-- Definitions based on conditions
def balanced_reaction_naoh_hcl : Prop :=
  ∀ (naoh hcl nacl h2o : Nat), 
    naoh + hcl = nacl + h2o

def reaction_produces_1_mole_water (h2o : Nat) : Prop :=
  h2o = 1

def one_mole_hcl_is_used (hcl : Nat) : Prop :=
  hcl = 1

-- The statement to prove
theorem moles_of_naoh_combined (naoh hcl nacl h2o : Nat) :
  balanced_reaction_naoh_hcl naoh hcl nacl h2o → 
  reaction_produces_1_mole_water h2o →
  one_mole_hcl_is_used hcl →
  naoh = 1 :=
by
  sorry

end moles_of_naoh_combined_l628_628418


namespace product_coprime_mod_120_l628_628183

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628183


namespace complement_of_union_l628_628203

-- Definitions based on conditions
def U := {1, 2, 3, 4, 5, 6, 7, 8}
def S := {1, 3, 5}
def T := {3, 6}

-- Define the complement operation for sets
def complement (universe set : Set ℕ) : Set ℕ :=
  universe \ set

-- Statement to be proved
theorem complement_of_union : 
  complement U (S ∪ T) = {2, 4, 7, 8} :=
by
  sorry

end complement_of_union_l628_628203


namespace polynomial_no_all_real_roots_l628_628197

theorem polynomial_no_all_real_roots (n : ℕ) (h : n > 2) : 
  ¬ ∀ x ∈ polynomial_roots (X ^ n + X ^ (n - 1) + 2 : polynomial ℝ), x ∈ ℝ :=
by
  sorry

end polynomial_no_all_real_roots_l628_628197


namespace solve_exp_eq_l628_628601

def exp_eq_condition (x : ℝ) : Prop := 2^(2 * x - 1) = 1 / 4

theorem solve_exp_eq (x : ℝ) (h : exp_eq_condition x) : x = -1 / 2 :=
by
  sorry

end solve_exp_eq_l628_628601


namespace John_Finishes_2_Meters_Ahead_l628_628551

-- Define the conditions
def john_speed := 4.2  -- John's speed in meters per second
def steve_speed := 3.8  -- Steve's speed in meters per second
def initial_gap := 15  -- Initial distance John is behind Steve in meters
def final_push_time := 42.5  -- Time taken for John's final push in seconds

-- Define the expected result
def john_finishing_distance := john_speed * final_push_time
def steve_effective_distance := (steve_speed * final_push_time) + initial_gap

-- State the theorem to prove John finishes 2 meters ahead of Steve
theorem John_Finishes_2_Meters_Ahead :
  john_finishing_distance - steve_effective_distance = 2 :=
by
  sorry

end John_Finishes_2_Meters_Ahead_l628_628551


namespace hike_length_l628_628484

-- Definitions of conditions
def initial_water : ℕ := 6
def final_water : ℕ := 1
def hike_duration : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_drunk : ℕ := 1
def first_part_drink_rate : ℚ := 2 / 3

-- Statement to prove
theorem hike_length (hike_duration : ℕ) (initial_water : ℕ) (final_water : ℕ) (leak_rate : ℕ) 
  (last_mile_drunk : ℕ) (first_part_drink_rate : ℚ) : 
  hike_duration = 2 → 
  initial_water = 6 → 
  final_water = 1 → 
  leak_rate = 1 → 
  last_mile_drunk = 1 → 
  first_part_drink_rate = 2 / 3 → 
  ∃ miles : ℕ, miles = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof placeholder
  sorry

end hike_length_l628_628484


namespace solve_for_x_l628_628496

theorem solve_for_x (x : ℝ) (h : 3 * x + 8 = -4 * x - 16) : x = -24 / 7 :=
sorry

end solve_for_x_l628_628496


namespace product_coprime_mod_120_l628_628187

-- Define the factorial of 5
def factorial_5 : ℕ := 5!

-- Define a set of integers less than 120 that are coprime to 120
def coprime_to_120 (n : ℕ) : Prop := Nat.gcd n 120 = 1

-- Define the product of all integers less than 120 that are coprime to 120
noncomputable def product_coprime_less_120 : ℕ :=
  ∏ i in Finset.filter coprime_to_120 (Finset.range factorial_5), i

-- State the theorem
theorem product_coprime_mod_120 : product_coprime_less_120 % factorial_5 = 1 := 
sorry

end product_coprime_mod_120_l628_628187
