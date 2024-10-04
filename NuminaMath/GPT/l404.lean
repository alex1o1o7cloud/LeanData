import Mathlib
import Mathlib.Algebra.Abs
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Cyclic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCDMonoid.Multiset
import Mathlib.Algebra.Order.Monoid
import Mathlib.Algebra.QuadraticEq
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Area
import Mathlib.Analysis.Calculus.Eccentricity
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Catalan
import Mathlib.Combinatorics.CombinatorialEnumeration
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Statistics
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Seq
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basics
import Mathlib.Geometry.Euclidean.Parallel
import Mathlib.Init.Data.Rat.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.ProbabilityTheory.Normal
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Real
import data.nat.sqrt
import data.real.basic

namespace binomial_coefficient_12_10_l404_404604

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404604


namespace two_by_two_subarray_sum_gt_3n_l404_404011

theorem two_by_two_subarray_sum_gt_3n {n : ℕ} (h1 : n ≥ 3) (a : Fin n → Fin n → ℝ)
  (h2 : ∀ i j, 0 < a i j) (h3 : ∑ i, ∑ j, a i j = n^3) :
  ∃ (i j : Fin n) (k l : Fin n), i ≠ k ∧ j ≠ l ∧ a i j + a i l + a k j + a k l > 3 * n :=
by
  sorry

end two_by_two_subarray_sum_gt_3n_l404_404011


namespace num_values_of_n_l404_404239

theorem num_values_of_n (a b c : ℕ) (h : 7 * a + 77 * b + 7777 * c = 8000) : 
  ∃ n : ℕ, (n = a + 2 * b + 4 * c) ∧ (110 * n ≤ 114300) ∧ ((8000 - 7 * a) % 70 = 7 * (10 * b + 111 * c) % 70) := 
sorry

end num_values_of_n_l404_404239


namespace integer_root_of_polynomial_l404_404101

theorem integer_root_of_polynomial 
  (b c : ℚ) 
  (h1 : is_root (λ x : ℚ, x^3 + b*x + c) (3 - real.sqrt 5))
  (h2 : ∃ x : ℤ, is_root (λ x : ℚ, x^3 + b*x + c) (x : ℚ)) :
  ∃ r : ℤ, r = -6 :=
by 
  sorry

end integer_root_of_polynomial_l404_404101


namespace num_real_solutions_eq_l404_404268

theorem num_real_solutions_eq (f g : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x / 150) →
  (∀ x, g x = Real.cos x) →
  a = -150 →
  b = 150 →
  (Set.Icc a b).countOnIntervalSolutions (λ x, f x = g x) = 93 :=
by
  intros hf hg ha hb
  sorry

end num_real_solutions_eq_l404_404268


namespace tom_made_washing_cars_l404_404391

-- Definitions of the conditions
def initial_amount : ℕ := 74
def final_amount : ℕ := 86

-- Statement to be proved
theorem tom_made_washing_cars : final_amount - initial_amount = 12 := by
  sorry

end tom_made_washing_cars_l404_404391


namespace find_c_max_value_l404_404378

-- Assume a, b, and c are the distinct values 1, 2, or 4
variables (a b c : ℕ)
-- The values of a, b, and c are in the set {1, 2, 4}.
variables (h_a : a ∈ {1, 2, 4})
variables (h_b : b ∈ {1, 2, 4})
variables (h_c : c ∈ {1, 2, 4})
-- a, b, and c are different
variables (h_diff : a ≠ b ∧ a ≠ c ∧ b ≠ c)
-- The largest possible value of the expression is 4.
theorem find_c_max_value :
  (a / 2) / (b / c) = 4 → c = 2 :=
sorry

end find_c_max_value_l404_404378


namespace find_AB_l404_404746

-- Definitions of geometric objects and conditions 
def is_rectangle (A B C D : Point) : Prop := sorry
def on_side_BC (P B C : Point) : Prop := sorry
def BP_length (P B : Point) : Real := 9
def CP_length (P C : Point) : Real := 15
def tan_angle_APD (A P D Point) : Real := 2

-- Statement to prove
theorem find_AB (A B C D P : Point) (h1 : is_rectangle A B C D)
                (h2 : on_side_BC P B C) 
                (h3 : BP_length P B = 9)
                (h4 : CP_length P C = 15)
                (h5 : tan_angle_APD A P D = 2) : 
                    ∃ (x : Real), AB_length A B = x ∧ x = 6 + Real.sqrt 171 := 
    sorry

end find_AB_l404_404746


namespace sum_of_first_n_terms_l404_404658

theorem sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 →
  (∀ n, a (n + 1) - a n = 2^n) →
  (∀ n, S n = ∑ k in finset.range n, a (k + 1)) →
  ∀ n, S n = 2^(n + 1) - n - 2 :=
by
  assume h1 h2 h3,
  sorry

end sum_of_first_n_terms_l404_404658


namespace friends_division_ways_l404_404719

theorem friends_division_ways : (4 ^ 8 = 65536) :=
by
  sorry

end friends_division_ways_l404_404719


namespace john_took_11_more_l404_404073

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l404_404073


namespace sales_volume_expression_reduction_for_desired_profit_l404_404172

-- Initial conditions definitions.
def initial_purchase_price : ℝ := 3
def initial_selling_price : ℝ := 5
def initial_sales_volume : ℝ := 100
def sales_increase_per_0_1_yuan : ℝ := 20
def desired_profit : ℝ := 300
def minimum_sales_volume : ℝ := 220

-- Question (1): Sales Volume Expression
theorem sales_volume_expression (x : ℝ) : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) = 100 + 200 * x :=
by sorry

-- Question (2): Determine Reduction for Desired Profit and Minimum Sales Volume
theorem reduction_for_desired_profit (x : ℝ) 
  (hx : (initial_selling_price - initial_purchase_price - x) * (initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x)) = desired_profit)
  (hy : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) >= minimum_sales_volume) :
  x = 1 :=
by sorry

end sales_volume_expression_reduction_for_desired_profit_l404_404172


namespace chocolate_bar_cost_l404_404254

theorem chocolate_bar_cost (x : ℝ) (total_bars : ℕ) (bars_sold : ℕ) (total_amount_made : ℝ)
    (h1 : total_bars = 7)
    (h2 : bars_sold = total_bars - 4)
    (h3 : total_amount_made = 9)
    (h4 : total_amount_made = bars_sold * x) : x = 3 :=
sorry

end chocolate_bar_cost_l404_404254


namespace packs_of_gum_bought_l404_404219

noncomputable def initial_amount : ℝ := 10.00
noncomputable def gum_cost : ℝ := 1.00
noncomputable def choc_bars : ℝ := 5.00
noncomputable def choc_bar_cost : ℝ := 1.00
noncomputable def candy_canes : ℝ := 2.00
noncomputable def candy_cane_cost : ℝ := 0.50
noncomputable def leftover_amount : ℝ := 1.00

theorem packs_of_gum_bought : (initial_amount - leftover_amount - (choc_bars * choc_bar_cost + candy_canes * candy_cane_cost)) / gum_cost = 3 :=
by
  sorry

end packs_of_gum_bought_l404_404219


namespace power_function_value_at_4_l404_404673
noncomputable theory

def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_function_value_at_4 :
  (∃ α : ℝ, power_function α 2 = (real.sqrt 2) / 2) →
  power_function (-1 / 2) 4 = 1 / 2 :=
by sorry

end power_function_value_at_4_l404_404673


namespace base_conversion_and_addition_l404_404986

theorem base_conversion_and_addition :
  let n1 := 2 * (8:ℕ)^2 + 4 * 8^1 + 3 * 8^0
  let d1 := 1 * 4^1 + 3 * 4^0
  let n2 := 2 * 7^2 + 0 * 7^1 + 4 * 7^0
  let d2 := 2 * 5^1 + 3 * 5^0
  n1 / d1 + n2 / d2 = 31 + 51 / 91 := by
  sorry

end base_conversion_and_addition_l404_404986


namespace total_clothing_ironed_l404_404437

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l404_404437


namespace tetrahedron_net_self_intersects_l404_404916

-- Define a structure representing a tetrahedron.
structure Tetrahedron (V : Type) :=
  (vertices : fin 4 → V)

-- Define conditions for cutting edges that are not on the same face.
def non_adjacent_edges {V : Type} (T : Tetrahedron V) (e1 e2 e3 : (V × V)) : Prop :=
  ∀ (v : V), v ∈ e1 ∨ v ∈ e2 → e1 ≠ e2 ∧ e2 ≠ e3 ∧ e1 ≠ e3

-- The main theorem expressing the possibility of a self-intersecting net.
theorem tetrahedron_net_self_intersects {V : Type} 
  (T : Tetrahedron V) (e1 e2 e3 : (V × V)) : 
  non_adjacent_edges T e1 e2 e3 → 
  ∃ (net : set (V × V)), self_intersecting net := 
sorry

end tetrahedron_net_self_intersects_l404_404916


namespace arrange_numbers_l404_404680

namespace MathProofs

theorem arrange_numbers (a b : ℚ) (h1 : a > 0) (h2 : b < 0) (h3 : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by
  -- Proof to be completed
  sorry

end MathProofs

end arrange_numbers_l404_404680


namespace cos_pi_plus_2alpha_value_l404_404290

theorem cos_pi_plus_2alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : 
    Real.cos (π + 2 * α) = 7 / 9 := sorry

end cos_pi_plus_2alpha_value_l404_404290


namespace total_savings_l404_404664

-- Define the given conditions
def number_of_tires : ℕ := 4
def sale_price : ℕ := 75
def original_price : ℕ := 84

-- State the proof problem
theorem total_savings : (original_price - sale_price) * number_of_tires = 36 :=
by
  -- Proof omitted
  sorry

end total_savings_l404_404664


namespace smallest_n_gcd_conditions_l404_404976

theorem smallest_n_gcd_conditions : 
  ∃ n : ℕ, n > 1000 ∧ 
  n = 1917 ∧ 
  (∑ d in (1917.digits 10), d) = 18 ∧ 
  (gcd 63 (n + 120) = 21) ∧ 
  (gcd (n + 63) 120 = 60) :=
sorry

end smallest_n_gcd_conditions_l404_404976


namespace find_max_possible_integer_l404_404178

noncomputable def max_integers_list (l : List ℕ) : Prop :=
  l.length = 5 ∧
  (∀ n, l.count n > 1 → n = 10) ∧
  (l.sorted.nth 2 = some 11) ∧
  (l.sum = 60)

theorem find_max_possible_integer :
  ∃ l : List ℕ, max_integers_list l ∧ List.maximum l = some 17 := 
by
  sorry

end find_max_possible_integer_l404_404178


namespace initial_card_count_l404_404169

theorem initial_card_count (r b : ℕ) (h₁ : (r : ℝ)/(r + b) = 1/4)
    (h₂ : (r : ℝ)/(r + (b + 6)) = 1/6) : r + b = 12 :=
by
  sorry

end initial_card_count_l404_404169


namespace inequality_must_hold_l404_404722

theorem inequality_must_hold (m n : ℝ) (h : m > n) : 2 + m > 2 + n :=
sorry

end inequality_must_hold_l404_404722


namespace abs_nonneg_position_l404_404848

theorem abs_nonneg_position (a : ℝ) : 0 ≤ |a| ∧ |a| ≥ 0 → (exists x : ℝ, x = |a| ∧ x ≥ 0) :=
by 
  sorry

end abs_nonneg_position_l404_404848


namespace number_of_zeros_l404_404096

def segment_1 (x : ℝ) : ℝ := x^2 + 2*x - 3
def segment_2 (x : ℝ) (hx : x > 0) : ℝ := Real.log x - 2

theorem number_of_zeros (f : ℝ → ℝ)
  (h₁ : ∀ x, x ≤ 0 → f x = segment_1 x)
  (h₂ : ∀ x, x > 0 → f x = segment_2 x)
  : ∃! x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 :=
sorry

end number_of_zeros_l404_404096


namespace inequality_solution_l404_404486

theorem inequality_solution :
  { x : ℝ | x - 2 < 0 ∧ 5x + 1 > 2 * (x - 1) } = { x : ℝ | -1/3 < x ∧ x < 2 } :=
by
  sorry

end inequality_solution_l404_404486


namespace coprime_with_form_l404_404632

theorem coprime_with_form (x : ℕ) (h_pos : x > 0) : (∀ n : ℕ, Nat.coprime x (2^n + 3^n + 6^n - 1)) ↔ x = 1 := by
  sorry

end coprime_with_form_l404_404632


namespace coterminal_angle_l404_404567

theorem coterminal_angle :
  ∀ θ : ℤ, (θ - 60) % 360 = 0 → θ = -300 ∨ θ = -60 ∨ θ = 600 ∨ θ = 1380 :=
by
  sorry

end coterminal_angle_l404_404567


namespace distinct_integer_values_in_interval_l404_404277

def f (x : ℝ) : ℝ := ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_integer_values_in_interval :
  ∀ x, 0 ≤ x ∧ x ≤ 100 → set.countable {y : ℝ | ∃ x, f x = y} = 734 :=
by sorry

end distinct_integer_values_in_interval_l404_404277


namespace cleaning_time_l404_404951

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l404_404951


namespace Lucy_needs_more_distance_John_needs_more_distance_Difference_between_Susan_and_Edna_l404_404035

def Mary_distance := (3 / 8) * 24
def Edna_distance := (2 / 3) * Mary_distance
def Lucy_distance := (5 / 6) * Edna_distance
def John_distance := (13 / 16) * Lucy_distance
def Susan_distance := (8 / 15) * John_distance

theorem Lucy_needs_more_distance : Mary_distance - Lucy_distance = 4 := by {
  -- Proof goes here
  sorry
}

theorem John_needs_more_distance : Edna_distance - John_distance = 1.75 := by {
  -- Proof goes here
  sorry
}

theorem Difference_between_Susan_and_Edna : Edna_distance - Susan_distance = 3.75 := by {
  -- Proof goes here
  sorry
}

end Lucy_needs_more_distance_John_needs_more_distance_Difference_between_Susan_and_Edna_l404_404035


namespace binomial_12_10_eq_66_l404_404612

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404612


namespace fold_line_equation_correct_l404_404099

def Point := (ℝ × ℝ)

def is_midpoint (M: Point) (P1 P2: Point) : Prop :=
  M = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def is_perpendicular_bisector (l : ℝ → ℝ) (P1 P2 : Point) : Prop :=
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2) in
  is_midpoint midpoint P1 P2 ∧ 
  ∀ x, l x = (-1) * x + (midpoint.2 + midpoint.1)

theorem fold_line_equation_correct :
  ∃ l : ℝ → ℝ, (is_perpendicular_bisector l (5, 3) (1, -1)) ∧ (∀ x, l x = -x + 4) :=
sorry

end fold_line_equation_correct_l404_404099


namespace function_bound_on_positive_measure_l404_404768

noncomputable def f : ℝ → ℝ := sorry
def n : ℕ := sorry

axiom integrable_f : integrable_on f (set.Icc 0 1) measure_theory.measure_space.volume
axiom integral_conditions (m : ℕ) (hm : m < n) : ∫ x in 0..1, x^m * f x = 0
axiom nth_integral_condition : ∫ x in 0..1, x^n * f x = 1

theorem function_bound_on_positive_measure : 
  ∃ (s : set ℝ), measure_theory.measure_space.volume s > 0 ∧ ∀ x ∈ s, |f x| ≥ 2^n * (n + 1) :=
sorry

end function_bound_on_positive_measure_l404_404768


namespace John_took_more_chickens_than_Ray_l404_404071

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l404_404071


namespace avg_scores_relation_l404_404232

variables (class_avg top8_avg other32_avg : ℝ)

theorem avg_scores_relation (h1 : 40 = 40) 
  (h2 : top8_avg = class_avg + 3) :
  other32_avg = top8_avg - 3.75 :=
sorry

end avg_scores_relation_l404_404232


namespace prime_reciprocal_sum_lt_four_l404_404279

open Real Classical

noncomputable def prime_reciprocal_sum (n : ℕ) : ℝ :=
∑ p in (Finset.filter Nat.Prime (Finset.Icc n (n^4))), (1 : ℝ) / p

theorem prime_reciprocal_sum_lt_four (n : ℕ) (h : 0 < n) : prime_reciprocal_sum n < 4 :=
sorry

end prime_reciprocal_sum_lt_four_l404_404279


namespace evaluate_expression_l404_404259

theorem evaluate_expression : 
    (13.factorial - 12.factorial - 2 * 11.factorial) / 10.factorial = 1430 := 
by
  sorry

end evaluate_expression_l404_404259


namespace MikiJuiceBlendResult_l404_404417

def MikiJuiceBlend (totalApples totalOranges totalPears : Nat)
                   (pearJuiceFromPears orangeJuiceFromOranges appleJuiceFromApples : Nat)
                   (pearJuiceAmount orangeJuiceAmount appleJuiceAmount : ℕ) 
                   (equalAmountFruits: ℕ ) 
                   (pearToJuice orangeToJuice appleToJuice : ℝ) 
                   (percentageOfOrangeJuice blendTotalJuice: ℝ) : Prop :=
  totalApples = 15 ∧ totalOranges = 12 ∧ totalPears = 12 ∧
  pearJuiceFromPears = 10 ∧ orangeJuiceFromOranges = 12 ∧ appleJuiceFromApples = 9 ∧
  pearJuiceAmount = 4 ∧ orangeJuiceAmount = 3 ∧ appleJuiceAmount = 5 ∧
  equalAmountFruits = 3 ∧ 
  pearToJuice = (10 / 4) ∧ orangeToJuice = (12 / 3) ∧ appleToJuice = (9 / 5) ∧
  blendTotalJuice = (3 * pearToJuice + 3 * orangeToJuice + 3 * appleToJuice) ∧
  percentageOfOrangeJuice = (3 * orangeToJuice / blendTotalJuice * 100)

theorem MikiJuiceBlendResult : MikiJuiceBlend 15 12 12 10 12 9 4 3 5 3 (10 / 4) (12 / 3) (9 / 5) (3 * (12 / 3) / (3 * (10 / 4) + 3 * (12 / 3) + 3 * (9 / 5)) * 100) :=
by
  -- skip the proof
  sorry

end MikiJuiceBlendResult_l404_404417


namespace equation_is_hyperbola_l404_404250

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

-- Theorem stating that the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ x y : ℝ, equation x y → (∃ A B : ℝ, A * x^2 - B * y^2 = 1) :=
by
  sorry

end equation_is_hyperbola_l404_404250


namespace asymptotes_of_hyperbola_l404_404314

noncomputable def focus_parabola (p : ℝ) : ℝ×ℝ :=
(2 * p, 0)

noncomputable def hyperbola_components (c : ℝ) : ℝ × ℝ :=
let a := sqrt(c^2 - 3) in (a, sqrt(3))

theorem asymptotes_of_hyperbola :
  let focus_par := focus_parabola 2 in
  let right_focus_hyp := (focus_par) in
  let c := 2 in
  let (a, b) := hyperbola_components c in
  a = 1 ∧ b = sqrt 3 →
  ∃ (f : ℝ → ℝ), ∀ x, f x = √3 * x ∨ f x = -√3 * x :=
begin
  sorry
end

end asymptotes_of_hyperbola_l404_404314


namespace problem_a_problem_b_l404_404739

-- Problem (a)
theorem problem_a (cities : Type*) [fintype cities] (roads : cities → cities → Prop)
  (hcount : fintype.card cities = 101)
  (hdeg : ∀ x : cities, (finset.filter (λ y, roads x y) finset.univ).card = 50 ∧ (finset.filter (λ y, roads y x) finset.univ).card = 50)
  (hconn : ∀ x y : cities, x ≠ y → roads x y ∨ roads y x):
  ∀ A B : cities, A ≠ B → ∃ C : cities, roads A C ∧ roads C B := 
by
  sorry

-- Problem (b)
theorem problem_b (cities : Type*) [fintype cities] (roads : cities → cities → Prop)
  (hcount : fintype.card cities = 101)
  (hdeg : ∀ x : cities, (finset.filter (λ y, roads x y) finset.univ).card = 40 ∧ (finset.filter (λ y, roads y x) finset.univ).card = 40):
  ∀ A B : cities, A ≠ B → ∃ C D : cities, roads A C ∧ roads C D ∧ roads D B := 
by
  sorry

end problem_a_problem_b_l404_404739


namespace solution_to_water_l404_404735

theorem solution_to_water (A W S T: ℝ) (h1: A = 0.04) (h2: W = 0.02) (h3: S = 0.06) (h4: T = 0.48) :
  (T * (W / S) = 0.16) :=
by
  sorry

end solution_to_water_l404_404735


namespace estimate_students_scores_l404_404566

noncomputable def students_scoring_within_interval 
  (xi : ℝ → ℝ) 
  (N : ℕ)
  (mu : ℝ) 
  (sigma : ℝ) 
  (P_interval1 : ℝ) 
  (P_interval2 : ℝ) : ℝ :=
100000 * (P_interval2 - P_interval1)

theorem estimate_students_scores 
  : students_scoring_within_interval 
      (Normal 70 25) 
      100000 
      70 
      5 
      0.3413 
      0.4772 = 13590 :=
sorry

end estimate_students_scores_l404_404566


namespace problem_I_problem_II_l404_404028

def function_f (a x : ℝ) : ℝ := x^2 - |x - a| + 1

theorem problem_I (a : ℝ) (ha : a = 0) :
  ∃ x ∈ Icc 0 2, (∀ y ∈ Icc 0 2, function_f a x ≤ function_f a y) ∧ (∀ y ∈ Icc 0 2, function_f a y ≤ function_f a 2) :=
by
  have hmin : ∃ x ∈ Icc 0 2, ∀ y ∈ Icc 0 2, function_f a x ≤ function_f a y, from sorry,
  have hmax : ∀ y ∈ Icc 0 2, function_f a y ≤ function_f a 2, from sorry,
  exact ⟨_, _, hmin, hmax⟩ -- construct the exact result ⟩

theorem problem_II (a : ℝ) :
  (a < 0 → ∃ x, (∀ y, function_f a x ≤ function_f a y) ∧ function_f a x = 3/4 + a) ∧ 
  (a ≥ 0 → ∃ x, (∀ y, function_f a x ≤ function_f a y) ∧ function_f a x = 3/4 - a) :=
by
  split
  { intro ha_neg; use 0; split
    { assume y, sorry -- f(x) minimum proof 
    } sorry -- function value proof
  }
  { intro ha_nonneg; use 0; split
    { assume y, sorry -- f(x) minimum proof 
    } sorry -- function value proof
  }

end problem_I_problem_II_l404_404028


namespace proposition_A_l404_404320

-- Proposition A
def SetA (n : ℤ) : ℤ := 2 * n - 1
def SetB (n : ℤ) : ℤ := 2 * n + 1

theorem proposition_A : ∀ x : ℤ, (∃ n : ℤ, x = SetA n) ↔ (∃ n : ℤ, x = SetB n) :=
by sorry

-- Proposition C
variables (a b : ℝ)

example (log2_3 : real.log 3 / real.log 2 = a) (log2_7 : real.log 7 / real.log 2 = b) :
  real.log 56 / real.log 42 = (3 + b) / (a + b + 1) :=
by sorry

end proposition_A_l404_404320


namespace two_person_subcommittees_l404_404713

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l404_404713


namespace geometric_sequence_properties_l404_404346

variable (a : ℕ → ℝ) (q : ℝ) (k : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

axiom h1 : is_geometric_sequence a q
axiom h2 : k ≠ 0

theorem geometric_sequence_properties :
  is_geometric_sequence (λ n, (a n) ^ 2) (q ^ 2) ∧
  is_geometric_sequence (λ n, k * a n) q ∧
  is_geometric_sequence (λ n, 1 / a n) (1 / q) :=
by
  sorry

end geometric_sequence_properties_l404_404346


namespace pencils_placed_by_sara_l404_404856

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l404_404856


namespace pots_calculation_l404_404111

def numPots (numFlowersEachPot totalFlowers : ℕ) : ℕ :=
  totalFlowers / numFlowersEachPot

theorem pots_calculation (h1 : ∀ p, p = 71) (h2 : ∀ t, t = 10011) : numPots 71 10011 = 141 := by
  rw [h1, h2]
  exact Nat.div_def' 10011 71 sorry sorry

end pots_calculation_l404_404111


namespace max_distance_and_area_of_coverage_ring_l404_404863

theorem max_distance_and_area_of_coverage_ring (
  radars : ℕ,
  radius : ℝ,
  width : ℝ,
  angle_div : ℝ
) 
(h_radars : radars = 9)
(h_radius : radius = 61)
(h_width : width = 22)
(h_angle_div : angle_div = 20) :
  let d := (60 / Real.sin (Real.pi * angle_div / 180))
  let A := (2640 * Real.pi) / Real.tan (Real.pi * angle_div / 180)
  in (d, A) = (60 / Real.sin (Real.pi * 20 / 180), 2640 * Real.pi / Real.tan (Real.pi * 20 / 180)) :=
by
  rw [h_radars, h_radius, h_width, h_angle_div]
  sorry

end max_distance_and_area_of_coverage_ring_l404_404863


namespace julie_total_earnings_l404_404005

def hourly_rate_mowing : ℝ := 4
def hourly_rate_weeding : ℝ := 8
def hours_mowing_in_september : ℝ := 25
def hours_weeding_in_september : ℝ := 3

def earnings_september : ℝ := (hourly_rate_mowing * hours_mowing_in_september) +
                              (hourly_rate_weeding * hours_weeding_in_september)

def earnings_both_months : ℝ := 2 * earnings_september

theorem julie_total_earnings : earnings_both_months = 248 :=
by
  calc
    earnings_both_months = 2 * earnings_september : by rfl
    ... = 2 * ((hourly_rate_mowing * hours_mowing_in_september)
              + (hourly_rate_weeding * hours_weeding_in_september)) : by rfl
    ... = 2 * ((4 * 25) + (8 * 3)) : by rfl
    ... = 2 * (100 + 24) : by rfl
    ... = 2 * 124 : by rfl
    ... = 248 : by rfl

end julie_total_earnings_l404_404005


namespace proof_problem_l404_404315

def f : ℝ → ℝ := sorry

def a : ℕ → ℝ 
| 1 := -1
| n := 2 * a (n - 1) - 1

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry

lemma periodicity (x : ℝ) : f (3 + x) = f x := sorry

lemma f_neg_two : f (-2) = -3 := sorry

lemma sequence_relation (n : ℕ) (Sn : ℝ) (a_n : ℝ) : (Sn / n = 2 * (a_n / n) + 1) → Sn = 2 * a_n + n := sorry

theorem proof_problem : f (a 5) + f (a 6) = 3 := sorry

end proof_problem_l404_404315


namespace total_students_l404_404112

theorem total_students (students_per_classroom : ℕ) (num_classrooms : ℕ) (h1 : students_per_classroom = 30) (h2 : num_classrooms = 13) : students_per_classroom * num_classrooms = 390 :=
by
  -- Begin the proof
  sorry

end total_students_l404_404112


namespace correlation_coefficient_measure_l404_404080

theorem correlation_coefficient_measure 
  (A : Prop := "The strength of the linear relationship between two variables")
  (B : Prop := "Whether the scatter plot shows a meaningful model")
  (C : Prop := "Whether there is a causal relationship between two variables")
  (D : Prop := "Whether there is a relationship between two variables")
  (correct_answer : Prop := A):
  correct_answer = A :=
by
  sorry

end correlation_coefficient_measure_l404_404080


namespace blue_balls_removed_l404_404176

theorem blue_balls_removed (x : ℕ) 
  (initial_total_balls : ℕ) 
  (initial_blue_balls : ℕ) 
  (probability_blue_after_removal : ℚ) 
  (h1 : initial_total_balls = 18) 
  (h2 : initial_blue_balls = 6) 
  (h3 : probability_blue_after_removal = 1/5) 
  (h4 : x > 0) : x = 3 :=
by
  -- Definitions and conditions
  have total_balls_left := initial_total_balls - x
  have blue_balls_left := initial_blue_balls - x
  have probability := (blue_balls_left : ℚ) / (total_balls_left : ℚ)
  -- Equation from the probability
  have eq : probability = probability_blue_after_removal := by rw [h3]; exact eq.rfl
  -- Solving the equation
  have calc : (6 - x : ℚ) / (18 - x : ℚ) = 1 / 5 := sorry
  -- Conclusion
  exact calc


end blue_balls_removed_l404_404176


namespace longer_diagonal_length_l404_404559

-- Define the side length of the rhombus
def side_length : ℝ := 40

-- Define the half-length of the shorter diagonal
def half_shorter_diagonal : ℝ := 28

-- Define the correct length of the half longer diagonal
def half_longer_diagonal : ℝ := 12 * Real.sqrt 17

-- Prove the length of the longer diagonal
theorem longer_diagonal_length (a b c : ℝ) (h1 : a = side_length) (h2 : b = half_shorter_diagonal) :
  2 * (Real.sqrt ((side_length ^ 2) - (half_shorter_diagonal ^ 2))) = 24 * Real.sqrt 17 := by
sorry

end longer_diagonal_length_l404_404559


namespace probability_of_sum_17_is_1_over_25_l404_404091
noncomputable def probability_sum_17_decagonal_dice : ℚ :=
  let dice := {x | 1 ≤ x ∧ x ≤ 10}
  let outcomes := (dice × dice).filter (λ p, p.fst + p.snd = 17)
  outcomes.card / (dice.card * dice.card)

theorem probability_of_sum_17_is_1_over_25 :
  probability_sum_17_decagonal_dice = 1 / 25 :=
sorry

end probability_of_sum_17_is_1_over_25_l404_404091


namespace area_of_triangle_ABC_l404_404324

noncomputable def f (x : ℝ) := cos x * (sin x - sqrt 3 * cos x)

def smallest_positive_period : ℝ := π

def decreasing_interval (k : ℤ) : set ℝ :=
  {x | ∃ y ∈ Icc (5 * π / 12 + k * π) (11 * π / 12 + k * π), x = y}

variables {a b c : ℝ} {A B C : ℝ}
variables (a := 3) (b + c := 2 * sqrt 3)
variables (A B C := sum_of_angles)

axiom side_opposites (A B C : ℝ) : A + B + C = π

axiom f_value (A : ℝ) : f (A / 2) = -sqrt 3 / 2

theorem area_of_triangle_ABC : 
  a = 3 → 
  b + c = 2 * sqrt 3 → 
  f (A / 2) = - sqrt 3 / 2 → 
  ∃ (S : ℝ), S = (sqrt 3 / 4) :=
by
  intro h1 h2 h3
  sorry

end area_of_triangle_ABC_l404_404324


namespace coord_sum_D_l404_404820

def is_midpoint (M C D : ℝ × ℝ) := M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem coord_sum_D (M C D : ℝ × ℝ) (h : is_midpoint M C D) (hM : M = (4, 6)) (hC : C = (10, 2)) :
  D.1 + D.2 = 8 :=
sorry

end coord_sum_D_l404_404820


namespace find_vector_l404_404243

def line_r (t : ℝ) : ℝ × ℝ :=
  (2 + 5 * t, 3 - 2 * t)

def line_s (u : ℝ) : ℝ × ℝ :=
  (1 + 5 * u, -2 - 2 * u)

def is_projection (w1 w2 : ℝ) : Prop :=
  w1 - w2 = 3

theorem find_vector (w1 w2 : ℝ) (h_proj : is_projection w1 w2) :
  (w1, w2) = (-2, -5) :=
sorry

end find_vector_l404_404243


namespace actual_total_area_in_acres_l404_404909

-- Define the conditions
def base_cm : ℝ := 20
def height_cm : ℝ := 12
def rect_length_cm : ℝ := 20
def rect_width_cm : ℝ := 5
def scale_cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

-- Define the total area in acres calculation
def total_area_cm_squared : ℝ := 120 + 100
def total_area_miles_squared : ℝ := total_area_cm_squared * (scale_cm_to_miles ^ 2)
def total_area_acres : ℝ := total_area_miles_squared * sq_mile_to_acres

-- The theorem statement
theorem actual_total_area_in_acres : total_area_acres = 1267200 :=
by
  sorry

end actual_total_area_in_acres_l404_404909


namespace find_number_l404_404544

theorem find_number (x : ℤ) (h1 : x - 2 + 4 = 9) : x = 7 :=
by
  sorry

end find_number_l404_404544


namespace find_f_value_l404_404316

noncomputable def f : ℝ → ℝ := sorry -- Conditions will define this function

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.logBase 2 (x + 1)

theorem find_f_value : f (-2008) + f 2009 = 1 := sorry

end find_f_value_l404_404316


namespace x_plus_y_equals_22_l404_404288

theorem x_plus_y_equals_22 (x y : ℕ) (h1 : 2^x = 4^(y + 2)) (h2 : 27^y = 9^(x - 7)) : x + y = 22 := 
sorry

end x_plus_y_equals_22_l404_404288


namespace divide_pentagon_into_square_l404_404394

-- Define the geometric setup and conditions
structure Rectangle (α : Type*) :=
(A B C D P : α)
(angle_BPC_eq_90 : ∠BPC = 90)
(area_ABPCD_eq_AB2 : area (polygon.mk [A, B, P, C, D]) = dist A B ^ 2)

-- Main proposition
theorem divide_pentagon_into_square {α : Type*} [MetricSpace α] (r : Rectangle α) :
  ∃ (pieces : list (set α)), (∀ p₁ p₂ ∈ pieces, disjoint p₁ p₂) ∧
  (⋃₀ pieces) = {A, B, P, C, D} ∧ -- Assuming {A, B, P, C, D} represents the pentagon
  ∃ s : set α, (is_square s) ∧ (⋃₀ pieces) = s :=
begin
  sorry, 
end

end divide_pentagon_into_square_l404_404394


namespace greatest_sum_faces_one_cube_l404_404426

theorem greatest_sum_faces_one_cube : 
  (∃ A B : ℕ → ℕ, (∀ k, 2 ≤ k ∧ k ≤ 12 → (A k + B k = 6 - |7 - k|))
    ∧ ((∑ i in finset.range 6, A i) = 33))
  :=
sorry

end greatest_sum_faces_one_cube_l404_404426


namespace max_k_l404_404789

theorem max_k (n : ℕ) (h : n ≥ 3) (M : Finset (fin n)) :
  ∃ k : ℕ, 
    (if n ≤ 5 then k = Nat.choose n 3 else k = Nat.choose (n-1) 2) := 
by
  sorry

end max_k_l404_404789


namespace equal_segments_in_equilateral_triangles_l404_404786

variables {A B C A' B' C' : Type} [metric_space Type]

-- Definitions of equilateral triangles

def is_equilateral_triangle (a b c : Type) [metric_space Type] : Prop :=
  dist a b = dist b c ∧ dist b c = dist c a ∧ dist c a = dist a b

noncomputable def triangle_equilateral_conditions
  (A B C A' B' C' : Type) [metric_space Type] :=
  is_equilateral_triangle A B C ∧
  is_equilateral_triangle C B A' ∧
  is_equilateral_triangle B A C' ∧
  is_equilateral_triangle A C B'

-- The theorem we need to prove

theorem equal_segments_in_equilateral_triangles 
  (h : triangle_equilateral_conditions A B C A' B' C') :
  dist A A' = dist B B' ∧ dist B B' = dist C C' :=
begin
  sorry
end

end equal_segments_in_equilateral_triangles_l404_404786


namespace dreamland_partition_l404_404082

theorem dreamland_partition :
  ∃ k : ℕ, (∀ (G : Type) [fintype G] [graph G (λ v, v)],
    (∀ (v : G), ∃! w : G, graph.reachable v w) →
    (∃ P : finset (finset G), P.card = k ∧ ∀ A ∈ P, ∀ v w ∈ A, ¬graph.reachable v w → false)) ∧ k = 57 :=
begin
  sorry
end

end dreamland_partition_l404_404082


namespace isle_of_unluckiness_l404_404039

-- Definitions:
def is_knight (i : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k = i * n / 100 ∧ k > 0

-- Main statement:
theorem isle_of_unluckiness (n : ℕ) (h : n ∈ [1, 2, 4, 5, 10, 20, 25, 50, 100]) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ is_knight i n := by
  sorry

end isle_of_unluckiness_l404_404039


namespace radius_of_circle_l404_404481

theorem radius_of_circle (K L P Q M : Point) (r : ℝ)  
  (h1 : ∃ O, circle O r ∧ diameter K L O) 
  (h2 : lies_on_circle P O ∧ lies_on_circle Q O ∧ same_side P Q KL)
  (h3 : ∠PKL = 60 * π / 180) 
  (h4 : dist M P = 1 ∧ dist M Q = 1) : 
  r = 1 :=
sorry

end radius_of_circle_l404_404481


namespace coffee_serving_time_between_1_and_2_is_correct_l404_404233

theorem coffee_serving_time_between_1_and_2_is_correct
    (x : ℝ)
    (h_pos: 0 < x)
    (h_lt: x < 60) :
    30 + (x / 2) = 360 - (6 * x) → x = 660 / 13 :=
by
  sorry

end coffee_serving_time_between_1_and_2_is_correct_l404_404233


namespace cm_eq_cn_l404_404478

variable (A B C D M N : Point)
variable (AB CD BC AD : Line)
variable [InscribedQuadrilateral ABCD : IsInscribedQuad A B C D]
variable [M_is_intersection : IntersectionPoint A B CD M]
variable [N_is_intersection : IntersectionPoint B C AD N]
variable (BM DN : Length)
variable [BM_eq_DN : BM = DN]
variable (CM CN : Length)

theorem cm_eq_cn : CM = CN := by
  sorry

end cm_eq_cn_l404_404478


namespace binomial_12_10_l404_404586

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404586


namespace no_integer_solutions_l404_404790

theorem no_integer_solutions (x y z : ℤ) : x^3 + y^6 ≠ 7 * z + 3 :=
by sorry

end no_integer_solutions_l404_404790


namespace intersection_is_solution_l404_404730

theorem intersection_is_solution (a b : ℝ) :
  (b = 3 * a + 6 ∧ b = 2 * a - 4) ↔ (3 * a - b = -6 ∧ 2 * a - b = 4) := 
by sorry

end intersection_is_solution_l404_404730


namespace probability_of_two_black_balls_is_one_fifth_l404_404162

noncomputable def probability_of_two_black_balls (W B : Nat) : ℚ :=
  let total_balls := W + B
  let prob_black1 := (B : ℚ) / total_balls
  let prob_black2_given_black1 := (B - 1 : ℚ) / (total_balls - 1)
  prob_black1 * prob_black2_given_black1

theorem probability_of_two_black_balls_is_one_fifth : 
  probability_of_two_black_balls 8 7 = 1 / 5 := 
by
  sorry

end probability_of_two_black_balls_is_one_fifth_l404_404162


namespace relationship_among_a_b_c_l404_404287

theorem relationship_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (4 : ℝ) ^ (1 / 2))
  (hb : b = (2 : ℝ) ^ (1 / 3))
  (hc : c = (5 : ℝ) ^ (1 / 2))
: b < a ∧ a < c := 
sorry

end relationship_among_a_b_c_l404_404287


namespace sum_of_squares_of_roots_of_polynomial_eq_zero_l404_404958

theorem sum_of_squares_of_roots_of_polynomial_eq_zero :
  let p := Polynomial.X ^ 1010 + 22 * Polynomial.X ^ 1007 + 6 * Polynomial.X ^ 6 + 808 in
  let roots := p.roots in
  ∑ i in roots, i^2 = 0 :=
by
  let p := Polynomial.X ^ 1010 + 22 * Polynomial.X ^ 1007 + 6 * Polynomial.X ^ 6 + 808
  let roots := p.roots
  sorry

end sum_of_squares_of_roots_of_polynomial_eq_zero_l404_404958


namespace students_answered_both_correctly_l404_404896

theorem students_answered_both_correctly
  (enrolled : ℕ)
  (did_not_take_test : ℕ)
  (answered_q1_correctly : ℕ)
  (answered_q2_correctly : ℕ)
  (total_students_answered_both : ℕ) :
  enrolled = 29 →
  did_not_take_test = 5 →
  answered_q1_correctly = 19 →
  answered_q2_correctly = 24 →
  total_students_answered_both = 19 :=
by
  intros
  sorry

end students_answered_both_correctly_l404_404896


namespace pascal_30th_31st_numbers_l404_404360

-- Definitions based on conditions
def pascal_triangle_row_34 (k : ℕ) : ℕ := Nat.choose 34 k

-- Problem statement in Lean 4: proving the equations
theorem pascal_30th_31st_numbers :
  pascal_triangle_row_34 29 = 278256 ∧
  pascal_triangle_row_34 30 = 46376 :=
by
  sorry

end pascal_30th_31st_numbers_l404_404360


namespace min_x8_x9_x10_eq_618_l404_404785

theorem min_x8_x9_x10_eq_618 (x : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 10 → x i < x j)
  (h2 : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10 = 2023) :
  x 8 + x 9 + x 10 = 618 :=
sorry

end min_x8_x9_x10_eq_618_l404_404785


namespace part_1_part_2_part_3_l404_404692

noncomputable def f (x : ℝ) := x^3 - 3 * x

/-- Part (1) -/
theorem part_1 :
  ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x^3 + b * x^2 - 3 * x ∧
  f'(-1) = 0 ∧ f'(1) = 0 := sorry

/-- Part (2) -/
theorem part_2 (x₁ x₂ : ℝ) (h₁ : x₁ ∈ Icc (-1) 1) (h₂ : x₂ ∈ Icc (-1) 1) :
  |f x₁ - f x₂| ≤ 4 := sorry

/-- Part (3) -/
theorem part_3 (m : ℝ) (hm : m ≠ -2) (h : ∃ c1 c2 : ℝ, 
  f(1) = m ∧ f' c1 = (m - f(1))/(1 - c1) ∧ f' c2 = (m - f(1))/(1 - c2)) :
  -3 < m ∧ m < -2 := sorry

end part_1_part_2_part_3_l404_404692


namespace calories_in_300g_of_lemonade_l404_404642

def lemonJuiceWeight := 150 -- grams
def honeyWeight := 100 -- grams
def waterWeight := 250 -- grams
def lemonJuiceCalories := 30 -- calories in 150 grams
def honeyCaloriesPer100g := 304 -- calories in 100 grams

def totalLemonadeWeight := lemonJuiceWeight + honeyWeight + waterWeight
def totalLemonadeCalories := lemonJuiceCalories + (honeyCaloriesPer100g * (honeyWeight / 100))

theorem calories_in_300g_of_lemonade : 
  (totalLemonadeCalories / totalLemonadeWeight) * 300 = 200.4 := 
by
  sorry

end calories_in_300g_of_lemonade_l404_404642


namespace solve_congruence_l404_404645

open Nat

theorem solve_congruence (x : ℕ) (h : x^2 + x - 6 ≡ 0 [MOD 143]) : 
  x = 2 ∨ x = 41 ∨ x = 101 ∨ x = 140 :=
by
  sorry

end solve_congruence_l404_404645


namespace zarnin_staffing_l404_404231

open Finset

theorem zarnin_staffing :
  let total_resumes := 30
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let positions := 5
  suitable_resumes = 20 → 
  positions = 5 → 
  Nat.factorial suitable_resumes / Nat.factorial (suitable_resumes - positions) = 930240 := by
  intro total_resumes unsuitable_resumes suitable_resumes positions h1 h2
  have hs : suitable_resumes = 20 := h1
  have hp : positions = 5 := h2
  sorry

end zarnin_staffing_l404_404231


namespace binom_12_10_eq_66_l404_404600

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404600


namespace find_values_of_b_l404_404647

noncomputable def condition1 (b x y : ℝ) := 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0

noncomputable def condition2 (b x y h r : ℝ) := x^2 + y^2 + 1 > 2 * h * r + 2 * y + h - b^2

theorem find_values_of_b (b x y h r : ℝ) :
  (∀ (x y : ℝ), condition1 b x y) ∧ (∀ (x y : ℝ), condition2 b x y h r) ↔ 
    b ∈ Set.Ioo (-∞ : ℝ) (-(1 / 2)) ∨ Set.Ioo (-1 - Real.sqrt 2 / 4) (0 : ℝ) := 
sorry

end find_values_of_b_l404_404647


namespace find_n_l404_404012

noncomputable def C (n : ℕ) : ℝ :=
  352 * (1 - 1 / 2 ^ n) / (1 - 1 / 2)

noncomputable def D (n : ℕ) : ℝ :=
  992 * (1 - 1 / (-2) ^ n) / (1 + 1 / 2)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 1 := by
  sorry

end find_n_l404_404012


namespace mixed_number_solution_l404_404228

noncomputable def mixed_number_problem : Prop :=
  let a := 4 + 2 / 7
  let b := 5 + 1 / 2
  let c := 3 + 1 / 3
  let d := 2 + 1 / 6
  (a * b) - (c + d) = 18 + 1 / 14

theorem mixed_number_solution : mixed_number_problem := by 
  sorry

end mixed_number_solution_l404_404228


namespace orangutoads_eventually_unable_to_move_l404_404010

theorem orangutoads_eventually_unable_to_move (n : ℕ) (h : n > 1)
    (positions : Fin n → ℤ) :
    ∃ (turn : ℕ), ∃ (i : Fin n), ∀ j : Fin n, 
        (positions j = positions i → positions j + 1 ≠ positions (j + 1) + 1) :=
sorry

end orangutoads_eventually_unable_to_move_l404_404010


namespace area_triangle_CKB_l404_404420

variable (a b : ℝ)

theorem area_triangle_CKB (h : a > 0) (h2 : b > 0) : 
  let AB := Real.sqrt (a^2 + b^2) in
  let S_ABC := (1 / 2) * a * b in
  let similarity_ratio := a / AB in
  let S_CKB := (similarity_ratio^2) * S_ABC in
  S_CKB = (a^3 * b) / (2 * (a^2 + b^2)) := sorry

end area_triangle_CKB_l404_404420


namespace clean_house_time_l404_404942

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l404_404942


namespace base_k_sum_l404_404834

theorem base_k_sum (k : ℕ) (t : ℕ) (h1 : (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5)
    (h2 : t = (k + 3) + (k + 4) + (k + 7)) :
    t = 50 := sorry

end base_k_sum_l404_404834


namespace choose_one_book_1_choose_one_book_2_choose_one_book_3_l404_404543

theorem choose_one_book_1 : ∀ (lit_books math_books : ℕ), lit_books = 5 → math_books = 4 → lit_books + math_books = 9 :=
by
  intros lit_books math_books h1 h2
  rw [h1, h2]
  exact rfl

theorem choose_one_book_2 : ∀ (lit_books math_books : ℕ), lit_books = 1 → math_books = 4 → lit_books + math_books = 5 :=
by
  intros lit_books math_books h1 h2
  rw [h1, h2]
  exact rfl

theorem choose_one_book_3 : ∀ (lit_books math_books : ℕ), lit_books = 1 → math_books = 1 → lit_books + math_books = 2 :=
by
  intros lit_books math_books h1 h2
  rw [h1, h2]
  exact rfl

end choose_one_book_1_choose_one_book_2_choose_one_book_3_l404_404543


namespace range_of_f_l404_404840

-- Define the function f and its domain
def f (x : ℝ) := x^2 / (x^2 + 1)

-- Define the domain as a set
def domain : Set ℝ := {0, 1}

-- Define the expected range
def expected_range : Set ℝ := {0, 1 / 2}

-- State the theorem
theorem range_of_f :
  ∀ (y : ℝ), (∃ (x : ℝ) (hx : x ∈ domain), f x = y) ↔ y ∈ expected_range :=
sorry

end range_of_f_l404_404840


namespace roots_difference_l404_404578

theorem roots_difference {a b c : ℝ} (h_eq : 2 * a^2 + 5 * a - 12 = 0) :
  let r1 := (-5 + Real.sqrt (5^2 - 4 * 2 * (-12))) / (2 * 2),
      r2 := (-5 - Real.sqrt (5^2 - 4 * 2 * (-12))) / (2 * 2)
  in r1 - r2 = 5.5 :=
by
  -- The proof will go here
  sorry

end roots_difference_l404_404578


namespace min_distance_midpoint_to_moving_point_l404_404356

theorem min_distance_midpoint_to_moving_point :
  ∀ (x1 y1 x2 y2 xn yn : ℝ),
    (x1 + y1 = 7) →
    (x2 + y2 = 5) →
    (xn^2 + yn^2 = 8) →
    ∃ (xm ym : ℝ),
    (xm + ym = 6) ∧
    (∃ m n : ℝ, m = (xn - xm)^2 + (yn - ym)^2 ∧ m = 2) :=
by 
  intros x1 y1 x2 y2 xn yn h1 h2 h3
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  have h4 : xm + ym = 6 := 
    calc 
      xm + ym = (x1 + x2)/2 + (y1 + y2)/2 : by ring 
            ... = (x1 + y1 + x2 + y2) / 2 : by ring 
            ... = (7 + 5) / 2 : by rw [h1, h2]
            ... = 6 : by norm_num
  have distance_formula : (xn - xm)^2 + (yn - ym)^2 = (sqrt ((xm - 0)^2 + (ym - 0)^2) - sqrt 8)^2 :=
    sorry
  exact ⟨xm, ym, h4, ⟨2⟩⟩


end min_distance_midpoint_to_moving_point_l404_404356


namespace digits_count_of_special_numbers_l404_404499

theorem digits_count_of_special_numbers
  (n : ℕ)
  (h1 : 8^n = 28672) : n = 5 := 
by
  sorry

end digits_count_of_special_numbers_l404_404499


namespace binom_12_10_eq_66_l404_404593

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404593


namespace minimum_value_abs_a_plus_2_abs_b_l404_404022

open Real

theorem minimum_value_abs_a_plus_2_abs_b 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (x₁ x₂ x₃ : ℝ)
  (f_def : ∀ x, f x = x^3 + a*x^2 + b*x)
  (roots_cond : x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1)
  (equal_values : f x₁ = f x₂ ∧ f x₂ = f x₃) :
  ∃ minimum, minimum = (sqrt 3) ∧ (∀ (a b : ℝ), |a| + 2*|b| ≥ sqrt 3) :=
by
  sorry

end minimum_value_abs_a_plus_2_abs_b_l404_404022


namespace solve_integral_equation_l404_404899

noncomputable def K (x t : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ t then (Real.sinh x * Real.sinh (t - 1)) / Real.sinh 1 
else if t ≤ x ∧ x ≤ 1 then (Real.sinh t * Real.sinh (x - 1)) / Real.sinh 1 
else 0

theorem solve_integral_equation (λ : ℝ) (φ : ℝ → ℝ) :
  (∀ x, φ x - λ * ∫ t in 0..1, K x t * φ t = Real.exp x) →
  (φ = 
    if λ = -1 then
      λ x, (Real.exp 1 - 1) * x + 1
    else if λ > -1 ∧ λ ≠ 0 then
      λ x, Real.cosh (Real.sqrt (λ + 1) * x) + (Real.exp 1 - Real.cosh (Real.sqrt (λ + 1))) / Real.sinh (Real.sqrt (λ + 1)) * Real.sinh (Real.sqrt (λ + 1) * x)
    else
      λ x, Real.cos (Real.sqrt (-λ - 1) * x) + (Real.exp 1 - Real.cos (Real.sqrt (-λ - 1))) / Real.sin (Real.sqrt (-λ - 1)) * Real.sin (Real.sqrt (-λ - 1) * x)
  )
:=
sorry

end solve_integral_equation_l404_404899


namespace find_circle_C_find_line_l_l404_404903

-- Problem 1
def center_of_circle_C : (ℝ × ℝ) :=
  let x := -1
  let y := 0
  (x, y)

def circle_C_tangent_line : (ℝ × ℝ × ℝ) :=
  let a := 1
  let b := 1
  let c := 3
  (a, b, c)

def circle_equation (C : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ (x y : ℝ), (x + C.1)^2 + (y + C.2)^2 = r^2

theorem find_circle_C : 
  let C := center_of_circle_C in
  let r := sqrt(2) in
  circle_equation C r := 
  by 
    sorry

-- Problem 2
def given_circle : (ℝ × ℝ × ℝ) :=
  let h := 0
  let k := 3
  let r := 2
  (h, k, r)

def point_A : (ℝ × ℝ) :=
  let x := -1
  let y := 0
  (x, y)

def line_passes_point (l : ℝ × ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  l.1 * A.1 + l.2 * A.2 + l.3 = 0

def line_intersects_circle (l : ℝ × ℝ × ℝ) (circle : ℝ × ℝ × ℝ) (d : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ≠ (x₂, y₂) ∧ 
    (x₁ + circle.1)^2 + (y₁ + circle.2)^2 = circle.3 ∧ 
    (x₂ + circle.1)^2 + (y₂ + circle.2)^2 = circle.3 ∧ 
    sqrt((x₁ - x₂)^2 + (y₁ - y₂)^2)= d

theorem find_line_l : 
  let l1 := (1, 0, 1) -- x = -1
  let l2 := (4, -3, 4) -- 4x - 3y + 4 = 0
  ∃ l : ℝ × ℝ × ℝ, 
  (line_passes_point l point_A ∧ line_intersects_circle l given_circle (2 * sqrt(3))) := 
  by 
    sorry

end find_circle_C_find_line_l_l404_404903


namespace smallest_n_l404_404461

open Nat

theorem smallest_n (m n : Nat) (h1 : RelativelyPrime m n) (h2 : m < n)
  (h3 : 741 * n ≤ 1000 * m) (h4 : 1000 * m < 742 * n) : n = 999 :=
sorry

end smallest_n_l404_404461


namespace coefficient_of_x6_in_q_squared_l404_404409

-- Define the polynomial q(x)
noncomputable def q (x : ℝ) : ℝ := x^4 - 5 * x^2 - 4 * x + 3

-- Define the expansion of (q(x))^2 and find the coefficient of x^6
theorem coefficient_of_x6_in_q_squared :
  (λ (x : ℝ), (q x)^2).coeff 6 = -10 :=
sorry

end coefficient_of_x6_in_q_squared_l404_404409


namespace cleaning_time_if_anne_doubled_l404_404953

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l404_404953


namespace minimal_cuts_l404_404214

-- Define the problem conditions
def is_valid_cut (net : list (list bool)) (cut : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), cut (i, j) → 
    (i = 0 ∨ i = net.length - 1 ∨ j = 0 ∨ j = net.head.length - 1) ∧ ¬net.getD i []!.getD j false

-- Formalize the question and answer using the conditions
theorem minimal_cuts (net : list (list bool)) (cut : ℕ × ℕ → Prop) :
  (∃ (cuts: ℕ), (∀ (p: ℕ × ℕ), cut p → is_valid_cut net cut) ∧ cuts = 8) :=
sorry

end minimal_cuts_l404_404214


namespace find_rate_percent_l404_404148

theorem find_rate_percent
  (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) 
  (hP : P = 1600)
  (hSI : SI = 200)
  (hT : T = 4)
  (hSI_eq : SI = (P * R * T) / 100) :
  R = 3.125 :=
by {
  sorry
}

end find_rate_percent_l404_404148


namespace total_length_of_segments_in_figure_2_l404_404202

theorem total_length_of_segments_in_figure_2 : 
  (length_left_vertical_segment : ℕ) 
  (length_top_horizontal_segment : ℕ) 
  (h_left : length_left_vertical_segment = 8)
  (h_top : length_top_horizontal_segment = 4) : 
  length_left_vertical_segment + length_top_horizontal_segment = 12 := 
by
  sorry

end total_length_of_segments_in_figure_2_l404_404202


namespace fraction_value_l404_404727

theorem fraction_value (p q x : ℚ) (h₁ : p / q = 4 / 5) (h₂ : 2 * q + p ≠ 0) (h₃ : 2 * q - p ≠ 0) :
  x + (2 * q - p) / (2 * q + p) = 2 → x = 11 / 7 :=
by
  sorry

end fraction_value_l404_404727


namespace brianne_yard_length_l404_404631

theorem brianne_yard_length 
  (derrick_yard_length : ℝ)
  (h₁ : derrick_yard_length = 10)
  (alex_yard_length : ℝ)
  (h₂ : alex_yard_length = derrick_yard_length / 2)
  (brianne_yard_length : ℝ)
  (h₃ : brianne_yard_length = 6 * alex_yard_length) :
  brianne_yard_length = 30 :=
by sorry

end brianne_yard_length_l404_404631


namespace sqrt_product_simplification_l404_404580

noncomputable def sqrt_product (a : ℝ) : ℝ :=
  real.sqrt (50 * a^3) * real.sqrt (18 * a^2) * real.sqrt (98 * a^5)

def simplified_form (a : ℝ) : ℝ :=
  42 * a^5 * real.sqrt 10

theorem sqrt_product_simplification (a : ℝ) : sqrt_product a = simplified_form a := by
  sorry

end sqrt_product_simplification_l404_404580


namespace clean_house_time_l404_404944

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l404_404944


namespace race_problem_l404_404736

noncomputable def race_time (distance : ℝ) (start : ℝ) (speed : ℝ) : ℝ :=
  (distance - start) / speed

theorem race_problem 
  (x : ℕ) 
  (distance : ℝ := 500) 
  (start_A : ℝ := 140) 
  (speed_A : ℝ := 3 * x)
  (start_B : ℝ := 0)
  (speed_B : ℝ := 4 * x)
  (start_C : ℝ := 60)
  (speed_C : ℝ := 5 * x)
  (start_D : ℝ := 20)
  (speed_D : ℝ := 6 * x) :
  race_time distance start_A speed_A = 120 / x ∧
  race_time distance start_B speed_B = 125 / x ∧
  race_time distance start_A speed_A - race_time distance start_B speed_B = -5 / x := 
begin
  sorry
end

end race_problem_l404_404736


namespace find_a_even_function_l404_404094

theorem find_a_even_function (a : ℝ) :
  (∀ x : ℝ, (x ^ 2 + a * x - 4) = ((-x) ^ 2 + a * (-x) - 4)) → a = 0 :=
by
  intro h
  sorry

end find_a_even_function_l404_404094


namespace probability_of_3_black_face_cards_l404_404653

-- Definitions based on conditions
def total_cards : ℕ := 36
def total_black_face_cards : ℕ := 8
def total_other_cards : ℕ := total_cards - total_black_face_cards
def draw_cards : ℕ := 6
def draw_black_face_cards : ℕ := 3
def draw_other_cards := draw_cards - draw_black_face_cards

-- Calculation using combinations
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_combinations : ℕ := combination total_cards draw_cards
noncomputable def favorable_combinations : ℕ := combination total_black_face_cards draw_black_face_cards * combination total_other_cards draw_other_cards

-- Calculating probability
noncomputable def probability : ℚ := favorable_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_3_black_face_cards : probability = 11466 / 121737 := by
  -- proof
  sorry

end probability_of_3_black_face_cards_l404_404653


namespace Sn_2014_eq_1006_5_l404_404322

noncomputable def f (x : ℝ) : ℝ := 1 / 2 + Real.log (x / (1 - x)) / Real.log 2

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n - 1), f (i / n)

theorem Sn_2014_eq_1006_5 : S 2014 = 2013 / 2 := by
  sorry

end Sn_2014_eq_1006_5_l404_404322


namespace packs_of_buns_needed_l404_404204

def friends := 10
def non_meat_eater := 1
def non_bun_eater := 1
def burgers_per_guest := 3
def buns_per_pack := 8

theorem packs_of_buns_needed : (3 * (friends - non_meat_eater) - burgers_per_guest) / buns_per_pack = 3 := 
by
suffices h : 24 / 8 = 3 by exact h
{
  have h_burgers : 3 * (friends - non_meat_eater) = 27 := by norm_num,
  have h_buns_needed : 27 - 3 = 24 := by norm_num,
  assumption
}

end packs_of_buns_needed_l404_404204


namespace rick_ironed_27_pieces_l404_404440

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l404_404440


namespace find_vector_b_l404_404781

-- Definitions and conditions as identified
def vector_a : ℝ × ℝ × ℝ := (3, 2, 4)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def vector_b : ℝ × ℝ × ℝ := (-12/5, -3/5, 4/5)

-- Lean statement to prove
theorem find_vector_b :
  dot_product vector_a vector_b = 14 ∧
  cross_product vector_a vector_b = (4, -12, 6) :=
sorry

end find_vector_b_l404_404781


namespace find_x_set_l404_404668

noncomputable def f (x : ℝ) := Real.tan (2 * x + Real.pi / 4)

theorem find_x_set :
  { x : ℝ | f x ≥ Real.sqrt 3 } = 
    { x : ℝ | ∃ k : ℤ, x ∈ set.Ico 
      (Real.pi / 24 + k * Real.pi / 2) 
      (Real.pi / 8 + k * Real.pi / 2) } :=
by sorry

end find_x_set_l404_404668


namespace sequence_general_term_l404_404300

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end sequence_general_term_l404_404300


namespace rachel_tokens_unique_sequences_l404_404825

theorem rachel_tokens_unique_sequences :
  let grid_size := 6,
      initial_positions := (1, 2),
      final_positions := (5, 6),
      move_rule := "A token can move one square to the right if the square is free and can leapfrog another token to a free square two spots to the right if the adjacent square is occupied"
  in
  number_of_sequences grid_size initial_positions final_positions move_rule = 42 := by
  sorry

end rachel_tokens_unique_sequences_l404_404825


namespace triangle_proof_l404_404734

noncomputable def triangle_properties (A B C a b c : ℝ) (h1 : 2 * c = √3 * a + 2 * b * cos A) : Prop :=
  -- Prove (1) Angle B
  B = π / 6 ∧
  -- Prove (2) Side b given c = 1 and area = √3/2
  (c = 1 → (1 / 2) * a * c * sin B = √3 / 2 → b = √7)

-- The statement:
theorem triangle_proof (A B C a b c : ℝ) (h1 : 2 * c = √3 * a + 2 * b * cos A)
  (h2 : c = 1) (h3 : (1 / 2) * a * c * sin B = √3 / 2) : triangle_properties A B C a b c h1 :=
by
  sorry

end triangle_proof_l404_404734


namespace equal_lengths_l404_404927

variables {Point : Type} [NonCoplanar Point]
variables (A B C D : Point)

-- Angles are equal
axiom angle_ABC_eq_ADC : angle A B C = angle A D C
axiom angle_BAD_eq_BCD : angle B A D = angle B C D

theorem equal_lengths 
  (h1 : NonCoplanar A B C D)
  (h2 : angle_ABC_eq_ADC)
  (h3 : angle_BAD_eq_BCD) :
  distance A B = distance C D ∧ distance B C = distance A D :=
sorry

end equal_lengths_l404_404927


namespace sin_value_l404_404289

variables {α : ℝ}

-- Condition given in the problem
def condition : Prop := cos (α + π / 4) = 2 / 3

-- The proof goal
theorem sin_value (h : condition) : sin (α - 5 * π / 4) = 2 / 3 :=
sorry

end sin_value_l404_404289


namespace remainder_div_x_plus_2_l404_404141

def f (x : ℤ) : ℤ := x^15 + 3

theorem remainder_div_x_plus_2 : f (-2) = -32765 := by
  sorry

end remainder_div_x_plus_2_l404_404141


namespace no_such_functions_exist_l404_404252

theorem no_such_functions_exist :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 :=
by
  sorry

end no_such_functions_exist_l404_404252


namespace find_a_l404_404274

variable (a x y : ℝ)

theorem find_a (h1 : x / (2 * y) = 3 / 2) (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) : a = 7 :=
sorry

end find_a_l404_404274


namespace total_amount_highest_spenders_Akeno_spent_less_l404_404929

noncomputable def Akeno := 2985
noncomputable def Lev := (1/3 : ℝ) * Akeno
noncomputable def Ambrocio := Lev - 177
noncomputable def Natasha := Akeno + 0.25 * Akeno
noncomputable def Jack := 2 * Lev
noncomputable def Hiroshi := (120000 : ℝ) / 110

theorem total_amount_highest_spenders :
  Natasha + Akeno + Jack = 8706.25 :=
by
  sorry

theorem Akeno_spent_less : 
  Akeno - (8706.25 - Akeno) = -2736.25 :=
by
  sorry

end total_amount_highest_spenders_Akeno_spent_less_l404_404929


namespace tangent_line_l404_404089

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 3 * x + 1

def p : ℝ × ℝ := (0, 1)

theorem tangent_line :
  ∃ (a b c : ℝ), (λ y x, a * x + b * y + c = 0) = (λ y x, 2 * x + y - 1 = 0) →
  tangent_line_to_curve_at_point f p a b c :=
sorry

end tangent_line_l404_404089


namespace alice_sum_l404_404207

def a : ℂ := 3 + 2 * complex.i
def b : ℂ := -1 + 5 * complex.i
def c : ℂ := 4 - 3 * complex.i
def conj_c : ℂ := complex.conj c

theorem alice_sum : a + b + conj_c = 6 + 10 * complex.i :=
by sorry

end alice_sum_l404_404207


namespace fox_catches_rabbit_after_40_steps_l404_404463

-- Define the speeds based on the conditions provided
def speed_dog := 0.4
def speed_fox := 0.5
def speed_rabbit := 1.2

-- Define the steps covered by each in given time frames
def steps_rabbit_in_t (t : ℝ) := 12 * (t / 1.0)
def steps_dog_in_t (t : ℝ) := 3 * (t / 1.0)
def steps_fox_in_t (t : ℝ) := 4 * (t / 1.0)

-- The problem to prove
theorem fox_catches_rabbit_after_40_steps 
  (distance_dog_fox distance_dog_rabbit distance_fox_rabbit : ℝ) 
  (time_dog_catch_fox : ℝ) :
  ∃ (t : ℝ), (speed_dog * t = distance_dog_fox ∧
              speed_fox * t = distance_dog_fox + distance_fox_rabbit) ∧
              steps_rabbit_in_t t - 10 = 40 :=
by
  sorry

end fox_catches_rabbit_after_40_steps_l404_404463


namespace students_average_age_l404_404077

theorem students_average_age (A : ℝ) (students_count teacher_age total_average new_count : ℝ) 
  (h1 : students_count = 30)
  (h2 : teacher_age = 45)
  (h3 : new_count = students_count + 1)
  (h4 : total_average = 15) 
  (h5 : total_average = (A * students_count + teacher_age) / new_count) : 
  A = 14 :=
by
  sorry

end students_average_age_l404_404077


namespace complex_number_in_third_quadrant_l404_404375

theorem complex_number_in_third_quadrant (i : ℂ) (hi : i^2 = -1) : 
  let z := i * (i - 1) in z.re < 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_in_third_quadrant_l404_404375


namespace acronym_XYZ_length_l404_404212

theorem acronym_XYZ_length :
  let X_length := 2 * Real.sqrt 2
  let Y_length := 1 + 2 * Real.sqrt 2
  let Z_length := 4 + Real.sqrt 5
  X_length + Y_length + Z_length = 5 + 4 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end acronym_XYZ_length_l404_404212


namespace binomial_12_10_l404_404588

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404588


namespace sum_of_fractions_l404_404684

theorem sum_of_fractions (p q : ℕ) (h1: 0 < p) (h2: p < q) (h3: q ≤ 2017) (h4: p + q > 2017) (h5: Nat.gcd p q = 1) :
  (∑ _ in {pq | ∃ (p q : ℕ), 0 < p ∧ p < q ∧ q ≤ 2017 ∧ p + q > 2017 ∧ Nat.gcd p q = 1}, (1 / (p * q) : ℚ)) = 1 / 2 :=
sorry

end sum_of_fractions_l404_404684


namespace binomial_coefficient_12_10_l404_404608

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404608


namespace percentage_reduction_is_correct_l404_404076

-- Definitions and initial conditions
def initial_price_per_model := 100
def models_for_kindergarten := 2
def models_for_elementary := 2 * models_for_kindergarten
def total_models := models_for_kindergarten + models_for_elementary
def total_cost_without_reduction := total_models * initial_price_per_model
def total_cost_paid := 570

-- Goal statement in Lean 4
theorem percentage_reduction_is_correct :
  (total_models > 5) →
  total_cost_paid = 570 →
  models_for_kindergarten = 2 →
  (total_cost_without_reduction - total_cost_paid) / total_models / initial_price_per_model * 100 = 5 :=
by
  -- sorry to skip the proof
  sorry

end percentage_reduction_is_correct_l404_404076


namespace distance_between_planes_l404_404979

theorem distance_between_planes :
  let plane1 := { p : ℝ × ℝ × ℝ | 3 * p.1 + p.2 - p.3 = 3 }
  let plane2 := { p : ℝ × ℝ × ℝ | 3 * (2 * p.1) + (2 * p.2) - (2 * p.3) = -2 }
  ∀ p : ℝ × ℝ × ℝ, (p ∈ plane1 → distance_from_point_to_plane p plane2 = 5 * real.sqrt 11 / 11) :=
begin
  -- sorry this will contain our proof
  sorry
end

end distance_between_planes_l404_404979


namespace jury_seating_arrangements_l404_404861

theorem jury_seating_arrangements (n : ℕ) (NN : n = 12)
  (next_clockwise : ∀ i, 0 ≤ i < n → (i + 1) % n = next_clockwise i)
  (seat_taken : ∀ i, 0 ≤ i < n → seat_taken i = false)
  (sit : ∀ j, 0 ≤ j < n → sit j = if seat_taken (next_clockwise j) then 
     (fun k, 0 ≤ k < n ∧ !(seat_taken k) → sit j = k) else sit j = next_clockwise j)
  : (∑ k : ℕ in range 11, (choose 10 k)) = 1024 := sorry

end jury_seating_arrangements_l404_404861


namespace sequence_nth_term_and_sum_l404_404997

noncomputable def a_n (n : ℕ) : ℝ := 1.5 * n^2 + 0.5 * n + 1

noncomputable def S_n (n : ℕ) : ℝ := 0.25 * n * (n + 1) * (2 * n + 7)

theorem sequence_nth_term_and_sum (n : ℕ) :
  (a_n n = 1.5 * n^2 + 0.5 * n + 1) ∧
  (S_n n = ∑ k in (Finset.range n).filter (λ k, true), a_n k) :=
sorry

end sequence_nth_term_and_sum_l404_404997


namespace max_quartets_5x5_max_quartets_mxn_l404_404029

open Nat

/-- Maximum number of quartets in a 5x5 grid is 5 -/
theorem max_quartets_5x5 : ∀ Q, (∃ Q, (4 * Q ≤ 25 ∧ 5 * 4 = 20 ∧ Q = 5)) ↔ Q = 5 :=
by
  sorry

/-- Maximum number of quartets in an mxn grid -/
theorem max_quartets_mxn (m n : ℕ) : ∀ Q, 
  ((∃ Q, (m % 2 = 0 ∧ n % 2 = 0 ∧ Q = m * n / 4) ∨
  (m % 2 = 0 ∧ n % 2 = 1 ∧ Q = m * (n - 1) / 4) ∨
  (m % 2 = 1 ∧ n % 2 = 1 ∧ Q = (m * (n - 1) - 2) / 4))) :=
by
  sorry

end max_quartets_5x5_max_quartets_mxn_l404_404029


namespace sin_angles_product_eq_one_eighth_l404_404968

theorem sin_angles_product_eq_one_eighth :
  sin (10 * π / 180) * sin (50 * π / 180) * sin (70 * π / 180) * sin (80 * π / 180) = 1 / 8 := by
  sorry

end sin_angles_product_eq_one_eighth_l404_404968


namespace recycled_bottles_l404_404275

theorem recycled_bottles (initial_bottles recycling_rate : ℕ) (h_initial: initial_bottles = 3125) (h_rate: recycling_rate = 5) 
: ∑ i in finset.range (nat.log recycling_rate initial_bottles + 1), (initial_bottles / recycling_rate ^ (i + 1)) = 781 :=
by {
 sorry
}

end recycled_bottles_l404_404275


namespace count_elements_with_digit_1_l404_404014

-- Define the set S
def S := {k | (∃ k : ℕ, 0 ≤ k ∧ k ≤ 1500)}

-- The main theorem statement
theorem count_elements_with_digit_1 : 
  set.count (λ k, ∃ m : ℤ, m ≤ real.log10 (3^k) ∧ real.log10 (3^k) < m + real.log10 2) S = 716 :=
sorry

end count_elements_with_digit_1_l404_404014


namespace least_number_of_table_entries_l404_404382

-- Given conditions
def num_towns : ℕ := 6

-- Theorem statement
theorem least_number_of_table_entries : (num_towns * (num_towns - 1)) / 2 = 15 := by
  -- Proof goes here.
  sorry

end least_number_of_table_entries_l404_404382


namespace perimeter_of_quadrilateral_l404_404238

theorem perimeter_of_quadrilateral (a : ℝ) (ha : a ≠ 0) :
  let p := (14 + 2 * Real.sqrt 10) / 3 in
  let quad_perimeter := (Dist.mk (a, -a) (a, a/3) + 2*a + Dist.mk (-a, -a/3) (-a, a) + 
                         Real.sqrt ((2*a) ^ 2 + (2*a / 3) ^ 2)) in
  (quad_perimeter / a) = p :=
begin
  sorry
end

end perimeter_of_quadrilateral_l404_404238


namespace optimal_start_time_for_max_distance_l404_404036

-- Define the velocity function before and after 16 hours
def velocity_before (t : ℝ) (m : ℝ) : ℝ := (m / 16) * t
def velocity_after (t : ℝ) (m : ℝ) : ℝ := m - (m / 8) * (t - 16)

-- Define the total distance function, integrating for maximum distance
noncomputable def total_distance (S : ℝ) (m : ℝ) : ℝ :=
  let x := 16 - S in
  let ASM := (x / 2) * m * (2 - (x / 16)) in
  let CMTB := ((8 - x) / 2) * m * (1 + (x / 8)) in
  ASM + CMTB

-- Theorem to prove the optimal starting time
theorem optimal_start_time_for_max_distance (m : ℝ) :
  (10 + 40/60 : ℝ) = S →
  S = 10 + 40/60 :=
by
  sorry

end optimal_start_time_for_max_distance_l404_404036


namespace parabola_equation_l404_404317

theorem parabola_equation (h_vertex : (0, 0) = (0, 0)) (h_symmetry : ∀ y : ℝ, ∃ x : ℝ, x = -2) :
  ∃ p > 0, (λ (x y : ℝ), y^2 = 2 * p * x) (8, 0) :=
by
  use 4
  split
  exact zero_lt_four
  simp
  sorry

end parabola_equation_l404_404317


namespace fundamental_events_in_A_expected_value_ξ_l404_404779

noncomputable def solution_set := {x : ℝ | x^2 - x - 6 ≤ 0}

theorem fundamental_events_in_A :
  {p : ℤ × ℤ | p.1 ∈ {m : ℤ | -2 ≤ m ∧ m ≤ 3} ∧ p.2 ∈ {n : ℤ | -2 ≤ n ∧ n ≤ 3} ∧ p.1 + p.2 = 0} =
  {(-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2)} := sorry

noncomputable def ξ_distribution := 
  {0, 1, 4, 9} →ᵣ (λ x : ℝ, 
    if x = 0 then (1 / 6 : ℝ)
    else if x = 1 then (1 / 3 : ℝ)
    else if x = 4 then (1 / 3 : ℝ)
    else if x = 9 then (1 / 6 : ℝ)
    else 0)

theorem expected_value_ξ : 
  ∑ x in {0, 1, 4, 9}, x * ξ_distribution x = (19 / 6 : ℝ) := sorry

end fundamental_events_in_A_expected_value_ξ_l404_404779


namespace sum_angles_lt_sum_plane_angles_sum_angles_gt_half_sum_plane_angles_l404_404824

-- Define a structure for a trihedral angle with vertex and edges
structure TrihedralAngle :=
(vertex : Point)
(edge1 edge2 edge3 : Point)
(plane_angle1 plane_angle2 plane_angle3 : ℝ) -- θ1, θ2, θ3
(angle1 angle2 angle3 : ℝ) -- α, β, γ

-- Condition: all plane angles are acute
def acute_plane_angles (ta : TrihedralAngle) : Prop :=
ta.plane_angle1 < π / 2 ∧ ta.plane_angle2 < π / 2 ∧ ta.plane_angle3 < π / 2

-- First part of the proof
theorem sum_angles_lt_sum_plane_angles (ta : TrihedralAngle) :
    ta.angle1 + ta.angle2 + ta.angle3 < 
    ta.plane_angle1 + ta.plane_angle2 + ta.plane_angle3 := 
begin
  sorry
end

-- Second part of the proof for acute plane angles
theorem sum_angles_gt_half_sum_plane_angles (ta : TrihedralAngle) (h : acute_plane_angles ta) :
    ta.angle1 + ta.angle2 + ta.angle3 > 
    (ta.plane_angle1 + ta.plane_angle2 + ta.plane_angle3) / 2 :=
begin
  sorry
end

end sum_angles_lt_sum_plane_angles_sum_angles_gt_half_sum_plane_angles_l404_404824


namespace number_of_2_dollar_socks_l404_404828

-- Given conditions
def total_pairs (a b c : ℕ) := a + b + c = 15
def total_cost (a b c : ℕ) := 2 * a + 4 * b + 5 * c = 41
def min_each_pair (a b c : ℕ) := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- To be proved
theorem number_of_2_dollar_socks (a b c : ℕ) (h1 : total_pairs a b c) (h2 : total_cost a b c) (h3 : min_each_pair a b c) : 
  a = 11 := 
  sorry

end number_of_2_dollar_socks_l404_404828


namespace sequence_integer_k_value_l404_404332

open Nat Int

theorem sequence_integer_k_value (k m : ℕ) (h_k_pos : 0 < k)
    (h_m_pos : 0 < m) (h_gcd : gcd k m = 1) :
    (∀t : ℤ, k = t * m - 1) → 
    (∀n : ℕ, a : ℕ → ℕ,
    (a 1 = 1) → (a 2 = 1) → (a 3 = m) →
    (∀ n ≥ 3, a (n + 1) = (k + a n * a (n - 1)) / a (n - 2)) →
    (∀ n : ℕ, n ≥ 1 → (∀ t : ℤ, k = t * m - 1) → ∃ a_n : ℕ, a n = a_n)) :=
by
  intros t h_k_eq a h_a1 h_a2 h_a3 h_recurrence n h_n h_k_eq
  sorry

end sequence_integer_k_value_l404_404332


namespace xyz_range_l404_404052

theorem xyz_range (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5 / 27 := 
sorry

end xyz_range_l404_404052


namespace cleaning_time_if_anne_doubled_l404_404955

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l404_404955


namespace John_took_more_chickens_than_Ray_l404_404070

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l404_404070


namespace daily_rate_proof_l404_404907

theorem daily_rate_proof :
  ∃ x : ℝ, (∀ distance : ℝ, (distance = 150.0) →
    (x + 0.19 * distance = 18.95 + 0.21 * distance)) →
    x = 21.95 :=
by
  intro h
  use 21.95
  intro distance hd
  rw hd
  sorry

end daily_rate_proof_l404_404907


namespace part1_part2_l404_404307

-- Part 1
theorem part1 (a : ℝ) : (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → (a ∈ Iic 0 ∪ Ici 4) :=
sorry

-- Part 2
theorem part2 (m : ℝ) (a : ℝ) : m > 0 → 
  ((∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) ∧ (∀ x ∈ Ioi (-1 : ℝ), x - a + m / (x + 1) ≥ 0)) → 
  (m ∈ Ioo 0 (1/4)) :=
sorry

end part1_part2_l404_404307


namespace partition_nats_100_subsets_l404_404823

theorem partition_nats_100_subsets :
  ∃ (S : ℕ → ℕ), (∀ n, 1 ≤ S n ∧ S n ≤ 100) ∧
    (∀ a b c : ℕ, a + 99 * b = c → S a = S c ∨ S a = S b ∨ S b = S c) :=
by
  sorry

end partition_nats_100_subsets_l404_404823


namespace kite_area_l404_404433

theorem kite_area (EF GH : ℝ) (FG EH : ℕ) (h1 : FG * FG + EH * EH = 25) : EF * GH = 12 :=
by
  sorry

end kite_area_l404_404433


namespace equation_of_l1_l404_404901

theorem equation_of_l1 (A B : ℝ × ℝ)
   (A_eq : A = (1, 1))
   (B_eq : B = (0, -1))
   (parallel : ∀ l1 l2 : ℝ → ℝ → Prop, parallel l1 l2) 
   (maximize_distance : ∀ d : ℝ, maximize_distance d) :
   ∃ (l1 : ℝ → ℝ → Prop), l1 x y ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end equation_of_l1_l404_404901


namespace permutation_exists_l404_404582

theorem permutation_exists (n : ℕ) (h_even : n % 2 = 0) : 
  ∃ g : ℕ → ℕ, function.bijective g ∧ (∀ i, (g i + g (i + 1) % n) % 4 = 0 ∨ (g i + g (i + 1) % n) % 7 = 0) :=
sorry

end permutation_exists_l404_404582


namespace cube_root_of_neg_125_l404_404839

theorem cube_root_of_neg_125 : (-5)^3 = -125 := 
by sorry

end cube_root_of_neg_125_l404_404839


namespace part1_AD_part2_perimeter_l404_404758

noncomputable def length_AD (b c : ℝ) (angle_BAC : ℝ) : ℝ :=
  if h : b = 1 ∧ c = 2 ∧ angle_BAC = 60 then
    2 * Math.sqrt 3 / 3
  else
    0

theorem part1_AD (b c : ℝ) (angle_BAC : ℝ) (h : b = 1 ∧ c = 2 ∧ angle_BAC = 60) : 
  length_AD b c angle_BAC = 2 * Math.sqrt 3 / 3 := sorry

noncomputable def perimeter_range (a b c : ℝ) (angle_B : ℝ) (angle_C : ℝ) : set ℝ :=
  if h : (∠ABC).right ∧ ∠ABC.left ∧ ∠ACB.right ∧ ∠ACB.left ∧ 
          a*2 / b = 1 + Math.tan angle_C / Math.tan angle_B then
    {x | 2 + 2 * Math.sqrt 3 < x ∧ x ≤ 6 }
  else
    ∅

theorem part2_perimeter (a b c : ℝ) (angle_B angle_C : ℝ) 
  (h : (∠ABC).right ∧ ∠ABC.left ∧ ∠ACB.right ∧ ∠ACB.left ∧ 
       a*2 / b = 1 + Math.tan angle_C / Math.tan angle_B) : 
  ∃ x ∈ perimeter_range a b c angle_B angle_C, true := 
  sorry

end part1_AD_part2_perimeter_l404_404758


namespace sum_of_coefficients_l404_404106

theorem sum_of_coefficients (
  x y : ℝ
) :
  let expr := (1*x - 3*y) ^ 5
  let terms_without_x := expr.coeffs.filter (λ t, ¬t.contains x)
  terms_without_x.sum = -32 := 
sorry

end sum_of_coefficients_l404_404106


namespace find_uncommon_cards_l404_404509

def numRare : ℕ := 19
def numCommon : ℕ := 30
def costRare : ℝ := 1
def costUncommon : ℝ := 0.50
def costCommon : ℝ := 0.25
def totalCostDeck : ℝ := 32

theorem find_uncommon_cards (U : ℕ) (h : U * costUncommon + numRare * costRare + numCommon * costCommon = totalCostDeck) : U = 11 := by
  sorry

end find_uncommon_cards_l404_404509


namespace range_of_f_l404_404877

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l404_404877


namespace equation_solution_1_equation_solution_2_equation_solution_3_l404_404448

def system_of_equations (x y : ℝ) : Prop :=
  (x * (x^2 - 3 * y^2) = 16) ∧ (y * (3 * x^2 - y^2) = 88)

theorem equation_solution_1 :
  system_of_equations 4 2 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_2 :
  system_of_equations (-3.7) 2.5 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_3 :
  system_of_equations (-0.3) (-4.5) :=
by
  -- The proof is skipped.
  sorry

end equation_solution_1_equation_solution_2_equation_solution_3_l404_404448


namespace clean_house_time_l404_404941

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l404_404941


namespace cistern_fill_time_l404_404550

/--
  A cistern can be filled by tap A in 4 hours,
  emptied by tap B in 6 hours,
  and filled by tap C in 3 hours.
  If all the taps are opened simultaneously,
  then the cistern will be filled in exactly 2.4 hours.
-/
theorem cistern_fill_time :
  let rate_A := 1 / 4
  let rate_B := -1 / 6
  let rate_C := 1 / 3
  let combined_rate := rate_A + rate_B + rate_C
  let fill_time := 1 / combined_rate
  fill_time = 2.4 := by
  sorry

end cistern_fill_time_l404_404550


namespace domain_of_g_l404_404249

-- Define the function g(x)
def g (x : ℝ) : ℝ := Real.sqrt(5 - 15 * x^2 - 10 * x)

-- The statement of the domain of g(x)
theorem domain_of_g : ∀ x : ℝ, (0 ≤ 5 - 15 * x^2 - 10 * x) ↔ (x ∈ Set.Icc (-1 / 3 : ℝ) 1) :=
by
  sorry

end domain_of_g_l404_404249


namespace net_profit_is_correct_l404_404477

-- Define the known quantities
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.20
def markup : ℝ := 45

-- Define the derived quantities based on the conditions
def overhead : ℝ := overhead_percentage * purchase_price
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + markup
def net_profit : ℝ := selling_price - total_cost

-- The statement to prove
theorem net_profit_is_correct : net_profit = 45 := by
  sorry

end net_profit_is_correct_l404_404477


namespace next_date_time_appears_same_digits_l404_404468

def digits_used (date : Nat × Nat × Nat) (time : Nat × Nat) : Set Nat :=
  let (day, month, year) := date
  let (hour, minute) := time
  day.digits ++ month.digits ++ year.digits ++ hour.digits ++ minute.digits

def valid_digits (d : Set Nat) (valid : Set Nat) : Bool :=
  d.to_finset ⊆ (valid.to_finset)

def next_valid_time (current_time : Nat × Nat) (valid_digits : Set Nat) : Option (Nat × Nat × Nat × Nat) :=
  sorry -- Function to compute the next valid date-time given current-time and valid digits

theorem next_date_time_appears_same_digits :
  ∃ date time, (date = (1, 8, 1994) ∧ time = (2, 45)) ∧
  let valid := {0, 1, 1, 4, 4, 5, 7, 9, 9} in
  valid_digits (digits_used (1, 8, 1994) (2, 45)) valid :=
begin
  -- We assume current time and valid digits as proved to be reused at least once in the year
  let current_time := (5, 1, 1994, 7 * 60 + 32)
  let valid := {0, 5, 1, 1, 9, 9, 4, 0, 7, 3}

  have h := next_valid_time current_time valid,
  sorry -- Lean steps to complete the proof go here
end

end next_date_time_appears_same_digits_l404_404468


namespace binomial_12_10_l404_404590

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404590


namespace solve_for_x_l404_404354

theorem solve_for_x (x : ℝ) (h : 3*x - 4*x + 5*x = 140) : x = 35 :=
by 
  sorry

end solve_for_x_l404_404354


namespace quadratic_graph_value_at_5_l404_404467

theorem quadratic_graph_value_at_5
  (a b c : ℝ)
  (h_vertex : ∀ x, (y = ax^2 + bx + c) → (y - 7) = a(x - 2)^2)
  (h_point_0_minus7 : ∀ y, (y = a * 0^2 + b * 0 + c) → (y = -7))
  (h_point_5_n : ∀ y, (y = a * 5^2 + b * 5 + c) → (y = n)) :
  n = -24.5 :=
sorry

end quadratic_graph_value_at_5_l404_404467


namespace intersection_eq_l404_404333

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by sorry

end intersection_eq_l404_404333


namespace proof_problem_l404_404303

noncomputable theory

-- Definitions for the given conditions
def is_sequence (a : ℕ → ℝ) (q : ℝ) (h_q : q > 1) (h1 : a 1 + a 3 = 20) (h2 : a 2 = 8) :=
∃ a1 : ℝ, ∃ q : ℝ, q > 1 ∧ a 1 = a1 ∧ a 2 = a1 * q ∧ a 3 = a1 * q * q ∧ a1 + a1 * q ^ 2 = 20 ∧ a1 * q = 8

-- The general term formula for the sequence.
def general_term (a : ℕ → ℝ) := ∀ n : ℕ, a n = 2^(n + 1)

-- Definitions for the second part of the problem
def S_n (b : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, b i

def b_n (a : ℕ → ℝ) (n : ℕ) := n / a n

def satisfies_inequality (S_n : ℕ → ℝ) (a : ℝ) := ∀ n : ℕ, S_n n + n / 2^(n + 1) > (-1)^n * a

-- The range of a that satisfies the inequality.
def range_of_a (a : ℝ) := -1/2 < a ∧ a < 3/4

-- Main theorem to prove
theorem proof_problem (a : ℕ → ℝ) (q : ℝ) (h_q : q > 1) (h1 : a 1 + a 3 = 20) (h2 : a 2 = 8) :
  (general_term a) ∧ (∃ b : ℕ → ℝ, (∀ n : ℕ, b n = b_n a n) ∧ satisfies_inequality (S_n b) a ↔ range_of_a a) :=
by { sorry }

end proof_problem_l404_404303


namespace cookies_in_each_bag_l404_404415

theorem cookies_in_each_bag (chocolate_chip oatmeal : ℕ) (baggies : ℕ)
  (h1 : chocolate_chip = 2)
  (h2 : oatmeal = 16)
  (h3 : baggies = 6) :
  (chocolate_chip + oatmeal) / baggies = 3 :=
by
  rw [h1, h2, h3]
  exact Nat.div_eq_of_eq_mul (Nat.mul_eq_mul_left_iff.mpr (Or.inl rfl)).symm
  sorry

end cookies_in_each_bag_l404_404415


namespace healthy_child_probability_l404_404934

-- Definitions based on conditions
def Family1_CarriesPhenylketonuria : Prop := ¬ (Family2_CarriesPhenylketonuria)
def Family2_CarriesHemophilia : Prop := ¬ (Family1_CarriesHemophilia)
def Family1_CarriesAlkaptonuria : Prop := ¬ (Family2_CarriesAlkaptonuria)

axiom Phenylketonuria_Recessive : Prop
axiom Alkaptonuria_Recessive : Prop
axiom Hemophilia_Recessive : Prop

structure Individual :=
  (phenylketonuria : Bool)
  (alkaptonuria : Bool)
  (hemophilia : Bool)
  (gender : String)

-- Assuming these conditions
axiom healthy_father_Ⅱ3 : Individual
axiom mother_with_phenylketonuria_hemophilia : Individual

-- Based on problem conditions
def individual_Ⅱ3 := Individual.mk false false false "female"

-- Statement to prove
theorem healthy_child_probability :
  (Probability (child_of individual_Ⅱ3).gender = "girl" -> 1)
  ∧ (Probability (child_of individual_Ⅱ3).gender = "boy"
      ∧ Probability (child_of individual_Ⅱ3).phenylketonuria = false 
      ∧ Probability (child_of individual_Ⅱ3).alkaptonuria = false
      ∧ Probability (child_of individual_Ⅱ3).hemophilia = false) = 3 / 4 :=
begin
  sorry
end

end healthy_child_probability_l404_404934


namespace transformed_coordinates_l404_404181

variables (ρ θ φ : ℝ) (x y z : ℝ)

/-- Cartesian to spherical coordinate transformation -/
def spherical_to_cartesian (ρ φ θ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

/-- Original point in Cartesian coordinates and corresponding spherical coordinates -/
variables (x₀ y₀ z₀ : ℝ)
axiom coords: x₀ = 3 ∧ y₀ = -4 ∧ z₀ = 12
axiom original_spherical: (x₀, y₀, z₀) = spherical_to_cartesian ρ φ θ

/-- We need to show that the transformed point remains consistent -/
theorem transformed_coordinates :
  spherical_to_cartesian ρ φ (-θ) = (3, 4, 12) :=
sorry

end transformed_coordinates_l404_404181


namespace domain_of_c_is_all_reals_l404_404633

theorem domain_of_c_is_all_reals (k : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 3 := 
by
  sorry

end domain_of_c_is_all_reals_l404_404633


namespace find_x_in_interval_l404_404992

theorem find_x_in_interval (x : ℝ) 
  (h₁ : 4 ≤ (x + 1) / (3 * x - 7)) 
  (h₂ : (x + 1) / (3 * x - 7) < 9) : 
  x ∈ Set.Ioc (32 / 13) (29 / 11) := 
sorry

end find_x_in_interval_l404_404992


namespace calculate_speed_train2_l404_404512

-- conditions
def length_train1 : ℝ := 140
def length_train2 : ℝ := 190
def speed_train1 : ℝ := 40 -- in km/hr
def crossing_time : ℝ := 11.879049676025918 -- in seconds

-- conversions
def distance_crossed := length_train1 + length_train2 -- in meters
def relative_speed_m_per_s := distance_crossed / crossing_time -- in m/s
def relative_speed_km_per_hr := relative_speed_m_per_s * 3600 / 1000 -- in km/hr

-- goal: speed of the other train
def speed_train2 := relative_speed_km_per_hr - speed_train1

-- statement to prove
theorem calculate_speed_train2 : speed_train2 = 60 := by
  sorry

end calculate_speed_train2_l404_404512


namespace max_distance_from_circle_to_line_l404_404095

theorem max_distance_from_circle_to_line
  (x y : ℝ)
  (h1 : x^2 + y^2 - 2*x - 2*y + 1 = 0)
  (h2 : ∀ x y : ℝ, x - y = 2 → True) :
  ∃ d : ℝ, d = 1 + real.sqrt 2 := sorry

end max_distance_from_circle_to_line_l404_404095


namespace carson_seed_l404_404963

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l404_404963


namespace cleaning_time_if_anne_doubled_l404_404954

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l404_404954


namespace find_y_l404_404472

noncomputable def x := -12
noncomputable def k := 37.5 * 12.5

theorem find_y (x y : ℝ) 
  (h1 : x * y = k)
  (h2 : x = 3 * y)
  (h3 : x + y = 50)
  (hx : x = -12) : 
  y = -39.0625 :=
sorry

end find_y_l404_404472


namespace fourth_person_height_l404_404109

theorem fourth_person_height 
  (height1 height2 height3 height4 : ℝ)
  (diff12 : height2 = height1 + 2)
  (diff23 : height3 = height2 + 2)
  (diff34 : height4 = height3 + 6)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 76) :
  height4 = 82 :=
by
  sorry

end fourth_person_height_l404_404109


namespace hyperbola_eccentricity_l404_404694

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : c^2 = a^2 + b^2)
  (h4 : ∃ x y : ℝ, (x - 3)^2 + y^2 = 4 ∧ y = b / a * x ∧ 2 * sqrt (4 - 9 * b^2 / (a^2 + b^2)) = 2) :
  c / a = sqrt 6 / 2 :=
by
  sorry

end hyperbola_eccentricity_l404_404694


namespace count_valid_sentences_l404_404451

-- Define the set of words and the conditions that restrict certain sequences
inductive Word
| elara 
| quen 
| silva 
| nore
deriving DecidableEq

-- Define valid sequences predicates based on the restrictions from the problem
def valid_seq (w1 w2 : Word) : Prop :=
  ¬(w1 = Word.elara ∧ w2 = Word.quen) ∧ ¬(w1 = Word.silva ∧ w2 = Word.nore)

-- Function to check if a 3-word sentence is valid
def valid_sentence : Word → Word → Word → Prop
| w1, w2, w3 => valid_seq w1 w2 ∧ valid_seq w2 w3

theorem count_valid_sentences : 
  (Finset.univ : Finset (Word × Word × Word)).filter (λ s => valid_sentence s.1.1 s.1.2 s.2).card = 48 :=
by
  sorry

end count_valid_sentences_l404_404451


namespace sum_of_d_k_64800_l404_404774

-- Define the function d(n) which denotes the number of positive divisors of n
def d (n : ℕ) : ℕ := (finset.range n).filter (λ k, n % k = 0).card

-- Define the sum of d(k) for all positive integral divisors k of n
def sum_d_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0).sum d

theorem sum_of_d_k_64800 :
  sum_d_divisors 64800 = 1890 := 
sorry

end sum_of_d_k_64800_l404_404774


namespace Felipe_time_l404_404124

theorem Felipe_time (together_years : ℝ) (felipe_ratio : ℝ) : 
  together_years = 7.5 → felipe_ratio = 0.5 → 
  (F : ℝ), 12 * F = 30 :=
by
  intro h_together h_ratio
  let F := 2.5
  sorry

end Felipe_time_l404_404124


namespace geometric_progression_theorem_l404_404048

theorem geometric_progression_theorem 
  (a b c d : ℝ) (q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2) 
  (h3 : d = a * q^3) 
  : (a - d)^2 = (a - c)^2 + (b - c)^2 + (b - d)^2 := 
by sorry

end geometric_progression_theorem_l404_404048


namespace seq_bounds_l404_404628

theorem seq_bounds (n : ℕ) (h : n > 1):
  let a : ℕ → ℝ := λ k, Nat.recOn k (1 / 2) (λ k ak, ak + (ak^2 / n)) in
  1 - (1 / n) < a n ∧ a n < 1 :=
begin
  sorry
end

end seq_bounds_l404_404628


namespace count_equilateral_triangles_with_outer_vertex_l404_404983

-- Given lattice configuration and set definitions
def is_hexagonal_lattice (G : Type) : Prop := sorry  -- This predicate describes the hexagonal lattice structure
def equilateral_triangle (t : G) : Prop := sorry  -- This predicate checks if t is an equilateral triangle within G

-- Statement of the problem
theorem count_equilateral_triangles_with_outer_vertex
  (G : Type)
  [is_hexagonal_lattice G]
  (H : set G)
  (condition : ∀ t, t ∈ H → equilateral_triangle t)
  (outer_vertex_condition : ∀ t ∈ H, ∃ v : G, v ∈ t ∧ is_outer_vertex v) : 
  ∃ n : ℕ, n = 32 := 
sorry

end count_equilateral_triangles_with_outer_vertex_l404_404983


namespace find_initial_amount_l404_404995

noncomputable def initial_amount (r: ℝ) (n: ℝ) (t: ℝ) (CI: ℝ) : ℝ :=
  let A := CI + P in
  CI / ((1 + r / n)^(n * t) - 1)

theorem find_initial_amount : 
  initial_amount 0.20 1 3 873.60 = 1200 :=
by sorry

end find_initial_amount_l404_404995


namespace items_sold_count_l404_404573

theorem items_sold_count (n m : ℕ) (hn : n = 13) (hm : m = 20) :
  n + m + 1 = 34 :=
by
  rw [hn, hm]
  rfl

end items_sold_count_l404_404573


namespace region_area_l404_404871

noncomputable def area_of_region : ℝ := 
  let region := { p : ℝ × ℝ | abs (4 * p.1 - 20) + abs (3 * p.2 + 9) ≤ 6 } in
  let vertices := {(6.5, -1), (3.5, -1), (5, 0), (5, -4)} : set (ℝ × ℝ) in
  if H : ∀ v ∈ vertices, v ∈ region then
    -- Calculate the area of the diamond
    6
  else
    -- This case should not happen according to our problem statement
    sorry

theorem region_area :
  let region := { p : ℝ × ℝ | abs (4 * p.1 - 20) + abs (3 * p.2 + 9) ≤ 6 } in
  let vertices := {(6.5, -1), (3.5, -1), (5, 0), (5, -4)} : set (ℝ × ℝ) in
  ∀ v ∈ vertices, v ∈ region → area_of_region = 6 :=
by
  sorry

end region_area_l404_404871


namespace binom_12_10_eq_66_l404_404598

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404598


namespace kelvin_jumps_ways_l404_404766

theorem kelvin_jumps_ways : 
  (∑ k in finset.filter Nat.prime (finset.Icc (0:ℕ) 10), 
   if k % 2 = 0 then nat.choose 10 ((10 + k) / 2) else 0) = 210 :=
by
  sorry

end kelvin_jumps_ways_l404_404766


namespace sum_powers_mod_div_l404_404775

theorem sum_powers_mod_div {n : ℕ} (h : n ≥ 2) :
  (∀ p : ℕ, p.Prime → p ∣ n → p ∣ (n / p - 1) ∧ p - 1 ∣ (n / p - 1)) ↔
  n ∣ (∑ i in finset.range (n-1), i^(n-1) + 1) := by
  sorry

end sum_powers_mod_div_l404_404775


namespace number_of_yellow_marbles_l404_404912

theorem number_of_yellow_marbles (total_marbles blue_marbles red_marbles green_marbles yellow_marbles : ℕ)
    (h_total : total_marbles = 164) 
    (h_blue : blue_marbles = total_marbles / 2)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27) :
    yellow_marbles = total_marbles - (blue_marbles + red_marbles + green_marbles) →
    yellow_marbles = 14 := by
  sorry

end number_of_yellow_marbles_l404_404912


namespace find_number_l404_404418

-- Define the condition given in the problem
def condition (x : ℤ) := 13 * x - 272 = 105

-- Prove that given the condition, x equals 29
theorem find_number : ∃ x : ℤ, condition x ∧ x = 29 :=
by
  use 29
  unfold condition
  sorry

end find_number_l404_404418


namespace football_campers_l404_404199

theorem football_campers {T S B F : ℕ} (hT : T = 88) (hS : S = 32) (hB : B = 24) : F = T - S - B → F = 32 :=
by
  intros hF
  rw [hT, hS, hB] at hF
  conv at hF => 
    lhs
    rw [← add_assoc, add_comm 32 24, add_comm 24 32]
  exact hF

end football_campers_l404_404199


namespace greatest_possible_value_of_a_l404_404465

theorem greatest_possible_value_of_a (a : ℤ) (h1 : ∃ x : ℤ, x^2 + a*x = -30) (h2 : 0 < a) :
  a ≤ 31 :=
sorry

end greatest_possible_value_of_a_l404_404465


namespace binomial_12_10_eq_66_l404_404618

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404618


namespace exists_infinite_male_lineage_l404_404050

-- Define what it means for a lineage to be infinite
def infinite_lineage (lineage : ℕ → ℕ) : Prop :=
  ∀ n, lineage n < lineage (n + 1)

-- Adam is the origin, represented by 0 for simplicity
def Adam := 0

-- Define the condition that humanity is immortal, implying an infinite number of men
axiom humanity_immortal : ∃ f : ℕ → ℕ, surjective f

-- Define the condition that all men are descendants of Adam
axiom all_men_descendants_of_adam : ∃ male_descendant : ℕ → ℕ, male_descendant 0 = Adam

theorem exists_infinite_male_lineage : ∃ lineage : ℕ → ℕ, 
  lineage 0 = Adam ∧ 
  infinite_lineage lineage := by
  sorry

end exists_infinite_male_lineage_l404_404050


namespace horse_food_amount_per_day_l404_404939

variable (num_sheep num_horses total_food food_per_horse : ℕ)
variable (ratio_sh : ℕ)
variable (ratio_h : ℕ)
variable (total_sheep : ℕ)
variable (total_food : ℕ)
variable (ratio_unit_sheep : ℕ)
variable (ratio_unit_horses : ℕ)

-- Definitions based on the conditions
def sheep_to_horses_ratio : Prop := 
  ratio_unit_sheep / ratio_unit_horses = 3 / 7

def total_sheep_farmer : Prop := 
  total_sheep = 24

def total_food_needed : Prop := 
  total_food = 12880

-- The main theorem
theorem horse_food_amount_per_day
  (h1 : sheep_to_horses_ratio)
  (h2 : total_sheep_farmer)
  (h3 : total_food_needed)
  : food_per_horse = 230 := sorry

end horse_food_amount_per_day_l404_404939


namespace find_x_l404_404677

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * f x - 19 = f (x - 4)) : x = 4 := 
by 
  sorry

end find_x_l404_404677


namespace sum_max_value_l404_404399

theorem sum_max_value (n : ℕ) (h₁ : 2 ≤ n) (a : Fin n → ℝ) (h₂ : ∀ i, 0 < a i ∧ a i < 1) :
  ∑ i in Finset.range n, (a i * (1 - a ((i + 1) % n))) ^ (1 / 6) ≤ (Real.sqrt 2 * n) / 3 :=
sorry

end sum_max_value_l404_404399


namespace sum_of_common_ratios_l404_404025

variable {k a2 a3 b2 b3 : ℝ}
variable (p q : ℝ)

-- Define what it means for sequences to be geometric with common ratios p and q and different.
def is_geometric (a b k p q : ℝ) : Prop :=
  p ≠ q ∧ ∀ n : ℕ, a = a2 * p^n ∧ b = b2 * q^n

-- The given condition: a₃ - b₃ = k²(a₂ - b₂)
def given_condition (a2 a3 b2 b3 k : ℝ) : Prop :=
  a3 - b3 = k^2 * (a2 - b2)

-- The final statement to prove: (sum of the common ratios of both sequences is k)
theorem sum_of_common_ratios
  (a2 a3 b2 b3 k p q : ℝ)
  (h : is_geometric a2 a3 b2 b3 k p q)
  (h_condition : given_condition a2 a3 b2 b3 k) :
  p + q = k :=
sorry

end sum_of_common_ratios_l404_404025


namespace range_of_y_function_l404_404879

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l404_404879


namespace sale_price_same_as_original_l404_404866

theorem sale_price_same_as_original (x : ℝ) :
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sale_price = x := 
by
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sorry

end sale_price_same_as_original_l404_404866


namespace cyclic_sum_inequality_l404_404296

variable (a b c : ℝ) 
variable (h1 : 4 * a * b * c = a + b + c + 1)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem cyclic_sum_inequality : 
    (a^2 + a) + (b^2 + b) + (c^2 + c) ≥ 2 * (a * b + b * c + c * a) := 
by 
  sorry

end cyclic_sum_inequality_l404_404296


namespace fraction_value_l404_404139

theorem fraction_value (b : ℝ) (hb : b = 2) : (3 * b⁻¹ + (b⁻¹ / 3) + 3) / (b + 1) = 14 / 9 :=
by
  rw hb
  sorry

end fraction_value_l404_404139


namespace count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l404_404554

-- Definitions based on conditions
def is_symmetric_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def symmetric_pair (a b : ℕ) : Prop :=
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) ∨ (a = 8 ∧ b = 8) ∨ (a = 6 ∧ b = 9) ∨ (a = 9 ∧ b = 6)

-- 1. Prove the total number of 7-digit symmetric numbers
theorem count_symmetric_numbers : ∃ n, n = 300 := by
  sorry

-- 2. Prove the number of symmetric numbers divisible by 4
theorem count_symmetric_divisible_by_4 : ∃ n, n = 75 := by
  sorry

-- 3. Prove the total sum of these 7-digit symmetric numbers
theorem sum_symmetric_numbers : ∃ s, s = 1959460200 := by
  sorry

end count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l404_404554


namespace hyperbola_condition_hyperbola_not_necessary_l404_404352

theorem hyperbola_condition (k : ℝ) : (k > 3) → ((k-3) * (k+3) > 0) :=
begin
  sorry,
end

theorem hyperbola_not_necessary (k : ℝ) : ¬(k > 3) → ((k-3) * (k+3) > 0) ∨ ((k-3) * (k+3) < 0):= 
begin
  sorry,
end

end hyperbola_condition_hyperbola_not_necessary_l404_404352


namespace rotate_D_90_clockwise_l404_404821

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℤ) : Point ℤ :=
  ⟨p.y, -p.x⟩

def D : Point ℤ := ⟨-3, 2⟩
def E : Point ℤ := ⟨0, 5⟩
def F : Point ℤ := ⟨0, 2⟩

theorem rotate_D_90_clockwise :
  rotate_90_clockwise D = Point.mk 2 (-3) :=
by
  sorry

end rotate_D_90_clockwise_l404_404821


namespace count_bad_arrangements_l404_404847

-- Defining what it means for an arrangement to be bad
def is_bad_arrangement (arr : List ℕ) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 20 → ¬ ∃ subset : List ℕ, 
                               subset ≠ [] ∧ 
                               subset.sum = n ∧ 
                               List.is_prefix_of subset arr ∨
                               List.is_prefix_of subset (arr.rotate 1) ∨
                               List.is_prefix_of subset (arr.rotate 2) ∨
                               List.is_prefix_of subset (arr.rotate 3) ∨
                               List.is_prefix_of subset (arr.rotate 4) ∨
                               List.is_prefix_of subset (arr.rotate 5)

-- The main theorem to prove
theorem count_bad_arrangements : 
  ∃ (bad_arrangements : ℕ), bad_arrangements = 3 :=
by
  sorry  -- Proof of the theorem

end count_bad_arrangements_l404_404847


namespace parabola_problem_l404_404311

noncomputable def parabola_equation : ℝ → ℝ := λ y => y^2 - 8 * (0 * y + 2)

theorem parabola_problem 
(O : Point) 
(F : Point) 
(A : Point) 
(l : Line) 
(P Q : Point) 
(M N : Point)
(h_eq : O = (0, 0)) 
(h_parab_eq : ∀ x y : ℝ, parabola_equation y = 8 * x) 
(hF : F = (2, 0)) 
(hA : A = (2, 4)) 
(h_line : ∀ x y : ℝ, x = 0 * y + 2) 
(h_PQ_on_parab : P ≠ A ∧ Q ≠ A ∧ parabola_equation P.y = 0 ∧ parabola_equation Q.y = 0)
(h_AP_inter_x : ∀ y : ℝ, parabola_equation y = 0 → AP.y = 0 → M = (-(y / 2), 0)) 
(h_AQ_inter_x : ∀ y : ℝ, parabola_equation y = 0 → AQ.y = 0 → N = (-(y / 2), 0)) :
| OM | * | ON | = 4 :=
sorry

end parabola_problem_l404_404311


namespace inequality_on_positive_reals_l404_404407

variable {a b c : ℝ}

theorem inequality_on_positive_reals (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_on_positive_reals_l404_404407


namespace derrick_yard_length_l404_404629

def alex_yard (derrick_yard : ℝ) := derrick_yard / 2
def brianne_yard (alex_yard : ℝ) := 6 * alex_yard

theorem derrick_yard_length : brianne_yard (alex_yard derrick_yard) = 30 → derrick_yard = 10 :=
by
  intro h
  sorry

end derrick_yard_length_l404_404629


namespace BD_length_l404_404761

theorem BD_length
  (A B C D : Type)
  (dist_AC : ℝ := 10)
  (dist_BC : ℝ := 10)
  (dist_AD : ℝ := 12)
  (dist_CD : ℝ := 5) : (BD : ℝ) = 95 / 12 :=
by
  sorry

end BD_length_l404_404761


namespace multiples_of_15_between_17_and_202_l404_404718

theorem multiples_of_15_between_17_and_202 : 
  ∃ n : ℕ, (∀ k : ℤ, 17 < k * 15 ∧ k * 15 < 202 → k = n + 1) ∧ n = 12 :=
sorry

end multiples_of_15_between_17_and_202_l404_404718


namespace eccentricity_range_l404_404318

noncomputable def main : Prop :=
  ∃ (a b e : ℝ),
  (a > b ∧ b > 0) ∧
  (a^2 * b^2 > 0) ∧
  (∃ x y : ℝ, (x = (sqrt 2) / 2) ∧ (y = (sqrt 2) / 2)) ∧
  (sqrt 5 / 2 ≤ a ∧ a ≤ sqrt 6 / 2) ∧ 
  (frac x^2 / a^2 + y^2 / b^2 = 1) ∧
  (e = sqrt (1 - b^2 / a^2)) ∧
  (frac sqrt 3 / 3 ≤ e ∧ e ≤ sqrt 2 / 2)

theorem eccentricity_range : main := 
  sorry

end eccentricity_range_l404_404318


namespace find_number_l404_404656

theorem find_number (N : ℝ) (h : (0.47 * N - 0.36 * 1412) + 66 = 6) : N = 953.87 :=
  sorry

end find_number_l404_404656


namespace eval_p_20_l404_404225

noncomputable theory
open Real

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem eval_p_20 (a b c : ℝ)
  (h_symmetry : 9 < 9.5 ∧ 9.5 < 10)
  (h_point : quadratic_function a b c 0 = -8) :
  quadratic_function a b c 20 = -8 :=
sorry

end eval_p_20_l404_404225


namespace sum_of_squares_of_reciprocal_roots_l404_404796

theorem sum_of_squares_of_reciprocal_roots :
  ∀ r s t : ℝ,
    (3 * r^3 + 2 * r^2 + 4 * r + 1 = 0) ∧
    (3 * s^3 + 2 * s^2 + 4 * s + 1 = 0) ∧
    (3 * t^3 + 2 * t^2 + 4 * t + 1 = 0) →
    (r + s + t = -2 / 3) →
    (r * s + r * t + s * t = 4 / 3) →
    (r * s * t = -1 / 3) →
    (1 / r)^2 + (1 / s)^2 + (1 / t)^2 = -20 :=
begin
  intros r s t h_r h_s h_t h_sum h_prod h_prod_all,
  sorry
end

end sum_of_squares_of_reciprocal_roots_l404_404796


namespace odd_function_inequalities_l404_404676

theorem odd_function_inequalities (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_ineq : ∀ x, x > 0 → 2 * f x - x * (derivative (derivative f) x) > 0) :
  2 * f (-real.sqrt 3) > 3 * f (-real.sqrt 2) ∧ 3 * f (-real.sqrt 2) > 6 * f (-1) :=
by 
  sorry

end odd_function_inequalities_l404_404676


namespace new_profit_percentage_l404_404742

variables {cost food_cost labor_cost overhead_cost new_food_cost new_labor_cost new_total_cost selling_price new_profit : ℝ}

def initial_conditions (cost profit food_cost labor_cost overhead_cost : ℝ) : Prop :=
  profit = 1.7 * cost ∧
  cost = food_cost + labor_cost + overhead_cost ∧
  food_cost = 0.65 * cost ∧
  labor_cost = 0.25 * cost ∧
  overhead_cost = 0.10 * cost

def updated_costs (food_cost labor_cost overhead_cost new_food_cost new_labor_cost new_total_cost : ℝ) : Prop :=
  new_food_cost = food_cost + 0.14 * food_cost ∧
  new_labor_cost = labor_cost + 0.05 * labor_cost ∧
  new_total_cost = new_food_cost + new_labor_cost + overhead_cost

def constant_selling_price (selling_price cost profit : ℝ) : Prop :=
  selling_price = cost + profit

def new_profit_conditions (selling_price new_total_cost new_profit : ℝ) : Prop :=
  new_profit = selling_price - new_total_cost

theorem new_profit_percentage :
  ∃ selling_price cost profit food_cost labor_cost overhead_cost new_food_cost new_labor_cost new_total_cost new_profit,
    initial_conditions cost profit food_cost labor_cost overhead_cost ∧
    constant_selling_price selling_price cost profit ∧
    updated_costs food_cost labor_cost overhead_cost new_food_cost new_labor_cost new_total_cost ∧
    new_profit_conditions selling_price new_total_cost new_profit ∧
    new_profit / selling_price * 100 ≈ 59.13 :=
by { sorry }

end new_profit_percentage_l404_404742


namespace cleaning_time_with_doubled_an_speed_l404_404948

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l404_404948


namespace find_new_coords_l404_404183

-- Define the initial conditions for the given point
variables (ρ θ φ : ℝ)

-- Condition for spherical coordinates related to rectangular coordinates
def rectangular_coords (x y z : ℝ) : Prop :=
  x = ρ * sin φ * cos θ ∧ y = ρ * sin φ * sin θ ∧ z = ρ * cos φ

-- The given point has rectangular coordinates (3, -4, 12)
variables (h1 : rectangular_coords 3 (-4) 12)

-- Define the target condition for the new spherical coordinates
def new_rectangular_coords (x y z : ℝ) : Prop :=
  x = ρ * sin φ * cos (-θ) ∧ y = ρ * sin φ * sin (-θ) ∧ z = ρ * cos φ

-- The target point has rectangular coordinates (3, 4, 12)
theorem find_new_coords : new_rectangular_coords 3 4 12 :=
by {
  -- The proof would be inserted here
  sorry
}

end find_new_coords_l404_404183


namespace binom_12_10_eq_66_l404_404592

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404592


namespace unique_function_satisfying_condition_l404_404651

theorem unique_function_satisfying_condition :
  ∃! f : ℤ → ℤ, (∀ a b : ℤ, f (a + b) + f (a * b) = f a * f b - 1) ∧ ∀ n : ℤ, f n = abs (2 - n) :=
begin
  -- proof will go here
  sorry
end

end unique_function_satisfying_condition_l404_404651


namespace range_of_f_l404_404876

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l404_404876


namespace part1_part2_l404_404911

-- Definition of concave function
def concave_function (f : ℝ → ℝ) (A : set ℝ) : Prop :=
∀ x1 x2 ∈ A, f ((x1 + x2) / 2) ≤ (f x1 + f x2) / 2

-- Part (1): Check if f(x) = 3x^2 + x is concave on ℝ
def f1 (x : ℝ) : ℝ := 3 * x^2 + x

theorem part1 : concave_function f1 set.univ := sorry

-- Part (2): Find the range of m for f(x) = m * x^2 + x to be concave on ℝ
def f2 (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x

theorem part2 (m : ℝ) : concave_function (f2 m) set.univ ↔ m ≥ 0 := sorry

end part1_part2_l404_404911


namespace watermelon_heavier_than_pineapple_l404_404494

noncomputable def watermelon_weight : ℕ := 1 * 1000 + 300 -- Weight of one watermelon in grams
noncomputable def pineapple_weight : ℕ := 450 -- Weight of one pineapple in grams

theorem watermelon_heavier_than_pineapple :
    (4 * watermelon_weight = 5 * 1000 + 200) →
    (3 * watermelon_weight + 4 * pineapple_weight = 5 * 1000 + 700) →
    watermelon_weight - pineapple_weight = 850 :=
by
    intros h1 h2
    sorry

end watermelon_heavier_than_pineapple_l404_404494


namespace total_students_shook_hands_l404_404114

theorem total_students_shook_hands (S3 S2 S1 : ℕ) (h1 : S3 = 200) (h2 : S2 = S3 + 40) (h3 : S1 = 2 * S2) : 
  S1 + S2 + S3 = 920 :=
by
  sorry

end total_students_shook_hands_l404_404114


namespace find_b_l404_404844

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end find_b_l404_404844


namespace Sydney_initial_rocks_l404_404450

variable (S₀ : ℕ)

def Conner_initial : ℕ := 723
def Sydney_collects_day1 : ℕ := 4
def Conner_collects_day1 : ℕ := 8 * Sydney_collects_day1
def Sydney_collects_day2 : ℕ := 0
def Conner_collects_day2 : ℕ := 123
def Sydney_collects_day3 : ℕ := 2 * Conner_collects_day1
def Conner_collects_day3 : ℕ := 27

def Total_Sydney_collects : ℕ := Sydney_collects_day1 + Sydney_collects_day2 + Sydney_collects_day3
def Total_Conner_collects : ℕ := Conner_collects_day1 + Conner_collects_day2 + Conner_collects_day3

def Total_Sydney_rocks : ℕ := S₀ + Total_Sydney_collects
def Total_Conner_rocks : ℕ := Conner_initial + Total_Conner_collects

theorem Sydney_initial_rocks :
  Total_Conner_rocks = Total_Sydney_rocks → S₀ = 837 :=
by
  sorry

end Sydney_initial_rocks_l404_404450


namespace spider_travel_distance_l404_404196

theorem spider_travel_distance
  (r : ℝ) (r_pos : r = 65) 
  (d : ℝ) (d_eq : d = 2 * r) 
  (a : ℝ) (a_eq : a = 90) 
  (b : ℝ) (b_eq : b = (real.sqrt (d^2 - a^2))) :
  d + a + b = 220 + 20 * real.sqrt 22 :=
by sorry

end spider_travel_distance_l404_404196


namespace y_increase_percentage_l404_404061

variables (x y k q : ℝ)
hypothesis h_inv_prop : x * y = k
hypothesis h_positive_x : 0 < x
hypothesis h_positive_y : 0 < y
hypothesis h_positive_q : 0 < q

theorem y_increase_percentage :
  (∀ x' y', (x' = x * (1 - q / 100)) → (x' * y' = k) → (100 * (y' - y) / y = 100 * q / (100 - q))) sorry

end y_increase_percentage_l404_404061


namespace problem_solution_l404_404357

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → |a * x^3 - log x| ≥ 1) → a ≥ Real.exp 2 / 3 :=
by
  assume h: ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |a * x^3 - log x| ≥ 1
  sorry

end problem_solution_l404_404357


namespace complex_number_in_third_quadrant_l404_404098

theorem complex_number_in_third_quadrant 
  (i : ℂ := complex.I)
  (z : ℂ := i * (1 + i)) :
  ∃ q : ℕ, z = -1 + i ∧ q = 3 :=
by
  sorry

end complex_number_in_third_quadrant_l404_404098


namespace age_problem_solution_l404_404913

namespace AgeProblem

variables (S M : ℕ) (k : ℕ)

-- Condition: The present age of the son is 22
def son_age (S : ℕ) := S = 22

-- Condition: The man is 24 years older than his son
def man_age (M S : ℕ) := M = S + 24

-- Condition: In two years, man's age will be a certain multiple of son's age
def age_multiple (M S k : ℕ) := M + 2 = k * (S + 2)

-- Question: The ratio of man's age to son's age in two years
def age_ratio (M S : ℕ) := (M + 2) / (S + 2)

theorem age_problem_solution (S M : ℕ) (k : ℕ) 
  (h1 : son_age S)
  (h2 : man_age M S)
  (h3 : age_multiple M S k)
  : age_ratio M S = 2 :=
by
  rw [son_age, man_age, age_multiple, age_ratio] at *
  sorry

end AgeProblem

end age_problem_solution_l404_404913


namespace ellipse_eq_form_l404_404304

noncomputable def ellipse_c (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ a = sqrt 2 ∧ b = 1 ∧ (e : ℝ) = sqrt 2 / 2 ∧ 
  P.x = 1 ∧ P.y = 1 / sqrt 2 ∧ |O - P| = sqrt 6 / 2

noncomputable def equation_of_line (k : ℝ) : Prop :=
  exists (a b : ℝ), a > b ∧ b > 0 ∧ 
  (a = sqrt 2 ∧ b = 1 ∧ (e : ℝ) = sqrt 2 / 2 ∧ 
  ( |O - P| = sqrt 6 / 2 ∧ 
    M.x = P.x / sqrt 2 / 2) ∧ 
    ((S : ℝ) = sqrt 2 / 2 ∧ (S_form: Formula.)triangle_area ((A : ℝ) + A.y k P = B) ∧ 2 / (1 + k^2) sqrt(1 + k^2) - intersect AOB = x

theorem ellipse_eq_form (C : ℝ): ellipse_c -> equation_of_line ->
 Prop = sorry

end ellipse_eq_form_l404_404304


namespace dance_lessons_l404_404127

theorem dance_lessons (cost_per_lesson : ℕ) (free_lessons : ℕ) (amount_paid : ℕ) 
  (H1 : cost_per_lesson = 10) 
  (H2 : free_lessons = 2) 
  (H3 : amount_paid = 80) : 
  (amount_paid / cost_per_lesson + free_lessons = 10) :=
by
  sorry

end dance_lessons_l404_404127


namespace ellipse_standard_eq_value_of_m_l404_404675

theorem ellipse_standard_eq :
  (∃ a b : ℝ, e = (sqrt 2 / 2) ∧ (3 * vec_qa (x1, y1 - 3) * vec_qb (x2, y2 - 3)) = 32) →
  (∀ a > 0, ∀ b > 0, ∃ (C : Set ℝ), C = { (x, y) | x^2 / a^2 + y^2 / b^2 = 1 }) →
  (ℝ) :=
by
  sorry

theorem value_of_m :
  (∃ m : ℝ, for the line l: y = x + m and point Q(0, 3), ∀ a, b, m = 1/3) :=
by
  sorry

end ellipse_standard_eq_value_of_m_l404_404675


namespace total_students_at_competition_l404_404121

def KnowItAllHigh : ℕ := 50
def KarenHigh : ℕ := (3 / 5 : ℚ) * KnowItAllHigh
def CombinedSchools : ℕ := KnowItAllHigh + KarenHigh
def NovelCoronaHigh : ℕ := 2 * CombinedSchools
def TotalStudents := CombinedSchools + NovelCoronaHigh

theorem total_students_at_competition : TotalStudents = 240 := by
  sorry

end total_students_at_competition_l404_404121


namespace isosceles_triangle_angle_l404_404483

theorem isosceles_triangle_angle {x : ℝ} (hx0 : 0 < x) (hx1 : x < 90) (hx2 : 2 * x = 180 / 7) : x = 180 / 7 :=
sorry

end isosceles_triangle_angle_l404_404483


namespace abs_neg_2023_l404_404452

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l404_404452


namespace polynomial_nonnegative_sum_of_squares_l404_404787

noncomputable def polynomial_sum_of_squares (P : Polynomial ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ P.eval x) → ∃ Q R : Polynomial ℝ, P = Q^2 + R^2

theorem polynomial_nonnegative_sum_of_squares {P : Polynomial ℝ} (h : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 :=
begin
  exact polynomial_sum_of_squares P h,
end

end polynomial_nonnegative_sum_of_squares_l404_404787


namespace john_took_more_chickens_l404_404068

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l404_404068


namespace eight_digit_numbers_count_l404_404341

theorem eight_digit_numbers_count :
  let valid_digit (d : ℕ) := d = 1 ∨ d = 2 ∨ d = 3
  ∃ (count : ℕ), count = 32 ∧
  ∀ (number : Fin 8 → ℕ), (∀ i, valid_digit (number i)) ∧ 
                          (∀ i, i < 7 → (abs (number i - number (i+1)) = 1)) → count = 32 :=
by
  let valid_digit (d : ℕ) := d = 1 ∨ d = 2 ∨ d = 3
  let f (number : Fin 8 → ℕ) := ∀ i, valid_digit (number i) ∧ 
                                  (∀ i, i < 7 → (abs (number i - number (i+1)) = 1))
  have count := 32
  existsi count
  apply And.intro
  . rfl
  . sorry

end eight_digit_numbers_count_l404_404341


namespace nested_sqrt_eq_two_l404_404723

theorem nested_sqrt_eq_two (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 :=
sorry

end nested_sqrt_eq_two_l404_404723


namespace arrangement_exists_for_n_eq_3_n_must_be_odd_l404_404563

-- Statement 1: Prove that an arrangement that meets the conditions exists for n = 3.
theorem arrangement_exists_for_n_eq_3 :
  ∃ (arrangement : Finset (Finset ℕ)), 
    (∀ x ∈ arrangement, x.card = 3) ∧ 
    (∀ (x y : ℕ), x ≠ y → (∃! z ∈ arrangement, {x, y} ⊆ z)) :=
sorry

-- Statement 2: Prove that n is an odd number given the conditions.
theorem n_must_be_odd (n : ℕ) (hpos : 0 < n) (hcond : ∀ (x y : ℕ), x ≠ y → (∃! z, {x, y} ⊆ z)) :
  n % 2 = 1 :=
sorry

end arrangement_exists_for_n_eq_3_n_must_be_odd_l404_404563


namespace teacher_C_correct_l404_404831

-- Define the constants for the contestants and teachers
def Contestants := {1, 2, 3, 4, 5, 6}
def Teachers := {A, B, C, D}

-- Guesses by each teacher
def guess_A (i : Contestants) : Prop := (i = 3 ∨ i = 5)
def guess_B (i : Contestants) : Prop := (i ≠ 6)
def guess_C (i : Contestants) : Prop := (i ≠ 2 ∧ i ≠ 3 ∧ i ≠ 4)
def guess_D (i : Contestants) : Prop := (i = 1 ∨ i = 2 ∨ i = 4)

-- Condition that only one teacher's guess is correct
def only_one_correct_guess (winner : Contestants) : Prop :=
  (guess_A winner ∧ ¬guess_B winner ∧ ¬guess_C winner ∧ ¬guess_D winner) ∨
  (¬guess_A winner ∧ guess_B winner ∧ ¬guess_C winner ∧ ¬guess_D winner) ∨
  (¬guess_A winner ∧ ¬guess_B winner ∧ guess_C winner ∧ ¬guess_D winner) ∨
  (¬guess_A winner ∧ ¬guess_B winner ∧ ¬guess_C winner ∧ guess_D winner)

-- Proving that Teacher C guessed correctly
theorem teacher_C_correct : ∃ winner : Contestants, guess_C winner ∧ only_one_correct_guess winner := by
  sorry

end teacher_C_correct_l404_404831


namespace inequality_proof_l404_404023

theorem inequality_proof
  (n : ℕ) 
  (a : Fin n.succ → ℝ) 
  (h : ∀ i, 0 < a i) :
  (∑ i in Finset.range n, a i) * (∑ i in Finset.range n, a (i + 1)) 
  ≥ (∑ i in Finset.range n, (a i * a (i + 1)) / (a i + a (i + 1))) 
  * (∑ i in Finset.range n, a i + a (i + 1)) :=
by sorry

end inequality_proof_l404_404023


namespace solution_to_quadratic_inequality_l404_404988

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l404_404988


namespace bruce_goals_l404_404530

theorem bruce_goals (B M : ℕ) (h1 : M = 3 * B) (h2 : B + M = 16) : B = 4 :=
by {
  -- Omitted proof
  sorry
}

end bruce_goals_l404_404530


namespace rotated_legs_common_part_length_l404_404428

-- Define the problem and its parameters
variables {a b c x r : ℝ}

-- Theorems and axioms used in the proof statements
constant right_triangle : Prop
constant legs_rotated : Prop

-- Conditions
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : c = real.sqrt (a^2 + b^2)
axiom right_triangle_condition : right_triangle
axiom legs_rotated_condition : legs_rotated

-- Define the radius of the inscribed circle
def inscribed_circle_radius (a b c : ℝ) : ℝ := (a + b - c) / 2

-- Define the diameter of the inscribed circle
def inscribed_circle_diameter (a b c : ℝ) : ℝ := 2 * inscribed_circle_radius a b c

-- Actual Lean proof goal to be proven
theorem rotated_legs_common_part_length :
  right_triangle_condition ∧ legs_rotated_condition ∧ a_pos ∧ b_pos ∧ c_pos →
  inscribed_circle_diameter a b c = a + b - c :=
by
  sorry

end rotated_legs_common_part_length_l404_404428


namespace maximum_k_l404_404278

open Finset

def not_power_of_three (n : ℕ) : Prop :=
  ∀ k : ℕ, 3 ^ k ≠ n

theorem maximum_k (S : Finset ℕ) (hS : S = (range 243).erase 0) :
  ∃ T ⊆ S, T.card = 121 ∧ ∀ a b ∈ T, not_power_of_three (a + b) := sorry

end maximum_k_l404_404278


namespace sequence_contains_square_l404_404398

theorem sequence_contains_square (m : ℕ) (hm : m > 0) : ∃ k : ℕ, ∃ n : ℕ, (iterate (λ n, n + nat.sqrt n) k m) = n * n := 
sorry

end sequence_contains_square_l404_404398


namespace triangle_A2B2C2_has_orthocenter_H_2_l404_404393

theorem triangle_A2B2C2_has_orthocenter_H_2 
    (ABC A1 B1 C1 : Type) [Inhabited ABC] [Nonempty ABC]
    (H : Point ABC) [orthocenter H] 
    (on_BC : Point ABC → Bool) (on_CA : Point ABC → Bool) (on_AB : Point ABC → Bool)
    (containsH_A1B1C1 : orthocenter H (triangle A1 B1 C1))
    (A2 : Point ABC) (B2 : Point ABC) (C2 : Point ABC)
    (AH_B1C1 : line_intersect (line_through A H) (line_through B1 C1) A2)
    (BH_C1A1 : line_intersect (line_through B H) (line_through C1 A1) B2)
    (CH_A1B1 : line_intersect (line_through C H) (line_through A1 B1) C2)
  : orthocenter H (triangle A2 B2 C2) := sorry

end triangle_A2B2C2_has_orthocenter_H_2_l404_404393


namespace equal_area_triangle_pairs_l404_404379

theorem equal_area_triangle_pairs (A B C D E F O : Type) (S : ℝ)
  (H1 : AD ∧ BE ∧ CF are medians of triangle ABC)
  (H2 : AD ∧ BE ∧ CF intersect at O, the centroid of triangle ABC):
  ∃ n : ℕ, n = 33 := 
sorry

end equal_area_triangle_pairs_l404_404379


namespace find_certain_number_l404_404157

theorem find_certain_number :
    ∃ x : ℕ, 3 * 16 + 3 * 17 + 3 * 20 + x = 170 ∧ x = 11 :=
begin
    use 11,
    split,
    {
        norm_num,
    },
    {
        refl,
    }
end

end find_certain_number_l404_404157


namespace dorothy_spent_on_ingredients_l404_404982

/-- Dorothy's expenditure on doughnut ingredients can be determined from her sales and profit. -/
theorem dorothy_spent_on_ingredients
  (doughnuts : ℕ)
  (price_per_doughnut : ℕ)
  (profit : ℕ)
  (total_revenue := doughnuts * price_per_doughnut)
  (cost_of_ingredients := total_revenue - profit) :
  cost_of_ingredients = 53 :=
by
  have h1 : doughnuts = 25 := rfl
  have h2 : price_per_doughnut = 3 := rfl
  have h3 : profit = 22 := rfl
  have h4 : total_revenue = 25 * 3 := rfl
  have h5 : total_revenue = 75 := by rw [h4]
  have h6 : cost_of_ingredients = 75 - 22 := by rw [h5]
  have h7 : cost_of_ingredients = 53 := by rw [h6]
  exact h7

end dorothy_spent_on_ingredients_l404_404982


namespace Luka_water_requirement_l404_404033

-- Declare variables and conditions
variables (L S W O : ℕ)  -- All variables are natural numbers
-- Conditions
variable (h1 : S = 2 * L)  -- Twice as much sugar as lemon juice
variable (h2 : W = 5 * S)  -- 5 times as much water as sugar
variable (h3 : O = S)      -- Orange juice equals the amount of sugar 
variable (L_eq_5 : L = 5)  -- Lemon juice is 5 cups

-- The goal statement to prove
theorem Luka_water_requirement : W = 50 :=
by
  -- Note: The proof steps would go here, but as per instructions, we leave it as sorry.
  sorry

end Luka_water_requirement_l404_404033


namespace perpendicular_lines_m_l404_404732

theorem perpendicular_lines_m (m : ℝ) :
  (∀ (x y : ℝ), x - 2 * y + 5 = 0 → 2 * x + m * y - 6 = 0) →
  m = 1 :=
by
  sorry

end perpendicular_lines_m_l404_404732


namespace eccentricity_is_sqrt2_div_2_l404_404662

variable (a b c : ℝ) (h_a_b : a > b) (h_b_zero : b > 0) (c_eq_b : c = b)

def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def perpendicular_to_x_axis (x c : ℝ) : Prop := x = -c

def point_P_on_ellipse (P : ℝ × ℝ) := P = (-c, b^2 / a)

def point_A : ℝ × ℝ := (a, 0)

def point_B : ℝ × ℝ := (0, b)

def parallel_lines (P A B : ℝ × ℝ) : Prop :=
  let k_OP := (P.2 - 0) / (P.1 - 0)
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  k_AB = k_OP

def eccentricity (c a : ℝ) : ℝ := c / Real.sqrt (a^2 - c^2)

theorem eccentricity_is_sqrt2_div_2 (h1 : ellipse (-c) (b^2 / a) a b)
  (h2 : perpendicular_to_x_axis (-c) c)
  (h3 : parallel_lines (-c, b^2 / a) (a, 0) (0, b)) :
  eccentricity c a = (Real.sqrt 2) / 2 := 
sorry

end eccentricity_is_sqrt2_div_2_l404_404662


namespace impossible_to_position_arcs_l404_404627

theorem impossible_to_position_arcs 
  (S : Type) [metric_space S] [inhabited S] 
  (n : ℕ) 
  (arcs : fin n → set S) 
  (length : ∀ i, ennreal) 
  (h_intersect : ∀ i j, i ≠ j → disjoint (arcs i) (arcs j)) 
  (h_length_gt : ∀ i, ennreal.to_real (length i) > π + 2 * π / n) : 
  false :=
by
  sorry

end impossible_to_position_arcs_l404_404627


namespace train_length_approx_90_l404_404200

noncomputable def speed_in_m_per_s := (124 : ℝ) * (1000 / 3600)

noncomputable def time_in_s := (2.61269421026963 : ℝ)

noncomputable def length_of_train := speed_in_m_per_s * time_in_s

theorem train_length_approx_90 : abs (length_of_train - 90) < 1e-9 :=
  by
  sorry

end train_length_approx_90_l404_404200


namespace petya_vasya_meet_at_lamp_l404_404565

theorem petya_vasya_meet_at_lamp :
  ∃ (n : ℕ), n = 64 ∧
  -- Conditions
  (∃ (lamps : ℕ), lamps = 100) ∧
  (∃ (petya_start : ℕ), petya_start = 1) ∧
  (∃ (vasya_start : ℕ), vasya_start = 100) ∧
  (∃ (petya_position : ℕ), petya_position = 22) ∧
  (∃ (vasya_position : ℕ), vasya_position = 88) ∧
  -- They meet at lamp 64
  (petya_position + 42 = 64) ∧ 
  (vasya_position - 24 = 64) :=
begin
  -- Proof would go here
  sorry,
end

end petya_vasya_meet_at_lamp_l404_404565


namespace find_p_plus_q_l404_404510

-- Define the lengths of the sides of the triangle
def DE : ℝ := 6
def EF : ℝ := 10
def FD : ℝ := 8

-- Define the points D, E, F, and L with the properties given
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
noncomputable def F : Point := sorry
noncomputable def L : Point := sorry

-- Define the circles ω1 and ω2 with the properties given
noncomputable def ω1 : Circle := {
  center := sorry,
  radius := sorry,
  passes_through := E,
  tangent_to := FD
}

noncomputable def ω2 : Circle := {
  center := sorry,
  radius := sorry,
  passes_through := F,
  tangent_to := DE
}

-- Define the distances between points
def DL : ℝ := distance D L

-- The theorem to prove
theorem find_p_plus_q : DL = 24 / 5 ∧ 24 + 5 = 29 := by
  sorry

end find_p_plus_q_l404_404510


namespace jane_conference_distance_l404_404000

theorem jane_conference_distance :
  ∃ d : ℕ, 
  ∃ t : ℕ,
  let speed1 := 45,
      speed2 := 70,
      late_time := 90 / 60, -- converted 90 minutes late to hours
      early_time := 10 / 60 in
  d = speed1 * (t + late_time) ∧
  d = speed1 + speed2 * (t - early_time) ∧
  d = 290 := by
  sorry

end jane_conference_distance_l404_404000


namespace axis_of_symmetry_of_f_l404_404323

def f (x : ℝ) : ℝ := sqrt 3 * abs (sin (x / 2)) + abs (cos (x / 2))

theorem axis_of_symmetry_of_f : ∀ k : ℤ, axis_of_symmetry f (k * Real.pi) :=
sorry

end axis_of_symmetry_of_f_l404_404323


namespace alpha_perp_beta_l404_404313

variables {l m n : Type} [IsLine l] [IsLine m] [IsLine n]
variables {α β : Type} [IsPlane α] [IsPlane β]

-- Assuming l is a line contained within plane α
axiom l_in_alpha : ∃ l, l ⊆ α

-- Assuming l is parallel to m
axiom l_parallel_m : ∃ l m, l ∥ m

-- Assuming m is perpendicular to β
axiom m_perp_beta : ∃ m, m ⊥ β

-- To prove that α is perpendicular to β
theorem alpha_perp_beta (l m α β : Type) [IsLine l] [IsLine m] [IsPlane α] [IsPlane β]
  (l_in_alpha : l ⊆ α) 
  (l_parallel_m : l ∥ m) 
  (m_perp_beta : m ⊥ β) : α ⊥ β := sorry

end alpha_perp_beta_l404_404313


namespace cost_of_cheaper_feed_l404_404506

theorem cost_of_cheaper_feed 
  (weight_mix : ℝ := 17)
  (value_mix_per_pound : ℝ := 0.22)
  (weight_cheaper : ℝ := 12.2051282051)
  (value_expensive_per_pound : ℝ := 0.50) :
  let total_value := weight_mix * value_mix_per_pound,
      weight_expensive := weight_mix - weight_cheaper,
      total_value_expensive := weight_expensive * value_expensive_per_pound,
      total_value_cheaper := total_value - total_value_expensive,
      cost_cheaper_per_pound := total_value_cheaper / weight_cheaper
  in cost_cheaper_per_pound ≈ 0.11 :=
by
  sorry

end cost_of_cheaper_feed_l404_404506


namespace cleaning_time_if_anne_doubled_l404_404956

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l404_404956


namespace number_of_common_tangents_l404_404703

def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem number_of_common_tangents : ∃ n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, circleM x y → circleN x y → false) :=
by
  sorry

end number_of_common_tangents_l404_404703


namespace largest_possible_m_l404_404473

theorem largest_possible_m (q : ℕ → polynomial ℝ) (m : ℕ) :
  (x^12 - 1 = q(1) * q(2) * q(3) * q(4) * q(5) * q(6) ∧ 
  (∀ i : ℕ, q(i) ≠ 0 ∧ ∃ n : ℕ, q(i).degree > 0)) → m ≤ 6 :=
begin
  sorry
end

end largest_possible_m_l404_404473


namespace product_of_slopes_slope_of_l_l404_404689

variables {a b : ℝ}
variables {A B M : (ℝ × ℝ)}

-- Prove: The product of the slopes of the lines OM and l is -a^2/b^2
theorem product_of_slopes (h1 : a > b) (h2 : b > 0) (hA : (A.1^2 / b^2 + A.2^2 / a^2 = 1))
  (hB : (B.1^2 / b^2 + B.2^2 / a^2 = 1))
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  let k_l := (A.2 - B.2) / (A.1 - B.1),
      k_OM := M.2 / M.1 in
  k_l * k_OM = - (a^2 / b^2) :=
sorry

-- Prove: If l passes through (b, a), slope of l is (4 ± √7)/3 * (a/b)
theorem slope_of_l (h1 : a > b) (h2 : b > 0) (h_l : B = (b, a))
  (h_intersect : ∃ P : (ℝ × ℝ), -- there exists a point of intersection P on OM extended to the ellipse
    (P.1^2 / b^2 + P.2^2 / a^2 = 1) ∧
    (M = ((O.1 + P.1) / 2, (O.2 + P.2) / 2))) :
  let k := (4 + sqrt 7) / 3 * (a / b) in
  -- Note: Handling the other possible slope would require another proof; omitted for brevity
  (A.2 - B.2) / (A.1 - B.1) = k :=
sorry

end product_of_slopes_slope_of_l_l404_404689


namespace sin_product_value_l404_404090

noncomputable def sin_product := ∏ (n : ℕ) in finset.range 45, real.sin (2 * (n + 1) * real.pi / 180)

theorem sin_product_value : sin_product = 192 * real.sqrt 5 / 2 ^ 50 :=
by
  sorry

end sin_product_value_l404_404090


namespace partition_into_57_groups_l404_404083

def Dreamland := Fin 2016

def flights (G : Dreamland → Dreamland) : Prop :=
  ∀ v : Dreamland, ∃! w : Dreamland, G v = w

theorem partition_into_57_groups (G : Dreamland → Dreamland) (h : flights G) :
  ∃ (groups : Fin 57 → set Dreamland),
  (∀ i : Fin 57, ∀ u v ∈ groups i, ∀ n ≤ 28, (G^[n]) u ≠ v) :=
sorry

end partition_into_57_groups_l404_404083


namespace difference_of_numbers_l404_404534

-- Definitions and conditions
def ratio_condition (a b : ℕ) : Prop := a / b = 2 / 3
def sum_of_cubes_condition (a b : ℕ) : Prop := a^3 + b^3 = 945

-- Objective: Prove the difference is 3
theorem difference_of_numbers (a b : ℕ) (h1 : ratio_condition a b) (h2 : sum_of_cubes_condition a b) :
  |a - b| = 3 :=
by
  sorry

end difference_of_numbers_l404_404534


namespace not_inscribed_if_black_vertices_exceed_half_l404_404057

theorem not_inscribed_if_black_vertices_exceed_half
  (V : Type) [Fintype V]
  (E : V → V → Prop)
  [DecidableRel E]
  (Black : V → Prop)
  (White : V → Prop)
  (convex_polyhedron : Prop)
  (H1 : ∀ v : V, Black v ∨ White v)
  (H2 : ∀ v w : V, E v w → White v ∨ White w)
  (H3 : Fintype.card {v : V | Black v} > Fintype.card V / 2) :
  ¬ (∃ R : V → Prop, (∀ v : V, E v v) → convex_polyhedron) := sorry

end not_inscribed_if_black_vertices_exceed_half_l404_404057


namespace shekar_marks_math_l404_404054

theorem shekar_marks_math (M : ℝ) : 
  let sci := 65
  let soc := 82
  let eng := 62
  let bio := 85
  let avg := 74
  let num_subjects := 5
in avg = (M + sci + soc + eng + bio) / num_subjects -> M = 76 :=
by
  let sci := 65
  let soc := 82
  let eng := 62
  let bio := 85
  let avg := 74
  let num_subjects := 5
  assume h : avg = (M + sci + soc + eng + bio) / num_subjects
  sorry

end shekar_marks_math_l404_404054


namespace remaining_figure_area_l404_404145

-- Definitions based on conditions
def original_semi_circle_radius_from_chord (L : ℝ) : ℝ := L / 2

def area_of_semi_circle (r : ℝ) : ℝ := (π * r^2) / 2

def remaining_area (L : ℝ) : ℝ :=
  let R := original_semi_circle_radius_from_chord L
  in area_of_semi_circle R - 2 * area_of_semi_circle (R / 2)

-- Given problem rewritten in Lean
theorem remaining_figure_area (hL : 8) : 
  abs (remaining_area 8 - 12.57) < 0.01 := by
  sorry

end remaining_figure_area_l404_404145


namespace mike_profit_l404_404801

theorem mike_profit 
  (num_acres_bought : ℕ) (price_per_acre_buy : ℤ) 
  (fraction_sold : ℚ) (price_per_acre_sell : ℤ) :
  num_acres_bought = 200 →
  price_per_acre_buy = 70 →
  fraction_sold = 1/2 →
  price_per_acre_sell = 200 →
  let cost_of_land := price_per_acre_buy * num_acres_bought,
      num_acres_sold := (fraction_sold * num_acres_bought),
      revenue_from_sale := price_per_acre_sell * num_acres_sold,
      profit := revenue_from_sale - cost_of_land
  in profit = 6000 := by
  intros h1 h2 h3 h4
  let cost_of_land := price_per_acre_buy * num_acres_bought
  let num_acres_sold := (fraction_sold * num_acres_bought)
  let revenue_from_sale := price_per_acre_sell * num_acres_sold
  let profit := revenue_from_sale - cost_of_land
  rw [h1, h2, h3, h4]
  sorry

end mike_profit_l404_404801


namespace rational_solution_cos_eq_l404_404446

theorem rational_solution_cos_eq {q : ℚ} (h0 : 0 < q) (h1 : q < 1) (heq : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2 / 3 := 
sorry

end rational_solution_cos_eq_l404_404446


namespace Venki_speed_from_X_to_Z_l404_404516

theorem Venki_speed_from_X_to_Z :
  ∀ (x z : Type)
  (distance_ZY time_Y distance_XZ : ℝ)
  (time_XZ : ℝ),
  (distance_ZY = 45 * 4.444444444444445) →
  (distance_XZ = 2 * distance_ZY) →
  (time_XZ = 5) →
  (Venki_speed := distance_XZ / time_XZ) →
  Venki_speed = 80 :=
by
  sorry

end Venki_speed_from_X_to_Z_l404_404516


namespace sum_of_distinct_prime_factors_l404_404272

theorem sum_of_distinct_prime_factors (a b c : ℕ) (h1 : a = 7^4 - 7^2) (h2 : b = 2) (h3 : c = 3) (h4 : 2 + 3 + 7 = 12): 
  ∃ d : ℕ, a.prime_factors.sum = d ∧ d = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l404_404272


namespace ironed_clothing_l404_404444

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l404_404444


namespace length_PQ_l404_404402

-- Representation of a point in 2D
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions
def R : Point := ⟨9, 7⟩

def is_midpoint (R P Q : Point) : Prop :=
  R.x = (P.x + Q.x) / 2 ∧ R.y = (P.y + Q.y) / 2

def on_line_1 (P : Point) : Prop :=
  9 * P.y = 14 * P.x

def on_line_2 (Q : Point) : Prop :=
  11 * Q.y = 4 * Q.x

def distance (P Q : Point) : ℚ :=
  real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

-- The statement of the theorem
theorem length_PQ : 
  ∀ (P Q : Point), 
  is_midpoint R P Q → on_line_1 P → on_line_2 Q → 
  distance P Q = 60 / 7 :=
by
  sorry

end length_PQ_l404_404402


namespace parallel_vectors_l404_404030

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_b : b = (-1, m)) (h_parallel : ∃ k : ℝ, b = k • a) : m = 2 :=
by {
  sorry
}

end parallel_vectors_l404_404030


namespace expected_value_correct_l404_404657

-- Define the problem conditions
def num_balls : ℕ := 5

def prob_swapped_twice : ℚ := (2 / 25)
def prob_never_swapped : ℚ := (9 / 25)
def prob_original_position : ℚ := prob_swapped_twice + prob_never_swapped

-- Define the expected value calculation
def expected_num_in_original_position : ℚ :=
  num_balls * prob_original_position

-- Claim: The expected number of balls that occupy their original positions after two successive transpositions is 2.2.
theorem expected_value_correct :
  expected_num_in_original_position = 2.2 :=
sorry

end expected_value_correct_l404_404657


namespace binomial_12_10_eq_66_l404_404615

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404615


namespace decreasing_function_l404_404693

def f (a x : ℝ) : ℝ := a * x^3 - x

theorem decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x < y → f a y ≤ f a x) : a ≤ 0 :=
by
  sorry

end decreasing_function_l404_404693


namespace cannot_cover_3x5_l404_404889

noncomputable theory

def is_even (n : ℕ) : Prop := n % 2 = 0

def even_covered (m n : ℕ) : Prop :=
  is_even (m * n)

theorem cannot_cover_3x5 : ¬ even_covered 3 5 :=
by unfold even_covered; unfold is_even; simp; exact dec_trivial

end cannot_cover_3x5_l404_404889


namespace binom_12_10_eq_66_l404_404597

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404597


namespace Felipe_time_l404_404123

theorem Felipe_time (together_years : ℝ) (felipe_ratio : ℝ) : 
  together_years = 7.5 → felipe_ratio = 0.5 → 
  (F : ℝ), 12 * F = 30 :=
by
  intro h_together h_ratio
  let F := 2.5
  sorry

end Felipe_time_l404_404123


namespace derivative_of_f_l404_404266

noncomputable def f (x : ℝ) : ℝ :=
  x - log (1 + exp x) - 2 * exp (-x / 2) * arctan (exp (x / 2)) - (arctan (exp (x / 2)))^2

noncomputable def f' (x : ℝ) : ℝ :=
  arctan (exp (x / 2)) / (exp (x / 2) * (1 + exp x))

theorem derivative_of_f (x : ℝ) : (deriv f x) = f' x :=
by
  sorry

end derivative_of_f_l404_404266


namespace part_a_part_b_part_c_part_d_l404_404051

variables {n : ℕ} {x : Fin n → ℝ}

-- Condition: All x_i are positive
axiom pos_xi : ∀ i, 0 < x i

-- Part a
theorem part_a : n * (∑ i, x i) ≥ (∑ i, real.sqrt (x i)) ^ 2 := sorry

-- Part b
theorem part_b : (n^3 : ℝ) / (∑ i, x i) ^ 2 ≤ ∑ i, (1 / (x i) ^ 2) := sorry

-- Part c
theorem part_c : n * (∏ i, x i) ≤ ∑ i, (x i) ^ n := sorry

-- Part d (Minkowski Inequality)
theorem part_d : (∑ i, x i) * (∑ i, 1 / (x i)) ≥ (n^2 : ℝ) := sorry

end part_a_part_b_part_c_part_d_l404_404051


namespace dreamland_partition_l404_404081

theorem dreamland_partition :
  ∃ k : ℕ, (∀ (G : Type) [fintype G] [graph G (λ v, v)],
    (∀ (v : G), ∃! w : G, graph.reachable v w) →
    (∃ P : finset (finset G), P.card = k ∧ ∀ A ∈ P, ∀ v w ∈ A, ¬graph.reachable v w → false)) ∧ k = 57 :=
begin
  sorry
end

end dreamland_partition_l404_404081


namespace root_of_quadratic_l404_404682

noncomputable def imaginary_unit : ℂ := complex.I

theorem root_of_quadratic (p q : ℝ) (h1 : is_root (λ x : ℂ, 2 * x^2 + (p : ℂ) * x + (q : ℂ)) (-2 * imaginary_unit - 3)) : 
  p - q = -14 := 
sorry

end root_of_quadratic_l404_404682


namespace fraction_draw_l404_404116

/-
Theorem: Given the win probabilities for Amy, Lily, and Eve, the fraction of the time they end up in a draw is 3/10.
-/

theorem fraction_draw (P_Amy P_Lily P_Eve : ℚ) (h_Amy : P_Amy = 2/5) (h_Lily : P_Lily = 1/5) (h_Eve : P_Eve = 1/10) : 
  1 - (P_Amy + P_Lily + P_Eve) = 3 / 10 := by
  sorry

end fraction_draw_l404_404116


namespace suff_but_not_necc_l404_404149

-- A noncomputable theory to include all necessary components
noncomputable theory

-- Variables used in the conditions and statement
variables {x : ℝ}

-- Definition of the conditions
def condition1 : Prop := x > 3
def condition2 : Prop := x^2 > 9

-- The Lean 4 statement proving the conditions and equivalence
theorem suff_but_not_necc : (condition1 → condition2) ∧ (¬condition2 → condition1) ↔ false :=
by {
  sorry
}

end suff_but_not_necc_l404_404149


namespace a_14_pow_14_eq_1_l404_404773

noncomputable def a : ℕ → ℕ 
| 1 := 11^11
| 2 := 12^12
| 3 := 13^13
| n := if n ≥ 4 then |a (n-1) - a (n-2)| + |a (n-2) - a (n-3)| else 0

theorem a_14_pow_14_eq_1 : a (14^14) = 1 := 
sorry

end a_14_pow_14_eq_1_l404_404773


namespace find_max_marks_l404_404893

theorem find_max_marks (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 := 
by
  sorry

end find_max_marks_l404_404893


namespace coverage_of_parallelogram_by_circles_l404_404851

theorem coverage_of_parallelogram_by_circles 
  (a α : ℝ) 
  (h_triangle_acute : ∀ {A B C : Type}, is_acute_triangle A B C)
  (h_cos_sin_range : ∀ x, (cos x) ^ 2 + (sin x) ^ 2 = 1) :
  a ≤ cos α + sqrt 3 * sin α :=
sorry

end coverage_of_parallelogram_by_circles_l404_404851


namespace binary_to_hexadecimal_l404_404244

theorem binary_to_hexadecimal (bin : String) (dec : ℕ) (hex : String) : 
  bin = "1010011" → 
  dec = 83 → 
  hex = "53" → 
  (convertToDecimal bin = dec) ∧ (convertToHexadecimal dec = hex) :=
by
  sorry

def convertToDecimal (bin : String) : ℕ :=
  let chars := bin.data.reverse.map (fun c => c.toNat - '0'.toNat)
  chars.enum.map (fun (i, b) => b * (2 ^ i)).sum

def convertToHexadecimal (dec : ℕ) : String :=
  if dec = 0 then "0" else
  let hexDigits := "0123456789ABCDEF"
  let rec go (dec : ℕ) : List Char :=
    if dec = 0 then [] else
    let (q, r) := divMod dec 16
    hexDigits.get! r :: go q
  go dec |> List.reverse |> String

end binary_to_hexadecimal_l404_404244


namespace consecutive_integer_sum_to_18_l404_404342

theorem consecutive_integer_sum_to_18 : 
  {n : ℕ // n ≥ 2} → 
  ∃ s : Set (Finset ℕ), (∀ f ∈ s, f.sum id = 18 ∧ ∃ l, (∀ x ∈ f, ∃ i, x = l + i ∧ i < f.card)) → 
  s.card = 2 :=
by
  sorry

end consecutive_integer_sum_to_18_l404_404342


namespace range_of_f_l404_404878

open Set

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f : range f = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_f_l404_404878


namespace all_a_n_are_integers_l404_404898

variable (a : ℕ → ℤ)
hypothesis (h1 : a 1 ∈ ℤ)
hypothesis (recurrence_relation : ∀ n, a (n + 1) = (n + 2) * (a n - 1) / n)

theorem all_a_n_are_integers : ∀ n ≥ 1, a n ∈ ℤ := 
by
  sorry

end all_a_n_are_integers_l404_404898


namespace cost_of_song_book_l404_404807

def cost_of_trumpet : ℝ := 145.16
def total_amount_spent : ℝ := 151.00

theorem cost_of_song_book : (total_amount_spent - cost_of_trumpet) = 5.84 := by
  sorry

end cost_of_song_book_l404_404807


namespace cookie_percentage_increase_l404_404220

theorem cookie_percentage_increase (cookies_Monday cookies_Tuesday cookies_Wednesday total_cookies : ℕ) 
  (h1 : cookies_Monday = 5)
  (h2 : cookies_Tuesday = 2 * cookies_Monday)
  (h3 : total_cookies = cookies_Monday + cookies_Tuesday + cookies_Wednesday)
  (h4 : total_cookies = 29) :
  (100 * (cookies_Wednesday - cookies_Tuesday) / cookies_Tuesday = 40) := 
by
  sorry

end cookie_percentage_increase_l404_404220


namespace last_digit_2_1992_last_digit_3_1992_last_digit_sum_l404_404469

theorem last_digit_2_1992 : Nat.mod (2^1992) 10 = 6 := by
  sorry

theorem last_digit_3_1992 : Nat.mod (3^1992) 10 = 1 := by
  sorry

theorem last_digit_sum :
  Nat.mod (2^1992 + 3^1992) 10 = 7 := by
  have h1 : Nat.mod (2^1992) 10 = 6 := last_digit_2_1992
  have h2 : Nat.mod (3^1992) 10 = 1 := last_digit_3_1992
  rw [h1, h2]
  -- Now we just need to handle the last digit of their sum
  exact Eq.symm (Nat.mod_add _ _ 10) ▸ by
    simp
    exact Eq.symm rfl

end last_digit_2_1992_last_digit_3_1992_last_digit_sum_l404_404469


namespace solution_set_ineq_l404_404264

theorem solution_set_ineq (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ 5 < x ∧ x ≤ 13 / 2 :=
sorry

end solution_set_ineq_l404_404264


namespace ratio_proof_l404_404319

variable (a b c d : ℚ)

-- Given conditions
axiom h1 : b / a = 3
axiom h2 : c / b = 4
axiom h3 : d = 5 * b

-- Theorem to be proved
theorem ratio_proof : (a + b + d) / (b + c + d) = 19 / 30 := 
by 
  sorry

end ratio_proof_l404_404319


namespace product_of_three_greater_than_two_or_four_of_others_l404_404309

theorem product_of_three_greater_than_two_or_four_of_others 
  (x : Fin 10 → ℕ) 
  (h_unique : ∀ i j : Fin 10, i ≠ j → x i ≠ x j) 
  (h_positive : ∀ i : Fin 10, 0 < x i) : 
  ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ a b : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ b ≠ i ∧ b ≠ j ∧ b ≠ k → 
      x i * x j * x k > x a * x b) ∨ 
    (∀ a b c d : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ 
      b ≠ i ∧ b ≠ j ∧ b ≠ k ∧ 
      c ≠ i ∧ c ≠ j ∧ c ≠ k ∧ 
      d ≠ i ∧ d ≠ j ∧ d ≠ k → 
      x i * x j * x k > x a * x b * x c * x d) := sorry

end product_of_three_greater_than_two_or_four_of_others_l404_404309


namespace math_problem_log_equality_l404_404541

theorem math_problem_log_equality :
  (3/4) * real.log10 25 + 2 ^ (real.log 3 / real.log 2) + real.log10 (2 * real.sqrt 2) = 9/2 :=
sorry

end math_problem_log_equality_l404_404541


namespace oranges_to_apples_l404_404765

noncomputable def weight_relation : Prop :=
∀ (oranges pears apples : ℕ),
  (9 * oranges = 6 * pears) →
  (2 * pears = 3 * apples) →
  45 * oranges = 45 * apples

theorem oranges_to_apples : weight_relation :=
by
  intros oranges pears apples h1 h2,
  sorry

end oranges_to_apples_l404_404765


namespace length_of_EC_l404_404542

variable (AB CD AC EC : ℝ)
variable (A B C D E : Type)

axiom is_trapezoid : ∥AB∥ = 3 * ∥CD∥
axiom parallel_bases : ∥AB∥ ∥ ∥CD∥ 
axiom intersection_point : E = (diagonal_intersection A C B D)
axiom diagonal_length : ∥AC∥ = 15

theorem length_of_EC 
  (h1 : ∥AB∥ = 3 * ∥CD∥)
  (h2 : parallel ∥AB∥ ∥CD∥)
  (h3 : E = diagonal_intersection ∥A, C, B, D∥)
  (h4 : ∥AC∥ = 15) 
  : ∥EC∥ = 5 := 
sorry

end length_of_EC_l404_404542


namespace function_continuous_at_x0_l404_404432

noncomputable def delta (ε : ℝ) : ℝ := ε / 36

theorem function_continuous_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 6| < δ → |(3 * x^2 + 7) - 115| < ε :=
by
  -- The following proof list is provided for context and will be replaced by the actual proof using Lean commands.
  -- sorry will be used to indicate the proof is omitted.
  exact sorry

end function_continuous_at_x0_l404_404432


namespace remaining_figure_area_l404_404144

-- Definitions based on conditions
def original_semi_circle_radius_from_chord (L : ℝ) : ℝ := L / 2

def area_of_semi_circle (r : ℝ) : ℝ := (π * r^2) / 2

def remaining_area (L : ℝ) : ℝ :=
  let R := original_semi_circle_radius_from_chord L
  in area_of_semi_circle R - 2 * area_of_semi_circle (R / 2)

-- Given problem rewritten in Lean
theorem remaining_figure_area (hL : 8) : 
  abs (remaining_area 8 - 12.57) < 0.01 := by
  sorry

end remaining_figure_area_l404_404144


namespace quadratic_polynomial_has_equal_roots_l404_404281

variable {R : Type*} [field R]

theorem quadratic_polynomial_has_equal_roots
  (a b c l t v : R)
  (f : R → R)
  (h1 : f l = t + v)
  (h2 : f t = l + v)
  (h3 : f v = l + t)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
: t = v ∨ l = t ∨ l = v := 
sorry

end quadratic_polynomial_has_equal_roots_l404_404281


namespace find_number_l404_404353

theorem find_number (x number : ℝ) (h₁ : 5 - (5 / x) = number + (4 / x)) (h₂ : x = 9) : number = 4 :=
by
  subst h₂
  -- proof steps
  sorry

end find_number_l404_404353


namespace similar_triangles_l404_404130

/-- 
   Given two circles k1 and k2 intersect at points A and B. 
   A line through B intersects k1 again at C and k2 again at E.
   Another line through B intersects k1 again at D and k2 again at F.
   Assuming B is between C and E, and between D and F.
   M and N are the midpoints of CE and DF respectively.
   Prove that triangles ACD, AEF, and AMN are similar to each other.
--/
theorem similar_triangles
  (k₁ k₂ : Circle)
  (A B C D E F M N : Point)
  (h₁ : k₁ ∩ k₂ = {A, B})
  (h₂ : B ∈ Line_through B C)
  (h₃ : B ∈ Line_through B E)
  (h₄ : B ∈ Line_through B D)
  (h₅ : B ∈ Line_through B F)
  (h₆ : B ∈ Segment C E)
  (h₇ : B ∈ Segment D F)
  (h₈ : M = midpoint C E)
  (h₉ : N = midpoint D F) :
  similar (triangle A C D) (triangle A E F) ∧ 
  similar (triangle A M N) (triangle A C D) ∧
  similar (triangle A M N) (triangle A E F) :=
by
  sorry

end similar_triangles_l404_404130


namespace area_triangle_DEF_l404_404759

theorem area_triangle_DEF 
  (DE EL EF : ℝ) (H1 : DE = 15) (H2 : EL = 12) (H3 : EF = 20) 
  (DL : ℝ) (H4 : DE^2 = EL^2 + DL^2) (H5 : DL * EF = DL * 20) :
  1/2 * EF * DL = 90 :=
by
  -- Use the assumptions and conditions to state the theorem.
  sorry

end area_triangle_DEF_l404_404759


namespace number_of_solutions_l404_404009

theorem number_of_solutions (m : ℕ) (h : m > 1) :
  {x : ℕ | ⌊x / m⌋ = ⌊x / (m - 1)⌋}.to_finset.card = (m * (m - 1)) / 2 :=
by
  sorry

end number_of_solutions_l404_404009


namespace shaded_area_correct_l404_404053

-- Define points as vectors in the 2D plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points K, L, M, J based on the given coordinates.
def K : Point := {x := 0, y := 0}
def L : Point := {x := 5, y := 0}
def M : Point := {x := 5, y := 6}
def J : Point := {x := 0, y := 6}

-- Define intersection point N based on the equations of lines.
def N : Point := {x := 2.5, y := 3}

-- Define the function to calculate area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Define the function to calculate area of a triangle.
def triangle_area (b h : ℝ) : ℝ :=
  0.5 * b * h

-- Compute total shaded area according to the problem statement.
def shaded_area (K L M J N : Point) : ℝ :=
  trapezoid_area 5 2.5 3 + triangle_area 2.5 1

theorem shaded_area_correct : shaded_area K L M J N = 12.5 := by
  sorry

end shaded_area_correct_l404_404053


namespace prove_math_problem_l404_404295

noncomputable def math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : Prop :=
  (x + y = 1) ∧ (x^5 + y^5 = 11)

theorem prove_math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : math_problem x y h1 h2 h3 :=
  sorry

end prove_math_problem_l404_404295


namespace mrs_hilt_travel_distance_l404_404808

theorem mrs_hilt_travel_distance :
  let distance_water_fountain := 30
  let distance_main_office := 50
  let distance_teacher_lounge := 35
  let trips_water_fountain := 4
  let trips_main_office := 2
  let trips_teacher_lounge := 3
  (distance_water_fountain * trips_water_fountain +
   distance_main_office * trips_main_office +
   distance_teacher_lounge * trips_teacher_lounge) = 325 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_travel_distance_l404_404808


namespace proof_A_intersection_C_U_B_l404_404700

open Set

-- Given sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Prove that the intersection of A and C_U_B is {2, 3}
theorem proof_A_intersection_C_U_B :
  A ∩ C_U_B = {2, 3} := by
  sorry

end proof_A_intersection_C_U_B_l404_404700


namespace arithmetic_sequence_common_difference_l404_404374

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ)
  (h_a4 : a₁ + 3 * d = -2)
  (h_sum : 10 * a₁ + 45 * d = 65) :
  d = 17 / 3 :=
sorry

end arithmetic_sequence_common_difference_l404_404374


namespace carlos_marbles_l404_404962

theorem carlos_marbles :
  ∃ N : ℕ, 
    (N % 9 = 2) ∧ 
    (N % 10 = 2) ∧ 
    (N % 11 = 2) ∧ 
    (N > 1) ∧ 
    N = 992 :=
by {
  -- We need this for the example; you would remove it in a real proof.
  sorry
}

end carlos_marbles_l404_404962


namespace min_area_bounded_by_tangents_parabola_l404_404395

theorem min_area_bounded_by_tangents_parabola (a b : ℝ) (h : a < b) :
  ∃ C : ℝ × ℝ, 
    (C = ((a + b) / 2, a * b) ∧ 
     C.2 = (1 / 2) * C.1 ^ 2 - C.1 - 2 ∧ 
     (∫ x in a..(a + b) / 2, (x ^ 2 - (2 * a * x - a ^ 2)) dx + 
      ∫ x in (a + b) / 2..b, (x ^ 2 - (2 * b * x - b ^ 2)) dx) =
     ((b - a) ^ 3) / 12) :=
begin
  sorry -- Proof is omitted as per instructions
end

end min_area_bounded_by_tangents_parabola_l404_404395


namespace max_b_c_in_triangle_l404_404384

noncomputable def max_sum_of_sides (a A : ℝ) (B C : ℝ) (b c : ℝ) :=
  a = Real.sqrt 3 ∧ A = 2 * Real.pi / 3 ∧ B + C = Real.pi / 3 ∧
  (∀ B C, b = 2 * Real.sin B ∧ c = 2 * Real.sin C → b + c = 2 * Real.sin (B + Real.pi / 3)) →
  b + c ≤ 2

theorem max_b_c_in_triangle (a := Real.sqrt 3) (A := 2 * Real.pi / 3) :
  ∃ B C b c, max_sum_of_sides a A B C b c :=
begin
  sorry
end

end max_b_c_in_triangle_l404_404384


namespace mrs_sheridan_cats_l404_404809

theorem mrs_sheridan_cats (initial_cats : ℝ) (given_away_cats : ℝ) (remaining_cats : ℝ) :
  initial_cats = 17.0 → given_away_cats = 14.0 → remaining_cats = (initial_cats - given_away_cats) → remaining_cats = 3.0 :=
by
  intros
  sorry

end mrs_sheridan_cats_l404_404809


namespace analytical_expression_of_f_sin_2_alpha_value_l404_404795

-- Definitions and conditions
def f (x : ℝ) := sin (ω * x + φ)
axiom ω_pos : ω > 0
axiom φ_range : 0 < φ ∧ φ < π
axiom axis_distance : ∀ x, f (x + (π / 2)) = f (x - (π / 2))
axiom f_even : ∀ x, f (x + (π / 2)) = f (-(x + (π / 2)))

-- Question 1: Find the analytical expression for f(x)
theorem analytical_expression_of_f : (∀ x, f x = cos (2 * x)) := sorry

-- Additional condition for Question 2
axiom α_acute : ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ f ((α/2) + (π / 12)) = 3 / 5

-- Question 2: Find the value of sin 2α
theorem sin_2_alpha_value : ∃ α : ℝ, 0 < α ∧ α < π / 2 → sin (2 * α) = (24 + 7 * sqrt 3) / 50 := sorry

end analytical_expression_of_f_sin_2_alpha_value_l404_404795


namespace max_sequence_is_ten_l404_404873

noncomputable def max_int_sequence_length : Prop :=
  ∀ (a : ℕ → ℤ), 
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) > 0) ∧
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) < 0) →
    (∃ n ≤ 10, ∀ i ≥ n, a i = 0)

theorem max_sequence_is_ten : max_int_sequence_length :=
sorry

end max_sequence_is_ten_l404_404873


namespace exists_ten_digit_number_divisible_by_11_l404_404637

open Nat

def is_ten_digit_number (n : ℕ) : Prop :=
  let digits := [9, 5, 7, 6, 8, 4, 3, 2, 1, 0] in
  let ds := Nat.digits 10 n in
  (ds.length = 10) ∧ (List.perm ds digits) 

def is_divisible_by_11 (n : ℕ) : Prop := 
  let ds := Nat.digits 10 n in
  (ds.enum.filterMap (λ ⟨i, d⟩ => if i % 2 = 0 then some d else none)).sum -
  (ds.enum.filterMap (λ ⟨i, d⟩ => if i % 2 = 1 then some d else none)).sum ≡ 0 [MOD 11]

theorem exists_ten_digit_number_divisible_by_11 :
  ∃ n : ℕ, is_ten_digit_number n ∧ is_divisible_by_11 n :=
  sorry

end exists_ten_digit_number_divisible_by_11_l404_404637


namespace find_factors_l404_404888

theorem find_factors :
  ∃ x y z : ℕ, 
    x = y + 10 ∧ 
    xy = z + 40 ∧ 
    z = 39y + 22 ∧ 
    x = 41 ∧ 
    y = 31 :=
by {
  -- Proof steps here
  sorry
}

end find_factors_l404_404888


namespace rectangular_solid_circumscribed_sphere_volume_l404_404557

noncomputable def max_circumscribed_sphere_volume : ℝ := 4 * real.sqrt 3 * real.pi

theorem rectangular_solid_circumscribed_sphere_volume :
  ∃ (x : ℝ) (h : ℝ), (4 * x + 4 * h = 24) ∧ 
  (∀ (y : ℝ), 0 < y ∧ y < 3 → 
    (2 * real.sqrt (y^2 + y^2 + (6 - 2*y)^2)) = 2 * real.sqrt 3) ∧ 
    (∀ (r : ℝ), r = real.sqrt 3 → 
      (4/3) * real.pi * r^3 = max_circumscribed_sphere_volume) :=
sorry

end rectangular_solid_circumscribed_sphere_volume_l404_404557


namespace hourly_rate_is_7_l404_404811

-- Define the fixed fee, the total payment, and the number of hours
def fixed_fee : ℕ := 17
def total_payment : ℕ := 80
def num_hours : ℕ := 9

-- Define the function calculating the hourly rate based on the given conditions
def hourly_rate (fixed_fee total_payment num_hours : ℕ) : ℕ :=
  (total_payment - fixed_fee) / num_hours

-- Prove that the hourly rate is 7 dollars per hour
theorem hourly_rate_is_7 :
  hourly_rate fixed_fee total_payment num_hours = 7 := 
by 
  -- proof is skipped
  sorry

end hourly_rate_is_7_l404_404811


namespace incorrect_statement_l404_404890

theorem incorrect_statement :
  ∀ (a b d R : ℝ) (x y z : ℕ),
    (a < 0 ∧ b < 0 ∧ |a| > |b| → a < b) ∧
    (¬ (π * d + 2 * π * R is quadratic binomial)) where
      legally_binomial (d R : ℝ) : Prop := d * d = 0 ∧ R * R = 0 :=
      ∃ c : ℝ, d + 2 * R != c :=
      ∀ (term : ℝ), (term = -2 → true ∧ (1 + 1 + 3 = 5 → true)).
{
  sorry
}

end incorrect_statement_l404_404890


namespace exponent_of_term_on_right_side_l404_404377

theorem exponent_of_term_on_right_side
  (s m : ℕ) 
  (h1 : (2^16) * (25^s) = 5 * (10^m))
  (h2 : m = 16) : m = 16 := 
by
  sorry

end exponent_of_term_on_right_side_l404_404377


namespace car_drive_time_60_kmh_l404_404548

theorem car_drive_time_60_kmh
  (t : ℝ)
  (avg_speed : ℝ := 80)
  (dist_speed_60 : ℝ := 60 * t)
  (time_speed_90 : ℝ := 2 / 3)
  (dist_speed_90 : ℝ := 90 * time_speed_90)
  (total_distance : ℝ := dist_speed_60 + dist_speed_90)
  (total_time : ℝ := t + time_speed_90)
  (avg_speed_eq : avg_speed = total_distance / total_time) :
  t = 1 / 3 := 
sorry

end car_drive_time_60_kmh_l404_404548


namespace total_interest_rate_is_correct_l404_404213

theorem total_interest_rate_is_correct :
  let total_investment := 100000
  let interest_rate_first := 0.09
  let interest_rate_second := 0.11
  let invested_in_second := 29999.999999999993
  let invested_in_first := total_investment - invested_in_second
  let interest_first := invested_in_first * interest_rate_first
  let interest_second := invested_in_second * interest_rate_second
  let total_interest := interest_first + interest_second
  let total_interest_rate := (total_interest / total_investment) * 100
  total_interest_rate = 9.6 :=
by
  sorry

end total_interest_rate_is_correct_l404_404213


namespace composite_body_volume_ratio_l404_404161

noncomputable def volume_ratio (α : Real) : Real :=
  2 * (Real.sin (2 * α))^2 * (Real.cos (α + Real.pi / 6)) * (Real.cos (α - Real.pi / 6))

theorem composite_body_volume_ratio (α : Real) :
  let r := 1 -- Assuming radius 1 for simplicity
  let V_cylinder_cones := (Real.pi * r^3 * Real.tan α) / 3 + Real.pi * r^3 * Real.cot (2 * α)
  let V_hemisphere := (2 / 3) * Real.pi * r^3
  V_cylinder_cones / V_hemisphere = volume_ratio α :=
sorry

end composite_body_volume_ratio_l404_404161


namespace main_statement_l404_404404

-- Define vectors a, b, c as non-zero vectors
variables {a b c : Type} [NonZero a] [NonZero b] [NonZero c]

-- Define dot product and parallel relations
variables (dot : a → b → ℝ) (parallel : a → b → Prop)

-- Define the propositions p and q
def p : Prop := dot a b = 0 ∧ dot b c = 0 → dot a c = 0
def q : Prop := parallel a b ∧ parallel b c → parallel a c

-- State that p is false and q is true
axiom p_false : ¬p
axiom q_true : q

-- The main statement to prove
theorem main_statement : p ∨ q :=
by {
  right,
  exact q_true,
}

end main_statement_l404_404404


namespace binomial_12_10_eq_66_l404_404619

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404619


namespace concyclic_points_l404_404539

open EuclideanGeometry

/-- Given an acute triangle ABC with altitudes AA₁ and CC₁. 
On the altitude AA₁, a point D is chosen such that A₁D = C₁D.
E is the midpoint of side AC. Prove that points A, C₁, D, and E
lie on the same circle. -/
theorem concyclic_points
  (A B C A₁ C₁ D E : Point) 
  (h_triangle : acute_triangle A B C)
  (h_alt_aa1 : is_altitude A A₁)
  (h_alt_cc1 : is_altitude C C₁)
  (h_midpoint_E : is_midpoint E A C)
  (h_equal_A1D_C1D : dist A₁ D = dist C₁ D) :
  concyclic {A, C₁, D, E} := 
sorry

end concyclic_points_l404_404539


namespace perfect_cubes_count_l404_404247

theorem perfect_cubes_count (a b : ℤ) (ha : a = 3 ^ 6 + 1) (hb : b = 3 ^ 12 + 1) : 
  (finset.Icc a b).filter (λ n, ∃ k : ℕ, n = k^3) .card = 72 :=
by
  sorry

end perfect_cubes_count_l404_404247


namespace sugar_needed_for_muffins_l404_404188

theorem sugar_needed_for_muffins :
  ( ∀ (muffins sugar : ℕ) (muffins_per_recipe sugar_per_recipe : ℕ),
      muffins_per_recipe = 45 → sugar_per_recipe = 3 → 
      muffins = 135 → 
      sugar = (135 * 3) / 45) → 
  ∃ (sugar_needed : ℕ), sugar_needed = 9 :=
by
  intro h
  exists (135 * 3) / 45
  specialize h 135 ((135 * 3) / 45) 45 3 rfl rfl rfl
  exact h

end sugar_needed_for_muffins_l404_404188


namespace max_value_of_symmetric_function_l404_404064

-- Define the function f(x) based on given parameters a and b
def f (x : ℝ) (a : ℝ) (b : ℝ) := (1 - (1/4) * x^2) * (x^2 + a * x + b)

-- State the main problem with the given conditions and expected answer
theorem max_value_of_symmetric_function :
  ∀ (a b : ℝ),
  (∀ x, f x a b = f (-x - 2) a b) →
  (∀ x, f x 0 4 ≤ 4) :=
by sorry

end max_value_of_symmetric_function_l404_404064


namespace find_a_b_l404_404325

noncomputable def f (a b x : ℝ) := a * x^3 + b * x
noncomputable def f' (a b x : ℝ) := (f a b x)' -- derivative

theorem find_a_b (a b : ℝ) :
  (f a b 1 = -2) ∧ (f' a b 1 = 0) → a = 1 ∧ b = -3 :=
by
  -- Definitions of f and f' must be used to prove the theorem
  sorry

end find_a_b_l404_404325


namespace incidence_rate_regression_conditional_probability_l404_404864

theorem incidence_rate_regression :
  let x := [25, 35, 45, 55, 65]
  let y := [0.09, 0.18, 0.30, 0.40, 0.53]
  let n := 5
  let sum_xi := x.sum
  let sum_yi := y.sum
  let mean_x := sum_xi / n
  let mean_y := sum_yi / n
  let sum_xi2 := 11125
  let sum_xi_yi := 78.5
  let b := (sum_xi_yi - n * mean_x * mean_y) / (sum_xi2 - n * mean_x^2)
  let a := mean_y - b * mean_x
  a = -0.195 := by
  compute sum_xi
  compute sum_yi
  compute mean_x
  compute mean_y
  compute b
  compute a
  sorry

theorem conditional_probability :
  let P_A_given_B := 0.99
  let P_not_A_given_not_B := 0.999
  let P_B := 0.0004
  let P_not_B := 1 - P_B
  let P_AB := P_B * P_A_given_B
  let P_A_not_B := P_not_B * (1 - P_not_A_given_not_B)
  let P_A := P_AB + P_A_not_B
  let P_B_given_A := P_AB / P_A
  abs (P_B_given_A - 0.284) < 0.001 := by
  compute P_A
  compute P_AB
  compute P_A_not_B
  compute P_B_given_A
  sorry

end incidence_rate_regression_conditional_probability_l404_404864


namespace binom_12_10_eq_66_l404_404591

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404591


namespace evaluate_f_5_minus_f_neg_5_l404_404349

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l404_404349


namespace concurrency_of_lines_l404_404936

variables (A B C D K M P T O : Type) 
variables [EuclideanPlane A B C D]

-- Assumptions
variable (h_perp_diag : perpendicular (line AC) (line BD))
variable (h_K_midpoint : midpoint A B K)
variable (h_M_midpoint : midpoint A D M)
variable (h_KP_perp_CD : perpendicular (line KP) (line CD))
variable (h_MT_perp_CB : perpendicular (line MT) (line CB))

theorem concurrency_of_lines : concurrent (line KP) (line MT) (line AC) :=
sorry

end concurrency_of_lines_l404_404936


namespace equal_diagonals_l404_404895

section
variables (A B C D M N : ℝ^2)
variables (midpoint_M : M = (B + C) / 2)
variables (midpoint_N : N = (A + D) / 2)
variables (equal_AM_BN : dist A M = dist B N)
variables (equal_DM_CN : dist D M = dist C N)

theorem equal_diagonals : dist A C = dist B D :=
by sorry
end

end equal_diagonals_l404_404895


namespace final_price_after_discounts_l404_404037

theorem final_price_after_discounts (original_price : ℝ)
  (first_discount_pct : ℝ) (second_discount_pct : ℝ) (third_discount_pct : ℝ) :
  original_price = 200 → 
  first_discount_pct = 0.40 → 
  second_discount_pct = 0.20 → 
  third_discount_pct = 0.10 → 
  (original_price * (1 - first_discount_pct) * (1 - second_discount_pct) * (1 - third_discount_pct) = 86.40) := 
by
  intros
  sorry

end final_price_after_discounts_l404_404037


namespace ironed_clothing_l404_404442

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l404_404442


namespace fraction_of_csc_members_in_pc_l404_404574

theorem fraction_of_csc_members_in_pc (x : ℚ) :
  (1 / 5) * (4 * x / 7) / x = 4 / 35 :=
by
  have h1 : (1 : ℚ) = (1 : ℚ) := rfl
  sorry

end fraction_of_csc_members_in_pc_l404_404574


namespace winner_lifted_weight_l404_404501

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l404_404501


namespace problem_statement_l404_404297

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x^2 + a / x

theorem problem_statement (a : ℝ) (k : ℝ) : 
  0 < a ∧ a ≤ 4 →
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 →
  |f x1 a - f x2 a| > k * |x1 - x2|) ↔
  k ≤ 2 - a^3 / 108 :=
by
  sorry

end problem_statement_l404_404297


namespace f_for_negative_x_l404_404695

-- Define the function f based on the given conditions
def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - x - 1
  else -x^2 - x + 1

-- State that the function is odd
axiom odd_function : ∀ (x : ℝ), f(-x) = -f(x)

-- Prove that f(x) = -x^2 - x + 1 when x < 0
theorem f_for_negative_x (x : ℝ) (h : x < 0) : f(x) = -x^2 - x + 1 :=
by sorry

end f_for_negative_x_l404_404695


namespace binomial_coefficient_divisible_by_prime_power_l404_404408

variable (p : ℕ) (α m : ℕ) (n : ℕ := p^(α - 2))

-- odd prime number condition
axiom odd_prime_p : nat.prime p ∧ (p % 2 = 1)

-- conditions on α and m
axiom alpha_ge_two : α ≥ 2
axiom m_ge_two : m ≥ 2

theorem binomial_coefficient_divisible_by_prime_power (hp : nat.prime p) (hα : α ≥ 2) (hm : m ≥ 2) :
  (p^(α - 2) choose m) % p^(α - m) = 0 := sorry

end binomial_coefficient_divisible_by_prime_power_l404_404408


namespace rect_pairs_blue_white_exists_l404_404459

theorem rect_pairs_blue_white_exists (cells : Fin 100 → Color) 
  (h_diff_adj : ∀ i j, adjacent i j → cells i ≠ cells j) 
  (h_red_count : (Finset.filter (λ i, cells i = Color.red) Finset.univ).card = 20) :
  ∃ rects : Finset (Finset (Fin 100)), 
    (∀ r ∈ rects, ∃ i j, adjacent i j ∧ (cells i = Color.blue ∧ cells j = Color.white 
      ∨ cells i = Color.white ∧ cells j = Color.blue)) ∧ rects.card = 30 :=
sorry

-- Some supporting definitions for clarity
inductive Color | red | blue | white

def adjacent (i j : Fin 100) : Prop :=
  (i.val % 10 = j.val % 10 ∧ (i.val / 10 = j.val / 10 ± 1 ∨ i.val / 10 = j.val / 10)) ∨
  (i.val / 10 = j.val / 10 ∧ (i.val % 10 = j.val % 10 ± 1 ∨ i.val % 10 = j.val % 10))

end rect_pairs_blue_white_exists_l404_404459


namespace problem_solution_l404_404625

noncomputable def g (x : ℝ) (P : ℝ) (Q : ℝ) (R : ℝ) : ℝ := x^2 / (P * x^2 + Q * x + R)

theorem problem_solution (P Q R : ℤ) 
  (h1 : ∀ x > 5, g x P Q R > 0.5)
  (h2 : P * (-3)^2 + Q * (-3) + R = 0)
  (h3 : P * 4^2 + Q * 4 + R = 0)
  (h4 : ∃ y : ℝ, y = 1 / P ∧ ∀ x : ℝ, abs (g x P Q R - y) < ε):
  P + Q + R = -24 :=
by
  sorry

end problem_solution_l404_404625


namespace soccer_team_percentage_l404_404920

theorem soccer_team_percentage (total_games won_games : ℕ) (h1 : total_games = 140) (h2 : won_games = 70) :
  (won_games / total_games : ℚ) * 100 = 50 := by
  sorry

end soccer_team_percentage_l404_404920


namespace probability_no_consecutive_heads_l404_404167

theorem probability_no_consecutive_heads :
  let f : ℕ → ℕ := λ n, if n = 1 then 2 else if n = 2 then 3 else (f (n - 1) + f (n - 2))
  ∃ (p : ℚ), p = f 10 / 1024 ∧ p = 9 / 64 :=
by {
  sorry
}

end probability_no_consecutive_heads_l404_404167


namespace total_students_l404_404740

theorem total_students
  (students_music : ℕ)
  (students_art : ℕ)
  (students_both : ℕ)
  (students_neither : ℕ)
  (h_music : students_music = 20)
  (h_art : students_art = 20)
  (h_both : students_both = 10)
  (h_neither : students_neither = 470) :
  students_music - students_both + students_art - students_both + students_both + students_neither = 500 := 
by
  rw [h_music, h_art, h_both, h_neither]
  -- the actual operations would be expanded in the proof
  sorry

end total_students_l404_404740


namespace selection_with_boys_and_girls_l404_404286

def boys := 4
def girls := 3
def total := boys + girls
def choose_total := Nat.choose total 4
def choose_boys_only := Nat.choose boys 4

theorem selection_with_boys_and_girls :
  choose_total - choose_boys_only = 34 :=
by
  -- Proof goes here
  sorry

end selection_with_boys_and_girls_l404_404286


namespace distance_covered_l404_404434

theorem distance_covered (t : ℝ) (s_kmph : ℝ) (distance : ℝ) (h1 : t = 180) (h2 : s_kmph = 18) : 
  distance = 900 :=
by 
  sorry

end distance_covered_l404_404434


namespace amount_of_water_formed_l404_404265

-- Define chemical compounds and reactions
def NaOH : Type := Unit
def HClO4 : Type := Unit
def NaClO4 : Type := Unit
def H2O : Type := Unit

-- Define the balanced chemical equation
def balanced_reaction (n_NaOH n_HClO4 : Int) : (n_NaOH = n_HClO4) → (n_NaOH = 1 → n_HClO4 = 1 → Int × Int × Int × Int) :=
  λ h_ratio h_NaOH h_HClO4 => 
    (n_NaOH, n_HClO4, 1, 1)  -- 1 mole of NaOH reacts with 1 mole of HClO4 to form 1 mole of NaClO4 and 1 mole of H2O

noncomputable def molar_mass_H2O : Float := 18.015 -- g/mol

theorem amount_of_water_formed :
  ∀ (n_NaOH n_HClO4 : Int), 
  (n_NaOH = 1 ∧ n_HClO4 = 1) →
  ((n_NaOH = n_HClO4) → molar_mass_H2O = 18.015) :=
by
  intros n_NaOH n_HClO4 h_condition h_ratio
  sorry

end amount_of_water_formed_l404_404265


namespace binomial_12_10_eq_66_l404_404610

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404610


namespace acute_triangle_lambda_range_l404_404367

noncomputable def acute_triangle_range (A B C a b c λ : ℝ) : Prop :=
  (0 < A ∧ A < π / 2) ∧
  (0 < B ∧ B < π / 2) ∧
  (0 < C ∧ C < π / 2) ∧
  (a^2 - b^2 = bc) ∧ 
  (C = π - A - B) ∧
  (λ > 0 ∧ λ < 2)

theorem acute_triangle_lambda_range {A B C a b c λ: ℝ} :
  acute_triangle_range A B C a b c λ → (λ > 0 ∧ λ < 2) :=
by
  intros h
  sorry

end acute_triangle_lambda_range_l404_404367


namespace measure_of_angle_y_l404_404752

-- Define the parallel lines and given angles
variables {m n : Line}
variables (p q : Point)

-- Lines m and n are parallel
axiom parallel_lines : parallel m n

-- Given angles in the problem
axiom angle_p_40 : angle p q (image.m_id p61) = 40
axiom angle_q_90 : angle q (image.m_id p61) (image.m_id p62) = 90
axiom angle_qp_40 : angle q (image.m_id p62) p = 40

-- The Lean statement to prove that the measure of angle y is 80 degrees
theorem measure_of_angle_y : angle_right.new_image.vision_117 پی is_liar 

end measure_of_angle_y_l404_404752


namespace problem1_problem2_problem3_l404_404175

-- Problem 1: Number of arrangements with A and B at the ends of the back row
theorem problem1 (instructors: ℕ) (students: ℕ) (A B: Type) (arrangements: ℕ) : 
  instructors = 3 ∧ students = 7 ∧ arrangements = 2 * 120 * 6 → arrangements = 1440 :=
by 
  sorry

-- Problem 2: Number of arrangements with A and B not adjacent
theorem problem2 (instructors: ℕ) (students: ℕ) (A B: Type) (arrangements: ℕ) : 
  instructors = 3 ∧ students = 7 ∧ arrangements = 120 * 15 * 2 * 6 → arrangements = 21600 :=
by 
  sorry

-- Problem 3: Number of adjustment methods moving 2 students from back to front row
theorem problem3 (students: ℕ) (adjustment_methods: ℕ) :
  students = 7 ∧ adjustment_methods = (7.choose 2) * 20 → adjustment_methods = 420 :=
by 
  sorry

end problem1_problem2_problem3_l404_404175


namespace mike_profit_l404_404799

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l404_404799


namespace area_percentage_increase_l404_404454

theorem area_percentage_increase (r1 r2 : ℝ) (π : ℝ) (area1 area2 : ℝ) (N : ℝ) :
  r1 = 6 → r2 = 4 → area1 = π * r1 ^ 2 → area2 = π * r2 ^ 2 →
  N = 125 →
  ((area1 - area2) / area2) * 100 = N :=
by {
  sorry
}

end area_percentage_increase_l404_404454


namespace staffing_starship_l404_404234

theorem staffing_starship : 
  let total_candidates := 10,
      chief_science_officer_candidates := 4 in
  (chief_science_officer_candidates * 
  (total_candidates - 1) * 
  (total_candidates - 2) * 
  (total_candidates - 3) = 2016) :=
by
  sorry

end staffing_starship_l404_404234


namespace farmer_brown_additional_cost_l404_404965

theorem farmer_brown_additional_cost:
  ∀ (bales_original bales_multiple cost_original cost_premium: ℕ),
  bales_original = 20 →
  bales_multiple = 5 →
  cost_original = 25 →
  cost_premium = 40 →
  let bales_needed := bales_multiple * bales_original in
  let cost_initial := bales_original * cost_original in
  let cost_new := bales_needed * cost_premium in
  cost_new - cost_initial = 3500 :=
by
  intros bales_original bales_multiple cost_original cost_premium
  intro h1 h2 h3 h4
  let bales_needed := bales_multiple * bales_original
  let cost_initial := bales_original * cost_original
  let cost_new := bales_needed * cost_premium
  have h_bales : bales_needed = 100 := by rw [h1, h2]; exact rfl
  have h_initial_cost : cost_initial = 500 := by rw [h1, h3]; exact rfl
  have h_new_cost : cost_new = 4000 := by rw [h_bales, h4]; exact rfl
  calc cost_new - cost_initial
      = 4000 - 500 : by rw [h_initial_cost, h_new_cost]
  ... = 3500 : by norm_num
  sorry

end farmer_brown_additional_cost_l404_404965


namespace pencils_placed_by_sara_l404_404857

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l404_404857


namespace cost_price_and_sales_correct_l404_404832

variable (C₁ C₂ : ℝ) -- cost price and second-month markup price
variable (P₁ P₂ : ℝ) -- profit in first and second month
variable (N₁ N₂ : ℕ) -- number of items sold in first and second month

-- Given conditions
def condition1 : Prop := P₁ = 6000
def condition2 : Prop := P₂ = 8000
def condition3 : Prop := N₂ = N₁ + 100
def condition4 : Prop := P₂ - P₁ = 2000

-- Definitions of P₂ and P₁
def profit_first_month (C₁ : ℝ) : ℝ := 0.2 * C₁
def profit_second_month (C₂ : ℝ) : ℝ := 0.1 * C₂

-- Proof problem
theorem cost_price_and_sales_correct :
  (∀ (C₁ C₂ : ℝ) (P₁ P₂ : ℝ) (N₁ N₂ : ℕ),
    condition1 → condition2 → condition3 → condition4 → 
    (C₁ = 500 ∧ N₂ = 160)) := by
  intros C₁ C₂ P₁ P₂ N₁ N₂ hC1 hC2 hC3 hC4
  sorry

end cost_price_and_sales_correct_l404_404832


namespace coronene_valid_configurations_l404_404484

def Carbon : Type :=  ℕ
def Hydrogen : Type := ℕ

constant Coronene : Set Carbon → Set Hydrogen → Prop

axiom carbon_count : ∀ (C_set : Set Carbon) (H_set : Set Hydrogen),
  Coronene C_set H_set → C_set.card = 24

axiom hydrogen_count : ∀ (C_set : Set Carbon) (H_set : Set Hydrogen),
  Coronene C_set H_set → H_set.card = 12

axiom carbon_bonds : ∀ (C_set : Set Carbon) (H_set : Set Hydrogen) (c : Carbon),
  Coronene C_set H_set → c ∈ C_set → ∃ (single double : ℕ), single + 2 * double = 4

axiom hydrogen_bonds : ∀ (C_set : Set Carbon) (H_set : Set Hydrogen) (h : Hydrogen),
  Coronene C_set H_set → h ∈ H_set → ∃ (single : ℕ), single = 1

theorem coronene_valid_configurations : ∃ (count : ℕ),
  (∀ (C_set : Set Carbon) (H_set : Set Hydrogen),
    Coronene C_set H_set →
    (C_set.card = 24 ∧ H_set.card = 12) →
    (∀ c ∈ C_set, ∃ (single double : ℕ), single + 2 * double = 4) →
    (∀ h ∈ H_set, ∃ single, single = 1)) →
  count = 20 :=
by
  sorry

end coronene_valid_configurations_l404_404484


namespace unique_function_number_of_functions_l404_404998

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem unique_function {f : ℝ → ℝ}
  (h1 : f 1 = 1)
  (h2 : ∀ (x y : ℝ), f (x^2 * y^2) = f (x^4 + y^4)) :
  ∀ x : ℝ, f x = 1 :=
  sorry

theorem number_of_functions :
  ∃! f : (ℝ → ℝ), f 1 = 1 ∧ (∀ (x y : ℝ), f (x^2 * y^2) = f (x^4 + y^4)) :=
begin
  use f,
  split,
  {
    split,
    { exact h1 },
    { exact h2 }
  },
  {
    intros g h,
    funext,
    apply unique_function,
    { exact h.1 },
    { exact h.2 }
  }
end

end unique_function_number_of_functions_l404_404998


namespace A_cubed_inv_l404_404345

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Given condition
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 7], ![-2, -4]]

-- Goal to prove
theorem A_cubed_inv :
  (A^3)⁻¹ = ![![11, 17], ![2, 6]] :=
  sorry

end A_cubed_inv_l404_404345


namespace max_self_intersections_polyline_7_l404_404521

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end max_self_intersections_polyline_7_l404_404521


namespace rate_of_current_is_3_l404_404487

-- Definitions of the given conditions
def boat_speed : ℝ := 15
def distance_downstream : ℝ := 7.2
def time_minutes : ℝ := 24

-- Downstream speed equation
def downstream_speed (c : ℝ) : ℝ := boat_speed + c

-- Convert minutes to hours
def time_hours : ℝ := time_minutes / 60

-- Proof statement
theorem rate_of_current_is_3 :
  ∃ c : ℝ, downstream_speed c * time_hours = distance_downstream ∧ c = 3 :=
by
  sorry

end rate_of_current_is_3_l404_404487


namespace daily_wage_of_a_man_l404_404852

theorem daily_wage_of_a_man (M W : ℝ) 
  (h1 : 24 * M + 16 * W = 11600) 
  (h2 : 12 * M + 37 * W = 11600) : 
  M = 350 :=
by
  sorry

end daily_wage_of_a_man_l404_404852


namespace equivalent_single_discount_l404_404059

theorem equivalent_single_discount (p : ℝ) : 
  let discount1 := 0.15
  let discount2 := 0.25
  let price_after_first_discount := (1 - discount1) * p
  let price_after_second_discount := (1 - discount2) * price_after_first_discount
  let equivalent_single_discount := 1 - price_after_second_discount / p
  equivalent_single_discount = 0.3625 :=
by
  sorry

end equivalent_single_discount_l404_404059


namespace sam_effective_avg_speed_l404_404827

theorem sam_effective_avg_speed
  (total_miles : ℕ)
  (total_time_minutes : ℕ)
  (first_segment_minutes : ℕ)
  (first_segment_speed_mph : ℕ)
  (second_segment_minutes : ℕ)
  (second_segment_speed_mph : ℕ)
  (stop_minutes : ℕ)
  (actual_last_segment_minutes : ℕ)
  (actual_last_segment_speed_mph : ℕ)
  (converted_last_segment_time_hours : ℕ)
  (distance_first_segment : ℚ)
  (distance_second_segment : ℚ)
  (total_distance_first_two_segments : ℚ)
  (distance_last_segment : ℚ)
  (effective_average_speed_last_segment : ℚ) : 
  total_miles = 120 ∧
  total_time_minutes = 120 ∧
  first_segment_minutes = 40 ∧
  first_segment_speed_mph = 50 ∧
  second_segment_minutes = 40 ∧
  second_segment_speed_mph = 55 ∧
  stop_minutes = 5 ∧
  actual_last_segment_minutes = 35 ∧
  effective_average_speed_last_segment = 85 :=
begin
  sorry  -- proof steps are omitted, only statement is required
end

end sam_effective_avg_speed_l404_404827


namespace basketball_club_members_l404_404361

theorem basketball_club_members :
  let sock_cost := 6
  let tshirt_additional_cost := 8
  let total_cost := 4440
  let cost_per_member := sock_cost + 2 * (sock_cost + tshirt_additional_cost)
  total_cost / cost_per_member = 130 :=
by
  sorry

end basketball_club_members_l404_404361


namespace extra_apples_l404_404850

theorem extra_apples 
  (red_apples : ℕ)
  (green_apples : ℕ)
  (students_wanting_fruit : ℕ)
  (taken_apples_per_student : ℕ)
  (total_apples := red_apples + green_apples)
  (apples_taken := students_wanting_fruit * taken_apples_per_student) 
  (extra_apples := total_apples - apples_taken) :
  red_apples = 43 →
  green_apples = 32 →
  students_wanting_fruit = 2 →
  taken_apples_per_student = 1 →
  extra_apples = 73 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end extra_apples_l404_404850


namespace ellipse_equation_max_area_ABC_l404_404686

noncomputable def eccentricity_of_ellipse := real.sqrt 2 / 2
def point_A : ℝ × ℝ := (1, real.sqrt 2)
def slope_of_l : ℝ := real.sqrt 2

-- Define the conditions related to the ellipse
def ellipse_eq (a b : ℝ) (x y : ℝ) := y^2 / a^2 + x^2 / b^2 = 1
def a_greater_b (a b : ℝ) := a > b ∧ b > 0

-- The proof problem statements
theorem ellipse_equation (a b : ℝ) (h1 : point_A ∈ set_of (λ p : ℝ × ℝ, ellipse_eq a b p.1 p.2))
  (h2 : eccentricity_of_ellipse = real.sqrt 2 / 2)
  (h3 : a_greater_b a b) :
  ellipse_eq 2 (real.sqrt 2) := 
sorry

theorem max_area_ABC (a b : ℝ) (h1 : point_A ∈ set_of (λ p : ℝ × ℝ, ellipse_eq a b p.1 p.2))
  (h2 : eccentricity_of_ellipse = real.sqrt 2 / 2)
  (h3 : a_greater_b a b)
  (h4 : ellipse_eq 2 (real.sqrt 2)) :
  ∀ (B C : ℝ × ℝ), 
  is_line slope_of_l B C → max_area A B C = real.sqrt 2 :=
sorry

-- Auxiliary definitions
def is_line (slope : ℝ) (B C : ℝ × ℝ) : Prop :=
  (C.2 - B.2) = slope * (C.1 - B.1)

def max_area (A B C : ℝ × ℝ) : ℝ :=
  let d := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
  (1/2) * d * ((A.1 - (B.1 + C.1)/2) + (A.2 - (B.2 + C.2)/2))

end ellipse_equation_max_area_ABC_l404_404686


namespace line_intersects_circle_l404_404849

theorem line_intersects_circle (a : ℝ) :
  ∃ (x y : ℝ), (y = a * x + 1) ∧ ((x - 1) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end line_intersects_circle_l404_404849


namespace common_number_in_sequences_l404_404216

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end common_number_in_sequences_l404_404216


namespace john_took_11_more_l404_404075

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l404_404075


namespace binomial_coefficient_12_10_l404_404607

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404607


namespace circumcircle_area_of_triangle_ABC_l404_404368

noncomputable def circumcircle_area : ℝ :=
  let R := (4 * Real.sqrt 3) / 3 in
  Real.pi * R^2

theorem circumcircle_area_of_triangle_ABC :
  ∀ (a b c : ℝ) (A B C : ℝ),
    A = 75 ∧ B = 45 ∧ c = 4 ∧
    (a / Real.sin (A * Real.pi / 180) = 2 * R ∧
     b / Real.sin (B * Real.pi / 180) = 2 * R ∧
     c / Real.sin (C * Real.pi / 180) = 2 * R ∧
     C = 60) →
    circumcircle_area = 16 * Real.pi / 3 :=
by
  sorry

end circumcircle_area_of_triangle_ABC_l404_404368


namespace jessica_purchase_cost_l404_404764

noncomputable def c_toy : Real := 10.22
noncomputable def c_cage : Real := 11.73
noncomputable def c_total : Real := c_toy + c_cage

theorem jessica_purchase_cost : c_total = 21.95 :=
by
  sorry

end jessica_purchase_cost_l404_404764


namespace sin_Z_in_right_triangle_l404_404372

-- Defining the setup of the right triangle and the trigonometric identities
structure RightTriangle :=
  (X Y Z : Point)
  (angle_Y : Angle)
  (right_angle : angle_Y = 90)
  (sin_X : Real)

-- The mathematical equivalent proof problem
theorem sin_Z_in_right_triangle (T : RightTriangle) (hT : T.right_angle) (hSin : T.sin_X = 8 / 17) :
  ∃ sin_Z : Real, sin_Z = 15 / 17 := 
  sorry

end sin_Z_in_right_triangle_l404_404372


namespace duty_roster_arrangements_l404_404858

theorem duty_roster_arrangements (students : Fin 5) : 
  let A := students 0 in
  (A = 0 ∨ A = 1) →
  ∀ (B C D E : Fin 5), B ≠ A → C ≠ A → D ≠ A → E ≠ A →
  B ≠ C → B ≠ D → B ≠ E → C ≠ D → C ≠ E → D ≠ E →
  ∃! roster : Fin 5 → Fin 5, (roster 0) ∈ ({0, 1}) ∧
  ∀ i, roster (students i) ≠ roster (students (i + 1) % 5) ∧
  (∑ i in Finset.range 5, 1) = 48 :=
by 
  -- {{ Assist the user to skip the proof step by adding sorry. }}
  sorry

end duty_roster_arrangements_l404_404858


namespace john_took_11_more_l404_404074

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l404_404074


namespace total_students_at_competition_l404_404120

def KnowItAllHigh : ℕ := 50
def KarenHigh : ℕ := (3 / 5 : ℚ) * KnowItAllHigh
def CombinedSchools : ℕ := KnowItAllHigh + KarenHigh
def NovelCoronaHigh : ℕ := 2 * CombinedSchools
def TotalStudents := CombinedSchools + NovelCoronaHigh

theorem total_students_at_competition : TotalStudents = 240 := by
  sorry

end total_students_at_competition_l404_404120


namespace train_crossing_time_l404_404385

theorem train_crossing_time
  (train_length : ℕ)           -- length of the train in meters
  (train_speed_kmh : ℕ)        -- speed of the train in kilometers per hour
  (conversion_factor : ℕ)      -- conversion factor from km/hr to m/s
  (train_speed_ms : ℕ)         -- speed of the train in meters per second
  (time_to_cross : ℚ)          -- time to cross in seconds
  (h1 : train_length = 60)
  (h2 : train_speed_kmh = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : train_speed_ms = train_speed_kmh * conversion_factor)
  (h5 : time_to_cross = train_length / train_speed_ms) :
  time_to_cross = 1.5 :=
by sorry

end train_crossing_time_l404_404385


namespace interest_rate_calculation_l404_404553

theorem interest_rate_calculation 
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = 12000)
  (hT : T = 3)
  (hSI : SI = 4320)
  : R = 12 :=
begin
  sorry
end

end interest_rate_calculation_l404_404553


namespace tan_theta_eq_sqrt2_div_4_l404_404691

theorem tan_theta_eq_sqrt2_div_4 
  (θ : ℝ) 
  (hθ1 : -π / 2 < θ) 
  (hθ2 : θ < π / 2)
  (h_max : ∃ x : ℝ, f x = sin (x + θ) * cos x ∧ ∀ x' : ℝ, f x' ≤ 2 / 3) : 
  tan θ = sqrt 2 / 4 :=
sorry

end tan_theta_eq_sqrt2_div_4_l404_404691


namespace original_total_cost_final_cost_after_discounts_and_tax_l404_404865

-- Define conditions as Lean definitions
def num_fandoms := 4
def shirts_per_fandom := 5
def cost_per_shirt := 15
def discount_20_percent := 0.20
def discount_additional_10_percent := 0.10
def tax_10_percent := 0.10

-- Define the proofs for the questions
theorem original_total_cost :
  let total_shirts := shirts_per_fandom * num_fandoms in
  let total_cost := total_shirts * cost_per_shirt in
  total_cost = 300 :=
by
  let total_shirts := shirts_per_fandom * num_fandoms
  let total_cost := total_shirts * cost_per_shirt
  have h1 : total_shirts = 20, sorry -- simplified calculation
  have h2 : total_cost = 300, sorry -- simplified calculation
  exact h2

theorem final_cost_after_discounts_and_tax :
  let total_cost := (shirts_per_fandom * num_fandoms) * cost_per_shirt in
  let cost_after_20_discount := total_cost * (1 - discount_20_percent) in
  let final_cost_after_additional_discount := cost_after_20_discount * (1 - discount_additional_10_percent) in
  let final_amount := final_cost_after_additional_discount * (1 + tax_10_percent) in
  final_amount = 237.60 :=
by
  let total_cost := (shirts_per_fandom * num_fandoms) * cost_per_shirt
  let cost_after_20_discount := total_cost * (1 - discount_20_percent)
  let final_cost_after_additional_discount := cost_after_20_discount * (1 - discount_additional_10_percent)
  let final_amount := final_cost_after_additional_discount * (1 + tax_10_percent)
  have h1 : total_cost = 300, sorry -- simplified calculation
  have h2 : cost_after_20_discount = 240, sorry -- simplified calculation
  have h3 : final_cost_after_additional_discount = 216, sorry -- simplified calculation
  have h4 : final_amount = 237.60, sorry -- simplified calculation
  exact h4

end original_total_cost_final_cost_after_discounts_and_tax_l404_404865


namespace solve_for_x_l404_404056

theorem solve_for_x (x : ℝ) (h : (3 * x - 15) / 4 = (x + 9) / 5) : x = 10 :=
by {
  sorry
}

end solve_for_x_l404_404056


namespace find_principal_l404_404887

-- define constants and conditions
def interest_amount := 14705.24
def annual_rate := 0.11
def time_years := 3
def compoundings_per_year := 1

-- define the expression for compound interest
def compound_interest_formula (P : ℝ) :=
  P * (1 + annual_rate / compoundings_per_year) ^ (compoundings_per_year * time_years) - P

-- define the proof statement
theorem find_principal (P : ℝ) (h₁ : compound_interest_formula P = interest_amount) : P ≈ 40000 :=
  sorry

end find_principal_l404_404887


namespace oxen_grazing_months_l404_404201

theorem oxen_grazing_months (total_rent : ℝ) (A_oxen A_months B_oxen C_oxen C_months C_rent : ℝ) 
  (hA : A_oxen * A_months)
  (hB : (B_oxen : ℝ) * (B_months : ℝ))
  (hC : C_oxen * C_months = 45) 
  (total_oxen_months : ℝ := hA + 12 * B_months + hC)
  (rent_proportion : C_rent / total_rent = hC / total_oxen_months)
  (total_rent : a := 280)
  (C_rent := 72)
  : B_months = 5 :=
begin
  sorry
end

end oxen_grazing_months_l404_404201


namespace coffee_consumption_thursday_l404_404185

theorem coffee_consumption_thursday :
  ∃ k : ℝ, (3 = (k * 3) / 8) → (g t = (k * 5) / 10) → g t = 4 := 
by sorry

end coffee_consumption_thursday_l404_404185


namespace area_of_remaining_figure_l404_404142
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end area_of_remaining_figure_l404_404142


namespace intersection_points_of_spheres_l404_404974

theorem intersection_points_of_spheres :
  ∃ (pts : Set (ℤ × ℤ × ℤ)),
  (∀ (x y z : ℤ), (x, y, z) ∈ pts ↔ ↑x^2 + ↑y^2 + ((↑z - 5)^2 : ℝ) ≤ 9 ∧ ↑x^2 + ↑y^2 + (↑z^2 : ℝ) ≤ 4) ∧
  pts.to_finset.card = 1 :=
by
  sorry

end intersection_points_of_spheres_l404_404974


namespace grinding_wheel_division_l404_404862

theorem grinding_wheel_division :
  ∀ (R r : ℝ) (companion1_ΔR companion2_ΔR : ℝ),
  R = 10 → r = 2 →
  (companion1_ΔR ≈ 1.754) ∧ (companion2_ΔR ≈ 2.246) ∧ (2 = 6 - companion2_ΔR) →
  ∃ companion3_ΔR, companion3_ΔR = 4 :=
by
  intro R r companion1_ΔR companion2_ΔR
  assume hr₁ : R = 10,
  assume hr₂ : r = 2,
  assume h_companions : companion1_ΔR ≈ 1.754 ∧ companion2_ΔR ≈ 2.246 ∧ 2 = 6 - companion2_ΔR,
  sorry  

end grinding_wheel_division_l404_404862


namespace range_of_a_l404_404308

theorem range_of_a (p q : Prop)
  (hp : ∀ a : ℝ, (1 < a ↔ p))
  (hq : ∀ a : ℝ, (2 ≤ a ∨ a ≤ -2 ↔ q))
  (hpq : ∀ a : ℝ, ∀ (p : Prop), ∀ (q : Prop), (p ∧ q) → p ∧ q) :
    ∀ a : ℝ, p ∧ q → 2 ≤ a :=
sorry

end range_of_a_l404_404308


namespace greatest_possible_value_x_l404_404872

theorem greatest_possible_value_x :
  ∀ x : ℚ, (∃ y : ℚ, y = (5 * x - 25) / (4 * x - 5) ∧ y^2 + y = 18) →
  x ≤ 55 / 29 :=
by sorry

end greatest_possible_value_x_l404_404872


namespace Chang_solution_A_amount_l404_404380

def solution_alcohol_content (A B : ℝ) (x : ℝ) : ℝ :=
  0.16 * x + 0.10 * (x + 500)

theorem Chang_solution_A_amount (x : ℝ) :
  solution_alcohol_content 0.16 0.10 x = 76 → x = 100 :=
by
  intro h
  sorry

end Chang_solution_A_amount_l404_404380


namespace sum_x_coordinates_eq_8950_l404_404504

theorem sum_x_coordinates_eq_8950 :
  let lines := (λ (θ : ℕ), θ < 180 → is_line_through_origin θ)
  let intersections := λ (θ : ℕ) (h : lines θ), exists_intersection θ
  let x_coords := λ (θ : ℕ) (h : intersections θ h ), is_x_coord θ
  (∑ θ in finset.range 179, x_coords θ sorry) = 8950 :=
by sorry

/-
Definitions:
is_line_through_origin: represents a line through origin at an angle.
exists_intersection: represents existence of intersection with y = 100 - x.
is_x_coord: represents the x-coordinate of the point of intersection.
-/

end sum_x_coordinates_eq_8950_l404_404504


namespace arithmetic_progression_rth_term_l404_404284

theorem arithmetic_progression_rth_term (S : ℕ → ℕ) (hS : ∀ n, S n = 5 * n + 4 * n ^ 2) 
  (r : ℕ) : S r - S (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l404_404284


namespace problem_solution_l404_404396

noncomputable def P (x y : ℕ) : ℕ := ...

theorem problem_solution :
  (∃ P : ℕ → ℕ → ℕ, 
    (∀ x y, P x y = ∑ i in Finset.range (x + y + 1), Nat.choose (x + y) i) ∧
    (∀ x, y, P x y = Finset.sum (Finset.range (x + y + 1)) (λ i, Nat.choose (x + y) i)) ∧
    (∀ x, y, P x y = ∏ i in Finset.range x, Nat.choose (i + y + x - i) x) ∧
    ∀ x, y, P(x,y).degree_x ≤ 2020 ∧ P(x,y).degree_y ≤ 2020 ) →
  P 4040 4040 % 2017 = 1555 :=
by mathlib_example sorry

end problem_solution_l404_404396


namespace translation_vector_l404_404729

variable (f : ℝ → ℝ)

/--
If the graph of the function y = f(2x - 1) + 1 is translated by vector a, and the equation of the translated function is 
y = f(2x + 1) - 1, then vector a equals (1, -2).
-/
theorem translation_vector (a : ℝ × ℝ) (h₀ : ∀ x, f(2 * (x + 1) - 1) - 2 = f(2 * x - 1) + 1) : a = (1, -2) :=
sorry

end translation_vector_l404_404729


namespace remainder_of_M_div_1000_l404_404778

def count_numbers_with_more_ones_than_zeros (limit : ℕ) : ℕ := sorry

theorem remainder_of_M_div_1000 :
  let M := count_numbers_with_more_ones_than_zeros 1500 in
  M % 1000 = 78 :=
by
  sorry

end remainder_of_M_div_1000_l404_404778


namespace binom_12_10_eq_66_l404_404599

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404599


namespace fill_pool_time_correct_l404_404184

-- Define the rates of the two pipes
def R1 : ℝ := 1 / 10
def R2 : ℝ := 1 / 6

-- Define the combined rate of both pipes turned on
def R_combined : ℝ := R1 + R2

-- Define the time to fill the pool with both pipes turned on
def T : ℝ := 1 / R_combined

-- The theorem to prove
theorem fill_pool_time_correct : T = 3.75 := by
  -- Use R1 and R2 to calculate and prove the theorem
  sorry

end fill_pool_time_correct_l404_404184


namespace distance_equal_axes_l404_404085

theorem distance_equal_axes (m : ℝ) :
  (abs (3 * m + 1) = abs (2 * m - 5)) ↔ (m = -6 ∨ m = 4 / 5) :=
by 
  sorry

end distance_equal_axes_l404_404085


namespace alex_buns_needed_l404_404206

def packs_of_buns_needed (burgers_per_guest guests non_meat_eating_friend non_bread_eating_friend buns_per_pack : ℕ) : ℕ := 
  (burgers_per_guest * (guests - non_meat_eating_friend - (if non_eat_bread_eating_friend > 0 then 1 else 0)) - two_friends_with_different_needs) / buns_per_pack

theorem alex_buns_needed (h1 : burgers_per_guest = 3) (h2 : guests = 10) (h3 : non_meat_eating_friend = 1)
                       (h4 : non_bread_eating_friend = 1) (h5 : buns_per_pack = 8) :
  packs_of_buns_needed burgers_per_guest guests non_meat_eating_friend non_bread_eating_friend buns_per_pack = 3 := 
by
  sorry

end alex_buns_needed_l404_404206


namespace max_value_bound_l404_404027

noncomputable def max_value_problem (x y z : ℝ) : ℝ :=
  3 * x * y * real.sqrt 5 + 9 * y * z

theorem max_value_bound (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_eq : x^2 + y^2 + z^2 = 1) :
  max_value_problem x y z ≤ (3 / 2) * real.sqrt 409 :=
sorry

end max_value_bound_l404_404027


namespace mike_profit_l404_404806

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l404_404806


namespace equivalent_expression_l404_404868

def evaluate_expression : ℚ :=
  let part1 := (2/3) * ((35/100) * 250)
  let part2 := ((75/100) * 150) / 16
  let part3 := (1/2) * ((40/100) * 500)
  part1 - part2 + part3

theorem equivalent_expression :
  evaluate_expression = 151.3020833333 :=  
by 
  sorry

end equivalent_expression_l404_404868


namespace odd_numbers_in_pascals_triangle_l404_404999

def count_ones_in_binary (n : ℕ) : ℕ :=
  Integer.to_nat (n.bits.filter (λ b => b = tt)).length

theorem odd_numbers_in_pascals_triangle (n : ℕ) :
  ∃ k : ℕ, k = count_ones_in_binary n ∧ 
  (let number_of_odds := 2^k in
   number_of_odds = (n.bits.filter (λ b => b = tt)).length) :=
sorry

end odd_numbers_in_pascals_triangle_l404_404999


namespace parallelogram_and_area_l404_404306

-- Defining the points A, B, C, and D in the 3D space
def A := (2, -5, 3)
def B := (6, -9, 7)
def C := (4, -2, 1)
def D := (8, -6, 5)

-- The main theorem stating that A, B, C, and D form a parallelogram and the area of this parallelogram
theorem parallelogram_and_area 
  (A B C D : ℝ × ℝ × ℝ)
  (hA : A = (2, -5, 3))
  (hB : B = (6, -9, 7))
  (hC : C = (4, -2, 1))
  (hD : D = (8, -6, 5)) :
  (B.1 - A.1, B.2 - A.2, B.3 - A.3) = (D.1 - C.1, D.2 - C.2, D.3 - C.3) ∧
  ∥(
    (B.2 - A.2) * (C.3 - A.3) - (B.3 - A.3) * (C.2 - A.2),
    (B.3 - A.3) * (C.1 - A.1) - (B.1 - A.1) * (C.3 - A.3),
    (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)
  )∥ = 4 * Real.sqrt 42 := 
sorry

end parallelogram_and_area_l404_404306


namespace angle_equality_l404_404429

open EuclideanGeometry

variable {P : Type*} [MetricSpace P] [NormedAddTorsor ℝ P]

-- Assume we have points A, B, C, D, E which form a convex pentagon
variables (A B C D E : P) (convex: Convex ℝ (Set.Polygon [A, B, C, D, E]))

-- Define the given conditions in the problem
variable (h1 : ∠(B -ᵥ A) (C -ᵥ B) = ∠(E -ᵥ D) (A -ᵥ E))
variable (h2 : ∠(E -ᵥ A) (C -ᵥ E) = ∠(B -ᵥ D) (A -ᵥ B))

-- State the theorem to be proven
theorem angle_equality : ∠(C -ᵥ A) (B -ᵥ A) = ∠(E -ᵥ A) (D -ᵥ A) :=
by
  -- Provide proof outline here
  sorry

end angle_equality_l404_404429


namespace Zachary_crunches_value_l404_404245

variable (David_pushups Zachary_pushups David_crunches Zachary_crunches : ℕ)

axiom (h1 : David_pushups = Zachary_pushups + 40)
axiom (h2 : David_crunches = Zachary_crunches - 17)
axiom (h3 : David_crunches = 45)
axiom (h4 : Zachary_pushups = 34)

theorem Zachary_crunches_value : Zachary_crunches = 62 := by
  -- proof here
  sorry

end Zachary_crunches_value_l404_404245


namespace AL_bisects_angle_BAC_l404_404756

open EuclideanGeometry

/--
In the triangle \( \triangle ABC \), the excircle \( \odot P \) touches the extensions of \( CB \) and \( CA \) at points \( D \) and \( E \), respectively.
The excircle \( \odot Q \) touches the extensions of \( BC \) and \( BA \) at points \( F \) and \( G \), respectively.
Lines \( DE \) and \( FG \) intersect \( PQ \) at points \( M \) and \( N \), respectively.
Lines \( BN \) and \( CM \) intersect at point \( L \).
Prove that \( AL \) bisects \(\angle BAC\).
-/
theorem AL_bisects_angle_BAC
  (A B C D E F G P Q M N L : Point)
  (h1 : Triangle A B C)
  (h2 : Excircle P A C B D E)
  (h3 : Excircle Q B A C F G)
  (h4 : IntersectAt DE FG M)
  (h5 : IntersectAt PQ DE N)
  (h6 : IntersectAt PQ FG M)
  (h7 : Collinear [A, P, Q])
  (h8 : LineThrough B N L)
  (h9 : LineThrough C M L) :
  Bisects A L (Angle A B C) :=
sorry

end AL_bisects_angle_BAC_l404_404756


namespace max_students_possible_l404_404918

open Set

-- Definitions
variable {Problem : Type} (Problems : Set Problem) (Students : Set (Set Problem))

-- Problem Conditions
def valid_problem_set (Problems : Set Problem) : Prop :=
  card Problems = 8

def valid_student_assignment (Problems : Set Problem) (S : Set (Set Problem)) : Prop :=
  ∀ s ∈ S, card s = 3 ∧ (∀ t ∈ S, s ≠ t → card (s ∩ t) ≤ 1)

-- Proof Objective
theorem max_students_possible (Problems : Set Problem) (S : Set (Set Problem)) 
  (h1 : valid_problem_set Problems) (h2 : valid_student_assignment Problems S) :
  card S ≤ 8 := 
sorry

end max_students_possible_l404_404918


namespace problem1_problem2_problem3_problem4_l404_404156

-- Definitions of conversion rates used in the conditions
def sq_m_to_sq_dm : Nat := 100
def hectare_to_sq_m : Nat := 10000
def sq_cm_to_sq_dm_div : Nat := 100
def sq_km_to_hectare : Nat := 100

-- The problem statement with the expected values
theorem problem1 : 3 * sq_m_to_sq_dm = 300 := by
  sorry

theorem problem2 : 2 * hectare_to_sq_m = 20000 := by
  sorry

theorem problem3 : 5000 / sq_cm_to_sq_dm_div = 50 := by
  sorry

theorem problem4 : 8 * sq_km_to_hectare = 800 := by
  sorry

end problem1_problem2_problem3_problem4_l404_404156


namespace DanielCandies_l404_404038

noncomputable def initialCandies (x : ℝ) : Prop :=
  (3 / 8) * x - (3 / 2) - 16 = 10

theorem DanielCandies : ∃ x : ℝ, initialCandies x ∧ x = 93 :=
by
  use 93
  simp [initialCandies]
  norm_num
  sorry

end DanielCandies_l404_404038


namespace non_similar_triangles_cyclic_quad_l404_404336

-- Definitions from the problem's conditions
variables {α : Type*} [EuclideanGeometry α]
variables {A B C D E F G : α}

-- Given triangle and midpoints condition
variable (is_midpoint : Midpoint B C D ∧ Midpoint A C E ∧ Midpoint A B F)

-- Given centroid condition
variable (is_centroid : Centroid A B C G)

-- Given cyclic quadrilateral condition
variable (is_cyclic : CyclicQuad A E G F)

-- Given fixed angle BAC
variable (BAC_angle_fixed : fixed_angle BAC)

-- Proving the mathematically equivalent problem statement
theorem non_similar_triangles_cyclic_quad :
  ∃ (n : ℕ), n = 2 ∧ ∀ (A B C : α), CyclicQuad A E G F → ¬(SimilarTriangles A B C A' B' C') :=
sorry

end non_similar_triangles_cyclic_quad_l404_404336


namespace five_digit_count_l404_404930

theorem five_digit_count :
  let digits := {0, 1, 2, 3, 4} in
  ∃ count : ℕ,
  (∀ n ∈ digits, n ≠ 0) →
  count = 240 :=
sorry

end five_digit_count_l404_404930


namespace problem_statement_l404_404024

noncomputable def f (x : ℝ) : ℝ := x + 1 / x - Real.sqrt 2

theorem problem_statement (x : ℝ) (h₁ : x ∈ Set.Ioc (Real.sqrt 2 / 2) 1) :
  Real.sqrt 2 / 2 < f (f x) ∧ f (f x) < x :=
by
  sorry

end problem_statement_l404_404024


namespace imaginary_numbers_count_l404_404663

theorem imaginary_numbers_count :
  let S := {0, 1, 2, 3, 4, 5}
  in 
  (finset.filter (λ z : ℂ, ∃ (x y : ℕ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ y ≠ 0 ∧ z = complex.mk x y)
                  (finset.image (λ xy, (complex.mk xy.fst xy.snd : ℂ))
                                 ((finset.product S S))).to_finset).card = 25 :=
by {
  let S := {0, 1, 2, 3, 4, 5},
  sorry
}

end imaginary_numbers_count_l404_404663


namespace find_c_of_perpendicular_lines_l404_404636

theorem find_c_of_perpendicular_lines (c : ℤ) :
  (∀ x y : ℤ, y = -3 * x + 4 → ∃ y' : ℤ, y' = (c * x + 18) / 9) →
  c = 3 :=
by
  sorry

end find_c_of_perpendicular_lines_l404_404636


namespace product_digit_sum_is_six_l404_404475

theorem product_digit_sum_is_six :
  let num1 := 606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606 : ℕ,
      num2 := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 : ℕ,
      prod := num1 * num2,
      units_digit := prod % 10,
      thousands_digit := (prod / 1000) % 10
  in thousands_digit + units_digit = 6 :=
by
  sorry

end product_digit_sum_is_six_l404_404475


namespace intersection_on_line_LM_l404_404392

set_option pp.all true

/-
Let ABC be a right-angled triangle with ∠ A = 90°. Let K be the midpoint of BC,
and let AKLM be a parallelogram with center C. Let T be the intersection of the line AC 
and the perpendicular bisector of BM. Let ω_1 be the circle with center C and radius CA 
and let ω_2 be the circle with center T and radius TB. Prove that one of the points
of intersection of ω_1 and ω_2 is on the line LM.
-/

variables {A B C K L M T : Type} [point A] [point B] [point C] [point K] [point L] [point M] [point T]
variables (ABC : triangle A B C) (K_eq_midpoint_BC : midpoint B C = K)
variables (parallelogram_AKLM : parallelogram A K L M) (center_C : center AKLM = C)
variables (T_eq_intersection : T = intersection (line A C) (perpendicular_bisector B M))
variables (ω1 : circle C (distance C A)) (ω2 : circle T (distance T B))
variables (intersection_point : point)

theorem intersection_on_line_LM :
  intersection_point ∈ (ω1 ∩ ω2) → 
  location intersection_point ∈ line L M :=
sorry

end intersection_on_line_LM_l404_404392


namespace compare_tan_neg_values_l404_404967

theorem compare_tan_neg_values :
  tan (- (13 * Real.pi / 7)) > tan (- (15 * Real.pi / 8)) :=
by
  sorry

end compare_tan_neg_values_l404_404967


namespace two_person_subcommittees_l404_404716

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l404_404716


namespace range_of_m_l404_404269

noncomputable def quadratic (x m : ℝ) : ℝ := x^2 + (m-2)*x + (m+6)

theorem range_of_m 
  (m : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, is_root (quadratic x1 m) 0 → is_root (quadratic x2 m) 0 → (x1 > 2 ∧ x2 < 2) ∨ (x1 < 2 ∧ x2 > 2))
  (h2 : ∀ x1 x2 : ℝ, is_root (quadratic x1 m) 0 → is_root (quadratic x2 m) 0 → x1 > 1 ∧ x2 > 1) :
  -5/2 < m ∧ m ≤ -2 :=
begin
  sorry
end

end range_of_m_l404_404269


namespace number_of_5_digit_palindromes_l404_404159

-- Define what a 5-digit palindrome is
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse ∧ digits.length = 5 ∧ digits.head ≠ some 0

-- Theorem statement
theorem number_of_5_digit_palindromes : 
  {n : ℕ | is_palindrome n}.card = 900 :=
sorry

end number_of_5_digit_palindromes_l404_404159


namespace isPossible_l404_404209

structure Person where
  firstName : String
  patronymic : String
  surname : String

def conditions (people : List Person) : Prop :=
  people.length = 4 ∧
  ∀ p1 p2 p3 : Person, 
    p1 ∈ people → p2 ∈ people → p3 ∈ people →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    p1.firstName ≠ p2.firstName ∨ p2.firstName ≠ p3.firstName ∨ p1.firstName ≠ p3.firstName ∧
    p1.patronymic ≠ p2.patronymic ∨ p2.patronymic ≠ p3.patronymic ∨ p1.patronymic ≠ p3.patronymic ∧
    p1.surname ≠ p2.surname ∨ p2.surname ≠ p3.surname ∨ p1.surname ≠ p3.surname ∧
  ∀ p1 p2 : Person, 
    p1 ∈ people → p2 ∈ people →
    p1 ≠ p2 →
    p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.surname = p2.surname

theorem isPossible : ∃ people : List Person, conditions people := by
  sorry

end isPossible_l404_404209


namespace concave_numbers_total_l404_404062

def is_concave (n : ℕ) : Prop :=
  let a₁ := n / 100
  let a₂ := (n / 10) % 10
  let a₃ := n % 10
  a₁ > a₂ ∧ a₂ < a₃

def concave_numbers_count : ℕ :=
  (List.range 900).countp (λ n, is_concave (n + 100))

theorem concave_numbers_total : concave_numbers_count = 285 := 
  sorry

end concave_numbers_total_l404_404062


namespace count_valid_partitions_l404_404777

def isNonEmpty (s : Set α) : Prop := s ≠ ∅

def isValidPartition {α : Type} (C D : Set α) :=
  (C ∪ D = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (C ∩ D = ∅) ∧
  (¬ (C.card ∈ C)) ∧
  (¬ (D.card ∈ D))

theorem count_valid_partitions : 
  (∃! pairs : Set ℕ × Set ℕ, isValidPartition pairs.1 pairs.2) = 93 := by
  sorry

end count_valid_partitions_l404_404777


namespace hexagon_triangles_l404_404192

def num_triangles (s : Finset (Fin 7)) : ℕ :=
s.card = 3 ∧ s.to_list ∧ cond := _
     ∧ ∃ n ∈ chosen_points, _ -- should specify the conditions and derive conclusions based on those
-- Specify the conditions for not being collinear
theorem hexagon_triangles (S : Finset (Fin 7)) :
  S.card = 7 ∧ (∀ s ⊆ S, s.card = 3 → ¬collinear s) →
  (∃ t : Finset (Finset (Fin 7)), t.card = 32 ∧ (∀ x ∈ t, x.card = 3)) :=
by
  sorry

end hexagon_triangles_l404_404192


namespace binomial_coefficient_12_10_l404_404603

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404603


namespace binom_12_10_eq_66_l404_404602

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404602


namespace trains_meet_in_3_33_seconds_l404_404513

def train_meet_time
  (L₁ : ℕ)         -- Length of the first train in meters
  (L₂ : ℕ)         -- Length of the second train in meters
  (D : ℕ)          -- Initial distance between the trains in meters
  (S₁ : ℕ)         -- Speed of the first train in km/h
  (S₂ : ℕ)         -- Speed of the second train in km/h
  (km_per_hour_to_meter_per_sec : ℕ → ℝ) -- Conversion function from km/h to m/s
  : ℝ := 
    let S₁_mps := km_per_hour_to_meter_per_sec S₁   -- Speed of the first train in m/s
    let S₂_mps := km_per_hour_to_meter_per_sec S₂   -- Speed of the second train in m/s
    let relative_speed := S₁_mps + S₂_mps           -- Relative speed when trains head towards each other in m/s
    let effective_distance := D - (L₁ + L₂)         -- Adjusted distance considering train lengths
    effective_distance / relative_speed             -- Time to meet in seconds

theorem trains_meet_in_3_33_seconds
  : train_meet_time 100 200 450 90 72 (λ x, x * 1000 / 3600) = 3.33 := 
by 
  sorry

end trains_meet_in_3_33_seconds_l404_404513


namespace prove_value_expression_l404_404020

theorem prove_value_expression (p q : ℝ) (h :  \(\forall p q, 3x^2 + 9x - 21 = 0\) : 3x^2 + 9 * x - 21)) (h_sum : p + q = -3) (h_prod : p * q = -7) :
  (3 * p - 4) * (6 * q - 8) = -22 :=
by {
  sorry
}

end prove_value_expression_l404_404020


namespace limit_f_eq_1_l404_404621

open Real

-- Define the function f(n)
def f (n : ℕ) : ℝ :=
  (sqrt (n + 2) - (n ^ 3 + 2) ^ (1 / 3)) /
  ((n + 2) ^ (1 / 7) - (n ^ 5 + 2) ^ (1 / 5))

-- State the theorem
theorem limit_f_eq_1 : 
  tendsto f atTop (𝓝 1) :=
sorry

end limit_f_eq_1_l404_404621


namespace card_game_strategy_l404_404168

theorem card_game_strategy :
  ∀ (P : ℕ → ℕ) (m : ℕ),
    P 0 = 1 →
    (∀ k, P (k + 1) ≥ 4 * P k) →
    ∑ k in range (2^(m-1) * m), k ≥ 4^m :=
by sorry

end card_game_strategy_l404_404168


namespace no_such_n_exists_l404_404518

def pandiagonal_heterosquare (n : ℕ) : Prop :=
  ∀ (A : matrix (fin n) (fin n) ℕ), 
      (∀ i j, A i j < n * n + 1) ∧ 
      (unique (∑ i, A i (i+j)%n)) ∧ 
      (unique (∑ i, A (i+j)%n i)) ∧ 
      (unique (∑ i, A i ((i+j)%n))) ∧ 
      (unique (∑ i, A ((i+j)%n) i))

theorem no_such_n_exists : ¬ ∃ n : ℕ, ∀ (A : matrix (fin n) (fin n) ℕ), 
  (∀ i j, A i j < n * n + 1) ∧ 
  (unique (∑ i, A i (i+j)%n)) ∧ 
  (unique (∑ i, A (i+j)%n i)) ∧ 
  (unique (∑ i, A i ((i+j)%n))) ∧ 
  (unique (∑ i, A ((i+j)%n) i)) ∧ 
  (∃ k : ℕ, list.range (4 * n) = list.range k u+ 4 * n - 1) :=
sorry

end no_such_n_exists_l404_404518


namespace integral_cos_cos_integral_sin_sin_integral_cos_sin_integral_cos_integral_sin_l404_404151

open Real

variables (k l : ℤ)

-- 1. ∫_{-π}^{π} cos(kx) cos(lx) dx = if k = l then π else 0
theorem integral_cos_cos : 
  (∫ x in -π..π, cos (k * x) * cos (l * x)) = if k = l then π else 0 :=
sorry

-- 2. ∫_{-π}^{π} sin(kx) sin(lx) dx = if k = l then π else 0
theorem integral_sin_sin : 
  (∫ x in -π..π, sin (k * x) * sin (l * x)) = if k = l then π else 0 :=
sorry

-- 3. ∫_{-π}^{π} cos(kx) sin(lx) dx = 0
theorem integral_cos_sin :
  (∫ x in -π..π, cos (k * x) * sin (l * x)) = 0 :=
sorry

-- 4. ∫_{-π}^{π} cos(kx) dx = 0
theorem integral_cos :
  (∫ x in -π..π, cos (k * x)) = 0 :=
sorry

-- 5. ∫_{-π}^{π} sin(kx) dx = 0
theorem integral_sin :
  (∫ x in -π..π, sin (k * x)) = 0 :=
sorry

end integral_cos_cos_integral_sin_sin_integral_cos_sin_integral_cos_integral_sin_l404_404151


namespace no_pairs_for_arithmetic_progression_l404_404634

-- Define the problem in Lean
theorem no_pairs_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (2 * a = 5 + b) ∧ (2 * b = a * (1 + b)) :=
sorry

end no_pairs_for_arithmetic_progression_l404_404634


namespace a_6_is_3_times_4_pow_4_l404_404482

noncomputable def a_seq : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+1) := 3 * (Finset.range (n+1)).sum a_seq

theorem a_6_is_3_times_4_pow_4 : a_seq 6 = 3 * 4^4 := sorry

end a_6_is_3_times_4_pow_4_l404_404482


namespace george_collected_50_marbles_l404_404665

theorem george_collected_50_marbles (w y g r total : ℕ)
  (hw : w = total / 2)
  (hy : y = 12)
  (hg : g = y / 2)
  (hr : r = 7)
  (htotal : total = w + y + g + r) :
  total = 50 := by
  sorry

end george_collected_50_marbles_l404_404665


namespace largest_common_term_in_range_1_to_200_l404_404218

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l404_404218


namespace two_person_subcommittees_l404_404711

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l404_404711


namespace sin_angles_product_eq_one_eighth_l404_404969

theorem sin_angles_product_eq_one_eighth :
  sin (10 * π / 180) * sin (50 * π / 180) * sin (70 * π / 180) * sin (80 * π / 180) = 1 / 8 := by
  sorry

end sin_angles_product_eq_one_eighth_l404_404969


namespace rational_inequality_solution_set_l404_404485

open Set

theorem rational_inequality_solution_set :
  (x : ℝ) → x ≠ 1 → (x + 5) / (x - 1)^2 ≥ 2 ↔ x ∈ Icc (-1/2) 1 ∪ Ioc 1 3 :=
by
  intro x h
  sorry

end rational_inequality_solution_set_l404_404485


namespace complement_intersection_l404_404667

variable (R : Type*) [PartialOrder R]

def M : Set R := {x | (-1 : R) ≤ x ∧ x ≤ (2 : R)}
def N : Set R := {x | x ≤ (3 : R)}
def CuM : Set R := {x | x < (-1 : R) ∨ x > (2 : R)}

theorem complement_intersection (U : Set R) (hU : U = Set.univ) :
  (CuM \ M) ∩ N = {x | x < (-1 : R) ∨ (2 : R) < x ∧ x ≤ (3 : R)} :=
by
  sorry

end complement_intersection_l404_404667


namespace find_n_l404_404538

theorem find_n (n : ℕ) (h : n > 0) : 
  (3^n + 5^n) % (3^(n-1) + 5^(n-1)) = 0 ↔ n = 1 := 
by sorry

end find_n_l404_404538


namespace photographer_choices_l404_404507

theorem photographer_choices : 
  ∑ k in {4, 5}, Nat.choose 7 k = 56 :=
by
  sorry

end photographer_choices_l404_404507


namespace dna_diameter_scientific_notation_l404_404413

def dna_diameter : ℝ := 0.0000002
def scientific_notation := 2 * 10 ^ (-7)

theorem dna_diameter_scientific_notation : dna_diameter = scientific_notation := 
sorry

end dna_diameter_scientific_notation_l404_404413


namespace probability_of_shortest_diagonal_l404_404875

open_locale big_operators

def regular_polygon (n : ℕ) := n ≥ 3

noncomputable def shortest_diagonals (n : ℕ) : ℕ := n / 2

noncomputable def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem probability_of_shortest_diagonal (n : ℕ) (h : regular_polygon n) (h21 : n = 21) : 
  (shortest_diagonals n : ℚ) / total_diagonals n = 10 / 189 := 
by {
  sorry
}

end probability_of_shortest_diagonal_l404_404875


namespace dot_product_parabola_l404_404626

def parabolaC (x y : ℝ) : Prop := y^2 = 4 * x

def line (x : ℝ) : ℝ := (2 / 3) * (x + 2)

def is_focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def intersection_points := [(1, 2), (4, 4)]

def vector_diff (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_parabola :
  let F := (1, 0),
      M := (1, 2),
      N := (4, 4),
      FM := vector_diff F M,
      FN := vector_diff F N
  in dot_product FM FN = 8 := sorry

end dot_product_parabola_l404_404626


namespace nth_equation_l404_404810

theorem nth_equation (n : ℕ) (h : 0 < n) : 9 * (n - 1) + n = 10 * n - 9 := 
  sorry

end nth_equation_l404_404810


namespace work_completion_days_l404_404533

theorem work_completion_days (W : ℝ) (h1 : W > 0) :
  (1 / (1 / 15 + 1 / 10)) = 6 :=
by
  sorry

end work_completion_days_l404_404533


namespace rhombus_area_of_square_4_l404_404381

theorem rhombus_area_of_square_4 :
  let A := (0, 4)
  let B := (0, 0)
  let C := (4, 0)
  let D := (4, 4)
  let F := (0, 2)  -- Midpoint of AB
  let E := (4, 2)  -- Midpoint of CD
  let FG := 2 -- Half of the side of the square (since F and E are midpoints)
  let GH := 2
  let HE := 2
  let EF := 2
  let rhombus_FGEH_area := 1 / 2 * FG * EH
  rhombus_FGEH_area = 4 := sorry

end rhombus_area_of_square_4_l404_404381


namespace solid_properties_l404_404457

-- Definition of constants and conditions
def edge_length (c : ℝ) := c

def cube_vertices (A B C D E F G H : Point) : Set Point :=
  {A, B, C, D, E, F, G, H}

def midpoints (AD AE BF CD CG EF EH : Point) : Set Point := 
  {AD, AE, BF, CD, CG, EF, EH}

def face_centers (CDHG EFGH : Point) : Set Point := 
  {CDHG, EFGH}

def solid_vertices (A B C AD AE BF CD CG EF EH CDHG EFGH : Point) : Set Point :=
  {A, B, C, AD, AE, BF, CD, CG, EF, EH, CDHG, EFGH}

-- The main theorem that encapsulates the proof problems
theorem solid_properties 
  (A B C D E F G H AD AE BF CD CG EF EH CDHG EFGH: Point) 
  (c : ℝ) 
  (hc : 0 < c) : 
  let s := solid_vertices A B C AD AE BF CD CG EF EH CDHG EFGH in
  (surface_area s = 4.506 * c^2) ∧ 
  (volume s = (11 * c^3) / 16) ∧ 
  (longest_diagonal s = (3 * c) / 2) := sorry

end solid_properties_l404_404457


namespace collinear_G_I_J_l404_404770

open EuclideanGeometry

variables {A B C D E F G H I J : Point}

-- Definitions of the points and their relationships
axiom D_interior_triangle_ABC : ∃ (A B C D : Point), D ≠ A ∧ D ≠ B ∧ D ≠ C ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
triangle A B C ∧ isInterior D (triangle A B C)

axiom BD_inter_AC_at_E : ∃ (B D E : Point), line B D ∧ line A C ∧ E = intersectionOfLines (line B D) (line A C)

axiom CD_inter_AB_at_F : ∃ (C D F : Point), line C D ∧ line A B ∧ F = intersectionOfLines (line C D) (line A B)

axiom EF_inter_BC_at_G : ∃ (E F G : Point), line E F ∧ line B C ∧ G = intersectionOfLines (line E F) (line B C)

axiom H_on_AD : ∃ (A D H : Point), line A D ∧ onLine H (line A D)

axiom HF_inter_BD_at_I : ∃ (H F I : Point), line H F ∧ line B D ∧ I = intersectionOfLines (line H F) (line B D)

axiom HE_inter_CD_at_J : ∃ (H E J : Point), line H E ∧ line C D ∧ J = intersectionOfLines (line H E) (line C D)

-- Prove G, I, and J are collinear
theorem collinear_G_I_J (A B C D E F G H I J : Point) 
    (h1 : triangle A B C)
    (h2 : isInterior D (triangle A B C))
    (h3 : E = intersectionOfLines (line B D) (line A C))
    (h4 : F = intersectionOfLines (line C D) (line A B))
    (h5 : G = intersectionOfLines (line E F) (line B C))
    (h6 : onLine H (line A D))
    (h7 : I = intersectionOfLines (line H F) (line B D))
    (h8 : J = intersectionOfLines (line H E) (line C D)) :
  collinear {G, I, J} :=
sorry

end collinear_G_I_J_l404_404770


namespace largest_4_digit_divisible_by_88_and_prime_gt_100_l404_404519

noncomputable def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem largest_4_digit_divisible_by_88_and_prime_gt_100 (p : ℕ) (hp : is_prime p) (h1 : 100 < p):
  ∃ n, is_4_digit n ∧ is_divisible_by n 88 ∧ is_divisible_by n p ∧
       (∀ m, is_4_digit m ∧ is_divisible_by m 88 ∧ is_divisible_by m p → m ≤ n) :=
sorry

end largest_4_digit_divisible_by_88_and_prime_gt_100_l404_404519


namespace sector_radius_l404_404560

theorem sector_radius (n : ℝ) (l : ℝ) (h1 : n = 90) (h2 : l = 3 * Real.pi) : 
  let π : ℝ := Real.pi in
  let r := (180 * l) / (n * π) in 
  r = 6 := 
by {
  -- This is where the proof would go.
  sorry
}

end sector_radius_l404_404560


namespace maximal_planar_is_planar_maximal_planar_edge_condition_l404_404154

-- Definition of planar graph based on Euler's formula
def is_planar (G : Graph) (V E F : ℕ) : Prop :=
  V - E + F = 2

-- Definition of a maximal planar graph
def is_maximal_planar (G : Graph) : Prop :=
  ∀ G' : Graph, (G ⊆ G' ∧ is_planar G' V' E' F') → G = G'

-- Definition for the number of edges in a simple planar graph with bounded faces
def num_edges_face_constraints (E F : ℕ) : Prop :=
  2 * E ≥ 3 * F

-- Proving Part (i)
theorem maximal_planar_is_planar (G : Graph) (V E F : ℕ) (H : is_maximal_planar G) : is_planar G V E F :=
sorry
  
-- Proving Part (ii)
theorem maximal_planar_edge_condition (G : Graph) (V E F : ℕ) (H1 : V ≥ 3) (H2 : is_planar G V E F) (H3 : num_edges_face_constraints E F) :
  (E = 3 * V - 6) ↔ is_maximal_planar G :=
sorry

end maximal_planar_is_planar_maximal_planar_edge_condition_l404_404154


namespace pencils_added_by_sara_l404_404854

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l404_404854


namespace prob_three_girls_l404_404908

theorem prob_three_girls (total_members boys girls : ℕ) (h_total: total_members = 12) (h_boys: boys = 7) (h_girls: girls = 5)
(h_comb: ∀ n k, nat.choose n k = nat.factorial n / (nat.factorial k * nat.factorial (n - k))) :
  (nat.choose girls 3 : ℚ) / (nat.choose total_members 3) = 1 / 22 := 
 by sorry

end prob_three_girls_l404_404908


namespace number_of_people_who_like_apple_is_40_l404_404158

variable {People : Type}

-- Define the subsets of people who like each combination of fruits
variable (A O M : set People)

-- Given conditions
axiom like_apple   : ∀ p : People, p ∈ A
axiom dislike_apple_orange_mango : ∀ p : People, p ∈ O ∩ M \ A
axiom like_mango_apple_dislike_orange : ∀ p : People, p ∈ M ∩ A \ O
axiom like_all : ∀ p : People, p ∈ A ∩ O ∩ M

-- Prove the number of people who like apple is 40
theorem number_of_people_who_like_apple_is_40 : 
  card (A : set People) = 40 := 
by sorry

end number_of_people_who_like_apple_is_40_l404_404158


namespace product_of_distinct_prime_factors_of_B_l404_404400

theorem product_of_distinct_prime_factors_of_B (B : ℕ) (hB : B = ∏ n in (finset.filter (∣ 60) (finset.range (60+1))), n) : 
    (∏ p in (finset.filter (prime) (nat.prime_divisors B)), p) = 30 :=
by
  sorry

end product_of_distinct_prime_factors_of_B_l404_404400


namespace integral_quarter_circle_l404_404230

theorem integral_quarter_circle :
  ∫ x in 0..2, (sqrt (4 - x^2)) = Real.pi :=
sorry

end integral_quarter_circle_l404_404230


namespace inversions_range_l404_404058

/-- Given any permutation of 10 elements, 
    the number of inversions (or disorders) in the permutation 
    can take any value from 0 to 45.
-/
theorem inversions_range (perm : List ℕ) (h_length : perm.length = 10):
  ∃ S, 0 ≤ S ∧ S ≤ 45 :=
sorry

end inversions_range_l404_404058


namespace f_nonsquare_formula_l404_404283

-- Define f(n) as the n-th positive nonsquare integer.
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def n_th_nonsquare (n : ℕ) : ℕ :=
  if h : n > 0 then
  (Nat.find (λ m, Nat.card (_root_.finset.filter (λ (x : ℕ), ¬ is_perfect_square x) (_root_.finset.range m)).card = n) : ℕ)
  else 0

-- Define the rounding to the nearest integer function.
def closest_int (x : ℝ) : ℤ :=
  round x

-- The theorem we want to prove.
theorem f_nonsquare_formula (n : ℕ) : n > 0 → n_th_nonsquare n = n + closest_int (real.sqrt n) := by
  sorry

end f_nonsquare_formula_l404_404283


namespace rain_probability_conditioned_l404_404707

theorem rain_probability_conditioned :
  let P_rain := (1 : ℚ) / 3,
  let P_consec_rain := (1 : ℚ) / 5,
  let p := P_consec_rain / P_rain
  in p = 3 / 5 := by
  sorry

end rain_probability_conditioned_l404_404707


namespace average_cost_per_trip_is_correct_l404_404422

def oldest_pass_cost : ℕ := 100
def second_oldest_pass_cost : ℕ := 90
def third_oldest_pass_cost : ℕ := 80
def youngest_pass_cost : ℕ := 70

def oldest_trips : ℕ := 35
def second_oldest_trips : ℕ := 25
def third_oldest_trips : ℕ := 20
def youngest_trips : ℕ := 15

def total_cost : ℕ := oldest_pass_cost + second_oldest_pass_cost + third_oldest_pass_cost + youngest_pass_cost
def total_trips : ℕ := oldest_trips + second_oldest_trips + third_oldest_trips + youngest_trips

def average_cost_per_trip : ℚ := total_cost / total_trips

theorem average_cost_per_trip_is_correct : average_cost_per_trip = 340 / 95 :=
by sorry

end average_cost_per_trip_is_correct_l404_404422


namespace no_solutions_for_a_l404_404993

theorem no_solutions_for_a (a : ℝ) : (∀ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 4 * a ≠ 0) ↔ a ∈ set.Iio (-8) ∪ set.Ioi 0 := by
  sorry

end no_solutions_for_a_l404_404993


namespace sixth_graders_bought_more_pencils_23_l404_404066

open Int

-- Conditions
def pencils_cost_whole_number_cents : Prop := ∃ n : ℕ, n > 0
def seventh_graders_total_cents := 165
def sixth_graders_total_cents := 234
def number_of_sixth_graders := 30

-- The number of sixth graders who bought more pencils than seventh graders
theorem sixth_graders_bought_more_pencils_23 :
  (seventh_graders_total_cents / 3 = 55) ∧
  (sixth_graders_total_cents / 3 = 78) →
  78 - 55 = 23 :=
by
  sorry

end sixth_graders_bought_more_pencils_23_l404_404066


namespace lucy_cookie_packs_l404_404414

theorem lucy_cookie_packs (total_packs noodle_packs cookie_packs : ℕ)
  (h_total : total_packs = 28)
  (h_noodles : noodle_packs = 16)
  (h_groceries : total_packs = cookie_packs + noodle_packs) : 
    cookie_packs = 12 := 
by
  rw [h_total, h_noodles, h_groceries]
  sorry

end lucy_cookie_packs_l404_404414


namespace eq_solutions_of_sqrt_eq_cbrt_l404_404397
  open Nat

  theorem eq_solutions_of_sqrt_eq_cbrt (a : ℕ) (x : ℚ) (h : a ≥ 0) :
    (√(1 + (a - 1) * (cbrt x)) = √(1 + (a - 1) * √x)) → (x = 0 ∨ x = 1) :=
  by
    sorry
  
end eq_solutions_of_sqrt_eq_cbrt_l404_404397


namespace sum_of_distinct_prime_factors_l404_404273

theorem sum_of_distinct_prime_factors (a b c : ℕ) (h1 : a = 7^4 - 7^2) (h2 : b = 2) (h3 : c = 3) (h4 : 2 + 3 + 7 = 12): 
  ∃ d : ℕ, a.prime_factors.sum = d ∧ d = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l404_404273


namespace count_multiples_of_3_count_remainder_1_mod_3_2_mod_4_l404_404210

-- Define the predicate for the multiples of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the predicate for leaving a remainder of 1 when divided by 3 and 2 when divided by 4
def leaves_remainder_1_mod_3_2_mod_4 (n : ℕ) : Prop :=
  n % 3 = 1 ∧ n % 4 = 2

-- Lean statement for proving the number of multiples of 3 from 1 to 2015
theorem count_multiples_of_3 : (finset.range 2016).filter is_multiple_of_3 .card = 671 := sorry

-- Lean statement for proving the number of integers from 1 to 2015 that leave a remainder of 1 when divided by 3 and 2 when divided by 4
theorem count_remainder_1_mod_3_2_mod_4 : (finset.range 2016).filter leaves_remainder_1_mod_3_2_mod_4 .card = 167 := sorry

end count_multiples_of_3_count_remainder_1_mod_3_2_mod_4_l404_404210


namespace expected_rainfall_l404_404179

theorem expected_rainfall (n : ℕ) (p_sun p_rain3 p_rain8 : ℕ) (rain3 rain8 : ℕ) (expected_weekly_rainfall : ℚ) :
  n = 7 →
  p_sun = 30 →
  p_rain3 = 35 →
  p_rain8 = 35 →
  rain3 = 3 →
  rain8 = 8 →
  expected_weekly_rainfall = (7 : ℚ) * ((p_sun / (100 : ℕ)) * (0 : ℚ) + (p_rain3 / (100 : ℕ)) * (rain3 : ℚ) + (p_rain8 / (100 : ℕ)) * (rain8 : ℚ)) →
  expected_weekly_rainfall ≈ 26.9 :=
by
  intros, 
  sorry -- The full proof can be completed here.

end expected_rainfall_l404_404179


namespace binom_12_10_eq_66_l404_404601

theorem binom_12_10_eq_66 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l404_404601


namespace sum_of_ns_l404_404622

theorem sum_of_ns (n m : Nat) (H : ∀ n m, n ≥ 0 ∧ m ≥ 0 → (5 * 3^m) + 4 = n^2) :
  ∑ n in {n ∈ Finset.range (100) | ∃ m, n ≥ 0 ∧ m ≥ 0 ∧ (5 * 3^m) + 4 = n^2}, n = 10 := 
sorry

end sum_of_ns_l404_404622


namespace meal_combinations_count_l404_404738

/-- Define the number of menu items -/
def num_menu_items : ℕ := 15

/-- Define the number of distinct combinations of meals Maryam and Jorge can order,
    considering they may choose the same dish and distinguishing who orders what -/
theorem meal_combinations_count (maryam_dishes jorge_dishes : ℕ) : 
  maryam_dishes = num_menu_items ∧ jorge_dishes = num_menu_items → 
  maryam_dishes * jorge_dishes = 225 :=
by
  intros h
  simp only [num_menu_items] at h -- Utilize the definition of num_menu_items
  sorry

end meal_combinations_count_l404_404738


namespace fraction_area_is_approx_0_1118_l404_404425

-- Define the coordinates of the points
def A := (2.5, 0.5)
def B := (7.8, 12.4)
def C := (14.6, 0.2)
def X := (6.1, 0.3)
def Y := (8.3, 4.7)
def Z := (9.7, 0.1)

-- Define a function to calculate the area of a triangle using determinant
def area (p1 p2 p3 : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Calculate the areas
def area_ABC : ℝ := area A B C
def area_XYZ : ℝ := area X Y Z

-- Calculate the fraction
def fraction_area : ℝ := area_XYZ / area_ABC

-- Prove that the fraction is approximately 0.1118
theorem fraction_area_is_approx_0_1118 : fraction_area ≈ 0.1118 :=
sorry

end fraction_area_is_approx_0_1118_l404_404425


namespace range_of_f_l404_404103

noncomputable def f (x : ℝ) : ℝ := (1/2)^(2*x - x^2)

theorem range_of_f :
  set.range f = set.Ici (1/2) :=
sorry

end range_of_f_l404_404103


namespace point_P_independent_of_X_l404_404021

variable {ABC : Type} [Triangle ABC]
variable {D X : ABC} (on_BC : ∃ bc : Segment ABC, D ∈ bc ∧ X ∈ bc ∧ D ≠ X)
variable {Y : Point} (Y_def : ∃ c : Circus (TriangleCircumcircle ABC), Y ∈ c ∧ LineThrough ABC X Y)
variable {P : Point} (P_def : ∃ c1 c2 : Circus, c1 = TriangleCircumcircle ABC ∧ c2 = CircusCircumcircle D X Y ∧ P ∈ c1 ∧ P ∈ c2 ∧ P ≠ LineThrough ABC X)

theorem point_P_independent_of_X : P_independent_of_X
  (on_BC : ∃ bc : Segment ABC, D ∈ bc ∧ X ∈ bc ∧ D ≠ X)
  (Y_def : ∃ c : Circus (TriangleCircumcircle ABC), Y ∈ c ∧ LineThrough ABC X Y)
  (P_def : ∃ c1 c2 : Circus, c1 = TriangleCircumcircle ABC ∧ c2 = CircusCircumcircle D X Y ∧ P ∈ c1 ∧ P ∈ c2 ∧ P ≠ LineThrough ABC X) : 
  ∀ (X' : ABC), (X' ∈ bc ∧ X' ≠ D) → P = P' :=
begin
  sorry
end

end point_P_independent_of_X_l404_404021


namespace at_least_four_white_rooks_l404_404814

-- Define a type for the chessboard position
structure Position where
  x : Fin 8
  y : Fin 8

-- Given: An 8x8 chessboard and 8 rooks with no two attacking each other
constant is_non_attacking : List Position → Prop

-- Given: Three specific positions of rooks on white squares
constant rook_positions : List Position

-- Check if a position is white
def is_white (pos : Position) : Bool :=
  (pos.x.1 + pos.y.1) % 2 = 1

-- The main theorem that needs to be proved
theorem at_least_four_white_rooks : 
  (is_non_attacking rook_positions) →
  (∃ p1 p2 p3 : Position, p1 ∈ rook_positions ∧ p2 ∈ rook_positions ∧ p3 ∈ rook_positions ∧
                         p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
                         is_white p1 ∧ is_white p2 ∧ is_white p3) →
  ∃ p4 : Position, p4 ∈ rook_positions ∧ is_white p4 := by
  sorry

end at_least_four_white_rooks_l404_404814


namespace lcm_Anthony_Bethany_Casey_Dana_l404_404569

theorem lcm_Anthony_Bethany_Casey_Dana : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 10) = 120 := 
by
  sorry

end lcm_Anthony_Bethany_Casey_Dana_l404_404569


namespace binomial_12_10_eq_66_l404_404613

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404613


namespace molecular_weight_not_related_to_osmotic_pressure_l404_404528

-- Definition for the content of protein in plasma
def protein_content (plasma: Type) : Prop := sorry

-- Definition for the content of Cl- in plasma
def cl_content (plasma: Type) : Prop := sorry

-- Definition for the molecular weight of plasma protein
def protein_molecular_weight (plasma: Type) : Prop := sorry

-- Definition for the content of Na+ in plasma
def na_content (plasma: Type) : Prop := sorry

-- The statement to prove
theorem molecular_weight_not_related_to_osmotic_pressure (plasma: Type) :
  ¬ (protein_molecular_weight plasma → plasma)=>
  ((protein_content plasma ∧ cl_content plasma ∧ na_content plasma) → plasma) :=
begin
  sorry
end

end molecular_weight_not_related_to_osmotic_pressure_l404_404528


namespace complex_sum_imaginary_l404_404495

theorem complex_sum_imaginary {a c d e f : ℝ} (h1 : e = -a - c)
  (h2 : (a + 2 * Complex.i) + (c + d * Complex.i) + (e + f * Complex.i) = 2 * Complex.i) :
  d + f = 0 :=
by sorry

end complex_sum_imaginary_l404_404495


namespace intersection_distance_iff_slope_l404_404177

variable {k : ℝ}
def line_through_origin (x : ℝ) : ℝ := k * x

def circle (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 2*y + 4 = 0

-- Given that the line through the origin with slope k intersects the circle at points A and B
theorem intersection_distance_iff_slope :
  ∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧
  line_through_origin A.1 = A.2 ∧ line_through_origin B.1 = B.2 ∧ 
  (dist A B = 2 ↔ k = 1/2) :=
sorry

end intersection_distance_iff_slope_l404_404177


namespace number_of_terms_added_l404_404514

theorem number_of_terms_added (k : ℕ) (h : 1 ≤ k) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k :=
by sorry

end number_of_terms_added_l404_404514


namespace triangle_area_range_l404_404792

noncomputable def tangent_lines_area (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < 1) (h3 : 1 < x2) : ℝ :=
  let y1 := -Real.log x1
  let y2 := Real.log x2
  let k1 := -1/x1
  let k2 := 1/x2
  let l1 (x : ℝ) := k1 * (x - x1) + y1
  let l2 (x : ℝ) := k2 * (x - x2) + y2
  let A_y := l1 0
  let B_y := l2 0
  let AB := Real.abs (A_y - B_y)
  let xP := 2 * x1 * x2 / (x1 + x2)
  (1/2) * AB * xP

theorem triangle_area_range : ∀ x1 x2 : ℝ, 
  0 < x1 → x1 < 1 → 1 < x2 →
  0 < tangent_lines_area x1 x2 (by linarith) (by linarith) (by linarith) < 1 :=
by
  intros x1 x2 h1 h2 h3
  sorry

end triangle_area_range_l404_404792


namespace independence_test_most_accurate_method_l404_404508

theorem independence_test_most_accurate_method
  (X Y : Type)
  (contingency_table : X → Y → ℕ)
  (methods : list (Type)) :
  let independence_test := independence_test_method contingency_table in
  independence_test ∈ methods → 
  (∀ method ∈ methods, method ≠ independence_test → ¬more_accurate method independence_test) :=
by
  sorry

end independence_test_most_accurate_method_l404_404508


namespace sum_of_squares_is_perfect_square_l404_404996

theorem sum_of_squares_is_perfect_square (n p k : ℤ) : 
  (∃ m : ℤ, n^2 + p^2 + k^2 = m^2) ↔ (n * k = (p / 2)^2) :=
by
  sorry

end sum_of_squares_is_perfect_square_l404_404996


namespace find_a_iset_l404_404146

noncomputable def theta (m : ℤ) (hm : 1 < m ∧ m % 2 = 1) : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I / (2 * m))

theorem find_a_iset (m : ℤ) (hm : 1 < m ∧ m % 2 = 1) (n : ℤ := 2 * m) (theta := theta m hm) :
  (∑ i in Finset.range m.filter (λ k, k % 2 = 1), theta ^ i) = 1 / (1 - theta) :=
by
  sorry

end find_a_iset_l404_404146


namespace train_cable_car_distance_and_speeds_l404_404187
-- Import necessary libraries

-- Defining the variables and conditions
variables (s v1 v2 : ℝ)
variables (half_hour_sym_dist additional_distance quarter_hour_meet : ℝ)

-- Defining the conditions
def conditions :=
  (half_hour_sym_dist = v1 * (1 / 2) + v2 * (1 / 2)) ∧
  (additional_distance = 2 / v2) ∧
  (quarter_hour_meet = 1 / 4) ∧
  (v1 + v2 = 2 * s) ∧
  (v2 * (additional_distance + half_hour_sym_dist) = (v1 * (additional_distance + half_hour_sym_dist) - s)) ∧
  ((v1 + v2) * (half_hour_sym_dist + additional_distance + quarter_hour_meet) = 2 * s)

-- Proving the statement
theorem train_cable_car_distance_and_speeds
  (h : conditions s v1 v2 half_hour_sym_dist additional_distance quarter_hour_meet) :
  s = 24 ∧ v1 = 40 ∧ v2 = 8 := sorry

end train_cable_car_distance_and_speeds_l404_404187


namespace min_AB_distance_l404_404737

open Real

variables {x y λ : ℝ}
namespace proof

-- Define the points and conditions
def point_P : (x=-1) := sorry
def point_F := (1:ℝ, 0:ℝ)
def point_Q (P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ × ℝ := ((P.1 + F.1) / 2, (P.2 + F.2) / 2)
def condition_perpendicular (Q M P : ℝ × ℝ) : Prop := ((M.2 - Q.2) / (M.1 - Q.1)) * ((P.2 - Q.2) / (P.1 - Q.1)) = -1
def condition_vector (M P F : ℝ × ℝ) (λ : ℝ) : Prop := (M.1 - P.1, M.2 - P.2) = λ *! (F.1, F.2)
def circle_center : ℝ × ℝ := (3, 0)
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2
def tangent_points (M A B : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) : Prop := 
  ((A.1 - C.1)^2 + (A.2 - C.2)^2 = r) ∧ ((B.1 - C.1)^2 + (B.2 - C.2)^2 = r) ∧ 
  (M.1 = (A.1 + B.1) / 2) ∧ (M.2 = (A.2 + B.2) / 2)

-- Define the theorem
theorem min_AB_distance {λ : ℝ} (P F : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (M A B : ℝ × ℝ) :
  point_P P →
  F = point_F →
  let Q := point_Q P F in 
  condition_perpendicular Q M P →
  condition_vector M P F λ →
  C = circle_center → 
  circle_eqn C.1 C.2 →
  tangent_points M A B C 2 →
  (distance A B = sqrt 6) := sorry

end proof

end min_AB_distance_l404_404737


namespace graph_not_in_third_quadrant_l404_404466

-- Define the conditions
variable (m : ℝ)
variable (h1 : 0 < m)
variable (h2 : m < 2)

-- Define the graph equation
noncomputable def line_eq (x : ℝ) : ℝ := (m - 2) * x + m

-- The proof problem: the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ¬ ∃ x y : ℝ, (x < 0 ∧ y < 0 ∧ y = (m - 2) * x + m) :=
sorry

end graph_not_in_third_quadrant_l404_404466


namespace integral_of_f_l404_404017

noncomputable def f (x : ℝ) : ℝ := 2^abs x

theorem integral_of_f :
  ∫ x in -2..4, f x = 18 / Real.log 2 :=
by
  sorry  -- Proof omitted

end integral_of_f_l404_404017


namespace seashell_count_l404_404128

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end seashell_count_l404_404128


namespace cost_of_cookies_and_board_game_l404_404390

theorem cost_of_cookies_and_board_game:
  let cost_bracelet := 1
  let cost_necklace := 2
  let cost_ring := 0.5
  let sell_bracelet := 1.5
  let sell_necklace := 3
  let sell_ring := 1
  let min_bracelets := 5
  let min_necklaces := 3
  let min_rings := 10
  let profit_margin := 0.5
  let remaining_money := 5
  let work_hours_per_day := 2
  let days_in_week := 7
  let time_bracelet := 10 / 60 -- 10 minutes converted to hours
  let time_necklace := 15 / 60 -- 15 minutes converted to hours
  let time_ring := 5 / 60 -- 5 minutes converted to hours
  in
  -- Calculate the profits
  let profit_bracelet := sell_bracelet - cost_bracelet
  let profit_necklace := sell_necklace - cost_necklace
  let profit_ring := sell_ring - cost_ring
  let total_profit := (min_bracelets * profit_bracelet) + (min_necklaces * profit_necklace) + (min_rings * profit_ring)
  -- Calculate the total cost of materials
  let total_cost := (min_bracelets * cost_bracelet) + (min_necklaces * cost_necklace) + (min_rings * cost_ring)
  -- Calculate total sales required for 50% profit margin
  let required_sales := total_cost * (1 + profit_margin)
  -- Calculate the total amount Josh had before purchase
  let total_amount_before_purchase := required_sales + remaining_money
  -- Deduce the cost of the box of cookies and board game together
  total_amount_before_purchase - remaining_money = 29 - 5 := 24 := sorry

end cost_of_cookies_and_board_game_l404_404390


namespace probability_of_prime_sum_l404_404639

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the sets of numbers on Spinner 1 and Spinner 2
def Spinner1 : Finset ℕ := {1, 2, 3, 4}
def Spinner2 : Finset ℕ := {3, 4, 5, 6}

-- Calculate the set of possible sums
def possible_sums : Finset ℕ := Finset.image₂ (· + ·) Spinner1 Spinner2

-- Filter out the sums that are prime
def prime_sums : Finset ℕ := possible_sums.filter is_prime

-- Prove that the probability of the sum being prime is 5/16
theorem probability_of_prime_sum : 
  (prime_sums.card : ℚ) / (Spinner1.card * Spinner2.card : ℚ) = 5 / 16 := 
by 
  -- Placeholder proof
  sorry

end probability_of_prime_sum_l404_404639


namespace parallel_planes_l404_404526

variable {α β γ : Type} [plane α] [plane β] [plane γ]

-- Definition of parallel planes
def planes_parallel (p1 p2 : Type) [plane p1] [plane p2] : Prop := sorry

-- Given conditions
variable (h : planes_parallel γ α ∧ planes_parallel γ β)

-- Prove that \(\alpha \parallel \beta\)
theorem parallel_planes (h : planes_parallel γ α ∧ planes_parallel γ β) : planes_parallel α β := by
  sorry

end parallel_planes_l404_404526


namespace range_of_function_l404_404882

def range_exclusion (x : ℝ) : Prop :=
  x ≠ 1

theorem range_of_function :
  set.range (λ x : ℝ, if x = -2 then (0 : ℝ) else x + 3) = {y : ℝ | range_exclusion y} :=
by 
  sorry

end range_of_function_l404_404882


namespace coefficient_of_x2_in_binomial_expansion_of_1_plus_2x_power_7_l404_404460

theorem coefficient_of_x2_in_binomial_expansion_of_1_plus_2x_power_7 :
  (coeff (expand (1 + 2 * X)^7) 2 = 84) :=
sorry

end coefficient_of_x2_in_binomial_expansion_of_1_plus_2x_power_7_l404_404460


namespace julie_earnings_l404_404006

def mowing_rate : ℝ := 4
def weeding_rate : ℝ := 8
def mowing_hours : ℝ := 25
def weeding_hours : ℝ := 3

theorem julie_earnings :
  let september_earnings := (mowing_hours * mowing_rate) + (weeding_hours * weeding_rate)
  in september_earnings + september_earnings = 248 :=
by
  let september_earnings := (mowing_hours * mowing_rate) + (weeding_hours * weeding_rate)
  show september_earnings + september_earnings = 248
  sorry

end julie_earnings_l404_404006


namespace intersection_of_sets_l404_404334

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def set_B : Set ℝ := {x | (x + 1) * (x - 4) > 0}

theorem intersection_of_sets :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ {x | (x + 1) * (x - 4) > 0} = {x | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l404_404334


namespace tent_height_max_volume_l404_404537

/-- Given an isosceles triangle ABC with AC = BC = 5 m and AB = 6 m,
    and D as the midpoint of AB, 
    prove that the height of the tent (the distance from D to the base of triangle ABC) 
    when the volume is maximized is 12 / sqrt 41 m. -/

theorem tent_height_max_volume :
  ∀ (A B C D : ℝ)
    (h1 : dist A C = 5)
    (h2 : dist B C = 5)
    (h3 : dist A B = 6)
    (h4 : D = (A + B) / 2),
  let h_max : ℝ := 12 / real.sqrt 41 in
  sorry

end tent_height_max_volume_l404_404537


namespace find_x_between_0_and_180_l404_404655

open Real

-- Condition: let x be a real number between 0 and 180 degrees.
variable (x : ℝ)
#check degree to radian conversion functions
def degrees (d : ℝ) : ℝ := d * (π / 180)

-- Conditions are in degrees, so we define and use a conversion to radians
def tan_condition : Prop :=
  tan (degrees (100 - x)) = (sin (degrees 100) - sin (degrees x)) / (cos (degrees 100) - cos (degrees x))

-- Main statement
theorem find_x_between_0_and_180 (h : 0 < x ∧ x < 180) (h1 : tan_condition x) : x = 80 :=
  sorry

end find_x_between_0_and_180_l404_404655


namespace dice_probability_l404_404577

theorem dice_probability :
  (∃ (d1 d2 d3 d4 d5 d6 : ℕ), 
       1 ≤ d1 ∧ d1 ≤ 10 ∧
       1 ≤ d2 ∧ d2 ≤ 10 ∧ 
       1 ≤ d3 ∧ d3 ≤ 10 ∧
       1 ≤ d4 ∧ d4 ≤ 10 ∧
       1 ≤ d5 ∧ d5 ≤ 10 ∧
       1 ≤ d6 ∧ d6 ≤ 10 ∧
       (λ n, n < 6).count ([d1, d2, d3, d4, d5, d6]) = 3) → 
  (probability that exactly three of the 6 dice show a number less than 6) = 5 / 16 := 
by sorry

end dice_probability_l404_404577


namespace hyperbola_eccentricity_proof_l404_404819

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let e := (c : ℝ) / a in
  e

theorem hyperbola_eccentricity_proof (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (intersect_condition : ∃ P : ℝ × ℝ, 
                          (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) 
                          ∧ (P.1 ^ 2 + P.2 ^ 2 = a ^ 2 + b ^ 2)
                          ∧ (0 ≤ P.1) ∧ (0 ≤ P.2))
  (F1 F2 : ℝ × ℝ) (foci_condition: |P.1 - F1.1| = 3 * |P.1 - F2.1|)
  : eccentricity_of_hyperbola a b h_a h_b = (real.sqrt 10) / 2 := 
sorry

end hyperbola_eccentricity_proof_l404_404819


namespace sum_of_exponents_l404_404830

-- Defining the given expression
def expr : ℕ → ℕ → ℕ → ℕ ≡ ℕ := λ (a b d : ℕ), 54 * a ^ 5 * b ^ 9 * d ^ 14

-- Stating the theorem to show that the sum of the exponents of the variables outside the radical is 8
theorem sum_of_exponents (a b d : ℕ) : 
  ∑ (x : ℕ) in ({(x → let c := x / 3 in c)}).erase 0, x = 8 := by 
  sorry

end sum_of_exponents_l404_404830


namespace lines_are_coplanar_l404_404242

-- Define the first line
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, 2 - m * t, 6 + t)

-- Define the second line
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (4 + m * u, 5 + 3 * u, 8 + 2 * u)

-- Define the vector connecting points on the lines when t=0 and u=0
def connecting_vector : ℝ × ℝ × ℝ :=
  (1, 3, 2)

-- Define the cross product of the direction vectors
def cross_product (m : ℝ) : ℝ × ℝ × ℝ :=
  ((-2 * m - 3), (m + 2), (6 + 2 * m))

-- Prove that lines are coplanar when m = -9/4
theorem lines_are_coplanar : ∃ k : ℝ, ∀ m : ℝ,
  cross_product m = (k * 1, k * 3, k * 2) → m = -9/4 :=
by
  sorry

end lines_are_coplanar_l404_404242


namespace determine_V_3034_l404_404282

variables {α : Type} [ordered_semiring α]

-- Define an arithmetic sequence as a function
def arith_seq (b : α) (e : α) (n : ℕ) : α := b + n * e

-- Define the sum of the first n terms of an arithmetic sequence
def U (b : α) (e : α) (n : ℕ) : α :=
  (n * (2 * b + (n - 1) * e)) / 2

-- Define the sum of U_k up to n
def V (b : α) (e : α) (n : ℕ) : α :=
  ∑ k in range(1 + n), U b e k

-- Problem statement: Given U_2023, uniquely determine V_3034
theorem determine_V_3034 (b e : α) (U2023 : α) :
  U b e 2023 = U2023 →
  ∃ V3034 : α, V b e 3034 = V3034 :=
by
  sorry -- Proof goes here

end determine_V_3034_l404_404282


namespace number_of_real_solutions_l404_404235

open Real

def f (x : ℝ) : ℝ := 3 * 2 ^ (floor (log (2 : ℝ) x)) - x

theorem number_of_real_solutions : 
  ∃! (n : ℕ), n = 9 ∧ (∃ x : ℝ, x > 0 ∧ (f x)^16 = 2022 * x^13) := 
sorry

end number_of_real_solutions_l404_404235


namespace binomial_12_10_eq_66_l404_404614

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404614


namespace common_number_in_sequences_l404_404215

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end common_number_in_sequences_l404_404215


namespace quadratic_function_value_at_18_l404_404226

noncomputable def p (d e f x : ℝ) : ℝ := d*x^2 + e*x + f

theorem quadratic_function_value_at_18
  (d e f : ℝ)
  (h_sym : ∀ x1 x2 : ℝ, p d e f 6 = p d e f 12)
  (h_max : ∀ x : ℝ, x = 10 → ∃ p_max : ℝ, ∀ y : ℝ, p d e f x ≤ p_max)
  (h_p0 : p d e f 0 = -1) : 
  p d e f 18 = -1 := 
sorry

end quadratic_function_value_at_18_l404_404226


namespace mike_profit_l404_404804

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l404_404804


namespace train_length_is_500_l404_404925

def speed_kmph : ℕ := 360
def time_sec : ℕ := 5

def speed_mps (v_kmph : ℕ) : ℕ :=
  v_kmph * 1000 / 3600

def length_of_train (v_mps : ℕ) (t_sec : ℕ) : ℕ :=
  v_mps * t_sec

theorem train_length_is_500 :
  length_of_train (speed_mps speed_kmph) time_sec = 500 := 
sorry

end train_length_is_500_l404_404925


namespace sides_equal_if_condition_l404_404760

noncomputable theory

-- Define the triangle angles and sides
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the given condition in the problem
def given_condition (A B : ℝ) (a b : ℝ) : Prop :=
  a * Real.tan A + b * Real.tan B = (a + b) * Real.tan ((A + B) / 2)

-- State the theorem to prove a = b
theorem sides_equal_if_condition (A B C : ℝ) (a b c : ℝ) (h : given_condition A B a b):
  a = b :=
sorry

end sides_equal_if_condition_l404_404760


namespace prob_large_apple_is_large_l404_404363

def large_to_small_prob : ℝ := 0.05
def small_to_large_prob : ℝ := 0.02
def large_apple_ratio : ℝ := 9/10
def small_apple_ratio : ℝ := 1/10

-- The probability that an apple sorted as large is actually large
def prob_large_given_sorted_large (large_to_small_prob small_to_large_prob large_apple_ratio small_apple_ratio : ℝ) : ℝ :=
  let P_B := large_apple_ratio * (1 - large_to_small_prob) + small_apple_ratio * small_to_large_prob
  let P_A1_B := large_apple_ratio * (1 - large_to_small_prob)
  P_A1_B / P_B

theorem prob_large_apple_is_large :
  prob_large_given_sorted_large large_to_small_prob small_to_large_prob large_apple_ratio small_apple_ratio = 855 / 857 :=
by
  sorry

end prob_large_apple_is_large_l404_404363


namespace even_number_of_segments_l404_404166

-- Define the vertices of the polygonal line as a sequence of lattice points
def lattice_point := ℤ × ℤ
def segment := (lattice_point × lattice_point)

-- Define conditions for the closed polygonal chain
structure ClosedPolygonalChain :=
  (vertices : List lattice_point)
  (segments_equal_length : ∀ (i : ℕ), i < vertices.length - 1 → 
    (vertices[i]⟩, vertices[(i+1) % vertices.length].fst \- vertices[i].fst)^2 +
    (vertices[i]⟩, vertices[(i+1) % vertices.length].snd \- vertices[i].snd)^2 = c^2)
  (closed : vertices.head = vertices.last)

theorem even_number_of_segments
  (P : ClosedPolygonalChain) : (P.vertices.length - 1) % 2 = 0 :=
by 
  sorry

end even_number_of_segments_l404_404166


namespace hexagon_coloring_l404_404923

-- Define the color types
inductive Color
| blue
| red
| yellow
| green

-- Define a structure for the hexagon grid system with conditions.
structure HexTiling where
  colors : List (List Color)   -- A 2D list to represent the grid of hexagon colors
  -- Constraints
  adjacent_diff : ∀ row col, (col < (List.length (List.nth colors row).getD []) - 1) →
                              ((List.nth (List.nth colors row).getD [] col) ≠ 
                              (List.nth (List.nth colors row).getD [] (col + 1))) -- No two adjacent hexagons can have the same color
  consecutive_seq : ∀ row col, (col < (List.length (List.nth colors row).getD []) - 2) →
                                  (List.nth (List.nth colors row).getD [] col, 
                                   List.nth (List.nth colors row).getD [] (col + 1),
                                   List.nth (List.nth colors row).getD [] (col + 2)) 
                                   ≠
                                  (List.nth (List.nth colors row).getD [] (col - 1), 
                                   List.nth (List.nth colors row).getD [] col,
                                   List.nth (List.nth colors row).getD [] (col + 1)) -- No three consecutive hex should repeat same color sequence
  initial_blue : List.nth (List.nth colors 0).getD [] 0 = Color.blue -- Initial hexagon is blue

-- Define the final assertion function
def validColorings : Nat :=
  -- Function to count the number of valid configurations
  4 -- We arrived at this value through the previous problem analysis and steps

-- The theorem statement 
theorem hexagon_coloring : validColorings = 4 := 
by
sorry

end hexagon_coloring_l404_404923


namespace part_one_part_two_l404_404783

noncomputable def f (x k : ℝ) : ℝ := Real.log x + k / x

theorem part_one (e : ℝ) (he : e ≠ 0) :
  (∀ k, (∃ (H : f e k = 2) k = e) →
    (∀ x, 
      (0 < x ∧ x < e) → (f x e > f e e) ∧
      (e < x) → (f x e > f e e))) ∧ 
  ∃ m, m = 2 :=
sorry

theorem part_two :
  (∀ x1 x2 k, (0 < x2 ∧ x2 < x1) → (f x1 k - f x2 k < x1 - x2)) →
  ∃ k, k ≥ 1/4 :=
sorry

end part_one_part_two_l404_404783


namespace consecutive_odd_integers_sum_l404_404524

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 138) :
  x + (x + 2) + (x + 4) = 207 :=
sorry

end consecutive_odd_integers_sum_l404_404524


namespace arrangement_probability_l404_404917

def class_arrangement_probability : Prop :=
  let total_arrangements := nat.factorial 6
  let valid_arrangements := nat.factorial 5 + nat.factorial 4 * nat.factorial 4 * nat.factorial 4
  valid_arrangements / total_arrangements = 7 / 10

theorem arrangement_probability (C M FL H P PE : Type) :
  class_arrangement_probability :=
by
  sorry

end arrangement_probability_l404_404917


namespace exterior_angle_theorem_for_line_ABC_l404_404376

theorem exterior_angle_theorem_for_line_ABC 
  (ABC_is_straight : ∃ A B C : Points, between A B C)
  (angle_ABD : ∠ABD = 148)
  (angle_BCD : ∠BCD = 58)
  : ∠BDC = 90 :=
by
  sorry

end exterior_angle_theorem_for_line_ABC_l404_404376


namespace rolls_combination_proof_l404_404546

-- Define the initial conditions
variables (total_rolls : ℕ) (num_kinds : ℕ) (min_each_kind : ℕ)
-- Assume the given conditions
def conditions (total_rolls = 10) (num_kinds = 4) (min_each_kind = 1) : Prop :=
  total_rolls = 10 ∧ num_kinds = 4 ∧ min_each_kind = 1

-- Define a function that calculates the combination of rolls
def combinations (total_rolls : ℕ) (num_kinds : ℕ) (min_each_kind : ℕ) : ℕ :=
  44 -- As deduced in the solution steps

-- Prove the number of different combinations of rolls that Jack could purchase is 44
theorem rolls_combination_proof (h : conditions 10  4  1) : combinations 10 4 1 = 44 :=
by
  -- carry out the proof here
  sorry

end rolls_combination_proof_l404_404546


namespace hyperbolas_properties_l404_404329

-- Definitions of the hyperbolas
def C₁ (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def C₂ (x y : ℝ) : Prop := (y^2 / 9) - (x^2 / 16) = 1

-- Proof statements
theorem hyperbolas_properties :
  ∃ (e₁ e₂ : ℝ), asymptotes_are_equal (C₁, C₂) ∧
                no_common_points (C₁, C₂) ∧
                focal_lengths_are_equal (C₁, C₂) :=
by sorry

end hyperbolas_properties_l404_404329


namespace shaded_area_eq_27pi_l404_404556

noncomputable def rectangle_width : ℝ := 15
noncomputable def rectangle_height : ℝ := 12
noncomputable def central_circle_radius : ℝ := 3

theorem shaded_area_eq_27pi :
  let quarter_circle_radius := rectangle_height / 2
      total_area_quarter_circles := π * quarter_circle_radius ^ 2
      area_central_circle := π * central_circle_radius ^ 2
      shaded_area := total_area_quarter_circles - area_central_circle
  in shaded_area = 27 * π :=
by
  -- Definitions
  let quarter_circle_radius := rectangle_height / 2
  let total_area_quarter_circles := π * quarter_circle_radius ^ 2
  let area_central_circle := π * central_circle_radius ^ 2
  let shaded_area := total_area_quarter_circles - area_central_circle

  -- Calculation of quarter circles' area
  have quarter_circles_area : total_area_quarter_circles = 4 * (π * (rectangle_height / 2)^2) / 4, by sorry
  have centralized : quarter_circles_area = 4 * (π * (6)^2) / 4, by sorry

  -- Central circle area
  have central_circle_area_value : area_central_circle = π * (3)^2, by sorry

  -- Shaded area calculation
  have shaded_area_value : shaded_area = 36 * π - 9 * π, by sorry
  have result : shaded_area_value = 27 * π, by sorry

  exact result

end shaded_area_eq_27pi_l404_404556


namespace solve_ratio_of_distances_on_ellipse_l404_404623

noncomputable def ellipse_foci_ratio (P : ℝ × ℝ) (F₁ : ℝ × ℝ) (F₂ : ℝ × ℝ) (ratio : ℝ) : Prop :=
  let (x, y) := P
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  let dist := λ A B : ℝ × ℝ, Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)
  ∃ (P : ℝ × ℝ), 
    (x, y) ∈ {P | P.1^2 / 12 + P.2^2 / 3 = 1} ∧
    F₁ = (-3, 0) ∧
    F₂ = (3, 0) ∧
    (x / 2 = 0) ∧
    dist P F₁ = ratio * dist P F₂

theorem solve_ratio_of_distances_on_ellipse :
  ∃ ratio : ℝ, 
    ellipse_foci_ratio (3, Real.sqrt 3 / 2) (-3, 0) (3, 0) 7 := 
by
  sorry

end solve_ratio_of_distances_on_ellipse_l404_404623


namespace quadratic_rewrite_ab_l404_404435

theorem quadratic_rewrite_ab : 
  ∃ (a b c : ℤ), (16*(x:ℝ)^2 - 40*x + 24 = (a*x + b)^2 + c) ∧ (a * b = -20) :=
by {
  sorry
}

end quadratic_rewrite_ab_l404_404435


namespace unique_prime_sum_and_diff_l404_404263

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def is_sum_of_two_primes (p : ℕ) : Prop :=
  ∃ q1 q2 : ℕ, is_prime q1 ∧ is_prime q2 ∧ p = q1 + q2

noncomputable def is_diff_of_two_primes (p : ℕ) : Prop :=
  ∃ q3 q4 : ℕ, is_prime q3 ∧ is_prime q4 ∧ q3 > q4 ∧ p = q3 - q4

theorem unique_prime_sum_and_diff :
  ∀ p : ℕ, is_prime p ∧ is_sum_of_two_primes p ∧ is_diff_of_two_primes p ↔ p = 5 := 
by
  sorry

end unique_prime_sum_and_diff_l404_404263


namespace sin4_mul_tan2_pos_l404_404701

-- Definitions of the conditions
def sin_four_neg : Prop := sin 4 < 0
def tan_two_neg : Prop := tan 2 < 0

-- Conjecture based on the conditions
theorem sin4_mul_tan2_pos (h1 : sin_four_neg) (h2 : tan_two_neg) : sin 4 * tan 2 > 0 := 
by sorry

end sin4_mul_tan2_pos_l404_404701


namespace checkerboard_domino_cover_l404_404163

def can_be_covered_by_dominoes : list (ℕ × ℕ) → list bool
| [(4, 5), (5, 5), (6, 5), (7, 3), (5, 4)] :=
  [false, true, false, true, false]

theorem checkerboard_domino_cover : can_be_covered_by_dominoes [(4, 5), (5, 5), (6, 5), (7, 3), (5, 4)] = [false, true, false, true, false] :=
by
  sorry

end checkerboard_domino_cover_l404_404163


namespace convex_polyhedral_angle_l404_404240

noncomputable def convex_polyhedral_angle_intersection {A : Type} [linear_ordered_field A]
  (n : ℕ) (Π : fin n → set (set A)) (S : set A) 
  (convex_ngon : set (set A)) : Prop :=
∀ (i : fin n), convex (Π i)

theorem convex_polyhedral_angle {A : Type} [linear_ordered_field A]
  (n : ℕ) (Π : fin n → set (set A)) (S : set A) 
  (convex_ngon : set (set A)) 
  (h_convex_ngon : convex convex_ngon) 
  (h_S : S ∉ convex_ngon) : 
  convex_polyhedral_angle_intersection n Π S convex_ngon :=
sorry

end convex_polyhedral_angle_l404_404240


namespace Felipe_time_to_build_house_l404_404126

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end Felipe_time_to_build_house_l404_404126


namespace cleaning_time_l404_404949

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l404_404949


namespace fuel_fraction_proof_l404_404034

noncomputable def fuel_fraction : Prop :=
  ∃ f : ℝ, 
    let total_fuel := 60 in
    let fuel_first := 30 in
    let fuel_second := f * total_fuel in
    let fuel_third := (f / 2) * total_fuel in
    fuel_first + fuel_second + fuel_third = total_fuel ∧ f = 1 / 3

theorem fuel_fraction_proof : fuel_fraction :=
sorry

end fuel_fraction_proof_l404_404034


namespace todd_borrowed_250_l404_404122

def borrowed_amount (remaining_money repayment spend cost_per_snow_cone income_from_sales : ℕ) :=
  remaining_money + repayment + spend + cost_per_snow_cone - income_from_sales = 250

theorem todd_borrowed_250 :
  borrowed_amount 65 110 75 0 150 = 250 :=
by
  sorry

end todd_borrowed_250_l404_404122


namespace triangular_sum_1000_l404_404026

def triangular_number (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def sum_inverse_triangular_numbers_up_to (k : ℕ) : ℚ :=
  ∑ n in Finset.range k, 1 / triangular_number (n + 1)

theorem triangular_sum_1000 :
  sum_inverse_triangular_numbers_up_to 1000 + 1 = 3001 / 1001 :=
by
  sorry

end triangular_sum_1000_l404_404026


namespace balanced_phrases_not_detected_l404_404131

def reduction_rules : Set String :=
  { "(()) -> A", "(A) -> A", "AA -> A" }

def formula_f (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 1
  else (formula_f (n-1)) + (formula_f (n-2)) + ∑ i in finset.range (n-2), (formula_f i) * (formula_f (n-i-1))

def C_n (n : ℕ) : ℕ :=
  (finset.range (n+1)).binom (2*n)

theorem balanced_phrases_not_detected (n : ℕ) (hn : n = 7) :
  C_n n - formula_f n = 392 :=
by
  rw [hn]
  have : C_n 7 = 429 := by
    rw [C_n]
    norm_num
  have : formula_f 7 = 37 := by
    norm_num
  norm_cast
  norm_num
  sorry

end balanced_phrases_not_detected_l404_404131


namespace fraction_of_income_from_tips_l404_404532

variable (S T I : ℝ)

theorem fraction_of_income_from_tips (h1 : T = (5 / 2) * S) (h2 : I = S + T) : 
  T / I = 5 / 7 := by
  sorry

end fraction_of_income_from_tips_l404_404532


namespace walking_time_in_minutes_l404_404869

/-- Walking speed of the man in kmph -/
def walking_speed : ℝ := 4

/-- Running speed of the man in kmph -/
def running_speed : ℝ := 16.5

/-- Time spent running in hours -/
def running_time : ℝ := 40 / 60

/-- Distance covered while running in km -/
def distance_running : ℝ := running_speed * running_time

/-- Time spent walking in hours -/
def walking_time : ℝ := distance_running / walking_speed

theorem walking_time_in_minutes : walking_time * 60 = 165 := by
  sorry

end walking_time_in_minutes_l404_404869


namespace pipe_C_empty_time_l404_404535

noncomputable def time_to_empty (time_fill_A time_fill_B time_fill_ABC : ℕ) (rate_C : ℚ) : ℚ := 1 / rate_C

theorem pipe_C_empty_time :
  ∀ (time_fill_A time_fill_B time_fill_ABC : ℕ) (rate_C : ℚ),
    time_fill_A = 60 → 
    time_fill_B = 120 →
    time_fill_ABC = 60 →
    (1/time_fill_A + 1/time_fill_B - rate_C = 1/time_fill_ABC) →
    (time_to_empty time_fill_A time_fill_B time_fill_ABC rate_C = 60) :=
begin
  intros,
  sorry
end

end pipe_C_empty_time_l404_404535


namespace count_chords_containing_point_l404_404423

theorem count_chords_containing_point (P O : Point) (r : ℝ) (d : ℝ) 
  (h₁ : distance O P = d) (h₂ : r = 20) (h₃ : d = 12) : 
  number_of_chords_with_integer_lengths O P r = 9 :=
sorry

end count_chords_containing_point_l404_404423


namespace seashell_count_l404_404129

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end seashell_count_l404_404129


namespace min_distance_to_line_eq_l404_404666

theorem min_distance_to_line_eq (x y : ℝ) (h : 5 * x + 12 * y = 60) : sqrt (x^2 + y^2) = 60 / 13 :=
sorry

end min_distance_to_line_eq_l404_404666


namespace standard_equation_of_circle_l404_404489

theorem standard_equation_of_circle (M N : ℝ × ℝ)
  (hM : M = (2, 0))
  (hN : N = (0, 4)) :
  ∃ (h : (ℝ × ℝ) × ℝ), h = (⟨1, 2⟩, sqrt 5) ∧ (λ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 5) := 
sorry

end standard_equation_of_circle_l404_404489


namespace problem_statement_l404_404705

noncomputable def triangle_abc_proof (A B C a b c S : ℝ) (h1 : a = 2) (h2 : c = 1) (h3 : sin B * (tan A + tan C) = tan A * tan C) : Prop :=
  (b^2 = a * c) ∧ (S = (sqrt 7) / 4)

theorem problem_statement 
  (A B C a b c S : ℝ)
  (h1 : a = 2) 
  (h2 : c = 1) 
  (h3 : sin B * (tan A + tan C) = tan A * tan C) : 
  triangle_abc_proof A B C a b c S h1 h2 h3 :=
begin
  -- proof goes here
  sorry
end

end problem_statement_l404_404705


namespace pencils_added_by_sara_l404_404855

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l404_404855


namespace lightest_ball_box_is_blue_l404_404496

-- Define the weights and counts of balls
def yellow_ball_weight : ℕ := 50
def yellow_ball_count_per_box : ℕ := 50
def white_ball_weight : ℕ := 45
def white_ball_count_per_box : ℕ := 60
def blue_ball_weight : ℕ := 55
def blue_ball_count_per_box : ℕ := 40

-- Calculate the total weight of balls per type
def yellow_box_weight : ℕ := yellow_ball_weight * yellow_ball_count_per_box
def white_box_weight : ℕ := white_ball_weight * white_ball_count_per_box
def blue_box_weight : ℕ := blue_ball_weight * blue_ball_count_per_box

theorem lightest_ball_box_is_blue :
  (blue_box_weight < yellow_box_weight) ∧ (blue_box_weight < white_box_weight) :=
by
  -- Proof can go here
  sorry

end lightest_ball_box_is_blue_l404_404496


namespace adjusted_yield_approx_l404_404905

-- Define the conditions from the problem statement:
def stockYield : ℝ := 0.10
def taxRate : ℝ := 0.03
def inflationRate : ℝ := 0.02

-- Calculate the after-tax yield:
def afterTaxYield : ℝ := stockYield * (1 - taxRate)

-- Calculate the adjusted yield considering inflation:
def adjustedYield : ℝ := ((1 + afterTaxYield) / (1 + inflationRate)) - 1

-- The target proof:
theorem adjusted_yield_approx :
  adjustedYield ≈ 0.07549 :=
by
  sorry

end adjusted_yield_approx_l404_404905


namespace remainder_2365947_div_8_l404_404136

theorem remainder_2365947_div_8 : (2365947 % 8) = 3 :=
by
  sorry

end remainder_2365947_div_8_l404_404136


namespace julie_earnings_l404_404007

def mowing_rate : ℝ := 4
def weeding_rate : ℝ := 8
def mowing_hours : ℝ := 25
def weeding_hours : ℝ := 3

theorem julie_earnings :
  let september_earnings := (mowing_hours * mowing_rate) + (weeding_hours * weeding_rate)
  in september_earnings + september_earnings = 248 :=
by
  let september_earnings := (mowing_hours * mowing_rate) + (weeding_hours * weeding_rate)
  show september_earnings + september_earnings = 248
  sorry

end julie_earnings_l404_404007


namespace last_two_digits_10_93_10_31_plus_3_eq_08_l404_404470

def last_two_digits_fraction_floor (n m d : ℕ) : ℕ :=
  let x := 10^n
  let y := 10^m + d
  (x / y) % 100

theorem last_two_digits_10_93_10_31_plus_3_eq_08 :
  last_two_digits_fraction_floor 93 31 3 = 08 :=
by
  sorry

end last_two_digits_10_93_10_31_plus_3_eq_08_l404_404470


namespace sequence_periodicity_l404_404841

-- Define the sequence
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 934
  else sum_of_digits (sequence (n - 1)) * 13

theorem sequence_periodicity : sequence 2019 = 130 :=
sorry

end sequence_periodicity_l404_404841


namespace smallest_n_l404_404255

theorem smallest_n(vc: ℕ) (n: ℕ) : 
    (vc = 25) ∧ ∃ y o i : ℕ, ((25 * n = 10 * y) ∨ (25 * n = 18 * o) ∨ (25 * n = 20 * i)) → 
    n = 16 := by
    -- We state that given conditions should imply n = 16.
    sorry

end smallest_n_l404_404255


namespace average_of_numbers_divisible_by_5_l404_404994

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

noncomputable def numbers := { n ∈ (set.Icc 11 31) | divisible_by_5 n }.to_finset

theorem average_of_numbers_divisible_by_5 : 
  (∑ i in numbers, i) / (numbers.card : ℝ) = 22.5 :=
by
  -- List all numbers
  have h_numbers : numbers = {15, 20, 25, 30} by sorry
  rw h_numbers

  -- Sum of numbers
  have h_sum : ∑ i in numbers, i = 90 by sorry
  rw h_sum

  -- Card (number of elements)
  have h_card : numbers.card = 4 by sorry
  rw h_card

  -- Average
  norm_num

end average_of_numbers_divisible_by_5_l404_404994


namespace surrounding_circle_radius_l404_404164

-- Define the conditions as assumptions
axiom central_circle_radius : ℝ
axiom r : ℝ
axiom centers_form_square : True

-- Assert the relationship to be proved
theorem surrounding_circle_radius (h₁ : central_circle_radius = 2)
                                  (h₂ : centers_form_square): r = Real.sqrt 2 + 1 :=
sorry

end surrounding_circle_radius_l404_404164


namespace binomial_coefficient_12_10_l404_404605

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404605


namespace count_ordered_pairs_l404_404652

theorem count_ordered_pairs :
  let P (a b : ℤ) := (1 ≤ a ∧ a ≤ 200) ∧ (b ≥ 0) ∧ 
                     (∃ r s : ℤ, r + s = -a ∧ r * s = b ∧ (r ≥ 0 ∨ s ≥ 0))
  in (Finset.univ.filter (λ (a b : ℤ), P a b)).card = 20301 :=
  by sorry

end count_ordered_pairs_l404_404652


namespace jerome_time_6_hours_l404_404762

theorem jerome_time_6_hours (T: ℝ) (s_J: ℝ) (t_N: ℝ) (s_N: ℝ)
  (h1: s_J = 4) 
  (h2: t_N = 3) 
  (h3: s_N = 8): T = 6 :=
by
  -- Given s_J = 4, t_N = 3, and s_N = 8,
  -- we need to prove that T = 6.
  sorry

end jerome_time_6_hours_l404_404762


namespace range_of_a_l404_404292

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 0 < x₁ → 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) ↔ (1 ≤ a) :=
by
  sorry

end range_of_a_l404_404292


namespace are_perpendicular_l404_404981

variable {R : Type} [LinearOrderedField R]

def line1 (c : R) : AffineSubspace R (R × R) := {
  carrier := { p | ∃ x y : R, p = (x, y) ∧ (2 * x + y + c = 0) },
  direction := ⊤
}

def line2 : AffineSubspace R (R × R) := {
  carrier := { p | ∃ x y : R, p = (x, y) ∧ (x - 2 * y + 1 = 0) },
  direction := ⊤
}

theorem are_perpendicular (c : R) : 
  ∃ m1 m2 : R, 
  (∀ x y : R, (2 * x + y + c = 0) → y = m1 * x + -c) ∧ 
  (∀ x y : R, (x - 2 * y + 1 = 0) → y = m2 * x + 0.5) ∧ 
  m1 * m2 = -1 := sorry

end are_perpendicular_l404_404981


namespace angle_of_inclination_l404_404731

theorem angle_of_inclination (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2 * real.sqrt 3) (h2 : y1 = -1) (h3 : x2 = real.sqrt 3) (h4 : y2 = 2) :
  ∃ θ : ℝ, real.tan θ = (y2 - y1) / (x2 - x1) ∧ θ = 120 :=
by
  sorry

end angle_of_inclination_l404_404731


namespace two_person_subcommittees_l404_404714

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l404_404714


namespace yuebao_scientific_notation_l404_404571

-- Definition of converting a number to scientific notation
def scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10 ^ n

-- The specific problem statement
theorem yuebao_scientific_notation :
  scientific_notation (1853 * 10 ^ 9) 1.853 11 :=
by
  sorry

end yuebao_scientific_notation_l404_404571


namespace jade_pieces_left_l404_404763

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end jade_pieces_left_l404_404763


namespace g_triple_of_10_l404_404978

def g (x : Int) : Int :=
  if x < 4 then x^2 - 9 else x + 7

theorem g_triple_of_10 : g (g (g 10)) = 31 := by
  sorry

end g_triple_of_10_l404_404978


namespace max_demand_decrease_l404_404102

theorem max_demand_decrease 
  (price_increase : ℝ = 1.20)
  (revenue_increase : ℝ = 1.10) :
  let q := revenue_increase / price_increase in
  1 - q = (1 : ℝ) / 12 :=
by
  sorry

end max_demand_decrease_l404_404102


namespace perimeter_WXYZ_approx_23_l404_404749

noncomputable def perimeter_WXYZ 
  (WX : ℝ) (XY : ℝ) (WZ : ℝ) (XZ : ℝ) (YZ : ℝ)
  (h1 : WX = 6 * Real.sqrt 2)
  (h2 : XY = 3 * Real.sqrt 2)
  (h3 : WZ = 6)
  (h4 : XZ = 6)
  (h5 : YZ = 3 * Real.sqrt 2) 
  (isosceles_WXZ : ∀ a b c : ℝ, a^2 + b^2 = c^2 → a = b)
  (isosceles_XYZ : ∀ a b c : ℝ, a^2 + b^2 = c^2 → a = b) 
  : ℝ :=
WX + XY + WZ + YZ

theorem perimeter_WXYZ_approx_23 
  (WX : ℝ) (XY : ℝ) (WZ : ℝ) (XZ : ℝ) (YZ : ℝ) 
  (h1 : WX = 6 * Real.sqrt 2)
  (h2 : XY = 3 * Real.sqrt 2)
  (h3 : WZ = 6)
  (h4 : XZ = 6)
  (h5 : YZ = 3 * Real.sqrt 2) 
  (isosceles_WXZ : ∀ a b c : ℝ, a^2 + b^2 = c^2 → a = b)
  (isosceles_XYZ : ∀ a b c : ℝ, a^2 + b^2 = c^2 → a = b) 
  : |(perimeter_WXYZ WX XY WZ XZ YZ h1 h2 h3 h4 h5 isosceles_WXZ isosceles_XYZ) - 23| < 1 :=
begin
  sorry
end

end perimeter_WXYZ_approx_23_l404_404749


namespace binomial_12_10_eq_66_l404_404617

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404617


namespace bouquet_cost_45_lilies_l404_404572

theorem bouquet_cost_45_lilies : 
  ∀ (price_per_lily : ℝ) (discount : ℝ) (cost_18_lilies : ℝ) (num_lilies_18 : ℝ) (num_lilies_45 : ℝ),
    price_per_lily = cost_18_lilies / num_lilies_18 →
    discount = 0.10 →
    num_lilies_18 = 18 →
    cost_18_lilies = 36 →
    num_lilies_45 = 45 →
    (num_lilies_45 > 30) →
    (num_lilies_45 * price_per_lily) * (1 - discount) = 81 :=
by
  intros price_per_lily discount cost_18_lilies num_lilies_18 num_lilies_45
  assume h1 h2 h3 h4 h5 h6
  have h7 : price_per_lily = 36 / 18, from h1,
  have h8 : price_per_lily = 2, from div_eq_mul_inv.trans (by norm_num),
  have h9 : num_lilies_45 * price_per_lily = 45 * 2, from congrArg (λ p, num_lilies_45 * p) h8,
  have h10 : num_lilies_45 * price_per_lily = 90, from by norm_num,
  have h11 : 90 * (1 - discount) = 90 * (1 - 0.10), from congrArg (λ d, 90 * (1 - d)) h2,
  have h12 : 90 * 0.90 = 81, from mul_eq_mul_right_iff.2 (or.inr (by norm_num)),
  exact h12

end bouquet_cost_45_lilies_l404_404572


namespace product_odd_integers_l404_404135

theorem product_odd_integers :
  (∏ i in finset.range 5000, if i % 2 = 1 then i else 1) = 5000.factorial / (2 ^ 2500 * 2500.factorial) :=
by
  sorry

end product_odd_integers_l404_404135


namespace domain_of_f_l404_404464

noncomputable def f (x : ℝ) := real.sqrt (2 * x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x + 1 ≥ 0} = set.Ici (-1/2) :=
by
  sorry

end domain_of_f_l404_404464


namespace repeating_decimal_0_1333_is_fraction_l404_404683

noncomputable def repeating_decimal_equiv_fraction : Prop :=
  let dec_0333 : ℚ := 1/3
  ∃ frac : ℚ, ∃ h : decimal_expand_frac frac = 0.1333, frac = 2/15

theorem repeating_decimal_0_1333_is_fraction :
  repeating_decimal_equiv_fraction :=
by
  sorry

end repeating_decimal_0_1333_is_fraction_l404_404683


namespace find_some_number_l404_404155

theorem find_some_number (some_number : ℝ) 
  (h : (some_number * 10^2) * (4 * 10^-2) = 12) : 
  some_number = 3 :=
sorry

end find_some_number_l404_404155


namespace two_person_subcommittees_l404_404712

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l404_404712


namespace length_of_segment_CD_l404_404493

theorem length_of_segment_CD (V : ℝ) (r : ℝ) (h : ℝ) (v_region : ℝ):
  v_region = 128 * π → r = 2 → 
  (let v_hemisphere := (2 * ((1 / 2) * ((4 / 3) * π * (2 ^ 3)))) in
  let v_cylinder := v_region - v_hemisphere in
  let h := v_cylinder / (π * 4) in
  h = 29 + 1/3) :=
begin
  intros h1 h2,
  have h3 : (2 * ((1 / 2) * ((4 / 3) * π * (2 ^ 3)))) = 32 * π / 3, by sorry,
  have h4 : 128 * π - 32 * π / 3 = 352 * π / 3, by sorry,
  have h5 : 352 * π / 3 / (π * 4) = 29 + 1/3, by sorry,
  rw h3,
  rw h4,
  rw h5,
  exact h_refl
end

end length_of_segment_CD_l404_404493


namespace convert_decimal_to_base_five_l404_404975

theorem convert_decimal_to_base_five (n : ℕ) (h : n = 256) : nat.base_repr 5 n = "2011" :=
by
  rw h
  -- skipped proof
  sorry

end convert_decimal_to_base_five_l404_404975


namespace ants_approx_73_million_l404_404190

section AntProblem

def feet_to_inches (length_in_feet : ℕ) : ℕ := length_in_feet * 12

def park_width_ft : ℕ := 200
def park_length_ft : ℕ := 500
def patch_side_ft : ℕ := 50

def park_width_in : ℕ := feet_to_inches park_width_ft
def park_length_in : ℕ := feet_to_inches park_length_ft
def patch_side_in : ℕ := feet_to_inches patch_side_ft

def area (width_in inches : ℕ) (length_in_in : ℕ ) : ℕ := width_in * length_in

def area_park_in : ℕ := area park_width_in park_length_in
def area_patch_in : ℕ := area patch_side_in patch_side_in

def density_norm : ℕ := 5 -- ants per square inch
def density_patch : ℕ := 8 -- ants per square inch

def ants_normal : ℕ := density_norm * area_park_in
def ants_patch_old : ℕ := density_norm * area_patch_in
def ants_patch_new : ℕ := density_patch * area_patch_in

def total_ants : ℕ := ants_normal - ants_patch_old + ants_patch_new

theorem ants_approx_73_million : abs (total_ants - 73000000) ≤ 1000000 := by
  sorry

end AntProblem

end ants_approx_73_million_l404_404190


namespace evaluate_expression_l404_404258

theorem evaluate_expression :
  ∀ (op : ℚ → ℚ → ℚ), 
    (∀ x y, op x y ≠ x + y) ∧ 
    (∀ x y, op x y ≠ x - y) ∧ 
    (∀ x y, op x y ≠ x * y) ∧ 
    (∀ x y, op x y ≠ x / y) → 
    ¬∃ op, op 8 4 = 0 :=
by
  sorry

end evaluate_expression_l404_404258


namespace hyperbola_eccentricity_range_l404_404328

variable (a b : ℝ) (f1 f2 : ℝ × ℝ)
variable (P : ℝ × ℝ)

def is_hyperbola (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

def conditions (x y : ℝ) : Prop :=
  x - y = 2 * a ∧ x = 2 * y

def triangle_inequality (x y c : ℝ) : Prop :=
  x + y > 2 * c ∧ x - y < 2 * c

noncomputable def c : ℝ := real.sqrt (a^2 + b^2)

def eccentricity_range (e : ℝ) : Prop :=
  1 < e ∧ e <= 3

theorem hyperbola_eccentricity_range (h1 : a > 0) (h2 : b > 0)
  (h3 : is_hyperbola f1.1 f1.2)
  (h4 : is_hyperbola f2.1 f2.2)
  (h5 : ∃ P, is_on_asymptote P ∧ |dist P f1| = 2 * |dist P f2|) :
  ∃ e, eccentricity a b c e ∧ eccentricity_range e :=
begin
  sorry
end

end hyperbola_eccentricity_range_l404_404328


namespace Annie_ride_distance_l404_404416

theorem Annie_ride_distance :
  ∀ (x : ℕ), 
  (let CostForMike := 2.5 + 0.25 * 36 in 
   let CostForAnnie := 2.5 + 5.0 + 0.25 * x in 
   CostForMike = CostForAnnie) → x = 16 :=
begin
  intros x h,
  sorry
end

end Annie_ride_distance_l404_404416


namespace perimeter_is_18_plus_2sqrt34_l404_404371

-- Definitions according to conditions
axiom is_trapezoid (ABCD : Type) [is_trapezoid ABCD] : Prop
def AB : ℝ := 12
def CD : ℝ := 6
def height : ℝ := 5 

-- Proving that the perimeter is 18 + 2 * sqrt(34)
theorem perimeter_is_18_plus_2sqrt34 (ABCD : Type) [is_trapezoid ABCD] : 
  AB = 2 * CD → 
  height = 5 → 
  CD = 6 → 
  perimeter ABCD = 18 + 2 * Real.sqrt 34 :=
by
  -- Skip the proof
  sorry

end perimeter_is_18_plus_2sqrt34_l404_404371


namespace cos_sin_power_l404_404584

noncomputable theory
open Complex
open Real

/-- Main statement of the problem -/
theorem cos_sin_power :
  ∀ (θ : ℝ), θ = 210 * (π / 180) → (cos θ + Complex.i * sin θ) ^ 30 = -1 :=
by 
  sorry

end cos_sin_power_l404_404584


namespace ellipse_dot_product_correct_l404_404931

noncomputable def ellipse_dot_product : ℝ :=
  let ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1 in
  let focus : (ℝ × ℝ) := (1, 0) in
  let line_eq : ∀ (x y : ℝ), y = x - 1 in
  let intersection_A : ℝ × ℝ := (0, -1) in
  let intersection_B : ℝ × ℝ := (4 / 3, 1 / 3) in
  let dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2 in
  dot_product intersection_A intersection_B

theorem ellipse_dot_product_correct :
  ellipse_dot_product = -1 / 3 := by
sorry

end ellipse_dot_product_correct_l404_404931


namespace trigonometric_identities_l404_404678

theorem trigonometric_identities
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : tan α = 4 / 3)
  (h4 : sin (α - β) = - (√5) / 5) :
  cos (2 * α) = -7 / 25 ∧ tan (α + β) = -41 / 38 := 
sorry

end trigonometric_identities_l404_404678


namespace derrick_yard_length_l404_404630

def alex_yard (derrick_yard : ℝ) := derrick_yard / 2
def brianne_yard (alex_yard : ℝ) := 6 * alex_yard

theorem derrick_yard_length : brianne_yard (alex_yard derrick_yard) = 30 → derrick_yard = 10 :=
by
  intro h
  sorry

end derrick_yard_length_l404_404630


namespace count_intersections_l404_404060

noncomputable def f : ℝ → ℝ := sorry

theorem count_intersections (hf : Function.Injective f) :
  { x : ℝ | f (x^2) = f (x^6) }.to_finset.card = 3 :=
begin
  sorry
end

end count_intersections_l404_404060


namespace largest_common_term_in_range_1_to_200_l404_404217

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l404_404217


namespace cheaper_store_for_22_books_max_books_for_24_yuan_l404_404870

-- Define the conditions
def cost_store_a (n : ℕ) : ℝ :=
  if n <= 10 then n
  else 10 + 0.7 * (n - 10)

def cost_store_b (n : ℕ) : ℝ :=
  0.85 * n

-- Problem 1: Prove that Store A is cheaper for buying 22 exercise books
theorem cheaper_store_for_22_books : cost_store_a 22 < cost_store_b 22 :=
by
  sorry

-- Problem 2: Prove that with 24 yuan, the maximum number of exercise books is 30 at Store A
theorem max_books_for_24_yuan : 
  let books_store_a := 10 + (24 - 10) / 0.7 in
  let books_store_b := 24 / 0.85 in
  floor books_store_a = 30 ∧ floor books_store_b = 28 :=
by
  sorry

end cheaper_store_for_22_books_max_books_for_24_yuan_l404_404870


namespace simplify_expression_1_combine_terms_l404_404153

variable (a b : ℝ)

-- Problem 1: Simplification
theorem simplify_expression_1 : 2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by 
  sorry

-- Problem 2: Combine like terms
theorem combine_terms : 3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 := by 
  sorry

end simplify_expression_1_combine_terms_l404_404153


namespace g_s_difference_l404_404355

def g (n : ℤ) : ℤ := n^3 + 3 * n^2 + 3 * n + 1

theorem g_s_difference (s : ℤ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end g_s_difference_l404_404355


namespace constant_term_binomial_expansion_l404_404079

theorem constant_term_binomial_expansion :
  (∃ (r : ℕ), 4 - 2 * r = 0 ∧ (2^r * Nat.choose 4 r = 24)) :=
by
  use 2 -- This is the value of r we found
  split
  · -- Proving the exponent condition
    norm_num
  · -- Proving the value of the term 
    norm_num
    sorry

end constant_term_binomial_expansion_l404_404079


namespace little_john_money_left_l404_404032

noncomputable def little_john_final_amount (initial : ℚ) : ℚ :=
  let after_toy := initial - 0.15 * initial
  let after_gifts := after_toy - 0.25 * after_toy
  let sweet_cost_with_discount := 12.5 - 0.1 * 12.5
  after_gifts - sweet_cost_with_discount

theorem little_john_money_left {initial : ℚ} (h_initial : initial = 325) :
  little_john_final_amount initial = 195.94 :=
by
  rw [h_initial, little_john_final_amount]
  norm_num
  split_ifs
  sorry

end little_john_money_left_l404_404032


namespace data_median_and_mode_l404_404674

theorem data_median_and_mode :
  let data := [3, 5, 7, 8, 8] in
  List.median data = 7 ∧ List.mode data = 8 :=
by
  sorry

end data_median_and_mode_l404_404674


namespace area_triangle_ABC_l404_404223

noncomputable def area_rect (A D E F : Type) [linear_order A] [ordered_ring D] (area: D) : D := 16
noncomputable def area_tri (A D B : Type) [linear_order A] [ordered_ring D] (area: D) : D := 3
noncomputable def area_tri' (A C F : Type) [linear_order A] [ordered_ring D] (area: D) : D := 4

theorem area_triangle_ABC (A B C D E F : Type) [linear_order (area_rect A D E F)]
  [ordered_ring (area_tri A D B)] [ordered_ring (area_tri' A C F)]
  (h1 : area_rect A D E F 16) (h2 : area_tri A D B 3) (h3 : area_tri' A C F 4) :
  ∃ (ABC_area : ℕ), ABC_area = 65 / 10 :=
by
  sorry

end area_triangle_ABC_l404_404223


namespace lines_coplanar_value_of_k_l404_404421

theorem lines_coplanar_value_of_k:
  ∃ (k : ℝ), 
  (∀ (s t : ℝ), (∃ u : ℝ, 
  (u*(2 + 2*s, -1 - 3*k*s, 4 + 2*k*s) = v*(1 + 3*t, 2 + 2*t, 5 - 2*t))) → 
  (k = -2/3))
:= sorry

end lines_coplanar_value_of_k_l404_404421


namespace no_such_integers_exists_l404_404772

theorem no_such_integers_exists :
  ∀ (P : ℕ → ℕ), (∀ x, P x = x ^ 2000 - x ^ 1000 + 1) →
  ¬(∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
  (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k))) := 
by
  intro P hP notExists
  have contra : ∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k)) := notExists
  sorry

end no_such_integers_exists_l404_404772


namespace sum_even_and_multiples_of_5_l404_404401

def num_even_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 5 -- even digits: {0, 2, 4, 6, 8}
  thousands * hundreds * tens * units

def num_multiples_of_5_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 2 -- multiples of 5 digits: {0, 5}
  thousands * hundreds * tens * units

theorem sum_even_and_multiples_of_5 : num_even_four_digit + num_multiples_of_5_four_digit = 6300 := by
  sorry

end sum_even_and_multiples_of_5_l404_404401


namespace binomial_12_10_l404_404587

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404587


namespace problem_l404_404721

theorem problem (m : ℝ) (h : 4 * x ^ 2 + m * x + 1 = (2 * x - 1) ^ 2) : m = -4 ∧ statement_1_correct ∧ ¬statement_2_correct ∧ ¬statement_3_correct :=
by 
  have h_exp : (2 * x - 1) ^ 2 = 4 * x ^ 2 - 4 * x + 1 := by 
    ring
  have h_eq : 4 * x ^ 2 + m * x + 1 = 4 * x ^ 2 - 4 * x + 1 := by
    rw [h, h_exp]
  have h_m : m = -4 := by 
    linarith
  split
  · exact h_m
  · split
    · sorry
    · split
      · sorry
      · sorry

end problem_l404_404721


namespace transformed_coordinates_l404_404180

variables (ρ θ φ : ℝ) (x y z : ℝ)

/-- Cartesian to spherical coordinate transformation -/
def spherical_to_cartesian (ρ φ θ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

/-- Original point in Cartesian coordinates and corresponding spherical coordinates -/
variables (x₀ y₀ z₀ : ℝ)
axiom coords: x₀ = 3 ∧ y₀ = -4 ∧ z₀ = 12
axiom original_spherical: (x₀, y₀, z₀) = spherical_to_cartesian ρ φ θ

/-- We need to show that the transformed point remains consistent -/
theorem transformed_coordinates :
  spherical_to_cartesian ρ φ (-θ) = (3, 4, 12) :=
sorry

end transformed_coordinates_l404_404180


namespace validate_statements_l404_404298

def circle : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

def line (x : ℝ) : ℝ := x + 1

theorem validate_statements :
  let C := circle
  let l := line in
  -- Condition A: The intercept of line l on the y-axis is 1
  (l 0 = 1) ∧
  -- Condition B: The slope of line l is π / 4
  ((l 1 - l 0) / (1 - 0) = 1) ∧
  -- Condition C: Line l intersects circle C at 2 points
  ((∃ x1 x2 : ℝ, (x1^2 + (line x1)^2 = 1) ∧ (x2^2 + (line x2)^2 = 1) ∧ (x1 ≠ x2)) ∧
  -- Condition D: The maximum distance from a point on circle C to line l is √2
  ¬ (∀ x : ℝ, x^2 + (line x)^2 = 1 → dist (x, line x) l = √2)) :=
by
  sorry

end validate_statements_l404_404298


namespace equation_holds_l404_404347

-- Positive integers less than 10
def is_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

theorem equation_holds (a b c : ℕ) (ha : is_lt_10 a) (hb : is_lt_10 b) (hc : is_lt_10 c) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 :=
by
  sorry

end equation_holds_l404_404347


namespace cleaning_time_l404_404950

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l404_404950


namespace find_winning_votes_l404_404897

-- Define the conditions of the problem
variables (V : ℝ) (votes_winner votes_loser : ℝ)

-- Conditions
def is_two_candidates : Prop := True -- There are two candidates

def winner_percentage : Prop := votes_winner = 0.62 * V -- The winner received 62% of the votes

def loser_percentage : Prop := votes_loser = 0.38 * V -- The loser received 38% of the votes

def win_by_votes : Prop := votes_winner - votes_loser = 384 -- The winner won by 384 votes

-- Problem to solve
theorem find_winning_votes
    (h1 : is_two_candidates)
    (h2 : winner_percentage)
    (h3 : loser_percentage)
    (h4 : win_by_votes) :
    votes_winner = 992 :=
sorry

end find_winning_votes_l404_404897


namespace least_possible_integral_qr_of_triangles_l404_404900

noncomputable def least_integral_qr (PQ PR SR SQ : ℝ) : ℕ :=
  if h : PQ > 0 ∧ PR > 0 ∧ SR > 0 ∧ SQ > 0 ∧ PQ + PR > QR ∧ SQ + SR > QR then
    let lower_bound_pqr := PR - PQ in
    let lower_bound_sqr := SQ - SR in
    nat.ceil (max lower_bound_pqr lower_bound_sqr)
  else 0

theorem least_possible_integral_qr_of_triangles (PQ PR SR SQ : ℝ) 
  (hPQ : PQ = 7.5) (hPR : PR = 14.5) (hSR : SR = 9.5) (hSQ : SQ = 23.5) : 
  least_integral_qr PQ PR SR SQ = 15 := by
  sorry

end least_possible_integral_qr_of_triangles_l404_404900


namespace percentage_daisies_is_62_l404_404173

-- Definitions based on conditions
variables (F : ℕ) -- Total number of flowers in the garden

def yellow_flowers : ℕ := (4 * F) / 10
def yellow_tulips : ℕ := (1 * yellow_flowers) / 5
def yellow_daisies : ℕ := yellow_flowers - yellow_tulips
def red_flowers : ℕ := F - yellow_flowers
def red_daisies : ℕ := (1 * red_flowers) / 2
def total_daisies : ℕ := yellow_daisies + red_daisies

-- Prove that the percentage of daisies in the garden is 62%
theorem percentage_daisies_is_62 : (total_daisies * 100) / F = 62 :=
by
  sorry

end percentage_daisies_is_62_l404_404173


namespace nap_hours_in_70_days_l404_404793

-- Define the variables and conditions
variable (n d a b c e : ℕ)  -- assuming they are natural numbers

-- Define the total nap hours function
noncomputable def total_nap_hours (n d a b c e : ℕ) : ℕ :=
  (a + b) * 10

-- The statement to prove
theorem nap_hours_in_70_days (n d a b c e : ℕ) :
  total_nap_hours n d a b c e = (a + b) * 10 :=
by sorry

end nap_hours_in_70_days_l404_404793


namespace range_of_y_function_l404_404881

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l404_404881


namespace inequality_proof_l404_404702

theorem inequality_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (x * y / Real.sqrt (x * y + y * z) + y * z / Real.sqrt (y * z + z * x) + z * x / Real.sqrt (z * x + x * y)) 
  ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end inequality_proof_l404_404702


namespace no_inscribed_sphere_if_black_faces_area_gt_white_faces_black_faces_area_eq_white_faces_possible_l404_404833

/--
  Given a convex polyhedron with some faces painted black and others white,
  and no two black faces share an edge,
  if the total area of the black faces is greater than the total area of the white faces,
  then a sphere cannot be inscribed in this polyhedron.
-/
theorem no_inscribed_sphere_if_black_faces_area_gt_white_faces
  (P : Polyhedron) 
  (black white: Finset Face)
  (no_shared_edge : ∀ f1 f2 : Face, f1 ∈ black → f2 ∈ black → f1 ≠ f2 → f1.disjoint f2)
  (area_black area_white: ℝ)
  (h_area : area_black > area_white) : 
  ¬ P.inscribe_sphere :=
sorry

/--
  Given a convex polyhedron with some faces painted black and some painted white,
  and no two black faces share an edge,
  it is possible for the area of the black faces to be equal to the area of the white faces.
-/
theorem black_faces_area_eq_white_faces_possible
  (P : Polyhedron) 
  (black white: Finset Face)
  (no_shared_edge : ∀ f1 f2 : Face, f1 ∈ black → f2 ∈ black → f1 ≠ f2 → f1.disjoint f2) :
  ∃ (area : ℝ), area_black = area ∧ area_white = area :=
sorry

end no_inscribed_sphere_if_black_faces_area_gt_white_faces_black_faces_area_eq_white_faces_possible_l404_404833


namespace binomial_12_10_eq_66_l404_404616

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404616


namespace find_denomination_l404_404924

-- Given conditions as definitions:
def total_checks : ℕ := 30
def total_worth : ℕ := 1800
def spendable_checks : ℕ := 18
def remaining_average : ℕ := 75
def remaining_checks : ℕ := total_checks - spendable_checks -- 12

-- Required to find:
def denomination := 50

-- Problem statement to prove denomination == 50
theorem find_denomination :
  let spendable_denom := ( (total_worth - remaining_average * remaining_checks) / spendable_checks ) in
  spendable_denom = denomination :=
by
  sorry

end find_denomination_l404_404924


namespace calculate_division_l404_404960

theorem calculate_division :
  (- (3 / 4) - 5 / 9 + 7 / 12) / (- 1 / 36) = 26 := by
  sorry

end calculate_division_l404_404960


namespace croatia_inequality_l404_404047

theorem croatia_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b) + sqrt ((ab + bc + ca) / (a^2 + b^2 + c^2))) 
    ≥ 5 / 2 := 
sorry

end croatia_inequality_l404_404047


namespace fluorescent_bulbs_switched_on_percentage_l404_404937

theorem fluorescent_bulbs_switched_on_percentage (I F : ℕ) (x : ℝ) (Inc_on F_on total_on Inc_on_ratio : ℝ) 
  (h1 : Inc_on = 0.3 * I) 
  (h2 : total_on = 0.7 * (I + F)) 
  (h3 : Inc_on_ratio = 0.08571428571428571) 
  (h4 : Inc_on_ratio = Inc_on / total_on) 
  (h5 : total_on = Inc_on + F_on) 
  (h6 : F_on = x * F) :
  x = 0.9 :=
sorry

end fluorescent_bulbs_switched_on_percentage_l404_404937


namespace number_of_true_propositions_is_3_l404_404211

-- Definitions for the conditions from the problem
def prop1 := ∀ (A B C D : Type) (quad : Quadrilateral A B C D),
  has_one_pair_parallel_sides quad ∧ has_one_pair_congruent_angles quad → is_parallelogram quad

def prop2 := ∀ (A B C D : Type) (quad : Quadrilateral A B C D),
  has_perpendicular_and_congruent_diagonals quad → is_square quad

def prop3 := ∀ (A B C D : Type) (rect : Quadrilateral A B C D),
  is_rectangle rect → is_rhombus (connect_midpoints rect)

def prop4 := ∀ (A B C D : Type) (rhombus : Quadrilateral A B C D),
  is_rhombus rhombus → (diagonal1_bisects_angles rhombus ∧ diagonal2_bisects_angles rhombus)

-- The main theorem combining the above conditions stating that the number of true propositions is 3.
theorem number_of_true_propositions_is_3 :
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) → true_propositions_count = 3 :=
by
  sorry

end number_of_true_propositions_is_3_l404_404211


namespace packs_of_buns_needed_l404_404203

def friends := 10
def non_meat_eater := 1
def non_bun_eater := 1
def burgers_per_guest := 3
def buns_per_pack := 8

theorem packs_of_buns_needed : (3 * (friends - non_meat_eater) - burgers_per_guest) / buns_per_pack = 3 := 
by
suffices h : 24 / 8 = 3 by exact h
{
  have h_burgers : 3 * (friends - non_meat_eater) = 27 := by norm_num,
  have h_buns_needed : 27 - 3 = 24 := by norm_num,
  assumption
}

end packs_of_buns_needed_l404_404203


namespace black_can_prevent_white_adjacency_l404_404812

-- Defining the grid size
def grid_size : ℕ := 23

-- Defining positions and their move constraints
structure Position := (x : ℕ) (y : ℕ)
def adjacent (p1 p2 : Position) : Prop :=
    (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y + 1 = p2.y)) ∨
    (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x + 1 = p2.x))

-- Initial positions
def bottom_left : Position := ⟨0, 0⟩
def top_right : Position := ⟨grid_size - 1, grid_size - 1⟩
def top_left : Position := ⟨0, grid_size - 1⟩
def bottom_right : Position := ⟨grid_size - 1, 0⟩

-- Moves
inductive Move : Type
| White
| Black

-- State of the game
structure GameState :=
    (white1 : Position) (white2 : Position)
    (black1 : Position) (black2 : Position)
    (next_move : Move)

-- Initial game state
def initial_state : GameState :=
    { white1 := bottom_left, white2 := top_right,
      black1 := top_left, black2 := bottom_right,
      next_move := Move.White }

-- The theorem we want to prove:
theorem black_can_prevent_white_adjacency : 
  ∀ (s : GameState),
  s = initial_state →
  ∃ (f : GameState → GameState), 
    ∀ (current_state : GameState), 
      f current_state ≠ sorry -- This placeholder states that each move ensures that white cannot reach adjacency.

end black_can_prevent_white_adjacency_l404_404812


namespace triangle_area_ratio_l404_404933

variable {α : Type*} [LinearOrderedField α]

theorem triangle_area_ratio
  (A B C D E M N : α)
  (β : α)
  (hβ : 0 < β ∧ β < π)
  (isosceles : abs (A - B) = abs (B - C))
  (angle_B : A + C = 2 * B + β)
  (midline_A_B_eq_C_B : abs (M - B) = abs (B - N))
  (AC_parallel_DE : abs (D - E) = 2 * abs (A - C)) :
  (S_ABC : α) / (S_DBE : α) = 4 * sqrt((1 - cos β) / (3 - cos β)) :=
sorry

end triangle_area_ratio_l404_404933


namespace solve_for_x_l404_404055

theorem solve_for_x (x : ℝ) (h : 1 / 2 + 1 / x^2 = 7 / 8) : x = sqrt (8 / 3) ∨ x = - sqrt (8 / 3) :=
by sorry

end solve_for_x_l404_404055


namespace problem1_part1_problem1_part2_l404_404581

theorem problem1_part1 : (3 - Real.pi)^0 - 2 * Real.cos (Real.pi / 6) + abs (1 - Real.sqrt 3) + (1 / 2)⁻¹ = 2 := by
  sorry

theorem problem1_part2 {x : ℝ} : x^2 - 2 * x - 9 = 0 -> (x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end problem1_part1_problem1_part2_l404_404581


namespace molecular_weight_compound_l404_404134

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def num_H : ℝ := 1
def num_Br : ℝ := 1
def num_O : ℝ := 3

def molecular_weight (num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O : ℝ) : ℝ :=
  (num_H * atomic_weight_H) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)

theorem molecular_weight_compound : molecular_weight num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O = 128.91 :=
by
  sorry

end molecular_weight_compound_l404_404134


namespace total_kids_at_camp_l404_404253

-- Definition of the conditions
def kids_from_lawrence_camp : ℕ := 34044
def kids_from_outside_camp : ℕ := 424944

-- The proof statement
theorem total_kids_at_camp : kids_from_lawrence_camp + kids_from_outside_camp = 459988 := by
  sorry

end total_kids_at_camp_l404_404253


namespace range_m_l404_404293

namespace MathProof

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_m
  (m : ℝ)
  (h : m > 0)
  (a b c : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 2)
  (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_triangle : f a m ^ 2 + f b m ^ 2 = f c m ^ 2 ∨
                f a m ^ 2 + f c m ^ 2 = f b m ^ 2 ∨
                f b m ^ 2 + f c m ^ 2 = f a m ^ 2) :
  0 < m ∧ m < 3 + 4 * Real.sqrt 2 :=
by
  sorry

end MathProof

end range_m_l404_404293


namespace vasya_choice_l404_404515

theorem vasya_choice (x : ℝ) (i : ℕ) (h1 : 1 ≤ i) (h2 : x = -1 ∨ x = 0 ∨ x = 1) :
  let a_i := 1 + x^(i+1) + x^(i+2) in a_i^2 = a_i * a_i :=
by
  sorry

end vasya_choice_l404_404515


namespace range_of_function_l404_404884

def range_exclusion (x : ℝ) : Prop :=
  x ≠ 1

theorem range_of_function :
  set.range (λ x : ℝ, if x = -2 then (0 : ℝ) else x + 3) = {y : ℝ | range_exclusion y} :=
by 
  sorry

end range_of_function_l404_404884


namespace sum_of_digits_961_l404_404915

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.to_list.map (λ c, c.to_digit.get_or_else 0).sum

theorem sum_of_digits_961 :
  ∃ x : ℕ, x >= 100 ∧ x <= 999 ∧ is_palindrome x ∧
           is_palindrome (x + 40) ∧
           x + 40 >= 1000 ∧ x + 40 <= 1039 ∧ sum_of_digits x = 16 :=
by
  sorry

end sum_of_digits_961_l404_404915


namespace limit_eval_l404_404150

noncomputable def limit_function (x : ℝ) := 
  (∛(1 + (Real.log x)^2) - 1) / (1 + Real.cos (Real.pi * x))

theorem limit_eval : 
  filter.tendsto limit_function (nhds 1) (nhds (2 / (3 * Real.pi^2))) :=
  sorry

end limit_eval_l404_404150


namespace complex_expression_value_l404_404412

open Complex Real

noncomputable def z : ℂ := cos (2 * π / 13) + Complex.i * sin (2 * π / 13)

theorem complex_expression_value : 
  (z ^ (-12) + z ^ (-11) + z ^ (-10)) * (z ^ 3 + 1) * (z ^ 6 + 1) = -1 :=
by
  have hz : z ^ 13 = 1 := by
    -- z is a 13-th root of unity
    sorry
  -- Substitute and simplify according to the roots of unity properties
  sorry

end complex_expression_value_l404_404412


namespace mike_profit_l404_404803

theorem mike_profit 
  (num_acres_bought : ℕ) (price_per_acre_buy : ℤ) 
  (fraction_sold : ℚ) (price_per_acre_sell : ℤ) :
  num_acres_bought = 200 →
  price_per_acre_buy = 70 →
  fraction_sold = 1/2 →
  price_per_acre_sell = 200 →
  let cost_of_land := price_per_acre_buy * num_acres_bought,
      num_acres_sold := (fraction_sold * num_acres_bought),
      revenue_from_sale := price_per_acre_sell * num_acres_sold,
      profit := revenue_from_sale - cost_of_land
  in profit = 6000 := by
  intros h1 h2 h3 h4
  let cost_of_land := price_per_acre_buy * num_acres_bought
  let num_acres_sold := (fraction_sold * num_acres_bought)
  let revenue_from_sale := price_per_acre_sell * num_acres_sold
  let profit := revenue_from_sale - cost_of_land
  rw [h1, h2, h3, h4]
  sorry

end mike_profit_l404_404803


namespace xiao_ming_speeds_l404_404529

-- Define the conditions of the problem
def distance_to_school : Real := 2 -- 2 kilometers
def cycling_speed_multiplier : Real := 4 -- Cycling speed is 4 times walking speed
def additional_time_minutes : Real := 20 -- Arrived 20 minutes later than usual

-- Noncomputable inference because we are not providing the actual solving steps here
noncomputable def xiao_ming_walking_speed : Real := 4.5 -- Walking speed in kilometers per hour
noncomputable def xiao_ming_cycling_speed : Real := cycling_speed_multiplier * xiao_ming_walking_speed -- Cycling speed

theorem xiao_ming_speeds :
  let x := xiao_ming_walking_speed
  let distance := distance_to_school
  let time_difference := additional_time_minutes / 60 -- Convert minutes to hours
  x = 4.5 ∧ xiao_ming_cycling_speed = 18 :=
by
  -- Definitions for time calculations
  let walking_time := distance / x
  let cycling_time := distance / (cycling_speed_multiplier * x)
  -- Formulate the given time difference condition
  have time_eq : walking_time - cycling_time = time_difference := sorry
  -- Assert the solution meets the given conditions
  exact sorry

end xiao_ming_speeds_l404_404529


namespace median_of_triangle_ABC_l404_404335

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

def median_length_correct (A B C : ℝ × ℝ × ℝ) (length_median : ℝ) : Prop :=
  distance A (midpoint B C) = length_median

theorem median_of_triangle_ABC :
  let A := (3, 3, 2)
  let B := (4, -3, 7)
  let C := (0, 5, 1)
  median_length_correct A B C 3 :=
by
  sorry

end median_of_triangle_ABC_l404_404335


namespace num_oranges_in_stack_l404_404552

def num_oranges_in_layer (rows cols : ℕ) : ℕ := rows * cols

-- Define the number of oranges in each layer based on the conditions
def layers : List ℕ := [ (7, 10), (6, 9), (5, 8), (4, 7), (3, 6), (2, 5), (1, 4), (1, 1) ]

-- Sum the number of oranges in all layers
def total_oranges : ℕ := (layers.map (λ (rc : ℕ × ℕ), num_oranges_in_layer rc.1 rc.2)).sum

theorem num_oranges_in_stack : total_oranges = 225 := by
  sorry

end num_oranges_in_stack_l404_404552


namespace sum_of_two_squares_l404_404431

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 :=
by sorry

end sum_of_two_squares_l404_404431


namespace rick_ironed_27_pieces_l404_404439

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l404_404439


namespace gcd_sum_l404_404935

theorem gcd_sum (n : ℕ) (hn : n > 0) : 
  let gcd_values := {d | ∃ n : ℕ, n > 0 ∧ d = gcd (3 * n + 2) n}
  in ∑ d in gcd_values, d = 3 :=
by
  sorry

end gcd_sum_l404_404935


namespace number_of_triples_is_real_l404_404972

def cyclic_4 (n : ℕ) : ℂ := match n % 4 with
  | 0 => 1
  | 1 => complex.I
  | 2 => -1
  | 3 => -complex.I
  | _ => 0  -- unreachable

noncomputable def count_ordered_triples : ℕ :=
  34225 + 703125

theorem number_of_triples_is_real :
  (count_ordered_triples) = 737350 :=
by
  -- proof steps will go here
  sorry

end number_of_triples_is_real_l404_404972


namespace value_of_S_plus_T_l404_404750

theorem value_of_S_plus_T :
  let l := 5
  let num_parts := 20
  let part_size := l / num_parts
  let S := 5 * part_size
  let T := l - 5 * part_size
  in S + T = l :=
by
  let l := 5
  let num_parts := 20
  let part_size := l / num_parts
  let S := 5 * part_size
  let T := l - 5 * part_size
  show S + T = l
  calc
    S + T = 5 * (l / num_parts) + (l - 5 * (l / num_parts)) := by rfl
    ...   = 5 * (5 / 20) + (5 - 5 * (5 / 20)) := by rfl
    ...   = 5 * 0.25 + (5 - 5 * 0.25) := by rfl
    ...   = 1.25 + 3.75 := by rfl
    ...   = 5 := by rfl

end value_of_S_plus_T_l404_404750


namespace evaluate_f_5_minus_f_neg_5_l404_404350

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end evaluate_f_5_minus_f_neg_5_l404_404350


namespace binomial_12_10_l404_404589

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404589


namespace domain_sqrt_sin_cos_l404_404087

open Real

theorem domain_sqrt_sin_cos (k : ℤ) :
  {x : ℝ | ∃ k : ℤ, (2 * k * π + π / 4 ≤ x) ∧ (x ≤ 2 * k * π + 5 * π / 4)} = 
  {x : ℝ | sin x - cos x ≥ 0} :=
sorry

end domain_sqrt_sin_cos_l404_404087


namespace perfect_square_trinomial_additions_l404_404961

theorem perfect_square_trinomial_additions : 
  ∃ (monomials : Finset Polynomial), 
    monomials.card = 5 ∧
    ∀ m ∈ monomials, ∃ p, (x^2 + 4 + m) = p^2 :=
sorry

end perfect_square_trinomial_additions_l404_404961


namespace required_decrease_l404_404198

noncomputable def price_after_increases (P : ℝ) : ℝ :=
  let P1 := 1.20 * P
  let P2 := 1.10 * P1
  1.15 * P2

noncomputable def price_after_discount (P : ℝ) : ℝ :=
  0.95 * price_after_increases P

noncomputable def price_after_tax (P : ℝ) : ℝ :=
  1.07 * price_after_discount P

theorem required_decrease (P : ℝ) (D : ℝ) : 
  (1 - D / 100) * price_after_tax P = P ↔ D = 35.1852 :=
by
  sorry

end required_decrease_l404_404198


namespace product_of_roots_l404_404788

theorem product_of_roots (Q : Polynomial ℚ) (hQ : Q.degree = 1) (h_root : Q.eval 6 = 0) :
  (Q.roots : Multiset ℚ).prod = 6 :=
sorry

end product_of_roots_l404_404788


namespace roots_equality_l404_404403

noncomputable def problem_statement (α β γ δ p q : ℝ) : Prop :=
(α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2 * p - 3 * q) ^ 2

theorem roots_equality (α β γ δ p q : ℝ)
  (h₁ : ∀ x, x^2 - 2 * p * x + 3 = 0 → (x = α ∨ x = β))
  (h₂ : ∀ x, x^2 - 3 * q * x + 4 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
sorry

end roots_equality_l404_404403


namespace sum_of_solutions_l404_404654

theorem sum_of_solutions : 
  (∀ x : ℝ, (x - 8)^2 = 16 → (x = 12 ∨ x = 4)) ∧ 
  (12 + 4 = 16) :=
by
  intros x h
  split
  { intro h
    rw ←sub_eq_zero at h
    rw sq_sub_sq at h
    have : ( x - 8) = 4 ∨ (x - 8) = -4 := by 
      rw abs_eq at h
      exact h,
    cases this
    { left
      exact eq_add_of_sub_eq this.symm },
    { right
      exact eq_sub_of_add_eq this.symm.symm } }
  { refl }

end sum_of_solutions_l404_404654


namespace total_students_at_competition_l404_404118
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end total_students_at_competition_l404_404118


namespace problem1_problem2_l404_404660

def operation (a b m n : ℤ) : ℤ := (a^b)^m + (b^a)^n

-- Problem 1
theorem problem1 
    (m : ℤ) (n : ℤ) (a b : ℤ)
    (h1 : a = 2)
    (h2 : b = 1)
    (hm : m = 1)
    (hn : n = 2023) :
    operation a b m n = 3 :=
by
  sorry

-- Problem 2 
theorem problem2 
    (m n : ℤ) 
    (h1 : operation 1 4 m n = 10)
    (h2 : operation 2 2 m n = 15) :
    4 ^ (2*m + n - 1) = 81 :=
by
  sorry

end problem1_problem2_l404_404660


namespace total_students_at_competition_l404_404119
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end total_students_at_competition_l404_404119


namespace regions_divided_by_eight_lines_l404_404984

-- Definitions:
def n : ℕ := 8
def no_parallel (lines : list (ℝ × ℝ × ℝ)) : Prop := 
  ∀ (l1 l2 : ℝ × ℝ × ℝ), l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → l1.1 * l2.2 ≠ l1.2 * l2.1
def no_concurrent (lines : list (ℝ × ℝ × ℝ)) : Prop := 
  ∀ (l1 l2 l3 : ℝ × ℝ × ℝ), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 → l2 ≠ l3 → 
  l1.1 * l2.2 ≠ l2.1 * l1.2 → l1.1 * l3.2 ≠ l3.1 * l1.2 → l2.1 * l3.2 ≠ l3.1 * l2.2

-- Statement to prove:
theorem regions_divided_by_eight_lines (lines : list (ℝ × ℝ × ℝ)) :
  no_parallel lines → no_concurrent lines → length lines = n → 1 + n + (n * (n - 1)) / 2 = 37 :=
by
  intros hnp hnc hl
  sorry

end regions_divided_by_eight_lines_l404_404984


namespace positive_difference_between_terms_l404_404957

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem positive_difference_between_terms :
  let a := 3 in
  let d := 7 in
  let a_1500 := arithmetic_sequence a d 1500 in
  let a_1510 := arithmetic_sequence a d 1510 in
  a_1510 - a_1500 = 70 :=
by
  sorry

end positive_difference_between_terms_l404_404957


namespace find_possible_values_of_a_l404_404449

noncomputable def find_a (x y a : ℝ) : Prop :=
  (x + y = a) ∧ (x^3 + y^3 = a) ∧ (x^5 + y^5 = a)

theorem find_possible_values_of_a (a : ℝ) :
  (∃ x y : ℝ, find_a x y a) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
sorry

end find_possible_values_of_a_l404_404449


namespace sin_A_in_right_triangle_l404_404757

theorem sin_A_in_right_triangle (A B C : Type) [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] (h : right_ang_triangle A B C) 
  (hAC : dist A C = Real.sqrt 34) (hAB : dist A B = 5) : 
  Real.sin (angle B A C) = 3 / Real.sqrt 34 :=
by
  sorry

end sin_A_in_right_triangle_l404_404757


namespace ball_returns_to_bella_after_13_throws_l404_404115

theorem ball_returns_to_bella_after_13_throws:
  ∀ (girls : Fin 13) (n : ℕ), (∃ k, k > 0 ∧ (1 + k * 5) % 13 = 1) → (n = 13) :=
by
  sorry

end ball_returns_to_bella_after_13_throws_l404_404115


namespace integral_semi_circle_l404_404236

theorem integral_semi_circle :
  ∫ x in -1..1, real.sqrt (1 - x^2) = real.pi / 2 :=
begin
  sorry
end

end integral_semi_circle_l404_404236


namespace n_div_18_repeating_decimals_l404_404285

theorem n_div_18_repeating_decimals :
  { n : ℕ | 1 ≤ n ∧ n ≤ 17 ∧ repeating_decimal (n / 18) }.to_finset.card = 16 :=
by sorry

-- Define what it means for n / 18 to be a repeating decimal.
def repeating_decimal (r : ℚ) : Prop :=
  ¬ terminating_decimal r

-- Define what it means for a fraction to be a terminating decimal.
def terminating_decimal (r : ℚ) : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ r = a / b ∧ ∀ p, prime p → p ∣ b → (p = 2 ∨ p = 5)

end n_div_18_repeating_decimals_l404_404285


namespace minimize_avg_cost_and_maximize_profit_l404_404549
-- Import necessary libraries

-- Define the variables and functions based on the given conditions
def daily_cost (x : ℝ) : ℝ :=
  (1 / 2) * x^2 + 40 * x + 3200

def average_cost_per_ton (x : ℝ) : ℝ :=
  daily_cost x / x

def selling_price_per_ton : ℝ := 110

def subsidy_scheme_1 : ℝ := 2300

def subsidy_scheme_2 (x : ℝ) : ℝ := 30 * x

-- Proof problem statement
theorem minimize_avg_cost_and_maximize_profit :
  (∀ x : ℝ, 70 ≤ x ∧ x ≤ 100 → average_cost_per_ton x = (x / 2 + 3200 / x + 40) ∧
            selling_price_per_ton = 110 ∧
            (∃ x_min : ℝ, x_min = 80 ∧ average_cost_per_ton x_min = 120) ∧
            (∀ x : ℝ, average_cost_per_ton x = 120 → 120 > 110) ∧

            (∀ x : ℝ, 70 ≤ x ∧ x ≤ 100 →
              ∃ max_profit_scheme_1 : ℝ, max_profit_scheme_1 = - (1 / 2) * (x - 70)^2 + 1550 ∧
              max_profit_scheme_1 = 1550 ) ∧

            (∀ x : ℝ, 70 ≤ x ∧ x ≤ 100 →
              ∃ max_profit_scheme_2 : ℝ, max_profit_scheme_2 = - (1 / 2) * (x - 100)^2 + 1800 ∧
              max_profit_scheme_2 = 1800 ))
 := 
begin
  sorry
end

end minimize_avg_cost_and_maximize_profit_l404_404549


namespace different_three_digit_numbers_l404_404860

theorem different_three_digit_numbers :
  ∃ (cards : list ℕ), (∀ x ∈ cards, x ∈ [1, 2, 3, 4, 5, 6]) ∧ (length cards = 3) → 
  (number_of_three_digit_numbers cards = 48) :=
sorry

end different_three_digit_numbers_l404_404860


namespace trig_identity_l404_404970

theorem trig_identity 
: sin (10 * Real.pi / 180) * sin (50 * Real.pi / 180) * sin (70 * Real.pi / 180) * sin (80 * Real.pi / 180) 
  = (cos (10 * Real.pi / 180) / 8) :=
by
  -- skipping the actual proof part
  sorry

end trig_identity_l404_404970


namespace bn_geometric_tn_sum_max_p_l404_404107

namespace MathProof
open Real

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Condition: S_n + a_n = - (1/2) * n^2 - (3/2) * n + 1
def S_ (n : ℕ) := - (1 / 2 : ℝ) * (n : ℝ) ^ 2 - (3 / 2 : ℝ) * (n : ℝ) + 1

theorem bn_geometric (n : ℕ) (h1 : S_ n + a n = - (1 / 2 : ℝ) * (n : ℝ) ^ 2 - (3 / 2 : ℝ) * (n : ℝ) + 1) :
  let b : ℕ → ℝ := λ n, a n + n in
  b n = (1 / 2) ^ n := 
sorry

theorem tn_sum (n : ℕ) (a : ℕ → ℝ) (h1 : S_ n + a n = - (1 / 2 : ℝ) * (n : ℝ) ^ 2 - (3 / 2 : ℝ) * (n : ℝ) + 1)
  (b : ℕ → ℝ) (hb : ∀ n, b n = (1 / 2) ^ n) :
  let T : ℕ → ℝ := λ n, ∑ i in Finset.range(n+1), (i:ℝ) * b i in
  T n = 2 - (n+2) / (2^(n : ℝ)) :=
sorry

theorem max_p (a : ℕ → ℝ) (h1 : ∀ n, S_ n + a n = - (1 / 2 : ℝ) * (n : ℝ) ^ 2 - (3 / 2 : ℝ) * (n : ℝ) + 1)
  (c : ℕ → ℝ) (hc : ∀ n, c n = (1 / 2) ^ n - a n)
  (P : ℝ := ∑ i in Finset.range 2013, sqrt (1 + 1 / ((c i) ^ 2) + 1 / ((c (i + 1)) ^ 2))) :
  ⌊P⌋ = 2013 :=
sorry

end MathProof

end bn_geometric_tn_sum_max_p_l404_404107


namespace trig_identity_l404_404971

theorem trig_identity 
: sin (10 * Real.pi / 180) * sin (50 * Real.pi / 180) * sin (70 * Real.pi / 180) * sin (80 * Real.pi / 180) 
  = (cos (10 * Real.pi / 180) / 8) :=
by
  -- skipping the actual proof part
  sorry

end trig_identity_l404_404971


namespace math_problem_l404_404267

noncomputable def least_positive_four_digit_solution :=
  let x := 1704
  have h1 : 5 * x % 20 = 10 := by sorry
  have h2 : (3 * x + 11) % 14 = 6 := by sorry
  have h3 : (-3 * x + 2) % 35 = 34 := by sorry
  ⟨x, h1, h2, h3⟩

theorem math_problem :
  ∃ x : ℕ, 5 * x % 20 = 30 % 20 ∧ (3 * x + 11) % 14 = 20 % 14 ∧ (-3 * x + 2) % 35 = x % 35 ∧ (1000 ≤ x ∧ x < 10000) ∧ x = 1704 :=
  begin
    use 1704,
    split,
    { exact sorry },
    split,
    { exact sorry },
    split,
    { exact sorry },
    split,
    { exact sorry },
    { refl }
  end

end math_problem_l404_404267


namespace incorrect_statement_d_l404_404791

variables (Plane Line : Type)
variables (perpendicular : Line → Line → Prop)
variables (parallel : Line → Line → Prop)
variables (containedIn : Line → Plane → Prop)
variables (perpendicular_plane : Line → Plane → Prop)
variables (parallel_plane : Plane → Plane → Prop)
variables (intersection : Plane → Plane → Option Line)

theorem incorrect_statement_d
  (α β : Plane) (m n : Line) :
  perpendicular m n →
  perpendicular_plane m α →
  perpendicular_plane n α →
  parallel_plane β (Plane := α) →
  ¬ (perpendicular_plane β (Plane := α)) :=
begin
  sorry
end

end incorrect_statement_d_l404_404791


namespace east_high_school_students_l404_404113

theorem east_high_school_students (S : ℝ) 
  (h1 : 0.52 * S * 0.125 = 26) :
  S = 400 :=
by
  -- The proof is omitted for this exercise
  sorry

end east_high_school_students_l404_404113


namespace max_elements_T_l404_404780

-- Define that a set T is a valid subset of the range {1, 2, ..., 2000} with required conditions.
def is_valid_T (T : Finset ℕ) : Prop :=
  T ⊆ Finset.range 2001 ∧ (∀ x y ∈ T, x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 6)

-- The theorem statement to prove.
theorem max_elements_T : ∀ T : Finset ℕ, is_valid_T T → T.card ≤ 835 :=
by
  sorry

end max_elements_T_l404_404780


namespace hyperbola_area_of_triangle_l404_404776

noncomputable def semi_major_axis := 3
noncomputable def semi_minor_axis := 4
noncomputable def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

def focus_1 : (ℝ × ℝ) := (5, 0)
def focus_2 : (ℝ × ℝ) := (-5, 0)

theorem hyperbola_area_of_triangle
  (P : ℝ × ℝ)
  (hP : hyperbola P.1 P.2)
  (angle_condition : ∃ angle : ℝ, angle = 90 ∧ ∠ (focus_1) P (focus_2) = angle) :
  (1 / 2) * (dist focus_1 P) * (dist focus_2 P) = 16 := 
sorry

end hyperbola_area_of_triangle_l404_404776


namespace angle_BAM_eq_angle_CAN_l404_404753

open EuclideanGeometry

variables (O A B C A' X D M N : Point)

-- Definitions based on the problem conditions
def circumcenter_of_triangle := is_circumcenter O A B C
def projection_of_A_onto_BC := is_projection A' A (line B C)
def X_on_ray_AA'_opposite := on_ray_opposite_side A A' X (line B C)
def D_on_angle_bisector := is_on_angle_bisector D A B C
def midpoint_of_DX := is_midpoint M D X
def N_intersection_parallel := is_intersection_parallel N (line_through O parallel AD) (line_through D X)

-- Theorem statement in Lean 4
theorem angle_BAM_eq_angle_CAN
  (h1 : circumcenter_of_triangle O A B C)
  (h2 : projection_of_A_onto_BC A' A (line B C))
  (h3 : X_on_ray_AA'_opposite A A' X (line B C))
  (h4 : D_on_angle_bisector D A B C)
  (h5 : midpoint_of_DX M D X)
  (h6 : N_intersection_parallel N (line_through O parallel (line_through A D)) (line_through D X)) :
  angle B A M = angle C A N :=
by
  sorry

end angle_BAM_eq_angle_CAN_l404_404753


namespace clean_house_time_l404_404943

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l404_404943


namespace calc_expression_l404_404579

variable {R : Type*} [CommSemiring R] (y : R)

theorem calc_expression : (18 * y^3) * (9 * y^2) * (1 / (6 * y)^3) = 3 / 4 * y^2 :=
by sorry

end calc_expression_l404_404579


namespace martian_words_bound_l404_404373

theorem martian_words_bound (n : ℕ) (A O : Type) (word : ℕ → ℕ → Prop) :
  (∀ (i j : ℕ), i ≠ j → ∃ k m l, word i k ∧ word j k ∧ m ≠ l ∧ (m < l) ∧ (j < l)) →
  ∀ k ≤ (2^n) / (n+1) := by
    sorry

end martian_words_bound_l404_404373


namespace max_number_in_cell_x_l404_404262

theorem max_number_in_cell_x (cells : Fin 9 → ℕ) (h_unique : Function.Injective cells) 
  (h_sum : ∀ i, (∑ j in adjacent_cells i, cells j) % cells i = 0) 
  (h_specific : ∃ i j, cells i = 4 ∧ cells j = 5) : 
  ∃ x, 1 ≤ x ∧ x ≤ 9 ∧ (∀ y, 1 ≤ y ∧ y ≤ 9 ∧ (∑ j in adjacent_cells i, cells j) % y = 0 → y ≤ x) ∧ x = 9 := 
sorry

end max_number_in_cell_x_l404_404262


namespace function_uniqueness_l404_404646

-- Define the problem conditions
def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = 0 ↔ x = 0

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x^2 + y * f(x)) + f(y^2 + x * f(y)) = f(x + y)^2

-- State the proof problem
theorem function_uniqueness (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) : ∀ x : ℝ, f(x) = x :=
by
  sorry

end function_uniqueness_l404_404646


namespace sum_of_distinct_prime_factors_l404_404271

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l404_404271


namespace distinct_exponents_l404_404624

def exp1 : ℕ := (3 ^ 3) ^ 3
def exp2 : ℕ := 3 ^ ((3 ^ 3) ^ 3)

theorem distinct_exponents : (Set.size ({3 ^ exp1, 3 ^ exp2} : Set ℕ) = 2) :=
by
  -- Exp1 yields 81 
  -- Exp2 yields 19683
  -- Since 81 and 19683 are distinct, the set {3^81, 3^19683} has 2 elements.
  sorry

end distinct_exponents_l404_404624


namespace inequality_2_pow_l404_404427

theorem inequality_2_pow (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 2^sin x + 2^tan x ≥ 2^(x + 1) :=
sorry

end inequality_2_pow_l404_404427


namespace solve_inequality_l404_404991

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l404_404991


namespace all_inequalities_hold_l404_404330

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x^2 < a^2
def condition2 : Prop := y^2 < b^2
def condition3 : Prop := z^2 < c^2

-- Inequalities to prove
def inequality1 : Prop := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a^2 * b^2 + b^2 * c^2 + c^2 * a^2
def inequality2 : Prop := x^4 + y^4 + z^4 < a^4 + b^4 + c^4
def inequality3 : Prop := x^2 * y^2 * z^2 < a^2 * b^2 * c^2

theorem all_inequalities_hold (h1 : condition1 a x) (h2 : condition2 b y) (h3 : condition3 c z) :
  inequality1 a b c x y z ∧ inequality2 a b c x y z ∧ inequality3 a b c x y z := by
  sorry

end all_inequalities_hold_l404_404330


namespace jimmy_pool_time_approx_l404_404001

-- Definitions of given conditions
def bucket_volume : ℝ := 2  -- gallons
def initial_time : ℝ := 20  -- seconds
def time_increase_rate : ℝ := 1.05  -- rate
def pool_volume : ℝ := 84  -- gallons
def trips : ℝ := pool_volume / bucket_volume  -- number of trips

-- Define the sum of time for the geometric series for n trips.
noncomputable def total_time_seconds (n : ℝ) : ℝ :=
  initial_time * (1 - time_increase_rate^n) / (1 - time_increase_rate)

-- Convert total time in seconds to minutes
noncomputable def total_time_minutes (n : ℝ) : ℝ :=
  total_time_seconds n / 60

-- The proof problem statement
theorem jimmy_pool_time_approx :
  total_time_minutes trips ≈ 40.27 :=
sorry

end jimmy_pool_time_approx_l404_404001


namespace autumn_found_pencils_l404_404575

def initial_pencils : ℕ := 20
def misplaced_pencils : ℕ := 7
def broken_pencils : ℕ := 3
def bought_pencils : ℕ := 2
def final_pencils : ℕ := 16

theorem autumn_found_pencils :
  (initial_pencils - misplaced_pencils - broken_pencils + bought_pencils) + found_pencils = final_pencils →
  found_pencils = 4 :=
by
  intros h
  apply eq_of_sub_eq_zero
  sorry

end autumn_found_pencils_l404_404575


namespace two_person_subcommittees_l404_404715

theorem two_person_subcommittees (n : ℕ) (hn : n = 8) : (finset.univ : finset (fin 8)).powerset.card = 28 :=
by
  rw hn
  -- use the fact that the number of k-combinations (sub-committees) is given by binomial coefficient
  exact (nat.choose_symm 8 2).symm ▸ nat.choose 8 2

end two_person_subcommittees_l404_404715


namespace max_area_of_ellipse_in_rectangle_l404_404251

theorem max_area_of_ellipse_in_rectangle :
  ∀ (a b : ℝ), a = 9 ∧ b = 7 → (π * a * b = 63 * π) :=
begin
  assume a b h,
  cases h with ha hb,
  rw [ha, hb],
  norm_num,
end

end max_area_of_ellipse_in_rectangle_l404_404251


namespace num_tents_needed_l404_404797

def count_people : ℕ :=
  let matts_family := 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2
  let joes_family := 1 + 1 + 3 + 1
  matts_family + joes_family

def house_capacity : ℕ := 6

def tent_capacity : ℕ := 2

theorem num_tents_needed : (count_people - house_capacity) / tent_capacity = 7 := by
  sorry

end num_tents_needed_l404_404797


namespace mike_profit_l404_404805

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l404_404805


namespace circles_intersect_l404_404474

def circle_center_radius (a b c : ℝ) : {center : ℝ × ℝ // r : ℝ} :=
  ⟨(-a / 2, -b / 2), sqrt (a^2 / 4 + b^2 / 4 - c)⟩

theorem circles_intersect :
  let C1 := circle_center_radius 1 4 0
  let C2 := circle_center_radius (-2) 0 1
  sqrt ((C1.center.1 - C2.center.1)^2 + (C1.center.2 - C2.center.2)^2) = 2 :=
by
  sorry

end circles_intersect_l404_404474


namespace total_production_by_june_l404_404914

def initial_production : ℕ := 10

def common_ratio : ℕ := 3

def production_june : ℕ :=
  let a := initial_production
  let r := common_ratio
  a * ((r^6 - 1) / (r - 1))

theorem total_production_by_june : production_june = 3640 :=
by sorry

end total_production_by_june_l404_404914


namespace evaluate_f_5_minus_f_neg_5_l404_404348

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l404_404348


namespace cosine_double_angle_identity_l404_404679

theorem cosine_double_angle_identity (α : ℝ) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end cosine_double_angle_identity_l404_404679


namespace soybeans_converted_to_soy_oil_l404_404902

theorem soybeans_converted_to_soy_oil (total_soybeans : ℕ) (tofu_conversion_rate : ℕ) (soy_oil_conversion_rate : ℕ)
  (tofu_price : ℕ) (soy_oil_price : ℕ) (total_revenue : ℕ) (soy_oil_kg : ℕ) : 
  total_soybeans = 460 →
  tofu_conversion_rate = 3 →
  soy_oil_conversion_rate = 6 →
  tofu_price = 3 →
  soy_oil_price = 15 →
  total_revenue = 1800 →
  9 * (total_soybeans - soy_oil_kg) + 2.5 * soy_oil_kg = total_revenue →
  soy_oil_kg = 360 :=
by {
  intros h_total_soybeans h_tofu_conversion_rate h_soy_oil_conversion_rate h_tofu_price h_soy_oil_price h_total_revenue h_revenue_formula,
  sorry
}

end soybeans_converted_to_soy_oil_l404_404902


namespace total_clothing_ironed_l404_404436

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l404_404436


namespace rick_ironed_27_pieces_l404_404441

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l404_404441


namespace problem1_problem2_problem3_l404_404671

-- Definitions and conditions for Problem 1
def symmetric (l : List ℤ) := l = l.reverse

def is_arithmetic (l : List ℤ) := 
∃ d, ∀ i < l.length - 1, l.nth_le i (by linarith) + d = l.nth_le (i + 1) (by linarith)

def b_seq := [2, 5, 8, 11, 8, 5, 2]
def b_seq_question : Prop := 
symmetric b_seq ∧ is_arithmetic b_seq.take 4 ∧ b_seq.head = 2 ∧ b_seq.nth_le 3 (by linarith) = 11

theorem problem1 : b_seq_question → b_seq = [2, 5, 8, 11, 8, 5, 2] := sorry

-- Definitions and conditions for Problem 2
def c_seq (k : ℕ) (n : ℕ) := 
if n < k then c_seq (k - n + 1) else 50 - 4 * (n - k)

def sum_seq (c : ℕ → ℤ) (n : ℕ) := 
∑ i in range n, c i

def S (k : ℕ) := 
sum_seq (c_seq k) (2 * k - 1)

theorem problem2 : (∀ k ≥ 1, symmetric (List.map (c_seq k) (List.range (2 * k - 1))) → 
∀ k ≥ 3, S k ≤ 626 → S k = 626 → k = 13 := sorry

-- Definitions and conditions for Problem 3
def sequence_m (m : ℕ) : List ℤ :=
let l := List.range m
l.concat1 (List.range' 1 m ++ List.reverse (List.range' 1 (m-1)))

def T (n : ℕ) := 
if n = 2008 then sequence_m n else []

def S2008 (m : ℕ) := sum_list (T (m))

theorem problem3 : ∀ m > 1500, one_possible_sum_of_first_2008_terms (sequence_m m) = 2 ^ 2008 - 1 := sorry

end problem1_problem2_problem3_l404_404671


namespace minimum_i_4_elements_l404_404536

-- Definition of the set Sn
def S (n : ℕ) : Set ℕ :=
  { x | ∃ (lines : Fin n → AffineLine ℝ), 
        (∀ i j k, i ≠ j → i ≠ k → j ≠ k → 
                 ¬(lines i).concurrent_3_lines (lines j) (lines k)) ∧ 
        (lines_intersect_regions lines = x) }

-- Proof of the main statement
theorem minimum_i_4_elements :
  ∃ i : ℕ, S i = S n ∧ |S i| ≥ 4 ∧ ∀ j : ℕ, j < i → |S j| < 4 := 
sorry

end minimum_i_4_elements_l404_404536


namespace grass_field_width_l404_404189

theorem grass_field_width
  (length : ℝ) (width_path : ℝ) (area_path : ℝ) (width : ℝ)
  (h_length : length = 75) (h_width_path : width_path = 3.5) (h_area_path : area_path = 1918) :
  (∃ w : ℝ, (82 * (w + 7) - 75 * w = 1918) ∧ w = 192) :=
by {
  have h_total_length : 82 = length + 2 * width_path, by simp [h_length, h_width_path],
  have h_total_width := λ w, w + 2 * width_path,
  have h_area_total := λ w, 82 * (h_total_width w),
  have h_area_field := λ w, 75 * w,
  use 192,
  split,
  { simp [h_area_total, h_area_field, h_area_path, h_length, h_width_path],
    ring },
  { refl }
}

end grass_field_width_l404_404189


namespace volume_of_prism_l404_404456

-- Basic definitions and setup
variables {a : ℝ} {α : ℝ} (H : ℝ)

def isosceles_base_triangle : Prop :=
  let β := α / 2 in 
  let BD := (a / 2) * tan α in 
  let AB := a / (2 * cos β) in 
  true

def lateral_surface_area (H : ℝ) : ℝ :=
  (a / cos α + a) * H

def base_area : ℝ :=
  (1 / 2) * a * (a / 2) * tan α

def condition : Prop :=
  let S_lat := lateral_surface_area H in
  let S_base := base_area in
  S_lat = 2 * S_base

-- The main statement to prove
theorem volume_of_prism (h_base : isosceles_base_triangle) (h_condition : condition) : 
  ∃ V : ℝ, V = (1 / 8) * a^3 * (tan α) * (tan (α / 2)) :=
by 
  sorry

end volume_of_prism_l404_404456


namespace product_fractions_eq_one_iff_perfect_square_l404_404280

theorem product_fractions_eq_one_iff_perfect_square (n : ℕ) (h : n > 2) :
  (∃ S : finset (ℕ × ℕ), (∀ (a b : ℕ), (a, b) ∈ S → 1 < a ∧ a ≤ n ∧ b = a - 1) 
    ∧ (∏ (p : ℕ × ℕ) in S, (p.1 : ℚ) / p.2) * (∏ (p : ℕ × ℕ) in (finset.univ \ S), (p.2 : ℚ) / p.1) = 1)
  ↔ ∃ k : ℕ, n = k^2 ∧ k > 1 := 
sorry

end product_fractions_eq_one_iff_perfect_square_l404_404280


namespace range_of_y_function_l404_404880

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l404_404880


namespace semi_circle_perimeter_l404_404479

-- Define the problem statement
theorem semi_circle_perimeter (r : ℝ) (π_approx : ℝ) (P : ℝ) :
  r = 6.7 → π_approx = 3.14159 → P = π_approx * r + 2 * r → P ≈ 34.45 :=
by
  intros
  sorry

end semi_circle_perimeter_l404_404479


namespace count_interesting_quadruples_l404_404977

def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d > b + c

def num_interesting_quadruples : ℕ :=
  (finset.range 11).card (λ quad : ℕ × ℕ × ℕ × ℕ, is_interesting_quadruple quad.1 quad.2.1 quad.2.2.1 quad.2.2.2)

theorem count_interesting_quadruples : num_interesting_quadruples = 80 :=
  sorry

end count_interesting_quadruples_l404_404977


namespace problem_statement_l404_404294

theorem problem_statement : 
  (1 > -2) ∧ (even 2) →
  ((1 > -2) ∨ (even 2) = True) ∧
  ((1 > -2) ∧ (even 2) = True) ∧
  (¬ (1 > -2) = False) :=
by
  sorry

end problem_statement_l404_404294


namespace cleaning_time_with_doubled_an_speed_l404_404945

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l404_404945


namespace CQRP_is_parallelogram_l404_404767

-- Define the conditions from the problem
variables {A B C R Q P : Type}
variables [is_triangle A B C]
variables {α β: ℝ}
variables (α_gt_45 : α > 45) (β_gt_45: β > 45)
variables (ABR: isosceles_right_triangle A B R) (R_in_ABC: inside_triangle R A B C)
variables (CBP: isosceles_right_triangle C B P) (outside_ABC_P: outside_triangle P A B C)
variables (ACQ: isosceles_right_triangle A C Q) (outside_ABC_Q: outside_triangle Q A B C)

-- State the problem to be proved
theorem CQRP_is_parallelogram :
  parallelogram C Q R P :=
sorry

end CQRP_is_parallelogram_l404_404767


namespace sequence_values_l404_404195

theorem sequence_values (a : ℕ → ℕ) (H1 : a 1 = 999) (H2 : a 2 < 999) (H3 : a 2006 = 1)
  (H4 : ∀ n ≥ 1, a (n + 2) = int.natAbs (a (n + 1) - a n)) :
  (finset.card {b : ℕ | b < 999 ∧ b % 2 = 1 ∧ (∀ d ∈ nat.factors b, d ≠ 3 ∧ d ≠ 37)} = 324) :=
by
  sorry

end sequence_values_l404_404195


namespace triangle_perimeter_l404_404921

theorem triangle_perimeter (s : ℕ) (h1 : 4 * s = 160) :
  let d := (Real.sqrt (s^2 + s^2)) in
  let triangle_perimeter := s + s + d in
  triangle_perimeter = 80 + 40 * Real.sqrt 2 :=
by
  -- fulfilling the type requirements for variables and sorry for the proof
  sorry

end triangle_perimeter_l404_404921


namespace bob_alice_can_see_each_other_again_l404_404208

noncomputable def time_until_visible : ℚ := 48
noncomputable def sum_of_numerator_and_denominator (q : ℚ) : ℤ :=
  q.num.natAbs + q.denom

theorem bob_alice_can_see_each_other_again :
  ∃ (t : ℚ), t = time_until_visible ∧ sum_of_numerator_and_denominator t = 53 :=
by
  use 48
  have hn := (48 : ℚ).num  -- numerator of 48/1
  have hd := (48 : ℚ).denom  -- denominator of 48/1
  calc
    hn.natAbs + hd = 48 + 1 := by norm_num
    ... = 53 := by norm_num
  exact ⟨rfl, rfl⟩

# The Lean statement checks the existence of the value 48 as the correct time interval and verifies
# the sum of the numerator and denominator of 48 when expressed as a fraction is 53.

end bob_alice_can_see_each_other_again_l404_404208


namespace sum_of_bases_l404_404369

theorem sum_of_bases 
  (R1 R2 : ℕ) 
  (F1_R1 : F1 = 0.454545454545.toRat(R1)) 
  (F2_R1 : F2 = 0.545454545454.toRat(R1)) 
  (F1_R2 : F1 = 0.343434343434.toRat(R2)) 
  (F2_R2 : F2 = 0.434343434343.toRat(R2)) 
  (Cond_1 : F1 = (4*R1 + 5)/(R1^2 - 1)) 
  (Cond_2 : F2 = (5*R1 + 4)/(R1^2 - 1))
  (Cond_3 : F1 = (3*R2 + 4)/(R2^2 - 1)) 
  (Cond_4 : F2 = (4*R2 + 3)/(R2^2 - 1)) :
  R1 + R2 = 16 := 
  sorry

end sum_of_bases_l404_404369


namespace oranges_count_l404_404919

variable {O : ℝ} -- The number of oranges bought by the shopkeeper

-- The conditions given in the problem
def conditions :=
  0.85 * O + 0.97 * 400 = 0.898 * (O + 400)

-- The to-be-proven statement
theorem oranges_count (h : conditions) : O = 600 :=
by
  sorry

end oranges_count_l404_404919


namespace jessica_not_work_days_l404_404387

theorem jessica_not_work_days:
  ∃ (x y z : ℕ), 
    (x + y + z = 30) ∧
    (80 * x - 40 * y + 40 * z = 1600) ∧
    (z = 5) ∧
    (y = 5) :=
by
  sorry

end jessica_not_work_days_l404_404387


namespace carson_seed_l404_404964

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l404_404964


namespace binom_12_10_eq_66_l404_404594

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404594


namespace solution_l404_404751

-- Define the total number of cards
def total_cards : ℕ := 63

-- Function to check if a collection of cards has an even number of each dot
def is_valid_collection (cards : Finset (Finset (Fin 6))) : Prop :=
  ∀ dot : Fin 6, (cards.filter (λ card, dot ∈ card)).card % 2 = 0

-- Define the number of valid collections of five cards
def num_valid_collections : ℕ :=
  (Finset.powerset_len 5 (Finset.range total_cards)).filter is_valid_collection.card

theorem solution :
  num_valid_collections = 109368 := 
by
  sorry -- Proof goes here

end solution_l404_404751


namespace quadrangular_pyramid_volume_l404_404078

theorem quadrangular_pyramid_volume (α h : ℝ) (hα : α < 60) : 
  let V := (8 * h^3 * tan (α / 2)) / (3 * (2 * (tan (α / 2))^2 + sqrt (1 - (tan (α / 2))^2) - 1))
  in True := 
by
  sorry

end quadrangular_pyramid_volume_l404_404078


namespace keystone_arch_larger_angle_l404_404741

-- Definitions based on the problem conditions
variable {α : Type} [LinearOrder α] [AddGroup α] [OrderedSub α]

def isosceles_trapezoid (T : Set α) : Prop :=
∃ a b c : α, T = {x | x = a ∨ x = b ∨ x = c} ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def keystone_arch (K : Set (Set α)) : Prop :=
  K = {T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8} ∧
  (∀ T ∈ K, isosceles_trapezoid T) ∧ 
  (∃ P : Set α, ∀ T_1 T_2 ∈ K, (T_1 ≠ T_2) → 
    (non_parallel_sides_meet_at T_1 P) ∧ (non_parallel_sides_meet_at T_2 P)) ∧
  (∀ E ∈ {T_1, T_8}, bottom_sides_horizontal E)

def measures_larger_interior_angle (T : Set α) (x : α) : Prop :=
  ∃ a b c d : α, T = {p | p = a ∨ p = b ∨ p = c ∨ p = d} ∧
  angle a c = x

-- Proof problem
theorem keystone_arch_larger_angle:
  ∀ (K : Set (Set α)) (T : Set α) (x : α), keystone_arch K → T ∈ K → measures_larger_interior_angle T x → x = 101.25 :=
by
sory

end keystone_arch_larger_angle_l404_404741


namespace tennis_matches_l404_404743

theorem tennis_matches (members matches : ℕ) (participants_per_match : ℕ) :
  members = 20 → matches = 14 →
  (∃ (G : Type) [graph G], ∀ (v : G), degree v ≥ 1) →
  ∃ (six_matches : finset (fin 14)), six_matches.card = 6 ∧
  ∀ (m ∈ six_matches), ∃ (participants : finset (fin 20)), participants.card = 2 ∧
  ∀ m₁ m₂ ∈ six_matches, m₁ ≠ m₂ → (participants m₁) ∩ (participants m₂) = ∅ :=
by
  sorry

end tennis_matches_l404_404743


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l404_404063

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 
  (x y : ℤ) (hx : ∃ m : ℤ, x = 6 * m) (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := 
by 
  sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l404_404063


namespace earthquake_amplitude_ratio_l404_404458

theorem earthquake_amplitude_ratio (A A0 : ℝ) (hA0 : A0 > 0) :
  let M8 := 8
      M5 := 5
      A1 := A0 * 10^M8
      A2 := A0 * 10^M5 in
  A1 / A2 = 1000 :=
by sorry

end earthquake_amplitude_ratio_l404_404458


namespace no_integer_n_satisfies_conditions_l404_404659

theorem no_integer_n_satisfies_conditions :
  ¬ ∃ n : ℕ, 0 < n ∧ 1000 ≤ n / 5 ∧ n / 5 ≤ 9999 ∧ 1000 ≤ 5 * n ∧ 5 * n ≤ 9999 :=
by
  sorry

end no_integer_n_satisfies_conditions_l404_404659


namespace triangle_ratio_inequality_l404_404302

/-- Given a triangle ABC, R is the radius of the circumscribed circle, 
    r is the radius of the inscribed circle, a is the length of the longest side,
    and h is the length of the shortest altitude. Prove that R / r > a / h. -/
theorem triangle_ratio_inequality
  (ABC : Triangle) (R r a h : ℝ)
  (hR : 2 * R ≥ a)
  (hr : 2 * r < h) :
  (R / r) > (a / h) :=
by
  -- sorry is used to skip the proof
  sorry

end triangle_ratio_inequality_l404_404302


namespace loss_calculation_l404_404480

-- Given conditions: 
-- The ratio of the amount of money Cara, Janet, and Jerry have is 4:5:6
-- The total amount of money they have is $75

theorem loss_calculation :
  let cara_ratio := 4
  let janet_ratio := 5
  let jerry_ratio := 6
  let total_ratio := cara_ratio + janet_ratio + jerry_ratio
  let total_money := 75
  let part_value := total_money / total_ratio
  let cara_money := cara_ratio * part_value
  let janet_money := janet_ratio * part_value
  let combined_money := cara_money + janet_money
  let selling_price := 0.80 * combined_money
  combined_money - selling_price = 9 :=
by
  sorry

end loss_calculation_l404_404480


namespace cosine_between_AB_AC_l404_404649

open Real Euclidean

-- Define the points
def A : EuclideanSpace ℝ (Fin 3) := ![-3, -7, -5]
def B : EuclideanSpace ℝ (Fin 3) := ![0, -1, -2]
def C : EuclideanSpace ℝ (Fin 3) := ![2, 3, 0]

-- Define vectors AB and AC
def AB : EuclideanSpace ℝ (Fin 3) := B - A
def AC : EuclideanSpace ℝ (Fin 3) := C - A

-- State the theorem
theorem cosine_between_AB_AC : cosAngle AB AC = 1 :=
sorry

end cosine_between_AB_AC_l404_404649


namespace units_digit_quotient_units_digit_quotient_4_1993_5_1993_units_digit_quotient_div_9_l404_404885

theorem units_digit_quotient (n : ℕ) (hn1: n % 3 = 1) (hn2: n % 2 = 1) :
  (4^n + 5^n) % 9 = 0 := by
  sorry

theorem units_digit_quotient_4_1993_5_1993 :
  (4^1993 + 5^1993) % 9 = 0 :=
units_digit_quotient 1993 (by norm_num) (by norm_num)

theorem units_digit_quotient_div_9 : (4^1993 + 5^1993) / 9 % 10 = 1 := by
  have h := units_digit_quotient_4_1993_5_1993
  rw [Nat.add_div, Nat.mul_div_cancel_left] at h
  exact h
  { exact (by dec_trivial) }
  { exact (by dec_trivial) }
  sorry

end units_digit_quotient_units_digit_quotient_4_1993_5_1993_units_digit_quotient_div_9_l404_404885


namespace find_general_formula_sum_of_b_l404_404299

noncomputable def a_n (n : ℕ) : ℕ := 2 ^ (n - 1)

def S_n (n : ℕ) : ℕ := (range n).sum (λ x, a_n (x + 1))

axiom a1_condition : a_n 1 = 1
axiom Sn_condition : S_n 6 = 9 * S_n 3

def b_n (n : ℕ) : ℕ := 1 + nat.log 2 (a_n n)

def sum_b_n (n : ℕ) : ℕ := (range n).sum (λ x, b_n (x + 1))

theorem find_general_formula:
  ∀ n, a_n n = 2^(n-1) :=
sorry

theorem sum_of_b:
  ∀ n, sum_b_n n = n * (n + 1) / 2 :=
sorry

end find_general_formula_sum_of_b_l404_404299


namespace triangle_lattice_points_l404_404932

theorem triangle_lattice_points :
  ∀ (A B C : ℕ) (AB AC BC : ℕ), 
    AB = 2016 → AC = 1533 → BC = 1533 → 
    ∃ lattice_points: ℕ, lattice_points = 1165322 := 
by
  sorry

end triangle_lattice_points_l404_404932


namespace minimum_deg_rotation_for_regular_pentagon_l404_404558

def regular_pentagon_min_rotation (sides : ℕ) (full_rotation : ℝ) : ℝ :=
  full_rotation / sides

theorem minimum_deg_rotation_for_regular_pentagon :
  regular_pentagon_min_rotation 5 360 = 72 :=
by 
  unfold regular_pentagon_min_rotation
  norm_num

end minimum_deg_rotation_for_regular_pentagon_l404_404558


namespace triangle_perimeter_l404_404688

theorem triangle_perimeter (a b c : ℝ) (h₁ : a + 2 = b) (h₂ : b + 2 = c) 
(h₃ : sin (angle a b c) = sqrt 3 / 2) : a + b + c = 15 := 
by {
  -- Definitions and assumptions about side lengths forming an AP
  -- Additional hypothesis and proof steps should be filled here
  sorry -- Proof proceeds here
}

end triangle_perimeter_l404_404688


namespace two_person_subcommittees_from_eight_l404_404709

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l404_404709


namespace a3_equals_3_div_2_l404_404301

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1 else 1 + 1 / (sequence (n - 1))

theorem a3_equals_3_div_2 : sequence 3 = 3 / 2 :=
by
  sorry

end a3_equals_3_div_2_l404_404301


namespace condition_A_neither_sufficient_nor_necessary_condition_B_l404_404237

variable (a θ : ℝ)

-- Define the conditions as predicates
def condition_A : Prop := sqrt (1 + sin θ) = a
def condition_B : Prop := sin (θ / 2) + cos (θ / 2) = a

-- Formalize the goal of the problem
theorem condition_A_neither_sufficient_nor_necessary_condition_B :
  ¬((condition_A a θ ∧ condition_B a θ) ∨ (condition_A a θ → condition_B a θ) ∨ (condition_B a θ → condition_A a θ)) :=
sorry

end condition_A_neither_sufficient_nor_necessary_condition_B_l404_404237


namespace John_ASMC_score_l404_404003

def ASMC_score (c w : ℕ) : ℕ := 25 + 5 * c - 2 * w

theorem John_ASMC_score (c w : ℕ) (h1 : ASMC_score c w = 100) (h2 : c + w ≤ 25) :
  c = 19 ∧ w = 10 :=
by {
  sorry
}

end John_ASMC_score_l404_404003


namespace raft_trip_days_l404_404561


theorem raft_trip_days (x : ℝ) (H1: (1/5 - 1/x = 1/7 + 1/x)): x = 35 :=
sorry

end raft_trip_days_l404_404561


namespace tangent_line_l404_404088

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 3 * x + 1

def p : ℝ × ℝ := (0, 1)

theorem tangent_line :
  ∃ (a b c : ℝ), (λ y x, a * x + b * y + c = 0) = (λ y x, 2 * x + y - 1 = 0) →
  tangent_line_to_curve_at_point f p a b c :=
sorry

end tangent_line_l404_404088


namespace isosceles_triangle_length_l404_404424

noncomputable def triangle_side_length (P Q R S : ℝ) (s : ℝ) 
  (AB AC : ℝ) (PQ PR PS : ℝ) : Prop :=
  PQ = 2 ∧ PR = 3 ∧ PS = 2 ∧ AB = AC ∧ AB = s ∧ 
  let h := Real.sqrt (s^2 - 4) in
  2 * s + 3 * AB / 2 = 1 / 2 * AB * h ->
  AB = Real.sqrt 53

theorem isosceles_triangle_length (P Q R S : ℝ) (PQ PR PS : ℝ) :
  triangle_side_length P Q R S (Real.sqrt 53) (Real.sqrt 53) (Real.sqrt 53) PQ PR PS :=
sorry

end isosceles_triangle_length_l404_404424


namespace highest_y_coordinate_l404_404980

theorem highest_y_coordinate : 
  (∀ x y : ℝ, ((x - 4)^2 / 25 + y^2 / 49 = 0) → y = 0) := 
by
  sorry

end highest_y_coordinate_l404_404980


namespace fraction_sent_afternoon_l404_404891

-- Defining the problem conditions
def total_fliers : ℕ := 1000
def fliers_sent_morning : ℕ := total_fliers * 1/5
def fliers_left_afternoon : ℕ := total_fliers - fliers_sent_morning
def fliers_left_next_day : ℕ := 600
def fliers_sent_afternoon : ℕ := fliers_left_afternoon - fliers_left_next_day

-- Proving the fraction of fliers sent in the afternoon
theorem fraction_sent_afternoon : (fliers_sent_afternoon : ℚ) / fliers_left_afternoon = 1/4 :=
by
  -- proof goes here
  sorry

end fraction_sent_afternoon_l404_404891


namespace proof_example_l404_404097

def my_op (a b : ℝ) : ℝ := a + (3 * a) / (2 * b)

theorem proof_example : my_op (my_op 5 3) 4 = 10.3125 := 
by 
  sorry

end proof_example_l404_404097


namespace cleaning_time_with_doubled_an_speed_l404_404947

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l404_404947


namespace part_I_part_II_part_III_l404_404327

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + a * x - 6 * log x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x + 4

theorem part_I (a : ℝ) :
  if a < 0 then 
    ∀ x, (0 < x ∧ x < -a) → f x a is decreasing ∧ ∀ x, (-a < x) → f x a is increasing
  else 
    ∀ x, (0 < x) → f x a is increasing :=
sorry

theorem part_II (a : ℝ) (h2 : ∀ x, 0 < x → g x a is increasing) :
  a ≥ 5 / 2 :=
sorry

theorem part_III (m : ℝ) (h1 : ∃ x1, (0 < x1 ∧ x1 < 1) ∧ (∀ x2, 1 ≤ x2 ∧ x2 ≤ 2 → g x1 2 ≥ h x2 m)) :
  m ≥ 8 - 5 * log 2 :=
sorry

end part_I_part_II_part_III_l404_404327


namespace max_f_alpha_side_a_l404_404338

noncomputable def a_vec (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)
noncomputable def b_vec (α : ℝ) : ℝ × ℝ := (6 * Real.sin α + Real.cos α, 7 * Real.sin α - 2 * Real.cos α)

noncomputable def f (α : ℝ) : ℝ := (a_vec α).1 * (b_vec α).1 + (a_vec α).2 * (b_vec α).2

theorem max_f_alpha : ∀ α : ℝ, f α ≤ 4 * Real.sqrt 2 + 2 :=
by
sorry

theorem side_a (A : ℝ) (b c : ℝ) (h1 : f A = 6) (h2 : 1/2 * b * c * Real.sin A = 3) (h3 : b + c = 2 + 3 * Real.sqrt 2) : 
  ∃ a : ℝ, a = Real.sqrt 10 :=
by
sorry

end max_f_alpha_side_a_l404_404338


namespace two_AP_eq_BP_l404_404362

variables {A B C D P : Type*}
variables [has_coe A] [has_coe B] [has_coe C] [has_coe D] [has_coe P]

-- Defining angles
def angle (x y z : Type*) := sorry

-- Given conditions
axiom angle_DAC_90 : angle D A C = 90
axiom angle_2ADB_eq_ACB : 2 * angle A D B = angle A C B
axiom angle_DBC_plus_2ADC_180 : angle D B C + 2 * angle A D C = 180

-- Prove that 2 * AP = BP
theorem two_AP_eq_BP : 2 * distance A P = distance B P :=
by sorry

end two_AP_eq_BP_l404_404362


namespace reciprocal_neg_sqrt_2_l404_404104

theorem reciprocal_neg_sqrt_2 : 1 / (-Real.sqrt 2) = -Real.sqrt 2 / 2 :=
by
  sorry

end reciprocal_neg_sqrt_2_l404_404104


namespace skew_lines_definition_l404_404132

noncomputable def are_skew_lines (L1 L2 : Line) : Prop :=
  ¬ (∃ (P : Point), L1.contains P ∧ L2.contains P) ∧ ¬ (∃ (P1 P2 : Point), 
  L1.contains P1 ∧ L1.contains P2 ∧ ∃ (P3 P4 : Point), 
  L2.contains P3 ∧ L2.contains P4 ∧ co_planar P1 P2 P3 P4)

-- The theorem to prove: two skew lines do not intersect and are not in the same plane
theorem skew_lines_definition (L1 L2 : Line) : 
  are_skew_lines L1 L2 ↔ (¬ (∃ (P : Point), L1.contains P ∧ L2.contains P) ∧ ¬ (co_planar_lines L1 L2)) :=
sorry

end skew_lines_definition_l404_404132


namespace julie_total_earnings_l404_404004

def hourly_rate_mowing : ℝ := 4
def hourly_rate_weeding : ℝ := 8
def hours_mowing_in_september : ℝ := 25
def hours_weeding_in_september : ℝ := 3

def earnings_september : ℝ := (hourly_rate_mowing * hours_mowing_in_september) +
                              (hourly_rate_weeding * hours_weeding_in_september)

def earnings_both_months : ℝ := 2 * earnings_september

theorem julie_total_earnings : earnings_both_months = 248 :=
by
  calc
    earnings_both_months = 2 * earnings_september : by rfl
    ... = 2 * ((hourly_rate_mowing * hours_mowing_in_september)
              + (hourly_rate_weeding * hours_weeding_in_september)) : by rfl
    ... = 2 * ((4 * 25) + (8 * 3)) : by rfl
    ... = 2 * (100 + 24) : by rfl
    ... = 2 * 124 : by rfl
    ... = 248 : by rfl

end julie_total_earnings_l404_404004


namespace two_person_subcommittees_from_eight_l404_404708

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l404_404708


namespace evaluate_expression_l404_404260

theorem evaluate_expression :
  (3025^2 : ℝ) / ((305^2 : ℝ) - (295^2 : ℝ)) = 1525.10417 :=
by
  sorry

end evaluate_expression_l404_404260


namespace smallest_terminating_with_digit_five_l404_404137

theorem smallest_terminating_with_digit_five :
  ∃ n : ℕ, n > 0 ∧ (∃ a b : ℕ, n = 2^a * 5^b) ∧ (∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) ∧ n = 5 :=
  sorry

end smallest_terminating_with_digit_five_l404_404137


namespace train_length_is_250_l404_404564

-- Define the length of the train
def train_length (L : ℝ) (V : ℝ) :=
  -- Condition 1
  (V = L / 10) → 
  -- Condition 2
  (V = (L + 1250) / 60) → 
  -- Question
  L = 250

-- Here's the statement that we expect to prove
theorem train_length_is_250 (L V : ℝ) : train_length L V :=
by {
  -- sorry is a placeholder to indicate the theorem proof is omitted
  sorry
}

end train_length_is_250_l404_404564


namespace chef_sold_12_meals_l404_404364

theorem chef_sold_12_meals
  (initial_meals_lunch : ℕ)
  (additional_meals_dinner : ℕ)
  (meals_left_after_lunch : ℕ)
  (meals_for_dinner : ℕ)
  (H1 : initial_meals_lunch = 17)
  (H2 : additional_meals_dinner = 5)
  (H3 : meals_for_dinner = 10) :
  ∃ (meals_sold_lunch : ℕ), meals_sold_lunch = 12 := by
  sorry

end chef_sold_12_meals_l404_404364


namespace area_square_EFGH_equiv_144_l404_404221

theorem area_square_EFGH_equiv_144 (a b : ℝ) (h : a = 6) (hb : b = 6)
  (side_length_EFGH : ℝ) (hs : side_length_EFGH = a + 3 + 3) : side_length_EFGH ^ 2 = 144 :=
by
  -- Given conditions
  sorry

end area_square_EFGH_equiv_144_l404_404221


namespace intervals_of_monotonicity_b_range_l404_404241

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x + (1 - a) / x - 1

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := x^2 - 2 * b * x - 5 / 9

theorem intervals_of_monotonicity (a : ℝ) (ha : 0 < a ∧ a < 1) :
    (0 < a ∧ a < 1/2 → (∀ x, f x a < 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1 ∧ x > 1/a - 1)))
  ∧ (a = 1/2 → ∀ x > 0, f x a < 0)
  ∧ (1/2 < a ∧ a < 1 → (∀ x, f x a < 0 ↔ (0 < x ∧ x < 1/a - 1) ∨ (x > 1))) :=
  sorry

theorem b_range (b : ℝ) :
  (for (x1 : ℝ), (1 ≤ x1 ∧ x1 ≤ 2) →
    (∃ x2, (0 ≤ x2 ∧ x2 ≤ 1) ∧ f x1 (1/3) ≥ g x2 b)) ↔ b ≥ 1/3 :=
  sorry

end intervals_of_monotonicity_b_range_l404_404241


namespace trigonometric_expression_l404_404959

theorem trigonometric_expression :
  arcsin (real.sqrt 2 / 2) + arctan (-1) + arccos (-real.sqrt 3 / 2) = 5 * real.pi / 6 :=
  sorry

end trigonometric_expression_l404_404959


namespace halfway_between_one_eighth_and_one_tenth_l404_404471

theorem halfway_between_one_eighth_and_one_tenth :
  (1 / 8 + 1 / 10) / 2 = 9 / 80 :=
by
  sorry

end halfway_between_one_eighth_and_one_tenth_l404_404471


namespace isosceles_triangle_circumcircle_area_l404_404568

noncomputable def isosceles_triangle_area (DE DF radius : ℝ) : ℝ :=
  let theta := real.arccos ((DE ^ 2 + DF ^ 2 - (DF + radius) ^ 2) / (2 * DE * DF))
  let EF := 2 * radius
  let R := EF / (2 * (real.sin theta * real.cos theta))
  let area := real.pi * R ^ 2
  area

theorem isosceles_triangle_circumcircle_area :
  isosceles_triangle_area (5 * real.sqrt 2) (5 * real.sqrt 2) (3 * real.sqrt 2) = 10 * real.pi :=
by sorry

end isosceles_triangle_circumcircle_area_l404_404568


namespace general_term_a_sum_first_10_terms_b_l404_404747

-- Definitions for the sequence {a_n}
def a (n : ℕ) : ℕ := n + 2

-- Definition for the sequence {b_n}
def b (n : ℕ) : ℕ := 2^(a n - 2) + n

-- Statement 1: Prove the general term formula for {a_n}
theorem general_term_a :
  (forall n, a n = n + 2) :=
sorry

-- Statement 2: Prove the sum of the first 10 terms of the sequence {b_n}
theorem sum_first_10_terms_b :
  (finset.range 10).sum (λ n, b (n + 1)) = 2101 :=
sorry

end general_term_a_sum_first_10_terms_b_l404_404747


namespace tan_70_sin_80_eq_neg1_l404_404447

theorem tan_70_sin_80_eq_neg1 :
  (Real.tan 70 * Real.sin 80 * (Real.sqrt 3 * Real.tan 20 - 1) = -1) :=
sorry

end tan_70_sin_80_eq_neg1_l404_404447


namespace number_of_integer_coordinate_points_l404_404197

open Point

structure Point (R : Type _) :=
(x : R)
(y : R)

def C : Point Int := {x := -5, y := 3}
def D : Point Int := {x := 4, y := -3}

def manhattan_dist (p1 p2 : Point Int) : Int :=
(abs (p2.x - p1.x)) + (abs (p2.y - p1.y))

def valid_path_length (p1 p2 : Point Int) (length : Int) : Prop :=
manhattan_dist p1 p2 <= length

theorem number_of_integer_coordinate_points (length : Int) : 
  valid_path_length C D length → 
  ∃ n : ℕ, n = 248 :=
by
  sorry

end number_of_integer_coordinate_points_l404_404197


namespace binom_12_10_eq_66_l404_404596

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404596


namespace units_digit_k_sq_plus_2_pow_k_l404_404406

-- Define the value of k
def k : ℤ := 2012^2 + 2^(2012)

-- The goal is to prove the units digit of k^2 + 2^k is 7
theorem units_digit_k_sq_plus_2_pow_k :
  (k^2 + 2^k) % 10 = 7 :=
by
  -- Use sorry to assume the proof
  sorry

end units_digit_k_sq_plus_2_pow_k_l404_404406


namespace dice_sum_configuration_l404_404491

theorem dice_sum_configuration (dices : ℕ) (sum_points : ℕ) (unit_cost : ℕ) : 
  (sum_points = 2023) → (unit_cost = 2022) → (∀ die, sum (opposite_faces die) = 7) → 
  (∃ configuration, sum_configuration configuration dices sum_points = unit_cost) := 
by
  intros
  sorry

end dice_sum_configuration_l404_404491


namespace polygon_with_13_sides_shares_vertex_only_with_14_sided_polygon_l404_404191

-- Define the condition of the problem.
def is_regular_polygon (sides : ℕ) : Prop := sides ≥ 3

def dodecagon := 12

-- The set of enclosing polygons with sides from 3 to 14.
def enclosing_polygons := {n : ℕ | n ≥ 3 ∧ n ≤ 14}

-- Define the angles
def interior_angle (sides : ℕ) := (sides - 2) * 180 / sides
def exterior_angle (sides : ℕ) := 180 - interior_angle sides

-- Statement of the proof problem
theorem polygon_with_13_sides_shares_vertex_only_with_14_sided_polygon :
  ∃ sides, sides = 13 ∧ ∀ sides_other, sides_other < 13 → enclosing_polygons sides_other → ¬ (shares_vertex sides_other 14) :=
by
  sorry

-- Helper definition to indicate sharing vertex condition
def shares_vertex (polygon1_sides polygon2_sides : ℕ) : Prop := 
  -- Placeholder for the actual geometrical relationship; 
  -- this should encode the correct geometric condition of sharing a vertex
  sorry

end polygon_with_13_sides_shares_vertex_only_with_14_sided_polygon_l404_404191


namespace option_C_not_like_terms_l404_404527

theorem option_C_not_like_terms :
  ¬ (2 * (m : ℝ) == 2 * (n : ℝ)) :=
by
  sorry

end option_C_not_like_terms_l404_404527


namespace line_parallel_to_polar_axis_l404_404754

theorem line_parallel_to_polar_axis (r θ : ℝ) (h : (r, θ) = (2, π / 3)) :
  (y = r * sin θ) → (r * sin θ = √3) :=
by
  sorry

end line_parallel_to_polar_axis_l404_404754


namespace number_of_pencils_in_store_l404_404365

-- Definitions based on the given conditions
def ratio_pens_pencils_erasers (x : ℕ) : Prop :=
  let pens := 5 * x
  let pencils := 6 * x
  let erasers := 7 * x
  pencils = pens + 6 ∧ erasers = 2 * pens

-- The main statement to prove
theorem number_of_pencils_in_store (x : ℕ) (h : ratio_pens_pencils_erasers x) : 6 * x = 36 :=
by
  cases h with h1 h2
  -- The proof is omitted
  sorry

end number_of_pencils_in_store_l404_404365


namespace peanut_butter_sandwiches_l404_404583

-- Define the conditions
variable (B : ℕ) (C : ℕ) (P : ℕ)
variable (B_val : B = 35) (total_sandwiches : C + B + P = 80)
variable (ratio : C = B / 7)

-- Define the proof statement
theorem peanut_butter_sandwiches : P = 40 := by
  rw [B_val, ratio]
  -- From B_val, we know B = 35
  have C_val : C = 35 / 7 := ratio
  -- Simplify the known ratio
  rw [C_val] at total_sandwiches
  -- From C = 5, we substitute it into the total_sandwich statement
  have simplified_total : 5 + 35 + P = 80 := total_sandwiches
  -- Simplify further to obtain P
  linarith
  sorry

end peanut_butter_sandwiches_l404_404583


namespace fraction_single_men_equals_l404_404938

-- Define the total number of employees as E (as a noncomputable constant for generality)
noncomputable def E : ℝ := sorry

-- Define the fraction of women employees
def fraction_women : ℝ := 0.61

-- Define the fraction of married employees
def fraction_married_employees : ℝ := 0.60

-- Define the fraction of married women employees from the information given
def fraction_married_women : ℝ := 0.7704918032786885

-- Define the number of employees as a function of the total
def num_women (E : ℝ) : ℝ := fraction_women * E
def num_married (E : ℝ) : ℝ := fraction_married_employees * E
def num_married_women (E : ℝ) : ℝ := fraction_married_women * num_women E
def num_men (E : ℝ) : ℝ := (1 - fraction_women) * E

-- Define the number of married men
def num_married_men (E : ℝ) : ℝ := num_married E - num_married_women E

-- Define the number of single men
def num_single_men (E : ℝ) : ℝ := num_men E - num_married_men E

-- Define the fraction of single men
def fraction_single_men (E : ℝ) : ℝ := num_single_men E / num_men E

-- The proof goal: Verify that the fraction of men who are single equals the given answer
theorem fraction_single_men_equals :
  fraction_single_men E = 0.6671794871794872 := sorry

end fraction_single_men_equals_l404_404938


namespace N2O3_weight_l404_404523

-- Definitions from the conditions
def molecularWeightN : Float := 14.01
def molecularWeightO : Float := 16.00
def molecularWeightN2O3 : Float := (2 * molecularWeightN) + (3 * molecularWeightO)
def moles : Float := 4

-- The main proof problem statement
theorem N2O3_weight (h1 : molecularWeightN = 14.01)
                    (h2 : molecularWeightO = 16.00)
                    (h3 : molecularWeightN2O3 = (2 * molecularWeightN) + (3 * molecularWeightO))
                    (h4 : moles = 4) :
                    (moles * molecularWeightN2O3) = 304.08 :=
by
  sorry

end N2O3_weight_l404_404523


namespace monotonically_increasing_interval_find_phi_given_max_value_l404_404310

-- Given function definition
noncomputable def f (x φ : ℝ) : ℝ := (√(3) / 2) * cos (2 * x + φ) + sin (x)^2

-- Question 1: If φ = π / 6, find the interval where f(x) is monotonically increasing.
theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, f x (π / 6) = (1 / 2) * cos (2 * x + π / 3) + (1 / 2) →
  π * k - (2 * π / 3) ≤ x ∧ x ≤ π * k - (π / 6) →
  (deriv (f x (π / 6)) > 0) :=
sorry

-- Question 2: If the maximum value of f(x) is 3 / 2, find the value of φ.
theorem find_phi_given_max_value :
  (∀ x, f x φ ≤ 3 / 2) → (φ = π / 2) :=
sorry

end monotonically_increasing_interval_find_phi_given_max_value_l404_404310


namespace quadratic_distinct_real_roots_l404_404784

noncomputable def distinct_real_roots_intervals (k : ℝ) : Set ℝ :=
  if h : 0 < k then Set.Ioo (-2*Real.sqrt k) (2*Real.sqrt k)ᶜ else ∅

theorem quadratic_distinct_real_roots (k : ℝ) :
  ∀ (m : ℝ), (∀ (x : ℝ), x^2 + m * x + k = 0 → x.is_root) ↔ m ∈ distinct_real_roots_intervals k :=
sorry

end quadratic_distinct_real_roots_l404_404784


namespace evaluate_expression_l404_404257

-- Given Condition
def power_condition (a : ℤ) (n : ℕ) (h : even n) : ℤ := 
if h : even n then (a ^ n) else (a ^ n)

theorem evaluate_expression : 
  (-2)^(4^2) + 2^(4^2) = 2^(17) :=
by
  have h : even 16 := by norm_num
  have h1 : (-2)^16 = 2^16 := by simp [power_condition, h]
  have h2 : 2^16 = 2^16 := by norm_num
  have h3 : 4^2 = 16 := by norm_num
  rw [h3] at h1 h2
  calc
    (-2)^(4^2) + 2^(4^2)
        = (-2)^16 + 2^16 : by rw [h3]
    ... = 2^16 + 2^16     : by rw [h1]
    ... = 2^17           : by ring

end evaluate_expression_l404_404257


namespace cleaning_time_l404_404952

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l404_404952


namespace range_a_l404_404635

theorem range_a (a : ℝ) :
  (∀ x : ℝ, a ≤ x ∧ x ≤ 3 → -1 ≤ -x^2 + 2 * x + 2 ∧ -x^2 + 2 * x + 2 ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l404_404635


namespace binomial_12_10_eq_66_l404_404609

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404609


namespace john_took_more_chickens_l404_404067

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l404_404067


namespace community_service_selection_l404_404165

-- Defining the problem conditions
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def total_students : ℕ := num_boys + num_girls
def selection_size : ℕ := 4

-- Lean theorem statement asserting the number of valid selection schemes is 14
theorem community_service_selection :
  (nat.choose total_students selection_size) - (nat.choose num_boys selection_size) = 14 :=
by
  -- sorry is used to skip the proof; actual proof omitted as per instructions
  sorry

end community_service_selection_l404_404165


namespace courses_students_problem_l404_404498

theorem courses_students_problem :
  let courses := Fin 6 -- represent 6 courses
  let students := Fin 20 -- represent 20 students
  (∀ (C C' : courses), ∀ (S : Finset students), S.card = 5 → 
    ¬ ((∀ s ∈ S, ∃ s_courses : Finset courses, C ∈ s_courses ∧ C' ∈ s_courses) ∨ 
       (∀ s ∈ S, ∃ s_courses : Finset courses, C ∉ s_courses ∧ C' ∉ s_courses))) :=
by sorry

end courses_students_problem_l404_404498


namespace find_levels_satisfying_surface_area_conditions_l404_404706

theorem find_levels_satisfying_surface_area_conditions (n : ℕ) :
  let A_total_lateral := n * (n + 1) * Real.pi
  let A_total_vertical := Real.pi * n^2
  let A_total := n * (3 * n + 1) * Real.pi
  A_total_lateral = 0.35 * A_total → n = 13 :=
by
  intros A_total_lateral A_total_vertical A_total h
  sorry

end find_levels_satisfying_surface_area_conditions_l404_404706


namespace gain_percentage_on_second_book_l404_404343

theorem gain_percentage_on_second_book
  (total_cost : ℕ) (cost1 : ℕ) (selling_price : ℝ)
  (h_total_cost : total_cost = 300)
  (h_cost1 : cost1 = 175)
  (h_selling_price : selling_price = 175 - 0.15 * 175) :
  selling_price = 148.75 → (148.75 - (total_cost - cost1)) / (total_cost - cost1) * 100 = 19 := 
by
  intro h_sell_eq
  sorry

end gain_percentage_on_second_book_l404_404343


namespace john_bought_packs_l404_404002

def students_in_classes : List ℕ := [20, 25, 18, 22, 15]
def packs_per_student : ℕ := 3

theorem john_bought_packs :
  (students_in_classes.sum) * packs_per_student = 300 := by
  sorry

end john_bought_packs_l404_404002


namespace conjugate_of_complex_expr_is_correct_l404_404838

def conjugate_of_complex_expr : ℂ :=
  let z : ℂ := (5 : ℂ) / (2 - (1 : ℂ) * complex.i)
  in (z ^ 2).conj

theorem conjugate_of_complex_expr_is_correct :
  conjugate_of_complex_expr = 3 - 4 * complex.i :=
by
  sorry

end conjugate_of_complex_expr_is_correct_l404_404838


namespace minimum_pieces_required_to_form_square_l404_404874

theorem minimum_pieces_required_to_form_square : 
  ∀ (n : ℕ), 
  (∃ (s : ℕ), s * s = 36 ∧ s % 2 = 0) → 
  n = 12 := 
begin
  sorry,
end

end minimum_pieces_required_to_form_square_l404_404874


namespace total_earnings_l404_404147

theorem total_earnings (x y : ℝ) (hx : 2 * x * y = 35000) : 
    (18 * x * y / 100 + 20 * x * y / 100 + 20 * x * y / 100) = 10150 := by
  have hxy : x * y = 17500 := by
    calc
      x * y = 35000 / 2 := by
        rw [hx]
      ... = 17500 := by ring
  calc
    18 * x * y / 100 + 20 * x * y / 100 + 20 * x * y / 100
      = (18 + 20 + 20) * x * y / 100 := by ring
    ... = 58 * x * y / 100 := by ring
    ... = 58 * 17500 / 100 := by rw [hxy]
    ... = 10150 := by norm_num

end total_earnings_l404_404147


namespace initial_pens_l404_404041

theorem initial_pens (pens_left pens_sold : ℕ) (h1 : pens_left = 14) (h2 : pens_sold = 92) : 
  pens_left + pens_sold = 106 :=
by
  rw [h1, h2]
  exact rfl

end initial_pens_l404_404041


namespace intersection_distance_l404_404745

-- Define the initial points and the end points
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def start : Point := ⟨1, 2, 3⟩
def end : Point := ⟨3, -1, -2⟩

-- Define the radius and center of the sphere
def radius : ℝ := 2
def center : Point := ⟨0, 0, 0⟩

-- Define the parameterized line equation
def line (t : ℝ) : Point :=
  ⟨1 + 2 * t, 2 - 3 * t, 3 - 5 * t⟩

-- Define the equation parameters
def a : ℝ := 38
def b : ℝ := -50
def c : ℝ := 13

-- Calculate the roots using Vieta's formulas
def t1 : ℝ := (50 + real.sqrt (50^2 - 4 * 38 * 13)) / (2 * 38)
def t2 : ℝ := (50 - real.sqrt (50^2 - 4 * 38 * 13)) / (2 * 38)

def distance : ℝ :=
  real.sqrt 38 * real.abs (t1 - t2)

-- The final theorem to be proven
theorem intersection_distance :
  distance = 23 * real.sqrt 722 / 361 := sorry

end intersection_distance_l404_404745


namespace find_b_value_l404_404835

noncomputable def validate_b (b : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + b * x ≤ -x) ∧ (b < -1) → (let f := λ x : ℝ, x^2 + b * x in
  ∀ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) → f x₀ = -1/2)

theorem find_b_value : validate_b (-3/2) :=
by sorry

end find_b_value_l404_404835


namespace every_integer_appears_exactly_once_l404_404411

-- Define the sequence of integers
variable (a : ℕ → ℤ)

-- Define the conditions
axiom infinite_positives : ∀ n : ℕ, ∃ i > n, a i > 0
axiom infinite_negatives : ∀ n : ℕ, ∃ i > n, a i < 0
axiom distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n) ≠ (a j % n)

-- The proof statement
theorem every_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! i : ℕ, a i = x :=
sorry

end every_integer_appears_exactly_once_l404_404411


namespace range_of_m_l404_404339

theorem range_of_m (m x : ℝ) (h : (x + m) / 3 - (2 * x - 1) / 2 = m) (hx : x ≤ 0) : m ≥ 3 / 4 := 
sorry

end range_of_m_l404_404339


namespace redistribute_apples_l404_404904

theorem redistribute_apples (total_apples : ℕ) (n_baskets : ℕ) 
  (h_total : total_apples = 1000)
  (h_baskets : ∃ (apples : vector ℕ n_baskets), apples.sum = total_apples ∧ ∀ x, apples.nth x > 0):
  ∃ (y : ℕ), y * n_baskets ≥ 100 ∧ total_apples ≥ 100 :=
by sorry

end redistribute_apples_l404_404904


namespace binomial_12_10_l404_404585

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l404_404585


namespace vertical_asymptote_at_4_over_3_l404_404276

def vertical_asymptote (f : ℚ → ℚ) (a : ℚ) : Prop :=
  filter.tendsto f (filter.tendsto_filter_iff filter (λ eps > 0, ∃ δ > 0, ∀ x, |x - a| < δ -> |f x - 0| > 1 / eps))

def my_function (x : ℚ) : ℚ := (2 * x + 3) / (6 * x - 8)

theorem vertical_asymptote_at_4_over_3 : vertical_asymptote my_function (4 / 3) :=
sorry

end vertical_asymptote_at_4_over_3_l404_404276


namespace cleaning_time_with_doubled_an_speed_l404_404946

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l404_404946


namespace striped_turtles_adult_percentage_l404_404505

noncomputable def percentage_of_adult_striped_turtles (total_turtles : ℕ) (female_percentage : ℝ) (stripes_per_male : ℕ) (baby_stripes : ℕ) : ℝ :=
  let total_male := total_turtles * (1 - female_percentage)
  let total_striped_male := total_male / stripes_per_male
  let adult_striped_males := total_striped_male - baby_stripes
  (adult_striped_males / total_striped_male) * 100

theorem striped_turtles_adult_percentage :
  percentage_of_adult_striped_turtles 100 0.60 4 4 = 60 := 
  by
  -- proof omitted
  sorry

end striped_turtles_adult_percentage_l404_404505


namespace surface_area_independent_of_P_l404_404818

-- Definitions and conditions
variable (P O : Point) (a r : ℝ)

-- Condition: P is outside the fixed sphere S with center O
variable (hP_outside_S : dist P O > a)

-- Condition: The distance from P to O is r
variable (h_r_eq_PO : dist P O = r)

-- Condition: The surface area of the sphere with center P and radius PO lies inside the sphere S
def sphere_surface_area_inside (P O : Point) (a r: ℝ) : ℝ :=
  if dist P O <= a then 2 * π * a^2 else 0

-- Proof statement
theorem surface_area_independent_of_P (P O : Point) (a r : ℝ)
  (hP_outside_S : dist P O > a) (h_r_eq_PO : dist P O = r) : 
  sphere_surface_area_inside P O a r = 2 * π * a^2 := by
  sorry

end surface_area_independent_of_P_l404_404818


namespace right_triangle_sides_l404_404340

-- Definitions based on the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2
def perimeter (a b c : ℕ) : ℕ := a + b + c
def inscribed_circle_radius (a b c : ℕ) : ℕ := (a + b - c) / 2

-- The theorem statement
theorem right_triangle_sides (a b c : ℕ) 
  (h_perimeter : perimeter a b c = 40)
  (h_radius : inscribed_circle_radius a b c = 3)
  (h_right : is_right_triangle a b c) :
  (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
by sorry

end right_triangle_sides_l404_404340


namespace largest_R_for_inequality_l404_404725

theorem largest_R_for_inequality :
  ∃ R : ℕ, (R < 12) ∧ R^{2000} < 5^{3000} ∧ ∀ (n : ℕ), ((n < 12) ∧ (n^{2000} < 5^{3000})) → (n ≤ R) :=
by
  sorry

end largest_R_for_inequality_l404_404725


namespace at_most_one_vertex_property_l404_404430

theorem at_most_one_vertex_property (T : Tetrahedron) : 
    (∃ v w : T.vertex, v ≠ w ∧ 
    (∀ u1 u2 ∈ T.angles_at v, u1 + u2 > 180) ∧ 
    (∀ u1 u2 ∈ T.angles_at w, u1 + u2 > 180)) → false :=
sorry

end at_most_one_vertex_property_l404_404430


namespace sum_first_11_terms_l404_404748

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ k in finset.range n, a k

theorem sum_first_11_terms (a : ℕ → ℤ) (d : ℤ) (h_seq : arithmetic_sequence a)
  (h_cond : a 5 + a 7 = 10) :
  sum_of_first_n_terms a 11 = 55 :=
by
  sorry

end sum_first_11_terms_l404_404748


namespace pencils_per_row_cannot_be_determined_l404_404261

theorem pencils_per_row_cannot_be_determined
  (rows : ℕ)
  (total_crayons : ℕ)
  (crayons_per_row : ℕ)
  (h_total_crayons: total_crayons = 210)
  (h_rows: rows = 7)
  (h_crayons_per_row: crayons_per_row = 30) :
  ∀ (pencils_per_row : ℕ), false :=
by
  sorry

end pencils_per_row_cannot_be_determined_l404_404261


namespace log_base_3_expression_l404_404985

-- Define the necessary conditions
def eighty_one_eq : ℝ := 81
def three_to_the_fourth : ℝ := 3^4

-- Given conditions
lemma eighty_one_is_three_to_the_fourth : eighty_one_eq = three_to_the_fourth := by
  calc
    eighty_one_eq = 81 : by simp
    ... = 3^4 : by norm_num

def sqrt_nine_eq : ℝ := sqrt 9
def three : ℝ := 3

lemma sqrt_nine_is_three : sqrt_nine_eq = three := by
  calc
    sqrt_nine_eq = sqrt 9 : by simp
    ... = 3 : by norm_num

-- Final statement to prove
theorem log_base_3_expression : log 3 (81 * sqrt 9) = 5 := by
  have h1 : 81 = 3^4 := eighty_one_is_three_to_the_fourth
  have h2 : sqrt 9 = 3 := sqrt_nine_is_three
  rw [h1, h2]
  calc
    log 3 (3^4 * 3) = log 3 (3^5)  : by rw [← pow_add, add_comm]
    ... = 5 : log_pow_self

end log_base_3_expression_l404_404985


namespace solve_inequality_l404_404990

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l404_404990


namespace Jose_played_football_l404_404389

theorem Jose_played_football :
  ∀ (total_hours : ℝ) (basketball_minutes : ℕ) (minutes_per_hour : ℕ), total_hours = 1.5 → basketball_minutes = 60 →
  (total_hours * minutes_per_hour - basketball_minutes = 30) :=
by
  intros total_hours basketball_minutes minutes_per_hour h1 h2
  sorry

end Jose_played_football_l404_404389


namespace evaluate_f_5_minus_f_neg_5_l404_404351

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end evaluate_f_5_minus_f_neg_5_l404_404351


namespace min_value_of_expression_l404_404782

noncomputable def min_value_expression (a b c d : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2

theorem min_value_of_expression (a b c d : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  min_value_expression a b c d = 1 / 4 :=
sorry

end min_value_of_expression_l404_404782


namespace proof_problem_l404_404358

variable (p q : Prop)

theorem proof_problem
  (h₁ : p ∨ q)
  (h₂ : ¬p) :
  ¬p ∧ q :=
by
  sorry

end proof_problem_l404_404358


namespace distance_between_foci_hyperbola_l404_404973

noncomputable def distance_between_foci_of_hyperbola (x y : ℝ) : ℝ :=
  let foc1 := (Real.sqrt 2, Real.sqrt 2)
  let foc2 := (-Real.sqrt 2, -Real.sqrt 2)
  Real.sqrt ((foc1.1 - foc2.1)^2 + (foc1.2 - foc2.2)^2)

theorem distance_between_foci_hyperbola : 
  (∀ x y : ℝ, x^2 - 2*x*y + y^2 = 2 → distance_between_foci_of_hyperbola x y = 4) :=
by
  intros x y h
  apply Eq.trans
  calc
    distance_between_foci_of_hyperbola x y = Real.sqrt ((Real.sqrt 2 + Real.sqrt 2)^2 + (Real.sqrt 2 + Real.sqrt 2)^2) : by
      rw [distance_between_foci_of_hyperbola]
      sorry
    ... = 4 : by sorry
  sorry

end distance_between_foci_hyperbola_l404_404973


namespace A_value_l404_404638

variable (A M E T H : ℤ)

-- Given conditions
def H_value : H = 8
def MATH_value : M + A + T + H = 30
def TEAM_value : T + E + A + M = 36
def MEET_value : M + E + E + T = 40

-- Statement of the problem
theorem A_value : A = 10 :=
  by
  -- all our input conditions and proof would be here
  sorry

end A_value_l404_404638


namespace quarters_added_l404_404445

theorem quarters_added (initial_quarters : ℕ) (final_quarters : ℕ) (quarters_given : ℕ) 
  (h1 : initial_quarters = 21) (h2 : final_quarters = 70) : quarters_given = 49 :=
by
  have h := final_quarters - initial_quarters,
  sorry

end quarters_added_l404_404445


namespace angle_LEM_right_angle_l404_404040

-- Defining the points and their coordinates
variables {A B C K L M N E : Type}
variables [RightTriangle A B C]

-- Defining the properties of the squares on the legs of the right triangle
variables (square_ACKL : Square A C K L)
variables (square_BCMN : Square B C M N)

-- Defining the altitude from C to AB
variables (C E_altitude : Altitude C E A B)

-- Main theorem statement
theorem angle_LEM_right_angle :
  ∀ A B C K L M N E,
  RightTriangle A B C ∧
  Square A C K L ∧
  Square B C M N ∧
  Altitude C E A B →
  angle L E M = 90 :=
by sorry

end angle_LEM_right_angle_l404_404040


namespace weightlifting_winner_l404_404502

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l404_404502


namespace probability_five_chords_form_convex_pentagon_l404_404640

-- Definitions of problem conditions
variable (n : ℕ) (k : ℕ)

-- Eight points on a circle
def points_on_circle : ℕ := 8

-- Number of chords selected
def selected_chords : ℕ := 5

-- Total number of ways to select 5 chords from 28 possible chords
def total_ways : ℕ := Nat.choose 28 5

-- Number of ways to select 5 points from 8, forming a convex pentagon
def favorable_ways : ℕ := Nat.choose 8 5

-- The probability computation
def probability_pentagon (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_five_chords_form_convex_pentagon :
  probability_pentagon total_ways favorable_ways = 1 / 1755 :=
by
  sorry

end probability_five_chords_form_convex_pentagon_l404_404640


namespace john_spent_on_cloth_l404_404388

-- Conditions
def length_of_cloth := 9.25
def cost_per_meter := 43

-- Question and correct answer
theorem john_spent_on_cloth : length_of_cloth * cost_per_meter = 397.75 :=
by
  sorry

end john_spent_on_cloth_l404_404388


namespace twenty_one_less_than_sixty_thousand_l404_404525

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 :=
by
  sorry

end twenty_one_less_than_sixty_thousand_l404_404525


namespace product_terms_l404_404105

noncomputable def geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem product_terms (
  a : ℕ → ℝ)
  (r : ℝ)
  (h1 : geometric_progression_common_ratio a 3)
  (h2 : ∏ i in (finset.range 10).map (λ n, 3 * n + 1), a i = 3 ^ 100) :
  (∏ i in (finset.range 10).map (λ n, 3 * n + 3), a i) = 3 ^ 120 :=
by {
  sorry
}

end product_terms_l404_404105


namespace sum_c_k_squared_l404_404246

noncomputable def c_k (k : ℕ) : ℚ :=
k + has_coe_to_fun.coe (1 / (3 * k + c_k k))

theorem sum_c_k_squared :
  (∑ k in finset.range 11, (c_k (k + 1))^2) = 517 :=
by
  sorry

end sum_c_k_squared_l404_404246


namespace football_team_girls_l404_404910

theorem football_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : B + (1 / 3 * G) = 18) : G = 18 :=
by {
  have h3 : B = 30 - G := by linarith,
  have h4 : (30 - G) + (1 / 3 * G) = 18 := by linarith,
  have h5 : 30 - (2 / 3 * G) = 18 := by linarith,
  have h6 : (2 / 3 * G) = 12 := by linarith,
  have h7 : G = 12 * (3 / 2) := by linarith,
  show G = 18, by linarith,
  sorry
}

end football_team_girls_l404_404910


namespace binom_12_10_eq_66_l404_404595

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l404_404595


namespace intersections_outside_polygon_l404_404670

theorem intersections_outside_polygon (n : ℕ) (h1: n ≥ 6) :
  let m := n * (n - 3) / 2,
      inside_intersections := (m * (m - 1) / 2 - (n * (n - 3) / 2) * (n - 4) / 2),
      total_intersections := inside_intersections - (n * (n-1) * (n-2) * (n-3) / 24) in
  m * (m - 1) / 2 - total_intersections = (1 / 12) * n * (n - 3) * (n - 4) * (n - 5) := 
sorry

end intersections_outside_polygon_l404_404670


namespace tan_C_value_l404_404681

variable (a b c A B C : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
variable (angle_A : 0 < A ∧ A < π)
variable (angle_B : 0 < B ∧ B < π)
variable (angle_C : 0 < C ∧ C < π)
variable (angle_sum : A + B + C = π)
variable (cos_rule : a^2 + b^2 - c^2 = - (2 / 3) * a * b)

theorem tan_C_value : 
  ∃ tanC : ℝ, tanC = -2 * real.sqrt 2 :=
sorry

end tan_C_value_l404_404681


namespace weightlifting_winner_l404_404503

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l404_404503


namespace intersection_eq_l404_404794

namespace SetIntersection

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Prove the intersection of A and B is {1, 2}
theorem intersection_eq : A ∩ B = {1, 2} :=
by
  sorry

end SetIntersection

end intersection_eq_l404_404794


namespace find_7th_term_l404_404065

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem find_7th_term 
    (a d : ℤ) 
    (h3 : a + 2 * d = 17) 
    (h5 : a + 4 * d = 39) : 
    arithmetic_sequence a d 7 = 61 := 
sorry

end find_7th_term_l404_404065


namespace mike_profit_l404_404802

theorem mike_profit 
  (num_acres_bought : ℕ) (price_per_acre_buy : ℤ) 
  (fraction_sold : ℚ) (price_per_acre_sell : ℤ) :
  num_acres_bought = 200 →
  price_per_acre_buy = 70 →
  fraction_sold = 1/2 →
  price_per_acre_sell = 200 →
  let cost_of_land := price_per_acre_buy * num_acres_bought,
      num_acres_sold := (fraction_sold * num_acres_bought),
      revenue_from_sale := price_per_acre_sell * num_acres_sold,
      profit := revenue_from_sale - cost_of_land
  in profit = 6000 := by
  intros h1 h2 h3 h4
  let cost_of_land := price_per_acre_buy * num_acres_bought
  let num_acres_sold := (fraction_sold * num_acres_bought)
  let revenue_from_sale := price_per_acre_sell * num_acres_sold
  let profit := revenue_from_sale - cost_of_land
  rw [h1, h2, h3, h4]
  sorry

end mike_profit_l404_404802


namespace sin_minus_cos_l404_404685

-- Define the conditions
variable (α : ℝ)
hypothesis (hα : π/2 < α ∧ α < π) -- α is in the second quadrant
hypothesis (h_cos_2α : cos (2 * α) = - (sqrt 5) / 3)

-- State the theorem
theorem sin_minus_cos (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_cos_2α : cos (2 * α) = - (sqrt 5) / 3) :
  sin (α) - cos (α) = sqrt 15 / 3 := 
sorry

end sin_minus_cos_l404_404685


namespace min_value_f_l404_404222

-- Definitions based on the given conditions
structure Triangle where
  A B C : Point
  side_length : ℝ
  eq_side: side_length = 5

constants (Point : Type)
          (Distance : Point → Point → ℝ)
          (Circumcircle_intersect : Point → Point → Point → Point → Point → Point)
          (D_on_segment_BC : Point → Point → Point → Point → Bool)
          (EA_DF_function : ℝ → ℝ)

noncomputable def point_on_extension (A B P : Point) :=
  Distance A P = 9

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 45) / (real.sqrt (x^2 - 5*x + 25))

theorem min_value_f :
  let T := Triangle.mk A B C 5 (by rfl)
  ∃ x : ℝ, f(x) = 4 * real.sqrt 5 := by
  sorry

end min_value_f_l404_404222


namespace sum_first_60_digits_of_fraction_l404_404140

theorem sum_first_60_digits_of_fraction (h : (1 : ℚ) / 1001 = 0.000999999999999) : 
  let digits : List ℕ := [0, 0, 0, 9, 9, 9].repeat 10
  (digits.take 60).sum = 270 :=
by 
  sorry

end sum_first_60_digits_of_fraction_l404_404140


namespace min_balls_to_ensure_three_of_same_color_l404_404160

theorem min_balls_to_ensure_three_of_same_color
  (white_balls black_balls blue_balls : ℕ)
  (h_white_balls : white_balls = 5)
  (h_black_balls : black_balls = 5)
  (h_blue_balls : blue_balls = 2)
  : ∃ n, n = 7 ∧ (∀ drawn_balls, drawn_balls ⊇ {white_balls, black_balls, blue_balls} → (∃ color ∈ {white, black, blue}, count drawn_balls color ≥ 3)) :=
  by
  sorry

end min_balls_to_ensure_three_of_same_color_l404_404160


namespace find_t_l404_404013

theorem find_t (t : ℝ) :
  let P := (t - 5, -2)
  let Q := (-3, t + 4)
  let M := ((t - 8) / 2, (t + 2) / 2)
  (dist M P) ^ 2 = t^2 / 3 →
  t = -12 + 2 * Real.sqrt 21 ∨ t = -12 - 2 * Real.sqrt 21 := sorry

end find_t_l404_404013


namespace non_dominated_distribution_favorite_toy_l404_404110

variable (n : ℕ)

-- We assume each child has a strict preference ordering on the toys.
structure PreferenceOrdering where
  preferred : Fin n → Fin n → Prop
  strict_preferred : ∀ {x y : Fin n}, preferred x y → ¬ preferred y x
  total_order : ∀ x y, x ≠ y → preferred x y ∨ preferred y x

-- We define a distribution as a mapping from children to toys.
def Distribution := Fin n → Fin n

-- A distribution A dominates a distribution B if every child considers their toy in A at least as preferable as in B.
def dominates (A B : Distribution) (pref : Fin n → PreferenceOrdering n) : Prop :=
  ∀ (i : Fin n), ∀ (j : Fin n), pref i j.preferred (A i) (B i)

-- The theorem to be proved.
theorem non_dominated_distribution_favorite_toy
  (A : Distribution)
  (pref : Fin n → PreferenceOrdering n)
  (non_dominated : ∀ B, B ≠ A → ¬ dominates B A pref) :
  ∃ i, ∀ j, ¬ pref i j.preferred (pref i).favorite_toy (A i) := 
  sorry

end non_dominated_distribution_favorite_toy_l404_404110


namespace value_of_livestock_l404_404540

variable (x y : ℝ)

theorem value_of_livestock :
  (5 * x + 2 * y = 10) ∧ (2 * x + 5 * y = 8) :=
sorry

end value_of_livestock_l404_404540


namespace two_person_subcommittees_from_eight_l404_404710

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l404_404710


namespace range_of_x0_proof_l404_404305

noncomputable def range_of_x0 : set ℝ := { x_0 | 0 ≤ x_0 ∧ x_0 ≤ 2 }

theorem range_of_x0_proof (x_0 : ℝ) (hx : ∃ (N : ℝ × ℝ), N.1^2 + N.2^2 = 1 ∧
                          ∃ M : ℝ × ℝ, M = (x_0, 2 - x_0) ∧
                          ∠ (0, 0) M N = 30) :
  x_0 ∈ range_of_x0 :=
begin
  sorry
end

end range_of_x0_proof_l404_404305


namespace value_of_f1_l404_404312

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x * real.log (3 - x) else sorry

theorem value_of_f1 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x <= 0 → f x = -x * real.log (3 - x)) →
  f 1 = -real.log 4 :=
begin
  intros h_odd h_fx,
  sorry
end

end value_of_f1_l404_404312


namespace manufacturing_cost_l404_404845

variable (M S TC : ℝ) (G : ℝ := 20)

-- Conditions
def transportation_cost_per_shoe : TC = 500 / 100 := by norm_num
def selling_price_per_shoe : S = 222 := by norm_num
def gain_percentage : G = 20 := by norm_num

-- Denote cost price
def cost_price : ℝ := M + TC

-- Selling price equation
def selling_price_definition : S = cost_price * (1 + G/100) := by
  simp [cost_price, G] 
  norm_num

theorem manufacturing_cost (h1 : selling_price_per_shoe S) (h2 : gain_percentage G) (h3 : transportation_cost_per_shoe TC) :
  M = 180 := by
  sorry -- proof will be here

end manufacturing_cost_l404_404845


namespace mike_profit_l404_404800

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l404_404800


namespace trapezoid_length_XY_l404_404383

noncomputable def trapezoid_problem (PQ QR RS PS : ℝ) (angle_P angle_S : ℝ) (mid_X mid_Y : ℝ) : Prop :=
  PQ ≠ 0 ∧ QR = 1500 ∧ PS = 3000 ∧ angle_P = 37 ∧ angle_S = 53 ∧
  let X := QR / 2 in
  let Y := PS / 2 in
  XY = Y - X ∧ XY = 750

theorem trapezoid_length_XY :
  trapezoid_problem PQ QR RS PS 37 53 mid_X mid_Y :=
by
  sorry

end trapezoid_length_XY_l404_404383


namespace indoor_table_chairs_l404_404906

theorem indoor_table_chairs (x : ℕ) :
  (9 * x) + (11 * 3) = 123 → x = 10 :=
by
  intro h
  sorry

end indoor_table_chairs_l404_404906


namespace area_of_triangle_AOB_l404_404696

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_line_eq : ℝ → ℝ := λ y, 1 - y

def parabola_curve_eq (x y : ℝ) : Prop := y^2 = 4 * x

theorem area_of_triangle_AOB
  (focus := parabola_focus)
  (line_eq := parabola_line_eq)
  (curve_eq := parabola_curve_eq) :
  let O := (0, 0) in
  let A := (1 - √(2+√8), √(2+√8)) in
  let B := (1 - √(2-√8), -√(2-√8)) in
  |A.2 - B.2| = 4 * sqrt 2 ∧
  (1 / 2) * |O.1 - focus.1| * |A.2 - B.2| = 2 * sqrt 2 :=
by
  sorry

end area_of_triangle_AOB_l404_404696


namespace winner_lifted_weight_l404_404500

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l404_404500


namespace cost_price_of_watch_l404_404926

theorem cost_price_of_watch :
  ∃ (CP : ℝ), (CP * 1.07 = CP * 0.88 + 250) ∧ CP = 250 / 0.19 :=
sorry

end cost_price_of_watch_l404_404926


namespace bouquet_combinations_l404_404562

theorem bouquet_combinations :
  let total_cost := 60
  let lily_cost := 4
  let daisy_cost := 2
  ∃ n : ℕ, (∀ l : ℕ, 0 ≤ l → l ≤ 15 → ∃ d : ℕ, daisy_cost * d + lily_cost * l = total_cost) ∧
              (n = 16) :=
by
  let total_cost := 60
  let lily_cost := 4
  let daisy_cost := 2
  have h : ∃ n : ℕ, (∀ l : ℕ, 0 ≤ l → l ≤ 15 → ∃ d : ℕ, daisy_cost * d + lily_cost * l = total_cost) ∧ (n = 16), from sorry
  exact h

end bouquet_combinations_l404_404562


namespace coordinates_on_y_axis_l404_404728

theorem coordinates_on_y_axis (m : ℝ) (h : m + 1 = 0) : (m + 1, m + 4) = (0, 3) :=
by
  sorry

end coordinates_on_y_axis_l404_404728


namespace sqrt_36_eq_6_cube_root_neg_a_125_l404_404836

theorem sqrt_36_eq_6 : ∀ (x : ℝ), 0 ≤ x ∧ x^2 = 36 → x = 6 :=
by sorry

theorem cube_root_neg_a_125 : ∀ (a y : ℝ), y^3 = - a / 125 → y = - (a^(1/3)) / 5 :=
by sorry

end sqrt_36_eq_6_cube_root_neg_a_125_l404_404836


namespace abs_neg_2023_l404_404453

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l404_404453


namespace angle_BAC_45_l404_404744

-- Define the type and hypotheses
variables {ABC : Type*} [triangle ABC] 
variables {A B C H : point ABC}
variables (acute_triangle : triangle.acute ABC) 
variables (orthocenter : orthocenter H A B C)
variables (altitude_eq : distance A H = distance B C)

-- Statement of the theorem
theorem angle_BAC_45 :
  angle BAC = 45 := 
sorry

end angle_BAC_45_l404_404744


namespace binomial_12_10_eq_66_l404_404611

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404611


namespace John_took_more_chickens_than_Ray_l404_404072

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l404_404072


namespace total_clothing_ironed_l404_404438

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l404_404438


namespace domain_of_sqrt_log_l404_404086

theorem domain_of_sqrt_log {x : ℝ} : (2 < x ∧ x ≤ 5 / 2) ↔ 
  (5 - 2 * x > 0 ∧ 0 ≤ Real.logb (1 / 2) (5 - 2 * x)) :=
sorry

end domain_of_sqrt_log_l404_404086


namespace area_of_vegetable_patch_l404_404256

theorem area_of_vegetable_patch : ∃ (a b : ℕ), 
  (2 * (a + b) = 24 ∧ b = 3 * a + 2 ∧ (6 * (a + 1)) * (6 * (b + 1)) = 576) :=
sorry

end area_of_vegetable_patch_l404_404256


namespace binomial_12_10_eq_66_l404_404620

theorem binomial_12_10_eq_66 : binomial 12 10 = 66 :=
by
  sorry

end binomial_12_10_eq_66_l404_404620


namespace jamshid_taimour_painting_problem_l404_404386

/-- Jamshid and Taimour Painting Problem -/
theorem jamshid_taimour_painting_problem (T : ℝ) (h1 : T > 0)
  (h2 : 1 / T + 2 / T = 1 / 5) : T = 15 :=
by
  -- solving the theorem
  sorry

end jamshid_taimour_painting_problem_l404_404386


namespace pyramid_stack_of_balls_total_count_l404_404555

theorem pyramid_stack_of_balls_total_count :
  ∃ n : ℕ, ∃ d : ℕ, 
  let a := 4 in  -- first term
  d = 3 ∧       -- common difference
  a + (n-1) * d = 40 ∧   -- nth term is 40
  (n * (a + 40)) / 2 = 286 := -- sum of the arithmetic series equals 286
begin
  -- Defining the variables and conditions
  let a := 4,
  let d := 3,
  -- finding the number of terms n
  have h1 : ∃ n : ℕ, a + (n-1) * d = 40,
  { use 13,
    linarith },
  cases h1 with n hn,
  -- showing that the sum of the series is 286
  have h2 : (n * (a + 40)) / 2 = 286,
  { rw [←hn], 
    norm_num },
  -- final proof
  use [n, d],
  split, -- split the conjunction
  { exact rfl }, -- d = 3
  split, -- split the conjunction
  { exact hn }, -- a + (n-1) * d = 40
  { exact h2 } -- (n * (a + 40)) / 2 = 286
end

end pyramid_stack_of_balls_total_count_l404_404555


namespace sn_value_l404_404015

variable {a : ℕ → ℕ}

-- Given condition
def S (n : ℕ) : ℕ := 2 * a n - 3

-- To prove
theorem sn_value (n : ℕ) : S n = 3 * 2^n - 3 := by
  sorry

end sn_value_l404_404015


namespace distinct_flags_l404_404171

axiom unique_colors (c : Nat) : c = 5

axiom middle_stripe_options : unique_colors 5

axiom top_stripe_options : unique_colors 4

axiom bottom_stripe_options : unique_colors 3

theorem distinct_flags : ∃ n, n = 5 * 4 * 3 ∧ n = 60 := by
  use 60
  split
  · exact Eq.refl (5 * 4 * 3)
  · exact Eq.refl 60

end distinct_flags_l404_404171


namespace even_exponents_l404_404008

theorem even_exponents (n : ℕ) (h1 : 2 ≤ n) (a : Π (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n → ℕ) 
(K : (fin n → ℝ) → ℝ)
(h2 : ∀ (x : fin n → ℝ), 0 ≤ K x) :
(∀ i j (hij : 1 ≤ i ∧ i < j ∧ j ≤ n), even (a i j hij)) :=
sorry

end even_exponents_l404_404008


namespace average_age_of_new_students_l404_404455

def original_class_strength : Nat := 18
def original_average_age : Nat := 40
def new_students : Nat := 18
def new_average_age (O A_O N : Nat) := (O * A_O + N * A_N) / (O + N) 

theorem average_age_of_new_students
  (O : Nat) (A_O : Nat) (N : Nat) (A_new : Nat)
  (h1 : O = original_class_strength)
  (h2 : A_O = original_average_age)
  (h3 : N = new_students)
  (h4 : A_new = new_average_age O A_O N) :
  A_new = 32 :=
by
  rw [h1, h2, h3]
  sorry

end average_age_of_new_students_l404_404455


namespace find_x_l404_404886

theorem find_x : ∃ (x a b : ℤ), 100 + x = a^2 ∧ 164 + x = b^2 ∧ x = 125 :=
by 
  exists 125, 17, 15
  split
  { 
    -- 100 + 125 = 225 = 15^2
    exact rfl
  }
  {
    -- 164 + 125 = 289 = 17^2
    exact rfl
  }
  {
    -- x equals to 125
    exact rfl
  }
  sorry

end find_x_l404_404886


namespace not_necessarily_true_inequality_l404_404291

theorem not_necessarily_true_inequality (a b : ℝ) (h : a > b) : ¬ (∀ a b, (a > b) → (a^2 > a * b)) :=
by sorry

end not_necessarily_true_inequality_l404_404291


namespace solve_for_x_l404_404724

theorem solve_for_x (x : ℝ) (h : x^4 = (-3)^4) : x = 3 ∨ x = -3 :=
sorry

end solve_for_x_l404_404724


namespace rocket_parachute_opens_l404_404193

theorem rocket_parachute_opens (h t : ℝ) : h = -t^2 + 12 * t + 1 ∧ h = 37 -> t = 6 :=
by sorry

end rocket_parachute_opens_l404_404193


namespace sin_inverse_tangent_sum_l404_404644

noncomputable def sin_sum (a b : ℝ) : ℝ := Real.sin (a + b)

theorem sin_inverse_tangent_sum :
  ∀ (a b : ℝ), a = Real.arcsin (3 / 5) → b = Real.arctan 2 → sin_sum a b = 11 * Real.sqrt 5 / 25 :=
by
  assume a b ha hb
  rw [ha, hb]
  sorry

end sin_inverse_tangent_sum_l404_404644


namespace integer_solutions_count_l404_404717

theorem integer_solutions_count :
  (∃ n : ℕ, 3* n * n.succ + 17 * n + 14 ≤ 25) → 11 = 11 := 
by {
  intro h,
  sorry
}

end integer_solutions_count_l404_404717


namespace find_an_l404_404490

variables {ℝ : Type*} [linear_ordered_field ℝ]
noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else if n = 1 then 3/5 else 3/5 * (2/5)^(n-1)

theorem find_an (n : ℕ) (Sn : ℕ → ℝ) (h_Sn : ∀ n, Sn n = 1 - 2/3 * a_sequence n)
  (h_an : ∀ n, a_sequence n = Sn n - Sn (n-1)) :
  a_sequence n = 3/5 * (2/5)^(n-1) :=
sorry

end find_an_l404_404490


namespace constant_term_is_208_l404_404100

noncomputable def P : ℤ[X] := sorry
axiom P_integer_coefficients : ∃ Q : ℤ[X], ∃ a_0 : ℤ, P = λ x => x * Q(x) + a_0
axiom P_at_19 : P 19 = 1994
axiom P_at_94 : P 94 = 1994
axiom constant_term_abs : ∃ a_0 : ℤ, (λ x => x * (∃ Q : ℤ[X], P = λ x => x * Q(x) + a_0)) 0 = a_0 ∧ abs a_0 < 1000

theorem constant_term_is_208 : ∃ a_0, (a_0 = 208) :=
  sorry

end constant_term_is_208_l404_404100


namespace compound_interest_second_year_is_976_04_l404_404837

noncomputable def principal_amount (CI_3 : ℝ) (rate : ℝ) : ℝ :=
  CI_3 / ((1 + rate)^3 - 1)

noncomputable def compound_interest_second_year (P : ℝ) (rate : ℝ) : ℝ :=
  P * ((1 + rate)^2 - 1)

theorem compound_interest_second_year_is_976_04 (CI_3 : ℝ) (rate : ℝ) (CI_2 : ℝ) :
  CI_3 = 1540 → rate = 0.10 → CI_2 = 0.21 * (1540 / 0.331) → CI_2 ≈ 976.04 :=
by
  intros h1 h2 h3
  unfold compound_interest_second_year
  sorry

end compound_interest_second_year_is_976_04_l404_404837


namespace measure_of_B_l404_404370

-- Define the conditions (angles and their relationships)
variable (angle_P angle_R angle_O angle_B angle_L angle_S : ℝ)
variable (sum_of_angles : angle_P + angle_R + angle_O + angle_B + angle_L + angle_S = 720)
variable (supplementary_O_S : angle_O + angle_S = 180)
variable (right_angle_L : angle_L = 90)
variable (congruent_angles : angle_P = angle_R ∧ angle_R = angle_B)

-- Prove the measure of angle B
theorem measure_of_B : angle_B = 150 := by
  sorry

end measure_of_B_l404_404370


namespace area_of_parallelogram_l404_404648

noncomputable def sine : ℝ → ℝ := λ x => Real.sin(x * Real.pi / 180)

constant base : ℝ := 32
constant side : ℝ := 18
constant angle_deg : ℝ := 75

theorem area_of_parallelogram : 
  let h := side * sine angle_deg in
  let area := base * h in
  abs (area - 556.36) < 0.01 :=
by sorry

end area_of_parallelogram_l404_404648


namespace Felipe_time_to_build_house_l404_404125

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end Felipe_time_to_build_house_l404_404125


namespace max_self_intersections_polyline_7_l404_404522

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end max_self_intersections_polyline_7_l404_404522


namespace angle_DCE_l404_404771

variables (A B C D E : Type)
variables [is_rhombus A B C D E]
variables (mEAB mEBC : ℝ)

def condition1 (E : Type) : Prop := E ∈ interior (convex_hull A B C D)

def condition2 (A E B : Type) : Prop := dist A E = dist E B

def condition3 (mEAB : ℝ) : Prop := mEAB = 11

def condition4 (mEBC : ℝ) : Prop := mEBC = 71

theorem angle_DCE (A B C D E : Type)
  [is_rhombus A B C D] (mEAB mEBC : ℝ)
  (h1 : condition1 E)
  (h2 : condition2 A E B)
  (h3 : condition3 mEAB)
  (h4 : condition4 mEBC) :
  angle A D E C = 68 :=
sorry

end angle_DCE_l404_404771


namespace relationship_of_points_on_inverse_proportion_l404_404419

theorem relationship_of_points_on_inverse_proportion :
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  sorry

end relationship_of_points_on_inverse_proportion_l404_404419


namespace probability_no_empty_boxes_is_correct_probability_one_empty_box_is_correct_l404_404044

-- Definitions and conditions
def balls := 4
def boxes := 4
def total_outcomes := boxes ^ balls

def no_empty_boxes_outcomes := (finset.range boxes).card.pmul (finset.range balls).card
def probability_no_empty_boxes := no_empty_boxes_outcomes / total_outcomes

def one_empty_box_outcomes := 
  (@finset.univ (fin (boxes - 1))).card * (@finset.univ (fin (balls - 2))).card.pmul (@finset.univ (fin (boxes - 2))).card

def probability_one_empty_box := one_empty_box_outcomes / total_outcomes

-- Proof statements
theorem probability_no_empty_boxes_is_correct : 
  probability_no_empty_boxes = 3 / 32 := by sorry

theorem probability_one_empty_box_is_correct : 
  probability_one_empty_box = 9 / 16 := by sorry

end probability_no_empty_boxes_is_correct_probability_one_empty_box_is_correct_l404_404044


namespace positive_number_equals_seven_l404_404892

theorem positive_number_equals_seven (x : ℝ) (h_pos : x > 0) (h_eq : x - 4 = 21 / x) : x = 7 :=
sorry

end positive_number_equals_seven_l404_404892


namespace work_completion_days_l404_404547

noncomputable def A_days : ℝ := 20
noncomputable def B_days : ℝ := 35
noncomputable def C_days : ℝ := 50

noncomputable def A_work_rate : ℝ := 1 / A_days
noncomputable def B_work_rate : ℝ := 1 / B_days
noncomputable def C_work_rate : ℝ := 1 / C_days

noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate + C_work_rate
noncomputable def total_days : ℝ := 1 / combined_work_rate

theorem work_completion_days : total_days = 700 / 69 :=
by
  -- Proof steps would go here
  sorry

end work_completion_days_l404_404547


namespace union_complement_l404_404699

def universalSet : Set ℤ := { x | x^2 < 9 }

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

def complement_I_B : Set ℤ := universalSet \ B

theorem union_complement :
  A ∪ complement_I_B = {0, 1, 2} :=
by
  sorry

end union_complement_l404_404699


namespace rectangular_field_diagonal_length_l404_404816

noncomputable def diagonal_length_of_rectangular_field (a : ℝ) (A : ℝ) : ℝ :=
  let b := A / a
  let d := Real.sqrt (a^2 + b^2)
  d

theorem rectangular_field_diagonal_length :
  let a : ℝ := 14
  let A : ℝ := 135.01111065390137
  abs (diagonal_length_of_rectangular_field a A - 17.002) < 0.001 := by
    sorry

end rectangular_field_diagonal_length_l404_404816


namespace problem_equivalence_l404_404019

def count_ordered_quadruples_odd_sum (n : ℕ) : ℕ :=
  if hn : n % 2 = 0 then 
    (n / 2 - 1).choose 3 else 
    0
  
theorem problem_equivalence : 
  let m := count_ordered_quadruples_odd_sum 84 in
  m / 100 = 123.41 := 
by
  sorry

end problem_equivalence_l404_404019


namespace sum_of_distinct_prime_factors_l404_404270

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l404_404270


namespace basketball_team_selection_l404_404817

def team : Finset String := { "Player₀", "Player₁", "Player₂", "Player₃", "Player₄", "Player₅", "Player₆", "Player₇", "Player₈", "Player₉", "Player₁₀", "Player₁₁", "Player₁₂", "Player₁₃", "Player₁₄", "Player₁₅", "Player₁₆", "Player₁₇" }

def quadruplets : Finset String := { "Brian", "Bruce", "Brad", "Billy" }

theorem basketball_team_selection :
  ∑ (S : Finset (Finset String)) in team.powerset.filter (λ s, s.card = 5 ∧ (s ∩ quadruplets).card ≤ 2), 1 = 8190 :=
by sorry

end basketball_team_selection_l404_404817


namespace find_a_l404_404726

theorem find_a (a : ℚ) (h : a + a / 3 + a / 4 = 4) : a = 48 / 19 := by
  sorry

end find_a_l404_404726


namespace commission_percentage_is_9_l404_404194

def initial_commission_percentage (total_commission : ℝ) (bonus : ℝ) (excess_sales_percentage : ℝ) : ℝ :=
  let excess_sales := bonus / (excess_sales_percentage / 100)
  let total_sales := 10000 + excess_sales
  let initial_commission := total_commission - bonus
  (initial_commission * 100) / total_sales

theorem commission_percentage_is_9 :
  initial_commission_percentage 1380 120 3 = 9 :=
by
  -- Proof goes here
  sorry

end commission_percentage_is_9_l404_404194


namespace sum_first_1995_terms_l404_404092

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 1 else 4 * Int.natAbs (real.sqrt (sequence_sum (n - 1))) + 4

def sequence_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence_term i

theorem sum_first_1995_terms :
  sequence_sum 1995 = 15912121 :=
sorry

end sum_first_1995_terms_l404_404092


namespace f_neg_one_l404_404405

noncomputable def f : ℝ → ℝ :=
λ x, if h : x > 0 then Real.log (x + 7) / Real.log 2 else 0

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_def_pos (x : ℝ) (h : x > 0) : f x = Real.log (x + 7) / Real.log 2

theorem f_neg_one : f (-1) = -3 := by
  sorry

end f_neg_one_l404_404405


namespace solution_to_quadratic_inequality_l404_404989

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l404_404989


namespace coordinates_OQ_quadrilateral_area_range_l404_404704

variables {p : ℝ} (p_pos : 0 < p)
variables {x0 x1 x2 y0 y1 y2 : ℝ} (h_parabola_A : y1^2 = 2*p*x1) (h_parabola_B : y2^2 = 2*p*x2) (h_parabola_M : y0^2 = 2*p*x0)
variables {a : ℝ} (h_focus_x : a = x0 + p) 

variables {FA FM FB : ℝ}
variables (h_arith_seq : ( FM = FA - (FA - FB) / 2 ))

-- Step 1: Prove the coordinates of OQ
theorem coordinates_OQ : (x0 + p, 0) = (a, 0) :=
by
  -- proof will be completed here
  sorry 

variables {x0_val : ℝ} (x0_eq : x0 = 2) {FM_val : ℝ} (FM_eq : FM = 5 / 2)

-- Step 2: Prove the area range of quadrilateral ABB1A1
theorem quadrilateral_area_range : ∀ (p : ℝ), 0 < p →
  ∀ (x0 x1 x2 y1 y2 FM OQ : ℝ), 
    x0 = 2 → FM = 5 / 2 → OQ = 3 → (y1^2 = 2*p*x1) → (y2^2 = 2*p*x2) →
  ( ∃ S : ℝ, 0 < S ∧ S ≤ 10) :=
by
  -- proof will be completed here
  sorry 

end coordinates_OQ_quadrilateral_area_range_l404_404704


namespace sum_coordinate_B_l404_404045

-- Define the conditions
def point (ℝ : Type) := ℝ × ℝ
def midpoint (M A B : point ℝ) : Prop :=
  (fst M = (fst A + fst B) / 2) ∧ (snd M = (snd A + snd B) / 2)

-- Given points A and M
def A : point ℝ := (10, 2)
def M : point ℝ := (5, 3)

-- Define the proof problem
theorem sum_coordinate_B (B : point ℝ) (hM : midpoint M A B) : (fst B + snd B) = 4 := 
sorry

end sum_coordinate_B_l404_404045


namespace eventually_decreasing_sequence_l404_404697

open Nat

def a (n : ℕ) : ℝ := 100 ^ n / n.factorial

theorem eventually_decreasing_sequence : ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a n < a (n - 1) :=
by
  sorry

end eventually_decreasing_sequence_l404_404697


namespace instantaneous_velocity_zero_at_65_div_98_l404_404755

variable (t : ℝ)

def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_zero_at_65_div_98 :
  ∃ t : ℝ, (h' : t → ℝ) = (-9.8 * t + 6.5) → h' 0 = 0 → t = 65 / 98 :=
by
  -- Define the derivative of h
  let h' := λ t, -9.8 * t + 6.5

  -- Suppose that instantaneous velocity is zero
  have h'_zero : h' t = 0,
  sorry

  -- Solve for t
  have t_val : t = 65 / 98,
  sorry

  -- Conclude existence
  exact ⟨65 / 98, h', h'_zero, t_val⟩

end instantaneous_velocity_zero_at_65_div_98_l404_404755


namespace smallest_number_in_list_l404_404497

theorem smallest_number_in_list :
  let numbers := [3.4, 7 / 2, 1.7, 27 / 10, 2.9] in
  ∃ x ∈ numbers, ∀ y ∈ numbers, x ≤ y ∧ x = 1.7 := by
  sorry

end smallest_number_in_list_l404_404497


namespace part1_part2_l404_404031

-- Define the lines and their intersection point
def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def intersection (x y : ℝ) := -2 = x ∧ 2 = y

-- Part 1: Prove the equation of line l passing through the intersection is parallel to 3x + y - 1 = 0
theorem part1 (a b m : ℝ) : l1 a b → l2 a b → m = 4 → 3 * a + b + m = 0:= by
  sorry

-- Part 2: Prove the equations of line l based on distance from point (3,1)
theorem part2 (k x y : ℝ) : (abs (3 * k - 1 + 2 * k + 2) / (sqrt (k ^ 2 + 1)) = 5) ∨ (x = -2) → 
  (12 * x - 5 * y + 34 = 0) ∨ (x = -2) := by
  sorry

end part1_part2_l404_404031


namespace correct_number_of_values_l404_404859

def mean (S : ℝ) (n : ℕ) : ℝ := S / n

theorem correct_number_of_values (n : ℕ) (S : ℝ) 
  (h1 : mean (S - 10) n = 140)
  (h2 : mean S n = 140.33333333333334) :
  n = 30 := by
  sorry

end correct_number_of_values_l404_404859


namespace parallel_lines_l404_404337

def l1 (m : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), 2 * x + m * y = 0
def l2 : ℝ × ℝ → Prop := λ (x y : ℝ), y = 3 * x - 1

theorem parallel_lines (m : ℝ) (h : ∀ x y : ℝ, l1 m (x, y) ↔ l2 (x, y)) : 
  m = -2 / 3 := 
sorry

end parallel_lines_l404_404337


namespace fifteenth_triangular_number_is_120_l404_404248

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem fifteenth_triangular_number_is_120 : triangular_number 15 = 120 := by
  sorry

end fifteenth_triangular_number_is_120_l404_404248


namespace expected_value_of_sum_of_marbles_l404_404344

theorem expected_value_of_sum_of_marbles : 
  ∀ (marbles : Finset ℕ), 
  marbles = {1, 2, 3, 4, 5, 6, 7} → 
  (∑ m in (marbles.subsets.filter (λ s, s.card = 3)).val, ∑ x in s, x) / (marbles.subsets.filter (λ s, s.card = 3)).card = 12 :=
by
  intros marbles hmarbles
  sorry

end expected_value_of_sum_of_marbles_l404_404344


namespace curve_is_circle_l404_404650

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) : 
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = k^2) → 
    (r^2 = x^2 + y^2 ∧ ∃ (θ : ℝ), x/r = Real.cos θ ∧ y/r = Real.sin θ) :=
sorry

end curve_is_circle_l404_404650


namespace ironed_clothing_l404_404443

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l404_404443


namespace assembly_line_arrangements_l404_404894

def task_arrangement_count : ℕ :=
  6!

theorem assembly_line_arrangements : task_arrangement_count / 2 = 120 := by
  sorry

end assembly_line_arrangements_l404_404894


namespace retail_store_paid_40_percent_more_l404_404170

variables (C R : ℝ)

-- Condition: The customer price is 96% more than manufacturing cost
def customer_price_from_manufacturing (C : ℝ) : ℝ := 1.96 * C

-- Condition: The customer price is 40% more than the retailer price
def customer_price_from_retail (R : ℝ) : ℝ := 1.40 * R

-- Theorem to be proved
theorem retail_store_paid_40_percent_more (C R : ℝ) 
  (h_customer_price : customer_price_from_manufacturing C = customer_price_from_retail R) :
  (R - C) / C = 0.40 :=
by
  sorry

end retail_store_paid_40_percent_more_l404_404170


namespace no_rectangle_with_20_marked_cells_l404_404152

theorem no_rectangle_with_20_marked_cells (marked_cells : finset (ℕ × ℕ)) (h_marked : marked_cells.card = 40) :
  ¬ (∃ rect : set (ℕ × ℕ), rect.finite ∧ rect ⊆ marked_cells ∧ rect.card = 20) :=
by
  sorry

end no_rectangle_with_20_marked_cells_l404_404152


namespace choose_3_out_of_13_l404_404720

theorem choose_3_out_of_13: (Nat.choose 13 3) = 286 :=
by
  sorry

end choose_3_out_of_13_l404_404720


namespace complex_cosine_sum_l404_404229

theorem complex_cosine_sum :
  ∑ (n : ℕ) in finset.range 31, (complex.i ^ n) * real.cos (↑((30 + 120 * n)) * real.pi / 180) = (5 * real.sqrt 3 - 7 * complex.i * real.sqrt 3) := 
sorry

end complex_cosine_sum_l404_404229


namespace ricardo_coins_difference_l404_404826

theorem ricardo_coins_difference :
  ∃ (x y : ℕ), (x + y = 2020) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ ((5 * x + y) - (x + 5 * y) = 8072) :=
by
  sorry

end ricardo_coins_difference_l404_404826


namespace tangent_line_properties_l404_404733

noncomputable def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem tangent_line_properties (a b : ℝ) :
  (∀ x : ℝ, curve 0 a b = b) →
  (∀ x : ℝ, x - (curve x a b - b) + 1 = 0 → (∀ x : ℝ, 2*0 + a = 1)) →
  a + b = 2 :=
by
  intros h_curve h_tangent
  have h_b : b = 1 := by sorry
  have h_a : a = 1 := by sorry
  rw [h_a, h_b]
  norm_num

end tangent_line_properties_l404_404733


namespace modulus_of_z_l404_404018

theorem modulus_of_z (z : ℂ) 
  (h : (complex.sqrt 3 + complex.I) * z = 1 - complex.sqrt 3 * complex.I) :
  complex.abs z = 1 :=
sorry

end modulus_of_z_l404_404018


namespace number_of_customers_l404_404227

def gallons_of_milk_total : Nat := 12 * 4
def price_per_gallon : Float := 3
def milk_to_butter_ratio : Nat := 2
def price_per_butter_stick : Float := 1.5
def gallons_per_customer : Nat := 6
def total_revenue : Float := 144

theorem number_of_customers 
  (total_milk : Nat = gallons_of_milk_total) 
  (price_gallon : Float = price_per_gallon) 
  (butter_ratio : Nat = milk_to_butter_ratio)
  (price_butter : Float = price_per_butter_stick)
  (milk_demand_per_customer : Nat = gallons_per_customer)
  (revenue : Float = total_revenue) :
  total_milk / milk_demand_per_customer = 8 := by
  sorry

end number_of_customers_l404_404227


namespace part_i_part_ii_l404_404174

noncomputable def a_n (n : ℕ) : ℝ := 2^(n+2)
def b_n (n : ℕ) : ℝ := 1 / (n * Real.log2 (a_n n))
def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, (b_n (i+1))

theorem part_i (n : ℕ) : a_n n = 2^(n+2) :=
by sorry

theorem part_ii (n : ℕ) : S_n n = (3/4) - ((2*n+3) / (2 * (n+1) * (n+2))) :=
by sorry

end part_i_part_ii_l404_404174


namespace trailing_zeros_of_7_factorial_in_base_8_l404_404846

def trailing_zeros_in_base (n : ℕ) (b : ℕ) : ℕ :=
  if b ≤ 1 then 0
  else (List.range n.succ).count (λ k, b^k | n)

theorem trailing_zeros_of_7_factorial_in_base_8 :
  trailing_zeros_in_base 7! 8 = 1 :=
by
  sorry

end trailing_zeros_of_7_factorial_in_base_8_l404_404846


namespace min_t_of_even_function_l404_404462

noncomputable def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def f_translated (t x : ℝ) : ℝ :=
  2 * sin (2 * x + (2 * π / 3) + 2 * t)

theorem min_t_of_even_function :
  ∃ t > 0, isEvenFunction (f_translated t) ∧ t = 5 * π / 12 := 
sorry

end min_t_of_even_function_l404_404462


namespace john_took_more_chickens_l404_404069

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l404_404069


namespace parallel_AO_JN_l404_404492

-- Given conditions
variables {A B C O J : Point}
variables {w1 w2 : Circle}
variables {N : Point}
variables (ABC_triangle : triangle A B C)
variables (circle_w1 : circumscribed_circle ABC_triangle = w1)
variables (incircle_touches_bc : incircle_touch_point BC w1 N)
variables (circle_w2 : inscribed_circle_segment BAC w1 N = w2)
variables (center_of_w2 : center_of_circle w2 = O)
variables (excircle_J_bc : excircle_center_triangle BC ABC_triangle = J)

-- Statement to prove
theorem parallel_AO_JN :
  parallel (line A O) (line J N) :=
sorry

end parallel_AO_JN_l404_404492


namespace dancers_earnings_l404_404661

noncomputable def earnings_of_dancers (x : ℚ) := (x, x - 16, 2 * x - 40, 3 * x - 40)

theorem dancers_earnings :
  ∃ x : ℚ, 
  let earnings := earnings_of_dancers x in
  earnings.1 + earnings.2 + earnings.3 + earnings.4 = 280 ∧
  x = 53 + 5/7 ∧ earnings.2 = 37 + 5/7 ∧ earnings.3 = 67 + 3/7 ∧ earnings.4 = 121 + 1/7 :=
sorry

end dancers_earnings_l404_404661


namespace one_gt_one_others_lt_one_l404_404476

theorem one_gt_one_others_lt_one 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_prod : a * b * c = 1)
  (h_ineq : a + b + c > (1 / a) + (1 / b) + (1 / c)) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
sorry

end one_gt_one_others_lt_one_l404_404476


namespace ratio_of_ages_l404_404829

theorem ratio_of_ages (S_age M_diff : ℕ) (h1 : S_age = 70) (h2 : ∀ M_age : ℕ, S_age + M_diff = M_age) : (S_age : M_age : ℕ) = 7 : 9 := 
by 
  sorry

end ratio_of_ages_l404_404829


namespace sum_numbers_l404_404138

theorem sum_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end sum_numbers_l404_404138


namespace sum_of_reciprocals_le_one_l404_404769

open Nat Finset

variable (n m : ℕ)
variable (A : Fin m (Finset (Fin n)))
variable (h1 : ∀ i, (A i).card ≤ n / 2)
variable (h2 : ∀ i j, i ≠ j → ¬ (A i) ⊆ (A j))
variable (h3 : ∀ i j, i ≠ j → (A i) ∩ (A j) ≠ ∅)

theorem sum_of_reciprocals_le_one : 
  (Finset.sum (Finset.univ : Finset (Fin m)) (λ i, 1 / (n - 1).choose ((A i).card - 1))) ≤ 1 :=
sorry

end sum_of_reciprocals_le_one_l404_404769


namespace quadratic_completing_square_sum_l404_404643

theorem quadratic_completing_square_sum (q t : ℝ) :
    (∃ (x : ℝ), 9 * x^2 - 54 * x - 36 = 0 ∧ (x + q)^2 = t) →
    q + t = 10 := sorry

end quadratic_completing_square_sum_l404_404643


namespace last_digit_1993_2002_plus_1995_2002_l404_404843

theorem last_digit_1993_2002_plus_1995_2002 :
  (1993 ^ 2002 + 1995 ^ 2002) % 10 = 4 :=
by sorry

end last_digit_1993_2002_plus_1995_2002_l404_404843


namespace smallest_integer_in_ratio_l404_404117

theorem smallest_integer_in_ratio {a b c : ℕ} (h1 : a = 2 * b / 3) (h2 : c = 5 * b / 3) (h3 : a + b + c = 60) : b = 12 := 
  sorry

end smallest_integer_in_ratio_l404_404117


namespace mike_profit_l404_404798

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l404_404798


namespace find_a_l404_404842

theorem find_a (a b c : ℤ) (vertex_cond : ∀ x : ℝ, -a * (x - 1)^2 + 3 = -a * (x - 1)^2 + 3)
    (point_cond : (0, 1) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2 + b * p.1 + c}) :
    a = -2 := by 
sorry

end find_a_l404_404842


namespace walter_percent_of_dollar_l404_404517

theorem walter_percent_of_dollar
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (penny_value : Nat := 1)
  (nickel_value : Nat := 5)
  (dime_value : Nat := 10)
  (dollar_value : Nat := 100)
  (total_value := pennies * penny_value + nickels * nickel_value + dimes * dime_value) :
  pennies = 2 ∧ nickels = 3 ∧ dimes = 2 →
  (total_value * 100) / dollar_value = 37 :=
by
  sorry

end walter_percent_of_dollar_l404_404517


namespace number_of_players_taking_chemistry_l404_404224

theorem number_of_players_taking_chemistry 
    (total_players : ℕ) 
    (taking_physics : ℕ) 
    (taking_both : ℕ) 
    (nonempty_of_subset_add : 0 < total_players ≤ taking_physics + (total_players - taking_physics + taking_both)) 
    (taking_physics_and_chemistry : taking_both ≤ taking_physics) 
    (chemistry_players_ge_zero : 0 ≤ (total_players - taking_physics + taking_both) ∧ 0 ≤ (taking_physics - taking_both)):
    total_players = 15 → 
    taking_physics = 9 → 
    taking_both = 3 → 
    (taking_physics + (total_players - taking_physics + taking_both)) = 9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end number_of_players_taking_chemistry_l404_404224


namespace dice_probability_l404_404966

theorem dice_probability :
  let prob := 1 - (5/6) * (5/6)
  prob = 11 / 36 :=
by
  -- Introducing the probability calculation steps
  let p_no_five := ((5 : ℚ) / 6) * ((5 : ℚ) / 6)
  have p_no_five_val : p_no_five = 25 / 36 := by
    -- Calculation of probability that neither die rolls a 5
    sorry
  
  let p_at_least_one_five := 1 - p_no_five
  have p_val : p_at_least_one_five = 11 / 36 := by
    -- Calculation of the probability that at least one die rolls a 5
    sorry
  
  exact p_val

end dice_probability_l404_404966


namespace exactly_one_correct_l404_404690

def proposition_1 (a_n : ℕ → ℝ) (A : ℝ) : Prop :=
  (∀ ε > 0, ∃ N, ∀ n > N, |a_n n| < ε) → (∀ ε > 0, ∃ N, ∀ n > N, |a_n n - A| < ε)

def proposition_2 (a_n : ℕ → ℝ) (A : ℝ) : Prop :=
  (∀ ε > 0, ∃ N, ∀ n > N, |a_n n - A| < ε ∧ ∀ n, a_n n > 0) → A > 0

def proposition_3 (a_n : ℕ → ℝ) (A : ℝ) : Prop :=
  (∀ ε > 0, ∃ N, ∀ n > N, |a_n n - A| < ε) → (∀ ε > 0, ∃ N, ∀ n > N, |(a_n n)^2 - A^2| < ε)

def proposition_4 (a_n b_n : ℕ → ℝ) : Prop :=
  (∀ ε > 0, ∃ N, ∀ n > N, |a_n n - b_n n| < ε) → (∀ ε > 0, ∃ MA, ∀ n > MA, ∃ MB, ∀ n > MB, |a_n n - b_n n| < ε)

theorem exactly_one_correct :
  ∃ i ∈ {1, 2, 3, 4}, ∀ j ∈ {1, 2, 3, 4}, j ≠ i → (¬(proposition_1 a_n A) ∧ ¬(proposition_2 a_n A) ∧ proposition_3 a_n A ∧ ¬(proposition_4 a_n b_n)) :=
sorry

end exactly_one_correct_l404_404690


namespace expected_value_of_8_sided_die_l404_404133

/-- Define the range of the die -/
def sides : ℕ := 8

/-- Define the probability of each side occurring -/
def probability (n : ℕ) : ℚ := 1 / n

/-- Define the expected value calculation -/
def expected_value (n : ℕ) (vals : List ℕ) : ℚ :=
  (vals.map ((· : ℕ → ℚ).⟫ * probability n).sum / n

theorem expected_value_of_8_sided_die :
  expected_value sides [1, 2, 3, 4, 5, 6, 7, 8] = 4.5 := by
  sorry

end expected_value_of_8_sided_die_l404_404133


namespace max_product_xy_l404_404410

theorem max_product_xy (x y : ℕ) (h : 69 * x + 54 * y ≤ 2008) : xy ≤ 270 :=
begin
  sorry
end

end max_product_xy_l404_404410


namespace cyclist_speed_l404_404867

variable (circumference : ℝ) (v₂ : ℝ) (t : ℝ)

theorem cyclist_speed (h₀ : circumference = 180) (h₁ : v₂ = 8) (h₂ : t = 12)
  (h₃ : (7 * t + v₂ * t) = circumference) : 7 = 7 :=
by
  -- From given conditions, we derived that v₁ should be 7
  sorry

end cyclist_speed_l404_404867


namespace time_to_cross_platform_l404_404545

-- Definitions based on conditions
def train_length : ℝ := 450
def platform_length : ℝ := 850
def speed_kmh : ℝ := 80
def speed_ms : ℝ := speed_kmh * 1000 / 3600
def total_distance : ℝ := train_length + platform_length

-- Theorem statement
theorem time_to_cross_platform : total_distance / speed_ms = 58.5 := 
by
  -- Proof omitted
  sorry

end time_to_cross_platform_l404_404545


namespace half_black_half_green_to_all_blue_one_thousand_black_one_thousand_sixteen_green_not_all_blue_two_adjacent_black_not_to_one_green_l404_404813

-- Define the type of the beads
inductive Color
  | black | blue | green

open Color

-- Define the transformation rule
def transform : List Color → List Color
  | [] => []
  | [c] => [c]
  | c1::c2::cs => 
      let new_color c1 c2 :=
        match (c1, c2) with
        | (black, black) => black
        | (green, green) => green
        | (blue, blue) => blue
        | _ => blue
      in new_color c1 c2 :: transform (c2::cs)

-- a) Prove that half black and half green transforms to all blue
theorem half_black_half_green_to_all_blue : 
  ∃ lst : List Color, lst.length = 2016 ∧
  (∃ cnt_black cnt_green, cnt_black = 1008 ∧ cnt_green = 1008) ∧
  transform lst = List.repeat blue 2016 :=
by
  sorry

-- b) Prove that 1000 black and 1016 green cannot transform to all blue
theorem one_thousand_black_one_thousand_sixteen_green_not_all_blue : 
  ¬(∃ lst : List Color, lst.length = 2016 ∧
  (∃ cnt_black cnt_green, cnt_black = 1000 ∧ cnt_green = 1016) ∧
  transform lst = List.repeat blue 2016) :=
by
  sorry

-- c) Prove that two consecutive black beads rest blue cannot transform to one green bead rest blue
theorem two_adjacent_black_not_to_one_green :
  ¬(∃ lst : List Color, lst.length = 2016 ∧
  (lst.count black = 2 ∧ lst.count blue = 2014) ∧
  transform lst = List.repeat blue 2015 ++ [green]) :=
by
  sorry

end half_black_half_green_to_all_blue_one_thousand_black_one_thousand_sixteen_green_not_all_blue_two_adjacent_black_not_to_one_green_l404_404813


namespace value_of_a_l404_404698

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end value_of_a_l404_404698


namespace probability_a_b_c_is_divisible_by_4_l404_404016

open Probability

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def probability_divisible_by_4 : ℚ := 
  let set := (Finset.range 4032).map Nat.succ in
  if h0 : ¬ Finset.Nonempty set then 0 else
  let S := Finset (set.product (set.product set)) in
  let events := S.filter (λ ⟨a, ⟨b, c⟩⟩, is_divisible_by_4 (a * (b * c + b + 1))) in
  (events.card : ℚ) / (S.card : ℚ)

theorem probability_a_b_c_is_divisible_by_4 : 
  probability_divisible_by_4 = 3 / 8 :=
sorry

end probability_a_b_c_is_divisible_by_4_l404_404016


namespace alex_buns_needed_l404_404205

def packs_of_buns_needed (burgers_per_guest guests non_meat_eating_friend non_bread_eating_friend buns_per_pack : ℕ) : ℕ := 
  (burgers_per_guest * (guests - non_meat_eating_friend - (if non_eat_bread_eating_friend > 0 then 1 else 0)) - two_friends_with_different_needs) / buns_per_pack

theorem alex_buns_needed (h1 : burgers_per_guest = 3) (h2 : guests = 10) (h3 : non_meat_eating_friend = 1)
                       (h4 : non_bread_eating_friend = 1) (h5 : buns_per_pack = 8) :
  packs_of_buns_needed burgers_per_guest guests non_meat_eating_friend non_bread_eating_friend buns_per_pack = 3 := 
by
  sorry

end alex_buns_needed_l404_404205


namespace ball_speed_is_20_l404_404576

def ball_flight_time : ℝ := 8
def collie_speed : ℝ := 5
def collie_catch_time : ℝ := 32

noncomputable def collie_distance : ℝ := collie_speed * collie_catch_time

theorem ball_speed_is_20 :
  collie_distance = ball_flight_time * 20 :=
by
  sorry

end ball_speed_is_20_l404_404576


namespace term_transition_addition_l404_404046

theorem term_transition_addition (k : Nat) :
  (2:ℚ) / ((k + 1) * (k + 2)) = ((2:ℚ) / ((k * (k + 1))) - ((2:ℚ) / ((k + 1) * (k + 2)))) := 
sorry

end term_transition_addition_l404_404046


namespace probability_exactly_five_green_marbles_l404_404043

open BigOperators

noncomputable def probability_five_green_marbles (total_draws : ℕ) (green_marbles : ℕ) (purple_marbles : ℕ) (green_draws : ℕ) : ℝ :=
  let total_marbles := green_marbles + purple_marbles
  let p_green := (green_marbles : ℝ) / total_marbles
  let p_purple := (purple_marbles : ℝ) / total_marbles
  (nat.choose total_draws green_draws) * (p_green ^ green_draws) * (p_purple ^ (total_draws - green_draws))

theorem probability_exactly_five_green_marbles :
  probability_five_green_marbles 8 8 4 5 ≈ 0.273 :=
sorry

end probability_exactly_five_green_marbles_l404_404043


namespace area_of_triangle_l404_404326

noncomputable def calculate_area (a : ℝ) : ℝ :=
let f := λ x : ℝ, a * x^3 + 3 * x in
let f' := λ x : ℝ, 3 * a * x^2 + 3 in
let tangent_slope := f' 1 in
let tangent_line := λ x : ℝ, -6*x + 6 in
let intersection_y := tangent_line 0 in
let intersection_x := (tangent_line (6)) in
(1 / 2) * intersection_x * intersection_y

theorem area_of_triangle (a : ℝ) (h_tangent_perpendicular : 3 * a + 3 = -6) : calculate_area a = 3 := 
by { rw calculate_area, sorry }

end area_of_triangle_l404_404326


namespace area_of_triangle_eq_one_l404_404359

noncomputable def triangle_area {a c : ℝ} {B : Real.Angle}
  (a_eq : a = Real.sqrt 2)
  (c_eq : c = 2 * Real.sqrt 2)
  (B_eq : B = Real.pi / 6) : ℝ :=
  (1 / 2) * a * c * Real.sin B

theorem area_of_triangle_eq_one :
  ∀ (a c : ℝ) (B : Real.Angle), 
  a = Real.sqrt 2 →
  c = 2 * Real.sqrt 2 →
  B = Real.pi / 6 →
  triangle_area a c B = 1 := 
by
  intros a c B a_eq c_eq B_eq
  rw [triangle_area, a_eq, c_eq, B_eq]
  sorry

end area_of_triangle_eq_one_l404_404359


namespace circumcircles_common_point_l404_404672

theorem circumcircles_common_point
  (A B C D P Q : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P] [Inhabited Q]
  (AB BC CD PQ : Type)
  [LinearOrder AB] [LinearOrder BC] [LinearOrder CD]
  (parallelogram_ABCD : Parallelogram A B C D)
  (h1 : AB < BC)
  (h2 : P ∈ Line[BC])
  (h3 : Q ∈ Line[CD])
  (h4 : CP = CQ) :
  ∃ A', (A' ≠ A) ∧ (∀ P Q on BC CD, ∀ h4, A' ∈ circumcircle (triangle A P Q)) :=
sorry

end circumcircles_common_point_l404_404672


namespace range_of_function_l404_404883

def range_exclusion (x : ℝ) : Prop :=
  x ≠ 1

theorem range_of_function :
  set.range (λ x : ℝ, if x = -2 then (0 : ℝ) else x + 3) = {y : ℝ | range_exclusion y} :=
by 
  sorry

end range_of_function_l404_404883


namespace probability_exactly_one_girl_two_boys_l404_404551

-- Defining the probability mass functions for boy and girl as 0.5 each
noncomputable def p_boy : ℝ := 0.5
noncomputable def p_girl : ℝ := 0.5

-- The main theorem stating the problem
theorem probability_exactly_one_girl_two_boys :
  (p_boy = 0.5) → (p_girl = 0.5) →
  let P_1G2B := 3 * (0.5 ^ 3) in
  P_1G2B = 0.375 :=
by
  intros h_boy h_girl
  -- Definitions and calculations would usually go here
  sorry

end probability_exactly_one_girl_two_boys_l404_404551


namespace beth_guaranteed_win_l404_404570

def nim_value (bricks : ℕ) : ℕ :=
  match bricks with
  | 0   => 0
  | 1   => 1
  | 2   => 2
  | 3   => 3
  | 4   => 1
  | 5   => 4
  | 6   => 3
  | 7   => 5
  | _   => sorry -- This case will not apply as the conditions restrict scenarios up to 7 bricks.

def nim_sum (configs : List ℕ) : ℕ :=
  configs.foldr xor 0

def Beth_wins (walls : List ℕ) : Prop :=
  nim_sum (walls.map nim_value) = 0

theorem beth_guaranteed_win : Beth_wins [7, 3, 2] :=
by
  -- calculating the nim sum step by step
  have n7 := nim_value 7
  have n3 := nim_value 3
  have n2 := nim_value 2
  have s := nim_sum [n7, n3, n2]
  show s = 0
  sorry -- nim_value [7,3,2] => 0, completing the proof manually is a direct calculation

end beth_guaranteed_win_l404_404570


namespace expression_divisible_by_11_l404_404049

theorem expression_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^(n+2) + 3^n) % 11 = 0 := 
sorry

end expression_divisible_by_11_l404_404049


namespace total_weight_of_load_l404_404922

def weight_of_crate : ℕ := 4
def weight_of_carton : ℕ := 3
def number_of_crates : ℕ := 12
def number_of_cartons : ℕ := 16

theorem total_weight_of_load :
  number_of_crates * weight_of_crate + number_of_cartons * weight_of_carton = 96 :=
by sorry

end total_weight_of_load_l404_404922


namespace productivity_increase_l404_404853

theorem productivity_increase (a b : ℝ) : (7 / 8) * (1 + 20 / 100) = 1.05 :=
by
  sorry

end productivity_increase_l404_404853


namespace max_gold_coins_l404_404531

theorem max_gold_coins (n : ℕ) : n % 12 = 3 ∧ n < 120 → n ≤ 111 :=
by
  intro h
  obtain ⟨m, hm⟩ := Nat.exists_eq_add_of_lt h.2
  rw [hm] at h
  have : n % 12 = 3 := h.1
  have : ∃ k, n = 12 * k + 3 :=
    ⟨m, by rw [hm, add_assoc, mul_comm, ← add_assoc]; exact rfl⟩
  obtain ⟨k, rfl⟩ := this
  by_cases hk : k < 10
  · calc
      12 * k + 3 ≤ 12 * 9 + 3 := by linarith
      ... = 111 := by linarith
  · exfalso
    exact hk (not_lt.1 hk)

end max_gold_coins_l404_404531


namespace candy_box_price_l404_404928

theorem candy_box_price (c s : ℝ) 
  (h1 : 1.50 * s = 6) 
  (h2 : c + s = 16) 
  (h3 : ∀ c, 1.25 * c = 1.25 * 12) : 
  (1.25 * c = 15) :=
by
  sorry

end candy_box_price_l404_404928


namespace largest_three_digit_multiple_of_7_with_digit_sum_21_is_966_l404_404520

theorem largest_three_digit_multiple_of_7_with_digit_sum_21_is_966 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 7 = 0) ∧ (digit_sum n = 21) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 7 = 0) ∧ (digit_sum m = 21) → m ≤ n) :=
sorry

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

end largest_three_digit_multiple_of_7_with_digit_sum_21_is_966_l404_404520


namespace possible_n_values_l404_404687

def a (n : Nat) : Int := 2 * n - 1
def b (n : Nat) : Nat := 2 ^ (n - 1)
def c (n : Nat) : Int := a (b n)
def T (n : Nat) : Int := (∑ i in Finset.range n, c (i + 1))

theorem possible_n_values (n : Nat) (h : T n < 2023) : n = 8 ∨ n = 9 := sorry

end possible_n_values_l404_404687


namespace kayak_total_until_May_l404_404940

noncomputable def kayak_number (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 3 * kayak_number (n - 1)

theorem kayak_total_until_May : kayak_number 0 + kayak_number 1 + kayak_number 2 + kayak_number 3 = 200 := by
  sorry

end kayak_total_until_May_l404_404940


namespace area_of_remaining_figure_l404_404143
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end area_of_remaining_figure_l404_404143


namespace second_pipe_filling_time_l404_404815

theorem second_pipe_filling_time (T : ℝ) :
  (∃ T : ℝ, (1 / 8 + 1 / T = 1 / 4.8) ∧ T = 12) :=
by
  sorry

end second_pipe_filling_time_l404_404815


namespace range_of_m_l404_404321

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) / x

theorem range_of_m (a m : ℝ) (h₀ : a > 0) (h₁ : f a (m^2 + 1) > f a (m^2 - m + 3)) 
  : m > 2 :=
sorry

end range_of_m_l404_404321


namespace find_new_coords_l404_404182

-- Define the initial conditions for the given point
variables (ρ θ φ : ℝ)

-- Condition for spherical coordinates related to rectangular coordinates
def rectangular_coords (x y z : ℝ) : Prop :=
  x = ρ * sin φ * cos θ ∧ y = ρ * sin φ * sin θ ∧ z = ρ * cos φ

-- The given point has rectangular coordinates (3, -4, 12)
variables (h1 : rectangular_coords 3 (-4) 12)

-- Define the target condition for the new spherical coordinates
def new_rectangular_coords (x y z : ℝ) : Prop :=
  x = ρ * sin φ * cos (-θ) ∧ y = ρ * sin φ * sin (-θ) ∧ z = ρ * cos φ

-- The target point has rectangular coordinates (3, 4, 12)
theorem find_new_coords : new_rectangular_coords 3 4 12 :=
by {
  -- The proof would be inserted here
  sorry
}

end find_new_coords_l404_404182


namespace binomial_coefficient_12_10_l404_404606

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l404_404606


namespace harmonic_series_bound_l404_404822

theorem harmonic_series_bound (n : ℕ) (hn : n > 0) : 
  (∑ i in finset.range (2^n), (1 : ℝ)/(i+1)) > (n / 2 : ℝ) :=
sorry

end harmonic_series_bound_l404_404822


namespace find_y_x_relationship_maximize_profit_profit_of_3910_l404_404042

-- Conditions
def initial_price : ℝ := 60
def initial_volume : ℝ := 100
def cost_per_piece : ℝ := 30
def price_decrease_per_piece : ℝ := 1
def volume_increase_per_decrease : ℝ := 10

-- 1. Functional relationship between y and x
theorem find_y_x_relationship (x : ℝ) : 
  ∃ y : ℝ, y = -10 * x + 700 :=
by 
  use (-10 * x + 700)
  rfl

-- 2. Determine the selling price that maximizes weekly sales profit, and the maximum profit
theorem maximize_profit :
  let profit (x : ℝ) := (x - cost_per_piece) * (-10 * x + 700) in
  ∃ max_profit_price max_profit, max_profit_price = 50 ∧ max_profit = 4000 :=
by 
  let profit := λ x : ℝ, (x - cost_per_piece) * (-10 * x + 700)
  use 50, 4000
  sorry

-- 3. Selling price that results in a weekly profit of 3910 yuan
theorem profit_of_3910 :
  let profit (x : ℝ) := (x - cost_per_piece) * (-10 * x + 700) in
  ∃ x1 x2, profit x1 = 3910 ∧ profit x2 = 3910 ∧ ((x1 = 47) ∧ (x2 = 53) ∨ (x1 = 53) ∧ (x2 = 47)) :=
by 
  let profit := λ x : ℝ, (x - cost_per_piece) * (-10 * x + 700)
  use 47, 53
  sorry

end find_y_x_relationship_maximize_profit_profit_of_3910_l404_404042


namespace partition_into_57_groups_l404_404084

def Dreamland := Fin 2016

def flights (G : Dreamland → Dreamland) : Prop :=
  ∀ v : Dreamland, ∃! w : Dreamland, G v = w

theorem partition_into_57_groups (G : Dreamland → Dreamland) (h : flights G) :
  ∃ (groups : Fin 57 → set Dreamland),
  (∀ i : Fin 57, ∀ u v ∈ groups i, ∀ n ≤ 28, (G^[n]) u ≠ v) :=
sorry

end partition_into_57_groups_l404_404084


namespace cylinder_original_radius_l404_404641

theorem cylinder_original_radius
    (r h: ℝ)
    (h₀: h = 4)
    (h₁: π * (r + 8)^2 * 4 = π * r^2 * 12) :
    r = 12 :=
by
  -- Insert your proof here
  sorry

end cylinder_original_radius_l404_404641


namespace max_height_reached_by_projectile_l404_404186

noncomputable def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached_by_projectile : ∃ t : ℝ, height t = 161 :=
by
  use 2.5
  sorry

end max_height_reached_by_projectile_l404_404186


namespace rabbit_carrots_l404_404366

theorem rabbit_carrots (h_r h_f x : ℕ) (H1 : 5 * h_r = x) (H2 : 6 * h_f = x) (H3 : h_r = h_f + 2) : x = 60 :=
by
  sorry

end rabbit_carrots_l404_404366


namespace general_term_correct_S_maximum_value_l404_404331

noncomputable def general_term (n : ℕ) : ℤ :=
  if n = 1 then -1 + 24 else (-n^2 + 24 * n) - (-(n - 1)^2 + 24 * (n - 1))

noncomputable def S (n : ℕ) : ℤ :=
  -n^2 + 24 * n

theorem general_term_correct (n : ℕ) (h : 1 ≤ n) : general_term n = -2 * n + 25 := by
  sorry

theorem S_maximum_value : ∃ n : ℕ, S n = 144 ∧ ∀ m : ℕ, S m ≤ 144 := by
  existsi 12
  sorry

end general_term_correct_S_maximum_value_l404_404331


namespace password_decryption_prob_l404_404511

theorem password_decryption_prob :
  let p1 : ℚ := 1 / 5
  let p2 : ℚ := 1 / 4
  let q1 : ℚ := 1 - p1
  let q2 : ℚ := 1 - p2
  (1 - (q1 * q2)) = 2 / 5 :=
by
  sorry

end password_decryption_prob_l404_404511


namespace number_of_true_propositions_l404_404093

def proposition_1 (p q : Prop) : Prop := (¬p ∧ (p ∨ q)) → q

def proposition_2 (x : ℝ) (k : ℤ) : Prop := (x ≠ k * Real.pi) → (Real.sin x + 1 / (Real.sin x) ≥ 2)

def proposition_3 : Prop := ∃ x₀ : ℝ, Real.log (x₀^2 + 1) < 0

def area (a b : ℝ) :=  ∫ t in a..b, t - 1 / t

def proposition_4 : Prop := 2 * (area 1 2) = 3 - 2 * Real.log 2

theorem number_of_true_propositions :
  (proposition_1 p q = true) ∧ 
  (proposition_2 x k = false) ∧ 
  (proposition_3 = false) ∧ 
  (proposition_4 = false) →
  1 = 1 :=
sorry

end number_of_true_propositions_l404_404093


namespace total_distance_travelled_downstream_l404_404488

def speed_boat_still_water : ℝ := 20
def speed_first_current : ℝ := 5
def time_first_current_minutes : ℝ := 12
def speed_second_current : ℝ := 2
def time_second_current_minutes : ℝ := 15

def time_first_current_hours : ℝ := time_first_current_minutes / 60
def time_second_current_hours : ℝ := time_second_current_minutes / 60

def effective_speed_first_current : ℝ := speed_boat_still_water + speed_first_current
def effective_speed_second_current : ℝ := speed_boat_still_water + speed_second_current

def distance_first_current : ℝ := effective_speed_first_current * time_first_current_hours
def distance_second_current : ℝ := effective_speed_second_current * time_second_current_hours

def total_distance_downstream : ℝ := distance_first_current + distance_second_current

theorem total_distance_travelled_downstream : total_distance_downstream = 10.5 := by
  sorry

end total_distance_travelled_downstream_l404_404488


namespace definite_integral_x_cubed_l404_404108

theorem definite_integral_x_cubed :
  ∫ x in -1..1, x^3 = 0 :=
by
sorry

end definite_integral_x_cubed_l404_404108


namespace find_q_x_l404_404987

noncomputable def q (x : ℝ) : ℝ := (8 / 5) * (x - 1) * (x + 1) * (x + 3)

theorem find_q_x :
  (q(1) = 0) ∧ (q(-1) = 0) ∧ (q(-3) = 0) ∧ (q(2) = 24) :=
by
  -- Proving each individual condition separately
  have h1 : q 1 = 0 := by
    unfold q
    rw [sub_self, mul_zero, mul_zero, mul_zero, mul_zero]
    norm_num
  have h2 : q (-1) = 0 := by
    unfold q
    norm_num
    rw [mul_neg_eq_neg_mul_symm, add_comm (-1 : ℝ), add_assoc (-1) 1, add_right_neg, 
        mul_zero, mul_zero, mul_zero]
  have h3 : q (-3) = 0 := by
    unfold q
    norm_num
    rw [mul_comm, add_comm (-3 : ℝ), add_assoc (-3) 1, add_right_neg, 
        mul_zero, mul_zero, mul_zero]
  have h4 : q 2 = 24 := by
    unfold q
    norm_num
    rw [mul_comm, norm_num]
    norm_num
    exact (ring_hom.map_mul _ _ _ ).symm
  exact ⟨h1, h2, h3, h4⟩

end find_q_x_l404_404987


namespace find_circle_equation_l404_404669

def dist_point_line (x₀ y₀ : ℝ) (a b c : ℝ) : ℝ := 
  |a * x₀ + b * y₀ + c| / (Real.sqrt (a ^ 2 + b ^ 2))

def circle_eq (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem find_circle_equation :
  let center := (1 : ℝ, -1 : ℝ)
  let tangent_line := (4 : ℝ, -3 : ℝ, 3 : ℝ)
  let radius := dist_point_line 1 (-1) 4 (-3) 3
  circle_eq 1 (-1) radius x y = circle_eq 1 (-1) 2 x y :=
by
  sorry

end find_circle_equation_l404_404669
