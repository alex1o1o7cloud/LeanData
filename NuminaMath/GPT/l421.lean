import Mathbin.Combinatorics.Basic
import Mathlib
import Mathlib.
import Mathlib.Algebra.Complex.Basic
import Mathlib.Algebra.GcdMonoid.Finset
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Graph.Degree
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Polynomial.Eval
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclid
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Manifold.
import Mathlib.Init.Algebra.Order
import Mathlib.Probability.Distributions.Normal
import Mathlib.RingTheory.Ideal.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Pigeonhole
import Mathlib.Topology.Basic

namespace product_of_two_numbers_l421_421055

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l421_421055


namespace part_one_part_two_l421_421783

open Real Nat

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem part_one (n : ℕ) : n * sqrt 2 - floor (n * sqrt 2) > 1 / (2 * n * sqrt 2) :=
by
  sorry

theorem part_two (ε : ℝ) (hε : ε > 0) : ∃ n : ℕ, n * sqrt 2 - floor (n * sqrt 2) < 1 / (2 * n * sqrt 2) + ε :=
by
  sorry

end part_one_part_two_l421_421783


namespace a4_value_l421_421619

noncomputable def sum_of_terms (n : ℕ) : ℚ :=
  (n + 1) / (n + 2)

theorem a4_value :
  let a4 := sum_of_terms 4 - sum_of_terms 3 in
  a4 = 1 / 30 :=
by
  sorry

end a4_value_l421_421619


namespace partition_nat_no_infinite_geo_seq_l421_421704

/-- A partition of the natural numbers into two sets such that neither contains an infinite geometric sequence. -/
theorem partition_nat_no_infinite_geo_seq :
  ∃ (A B : set ℕ), (∀ x, x ∈ A ∨ x ∈ B) ∧ (∀ x, x ∉ A ∨ x ∉ B) ∧
  (∀ (q : ℕ), q > 1 → ¬∃ (m : ℕ), ∀ k, m * q^k ∈ A) ∧
  (∀ (q : ℕ), q > 1 → ¬∃ (m : ℕ), ∀ k, m * q^k ∈ B) := by
  sorry

end partition_nat_no_infinite_geo_seq_l421_421704


namespace area_of_triangle_l421_421400

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l421_421400


namespace tom_strokes_over_par_l421_421088

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l421_421088


namespace area_of_triangle_l421_421720

noncomputable def hyperbola_focus (F1 F2 P : ℝ × ℝ) : Prop :=
  let a := 3
  let c := 5
  let e := c / a
  let cosθ := real.cos (real.pi / 3)
  F1 = (-c, 0) ∧ F2 = (c, 0) ∧
  (P.1^2 / (a^2) - P.2^2 / (16) = 1) ∧
  (real.angle (P, F1) (P, F2) = 60)

noncomputable def triangle_area (F1 F2 P : ℝ × ℝ) : ℝ :=
  let PF1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
  let PF2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  0.5 * PF1 * PF2 * real.sin (real.pi / 3)

theorem area_of_triangle (F1 F2 P : ℝ × ℝ) :
  hyperbola_focus F1 F2 P → triangle_area F1 F2 P = 16 * real.sqrt 3 :=
by
  intros h
  sorry

end area_of_triangle_l421_421720


namespace functions_satisfy_condition_l421_421697

theorem functions_satisfy_condition :
  (∀ x1 x2 : ℝ, x1 > x2 → (∃ (f : ℝ → ℝ), 
  (f = (λ x, (1/2)^x) ∨ f = (λ x, -Real.sqrt x)) ∧ 
    (f x1 - f x2) / (x1 - x2) < 2)) := 
sorry

end functions_satisfy_condition_l421_421697


namespace area_of_voice_is_correct_l421_421166

-- Define the conditions
def side_length_of_square : ℝ := 25
def radius_of_voice_range : ℝ := 140

-- Define the area calculation for the voice range given as a circle
noncomputable def area_of_voice_range : ℝ := Real.pi * (radius_of_voice_range ^ 2)

-- The theorem statement
theorem area_of_voice_is_correct : area_of_voice_range ≈ 61575.44 := 
by
  sorry

end area_of_voice_is_correct_l421_421166


namespace sum_of_fractions_l421_421980

theorem sum_of_fractions :
  (finset.sum (finset.range 3010) (λ n, 3 / (n + 1)*(n + 1 + 3))) ≈ 1.833 :=
sorry

end sum_of_fractions_l421_421980


namespace remainder_thirty_l421_421295

theorem remainder_thirty (k : ℤ) : 
    let n := 30 * k - 1 in
    (n^2 + 2 * n + n^3 + 3) % 30 = 1 := by
    sorry

end remainder_thirty_l421_421295


namespace product_of_three_numbers_l421_421074

theorem product_of_three_numbers (a b c : ℝ) 
  (h₁ : a + b + c = 45)
  (h₂ : a = 2 * (b + c))
  (h₃ : c = 4 * b) : 
  a * b * c = 1080 := 
sorry

end product_of_three_numbers_l421_421074


namespace number_of_girls_l421_421866

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l421_421866


namespace arrived_new_stock_l421_421036

variable (initial_stock : ℕ) (sold_fish : ℕ) (spoiled_fraction : ℚ) (final_stock : ℕ)
variable (remaining_fish_after_sale : ℕ := initial_stock - sold_fish)
variable (spoiled_fish : ℕ := (remaining_fish_after_sale * spoiled_fraction).toNat)
variable (good_fish_left : ℕ := remaining_fish_after_sale - spoiled_fish)
variable (new_stock : ℕ := final_stock - good_fish_left)

theorem arrived_new_stock :
  initial_stock = 200 →
  sold_fish = 50 →
  spoiled_fraction = 1 / 3 →
  final_stock = 300 →
  new_stock = 200 :=
by
  intros h_initial h_sold h_spoiled_fraction h_final
  rw [h_initial, h_sold, h_spoiled_fraction, h_final]
  -- Automated steps to handle rational arithmetic is omitted for clarity
  sorry

end arrived_new_stock_l421_421036


namespace find_cost_of_potting_soil_l421_421171

-- Conditions
def cost_of_seeds : ℝ := 2.00
def number_of_plants : ℝ := 20.00
def sale_price_per_plant : ℝ := 5.00
def net_profit : ℝ := 90.00

-- Total revenue from selling basil plants
def total_revenue : ℝ := number_of_plants * sale_price_per_plant

-- Define the cost of potting soil based on the given conditions
def cost_of_potting_soil : ℝ := cost_of_potting_soil

theorem find_cost_of_potting_soil : cost_of_potting_soil = 8.00 :=
by
  have h1 : total_revenue = 100.00 := by norm_num [total_revenue, number_of_plants, sale_price_per_plant]
  have h2 : net_profit = total_revenue - (cost_of_seeds + cost_of_potting_soil) := by norm_num [net_profit, total_revenue, cost_of_seeds]
  have h3 : net_profit = 100.00 - 2.00 - cost_of_potting_soil := by norm_num [h2, h1]
  have h4 : cost_of_potting_soil = 8.00 := by norm_num [h3, net_profit]
  exact h4

end find_cost_of_potting_soil_l421_421171


namespace required_principal_l421_421244

theorem required_principal (i : ℝ) (h : i ≠ 0) : 
  let x := 1 / (1 + i) in
  let P := ∑ n in (nat.filter (λ n, n > 0)), n^2 / (1 + i)^n in
  P = (1 + i) * (2 + i) / i^3 := 
by 
  sorry

end required_principal_l421_421244


namespace find_y_l421_421484

theorem find_y 
  (x y : ℕ) 
  (hx : x % y = 9) 
  (hxy : (x : ℝ) / y = 96.12) : y = 75 :=
sorry

end find_y_l421_421484


namespace simplify_and_evaluate_l421_421386

-- Define the expression
def expr (x : ℝ) : ℝ := x^2 * (x + 1) - x * (x^2 - x + 1)

-- The main theorem stating the equivalence
theorem simplify_and_evaluate (x : ℝ) (h : x = 5) : expr x = 45 :=
by {
  sorry
}

end simplify_and_evaluate_l421_421386


namespace single_intersection_not_necessarily_tangent_l421_421551

structure Hyperbola where
  -- Placeholder for hyperbola properties
  axis1 : Real
  axis2 : Real

def is_tangent (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for tangency
  ∃ p : Real × Real, l = { p }

def is_parallel_to_asymptote (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for parallelism to asymptote 
  ∃ A : Real, l = { (x, A * x) | x : Real }

theorem single_intersection_not_necessarily_tangent
  (l : Set (Real × Real)) (H : Hyperbola) (h : ∃ p : Real × Real, l = { p }) :
  ¬ is_tangent l H ∨ is_parallel_to_asymptote l H :=
sorry

end single_intersection_not_necessarily_tangent_l421_421551


namespace marek_score_difference_l421_421076

variable (n : ℕ := 17)
variable (s : Fin n.succ → ℝ)
variable (M : ℝ)

-- Conditions
axiom all_students_took_test : ∀ i, 0 ≤ s i
axiom others_mean : M = (∑ i, (s i) - s ⟨0, by linarith⟩) / 16 + 17

-- To prove
theorem marek_score_difference :
  let total_score := (∑ i, s i) in
  let class_mean := total_score / n in
  M - class_mean = 16 :=
by
  sorry

end marek_score_difference_l421_421076


namespace product_of_two_numbers_l421_421045

-- Definitions and conditions
variables {x y : ℤ}
def condition1 := x - y = 7
def condition2 := x^2 + y^2 = 85

-- Theorem statement
theorem product_of_two_numbers : condition1 → condition2 → x * y = 18 :=
by
  intros h1 h2
  sorry

end product_of_two_numbers_l421_421045


namespace translation_of_graph_right_l421_421423

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem translation_of_graph_right (f : ℝ → ℝ) (x : ℝ) :
  f(-(x - 1)) = f(-x + 1) ∧ f(-(x - 1)) ≠ f(-x - 1) :=
by
  split
  -- Part 1: Proof that f(-(x - 1)) = f(-x + 1)
  -- This should be straightforward as it follows from the arithmetic on the inside of the function.
  sorry,
  -- Part 2: Proof that f(-(x - 1)) ≠ f(-x - 1) 
  -- This shoud also follow from the arithmetic on the inside of the function.
  sorry

end translation_of_graph_right_l421_421423


namespace polynomial_divisible_l421_421065

theorem polynomial_divisible (A B : ℝ) (h : ∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by
  sorry

end polynomial_divisible_l421_421065


namespace length_of_Q_l421_421450

theorem length_of_Q'R'_is_correct (P Q R : Point)
  (s : ℝ) (h1 : 2 < s ∧ s < 3)
  (h2 : dist P Q = 3 ∧ dist Q R = 3 ∧ dist P R = 3)
  (Q' R' : Point)
  (h3 : Q' ∈ circle_intersection P R ∧ Q' ∉ circle Q)
  (h4 : R' ∈ circle_intersection P Q ∧ R' ∉ circle R) :
  dist Q' R' = 1.5 + sqrt (6 * (s^2 - 2.25)) :=
sorry

end length_of_Q_l421_421450


namespace inclination_angle_range_l421_421187

theorem inclination_angle_range (α θ : ℝ) (sin_range : -1 ≤ Real.sin α ∧ Real.sin α ≤ 1) :
  (θ ∈ set.Icc 0 (Real.pi / 4) ∨ θ ∈ set.Icc (3 * Real.pi / 4) Real.pi) ↔
  ∃ (k : ℝ), k = -Real.sin α ∧ k = Real.tan θ := sorry

end inclination_angle_range_l421_421187


namespace determine_a_value_l421_421744

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem determine_a_value (a : ℝ) : 
  A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 :=
by
  intro h
  sorry

end determine_a_value_l421_421744


namespace neg_P_equiv_correct_answer_l421_421283

-- Define the proposition P
def P : Prop := ∃ x : ℝ, exp x - x - 1 ≤ 0

-- State the theorem that the negation of P is equivalent to the correct answer
theorem neg_P_equiv_correct_answer : ¬P ↔ ∀ x : ℝ, exp x - x - 1 > 0 :=
by sorry

end neg_P_equiv_correct_answer_l421_421283


namespace polynomial_roots_l421_421204

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end polynomial_roots_l421_421204


namespace strokes_over_par_l421_421092

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l421_421092


namespace sewer_runoff_capacity_l421_421440

theorem sewer_runoff_capacity (gallons_per_hour : ℕ) (hours_per_day : ℕ) (days_till_overflow : ℕ)
  (h1 : gallons_per_hour = 1000)
  (h2 : hours_per_day = 24)
  (h3 : days_till_overflow = 10) :
  gallons_per_hour * hours_per_day * days_till_overflow = 240000 := 
by
  -- We'll use sorry here as the placeholder for the actual proof steps
  sorry

end sewer_runoff_capacity_l421_421440


namespace calculate_ratio_milk_l421_421754

def ratio_milk_saturdays_weekdays (S : ℕ) : Prop :=
  let Weekdays := 15 -- total milk on weekdays
  let Sundays := 9 -- total milk on Sundays
  S + Weekdays + Sundays = 30 → S / Weekdays = 2 / 5

theorem calculate_ratio_milk : ratio_milk_saturdays_weekdays 6 :=
by
  unfold ratio_milk_saturdays_weekdays
  intros
  apply sorry -- Proof goes here

end calculate_ratio_milk_l421_421754


namespace measure_of_angle_A_possibilities_l421_421816

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l421_421816


namespace sequence_sum_eq_seven_l421_421958

noncomputable def b : ℕ → ℚ
| 1     := 2
| 2     := 3
| (k+3) := (1 / 2) * b (k + 2) + (1 / 3) * b (k + 1)

theorem sequence_sum_eq_seven : (∑' n, b n) = 7 := 
sorry

end sequence_sum_eq_seven_l421_421958


namespace tom_strokes_over_par_l421_421086

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l421_421086


namespace triangle_equation_no_real_roots_l421_421629

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h1: 0 < a) 
  (h2: 0 < b) 
  (h3: 0 < c) 
  (h4: a + b > c) 
  (h5: a + c > b) 
  (h6: b + c > a) : 
  (∆ < 0) := 
by
  let ∆ := (c^2 - a^2 - b^2)^2 - 4 * a^2 * b^2 
  sorry

end triangle_equation_no_real_roots_l421_421629


namespace smallest_n_satisfying_condition_l421_421375

def within_unit_interval (x y z : ℝ) (n : ℕ) :=
  (0 ≤ x ∧ x ≤ n) ∧ (0 ≤ y ∧ y ≤ n) ∧ (0 ≤ z ∧ z ≤ n)

def no_two_within_1_unit (x y z : ℝ) :=
  abs(x - y) ≥ 1 ∧ abs(y - z) ≥ 1 ∧ abs(z - x) ≥ 1

def at_least_one_pair_more_than_2_units (x y z : ℝ) :=
  abs(x - y) > 2 ∨ abs(y - z) > 2 ∨ abs(z - x) > 2

def required_probability_condition (n : ℕ) : Prop :=
  ∃ x y z : ℝ, within_unit_interval x y z n ∧ no_two_within_1_unit x y z ∧ at_least_one_pair_more_than_2_units x y z

theorem smallest_n_satisfying_condition : ∃ n : ℕ, n = 12 ∧ required_probability_condition n ∧ 
  (∀ k : ℕ, k < 12 → ¬ (required_probability_condition k)) :=
begin
  sorry
end

end smallest_n_satisfying_condition_l421_421375


namespace james_artifact_time_l421_421336

theorem james_artifact_time:
  let first_research := 6 in
  let first_expedition := 24 in
  let second_research := 3 * first_research in
  let second_expedition := 3 * first_expedition in
  (first_research + first_expedition) + (second_research + second_expedition) = 120 :=
by
  let first_research := 6
  let first_expedition := 24
  let second_research := 3 * first_research
  let second_expedition := 3 * first_expedition
  -- simplified as follows:
  have total_first := first_research + first_expedition
  have total_second := second_research + second_expedition
  -- statement = total_first + total_second = 120
  rw [←total_first, ←total_second]
  -- values substituted
  exact calc
    (6 + 24) + (18 + 72) = 30 + 90     : by rfl
                     ...       = 120  : by rfl
  sorry  -- note: proving this part is left as "sorry"

end james_artifact_time_l421_421336


namespace mike_falls_short_l421_421362

theorem mike_falls_short : 
  ∀ (max_marks mike_score : ℕ) (pass_percentage : ℚ),
  pass_percentage = 0.30 → 
  max_marks = 800 → 
  mike_score = 212 → 
  (pass_percentage * max_marks - mike_score) = 28 :=
by
  intros max_marks mike_score pass_percentage h1 h2 h3
  sorry

end mike_falls_short_l421_421362


namespace tan_alpha_plus_pi_l421_421721

-- Define the given conditions and prove the desired equality.
theorem tan_alpha_plus_pi 
  (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (π - α) = 1 / 3) : 
  Real.tan (α + π) = -2 * Real.sqrt 2 :=
by
  sorry

end tan_alpha_plus_pi_l421_421721


namespace arithmetic_sequence_formula_geometric_sequence_sum_formula_l421_421245

noncomputable def arithmetic_sequence_a_n (n : ℕ) : ℤ :=
  sorry

noncomputable def geometric_sequence_T_n (n : ℕ) : ℤ :=
  sorry

theorem arithmetic_sequence_formula :
  (∃ a₃ : ℤ, a₃ = 5) ∧ (∃ S₃ : ℤ, S₃ = 9) →
  -- Suppose we have an arithmetic sequence $a_n$
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence_a_n n = 2 * n - 1) := 
sorry

theorem geometric_sequence_sum_formula :
  (∃ q : ℤ, q > 0 ∧ q = 3) ∧ (∃ b₃ : ℤ, b₃ = 9) ∧ (∃ T₃ : ℤ, T₃ = 13) →
  -- Suppose we have a geometric sequence $b_n$ where $b_3 = a_5$
  (∀ n : ℕ, n ≥ 1 → geometric_sequence_T_n n = (3 ^ n - 1) / 2) := 
sorry

end arithmetic_sequence_formula_geometric_sequence_sum_formula_l421_421245


namespace oliver_shirts_problem_l421_421133

-- Defining the quantities of short sleeve shirts, long sleeve shirts, and washed shirts.
def shortSleeveShirts := 39
def longSleeveShirts  := 47
def shirtsWashed := 20

-- Stating the problem formally.
theorem oliver_shirts_problem :
  shortSleeveShirts + longSleeveShirts - shirtsWashed = 66 :=
by
  -- Proof goes here.
  sorry

end oliver_shirts_problem_l421_421133


namespace number_of_girls_l421_421857

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l421_421857


namespace value_before_decrease_l421_421774

theorem value_before_decrease
  (current_value decrease : ℤ)
  (current_value_equals : current_value = 1460)
  (decrease_equals : decrease = 12) :
  current_value + decrease = 1472 :=
by
  -- We assume the proof to follow here.
  sorry

end value_before_decrease_l421_421774


namespace locus_orthocenter_symmetric_circle_l421_421004

theorem locus_orthocenter_symmetric_circle (A B : Point) (circle_center : Point) (circle_radius : ℝ) :
  ∃ (SymCircle : Circle), ∀ C : Point,
    (OnCircle C circle_center circle_radius) →
    (SymmetricCircle SymCircle circle_center circle_radius A B ∧
    C ≠ A ∧ C ≠ B → OnCircle (orthocenter A B C) SymCircle.center SymCircle.radius ∧
    orthocenter A B C ≠ A ∧ orthocenter A B C ≠ B) := 
sorry

end locus_orthocenter_symmetric_circle_l421_421004


namespace math_problem_l421_421257

noncomputable def find_min_value (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2)
  (h_b : -a^2 / 2 + 3 * Real.log a = -1 / 2) : ℝ :=
  (3 * Real.sqrt 5 / 5) ^ 2

theorem math_problem (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2) :
  ∃ b : ℝ, b = -a^2 / 2 + 3 * Real.log a →
  (a - m) ^ 2 + (b - n) ^ 2 = 9 / 5 :=
by
  sorry

end math_problem_l421_421257


namespace find_f_3_l421_421617

def f (x : ℝ) : ℝ := x + 3  -- define the function as per the condition

theorem find_f_3 : f (3) = 7 := by
  sorry

end find_f_3_l421_421617


namespace find_k_l421_421654

theorem find_k (θ : ℝ) (h₁ : tan θ = 3) (h₂ : k = (3 * sin θ + 5 * cos θ) / (2 * sin θ + cos θ)) : k = 2 :=
sorry

end find_k_l421_421654


namespace total_cards_beginning_l421_421377

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l421_421377


namespace measure_of_angle_A_possibilities_l421_421817

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l421_421817


namespace number_of_girls_l421_421859

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l421_421859


namespace volume_of_PQRS_is_32_l421_421992

noncomputable def volume_of_tetrahedron (P Q R S : ℝ × ℝ × ℝ)
  (hPQ : dist P Q = 4)
  (hPR : dist P R = 6)
  (hPS : dist P S = 8)
  (hQR : dist Q R = sqrt 52)
  (hQS : dist Q S = 3 * sqrt 13)
  (hRS : dist R S = 10) : ℝ :=
  sorry

theorem volume_of_PQRS_is_32 
  (P Q R S : ℝ × ℝ × ℝ)
  (hPQ : dist P Q = 4)
  (hPR : dist P R = 6)
  (hPS : dist P S = 8)
  (hQR : dist Q R = sqrt 52)
  (hQS : dist Q S = 3 * sqrt 13)
  (hRS : dist R S = 10) : 
  volume_of_tetrahedron P Q R S hPQ hPR hPS hQR hQS hRS = 32 :=
  sorry

end volume_of_PQRS_is_32_l421_421992


namespace find_a_l421_421879

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end find_a_l421_421879


namespace reflect_y_axis_P_l421_421325

-- Define the point P with coordinates (3,5)
def P := (3, 5)

-- Define the reflection function across the y-axis
def reflect_y_axis (point : ℤ × ℤ) : ℤ × ℤ :=
  (-point.fst, point.snd)

-- Define the expected reflected point
def reflected_P := (-3, 5)

-- The theorem to prove that reflecting P across the y-axis gives the correct coordinates
theorem reflect_y_axis_P :
  reflect_y_axis P = reflected_P :=
by
  sorry

end reflect_y_axis_P_l421_421325


namespace steve_took_4_berries_l421_421790

theorem steve_took_4_berries (s t : ℕ) (H1 : s = 32) (H2 : t = 21) (H3 : s - 7 = t + x) :
  x = 4 :=
by
  sorry

end steve_took_4_berries_l421_421790


namespace minimum_cells_required_l421_421675

variable (n : ℕ)
variable (table : ℕ → ℕ → ℕ)
variable (H_table : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)

theorem minimum_cells_required : (∃ (cells : fin n → fin n), 
  (∀ i j, cells i = cells j → i = j) ∧
  (∀ i, 1 ≤ table (cells i).val (cells i).val ∧ table (cells i).val (cells i).val ≤ 9) ∧
  (∀ number : ℕ, (∀ i : fin n, number = (table (cells i).val (cells i).val)) 
    → (∀ j : fin n, (∀ k : fin n, table j k = number) ∨ (∀ k : fin n, table k j = number) → false))) := 
begin
  sorry
end

end minimum_cells_required_l421_421675


namespace calculation_l421_421212

def star_operation (m n : ℝ) : ℝ :=
  if m > n then (Real.sqrt m - Real.sqrt n) else (Real.sqrt m + Real.sqrt n)

theorem calculation : (star_operation 3 2) * (star_operation 8 12) = 2 := by
  sorry

end calculation_l421_421212


namespace price_difference_l421_421357

-- Define the prices of commodity X and Y in the year 2001 + n.
def P_X (n : ℕ) (a : ℝ) : ℝ := 4.20 + 0.45 * n + a * n
def P_Y (n : ℕ) (b : ℝ) : ℝ := 6.30 + 0.20 * n + b * n

-- Define the main theorem to prove
theorem price_difference (n : ℕ) (a b : ℝ) :
  P_X n a = P_Y n b + 0.65 ↔ (0.25 + a - b) * n = 2.75 :=
by
  sorry

end price_difference_l421_421357


namespace solve_for_A_plus_B_l421_421728

-- Define f and g as given
def f (A B x : ℝ) := A * x + B + 1
def g (A B x : ℝ) := B * x + A - 1

-- Use a broader scope of import for necessary libraries.
noncomputable theory

-- The main theorem according to our problem statement
theorem solve_for_A_plus_B (A B : ℝ) (h₀ : A ≠ -B) (h₁ : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = A - 2 * B) :
  A + B = 2 * A :=
sorry

end solve_for_A_plus_B_l421_421728


namespace binom_15_4_l421_421986

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l421_421986


namespace ratio_of_areas_of_concentric_circles_l421_421454

theorem ratio_of_areas_of_concentric_circles 
  (C1 C2 : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * C1 = 2 * π * r1)
  (h2 : r2 * C2 = 2 * π * r2)
  (h_c1 : 60 / 360 * C1 = 48 / 360 * C2) :
  (π * r1^2) / (π * r2^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l421_421454


namespace number_of_girls_l421_421853

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l421_421853


namespace solution_set_f_cos_x_l421_421259

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 3 then -(x-2)^2 + 1
else if x = 0 then 0
else if -3 < x ∧ x < 0 then (x+2)^2 - 1
else 0 -- Defined as 0 outside the given interval for simplicity

theorem solution_set_f_cos_x :
  {x : ℝ | f x * Real.cos x < 0} = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)} :=
sorry

end solution_set_f_cos_x_l421_421259


namespace monotonic_f_on_interval_l421_421395

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x - 1|

theorem monotonic_f_on_interval (a : ℝ) :
  (∀ x1 x2 ∈ set.Ici (0 : ℝ), x1 ≤ x2 → f a x1 ≤ f a x2) ↔ (-2 : ℝ) ≤ a ∧ a ≤ (0 : ℝ) :=
begin
  sorry
end

end monotonic_f_on_interval_l421_421395


namespace regular_octahedron_faces_l421_421639

def is_regular_octahedron (shape : Type) : Prop :=
  -- define what it means to be a regular octahedron
  ∃ (f : shape → ℕ), -- exists a function that assigns each face a unique number
  ∀ x : shape, f x = 8

theorem regular_octahedron_faces (shape : Type) (h : is_regular_octahedron shape) : 
  ∃ (n : ℕ), n = 8 :=
by
  unfold is_regular_octahedron at h
  obtain ⟨f, hf⟩ := h
  use 8
  sorry

end regular_octahedron_faces_l421_421639


namespace circumcircle_excircle_distance_squared_l421_421263

variable (R r_A d_A : ℝ)

theorem circumcircle_excircle_distance_squared 
  (h : R ≥ 0)
  (h1 : r_A ≥ 0)
  (h2 : d_A^2 = R^2 + 2 * R * r_A) : d_A^2 = R^2 + 2 * R * r_A := 
by
  sorry

end circumcircle_excircle_distance_squared_l421_421263


namespace arithmetic_sequence_sum_11_l421_421687

open Real

variable (a : ℕ → ℝ) 
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable h : ∀ n, a (n+1) = a n + d
variable h_seq : a 4 + a 5 + a 6 + a 7 + a 8 = 150

theorem arithmetic_sequence_sum_11 :
  S 11 = 330 :=
sorry

end arithmetic_sequence_sum_11_l421_421687


namespace solve_for_x_l421_421389

theorem solve_for_x : ∃ x : ℚ, (1/4 : ℚ) + (1/x) = 7/8 ∧ x = 8/5 :=
by {
  sorry
}

end solve_for_x_l421_421389


namespace circumcenter_parallelogram_l421_421330

open EuclideanGeometry

/--
Given a parallelogram ABCD, with points P, Q, R, and S chosen
on sides AB, BC, CD, and DA respectively, forming triangles
APD, BPQ, BRC, and DQC. Prove that the centers of the circumcircles
of these four triangles form vertices of another parallelogram.
-/
theorem circumcenter_parallelogram (A B C D P Q R S : Point ℝ 2):
  is_parallelogram A B C D →
  is_on_line_segment A B P →
  is_on_line_segment B C Q →
  is_on_line_segment C D R →
  is_on_line_segment D A S →
  ∃ O₁ O₂ O₃ O₄ : Point ℝ 2,
    is_circumcenter O₁ (triangle A P D) ∧
    is_circumcenter O₂ (triangle B P Q) ∧
    is_circumcenter O₃ (triangle B R C) ∧
    is_circumcenter O₄ (triangle D Q C) ∧
    is_parallelogram O₁ O₂ O₃ O₄ :=
begin
  sorry
end

end circumcenter_parallelogram_l421_421330


namespace total_cost_paid_l421_421359

-- Definition of the given conditions
def number_of_DVDs : ℕ := 4
def cost_per_DVD : ℝ := 1.2

-- The theorem to be proven
theorem total_cost_paid : number_of_DVDs * cost_per_DVD = 4.8 := by
  sorry

end total_cost_paid_l421_421359


namespace rhombus_diagonal_length_l421_421043

theorem rhombus_diagonal_length (d1 : ℝ) : 
  (d1 * 12) / 2 = 60 → d1 = 10 := 
by 
  sorry

end rhombus_diagonal_length_l421_421043


namespace part1_exists_infinite_rationals_part2_rationals_greater_bound_l421_421488

theorem part1_exists_infinite_rationals 
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2):
  ∀ ε > 0, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ abs (q / p - sqrt5_minus1_div2) < 1 / p ^ 2 :=
by sorry

theorem part2_rationals_greater_bound
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2)
  (sqrt5_plus1_inv := 1 / (Real.sqrt 5 + 1)):
  ∀ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 → abs (q / p - sqrt5_minus1_div2) > sqrt5_plus1_inv / p ^ 2 :=
by sorry

end part1_exists_infinite_rationals_part2_rationals_greater_bound_l421_421488


namespace relay_order_count_l421_421358

-- Definitions are required for the four team members and the condition
-- that Linda runs the last lap.

-- Assume the four team members are represented as a set.
noncomputable def team_members : Set String := { "Linda", "MemberA", "MemberB", "MemberC" }

-- Linda runs the last lap
def linda_last_lap (laps : List String) : Prop :=
  laps.length = 4 ∧ last laps = "Linda"

-- The number of permutations of the team members with Linda running the last lap:
noncomputable def permutations_count : ℕ := 6

-- Theorem that we need to state and prove
theorem relay_order_count (laps : List String) :
  (linda_last_lap laps → laps.permutations.length = permutations_count) := 
  sorry

end relay_order_count_l421_421358


namespace find_range_a_l421_421306

def bounded_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 2 → a * (4 ^ x) + 2 ^ x + 1 ≥ 0

theorem find_range_a :
  ∃ (a : ℝ), bounded_a a ↔ a ≥ -5 / 16 :=
sorry

end find_range_a_l421_421306


namespace tan_phi_of_right_triangle_l421_421668

theorem tan_phi_of_right_triangle (ABC : Type) [IsTriangle ABC] (angle_C : angle ABC = 90) (beta : Real)
    (tan_half_beta : Real) (phi : Real) (h1 : tan(beta / 2) = 2 / (3^(1/3))) :
    tan(phi) = (3^(1/3)) / 2 := by
  sorry

end tan_phi_of_right_triangle_l421_421668


namespace decimal_150th_digit_of_5_over_11_l421_421109

theorem decimal_150th_digit_of_5_over_11 :
  let repeating_sequence : string := "45" in
  let sequence_length := 2 in
  ∀ n, n = 150 →
  let digit_position := n % sequence_length in
  (if digit_position = 0 then repeating_sequence.get 1 else repeating_sequence.get (digit_position - 1)) = '5' :=
by
  sorry

end decimal_150th_digit_of_5_over_11_l421_421109


namespace fraction_multiplication_subtraction_l421_421486

theorem fraction_multiplication_subtraction :
  (3 + 1 / 117) * (4 + 1 / 119) - (2 - 1 / 117) * (6 - 1 / 119) - (5 / 119) = 10 / 117 :=
by
  sorry

end fraction_multiplication_subtraction_l421_421486


namespace volume_of_intersection_l421_421805

theorem volume_of_intersection (a : ℝ) (r : ℝ) (h : ℝ) 
  (eq_triangle : ∀ x y : ℝ, x = y) 
  (center_S : ∀ x y : ℝ, x = y) 
  (height_SH : ∀ x y : ℝ, x = y) :
  (a = 4) ∧ (r = 1) ∧ (h = 1 - sqrt(6) / 3) →
   volume_of_intersection a r = π * ((7 * sqrt(6) - 9) / 27) := 
by
  sorry

end volume_of_intersection_l421_421805


namespace waiter_customers_l421_421965

-- Define initial conditions
def initial_customers : ℕ := 47
def customers_left : ℕ := 41
def new_customers : ℕ := 20

-- Calculate remaining customers after some left
def remaining_customers : ℕ := initial_customers - customers_left

-- Calculate the total customers after getting new ones
def total_customers : ℕ := remaining_customers + new_customers

-- State the theorem to prove the final total customers
theorem waiter_customers : total_customers = 26 := by
  -- We include sorry for the proof placeholder
  sorry

end waiter_customers_l421_421965


namespace lemoine_segments_ratio_l421_421743

theorem lemoine_segments_ratio 
  (A B C P Q R S T U : Type) 
  (L Lemoine_point : triangle L_Coincides L) 
  (a b c ur qt sp : length) : Prop :=
begin
  have h1 := parallel_lines_through_lemoine,
  have h2 := intersection_points_Lemoine a b c A B C P Q R S T U,
  have h3 := proportional_ratios h1 h2,
  exact (ur / qt = qt / sp ∧ qt / sp = sp / a^3 ∧ sp / a^3 = b^3 / c^3),
end

end lemoine_segments_ratio_l421_421743


namespace vans_taken_l421_421834

theorem vans_taken (cars taxis vans : ℕ)
                   (people_per_car people_per_taxi people_per_van total_people : ℕ)
                   (h1 : cars = 3)
                   (h2 : taxis = 6)
                   (h3 : people_per_car = 4)
                   (h4 : people_per_taxi = 6)
                   (h5 : people_per_van = 5)
                   (h6 : total_people = 58) :
  vans = 2 :=
by 
  let total_people_in_cars_and_taxis := cars * people_per_car + taxis * people_per_taxi
  let people_in_vans := total_people - total_people_in_cars_and_taxis
  let vans_needed := people_in_vans / people_per_van
  have h_total_people_in_cars_and_taxis : total_people_in_cars_and_taxis = 12 + 36,
  { calc
      total_people_in_cars_and_taxis = cars * people_per_car + taxis * people_per_taxi : rfl
      ... = 3 * 4 + 6 * 6 : by simp [h1, h2, h3, h4]
      ... = 12 + 36 : by simp },
  have h_people_in_vans : people_in_vans = 10,
  { calc
      people_in_vans = total_people - total_people_in_cars_and_taxis : rfl
      ... = 58 - 48 : by simp [h6, h_total_people_in_cars_and_taxis] },
  have h_vans_needed : vans_needed = 2,
  { calc
      vans_needed = people_in_vans / people_per_van : rfl
      ... = 10 / 5 : by simp [h5, h_people_in_vans]
      ... = 2 : by norm_num },
  exact sorry

end vans_taken_l421_421834


namespace line_l_rect_eq_curve_C_ord_eq_line_curve_intersect_chord_length_l421_421685

open Real

-- Definitions of the given conditions
def line_l_polar (ρ θ : ℝ) : Prop := ρ * sin (θ - π / 6) = 1 / 2
def curve_C_param (x y α : ℝ) : Prop := x = 1 + 3 * cos α ∧ y = 3 * sin α

-- Proving the rectangular coordinate equation of line l and ordinary equation of curve C
theorem line_l_rect_eq (ρ θ : ℝ) : (∃ x y : ℝ, line_l_polar ρ θ ∧ ρ * cos θ = x ∧ ρ * sin θ = y) →
  (∃ x y : ℝ, x - √3 * y + 1 = 0) :=
sorry

theorem curve_C_ord_eq (x y α : ℝ) : curve_C_param x y α →
  (x - 1) ^ 2 + y ^ 2 = 9 :=
sorry

-- Proving intersection and calculating the chord length
theorem line_curve_intersect_chord_length : 
  (∃ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 9 ∧ x - √3 * y + 1 = 0) →
  (∃ d : ℝ, d < 3 ∧ d = 1) →
  (∃ chord_len : ℝ, chord_len = 4 * sqrt 2) :=
sorry

end line_l_rect_eq_curve_C_ord_eq_line_curve_intersect_chord_length_l421_421685


namespace cos_alpha_value_l421_421645

theorem cos_alpha_value (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
                        (hβ1 : π / 2 < β) (hβ2 : β < π)
                        (hcosβ : cos β = -3 / 5)
                        (hsinalphabeta : sin (α + β) = 5 / 13) :
  cos α = 56 / 65 := 
sorry

end cos_alpha_value_l421_421645


namespace find_a_l421_421441

theorem find_a (a : ℝ) :
  let θ := 120
  let tan120 := -Real.sqrt 3
  (∀ x y: ℝ, 2 * x + a * y + 3 = 0) →
  a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end find_a_l421_421441


namespace magnitude_of_1_plus_z_l421_421269

noncomputable def z : ℂ := (1 - complex.I) / (complex.I + 1)

theorem magnitude_of_1_plus_z : |1 + z| = real.sqrt 2 := by
  -- Declaration of z is needed due to its noncomputable nature.
  sorry

end magnitude_of_1_plus_z_l421_421269


namespace solve_absolute_value_eq_l421_421914

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421914


namespace number_of_arrangements_l421_421210

def students := {A, B, C, D, E}

-- Condition: Students A and B are neither first nor last.
def valid_arrangement (arr: List char) : Prop :=
  arr.length = 5 ∧ (∀ i, arr[i] ∈ students) ∧
  arr[0] ≠ 'A' ∧ arr[0] ≠ 'B' ∧
  arr[4] ≠ 'A' ∧ arr[4] ≠ 'B'

theorem number_of_arrangements : 
  (finset.univ.filter valid_arrangement).card = 36 :=
  by
    sorry

end number_of_arrangements_l421_421210


namespace sequence_solution_l421_421230

-- Define the sequence a_{n}
def a : ℕ → ℝ
| 0       := 1     -- a_1 = 1
| (n + 1) := (1/16) * (1 + 4 * a n + real.sqrt (1 + 24 * a n))

-- Theorem stating the equivalence that we need to prove
theorem sequence_solution (n : ℕ) : 
  a n = (1 / 24) * (2^(2*n - 1) + 3 * 2^n - 1 + 1) := 
sorry

end sequence_solution_l421_421230


namespace find_f_l421_421228

def f (x : ℝ) (f'_1 : ℝ) : ℝ := x^2 + 3 * x * f'_1

theorem find_f'_1 (f'_1 : ℝ) (h : ∀ x : ℝ, deriv (λ x, f x f'_1) x = 2 * x + 3 * f'_1) : f'_1 = -1 :=
by
  have h1 : deriv (λ x, f x f'_1) 1 = 2 * 1 + 3 * f'_1 := h 1
  sorry

end find_f_l421_421228


namespace power_value_l421_421294

theorem power_value (m n : ℤ) (h : |m - 2| + (n + 3)^2 = 0) : n ^ m = 9 :=
sorry

end power_value_l421_421294


namespace rate_percent_simple_interest_l421_421464

theorem rate_percent_simple_interest:
  ∀ (P SI T R : ℝ), SI = 400 → P = 1000 → T = 4 → (SI = P * R * T / 100) → R = 10 :=
by
  intros P SI T R h_si h_p h_t h_formula
  -- Proof skipped
  sorry

end rate_percent_simple_interest_l421_421464


namespace largest_prime_factor_2999_l421_421886

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  -- Note: This would require actual computation logic to find the largest prime factor.
  sorry

theorem largest_prime_factor_2999 :
  largest_prime_factor 2999 = 103 :=
by 
  -- Given conditions:
  -- 1. 2999 is an odd number (doesn't need explicit condition in proof).
  -- 2. Sum of digits is 29, thus not divisible by 3.
  -- 3. 2999 is not divisible by 11.
  -- 4. 2999 is not divisible by 7, 13, 17, 19.
  -- 5. Prime factorization of 2999 is 29 * 103.
  admit -- actual proof will need detailed prime factor test results 

end largest_prime_factor_2999_l421_421886


namespace spherical_triangle_area_correct_l421_421569

noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ :=
  R^2 * (α + β + γ - Real.pi)

theorem spherical_triangle_area_correct (R α β γ : ℝ) :
  spherical_triangle_area R α β γ = R^2 * (α + β + γ - Real.pi) := by
  sorry

end spherical_triangle_area_correct_l421_421569


namespace g_n_2_formula_g_n_plus_1_k_g_7_3_l421_421346

/- Define g(n, k) as the number of ways to partition an n-element set into k subsets each containing at least 2 elements. -/
namespace PartitionProblem

open Nat

/-- The formula for g(n, 2) when n ≥ 4 is 2^(n-1) - n - 1. -/
theorem g_n_2_formula (n : ℕ) (h : n ≥ 4) : (g n 2) = 2^(n - 1) - n - 1 :=
sorry

/-- The recursive relation for g(n+1, k) is given by:
    - g(n+1, k) = n * g(n-1, k-1) + k * g(n, k) for n ≥ 2k ≥ 6. -/
theorem g_n_plus_1_k (n k : ℕ) (h : n ≥ 2 * k ∧ 2 * k ≥ 6) : g (n + 1) k = n * g (n - 1) (k - 1) + k * g n k :=
sorry

/-- The number of ways to partition a 7-element set into 3 subsets each containing at least 2 elements is 105. -/
theorem g_7_3 : g 7 3 = 105 :=
sorry

end PartitionProblem

end g_n_2_formula_g_n_plus_1_k_g_7_3_l421_421346


namespace or_necessary_not_sufficient_for_and_l421_421806

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∨ q) → (p ∧ q) → False :=
by
  intros h₁ h₂
  exact h₁.elim (λ hp, h₂.elim (λ hp _ , False) (λ _ _, False)) (λ hq, h₂.elim (λ _ _, False) (λ _, hq.symm.elim (λ _, False)))

end or_necessary_not_sufficient_for_and_l421_421806


namespace tom_strokes_over_par_l421_421084

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l421_421084


namespace number_of_girls_l421_421864

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l421_421864


namespace sufficient_but_not_necessary_l421_421727

noncomputable def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y
noncomputable def is_increasing (g : ℝ → ℝ) := ∀ x y : ℝ, x < y → g x < g y

theorem sufficient_but_not_necessary (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  (is_decreasing (λ x, a^x) → is_increasing (λ x, (2 - a) * x^3)) ∧ 
  (¬ is_decreasing (λ x, a^x) → is_increasing (λ x, (2 - a) * x^3)) :=
sorry

end sufficient_but_not_necessary_l421_421727


namespace original_price_l421_421433

theorem original_price (x: ℝ) (h1: x * 1.1 * 0.8 = 2) : x = 25 / 11 :=
by
  sorry

end original_price_l421_421433


namespace total_coronavirus_cases_l421_421332

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l421_421332


namespace pete_minimum_cells_l421_421677

theorem pete_minimum_cells (n : ℕ) (table : ℕ → ℕ → ℕ) (h1 : ∀ (i j : ℕ), 1 ≤ table i j ∧ table i j ≤ 9) : 
  ∃ (cells : fin n → fin n), 
  (∀ i, 1 ≤ table (cells i) (cells i) ∧ table (cells i) (cells i) ≤ 9) ∧ 
  (∀ (a : string), (a.length = n) → 
                   (∀ k, 1 ≤ string.to_nat (string.get_digit a k) ∧ string.to_nat (string.get_digit a k) ≤ 9) → 
                   (a ≠ a.reverse)) :=
sorry

end pete_minimum_cells_l421_421677


namespace possible_measures_of_angle_A_l421_421827

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l421_421827


namespace measure_of_angle_A_possibilities_l421_421815

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l421_421815


namespace excenter_opposite_angle_ABC_l421_421701

theorem excenter_opposite_angle_ABC :
  ∀ {A B C : Type} [angle_points A B C],
    ∠ BAC = 48 →
    ∠ ABC = 65 →
    ∃ (I_a : Type), is_excenter_opposite I_a A B C ∧ ∠ BIC_a = 56.5 :=
by
  sorry

end excenter_opposite_angle_ABC_l421_421701


namespace cement_bag_weight_in_pounds_l421_421142

theorem cement_bag_weight_in_pounds :
  let kg_to_pounds (kg : ℕ) : ℝ := kg / 0.454
  let weight_in_kg := 150
  let weight_in_pounds := kg_to_pounds weight_in_kg
  Int.nearest weight_in_pounds = 330 := 
by 
  -- Definitions and assumptions
  let kg_to_pounds (kg : ℕ) : ℝ := kg / 0.454
  let weight_in_kg := 150
  let weight_in_pounds := kg_to_pounds weight_in_kg
  
  -- Convert the weight
  have h_weight_in_pounds : weight_in_pounds = 150 / 0.454 := by simp [kg_to_pounds, weight_in_kg]
  have h_rounded_weight : Int.nearest weight_in_pounds = 330 := 
    by 
    calc 
      Int.nearest (150.0 / 0.454) = 330 : sorry -- Rounding to nearest whole number

  -- Final assertion
  exact h_rounded_weight

end cement_bag_weight_in_pounds_l421_421142


namespace number_of_satisfying_functions_l421_421971

-- Define all functions to be checked
def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := 1 - x^2
def f3 (x : ℝ) [h : x > -π/2 ∧ x < π/2] : ℝ := abs (tan x)
def f4 (x : ℝ) : ℝ := 2 * x + 1

-- Define the condition to be checked
def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, f((x + 2 * y) / 3) ≥ (1 / 3) * f(x) + (2 / 3) * f(y)

-- The statement to be proven
theorem number_of_satisfying_functions : 
  (if satisfies_condition f1 then 1 else 0) +
  (if satisfies_condition f2 then 1 else 0) +
  (if satisfies_condition (λ x, f3 x sorry) then 1 else 0) +
  (if satisfies_condition f4 then 1 else 0) = 2 := 
sorry

end number_of_satisfying_functions_l421_421971


namespace point_in_fourth_quadrant_l421_421599

def z1 : ℂ := 2 + I
def z2 : ℂ := 1 + I
def z := z1 / z2
def point : ℝ × ℝ := (z.re, z.im)

theorem point_in_fourth_quadrant : point.snd < 0 ∧ point.fst > 0 :=
by
  have h1 : z = (3/2) - (1/2) * I := by sorry
  have h2 : point = (3/2, -1/2) := by sorry
  exact ⟨by norm_num, by norm_num⟩

end point_in_fourth_quadrant_l421_421599


namespace closest_fraction_is_one_sixth_l421_421528

def fraction_of_medals_won : ℚ := 17 / 100

def fractions_to_compare : List ℚ := [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]

def closest_fraction (target : ℚ) (fractions : List ℚ) : ℚ :=
fractions.argmin (λ f, abs (target - f))

theorem closest_fraction_is_one_sixth :
  closest_fraction fraction_of_medals_won fractions_to_compare = 1 / 6 :=
by sorry

end closest_fraction_is_one_sixth_l421_421528


namespace equilateral_triangle_area_l421_421407

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l421_421407


namespace tim_income_less_juan_l421_421757

variable {T M J : ℝ}

theorem tim_income_less_juan :
  (M = 1.60 * T) → (M = 0.6400000000000001 * J) → T = 0.4 * J :=
by
  sorry

end tim_income_less_juan_l421_421757


namespace lines_perpendicular_to_line_in_plane_l421_421303

theorem lines_perpendicular_to_line_in_plane
  (a : Line) (alpha : Plane) :
  (¬ is_perpendicular a alpha) → (∃ (l : Set Line), ∀ l ∈ l, is_perpendicular l a ∧ infinite l) :=
by
  sorry

end lines_perpendicular_to_line_in_plane_l421_421303


namespace simplify_fraction_l421_421031

theorem simplify_fraction (k : ℤ) : 
  (∃ (a b : ℤ), a = 1 ∧ b = 2 ∧ (6 * k + 12) / 6 = a * k + b) → (1 / 2 : ℚ) = (1 / 2 : ℚ) := 
by
  intro h
  sorry

end simplify_fraction_l421_421031


namespace average_percentage_all_students_l421_421652

theorem average_percentage_all_students
  (n1 n2 : ℕ) (avg1 avg2 : ℝ)
  (h1 : n1 = 15) (h2 : n2 = 10)
  (h3 : avg1 = 80) (h4 : avg2 = 90)
  (h_students : n1 + n2 = 25) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 84 :=
by
  have h_total_points : (n1 * avg1 + n2 * avg2) = 1200 + 900 := by sorry
  have h_total_students : (n1 + n2) = 25 := h_students
  have h_average : (1200 + 900) / 25 = 84 := by sorry
  show (n1 * avg1 + n2 * avg2) / (n1 + n2) = 84, from h_average

end average_percentage_all_students_l421_421652


namespace possible_remainder_degrees_l421_421473

-- Defining the polynomial divisor
def divisor : Polynomial ℝ := 3 * X^2 - 4 * X + 5

-- The degree of the divisor
def degree_divisor := Polynomial.degree divisor

-- The mathematically equivalent proof problem in Lean statement
theorem possible_remainder_degrees (f : Polynomial ℝ) :
  ∃ (r : Polynomial ℝ), Polynomial.degree r < degree_divisor ∧ (0 ≤ Polynomial.degree r ∧ Polynomial.degree r ≤ 1) :=
sorry

end possible_remainder_degrees_l421_421473


namespace angle_measures_possible_l421_421820

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l421_421820


namespace elementary_school_classes_count_l421_421955

theorem elementary_school_classes_count (E : ℕ) (donate_per_class : ℕ) (middle_school_classes : ℕ) (total_balls : ℕ) :
  donate_per_class = 5 →
  middle_school_classes = 5 →
  total_balls = 90 →
  5 * 2 * E + 5 * 2 * middle_school_classes = total_balls →
  E = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end elementary_school_classes_count_l421_421955


namespace segment_theorem_l421_421590

def Segment := (ℝ × ℝ)

noncomputable def segments := (list Segment)

theorem segment_theorem (L : segments) (hL : L.length = 50) :
  ∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Segment), 
  (∃ p : ℝ, 
    (p ∈ (set_of points of s1)) ∧ 
    (p ∈ (set_of points of s2)) ∧ 
    (p ∈ (set_of points of s3)) ∧ 
    (p ∈ (set_of points of s4)) ∧ 
    (p ∈ (set_of points of s5)) ∧ 
    (p ∈ (set_of points of s6)) ∧ 
    (p ∈ (set_of points of s7)) ∧ 
    (p ∈ (set_of points of s8))) ∨
  (¬ ∃ (p1 p2 : Segment), p1 ∩ p2 ≠ ∅ ∧ 
    {s1, s2, s3, s4, s5, s6, s7, s8}.pairwise (λ x y, x ∩ y = ∅)) :=
sorry

end segment_theorem_l421_421590


namespace max_chord_length_l421_421621

theorem max_chord_length (θ : ℝ) : 
  let f := 2 * (2 * Real.sin θ - Real.cos θ + 3)
  let g := 8 * Real.sin θ + Real.cos θ + 1
  (∃ x y, (2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0) ∧ y = 2 * x) →
  (∀ x, (x ^ 2 + 16 * x - 16 ≤ 0) → |x| ≤ 8) →
  8 - 2 * x ≤ 0 →
  chord_length = 8 := sorry

end max_chord_length_l421_421621


namespace count_valid_n_l421_421215

theorem count_valid_n : (∃ (n_set : Finset ℕ), (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 60 ∧ ( (factorial ((n^3) - 1)) / (factorial n)^n ).is_nat) ∧ n_set.card = 40) :=
sorry

end count_valid_n_l421_421215


namespace auctioneer_price_increase_per_bid_l421_421972

-- Definitions based on conditions
def P_initial : ℝ := 15
def P_final : ℝ := 65
def N : ℕ := 10

-- The statement to be proved
theorem auctioneer_price_increase_per_bid :
  (P_final - P_initial) / N = 5 := 
begin
  sorry,
end

end auctioneer_price_increase_per_bid_l421_421972


namespace proof_problem_l421_421723

theorem proof_problem (a b c : ℝ) (h1 : 4 * a - 2 * b + c > 0) (h2 : a + b + c < 0) : b^2 > a * c :=
sorry

end proof_problem_l421_421723


namespace coefficient_x5_in_product_l421_421103

noncomputable def P : Polynomial ℤ := 
  Polynomial.C 1 * Polynomial.X ^ 6 +
  Polynomial.C (-2) * Polynomial.X ^ 5 +
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C (-6) * Polynomial.X +
  Polynomial.C 7

noncomputable def Q : Polynomial ℤ := 
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C 6 * Polynomial.X +
  Polynomial.C (-8)

theorem coefficient_x5_in_product (p q : Polynomial ℤ) :
  (p * q).coeff 5 = 2 :=
by
  have P := 
    Polynomial.C 1 * Polynomial.X ^ 6 +
    Polynomial.C (-2) * Polynomial.X ^ 5 +
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C (-6) * Polynomial.X +
    Polynomial.C 7
  have Q := 
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C 6 * Polynomial.X +
    Polynomial.C (-8)

  sorry

end coefficient_x5_in_product_l421_421103


namespace negation_of_proposition_l421_421063

theorem negation_of_proposition :
    (¬ ∃ (x : ℝ), (Real.exp x - x - 1 < 0)) ↔ (∀ (x : ℝ), Real.exp x - x - 1 ≥ 0) :=
by
  sorry

end negation_of_proposition_l421_421063


namespace minimum_cells_required_l421_421674

variable (n : ℕ)
variable (table : ℕ → ℕ → ℕ)
variable (H_table : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 9)

theorem minimum_cells_required : (∃ (cells : fin n → fin n), 
  (∀ i j, cells i = cells j → i = j) ∧
  (∀ i, 1 ≤ table (cells i).val (cells i).val ∧ table (cells i).val (cells i).val ≤ 9) ∧
  (∀ number : ℕ, (∀ i : fin n, number = (table (cells i).val (cells i).val)) 
    → (∀ j : fin n, (∀ k : fin n, table j k = number) ∨ (∀ k : fin n, table k j = number) → false))) := 
begin
  sorry
end

end minimum_cells_required_l421_421674


namespace possible_values_of_a_l421_421254

noncomputable def M : set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N (a : ℝ) : set ℝ := {x | x^2 - a * x + 3 * a - 5 = 0}

theorem possible_values_of_a (a : ℝ) (h : M ∪ N a = M) : a ∈ {a | 2 ≤ a ∧ a < 10} :=
sorry

end possible_values_of_a_l421_421254


namespace triangle_height_l421_421801

theorem triangle_height (base height : ℝ) (area : ℝ) (h_base : base = 4) (h_area : area = 12) (h_area_eq : area = (base * height) / 2) :
  height = 6 :=
by
  sorry

end triangle_height_l421_421801


namespace minimum_distance_l421_421594

theorem minimum_distance (a : ℝ) :
  ∃ N M : ℝ × ℝ × ℝ,
  (\N \in diagonal_side_face ∧ M \in circle_base_plane_center_radius) →
  min_distance a N M = a * real.sqrt 34 / 24 :=
  sorry

end minimum_distance_l421_421594


namespace max_items_for_2019_students_l421_421489

noncomputable def max_items (students : ℕ) : ℕ :=
  students / 2

theorem max_items_for_2019_students : max_items 2019 = 1009 := by
  sorry

end max_items_for_2019_students_l421_421489


namespace ratio_AF_AT_l421_421702

theorem ratio_AF_AT {A B C D E T F : Type} 
  [geometry] (h1 : segment A B D = 2) (h2 : segment D B B = 2)
  (h3 : segment A C E = 3) (h4 : segment E C C = 3) 
  (h5 : angle_bisector A T (E, F) D) :
  segment_ratio A F A T = 1 / 2 ∧ segment_ratio B F F D = 1 := 
by
  sorry

end ratio_AF_AT_l421_421702


namespace number_of_sets_A_l421_421431

open Set

theorem number_of_sets_A : 
  {A : Set ℕ // insert 0 (insert 1 A) = insert 0 (singleton 1)}.finite 
  ∧ {A : Set ℕ // insert 0 (insert 1 A) = insert 0 (singleton 1)}.to_finset.card = 4 :=
begin
  sorry
end

end number_of_sets_A_l421_421431


namespace green_flower_percentage_l421_421756

theorem green_flower_percentage (yellow purple green total : ℕ)
  (hy : yellow = 10)
  (hp : purple = 18)
  (ht : total = 35)
  (hgreen : green = total - (yellow + purple)) :
  ((green * 100) / (yellow + purple)) = 25 := 
by {
  sorry
}

end green_flower_percentage_l421_421756


namespace donna_paid_165_l421_421951

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end donna_paid_165_l421_421951


namespace Xiaokang_position_l421_421123

theorem Xiaokang_position :
  let east := 150
  let west := 100
  let total_walks := 3
  (east - west - west = -50) :=
sorry

end Xiaokang_position_l421_421123


namespace single_discount_percentage_l421_421146

noncomputable def original_price : ℝ := 9795.3216374269
noncomputable def sale_price : ℝ := 6700
noncomputable def discount_percentage (p₀ p₁ : ℝ) : ℝ := ((p₀ - p₁) / p₀) * 100

theorem single_discount_percentage :
  discount_percentage original_price sale_price = 31.59 := 
by
  sorry

end single_discount_percentage_l421_421146


namespace solve_diamond_l421_421646

theorem solve_diamond (d : ℕ) (hd : d < 10) (h : d * 9 + 6 = d * 10 + 3) : d = 3 :=
sorry

end solve_diamond_l421_421646


namespace calories_left_for_dinner_l421_421555

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l421_421555


namespace inequality_solution_set_l421_421789

theorem inequality_solution_set (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) ↔ (-1 ≤ x ∧ x < 3) := 
by sorry

end inequality_solution_set_l421_421789


namespace find_x_l421_421896

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421896


namespace marble_selection_l421_421337

theorem marble_selection :
  let total_marbles := 15
  let red_marbles := 2
  let green_marbles := 2
  let blue_marbles := 2
  let other_marbles := total_marbles - (red_marbles + green_marbles + blue_marbles)
  in (finset.card ((finset.subsets (finset.range total_marbles) 6).filter (λ s, finset.count red_marbles s + finset.count green_marbles s + finset.count blue_marbles s = 2)) = 1485 :=
sorry

end marble_selection_l421_421337


namespace digit_in_150th_place_of_5_over_11_l421_421114

theorem digit_in_150th_place_of_5_over_11 :
  let dec_rep := "45" -- represents the repeating part of the decimal
  (dec_rep[(150 - 1) % 2] = '5') := 
  sorry

end digit_in_150th_place_of_5_over_11_l421_421114


namespace problem_statement_l421_421354

variables {α β : Type*} [TopologicalSpace α] [TopologicalSpace β]
variables (a b c : Set α) (h : α → β)

/-- Given two different planes \(α\) and \(β\), and a line \(a\), 
    if \(a \perp α\) and \(a \perp β\), then \(α \parallel β\). -/
theorem problem_statement 
  (α β : Set α) (a : Set α) 
  (h₁ : a ⊥ α) (h₂ : a ⊥ β) : α ∥ β :=
sorry

end problem_statement_l421_421354


namespace real_to_fraction_l421_421884

noncomputable def real_num : ℚ := 3.675

theorem real_to_fraction : real_num = 147 / 40 :=
by
  -- convert 3.675 to a mixed number
  have h1 : real_num = 3 + 675 / 1000 := by sorry
  -- find gcd of 675 and 1000
  have h2 : Nat.gcd 675 1000 = 25 := by sorry
  -- simplify 675/1000 to 27/40
  have h3 : 675 / 1000 = 27 / 40 := by sorry
  -- convert mixed number to improper fraction 147/40
  have h4 : 3 + 27 / 40 = 147 / 40 := by sorry
  -- combine the results to prove the required equality
  exact sorry

end real_to_fraction_l421_421884


namespace b_work_alone_days_l421_421477

variable (a_rate b_rate c_rate : ℝ)
variable (work_time total_days : ℝ)
variable (a_days c_days : ℝ)
variable (x : ℝ) -- days b can do the work alone

-- Conditions
def a_can_complete_work : Prop := a_rate = 1 / 24
def c_can_complete_work : Prop := c_rate = 1 / 60
def b_rate_is_defined : Prop := b_rate = 1 / x
def total_work_is_done : Prop :=
  (a_rate + b_rate + c_rate) * (total_days - 4) + (a_rate + b_rate) * 4 = 1
def work_completion_time : Prop := total_days = 11

theorem b_work_alone_days 
  (h1 : a_can_complete_work)
  (h2 : c_can_complete_work)
  (h3 : b_rate_is_defined)
  (h4 : total_work_is_done)
  (h5 : work_completion_time) : 
  x = 125 := 
sorry

end b_work_alone_days_l421_421477


namespace find_starting_number_l421_421034

theorem find_starting_number : 
  ∃ x : ℕ, (∀ k : ℕ, (k < 12 → (x + 3 * k) ≤ 46) ∧ 12 = (46 - x) / 3 + 1) 
  ∧ x = 12 := 
by 
  sorry

end find_starting_number_l421_421034


namespace simplify_expression_l421_421384

theorem simplify_expression :
  (123 / 999) * 27 = 123 / 37 :=
by sorry

end simplify_expression_l421_421384


namespace ice_cream_sale_necessity_l421_421028

theorem ice_cream_sale_necessity 
  (game_cost : ℝ) (selling_price : ℝ) (cost_per_ice_cream : ℝ) (tax_rate : ℝ) (total_cost : ℝ)
  (profit_per_ice_cream : selling_price - cost_per_ice_cream = 3.5)
  (games_needed : total_cost = 2 * game_cost)
  (game_cost_value : game_cost = 60)
  (selling_price_value : selling_price = 5)
  (cost_per_ice_cream_value : cost_per_ice_cream = 1.5)
  (tax_rate_value : tax_rate = 0.1) :
  let P := total_cost / (1 - tax_rate) in
  let number_of_ice_creams_needed := P / 3.5 in
  ceil number_of_ice_creams_needed = 39 :=
by
  sorry

end ice_cream_sale_necessity_l421_421028


namespace additional_discount_percentage_l421_421561

-- Define constants representing the conditions
def price_shoes : ℝ := 200
def discount_shoes : ℝ := 0.30
def price_shirt : ℝ := 80
def number_shirts : ℕ := 2
def final_spent : ℝ := 285

-- Define the theorem to prove the additional discount percentage
theorem additional_discount_percentage :
  let discounted_shoes := price_shoes * (1 - discount_shoes)
  let total_before_additional_discount := discounted_shoes + number_shirts * price_shirt
  let additional_discount := total_before_additional_discount - final_spent
  (additional_discount / total_before_additional_discount) * 100 = 5 :=
by
  -- Lean proof goes here, but we'll skip it for now with sorry
  sorry

end additional_discount_percentage_l421_421561


namespace arrangement_ways_l421_421446

-- Define the problem's conditions
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def girls_adjacent : Bool := true

-- Now, state our theorem based on the problem and its conditions
theorem arrangement_ways (b g : ℕ) (adjacent : Bool) (h1 : b = num_boys) (h2 : g = num_girls) 
  (h3 : adjacent = girls_adjacent) : 
  (number_of_ways : ℕ) := 48 :=
by {
  sorry
}

end arrangement_ways_l421_421446


namespace horner_eval_and_operations_count_l421_421173

noncomputable def polynomial_value : ℚ :=
  let f : ℚ[X] := 3 * X^6 + 4 * X^5 + 5 * X^4 + 6 * X^3 + 7 * X^2 + 8 * X + 1
  in f.eval (4 / 10)

theorem horner_eval_and_operations_count :
  let f : ℚ[X] := 3 * X^6 + 4 * X^5 + 5 * X^4 + 6 * X^3 + 7 * X^2 + 8 * X + 1
  let x : ℚ := 4 / 10
  (f.eval x = polynomial_value)
  ∧ ((6 = 6) ∧ (6 = 6)) :=
by
  sorry

end horner_eval_and_operations_count_l421_421173


namespace irrational_sqrt_two_l421_421120

theorem irrational_sqrt_two :
  ∃ x ∈ ({3.14, -2, sqrt 2, 1 / 3} : Set ℝ), x = sqrt 2 ∧ ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b :=
by
  sorry

end irrational_sqrt_two_l421_421120


namespace find_a_l421_421271

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = abs (2 * x - a) + a)
  (h2 : ∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 := by
  sorry

end find_a_l421_421271


namespace tangent_line_at_fixed_point_range_of_m_l421_421277

variable (m : ℝ)
noncomputable def f (x : ℝ) := m * x * real.log (x + 1) + x + 1

theorem tangent_line_at_fixed_point :
  ∀ x, f m x = m * x * real.log (x + 1) + x + 1 →
  (∀ x, (∃ a b : ℝ, line a b x = a * x + b ∧ f m x = a * x + b)) →
  (∀ x, line 1 1 x = x + 1) :=
sorry

theorem range_of_m :
  ∀ (m : ℝ), (∀ x ≥ 0, f m x ≤ real.exp x) → m ≤ (1/2) :=
sorry

end tangent_line_at_fixed_point_range_of_m_l421_421277


namespace value_of_m_l421_421060

theorem value_of_m :
  ∃ m : ℝ, (3 - 1) / (m + 2) = 1 → m = 0 :=
by 
  sorry

end value_of_m_l421_421060


namespace incorrect_permutations_l421_421462

/-- The word "error" consists of 5 letters with 'r' repeated 3 times -/
def word := "error"
def total_letters := 5
def repeated_r := 3
def correct_permutations := 1

/-- Calculate the number of incorrect permutations of the word "error" -/
theorem incorrect_permutations : nat := fact(total_letters) / fact(repeated_r) - correct_permutations

example : incorrect_permutations = 19 := sorry

end incorrect_permutations_l421_421462


namespace magnitude_conjugate_of_z_l421_421745

noncomputable def z : ℂ := (1 / (1 - complex.I)) + complex.I

theorem magnitude_conjugate_of_z : complex.abs (conj z) = real.sqrt 10 / 2 := by
  sorry

end magnitude_conjugate_of_z_l421_421745


namespace second_player_wins_with_perfect_play_l421_421869

def ball_colors := ["red", "blue", "white", "black"]
def num_balls_per_color := 2

noncomputable def total_balls := ball_colors.length * num_balls_per_color
noncomputable def cube_vertices := 8

structure GameState :=
  (vertex_filled : Fin cube_vertices → Option String) -- Keeps track of which vertex has which colored ball

def PlayerA_goal (gs : GameState) (v : Fin cube_vertices) : Prop :=
  let neighbors : List (Fin cube_vertices) := [] -- Assume neighbors can be computed
  let colors_at_v := [gs.vertex_filled v, Option.bind (List.head? neighbors) (gs.vertex_filled), Option.bind (List.nth? neighbors 1) (gs.vertex_filled), Option.bind (List.nth? neighbors 2) (gs.vertex_filled)]
  colors_at_v.eraseDups.length = ball_colors.length

def PlayerB_goal (gs : GameState) : Prop :=
  ∀ v : Fin cube_vertices, ¬ PlayerA_goal gs v

theorem second_player_wins_with_perfect_play :
  ∃ strategy_b : (Fin cube_vertices → String) → (GameState → Fin cube_vertices * String),
  ∀ state : GameState, PlayerB_goal (strategy_b (state.vertex_filled)) :=
sorry

end second_player_wins_with_perfect_play_l421_421869


namespace number_of_girls_l421_421849

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l421_421849


namespace four_digit_numbers_2033_l421_421637

theorem four_digit_numbers_2033 : 
  let digits := [2, 0, 3, 3],
  let is_valid_number (d : List ℕ) := d.head != 0,
  ∃ (n : ℕ), n = 18 ∧ (∀ (L : List (List ℕ)), 
    (∀ l ∈ L, (l ~ digits ∧ is_valid_number l)) → L.length = n) :=
by
  sorry

end four_digit_numbers_2033_l421_421637


namespace first_term_exceeds_10000_l421_421048

noncomputable def sequence : ℕ → ℕ
| 0       := 2
| 1       := 2
| (n + 2) := if (n + 2) % 2 = 1 then (Finset.range (n + 2)).sum sequence
              else (Finset.range (n + 2)).sum sequence + sequence (n+1) * sequence n

theorem first_term_exceeds_10000 : ∃ n, sequence n > 10000 ∧ ∀ k < n, sequence k ≤ 10000 :=
sorry

end first_term_exceeds_10000_l421_421048


namespace no_solution_for_y_l421_421787

theorem no_solution_for_y : ¬ ∃ y : ℝ, 16^(3 * y) = 64^(2 * y + 1) :=
by
  sorry

end no_solution_for_y_l421_421787


namespace calories_remaining_for_dinner_l421_421560

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l421_421560


namespace tom_strokes_over_par_l421_421085

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l421_421085


namespace draw_segment_equal_to_given_segment_l421_421583

variable {Point : Type*}
variable {Line : Type*}
variable (A : Point)
variable (l : Line)
variable (segment_length : ℝ)

-- Assuming the existence of a function that tells us if a point is on a line
variable (is_on_line : Point → Line → Prop)
-- Assuming the existence of a function that calculates the distance between two points
variable (distance : Point → Point → ℝ)
-- Assuming the existence of a function that draws the circle
variable (circle_centered_at : Point → ℝ → set Point)
-- Assuming the existence of a function that provides line intersections
variable (line_intersections_circle : Line → set Point → set Point)

theorem draw_segment_equal_to_given_segment :
  ∃ B C : Point, 
    is_on_line A l → 
    (B ∈ line_intersections_circle l (circle_centered_at A segment_length) ∧ distance A B = segment_length) ∨
    (C ∈ line_intersections_circle l (circle_centered_at A segment_length) ∧ distance A C = segment_length) :=
sorry

end draw_segment_equal_to_given_segment_l421_421583


namespace initial_ratio_is_2_63_l421_421501

noncomputable theory

def initial_ratio_of_firm_partners_associates : Prop :=
  ∃ (a : ℕ), ∃ (g : ℕ), g > 0 ∧ 14 * 34 = a + 35 ∧ g.gcd 14 = 1 ∧ g.gcd a = 1 ∧ (14 / g : ℤ) = 2 ∧ (a / g : ℤ) = 63

theorem initial_ratio_is_2_63 : initial_ratio_of_firm_partners_associates :=
sorry

end initial_ratio_is_2_63_l421_421501


namespace maximum_value_le_maximizing_values_combined_value_l421_421350

noncomputable def maximum_value (x y z v w : ℝ) : ℝ :=
  x * z + 2 * y * z + 4 * z * v + 8 * z * w

theorem maximum_value_le (x y z v w : ℝ) (h : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) :
  maximum_value x y z v w ≤ 3075 * Real.sqrt 17 := 
begin
  sorry
end

theorem maximizing_values (x y z v w : ℝ) (z_val : z = Real.sqrt 1012.5) (h : x = y ∧ y = v ∧ v = w ∧ w = z) :
  maximum_value x y z v w = 3075 * Real.sqrt 17 :=
begin
  sorry
end

theorem combined_value (M x_M y_M z_M v_M w_M : ℝ) (h_max : maximum_value x_M y_M z_M v_M w_M = M)
  (h_eq : x_M = Real.sqrt 1012.5) (h_x : x_M = y_M ∧ y_M = z_M ∧ z_M = v_M ∧ v_M = w_M) :
  M + x_M + y_M + z_M + v_M + w_M = 3075 * Real.sqrt 17 + 5 * Real.sqrt 1012.5 :=
begin
  sorry
end

end maximum_value_le_maximizing_values_combined_value_l421_421350


namespace triangle_incenter_midpoint_eq_distance_l421_421699

theorem triangle_incenter_midpoint_eq_distance
    {A B C : Point}
    (hABC : right_triangle A B C)
    (angleBAC_30 : ∠BAC = 30)
    (S : Point)
    (hS : incenter S A B C)
    (D : Point)
    (hD : midpoint D A B) :
    distance C S = distance D S :=
by
  sorry

end triangle_incenter_midpoint_eq_distance_l421_421699


namespace length_of_NM_is_3_cm_l421_421218

theorem length_of_NM_is_3_cm :
  ∀ (AB AM NM : ℝ) (M N : Point) (AB_total_length : AB = 12)
  (M_midpoint : ∀ A B : Point, Midpoint A B M)
  (N_midpoint : ∀ A M : Point, Midpoint A M N),
  NM = 3 :=
by
  intros AB AM NM M N AB_total_length M_midpoint N_midpoint
  sorry

end length_of_NM_is_3_cm_l421_421218


namespace cookies_on_third_plate_l421_421844

theorem cookies_on_third_plate :
  ∀ (a5 a7 a14 a19 a25 : ℕ),
  (a5 = 5) ∧ (a7 = 7) ∧ (a14 = 14) ∧ (a19 = 19) ∧ (a25 = 25) →
  ∃ (a12 : ℕ), a12 = 12 :=
by
  sorry

end cookies_on_third_plate_l421_421844


namespace degree_at_least_p_minus_1_l421_421235

noncomputable def degree_of_polynomial_at_least (p : ℕ) (q : Polynomial ℤ) : Prop :=
  Prime p ∧ (∀ k : ℕ, q.eval k % ↑p = 0 ∨ q.eval k % ↑p = 1) ∧ 
  q.eval 0 = 0 ∧ q.eval 1 = 1 → q.natDegree ≥ p - 1

-- Statement of the problem
theorem degree_at_least_p_minus_1 (p : ℕ) (q : Polynomial ℤ) : degree_of_polynomial_at_least p q := by
  sorry

end degree_at_least_p_minus_1_l421_421235


namespace compound_interest_principal_l421_421520

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_principal :
  (∃ P : ℝ, compoundInterest P 0.05 1 2 = 8820) ↔ P = 8000 :=
begin
  sorry
end

end compound_interest_principal_l421_421520


namespace rotation_of_OA_eq_OB_l421_421288

theorem rotation_of_OA_eq_OB:
  ∃ (x y : ℝ), ( ∀(OA : ℝ × ℝ), OA = (1, 1) → 
    (∃θ : ℝ, θ = 60 * real.pi / 180 → 
      ∃(OB : ℝ × ℝ), OB = (x, y) ∧ 
        OB.1 = (1 - real.sqrt 3) / 2 ∧ OB.2 = (1 + real.sqrt 3) / 2 ) ) :=
  sorry

end rotation_of_OA_eq_OB_l421_421288


namespace sum_and_product_of_roots_l421_421660

theorem sum_and_product_of_roots (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b = 0 → x = -2 ∨ x = 3) → a + b = -7 :=
by
  sorry

end sum_and_product_of_roots_l421_421660


namespace recreation_percentage_l421_421129

-- Definitions and conditions
variable (W' : ℝ)  -- Last week's wages
def R' : ℝ := 0.10 * W'  -- Last week's recreation amount
def W : ℝ := 0.90 * W'  -- This week's wages
def R : ℝ := 0.40 * W  -- This week's recreation amount

-- Theorem statement
theorem recreation_percentage :
  (R / R') * 100 = 360 := by
-- proof would go here
sorry

end recreation_percentage_l421_421129


namespace sum_sequence_1999_l421_421285

def a (n : ℕ) : ℝ
def sum_first_n_terms (n : ℕ) : ℝ

axiom h_initial_1 : a 1 = 1
axiom h_initial_2 : a 2 = 2
axiom h_recurrence : ∀ n : ℕ, a n * a (n+1) * a (n+2) = a n + a (n+1) + a (n+2)
axiom h_not_equal_1 : ∀ n : ℕ, a (n+1) * a (n+2) ≠ 1

-- The sum S_{1999} is defined as the sum of the first 1999 terms of the sequence a_n
def S_1999 : ℝ := sum_first_n_terms 1999

-- Statement to prove
theorem sum_sequence_1999 : S_1999 = 3997 := by
  sorry

end sum_sequence_1999_l421_421285


namespace misread_number_l421_421802

theorem misread_number (X : ℕ) :
  (average_10_initial : ℕ) = 18 →
  (incorrect_read : ℕ) = 26 →
  (average_10_correct : ℕ) = 22 →
  (10 * 22 - 10 * 18 = X + 26 - 26) →
  X = 66 :=
by sorry

end misread_number_l421_421802


namespace sum_of_reciprocals_l421_421478

theorem sum_of_reciprocals {a b : ℕ} (h_sum: a + b = 55) (h_hcf: Nat.gcd a b = 5) (h_lcm: Nat.lcm a b = 120) :
  1 / (a : ℚ) + 1 / (b : ℚ) = 11 / 120 :=
by
  sorry

end sum_of_reciprocals_l421_421478


namespace ratio_of_areas_l421_421391

-- Definitions for the problem
variable (s : ℝ)

-- We define squares WXYZ and JKLM in terms of their side lengths
def side_length_WXYZ := 4 * s
def side_length_JW := 3 * s
def side_length_WJ := s

-- Calculate the areas
def area_WXYZ := (side_length_WXYZ) ^ 2
def side_length_JK := s * Real.sqrt 2
def area_JKLM := (side_length_JK) ^ 2

-- The theorem to be proved
theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  -- sorry part skips the proof
  sorry

end ratio_of_areas_l421_421391


namespace number_of_girls_l421_421861

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l421_421861


namespace shrub_height_at_end_of_2_years_l421_421160

theorem shrub_height_at_end_of_2_years (h₅ : ℕ) (h : ∀ n : ℕ, 0 < n → 243 = 3^5 * h₅) : ∃ h₂ : ℕ, h₂ = 9 :=
by sorry

end shrub_height_at_end_of_2_years_l421_421160


namespace conjugate_of_complex_num_l421_421042

-- Define the given complex number
def complex_num : ℂ := 5 / (3 + 4 * complex.I)

-- Define the expected result of the conjugate
def expected_result : ℂ := (3 / 5) + (4 / 5) * complex.I

-- State the theorem that conjugate of the given complex number equals the expected result
theorem conjugate_of_complex_num : complex.conj complex_num = expected_result :=
by
  sorry

end conjugate_of_complex_num_l421_421042


namespace intersection_on_circumcircle_l421_421614

open EuclideanGeometry

variables {A B C P Q M N S : Point}
variables (h_triangle : Triangle A B C)
variables (h_acute : acute A B C)
variables (h_PQ_on_BC : OnLine P B C ∧ OnLine Q B C)
variables (h_angles : angle P A B = angle B C A ∧ angle C A Q = angle A B C)
variables (h_midpoints : Midpoint P A M ∧ Midpoint Q A N)

theorem intersection_on_circumcircle (h : ∃ S, Intersection S (Line B M) (Line C N)) : 
  OnCircumcircle S (Circumcircle A B C) :=
by 
  cases h with S h_intersection,
  sorry

end intersection_on_circumcircle_l421_421614


namespace opponent_total_runs_correct_l421_421960

-- Definitions based on the problem's conditions
def game_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def lost_by_one_run (n : ℕ) : Prop := n ∈ [1, 3, 5, 7, 9, 11]

def tie_game (n : ℕ) : Prop := n = 12

def double_score_game (n : ℕ) : Prop := n ∈ [2, 4, 6, 8, 10]

-- Definition of the score of the opponent in a given game
def opponent_score (n : ℕ) : ℕ :=
  if lost_by_one_run n then n + 1
  else if tie_game n then n
  else if double_score_game n then n / 2
  else 0

-- Total runs scored by the opponents
def total_opponent_score : ℕ :=
  (game_scores.map opponent_score).sum

-- Stating the theorem to be proven
theorem opponent_total_runs_correct : total_opponent_score = 69 := by
  sorry

end opponent_total_runs_correct_l421_421960


namespace round_trip_time_l421_421154

def speed_uphill : ℝ := 50
def speed_downhill : ℝ := 100
def total_distance : ℝ := 800

theorem round_trip_time : 
  let D := total_distance / 2 in
  (D / speed_uphill + D / speed_downhill) = 12 :=
by
  let D := total_distance / 2
  have D_def : D = 400 := by sorry
  calc (D / speed_uphill + D / speed_downhill)
    = (400 / 50 + 400 / 100) : by rw [D_def]
    ... = 8 + 4 : by sorry
    ... = 12 : by sorry

end round_trip_time_l421_421154


namespace squares_of_natural_numbers_l421_421019

theorem squares_of_natural_numbers (x y z : ℕ) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) : ∃ a b c : ℕ, x = a^2 ∧ y = b^2 ∧ z = c^2 := 
by
  sorry

end squares_of_natural_numbers_l421_421019


namespace total_trip_duration_correct_l421_421082

-- Definitions based on the conditions provided.
def drivingTime : ℝ := 5 -- Total driving time is 5 hours.
def sectionCount : ℝ := 4 -- Time is split into four equal sections.
def sectionDuration := drivingTime / sectionCount -- Duration of each section.

def factorA := 1.5 * sectionDuration -- Factor A is 1.5 times the duration of the first driving section.
def factorB := 2 * sectionDuration -- Factor B is 2 times the duration of the third driving section.

def trafficJam1Duration := sectionDuration * factorA -- First traffic jam duration.
def trafficJam2Duration := sectionDuration * factorB -- Second traffic jam duration.

def pitStop1Duration := sectionDuration * 0.5 -- Duration of the first pit stop.
def pitStop2Duration := sectionDuration * 0.75 -- Duration of the second pit stop.

def totalTripDuration : ℝ := drivingTime + trafficJam1Duration + trafficJam2Duration + pitStop1Duration + pitStop2Duration

-- The statement to prove.
theorem total_trip_duration_correct : totalTripDuration = 10.9375 := by
  -- The proof would go here.
  sorry

end total_trip_duration_correct_l421_421082


namespace y1_gt_y2_l421_421307

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 4*(-1) + k) 
  (h2 : y2 = 3^2 - 4*3 + k) : 
  y1 > y2 := 
by
  sorry

end y1_gt_y2_l421_421307


namespace parabola_chord_constant_l421_421079

noncomputable def parabola_chord_sum (d : ℝ) : ℝ := 
let PD := sqrt d in  -- Distance from P ( √d, d) to D (0, d)
let QD := sqrt d in  -- Distance from Q (-√d, d) to D (0, d)
1 / PD + 1 / QD

theorem parabola_chord_constant (d : ℝ) (PQ_through_D : ∃ P Q, 
  P.2 = Q.2 ∧ P.2 = d ∧ (P.1^2 = d ∧ Q.1^2 = d) ∧ (P.1 ≠ Q.1) ∧ (0 < d)) : parabola_chord_sum d = 4 :=
by 
  sorry

end parabola_chord_constant_l421_421079


namespace gcd_sum_product_pairwise_coprime_l421_421658

theorem gcd_sum_product_pairwise_coprime 
  (a b c : ℤ) 
  (h1 : Int.gcd a b = 1)
  (h2 : Int.gcd b c = 1)
  (h3 : Int.gcd a c = 1) : 
  Int.gcd (a * b + b * c + a * c) (a * b * c) = 1 := 
sorry

end gcd_sum_product_pairwise_coprime_l421_421658


namespace words_with_at_least_one_consonant_l421_421471

-- Define the letters available and classify them as vowels and consonants
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Define the total number of 5-letter words using the given letters
def total_words : ℕ := 6^5

-- Define the total number of 5-letter words composed exclusively of vowels
def vowel_words : ℕ := 2^5

-- Define the number of 5-letter words that contain at least one consonant
noncomputable def words_with_consonant : ℕ := total_words - vowel_words

-- The theorem to prove
theorem words_with_at_least_one_consonant : words_with_consonant = 7744 := by
  sorry

end words_with_at_least_one_consonant_l421_421471


namespace negation_existence_l421_421430

-- The problem requires showing the equivalence between the negation of an existential
-- proposition and a universal proposition in the context of real numbers.

theorem negation_existence (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) → (∀ x : ℝ, x^2 - m * x - m ≥ 0) :=
by
  sorry

end negation_existence_l421_421430


namespace evaluate_expression_l421_421563

theorem evaluate_expression : 
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  a * b = 63 := 
by
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  sorry

end evaluate_expression_l421_421563


namespace prob_x_plus_y_le_5_l421_421949

theorem prob_x_plus_y_le_5 :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → (x + y ≤ 5 → true) ∧ (x + y > 5 → false)) →
  3 / 8 :=
by sorry

end prob_x_plus_y_le_5_l421_421949


namespace product_of_two_numbers_l421_421057

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l421_421057


namespace hotel_elevator_cubic_value_l421_421999

noncomputable def hotel_elevator_cubic : ℚ → ℚ := sorry

theorem hotel_elevator_cubic_value :
  hotel_elevator_cubic 11 = 11 ∧
  hotel_elevator_cubic 12 = 12 ∧
  hotel_elevator_cubic 13 = 14 ∧
  hotel_elevator_cubic 14 = 15 →
  hotel_elevator_cubic 15 = 13 :=
sorry

end hotel_elevator_cubic_value_l421_421999


namespace no_possible_values_of_k_l421_421532

theorem no_possible_values_of_k : 
  ¬ ∃ k p q : ℕ, (p * p - 79 * p + k = 0) ∧ (Nat.prime p) ∧ (Nat.prime q) ∧ (p + q = 79) := by
  sorry

end no_possible_values_of_k_l421_421532


namespace max_true_statements_maximum_true_conditions_l421_421734

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end max_true_statements_maximum_true_conditions_l421_421734


namespace sheldon_investment_l421_421382

namespace Investment

noncomputable def growth (rate : ℝ) := (1 + rate) ^ 3

theorem sheldon_investment :
  ∀ (total_amount rate_secure rate_high_growth final_amount x : ℝ),
    total_amount = 1500 →
    rate_secure = 0.04 →
    rate_high_growth = 0.06 →
    final_amount = 1824.89 →
    x * growth rate_secure + (total_amount - x) * growth rate_high_growth = final_amount →
    x = 580 :=
begin
  intros total_amount rate_secure rate_high_growth final_amount x,
  assume h1 h2 h3 h4 h5,
  sorry
end

end Investment

end sheldon_investment_l421_421382


namespace max_true_statements_l421_421735

theorem max_true_statements (x y : ℝ) :
  ∀ s : Finset ℕ, ∀ h : s ⊆ {1, 2, 3, 4, 5},
  (∀ i ∈ s, (i = 1 → 1 / x > 1 / y) ∧
            (i = 2 → x^2 < y^2) ∧
            (i = 3 → x > y) ∧
            (i = 4 → x > 0) ∧
            (i = 5 → y > 0)) →
  s.card ≤ 3 := 
begin
  sorry
end

end max_true_statements_l421_421735


namespace g_five_l421_421812

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_one : g 1 = 2

theorem g_five : g 5 = 10 :=
by sorry

end g_five_l421_421812


namespace five_fridays_in_june_l421_421038

theorem five_fridays_in_june (N : ℕ) 
  (h_may_has_five_tuesdays : ∃ ts : list ℕ, ts.length = 5 ∧ ∀ t ∈ ts, t < 32 ∧ nat.pred t % 7 = (3 : ℕ)) 
  (h_may_days : ∀ d, d ∈ (finset.range 31).map (λ x, x + 1)) 
  (h_june_days : ∀ d, d ∈ (finset.range 31).map (λ x, x + 1)) : 
  ∃ fs : list ℕ, fs.length = 5 ∧ ∀ f ∈ fs, f < 32 ∧ nat.pred f % 7 = (4 : ℕ) := 
sorry

end five_fridays_in_june_l421_421038


namespace trajectory_eq_sum_lambdas_l421_421601
-- Define the points M and N
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the conditions
variables {P : ℝ × ℝ} (h : (N.1 - M.1) * (P.1 - M.1) = 6 * Real.sqrt ( (P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2 )) 

-- Prove the trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (hP : (N.1 - M.1) * (P.1 - M.1) = 6 * Real.sqrt ( (P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2 )) :
  (P.1 ^ 2) / 4 + (P.2 ^ 2) / 3 = 1 := 
sorry

-- Define the line passing through N which intersects C
variables {A B H : ℝ × ℝ}
variables (l : ℝ → ℝ) -- equation of line
variables (λ₁ λ₂ : ℝ)
variables (hAN : A = (A.1, A.2 + (1/l A.2)) ∧ (H.1, H.2 - -N.2) = λ₁ * (N.1-A.1, -A.2))
variables (hBN : B = (B.1, B.2 + (1/l B.2)) ∧ (H.1, H.2 - -)\Nep B.2) = λ₂ * (N.1-B.1, -B.2))

-- Prove the concatenation of constants
theorem sum_lambdas (hHNL : (N.1-M.1) * \sqrt ((N.1-N.2) * (H.1-H.2))) :
  λ₁ + λ₂ = -8/3 := 
sorry

end trajectory_eq_sum_lambdas_l421_421601


namespace base_cost_for_toll_l421_421845

theorem base_cost_for_toll (x : ℕ) (t base_cost : ℝ) : 
  t = base_cost + 0.5 * (x - 2) → 
  t = 5 → 
  x = 5 →
  base_cost = 3.5 :=
by {
  intros ht ht5 hx5,
  -- proof goes here
  sorry
}

end base_cost_for_toll_l421_421845


namespace value_of_a_plus_b_l421_421647

variables (a b : ℝ)

theorem value_of_a_plus_b (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : a + b = 12 := 
by
  sorry

end value_of_a_plus_b_l421_421647


namespace sum_first_2017_terms_l421_421275

def f (x : ℝ) : ℝ := (x + 1) / (2 * x - 1)

def a_n (n : ℕ) : ℝ := f (n / 2017)

def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n (i + 1)

theorem sum_first_2017_terms : S_n 2017 = 1010 := sorry

end sum_first_2017_terms_l421_421275


namespace josh_paid_6_dollars_l421_421714

def packs : ℕ := 3
def cheesePerPack : ℕ := 20
def costPerCheese : ℕ := 10 -- cost in cents

theorem josh_paid_6_dollars :
  (packs * cheesePerPack * costPerCheese) / 100 = 6 :=
by
  sorry

end josh_paid_6_dollars_l421_421714


namespace angle_x_first_figure_angle_x_second_figure_l421_421981

-- Problem 1: First Figure
theorem angle_x_first_figure (x : ℝ) (A B C D E F : Type) [parallel : parallel AB ED] 
  (angle_A_B_F : A ∠ B ∠ F = 25) (angle_C_D_F : C ∠ D ∠ F = 55) :
  x = 80 := 
begin
  sorry
end

-- Problem 2: Second Figure
theorem angle_x_second_figure (x : ℝ) (A B C D E F : Type) [parallel : parallel AB ED]
  (angle_A_B_F : A ∠ B ∠ F = 160) (angle_E_D_C : E ∠ D ∠ C = 150) :
  x = 50 := 
begin
  sorry
end

end angle_x_first_figure_angle_x_second_figure_l421_421981


namespace highest_point_difference_l421_421799

theorem highest_point_difference :
  let A := -112
  let B := -80
  let C := -25
  max A (max B C) - min A (min B C) = 87 :=
by
  sorry

end highest_point_difference_l421_421799


namespace negation_is_correct_l421_421889

variable (α : Type*) (S : Set α)

def original_statement := ∀ x ∈ S, |x - 1| ≠ 2

def negation_statement := ∃ x ∈ S, |x - 1| = 2

theorem negation_is_correct : ¬ original_statement ↔ negation_statement :=
by sorry

end negation_is_correct_l421_421889


namespace smallest_product_l421_421016

theorem smallest_product : 
  ∃ a b c d : ℕ, 
    {a, b, c, d} = {1, 2, 3, 4} ∧ 
    (a * 10 + b) * (c * 10 + d) = 312 := 
by
  have h1 : (1 * 10 + 3) * (2 * 10 + 4) = 312 := by norm_num
  existsi 1
  existsi 3
  existsi 2
  existsi 4
  split
  { exact finset.mk [1, 2, 3, 4] finset.nodup.cons:[1, 2, 3, 4] }
  exact h1

end smallest_product_l421_421016


namespace unknown_number_value_l421_421296

theorem unknown_number_value (x n : ℝ) (h1 : 0.75 / x = n / 8) (h2 : x = 2) : n = 3 :=
by
  sorry

end unknown_number_value_l421_421296


namespace number_of_true_propositions_is_2_l421_421323

-- Definitions of the conditions
variables (a b c : Vector) (λ μ : ℝ) (hab_noncollinear : ¬(a = b)) (hλμ_nonzero : λ ≠ 0 ∧ μ ≠ 0)

def prop1 := ¬(LinearIndependent ℝ ![a, b, c]) → Coplanar ![a, b, c]
def prop2 := ¬(∃ (d : Vector), LinearIndependent ℝ ![a, b, d]) → Collinear a b
def prop3 := LinearIndependent ℝ ![a, b, λ • a + μ • b]

-- Proof statement
theorem number_of_true_propositions_is_2 :
  (if prop1 a b c then 1 else 0) + (if prop2 a b then 1 else 0) + (if prop3 a b λ μ then 1 else 0) = 2 :=
by
  sorry

end number_of_true_propositions_is_2_l421_421323


namespace milk_tea_equality_l421_421487

-- Definitions based on the problem conditions
def v : ℝ := sorry  -- volume of a spoonful
def V : ℝ := sorry  -- volume of the glass

-- Initial volumes
def initial_milk_volume : ℝ := V
def initial_tea_volume : ℝ := V

-- Volume after the first transfer
def milk_after_first_transfer : ℝ := V - 3 * v
def tea_after_first_transfer : ℝ := V + 3 * v

-- Volumes after the mixture transfer
def concentration_milk_in_mixture : ℝ := (3 * v) / (V + 3 * v)
def transferred_milk_volume_back : ℝ := 3 * v * concentration_milk_in_mixture
def transferred_tea_volume_back : ℝ := 3 * v * (1 - concentration_milk_in_mixture)

-- Final volumes
def final_milk_volume : ℝ := (V - 3 * v) + transferred_milk_volume_back
def final_tea_volume : ℝ := (V + 3 * v) - transferred_milk_volume_back

theorem milk_tea_equality :
  (initial_milk_volume - (initial_milk_volume - milk_after_first_transfer) + transferred_milk_volume_back)
  = 
  (initial_tea_volume - (initial_tea_volume - tea_after_first_transfer) - transferred_tea_volume_back) :=
by
  sorry

end milk_tea_equality_l421_421487


namespace donna_paid_165_l421_421950

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end donna_paid_165_l421_421950


namespace parallel_lines_intersect_parabola_l421_421096

theorem parallel_lines_intersect_parabola {a k b c x1 x2 x3 x4 : ℝ} 
    (h₁ : x1 < x2) 
    (h₂ : x3 < x4) 
    (intersect1 : ∀ y : ℝ, y = k * x1 + b ∧ y = a * x1^2 ∧ y = k * x2 + b ∧ y = a * x2^2) 
    (intersect2 : ∀ y : ℝ, y = k * x3 + c ∧ y = a * x3^2 ∧ y = k * x4 + c ∧ y = a * x4^2) :
    (x3 - x1) = (x2 - x4) := 
by 
    sorry

end parallel_lines_intersect_parabola_l421_421096


namespace dot_product_calculation_l421_421225

def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (2, 3)
def vec_s : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)
def vec_t : ℝ × ℝ := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_calculation :
  dot_product vec_s vec_t = -9 := by
  sorry

end dot_product_calculation_l421_421225


namespace minimum_distance_AB_l421_421718

noncomputable def circle_center : (ℝ × ℝ) := (2, 3)
noncomputable def circle_radius : ℝ := 2
noncomputable def parabola : (ℝ × ℝ) → Prop := λ (x y : ℝ), y^2 = 8 * x

theorem minimum_distance_AB : ∃ (A B : ℝ × ℝ), 
  ((A.1 - circle_center.1)^2 + (A.2 - circle_center.2)^2 = circle_radius^2) ∧
  parabola B.1 B.2 ∧
  (λ (dist : ℝ), dist = (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))) (0.875) :=
begin
  sorry
end

end minimum_distance_AB_l421_421718


namespace find_a_l421_421880

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end find_a_l421_421880


namespace sum_of_solutions_eq_9_l421_421891

theorem sum_of_solutions_eq_9 (x_1 x_2 : ℝ) (h : x^2 - 9 * x + 20 = 0) :
  x_1 + x_2 = 9 :=
sorry

end sum_of_solutions_eq_9_l421_421891


namespace line_circle_intersection_l421_421185

theorem line_circle_intersection (x y : ℝ) (h1 : 7 * x + 5 * y = 14) (h2 : x^2 + y^2 = 4) :
  ∃ (p q : ℝ), (7 * p + 5 * q = 14) ∧ (p^2 + q^2 = 4) ∧ (7 * p + 5 * q = 14) ∧ (p ≠ q) :=
sorry

end line_circle_intersection_l421_421185


namespace floor_double_l421_421029

theorem floor_double (a : ℝ) (h : 0 < a) : 
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ :=
sorry

end floor_double_l421_421029


namespace sequence_sum_l421_421241

def sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ+, S n = n * n * a n)

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : sequence a S) :
  (∀ n : ℕ+, S n = (2 * n) / (n + 1)) ∧ (∀ n : ℕ+, a n = 2 / (n * (n + 1))) :=
sorry

end sequence_sum_l421_421241


namespace cube_removal_minimum_l421_421512

theorem cube_removal_minimum (l w h : ℕ) (hu : l = 4) (hv : w = 5) (hw : h = 6) :
  ∃ num_cubes_removed : ℕ, 
    (l * w * h - num_cubes_removed = 4 * 4 * 4) ∧ 
    num_cubes_removed = 56 := 
by
  sorry

end cube_removal_minimum_l421_421512


namespace area_region_eq_6_25_l421_421439

noncomputable def area_of_region : ℝ :=
  ∫ x in -0.5..4.5, (5 - |x - 2| - |x - 2|)

theorem area_region_eq_6_25 :
  area_of_region = 6.25 :=
sorry

end area_region_eq_6_25_l421_421439


namespace shape_invariant_under_translation_l421_421159

/-- A shape does not change in size after translation. -/
theorem shape_invariant_under_translation :
  ∀ (s : Type) [has_translation s] (T : translation s),
  invariant s T (size s) :=
by sorry

end shape_invariant_under_translation_l421_421159


namespace percentage_increase_correct_l421_421153

noncomputable def originalWattage : ℕ := 110
noncomputable def newWattage : ℕ := 143

def percentageIncrease (origWatt : ℕ) (newWatt : ℕ) : ℝ :=
  ((newWatt.toReal - origWatt.toReal) / origWatt.toReal) * 100

theorem percentage_increase_correct :
  percentageIncrease originalWattage newWattage = 30 := by
  sorry

end percentage_increase_correct_l421_421153


namespace avg_multiples_of_10_from_10_to_500_l421_421483

theorem avg_multiples_of_10_from_10_to_500 : 
  (∑ i in (finset.range 50).map (λ i, 10 * (i + 1)), i : ℝ) / 50 = 255 :=
by
  sorry

end avg_multiples_of_10_from_10_to_500_l421_421483


namespace inequality_solution_l421_421788

theorem inequality_solution (x : ℝ) (h : x ≠ -5) : 
  (x^2 - 25) / (x + 5) < 0 ↔ x ∈ Set.union (Set.Iio (-5)) (Set.Ioo (-5) 5) := 
by
  sorry

end inequality_solution_l421_421788


namespace quadratic_roots_algebraic_expression_value_l421_421134

-- Part 1: Proof statement for the roots of the quadratic equation
theorem quadratic_roots : (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧ (∀ x : ℝ, x^2 - 4 * x - 3 = 0 → x = x₁ ∨ x = x₂)) :=
by
  sorry

-- Part 2: Proof statement for the algebraic expression value
theorem algebraic_expression_value (a : ℝ) (h : a^2 = 3 * a + 10) :
  (a + 4) * (a - 4) - 3 * (a - 1) = -3 :=
by
  sorry

end quadratic_roots_algebraic_expression_value_l421_421134


namespace ellen_dinner_calories_proof_l421_421554

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l421_421554


namespace simplify_expression_l421_421785

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 :=
by
  sorry

end simplify_expression_l421_421785


namespace eventually_reaches_2011_expected_y_coordinate_2011_l421_421150

-- Define the initial conditions
def fly_movement : ℕ × ℕ → ℕ × ℕ
| (x, y) => if (1 : random ℕ) = 0 then (x + 1, y) else (x, y + 1)

-- Define the proof problem for Part (a)
theorem eventually_reaches_2011 :
  ∃ y, ∃ n, (n > 2010) → (1 / 2) ^ n = 0 → n = 2011 → (2011 = n) :=
sorry

-- Define the proof problem for Part (b)
theorem expected_y_coordinate_2011 :
  ∃ y, (y = 2011 ∧ (2011 : ℕ / 2) = y ) :=
sorry

end eventually_reaches_2011_expected_y_coordinate_2011_l421_421150


namespace find_radius_probability_l421_421963

theorem find_radius_probability :
  ∃ d : ℝ, (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4040 ∧ 0 ≤ y ∧ y ≤ 4040 → 
    (∃ (i j : ℤ), (x - i)^2 + (y - j)^2 ≤ (2 * d)^2) → 
    4 * π * d^2 = 1 / 4 ∧ d ≈ 0.14) := sorry

end find_radius_probability_l421_421963


namespace cricketer_initial_average_l421_421939

def initial_bowling_average
  (runs_for_last_5_wickets : ℝ)
  (decreased_average : ℝ)
  (final_wickets : ℝ)
  (initial_wickets : ℝ)
  (initial_average : ℝ) : Prop :=
  (initial_average * initial_wickets + runs_for_last_5_wickets) / final_wickets =
    initial_average - decreased_average

theorem cricketer_initial_average :
  initial_bowling_average 26 0.4 85 80 12 :=
by
  unfold initial_bowling_average
  sorry

end cricketer_initial_average_l421_421939


namespace parabola_point_b_l421_421156

variable {a b : ℝ}

theorem parabola_point_b (h1 : 6 = 2^2 + 2*a + b) (h2 : -14 = (-2)^2 - 2*a + b) : b = -8 :=
by
  -- sorry as a placeholder for the actual proof.
  sorry

end parabola_point_b_l421_421156


namespace four_digit_integers_with_repeated_digits_l421_421641

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l421_421641


namespace log_5_of_4850_l421_421101

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_5_of_4850 :
  5^5 = 3125 →
  5^6 = 15625 →
  3125 < 4850 →
  4850 < 15625 →
  5 < log_base 5 4850 →
  log_base 5 4850 < 6 →
  Int.round (log_base 5 4850) = 5 :=
by
  intros h5_5 h5_6 h3 h4 h5 h6
  sorry

end log_5_of_4850_l421_421101


namespace range_for_a_l421_421282

theorem range_for_a (f : ℝ → ℝ) (a : ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 8 = 1/4 →
  f (a+1) < f 2 →
  a < -3 ∨ a > 1 :=
by
  intros h1 h2 h3
  sorry

end range_for_a_l421_421282


namespace product_of_two_numbers_l421_421054

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l421_421054


namespace number_of_girls_l421_421852

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l421_421852


namespace average_salary_l421_421481

theorem average_salary (a b c d e : ℕ) (h₁ : a = 8000) (h₂ : b = 5000) (h₃ : c = 15000) (h₄ : d = 7000) (h₅ : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by sorry

end average_salary_l421_421481


namespace equivalent_form_l421_421535

variable {x y z : ℝ}

theorem equivalent_form :
  let P := x + y
  let Q := x - y
  (P + Q + z) / (P - Q - z) - (P - Q - z) / (P + Q + z) = 4 * (x^2 + y^2 + xz) / ((2y - z) * (2x + z)) := by
  sorry

end equivalent_form_l421_421535


namespace back_wheel_revolutions_l421_421761

theorem back_wheel_revolutions (front_diameter back_diameter : ℕ) (front_revolutions : ℕ) (no_slippage : Prop)
  (h1 : front_diameter = 40) (h2 : back_diameter = 20) (h3 : front_revolutions = 150) :
  ∃ back_revolutions : ℕ, back_revolutions = 300 :=
by {
  have front_radius : ℕ := front_diameter / 2,
  have back_radius : ℕ := back_diameter / 2,
  have front_circumference : ℕ := 2 * 3.141592653589793 * front_radius,
  have back_circumference : ℕ := 2 * 3.141592653589793 * back_radius,
  have distance_traveled : ℕ := front_revolutions * front_circumference,
  existsi (distance_traveled / back_circumference),
  sorry 
}

end back_wheel_revolutions_l421_421761


namespace matrix_equation_solution_l421_421657

noncomputable def A : Type* := sorry  -- Placeholder for the type of the matrix A

variables (A : A) [invertible A]

-- defining the condition that A - 3I and A - 5I multipled together equals 0
axiom matrix_equation : (A - 3 * 1) * (A - 5 * 1) = 0

theorem matrix_equation_solution : A + 9 * A⁻¹ = 7 * 1 :=
by 
  sorry -- proof goes here

end matrix_equation_solution_l421_421657


namespace total_packages_l421_421868

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) (h1 : num_trucks = 7) (h2 : packages_per_truck = 70) : num_trucks * packages_per_truck = 490 := by
  sorry

end total_packages_l421_421868


namespace number_of_girls_l421_421854

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l421_421854


namespace megan_files_in_folder_l421_421361

theorem megan_files_in_folder :
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  (total_files / total_folders) = 8.0 :=
by
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  have h1 : total_files = initial_files + added_files := rfl
  have h2 : total_files = 114.0 := by sorry -- 93.0 + 21.0 = 114.0
  have h3 : total_files / total_folders = 8.0 := by sorry -- 114.0 / 14.25 = 8.0
  exact h3

end megan_files_in_folder_l421_421361


namespace average_score_of_class_l421_421479

theorem average_score_of_class (total_students : ℕ)
  (perc_assigned_day perc_makeup_day : ℝ)
  (average_assigned_day average_makeup_day : ℝ)
  (h_total : total_students = 100)
  (h_perc_assigned_day : perc_assigned_day = 0.70)
  (h_perc_makeup_day : perc_makeup_day = 0.30)
  (h_average_assigned_day : average_assigned_day = 55)
  (h_average_makeup_day : average_makeup_day = 95) :
  ((perc_assigned_day * total_students * average_assigned_day + perc_makeup_day * total_students * average_makeup_day) / total_students) = 67 := by
  sorry

end average_score_of_class_l421_421479


namespace max_rational_products_1250_l421_421934
-- Import necessary libraries

-- Define conditions
def is_rational (x : ℝ) : Prop := ∃ a b : ℚ, x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

noncomputable def count_rational_products (rows : Fin 50 → ℝ) (cols : Fin 50 → ℝ) : ℕ :=
  ∑ i in Finset.range 50, ∑ j in Finset.range 50, if is_rational (rows i * cols j) then 1 else 0

-- Define the main theorem
theorem max_rational_products_1250 :
  ∀ (rows cols : Fin 50 → ℝ),
    (∀ i j, rows i ≠ rows j ∧ cols i ≠ cols j) →
    (∃ S : Finset ℝ, S.card = 50 ∧ (∀ x ∈ S, is_rational x) ∧ (∀ y ∉ S, is_irrational y)) →
    count_rational_products rows cols ≤ 1250 :=
by
  sorry

end max_rational_products_1250_l421_421934


namespace cars_will_not_interfere_l421_421095

def car1_speed : ℝ := 60 -- km/h
def car2_speed : ℝ := car1_speed * (6 / 5) -- 1/5 greater than car1_speed
def car1_distance_to_bridge : ℝ := 120 -- km
def car2_distance_to_bridge : ℝ := 180 -- km
def bridge_length : ℝ := 2 -- km

def car1_time_to_bridge : ℝ := car1_distance_to_bridge / car1_speed
def car2_time_to_bridge : ℝ := car2_distance_to_bridge / car2_speed

def car1_distance_after_car2_arrive : ℝ := car1_speed * car2_time_to_bridge

theorem cars_will_not_interfere : car1_distance_after_car2_arrive > car1_distance_to_bridge + bridge_length :=
by
  sorry

end cars_will_not_interfere_l421_421095


namespace collinear_and_parallel_l421_421542

variables {A B C D O P P' Q : Type} [EuclideanGeometrySpace A B C D O P P' Q]

theorem collinear_and_parallel (h : rhomboid A B C D)
    (h_diagonal_intersections : is_diagonal_intersection A B C D O P P' Q) :
  collinear {O, P, P', Q} ∧ parallel_line {O, P, P', Q} (diagonal_line A C) :=
sorry

end collinear_and_parallel_l421_421542


namespace wendy_dentist_bill_l421_421883

theorem wendy_dentist_bill : 
  let cost_cleaning := 70
  let cost_filling := 120
  let num_fillings := 3
  let cost_root_canal := 400
  let cost_dental_crown := 600
  let total_bill := 9 * cost_root_canal
  let known_costs := cost_cleaning + (num_fillings * cost_filling) + cost_root_canal + cost_dental_crown
  let cost_tooth_extraction := total_bill - known_costs
  cost_tooth_extraction = 2170 := by
  sorry

end wendy_dentist_bill_l421_421883


namespace maximum_d_value_l421_421948

theorem maximum_d_value :
  ∃ (d : ℚ), (∀ (stones : list ℚ), (∑ stone in stones, stone ≤ 100) → (∀ stone in stones, stone ≤ 2) → 
  (d = (inf (abs (∑ stone in stones - 10))) → (d = 10 / 11)) :=
begin
  sorry
end

end maximum_d_value_l421_421948


namespace clark_family_ticket_cost_l421_421581

theorem clark_family_ticket_cost
  (regular_price children's_price seniors_price : ℝ)
  (number_youngest_gen number_second_youngest_gen number_second_oldest_gen number_oldest_gen : ℕ)
  (h_senior_discount : seniors_price = 0.7 * regular_price)
  (h_senior_ticket_cost : seniors_price = 7)
  (h_child_discount : children's_price = 0.6 * regular_price)
  (h_number_youngest_gen : number_youngest_gen = 3)
  (h_number_second_youngest_gen : number_second_youngest_gen = 1)
  (h_number_second_oldest_gen : number_second_oldest_gen = 2)
  (h_number_oldest_gen : number_oldest_gen = 1)
  : 3 * children's_price + 1 * regular_price + 2 * seniors_price + 1 * regular_price = 52 := by
  sorry

end clark_family_ticket_cost_l421_421581


namespace area_of_rectangle_l421_421172

-- Define the lengths in meters
def length : ℝ := 1.2
def width : ℝ := 0.5

-- Define the function to calculate the area of a rectangle
def area (l w : ℝ) : ℝ := l * w

-- Prove that the area of the rectangle with given length and width is 0.6 square meters
theorem area_of_rectangle :
  area length width = 0.6 := by
  -- This is just the statement. We omit the proof with sorry.
  sorry

end area_of_rectangle_l421_421172


namespace circumcircles_intersect_on_BC_l421_421353

variable {A B C M N O R X : Type}

noncomputable def midpoint (A B : Type) : Type := sorry -- midpoint definition
noncomputable def circle_diameter_bc : Type := sorry -- circle with diameter BC

axiom is_angle_bisector (P Q R : Type) : Prop -- angle bisector definition
axiom is_circumcircle (P Q R : Type) : Prop -- circumcircle definition
axiom is_on_segment (P segment : Type) : Prop -- point on segment definition

-- Main statement
theorem circumcircles_intersect_on_BC 
  (hABC_non_isosceles_acute : ¬is_isosceles_acuteABC A B C)
  (hM : is_on_circle_diameter_bc M)
  (hN : is_on_circle_diameter_bc N)
  (hO_midpoint : O = midpoint B C)
  (hR_intersect : R = intersection (is_angle_bisector A B C) (is_angle_bisector M O N))
  (hBMR_circumcircle : is_circumcircle B M R)
  (hCNR_circumcircle : is_circumcircle C N R)
  : ∃ (X : Type), is_on_segment X (B, C) ∧ is_on_circumcircle X (B M R) ∧ is_on_circumcircle X (C N R) :=
sorry

end circumcircles_intersect_on_BC_l421_421353


namespace riemann_function_f_l421_421798

noncomputable def RiemannR (x : ℝ) : ℝ :=
if h : ∃ p q : ℕ+, ¬(q % p).val ∧ x = (q.val : ℝ) / (p.val : ℝ)
then 1 / ((h.some_spec.some : ℕ+).val : ℝ)
else if x = 0 ∨ x = 1 then 0 else 0

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 1 then RiemannR x
else if x < 0 then -f (-x)
else -f (1 - x)

theorem riemann_function_f (x : ℝ) (y : ℝ) :
  (∀ x, f(x + 2) = f(x)) →
  f(2023) + f(1011.5) + f(-674.333) = -5 / 6 :=
by sorry

end riemann_function_f_l421_421798


namespace number_of_ages_is_180_l421_421942

-- Conditions: The given digits and the requirement of starting with a prime number
def digits : List ℕ := [1, 1, 2, 3, 7, 9]
def primes : List ℕ := [2, 3, 7]

-- Statement of the theorem
theorem number_of_ages_is_180 (h_digits : digits) (h_primes : primes) : 
  let prime_choices := 3 -- prime numbers from [2, 3, 7]
  let remaining_digits := 5 -- remaining five positions
  let repetitions := 2 -- '1' repeats twice
  (prime_choices * (Finset.univ.card.factorial / (Finset.univ.cardfactorial))) = 180 := 
  sorry

end number_of_ages_is_180_l421_421942


namespace max_balls_of_clay_l421_421887

theorem max_balls_of_clay
  (r : ℝ)
  (side_length : ℝ)
  (volume_ball : ℝ := (4 / 3) * Real.pi * r ^ 3)
  (volume_cube : ℝ := side_length ^ 3):
  (r = 3) →
  (side_length = 8) →
  (⌊volume_cube / volume_ball⌋ = 4) := 
by
  intros hr hside_length
  rw [hr, hside_length]
  have hvc : volume_cube = 8 ^ 3 := by norm_num
  have hvb : volume_ball = (4 / 3) * Real.pi * 3 ^ 3 := by norm_num [Real.pi]
  have hdivision : ⌊8 ^ 3 / ((4 / 3) * Real.pi * 3 ^ 3)⌋ = 4 := by 
    calc ⌊8 ^ 3 / ((4 / 3) * Real.pi * 3 ^ 3)⌋ 
        = ⌊512 / ((4 / 3) * Real.pi * 27)⌋ : by norm_num
    ... = ⌊512 / (36 * Real.pi)⌋ : by norm_num
    ... = ⌊4.527...⌋ : by sorry -- Numerical calculation step here
    ... = 4 : by norm_num
  exact hdivision

end max_balls_of_clay_l421_421887


namespace eccentricity_sum_l421_421255

-- Define the given conditions
variables (F1 F2 P : Type*) [euclidean_geometry] 
variables (a1 a2 c e1 e2 : ℝ)
variables (eccen_ellipse : ∀ point : Type*, (point = P) → eccen F1 P F2 e1)
variables (eccen_hyperbola : ∀ point : Type*, (point = P) → eccen F1 P F2 e2)

-- Assume angle F1PF2 = 2π/3
def angle_F1_P_F2 := (angle F1 P F2 = (2 * π) / 3)

-- Define the proof problem
theorem eccentricity_sum (h : angle_F1_P_F2 F1 F2 P) 
  (he1 : eccentricity_ellipse e1 a1 c) 
  (he2 : eccentricity_hyperbola e2 a2 c) : 
  3 / e1^2 + 1 / e2^2 = 4 := 
sorry

end eccentricity_sum_l421_421255


namespace range_of_m_line_eq_if_circle_passes_origin_l421_421630

namespace ProofProblem

noncomputable def line_eq (m : ℝ) : ℝ → ℝ := λ x, x + m
noncomputable def circle_eq : ℝ × ℝ → ℝ := λ ⟨x, y⟩, x^2 + y^2 - 2*x + 4*y - 4

def intersects_at_two_points (m : ℝ) : Prop :=
    let discriminant := 4 * (m + 1)^2 - 8 * (4 * m - 4)
    discriminant > 0

theorem range_of_m :
    (∀ (m : ℝ), intersects_at_two_points m → m ∈ Set.Ioo (-3 - 3 * Real.sqrt 2) (-3 + 3 * Real.sqrt 2)) :=
sorry

theorem line_eq_if_circle_passes_origin (m : ℝ) :
    (intersects_at_two_points m) →
    (let A B : ℝ × ℝ := sorry, sorry -- Omitted the explicit points A and B for brevity
     let circle_with_diameter_AB : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x - A.1) * (x - B.1) + (y - A.2) * (y - B.2) = 0
     circle_with_diameter_AB (0, 0) → (line_eq m = λ x, x - 4) ∨ (line_eq m = λ x, x + 1)) :=
sorry

end ProofProblem

end range_of_m_line_eq_if_circle_passes_origin_l421_421630


namespace total_cases_l421_421334

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l421_421334


namespace bob_wins_strategy_l421_421364

-- Define the alternating sequence condition
def alternating_sequence (n : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ i, i < n → (if even i then f i = 0 else f i = 1)

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a*a + b*b = n

-- Theorem: Bob has a winning strategy
theorem bob_wins_strategy : ∀ (f : ℕ → ℕ), alternating_sequence 4042 f →
                       ¬ is_sum_of_two_squares (nat.binary_to_nat 4042 f) :=
by
  sorry

end bob_wins_strategy_l421_421364


namespace truncated_cone_surface_area_l421_421167

theorem truncated_cone_surface_area (R r : ℝ) (S : ℝ)
  (h1: S = 4 * Real.pi * (R^2 + R * r + r^2)) :
  2 * Real.pi * (R^2 + R * r + r^2) = S / 2 :=
by
  sorry

end truncated_cone_surface_area_l421_421167


namespace find_length_NY_l421_421328

-- Define the entities and given lengths
def triangle_XYZ (X Y Z : Type) (XY : ℝ) (XZ : Type) (YZ : Type) : Prop :=
  XY = 10

def line_MNP_parallel_XY (MNP XY : Type) : Prop := sorry

def segment_MN (MN : ℝ) : Prop :=
  MN = 6

def XM_bisects_PNY (XM PNY : Type) : Prop := sorry

-- Final proof problem statement
theorem find_length_NY
  (X Y Z M N P : Type)
  (XY : ℝ) (MN : ℝ) (NY : ℝ)
  (line_parallel : line_MNP_parallel_XY MNP XY)
  (triangleXYZ : triangle_XYZ X Y Z XY (XZ : Type) (YZ : Type))
  (segmentMN : segment_MN MN)
  (bisect : XM_bisects_PNY XM (PNY : Type)) :
  NY = 15 :=
sorry

end find_length_NY_l421_421328


namespace kangaroo_meetings_l421_421878

/-- 
Two kangaroos, A and B, start at point A and jump in specific sequences:
- Kangaroo A jumps in the sequence A, B, C, D, E, F, G, H, I, A, B, C, ... in a loop every 9 jumps.
- Kangaroo B jumps in the sequence A, B, D, E, G, H, A, B, D, ... in a loop every 6 jumps.
They start at point A together. Prove that they will land on the same point 226 times after 2017 jumps.
-/
theorem kangaroo_meetings (n : Nat) (ka : Fin 9 → Fin 9) (kb : Fin 6 → Fin 6)
  (hka : ∀ i, ka i = (i + 1) % 9) (hkb : ∀ i, kb i = (i + 1) % 6) :
  n = 2017 →
  -- Prove that the two kangaroos will meet 226 times after 2017 jumps
  ∃ k, k = 226 :=
by
  sorry

end kangaroo_meetings_l421_421878


namespace roots_real_distinct_l421_421739

noncomputable def P : ℕ → (ℝ → ℝ)
| 0     := λ x, x
| (n+1) := λ x, P n (λ x, x^2 - 2) x 

theorem roots_real_distinct (n : ℕ) :
  ∀ n, ∃ s : finset ℝ, (∀ x ∈ s, (P n x) = x) ∧ (∀ x y ∈ s, x ≠ y → x ≠ y) :=
sorry

end roots_real_distinct_l421_421739


namespace length_of_DB_l421_421321

/-- Statement of the problem with conditions and conclusion. -/
theorem length_of_DB :
  ∀ (A B C D : Type)
    (d_AC : ℝ) (d_AD : ℝ)
    (angle_ABC : Real.angle) (angle_ADB : Real.angle),
    angle_ABC = Real.pi / 2 ∧ angle_ADB = Real.pi / 2 ∧
    d_AC = 15.7 ∧ d_AD = 4.5 →
    ∃ d_DB : ℝ, d_DB = 7.1 :=
begin
  intros A B C D d_AC d_AD angle_ABC angle_ADB,
  rintro ⟨h1, h2, h3, h4⟩,
  /- The detailed proof would go here -/
  sorry
end

end length_of_DB_l421_421321


namespace find_111th_digit_in_fraction_l421_421102

theorem find_111th_digit_in_fraction :
  let frac : ℚ := 33 / 555
  (decimal_rep : String) := "0.0overline594"
  (repeating_cycle_len : ℕ := 3)
  (position_mod : ℕ := 110 % repeating_cycle_len)
  (digit := "594".nth (position_mod).getD '0')
in digit = '9' :=
by
  sorry

end find_111th_digit_in_fraction_l421_421102


namespace find_equation_line_AB_find_equation_median_AD_l421_421329

noncomputable def point := ℝ × ℝ
noncomputable def line := ℝ × ℝ × ℝ

def B : point := (3, 4)
def C : point := (5, 2)
def line_AC : line := (1, -4, 3)
def altitude_A_to_AB : line := (2, 3, -16)

-- Question 1: Equation of the line containing side AB
def equation_line_AB (l : line) : Prop :=
  ∃ m b, l = (m, -1, b) ∧ 3 * (fst l - 3) - 2 * (snd l - 4) = 0

-- Question 2: Equation of the line containing the median from A to side AC
def equation_median_AD (l : line) : Prop :=
  ∃ A D : point,
    (1, 1) ∉ {B, C} ∧ D = ((fst B + fst C)/2, (snd B + snd C)/2) ∧
    l = (fst A - fst D, snd A - snd D, 0)

-- Proofs to be filled in
theorem find_equation_line_AB : ∃ l, equation_line_AB l := sorry
theorem find_equation_median_AD : ∃ l, equation_median_AD l := sorry

end find_equation_line_AB_find_equation_median_AD_l421_421329


namespace sculptor_needs_blocks_l421_421516

noncomputable def volume_cylinder (r h : ℝ) : ℝ := real.pi * r^2 * h
noncomputable def volume_block (l w h : ℝ) : ℝ := l * w * h
noncomputable def blocks_required (v_trophy v_block : ℝ) : ℕ := (v_trophy / v_block).ceil.to_nat

theorem sculptor_needs_blocks
  (h_cylinder r_cylinder : ℝ)
  (l_block w_block h_block : ℝ)
  (H_pos : h_cylinder > 0)
  (R_pos : r_cylinder > 0)
  (L_pos : l_block > 0)
  (W_pos : w_block > 0)
  (H_b_pos : h_block > 0)
  (H_cylinder : h_cylinder = 10)
  (R_cylinder : r_cylinder = 3)
  (L_block : l_block = 8)
  (W_block : w_block = 3)
  (H_block : h_block = 2) :
  blocks_required (volume_cylinder r_cylinder h_cylinder) (volume_block l_block w_block h_block) = 6 :=
sorry

end sculptor_needs_blocks_l421_421516


namespace inequality_problem_l421_421725

theorem inequality_problem 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
by sorry

end inequality_problem_l421_421725


namespace Andrena_more_than_Debelyn_l421_421998

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l421_421998


namespace sum_of_powers_inequality_l421_421610

theorem sum_of_powers_inequality {n : ℕ} {a : Fin n → ℕ} (h_distinct: Function.Injective a) (h_pos: ∀ i, 0 < a i) :
    (∑ i, (a i) ^ 7) + (∑ i, (a i) ^ 5) ≥ 2 * (∑ i, (a i) ^ 3) ^ 2 := 
sorry

end sum_of_powers_inequality_l421_421610


namespace inequality_solution_I_inequality_solution_II_l421_421136

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| - |x + 1|

theorem inequality_solution_I (x : ℝ) : f x 1 > 2 ↔ x < -2 / 3 ∨ x > 4 :=
sorry 

noncomputable def g (x a : ℝ) : ℝ := f x a + |x + 1| + x

theorem inequality_solution_II (a : ℝ) : (∀ x, g x a > a ^ 2 - 1 / 2) ↔ (-1 / 2 < a ∧ a < 1) :=
sorry

end inequality_solution_I_inequality_solution_II_l421_421136


namespace length_of_diagonal_l421_421196

theorem length_of_diagonal (h1 h2 area : ℝ) (h1_val : h1 = 7) (h2_val : h2 = 3) (area_val : area = 50) :
  ∃ d : ℝ, d = 10 :=
by
  sorry

end length_of_diagonal_l421_421196


namespace tom_strokes_over_par_l421_421083

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l421_421083


namespace cab_driver_income_l421_421494

theorem cab_driver_income (incomes : Fin 5 → ℝ)
  (h1 : incomes 0 = 250)
  (h2 : incomes 1 = 400)
  (h3 : incomes 2 = 750)
  (h4 : incomes 3 = 400)
  (avg_income : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 460) : 
  incomes 4 = 500 :=
sorry

end cab_driver_income_l421_421494


namespace find_roots_l421_421203

noncomputable def polynomial_roots : set ℝ :=
  {((1 - Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 - Real.sqrt 43 - 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 - 2 * Real.sqrt 34) / 6)}

theorem find_roots (x : ℝ) :
  (3 * x ^ 4 + 2 * x ^ 3 - 8 * x ^ 2 + 2 * x + 3 = 0) ↔ (x ∈ polynomial_roots) :=
by sorry

end find_roots_l421_421203


namespace abs_eq_condition_l421_421909

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421909


namespace joes_speed_second_part_l421_421340

theorem joes_speed_second_part
  (d1 d2 t1 t_total: ℝ)
  (s1 s_avg: ℝ)
  (h_d1: d1 = 420)
  (h_d2: d2 = 120)
  (h_s1: s1 = 60)
  (h_s_avg: s_avg = 54) :
  (d1 / s1 + d2 / (d2 / 40) = t_total ∧ t_total = (d1 + d2) / s_avg) →
  d2 / (t_total - d1 / s1) = 40 :=
by
  sorry

end joes_speed_second_part_l421_421340


namespace expression_divisible_by_1999_l421_421021

theorem expression_divisible_by_1999 : 
  let E := (1 * 3) - (List.product (List.filter odd (List.rangeWith 5 (1997-5+1) 2))) 
    + (List.product (List.filter even (List.rangeWith 2 (1998-2+1) 2)))
  in E % 1999 = 0 := 
by
  let E := (1 * 3) - (List.product (List.filter odd (List.rangeWith 5 (1997-5+1) 2))) 
    + (List.product (List.filter even (List.rangeWith 2 (1998-2+1) 2)))
  sorry

end expression_divisible_by_1999_l421_421021


namespace product_csc_sec_eqn_pq_sum_l421_421475

noncomputable def complex_csc_square_sec_square_prod : ℤ := 
  ∑ k in (finset.range 1 31), 
    ((csc (3 * k : ℝ)).square) * ((sec (6 * k : ℝ)).square)

theorem product_csc_sec_eqn :
  complex_csc_square_sec_square_prod = 2^60 := sorry

theorem pq_sum :
  2 + 60 = 62 := 
by sorry

end product_csc_sec_eqn_pq_sum_l421_421475


namespace find_remainder_l421_421207

def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 1

theorem find_remainder : p 2 = 41 :=
by sorry

end find_remainder_l421_421207


namespace jessica_rearrangements_fraction_l421_421339

noncomputable def jessica_name : String := "Jessica Brown"

def letter_count (name : String) : ℕ := 12 -- Since it's given in the problem that the count is 12 for simplicity

def arrangements_per_minute : ℕ := 10

def calculate_hours (total_permutations : ℕ) (arr_per_minute : ℕ) : ℕ :=
  (total_permutations / arr_per_minute) / 60

theorem jessica_rearrangements_fraction :
  let total_permutations := Nat.factorial (letter_count jessica_name) in
  calculate_hours total_permutations arrangements_per_minute = 798336 :=
by
  sorry

end jessica_rearrangements_fraction_l421_421339


namespace riemann_function_f_l421_421797

noncomputable def RiemannR (x : ℝ) : ℝ :=
if h : ∃ p q : ℕ+, ¬(q % p).val ∧ x = (q.val : ℝ) / (p.val : ℝ)
then 1 / ((h.some_spec.some : ℕ+).val : ℝ)
else if x = 0 ∨ x = 1 then 0 else 0

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 1 then RiemannR x
else if x < 0 then -f (-x)
else -f (1 - x)

theorem riemann_function_f (x : ℝ) (y : ℝ) :
  (∀ x, f(x + 2) = f(x)) →
  f(2023) + f(1011.5) + f(-674.333) = -5 / 6 :=
by sorry

end riemann_function_f_l421_421797


namespace sum_tan_alpha_product_eq_neg7_l421_421260

theorem sum_tan_alpha_product_eq_neg7 (z : ℂ)
  (roots : Fin 7 → ℂ)
  (args : Fin 7 → ℝ) :
  (∀ i, roots i ^ 7 = z)
  → z = 2021 + Complex.i
  → (∀ i, args (Fin.mk i sorry) = Complex.arg (roots (Fin.mk i sorry)))
  → (∀ i, args (Fin.mk i sorry) < args (Fin.mk (i + 1) sorry))
  → ∑ i in Finset.range 7, Real.tan (args (Fin.mk i sorry)) * Real.tan (args (Fin.mk (i + 2) sorry)) = -7 := by
  sorry

end sum_tan_alpha_product_eq_neg7_l421_421260


namespace abs_eq_condition_l421_421906

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421906


namespace min_product_of_prime_triplet_l421_421613

theorem min_product_of_prime_triplet
  (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1)
  (h1 : x ∣ (y^5 + 1)) (h2 : y ∣ (z^5 + 1)) (h3 : z ∣ (x^5 + 1)) :
  (x * y * z) = 2013 := by
  sorry

end min_product_of_prime_triplet_l421_421613


namespace louisa_average_speed_l421_421009

theorem louisa_average_speed :
  ∃ v : ℝ, (350 / v) - (200 / v) = 3 ∧ v = 50 :=
by
  use 50
  split
  . finish
  . rfl

end louisa_average_speed_l421_421009


namespace prob_not_same_group_l421_421445

variable {A B : Type}

/-- Define an event E where students A and B are in the same group.
    The probability of E can be calculated directly from the given conditions. -/
def prob_same_group (n : ℕ) : ℚ :=
  1 / n

/-- Define the main theorem: the probability that students A and B are not in the same group. -/
theorem prob_not_same_group (n : ℕ) (h: n = 3) :
  1 - prob_same_group n = 2 / 3 :=
by
  rw [h, prob_same_group]
  sorry

end prob_not_same_group_l421_421445


namespace cylinder_volume_is_6pi_l421_421262

-- Defining the problem using Lean formalism
noncomputable def volume_of_cylinder (h : ℝ) (R : ℝ) (r : ℝ) : ℝ :=
  π * r^2 * h

theorem cylinder_volume_is_6pi :
  let h := 2
  let R := 2  -- radius of the sphere with diameter 4
  let r := sqrt (R^2 - (h / 2)^2)
  volume_of_cylinder h R r = 6 * π := by
  sorry

end cylinder_volume_is_6pi_l421_421262


namespace olafs_dad_points_l421_421001

-- Let D be the number of points Olaf's dad scored.
def dad_points : ℕ := sorry

-- Olaf scored three times more points than his dad.
def olaf_points (dad_points : ℕ) : ℕ := 3 * dad_points

-- Total points scored is 28.
def total_points (dad_points olaf_points : ℕ) : Prop := dad_points + olaf_points = 28

theorem olafs_dad_points (D : ℕ) :
  (D + olaf_points D = 28) → (D = 7) :=
by
  sorry

end olafs_dad_points_l421_421001


namespace contrapositive_proposition_l421_421416

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → tan α = 1) ↔ (tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end contrapositive_proposition_l421_421416


namespace car_initial_time_l421_421145

variable (t : ℝ)

theorem car_initial_time (h : 80 = 720 / (3/2 * t)) : t = 6 :=
sorry

end car_initial_time_l421_421145


namespace max_terms_eq_2022_l421_421242

noncomputable def max_terms_sequence (a : ℕ → ℝ) : ℕ :=
  ∃ n, a 1 = 3 ∧ a 2 = 46 ∧ 
  (∀ k ≥ 1, a (k + 2) = real.sqrt (a (k + 1) * a k - π / a (k + 1))) ∧ 
  a n ≥ 0

theorem max_terms_eq_2022 : max_terms_sequence = 2022 := 
sorry

end max_terms_eq_2022_l421_421242


namespace free_endpoints_can_be_1001_l421_421239

variables (initial_segs : ℕ) (total_free_ends : ℕ) (k : ℕ)

-- Initial setup: one initial segment.
def initial_segment : ℕ := 1

-- Each time 5 segments are drawn from a point, the number of free ends increases by 4.
def free_ends_after_k_actions (k : ℕ) : ℕ := initial_segment + 4 * k

-- Question: Can the number of free endpoints be exactly 1001?
theorem free_endpoints_can_be_1001 : free_ends_after_k_actions 250 = 1001 := by
  sorry

end free_endpoints_can_be_1001_l421_421239


namespace root_of_quadratic_is_4_l421_421349

noncomputable theory

open Real

theorem root_of_quadratic_is_4 (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) 
  (h4 : ∃ d, y = x - d ∧ z = x - 2 * d)
  (h5 : ∃ r, ∀ t, y * t^2 + z * t + y = 0 ↔ t = r) : 
  r = 4 := 
sorry

end root_of_quadratic_is_4_l421_421349


namespace kaleb_toys_l421_421715

def initial_savings : ℕ := 21
def allowance : ℕ := 15
def cost_per_toy : ℕ := 6

theorem kaleb_toys : (initial_savings + allowance) / cost_per_toy = 6 :=
by
  sorry

end kaleb_toys_l421_421715


namespace min_product_of_digits_1_2_3_4_l421_421013

theorem min_product_of_digits_1_2_3_4 : 
  let nums := (1, 2, 3, 4) in
  ∃ (a b c d: ℕ), 
    (a ∈ nums) ∧ (b ∈ nums) ∧ (c ∈ nums) ∧ (d ∈ nums) ∧ 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
    (b ≠ c) ∧ (b ≠ d) ∧
    (c ≠ d) ∧
    (a * 10 + b) * (c * 10 + d) = 312 ∧
    ((a * 10 + b) = 13 ∧ (c * 10 + d) = 24 ∨ 
     (a * 10 + b) = 24 ∧ (c * 10 + d) = 13) :=
  sorry

end min_product_of_digits_1_2_3_4_l421_421013


namespace math_competition_l421_421957

noncomputable def ξ : ℝ → Prop := sorry -- Placeholder for the normal distribution N(100, σ^2)

def condition1 (ξ : ℝ → Prop) : Prop := ∀ x, P (ξ x) = normalDist 100 σ^2 x
def condition2 (a : ℝ) := P (ξ ≥ 120) = a
def condition3 (b : ℝ) := P (80 < ξ ≤ 100) = b 

theorem math_competition (a b : ℝ) (ξ : ℝ → Prop)
    (h1: condition1 ξ)
    (h2: condition2 a)
    (h3: condition3 b)
    : a + b = 0.5 :=
by
  sorry

end math_competition_l421_421957


namespace angle_Q_of_regular_hexagon_l421_421030

theorem angle_Q_of_regular_hexagon
  (A B C D E F Q: Type)
  (h_hex : regular_hexagon A B C D E F)
  (h_meet : extends_meet_at Q CD AB): 
  angle_measure Q = 60 := 
sorry

end angle_Q_of_regular_hexagon_l421_421030


namespace sum_cosec_identity_l421_421770

theorem sum_cosec_identity (n : ℕ) (h_odd : n % 2 = 1) (h_gt_one : n > 1):
  ∑ m in Finset.range (n - 1), 1 / (Real.sin (m * Real.pi / n))^2 = (n^2 - 1) / 3 := 
by sorry

end sum_cosec_identity_l421_421770


namespace prob_green_ball_is_correct_l421_421184

/-- Define the probabilities involved in selecting a green ball from each container given the problem conditions. -/
def prob_green_I := 6 / 16
def prob_green_II := 5 / 8
def prob_green_III := 6 / 8
def prob_green_IV := 4 / 8

/-- Define the probability of selecting any container. -/
def prob_container := 1 / 4

/-- Define the total probability of selecting a green ball from a randomly chosen container. -/
def total_prob_green_ball : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III +
  prob_container * prob_green_IV

/-- The main theorem to prove that the total probability of selecting a green ball is 9/16. -/
theorem prob_green_ball_is_correct : total_prob_green_ball = 9 / 16 :=
begin
  sorry
end

end prob_green_ball_is_correct_l421_421184


namespace john_total_payment_and_brother_ratio_l421_421711

def total_fee (court_hours prep_hours : ℕ) (hourly_rate upfront_fee : ℕ) : ℕ :=
  upfront_fee + court_hours * hourly_rate + prep_hours * hourly_rate

theorem john_total_payment_and_brother_ratio :
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let john_total_payment := 8000
  let total_payment := total_fee court_hours prep_hours hourly_rate upfront_fee
  let john_remaining_payment := john_total_payment - upfront_fee
  let brother_payment := total_payment - john_total_payment
  let ratio := brother_payment / brother_payment : total_payment / brother_payment = 1 : 2 
  john_remaining_payment = 7000 ∧ ratio = 1 / 2 := 
  by
  sorry

end john_total_payment_and_brother_ratio_l421_421711


namespace number_of_solutions_l421_421249

-- Defining the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

-- Statement of the problem
theorem number_of_solutions (a : ℝ) : {a : ℝ | A a ∪ B a = A a}.card = 1 :=
by
  sorry

end number_of_solutions_l421_421249


namespace parking_lot_perimeter_l421_421513

theorem parking_lot_perimeter (a b : ℝ) (h₁ : a^2 + b^2 = 625) (h₂ : a * b = 168) :
  2 * (a + b) = 62 :=
sorry

end parking_lot_perimeter_l421_421513


namespace secretary_worked_longest_l421_421923

theorem secretary_worked_longest
  (h1 : ∀ (x : ℕ), 3 * x + 5 * x + 7 * x + 11 * x = 2080)
  (h2 : ∀ (a b c d : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x ∧ d = 11 * x → d = 11 * x):
  ∃ y : ℕ, y = 880 :=
by
  sorry

end secretary_worked_longest_l421_421923


namespace total_ticket_sales_l421_421080

def ticket_price : Type := 
  ℕ → ℕ

def total_individual_sales (student_count adult_count child_count senior_count : ℕ) (prices : ticket_price) : ℝ :=
  (student_count * prices 6 + adult_count * prices 8 + child_count * prices 4 + senior_count * prices 7)

def total_group_sales (group_student_count group_adult_count group_child_count group_senior_count : ℕ) (prices : ticket_price) : ℝ :=
  let total_price := (group_student_count * prices 6 + group_adult_count * prices 8 + group_child_count * prices 4 + group_senior_count * prices 7)
  if (group_student_count + group_adult_count + group_child_count + group_senior_count) > 10 then 
    total_price - 0.10 * total_price 
  else 
    total_price

theorem total_ticket_sales
  (prices : ticket_price)
  (student_count adult_count child_count senior_count : ℕ)
  (group_student_count group_adult_count group_child_count group_senior_count : ℕ)
  (total_sales : ℝ) :
  student_count = 20 →
  adult_count = 12 →
  child_count = 15 →
  senior_count = 10 →
  group_student_count = 5 →
  group_adult_count = 8 →
  group_child_count = 10 →
  group_senior_count = 9 →
  prices 6 = 6 →
  prices 8 = 8 →
  prices 4 = 4 →
  prices 7 = 7 →
  total_sales = (total_individual_sales student_count adult_count child_count senior_count prices) + (total_group_sales group_student_count group_adult_count group_child_count group_senior_count prices) →
  total_sales = 523.30 := by
  sorry

end total_ticket_sales_l421_421080


namespace total_coronavirus_cases_l421_421333

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l421_421333


namespace required_bandwidth_l421_421932

/-- Given the session duration in minutes, sampling rate in Hz, bit depth in bits,
    and metadata volume in bytes per 5 KB of audio, prove that the required bandwidth
    of this channel in kilobits per second for stereo audio signals is 2.25 Kbit/s.
-/
theorem required_bandwidth (session_duration_min : ℕ) (sampling_rate_hz : ℕ) (bit_depth_bits : ℕ)
(metadata_volume_bytes : ℕ) (stereo_multiplier : ℕ) :
   session_duration_min = 51 →
   sampling_rate_hz = 63 →
   bit_depth_bits = 17 →
   metadata_volume_bytes = 47 →
   stereo_multiplier = 2 →
   let session_duration_sec := session_duration_min * 60 in
   let data_volume_bits := sampling_rate_hz * bit_depth_bits * session_duration_sec in
   let metadata_volume_bits := (metadata_volume_bytes * 8 * data_volume_bits) / (5 * 1024) in
   let total_data_volume_bits := (data_volume_bits + metadata_volume_bits) * stereo_multiplier in
   let throughput_kbps := total_data_volume_bits / (session_duration_sec * 1024) in
   throughput_kbps = 2.25 :=
begin
  intros,
  let session_duration_sec := session_duration_min * 60,
  let data_volume_bits := sampling_rate_hz * bit_depth_bits * session_duration_sec,
  let metadata_volume_bits := (metadata_volume_bytes * 8 * data_volume_bits) / (5 * 1024),
  let total_data_volume_bits := (data_volume_bits + metadata_volume_bits) * stereo_multiplier,
  let throughput_kbps := total_data_volume_bits / (session_duration_sec * 1024),
  sorry
end

end required_bandwidth_l421_421932


namespace current_selling_price_is_correct_profit_per_unit_is_correct_l421_421954

variable (a : ℝ)

def original_selling_price (a : ℝ) : ℝ :=
  a * 1.22

def current_selling_price (a : ℝ) : ℝ :=
  original_selling_price a * 0.85

def profit_per_unit (a : ℝ) : ℝ :=
  current_selling_price a - a

theorem current_selling_price_is_correct : current_selling_price a = 1.037 * a :=
by
  unfold current_selling_price original_selling_price
  sorry

theorem profit_per_unit_is_correct : profit_per_unit a = 0.037 * a :=
by
  unfold profit_per_unit current_selling_price original_selling_price
  sorry

end current_selling_price_is_correct_profit_per_unit_is_correct_l421_421954


namespace correct_value_of_wrongly_read_number_l421_421041

def average_problem (x : ℕ) : Prop :=
  let wrong_sum := 180
  let correct_sum := 190
  let wrongly_read_number := 26
  correct_sum - wrong_sum = 10 ∧ x = wrongly_read_number + 10

theorem correct_value_of_wrongly_read_number : average_problem 36 :=
by
  let wrong_sum := 180
  let correct_sum := 190
  let wrongly_read_number := 26
  have : correct_sum - wrong_sum = 10 := by simp [correct_sum, wrong_sum]
  have : 36 = wrongly_read_number + 10 := by simp [wrongly_read_number]
  exact ⟨this, by simp⟩
  sorry

end correct_value_of_wrongly_read_number_l421_421041


namespace quadrilateral_EFGH_EH_is_24_l421_421680

theorem quadrilateral_EFGH_EH_is_24 (E F G H : Type) [MetricSpace E]
  (hEF : dist E F = 7) (hFG : dist F G = 21) (hGH : dist G H = 7)
  (angleHEF : ∀ (p q r : E), right_angle (angle p q r)) :
  ∃ (EH : ℕ), EH = 24 :=
by
  sorry

end quadrilateral_EFGH_EH_is_24_l421_421680


namespace oliver_first_coupon_redeem_on_friday_l421_421002

-- Definitions of conditions in the problem
def has_coupons (n : ℕ) := n = 8
def uses_coupon_every_9_days (days : ℕ) := days = 9
def is_closed_on_monday (day : ℕ) := day % 7 = 1  -- Assuming 1 represents Monday
def does_not_redeem_on_closed_day (redemption_days : List ℕ) :=
  ∀ day ∈ redemption_days, day % 7 ≠ 1

-- Main theorem statement
theorem oliver_first_coupon_redeem_on_friday : 
  ∃ (first_redeem_day: ℕ), 
  has_coupons 8 ∧ uses_coupon_every_9_days 9 ∧
  is_closed_on_monday 1 ∧ 
  does_not_redeem_on_closed_day [first_redeem_day, first_redeem_day + 9, first_redeem_day + 18, first_redeem_day + 27, first_redeem_day + 36, first_redeem_day + 45, first_redeem_day + 54, first_redeem_day + 63] ∧ 
  first_redeem_day % 7 = 5 := sorry

end oliver_first_coupon_redeem_on_friday_l421_421002


namespace symmetric_points_sum_l421_421304

-- Define the conditions
def points_symmetric_about_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ P.2 = -Q.2

-- Given points
def P : ℝ × ℝ := (-2, b)
def Q : ℝ × ℝ := (a, -3)

-- Proof statement
theorem symmetric_points_sum (a b : ℝ) (h : points_symmetric_about_x_axis P Q) : a + b = 1 :=
  by sorry

end symmetric_points_sum_l421_421304


namespace find_a_find_range_of_g_l421_421626

-- Definition and conditions
def f (a : ℝ) (x : ℝ) := a^(x - 1)
def g (a : ℝ) (x : ℝ) := a^(2*x) - a^(x-2) + 8

-- Lean statement for the first question
theorem find_a (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : f a 2 = 1/2) : a = 1/2 :=
  sorry

-- Lean statement for the second question
theorem find_range_of_g (a : ℝ) (x : ℝ) (h₀ : a = 1/2) (h₁ : -2 ≤ x ∧ x ≤ 1) :
  4 ≤ g a x ∧ g a x ≤ 8 :=
  sorry

end find_a_find_range_of_g_l421_421626


namespace amusement_park_people_l421_421936

theorem amusement_park_people (students adults free : ℕ) (total_people paid : ℕ) :
  students = 194 →
  adults = 235 →
  free = 68 →
  total_people = students + adults →
  paid = total_people - free →
  paid - free = 293 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end amusement_park_people_l421_421936


namespace max_extra_credit_students_l421_421663

noncomputable def largest_extra_credit_students (N : ℕ) (scores : Fin N → ℝ) : ℕ :=
  let mean := (∑ i, scores i) / N
  ∑ i, if scores i > mean then 1 else 0

theorem max_extra_credit_students (N : ℕ) (scores : Fin N → ℝ)
  (hN : N = 120) (h_scores : ∃ i, ∀ j ≠ i, scores j = 10 ∧ scores i = 0) :
  largest_extra_credit_students N scores = 119 := by
  sorry

end max_extra_credit_students_l421_421663


namespace find_x_l421_421897

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421897


namespace coeff_x2_in_product_l421_421461

def P (x : ℚ) : ℚ := x^5 - 2 * x^4 + 4 * x^3 - 5 * x + 2
def Q (x : ℚ) : ℚ := 3 * x^4 - x^3 + x^2 + 4 * x - 1

theorem coeff_x2_in_product : coefficient (P * Q) 2 = -18 :=
by sorry

end coeff_x2_in_product_l421_421461


namespace problem_statement_l421_421589

def f (x : ℝ) : ℝ := x^2 + 2 * x * f' 1

theorem problem_statement :
  let f' (x : ℝ) : ℝ := 2 * x + 2 * f' 1 in
  f 1 = - 3 :=
by
  sorry

end problem_statement_l421_421589


namespace Oleg_older_than_Ekaterina_oldest_is_Roman_and_married_to_Zhanna_l421_421024

def Roman : ℕ := sorry
def Oleg : ℕ := sorry
def Ekaterina : ℕ := sorry
def Zhanna : ℕ := sorry

axiom diff_ages : Roman ≠ Oleg ∧ Roman ≠ Ekaterina ∧ Roman ≠ Zhanna ∧ Oleg ≠ Ekaterina ∧ Oleg ≠ Zhanna ∧ Ekaterina ≠ Zhanna
axiom husband_older (h w : ℕ) : (h = Roman ∨ h = Oleg) → (w = Zhanna ∨ w = Ekaterina) → h > w
axiom zhanna_older_than_oleg : Zhanna > Oleg

theorem Oleg_older_than_Ekaterina : Oleg > Ekaterina :=
by {
  sorry
}

theorem oldest_is_Roman_and_married_to_Zhanna : ∀ p, (p = Roman ∧ (∀ q, q ≠ Roman → Roman > q)) ∧ (p = Roman → ∃ w, w = Zhanna ∧ husband_older Roman w) :=
by {
  sorry
}

end Oleg_older_than_Ekaterina_oldest_is_Roman_and_married_to_Zhanna_l421_421024


namespace abs_eq_condition_l421_421908

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421908


namespace average_grade_multiple_of_5_l421_421460

theorem average_grade_multiple_of_5 
(grades : List ℕ) 
(h_avg: (grades.sum / grades.length) = 4.6) : 
  ∃ k : ℕ, grades.length = 5 * k := 
sorry

end average_grade_multiple_of_5_l421_421460


namespace shooter_probability_stabilizes_l421_421147

def frequency_of_hits (n m : ℕ) : ℚ := m / n

theorem shooter_probability_stabilizes :
  ∀ (n m : ℕ), (n, m) ∈ [(10, 8), (20, 17), (50, 40), (100, 79), (200, 158), (500, 390), (1000, 780)] →
  frequency_of_hits n m ≈ 0.78 :=
by
  intros n m H
  cases H
  case 1 => sorry
  case 2 => sorry
  case 3 => sorry
  case 4 => sorry
  case 5 => sorry
  case 6 => sorry
  case 7 => sorry

end shooter_probability_stabilizes_l421_421147


namespace sum_first_100_terms_is_5_l421_421327

def seq (n : ℕ) : ℤ :=
if n = 1 then 1 else
if n = 2 then 3 else
seq (n - 1) - seq (n - 2)

def sum_first_100_terms : ℤ :=
(List.range 100).sum (λ n, seq (n + 1))

theorem sum_first_100_terms_is_5 : sum_first_100_terms = 5 := by
  sorry

end sum_first_100_terms_is_5_l421_421327


namespace circle_radius_tangent_to_ellipse_l421_421875

theorem circle_radius_tangent_to_ellipse (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4*y^2 = 8) ↔ r = (Real.sqrt 6) / 2 :=
by
  sorry

end circle_radius_tangent_to_ellipse_l421_421875


namespace area_of_triangle_l421_421399

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l421_421399


namespace calories_left_for_dinner_l421_421556

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l421_421556


namespace Andrena_more_than_Debelyn_l421_421996

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l421_421996


namespace locus_of_centers_of_circles_l421_421243

-- Definitions: Triangle, Point D on BC, E on AC, F on AB

variable {ABC : Type} [triangle ABC]
variable {D E F : Point}
variable {BC AB AC : Line}
variable (lineD_AB : ∀ (D : Point), parallel (line D) AB)
variable (lineD_AC : ∀ (D : Point), parallel (line D) AC)
variable (intersectE_AC : ∀ (D : Point), ∃ E : Point, intersects (line D) AC E)
variable (intersectF_AB : ∀ (D : Point), ∃ F : Point, intersects (line D) AB F)

-- The locus of the centers of circles passing through points D, E, and F is the line MN
theorem locus_of_centers_of_circles {M N : Line} (D E F : Point) :
  locus_of_centers (circle_through D E F) = line MN :=
sorry

end locus_of_centers_of_circles_l421_421243


namespace total_cases_l421_421335

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l421_421335


namespace locus_of_points_l421_421716

variables {P Q R M X Y Z O : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace M] [MetricSpace X]
          [MetricSpace Y] [MetricSpace Z] [MetricSpace O] (L : Set P) (C : Set P) (L' : Set P)

def is_tangent (L : Set P) (C : Set P) : Prop := sorry  -- definition of tangent line
def is_incircle (C : Set P) (P Q R : P) : Prop := sorry  -- definition of incircle
def is_equidistant (M Q R : P) : Prop := sorry  -- definition of equidistant points

-- Given conditions
axiom TangentCondition : is_tangent L C
axiom MOnLine : M ∈ L 
axiom EquidistantPoints : is_equidistant M Q R
axiom IncircleCondition : is_incircle C P Q R

-- Goal: Prove the locus of points
theorem locus_of_points : ∃ (Z : P) (L' : Set P), (locus_of P Q R M X Y Z O L C L' = sorry) ∈ ray Z L' := sorry

end locus_of_points_l421_421716


namespace total_seedlings_transferred_l421_421775

-- Define the number of seedlings planted on the first day
def seedlings_day_1 : ℕ := 200

-- Define the number of seedlings planted on the second day
def seedlings_day_2 : ℕ := 2 * seedlings_day_1

-- Define the total number of seedlings planted on both days
def total_seedlings : ℕ := seedlings_day_1 + seedlings_day_2

-- The theorem statement
theorem total_seedlings_transferred : total_seedlings = 600 := by
  -- The proof goes here
  sorry

end total_seedlings_transferred_l421_421775


namespace balance_increase_second_year_l421_421881

variable (initial_deposit : ℝ) (balance_first_year : ℝ) 
variable (total_percentage_increase : ℝ)

theorem balance_increase_second_year
  (h1 : initial_deposit = 1000)
  (h2 : balance_first_year = 1100)
  (h3 : total_percentage_increase = 0.32) : 
  (balance_first_year + (initial_deposit * total_percentage_increase) - balance_first_year) / balance_first_year * 100 = 20 :=
by
  sorry

end balance_increase_second_year_l421_421881


namespace ellipse_major_axis_length_l421_421973

-- Definition of points representing the foci of the ellipse
def F1 : (ℝ × ℝ) := (9, 20)
def F2 : (ℝ × ℝ) := (49, 55)
def F2' : (ℝ × ℝ) := (49, -55)

-- Distance formula definition
def dist (p1 p2 : (ℝ × ℝ)) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Length of the major axis
def major_axis_length : ℝ := dist F1 F2'

-- Proof statement for Lean
theorem ellipse_major_axis_length :
  major_axis_length = 85 :=
sorry

end ellipse_major_axis_length_l421_421973


namespace longest_side_of_similar_triangle_l421_421522

theorem longest_side_of_similar_triangle :
  ∀ (x : ℝ),
    let a := 8
    let b := 10
    let c := 12
    let s₁ := a * x
    let s₂ := b * x
    let s₃ := c * x
    a + b + c = 30 → 
    30 * x = 150 → 
    s₁ > 30 → 
    max s₁ (max s₂ s₃) = 60 :=
by
  intros x a b c s₁ s₂ s₃ h₁ h₂ h₃
  sorry

end longest_side_of_similar_triangle_l421_421522


namespace orthocenter_of_triangle_l421_421929

variables (A B C X A₁ B₁ C₁ : Type*)

-- Define points and intersections
def point_in_triangle (A B C X : Type*) : Prop :=
  -- X is a point inside triangle ABC
  true -- Since X is a point inside, adding this as a placeholder

def intersect_sides (A B C X A₁ B₁ C₁ : Type*) : Prop := 
  -- AX intersects BC at A₁
  -- BX intersects CA at B₁
  -- CX intersects AB at C₁
  true -- These are lines and intersection points, added placeholder

def circumcircles_intersect (AB₁C₁ A₁BC₁ A₁B₁C : Type*) (X : Type*) : Prop :=
  -- Circumcircles of △AB₁C₁, △A₁BC₁, and △A₁B₁C intersect at point X
  true -- Placeholder for intersection property of circumcircles

-- The main theorem statement
theorem orthocenter_of_triangle 
  (h1 : point_in_triangle A B C X)
  (h2 : intersect_sides A B C X A₁ B₁ C₁)
  (h3 : circumcircles_intersect (AB₁C₁) (A₁BC₁) (A₁B₁C) X) : 
  -- Prove that X is the orthocenter of the triangle ABC
  ord := true :=
begin
  sorry -- Proof placeholder
end

end orthocenter_of_triangle_l421_421929


namespace find_point_P_l421_421248

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 2⟩
def N : Point := ⟨5, -2⟩

def is_on_x_axis (P : Point) : Prop :=
  P.y = 0

def is_right_angle (M N P : Point) : Prop :=
  (M.x - P.x)*(N.x - P.x) + (M.y - P.y)*(N.y - P.y) = 0

noncomputable def P1 : Point := ⟨1, 0⟩
noncomputable def P2 : Point := ⟨6, 0⟩

theorem find_point_P :
  ∃ P : Point, is_on_x_axis P ∧ is_right_angle M N P ∧ (P = P1 ∨ P = P2) :=
by
  sorry

end find_point_P_l421_421248


namespace mod_arith_proof_l421_421392

theorem mod_arith_proof (m : ℕ) (hm1 : 0 ≤ m) (hm2 : m < 50) : 198 * 935 % 50 = 30 := 
by
  sorry

end mod_arith_proof_l421_421392


namespace households_in_city_l421_421706

theorem households_in_city (ha_absorbs : ℕ → ℝ) (ha_forests : ℕ)
    (ac_co2_reduction : ℕ → ℝ) (ac_per_house : ℕ) (co2_red_equiv : ℕ) (expected_households : ℕ) : 
    ha_absorbs 1 = 14 ∧ 
    ac_co2_reduction 1 = 21 ∧ 
    co2_red_equiv = 25000 * ha_absorbs 1 ∧ 
    ac_per_house = 3 ∧ 
    3 * ac_co2_reduction 1 * expected_households = co2_red_equiv * 1000 →
    expected_households = 5555556 := 
by
    intros conditions,
    sorry

end households_in_city_l421_421706


namespace find_number_l421_421297

theorem find_number (x : ℤ) (h : 3 * x - 6 = 2 * x) : x = 6 :=
by
  sorry

end find_number_l421_421297


namespace rose_bushes_l421_421656

theorem rose_bushes (l w spacing perimeter : ℝ) (h1 : l = 24) (h2 : w = 10) (h3 : spacing = 1.2) (h4 : perimeter = 2 * (l + w)) :
  Nat.round (perimeter / spacing) = 57 :=
by
  sorry

end rose_bushes_l421_421656


namespace tom_seashells_found_l421_421779

/-- 
Given:
- sally_seashells = 9 (number of seashells Sally found)
- jessica_seashells = 5 (number of seashells Jessica found)
- total_seashells = 21 (number of seashells found together)

Prove that the number of seashells that Tom found (tom_seashells) is 7.
-/
theorem tom_seashells_found (sally_seashells jessica_seashells total_seashells tom_seashells : ℕ)
  (h₁ : sally_seashells = 9) (h₂ : jessica_seashells = 5) (h₃ : total_seashells = 21) :
  tom_seashells = 7 :=
by
  sorry

end tom_seashells_found_l421_421779


namespace ellipse_eccentricity_l421_421814

theorem ellipse_eccentricity (a b e : ℝ) (h_eq_ellipse : ∀ (x y : ℝ), x + 2 * y = 1 → x^2 / a^2 + y^2 / b^2 = 1 → true) 
(h_slope_product : (2 * b^2 / a^2) * -1 / 2 = -1 / 4) :
  e = sqrt 3 / 2 := sorry

end ellipse_eccentricity_l421_421814


namespace exponent_property_l421_421293

-- Define the conditions as Lean expressions
variables (a b : ℝ)
axiom h1 : 10^a = 3
axiom h2 : 10^b = 5

-- The statement we want to prove
theorem exponent_property : 10^(b - a) = 5 / 3 := 
by sorry

end exponent_property_l421_421293


namespace strokes_over_par_l421_421091

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l421_421091


namespace area_of_region_region_area_eq_sixteen_pi_l421_421548

theorem area_of_region (x y : ℝ) :
  (x - 4) ^ 2 + (y + 5) ^ 2 = 16 ↔
  x^2 + y^2 - 8x + 10y = -25 :=
begin
  sorry
end

theorem region_area_eq_sixteen_pi :
  ∀ (x y : ℝ), (x - 4) ^ 2 + (y + 5) ^ 2 = 16 → 
  ∃ (r : ℝ), r = 4 ∧ real.pi * r^2 = 16 * real.pi :=
begin
  sorry
end

end area_of_region_region_area_eq_sixteen_pi_l421_421548


namespace least_number_with_remainder_l421_421928

theorem least_number_with_remainder (x : ℕ) :
  (x % 6 = 4) ∧ (x % 7 = 4) ∧ (x % 9 = 4) ∧ (x % 18 = 4) ↔ x = 130 :=
by
  sorry

end least_number_with_remainder_l421_421928


namespace required_bandwidth_l421_421933

/-- Given the session duration in minutes, sampling rate in Hz, bit depth in bits,
    and metadata volume in bytes per 5 KB of audio, prove that the required bandwidth
    of this channel in kilobits per second for stereo audio signals is 2.25 Kbit/s.
-/
theorem required_bandwidth (session_duration_min : ℕ) (sampling_rate_hz : ℕ) (bit_depth_bits : ℕ)
(metadata_volume_bytes : ℕ) (stereo_multiplier : ℕ) :
   session_duration_min = 51 →
   sampling_rate_hz = 63 →
   bit_depth_bits = 17 →
   metadata_volume_bytes = 47 →
   stereo_multiplier = 2 →
   let session_duration_sec := session_duration_min * 60 in
   let data_volume_bits := sampling_rate_hz * bit_depth_bits * session_duration_sec in
   let metadata_volume_bits := (metadata_volume_bytes * 8 * data_volume_bits) / (5 * 1024) in
   let total_data_volume_bits := (data_volume_bits + metadata_volume_bits) * stereo_multiplier in
   let throughput_kbps := total_data_volume_bits / (session_duration_sec * 1024) in
   throughput_kbps = 2.25 :=
begin
  intros,
  let session_duration_sec := session_duration_min * 60,
  let data_volume_bits := sampling_rate_hz * bit_depth_bits * session_duration_sec,
  let metadata_volume_bits := (metadata_volume_bytes * 8 * data_volume_bits) / (5 * 1024),
  let total_data_volume_bits := (data_volume_bits + metadata_volume_bits) * stereo_multiplier,
  let throughput_kbps := total_data_volume_bits / (session_duration_sec * 1024),
  sorry
end

end required_bandwidth_l421_421933


namespace x_y_sum_l421_421737

theorem x_y_sum (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end x_y_sum_l421_421737


namespace problem1_problem2_l421_421253

def M (x : ℝ) : Prop := (x + 5) / (x - 8) ≥ 0

def N (x : ℝ) (a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

theorem problem1 : ∀ (x : ℝ), (M x ∨ (N x 9)) ↔ (x ≤ -5 ∨ x ≥ 8) :=
by
  sorry

theorem problem2 : ∀ (a : ℝ), (∀ (x : ℝ), N x a → M x) ↔ (a ≤ -6 ∨ 9 < a) :=
by
  sorry

end problem1_problem2_l421_421253


namespace product_of_second_largest_and_smallest_l421_421870

def numbers : List ℕ := [10, 11, 12, 13, 14]

def secondLargest (l : List ℕ) : ℕ :=
  l.sortedRev.nth 1

def secondSmallest (l : List ℕ) : ℕ :=
  l.sorted.nth 1

theorem product_of_second_largest_and_smallest : 
  secondLargest numbers * secondSmallest numbers = 143 := by
  sorry

end product_of_second_largest_and_smallest_l421_421870


namespace cucumbers_in_salad_l421_421977

theorem cucumbers_in_salad :
  ∃ C T B : ℕ, 
    (T : ℚ) / C = 3 / 2 ∧ 
    B = T + C ∧ 
    T + C + B = 560 ∧ 
    C = 112 :=
by 
  use 112, 168, 280
  split
  exact 168 / 112 = 3 / 2
  split
  exact 280 = 168 + 112
  split
  exact 168 + 112 + 280 = 560
  exact 112


end cucumbers_in_salad_l421_421977


namespace first_player_loses_optimal_strategy_l421_421763

def largest_power_of_2_divisor (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.findGreatest (λ k, 2^k ∣ n) n

theorem first_player_loses_optimal_strategy :
  let n := 100
  let m := 252
  ∀ pick : ℕ → ℕ → Prop, 
    (pick n m → 
    (∃ r ≤ n, r ∣ m ∧ pick (n - r) m) ∨ 
    (∃ r ≤ m, r ∣ n ∧ pick n (m - r))) → 
  (∃ k, largest_power_of_2_divisor n = 2 ∧
            largest_power_of_2_divisor m = 2 ∧ 
            ∀ (n m : ℕ), (largest_power_of_2_divisor n = largest_power_of_2_divisor m →
                                   (pick (n - r) m → ¬ pick n (m - r)) ∧ 
                                   (pick n (m - r) → ¬ pick (n - r) m))) 
  → 
  false := 
by
  intros
  let n := 100
  let m := 252
  use pick
  sorry

end first_player_loses_optimal_strategy_l421_421763


namespace monotonic_intervals_range_of_a_l421_421625

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.log x
noncomputable def g (x a : ℝ) : ℝ := f x a / x

-- Prove the intervals where g(x) is monotonic given a = 3 * sqrt(e)
theorem monotonic_intervals (a : ℝ) (h_a : a = 3 * Real.sqrt Real.exp 1) :
  (∀ x, x ∈ Set.Ioo 0 (Real.sqrt (Real.exp 1)) → g x a > 0) ∧
  (∀ x, x ∈ Set.Ioo (Real.sqrt (Real.exp 1)) (3 * Real.sqrt (Real.exp 1)) → g x a < 0) ∧
  (∀ x, x ∈ Set.Ioo (3 * Real.sqrt (Real.exp 1)) (Real.exp 1) → g x a > 0) := sorry

-- Prove the range of values for a such that f(x) has both a maximum and minimum value
theorem range_of_a (a : ℝ) :
  (∃ x, f x a = Real.sup (Set.range (λ x => f x a))) ∧
  (∃ x, f x a = Real.inf (Set.range (λ x => f x a))) ↔
  a > -2 * Real.exp (-3 / 2) ∧ a ≠ 0 ∧ a ≠ 1 := sorry

end monotonic_intervals_range_of_a_l421_421625


namespace AE_six_l421_421540

namespace MathProof

-- Definitions of the given conditions
variables {A B C D E : Type}
variables (AB CD AC AE : ℝ)
variables (triangleAED_area triangleBEC_area : ℝ)

-- Given conditions
def conditions : Prop := 
  convex_quadrilateral A B C D ∧
  AB = 9 ∧
  CD = 12 ∧
  AC = 14 ∧
  intersect_at E AC BD ∧
  areas_equal triangleAED_area triangleBEC_area

-- Theorem to prove AE = 6
theorem AE_six (h : conditions AB CD AC AE triangleAED_area triangleBEC_area) : 
  AE = 6 :=
by sorry  -- proof omitted

end MathProof

end AE_six_l421_421540


namespace find_scalars_l421_421620

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, -3, 1)
noncomputable def vector_c : ℝ × ℝ × ℝ := (4, 1, -5)
noncomputable def target_vector : ℝ × ℝ × ℝ := (-3, 6, 2)

theorem find_scalars (p q r : ℝ) 
  (h1 : p = 5 / 3) 
  (h2 : q = 3 / 14) 
  (h3 : r = -1 / 21) : 
  target_vector = 
  (p * vector_a.1 + q * vector_b.1 + r * vector_c.1,
   p * vector_a.2 + q * vector_b.2 + r * vector_c.2,
   p * vector_a.3 + q * vector_b.3 + r * vector_c.3) :=
sorry

end find_scalars_l421_421620


namespace neighbor_behind_ratio_l421_421534

theorem neighbor_behind_ratio
  (backyard_side : ℕ) (backyard_back : ℕ)
  (fencing_cost_per_foot : ℕ) (cole_payment : ℕ)
  (left_neighbor_fraction : ℚ)
  (left_neighbor_length : ℕ) :
  backyard_side = 9 ∧ backyard_back = 18 ∧
  fencing_cost_per_foot = 3 ∧ cole_payment = 72 ∧
  left_neighbor_fraction = 1/3 ∧ left_neighbor_length = 9 →
  let total_length := 2 * backyard_side + backyard_back in
  let total_cost := total_length * fencing_cost_per_foot in
  let neighbors_contribution := total_cost - cole_payment in
  let left_neighbor_contribution := left_neighbor_fraction * left_neighbor_length * fencing_cost_per_foot in
  let behind_neighbor_contribution := neighbors_contribution - left_neighbor_contribution in
  let back_side_cost := backyard_back * fencing_cost_per_foot in
  (behind_neighbor_contribution / back_side_cost : ℚ) = 1/2 :=
begin
  sorry
end

end neighbor_behind_ratio_l421_421534


namespace find_m_l421_421266

-- Definitions for vectors and dot products
structure Vector :=
  (i : ℝ)
  (j : ℝ)

def dot_product (a b : Vector) : ℝ :=
  a.i * b.i + a.j * b.j

-- Given conditions
def i : Vector := ⟨1, 0⟩
def j : Vector := ⟨0, 1⟩

def a : Vector := ⟨2, 3⟩
def b (m : ℝ) : Vector := ⟨1, -m⟩

-- The main goal
theorem find_m (m : ℝ) (h: dot_product a (b m) = 1) : m = 1 / 3 :=
by {
  -- Calculation reaches the same \(m = 1/3\)
  sorry
}

end find_m_l421_421266


namespace folded_segment_square_length_eq_225_div_4_l421_421506

noncomputable def square_of_fold_length : ℝ :=
  let side_length := 15
  let distance_from_B := 5
  (side_length ^ 2 - distance_from_B * (2 * side_length - distance_from_B)) / 4

theorem folded_segment_square_length_eq_225_div_4 :
  square_of_fold_length = 225 / 4 :=
by
  sorry

end folded_segment_square_length_eq_225_div_4_l421_421506


namespace rationalize_denominator_l421_421385

noncomputable def simplify (x : ℝ) : ℝ :=
  1 / (2 + 1 / (sqrt 5 + 2))

theorem rationalize_denominator :
  simplify (1 / (2 + 1 / (sqrt 5 + 2))) = sqrt 5 / 5 :=
by
  sorry

end rationalize_denominator_l421_421385


namespace calc_expression_l421_421175

theorem calc_expression :
  2 * Real.sin (Real.pi / 3) + Real.abs (Real.sqrt 3 - 2) + (-1)^(-1 : ℤ) - Real.cbrt (-8) = 3 :=
by
  sorry

end calc_expression_l421_421175


namespace exists_balanced_num1_exists_balanced_num2_exists_balanced_num2_alternatives_l421_421459

-- Define what it means for a number (represented as a structure) to be balanced
structure BalancedNum where
  before_decimal : List ℕ
  after_decimal : List ℕ
  before_sum_eq_after_sum : (before_decimal.sum) = (after_decimal.sum)

-- Define specific instances for the problem
-- instance for 497365.198043 transformed into 47365.198043
def num1 := BalancedNum.mk [4, 7, 3, 6, 5] [1, 9, 8, 0, 4, 3] (by simp)
-- instance for 197352.598062 transformed into multiple possibilities
def num2_a := BalancedNum.mk [1, 9, 7, 3, 5, 2] [5, 9, 8, 2] (by simp)
def num2_b := BalancedNum.mk [1, 9, 7, 3, 2] [5, 9, 0, 6, 2] (by simp)
def num2_c := BalancedNum.mk [1, 9, 7, 5, 2] [5, 9, 8, 0, 2] (by simp)
def num2_d := BalancedNum.mk [1, 9, 7, 2] [5, 9, 8, 0, 6, 2] (by simp)

-- Theorem stating that there exists a balanced number for given conditions
theorem exists_balanced_num1 : ∃ (n : BalancedNum), n = num1 := by
  use num1
  exact rfl

theorem exists_balanced_num2 : ∃ (n : BalancedNum), n = num2_a ∨ n = num2_b ∨ n = num2_c ∨ n = num2_d := by
  use num2_a
  left; exact rfl
-- Additional balance possibilities
theorem exists_balanced_num2_alternatives : ∃ (n : BalancedNum), n = num2_b ∨ n = num2_c ∨ n = num2_d := by
  right; left; exact rfl

end exists_balanced_num1_exists_balanced_num2_exists_balanced_num2_alternatives_l421_421459


namespace problem1_problem2_l421_421984

theorem problem1 : (sqrt 24 - sqrt 2) - (sqrt 8 + sqrt 6) = sqrt 6 - 3 * sqrt 2 :=
by 
  sorry

theorem problem2 : (4 * sqrt 2 - 8 * sqrt 6) / (2 * sqrt 2) = 2 - 4 * sqrt 3 :=
by 
  sorry

end problem1_problem2_l421_421984


namespace length_AE_l421_421598

-- The given conditions:
def isosceles_triangle (A B C : Type*) (AB BC : ℝ) (h : AB = BC) : Prop := true

def angles_and_lengths (A D C E : Type*) (angle_ADC angle_AEC AD CE DC : ℝ) 
  (h_angles : angle_ADC = 60 ∧ angle_AEC = 60)
  (h_lengths : AD = 13 ∧ CE = 13 ∧ DC = 9) : Prop := true

variables {A B C D E : Type*} (AB BC AD CE DC : ℝ)
  (h_isosceles_triangle : isosceles_triangle A B C AB BC (by sorry))
  (h_angles_and_lengths : angles_and_lengths A D C E 60 60 AD CE DC 
    (by split; norm_num) (by repeat {split}; norm_num))

-- The proof problem:
theorem length_AE : ∃ AE : ℝ, AE = 4 :=
  by sorry

end length_AE_l421_421598


namespace num_real_a_satisfy_union_l421_421251

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_l421_421251


namespace binom_15_4_l421_421989

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l421_421989


namespace solve_absolute_value_eq_l421_421910

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421910


namespace parallelogram_BPLQ_l421_421874

/-- Given two circles internally tangent at a point N, and chords BA and BC of the larger circle 
tangent to the smaller circle at points K and M respectively, and points Q and P as midpoints of arcs AB and BC, 
with the circumcircles of triangles BQK and BPM intersecting at point L, 
we are to prove that the quadrilateral BPLQ is a parallelogram. --/
theorem parallelogram_BPLQ 
    (circle_large : Circle)
    (circle_small : Circle)
    (N : Point)
    (int_tangent : internally_tangent circle_large circle_small N)
    (A B C K M Q P L : Point)
    (BA_tangent_small : tangent circle_large BA circle_small K)
    (BC_tangent_small : tangent circle_large BC circle_small M)
    (mid_AB : midpoint_of_arc circle_large A B Q)
    (mid_BC : midpoint_of_arc circle_large B C P)
    (circumcircle_BQK : circle)
    (circumcircle_BPM : circle)
    (L_on_BQK : on_circle circumcircle_BQK L)
    (L_on_BPM : on_circle circumcircle_BPM L) :
    parallelogram B P L Q := sorry

end parallelogram_BPLQ_l421_421874


namespace area_of_triangle_formed_by_tangent_l421_421198

open Real

/-- The curve is defined as y = (1/3)x³ + x. -/
def curve (x : ℝ) : ℝ := (1 / 3) * x ^ 3 + x

/-- The point of tangency is (1, 4/3). -/
def point_of_tangency : ℝ × ℝ := (1, 4 / 3)

/-- The solution to the problem is to prove that the area of the triangle formed by the tangent line to the curve at the point (1, 4/3) and the coordinate axes is 1/9. -/
theorem area_of_triangle_formed_by_tangent :
  let slope := (fun x => x^2 + 1) 1,
      tangent_line := (fun x => slope * (x - 1) + 4 / 3),
      x_intercept := 1 / 3,
      y_intercept := -2 / 3,
      triangle_area := 1 / 2 * x_intercept * (-y_intercept)
  in triangle_area = 1 / 9 :=
by
  sorry

end area_of_triangle_formed_by_tangent_l421_421198


namespace possible_values_of_AD_l421_421365

-- Define the conditions as variables
variables {A B C D : ℝ}
variables {AB BC CD : ℝ}

-- Assume the given conditions
def conditions (A B C D : ℝ) (AB BC CD : ℝ) : Prop :=
  AB = 1 ∧ BC = 2 ∧ CD = 4

-- Define the proof goal: proving the possible values of AD
theorem possible_values_of_AD (h : conditions A B C D AB BC CD) :
  ∃ AD, AD = 1 ∨ AD = 3 ∨ AD = 5 ∨ AD = 7 :=
sorry

end possible_values_of_AD_l421_421365


namespace fg_of_2_l421_421648

def f (x : ℤ) : ℤ := 4 * x + 3
def g (x : ℤ) : ℤ := x ^ 3 + 1

theorem fg_of_2 : f (g 2) = 39 := by
  sorry

end fg_of_2_l421_421648


namespace maximum_value_of_y_over_x_l421_421267

noncomputable def maxSlope := (λ x y : ℝ , y / x)

theorem maximum_value_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : maxSlope x y ≤ real.sqrt 3 := sorry

end maximum_value_of_y_over_x_l421_421267


namespace sum_binom_specific_final_answer_l421_421603

def sum_binom : ℚ :=
  ∑ n in finset.range  8 \ finset.range 3, 
  (Nat.choose n 2 : ℚ) / 
  ((Nat.choose n 3 : ℚ) * (Nat.choose (n + 1) 3 : ℚ))

theorem sum_binom_specific :
  sum_binom = 164 / 165 :=
by
  sorry

theorem final_answer :
  164 + 165 = 329 :=
by
  exact rfl

end sum_binom_specific_final_answer_l421_421603


namespace adam_apples_count_l421_421966

variable (Jackie_apples : ℕ)
variable (extra_apples : ℕ)
variable (Adam_apples : ℕ)

theorem adam_apples_count (h1 : Jackie_apples = 9) (h2 : extra_apples = 5) (h3 : Adam_apples = Jackie_apples + extra_apples) :
  Adam_apples = 14 := 
by 
  sorry

end adam_apples_count_l421_421966


namespace binom_15_4_l421_421988

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l421_421988


namespace find_kth_term_l421_421073

-- Definitions based on conditions
def partial_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum a

def sequence : ℕ → ℚ
| n := let m := nat.find_greatest (λ k, k * (k + 1) / 2 < n) (nat.sub (nat.succ n) 1) in
       let k := n - (m * (m + 1) / 2 + 1) in
       (↑(nat.succ k) / ↑(nat.succ m))

-- Main statement, with conditions
theorem find_kth_term (S : ℕ → ℚ) (a : ℕ → ℚ)
  (hS : ∀ n, S n = partial_sum a n)
  (ha : ∀ n, a n = sequence n)
  (h1 : ∃ k : ℕ, S (k - 1) < 10 ∧ S k > 10) :
  ∃ k : ℕ, a k = 6 / 7 :=
sorry

end find_kth_term_l421_421073


namespace profit_function_expression_profit_range_max_profit_l421_421500

noncomputable def G (x : ℝ) : ℝ := 2.8 + x

noncomputable def R (x : ℝ) : ℝ :=
  if x ≤ 5 then -0.4 * x^2 + 3.4 * x + 0.8 else 9

noncomputable def f (x : ℝ) : ℝ :=
  R(x) - G(x)

theorem profit_function_expression : 
  ∀ x : ℝ, f(x) =
  if x ≤ 5 then -0.4 * x^2 + 2.4 * x - 2 else 6.2 - x :=
by
  sorry

theorem profit_range (x : ℝ) : 
  f(x) > 0 ↔ (1 < x ∧ x < 5) ∨ (5 < x ∧ x < 6.2) :=
by
  sorry

theorem max_profit : 
  ∃ x : ℝ, f(x) = 1.6 ∧ x = 3 :=
by
  sorry

end profit_function_expression_profit_range_max_profit_l421_421500


namespace find_b_given_tangent_circle_l421_421684

noncomputable def monge_circle_radius (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

def centers_distance (cx1 cy1 cx2 cy2 : ℝ) : ℝ := real.sqrt ((cx2 - cx1)^2 + (cy2 - cy1)^2)

theorem find_b_given_tangent_circle :
  let a := real.sqrt 3,
      b := real.sqrt 1,
      r1 := monge_circle_radius a b,
      cx1 := 0,
      cy1 := 0,
      r2 := 3,
      cx2 := 3,
      b_candidate := find_b_in_circle
  in 
  centers_distance cx1 cy1 cx2 b_candidate = r1 + r2 -> 
  b_candidate ^ 2 = 16 :=
sorry

end find_b_given_tangent_circle_l421_421684


namespace ellen_dinner_calories_proof_l421_421552

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l421_421552


namespace maximum_area_of_triangle_AOB_l421_421280

-- Definitions
def hyperbola (x y : ℝ) := x^2 / 3 - y^2 = 1
def asymptote1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
def asymptote2 (x y : ℝ) := y = -(Real.sqrt 3 / 3) * x
def A_in_first_quadrant (A : ℝ × ℝ) := A.1 > 0 ∧ A.2 = (Real.sqrt 3 / 3) * A.1
def B_in_fourth_quadrant (B : ℝ × ℝ) := B.1 > 0 ∧ B.2 = -(Real.sqrt 3 / 3) * B.1
def vector_relation (A P B : ℝ × ℝ) (λ : ℝ) := P.1 - A.1 = λ * (B.1 - P.1) ∧ P.2 - A.2 = λ * (B.2 - P.2)

-- Problem statement
theorem maximum_area_of_triangle_AOB (A B P : ℝ × ℝ) (λ : ℝ) 
  (hA : A_in_first_quadrant A) (hB : B_in_fourth_quadrant B) 
  (hP : hyperbola P.1 P.2) (hλ : λ ∈ Set.Icc (1/3) 2) 
  (hvec : vector_relation A P B λ) :
  ∃ max_area : ℝ, max_area = (4 * Real.sqrt 3) / 3 :=
sorry

end maximum_area_of_triangle_AOB_l421_421280


namespace Andrena_more_than_Debelyn_l421_421997

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l421_421997


namespace distance_A_B_l421_421584

-- Define the initial and final positions.
def A : ℝ × ℝ := (0, 0)
def south : ℝ × ℝ := (0, -50)
def west : ℝ × ℝ := (-80, 0)
def north : ℝ × ℝ := (0, 20)
def east : ℝ × ℝ := (40, 0)

-- Compute the final position B after all movements.
def B : ℝ × ℝ := A + south + west + north + east

-- State the theorem to prove the distance between A and B.
theorem distance_A_B : 
  let d := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) in
  d = 50 := by
  sorry

end distance_A_B_l421_421584


namespace sum_of_squares_l421_421773

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 40) (h₂ : x * y = 120) : x^2 + y^2 = 1360 :=
by
  sorry

end sum_of_squares_l421_421773


namespace total_soldiers_correct_l421_421320

-- Definitions based on conditions
def num_generals := 8
def num_vanguards := 8^2
def num_flags := 8^3
def num_team_leaders := 8^4
def num_armored_soldiers := 8^5
def num_soldiers := 8 + 8^2 + 8^3 + 8^4 + 8^5 + 8^6

-- Prove total number of soldiers
theorem total_soldiers_correct : num_soldiers = (1 / 7 : ℝ) * (8^7 - 8) := by
  sorry

end total_soldiers_correct_l421_421320


namespace line_passes_through_vertex_of_parabola_l421_421217

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l421_421217


namespace volume_of_mixture_l421_421444

/-
Conditions:
1. The weight of one liter vegetable ghee packet of brand 'a' is 900 gm.
2. The weight of one liter vegetable ghee packet of brand 'b' is 850 gm.
3. They are mixed in the ratio of 3:2 by volumes.
4. The weight of the mixture is 3520 gm.
-/
variables (weight_a weight_b weight_mixture : ℕ)
variables (ratio_a ratio_b : ℕ)

def mixed_weight (V_a V_b : ℕ) : ℕ :=
  900 * V_a + 850 * V_b

theorem volume_of_mixture :
  let V_a := 24 / 10 in
  let V_b := 16 / 10 in
  let total_volume := V_a + V_b in
  total_volume = 4 :=
by
  have Va := 24/10
  have Vb := 16/10
  have total_volume := Va + Vb
  sorry

end volume_of_mixture_l421_421444


namespace tangent_normal_lines_l421_421571

theorem tangent_normal_lines (x0 : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x, 2 * x ^ 3 - x - 4)
  (h2 : x0 = 1) :
  (∃ m b, ∀ x : ℝ, m = 5 ∧ b = -8 ∧ (∀ y, y = m * x + b → y = 5 * x - 8)) ∧
  (∃ m b, ∀ x : ℝ, m = -1 / 5 ∧ b = -14 / 5 ∧ (∀ y, y = m * x + b → y = -0.2 * x - 2.8)) :=
sorry

end tangent_normal_lines_l421_421571


namespace broccoli_pounds_l421_421708

theorem broccoli_pounds:
  ∀ (broccoli_cost : ℝ) (oranges_cost : ℝ) (cabbage_cost : ℝ) (bacon_cost : ℝ) (chicken_cost_per_pound : ℝ) (chicken_pounds : ℝ) (meat_percentage : ℝ)
  (total_groceries_cost : ℝ) (other_items_cost : ℝ) (total_cost : ℝ) (money_spent_broccoli : ℝ),

  broccoli_cost = 4 ∧
  oranges_cost = 3 * 0.75 ∧
  cabbage_cost = 3.75 ∧
  bacon_cost = 3 ∧
  chicken_cost_per_pound = 3 ∧
  chicken_pounds = 2 ∧
  meat_percentage = 0.33 ∧
  total_groceries_cost = bacon_cost + chicken_cost_per_pound * chicken_pounds ∧
  total_cost = oranges_cost + cabbage_cost + bacon_cost + chicken_cost_per_pound * chicken_pounds ∧
  money_spent_broccoli = total_groceries_cost / meat_percentage - total_cost

  → nat.floor (money_spent_broccoli / broccoli_cost) = 3 :=
by
  intros broccoli_cost oranges_cost cabbage_cost bacon_cost chicken_cost_per_pound chicken_pounds meat_percentage total_groceries_cost other_items_cost total_cost money_spent_broccoli
  assume h_broccoli_cost h_oranges_cost h_cabbage_cost h_bacon_cost h_chicken_cost_per_pound h_chicken_pounds h_meat_percentage h_total_groceries_cost h_total_cost h_money_spent_broccoli
  rw [h_broccoli_cost, h_oranges_cost, h_cabbage_cost, h_bacon_cost, h_chicken_cost_per_pound, h_chicken_pounds, h_meat_percentage, h_total_groceries_cost, h_total_cost, h_money_spent_broccoli]
  sorry

end broccoli_pounds_l421_421708


namespace miles_to_mall_l421_421755

noncomputable def miles_to_grocery_store : ℕ := 10
noncomputable def miles_to_pet_store : ℕ := 5
noncomputable def miles_back_home : ℕ := 9
noncomputable def miles_per_gallon : ℕ := 15
noncomputable def cost_per_gallon : ℝ := 3.50
noncomputable def total_cost_of_gas : ℝ := 7.00
noncomputable def total_miles_driven := 2 * miles_per_gallon

theorem miles_to_mall : total_miles_driven -
  (miles_to_grocery_store + miles_to_pet_store + miles_back_home) = 6 :=
by
  -- proof omitted 
  sorry

end miles_to_mall_l421_421755


namespace value_of_c_over_ab_l421_421231

theorem value_of_c_over_ab
  (a b c : ℚ)
  (h1 : ab / (a + b) = 3)
  (h2 : bc / (b + c) = 6)
  (h3 : ac / (a + c) = 9)
  : c / (ab) = -35/36 := 
by
  sorry

end value_of_c_over_ab_l421_421231


namespace triangle_perimeter_inequality_l421_421058

theorem triangle_perimeter_inequality (x : ℕ) (h₁ : 15 + 24 > x) (h₂ : 15 + x > 24) (h₃ : 24 + x > 15) 
    (h₄ : ∃ x : ℕ, x > 9 ∧ x < 39) : 15 + 24 + x = 49 :=
by { sorry }

end triangle_perimeter_inequality_l421_421058


namespace evaluate_polynomial_l421_421562

theorem evaluate_polynomial (x : ℝ) (h : x = 2) : x^3 + x^2 + x + real.exp x = 14 + real.exp 2 :=
by
  rw [h]
  sorry

end evaluate_polynomial_l421_421562


namespace false_prop_range_of_a_l421_421835

theorem false_prop_range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (a < -2 * Real.sqrt 2 ∨ a > 2 * Real.sqrt 2) :=
by
  sorry

end false_prop_range_of_a_l421_421835


namespace maximum_obtuse_vectors_l421_421888

-- Definition: A vector in 3D space
structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition: Dot product of two vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition: Two vectors form an obtuse angle if their dot product is negative
def obtuse_angle (v1 v2 : Vector3D) : Prop :=
  dot_product v1 v2 < 0

-- Main statement incorporating the conditions and the conclusion
theorem maximum_obtuse_vectors :
  ∀ (v1 v2 v3 v4 : Vector3D),
  (obtuse_angle v1 v2) →
  (obtuse_angle v1 v3) →
  (obtuse_angle v1 v4) →
  (obtuse_angle v2 v3) →
  (obtuse_angle v2 v4) →
  (obtuse_angle v3 v4) →
  -- Conclusion: At most 4 vectors can be pairwise obtuse
  ∃ (v5 : Vector3D),
  ¬ (obtuse_angle v1 v5 ∧ obtuse_angle v2 v5 ∧ obtuse_angle v3 v5 ∧ obtuse_angle v4 v5) :=
sorry

end maximum_obtuse_vectors_l421_421888


namespace decimal_150th_place_of_5_over_11_l421_421115

theorem decimal_150th_place_of_5_over_11 :
  let r := "45" in  -- The repeating decimal part
  let n := 150 in   -- The 150th place to find
  let repeat_len := 2 in -- Length of the repeating cycle
  cycle_digit r (n % repeat_len) = '5' := 
by
  sorry

/-- Helper function to get the nth digit of a repeating decimal cycle -/
def cycle_digit (cycle: String) (n: Nat) : Char := 
  cycle.get (n % cycle.length)

end decimal_150th_place_of_5_over_11_l421_421115


namespace num_real_solutions_l421_421186

noncomputable def f (x : ℝ) : ℝ := 
  (Finset.range 100).sum (λ n, (2 * (n + 1)) / (x - (2 * (n + 1))))

theorem num_real_solutions : 
  (∃ n : ℕ, n = 101 ∧ (∀ x : ℝ, f x = 2 * x → True)) :=
sorry

end num_real_solutions_l421_421186


namespace a_2023_eq_3_S_2024_eq_neg_2530_l421_421302

noncomputable def a : ℕ → ℤ := sorry
noncomputable def b (n : ℕ) : ℤ := (-1) ^ n * a n
def S (n : ℕ) : ℤ := (List.range n).sum (λ i, b i)

axiom h1 : ∀ {p q : ℕ}, a p = a q → a (p + 1) = a (q + 1)
axiom h2 : a 1 = 1
axiom h3 : a 3 = 3
axiom h4 : a 5 = 1
axiom h5 : a 4 + a 7 + a 10 = 2

theorem a_2023_eq_3 : a 2023 = 3 := sorry
theorem S_2024_eq_neg_2530 : S 2024 = -2530 := sorry

end a_2023_eq_3_S_2024_eq_neg_2530_l421_421302


namespace distribute_cakes_count_l421_421367

def numWaysToDistributeCakes (n : ℕ) (d : ℕ → ℕ) : ℕ :=
  n + d n

theorem distribute_cakes_count (n : ℕ) (d : ℕ → ℕ)
  (h_divisors : ∀ k, k ∣ n ↔ k ≤ n + d n) :
  ∑ k in finset.range (n + 1), 1 + d n =
  numWaysToDistributeCakes n d :=
by
  sorry

end distribute_cakes_count_l421_421367


namespace b_21_equals_861_l421_421607

-- Definitions based on given conditions
def a (n : ℕ) := n * (n + 1) / 2

def is_divisible_by_2 (n : ℕ) := ∃ k : ℕ, n = 2 * k

def b : ℕ → ℕ
| 0     := 0
| (n+1) := let m := (n+1) * (n+2) / 2 in if is_divisible_by_2 m then b n else m

-- Problem statement
theorem b_21_equals_861 : b 21 = 861 := 
sorry

end b_21_equals_861_l421_421607


namespace singleCirclePercentage_is_48_percent_l421_421132

noncomputable def doubleFactorial (n : ℕ) : ℕ :=
  if n = 0 ∨ n = 1 then 1 else n * doubleFactorial (n - 2)

def numSingleCircleConfigs (n : ℕ) : ℕ :=
  doubleFactorial (2 * n - 2)

def numAllConfigs (n : ℕ) : ℕ :=
  let singleCircle := numSingleCircleConfigs n
  -- A placeholder function to calculate all configurations
  387099936  -- Precomputed total number of configurations for n = 10

def singleCirclePercentage (n : ℕ) : ℚ :=
  numSingleCircleConfigs n / numAllConfigs n

theorem singleCirclePercentage_is_48_percent : 
  singleCirclePercentage 10 = 48 / 100 := 
by
  -- Based on the problem, precompute as per solution
  sorry

end singleCirclePercentage_is_48_percent_l421_421132


namespace line_parallel_plane_implies_perpendicular_l421_421587

variable {α : Type*}
variables (a b : set α) -- Typically would denote lines with a specific type, e.g., affine subspaces, but we simplify as sets here
variables (α_plane : set α) -- Similarly, a plane is simplified as a set

-- Given
axiom plane (hα : is_plane α_plane)
axiom lines_are_different (ha : a ≠ b)
axiom line_perpendicular_to_plane (ha_perp : is_perpendicular a α_plane)
axiom lines_are_parallel (ha_parallel_b : is_parallel a b)

-- Prove
theorem line_parallel_plane_implies_perpendicular (hb_perp : is_perpendicular b α_plane) : 
  is_perpendicular b α_plane :=
sorry

end line_parallel_plane_implies_perpendicular_l421_421587


namespace find_x_l421_421895

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421895


namespace magic_square_sum_l421_421326

theorem magic_square_sum (a b c d e f S : ℕ) 
  (h1 : 30 + b + 22 = S) 
  (h2 : 19 + c + d = S) 
  (h3 : a + 28 + f = S)
  (h4 : 30 + 19 + a = S)
  (h5 : b + c + 28 = S)
  (h6 : 22 + d + f = S)
  (h7 : 30 + c + f = S)
  (h8 : 22 + c + a = S)
  (h9 : e = b) :
  d + e = 54 := 
by 
  sorry

end magic_square_sum_l421_421326


namespace problem_concentric_circles_chord_probability_l421_421455

open ProbabilityTheory

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h : r1 < r2) : ℝ :=
1/6

theorem problem_concentric_circles_chord_probability :
  probability_chord_intersects_inner_circle 1.5 3 
  (by norm_num) = 1/6 :=
sorry

end problem_concentric_circles_chord_probability_l421_421455


namespace solve_abs_eq_l421_421898

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421898


namespace necessary_and_sufficient_l421_421490

def point_on_curve (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : Prop :=
  f P = 0

theorem necessary_and_sufficient (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) :
  (point_on_curve P f ↔ f P = 0) :=
by
  sorry

end necessary_and_sufficient_l421_421490


namespace fraction_went_on_field_trip_l421_421926

-- Definitions based on conditions
variables {x : ℕ}
def fraction_left_on_trip := (4 : ℚ) / 5
def fraction_stayed_behind := 1 - fraction_left_on_trip
def fraction_did_not_want_to_go := (1 : ℚ) / 3
def fraction_did_want_to_go := fraction_stayed_behind * (1 - fraction_did_not_want_to_go)
def fraction_additional_students := fraction_did_want_to_go / 2

-- Main statement to prove
theorem fraction_went_on_field_trip
  (hx : x > 0) :
  let initial_trip = fraction_left_on_trip * x,
      additional_trip = fraction_additional_students * x
  in (initial_trip + additional_trip) / x = (13 : ℚ) / 15 :=
by
  sorry

end fraction_went_on_field_trip_l421_421926


namespace infinitely_many_MTRP_numbers_l421_421355

def sum_of_digits (n : ℕ) : ℕ := 
n.digits 10 |>.sum

def is_MTRP_number (m n : ℕ) : Prop :=
  n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_MTRP_numbers (m : ℕ) : 
  ∀ N : ℕ, ∃ n > N, is_MTRP_number m n :=
by sorry

end infinitely_many_MTRP_numbers_l421_421355


namespace abs_eq_condition_l421_421907

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421907


namespace find_xyz_l421_421751

variable {x y z : ℝ}

axiom h_x_y_rational : ∀ x y : ℝ, x.is_rational ∧ y.is_rational
axiom h_z_irrational : z.is_irrational
axiom h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x
axiom h_correct_eq : x^2 + y^2 + z^2 = (73 + 6*z - 6*(x + y))^2

theorem find_xyz (x y z : ℝ) [hx : h_x_y_rational x y] [hz : h_z_irrational] [hd : h_distinct] : 
  x^2 + y^2 + z^2 = (73 + 6*z - 6*(x + y))^2 :=
by sorry 

end find_xyz_l421_421751


namespace equilibrium_wage_increase_equilibrium_price_decrease_l421_421130

section TeacherLaborMarket

-- We define the main entities and assumptions based on the conditions.
variables (supply demand : ℕ → ℕ)
variables (locality : Type) (teachers universityGraduates : locality → ℕ)
variables (governmentPolicy : Prop)

-- Government policy condition
def gov_policy (universityGraduates mandatoryServiceYears : locality → ℕ) : Prop :=
  ∀ l, universityGraduates l > mandatoryServiceYears l

-- Theorems based on given conditions and correct answers
theorem equilibrium_wage_increase
  (hpolicy : gov_policy universityGraduates (λ l, 0))
  (h1 : ∀ l, teachers l < universityGraduates l) :
  ∃ wage : ℕ, ∀ l, wage > teachers l := sorry

theorem equilibrium_price_decrease
  (hpolicy : gov_policy universityGraduates (λ l, 0))
  (h1 : ∀ l, teachers l > universityGraduates l) :
  ∃ price : ℕ, ∀ l, price < teachers l := sorry

end TeacherLaborMarket

end equilibrium_wage_increase_equilibrium_price_decrease_l421_421130


namespace decreasing_interval_l421_421831

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem decreasing_interval : ∀ x : ℝ, f'(x) < 0 ↔ -1 < x ∧ x < 11 := by
  sorry

end decreasing_interval_l421_421831


namespace sum_of_integers_l421_421435

theorem sum_of_integers (x y : ℕ) (h1 : x * y + x + y = 187) 
  (h2 : Nat.coprime x y) (h3 : x < 30) (h4 : y < 30) : x + y = 49 := 
sorry

end sum_of_integers_l421_421435


namespace savings_when_purchased_together_l421_421518

def window_price : ℕ := 100

def free_window_for_each_three_purchased := 4

def cost_with_discount (windows_needed : ℕ) : ℕ :=
  let groups_of_four := windows_needed / free_window_for_each_three_purchased
  let extra_windows := windows_needed % free_window_for_each_three_purchased
  (groups_of_four * 3 + min extra_windows 3) * window_price

def dave_windows : ℕ := 9
def doug_windows : ℕ := 10

def separate_cost : ℕ := cost_with_discount dave_windows + cost_with_discount doug_windows

def joint_cost : ℕ := cost_with_discount (dave_windows + doug_windows)

theorem savings_when_purchased_together : separate_cost - joint_cost = 0 :=
by
  have s_cost := separate_cost
  have j_cost := joint_cost
  calc
  s_cost - j_cost = 0 : by sorry

end savings_when_purchased_together_l421_421518


namespace problem1_problem2_l421_421278

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 9 * x

theorem problem1 :
  ∀ (x : ℝ), (f x 3)'
  → (1, +∞)

theorem problem2 :
  ∀ a,
  ∀ x ∈ [1/2, 2],
  g x a ≤ 0
  → a ∈ [6, +∞]

end problem1_problem2_l421_421278


namespace area_relation_l421_421974

open Real

noncomputable def S_OMN (a b c d θ : ℝ) : ℝ := 1 / 2 * abs (b * c - a * d) * sin θ
noncomputable def S_ABCD (a b c d θ : ℝ) : ℝ := 2 * abs (b * c - a * d) * sin θ

theorem area_relation (a b c d θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
    4 * (S_OMN a b c d θ) = S_ABCD a b c d θ :=
by
  sorry

end area_relation_l421_421974


namespace tan_four_positive_l421_421121

theorem tan_four_positive :
  tan 4 > 0 :=
by
  sorry

end tan_four_positive_l421_421121


namespace first_train_length_l421_421098

noncomputable def length_of_first_train
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_seconds : ℕ)
  (relative_speed_kmph : ℕ) : ℕ :=
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  in relative_speed_mps * time_seconds

theorem first_train_length:
  ∀ (speed_first_train_kmph speed_second_train_kmph time_seconds : ℕ),
  speed_first_train_kmph = 60 →
  speed_second_train_kmph = 80 →
  time_seconds = 6 →
  length_of_first_train speed_first_train_kmph speed_second_train_kmph time_seconds (speed_first_train_kmph + speed_second_train_kmph) = 233 :=
by
  intros speed_first_train_kmph speed_second_train_kmph time_seconds h1 h2 h3
  unfold length_of_first_train
  rw [h1, h2, h3]
  norm_num
  sorry

end first_train_length_l421_421098


namespace donna_paid_correct_amount_l421_421952

-- Define the original price, discount rate, and sales tax rate
def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.10

-- Define the total amount Donna paid
def total_amount_donna_paid : ℝ := 165

-- Define a theorem to express the proof problem
theorem donna_paid_correct_amount :
  let discount_amount := original_price * discount_rate in
  let sale_price := original_price - discount_amount in
  let sales_tax_amount := sale_price * sales_tax_rate in
  let total_amount := sale_price + sales_tax_amount in
  total_amount = total_amount_donna_paid :=
by
  sorry

end donna_paid_correct_amount_l421_421952


namespace shaded_region_area_correct_l421_421181

-- Given that each semicircle has a diameter of 3 inches,
-- aligned in a pattern that extends for 18 inches.
def diameter : ℝ := 3
def length_of_pattern : ℝ := 18

-- The number of semicircles in the given length of the pattern
def number_of_semicircles := length_of_pattern / diameter

-- Each pair of semicircles forms a complete circle
def number_of_circles := number_of_semicircles / 2

-- The radius of each circle
def radius : ℝ := diameter / 2

-- Calculate the total area of the shaded region
def area_of_shaded_region : ℝ := number_of_circles * π * radius^2

-- Prove that the area of the shaded region in an 18-inch length of the pattern
-- is equal to 27/4 * π
theorem shaded_region_area_correct :
  area_of_shaded_region = (27 / 4) * π :=
by
  sorry

end shaded_region_area_correct_l421_421181


namespace find_AE_l421_421539

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end find_AE_l421_421539


namespace find_x_given_ratio_l421_421005

noncomputable def surface_area_F1 (a x : ℝ) : ℝ :=
  π * (a * x * (a - x)) / (a - 2 * x)

noncomputable def surface_area_F2 (a x : ℝ) : ℝ :=
  2 * π * (x * a * x) / (a - x)

theorem find_x_given_ratio (a : ℝ) (h : a > 0) : 
  (surface_area_F2 a (a / 3)) / (surface_area_F1 a (a / 3)) = 1 / 2 :=
by
  let x := a / 3
  have x_nonzero : x ≠ 0 := by exact div_ne_zero h (by norm_num)
  have ha : a ≠ 0 := h.ne'
  calc
    (surface_area_F2 a x) / (surface_area_F1 a x)
        = (2 * π * (x * a * x) / (a - x)) / (π * (a * x * (a - x)) / (a - 2 * x)) : by sorry
    ... = 1 / 2 : by sorry

end find_x_given_ratio_l421_421005


namespace daily_average_rain_is_four_l421_421189

def monday_rain_morning : ℝ := 2
def monday_rain_afternoon : ℝ := 1
def tuesday_rain_total : ℝ := 2 * (monday_rain_morning + monday_rain_afternoon)
def wednesday_rain_total : ℝ := 0
def thursday_rain_total : ℝ := 1
def friday_rain_total : ℝ := monday_rain_morning + monday_rain_afternoon + tuesday_rain_total + wednesday_rain_total + thursday_rain_total

def total_rain_week : ℝ := monday_rain_morning + monday_rain_afternoon + tuesday_rain_total + wednesday_rain_total + thursday_rain_total + friday_rain_total

def daily_average_rain_week : ℝ := total_rain_week / 5

theorem daily_average_rain_is_four :
  daily_average_rain_week = 4 := sorry

end daily_average_rain_is_four_l421_421189


namespace P_evaluation_at_1983_l421_421741

open Polynomial

-- Fibonacci sequence definition
def F : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := F (n+1) + F n

-- Polynomial P of degree 990 defined on specific range
noncomputable def P : Polynomial ℕ :=
  sorry  -- P(x) is explicitly given in the problem but we use sorry

-- Main theorem to prove
theorem P_evaluation_at_1983 : ∀ P : Polynomial ℕ, (deg P ≤ 990) → 
    (∀ k, 992 ≤ k ∧ k ≤ 1982 → eval (k : ℕ) P = F k) → eval (1983 : ℕ) P = F 1983 - 1 :=
sorry

end P_evaluation_at_1983_l421_421741


namespace place_20_knights_l421_421582

structure Board :=
  (width : ℕ)
  (height : ℕ)
  (blocked: list (ℕ × ℕ))
  (open_positions : list (ℕ × ℕ))

def knight_moves : List (ℕ × ℕ) := 
  [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

def knight_attacks (pos₁ pos₂ : ℕ × ℕ) : Prop :=
  (pos₁.1 + 2 = pos₂.1 ∧ pos₁.2 + 1 = pos₂.2) ∨
  (pos₁.1 + 2 = pos₂.1 ∧ pos₁.2 - 1 = pos₂.2) ∨
  (pos₁.1 - 2 = pos₂.1 ∧ pos₁.2 + 1 = pos₂.2) ∨
  (pos₁.1 - 2 = pos₂.1 ∧ pos₁.2 - 1 = pos₂.2) ∨
  (pos₁.1 + 1 = pos₂.1 ∧ pos₁.2 + 2 = pos₂.2) ∨
  (pos₁.1 + 1 = pos₂.1 ∧ pos₁.2 - 2 = pos₂.2) ∨
  (pos₁.1 - 1 = pos₂.1 ∧ pos₁.2 + 2 = pos₂.2) ∨
  (pos₁.1 - 1 = pos₂.1 ∧ pos₁.2 - 2 = pos₂.2)

def board_config : Board :=
  { width := 6,
    height := 6,
    blocked := [(0, 0), (0, 1), (1, 0), (1, 1),
                (0, 5), (0, 4), (1, 5), (1, 4),
                (5, 0), (5, 1), (4, 0), (4, 1),
                (5, 5), (5, 4), (4, 5), (4, 4)],
    open_positions := [(x, y) | x <- [0..5], y <- [0..5], (x, y) ∉ [(0, 0), (0, 1), (1, 0), (1, 1),
                                                                    (0, 4), (0, 5), (1, 4), (1, 5), 
                                                                    (4, 0), (4, 1), (5, 0), (5, 1),
                                                                    (4, 4), (4, 5), (5, 4), (5, 5)]] }

noncomputable def place_knights (colors : fin 10) (config : Board) :
  Prop :=
  ∀ c ∈ colors, ∃ k1 k2 ∈ config.open_positions, knight_attacks k1 k2

theorem place_20_knights :
  place_knights (fin 10) board_config :=
sorry

end place_20_knights_l421_421582


namespace sum_digits_of_t_l421_421347

noncomputable def factorial_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + factorial_trailing_zeros (n / 5)

noncomputable def trailing_zeros_squared (n : ℕ) : ℕ :=
  2 * factorial_trailing_zeros n

noncomputable def trailing_zeros_threefactorial (n : ℕ) : ℕ :=
  factorial_trailing_zeros (3 * n)

theorem sum_digits_of_t {n k t : ℕ} (hn : n ≥ 10)
  (h3n : trailing_zeros_threefactorial n = 4 * k)
  (hn2 : trailing_zeros_squared n = 3 * k)
  (ht : t = 15 + 20 + 25 + 30) :
  sum_digits t = 9 :=
by 
  sorry

end sum_digits_of_t_l421_421347


namespace ways_A_not_head_is_600_l421_421139

-- Definitions for the problem conditions
def num_people : ℕ := 6
def valid_positions_for_A : ℕ := 5
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The total number of ways person A can be placed in any position except the first
def num_ways_A_not_head : ℕ := valid_positions_for_A * factorial (num_people - 1)

-- The theorem to prove
theorem ways_A_not_head_is_600 : num_ways_A_not_head = 600 := by
  sorry

end ways_A_not_head_is_600_l421_421139


namespace problem_1_problem_2_l421_421624

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem problem_1 (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m * n > 1) :
  f(m) ≥ 0 ∨ f(n) ≥ 0 :=
sorry

theorem problem_2 (a b : ℝ) (hab : a ≠ b) (ha : 0 < a) (hb : 0 < b) (hfab : f(a) = f(b)) :
  a + b > 1 :=
sorry

end problem_1_problem_2_l421_421624


namespace grasshopper_distance_l421_421767

theorem grasshopper_distance :
  ∀ (a b : ℝ), 0 < a ∧ 0 < b ∧ irrational (b / a) → 
  ∃ (n : ℕ), 
  let pos := if n % 2 = 0 then (-a) else b in
  abs pos < 10 ^ (-6) :=
by sorry

end grasshopper_distance_l421_421767


namespace sum_of_C_100_terms_l421_421634

def sequence (n : ℕ) := ℝ

def S : ℕ → ℝ
def T : ℕ → ℝ
def a (n : ℕ) : ℝ := S n - S (n - 1)
def b (n : ℕ) : ℝ := T n - T (n - 1)
def C (n : ℕ) : ℝ := a n * T n + b n * S n - a n * b n

theorem sum_of_C_100_terms :
  S 100 = 41 →
  T 100 = 49 →
  ∑ i in Finset.range 100, C (i + 1) = 2009 :=
by
  intros hS100 hT100
  sorry

end sum_of_C_100_terms_l421_421634


namespace inequality_l421_421517

variable {α : Type*} 

/-- A sequence of positive real numbers that satisfies the given condition -/
variable (x : ℕ → ℝ) 

/-- The condition that the sequence should satisfy -/
axiom pos_real (n : ℕ) (hn : n > 0) : 0 < x n
axiom seq_condition (n : ℕ) (hn : n > 0) : x (n-1) * x (n+1) ≤ (x n) ^ 2

/-- Definitions of a_n and b_n -/
noncomputable def a_n (n : ℕ) : ℝ := (finset.range (n+1)).sum (λ i, x i) / (n+1)
noncomputable def b_n (n : ℕ) : ℝ := (finset.range n).sum (λ i, x (i + 1)) / n

/-- The task is to show the inequality -/
theorem inequality (n : ℕ) (hn : n > 0) : a_n x n * b_n x (n-1) ≥ a_n x (n-1) * b_n x n := 
sorry

end inequality_l421_421517


namespace polynomial_has_rational_root_l421_421343

theorem polynomial_has_rational_root {P : Polynomial ℚ} (hP_deg : P.degree = 5) 
  (h_double_root : ∃ z : ℂ, P.eval z = 0 ∧ P.derivative.eval z = 0) : 
  ∃ r : ℚ, P.eval r = 0 := 
sorry

end polynomial_has_rational_root_l421_421343


namespace infinite_non_prime_numbers_l421_421373

theorem infinite_non_prime_numbers : ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ (¬(Nat.Prime (2 ^ (2 ^ m) + 1) ∨ ¬Nat.Prime (2018 ^ (2 ^ m) + 1))) := sorry

end infinite_non_prime_numbers_l421_421373


namespace number_of_girls_l421_421858

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l421_421858


namespace solve_absolute_value_eq_l421_421913

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421913


namespace basketball_player_height_l421_421068

noncomputable def player_height (H : ℝ) : Prop :=
  let reach := 22 / 12
  let jump := 32 / 12
  let total_rim_height := 10 + (6 / 12)
  H + reach + jump = total_rim_height

theorem basketball_player_height : ∃ H : ℝ, player_height H → H = 6 :=
by
  use 6
  sorry

end basketball_player_height_l421_421068


namespace remainder_of_product_mod_7_l421_421574

   theorem remainder_of_product_mod_7 :
     (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := 
   by
     sorry
   
end remainder_of_product_mod_7_l421_421574


namespace sophie_bought_4_boxes_l421_421033

theorem sophie_bought_4_boxes :
  (exists (boxes_donuts : Nat) (each_box : Nat) (gave_away : Nat) (left_for_her : Nat),
    each_box = 12 ∧
    gave_away = 18 ∧
    left_for_her = 30 ∧
    (boxes_donuts * each_box) = (gave_away + left_for_her) ∧
    boxes_donuts = 4) :=
begin
  sorry -- Proof omitted as per instructions
end

end sophie_bought_4_boxes_l421_421033


namespace second_offset_length_l421_421197

noncomputable def quadrilateral_area (d o1 o2 : ℝ) : ℝ :=
  (1 / 2) * d * (o1 + o2)

theorem second_offset_length (d o1 A : ℝ) (h_d : d = 22) (h_o1 : o1 = 9) (h_A : A = 165) :
  ∃ o2, quadrilateral_area d o1 o2 = A ∧ o2 = 6 := by
  sorry

end second_offset_length_l421_421197


namespace minimum_number_is_correct_l421_421833

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct_l421_421833


namespace prove_every_integer_in_S_l421_421740

noncomputable def problem_statement (a b : ℕ) (S : Set ℕ) [a ≠ 1 ∨ b ≠ 1] : Prop :=
  gcd a b = 1 ∧ a ∈ S ∧ b ∈ S ∧
  (∀ x y z : ℕ, x ∈ S → y ∈ S → z ∈ S → x + y + z ∈ S) →
  (∀ n : ℕ, n > 2 * a * b → n ∈ S)

theorem prove_every_integer_in_S (a b : ℕ) (S : Set ℕ)
  [a_pos : 0 < a] [b_pos : 0 < b] [a_not_one : a ≠ 1] [b_not_one : b ≠ 1] :
  problem_statement a b S :=
begin
  sorry
end

end prove_every_integer_in_S_l421_421740


namespace increasing_sequence_lambda_range_l421_421605

theorem increasing_sequence_lambda_range (λ : ℝ) :
  (∀ n : ℕ, 0 < n ⟹ a (n + 1) - a n > 0) ⟹ λ > -3 :=
by
  -- We define the sequence a_n
  let a : ℕ → ℝ := λ n, n^2 + λ * n
  -- Since the sequence a_n is increasing
  assume h : ∀ n : ℕ, 0 < n ⟹ a (n + 1) - a n > 0
  -- Show that λ > -3
  sorry  -- Proof is omitted

end increasing_sequence_lambda_range_l421_421605


namespace max_true_statements_l421_421736

theorem max_true_statements (x y : ℝ) :
  ∀ s : Finset ℕ, ∀ h : s ⊆ {1, 2, 3, 4, 5},
  (∀ i ∈ s, (i = 1 → 1 / x > 1 / y) ∧
            (i = 2 → x^2 < y^2) ∧
            (i = 3 → x > y) ∧
            (i = 4 → x > 0) ∧
            (i = 5 → y > 0)) →
  s.card ≤ 3 := 
begin
  sorry
end

end max_true_statements_l421_421736


namespace smallest_m_for_integral_solutions_l421_421470

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), (∀ (p q : ℤ), (15 * (p * p) - m * p + 630 = 0 ∧ 15 * (q * q) - m * q + 630 = 0) → (m = 195)) :=
sorry

end smallest_m_for_integral_solutions_l421_421470


namespace cosine_of_angle_between_AB_AC_l421_421200

-- Definitions of the points A, B, C
def A : ℝ × ℝ × ℝ := (2, -8, -1)
def B : ℝ × ℝ × ℝ := (4, -6, 0)
def C : ℝ × ℝ × ℝ := (-2, -5, -1)

-- Definitions of the vectors AB and AC
def vecAB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def vecAC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Function to find the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Function to find the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- The final proof statement
theorem cosine_of_angle_between_AB_AC :
  let cos_theta := dot_product vecAB vecAC / (magnitude vecAB * magnitude vecAC)
  in cos_theta = -2 / 15 :=
sorry

end cosine_of_angle_between_AB_AC_l421_421200


namespace no_non_constant_polynomial_takes_powers_of_two_at_positive_integers_l421_421705

open BigOperators

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

theorem no_non_constant_polynomial_takes_powers_of_two_at_positive_integers :
  ¬ ∃ (f : ℤ[X]), (¬ f.degree = 0) ∧ (∀ n : ℕ, 0 < n → is_power_of_two (f.eval n)) := 
sorry

end no_non_constant_polynomial_takes_powers_of_two_at_positive_integers_l421_421705


namespace find_percentage_per_annum_l421_421804

theorem find_percentage_per_annum (BG BD : ℝ) (T : ℝ) : 
  BG = 684 → BD = 1634 → T = 6 → let TD := BD - BG in TD = (BD - BG) → 
  (TD * 100) / (BD * T) ≈ 28.67 :=
by
  intros hBG hBD hT hTD
  rw [hBG, hBD, hT, hTD]
  sorry

end find_percentage_per_annum_l421_421804


namespace digit_in_150th_place_of_5_over_11_l421_421113

theorem digit_in_150th_place_of_5_over_11 :
  let dec_rep := "45" -- represents the repeating part of the decimal
  (dec_rep[(150 - 1) % 2] = '5') := 
  sorry

end digit_in_150th_place_of_5_over_11_l421_421113


namespace card_B_eq_10_l421_421287

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set (ℕ × ℕ) := { p | p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 - p.2) ∈ A }

theorem card_B_eq_10 : (B.toFinset.card = 10) :=
by
  sorry

end card_B_eq_10_l421_421287


namespace uphill_speed_correct_l421_421938

noncomputable def uphill_speed (distance_up: ℕ) (speed_down: ℕ) (distance_down: ℕ) (average_speed: ℝ) : ℝ :=
  let total_distance := distance_up + distance_down
  let time_down := distance_down / speed_down.to_real
  let time_up := distance_up.to_real / average_speed
  total_distance.to_real / (time_up + time_down)

theorem uphill_speed_correct :
  uphill_speed 100 40 50 32.73 = 30 :=
by
  sorry

end uphill_speed_correct_l421_421938


namespace tangent_line_equation_at_A_l421_421811

theorem tangent_line_equation_at_A :
  ∀ (f : ℝ → ℝ) (A : ℝ × ℝ), 
  f = (λ x, x^2 - 2 * x + 3) ∧ 
  A = (-1, 6) → 
  ∃ (m b : ℝ), 
  m = -4 ∧ b = 2 ∧ 
  (∀ x : ℝ, m * x + b = 4 * x + 6 - 2) :=
begin
  sorry
end

end tangent_line_equation_at_A_l421_421811


namespace symmetric_point_origin_l421_421417

-- Define the coordinates of point A and the relation of symmetry about the origin
def A : ℝ × ℝ := (2, -1)
def symm_origin (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Theorem statement: Point B is the symmetric point of A about the origin
theorem symmetric_point_origin : symm_origin A = (-2, 1) :=
  sorry

end symmetric_point_origin_l421_421417


namespace equilateral_triangle_area_l421_421405

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l421_421405


namespace larger_integer_exists_l421_421097

theorem larger_integer_exists : ∃ (a b : ℕ), a * b = 198 ∧ abs (a - b) = 8 ∧ max a b = 18 := by
  sorry

end larger_integer_exists_l421_421097


namespace find_r2_l421_421792

noncomputable def p (x : ℝ) : ℝ := x ^ 8
noncomputable def d1 (x : ℝ) : ℝ := x + 1/2
noncomputable def d2 (x : ℝ) : ℝ := x - 1/2

theorem find_r2 :
  let q1_x_r1 := div_rem p d1,
      q1 := q1_x_r1.1,
      r1 := q1_x_r1.2,
      q2_x_r2 := div_rem q1 d2,
      q2 := q2_x_r2.1,
      r2 := q2_x_r2.2 in
  r2 = 0 := by
  sorry

end find_r2_l421_421792


namespace problem_proof_l421_421310

noncomputable def abc_triangle (A B C a b c S : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧
  C < π ∧ A + B + C = π ∧ 
  4 * S = Real.sqrt(3) * (a^2 + b^2 - c^2) ∧
  1 + Real.tan A / Real.tan B = 2 * c / b ∧ 
  ∥(V2.mk b 0) - (V2.mk 0 a)∥ * ∥(V2.mk b 0) - (V2.mk c 0)∥ * Real.cos B = -8

theorem problem_proof (A B C a b c S : ℝ) 
  (h : abc_triangle A B C a b c S) 
  : C = π/3 ∧ c = 4 :=
by
  cases h with a_pos b_pos c_pos A_pos B_pos C_pos angles_sum S_eq tan_eq dot_eq
  sorry

end problem_proof_l421_421310


namespace volume_of_prism_l421_421514

variable (l w h : ℝ)

def area1 (l w : ℝ) : ℝ := l * w
def area2 (w h : ℝ) : ℝ := w * h
def area3 (l h : ℝ) : ℝ := l * h
def volume (l w h : ℝ) : ℝ := l * w * h

axiom cond1 : area1 l w = 15
axiom cond2 : area2 w h = 20
axiom cond3 : area3 l h = 30

theorem volume_of_prism : volume l w h = 30 * Real.sqrt 10 :=
by
  sorry

end volume_of_prism_l421_421514


namespace digit_in_150th_place_of_5_over_11_l421_421112

theorem digit_in_150th_place_of_5_over_11 :
  let dec_rep := "45" -- represents the repeating part of the decimal
  (dec_rep[(150 - 1) % 2] = '5') := 
  sorry

end digit_in_150th_place_of_5_over_11_l421_421112


namespace find_phi_l421_421247

theorem find_phi (ϕ : ℝ) (h0 : 0 ≤ ϕ) (h1 : ϕ < π)
    (H : 2 * Real.cos (π / 3) = 2 * Real.sin (2 * (π / 3) + ϕ)) : ϕ = π / 6 :=
by
  sorry

end find_phi_l421_421247


namespace number_of_possible_measures_angle_C_l421_421428

theorem number_of_possible_measures_angle_C : 
  ∃ (C D : ℕ) (k : ℕ), (C > 0) ∧ (D > 0) ∧ (k ≥ 1) ∧ (C = k * D) ∧ (C + D = 180) ∧
  (17 = (do_n_count ((d : ℕ) → ∃ (k : ℕ), (k ≥ 1) ∧ ((k + 1) * d = 180)))) :=
sorry

end number_of_possible_measures_angle_C_l421_421428


namespace clara_bikes_more_than_david_l421_421424

-- Definitions for the conditions
noncomputable def clara_line : ℝ → ℝ := λ t, (4.5 / 6) * t
noncomputable def david_line : ℝ → ℝ := λ t, (3.6 / 6) * t
def hour_to_coordinate_unit (h : ℝ) : ℝ := h
def mile_to_coordinate_unit (m : ℝ) : ℝ := m / 15

-- Time in hours and converted to coordinate unit
def t : ℝ := 6

-- Proof statement
theorem clara_bikes_more_than_david :
  (mile_to_coordinate_unit (clara_line t * 15) - mile_to_coordinate_unit (david_line t * 15)) = 13.5 :=
by {
  -- Skipping the proof
  sorry
}

end clara_bikes_more_than_david_l421_421424


namespace price_of_third_variety_l421_421922

-- Define the constants
def price_first_variety := 126
def price_second_variety := 135
def ratio_a := 1
def ratio_b := 1
def ratio_c := 2
def mixture_price := 152

-- Define the proof problem statement
theorem price_of_third_variety :
  let P := ((mixture_price * (ratio_a + ratio_b + ratio_c)) - price_first_variety * ratio_a - price_second_variety * ratio_b) / (2 * ratio_c)
  P = 173.5 :=
  by
    -- We skip the proof itself with 'sorry'
    sorry

end price_of_third_variety_l421_421922


namespace geometric_sequence_solution_l421_421841

theorem geometric_sequence_solution (a : ℕ → ℝ) (n : ℕ) (q : ℝ) :
  a 1 = 2 →
  a 4 = -2 →
  ∀ n : ℕ, a n = 2 * (-1)^(n-1) ∧ (∑ i in finset.range (9+1), a i) = 2 :=
by { 
  sorry 
}

end geometric_sequence_solution_l421_421841


namespace books_per_week_l421_421764

-- Define the conditions
def total_books_read : ℕ := 20
def weeks : ℕ := 5

-- Define the statement to be proved
theorem books_per_week : (total_books_read / weeks) = 4 := by
  -- Proof omitted
  sorry

end books_per_week_l421_421764


namespace counterexample_sum_acute_angles_l421_421970

-- Definitions for angle types
def is_acute (θ : ℝ) := θ < 90
def is_angle := {θ : ℝ // 0 ≤ θ ∧ θ ≤ 180}

-- Given angles
def angle_A1 : is_angle := ⟨20, by linarith⟩
def angle_B1 : is_angle := ⟨60, by linarith⟩
def angle_A2 : is_angle := ⟨50, by linarith⟩
def angle_B2 : is_angle := ⟨90, by linarith⟩
def angle_A3 : is_angle := ⟨40, by linarith⟩
def angle_B3 : is_angle := ⟨50, by linarith⟩
def angle_A4 : is_angle := ⟨40, by linarith⟩
def angle_B4 : is_angle := ⟨100, by linarith⟩

-- Proving option (A3, B3) is the counterexample
theorem counterexample_sum_acute_angles :
  ¬ is_acute (angle_A3.val + angle_B3.val) :=
begin
  have hA3 : is_acute angle_A3.val := by linarith,
  have hB3 : is_acute angle_B3.val := by linarith,
  show ¬ is_acute (angle_A3.val + angle_B3.val),
  -- skip the proof
  sorry
end

end counterexample_sum_acute_angles_l421_421970


namespace totalProblemsSolved_l421_421523

-- Given conditions
def initialProblemsSolved : Nat := 45
def additionalProblemsSolved : Nat := 18

-- Statement to prove the total problems solved equals 63
theorem totalProblemsSolved : initialProblemsSolved + additionalProblemsSolved = 63 := 
by
  sorry

end totalProblemsSolved_l421_421523


namespace num_of_integer_terms_l421_421069

-- Sequence definition
def seq (n : ℕ) : ℤ := 6075 / (3 ^ n)

-- Condition: The sequence starts at 6075 and is repeatedly divided by 3
lemma sequence_property (n : ℕ) : (3 ^ 5 % 3 ^ (5 - n) = 0) ↔ (5 ≥ n) :=
begin
  sorry
end

-- Proving that there are exactly 6 integers in the sequence
lemma num_integers_in_sequence : set.countable {n : ℕ | ∃ (k : ℕ), seq k = n} :=
begin
  sorry
end

-- Final result
theorem num_of_integer_terms : (set.finite {n : ℕ | ∃ (k : ℕ), seq k = n}) ∧ finset.card {n : ℕ | ∃ (k : ℕ), seq k = n} = 6 :=
begin
  sorry
end

end num_of_integer_terms_l421_421069


namespace stratified_sampling_l421_421438

theorem stratified_sampling (N : ℕ) (r1 r2 r3 : ℕ) (sample_size : ℕ) 
  (ratio_given : r1 = 5 ∧ r2 = 2 ∧ r3 = 3) 
  (total_sample_size : sample_size = 200) :
  sample_size * r3 / (r1 + r2 + r3) = 60 := 
by
  sorry

end stratified_sampling_l421_421438


namespace sin_cos_eq_negative_one_has_solution_set_l421_421840

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (2 * n - 1) * Real.pi ∨ x = 2 * n * Real.pi - Real.pi / 2

theorem sin_cos_eq_negative_one_has_solution_set :
  {x : ℝ | sin x + cos x = -1} = {x : ℝ | solution_set x} :=
by
  sorry

end sin_cos_eq_negative_one_has_solution_set_l421_421840


namespace possible_measures_A_l421_421825

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l421_421825


namespace quadratic_inequality_solution_correct_l421_421606

noncomputable
def quadratic_inequality_solution (a : ℝ) : set ℝ :=
  if a = 0 then {x | x <= 0}
  else if a < 0 then {x | x <= (a - 1 + Real.sqrt (1 - 2 * a)) / a} ∪ {x | x >= (a - 1 - Real.sqrt (1 - 2 * a)) / a}
  else if a = 1 / 2 then {-1}
  else if a > 1 / 2 then ∅
  else {x | (a - 1 - Real.sqrt (1 - 2 * a)) / a ≤ x ∧ x ≤ (a - 1 + Real.sqrt (1 - 2 * a)) / a}

theorem quadratic_inequality_solution_correct (a : ℝ) :
  ∀ x : ℝ, x ∈ quadratic_inequality_solution a ↔ a * x^2 - 2 * (a - 1) * x + a ≤ 0 :=
sorry

end quadratic_inequality_solution_correct_l421_421606


namespace sum_of_solutions_l421_421683

theorem sum_of_solutions (y : ℝ) (h1 : y = 8) (h2 : ∀ x : ℝ, x^2 + y^2 = 289 → x = 15 ∨ x = -15) : 15 + (-15) = 0 :=
by
  simp
  exact 0

end sum_of_solutions_l421_421683


namespace sum_of_roots_is_zero_all_roots_real_third_root_is_neg_a_l421_421383

variable (a b : ℝ)

/-- Prove that for the polynomial equation x^3 + a x^2 + b x + a b = 0,
    the sum of the two roots from the factor x^2 + b is zero -/
theorem sum_of_roots_is_zero (h : x^3 + a * x^2 + b * x + a * b = 0) :
  let r1 := sqrt (-b)
  let r2 := -sqrt (-b)
  r1 + r2 = 0 :=
sorry

/-- Show that given a ∈ ℝ and b ≤ 0, all roots of the equation are real -/
theorem all_roots_real (h1 : a ∈ ℝ) (h2 : b ≤ 0) :
  ∀ (x1 x2 x3 : ℝ), x1 = -a ∧ x2 = sqrt (-b) ∧ x3 = - sqrt (-b) :=
sorry

/-- Prove that the third root from the factor x + a is -a -/
theorem third_root_is_neg_a (h : x + a = 0) : 
  x = -a :=
sorry

end sum_of_roots_is_zero_all_roots_real_third_root_is_neg_a_l421_421383


namespace region_area_proof_l421_421990

noncomputable def compute_region_area (r : ℝ) (angle : ℝ) : ℝ := 
  let triangle_area := 1/2 * r * r
  let sector_area := (angle / 360) * π * r * r
  sector_area - triangle_area

theorem region_area_proof (r : ℝ) (angle : ℝ) (a b c : ℝ) (hb : b = 1) (hc : c = 18.75)
  (ha : a = 0) :
  r = 5 ∧ angle = 90 → a + b + c = 19.75 :=
by
  intros hr_angle
  sorry

end region_area_proof_l421_421990


namespace period_of_f_monotonic_increase_intervals_of_f_max_value_of_f_in_interval_l421_421746

def f (x : ℝ) : ℝ := 2 * cos x * (cos x + sqrt 3 * sin x)

theorem period_of_f : ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem monotonic_increase_intervals_of_f : ∀ k : ℤ, 
  ∀ x : ℝ, kπ - π / 3 < x ∧ x < kπ + π / 6 → f x < f (x + h) :=
by sorry 

theorem max_value_of_f_in_interval : ∀ x : ℝ, 
  0 ≤ x ∧ x ≤ π / 2 → f x ≤ 3 :=
by sorry

end period_of_f_monotonic_increase_intervals_of_f_max_value_of_f_in_interval_l421_421746


namespace smallest_product_l421_421015

theorem smallest_product : 
  ∃ a b c d : ℕ, 
    {a, b, c, d} = {1, 2, 3, 4} ∧ 
    (a * 10 + b) * (c * 10 + d) = 312 := 
by
  have h1 : (1 * 10 + 3) * (2 * 10 + 4) = 312 := by norm_num
  existsi 1
  existsi 3
  existsi 2
  existsi 4
  split
  { exact finset.mk [1, 2, 3, 4] finset.nodup.cons:[1, 2, 3, 4] }
  exact h1

end smallest_product_l421_421015


namespace remainder_when_divided_by_x_minus_2_l421_421106

def f (x : ℝ) : ℝ := x^5 - 6 * x^4 + 11 * x^3 + 21 * x^2 - 17 * x + 10

theorem remainder_when_divided_by_x_minus_2 : (f 2) = 84 := by
  sorry

end remainder_when_divided_by_x_minus_2_l421_421106


namespace tank_fill_time_l421_421369

-- Define the conditions
def capacity := 800
def rate_A := 40
def rate_B := 30
def rate_C := -20

def net_rate_per_cycle := rate_A + rate_B + rate_C
def cycle_duration := 3
def total_cycles := capacity / net_rate_per_cycle
def total_time := total_cycles * cycle_duration

-- The proof that tank will be full after 48 minutes
theorem tank_fill_time : total_time = 48 := by
  sorry

end tank_fill_time_l421_421369


namespace inequality_solution_l421_421222

theorem inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
sorry

end inequality_solution_l421_421222


namespace least_four_digit_palindrome_divisible_by_5_is_1551_l421_421012

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in
  s = s.reverse

noncomputable def least_four_digit_palindrome_divisible_by_5 : ℕ :=
  if h : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ is_palindrome n ∧ n % 5 = 0 
  then @WellFounded.min ℕ lt (by {apply_instance}) {n | 1000 ≤ n ∧ n < 10000 ∧ is_palindrome n ∧ n % 5 = 0} h
  else 0

theorem least_four_digit_palindrome_divisible_by_5_is_1551 : least_four_digit_palindrome_divisible_by_5 = 1551 := 
by
  sorry

end least_four_digit_palindrome_divisible_by_5_is_1551_l421_421012


namespace merchant_markup_percentage_l421_421946

theorem merchant_markup_percentage (CP MP SP : ℝ) (x : ℝ) (H_CP : CP = 100)
  (H_MP : MP = CP + (x / 100 * CP)) 
  (H_SP_discount : SP = MP * 0.80) 
  (H_SP_profit : SP = CP * 1.12) : 
  x = 40 := 
by
  sorry

end merchant_markup_percentage_l421_421946


namespace find_log3_x14_l421_421425

def is_geometric_sequence {α : Type} [linear_ordered_field α] (x : ℕ → α) (a r : α) : Prop :=
∀ n, x n = a * r^n

theorem find_log3_x14 
    (x : ℕ → ℚ)
    (h_geom : ∃ (a r : ℚ), a > 0 ∧ r > 1 ∧ is_geometric_sequence x a r)
    (h_sum_logs : ∑ n in finset.range 8, real.logb 3 (x n) = 308)
    (h_log_sum_bounds : 56 ≤ real.logb 3 (∑ n in finset.range 8, x n) ∧ 
                        real.logb 3 (∑ n in finset.range 8, x n) ≤ 57) :
    real.logb 3 (x 14) = 91 :=
sorry

end find_log3_x14_l421_421425


namespace sum_of_primes_between_30_and_40_l421_421108

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_primes_between_30_and_40 :
  (∑ n in [31, 37].toFinset, n) = 68 :=
by
  have h1 : is_prime 31 := sorry
  have h2 : is_prime 37 := sorry
  have h3 : ∀ n, n ∈ [31, 37].toFinset → is_prime n := by 
    intros n hn
    cases hn
    · exact h1
    · cases hn
      · exact h2
      · cases hn
  exact by simp [Finset.sum_singleton, add_comm]; exact 68

end sum_of_primes_between_30_and_40_l421_421108


namespace abs_eq_condition_l421_421904

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421904


namespace volume_of_cylinder_l421_421813

noncomputable def cylinder_volume (a α : ℝ) : ℝ :=
  (a^3 * (Real.cos α)^2 * Real.sin α) / (4 * Real.pi)

theorem volume_of_cylinder (a α V : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ Real.pi / 2) -- condition for angle α in the range of valid angles.
  : cylinder_volume a α = V 
    → V = (a^3 * (Real.cos α)^2 * Real.sin α) / (4 * Real.pi) :=
by 
  sorry

end volume_of_cylinder_l421_421813


namespace increasing_interval_f_l421_421622

def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def is_increasing (a b : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x y, a < x → x < y → y < b → f x < f y

theorem increasing_interval_f : is_increasing 0 e f :=
sorry

end increasing_interval_f_l421_421622


namespace findYearsForTwiceAge_l421_421155

def fatherSonAges : ℕ := 33

def fatherAge : ℕ := fatherSonAges + 35

def yearsForTwiceAge (x : ℕ) : Prop :=
  fatherAge + x = 2 * (fatherSonAges + x)

theorem findYearsForTwiceAge : ∃ x, yearsForTwiceAge x :=
  ⟨2, sorry⟩

end findYearsForTwiceAge_l421_421155


namespace compute_c_plus_d_l421_421219

-- Define the conditions
variables (c d : ℕ) 

-- Conditions:
-- Positive integers
axiom pos_c : 0 < c
axiom pos_d : 0 < d

-- Contains 630 terms
axiom term_count : d - c = 630

-- The product of the logarithms equals 2
axiom log_product : (Real.log d) / (Real.log c) = 2

-- Theorem to prove
theorem compute_c_plus_d : c + d = 1260 :=
sorry

end compute_c_plus_d_l421_421219


namespace original_amount_water_l421_421943

theorem original_amount_water (O : ℝ) (h1 : (0.75 = 0.05 * O)) : O = 15 :=
by sorry

end original_amount_water_l421_421943


namespace num_positive_integers_le_500_l421_421572

-- Define a predicate to state that a number is a perfect square
def is_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- Define the main theorem
theorem num_positive_integers_le_500 (n : ℕ) :
  (∃ (ns : Finset ℕ), (∀ x ∈ ns, x ≤ 500 ∧ is_square (21 * x)) ∧ ns.card = 4) :=
by
  sorry

end num_positive_integers_le_500_l421_421572


namespace pete_minimum_cells_l421_421676

theorem pete_minimum_cells (n : ℕ) (table : ℕ → ℕ → ℕ) (h1 : ∀ (i j : ℕ), 1 ≤ table i j ∧ table i j ≤ 9) : 
  ∃ (cells : fin n → fin n), 
  (∀ i, 1 ≤ table (cells i) (cells i) ∧ table (cells i) (cells i) ≤ 9) ∧ 
  (∀ (a : string), (a.length = n) → 
                   (∀ k, 1 ≤ string.to_nat (string.get_digit a k) ∧ string.to_nat (string.get_digit a k) ≤ 9) → 
                   (a ≠ a.reverse)) :=
sorry

end pete_minimum_cells_l421_421676


namespace pears_sold_in_afternoon_l421_421515

theorem pears_sold_in_afternoon (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : m + a = total) (h4 : total = 360) :
  a = 240 :=
by
  sorry

end pears_sold_in_afternoon_l421_421515


namespace square_area_of_triangle_on_hyperbola_l421_421443

noncomputable def centroid_is_vertex (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ triangle ∧ v.1 * v.2 = 4

noncomputable def triangle_properties (triangle : Set (ℝ × ℝ)) : Prop :=
  centroid_is_vertex triangle ∧
  (∃ centroid : ℝ × ℝ, 
    centroid_is_vertex triangle ∧ 
    (∀ p ∈ triangle, centroid ∈ triangle))

theorem square_area_of_triangle_on_hyperbola :
  ∃ triangle : Set (ℝ × ℝ), triangle_properties triangle ∧ (∃ area_sq : ℝ, area_sq = 1728) :=
by
  sorry

end square_area_of_triangle_on_hyperbola_l421_421443


namespace triangle_perimeter_exradius_l421_421836

theorem triangle_perimeter_exradius (r_a r_b r_c : ℝ) (h1 : r_a = 3) (h2 : r_b = 10) (h3 : r_c = 15) :
  let s_perimeter := (2 * (r_a + r_b + r_c)) / 3 in
  s_perimeter = 30 :=
by
  sorry

end triangle_perimeter_exradius_l421_421836


namespace shaded_area_of_circle_l421_421693

theorem shaded_area_of_circle (r : ℝ) (h_r : r = 6) (angle : ℝ) (h_angle : angle = 120) :
  let area_triangles := 72 * Real.sqrt 3 in
  let area_sectors := 48 * Real.pi in
  area_triangles + area_sectors = 72 * Real.sqrt 3 + 48 * Real.pi :=
by 
  sorry

end shaded_area_of_circle_l421_421693


namespace students_suggested_bacon_l421_421978

-- Defining the conditions
def total_students := 310
def mashed_potatoes_students := 185

-- Lean statement for proving the equivalent problem
theorem students_suggested_bacon : total_students - mashed_potatoes_students = 125 := by
  sorry -- Proof is omitted

end students_suggested_bacon_l421_421978


namespace solution_set_l421_421194

theorem solution_set (x : ℝ) : (1/6 + |x - 1/3| < 1/2) ↔ x ∈ set.Ioo 0 (2/3) :=
by
  sorry

end solution_set_l421_421194


namespace find_p_plus_q_l421_421039

theorem find_p_plus_q
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 25 / 12)
  (p q : ℕ)
  (pq_lowest_terms : Nat.coprime p q)
  (h2 : Real.csc x + Real.cot x = p.to_rat / q.to_rat) :
  p + q = 1348 :=
by
  sorry

end find_p_plus_q_l421_421039


namespace number_of_girls_l421_421867

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l421_421867


namespace bird_population_in_1997_l421_421664

theorem bird_population_in_1997 
  (k : ℝ)
  (pop_1995 pop_1996 pop_1998 : ℝ)
  (h1 : pop_1995 = 45)
  (h2 : pop_1996 = 70)
  (h3 : pop_1998 = 145)
  (h4 : pop_1997 - pop_1995 = k * pop_1996)
  (h5 : pop_1998 - pop_1996 = k * pop_1997) : 
  pop_1997 = 105 :=
by
  sorry

end bird_population_in_1997_l421_421664


namespace area_of_triangle_l421_421398

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l421_421398


namespace angle_measures_possible_l421_421819

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l421_421819


namespace mass_percentage_of_Ba_l421_421703

theorem mass_percentage_of_Ba {BaX : Type} {molar_mass_Ba : ℝ} {compound_mass : ℝ} {mass_Ba : ℝ}:
  molar_mass_Ba = 137.33 ∧ 
  compound_mass = 100 ∧
  mass_Ba = 66.18 →
  (mass_Ba / compound_mass * 100) = 66.18 :=
by
  sorry

end mass_percentage_of_Ba_l421_421703


namespace tan_alpha_value_tan_beta_value_sum_angles_l421_421686

open Real

noncomputable def tan_alpha (α : ℝ) : ℝ := sin α / cos α
noncomputable def tan_beta (β : ℝ) : ℝ := sin β / cos β

def conditions (α β : ℝ) :=
  α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2 ∧ 
  sin α = 1 / sqrt 10 ∧ tan β = 1 / 7

theorem tan_alpha_value (α β : ℝ) (h : conditions α β) : tan_alpha α = 1 / 3 := sorry

theorem tan_beta_value (α β : ℝ) (h : conditions α β) : tan_beta β = 1 / 7 := sorry

theorem sum_angles (α β : ℝ) (h : conditions α β) : 2 * α + β = π / 4 := sorry

end tan_alpha_value_tan_beta_value_sum_angles_l421_421686


namespace most_suitable_student_l421_421695

theorem most_suitable_student {α : Type}
  (avg_score : α → ℝ)
  (var_score : α → ℝ)
  (A B C D : α)
  (h_avg : ∀ x, avg_score x = 180)
  (h_var_A : var_score A = 65)
  (h_var_B : var_score B = 56.5)
  (h_var_C : var_score C = 53)
  (h_var_D : var_score D = 50.5) :
  D = (argmin var_score {A, B, C, D}) := 
by
  unfold argmin
  sorry

end most_suitable_student_l421_421695


namespace prime_k_values_l421_421595

theorem prime_k_values (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by
  sorry

end prime_k_values_l421_421595


namespace number_of_special_functions_l421_421342

-- Defining the set A and the function f
def A (n : ℕ) := Fin n

-- Defining what it means for a function to be nondecreasing
def nondecreasing {n : ℕ} (f : A n → A n) : Prop :=
  ∀ ⦃x y : A n⦄, x ≤ y → f x ≤ f y

-- Defining the specific property of the function f
def special_property {n : ℕ} (f : A n → A n) : Prop :=
  ∀ ⦃x y : A n⦄, |f x - f y| ≤ |x - y|

-- Counting the number of such functions
def count_special_functions (n : ℕ) : ℕ :=
  n * 2^(n-1) - (n-1) * 2^(n-2)

-- The main statement to prove
theorem number_of_special_functions (n : ℕ) :
  ∃ count : ℕ, count = count_special_functions n := sorry

end number_of_special_functions_l421_421342


namespace arithmetic_sequence_sum_11_l421_421688

open Real

variable (a : ℕ → ℝ) 
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable h : ∀ n, a (n+1) = a n + d
variable h_seq : a 4 + a 5 + a 6 + a 7 + a 8 = 150

theorem arithmetic_sequence_sum_11 :
  S 11 = 330 :=
sorry

end arithmetic_sequence_sum_11_l421_421688


namespace median_of_remaining_rooms_is_16_l421_421527

-- Conditions
def rooms := (Finset.range 32) \ (Finset.singleton 15 ∪ Finset.singleton 20)

-- Proposition (Median calculation)
def median_room_number (r : Finset ℕ) : ℕ :=
  if h : r.card % 2 = 1 then
    r.sort (≤) (((r.card + 1) / 2) - 1)
  else
    sorry  -- Here we only care about the case where the number of elements is odd

theorem median_of_remaining_rooms_is_16 :
  median_room_number rooms = 16 :=
by
  sorry

end median_of_remaining_rooms_is_16_l421_421527


namespace orthogonal_projection_area_l421_421839

theorem orthogonal_projection_area (a b c : ℝ) : 
  a = 5 → b = 6 → c = 7 → 
  let cos_smallest_angle := (b^2 + c^2 - a^2) / (2 * b * c) in
  let S := Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) in
  let S1 := S * cos_smallest_angle in
  S1 = (30 * Real.sqrt 6) / 7 
:=
by {
  intros,
  sorry,
}

end orthogonal_projection_area_l421_421839


namespace possible_measures_of_angle_A_l421_421828

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l421_421828


namespace solve_absolute_value_eq_l421_421912

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421912


namespace hyperbola_eq_l421_421265

noncomputable def hyperbola_foci (F1 F2 : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  let MF1 := (M.1 - F1.1, M.2 - F1.2)
  let MF2 := (M.1 - F2.1, M.2 - F2.2)
  (MF1.1 * MF2.1 + MF1.2 * MF2.2 = 0) ∧
  (sqrt ((MF1.1)^2 + (MF1.2)^2) * sqrt ((MF2.1)^2 + (MF2.2)^2) = 2)
  
theorem hyperbola_eq :
  ∀ (M : ℝ × ℝ),
  hyperbola_foci (-(sqrt 10), 0) (sqrt 10, 0) M →
  ∀ (x y : ℝ), 
  (y = 0) → 
  (F1 := (-(sqrt 10), 0)) → 
  (F2 := (sqrt 10, 0)) →
  (x^2 / 9 - y^2 = 1) :=
by 
  sorry

end hyperbola_eq_l421_421265


namespace fn_conjecture_l421_421628

noncomputable def f0 (a b c d x : ℝ) : ℝ := (c * x + d) / (a * x + b)

def derivative_f0 (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : a * c - b * d ≠ 0) : (x : ℝ) → ℝ :=
  fun x => (b * c - a * d) / (a * x + b) ^ 2

def conjecture_fn (a b c d : ℝ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : a * c - b * d ≠ 0) : (x : ℝ) → ℝ :=
  fun x => (-1)^(n - 1) * a^(n - 1) * (b * c - a * d) * Nat.factorial n / (a * x + b) ^ (n + 1)

theorem fn_conjecture
  (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : a * c - b * d ≠ 0) (n : ℕ) (hn : 0 < n) :
  (derivative_f0 a b c d h₀ h₁ n x) = (conjecture_fn a b c d n h₀ h₁ x) :=
sorry

end fn_conjecture_l421_421628


namespace system_solution_l421_421390

theorem system_solution (n : ℕ) (n_pos : n > 0) (x : ℕ → ℚ) 
  (h1 : x 1 + 2 * x 2 + 3 * x 3 + ∀ (k : fin (n - 4)), k + 3 * x (k + 3) + (n - 1) * x (n - 1) + n * x n = n)
  (h2 : 2 * x 1 +  3 * x 2 + 4 * x 3 + ∀ (k : fin (n - 4)), (k + 3) * x (k + 4) + n * x n-1 + x n = n-1)
  (h3 : 3 * x 1 +  4 * x 2 + 5 * x 3 + ∀ (k : fin (n - 4)), (k + 3) * x (k + 5) + x n-2 + 2 * x n = n-2)   
  :
  x 1 = 2 / (n: ℚ) - 1 ∧ ∀ i ∈ fin (n-1), x (i+1) = 2 / (n: ℚ) := sorry

end system_solution_l421_421390


namespace positive_sequence_inequality_l421_421771

theorem positive_sequence_inequality (n : ℕ) (h : 2 ≤ n)
  (a : Fin n → ℝ) (pos : ∀ i : Fin n, 0 < a i) :
  (∑ i : Fin n, a i / (a ((i + 1) % n) + a ((i + 2) % n))) > n / 4 :=
by sorry

end positive_sequence_inequality_l421_421771


namespace find_long_tennis_players_l421_421313

variables (total_students football_players both_players neither_players : ℕ) (long_tennis_players : ℕ)

-- Given conditions
def conditions := 
  total_students = 40 ∧ 
  football_players = 26 ∧ 
  both_players = 17 ∧ 
  neither_players = 11

-- The goal to prove
def goal := long_tennis_players = 20

-- Proof statement
theorem find_long_tennis_players : 
  conditions total_students football_players both_players neither_players → 
  (total_students - neither_players) = (football_players + long_tennis_players - both_players) → 
  goal long_tennis_players :=
by
  intro h_conditions
  intro h_inclusion_exclusion
  split
  cases h_conditions with h1 h_rest
  cases h_rest with h2 h_rest'
  cases h_rest' with h3 h4
  rw [h1, h2, h3, h4] at h_inclusion_exclusion
  -- Now we have 29 = 26 + L - 17
  -- which simplifies to L = 20
  sorry

end find_long_tennis_players_l421_421313


namespace area_of_figure_l421_421078

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - 0) + x2 * (0 - y1) + 0 * (y1 - y2))

theorem area_of_figure : 
  let x1 := -5 in
  let y1 := -5 in
  let x2 := 0 in
  let y2 := 0 in
  area_of_triangle x1 y1 x2 y2 = 12.5 :=
by
  let x1 := -5
  let y1 := -5
  let x2 := 0
  let y2 := 0
  have area := area_of_triangle x1 y1 x2 y2
  simp [area_of_triangle, x1, y1, x2, y2, abs]
  norm_num
  sorry

end area_of_figure_l421_421078


namespace range_of_a_l421_421421

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- State the theorem that describes the condition and proves the answer
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 4 → x₂ < 4 → f a x₁ ≥ f a x₂) → a ≤ -3 :=
by
  -- The proof would go here; for now, we skip it
  sorry

end range_of_a_l421_421421


namespace probability_x_leq_neg_2_l421_421618

open ProbabilityTheory

noncomputable def random_variable_xi {Ω : Type*} [MeasureSpace Ω] : Ω → ℝ := sorry

axiom normal_distribution (σ : ℝ) (hσ : σ > 0) :
  is_normal random_variable_xi 1 σ

axiom probability_x_leq_4 (σ : ℝ) (hσ : σ > 0) :
  ProbabilityTheory.prob (random_variable_xi ≤ 4) = 0.86

theorem probability_x_leq_neg_2 (σ : ℝ) (hσ : σ > 0) :
  ProbabilityTheory.prob (random_variable_xi ≤ -2) = 0.14 := by
  sorry

end probability_x_leq_neg_2_l421_421618


namespace negation_of_exists_l421_421062

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ ln x > 0) ↔ ∀ x : ℝ, x > 0 → ln x ≤ 0 :=
by
  sorry

end negation_of_exists_l421_421062


namespace true_statements_sum_l421_421724

noncomputable def arithmetic_mean (x y : ℝ) : ℝ := (x + y) / 2
noncomputable def geometric_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)
noncomputable def harmonic_mean (x y : ℝ) : ℝ := 2 / ((1 / x) + (1 / y))

-- Given the sequences Aₙ, Gₙ, Hₙ defined as specified
def seq_A (a b c d : ℝ) : ℕ → ℝ
| 0     := (a + b + c + d) / 4
| n + 1 := arithmetic_mean (seq_A a b c d n) (seq_H a b c d n)

def seq_G (a b c d : ℝ) : ℕ → ℝ
| 0     := Real.sqrt (Real.sqrt (a * b * c * d))
| n + 1 := geometric_mean (seq_A a b c d n) (seq_H a b c d n)

def seq_H (a b c d : ℝ) : ℕ → ℝ
| 0     := 4 / (1 / a + 1 / b + 1 / c + 1 / d)
| n + 1 := harmonic_mean (seq_A a b c d n) (seq_A a b c d n)

-- Conditions
variables (a b c d : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)

-- The proof problem statement
theorem true_statements_sum :
  (seq_A a b c d 0 > seq_A a b c d 1 ∧ seq_A a b c d 1 > seq_A a b c d 2 ∧ ∀ n, seq_A a b c d n > seq_A a b c d (n + 1))
∧ (seq_G a b c d 0 = seq_G a b c d 1 ∧ seq_G a b c d 1 = seq_G a b c d 2 ∧ ∀ n, seq_G a b c d n = seq_G a b c d (n + 1))
∧ (seq_H a b c d 0 < seq_H a b c d 1 ∧ seq_H a b c d 1 < seq_H a b c d 2 ∧ ∀ n, seq_H a b c d n < seq_H a b c d (n + 1)) :=
sorry

end true_statements_sum_l421_421724


namespace time_taken_by_C_l421_421495

theorem time_taken_by_C (days_A B C : ℕ) (work_done_A work_done_B work_done_C : ℚ) 
  (h1 : days_A = 40) (h2 : work_done_A = 10 * (1/40)) 
  (h3 : days_B = 40) (h4 : work_done_B = 10 * (1/40)) 
  (h5 : work_done_C = 1/2)
  (h6 : 10 * work_done_C = 1/2) :
  (10 * 2) = 20 := 
sorry

end time_taken_by_C_l421_421495


namespace vector_at_s_neg2_l421_421504

noncomputable def vector_on_line (s : ℝ) : ℝ × ℝ :=
  let b := (0, 9)
  let e := (2, -4)
  (b.1 + s * e.1, b.2 + s * e.2)

theorem vector_at_s_neg2 :
  (vector_on_line (-2)) = (-4, 17) :=
by
  sorry

end vector_at_s_neg2_l421_421504


namespace cheryl_probability_same_color_l421_421143

noncomputable def probability_cheryl_same_color (total_marbles : ℕ) [decidable_eq ℕ] (draws : ℕ) :=
  let red_probability    := (3 / total_marbles) ^ draws in
  let green_probability  := (3 / total_marbles) ^ draws in
  let yellow_probability := (3 / total_marbles) ^ draws in
  red_probability + green_probability + yellow_probability

theorem cheryl_probability_same_color : probability_cheryl_same_color 9 3 = 1 / 9 :=
by
  have red_probability    := (3 / 9) ^ 3
  have green_probability  := (3 / 9) ^ 3
  have yellow_probability := (3 / 9) ^ 3
  calc
    probability_cheryl_same_color 9 3
        = red_probability + green_probability + yellow_probability : by simp [probability_cheryl_same_color]
    ... = (3 / 9) ^ 3 + (3 / 9) ^ 3 + (3 / 9) ^ 3 : by simp [red_probability, green_probability, yellow_probability]
    ... = 1/27 + 1/27 + 1/27 : by norm_num
    ... = 3 * (1/27) : by ring
    ... = 1/9 : by norm_num

end cheryl_probability_same_color_l421_421143


namespace equation_solution_l421_421220

theorem equation_solution (x : ℝ) (h : x + 1/x = 2.5) : x^2 + 1/x^2 = 4.25 := 
by sorry

end equation_solution_l421_421220


namespace ratio_XY_WZ_l421_421666

/-- Given conditions in the problem: -/
def side_length : ℝ := 4
def quarter_point (a b : ℝ) : Prop := a = b / 4
def is_perpendicular (a b : ℝ) : Prop := a * b = 0

/-- Given: -/
variable (A B E F G : ℝ)
variables (XY WZ : ℝ)

/-- Given conditions encoded in Lean definitions -/
axiom quarter_point_E : quarter_point E side_length
axiom quarter_point_F : quarter_point F side_length
axiom perpendicular_AG_BF : is_perpendicular G side_length

/-- Proving the ratio XY / WZ is 1 -/
theorem ratio_XY_WZ : XY / WZ = 1 :=
sorry

end ratio_XY_WZ_l421_421666


namespace distinct_nonzero_reals_product_l421_421609

theorem distinct_nonzero_reals_product 
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy: x ≠ y)
  (h : x + 3 / x = y + 3 / y) :
  x * y = 3 :=
sorry

end distinct_nonzero_reals_product_l421_421609


namespace value_of_a_l421_421126

variable (a : ℝ)

theorem value_of_a (h1 : (0.5 / 100) * a = 0.80) : a = 160 := by
  sorry

end value_of_a_l421_421126


namespace derivative_at_2_l421_421729

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 + 1)

theorem derivative_at_2 : deriv f 2 = 2 * real.sqrt 5 / 5 := by
  -- proof
  sorry

end derivative_at_2_l421_421729


namespace rank_friends_oldest_youngest_l421_421994

variable (Ages : String → Nat)
variable (h1 : Ages "Emma" ≠ Ages "Fiona")
variable (h2 : Ages "Emma" ≠ Ages "George")
variable (h3 : Ages "Emma" ≠ Ages "David")
variable (h4 : Ages "Fiona" ≠ Ages "George")
variable (h5 : Ages "Fiona" ≠ Ages "David")
variable (h6 : Ages "George" ≠ Ages "David")
variable (I : Ages "Emma" = max (Ages "Emma") (max (Ages "Fiona") (max (Ages "David") (Ages "George"))))
variable (II : not (Ages "Fiona" = max (Ages "Emma") (max (Ages "Fiona") (max (Ages "David") (Ages "George")))))
variable (III : not (Ages "David" = min (Ages "Emma") (min (Ages "Fiona") (min (Ages "David") (Ages "George")))))
variable (IV : not (Ages "George" = max (Ages "Emma") (max (Ages "Fiona") (max (Ages "David") (Ages "George")))))

theorem rank_friends_oldest_youngest : Exists.unique (λ (true_statement : Prop), true_statement ∈ [I, II, III, IV]) → 
  (Ages "Fiona" > Ages "Emma" ∧ Ages "Emma" > Ages "George" ∧ Ages "George" > Ages "David") :=
by
  sorry

end rank_friends_oldest_youngest_l421_421994


namespace product_of_numbers_l421_421434

theorem product_of_numbers (a b : ℕ) (hcf_val lcm_val : ℕ) 
  (h_hcf : Nat.gcd a b = hcf_val) 
  (h_lcm : Nat.lcm a b = lcm_val) 
  (hcf_eq : hcf_val = 33) 
  (lcm_eq : lcm_val = 2574) : 
  a * b = 84942 := 
by
  sorry

end product_of_numbers_l421_421434


namespace minimize_cost_ratio_l421_421882

noncomputable def ratio_min_cost (V a : ℝ) : ℝ :=
  let r := (V / (4 * π))^(1/3) in
  let h := V / (π * r^2) in
  r / h

theorem minimize_cost_ratio (V a : ℝ) (h : V > 0) (r : a > 0) :
  ratio_min_cost V a = 1 / 4 := by 
  sorry

end minimize_cost_ratio_l421_421882


namespace surface_area_of_sphere_cylinder_inscribed_l421_421331

theorem surface_area_of_sphere_cylinder_inscribed (d h : ℝ) (hd : d = 2) (hh : h = 2) :
  let r := Real.sqrt (d^2 / 4 + h^2 / 4)
  in 4 * Real.pi * r^2 = 8 * Real.pi :=
by
  sorry

end surface_area_of_sphere_cylinder_inscribed_l421_421331


namespace restore_digits_l421_421776

-- Problem statement: Restore the digits in the product of three consecutive even numbers such that the product has the form 87*****8

theorem restore_digits:
  ∃ (n : ℕ), let a := 2 * n in let b := 2 * n + 2 in let c := 2 * n + 4 in
             let product := a * b * c in
             product = 87526608
  ∧ (product / 10000000 = 87)
  ∧ (product % 10 = 8) := sorry

end restore_digits_l421_421776


namespace joy_choices_count_l421_421678

theorem joy_choices_count :
  {d : ℕ | 1 ≤ d ∧ d ≤ 40 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 18 ∧ 6 ≤ d ∧ d < 31}.card = 25 :=
by
  sorry

end joy_choices_count_l421_421678


namespace hall_area_relation_l421_421152

theorem hall_area_relation :
  ∀ (L W V : ℝ), 
  L = 6 ∧ W = 6 ∧ V = 108 → 
  let floor_area : ℝ := L * W,
      h : ℝ := V / floor_area,
      wall_area : ℝ := 4 * L * h
  in wall_area = 2 * floor_area :=
by
  intros L W V h_cond
  cases h_cond with hL h_rest
  cases h_rest with hW hV
  let floor_area := L * W
  let h := V / floor_area
  let wall_area := 4 * L * h
  rw [hL, hW, hV]
  sorry

end hall_area_relation_l421_421152


namespace no_five_in_sequence_l421_421545

noncomputable def sequence : ℕ → ℕ
| 1       := 2
| (n + 2) := Nat.largestPrime (sequence (n + 1) * sequence n + 1)

theorem no_five_in_sequence : ∀ n : ℕ, sequence n ≠ 5 :=
sorry

end no_five_in_sequence_l421_421545


namespace min_n_divides_P_2010_l421_421738

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def P (n : ℕ) : ℕ :=
  ∏ c in (Finset.range (n+1)).filter (λ c, Nat.squarefree c), 
    Nat.factorial (Nat.sqrt (n / c))

theorem min_n_divides_P_2010 : ∃ n : ℕ, P(n) % 2010 = 0 ∧ ∀ m : ℕ, m < n → P(m) % 2010 ≠ 0 :=
  ∃ n, n = 4489 ∧ P(4489) % 2010 = 0 ∧ ∀ m : ℕ, m < 4489 → P(m) % 2010 ≠ 0 :=
by
  sorry

end min_n_divides_P_2010_l421_421738


namespace midpoint_translation_correct_l421_421781

def midpoint (p1 p2 : (ℤ × ℤ)) : (ℤ × ℤ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def translate (p : (ℤ × ℤ)) (v : (ℤ × ℤ)) : (ℤ × ℤ) :=
  (p.1 + v.1, p.2 + v.2)

theorem midpoint_translation_correct :
  let s1 : (ℤ × ℤ) × (ℤ × ℤ) := ((3, -2), (-7, 4))
  let translation_vector : (ℤ × ℤ) := (3, -4)
  let s2_midpoint := translate (midpoint s1.1 s1.2) translation_vector
  s2_midpoint = (1, -3) :=
by
  sorry

end midpoint_translation_correct_l421_421781


namespace possible_measures_A_l421_421826

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l421_421826


namespace mean_age_l421_421040

theorem mean_age (ages : List ℕ) (h : ages = [8, 8, 8, 8, 11, 11, 16]) : (ages.sum : ℚ) / ages.length = 10 := by
  rw [h]
  norm_num
  rfl

end mean_age_l421_421040


namespace common_point_of_lines_l421_421524

theorem common_point_of_lines (a b c d x y : ℝ) (h1 : b = a - d) (h2 : c = a + d) :
    ∀ (x y : ℝ), (∃ d : ℝ, ax + (a - d)y = a + d) ↔ (x = 1 ∧ y = -1) := 
by
  intro x y
  constructor
  { -- prove (x = 1 ∧ y = -1) given (ax + (a - d)y = a + d)
    intros h
    -- here, we would input the steps to prove x = 1 and y = -1
    sorry
  }
  { -- prove reverse implication (using ∃ d)
    intros h
    existsi d
    -- here, we would input the steps to prove the line equation holds for some d
    sorry
  }

end common_point_of_lines_l421_421524


namespace danny_threw_away_l421_421543

-- Problem conditions as mathematical definitions
def num_new_caps : ℕ := 50
def total_caps : ℕ := 60
def more_caps_found : ℕ := 44

-- To prove: The number of bottle caps Danny threw away is 6
theorem danny_threw_away : ∃ x : ℕ, x + more_caps_found = num_new_caps ∧ 60 = total_caps :=
begin
  use 6,
  split,
  { exact nat.add_left_cancel_iff.2 rfl, },
  { refl, }
end

end danny_threw_away_l421_421543


namespace length_of_fourth_side_in_cyclic_quadrilateral_l421_421509

theorem length_of_fourth_side_in_cyclic_quadrilateral :
  ∀ (r a b c : ℝ), r = 300 ∧ a = 300 ∧ b = 300 ∧ c = 150 * Real.sqrt 2 →
  ∃ d : ℝ, d = 450 :=
by
  sorry

end length_of_fourth_side_in_cyclic_quadrilateral_l421_421509


namespace smallest_four_digit_palindrome_divisible_by_5_l421_421468

theorem smallest_four_digit_palindrome_divisible_by_5 :
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 5 = 0) ∧ (n = 5005 ∨ n = 5335 ∨ n = 5445 ∨ n = 5555 ∨ n = 5665 ∨ n = 5775 ∨ n = 5885 ∨ n = 5995) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m % 5 = 0) ∧ (m = 5005 ∨ m = 5335 ∨ m = 5445 ∨ m = 5555 ∨ m = 5665 ∨ m = 5775 ∨ m = 5885 ∨ m = 5995) → 5005 ≤ m :=
begin
  sorry
end

end smallest_four_digit_palindrome_divisible_by_5_l421_421468


namespace ratio_of_distances_l421_421270

open Real

noncomputable def ellipse : Set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in (x^2 / 1 + y^2 / 4 = 1)}

def foci_1 : ℝ × ℝ := ( -2 * sqrt 3, 0)
def foci_2 : ℝ × ℝ := ( 2 * sqrt 3, 0)

def line (p : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := p
  in (x - sqrt 3 * y + 8 + 2 * sqrt 3 = 0)

theorem ratio_of_distances (P : ℝ × ℝ)
  (hP : P ∈ {p | let ⟨x, y⟩ := p in x - sqrt 3 * y + 8 + 2 * sqrt 3 = 0})
  (hP_el : P ∈ ellipse) :
  let F1 := foci_1 in let F2 := foci_2 in
  (dist P F1 / dist P F2) = sqrt 3 - 1 :=
sorry

end ratio_of_distances_l421_421270


namespace equilateral_triangle_area_l421_421403

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l421_421403


namespace chess_tournament_perfect_square_l421_421314

theorem chess_tournament_perfect_square 
  (L F : ℕ)
  (total_participants : ℕ := L + F)
  (points_distribution_symmetry : ∀ p (hp : p ∈ finset.range (L + F)), 
    ∑ q in finset.range L, (1 if p < q else 0.5 if p = q else 0) = 
    ∑ r in finset.range F, (1 if p < L + r else 0.5 if p = L + r else 0))
  (games_played_once : ∀ (p q : ℕ), p ≠ q → p < total_participants → q < total_participants → 
    ∃ result, result = if p < q then 0.5 else 1): 
  ∃ k : ℕ, total_participants = k^2 :=
sorry

end chess_tournament_perfect_square_l421_421314


namespace incorrect_deduction_l421_421256

-- Definitions of lines and planes
variable {a b : Line}
variable {α β γ : Plane}

-- Problem statement
theorem incorrect_deduction :
  (a ⊆ α) ∧ (b ⊆ α) ∧ (¬ Parallel a β) ∧ (¬ Parallel b β) → ¬ Parallel α β :=
sorry

end incorrect_deduction_l421_421256


namespace unique_solution_of_quadratics_l421_421742

theorem unique_solution_of_quadratics (y : ℚ) 
    (h1 : 9 * y^2 + 8 * y - 3 = 0) 
    (h2 : 27 * y^2 + 35 * y - 12 = 0) : 
    y = 1 / 3 :=
sorry

end unique_solution_of_quadratics_l421_421742


namespace Andrena_more_than_Debelyn_l421_421995

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l421_421995


namespace fixed_monthly_fee_l421_421000

theorem fixed_monthly_fee (f h : ℝ) 
  (feb_bill : f + h = 18.72)
  (mar_bill : f + 3 * h = 33.78) :
  f = 11.19 :=
by
  sorry

end fixed_monthly_fee_l421_421000


namespace evaluate_expression_l421_421982

theorem evaluate_expression :
  (1 / 3)⁻¹ - abs (sqrt 3 - 3) = sqrt 3 := 
sorry

end evaluate_expression_l421_421982


namespace strange_numbers_are_correct_l421_421752

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_strange : ℕ → Prop
| n := if n < 10 then is_prime n
        else is_prime n ∧ (is_strange (n / 10) ∨ is_strange (n % 10))

def strange_numbers : set ℕ := { n | is_strange n }

theorem strange_numbers_are_correct :
  strange_numbers = {2, 3, 5, 7, 23, 37, 53, 73, 373} :=
by {
  sorry
}

end strange_numbers_are_correct_l421_421752


namespace weight_of_new_person_l421_421924

theorem weight_of_new_person (W : ℝ) (N : ℝ) (h1 : (W + (8 * 2.5)) = (W - 20 + N)) : N = 40 :=
by
  sorry

end weight_of_new_person_l421_421924


namespace derivative_zero_at_x0_l421_421809

noncomputable def func (x a : ℝ) : ℝ := (x^2 + a^2) / x

theorem derivative_zero_at_x0 (a x0 : ℝ) (h_a_pos : a > 0) :
  (deriv (λ x : ℝ, func x a) x0 = 0) ↔ (x0 = a ∨ x0 = -a) :=
begin
  sorry
end

end derivative_zero_at_x0_l421_421809


namespace range_of_a_l421_421748

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (2^x) = x^2 - 2 * a * x + a^2 - 1) →
  (∀ x, 2^(a-1) ≤ x ∧ x ≤ 2^(a^2 - 2*a + 2) → -1 ≤ f x ∧ f x ≤ 0) →
  ((3 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a ∧ a ≤ (3 + Real.sqrt 5) / 2) :=
by
  sorry

end range_of_a_l421_421748


namespace sandy_age_l421_421480

-- Given conditions
variables (S M : ℕ)
hypothesis h1 : M = S + 14
hypothesis h2 : (S : ℚ) / M = 7 / 9

-- Prove that Sandy's age is 49 years
theorem sandy_age : S = 49 :=
by sorry

end sandy_age_l421_421480


namespace equilateral_triangle_area_l421_421408

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l421_421408


namespace angle_bisector_theorem_l421_421234

noncomputable def find_point_on_line (A B : Point) (l : Line) : Point :=
  -- Assume necessary geometric functions and constructions are available.

theorem angle_bisector_theorem (l : Line) (A B : Point) (hA : !collinear l A) (hB : !collinear l B) :
  ∃ M : Point, lies_on M l ∧ angle_bisector (ray M A) (angle (ray M B) (ray_from_line M l)) :=
sorry

end angle_bisector_theorem_l421_421234


namespace monotonic_intervals_range_of_a_two_zeros_product_of_zeros_gt_e_squared_l421_421274

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / x - a

theorem monotonic_intervals (a : ℝ) : 
  (∀ x, f' x a > 0 → (0 < x ∧ x < Real.exp 1)) ∧ 
  (∀ x, f' x a < 0 → (Real.exp 1 < x)) := 
sorry

theorem range_of_a_two_zeros (a : ℝ) :
  (∃ m n : ℝ, m ≠ n ∧ f m a = 0 ∧ f n a = 0) → (0 < a ∧ a < 1 / Real.exp 1) := 
sorry

theorem product_of_zeros_gt_e_squared (a m n : ℝ) :
  f m a = 0 ∧ f n a = 0 → m * n > Real.exp 2 :=
sorry

end monotonic_intervals_range_of_a_two_zeros_product_of_zeros_gt_e_squared_l421_421274


namespace number_of_solutions_l421_421250

-- Defining the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

-- Statement of the problem
theorem number_of_solutions (a : ℝ) : {a : ℝ | A a ∪ B a = A a}.card = 1 :=
by
  sorry

end number_of_solutions_l421_421250


namespace prime_p_satisfies_conditions_l421_421193

theorem prime_p_satisfies_conditions (p : ℕ) (hp : Nat.Prime p) (h1 : Nat.Prime (4 * p^2 + 1)) (h2 : Nat.Prime (6 * p^2 + 1)) : p = 5 :=
sorry

end prime_p_satisfies_conditions_l421_421193


namespace min_u_value_l421_421299

theorem min_u_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x + 1 / x) * (y + 1 / (4 * y)) ≥ 25 / 8 :=
by
  sorry

end min_u_value_l421_421299


namespace count_quadruples_l421_421290

open Real

theorem count_quadruples:
  ∃ qs : Finset (ℝ × ℝ × ℝ × ℝ),
  (∀ (a b c k : ℝ), (a, b, c, k) ∈ qs ↔ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a^k = b * c ∧
    b^k = c * a ∧
    c^k = a * b
  ) ∧
  qs.card = 8 :=
sorry

end count_quadruples_l421_421290


namespace meadow_area_l421_421163

theorem meadow_area (x : ℝ) (h1 : ∀ y : ℝ, y = x / 2 + 3) (h2 : ∀ z : ℝ, z = 1 / 3 * (x / 2 - 3) + 6) :
  (x / 2 + 3) + (1 / 3 * (x / 2 - 3) + 6) = x → x = 24 := by
  sorry

end meadow_area_l421_421163


namespace infinite_pairs_of_sides_l421_421877

theorem infinite_pairs_of_sides (x : ℝ) (n_1 n_2 : ℕ) (h1 : P_1_interior_angle = 180 - 360 / n_1)
    (h2 : P_2_interior_angle = 180 - 360 / n_2) (h3 : x + 3 * x < 180) (h4 : P_1_interior_angle = x)
    (h5 : P_2_interior_angle = 3 * x) :
    { (n_1, n_2) : ℕ × ℕ | n_1 ≥ 3 ∧ n_2 ≥ 9 }.infinite := by
  sorry

end infinite_pairs_of_sides_l421_421877


namespace solve_for_x_l421_421032

theorem solve_for_x :
  ∀ x : ℕ, 100^4 = 5^x → x = 8 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l421_421032


namespace min_days_to_sun_l421_421374

def active_days_for_level (N : ℕ) : ℕ :=
  N * (N + 4)

def days_needed_for_upgrade (current_days future_days : ℕ) : ℕ :=
  future_days - current_days

theorem min_days_to_sun (current_level future_level : ℕ) :
  current_level = 9 →
  future_level = 16 →
  days_needed_for_upgrade (active_days_for_level current_level) (active_days_for_level future_level) = 203 :=
by
  intros h1 h2
  rw [h1, h2, active_days_for_level, active_days_for_level]
  sorry

end min_days_to_sun_l421_421374


namespace lifespan_difference_l421_421059

variable (H : ℕ)

theorem lifespan_difference (H : ℕ) (bat_lifespan : ℕ) (frog_lifespan : ℕ) (total_lifespan : ℕ) 
    (hb : bat_lifespan = 10)
    (hf : frog_lifespan = 4 * H)
    (ht : H + bat_lifespan + frog_lifespan = total_lifespan)
    (t30 : total_lifespan = 30) :
    bat_lifespan - H = 6 :=
by
  -- here would be the proof
  sorry

end lifespan_difference_l421_421059


namespace f_sum_value_l421_421796

-- Definition of the Riemann function R(x)
def R (x : ℝ) : ℝ :=
  if ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ x = q / p ∧ nat.coprime q p ∧ q < p then
    let ⟨p, q, hp, hq, hx, hc, hlt⟩ := classical.some_spec (classical.some_spec (classical.some x).some_spec) in 1 / p
  else if x = 0 ∨ x = 1 then 0
  else 0

-- Given conditions
def f (x : ℝ) : ℝ := sorry

-- Hypotheses about f(x)
axiom odd_f {x : ℝ} : f (-x) = -f x
axiom periodic_f {x : ℝ} : f (1 + x) = -f (1 - x)
axiom initial_f {x : ℝ} (h : x ∈ set.Icc 0 1) : f x = R x

-- The goal
theorem f_sum_value :
  f 2023 + f (2023 / 2) + f (-2023 / 3) = -5 / 6 :=
sorry

end f_sum_value_l421_421796


namespace range_of_a_l421_421442

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y = log 0.5 (x^2 + a * x + 1))
  ↔ a ∈ set.Iic (-2) ∪ set.Ici 2 :=
begin
  sorry
end

end range_of_a_l421_421442


namespace distance_from_P_to_XYZ_l421_421765

theorem distance_from_P_to_XYZ 
  (P X Y Z : ℝ^3)
  (h_XY_perp : is_perpendicular (X - P) (Y - P))
  (h_YZ_perp : is_perpendicular (Y - P) (Z - P))
  (h_ZX_perp : is_perpendicular (Z - P) (X - P))
  (PX : ℝ) (PY : ℝ) (PZ : ℝ)
  (h_PX : dist P X = 15)
  (h_PY : dist P Y = 10)
  (h_PZ : dist P Z = 8) :
  distance_from_point_to_face P {X, Y, Z} = 8 :=
sorry

end distance_from_P_to_XYZ_l421_421765


namespace abc_values_l421_421344

open Complex

theorem abc_values (a b c : ℂ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = 1) :
  ({a, b, c} = {1, Complex.I, -Complex.I}) :=
by
  sorry

end abc_values_l421_421344


namespace number_of_girls_l421_421855

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l421_421855


namespace remainder_of_sum_l421_421712

theorem remainder_of_sum (p q : ℤ) (c d : ℤ) 
  (hc : c = 100 * p + 78)
  (hd : d = 150 * q + 123) :
  (c + d) % 50 = 1 :=
sorry

end remainder_of_sum_l421_421712


namespace smallest_g_l421_421214

noncomputable def r_n (n : ℕ) : ℝ := 
  Real.Inf ((λ ⟨c, d⟩, abs (c - d * Real.sqrt 3)) '' { x : ℕ × ℕ | x.1 + x.2 = n })

theorem smallest_g :
  (∀ n : ℕ, r_n n ≤ (1 + Real.sqrt 3) / 2) ∧ 
  (∀ ε > 0, ∃ n : ℕ, r_n n > (1 + Real.sqrt 3) / 2 - ε) :=
sorry

end smallest_g_l421_421214


namespace find_length_of_longer_parallel_side_of_trapezoid_l421_421161

-- Conditions
def length_of_sides_of_square : ℝ := 2
def area_of_square : ℝ := length_of_sides_of_square ^ 2
def area_of_each_region : ℝ := area_of_square / 3
def length_of_segments : ℝ := 1
def height_of_trapezoid : ℝ := length_of_segments
def sum_of_parallel_sides (x : ℝ) : ℝ := x + 1

-- Goal to prove
theorem find_length_of_longer_parallel_side_of_trapezoid (x : ℝ) :
  area_of_each_region = 0.5 * sum_of_parallel_sides x * height_of_trapezoid → 
  x = 5 / 3 :=
by
  sorry

end find_length_of_longer_parallel_side_of_trapezoid_l421_421161


namespace stream_speed_is_2_l421_421919

variable (v : ℝ) -- Let v be the speed of the stream in km/h

-- Condition 1: Man's swimming speed in still water
def swimming_speed_still : ℝ := 6

-- Condition 2: It takes him twice as long to swim upstream as downstream
def condition : Prop := (swimming_speed_still + v) / (swimming_speed_still - v) = 2

theorem stream_speed_is_2 : condition v → v = 2 := by
  intro h
  -- Proof goes here
  sorry

end stream_speed_is_2_l421_421919


namespace lassis_from_mangoes_l421_421177

-- Define the given ratio
def lassis_per_mango := 15 / 3

-- Define the number of mangoes
def mangoes := 15

-- Define the expected number of lassis
def expected_lassis := 75

-- Prove that with 15 mangoes, 75 lassis can be made given the ratio
theorem lassis_from_mangoes (h : lassis_per_mango = 5) : mangoes * lassis_per_mango = expected_lassis :=
by
  sorry

end lassis_from_mangoes_l421_421177


namespace complex_distance_to_origin_l421_421491

open Complex

theorem complex_distance_to_origin :
  let z := (2 * Complex.I) / (1 - Complex.I) in
  Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_distance_to_origin_l421_421491


namespace number_of_girls_l421_421863

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l421_421863


namespace train_speed_proof_l421_421927

theorem train_speed_proof :
  (∀ (speed : ℝ), 
    let train_length := 120
    let cross_time := 16
    let total_distance := 240
    let relative_speed := total_distance / cross_time
    let individual_speed := relative_speed / 2
    let speed_kmh := individual_speed * 3.6
    (speed_kmh = 27) → speed = 27
  ) :=
by
  sorry

end train_speed_proof_l421_421927


namespace number_of_solid_circles_among_first_2019_l421_421519

theorem number_of_solid_circles_among_first_2019 (n : ℕ) :
  (∑ k in finset.range 64, k + 2) > 2019 ∧ (∑ k in finset.range 63, k + 2) ≤ 2019 →
  (n = 2019 → (n - 2015 + 1 = 62)) := 
by
  sorry

end number_of_solid_circles_among_first_2019_l421_421519


namespace integer_solutions_5_lt_sqrt_x_lt_6_l421_421195

theorem integer_solutions_5_lt_sqrt_x_lt_6 :
  {x : ℤ | 5 < real.sqrt x ∧ real.sqrt x < 6}.to_finset.card = 10 :=
by
  sorry

end integer_solutions_5_lt_sqrt_x_lt_6_l421_421195


namespace coeffs_divisible_by_5_l421_421784

theorem coeffs_divisible_by_5
  (a b c d : ℤ)
  (h1 : a + b + c + d ≡ 0 [ZMOD 5])
  (h2 : -a + b - c + d ≡ 0 [ZMOD 5])
  (h3 : 8 * a + 4 * b + 2 * c + d ≡ 0 [ZMOD 5])
  (h4 : d ≡ 0 [ZMOD 5]) :
  a ≡ 0 [ZMOD 5] ∧ b ≡ 0 [ZMOD 5] ∧ c ≡ 0 [ZMOD 5] ∧ d ≡ 0 [ZMOD 5] :=
sorry

end coeffs_divisible_by_5_l421_421784


namespace equidistant_Q_R_midpoint_MN_l421_421766

variables (A B C D M N Q R : Type*)
variables [right_trapezoid ABCD, midpoint M AC, midpoint N BD]
variables [circumcircle ABN Q, circumcircle CDM R]
variables (X Y : Type*)
variables [projection N X BC, projection M Y BC]

theorem equidistant_Q_R_midpoint_MN :
  distance Q (midpoint MN) = distance R (midpoint MN) :=
sorry

end equidistant_Q_R_midpoint_MN_l421_421766


namespace infinite_solutions_a_value_l421_421188

theorem infinite_solutions_a_value (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) ↔ a = 5 := 
by 
  sorry

end infinite_solutions_a_value_l421_421188


namespace pyramid_in_cube_volume_l421_421968

theorem pyramid_in_cube_volume (height base_side_length : ℕ) (h_height : height = 15) (h_base : base_side_length = 14) :
  let cube_side := max height base_side_length in
  cube_side ^ 3 = 3375 := by
  sorry

end pyramid_in_cube_volume_l421_421968


namespace math_problem_l421_421722

theorem math_problem (x a b p q : ℝ) (hp : p = 2) (hq : q = 9) 
  (h1 : a = real.cbrt x) (h2 : b = real.cbrt (24 - x)) (h3 : a + b = 0) 
  (h4 : a^3 + b^3 = 24) : p + q = 13 :=
by {
  -- Set up definitions and equations based on the math problem
  sorry
}

end math_problem_l421_421722


namespace vera_cannot_get_novo_sibirsk_from_good_number_l421_421505

def good_number (n : list ℕ) : Prop :=
  ∀ i ∈ list.range (n.length - 1), abs (n.nth_le i sorry - n.nth_le (i + 1) sorry) ≥ 5

-- Translate letters in "NOVOSIBIRSK" to digits
def novo_sibirsk : list ℕ := [5, 0, 5, 8, 0, 8, 9, 1, 8, 9, 1, 9]

theorem vera_cannot_get_novo_sibirsk_from_good_number :
  ¬ good_number novo_sibirsk :=
by
  sorry

end vera_cannot_get_novo_sibirsk_from_good_number_l421_421505


namespace max_LShapes_placement_l421_421311

def Grid := ℕ × ℕ

def LShape := List (ℕ × ℕ)

noncomputable def fitsInGrid (shape : LShape) (grid : Grid) : Prop := 
  ∀ (x y : ℕ), (x, y) ∈ shape → x < grid.fst ∧ y < grid.snd

def noOverlap (shapes : List LShape) : Prop := 
  ∀ s1 s2 ∈ shapes, s1 ≠ s2 → ∃ p ∈ s1, ∃ q ∈ s2, p ≠ q

def isLShape (shape : LShape) : Prop := 
  -- Assuming a definition that verifies shape conforms to 'L' 
  sorry 

def maxLShapes (grid : Grid) : ℕ := 
  6 -- Based on the correct answer identified

theorem max_LShapes_placement : 
  ∀ (shapes : List LShape), 
    (∀ s ∈ shapes, isLShape s ∧ fitsInGrid s (5, 5)) → 
    noOverlap shapes → 
    shapes.length ≤ maxLShapes (5, 5) := 
begin
  sorry
end

end max_LShapes_placement_l421_421311


namespace simplify_fraction_l421_421179

/-- Definition of the problem statement to be proved in Lean 4 -/
theorem simplify_fraction (N : ℤ) : (factorial (N + 1) / (factorial (N - 1) * (N + 2))) = (N * (N + 1) / (N + 2)) :=
by
  sorry

end simplify_fraction_l421_421179


namespace distinct_positive_factors_13200_l421_421638

theorem distinct_positive_factors_13200 : 
  let factors_count (n: ℕ) : ℕ := 
    let prime_factors := [(2, 4), (3, 1), (5, 2), (11, 1)]
    prime_factors.foldr (λ factor acc => acc * (factor.snd + 1)) 1
  in factors_count 13200 = 60 := 
by
  sorry

end distinct_positive_factors_13200_l421_421638


namespace stubborn_robot_returns_to_start_l421_421485

inductive Direction
| East | North | West | South

inductive Command
| STEP | LEFT

structure Robot :=
  (position : ℤ × ℤ)
  (direction : Direction)

def turnLeft : Direction → Direction
| Direction.East  => Direction.North
| Direction.North => Direction.West
| Direction.West  => Direction.South
| Direction.South => Direction.East

def moveStep : Robot → Robot
| ⟨(x, y), Direction.East⟩  => ⟨(x + 1, y), Direction.East⟩
| ⟨(x, y), Direction.North⟩ => ⟨(x, y + 1), Direction.North⟩
| ⟨(x, y), Direction.West⟩  => ⟨(x - 1, y), Direction.West⟩
| ⟨(x, y), Direction.South⟩ => ⟨(x, y - 1), Direction.South⟩

def executeCommand : Command → Robot → Robot
| Command.STEP, robot => moveStep robot
| Command.LEFT, robot => ⟨robot.position, turnLeft robot.direction⟩

def invertCommand : Command → Command
| Command.STEP => Command.LEFT
| Command.LEFT => Command.STEP

def executeSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand cmd r) robot

def executeInvertedSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand (invertCommand cmd) r) robot

def initialRobot : Robot := ⟨(0, 0), Direction.East⟩

def exampleProgram : List Command :=
  [Command.LEFT, Command.LEFT, Command.LEFT, Command.LEFT, Command.STEP, Command.STEP,
   Command.LEFT, Command.LEFT]

theorem stubborn_robot_returns_to_start :
  let robot := executeSequence exampleProgram initialRobot
  executeInvertedSequence exampleProgram robot = initialRobot :=
by
  sorry

end stubborn_robot_returns_to_start_l421_421485


namespace solve_absolute_value_eq_l421_421911

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421911


namespace equilateral_triangle_area_l421_421411

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l421_421411


namespace oleg_older_than_wife_ekaterina_l421_421026

-- Define the four individuals
constants (Roman Oleg Ekaterina Zhanna : Type)

-- Define age as a relationship
constant age : Roman → Oleg → Ekaterina → Zhanna → Prop

-- Each person has a different age
axiom different_ages : ∀ r o e z : Prop, r ≠ o ∧ r ≠ e ∧ r ≠ z ∧ o ≠ e ∧ o ≠ z ∧ e ≠ z

-- Each married man is older than his wife
axiom roman_older_than_wife : ∀ r z : Prop, r > z → r = Ekaterina
axiom oleg_older_than_wife : ∀ o e : Prop, o > e → o = Zhanna

-- Zhanna is older than Oleg
axiom zhanna_older_than_oleg : ∀ z o : Prop, z > o

-- We need to prove that: Oleg is older than his wife Ekaterina
theorem oleg_older_than_wife_ekaterina : age o e → o > e :=
sorry

end oleg_older_than_wife_ekaterina_l421_421026


namespace time_to_cross_bridge_l421_421164

def train_length : ℕ := 600  -- train length in meters
def bridge_length : ℕ := 100  -- overbridge length in meters
def speed_km_per_hr : ℕ := 36  -- speed of the train in kilometers per hour

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_km_per_hr * 1000 / 3600

-- Compute the total distance
def total_distance : ℕ := train_length + bridge_length

-- Prove the time to cross the overbridge
theorem time_to_cross_bridge : total_distance / speed_m_per_s = 70 := by
  sorry

end time_to_cross_bridge_l421_421164


namespace polynomial_roots_l421_421205

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end polynomial_roots_l421_421205


namespace smallest_palindrome_divisible_by_5_l421_421466

theorem smallest_palindrome_divisible_by_5 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ m, 1000 ≤ m ∧ m < 10000 ∧ is_palindrome m ∧ 5 ∣ m → n ≤ m) ∧ is_palindrome n ∧ 5 ∣ n ∧ n = 5005 :=
sorry

def is_palindrome (n : ℕ) : Prop :=
  let n_str := n.to_string in
  n_str = n_str.reverse

end smallest_palindrome_divisible_by_5_l421_421466


namespace problem_solution_l421_421192

theorem problem_solution:
  ∃(pairs : List (ℕ × ℕ)),
    pairs = [(8, 513), (513, 8), (215, 2838), (2838, 215), (258, 1505), (1505, 258), (235, 2961), (2961, 235)] ∧
    ∀ (α β : ℕ), (α, β) ∈ pairs →
      let δ := Nat.gcd α β in
      let Δ := Nat.lcm α β in
      δ + Δ = 4 * (α + β) + 2021 := 
by
  sorry

end problem_solution_l421_421192


namespace Jolyn_older_than_Clarisse_l421_421575

namespace AgeComparison

-- Define each person's age in terms of months, relative to an arbitrary baseline
variables {Jolyn Therese Aivo Leon Clarisse : ℕ}

-- Conditions provided in the problem
def condition1 : Prop := Jolyn = Therese + 2
def condition2 : Prop := Therese = Aivo + 5
def condition3 : Prop := Leon = Aivo + 2
def condition4 : Prop := Clarisse = Leon + 3

-- The statement to prove: Prove that Jolyn is 2 months older than Clarisse given the conditions
theorem Jolyn_older_than_Clarisse :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → Jolyn = Clarisse + 2 :=
by
  sorry

end AgeComparison

end Jolyn_older_than_Clarisse_l421_421575


namespace find_number_of_non_officers_l421_421803

theorem find_number_of_non_officers
  (avg_salary_all : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℕ) :
  avg_salary_all = 120 ∧
  avg_salary_officers = 450 ∧
  avg_salary_non_officers = 110 ∧
  num_officers = 15 →
  ∃ N : ℕ, (120 * (15 + N) = 450 * 15 + 110 * N) ∧ N = 495 :=
by
  sorry

end find_number_of_non_officers_l421_421803


namespace final_slices_leftover_l421_421791

def total_slices (pizzas slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def stephen_ate (total_slices : ℕ) (percentage : ℝ) : ℕ := (total_slices : ℝ) * percentage |> Int.toNat
def remaining_slices_after_stephen (total_slices stephen_ate_slices : ℕ) : ℕ := total_slices - stephen_ate_slices
def pete_ate (remaining_slices : ℕ) (percentage : ℝ) : ℕ := (remaining_slices : ℝ) * percentage |> Int.toNat
def remaining_slices_after_pete (remaining_slices pete_ate_slices : ℕ) : ℕ := remaining_slices - pete_ate_slices

theorem final_slices_leftover: 
  ∀ (pizzas slices_per_pizza : ℕ) 
    (stephen_percentage pete_percentage : ℝ), 
    pizzas = 2 → slices_per_pizza = 12 → stephen_percentage = 0.25 → pete_percentage = 0.50 → 
    remaining_slices_after_pete 
      (remaining_slices_after_stephen 
        (total_slices pizzas slices_per_pizza) 
        (stephen_ate (total_slices pizzas slices_per_pizza) stephen_percentage))
      (pete_ate 
        (remaining_slices_after_stephen 
          (total_slices pizzas slices_per_pizza) 
          (stephen_ate (total_slices pizzas slices_per_pizza) stephen_percentage)) 
        pete_percentage) 
    = 9 := 
by 
  intros pizzas slices_per_pizza stephen_percentage pete_percentage  
  intros h_pizzas h_slices_per_pizza h_stephen_percentage h_pete_percentage 
  simp [total_slices, stephen_ate, remaining_slices_after_stephen, pete_ate, remaining_slices_after_pete]
  sorry

end final_slices_leftover_l421_421791


namespace max_draw_triples_l421_421762

def choose (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem max_draw_triples (n : ℕ) (h : n ≥ 3) :
  (n % 2 = 1 → ∃ k, k = (choose n 3 - ((n^3 - 4*n^2 + 3*n) / 8))) ∧
  (n % 2 = 0 → ∃ k, k = (choose n 3 - ((n^3 - 4*n^2 + 4*n) / 8))) :=
  sorry

end max_draw_triples_l421_421762


namespace ladder_slides_out_l421_421140

theorem ladder_slides_out (ladder_length foot_initial_dist ladder_slip_down foot_final_dist : ℝ) 
  (h_ladder_length : ladder_length = 25)
  (h_foot_initial_dist : foot_initial_dist = 7)
  (h_ladder_slip_down : ladder_slip_down = 4)
  (h_foot_final_dist : foot_final_dist = 15) :
  foot_final_dist - foot_initial_dist = 8 :=
  by
  simp [h_ladder_length, h_foot_initial_dist, h_ladder_slip_down, h_foot_final_dist]
  sorry

end ladder_slides_out_l421_421140


namespace length_of_train_l421_421964

-- Definitions based on the conditions
def train_speed_km_per_hr : ℝ := 63
def man_speed_km_per_hr : ℝ := 3
def crossing_time_seconds : ℝ := 29.997600191984642

-- Convert speeds from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * 1000 / 3600

def train_speed_m_per_s : ℝ := km_per_hr_to_m_per_s train_speed_km_per_hr
def man_speed_m_per_s : ℝ := km_per_hr_to_m_per_s man_speed_km_per_hr

-- Calculate relative speed
def relative_speed_m_per_s : ℝ := train_speed_m_per_s - man_speed_m_per_s

-- Prove the length of the train
theorem length_of_train : relative_speed_m_per_s * crossing_time_seconds = 500 := by
  simp [relative_speed_m_per_s, train_speed_m_per_s, man_speed_m_per_s, crossing_time_seconds, km_per_hr_to_m_per_s]
  sorry

end length_of_train_l421_421964


namespace add_pure_alcohol_and_glycerin_l421_421937

structure Mixture :=
  (initial_volume : ℝ)
  (initial_alcohol_concentration : ℝ)
  (initial_glycerin_concentration : ℝ)
  (desired_alcohol_concentration : ℝ)
  (desired_glycerin_concentration : ℝ)

def initial_mixture := Mixture.mk 12 0.30 0.10 0.45 0.15

theorem add_pure_alcohol_and_glycerin 
  (A : ℝ) (G : ℝ)
  (hA : A ≈ 4.49) (hG : G ≈ 1.496) :
  (3.6 + A) / (12 + A + G) = 0.45 ∧ (1.2 + G) / (12 + A + G) = 0.15 :=
sorry

end add_pure_alcohol_and_glycerin_l421_421937


namespace dot_product_calculation_l421_421635

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (u v w : V)
(h1 : inner u v = 5)
(h2 : inner u w = -2)
(h3 : inner v w = -7)

-- Mathematically equivalent proof problem
theorem dot_product_calculation : inner v (3 • w - 4 • u) = -41 :=
by {
  sorry,
}

end dot_product_calculation_l421_421635


namespace func_eq_id_l421_421018

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f.

theorem func_eq_id (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) ≤ x)
  (h2 : ∀ x y, f(x + y) ≤ f(x) + f(y)) :
  ∀ x, f(x) = x :=
begin
  sorry -- Proof goes here
end

end func_eq_id_l421_421018


namespace range_of_a_l421_421071

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f(x) ≥ a^2 - 4*a) ↔ (1 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l421_421071


namespace tom_strokes_over_par_l421_421089

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l421_421089


namespace real_solution_of_equation_l421_421568

theorem real_solution_of_equation :
  ∀ x : ℝ, (x ≠ 5) → (x ≠ 3) →
  ((x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3)) 
  / ((x - 5) * (x - 3) * (x - 5)) = 1 ↔ x = 1 :=
by sorry

end real_solution_of_equation_l421_421568


namespace number_of_girls_l421_421850

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l421_421850


namespace hare_weights_problem_l421_421847

theorem hare_weights_problem :
  ∀ (weights : Finset ℕ) (m n : ℕ),
    weights = Finset.range (2018 + 1) ∧ m ∈ weights ∧ n ∈ weights ∧ m < n →
    n = m + 1 ∧ (m = 1 ∧ n = 2 ∨ m = 1 ∧ n = 3 ∨ m = 2016 ∧ n = 2018 ∨ m = 2017 ∧ n = 2018) ∨
    (∀ (a b : ℕ), a ∈ weights \ {m, n} ∧ b ∈ weights \ {m, n} ∧ a < b →
      a + b ≠ m + n) :=
begin
  sorry
end

end hare_weights_problem_l421_421847


namespace xiao_qian_has_been_to_great_wall_l421_421917

-- Define the four students
inductive Student
| XiaoZhao
| XiaoQian
| XiaoSun
| XiaoLi

open Student

-- Define the relations for their statements
def has_been (s : Student) : Prop :=
  match s with
  | XiaoZhao => false
  | XiaoQian => true
  | XiaoSun => true
  | XiaoLi => false

def said (s : Student) : Prop :=
  match s with
  | XiaoZhao => ¬has_been XiaoZhao
  | XiaoQian => has_been XiaoLi
  | XiaoSun => has_been XiaoQian
  | XiaoLi => ¬has_been XiaoLi

axiom only_one_lying : ∃ l : Student, ∀ s : Student, said s → (s ≠ l)

theorem xiao_qian_has_been_to_great_wall : has_been XiaoQian :=
by {
  sorry -- Proof elided
}

end xiao_qian_has_been_to_great_wall_l421_421917


namespace systematic_sampling_camp_distribution_l421_421670

theorem systematic_sampling_camp_distribution :
  ∀ (n : ℕ) (a1 d : ℕ) (c_1_lim c_2_lim c_3_lim : ℕ) (c_1_total c_2_total c_3_total : ℕ),
  n = 50 →
  a1 = 3 →
  d = 12 →
  c_1_lim = 200 →
  c_2_lim = 500 →
  c_3_lim = 600 →
  c_1_total = 17 →
  c_2_total = 25 →
  c_3_total = 8 →
  c_1_total = Nat.div (c_1_lim - a1 + d) d - 1 + 1 ∧
  c_2_total = Nat.div (c_2_lim - a1 + d) d - Nat.div (c_1_lim - a1 + d) d ∧
  c_3_total = Nat.div (c_3_lim - a1 + d) d - Nat.div (c_2_lim - a1 + d) d
 :=
begin
  intros n a1 d c_1_lim c_2_lim c_3_lim c_1_total c_2_total c_3_total,
  intro h_n, intro h_a1, intro h_d, intro h_c_1_lim, intro h_c_2_lim, intro h_c_3_lim,
  intro h_c_1_total, intro h_c_2_total, intro h_c_3_total,
  sorry
end

end systematic_sampling_camp_distribution_l421_421670


namespace smallest_palindrome_divisible_by_5_l421_421467

theorem smallest_palindrome_divisible_by_5 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ m, 1000 ≤ m ∧ m < 10000 ∧ is_palindrome m ∧ 5 ∣ m → n ≤ m) ∧ is_palindrome n ∧ 5 ∣ n ∧ n = 5005 :=
sorry

def is_palindrome (n : ℕ) : Prop :=
  let n_str := n.to_string in
  n_str = n_str.reverse

end smallest_palindrome_divisible_by_5_l421_421467


namespace prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l421_421157

/-
Prove that if a person forgets the last digit of their 6-digit password, which can be any digit from 0 to 9,
the probability of pressing the correct last digit in no more than 2 attempts is 1/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts :
  let correct_prob := 1 / 10 
  let incorrect_prob := 9 / 10 
  let second_attempt_prob := 1 / 9 
  correct_prob + (incorrect_prob * second_attempt_prob) = 1 / 5 :=
by
  sorry

/-
Prove that if a person forgets the last digit of their 6-digit password, but remembers that the last digit is an even number,
the probability of pressing the correct last digit in no more than 2 attempts is 2/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts_if_even :
  let correct_prob := 1 / 5 
  let incorrect_prob := 4 / 5 
  let second_attempt_prob := 1 / 4 
  correct_prob + (incorrect_prob * second_attempt_prob) = 2 / 5 :=
by
  sorry

end prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l421_421157


namespace store_A_cost_store_B_cost_cost_comparison_store_more_favorable_option_l421_421396

section heaters
variables (x : ℕ)

-- Definitions for costs
def cost_A (x : ℕ) : ℕ := 21000 - 100 * x
def cost_B (x : ℕ) : ℕ := 20200 - 82 * x
def mixed_cost (x : ℕ) : ℕ := 20200 - 92 * x

-- a) Proof of total cost from Store A
theorem store_A_cost (x : ℕ) : cost_A x = 21000 - 100 * x :=
by simp [cost_A]

-- b) Proof of total cost from Store B
theorem store_B_cost (x : ℕ) : cost_B x = 20200 - 82 * x :=
by simp [cost_B]

-- c) Store comparison for x = 60
theorem cost_comparison_store (x : ℕ) (h : x = 60) : cost_A x < cost_B x :=
by { rw h, simp [cost_A, cost_B], linarith }

-- d) Proof of a more favorable option with a total cost
theorem more_favorable_option (x : ℕ) (h : x = 60) : mixed_cost x = 14680 :=
by { rw h, simp [mixed_cost], linarith }
end heaters

end store_A_cost_store_B_cost_cost_comparison_store_more_favorable_option_l421_421396


namespace decimal_150th_digit_of_5_over_11_l421_421110

theorem decimal_150th_digit_of_5_over_11 :
  let repeating_sequence : string := "45" in
  let sequence_length := 2 in
  ∀ n, n = 150 →
  let digit_position := n % sequence_length in
  (if digit_position = 0 then repeating_sequence.get 1 else repeating_sequence.get (digit_position - 1)) = '5' :=
by
  sorry

end decimal_150th_digit_of_5_over_11_l421_421110


namespace basketball_team_selection_l421_421838

theorem basketball_team_selection :
  (∃ (S : Finset (Fin 16)), S.card = 7 ∧ 
    (∃ (Q : Finset (Fin 16)), Q.card = 4 ∧ 
      {0, 1, 2, 3} ⊆ Q ∧ 
      (Q \ {0, 1, 2, 3}).card = 3 ∧
      (∀ x ∈ Q, x ∈ S) ∧ 
      (∃ (R : Finset (Fin 16)), R = S \ Q ∧ R.card = 4 ∧
        (R ⊆ {4, 5, ..., 15}))) → 
    (S.card = 7 ∧ Q.card = 4 ∧ ({0, 1, 2, 3} ⊆ Q ∧ (Q \ {0, 1, 2, 3}).card = 3 ∧ R = S \ Q ∧
    R.card = 4 ∧ R ⊆ {4, 5, ..., 15}))) := 1980 :=
by sorry

end basketball_team_selection_l421_421838


namespace find_n_l421_421604

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -0.7)
  (h2 : log 10 (sin x + cos x) = 0.5 * (log 10 n - 0.7)) :
  n = 3 :=
sorry

end find_n_l421_421604


namespace harmonic_mean_of_1_3_1_div_2_l421_421104

noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  let reciprocals := [1 / a, 1 / b, 1 / c]
  (reciprocals.sum) / reciprocals.length

theorem harmonic_mean_of_1_3_1_div_2 : harmonicMean 1 3 (1 / 2) = 9 / 10 :=
  sorry

end harmonic_mean_of_1_3_1_div_2_l421_421104


namespace increasing_function_in_0_infty_l421_421969

def function_A (x : ℝ) := Real.sin x

noncomputable def function_B (x : ℝ) := x * Real.exp 2

def function_C (x : ℝ) := x^3 - x

noncomputable def function_D (x : ℝ) := Real.log x - x

theorem increasing_function_in_0_infty :
  (∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → function_A x ≤ function_A y) = false ∧
  (∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → function_B x ≤ function_B y) ∧
  (∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → function_C x ≤ function_C y) = false ∧
  (∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → function_D x ≤ function_D y) = false :=
by
  sorry

end increasing_function_in_0_infty_l421_421969


namespace satisfy_eq_pairs_l421_421191

theorem satisfy_eq_pairs (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ (x = 4 ∧ (y = 1 ∨ y = -3) ∨ x = -4 ∧ (y = 1 ∨ y = -3)) :=
by
  sorry

end satisfy_eq_pairs_l421_421191


namespace second_discount_is_five_percent_l421_421837

-- Defining the parameters and conditions
def Rs := ℝ  -- using ℝ to represent currency values for precision
def initial_price : Rs := 200
def first_discount_rate : ℝ := 0.20
def final_price : Rs := 152
def after_first_discount_price : Rs := initial_price * (1 - first_discount_rate)

-- Define the second discount rate
def second_discount_rate (D : ℝ) : Prop :=
  after_first_discount_price * (1 - D) = final_price

-- Prove that the second discount rate is 0.05 (or 5%)
theorem second_discount_is_five_percent : second_discount_rate 0.05 :=
by
  sorry

end second_discount_is_five_percent_l421_421837


namespace tom_strokes_over_par_l421_421087

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l421_421087


namespace shortest_distance_midpoint_y_axis_l421_421238

-- Define the points and their properties
variables {M A B : Point}
variables {x0 y0 d1 d2 : ℝ}

-- Conditions
def midpoint (M : Point) (A B : Point) : Prop := 
  M = Point.mk (x0, y0) /\
  A = Point.mk (x0 + d1, y0 + d2) /\
  B = Point.mk (x0 - d1, y0 - d2)

def on_parabola (P : Point) : Prop := 
  let ⟨x, y⟩ := P in y^2 = x

def length_eq_three (A B : Point) : Prop :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  (x2 - x1)^2 + (y2 - y1)^2 = 3^2

-- Proof statement
theorem shortest_distance_midpoint_y_axis : 
  ∀ (A B M : Point),
  midpoint M A B →
  on_parabola A →
  on_parabola B →
  length_eq_three A B →
  let ⟨x, y⟩ := M in x = 5 / 4 :=
sorry

end shortest_distance_midpoint_y_axis_l421_421238


namespace distances_proportional_to_sides_l421_421510

variables {A B C D M : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables {AB BC CD DA : ℝ}
variables {m_a m_b m_c m_d : ℝ}
variables (cyclic : inscribed_in_circle A B C D)
variables (equal_products : AB * CD = BC * DA)
variables (dist_M_AB dist_M_BC dist_M_CD dist_M_DA: ℝ)

theorem distances_proportional_to_sides :
    dist_M_AB / AB = dist_M_BC / BC ∧
    dist_M_BC / BC = dist_M_CD / CD ∧
    dist_M_CD / CD = dist_M_DA / DA := 
sorry

end distances_proportional_to_sides_l421_421510


namespace arcs_equivalence_l421_421793

variable (A B C D A1 A2 B1 B2 C1 C2 D1 D2 : Point)
variable (AB BC CD DA : Line)
variable (O : Circle)

-- Conditions
axiom intersects_AB : Circle.ints_Segment O AB = {A1, B2}
axiom intersects_BC : Circle.ints_Segment O BC = {B1, C2}
axiom intersects_CD : Circle.ints_Segment O CD = {C1, D2}
axiom intersects_DA : Circle.ints_Segment O DA = {D1, A2}
axiom cyclic_quad : CyclicQuadrilateral A B C D

-- To Prove
theorem arcs_equivalence :
  arcLength O A1 B1 + arcLength O C1 D1 = arcLength O B1 C1 + arcLength O D1 A1 := 
sorry

end arcs_equivalence_l421_421793


namespace minimum_perimeter_of_rectangle_ABCD_l421_421682

theorem minimum_perimeter_of_rectangle_ABCD
  (a : ℝ) (h_a_ne_zero : a ≠ 0)
  (AB_perp_x : ∀ (x : ℝ), x ≠ 0 → is_perpendicular (ABCD a x)) :
  let period := (2 * real.pi) / (|a|)
  let width := 2 * |a|
  let perimeter := 2 * period + 2 * width in
  (perimeter ≥ 8 * real.sqrt real.pi) :=
begin
  -- Proof omitted
  sorry
end

end minimum_perimeter_of_rectangle_ABCD_l421_421682


namespace domain_of_f_l421_421046

noncomputable def domain_of_function (x : ℝ) :=
  ∃ (f : ℝ → ℝ), f = λ x, real.sqrt (4 - x) + (x - 2)^0 ∧ x ≤ 4 ∧ x ≠ 2

theorem domain_of_f :
  ∀ x : ℝ, (domain_of_function x ↔ (x ≤ 4 ∧ x ≠ 2)) :=
by
  sorry

end domain_of_f_l421_421046


namespace max_ABCD_value_l421_421564

noncomputable def max_four_digit_number : ℕ :=
  -- Define the numbers in the circles A, B, C, D
  let A := 5 in
  let B := 4 in
  let C := 0 in
  let D := 3 in
  -- Calculate the four-digit number ABCD
  1000 * A + 100 * B + 10 * C + D

theorem max_ABCD_value : max_four_digit_number = 5304 := 
by {
  -- Proof is required here but omitted (sorry)
  sorry
}

end max_ABCD_value_l421_421564


namespace intersect_on_median_l421_421131

variable {P A B C A' B' C' M : Point}
variable {BC CA AB : Line}
variable (P_on_bisector : OnAngleBisector P A B C)
variable (PA'_on_BC : Perpendicular PA' P BC)
variable (PB'_on_CA : Perpendicular PB' P CA)
variable (PC'_on_AB : Perpendicular PC' P AB)
variable (M_middle_BC : Midpoint M B C)

theorem intersect_on_median 
  (h1 : OnLine A' BC) 
  (h2 : OnLine B' CA) 
  (h3 : OnLine C' AB) 
  (h4 : Perpendicular PA' BC)
  (h5 : Perpendicular PB' CA)
  (h6 : Perpendicular PC' AB) :
  ∃ E, Intersect PA' B'C' E ∧ OnLine E (Median A M) := 
sorry

end intersect_on_median_l421_421131


namespace Zilla_savings_l421_421476

/-- Zilla's monthly savings based on her spending distributions -/
theorem Zilla_savings
  (rent : ℚ) (monthly_earnings_percentage : ℚ)
  (other_expenses_fraction : ℚ) (monthly_rent : ℚ)
  (monthly_expenses : ℚ) (total_monthly_earnings : ℚ)
  (half_monthly_earnings : ℚ) (savings : ℚ)
  (h1 : rent = 133)
  (h2 : monthly_earnings_percentage = 0.07)
  (h3 : other_expenses_fraction = 0.5)
  (h4 : total_monthly_earnings = monthly_rent / monthly_earnings_percentage)
  (h5 : half_monthly_earnings = total_monthly_earnings * other_expenses_fraction)
  (h6 : savings = total_monthly_earnings - (monthly_rent + half_monthly_earnings))
  : savings = 817 :=
sorry

end Zilla_savings_l421_421476


namespace car_trip_time_l421_421144

theorem car_trip_time (T A : ℕ) (h1 : 50 * T = 140 + 53 * A) (h2 : T = 4 + A) : T = 24 := by
  sorry

end car_trip_time_l421_421144


namespace solve_abs_eq_l421_421900

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421900


namespace race_time_l421_421127

theorem race_time (v_A v_B : ℝ) (t tB : ℝ) (h1 : 200 / v_A = t) (h2 : 144 / v_B = t) (h3 : 200 / v_B = t + 7) : t = 18 :=
by
  sorry

end race_time_l421_421127


namespace probability_multiple_of_4_l421_421753

def prob_at_least_one_multiple_of_4 : ℚ :=
  1 - (38/50)^3

theorem probability_multiple_of_4 (n : ℕ) (h : n = 3) : 
  prob_at_least_one_multiple_of_4 = 28051 / 50000 :=
by
  rw [prob_at_least_one_multiple_of_4, ← h]
  sorry

end probability_multiple_of_4_l421_421753


namespace number_of_girls_l421_421862

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l421_421862


namespace trigonometric_identity_l421_421983

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 := 
by
  sorry

end trigonometric_identity_l421_421983


namespace true_propositions_l421_421550

-- Proposition ①
def proposition1 : Prop := 5 > 4 ∨ 4 > 5

-- Proposition ② (negation of "If a > b, then a + c > b + c")
def proposition2 (a b c : ℝ) : Prop := a ≤ b → a + c ≤ b + c

-- Proposition ③ (converse of "The diagonals of a rectangle are equal")
def is_rectangle (q : Type) [PlaneGeometry q] : Prop := 
  ∃ (a b c d : q), rectangle a b c d ∧ diagonals_equal a b c d

def proposition3 (q : Type) [PlaneGeometry q] : Prop :=
  (∃ (a b c d : q), diagonals_equal a b c d) → 
  is_rectangle q

-- The main theorem to state that proposition ① and ② are true, and proposition ③ is false.
theorem true_propositions : proposition1 ∧ (∀ a b c : ℝ, proposition2 a b c) ∧ ¬ proposition3 :=
by
  sorry

end true_propositions_l421_421550


namespace triangle_shape_l421_421308

variables {α : Type} [Preorder α] [AddGroup α] [MulGroup α] [MulAction α α]
variables (a b c A B C : α)

theorem triangle_shape (h1 : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  (h2 : A + B + C = π) (h3 : a⁄(Real.sin A) = b⁄(Real.sin B)  = c⁄(Real.sin C)) : 
  A = B ∨ 2 * (A + B) = π :=
sorry

end triangle_shape_l421_421308


namespace ratio_of_areas_l421_421665

variables {AB CD : ℝ} {A B C D E θ : ℝ}
variables (is_diameter : True) (is_parallel : True) (AC_ BD_intersect_E : True)
          (angle_AED_theta : ∠ E A D = θ) (angle_ADE_90: ∠ A D E = π / 2)

theorem ratio_of_areas (is_diameter : True) (is_parallel : True) (AC_BD_intersect_E : True)
  (angle_AED_theta : ∠ E A D = θ) (angle_ADE_90 : ∠ A D E = π / 2) :
  (triangle_area C D E) / (triangle_area A B E) = (Real.sin θ) ^ 2 := 
sorry

end ratio_of_areas_l421_421665


namespace fraction_eaten_by_javier_l421_421709

-- Given conditions and calculations in Lean
def total_cookies : ℕ := 200
def wife_fraction : ℚ := 30 / 100
def daughter_cookies : ℕ := 40
def uneaten_cookies : ℕ := 50

-- Number of cookies taken by wife
def cookies_taken_by_wife (total : ℕ) (fraction : ℚ) := (fraction * total).toNat
def remaining_after_wife (total : ℕ) (taken_by_wife : ℕ) := total - taken_by_wife

-- Number of cookies taken by daughter
def remaining_after_daughter (remaining : ℕ) (taken_by_daughter : ℕ) := remaining - taken_by_daughter

-- Prove that Javier ate half of the remaining cookies
theorem fraction_eaten_by_javier :
  let wife_taken := cookies_taken_by_wife total_cookies wife_fraction in
  let remaining_wife := remaining_after_wife total_cookies wife_taken in
  let remaining_daughter := remaining_after_daughter remaining_wife daughter_cookies in
  let javier_eaten := remaining_daughter - uneaten_cookies in
  javier_eaten.to_rat / remaining_daughter.to_rat = (1 / 2 : ℚ) := by {
  sorry
}

end fraction_eaten_by_javier_l421_421709


namespace players_quit_l421_421871

theorem players_quit (initial_players remaining_lives lives_per_player : ℕ) 
  (h1 : initial_players = 8) (h2 : remaining_lives = 15) (h3 : lives_per_player = 5) :
  initial_players - (remaining_lives / lives_per_player) = 5 :=
by
  -- A proof is required here
  sorry

end players_quit_l421_421871


namespace a_1994_is_7_l421_421611

-- Sequence definition as per the given conditions
def a : ℕ → ℕ
| 0 := 3
| 1 := 7
| (n+2) := (a n * a (n+1)) % 10

-- The proof statement
theorem a_1994_is_7 : a 1994 = 7 := 
by 
  sorry

end a_1994_is_7_l421_421611


namespace union_complement_l421_421633

universe u

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1}

theorem union_complement (U A B : Set ℕ) (hU : U = {0, 2, 4, 6, 8, 10}) (hA : A = {2, 4, 6}) (hB : B = {1}) :
  (U \ A) ∪ B = {0, 1, 8, 10} :=
by
  -- The proof is omitted.
  sorry

end union_complement_l421_421633


namespace math_proof_problem_l421_421393

noncomputable def sum_of_distinct_squares (a b c : ℕ) : ℕ :=
3 * ((a^2 + b^2 + c^2 : ℕ))

theorem math_proof_problem (a b c : ℕ)
  (h1 : a + b + c = 27)
  (h2 : Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 11) :
  sum_of_distinct_squares a b c = 2274 :=
sorry

end math_proof_problem_l421_421393


namespace part1_part2_l421_421662

-- Define the conditions and question for part (1)
theorem part1 (A : ℝ) (hA : A = π / 2) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (2 * x + A)) : 
  f (-π / 6) = 1 / 2 :=
by
  have : f (-π / 6) = Real.sin (2 * (-π / 6) + A) := hf (-π / 6)
  rw [hA] at this
  simp at this
  rw [this, Real.sin_pi_div_6]
  norm_num
  sorry

-- Define the conditions and question for part (2)
theorem part2 (A B a b : ℝ) (f : ℝ → ℝ)
  (hf₁ : f (π / 12) = 1)
  (hf₂ : ∀ x, f x = Real.sin (2 * x + A))
  (ha : a = 3)
  (hcosB : Real.cos B = 4 / 5) :
  b = 6 * Real.sqrt 3 / 5 :=
by
  have sinB : Real.sin B = 3 / 5 :=
    by sorry  -- Prove that sin B = 3 / 5 from cos B using trigonometric identities.
  have sinA : Real.sin (π / 3) = Real.sqrt 3 / 2 :=
    by simp
  have : f (π / 12) = Real.sin (2 * (π / 12) + A) := hf₂ (π / 12)
  rw [hf₁] at this
  have hA : A = π / 3 :=
    by sorry  -- Solve 2(π / 12) + A = π / 2 for A.
  rw [sinA] at ha
  have : b / (3 / 5) = 3 / (Real.sqrt 3 / 2) :=
    by sorry  -- Apply the sine law.
  rw [Real.sqrt_div((3:ℝ), 2)] -- √3 / 2 simplification
  norm_num
  sorry

end part1_part2_l421_421662


namespace decimal_150th_digit_of_5_over_11_l421_421111

theorem decimal_150th_digit_of_5_over_11 :
  let repeating_sequence : string := "45" in
  let sequence_length := 2 in
  ∀ n, n = 150 →
  let digit_position := n % sequence_length in
  (if digit_position = 0 then repeating_sequence.get 1 else repeating_sequence.get (digit_position - 1)) = '5' :=
by
  sorry

end decimal_150th_digit_of_5_over_11_l421_421111


namespace concyclic_points_K_M_L_N_l421_421975

theorem concyclic_points_K_M_L_N
  (ΔABC : Type)
  [triangle ΔABC]
  (A B C O H M N L K : ΔABC)
  (AB AC : ℝ)
  (H_parallel_AB_M_inter_AC : ∃p, line_parallel_ab_passing_through_H_and_intersects_AC_at_M A B C H M)
  (H_parallel_AC_N_inter_AB : ∃q, line_parallel_ac_passing_through_H_and_intersects_AB_at_N A B C H N)
  (L_reflection_H_MN : reflection_of_H_across_MN H M N L)
  (OL_intersects_AH_at_K : line_intersects A H O L K)
  (acute_angled_ΔABC : is_acute_angled_triangle A B C)
  (AB_gt_AC : AB > AC) :
  concyclic_points K M L N := sorry

end concyclic_points_K_M_L_N_l421_421975


namespace Mp_not_square_l421_421596

open Nat

def balanced_sequences_count (p : ℕ) : ℕ :=
  -- placeholder definition to represent M_p
  sorry

theorem Mp_not_square {p : ℕ} (hp_prime : Prime p) (hp_mod : p % 4 = 3) : 
  ¬ ∃ (k : ℕ), k * k = balanced_sequences_count p :=
by sorry

end Mp_not_square_l421_421596


namespace sin_13pi_over_4_eq_neg_sqrt2_over_2_l421_421566

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end sin_13pi_over_4_eq_neg_sqrt2_over_2_l421_421566


namespace sum_floor_ceiling_difference_l421_421213

theorem sum_floor_ceiling_difference :
    (∑ k in Finset.range 2010 \.succ, ((2010 : ℝ) / k - (Real.floor ((2010 : ℝ) / k)))) = 1994 :=
by sorry

end sum_floor_ceiling_difference_l421_421213


namespace decreased_revenue_l421_421482

variable (T C : ℝ)
def Revenue (tax consumption : ℝ) : ℝ := tax * consumption

theorem decreased_revenue (hT_new : T_new = 0.9 * T) (hC_new : C_new = 1.1 * C) :
  Revenue T_new C_new = 0.99 * (Revenue T C) := 
sorry

end decreased_revenue_l421_421482


namespace breadth_reduction_percentage_l421_421426

theorem breadth_reduction_percentage (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    (∃ x : ℝ, 
      x = 13.33 ∧ 
      (let L' := 1.20 * L in
       let B' := B * (1 - x / 100) in
       L' * B' = 1.04 * L * B)) :=
by
  let x := 13.33
  use x
  let L' := 1.20 * L
  let B' := B * (1 - x / 100)
  have h : L' * B' = 1.04 * L * B := sorry
  use h
  sorry

end breadth_reduction_percentage_l421_421426


namespace carol_first_6_l421_421168

def probability_carl_first_6_roll 
  (prob_roll_6 : ℚ := 1/6) 
  (prob_not_roll_6 : ℚ := 5/6) : ℚ := 
  (1/6) * (prob_not_roll_6^2 * ∑' (n : ℕ), (prob_not_roll_6^(3 * n)))

theorem carol_first_6 (prob_roll_6 : ℚ := 1/6) 
  (prob_not_roll_6 : ℚ := 5/6) :
  probability_carl_first_6_roll = 25/91 := 
by 
  sorry

end carol_first_6_l421_421168


namespace strokes_over_par_l421_421094

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l421_421094


namespace apple_slices_count_l421_421780

theorem apple_slices_count :
  let boxes := 7
  let apples_per_box := 7
  let slices_per_apple := 8
  let total_apples := boxes * apples_per_box
  let total_slices := total_apples * slices_per_apple
  total_slices = 392 :=
by
  sorry

end apple_slices_count_l421_421780


namespace area_of_rectangle_area_of_triangle_l421_421010

namespace geometry

def points_on_rectangle (A B C D M N : Type) :=
  (AN NC AM MB : ℕ) -- Assume these are natural numbers for the lengths
  (h1 : AN = 3) -- Condition 1
  (h2 : NC = 39) -- Condition 2
  (h3 : AM = 10) -- Condition 3
  (h4 : MB = 5) -- Condition 4
  (ABCD_is_rectangle : true) -- Condition 5, assume it's given

-- Define the problem statement
theorem area_of_rectangle (A B C D M N : Type)
  [points_on_rectangle A B C D M N]:
  ∀ {AN NC AM MB : ℕ},
  AN = 3 → NC = 39 → AM = 10 → MB = 5 →
  rectangle_area : ℕ :=
  λ AN NC AM MB h1 h2 h3 h4, 630 := sorry

theorem area_of_triangle (A B C D M N : Type)
  [points_on_rectangle A B C D M N]:
  ∀ {AN NC AM MB : ℕ},
  AN = 3 → NC = 39 → AM = 10 → MB = 5 →
  triangle_area : ℝ :=
  λ AN NC AM MB h1 h2 h3 h4, 202.5 := sorry

end geometry

end area_of_rectangle_area_of_triangle_l421_421010


namespace t_shape_figure_perimeter_l421_421180

-- Define the geometric problem with conditions
def is_t_shape_figure (total_area : ℕ) (num_squares : ℕ) (perimeter : ℕ) : Prop :=
  total_area = 125 ∧ 
  num_squares = 5 ∧ 
  perimeter = 35

-- The main statement to be proved
theorem t_shape_figure_perimeter :
  ∃ (perimeter : ℕ), let total_area := 125 in
                       let num_squares := 5 in
                       is_t_shape_figure total_area num_squares perimeter :=
by
  use 35
  unfold is_t_shape_figure
  simp
  sorry

end t_shape_figure_perimeter_l421_421180


namespace smallest_value_of_expression_l421_421726

noncomputable def smallest_value_problem
  (a b c d : ℤ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (ω : ℂ) (hω : ω^4 = 1 ∧ ω ≠ 1) : ℝ :=
|a + b * ω + c * (ω^2) + d * (ω^3)|

theorem smallest_value_of_expression 
  (a b c d : ℤ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (ω : ℂ) (hω : ω^4 = 1 ∧ ω ≠ 1) :
  smallest_value_problem a b c d h ω hω = sqrt 2 :=
sorry

end smallest_value_of_expression_l421_421726


namespace f_sum_value_l421_421795

-- Definition of the Riemann function R(x)
def R (x : ℝ) : ℝ :=
  if ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ x = q / p ∧ nat.coprime q p ∧ q < p then
    let ⟨p, q, hp, hq, hx, hc, hlt⟩ := classical.some_spec (classical.some_spec (classical.some x).some_spec) in 1 / p
  else if x = 0 ∨ x = 1 then 0
  else 0

-- Given conditions
def f (x : ℝ) : ℝ := sorry

-- Hypotheses about f(x)
axiom odd_f {x : ℝ} : f (-x) = -f x
axiom periodic_f {x : ℝ} : f (1 + x) = -f (1 - x)
axiom initial_f {x : ℝ} (h : x ∈ set.Icc 0 1) : f x = R x

-- The goal
theorem f_sum_value :
  f 2023 + f (2023 / 2) + f (-2023 / 3) = -5 / 6 :=
sorry

end f_sum_value_l421_421795


namespace Oleg_older_than_Ekaterina_oldest_is_Roman_and_married_to_Zhanna_l421_421025

def Roman : ℕ := sorry
def Oleg : ℕ := sorry
def Ekaterina : ℕ := sorry
def Zhanna : ℕ := sorry

axiom diff_ages : Roman ≠ Oleg ∧ Roman ≠ Ekaterina ∧ Roman ≠ Zhanna ∧ Oleg ≠ Ekaterina ∧ Oleg ≠ Zhanna ∧ Ekaterina ≠ Zhanna
axiom husband_older (h w : ℕ) : (h = Roman ∨ h = Oleg) → (w = Zhanna ∨ w = Ekaterina) → h > w
axiom zhanna_older_than_oleg : Zhanna > Oleg

theorem Oleg_older_than_Ekaterina : Oleg > Ekaterina :=
by {
  sorry
}

theorem oldest_is_Roman_and_married_to_Zhanna : ∀ p, (p = Roman ∧ (∀ q, q ≠ Roman → Roman > q)) ∧ (p = Roman → ∃ w, w = Zhanna ∧ husband_older Roman w) :=
by {
  sorry
}

end Oleg_older_than_Ekaterina_oldest_is_Roman_and_married_to_Zhanna_l421_421025


namespace work_combined_days_l421_421124

theorem work_combined_days (W : ℝ) (D : ℝ) :
  (∀ W > 0, ∀ D > 0,
  (A_work_rate = W / 6) → (B_work_rate = W / 6) → 
  ((A_work_rate + B_work_rate) * D = W) → D = 3) := 
begin
  intros W W_pos D D_pos A_rate B_rate total_work,
  sorry
end

end work_combined_days_l421_421124


namespace probability_red_side_first_on_third_roll_l421_421125

noncomputable def red_side_probability_first_on_third_roll : ℚ :=
  let p_non_red := 7 / 10
  let p_red := 3 / 10
  (p_non_red * p_non_red * p_red)

theorem probability_red_side_first_on_third_roll :
  red_side_probability_first_on_third_roll = 147 / 1000 := 
sorry

end probability_red_side_first_on_third_roll_l421_421125


namespace sum_of_c_values_l421_421422

def g (x : ℝ) : ℝ := ((x - 3) * (x - 1) * (x + 1) * (x + 3)) / 24 - 1.5

theorem sum_of_c_values : 
  let c_values := {c : ℝ | ∃ x1 x2 x3 x4 : ℝ, g x1 = c ∧ g x2 = c ∧ g x3 = c ∧ g x4 = c ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4}
  let integer_c_values := {c : ℝ | c ∈ c_values ∧ c.floor = c}
  ∑ c in integer_c_values, c = -4 := 
by
  sorry

end sum_of_c_values_l421_421422


namespace fair_coin_second_head_l421_421122

theorem fair_coin_second_head (P : ℝ) 
  (fair_coin : ∀ outcome : ℝ, outcome = 0.5) :
  P = 0.5 :=
by
  sorry

end fair_coin_second_head_l421_421122


namespace total_cost_of_tickets_l421_421363

def number_of_adults := 2
def number_of_children := 3
def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6

theorem total_cost_of_tickets :
  let total_cost := number_of_adults * cost_of_adult_ticket + number_of_children * cost_of_child_ticket
  total_cost = 77 :=
by
  sorry

end total_cost_of_tickets_l421_421363


namespace product_of_two_numbers_l421_421056

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l421_421056


namespace number_of_correct_propositions_l421_421077

def line_parallel_to_plane (a : Type) (α : Type) : Prop := sorry
def planes_intersect_line (α β : Type) : Type := sorry
def planes_parallel (α β : Type) : Prop := sorry
def lines_parallel (a b : Type) : Prop := sorry
def unique_parallel_planes_for_skew_lines (a b : Type) (α β : Type) : Prop := sorry

theorem number_of_correct_propositions :
  let a := Type
  let b := Type
  let m := Type
  let α := Type
  let β := Type in
  ((line_parallel_to_plane a α) ∧ (line_parallel_to_plane a β) ∧ (planes_intersect_line α β = m) → (planes_parallel α m)) ∧
  ((line_parallel_to_plane a α) → (∀ l : Type, (l ∈ α) → (lines_parallel a l))) ∧
  ((line_parallel_to_plane a α) ∧ (planes_parallel α β) → (line_parallel_to_plane a β)) ∧
  ((lines_parallel a b) ∧ (line_parallel_to_plane a α) → (line_parallel_to_plane b α)) ∧
  (unique_parallel_planes_for_skew_lines a b α β) →
  2 := sorry

end number_of_correct_propositions_l421_421077


namespace part1_part2_l421_421588

variable (a : ℝ)

def p : Prop := ∀ x ∈ Set.Icc (-2 : ℝ) (-1), x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x - (a - 2) = 0

-- Part 1
theorem part1 (h : p a) : a ≤ 1 :=
sorry

-- Part 2
theorem part2 (h : p a ∨ q a) (hn : ¬ (p a ∧ q a)) : a ∈ Set.Ioo (-2 : ℝ) 1 ∪ Set.Ioo 1 ⊤ :=
sorry

end part1_part2_l421_421588


namespace discriminant_formula_l421_421356

def discriminant_cubic_eq (x1 x2 x3 p q : ℝ) : ℝ :=
  (x1 - x2)^2 * (x2 - x3)^2 * (x3 - x1)^2

theorem discriminant_formula (x1 x2 x3 p q : ℝ)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x1 * x3 + x2 * x3 = p)
  (h3 : x1 * x2 * x3 = -q) :
  discriminant_cubic_eq x1 x2 x3 p q = -4 * p^3 - 27 * q^2 :=
by sorry

end discriminant_formula_l421_421356


namespace find_inradius_of_inscribed_circle_l421_421463

-- Define a triangle with sides AB, AC, and BC given by the problem conditions.
variables (A B C : Type) [metric_space A]

open_locale big_operators

-- Define the side lengths of the triangle.
def AB : ℝ := 8
def AC : ℝ := 8
def BC : ℝ := 5

-- Define the semi-perimeter of the triangle.
def semiperimeter : ℝ := (AB + AC + BC) / 2

-- Define Heron's formula for the area of the triangle.
def herons_area : ℝ := real.sqrt (semiperimeter * (semiperimeter - AB) * (semiperimeter - AC) * (semiperimeter - BC))

-- Define the radius of the inscribed circle.
def inradius : ℝ := herons_area / semiperimeter

-- The theorem to be proven.
theorem find_inradius_of_inscribed_circle (h₁ : AB = 8) (h₂ : AC = 8) (h₃ : BC = 5) :
  inradius = 38 / 21 :=
begin
  sorry
end

end find_inradius_of_inscribed_circle_l421_421463


namespace four_student_round_table_l421_421956

theorem four_student_round_table
  (G : SimpleGraph (Fin 2021))
  (h_deg : ∀ v, G.degree v ≥ 45) :
  ∃ (v1 v2 v3 v4 : Fin 2021), 
    G.Adj v1 v2 ∧ G.Adj v2 v3 ∧ G.Adj v3 v4 ∧ G.Adj v4 v1 ∧ 
    ¬ v1 = v3 ∧ ¬ v2 = v4 :=
by
  sorry

end four_student_round_table_l421_421956


namespace harold_total_cost_l421_421760

theorem harold_total_cost 
  (d : ℝ) (c : ℝ) 
  (cost_melinda : ℝ) 
  (cost_doughnut : ℝ) 
  (cost_coffee : ℝ) 
  (num_doughnuts_melinda : ℕ) 
  (num_coffees_melinda : ℕ) 
  (num_doughnuts_harold : ℕ) 
  (num_coffees_harold : ℕ) :
  cost_melinda = 7.59 →
  cost_doughnut = 0.45 →
  num_doughnuts_melinda = 5 →
  num_coffees_melinda = 6 →
  num_doughnuts_harold = 3 →
  num_coffees_harold = 4 →
  (num_doughnuts_melinda * cost_doughnut + num_coffees_melinda * cost_coffee) = cost_melinda →
  ∃ (total_harold : ℝ), total_harold = (num_doughnuts_harold * cost_doughnut + num_coffees_harold * cost_coffee) ∧ total_harold = 4.91 :=
begin
  sorry
end

end harold_total_cost_l421_421760


namespace part1_l421_421750

def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

theorem part1 (a : ℝ) (h : a = 1) :
  (Set.compl B ∪ A a) = {x | x ≤ 1 ∨ x ≥ 2} :=
by
  sorry

end part1_l421_421750


namespace three_people_same_topic_l421_421782

open Classical

theorem three_people_same_topic (people : Fin 17 → Type) (topics : Fin 3 → Type) 
  (corresponds : (Fin 17 → Fin 17 → Fin 3) → Prop) :
    ∃ (a b c : Fin 17), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
      (∃ (t : Fin 3), corresponds (λ x y, t) a b ∧ corresponds (λ x y, t) b c ∧ corresponds (λ x y, t) c a) :=
  sorry

end three_people_same_topic_l421_421782


namespace angle_AHE_degrees_l421_421236

-- Define the problem conditions and theorem in Lean 4
noncomputable theory

def regular_octagon (A B C D E F G H : Point) : Prop :=
  is_regular_polygon 8 [A, B, C, D, E, F, G, H]

theorem angle_AHE_degrees (A B C D E F G H : Point) (h : regular_octagon A B C D E F G H) :
  angle A H E = 22.5 :=
sorry

end angle_AHE_degrees_l421_421236


namespace probability_odd_sum_l421_421223

theorem probability_odd_sum:
  (∃ E : Finset (Finset ℕ), E.card = 10 ∧ 
      (E.filter (λ S, (S.card = 2) ∧ (∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a + b) % 2 = 1))).card = 6) ->
  (6 / 10 = (3 / 5)) :=
by
  sorry

end probability_odd_sum_l421_421223


namespace isosceles_triangle_angle_measure_l421_421170

theorem isosceles_triangle_angle_measure
  (isosceles : Triangle → Prop)
  (exterior_angles : Triangle → ℝ → ℝ → Prop)
  (ratio_1_to_4 : ∀ {T : Triangle} {a b : ℝ}, exterior_angles T a b → b = 4 * a)
  (interior_angles : Triangle → ℝ → ℝ → ℝ → Prop) :
  ∀ (T : Triangle), isosceles T → ∃ α β γ : ℝ, interior_angles T α β γ ∧ α = 140 ∧ β = 20 ∧ γ = 20 := 
by
  sorry

end isosceles_triangle_angle_measure_l421_421170


namespace perimeter_of_triangle_is_16_l421_421573

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem perimeter_of_triangle_is_16 :
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  distance A B + distance B C + distance C A = 16 :=
by
  let A := (2, 3)
  let B := (2, 9)
  let C := (6, 6)
  have dAB : distance A B = 6 := by
    simp [distance]
    norm_num
  have dBC : distance B C = 5 := by
    simp [distance]
    norm_num
  have dCA : distance C A = 5 := by
    simp [distance]
    norm_num
  calc
    distance A B + distance B C + distance C A = 6 + 5 + 5 := by rw [dAB, dBC, dCA]
    ... = 16 := by norm_num

end perimeter_of_triangle_is_16_l421_421573


namespace measure_of_angle_A_possibilities_l421_421818

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l421_421818


namespace average_age_decrease_l421_421414

theorem average_age_decrease (N : ℕ) (T : ℝ) 
  (h1 : T = 40 * N) 
  (h2 : ∀ new_average_age : ℝ, (T + 12 * 34) / (N + 12) = new_average_age → new_average_age = 34) :
  ∃ decrease : ℝ, decrease = 6 :=
by
  sorry

end average_age_decrease_l421_421414


namespace jolyn_older_than_leon_l421_421579

open Nat

def Jolyn := Nat
def Therese := Nat
def Aivo := Nat
def Leon := Nat

-- Conditions
variable (jolyn therese aivo leon : Nat)
variable (h1 : jolyn = therese + 2) -- Jolyn is 2 months older than Therese
variable (h2 : therese = aivo + 5) -- Therese is 5 months older than Aivo
variable (h3 : leon = aivo + 2) -- Leon is 2 months older than Aivo

theorem jolyn_older_than_leon :
  jolyn = leon + 5 := by
  sorry

end jolyn_older_than_leon_l421_421579


namespace problem1_problem2_l421_421533

/- Problem 1 -/
theorem problem1 {a : ℝ} (h : a + a⁻¹ = 4) : a^(1/2) + a^(-1/2) = real.sqrt 6 :=
  sorry

/- Problem 2 -/
theorem problem2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * real.log10 ((x - y) / 2) = real.log10 x + real.log10 y) :
  real.log (3 - 2 * real.sqrt 2) (x / y) = -1 :=
  sorry

end problem1_problem2_l421_421533


namespace arithmetic_geometric_proof_l421_421448

variable {A : ℕ → ℕ} {G : ℕ → ℕ}
variables (a1 a2 g1 g2 g3 : ℕ)
variable {m : ℕ}

-- The conditions
axiom h1 : a1 = 1
axiom h2 : g1 = 1
axiom h3 : a2 = g2 ∧ a2 ≠ 1
axiom h4 : a m = G 3
axiom h5 : m > 3

-- The math proof problem
theorem arithmetic_geometric_proof :
  let d := m - 3
  let q := m - 2
  g4_eq_a_term : (G 4) = A (Nat.succ (Nat.pred (Nat.pred m * m - 2 * m + 2)))
  (∀ j : ℕ, ∃ k : ℕ, G (j+1) = A k) :=
by
  intro d q h1 h2 h3 h4 h5
  split
  { sorry } -- Proof 1: Prove d = m - 3 and q = m - 2
  { sorry } -- Proof 2: Prove that g4 = A term
  { sorry } -- Proof 3: Prove that every term of G appears in A

end arithmetic_geometric_proof_l421_421448


namespace find_x_l421_421892

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421892


namespace rectangle_perimeter_l421_421667

theorem rectangle_perimeter (A : Point) (distances : List ℝ) :
  (length distances = 4) →
  (sorted distances) →
  (distances = [1, 2, 3, 4]) →
  (∃ (length width : ℝ),
   length = 4 ∧ width = 6 ∧
   2 * (length + width) = 20) := 
by
  sorry

end rectangle_perimeter_l421_421667


namespace regular_polyhedron_concentric_spheres_l421_421020

-- Define a regular polyhedron and its properties
variables {P : Type} [RegularPolyhedron P]

-- Define the center of the polyhedron
def center (P : Type) [RegularPolyhedron P] : P := sorry

-- Define the conditions for the concentric spheres
def circumscribed_sphere (c : P) (r : ℝ) : Prop := sorry
def edge_sphere (c : P) (r : ℝ) : Prop := sorry
def inscribed_sphere (c : P) (r : ℝ) : Prop := sorry

-- Prove the existence of these spheres
theorem regular_polyhedron_concentric_spheres (P : Type) [RegularPolyhedron P] :
  ∃ (c : P) (r1 r2 r3 : ℝ),
    circumscribed_sphere c r1 ∧
    edge_sphere c r2 ∧
    inscribed_sphere c r3 :=
sorry

end regular_polyhedron_concentric_spheres_l421_421020


namespace gary_new_repayment_plan_l421_421224

theorem gary_new_repayment_plan :
  ∃ y : ℕ, y = 2 ∧ -- Gary's new repayment plan is 2 years
  let total_amount := 6000 in
  let original_years := 5 in
  let extra_per_month := 150 in
  let months_in_year := 12 in
  let original_months := months_in_year * original_years in
  let original_monthly_payment := total_amount / original_months in
  let new_monthly_payment := original_monthly_payment + extra_per_month in
  let new_months := total_amount / new_monthly_payment in
  y = new_months / months_in_year :=
begin
  sorry -- Proof to be filled in.
end

end gary_new_repayment_plan_l421_421224


namespace calories_left_for_dinner_l421_421557

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l421_421557


namespace required_bandwidth_channel_l421_421931

theorem required_bandwidth_channel
  (session_duration_min : ℕ)
  (sampling_rate : ℕ)
  (sampling_depth : ℕ)
  (metadata_volume : ℕ)
  (audio_per_kibibit : ℕ)
  (stereo_factor : ℕ) :
  session_duration_min = 51 →
  sampling_rate = 63 →
  sampling_depth = 17 →
  metadata_volume = 47 →
  audio_per_kibibit = 5 →
  stereo_factor = 2 →
  let time_in_seconds := session_duration_min * 60 in
  let data_volume := sampling_rate * sampling_depth * time_in_seconds in
  let metadata_bits := metadata_volume * 8 in
  let metadata_volume_bits := (metadata_bits * data_volume) / (audio_per_kibibit * 1024) in
  let total_data_volume := (data_volume + metadata_volume_bits) * stereo_factor in
  (total_data_volume / 1024) / time_in_seconds = 2.25 :=
sorry

end required_bandwidth_channel_l421_421931


namespace friend_gives_amount_l421_421209

theorem friend_gives_amount :
  let earnings := [12, 18, 24, 30, 45] in
  let total := List.sum earnings in
  let share := total / (earnings.length : Int) in
  let diff := (45 : Int) - share in
  diff = 19.2 :=
by
  sorry

end friend_gives_amount_l421_421209


namespace last_remaining_number_l421_421366

theorem last_remaining_number :
  ∃ n, n ∈ range (1, 2^10) ∧ (n % 2 = 1) :=
by
  sorry

end last_remaining_number_l421_421366


namespace difference_mean_median_is_neg_half_l421_421007

-- Definitions based on given conditions
def scoreDistribution : List (ℕ × ℚ) :=
  [(65, 0.05), (75, 0.25), (85, 0.4), (95, 0.2), (105, 0.1)]

-- Defining the total number of students as 100 for easier percentage calculations
def totalStudents := 100

-- Definition to compute mean
def mean : ℚ :=
  scoreDistribution.foldl (λ acc (score, percentage) => acc + (↑score * percentage)) 0

-- Median score based on the distribution conditions
def median : ℚ := 85

-- Proving the proposition that the difference between the mean and the median is -0.5
theorem difference_mean_median_is_neg_half :
  median - mean = -0.5 :=
sorry

end difference_mean_median_is_neg_half_l421_421007


namespace volume_of_water_in_cylindrical_tank_l421_421149

theorem volume_of_water_in_cylindrical_tank
  (r : ℝ) (h : ℝ) (depth : ℝ)
  (r_eq : r = 5) (h_eq : h = 10) (depth_eq : depth = 3) :
  volume_of_water r h depth = 290.73 * real.pi - 91.65 := 
by 
  -- Conditions
  have r_val : r = 5 := r_eq,
  have h_val : h = 10 := h_eq,
  have depth_val : depth = 3 := depth_eq,
  -- Calculate volume (skipped)
  sorry

end volume_of_water_in_cylindrical_tank_l421_421149


namespace range_of_m_l421_421644

theorem range_of_m 
  (m : ℝ) 
  (h1 : 2 * m + 1 ≥ 0)
  (h2 : m^2 + m - 1 ≥ 0) 
  (h3 : sqrt (2 * m + 1) > sqrt (m^2 + m - 1)) :
  (sqrt 5 - 1) / 2 ≤ m ∧ m < 2 :=
sorry

end range_of_m_l421_421644


namespace maximum_unique_walks_l421_421044

-- Define the conditions
def starts_at_A : Prop := true
def crosses_bridge_1_first : Prop := true
def finishes_at_B : Prop := true
def six_bridges_linking_two_islands_and_banks : Prop := true

-- Define the theorem to prove the maximum number of unique walks is 6
theorem maximum_unique_walks : starts_at_A ∧ crosses_bridge_1_first ∧ finishes_at_B ∧ six_bridges_linking_two_islands_and_banks → ∃ n, n = 6 :=
by
  intros
  existsi 6
  sorry

end maximum_unique_walks_l421_421044


namespace intersection_distance_l421_421201

-- Define the parabola equation
def parabola (x y : ℝ) : Prop :=
  y^2 = 12 * x

-- Define the circle equation
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Points of intersection
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (3, -6)

-- Statement of the proof problem
theorem intersection_distance :
  parabola pointA.1 pointA.2 ∧ parabola pointB.1 pointB.2 ∧
  circle pointA.1 pointA.2 ∧ circle pointB.1 pointB.2 ∧
  distance pointA.1 pointA.2 pointB.1 pointB.2 = 3 * Real.sqrt 5 :=
by
  sorry

end intersection_distance_l421_421201


namespace first_expression_eq_second_expression_eq_l421_421979

-- Definition of the first expression
def first_expression :=
  (9 / 4)^(1 / 2) - 1 - (-27 / 8)^(-2 / 3) + (2 / 3)^2

-- Definition of the second expression
def second_expression :=
  2 * (Real.log 5 / Real.log 2) * (3 / 2 * 1 + Real.log 2 / Real.log 3) * (2 * Real.log 3 / Real.log 5)

-- Theorem to state the first expression equals 1/2
theorem first_expression_eq : first_expression = 1 / 2 :=
by
  sorry

-- Theorem to state the second expression equals 6
theorem second_expression_eq : second_expression = 6 :=
by
  sorry

end first_expression_eq_second_expression_eq_l421_421979


namespace sum_odd_powers_binom_eqn_l421_421691

theorem sum_odd_powers_binom_eqn (a : ℕ) (f : ℕ → ℕ) :
  let S := (range (2015 + 1)).sum (λ k, if k % 2 = 1 then a * (binomial 2015 k) * 2 ^ (2015 - k) else 0)
  then S = 2 ^ 4029 := by 
sorry

end sum_odd_powers_binom_eqn_l421_421691


namespace find_circle_equation_l421_421592

-- Define the basic conditions of the problem
def is_center_on_x_axis (a : ℝ) : Prop := ∃y : ℝ, y = 0

def passes_through_point (a : ℝ) (p : ℝ × ℝ) : Prop := 
  let r := Real.abs (a - 1)
  (p.1 - a)^2 + p.2^2 = r^2

def length_of_chord (a : ℝ) : Prop :=
  let r := Real.abs (a - 1)
  let d := Real.abs (a - 1) / 2
  let chord_length := 2 * Real.sqrt 3
  (chord_length / 2)^2 + d^2 = r^2

-- The main theorem with conditions and result
theorem find_circle_equation (a : ℝ) :
  is_center_on_x_axis a →
  passes_through_point a (1, 0) →
  length_of_chord a →
  (∃ k : ℝ, k = 4 ∧ ((a = 3 ∧ (x - 3)^2 + y^2 = k) ∨ (a = -1 ∧ (x + 1)^2 + y^2 = k))) :=
by 
  intro h_center h_passes h_chord
  by_cases h1 : a = 3
  · use 4
    simp [h1]
    left
    split
    · exact h1
    · rfl
  by_cases h2 : a = -1
  · use 4
    simp [h2]
    right
    split
    · exact h2
    · rfl
  · exfalso
    sorry

end find_circle_equation_l421_421592


namespace total_songs_performed_l421_421580

theorem total_songs_performed (lucy_songs : ℕ) (sarah_songs : ℕ) (beth_songs : ℕ) (jane_songs : ℕ) 
  (h1 : lucy_songs = 8)
  (h2 : sarah_songs = 5)
  (h3 : sarah_songs < beth_songs)
  (h4 : sarah_songs < jane_songs)
  (h5 : beth_songs < lucy_songs)
  (h6 : jane_songs < lucy_songs)
  (h7 : beth_songs = 6 ∨ beth_songs = 7)
  (h8 : jane_songs = 6 ∨ jane_songs = 7) :
  (lucy_songs + sarah_songs + beth_songs + jane_songs) / 3 = 9 :=
by
  sorry

end total_songs_performed_l421_421580


namespace max_sum_terms_l421_421576

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_d : d < 0)
variable (h_a1_a11 : a 1 ^ 2 = a 11 ^ 2)

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

theorem max_sum_terms (h_seq : ∀ n, a (n + 1) = a n + d) :
  ∀ n, S a n ≤ S a 5 ∨ S a n ≤ S a 6 :=
sorry

end max_sum_terms_l421_421576


namespace part1_part2_part3_l421_421747

-- Define f(x) = ln(1 + x) - m * x, where m > 0
def f (x : ℝ) (m : ℝ) : ℝ := Real.log (1 + x) - m * x

-- Part 1: Prove the function f(x) is monotonically decreasing on (0, +∞) when m = 1
theorem part1 (x : ℝ) : 
(m = 1) → (0 < x → monotone_decreasing (f x 1)) :=
sorry

-- Part 2: Find the maximum value of f(x) for m > 0
theorem part2 (m : ℝ) (hm : 0 < m) :
  (∀ x, -1 < x → x ≤ (1 / m) - 1 → monotone_increasing (f x m)) ∧
  (∀ x, (1 / m) - 1 ≤ x → monotone_decreasing (f x m)) ∧ 
  (∀ x, x = (1 / m) - 1 → f x m = m - Real.log m - 1) :=
sorry

-- Part 3: Prove the range of m for exactly two zeros of f(x) in [0, e^2 - 1]
theorem part3 (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.exp 2 - 1 → (has_two_zeros (f x m) ∧ (2 / (Real.exp 2 - 1) ≤ m ∧ m < 1))) :=
sorry

end part1_part2_part3_l421_421747


namespace tens_digit_of_binary_result_l421_421794

def digits_tens_digit_subtraction (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) : ℕ :=
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let difference := original_number - reversed_number
  (difference % 100) / 10

theorem tens_digit_of_binary_result (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) :
  digits_tens_digit_subtraction a b c h1 h2 = 9 :=
sorry

end tens_digit_of_binary_result_l421_421794


namespace num_real_a_satisfy_union_l421_421252

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_l421_421252


namespace steve_reads_book_l421_421035

theorem steve_reads_book (pages_per_day : ℕ) (book_length : ℕ) (days_per_week : ℕ) :
  (pages_per_day = 100) →
  (book_length = 2100) →
  (days_per_week = 3) →
  (book_length / (pages_per_day * days_per_week) = 7) :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end steve_reads_book_l421_421035


namespace fraction_neither_cell_phones_nor_pagers_l421_421976

theorem fraction_neither_cell_phones_nor_pagers
  (E : ℝ) -- total number of employees (E must be positive)
  (h1 : 0 < E)
  (frac_cell_phones : ℝ)
  (H1 : frac_cell_phones = (2 / 3))
  (frac_pagers : ℝ)
  (H2 : frac_pagers = (2 / 5))
  (frac_both : ℝ)
  (H3 : frac_both = 0.4) :
  (1 / 3) = (1 - frac_cell_phones - frac_pagers + frac_both) :=
by
  -- setup definitions, conditions and final proof
  sorry

end fraction_neither_cell_phones_nor_pagers_l421_421976


namespace number_of_C_atoms_l421_421498

theorem number_of_C_atoms (n C H O : ℕ) (C_weight H_weight O_weight molecular_weight : ℝ)
  (h1 : H = 8) (h2 : O = 7) (h3 : molecular_weight = 192)
  (h4 : C_weight = 12.01) (h5 : H_weight = 1.01) (h6 : O_weight = 16.00) :
  ((molecular_weight - (H * H_weight + O * O_weight)) / C_weight).round = 6 :=
by
  sorry

end number_of_C_atoms_l421_421498


namespace number_of_prime_divisors_1421_l421_421292

-- Conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def prime_divisors (n : ℕ) : List ℕ :=
List.filter is_prime (List.range (n + 1))

-- Statement
theorem number_of_prime_divisors_1421 : List.length (prime_divisors 1421) = 3 := 
sorry

end number_of_prime_divisors_1421_l421_421292


namespace probability_two_even_is_1_div_6_l421_421585

-- Define the problem conditions
def numbers : List ℕ := [1, 2, 3, 4]
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define what it means to draw two numbers sequentially without replacement and find probability
def probability_two_even : ℚ := 
  let all_pairs := numbers.product numbers
  let valid_pairs := all_pairs.filter (λ p, p.fst ≠ p.snd ∧ is_even p.fst ∧ is_even p.snd)
  valid_pairs.length / all_pairs.filter (λ p, p.fst ≠ p.snd).length

-- The theorem stating the probability is 1/6
theorem probability_two_even_is_1_div_6 : probability_two_even = 1 / 6 := 
by
  sorry

end probability_two_even_is_1_div_6_l421_421585


namespace unique_isometry_and_minimal_distance_points_l421_421502

variable {Point : Type}
variable [MetricSpace Point]
variables {A B C D E : Point} {hexahedron : Surface}

-- Defining the structure of the hexahedron composed of two regular tetrahedra
def is_regular_tetrahedron {A B C D : Point} : Prop := sorry
def composed_of_congruent_tetrahedra (hexahedron : Surface) : Prop :=
  is_regular_tetrahedron ∧ hexahedron = {A, B, C, D, E}

-- Define the isometry Z
def isometry_Z (Z : Point → Point) :=
  Z A = B ∧ Z B = C ∧ Z C = A ∧ Z D = E ∧ Z E = D ∧ 
  (∀ (X Y : Point), dist X Y = dist (Z X) (Z Y))

-- Define the points on the hexahedron whose distance from their image under Z is minimal
def minimal_distance_points (Z : Point → Point) (X : Point) : Prop :=
  ∀ (Y : Point) (hY : Y ∈ hexahedron), dist X (Z X) ≤ dist Y (Z Y)

theorem unique_isometry_and_minimal_distance_points :
  (∃! Z : Point → Point, isometry_Z Z) ∧ 
  (∀ (Z : Point → Point) (hZ : isometry_Z Z) (X : Point), X ∈ hexahedron → minimal_distance_points Z X) := sorry

end unique_isometry_and_minimal_distance_points_l421_421502


namespace find_equation_of_tangent_line_l421_421300

def is_tangent_at_point (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := 
  ∃ x y, (x - 1)^2 + (y + 2)^2 = 1 ∧ l x₀ y₀ ∧ l x y

def equation_of_line (l : ℝ → ℝ → Prop) := 
  ∀ x y, l x y ↔ (x = 2 ∨ 12 * x - 5 * y - 9 = 0)

theorem find_equation_of_tangent_line : 
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ (x - 1)^2 + (y + 2)^2 ≠ 1 ∧ (x, y) = (2,3))
  → is_tangent_at_point l 2 3
  → equation_of_line l := 
sorry

end find_equation_of_tangent_line_l421_421300


namespace count_perfect_powers_lt_1000_l421_421291

/--
There are 42 positive integers less than 1000 that are either a perfect square, a perfect cube, or a perfect fourth power.
-/
theorem count_perfect_powers_lt_1000 : 
  let perfect_square (n : ℕ) := ∃ k : ℕ, k^2 = n
  let perfect_cube (n : ℕ) := ∃ k : ℕ, k^3 = n
  let perfect_fourth_power (n : ℕ) := ∃ k : ℕ, k^4 = n
  {n : ℕ | n < 1000 ∧ (perfect_square n ∨ perfect_cube n ∨ perfect_fourth_power n)}.to_finset.card = 42 :=
by
  sorry

end count_perfect_powers_lt_1000_l421_421291


namespace remainder_of_sum_of_powers_l421_421465

theorem remainder_of_sum_of_powers :
  (finset.range 1000).sum (λ n, 5^n) % 7 = 2 := 
sorry

end remainder_of_sum_of_powers_l421_421465


namespace first_group_number_l421_421099

theorem first_group_number (x : ℕ) (h1 : x + 120 = 126) : x = 6 :=
by
  sorry

end first_group_number_l421_421099


namespace max_k_for_rooks_knights_l421_421003

theorem max_k_for_rooks_knights (k : ℕ) : 
  (∃ (positions : fin 8 × fin 8 → bool), 
    (∀ (i j : fin 8), i ≠ j → ¬(positions i.1 j.1 = true ∧ positions i.2 j.2 = true)) ∧ 
    (∀ (i j : fin 8), (positions i j = true → 
      ¬∀ dx dy, (dx,dy) ∈ {(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)} → 
      positions (i + dx) (j + dy) = true))
  ) → k ≤ 5 :=
sorry

end max_k_for_rooks_knights_l421_421003


namespace words_with_B_at_least_once_l421_421636

theorem words_with_B_at_least_once :
  ∃ (s : Finset (String)), s.card = 369 ∧
  ∀ w ∈ s, w.length = 4 ∧ w.toList.all (λ c, c ∈ ['A', 'B', 'C', 'D', 'E']) ∧
    ('B' ∈ w.toList) :=
by { sorry }

end words_with_B_at_least_once_l421_421636


namespace avg_marks_class_20_students_l421_421312

theorem avg_marks_class_20_students  :
  ∃ (x : ℝ), 
    (let total_marks := 20 * x,
         total_marks_with_50 := 50 * 60,
         combined_total := 70 * 54.285714285714285 in
     total_marks + total_marks_with_50 = combined_total) 
     ∧ x = 40 :=
sorry

end avg_marks_class_20_students_l421_421312


namespace area_of_circular_ring_l421_421800

open Real

variables (n : ℕ) (t : ℝ)

noncomputable def area_ring (n : ℕ) (t : ℝ) : ℝ :=
  (π * t * tan (pi / n)) / n

theorem area_of_circular_ring (n : ℕ) (t : ℝ) :
  ∀ (R r : ℝ), R = r / cos (π / n) → 
  area_ring n t = (π * t * tan (π / n)) / n :=
by
  intros R r h
  sorry

end area_of_circular_ring_l421_421800


namespace coupon_savings_difference_l421_421959

-- Definitions based on conditions
def P (p : ℝ) := 120 + p
def savings_coupon_A (p : ℝ) := 24 + 0.20 * p
def savings_coupon_B := 35
def savings_coupon_C (p : ℝ) := 0.30 * p

-- Conditions
def condition_A_saves_at_least_B (p : ℝ) := savings_coupon_A p ≥ savings_coupon_B
def condition_A_saves_at_least_C (p : ℝ) := savings_coupon_A p ≥ savings_coupon_C p

-- Proof problem
theorem coupon_savings_difference :
  ∀ (p : ℝ), 55 ≤ p ∧ p ≤ 240 → (P 240 - P 55) = 185 :=
by
  sorry

end coupon_savings_difference_l421_421959


namespace james_total_expenditure_l421_421148

theorem james_total_expenditure :
  let entry_fee := 30
  let drinks_for_friends := 3 * 10
  let drinks_for_himself := 8
  let total_drinks := drinks_for_friends + drinks_for_himself
  let cocktails := 7
  let non_alcoholic_drinks := total_drinks - cocktails
  let cocktail_cost := 10
  let non_alcoholic_cost := 5
  let cocktail_cost_initial := cocktails * cocktail_cost
  let cocktail_discount := 0.20 * cocktail_cost_initial
  let cocktail_cost_final := cocktail_cost_initial - cocktail_discount
  let non_alcoholic_cost_total := non_alcoholic_drinks * non_alcoholic_cost
  let food_gourmet_burger := 20
  let food_fries := 8
  let food_cost := food_gourmet_burger + food_fries
  let food_tip := 0.20 * food_cost
  let drinks_tip := 0.15 * (cocktail_cost_initial + non_alcoholic_cost_total)
  let total_cost := entry_fee + cocktail_cost_final + non_alcoholic_cost_total + food_cost + food_tip + drinks_tip
  total_cost = 308.35 :=
by 
  let entry_fee := 30
  let drinks_for_friends := 3 * 10
  let drinks_for_himself := 8
  let total_drinks := drinks_for_friends + drinks_for_himself
  let cocktails := 7
  let non_alcoholic_drinks := total_drinks - cocktails
  let cocktail_cost := 10
  let non_alcoholic_cost := 5
  let cocktail_cost_initial := cocktails * cocktail_cost
  let cocktail_discount := 0.20 * cocktail_cost_initial
  let cocktail_cost_final := cocktail_cost_initial - cocktail_discount
  let non_alcoholic_cost_total := non_alcoholic_drinks * non_alcoholic_cost
  let food_gourmet_burger := 20
  let food_fries := 8
  let food_cost := food_gourmet_burger + food_fries
  let food_tip := 0.20 * food_cost
  let drinks_tip := 0.15 * (cocktail_cost_initial + non_alcoholic_cost_total)
  let total_cost := entry_fee + cocktail_cost_final + non_alcoholic_cost_total + food_cost + food_tip + drinks_tip
  exact sorry

end james_total_expenditure_l421_421148


namespace probability_X_lt_0_l421_421749

open ProbabilityTheory MeasureTheory

-- Given conditions
variables (X : ℝ → ℝ) (σ : ℝ)
  [NormalDistribution X 2 (σ^2)]
  (h : (probability {x | 0 < X x ∧ X x < 4}) = 0.3)

-- The statement to prove
theorem probability_X_lt_0 : (probability {x | X x < 0}) = 0.35 :=
sorry

end probability_X_lt_0_l421_421749


namespace neither_cable_nor_vcr_fraction_l421_421921

variable (T : ℕ) -- Let T be the total number of housing units

def cableTV_fraction : ℚ := 1 / 5
def VCR_fraction : ℚ := 1 / 10
def both_fraction_given_cable : ℚ := 1 / 4

theorem neither_cable_nor_vcr_fraction : 
  (T : ℚ) * (1 - ((1 / 5) + ((1 / 10) - ((1 / 4) * (1 / 5))))) = (T : ℚ) * (3 / 4) :=
by sorry

end neither_cable_nor_vcr_fraction_l421_421921


namespace least_possible_cost_l421_421777

theorem least_possible_cost : 
  let R := [12, 6, 15, 20, 21]  -- areas of the regions
  let C := [3, 2.50, 2, 1.50, 1] -- costs per square foot
  (∑ i in [0, 1, 2, 3, 4], R[i] * C[i]) = 108 := by 
    sorry

end least_possible_cost_l421_421777


namespace mass_of_substance_l421_421061

/- Define the conditions -/

/-- The mass (in kilograms) of 1 cubic meter of the substance -/
def mass_per_cubic_meter : ℝ := 500

/-- The volume (in cubic meters) of the given mass of the substance -/
def volume_cm_cubed : ℝ := 2 * 10^-6

/- Theorem statement -/

/-- The mass of the substance for the given volume of 2 cm³ is 1 * 10^-3 kg -/
theorem mass_of_substance :
  (mass_per_cubic_meter * volume_cm_cubed) = 1 * 10^-3 :=
sorry

end mass_of_substance_l421_421061


namespace ellipses_intersect_at_most_two_points_l421_421876

-- Definitions of ellipses and their properties
structure Ellipse where
  F : Point
  F1 : Point
  major_axis_length : ℝ

-- Definition of the problem conditions
variable (E1 E2 : Ellipse) (P1 P2 P3 : Point)

-- The key property of ellipses
def is_on_ellipse (P : Point) (E : Ellipse) : Prop :=
  (dist P E.F + dist P E.F1) = E.major_axis_length

-- The theorem to prove
theorem ellipses_intersect_at_most_two_points (h1 : E1.F = E2.F)
    (h2 : is_on_ellipse P1 E1) (h3 : is_on_ellipse P2 E1) (h4 : is_on_ellipse P3 E1)
    (h5 : is_on_ellipse P1 E2) (h6 : is_on_ellipse P2 E2) (h7 : is_on_ellipse P3 E2) :
    False :=
sorry

end ellipses_intersect_at_most_two_points_l421_421876


namespace min_product_of_digits_1_2_3_4_l421_421014

theorem min_product_of_digits_1_2_3_4 : 
  let nums := (1, 2, 3, 4) in
  ∃ (a b c d: ℕ), 
    (a ∈ nums) ∧ (b ∈ nums) ∧ (c ∈ nums) ∧ (d ∈ nums) ∧ 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
    (b ≠ c) ∧ (b ≠ d) ∧
    (c ≠ d) ∧
    (a * 10 + b) * (c * 10 + d) = 312 ∧
    ((a * 10 + b) = 13 ∧ (c * 10 + d) = 24 ∨ 
     (a * 10 + b) = 24 ∧ (c * 10 + d) = 13) :=
  sorry

end min_product_of_digits_1_2_3_4_l421_421014


namespace probability_BD_greater_than_6_correct_l421_421873

structure Triangle (α : Type _) :=
(A B C : α)

structure IsRightTriangle {α : Type _} [EuclideanGeometry α] (T : Triangle α) :=
(right_angle_C : angle T.A T.C T.B = π/2)
(angle_B_45 : angle T.A T.B T.C = π/4)
(hypotenuse_AB_12 : distance T.A T.B = 12)

axiom exists_point_in_triangle {α : Type _} [EuclideanGeometry α] (T : Triangle α) : ∃ P : α, in_triangle P T

noncomputable def probability_BD_greater_than_6 {α : Type _} [EuclideanGeometry α] (T : Triangle α) (hT : IsRightTriangle T) : ℝ :=
  let P := classical.some (exists_point_in_triangle T)
  let D := intersection_point_extending_BP_meeting_AC T P  -- hypothetical function
  if distance T.B D > 6 then 1 else 0  -- this is simplified and needs actual probability computation
  
theorem probability_BD_greater_than_6_correct {α : Type _} [EuclideanGeometry α] (T : Triangle α) (hT : IsRightTriangle T) :
  probability_BD_greater_than_6 T hT = (2 - Real.sqrt 2) / 2 := 
sorry

end probability_BD_greater_than_6_correct_l421_421873


namespace scientific_notation_110_billion_l421_421696

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ 110 * 10^8 = a * 10^n

theorem scientific_notation_110_billion :
  ∃ (a : ℝ) (n : ℤ), scientific_notation_form a n ∧ a = 1.1 ∧ n = 10 :=
by
  sorry

end scientific_notation_110_billion_l421_421696


namespace length_of_platform_l421_421165

theorem length_of_platform (l t p : ℝ) (h1 : (l / t) = (l + p) / (5 * t)) : p = 4 * l :=
by
  sorry

end length_of_platform_l421_421165


namespace midpoint_of_points_l421_421570

theorem midpoint_of_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 10) (h3 : x2 = 8) (h4 : y2 = 4) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 7) := 
by
  rw [h1, h2, h3, h4]
  norm_num

end midpoint_of_points_l421_421570


namespace neg_p_l421_421284

-- Define the initial proposition p
def p : Prop := ∀ (m : ℝ), m ≥ 0 → 4^m ≥ 4 * m

-- State the theorem to prove the negation of p
theorem neg_p : ¬p ↔ ∃ (m_0 : ℝ), m_0 ≥ 0 ∧ 4^m_0 < 4 * m_0 :=
by
  sorry

end neg_p_l421_421284


namespace find_phi_l421_421279

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + π / 4)

theorem find_phi (ω φ : ℝ) (hω : 0 < ω) (hφ : |φ| < π / 2)
  (h_shift : ∀ x : ℝ, f ω φ (x + π / 3) = g x) : φ = π / 12 :=
by sorry

end find_phi_l421_421279


namespace part1_part2_part3_part4_l421_421370

open Nat

section balls_and_boxes

variables (n : ℕ) (m : ℕ) (balls boxes : Finset ℕ) (h₁ : n = 4) (h₂ : m = 4) (h₃ : balls.card = 4) (h₄ : boxes.card = 4)

/-- There are 256 ways to place 4 balls into 4 boxes. -/
theorem part1 : (boxes.card ^ balls.card) = 256 := 
by
  sorry

/-- With each box having exactly one ball, there are 24 ways to place the balls. -/
theorem part2 : (Fintype.card (ball_equiv_perms balls boxes)) = 24 := 
by
  sorry

/-- If exactly one box is empty, there are 144 ways to place the balls. -/
theorem part3 : (choose 4 2) * (factorial 3) = 144 := 
by
  sorry

/-- If the balls are identical and exactly one box is empty, there are 12 ways to place the balls. -/
theorem part4 : (choose 4 1) * (choose 3 1) = 12 := 
by
  sorry

end balls_and_boxes

/- helper definitions for theorems -/

def ball_equiv_perms : Equiv.Perm (Fin 4) :=
{ to_fun := λ x, x.succ,
  inv_fun := λ x, x.pred,
  left_inv := λ x, by { simp [Fin.succ_pred] },
  right_inv := λ x, by { simp [Fin.pred_succ] }
}

end part1_part2_part3_part4_l421_421370


namespace problem_statement_l421_421229

def f : ℝ → ℝ :=
  λ x, if x > 0 then Real.log x / Real.log 3 else (1 / 2) ^ x + 1

theorem problem_statement (h₀ : f 0 = 2) (h₁ : f (-1) = 3) : f (f (-3)) = 2 :=
by
  have hf0 : ∀ x, x > 0 → f x = Real.log x / Real.log 3 := sorry
  have hle0 : ∀ x, x ≤ 0 → f x = (1 / 2) ^ x + 1 := sorry
  have h2 : (1 / 2) ^ 0 + 1 = 2 := by norm_num
  have h4 : (1 / 2) ^ (-1) + 1 = 3 := by norm_num
  sorry

end problem_statement_l421_421229


namespace solve_abs_eq_l421_421899

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421899


namespace rook_tour_possible_iff_l421_421692

-- Defining the conditions for the problem
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Theorem statement
theorem rook_tour_possible_iff (n : ℕ) :
  (∃ path : list (ℕ × ℕ), 
    -- Ensure the path length is exactly n^2
    path.length = n^2 ∧ 
    -- Ensure the rook starts from the top-left corner (1, 1)
    path.head = some (1, 1) ∧ 
    -- Ensure the rook ends at the top-left corner (1, 1)
    path.last = some (1, 1) ∧ 
    -- Ensure every square on the chessboard is visited exactly once
    (∀ x, x ∈ path) = ∀ x' y', 1 ≤ x' ∧ x' ≤ n ∧ 1 ≤ y' ∧ y' ≤ n → (x', y') ∈ path ∧ 
    -- Ensure alternating between horizontal and vertical moves
    (∀ i < path.length - 1, 
      (fst path.nth_le i = fst path.nth_le (i+1) ∧ abs (snd path.nth_le (i+1) - snd path.nth_le i) = 1) ∨ 
      (snd path.nth_le i = snd path.nth_le (i+1) ∧ abs (fst path.nth_le (i+1) - fst path.nth_le i) = 1))) ↔ is_even n :=
sorry

end rook_tour_possible_iff_l421_421692


namespace area_of_region_eq_24π_l421_421885

theorem area_of_region_eq_24π :
  (∃ R, R > 0 ∧ ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 8 * x + 18 * y + 73 = R ^ 2) →
  ∃ π : ℝ, π > 0 ∧ area = 24 * π :=
by
  sorry

end area_of_region_eq_24π_l421_421885


namespace possible_measures_A_l421_421823

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l421_421823


namespace tissue_properties_l421_421119

noncomputable def actual_diameter (magnified_diameter magnification_factor : ℝ) : ℝ :=
  magnified_diameter / magnification_factor

noncomputable def radius (diameter : ℝ) : ℝ :=
  diameter / 2

noncomputable def area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

noncomputable def circumference (radius : ℝ) : ℝ :=
  2 * Real.pi * radius

theorem tissue_properties :
  ∀ (magnified_diameter magnification_factor : ℝ),
  magnified_diameter = 5 →
  magnification_factor = 1000 →
  actual_diameter magnified_diameter magnification_factor = 0.005 ∧
  area (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.000019635 ∧
  circumference (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.01570795 :=
by
  intros magnified_diameter magnification_factor h1 h2
  have h_diameter : actual_diameter magnified_diameter magnification_factor = 0.005
  {
    rw [h1, h2]
    norm_num
  }
  have h_radius : radius (actual_diameter magnified_diameter magnification_factor) = 0.0025
  {
    rw [h_diameter]
    norm_num
  }
  have h_area : area (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.000019635
  {
    simp [h_radius, area]
    norm_num
    simp [Real.pi]
  }
  have h_circumference : circumference (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.01570795
  {
    simp [h_radius, circumference]
    norm_num
    simp [Real.pi]
  }
  exact ⟨h_diameter, h_area, h_circumference⟩

end tissue_properties_l421_421119


namespace mike_total_spent_l421_421758

def cost_of_rose_bushes := 75
def num_rose_bushes_total := 6
def num_rose_bushes_friend := 2
def num_rose_bushes_self := num_rose_bushes_total - num_rose_bushes_friend
def tax_rate_rose_bushes := 0.05
def cost_of_rose_bushes_self := num_rose_bushes_self * cost_of_rose_bushes
def tax_rose_bushes_self := cost_of_rose_bushes_self * tax_rate_rose_bushes

def cost_of_tiger_tooth_aloe := 100
def num_tiger_tooth_aloes := 2
def tax_rate_tiger_tooth_aloes := 0.07
def cost_of_tiger_tooth_aloes := num_tiger_tooth_aloes * cost_of_tiger_tooth_aloe
def tax_tiger_tooth_aloes := cost_of_tiger_tooth_aloes * tax_rate_tiger_tooth_aloes

def total_cost_self := cost_of_rose_bushes_self + tax_rose_bushes_self + cost_of_tiger_tooth_aloes + tax_tiger_tooth_aloes

theorem mike_total_spent : total_cost_self = 529 := by
  sorry

end mike_total_spent_l421_421758


namespace percentage_increase_correct_l421_421713

def bookstore_earnings : ℕ := 60
def tutoring_earnings : ℕ := 40
def new_bookstore_earnings : ℕ := 100
def additional_tutoring_fee : ℕ := 15
def old_total_earnings : ℕ := bookstore_earnings + tutoring_earnings
def new_total_earnings : ℕ := new_bookstore_earnings + (tutoring_earnings + additional_tutoring_fee)
def overall_percentage_increase : ℚ := (((new_total_earnings - old_total_earnings : ℚ) / old_total_earnings) * 100)

theorem percentage_increase_correct :
  overall_percentage_increase = 55 := sorry

end percentage_increase_correct_l421_421713


namespace problem_l421_421432

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : odd_function f
axiom f_property : ∀ x : ℝ, f (x + 2) = -f x
axiom f_at_1 : f 1 = 8

theorem problem : f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end problem_l421_421432


namespace find_value_of_expression_l421_421653

theorem find_value_of_expression (x y z : ℚ)
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y = 10) : (x + y - z) / 3 = -4 / 9 :=
by sorry

end find_value_of_expression_l421_421653


namespace robie_initial_cards_l421_421380

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l421_421380


namespace no_real_solution_l421_421549

noncomputable def quadratic_eq (x : ℝ) : ℝ := (2*x^2 - 3*x + 5)

theorem no_real_solution : 
  ∀ x : ℝ, quadratic_eq x ^ 2 + 1 ≠ 1 :=
by
  intro x
  sorry

end no_real_solution_l421_421549


namespace sum_of_angles_l421_421067

-- Definitions of points and triangles
variables (point : Type) [has_zero point] [has_one point]
variables (M D C A K : point)
variables (angle : Type) [add_comm_group angle] [has_of_nat angle] [linear_ordered_field angle]

-- Conditions
axiom common_right_angle : ∀ (p q r : point), triangle p q r → right_angle (angle_at q)
axiom ratio_AD_CD : AD / CD = 2 / 5
axiom ratio_K_on_CD : ∃ (k : ℝ), K = C + k * (D - C) ∧ 2/5 = k
axiom midpoint_A_D : M = (A + D) / 2

-- The statement to prove using conditions
theorem sum_of_angles : ∠ (angle A K D) + ∠ (angle M C D) = 45 :=
by
  sorry

end sum_of_angles_l421_421067


namespace determine_c_l421_421070

-- Definitions of the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- Hypothesis for the sequence to be geometric
def geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∃ c, ∀ n, a (n + 1) + c = r * (a n + c)

-- The goal to prove
theorem determine_c (a : ℕ → ℕ) (c : ℕ) (r := 2) :
  seq a →
  geometric_seq a c →
  c = 1 :=
by
  intros h_seq h_geo
  sorry

end determine_c_l421_421070


namespace correct_props_l421_421731

def line := ℕ
def plane := ℕ

variables (m n : line) (α β γ : plane)

-- Conditions
axiom diff_lines : m ≠ n
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions
def prop1 (m_perp_alpha : Prop) (n_par_alpha : Prop) : Prop :=
  m_perp_alpha → n_par_alpha → ⊥

def prop2 (alpha_perp_gamma : Prop) (beta_perp_gamma : Prop) : Prop :=
  alpha_perp_gamma → beta_perp_gamma → ⊥

def prop3 (alpha_par_beta : Prop) (beta_par_gamma : Prop) (m_perp_alpha : Prop) : Prop :=
  alpha_par_beta → beta_par_gamma → m_perp_alpha → ⊥

def prop4 (alpha_cap_gamma_eq_m : Prop) (beta_cap_gamma_eq_n : Prop) (m_par_n : Prop) : Prop :=
  alpha_cap_gamma_eq_m → beta_cap_gamma_eq_n → m_par_n → ⊥

-- Main theorem
theorem correct_props : prop1 (m ⊥ α) (n ∥ α) ∧
                        ¬ prop2 (α ⊥ γ) (β ⊥ γ) ∧
                        prop3 (α ∥ β) (β ∥ γ) (m ⊥ α) ∧
                        ¬ prop4 (α ∩ γ = m) (β ∩ γ = n) (m ∥ n)
:= sorry

end correct_props_l421_421731


namespace trip_length_is_440_l421_421778

noncomputable def total_trip_length (d : ℝ) : Prop :=
  55 * 0.02 * (d - 40) = d

theorem trip_length_is_440 :
  total_trip_length 440 :=
by
  sorry

end trip_length_is_440_l421_421778


namespace subset_condition_necessary_but_not_sufficient_l421_421602

open Set

variable {A B : Set ℕ}
variable {a : ℕ}

theorem subset_condition_necessary_but_not_sufficient :
  (A = {1, a}) → (B = {1, 2, 3}) → (A ⊆ B) ↔ (a = 3 ∨ a = 2) ∧ a = 3 ∨ a = 2 ∧ ¬(A ⊆ B)  :=
by
  intros hA hB hAB
  split
  { sorry } -- Proof required here
  { sorry } -- Proof required here

end subset_condition_necessary_but_not_sufficient_l421_421602


namespace number_of_girls_l421_421860

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l421_421860


namespace tom_strokes_over_par_l421_421090

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l421_421090


namespace negation_false_l421_421436

theorem negation_false (a b : ℝ) : ¬ ((a ≤ 1 ∨ b ≤ 1) → a + b ≤ 2) :=
sorry

end negation_false_l421_421436


namespace sin_2alpha_eq_one_half_l421_421261

variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < π / 2)
variables (h3 : cos (2 * α) = cos (π / 4 - α))

theorem sin_2alpha_eq_one_half : sin (2 * α) = 1 / 2 :=
by sorry

end sin_2alpha_eq_one_half_l421_421261


namespace g_properties_l421_421453

def f (x : ℝ) : ℝ := cos (π + x) * (cos x - 2 * sin x) + sin x ^ 2

def g (x : ℝ) : ℝ := f (x + π / 8)

theorem g_properties :
  (∀ x, g(x) = sqrt 2 * sin (2 * x)) ∧ 
  (∀ x ∈ Ioo (0 : ℝ) (π / 4), strict_mono_on g x ∧ odd g) := 
sorry

end g_properties_l421_421453


namespace point_O_dot_sum_vectors_l421_421017

-- Define the points and the lengths of the sides
variables (A B C O : Type) [affine_space ℝ ℝ]
noncomputable def length_AB : ℝ := 6
noncomputable def length_AC : ℝ := 2

-- Define vectors for A, B, C, and circumcenter O
variables (OA OB OC : ℝ → Affine ℝ ℝ)
variables {a b c o : ℝ}

@[vector_space ℝ ℝ]
noncomputable def dot_product (x y : ℝ) := x * y

-- State the theorem
theorem point_O_dot_sum_vectors (h1 : O = circumcenter A B C) 
                                (h2 : length_AB AB = 6) 
                                (h3 : length_AC AC = 2) :
  dot_product (OA o) ((OB b) + (OC c)) = 20 :=
begin
  -- placeholder for proof
  sorry, 
end

end point_O_dot_sum_vectors_l421_421017


namespace area_enclosed_by_line_and_curve_l421_421547

theorem area_enclosed_by_line_and_curve :
  ∫ x in -2..1, (4 - 2 * x^2 - 2 * x) = 9 :=
by
  -- Proof steps omitted
  sorry

end area_enclosed_by_line_and_curve_l421_421547


namespace collinear_points_XYZ_l421_421935

open EuclideanGeometry

theorem collinear_points_XYZ
  (O A B C Z X Y : Point)
  (hOA : Collinear O A)
  (hOB : Collinear O B)
  (hOC : Collinear O C)
  (hZ : ∃ (circle1 circle2 : Circle), circle1.Diameter = lineSegment O A ∧ circle2.Diameter = lineSegment O B ∧ circle1 ∩ circle2 = {Z})
  (hX : ∃ (circle3 circle4 : Circle), circle3.Diameter = lineSegment O B ∧ circle4.Diameter = lineSegment O C ∧ circle3 ∩ circle4 = {X})
  (hY : ∃ (circle5 circle6 : Circle), circle5.Diameter = lineSegment O C ∧ circle6.Diameter = lineSegment O A ∧ circle5 ∩ circle6 = {Y}) :
  Collinear X Y Z :=
sorry

end collinear_points_XYZ_l421_421935


namespace decimal_150th_place_of_5_over_11_l421_421117

theorem decimal_150th_place_of_5_over_11 :
  let r := "45" in  -- The repeating decimal part
  let n := 150 in   -- The 150th place to find
  let repeat_len := 2 in -- Length of the repeating cycle
  cycle_digit r (n % repeat_len) = '5' := 
by
  sorry

/-- Helper function to get the nth digit of a repeating decimal cycle -/
def cycle_digit (cycle: String) (n: Nat) : Char := 
  cycle.get (n % cycle.length)

end decimal_150th_place_of_5_over_11_l421_421117


namespace f_one_over_f_two_eq_one_over_sixteen_f_x_eq_three_implies_x_eq_sqrt_three_l421_421273

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2 * x

theorem f_one_over_f_two_eq_one_over_sixteen : f (1 / f 2) = 1 / 16 := by
  sorry

theorem f_x_eq_three_implies_x_eq_sqrt_three (x : ℝ) : f x = 3 → x = Real.sqrt 3 := by
  sorry

end f_one_over_f_two_eq_one_over_sixteen_f_x_eq_three_implies_x_eq_sqrt_three_l421_421273


namespace alpha_proportional_l421_421037

theorem alpha_proportional (alpha beta gamma : ℝ) (h1 : ∀ β γ, (β = 15 ∧ γ = 3) → α = 5)
    (h2 : beta = 30) (h3 : gamma = 6) : alpha = 2.5 :=
sorry

end alpha_proportional_l421_421037


namespace find_n_l421_421672

/-- Define a structure for an arithmetic sequence -/
structure ArithSeq where
  a1 : ℝ
  d : ℝ
  n : ℕ

/-- Compute the sum of odd-numbered terms of an arithmetic sequence -/
def sum_odd (seq : ArithSeq) : ℝ :=
  let sum_odd_n := (seq.a1 + seq.a1 + 2*seq.d*seq.n) * (seq.n + 1) / 2
  sum_odd_n

/-- Compute the sum of even-numbered terms of an arithmetic sequence -/
def sum_even (seq : ArithSeq) : ℝ :=
  let sum_even_n := (seq.a1 + seq.d + seq.a1 + (2*seq.n - 1)*seq.d) * seq.n / 2
  sum_even_n

/-- Define the conditions provided -/
def conditions (seq : ArithSeq) : Prop :=
  sum_odd seq = 165 ∧ sum_even seq = 150

/-- Main theorem stating that under given conditions, value of n is 10 -/
theorem find_n (seq : ArithSeq) (h : conditions seq) : seq.n = 10 :=
  sorry

end find_n_l421_421672


namespace number_of_girls_l421_421848

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l421_421848


namespace unique_triple_solution_l421_421546

/--
The only triples \((p, q, n)\) that satisfy the conditions 
\( q^{n+2} \equiv 3^{n+2} \pmod{p^n} \) and \( p^{n+2} \equiv 3^{n+2} \pmod{q^n} \)
when \( p \) and \( q \) are odd prime numbers, and \( n \) is an integer greater than 1,
are \((3, 3, n)\) for \( n = 2, 3, \ldots \).
-/
theorem unique_triple_solution (p q : ℕ) (n : ℕ) : 
  prime p ∧ prime q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ n > 1 →
  (q^(n+2) % (p^n) = 3^(n+2) % (p^n) ∧ p^(n+2) % (q^n) = 3^(n+2) % (q^n)) ↔ 
  (p = 3 ∧ q = 3 ∧ ∃ k, n = k ∧ k > 1) :=
by
  sorry

end unique_triple_solution_l421_421546


namespace rounding_29_6_to_30_l421_421138

theorem rounding_29_6_to_30 : round 29.6 = 30 := 
sorry

end rounding_29_6_to_30_l421_421138


namespace find_x_l421_421894

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421894


namespace circumference_to_diameter_ratio_l421_421158

-- Definitions from the conditions
def r : ℝ := 15
def C : ℝ := 90
def D : ℝ := 2 * r

-- The proof goal
theorem circumference_to_diameter_ratio : C / D = 3 := 
by sorry

end circumference_to_diameter_ratio_l421_421158


namespace find_expression_for_f_check_m_n_range_l421_421246

def a : ℝ := -1/2
def b : ℝ := 1
def f (x : ℝ) : ℝ := a * x^2 + b * x
def f2 : Prop := f 2 = 0
def f_eq_x_two_roots : Prop := ∀ x : ℝ, x^2 + (b - 1) * x = 0 → x = 1 -- ensures two equal roots

-- Find the expression for f(x)
theorem find_expression_for_f : f = λ x, -1/2 * x^2 + x :=
by
  sorry

-- Check if there exist m, n such that range of f(x) is [2m, 2n]
theorem check_m_n_range :
  ∃ (m n : ℝ), m < n ∧ (∀ x ∈ set.Icc m n, f x ∈ set.Icc (2 * m) (2 * n)) ∧ m = -2 ∧ n = 0 :=
by
  sorry

end find_expression_for_f_check_m_n_range_l421_421246


namespace total_cards_beginning_l421_421378

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l421_421378


namespace compare_pow_value_l421_421474

theorem compare_pow_value : 
  ∀ (x : ℝ) (n : ℕ), x = 0.01 → n = 1000 → (1 + x)^n > 1000 := 
by 
  intros x n hx hn
  rw [hx, hn]
  sorry

end compare_pow_value_l421_421474


namespace solution_to_quadratic_inequality_l421_421281

theorem solution_to_quadratic_inequality 
  (a : ℝ)
  (h : ∀ x : ℝ, x^2 - a * x + 1 < 0 ↔ (1 / 2 : ℝ) < x ∧ x < 2) :
  a = 5 / 2 :=
sorry

end solution_to_quadratic_inequality_l421_421281


namespace triangle_angle_eq_pi_over_3_l421_421309

theorem triangle_angle_eq_pi_over_3
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = a * b)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ C : ℝ, C = 2 * Real.pi / 3 ∧ 
            Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  -- Proof goes here
  sorry

end triangle_angle_eq_pi_over_3_l421_421309


namespace distinct_seatings_equiv_rotations_l421_421759

-- Define the number of people and the number of seats
def num_people : ℕ := 9
def num_seats : ℕ := 8

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the factorial function
def factorial (n : ℕ) : ℕ := nat.factorial n

-- Define the proof statement
theorem distinct_seatings_equiv_rotations : 
  (binom num_people num_seats) * (factorial (num_seats - 1)) = 45360 :=
by
  -- The proof will be added here
  sorry

end distinct_seatings_equiv_rotations_l421_421759


namespace seating_arrangement_correct_l421_421492

noncomputable def seatingArrangements (committee : Fin 10) : Nat :=
  Nat.factorial 9

theorem seating_arrangement_correct :
  seatingArrangements committee = 362880 :=
by sorry

end seating_arrangement_correct_l421_421492


namespace auntie_em_can_park_l421_421507

theorem auntie_em_can_park (spaces cars : ℕ) (adjacent_spaces : ℕ)
  (h_spaces : spaces = 20) (h_cars : cars = 15) (h_adjacent_spaces : adjacent_spaces = 2) :
  let total_parking_ways := choose spaces cars,
      total_unavailable_ways := choose (spaces - adjacent_spaces + 1) (cars - adjacent_spaces + 1),
      probability_cannot_park := (total_unavailable_ways : ℚ) / total_parking_ways,
      probability_can_park := 1 - probability_cannot_park in
  probability_can_park = 232 / 323 :=
by
  sorry

end auntie_em_can_park_l421_421507


namespace marble_problem_l421_421503

theorem marble_problem
  (total_marbles : ℕ)
  (h_total_marbles : total_marbles = 267)
  (red_percentage white_percentage : ℝ)
  (h_red_percentage : red_percentage = 0.20)
  (h_white_percentage : white_percentage = 0.10)
  (num_red_marbles : ℕ)
  (num_white_marbles : ℕ)
  (h_num_red_marbles : num_red_marbles = (red_percentage * total_marbles).natAbs)
  (h_num_white_marbles : num_white_marbles = (white_percentage * total_marbles).natAbs)
  (num_white_add : ℕ)
  (h_num_white_add : num_white_add = (num_red_marbles / 3).natAbs):
  num_white_marbles + num_white_add = 45 := 
sorry

end marble_problem_l421_421503


namespace exist_point_B_l421_421008

-- Defining the acute angle and the point A
variables {O A : Point} (α : Angle) (h₁ : acute α) (h₂ : A ∈ α.side1)

-- The point B to be constructed
noncomputable def construct_point_B (O A : Point) (α : Angle) (h₁ : acute α) (h₂ : A ∈ α.side1) : Point :=
  let B := -- The point on the same side of the angle
    (λ B, B ∈ α.side1 ∧ (distance B A = distance (foot_of_perpendicular B α.side2) B)) in
  B

-- Statement of the proof problem
theorem exist_point_B (O A : Point) (α : Angle) (h₁ : acute α) (h₂ : A ∈ α.side1) :
  ∃ B, B ∈ α.side1 ∧ distance B A = distance (foot_of_perpendicular B α.side2) B :=
by 
  -- We start with the conditions and the definition of point B
  use construct_point_B O A α h₁ h₂
  sorry

end exist_point_B_l421_421008


namespace number_of_girls_l421_421865

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l421_421865


namespace cakes_sold_l421_421530

/-- If a baker made 54 cakes and has 13 cakes left, then the number of cakes he sold is 41. -/
theorem cakes_sold (original_cakes : ℕ) (cakes_left : ℕ) 
  (h1 : original_cakes = 54) (h2 : cakes_left = 13) : 
  original_cakes - cakes_left = 41 := 
by 
  sorry

end cakes_sold_l421_421530


namespace phenotype_ratios_correct_l421_421050

universe u
variable {α : Type u}

-- Definitions for dominance and independent inheritance
def dominant (a b : α) : Prop := ∃ c : α, a = b ∧ a = c
def independent_inheritance (a b : α) : Prop := ∃ d : α, a = b ∧ d = a

-- Definitions for our traits
def green_cotyledons (Y y : α) := dominant Y y
def brown_cotyledons (y : α)
def round_seeds (R r : α) := dominant R r
def kidney_seeds (r : α)

-- Phenotypic ratios
def phenotype_ratio_1_1_1_1 (a b c d : ℕ) := a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1

-- Definitions for parental combinations
def combination_A (P1 P2 : α) := P1 = ⟨YyRr⟩ ∧ P2 = ⟨YyRr⟩
def combination_B (P1 P2 : α) := P1 = ⟨YyRr⟩ ∧ P2 = ⟨yyrr⟩
def combination_C (P1 P2 : α) := P1 = ⟨Yyrr⟩ ∧ P2 = ⟨Yyrr⟩
def combination_D (P1 P2 : α) := P1 = ⟨Yyrr⟩ ∧ P2 = ⟨yyRr⟩

theorem phenotype_ratios_correct :
  (combination_B (YyRr yyrr) ∨ combination_D (Yyrr yyRr)) →
  phenotype_ratio_1_1_1_1 1 1 1 1 :=
sorry

end phenotype_ratios_correct_l421_421050


namespace round_table_seating_l421_421679

theorem round_table_seating (n : ℕ) (h : n = 10) : 
  ∃ k : ℕ, k = 362880 ∧ (∃! m : ℕ, m * n = nat.factorial n ∧ m = k) :=
by
  sorry

end round_table_seating_l421_421679


namespace find_x_l421_421591

theorem find_x 
  (x : ℝ) 
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : 1 / (Real.sin x) = 1 / (Real.sin (2 * x)) + 1 / (Real.sin (4 * x)) + 1 / (Real.sin (8 * x))) : 
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 :=
by
  sorry

end find_x_l421_421591


namespace sum_f_eq_n_l421_421730

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

noncomputable def f_n (n : ℕ+) (x : ℝ) : ℝ :=
  if n = 1 then f x else f_n (n - 1) (f x)

theorem sum_f_eq_n (n : ℕ+) :
  (∑ i in Finset.range n, f (i + 1)) + (∑ i in Finset.range n, f_n (i + 1) 1) = n :=
by
  sorry

end sum_f_eq_n_l421_421730


namespace referendum_proof_l421_421526

variable (U A B : Finset ℕ)
variable (n_U n_A n_B n_AcBc : ℕ)

theorem referendum_proof 
  (hU : n_U = 250)
  (hA : n_A = 175)
  (hB : n_B = 145)
  (hAcBc : n_AcBc = 35) :
  ((|U| = n_U) ∧ (|A| = n_A) ∧ (|B| = n_B) ∧ (|U \ (A ∪ B)| = n_AcBc)) →
  (|A ∩ B| = n_A + n_B - (n_U - n_AcBc)) :=
by
  sorry

end referendum_proof_l421_421526


namespace find_c_for_radius_of_circle_l421_421221

theorem find_c_for_radius_of_circle :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 6 * y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25 - c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 25 → c = 0) :=
sorry

end find_c_for_radius_of_circle_l421_421221


namespace construct_equilateral_triangles_l421_421183

def Point := ℝ × ℝ

structure Circle where
  center : Point
  radius : ℝ

def dist (p q : Point) : ℝ := 
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

theorem construct_equilateral_triangles (
  S D : Point) (k : Circle)
  (hk : k.center = S ∧ k.radius = dist S D) :
  ∃ A B C, 
    (dist S A = dist A D ∧ dist A D = dist D S ∧ dist D S = dist S A) ∧
    equilateral_triangle D A S ∧
    dist S B = k.radius ∧ dist S C = k.radius ∧
    equilateral_triangle A B C :=
sorry

end construct_equilateral_triangles_l421_421183


namespace rectangle_circle_intersection_l421_421511

-- Define the conditions as assumptions
variables (A B C D E F : Type) [HasDist A B] [HasDist B C] [HasDist D E]

-- Given conditions
def AB : Real := 4
def BC : Real := 5
def DE : Real := 3

-- The length to prove
noncomputable def EF : Real := 7

-- Proof statement
theorem rectangle_circle_intersection : 
  AB = 4 ∧ BC = 5 ∧ DE = 3 → EF = 7 := 
by
  sorry

end rectangle_circle_intersection_l421_421511


namespace square_length_QP_l421_421324

noncomputable def chord_square_length 
  (r1 r2 d: ℝ) (cos_theta: ℝ) (angle: ℝ) 
  (chord_eq: ∀ (x : ℝ), x = chord_eq x) : ℝ :=
  let PQ_sq := r1^2 + r2^2 + 2 * r1 * r2 * cos_theta in
  PQ_sq / 3

theorem square_length_QP : 
  chord_square_length 10 7 15 (-19/35) 120 (λ x, x) = 75 :=
sorry

end square_length_QP_l421_421324


namespace equilateral_triangle_area_l421_421409

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l421_421409


namespace max_true_statements_maximum_true_conditions_l421_421733

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end max_true_statements_maximum_true_conditions_l421_421733


namespace exists_smaller_circle_with_at_least_as_many_lattice_points_l421_421593

theorem exists_smaller_circle_with_at_least_as_many_lattice_points
  (R : ℝ) (hR : 0 < R) :
  ∃ R' : ℝ, (R' < R) ∧ (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 → ∃ (x' y' : ℤ), (x')^2 + (y')^2 ≤ (R')^2) := sorry

end exists_smaller_circle_with_at_least_as_many_lattice_points_l421_421593


namespace solve_for_x_l421_421388

theorem solve_for_x : ∃ x : ℤ, 24 - 4 = 3 + x ∧ x = 17 := 
begin
  use 17,
  split,
  { linarith, },
  { refl, },
end

end solve_for_x_l421_421388


namespace pentagon_perimeter_l421_421890

noncomputable def perimeter_pentagon (FG GH HI IJ : ℝ) (FH FI FJ : ℝ) : ℝ :=
  FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : 
  ∀ (FG GH HI IJ : ℝ), 
  ∀ (FH FI FJ : ℝ),
  FG = 1 → GH = 1 → HI = 1 → IJ = 1 →
  FH^2 = FG^2 + GH^2 → FI^2 = FH^2 + HI^2 → FJ^2 = FI^2 + IJ^2 →
  perimeter_pentagon FG GH HI IJ FJ = 6 :=
by
  intros FG GH HI IJ FH FI FJ
  intros H_FG H_GH H_HI H_IJ
  intros H1 H2 H3
  sorry

end pentagon_perimeter_l421_421890


namespace find_largest_M_l421_421597

noncomputable def largest_M (n : ℕ) (hn : n ≥ 3) : ℝ :=
  n - 1

theorem find_largest_M (n : ℕ) (hn : n ≥ 3) (x : Fin n → ℝ) (hx_pos : ∀ i, 0 < x i) :
  ∃ y : Fin n → ℝ, 
  (∑ i, y i ^ 2 / (y ((i + 1) % n) ^ 2 - y ((i + 1) % n) * y ((i + 2) % n) + y ((i + 2) % n) ^ 2)) ≥ largest_M n hn := 
sorry

end find_largest_M_l421_421597


namespace intersection_parabola_y_axis_l421_421807

theorem intersection_parabola_y_axis :
  let parabola : ℝ → ℝ := λ x, (1/2)*((x - 2)^2) - 1
  in parabola 0 = 1 :=
by {
  sorry
}

end intersection_parabola_y_axis_l421_421807


namespace car_speed_proof_l421_421521

noncomputable def car_speed_in_kmh (rpm : ℕ) (circumference : ℕ) : ℕ :=
  (rpm * circumference * 60) / 1000

theorem car_speed_proof : 
  car_speed_in_kmh 400 1 = 24 := 
by
  sorry

end car_speed_proof_l421_421521


namespace savings_increase_100_percent_l421_421945

variable (I : ℝ) -- income in the first year
variable (S : ℝ) -- savings in the first year
variable (E1 : ℝ) -- expenditure in the first year
variable (I2 : ℝ) -- income in the second year
variable (Etotal : ℝ) -- total expenditure over two years
variable (E2 : ℝ) -- expenditure in the second year
variable (S2 : ℝ) -- savings in the second year

noncomputable def percentage_increase_savings : ℝ :=
  ((S2 - S) / S) * 100

theorem savings_increase_100_percent
  (h1 : S = 0.5 * I)
  (h2 : I2 = 1.5 * I)
  (h3 : Etotal = 2 * E1)
  (h4 : E1 = 0.5 * I)
  (h5 : E2 = Etotal - E1)
  (h6 : S2 = I2 - E2) :
  percentage_increase_savings I S E1 I2 Etotal E2 S2 = 100 := by
  sorry

end savings_increase_100_percent_l421_421945


namespace largest_sum_of_three_faces_l421_421669

theorem largest_sum_of_three_faces (faces : Fin 6 → ℕ)
  (h_unique : ∀ i j, i ≠ j → faces i ≠ faces j)
  (h_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6)
  (h_opposite_sum : ∀ i, faces i + faces (5 - i) = 10) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ faces i + faces j + faces k = 12 :=
by sorry

end largest_sum_of_three_faces_l421_421669


namespace angle_LOM_l421_421360

-- Define the points L and M with their coordinates as per their latitude and longitude.
def L := (0, 45)
def M := (23.5, -90)
def O := (0, 0)  -- Earth's center

-- State the theorem 
theorem angle_LOM (L M O : ℝ × ℝ) (hL : L = (0, 45)) (hM : M = (23.5, -90)) (hO : O = (0, 0)) :
  angle L O M = 135 :=
by
  -- Proof would go here
  sorry

end angle_LOM_l421_421360


namespace exists_abc_sums_eq_l421_421372

def S (n : ℕ) : ℕ :=
  -- Implement the sum of the decimal digits of n
  sorry

theorem exists_abc_sums_eq :
  ∃ (a b c : ℕ), (1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25) ∧ S(a^6 + 2014) = S(b^6 + 2014) ∧ S(a^6 + 2014) = S(c^6 + 2014) :=
by sorry

end exists_abc_sums_eq_l421_421372


namespace turkey_2003_problem_l421_421135

theorem turkey_2003_problem (x m n : ℕ) (hx : 0 < x) (hm : 0 < m) (hn : 0 < n) (h : x^m = 2^(2 * n + 1) + 2^n + 1) :
  x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1 ∨ x = 23 ∧ m = 2 ∧ n = 4 :=
sorry

end turkey_2003_problem_l421_421135


namespace intersection_point_on_square_diagonal_l421_421961

theorem intersection_point_on_square_diagonal (a b c : ℝ) (h : c = (a + b) / 2) :
  (b / 2) = (-a / 2) + c :=
by
  sorry

end intersection_point_on_square_diagonal_l421_421961


namespace find_min_value_l421_421206

theorem find_min_value (a : ℝ) : 
  (a ≤ 2 → ∀ x : ℝ, x ∈ set.Ici 2 → x^2 - 2 * a * x - 1 ≥ 3 - 4 * a) ∧ 
  (a > 2 → ∀ x : ℝ, x ∈ set.Ici 2 → x^2 - 2 * a * x - 1 ≥ -a^2 - 1) :=
by sorry

end find_min_value_l421_421206


namespace number_of_days_l421_421178

def burger_meal_cost : ℕ := 6
def upsize_cost : ℕ := 1
def total_spending : ℕ := 35

/-- The number of days Clinton buys the meal. -/
theorem number_of_days (h1 : burger_meal_cost + upsize_cost = 7) (h2 : total_spending = 35) : total_spending / (burger_meal_cost + upsize_cost) = 5 :=
by
  -- The proof will go here
  sorry

end number_of_days_l421_421178


namespace equilateral_triangle_area_l421_421406

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l421_421406


namespace area_of_triangle_l421_421397

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l421_421397


namespace x_cubed_plus_y_cubed_l421_421651

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end x_cubed_plus_y_cubed_l421_421651


namespace f_f_2pi_l421_421272

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

-- Prove that f(f(2π)) = 1
theorem f_f_2pi : f (f (2 * Real.pi)) = 1 :=
by sorry

end f_f_2pi_l421_421272


namespace hexagon_area_is_correct_l421_421991

-- Define the length of the side of the hexagon
def side_length : ℝ := 3

-- Define the area of an equilateral triangle with given side length
def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

-- Define the number of equilateral triangles in the hexagon
def num_triangles : ℕ := 6

-- Define the total area of the hexagon
def hexagon_area (s : ℝ) : ℝ := num_triangles * equilateral_triangle_area s

-- State the theorem
theorem hexagon_area_is_correct : hexagon_area side_length = (27 * sqrt 3) / 2 :=
by
  sorry

end hexagon_area_is_correct_l421_421991


namespace theater_seat_representation_l421_421315

theorem theater_seat_representation :
  ∀ {row seat : ℕ}, ((row, seat) = (6, 2)) → (seat = 2 ∧ row = 6) :=
by
  intros row seat h
  have h1: row = 6 := by injection h with h_row h_seat; exact h_row
  have h2: seat = 2 := by injection h with h_row h_seat; exact h_seat
  exact ⟨h2, h1⟩

end theater_seat_representation_l421_421315


namespace polynomials_same_roots_iff_same_sign_of_f_l421_421772

theorem polynomials_same_roots_iff_same_sign_of_f 
  (P Q : ℂ[X]) (hP : P ≠ 0) (hQ : Q ≠ 0) :
  (∀ z : ℂ, z ≠ 0 → (|P.eval z| - |Q.eval z| = 0)) ↔ 
  (∀ (z : ℂ), (z ∈ P.roots.multiplicity) ↔ (z ∈ Q.roots.multiplicity)) :=
sorry

end polynomials_same_roots_iff_same_sign_of_f_l421_421772


namespace identify_homologous_functions_l421_421301

def homologous_functions (f : ℝ → ℝ) (range_1 range_2 : set ℝ) : Prop :=
∃ (domain_1 domain_2 : set ℝ), (domain_1 ∩ domain_2 = ∅) ∧ 
  (∀ x ∈ domain_1, f x ∈ range_1) ∧ (∀ x ∈ domain_2, f x ∈ range_2) ∧ 
  (range_1 = range_2) ∧ (range_1 ≠ ∅)

theorem identify_homologous_functions : homologous_functions (λ x, Real.sin x) (Ioo 0 1) (Ioo 0 1) ∧ 
  ¬ (homologous_functions (λ x, x) (Ioo 0 1) (Ioo 0 1)) ∧ 
  ¬ (homologous_functions (λ x, (2:ℝ)^x) (Ioo 0 1) (Ioo 0 1)) ∧ 
  ¬ (homologous_functions (λ x, Real.log x / Real.log 2) (Ioo 0 1) (Ioo 0 1)) :=
by {
  sorry
}

end identify_homologous_functions_l421_421301


namespace area_ratio_of_squares_l421_421047

theorem area_ratio_of_squares (s : ℝ) (hs : s > 0) :
  let small_square_area := s^2,
      total_fence_length := 16 * s,
      large_square_side := 4 * s,
      large_square_area := (4 * s)^2,
      total_small_squares_area := 4 * s^2
  in total_small_squares_area / large_square_area = 1 / 4 :=
by
  let small_square_area := s^2
  let total_fence_length := 16 * s
  let large_square_side := 4 * s
  let large_square_area := (4 * s)^2
  let total_small_squares_area := 4 * s^2
  show total_small_squares_area / large_square_area = 1 / 4
  sorry

end area_ratio_of_squares_l421_421047


namespace evening_customers_l421_421947

-- Define the conditions
def matinee_price : ℕ := 5
def evening_price : ℕ := 7
def opening_night_price : ℕ := 10
def popcorn_price : ℕ := 10
def num_matinee_customers : ℕ := 32
def num_opening_night_customers : ℕ := 58
def total_revenue : ℕ := 1670

-- Define the number of evening customers as a variable
variable (E : ℕ)

-- Prove that the number of evening customers E equals 40 given the conditions
theorem evening_customers :
  5 * num_matinee_customers +
  7 * E +
  10 * num_opening_night_customers +
  10 * (num_matinee_customers + E + num_opening_night_customers) / 2 = total_revenue
  → E = 40 :=
by
  intro h
  sorry

end evening_customers_l421_421947


namespace find_remainder_l421_421128

theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by
  sorry

end find_remainder_l421_421128


namespace decimal_150th_place_of_5_over_11_l421_421116

theorem decimal_150th_place_of_5_over_11 :
  let r := "45" in  -- The repeating decimal part
  let n := 150 in   -- The 150th place to find
  let repeat_len := 2 in -- Length of the repeating cycle
  cycle_digit r (n % repeat_len) = '5' := 
by
  sorry

/-- Helper function to get the nth digit of a repeating decimal cycle -/
def cycle_digit (cycle: String) (n: Nat) : Char := 
  cycle.get (n % cycle.length)

end decimal_150th_place_of_5_over_11_l421_421116


namespace proof_OPQ_Constant_l421_421616

open Complex

def OPQ_Constant :=
  ∀ (z1 z2 : ℂ) (θ : ℝ), abs z1 = 5 ∧
    (z1^2 - z1 * z2 * Real.sin θ + z2^2 = 0) →
      abs z2 = 5

theorem proof_OPQ_Constant : OPQ_Constant :=
by
  sorry

end proof_OPQ_Constant_l421_421616


namespace jane_reads_105_pages_in_a_week_l421_421338

-- Define the pages read in the morning and evening
def pages_morning := 5
def pages_evening := 10

-- Define the number of pages read in a day
def pages_per_day := pages_morning + pages_evening

-- Define the number of days in a week
def days_per_week := 7

-- Define the total number of pages read in a week
def pages_per_week := pages_per_day * days_per_week

-- The theorem that sums up the proof
theorem jane_reads_105_pages_in_a_week : pages_per_week = 105 := by
  sorry

end jane_reads_105_pages_in_a_week_l421_421338


namespace nabla_example_l421_421544

def nabla (a b : ℕ) : ℕ := 2 + b ^ a

theorem nabla_example : nabla (nabla 1 2) 3 = 83 :=
  by
  sorry

end nabla_example_l421_421544


namespace solve_abs_eq_l421_421901

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421901


namespace coordinates_of_P_with_respect_to_origin_l421_421418

def point (x y : ℝ) : Prop := True

theorem coordinates_of_P_with_respect_to_origin :
  point 2 (-3) ↔ point 2 (-3) := by
  sorry

end coordinates_of_P_with_respect_to_origin_l421_421418


namespace slope_is_7_over_4_sum_of_m_and_n_is_11_l421_421537

-- Define the vertices of the parallelogram
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (0,30)
def C : ℤ × ℤ := (20,50)
def D : ℤ × ℤ := (20,20)

-- Define the slope function
def slope (p1 p2 : ℤ × ℤ) : ℚ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Given that the slope should divide the parallelogram into two congruent trapezoids
noncomputable def slope_of_congruent_dividing_line (A B C D : ℤ × ℤ) : ℚ := slope A (20, 35)

theorem slope_is_7_over_4 :
  slope_of_congruent_dividing_line A B C D = 7/4 := 
sorry

theorem sum_of_m_and_n_is_11 :
  let m := 7
  let n := 4
  m + n = 11 := 
by
  sorry

end slope_is_7_over_4_sum_of_m_and_n_is_11_l421_421537


namespace problem1_problem2_l421_421176

-- Problem 1: Prove that (2a^2 b) * a b^2 / 4a^3 = 1/2 b^3
theorem problem1 (a b : ℝ) : (2 * a^2 * b) * (a * b^2) / (4 * a^3) = (1 / 2) * b^3 :=
  sorry

-- Problem 2: Prove that (2x + 5)(x - 3) = 2x^2 - x - 15
theorem problem2 (x : ℝ): (2 * x + 5) * (x - 3) = 2 * x^2 - x - 15 :=
  sorry

end problem1_problem2_l421_421176


namespace multiply_add_distribute_l421_421985

theorem multiply_add_distribute :
  42 * 25 + 58 * 42 = 3486 := by
  sorry

end multiply_add_distribute_l421_421985


namespace denominator_of_repeating_decimal_in_lowest_terms_l421_421808

-- Definition of the repeating decimal 0.454545...
def repeating_decimal_0_45 : Real := 0.454545...

-- Definition of lowest terms conversion
def lowest_terms (n d : Nat) : Nat × Nat :=
  let g := Nat.gcd n d
  (n / g, d / g)

-- Definition of the problem: Finding the denominator of the decimal in lowest terms
theorem denominator_of_repeating_decimal_in_lowest_terms :
  ∃ d, d = (lowest_terms 45 99).2 ∧ d = 11 :=
by
  sorry

end denominator_of_repeating_decimal_in_lowest_terms_l421_421808


namespace Sum_S11_l421_421689

-- Define the arithmetic sequence and necessary conditions
def arithmetic_sequence (a d : ℕ → ℕ) : Prop :=
  ∀ n, a(n + 1) = a(n) + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  n * (a 1 + a n) / 2 

theorem Sum_S11 (a : ℕ → ℕ) (d : ℕ → ℕ) (S₁₁ : ℕ) :
    arithmetic_sequence a d →
    (a 4 + a 5 + a 6 + a 7 + a 8 = 150) →
    S a 11 = 330 :=
  by
    intro h_sequence h_sum
    sorry

end Sum_S11_l421_421689


namespace angle_ADC_acute_of_angle_ACB_obtuse_l421_421681

theorem angle_ADC_acute_of_angle_ACB_obtuse
  (A B C D : Type)
  [InnerProductSpace ℝ A]
  (hAB_CD : dist A B = dist C D)
  (hAC_diagonal : ¬ collinear A C)
  (hACB_obtuse : ∃ ε > 0, ∡ B C A > π/2 + ε) :
  ∃ δ > 0, ∡ D C A < π/2 + δ := 
sorry

end angle_ADC_acute_of_angle_ACB_obtuse_l421_421681


namespace range_of_m_l421_421258

theorem range_of_m (m : ℝ) 
  (hp : ∀ x : ℝ, 2 * x > m * (x ^ 2 + 1)) 
  (hq : ∃ x0 : ℝ, x0 ^ 2 + 2 * x0 - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l421_421258


namespace missing_fraction_is_73_div_60_l421_421072

-- Definition of the given fractions
def fraction1 : ℚ := 1/3
def fraction2 : ℚ := 1/2
def fraction3 : ℚ := -5/6
def fraction4 : ℚ := 1/5
def fraction5 : ℚ := 1/4
def fraction6 : ℚ := -5/6

-- Total sum provided in the problem
def total_sum : ℚ := 50/60  -- 0.8333333333333334 in decimal form

-- The summation of given fractions
def sum_of_fractions : ℚ := fraction1 + fraction2 + fraction3 + fraction4 + fraction5 + fraction6

-- The statement to prove that the missing fraction is 73/60
theorem missing_fraction_is_73_div_60 : (total_sum - sum_of_fractions) = 73/60 := by
  sorry

end missing_fraction_is_73_div_60_l421_421072


namespace find_abc_sum_eq_14_l421_421420

theorem find_abc_sum_eq_14 : 
  ∃ (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c),
  (∀ x : ℝ, sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (6 * x) ^ 2 = 81 / 32 → 
  cos (a * x) * cos (b * x) * cos (c * x) = 0) ∧ a + b + c = 14 :=
by
  sorry

end find_abc_sum_eq_14_l421_421420


namespace total_cows_l421_421531

theorem total_cows (cows_per_herd : Nat) (herds : Nat) (total_cows : Nat) : 
  cows_per_herd = 40 → herds = 8 → total_cows = cows_per_herd * herds → total_cows = 320 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_cows_l421_421531


namespace reduction_factor_ice_melts_l421_421786

noncomputable def initial_volume : ℝ := 1
noncomputable def volume_ice : ℝ := initial_volume + initial_volume / 9
noncomputable def reduction_factor : ℝ := 1/10
noncomputable def reduced_volume : ℝ := reduction_factor * volume_ice

theorem reduction_factor_ice_melts :
  reduced_volume = initial_volume / 9 :=
begin
  sorry
end

end reduction_factor_ice_melts_l421_421786


namespace sum_of_integers_l421_421419

theorem sum_of_integers (a b : ℕ) (h_diff : a - b = 15) (h_prod : a * b = 56) (h_even : even a ∨ even b) : a + b = 29 :=
sorry

end sum_of_integers_l421_421419


namespace f_doesnt_take_a_n_l421_421768

-- Define the function f(n)
def f (n : ℕ) : ℕ :=
  ⌊ n + Real.sqrt (n / 3) + 1/2 ⌋

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  3 * n ^ 2 - 2 * n

theorem f_doesnt_take_a_n (n : ℕ) : ∃ m, f m ≠ 3 * n ^ 2 - 2 * n :=
by
  sorry

end f_doesnt_take_a_n_l421_421768


namespace aleksey_divisible_l421_421843

theorem aleksey_divisible
  (x y a b S : ℤ)
  (h1 : x + y = S)
  (h2 : S ∣ (a * x + b * y)) :
  S ∣ (b * x + a * y) := 
sorry

end aleksey_divisible_l421_421843


namespace abs_eq_condition_l421_421905

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l421_421905


namespace quadratic_root_relationship_l421_421577

theorem quadratic_root_relationship (a b c : ℂ) (alpha beta : ℂ) (h1 : a ≠ 0) (h2 : alpha + beta = -b / a) (h3 : alpha * beta = c / a) (h4 : beta = 3 * alpha) : 3 * b ^ 2 = 16 * a * c := by
  sorry

end quadratic_root_relationship_l421_421577


namespace exterior_angle_BAC_l421_421962

-- Definitions for the problem conditions
def regular_nonagon_interior_angle :=
  140

def square_interior_angle :=
  90

-- The proof statement
theorem exterior_angle_BAC (regular_nonagon_interior_angle square_interior_angle : ℝ) : 
  regular_nonagon_interior_angle = 140 ∧ square_interior_angle = 90 -> 
  ∃ (BAC : ℝ), BAC = 130 :=
by
  sorry

end exterior_angle_BAC_l421_421962


namespace angle_bisector_intersect_equal_angles_l421_421700

theorem angle_bisector_intersect_equal_angles (A B C A1 B1 C1 M N : Point)
  (h1: Line.tangentAt AA1)
  (h2: Line.tangentAt BB1)
  (h3: Line.tangentAt CC1)
  (hM: Segment.intersect M (Line.segment C1 B1))
  (hN: Segment.intersect N (Line.segment B1 A1))
  : ∠ MBB1 = ∠ NBB1 := by
  sorry

end angle_bisector_intersect_equal_angles_l421_421700


namespace johns_weekly_quarts_l421_421710

def daily_water_gal := 1.5 -- gallons/day
def daily_milk_pints := 3 -- pints/day
def daily_juice_oz := 20 -- fluid ounces/day

def gal_to_quarts := 4.0 -- quarts/gallon
def pint_to_quarts := 0.5 -- quarts/pint
def oz_to_quarts := 0.03125 -- quarts/fluid ounce

def daily_water_quarts := daily_water_gal * gal_to_quarts
def daily_milk_quarts := daily_milk_pints * pint_to_quarts
def daily_juice_quarts := daily_juice_oz * oz_to_quarts

def total_daily_quarts := daily_water_quarts + daily_milk_quarts + daily_juice_quarts
def days_per_week := 7
def total_weekly_quarts := total_daily_quarts * days_per_week

theorem johns_weekly_quarts :
  total_weekly_quarts = 56.875 := by
  sorry

end johns_weekly_quarts_l421_421710


namespace robie_initial_cards_l421_421379

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l421_421379


namespace direction_vector_correctness_l421_421211

def sequence (n : ℕ) : ℝ := 4 * n - 1

theorem direction_vector_correctness (a_1 a_2 a_3 a_4 : ℝ) (d : ℝ)
  (h1 : a_1 + a_2 = 10)  
  (h2 : a_3 + a_4 = 26)
  (h3 : ∀ n : ℕ, a_1 + d * (n - 1) = sequence n)
  (h4 : d = 4)
  (h5 : a_1 = 3)
  (n : ℕ) (hn : 0 < n) :
  let P := (n, sequence n)
  let Q := (n + 1, sequence (n + 2)) in
  (Q.1 - P.1) * -4 = (Q.2 - P.2) * (-1/2) :=
sorry

end direction_vector_correctness_l421_421211


namespace strokes_over_par_l421_421093

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l421_421093


namespace parabola_chord_constant_l421_421449

noncomputable def calcT (x₁ x₂ c : ℝ) : ℝ :=
  let a := x₁^2 + (2*x₁^2 - c)^2
  let b := x₂^2 + (2*x₂^2 - c)^2
  1 / Real.sqrt a + 1 / Real.sqrt b

theorem parabola_chord_constant (c : ℝ) (m x₁ x₂ : ℝ) 
    (h₁ : 2*x₁^2 - m*x₁ - c = 0) 
    (h₂ : 2*x₂^2 - m*x₂ - c = 0) : 
    calcT x₁ x₂ c = -20 / (7 * c) :=
by
  sorry

end parabola_chord_constant_l421_421449


namespace B_received_profit_l421_421141

/-- Proof problem: given the conditions stated, 
    we need to prove that B received Rs 3000 as profit. -/
theorem B_received_profit 
  (B_investment : ℝ) 
  (B_time : ℝ) 
  (total_profit : ℝ) 
  (A_investment : ℝ := 3 * B_investment) 
  (A_time : ℝ := 2 * B_time) 
  (profit_ratio : ℝ := (A_investment * A_time) / (B_investment * B_time)) 
  (total_ratio_parts : ℝ := profit_ratio + 1) :
  total_profit = 21000 →
  B_investment * B_time ≠ 0 →
  B_time ≠ 0 →
  B_investment ≠ 0 →
  B_received : ℝ := total_profit / total_ratio_parts 
  → B_received = 3000 := by
  sorry

end B_received_profit_l421_421141


namespace x_cubed_plus_y_cubed_l421_421650

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end x_cubed_plus_y_cubed_l421_421650


namespace hyperbola_distance_to_left_focus_l421_421810

theorem hyperbola_distance_to_left_focus (P : ℝ × ℝ)
  (h1 : (P.1^2) / 9 - (P.2^2) / 16 = 1)
  (dPF2 : dist P (4, 0) = 4) : dist P (-4, 0) = 10 := 
sorry

end hyperbola_distance_to_left_focus_l421_421810


namespace exists_equilateral_triangles_l421_421508

structure Point :=
(x : ℝ) (y : ℝ)

def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

def is_isosceles (A M B : Point) (angleAMB : ℝ) : Prop :=
(A.x - M.x) ^ 2 + (A.y - M.y) ^ 2 = (B.x - M.x) ^ 2 + (B.y - M.y) ^ 2 ∧ angleAMB = 120

theorem exists_equilateral_triangles (A B C D M : Point) :
  is_convex_quadrilateral A B C D →
  is_isosceles A M B (120) →
  is_isosceles C M D (120) →
  ∃ N : Point, 
    (equilateral (B N C) ∧ equilateral (D N A)) := sorry

end exists_equilateral_triangles_l421_421508


namespace equilateral_triangle_area_l421_421412

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l421_421412


namespace min_MN_length_l421_421351

noncomputable def cube_min_distance (x y : ℝ) :=
  real.sqrt ((x - 1) ^ 2 + y ^ 2 + 1)

theorem min_MN_length :
  ∃ x y, (y = x ∧ x ≥ 0 ∧ cube_min_distance x y = 3) :=
begin
  use [1, 1],
  split,
  { exact rfl, },
  split,
  { linarith, },
  { simp [cube_min_distance],
    norm_num, },
end

end min_MN_length_l421_421351


namespace angle_measures_possible_l421_421822

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l421_421822


namespace intervals_of_monotonicity_minimum_value_l421_421276

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem intervals_of_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x, 0 < x ∧ x ≤ 1 / a → f a x ≤ f a (1 / a)) ∧
  (∀ x, x ≥ 1 / a → f a x ≥ f a (1 / a)) :=
sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (a < Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = -a) ∧
  (a ≥ Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = Real.log 2 - 2 * a) :=
sorry

end intervals_of_monotonicity_minimum_value_l421_421276


namespace ellen_dinner_calories_proof_l421_421553

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l421_421553


namespace original_function_eqn_l421_421452

theorem original_function_eqn
  (a : ℝ × ℝ := (3, -2))
  (translated_function : ℝ → ℝ := λ x, log 2 (x + 3) + 2) :
  (∀ x, translated_function(x) = (log 2 (x + 6) + 4)) := 
by
  sorry

end original_function_eqn_l421_421452


namespace monotonic_decreasing_interval_ln_quadratic_l421_421429

noncomputable def u (x : ℝ) : ℝ := 4 + 3 * x - x^2

def decreasing_interval (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem monotonic_decreasing_interval_ln_quadratic :
  ∀ (x : ℝ), x ∈ Set.Ico (3 / 2) 4 → decreasing_interval (λ x, Real.log (u x)) (Set.Ico (3 / 2) 4) := 
by
  sorry

end monotonic_decreasing_interval_ln_quadratic_l421_421429


namespace identify_counterfeit_coins_with_six_weighings_l421_421447

noncomputable section

-- Definitions of coins and weights
structure Coin :=
  (is_gold : Bool)
  (is_real : Bool)

def weight (c : Coin) : ℕ :=
  if c.is_gold then
    if c.is_real then 20 else 19
  else
    if c.is_real then 10 else 9

-- Initial set of coins
def gold_coins : List Coin := List.replicate 7 ⟨true, true⟩ ++ [⟨true, false⟩]
def silver_coins : List Coin := List.replicate 7 ⟨false, true⟩ ++ [⟨false, false⟩]

-- two-pan balance function
def balance (left right : List Coin) : Ordering :=
  compare (left.map weight).sum (right.map weight).sum

-- The proof problem statement
theorem identify_counterfeit_coins_with_six_weighings : ∃ (strategy : List (List Coin × List Coin)), strategy.length = 6 ∧ 
  ∀ (gold_coins silver_coins : List Coin), 
    (∃ g ∈ gold_coins, ¬ g.is_real) → 
    (∃ s ∈ silver_coins, ¬ s.is_real) → 
    (∀ (step : List Coin × List Coin) ∈ strategy, 
       match balance step.1 step.2 with
       | Ordering.eq => true
       | Ordering.gt => true
       | Ordering.lt => true) → 
    (∃ (cg : Coin) (cs : Coin), cg ∈ gold_coins ∧ ¬ cg.is_real ∧ cs ∈ silver_coins ⟨noncomputable⟩.
-- Sorry is used to complete the proof placeholer verification
sorry

end identify_counterfeit_coins_with_six_weighings_l421_421447


namespace sum_of_squares_l421_421394

theorem sum_of_squares (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h1 : x + y + z = 30) (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  ∃ (s : ℕ), s = 710 ∧ ∀ (a₁ a₂ a₃ : ℕ), 
    (a₁ = x² + y² + z² ∨ a₂ = x² + y² + z² ∨ a₃ = x² + y² + z²) →
    a₁ + a₂ + a₃ = s :=
sorry

end sum_of_squares_l421_421394


namespace base_8_addition_l421_421967

-- Definitions
def five_base_8 : ℕ := 5
def thirteen_base_8 : ℕ := 1 * 8 + 3 -- equivalent of (13)_8 in base 10

-- Theorem statement
theorem base_8_addition :
  (five_base_8 + thirteen_base_8) = 2 * 8 + 0 :=
sorry

end base_8_addition_l421_421967


namespace men_in_second_group_l421_421137

theorem men_in_second_group (M : ℕ) (W : ℝ) (h1 : 15 * 25 = W) (h2 : M * 18.75 = W) : M = 20 :=
sorry

end men_in_second_group_l421_421137


namespace gradient_descent_converges_to_minimum_l421_421376

-- Define the error function I(a, b)
def error_function (a b : ℝ) (data : list (ℝ × ℝ)) : ℝ :=
  data.foldl (λ acc (x, y), acc + (y - (Real.cos (a * x) + b))^2) 0

-- Define the gradient of the error function
def grad_I (a b : ℝ) (data : list (ℝ × ℝ)) : ℝ × ℝ :=
  let partial_a := data.foldl (λ acc (x, y),
    acc + 2 * (y - (Real.cos (a * x) + b)) * (Real.sin (a * x) * x)) 0
  let partial_b := data.foldl (λ acc (x, y),
    acc - 2 * (y - (Real.cos (a * x) + b))) 0
  (partial_a, partial_b)

-- Define a single iteration of the algorithm
def gradient_descent_step (a b α : ℝ) (data : list (ℝ × ℝ)) : ℝ × ℝ :=
  let (grad_a, grad_b) = grad_I a b data
  let a_new := a - α * grad_a
  let b_new := b - α * grad_b
  (a_new, b_new)

-- Define the main function which runs the gradient descent algorithm
noncomputable def run_gradient_descent (data : list (ℝ × ℝ)) : ℝ × ℝ :=
  let iterations := 5000
  let initial_a := 2.0
  let initial_b := 0.5
  let (final_a, final_b) := (List.range iterations).foldl
    (λ (ab : ℝ × ℝ) i,
      let (a, b) := ab
      let α := 1 / (10 + i)
      gradient_descent_step a b α data) 
    (initial_a, initial_b)
  (final_a, final_b)

-- Define the final error after running gradient descent
noncomputable def final_error (data : list (ℝ × ℝ)) : ℝ :=
  let (a, b) := run_gradient_descent data
  error_function a b data

-- The main theorem which states the final error is approximately 33.58 ± 0.02
theorem gradient_descent_converges_to_minimum (data : list (ℝ × ℝ)) :
  final_error data ≈ 33.58 ∨ final_error data ≈ (33.58 - 0.02) ∨ final_error data ≈ (33.58 + 0.02) :=
sorry

end gradient_descent_converges_to_minimum_l421_421376


namespace part1_part2_l421_421240

noncomputable def a : ℕ → ℤ
| 0 => -1
| n + 1 => 2 * a n + 3

def b (n : ℕ) : ℤ := a n + 3

def bn_geometric : Prop :=
  ∀ n, b n = 2 ^ n

def sum_Sn (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, (1 : ℚ) / ((Real.log 2 (b k) * Real.log 2 (b (k+1)))))

theorem part1 (n : ℕ) : bn_geometric :=
  sorry

theorem part2 (n : ℕ) : sum_Sn n = n / (n + 1 : ℚ) :=
  sorry

end part1_part2_l421_421240


namespace change_correct_l421_421707

-- Define the conditions
def number_of_packs : ℕ := 3
def cost_per_pack : ℕ := 3
def initial_payment : ℕ := 20

-- Calculate the total cost of the candy
def total_cost : ℕ := number_of_packs * cost_per_pack

-- Define the expected change
def expected_change : ℕ := 11

-- Prove that the change received is as expected
theorem change_correct : initial_payment - total_cost = expected_change := by
  -- calculation of change
  have h1 : total_cost = 9 := by
    simp [total_cost]
  have h2 : initial_payment - total_cost = 11 := by
    simp [initial_payment, h1]
  exact h2

end change_correct_l421_421707


namespace problem_statement_l421_421698

-- Definition of the regular tetrahedron and relevant points
structure Tetrahedron (α : Type _) :=
(A B C D : α)
(is_regular : true) -- Placeholder for the property that the tetrahedron is regular

variables {α : Type _} [inner_product_space ℝ α] {A B C D E F : α}

-- Points E and F with given ratios lambda
def is_point_on_edge (p1 p2 E : α) (λ : ℝ) := E = (1 / (1 + λ)) • p1 + (λ / (1 + λ)) • p2

-- Function f
def f (λ : ℝ) [0 < λ] (E F A C B D : α) :=
let α_λ := inner_product_space.angle E F A C, -- Angle between EF and AC
    β_λ := inner_product_space.angle E F B D -- Angle between EF and BD
in α_λ + β_λ 

-- Statement to be proved
theorem problem_statement (t : Tetrahedron α) (λ : ℝ)
  (h_λ : 0 < λ) (hE : is_point_on_edge t.A t.B E λ) (hF : is_point_on_edge t.C t.D F λ) :
  f λ E F t.A t.C t.B t.D = 90 :=
sorry

end problem_statement_l421_421698


namespace trigonometric_identity_l421_421226

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 :=
by
  sorry

end trigonometric_identity_l421_421226


namespace angle_measures_possible_l421_421821

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l421_421821


namespace line_passes_through_vertex_of_parabola_l421_421216

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l421_421216


namespace subtraction_of_smallest_from_largest_l421_421916

open Finset

def three_digit_numbers (s : Finset ℕ) : Finset ℕ :=
  (s.product (s.filter (· ≠ _)).product (s.filter (· ≠ _)).map 
    (λ x, 100 * x.1.1 + 10 * x.1.2 + x.2)).filter (λ n, n >= 100 ∧ n < 1000)

theorem subtraction_of_smallest_from_largest : 
  let numbers := {1, 2, 6, 7, 8} in
  let largest := max' (three_digit_numbers numbers) sorry in
  let smallest := min' (three_digit_numbers numbers) sorry in
  largest - smallest = 750 := 
by
  sorry

end subtraction_of_smallest_from_largest_l421_421916


namespace angle_ACB_is_60_degrees_l421_421316

noncomputable def angle_bisector_intersection (A B C L M K : Point)
  (hAngleBisectorAL : is_angle_bisector A C L)
  (hAngleBisectorBM : is_angle_bisector B C M)
  (hIntersection : lies_on_segment K A B)
  (hCircumcircle1 : on_circumcircle K A C L)
  (hCircumcircle2 : on_circumcircle K B C M) : Prop :=
  ∠ACB = 60

theorem angle_ACB_is_60_degrees (A B C L M K : Point)
  (hAngleBisectorAL : is_angle_bisector A C L)
  (hAngleBisectorBM : is_angle_bisector B C M)
  (hIntersection : lies_on_segment K A B)
  (hCircumcircle1 : on_circumcircle K A C L)
  (hCircumcircle2 : on_circumcircle K B C M) :
  angle_bisector_intersection A B C L M K hAngleBisectorAL hAngleBisectorBM hIntersection hCircumcircle1 hCircumcircle2 :=
by
  sorry

end angle_ACB_is_60_degrees_l421_421316


namespace find_AE_l421_421538

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end find_AE_l421_421538


namespace problem_solution_l421_421066

noncomputable def p_q_sum : ℤ :=
  let a := 5
  let b := -9
  let c := -11
  let Δ := b^2 - 4 * a * c
  let q := 5
  let p := Δ
  p + q

-- Proposition statement to be proved
theorem problem_solution : p_q_sum = 306 := 
by
  let a := 5
  let b := -9
  let c := -11
  let Δ := b^2 - 4 * a * c
  have h1 : Δ = 301 := by norm_num
  have h2 : q = 5 := rfl
  have h3 : p = Δ := rfl
  have h4 : p + q = 301 + 5 := by norm_num
  show p_q_sum = 306
  rw [←h3, ←h2, ←h1]
  exact h4
  sorry

end problem_solution_l421_421066


namespace correct_option_d_l421_421608

variables (m l : Line) (α β : Plane)

-- Hypotheses
axiom h1 : m ≠ l
axiom h2 : α ≠ β
axiom h3 : Perpendicular m α
axiom h4 : Parallel l β
axiom h5 : Parallel α β

-- Statement to prove
theorem correct_option_d : Perpendicular m l := 
sorry

end correct_option_d_l421_421608


namespace least_number_divisible_l421_421925

theorem least_number_divisible (x : ℕ) :
  (∀ d ∈ ({48, 64, 72, 108, 125} : set ℕ), (x + 12) % d = 0) → x = 215988 :=
by
  sorry -- Proof is omitted

end least_number_divisible_l421_421925


namespace sum_of_interior_angles_l421_421655

theorem sum_of_interior_angles (n : ℕ) : 
  (∀ θ, θ = 40 ∧ (n = 360 / θ)) → (n - 2) * 180 = 1260 :=
by
  sorry

end sum_of_interior_angles_l421_421655


namespace simplify_expr_l421_421387

def expr1 : ℚ := (3 + 4 + 6 + 7) / 3
def expr2 : ℚ := (3 * 6 + 9) / 4

theorem simplify_expr : expr1 + expr2 = 161 / 12 := by
  sorry

end simplify_expr_l421_421387


namespace shift_graph_l421_421451

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 3)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 3)

theorem shift_graph :
  ∀ x, f(x) = g(x + π / 4) :=
by
  sorry

end shift_graph_l421_421451


namespace probability_of_expression_equality_l421_421022

noncomputable def valid_rationals (n d : ℕ) : set ℚ :=
  { x | 0 ≤ x ∧ x < 3 ∧ ∃ (n d : ℤ), n.nat_abs ≤ 3 ∧ 1 ≤ d.nat_abs ∧ d.nat_abs ≤ 4 ∧ x = n / d }

noncomputable def valid_a (a : ℚ) : Prop :=
  a ∈ valid_rationals ∧ (∃ (k : ℤ), a = k / 4)

noncomputable def valid_b (b : ℚ) : Prop :=
  b ∈ valid_rationals ∧ (∃ (k : ℤ), b = k / 8)

theorem probability_of_expression_equality :
  let S := valid_rationals in
  let valid_pairs := { (a, b) | valid_a a ∧ valid_b b } in
  let total_pairs := set.prod valid_rationals valid_rationals in
  (card valid_pairs : ℚ) / (card total_pairs : ℚ) = 27 / 289 :=
sorry

end probability_of_expression_equality_l421_421022


namespace find_second_number_l421_421842

theorem find_second_number (x y z : ℚ) (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 240 / 7 := by
  sorry

end find_second_number_l421_421842


namespace intersections_of_segments_l421_421011

theorem intersections_of_segments (m n : ℕ) : 
  let num_intersections := (m * (m - 1) / 2) * (n * (n - 1) / 2) in
  (∀ a b : Set Point, ∃ A : Finset a, A.card = m ∧ ∃ B : Finset b, B.card = n) →
  (∀ segs : Finset (Σ (i : Fin m) (j : Fin n), Segment A B), 
    ∀ s1 s2 ∈ segs, s1 ≠ s2 → s1 ∩ s2 ∉ (some three points intersecting at the same time)) →
  num_intersections = binom m 2 * binom n 2 := 
by
  sorry

end intersections_of_segments_l421_421011


namespace isosceles_triangle_perimeter_l421_421673

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 5) (h2 : b = 8) :
    (∃ x : ℕ, x = 18 ∨ x = 21) :=
by
  use [18, 21]
  split
  · left
    sorry
  · right
    sorry

end isosceles_triangle_perimeter_l421_421673


namespace usable_area_l421_421940

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def pond_side : ℕ := 4

theorem usable_area :
  garden_length * garden_width - pond_side * pond_side = 344 :=
by
  sorry

end usable_area_l421_421940


namespace geometric_series_mod_l421_421472

theorem geometric_series_mod :
  let a := 1
  let r := 7
  let n := 2005
  let sum := (r ^ n - 1) * a / (r - 1)
  (sum % 1000) = 801 :=
by
  let a := 1
  let r := 7
  let n := 2005
  let sum := (r ^ n - 1) * a / (r - 1)
  show (sum % 1000) = 801 from sorry

end geometric_series_mod_l421_421472


namespace product_of_two_numbers_l421_421053

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l421_421053


namespace identify_counterfeit_bag_l421_421846

-- Definitions based on problem conditions
def num_bags := 10
def genuine_weight := 10
def counterfeit_weight := 11
def expected_total_weight := genuine_weight * ((num_bags * (num_bags + 1)) / 2 : ℕ)

-- Lean theorem for the above problem
theorem identify_counterfeit_bag (W : ℕ) (Δ := W - expected_total_weight) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ num_bags ∧ Δ = i :=
by sorry

end identify_counterfeit_bag_l421_421846


namespace matrix_det_equals_l421_421190

noncomputable def matrix_det : ℝ → ℝ := 
  λ x, (Matrix.det ![
    ![x + 2, x - 1, x],
    ![x - 1, x + 2, x],
    ![x, x, x + 3]
  ])

theorem matrix_det_equals : ∀ x : ℝ, matrix_det x = 14 * x + 9 := by
  sorry

end matrix_det_equals_l421_421190


namespace general_term_formula_Tn_bound_find_a_l421_421286

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 3 else if n = 2 then 5 else 2^n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
1 / (a_n n * a_n (n + 1))

noncomputable def f (x : ℕ) : ℝ :=
2^(x - 1)

noncomputable def T_n (n : ℕ) : ℝ :=
∑ i in Finset.range n, b_n (i + 1) * f (i + 1)

noncomputable def T_n_a (n : ℕ) (a : ℝ) : ℝ :=
(1/2) * (∑ i in Finset.range n, b_n (i + 1) * a^(i + 1))

theorem general_term_formula : ∀ n : ℕ, a_n n = 2^n + 1 :=
sorry

theorem Tn_bound (n : ℕ) : T_n n < 1/6 :=
sorry

theorem find_a : ∀ a : ℝ, (∀ n : ℕ , T_n_a n a < 1/6) ∧ (∀ m ∈ (0, 1/6), ∃ n_0 : ℕ, ∀ n ≥ n_0, T_n_a n a > m) ↔ a = 2 :=
sorry

end general_term_formula_Tn_bound_find_a_l421_421286


namespace general_term_of_sequence_l421_421631

theorem general_term_of_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, a (n + 1) = (n^2 * (a n)^2 + 5) / ((n^2 - 1) * a (n - 1))) :
  ∀ n : ℕ, a n = 
    if n = 0 then 0 else
    (1 / n) * ( (63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2) ^ n + 
                (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2) ^ n) :=
by
  sorry

end general_term_of_sequence_l421_421631


namespace cube_edge_length_l421_421612

theorem cube_edge_length (V : ℝ) (a : ℝ) (h₁ : V = 9 * π / 2) (h₂ : V = (4 / 3 ) * π * ( (√3 * a / 2) ^ 3 )) : 
  a = √3 :=
  sorry

end cube_edge_length_l421_421612


namespace possible_measures_of_angle_A_l421_421829

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l421_421829


namespace initial_population_l421_421671

theorem initial_population (P : ℝ) 
  (h1 : P * 0.90 * 0.95 * 0.85 * 1.08 = 6514) : P = 8300 :=
by
  -- Given conditions lead to the final population being 6514
  -- We need to show that the initial population P was 8300
  sorry

end initial_population_l421_421671


namespace max_value_of_f_l421_421427

-- Define the function
def f (x : ℝ) : ℝ := x / (x - 1)

-- Define the condition
def condition (x : ℝ) : Prop := x ≥ 2

-- State the theorem that the maximum value of the function for x ≥ 2 is 2
theorem max_value_of_f : ∃ M, ∀ x : ℝ, condition x → f x ≤ M ∧ (f 2 = M) :=
sorry

end max_value_of_f_l421_421427


namespace solve_abs_eq_l421_421903

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421903


namespace simplify_expression_l421_421345

theorem simplify_expression (a b c d : ℝ) (h₁ : a + b + c + d = 0) (h₂ : a ≠ 0) (h₃ : b ≠ 0) (h₄ : c ≠ 0) (h₅ : d ≠ 0) :
  (1 / (b^2 + c^2 + d^2 - a^2) + 
   1 / (a^2 + c^2 + d^2 - b^2) + 
   1 / (a^2 + b^2 + d^2 - c^2) + 
   1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := 
sorry

end simplify_expression_l421_421345


namespace tile_IV_in_C_l421_421081

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)
  (sum : ℕ := top + right + bottom + left)

def TileIV : Tile := {top := 3, right := 5, bottom := 6, left := 4}

axiom all_tiles : List Tile := [
  {top := 5, right := 3, bottom := 6, left := 2},
  {top := 3, right := 5, bottom := 2, left := 6},
  {top := 0, right := 7, bottom := 1, left := 3},
  TileIV
]

-- Define the conditions given in the problem here
def adjacent_sums_equal (t1 t2 : Tile) := t1.sum = t2.sum
def sides_match (t1 t2 : Tile) : Prop :=
  t1.right = t2.left ∧ t1.left = t2.right ∧ t1.top = t2.bottom ∧ t1.bottom = t2.top

-- Given the conditions
def tile_placement_condition (tC tD : Tile) : Prop :=
  adjacent_sums_equal tC tD ∧ sides_match tC tD

-- Problem to be proved
theorem tile_IV_in_C : ∃ tC tD : Tile, tile_placement_condition tC tD ∧ tC = TileIV :=
by
  sorry

end tile_IV_in_C_l421_421081


namespace sum_evaluation_l421_421348

open Complex

noncomputable def sum_problem (x : ℂ) (hx1 : x ^ 2009 = 1) (hx2 : x ≠ 1) : ℂ :=
  ∑ k in finset.range 2008, x ^ (4 * (k + 1)) / (x ^ (k + 1) - 1)

theorem sum_evaluation (x : ℂ) (hx1 : x ^ 2009 = 1) (hx2 : x ≠ 1) :
  sum_problem x hx1 hx2 = 1004 := 
sorry

end sum_evaluation_l421_421348


namespace system_of_equations_l421_421318

theorem system_of_equations (x y : ℝ) (h1 : 3 * x = 4 * y) (h2 : 6 * x + 9 * y = 76500) : 
    (3 * x = 4 * y ∧ 6 * x + 9 * y = 76500) :=
by
  split
  · exact h1
  · exact h2

end system_of_equations_l421_421318


namespace hans_room_count_l421_421289

theorem hans_room_count :
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  available_floors * rooms_per_floor = 90 := by
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  show available_floors * rooms_per_floor = 90
  sorry

end hans_room_count_l421_421289


namespace sin_13pi_over_4_eq_neg_sqrt2_over_2_l421_421565

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end sin_13pi_over_4_eq_neg_sqrt2_over_2_l421_421565


namespace inscribed_circle_radius_l421_421317

variable (A p s r : ℝ)

-- Condition: Area is twice the perimeter
def twice_perimeter_condition : Prop := A = 2 * p

-- Condition: The formula connecting the area, inradius, and semiperimeter
def area_inradius_semiperimeter_relation : Prop := A = r * s

-- Condition: The perimeter is twice the semiperimeter
def perimeter_semiperimeter_relation : Prop := p = 2 * s

-- Prove the radius of the inscribed circle is 4
theorem inscribed_circle_radius (h1 : twice_perimeter_condition A p)
                                (h2 : area_inradius_semiperimeter_relation A r s)
                                (h3 : perimeter_semiperimeter_relation p s) :
  r = 4 :=
by
  sorry

end inscribed_circle_radius_l421_421317


namespace ratio_surface_area_l421_421233

variable (r h : ℝ) (π : ℝ) [Real.pi_def] -- Treat pi as a constant, can use Mathlib's pi definition

-- Conditions

-- 1. The height h is equal to πr
axiom height_eq_pi_mul_radius : h = π * r

-- 2. h < 2πr
axiom height_lt_2pi_mul_radius : h < 2 * π * r

-- Question: Prove the ratio of total surface area to lateral surface area is (1+π)/π
theorem ratio_surface_area (r h π : ℝ) [Real.pi_def]
  (height_eq_pi_mul_radius : h = π * r)
  (height_lt_2pi_mul_radius : h < 2 * π * r) :
  (2 * π * r * (r + h)) / (2 * π * r * h) = (1 + π) / π :=
by
  sorry

end ratio_surface_area_l421_421233


namespace possible_measures_A_l421_421824

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l421_421824


namespace inequality_lemma_l421_421659

theorem inequality_lemma (a b c d : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) (hd : 0 < d ∧ d < 1) :
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d :=
by 
  sorry

end inequality_lemma_l421_421659


namespace find_x_plus_y_l421_421732

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y) 
    (h2 : det ![
      ![2, 3, 7],
      ![4, x, y],
      ![4, y, x + 1]
    ] = 0) : x + y = 20 :=
sorry

end find_x_plus_y_l421_421732


namespace solve_absolute_value_eq_l421_421915

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l421_421915


namespace max_number_soap_boxes_l421_421918

-- Definition of dimensions and volumes
def carton_length : ℕ := 25
def carton_width : ℕ := 42
def carton_height : ℕ := 60
def soap_box_length : ℕ := 7
def soap_box_width : ℕ := 12
def soap_box_height : ℕ := 5

def volume (l w h : ℕ) : ℕ := l * w * h

-- Volumes of the carton and soap box
def carton_volume : ℕ := volume carton_length carton_width carton_height
def soap_box_volume : ℕ := volume soap_box_length soap_box_width soap_box_height

-- The maximum number of soap boxes that can be placed in the carton
def max_soap_boxes : ℕ := carton_volume / soap_box_volume

theorem max_number_soap_boxes :
  max_soap_boxes = 150 :=
by
  -- Proof here
  sorry

end max_number_soap_boxes_l421_421918


namespace coeff_x_term_in_expansion_l421_421199

theorem coeff_x_term_in_expansion :
  let f := λ (x : ℝ), (x ^ 2 - x - 2) ^ 4 in
  (f x).coeff 1 = 32 :=
by 
  sorry

end coeff_x_term_in_expansion_l421_421199


namespace calories_remaining_for_dinner_l421_421558

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l421_421558


namespace triangle_angle_sum_l421_421371

theorem triangle_angle_sum (α β γ : ℝ) (h : α + β + γ = 180) (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
sorry

end triangle_angle_sum_l421_421371


namespace possible_for_14_teeth_impossible_for_13_teeth_l421_421456

-- Define the generic problem with conditions
def Gear (n : ℕ) : Type := { gears : Fin n // gears.val < n }

-- Assumptions based on the setup
variable (A B : Gear n)
variable (removed_teeth : Finset (Fin n))

-- Conditions specific to the problem
axiom teeth_equal : (A.gears.val = B.gears.val)
axiom num_teeth_14 : n = 14
axiom num_teeth_13 : n = 13
axiom four_pairs_removed : removed_teeth.card = 4

-- Proving the two cases
theorem possible_for_14_teeth (h : n = 14) : 
  ∃ θ : ℕ, (rotated_projection A θ) ∩ (B.gears) = complete_projection B := 
by 
  sorry

theorem impossible_for_13_teeth (h : n = 13) : 
  ¬ ∃ θ : ℕ, (rotated_projection A θ) ∩ (B.gears) = complete_projection B := 
by 
  sorry

end possible_for_14_teeth_impossible_for_13_teeth_l421_421456


namespace pawns_placement_possible_l421_421381

-- Definitions
def Square := (ℕ × ℕ)
def Board := Fin₈ × Fin₈

variable {S : set Square}
variable (S_cond1 : ∀ r : Fin₈, 2 = S.to_finset.filter (λ s, s.1 = r).card)
variable (S_cond2 : ∀ c : Fin₈, 2 = S.to_finset.filter (λ s, s.2 = c).card)

-- Theorem statement
theorem pawns_placement_possible (S : set Board) 
  (h1 : ∀ r : Fin₈, 2 = (S.filter (λ s, s.1 = r)).to_finset.card)
  (h2 : ∀ c : Fin₈, 2 = (S.filter (λ s, s.2 = c)).to_finset.card) 
  : ∃ P : Board → Prop, 
      (∀ r : Fin₈, (λ B : Prop, (∃ (x : Fin₈), ((r, x) ∈ S ∧ B (r, x))) ∧ 
      (∃ (y : Fin₈), ((r, y) ∈ S ∧ ¬ B (r, y))))) P ∧ 
      (∀ c : Fin₈, (λ W : Prop, (∃ (x : Fin₈), ((x, c) ∈ S ∧ W (x, c))) ∧ 
      (∃ (y : Fin₈), ((y, c) ∈ S ∧ ¬ W (y, c))))) P :=
sorry

end pawns_placement_possible_l421_421381


namespace acute_triangle_in_right_triangle_area_le_sqrt3_l421_421769

theorem acute_triangle_in_right_triangle_area_le_sqrt3
  (ABC : Type*)
  [triangle ABC]
  [acute_angled ABC]
  (A B C : Point ABC)
  (h_area : triangle.area ABC = 1) :
  ∃ (α β γ : Point ABC), right_triangle α β γ ∧ triangle.area α β γ ≤ √3 :=
sorry

end acute_triangle_in_right_triangle_area_le_sqrt3_l421_421769


namespace solution_l421_421832

theorem solution (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 12) : (12 * y - 4)^2 = 128 :=
sorry

end solution_l421_421832


namespace function_pass_point_l421_421049

theorem function_pass_point (f g : ℝ → ℝ) 
  (h_symm : ∀ x y, y = g(x) ↔ x = f(y - 1))
  (h_g : g 2 = 0) : 
  f (-1) = 2 :=
by sorry

end function_pass_point_l421_421049


namespace max_sqrt_min_value_l421_421578

def max (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

def f (x : ℝ) : ℝ :=
  max (x + 4) (-x + 2)

noncomputable def sqrt (x : ℝ) : ℝ := sorry -- assuming we have a sqrt function

theorem max_sqrt_min_value :
  let m := 3 in max (sqrt m) (m - 1) = 2 :=
by
  -- We assume the proof is done here
  sorry

end max_sqrt_min_value_l421_421578


namespace sum_real_imag_l421_421268

theorem sum_real_imag (z : ℂ) (hz : z = 3 - 4 * I) : z.re + z.im = -1 :=
by {
  -- Because the task asks for no proof, we're leaving it with 'sorry'.
  sorry
}

end sum_real_imag_l421_421268


namespace correct_option_l421_421023

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log (1/2)

theorem correct_option : 
  ∀ (x : ℝ), 
  (∀(y z : ℝ), f y = f z → abs y = abs z) ∧ -- Symmetric about y-axis
  (∀x, x ≠ 0 → x ∈ set_of (λ x, f x ∈ set.univ)) ∧ -- Domain is (-∞, 0) ∪ (0, ∞)
  (∀ x y, x < y ∧ x < 0 ∧ y < 0 → f x < f y) ∧ -- Monotonically increasing in (-∞, 0)
  (f x ∈ set.univ) → -- Range is ℝ
  (∃ (d : char), d = 'D') := 
by
  sorry

end correct_option_l421_421023


namespace radius_of_circle_polar_l421_421437

theorem radius_of_circle_polar (ρ θ : ℝ) (h : ρ = 2 * cos θ) :
  ∃ r, r = 1 :=
by
  sorry

end radius_of_circle_polar_l421_421437


namespace magnitude_problem_l421_421615

variables (a b : ℝ^3)
variables (angle_eq : real.angle a b = ⟨1/2, by norm_num⟩) -- (cos 60° = 1/2)
variables (norm_a : ∥a∥ = 2)
variables (norm_b : ∥b∥ = 1)

theorem magnitude_problem : ∥a - 2 • b∥ = 2 :=
sorry

end magnitude_problem_l421_421615


namespace average_shirts_sold_per_day_l421_421162

theorem average_shirts_sold_per_day (shirts_day1_morning shirts_day1_afternoon shirts_day2 : ℕ) 
  (h1 : shirts_day1_morning = 250) 
  (h2 : shirts_day1_afternoon = 20) 
  (h3 : shirts_day2 = 320) : 
  (shirts_day1_morning + shirts_day1_afternoon + shirts_day2) / 2 = 295 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end average_shirts_sold_per_day_l421_421162


namespace perimeter_triangle_ab_f2_area_triangle_ab_f2_l421_421993

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1
def f1 : ℝ × ℝ := (-real.sqrt 7, 0)
def f2 : ℝ × ℝ := (real.sqrt 7, 0)
def line_through_f1 (x y : ℝ) : Prop := y = x + real.sqrt 7
def A : ℝ × ℝ := sorry -- assuming coordinates are calculated as part of proof
def B : ℝ × ℝ := sorry -- assuming coordinates are calculated as part of proof

theorem perimeter_triangle_ab_f2 : 
  let AB := dist A B,
      AF2 := dist A f2, 
      BF2 := dist B f2 in
  AB + AF2 + BF2 = 16 := sorry

theorem area_triangle_ab_f2 : 
  let AB := dist A B in 
  let d := abs (f2.1 - f1.1) in
  0.5 * AB * d = 72 * real.sqrt 14 / 25 := sorry

end perimeter_triangle_ab_f2_area_triangle_ab_f2_l421_421993


namespace donna_paid_correct_amount_l421_421953

-- Define the original price, discount rate, and sales tax rate
def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.10

-- Define the total amount Donna paid
def total_amount_donna_paid : ℝ := 165

-- Define a theorem to express the proof problem
theorem donna_paid_correct_amount :
  let discount_amount := original_price * discount_rate in
  let sale_price := original_price - discount_amount in
  let sales_tax_amount := sale_price * sales_tax_rate in
  let total_amount := sale_price + sales_tax_amount in
  total_amount = total_amount_donna_paid :=
by
  sorry

end donna_paid_correct_amount_l421_421953


namespace intercept_sum_l421_421944

theorem intercept_sum (x y : ℝ) (h1 : y + 3 = -3 * (x - 5)) (hx : y = 0) (hy: x = 0) : 
  let x_intercept := x, y_intercept := y
  x_intercept + y_intercept = 16 := 
by
  -- Assuming conditions for x-intercept and y-intercept
  have hx_val : x_intercept = 4 := by
    rw [hy] at h1
    linarith
  
  have hy_val : y_intercept = 12 := by
    rw [hx] at h1
    linarith
  
  -- Proving the sum of intercepts
  rw [hx_val, hy_val]
  norm_num

end intercept_sum_l421_421944


namespace smallest_four_digit_palindrome_divisible_by_5_l421_421469

theorem smallest_four_digit_palindrome_divisible_by_5 :
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 5 = 0) ∧ (n = 5005 ∨ n = 5335 ∨ n = 5445 ∨ n = 5555 ∨ n = 5665 ∨ n = 5775 ∨ n = 5885 ∨ n = 5995) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m % 5 = 0) ∧ (m = 5005 ∨ m = 5335 ∨ m = 5445 ∨ m = 5555 ∨ m = 5665 ∨ m = 5775 ∨ m = 5885 ∨ m = 5995) → 5005 ≤ m :=
begin
  sorry
end

end smallest_four_digit_palindrome_divisible_by_5_l421_421469


namespace reinforcement_size_l421_421941

theorem reinforcement_size
    (garrison : ℕ)
    (initial_days : ℕ)
    (days_after : ℕ)
    (remaining_days : ℕ)
    (additional_days : ℕ)
    (initial_men : ℕ)
    (reinforcement_size : ℕ) :
    garrison = 2000 →
    initial_days = 54 →
    days_after = 15 →
    remaining_days = 20 →
    additional_days = 10 →
    initial_men = 2000 →
    let remaining_provisions := garrison * (initial_days - days_after) in
    let total_men := initial_men + reinforcement_size in
    remaining_provisions = total_men * remaining_days →
    reinforcement_size = 1900 := sorry

end reinforcement_size_l421_421941


namespace pete_bus_ride_blocks_l421_421368

theorem pete_bus_ride_blocks : 
  ∀ (total_walk_blocks bus_blocks total_blocks : ℕ), 
  total_walk_blocks = 10 → 
  total_blocks = 50 → 
  total_walk_blocks + 2 * bus_blocks = total_blocks → 
  bus_blocks = 20 :=
by
  intros total_walk_blocks bus_blocks total_blocks h1 h2 h3
  sorry

end pete_bus_ride_blocks_l421_421368


namespace correct_function_by_conditions_l421_421525

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)
noncomputable def fB (x : ℝ) : ℝ := sin x * cos x
noncomputable def fC (x : ℝ) : ℝ := abs (cos (2 * x))
noncomputable def fD (x : ℝ) : ℝ := sin (2 * x + π / 2)

-- Define the conditions
def is_symmetric_about_y (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x

-- Main theorem statement
theorem correct_function_by_conditions :
  (is_symmetric_about_y fD) ∧ (has_period fD π) ∧
  (¬ (is_symmetric_about_y fA ∧ has_period fA π)) ∧
  (¬ (is_symmetric_about_y fB ∧ has_period fB π)) ∧
  (¬ (is_symmetric_about_y fC ∧ has_period fC π)) :=
sorry

end correct_function_by_conditions_l421_421525


namespace reciprocal_sum_l421_421305

theorem reciprocal_sum (a b : ℝ) (h1 : a^2 + 2*a = 2) (h2 : b^2 + 2*b = 2) : 
  (1 / a + 1 / b = 1) ∨ 
  (1 / a + 1 / b = sqrt 3 + 1) ∨ 
  (1 / a + 1 / b = -sqrt 3 + 1) :=
sorry

end reciprocal_sum_l421_421305


namespace average_of_added_numbers_l421_421413

theorem average_of_added_numbers (sum_twelve : ℕ) (new_sum : ℕ) (x y z : ℕ) 
  (h_sum_twelve : sum_twelve = 12 * 45) 
  (h_new_sum : new_sum = 15 * 60) 
  (h_addition : x + y + z = new_sum - sum_twelve) : 
  (x + y + z) / 3 = 120 :=
by 
  sorry

end average_of_added_numbers_l421_421413


namespace compute_g_x_h_l421_421298

def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

theorem compute_g_x_h (x h : ℝ) : 
  g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end compute_g_x_h_l421_421298


namespace count_inverses_mod_21_l421_421640

theorem count_inverses_mod_21 : 
  (Finset.filter (λ n : ℕ, Nat.gcd n 21 = 1) (Finset.range 21)).card = 12 := 
by
  sorry

end count_inverses_mod_21_l421_421640


namespace circle_center_radius_l421_421415

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y - 4 = 0 → (1, -2) ∧ 3 :=
by {
  sorry
}

end circle_center_radius_l421_421415


namespace chord_length_l421_421496

-- Define the circle with center O and radius 15 units
def Circle (O : Point) (r : ℝ) : Prop := sorry

-- Define the chord CD which is the perpendicular bisector of the radius OA.
def isPerpendicularBisector (O A C D : Point) : Prop :=
  (OY == 15) ∧ (perpendicularBisector O A C D) ∧ (bisect OA C D)

-- If the chord CD perpendicularly bisects the radius OA
-- then the length of the chord is 26 * sqrt(1.1)
theorem chord_length (O A C D : Point) : 
  (Circle O 15) ∧ (isPerpendicularBisector O A C D) → length CD = 26 * sqrt(1.1) :=
  sorry

end chord_length_l421_421496


namespace Sum_S11_l421_421690

-- Define the arithmetic sequence and necessary conditions
def arithmetic_sequence (a d : ℕ → ℕ) : Prop :=
  ∀ n, a(n + 1) = a(n) + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  n * (a 1 + a n) / 2 

theorem Sum_S11 (a : ℕ → ℕ) (d : ℕ → ℕ) (S₁₁ : ℕ) :
    arithmetic_sequence a d →
    (a 4 + a 5 + a 6 + a 7 + a 8 = 150) →
    S a 11 = 330 :=
  by
    intro h_sequence h_sum
    sorry

end Sum_S11_l421_421690


namespace equilateral_triangle_area_l421_421404

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l421_421404


namespace problem_1_problem_2_l421_421623

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.ln x

theorem problem_1 :
  f (f (1 / 2)) = 1 / 2 :=
  sorry

theorem problem_2 :
  { x : ℝ | f (f x) = 1 } = {1, Real.exp Real.e} :=
  sorry

end problem_1_problem_2_l421_421623


namespace determine_angle_ACB_l421_421322

-- Assuming the geometrical context and necessary definitions are available.

variable {Point : Type} [AffineSpace ℝ Point]

variable {A B C D : Point}
variable {Angle : Type} [AngleSpace ℝ Angle]

-- Definitions for parallel lines and angles
variable (DC_AB_parallel : parallel Line_DC Line_AB)
variable (angle_DCA : Angle) (angle_ABC : Angle)
variable [isAngle (angle_DCA = 30)]
variable [isAngle (angle_ABC = 80)]
variable [isLineSegment (LineSegment A D) Line_DC]
variable [isLineSegment (LineSegment A B) Line_AB]
variable [isAngleInTriangle A B C angle_ABC angle_DCA]

theorem determine_angle_ACB (h1 : parallel Line_DC Line_AB)
    (h2 : isAngle (angle_DCA = 30))
    (h3 : isAngle (angle_ABC = 80)) :
    ∃ angle_ACB : Angle, angle_ACB = 70 :=
by
  sorry

end determine_angle_ACB_l421_421322


namespace find_age_l421_421920

theorem find_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 := 
by 
  sorry

end find_age_l421_421920


namespace line_does_not_intersect_staircase_l421_421051

/-- Define the line segment connecting (0, 0) to (2019, 2019) --/
def line_segment (x : ℝ) : ℝ := x

/-- Define the steps of the staircase --/
def staircase_step (i : ℤ) : set (ℝ × ℝ) :=
  { (x, y) : ℝ × ℝ | i - 1 ≤ x ∧ x < i ∧ y = i } ∪
  { (x, y) : ℝ × ℝ | i ≤ y ∧ y < i + 1 ∧ x = i }

def staircase := ⋃ i in (1 : ℕ)..2019, staircase_step i

theorem line_does_not_intersect_staircase :
  ¬ ∃ (x y : ℝ), (line_segment x = y) ∧ ((x, y) ∈ staircase) :=
sorry

end line_does_not_intersect_staircase_l421_421051


namespace equilateral_triangle_area_l421_421402

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l421_421402


namespace imo1965_cmo6511_l421_421567

theorem imo1965_cmo6511 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ∧
  |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ≤ Real.sqrt 2 ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
sorry

end imo1965_cmo6511_l421_421567


namespace possible_measures_of_angle_A_l421_421830

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l421_421830


namespace seven_a_plus_seven_b_l421_421352

noncomputable def g (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem seven_a_plus_seven_b (a b : ℝ) (h₁ : ∀ x, g x = f_inv x - 2) (h₂ : ∀ x, f_inv (f x a b) = x) :
  7 * a + 7 * b = 5 :=
by
  sorry

end seven_a_plus_seven_b_l421_421352


namespace calculation_equiv_l421_421174

theorem calculation_equiv :
  (-2: ℝ)^3 + real.sqrt 12 + (1/3: ℝ)⁻¹ = 2 * real.sqrt 3 - 5 :=
by sorry

end calculation_equiv_l421_421174


namespace vertex_of_parabola_l421_421075
-- Bring in the necessary library to handle basic algebra and real numbers

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 18

-- Define the vertex components p and q
def vertex_p (a b : ℝ) : ℝ := -b / (2 * a)
def vertex_q (f : ℝ → ℝ) (p : ℝ) : ℝ := f p

-- Problem statement in Lean 4
theorem vertex_of_parabola : 
  vertex_p 2 8 = -2 ∧ vertex_q parabola (-2) = 10 :=
by
  sorry

end vertex_of_parabola_l421_421075


namespace bishop_covers_all_light_squares_l421_421006

theorem bishop_covers_all_light_squares (n k : ℕ) (hn : n > 1) (hk : k > 1) :
  gcd (n - 1) (k - 1) = 1 ↔ bishop_covers_all_light_squares_on_n_k_chessboard n k :=
sorry

end bishop_covers_all_light_squares_l421_421006


namespace transformed_matrix_determinant_is_zero_l421_421719

-- Define the vectors and the determinant D
variables {ℝ : Type*} [field ℝ] [vector_space ℝ ℝ]

variables (a b c : ℝ × ℝ × ℝ)
variables (k : ℝ)

-- Assume the determinant of matrix with column vectors a, b, c is D
def determinant_abc : ℝ := (a.1 * (b.2 * c.3 - b.3 * c.2)) - (a.2 * (b.1 * c.3 - b.3 * c.1)) + (a.3 * (b.1 * c.2 - b.2 * c.1))

-- Definition for the modified matrix determinant
@[simp] def det_transformed_matrix : ℝ :=
  let v1 := (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3) in
  let v2 := (b.1 + c.1, b.2 + c.2, b.3 + c.3) in
  let v3 := (c.1 + k * a.1, c.2 + k * a.2, c.3 + k * a.3) in
  (v1.1 * (v2.2 * v3.3 - v2.3 * v3.2)) - (v1.2 * (v2.1 * v3.3 - v2.3 * v3.1)) + (v1.3 * (v2.1 * v3.2 - v2.2 * v3.1))
  
theorem transformed_matrix_determinant_is_zero (a b c : ℝ × ℝ × ℝ) (k D : ℝ) (h : determinant_abc a b c = D) :
  det_transformed_matrix a b c k = 0 := 
  sorry

end transformed_matrix_determinant_is_zero_l421_421719


namespace number_of_three_digit_integers_ending_in_5_l421_421643

theorem number_of_three_digit_integers_ending_in_5 :
  (number_of_integers (λ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
                                   (∀ i, i ∈ digits n → i > 4) ∧ 
                                   (n % 10 = 5))) = 25 := 
sorry

end number_of_three_digit_integers_ending_in_5_l421_421643


namespace boys_tried_out_l421_421872

theorem boys_tried_out (G B C N : ℕ) (hG : G = 9) (hC : C = 2) (hN : N = 21) (h : G + B - C = N) : B = 14 :=
by
  -- The proof is omitted, focusing only on stating the theorem
  sorry

end boys_tried_out_l421_421872


namespace rectangle_rotation_cylinder_l421_421118

theorem rectangle_rotation_cylinder (r : ℝ) (h : ℝ) :
  (rotate_around_side (rectangle r h) (side r)) = cylinder r h :=
sorry

end rectangle_rotation_cylinder_l421_421118


namespace product_of_two_numbers_l421_421052

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l421_421052


namespace factorization_identity_l421_421586

theorem factorization_identity (x : ℝ) : 
  3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 :=
by
  sorry

end factorization_identity_l421_421586


namespace sum_of_tangency_points_l421_421182

def f (x : ℝ) : ℝ := max (-7 * x - 15) (max (5 * x - 8) (3 * x + 10))

theorem sum_of_tangency_points (q : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (hq1 : q x1 = f x1) (hq2 : q x2 = f x2) (hq3 : q x3 = f x3)
  (hqt1 : ∀ x, x ≠ x1 → q x ≥ f x)
  (hqt2 : ∀ x, x ≠ x2 → q x ≥ f x)
  (hqt3 : ∀ x, x ≠ x3 → q x ≥ f x)
  (hx_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :
  x1 + x2 + x3 = -41 / 24 := 
sorry

end sum_of_tangency_points_l421_421182


namespace find_x_l421_421694

/--
Given the following conditions:
1. The sum of angles around a point is 360 degrees.
2. The angles are 7x, 6x, 3x, and (2x + y).
3. y = 2x.

Prove that x = 18 degrees.
-/
theorem find_x (x y : ℝ) (h : 18 * x + y = 360) (h_y : y = 2 * x) : x = 18 :=
by
  sorry

end find_x_l421_421694


namespace height_of_remaining_cube_l421_421499

theorem height_of_remaining_cube (a : ℝ) (h : a = 2) : 
  ∃ height:ℝ, height = 2 - real.sqrt 3 :=
by
  use 2 - real.sqrt 3
  sorry

end height_of_remaining_cube_l421_421499


namespace binom_15_4_l421_421987

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l421_421987


namespace cute_2020_all_integers_cute_l421_421100

-- Definition of "cute" integer
def is_cute (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), n = a^2 + b^3 + c^3 + d^5

-- Proof problem 1: Assert that 2020 is cute
theorem cute_2020 : is_cute 2020 :=
sorry

-- Proof problem 2: Assert that every integer is cute
theorem all_integers_cute (n : ℤ) : is_cute n :=
sorry

end cute_2020_all_integers_cute_l421_421100


namespace count_equilateral_triangles_l421_421151

theorem count_equilateral_triangles 
  (side_length_large : ℕ) 
  (side_length_small : ℕ) 
  (num_small_triangles : ℕ) 
  (num_parallel_lines : ℕ) 
  (total_equilateral_triangles : ℕ) 
  (h1 : side_length_large = 10) 
  (h2 : side_length_small = 1) 
  (h3 : num_small_triangles = 100) 
  (h4 : num_parallel_lines = side_length_large) 
  (h5 : total_equilateral_triangles = 200) :
  total_equilateral_triangles = ∑ i in (range num_parallel_lines).tail, (num_parallel_lines - i) := 
sorry

end count_equilateral_triangles_l421_421151


namespace PR_RQ_minimized_l421_421600

noncomputable def P := (-1, -2)
noncomputable def Q := (4, 2)
variable m : ℝ
noncomputable def R := (1, m)

theorem PR_RQ_minimized : m = -2/5 :=
by
  -- load the coordinates of the points
  have P_coords : ℝ × ℝ := P
  have Q_coords : ℝ × ℝ := Q
  have : P_coords.1 ≠ Q_coords.1 := by sorry -- (-1) ≠ 4

  -- calculate the slope of PQ
  let m_PQ := (Q.2 - P.2) / (Q.1 - P.1)

  -- find the equation of the line passing through P and Q
  let y := fun x => m_PQ * (x - P.1) + P.2

  -- verify that R lies on the line PQ
  have h : y R.1 = m := by
    calc
      y R.1 = m_PQ * (R.1 - P.1) + P.2 : by rfl
      ... = 4/5 * (1 + 1) - 2 : by sorry -- intermediate calculation steps
      ... = 8/5 - 2 : by rfl
      ... = 8/5 - 10/5 : by rfl
      ... = -2/5 : by sorry

  -- prove m = -2/5 
  exact h

end PR_RQ_minimized_l421_421600


namespace product_of_roots_of_quadratic_l421_421105

theorem product_of_roots_of_quadratic :
  ∀ (x : ℝ), (45 = -x^2 - 4x) → let y := x^2 + 4x - 45 in 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = -45 → y = 0 → (  
  (α β : ℝ), (a * α^2 + b * α + c = 0) ∧ (a * β^2 + b * β + c = 0) → α * β = -45 )

end product_of_roots_of_quadratic_l421_421105


namespace part1_part2_l421_421627

noncomputable def f (x : ℝ) (a : ℝ) := exp x - a * x

theorem part1 (a : ℝ) (h : deriv (λ x, f x a) 0 = -1) :
    a = 2 :=
begin
  change deriv (λ x, exp x - a * x) 0 = -1 at h,
  rw [deriv_sub, deriv_const_mul, deriv_exp, deriv_id'] at h,
  simp [-sub_eq_add_neg] at h,
  linarith,
end

theorem part2 : ∀ (x : ℝ), x > 0 → x^2 < exp x :=
begin
  intro x,
  intro hx_pos,
  have h : ∀ x > 0, ∃ c > 0, (λ y, exp y - y^2)' c > 0,
  {
    intros x hx,
    refine ⟨ln 4 / 2, div_pos (log_pos (four_pos)) zero_lt_two⟩,
  },
  sorry,
end

end part1_part2_l421_421627


namespace number_of_girls_l421_421856

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l421_421856


namespace solve_abs_eq_l421_421902

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l421_421902


namespace sum_mod_9237_9241_l421_421208

theorem sum_mod_9237_9241 :
  (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 :=
by
  sorry

end sum_mod_9237_9241_l421_421208


namespace equilateral_triangle_area_l421_421410

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l421_421410


namespace sum_remainder_is_zero_l421_421107

theorem sum_remainder_is_zero (a₁ d n : ℕ) (hn : n = 52) (an : 3 + (n - 1) * 6 = 309) :
  let S := n * (a₁ + an) / 2 in
  S % 6 = 0 :=
by sorry

end sum_remainder_is_zero_l421_421107


namespace task_completion_l421_421341

noncomputable def john_work_rate (J : ℝ) : ℝ := 1 / J
def jane_work_rate : ℝ := 1 / 12
def combined_work_rate (J : ℝ) : ℝ := john_work_rate J + jane_work_rate

theorem task_completion (J : ℝ) (h : combined_work_rate J * 6 + john_work_rate J * 4 = 1) : J = 20 :=
begin
  sorry
end

end task_completion_l421_421341


namespace four_digit_integers_with_repeated_digits_l421_421642

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l421_421642


namespace required_bandwidth_channel_l421_421930

theorem required_bandwidth_channel
  (session_duration_min : ℕ)
  (sampling_rate : ℕ)
  (sampling_depth : ℕ)
  (metadata_volume : ℕ)
  (audio_per_kibibit : ℕ)
  (stereo_factor : ℕ) :
  session_duration_min = 51 →
  sampling_rate = 63 →
  sampling_depth = 17 →
  metadata_volume = 47 →
  audio_per_kibibit = 5 →
  stereo_factor = 2 →
  let time_in_seconds := session_duration_min * 60 in
  let data_volume := sampling_rate * sampling_depth * time_in_seconds in
  let metadata_bits := metadata_volume * 8 in
  let metadata_volume_bits := (metadata_bits * data_volume) / (audio_per_kibibit * 1024) in
  let total_data_volume := (data_volume + metadata_volume_bits) * stereo_factor in
  (total_data_volume / 1024) / time_in_seconds = 2.25 :=
sorry

end required_bandwidth_channel_l421_421930


namespace root_of_function_l421_421227

-- Problem conditions
def f (x : ℝ) : ℝ := 2 - Real.logb 2 x

-- The statement we want to prove
theorem root_of_function (a : ℝ) (h : f a = 0) : a = 4 := by
  sorry

end root_of_function_l421_421227


namespace f_3_eq_4_l421_421649

noncomputable def f : ℝ → ℝ := sorry

theorem f_3_eq_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 3 = 4 :=
by
  sorry

end f_3_eq_4_l421_421649


namespace find_roots_l421_421202

noncomputable def polynomial_roots : set ℝ :=
  {((1 - Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 - Real.sqrt 43 - 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 - 2 * Real.sqrt 34) / 6)}

theorem find_roots (x : ℝ) :
  (3 * x ^ 4 + 2 * x ^ 3 - 8 * x ^ 2 + 2 * x + 3 = 0) ↔ (x ∈ polynomial_roots) :=
by sorry

end find_roots_l421_421202


namespace increase_in_deductibles_next_year_l421_421529

def average_family_deductible_current : ℝ := 3000
def increase_fraction : ℝ := 2/3

theorem increase_in_deductibles_next_year : increase_fraction * average_family_deductible_current = 2000 :=
by
  sorry

end increase_in_deductibles_next_year_l421_421529


namespace equilateral_triangle_area_l421_421401

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l421_421401


namespace brothers_meeting_time_l421_421064

theorem brothers_meeting_time :
  ∀ (S : ℝ), 
  (older_brother_time younger_brother_time : ℝ) (delay : ℝ),
  older_brother_time = 12 →
  younger_brother_time = 20 →
  delay = 5 →
  let younger_brother_speed = S / younger_brother_time in
  let older_brother_speed = S / older_brother_time in
  let relative_speed = older_brother_speed - younger_brother_speed in
  let distance_between_at_start = younger_brother_speed * delay in
  let time_to_meet = distance_between_at_start / relative_speed in
  time_to_meet + delay = 12.5 :=
λ S older_brother_time younger_brother_time delay 
  hob hyb hdelay younger_brother_speed older_brother_speed
  relative_speed distance_between_at_start time_to_meet,
  by
    rw [hob, hyb, hdelay] at *
    let younger_brother_speed := S / younger_brother_time
    let older_brother_speed := S / older_brother_time
    let relative_speed := older_brother_speed - younger_brother_speed
    let distance_between_at_start := younger_brother_speed * delay
    let time_to_meet := distance_between_at_start / relative_speed
    sorry

end brothers_meeting_time_l421_421064


namespace win_sector_area_l421_421497

-- Definitions and conditions
def radius : ℝ := 15
def probability_of_winning : ℝ := 3 / 8
def circle_area : ℝ := π * radius ^ 2

-- Statement of the proof problem
theorem win_sector_area :
  (probability_of_winning * circle_area) = (675 * π / 8) :=
  by sorry

end win_sector_area_l421_421497


namespace AE_six_l421_421541

namespace MathProof

-- Definitions of the given conditions
variables {A B C D E : Type}
variables (AB CD AC AE : ℝ)
variables (triangleAED_area triangleBEC_area : ℝ)

-- Given conditions
def conditions : Prop := 
  convex_quadrilateral A B C D ∧
  AB = 9 ∧
  CD = 12 ∧
  AC = 14 ∧
  intersect_at E AC BD ∧
  areas_equal triangleAED_area triangleBEC_area

-- Theorem to prove AE = 6
theorem AE_six (h : conditions AB CD AC AE triangleAED_area triangleBEC_area) : 
  AE = 6 :=
by sorry  -- proof omitted

end MathProof

end AE_six_l421_421541


namespace dog_water_per_hour_l421_421458

theorem dog_water_per_hour (violet_water_per_hour : ℕ) (total_water_ml : ℕ) 
  (hiking_hours : ℕ) (violet_water_needs : ℕ) (dog_water_per_hour : ℕ) :
  violet_water_per_hour = 800 → total_water_ml = 4800 → 
  hiking_hours = 4 → violet_water_needs = violet_water_per_hour * hiking_hours → 
  dog_water_per_hour = (total_water_ml - violet_water_needs) / hiking_hours → 
  dog_water_per_hour = 400 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  rw mul_comm at h4
  sorry

end dog_water_per_hour_l421_421458


namespace triangle_side_ratio_l421_421661

variables (a b c S : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
    and given a=1, B=π/4, and the area S=2, we prove that b / sin(B) = 5√2. -/
theorem triangle_side_ratio (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : b / Real.sin B = 5 * Real.sqrt 2 :=
sorry

end triangle_side_ratio_l421_421661


namespace find_c_l421_421493

-- Define the conditions of the problem
variables {a b c : ℝ}
def parabola_eq (y : ℝ) : ℝ := a * y^2 + b * y + c

-- Define the vertex of the parabola
def vertex_condition (x_v y_v : ℝ) : Prop := x_v = parabola_eq y_v

-- Define the passing point condition
def passing_point_condition (x_p y_p : ℝ) : Prop := x_p = parabola_eq y_p

theorem find_c (h_v : vertex_condition (-3) 1) (h_p : passing_point_condition (-1) 3) : c = -5/2 :=
by
  sorry

end find_c_l421_421493


namespace oleg_older_than_wife_ekaterina_l421_421027

-- Define the four individuals
constants (Roman Oleg Ekaterina Zhanna : Type)

-- Define age as a relationship
constant age : Roman → Oleg → Ekaterina → Zhanna → Prop

-- Each person has a different age
axiom different_ages : ∀ r o e z : Prop, r ≠ o ∧ r ≠ e ∧ r ≠ z ∧ o ≠ e ∧ o ≠ z ∧ e ≠ z

-- Each married man is older than his wife
axiom roman_older_than_wife : ∀ r z : Prop, r > z → r = Ekaterina
axiom oleg_older_than_wife : ∀ o e : Prop, o > e → o = Zhanna

-- Zhanna is older than Oleg
axiom zhanna_older_than_oleg : ∀ z o : Prop, z > o

-- We need to prove that: Oleg is older than his wife Ekaterina
theorem oleg_older_than_wife_ekaterina : age o e → o > e :=
sorry

end oleg_older_than_wife_ekaterina_l421_421027


namespace range_of_y_over_x_l421_421232

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (h : abs (x - 2 + y * complex.I) = sqrt 3)

theorem range_of_y_over_x : -sqrt 3 ≤ y / x ∧ y / x ≤ sqrt 3 := by
  sorry

end range_of_y_over_x_l421_421232


namespace number_of_girls_l421_421851

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l421_421851


namespace area_of_rectangle_ABCD_l421_421237

noncomputable def point := (ℕ, ℕ)
noncomputable def rightTriangle (A B C : point) : Prop :=
  (A = (0, 0)) ∧ (B = (3, 0)) ∧ (C = (3, 4)) ∧ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2)

noncomputable def rectangle (A B C D : point) : Prop := 
  rightTriangle A B C ∧ 
  (D.1 - A.1 = 0) ∧
  (D.2 - A.2 = C.2) ∧
  (B.1 - A.1 = C.1 - D.1) ∧
  (B.2 - A.2 = C.2 - D.2)

theorem area_of_rectangle_ABCD (A B C D : point) 
  (h1 : A = (0,0)) 
  (h2 : B = (3,0)) 
  (h3 : C = (3,4)) 
  (h4 : rectangle A B C D) : 
  (abs (B.1 - A.1)) * (abs (C.2 - A.2)) = 12 :=
by
  sorry

end area_of_rectangle_ABCD_l421_421237


namespace algebraic_expression_multiplied_l421_421457

theorem algebraic_expression_multiplied (n k : ℕ) (h : n = k) :
    (∏ i in finset.range (n + 1) \ finset.range 1, (n + i)) = (2^n * ∏ j in finset.range n, (2 * j + 1)) ->
    ((∏ i in finset.range ((k+1) + 1) \ finset.range 1, ((k + 1) + i)) / (∏ i in finset.range (k + 1) \ finset.range 1, (k + i))) = 2 * (2*k + 1) :=
by
  sorry 

end algebraic_expression_multiplied_l421_421457


namespace triangle_pentagon_ratio_l421_421169

theorem triangle_pentagon_ratio (p : ℝ) (h1 : p = 60) :
  let triangle_side_length := p / 3,
      pentagon_side_length := p / 5
  in triangle_side_length / pentagon_side_length = 5 / 3 :=
by
  sorry

end triangle_pentagon_ratio_l421_421169


namespace calories_remaining_for_dinner_l421_421559

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l421_421559


namespace min_distance_to_line_value_of_AB_l421_421319

noncomputable def point_B : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 4)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def polar_line_l (a : ℝ) (θ : ℝ) : ℝ :=
  a * Real.cos (θ - Real.pi / 4)

noncomputable def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + y + m = 0

theorem min_distance_to_line {θ : ℝ} (a : ℝ) :
  polar_line_l a θ = 4 * Real.sqrt 2 → 
  ∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 :=
by
  sorry

theorem value_of_AB :
  ∃ AB, AB = 12 * Real.sqrt 2 / 7 :=
by
  sorry

end min_distance_to_line_value_of_AB_l421_421319


namespace f_iterated_result_l421_421717

def f (x : ℕ) : ℕ :=
  if Even x then 3 * x / 2 else 2 * x + 1

theorem f_iterated_result : f (f (f (f 1))) = 31 := by
  sorry

end f_iterated_result_l421_421717


namespace find_x_l421_421893

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l421_421893


namespace cos_B_l421_421632

-- Definition of the given conditions.
def a_b (a b : ℝ) : Prop := a = b
def b_squared_eq_3ac (a b c : ℝ) : Prop := b^2 = 3 * a * c

-- Statement to prove: If a = b and b^2 = 3ac, then cos B = 1/6.
theorem cos_B (a b c : ℝ) (h1 : a_b a b) (h2 : b_squared_eq_3ac a b c) :
  real.cos (real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 1 / 6 := by
  sorry

end cos_B_l421_421632


namespace arithmetic_sequence_property_l421_421264

variable {a : ℕ → ℝ} -- Let a be an arithmetic sequence
variable {S : ℕ → ℝ} -- Let S be the sum of the first n terms of the sequence

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + (n - 1) * (a 2 - a 1) / 2)
axiom a_5 : a 5 = 3
axiom S_13 : S 13 = 91

-- Question to prove
theorem arithmetic_sequence_property : a 1 + a 11 = 10 :=
by
  sorry

end arithmetic_sequence_property_l421_421264


namespace vector_exists_l421_421536

-- Define the given conditions as stated in the problem
def parametric_vector (t : ℝ) : ℝ × ℝ := (3 * t + 3, 2 * t + 3)

-- Define the vector with the correct answer
def target_vector : ℝ × ℝ := (9, 6)

-- State the theorem that proves the question given the conditions
theorem vector_exists (t : ℝ) :
  let k := t + 1 in
  (3 * k, 2 * k) = target_vector :=
by
  sorry

end vector_exists_l421_421536
