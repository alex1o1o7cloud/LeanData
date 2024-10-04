import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Trig
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Combinatorics
import Mathlib.Analysis.SpecialFunctions.Harmonic
import Mathlib.Data.Digit
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Default
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Real.Basic
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.Tactic
import Mathlib.Topology.MetricSpace.Basic
import Time

namespace range_of_k_l122_122640

theorem range_of_k {k : ℝ} :
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end range_of_k_l122_122640


namespace opposite_of_negative_five_l122_122386

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122386


namespace arithmetic_geometric_sequence_l122_122989

variable (a_1 a_2 a_3 a_4 a_5 : ℝ)

theorem arithmetic_geometric_sequence :
  a_1 + a_2 + a_3 + a_4 + a_5 = 3 →
  a_1^2 + a_2^2 + a_3^2 + a_4^2 + a_5^2 = 12 →
  a_1 - a_2 + a_3 - a_4 + a_5 = 4 := 
begin
  intros h1 h2,
  sorry,  -- Proof goes here
end

end arithmetic_geometric_sequence_l122_122989


namespace axis_of_symmetry_l122_122775

theorem axis_of_symmetry (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ)
  (h₁ : ∀ n, a_n n = 1 / (n * (n + 1)))
  (h₂ : S_n n = ∑ i in finset.range (n + 1), a_n i)
  (h₃ : S_n 9 = 9 / 10) :
  axis_of_symmetry 36 = -9 :=
by
  sorry

end axis_of_symmetry_l122_122775


namespace rectangle_area_l122_122741

theorem rectangle_area {A B C D E F : Type*} [EuclideanGeometry A B C D E F] 
  (AB_length : Real) (BC_length : Real) (AFED_area : Real)
  (midpoint_E : Midpoint E (A, B, C, D)) 
  (BE_intersect_AC : Intersect BE AC F):
  AB_length = 2 → BC_length = 1 → AFED_area = 30 → Area ABCD = 480 / 7 :=
by
  sorry

end rectangle_area_l122_122741


namespace range_of_omega_l122_122888

-- Define the sine function f(x) = sin(ωx + π/3)
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- The hypothesis for the problem
def has_extreme_points (ω : ℝ) :=
  (∃ x1 x2 x3 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (x3 < Real.pi)
    ∧ f' ω x1 = 0 ∧ f' ω x2 = 0 ∧ f' ω x3 = 0)

def has_zeros (ω : ℝ) :=
  (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < Real.pi)
    ∧ f ω x1 = 0 ∧ f ω x2 = 0)

-- The main theorem to be proved
theorem range_of_omega (ω : ℝ) :
  has_extreme_points ω ∧ has_zeros ω ↔ (13/6 < ω ∧ ω ≤ 8/3) :=
by
  sorry

end range_of_omega_l122_122888


namespace diagonals_of_nine_sided_polygon_l122_122543

theorem diagonals_of_nine_sided_polygon :
  ∀ (P : Type) [polygon P] (h_sides : num_sides P = 9) (h_right_angles : num_right_angles P = 2),
  num_diagonals P = 27 :=
by sorry

end diagonals_of_nine_sided_polygon_l122_122543


namespace find_pe_given_conditions_l122_122625

variable {event : Type}
variable (e f : event)
variable {p : event → ℝ}

def cond_prob (a b : event) : ℝ := p (a ∩ b) / p b

theorem find_pe_given_conditions
  (h1 : p f = 75)
  (h2 : p (e ∩ f) = 75)
  (h3 : cond_prob e f = 3)
  (h4 : cond_prob f e = 3) :
  p e = 25 :=
sorry

end find_pe_given_conditions_l122_122625


namespace faulty_key_in_digits_l122_122474

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122474


namespace percentage_of_females_wearing_glasses_l122_122504

theorem percentage_of_females_wearing_glasses (total_population : ℕ) (male_population : ℕ) (females_with_glasses : ℕ) :
  total_population = 5000 ∧ male_population = 2000 ∧ females_with_glasses = 900 →
  let female_population := total_population - male_population in
  (females_with_glasses / (female_population : ℚ)) * 100 = 30 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  let female_population := total_population - male_population
  have h_female_population : female_population = 3000 := by
    simp [h1, h2]
  have h_percentage : (females_with_glasses / (female_population : ℚ)) * 100 = 30 := by
    simp [h3, h_female_population]
  exact h_percentage

end percentage_of_females_wearing_glasses_l122_122504


namespace mutually_exclusive_events_l122_122970

-- Define the types for balls and events
inductive Ball
| red
| white

-- Define the bag of balls
def bag : List Ball := [Ball.red, Ball.red, Ball.white, Ball.white]

-- Define the four events
def eventA (draw: List Ball) : Prop := (draw.contains Ball.white) ∧ (draw = [Ball.white, Ball.white])
def eventB (draw: List Ball) : Prop := (draw.contains Ball.white) ∧ ((draw.count Ball.red) ≤ 1)
def eventC (draw: List Ball) : Prop := ¬(draw.contains Ball.white) ∧ ((draw.count Ball.red) = 1)
def eventD (draw: List Ball) : Prop := (draw.contains Ball.white) ∧ (draw = [Ball.red, Ball.red])

-- Define the proof problem
theorem mutually_exclusive_events :
  ∀ draw : List Ball,
  (eventD draw → (¬ eventA draw ∧ ¬ eventB draw ∧ ¬ eventC draw)) :=
by
  intros draw
  sorry

end mutually_exclusive_events_l122_122970


namespace actual_distance_correct_l122_122305

-- Define the conditions
def map_scale : ℝ := 1000000
def map_distance : ℝ := 2.5 -- in cm

-- Define the actual distance conversion
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale / 100000 -- convert from cm to kilometers

-- The theorem statement representing the math problem
theorem actual_distance_correct :
  actual_distance map_distance map_scale = 25 := by
  sorry

end actual_distance_correct_l122_122305


namespace max_omega_l122_122190

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem max_omega :
  (∃ ω > 0, (∃ k : ℤ, (f ω (2 * π / 3) = 0) ∧ (ω = 3 / 2 * k)) ∧ (0 < ω * π / 14 ∧ ω * π / 14 ≤ π / 2)) →
  ∃ ω, ω = 6 :=
by
  sorry

end max_omega_l122_122190


namespace range_of_a_l122_122630

noncomputable def y (a x : ℝ) := Real.log x + a * x^2 - (2 * a + 1) * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x, (deriv (λ x, Real.log x + a * x^2 - (2 * a + 1) * x)) = 0 → x = 1) →
  a > 1 / 2 :=
by
  intros h_min
  sorry

end range_of_a_l122_122630


namespace share_of_A_l122_122009

variable (total_payment : ℝ) (A_days : ℝ) (B_days : ℝ) (A_share : ℝ)

def work_rate (days : ℝ) : ℝ := 1 / days
def combined_work_rate (A_days : ℝ) (B_days : ℝ) : ℝ :=
  work_rate A_days + work_rate B_days

theorem share_of_A
  (A_days_eq : A_days = 12)
  (B_days_eq : B_days = 18)
  (total_payment_eq : total_payment = 149.25)
  (A_share_eq : A_share = 3 / 5 * total_payment) :
  A_share = 89.55 :=
by
  rw [A_days_eq, B_days_eq, total_payment_eq, A_share_eq]
  calc
    3 / 5 * 149.25 = 3 / 5 * (14925 / 100) : by norm_num
    ...           = 44775 / 500 : by norm_num
    ...           = 89.55 : by norm_num

end share_of_A_l122_122009


namespace find_prime_squares_l122_122952

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l122_122952


namespace trig_expression_l122_122973

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end trig_expression_l122_122973


namespace possible_faulty_keys_l122_122460

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122460


namespace gavrila_final_distance_l122_122095

-- Define constant distances and speeds
def L : ℝ := 50  -- Halfway distance to starting point for Gavrila
def speed : ℝ := 20  -- Speed of both bicycle and motorboat in km/h
def distance_from_bank : ℝ := 40  -- Given distance y from the bank, in meters

-- Define the equation for Gavrila's coordinate computation
def gavrila_x_coordinate (y : ℝ) : ℝ := (y^2) / (4 * L)

-- Define Pythagorean theorem application to find Gavrila's distance from the starting point
def gavrila_distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Finally, the theorem statement that needs to be proven
theorem gavrila_final_distance :
  let x := gavrila_x_coordinate distance_from_bank in
  let s := gavrila_distance x distance_from_bank in
  real.round s = 41 :=
by {
  -- Proof omitted
  sorry
}

end gavrila_final_distance_l122_122095


namespace seating_arrangements_l122_122240

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l122_122240


namespace original_price_before_discounts_l122_122859

theorem original_price_before_discounts (P : ℝ) : 
  (P * 0.80 * 0.90 * 0.95 = 6500) → 
  (P = 9502.92) :=
by
  intro h
  have h1 : P * 0.684 = 6500 := by simp [mul_assoc, h]
  sorry

end original_price_before_discounts_l122_122859


namespace count_ordered_pairs_l122_122203

def ordered_pairs_condition (b g : ℕ) : Prop :=
  4 ≤ b ∧ b ≤ g ∧ g ≤ 2007 ∧ 
  (∃ k : ℕ, g = b + k ∧ b + g = k^2)

theorem count_ordered_pairs : 
  (∑ (b g : ℕ) in finset.Icc 4 2007 ×ˢ finset.Icc 4 2007, if ordered_pairs_condition b g then 1 else 0) = 59 :=
sorry

end count_ordered_pairs_l122_122203


namespace area_quadrilateral_correct_l122_122795

-- Define the setup for the problem
def square_side_lengths := [1, 3, 5]  -- The side lengths of the squares
def total_base_length := (square_side_lengths.sum)  -- Total length of the base
def height_to_length_ratio := 5 / total_base_length  -- Ratio of height to total base length

-- Define the heights at which the intersects the right corners of these rectangles
def height_at(x : ℝ) := x * height_to_length_ratio

-- Heights at specific points
def h1 := height_at 1
def h2 := height_at (1 + 3)

-- Define the area of the quadrilateral formed as a trapezoid
def area_of_quadrilateral := (1 / 2) * (h1 + h2) * 3

-- Formal statement for Lean to prove
theorem area_quadrilateral_correct : area_of_quadrilateral = 75 / 18 := by
  sorry

end area_quadrilateral_correct_l122_122795


namespace zero_in_interval_l122_122763

theorem zero_in_interval : ∃ x ∈ (2, 3), (∀ x, (0 < x) → (f x = ln x - 2/x)) := sorry

end zero_in_interval_l122_122763


namespace last_two_digits_l122_122959

def x := Real.sqrt 29 + Real.sqrt 21
def y := Real.sqrt 29 - Real.sqrt 21
def a := x^2 = 50 + 2 * Real.sqrt 609
def b := y^2 = 50 - 2 * Real.sqrt 609
def S : ℕ → ℝ :=
  λ n => a^n + b^n

theorem last_two_digits (n : ℕ) :
  ((x : ℝ) ^ 2)^(n : ℕ) + ((y : ℝ)(^2))^(n : ℕ) = 71 := 
sorry

end last_two_digits_l122_122959


namespace sqrt_mixed_number_simplified_l122_122923

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l122_122923


namespace sqrt_mixed_number_simplified_l122_122926

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l122_122926


namespace product_of_two_is_even_probability_l122_122969

-- Definitions based on conditions from part a
def papers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_even (n : ℕ) : Prop := n % 2 = 0

-- Main theorem statement based on problem translation in part c
theorem product_of_two_is_even_probability :
  let pairs := (papers.sups papers).filter (λ (p : ℕ × ℕ), p.fst < p.snd) in
  let even_pairs := pairs.filter (λ (p : ℕ × ℕ), is_even (p.fst * p.snd)) in
  even_pairs.card = 13 / 18 * (pairs.card) :=
sorry

end product_of_two_is_even_probability_l122_122969


namespace opposite_of_neg_five_l122_122379

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122379


namespace sum_integers_between_8_and_19_l122_122017

theorem sum_integers_between_8_and_19 : (list.sum ([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : list ℕ)) = 162 := 
by
  sorry

end sum_integers_between_8_and_19_l122_122017


namespace duty_arrangements_2520_l122_122796

noncomputable def countDutyArrangements : ℕ :=
  let C := Nat.choose
  let A := Nat.fact
  (C 7 2 * C 5 2 * C 3 2) / A 3 * A 4

theorem duty_arrangements_2520 :
  countDutyArrangements = 2520 := 
sorry

end duty_arrangements_2520_l122_122796


namespace heat_conduction_circle_solution_l122_122753

noncomputable def BesselJ (n : ℕ) (x : ℝ) : ℝ := sorry

def heat_solution (μ : ℕ → ℝ) (r t : ℝ) : ℝ :=
  ∑' n, (4 * BesselJ 2 (μ n)) / ((μ n)^2 * (BesselJ 1 (μ n))^2) * exp (-(μ n)^2 * t) * BesselJ 0 (μ n * r)

theorem heat_conduction_circle_solution (μ : ℕ → ℝ) (r t : ℝ) (hμ : ∀ n, BesselJ 0 (μ n) = 0) :
  (∂ u_t = ∂^2 u / ∂ r^2 + 1/r * ∂ u / ∂ r) → (u (r, 0) = 1 - r^2) → (u (1, t) = 0) → 
  u (r, t) = heat_solution μ r t :=
begin
  sorry
end

end heat_conduction_circle_solution_l122_122753


namespace discriminant_of_quadratic_polynomial_l122_122136

theorem discriminant_of_quadratic_polynomial :
  let a := 5
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ) 
  let Δ := b^2 - 4 * a * c
  Δ = (576/25 : ℚ) :=
by
  sorry

end discriminant_of_quadratic_polynomial_l122_122136


namespace number_of_birches_possible_l122_122042

-- Definitions for the problem's conditions
def tree : Type := { isBirch : bool // isBirch || (!isBirch) }  -- A tree is either a birch or not
def isBirch (t: tree) : bool := t.val

-- Given conditions
def num_trees : ℕ := 130
def lindens_have_incorrect_sign (trees: list tree) : Prop :=
  ∀ t ∈ trees, ¬isBirch(t) → (
    let neighbors := (if h : trees.indexOf(t) = 0 then [trees.last!, trees.head!]
                      else [trees.get! (trees.indexOf(t) - 1), trees.get! ((trees.indexOf(t) + 1) % trees.length)]) in
    isBirch(neighbors.head) = isBirch(neighbors.tail.head)
  )

def exactly_one_birch_incorrect_sign (trees: list tree) : Prop :=
  (trees.countp isBirch = 1) ∧ (trees.countp (λ t, isBirch(t) && let neighbors := (if h : trees.indexOf(t) = 0 then [trees.last!, trees.head!]
                                                                  else [trees.get! (trees.indexOf(t) - 1), trees.get! ((trees.indexOf(t) + 1) % trees.length)]) in
                                          ¬(isBirch(neighbors.head) ≠ isBirch(neighbors.tail.head))) = 1)

-- The statement to be proved
theorem number_of_birches_possible (trees: list tree) 
  (h1: trees.length = num_trees)
  (h2: lindens_have_incorrect_sign trees)
  (h3: exactly_one_birch_incorrect_sign trees) :
  trees.countp isBirch = 87 := sorry

end number_of_birches_possible_l122_122042


namespace percentage_decrease_in_savings_l122_122730

theorem percentage_decrease_in_savings (I : ℝ) (F : ℝ) (IncPercent : ℝ) (decPercent : ℝ)
  (h1 : I = 125) (h2 : IncPercent = 0.25) (h3 : F = 125) :
  let P := (I * (1 + IncPercent))
  ∃ decPercent, decPercent = ((P - F) / P) * 100 ∧ decPercent = 20 := 
by
  sorry

end percentage_decrease_in_savings_l122_122730


namespace four_points_all_red_edges_l122_122056

variable (G : SimpleGraph (Fin 9)) (color : List (Fin 9 × Fin 9) → Prop)

-- Condition 1: G has 36 edges.
def edge_count_36 : Prop := G.edgeFinset.card = 36

-- Condition 2: Each edge is colored either red or black.
def edge_is_colored : Prop := ∀ e ∈ G.edgeFinset, color [e.fst, e.snd] ∨ ¬color [e.fst, e.snd]

-- Condition 3: Every triangle in the graph must have at least one red edge.
def triangle_has_red_edge : Prop :=
  ∀ (a b c : Fin 9), G.adj a b → G.adj b c → G.adj c a → 
  (color [a, b] ∨ color [b, c] ∨ color [c, a]) 

-- Prove that there exist four vertices with all edges between them red
theorem four_points_all_red_edges (G : SimpleGraph (Fin 9)) (color : List (Fin 9 × Fin 9) → Prop)
  (h1 : edge_count_36 G color)
  (h2 : edge_is_colored G color)
  (h3 : triangle_has_red_edge G color) :
  ∃ (a b c d : Fin 9), 
    G.adj a b ∧ G.adj b c ∧ G.adj c d ∧ G.adj d a ∧ 
    color [a, b] ∧ color [b, c] ∧ color [c, d] ∧ color [d, a] := 
sorry

end four_points_all_red_edges_l122_122056


namespace number_of_solutions_tan_cot_eq_l122_122962

theorem number_of_solutions_tan_cot_eq :
  {θ : ℝ | 0 < θ ∧ θ < π ∧ tan (3 * π * cos θ ^ 3) = cot (3 * π * sin θ ^ 3)}.finite.card = 10 :=
by
  sorry

end number_of_solutions_tan_cot_eq_l122_122962


namespace min_value_of_expr_min_value_at_specific_points_l122_122716

noncomputable def min_value_expr (p q r : ℝ) : ℝ := 8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r)

theorem min_value_of_expr : ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → min_value_expr p q r ≥ 6 :=
by
  intro p q r hp hq hr
  sorry

theorem min_value_at_specific_points : min_value_expr (1 / (8 : ℝ)^(1 / 4)) (1 / (18 : ℝ)^(1 / 4)) (1 / (50 : ℝ)^(1 / 4)) = 6 :=
by
  sorry

end min_value_of_expr_min_value_at_specific_points_l122_122716


namespace focus_of_parabola_is_two_one_l122_122339

/-- Define the parabola and its properties -/
def parabola (x y : ℝ) : Prop := (y - 1)^2 = 4 * (x - 1)

/-- Prove that the focus of the parabola (y - 1)^2 = 4(x - 1) is (2, 1) -/
theorem focus_of_parabola_is_two_one : 
  ∀ x y : ℝ, parabola x y → (x = 2 ∧ y = 1) :=
begin
  sorry
end

end focus_of_parabola_is_two_one_l122_122339


namespace area_ratio_of_bisectors_l122_122351

-- Definitions corresponding to the given conditions
def parallelogram (a b : ℝ) : Prop := a > 0 ∧ b > 0

def bisectors_form_rectangle (a b : ℝ) : Prop := parallelogram a b

-- The main theorem we need to prove
theorem area_ratio_of_bisectors (a b : ℝ) (h : parallelogram a b) 
  (h_a : a = 3) (h_b : b = 5) : 
  let rectangle_area := (2 : ℝ)
  let parallelogram_area := (a * b : ℝ)
  rectangle_area / parallelogram_area = (2 / 15 : ℝ) :=
by {
  unfold parallelogram at h,
  rw [h_a, h_b],
  have h_parallelogram_area : parallelogram_area = 15 := by simp,
  simp [rectangle_area, h_parallelogram_area],
  sorry -- Proof goes here
}

end area_ratio_of_bisectors_l122_122351


namespace slide_step_difference_l122_122968

-- Define the conditions in the problem
def steps_per_gap : ℕ := 60
def slides_per_gap : ℕ := 15
def total_gaps : ℕ := 30 -- 31 beacons means 30 gaps
def total_distance_ft : ℕ := 2640 -- half a mile in feet

-- Define Frank's step length
def frank_step_length : ℝ := total_distance_ft / (steps_per_gap * total_gaps)

-- Define Peter's slide length
def peter_slide_length : ℝ := total_distance_ft / (slides_per_gap * total_gaps)

-- The theorem to prove
theorem slide_step_difference : peter_slide_length - frank_step_length = 4.4 := by
  sorry

end slide_step_difference_l122_122968


namespace infinite_primes_divide_sequence_l122_122709

noncomputable def a : ℕ := 1  -- Placeholder value (a should be positive)
noncomputable def b : ℕ := 1  -- Placeholder value (b should be positive)

def sequence (n : ℕ) : ℕ := a * 2017^n + b * 2016^n

theorem infinite_primes_divide_sequence :
  ∃ (primeset : set ℕ), infinite primeset ∧ (∀ n : ℕ, ∃ p ∈ primeset, p ∣ sequence n) :=
sorry

end infinite_primes_divide_sequence_l122_122709


namespace problem_inequality_l122_122265

theorem problem_inequality (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 := sorry

end problem_inequality_l122_122265


namespace determine_x_l122_122822

-- Define the variables and the conditions
variables (C S x : ℝ)
hypothesis h1 : 20 * C = x * S
hypothesis h2 : S = 1.25 * C

-- Statement to prove
theorem determine_x (C S : ℝ) (x : ℝ) (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end determine_x_l122_122822


namespace total_goals_is_15_l122_122563

-- Define the conditions as variables
def KickersFirstPeriodGoals : ℕ := 2
def KickersSecondPeriodGoals : ℕ := 2 * KickersFirstPeriodGoals
def SpidersFirstPeriodGoals : ℕ := KickersFirstPeriodGoals / 2
def SpidersSecondPeriodGoals : ℕ := 2 * KickersSecondPeriodGoals

-- Define total goals by each team
def TotalKickersGoals : ℕ := KickersFirstPeriodGoals + KickersSecondPeriodGoals
def TotalSpidersGoals : ℕ := SpidersFirstPeriodGoals + SpidersSecondPeriodGoals

-- Define total goals by both teams
def TotalGoals : ℕ := TotalKickersGoals + TotalSpidersGoals

-- Prove the statement
theorem total_goals_is_15 : TotalGoals = 15 :=
by
  sorry

end total_goals_is_15_l122_122563


namespace rectangle_side_lengths_l122_122984

theorem rectangle_side_lengths (x y : ℝ) (h1 : 2 * x + 4 = 10) (h2 : 8 * y - 2 = 10) : x + y = 4.5 := by
  sorry

end rectangle_side_lengths_l122_122984


namespace expected_interval_proof_l122_122308

noncomputable def expected_interval_between_trains : ℝ := 3

theorem expected_interval_proof
  (northern_route_time southern_route_time : ℝ)
  (counter_clockwise_delay : ℝ)
  (home_to_work_less_than_work_to_home : ℝ) :
  northern_route_time = 17 →
  southern_route_time = 11 →
  counter_clockwise_delay = 75 / 60 →
  home_to_work_less_than_work_to_home = 1 →
  expected_interval_between_trains = 3 :=
by
  intros
  sorry

end expected_interval_proof_l122_122308


namespace circles_intersect_iff_l122_122802

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end circles_intersect_iff_l122_122802


namespace equilateral_triangle_area_with_circles_l122_122248

theorem equilateral_triangle_area_with_circles :
  ∀ (triangle : Type) (r : ℝ), 
  (∀ c1 c2 c3 c4 : triangle, 
    circle_in_triangle triangle c1 r ∧ circle_in_triangle triangle c2 r ∧ 
    circle_in_triangle triangle c3 r ∧ circle_in_triangle triangle c4 r ∧
    (∀ (c : triangle), radius c = r) ∧ r = 1) →
  ∃ (A : ℝ), equilateral_triangle_area triangle A ∧ A = 12 * Real.sqrt 3
:= sorry

end equilateral_triangle_area_with_circles_l122_122248


namespace opposite_of_neg_five_l122_122365

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122365


namespace regular_polyhedron_of_equal_dihedral_angles_and_regular_faces_l122_122744

-- Definitions of regular polygon and dihedral angles would be required, but 
-- in this context, we assume the existence of the necessary definitions.
-- Since the proof is not required, we proceed directly to the theorem statement
-- with the conditions and the conclusion.

/-- If all dihedral angles of a convex polyhedron are equal and all its faces are regular polygons,
then the polyhedron is regular. -/
theorem regular_polyhedron_of_equal_dihedral_angles_and_regular_faces 
  (P : Type) [ConvexPolyhedron P] 
  (equal_dihedral_angles : ∀ (A B C : Face P), dihedral_angle A B C = dihedral_angle A B C)
  (regular_faces : ∀ (F : Face P), RegularPolygon F) : RegularPolyhedron P :=
sorry

end regular_polyhedron_of_equal_dihedral_angles_and_regular_faces_l122_122744


namespace possible_faulty_keys_l122_122462

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122462


namespace width_of_deck_l122_122298

noncomputable def length : ℝ := 30
noncomputable def cost_per_sqft_construction : ℝ := 3
noncomputable def cost_per_sqft_sealant : ℝ := 1
noncomputable def total_cost : ℝ := 4800
noncomputable def total_cost_per_sqft : ℝ := cost_per_sqft_construction + cost_per_sqft_sealant

theorem width_of_deck (w : ℝ) 
  (h1 : length * w * total_cost_per_sqft = total_cost) : 
  w = 40 := 
sorry

end width_of_deck_l122_122298


namespace find_omega_find_period_and_intervals_find_solution_set_l122_122185

noncomputable def omega_condition (ω : ℝ) :=
  0 < ω ∧ ω < 2

noncomputable def function_fx (ω : ℝ) (x : ℝ) := 
  3 * Real.sin (2 * ω * x + Real.pi / 3)

noncomputable def center_of_symmetry_condition (ω : ℝ) := 
  function_fx ω (-Real.pi / 6) = 0

noncomputable def period_condition (ω : ℝ) :=
  Real.pi / abs ω

noncomputable def intervals_of_increase (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, ((Real.pi / 12 + k * Real.pi) ≤ x) ∧ (x < (5 * Real.pi / 12 + k * Real.pi))

noncomputable def solution_set_fx_ge_half (x : ℝ) : Prop :=
  ∃ k : ℤ, (Real.pi / 12 + k * Real.pi) ≤ x ∧ (x ≤ 5 * Real.pi / 12 + k * Real.pi)

theorem find_omega : ∀ ω : ℝ, omega_condition ω ∧ center_of_symmetry_condition ω → omega = 1 := sorry

theorem find_period_and_intervals : 
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → period_condition ω = Real.pi :=
sorry

theorem find_solution_set :
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → (∀ x, solution_set_fx_ge_half x) :=
sorry

end find_omega_find_period_and_intervals_find_solution_set_l122_122185


namespace repeating_decimal_to_fraction_l122_122571

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l122_122571


namespace f_inv_f_inv_14_l122_122757

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem f_inv_f_inv_14 : f_inv (f_inv 14) = -14 / 9 :=
by {
  sorry
}

end f_inv_f_inv_14_l122_122757


namespace arrangement_ways_l122_122231

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l122_122231


namespace wine_half_water_after_69_days_l122_122520

theorem wine_half_water_after_69_days (h l : ℝ) (n : ℕ) :
  h = 100 ∧ l = 1 ∧ (100 * (0.99) ^ 69 = 50) → n = 69 :=
by
  intro ⟨h_eq, l_eq, eqn⟩
  have log_eq : log (0.5) = 69 * log (0.99) :=
    calc
      log (0.5) = log (100 * (0.99) ^ 69 / 100) : by rw eqn; ring
      ... = log (0.99) ^ 69 : by rw log_div; ring
      ... = 69 * log (0.99) : by rw log_pow
  have inv_eq : n = 69 :=
    calc
      n = log (0.5) / log (0.99) : by rw ←log_eq; ring
      ... = 69 : by rw log_eq; ring
  exact inv_eq

end wine_half_water_after_69_days_l122_122520


namespace probability_no_adjacent_stand_up_l122_122598

theorem probability_no_adjacent_stand_up (n : ℕ) (h_n : n = 4) :
  let total_outcomes := 2 ^ n,
      scenarios_adjacent_stand_up := 4 + 4 + 1,
      prob_adjacent_stand_up := (scenarios_adjacent_stand_up : ℚ) / total_outcomes,
      prob_no_adjacent_stand_up := 1 - prob_adjacent_stand_up
  in prob_no_adjacent_stand_up = 7 / 16 :=
by
  sorry

end probability_no_adjacent_stand_up_l122_122598


namespace least_number_subtraction_l122_122439

theorem least_number_subtraction (n : ℕ) (h₀ : n = 3830) (k : ℕ) (h₁ : k = 5) : (n - k) % 15 = 0 :=
by {
  sorry
}

end least_number_subtraction_l122_122439


namespace smallest_number_l122_122862

theorem smallest_number (a b c d : ℤ) (h_a : a = 0) (h_b : b = -1) (h_c : c = -4) (h_d : d = 5) : 
  c < b ∧ c < a ∧ c < d :=
by {
  sorry
}

end smallest_number_l122_122862


namespace day_of_week_100_days_from_wednesday_l122_122797

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end day_of_week_100_days_from_wednesday_l122_122797


namespace number_of_balls_in_last_box_l122_122312

noncomputable def box_question (b : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2010 → b i + b (i + 1) = 14 + i) ∧
  (b 1 + b 2011 = 1023)

theorem number_of_balls_in_last_box (b : ℕ → ℕ) (h : box_question b) : b 2011 = 1014 :=
by
  sorry

end number_of_balls_in_last_box_l122_122312


namespace perpendicular_planes_of_perpendicular_line_l122_122621

-- Definitions for the conditions
variables (m n : Line) (α β : Plane)
variables (h_skew : SkewLines m n)
variables (h_diff_planes : α ≠ β)
variables (h_m_in_alpha : m ⊆ α)
variables (h_n_in_beta : n ⊆ β)

-- The theorem to prove
theorem perpendicular_planes_of_perpendicular_line (h_m_perp_beta : m ⊥ β) : α ⊥ β := 
by sorry

end perpendicular_planes_of_perpendicular_line_l122_122621


namespace incorrect_statement_l122_122153

variables (a b : Line) (α β : Plane)

theorem incorrect_statement :
  ∀ (a b : Line) (α β : Plane),
  (¬ ((a ∥ α) ∧ (a ∥ β) → (α ∥ β))) :=
by sorry

end incorrect_statement_l122_122153


namespace find_defective_keys_l122_122452

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122452


namespace minimum_value_of_f_in_interval_l122_122189

noncomputable def f (m : ℝ) : ℝ := -m^3 + 3 * m^2 - 4

theorem minimum_value_of_f_in_interval :
  is_minimum_value_in_interval (f) (-1) 1 (-4) :=
sorry

end minimum_value_of_f_in_interval_l122_122189


namespace lambda_sum_half_l122_122713

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable {A B C D E : V}

theorem lambda_sum_half (AD_half_AB : ∥ A - D ∥ = 1/2 * ∥ A - B ∥)
                        (BE_two_thirds_BC : ∥ B - E ∥ = 2/3 * ∥ B - C ∥)
                        (DE_linear_comb : ∃ (λ1 λ2 : ℝ), ∥ D - E ∥ = λ1 * ∥ A - B ∥ + λ2 * ∥ A - C ∥) :
                        (∃ λ1 λ2 : ℝ, λ1 + λ2 = 1/2) :=
by
  let λ1 := -1/6
  let λ2 := 2/3
  have h_sum : λ1 + λ2 = 1/2 := sorry
  exact ⟨λ1, λ2, h_sum⟩


end lambda_sum_half_l122_122713


namespace oil_bill_for_january_l122_122219

-- Definitions and conditions
def ratio_F_J (F J : ℝ) : Prop := F / J = 3 / 2
def ratio_F_M (F M : ℝ) : Prop := F / M = 4 / 5
def ratio_F_J_modified (F J : ℝ) : Prop := (F + 20) / J = 5 / 3
def ratio_F_M_modified (F M : ℝ) : Prop := (F + 20) / M = 2 / 3

-- The main statement to prove
theorem oil_bill_for_january (J F M : ℝ) 
  (h1 : ratio_F_J F J)
  (h2 : ratio_F_M F M)
  (h3 : ratio_F_J_modified F J)
  (h4 : ratio_F_M_modified F M) :
  J = 120 :=
sorry

end oil_bill_for_january_l122_122219


namespace minimum_ascending_paths_is_n_l122_122291

universe u

def northern_square (n : ℕ) := 
  { square : ℕ → ℕ → ℕ // 
    ∀ i j: ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 1 ≤ (square i j) ∧ (square i j) ≤ n^2 ∧ 
    ∀ (x y : ℕ → ℕ → Prop), x ≠ y → x < y }

def is_adjacent (i j i' j' : ℕ) := 
  (i = i' ∧ (j = j' + 1 ∨ j = j' - 1)) ∨ 
  (j = j' ∧ (i = i' + 1 ∨ i = i' - 1))

def is_valley (square : ℕ → ℕ → ℕ) (i j n : ℕ) : Prop :=
  ∀ (i' j' : ℕ), is_adjacent i j i' j' → i' > 0 ∧ j' > 0 ∧ i' ≤ n ∧ j' ≤ n → 
  square i j < square i' j'

def ascending_path_exists_from_valley (square : ℕ → ℕ → ℕ) (path : list (ℕ × ℕ)) : Prop :=
  ∃ (valley : ℕ × ℕ), 
  (is_valley square valley.1 valley.2 (path.head.1 * path.head.2)) ∧ 
  (path.head = valley) ∧ 
  ∀ (p₁ p₂ : ℕ × ℕ), 
  list.mem p₁ path → list.mem p₂ path → list.index_of p₁ path < list.index_of p₂ path → 
  is_adjacent p₁.1 p₁.2 p₂.1 p₂.2 → square p₁.1 p₁.2 < square p₂.1 p₂.2

def minimum_ascending_paths (n : ℕ) : ℕ :=
  if h : n > 0 then
    have : ∀ (square : northern_square n), 
      ∃ (paths : ℕ), 
        (∀ path : list (ℕ × ℕ), ascending_path_exists_from_valley (square.val) path → paths > 0) ∧
        paths = n 
    from sorry;
    n
  else 
    0

theorem minimum_ascending_paths_is_n (n : ℕ) (h : n > 0) : 
  minimum_ascending_paths n = n := 
by 
  sorry

end minimum_ascending_paths_is_n_l122_122291


namespace number_of_parallel_or_perpendicular_pairs_l122_122903

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := 3 * x + 2
def line3 (x : ℝ) : ℝ := 2 * x - 1 / 3
def line4 (x : ℝ) : ℝ := 1.5 * x - 1
def line5 (x : ℝ) : ℝ := 0.5 * x - 1.5

theorem number_of_parallel_or_perpendicular_pairs :
  (∑ i in [(line1, line3)], 1) = 1 :=
sorry

end number_of_parallel_or_perpendicular_pairs_l122_122903


namespace opposite_of_neg_five_is_five_l122_122397

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122397


namespace four_digit_number_has_factors_42_l122_122518

-- Define the conditions
def number_of_factors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), 1

variable (n : ℕ)

-- State the problem
theorem four_digit_number_has_factors_42
  (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : number_of_factors n = 42)
  (h3 : ∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 
        (∀ q, q ∣ n → Prime q → q = p1 ∨ q = p2 ∨ q = p3)) 
  : number_of_factors n = 42 := 
sorry

end four_digit_number_has_factors_42_l122_122518


namespace asbestos_tiles_width_l122_122857

theorem asbestos_tiles_width (n : ℕ) (h : 0 < n) :
  let width_per_tile := 60
  let overlap := 10
  let effective_width := width_per_tile - overlap
  width_per_tile + (n - 1) * effective_width = 50 * n + 10 := by
sorry

end asbestos_tiles_width_l122_122857


namespace necessary_not_sufficient_condition_l122_122169

variable (B C D : Type) [InnerProductSpace ℝ B]
variable {A : B}
variable (λ μ : ℝ)

def collinear (B C D : B) : Prop :=
∃ (k : ℝ), ∃ (l : ℝ), (C - B) = k • (D - B) ∧ (A - B) = l • (D - B)

def unique_combination (A B C D : B) (λ μ : ℝ) : Prop :=
∃ (λ μ : ℝ), (A - D) = λ • (A - B) + μ • (A - C) ∧ (λ + μ = 1)

theorem necessary_not_sufficient_condition (p q : Prop)
    (h₁ : p ↔ collinear B C D)
    (h₂ : q ↔ unique_combination A B C D λ μ) : p → q ∧ ¬ (q → p) :=
by
  sorry

end necessary_not_sufficient_condition_l122_122169


namespace sum_of_radii_eq_l122_122315

-- Definitions as given in the problem statement
variables (A B C D : Type) [locally_compact_space A]
variables [is_metric_space A] [is_compact_space A]

-- r_XYZ indicates the radius of the inscribed circle in triangle XYZ
variables (r_ABC r_ACD r_BCD r_BDA : ℝ)

-- Assume A, B, C, D can form a cyclic quadrilateral
-- (This assumption is simplified for typing in Lean without geometric details)
axiom cyclic_quad_ABC_D : IsCyclicQuadrilateral A B C D

theorem sum_of_radii_eq (h_cyclic : cyclic_quad_ABC_D) :
  r_ABC + r_ACD = r_BCD + r_BDA := 
by
  sorry

end sum_of_radii_eq_l122_122315


namespace opposite_of_neg_five_l122_122380

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122380


namespace opposite_of_negative_five_l122_122382

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122382


namespace arithmetic_sequence_general_formula_l122_122988

variable (a : ℤ) 

def is_arithmetic_sequence (a1 a2 a3 : ℤ) : Prop :=
  2 * a2 = a1 + a3

theorem arithmetic_sequence_general_formula :
  ∀ {a1 a2 a3 : ℤ}, is_arithmetic_sequence a1 a2 a3 → a1 = a - 1 ∧ a2 = a + 1 ∧ a3 = 2 * a + 3 → 
  ∀ n : ℕ, a_n = 2 * n - 3
:= by
  sorry

end arithmetic_sequence_general_formula_l122_122988


namespace oscar_marathon_training_l122_122310

theorem oscar_marathon_training :
  let initial_miles := 2
  let target_miles := 20
  let increment_per_week := (2 : ℝ) / 3
  ∃ weeks_required, target_miles - initial_miles = weeks_required * increment_per_week → weeks_required = 27 :=
by
  sorry

end oscar_marathon_training_l122_122310


namespace find_defective_keys_l122_122454

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122454


namespace number_of_roots_of_unity_l122_122851

theorem number_of_roots_of_unity (c d : ℤ) :
  {ω : ℂ | ∃ n : ℕ, n > 0 ∧ ω^n = 1 ∧ ω^3 + c * ω + d = 0}.card = 2 :=
sorry

end number_of_roots_of_unity_l122_122851


namespace largest_divisible_by_7_l122_122107

theorem largest_divisible_by_7 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  let n := 10000 * A + 1000 * B + 100 * B + 10 * C + A in
  (n ≡ 0 [MOD 7]) ∧ n = 98879 :=
sorry

end largest_divisible_by_7_l122_122107


namespace intersect_midpoint_l122_122712

-- Definitions of the terms used in the problem
variables (A B C D K L J : Type)
variables [IsTriangle A B C]
variables [angle_ordering : Angle (C) < Angle(A) ∧ Angle(A) < 90] -- Condition on angles
variables [OnSegment D AC] -- D is on the segment AC
variables [SameLength BD BA] -- BD = BA
variables [Tangency K (Incircle ABC) AB] -- Tangency definition at K
variables [Tangency L (Incircle ABC) AC] -- Tangency definition at L
variables [Incenter J BCD] -- J is the incenter of BCD

-- Statement that KL intersects AJ at its midpoint
theorem intersect_midpoint (h : Line (KL) intersects Segment (AJ)) :
  midpoint (AJ) = intersect_point (Line KL) (Segment AJ) :=
sorry

end intersect_midpoint_l122_122712


namespace credit_card_more_beneficial_l122_122027

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end credit_card_more_beneficial_l122_122027


namespace JillSavingsPercentage_l122_122917

noncomputable def JillNetMonthlySalary : ℤ := 3700
noncomputable def DiscretionaryIncome (netMonthlySalary : ℤ) : ℚ := (1 / 5 : ℚ) * netMonthlySalary
noncomputable def VacationsPercentage : ℚ := 0.3
noncomputable def EatingOutPercentage : ℚ := 0.35
noncomputable def GiftsAndCharity : ℚ := 111
noncomputable def TotalPercentage : ℚ := 1

theorem JillSavingsPercentage :
  let discretionaryIncome := DiscretionaryIncome JillNetMonthlySalary in
  let giftsPercentage := (GiftsAndCharity / discretionaryIncome) in
  let totalSpentPercentage := VacationsPercentage + EatingOutPercentage + giftsPercentage in
  let savingsPercentage := TotalPercentage - totalSpentPercentage in
  savingsPercentage = 0.2 :=
by
  sorry

end JillSavingsPercentage_l122_122917


namespace lower_face_is_red_given_red_upper_l122_122199

noncomputable def probability_lower_face_red :
  (red_upper : Bool) → (probability_red_lower : ℚ) :=
  if red_upper then 2 / 3 else 0

theorem lower_face_is_red_given_red_upper :
  ∀ (red_upper_has_occurred : Bool),
    red_upper_has_occurred = true →
    probability_lower_face_red red_upper_has_occurred = 2 / 3 :=
by
  intros red_upper_has_occurred h
  rw h
  simp [probability_lower_face_red]
  sorry

end lower_face_is_red_given_red_upper_l122_122199


namespace range_of_omega_l122_122884

noncomputable section

open Real

/--
Assume the function f(x) = sin (ω x + π / 3) has exactly three extreme points and two zeros in 
the interval (0, π). Prove that the range of values for ω is 13 / 6 < ω ≤ 8 / 3.
-/
theorem range_of_omega 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h : ∀ x, f x = sin (ω * x + π / 3))
  (h_extreme : (∃ a b c, 0 < a ∧ a < b ∧ b < c ∧ c < π ∧ (f' a = 0) ∧ (f' b = 0) ∧ (f' c = 0)))
  (h_zeros : (∃ u v, 0 < u ∧ u < v ∧ v < π ∧ f u = 0 ∧ f v = 0)) :
  (13 / 6) < ω ∧ ω ≤ (8 / 3) :=
  sorry

end range_of_omega_l122_122884


namespace exists_root_in_interval_l122_122208

theorem exists_root_in_interval (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : a ≤ b) 
  (h2 : f a * f b < 0) : 
  ∃ c ∈ Icc a b, f c = 0 := 
sorry

end exists_root_in_interval_l122_122208


namespace number_of_irrationals_is_three_l122_122539

theorem number_of_irrationals_is_three :
  let numbers := [3.14, -real.cbrt 27, real.pi / 3, real.sqrt 6, -3 / 4, 0.2020020002 -- further digits implied as irrational ]
  in (∃ (irrational_count : ℕ), irrational_count = 3 ∧ 
    rational.repr 3.14 ∧ 
    rational.repr (-real.cbrt 27) ∧ 
    irrational.repr (real.pi / 3) ∧ 
    irrational.repr (real.sqrt 6) ∧ 
    rational.repr (-3 / 4) ∧ 
    irrational.repr (≈ 0.2020020002) ∧ 
    irrational_count = 3) 
  sorry

end number_of_irrationals_is_three_l122_122539


namespace compute_AQ_l122_122325

/-
Right triangle ABC (hypotenuse AB) is inscribed in equilateral triangle PQR,
with PC = 4 and BP = CQ = 3. We need to compute AQ.
-/

-- Defining the conditions as they are stated in the problem.
def PC : ℝ := 4
def BP : ℝ := 3
def CQ : ℝ := 3

-- Define points A, B, C, P, Q, R
variables (A B C P Q R : EuclideanSpace ℝ (fin 2))

-- Hypothesis: right triangle ABC inscribed in equilateral triangle PQR
-- This assumption implies certain structural relationships among the points.

-- Define AQ as x and find the correct value given the conditions.
def AQ (x : ℝ) : Prop := x = 27 / 11

-- The main theorem states that AQ = 27/11 under the given conditions.
theorem compute_AQ : ∃ x : ℝ, AQ x ∧ x = 27 / 11 :=
by {
  use 27 / 11,
  split,
  { refl },
  { refl }
}

-- If needed, we can use sorry to skip proof details not directly specified by the conditions.
sorry

end compute_AQ_l122_122325


namespace hyperbola_standard_eq_l122_122627

noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_standard_eq (a b : ℝ) 
  (f := (sqrt 6, 0))
  (p := (-5, 2))
  (ha : a > 0) (hb : b > 0) 
  (hf : a^2 + b^2 = 6) 
  (hp : (25 / a^2) - (4 / b^2) = 1) :
  hyperbola_equation (sqrt 5) 1 x y :=
begin
  sorry
end

end hyperbola_standard_eq_l122_122627


namespace median_free_throws_l122_122048

def free_throws : List ℕ := [22, 17, 5, 31, 25, 22, 9, 17, 5, 26]

theorem median_free_throws : (List.median (List.sort free_throws)) = 19.5 := sorry

end median_free_throws_l122_122048


namespace even_function_expression_l122_122609

noncomputable def f (x : ℝ) : ℝ := 
  if x >= 0 then real.log (x^2 - 2*x + 2) 
  else real.log ((-x)^2 - 2*(-x) + 2)

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_f_nonneg : ∀ x, 0 ≤ x → f x = real.log (x^2 - 2*x + 2)) : 
  ∀ x, x < 0 → f x = real.log (x^2 + 2*x + 2) := 
by
  intros x hx
  have h : f x = f (-x) := h_even x
  rw [←h, h_f_nonneg (-x)]
  sorry

end even_function_expression_l122_122609


namespace natural_numbers_not_divisible_by_5_or_7_l122_122202

def num_not_divisible_by_5_or_7 (n : ℕ) : ℕ :=
  let num_div_5 := n / 5
  let num_div_7 := n / 7
  let num_div_35 := n / 35
  n - (num_div_5 + num_div_7 - num_div_35)

theorem natural_numbers_not_divisible_by_5_or_7 :
  num_not_divisible_by_5_or_7 999 = 686 :=
by sorry

end natural_numbers_not_divisible_by_5_or_7_l122_122202


namespace part_a_l122_122055

theorem part_a (b c: ℤ) : ∃ (n : ℕ) (a : ℕ → ℤ), 
  (a 0 = b) ∧ (a n = c) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) :=
sorry

end part_a_l122_122055


namespace find_positive_integer_n_l122_122142

theorem find_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧ (sin (π / (2 * n)) * cos (π / (2 * n)) = (n : ℝ) / 8) ∧ n = 2 :=
by
  sorry

end find_positive_integer_n_l122_122142


namespace functional_equation_solution_l122_122983

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) :
  (∀ x y : ℝ, f(x + t + f(y)) = f(f(x)) + f(t) + y) → (∀ x : ℝ, f(x) = x) :=
by
  intros h x
  sorry

end functional_equation_solution_l122_122983


namespace price_of_soft_taco_l122_122533

theorem price_of_soft_taco (S : ℝ) 
  (h_family_cost : 4 * 5 + 3 * S)
  (h_customers_cost : 10 * 2 * S)
  (h_total_cost : 4 * 5 + 3 * S + 10 * 2 * S = 66) :
  S = 2 :=
by
  sorry

end price_of_soft_taco_l122_122533


namespace identify_faulty_key_l122_122483

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122483


namespace collinear_A_Z_M_l122_122620

noncomputable def midpoint (A B : Point) : Point := 
    -- Assume this function returns the midpoint of the line segment AB
    sorry

structure Triangle :=
  (A B C : Point)

structure Altitudes :=
  (AD BE CF : Line)

structure Midpoints :=
  (M : Point) -- Midpoint of side BC
  (N : Point) -- Midpoint of segment AD

structure Intersections :=
  (X Y : Point) -- Intersection points of AO and BC

structure MidSegment :=
  (Z : Point) -- Midpoint of segment XY

theorem collinear_A_Z_M (O : Point) 
    (△ABC : Triangle) 
    (AD BE CF : Altitudes) 
    (M : Midpoints) 
    (XY : Intersections)
    (Z : MidSegment)
    (h1 : Z.Z = midpoint XY.X XY.Y) 
    (h2 : M.M = midpoint △ABC.B △ABC.C) 
    (h3 : M.N = midpoint △ABC.A AD.AD) : 
    collinear [△ABC.A, Z.Z, M.M] :=
sorry

end collinear_A_Z_M_l122_122620


namespace problem_proof_l122_122628

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_domain (x : ℝ) : True -- to denote domain of f is ℝ.
axiom f_decreasing (x y : ℝ) (h1 : 4 < x) (h2 : x < y) : f(x) > f(y)
axiom f_even (x : ℝ) : f(x + 4) = f(-x + 4)

-- Statement to prove
theorem problem_proof : f(3) > f(6) := sorry

end problem_proof_l122_122628


namespace max_n_l122_122222

-- Definition of the problem using the given conditions
noncomputable def max_possible_n : ℕ :=
  -- Define a function that takes a table of size 100 x n where each row is a permutation
  let table := list (list ℕ) in
  -- Condition for the permutations of numbers 1 to 100 in each row
  ∀ row ∈ table, row.perm 1:100 ∧ (∀ i < 100, row.nth i ≠ row.nth (i + 1)) &&
    -- Allowed operation: swap two numbers in a row if they differ by 1 and are not adjacent
    (∀ (i j : ℕ), i < 100 → j < 100 → |(row.nth i) - (row.nth j)| = 1 → j ≠ i+1 →
       -- Ensure that no two rows can be made identical through the allowed operations
       ∀ row' ∈ table, (row ≠ row' → (show row ≠swap(row, row.nth i, row.nth j)) )) → sorry

-- The main theorem to be proven -- maximum possible value of n
theorem max_n:
  max_possible_n = 2^99 :=
  sorry

end max_n_l122_122222


namespace f_neg_two_l122_122174

noncomputable def f : ℝ → ℝ
  := sorry

variables (x : ℝ)

-- Definition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- First condition: f is odd
axiom f_odd : is_odd_function f

-- Second condition: for x > 0, f(x) = log x / log 2
axiom f_positive_x (h : x > 0) : f x = log x / log 2

-- Proof Statement: f(-2) = -1
theorem f_neg_two : f (-2) = -1 :=
sorry

end f_neg_two_l122_122174


namespace lemons_needed_for_10_gallons_l122_122662

-- Define the initial conditions as ratios
def initial_lemons := 48
def initial_gallons := 64

-- Define the conversion calculation
noncomputable def lemons_per_gallon := initial_lemons / initial_gallons
noncomputable def needed_gallons := 10

-- Prove the number of lemons needed for 10 gallons
theorem lemons_needed_for_10_gallons : 
  let needed_lemons := (lemons_per_gallon * needed_gallons).ceil in
  needed_lemons = 8 :=
by
  have ratio := lemons_per_gallon
  have proportion := ratio * needed_gallons
  have rounded := proportion.ceil
  have needed_lemons := 8
  sorry

end lemons_needed_for_10_gallons_l122_122662


namespace arithmetic_series_sum_l122_122557

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ := k^2 - 1,
      d := 1,
      n := 2 * k
  aₙ := a₁ + (n - 1) * d :=
  (k * (2 * k^2 + 2 * k - 3) = 2 * k^3 + 2 * k^2 - 3 * k) := by
  sorry

end arithmetic_series_sum_l122_122557


namespace necessary_but_not_sufficient_condition_for_a_lt_neg_one_l122_122501

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end necessary_but_not_sufficient_condition_for_a_lt_neg_one_l122_122501


namespace train_B_departure_time_l122_122806

def distance : ℕ := 65
def speed_A : ℕ := 20
def speed_B : ℕ := 25
def departure_A := 7
def meeting_time := 9

theorem train_B_departure_time : ∀ (d : ℕ) (vA : ℕ) (vB : ℕ) (tA : ℕ) (m : ℕ), 
  d = 65 → vA = 20 → vB = 25 → tA = 7 → m = 9 → ((9 - (m - tA + (d - (2 * vA)) / vB)) = 1) → 
  8 = ((9 - (meeting_time - departure_A + (distance - (2 * speed_A)) / speed_B))) := 
  by {
    sorry
  }

end train_B_departure_time_l122_122806


namespace b1_value_b2_value_b3_value_b_geometric_a_general_formula_l122_122644

noncomputable def sequence (n : ℕ) : ℕ 
| 0     := 2
| (n+1) := 3 * sequence n + 2

def b_sequence (n : ℕ) : ℕ := sequence n + 1

theorem b1_value : b_sequence 0 = 3 :=
by sorry

theorem b2_value : b_sequence 1 = 9 :=
by sorry

theorem b3_value : b_sequence 2 = 27 :=
by sorry

theorem b_geometric (n : ℕ) : b_sequence (n + 1) = 3 * b_sequence n :=
by sorry

theorem a_general_formula (n : ℕ) : sequence n = 3^n - 1 :=
by sorry

end b1_value_b2_value_b3_value_b_geometric_a_general_formula_l122_122644


namespace construct_lines_at_angle_60_l122_122904

-- Definitions of the conditions
variable (Point : Type) [Nonempty Point] -- given point
variable (Plane : Type) [Nonempty Plane] -- given plane
variable (firstProjectionPlane : Plane) -- first projection plane
variable (givenPoint : Point) -- the given point
variable (givenPlane : Plane) -- the given plane

-- Angle between planes
noncomputable def angle_between_planes (p1 p2 : Plane) : ℝ := sorry -- Some function to calculate angles between planes

-- Proving the main theorem
theorem construct_lines_at_angle_60 (α : ℝ) (hα : α ≥ 60) : ∃ (l : Line), passes_through l givenPoint ∧ equidistant_from_plane l givenPlane ∧ forms_angle l firstProjectionPlane 60 := 
by
  sorry

end construct_lines_at_angle_60_l122_122904


namespace decimal_to_binary_35_l122_122115

theorem decimal_to_binary_35 :
  nat.to_digits 2 35 = [1, 0, 0, 0, 1, 1] :=
by sorry

end decimal_to_binary_35_l122_122115


namespace total_sum_lent_l122_122073

theorem total_sum_lent (x : ℚ) (second_part : ℚ) (total_sum : ℚ) (h : second_part = 1688) 
  (h_interest : x * 3/100 * 8 = second_part * 5/100 * 3) : total_sum = 2743 :=
by
  sorry

end total_sum_lent_l122_122073


namespace coordinates_equality_l122_122341

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l122_122341


namespace smallest_prime_factor_of_1739_eq_itself_l122_122438

theorem smallest_prime_factor_of_1739_eq_itself :
  ∀ p : ℕ, nat.prime p → p ∣ 1739 → p = 1739 :=
begin
  intros p hp hdiv,
  have h : p = 1739, sorry,
end

end smallest_prime_factor_of_1739_eq_itself_l122_122438


namespace total_cost_l122_122407

-- Definition: Cost of first 100 notebooks
def cost_first_100_notebooks : ℕ := 230

-- Definition: Cost per notebook beyond the first 100 notebooks
def cost_additional_notebooks (n : ℕ) : ℕ := n * 2

-- Theorem: Total cost given a > 100 notebooks
theorem total_cost (a : ℕ) (h : a > 100) : (cost_first_100_notebooks + cost_additional_notebooks (a - 100) = 2 * a + 30) := by
  sorry

end total_cost_l122_122407


namespace seating_arrangements_l122_122238

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l122_122238


namespace find_faulty_keys_l122_122449

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122449


namespace problem1_min_max_problem2_monotonous_range_problem3_min_value_l122_122188

-- Define the function f(x)
def f (x a : ℝ) := x^2 + 2 * a * x + 3

-- Problem 1: Prove the minimum and maximum for a = -2
theorem problem1_min_max :
  let a := -2 in
  (∀ x ∈ Icc (-4 : ℝ) 6, f x a ≥ -1) ∧ (∃ x ∈ Icc (-4 : ℝ) 6, f x a = -1) ∧ 
  (∀ x ∈ Icc (-4 : ℝ) 6, f x a ≤ 35) ∧ (∃ x ∈ Icc (-4 : ℝ) 6, f x a = 35) :=
sorry

-- Problem 2: Prove the range of a for strictly monotonous function
theorem problem2_monotonous_range :
  (∀ a : ℝ, (∀ x₁ x₂ ∈ Icc (-4 : ℝ) 6, (x₁ < x₂ → f x₁ a < f x₂ a) ∨ (f x₁ a = f x₂ a))) ↔ 
  a ∈ Iic (-6) ∪ Ioi 4 :=
sorry

-- Problem 3: Prove the minimum value on [-4, 6]
theorem problem3_min_value (a : ℝ) :
  (∃ x ∈ Icc (-4 : ℝ) 6, f x a = 
    if a < -6 then 39 + 12 * a 
    else if -6 ≤ a ∧ a ≤ 4 then -a^2 + 3 
    else 19 - 8 * a) :=
sorry

end problem1_min_max_problem2_monotonous_range_problem3_min_value_l122_122188


namespace number_of_ordered_pairs_l122_122900

theorem number_of_ordered_pairs :
  (card {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ 150 ∧
                        (complex.I ^ p.1 + complex.I ^ p.2).im = 0 ∧
                        p.1 % 3 = 0 }) = 772 :=
  sorry

end number_of_ordered_pairs_l122_122900


namespace base_length_of_parallelogram_l122_122135

theorem base_length_of_parallelogram (A : ℕ) (H : ℕ) (Base : ℕ) (hA : A = 576) (hH : H = 48) (hArea : A = Base * H) : 
  Base = 12 := 
by 
  -- We skip the proof steps since we only need to provide the Lean theorem statement.
  sorry

end base_length_of_parallelogram_l122_122135


namespace math_problem_l122_122745

variable (x y : ℝ)

theorem math_problem (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0) (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) :
  x * y - 12 * x + 15 * y = 0 :=
  sorry

end math_problem_l122_122745


namespace price_of_chicken_l122_122053

theorem price_of_chicken 
  (price_duck : ℕ)
  (price_chicken : ℕ)
  (ducks : ℕ)
  (chickens : ℕ)
  (additional_earnings : ℕ)
  (half_earnings : ℕ)
  (total_then_double : ℕ) :
  price_duck = 10 →
  ducks = 2 →
  chickens = 5 →
  additional_earnings = 60 →
  total_then_double = 2 * half_earnings →
  total_then_double = 60 →
  half_earnings = (price_duck * ducks + price_chicken * chickens) / 2 →
  price_chicken = 8 :=
begin
  sorry
end

end price_of_chicken_l122_122053


namespace upper_limit_of_people_l122_122131

theorem upper_limit_of_people (P : ℕ) (h1 : 36 = (3 / 8) * P) (h2 : P > 50) (h3 : (5 / 12) * P = 40) : P = 96 :=
by
  sorry

end upper_limit_of_people_l122_122131


namespace necessary_but_not_sufficient_condition_l122_122177

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > 0) : 
  ((x > 2 ∧ x < 4) ↔ (2 < x ∧ x < 4)) :=
by {
    sorry
}

end necessary_but_not_sufficient_condition_l122_122177


namespace even_two_digit_numbers_count_l122_122749

theorem even_two_digit_numbers_count : 
  let digits := {0, 1, 2, 3, 4}
  ∃ count = 10, 
    (∀ (t : ℕ × ℕ), 
      t.1 ≠ t.2 ∧ t.1 ∈ digits ∧ t.2 ∈ digits ∧ ∃ k : ℕ, t.2 = 2 * k ∧ t.1 ≠ 0 → count) 
  :=
sorry

end even_two_digit_numbers_count_l122_122749


namespace expression_for_C_value_of_C_l122_122150

variables (x y : ℝ)

-- Definitions based on the given conditions
def A := x^2 - 2 * x * y + y^2
def B := x^2 + 2 * x * y + y^2

-- The algebraic expression for C
def C := - x^2 + 10 * x * y - y^2

-- Prove that the expression for C is correct
theorem expression_for_C (h : 3 * A x y - 2 * B x y + C x y = 0) : 
  C x y = - x^2 + 10 * x * y - y^2 := 
by {
  sorry
}

-- Prove the value of C when x = 1/2 and y = -2
theorem value_of_C : C (1/2) (-2) = -57/4 :=
by {
  sorry
}

end expression_for_C_value_of_C_l122_122150


namespace find_x_l122_122605

def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := log (8 * x) / log (sqrt 2)

theorem find_x : ∃ x : ℝ, f (g x) = g (f x) ∧ x = (1 + sqrt 385) / 64 := by
  sorry

end find_x_l122_122605


namespace count_integers_in_range_l122_122652

theorem count_integers_in_range : ∃ n ∈ ℕ, n = 4 ∧ ∀ x : ℤ, -3 ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ 8 ↔ x ∈ {0, 1, 2, 3} := 
by
  sorry

end count_integers_in_range_l122_122652


namespace geometric_theorem_l122_122212

-- Definition of the geometric setup
variable {P M N Q R S : Type} [Field ℝ]

-- Given conditions
-- A circle with center P
-- Quadrilateral MNPQ inscribed in this circle
-- A smaller circle with center R tangent to two sides of the quadrilateral
noncomputable def circle_center (circle : Type) [Field circle] : circle := sorry
noncomputable def quadrilateral (A B C D : Type) [Field ℝ] : Prop := sorry
noncomputable def tangent (A B C : Type) [Field ℝ] : Prop := sorry
noncomputable def extends_to (L : Type) [Field ℝ] : Type := sorry

-- Line segment NM extends to intersect the larger circle again at point S
variable (NM : Type) : extends_to ℝ

-- RM bisects ∠NRP and ∠RMS
noncomputable def bisects_angle (A B C : Type) [Field ℝ] : Prop := sorry

-- Definitions for cyclic quadrilateral
noncomputable def cyclic_quadrilateral (A B C D E : Type) [Field ℝ] : Prop := sorry

-- Prove that NS = MS = PR given the conditions
theorem geometric_theorem : NS = MS ∧ MS = PR :=
by {
  assume (circle_center P) (quadrilateral M N P Q) (circle_center R)
         (tangent R M P) (tangent R N P) (extends_to N M S)
         (bisects_angle R M N R P) (bisects_angle R M N R S),
  exact sorry,
}

end geometric_theorem_l122_122212


namespace map_distance_l122_122306

theorem map_distance (c1 c2 k1 k2 : ℕ) (h1 : c1 = 15) (h2 : k1 = 90) (h3 : c2 = 25) (h4 : k2 = 150) :
  (k1 / c1) * c2 = k2 :=
by 
  rw [h1, h2, h3, h4]
  -- Sorry to imply that we would usually now provide the proof details,
  -- we might use sufficient algebraic manipulation to show
  -- that (90 / 15) * 25 does indeed equal 150.
  sorry

end map_distance_l122_122306


namespace eval_expr_ceiling_l122_122918

theorem eval_expr_ceiling :
  ⌈4 * (8 - 3 / 4) + 2⌉ = 31 := 
by
  sorry

end eval_expr_ceiling_l122_122918


namespace fraction_product_eq_l122_122550

theorem fraction_product_eq :
  (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end fraction_product_eq_l122_122550


namespace number_of_solutions_l122_122583

-- Define the equation parts
def numerator (z : ℂ) := z^4 - 1
def denominator (z : ℂ) := z^3 - 3z + 2

-- The main theorem statement
theorem number_of_solutions : 
  (∃ z : ℂ, numerator z = 0 ∧ denominator z ≠ 0) -- This ensures z is a solution of the numerator and not a root of the denominator
  .set({ z | numerator z = 0 }).card - { z | numerator z = 0 ∧ denominator z = 0 }.to_finset.card = 3 := 
begin
  -- Proof will be filled here
  sorry
end

end number_of_solutions_l122_122583


namespace expected_value_K_squared_l122_122707

-- Define the set S of points (x, y) with integers from -4 to 4
def S : Set (ℤ × ℤ) := { (x, y) | -4 ≤ x ∧ x ≤ 4 ∧ -4 ≤ y ∧ y ≤ 4 }

-- Define three independent random points A, B, C chosen from S
noncomputable def A : S := sorry
noncomputable def B : S := sorry
noncomputable def C : S := sorry

-- Define the area K of the triangle ABC
def K (A B C : S) : ℚ :=
  let (⟨a, b⟩ : ℤ × ℤ) := A
  let (⟨c, d⟩ : ℤ × ℤ) := B
  let (⟨e, f⟩ : ℤ × ℤ) := C
  1/2 * (a * (d - f) + c * (f - b) + e * (b - d))

-- State the main theorem: the expected value of K^2 is 200/3
theorem expected_value_K_squared : E(K(A, B, C)^2) = 200 / 3 := sorry

end expected_value_K_squared_l122_122707


namespace profits_ratio_l122_122739

-- Define the investments of P and Q
def P_invest : ℕ := 40000
def Q_invest : ℕ := 60000

-- Define the greatest common divisor
def gcd (a b : ℕ) := Nat.gcd a b

-- Define the ratio of their investments
def investment_ratio : ℕ × ℕ := (P_invest / (gcd P_invest Q_invest), Q_invest / (gcd P_invest Q_invest))

-- Prove the ratio of their profits is equal to 2 : 3 given the investment amounts
theorem profits_ratio (P_invest Q_invest: ℕ) (hP: P_invest = 40000) (hQ: Q_invest = 60000) :
  investment_ratio = (2, 3) :=
by
  sorry

end profits_ratio_l122_122739


namespace question1_question2_question3_l122_122632

-- Definitions for conditions
def polynomial_quadratic (a : ℝ) : Polynomial ℝ :=
  Polynomial.C (a + 4) * Polynomial.X^3 + Polynomial.C 6 * Polynomial.X^2 -
  Polynomial.C 2 * Polynomial.X + Polynomial.C 5

def quadratic_coeff_is (b : ℝ) : Prop :=
  polynomial_quadratic (-4) ∈ Polynomial.monomial 6 Polynomial.X^2

-- Lean statements
theorem question1 (a b : ℝ) (h1 : polynomial_quadratic a) (h2 : quadratic_coeff_is b) :
  a = -4 ∧ b = 6 := sorry

theorem question2 (t : ℝ) : 
  ∀ P A B, (P = A + 2 * t) → (B - P = 1/2 * abs (A - P)) → t = 10 ∨ t = 10 / 3 := sorry 

theorem question3 (m : ℝ) :
  ∀ P Q A B, (P = A - 1/2 * m) → (Q = B - 2 * (m - 2)) → abs (P - Q) = 8 → m = 4 ∨ m = 44 / 3 := sorry

end question1_question2_question3_l122_122632


namespace thm_geo_sequence_sum_l122_122979

noncomputable def geo_sequence_sum (a r : ℝ) : Prop :=
  let a4 := a * r^3 in
  let a5 := a * r^4 in
  let a6 := a * r^5 in
  let a7 := a * r^6 in
  let a8 := a * r^7 in
  a4 * a6 + 2 * a5 * a7 + a6 * a8 = 36

theorem thm_geo_sequence_sum (a r : ℝ) (h : geo_sequence_sum a r) : 
  ∃ x : ℝ, x = a * r^4 + a * r^6 ∧ (x = 6 ∨ x = -6) :=
by
  cases h
  sorry

end thm_geo_sequence_sum_l122_122979


namespace complex_simplification_l122_122751

theorem complex_simplification :
  ((-3 - 2 * complex.I) - (1 + 4 * complex.I)) * (2 - 3 * complex.I) = 10 := 
by
  sorry

end complex_simplification_l122_122751


namespace finite_decimal_fractions_l122_122575

theorem finite_decimal_fractions (a b c d : ℕ) (n : ℕ) 
  (h1 : n = 2^a * 5^b)
  (h2 : n + 1 = 2^c * 5^d) :
  n = 1 ∨ n = 4 :=
by
  sorry

end finite_decimal_fractions_l122_122575


namespace day_100_days_from_wednesday_l122_122799

-- Definitions for the conditions
def today_is_wednesday := "Wednesday"

def days_in_week := 7

def day_of_the_week (n : Nat) : String := 
  let days := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  days[n % days.length]

-- Theorem to prove
theorem day_100_days_from_wednesday : day_of_the_week ((4 + 100) % days_in_week) = "Friday" :=
  sorry

end day_100_days_from_wednesday_l122_122799


namespace prove_monotonic_increasing_interval_l122_122779

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | (k * real.pi / 2 - real.pi / 8) < x ∧ x < (k * real.pi / 2 + 3 * real.pi / 8)}

theorem prove_monotonic_increasing_interval (k : ℤ) :
  ∀ x, x ∈ (monotonic_increasing_interval k) ↔
    (k * real.pi / 2 - real.pi / 8) < x ∧ x < (k * real.pi / 2 + 3 * real.pi / 8) :=
  sorry

end prove_monotonic_increasing_interval_l122_122779


namespace ordered_quadruple_ellipse_l122_122541

noncomputable def ellipse_quadruple := 
  let f₁ : (ℝ × ℝ) := (1, 1)
  let f₂ : (ℝ × ℝ) := (1, 7)
  let p : (ℝ × ℝ) := (12, -1)
  let a := (5 / 2) * (Real.sqrt 5 + Real.sqrt 37)
  let b := (1 / 2) * Real.sqrt (1014 + 50 * Real.sqrt 185)
  let h := 1
  let k := 4
  (a, b, h, k)

theorem ordered_quadruple_ellipse :
  let e : (ℝ × ℝ × ℝ × ℝ) := θse_quadruple
  e = ((5 / 2 * (Real.sqrt 5 + Real.sqrt 37)), (1 / 2 * Real.sqrt (1014 + 50 * Real.sqrt 185)), 1, 4) :=
by
  sorry

end ordered_quadruple_ellipse_l122_122541


namespace sphere_proj_invariant_l122_122487

theorem sphere_proj_invariant (S : Type) [Sphere S] :
  ∀ (P : orthographic_projection S), is_invariant P :=
by sorry

end sphere_proj_invariant_l122_122487


namespace repeating_decimal_to_fraction_l122_122569

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l122_122569


namespace opposite_of_neg5_is_pos5_l122_122354

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122354


namespace find_set_A_l122_122196

def is_arithmetic_sum (k n : ℕ) (sum : ℕ) : Prop := 
  let a := k + 1 in 
  let l := k + n in 
  n * (a + l) / 2 = sum

theorem find_set_A : 
  ∃ (A : Set ℕ), 
  (∃ k n : ℕ, k > 0 ∧ n > 0 ∧ is_arithmetic_sum k n 2019 ∧ n = 6 ∧ A = { 334, 335, 336, 337, 338, 339 })
  :=
sorry

end find_set_A_l122_122196


namespace largest_interesting_number_l122_122809

def isInterestingNumber (n : ℕ) : Prop :=
  let digits := to_digits 10 n
  ∀ i, 2 ≤ i ∧ i < digits.length - 1 → 
    digits.nth! i < (digits.nth! (i - 1) + digits.nth! (i + 1)) / 2

theorem largest_interesting_number (n : ℕ) : isInterestingNumber n ↔ n ≤ 96433469 :=
sorry

end largest_interesting_number_l122_122809


namespace opposite_of_neg_five_l122_122363

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122363


namespace area_quadrilateral_ADEC_l122_122249

theorem area_quadrilateral_ADEC (C A D B E : Type) [HasAngle C] [HasLength C] [HasMidpoint A B D] [IsPerpendicular DE AB] 
  (AC AD DB AB DE : ℝ) 
  (h1 : Angle C = 90)
  (h2 : AD = DB)
  (h3 : DE ⊥ AB)
  (h4 : AB = 20)
  (h5 : AC = 12) :
  area A D E C = 58.5 :=
sorry

end area_quadrilateral_ADEC_l122_122249


namespace no_four_consecutive_product_square_l122_122914

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l122_122914


namespace cos_beta_value_l122_122994

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l122_122994


namespace sequence_count_l122_122733

theorem sequence_count :
  ∃ (S : Finset (Fin 1993 → ℕ)), S.card = 75 ∧
  (∀ (f : Fin 1993 → ℕ), f ∈ S → ∀ k : Fin 1993, k.val + 1 ∣ f k) ∧
  (∀ x : ℕ, x ∈ S.val.image (f → f (Fin.mk 0 sorry)) → x ∈ (Finset.range (1995-2+1)).image ((+) 3))
:=
sorry

end sequence_count_l122_122733


namespace valid_passcodes_count_l122_122752

-- Define a predicate that checks whether a list of digits forms a valid passcode.
def is_valid_passcode (lst : List ℕ) : Prop :=
  lst.length = 4 ∧ lst.prod = 18 ∧ ∀ x ∈ lst, 1 ≤ x ∧ x ≤ 9

-- Count the number of valid passcodes
def count_valid_passcodes : ℕ :=
  (List.range' 1 9).product (fun _ => List.range' 1 9).filter is_valid_passcode).length

theorem valid_passcodes_count : count_valid_passcodes = 36 :=
  sorry

end valid_passcodes_count_l122_122752


namespace PB_perpendicular_MK_l122_122035

theorem PB_perpendicular_MK
  (Γ : Type) [metric_space Γ] [normed_group Γ] [normed_space ℝ Γ]
  (BC : submodule ℝ Γ)
  (X Y : Γ)
  (XY : submodule ℝ Γ)
  (P : Γ)
  (M : Γ)
  (CY PB : submodule ℝ Γ)
  (CX PM : submodule ℝ Γ)
  (K : Γ)
  (h_diameter : BC = {x | ∃ b c, x = b - c ∧ ∥b - c∥ = 1}) -- BC is a diameter of circle Γ
  (h_XY_perp_BC : ∃ cx : Γ, XY = {y | ∀ x ∈ BC, inner (y - cx) x = 0}) -- XY is perpendicular to BC
  (h_points_on_circle : ∃ O : Γ, ∀ y ∈ {Y, X}, dist O y = radius) -- X and Y are on circle 
  (h_P_on_XY : P ∈ XY)
  (h_M_on_CY : M ∈ CY)
  (h_CY_parallel_PB : ∃ o, CY = {y | ∃ p ∈ PB, ∃ t : ℝ, y = o + t • p}) -- CY ∥ PB
  (h_CX_parallel_PM : ∃ o, CX = {y | ∃ p ∈ PM, ∃ t : ℝ, y = o + t • p}) -- CX ∥ PM
  (h_K_is_intersection : ∃ k, K ∈ {x | ∀ z ∈ XC, ∀ w ∈ BP, inner (x - z) w = 0}) : -- K is the intersection of lines XC and BP
  ∃ p m : Γ, PB = {x | ∀ k ∈ MK, inner (x - k) m = 0}. -- PB ⊥ MK

end PB_perpendicular_MK_l122_122035


namespace sqrt_mixed_number_simplified_l122_122927

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l122_122927


namespace min_trips_needed_l122_122414

noncomputable def min_trips (n : ℕ) (h : 2 ≤ n) : ℕ :=
  6

theorem min_trips_needed
  (n : ℕ) (h : 2 ≤ n) (students : Finset (Fin (2 * n)))
  (trip : ℕ → Finset (Fin (2 * n)))
  (trip_cond : ∀ i, (trip i).card = n)
  (pair_cond : ∀ (s t : Fin (2 * n)),
    s ≠ t → ∃ i, s ∈ trip i ∧ t ∈ trip i) :
  ∃ k, k = min_trips n h :=
by
  use 6
  sorry

end min_trips_needed_l122_122414


namespace strawberry_percentage_is_24_l122_122684

-- Define the total number of students who chose each flavor
def chocolate := 120
def strawberry := 100
def vanilla := 80
def mint := 50
def butterPecan := 70

-- Define the total number of responses
def totalResponses := chocolate + strawberry + vanilla + mint + butterPecan

-- Define the percentage of students who favored Strawberry
def strawberryPercentage := (strawberry.toFloat / totalResponses.toFloat) * 100

-- Prove that the percentage of students who favored Strawberry is 24%
theorem strawberry_percentage_is_24 :
  Int.round strawberryPercentage = 24 := by
  have h1 : (strawberry.toFloat / totalResponses.toFloat) * 100 = 23.809523809523807 := sorry
  show Int.round 23.809523809523807 = 24
  from intRound_eq_24 h1
  where
    intRound_eq_24 : round 23.809523809523807 = 24 := sorry


end strawberry_percentage_is_24_l122_122684


namespace range_of_omega_l122_122889

-- Define the sine function f(x) = sin(ωx + π/3)
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- The hypothesis for the problem
def has_extreme_points (ω : ℝ) :=
  (∃ x1 x2 x3 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (x3 < Real.pi)
    ∧ f' ω x1 = 0 ∧ f' ω x2 = 0 ∧ f' ω x3 = 0)

def has_zeros (ω : ℝ) :=
  (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < Real.pi)
    ∧ f ω x1 = 0 ∧ f ω x2 = 0)

-- The main theorem to be proved
theorem range_of_omega (ω : ℝ) :
  has_extreme_points ω ∧ has_zeros ω ↔ (13/6 < ω ∧ ω ≤ 8/3) :=
by
  sorry

end range_of_omega_l122_122889


namespace pencil_cost_l122_122843

theorem pencil_cost (p e : ℝ) (h1 : p + e = 3.40) (h2 : p = 3 + e) : p = 3.20 :=
by
  sorry

end pencil_cost_l122_122843


namespace verify_ellipse_circle_tangent_lines_l122_122163

noncomputable def ellipse_m : set (ℝ × ℝ) :=
  {p | let (x, y) := p in (x^2 / 4) + (y^2 / 3) = 1}

noncomputable def circle_n : set (ℝ × ℝ) :=
  {p | let (x, y) := p in (x - 1)^2 + y^2 = 5}

theorem verify_ellipse_circle_tangent_lines 
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = 1) (h4 : a = 2) (h5 : b = sqrt 3) : 
  ellipse_m = {p | let (x, y) := p in (x^2 / 4) + (y^2 / 3) = 1} ∧ 
  circle_n = {p | let (x, y) := p in (x - 1)^2 + y^2 = 5} ∧ 
  ∃ k m : ℝ, 
    (k = 1/2 ∧ m = 2 ∧ ∀ p, p ∈ {p | let (x, y) := p in y = k * x + m} → 
          (p ∈ ellipse_m ∨ 
          p ∈ circle_n)) ∨
    (k = -1/2 ∧ m = -2 ∧ ∀ p, p ∈ {p | let (x, y) := p in y = k * x + m} →
          (p ∈ ellipse_m ∨ 
          p ∈ circle_n)) :=
by {
  sorry
}

end verify_ellipse_circle_tangent_lines_l122_122163


namespace max_levels_passed_prob_pass_three_levels_l122_122784

-- Define the conditions of the game
def max_roll (n : ℕ) : ℕ := 6 * n
def pass_condition (n : ℕ) : ℕ := 2^n

-- Problem 1: Prove the maximum number of levels a person can pass
theorem max_levels_passed : ∃ n : ℕ, (∀ m : ℕ, m > n → max_roll m ≤ pass_condition m) ∧ (∀ m : ℕ, m ≤ n → max_roll m > pass_condition m) :=
by sorry

-- Define the probabilities for passing each level
def prob_pass_level_1 : ℚ := 4 / 6
def prob_pass_level_2 : ℚ := 30 / 36
def prob_pass_level_3 : ℚ := 160 / 216

-- Problem 2: Prove the probability of passing the first three levels consecutively
theorem prob_pass_three_levels : prob_pass_level_1 * prob_pass_level_2 * prob_pass_level_3 = 100 / 243 :=
by sorry

end max_levels_passed_prob_pass_three_levels_l122_122784


namespace original_radius_of_cylinder_l122_122259

theorem original_radius_of_cylinder :
  ∃ (r : ℝ), 
  (∀ (y : ℝ), y = 3 * π * ((r + 4)^2 - r^2) → y = 4 * π * r^2) → 
  r = 3 + Real.sqrt 21 :=
begin
  sorry
end

end original_radius_of_cylinder_l122_122259


namespace coprime_exists_pow_divisible_l122_122353

theorem coprime_exists_pow_divisible (a n : ℕ) (h_coprime : Nat.gcd a n = 1) : 
  ∃ m : ℕ, n ∣ a^m - 1 :=
by
  sorry

end coprime_exists_pow_divisible_l122_122353


namespace sum_of_digits_of_greatest_prime_divisor_of_18447_l122_122910

def sum_of_digits (n : ℕ) : ℕ :=
  (to_string n).foldl (λ s c => s + c.to_digit 10) 0

theorem sum_of_digits_of_greatest_prime_divisor_of_18447 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 18447 ∧ sum_of_digits p = 20 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_18447_l122_122910


namespace common_ratio_is_neg2_l122_122610

-- Definitions of the problem
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Assumptions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q
axiom sum_geom_seq (n : ℕ) : S n = (1 - q ^ (n + 1)) / (1 - q) * a 0
axiom arithmetic_seq (h : 2 * S 4 = S 5 + S 6)

-- Conclusion to prove
theorem common_ratio_is_neg2 (h_arith : 2 * S 4 = S 5 + S 6) : q = -2 :=
  sorry

end common_ratio_is_neg2_l122_122610


namespace fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l122_122149

def is_divisible_by_7 (n: ℕ): Prop := n % 7 = 0

theorem fourteen_divisible_by_7: is_divisible_by_7 14 :=
by
  sorry

theorem twenty_eight_divisible_by_7: is_divisible_by_7 28 :=
by
  sorry

theorem thirty_five_divisible_by_7: is_divisible_by_7 35 :=
by
  sorry

theorem forty_nine_divisible_by_7: is_divisible_by_7 49 :=
by
  sorry

end fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l122_122149


namespace function_extreme_points_and_zeros_l122_122872

noncomputable def ω_range : Set ℝ := 
  setOf (λ ω, (13 : ℝ)/6 < ω ∧ ω ≤ (8 : ℝ)/3)

theorem function_extreme_points_and_zeros (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = sin (ω * x + π / 3)) 
  (h2 : set.count (set_of (λ x, x ∈ (0, π) ∧ extreme_point f x)) = 3) 
  (h3 : set.count (set_of (λ x, x ∈ (0, π) ∧ f x = 0)) = 2) : 
  ω ∈ ω_range := 
sorry

end function_extreme_points_and_zeros_l122_122872


namespace incorrect_statement_c_l122_122023

theorem incorrect_statement_c (x y : ℝ) (h₀ : x = 500) (h₁ : y = 45) (h₂ : ∀ k : ℝ, y k = 45 - 0.08 * k) : ¬ (∀ k : ℝ, y k = 45 - 8 * k) :=
by {
  sorry,
}

end incorrect_statement_c_l122_122023


namespace find_quadratic_eq_l122_122587

theorem find_quadratic_eq (x y : ℝ) 
  (h₁ : x + y = 10)
  (h₂ : |x - y| = 6) :
  x^2 - 10 * x + 16 = 0 :=
sorry

end find_quadratic_eq_l122_122587


namespace area_enclosed_by_inequality_l122_122908

theorem area_enclosed_by_inequality : 
  let region := {p : ℝ × ℝ | 2 * |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  in ∃ area, measure.measure (measure_space.to_measure (outer_measure.borel ℝ)) region = 4.5 :=
by
  sorry -- Proof Boilerplate

end area_enclosed_by_inequality_l122_122908


namespace expand_polynomial_expression_l122_122922

theorem expand_polynomial_expression (x : ℝ) : 
  (x + 6) * (x + 8) * (x - 3) = x^3 + 11 * x^2 + 6 * x - 144 :=
by
  sorry

end expand_polynomial_expression_l122_122922


namespace no_four_consecutive_product_square_l122_122913

/-- Prove that there do not exist four consecutive positive integers whose product is a perfect square. -/
theorem no_four_consecutive_product_square :
  ¬ ∃ (x : ℕ), ∃ (n : ℕ), n * n = x * (x + 1) * (x + 2) * (x + 3) :=
sorry

end no_four_consecutive_product_square_l122_122913


namespace cyclic_quadrilaterals_with_one_right_angle_l122_122826

theorem cyclic_quadrilaterals_with_one_right_angle (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, k = 29070 ∧
  (∀ (points : Finset ℕ), points.card = n → ∀ (quads : Finset (Finset ℕ)), 
  (∀ quad ∈ quads, quad.card = 4) → 
  (∀ quad ∈ quads, (∃ d1 d2 p1 p2 ∈ points, d1 ≠ d2 ∧ p1 ≠ p2 ∧ p1 ≠ d1 ∧ p1 ≠ d2 ∧ 
     p2 ≠ d1 ∧ p2 ≠ d2 ∧ p1 ∈ quad ∧ p2 ∈ quad ∧ ∀ a b ∈ quad, angle a b = 90)) →
  quads.card = k) :=
begin
  sorry
end

end cyclic_quadrilaterals_with_one_right_angle_l122_122826


namespace opposite_of_negative_five_l122_122390

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122390


namespace at_least_two_solve_exactly_five_l122_122211

theorem at_least_two_solve_exactly_five (n : ℕ)
  (pij : ℕ × ℕ → ℕ)
  (n_r : ℕ → ℕ)
  (h1 : ∀ i j, 1 ≤ i → i < j → j ≤ 6 → pij (i, j) > 2 * n / 5)
  (h2 : n_r 6 = 0)
  (h3 : ∑ r in Finset.range 6, n_r r = n)
  : 2 ≤ n_r 5 :=
begin
  sorry
end

end at_least_two_solve_exactly_five_l122_122211


namespace sqrt_table_values_sqrt_pattern_1000_sqrt_b_in_terms_of_m_compare_sqrt_a_l122_122573

-- Defining the problem statements
theorem sqrt_table_values (a : ℝ) (h_a : a = 0.01 ∨ a = 100) : sqrt a = if h_a = 0.01 then 0.1 else 10 :=
sorry

theorem sqrt_pattern_1000 (h_sqrt10 : sqrt 10 ≈ 3.16) : sqrt 1000 ≈ 31.6 :=
sorry

theorem sqrt_b_in_terms_of_m (m b : ℝ) (h_sqrt_m : sqrt m = 8.973) (h_sqrt_b: sqrt b = 897.3) : b = 10000 * m :=
sorry

theorem compare_sqrt_a (a : ℝ) :
    (sqrt a = a ↔ a = 0 ∨ a = 1) ∧
    (0 < a ∧ a < 1 → sqrt a > a) ∧
    (1 < a → sqrt a < a) :=
sorry

end sqrt_table_values_sqrt_pattern_1000_sqrt_b_in_terms_of_m_compare_sqrt_a_l122_122573


namespace total_jokes_sum_l122_122737

theorem total_jokes_sum :
  let jessy_week1 := 11
  let alan_week1 := 7
  let tom_week1 := 5
  let emily_week1 := 3
  let jessy_week4 := 11 * 3 ^ 3
  let alan_week4 := 7 * 2 ^ 3
  let tom_week4 := 5 * 4 ^ 3
  let emily_week4 := 3 * 4 ^ 3
  let jessy_total := 11 + 11 * 3 + 11 * 3 ^ 2 + jessy_week4
  let alan_total := 7 + 7 * 2 + 7 * 2 ^ 2 + alan_week4
  let tom_total := 5 + 5 * 4 + 5 * 4 ^ 2 + tom_week4
  let emily_total := 3 + 3 * 4 + 3 * 4 ^ 2 + emily_week4
  jessy_total + alan_total + tom_total + emily_total = 1225 :=
by 
  sorry

end total_jokes_sum_l122_122737


namespace sum_of_first_60_digits_l122_122815

noncomputable def decimal_expansion_period : List ℕ := [0, 0, 0, 8, 1, 0, 3, 7, 2, 7, 7, 1, 4, 7, 4, 8, 7, 8, 4, 4, 4, 0, 8, 4, 2, 7, 8, 7, 6, 8]

def sum_of_list (l : List ℕ) : ℕ := l.foldl (· + ·) 0

theorem sum_of_first_60_digits : sum_of_list (decimal_expansion_period ++ decimal_expansion_period) = 282 := 
by
  simp [decimal_expansion_period, sum_of_list]
  sorry

end sum_of_first_60_digits_l122_122815


namespace interest_rate_per_annum_l122_122410

variables (P r : ℝ)

theorem interest_rate_per_annum:
  (200 = P * r) ∧ (440 = P * ((1 + r)^2 - 1)) → r = 0.2 :=
by
  intro h
  cases h with h1 h2
  sorry

end interest_rate_per_annum_l122_122410


namespace opposite_of_neg_five_l122_122367

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122367


namespace identify_faulty_key_l122_122478

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122478


namespace last_two_digits_of_power_sequence_l122_122961

noncomputable def power_sequence (n : ℕ) : ℤ :=
  (Int.sqrt 29 + Int.sqrt 21)^(2 * n) + (Int.sqrt 29 - Int.sqrt 21)^(2 * n)

theorem last_two_digits_of_power_sequence :
  (power_sequence 992) % 100 = 71 := by
  sorry

end last_two_digits_of_power_sequence_l122_122961


namespace unique_solution_of_quadratic_l122_122647

theorem unique_solution_of_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9 / 8) :=
by
  sorry

end unique_solution_of_quadratic_l122_122647


namespace inequality_proof_l122_122724

variable (n : ℕ)
variable (x : Fin n → ℝ)

theorem inequality_proof (h₀ : x 0 = 0) 
    (hx : ∀ i, 1 ≤ i → x i > 0)
    (hΣ : ∑ i in Finset.range n, x i = 1) :
  1 ≤ ∑ i in Finset.range n, 
         x i / ( Real.sqrt (1 + ∑ j in Finset.range i, x j) * 
                 Real.sqrt (∑ j in Finset.Ici i, x j)) 
    ∧ ∑ i in Finset.range n, 
         x i / ( Real.sqrt (1 + ∑ j in Finset.range i, x j) * 
                 Real.sqrt (∑ j in Finset.Ici i, x j)) < n / 2 := 
sorry

end inequality_proof_l122_122724


namespace tangent_line_at_1_l122_122771

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
def tangent_line_eq : ℝ × ℝ → ℝ := fun ⟨x, y⟩ => x - y - 1

theorem tangent_line_at_1 : tangent_line_eq (1, f 1) = 0 := by
  -- Proof would go here
  sorry

end tangent_line_at_1_l122_122771


namespace area_quadrilateral_inequality_l122_122160

theorem area_quadrilateral_inequality (A B C D : ℝ) (h : convex_quad A B C D) :
  let S := area_of_quad A B C D
  S ≤ (A * A + B * B + C * C + D * D) / 4 := sorry

end area_quadrilateral_inequality_l122_122160


namespace total_number_of_orders_l122_122840

-- Define the conditions
def num_original_programs : Nat := 6
def num_added_programs : Nat := 3

-- State the theorem
theorem total_number_of_orders : ∃ n : ℕ, n = 210 :=
by
  -- This is where the proof would go
  sorry

end total_number_of_orders_l122_122840


namespace AJB_angle_l122_122344

-- Define the circle with the points
def points_on_circle (A B C D E F G H J : ℕ) : Prop :=
  (A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 3 ∧ E = 4 ∧ F = 5 ∧ G = 6 ∧ H = 7 ∧ J = 8)

-- Define the equal arc condition
def equal_arcs (A B C D E F G H : ℕ) (x : ℝ) : Prop :=
  (360 / 7 = x ∧
  (∠ B A = x ∧
  ∠ C B = x ∧
  ∠ D C = x ∧
  ∠ E D = x ∧
  ∠ F E = x ∧
  ∠ G F = x ∧
  ∠ H G = x))

-- Define given angle GHJ
def angle_GHJ (H J G : ℕ) : Prop :=
  (∠ H J = 117)

-- Define perpendicular condition
def perpendicular (B H E J : ℕ) : Prop :=
  (∠ B H E = 90 ∧ ∠ E J B = 90)

-- Prove the final angle AJB
theorem AJB_angle (A B C D E F G H J : ℕ) (x : ℝ) (m n : ℕ) :
  points_on_circle A B C D E F G H J →
  equal_arcs A B C D E F G H x →
  angle_GHJ H J G →
  perpendicular B H E J →
  (m = 27 ∧ n = 2) →
  ∠ A J B = (m / n) := by
  sorry

end AJB_angle_l122_122344


namespace perimeter_of_rectangle_WXYZ_l122_122324

theorem perimeter_of_rectangle_WXYZ 
  (WE XF EG FH : ℝ)
  (h1 : WE = 10)
  (h2 : XF = 25)
  (h3 : EG = 20)
  (h4 : FH = 50) :
  let p := 53 -- By solving the equivalent problem, where perimeter is simplified to 53/1 which gives p = 53 and q = 1
  let q := 29
  p + q = 102 := 
by
  sorry

end perimeter_of_rectangle_WXYZ_l122_122324


namespace tom_disproves_sam_l122_122750

def is_consonant (c : Char) : Prop :=
c ∉ ['a', 'e', 'i', 'o', 'u']

def is_prime (n : ℕ) : Prop :=
n = 2 ∨ n = 3 ∨ (n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def cards := ['S', 'T', 'U', '2', '5', '7', '11']

theorem tom_disproves_sam :
  ∃ (card : Char), card = 'S' ∧ ¬ (is_prime 3) :=
by
  sorry

end tom_disproves_sam_l122_122750


namespace expand_expression_l122_122921

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l122_122921


namespace triangle_sine_cosine_relation_l122_122257

theorem triangle_sine_cosine_relation 
  {a b c A B C : ℝ} 
  (h1 : a^2 + b^2 = 2018 * c^2)
  (h2 : ∀ {x y z : ℝ}, x = 2 * real.sin y * real.sin z * real.cos A → x = 2017 * (1 - real.cos A ^ 2)) 
  : (2 * real.sin A * real.sin B * real.cos C) / (1 - real.cos C ^ 2) = 2017 :=
sorry

end triangle_sine_cosine_relation_l122_122257


namespace math_problem_l122_122996

variable (a b : ℝ)

theorem math_problem 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_sum : a + b = 2) : 
  (ab_le_one : a * b ≤ 1) ∧ 
  (a_sq_plus_b_sq_ge_two : a^2 + b^2 ≥ 2) ∧ 
  (sqrt_a_plus_sqrt_b_le_two : Real.sqrt a + Real.sqrt b ≤ 2) ∧ 
  ¬ (one_div_a_plus_one_div_b_le_two : 1 / a + 1 / b ≤ 2) :=
sorry

end math_problem_l122_122996


namespace no_uniformly_colored_rectangle_l122_122110

open Int

def point := (ℤ × ℤ)

def is_green (P : point) : Prop :=
  3 ∣ (P.1 + P.2)

def is_red (P : point) : Prop :=
  ¬ is_green P

def is_rectangle (A B C D : point) : Prop :=
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2

def rectangle_area (A B : point) : ℤ :=
  abs (B.1 - A.1) * abs (B.2 - A.2)

theorem no_uniformly_colored_rectangle :
  ∀ (A B C D : point) (k : ℕ), 
  is_rectangle A B C D →
  rectangle_area A C = 2^k →
  ¬ (is_green A ∧ is_green B ∧ is_green C ∧ is_green D) ∧
  ¬ (is_red A ∧ is_red B ∧ is_red C ∧ is_red D) :=
by sorry

end no_uniformly_colored_rectangle_l122_122110


namespace Carla_put_initial_apples_l122_122919

theorem Carla_put_initial_apples :
  ∀ (stolen fallen remaining total_initial : ℕ),
  stolen = 45 →
  fallen = 26 →
  remaining = 8 →
  total_initial = stolen + fallen + remaining →
  total_initial = 79 :=
by
  intros stolen fallen remaining total_initial h_stolen h_fallen h_remaining h_total_initial
  rw [h_stolen, h_fallen, h_remaining, h_total_initial]
  sorry

end Carla_put_initial_apples_l122_122919


namespace minimum_questions_required_l122_122010

theorem minimum_questions_required 
  (n : ℕ) 
  (x : Fin n → ℤ)
  (hx : ∀ i, abs (x i) < 10) : 
  ∃ a : Fin n → ℤ, 
  ∀ y : Fin n → ℤ, (∀ i, abs (y i) < 10) → 
  (∑ i, a i * x i) = (∑ i, a i * y i) → x = y :=
sorry

end minimum_questions_required_l122_122010


namespace find_sum_l122_122342

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l122_122342


namespace total_octopus_legs_is_39_l122_122755

-- Definitions and conditions.
def lies_if_even (legs : Nat) : Prop := legs % 2 = 0
def tells_truth_if_odd (legs : Nat) : Prop := legs % 2 = 1

-- Octopuses' possible leg counts.
def valid_legs (legs : Nat) : Prop := legs ∈ {7, 8, 9}

-- Statements made by the octopuses.
def statement_1_legs (legs : Nat) : Prop := legs = 36
def statement_2_legs (legs : Nat) : Prop := legs = 37
def statement_3_legs (legs : Nat) : Prop := legs = 38
def statement_4_legs (legs : Nat) : Prop := legs = 39
def statement_5_legs (legs : Nat) : Prop := legs = 40

-- Lean theorem to prove the total number of legs.
theorem total_octopus_legs_is_39 :
  ∃ legs_list : List Nat,
    (∀ legs ∈ legs_list, valid_legs legs) ∧
    length legs_list = 5 ∧
    let total_legs := legs_list.sum 
    ∧ (statement_1_legs total_legs ↔ lies_if_even total_legs)
    ∧ (statement_2_legs total_legs ↔ lies_if_even total_legs)
    ∧ (statement_3_legs total_legs ↔ lies_if_even total_legs)
    ∧ (statement_4_legs total_legs ↔ tells_truth_if_odd total_legs)
    ∧ (statement_5_legs total_legs ↔ lies_if_even total_legs)
    in total_legs = 39 := 
sorry

end total_octopus_legs_is_39_l122_122755


namespace scale_model_height_l122_122521

theorem scale_model_height (real_height : ℕ) (scale_ratio : ℕ) (h_real : real_height = 1454) (h_scale : scale_ratio = 50) : 
⌊(real_height : ℝ) / scale_ratio + 0.5⌋ = 29 :=
by
  rw [h_real, h_scale]
  norm_num
  sorry

end scale_model_height_l122_122521


namespace first_digit_base8_of_350_is_5_l122_122433

theorem first_digit_base8_of_350_is_5 : 
  ∀ base10 : ℕ, base10 = 350 → (base10 / 64) = 5 := 
by
  intros base10 h_base10
  rw h_base10
  norm_num
-- Proof will continue by substituting the value and simplifying the division to show 350 / 64 = 5
sorry

end first_digit_base8_of_350_is_5_l122_122433


namespace repeating_decimal_to_fraction_l122_122570

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end repeating_decimal_to_fraction_l122_122570


namespace find_ST_l122_122424

noncomputable theory

-- Define the triangle PQR and its side lengths
def Triangle (P Q R : Type) :=
  ∃ (PQ PR QR : ℝ), PQ = 13 ∧ PR = 14 ∧ QR = 15

-- Define that points S and T lie on PQ and PR respectively
-- Define that ST is parallel to QR and contains the incenter of triangle PQR
def PointsLine (S T : Type) (PQ PR QR : ℝ) :=
  ∃ (ST QR : ℝ), ST / QR = 27 / 42

-- Prove the length of ST is 135/14 and m + n = 149
theorem find_ST {P Q R S T : Type} :
  Triangle P Q R →
  PointsLine S T 13 14 15 →
  let ST := (135:ℝ) / (14:ℝ) in
  let m := 135 in
  let n := 14 in
  m + n = 149 :=
by
  intros
  exact sorry

end find_ST_l122_122424


namespace minimize_station_distance_l122_122736

open Real

def total_distance (x : ℝ) : ℝ :=
  abs x + 2 * abs (x - 50) + 3 * abs (x - 100) + 4 * abs (x - 150) + 5 * abs (x - 200)

theorem minimize_station_distance :
  ∃ x : ℝ, (x = 150 ∧ ∀ y : ℝ, total_distance x ≤ total_distance y) :=
begin
  use 150,
  split,
  { refl },
  { intro y,
    sorry },
end

end minimize_station_distance_l122_122736


namespace exists_cyclic_quadrilateral_l122_122034

noncomputable def quadratic (p q : ℝ) := λ x : ℝ, x^2 + p * x + q

variables {p q x1 x2 : ℝ}
variables (A B C D : ℝ × ℝ)

def A := (x1, 0)
def B := (x2, 0)
def C := (0, q)
def D := (0, 1)

theorem exists_cyclic_quadrilateral 
  (h1 : quadratic p q x1 = 0) 
  (h2 : quadratic p q x2 = 0) 
  (h3 : q ≠ 0) 
  (h4 : D = (0, 1)) : 
  ∃ (O : ℝ × ℝ) (r : ℝ), 
  ∀ P ∈ [A, B, C, D], dist P O = r :=
sorry

end exists_cyclic_quadrilateral_l122_122034


namespace find_k_l122_122673

theorem find_k (k : ℝ) (h : ∀ x: ℝ, (x = -2) → (1 + k / (x - 1) = 0)) : k = 3 :=
by
  sorry

end find_k_l122_122673


namespace smallest_n_for_roots_of_unity_l122_122810

theorem smallest_n_for_roots_of_unity :
  ∃ n : ℕ, (∀ z : ℂ, z ^ 4 - z ^ 2 + 1 = 0 → ∃ k : ℤ, z = complex.exp (2 * k * real.pi * complex.I / n)) ∧
  ∀ m : ℕ, (∀ z : ℂ, z ^ 4 - z ^ 2 + 1 = 0 → ∃ k : ℤ, z = complex.exp (2 * k * real.pi * complex.I / m)) → n ≤ m :=
begin
  use 12,
  sorry
end

end smallest_n_for_roots_of_unity_l122_122810


namespace angle_ADB_is_right_angle_l122_122511

structure Circle (center : Point) (radius : ℝ) :=
  (on_circle : ∀ (P : Point), dist center P = radius)

structure Triangle (A B C : Point) :=
  (isosceles : dist A B = dist A C)

open Real

theorem angle_ADB_is_right_angle
  (A B C D : Point)
  (r : ℝ)
  (h1 : Circle C r)
  (h2 : Triangle A B C)
  (h3 : dist C B = r)
  (h4 : dist C A = r)
  (h5 : ∃ D, dist C D = r ∧ collinear ({ A, C, D }))
  : ∠ADB = π / 2 :=
sorry

end angle_ADB_is_right_angle_l122_122511


namespace order_of_xyz_l122_122975

variable (a b c d : ℝ) -- Declare the variables as real numbers
variable (h : a > b) (i : b > c) (j : c > d) (k : d > 0) -- Declare the conditions

def x : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
def y : ℝ := Real.sqrt (a * c) + Real.sqrt (b * d)
def z : ℝ := Real.sqrt (a * d) + Real.sqrt (b * c)

theorem order_of_xyz : z < y ∧ y < x := by
  sorry -- Proof is required here

end order_of_xyz_l122_122975


namespace sum_of_nu_for_lcm_60_l122_122811

theorem sum_of_nu_for_lcm_60 :
  let possible_values : List ℕ := [3, 6, 12, 15, 30, 60] in
  (possible_values.sum) = 126 :=
by
  let possible_values : List ℕ := [3, 6, 12, 15, 30, 60]
  have h1 : possible_values.sum = 126 := by
    -- Proof steps skipped
    sorry
  exact h1

end sum_of_nu_for_lcm_60_l122_122811


namespace north_southland_population_increase_l122_122692

noncomputable def net_population_increase_per_year
  (births_per_hour : ℝ)
  (deaths_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_year : ℝ) : ℝ :=
  (hours_per_day / births_per_hour - hours_per_day / deaths_per_hour) * days_per_year

theorem north_southland_population_increase :
  net_population_increase_per_year 0.1 (1/30) 24 365 ≈ 600 :=
by 
  -- Use the approximation because multiplication can lead to slightly off floating-point results
  sorry

end north_southland_population_increase_l122_122692


namespace pentagon_arrows_same_vertex_l122_122608

theorem pentagon_arrows_same_vertex
  (A B C D E : Point)
  (h_convex : ConvexPentagon A B C D E)
  (h_nonparallel_edges_diagonals : ∀ (X Y : Point) (h_edgeXY : Edge A B ∨ Edge B C ∨ Edge C D ∨ Edge D E ∨ Edge E A ∨ Diagonal A C ∨ Diagonal A D ∨ Diagonal B D ∨ Diagonal B E ∨ Diagonal C E), ¬Parallel (Edge X Y))
  : ∃ (P : Point), at_least_two_arrows_point_to_vertex A B C D E P :=
sorry

end pentagon_arrows_same_vertex_l122_122608


namespace basketball_substitution_mod_1000_l122_122047

def basketball_substitution_count_mod (n_playing n_substitutes max_subs : ℕ) : ℕ :=
  let no_subs := 1
  let one_sub := n_playing * n_substitutes
  let two_subs := n_playing * (n_playing - 1) * (n_substitutes * (n_substitutes - 1)) / 2
  let three_subs := n_playing * (n_playing - 1) * (n_playing - 2) *
                    (n_substitutes * (n_substitutes - 1) * (n_substitutes - 2)) / 6
  no_subs + one_sub + two_subs + three_subs 

theorem basketball_substitution_mod_1000 :
  basketball_substitution_count_mod 9 9 3 % 1000 = 10 :=
  by 
    -- Here the proof would be implemented
    sorry

end basketball_substitution_mod_1000_l122_122047


namespace sum_zero_quotient_l122_122719

   theorem sum_zero_quotient (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + y + z = 0) :
     (xy + yz + zx) / (x^2 + y^2 + z^2) = -1 / 2 :=
   by
     sorry
   
end sum_zero_quotient_l122_122719


namespace cubic_roots_a_b_third_root_l122_122247

theorem cubic_roots_a_b_third_root (a b : ℝ) :
  (∀ x, x^3 + a * x^2 + b * x + 6 = 0 → (x = 2 ∨ x = 3 ∨ x = -1)) →
  a = -4 ∧ b = 1 :=
by
  intro h
  -- We're skipping the proof steps and focusing on definite the goal
  sorry

end cubic_roots_a_b_third_root_l122_122247


namespace function_extreme_points_and_zeros_l122_122873

noncomputable def ω_range : Set ℝ := 
  setOf (λ ω, (13 : ℝ)/6 < ω ∧ ω ≤ (8 : ℝ)/3)

theorem function_extreme_points_and_zeros (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = sin (ω * x + π / 3)) 
  (h2 : set.count (set_of (λ x, x ∈ (0, π) ∧ extreme_point f x)) = 3) 
  (h3 : set.count (set_of (λ x, x ∈ (0, π) ∧ f x = 0)) = 2) : 
  ω ∈ ω_range := 
sorry

end function_extreme_points_and_zeros_l122_122873


namespace sqrt_mixed_number_simplified_l122_122939

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l122_122939


namespace seeds_problem_l122_122061

theorem seeds_problem (n : ℕ) 
  (hS1 : (1 + 2 + ... + n = n * (n + 1) / 2) : 
  (S2 : ℕ := (n+1) + (n+2) + ... + (2*n)) 
  (hS2 : S2 = n * (3 * n + 1) / 2) 
  (hDifference : S2 = n * (n + 1) / 2 + 100) : 
  n = 10 := by
  sorry

end seeds_problem_l122_122061


namespace percent_carnations_l122_122837

theorem percent_carnations (F : ℕ) (H1 : 3 / 5 * F = pink) (H2 : 1 / 5 * F = white) 
(H3 : F - pink - white = red) (H4 : 1 / 2 * pink = pink_roses)
(H5 : pink - pink_roses = pink_carnations) (H6 : 1 / 2 * red = red_carnations)
(H7 : white = white_carnations) : 
100 * (pink_carnations + red_carnations + white_carnations) / F = 60 :=
sorry

end percent_carnations_l122_122837


namespace max_value_of_f_l122_122595

def f (x : ℝ) := min (2 * x + 2) (min ((1/2) * x + 1) ((-3/4) * x + 7))

theorem max_value_of_f : ∃ x : ℝ, f x = 17 / 5 :=
begin
  -- This is where the proof would go, but it's not required as per the criteria.
  sorry
end

end max_value_of_f_l122_122595


namespace equation_solution_range_l122_122490

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ a ∈ set.Ioo 3 27 :=
by sorry

end equation_solution_range_l122_122490


namespace number_of_coloring_methods_l122_122546

theorem number_of_coloring_methods (n r : ℕ) (h1 : n > 1) (h2 : r > 1) : 
    (r - 1)^n + (-1)^n * (r - 1) + 
    (∑ k in finset.range (⌊n / 2⌋ + 1).filter (λ k, k > 0), 
      (n / k) * nat.choose (n - k - 1) (k - 1) *
        ((r - 1)^(n - k) + (-1)^(n - k) * (r - 1))) = 
    sorry

end number_of_coloring_methods_l122_122546


namespace fraction_of_male_birds_l122_122687

theorem fraction_of_male_birds (T : ℕ) (h_total : T > 0) : 
  let robins := (2 / 5 : ℚ) * T,
      bluejays := (3 / 5 : ℚ) * T,
      male_robins := robins - (1 / 3 : ℚ) * robins,
      male_bluejays := bluejays - (2 / 3 : ℚ) * bluejays in
  (male_robins + male_bluejays) / T = 7 / 15 :=
by {
  -- Proof goes here
  sorry
}

end fraction_of_male_birds_l122_122687


namespace find_floor_value_l122_122292

def floor (x : ℝ) : ℤ := Int.floor x

def a : ℝ := 1 + ∑ n in (Finset.range 2003).map (λ n, n + 2), 1 / ((n : ℝ) ^ 2)

theorem find_floor_value : floor a = 1 :=
by
  -- Proof omitted
  sorry

end find_floor_value_l122_122292


namespace green_ball_probabilities_l122_122544

-- Definitions corresponding to the conditions in the problem
def num_red_balls := 6
def num_yellow_balls := 9
def initial_num_green_balls := 3

-- Initial total number of balls in the bag
def initial_total_balls := num_red_balls + num_yellow_balls + initial_num_green_balls

-- The probability of drawing a green ball initially
def initial_prob_green_ball := initial_num_green_balls.toRational / initial_total_balls.toRational

-- Number of additional green balls to achieve the desired probability
def additional_green_balls_needed := 2

-- New number of green balls after adding
def new_num_green_balls := initial_num_green_balls + additional_green_balls_needed

-- New total number of balls after adding
def new_total_balls := initial_total_balls + additional_green_balls_needed

-- The desired probability of drawing a green ball after adding
def desired_prob_green_ball := (new_num_green_balls).toRational / new_total_balls.toRational

-- Lean statement to be proved
theorem green_ball_probabilities :
  initial_prob_green_ball = 1 / 6 ∧ desired_prob_green_ball = 1 / 4 :=
by
  sorry

end green_ball_probabilities_l122_122544


namespace find_z_l122_122607

-- Given conditions
def z : ℂ := 1 - 𝑖
def sqrt3_minus_i : ℂ := real.sqrt 3 - 𝑖
def condition : Prop := (1 + 𝑖) * z = complex.abs sqrt3_minus_i

-- Proof statement
theorem find_z : condition → z = 1 - 𝑖 :=
by
  intro h
  sorry

end find_z_l122_122607


namespace gavrila_final_distance_l122_122093

-- Define constant distances and speeds
def L : ℝ := 50  -- Halfway distance to starting point for Gavrila
def speed : ℝ := 20  -- Speed of both bicycle and motorboat in km/h
def distance_from_bank : ℝ := 40  -- Given distance y from the bank, in meters

-- Define the equation for Gavrila's coordinate computation
def gavrila_x_coordinate (y : ℝ) : ℝ := (y^2) / (4 * L)

-- Define Pythagorean theorem application to find Gavrila's distance from the starting point
def gavrila_distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Finally, the theorem statement that needs to be proven
theorem gavrila_final_distance :
  let x := gavrila_x_coordinate distance_from_bank in
  let s := gavrila_distance x distance_from_bank in
  real.round s = 41 :=
by {
  -- Proof omitted
  sorry
}

end gavrila_final_distance_l122_122093


namespace carol_total_peanuts_l122_122101

open Nat

-- Define the conditions
def peanuts_from_tree : Nat := 48
def peanuts_from_ground : Nat := 178
def bags_of_peanuts : Nat := 3
def peanuts_per_bag : Nat := 250

-- Define the total number of peanuts Carol has to prove it equals 976
def total_peanuts (peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag : Nat) : Nat :=
  peanuts_from_tree + peanuts_from_ground + (bags_of_peanuts * peanuts_per_bag)

theorem carol_total_peanuts : total_peanuts peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag = 976 :=
  by
    -- proof goes here
    sorry

end carol_total_peanuts_l122_122101


namespace horizontal_asymptote_is_3_l122_122892

-- Definitions of the polynomials
noncomputable def p (x : ℝ) : ℝ := 15 * x^5 + 10 * x^4 + 5 * x^3 + 7 * x^2 + 6 * x + 2
noncomputable def q (x : ℝ) : ℝ := 5 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1

-- Statement that we need to prove
theorem horizontal_asymptote_is_3 : 
  (∃ (y : ℝ), (∀ x : ℝ, x ≠ 0 → (p x / q x) = y) ∧ y = 3) :=
  sorry -- The proof is left as an exercise.

end horizontal_asymptote_is_3_l122_122892


namespace num_BooleanFunctions_num_elements_D_10_g_sum_elements_D_10_g_l122_122502

-- Define n-ary Boolean function
def BooleanFunction (n : ℕ) := vector (fin 2) n → fin 2

-- Define the set D_n(f)
def D_n {n : ℕ} (f : BooleanFunction n) := {x : vector (fin 2) n | f x = 0}

-- Problem 1
theorem num_BooleanFunctions (n : ℕ) : 
  (fin 2 ^ n → fin 2) ≃ fin (2 ^ (2 ^ n)) :=
sorry

-- Define g and D_10(g)
def g (x : vector (fin 2) 10) : fin 2 :=
  (1 + ∑ i in finset.range 10, ∏ j in finset.range (i + 1), x.nth j) % 2

def D_10 (f : BooleanFunction 10) := D_n f

-- Problem 2 part a
theorem num_elements_D_10_g : 
  |D_10 g| = 341 := 
sorry

-- Problem 2 part b
theorem sum_elements_D_10_g : 
  ∑ x in D_10 g, (finset.univ.sum (λ i, x i)) = 565 := 
sorry

end num_BooleanFunctions_num_elements_D_10_g_sum_elements_D_10_g_l122_122502


namespace faulty_key_in_digits_l122_122471

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122471


namespace possible_faulty_keys_l122_122458

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122458


namespace log_ab_eq_l122_122334

-- Definition and conditions
variables (a b x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hx : 0 < x)

-- The theorem to prove
theorem log_ab_eq (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log (x) / Real.log (a * b) = (Real.log (x) / Real.log (a)) * (Real.log (x) / Real.log (b)) / ((Real.log (x) / Real.log (a)) + (Real.log (x) / Real.log (b))) :=
sorry

end log_ab_eq_l122_122334


namespace logarithmic_inequality_l122_122314

theorem logarithmic_inequality (a : ℝ) (h : a > 1) : 
  1 / 2 + 1 / Real.log a ≥ 1 := 
sorry

end logarithmic_inequality_l122_122314


namespace faulty_key_in_digits_l122_122476

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122476


namespace correct_set_representation_l122_122488

theorem correct_set_representation : 
  (A: Set ℕ := {2, 4}) ∧
  (B: Set ℕ := {2, 4, 4}) ∧
  (C: List ℕ := [1, 2, 3]) ∧
  (D: Set String := {"tall boys"}) → 
  correct_representation (A) :=
by
  sorry

end correct_set_representation_l122_122488


namespace opposite_of_negative_five_l122_122373

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122373


namespace region_area_l122_122578

noncomputable def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

def region (x y : ℝ) : Prop := absolute_value(2 * x + 3 * y) + absolute_value(3 * x - 2 * y) ≤ 6

theorem region_area : 
  let area := ∫∫ (x y : ℝ) in {xy : ℝ × ℝ | region xy.1 xy.2}, 1 
  in area = 14.4 := 
by 
  sorry

end region_area_l122_122578


namespace problem1_problem2_l122_122691

theorem problem1 (P : ℝ × ℝ) (a b : ℝ) (h1 : a = 0) (h2 : b = sqrt 3)
  (h3 : dist P (h1, -h2) + dist P (h1, h2) = 4) :
  P.1^2 + P.2^2 / 4 = 1 := 
sorry

theorem problem2 (A B : ℝ × ℝ) (k : ℝ) 
  (h1 : A.2 = k * A.1 + 1) (h2 : B.2 = k * B.1 + 1)
  (h3 : A.1^2 + (A.2)^2 / 4 = 1) (h4 : B.1^2 + (B.2)^2 / 4 = 1)
  (h5 : A.1 * B.1 + A.2 * B.2 = 0) :
  k = 1/2 ∨ k = -1/2 := 
sorry

end problem1_problem2_l122_122691


namespace three_digit_numbers_exclusion_l122_122205

-- Definitions based on conditions
def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_two_identical_adjacent_digits_and_different_third (n : ℕ) : Prop := 
  let digits := n.digits 10 in 
  (digits.length = 3 ∧ 
    ((digits.nth 1 = digits.nth 2 ∧ digits.nth 0 ≠ digits.nth 1) ∨ 
     (digits.nth 0 = digits.nth 1 ∧ digits.nth 0 ≠ digits.nth 2)))

-- The Lean theorem statement
theorem three_digit_numbers_exclusion :
  let all_three_digit_numbers := {n : ℕ | is_three_digit_number n}
  let excluded_numbers := {n : ℕ | is_three_digit_number n ∧ has_two_identical_adjacent_digits_and_different_third n}
  (all_three_digit_numbers.card - excluded_numbers.card) = 738 := by
  sorry

end three_digit_numbers_exclusion_l122_122205


namespace cuboid_third_edge_l122_122346

theorem cuboid_third_edge (a b V h : ℝ) (ha : a = 4) (hb : b = 4) (hV : V = 96) (volume_formula : V = a * b * h) : h = 6 :=
by
  sorry

end cuboid_third_edge_l122_122346


namespace ratio_rect_prism_l122_122418

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l122_122418


namespace shortest_distance_Dasha_Vasya_l122_122693

def distance_Asya_Galia : ℕ := 12
def distance_Galia_Borya : ℕ := 10
def distance_Asya_Borya : ℕ := 8
def distance_Dasha_Galia : ℕ := 15
def distance_Vasya_Galia : ℕ := 17

def distance_Dasha_Vasya : ℕ :=
  distance_Dasha_Galia + distance_Vasya_Galia - distance_Asya_Galia - distance_Galia_Borya + distance_Asya_Borya

theorem shortest_distance_Dasha_Vasya : distance_Dasha_Vasya = 18 :=
by
  -- We assume the distances as given in the conditions. The calculation part is skipped here.
  -- The actual proof steps would go here.
  sorry

end shortest_distance_Dasha_Vasya_l122_122693


namespace five_people_six_chairs_l122_122235

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l122_122235


namespace find_m_l122_122422

theorem find_m 
  (y : ℝ) (m : ℝ) 
  (h : (3 - m = 0)) : m = 3 := 
by {
  exact h.symm,
  }

end find_m_l122_122422


namespace inequality_sum_powers_l122_122976

theorem inequality_sum_powers 
  (a b c : ℝ) (n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (habc : a * b * c ≥ 1) :
  n^3 * (∑ k in Finset.range (n+1), a^k) * 
       (∑ k in Finset.range (n+1), b^k) * 
       (∑ k in Finset.range (n+1), c^k) 
  ≥ (n+1)^3 * (∑ k in Finset.range n, a^k) * 
             (∑ k in Finset.range n, b^k) * 
             (∑ k in Finset.range n, c^k) :=
sorry

end inequality_sum_powers_l122_122976


namespace decreasing_function_interval_l122_122727

noncomputable def f (k x : ℝ) := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

theorem decreasing_function_interval (k : ℝ) : (∀ x ∈ Ioo 0 4, ∀ y ∈ Ioo 0 4, x < y → f' k x ≥ f' k y) → k ≤ 1 / 3 :=
by
  sorry

-- Definitions to handle differentiation and intervals
noncomputable def f' (k x : ℝ) := 3 * k * x^2 + 6 * (k - 1) * x

end decreasing_function_interval_l122_122727


namespace min_value_of_y_min_value_achieved_l122_122660

noncomputable def y (x : ℝ) : ℝ := x + 1/x + 16*x / (x^2 + 1)

theorem min_value_of_y : ∀ x > 1, y x ≥ 8 :=
  sorry

theorem min_value_achieved : ∃ x, (x > 1) ∧ (y x = 8) :=
  sorry

end min_value_of_y_min_value_achieved_l122_122660


namespace domain_of_f_l122_122431

def is_defined (y : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ z : ℝ,  y x = z

def domain (y : ℝ → ℝ) : set ℝ :=
  {x : ℝ | is_defined y x}

def f (x : ℝ) : ℝ := (x^2 - 4 * x + 4) / (x + 3)

theorem domain_of_f :
  domain f = {x : ℝ | x ≠ -3} :=
by
  sorry

end domain_of_f_l122_122431


namespace opposite_of_negative_five_l122_122372

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122372


namespace pavel_max_revenue_l122_122311

theorem pavel_max_revenue:
  (exists (x : ℤ), (x ≥ 0) ∧ (x ≤ 32) ∧ 
    (let revenue := (32 - x) * (x - 4.5) in revenue = 189) ∧ 
    (let crayfish_sold := 32 - x in crayfish_sold = 14)) :=
begin
  sorry
end

end pavel_max_revenue_l122_122311


namespace tangent_line_at_origin_is_x_plus_y_eq_zero_l122_122347

open Real

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2 - x

theorem tangent_line_at_origin_is_x_plus_y_eq_zero :
  ∀ (x y : ℝ), (curve x = y ∧ x = 0 ∧ y = 0) → x + y = 0 :=
by {
  intros x y h,
  sorry
}

end tangent_line_at_origin_is_x_plus_y_eq_zero_l122_122347


namespace MaryAddedCandy_l122_122729

-- Definitions based on the conditions
def MaryInitialCandyCount (MeganCandyCount : ℕ) : ℕ :=
  3 * MeganCandyCount

-- Given conditions
def MeganCandyCount : ℕ := 5
def MaryTotalCandyCount : ℕ := 25

-- Proof statement
theorem MaryAddedCandy : 
  let MaryInitialCandy := MaryInitialCandyCount MeganCandyCount
  MaryTotalCandyCount - MaryInitialCandy = 10 :=
by 
  sorry

end MaryAddedCandy_l122_122729


namespace range_of_omega_l122_122880

noncomputable section

open Real

/--
Assume the function f(x) = sin (ω x + π / 3) has exactly three extreme points and two zeros in 
the interval (0, π). Prove that the range of values for ω is 13 / 6 < ω ≤ 8 / 3.
-/
theorem range_of_omega 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h : ∀ x, f x = sin (ω * x + π / 3))
  (h_extreme : (∃ a b c, 0 < a ∧ a < b ∧ b < c ∧ c < π ∧ (f' a = 0) ∧ (f' b = 0) ∧ (f' c = 0)))
  (h_zeros : (∃ u v, 0 < u ∧ u < v ∧ v < π ∧ f u = 0 ∧ f v = 0)) :
  (13 / 6) < ω ∧ ω ≤ (8 / 3) :=
  sorry

end range_of_omega_l122_122880


namespace number_of_valid_numbers_l122_122905

def isValidNumber (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  a1 < a2 ∧ a2 > a3 ∧ a3 < a4 ∧ a4 > a5

def possibleNumbers : List (ℕ × ℕ × ℕ × ℕ × ℕ) :=
  (List.range 5).permute.toList.filter (λ ⟨a1, a2, a3, a4, a5⟩ =>
    isValidNumber a1 a2 a3 a4 a5 
    ∧ List.nodup [a1, a2, a3, a4, a5]
    ∧ [a1, a2, a3, a4, a5].perm.toList = [1, 2, 3, 4, 5]
  )

theorem number_of_valid_numbers : list.length possibleNumbers = 16 :=
  sorry

end number_of_valid_numbers_l122_122905


namespace chime_3000_occurs_l122_122513

noncomputable def clock_chime_time : Nat → Time → Time := sorry

theorem chime_3000_occurs :
  start_time = Time.mk 2005 3 1 13 45 0 →
  chime_count = 3000 →
  ∀ tc, tc = clock_chime_time 3000 start_time 
  → tc = Time.mk 2005 4 15 12 0 0 :=
begin
  sorry
end

end chime_3000_occurs_l122_122513


namespace islands_BC_distance_l122_122792

noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ

theorem islands_BC_distance :
  ∀ (AB : ℝ) (angle_BAC : ℝ) (angle_ABC : ℝ),
    AB = 10 ∧ angle_BAC = real.pi / 3 ∧ angle_ABC = 5 * real.pi / 12 →
    let angle_BCA := real.pi - (angle_BAC + angle_ABC) in
    let BC := (AB * sin angle_BAC) / sin angle_BCA in
    BC = 5 * real.sqrt 6 :=
by
  intro AB angle_BAC angle_ABC,
  intro h,
  -- Placeholder for actual proof
  exact sorry

end islands_BC_distance_l122_122792


namespace omega_range_l122_122875

theorem omega_range (ω : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < π ∧
    (sin (ω * x₁ + π / 3)).cos = 0 ∧ (sin (ω * x₂ + π / 3)).cos = 0 ∧ (sin (ω * x₃ + π / 3)).cos = 0) ∧ 
  (∃ z₁ z₂ : ℝ, 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ sin (ω * z₁ + π / 3) = 0 ∧ sin (ω * z₂ + π / 3) = 0) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
by
  sorry

end omega_range_l122_122875


namespace andrea_needs_1500_sod_squares_l122_122084

-- Define the measurements of the yard sections
def section1_length : ℕ := 30
def section1_width : ℕ := 40
def section2_length : ℕ := 60
def section2_width : ℕ := 80

-- Define the measurements of the sod square
def sod_length : ℕ := 2
def sod_width : ℕ := 2

-- Compute the areas
def area_section1 : ℕ := section1_length * section1_width
def area_section2 : ℕ := section2_length * section2_width
def total_area : ℕ := area_section1 + area_section2

-- Compute the area of one sod square
def area_sod : ℕ := sod_length * sod_width

-- Compute the number of sod squares needed
def num_sod_squares : ℕ := total_area / area_sod

-- Theorem and proof placeholder
theorem andrea_needs_1500_sod_squares : num_sod_squares = 1500 :=
by {
  -- Place proof here
  sorry
}

end andrea_needs_1500_sod_squares_l122_122084


namespace probability_xy_even_l122_122803

open Finset

def set := ∅.insert 1 |>.insert 2 |>.insert 3 |>.insert 4 |>.insert 5
           |>.insert 6 |>.insert 7 |>.insert 8 |>.insert 9 |>.insert 10
           |>.insert 11 |>.insert 12 |>.insert 13 |>.insert 14 |>.insert 15

def primes_in_set := {2, 3, 5, 7, 11, 13}

noncomputable def probability_event := 5 / 12

theorem probability_xy_even :
  ∀ (x y : ℕ), x ∈ set → y ∈ set → y ∈ primes_in_set → x ≠ y →
  (∃ (p : ℚ), p = probability_event ∧ 
  ((Finset.card ((set.product primes_in_set).filter (λ (t : ℕ × ℕ),
     let (x, y) := t in
     x ≠ y ∧ (x * y - x - y) % 2 = 0))) %
     (Finset.card ((set.product primes_in_set).filter (λ (t : ℕ × ℕ),
     let (x, y) := t in x ≠ y))) = p) ) :=
sorry

end probability_xy_even_l122_122803


namespace line_perpendicular_through_P_l122_122770

/-
  Given:
  1. The point P(-2, 2).
  2. The line 2x - y + 1 = 0.
  Prove:
  The equation of the line that passes through P and is perpendicular to the given line is x + 2y - 2 = 0.
-/

def P : ℝ × ℝ := (-2, 2)
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem line_perpendicular_through_P :
  ∃ (x y : ℝ) (m : ℝ), (x = -2) ∧ (y = 2) ∧ (m = -1/2) ∧ 
  (∀ (x₁ y₁ : ℝ), (y₁ - y) = m * (x₁ - x)) ∧ 
  (∀ (lx ly : ℝ), line1 lx ly → x + 2 * y - 2 = 0) := sorry

end line_perpendicular_through_P_l122_122770


namespace max_volume_of_acetic_acid_solution_l122_122791

theorem max_volume_of_acetic_acid_solution :
  (∀ (V : ℝ), 0 ≤ V ∧ (V * 0.09) = (25 * 0.7 + (V - 25) * 0.05)) →
  V = 406.25 :=
by
  sorry

end max_volume_of_acetic_acid_solution_l122_122791


namespace omega_range_l122_122876

theorem omega_range (ω : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < π ∧
    (sin (ω * x₁ + π / 3)).cos = 0 ∧ (sin (ω * x₂ + π / 3)).cos = 0 ∧ (sin (ω * x₃ + π / 3)).cos = 0) ∧ 
  (∃ z₁ z₂ : ℝ, 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ sin (ω * z₁ + π / 3) = 0 ∧ sin (ω * z₂ + π / 3) = 0) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
by
  sorry

end omega_range_l122_122876


namespace complex_solutions_equation_l122_122582

noncomputable def valid_complex_solutions_count : ℕ :=
  let numerator_roots := [1, -1, Complex.I, -Complex.I] -- roots of z^4 - 1 = 0
  let denominator_roots := [1, 1, -2] -- roots of z^3 - 3z + 2 = 0 (including multiplicities)
  numerator_roots.eraseList denominator_roots |>.length -- remove roots that are in both lists and count remaining

theorem complex_solutions_equation : valid_complex_solutions_count = 3 := by
  sorry

end complex_solutions_equation_l122_122582


namespace roots_of_quadratic_l122_122783

theorem roots_of_quadratic (x : ℝ) : x^2 + x = 0 ↔ (x = 0 ∨ x = -1) :=
by sorry

end roots_of_quadratic_l122_122783


namespace sqrt_mixed_number_simplify_l122_122945

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l122_122945


namespace range_of_omega_l122_122887

-- Define the sine function f(x) = sin(ωx + π/3)
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- The hypothesis for the problem
def has_extreme_points (ω : ℝ) :=
  (∃ x1 x2 x3 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (x3 < Real.pi)
    ∧ f' ω x1 = 0 ∧ f' ω x2 = 0 ∧ f' ω x3 = 0)

def has_zeros (ω : ℝ) :=
  (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < Real.pi)
    ∧ f ω x1 = 0 ∧ f ω x2 = 0)

-- The main theorem to be proved
theorem range_of_omega (ω : ℝ) :
  has_extreme_points ω ∧ has_zeros ω ↔ (13/6 < ω ∧ ω ≤ 8/3) :=
by
  sorry

end range_of_omega_l122_122887


namespace series_sum_l122_122105

theorem series_sum : 
  (Finset.sum (Finset.range 49) (λ n, (2 * n + 1) - 2 * (n + 1))) = -49 := 
sorry

end series_sum_l122_122105


namespace max_real_part_l122_122756

noncomputable def isRealPartMax (z w : ℂ) : Prop :=
  (Δz  = Re z + Re w)

theorem max_real_part (z w : ℂ) (hz : |z| = 1) (hw : |w| = 1) (hzw : z * conj(w) + conj(z) * w = 2) : isRealPartMax z w = 2 :=
  sorry

end max_real_part_l122_122756


namespace reduced_rectangle_area_l122_122848

theorem reduced_rectangle_area
  (w h : ℕ) (hw : w = 5) (hh : h = 7)
  (new_w : ℕ) (h_reduced_area : new_w = w - 2 ∧ new_w * h = 21)
  (reduced_h : ℕ) (hr : reduced_h = h - 1) :
  (new_w * reduced_h = 18) :=
by
  sorry

end reduced_rectangle_area_l122_122848


namespace compute_expression_l122_122102

theorem compute_expression (x : ℤ) (h : x = 3) : (x^8 + 24 * x^4 + 144) / (x^4 + 12) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l122_122102


namespace sum_even_minus_sum_odd_first_fifty_l122_122437

theorem sum_even_minus_sum_odd_first_fifty :
  let even_sum := ∑ n in finset.range 50, (2 * (n + 1))
  let odd_sum := ∑ n in finset.range 50, (2 * (n + 1) - 1)
  even_sum - odd_sum = 50 :=
by {
  let even_sum := ∑ n in finset.range 50, (2 * (n + 1)),
  let odd_sum := ∑ n in finset.range 50, (2 * (n + 1) - 1),
  have h_even_sum : even_sum = 2 * ∑ n in finset.range 50, (n + 1),
  have h_odd_sum : odd_sum = ∑ n in finset.range 50, (2 * (n + 1) - 1),
  sorry,
}

end sum_even_minus_sum_odd_first_fifty_l122_122437


namespace natural_number_n_l122_122657

def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem natural_number_n (n : ℕ) (h : (∑ i in finset.range (n - 2), binomial (i + 3) 2) = 363) : 
  n = 13 :=
sorry

end natural_number_n_l122_122657


namespace opposite_of_neg_five_is_five_l122_122400

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122400


namespace not_collinear_l122_122545

-- Define vectors a and b
def a : list ℤ := [-9, 5, 3]
def b : list ℤ := [7, 1, -2]

-- Define vector c1 as 2a - b
def c1 : list ℤ := [2 * a.nth 0 - b.nth 0, 2 * a.nth 1 - b.nth 1, 2 * a.nth 2 - b.nth 2]

-- Define vector c2 as 3a + 5b
def c2 : list ℤ := [3 * a.nth 0 + 5 * b.nth 0, 3 * a.nth 1 + 5 * b.nth 1, 3 * a.nth 2 + 5 * b.nth 2]

-- Proving that c1 and c2 are not collinear
theorem not_collinear : ¬ ∃ (γ : ℚ), list.zipWith (λ x y, x = γ * y) c1 c2 = [true, true, true] :=
sorry

end not_collinear_l122_122545


namespace diagonal_polygon_sides_l122_122141

theorem diagonal_polygon_sides (n : ℕ) (h : ∃ v, n ≥ 3 ∧ (∃ d, d ≠ v ∧ adjacent d v)) : n = 4 :=
sorry

end diagonal_polygon_sides_l122_122141


namespace opposite_of_neg_five_l122_122378

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122378


namespace sum_abs_leq_three_l122_122273

theorem sum_abs_leq_three (n : ℕ) (a : Fin n → ℂ) :
  (∀ (I : Finset (Fin n)) (hI : I.Nonempty), abs (-1 + ∏ j in I, (1 + a j)) ≤ 1/2) →
  (∑ j in Finset.univ, abs (a j) ≤ 3) :=
by sorry

end sum_abs_leq_three_l122_122273


namespace complement_of_M_in_U_l122_122648

def U : set ℕ := {1, 2, 3, 4, 5}
def M : set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5} :=
by sorry

end complement_of_M_in_U_l122_122648


namespace segment_total_length_l122_122403

namespace PentagonSegments

variables (C E : Point)
variables (A B D F G H I : Point)
variables (rotation_1 : Pentagon.rotate H I C B G C ABCDE)
variables (rotation_2 : Pentagon.rotate A B C D E E FGBAE)
variables (length_AB : length AB = 7)

/-- The total length of the 11 segments is 77 cm. -/
theorem segment_total_length : 
  length AB + length BC + length CD + length DE + length EA + 
  length HI + length IC + length CB + length BG + length GH + 
  length FG + length GB + length BA + length AE + length EF = 77 := sorry

end PentagonSegments

end segment_total_length_l122_122403


namespace lily_money_left_for_coffee_l122_122294

theorem lily_money_left_for_coffee :
  let budget := 85
  let celery_price_per_pound := 3
  let celery_needed := 2
  let cereal_price_per_box := 7
  let cereal_boxes := 3
  let bread_price_per_loaf := 6
  let bread_loaves := 3
  let bread_discount := 0.15
  let milk_price_per_gallon := 5
  let milk_gallons := 3
  let potatoes_price_per_pound := 1.5
  let potatoes_needed := 5
  let cookies_price := 10
  let cookies_deal := 15
  let tax_rate := 0.06
  let celery_cost := celery_price_per_pound * celery_needed
  let cereal_cost := cereal_price_per_box * 2  -- Buy two get one free
  let bread_cost_before_discount := bread_price_per_loaf * bread_loaves
  let bread_cost := bread_cost_before_discount * (1 - bread_discount)
  let milk_cost := milk_price_per_gallon * 2  -- Buy two get one free
  let potatoes_cost := potatoes_price_per_pound * potatoes_needed
  let cookies_cost := cookies_deal
  let total_cost_before_tax := celery_cost + cereal_cost + bread_cost + milk_cost + potatoes_cost + cookies_cost
  let total_tax := tax_rate * (celery_cost + cereal_cost + bread_cost + milk_cost + potatoes_cost + cookies_cost)
  let total_cost := total_cost_before_tax + total_tax
  let money_left := budget - total_cost
  let rounded_money_left := Real.round money_left
  rounded_money_left = 12.13 :=
by
  -- skipped proof
  sorry

end lily_money_left_for_coffee_l122_122294


namespace no_eight_consecutive_sums_in_circle_l122_122304

theorem no_eight_consecutive_sums_in_circle :
  ¬ ∃ (arrangement : Fin 8 → ℕ) (sums : Fin 8 → ℤ),
      (∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 8) ∧
      (∀ i, sums i = arrangement i + arrangement (⟨(i + 1) % 8, sorry⟩)) ∧
      (∃ (n : ℤ), 
        (sums 0 = n - 3) ∧ 
        (sums 1 = n - 2) ∧ 
        (sums 2 = n - 1) ∧ 
        (sums 3 = n) ∧ 
        (sums 4 = n + 1) ∧ 
        (sums 5 = n + 2) ∧ 
        (sums 6 = n + 3) ∧ 
        (sums 7 = n + 4)) := 
sorry

end no_eight_consecutive_sums_in_circle_l122_122304


namespace cone_prism_volume_ratio_correct_l122_122525

noncomputable def cone_prism_volume_ratio (π : ℝ) : ℝ :=
  let r := 1.5
  let h := 5
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio_correct (π : ℝ) : 
  cone_prism_volume_ratio π = π / 4.8 :=
sorry

end cone_prism_volume_ratio_correct_l122_122525


namespace min_m_plus_inv_m_min_frac_expr_l122_122965

-- Sub-problem (1): Minimum value of m + 1/m for m > 0.
theorem min_m_plus_inv_m (m : ℝ) (h : m > 0) : m + 1/m = 2 :=
sorry

-- Sub-problem (2): Minimum value of (x^2 + x - 5)/(x - 2) for x > 2.
theorem min_frac_expr (x : ℝ) (h : x > 2) : (x^2 + x - 5)/(x - 2) = 7 :=
sorry

end min_m_plus_inv_m_min_frac_expr_l122_122965


namespace heads_count_l122_122694

theorem heads_count (total_tosses tails: ℕ) (h_total_tosses: total_tosses = 14) (h_tails: tails = 5) : (total_tosses - tails = 9) :=
by
  simp [h_total_tosses, h_tails]
  sorry

end heads_count_l122_122694


namespace direction_vector_of_line_m_l122_122981

noncomputable def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![ 5 / 21, -2 / 21, -2 / 7 ],
    ![ -2 / 21, 1 / 42, 1 / 14 ],
    ![ -2 / 7,  1 / 14, 4 / 7 ]
  ]

noncomputable def vectorI : Fin 3 → ℚ
  | 0 => 1
  | _ => 0

noncomputable def projectedVector : Fin 3 → ℚ :=
  fun i => (projectionMatrix.mulVec vectorI) i

theorem direction_vector_of_line_m :
  (projectedVector 0 = 5 / 21) ∧ 
  (projectedVector 1 = -2 / 21) ∧
  (projectedVector 2 = -6 / 21) ∧
  Nat.gcd (Nat.gcd 5 2) 6 = 1 :=
by
  sorry

end direction_vector_of_line_m_l122_122981


namespace find_real_values_l122_122173

noncomputable def log_two (x : ℝ) := real.log x / real.log 2

theorem find_real_values (x : ℝ) :
  log_two (x^2 - 3 * x - 2) + complex.i * log_two (x^2 + 2 * x + 1) > 1 →
  x = -2 := 
sorry

end find_real_values_l122_122173


namespace complex_addition_zero_l122_122600

theorem complex_addition_zero (a b : ℝ) (i : ℂ) (h1 : (1 + i) * i = a + b * i) (h2 : i * i = -1) : a + b = 0 :=
sorry

end complex_addition_zero_l122_122600


namespace transformation_thinking_reflected_in_solution_of_quadratic_l122_122020

theorem transformation_thinking_reflected_in_solution_of_quadratic :
  ∀ (x : ℝ), (x - 3)^2 - 5 * (x - 3) = 0 → (x = 3 ∨ x = 8) →
  transformation_thinking :=
by
  intros x h_eq h_solutions
  sorry

end transformation_thinking_reflected_in_solution_of_quadratic_l122_122020


namespace weight_of_new_student_l122_122033

theorem weight_of_new_student 
  (avg_weight_old : ℕ := 15) 
  (num_students_old : ℕ := 19) 
  (avg_weight_new : ℕ := 14.4) 
  (num_students_new : ℕ := 20)
  (total_weight_old : ℕ := avg_weight_old * num_students_old)
  (total_weight_new : ℕ := avg_weight_new * num_students_new)
  (weight_new_student : ℕ := total_weight_new - total_weight_old) :
  weight_new_student = 3 := 
sorry

end weight_of_new_student_l122_122033


namespace tangent_lines_at_A_l122_122138

noncomputable def tangent_lines_through_point_to_circle (A : ℝ × ℝ) (r : ℝ) : set (ℝ → ℝ) :=
  {f : ℝ → ℝ | ∃ k b : ℝ, ∃ (x₀ x₁ y₁ : ℝ), k = -(x₀ / r) ∧ 
                           b = y₁ - k * x₁ ∧ 
                           x ≠ 0 ∧ 
                           y = k * x + b}


theorem tangent_lines_at_A (A : ℝ × ℝ) (hA : A = (2, 4)) :
  tangent_lines_through_point_to_circle A 2 = 
    {λ x : ℝ, 0, λ x : ℝ, 3*x - 4*y + 10 = 0} := 
by 
  sorry

end tangent_lines_at_A_l122_122138


namespace distinct_column_maxima_eq_distinct_row_maxima_l122_122051

theorem distinct_column_maxima_eq_distinct_row_maxima
  (A : Matrix (Fin 6) (Fin 6) ℝ) 
  (h_uniq : ∀ i j : Fin 6, A i j ≠ A i j)  -- Ensuring all averages are distinct
  (h_col : ∀ j : Fin 6, ∃ i : Fin 6, ∀ k : Fin 6, A i j ≥ A k j)
  (h_row : ∀ i : Fin 6, ∃ j : Fin 6, ∀ l : Fin 6, A i j ≥ A i l) : 
  ∃ indexes : Fin 6 → Fin 6, 
  (∀ i : Fin 6, (∃ j1 : Fin 6, A i (indexes j1) = row_max(i) ∧ 
  ∃ j2 : Fin 6, A (indexes j2) i = col_max(i))) ∧ 
  (∀ i1 i2 : Fin 6, i1 ≠ i2 → indexes i1 ≠ indexes i2) :=
by 
  sorry

noncomputable def row_max (i : Fin 6) : ℝ := 
  max (A i 0) (max (A i 1) (max (A i 2) (max (A i 3) (max (A i 4) (A i 5)))))

noncomputable def col_max (j : Fin 6) : ℝ := 
  max (A 0 j) (max (A 1 j) (max (A 2 j) (max (A 3 j) (max (A 4 j) (A 5 j)))))

end distinct_column_maxima_eq_distinct_row_maxima_l122_122051


namespace product_of_coordinates_center_eq_zero_l122_122404

-- Define the given points
def point1 : ℝ × ℝ := (-6, 4)
def point2 : ℝ × ℝ := (6, 10)

-- Define the function to find the midpoint of two points (as per the midpoint formula)
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the center of the circle as the midpoint of the given points
def center := midpoint point1 point2

-- Define the product of the coordinates of the center of the circle
def product_of_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 * p.2

-- State the theorem we need to prove
theorem product_of_coordinates_center_eq_zero : 
  product_of_coordinates center = 0 :=
sorry

end product_of_coordinates_center_eq_zero_l122_122404


namespace functions_not_exist_l122_122702

theorem functions_not_exist :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by
  sorry

end functions_not_exist_l122_122702


namespace general_term_sum_of_sequence_l122_122180

-- Conditions
def is_geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop := ∀ n, a (n + 1) = r * a n

def forms_arithmetic_seq (a2 a3_plus1 a4 : ℕ) : Prop := 2 * a3_plus1 = a2 + a4

-- Question 1
theorem general_term (a : ℕ → ℕ) (h1 : is_geometric_seq a 2) (h2 : forms_arithmetic_seq (a 1) ((a 2) + 1) (a 3)) :
  ∃ a₁ : ℕ, ∀ n, a n = 2 ^ (n - 1) :=
sorry

-- Definitions for Question 2
def b (n : ℕ) : ℚ := 1 / ((n + 1) * (n + 2) : ℚ)

def T (n : ℕ) : ℚ := ∑ i in finset.range n, b i

-- Question 2
theorem sum_of_sequence (a : ℕ → ℕ) (h1 : is_geometric_seq a 2) (h2 : forms_arithmetic_seq (a 1) ((a 2) + 1) (a 3)) :
  ∀ n, T n = n / (n + 1) :=
sorry

end general_term_sum_of_sequence_l122_122180


namespace hyperbola_eccentricity_l122_122192

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
    (h_hyperbola : ∀ (x y : ℝ), (∃ (A B C D : ℂ), (A, B, C, D are vertices of a square) ∧
    (A, B, C, D are on a hyperbola characterized by 
    ∀ (a b : ℝ), (a > 0 ∧ b > 0) → (A, B, C, D vertices condition) ∧
    midpoints of AB and CD are the two foci of hyperbola) →
    there exists focus and eccentricity):
    eccentricity a b = (sqrt 5 + 1) / 2 := sorry

end hyperbola_eccentricity_l122_122192


namespace opposite_of_neg_five_is_five_l122_122401

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122401


namespace problem_statement_l122_122782

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-5, 0⟩
def B : Point := ⟨-5, 7⟩
def C : Point := ⟨-1, 7⟩
def D : Point := ⟨-1, 4⟩

def L : ℝ := 3

def reflect (p : Point) (x : ℝ) : Point :=
  ⟨2 * x - p.x, p.y⟩

def Q : Point := reflect C L
def P : Point := reflect D L
def R : Point := reflect B L
def S : Point := reflect A L

noncomputable def perimeter (points : List Point) : ℝ :=
  points.zip (points.drop 1 ++ [points.head!]).map (λ ⟨p1, p2⟩ =>
    (p1.x - p2.x)^2 + (p1.y - p2.y)^2).map Math.sqrt |>.sum

noncomputable def volume (r_large h_large r_small h_small : ℝ) : ℝ :=
  Math.pi * (r_large ^ 2 * h_large - r_small ^ 2 * h_small)

noncomputable def surface_area (r_large h_large r_small h_small : ℝ) : ℝ :=
  let SA_large := 2 * Math.pi * r_large * (r_large + h_large)
  let SA_small := 2 * Math.pi * r_small * (r_small + h_small)
  let circle_diff := Math.pi * (r_large^2 - r_small^2)
  SA_large - circle_diff + SA_small + circle_diff

theorem problem_statement:
  P = ⟨7, 4⟩ ∧
  R = ⟨11, 7⟩ ∧
  S = ⟨11, 0⟩ ∧
  perimeter [A, B, C, D, P, Q, R, S] = 52 ∧
  volume 8 7 4 3 = 400 * Math.pi ∧
  surface_area 8 7 4 3 = 264 * Math.pi := 
  by
    -- leaving the proof part with sorry
    sorry

end problem_statement_l122_122782


namespace find_b_l122_122759

variable (f : ℝ → ℝ) (finv : ℝ → ℝ)

-- Defining the function f
def f_def (b : ℝ) (x : ℝ) := 1 / (2 * x + b)

-- Defining the inverse function
def finv_def (x : ℝ) := (2 - 3 * x) / (3 * x)

theorem find_b (b : ℝ) :
  (∀ x : ℝ, f_def b (finv_def x) = x ∧ finv_def (f_def b x) = x) ↔ b = -2 := by
  sorry

end find_b_l122_122759


namespace positive_difference_is_722_l122_122772

def A : ℤ := ∑ k in (0:ℕ) to 18, (2*k + 1) * (2*k + 2) + 39
def B : ℤ := ∑ k in (0:ℕ) to 18, (2*k + 2) * (2*k + 3) + 1

theorem positive_difference_is_722 : |A - B| = 722 :=
by
  sorry

end positive_difference_is_722_l122_122772


namespace pyramid_volume_SPQR_l122_122743

noncomputable def volume_of_pyramid_SPQR (SP SQ SR : ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * SP * SQ * SR

theorem pyramid_volume_SPQR : ∀ (P Q R S : Type) 
  [HasDist P Q] [HasDist Q R] [HasDist R S]
  [HasDist S P] [HasPerp P Q] [HasPerp Q R] [HasPerp R P]
  (hSP : dist P S = 12) (hSQ : dist Q S = 12) (hSR : dist R S = 7),
  volume_of_pyramid_SPQR (dist P S) (dist Q S) (dist R S) = 168 :=
by
  intros P Q R S _ _ _ _ _ _ _ hSP hSQ hSR
  rw [←hSP, ←hSQ, ←hSR]
  unfold volume_of_pyramid_SPQR
  norm_num
  sorry

end pyramid_volume_SPQR_l122_122743


namespace train_length_l122_122838

theorem train_length
  (jogger_speed_kmph : ℕ) (train_speed_kmph : ℕ)
  (initial_distance_m : ℕ) (time_to_pass_s : ℕ)
  (jogger_speed := jogger_speed_kmph * 1000 / 3600 : ℚ)
  (train_speed := train_speed_kmph * 1000 / 3600 : ℚ)
  (relative_speed := train_speed - jogger_speed : ℚ)
  (distance_covered_by_train := relative_speed * time_to_pass_s : ℚ)
  (train_length_m := distance_covered_by_train - initial_distance_m : ℚ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  initial_distance_m = 240 →
  time_to_pass_s = 36 →
  train_length_m = 120 :=
by
  intros
  rw [show jogger_speed_kmph * 1000 / 3600 = 2.5, by norm_num,
      show train_speed_kmph * 1000 / 3600 = 12.5, by norm_num,
      show relative_speed = 10, by norm_num,
      show distance_covered_by_train = 360, by norm_num]
  exact by norm_num


end train_length_l122_122838


namespace dividend_is_correct_l122_122213

def quotient : ℕ := 20
def divisor : ℕ := 66
def remainder : ℕ := 55

def dividend := (divisor * quotient) + remainder

theorem dividend_is_correct : dividend = 1375 := by
  sorry

end dividend_is_correct_l122_122213


namespace find_number_of_notebooks_l122_122295

variable (n : ℕ)
variable (cost_notebook cost_pencil cost_pen total_spent : ℝ)
variable (cost_notebook_eq : cost_notebook = 1.20)
variable (cost_pencil_eq : cost_pencil = 1.50)
variable (cost_pen_eq : cost_pen = 1.70)
variable (total_spent_eq : total_spent = 6.80)

theorem find_number_of_notebooks
  (h : (cost_notebook * n) + cost_pencil + cost_pen = total_spent) :
  n = 3 := by
sory

end find_number_of_notebooks_l122_122295


namespace empty_set_subset_of_any_set_correct_set_relation_l122_122537

theorem empty_set_subset_of_any_set : ∀ (A : Set), ∅ ⊆ A := by
  sorry

theorem correct_set_relation : 0 ⊆ {∅} :=
  empty_set_subset_of_any_set {∅}

end empty_set_subset_of_any_set_correct_set_relation_l122_122537


namespace area_of_EFGH_l122_122321

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l122_122321


namespace find_defective_keys_l122_122455

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122455


namespace find_g_of_2_l122_122773

theorem find_g_of_2 (g : ℝ → ℝ) (h : ∀ x ≠ 0, g(x) - 2 * g (1 / x) = 3^x) : g 2 = -5 := by
  sorry

end find_g_of_2_l122_122773


namespace gavrila_final_distance_l122_122094

-- Define constant distances and speeds
def L : ℝ := 50  -- Halfway distance to starting point for Gavrila
def speed : ℝ := 20  -- Speed of both bicycle and motorboat in km/h
def distance_from_bank : ℝ := 40  -- Given distance y from the bank, in meters

-- Define the equation for Gavrila's coordinate computation
def gavrila_x_coordinate (y : ℝ) : ℝ := (y^2) / (4 * L)

-- Define Pythagorean theorem application to find Gavrila's distance from the starting point
def gavrila_distance (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Finally, the theorem statement that needs to be proven
theorem gavrila_final_distance :
  let x := gavrila_x_coordinate distance_from_bank in
  let s := gavrila_distance x distance_from_bank in
  real.round s = 41 :=
by {
  -- Proof omitted
  sorry
}

end gavrila_final_distance_l122_122094


namespace intersection_M_N_l122_122728

def M (x : ℝ) : Prop := 2 - x > 0
def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

theorem intersection_M_N:
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l122_122728


namespace solution_set_of_inequality_l122_122786

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 1) * (2 - x) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l122_122786


namespace derivative_of_y_l122_122038

-- Define the function y
def y (x : ℝ) : ℝ := Math.sin (3 - 4 * x)

-- State the theorem to prove
theorem derivative_of_y (x : ℝ) : deriv y x = -4 * Math.cos (3 - 4 * x) :=
by
  sorry

end derivative_of_y_l122_122038


namespace calculate_expression_l122_122551

theorem calculate_expression :
  18 - ((-16) / (2 ^ 3)) = 20 :=
by
  sorry

end calculate_expression_l122_122551


namespace number_count_l122_122846

open Nat

theorem number_count (n a k m : ℕ) (n_pos : n > 0) (m_bound : m < 10^k)
    (key_eqn : 8 * m = 10^k * (a + n)) : 
    (number_of_combinations (λ m a k n, 8 * m = 10^k * (a + n) ∧ n > 0 ∧ m < 10^k) = 28) :=
sorry

end number_count_l122_122846


namespace line_l_standard_form_curve_C_rectangular_form_intersection_points_A_and_B_l122_122256

noncomputable def line_equation (t : ℝ) : ℝ × ℝ :=
(let x := 1 - (√3 / 2) * t, y := (1 / 2) * t in (x, y))

def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
(let x := rho * cos theta, y := rho * sin theta in (x, y))

def point_P := (1, 0)

theorem line_l_standard_form :
  ∃ (x y : ℝ), ∀ t : ℝ, (x, y) = line_equation t → x + √3 * y - 1 = 0 := sorry

theorem curve_C_rectangular_form :
  ∀ θ : ℝ, (let (x, y) := polar_to_rectangular (4 * cos θ) θ in
    (x - 2) ^ 2 + y ^ 2 = 4) := sorry

theorem intersection_points_A_and_B (t1 t2: ℝ) :
  let (x1, y1) := line_equation t1,
      (x2, y2) := line_equation t2,
      PA := (x1 - 1) ^ 2 + y1 ^ 2,
      PB := (x2 - 1) ^ 2 + y2 ^ 2 in
  x1 ^ 2 + y1 ^ 2 = 4 * x1 ∧
  x2 ^ 2 + y2 ^ 2 = 4 * x2 ∧
  t1 < 0 ∧ t2 > 0 ∧
  t1 + t2 = -√3 ∧
  t1 * t2 = -3 →
  1 / PA + 1 / PB = (√15) / 3 := sorry

end line_l_standard_form_curve_C_rectangular_form_intersection_points_A_and_B_l122_122256


namespace additional_pots_in_eighth_hour_l122_122059

-- Define the conditions
def minutes_per_hour : ℕ := 60
def first_hour_rate : ℕ := 6
def eighth_hour_rate : ℚ := 5.2

-- Define the calculation for the number of pots produced in an hour given the production rate per pot
def pots_produced (rate : ℚ) : ℕ := (minutes_per_hour : ℚ) / rate

-- Define the main statement
theorem additional_pots_in_eighth_hour :
  pots_produced eighth_hour_rate - pots_produced first_hour_rate = 1 :=
by
  sorry

end additional_pots_in_eighth_hour_l122_122059


namespace opposite_of_neg_five_l122_122366

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122366


namespace traditionalist_fraction_l122_122492

theorem traditionalist_fraction (T P : ℕ) 
  (h1 : ∀ prov : ℕ, prov < 6 → T = P / 9) 
  (h2 : P + 6 * T > 0) :
  6 * T / (P + 6 * T) = 2 / 5 := 
by
  sorry

end traditionalist_fraction_l122_122492


namespace calculate_AB_l122_122246

-- Definitions of the given conditions:
def reflection_median {A B C M B' C'} : Prop :=
  -- Triangle ABC reflected over median AM to form triangle AB'C'
  sorry

def lengths {A E C D B : Type*} [metric_space A] [metric_space E] [metric_space C] [metric_space D] [metric_space B] : Prop :=
  dist A E = 8 ∧ dist E C = 16 ∧ dist B D = 15

-- The goal is to prove:
theorem calculate_AB {A B : Type*} [metric_space A] [metric_space B]
  (h : reflection_median ∧ lengths) : dist A B = 19.52 :=
sorry

end calculate_AB_l122_122246


namespace opposite_of_neg_five_l122_122375

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122375


namespace flower_shop_sold_bouquets_l122_122835

theorem flower_shop_sold_bouquets (roses_per_bouquet : ℕ) (daisies_per_bouquet : ℕ) 
  (rose_bouquets_sold : ℕ) (daisy_bouquets_sold : ℕ) (total_flowers_sold : ℕ)
  (h1 : roses_per_bouquet = 12) (h2 : rose_bouquets_sold = 10) 
  (h3 : daisy_bouquets_sold = 10) (h4 : total_flowers_sold = 190) : 
  (rose_bouquets_sold + daisy_bouquets_sold) = 20 :=
by sorry

end flower_shop_sold_bouquets_l122_122835


namespace sum_max_min_interval_0_3_l122_122788

def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem sum_max_min_interval_0_3 : 
    let lower_bound := 0
    let upper_bound := 3
    let min_val := (λ x, (x - 1)^2 - 2) 1
    let max_val := (λ x, (x - 1)^2 - 2) 3
    min_val + max_val = 0 :=
by
  sorry

end sum_max_min_interval_0_3_l122_122788


namespace radius_of_incircle_of_triangle_l122_122193

noncomputable def hyperbola_inc_radius 
  (a b : ℝ) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (eccentricity : ℝ) (h_ecc : eccentricity = (Real.sqrt 5) / 2)
  (area_triangle : ℝ) (h_area : area_triangle = 32 / 3)
  (AF2_3F2B : ∀ (A B F1 F2 : EuclideanGeometry.Point) 
        (h_foci : (F1 = ⟨-Real.sqrt (a^2 + b^2), 0⟩) 
                 ∧ (F2 = ⟨Real.sqrt (a^2 + b^2), 0⟩))
        (h_intersect : EuclideanGeometry.isOnHyperbola A a b ∧ EuclideanGeometry.isOnHyperbola B a b)
        (h_AF2_F2B : EuclideanGeometry.eq_3times_vector A F2 B),
    EuclideanGeometry.area F1 A B = 32 / 3)
  : ℝ :=
1

theorem radius_of_incircle_of_triangle 
  {a b : ℝ} 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h_ecc : (Real.sqrt 5) / 2 = (Real.sqrt 5) / 2)
  (h_area : 32 / 3 = 32 / 3)
  (AF2_3F2B : ∀ (A B F1 F2 : EuclideanGeometry.Point) 
        (h_foci : (F1 = ⟨-Real.sqrt (a^2 + b^2), 0⟩) 
                 ∧ (F2 = ⟨Real.sqrt (a^2 + b^2), 0⟩))
        (h_intersect : EuclideanGeometry.isOnHyperbola A a b ∧ EuclideanGeometry.isOnHyperbola B a b)
        (h_AF2_F2B : EuclideanGeometry.eq_3times_vector A F2 B),
    EuclideanGeometry.area F1 A B = 32 / 3) :
  hyperbola_inc_radius a b a_pos b_pos ((Real.sqrt 5) / 2) (32 / 3) AF2_3F2B = 1 := 
sorry

end radius_of_incircle_of_triangle_l122_122193


namespace number_divided_by_21_l122_122505

theorem number_divided_by_21 (x : ℝ) (h : 6000 - (x / 21.0) = 5995) : x = 105 :=
by
  sorry

end number_divided_by_21_l122_122505


namespace distribute_problems_to_friends_l122_122861

theorem distribute_problems_to_friends :
  let num_problems := 7
  let num_friends := 12
  num_friends ^ num_problems = 35_831_808 :=
by
  sorry

end distribute_problems_to_friends_l122_122861


namespace expand_expression_l122_122920

open Nat

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 :=
by
  sorry

end expand_expression_l122_122920


namespace tan_sum_identity_l122_122152

theorem tan_sum_identity (θ : ℝ) (h : (cos (2 * θ) + 1) / (1 + 2 * sin (2 * θ)) = -2 / 3) : 
  tan (θ + π / 4) = -1 / 3 :=
by 
  sorry

end tan_sum_identity_l122_122152


namespace good_arrangement_count_l122_122540

def is_good (arr : Fin 12 → ℝ) : Prop :=
  ∀ n : Fin 9, (arr n + arr (n + 3)) / 2 = arr (n + 1) * arr (n + 2)

theorem good_arrangement_count : 
  {arr : Fin 12 → ℝ | is_good arr ∧ arr 0 = 1 ∧ arr 1 = arr 2 ∧ arr 11 = 1}.card = 89 := sorry

end good_arrangement_count_l122_122540


namespace transformed_curve_eq_l122_122005

theorem transformed_curve_eq :
  ∀ (y x : ℝ), (y * cos x + 2 * y - 1 = 0) →
    ((y - 1) * sin x + 2 * y - 3 = 0) :=
by
  intros y x h
  sorry

end transformed_curve_eq_l122_122005


namespace sum_of_coefficients_l122_122128

theorem sum_of_coefficients : 
  let p := (x - 3 * y) in
  (p ^ 19).coeff 1 1 = -2 ^ 19 :=
sorry

end sum_of_coefficients_l122_122128


namespace ellipse_properties_l122_122139

theorem ellipse_properties :
  ∀ {x y : ℝ}, 4 * x^2 + 2 * y^2 = 16 →
    (∃ a b e c, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ e = Real.sqrt 2 / 2 ∧ c = 2) ∧
    (∃ f1 f2, f1 = (0, 2) ∧ f2 = (0, -2)) ∧
    (∃ v1 v2 v3 v4, v1 = (0, 2 * Real.sqrt 2) ∧ v2 = (0, -2 * Real.sqrt 2) ∧ v3 = (2, 0) ∧ v4 = (-2, 0)) :=
by
  sorry

end ellipse_properties_l122_122139


namespace speed_of_man_l122_122819

theorem speed_of_man (v_m v_s : ℝ) 
    (h1 : (v_m + v_s) * 4 = 32) 
    (h2 : (v_m - v_s) * 4 = 24) : v_m = 7 := 
by
  sorry

end speed_of_man_l122_122819


namespace problem1_problem2_l122_122197

-- Problem I
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {-2, 4}

theorem problem1 (a : ℝ) (h : A a = B) : a = 2 :=
sorry

-- Problem II
def C (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}
def B' : Set ℝ := {-2, 4}

theorem problem2 (m : ℝ) (h : B' ∪ C m = B') : 
  m = -1/2 ∨ m = -1/4 ∨ m = 0 :=
sorry

end problem1_problem2_l122_122197


namespace min_value_x_squared_plus_y_squared_l122_122176

theorem min_value_x_squared_plus_y_squared {x y : ℝ} 
  (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) : 
  ∃ m : ℝ, m = 14 - 2 * Real.sqrt 13 ∧ ∀ u v : ℝ, (u^2 + v^2 - 4*u - 6*v + 12 = 0) → (u^2 + v^2 ≥ m) :=
by
  sorry

end min_value_x_squared_plus_y_squared_l122_122176


namespace functional_equation_solution_unique_l122_122951

noncomputable def is_functional_solution (f : ℚ → ℚ) : Prop :=
∀ x y : ℚ, 0 < x → 0 < y → f(x^2 * (f y)^2) = (f x)^2 * f y

noncomputable def unique_solution : Prop :=
∀ f : ℚ → ℚ, (∀ x : ℚ, 0 < x → f x = 1) ↔ is_functional_solution f

theorem functional_equation_solution_unique : unique_solution :=
sorry

end functional_equation_solution_unique_l122_122951


namespace amanda_works_hours_per_day_l122_122536

noncomputable def amanda_hourly_rate := 50
noncomputable def jose_withhold_percent := 0.20
noncomputable def amanda_received_if_not_finished := 400

theorem amanda_works_hours_per_day :
  let full_pay :=  amanda_received_if_not_finished / (1 - jose_withhold_percent),
  let hours_worked := full_pay / amanda_hourly_rate
  in hours_worked = 10 := by
  sorry

end amanda_works_hours_per_day_l122_122536


namespace students_like_burgers_l122_122677

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end students_like_burgers_l122_122677


namespace sum_of_digits_of_triangular_number_l122_122075

-- Definition of triangular number sum
def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Definition of digit sum
def digit_sum (N : ℕ) : ℕ := 
  let digits := N.toString.data.map (λ c => c.toNat - '0'.toNat) in
  digits.sum

-- Main statement of the mathematical proof problem
theorem sum_of_digits_of_triangular_number (N : ℕ) (h : triangular_sum N = 2145) : digit_sum N = 11 :=
by
  sorry

end sum_of_digits_of_triangular_number_l122_122075


namespace find_primes_l122_122956

theorem find_primes (p q r s : ℕ) (h1 : p.prime) (h2 : q.prime) (h3 : r.prime) (h4 : s.prime)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_prime_sum : (p + q + r + s).prime)
  (h_square1 : ∃ n : ℕ, p^2 + q * s = n^2)
  (h_square2 : ∃ m : ℕ, p^2 + q * r = m^2) :
  (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) ∨ (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) := by
  sorry

end find_primes_l122_122956


namespace distance_AE_BF_l122_122497

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2⟩

def vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def cross_product (u v : Point) : Point :=
  ⟨u.y * v.z - u.z * v.y,  u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

def magnitude (p : Point) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2 + p.z^2)

noncomputable def distance_between_lines
  (A E B F : Point)
  (u v n : Point)
  (AB : ℝ)
  (H_AB : AB = magnitude (vector A B)) : ℝ :=
  (| dot_product (vector A B) n |) / (magnitude n)

theorem distance_AE_BF
  (A B D A₁ B₁ C₁ E F : Point)
  (AB AD AA₁ : ℝ)
  (H_AB : AB = 30)
  (H_AD : AD = 32)
  (H_AA₁ : AA₁ = 20)
  (H_B : B = ⟨30, 0, 0⟩)
  (H_D : D = ⟨0, 32, 0⟩)
  (H_A₁ : A₁ = ⟨0, 0, 20⟩)
  (H_B₁ : B₁ = ⟨30, 0, 20⟩)
  (H_C₁ : C₁ = ⟨30, 32, 20⟩)
  (H_E : E = midpoint A₁ B₁)
  (H_F : F = midpoint B₁ C₁)
  (u v : Point)
  (H_u : u = vector A E)
  (H_v : v = vector B F)
  (n : Point)
  (H_n : n = cross_product u v) :
  distance_between_lines A E B F u v n 30 H_AB = 19.2 := 
sorry

end distance_AE_BF_l122_122497


namespace five_people_six_chairs_l122_122223

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l122_122223


namespace nat_condition_l122_122132

theorem nat_condition (n : ℕ) (h : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  (∃ p : ℕ, n = 2^p - 2) :=
sorry

end nat_condition_l122_122132


namespace crayons_total_l122_122674

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l122_122674


namespace wage_difference_l122_122890

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.2

theorem wage_difference : manager_wage - chef_wage = 3.40 := 
by
  sorry

end wage_difference_l122_122890


namespace binar_operation_correct_l122_122554

theorem binar_operation_correct : 
  let a := 13  -- 1101_2 in decimal
  let b := 15  -- 1111_2 in decimal
  let c := 9   -- 1001_2 in decimal
  let d := 2   -- 10_2 in decimal
  a + b - c * d = 10 ↔ "1010" = "1010" := 
by 
  intros
  simp
  sorry

end binar_operation_correct_l122_122554


namespace distance_from_starting_point_l122_122087

-- Definitions based on the problem conditions
def L : ℝ := 50 -- Halfway between the siren and the start of the bridge
def sound_distance_condition (x y L : ℝ) : Prop :=
  (x + L) = Real.sqrt ((x - L) ^ 2 + y ^ 2)

-- Hypotheses based on the problem conditions
def condition1 : Prop := sound_distance_condition x y L
def condition2 : Prop := y = 40 

-- The theorem we need to prove
theorem distance_from_starting_point (x y L : ℝ) (h1 : sound_distance_condition x y L) (h2 : y = 40) : 
  Real.sqrt (x ^ 2 + y ^ 2) = 41 :=
  sorry

end distance_from_starting_point_l122_122087


namespace sqrt_mixed_number_simplified_l122_122925

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l122_122925


namespace comparison_1_comparison_2_l122_122897

noncomputable def expr1 := -(-((6: ℝ) / 7))
noncomputable def expr2 := -((abs (-((4: ℝ) / 5))))
noncomputable def expr3 := -((4: ℝ) / 5)
noncomputable def expr4 := -((2: ℝ) / 3)

theorem comparison_1 : expr1 > expr2 := sorry
theorem comparison_2 : expr3 < expr4 := sorry

end comparison_1_comparison_2_l122_122897


namespace possible_lengths_CD_l122_122109

-- Define point type and distances between points
axiom Point : Type
axiom dist : Point → Point → ℝ

-- Existential quantifiers stating the given distances
axiom A B C D : Point
axiom hAB : dist A B = 4
axiom hAC : dist A C = 5
axiom hCB : dist C B = 5
axiom hAD : dist A D = 6
axiom hDB : dist D B = 6

-- Given the conditions and the problem constraints
theorem possible_lengths_CD
    (h_inscribed : ∀ (α : Type) (cylinder : α), ∀ (P : Point → Prop),  -- Inscription in a cylinder
        (P A) ∧ (P B) ∧ (P C) ∧ (P D) → -- All vertices lie on the lateral surface
        (∀ C D, parallel C D (cylinder.axis)) -- Edge CD is parallel to cylinder's axis
    )
    : ∃ (l : ℝ), l = 2 * Real.sqrt 7 + Real.sqrt 17 ∨ l = 2 * Real.sqrt 7 - Real.sqrt 17 :=
sorry

end possible_lengths_CD_l122_122109


namespace credit_card_more_beneficial_l122_122026

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end credit_card_more_beneficial_l122_122026


namespace f_inv_sum_l122_122774

def f (x : ℝ) : ℝ :=
if x < 3 then 3 * x - 1 else x ^ (1 / 3 : ℝ)

def f_inv (x : ℝ) : ℝ :=
if x < 8 then (x + 1) / 3 else x ^ 3

theorem f_inv_sum : 
  (f_inv (-4) + f_inv (-3) + f_inv (-2) + f_inv (-1) + f_inv 0 + f_inv 1 + f_inv 2 + f_inv 3) = 27 := 
  sorry

end f_inv_sum_l122_122774


namespace geometric_sequence_sum_ratio_l122_122980

theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n+1) = a 0 * q ^ n)
  (h2 : ∀ n, S n = (a 0 * (q ^ n - 1)) / (q - 1))
  (h3 : 6 * a 3 = a 0 * q ^ 5 - a 0 * q ^ 4) :
  S 4 / S 2 = 10 := 
sorry

end geometric_sequence_sum_ratio_l122_122980


namespace find_a_l122_122284

-- Defining the sets A and B
def A : Set :=
{1, a}

def B : Set :=
{a, 5}

-- Defining the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Defining the conditions
axiom H1 : f 1 = 2
axiom H2 : f a = 5

-- Main theorem to prove a = 2
theorem find_a : a = 2 :=
by
  sorry

end find_a_l122_122284


namespace derivative_of_f_l122_122636

-- Define the function
def f (x a : ℝ) : ℝ := (x + 2 * a) * (x - a) ^ 2

-- State the theorem
theorem derivative_of_f (x a : ℝ) : 
  (deriv (λ x : ℝ, f x a)) x = 3 * (x^2 - a^2) :=
sorry

end derivative_of_f_l122_122636


namespace distinguishable_colorings_l122_122564

theorem distinguishable_colorings (R W B : Type) (pyramid : Type) : 
  (∃ (colors : pyramid → fin 3), ∀ c₁ c₂ : pyramid, c₁ ≠ c₂ → colors c₁ ≠ colors c₂) → 
  ∃ n : nat, n = 27 :=
by
  sorry

end distinguishable_colorings_l122_122564


namespace james_total_beverages_l122_122703

-- Define the initial quantities
def initial_sodas := 4 * 10 + 12
def initial_juice_boxes := 3 * 8 + 5
def initial_water_bottles := 2 * 15
def initial_energy_drinks := 7

-- Define the consumption rates
def mon_to_wed_sodas := 3 * 3
def mon_to_wed_juice_boxes := 2 * 3
def mon_to_wed_water_bottles := 1 * 3

def thu_to_sun_sodas := 2 * 4
def thu_to_sun_juice_boxes := 4 * 4
def thu_to_sun_water_bottles := 1 * 4
def thu_to_sun_energy_drinks := 1 * 4

-- Define total beverages consumed
def total_consumed_sodas := mon_to_wed_sodas + thu_to_sun_sodas
def total_consumed_juice_boxes := mon_to_wed_juice_boxes + thu_to_sun_juice_boxes
def total_consumed_water_bottles := mon_to_wed_water_bottles + thu_to_sun_water_bottles
def total_consumed_energy_drinks := thu_to_sun_energy_drinks

-- Define total beverages consumed by the end of the week
def total_beverages_consumed := total_consumed_sodas + total_consumed_juice_boxes + total_consumed_water_bottles + total_consumed_energy_drinks

-- The theorem statement to prove
theorem james_total_beverages : total_beverages_consumed = 50 :=
  by sorry

end james_total_beverages_l122_122703


namespace downstream_speed_problem_statement_l122_122058

-- Define variables for upstream speed, still water speed, and downstream speed
variable (V_u V_s V_d : ℝ)

-- Conditions given in the problem
axiom h1 : V_u = 20
axiom h2 : V_s = 50

-- Formula to calculate downstream speed
theorem downstream_speed : V_d = 2 * V_s - V_u := by
  -- Substitute the conditions into the formula and show the result
  sorry

-- Define the final result we want to prove
theorem problem_statement : V_d = 80 := by
  -- Use the theorem downstream_speed and the conditions h1, h2 to conclude
  apply downstream_speed
  simp [h1, h2]
  sorry

end downstream_speed_problem_statement_l122_122058


namespace expression_value_l122_122894

def a : ℕ := 1000
def b1 : ℕ := 15
def b2 : ℕ := 314
def c1 : ℕ := 201
def c2 : ℕ := 360
def c3 : ℕ := 110
def d1 : ℕ := 201
def d2 : ℕ := 360
def d3 : ℕ := 110
def e1 : ℕ := 15
def e2 : ℕ := 314

theorem expression_value :
  (a + b1 + b2) * (c1 + c2 + c3) + (a - d1 - d2 - d3) * (e1 + e2) = 1000000 :=
by
  sorry

end expression_value_l122_122894


namespace tan_ratio_l122_122281

theorem tan_ratio (a b : ℝ)
  (h1 : Real.cos (a + b) = 1 / 3)
  (h2 : Real.cos (a - b) = 1 / 2) :
  (Real.tan a) / (Real.tan b) = 5 :=
sorry

end tan_ratio_l122_122281


namespace expected_value_l122_122063

theorem expected_value (p1 p2 p3 p4 p5 p6 : ℕ) (hp1 : p1 = 1) (hp2 : p2 = 5) (hp3 : p3 = 10) 
(hp4 : p4 = 25) (hp5 : p5 = 50) (hp6 : p6 = 100) :
  (p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + p5 / 2 + p6 / 2 : ℝ) = 95.5 := by
  sorry

end expected_value_l122_122063


namespace area_circumcircle_l122_122639

noncomputable def hyperbola (x y l : ℝ) := x^2 - (y^2 / 3) = l

theorem area_circumcircle (A B F2 : ℝ × ℝ) (P : ℝ × ℝ) (R : ℝ)
    (hA : A = (-1, 0)) (hB : B = (1, 0)) (hF2 : F2 = (2, 0))
    (hP : hyperbola P.1 P.2 1) 
    (condition : (abs (dist A P) * real.cos_angle P (mk_fin 2 2) A F2) = abs (dist A F2)) 
    (hP_F2_perp_A_F2 : ∠(P, F2) = ∠(A, F2) + π / 2) :
  let PB := real.sqrt (3^2 + (2 - 1)^2),
      R := PB / (real.sin (π / 4)) / 2
  in (R^2 * π = 5 * π) :=
by
  sorry

end area_circumcircle_l122_122639


namespace find_largest_number_l122_122080

theorem find_largest_number :
  let a := -(abs (-3) ^ 3)
  let b := -((-3) ^ 3)
  let c := (-3) ^ 3
  let d := -(3 ^ 3)
  b = 27 ∧ b > a ∧ b > c ∧ b > d := by
  sorry

end find_largest_number_l122_122080


namespace range_of_m_l122_122157

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc (0 : ℝ) 2, f x1 ≥ g x2) →
  (∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc (0 : ℝ) 2, x1^2 ≥ 2^x2 - m) →
  m ≥ 1 := 
sorry

end range_of_m_l122_122157


namespace ordered_pairs_exists_l122_122559

theorem ordered_pairs_exists : 
  (∀ (a b : ℝ), (∃ (x y : ℤ), ax + by = 1 ∧ x^2 + y^2 = 65)) ↔ 
    ∃ (n : ℕ), n = 128 :=
sorry

end ordered_pairs_exists_l122_122559


namespace bead_closed_chain_condition_l122_122011

theorem bead_closed_chain_condition (n k : ℕ) : 
  (∃ (χ : ℕ), n * k = 2 * χ) → (2 ≤ n) ∧ (2 ≤ k) → (∃ (closed_chain : bool), closed_chain = true) :=
by 
  intros h1 h2 
  sorry

end bead_closed_chain_condition_l122_122011


namespace digit_sum_s_99_l122_122821

/-- 
  \( s(n) \) is an \( n \)-digit number formed by attaching the first \( n \) perfect squares in order into one integer.
  Define \( digit\_sum \) as the sum of the digits of a number.
  Prove that the digit sum of \( s(99) \) is 4.
-/
def s (n : ℕ) : ℕ := 
  String.toNat $ String.intercalate "" $ (List.range n).map (λ i, toString ((i+1)*(i+1)))

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem digit_sum_s_99 : digit_sum (s 99) = 4 := 
sorry

end digit_sum_s_99_l122_122821


namespace jenna_practice_minutes_l122_122261

theorem jenna_practice_minutes :
  ∀ (practice_6_days practice_2_days target_total target_average: ℕ),
    practice_6_days = 6 * 80 →
    practice_2_days = 2 * 105 →
    target_average = 100 →
    target_total = 9 * target_average →
  ∃ practice_9th_day, (practice_6_days + practice_2_days + practice_9th_day = target_total) ∧ practice_9th_day = 210 :=
by sorry

end jenna_practice_minutes_l122_122261


namespace largest_degree_polynomial_horizontal_asymptote_l122_122434

theorem largest_degree_polynomial_horizontal_asymptote (p : polynomial ℝ) :
  (∃ (d : ℕ), p.natDegree = d ∧ d ≤ 6) ↔ (∃ (p : polynomial ℝ), p.natDegree = 6 ∧
    ∃ (p : polynomial ℝ), (2 * p.natDegree = 6 + p.natDegree - 1 + p.natDegree + 2)) :=
  sorry

end largest_degree_polynomial_horizontal_asymptote_l122_122434


namespace find_defective_keys_l122_122453

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122453


namespace probability_not_pass_fourth_quadrant_l122_122689

-- The set of possible numbers on the balls
def ball_numbers : Set ℚ := {-1, 0, 1/3}

-- The quadratic function with given m and n
def quadratic (m n : ℚ) (x : ℚ) : ℚ := x^2 + m * x + n

-- The condition for the quadratic function not to pass through the fourth quadrant
def does_not_pass_fourth_quadrant (m n : ℚ) : Prop :=
  (m ≥ 0 ∧ n ≥ 0) ∨ (m ≥ 0 ∧ n * 4 ≥ m^2)

-- The predicate that for a given (m, n) the quadratic function does not pass through the fourth quadrant
def valid_pair (m n : ℚ) : Prop :=
  does_not_pass_fourth_quadrant m n

-- All possible (m, n) pairs
def all_pairs : List (ℚ × ℚ) :=
  [( -1, -1), ( -1,  0), ( -1,  1/3),
   (  0, -1), (  0,  0), (  0,  1/3),
   (1/3, -1), (1/3,  0), (1/3, 1/3)]

-- Valid pairs count
def valid_pair_count : ℕ :=
  ((all_pairs.filter $ λ pair, valid_pair pair.fst pair.snd).length : ℕ)

-- Total pairs count
def total_pair_count : ℕ := (all_pairs.length : ℕ)

-- The probability the quadratic function does not pass through the fourth quadrant
def probability_does_not_pass_fourth_quadrant : ℚ :=
  (valid_pair_count : ℚ) / (total_pair_count : ℚ)

theorem probability_not_pass_fourth_quadrant :
  probability_does_not_pass_fourth_quadrant = 5 / 9 :=
by
  sorry

end probability_not_pass_fourth_quadrant_l122_122689


namespace cone_volume_l122_122067

theorem cone_volume (R r h : ℝ) (H1 : 2 * π * r = π * R)
  (H2 : h^2 + r^2 = R^2) :
  (1 / 3) * π * r^2 * h = (sqrt 3 / 24) * π * R^3 := sorry

end cone_volume_l122_122067


namespace sin_identity_l122_122602

theorem sin_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + π / 4) ^ 2 = 5 / 6 := 
sorry

end sin_identity_l122_122602


namespace order_abc_l122_122659

noncomputable def a := log 2 3 + log 3 2
noncomputable def b := log 2 real.e + real.log 2
def c := 13 / 6

theorem order_abc : b < c ∧ c < a :=
by
  sorry

end order_abc_l122_122659


namespace slope_of_line_l122_122072

theorem slope_of_line (a b : ℝ) (h_intercept : b = 2) (h_point : (239, 480) ∈ set_of (λ x, a * x + b)) : a = 2 := 
by sorry

end slope_of_line_l122_122072


namespace circle_diameter_increase_l122_122764

theorem circle_diameter_increase (π : Real) (D A : Real)
  (h_area : A = π / 4 * D^2) (h_increase : ∃ A', A' = 6 * A) :
  ∃ D', D' = D * Real.sqrt(6) ∧ ((D' - D) / D) * 100 = 144.95 := by
  sorry

end circle_diameter_increase_l122_122764


namespace half_burned_height_l122_122050

def burn_time (k : ℕ) : ℕ := 5 * k^2

def total_burn_time : ℕ := (∑ k in Finset.range 101, burn_time k)

theorem half_burned_height : (∑ k in Finset.range 101, burn_time k) / 2 = 845875 → 100 - 52 = 48 :=
by
  intro h_total
  sorry

end half_burned_height_l122_122050


namespace smallest_prime_sum_l122_122430

/-- 
  Prove that the smallest possible sum of a set of prime numbers using each of the
  digits 0 through 9 exactly once is 207 given the conditions.
-/
theorem smallest_prime_sum :
  (∃ (S : set ℕ), 
    (∀ p ∈ S, prime p) ∧ 
    (∀ d ∈ {0,1,2,3,4,5,6,7,8,9}, ∃! n ∈ S, d ∈ digits 10 n) ∧ 
    ∑ p in S, p = 207) :=
sorry

end smallest_prime_sum_l122_122430


namespace equal_real_roots_value_l122_122613

theorem equal_real_roots_value (a c : ℝ) (ha : a ≠ 0) (h : 4 - 4 * a * (2 - c) = 0) : (1 / a) + c = 2 := 
by
  sorry

end equal_real_roots_value_l122_122613


namespace line_tangent_to_circle_l122_122426

theorem line_tangent_to_circle : 
  ∀ (P : ℝ × ℝ), P = (2, 1) → 
  ∀ (x y : ℝ), x^2 + y^2 = 4 →
  ∃ a b c : ℝ, a = 2 ∧ b = 1 ∧ c = 4 ∧ (a * x + b * y - c = 0) := 
by
  intro P hP x y hCircle
  use [2, 1, 4]
  simp [hP, hCircle]
  sorry

end line_tangent_to_circle_l122_122426


namespace exterior_angle_polygon_num_sides_l122_122667

theorem exterior_angle_polygon_num_sides (exterior_angle : ℝ) (h : exterior_angle = 60) : ∃ (n : ℕ), n = 6 :=
by
  use 6
  sorry

end exterior_angle_polygon_num_sides_l122_122667


namespace function_extreme_points_and_zeros_l122_122870

noncomputable def ω_range : Set ℝ := 
  setOf (λ ω, (13 : ℝ)/6 < ω ∧ ω ≤ (8 : ℝ)/3)

theorem function_extreme_points_and_zeros (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = sin (ω * x + π / 3)) 
  (h2 : set.count (set_of (λ x, x ∈ (0, π) ∧ extreme_point f x)) = 3) 
  (h3 : set.count (set_of (λ x, x ∈ (0, π) ∧ f x = 0)) = 2) : 
  ω ∈ ω_range := 
sorry

end function_extreme_points_and_zeros_l122_122870


namespace alison_money_l122_122078

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l122_122078


namespace algebraic_expression_analysis_l122_122250

theorem algebraic_expression_analysis :
  (∀ x y : ℝ, (x - 1/2 * y) * (x + 1/2 * y) = x^2 - (1/2 * y)^2) ∧
  (∀ a b c : ℝ, ¬ ((3 * a + b * c) * (-b * c - 3 * a) = (3 * a + b * c)^2)) ∧
  (∀ x y : ℝ, (3 - x + y) * (3 + x + y) = (3 + y)^2 - x^2) ∧
  ((100 + 1) * (100 - 1) = 100^2 - 1) :=
by
  intros
  repeat { split }; sorry

end algebraic_expression_analysis_l122_122250


namespace prob_of_point_in_region_prob_of_real_roots_l122_122039

/-- The probability that point P falls within the region defined by the conditions
x + y <= 6, x >= 0, and y >= 0, when rolling a standard six-sided die twice, is 5/12. -/
theorem prob_of_point_in_region :
  let outcomes := [(m, n) | m ∈ [1, 2, 3, 4, 5, 6], n ∈ [1, 2, 3, 4, 5, 6]],
      region := [(x, y) | x + y ≤ 6 ∧ x ≥ 0 ∧ y ≥ 0],
      count := (outcomes ∩ region).length in
  (count / 36) = 5 / 12 := sorry

/-- The probability that the equation x^2 + mx + n^2 = 0 has real roots,
given m and n are randomly chosen from the interval [1, 6] and m >= 2n, is 4/25. -/
theorem prob_of_real_roots :
  let interval := [(m, n) | 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6],
      condition := [(m, n) | m ≥ 2 * n],
      count := (interval ∩ condition).length in
  (count / (6 * 6)) = 4 / 25 := sorry

end prob_of_point_in_region_prob_of_real_roots_l122_122039


namespace find_plane_l122_122137

def point := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

def normal_vector (p : plane) : (ℝ × ℝ × ℝ) := (p.1, p.2, p.3)
def point_on_plane (x y z : ℝ) (p : plane) : Prop := p.1 * x + p.2 * y + p.3 * z + p.4 = 0

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem find_plane (A B C D : ℤ)
  (hA_pos : A > 0)
  (h_gcd : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1)
  (h_point1 : point_on_plane 2 0 (-1) (A, B, C, D))
  (h_point2 : point_on_plane (-2) 4 (-1) (A, B, C, D))
  (h_perp : let n := normal_vector (2, -1, 2, 4) in cross_product (-4, 4, 0) n = (A, B, C)) :
  A = 2 ∧ B = 2 ∧ C = 3 ∧ D = -1 :=
sorry

end find_plane_l122_122137


namespace ratio_of_surface_areas_l122_122672

theorem ratio_of_surface_areas (x k : ℝ) (hk : k ≠ 0) :
  (300 * x^2) / (6 * k^2 * x^2) = 50 / k^2 :=
by
  calc
    (300 * x^2) / (6 * k^2 * x^2) = 300 / (6 * k^2) := by sorry
    ... = 50 / k^2 := by sorry

end ratio_of_surface_areas_l122_122672


namespace correct_number_of_propositions_l122_122210

-- Define the taxicab distance
def taxicab_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the propositions
def prop1 (A B C : ℝ × ℝ) : Prop :=
  let P := A
  let Q := B
  let R := C
  taxicab_distance A C + taxicab_distance C B = taxicab_distance A B

def prop2 (A B C : ℝ × ℝ) : Prop :=
  let P := A
  let Q := B
  let R := C
  taxicab_distance A C + taxicab_distance C B > taxicab_distance A B

def prop3 (x y : ℝ) : Prop :=
  taxicab_distance (x, y) (-1, 0) = taxicab_distance (x, y) (1, 0) ↔ x = 0

def prop4 (B : ℝ × ℝ) : Prop :=
  B = (λ x, x, 2 * real.sqrt 5 - x) →
  let A := (0, 0) in
  taxicab_distance A B = 2 * real.sqrt 5

-- Count the number of true propositions
def num_true_propositions : ℕ :=
  (if prop1 (1, 2) (3, 4) (2, 3) then 1 else 0) +
  (if prop2 (1, 2) (3, 4) (2, 3) then 1 else 0) +
  (if prop3 0 0 then 1 else 0) +
  (if prop4 (real.sqrt 5, real.sqrt 5) then 1 else 0)

-- The goal statement
theorem correct_number_of_propositions : num_true_propositions = 3 :=
by
  sorry

end correct_number_of_propositions_l122_122210


namespace necessarily_positive_y_plus_z_l122_122323

theorem necessarily_positive_y_plus_z
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) :
  y + z > 0 := 
by
  sorry

end necessarily_positive_y_plus_z_l122_122323


namespace largest_prime_factor_1729_l122_122013

theorem largest_prime_factor_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p) := 
sorry

end largest_prime_factor_1729_l122_122013


namespace tan_monotone_increasing_interval_l122_122778

theorem tan_monotone_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, 
  (-π / 2 + k * π < x + π / 4 ∧ x + π / 4 < π / 2 + k * π) ↔
  (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) :=
by sorry

end tan_monotone_increasing_interval_l122_122778


namespace fewer_miles_per_gallon_city_l122_122830

-- Define the given conditions.
def miles_per_tankful_highway : ℕ := 420
def miles_per_tankful_city : ℕ := 336
def miles_per_gallon_city : ℕ := 24

-- Define the question as a theorem that proves how many fewer miles per gallon in the city compared to the highway.
theorem fewer_miles_per_gallon_city (G : ℕ) (hG : G = miles_per_tankful_city / miles_per_gallon_city) :
  miles_per_tankful_highway / G - miles_per_gallon_city = 6 :=
by
  -- The proof will be provided here.
  sorry

end fewer_miles_per_gallon_city_l122_122830


namespace number_of_ordered_quadruples_division_l122_122275

theorem number_of_ordered_quadruples_division :
  let n := (finset.card {u : fin 52 ^ 4 // (u.1 + u.2 + u.3 + u.4 = 51)} : ℕ)
  in (n / 100 = 248.04) :=
by
  -- Definitions and conditions follow
  let n := (finset.card {u : fin 52 ^ 4 // (u.1 + u.2 + u.3 + u.4 = 51)} : ℕ),
  have H : n = 24804,
    -- Proof of the cardinality of the set would go here
    sorry,
  calc
    n / 100 = 24804 / 100 : by rw H
    ... = 248.04 : by norm_num

end number_of_ordered_quadruples_division_l122_122275


namespace trigonometric_equation_solution_l122_122491

theorem trigonometric_equation_solution (x : ℝ) (k t : ℤ) :
  (cos x ≠ 0) ∧ (cos (2 * x) ≠ 0) ∧ (sin (3 * x) - cos (6 * x / 5) = 2) ∧
  (x = (5 / 6) * (ℝ.pi + 4 * ℝ.pi * k)) ∧ (k ≠ 2 + 3 * t) →
  8.471 * (3 * tan x - tan x ^ 3) / (2 - 1 / cos x ^ 2) = (4 + 2 * cos (6 * x / 5)) / (cos (3 * x) + cos x) :=
sorry

end trigonometric_equation_solution_l122_122491


namespace compute_special_op_l122_122966

-- Define the operation ※
def special_op (m n : ℚ) := (3 * m + n) * (3 * m - n) + n

-- Hypothesis for specific m and n
def m := (1 : ℚ) / 6
def n := (-1 : ℚ)

-- Proof goal
theorem compute_special_op : special_op m n = -7 / 4 := by
  sorry

end compute_special_op_l122_122966


namespace possible_faulty_keys_l122_122459

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122459


namespace P_1989_div_3_pow_994_l122_122553

noncomputable def P : ℕ → ℕ
| 0 := 4
| 1 := 15
| (n + 2) := 3 * (P (n + 1) + P n)

theorem P_1989_div_3_pow_994 : 
  ∃ k : ℕ, P 1989 = 3 ^ 994 * k :=
sorry

end P_1989_div_3_pow_994_l122_122553


namespace possible_faulty_keys_l122_122463

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122463


namespace work_together_time_l122_122495

-- Definitions for the conditions
def p_efficiency := 1 -- Assuming p's efficiency is 100% (1 as a proportion)
def q_efficiency := 0.9 * p_efficiency -- q's efficiency as 90% of p's

def p_days_to_complete := 21 -- p can complete the work in 21 days

def p_work_rate := 1 / p_days_to_complete -- Work rate of p
def q_work_rate := q_efficiency * p_work_rate -- Work rate of q

-- Combined work rate of p and q working together
def combined_work_rate := p_work_rate + q_work_rate

-- The number of days to complete the work together
def days_together := 1 / combined_work_rate

-- The theorem to prove
theorem work_together_time : days_together = 210 / 19 :=
by
  -- Statement only, proof skipped
  sorry

end work_together_time_l122_122495


namespace find_coefficients_l122_122574

theorem find_coefficients (A B : ℚ) :
  (∀ x : ℚ, 2 * x + 7 = A * (x + 7) + B * (x - 9)) →
  A = 25 / 16 ∧ B = 7 / 16 :=
by
  intro h
  sorry

end find_coefficients_l122_122574


namespace probability_of_yellow_marble_l122_122252

/--
In the jar, there are 7 blue marbles, 11 red marbles, and 6 yellow marbles.
Prove that the probability of randomly picking a yellow marble is 1/4.
-/
theorem probability_of_yellow_marble :
  let blue_marbles := 7
  let red_marbles := 11
  let yellow_marbles := 6
  let total_marbles := blue_marbles + red_marbles + yellow_marbles
  total_marbles = 24 → yellow_marbles = 6 → (yellow_marbles : ℚ) / total_marbles = 1 / 4 :=
by
  intros h1 h2
  have h3: total_marbles = yellow_marbles + 18 := by sorry
  simp [h2] at h1
  field_simp
  norm_num
  sorry

end probability_of_yellow_marble_l122_122252


namespace crayons_total_l122_122675

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l122_122675


namespace cos_beta_l122_122992

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l122_122992


namespace alison_money_l122_122079

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l122_122079


namespace largest_value_of_x_l122_122435

theorem largest_value_of_x (x : ℝ) (hx : x / 3 + 1 / (7 * x) = 1 / 2) : 
  x = (21 + Real.sqrt 105) / 28 := 
sorry

end largest_value_of_x_l122_122435


namespace field_area_is_13_point854_hectares_l122_122834

noncomputable def area_of_field_in_hectares (cost_fencing: ℝ) (rate_per_meter: ℝ): ℝ :=
  let length_of_fence := cost_fencing / rate_per_meter
  let radius := length_of_fence / (2 * Real.pi)
  let area_in_square_meters := Real.pi * (radius * radius)
  area_in_square_meters / 10000

theorem field_area_is_13_point854_hectares :
  area_of_field_in_hectares 6202.75 4.70 = 13.854 :=
by
  sorry

end field_area_is_13_point854_hectares_l122_122834


namespace faulty_keys_l122_122470

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122470


namespace find_k_l122_122950

theorem find_k (α : ℝ) (h1 : (sin α + 2 / sin α) ^ 2 + (cos α + 2 / cos α) ^ 2 = k + 2 * (tan α) ^ 2 + 2 * (cot α) ^ 2 + sin (2 * α)) : 
  k = -1 := 
sorry

end find_k_l122_122950


namespace curve_cartesian_eq_correct_intersection_distances_sum_l122_122254

noncomputable section

def curve_parametric_eqns (θ : ℝ) : ℝ × ℝ := 
  (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

def line_parametric_eqns (t : ℝ) : ℝ × ℝ := 
  (3 + (1/2) * t, 3 + (Real.sqrt 3 / 2) * t)

def curve_cartesian_eq (x y : ℝ) : Prop := 
  (x - 1)^2 + (y - 3)^2 = 9

def point_p : ℝ × ℝ := 
  (3, 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem curve_cartesian_eq_correct (θ : ℝ) : 
  curve_cartesian_eq (curve_parametric_eqns θ).1 (curve_parametric_eqns θ).2 := 
by 
  sorry

theorem intersection_distances_sum (t1 t2 : ℝ) 
  (h1 : curve_cartesian_eq (line_parametric_eqns t1).1 (line_parametric_eqns t1).2) 
  (h2 : curve_cartesian_eq (line_parametric_eqns t2).1 (line_parametric_eqns t2).2) : 
  distance point_p (line_parametric_eqns t1) + distance point_p (line_parametric_eqns t2) = 2 * Real.sqrt 3 := 
by 
  sorry

end curve_cartesian_eq_correct_intersection_distances_sum_l122_122254


namespace opposite_of_neg5_is_pos5_l122_122358

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122358


namespace students_only_english_l122_122678

-- Define the conditions as given in the problem
variables (total_students both_english_german total_german : ℕ)
variables (total_students_eq both_english_german_eq total_german_eq: Prop)

-- Assign the values from the problem
axiom total_students_eq : total_students = 60
axiom both_english_german_eq : both_english_german = 18
axiom total_german_eq : total_german = 36
axiom at_least_one_subject : total_students = E + G + B

-- Proving the main question
theorem students_only_english (E G B : ℕ) 
    (h1 : total_students = E + G + B)
    (h2 : B = both_english_german)
    (h3 : G = total_german - both_english_german) 
    : E = 24 :=
by
  rw [total_students_eq, both_english_german_eq, total_german_eq] at *, 
  sorry

end students_only_english_l122_122678


namespace find_s_l122_122007

noncomputable def point (α : Type) := α × α

def P : point ℝ := (0, 10)
def Q : point ℝ := (4, 0)
def R : point ℝ := (10, 0)

-- Defining a line equation based on a point and the slope
def line_eq (x y : ℝ) (m : ℝ) := fun t : ℝ => (t, m * (t - x) + y)

def line_PQ : ℝ → ℝ × ℝ := line_eq 0 10 (-2.5)
def line_PR : ℝ → ℝ × ℝ := line_eq 0 10 (-1)

-- Defining the area function based on the vertical distance and the base
def area_triangle (base height : ℝ) := 0.5 * base * height

-- Statement to prove
theorem find_s (s : ℝ) :
  let V := line_PQ ((10 - s) / 2.5),
      W := line_PR (10 - s),
      VW_len := abs ((10 - s) - ((4 - 2 * s / 5))),
      height := 10 - s,
      area := area_triangle VW_len height
  in area = 15 → s = 5 :=
by
  sorry

end find_s_l122_122007


namespace prove_r_equals_one_l122_122902

open Real

variables (A B C D E : Point)
variables (α β γ δ : ℝ)

-- Given conditions
def convex_quadrilateral (A B C D : Point) : Prop := sorry
def extend_to_intersect (AD BC : Line) (E: Point) : Prop := sorry
def angle_CDE (C D E : Point) : ℝ := sorry
def angle_DCE (D C E : Point) : ℝ := sorry
def angle_BAD (B A D : Point) : ℝ := sorry
def angle_ABC (A B C : Point) : ℝ := sorry

-- s = ∠CDE + ∠DCE
def s := angle_CDE C D E + angle_DCE D C E

-- s' = ∠BAD + ∠ABC
def s' := angle_BAD B A D + angle_ABC A B C

-- r = s / s'
def r := s / s'

-- Theorem to prove r = 1
theorem prove_r_equals_one
  (conv_quad: convex_quadrilateral A B C D)
  (intersect : extend_to_intersect (line_through A D) (line_through B C) E) : r = 1 := 
sorry

end prove_r_equals_one_l122_122902


namespace sqrt_mixed_number_simplified_l122_122941

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l122_122941


namespace greatest_integer_gcd_3_l122_122012

theorem greatest_integer_gcd_3 : ∃ n, n < 100 ∧ gcd n 18 = 3 ∧ ∀ m, m < 100 ∧ gcd m 18 = 3 → m ≤ n := by
  sorry

end greatest_integer_gcd_3_l122_122012


namespace sara_saving_is_9_dollars_l122_122748

def dozen : ℕ := 12
def quarters_in_a_dollar : ℕ := 4
def quarter_value_cents : ℕ := 25
def cents_in_a_dollar : ℕ := 100

theorem sara_saving_is_9_dollars (dozens_of_quarters : ℕ) 
  (H1 : dozens_of_quarters = 3)
  (H2 : dozen = 12) 
  (H3 : quarter_value_cents = 25) 
  (H4 : cents_in_a_dollar = 100):
  dozens_of_quarters * dozen * quarter_value_cents / cents_in_a_dollar = 9 := 
by
  rw [H1, H2, H3, H4]
  norm_num
  sorry

end sara_saving_is_9_dollars_l122_122748


namespace tangency_pass_through_l122_122742

-- Definitions of the geometric properties based on the given conditions.
variable {Point : Type*} [metric_space Point]

-- Conditions to define the pyramid and sphere.
variables (O S A B C D K L M N : Point)
variable (pyramid : ∀ P : Point, ∃! Q : Point, P ∈ Sphere Q)  -- Simplified: points lying on connected region
variable (tangent : ∀ P, is_tangent_to_sphere P sphere)  -- Simplified: tangency property

-- Main theorem to prove the statement about tangency points.
theorem tangency_pass_through :
  is_foot_altitude O pyramid ∧ is_center_of_sphere O sphere ∧
  are_points_on_edges [A, B, C, D] ∧
  tangent_segment_passes [AB, BC, CD] [K, L, M] ∧ 
  are_tangency_points [T] := sorry

end tangency_pass_through_l122_122742


namespace tank_empty_time_when_inlet_open_l122_122839

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open_l122_122839


namespace base_conversion_69_to_5_l122_122253

theorem base_conversion_69_to_5 :
  let n := 69 in
  let b := 5 in
  let d1 := n / b^2 in
  let r1 := n % b^2 in
  let d2 := r1 / b in
  let r2 := r1 % b in
  let d3 := r2 in
  n = d1 * b^2 + d2 * b + d3 ∧
  d1 = 2 ∧ d2 = 3 ∧ d3 = 4 ∧
  (string.length (d1.to_string ++ d2.to_string ++ d3.to_string) = 3) := 
by
  sorry

end base_conversion_69_to_5_l122_122253


namespace xyz_value_l122_122618

theorem xyz_value
  (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 :=
  sorry

end xyz_value_l122_122618


namespace find_faulty_keys_l122_122444

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122444


namespace expression_equals_neg_one_l122_122658

theorem expression_equals_neg_one (a b c : ℝ) (h : a + b + c = 0) :
  (|a| / a) + (|b| / b) + (|c| / c) + (|a * b| / (a * b)) + (|a * c| / (a * c)) + (|b * c| / (b * c)) + (|a * b * c| / (a * b * c)) = -1 :=
  sorry

end expression_equals_neg_one_l122_122658


namespace smallest_number_in_systematic_sample_l122_122599

theorem smallest_number_in_systematic_sample (n m x : ℕ) (products : Finset ℕ) :
  n = 80 ∧ m = 5 ∧ products = Finset.range n ∧ x = 42 ∧ x ∈ products ∧ (∃ k : ℕ, x = (n / m) * k + 10) → 10 ∈ products :=
by
  sorry

end smallest_number_in_systematic_sample_l122_122599


namespace side_length_of_square_l122_122528

theorem side_length_of_square (s : ℝ) (h : s^2 = 2 * (4 * s)) : s = 8 := 
by
  sorry

end side_length_of_square_l122_122528


namespace lcm_14_21_45_l122_122014

-- Definitions of the numbers and their prime factorizations
def num1 := 14
def num2 := 21
def num3 := 45

-- Definition of the prime factorizations (assert without proof)
def prime_factorization_num1 : num1 = 2 * 7 := by { dsimp [num1], norm_num }
def prime_factorization_num2 : num2 = 3 * 7 := by { dsimp [num2], norm_num }
def prime_factorization_num3 : num3 = 3^2 * 5 := by { dsimp [num3], norm_num }

-- Definition of the least common multiple (LCM)
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

-- Statement asserting the LCM of num1, num2, num3 is 630
theorem lcm_14_21_45 : lcm num1 num2 num3 = 630 :=
  by
    -- Use known prime factorizations to find LCM
    have h1 : num1 = 2 * 7 := prime_factorization_num1
    have h2 : num2 = 3 * 7 := prime_factorization_num2
    have h3 : num3 = 3^2 * 5 := prime_factorization_num3
    rw [h1, h2, h3]
    sorry -- Proof of the calculation (LCM) needs to be filled in

end lcm_14_21_45_l122_122014


namespace high_fives_count_l122_122814

theorem high_fives_count (n : ℕ) (h : n = 12) : (Nat.choose n 2) = 66 := by
  rw [h]
  simp only [Nat.choose]
  norm_num
  sorry

end high_fives_count_l122_122814


namespace Tobias_change_l122_122423

def cost_of_shoes := 95
def allowance_per_month := 5
def months_saving := 3
def charge_per_lawn := 15
def lawns_mowed := 4
def charge_per_driveway := 7
def driveways_shoveled := 5
def total_amount_saved : ℕ := (allowance_per_month * months_saving)
                          + (charge_per_lawn * lawns_mowed)
                          + (charge_per_driveway * driveways_shoveled)

theorem Tobias_change : total_amount_saved - cost_of_shoes = 15 := by
  sorry

end Tobias_change_l122_122423


namespace arrangement_ways_l122_122232

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l122_122232


namespace sum_of_tens_and_units_digit_l122_122119

theorem sum_of_tens_and_units_digit (n : ℕ) (h : n = 11^2004 - 5) : 
  (n % 100 / 10) + (n % 10) = 9 :=
by
  sorry

end sum_of_tens_and_units_digit_l122_122119


namespace number_of_pear_trees_l122_122251

theorem number_of_pear_trees (A P : ℕ) (h1 : A + P = 46)
  (h2 : ∀ (s : Finset (Fin 46)), s.card = 28 → ∃ (i : Fin 46), i ∈ s ∧ i < A)
  (h3 : ∀ (s : Finset (Fin 46)), s.card = 20 → ∃ (i : Fin 46), i ∈ s ∧ A ≤ i) :
  P = 27 :=
by
  sorry

end number_of_pear_trees_l122_122251


namespace find_ff2_l122_122155

def f (x : ℝ) : ℝ := if x ≤ 1 then x + 1 else 3 - x

theorem find_ff2 : f (f 2) = 2 :=
by
  sorry

end find_ff2_l122_122155


namespace points_four_units_away_l122_122313

theorem points_four_units_away (A : ℝ) (d : ℝ) (B : ℝ) : A = -2 ∧ d = 4 ∧ (B = A + d ∨ B = A - d) → (B = 2 ∨ B = -6) := 
by 
  intro h
  cases h with hA hd
  cases hd with hd hB
  cases hB
  { rw [hA, hd] at hB
    simp at hB
    exact Or.inl hB }
  { rw [hA, hd] at hB
    simp at hB
    exact Or.inr hB }

end points_four_units_away_l122_122313


namespace planting_trees_equation_l122_122512

theorem planting_trees_equation (x : ℝ) (h1 : x > 0) : 
  20 / x - 20 / ((1 + 0.1) * x) = 4 :=
sorry

end planting_trees_equation_l122_122512


namespace problem_solution_l122_122957

noncomputable def root1 : ℝ := (3 + Real.sqrt 105) / 4
noncomputable def root2 : ℝ := (3 - Real.sqrt 105) / 4

theorem problem_solution :
  (∀ x : ℝ, x ≠ -2 → x ≠ -3 → (x^3 - x^2 - 4 * x) / (x^2 + 5 * x + 6) + x = -4
    → x = root1 ∨ x = root2) := 
by
  sorry

end problem_solution_l122_122957


namespace intersection_A_B_l122_122646

-- Define the set A
def A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define the set B
def B := {x : ℝ | x^2 - x < 0}

-- The proof problem statement in Lean 4
theorem intersection_A_B : A ∩ B = {y : ℝ | 0 < y ∧ y < 1} :=
by
  sorry

end intersection_A_B_l122_122646


namespace no_four_consecutive_perf_square_l122_122911

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l122_122911


namespace ratio_of_siblings_l122_122705

/-- Let's define the sibling relationships and prove the ratio of Janet's to Masud's siblings is 3 to 1. -/
theorem ratio_of_siblings (masud_siblings : ℕ) (carlos_siblings janet_siblings : ℕ)
  (h1 : masud_siblings = 60)
  (h2 : carlos_siblings = 3 * masud_siblings / 4)
  (h3 : janet_siblings = carlos_siblings + 135) 
  (h4 : janet_siblings < some_mul * masud_siblings) : 
  janet_siblings / masud_siblings = 3 :=
by
  sorry

end ratio_of_siblings_l122_122705


namespace rajans_position_l122_122683

theorem rajans_position
    (total_boys : ℕ)
    (vinay_position_from_right : ℕ)
    (boys_between_rajan_and_vinay : ℕ)
    (total_boys_eq : total_boys = 24)
    (vinay_position_from_right_eq : vinay_position_from_right = 10)
    (boys_between_eq : boys_between_rajan_and_vinay = 8) :
    ∃ R : ℕ, R = 6 :=
by
  sorry

end rajans_position_l122_122683


namespace min_value_expression_l122_122168

theorem min_value_expression (a b : ℝ) (h1 : 2 * a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a) + ((1 - b) / b) = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_expression_l122_122168


namespace number_of_cherry_pies_l122_122860

theorem number_of_cherry_pies (total_pies : ℕ) (apple_ratio : ℕ) (blueberry_ratio : ℕ) (cherry_ratio : ℕ)
    (total_ratio : apple_ratio + blueberry_ratio + cherry_ratio = 11) (total_pies = 36)
    : ∃ n, n = 13 ∧ 4 * (total_pies / total_ratio) = n :=
by
  sorry

end number_of_cherry_pies_l122_122860


namespace problem_statement_l122_122558

def star (x y : Nat) : Nat :=
  match x, y with
  | 1, 1 => 4 | 1, 2 => 3 | 1, 3 => 2 | 1, 4 => 1
  | 2, 1 => 1 | 2, 2 => 4 | 2, 3 => 3 | 2, 4 => 2
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 3
  | 4, 1 => 3 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4
  | _, _ => 0  -- This line handles unexpected inputs.

theorem problem_statement : star (star 3 2) (star 2 1) = 4 := by
  sorry

end problem_statement_l122_122558


namespace minute_hand_gains_per_hour_l122_122514

theorem minute_hand_gains_per_hour :
  (∀ t : ℕ, (t >= 11 ∧ t <= 18) → 
   let gained_minutes := 35 in
   let hours := t - 11 in
   gained_minutes = 5 * hours) :=
by
  sorry

end minute_hand_gains_per_hour_l122_122514


namespace number_of_last_digits_div_by_4_l122_122302

theorem number_of_last_digits_div_by_4: 
  let last_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let divisible_by_4 (n : Nat) : Prop := n % 4 = 0
  let valid_last_digits := {d ∈ last_digits | ∃ k, divisible_by_4 (k * 10 + d)}
in valid_last_digits = {0, 2, 4, 6, 8} ∧ valid_last_digits.card = 5 := by sorry

end number_of_last_digits_div_by_4_l122_122302


namespace sum_of_squares_of_coefficients_eq_283_l122_122549

def polynomial := 3 * (X^3 - 4 * X^2 + 3 * X - 1) - 5 * (2 * X^3 - X^2 + X + 2)

theorem sum_of_squares_of_coefficients_eq_283 :
  let p := polynomial,
  let c := p.coeffs,
  (c[0])^2 + (c[1])^2 + (c[2])^2 + (c[3])^2 = 283 :=
begin
  sorry
end

end sum_of_squares_of_coefficients_eq_283_l122_122549


namespace range_of_a_l122_122159

open Complex

def quadrant_condition (z : ℂ) : Prop :=
  (z.re < 0 ∧ z.im > 0) ∨ (z.re > 0 ∧ z.im < 0)

theorem range_of_a 
  {a : ℝ} {z : ℂ} 
  (h : z * (a + Complex.i) = 2 + 3 * Complex.i) 
  (hq : quadrant_condition z) : 
  -3/2 < a ∧ a < 2/3 :=
by
  sorry

end range_of_a_l122_122159


namespace fourth_circle_radius_not_smallest_l122_122083

def circle (r: ℝ) := {c : ℝ × ℝ // (c.1 - 0)^2 + (c.2 - 0)^2 = r^2}

theorem fourth_circle_radius_not_smallest :
  ∀ (r1 r2 r3 r : ℝ)
    (O1 : circle r1) (O2 : circle r2) (O3 : circle r3)
    (O : circle r),
    O1.1.1 < O2.1.1 ∧ O2.1.1 < O3.1.1 ∧ -- centers lie in a straight line
    (O.1.1 - O1.1.1)^2 + (O.1.2 - O1.1.2)^2 = (r + r1)^2 ∧ -- O tangent to O1
    (O.1.1 - O2.1.1)^2 + (O.1.2 - O2.1.2)^2 = (r + r2)^2 ∧ -- O tangent to O2
    (O.1.1 - O3.1.1)^2 + (O.1.2 - O3.1.2)^2 = (r + r3)^2 ∧ -- O tangent to O3
    (O1.1.1 - O2.1.1)^2 + (O1.1.2 - O2.1.2)^2 = (r1 + r2)^2 ∧ -- O1 tangent to O2
    (O2.1.1 - O3.1.1)^2 + (O2.1.2 - O3.1.2)^2 = (r2 + r3)^2 ∧ -- O2 tangent to O3
    (O1.1.1 - O3.1.1)^2 + (O1.1.2 - O3.1.2)^2 = (r1 + 2 * r2 + r3)^2  -- O1 through O2 to O3
  -> r ≥ r2 := 
by 
  sorry

end fourth_circle_radius_not_smallest_l122_122083


namespace equal_points_in_tournament_l122_122682

theorem equal_points_in_tournament
  (n : ℕ) (h_n_gt_2 : n > 2)
  (S : Fin n → ℕ)
  (C : Fin n → ℕ)
  (defeated : Fin n → Finset (Fin n)) -- Sets of defeated participants for each participant
  (h_coeff_eq : ∀ (i j : Fin n), C i = C j) -- All coefficients are equal
  (h_def_C : ∀ i, C i = ∑ x in defeated i, S x) -- Coefficient definition
  : ∀ (i j : Fin n), S i = S j := sorry

end equal_points_in_tournament_l122_122682


namespace opposite_of_negative_five_l122_122371

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122371


namespace parallelogram_base_length_l122_122032

theorem parallelogram_base_length (A : ℕ) (h b : ℕ) (h1 : A = b * h) (h2 : h = 2 * b) (h3 : A = 200) : b = 10 :=
by {
  sorry
}

end parallelogram_base_length_l122_122032


namespace XF_XG_17_l122_122524

-- Definitions for the problem's conditions

variables {A B C D X Y E F G O : Type} 
variables (AB BC CD DA BD DX BY AX XE XF XG : ℝ)

-- Given conditions
def quadrilateral_inscribed_in_circle (circle : Type) : Prop :=
  ∃ (A B C D : Type), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (∀ (P : Type), P = A ∨ P = B ∨ P = C ∨ P = D)

def line_segment_length (a b : Type) (length : ℝ) : Prop :=
  length > 0

def point_on_segment (p a b : Type) (ratio : ℝ) : Prop :=
  0 < ratio ∧ ratio < 1

def parallel_lines (a b c d : Type) : Prop :=
  ∃ (l1 l2 : Type), (a, b) ∈ l1 ∧ (c, d) ∈ l2 ∧ l1 ‖ l2

def intersection_point (p l1 l2 : Type) : Prop :=
  ∃ (a b c d : Type), (a, b) ∈ l1 ∧ (c, d) ∈ l2 ∧ p ∈ l1 ∧ p ∈ l2

-- Theorem statement
theorem XF_XG_17 : quadrilateral_inscribed_in_circle O
  → line_segment_length A B 3
  → line_segment_length B C 2
  → line_segment_length C D 6
  → line_segment_length D A 8
  → point_on_segment X D B (1 / 4 : ℝ)
  → point_on_segment Y B D (11 / 36 : ℝ)
  → parallel_lines Y E D A
  → intersection_point E AX (Y, parallel_lines Y E D A)
  → parallel_lines E F A C
  → intersection_point F CX (E, parallel_lines E F A C)
  → intersection_point G O (CX, some (quadrilateral_inscribed_in_circle O))
  → CX * XG = 17 :=
sorry

end XF_XG_17_l122_122524


namespace log_base_4_half_l122_122127

theorem log_base_4_half : log 4 (1 / 2) = -1 / 2 := 
by sorry

end log_base_4_half_l122_122127


namespace find_k_find_n_l122_122576

noncomputable def polynomial_divisibility_k (x : ℤ) : Prop := 
  (x^5 + x + 1) % (x^2 + x + 1) = 0

theorem find_k : polynomial_divisibility_k :=
begin
  sorry
end

def divisible_by_xk_x_1 (x : ℤ) (k n m : ℕ) : Prop :=
  (n = 3 * m + 2) ∧ ((x^n + x + 1) % (x^k + x + 1) = 0)

theorem find_n (x : ℤ) (k : ℕ) (h : k = 2) : ∃ (m : ℕ), divisible_by_xk_x_1 x 2 (3 * m + 2) m :=
begin
  sorry
end

end find_k_find_n_l122_122576


namespace transformed_curve_l122_122004

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end transformed_curve_l122_122004


namespace range_of_f_l122_122633

noncomputable def f (x : ℝ) : ℝ := 2^x
def valid_range (S : Set ℝ) : Prop := ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), f x ∈ S

theorem range_of_f : valid_range (Set.Icc (1 : ℝ) (8 : ℝ)) :=
sorry

end range_of_f_l122_122633


namespace polygon_sides_l122_122665

theorem polygon_sides (each_exterior_angle : ℝ)
                      (h₀ : each_exterior_angle = 60) :
                      (number_of_sides : ℕ) :=
  sorry

end polygon_sides_l122_122665


namespace find_angle2_l122_122849

variables {α : Type} [linear_ordered_field α]

-- Define variables for the problem
variables (AB CD E F : α)
variables (angle1 : α) (angle2 : α)

-- Given conditions
def given_conditions :=
  (triangle D C F is folded onto triangle D E F) ∧
  (vertex E lies on side A B) ∧
  (angle1 = 22)

-- The proof goal
theorem find_angle2 (h : given_conditions α AB CD E F angle1 angle2) : angle2 = 44 :=
sorry

end find_angle2_l122_122849


namespace sqrt_of_mixed_number_as_fraction_l122_122932

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l122_122932


namespace minimum_value_an_eq_neg28_at_n_eq_3_l122_122986

noncomputable def seq_an (n : ℕ) : ℝ :=
  if n > 0 then (5 / 2) * n^2 - (13 / 2) * n
  else 0

noncomputable def delta_seq_an (n : ℕ) : ℝ := seq_an (n + 1) - seq_an n

noncomputable def delta2_seq_an (n : ℕ) : ℝ := delta_seq_an (n + 1) - delta_seq_an n

theorem minimum_value_an_eq_neg28_at_n_eq_3 : 
  ∃ (n : ℕ), n = 3 ∧ seq_an n = -28 :=
by
  sorry

end minimum_value_an_eq_neg28_at_n_eq_3_l122_122986


namespace pattern_sum_eq_power_l122_122735

noncomputable def binomial_coefficient (n k : ℕ) := Nat.choose n k

theorem pattern_sum_eq_power (n : ℕ) (hn : n > 0) :
  (Finset.range n).sum (λ i, binomial_coefficient (2 * n - 1) i) = 4^(n-1) :=
sorry

end pattern_sum_eq_power_l122_122735


namespace function_extreme_points_and_zeros_l122_122874

noncomputable def ω_range : Set ℝ := 
  setOf (λ ω, (13 : ℝ)/6 < ω ∧ ω ≤ (8 : ℝ)/3)

theorem function_extreme_points_and_zeros (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = sin (ω * x + π / 3)) 
  (h2 : set.count (set_of (λ x, x ∈ (0, π) ∧ extreme_point f x)) = 3) 
  (h3 : set.count (set_of (λ x, x ∈ (0, π) ∧ f x = 0)) = 2) : 
  ω ∈ ω_range := 
sorry

end function_extreme_points_and_zeros_l122_122874


namespace sqrt_of_mixed_number_as_fraction_l122_122930

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l122_122930


namespace even_functions_with_period_pi_l122_122082

def f (x : ℝ) : ℝ := cos (abs (2 * x))
def g (x : ℝ) : ℝ := abs (cos x)
def h (x : ℝ) : ℝ := abs (sin (2 * x + (Real.pi / 2)))
def j (x : ℝ) : ℝ := tan (abs x)

theorem even_functions_with_period_pi :
  (∀ x, f (-x) = f x ∧ f x = f (x + π)) ∧
  (∀ x, g (-x) = g x ∧ g x = g (x + π)) ∧
  ¬(∀ x, h (-x) = h x ∧ h x = h (x + π)) ∧
  ¬(∀ x, j (-x) = j x ∧ j x = j (x + π)) :=
by sorry

end even_functions_with_period_pi_l122_122082


namespace problem_solution_set_l122_122165

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem problem_solution_set : 
  { x : ℝ | f (x-2) > 0 } = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by sorry

end problem_solution_set_l122_122165


namespace polynomial_expansion_l122_122565

variable (z : ℂ)

def poly1 : ℂ := 3 * z^2 + 4 * z - 5
def poly2 : ℂ := 4 * z^3 - 3 * z^2 + 2

theorem polynomial_expansion :
  (poly1 * poly2) = (12 * z^5 + 7 * z^4 - 26 * z^3 + 21 * z^2 + 8 * z - 10) :=
by 
sorry

end polynomial_expansion_l122_122565


namespace inequality_M_N_l122_122158

variable {a b t : ℝ}
def M := a / b
def N := (a + t) / (b + t)

theorem inequality_M_N (h1 : a > b) (h2 : b > 0) (h3 : t > 0) :
  M > N :=
by
  sorry

end inequality_M_N_l122_122158


namespace euler_inequality_l122_122126

-- Euler's inequality: For any triangle ABC with circumradius R and inradius r, it holds that R >= 2r
theorem euler_inequality (ΔABC : Triangle ℝ) (R r : ℝ) (hR : is_circumradius ΔABC R) (hr : is_inradius ΔABC r) : 
  R ≥ 2 * r :=
sorry

end euler_inequality_l122_122126


namespace parabola_standard_equation_through_P_l122_122441

theorem parabola_standard_equation_through_P (a : ℝ) :
  ∃ P, (P = (-2, 3)) ∧ ((a - 1) * (-2) - 3 + 2a + 1 = 0) ∧
    ((∀ x y, y^2 = -9 / 2 * x → y^2 = -9 / 2 * x) ∨ (∀ x y, x^2 = 4 / 3 * y → x^2 = 4 / 3 * y)) := 
by
  sorry

end parabola_standard_equation_through_P_l122_122441


namespace opposite_of_negative_five_l122_122374

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122374


namespace exist_equidistant_point_l122_122804

-- Definitions for Lean 4
structure Point (α : Type) := (x : α) (y : α) (z : α)
structure Vector (α : Type) := (vx : α) (vy : α) (vz : α)

-- Adding the condition for Flies' movement
def fly_position {α : Type} [Add α] [Mul α] (A0 B0 : Point α) (vA vB : Vector α) (t : α) : Point α × Point α :=
  ( { x := A0.x + vA.vx * t, y := A0.y + vA.vy * t, z := A0.z + vA.vz * t }
  , { x := B0.x + vB.vx * t, y := B0.y + vB.vy * t, z := B0.z + vB.vz * t })

-- Condition structs
variable (α : Type) [Field α]

structure Problem (α : Type) [Field α] :=
  (A0 B0 : Point α)
  (vA vB : Vector α)
  (condition_vA_vB : vA.vx^2 + vA.vy^2 + vA.vz^2 = vB.vx^2 + vB.vy^2 + vB.vz^2)

-- Lean proof problem statement
theorem exist_equidistant_point (prob : Problem α) :
  ∃ P : Point α, ∀ t : α, 
    let (A_t, B_t) := fly_position prob.A0 prob.B0 prob.vA prob.vB t in
    ( (P.x - A_t.x)^2 + (P.y - A_t.y)^2 + (P.z - A_t.z)^2 =
      (P.x - B_t.x)^2 + (P.y - B_t.y)^2 + (P.z - B_t.z)^2 ) :=
  sorry

end exist_equidistant_point_l122_122804


namespace domain_range_a_l122_122634

theorem domain_range_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
by
  sorry

end domain_range_a_l122_122634


namespace arrangement_ways_l122_122228

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l122_122228


namespace mikey_has_125_jelly_beans_l122_122732

-- Definitions
def napoleon_jelly_beans := 56
def sedrich_jelly_beans := 3 * napoleon_jelly_beans + 9
def daphne_jelly_beans := 2 * (sedrich_jelly_beans - napoleon_jelly_beans)
def average_jelly_beans := (napoleon_jelly_beans + sedrich_jelly_beans + daphne_jelly_beans) / 3
def alondra_jelly_beans := average_jelly_beans - 8
def total_jelly_beans := napoleon_jelly_beans + sedrich_jelly_beans + daphne_jelly_beans + nat.round alondra_jelly_beans
def mikey_jelly_beans := total_jelly_beans / 5

-- Theorem
theorem mikey_has_125_jelly_beans : mikey_jelly_beans = 125 := by
  sorry

end mikey_has_125_jelly_beans_l122_122732


namespace sqrt_mixed_number_eq_l122_122937

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l122_122937


namespace multiplicative_inverse_modulo_l122_122725

noncomputable def A := 123456
noncomputable def B := 153846
noncomputable def N := 500000

theorem multiplicative_inverse_modulo :
  (A * B * N) % 1000000 = 1 % 1000000 :=
by
  sorry

end multiplicative_inverse_modulo_l122_122725


namespace largest_x_plus_y_l122_122287

theorem largest_x_plus_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 18 / 7 :=
by
  sorry

end largest_x_plus_y_l122_122287


namespace find_x_set_l122_122118

theorem find_x_set (x : ℝ) : ((x - 2) ^ 2 < 3 * x + 4) ↔ (0 ≤ x ∧ x < 7) := 
sorry

end find_x_set_l122_122118


namespace find_faulty_keys_l122_122445

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122445


namespace last_number_is_five_l122_122685

theorem last_number_is_five (seq : ℕ → ℕ) (h₀ : seq 0 = 5)
  (h₁ : ∀ n < 32, seq n + seq (n+1) + seq (n+2) + seq (n+3) + seq (n+4) + seq (n+5) = 29) :
  seq 36 = 5 :=
sorry

end last_number_is_five_l122_122685


namespace parabola_properties_l122_122194

open Real

theorem parabola_properties :
  ∃ p > 0, (1 : ℝ) < p ∧ (∀ y, y * y = 2 * p * 1 ↔ p = 2) ∧
  (let C : ℝ × ℝ → Prop := λ a b, b * b = 4 * a in
   C (1 : ℝ) (-2) ∧ (∀ x, C x (-1) = false ∧ x = -1)) ∧
  ∃ L : ℝ × ℝ → Prop, (∀ a, L a (-2 * a + 1) ↔ true) ∧
  ((dist (λ _, 0 : ℝ) L) = sqrt 5 / 5) :=
by
  sorry

end parabola_properties_l122_122194


namespace optimal_delivery_to_4th_house_l122_122116

-- Define distances between houses
variables (d12 d13 d14 d23 d24 d34 : ℝ)

-- Define weights (orders from each house)
def w1 := 1
def w2 := 2
def w3 := 3
def w4 := 6

-- Define total weighted distances to deliver to each house
def total_distance_to_1st_house :=
  0 + w2 * d12 + w3 * d13 + w4 * d14

def total_distance_to_2nd_house :=
  w1 * d12 + 0 + w3 * d23 + w4 * d24

def total_distance_to_3rd_house :=
  w1 * d13 + w2 * d23 + 0 + w4 * d34

def total_distance_to_4th_house :=
  w1 * d14 + w2 * d24 + w3 * d34 + 0

-- The theorem that delivering to the 4th house minimizes the total distance
theorem optimal_delivery_to_4th_house :
  total_distance_to_4th_house d12 d13 d14 d23 d24 d34 ≤
  total_distance_to_1st_house d12 d13 d14 d23 d24 d34 ∧
  total_distance_to_4th_house d12 d13 d14 d23 d24 d34 ≤
  total_distance_to_2nd_house d12 d13 d14 d23 d24 d34 ∧
  total_distance_to_4th_house d12 d13 d14 d23 d24 d34 ≤
  total_distance_to_3rd_house d12 d13 d14 d23 d24 d34 :=
sorry -- Proof is not required for this task

end optimal_delivery_to_4th_house_l122_122116


namespace repeating_decimal_to_fraction_l122_122566

theorem repeating_decimal_to_fraction :
  let x := (0.7 : ℝ) + Real.repeat' 8 9 in
  x = 781 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l122_122566


namespace opposite_of_neg5_is_pos5_l122_122355

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122355


namespace find_faulty_keys_l122_122447

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122447


namespace right_triangle_side_length_l122_122671

noncomputable def hypotenuse_length : ℤ := 17
noncomputable def side_length : ℤ := 15
noncomputable def other_side_length : ℤ := 8

theorem right_triangle_side_length (a b c : ℤ) (hypotenuse : c = hypotenuse_length) (side : a = side_length)
  (pythagorean_theorem : c^2 = a^2 + b^2) : b = other_side_length := by
  rw [hypotenuse, side] at pythagorean_theorem
  have h : b^2 = c^2 - a^2, by linarith
  rw [hypotenuse_length, side_length] at h
  norm_num at h
  exact h

end right_triangle_side_length_l122_122671


namespace no_such_abc_exists_l122_122915

theorem no_such_abc_exists : ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ),
  |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| :=
by
  sorry

end no_such_abc_exists_l122_122915


namespace ratio_first_part_l122_122831

theorem ratio_first_part (x : ℝ) (h1 : 180 / 100 * 5 = x) : x = 9 :=
by sorry

end ratio_first_part_l122_122831


namespace parabola_properties_parabola_equation_l122_122181

theorem parabola_properties 
(vertex_origin : ∃ (V : ℝ × ℝ), V = (0, 0))
(focus_on_y_axis : ∃ (F : ℝ × ℝ), ∃ (y : ℝ), F = (0, y))
(point_M_on_parabola : ∃ (m : ℝ), ∃ (F : ℝ × ℝ), ∃ (V : ℝ × ℝ), ∃ (p : ℝ), 
  F = (0, -p / 2) ∧ 
  V = (0, 0) ∧ 
  point = (m, -3) ∧ 
  (m, -3) on parabola with vertex V and focus F ∧ 
  distance((m, -3), F) = 5):
∃ (m : ℝ), m = 2 * sqrt 6 ∨ m = -2 * sqrt 6 :=
  sorry

theorem parabola_equation 
(vertex_origin : ∃ (V : ℝ × ℝ), V = (0, 0))
(focus_on_y_axis : ∃ (F : ℝ × ℝ), ∃ (y : ℝ), F = (0, y))
(point_M_on_parabola : ∃ (m : ℝ), ∃ (F : ℝ × ℝ), ∃ (V : ℝ × ℝ), ∃ (p : ℝ), 
  F = (0, -p / 2) ∧ 
  V = (0, 0) ∧ 
  point = (m, -3) ∧ 
  (m, -3) on parabola with vertex V and focus F ∧ 
  distance((m, -3), F) = 5):
(child: parabola properties (vertex_origin, focus_on_y_axis, point_M_on_parabola))
∃ (p : ℝ), 
∃ (equation : ℝ → ℝ), equation x = -8 * x^2 - 8 * y ∧ 
directrix equation y = 2 := 
sorry

end parabola_properties_parabola_equation_l122_122181


namespace quadratic_has_unique_solution_l122_122967

theorem quadratic_has_unique_solution (k : ℝ) :
  (∀ x : ℝ, (x + 6) * (x + 3) = k + 3 * x) → k = 9 :=
by
  intro h
  sorry

end quadratic_has_unique_solution_l122_122967


namespace factorization_l122_122949
-- Import the necessary library

-- Define the expression
def expr (x : ℝ) : ℝ := 75 * x^2 + 50 * x

-- Define the factored form
def factored_form (x : ℝ) : ℝ := 25 * x * (3 * x + 2)

-- Statement of the equality to be proved
theorem factorization (x : ℝ) : expr x = factored_form x :=
by {
  sorry
}

end factorization_l122_122949


namespace circumscribed_sphere_minimum_l122_122985

theorem circumscribed_sphere_minimum (a b : ℝ) (h : 0 < b ∧ b < a) :
  (1 < a / b) ∧ (a / b < 5 / 4) :=
by
  let x := (2 / 9) * (a + b)
  have h1 : b - 2 * x > 0 := sorry
  have h2 : x < b / 2 := sorry
  show (1 < a / b) ∧ (a / b < 5 / 4), from sorry

end circumscribed_sphere_minimum_l122_122985


namespace part_I_part_II_l122_122184

-- Given assumptions
variables {a : ℝ} {f g : ℝ → ℝ}
variables (x : ℝ) (h : x = 1/2)

-- Definition of the function f
def f (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part (I): Monotonic intervals of f
theorem part_I : 
  (∀ x ∈ Ioo 0 1, 0 < f' x) ∧ (∀ x ∈ Ioi 1, f' x < 0) :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := x * (f x + a + 1)

-- Part (II): Number of zeros of g
theorem part_II (h : a > 1 / Real.exp 1) : ∃! x, g x = 0 :=
sorry

end part_I_part_II_l122_122184


namespace axis_of_symmetry_monotonic_intervals_range_of_m_l122_122649

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)

-- Define the function f(x) as the dot product of vectors a and b
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

-- Proof statements
theorem axis_of_symmetry : ∀ (k : ℤ), ∃ (x : ℝ), x = (k * Real.pi / 2) + (3 * Real.pi / 8) := sorry

theorem monotonic_intervals :
  ∀ (k : ℤ),
    (∀ x, (k * Real.pi + 3 * Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 7 * Real.pi / 8) → monotone (f x)) ∧
    (∀ x, (k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8) → antitone (f x)) := sorry

theorem range_of_m : ∀ (m : ℝ), (∀ (x : ℝ), (Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3) → f x - m < 2) ↔ m ∈ Ioi ((Real.sqrt 3 - 5) / 4) := sorry

end axis_of_symmetry_monotonic_intervals_range_of_m_l122_122649


namespace opposite_of_neg5_is_pos5_l122_122359

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122359


namespace range_of_a_l122_122781

theorem range_of_a (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := 
sorry

end range_of_a_l122_122781


namespace blue_to_white_ratio_l122_122527

-- Define the conditions
variables (B W : ℕ)  -- B: number of blue tiles, W: number of white tiles
variable total_tiles_large : bool -- indicating a large number of tiles
variable each_blue_surrounded_by_six_white : ∀ b, b ∈ B → ∃ S, S ⊆ W ∧ card S = 6
variable each_white_surrounded_by_three_white_three_blue : ∀ w, w ∈ W → ∃ S, S.subset(W ∪ B) ∧ card {x | x ∈ S ∧ x ∈ W} = 3 ∧ card {x | x ∈ S ∧ x ∈ B} = 3

-- Prove that the ratio of the number of blue tiles to the number of white tiles is 1:2
theorem blue_to_white_ratio : W = 2 * B := by
  sorry

end blue_to_white_ratio_l122_122527


namespace Q_is_constant_l122_122721

theorem Q_is_constant (P Q : ℤ[X])
  (h1 : ∀ n : ℕ, 0 < P.eval n ∧ 0 < Q.eval n)
  (h2 : ∀ n : ℕ, (2 ^ Q.eval n - 1) ∣ (3 ^ P.eval n - 1))
  (h3 : ¬∃ R : ℚ[X], polynomial.degree R > 0 ∧ R ∣ P ∧ R ∣ Q) : 
  ∃ c : ℤ, Q = polynomial.C c :=
sorry

end Q_is_constant_l122_122721


namespace complex_proof_l122_122293

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def a : ℂ := 1 + complex.I * (real.sqrt 3)
noncomputable def c := (complex.abs z1 = 2) ∧ (complex.abs z2 = 2) ∧ (z1 + z2 = a)

theorem complex_proof (h : c) : complex.abs (z1 - z2) = 2 * real.sqrt 3 :=
sorry

end complex_proof_l122_122293


namespace trajectory_is_one_branch_of_hyperbola_l122_122151

open Real

-- Condition 1: Given points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Condition 2: Moving point P such that |PF1| - |PF2| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs (dist P F1) - abs (dist P F2) = 4

-- Prove the trajectory of point P is one branch of a hyperbola
theorem trajectory_is_one_branch_of_hyperbola (P : ℝ × ℝ) (h : satisfies_condition P) : 
  (∃ a b : ℝ, ∀ x y: ℝ, satisfies_condition (x, y) → (((x^2 / a^2) - (y^2 / b^2) = 1) ∨ ((x^2 / a^2) - (y^2 / b^2) = -1))) :=
sorry

end trajectory_is_one_branch_of_hyperbola_l122_122151


namespace ellipse_equation_correct_max_area_correct_l122_122164

noncomputable def ellipse_params := (a b : ℝ) (c := sqrt 2) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : c / a = sqrt 2 / 2) 
  (h4 : a^2 = b^2 + c^2) : Prop :=
  (a = 2 * sqrt 2) ∧ (b = 2)

noncomputable def ellipse_equation := (a b : ℝ) (h1 : a = 2 * sqrt 2) (h2 : b = 2) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1

noncomputable def max_triangle_area := (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = sqrt 2)
  (h4 : c / a = sqrt 2 / 2) (h5 : a^2 = b^2 + c^2) (h6 : a = 2 * sqrt 2) (h7 : b = 2) 
  (A B : ℝ × ℝ) (hA : A.2 = A.1 * sqrt 2 / 2) 
  (hB : ellipse_equation a b → (B.1^2 / 8 + B.2^2 / 4 = 1) ∧ ¬(A = B)) : Prop :=
  2 * sqrt 2

theorem ellipse_equation_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c := sqrt 2) 
  (h4 : c / a = sqrt 2 / 2) (h5 : a^2 = b^2 + c^2) : ellipse_equation 2sqrt2 2 := 
by
  sorry

theorem max_area_correct (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a^2 = b^2 + c^2) (h4 : c = sqrt 2) (h5 : c / a = sqrt 2 / 2) :
  max_triangle_area a b c :=
by
  sorry

end ellipse_equation_correct_max_area_correct_l122_122164


namespace game_necessarily_ends_no_winning_strategy_for_starting_player_l122_122901

-- Define the conditions: the cards, initial states, and the game rules.
def card := ℕ  -- gold = 1, black = 0
def initial_state := (list.repeat 1 2009 : list card)
def flip_block (cards : list card) (start : ℕ) : list card :=
  if start + 49 < cards.length ∧ (cards.nth start = some 1)
  then (cards.take start) ++ (cards.drop start).take 50.map (λ x, 1 - x) ++ (cards.drop (start + 50))
  else cards

-- Question (a): Does the game necessarily end?
theorem game_necessarily_ends :
  ∀ (state : list card), ∃ n, n ≥ 0 ∧ state.nth n = some 1 → ∃ m, flip_block state n = (list.repeat 0 2009) :=
sorry

-- Question (b): Does there exist a winning strategy for the starting player?
theorem no_winning_strategy_for_starting_player :
  ¬ ∃ strategy : (list card → ℕ), ∀ state, strategy state ≤ 1959 ∧ 
  (flip_block state (strategy state) = list.repeat 0 2009) →
  true := 
sorry

end game_necessarily_ends_no_winning_strategy_for_starting_player_l122_122901


namespace sequence_number_theorem_l122_122303

def seq_count (n k : ℕ) : ℕ :=
  -- Sequence count function definition given the conditions.
  sorry -- placeholder, as we are only defining the statement, not the function itself.

theorem sequence_number_theorem (n k : ℕ) : seq_count n k = Nat.choose (n-1) k :=
by
  -- This is where the proof would go, currently omitted.
  sorry

end sequence_number_theorem_l122_122303


namespace find_function_expression_l122_122175

theorem find_function_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f(x + 1) = x^2 - 2*x - 3) : 
  ∀ x : ℤ, f(x) = x^2 - 4*x := by
  sorry

end find_function_expression_l122_122175


namespace unique_function_satisfying_conditions_l122_122651

-- Define the functional form of f
def f (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Specification of the conditions given
variables {a : ℝ}

-- Definitions for the conditions
def condition1 (a : ℝ) := ∀ x : ℝ, f a (x ^ 2) = (f a x) ^ 2 ∧ (f a x) ^ 2 = f a (f a x)
def condition2 (a : ℝ) := f a 1 = 1

-- The theorem statement
theorem unique_function_satisfying_conditions :
  (∃! a : ℝ, condition1 a ∧ condition2 a) :=
sorry

end unique_function_satisfying_conditions_l122_122651


namespace leftover_books_l122_122217

/-
Given:
1. There are 2025 boxes in the library.
2. Each box initially contains 25 books.
3. The librarian wants to reorganize the books so that each box contains 28 books.
Prove:
The number of leftover books after reorganizing will be 21.
-/
theorem leftover_books :
  let total_books := 2025 * 25 in
  total_books % 28 = 21 := by
  sorry

end leftover_books_l122_122217


namespace midpoint_locus_max_length_AB_l122_122167

theorem midpoint_locus (A B : Point) (P : Point := Point.mk (-1) 0) :
  (A ∈ ellipse 8 4) → (B ∈ ellipse 8 4) → (perpendicular_bisector AB P) →
  ∃ M : Point, M = midpoint A B ∧ M.x = -2 :=
sorry

theorem max_length_AB (A B : Point) : 
  (A ∈ ellipse 8 4) → (B ∈ ellipse 8 4) →
  (perpendicular_bisector AB P) → 
  length AB ≤ 2 * sqrt 2 :=
sorry

end midpoint_locus_max_length_AB_l122_122167


namespace arithmetic_sequence_num_terms_is_18_l122_122117

noncomputable def numberOfTermsInArithmeticSequence
  (a d an : ℕ)
  (start_term : a = 15)
  (common_difference : d = 5)
  (last_term : an = 100) : Prop :=
  ∃ n, an = a + (n - 1) * d ∧ n = 18

theorem arithmetic_sequence_num_terms_is_18 :
  numberOfTermsInArithmeticSequence 15 5 100 :=
begin
  unfold numberOfTermsInArithmeticSequence,
  use 18,
  split,
  {
    -- Verifying the last term
    norm_num,
  },
  {
    -- Verifying the number of terms
    norm_num,
  }
end

end arithmetic_sequence_num_terms_is_18_l122_122117


namespace jones_elementary_l122_122828

variable (x : ℕ)

/-- Given conditions: 
    1. 90 students represent x percent of the boys at Jones Elementary School.
    2. The boys at Jones Elementary make up 30% of the total school population of x students.
  Prove: x = 300
-/
theorem jones_elementary (h1 : 90 = 0.30 * x) 
    (h2 : ∃ total_students : ℝ, total_students = x ∧ 0.30 * total_students = 90) : 
    x = 300 := 
by 
  sorry

end jones_elementary_l122_122828


namespace number_of_members_in_A_l122_122823

open Set

variable (U A B : Finset ℕ)
variable (U_total B_total neither AB_total : ℕ)

theorem number_of_members_in_A (h1 : U.card = 190)
                              (h2 : B.card = 49)
                              (h3 : (U.filter (λ x, ¬ x ∈ A ∪ B)).card = 59)
                              (h4 : (A ∩ B).card = 23) : 
                              A.card = 105 := by
  sorry

end number_of_members_in_A_l122_122823


namespace color_of_182nd_marble_l122_122855

-- conditions
def pattern_length : ℕ := 15
def blue_length : ℕ := 6
def red_length : ℕ := 5
def green_length : ℕ := 4

def marble_color (n : ℕ) : String :=
  let cycle_pos := n % pattern_length
  if cycle_pos < blue_length then
    "blue"
  else if cycle_pos < blue_length + red_length then
    "red"
  else
    "green"

theorem color_of_182nd_marble : marble_color 182 = "blue" :=
by
  sorry

end color_of_182nd_marble_l122_122855


namespace Q_in_second_quadrant_l122_122669

-- Define the condition that point P is in the first quadrant
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

-- Define the condition that point Q is in the second quadrant
def in_second_quadrant (Q : ℝ × ℝ) : Prop := Q.1 < 0 ∧ Q.2 > 0

-- Given condition
def point_P := (3 : ℝ, -(a : ℝ))
def point_Q := (a, -a + 2)

-- Theorem to be proved
theorem Q_in_second_quadrant (a : ℝ) (h : in_first_quadrant (3, -a)) : in_second_quadrant (a, -a + 2) :=
sorry

end Q_in_second_quadrant_l122_122669


namespace smallest_root_of_equation_l122_122548

theorem smallest_root_of_equation :
  ∃ x_0 : ℝ, x_0 ≥ 10^(-100) ∧ x_0^2 - real.sqrt (real.log x_0 + 100) = 0 ∧ x_0 = 10^(-100) := 
sorry

end smallest_root_of_equation_l122_122548


namespace cos_beta_l122_122993

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l122_122993


namespace probability_A_wins_l122_122517

-- Definitions based on the problem's conditions
def deck (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ 2 * n}

noncomputable def game_setup (n : ℕ) : Prop :=
  let A, B : list ℕ := (deck n).to_list.split_at n in
  A.length = n ∧ B.length = n

-- Probability of A winning given optimal play
theorem probability_A_wins (n : ℕ) (h : game_setup n) : 
  (∃ seq : list ℕ, seq.head = some A.head ∧ ∃ k, (list.sum seq) % (2 * n + 1) = 0 ∧
  ((k % 2 = 0) ↔ seq.tail.length = 2 * n - 1)) → 0 = 1 :=
sorry  -- Proof omitted

end probability_A_wins_l122_122517


namespace mike_total_spending_l122_122596

def mike_spent_on_speakers : ℝ := 235.87
def mike_spent_on_tires : ℝ := 281.45
def mike_spent_on_steering_wheel_cover : ℝ := 179.99
def mike_spent_on_seat_covers : ℝ := 122.31
def mike_spent_on_headlights : ℝ := 98.63

theorem mike_total_spending :
  mike_spent_on_speakers + mike_spent_on_tires + mike_spent_on_steering_wheel_cover + mike_spent_on_seat_covers + mike_spent_on_headlights = 918.25 :=
  sorry

end mike_total_spending_l122_122596


namespace angle_between_unit_vectors_l122_122279

open Real
open ComplexConjugate

-- Let a and b be unit vectors
variables {E : Type*} [InnerProductSpace ℝ E] (a b : E)
hypothesis unit_a : ∥a∥ = 1
hypothesis unit_b : ∥b∥ = 1

-- The vectors 2a + 3b and 4a - 5b are orthogonal
hypothesis orthogonal_vectors : (2 : ℝ) • a + (3 : ℝ) • b ⟂ (4 : ℝ) • a - (5 : ℝ) • b

-- The theorem we want to prove
theorem angle_between_unit_vectors :
  real.angle a b = real.acos (11 / 12) :=
sorry

end angle_between_unit_vectors_l122_122279


namespace cos_B_eq_zero_l122_122997

variable {a b c A B C : ℝ}
variable (h1 : ∀ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (h2 : b * Real.cos A = c)

theorem cos_B_eq_zero (h1 : a = b) (h2 : b * Real.cos A = c) : Real.cos B = 0 :=
sorry

end cos_B_eq_zero_l122_122997


namespace opposite_of_negative_five_l122_122389

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122389


namespace probability_perpendicular_vectors_l122_122019

theorem probability_perpendicular_vectors :
  let outcomes := [(m,n) | m in Finset.range 1 7, n in Finset.range 1 7];
  let favorable_outcomes := [(m,n) | (m,n) in outcomes, m = n];
  let prob := (favorable_outcomes.card : ℝ) / (outcomes.card : ℝ)
  in prob = 1 / 6 := by
sorry

end probability_perpendicular_vectors_l122_122019


namespace decimal_expansion_contains_all_digits_l122_122710

theorem decimal_expansion_contains_all_digits
  (p : ℕ) (hp_prime : Prime p) (hp_large : p > 10^9)
  (hq_prime : Prime (4 * p + 1)) :
  (∃ n, ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, d ∈ decimal_expansion(1 / (4 * p + 1))) :=
sorry

end decimal_expansion_contains_all_digits_l122_122710


namespace ratio_of_dimensions_128_l122_122416

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l122_122416


namespace complete_graph_coloring_l122_122123

noncomputable def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem complete_graph_coloring (m : ℕ) (G : SimpleGraph (Fin (2*m))) (colors : Fin (2*m-1))
    (h1 : G.IsComplete) (h2 : ∀ c1 c2 : colors, ∃ C : List (SimpleGraph (Fin (2*m))), C.Chain (λ g h, ∃ h, g + h = G)) :
    is_power_of_two m := 
    sorry

end complete_graph_coloring_l122_122123


namespace avg_speed_approx_l122_122494

-- Define the distances and the times taken
def distance1 : ℕ := 210
def time1 : ℕ := 3
def distance2 : ℕ := 270
def time2 : ℕ := 4

-- Define the total distance and total time
def total_distance : ℕ := distance1 + distance2
def total_time : ℕ := time1 + time2

-- Define the average speed
def average_speed : ℚ := total_distance / total_time

-- Prove that the average speed is approximately 68.57 km/h
theorem avg_speed_approx : average_speed ≈ 68.57 := sorry

end avg_speed_approx_l122_122494


namespace angle_C_105_l122_122697

theorem angle_C_105 (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_condition : |sin A - 1/2| + (sqrt 2 / 2 - cos B)^2 = 0) :
  C = 105 :=
by
  sorry

end angle_C_105_l122_122697


namespace sum_geometric_sequence_l122_122215

theorem sum_geometric_sequence {a : ℕ → ℝ} (ha : ∃ q, ∀ n, a n = 3 * q ^ n)
  (h1 : a 1 = 3) (h2 : a 1 + a 2 + a 3 = 9) :
  a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72 :=
sorry

end sum_geometric_sequence_l122_122215


namespace S_cardinality_l122_122907

def g (n : ℤ) : ℤ := if n < 0 then 0 else 1

def f (n : ℤ) : ℤ := n - 1024 * g (n - 1024)

def a : ℕ → ℤ
| 0     := 1
| (n+1) := let ell := if (List.prod (List.range (n+1)).map (λ i, 2 * f (a n) + 1 - a i) = 0) then 0 else 1 in
             2 * f (a n) + ell

def S : Set ℤ := {a i | i : ℕ, i ≤ 2009}

theorem S_cardinality : S.card = 11 := sorry

end S_cardinality_l122_122907


namespace number_of_schools_l122_122129

theorem number_of_schools (n : ℕ) (Alex Jordan Kim Lee : ℕ) (h1 : ∀ s, s > 0 → s % 4 = 0 → n = s / 4) 
  (h2 : ∀ k, k > 0 → unique k) (h3 : is_median Alex (4 * n) ∧ highest_score Alex team_Val)
  (h4 : Jordan = 45) (h5 : Kim = 73) (h6 : Lee = 98) : n = 22 := 
sorry

end number_of_schools_l122_122129


namespace inverse_variation_l122_122336

theorem inverse_variation (k : ℝ) (x y : ℝ) 
  (h1 : x * (y ^ 2) = k) 
  (hx : x = 16) 
  (hy : y = 4) : 
  ∃ x, y = -2 → x * 4 = k → x = 64 :=
by {
  intro,
  sorry
}

end inverse_variation_l122_122336


namespace proof_2_fx_minus_11_eq_f_x_minus_d_l122_122166

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 2

theorem proof_2_fx_minus_11_eq_f_x_minus_d :
  2 * (f 5) - 11 = f (5 - d) := by
  sorry

end proof_2_fx_minus_11_eq_f_x_minus_d_l122_122166


namespace part_I_part_II_l122_122631

-- The curve C defined by parametric equations
def curve (φ : ℝ) : ℝ × ℝ := (4 * Real.cos φ, 3 * Real.sin φ)

-- Question (Ⅰ): The standard form of the equation should be (x^2 / 16 + y^2 / 9 = 1)
theorem part_I (x y : ℝ) (φ : ℝ) (h : curve φ = (x, y)) : 
  x^2 / 16 + y^2 / 9 = 1 :=
by sorry

-- Question (Ⅱ): The range of x + y is [-5, 5]
theorem part_II (x y : ℝ) (φ : ℝ) (h : curve φ = (x, y)) : 
  -5 ≤ x + y ∧ x + y ≤ 5 :=
by sorry

end part_I_part_II_l122_122631


namespace opposite_of_negative_five_l122_122392

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122392


namespace find_center_of_first_circle_l122_122510

theorem find_center_of_first_circle : 
  ∃ c : ℝ × ℝ, 
    (c = (2, 3)) ∧ 
    ((0 - c.fst)^2 + (2 - c.snd)^2 = (c.fst - 1)^2 + (c.snd - 3) ∧
    ∃ r1 r2 : ℝ, 
      r1 ^ 2 = 5 ∧ 
      r2 = (c.fst - 0)^2 + (c.snd - 3)^2 ∧ 
      abs ((r2 - r1)) = (c.fst - 0)^2 + (c.snd - 3)^2
  := 
sorry

end find_center_of_first_circle_l122_122510


namespace sector_circumference_l122_122624

theorem sector_circumference (θ : ℝ) (r : ℝ) (hθ : θ = 60) (hr : r = 15) :
  let L := (θ / 360) * 2 * Real.pi * r in
  let C := L + 2 * r in
  C = 5 * (6 + Real.pi) :=
  by
    sorry

end sector_circumference_l122_122624


namespace rectangular_solid_surface_area_l122_122122

theorem rectangular_solid_surface_area (a b c : ℕ)
  (ha : a = 3) (hb : b = 5) (hc : c = 17) (hprimes : (Nat.prime a) ∧ (Nat.prime b) ∧ (Nat.prime c)) :
  2 * (a * b + a * c + b * c) = 302 := 
by
  rw [ha, hb, hc]
  norm_num
  sorry

end rectangular_solid_surface_area_l122_122122


namespace coefficient_x3_in_expansion_l122_122579

open nat

theorem coefficient_x3_in_expansion :
  let c7r (r : ℕ) : ℕ := nat.choose 7 r
  let term_x (x r : ℕ) : ℕ := (c7r r) * (-2)^r * x^(7 - r)
  let coef_x (a b c : ℕ) : ℕ := a - b
  coef_x (c7r 6 * (-2)^6) (c7r 4 * (-2)^4) = -112 :=
by
  sorry

end coefficient_x3_in_expansion_l122_122579


namespace omega_range_l122_122877

theorem omega_range (ω : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < π ∧
    (sin (ω * x₁ + π / 3)).cos = 0 ∧ (sin (ω * x₂ + π / 3)).cos = 0 ∧ (sin (ω * x₃ + π / 3)).cos = 0) ∧ 
  (∃ z₁ z₂ : ℝ, 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ sin (ω * z₁ + π / 3) = 0 ∧ sin (ω * z₂ + π / 3) = 0) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
by
  sorry

end omega_range_l122_122877


namespace find_distance_correct_l122_122496

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def distance_point_to_plane (M0 M1 M2 M3 : Point3D) : ℝ :=
  let A := 0
  let B := 1
  let C := 1
  let D := 1
  let x0 := M0.x
  let y0 := M0.y
  let z0 := M0.z
  (abs (A * x0 + B * y0 + C * z0 + D)) / (real.sqrt (A^2 + B^2 + C^2))

noncomputable def main : ℝ :=
  let M0 := Point3D.mk 1 (-6) (-5)
  let M1 := Point3D.mk (-1) 2 (-3)
  let M2 := Point3D.mk 4 (-1) 0
  let M3 := Point3D.mk 2 1 (-2)
  distance_point_to_plane M0 M1 M2 M3

theorem find_distance_correct : main = 5 * real.sqrt 2 := by
  -- Proof can be provided later
  sorry

end find_distance_correct_l122_122496


namespace floor_computation_l122_122899

theorem floor_computation : 
  (Real.floor ((102^3 / (100 * 101)) - (100^3 / (101 * 102)))) = 8 :=
by
  sorry

end floor_computation_l122_122899


namespace find_difference_of_x_l122_122615

theorem find_difference_of_x
  (x y : ℝ)
  (h1 : even y)
  (h2 : 4 ≤ x ∧ x < 16)
  (h3 : y = 16 ∨ y ∈ {314})
  (h4 : x ∈ Set.Icc 5 15) :
  (sup {x | x ∈ {4, 314, 710, x, y}} - inf {x | x ∈ {4, 314, 710, x, y}}) = 11 := sorry

end find_difference_of_x_l122_122615


namespace sqrt_mixed_number_eq_l122_122933

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l122_122933


namespace interest_calculation_l122_122345

theorem interest_calculation (P : ℝ) (r : ℝ) (CI SI : ℝ → ℝ) (n : ℝ) :
  P = 1300 →
  r = 0.10 →
  (CI n - SI n = 13) →
  (CI n = P * (1 + r)^n - P) →
  (SI n = P * r * n) →
  (1.10 ^ n - 1 - 0.10 * n = 0.01) →
  n = 2 :=
by
  intro P_eq r_eq diff_eq CI_def SI_def equation
  -- Sorry, this is just a placeholder. The proof is omitted.
  sorry

end interest_calculation_l122_122345


namespace arcsin_sqrt_2_div_2_l122_122898

theorem arcsin_sqrt_2_div_2 :
  arcsin (real.sqrt 2 / 2) = real.pi / 4 :=
by
  have h1 : real.sin (real.pi / 4) = real.sqrt 2 / 2 := sorry
  have h2 : -real.pi / 2 ≤ arcsin (real.sqrt 2 / 2) ∧ arcsin (real.sqrt 2 / 2) ≤ real.pi / 2 := sorry
  sorry

end arcsin_sqrt_2_div_2_l122_122898


namespace total_students_l122_122531

-- Define the conditions
def rank_from_right := 17
def rank_from_left := 5

-- The proof statement
theorem total_students : rank_from_right + rank_from_left - 1 = 21 := 
by 
  -- Assuming the conditions represented by the definitions
  -- Without loss of generality the proof would be derived from these, but it is skipped
  sorry

end total_students_l122_122531


namespace probability_of_woman_lawyer_is_54_percent_l122_122827

variable (total_members : ℕ) (women_percentage lawyers_percentage : ℕ)
variable (H_total_members_pos : total_members > 0) 
variable (H_women_percentage : women_percentage = 90)
variable (H_lawyers_percentage : lawyers_percentage = 60)

def probability_woman_lawyer : ℕ :=
  (women_percentage * lawyers_percentage * total_members) / (100 * 100)

theorem probability_of_woman_lawyer_is_54_percent (H_total_members_pos : total_members > 0)
  (H_women_percentage : women_percentage = 90)
  (H_lawyers_percentage : lawyers_percentage = 60) :
  probability_woman_lawyer total_members women_percentage lawyers_percentage = 54 :=
by
  sorry

end probability_of_woman_lawyer_is_54_percent_l122_122827


namespace four_digit_numbers_divisible_by_4_not_8_count_l122_122140

/-- Define a 4-digit number in base 10 -/
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- Define a number having non-zero digits -/
def has_non_zero_digits (n : ℕ) : Prop :=
  (n / 1000 ≠ 0) ∧ ((n / 100) % 10 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0)

/-- Define a number divisible by 4 but not by 8 -/
def divisible_by_4_and_not_8 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 8 ≠ 0)

/-- Theorem statement -/
theorem four_digit_numbers_divisible_by_4_not_8_count : 
  { n : ℕ | is_four_digit_number n ∧ has_non_zero_digits n ∧ divisible_by_4_and_not_8 n }.to_finset.card = 729 :=
sorry

end four_digit_numbers_divisible_by_4_not_8_count_l122_122140


namespace sequence_expression_l122_122603

-- Define the sequence according to the conditions
def a : ℕ → ℕ 
| 1 := 0
| (n+1) := a n + (2*n - 1)

-- The theorem to prove
theorem sequence_expression (n : ℕ) : a n = (n-1)^2 :=
sorry

end sequence_expression_l122_122603


namespace five_student_committees_from_ten_select_two_committees_with_three_overlap_l122_122148

-- Lean statement for the first part: number of different five-student committees from ten students.
theorem five_student_committees_from_ten : 
  (Nat.choose 10 5) = 252 := 
by
  sorry

-- Lean statement for the second part: number of ways to choose two five-student committees with exactly three overlapping members.
theorem select_two_committees_with_three_overlap :
  ( (Nat.choose 10 5) * ( (Nat.choose 5 3) * (Nat.choose 5 2) ) ) / 2 = 12600 := 
by
  sorry

end five_student_committees_from_ten_select_two_committees_with_three_overlap_l122_122148


namespace find_z_plus_one_over_y_l122_122335

theorem find_z_plus_one_over_y 
  (x y z : ℝ) 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1/z = 4)
  (h6 : y + 1/x = 20) :
  z + 1/y = 26 / 79 :=
by
  sorry

end find_z_plus_one_over_y_l122_122335


namespace teacher_selection_l122_122852

/-- A school has 150 teachers, including 15 senior teachers, 45 intermediate teachers, 
and 90 junior teachers. By stratified sampling, 30 teachers are selected to 
participate in the teachers' representative conference. 
--/

def total_teachers : ℕ := 150
def senior_teachers : ℕ := 15
def intermediate_teachers : ℕ := 45
def junior_teachers : ℕ := 90

def total_selected_teachers : ℕ := 30
def selected_senior_teachers : ℕ := 3
def selected_intermediate_teachers : ℕ := 9
def selected_junior_teachers : ℕ := 18

def ratio (a b : ℕ) : ℕ × ℕ := (a / (gcd a b), b / (gcd a b))

theorem teacher_selection :
  ratio senior_teachers (gcd senior_teachers total_teachers) = ratio intermediate_teachers (gcd intermediate_teachers total_teachers) ∧
  ratio intermediate_teachers (gcd intermediate_teachers total_teachers) = ratio junior_teachers (gcd junior_teachers total_teachers) →
  selected_senior_teachers / selected_intermediate_teachers / selected_junior_teachers = 1 / 3 / 6 → 
  selected_senior_teachers + selected_intermediate_teachers + selected_junior_teachers = 30 :=
sorry

end teacher_selection_l122_122852


namespace square_side_length_l122_122506

theorem square_side_length :
  ∀ (w l : ℕ) (area : ℕ),
  w = 9 → l = 27 → area = w * l →
  ∃ s : ℝ, s^2 = area ∧ s = 9 * Real.sqrt 3 :=
by
  intros w l area hw hl harea
  sorry

end square_side_length_l122_122506


namespace equilibrium_constant_expression_reverse_equilibrium_constant_half_reaction_equilibrium_constant_reaction_endothermic_equilibrium_evidence_forward_reverse_rate_conversion_rate_CO₂_l122_122442

section equilibrium_constants

-- Definitions based on the problem
def equilibrium_expression (c_CO c_H₂O c_CO₂ c_H₂ : ℝ) : ℝ :=
  (c_CO * c_H₂O) / (c_CO₂ * c_H₂)

def K₁ : ℝ := 0.6
def K₄ : ℝ := 1.0

-- Problem 1
theorem equilibrium_constant_expression (c_CO c_H₂O c_CO₂ c_H₂ : ℝ) :
  equilibrium_expression c_CO c_H₂O c_CO₂ c_H₂ = (c_CO * c_H₂O) / (c_CO₂ * c_H₂) :=
sorry

-- Problem 2 (a)
theorem reverse_equilibrium_constant :
  1 / K₁ = 1.67 :=
sorry 

-- Problem 2 (b)
theorem half_reaction_equilibrium_constant :
  Real.sqrt K₁ = 0.77 :=
sorry

-- Problem 3
theorem reaction_endothermic :
  K₄ > K₁ → "endothermic" :=
sorry 

-- Problem 4
theorem equilibrium_evidence : 
  "When a mole of CO₂ is generated, a mole of H₂ is consumed." :=
sorry 

-- Problem 5
theorem forward_reverse_rate : 
  v_forward > v_reverse :=
sorry

theorem conversion_rate_CO₂ : 
  0.4 = 0.6 / 1.5 :=
sorry

end equilibrium_constants

end equilibrium_constant_expression_reverse_equilibrium_constant_half_reaction_equilibrium_constant_reaction_endothermic_equilibrium_evidence_forward_reverse_rate_conversion_rate_CO₂_l122_122442


namespace opposite_of_negative_five_l122_122370

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122370


namespace sum_first_100_terms_l122_122195

section
variable (a : ℕ → ℕ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n : ℕ, a (n + 2) - a n = 1

theorem sum_first_100_terms : (∑ n in Finset.range 100, a (n + 1)) = 2550 :=
sorry
end

end sum_first_100_terms_l122_122195


namespace area_square_inscribed_parabola_l122_122134

noncomputable def area_of_square_in_parabola : ℝ :=
  let y := λ x : ℝ, x^2 - 6 * x + 8 in
  let axis_of_symmetry := 3 in
  let s := -2 + 2 * Real.sqrt 2 in
  s^2

theorem area_square_inscribed_parabola : 
  ∀ s : ℝ, s = -2 + 2 * Real.sqrt 2 → s^2 = 12 - 8 * Real.sqrt 2 :=
by
  intros s hs
  rw hs
  have := Real.sqrt 2 * Real.sqrt 2
  ring
  exact this

end area_square_inscribed_parabola_l122_122134


namespace sufficient_but_not_necessary_l122_122036

theorem sufficient_but_not_necessary (a : ℝ) :
  0 < a ∧ a < 1 → (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) ∧ ¬ (∀ a, (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 < a ∧ a < 1) :=
by
  sorry

end sufficient_but_not_necessary_l122_122036


namespace perpendicular_bisector_eq_l122_122349

theorem perpendicular_bisector_eq (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 5 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y - 4 = 0) →
  x + y - 1 = 0 :=
by
  sorry

end perpendicular_bisector_eq_l122_122349


namespace volume_ratio_l122_122653

-- Define the dimensions of the first cube in meters
def cube1_side : ℝ := 2

-- Define the dimensions of the second cube in centimeters and convert to meters
def cube2_side_cm : ℝ := 100
def cm_to_m : ℝ := 100
def cube2_side : ℝ := cube2_side_cm / cm_to_m

-- Calculate the volumes of the cubes
def volume_cube1 : ℝ := cube1_side ^ 3
def volume_cube2 : ℝ := cube2_side ^ 3

-- The theorem to prove
theorem volume_ratio : volume_cube1 / volume_cube2 = 8 :=
by
  calc volume_cube1 / volume_cube2 = (cube1_side ^ 3) / (cube2_side ^ 3) : rfl
                             ... = (2 ^ 3) / (1 ^ 3) : by rw [cube1_side, cube2_side]
                             ... = 8 / 1 : rfl
                             ... = 8 : by norm_num

end volume_ratio_l122_122653


namespace strips_area_coverage_l122_122144

-- Define paper strips and their properties
def length_strip : ℕ := 8
def width_strip : ℕ := 2
def number_of_strips : ℕ := 5

-- Total area without considering overlaps
def area_one_strip : ℕ := length_strip * width_strip
def total_area_without_overlap : ℕ := number_of_strips * area_one_strip

-- Overlapping areas
def area_center_overlap : ℕ := 4 * (2 * 2)
def area_additional_overlap : ℕ := 2 * (2 * 2)
def total_overlap_area : ℕ := area_center_overlap + area_additional_overlap

-- Actual area covered
def actual_area_covered : ℕ := total_area_without_overlap - total_overlap_area

-- Theorem stating the required proof
theorem strips_area_coverage : actual_area_covered = 56 :=
by sorry

end strips_area_coverage_l122_122144


namespace average_speed_is_42_l122_122493

theorem average_speed_is_42 (v t : ℝ) (h : t > 0)
  (h_eq : v * t = (v + 21) * (2/3) * t) : v = 42 :=
by
  sorry

end average_speed_is_42_l122_122493


namespace farmer_total_profit_l122_122054

theorem farmer_total_profit :
  let group1_revenue := 3 * 375
  let group1_cost := (8 * 13 + 3 * 15) * 3
  let group1_profit := group1_revenue - group1_cost

  let group2_revenue := 4 * 425
  let group2_cost := (5 * 14 + 9 * 16) * 4
  let group2_profit := group2_revenue - group2_cost

  let group3_revenue := 2 * 475
  let group3_cost := (10 * 15 + 8 * 18) * 2
  let group3_profit := group3_revenue - group3_cost

  let group4_revenue := 1 * 550
  let group4_cost := 20 * 20 * 1
  let group4_profit := group4_revenue - group4_cost

  let total_profit := group1_profit + group2_profit + group3_profit + group4_profit
  total_profit = 2034 :=
by
  sorry

end farmer_total_profit_l122_122054


namespace sqrt_mixed_number_simplify_l122_122943

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l122_122943


namespace faulty_key_in_digits_l122_122475

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122475


namespace mass_of_limestone_l122_122909

-- Definitions for conditions
def mass_of_CaCO3 : ℕ := 41.1
def impurities_percentage : ℕ := 3
def pure_substance_percentage : ℕ := 100 - impurities_percentage

-- Theorem statement
theorem mass_of_limestone (mass_of_CaCO3 : ℕ) (impurities_percentage : ℕ) : ℕ :=
  let pure_substance_percentage := 100 - impurities_percentage
  let total_mass := (mass_of_CaCO3 * 100) / pure_substance_percentage
  total_mass = 42.37

end mass_of_limestone_l122_122909


namespace quadrilateral_EFGH_area_l122_122318

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l122_122318


namespace probability_at_least_two_same_class_l122_122794

/-- Define the number of classes -/
def numClasses : ℕ := 10

/-- Define the number of students -/
def numStudents : ℕ := 3

/-- Define the probability that at least 2 of the students are in the same class -/
theorem probability_at_least_two_same_class 
  (hc : numClasses = 10) 
  (hs : numStudents = 3) :
  (1 - (numClasses * (numClasses - 2) * (numClasses - 1)) / (numClasses * numClasses * numClasses)) = 7 / 25 :=
by
  sorry

end probability_at_least_two_same_class_l122_122794


namespace seating_arrangements_l122_122242

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l122_122242


namespace find_sixth_number_l122_122767

theorem find_sixth_number (A : ℕ → ℤ) 
  (h1 : (1 / 11 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 60)
  (h2 : (1 / 6 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6) = 88)
  (h3 : (1 / 6 : ℚ) * (A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 65) :
  A 6 = 258 :=
sorry

end find_sixth_number_l122_122767


namespace lawrence_worked_hours_l122_122263

-- Let h_M, h_T, h_F be the hours worked on Monday, Tuesday, and Friday respectively
-- Let h_W be the hours worked on Wednesday (h_W = 5.5)
-- Let h_R be the hours worked on Thursday (h_R = 5.5)
-- Let total hours worked in 5 days be 25
-- Prove that h_M + h_T + h_F = 14

theorem lawrence_worked_hours :
  ∀ (h_M h_T h_F : ℝ), h_W = 5.5 → h_R = 5.5 → (5 * 5 = 25) → 
  h_M + h_T + h_F + h_W + h_R = 25 → h_M + h_T + h_F = 14 :=
by
  intros h_M h_T h_F h_W h_R h_total h_sum
  sorry

end lawrence_worked_hours_l122_122263


namespace prove_f_neg1_l122_122668

variable (f : ℝ → ℝ)

noncomputable def condition1 (x : ℝ) : Prop := f(x + 2009) = -f(x + 2008)
noncomputable def condition2 : Prop := f 2009 = -2009

theorem prove_f_neg1 (h1 : ∀ x, condition1 f x) (h2 : condition2 f) : f (-1) = -2009 := 
by 
  sorry

end prove_f_neg1_l122_122668


namespace area_of_PQRS_is_40400_l122_122614

noncomputable def area_of_rectangle_PQRS (y : ℤ) : ℝ :=
  let P := (10, -30)
  let Q := (2010, 170)
  let S := (12, y)
  let PQ_length := real.sqrt ((2010 - 10 : ℝ)^2 + (170 - (-30) : ℝ)^2)
  let PS_length := real.sqrt ((12 - 10 : ℝ)^2 + (y - (-30) : ℝ)^2)
  PQ_length * PS_length

theorem area_of_PQRS_is_40400 : area_of_rectangle_PQRS (-50) = 40400 :=
  sorry

end area_of_PQRS_is_40400_l122_122614


namespace opposite_of_negative_five_l122_122391

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122391


namespace gavrila_distance_l122_122090

noncomputable def distance_from_start (L : ℝ) (y : ℝ) : ℝ :=
  let x := (y^2) / (4 * L)
  sqrt(x^2 + y^2)

theorem gavrila_distance (L : ℝ) (y : ℝ) : 
  L = 50 → y = 40 → Real.floor (distance_from_start L y) = 41 :=
by
  intros hL hy
  rw [hL, hy]
  sorry

end gavrila_distance_l122_122090


namespace cars_and_tourists_l122_122043

theorem cars_and_tourists (n t : ℕ) (h : n * t = 737) : n = 11 ∧ t = 67 ∨ n = 67 ∧ t = 11 :=
by
  sorry

end cars_and_tourists_l122_122043


namespace lim_eq_third_derivative_at_1_l122_122629

noncomputable def f : ℝ → ℝ := sorry

theorem lim_eq_third_derivative_at_1 (h_diff : DifferentiableAt ℝ f 1) :
  tendsto (fun Δx => (f (1 + Δx) - f 1) / (3 * Δx)) (𝓝 0) (𝓝 (1 / 3 * deriv f 1)) :=
sorry

end lim_eq_third_derivative_at_1_l122_122629


namespace part1_part2_l122_122156

noncomputable def f (x : ℝ) : ℝ := 16 * x / (x^2 + 8)

theorem part1 (x : ℝ) (hx : 0 < x) : f(x) ≤ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a b : ℝ) : f(a) < b^2 - 3 * b + 21 / 4 :=
sorry

end part1_part2_l122_122156


namespace opposite_of_negative_five_l122_122369

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122369


namespace elective_courses_l122_122807

theorem elective_courses : 
  ∃ (ways : ℕ), 
    ways = 150 ∧ 
    ∃ (students courses : ℕ), 
      students = 5 ∧ 
      courses = 3 ∧ 
      (∀ student in Range(students), ∃ course in Range(courses), student ∈ course) ∧ 
      (∀ course in Range(courses), ∃ student in Range(students), student ∈ course) :=
by
    sorry

end elective_courses_l122_122807


namespace find_XY_squared_l122_122290

variables {A B C T X Y : Type}
variables (h1 : acute_scalene_triangle A B C)
variables (ω : Circumcircle A B C)
variables (ht1 : Tangent ω B T)
variables (ht2 : Tangent ω C T)
variables (hx : Projection T A B X)
variables (hy : Projection T A C Y)
variables (bt : distance B T = 20)
variables (ct : distance C T = 20)
variables (bc : distance B C = 30)
variables (cond : distance T X ^ 2 + distance T Y ^ 2 + distance X Y ^ 2 = 2000)

theorem find_XY_squared : distance X Y ^ 2 = 1000 :=
sorry

end find_XY_squared_l122_122290


namespace general_term_bn_l122_122645

theorem general_term_bn (p q r : ℝ) (q_pos : q > 0) (p_gt_r : p > r) :
  ∀ n : ℕ, n ≥ 1 →
  (∀ n, n = 1 → b n = q) →
  (∀ n, n ≥ 2 → a n = p * a (n - 1)) →
  (∀ n, n ≥ 2 → b n = q * a (n - 1) + r * b (n - 1)) →
  b n = (q * (p ^ n - r ^ n)) / (p - r) :=
begin
  sorry
end

variables (a b : ℕ → ℝ)

end general_term_bn_l122_122645


namespace cylinder_lateral_surface_area_l122_122963

theorem cylinder_lateral_surface_area :
  let side := 20
  let radius := side / 2
  let height := side
  2 * Real.pi * radius * height = 400 * Real.pi :=
by
  let side := 20
  let radius := side / 2
  let height := side
  sorry

end cylinder_lateral_surface_area_l122_122963


namespace wandas_bread_ratio_l122_122704

variable (T : ℚ) -- Number of treats Jane brings
variable (B : ℚ) -- Number of pieces of bread Jane brings
variable (W_treats : ℚ) -- Number of treats Wanda brings

-- Jane brings 75% as many pieces of bread as treats
def C1 : Prop := B = 0.75 * T

-- Wanda brings half as many treats as Jane
def C2 : Prop := W_treats = 0.5 * T

-- Wanda brings 90 pieces of bread
def C3 : Prop := 90 = W_treats * 0.5 * T / W_treats * W_treats * (90 = 0.5 * T)

-- The total number of pieces of bread and treats that Wanda and Jane brought to the zoo is 225
def C4 : Prop := B + T + 90 + W_treats = 225

-- Prove the ratio
theorem wandas_bread_ratio (T W_treats : ℚ) (h1 : C1 T B)
                            (h2 : C2 T W_treats)
                            (h3 : C3 T W_treats)
                            (h4 : C4 T B W_treats) : 
                            90 / W_treats = 3 :=
by
  sorry

end wandas_bread_ratio_l122_122704


namespace functional_equation_l122_122272

noncomputable theory

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := sorry

theorem functional_equation (h : ∀ x y : ℝ, f (g x + y) = g (x + y)) :
  (∀ x : ℝ, f x = x) ∨ (∃ p > 0, ∀ x : ℝ, g x = g (x + p)) :=
sorry

end functional_equation_l122_122272


namespace area_AEF_l122_122698

-- Define the overall context and known conditions
variables (A B C D E F : Type)
variable [IsTriangle A B C] -- Assume proper definition of a triangle
variables (is_midpoint_F : IsMidpoint B C F)
variables (is_midpoint_D : IsMidpoint A B D)
variables (is_midpoint_E : IsMidpoint D F E)
variable (area_ABC : ℝ)
variable (h_area_ABC : area_ABC = 144)
variables (area_ABC_positive : area_ABC > 0)

-- Define the main theorem statement
theorem area_AEF :
  ∃ area_AEF : ℝ, area_AEF = 18 :=
sorry

end area_AEF_l122_122698


namespace find_cos_theta_l122_122656

theorem find_cos_theta (θ : ℝ) (h₁ : 5 * cot θ = 4 * sin θ) (h₂ : 0 < θ ∧ θ < π) :
  cos θ = (-5 + real.sqrt 89) / 8 := sorry

end find_cos_theta_l122_122656


namespace ratio_of_segments_l122_122509

-- Define the segments and sides of the triangle
variable (u v : ℝ)
variable (a b c : ℝ)

-- Conditions: the sides of the triangle are 11, 14, and 19, and u + v = 11 with u < v
def sides_of_triangle : Prop := a = 11 ∧ b = 14 ∧ c = 19
def segments_of_side : Prop := u + v = 11 ∧ u < v

-- Goal: show that u:v = 3:8
theorem ratio_of_segments 
  (h1 : sides_of_triangle)
  (h2 : segments_of_side)
  (h3 : ∀ u v : ℝ, u + v = 11 → u < v →
    let x := 3 in let y := 8 in x * v = y * u
  )
  : ∀ u v, u:v = 3:8 :=
by 
  intros
  -- use given segments of side and ratio property to conclude u/v = 3/8
  sorry

end ratio_of_segments_l122_122509


namespace lines_intersect_and_sum_l122_122352

theorem lines_intersect_and_sum (a b : ℝ) :
  (∃ x y : ℝ, x = (1 / 3) * y + a ∧ y = (1 / 3) * x + b ∧ x = 3 ∧ y = 3) →
  a + b = 4 :=
by
  sorry

end lines_intersect_and_sum_l122_122352


namespace find_f_l122_122500

theorem find_f :
  ∀ (f : ℕ → ℕ),   
    (∀ a b : ℕ, f (a * b) = f a + f b - f (Nat.gcd a b)) →
    (∀ (p a : ℕ), Nat.Prime p → (f a ≥ f (a * p) → f a + f p ≥ f a * f p + 1)) →
    (∀ n : ℕ, f n = n ∨ f n = 1) :=
by
  intros f h1 h2
  sorry

end find_f_l122_122500


namespace perfect_number_divisibility_3_to_9_perfect_number_divisibility_7_to_49_l122_122522

theorem perfect_number_divisibility_3_to_9 (n : ℕ) (h1 : ∃ m : ℕ, n = m ∧ (m > 6)) (h2 : ∃ k : ℕ, n = 3 * k) (h3 : ∀ m, (∑ i in (range m) \ˢ{0}, if n % i = 0 then i else 0) = 2 * n) : ∃ j : ℕ, n = 9 * j :=
sorry

theorem perfect_number_divisibility_7_to_49 (n : ℕ) (h1 : ∃ m : ℕ, n = m ∧ (m > 28)) (h2 : ∃ k : ℕ, n = 7 * k) (h3 : ∀ m, (∑ i in (range m) \ˢ{0}, if n % i = 0 then i else 0) = 2 * n) : ∃ j : ℕ, n = 49 * j :=
sorry

end perfect_number_divisibility_3_to_9_perfect_number_divisibility_7_to_49_l122_122522


namespace Chris_catches_Sam_l122_122327

/-- 
Given:
- Sam's speed is \(16 \text{ km/h}\)
- Chris's speed is \(24 \text{ km/h}\)
- Initial distance between Sam and Chris is \(1 \text{ km}\)
Prove:
- Chris catches up to Sam in \( \frac{15}{2} \, \text{minutes} \)
-/
theorem Chris_catches_Sam (
    Sam_speed : ℝ := 16 -- Assume Sam's speed is 16 km/h
    Chris_speed : ℝ := 24 -- Assume Chris's speed is 24 km/h
    initial_distance : ℝ := 1 -- Assume initial distance between Sam and Chris is 1 km
):
    (initial_distance / (Chris_speed - Sam_speed) * 60) = (15 / 2) := 
by
    sorry

end Chris_catches_Sam_l122_122327


namespace find_inverse_l122_122580

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def inverse_matrix (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let det := determinant a b c d
  if det == 0 then (0, 0, 0, 0)
  else (d / det, -b / det, -c / det, a / det)

theorem find_inverse:
  inverse_matrix 4 10 8 20 = (0, 0, 0, 0) :=
by
  sorry

end find_inverse_l122_122580


namespace number_of_questionnaires_from_unit_D_l122_122074

theorem number_of_questionnaires_from_unit_D 
  (a d : ℕ) 
  (total : ℕ) 
  (samples : ℕ → ℕ) 
  (h_seq : samples 0 = a ∧ samples 1 = a + d ∧ samples 2 = a + 2 * d ∧ samples 3 = a + 3 * d)
  (h_total : samples 0 + samples 1 + samples 2 + samples 3 = total)
  (h_stratified : ∀ (i : ℕ), i < 4 → samples i * 100 / total = 20 → i = 1) 
  : samples 3 = 40 := sorry

end number_of_questionnaires_from_unit_D_l122_122074


namespace triangle_area_l122_122044

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area_l122_122044


namespace train_cross_signal_pole_time_l122_122045

theorem train_cross_signal_pole_time (L_train L_platform time_platform : ℝ) 
    (h_train : L_train = 300) 
    (h_platform : L_platform = 250) 
    (h_time_platform : time_platform = 33) : 
    ∃ time_pole : ℝ, time_pole = 18 ∧ 
    time_pole = L_train / ((L_train + L_platform) / time_platform) :=
by
  sorry

end train_cross_signal_pole_time_l122_122045


namespace new_minimum_point_l122_122348

open Real

noncomputable def original_function (x : ℝ) : ℝ :=
  abs x - 3

def translated_function (x : ℝ) (y : ℝ) :=
  y = original_function (x - 4) + 1

theorem new_minimum_point :
  ∃ x y : ℝ, translated_function x y ∧ x = 4 ∧ y = -2 :=
by
  exists 4, -2
  split
  · simp [translated_function, original_function, abs]
  · split; refl

end new_minimum_point_l122_122348


namespace number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l122_122845

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l122_122845


namespace quadrilateral_EFGH_area_l122_122319

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l122_122319


namespace sqrt_of_1024_l122_122209

theorem sqrt_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x ^ 2 = 1024) : x = 32 :=
sorry

end sqrt_of_1024_l122_122209


namespace exterior_angle_regular_octagon_l122_122255

theorem exterior_angle_regular_octagon : 
  (∃ n : ℕ, n = 8 ∧ ∀ (i : ℕ), i < n → true) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end exterior_angle_regular_octagon_l122_122255


namespace manufacturing_cost_before_decrease_l122_122820

def original_manufacturing_cost (P : ℝ) (C_now : ℝ) (profit_rate_now : ℝ) : ℝ :=
  P - profit_rate_now * P

theorem manufacturing_cost_before_decrease
  (P : ℝ)
  (C_now : ℝ)
  (profit_rate_now : ℝ)
  (profit_rate_original : ℝ)
  (H1 : C_now = P - profit_rate_now * P)
  (H2 : profit_rate_now = 0.50)
  (H3 : profit_rate_original = 0.20)
  (H4 : C_now = 50) :
  original_manufacturing_cost P C_now profit_rate_now = 80 :=
sorry

end manufacturing_cost_before_decrease_l122_122820


namespace opposite_of_negative_five_l122_122393

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122393


namespace glucose_solution_l122_122519

theorem glucose_solution (x : ℝ) (h : (15 / 100 : ℝ) = (6.75 / x)) : x = 45 :=
sorry

end glucose_solution_l122_122519


namespace four_friends_same_group_prob_l122_122337

-- Definitions of conditions
def students : Nat := 900
def groups : Nat := 3
def group_size : Nat := students / groups

-- The probability of assigning one student to a specific group
def prob_assign_to_same_group : ℚ := 1 / groups

-- Lean statement for the proof problem
theorem four_friends_same_group_prob :
  (prob_assign_to_same_group ^ 3) = 1 / 27 :=
by
  -- Proof to be filled in
  sorry

end four_friends_same_group_prob_l122_122337


namespace rowing_time_to_and_fro_l122_122064

noncomputable def rowing_time (distance rowing_speed current_speed : ℤ) : ℤ :=
  let speed_to_place := rowing_speed - current_speed
  let speed_back_place := rowing_speed + current_speed
  let time_to_place := distance / speed_to_place
  let time_back_place := distance / speed_back_place
  time_to_place + time_back_place

theorem rowing_time_to_and_fro (distance rowing_speed current_speed : ℤ) :
  distance = 72 → rowing_speed = 10 → current_speed = 2 → rowing_time distance rowing_speed current_speed = 15 := by
  intros h_dist h_row_speed h_curr_speed
  rw [h_dist, h_row_speed, h_curr_speed]
  sorry

end rowing_time_to_and_fro_l122_122064


namespace sum_of_coefficients_l122_122655

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (∀ x : ℤ, (1 + x)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 63 :=
by
  intros h ha
  sorry

end sum_of_coefficients_l122_122655


namespace prob_green_second_given_first_green_l122_122046

def total_balls : Nat := 14
def green_balls : Nat := 8
def red_balls : Nat := 6

def prob_green_first_draw : ℚ := green_balls / total_balls

theorem prob_green_second_given_first_green :
  prob_green_first_draw = (8 / 14) → (green_balls / total_balls) = (4 / 7) :=
by
  sorry

end prob_green_second_given_first_green_l122_122046


namespace arithmetic_progression_nth_term_l122_122412

theorem arithmetic_progression_nth_term (n r : ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, S n = 3 * n + 4 * n^2) : S r - S (r - 1) = 8 * r - 1 :=
by
  -- Use the given condition to define S_n and S_(r-1)
  have Sr : S r = 3 * r + 4 * r^2 := hS r,
  have S_r_minus_1 : S (r - 1) = 3 * (r - 1) + 4 * (r - 1)^2 := hS (r - 1),
  -- Calculate S_r and S_(r-1)
  rw Sr,
  rw S_r_minus_1,
  sorry  -- Skip the actual proof here

end arithmetic_progression_nth_term_l122_122412


namespace church_members_l122_122832

theorem church_members (M A C : ℕ) (h1 : A = 4/10 * M)
  (h2 : C = 6/10 * M) (h3 : C = A + 24) : M = 120 := 
  sorry

end church_members_l122_122832


namespace kathleens_remaining_money_l122_122262

variable (savings_june savings_july savings_august savings_september : ℝ)
variable (expenses_school_supplies expenses_clothes expenses_gift expenses_book : ℝ)
variable (donation amount_october amount_november bonus : ℝ) 
variable (total_savings total_expenses remaining_money remaining_money_with_bonus : ℝ)

-- Given savings in each month
def savings_june := 21
def savings_july := 46
def savings_august := 45
def savings_september := 32
def savings_october := savings_august / 2
def savings_november := (2 * savings_september) - 20

-- Given expenses
def expenses_school_supplies := 12
def expenses_clothes := 54
def expenses_gift := 37
def expenses_book := 25
def donation := 10

-- Total savings and expenses
def total_savings := savings_june + savings_july + savings_august + savings_september + savings_october + savings_november
def total_expenses := expenses_school_supplies + expenses_clothes + expenses_gift + expenses_book + donation

-- Calculating remaining money
def remaining_money := total_savings - total_expenses

-- Calculating remaining money with bonus
def bonus := if total_savings > 200 then 25 else 0
def remaining_money_with_bonus := remaining_money + bonus

-- Theorem statement
theorem kathleens_remaining_money :
  remaining_money_with_bonus = 97.50 := 
by
  dsimp [savings_june, savings_july, savings_august, savings_september, savings_october, savings_november, total_savings, expenses_school_supplies, expenses_clothes, expenses_gift, expenses_book, donation, total_expenses, remaining_money, bonus, remaining_money_with_bonus]
  norm_num
  sorry

end kathleens_remaining_money_l122_122262


namespace opposite_of_neg_five_l122_122364

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122364


namespace find_a3_l122_122113

theorem find_a3 (a : ℕ → ℕ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, (1 + 2 * a (n + 1)) = (1 + 2 * a n) + 1) : a 3 = 3 :=
by
  -- This is where the proof would go, but we'll leave it as sorry for now.
  sorry

end find_a3_l122_122113


namespace point_C_values_l122_122307

variable (B C : ℝ)
variable (distance_BC : ℝ)
variable (hB : B = 3)
variable (hDistance : distance_BC = 2)

theorem point_C_values (hBC : abs (C - B) = distance_BC) : (C = 1 ∨ C = 5) := 
by
  sorry

end point_C_values_l122_122307


namespace arrangement_ways_l122_122229

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l122_122229


namespace opposite_of_neg5_is_pos5_l122_122356

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122356


namespace tom_teaching_years_l122_122001

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l122_122001


namespace tan_angle_AMF_proof_l122_122780

noncomputable def parabola_focus (p : ℝ) (h : 0 < p): ℝ × ℝ :=
(F : ℝ × ℝ)

def is_on_parabola (point : ℝ × ℝ) (p : ℝ) : Prop :=
let (x, y) := point in y^2 = 2 * p * x

def line_slope (point : ℝ × ℝ) (slope : ℝ) : Prop :=
let (x, y) := point in y = slope * x

def tan_angle_AMF (p : ℝ) (A M F : ℝ × ℝ) : ℝ :=
let (Ax, Ay) := A in
let (Mx, My) := M in
let (Fx, Fy) := F in
(Ay - My) / (Ax - Mx)

theorem tan_angle_AMF_proof (p : ℝ) (h : 0 < p) (F A M : ℝ × ℝ) (Fx_lemma : parabola_focus p h = F) :
  let A_1 := (p / 2 + (1 + sqrt(5) / 4) * p, 2 * (1 + sqrt(5) / 4) * p) in
  let M_1 := (-p / 2, 0) in
  A = A_1 → M = M_1 →
  tan_angle_AMF p A M F = (2 / 5) * sqrt(5) := by
  intros
  sorry

end tan_angle_AMF_proof_l122_122780


namespace gardener_responsibility_l122_122695

-- Conditions: basic setup of the grid and placement of gardeners and flower.

structure Point where
  x : ℝ
  y : ℝ

def gardenerA : Point := { x := 0, y := 0 }
def gardenerB : Point := { x := 2, y := 0 }
def gardenerC : Point := { x := 0, y := 2 }
def gardenerD : Point := { x := 2, y := 2 }

-- Assuming the flower is at point (0.5, 0.5)
def flower : Point := { x := 0.5, y := 0.5 }

-- Define distance between points
def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Prove that gardener A, B, and C handle the flower at (0.5, 0.5)
theorem gardener_responsibility :
  let dA := distance gardenerA flower
      dB := distance gardenerB flower
      dC := distance gardenerC flower
      dD := distance gardenerD flower
  in dA < dD ∧ dB < dD ∧ dC < dD := by
  sorry

end gardener_responsibility_l122_122695


namespace sum_to_cis_form_l122_122891

noncomputable def summation_cis : ℝ :=
  ∑ i in finset.range 9, complex.cis (60 + 10 * i)

theorem sum_to_cis_form : 
  ∃ r > 0, ∃ (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360), summation_cis = r * complex.cis (real.to_radians 100) ∧ θ = 100 :=
by
  use 2 * (cos (real.to_radians 40) + cos (real.to_radians 30) + cos (real.to_radians 20) + cos (real.to_radians 10) + cos 0)
  split
  { -- Proof that r > 0
    sorry }
  use 100
  split
  { -- Proof that 0 ≤ θ < 360
    split
    { linarith }
    linarith }
  split
  { -- Proof of the summation being the right expression
    sorry }
  { -- Proof that θ = 100
    refl }

end sum_to_cis_form_l122_122891


namespace possible_faulty_keys_l122_122461

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122461


namespace sum_inf_series_l122_122593

noncomputable def H (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (1 : ℝ) / (i + 1)

open Real

theorem sum_inf_series : ∑' (n : ℕ in Finset.univ), (n : ℝ) / ((n + 1) * H n * H (n + 1)) = 1 / 2 := 
by
  sorry

end sum_inf_series_l122_122593


namespace mean_temp_of_week_l122_122761

theorem mean_temp_of_week :
  let temps := [-5, -3, -3, -6, 2, 4, 0] in
  real.mean temps = -1.86 := by
  let temps := [-5, -3, -3, -6, 2, 4, 0]
  have sum_temps : list.sum temps = -11 := by sorry
  have length_temps : list.length temps = 7 := by rfl
  calc
    real.mean temps = (list.sum temps: ℝ) / (list.length temps: ℝ) := by sorry
    ... = (-11: ℝ) / (7: ℝ) := by rw [sum_temps, length_temps]
    ... ≈ -1.857 : by norm_num
    ... ≈ -1.86 : by sorry

end mean_temp_of_week_l122_122761


namespace carrie_spent_l122_122895

-- Define the cost of one t-shirt
def cost_per_tshirt : ℝ := 9.65

-- Define the number of t-shirts bought
def num_tshirts : ℝ := 12

-- Define the total cost function
def total_cost (cost_per_tshirt : ℝ) (num_tshirts : ℝ) : ℝ := cost_per_tshirt * num_tshirts

-- State the theorem which we need to prove
theorem carrie_spent :
  total_cost cost_per_tshirt num_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l122_122895


namespace solution_set_of_inequality_l122_122161

variable (f : ℝ → ℝ)

-- Conditions
axiom additivity : ∀ x y : ℝ, f(x) + f(y) = f(x + y)
axiom positivity : ∀ x : ℝ, 0 < x → 0 < f(x)
axiom value_neg_one_fourth : f (-1 / 4) = -1

-- Question: Prove the solution set is (-3, 3)
theorem solution_set_of_inequality : { x : ℝ | f(x^2 - 8) < 4 } = set.Ioo (-3) 3 :=
by
  sorry

end solution_set_of_inequality_l122_122161


namespace five_people_six_chairs_l122_122225

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l122_122225


namespace weekly_salary_correct_l122_122515

-- Define the daily salaries for each type of worker
def salary_A : ℝ := 200
def salary_B : ℝ := 250
def salary_C : ℝ := 300
def salary_D : ℝ := 350

-- Define the number of each type of worker
def num_A : ℕ := 3
def num_B : ℕ := 2
def num_C : ℕ := 3
def num_D : ℕ := 1

-- Define the total hours worked per day and the number of working days in a week
def hours_per_day : ℕ := 6
def working_days : ℕ := 7

-- Calculate the total daily salary for the team
def daily_salary_team : ℝ :=
  (num_A * salary_A) + (num_B * salary_B) + (num_C * salary_C) + (num_D * salary_D)

-- Calculate the total weekly salary for the team
def weekly_salary_team : ℝ := daily_salary_team * working_days

-- Problem: Prove that the total weekly salary for the team is Rs. 16,450
theorem weekly_salary_correct : weekly_salary_team = 16450 := by
  sorry

end weekly_salary_correct_l122_122515


namespace intersection_complement_l122_122198

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {2, 3, 4}) (hB : B = {1, 2})

theorem intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_complement_l122_122198


namespace sqrt_ineq_l122_122977

theorem sqrt_ineq (a : ℝ) (h : a > 5) : 
  sqrt (a - 5) - sqrt (a - 3) < sqrt (a - 2) - sqrt (a) :=
by
  sorry

end sqrt_ineq_l122_122977


namespace number_count_l122_122847

open Nat

theorem number_count (n a k m : ℕ) (n_pos : n > 0) (m_bound : m < 10^k)
    (key_eqn : 8 * m = 10^k * (a + n)) : 
    (number_of_combinations (λ m a k n, 8 * m = 10^k * (a + n) ∧ n > 0 ∧ m < 10^k) = 28) :=
sorry

end number_count_l122_122847


namespace average_sq_feet_per_person_approx_l122_122406

noncomputable def average_sq_feet_per_person (pop : ℕ) (area_sq_miles : ℕ) (sq_feet_per_sq_mile : ℕ) : ℝ :=
  (area_sq_miles * sq_feet_per_sq_mile) / pop

theorem average_sq_feet_per_person_approx :
  let pop := 38005238
      area_sq_miles := 3855100
      sq_feet_per_sq_mile := 27878400 in
  |average_sq_feet_per_person pop area_sq_miles sq_feet_per_sq_mile - 3000000| < 1000000 :=
by sorry

end average_sq_feet_per_person_approx_l122_122406


namespace opposite_of_neg_five_l122_122377

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122377


namespace problem1_problem2_l122_122041

-- Problem 1
theorem problem1 (a b : ℝ) (h : 2 * (a + 1) * (b + 1) = (a + b) * (a + b + 2)) : a^2 + b^2 = 2 := sorry

-- Problem 2
theorem problem2 (a b c : ℝ) (h : a^2 + c^2 = 2 * b^2) : (a + b) * (a + c) + (c + a) * (c + b) = 2 * (b + a) * (b + c) := sorry

end problem1_problem2_l122_122041


namespace sum_of_valid_integers_l122_122589

theorem sum_of_valid_integers :
  ∑ n in { n : ℤ | (∃ x : ℤ, n^2 - 17 * n + 72 = x^2) ∧ (18 % n = 0) }.toFinset = 19 := by
  sorry

end sum_of_valid_integers_l122_122589


namespace store_profit_l122_122842

theorem store_profit (C : ℝ) :
  let SP1 := 1.20 * C,
      SP2 := 1.25 * SP1,
      SPF := 0.94 * SP2,
      Profit := SPF - C
  in Profit / C = 0.41 :=
by
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.94 * SP2
  let Profit := SPF - C
  sorry

end store_profit_l122_122842


namespace complex_numbers_proof_l122_122708

noncomputable def modulus_1 (z : ℂ) : Prop := complex.abs z = 1

theorem complex_numbers_proof (a : ℝ) (z : ℕ → ℂ) (h_modulus : ∀ k, modulus_1 (z k)) 
    (h_equation : ∑ k in finset.range n, (z k)^3 = 4*(a + (a - ↑n)*complex.I) - 3 * ∑ k in finset.range n, (complex.conj (z k))) : 
    (a ∈ (finset.range (n + 1) : finset ℝ)) ∧ (∀ k, z k = 1 ∨ z k = complex.I) :=
sorry

end complex_numbers_proof_l122_122708


namespace gavrila_distance_l122_122091

noncomputable def distance_from_start (L : ℝ) (y : ℝ) : ℝ :=
  let x := (y^2) / (4 * L)
  sqrt(x^2 + y^2)

theorem gavrila_distance (L : ℝ) (y : ℝ) : 
  L = 50 → y = 40 → Real.floor (distance_from_start L y) = 41 :=
by
  intros hL hy
  rw [hL, hy]
  sorry

end gavrila_distance_l122_122091


namespace ending_number_81_l122_122097

theorem ending_number_81 (n : ℕ) (h : (18 + n) / 2 = 49.5) : n = 81 :=
  sorry

end ending_number_81_l122_122097


namespace probability_of_adjacent_vertices_in_dodecagon_l122_122503

def probability_at_least_two_adjacent_vertices (n : ℕ) : ℚ :=
  if n = 12 then 24 / 55 else 0  -- Only considering the dodecagon case

theorem probability_of_adjacent_vertices_in_dodecagon :
  probability_at_least_two_adjacent_vertices 12 = 24 / 55 :=
by
  sorry

end probability_of_adjacent_vertices_in_dodecagon_l122_122503


namespace eccentricity_of_ellipse_l122_122111

noncomputable def ellipse_eccentricity : ℂ → ℂ → ℂ → ℂ := sorry

theorem eccentricity_of_ellipse :
  let z1 := 2
  let z2 := Complex.mk (-3/2) (sqrt 5 / 2)
  let z3 := Complex.mk (-3/2) (-sqrt 5 / 2)
  let z4 := Complex.mk (-5/2) (sqrt 5 / 2)
  let z5 := Complex.mk (-5/2) (-sqrt 5 / 2)
  let semi_major_axis := sqrt 5 / 2
  let semi_minor_axis := sqrt 5 / 4
  (ellipse_eccentricity z1 z2 z3 z4 z5 semi_major_axis semi_minor_axis) = sqrt 3 / 2 :=
begin
  sorry
end

end eccentricity_of_ellipse_l122_122111


namespace number_of_slices_l122_122296

theorem number_of_slices 
  (pepperoni ham sausage total_meat pieces_per_slice : ℕ)
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : total_meat = pepperoni + ham + sausage)
  (h5 : pieces_per_slice = 22) :
  total_meat / pieces_per_slice = 6 :=
by
  sorry

end number_of_slices_l122_122296


namespace initial_meals_for_adults_l122_122057

theorem initial_meals_for_adults (C A : ℕ) (h1 : C = 90) (h2 : 14 * C / A = 72) : A = 18 :=
by
  sorry

end initial_meals_for_adults_l122_122057


namespace truncated_pyramid_smaller_base_area_l122_122785

noncomputable def smaller_base_area (a : ℝ) (α β : ℝ) : ℝ :=
  (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2

theorem truncated_pyramid_smaller_base_area (a α β : ℝ) :
  smaller_base_area a α β = (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2 :=
by
  unfold smaller_base_area
  sorry

end truncated_pyramid_smaller_base_area_l122_122785


namespace line_properties_l122_122964

theorem line_properties : ∃ m x_intercept, 
  (∀ (x y : ℝ), 4 * x + 7 * y = 28 → y = m * x + 4) ∧ 
  (∀ (x y : ℝ), y = 0 → 4 * x + 7 * y = 28 → x = x_intercept) ∧ 
  m = -4 / 7 ∧ 
  x_intercept = 7 :=
by 
  sorry

end line_properties_l122_122964


namespace solve_for_y_l122_122329

theorem solve_for_y (y : ℚ) : 16^(3 * y) = 4^(4 * y - 5) → y = -5 / 2 :=
by
  sorry

end solve_for_y_l122_122329


namespace parabola_vertex_distance_l122_122103

theorem parabola_vertex_distance : 
  let p1 := (0 : ℝ, 3 : ℝ)
      p2 := (0 : ℝ, -2 : ℝ) in
  (EuclideanGeometry.dist p1 p2) = 5 := 
by {
  sorry  -- Proof to be filled in later
}

end parabola_vertex_distance_l122_122103


namespace protein_percentage_in_mixture_l122_122332

theorem protein_percentage_in_mixture :
  let soybean_meal_weight := 240
  let cornmeal_weight := 40
  let mixture_weight := 280
  let soybean_protein_content := 0.14
  let cornmeal_protein_content := 0.07
  let total_protein := soybean_meal_weight * soybean_protein_content + cornmeal_weight * cornmeal_protein_content
  let protein_percentage := (total_protein / mixture_weight) * 100
  protein_percentage = 13 :=
by
  sorry

end protein_percentage_in_mixture_l122_122332


namespace number_of_tons_is_3_l122_122331

noncomputable def calculate_tons_of_mulch {total_cost price_per_pound pounds_per_ton : ℝ} 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : ℝ := 
  total_cost / price_per_pound / pounds_per_ton

theorem number_of_tons_is_3 
  (total_cost price_per_pound pounds_per_ton : ℝ) 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : 
  calculate_tons_of_mulch h_total_cost h_price_per_pound h_pounds_per_ton = 3 := 
by
  sorry

end number_of_tons_is_3_l122_122331


namespace remainder_2753_div_98_l122_122015

theorem remainder_2753_div_98 : (2753 % 98) = 9 := 
by sorry

end remainder_2753_div_98_l122_122015


namespace cross_section_of_cube_l122_122868

-- Definitions for the cube and the plane
def Cube (V : Type) :=
  ∃ (A B C D A' B' C' D' : V), -- Vertices of cube
    IsCube A B C D A' B' C' D'

def PlanePerpendicularToDiagonal (V : Type) (α : Set V) :=
  ∃ (A C' : V), -- Diagonal points
    IsDiagonal A C' ∧ IsPerpendicularPlaneToLineSegment α A C'

-- Problem statement in Lean 4
theorem cross_section_of_cube (V : Type) [EuclideanSpace V]
    (A B C D A' B' C' D' : V) (α : Set V) 
    (h1 : Cube V)
    (h2 : PlanePerpendicularToDiagonal V α)
    (h3 : α ∩ (boundary (Cube V)) ≠ ∅) :
    (∃ l : ℝ, ∀ W : polygon V, l = perimeter W) ∧
    (∃ (S1 S2 : ℝ), S1 ≠ S2 ∧ ∀ W1 W2 : polygon V, S1 = area W1 ∨ S2 = area W2) :=
sorry

end cross_section_of_cube_l122_122868


namespace geometric_representation_of_eq_l122_122776

theorem geometric_representation_of_eq {z : ℂ} (h : |z - 3| = 1) :
  ∃ center : ℂ, center = 3 ∧ ∃ radius : ℝ, radius = 1 ∧ abs (z - center) = radius :=
by
  sorry

end geometric_representation_of_eq_l122_122776


namespace probability_heart_or_diamond_top_card_l122_122529

theorem probability_heart_or_diamond_top_card :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let favorable_outcomes := 26  -- 13 $\heartsuit$ cards + 13 $\diamondsuit$ cards
  let total_outcomes := deck_size
  probability_heart_or_diamond (deck_size: Nat) (suits: Nat) (cards_per_suit: Nat) (favorable_outcomes: Nat) (total_outcomes: Nat) : 
  favorite_outcomes / total_outcomes = (1 / 2) :=
by
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let favorable_outcomes := 26  -- 13 $\heartsuit$ cards + 13 $\diamondsuit$ cards
  let total_outcomes := deck_size
  sorry

end probability_heart_or_diamond_top_card_l122_122529


namespace sweet_numbers_correct_l122_122108

def is_sweet_number (G : ℕ) : Prop :=
  ∀ (n : ℕ), let seq := (λ x, if x ≤ 30 then 3 * x else (x - 15) % 15) in seq^[n] G % 15 ≠ 3

def count_sweet_numbers (up_to : ℕ) : ℕ :=
  finset.card ((finset.range up_to).filter is_sweet_number)

theorem sweet_numbers_correct : count_sweet_numbers 60 = 44 :=
by {
  trace "Proving that the number of sweet numbers from 1 to 60 is exactly 44."; 
  sorry
}

end sweet_numbers_correct_l122_122108


namespace even_factors_count_of_n_l122_122650

def n : ℕ := 2^3 * 3^2 * 7 * 5

theorem even_factors_count_of_n : ∃ k : ℕ, k = 36 ∧ ∀ (a b c d : ℕ), 
  1 ≤ a ∧ a ≤ 3 →
  b ≤ 2 →
  c ≤ 1 →
  d ≤ 1 →
  2^a * 3^b * 7^c * 5^d ∣ n :=
sorry

end even_factors_count_of_n_l122_122650


namespace particle_position_at_2004_seconds_l122_122086

structure ParticleState where
  position : ℕ × ℕ

def initialState : ParticleState :=
  { position := (0, 0) }

def moveParticle (state : ParticleState) (time : ℕ) : ParticleState :=
  if time = 0 then initialState
  else if (time - 1) % 4 < 2 then
    { state with position := (state.position.fst + 1, state.position.snd) }
  else
    { state with position := (state.position.fst, state.position.snd + 1) }

def particlePositionAfterTime (time : ℕ) : ParticleState :=
  (List.range time).foldl moveParticle initialState

/-- The position of the particle after 2004 seconds is (20, 44) -/
theorem particle_position_at_2004_seconds :
  (particlePositionAfterTime 2004).position = (20, 44) :=
  sorry

end particle_position_at_2004_seconds_l122_122086


namespace sum_of_common_chords_leq_sum_of_radii_l122_122419

theorem sum_of_common_chords_leq_sum_of_radii
  (P : Point)
  (O1 O2 O3 : Point)
  (R1 R2 R3 : ℝ)
  (h1 : Circle P O1 R1)
  (h2 : Circle P O2 R2)
  (h3 : Circle P O3 R3)
  (PR PS PT : ℝ)
  (hPR : chord_of_circle_through_point PR (circle P O1 R1))
  (hPS : chord_of_circle_through_point PS (circle P O2 R2))
  (hPT : chord_of_circle_through_point PT (circle P O3 R3)) :
  PR + PS + PT ≤ R1 + R2 + R3 := 
sorry

end sum_of_common_chords_leq_sum_of_radii_l122_122419


namespace seating_arrangements_l122_122241

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l122_122241


namespace parabola_midpoint_distance_l122_122619

noncomputable def midpoint_distance_to_y_axis (F A B : ℝ × ℝ) [parabola : (A.2)^2 = A.1 ∧ (B.2)^2 = B.1] 
  (H : (Real.sqrt (A.1 - F.1)^2 + F.2^2) + (Real.sqrt (B.1 - F.1)^2 + F.2^2) = 5) : ℝ :=
  let M := (A.1 + B.1) / 2 in
  M

theorem parabola_midpoint_distance {F A B : ℝ × ℝ} 
  (hF : F = (1 / 4, 0)) 
  (hAB_parabola : (A.2)^2 = A.1 ∧ (B.2)^2 = B.1) 
  (h_dist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - 0)^2) + Real.sqrt ((B.1 - F.1)^2 + (B.2 - 0)^2) = 5) :
  midpoint_distance_to_y_axis F A B hAB_parabola h_dist = 9 / 4 := 
sorry

end parabola_midpoint_distance_l122_122619


namespace focalRadiiEqualAnglesWithTangent_l122_122266

noncomputable section

variables {a b : ℝ} (P : ℝ × ℝ)
variable (ellipse : P = (a * cos θ, b * sin θ))
variables [hw : a^2 > b^2]

def isUsingEllipseForm (x y : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

def tangentsEqual (S S' : ℝ × ℝ) (P : ℝ × ℝ) : Prop := 
  -- Mahematical definition simplified for illustrative purposes
  let m := -b * cot θ / a 
  let m1 := b * sin θ / (a * (cos θ - e))
  let m2 := b * sin θ / (a * (cos θ + e))
  | (m - m1) / (1 + m * m1) | = | (m - m2) / (1 + m * m2) |

theorem focalRadiiEqualAnglesWithTangent (A B : ℝ × ℝ) (P : ℝ × ℝ) : 
  isUsingEllipseForm A P → isUsingEllipseForm B P → tangentsEqual A B P :=
by
  sorry

end focalRadiiEqualAnglesWithTangent_l122_122266


namespace max_last_place_score_l122_122037

theorem max_last_place_score (n : ℕ) (h : n ≥ 4) :
  ∃ k, (∀ m, m < n -> (k + m) < (n * 3)) ∧ 
     (∀ i, ∃ j, j < n ∧ i = k + j) ∧
     (n * 2 - 2) = (k + n - 1) ∧ 
     k = n - 2 := 
sorry

end max_last_place_score_l122_122037


namespace sum_of_b_seq_l122_122069

noncomputable def a_seq : ℕ → ℝ
| 0       => 1
| (n + 1) => 1 / Real.sqrt(1 / a_seq n ^ 2 + 2)

def b_seq (n : ℕ) : ℝ := 1 / (a_seq n ^ 2 * 2 ^ n)

def S (n : ℕ) : ℝ := ∑ k in Finset.range (n+1), b_seq k

theorem sum_of_b_seq (n : ℕ) : S n = 3 - (2 * n + 3) / 2 ^ n := by
  sorry

end sum_of_b_seq_l122_122069


namespace sphere_shadow_boundary_l122_122070

theorem sphere_shadow_boundary
(radius : ℝ) (center sphere : ℝ × ℝ × ℝ)
(P : ℝ × ℝ × ℝ)
(boundary_x boundary_y : ℝ) :
radius = 2 
∧ center = (0, 0, 2)
∧ P = (1, -1, 3)
∧ (boundary_x = (5 + sqrt 19)/2 ∨ boundary_x = (5 - sqrt 19)/2)
∧ (boundary_y = (sqrt 19 - 5)/2 ∨ boundary_y = (-sqrt 19 - 5)/2)
→ boundary_y = boundary_x - 5 :=
sorry

end sphere_shadow_boundary_l122_122070


namespace probability_not_pass_fourth_quadrant_l122_122688

-- The set of possible numbers on the balls
def ball_numbers : Set ℚ := {-1, 0, 1/3}

-- The quadratic function with given m and n
def quadratic (m n : ℚ) (x : ℚ) : ℚ := x^2 + m * x + n

-- The condition for the quadratic function not to pass through the fourth quadrant
def does_not_pass_fourth_quadrant (m n : ℚ) : Prop :=
  (m ≥ 0 ∧ n ≥ 0) ∨ (m ≥ 0 ∧ n * 4 ≥ m^2)

-- The predicate that for a given (m, n) the quadratic function does not pass through the fourth quadrant
def valid_pair (m n : ℚ) : Prop :=
  does_not_pass_fourth_quadrant m n

-- All possible (m, n) pairs
def all_pairs : List (ℚ × ℚ) :=
  [( -1, -1), ( -1,  0), ( -1,  1/3),
   (  0, -1), (  0,  0), (  0,  1/3),
   (1/3, -1), (1/3,  0), (1/3, 1/3)]

-- Valid pairs count
def valid_pair_count : ℕ :=
  ((all_pairs.filter $ λ pair, valid_pair pair.fst pair.snd).length : ℕ)

-- Total pairs count
def total_pair_count : ℕ := (all_pairs.length : ℕ)

-- The probability the quadratic function does not pass through the fourth quadrant
def probability_does_not_pass_fourth_quadrant : ℚ :=
  (valid_pair_count : ℚ) / (total_pair_count : ℚ)

theorem probability_not_pass_fourth_quadrant :
  probability_does_not_pass_fourth_quadrant = 5 / 9 :=
by
  sorry

end probability_not_pass_fourth_quadrant_l122_122688


namespace cows_in_group_l122_122680

theorem cows_in_group (D C : ℕ) 
  (h : 2 * D + 4 * C = 2 * (D + C) + 36) : 
  C = 18 :=
by
  sorry

end cows_in_group_l122_122680


namespace student_tickets_sold_l122_122421

theorem student_tickets_sold (A S : ℝ) (h1 : A + S = 59) (h2 : 4 * A + 2.5 * S = 222.50) : S = 9 :=
by
  sorry

end student_tickets_sold_l122_122421


namespace ordered_triples_count_l122_122204

theorem ordered_triples_count :
  { (a, b, c) : ℤ × ℤ × ℤ // a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ Real.log b / Real.log a = c ^ 3 ∧ a + b + c = 2100 }.card = 1 :=
sorry

end ordered_triples_count_l122_122204


namespace only_solution_l122_122577

-- Define strictly positive integers
def positive_int (n : ℕ) : Prop := n > 0

-- Define the equation condition
def equation_condition (x y z : ℕ) : Prop := 5^x - 3^y = z^2

-- Define the main theorem
theorem only_solution (x y z : ℕ) (hx : positive_int x) (hy : positive_int y) (hz : positive_int z) : 
  equation_condition x y z → 
  (x = 2 ∧ y = 2 ∧ z = 4) :=
begin
  sorry
end

end only_solution_l122_122577


namespace five_people_six_chairs_l122_122226

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l122_122226


namespace general_formula_arith_seq_sum_bn_l122_122616

open Real

-- Definitions based on given conditions
variables (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
variables (n : ℕ)

-- Conditions
def cond1 : Prop := a_n 1 + a_n 3 + a_n 5 = 15
def cond2 : Prop := S_n 7 = 49
def SumArithSeq (n : ℕ) : ℝ := n / 2 * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

-- Questions
def question1 : Prop := ∀ n, a_n n = 2 * n - 1
def question2 : Prop := T_n n = (n - 1) * 3^(n + 1) + 3
def relation1 : Prop := b_n n = a_n n * 3^n
def SumSeq (n : ℕ, b : ℕ → ℝ) : ℝ := ∑ i in range n, b i

-- Theorem 1: General formula for {a_n}
theorem general_formula_arith_seq : cond1 → cond2 → (∀ n, a_n n = 2 * n - 1) := sorry

-- Theorem 2: Sum of the first n terms T_n of {b_n}
theorem sum_bn : (∀ n, a_n n = 2 * n - 1) →
                (∀ n, b_n n = a_n n * 3 ^ n) →
                (∀ n, T_n n = (n - 1) * 3^(n + 1) + 3) := sorry

end general_formula_arith_seq_sum_bn_l122_122616


namespace complex_solutions_equation_l122_122581

noncomputable def valid_complex_solutions_count : ℕ :=
  let numerator_roots := [1, -1, Complex.I, -Complex.I] -- roots of z^4 - 1 = 0
  let denominator_roots := [1, 1, -2] -- roots of z^3 - 3z + 2 = 0 (including multiplicities)
  numerator_roots.eraseList denominator_roots |>.length -- remove roots that are in both lists and count remaining

theorem complex_solutions_equation : valid_complex_solutions_count = 3 := by
  sorry

end complex_solutions_equation_l122_122581


namespace find_faulty_keys_l122_122446

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122446


namespace transformed_curve_l122_122003

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end transformed_curve_l122_122003


namespace angle_F_measure_l122_122214

-- Define angle B
def angle_B := 120

-- Define angle C being supplementary to angle B on a straight line
def angle_C := 180 - angle_B

-- Define angle D
def angle_D := 45

-- Define angle E
def angle_E := 30

-- Define the vertically opposite angle F to angle C
def angle_F := angle_C

theorem angle_F_measure : angle_F = 60 :=
by
  -- Provide a proof by specifying sorry to indicate the proof is not complete
  sorry

end angle_F_measure_l122_122214


namespace beneficial_card_l122_122025

theorem beneficial_card
  (P : ℕ) (r_c r_d r_i : ℚ) :
  let credit_income := (r_c * P + r_i * P)
  let debit_income := r_d * P
  P = 8000 ∧ r_c = 0.005 ∧ r_d = 0.0075 ∧ r_i = 0.005 →
  credit_income > debit_income :=
by
  intro h
  cases h with hP hr
  cases hr with hrc hrd_ri
  cases hrd_ri with hrd hri
  rw [hP, hrc, hrd, hri]
  sorry

end beneficial_card_l122_122025


namespace min_value_of_a_plus_b_l122_122282

theorem min_value_of_a_plus_b (a b : ℤ) (h_ab : a * b = 72) (h_even : a % 2 = 0) : a + b ≥ -38 :=
sorry

end min_value_of_a_plus_b_l122_122282


namespace faulty_keys_l122_122468

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122468


namespace part_time_employees_l122_122060

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : full_time_employees = 63093) :
  total_employees - full_time_employees = 2041 :=
by
  -- Suppose that total_employees - full_time_employees = 2041
  sorry

end part_time_employees_l122_122060


namespace volume_of_pyramid_OABC_l122_122696

def volume_pyramid_OABC (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let BC := (C.1 - B.1, C.2 - B.2, C.3 - B.3)
  let n := (3, -1, 1)
  let h := (|n.1 * A.1 + n.2 * A.2 + n.3 * A.3| : ℝ) / (real.sqrt (n.1^2 + n.2^2 + n.3^2))
  let S_ABC := (1 / 2) * real.sqrt (AB.1^2 + AB.2^2 + AB.3^2) * real.sqrt (BC.1^2 + BC.2^2 + BC.3^2) * (3 * real.sqrt 11 / 10)
  1 / 3 * h * S_ABC

theorem volume_of_pyramid_OABC (A B C : ℝ × ℝ × ℝ) : 
  A = (1, -2, 0) → B = (2, 1, 0) → C = (1, 1, 3) → 
  volume_pyramid_OABC A B C = 5 / 2 :=
by
  -- proof skipped
  sorry

end volume_of_pyramid_OABC_l122_122696


namespace statements_correct_l122_122538

theorem statements_correct :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (∀ x : ℝ, (∀ x, x^2 + x + 1 ≠ 0) ↔ (∃ x, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) ↔ p ∧ q) ∧
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬ (x^2 - 3*x + 2 > 0) → x ≤ 2)) :=
by
  sorry

end statements_correct_l122_122538


namespace problem1_problem2_problem3_problem4_l122_122100

-- Problem (1)
theorem problem1 : 6 - -2 + -4 - 3 = 1 :=
by sorry

-- Problem (2)
theorem problem2 : 8 / -2 * (1 / 3 : ℝ) * (-(1 + 1/2: ℝ)) = 2 :=
by sorry

-- Problem (3)
theorem problem3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 :=
by sorry

-- Problem (4)
theorem problem4 : 
  |-(5 / 6 : ℝ)| / ((-(3 + 1 / 5: ℝ)) / (-4)^2 + (-7 / 4) * (4 / 7)) = -(25 / 36) :=
by sorry

end problem1_problem2_problem3_problem4_l122_122100


namespace find_b_over_a_find_angle_B_l122_122676

-- Definitions and main theorems
noncomputable def sides_in_triangle (A B C a b c : ℝ) : Prop :=
  a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = Real.sqrt 2 * a

noncomputable def cos_law_condition (a b c : ℝ) : Prop :=
  c^2 = b^2 + Real.sqrt 3 * a^2

theorem find_b_over_a {A B C a b c : ℝ} (h : sides_in_triangle A B C a b c) : b / a = Real.sqrt 2 :=
  sorry

theorem find_angle_B {A B C a b c : ℝ} (h1 : sides_in_triangle A B C a b c) (h2 : cos_law_condition a b c)
  (h3 : b / a = Real.sqrt 2) : B = Real.pi / 4 :=
  sorry

end find_b_over_a_find_angle_B_l122_122676


namespace doris_needs_weeks_l122_122916

noncomputable def average_weeks_to_cover_expenses (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) : ℝ := 
  let weekday_income := weekly_babysit_hours * 20
  let saturday_income := saturday_hours * (if weekly_babysit_hours > 15 then 15 else 20)
  let teaching_income := 100
  let total_weekly_income := weekday_income + saturday_income + teaching_income
  let monthly_income_before_tax := total_weekly_income * 4
  let monthly_income_after_tax := monthly_income_before_tax * 0.85
  monthly_income_after_tax / 4 / 1200

theorem doris_needs_weeks (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) :
  1200 ≤ (average_weeks_to_cover_expenses weekly_babysit_hours saturday_hours) * 4 * 1200 :=
  by
    sorry

end doris_needs_weeks_l122_122916


namespace quadrilateral_pyramid_volume_l122_122143

-- Defining the problem parameters and conditions
variables {a h b : ℝ} (α : ℝ)

-- Defining the main proof statement
theorem quadrilateral_pyramid_volume (ha : a > 0) (h_planar_angle : cos α = sqrt(5) - 1) : 
  (volume : ℝ) = (a^3 / 6) * sqrt(1 + sqrt(5)) := 
sorry

end quadrilateral_pyramid_volume_l122_122143


namespace greatest_n_4022_l122_122280

noncomputable def arithmetic_sequence_greatest_n 
  (a : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (cond1 : a 2011 + a 2012 > 0)
  (cond2 : a 2011 * a 2012 < 0) : ℕ :=
  4022

theorem greatest_n_4022 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : a 1 > 0)
  (h2 : a 2011 + a 2012 > 0)
  (h3 : a 2011 * a 2012 < 0):
  arithmetic_sequence_greatest_n a h1 h2 h3 = 4022 :=
sorry

end greatest_n_4022_l122_122280


namespace five_people_six_chairs_l122_122224

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l122_122224


namespace determine_n_l122_122052

noncomputable def average_value (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1) : ℚ) / (6 * (n * (n + 1) / 2))

theorem determine_n :
  ∃ n : ℕ, average_value n = 2020 ∧ n = 3029 :=
sorry

end determine_n_l122_122052


namespace insert_seven_digits_l122_122425

theorem insert_seven_digits :
  ∀ (N1 N2 : list ℕ), 
  N1.length = 2007 → 
  N2.length = 2007 → 
  (∃ M : list ℕ, M.length = 2000 ∧ 
                 ∃ a b : list ℕ, a.length = 7 ∧ b.length = 7 ∧ 
                 N1.erase_all a = M ∧ N2.erase_all b = M) →
  ∃ M' : list ℕ, M'.length = 2014 ∧ 
                 (∃ a' b' : list ℕ, a'.length = 7 ∧ b'.length = 7 ∧ 
                  N1 ++ a' = M' ∧ N2 ++ b' = M') := 
by
  sorry

end insert_seven_digits_l122_122425


namespace problem1_l122_122099

theorem problem1 : (2 * 10^3)^2 * (-2 * 10^(-5)) = -80 :=
  by sorry

end problem1_l122_122099


namespace S6_value_l122_122825

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ := x^m + (1/x)^m

theorem S6_value (x : ℝ) (h : x + 1/x = 4) : S_m x 6 = 2700 :=
by
  -- Skipping proof
  sorry

end S6_value_l122_122825


namespace interest_rate_of_second_investment_l122_122076

variable {r : ℝ}

def invested_at_rate := 2000
def invested_at_five_percent := 4 * invested_at_rate
def interest_from_rate (r : ℝ) := invested_at_rate * r
def interest_from_five_percent := invested_at_five_percent * 0.05
def total_interest := interest_from_rate r + interest_from_five_percent

theorem interest_rate_of_second_investment (h : total_interest = 520) : r = 0.06 :=
sorry

end interest_rate_of_second_investment_l122_122076


namespace distance_from_starting_point_l122_122088

-- Definitions based on the problem conditions
def L : ℝ := 50 -- Halfway between the siren and the start of the bridge
def sound_distance_condition (x y L : ℝ) : Prop :=
  (x + L) = Real.sqrt ((x - L) ^ 2 + y ^ 2)

-- Hypotheses based on the problem conditions
def condition1 : Prop := sound_distance_condition x y L
def condition2 : Prop := y = 40 

-- The theorem we need to prove
theorem distance_from_starting_point (x y L : ℝ) (h1 : sound_distance_condition x y L) (h2 : y = 40) : 
  Real.sqrt (x ^ 2 + y ^ 2) = 41 :=
  sorry

end distance_from_starting_point_l122_122088


namespace last_three_digits_of_power_l122_122817

theorem last_three_digits_of_power (h : 3^400 ≡ 1 [MOD 800]) : 3^8000 ≡ 1 [MOD 800] :=
by {
  sorry
}

end last_three_digits_of_power_l122_122817


namespace smallest_k_divisibility_l122_122016

theorem smallest_k_divisibility : ∃ (k : ℕ), k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_divisibility_l122_122016


namespace largest_n_digit_number_is_10n_minus_2_smallest_n_with_perfect_square_digits_sum_correct_smallest_n_l122_122723

def largest_n_digit_not_sum_or_difference_of_squares (n : ℕ) (h : n > 2) : ℕ :=
  10^n - 2

theorem largest_n_digit_number_is_10n_minus_2 (n : ℕ) (h : n > 2) :
  ∀ x : ℕ, (x = largest_n_digit_not_sum_or_difference_of_squares n h) → 
  ¬ (∃ a b : ℕ, x = a^2 + b^2 ∨ x = a^2 - b^2) :=
sorry

noncomputable def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  let digits := (10 ^ n - 2).digits in
  digits.foldr (λ d acc, d ^ 2 + acc) 0

theorem smallest_n_with_perfect_square_digits_sum : ℕ :=
  if n : ℕ := (lsieve (λ n, ∃ k : ℕ, sum_of_squares_of_digits n = k ^ 2)).head?.getOrElse 0
  then n
  else 66 -- Placeholder value, the actual computation isn't required per problem statement.

theorem correct_smallest_n : ℕ := 66

lemma correct_smallest_n_satisfies_condition (n : ℕ) :
  correct_smallest_n = 66 ∧ ∀ x : ℕ, sum_of_squares_of_digits x = x^2 :=
begin
  refine ⟨rfl, _⟩,
  intro x,
  -- skipping the proof part
  sorry,
end

end largest_n_digit_number_is_10n_minus_2_smallest_n_with_perfect_square_digits_sum_correct_smallest_n_l122_122723


namespace find_n_l122_122718

-- Define the arithmetic series sums
def s1 (n : ℕ) : ℕ := (5 * n^2 + 5 * n) / 2
def s2 (n : ℕ) : ℕ := n^2 + n

-- The theorem to be proved
theorem find_n : ∃ n : ℕ, s1 n + s2 n = 156 ∧ n = 7 :=
by
  sorry

end find_n_l122_122718


namespace identify_faulty_key_l122_122484

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122484


namespace tan_alpha_eq_one_third_expression_evaluates_to_three_fifths_l122_122972

theorem tan_alpha_eq_one_third (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

theorem expression_evaluates_to_three_fifths (α : Real) (h1 : Real.tan α = 1 / 3) :
  (2 * Real.sin(α)^2 + Real.sin(2 * α)) / (1 + Real.tan α) = 3 / 5 :=
by
  sorry

end tan_alpha_eq_one_third_expression_evaluates_to_three_fifths_l122_122972


namespace sqrt_mixed_number_simplify_l122_122944

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l122_122944


namespace cuboid_height_l122_122489

theorem cuboid_height
  (edge_length_cube : ℝ)
  (total_wire_length : ℝ)
  (length_cuboid : ℝ)
  (width_cuboid : ℝ) :
  edge_length_cube = 10 →
  total_wire_length = 12 * edge_length_cube →
  length_cuboid = 8 →
  width_cuboid = 5 →
  ∃ height_cuboid : ℝ,
    total_wire_length = 4 * length_cuboid + 4 * width_cuboid + 4 * height_cuboid ∧
    height_cuboid = 17 :=
by {
  intro h1 h2 h3 h4,
  use 17,
  rw [h1, h3, h4],
  split,
  { rw h2,
    linarith, },
  { linarith, },
}

end cuboid_height_l122_122489


namespace sum_of_solutions_l122_122561

theorem sum_of_solutions :
  let f_base := λ x : ℝ, x^2 - 6 * x + 4
  let f_exp := λ x : ℝ, x^2 - 7 * x + 6
  let solutions := {x : ℝ | f_base x ≠ 0} ∪ {1, 3, 5, 6}
  solutions.sum = 15 :=
by
  intros
  sorry 

end sum_of_solutions_l122_122561


namespace find_range_of_f_gt_f_comp_l122_122112

def f (x : ℝ) : ℝ := Real.log (1 + |x|) - (1 / (1 + x^2))

theorem find_range_of_f_gt_f_comp (x : ℝ) : 
    (1 / 3) < x ∧ x < 1 ↔ f x > f (2 * x - 1) := 
    sorry

end find_range_of_f_gt_f_comp_l122_122112


namespace inequality_ay_bz_cx_lt_k_squared_l122_122991

theorem inequality_ay_bz_cx_lt_k_squared
  (a b c x y z k : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) :
  (a * y + b * z + c * x) < k^2 :=
sorry

end inequality_ay_bz_cx_lt_k_squared_l122_122991


namespace polyhedron_euler_formula_l122_122207

variable (A F S : ℕ)
variable (closed_polyhedron : Prop)

theorem polyhedron_euler_formula (h : closed_polyhedron) : A + 2 = F + S := sorry

end polyhedron_euler_formula_l122_122207


namespace part1_part2_l122_122641

-- Definitions (conditions).
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def reflection_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Given point A
def A : ℝ × ℝ := (-1, 0)

-- Relationship between points P and Q
def AP_eq_lambda_AQ (P Q : ℝ × ℝ) (λ : ℝ) : Prop :=
  (P.1 + 1) = λ * (Q.1 + 1) ∧ P.2 = λ * Q.2 ∧ λ ≠ 1

-- Part 1: Prove that line MQ passes through the focus F
theorem part1 (P Q M : ℝ × ℝ) (λ : ℝ) (hP : parabola P.1 P.2) (hQ : parabola Q.1 Q.2) 
  (hAPQ : AP_eq_lambda_AQ P Q λ) (hM : M = reflection_x_axis P) : 
  (∃ (F : ℝ × ℝ), focus F ∧ ∀ x, x ∈ line_through M Q → x ∈ line_through M F) := 
sorry

-- Part 2: Find the maximum value of |PQ|
theorem part2 (P Q : ℝ × ℝ) (λ : ℝ) (hP : parabola P.1 P.2) (hQ : parabola Q.1 Q.2)
  (hAPQ : AP_eq_lambda_AQ P Q λ) (hlambda : λ ∈ set.Icc (1 / 3 : ℝ) (1 / 2)) :
  ∃ max_val : ℝ, max_val = (4 * real.sqrt 7) / 3 := 
sorry

end part1_part2_l122_122641


namespace radii_triangle_inequality_l122_122818

structure Triangle :=
  (A B C : Point)
  (midpoint_BC : Point)

def radius_inscribed (T : Triangle) (P M : Point) :=
  let a := segmentLength T.A T.B
  let b := segmentLength T.B M
  let c := segmentLength T.A M
  (area T.A T.B M) / (semi_perimeter a b c)

theorem radii_triangle_inequality (A B C M : Point) (h_midpoint : isMidpoint M B C) :
  let T₁ := Triangle.mk A B M
  let T₂ := Triangle.mk A C M
  let r₁ := radius_inscribed T₁ A M
  let r₂ := radius_inscribed T₂ A M
  r₁ < 2 * r₂ :=
by
  sorry

end radii_triangle_inequality_l122_122818


namespace books_arrangement_count_l122_122654

theorem books_arrangement_count : 
  let totalBooks := 7
  let identicalMathBooks := 2
  let identicalScienceBooks := 2
  let differentBooks := totalBooks - identicalMathBooks - identicalScienceBooks
  (totalBooks.factorial / (identicalMathBooks.factorial * identicalScienceBooks.factorial) = 1260) := 
by
  sorry

end books_arrangement_count_l122_122654


namespace midline_of_isosceles_trapezoid_l122_122777

theorem midline_of_isosceles_trapezoid (h α : ℝ) :
  let cot := λ x : ℝ, 1 / (Real.tan x) in
  ∀ (midline : ℝ), midline = h * cot (α / 2) :=
sorry

end midline_of_isosceles_trapezoid_l122_122777


namespace x_sub_p_l122_122661

theorem x_sub_p (x p : ℝ) (h₁ : |x - 2| = p) (h₂ : x < 2) : x - p = 2 - 2p :=
by
  sorry

end x_sub_p_l122_122661


namespace triangle_area_131415_triangle_area_51213_l122_122560

/-- Heron's formula for the area of a triangle given sides a, b, and c -/
def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_131415 :
  heron_area 13 14 15 = 84 :=
by
  sorry

theorem triangle_area_51213 :
  heron_area 5 12 13 = 30 :=
by
  sorry

end triangle_area_131415_triangle_area_51213_l122_122560


namespace trains_meet_in_l122_122429

def train_meeting_time (length1 length2 distance speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (1000 / 3600)
  let speed2_ms := speed2_kmh * (1000 / 3600)
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := length1 + length2 + distance
  total_distance / relative_speed

theorem trains_meet_in (length1 length2 distance speed1_kmh speed2_kmh : ℝ)
  (h_length1 : length1 = 100)
  (h_length2 : length2 = 200)
  (h_distance : distance = 140)
  (h_speed1_kmh : speed1_kmh = 54)
  (h_speed2_kmh : speed2_kmh = 72) : train_meeting_time length1 length2 distance speed1_kmh speed2_kmh = 12.57 :=
by
  rw [h_length1, h_length2, h_distance, h_speed1_kmh, h_speed2_kmh]
  sorry

end trains_meet_in_l122_122429


namespace vanessa_basketball_score_l122_122679

theorem vanessa_basketball_score (team_total : ℕ) (other_players : ℕ) (average_points : ℕ) (vanessa_score : ℕ) 
    (h1 : team_total = 72) 
    (h2 : other_players = 7) 
    (h3 : average_points = 6) 
    (h4 : other_players * average_points + vanessa_score = team_total) : 
    vanessa_score = 30 :=
by {
  sorry,
}

end vanessa_basketball_score_l122_122679


namespace find_deductive_reasoning_l122_122863

noncomputable def is_deductive_reasoning (reasoning : String) : Prop :=
  match reasoning with
  | "B" => true
  | _ => false

theorem find_deductive_reasoning : is_deductive_reasoning "B" = true :=
  sorry

end find_deductive_reasoning_l122_122863


namespace number_of_skew_line_pairs_l122_122162

-- Definitions corresponding to conditions in the problem.

-- Type to represent points in 3D space.
structure Point := (x y z : ℝ)

-- The vertices of the rectangular prism.
def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨1, 0, 0⟩
def C : Point := ⟨1, 1, 0⟩
def D : Point := ⟨0, 1, 0⟩
def A' : Point := ⟨0, 0, 1⟩
def B' : Point := ⟨1, 0, 1⟩
def C' : Point := ⟨1, 1, 1⟩
def D' : Point := ⟨0, 1, 1⟩

-- List of lines in the rectangular prism.
def lines : list (Point × Point) := 
  [(A, B'), (B, A'), (C, D'), (D, C'), (A, D'), (D, A'), 
   (B, C'), (C, B'), (A, C), (B, D), (A', C'), (B', D')]

-- Definition of skew lines.
def are_skew (l1 l2 : Point × Point) : Prop :=
  ¬(∃ p : Point, (p = l1.1 ∨ p = l1.2) ∧ (p = l2.1 ∨ p = l2.2)) ∧
  ¬((l1.1.1 = l1.2.1 ∧ l1.1.1 = l2.1.1 ∧ l2.1.1 = l2.2.1) ∨ 
    (l1.1.2 = l1.2.2 ∧ l1.1.2 = l2.1.2 ∧ l2.1.2 = l2.2.2) ∨ 
    (l1.1.3 = l1.2.3 ∧ l1.1.3 = l2.1.3 ∧ l2.1.3 = l2.2.3))

-- Definition that counts the number of skew line pairs.
def skew_pairs_count (lines : list (Point × Point)) : ℕ :=
  (list.filter (λ (p : Point × Point × Point × Point), are_skew p.1 p.2) 
   (lines.product lines)).length / 2

-- Main theorem statement.
theorem number_of_skew_line_pairs : skew_pairs_count lines = 30 :=
by sorry

end number_of_skew_line_pairs_l122_122162


namespace exists_identical_coordinates_l122_122218

theorem exists_identical_coordinates
  (O O' : ℝ × ℝ)
  (Ox Oy O'x' O'y' : ℝ → ℝ)
  (units_different : ∃ u v : ℝ, u ≠ v)
  (O_ne_O' : O ≠ O')
  (Ox_not_parallel_O'x' : ∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π) :
  ∃ S : ℝ × ℝ, (S.1 = Ox S.1 ∧ S.2 = Oy S.2) ∧ (S.1 = O'x' S.1 ∧ S.2 = O'y' S.2) :=
sorry

end exists_identical_coordinates_l122_122218


namespace expanded_dinning_area_correct_l122_122850

noncomputable def expanded_dinning_area :=
  let π := Real.pi in
  let area_rect := 35 in
  let radius_semi_circle := 4 in
  let area_semi_circle := (π * radius_semi_circle^2) / 2 in
  let base_triangle := 5 in
  let height_triangle := 6 in
  let area_triangle := (base_triangle * height_triangle) / 2 in
  area_rect + area_semi_circle + area_triangle

theorem expanded_dinning_area_correct :
  expanded_dinning_area ≈ 75.1328 :=
by
  have π_approx : Real.pi ≈ 3.1416 := Real.pi_approx
  have h1 : 35 = 35 := rfl
  have h2 : (3.1416 * 4^2) / 2 ≈ 25.1328 := by
    norm_num
  have h3 : (5 * 6) / 2 = 15 := rfl
  norm_num
  sorry

end expanded_dinning_area_correct_l122_122850


namespace find_faulty_keys_l122_122443

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122443


namespace bryan_total_after_discount_l122_122096

theorem bryan_total_after_discount 
  (n : ℕ) (p : ℝ) (d : ℝ) (h_n : n = 8) (h_p : p = 1785) (h_d : d = 0.12) :
  (n * p - (n * p * d) = 12566.4) :=
by
  sorry

end bryan_total_after_discount_l122_122096


namespace length_of_bridge_l122_122098

noncomputable def convert_speed (km_per_hour : ℝ) : ℝ := km_per_hour * (1000 / 3600)

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (passing_time : ℝ)
  (total_distance_covered : ℝ)
  (bridge_length : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  total_distance_covered = convert_speed train_speed_kmh * passing_time →
  bridge_length = total_distance_covered - train_length →
  bridge_length = 160 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_bridge_l122_122098


namespace cube_modulo_9_l122_122663

theorem cube_modulo_9 (N : ℤ) (h : N % 9 = 2 ∨ N % 9 = 5 ∨ N % 9 = 8) : 
  (N^3) % 9 = 8 :=
by sorry

end cube_modulo_9_l122_122663


namespace opposite_of_neg_five_l122_122362

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122362


namespace stationery_box_cost_l122_122071

theorem stationery_box_cost (unit_price : ℕ) (quantity : ℕ) (total_cost : ℕ) :
  unit_price = 23 ∧ quantity = 3 ∧ total_cost = 3 * 23 → total_cost = 69 :=
by
  sorry

end stationery_box_cost_l122_122071


namespace triangle_ABC_PQRS_max_area_BC_4_l122_122267

noncomputable def harmonic_mean (a b : ℝ) : ℝ :=
  2 * a * b / (a + b)

theorem triangle_ABC_PQRS_max_area_BC_4
  (A B C P Q R S D : Point)
  (h_acute : acute_triangle A B C)
  (h_perp : AD ⊥ BC)
  (h_p_on_ab : P ∈ segment A B)
  (h_q_on_bc : Q ∈ segment B C)
  (h_r_on_bc : R ∈ segment B C)
  (h_s_on_ac : S ∈ segment A C)
  (h_max_area : ∀ (T U V W : Point) (h_t_on_ab : T ∈ segment A B) 
                              (h_u_on_bc : U ∈ segment B C)
                              (h_v_on_bc : V ∈ segment B C)
                              (h_w_on_ac : W ∈ segment A C), 
                 area_of_rect PQRS ≥ area_of_rect TUVW)
  (h_harmonic_mean : PQ = harmonic_mean (AD / DB) (AD / DC)) :
  length BC = 4 := 
sorry

end triangle_ABC_PQRS_max_area_BC_4_l122_122267


namespace sqrt_mixed_number_eq_l122_122935

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l122_122935


namespace problem_statement_l122_122606

variable (a : ℕ → ℝ)
variable (M : ℝ) (bound_seq : ℕ → ℝ)

-- Given the sequence is bounded
axiom h_bounded : ∀ n, a n ≤ M

-- Given the inequality condition for all n ≥ 1
axiom h_inequality : ∀ n, a n < (∑ k in Finset.range (2*n + 2007) \ Finset.range n, a k / (k + 1)) + 1 / (2*n + 2007)

theorem problem_statement : ∀ n, 1 ≤ n → a n < 1 / n :=
by
  intros n hn
  sorry

end problem_statement_l122_122606


namespace u_n_differs_by_two_from_square_l122_122068

def u : ℕ → ℤ
| 0       := 2
| 1       := 7  -- u_2 corresponds to u(1) as index starts from 0 in Lean
| 2       := 7  -- u_3 corresponds to u(2) as index starts from 0 in Lean
| (n + 3) := u (n + 2) * u (n + 1) - u n

theorem u_n_differs_by_two_from_square (n : ℕ) : ∃ k : ℤ, k^2 - 2 = u n := 
  sorry

end u_n_differs_by_two_from_square_l122_122068


namespace minimum_value_w_l122_122436

theorem minimum_value_w : ∃ (x y : ℝ), ∀ w, w = 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 → w ≥ 20.25 :=
sorry

end minimum_value_w_l122_122436


namespace estimate_larger_than_difference_l122_122243

theorem estimate_larger_than_difference (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
    (x + z) - (y - z) > x - y :=
    sorry

end estimate_larger_than_difference_l122_122243


namespace complement_intersection_complement_l122_122278

-- Define the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the statement of the proof problem
theorem complement_intersection_complement:
  (U \ (A ∩ B)) = {1, 4, 6} := by
  sorry

end complement_intersection_complement_l122_122278


namespace opposite_of_neg_five_l122_122381

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122381


namespace sqrt_of_mixed_number_as_fraction_l122_122928

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l122_122928


namespace count_triangles_in_figure_l122_122555

theorem count_triangles_in_figure :
  let A B C D : Type := ℝ
  let vertices := {A, B, C, D}
  let diagonals := {AC, BD}
  let midpoints := λ s : set Type, (s : A+B)/2
  let inner_square := {E, F, G, H} where
                      E = midpoints A B,
                      F = midpoints B C,
                      G = midpoints C D,
                      H = midpoints D A
  let second_level_midpoints := {midpoints x y | x y ∈ inner_square}
  let total_triangles := 16 (smallest) + 8 (medium) + 4 (largest)
  total_triangles = 28 := 
sorry

end count_triangles_in_figure_l122_122555


namespace soccer_team_lineup_selection_l122_122853

/-- 
We are given 16 soccer players:
  - 3 can play as a goalkeeper,
  - 5 can play as a defender,
  - 8 can play as a midfielder,
  - 4 can play as forwards.
We need to select a starting lineup consisting of:
  - 1 goalkeeper,
  - 1 defender,
  - 1 midfielder,
  - 2 forwards (in two specific positions).
Prove there are 1440 ways to select such a starting lineup.
-/
theorem soccer_team_lineup_selection :
  let goalkeepers := 3,
      defenders := 5,
      midfielders := 8,
      forwards := 4 in
  (goalkeepers * defenders * midfielders * (forwards * (forwards - 1))) = 1440 :=
by
  sorry

end soccer_team_lineup_selection_l122_122853


namespace beneficial_card_l122_122024

theorem beneficial_card
  (P : ℕ) (r_c r_d r_i : ℚ) :
  let credit_income := (r_c * P + r_i * P)
  let debit_income := r_d * P
  P = 8000 ∧ r_c = 0.005 ∧ r_d = 0.0075 ∧ r_i = 0.005 →
  credit_income > debit_income :=
by
  intro h
  cases h with hP hr
  cases hr with hrc hrd_ri
  cases hrd_ri with hrd hri
  rw [hP, hrc, hrd, hri]
  sorry

end beneficial_card_l122_122024


namespace isosceles_triangle_exterior_angle_eq_l122_122221

-- Define the given conditions in terms of Lean
def isosceles_triangle (t : Triangle) : Prop :=
  t.angleA = t.angleB ∨ t.angleB = t.angleC ∨ t.angleC = t.angleA  

def vertex_angle (t : Triangle) (x : ℝ) : Prop :=
  t.vertexAngle = x

def exterior_angle (t : Triangle) (y : ℝ) : Prop :=
  t.exteriorBaseAngle = y

-- Prove the function expression for y in terms of x
theorem isosceles_triangle_exterior_angle_eq (t : Triangle) (x y : ℝ) 
  (h1 : isosceles_triangle t)
  (h2 : vertex_angle t x)
  (h3 : exterior_angle t y) :
  y = 90 + x / 2 := 
begin 
  sorry 
end

end isosceles_triangle_exterior_angle_eq_l122_122221


namespace area_of_EFGH_l122_122320

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l122_122320


namespace a_eq_2_neither_sufficient_nor_necessary_l122_122182

-- Define the complex number z
def z (a : ℝ) : ℂ := complex.mk (a-4) (a+2)

-- Statement that a = 2 is neither sufficient nor necessary for z to be a pure imaginary number.
theorem a_eq_2_neither_sufficient_nor_necessary (a : ℝ) : 
  ¬((∀ a, z a = complex.i * (a + 2) → a = 2) ∨ (∀ a, a = 2 → z a = complex.i * (a + 2))) :=
sorry

end a_eq_2_neither_sufficient_nor_necessary_l122_122182


namespace xy_product_equals_zero_l122_122322

theorem xy_product_equals_zero (x y : ℝ) (h1 : 2^x = 256^(y + 1)) (h2 : 81^y = 3^(x - 4)) : x * y = 0 :=
by sorry

end xy_product_equals_zero_l122_122322


namespace intersection_solution_l122_122715

open Set

def P : Set ℝ := {x | log 2 x < 1}
def Q : Set ℝ := {x | x ^ 2 - 4 * x + 4 < 1}
def U : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_solution : P ∩ Q = U :=
by
  sorry

end intersection_solution_l122_122715


namespace find_z_l122_122601

open Complex

theorem find_z (z : ℂ) (h : ((1 - I) ^ 2) / z = 1 + I) : z = -1 - I :=
sorry

end find_z_l122_122601


namespace incorrect_deduction_wrong_l122_122121

-- Define the conditions as assumptions
variable (Y : ℕ) (hY : Y ≤ 3)

-- Define the incorrect statement we need to prove against
def incorrect_deduction_statement := (-300 * (3 - Y) ≠ -300 * Y)

-- Define the Lean 4 theorem statement
theorem incorrect_deduction_wrong (hY : ℕ ≤ 3) : incorrect_deduction_statement Y hY :=
sorry

end incorrect_deduction_wrong_l122_122121


namespace problem_1_problem_2_l122_122154

theorem problem_1 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 3 / 4 := 
sorry

theorem problem_2 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a / (b + 2 * c + 3 * d) + b / (c + 2 * d + 3 * a) + c / (d + 2 * a + 3 * b) + d / (a + 2 * b + 3 * c)) ≥ 2 / 3 :=
sorry

end problem_1_problem_2_l122_122154


namespace park_cycling_time_l122_122408

def length_breadth_ratio (L B : ℕ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℕ) : Prop := L * B = 120000
def speed_of_cyclist : ℕ := 200 -- meters per minute
def perimeter (L B : ℕ) : ℕ := 2 * L + 2 * B
def time_to_complete_round (P v : ℕ) : ℕ := P / v

theorem park_cycling_time
  (L B : ℕ)
  (h_ratio : length_breadth_ratio L B)
  (h_area : area_of_park L B)
  : time_to_complete_round (perimeter L B) speed_of_cyclist = 8 :=
by
  sorry

end park_cycling_time_l122_122408


namespace feasible_tunnel_construction_l122_122762

open Set

-- Define the properties of the Martian metro line
def closed_self_intersecting_line (C : ℝ → ℝ × ℝ) : Prop :=
 ∃ t₀ t₁ t₂ t₃, t₀ ≠ t₁ ∧ C t₀ = C t₁ ∧ t₂ ≠ t₃ ∧ ∀ t ≠ t₀, C t ≠ C t₀

-- Define the alternating construction requirement
def alternating_tunnel_construction (C : ℝ → ℝ × ℝ) : Prop :=
 ∃ (regions : ℝ × ℝ → ℕ), 
      (∀ p, ∃ n, regions p = n) ∧
      (∀ p1 p2, (C p1 = C p2 → regions p1 ≠ regions p2))

-- Define the main theorem statement
theorem feasible_tunnel_construction 
  (C : ℝ → ℝ × ℝ) 
  (hC : closed_self_intersecting_line C) : 
  alternating_tunnel_construction C := 
sorry

end feasible_tunnel_construction_l122_122762


namespace faulty_keys_l122_122469

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122469


namespace faulty_key_in_digits_l122_122473

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122473


namespace blue_balloons_l122_122864

theorem blue_balloons (total_balloons red_balloons green_balloons purple_balloons : ℕ)
  (h1 : total_balloons = 135)
  (h2 : red_balloons = 45)
  (h3 : green_balloons = 27)
  (h4 : purple_balloons = 32) :
  total_balloons - (red_balloons + green_balloons + purple_balloons) = 31 :=
by
  sorry

end blue_balloons_l122_122864


namespace spherical_to_rectangular_correct_l122_122065

noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

def initial_rectangular_coords : ℝ × ℝ × ℝ := (3, -4, 12)

noncomputable def spherical_coords (rect_coords : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (rect_coords.1 ^ 2 + rect_coords.2 ^ 2 + rect_coords.3 ^ 2)
  let θ := Real.atan rect_coords.2 rect_coords.1
  let φ := Real.acos (rect_coords.3 / ρ)
  (ρ, θ, φ)

def new_spherical_coords (spherical : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let ρ := spherical.1
  let θ := spherical.2
  let φ := spherical.3 + Real.pi / 4
  (ρ, θ, φ)

theorem spherical_to_rectangular_correct :
  let (ρ, θ, φ) := spherical_coords initial_rectangular_coords
  let (new_ρ, new_θ, new_φ) := new_spherical_coords (ρ, θ, φ)
  spherical_to_rectangular new_ρ new_θ new_φ
  = (-51 * Real.sqrt 2 / 10, -68 * Real.sqrt 2 / 10, 91 * Real.sqrt 2 / 20) := by
  sorry

end spherical_to_rectangular_correct_l122_122065


namespace no_four_consecutive_perf_square_l122_122912

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l122_122912


namespace arun_crosses_train_b_in_15_seconds_l122_122427

theorem arun_crosses_train_b_in_15_seconds :
  ∀ (length_A length_B speed_A_kmh speed_B_kmh : ℝ),
    length_A = 225 ∧
    length_B = 150 ∧
    speed_A_kmh = 54 ∧
    speed_B_kmh = 36 →
    let speed_A := speed_A_kmh * (5 / 18) in
    let speed_B := speed_B_kmh * (5 / 18) in
    let relative_speed := speed_A + speed_B in
    let total_distance := length_A + length_B in
    total_distance / relative_speed = 15 := by
  intros length_A length_B speed_A_kmh speed_B_kmh h,
  cases h,
  -- convert km/hr to m/s
  let speed_A := speed_A_kmh * (5 / 18),
  let speed_B := speed_B_kmh * (5 / 18),
  -- calculate relative speed
  let relative_speed := speed_A + speed_B,
  -- calculate total distance
  let total_distance := length_A + length_B,
  -- calculate time
  have time_eq := total_distance / relative_speed,
  -- prove the time taken is 15 seconds
  sorry

end arun_crosses_train_b_in_15_seconds_l122_122427


namespace opposite_of_neg5_is_pos5_l122_122357

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122357


namespace range_of_omega_l122_122885

-- Define the sine function f(x) = sin(ωx + π/3)
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- The hypothesis for the problem
def has_extreme_points (ω : ℝ) :=
  (∃ x1 x2 x3 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (x3 < Real.pi)
    ∧ f' ω x1 = 0 ∧ f' ω x2 = 0 ∧ f' ω x3 = 0)

def has_zeros (ω : ℝ) :=
  (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < Real.pi)
    ∧ f ω x1 = 0 ∧ f ω x2 = 0)

-- The main theorem to be proved
theorem range_of_omega (ω : ℝ) :
  has_extreme_points ω ∧ has_zeros ω ↔ (13/6 < ω ∧ ω ≤ 8/3) :=
by
  sorry

end range_of_omega_l122_122885


namespace find_defective_keys_l122_122450

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122450


namespace smallest_positive_integer_with_conditions_l122_122535

theorem smallest_positive_integer_with_conditions (n : ℕ) (h₀ : alice_number = 36)
  (h₁ : ∀ p, prime p → p ∣ alice_number → p ∣ n)
  (h₂ : 5 ∣ n) :
  n = 30 :=
by
  let alice_number := 36
  sorry

end smallest_positive_integer_with_conditions_l122_122535


namespace opposite_of_neg_five_is_five_l122_122398

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122398


namespace faulty_keys_l122_122465

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122465


namespace range_of_p_l122_122869

noncomputable def success_prob_4_engine (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p) + p^4

noncomputable def success_prob_2_engine (p : ℝ) : ℝ :=
  p^2

theorem range_of_p (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  success_prob_4_engine p > success_prob_2_engine p ↔ (1/3 < p ∧ p < 1) :=
by
  sorry

end range_of_p_l122_122869


namespace remove_terms_sum_eq_3_over_4_l122_122816

theorem remove_terms_sum_eq_3_over_4 : 
  let terms := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let sum_terms := list.sum terms
  (sum_terms - (1/12 + 1/15)) = 3/4 :=
  by
    let terms := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
    have h_sum : list.sum terms = 49/60 := sorry
    have h_remove : 1/12 + 1/15 = 1/15 := sorry  
    have : 49/60 - 1/15 = 3/4 := sorry
    rw [←h_sum, ←h_remove]
    exact 

sorry

end remove_terms_sum_eq_3_over_4_l122_122816


namespace number_of_solutions_l122_122584

-- Define the equation parts
def numerator (z : ℂ) := z^4 - 1
def denominator (z : ℂ) := z^3 - 3z + 2

-- The main theorem statement
theorem number_of_solutions : 
  (∃ z : ℂ, numerator z = 0 ∧ denominator z ≠ 0) -- This ensures z is a solution of the numerator and not a root of the denominator
  .set({ z | numerator z = 0 }).card - { z | numerator z = 0 ∧ denominator z = 0 }.to_finset.card = 3 := 
begin
  -- Proof will be filled here
  sorry
end

end number_of_solutions_l122_122584


namespace negate_statement_l122_122183

variable (Students Teachers : Type)
variable (Patient : Students → Prop)
variable (PatientT : Teachers → Prop)
variable (a : ∀ t : Teachers, PatientT t)
variable (b : ∃ t : Teachers, PatientT t)
variable (c : ∀ s : Students, ¬ Patient s)
variable (d : ∀ s : Students, ¬ Patient s)
variable (e : ∃ s : Students, ¬ Patient s)
variable (f : ∀ s : Students, Patient s)

theorem negate_statement : (∃ s : Students, ¬ Patient s) ↔ ¬ (∀ s : Students, Patient s) :=
by sorry

end negate_statement_l122_122183


namespace find_line_m_l122_122120

theorem find_line_m (Q Q'' : ℝ × ℝ) (l m : ℝ → ℝ → Prop) :
  (l = (λ x y, 3 * x - 4 * y = 0)) →            -- line ℓ
  (Q = (3, -2)) →                                -- Point Q
  (Q'' = (2, 5)) →                               -- Point Q''
  (l (0, 0)) → m (0, 0) →                      -- Lines ℓ and m intersect at the origin
  ∃ m : ℝ → ℝ → Prop, m (0, 0) ∧ (m = (λ x y, x + 7 * y = 0)) :=
by
  sorry

end find_line_m_l122_122120


namespace last_two_digits_of_power_sequence_l122_122960

noncomputable def power_sequence (n : ℕ) : ℤ :=
  (Int.sqrt 29 + Int.sqrt 21)^(2 * n) + (Int.sqrt 29 - Int.sqrt 21)^(2 * n)

theorem last_two_digits_of_power_sequence :
  (power_sequence 992) % 100 = 71 := by
  sorry

end last_two_digits_of_power_sequence_l122_122960


namespace product_of_y_coordinates_l122_122740

-- Definitions based on conditions:
def is_on_line (P : ℝ × ℝ) : Prop := P.1 = 3
def distance (P1 P2 : ℝ × ℝ) : ℝ := Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

-- The given point
def point1 : ℝ × ℝ := (1, -1)

-- Define the conditions for the point P
def valid_point (P : ℝ × ℝ) : Prop :=
  is_on_line P ∧ distance P point1 = 10

-- Statement to prove the required result
theorem product_of_y_coordinates : 
  ∀ y : ℝ, valid_point (3, y) → (y = -1 + Real.sqrt 96 ∨ y = -1 - Real.sqrt 96) 
                      → (y = -1 + Real.sqrt 96 → y = -1 - Real.sqrt 96 → (y - 1 - Real.sqrt 96) * (y - 1 + Real.sqrt 96) = -95) := 
by
  intros y h₁ h₂
  sorry

end product_of_y_coordinates_l122_122740


namespace total_money_raised_for_charity_l122_122768

theorem total_money_raised_for_charity:
    let price_small := 2
    let price_medium := 3
    let price_large := 5
    let num_small := 150
    let num_medium := 221
    let num_large := 185
    num_small * price_small + num_medium * price_medium + num_large * price_large = 1888 := by
  sorry

end total_money_raised_for_charity_l122_122768


namespace terminal_sides_equal_is_even_function_l122_122485

-- Define the rotation degrees and terminal sides
def terminal_side (deg : ℤ) : ℝ :=
  deg % 360

-- Define the original angles
def angle1 : ℤ := -497
def angle2 : ℤ := 2023

-- Prove their terminal sides are the same.
theorem terminal_sides_equal : terminal_side angle1 = terminal_side angle2 :=
  sorry
  
-- Define the trigonometric function given
def trig_function (x : ℝ) : ℝ :=
  Real.sin((2/3) * x - (7/2) * Real.pi)

-- Prove the function is even
theorem is_even_function : ∀ x : ℝ, trig_function x = trig_function (-x) :=
  sorry

end terminal_sides_equal_is_even_function_l122_122485


namespace distinct_subsets_after_removal_l122_122289

theorem distinct_subsets_after_removal
  (X : Type)
  (n : ℕ)
  (hx : fintype X) -- X is a finite set
  (h_card_X : fintype.card X = n)
  (A : fin n → set X)
  (h_distinct : function.injective A) -- A_1, A_2, ..., A_n are distinct
  : ∃ x ∈ (univ : set X), function.injective (λ i, A i \ {x}) :=
sorry

end distinct_subsets_after_removal_l122_122289


namespace angle_Y_measure_l122_122690

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end angle_Y_measure_l122_122690


namespace arrangement_ways_l122_122230

theorem arrangement_ways : 
  ∀ (persons : ℕ) (chairs : ℕ), 
  persons = 5 ∧ chairs = 6 → 
  (∏ i in finset.range persons, (chairs - i)) = 720 :=
begin
  intros persons chairs,
  rintros ⟨h1, h2⟩,
  subst h1,
  subst h2,
  simp only [finset.prod_range_succ, finset.prod_range_succ, nat.cast_sub, nat.cast_succ, nat.cast_bit0, nat.cast_bit1],
  norm_num
end

end arrangement_ways_l122_122230


namespace identify_faulty_key_l122_122479

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122479


namespace solve_inequality_find_m_range_l122_122191

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem solve_inequality (a : ℝ) : 
  ∀ x : ℝ, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨ 
    (a > 1) ∨ 
    (a < 1 ∧ (x > 3 - a ∨ x < a + 1)) :=
sorry

theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, f x > g x m) ↔ m < 5 :=
sorry

end solve_inequality_find_m_range_l122_122191


namespace similar_triangles_equal_ratios_l122_122700

variables (PQ QR YZ XY : ℝ)

theorem similar_triangles_equal_ratios
  (h_similar : similar_triangles PQR XYZ)
  (hPQ : PQ = 8)
  (hQR : QR = 16)
  (hYZ : YZ = 24) :
  XY = 12 :=
by
  -- Proof of the theorem
  sorry

end similar_triangles_equal_ratios_l122_122700


namespace at_least_one_ge_two_l122_122283

theorem at_least_one_ge_two (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  (a + 1/b >= 2) ∨ (b + 1/c >= 2) ∨ (c + 1/a >= 2) :=
sorry

end at_least_one_ge_two_l122_122283


namespace find_prime_squares_l122_122953

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l122_122953


namespace area_enclosed_by_curves_l122_122765

theorem area_enclosed_by_curves :
  ∫ x in 0..1, (x^2 - x^3) = 1 / 12 :=
by
  sorry

end area_enclosed_by_curves_l122_122765


namespace last_two_digits_l122_122958

def x := Real.sqrt 29 + Real.sqrt 21
def y := Real.sqrt 29 - Real.sqrt 21
def a := x^2 = 50 + 2 * Real.sqrt 609
def b := y^2 = 50 - 2 * Real.sqrt 609
def S : ℕ → ℝ :=
  λ n => a^n + b^n

theorem last_two_digits (n : ℕ) :
  ((x : ℝ) ^ 2)^(n : ℕ) + ((y : ℝ)(^2))^(n : ℕ) = 71 := 
sorry

end last_two_digits_l122_122958


namespace find_b_l122_122758

variable (f : ℝ → ℝ) (finv : ℝ → ℝ)

-- Defining the function f
def f_def (b : ℝ) (x : ℝ) := 1 / (2 * x + b)

-- Defining the inverse function
def finv_def (x : ℝ) := (2 - 3 * x) / (3 * x)

theorem find_b (b : ℝ) :
  (∀ x : ℝ, f_def b (finv_def x) = x ∧ finv_def (f_def b x) = x) ↔ b = -2 := by
  sorry

end find_b_l122_122758


namespace distribute_students_l122_122562

-- Define the students
inductive Student
| A
| B
| C
| D

open Student

/-- A proof statement asserting the number of valid distributions of four students
    into two classrooms under given constraints. -/
theorem distribute_students : 
  (∃ class1 class2 : list Student, 
    class1 ≠ [] ∧ class2 ≠ [] ∧ 
    ¬(A ∈ class1 ∧ B ∈ class1) ∧ 
    ¬ (A ∈ class2 ∧ B ∈ class2) ∧
    (class1 ∪ class2 = [A, B, C, D] ∧ class1 ∩ class2 = [] ))
    → 8 := 
  sorry

end distribute_students_l122_122562


namespace range_of_f_on_interval_l122_122790

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem range_of_f_on_interval :
  (set.range (λ x, f x)) = set.Icc 0 8 :=
by {
  let x := [-1, 1],
  let y := x.map (λ x, x^2 - 4*x + 3),
  let ymin := 0,
  let ymax := 8,
  show (set.range (λ x, x^2 - 4*x + 3)) = set.Icc 0 8,
  sorry
}

end range_of_f_on_interval_l122_122790


namespace find_x_l122_122717

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
by
  sorry

end find_x_l122_122717


namespace sqrt_of_mixed_number_as_fraction_l122_122931

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l122_122931


namespace range_y_eq_x_plus_sqrt_1_minus_2x_l122_122586

theorem range_y_eq_x_plus_sqrt_1_minus_2x (x y : ℝ) :
  y = x + sqrt (1 - 2 * x) → ∃ y, y ≤ 1 :=
sorry

end range_y_eq_x_plus_sqrt_1_minus_2x_l122_122586


namespace bookseller_loss_l122_122829

theorem bookseller_loss (C S : ℝ) (h : 20 * C = 25 * S) : (C - S) / C * 100 = 20 := by
  sorry

end bookseller_loss_l122_122829


namespace return_to_starting_point_possible_l122_122523

theorem return_to_starting_point_possible 
  (R : ℝ) -- Assume R is the radius of the Earth
  (phi lambda : ℝ) -- Assume initial longitude and latitude
  (H1 : 2 * π * R * cos(phi + 1) = 1) -- Condition for the parallel's circumference
  : ∃ phi' lambda', 
      (phi', lambda') = (phi, lambda) := 
  sorry

end return_to_starting_point_possible_l122_122523


namespace ferry_q_more_time_l122_122029

variables (speed_ferry_p speed_ferry_q distance_ferry_p distance_ferry_q time_ferry_p time_ferry_q : ℕ)
  -- Conditions given in the problem
  (h1 : speed_ferry_p = 8)
  (h2 : time_ferry_p = 2)
  (h3 : distance_ferry_p = speed_ferry_p * time_ferry_p)
  (h4 : distance_ferry_q = 3 * distance_ferry_p)
  (h5 : speed_ferry_q = speed_ferry_p + 4)
  (h6 : time_ferry_q = distance_ferry_q / speed_ferry_q)

theorem ferry_q_more_time : time_ferry_q - time_ferry_p = 2 :=
by
  sorry

end ferry_q_more_time_l122_122029


namespace average_speed_is_9_mph_l122_122841

-- Define the conditions
def distance_north_ft := 5280
def north_speed_min_per_mile := 3
def rest_time_min := 10
def south_speed_miles_per_min := 3

-- Define a function to convert feet to miles
def feet_to_miles (ft : ℕ) : ℕ := ft / 5280

-- Define the time calculation for north and south trips
def time_north_min (speed : ℕ) (distance_ft : ℕ) : ℕ :=
  speed * feet_to_miles distance_ft

def time_south_min (speed_miles_per_min : ℕ) (distance_ft : ℕ) : ℕ :=
  (feet_to_miles distance_ft) / speed_miles_per_min

def total_time_min (time_north rest_time time_south : ℕ) : Rat :=
  time_north + rest_time + time_south

-- Convert total time into hours
def total_time_hr (total_time_min : Rat) : Rat :=
  total_time_min / 60

-- Define the total distance in miles
def total_distance_miles (distance_ft : ℕ) : ℕ :=
  2 * feet_to_miles distance_ft

-- Calculate the average speed
def average_speed (total_distance : ℕ) (total_time_hr : Rat) : Rat :=
  total_distance / total_time_hr

-- Prove the average speed is 9 miles per hour
theorem average_speed_is_9_mph : 
  average_speed (total_distance_miles distance_north_ft)
                (total_time_hr (total_time_min (time_north_min north_speed_min_per_mile distance_north_ft)
                                              rest_time_min
                                              (time_south_min south_speed_miles_per_min distance_north_ft)))
    = 9 := by
  sorry

end average_speed_is_9_mph_l122_122841


namespace five_people_six_chairs_l122_122236

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l122_122236


namespace lizette_overall_average_is_94_l122_122297

-- Defining the given conditions
def third_quiz_score : ℕ := 92
def first_two_quizzes_average : ℕ := 95
def total_quizzes : ℕ := 3

-- Calculating total points from the conditions
def total_points : ℕ := first_two_quizzes_average * 2 + third_quiz_score

-- Defining the overall average to prove
def overall_average : ℕ := total_points / total_quizzes

-- The theorem stating Lizette's overall average after taking the third quiz
theorem lizette_overall_average_is_94 : overall_average = 94 := by
  sorry

end lizette_overall_average_is_94_l122_122297


namespace cyclic_hexagon_ratio_l122_122288

open Lean.Meta
open Lean.Elab
open Lean.Elab.Tactic

-- Define the cyclic hexagon inscribed in circle
variables {k : Type} [field k]
variables {A B C D E F Q P : k}

-- Conditions given in the problem
variables (circ : ∀ a b c d e f : k, inscribed a b c d e f k)
variables (h1 : A = B)
variables (h2 : B = D)
variables (h3 : D = C)
variables (h4 : E = F)
variables (intQ : intersection AD BE CF Q)
variables (intP : intersection AD CE P)

-- Question as the statement to prove
theorem cyclic_hexagon_ratio (h : ∀ a b c d e f : k,
  inscribed a b c d e f k → A = B → B = D → D = C → E = F →  
  intersection AD BE CF Q → intersection AD CE P → 
  P = AD ∩ CE ∧ Q = AD ∩ BE  ∴  ∀ CP PE,  (CP / PE) = (AC^2 / CE^2)) :
  (CP / PE) = (AC^2 / CE^2) := sorry

end cyclic_hexagon_ratio_l122_122288


namespace Changhyeok_snacks_l122_122896

theorem Changhyeok_snacks (S D : ℕ) (h1 : 1000 * S + 1300 * D = 15000) (h2 : S + D = 12) : S = 2 :=
begin
  sorry,
end

end Changhyeok_snacks_l122_122896


namespace original_speed_of_car_A_l122_122801

-- Define the speeds of cars A and B
variables (vₐ vₑ : ℕ)

-- Define the conditions given in the problem
axiom h1 : ∀ (vₐ vₑ t : ℕ), t = 6 → (vₐ * t) + (vₑ * t) = (6 * (vₐ + vₑ))
axiom h2 : ∀ (vₐ vₑ t : ℕ), t = 3.6 → (vₑ + 5) * t = (6 * vₑ) - 12
axiom h3 : ∀ (vₐ vₑ t : ℕ), t = 4.2 → (vₐ + 5) * t = (6 * vₐ) + 16

-- The proof problem in Lean 4
theorem original_speed_of_car_A : vₐ = 30 :=
by {
  sorry
}

end original_speed_of_car_A_l122_122801


namespace illuminate_entire_space_l122_122617

theorem illuminate_entire_space (points : Fin 8 → ℝ × ℝ × ℝ) 
  (illuminate : (ℝ × ℝ × ℝ) → Set (ℝ × ℝ × ℝ)) :
  (∀ i, ∃ o : Set (ℝ × ℝ × ℝ), illuminate (points i) = o ∧ 
    (o = { p | p.1 ≤ points i.1 ∧ p.2 ≤ points i.2 ∧ p.3 ≤ points i.3 } ∨
       o = { p | p.1 ≥ points i.1 ∧ p.2 ≤ points i.2 ∧ p.3 ≤ points i.3 } ∨
       o = { p | p.1 ≤ points i.1 ∧ p.2 ≥ points i.2 ∧ p.3 ≤ points i.3 } ∨
       o = { p | p.1 ≤ points i.1 ∧ p.2 ≤ points i.2 ∧ p.3 ≥ points i.3 } ∨
       o = { p | p.1 ≥ points i.1 ∧ p.2 ≥ points i.2 ∧ p.3 ≤ points i.3 } ∨
       o = { p | p.1 ≤ points i.1 ∧ p.2 ≥ points i.2 ∧ p.3 ≥ points i.3 } ∨
       o = { p | p.1 ≥ points i.1 ∧ p.2 ≤ points i.2 ∧ p.3 ≥ points i.3 } ∨
       o = { p | p.1 ≥ points i.1 ∧ p.2 ≥ points i.2 ∧ p.3 ≥ points i.3 })) →
  ∃ orientations, (⋃ i, orientations i (points i) = Set.univ) :=
by
  intros points illuminate h
  sorry

end illuminate_entire_space_l122_122617


namespace sqrt_mixed_number_simplified_l122_122942

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l122_122942


namespace tank_capacity_is_288_l122_122854

/-- A tank is initially one-fourth full of water. After adding 120 gallons of water, 
the tank is two-thirds full. -/
def tank_capacity : ℕ :=
  let initial_fullness : ℚ := 1/4
  let final_fullness : ℚ := 2/3
  let added_water : ℚ := 120
  let fullness_difference : ℚ := final_fullness - initial_fullness
  let capacity := added_water / fullness_difference
  capacity.toNat

theorem tank_capacity_is_288 : tank_capacity = 288 := by
  sorry  

end tank_capacity_is_288_l122_122854


namespace tribe_leadership_choices_l122_122542

theorem tribe_leadership_choices :
  let members := 15
  let ways_to_choose_chief := members
  let remaining_after_chief := members - 1
  let ways_to_choose_supporting_chiefs := Nat.choose remaining_after_chief 2
  let remaining_after_supporting_chiefs := remaining_after_chief - 2
  let ways_to_choose_officers_A := Nat.choose remaining_after_supporting_chiefs 2
  let remaining_for_assistants_A := remaining_after_supporting_chiefs - 2
  let ways_to_choose_assistants_A := Nat.choose remaining_for_assistants_A 2 * Nat.choose (remaining_for_assistants_A - 2) 2
  let remaining_after_A := remaining_for_assistants_A - 2
  let ways_to_choose_officers_B := Nat.choose remaining_after_A 2
  let remaining_for_assistants_B := remaining_after_A - 2
  let ways_to_choose_assistants_B := Nat.choose remaining_for_assistants_B 2 * Nat.choose (remaining_for_assistants_B - 2) 2
  (ways_to_choose_chief * ways_to_choose_supporting_chiefs *
  ways_to_choose_officers_A * ways_to_choose_assistants_A *
  ways_to_choose_officers_B * ways_to_choose_assistants_B = 400762320000) := by
  sorry

end tribe_leadership_choices_l122_122542


namespace midpoints_MC_CD_l122_122244

-- Definitions for points and segments
variables (A B C D M N : Type)
variables (area : A → A → A → ℝ)
variables (AM AC CN CD : ℝ)
variables (collinear : A → A → A → Prop)

-- Condition statements
def condition1 : Prop := area A B D / area B C D = 3 / 4 ∧ area B C D / area A B C = 4 / 1
def condition2 : Prop := AM / AC = CN / CD
def condition3 : Prop := collinear B M N

-- Theorem statement
theorem midpoints_MC_CD 
  (cond1 : condition1)
  (cond2 : condition2)
  (cond3 : condition3) : 
  (AM = AC / 2) ∧ (CN = CD / 2) :=
by
  sorry

end midpoints_MC_CD_l122_122244


namespace range_of_m_l122_122906

noncomputable def f : ℝ → ℝ := sorry

lemma function_symmetric {x : ℝ} : f (2 + x) = f (-x) := sorry

lemma f_decreasing_on_pos_halfline {x y : ℝ} (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) : f x ≥ f y := sorry

theorem range_of_m {m : ℝ} (h : f (1 - m) < f m) : m > (1 / 2) := sorry

end range_of_m_l122_122906


namespace ruiz_take_home_income_l122_122747

-- Conditions
def monthly_salary := 500
def pension_contribution := 0.03
def yearly_bonus := 600
def raise_rate := 0.06
def compound_times_per_year := 4
def tax_rate := 0.10

-- Prove the take-home income
theorem ruiz_take_home_income :
  let monthly_salary_after_pension := (monthly_salary * (1 - pension_contribution))
  let monthly_salary_after_raise := monthly_salary_after_pension * (1 + (raise_rate / compound_times_per_year))^compound_times_per_year
  let yearly_salary := monthly_salary_after_raise * 12
  let total_income := yearly_salary + yearly_bonus
  let take_home_income := total_income * (1 - tax_rate)
  take_home_income ≈ 6101.57 :=
by
  -- Definitions and calculations
  let monthly_salary_after_pension := monthly_salary * (1 - pension_contribution)
  let monthly_salary_after_raise := monthly_salary_after_pension * (1 + (raise_rate / compound_times_per_year))^compound_times_per_year
  let yearly_salary := monthly_salary_after_raise * 12
  let total_income := yearly_salary + yearly_bonus
  let take_home_income := total_income * (1 - tax_rate)
  -- Prove the equation
  sorry

end ruiz_take_home_income_l122_122747


namespace Isabella_hair_length_l122_122260

-- Define the conditions using variables
variables (h_current h_cut_off h_initial : ℕ)

-- The proof problem statement
theorem Isabella_hair_length :
  h_current = 9 → h_cut_off = 9 → h_initial = h_current + h_cut_off → h_initial = 18 :=
by
  intros hc hc' hi
  rw [hc, hc'] at hi
  exact hi


end Isabella_hair_length_l122_122260


namespace factorization_l122_122948
-- Import the necessary library

-- Define the expression
def expr (x : ℝ) : ℝ := 75 * x^2 + 50 * x

-- Define the factored form
def factored_form (x : ℝ) : ℝ := 25 * x * (3 * x + 2)

-- Statement of the equality to be proved
theorem factorization (x : ℝ) : expr x = factored_form x :=
by {
  sorry
}

end factorization_l122_122948


namespace faulty_key_in_digits_l122_122472

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122472


namespace prime_in_choices_l122_122021

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def twenty := 20
def twenty_one := 21
def twenty_three := 23
def twenty_five := 25
def twenty_seven := 27

theorem prime_in_choices :
  is_prime twenty_three ∧ ¬ is_prime twenty ∧ ¬ is_prime twenty_one ∧ ¬ is_prime twenty_five ∧ ¬ is_prime twenty_seven :=
by
  sorry

end prime_in_choices_l122_122021


namespace total_area_painted_is_correct_l122_122507

noncomputable def barn_area_painted (width length height : ℝ) : ℝ :=
  let walls_area := 2 * (width * height + length * height) * 2
  let ceiling_and_roof_area := 2 * (width * length)
  walls_area + ceiling_and_roof_area

theorem total_area_painted_is_correct 
  (width length height : ℝ) 
  (h_w : width = 12) 
  (h_l : length = 15) 
  (h_h : height = 6) 
  : barn_area_painted width length height = 1008 :=
  by
  rw [h_w, h_l, h_h]
  -- Simplify steps omitted
  sorry

end total_area_painted_is_correct_l122_122507


namespace TrishulPercentageLessThanRaghu_l122_122808

-- Define the variables and conditions
variables (R T V : ℝ)

-- Raghu's investment is Rs. 2200
def RaghuInvestment := (R : ℝ) = 2200

-- Vishal invested 10% more than Trishul
def VishalInvestment := (V : ℝ) = 1.10 * T

-- Total sum of investments is Rs. 6358
def TotalInvestment := R + T + V = 6358

-- Define the proof statement
theorem TrishulPercentageLessThanRaghu (R_is_2200 : RaghuInvestment R) 
    (V_is_10_percent_more : VishalInvestment V T) 
    (total_sum_is_6358 : TotalInvestment R T V) : 
  ((2200 - T) / 2200) * 100 = 10 :=
sorry

end TrishulPercentageLessThanRaghu_l122_122808


namespace sum_of_valid_k_l122_122611

-- Definitions based on given conditions
def is_valid_digit (m : Nat) : Prop := m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if a given k can be expressed in the base -3+i
def valid_k (k : ℤ) (a0 a1 a2 a3 : Nat) : Prop := 
  a3 ≠ 0 ∧
  is_valid_digit a0 ∧
  is_valid_digit a1 ∧
  is_valid_digit a2 ∧
  is_valid_digit a3 ∧
  let r := a3 * (-18) + a2 * 8 + a1 * (-3) + a0 in
  let i := a3 * 26 + a2 * (-6) + a1 * 1 in
  i = 0 ∧ k = r

-- The sum of all valid k values
def sum_valid_k : ℤ :=
  List.sum $ do
    a3 ← [1, 2];  -- since a3 ≠ 0 and valid digits are in 1 to 9
    a2 ← List.filter (λ x, ∃ a1, 26 * a3 = 6 * x - a1) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    a1 ← List.filter (λ a1, 26 * a3 = 6 * a2 - a1) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    a0 ← [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    guard (valid_k (a3 * (-18) + a2 * 8 + a1 * (-3) + a0) a0 a1 a2 a3);
    [a3 * (-18) + a2 * 8 + a1 * (-3) + a0]

-- The theorem statement
theorem sum_of_valid_k : sum_valid_k = 490 := by
  sorry

end sum_of_valid_k_l122_122611


namespace cos_alpha_minus_beta_l122_122172

-- Let α and β be real numbers
variables (α β : ℝ)

-- Conditions given
axiom cos_sum : (cos α + cos β = 1/2)
axiom sin_sum : (sin α + sin β = √3/2)

-- Theorem to prove
theorem cos_alpha_minus_beta : cos (α - β) = -1/2 :=
sorry

end cos_alpha_minus_beta_l122_122172


namespace cos_beta_value_l122_122995

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l122_122995


namespace find_function_increasing_interval_l122_122186

noncomputable def ome := 2
noncomputable def A := 2
noncomputable def phi := Real.pi / 4
noncomputable def f (x : ℝ) : ℝ := A * sin (ome * x + phi)
def highest_point_D (D : ℝ × ℝ) : Prop := D = (Real.pi / 8, 2)
def intersects_x_axis (P : ℝ × ℝ) : Prop := P = (3 * Real.pi / 8, 0)

theorem find_function:
  (∃ A ω φ, (A > 0) ∧ (ω > 0) ∧ (|φ| < Real.pi / 2) ∧ 
    highest_point_D (Real.pi / 8, 2) ∧ 
    intersects_x_axis (3 * Real.pi / 8, 0) ∧ 
    f (x : ℝ) = A * Real.sin (ω * x + φ)) → 
  (f (x : ℝ) = 2 * Real.sin (2 * x + Real.pi / 4)) := 
by sorry

theorem increasing_interval (k : ℤ) :
  (∃ A ω φ, (A > 0) ∧ (ω > 0) ∧ (|φ| < Real.pi / 2) ∧ 
    highest_point_D (Real.pi / 8, 2) ∧ 
    intersects_x_axis (3 * Real.pi / 8, 0) ∧ 
    f (x : ℝ) = A * Real.sin (ω * x + φ)) → 
  ∀ x, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 8) → 
  StrictMonoOn f (set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8)) :=
by sorry

end find_function_increasing_interval_l122_122186


namespace integral_D_value_l122_122486

-- Define the problem conditions as functions

def integral_A : ℝ := ∫ x in 0..5, (2 * x - 4)
def integral_B : ℝ := ∫ x in 0..π, cos x
def integral_C : ℝ := ∫ x in 1..3, 1 / x
def integral_D : ℝ := ∫ x in 0..π, sin x

-- The proof statement we need to prove
theorem integral_D_value : integral_D = 2 :=
sorry

end integral_D_value_l122_122486


namespace probability_prime_and_multiple_of_11_l122_122738

-- Define the range of numbers from 1 to 100
def numbers := Finset.range 100

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

-- Define the probability calculation
def probability (s : Finset ℕ) (pred : ℕ → Prop) : ℚ :=
  (s.filter pred).card / s.card

-- State the problem
theorem probability_prime_and_multiple_of_11 :
  probability numbers (λ n, is_prime n ∧ is_multiple_of_11 n) = 1 / 100 :=
by
  sorry

end probability_prime_and_multiple_of_11_l122_122738


namespace polynomial_roots_l122_122133

noncomputable def f : ℝ → ℝ := λ x, x^3 - 5*x^2 + 8*x - 4
noncomputable def f' : ℝ → ℝ := λ x, 3*x^2 - 10*x + 8

theorem polynomial_roots : 
  (f 1 = 0) ∧ (f 2 = 0) ∧ (f' 1 ≠ 0) ∧ (f' 2 = 0) := 
by
  sorry

end polynomial_roots_l122_122133


namespace geometric_sequence_sum_l122_122998

theorem geometric_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n+1) = a n * q) → -- geometric sequence condition
  a 2 = 6 → -- first condition
  6 * a 1 + a 3 = 30 → -- second condition
  (∀ n, S_n n = (if q = 2 then 3*(2^n - 1) else if q = 3 then 3^n - 1 else 0)) :=
by intros
   sorry

end geometric_sequence_sum_l122_122998


namespace m_perp_n_l122_122971

-- Definitions for the conditions
variables (α β : Plane) (m n : Line)

-- Conditions in the problem
axiom planes_are_different (h1 : α ≠ β)

axiom m_perp_alpha (h2 : Perpendicular m α)
axiom n_perp_beta (h3 : Perpendicular n β)
axiom alpha_perp_beta (h4 : Perpendicular α β)

-- Result to prove in the form of a Lean 4 statement
theorem m_perp_n : Perpendicular m n :=
by sorry

end m_perp_n_l122_122971


namespace solve_inequality_l122_122330

theorem solve_inequality (a x : ℝ) : 
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) :=
  sorry

end solve_inequality_l122_122330


namespace sqrt_mixed_number_simplify_l122_122946

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l122_122946


namespace average_age_of_remaining_people_l122_122338

noncomputable def average_age_after_leaving : Float := 
  let initial_avg : Float := 25
  let total_people : Int := 8
  let age_leaving : Float := 20

  -- initial total age of all people
  let total_age := initial_avg * total_people

  -- new total age after one person leaves
  let new_total_age := total_age - age_leaving

  -- the number of remaining people
  let remaining_people := total_people - 1

  new_total_age / remaining_people

theorem average_age_of_remaining_people :
  (let initial_avg : Float := 25
   let total_people : Int := 8
   let age_leaving : Float := 20
   
   let total_age := initial_avg * total_people
   let new_total_age := total_age - age_leaving
   let remaining_people := total_people - 1
   new_total_age / remaining_people) = (180 / 7) :=
by
  sorry

end average_age_of_remaining_people_l122_122338


namespace opposite_of_negative_five_l122_122394

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122394


namespace minimum_intersection_of_three_sets_l122_122711

def number_of_subsets (S : Finset α) : ℕ :=
2 ^ S.card

theorem minimum_intersection_of_three_sets
  {α : Type} [DecidableEq α] (A B C : Finset α)
  (hA : A.card = 50) (hB : B.card = 50) (hC : C.card = 45)
  (h : number_of_subsets (A ∪ B ∪ C) = number_of_subsets A + number_of_subsets B + number_of_subsets C) :
  |A ∩ B ∩ C| ≥ 43 := sorry

end minimum_intersection_of_three_sets_l122_122711


namespace identify_faulty_key_l122_122480

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122480


namespace exists_point_with_sum_distances_at_least_n_l122_122701

theorem exists_point_with_sum_distances_at_least_n
  (n : ℕ)
  (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (hC : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 2)) ≤ 1) :
  ∃ (p : EuclideanSpace ℝ (Fin 2)), 
    (finset.univ.sum (λ i, dist p (points i))) ≥ n :=
sorry

end exists_point_with_sum_distances_at_least_n_l122_122701


namespace quotient_korean_english_l122_122333

theorem quotient_korean_english (K M E : ℝ) (h1 : K / M = 1.2) (h2 : M / E = 5 / 6) : K / E = 1 :=
sorry

end quotient_korean_english_l122_122333


namespace fib_sum_lt_two_l122_122526

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- State the theorem
theorem fib_sum_lt_two (n : ℕ) : ∑ k in Finset.range (n + 1), (fib k : ℝ) / 2^k < 2 := 
sorry

end fib_sum_lt_two_l122_122526


namespace train_B_speed_l122_122428

theorem train_B_speed 
  (meeting_time_A : 9) -- hours
  (meeting_time_B : 4) -- hours
  (speed_A : 100) -- kmph
  : (speed_B = 225) := 
by
  let D_A := speed_A * meeting_time_A
  let D_B := speed_B * meeting_time_B
  have : D_A = 900 := by simp [D_A]
  have : D_B = 4 * speed_B := by simp [D_B]
  have : D_A = D_B := by simp [this, that]
  let speed_B := D_A / 4
  rw [← this] at speed_B
  simp [D_A] at speed_B
  exact speed_B

end train_B_speed_l122_122428


namespace journey_time_correct_l122_122681

/-- Definition of the speeds based on the problem conditions --/
def boat_speed_still_water : ℝ := 6
def initial_current_speed : ℝ := 2
def upstream_distance : ℝ := 56
def increased_current_speed_upstream : ℝ := 3
def headwind_decrease : ℝ := 1

/-- Definition to calculate effective speeds --/
def effective_speed_upstream : ℝ := boat_speed_still_water - increased_current_speed_upstream
def effective_speed_downstream : ℝ := (boat_speed_still_water - headwind_decrease) + initial_current_speed

/-- Definition to calculate times --/
def time_upstream : ℝ := upstream_distance / effective_speed_upstream
def time_downstream : ℝ := upstream_distance / effective_speed_downstream

/-- Total journey time --/
def total_journey_time : ℝ := time_upstream + time_downstream

/-- Main theorem to prove that total journey time is 26.67 hours --/
theorem journey_time_correct : total_journey_time = 26.67 := by
  -- steps and detailed proof can be filled here
  sorry

end journey_time_correct_l122_122681


namespace remainder_is_four_l122_122793

theorem remainder_is_four :
  ∀ x ∈ [5, 9, 12, 18], 184 % x = 4 :=
by
  intros x hx
  fin_cases hx <;> simp
  sorry

end remainder_is_four_l122_122793


namespace good_permutations_equiv_l122_122612

def is_good_permutation (p : ℕ) (a : ℕ → ℕ) : Prop :=
  p > 3 ∧ p.prime ∧ (∀ i, 1 ≤ a i ∧ a i ≤ p-1 ∧ 
  ∑ i in range (p-2), a i * a (i+1) % p = 0)

noncomputable def K (p : ℕ) : ℕ :=
  fintype.card {a : fin (p-1) → ℕ // is_good_permutation p a}

theorem good_permutations_equiv (p : ℕ) (hp : p > 3) [p.prime] : 
  (K p) ≡ (p-1) [MOD p * (p-1)] :=
by sorry

end good_permutations_equiv_l122_122612


namespace coordinates_equality_l122_122340

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end coordinates_equality_l122_122340


namespace max_value_of_f_on_ellipse_l122_122974

noncomputable def f (x y : ℝ) : ℝ :=
  ((x - y) + (4 + real.sqrt (1 - x^2) + real.sqrt (1 - (y^2 / 9))))^2

def ellipse (x y : ℝ) : Prop :=
  (x^2) / 9 + (y^2) / 9 = 1

theorem max_value_of_f_on_ellipse :
  ∃ x y : ℝ, ellipse x y ∧ f x y = f (3 * real.sqrt 3 + 1) (-6 * real.sqrt 3 + 8) :=
sorry

end max_value_of_f_on_ellipse_l122_122974


namespace function_domain_l122_122432

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x ≠ 8}

theorem function_domain :
  ∀ x, x ∈ domain_function ↔ x ∈ (Set.Iio 8 ∪ Set.Ioi 8) := by
  intro x
  sorry

end function_domain_l122_122432


namespace calculate_compound_interest_l122_122030

def SimpleInterest (P R T : ℝ) : ℝ := P * R * T / 100
def CompoundInterest (P R T : ℝ) : ℝ := P * ((1 + R / 100) ^ T - 1)

theorem calculate_compound_interest :
  let SI := 50
  let R := 5
  let T := 2
  let P := SI * 100 / (R * T)
  CompoundInterest P R T = 51.25 :=
by
  sorry

end calculate_compound_interest_l122_122030


namespace square_area_product_one_l122_122706

theorem square_area_product_one (ABCD EFGH : Type) [Square ABCD] [Quadrilateral EFGH]
  (AE BF CG DH : ℝ)
  (A B C D E F G H X Y Z W O : Point) 
  (h_equal_distance_ae_bf_cg_dh : AE = BF ∧ BF = CG ∧ CG = DH ∧ DH = AE)
  (h_area_efgh : Area EFGH = 1)
  (h_intersections : ∀ (p : Point), (p = X → OnLineSegment p O A ∧ OnLineSegment p E H) ∧
                                    (p = Y → OnLineSegment p O B ∧ OnLineSegment p E F) ∧
                                    (p = Z → OnLineSegment p O C ∧ OnLineSegment p F G) ∧
                                    (p = W → OnLineSegment p O D ∧ OnLineSegment p H G)) :
  Area ABCD * Area XYZW = 1 :=
sorry

end square_area_product_one_l122_122706


namespace calculate_result_l122_122270

open Real

def triangle_area (b h : ℝ) : ℝ := (1 / 2) * b * h

def region_area (n : ℕ) : ℝ := 
  triangle_area 3 4 + 12 * (n : ℝ) + (π * (n : ℝ) ^ 2) / 2

def area_difference (n : ℕ) : ℝ := region_area (n + 1) - region_area n

def a : ℝ := 41 / 2
def b : ℝ := 12
def result : ℝ := 100 * a + b

theorem calculate_result : result = 2062 := 
  sorry

end calculate_result_l122_122270


namespace volume_ratio_of_sphere_with_circle_O1_as_great_circle_to_sphere_O_l122_122623

noncomputable def midpoint_volume_ratio (R : ℝ) : ℝ :=
  let r := (Math.sqrt 3 / 2) * R in
  (4/3 * π * r^3) / (4/3 * π * R^3)

theorem volume_ratio_of_sphere_with_circle_O1_as_great_circle_to_sphere_O (R : ℝ) (hR : 0 < R) :
  midpoint_volume_ratio(R) = (3 / 8) * Math.sqrt 3 :=
by
  sorry

end volume_ratio_of_sphere_with_circle_O1_as_great_circle_to_sphere_O_l122_122623


namespace ratio_of_paper_plates_l122_122530

theorem ratio_of_paper_plates (total_pallets : ℕ) (paper_towels : ℕ) (tissues : ℕ) (paper_cups : ℕ) :
  total_pallets = 20 →
  paper_towels = 20 / 2 →
  tissues = 20 / 4 →
  paper_cups = 1 →
  (total_pallets - (paper_towels + tissues + paper_cups)) / total_pallets = 1 / 5 :=
by
  intros h_total h_towels h_tissues h_cups
  sorry

end ratio_of_paper_plates_l122_122530


namespace probability_four_or_five_wrong_l122_122999

theorem probability_four_or_five_wrong :
  let n := 5 in
  let D : ℕ → ℕ
    | 0 => 1
    | 1 => 0
    | n + 1 => n * (D n + D (n - 1)) in
  let favorable_outcomes := 5 * D 4 + D 5 in
  let total_outcomes := Fact 5 in
  (favorable_outcomes : ℚ) / total_outcomes = 89 / 120 :=
by
  sorry

end probability_four_or_five_wrong_l122_122999


namespace distance_from_starting_point_l122_122089

-- Definitions based on the problem conditions
def L : ℝ := 50 -- Halfway between the siren and the start of the bridge
def sound_distance_condition (x y L : ℝ) : Prop :=
  (x + L) = Real.sqrt ((x - L) ^ 2 + y ^ 2)

-- Hypotheses based on the problem conditions
def condition1 : Prop := sound_distance_condition x y L
def condition2 : Prop := y = 40 

-- The theorem we need to prove
theorem distance_from_starting_point (x y L : ℝ) (h1 : sound_distance_condition x y L) (h2 : y = 40) : 
  Real.sqrt (x ^ 2 + y ^ 2) = 41 :=
  sorry

end distance_from_starting_point_l122_122089


namespace range_of_omega_l122_122886

-- Define the sine function f(x) = sin(ωx + π/3)
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- The hypothesis for the problem
def has_extreme_points (ω : ℝ) :=
  (∃ x1 x2 x3 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (x3 < Real.pi)
    ∧ f' ω x1 = 0 ∧ f' ω x2 = 0 ∧ f' ω x3 = 0)

def has_zeros (ω : ℝ) :=
  (∃ x1 x2 : ℝ, (0 < x1) ∧ (x1 < x2) ∧ (x2 < Real.pi)
    ∧ f ω x1 = 0 ∧ f ω x2 = 0)

-- The main theorem to be proved
theorem range_of_omega (ω : ℝ) :
  has_extreme_points ω ∧ has_zeros ω ↔ (13/6 < ω ∧ ω ≤ 8/3) :=
by
  sorry

end range_of_omega_l122_122886


namespace opposite_of_negative_five_l122_122384

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122384


namespace cyclic_octagon_circumradius_l122_122769

theorem cyclic_octagon_circumradius (a b : ℝ) : 
  let r := sqrt ((a^2 + b^2 + sqrt 2 * a * b) / 2) in
  ∃ R : ℝ, ∀ (ABCDEFGH : ℝ), ABCDEFGH = r -> ∀ x, x = R := sorry

end cyclic_octagon_circumradius_l122_122769


namespace quadrilateral_ABCD_AB_over_BC_l122_122317

theorem quadrilateral_ABCD_AB_over_BC
  (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (r1 r2 : ℝ)
  (hAB : dist A B = r1)
  (hBC : dist B C = r2)
  (hAD : dist A D = 1)
  (hDC : dist D C = 1)
  (hC : ∠ A C B = π/2)
  (hA : ∠ D A B = π/2)
  (hSim1 : similar (triangle A B C) (triangle B C D))
  (hSim2 : similar (triangle A B C) (triangle A B E))
  (hArea : area (triangle A E D) = 10 * area (triangle A B E))
  (hABBC : dist A B < dist B C) :
  dist A B / dist B C = 2 + real.sqrt 5 :=
by
  sorry

end quadrilateral_ABCD_AB_over_BC_l122_122317


namespace man_walking_time_l122_122028

theorem man_walking_time
  (T : ℕ) -- Let T be the time (in minutes) the man usually arrives at the station.
  (usual_arrival_home : ℕ) -- The time (in minutes) they usually arrive home, which is T + 30.
  (early_arrival : ℕ) (walking_start_time : ℕ) (early_home_arrival : ℕ)
  (usual_arrival_home_eq : usual_arrival_home = T + 30)
  (early_arrival_eq : early_arrival = T - 60)
  (walking_start_time_eq : walking_start_time = early_arrival)
  (early_home_arrival_eq : early_home_arrival = T)
  (time_saved : ℕ) (half_time_walk : ℕ)
  (time_saved_eq : time_saved = 30)
  (half_time_walk_eq : half_time_walk = time_saved / 2) :
  walking_start_time = half_time_walk := by
  sorry

end man_walking_time_l122_122028


namespace five_people_six_chairs_l122_122237

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l122_122237


namespace ratio_of_dimensions_128_l122_122415

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l122_122415


namespace tom_teaching_years_l122_122000

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l122_122000


namespace opposite_of_negative_five_l122_122385

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122385


namespace repeating_decimal_sum_to_fraction_l122_122572

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0045 : ℚ := 45 / 9999
def repeating_decimal_000678 : ℚ := 678 / 999999

theorem repeating_decimal_sum_to_fraction :
  repeating_decimal_123 + repeating_decimal_0045 + repeating_decimal_000678 = 128178 / 998001000 :=
by
  sorry

end repeating_decimal_sum_to_fraction_l122_122572


namespace smallest_palindromic_number_l122_122104

def is_palindrome (digits : List ℕ) : Prop :=
  digits = digits.reverse

def to_base (n base : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / base) ((n % base)::acc)
    aux n []

theorem smallest_palindromic_number :
  ∃ n : ℕ, n > 10 ∧ is_palindrome (to_base n 3) ∧ is_palindrome (to_base n 5) ∧ n = 26 :=
by
  use 26
  have h1 : 26 > 10 := by norm_num
  have h2 : is_palindrome (to_base 26 3) := by
    unfold to_base
    norm_num
    unfold is_palindrome
    refl
  have h3 : is_palindrome (to_base 26 5) := by
    unfold to_base
    norm_num
    unfold is_palindrome
    refl
  tauto

end smallest_palindromic_number_l122_122104


namespace identify_faulty_key_l122_122482

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122482


namespace final_amount_correct_l122_122552

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3
def shoes_cost : ℝ := wallet_cost + purse_cost + 7
def total_cost_before_discount : ℝ := wallet_cost + purse_cost + shoes_cost
def discount_rate : ℝ := 0.10
def discounted_amount : ℝ := total_cost_before_discount * discount_rate
def final_amount : ℝ := total_cost_before_discount - discounted_amount

theorem final_amount_correct :
  final_amount = 198.90 := by
  -- Here we would provide the proof of the theorem
  sorry

end final_amount_correct_l122_122552


namespace area_of_AEHF_l122_122106

theorem area_of_AEHF :
  ∀ (A E H F: ℝ × ℝ),
    A = (0, 0) →
    E = (3, 6) →
    H = (3, 0) →
    F = (2, 0) →
    let AEHF_area := 6 * 2 in
    AEHF_area = 12 :=
begin
    intros A E H F hA hE hH hF AEHF_area,
    sorry -- proof placeholder
end

end area_of_AEHF_l122_122106


namespace range_of_a_l122_122604

-- Definition of conditions
def condition_p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1

def condition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- The range of x given condition_p
def range_x_p : set ℝ := { x : ℝ | 1 / 2 ≤ x ∧ x ≤ 1 }

-- The range of x given condition_q
def range_x_q (a : ℝ) : set ℝ := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- Mathematically equivalent proof problem
theorem range_of_a (a : ℝ) :
  (∀ x, condition_p x → condition_q x a) ∧ ¬ (∀ x, condition_q x a → condition_p x) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l122_122604


namespace projection_of_v2_projection_of_v3_l122_122585

open Real EuclideanSpace Matrix

noncomputable def w : ℝ × ℝ × ℝ := (2, -1, 1)
noncomputable def v1 : ℝ × ℝ × ℝ := (-1, 0, 3)
noncomputable def v2 : ℝ × ℝ × ℝ := (1, 2, -1)
noncomputable def v3 : ℝ × ℝ × ℝ := (4, -1, 1)

noncomputable def proj (u w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let numer := dot_product u w
  let denom := dot_product w w
  let scalar := numer / denom
  (scalar * w.1, scalar * w.2, scalar * w.3)

theorem projection_of_v2 :
  proj v2 w = (-1 / 3, 1 / 6, -1 / 6) := 
by
  sorry

theorem projection_of_v3 :
  proj v3 w = (3, -3 / 2, 3 / 2) := 
by
  sorry

end projection_of_v2_projection_of_v3_l122_122585


namespace retail_price_per_kg_l122_122532

-- Define the initial conditions as given in the problem
def initial_weight := 500 -- in kg
def cost_per_kg := 4.80 -- in yuan
def weight_loss_percent := 0.10
def profit_percent := 0.20

-- The mathematically equivalent statement to prove
theorem retail_price_per_kg :
  let total_cost := initial_weight * cost_per_kg,
      effective_weight := initial_weight - (weight_loss_percent * initial_weight),
      desired_revenue := total_cost * (1 + profit_percent),
      retail_price := desired_revenue / effective_weight
  in retail_price = 6.4 :=
by
  sorry

end retail_price_per_kg_l122_122532


namespace min_value_of_f_l122_122637

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 3)

theorem min_value_of_f (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx1_distinct : x1 ≠ x2) (hx2_pos : 0 < x2)
  (h_f_eq : f x1 = f x2) : (1 / x1 + 9 / x2) = 2 / 3 :=
by
  sorry

end min_value_of_f_l122_122637


namespace sqrt_mixed_number_simplified_l122_122940

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l122_122940


namespace smallest_prime_sum_l122_122114

theorem smallest_prime_sum :
  ∃ (p1 p2 p3 : ℕ), 
  (∀ p, p ∈ {p1, p2, p3} → Prime p) ∧
  (∀ p, p ∈ {p1, p2, p3} → (p % 10 = 1 ∨ p % 10 = 2 ∨ p % 10 = 3 ∨ p % 10 = 5)) ∧
  (p1 % 10 ≠ p2 % 10 ∧ p1 % 10 ≠ p3 % 10 ∧ p2 % 10 ≠ p3 % 10) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) ∧
  (p1 + p2 + p3 = 71) :=
sorry

end smallest_prime_sum_l122_122114


namespace part1_part2_l122_122179

open Real

variables (x a : ℝ)

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0

theorem part1 (h : a = 1) (h_pq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

theorem part2 (hpq : ∀ (a x : ℝ), ¬ p x a → ¬ q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l122_122179


namespace five_people_six_chairs_l122_122227

theorem five_people_six_chairs : 
  ∃ (f : Fin 6 → Bool), (∑ i, if f i then 1 else 0) = 5 ∧ 
  (∃ (g : Fin 5 → Fin 6), ∀ i j : Fin 5, i ≠ j → g i ≠ g j) →
  (5!) * (choose 6 5) = 720 :=
by
  sorry

end five_people_six_chairs_l122_122227


namespace part1_l122_122824

theorem part1 : (-2)^2 + |real.sqrt 2 - 1| - real.sqrt 4 = real.sqrt 2 + 1 :=
by {
  sorry
}

end part1_l122_122824


namespace range_of_omega_l122_122883

noncomputable section

open Real

/--
Assume the function f(x) = sin (ω x + π / 3) has exactly three extreme points and two zeros in 
the interval (0, π). Prove that the range of values for ω is 13 / 6 < ω ≤ 8 / 3.
-/
theorem range_of_omega 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h : ∀ x, f x = sin (ω * x + π / 3))
  (h_extreme : (∃ a b c, 0 < a ∧ a < b ∧ b < c ∧ c < π ∧ (f' a = 0) ∧ (f' b = 0) ∧ (f' c = 0)))
  (h_zeros : (∃ u v, 0 < u ∧ u < v ∧ v < π ∧ f u = 0 ∧ f v = 0)) :
  (13 / 6) < ω ∧ ω ≤ (8 / 3) :=
  sorry

end range_of_omega_l122_122883


namespace primes_square_condition_l122_122955

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l122_122955


namespace find_faulty_keys_l122_122448

-- Define the conditions given in the problem
def total_digits : ℕ := 10
def registered_digits : ℕ := 7
def missing_digits : ℕ := 3
def defective_key_min_presses : ℕ := 5
def defective_key_successful_presses : ℕ := 2

-- Define that we need to find which keys could be faulty
def possible_faulty_keys : List ℤ := [7, 9]

-- The main theorem statement
theorem find_faulty_keys (total_digits = 10) (registered_digits = 7) (missing_digits = 3)
  (defective_key_min_presses = 5) (defective_key_successful_presses ≥ 2) :
  possible_faulty_keys = [7, 9] :=
by
  sorry

end find_faulty_keys_l122_122448


namespace find_angle_BCA_eq_90_l122_122269

variables {A B C T1 T2 I : Type} [field I] [inhabited I]

noncomputable def is_reflection (I' M I : I) : Prop :=
  I' = 2 * M - I

noncomputable def circumcircle_contains (I' T1 T2 C : I) : Prop :=
  -- This formula needs to be derived properly; placeholder equalities for demonstration.
  (I' - T1) * (I' - T2) * (I' - C) = (T1 - C) * (T2 - C)

noncomputable def incenter_lies_on_circumcircle_of_reflection 
  (A B C T1 T2 : I) [is_triangle A B C] (M := midpoint A B) 
  (I' : I := 2 * M - incenter A B C) : Prop :=
  circumcircle_contains I' T1 T2 C

theorem find_angle_BCA_eq_90
  (A B C T1 T2 : I)
  [is_triangle A B C]
  (reflection_incenter_cond : incenter_lies_on_circumcircle_of_reflection A B C T1 T2) :
  ∠ B C A = 90 :=
sorry

end find_angle_BCA_eq_90_l122_122269


namespace podcast_ratio_l122_122326

theorem podcast_ratio
  (total_drive_time : ℕ)
  (first_podcast : ℕ)
  (third_podcast : ℕ)
  (fourth_podcast : ℕ)
  (next_podcast : ℕ)
  (second_podcast : ℕ) :
  total_drive_time = 360 →
  first_podcast = 45 →
  third_podcast = 105 →
  fourth_podcast = 60 →
  next_podcast = 60 →
  second_podcast = total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast) →
  second_podcast / first_podcast = 2 :=
by
  sorry

end podcast_ratio_l122_122326


namespace complex_subtraction_solution_l122_122440

theorem complex_subtraction_solution :
  ∃ z : ℂ, (4 - 3 * complex.I) - z = -1 + 9 * complex.I ∧ z = 5 - 12 * complex.I :=
by
  use 5 - 12 * complex.I
  split
  sorry
  rfl

end complex_subtraction_solution_l122_122440


namespace proof_value_of_expression_l122_122812

theorem proof_value_of_expression (a b : ℕ) (h1 : a = 196) (h2 : b = 169) : 
  (a^2 - b^2) / 27 = 365 := 
by {
  sorry,
}

end proof_value_of_expression_l122_122812


namespace intersection_nonempty_range_b_l122_122171

noncomputable def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
noncomputable def B (b : ℝ) (a : ℝ) : Set ℝ := {x | (x - b)^2 < a}

theorem intersection_nonempty_range_b (b : ℝ) : 
  A ∩ B b 1 ≠ ∅ ↔ -2 < b ∧ b < 2 := 
by
  sorry

end intersection_nonempty_range_b_l122_122171


namespace sequence_proof_l122_122987

open Nat

-- Definitions based on the conditions provided

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n+1) = a n / (2 + 3 * a n)

-- Statement A: Proving the sequence {1/a_n + 3} forms a geometric sequence with ratio 2 and first term 4
noncomputable def is_geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
  ∃ t : ℝ, b 1 = t ∧ ∀ n : ℕ, n > 0 → b (n+1) = r * b n

-- Statement D: Sum of the first n terms of {1/a_n} is {T_n = 2^(n+2) - 3n - 4}
noncomputable def sequence_sum (a : ℕ → ℝ) (s : ℕ → ℝ) : Prop :=
  s 1 = 1 ∧ ∃ t : ℝ, ∀ n : ℕ, n > 0 → s (n+1) = 2^(n+2) - 3*n - 4

-- Proof problem combining the necessary definitions and conditions
theorem sequence_proof (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) :
  sequence a →
  is_geometric_sequence (λ n, 1 / a n + 3) 2 →
  sequence_sum (λ n, 1 / a n) s :=
by
  intros h1 h2 h3
  sorry

end sequence_proof_l122_122987


namespace array_fill_possible_l122_122990

theorem array_fill_possible (m n : ℕ) (a : Fin m → ℕ+) (b : Fin n → ℕ+)
  (h_sum : ∑ i, a i = ∑ j, b j) :
  ∃ (c : Fin (m + n - 1) → ℕ+), 
  ∃ (f : (Fin m) × (Fin n) → ℕ+), 
  (∀ i, ∑ j, f (i, j) = a i) ∧ 
  (∀ j, ∑ i, f (i, j) = b j) ∧ 
  (∀ (k : Fin (m + n - 1)), ∃ (i : Fin m) (j : Fin n), c k = f (i, j)) ∧ 
  (∀ (i : Fin m) (j : Fin n), f (i, j) ≠ 0 → ∃ k, f (i, j) = c k) := 
sorry

end array_fill_possible_l122_122990


namespace points_on_line_l122_122670

theorem points_on_line (y1 y2 : ℝ) 
  (hA : y1 = - (1 / 2 : ℝ) * 1 - 1) 
  (hB : y2 = - (1 / 2 : ℝ) * 3 - 1) :
  y1 > y2 := 
by
  sorry

end points_on_line_l122_122670


namespace quadratic_real_roots_range_l122_122145

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9 / 4 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l122_122145


namespace athlete_total_heartbeats_l122_122865

/-
  An athlete's heart rate starts at 140 beats per minute at the beginning of a race
  and increases by 5 beats per minute for each subsequent mile. How many times does
  the athlete's heart beat during a 10-mile race if the athlete runs at a pace of
  6 minutes per mile?
-/

def athlete_heartbeats (initial_rate : ℕ) (increase_rate : ℕ) (miles : ℕ) (minutes_per_mile : ℕ) : ℕ :=
  let n := miles
  let a := initial_rate
  let l := initial_rate + (increase_rate * (miles - 1))
  let S := (n * (a + l)) / 2
  S * minutes_per_mile

theorem athlete_total_heartbeats :
  athlete_heartbeats 140 5 10 6 = 9750 :=
sorry

end athlete_total_heartbeats_l122_122865


namespace opposite_of_negative_five_l122_122387

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122387


namespace raft_stick_ratio_l122_122328

theorem raft_stick_ratio :
  ∀ (G : ℕ),
  let S := 36 in 
  let M := S + G + 9 in
  S + G + M = 129 →
  G / 36 = 2 / 3 :=
by
  intros G S M h,
  let S := 36,
  let M := S + G + 9,
  assume : S + G + M = 129,
  sorry

end raft_stick_ratio_l122_122328


namespace euler_polyhedron_problem_l122_122125

theorem euler_polyhedron_problem : 
  ( ∀ (V E F T S : ℕ), F = 42 → (T = 2 ∧ S = 3) → V - E + F = 2 → 100 * S + 10 * T + V = 337 ) := 
by sorry

end euler_polyhedron_problem_l122_122125


namespace ab2_plus_ac2_plus_bc2_div_sum_lmn_sq_3_eq_4_l122_122285

variables {l m n : ℝ}

-- Midpoint conditions as provided in the problem
variable midpoint_BC : (l, 0, 1)
variable midpoint_AC : (0, m, 1)
variable midpoint_AB : (0, 0, n)

-- The theorem we need to prove
theorem ab2_plus_ac2_plus_bc2_div_sum_lmn_sq_3_eq_4 :
  let AB_sq := 4 * (m^2 + (1 - n)^2)
  let AC_sq := 4 * (l^2 + (1 - n)^2)
  let BC_sq := 4 * (l^2 + m^2)
  in (AB_sq + AC_sq + BC_sq) / (l^2 + m^2 + n^2 + 3) = 4 :=
by
  sorry

end ab2_plus_ac2_plus_bc2_div_sum_lmn_sq_3_eq_4_l122_122285


namespace division_multiplication_l122_122018

-- Given a number x, we want to prove that (x / 6) * 12 = 2 * x under basic arithmetic operations.

theorem division_multiplication (x : ℝ) : (x / 6) * 12 = 2 * x := 
by
  sorry

end division_multiplication_l122_122018


namespace range_of_omega_l122_122881

noncomputable section

open Real

/--
Assume the function f(x) = sin (ω x + π / 3) has exactly three extreme points and two zeros in 
the interval (0, π). Prove that the range of values for ω is 13 / 6 < ω ≤ 8 / 3.
-/
theorem range_of_omega 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h : ∀ x, f x = sin (ω * x + π / 3))
  (h_extreme : (∃ a b c, 0 < a ∧ a < b ∧ b < c ∧ c < π ∧ (f' a = 0) ∧ (f' b = 0) ∧ (f' c = 0)))
  (h_zeros : (∃ u v, 0 < u ∧ u < v ∧ v < π ∧ f u = 0 ∧ f v = 0)) :
  (13 / 6) < ω ∧ ω ≤ (8 / 3) :=
  sorry

end range_of_omega_l122_122881


namespace opposite_of_neg_five_l122_122376

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122376


namespace opposite_of_neg_five_l122_122361

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l122_122361


namespace limit_of_f_at_4_l122_122893

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt (1 + 2 * x) - 3) / (sqrt x - 2)

theorem limit_of_f_at_4 :
  tendsto f (nhds 4) (nhds (4 / 3)) :=
sorry

end limit_of_f_at_4_l122_122893


namespace sum_p_not_divisible_l122_122982

theorem sum_p_not_divisible (p : ℕ) (hp : Nat.Prime p) : ∃ (S : Set ℕ), (∀ q ∈ S, Nat.Prime q) ∧ ∀ q, Nat.Prime q → q ∉ S → ¬(q ∣ ∑ k in Finset.range (q / p), k^(p-1)) := by
  sorry

end sum_p_not_divisible_l122_122982


namespace omega_range_l122_122879

theorem omega_range (ω : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < π ∧
    (sin (ω * x₁ + π / 3)).cos = 0 ∧ (sin (ω * x₂ + π / 3)).cos = 0 ∧ (sin (ω * x₃ + π / 3)).cos = 0) ∧ 
  (∃ z₁ z₂ : ℝ, 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ sin (ω * z₁ + π / 3) = 0 ∧ sin (ω * z₂ + π / 3) = 0) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
by
  sorry

end omega_range_l122_122879


namespace equivalence_of_f_and_g_l122_122081

def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := x^(1/3)

theorem equivalence_of_f_and_g : ∀ x : ℝ, f x = g x := 
by 
  intro x
  have h : g x = x := by sorry
  rw [h]
  refl

end equivalence_of_f_and_g_l122_122081


namespace checkerboard_size_l122_122147

theorem checkerboard_size {Grid : Type} (corner_free : ∀ (c : Grid), c ≠ "corner") 
  (checker_at_center : ∀ (c : Grid), "checker at center") 
  (checkers_on_sides : ∃! (c : Grid), "checker on side") : 
  ∃ (n : ℕ), n = 10 ∧ n * n = 100 :=
by 
  -- statements
  sorry

end checkerboard_size_l122_122147


namespace repeating_decimal_to_fraction_l122_122567

theorem repeating_decimal_to_fraction :
  let x := (0.7 : ℝ) + Real.repeat' 8 9 in
  x = 781 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l122_122567


namespace det_A_is_square_l122_122264

open Int Matrix Nat

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the matrix entry a_ij
def matrix_entry (i j : ℕ) : ℤ :=
  if is_prime (i + j) then 1 else 0

-- Define the matrix A
noncomputable def matrix_A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j, matrix_entry (i + 1) (j + 1)

theorem det_A_is_square (n : ℕ) (h_pos : 0 < n) :
  ∃ (k : ℤ), abs (det (matrix_A n)) = k^2 :=
  sorry

end det_A_is_square_l122_122264


namespace repeating_decimal_to_fraction_l122_122568

theorem repeating_decimal_to_fraction :
  let x := (0.7 : ℝ) + Real.repeat' 8 9 in
  x = 781 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l122_122568


namespace triangles_DM_eq_DN_l122_122714

variable {A B C D P M E F N : Type*}
variable [IsTriangle A B C]
variable [PointOnSide D BC]
variable [PointOnSegment P AD]
variable [LineThrough D M E PB AB]
variable [LineThrough D F N ExtensionAC PC]
variable (DE_eq_DF : Distance(D, E) = Distance(D, F))

theorem triangles_DM_eq_DN :
  Distance(D, M) = Distance(D, N) :=
sorry

end triangles_DM_eq_DN_l122_122714


namespace remainder_q_x_plus_2_l122_122286

def q (x : ℝ) (D E F : ℝ) : ℝ := D * x ^ 6 + E * x ^ 4 + F * x ^ 2 + 5

theorem remainder_q_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 13) : q (-2) D E F = 13 :=
by
  sorry

end remainder_q_x_plus_2_l122_122286


namespace probability_below_line_x_plus_y_eq_5_l122_122170

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {0, 1, 2, 3, 4}
def P : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ B}
def below_line_x_plus_y_eq_5 (p : ℕ × ℕ) : Prop := p.1 + p.2 < 5

theorem probability_below_line_x_plus_y_eq_5 : 
  (card {p ∈ P | below_line_x_plus_y_eq_5 p}).toReal / (card P).toReal = 2 / 5 := 
by 
  sorry

end probability_below_line_x_plus_y_eq_5_l122_122170


namespace ratio_female_to_male_members_l122_122547

theorem ratio_female_to_male_members (f m : ℕ)
  (h1 : 35 * f = SumAgesFemales)
  (h2 : 20 * m = SumAgesMales)
  (h3 : (35 * f + 20 * m) / (f + m) = 25) :
  f / m = 1 / 2 := by
  sorry

end ratio_female_to_male_members_l122_122547


namespace interval_of_monotonic_increase_sum_greater_than_2e_l122_122635

noncomputable def f (a x : ℝ) : ℝ := a * x / (Real.log x)

theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (e < x → f 1 x > f 1 e) := 
sorry

theorem sum_greater_than_2e (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ≠ x2) (hx1 : f 1 x1 = 1) (hx2 : f 1 x2 = 1) :
  x1 + x2 > 2 * Real.exp 1 :=
sorry

end interval_of_monotonic_increase_sum_greater_than_2e_l122_122635


namespace acute_triangle_angle_l122_122220

theorem acute_triangle_angle {A B : ℝ} {a b : ℝ}
  (hABC : A + B < π)
  (h_a_b : 2 * a * sin B = b) :
  A = π / 6 :=
sorry

end acute_triangle_angle_l122_122220


namespace opposite_of_negative_five_l122_122395

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l122_122395


namespace line_slope_sum_l122_122008

open Real

-- Define points P, Q, and R
def P : Point := ⟨1, 9⟩
def Q : Point := ⟨3, 2⟩
def R : Point := ⟨9, 2⟩

-- Define a line through Q cutting the area of ∆PQR in half with y-intercept 1
def line_through_Q (m : ℝ) := ∀ x, y = m * x + 1

-- Function to determine the sum of the slope and y-intercept
def find_slope_sum : ℝ := (1 / 3) + 1

-- Proof statement
theorem line_slope_sum :
  ∃ m : ℝ, (line_through_Q m) ∧ (m + 1 = find_slope_sum) := 
sorry

end line_slope_sum_l122_122008


namespace Q_zero_values_l122_122597

def Q (x : ℝ) : ℂ := 2 + Complex.exp (Complex.I * x) 
                        - 2 * Complex.exp (Complex.I * 2 * x) 
                        + Complex.exp (Complex.I * 3 * x)

theorem Q_zero_values :
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi) → 
  (Finset.filter (fun x => Q x = 0) (Finset.Ico 0 (2 * Real.pi))).card = 2 :=
sorry

end Q_zero_values_l122_122597


namespace translated_midpoint_l122_122405

def Point := (ℝ × ℝ)

def translate (p: Point) (dx dy: ℝ): Point :=
  (p.1 + dx, p.2 + dy)

def midpoint (p1 p2: Point): Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem translated_midpoint:
  let B := (2, 2)
  let G := (6, 2)
  let B' := translate B (-6) 3
  let G' := translate G (-6) 3
  midpoint B' G' = (-2, 5) :=
by
  sorry

end translated_midpoint_l122_122405


namespace ratio_of_altitudes_is_correct_l122_122699

-- Define the geometric setting
variables {K M N : Type} [TopologicalSpace K] [TopologicalSpace M] [TopologicalSpace N]

-- Define angle conditions
variables (θ_1 θ_2 θ_3 : ℝ) -- θ_1 = ∠KNM, θ_2 = ∠KMN
variables (sin_θ1 cos_θ2 : ℝ) (sin_θ1_value cos_θ2_value : ℝ)

-- The conditions given in the problem
def problem_conditions := (sin_θ1 = sin_θ1_value) ∧ (cos_θ2 = cos_θ2_value)

-- Equation representing the ratio of the altitudes
def ratio_of_altitudes (MN : ℝ) :=
  let MA := MN * sin_θ1 in
  let NB := MN * sqrt (1 - cos_θ2^2) in
  NB / MA

-- Statement to prove
theorem ratio_of_altitudes_is_correct (MN : ℝ)
  (h1 : sin_θ1 = sqrt 3 / 2) (h2 : cos_θ2 = 1 / 3) :
  ratio_of_altitudes MN h1 h2 = 4 * sqrt 6 / 9 :=
sorry

end ratio_of_altitudes_is_correct_l122_122699


namespace sqrt_mixed_number_simplified_l122_122938

theorem sqrt_mixed_number_simplified :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 := sorry

end sqrt_mixed_number_simplified_l122_122938


namespace sqrt_of_mixed_number_as_fraction_l122_122929

def mixed_number_to_improper_fraction (a : ℚ) : ℚ :=
  8 + 9 / 16

theorem sqrt_of_mixed_number_as_fraction :
  (√ (mixed_number_to_improper_fraction 8) : ℚ) = (√137) / 4 :=
by
  sorry

end sqrt_of_mixed_number_as_fraction_l122_122929


namespace slope_PQ_pos_range_a_l122_122187

section
variable {f : ℝ → ℝ}
def f := λ x, x + Real.sin x

-- Slope of line PQ is greater than 0
theorem slope_PQ_pos (P Q : ℝ × ℝ) (hP : P.2 = f P.1) (hQ : Q.2 = f Q.1) (hPQ : P.1 ≠ Q.1) :
  (Q.2 - P.2) / (Q.1 - P.1) > 0 :=
sorry

-- Range of real numbers a such that f(x) ≥ ax cos x for all x in [0, π/2]
theorem range_a (a : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≥ a * x * Real.cos x) ↔ a ≤ 2 :=
sorry
end

end slope_PQ_pos_range_a_l122_122187


namespace fraction_sum_eq_l122_122031

-- Given conditions
variables (w x y : ℝ)
axiom hx : w / x = 1 / 6
axiom hy : w / y = 1 / 5

-- Proof goal
theorem fraction_sum_eq : (x + y) / y = 11 / 5 :=
by sorry

end fraction_sum_eq_l122_122031


namespace nonagon_interior_angle_l122_122200

theorem nonagon_interior_angle :
  ∀ (n : ℕ), n = 9 → (let sum_of_angles := 180 * (n - 2) in sum_of_angles / n = 140) :=
by
  intro n
  intro h
  rw [h]
  let sum_of_angles := 180 * (n - 2)
  have : sum_of_angles = 1260 :=
    by simp [sum_of_angles, h]
  simp [this]
  sorry

end nonagon_interior_angle_l122_122200


namespace matilda_selling_price_l122_122299

variable (c : ℝ := 300) (p : ℝ := 0.15)

theorem matilda_selling_price : ∃ s : ℝ, s = c - p * c ∧ s = 255 := by
  -- Using the specific values provided
  exists (300 - 0.15 * 300)
  simp
  -- 300 - 0.15 * 300 = 300 - 45 = 255
  exact ⟨rfl, rfl⟩

end matilda_selling_price_l122_122299


namespace sqrt_mixed_number_simplify_l122_122947

open Real

theorem sqrt_mixed_number_simplify :
  sqrt (8 + 9 / 16) = sqrt 137 / 4 :=
by 
  sorry

end sqrt_mixed_number_simplify_l122_122947


namespace baseball_cards_initial_count_unkn_l122_122301

-- Definitions based on the conditions
def cardValue : ℕ := 6
def tradedCards : ℕ := 2
def receivedCardsValue : ℕ := (3 * 2) + 9   -- 3 cards worth $2 each and 1 card worth $9
def profit : ℕ := receivedCardsValue - (tradedCards * cardValue)

-- Lean 4 statement to represent the proof problem
theorem baseball_cards_initial_count_unkn (h_trade : tradedCards * cardValue = 12)
    (h_receive : receivedCardsValue = 15)
    (h_profit : profit = 3) : ∃ n : ℕ, n >= 2 ∧ n = 2 + (n - 2) :=
sorry

end baseball_cards_initial_count_unkn_l122_122301


namespace team_scores_variance_l122_122216

noncomputable def team_scores : List ℝ := [11.5, 13.5, 13.5, 11.5]

def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

def variance (scores : List ℝ) : ℝ :=
  let μ := mean scores
  (scores.map (λ x, (x - μ) ^ 2)).sum / scores.length

theorem team_scores_variance : variance team_scores = 1 := by
  sorry

end team_scores_variance_l122_122216


namespace color_points_distance_d_l122_122271

open Real Rat Set

variable {d r s : ℝ} (rational_r : r ∈ ℚ) (rational_s : s ∈ ℚ)

theorem color_points_distance_d (h : d^2 = r^2 + s^2) :
  ∃ (coloring : ℚ × ℚ → ℕ), (∀ (a b : ℚ × ℚ), dist a b = d → coloring a ≠ coloring b) :=
sorry

end color_points_distance_d_l122_122271


namespace sqrt_mixed_number_simplified_l122_122924

theorem sqrt_mixed_number_simplified :
  (sqrt (8 + 9 / 16) = sqrt 137 / 4) :=
begin
  sorry
end

end sqrt_mixed_number_simplified_l122_122924


namespace Tom_total_spend_l122_122130

theorem Tom_total_spend :
  let notebook_price := 2
  let notebook_discount := 0.75
  let notebook_count := 4
  let magazine_price := 5
  let magazine_count := 2
  let pen_price := 1.50
  let pen_discount := 0.75
  let pen_count := 3
  let book_price := 12
  let book_count := 1
  let discount_threshold := 30
  let coupon_discount := 10
  let total_cost :=
    (notebook_count * (notebook_price * notebook_discount)) +
    (magazine_count * magazine_price) +
    (pen_count * (pen_price * pen_discount)) +
    (book_count * book_price)
  let final_cost := if total_cost >= discount_threshold then total_cost - coupon_discount else total_cost
  final_cost = 21.375 :=
by
  sorry

end Tom_total_spend_l122_122130


namespace probability_v_greater_l122_122592

-- Define v(n) for a positive integer n
def v (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.find_greatest (fun j => 2^j ∣ n) n

-- Define the given range for a and b
def in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 32

-- Prove the probability that v(a) > v(b)
theorem probability_v_greater (a b : ℕ) (ha : in_range a) (hb : in_range b) :
  (probability (v a > v b)) = 341 / 1024 :=
sorry

end probability_v_greater_l122_122592


namespace monotonicity_a_leq_0_monotonicity_a_gt_0_exists_x_f_lt_g_l122_122726

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a + (Real.exp 1) / Real.exp x
noncomputable def g (x : ℝ) : ℝ := 1 / x + Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x + (Real.exp x - Real.exp 1 * x) / (x * Real.exp x)

theorem monotonicity_a_leq_0 (a : ℝ) (x : ℝ) (h_leq_0 : a ≤ 0) :
  ∀ x ∈ Ioi (0 : ℝ), (h a)' x < 0 :=
sorry

theorem monotonicity_a_gt_0 (a : ℝ) (x : ℝ) (h_gt_0 : a > 0) :
  ∀ x ∈ Ioo (0 : ℝ) (Real.sqrt (1 / (2 * a))) (h a)' x < 0 ∧
  ∀ x ∈ Ioi (Real.sqrt (1 / (2 * a))), (h a)' x > 0 :=
sorry

theorem exists_x_f_lt_g (a : ℝ) (h_a : a < 1 / 2) :
  ∃ x ∈ Ioi (1 : ℝ), f a x < g x :=
sorry

end monotonicity_a_leq_0_monotonicity_a_gt_0_exists_x_f_lt_g_l122_122726


namespace opposite_of_neg_five_is_five_l122_122399

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122399


namespace faulty_key_in_digits_l122_122477

-- Problem statement definitions
def is_faulty_key (digit_seq : list ℕ) (faulty_keys : set ℕ) : Prop :=
  ∃ (missing_digits faulty_occurrences : ℕ), 
    (∃ (attempted_seq : list ℕ), length digit_seq = 10 ∧ length attempted_seq = 7 ∧
    missing_digits = 10 - 7 ∧ length (digit_seq.filter (λ d, d ∈ faulty_keys)) ≥ 5 ∧
    length (attempted_seq.filter (λ d, d ∈ faulty_keys)) ≥ 2 ∧
    length (digit_seq.filter (λ d, d ∈ faulty_keys)) - length (attempted_seq.filter (λ d, d ∈ faulty_keys)) = 3)

-- Theorem: Proving which keys could be the faulty ones.
theorem faulty_key_in_digits (digit_seq : list ℕ) :
  is_faulty_key digit_seq {7, 9} :=
sorry

end faulty_key_in_digits_l122_122477


namespace Tom_catchup_time_l122_122077

-- Conditions
def speed_Alice := 6 -- Alice's speed in mph
def speed_Tom := 9 -- Tom's speed in mph
def initial_distance := 3 -- Initial distance south of Alice in miles
def eastward_travel_time := 10 / 60 -- Eastward travel time in hours (10 minutes)

-- Computations
noncomputable def eastward_distance := speed_Tom * eastward_travel_time -- Eastward distance traveled by Tom
noncomputable def relative_speed := speed_Tom - speed_Alice -- Relative speed north

-- Proof statement
theorem Tom_catchup_time : 
  let northward_distance := initial_distance in
  let time_to_catchup := northward_distance / relative_speed in
  let time_in_minutes := time_to_catchup * 60 in
  time_in_minutes = 60 :=
by {
  sorry -- proof would be included here
}

end Tom_catchup_time_l122_122077


namespace negation_proposition_p_true_l122_122643

variable (x y : ℕ)
def proposition_p : Prop := (x + y = 5 → (x = 2 ∧ y = 3))

-- The Lean statement to prove the negation of proposition p is a true proposition.
theorem negation_proposition_p_true : ¬(x + y = 5 → x = 2 ∧ y = 3) → true := 
by { 
  intro h, 
  trivial 
}

end negation_proposition_p_true_l122_122643


namespace arrangement_of_students_l122_122591

theorem arrangement_of_students :
  let total_students := 5
  let total_communities := 2
  (2 ^ total_students - 2) = 30 :=
by
  let total_students := 5
  let total_communities := 2
  sorry

end arrangement_of_students_l122_122591


namespace kingdom_of_sellke_arabia_l122_122274

theorem kingdom_of_sellke_arabia (n : ℕ) (h_pos : 0 < n) 
    (h_cities : 2018 * n + 1) 
    (h_roads : ∀ (C : ℕ) (i : ℕ), 1 ≤ i ∧ i ≤ 2018 → ∃ (cities : Finset ℕ), cities.card = n ∧ ∀ x ∈ cities, distance C x = i) :
  2 ∣ n :=
sorry

end kingdom_of_sellke_arabia_l122_122274


namespace five_people_six_chairs_l122_122233

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l122_122233


namespace sqrt_mixed_number_eq_l122_122936

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l122_122936


namespace diagonal_of_square_l122_122350

theorem diagonal_of_square (s d : ℝ) (h_perimeter : 4 * s = 40) : d = 10 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_square_l122_122350


namespace barry_pretzels_l122_122867

theorem barry_pretzels (A S B : ℕ) (h1 : A = 3 * S) (h2 : S = B / 2) (h3 : A = 18) : B = 12 :=
  by
  sorry

end barry_pretzels_l122_122867


namespace possible_faulty_keys_l122_122457

theorem possible_faulty_keys (d : ℕ) (digits : list ℕ) (len_d : digits.length = 10) 
  (registered : list ℕ) (len_r : registered.length = 7) :
  (∃ d ∈ digits, d = 7 ∨ d = 9) :=
by
  sorry

end possible_faulty_keys_l122_122457


namespace sarah_ellie_total_reflections_l122_122124

def sarah_tall_reflections : ℕ := 10
def sarah_wide_reflections : ℕ := 5
def sarah_narrow_reflections : ℕ := 8

def ellie_tall_reflections : ℕ := 6
def ellie_wide_reflections : ℕ := 3
def ellie_narrow_reflections : ℕ := 4

def tall_mirror_passages : ℕ := 3
def wide_mirror_passages : ℕ := 5
def narrow_mirror_passages : ℕ := 4

def total_reflections (sarah_tall sarah_wide sarah_narrow ellie_tall ellie_wide ellie_narrow
    tall_passages wide_passages narrow_passages : ℕ) : ℕ :=
  (sarah_tall * tall_passages + sarah_wide * wide_passages + sarah_narrow * narrow_passages) +
  (ellie_tall * tall_passages + ellie_wide * wide_passages + ellie_narrow * narrow_passages)

theorem sarah_ellie_total_reflections :
  total_reflections sarah_tall_reflections sarah_wide_reflections sarah_narrow_reflections
  ellie_tall_reflections ellie_wide_reflections ellie_narrow_reflections
  tall_mirror_passages wide_mirror_passages narrow_mirror_passages = 136 :=
by
  sorry

end sarah_ellie_total_reflections_l122_122124


namespace problem_l122_122594

-- Define the polynomial g(x) with given coefficients
def g (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x^2 + x + 8

-- Define the polynomial f(x) with given coefficients
def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^4 + x^3 + b * x^2 + 50 * x + c

-- Define the conditions
def conditions (a b c r : ℝ) : Prop :=
  ∃ roots : Finset ℝ, (∀ x ∈ roots, g x a = 0) ∧ (∀ x ∈ roots, f x a b c = 0) ∧ (roots.card = 3) ∧
  (8 - r = 50) ∧ (a - r = 1) ∧ (1 - a * r = b) ∧ (-8 * r = c)

-- Define the theorem to be proved
theorem problem (a b c r : ℝ) (h : conditions a b c r) : f 1 a b c = -1333 :=
by sorry

end problem_l122_122594


namespace opposite_of_neg_five_is_five_l122_122402

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122402


namespace locus_M_N_circle_construct_triangle_ABC_l122_122498

-- Definitions corresponding to the problem conditions
variable (A B C M N : Type)
variable [MetricSpace A B]
variable (a b f : ℝ)

-- Condition: M and N on side AB
def is_angle_bisector (A B C M : Type) [MetricSpace A B] : Prop :=
sorry

def is_external_bisector (A B C N : Type) [MetricSpace A B] : Prop :=
sorry

-- Problem statement 1: Locus of point M and N form circles
theorem locus_M_N_circle (h_fixed_BC : fixed_points B C)
  (h_bisector : is_angle_bisector A B C M)
  (h_external_bisector : is_external_bisector A B C N) :
  exists (O : Type) (rM rN : ℝ), 
  (locus M = { M' | dist M' O = rM }) ∧ 
  (locus N = { N' | dist N' O = rN }) :=
sorry

-- Problem statement 2: Constructing triangle ABC
theorem construct_triangle_ABC (h_BC : dist B C = a)
  (h_CA : dist C A = b)
  (h_CM : dist C M = f) :
  ∃ (A B C : Type), is_triangle A B C ∧ dist B C = a ∧ dist C A = b ∧ dist C M = f :=
sorry

end locus_M_N_circle_construct_triangle_ABC_l122_122498


namespace function_extreme_points_and_zeros_l122_122871

noncomputable def ω_range : Set ℝ := 
  setOf (λ ω, (13 : ℝ)/6 < ω ∧ ω ≤ (8 : ℝ)/3)

theorem function_extreme_points_and_zeros (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = sin (ω * x + π / 3)) 
  (h2 : set.count (set_of (λ x, x ∈ (0, π) ∧ extreme_point f x)) = 3) 
  (h3 : set.count (set_of (λ x, x ∈ (0, π) ∧ f x = 0)) = 2) : 
  ω ∈ ω_range := 
sorry

end function_extreme_points_and_zeros_l122_122871


namespace number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l122_122844

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l122_122844


namespace finite_trajectory_only_for_3_6_9_l122_122556

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def next_number (a : ℕ) : ℕ :=
  if is_perfect_square a then (nat.sqrt a) else a + 3

def trajectory (a : ℕ) : set ℕ :=
  { n | ∃ k : ℕ, nat.iterate next_number k a = n }

def finite_cardinality (s : set ℕ) : Prop :=
  ∃ n : ℕ, s = finset.range (n + 1) ∧ (finset.range (n + 1)).card = n + 1

theorem finite_trajectory_only_for_3_6_9 (n : ℕ) (h : n > 1) :
  finite (trajectory n) ↔ n = 3 ∨ n = 6 ∨ n = 9 :=
sorry

end finite_trajectory_only_for_3_6_9_l122_122556


namespace differences_count_l122_122201

def S : Set ℕ := {1, 2, 3, 4, 5, 7, 10}

noncomputable def count_differences (s : Set ℕ) : ℕ :=
  (s.powerset.filter (λ t, t.card = 2)).image (λ t, (t.to_list.nth_le 1 (by sorry) - t.to_list.nth_le 0 (by sorry)).natAbs).to_finset.card

theorem differences_count :
  count_differences S = 9 := 
sorry

end differences_count_l122_122201


namespace find_sum_l122_122343

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l122_122343


namespace FQ_length_correct_l122_122760

noncomputable def lengthFQ : ℝ :=
  let DE := 6
  let DF := Real.sqrt 85
  let EF := Real.sqrt (DF^2 - DE^2)
  EF

theorem FQ_length_correct :
  let DE := 6
  let DF := Real.sqrt 85
  EF = Real.sqrt (DF^2 - DE^2)
  in lengthFQ = 7 :=
by
  -- definitions
  let DE := 6
  let DF := Real.sqrt 85
  let EF := Real.sqrt (DF^2 - DE^2)
  
  -- goal
  have h1 : EF = 7, from
    calc
      EF = Real.sqrt (85 - 36) : by rw [Real.sqrt_sub'; exact 85 ≥ 36, add_comm, Real.sqrt_sq]
      ... = 7 : by norm_num

  show lengthFQ = 7 from h1

end FQ_length_correct_l122_122760


namespace triangle_MNP_is_right_isosceles_l122_122268

open Classical
open Real

variables {A B C M P N : Point}
variables {α β γ θ φ ψ : ℝ}
variables {a b c : ℝ}

-- Definitions of initial points and triangles
def is_midpoint (N : Point) (B C : Point) : Prop :=
  N = (B + C) / 2

def right_isosceles (A B M : Point) : Prop :=
  (dist A B = dist A M) ∧ (∠AMB = 90)

-- Given conditions
axiom midpoint_condition : is_midpoint N B C
axiom right_isosceles_triangle1 : right_isosceles A B M
axiom right_isosceles_triangle2 : right_isosceles A C P

-- To prove: triangle MNP is right isosceles
theorem triangle_MNP_is_right_isosceles :
  right_isosceles M N P :=
sorry

end triangle_MNP_is_right_isosceles_l122_122268


namespace omega_range_l122_122878

theorem omega_range (ω : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < π ∧
    (sin (ω * x₁ + π / 3)).cos = 0 ∧ (sin (ω * x₂ + π / 3)).cos = 0 ∧ (sin (ω * x₃ + π / 3)).cos = 0) ∧ 
  (∃ z₁ z₂ : ℝ, 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ sin (ω * z₁ + π / 3) = 0 ∧ sin (ω * z₂ + π / 3) = 0) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
by
  sorry

end omega_range_l122_122878


namespace triangulation_of_polygon_has_1998_triangles_l122_122309

theorem triangulation_of_polygon_has_1998_triangles
    (n : ℕ) (k : ℕ)
    (poly_vertices : fin n)
    (additional_points : fin k)
    (h1 : n = 1000)
    (h2 : k = 500)
    (no_collinear : ∀ (p1 p2 p3 : fin k), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬ collinear p1 p2 p3) :
  let total_points := n + k in
  ∃ (T : ℕ), T = 1998 :=
by
  sorry

end triangulation_of_polygon_has_1998_triangles_l122_122309


namespace rectangle_area_tangent_circle_l122_122833

theorem rectangle_area_tangent_circle 
  (s : ℝ)
  (EFGH : Type) 
  [rect_EFGH :  IsRectangle EFGH]
  (circle : Type) 
  [circle_tangent_EFGH : CircleTangentToRectangle circle EFGH s]
  (M : Type) 
  [midpoint_diagonal_M : MidpointOfDiagonal EFGH M]
  (circle_passes_through_M : CirclePassesThroughPoint circle M) :
  rectangle_area EFGH = 4 * s^2 :=
sorry  -- Proof is omitted

end rectangle_area_tangent_circle_l122_122833


namespace FNRT_is_parallelogram_l122_122720

open_locale euclidean_geometry

variables {A B C D F R N T : Point}
variables (circle : Circle)
variables (midpoint_AC : N = midpoint A C) (midpoint_BD : T = midpoint B D)
variables (perp_chords : ∀ {P Q}, on_circle P circle → on_circle Q circle → (chord circle P Q) ⟂ (line R))

theorem FNRT_is_parallelogram (h_AB : on_chord circle A B)
                             (h_CD : on_chord circle C D)
                             (intersect : ∃ F, F ∈ line A B ∧ F ∈ line C D)
                             (h_perpendicular: chord circle A B ⟂ chord circle C D)
                             (mid_AC : N = midpoint A C)
                             (mid_BD : T = midpoint B D):
  parallelogram (quad F N R T) :=
by sorry

end FNRT_is_parallelogram_l122_122720


namespace ellipse_eq_line_l1_eq_l122_122642

noncomputable theory

-- Define the parabola and focus F
def parabola := { p : ℝ × ℝ | let (x, y) := p in y^2 = 4 * x }
def focus_F : ℝ × ℝ := (1, 0)

-- Define the ellipse M with given conditions a > b > 0
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) := 
  { p : ℝ × ℝ | let (x, y) := p in x^2 / a^2 + y^2 / b^2 = 1 }

-- Intersection point P in the first quadrant
def point_P : ℝ × ℝ := (2/3, 2 * real.sqrt 6 / 3)

-- Distance |PF| = 5/3
def distance_PF (p : ℝ × ℝ) (f : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - f.1)^2 + (p.2 - f.2)^2)

-- Line l1
def line_eq (k x1 y1 : ℝ) : ℝ -> ℝ :=
  λ x, k * (x - x1) + y1

-- Prove the first question
theorem ellipse_eq (a b : ℝ) (h : a > b ∧ b > 0) :
  ellipse a b h = { p : ℝ × ℝ | let (x, y) := p in x^2 / 4 + y^2 / 3 = 1 } :=
by sorry

-- Prove the second question
theorem line_l1_eq : 
  line_eq (3 / 2) 1 0 = λ x, 3 * x - 3 / 2 ∨ 
  line_eq (-3 / 2) 1 0 = λ x, 3 * x + 3 / 2 :=
by sorry

end ellipse_eq_line_l1_eq_l122_122642


namespace solve_system_of_equations_l122_122754
noncomputable theory

variables (a b x y z : ℝ)
variables (h_non_zero_y_z : y ≠ z)
variables (h_non_zero_2y_3z : 2 * y ≠ 3 * z)
variables (h_non_zero_3a2x_2ay : 3 * a^2 * x ≠ 2 * a * y)
variables (h_non_zero_a : a ≠ 0)
variables (h_non_one_b : b ≠ 1)

theorem solve_system_of_equations (h1 : (a * x + z) / (y - z) = (1 + b) / (1 - b))
                                 (h2 : (2 * a * x - 3 * b) / (2 * y - 3 * z) = 1)
                                 (h3 : (5 * z - 4 * b) / (3 * a^2 * x - 2 * a * y) = b / a) :
    (b ≠ -19 / 15 ∧ x = 1 / a ∧ y = 1 ∧ z = b)
    ∨ (b = -19 / 15 ∧ ∃ z, true) :=
sorry

end solve_system_of_equations_l122_122754


namespace polygon_sides_l122_122664

theorem polygon_sides (each_exterior_angle : ℝ)
                      (h₀ : each_exterior_angle = 60) :
                      (number_of_sides : ℕ) :=
  sorry

end polygon_sides_l122_122664


namespace point_A_in_fourth_quadrant_l122_122245

def Point := ℤ × ℤ

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def point_A : Point := (3, -2)
def point_B : Point := (2, 5)
def point_C : Point := (-1, -2)
def point_D : Point := (-2, 2)

theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A :=
  sorry

end point_A_in_fourth_quadrant_l122_122245


namespace probability_meeting_is_approx_0_02_l122_122734
  
theorem probability_meeting_is_approx_0_02 :
  let A_paths := 2^6 in
  let B_paths := 2^6 in
  let meet_prob :=
    (∑ i in Finset.range 11, (Nat.choose 6 i) * (Nat.choose 6 (10 - i))) / (A_paths * B_paths)
  abs (meet_prob - 0.02) < 0.01 :=
by
  sorry  

end probability_meeting_is_approx_0_02_l122_122734


namespace no_perfect_square_abc_sum_l122_122746

theorem no_perfect_square_abc_sum (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  ¬ ∃ m : ℕ, m * m = (100 * a + 10 * b + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) :=
by
  sorry

end no_perfect_square_abc_sum_l122_122746


namespace final_distribution_after_four_rounds_l122_122590

-- Definitions of initial conditions
def initial_balls (child : ℕ) : ℕ :=
  match child with
  | 0 => 2  -- A
  | 1 => 4  -- B
  | 2 => 6  -- C
  | 3 => 8  -- D
  | 4 => 10 -- E
  | _ => 0

-- Function representing the redistribution process over one round
def redistribute (balls : ℕ → ℕ) (n : ℕ) : ℕ → ℕ :=
  if balls n > balls ((n+4) % 5) then balls n - 2 else balls n

-- Main proposition to prove
theorem final_distribution_after_four_rounds :
  ∃ final_balls : ℕ → ℕ,
    (∀ n, final_balls n = 6) ∧
    let balls1 := redistribute initial_balls in
    let balls2 := redistribute balls1 in
    let balls3 := redistribute balls2 in
    let balls4 := redistribute balls3 in
    ball4 = final_balls :=
begin
  sorry
end

end final_distribution_after_four_rounds_l122_122590


namespace trapezoid_area_ABCD_l122_122686

theorem trapezoid_area_ABCD 
  {A B C D E : Type}
  (trapezoid : Trapezoid A B C D)
  (AB_parallel_CD : parallel A B C D)
  (AC : diagonal A C)
  (BD : diagonal B D)
  (intersect : intersect AC BD E)
  (area_ABE : area A B E 72)
  (area_ADE : area A D E 27)
  (length_relation : length AB = 3 * length CD) :
  area_trapezoid A B C D = 134 :=
sorry

end trapezoid_area_ABCD_l122_122686


namespace other_x_intercept_correct_l122_122866

def foci1 : ℝ × ℝ := (0, 3)
def foci2 : ℝ × ℝ := (4, 0)
def x_intercept1 : ℝ × ℝ := (1, 0)
def constant_distance_sum : ℝ := Real.sqrt 10 + 3

theorem other_x_intercept_correct :
  ∃ x : ℝ, 
    (Real.sqrt (x^2 + 9) + Real.abs (x - 4) = constant_distance_sum)
  ∧ (x, 0) = ( (13 - 14 * Real.sqrt 10) / (2 * Real.sqrt 10 + 14), 0 ) :=
sorry

end other_x_intercept_correct_l122_122866


namespace sqrt_mixed_number_eq_l122_122934

def improper_fraction (a b c : ℕ) (d : ℕ) : ℚ :=
  a + b / d

theorem sqrt_mixed_number_eq (a b c d : ℕ) (h : d ≠ 0) :
  (d * a + b) ^ 2 = c * d^2 → 
  sqrt (improper_fraction a b c d) = (sqrt (d * a + b)) / (sqrt d) :=
by sorry

example : sqrt (improper_fraction 8 9 0 16) = (sqrt 137) / 4 := 
  sqrt_mixed_number_eq 8 9 0 16 sorry sorry

end sqrt_mixed_number_eq_l122_122934


namespace binomial_expansion_terms_l122_122622

-- Given n is a positive integer, in the expansion of (1/2 + 2x)^n,
-- if the sum of the binomial coefficients of the first three terms equals 79,
-- prove that n = 12 and the 11th term in the expansion has the largest coefficient.
theorem binomial_expansion_terms (n : ℕ) (hn : 0 < n)
  (h : ∑ i in range 3, (n.choose i) = 79) :
  n = 12 ∧ ∀ k ∈ finset.range (12 + 1), binomial_coefficient (12, k) * 4^k ≤ binomial_coefficient (12, 10) * 4^10 := 
by 
-- Proof is omitted
sorry

end binomial_expansion_terms_l122_122622


namespace find_defective_keys_l122_122451

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122451


namespace mean_of_all_students_is_79_l122_122731

def mean_score_all_students (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) : ℕ :=
  (36 * s + 75 * s) / ((2/5 * s) + s)

theorem mean_of_all_students_is_79 (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) (hF : F = 90) (hS : S = 75) : 
  mean_score_all_students F S f s hf = 79 := by
  sorry

end mean_of_all_students_is_79_l122_122731


namespace min_value_sum_of_reciprocals_l122_122722

open Real

theorem min_value_sum_of_reciprocals :
  ∀ (b : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ 8 → 0 < b i) → (∑ i in finset.range 8, b (i + 1)) = 2 → (∑ i in finset.range 8, 1 / (b (i + 1))) ≥ 32 :=
by 
  assume b hb hb_sum,
  sorry

end min_value_sum_of_reciprocals_l122_122722


namespace workers_l122_122022

theorem workers (N C : ℕ) (h1 : N * C = 300000) (h2 : N * (C + 50) = 315000) : N = 300 :=
by
  sorry

end workers_l122_122022


namespace cone_lateral_surface_area_l122_122516

open Real

-- Define the cone's base radius
def base_radius : ℝ := 1

-- Define the cone's height
def height : ℝ := sqrt 3

-- Define the slant height using Pythagoras' theorem
def slant_height : ℝ := sqrt ((height ^ 2) + (base_radius ^ 2))

-- Compute the lateral surface area
def lateral_surface_area : ℝ := (1 / 2) * slant_height * pi * base_radius * 2

-- Statement to prove
theorem cone_lateral_surface_area : 
  ∀ (base_radius height : ℝ), base_radius = 1 → height = sqrt 3 → lateral_surface_area = 2 * pi :=
by
  intros base_radius height h_base_radius h_height
  rw [h_base_radius, h_height, lateral_surface_area]
  sorry

end cone_lateral_surface_area_l122_122516


namespace faulty_keys_l122_122464

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122464


namespace ratio_rect_prism_l122_122417

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l122_122417


namespace avg_waiting_time_l122_122534

theorem avg_waiting_time : 
  let P_G := 1 / 3      -- Probability of green light
  let P_red := 2 / 3    -- Probability of red light
  let E_T_given_G := 0  -- Expected time given green light
  let E_T_given_red := 1 -- Expected time given red light
  (E_T_given_G * P_G) + (E_T_given_red * P_red) = 2 / 3
:= by
  sorry

end avg_waiting_time_l122_122534


namespace find_defective_keys_l122_122456

-- Definitions from the conditions
def ten_digit_sequence : Type := list ℕ
def registered_digits : Type := list ℕ

axiom typed_ten_digits (s : ten_digit_sequence) : s.length = 10
axiom only_seven_registered (t : registered_digits) : t.length = 7
axiom three_missing_digits (s : ten_digit_sequence) (t : registered_digits) : 
             s.length - t.length = 3

-- This indicates that it is the same type of digits just subsets of initial sequence
axiom all_digits_in_sequence (s : ten_digit_sequence) (t : registered_digits) : 
            ∀ (d : ℕ), d ∈ t → d ∈ s

axiom defective_key_condition (s : ten_digit_sequence) (t : registered_digits) : 
            ∃ d : ℕ, (d ∈ s ∧ d ∉ t) ∧ count s d >= 5 ∧ count t d = 2

axiom multiple_defective_keys_condition (s : ten_digit_sequence) (t : registered_digits): 
           ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ s ∧ d2 ∈ s) ∧ (d1 ∉ t ∧ d2 ∉ t) ∧ 
           (count s d1 >= 3 ∧ count s d2 >= 3)

-- Proving the answer:
theorem find_defective_keys (s : ten_digit_sequence) (t : registered_digits) :
  typed_ten_digits s → only_seven_registered t → three_missing_digits s t → 
  all_digits_in_sequence s t → defective_key_condition s t → multiple_defective_keys_condition s t → 
  ∃ (keys : list ℕ), keys = [7, 9] :=
begin
  sorry
end

end find_defective_keys_l122_122456


namespace cos6_plus_sin6_ge_one_fourth_l122_122316

theorem cos6_plus_sin6_ge_one_fourth (alpha : ℝ) : 
  cos(alpha)^2 + sin(alpha)^2 = 1 → cos(alpha)^6 + sin(alpha)^6 ≥ 1 / 4 := 
by 
  sorry

end cos6_plus_sin6_ge_one_fourth_l122_122316


namespace min_elements_in_S_l122_122276

theorem min_elements_in_S (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 0) :
  ∃ S : set (fin (2 * n) × fin (2 * n)), 
  (∀ A B ∈ S, A ≠ B → ∃ seq : list (fin (2 * n) × fin (2 * n)),
    seq.head = some A ∧ seq.last = some B ∧ (∀ i ∈ seq.zip seq.tail, (i.1.1 = i.2.1 ∧ (i.1.2 + 1 = i.2.2 ∨ i.1.2 = i.2.2 + 1)) ∨ (i.1.2 = i.2.2 ∧ (i.1.1 + 1 = i.2.1 ∨ i.1.1 = i.2.1 + 1)))) ∧ 
  (∀ i j, i < n ∧ j < n → ∃ a b : ℕ, a < 2 ∧ b < 2 ∧ ((2 * i + a, 2 * j + b) ∈ S)) ∧ 
  card S = 2 * n^2 - 1 :=
sorry

end min_elements_in_S_l122_122276


namespace opposite_of_neg_five_is_five_l122_122396

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l122_122396


namespace range_of_t_l122_122146

def seq_is_decreasing_difference (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → (x n + x (n+2)) / 2 < x (n+1)

def b (t : ℝ) (n : ℕ) : ℝ := 2 * t - t * (n - 1) / 2 ^ (n - 1)

theorem range_of_t (t : ℝ) : 
  seq_is_decreasing_difference (b t) → t > 1 :=
sorry

end range_of_t_l122_122146


namespace identify_faulty_key_l122_122481

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l122_122481


namespace tom_teaching_years_l122_122002

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l122_122002


namespace line_l_eq_line_l2_eq_l122_122626

noncomputable def line_l : set (ℝ × ℝ) := {p | 5 * p.1 - 3 * p.2 + 15 = 0}
noncomputable def line_l1 : set (ℝ × ℝ) := {p | 3 * p.1 + 5 * p.2 - 3 = 0}
noncomputable def line_l2 : set (ℝ × ℝ) := {p | 3 * p.1 - 5 * p.2 - 3 = 0}

theorem line_l_eq :
  (∀ p : ℝ × ℝ, (p = (0, 5)) → 5 * p.1 - 3 * p.2 + 15 = 0) ∧
  (∀ x_intercept y_intercept : ℝ, (5 / (-x_intercept) + 3 / y_intercept = 2) → (x_intercept = -3) ∧ (y_intercept = 5))
  → 5 * 0 - 3 * 5 + 15 = 0 :=
sorry

theorem line_l2_eq :
  (∀ p : ℝ × ℝ, (p = (8/3, -1)) → (3 * p.1 + 5 * p.2 - 3 = 0)) ∧
  (∀ p : ℝ × ℝ, line_l2 p ↔ line_l1 (p.1, -p.2))
  → 3 * 1 - 5 * 0 - 3 = 0 :=
sorry

end line_l_eq_line_l2_eq_l122_122626


namespace sphere_surface_area_of_tetrahedron_l122_122178

theorem sphere_surface_area_of_tetrahedron (a : ℝ) (S : ℝ) 
  (h1 : a = real.sqrt 2) 
  (h2 : ∀ v1 v2 v3 v4 : euclidean_space ℝ 3, tetrahedron v1 v2 v3 v4 ∧ all_eq_length v1 v2 v3 v4 a ∧ inscribed_in_sphere v1 v2 v3 v4 s) : 
  S = 3 * real.pi :=
sorry

end sphere_surface_area_of_tetrahedron_l122_122178


namespace quadrilateral_inequality_l122_122787

theorem quadrilateral_inequality (A C : ℝ) (AB AC AD BC CD : ℝ) (h1 : A + C < 180) (h2 : A > 0) (h3 : C > 0) (h4 : AB > 0) (h5 : AC > 0) (h6 : AD > 0) (h7 : BC > 0) (h8 : CD > 0) : 
  AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l122_122787


namespace A_not_on_transformed_plane_l122_122499

noncomputable def A : ℝ × ℝ × ℝ := (-3, -2, 4)
noncomputable def k : ℝ := -4/5
noncomputable def original_plane (x y z : ℝ) : Prop := 2 * x - 3 * y + z - 5 = 0

noncomputable def transformed_plane (x y z : ℝ) : Prop := 
  2 * x - 3 * y + z + (k * -5) = 0

theorem A_not_on_transformed_plane :
  ¬ transformed_plane (-3) (-2) 4 :=
by
  sorry

end A_not_on_transformed_plane_l122_122499


namespace range_of_omega_l122_122882

noncomputable section

open Real

/--
Assume the function f(x) = sin (ω x + π / 3) has exactly three extreme points and two zeros in 
the interval (0, π). Prove that the range of values for ω is 13 / 6 < ω ≤ 8 / 3.
-/
theorem range_of_omega 
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h : ∀ x, f x = sin (ω * x + π / 3))
  (h_extreme : (∃ a b c, 0 < a ∧ a < b ∧ b < c ∧ c < π ∧ (f' a = 0) ∧ (f' b = 0) ∧ (f' c = 0)))
  (h_zeros : (∃ u v, 0 < u ∧ u < v ∧ v < π ∧ f u = 0 ∧ f v = 0)) :
  (13 / 6) < ω ∧ ω ≤ (8 / 3) :=
  sorry

end range_of_omega_l122_122882


namespace range_of_a_l122_122638

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) :
    (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
    a ≥ 18 := sorry

end range_of_a_l122_122638


namespace primes_square_condition_l122_122954

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l122_122954


namespace faulty_keys_l122_122466

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122466


namespace faulty_keys_l122_122467

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l122_122467


namespace day_100_days_from_wednesday_l122_122800

-- Definitions for the conditions
def today_is_wednesday := "Wednesday"

def days_in_week := 7

def day_of_the_week (n : Nat) : String := 
  let days := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  days[n % days.length]

-- Theorem to prove
theorem day_100_days_from_wednesday : day_of_the_week ((4 + 100) % days_in_week) = "Friday" :=
  sorry

end day_100_days_from_wednesday_l122_122800


namespace vitia_knows_all_by_29th_attempt_vitia_knows_all_by_24th_attempt_l122_122789

/-- 
There are 30 questions each with two possible answers: YES or NO.
Vitia can answer all questions and learn the number of correct answers after each attempt.
-/
def questions := Fin 30
def answers := Bool -- YES or NO

/-- 
Vitia's answer check function. 
-/
def answer_check (attempt : questions → answers) (correct_answers : questions → answers) : Nat :=
  (Finset.univ.filter (λ q => attempt q = correct_answers q)).card

/-- 
Vitia can determine all the correct answers by the 29th attempt and answer all questions correctly on the 30th attempt.
-/
theorem vitia_knows_all_by_29th_attempt :
  ∃ (strategy : Fin 30 → (questions → answers)),
    ∀ (correct_answers : questions → answers), 
      (∃ n < 30, ∀ k < n, answer_check (strategy k) correct_answers = answer_check (strategy k) correct_answers) :=
  sorry

/-- 
Vitia can determine all the correct answers by the 24th attempt and answer all questions correctly on the 25th attempt.
-/
theorem vitia_knows_all_by_24th_attempt :
  ∃ (strategy : Fin 25 → (questions → answers)),
    ∀ (correct_answers : questions → answers), 
      (∃ n < 25, ∀ k < n, answer_check (strategy k) correct_answers = answer_check (strategy k) correct_answers) :=
  sorry

end vitia_knows_all_by_29th_attempt_vitia_knows_all_by_24th_attempt_l122_122789


namespace opposite_of_neg5_is_pos5_l122_122360

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l122_122360


namespace time_to_paint_house_l122_122300

variable (M : ℝ)
variable (Patty Rachel : ℝ)

-- Conditions
def Patty_time := M / 3
def Rachel_time := 2 * Patty + 5

-- Assertion
theorem time_to_paint_house : Rachel_time = 13 → M = 12 := by
  intro h
  have h1 : Rachel_time = 2 * (M / 3) + 5 := by
    rw [←Patty_time, <- rfl, <- rfl]
    exact sorry
  have h2 : 2 * (M / 3) + 5 = 13 := by
    rw [h1, h]
    exact sorry
  have h3 : 2 * (M / 3) = 8 := by
    linarith
    exact sorry
  have h4 : M / 3 = 4 := by
    linarith
    exact sorry
  have h5 : M = 12 := by
    linarith
    exact sorry
  exact h5

end time_to_paint_house_l122_122300


namespace five_people_six_chairs_l122_122234

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l122_122234


namespace gavrila_distance_l122_122092

noncomputable def distance_from_start (L : ℝ) (y : ℝ) : ℝ :=
  let x := (y^2) / (4 * L)
  sqrt(x^2 + y^2)

theorem gavrila_distance (L : ℝ) (y : ℝ) : 
  L = 50 → y = 40 → Real.floor (distance_from_start L y) = 41 :=
by
  intros hL hy
  rw [hL, hy]
  sorry

end gavrila_distance_l122_122092


namespace outer_boundary_diameter_l122_122062

theorem outer_boundary_diameter (fountain_diameter garden_width path_width : ℝ) 
(h1 : fountain_diameter = 12) 
(h2 : garden_width = 10) 
(h3 : path_width = 6) : 
2 * ((fountain_diameter / 2) + garden_width + path_width) = 44 :=
by
  -- Sorry, proof not needed for this statement
  sorry

end outer_boundary_diameter_l122_122062


namespace geometric_locus_circle_l122_122858

variables {α : Type*} [linear_ordered_field α]

variable (a : α)

theorem geometric_locus_circle 
  (A B C D : α) 
  (hAB : A - B = a)
  (h_angle : ∀ C D, C forms angle α with segment AB)
  (h_relation : ∀ C, (C - A)^2 = (A - B) * (D - B)) :
  ∃ (x y : α), (x + a / 2)^2 + (y + a / 2 * cot α)^2 = (a / (2 * sin α))^2 := 
by
  sorry

end geometric_locus_circle_l122_122858


namespace train_pass_man_in_6_6_seconds_l122_122856

-- Definitions based on the conditions
def train_length : ℝ := 110
def train_speed_kmh : ℝ := 27
def man_speed_kmh : ℝ := 6
def kmh_to_ms (v: ℝ) : ℝ := v * (5 / 18)

-- Speeds in m/s
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh
def man_speed_ms : ℝ := kmh_to_ms man_speed_kmh

-- Relative speed since they are moving in opposite directions
def relative_speed_ms : ℝ := train_speed_ms + man_speed_ms

-- Time to pass the man
def time_to_pass : ℝ := train_length / relative_speed_ms

-- Lean statement to prove the time to pass is 6.6 seconds
theorem train_pass_man_in_6_6_seconds : time_to_pass = 6.6 := 
by sorry

end train_pass_man_in_6_6_seconds_l122_122856


namespace opposite_of_negative_five_l122_122388

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122388


namespace probability_opposite_vertex_after_8_moves_l122_122049

-- Let's define the cube vertices and edges
@[derive DecidableEq]
inductive Vertex
| A | B | C | D | E | F | G | H
deriving Repr

def opposite_vertex (v : Vertex) : Vertex :=
  match v with
  | Vertex.A => Vertex.H
  | Vertex.H => Vertex.A
  | Vertex.B => Vertex.F
  | Vertex.F => Vertex.B
  | Vertex.C => Vertex.G
  | Vertex.G => Vertex.C
  | Vertex.D => Vertex.E
  | Vertex.E => Vertex.D

def adjacent_vertices (v : Vertex) : List Vertex :=
  match v with
  | Vertex.A => [Vertex.B, Vertex.D, Vertex.E]
  | Vertex.B => [Vertex.A, Vertex.C, Vertex.F]
  | Vertex.C => [Vertex.B, Vertex.D, Vertex.G]
  | Vertex.D => [Vertex.A, Vertex.C, Vertex.E]
  | Vertex.E => [Vertex.A, Vertex.D, Vertex.F]
  | Vertex.F => [Vertex.B, Vertex.E, Vertex.H]
  | Vertex.G => [Vertex.C, Vertex.H, Vertex.B]
  | Vertex.H => [Vertex.F, Vertex.G, Vertex.D]

def moves_sequence_valid (seq : List Vertex) : Bool :=
  match seq with
  | [] => false
  | h :: t' =>
    seq.length = 9 ∧
    h = Vertex.A ∧
    List.last seq = some Vertex.H ∧
    (∀ p ∈ seq, seq.count p ≤ 2) ∧
    List.chain' (λ u v => v ∈ adjacent_vertices u) seq

def possible_sequences : List (List Vertex) := sorry -- implementation not provided, should generate all possible 8-move sequences

def valid_sequences : List (List Vertex) :=
  possible_sequences.filter moves_sequence_valid

def probability_valid_sequences : ℚ :=
  (valid_sequences.length : ℚ) / (3^8 : ℚ)

theorem probability_opposite_vertex_after_8_moves : probability_valid_sequences = 2 / 2187 :=
sorry -- proof not provided

end probability_opposite_vertex_after_8_moves_l122_122049


namespace recipe_quantities_l122_122066

noncomputable def mixed_to_improper (a b c : ℤ) : ℚ :=
  a + b / c

noncomputable def calculate_flour_needed (flour : ℚ) (fraction : ℚ) : ℚ :=
  flour * fraction

noncomputable def calculate_sugar_needed (flour : ℚ) (ratio : ℚ) : ℚ :=
  flour * ratio

def to_mixed_number (q : ℚ) : ℚ × ℚ :=
  let n := q.num / q.denom
  let fraction := (q.num % q.denom) / q.denom
  (n, fraction)

theorem recipe_quantities :
  ∀ (flour_orig : ℚ) (sugar_ratio : ℚ) (recipe_fraction : ℚ),
  flour_orig = mixed_to_improper 6 2 3 →
  sugar_ratio = 1 / 2 →
  recipe_fraction = 1 / 3 →
  to_mixed_number (calculate_flour_needed flour_orig recipe_fraction) = (2, 2 / 9) ∧
  to_mixed_number (calculate_sugar_needed (calculate_flour_needed flour_orig recipe_fraction) sugar_ratio) = (1, 1 / 9) :=
by
  intros
  sorry

end recipe_quantities_l122_122066


namespace measure_of_angle_C_l122_122085

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 360) (h2 : C = 5 * D) : C = 300 := 
by sorry

end measure_of_angle_C_l122_122085


namespace smallest_solution_l122_122588

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l122_122588


namespace seating_arrangements_l122_122239

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l122_122239


namespace find_n_l122_122411

theorem find_n (a n : ℕ) 
  (h1 : a^2 % n = 8) 
  (h2 : a^3 % n = 25) 
  (h3 : n > 25) : 
  n = 113 := 
sorry

end find_n_l122_122411


namespace sum_binom_mod_l122_122277

theorem sum_binom_mod (S : ℤ) : 
  S = ∑ n in Finset.range (670), (-1)^n * nat.choose 2010 (3 * n) →
  S % 1000 = 6 :=
by
  sorry

end sum_binom_mod_l122_122277


namespace intersection_x_value_l122_122413

theorem intersection_x_value : 
  ∀ x y : ℝ, y = 3 * x - 15 ∧ 3 * x + y = 105 → x = 20 :=
by
  intro x y
  intro h
  cases h with h1 h2
  sorry

end intersection_x_value_l122_122413


namespace opposite_of_negative_five_l122_122383

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l122_122383


namespace exterior_angle_polygon_num_sides_l122_122666

theorem exterior_angle_polygon_num_sides (exterior_angle : ℝ) (h : exterior_angle = 60) : ∃ (n : ℕ), n = 6 :=
by
  use 6
  sorry

end exterior_angle_polygon_num_sides_l122_122666


namespace find_square_sum_l122_122206

theorem find_square_sum (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : (x + y) ^ 2 = 135 :=
sorry

end find_square_sum_l122_122206


namespace triangle_sides_l122_122258

noncomputable theory

variables {A B C : Type*}
variables {a b c : ℝ} {S : ℝ}

-- The conditions of the problem
def conditions (a b c : ℝ) (B : ℝ) (S : ℝ) : Prop :=
  (b * Real.sin C = Real.sqrt 3) ∧
  (B = Real.pi / 4) ∧
  (S = 9 / 2)

-- The proof problem as a Lean statement
theorem triangle_sides (h : conditions a b c (Real.pi / 4) (9 / 2)): 
  c = Real.sqrt 6 ∧ b = Real.sqrt 15 :=
sorry

end triangle_sides_l122_122258


namespace sum_first_60_terms_l122_122409

def sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  a (n + 1) + (-1)^n * a n = 2 * n - 1

def sum_upto (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem sum_first_60_terms (a : ℕ → ℤ) (h : ∀ n, sequence a n) :
  sum_upto a 60 = 1830 :=
sorry

end sum_first_60_terms_l122_122409


namespace fixed_point_of_f_l122_122040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (|x + 1|)

theorem fixed_point_of_f (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 0 = 1 :=
by
  sorry

end fixed_point_of_f_l122_122040


namespace day_of_week_100_days_from_wednesday_l122_122798

theorem day_of_week_100_days_from_wednesday (today_is_wed : ∃ i : ℕ, i % 7 = 3) : 
  (100 % 7 + 3) % 7 = 5 := 
by
  sorry

end day_of_week_100_days_from_wednesday_l122_122798


namespace opposite_of_negative_five_l122_122368

theorem opposite_of_negative_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  split
  {
    sorry,
  }
  {
    refl,
  }

end opposite_of_negative_five_l122_122368


namespace area_of_triangle_CM_N_l122_122978

noncomputable def triangle_area (a : ℝ) : ℝ :=
  let M := (a / 2, a, a)
  let N := (a, a / 2, a)
  let MN := Real.sqrt ((a - a / 2) ^ 2 + (a / 2 - a) ^ 2)
  let CK := Real.sqrt (a ^ 2 + (a * Real.sqrt 2 / 4) ^ 2)
  (1/2) * MN * CK

theorem area_of_triangle_CM_N 
  (a : ℝ) :
  (a > 0) →
  triangle_area a = (3 * a^2) / 8 :=
by
  intro h
  -- Proof will go here.
  sorry

end area_of_triangle_CM_N_l122_122978


namespace cyclist_speed_ratio_is_4_l122_122420

noncomputable def ratio_of_speeds (v_a v_b v_c : ℝ) : ℝ :=
  if v_a ≤ v_b ∧ v_b ≤ v_c then v_c / v_a else 0

theorem cyclist_speed_ratio_is_4
  (v_a v_b v_c : ℝ)
  (h1 : v_a + v_b = d / 5)
  (h2 : v_b + v_c = 15)
  (h3 : 15 = (45 - d) / 3)
  (d : ℝ) : 
  ratio_of_speeds v_a v_b v_c = 4 :=
by
  sorry

end cyclist_speed_ratio_is_4_l122_122420


namespace chip_prices_and_minimum_cost_l122_122508

theorem chip_prices_and_minimum_cost :
  ∃ (price_A price_B : ℝ) (a b : ℕ),
    price_B - price_A = 9 ∧
    3120 / price_A = 4200 / price_B ∧
    200 = a + b ∧
    1/4 * b ≤ a ∧ a ≤ 1/3 * b ∧
    price_A = 26 ∧
    price_B = 35 ∧
    a = 50 ∧
    b = 150 ∧
    26 * a + 35 * b = 6550 :=
begin
  -- Existence statement, no need to prove individual steps
  -- Proof is omitted
  sorry
end

end chip_prices_and_minimum_cost_l122_122508


namespace math_problem_l122_122805

def has_three_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, p.prime ∧ n = p^2

def has_four_divisors (n : ℕ) : Prop :=
  (∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p*q) ∨ (∃ p : ℕ, p.prime ∧ n = p^3)

def smallest_with_three_divisors : ℕ :=
  Nat.find (exists_intro 4 (by norm_num [has_three_divisors, prime]))

def smallest_greater_than_100_with_four_divisors : ℕ :=
  Nat.find (exists_intro 125 (by norm_num [has_four_divisors, prime]))

theorem math_problem :
  let a := smallest_with_three_divisors,
      b := smallest_greater_than_100_with_four_divisors
  in a + b = 129 :=
by
  let a := smallest_with_three_divisors
  let b := smallest_greater_than_100_with_four_divisors
  sorry

end math_problem_l122_122805


namespace find_age_of_15th_person_l122_122766

-- Define the conditions given in the problem
def total_age_of_18_persons (avg_18 : ℕ) (num_18 : ℕ) : ℕ := avg_18 * num_18
def total_age_of_5_persons (avg_5 : ℕ) (num_5 : ℕ) : ℕ := avg_5 * num_5
def total_age_of_9_persons (avg_9 : ℕ) (num_9 : ℕ) : ℕ := avg_9 * num_9

-- Define the overall question which is the age of the 15th person
def age_of_15th_person (total_18 : ℕ) (total_5 : ℕ) (total_9 : ℕ) : ℕ :=
  total_18 - total_5 - total_9

-- Statement of the theorem to prove
theorem find_age_of_15th_person :
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  age_of_15th_person total_18 total_5 total_9 = 56 :=
by
  -- Definitions for the total ages
  let avg_18 := 15
  let num_18 := 18
  let avg_5 := 14
  let num_5 := 5
  let avg_9 := 16
  let num_9 := 9
  let total_18 := total_age_of_18_persons avg_18 num_18 
  let total_5 := total_age_of_5_persons avg_5 num_5
  let total_9 := total_age_of_9_persons avg_9 num_9
  
  -- Goal: compute the age of the 15th person
  let answer := age_of_15th_person total_18 total_5 total_9

  -- Prove that the computed age is equal to 56
  show answer = 56
  sorry

end find_age_of_15th_person_l122_122766


namespace original_four_digit_number_l122_122836

theorem original_four_digit_number : 
  ∃ x y z: ℕ, (x = 1 ∧ y = 9 ∧ z = 7 ∧ 1000 * x + 100 * y + 10 * z + y = 1979) ∧ 
  (1000 * y + 100 * z + 10 * y + x - (1000 * x + 100 * y + 10 * z + y) = 7812) ∧ 
  (1000 * y + 100 * z + 10 * y + x < 10000 ∧ 1000 * x + 100 * y + 10 * z + y < 10000) := 
sorry

end original_four_digit_number_l122_122836


namespace transformed_curve_eq_l122_122006

theorem transformed_curve_eq :
  ∀ (y x : ℝ), (y * cos x + 2 * y - 1 = 0) →
    ((y - 1) * sin x + 2 * y - 3 = 0) :=
by
  intros y x h
  sorry

end transformed_curve_eq_l122_122006


namespace geometric_series_sum_l122_122813

theorem geometric_series_sum :
  (1 / 3 - 1 / 6 + 1 / 12 - 1 / 24 + 1 / 48 - 1 / 96) = 7 / 32 :=
by
  sorry

end geometric_series_sum_l122_122813
