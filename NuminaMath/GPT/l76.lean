import Mathlib
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Choose
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Ln
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace area_between_chords_eq_l76_76078

-- Define the circle, its radius, and the distance between the chords
def radius : ℝ := 10
def distance_between_chords : ℝ := 10
def answer : ℝ := (100 * Real.pi) / 3 - 25 * Real.sqrt 3

-- The theorem that needs to be proved
theorem area_between_chords_eq :
  ∃ (r d : ℝ), r = 10 ∧ d = 10 ∧ 
  r = radius ∧ d = distance_between_chords ∧
  let h := d / 2,
      chord_length := 2 * Real.sqrt (r^2 - h^2),
      sector_area := 2 * (1 / 6) * Real.pi * r^2,
      triangle_area := 2 * (1 / 2) * h * (chord_length / 2),
      total_area := sector_area - triangle_area
  in total_area = answer :=
begin
  sorry
end

end area_between_chords_eq_l76_76078


namespace initial_winning_percentage_calc_l76_76384

variable (W : ℝ)
variable (initial_matches : ℝ := 120)
variable (additional_wins : ℝ := 70)
variable (final_matches : ℝ := 190)
variable (final_average : ℝ := 0.52)
variable (initial_wins : ℝ := 29)

noncomputable def winning_percentage_initial :=
  (initial_wins / initial_matches) * 100

theorem initial_winning_percentage_calc :
  (W = initial_wins) →
  ((W + additional_wins) / final_matches = final_average) →
  winning_percentage_initial = 24.17 :=
by
  intros
  sorry

end initial_winning_percentage_calc_l76_76384


namespace distance_between_circumcenters_l76_76447

theorem distance_between_circumcenters 
  (A B C M : Point)
  (h_collinear : collinear A B C)
  (line_through_B : ∃ line, (B ∈ line) ∧ (M ∈ line))
  (distance_AC : dist A C = a)
  (angle_MBC : ∠ M B C = α) :
  dist (circumcenter (triangle A B M)) (circumcenter (triangle C B M)) = a / (2 * Real.sin α) := 
sorry

end distance_between_circumcenters_l76_76447


namespace opposite_of_a_is_2022_l76_76382

theorem opposite_of_a_is_2022 (a : Int) (h : -a = -2022) : a = 2022 := by
  sorry

end opposite_of_a_is_2022_l76_76382


namespace cycle_not_multiple_of_3_l76_76903

theorem cycle_not_multiple_of_3
  (G : Type) [graph G]
  (Towns : set G) 
  (Roads : G → G → Prop) 
  (h1 : ∀ x ∈ Towns, ∃ y z w ∈ Towns, Roads x y ∧ Roads x z ∧ Roads x w)
  : ∃ (cycle : list G), (∀ (v : G), v ∈ cycle → v ∈ Towns ∧ (∀ u ∈ cycle, Roads v u)) ∧ ¬ (length cycle % 3 = 0) :=
sorry

end cycle_not_multiple_of_3_l76_76903


namespace circumcircle_tangent_to_BD_l76_76868

open EuclideanGeometry

noncomputable def proof_problem_statement : Prop :=
  ∀ (A B C D H S T : Point), 
  convex_quadrilateral A B C D ∧ 
  ∠ B A C = 90 ∧ 
  ∠ A D C = 90 ∧ 
  foot_of_perpendicular A B D = H ∧ 
  S ∈ segment A B ∧ 
  T ∈ segment A D ∧ 
  H ∈ triangle S C T ∧ 
  ∠ S H C - ∠ B S C = 90 ∧ 
  ∠ T H C - ∠ D T C = 90 →
  tangent_to_line (circumcircle_of_triangle S H T) (line B D)

theorem circumcircle_tangent_to_BD : proof_problem_statement :=
begin
  sorry
end

end circumcircle_tangent_to_BD_l76_76868


namespace shared_candy_equally_l76_76750

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l76_76750


namespace opposite_of_neg_two_l76_76528

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76528


namespace opposite_of_neg_two_l76_76532

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76532


namespace small_pump_time_to_fill_tank_l76_76214

-- Define the problem conditions
def large_pump_rate := 4 -- tanks per hour
def combined_time := 0.23076923076923078 -- hours
def combined_rate := 1 / combined_time -- tanks per hour
def small_pump_rate := combined_rate - large_pump_rate -- tanks per hour

-- Question: How many hours would it take the small water pump to fill the empty tank alone?
theorem small_pump_time_to_fill_tank :
  (1 / small_pump_rate) ≈ 3 := sorry

end small_pump_time_to_fill_tank_l76_76214


namespace goat_cannot_return_l76_76213

theorem goat_cannot_return (R d L : ℝ) (R_pos : R = 0.5) (d_pos : d = 1) (L_pos : L = 4.7)
  (horiz_rope : ∀ (x y z : ℝ), R > 0 ∧ d > 0 ∧ L > 0) :
  let α := Real.arcsin (R / (R + d)) in
  2 * R * Real.cot α + R * (Real.pi + 2 * α) > L :=
by
  sorry

end goat_cannot_return_l76_76213


namespace curved_surface_area_approximation_l76_76255

-- Define the radius and slant height
def radius : ℝ := 35
def slant_height : ℝ := 30

-- Formula for curved surface area of a cone
def CSA (r l : ℝ) : ℝ := π * r * l

-- Realistically, we can't usually prove approximations exactly; however, we state the goal
theorem curved_surface_area_approximation : 
  abs (CSA radius slant_height - 3299.34) < 0.01 :=
by
  -- leaving proof as sorry to focus on the statement construction
  sorry

end curved_surface_area_approximation_l76_76255


namespace length_diff_width_8m_l76_76988

variables (L W : ℝ)

theorem length_diff_width_8m (h1: W = (1/2) * L) (h2: L * W = 128) : L - W = 8 :=
by sorry

end length_diff_width_8m_l76_76988


namespace albert_needs_more_money_l76_76221

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l76_76221


namespace opposite_of_neg2_l76_76504

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76504


namespace polyhedron_edge_sum_leq_l76_76887

-- Define distance between two points in a polyhedron
variable {P : Type} [polyhedron P]
variable (d : P → P → ℝ)

-- Define vertices and edges of the polyhedron
variable (A B : P)
variable (edges : List (P × P))

-- Assume A and B are the points at the greatest distance in P
variable (h1 : ∀ x y : P, d x y ≤ d A B)

-- Define a function to compute the length of an edge
def edge_length : (P × P) → ℝ := λ (x y), d x y

-- Sum of lengths of all edges in polyhedron P
def sum_edge_lengths : ℝ := edges.map edge_length |>.sum

theorem polyhedron_edge_sum_leq : sum_edge_lengths edges ≥ 3 * d A B :=
  sorry

end polyhedron_edge_sum_leq_l76_76887


namespace limit_M_n_l76_76009

noncomputable def circumradius (a b c : ℝ) (K : ℝ) : ℝ :=
  a * b * c / (4 * K)

noncomputable def herons_formula_area (a b c : ℝ) (s : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))
  
def find_s (a b c : ℝ) : ℝ := (a + b + c) / 2

def K (a b c s : ℝ) : ℝ := herons_formula_area a b c s

def f_n (A B C n : ℝ) (P : ℝ^2) : ℝ :=
  (A ^ n + B ^ n + C ^ n) ^ (1 / n)

theorem limit_M_n :
  let a := 13
  let b := 14
  let c := 15
  let s := find_s a b c
  let K := K a b c s
  let R := circumradius a b c K
  ∀ P : ℝ^2, 
  (∀ n : ℝ, 
    let M_n := λ P, (A(P) ^ n + B(P) ^ n + C(P) ^ n) ^ (1 / n),
    lim n → ∞ (inf (M_n P)) = R)
  := 
begin
  sorry
end

end limit_M_n_l76_76009


namespace max_discount_rate_l76_76148

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76148


namespace h_eq_transformed_f_l76_76655

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2) ^ 2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

noncomputable def h (x : ℝ) : ℝ := f (3 - x) + 2

theorem h_eq_transformed_f :
  ∀ x : ℝ, h(x) = f(3 - x) + 2 :=
by
  intro x
  exact rfl

end h_eq_transformed_f_l76_76655


namespace younger_son_age_after_30_years_l76_76917

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l76_76917


namespace jackson_money_l76_76849

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l76_76849


namespace no_color_match_in_3x3_squares_l76_76387

theorem no_color_match_in_3x3_squares 
  (colors : Fin 10) (grid : Fin 101 × Fin 101 → Option colors) :
  (∀ i j, 1 ≤ i ∧ i ≤ 99 ∧ 1 ≤ j ∧ j ≤ 99 → 
    ∃! c, ∃ i' j', 
      ((i - 1 ≤ i' ∧ i' ≤ i + 1) ∧ (j - 1 ≤ j' ∧ j' ≤ j + 1) ∧ (grid (i, j) = some c) ∧ (i ≠ i' ∨ j ≠ j'))) → 
  False :=
by
  sorry

end no_color_match_in_3x3_squares_l76_76387


namespace tan_alpha_minus_beta_l76_76361

theorem tan_alpha_minus_beta (α β : ℝ) (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_l76_76361


namespace tan_alpha_one_value_l76_76374

theorem tan_alpha_one_value {α : ℝ} (h : Real.tan α = 1) :
    (2 * Real.sin α ^ 2 + 1) / Real.sin (2 * α) = 2 :=
by
  sorry

end tan_alpha_one_value_l76_76374


namespace total_value_correct_l76_76588

-- Define conditions
def import_tax_rate : ℝ := 0.07
def tax_paid : ℝ := 109.90
def tax_exempt_value : ℝ := 1000

-- Define total value
def total_value (V : ℝ) : Prop :=
  V - tax_exempt_value = tax_paid / import_tax_rate

-- Theorem stating that the total value is $2570
theorem total_value_correct : total_value 2570 := by
  sorry

end total_value_correct_l76_76588


namespace count_multiples_4_6_10_less_300_l76_76745

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l76_76745


namespace problem_statement_l76_76319

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ
| x => if x > 0 then f (x - 3) else exp x + log (8^(x+1) * (1/4)^(-2)) / log 2

-- Formalize the problem as a Lean statement:
theorem problem_statement : f 2016 = 8 :=
by sorry

end problem_statement_l76_76319


namespace different_flavors_count_l76_76676

-- Definitions as per the given conditions
def red_candies : Nat := 5
def green_candies : Nat := 4

-- Total different flavors
def flavors_count := 17

-- Problem statement to prove
theorem different_flavors_count : 
  (∃ flavors : Finset (Nat × Nat), ∀ x y : Nat, (x ≤ red_candies ∧ y ≤ green_candies) → 
    (flavors.contains (x, y) = ((x = 0 ∧ y = 0) → false) ∧
    ∀ (a b : Nat), (a * y = b * x → (x, y) = (a, b))) ∧ 
    flavors.card = flavors_count) :=
sorry

end different_flavors_count_l76_76676


namespace median_time_is_135_l76_76066

-- Definitions and conditions
def times_in_seconds : List ℕ := [23, 23, 45, 55, 60, 70, 90, 105, 110, 135, 145, 148, 160, 170, 185, 185, 195, 360]

-- Statement to prove
theorem median_time_is_135 : 
  (List.sort times_in_seconds).nth (times_in_seconds.length / 2) = some 135 :=
by
  sorry

end median_time_is_135_l76_76066


namespace complex_modulus_squared_l76_76008

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6 * Complex.I) : Complex.abs z^2 = 13 / 2 :=
by
  sorry

end complex_modulus_squared_l76_76008


namespace count_subsets_l76_76817

open Set

def A : Set ℕ := {x | x ∈ Finset.range 11}
def B : Set ℕ := {1, 2, 3, 4}

theorem count_subsets (C : Set ℕ) (h₁ : C ⊆ A) (h₂ : C ∩ B ≠ ∅) :
  (Finset.powerset (Finset.range 11)).filter (λ s, (↑s ∩ B).nonempty).card = 960 :=
  sorry

end count_subsets_l76_76817


namespace farm_horses_cows_l76_76110

theorem farm_horses_cows (x : ℕ) :
  let h := 4 * x in
  let c := x in
  let h' := h - 15 in
  let c' := c + 15 in
  (4 * x - 15) * 7 = 13 * (x + 15) →
  h' - c' = 30 :=
by
  intro x h c h' c'
  intro h_c_rel
  sorry

end farm_horses_cows_l76_76110


namespace max_discount_rate_l76_76155

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76155


namespace minimum_soldiers_to_add_l76_76777

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76777


namespace distinct_collections_proof_l76_76028

noncomputable def distinct_collections_count : ℕ := 240

theorem distinct_collections_proof : distinct_collections_count = 240 := by
  sorry

end distinct_collections_proof_l76_76028


namespace baking_problem_l76_76857

theorem baking_problem (flour_recipe : ℚ) (sugar_recipe : ℚ) (cocoa_recipe : ℚ) (milk_recipe : ℚ)
                       (flour_added : ℚ) (sugar_added : ℚ) :
  flour_recipe = 3/4 ∧ sugar_recipe = 2/3 ∧ cocoa_recipe = 1/3 ∧ milk_recipe = 1/2 ∧
  flour_added = 1/2 ∧ sugar_added = 1/4 →
  let flour_needed := flour_recipe - flour_added in
  let sugar_needed := sugar_recipe - sugar_added in
  let cocoa_needed := cocoa_recipe in
  let milk_needed := milk_recipe in
  flour_needed = 1/4 ∧ sugar_needed = 5/12 ∧ cocoa_needed = 1/3 ∧ milk_needed = 1/2 :=
by
  sorry

end baking_problem_l76_76857


namespace pounds_in_one_ton_is_2600_l76_76029

variable (pounds_in_one_ton : ℕ)
variable (ounces_in_one_pound : ℕ := 16)
variable (packets : ℕ := 2080)
variable (weight_per_packet_pounds : ℕ := 16)
variable (weight_per_packet_ounces : ℕ := 4)
variable (gunny_bag_capacity_tons : ℕ := 13)

theorem pounds_in_one_ton_is_2600 :
  (packets * (weight_per_packet_pounds + weight_per_packet_ounces / ounces_in_one_pound)) = (gunny_bag_capacity_tons * pounds_in_one_ton) →
  pounds_in_one_ton = 2600 :=
sorry

end pounds_in_one_ton_is_2600_l76_76029


namespace find_lunch_break_duration_l76_76892

def lunch_break_duration : ℝ → ℝ → ℝ → ℝ
  | s, a, L => L

theorem find_lunch_break_duration (s a L : ℝ) :
  (8 - L) * (s + a) = 0.6 ∧ (6.4 - L) * a = 0.28 ∧ (9.6 - L) * s = 0.12 →
  lunch_break_duration s a L = 1 :=
  by
    sorry

end find_lunch_break_duration_l76_76892


namespace XYM1M2_concyclic_l76_76000

variables (X Y A1 A2 B1 B2 M1 M2 : Point)

-- Conditions
variables (d1 d2 : Line) 
variables (C C' : Circle)
variable (intersects_X : X ∈ (d1 ∩ d2))
variable (passes_X_C : X ∈ C)
variable (passes_X_C' : X ∈ C')
variable (intersects_A1_d1 : A1 ∈ (C ∩ d1))
variable (intersects_A2_d2 : A2 ∈ (C ∩ d2))
variable (intersects_B1_d1 : B1 ∈ (C' ∩ d1))
variable (intersects_B2_d2 : B2 ∈ (C' ∩ d2))
variable (intersects_Y : Y ∈ (C ∩ C'))
variable (midpoint_M1 : M1 = midpoint A1 B1)
variable (midpoint_M2 : M2 = midpoint A2 B2)

-- Theorem statement
theorem XYM1M2_concyclic : 
  cyclic [X, Y, M1, M2] :=
sorry

end XYM1M2_concyclic_l76_76000


namespace sequence_formula_l76_76298

-- Define the initial conditions and sequence
def θ : ℝ := sorry  -- θ : real number
def a : ℕ → ℝ
| 1     := 2 * Real.cos θ
| (n+2) := Real.sqrt (2 + a (n + 1))

-- State the theorem to be proven
theorem sequence_formula
  (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  ∀ n : ℕ, a (n + 1) = 2 * Real.cos (θ / 2^n) :=
sorry

end sequence_formula_l76_76298


namespace tangent_identity_l76_76369

theorem tangent_identity (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tangent_identity_l76_76369


namespace opposite_of_neg2_l76_76508

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76508


namespace range_of_f_le_1_l76_76726

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x - cos x

theorem range_of_f_le_1 :
  {x : ℝ | (∃ k : ℤ, 2 * k * π + π / 3 ≤ x ∧ x ≤ 2 * k * π + π) } =
  {x : ℝ | f x ≥ 1} :=
by
  sorry

end range_of_f_le_1_l76_76726


namespace relationship_among_abc_l76_76303

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := f (Real.log 7 / Real.log 4)
def b : ℝ := f (-Real.log 3 / Real.log 2)
def c : ℝ := f (2 ^ 16)

lemma logarithmic_relationship : (Real.log 7 / Real.log 4) < (Real.log 9 / Real.log 4) := sorry

lemma function_properties (x : ℝ) (hx : x ≤ 0) : Monotone f := sorry
lemma function_properties_even (x : ℝ) : f x = f (-x) := sorry

theorem relationship_among_abc : c < b ∧ b < a :=
by {
  have hb : f (-Real.log 3 / Real.log 2) = f (Real.log 3 / Real.log 2),
  { apply function_properties_even, },
  have hlog : Real.log 7 / Real.log 4 < Real.log 9 / Real.log 4,
  { exact logarithmic_relationship, },
  have f_decreasing : ∀ x (hx : x ≥ 0), f x ≤ f (2 ^ 16),
  { sorry },
  
  split,
  { sorry, }, -- to show c < b
  { sorry, }, -- to show b < a
}

end relationship_among_abc_l76_76303


namespace count_ordered_pairs_l76_76734

-- Define the sets M and N and the condition M ∪ N = {1, 2}
def M : Set ℕ := sorry
def N : Set ℕ := sorry

-- Hypothesize that the union of M and N is {1, 2}
theorem count_ordered_pairs (h : M ∪ N = {1, 2}) : 
  let pairs := { (M, N) : Set ℕ × Set ℕ // M ∪ N = {1, 2} }
  in pairs.card = 9 :=
sorry

end count_ordered_pairs_l76_76734


namespace find_gain_percent_l76_76981

theorem find_gain_percent (CP SP : ℝ) (h1 : CP = 20) (h2 : SP = 25) : 100 * ((SP - CP) / CP) = 25 := by
  sorry

end find_gain_percent_l76_76981


namespace trigonometric_identity_l76_76355

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 :=
sorry

end trigonometric_identity_l76_76355


namespace price_of_first_shirt_l76_76623

theorem price_of_first_shirt
  (price1 price2 price3 : ℕ)
  (total_shirts : ℕ)
  (min_avg_price_of_remaining : ℕ)
  (total_avg_price_of_all : ℕ)
  (prices_of_first_3 : price1 = 100 ∧ price2 = 90 ∧ price3 = 82)
  (condition1 : total_shirts = 10)
  (condition2 : min_avg_price_of_remaining = 104)
  (condition3 : total_avg_price_of_all > 100) :
  price1 = 100 :=
by
  sorry

end price_of_first_shirt_l76_76623


namespace weight_of_each_bag_l76_76226

theorem weight_of_each_bag 
  (total_potatoes_weight : ℕ) (damaged_potatoes_weight : ℕ) 
  (bag_price : ℕ) (total_revenue : ℕ) (sellable_potatoes_weight : ℕ) (number_of_bags : ℕ) 
  (weight_of_each_bag : ℕ) :
  total_potatoes_weight = 6500 →
  damaged_potatoes_weight = 150 →
  sellable_potatoes_weight = total_potatoes_weight - damaged_potatoes_weight →
  bag_price = 72 →
  total_revenue = 9144 →
  number_of_bags = total_revenue / bag_price →
  weight_of_each_bag * number_of_bags = sellable_potatoes_weight →
  weight_of_each_bag = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end weight_of_each_bag_l76_76226


namespace tan_alpha_minus_beta_l76_76352

variable (α β : ℝ)

theorem tan_alpha_minus_beta
  (h : sin (α + β) + cos (α + β) = 2 * real.sqrt 2 * cos (α + π/4) * sin β) : 
  real.tan (α - β) = -1 :=
sorry

end tan_alpha_minus_beta_l76_76352


namespace tangent_identity_l76_76370

theorem tangent_identity (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tangent_identity_l76_76370


namespace circles_intersect_l76_76558

theorem circles_intersect :
  1 - 1 -
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 2*y - 2 = 0) ∧ (x^2 + y^2 - 4*x - 2*y + 1 = 0) →
    ∃ c1 c2 : ℝ × ℝ, 
      let C1 := (-1, -1),
          C2 := (2, 1),
          r1 := 2,
          r2 := 2 in
      let dist := Real.sqrt ((C2.1 - C1.1)^2 + (C2.2 - C1.2)^2) in
      dist < r1 + r2 ∧ dist > 0 := r1 - r2 :=
by
  sorry

end circles_intersect_l76_76558


namespace trigonometric_identity_l76_76354

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 :=
sorry

end trigonometric_identity_l76_76354


namespace max_pens_min_pens_l76_76978

def pen_prices : List ℕ := [2, 3, 4]
def total_money : ℕ := 31

/-- Given the conditions of the problem, prove the maximum number of pens -/
theorem max_pens  (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 14 := by
  sorry

/-- Given the conditions of the problem, prove the minimum number of pens -/
theorem min_pens (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 9 := by
  sorry

end max_pens_min_pens_l76_76978


namespace sum_of_relatively_prime_integers_l76_76560

theorem sum_of_relatively_prime_integers (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x * y + x + y = 154) (h4 : Nat.gcd x y = 1) (h5 : x < 30) (h6 : y < 30) : 
  x + y = 34 :=
sorry -- proof

end sum_of_relatively_prime_integers_l76_76560


namespace correct_statements_l76_76100

theorem correct_statements :
  let data_mode := [1, 1, 2, 3, 4, 4]
  let data_median := [2, 3, 4, 5, 6, 7]
  let first_group_avg := 5
  let second_group_avg := 4 
  (list.mode data_mode = [1, 4]) ∧
  ¬ (list.median data_median = 4.5) ∧
  (∃ data_set, list.median data_set = data_set.head ∧ list.mode data_set = data_set.head ∧ list.mean data_set = data_set.head) ∧
  ((3 * first_group_avg + 4 * second_group_avg) / 7 = (3 * 5 + 4 * 4) / 7) := 
by
  sorry

end correct_statements_l76_76100


namespace solve_sqrt_eq_l76_76041

theorem solve_sqrt_eq (x : ℝ) : sqrt (7 * x - 3) + sqrt (2 * x - 2) = 3 ↔ (x = 2 ∨ x = 172 / 25) := 
sorry

end solve_sqrt_eq_l76_76041


namespace max_discount_rate_l76_76162

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76162


namespace equal_green_purple_shoes_l76_76570

theorem equal_green_purple_shoes :
  ∃ (G P : ℕ), 
    let T := 1250 in
    let B := 540 in
    let P := 355 in
    G = T - B - P ∧ P = 355 ∧ G = P :=
by
  sorry

end equal_green_purple_shoes_l76_76570


namespace grid_blue_probability_correct_l76_76610

noncomputable def probability_grid_blue : ℚ :=
  let p_center := (1 / 2) ^ 4 in
  let p_edges := (1 / 2) ^ 12 in
  p_center * p_edges

theorem grid_blue_probability_correct :
  probability_grid_blue = 1 / 65536 :=
by
  -- proof is omitted
  sorry

end grid_blue_probability_correct_l76_76610


namespace r_at_5_l76_76866

def r (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 - 1

theorem r_at_5 :
  r 5 = 48 := by
  sorry

end r_at_5_l76_76866


namespace misha_cards_digit_example_l76_76877

-- Define variables representing the digits
variables {L O M H C : ℕ}

-- Define the main equation
def main_equation : Prop := L + O / M + O + H + O / C = 20

-- Conditions
def digit_conditions: Prop :=
  M > O ∧ C > O

theorem misha_cards_digit_example
  (digit_conditions)
  : main_equation :=
sorry

end misha_cards_digit_example_l76_76877


namespace max_discount_rate_l76_76170

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76170


namespace trig_identity_proof_l76_76262

theorem trig_identity_proof :
  sin 50 * cos 170 - sin 40 * sin 170 = - (sqrt 3 / 2) :=
by sorry

end trig_identity_proof_l76_76262


namespace sum_ratios_l76_76895

open Finset Nat

def a (i : ℕ) : ℚ := (choose 2019 i : ℚ)
def b (i : ℕ) : ℚ := (choose 2020 i : ℚ)
def c (i : ℕ) : ℚ := (choose 2021 i : ℚ)

theorem sum_ratios :
  ∑ i in range (2020), (b i / c i) - ∑ i in range (2019), (a i / b i) = 1 / 2 := by
  sorry

end sum_ratios_l76_76895


namespace ellipse_equation_intersection_diagonals_fixed_point_l76_76314

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : |a - 1| = 2) (h4 : sqrt(a^2 - b^2) = 1) : 
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by 
  sorry 

theorem intersection_diagonals_fixed_point (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (M N P Q : ℝ × ℝ) (h3 : ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1))
  (h4 : M.1 * M.1 / 4 + M.2 * M.2 / 3 = 1)
  (h5 : N.1 * N.1 / 4 + N.2 * N.2 / 3 = 1)
  (h6 : P.1 * P.1 / 4 + P.2 * P.2 / 3 = 1)
  (h7 : Q.1 * Q.1 / 4 + Q.2 * Q.2 / 3 = 1)
  (h8 : (M.1 - N.1) = (Q.1 - P.1))
  (h9 : M.2 = -Q.2)
  (h10 : N.2 = -P.2)
  (h11 : M.2 ≠ 0)
  (h12 : N.2 ≠ 0)
  (h13 : 4 < 0)
  (h14 : ∃ t : ℝ, t ≠ 0 ∧ ((M.1 - P.1) * t) = 0 ∧ ((N.1 - Q.1) * t) = 0): 
  (∃ D : ℝ × ℝ, D = (1, 0)) :=
by 
  sorry 

end ellipse_equation_intersection_diagonals_fixed_point_l76_76314


namespace work_days_C_l76_76124

theorem work_days_C :
  let total_work := 1 in 
  let rate_A := 1 / 30 in
  let rate_B := 1 / 30 in
  let rate_C := 1 / 29.999999999999996 in
  let work_A := 10 * rate_A in
  let work_B := 10 * rate_B in
  let remaining_work := total_work - (work_A + work_B) in
  let days_C := remaining_work * 29.999999999999996 in
  days_C = 10 :=
by 
  sorry

end work_days_C_l76_76124


namespace opposite_of_neg_two_l76_76513

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76513


namespace total_cost_of_returned_packets_l76_76019

/--
  Martin bought 10 packets of milk with varying prices.
  The average price (arithmetic mean) of all the packets is 25¢.
  If Martin returned three packets to the retailer, and the average price of the remaining packets was 20¢,
  then the total cost, in cents, of the three returned milk packets is 110¢.
-/
theorem total_cost_of_returned_packets 
  (T10 : ℕ) (T7 : ℕ) (average_price_10 : T10 / 10 = 25)
  (average_price_7 : T7 / 7 = 20) :
  (T10 - T7 = 110) := 
sorry

end total_cost_of_returned_packets_l76_76019


namespace gcf_of_180_240_300_l76_76970

def prime_factors (n : ℕ) : ℕ → Prop :=
λ p, p ^ (nat.factorization n p)

def gcf (n1 n2 n3 : ℕ) : ℕ :=
nat.gcd n1 (nat.gcd n2 n3)

theorem gcf_of_180_240_300 : gcf 180 240 300 = 60 := by
  sorry

end gcf_of_180_240_300_l76_76970


namespace transylvanian_is_sane_human_l76_76112

def Transylvanian : Type := sorry -- Placeholder type for Transylvanian
def Human : Transylvanian → Prop := sorry
def Sane : Transylvanian → Prop := sorry
def InsaneVampire : Transylvanian → Prop := sorry

/-- The Transylvanian stated: "Either I am a human, or I am sane." -/
axiom statement (T : Transylvanian) : Human T ∨ Sane T

/-- Insane vampires only make true statements. -/
axiom insane_vampire_truth (T : Transylvanian) : InsaneVampire T → (Human T ∨ Sane T)

/-- Insane vampires cannot be sane or human. -/
axiom insane_vampire_condition (T : Transylvanian) : InsaneVampire T → ¬ Human T ∧ ¬ Sane T

theorem transylvanian_is_sane_human (T : Transylvanian) :
  ¬ (InsaneVampire T) → (Human T ∧ Sane T) := sorry

end transylvanian_is_sane_human_l76_76112


namespace class_size_l76_76393

theorem class_size (g : ℕ) (h1 : g + (g + 3) = 44) (h2 : g^2 + (g + 3)^2 = 540) : g + (g + 3) = 44 :=
by
  sorry

end class_size_l76_76393


namespace sum_geq_three_implies_one_geq_two_l76_76958

theorem sum_geq_three_implies_one_geq_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by { sorry }

end sum_geq_three_implies_one_geq_two_l76_76958


namespace find_radius_and_angle_l76_76606

-- Definitions within the given problem
variables {A B C D K T O O₁ O₂ : Point}

-- Conditions definitions
def quadrilateral_inscribed (A B C D : Point) (O : Point) := 
  ∃ (circle : Circle), circle.contains A ∧ circle.contains B ∧ circle.contains C ∧ circle.contains D ∧ circle.center = O

def circles_equal_radius (O₁ O₂ : Point) (r : ℝ) := (∃ (Ω₁ Ω₂ : Circle), Ω₁.radius = r ∧ Ω₂.radius = r ∧ Ω₁.center = O₁ ∧ Ω₂.center = O₂)

def angles_circle_tangent (A D K C T : Point) (Ω₁ Ω₂ : Circle) :=
  ∃ (angleBAD angleBCD: Angle),
  Ω₁.is_tangent_to AD K ∧ Ω₂.is_tangent_to BC T ∧ Ω₁.inscribed_in angleBAD ∧ Ω₂.inscribed_in angleBCD ∧ AK = 2 ∧ CT = 8

def circumcenter (O₂ : Point) (B O C : Point) := Collinear O O₂ (midpoint B C)

-- Proof Problem
theorem find_radius_and_angle (A B C D K T O O₁ O₂ : Point)
  (h1 : quadrilateral_inscribed A B C D O)
  (h2 : ∃ r, circles_equal_radius O₁ O₂ r)
  (h3 : angles_circle_tangent A D K C T Ω₁ Ω₂)
  (h4 : circumcenter O₂ B O C):
  (radius Ω₁ = 4) ∧ 
  (angle B D C = arctan ( ( sqrt 5 - 1 ) / 2 ) ∨ angle B D C = π - arctan ( ( sqrt 5 + 1 ) / 2 )) :=
sorry

end find_radius_and_angle_l76_76606


namespace max_discount_rate_l76_76192

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76192


namespace train_cross_signal_pole_in_18_sec_l76_76611

variable (length_train length_platform time_platform_cross length_signal_pole : ℝ)

def speed (length_train length_platform time_platform_cross : ℝ) : ℝ := 
  (length_train + length_platform) / time_platform_cross

def time_signal_pole_cross (length_train length_platform time_platform_cross length_signal_pole : ℝ) : ℝ :=
  length_signal_pole / (speed length_train length_platform time_platform_cross)

theorem train_cross_signal_pole_in_18_sec (len_train : length_train = 600) (len_platform : length_platform = 700) (time_cross_platform : time_platform_cross = 39) (len_signal_pole : length_signal_pole = 600) :
  time_signal_pole_cross length_train length_platform time_platform_cross length_signal_pole = 18 :=
by
  simp [length_train, length_platform, time_platform_cross, length_signal_pole]
  sorry

end train_cross_signal_pole_in_18_sec_l76_76611


namespace percent_decrease_in_price_l76_76409

theorem percent_decrease_in_price 
  (price_last_week : ℝ := 7/3)
  (price_this_week : ℝ := 8/6) :
  (price_last_week - price_this_week) / price_last_week * 100 ≈ 42.9 :=
by
  sorry

end percent_decrease_in_price_l76_76409


namespace g_value_l76_76423

theorem g_value (g : ℝ → ℝ) (h : ∀ x y : ℝ, g ((x - y) ^ 2) = g x ^ 2 - 3 * x * g y + y ^ 2) :
  let values := {g 1 | c ∈ ({0, 1} : set ℝ) ∧ (∀ x : ℝ, g x = x + c)} in
  let n := values.to_finset.card in
  let s := values.sum id in
  n * s = 6 :=
by
  sorry

end g_value_l76_76423


namespace find_x_l76_76585

theorem find_x :
  ∃ x : ℤ, x + 3 * 12 + 3 * 13 + 3 * 16 = 134 ∧ x = 11 :=
by
  use 11
  calc
    11 + 3 * 12 + 3 * 13 + 3 * 16
        = 11 + 36 + 3 * 13 + 3 * 16 : by rw [three_mul_12]
    ... = 11 + 36 + 39 + 3 * 16 : by rw [three_mul_13]
    ... = 11 + 36 + 39 + 48 : by rw [three_mul_16]
    ... = 134 : by norm_num
  sorry

end find_x_l76_76585


namespace max_min_x2_sub_xy_add_y2_l76_76869

/-- Given a point \((x, y)\) on the curve defined by \( |5x + y| + |5x - y| = 20 \), prove that the maximum value of \(x^2 - xy + y^2\) is 124 and the minimum value is 3. -/
theorem max_min_x2_sub_xy_add_y2 (x y : ℝ) (h : abs (5 * x + y) + abs (5 * x - y) = 20) :
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_sub_xy_add_y2_l76_76869


namespace relationship_abc_l76_76318

-- Assumptions about the function f
variables {f : ℝ → ℝ}
hypothesis h1 : ∀ x, f x = f (-x)
hypothesis h2 : ∀ x, x < 0 → f x + x * (deriv^[2] f x) < 0

-- Definitions of a, b, c
noncomputable def a := (2^(0.6)) * f (2^(0.6))
noncomputable def b := (Real.log 2) * f (Real.log 2)
noncomputable def c := (Real.log 2⁻³) * f (Real.log 2⁻³)


theorem relationship_abc : c > b > a :=
sorry

end relationship_abc_l76_76318


namespace chess_tournament_l76_76824

theorem chess_tournament (n : ℕ) (points : Fin n → ℕ) (games_with_white : Fin n → Fin n → ℕ) :
  (∀ i j : Fin n, i ≠ j → games_with_white i j + games_with_white j i = 1) →
  (∀ i : Fin n, points i = n - 1) →
  ∃ i j : Fin n, i ≠ j ∧ ∑ k : Fin n, games_with_white i k = ∑ k : Fin n, games_with_white j k :=
by
  sorry

end chess_tournament_l76_76824


namespace percentage_increase_l76_76853

theorem percentage_increase (x distance_first_hour distance_second_hour distance_third_hour total_distance: ℕ) 
  (h1 : distance_second_hour = 24)
  (h2 : distance_third_hour = distance_second_hour + distance_second_hour / 4) 
  (h3 : total_distance = x + distance_second_hour + distance_third_hour) 
  (h4 : total_distance = 74) :
  (distance_second_hour - x) * 100 / x = 20 :=
by
  -- Definitions based on conditions
  rw [h1] at h2
  have h5 : distance_third_hour = 30 := by rw [h2, ←nat.add_comm, add_assoc, nat.mul_div_cancel, nat.div_self] <;> assumption
  rw [h5] at h3 h4
  have h6 : x + 54 = 74 := by linarith
  have h7 : x = 20 := by linarith
  sorry

end percentage_increase_l76_76853


namespace opposite_of_neg_two_l76_76550

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76550


namespace exp_neg_eq_l76_76758

theorem exp_neg_eq (θ φ : ℝ) (h : Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 2 : ℂ) + (1 / 3 : ℂ) * Complex.I) :
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 2 : ℂ) - (1 / 3 : ℂ) * Complex.I :=
by sorry

end exp_neg_eq_l76_76758


namespace part_a_part_b_part_c_l76_76119

def is_digit_5 (d : ℕ) : Prop := d = 5
def is_nonzero_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9
def is_valid_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9

theorem part_a (count_exactly_one_5 : ℕ) :
  (∃ f : fin 5 → ℕ, 
    (∀ (i : fin 5), is_valid_digit (f i)) ∧ 
    is_nonzero_digit (f 0) ∧ 
    (finset.filter is_digit_5 (finset.univ.image f)).card = 1
  ) → count_exactly_one_5 = 41 * 9^3 := 
sorry

theorem part_b (count_at_most_one_5 : ℕ) :
  (∃ f : fin 5 → ℕ, 
    (∀ (i : fin 5), is_valid_digit (f i)) ∧ 
    is_nonzero_digit (f 0) ∧ 
    (finset.filter is_digit_5 (finset.univ.image f)).card ≤ 1
  ) → count_at_most_one_5 = 113 * 9^3 := 
sorry

theorem part_c (total_five_digit_numbers : ℕ) :
  (∃ f : fin 5 → ℕ, 
    (∀ (i : fin 5), is_valid_digit (f i)) ∧ 
    is_nonzero_digit (f 0)
  ) →
  (∃ g : fin 5 → ℕ, 
    (∀ (i : fin 5), is_valid_digit (g i)) ∧ 
    is_nonzero_digit (g 0) ∧ 
    (finset.filter is_digit_5 (finset.univ.image g)).card = 0
  ) →
  total_five_digit_numbers - count_at_most_one_0_5 = 37512 :=
sorry

end part_a_part_b_part_c_l76_76119


namespace pentagon_ratio_l76_76449

open Real

section
variables (A B C D E F G H I J : Point)
-- Given conditions
def rectangles_placed_adjacent := 
  (segment_length A B = 2) ∧ (segment_length G H = 2)
  ∧ (segment_length B C = 1) ∧ (segment_length H I = 1)
  ∧ (point_on_segment_fraction C H ((1:Real)/3)) 
  ∧ (point_on_segment_fraction D H ((1:Real)/3))

-- area calculations
def area_pentagon (A J I C B : Point) := (3 / 2 : Real) - (1 / 3 : Real) - (1 / 3 : Real)

-- sum of areas of the three rectangles
def total_area_rectangles := 6

-- proof statement
theorem pentagon_ratio (h : rectangles_placed_adjacent A B C D E F G H I J) :
  area_pentagon A J I C B / total_area_rectangles = (5 / 36 : Real) :=
sorry
end

end pentagon_ratio_l76_76449


namespace maximize_variance_l76_76415

noncomputable def bernoulli_vars (n : ℕ) (p : Fin n → ℝ) : Prop :=
∀ i, 0 ≤ p i ∧ p i ≤ 1

noncomputable def avg_is_a (n : ℕ) (p : Fin n → ℝ) (a : ℝ) : Prop :=
1 / n * ∑ i : Fin n, p i = a

noncomputable def variance_of_sum (n : ℕ) (p : Fin n → ℝ) : ℝ :=
∑ i : Fin n, p i * (1 - p i)

theorem maximize_variance 
    (n : ℕ) (p : Fin n → ℝ) (a : ℝ)
    (h1 : bernoulli_vars n p)
    (h2 : 0 < a ∧ a < 1)
    (h3 : avg_is_a n p a) : 
    variance_of_sum n p ≤ n * a * (1 - a) :=
sorry

end maximize_variance_l76_76415


namespace opposite_of_neg_two_is_two_l76_76522

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76522


namespace quadratic_function_is_parabola_l76_76035

def quadratic_function (x : ℝ) : ℝ :=
  3 * (x - 2) ^ 2 + 6

theorem quadratic_function_is_parabola :
  ∃ A B, quadratic_function = λ x, A * (x - B) ^ 2 + C := by
  sorry

end quadratic_function_is_parabola_l76_76035


namespace max_discount_rate_l76_76187

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76187


namespace find_phi_l76_76483

def symmetry_axis_phi (phi : ℝ) : Prop := 
  ∀ k : ℤ, let x := (k * Real.pi - phi) / 3 + Real.pi / 6 in x = Real.pi / 12

theorem find_phi (phi : ℝ) (h1 : |phi| < Real.pi / 2) 
  (h2 : symmetry_axis_phi phi) : 
  phi = Real.pi / 4 := 
by 
  sorry

end find_phi_l76_76483


namespace g_75_is_1997_l76_76658

def g : ℤ → ℤ
| n := if n ≥ 2000 then n - 4 else g (g (n + 7))

theorem g_75_is_1997 : g 75 = 1997 :=
by sorry

end g_75_is_1997_l76_76658


namespace b_nonnegative_range_length_segment_positive_f_m1_m2_plus_3_l76_76728

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

variables (a b c m1 m2 : ℝ)
variable (h_abc : a > b ∧ b > c)

-- Given conditions
variable (h1 : a^2 + (f a b c m1 + f a b c m2) * a + f a b c m1 * f a b c m2 = 0)
variable (h2 : f a b c 1 = 0)

-- Prove b ≥ 0
theorem b_nonnegative (h_abc : a > b ∧ b > c) (h1 : a^2 + (f a b c m1 + f a b c m2) * a + f a b c m1 * f a b c m2 = 0)
  (h2 : f a b c 1 = 0) : b ≥ 0 :=
sorry

-- Prove the range of possible lengths of the intercepted segment
theorem range_length_segment (h_abc : a > b ∧ b > c) (h1 : a^2 + (f a b c m1 + f a b c m2) * a + f a b c m1 * f a b c m2 = 0)
  (h2 : f a b c 1 = 0) : ∃ l, l ∈ set.Ico 2 3 :=
sorry

-- Prove at least one of f(m1 + 3) or f(m2 + 3) is positive
theorem positive_f_m1_m2_plus_3 (h_abc : a > b ∧ b > c) (h1 : a^2 + (f a b c m1 + f a b c m2) * a + f a b c m1 * f a b c m2 = 0)
  (h2 : f a b c 1 = 0) : f a b c (m1 + 3) > 0 ∨ f a b c (m2 + 3) > 0 :=
sorry

end b_nonnegative_range_length_segment_positive_f_m1_m2_plus_3_l76_76728


namespace problem1_problem2_problem3_l76_76488

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f x = -2*x^2 - 3*x + 1) : f (-2) = -1 :=
by {
  rw h,
  sorry
}

-- Problem (2)
theorem problem2 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a*x^3 + 2*x^2 - a*x - 6) (hf : f (1/2) = a) : a = -4 :=
by {
  rw [h, hf],
  sorry
}

-- Problem (3)
theorem problem3 (a b : ℝ) (f : ℝ → ℝ) (k : ℚ) (h : ∀ x, f x = (2 * b * x + a)/3 - (x - b * k)/6 -2) (hf : f 1 = 0) : a = 6.5 ∧ b = -4 :=
by {
  rw [h, hf],
  sorry
}

end problem1_problem2_problem3_l76_76488


namespace maximum_discount_rate_l76_76179

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76179


namespace associative_for_t_zero_associative_for_t_two_l76_76680

def binary_operation (t : ℝ) (x y : ℝ) :=
  x + y + t * Real.sqrt (x * y)

theorem associative_for_t_zero (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  binary_operation 0 (binary_operation 0 x y) z = binary_operation 0 x (binary_operation 0 y z) :=
sorry

theorem associative_for_t_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  binary_operation 2 (binary_operation 2 x y) z = binary_operation 2 x (binary_operation 2 y z) :=
sorry

end associative_for_t_zero_associative_for_t_two_l76_76680


namespace degree_of_p_x2_q_x4_l76_76464

-- Definitions to capture the given problem conditions
def is_degree_3 (p : Polynomial ℝ) : Prop := p.degree = 3
def is_degree_6 (q : Polynomial ℝ) : Prop := q.degree = 6

-- Statement of the proof problem
theorem degree_of_p_x2_q_x4 (p q : Polynomial ℝ) (hp : is_degree_3 p) (hq : is_degree_6 q) :
  (p.comp (Polynomial.X ^ 2) * q.comp (Polynomial.X ^ 4)).degree = 30 :=
sorry

end degree_of_p_x2_q_x4_l76_76464


namespace terms_before_five_l76_76343

theorem terms_before_five (a₁ : ℤ) (d : ℤ) (n : ℤ) :
  a₁ = 75 → d = -5 → (a₁ + (n - 1) * d = 5) → n - 1 = 14 :=
by
  intros h1 h2 h3
  sorry

end terms_before_five_l76_76343


namespace infinite_either_interval_exists_rational_infinite_elements_l76_76238

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end infinite_either_interval_exists_rational_infinite_elements_l76_76238


namespace range_of_a_l76_76325

def A (a : ℝ) : set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) (x : ℝ) : set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}
def C (a : ℝ) : set ℝ := {z | ∃ x ∈ A a, z = x^2}

theorem range_of_a (a : ℝ) : (C a ⊆ B a (x := x)) → 1/2 ≤ a ∧ a ≤ 3 :=
begin
  sorry
end

end range_of_a_l76_76325


namespace count_sequences_returning_to_original_l76_76044

-- Definitions of the transformations as functions on the vertices could be implemented here, but for the simplicity of the example, we use generic terms.
def L (p : ℕ) := sorry
def R (p : ℕ) := sorry
def H (p : ℕ) := sorry
def V (p : ℕ) := sorry

-- Assume E(2,2), F(-2,2), G(-2,-2), H(2,-2) are represented by their indices 1, 2, 3, 4 respectively
def E := 1
def F := 2
def G := 3
def H := 4

noncomputable def is_original_configuration (trans_seq : List (ℕ → ℕ)) : Prop :=
  let final_positions := trans_seq.foldl (λ positions trans => positions.map trans) [E, F, G, H]
  final_positions = [1, 2, 3, 4]

theorem count_sequences_returning_to_original : 
  ∃ n : ℕ, n = 455 ∧ ∀ (seq : List (ℕ → ℕ)), 
    seq.length = 24 ∧ 
    (∀ f ∈ seq, f = L ∨ f = R ∨ f = H ∨ f = V) →
    (is_original_configuration seq ↔ seq.count L % 4 = 0 ∧ seq.count R % 4 = 0 ∧ seq.count H % 2 = 0 ∧ seq.count V % 2 = 0) :=
sorry

end count_sequences_returning_to_original_l76_76044


namespace range_of_x_l76_76731

theorem range_of_x (x : ℝ)
  (h : ∀ (a b : ℝ), a^2 + b^2 = 1 → a + sqrt 3 * b ≤ abs (x^2 - 1)) :
  x ≤ -sqrt 3 ∨ x ≥ sqrt 3 :=
sorry

end range_of_x_l76_76731


namespace decompose_space_tetrahedra_octahedra_l76_76847

theorem decompose_space_tetrahedra_octahedra : 
  ∃ (partition : set (set ℝ^3)), 
    (∀ p ∈ partition, p ∈ {regular_tetrahedron, regular_octahedron}) →
    (∀ x y ∈ partition, ¬(x ∩ y ≠ ∅) → x = y) →
    (⋃ p ∈ partition, p) = univ := 
sorry

end decompose_space_tetrahedra_octahedra_l76_76847


namespace equal_incircle_excircle_radius_implies_quarter_height_l76_76705

variables {A B C D : Point}
variables (h r : ℝ)

-- Define the triangle ABC as isosceles with base AB.
def is_isosceles (A B C : Point) := dist B A = dist C A ∧ dist B C = dist C A

-- Define the height of the isosceles triangle ABC.
def height (A B C : Point) (h : ℝ) := ∃ M, midpoint A B M ∧ dist C M = h

-- Define the condition that the radius of the incircle and the excircle are equal.
def equal_incircle_excircle_radius (B C D : Point) (r : ℝ) := 
incircle_radius B C D = r ∧ excircle_radius A C D = r

theorem equal_incircle_excircle_radius_implies_quarter_height
(isosceles_triangle : is_isosceles A B C)
(point_on_base : ∃ D, collinear A B D)
(equal_radii : equal_incircle_excircle_radius B C D r)
(triangle_height : height A B C h) :
r = h / 4 :=
by
  sorry

end equal_incircle_excircle_radius_implies_quarter_height_l76_76705


namespace deduce_reasoning_optionC_is_deductive_l76_76099

-- Definitions for each option as conditions
def optionA := "Inferring that the sum of the interior angles of all triangles is 180° based on the fact that the sum of the interior angles of equilateral triangles and isosceles triangles is 180°."

def optionB := "Inferring that the sum of the areas of any three faces of a tetrahedron is greater than the area of the fourth face based on the fact that the sum of the lengths of two sides of a triangle is greater than the length of the third side."

def optionC := "Inferring that the diagonals of a parallelogram bisect each other, a rhombus is a parallelogram, so the diagonals of a rhombus bisect each other."

def optionD := "Inducing the general formula of a sequence {a_n} from the given conditions a_1 = 1 and a_n = 1/2 (a_{n-1} + 1/a_{n-1}) for n ≥ 2."

-- Target proof statement
theorem deduce_reasoning_optionC_is_deductive :
  optionC = "Inferring that the diagonals of a parallelogram bisect each other, a rhombus is a parallelogram, so the diagonals of a rhombus bisect each other." → 
  "optionC represents deductive reasoning" := 
by
  sorry

end deduce_reasoning_optionC_is_deductive_l76_76099


namespace max_discount_rate_l76_76151

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76151


namespace zero_function_solution_l76_76246

theorem zero_function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3 * x^2 * f (x) * f (y) + 3 * (f (x) * f (y))^2 + y^6 * f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_solution_l76_76246


namespace tan_theta_value_cos_2theta_minus_pi_over_3_value_l76_76300

open Real

variables (θ : ℝ)
noncomputable def sin_val : ℝ := 3 / 5

-- Conditions
axiom sin_theta : sin θ = sin_val
axiom theta_quadrant : θ ∈ Icc (π / 2) π

-- Proof statements
theorem tan_theta_value : tan θ = -3 / 4 := by
  sorry

theorem cos_2theta_minus_pi_over_3_value : cos (2 * θ - π / 3) = (7 - 24 * sqrt 3) / 50 := by
  sorry

end tan_theta_value_cos_2theta_minus_pi_over_3_value_l76_76300


namespace opposite_of_neg_2_is_2_l76_76494

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76494


namespace average_of_remaining_primes_l76_76811

theorem average_of_remaining_primes (avg30: ℕ) (avg15: ℕ) (h1 : avg30 = 110) (h2 : avg15 = 95) : 
  ((30 * avg30 - 15 * avg15) / 15) = 125 := 
by
  -- Proof
  sorry

end average_of_remaining_primes_l76_76811


namespace digits_difference_l76_76911

/-- Given a two-digit number represented as 10X + Y and the number obtained by interchanging its digits as 10Y + X,
    if the difference between the original number and the interchanged number is 81, 
    then the difference between the tens digit X and the units digit Y is 9. -/
theorem digits_difference (X Y : ℕ) (h : (10 * X + Y) - (10 * Y + X) = 81) : X - Y = 9 :=
by
  sorry

end digits_difference_l76_76911


namespace soldiers_to_add_l76_76764

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76764


namespace sum_of_coeffs_eq_cos_two_alpha_l76_76756

theorem sum_of_coeffs_eq_cos_two_alpha (a b : ℝ) (α : ℝ) 
  (h_roots : ∀ x : ℝ, x^2 + a*x + b = 0 → x = sin α ∨ x = cos α) : 
  a + b = cos (2 * α) := 
sorry

end sum_of_coeffs_eq_cos_two_alpha_l76_76756


namespace abs_a_lt_abs_b_add_abs_c_l76_76380

theorem abs_a_lt_abs_b_add_abs_c (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_a_lt_abs_b_add_abs_c_l76_76380


namespace thirty_percent_less_than_ninety_l76_76946

theorem thirty_percent_less_than_ninety : 
  ∃ n : ℝ, (3 / 2) * n = 63 ∧ n = 42 :=
by
  have h : (3 / 2) * 42 = 63,
  { norm_num },
  use 42,
  exact ⟨h, rfl⟩

end thirty_percent_less_than_ninety_l76_76946


namespace find_f_2016_l76_76727

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem find_f_2016 (a α b β : ℝ) (h : f 3 a α b β = 3) : f 2016 a α b β = -3 :=
by
  sorry

end find_f_2016_l76_76727


namespace centroid_not_in_bisector_base_triangle_l76_76403

theorem centroid_not_in_bisector_base_triangle 
  (ABC : Type) [triangle ABC]
  (G : centroid ABC)
  (DE : base_of_angle_bisectors ABC) :
  ¬ G ∈ DE :=
sorry

end centroid_not_in_bisector_base_triangle_l76_76403


namespace ones_digit_of_prime_sequence_l76_76686

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l76_76686


namespace max_discount_rate_l76_76133

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76133


namespace gcd_multiple_not_prime_less_70_l76_76580

theorem gcd_multiple_not_prime_less_70 : ∀ n, n < 70 ∧ (∃ k, n = k * Nat.lcm 10 15) → ¬ Nat.Prime n := by
  intro n h
  cases h with h1 h2
  rcases h2 with ⟨k, rfl⟩
  have lcm_value : Nat.lcm 10 15 = 30 := by
    rw [Nat.lcm, Nat.gcd_rec, Nat.gcd_one_right, Nat.mul_comm, Nat.div_one, Nat.gcd_mul_left, Nat.gcd_mul_right_left, Nat.gcd_self, Nat.mul_one, Nat.gcd_comm]
    norm_num
  have multiples_less_70 := ((gt_one_iff_div.1 (norm_num.divide_iff.1 ((Nat.Divisors.mul_left _ _).2 lcm_value.ne_zero.le (le_of_dvd (Nat.Prime.not_zero (Nat.Prime.of_mem_divisors lcm_value.zero_ne)).not_zero_zero) ((le_off_pred zero_pred_gt1).mp (zero_lt_pred lcm_value.gt))).2 (le_of_lt_succ pred.compl.pred_mem.compl0)).
  sorry

end gcd_multiple_not_prime_less_70_l76_76580


namespace Jackson_money_is_125_l76_76851

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l76_76851


namespace greatest_possible_y_l76_76905

theorem greatest_possible_y (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 :=
sorry

end greatest_possible_y_l76_76905


namespace count_diff_squares_not_representable_1_to_1000_l76_76338

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l76_76338


namespace percent_boys_study_science_l76_76388

theorem percent_boys_study_science (total_boys camp: ℕ) (percent_school_A: ℝ) (not_study_science: ℕ):
    total_boys = 200 →
    percent_school_A = 0.20 →
    not_study_science = 28 →
    let boys_school_A := percent_school_A * (total_boys: ℝ);
    let study_science := boys_school_A - not_study_science;
    let percentage := (study_science / boys_school_A) * 100 in
    percentage = 30 := by
  sorry

end percent_boys_study_science_l76_76388


namespace opposite_of_neg_two_is_two_l76_76517

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76517


namespace min_soldiers_to_add_l76_76810

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76810


namespace range_of_a_l76_76701

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
| 0       := 0
| 1       := a
| (n + 2) := 6 * n + 3 + (sequence_a a (n + 1))

noncomputable def sum_S (a : ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := sum_S a n + sequence_a a (n + 1)

theorem range_of_a (a : ℝ) 
  (H1 : ∀ n ≥ 2, sum_S a n + sum_S a (n - 1) = 3 * (n * n))
  (H2 : ∀ n : ℕ, a_n a n < a_n a (n + 1)) :
  9 / 4 < a ∧ a < 15 / 4 :=
sorry

end range_of_a_l76_76701


namespace repeating_decimal_to_fraction_l76_76961

theorem repeating_decimal_to_fraction : ∃ (r : ℚ), r = 0.4 + 0.0036 * (1/(1 - 0.01)) ∧ r = 42 / 55 :=
by
  sorry

end repeating_decimal_to_fraction_l76_76961


namespace words_difference_proof_l76_76874

noncomputable def words_per_min_both_hands := 11
noncomputable def words_written_right_hand := 10 * (11 / 2)
noncomputable def words_written_left_hand := 15 * (11 / 2)
noncomputable def words_written_both_hands := 8 * 11
noncomputable def words_difference := words_written_right_hand - words_written_left_hand

theorem words_difference_proof : words_difference = -27.5 := by
  sorry

end words_difference_proof_l76_76874


namespace pqrsum_l76_76212

def sequence (t : ℕ → ℤ) : Prop :=
  t 1 = 14 ∧ ∀ k ≥ 2, t k = 24 - 5 * t (k - 1)

theorem pqrsum :
  (∃ (p q r : ℤ), ∀ (n : ℕ), n > 0 → (∃ t : ℕ → ℤ, sequence t ∧ t n = p * q^n + r)) →
  (-2 + -5 + 4 = -3) :=
by
  sorry

end pqrsum_l76_76212


namespace repeating_decimal_to_fraction_l76_76964

theorem repeating_decimal_to_fraction : 
  (x : ℝ) (h : x = 0.4 + 36 / (10^1 + 10^2 + 10^3 + ...)) : x = 24 / 55 :=
sorry

end repeating_decimal_to_fraction_l76_76964


namespace shares_owned_l76_76125

variables (expected_earnings : ℝ) (excess_dividend_per_share : ℝ) (additional_dividend_per_dime : ℝ)
          (actual_earnings : ℝ) (total_dividend_paid : ℝ)

def calculate_shares (expected_earnings actual_earnings excess_dividend_per_share additional_dividend_per_dime total_dividend_paid : ℝ) : ℝ :=
  let extra_earnings := actual_earnings - expected_earnings
  let extra_blocks := extra_earnings / excess_dividend_per_share
  let additional_dividend := extra_blocks * additional_dividend_per_dime
  let base_dividend := expected_earnings / 2
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend_paid / total_dividend_per_share

theorem shares_owned : calculate_shares 0.8 1.10 0.1 0.04 260 = 500 :=
sorry

end shares_owned_l76_76125


namespace find_ratio_l76_76904

-- Define the main theorem
theorem find_ratio k_exists
  (a1 a2 a3 b1 b2 b3 : ℕ) (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3 ∧ b1 ≠ b2 ∧ b2 ≠ b3 ∧ b1 ≠ b3)
  (h_pos_a1 : 0 < a1) (h_pos_a2 : 0 < a2) (h_pos_a3 : 0 < a3)
  (h_pos_b1 : 0 < b1) (h_pos_b2 : 0 < b2) (h_pos_b3 : 0 < b3)
  (h_condition : ∀ n : ℕ, 0 < n → (n + 1) * a1^n + n * a2^n + (n - 1) * a3^n ∣ (n + 1) * b1^n + n * b2^n + (n - 1) * b3^n) :
  ∃ k : ℕ, b1 = k * a1 ∧ b2 = k * a2 ∧ b3 = k * a3 :=
begin
  sorry
end

end find_ratio_l76_76904


namespace correct_statement_C_l76_76679

-- Define the function
def linear_function (x : ℝ) : ℝ := -3 * x + 1

-- Define the condition for statement C
def statement_C (x : ℝ) : Prop := x > 1 / 3 → linear_function x < 0

-- The theorem to be proved
theorem correct_statement_C : ∀ x : ℝ, statement_C x := by
  sorry

end correct_statement_C_l76_76679


namespace ada_original_seat_l76_76265

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end ada_original_seat_l76_76265


namespace max_median_of_groups_l76_76248

theorem max_median_of_groups (numbers : Finset ℕ) (h : numbers = Finset.range 51 \ {0}) :
  ∃ (groups : list (Finset ℕ)), 
  (∀ g ∈ groups, g.card = 5 ∧ ∀ i j : ℕ, i ≠ j → disjoint g (groups.nth_le (j % 10) (by linarith)) ∧ ∃ median, median ∈ g ∧ 
  (∀ m ∈ groups, (median_of_five m).max = 48)) :=
sorry

end max_median_of_groups_l76_76248


namespace opposite_of_neg_2_is_2_l76_76496

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76496


namespace calculate_transport_cost_l76_76471

def cost_per_kg := 24000 -- Cost per kilogram in dollars
def instrument_weight_g := 350 -- Weight of the instrument in grams
def grams_to_kilograms (g : ℕ) : ℝ := g / 1000.0 -- Conversion from grams to kilograms
def transport_cost (weight_kg : ℝ) (cost_per_kg : ℕ) : ℕ := (weight_kg * cost_per_kg).to_nat

theorem calculate_transport_cost : transport_cost (grams_to_kilograms instrument_weight_g) cost_per_kg = 8400 :=
by
  sorry

end calculate_transport_cost_l76_76471


namespace sequence_sum_l76_76732

theorem sequence_sum (a : ℕ → ℕ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = 1)
  (h₃ : a 3 = 2)
  (h₄ : ∀ n : ℕ, a (n+1) * a (n+2) * a (n+3) ≠ 1)
  (h₅ : ∀ n : ℕ, a n * a (n+1) * a (n+2) * a (n+3) = a 1 + a (n+1) + a (n+2) + a (n+3)) :
  ∑ i in finset.range 100, a (i + 1) = 200 :=
by
  sorry

end sequence_sum_l76_76732


namespace number_of_students_with_at_least_two_pets_l76_76389

-- Definitions for the sets of students
def total_students := 50
def dog_students := 35
def cat_students := 40
def rabbit_students := 10
def dog_and_cat_students := 20
def dog_and_rabbit_students := 5
def cat_and_rabbit_students := 0  -- Assuming minimal overlap

-- Problem Statement
theorem number_of_students_with_at_least_two_pets :
  (dog_and_cat_students + dog_and_rabbit_students + cat_and_rabbit_students) = 25 :=
by
  sorry

end number_of_students_with_at_least_two_pets_l76_76389


namespace marbles_per_pack_l76_76410

theorem marbles_per_pack (total_marbles : ℕ) (leo_packs manny_packs neil_packs total_packs : ℕ) 
(h1 : total_marbles = 400) 
(h2 : leo_packs = 25) 
(h3 : manny_packs = total_packs / 4) 
(h4 : neil_packs = total_packs / 8) 
(h5 : leo_packs + manny_packs + neil_packs = total_packs) : 
total_marbles / total_packs = 10 := 
by sorry

end marbles_per_pack_l76_76410


namespace domain_of_log_function_l76_76054

open Real

noncomputable def domain_of_function : Set ℝ :=
  {x | x > 2 ∨ x < -1}

theorem domain_of_log_function :
  ∀ x : ℝ, (x^2 - x - 2 > 0) ↔ (x > 2 ∨ x < -1) :=
by
  intro x
  exact sorry

end domain_of_log_function_l76_76054


namespace opposite_of_neg_two_is_two_l76_76536

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76536


namespace system_has_solution_l76_76102

theorem system_has_solution :
  ∃ (x y : ℤ), 3 * x + y = 3 ∧ 4 * x - y = 11 :=
by
  use 2, -3
  split
  · norm_num
  · norm_num

end system_has_solution_l76_76102


namespace solve_integers_l76_76042

theorem solve_integers (x y : ℤ) : 
  (∃ n : ℤ, x = 2 * n) → 3^x - 5 = 4 * y → ∃ n : ℤ, y = (3^(2 * n) - 5) / 4 :=
by 
  intro hx hy
  cases hx with n hn
  rw hn at hy
  use n
  rw [pow_mul, ←hy]
  sorry

end solve_integers_l76_76042


namespace a_2004_value_l76_76291

-- Define the sequence with its initial conditions
def a : ℕ → ℚ
| 0     := 1
| 1     := 1
| (n+2) := 1 / a (n+1) + a n

-- State the theorem to be proven
theorem a_2004_value : a 2004 = (2003!! / 2002!!) := by
  sorry

end a_2004_value_l76_76291


namespace smallest_integer_n_l76_76090

theorem smallest_integer_n (n : ℤ) (h : n^2 - 9 * n + 20 > 0) : n ≥ 6 := 
sorry

end smallest_integer_n_l76_76090


namespace tangent_identity_l76_76371

theorem tangent_identity (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tangent_identity_l76_76371


namespace positive_difference_of_perimeters_l76_76071

theorem positive_difference_of_perimeters :
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  (perimeter1 - perimeter2) = 4 :=
by
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  show (perimeter1 - perimeter2) = 4
  sorry

end positive_difference_of_perimeters_l76_76071


namespace proportional_function_l76_76308

theorem proportional_function (k m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = k * x) →
  f 2 = -4 →
  (∀ x, f x + m = -2 * x + m) →
  f 2 = -4 ∧ (f 1 + m = 1) →
  k = -2 ∧ m = 3 := 
by
  intros h1 h2 h3 h4
  sorry

end proportional_function_l76_76308


namespace convert_to_rectangular_coords_l76_76242

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)
open Real

theorem convert_to_rectangular_coords :
  let ρ := 5
  let θ := 7 * π / 4
  let φ := 3 * π / 4
  spherical_to_rectangular ρ θ φ = (5 / 2, -5 / 2, -5 * sqrt 2 / 2) :=
by {
  sorry
}

end convert_to_rectangular_coords_l76_76242


namespace count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76740

-- The problem states the following conditions
def is_divisible_by (m n : ℕ) : Prop := m % n = 0
def less_than_300 (n : ℕ) : Prop := n < 300

-- We want to prove the number of positive integers less than 300 that are divisible by 4, 6, and 10
theorem count_positive_integers_divisible_by_4_6_10_less_than_300 :
  (Finset.card (Finset.filter 
    (λ n, is_divisible_by n 4 ∧ is_divisible_by n 6 ∧ is_divisible_by n 10 ∧ less_than_300 n)
    ((Finset.range 300).filter (λ n, n ≠ 0)))) = 4 :=
by
  sorry

end count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76740


namespace ram_percentage_l76_76034

variable (marks_obtained : ℝ)
variable (total_marks : ℝ)

def percentage (marks_obtained total_marks : ℝ) : ℝ :=
  (marks_obtained / total_marks) * 100

theorem ram_percentage :
  marks_obtained = 450 ∧ total_marks = 500 → percentage marks_obtained total_marks = 90 :=
by
  sorry

end ram_percentage_l76_76034


namespace periodic_continued_fraction_is_quadratic_irrational_l76_76890

noncomputable def periodic_continued_fraction (P Q : ℕ → ℤ) (a_0 a_1 a_2 a_k : ℤ) : Prop :=
  ∃ α : ℝ,
    α = [a_0; a_1, a_2, ..., a_k, α] ∧
    α = (α * P (k - 1) + P (k - 2)) / (α * Q (k - 1) + Q (k - 2)) ∧
    ∃ c0 c1 c2 : ℤ, (α : ℤ)^2 * Q (k - 1) + (α : ℤ) * (Q (k - 2) - P (k - 1)) - P (k - 2) = 0

theorem periodic_continued_fraction_is_quadratic_irrational 
  (P Q : ℕ → ℤ) (a_0 a_1 a_2 a_k : ℤ) :
  periodic_continued_fraction P Q a_0 a_1 a_2 a_k :=
by
  sorry

end periodic_continued_fraction_is_quadratic_irrational_l76_76890


namespace statement_1_statement_2_statement_3_statement_4_l76_76821

theorem statement_1 (f : ℝ → ℝ) (h : f (-2) = f 2) : ¬ ∀ x, f (-x) = f x := sorry

theorem statement_2 {f : ℝ → ℝ}
  (h1 : ∀ x y, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y)
  (h2 : ∀ x y, x < y → 0 ≤ x → 0 ≤ y → f x ≤ f y) :
  ∀ x y, x < y → f x ≤ f y :=
sorry

theorem statement_3 {f : ℝ → ℝ} {a b c : ℝ} (h1 : a < c) (h2 : c < b)
  (h3 : ∀ x y, a ≤ x → x < y → y < c → f x ≤ f y)
  (h4 : ∀ x y, c ≤ x → x < y → y ≤ b → f x ≥ f y) : 
  ∀ x ∈ set.Icc a b, f x ≤ f c :=
sorry

theorem statement_4 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (sqrt x1 + sqrt x2) / 2 ≤ sqrt ((x1 + x2) / 2) :=
sorry

end statement_1_statement_2_statement_3_statement_4_l76_76821


namespace evaluate_expression_at_neg_third_l76_76039

variable (a : ℚ)

theorem evaluate_expression_at_neg_third (h : a = -1 / 3) :
  (2 - a) * (2 + a) - 2 * a * (a + 3) + 3 * a^2 = 6 :=
by
  rw h
  sorry

end evaluate_expression_at_neg_third_l76_76039


namespace largest_number_is_sqrt_7_l76_76073

noncomputable def largest_root (d e f : ℝ) : ℝ :=
if d ≥ e ∧ d ≥ f then d else if e ≥ d ∧ e ≥ f then e else f

theorem largest_number_is_sqrt_7 :
  ∃ (d e f : ℝ), (d + e + f = 3) ∧ (d * e + d * f + e * f = -14) ∧ (d * e * f = 21) ∧ (largest_root d e f = Real.sqrt 7) :=
sorry

end largest_number_is_sqrt_7_l76_76073


namespace number_of_subsets_l76_76753

-- Define the set
def my_set : Set ℕ := {1, 2, 3}

-- Theorem statement
theorem number_of_subsets : Finset.card (Finset.powerset {1, 2, 3}) = 8 :=
by
  sorry

end number_of_subsets_l76_76753


namespace round_14_9953_to_nearest_tenth_l76_76455

-- Definition that states the conditions for rounding.
def round_to_nearest_tenth (x : ℝ) : ℝ :=
  (Real.floor (10 * x) / 10) + 
  if (10 * x - Real.floor (10 * x)) >= 0.5 then 0.1 else 0

-- Theorem statement to prove that rounding 14.9953 to the nearest tenth results in 15.0
theorem round_14_9953_to_nearest_tenth :
  round_to_nearest_tenth 14.9953 = 15.0 :=
sorry

end round_14_9953_to_nearest_tenth_l76_76455


namespace equilateral_triangle_condition_l76_76702

theorem equilateral_triangle_condition (ABC : Triangle) 
  (h : ∀ (P : Point), P ∈ inside ABC → (PA P ABC) + (PB P ABC) > (PC P ABC) 
                       ∧ (PB P ABC) + (PC P ABC) > (PA P ABC) 
                       ∧ (PC P ABC) + (PA P ABC) > (PB P ABC)) : 
  is_equilateral ABC :=
sorry

end equilateral_triangle_condition_l76_76702


namespace largest_factor_11_fact_6k_plus_1_l76_76224

theorem largest_factor_11_fact_6k_plus_1 :
  ∃ k : ℕ, (k > 0) ∧ (385 = 6 * k + 1) ∧ ∀ m : ℕ, (m > 0) ∧ (m ∣ 11 !) ∧ (∃ l : ℕ, m = 6 * l + 1) → (m ≤ 385) := by
  sorry

end largest_factor_11_fact_6k_plus_1_l76_76224


namespace younger_son_age_30_years_later_eq_60_l76_76919

variable (age_diff : ℕ) (elder_age : ℕ) (younger_age_30_years_later : ℕ)

-- Conditions
axiom h1 : age_diff = 10
axiom h2 : elder_age = 40

-- Definition of younger son's current age
def younger_age : ℕ := elder_age - age_diff

-- Definition of younger son's age 30 years from now
def younger_age_future : ℕ := younger_age + 30

-- Proving the required statement
theorem younger_son_age_30_years_later_eq_60 (h_age_diff : age_diff = 10) (h_elder_age : elder_age = 40) :
  younger_age_future elder_age age_diff = 60 :=
by
  unfold younger_age
  unfold younger_age_future
  rw [h_age_diff, h_elder_age]
  sorry

end younger_son_age_30_years_later_eq_60_l76_76919


namespace opposite_of_neg_two_is_two_l76_76521

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76521


namespace opposite_of_neg_two_l76_76545

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76545


namespace equilateral_triangle_cd_l76_76557

theorem equilateral_triangle_cd (c d : ℝ) (h : (c, 15), (d, 43) are the vertices of an equilateral triangle) : 
  c * d = (5051 - 3195 * Real.sqrt 3) / (18 * Real.sqrt 3) :=
by
  sorry

end equilateral_triangle_cd_l76_76557


namespace find_BI_l76_76951

variables (A B C I : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space I]

-- Conditions
variables {AB AC BC: ℝ}
variable (triangle_ABC : AB = 28 ∧ AC = 24 ∧ BC = 20)
variable (I_incenter : ∃ (I : A), is_intersection_of_angle_bisectors I A B C)

-- Theorem statement
theorem find_BI
  (hABC : triangle_ABC)
  (hI : I_incenter) :
  BI = 2 * sqrt 11 :=
sorry

end find_BI_l76_76951


namespace albert_needs_more_money_l76_76219

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l76_76219


namespace polygon_sides_l76_76208

-- Definition: the interior angle of the polygon is 140 degrees
def interior_angle : ℝ := 140

-- Definition: the measure of each exterior angle (in degrees)
def exterior_angle := 180 - interior_angle

-- Theorem to prove the number of sides
theorem polygon_sides (h1 : interior_angle = 140) : 360 / exterior_angle = 9 :=
by
-- Since, by definition, exterior_angle = 40 degrees
have h2 : exterior_angle = 40 := by rw [exterior_angle, h1]
-- Plugging this value in, we get:
calc
  360 / exterior_angle
      = 360 / 40 : by rw h2
  ... = 9       : by norm_num -- This should verify the arithmetic
sorry -- This proof is left as an exercise.

end polygon_sides_l76_76208


namespace min_M_l76_76861

def frac_part (x : ℝ) : ℝ := x - ⌊x⌋

def f_a_b (a b x : ℝ) : ℝ := frac_part (x + a) + 2 * frac_part (x + b)

def range_f (a b : ℝ) := {y : ℝ | ∃ x : ℝ, y = f_a_b a b x}

def M_a_b (a b : ℝ) := Sup (range_f a b)

theorem min_M : (inf (λ (a b : ℝ), M_a_b a b) = 7/3) :=
by {
  have h : ∀ (a b : ℝ), M_a_b a b ≥ 7/3,
  { intros a b,
    sorry -- proof steps based on the given solution steps
  },
  have h' : ∃ (a b : ℝ), M_a_b a b = 7/3,
  { use (1/3),
    use 0,
    sorry -- proof steps based on the given solution steps
  },
  exact infi_eq_of_forall_ge_of_le ⟨1/3, 0⟩ 7/3 h h'
}

end min_M_l76_76861


namespace regression_residual_l76_76934

theorem regression_residual (x y : ℝ) (b m : ℝ) (hx1 : x = 4) (hy1 : y = 1.2)
                            (hb : b = 0.31) (hm : m = 2.5) :
  let y_hat := m * x + b in
  y - y_hat = -9.11 :=
by
  -- Definitions
  let y_hat := m * x + b
  -- Stating the goal
  have h_goal : y - y_hat = -9.11 := sorry
  exact h_goal

end regression_residual_l76_76934


namespace four_op_two_l76_76005

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l76_76005


namespace coeff_of_x_squared_in_expansion_l76_76399

theorem coeff_of_x_squared_in_expansion :
  let x : ℝ := x;
  let expansion : ℝ := (sqrt x - 2 / x) ^ 7;
  ∃ r : ℕ, (7 - 3 * r) / 2 = 2 ∧ ((-2) ^ r * nat.choose 7 r) = -14 :=
by
  sorry

end coeff_of_x_squared_in_expansion_l76_76399


namespace sector_arc_length_l76_76289

theorem sector_arc_length (r : ℝ) (θ : ℝ) (L : ℝ) (h₁ : r = 1) (h₂ : θ = 60 * π / 180) : L = π / 3 :=
by
  sorry

end sector_arc_length_l76_76289


namespace max_discount_rate_l76_76197

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76197


namespace timesToFillBottlePerWeek_l76_76022

noncomputable def waterConsumptionPerDay : ℕ := 4 * 5
noncomputable def waterConsumptionPerWeek : ℕ := 7 * waterConsumptionPerDay
noncomputable def bottleCapacity : ℕ := 35

theorem timesToFillBottlePerWeek : 
  waterConsumptionPerWeek / bottleCapacity = 4 := 
by
  sorry

end timesToFillBottlePerWeek_l76_76022


namespace zachary_needs_more_money_l76_76270

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_shoes : ℝ := 11.85
def zachary_money : ℝ := 10.00
def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_shoes
def amount_needed : ℝ := total_cost - zachary_money

theorem zachary_needs_more_money : amount_needed = 7.00 := by
  sorry

end zachary_needs_more_money_l76_76270


namespace sum_of_integers_condition_l76_76091

theorem sum_of_integers_condition (π_approx : ℝ) (hπ : π_approx ≈ Real.pi) : 
  let cond := λ (n : ℤ), (abs (↑n - 1) < π_approx)
  in ((Finset.filter cond (Finset.range 8)).sum (λ x, x) = 7) :=
by sorry

end sum_of_integers_condition_l76_76091


namespace maximum_discount_rate_l76_76141

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76141


namespace not_a_random_event_is_A_l76_76225

def Event (description : String) : Type := String

def certain_event (e : Event) : Prop :=
  match e with
  | "The sun rises in the east and it rains in the west" => true
  | _ => false

theorem not_a_random_event_is_A :
  ∃ e : Event, certain_event e ∧ e = "The sun rises in the east and it rains in the west" := by
    exists "The sun rises in the east and it rains in the west"
    constructor
    . simp [certain_event]
    . rfl

end not_a_random_event_is_A_l76_76225


namespace calc_eccentricity_l76_76703

noncomputable def point := (ℝ × ℝ)

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : set point :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)}

noncomputable def foci_distance (a : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - (a^2 * (1 - (a/(2*b))^2)))

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

variables (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variables
  (F1 F2 O A M : point)
  (h3 : F1 = ( - (foci_distance a) / 2, 0))
  (h4 : F2 = ( (foci_distance a) / 2, 0))
  (h5 : O = (0, 0))
  (h6 : A ∈ ellipse a b ⟨h1, h2⟩)
  (h7 : (A.1 - F1.1) * (A.1 - F2.1) + (A.2 - F1.2) * (A.2 - F2.2) = 0)
  (h8 : M = (0, A.2))
  (h9 : dist F1 F2 = 6 * dist O M)
  
theorem calc_eccentricity : eccentricity a (real.sqrt 10 / 4) = real.sqrt 10 / 4 :=
  by sorry

end calc_eccentricity_l76_76703


namespace Bohan_bill_given_to_Ann_l76_76228

def people := {Ann, Bohan, Che, Devi, Eden}
def bills := {Ann's_bill, Bohan's_bill, Che's_bill, Devi's_bill, Eden's_bill}
variables (person bill : Type) [DecidableEq person] [DecidableEq bill]

axiom diff_meals : ∀ (p1 p2 : person), p1 ≠ p2 → ∃ (b1 b2 : bill), b1 ≠ b2
axiom correct_dist : ∀ (p : person), ∃ (b : bill), p ≠ who_got(b)

noncomputable def who_got : bill → person := sorry
noncomputable def given_to : person → bill := sorry

axiom cond1 : who_got(Che's_bill) = who_got(Eden's_bill)
axiom cond2 : who_got(Devi's_bill) = Devi
axiom cond3 : ∀ p, (p ≠ Ann ∧ p ≠ Bohan ∧ p ≠ Eden) → given_to(p) ≠ p
axiom cond4 : given_to(Ann) = Che's_bill

theorem Bohan_bill_given_to_Ann : given_to(Bohan) = Ann's_bill := sorry

end Bohan_bill_given_to_Ann_l76_76228


namespace repeating_decimal_to_fraction_l76_76962

theorem repeating_decimal_to_fraction : ∃ (r : ℚ), r = 0.4 + 0.0036 * (1/(1 - 0.01)) ∧ r = 42 / 55 :=
by
  sorry

end repeating_decimal_to_fraction_l76_76962


namespace teammates_average_points_per_game_l76_76578

-- Defining the conditions as Lean definitions
def wade_average_points_per_game : ℕ := 20
def total_points_after_5_games : ℕ := 300

-- Stating the problem: the average points per game of the teammates
theorem teammates_average_points_per_game (wade_average : ℕ) (team_total : ℕ) : Prop :=
  let total_wade_points := 5 * wade_average in
  let total_teammates_points := team_total - total_wade_points in
  let teammates_average := total_teammates_points / 5 in
  teammates_average = 40

-- Substituting the given conditions
lemma example : teammates_average_points_per_game wade_average_points_per_game total_points_after_5_games :=
by 
  let total_wade_points := 5 * wade_average_points_per_game
  let total_teammates_points := total_points_after_5_games - total_wade_points
  let teammates_average := total_teammates_points / 5
  show teammates_average = 40, from sorry

end teammates_average_points_per_game_l76_76578


namespace find_scalar_d_l76_76072

variables (v : ℝ^3)
variables (i j k : ℝ^3)
variables [IsOrthoBasis (i, j, k)]

noncomputable def scalar_d : ℝ := 1

theorem find_scalar_d (v : ℝ^3) (i j k : ℝ^3) [h : IsOrthoBasis (i, j, k)] :
  i.dot (dotProd v i) + j.dot (dotProd v j) + k.dot (dotProd v k) = scalar_d v * (norm v)^2 := by
  sorry

end find_scalar_d_l76_76072


namespace maximum_discount_rate_l76_76136

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76136


namespace max_discount_rate_l76_76183

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76183


namespace max_discount_rate_l76_76128

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76128


namespace min_soldiers_needed_l76_76774

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76774


namespace b_not_necessary_nor_sufficient_log_a_b_gt_1_l76_76302

theorem b_not_necessary_nor_sufficient_log_a_b_gt_1 (a b : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ¬ ((∀ b > a, ∃ b, log a b > 1 → b > a) ∧ (∃ b > a, log a b > 1)) :=
by
  sorry

end b_not_necessary_nor_sufficient_log_a_b_gt_1_l76_76302


namespace max_discount_rate_l76_76195

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76195


namespace friends_total_sales_l76_76594

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end friends_total_sales_l76_76594


namespace cos_squared_identity_l76_76448

theorem cos_squared_identity (α β ω : ℝ) (h : α + β = ω) :
  cos α * cos α + cos β * cos β - 2 * cos α * cos β * cos ω = sin ω * sin ω := 
by
  sorry

end cos_squared_identity_l76_76448


namespace solve_inequality_l76_76043

theorem solve_inequality (x : ℝ) : 
  (x - 3) / (x - 1)^2 < 0 ↔ x ∈ Set.Ioo (⊥ : ℝ) 1 ∪ Set.Ioo 1 3 := by
sor

end solve_inequality_l76_76043


namespace original_number_exists_l76_76378

theorem original_number_exists (x : ℤ) (h1 : x * 16 = 3408) (h2 : 0.016 * 2.13 = 0.03408) : x = 213 := 
by 
  sorry

end original_number_exists_l76_76378


namespace max_discount_rate_l76_76182

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76182


namespace min_soldiers_needed_l76_76769

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76769


namespace max_discount_rate_l76_76180

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76180


namespace total_cages_used_l76_76206

def num_puppies : Nat := 45
def num_adult_dogs : Nat := 30
def num_kittens : Nat := 25

def puppies_sold : Nat := 39
def adult_dogs_sold : Nat := 15
def kittens_sold : Nat := 10

def cage_capacity_puppies : Nat := 3
def cage_capacity_adult_dogs : Nat := 2
def cage_capacity_kittens : Nat := 2

def remaining_puppies : Nat := num_puppies - puppies_sold
def remaining_adult_dogs : Nat := num_adult_dogs - adult_dogs_sold
def remaining_kittens : Nat := num_kittens - kittens_sold

def cages_for_puppies : Nat := (remaining_puppies + cage_capacity_puppies - 1) / cage_capacity_puppies
def cages_for_adult_dogs : Nat := (remaining_adult_dogs + cage_capacity_adult_dogs - 1) / cage_capacity_adult_dogs
def cages_for_kittens : Nat := (remaining_kittens + cage_capacity_kittens - 1) / cage_capacity_kittens

def total_cages : Nat := cages_for_puppies + cages_for_adult_dogs + cages_for_kittens

-- Theorem stating the final goal
theorem total_cages_used : total_cages = 18 := by
  sorry

end total_cages_used_l76_76206


namespace odd_function_f_l76_76057

noncomputable def f (x : ℝ) := 
  if x > 0 then 2 * x^2 - x + 3 else function.sorry

theorem odd_function_f {x : ℝ} : (∀ x : ℝ, f (-x) = -f x) → ∀ x < 0, f x = -2 * x^2 - x - 3 :=
by
  intro h_odd x_neg
  sorry

end odd_function_f_l76_76057


namespace younger_son_age_after_30_years_l76_76916

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l76_76916


namespace quadratic_has_solutions_l76_76079

theorem quadratic_has_solutions :
  (1 + Real.sqrt 2)^2 - 2 * (1 + Real.sqrt 2) - 1 = 0 ∧ 
  (1 - Real.sqrt 2)^2 - 2 * (1 - Real.sqrt 2) - 1 = 0 :=
by
  sorry

end quadratic_has_solutions_l76_76079


namespace factorization_ce_sum_eq_25_l76_76479

theorem factorization_ce_sum_eq_25 {C E : ℤ} (h : (C * x - 13) * (E * x - 7) = 20 * x^2 - 87 * x + 91) : 
  C * E + C = 25 :=
sorry

end factorization_ce_sum_eq_25_l76_76479


namespace monochromatic_arithmetic_progression_l76_76491

open Finset

theorem monochromatic_arithmetic_progression :
  ∀ (S : Finset ℕ), S = (finset.range 10).filter (λ x, x > 0) → 
  (∀ f : ℕ → Prop, (∀ x ∈ S, f x → f x = or.bnot (f x)) →
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ (b - a = c - b) ∧ f a = f b ∧ f b = f c). 
sorry

end monochromatic_arithmetic_progression_l76_76491


namespace triangle_stick_sum_l76_76626

theorem triangle_stick_sum (m : ℕ) 
  (h1 : 7 + 11 > m) 
  (h2 : 7 + m > 11) 
  (h3 : 11 + m > 7) :
  ∃ (m_values : Set ℕ), (∀ m ∈ m_values, 4 < m ∧ m < 18) ∧ m_values.sum = 132 :=
by
  sorry

end triangle_stick_sum_l76_76626


namespace find_b_l76_76637

def oscillation_period (a b c d : ℝ) (oscillations : ℝ) : Prop :=
  oscillations = 5 * (2 * Real.pi) / b

theorem find_b
  (a b c d : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0)
  (osc_complexity: oscillation_period a b c d 5):
  b = 5 := by
  sorry

end find_b_l76_76637


namespace opposite_of_neg_two_l76_76555

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76555


namespace soldiers_to_add_l76_76765

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76765


namespace probability_both_in_section_between_15_16_minutes_l76_76827

def alice_path (t : ℝ) : ℝ := (t / 120) - ⌊t / 120⌋
def bob_path (t : ℝ) : ℝ := 1 - (t / 75) + ⌊t / 75⌋

def is_in_section (position : ℝ) : Prop := 0.1 ≤ position ∧ position ≤ 0.4

def both_in_section (t : ℝ) : Prop := is_in_section (alice_path t) ∧ is_in_section (bob_path t)

theorem probability_both_in_section_between_15_16_minutes : 
  (∫ t in 900..960, if both_in_section t then 1 else 0) / 60 = 11 / 60 := 
sorry

end probability_both_in_section_between_15_16_minutes_l76_76827


namespace five_letter_arrangements_fixed_A_and_one_C_l76_76330

theorem five_letter_arrangements_fixed_A_and_one_C :
  ∃ n : ℕ, n = 240 ∧
  (∀ (l : List Char), l.length = 5 →
    l.head = 'A' →
    'C' ∈ l.tail →
    l.nodup →
    l.toFinset ⊆ {'A', 'B', 'C', 'D', 'E', 'F', 'G'}) :=
by
  use 240
  split
  { exact rfl } -- The correct number of arrangements is 240
  sorry

end five_letter_arrangements_fixed_A_and_one_C_l76_76330


namespace Jackson_money_is_125_l76_76852

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l76_76852


namespace opposite_of_neg_two_l76_76515

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76515


namespace maximum_discount_rate_l76_76137

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76137


namespace rotated_line_eq_l76_76454

theorem rotated_line_eq :
  ∀ (x y : ℝ), (x - y + sqrt 3 - 1 = 0) ∧ 
                (let θ := π/12 in
                 let T := λ (p : ℝ × ℝ), (cos θ * (p.1 - 1) - sin θ * (p.2 - sqrt 3) + 1, 
                                            sin θ * (p.1 - 1) + cos θ * (p.2 - sqrt 3) + sqrt 3) in
                 ∀ p ∈ {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + sqrt 3 - 1 = 0},
                    let p' := T p in 
                    (p'.2 = sqrt 3 * p'.1)) :=
by
  sorry

end rotated_line_eq_l76_76454


namespace find_x_l76_76609

-- Define the conditions as hypotheses
def problem_statement (x : ℤ) : Prop :=
  (3 * x > 30) ∧ (x ≥ 10) ∧ (x > 5) ∧ 
  (x = 9)

-- Define the theorem statement
theorem find_x : ∃ x : ℤ, problem_statement x :=
by
  -- Sorry to skip proof as instructed
  sorry

end find_x_l76_76609


namespace negation_proof_l76_76928

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 2 * x)) ↔ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proof_l76_76928


namespace count_zeros_f_on_interval_0_to_6_l76_76719

-- Definition of the function f with the given conditions
def f : ℝ → ℝ :=
  λ x, if h : x = 3 / 2 then 0 
       else if x < 3 / 2 then sin (π * x) 
       else -sin (π * (x - 3))

-- The main theorem stating the number of zeros of f on the interval [0, 6]
theorem count_zeros_f_on_interval_0_to_6 : 
  number_of_zeros_of_function_on_interval f 0 6 = 9 := sorry

end count_zeros_f_on_interval_0_to_6_l76_76719


namespace angleAPB_is_right_angle_l76_76839

noncomputable def angleAPB : Real := 90

theorem angleAPB_is_right_angle
  (PA_tangent_SAR : True) -- Simplified by True for tangency condition
  (PB_tangent_RBT : True) -- Simplified by True for tangency condition
  (SRT_straight : True) -- Simplified by True for straight line condition
  (arc_AS_40 : ∀ (O1 A S : Point), angle O1 A S = 40)
  (arc_BT_50 : ∀ (O2 B T : Point), angle O2 B T = 50) :
  angle APB = 90 :=
sorry

end angleAPB_is_right_angle_l76_76839


namespace perp_GA_DC_l76_76842

theorem perp_GA_DC (A B C D E F G : Point)
  (h1 : triangle_right A B C)
  (h2 : square_on_side ABDE)
  (h3 : square_on_side BCFG) :
  is_perpendicular G A D C :=
  sorry

end perp_GA_DC_l76_76842


namespace abs_sqrt3_minus_2_pm_sqrt_25_over_9_cube_root_neg_8_over_27_l76_76997

theorem abs_sqrt3_minus_2 : abs (sqrt 3 - 2) = 2 - sqrt 3 := 
by
  sorry

theorem pm_sqrt_25_over_9 : 
  (sqrt (25 / 9) = 5 / 3) ∨ (sqrt (25 / 9) = - (5 / 3)) := 
by
  sorry

theorem cube_root_neg_8_over_27 : cbrt (-8 / 27) = -(2 / 3) := 
by
  sorry

end abs_sqrt3_minus_2_pm_sqrt_25_over_9_cube_root_neg_8_over_27_l76_76997


namespace solution_set_of_inequality_l76_76670

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l76_76670


namespace minimum_soldiers_to_add_l76_76793

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76793


namespace opposite_of_neg_two_l76_76531

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76531


namespace fencing_cost_l76_76925

theorem fencing_cost (w l : ℕ) (rate : ℕ) (P : ℕ) (cost : ℕ)
  (h1 : l = w + 10)
  (h2 : P = 2 * (w + l))
  (h3 : P = 180)
  (h4 : rate = 6.5)
  (h5 : cost = P * rate) :
  cost = 1170 := by
sorry

end fencing_cost_l76_76925


namespace number_of_special_integers_l76_76738

theorem number_of_special_integers : 
  ∃ S : finset ℕ, S.card = 5 ∧ ∀ N ∈ S, 
    (N < 2000) ∧ (∃ (M : finset ℕ), M.card = 6 ∧ ∀ j ∈ M, ∃ n, N = j * (2 * n + j)) :=
sorry

end number_of_special_integers_l76_76738


namespace younger_son_age_in_30_years_l76_76912

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l76_76912


namespace opposite_of_neg_two_is_two_l76_76540

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76540


namespace sum_of_corners_is_164_l76_76439

section CheckerboardSum

-- Define the total number of elements in the 9x9 grid
def num_elements := 81

-- Define the positions of the corners
def top_left : ℕ := 1
def top_right : ℕ := 9
def bottom_left : ℕ := 73
def bottom_right : ℕ := 81

-- Define the sum of the corners
def corner_sum : ℕ := top_left + top_right + bottom_left + bottom_right

-- State the theorem
theorem sum_of_corners_is_164 : corner_sum = 164 :=
by
  exact sorry

end CheckerboardSum

end sum_of_corners_is_164_l76_76439


namespace interval_for_a_l76_76015

namespace Proof

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem interval_for_a (a : ℝ) (h : a ≠ 0) :
  Set.Icc (-3 / 2) 6 ⊆ {x : ℝ | a * x^2 + 3 * x - 2 * a = 0} →
  {a | a ∈ Set.Ioi (-∞) ∪ Set.Ioi (-9 / 17) ∪ Set.Ioi (42 / 41) ∪ Set.Ioi (∞)} :=
begin
  sorry
end

end Proof

end interval_for_a_l76_76015


namespace problem1_problem2_l76_76711

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l76_76711


namespace chengdu_chongqing_scientific_notation_l76_76385

theorem chengdu_chongqing_scientific_notation:
  (185000 : ℝ) = 1.85 * 10^5 :=
sorry

end chengdu_chongqing_scientific_notation_l76_76385


namespace opposite_of_neg2_l76_76503

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76503


namespace max_wickets_per_over_l76_76122

variable (max_wickets_in_6_overs : ℕ) (overs_in_innings : ℕ) (balls_per_over : ℕ)

theorem max_wickets_per_over :
  max_wickets_in_6_overs = 10 → overs_in_innings = 6 → balls_per_over = 6 → ∃ max_wickets_in_1_over : ℕ, max_wickets_in_1_over = 6 :=
by
  intros h1 h2 h3
  use 6
  sorry

end max_wickets_per_over_l76_76122


namespace determine_x_l76_76654

noncomputable def f (x : ℝ) : ℝ := (↑ ((x + 5) / 5)) ^ (1 / 4)

theorem determine_x (x : ℝ) (h₀ : f(3 * x) = 3 * f(x)) : x = -200 / 39 :=
  sorry

end determine_x_l76_76654


namespace equal_saturdays_and_sundays_l76_76205

theorem equal_saturdays_and_sundays: 
  ∃ days: Finset Nat, days.card = 2 ∧ 
  ∀ d ∈ days, let s := (List.range 31).map (λ i, (d + i) % 7) in 
  s.countp (λ x, x = 5) = s.countp (λ x, x = 6) :=
by {
  sorry
}

end equal_saturdays_and_sundays_l76_76205


namespace a_parallel_b_implies_t_half_a_orthogonal_b_implies_t_three_or_neg_one_magnitude_difference_at_t_one_angle_acute_implies_range_t_l76_76328

-- Definitions of the given vectors
def a (t : ℝ) : ℝ × ℝ := (2 - t, 3)
def b (t : ℝ) : ℝ × ℝ := (t, 1)

-- Define orthogonality (dot product is zero)
def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Define parallelism (components are proportional)
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statements for Lean Proof
theorem a_parallel_b_implies_t_half : ∀ (t : ℝ), parallel (a t) (b t) → t = 1 / 2 :=
by
  sorry

theorem a_orthogonal_b_implies_t_three_or_neg_one : ∀ (t : ℝ), orthogonal (a t) (b t) → t = 3 ∨ t = -1 :=
by
  sorry

theorem magnitude_difference_at_t_one : ∀ (t : ℝ), t = 1 → (real.sqrt ((a t).1 - 4* (b t).1)^2 + ((a t).2 - 4* (b t).2)^2) = real.sqrt 10 :=
by
  sorry

theorem angle_acute_implies_range_t : ∀ (t : ℝ), (2 * t - t ^ 2 + 3 > 0) → (-1 < t ∧ t < 3) → t ≠ 1 / 2 :=
by
  sorry

end a_parallel_b_implies_t_half_a_orthogonal_b_implies_t_three_or_neg_one_magnitude_difference_at_t_one_angle_acute_implies_range_t_l76_76328


namespace cone_surface_area_ratio_l76_76563

-- Definitions
def base_area (r : ℝ) : ℝ := π * r^2
def lateral_area (r l : ℝ) : ℝ := (1 / 2) * π * l^2

-- Conditions
variables (r l : ℝ) (h : 2 * π * r = (1 / 2) * π * l)

-- Correct ratio of surface area to lateral area
theorem cone_surface_area_ratio (h : l = 4 * r) :
  (base_area r + lateral_area r l) / lateral_area r l = 5 / 4 := by
  sorry

end cone_surface_area_ratio_l76_76563


namespace ada_original_seat_l76_76266

theorem ada_original_seat {a b c d e : ℕ} : 
  (a ∈ {1, 5}) → 
  b = a + 2 → 
  c = a - 1 → 
  d ≠ e → 
  (a, 5) ∈ (1, 5) → 
  a = 2 :=
by
  -- Placeholder proof.
  sorry

end ada_original_seat_l76_76266


namespace max_discount_rate_l76_76145

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76145


namespace swim_upstream_distance_l76_76627

-- Defining the parameters
def t_d := 8 -- time taken downstream in hours
def d_d := 64 -- distance downstream in km
def c := 2.5 -- speed of the current in km/h

-- Statement to be proved
theorem swim_upstream_distance :
  ∃ (d v : ℝ), (v + c) * t_d = d_d ∧ (v - c) * t_d = d ∧ d = 24 :=
by
  sorry

end swim_upstream_distance_l76_76627


namespace max_discount_rate_l76_76146

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76146


namespace equal_share_candy_l76_76752

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l76_76752


namespace problem1_problem2_l76_76232

-- Problem 1: Prove the expression evaluates to 8
theorem problem1 : (1:ℝ) * (- (1 / 2)⁻¹) + (3 - Real.pi)^0 + (-3)^2 = 8 := 
by
  sorry

-- Problem 2: Prove the expression simplifies to 9a^6 - 2a^2
theorem problem2 (a : ℝ) : a^2 * a^4 - (-2 * a^2)^3 - 3 * a^2 + a^2 = 9 * a^6 - 2 * a^2 := 
by
  sorry

end problem1_problem2_l76_76232


namespace fill_latin_square_l76_76607

-- Define a 4x4 matrix with the given initial conditions
def initial_matrix : Matrix (Fin 4) (Fin 4) ℕ :=
  ![
    ![1, 2, 3, 4],
    ![2, 0, 0, 0],
    ![3, 0, 0, 0],
    ![4, 0, 0, 0]
  ]

-- Function to check if a matrix is a Latin square
def is_latin_square {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∀ i : Fin n, (Finset.univ.image (λ j, A i j)).card = n ∧
               ∀ j : Fin n, (Finset.univ.image (λ i, A i j)).card = n

-- Prove that we can fill in the remaining cells to form a valid Latin square
theorem fill_latin_square : ∃ M : Matrix (Fin 4) (Fin 4) ℕ, 
  (∀ i : Fin 4, ∀ j : Fin 4, (i = ⟨0, _⟩ → M i j = initial_matrix i j) ∧ 
                              (j = ⟨0, _⟩ → M i j = initial_matrix i j)) ∧
  is_latin_square M :=
by {
  -- The proof steps would be implemented here.
  sorry
}

end fill_latin_square_l76_76607


namespace opposite_of_neg_two_is_two_l76_76520

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76520


namespace distribution_plans_l76_76663

theorem distribution_plans :
  let classes := range 3,
    teachers := range 5 in
  (∃ (class_assignments : teachers → classes → Prop),
    (∀ t, ∃ c, class_assignments t c) ∧ -- every teacher is assigned to a class
    (∀ c, 1 ≤ (Finset.card (Finset.filter (class_assignments · c) teachers)) ∧ -- each class has at least 1 teacher
        (Finset.card (Finset.filter (class_assignments · c) teachers) ≤ 2))) →
  ((Finset.card (Finset.filter (λ c, Finset.card (Finset.filter (class_assignments · c) teachers) = 2) classes) = 1) ∧ -- one class has exactly 2 teachers
   ∀ c, (1 ≤ Finset.card (Finset.filter (class_assignments · c) teachers) ∧
        Finset.card (Finset.filter (class_assignments · c) teachers) ≤ 2)) →
  ∃ N, N = 30 := 
sorry

end distribution_plans_l76_76663


namespace max_additional_license_plates_l76_76677

theorem max_additional_license_plates :
  let initial_count := 6 * 4 * 5 in
  let new_count := max (6 * 7 * 5) (max (6 * 6 * 6) (7 * 5 * 6)) in
  new_count - initial_count = 96 := 
by {
  let initial_count := 6 * 4 * 5;
  let new_count := max (6 * 7 * 5) (max (6 * 6 * 6) (7 * 5 * 6));
  calc
    new_count - initial_count
        = max (6 * 7 * 5) (max (6 * 6 * 6) (7 * 5 * 6)) - 6 * 4 * 5 : by rfl
    ... = 216 - 120 : by sorry
    ... = 96 : by rfl
}

end max_additional_license_plates_l76_76677


namespace maximum_discount_rate_l76_76171

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76171


namespace pyramid_volume_l76_76694

noncomputable def volume_pyramid (a b : ℝ) : ℝ :=
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2))

theorem pyramid_volume (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 < 4 * b^2) :
  volume_pyramid a b =
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2)) :=
sorry

end pyramid_volume_l76_76694


namespace opposite_of_neg_two_l76_76527

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76527


namespace pressures_determined_l76_76604

noncomputable def gas_flow_proof_problem
  (V1 V2 P1 P2 : ℝ)
  (a b : ℝ) (C1 C2 k : ℝ)
  (p1 p2 : ℝ → ℝ)
  (t : ℝ) : Prop :=
∃ (p1 p2 : ℝ → ℝ),
  (∀ t, V1 * p1 t + V2 * p2 t = C1) ∧
  (∀ t, a * (p1 t ^ 2 - p2 t ^ 2) = b * V2 * (p2 t).derivative) ∧
  (∀ t, a * (p1 t ^ 2 - p2 t ^ 2) = -b * V1 * (p1 t).derivative) ∧
  (∃ C2, ∀ t, (p1(t) + p2(t)) / (p1(t) - p2(t)) = C2 * exp(2 * k * t)) ∧
  (p1(0) = P1) ∧ (p2(0) = P2)

-- statement to show that p1 and p2 are determined by initial conditions
theorem pressures_determined (V1 V2 P1 P2 : ℝ) (a b : ℝ) (k : ℝ):
  ∀ t, ∃ p1 p2 : ℝ → ℝ, gas_flow_proof_problem V1 V2 P1 P2 a b (V1 * P1 + V2 * P2) (P1 + P2) k p1 p2 t := 
sorry

end pressures_determined_l76_76604


namespace triangle_ratio_l76_76446

theorem triangle_ratio 
  (A B C D E : Type) 
  [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E]
  (triangle : A → B → C → Type) 
  (midpoint : A → C → D → Prop)
  (angle_eq : B → E → A → C → E → D → Prop)
  (A E_eq : E → A) (D_eq : E → D) : 
  midpoint A C D →
  angle_eq B E A C E D →
  exists k : ℝ, k = (AE / ED) ∧ k = 2 :=
begin
  assume h1 : midpoint A C D,
  assume h2 : angle_eq B E A C E D,
  sorry
end

end triangle_ratio_l76_76446


namespace five_digit_even_numbers_l76_76959

theorem five_digit_even_numbers : 
  let digits := {0, 1, 2, 3, 4, 5}
  ∃ (s : Finset ℕ) (h : ∀ x ∈ s, x ≥ 20000 ∧ x % 2 = 0 ∧ (∀ y ∈ s, y ≠ x → x ≠ y)),
    s.card = 240 :=
by
  let digits := {0, 1, 2, 3, 4, 5}
  sorry

end five_digit_even_numbers_l76_76959


namespace arithmetic_sequence_term_count_l76_76342

theorem arithmetic_sequence_term_count :
  ∀ (a1 d an : ℕ), a1 = 20 → d = 5 → an = 140 → ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 25 :=
by
  intros a1 d an h1 h2 h3
  use 25
  split
  calc
    an = 20 + (25 - 1) * 5 : by simp [h1, h2]
    ... = 140             : by simp [h3]
  simp only [Nat.succ_pred_eq_of_pos, ge_iff_le, Nat.succ_pos', tsub_le_iff_right]
  done

end arithmetic_sequence_term_count_l76_76342


namespace hyperbola_eccentricity_squared_l76_76305

/-- Given that F is the right focus of the hyperbola 
    \( C: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 \) with \( a > 0 \) and \( b > 0 \), 
    a line perpendicular to the x-axis is drawn through point F, 
    intersecting one asymptote of the hyperbola at point M. 
    If \( |FM| = 2a \), denote the eccentricity of the hyperbola as \( e \). 
    Prove that \( e^2 = \frac{1 + \sqrt{17}}{2} \).
 -/
theorem hyperbola_eccentricity_squared (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: c^2 = a^2 + b^2) (h4: b * c = 2 * a^2) : 
  (c / a)^2 = (1 + Real.sqrt 17) / 2 := 
sorry

end hyperbola_eccentricity_squared_l76_76305


namespace count_multiples_4_6_10_less_300_l76_76747

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l76_76747


namespace grass_sheet_cost_per_cubic_meter_l76_76908

variable (area depth total_cost : ℝ)

def volume (area depth : ℝ) : ℝ := area * depth
def cost_per_cubic_meter (total_cost volume : ℝ) : ℝ := total_cost / volume

theorem grass_sheet_cost_per_cubic_meter
  (h_area : area = 5900)
  (h_depth : depth = 0.01)
  (h_total_cost : total_cost = 165.2)
  : cost_per_cubic_meter total_cost (volume area depth) = 2.8 := by
  sorry

end grass_sheet_cost_per_cubic_meter_l76_76908


namespace one_plus_i_squared_eq_two_i_l76_76281

theorem one_plus_i_squared_eq_two_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end one_plus_i_squared_eq_two_i_l76_76281


namespace solve_equation_1_solve_equation_2_l76_76461

open Real

theorem solve_equation_1 (x : ℝ) (h_ne1 : x + 1 ≠ 0) (h_ne2 : x - 3 ≠ 0) : 
  (5 / (x + 1) = 1 / (x - 3)) → x = 4 :=
by
    intro h
    sorry

theorem solve_equation_2 (x : ℝ) (h_ne1 : x - 4 ≠ 0) (h_ne2 : 4 - x ≠ 0) :
    (3 - x) / (x - 4) = 1 / (4 - x) - 2 → False :=
by
    intro h
    sorry

end solve_equation_1_solve_equation_2_l76_76461


namespace minimum_soldiers_to_add_l76_76781

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76781


namespace integral_of_exponential_absolute_value_l76_76665

theorem integral_of_exponential_absolute_value :
  ∫ x in -2..4, Real.exp (|x|) = Real.exp 4 + Real.exp 2 - 2 :=
by
  sorry

end integral_of_exponential_absolute_value_l76_76665


namespace number_of_solutions_l76_76660

theorem number_of_solutions :
  { x : ℝ | |x + 1| = |x - 2| + |x - 5| + |x - 6| }.to_finset.card = 2 :=
by
  sorry

end number_of_solutions_l76_76660


namespace trees_to_plant_l76_76105

def road_length : ℕ := 156
def interval : ℕ := 6
def trees_needed (road_length interval : ℕ) := road_length / interval + 1

theorem trees_to_plant : trees_needed road_length interval = 27 := by
  sorry

end trees_to_plant_l76_76105


namespace candle_problem_l76_76954

theorem candle_problem :
  ∃ x : ℚ,
    (1 - x / 6 = 3 * (1 - x / 5)) ∧
    x = 60 / 13 :=
by
  -- let initial_height_first_candle be 1
  -- let rate_first_burns be 1 / 6
  -- let initial_height_second_candle be 1
  -- let rate_second_burns be 1 / 5
  -- We want to prove:
  -- 1 - x / 6 = 3 * (1 - x / 5) ∧ x = 60 / 13
  sorry

end candle_problem_l76_76954


namespace max_value_sqrt_sum_l76_76867

open Real

noncomputable def max_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) : ℝ :=
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) :
  max_sqrt_sum x y z h1 h2 h3 h_sum ≤ 3 * sqrt 8 :=
sorry

end max_value_sqrt_sum_l76_76867


namespace count_non_representable_as_diff_of_squares_l76_76331

theorem count_non_representable_as_diff_of_squares :
  let count := (Finset.filter (fun n => ∃ k, n = 4 * k + 2 ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card in
  count = 250 :=
by
  sorry

end count_non_representable_as_diff_of_squares_l76_76331


namespace minimum_soldiers_to_add_l76_76800

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76800


namespace jack_marathon_time_l76_76848

theorem jack_marathon_time :
  ∀ (D S_Jill S_Jack : ℝ), 
  D = 42 ∧ S_Jill = 10 ∧ S_Jack = 0.7 * S_Jill → 
  (D / S_Jack = 6) := 
by 
  intro D S_Jill S_Jack
  intro h
  cases h with h_D h
  cases h with h_SJill h_SJack
  have h_speedJack : S_Jack = 7 := by rw [h_SJill, h_SJack]; norm_num
  rw [h_D, h_speedJack]
  norm_num
  sorry

end jack_marathon_time_l76_76848


namespace opposite_of_neg_two_l76_76556

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76556


namespace count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76739

-- The problem states the following conditions
def is_divisible_by (m n : ℕ) : Prop := m % n = 0
def less_than_300 (n : ℕ) : Prop := n < 300

-- We want to prove the number of positive integers less than 300 that are divisible by 4, 6, and 10
theorem count_positive_integers_divisible_by_4_6_10_less_than_300 :
  (Finset.card (Finset.filter 
    (λ n, is_divisible_by n 4 ∧ is_divisible_by n 6 ∧ is_divisible_by n 10 ∧ less_than_300 n)
    ((Finset.range 300).filter (λ n, n ≠ 0)))) = 4 :=
by
  sorry

end count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76739


namespace cell_division_result_l76_76284

theorem cell_division_result (divisions : ℕ) (initial_cells : ℕ) : 
  (initial_cells = 1) → (divisions = 3) → ((2 ^ divisions) = 8) :=
by
  intros h_initial h_divisions
  rw [h_initial, h_divisions]
  rfl

end cell_division_result_l76_76284


namespace cos_double_angle_given_tan_sum_l76_76115

theorem cos_double_angle_given_tan_sum (α : ℝ) (h : Float.tan (α + Float.pi / 4) = 2) : Float.cos (2 * α) = 4 / 5 := 
sorry

end cos_double_angle_given_tan_sum_l76_76115


namespace range_of_t_condition_l76_76481

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a ≤ b → f a ≤ f b
def range_of_t (f : ℝ → ℝ) : Set ℝ := {t | ∀ x ∈ Icc (-1:ℝ) (1:ℝ), ∀ a ∈ Icc (-1:ℝ) (1:ℝ), f x ≤ t^2 + 2*a*t + 1}

theorem range_of_t_condition (f : ℝ → ℝ) 
  (hf1 : odd f)
  (hf2 : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → monotonically_increasing f x y)
  (hf3 : f (-1) = -1) : 
  range_of_t f = {t | t ∈ Iio (-2) ∪ ({0} : Set ℝ) ∪ Ioi 2} := 
by 
  sorry

end range_of_t_condition_l76_76481


namespace opposite_of_neg_two_l76_76541

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76541


namespace sum_of_reciprocals_of_roots_l76_76258

theorem sum_of_reciprocals_of_roots:
  let r1 r2 : ℝ := roots_of_quadratic 1 (-13) 4 in
  (r1 + r2 = 13 ∧ r1 * r2 = 4) → (1 / r1 + 1 / r2 = 13 / 4) :=
by
  intros r1 r2 h
  cases h with h_sum h_prod
  have : 1 / r1 + 1 / r2 = (r1 + r2) / (r1 * r2) :=
    by rw [←add_div, div_self (mul_ne_zero (ne_of_eq_of_ne h_sum) (ne_of_eq_of_ne h_prod))]
  rw [h_sum, h_prod, this]
  done

end sum_of_reciprocals_of_roots_l76_76258


namespace solve_problem_l76_76678

-- Define the concept of a descent in a permutation
def has_exactly_one_descent (l : List ℕ) : Prop :=
  ∃! i, (1 ≤ i ∧ i < l.length ∧ l.nth i > l.nth (i + 1))

-- Define the number of permutations of {1, 2, ..., n}
def number_of_permutations_with_one_descent (n : ℕ) : ℕ :=
  let total_perms := 2^n
  let invalid_perms := n + 1
  total_perms - invalid_perms

-- Define the specific condition for n = 2007
def problem_statement : Prop :=
  number_of_permutations_with_one_descent 2007 = 2^3 * (2^2004 - 251)

theorem solve_problem : problem_statement :=
  sorry

end solve_problem_l76_76678


namespace max_discount_rate_l76_76184

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76184


namespace problem_l76_76296

variable (x : ℝ)

def p := ∃ (x_0 : ℝ), x_0 - 2 > 0
def q := ∀ (x : ℝ), 2^x > x^2

theorem problem : p ∧ ¬q :=
by
  have hp : p := by
    use 3
    exact by norm_num
  have ¬hq : ¬q := by
    intro h
    have hc := h 2
    norm_num at hc
  exact ⟨hp, ¬hq⟩

end problem_l76_76296


namespace ratio_of_areas_l76_76707

noncomputable def rectangle_ABCD : set (ℝ × ℝ) := {(0, 0), (0, 1), (1, 1), (1, 0)}

def is_midpoint (p1 p2 p3 : ℝ × ℝ) : Prop := p3 = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def belongs_to_segment (p1 p2 p : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * p1.1 + (1 - t) * p2.1, t * p1.2 + (1 - t) * p2.2)

def DF_ratio_DA (d f : ℝ × ℝ) (ratio : ℝ) : Prop := (f.1 - d.1) ^ 2 + (f.2 - d.2) ^ 2 = (ratio * (((1 : ℝ) - d.1) ^ 2 + (0 - d.2) ^ 2))

theorem ratio_of_areas (A B C D E F : ℝ × ℝ) :
  A = (0, 0) → B = (0, 1) → C = (1, 1) → D = (1, 0) →
  is_midpoint B D E → belongs_to_segment D A F ∧ DF_ratio_DA D F (1 / 4) →
  (area_of_triangle D F E / area_of_quadrilateral A B E F = 1 / 7) :=
begin
  sorry
end

end ratio_of_areas_l76_76707


namespace speed_ratio_l76_76434

def distance_lou := 3 -- Lou runs 3 miles
def laps_rosie := 24 -- Rosie runs 24 laps
def track_length := 0.25 -- Each lap is 1/4 mile

def distance_rosie := laps_rosie * track_length -- Distance Rosie runs

theorem speed_ratio 
  (lou_distance : ℝ := distance_lou)
  (rosie_distance : ℝ := distance_rosie)
  (rosie_laps : ℝ := laps_rosie)
  (track_len : ℝ := track_length)
  (rosie_total_distance : ℝ := rosie_laps * track_len) :
  rosie_total_distance / lou_distance = 2 :=
by 
  sorry

end speed_ratio_l76_76434


namespace correct_inequality_relation_l76_76723

theorem correct_inequality_relation :
  ¬(∀ (a b c : ℝ), a > b ↔ a * (c^2) > b * (c^2)) ∧
  ¬(∀ (a b : ℝ), a > b → (1/a) < (1/b)) ∧
  ¬(∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∀ (a b c : ℝ), a > b ∧ b > 1 ∧ c < 0 → a^c < b^c) := sorry

end correct_inequality_relation_l76_76723


namespace geometric_sequence_properties_l76_76714

variables {a1 a2 a3 a4 : ℝ}
variable (q : ℝ)
variables (h1 : a1 > 1) 
variables (h2 : a2 = a1 * q) 
variable (h3 : a3 = a2 * q) 
variable (h4 : a4 = a3 * q)
variable (h5 : a1 + a2 + a3 + a4 = log (a1 + a2 + a3))

theorem geometric_sequence_properties (h1 : a1 > 1) 
  (h2 : a2 = a1 * q) 
  (h3 : a3 = a2 * q) 
  (h4 : a4 = a3 * q)
  (h5 : a1 + a2 + a3 + a4 = log (a1 + a2 + a3)) : 
  a1 > a3 ∧ a2 < a4 :=
sorry

end geometric_sequence_properties_l76_76714


namespace minimum_value_of_f_on_interval_l76_76927

def f (x : ℝ) : ℝ := 12 * x - x^3

theorem minimum_value_of_f_on_interval :
  (∃ x ∈ set.Icc (-3 : ℝ) (3 : ℝ), ∀ y ∈ set.Icc (-3 : ℝ) (3 : ℝ), 
  f x ≤ f y) ∧ f (-2) = -16 :=
by
  sorry

end minimum_value_of_f_on_interval_l76_76927


namespace maximum_discount_rate_l76_76139

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76139


namespace all_locations_entered_summer_l76_76059

def meteorological_criterion (temps : List ℕ) : Prop :=
  (temps.length = 5) ∧ (temps.sum / temps.length ≥ 22)

lemma location_A_entered_summer :
  ∃ temps : List ℕ, (temps.length = 5) ∧ (temps.median = 24)
  ∧ (temps.mode = 22) ∧ meteorological_criterion temps :=
sorry

lemma location_B_entered_summer :
  ∃ temps : List ℕ, (temps.length = 5) ∧ (temps.median = 27)
  ∧ (temps.sum / temps.length = 24) ∧ meteorological_criterion temps :=
sorry

lemma location_C_entered_summer :
  ∃ temps : List ℕ, (temps.length = 5) ∧ (32 ∈ temps)
  ∧ (temps.sum / temps.length = 26) ∧ (temps.variance = 10.8)
  ∧ meteorological_criterion temps :=
sorry

theorem all_locations_entered_summer :
  location_A_entered_summer ∧ location_B_entered_summer ∧ location_C_entered_summer :=
by
  apply And.intro
  { apply location_A_entered_summer }
  apply And.intro
  { apply location_B_entered_summer }
  { apply location_C_entered_summer }

end all_locations_entered_summer_l76_76059


namespace opposite_of_neg_two_l76_76544

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76544


namespace power_point_relative_to_circle_l76_76032

noncomputable def circle_power (a b R x1 y1 : ℝ) : ℝ :=
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2

theorem power_point_relative_to_circle (a b R x1 y1 : ℝ) :
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2 = circle_power a b R x1 y1 := by
  unfold circle_power
  sorry

end power_point_relative_to_circle_l76_76032


namespace matrix_power_A_2023_l76_76411

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_A_2023 :
  A ^ 2023 = ![
    ![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]
  ] :=
sorry

end matrix_power_A_2023_l76_76411


namespace sin_alpha_cond_l76_76348

theorem sin_alpha_cond (α : ℝ) (h₁ : cos (π + α) = -1 / 2) (h₂ : (3 / 2) * π < α ∧ α < 2 * π) :
  sin α = -sqrt 3 / 2 :=
by sorry

end sin_alpha_cond_l76_76348


namespace area_ratio_of_triangles_l76_76952

theorem area_ratio_of_triangles
  (A B C D : Type)
  (AC BC AB : ℝ)
  (h1 : AC = 8)
  (h2 : BC = 6)
  (h3 : AB = 10)
  (h4 : ∠ ACB = 90)
  (h5 : Point D is on line segment AB)
  (h6 : CD bisects the right angle at C) :
  (let A_a := area_of_ADT(ADC)
   let A_b := area_of_BDT(BCD)
   (A_a / A_b) = (16 / 9)) := 
  sorry

end area_ratio_of_triangles_l76_76952


namespace infinite_seq_partition_infinite_seq_within_interval_l76_76239

open Mathlib

variable (x : ℕ → ℝ) (ε : ℝ)

-- Condition: Infinite sequence of real numbers in [0, 1)
constant seq_bounded : ∀ n, 0 ≤ x n ∧ x n < 1

-- Condition: ε is strictly between 0 and 1/2
constant eps_cond : ε > 0 ∧ ε < 1/2

-- Problem 1: Prove that either [0, 1/2) or [1/2, 1) contains infinitely many elements
theorem infinite_seq_partition :
  (∃∞ n, 0 ≤ x n ∧ x n < 1/2) ∨ (∃∞ n, 1/2 ≤ x n ∧ x n < 1) := sorry

-- Problem 2: Prove that there exists a rational number α ∈ [0, 1] such that infinitely many elements are within [α - ε, α + ε]
theorem infinite_seq_within_interval :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃∞ n, (α.toReal - ε ≤ x n ∧ x n ≤ α.toReal + ε) := sorry

end infinite_seq_partition_infinite_seq_within_interval_l76_76239


namespace perimeter_of_figure_l76_76834

variable (x y : ℝ)
variable (lengths : Set ℝ)
variable (perpendicular_adjacent : Prop)
variable (area : ℝ)

-- Conditions
def condition_1 : Prop := ∀ l ∈ lengths, l = x ∨ l = y
def condition_2 : Prop := perpendicular_adjacent
def condition_3 : Prop := area = 252
def condition_4 : Prop := x = 2 * y

-- Problem statement
theorem perimeter_of_figure
  (h1 : condition_1 x y lengths)
  (h2 : condition_2 perpendicular_adjacent)
  (h3 : condition_3 area)
  (h4 : condition_4 x y) :
  ∃ perimeter : ℝ, perimeter = 96 := by
  sorry

end perimeter_of_figure_l76_76834


namespace find_angle_A_max_area_l76_76383

-- Problem definition and conditions
variables (a b c : ℝ) (A B : ℝ)
variables (cosB : ℝ) (sinB : ℝ)
variables (cosB_eq : cosB = 4 / 5)
variables (b_eq : b = 2)
variables (a_eq : a = 5 / 3)

-- Proving part (I) -- Angle A
theorem find_angle_A (h : a = 5 / 3) (h_b : b = 2) (h_cosB : cosB = 4 / 5) (h_B : B = real.arccos 4 / 5) : 
  A = 30 := sorry

-- Proving part (II) -- Maximum area
theorem max_area (h : a = 5 / 3) (h_b : b = 2) (h_cosB : cosB = 4 / 5) (h_A : A = 30) :
  ∃ (ac : ℝ), (1 / 2) * ac * sin (real.arccos 4 / 5) = 3 := sorry

end find_angle_A_max_area_l76_76383


namespace opposite_of_neg_two_l76_76529

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76529


namespace program_output_l76_76975

theorem program_output :
  let i := 1
      s := 2
  in
  (∃ S, 
    (∀ i, 1 ≤ i ∧ i ≤ 2017 → 
      (S = 1 / (1 - S)) ∧ (i' = i + 1)) → 
    S = -1)
  :=
begin
  let update_S : ℤ → ℤ → ℤ → ℤ := λ i s max_i,
    if i > max_i then s
    else update_S (i + 1) (1 / (1 - s)) max_i,
  have S : ℤ := update_S 1 2 2017,
  show S = -1,
  sorry
end

end program_output_l76_76975


namespace total_surface_area_is_900π_l76_76472

-- Definitions based on conditions
def area_of_base (r : ℝ) : ℝ := π * r ^ 2

def normal_hemisphere_surface_area (r : ℝ) : ℝ := 2 * π * r ^ 2

def adjusted_curved_surface_area (r : ℝ) := (3/2) * 2 * π * r ^ 2

def total_surface_area (r : ℝ) := adjusted_curved_surface_area r + area_of_base r

-- Given the base area condition
def base_area_condition : ℝ := 225 * π

-- Our target proof statement
theorem total_surface_area_is_900π (r : ℝ) (h : area_of_base r = base_area_condition) :
  total_surface_area r = 900 * π :=
sorry

end total_surface_area_is_900π_l76_76472


namespace negation_proposition_l76_76061

theorem negation_proposition : ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ ∃ x : ℝ, x > 0 ∧ x < 1 := 
by
  sorry

end negation_proposition_l76_76061


namespace sum_union_eq_l76_76425

open Set

namespace Proof

def B : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

noncomputable def sum_union (F : Finset (Finset (Set (Fin (10))))) : ℕ :=
  ∑ f in F, |⋃ (A ∈ f), A|

-- Define the problem condition where B and F are given, and we need to prove the sum
theorem sum_union_eq :
  sum_union (Finset.univ : Finset (Set (Fin 10))) = 10 * (2 ^ (10 * n) - 2 ^ (9 * n)) :=
  sorry

end Proof

end sum_union_eq_l76_76425


namespace wendy_tooth_extraction_cost_eq_290_l76_76083

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l76_76083


namespace part1_part2_l76_76709

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l76_76709


namespace g_crosses_asymptote_l76_76681

def g (x : ℝ) : ℝ := (3 * x^3 - 5 * x^2 + x - 1) / (x^2 - 3 * x + 4)

theorem g_crosses_asymptote :
  g 1 = 0 ∧ g (-1/3) = 0 :=
sorry

end g_crosses_asymptote_l76_76681


namespace segment_length_at_least_11_l76_76026

theorem segment_length_at_least_11 
  (five_points : Finset ℕ) 
  (h_card : five_points.card = 5)
  (edge_lengths : Finset ℕ)
  (edge_lengths_distinct : ∀ p₁ p₂ ∈ five_points, p₁ ≠ p₂ → (abs (p₁ - p₂)) ∈ edge_lengths)
  (edges_integers : ∀ d ∈ edge_lengths, d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) :
  ∃ l ≥ 11, ∀ p₁ p₂ ∈ five_points, p₁ ≠ p₂ → (abs (p₁ - p₂)) ≠ (abs (p₁ - p₂)) → edge_lengths ∈ edge_lengths  :=
begin
  sorry
end

end segment_length_at_least_11_l76_76026


namespace cot_sum_inverse_cot_l76_76672

theorem cot_sum_inverse_cot (a b c d : ℝ) (ha : a = 5) (hb : b = 11) (hc : c = 17) (hd : d = 29) :
  Real.cot (Real.arccot a + Real.arccot b + Real.arccot c + Real.arccot d) = 39 / 16 :=
by {
  -- Using the given values
  have ha : a = 5 := ha,
  have hb : b = 11 := hb,
  have hc : c = 17 := hc,
  have hd : d = 29 := hd,
  -- Apply the steps reasoning from the solution
  sorry
}

end cot_sum_inverse_cot_l76_76672


namespace charlie_third_week_data_l76_76024

theorem charlie_third_week_data (d3 : ℕ) : 
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  overage_GB = total_extra_GB -> d3 = 5 := 
by
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  have : overage_GB = total_extra_GB := sorry
  have : d3 = 5 := sorry
  sorry

end charlie_third_week_data_l76_76024


namespace tan_alpha_minus_beta_neg_one_l76_76365

theorem tan_alpha_minus_beta_neg_one 
  (α β : ℝ)
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_neg_one_l76_76365


namespace max_discount_rate_l76_76166

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76166


namespace violet_balloons_count_l76_76406

-- Define the initial number of violet balloons
def initial_violet_balloons := 7

-- Define the number of violet balloons Jason lost
def lost_violet_balloons := 3

-- Define the remaining violet balloons after losing some
def remaining_violet_balloons := initial_violet_balloons - lost_violet_balloons

-- Prove that the remaining violet balloons is equal to 4
theorem violet_balloons_count : remaining_violet_balloons = 4 :=
by
  sorry

end violet_balloons_count_l76_76406


namespace maximum_discount_rate_l76_76138

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76138


namespace sum_of_squares_of_multiples_of_10_l76_76584

theorem sum_of_squares_of_multiples_of_10 :
  (∑ k in finset.range 16, (10 * (k + 1))^2) = 149600 :=
by
  -- Add actual proof steps here
  sorry

end sum_of_squares_of_multiples_of_10_l76_76584


namespace total_wet_surface_area_is_62_l76_76616

-- Define the dimensions of the cistern
def length_cistern : ℝ := 8
def width_cistern : ℝ := 4
def depth_water : ℝ := 1.25

-- Define the calculation of the wet surface area
def bottom_surface_area : ℝ := length_cistern * width_cistern
def longer_side_surface_area : ℝ := length_cistern * depth_water * 2
def shorter_end_surface_area : ℝ := width_cistern * depth_water * 2

-- Sum up all wet surface areas
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_surface_area + shorter_end_surface_area

-- The theorem stating that the total wet surface area is 62 m²
theorem total_wet_surface_area_is_62 : total_wet_surface_area = 62 := by
  sorry

end total_wet_surface_area_is_62_l76_76616


namespace not_diff_of_squares_count_l76_76335

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l76_76335


namespace opposite_of_neg_two_l76_76516

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76516


namespace diff_10_bills_is_17_l76_76435

-- Define the initial amounts of bills each person has
def Mandy_bills : ℕ := 3 * 20
def Manny_bills : ℕ := 2 * 50
def Mary_bills : ℕ := 4 * 10 + 100

-- Define the service fees
def Mandy_fee : ℚ := 2 / 100 * Mandy_bills
def Manny_fee : ℚ := 3 / 100 * Manny_bills
def Mary_fee : ℚ := 5 / 100 * 100

-- Define the amounts after fees
def Mandy_after_fee : ℚ := Mandy_bills - Mandy_fee
def Manny_after_fee : ℚ := Manny_bills - Manny_fee
def Mary_after_fee : ℚ := Mary_bills - Mary_fee

-- Define the number of $10 bills each person will get
def Mandy_10_bills : ℕ := Mandy_after_fee.toNat / 10
def Manny_10_bills : ℕ := Manny_after_fee.toNat / 10
def Mary_10_bills : ℕ := Mary_after_fee.toNat / 10

-- Define the combined number of $10 bills for Manny and Mary
def combined_10_bills : ℕ := Manny_10_bills + Mary_10_bills

-- Define the difference in the number of $10 bills
def difference_10_bills : ℕ := combined_10_bills - Mandy_10_bills

-- Prove that the difference is 17
theorem diff_10_bills_is_17 : difference_10_bills = 17 := by
  -- The proof will be filled here
  sorry

end diff_10_bills_is_17_l76_76435


namespace solve_f_l76_76278

def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

noncomputable def G : Set (ℝ × ℝ) := {(x, y) | x ∈ I ∧ y ∈ I}

theorem solve_f (f : ℝ × ℝ → ℝ) (k : ℝ) (hk : 0 < k) :
  (∀ x y z ∈ I, f (f (x, y), z) = f (x, f (y, z))) ∧
  (∀ x ∈ I, f (x, 1) = x) ∧
  (∀ y ∈ I, f (1, y) = y) ∧
  (∀ z ∈ I, ∀ x y ∈ I, f (z * x, z * y) = z ^ k * f (x, y)) →
  (∃ (f1 : ℝ × ℝ → ℝ), k = 1 ∧ f1 = λ p, min p.1 p.2 ∨
   ∃ (f2 : ℝ × ℝ → ℝ), k = 2 ∧ f2 = λ p, p.1 * p.2) :=
sorry

end solve_f_l76_76278


namespace least_area_of_triangle_PQR_in_decagon_l76_76565

noncomputable def minimal_triangle_area_in_decagon : ℝ :=
  (real.sqrt (2:ℝ)).sqrt.sqrt 
  * (real.sqrt 5 - 1) / (4:ℝ) 
  * (1 - (1 + real.sqrt 5) / (4:ℝ))

theorem least_area_of_triangle_PQR_in_decagon :
  ∃ (z : ℝ → ℂ), 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 9 → 
    z (k : ℝ) = complex.exp (2 * complex.pi * I * k / 10) * (real.sqrt (2:ℝ)).sqrt ∧
    minimal_triangle_area_in_decagon = 
      (real.sqrt (2:ℝ)).sqrt.sqrt * (real.sqrt 5 - 1) / 4 * (1 - (1 + real.sqrt 5) / 4) :=
sorry

end least_area_of_triangle_PQR_in_decagon_l76_76565


namespace opposite_of_neg_two_l76_76552

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76552


namespace younger_son_age_in_30_years_l76_76913

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l76_76913


namespace candy_unclaimed_l76_76628

theorem candy_unclaimed (x : ℝ) : 
  4 / 9 * x + 1 / 3 * x + 2 / 9 * x = x :=
by 
  have h1 : 4 / 9 + 1 / 3 + 2 / 9 = 1, sorry,
  rw [mul_add, ← mul_assoc, ← mul_assoc, h1, one_mul]

end candy_unclaimed_l76_76628


namespace elena_novel_pages_l76_76664

theorem elena_novel_pages
  (days_vacation : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_three_days : ℕ)
  (pages_last_day : ℕ)
  (h1 : days_vacation = 6)
  (h2 : pages_first_two_days = 2 * 42)
  (h3 : pages_next_three_days = 3 * 35)
  (h4 : pages_last_day = 15) :
  pages_first_two_days + pages_next_three_days + pages_last_day = 204 := by
  sorry

end elena_novel_pages_l76_76664


namespace max_integer_is_twelve_l76_76944

theorem max_integer_is_twelve
  (a b c d e : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : (a + b + c + d + e) / 5 = 9)
  (h6 : ((a - 9)^2 + (b - 9)^2 + (c - 9)^2 + (d - 9)^2 + (e - 9)^2) / 5 = 4) :
  e = 12 := sorry

end max_integer_is_twelve_l76_76944


namespace log_sum_eq_five_l76_76323

variable {a : ℕ → ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 3 * a n 

def sum_condition (a : ℕ → ℝ) : Prop :=
a 2 + a 4 + a 9 = 9

-- The mathematical statement to prove
theorem log_sum_eq_five (h1 : geometric_seq a) (h2 : sum_condition a) :
  Real.logb 3 (a 5 + a 7 + a 9) = 5 := 
sorry

end log_sum_eq_five_l76_76323


namespace ellipse_line_properties_l76_76706

theorem ellipse_line_properties :
  (∀ (M : ℝ × ℝ), M ∈ { p : ℝ × ℝ | (p.2^2 / 16 + p.1^2 / 4 = 1) } →
    ∃ α : ℝ, (M = (2 * Real.cos α, 4 * Real.sin α)))
  ∧ (∀ (ρ θ : ℝ), ρ * Real.sin (θ + Real.pi / 3) = 3 →
    ∃ x y : ℝ, (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (sqrt(3) * x + y - 6 = 0))
  ∧ (∀ (M : ℝ × ℝ), 
      M ∈ { p : ℝ × ℝ | (p.2^2 / 16 + p.1^2 / 4 = 1) } →
      (∃ max_val : ℝ, max_val = 9 ∧ ∀ (⟨x, y⟩ : ℝ × ℝ), y^2 / 16 + x^2 / 4 = 1 → |2 * sqrt(3) * x + y - 1| ≤ max_val)) :=
by
  sorry

end ellipse_line_properties_l76_76706


namespace train_crossing_time_l76_76216

-- Length of the train in meters
def train_length : ℕ := 145

-- Speed of the train in km/hr
def train_speed_km_per_hr : ℕ := 45

-- Length of the bridge in meters
def bridge_length : ℕ := 230

-- Convert speed from km/h to m/s
def train_speed_m_per_s : ℝ := (train_speed_km_per_hr : ℝ) * (1000 / 3600)

-- Total distance train needs to travel (length of train + length of bridge)
def total_distance : ℕ := train_length + bridge_length

-- Compute the time to cross the bridge in seconds
def time_to_cross_bridge : ℝ := (total_distance : ℝ) / train_speed_m_per_s

-- The statement to prove
theorem train_crossing_time : time_to_cross_bridge = 30 := 
by 
  sorry

end train_crossing_time_l76_76216


namespace find_interest_rate_l76_76875

theorem find_interest_rate
  (P : ℝ)         -- principal amount
  (t : ℝ)         -- time in years, t = 2.5
  (n : ℝ)         -- number of times interest is compounded per year, n = 2
  (A : ℝ)         -- final amount, A = 1.2762815625000003 * P
  (h_t : t = 2.5)
  (h_n : n = 2)
  (h_A : A = 1.2762815625000003 * P) :
  ∃ r : ℝ, (1 + r/2)^5 = 1.2762815625000003 ∧ r ≈ 0.1 :=
sorry

end find_interest_rate_l76_76875


namespace parabola_ratio_l76_76288

theorem parabola_ratio (P : ℝ) (A B F : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : B = (8, -8))
  (h_parabola : ∀ {x y : ℝ}, y^2 = 2 * P * x → true)
  (hF : F = (P, 0))
  (hA_on_parabola : (A.snd)^2 = 2 * P * A.fst)
  (hP_val : P = 4) :
  |dist A F : dist B F| = 2 / 5 :=
by
  sorry

end parabola_ratio_l76_76288


namespace race_time_comparison_l76_76567

noncomputable def townSquare : ℝ := 3 / 4 -- distance of one lap in miles
noncomputable def laps : ℕ := 7 -- number of laps
noncomputable def totalDistance : ℝ := laps * townSquare -- total distance of the race in miles
noncomputable def thisYearTime : ℝ := 42 -- time taken by this year's winner in minutes
noncomputable def lastYearTime : ℝ := 47.25 -- time taken by last year's winner in minutes

noncomputable def thisYearPace : ℝ := thisYearTime / totalDistance -- pace of this year's winner in minutes per mile
noncomputable def lastYearPace : ℝ := lastYearTime / totalDistance -- pace of last year's winner in minutes per mile
noncomputable def timeDifference : ℝ := lastYearPace - thisYearPace -- the difference in pace

theorem race_time_comparison : timeDifference = 1 := by
  sorry

end race_time_comparison_l76_76567


namespace symmetric_point_coordinates_l76_76051

theorem symmetric_point_coordinates (M N : ℝ × ℝ) (x y : ℝ) 
  (hM : M = (-2, 1)) 
  (hN_symmetry : N = (M.1, -M.2)) : N = (-2, -1) :=
by
  sorry

end symmetric_point_coordinates_l76_76051


namespace max_discount_rate_l76_76130

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76130


namespace p_is_necessary_and_sufficient_for_q_l76_76699

def circle (x y r : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = r ^ 2

def line (x y : ℝ) : Prop := x - (Real.sqrt 3) * y + 3 = 0

def condition_p (r : ℝ) : Prop := 0 < r ∧ r < 3

def distance_from_center_to_line (x y : ℝ) : ℝ:= abs (x - (Real.sqrt 3) * y + 3) / 2

def condition_q (x y r : ℝ) : Prop :=
  ∃ (count : ℕ), circle x y r ∧ (distance_from_center_to_line x y = 1) ∧ count ≤ 2

theorem p_is_necessary_and_sufficient_for_q (r : ℝ) (h₁ : condition_p r) (h₂ : ∃ x y, circle x y r)
  : (condition_q r x y ↔ condition_p r) :=
by sorry

end p_is_necessary_and_sufficient_for_q_l76_76699


namespace find_s_t_l76_76907

noncomputable def problem_constants (a b c : ℝ) : Prop :=
  (a^3 + 3 * a^2 + 4 * a - 11 = 0) ∧
  (b^3 + 3 * b^2 + 4 * b - 11 = 0) ∧
  (c^3 + 3 * c^2 + 4 * c - 11 = 0)

theorem find_s_t (a b c s t : ℝ) (h1 : problem_constants a b c) (h2 : (a + b) * (b + c) * (c + a) = -t)
  (h3 : (a + b) * (b + c) + (b + c) * (c + a) + (c + a) * (a + b) = s) :
s = 8 ∧ t = 23 :=
sorry

end find_s_t_l76_76907


namespace average_side_lengths_of_squares_l76_76473

theorem average_side_lengths_of_squares (A1 A2 A3 : ℝ) (hA1 : A1 = 25) (hA2 : A2 = 64) (hA3 : A3 = 121) : 
  (real.sqrt A1 + real.sqrt A2 + real.sqrt A3) / 3 = 8 := 
by 
  sorry

end average_side_lengths_of_squares_l76_76473


namespace larger_number_is_50_l76_76379

theorem larger_number_is_50 (x y : ℤ) (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 :=
sorry

end larger_number_is_50_l76_76379


namespace Gloin_is_knight_l76_76249

structure Dwarf :=
  (is_knight : Bool) -- True if the dwarf is a knight, False if the dwarf is a liar.

noncomputable def condition (dwarves : List Dwarf) : Prop :=
  dwarves.length = 10 ∧ (∃ dwarf, dwarf ∈ dwarves ∧ dwarf.is_knight = true) ∧
  (∀ i, i < 9 → (
    if dwarves.get! i = dwarf
      then (∃ k, k < i ∧ dwarves.get! k = dwarf)
      else false)) ∧
  (
    if dwarves.get! 9 = dwarf
      then (∃ k, k > 9 ∧ dwarves.get! k = dwarf)
      else false)
    
theorem Gloin_is_knight (dwarves : List Dwarf) (h : condition dwarves) : dwarves.get! 9.is_knight = true :=
by {
  sorry
}

end Gloin_is_knight_l76_76249


namespace count_diff_squares_not_representable_1_to_1000_l76_76339

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l76_76339


namespace range_of_a_l76_76317

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * Real.log (1 + x) + x^2 else -x * Real.log (1 - x) + x^2

theorem range_of_a (a : ℝ) : f (-a) + f (a) ≤ 2 * f (1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l76_76317


namespace identify_vintik_l76_76107

-- Definitions based on the conditions
def is_vintik (A B : Type) := ∃ v : A ⊕ B, True

def lies (A : Type) : Prop := ∀ x : A, ¬ x

def truthfulness_condition (A B : Type) (instance : A ⊕ B): Prop :=
  match instance with
  | Sum.inl a => lies A ∨ lies B
  | Sum.inr b => lies A ∨ lies B

-- Main statement
theorem identify_vintik {A B : Type} (instance : A ⊕ B)
  (h1 : is_vintik A B) (h2 : truthfulness_condition A B instance)
  (h_answer_A : instance = Sum.inl A) (h_answer_B : instance = Sum.inr B) :
  instance = Sum.inr B :=
by
  sorry

end identify_vintik_l76_76107


namespace shaded_area_of_rectangle_l76_76085

theorem shaded_area_of_rectangle (base height : ℕ) (total_area : ℕ) :
  base = 7 → height = 4 → total_area = 56 →
  let triangle_area := (base * height) / 2 in
  let unshaded_area := 2 * triangle_area in
  let shaded_area := total_area - unshaded_area in
  shaded_area = 28 :=
by
  intros h_base h_height h_total_area
  let triangle_area := (base * height) / 2
  have h_triangle_area : triangle_area = 14 := by
    simp [h_base, h_height]
  let unshaded_area := 2 * triangle_area
  have h_unshaded_area : unshaded_area = 28 := by
    simp [h_triangle_area]
  let shaded_area := total_area - unshaded_area
  have h_shaded_area : shaded_area = 28 := by
    simp [h_total_area, h_unshaded_area]
  exact h_shaded_area

end shaded_area_of_rectangle_l76_76085


namespace sum_of_13th_positions_l76_76991

/-- Consider a regular 100-sided polygon with vertices numbered from 1 to 100. These vertices are 
rewritten in order based on their distance from a fixed edge such that if two vertices are at equal 
distance, the left vertex number appears first. We are to compute the sum of the numbers appearing 
at the 13th position from the left in all possible configurations of this 100-sided polygon rotated 
about its center. -/
theorem sum_of_13th_positions (n : ℕ) (h : n = 100) :
  let positions := list.finRange n,
      rot_sum := list.sum $ positions.map (λ k, 2 * k.succ) 
  in rot_sum = 10100 :=
by
  sorry

end sum_of_13th_positions_l76_76991


namespace light_bulbs_on_possible_l76_76082

theorem light_bulbs_on_possible (G : SimpleGraph V) (all_bulbs_off: ∀ v : V, state[v] = off) : 
  ∃ n : ℕ, step n
  sorry

end light_bulbs_on_possible_l76_76082


namespace exists_n0_l76_76830

noncomputable def seq (n : ℕ) : ℕ := sorry  -- Define the sequence (details skipped)

def S (n : ℕ) : ℕ := (finset.range n).sum seq  -- Define the sum of the first n terms

theorem exists_n0 (h : ∀ n ≥ 2002, seq (n + 1) ∣ S n) :
  ∃ n0, ∀ n ≥ n0, seq (n + 1) = S n := 
sorry

end exists_n0_l76_76830


namespace maximum_discount_rate_l76_76177

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76177


namespace opposite_of_neg2_l76_76507

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76507


namespace maximum_discount_rate_l76_76173

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76173


namespace sum_of_cubes_l76_76906

theorem sum_of_cubes (p q r : ℝ) (h1 : p + q + r = 7) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 :=
by
  sorry

end sum_of_cubes_l76_76906


namespace probability_at_least_one_boy_one_girl_l76_76101

theorem probability_at_least_one_boy_one_girl :
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let all_boys := (p_boy ^ 4)
  let all_girls := (p_girl ^ 4)
  let at_least_one_boy_one_girl := 1 - (all_boys + all_girls)
  in at_least_one_boy_one_girl = 7 / 8 :=
by
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let all_boys := (p_boy ^ 4)
  let all_girls := (p_girl ^ 4)
  let at_least_one_boy_one_girl := 1 - (all_boys + all_girls)
  show at_least_one_boy_one_girl = 7 / 8
  sorry

end probability_at_least_one_boy_one_girl_l76_76101


namespace opposite_of_neg_two_is_two_l76_76538

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76538


namespace BE_perpendicular_NC_l76_76576

open_locale classical

variable {Point : Type*}

variables [inner_product_space ℝ Point]
variables (B C D A K M N : Point) (E F : Point)

-- Two squares BCDA and BKMN share a common vertex B
def is_square (B C D A : Point) : Prop :=
  (dist B C = dist C D) ∧ (dist C D = dist D A) ∧ (dist D A = dist A B) ∧ 
  (∠ B C D = π / 2) ∧ (∠ C D A = π / 2) ∧ (∠ D A B = π / 2) ∧ (∠ A B C = π / 2)

-- B, C, D, A form a square
axiom h1 : is_square B C D A

-- B, K, M, N form a square
axiom h2 : is_square B K M N

-- Define midpoint E of AK and midpoint F of CN
def is_midpoint (P Q R : Point) : Prop := ∃ S : Point, dist P S = dist S Q ∧ segment S R

axiom is_median_E : is_midpoint A K E
axiom is_altitude_F : dist B F = dist B N ∧ segment F C ×
                       ∠ B F C = π / 2

-- Main lemma: BE and BF lie on the same line
theorem BE_perpendicular_NC : ⟦BE⟧ ∋ ⟦NC⟧ :=
sorry

end BE_perpendicular_NC_l76_76576


namespace remainder_x500_is_x2_l76_76257

open Polynomial

noncomputable def remainder_x500 (R : Type) [CommRing R] : Polynomial R :=
  let f := (C (1 : R)) * X ^ 3 + (C (-1 : R)) * X ^ 2 + (C (-1 : R)) * X + (C (1 : R))
  let p := X ^ 500
  p % f

theorem remainder_x500_is_x2 (R : Type) [CommRing R] : 
  remainder_x500 R = C (1 : R) * X ^ 2 :=
by sorry

end remainder_x500_is_x2_l76_76257


namespace pirate_coins_l76_76619

def coins_remain (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then x else coins_remain (k - 1) x * (15 - k) / 15

theorem pirate_coins (x : ℕ) :
  (∀ k < 15, (k + 1) * coins_remain k x % 15 = 0) → 
  coins_remain 14 x = 8442 :=
sorry

end pirate_coins_l76_76619


namespace max_discount_rate_l76_76127

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76127


namespace cat_max_distance_from_origin_l76_76440

theorem cat_max_distance_from_origin :
  let center := (6, 8)
  let radius := 15
  let origin := (0, 0)
  let distance (p q : ℕ × ℕ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let center_distance := distance origin center
  center_distance + radius = 25 :=
by
  let center := (6, 8)
  let radius := 15
  let origin := (0, 0)
  let distance (p q : ℕ × ℕ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let center_distance := distance origin center
  have h1 : center_distance = 10 := by sorry
  have h2 : 10 + 15 = 25 := by norm_num
  show 10 + 15 = 25 from h2

end cat_max_distance_from_origin_l76_76440


namespace circle_area_diameter_six_l76_76087

theorem circle_area_diameter_six :
  let d := 6 in
  let radius := d / 2 in
  let area := Real.pi * (radius ^ 2) in
  area = 9 * Real.pi := by
  let d := 6
  let radius := d / 2
  have : radius = 3 := by norm_num
  let area := Real.pi * (radius ^ 2)
  have : area = Real.pi * (3 ^ 2) := by rw [this]
  have : area = 9 * Real.pi := by norm_num
  exact this

end circle_area_diameter_six_l76_76087


namespace count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76741

-- The problem states the following conditions
def is_divisible_by (m n : ℕ) : Prop := m % n = 0
def less_than_300 (n : ℕ) : Prop := n < 300

-- We want to prove the number of positive integers less than 300 that are divisible by 4, 6, and 10
theorem count_positive_integers_divisible_by_4_6_10_less_than_300 :
  (Finset.card (Finset.filter 
    (λ n, is_divisible_by n 4 ∧ is_divisible_by n 6 ∧ is_divisible_by n 10 ∧ less_than_300 n)
    ((Finset.range 300).filter (λ n, n ≠ 0)))) = 4 :=
by
  sorry

end count_positive_integers_divisible_by_4_6_10_less_than_300_l76_76741


namespace shortest_distance_to_left_focus_l76_76562

def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

theorem shortest_distance_to_left_focus : 
  ∃ P : ℝ × ℝ, 
  hyperbola P.1 P.2 ∧ 
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → dist Q left_focus ≥ dist P left_focus) ∧ 
  dist P left_focus = 2 :=
sorry

end shortest_distance_to_left_focus_l76_76562


namespace sam_collected_42_cans_l76_76996

noncomputable def total_cans_collected (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  bags_saturday + bags_sunday * cans_per_bag

theorem sam_collected_42_cans :
  total_cans_collected 4 3 6 = 42 :=
by
  sorry

end sam_collected_42_cans_l76_76996


namespace Talia_total_distance_l76_76469

variable (Talia : Type)
variable (house park store : Talia)

-- Define the distances given in the conditions
variable (distance : Talia → Talia → ℕ)
variable (h2p : distance house park = 5)
variable (p2s : distance park store = 3)
variable (s2h : distance store house = 8)

-- Define the total distance function
def total_distance (t : Talia) : ℕ :=
  distance house park + distance park store + distance store house

-- Lean 4 theorem statement
theorem Talia_total_distance : total_distance Talia house park store distance = 16 :=
by
  simp [total_distance, h2p, p2s, s2h]
  sorry

end Talia_total_distance_l76_76469


namespace tan_alpha_minus_beta_neg_one_l76_76367

theorem tan_alpha_minus_beta_neg_one 
  (α β : ℝ)
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_neg_one_l76_76367


namespace f_periodicity_f_def_0_2_f_neg_2017_l76_76307

namespace Mathlib

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 2 then x * (2 - x) else 0

theorem f_periodicity (x : ℝ) : f x = - f (x + 2) :=
sorry

theorem f_def_0_2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x = x * (2 - x) :=
by
  unfold f
  split_ifs
  . refl
  . contradiction

theorem f_neg_2017 : f (-2017) = -1 :=
sorry

end Mathlib

end f_periodicity_f_def_0_2_f_neg_2017_l76_76307


namespace ones_digit_of_p_is_3_l76_76689

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l76_76689


namespace Q_subset_P_l76_76712

-- Definitions
def P : Set ℝ := {x : ℝ | x ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Statement to prove
theorem Q_subset_P : Q ⊆ P :=
sorry

end Q_subset_P_l76_76712


namespace minimum_value_a_plus_3b_plus_9c_l76_76007

open Real

theorem minimum_value_a_plus_3b_plus_9c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
sorry

end minimum_value_a_plus_3b_plus_9c_l76_76007


namespace max_sum_of_products_l76_76941

theorem max_sum_of_products (p q r s : ℕ) 
  (h1 : p ∈ {2, 3, 4, 5}) 
  (h2 : q ∈ {2, 3, 4, 5}) 
  (h3 : r ∈ {2, 3, 4, 5}) 
  (h4 : s ∈ {2, 3, 4, 5}) 
  (h_neq : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  pq + qr + rs + sp ≤ 49 :=
by
  sorry

end max_sum_of_products_l76_76941


namespace minimum_soldiers_to_add_l76_76791

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76791


namespace find_PF_2_l76_76381

-- Define the hyperbola and points
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def PF_1 := 3
def a := 2
def two_a := 2 * a

-- State the theorem
theorem find_PF_2 (PF_2 : ℝ) (cond1 : PF_1 = 3) (cond2 : abs (PF_1 - PF_2) = two_a) : PF_2 = 7 :=
sorry

end find_PF_2_l76_76381


namespace count_non_representable_as_diff_of_squares_l76_76332

theorem count_non_representable_as_diff_of_squares :
  let count := (Finset.filter (fun n => ∃ k, n = 4 * k + 2 ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card in
  count = 250 :=
by
  sorry

end count_non_representable_as_diff_of_squares_l76_76332


namespace number_of_valid_values_of_M_l76_76693

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem number_of_valid_values_of_M :
  ∃ (M : ℕ), M ∈ {n | 10 ≤ n ∧ n ≤ 99} ∧ 
  (∃ (a b : ℕ), M = 10 * a + b ∧ M - (10 * b + a) = 64 ∧
  is_perfect_cube 64) ↔
  ∃! (M : ℕ), M ∈ {81, 92} :=
sorry

end number_of_valid_values_of_M_l76_76693


namespace younger_son_age_30_years_later_eq_60_l76_76920

variable (age_diff : ℕ) (elder_age : ℕ) (younger_age_30_years_later : ℕ)

-- Conditions
axiom h1 : age_diff = 10
axiom h2 : elder_age = 40

-- Definition of younger son's current age
def younger_age : ℕ := elder_age - age_diff

-- Definition of younger son's age 30 years from now
def younger_age_future : ℕ := younger_age + 30

-- Proving the required statement
theorem younger_son_age_30_years_later_eq_60 (h_age_diff : age_diff = 10) (h_elder_age : elder_age = 40) :
  younger_age_future elder_age age_diff = 60 :=
by
  unfold younger_age
  unfold younger_age_future
  rw [h_age_diff, h_elder_age]
  sorry

end younger_son_age_30_years_later_eq_60_l76_76920


namespace necessary_not_sufficient_condition_l76_76419

theorem necessary_not_sufficient_condition
  (a b : ℝ) (α β : ℝ)
  (h₁ : a = 1)
  (h₂ : b = real.sqrt 3)
  (h₃ : α = 30)
  (h₄ : β = 60) :
  ¬ ((a = 1 ∧ b = real.sqrt 3 ∧ α = 30 ∧ β = 60) ↔ (necessary_condition_for α β a b)) :=
sorry

end necessary_not_sufficient_condition_l76_76419


namespace geometric_sequence_sum_10_l76_76404

theorem geometric_sequence_sum_10 (a : ℕ) (r : ℕ) (h : r = 2) (sum5 : a + r * a + r^2 * a + r^3 * a + r^4 * a = 1) : 
    a * (1 - r^10) / (1 - r) = 33 := 
by 
    sorry

end geometric_sequence_sum_10_l76_76404


namespace max_discount_rate_l76_76158

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76158


namespace supplement_of_supplement_l76_76965

def supplement (angle : ℝ) : ℝ :=
  180 - angle

theorem supplement_of_supplement (θ : ℝ) (h : θ = 35) : supplement (supplement θ) = 35 := by
  -- It is enough to state the theorem; the proof is not required as per the instruction.
  sorry

end supplement_of_supplement_l76_76965


namespace min_delivery_time_l76_76568

def elf_delivery_time (distances : Fin 63 → ℕ) (times : Fin 63 → ℕ) : ℕ :=
  (Finset.univ : Finset (Fin 63)).sup (λ i, i.val * distances i)

theorem min_delivery_time (distances : Fin 63 → ℕ) (times : Fin 63 → ℕ)
  (h_distinct : Function.Injective distances)
  (h_range : ∀ i, distances i ∈ Finset.range 64 \ Finset.range 1) :
  elf_delivery_time distances times = 1024 :=
begin
  sorry
end

end min_delivery_time_l76_76568


namespace fixed_point_l76_76235

def Circle (C : Type) := {p : C × C // p.1 * p.1 + (p.2 - 2) * (p.2 - 2) = 16}

def symmetric_with (L : Type) (a b : ℕ) := L = λ x y : ℕ, ax + by - 12

def Point (P : Type) := {p : P // p.2 = -6}

def Tangent_line (S : Type) := {l : S × S // l.1 ≠ l.2}

theorem fixed_point (a b : ℕ) (C : Circle ℕ) (L : symmetric_with ℕ a b) (S : Point ℕ) (A B : Tangent_line ℕ) :
  ∃ p : ℕ × ℕ, p = (0, 0) :=
sorry

end fixed_point_l76_76235


namespace ones_digit_of_prime_p_l76_76691

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l76_76691


namespace min_soldiers_needed_l76_76775

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76775


namespace smallest_x_divisibility_l76_76094

theorem smallest_x_divisibility (x : ℕ) : 
  (x % 6 = 5) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 167 :=
begin
  sorry
end

end smallest_x_divisibility_l76_76094


namespace ada_original_seat_l76_76267

theorem ada_original_seat {a b c d e : ℕ} : 
  (a ∈ {1, 5}) → 
  b = a + 2 → 
  c = a - 1 → 
  d ≠ e → 
  (a, 5) ∈ (1, 5) → 
  a = 2 :=
by
  -- Placeholder proof.
  sorry

end ada_original_seat_l76_76267


namespace student_game_incorrect_statement_l76_76047

theorem student_game_incorrect_statement (a : ℚ) : ¬ (∀ a : ℚ, -a - 2 < 0) :=
by
  -- skip the proof for now
  sorry

end student_game_incorrect_statement_l76_76047


namespace find_A_for_diamondsuit_l76_76347

-- Define the operation
def diamondsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- Define the specific instance of the operation equated to 57
theorem find_A_for_diamondsuit :
  ∃ A : ℝ, diamondsuit A 10 = 57 ↔ A = 20 := by
  sorry

end find_A_for_diamondsuit_l76_76347


namespace range_sin_cos_range_sin_cos_sin2x_l76_76998

theorem range_sin_cos (x : ℝ) : -real.sqrt 2 ≤ real.sin x + real.cos x ∧ real.sin x + real.cos x ≤ real.sqrt 2 := sorry

theorem range_sin_cos_sin2x (x : ℝ) : -1 - real.sqrt 2 ≤ (real.sin x + real.cos x - real.sin (2 * x)) ∧ (real.sin x + real.cos x - real.sin (2 * x)) ≤ 5 / 4 := sorry

end range_sin_cos_range_sin_cos_sin2x_l76_76998


namespace max_value_of_f_on_interval_l76_76316

noncomputable def f (x : ℝ) : ℝ := - (1/3) * x^3 + x^2 + 3 * x - 5

theorem max_value_of_f_on_interval :
  f 3 = 4 ∧ (f' 3 = 0) → ∃ x ∈ Icc (-2 : ℝ) (1 : ℝ), f x = -4 / 3 := 
sorry

end max_value_of_f_on_interval_l76_76316


namespace find_length_LB_l76_76441

open Real -- We will make use of real numbers, hence open the real namespace

/-- 
Given the conditions:
1. The angle \( \angle LKC = 45^\circ \)
2. The distance \( AK = 1 \)
3. The distance \( KD = 2 \)
Prove that the length \( LB = 2 \).
-/
theorem find_length_LB
  (A B C D K L : ℝ) -- Substitute geometric points with real values
  (aK: ℝ) : aK = 1 -> -- AK = 1
  (kD: ℝ) : kD = 2 -> -- KD = 2
  (angle_LKC: ℝ) : angle_LKC = 45 -> -- Angle LKC = 45 degrees
  LB = 2 := sorry

end find_length_LB_l76_76441


namespace solve_primes_l76_76667

open Int

theorem solve_primes :
  ∀ p : ℕ, prime p → ∃ k : ℕ, (2^(p+1) - 4) = p * k^2 ↔ p = 3 ∨ p = 7 :=
by sorry

end solve_primes_l76_76667


namespace distance_sum_correct_l76_76682

noncomputable def distance_sum
  (rA rB rC rD : ℝ)
  (A B C D P Q R : Point)
  (h1 : r_A = 2 / 3 * r_B)
  (h2 : r_C = 2 / 3 * r_D)
  (h3 : distance A B = 45)
  (h4 : distance C D = 45)
  (h5 : distance P Q = 50)
  (h6 : midpoint R P Q)
  : ℝ :=
  distance A R + distance B R + distance C R + distance D R

theorem distance_sum_correct
  (rA rB rC rD : ℝ)
  (A B C D P Q R : Point)
  (h1 : r_A = 2 / 3 * r_B)
  (h2 : r_C = 2 / 3 * r_D)
  (h3 : distance A B = 45)
  (h4 : distance C D = 45)
  (h5 : distance P Q = 50)
  (h6 : midpoint R P Q)
  : distance A R + distance B R + distance C R + distance D R = 140 :=
sorry

end distance_sum_correct_l76_76682


namespace remainder_of_sum_150_div_5550_l76_76088

theorem remainder_of_sum_150_div_5550 : 
  let sum := (150 * (150 + 1)) / 2 in
  (sum % 5550) = 225 :=
by
  let sum := (150 * (150 + 1)) / 2
  have h : sum = 11325 := by norm_num
  rw [h]
  have r : 11325 % 5550 = 225 := by norm_num
  exact r

end remainder_of_sum_150_div_5550_l76_76088


namespace L_geq_one_third_l76_76428

theorem L_geq_one_third (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
    (h : a * b + b * c + c * d + d * a = 1) : 
    (a^3 / (b + c + d)) + 
    (b^3 / (c + d + a)) + 
    (c^3 / (a + b + d)) + 
    (d^3 / (a + b + c)) ≥ (1 /  3) :=
begin
  sorry
end

end L_geq_one_third_l76_76428


namespace solve_x_in_equation_l76_76900

theorem solve_x_in_equation (a b x : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : x ≠ 0) : 
  (b ≠ 0 ∧ (1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) → x = a^2 - b^2) ∧ 
  (b = 0 ∧ a ≠ 0 ∧ (1 / a + a / x = 1 / a + a / x) → x ≠ 0) := 
by
  sorry

end solve_x_in_equation_l76_76900


namespace maximum_discount_rate_l76_76140

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76140


namespace sufficient_not_necessary_l76_76754

def alpha : ℝ := Real.pi / 6
def tan_pi_minus_alpha := Real.tan (Real.pi - alpha)

theorem sufficient_not_necessary :
  (alpha = Real.pi / 6 → tan_pi_minus_alpha = - Real.sqrt 3 / 3) ∧ 
  ¬ (tan_pi_minus_alpha = - Real.sqrt 3 / 3 → alpha = Real.pi / 6) :=
by
  sorry

end sufficient_not_necessary_l76_76754


namespace license_plates_count_l76_76236

-- Define the alphabet in the given language
def alphabet : List Char := ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'L', 'M']

-- Function to count the number of valid license plates
def count_license_plates (alphabet : List Char) : Nat :=
  let letters := alphabet.erase 'A'
  let choices_for_first := letters.filter (fun c => c = 'B' ∨ c = 'D').length
  let choices_for_last := 1
  let remaining_choices := letters.erase 'J'
  choices_for_first * choices_for_last *
  (remaining_choices.length - 0) *
  (remaining_choices.length - 1) *
  (remaining_choices.length - 2) *
  (remaining_choices.length - 3) *
  (remaining_choices.length - 4)

theorem license_plates_count :
  count_license_plates alphabet = 1680 := by
  sorry

end license_plates_count_l76_76236


namespace opposite_of_neg_2_is_2_l76_76499

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76499


namespace angle_equality_l76_76030

-- Define the problem conditions and prove the required angle equality
theorem angle_equality (A B C D M P Q K : Point) 
  (hM_midpoint : M = midpoint B C)
  (hP_on_AD : P ∈ line_through A D)
  (hPM_intersects_DC : ∃ Q, Q ∈ line_through P M ∧ Q ∈ line_through D C)
  (hK_perpendicular : ∃ K, K ∈ line_through P (foot P (line_through A D)) ∧ K ∈ line_through B Q)
  : ∠ Q B C = ∠ K D A :=
sorry

end angle_equality_l76_76030


namespace probability_in_math_l76_76761

-- Define the total number of letters in the alphabet.
def total_letters : ℕ := 26

-- Define the unique letters in "MATHEMATICS".
def unique_letters_math : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Define the count of unique letters in the word "MATHEMATICS".
def count_unique_letters : ℕ := unique_letters_math.card

-- Define the probability of picking a letter from "MATHEMATICS".
def probability_math_letter : ℚ := count_unique_letters.to_nat / total_letters

-- The theorem statement
theorem probability_in_math (total_letters = 26) (count_unique_letters = 8) : probability_math_letter = 4 / 13 := 
by sorry

end probability_in_math_l76_76761


namespace solve_consecutive_integers_solve_consecutive_even_integers_l76_76315

-- Conditions: x, y, z, w are positive integers and x + y + z + w = 46.
def consecutive_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) ∧ (x + y + z + w = 46)

def consecutive_even_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 2 = y) ∧ (y + 2 = z) ∧ (z + 2 = w) ∧ (x + y + z + w = 46)

-- Proof that consecutive integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_integers : ∃ x y z w : ℕ, consecutive_integers_solution x y z w :=
sorry

-- Proof that consecutive even integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_even_integers : ∃ x y z w : ℕ, consecutive_even_integers_solution x y z w :=
sorry

end solve_consecutive_integers_solve_consecutive_even_integers_l76_76315


namespace total_number_of_coins_l76_76070

theorem total_number_of_coins (total_paise := 7100) (n20 := 290) (value20 := 20) (value25 := 25) :
  total_paise = (71 * 100) ∧ 
  n20 * value20 = 5800 ∧
  (total_paise - n20 * value20) / value25 = 52 ∧
  (n20 + ((total_paise - n20 * value20) / value25)) = 342 :=
begin
  sorry
end

end total_number_of_coins_l76_76070


namespace petya_can_write_divisible_by_2019_l76_76444

open Nat

theorem petya_can_write_divisible_by_2019 (M : ℕ) (h : ∃ k : ℕ, M = (10^k - 1) / 9) : ∃ N : ℕ, (N = (10^M - 1) / 9) ∧ 2019 ∣ N :=
by
  sorry

end petya_can_write_divisible_by_2019_l76_76444


namespace shared_candy_equally_l76_76749

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l76_76749


namespace prob_f_is_increasing_l76_76209

def f (a x : ℝ) : ℝ :=
  if x ≤ -1 then a^(x+3) else (3 - a) * x - a + 7

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem prob_f_is_increasing :
  ∀ (a : ℝ), (a ∈ Set.Ioo 0 6) →
  (is_increasing_on (f a) Set.univ) →
  (∃ (p : ℝ), p = (2 - 1) / (6 - 0) ∧ p = 1 / 6) :=
by
  sorry

end prob_f_is_increasing_l76_76209


namespace arithmetic_sqrt_of_linear_combination_l76_76310

theorem arithmetic_sqrt_of_linear_combination (a b : ℝ)
  (h1 : real.sqrt (a + 9) = -5)
  (h2 : real.cbrt (2 * b - a) = -2) :
  real.sqrt (2 * a + b) = 6 :=
by
  sorry

end arithmetic_sqrt_of_linear_combination_l76_76310


namespace parallel_planes_implies_parallel_lines_l76_76016

variables (α β : Plane) (m n : Line)

-- Defining the relationship between a line and a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Axiom stating m is in plane α
axiom hm_in_alpha : line_in_plane m α

-- Axiom stating n is in plane β
axiom hn_in_beta : line_in_plane n β

-- Theorem: If α is parallel to β, then m is parallel to β
theorem parallel_planes_implies_parallel_lines :
  (α ∥ β) → (m ∥ β) := sorry

end parallel_planes_implies_parallel_lines_l76_76016


namespace max_observing_relations_lemma_l76_76942

/-- There are 24 robots on a plane, each with a 70-degree field of view. -/
def robots : ℕ := 24

/-- Definition of field of view for each robot. -/
def field_of_view : ℝ := 70

/-- Maximum number of observing relations. Observing is a one-sided relation. -/
def max_observing_relations := 468

/-- Theorem: The maximum number of observing relations among 24 robots,
each with a 70-degree field of view, is 468. -/
theorem max_observing_relations_lemma : max_observing_relations = 468 :=
by
  sorry

end max_observing_relations_lemma_l76_76942


namespace cyclic_quadrilateral_diagonal_ratios_l76_76936

theorem cyclic_quadrilateral_diagonal_ratios {A B C D M : Type*} 
  (cyclic_quad : (AB = 2) ∧ (BC = 3) ∧ (CD = 6) ∧ (DA = 4))
  (AC_diagonal : LineSegment A C)
  (BD_diagonal : LineSegment B D)
  (intersection_point : collinear M [] [A C] ∧ collinear M [] [B D]) :
  (ratio (A, M, C) = 4 : 1) ∧ (ratio (D, M, B) = 1 : 1) :=
sorry

end cyclic_quadrilateral_diagonal_ratios_l76_76936


namespace frog_jump_distance_l76_76058

theorem frog_jump_distance (G F : ℕ) (hG : G = 36) (hF : F = G + 17) : F = 53 := by
  rw [hG] at hF
  rw [←hF]
  sorry

end frog_jump_distance_l76_76058


namespace friends_total_sales_l76_76593

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end friends_total_sales_l76_76593


namespace minimum_soldiers_to_add_l76_76798

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76798


namespace find_angle_and_area_l76_76038

structure Triangle (α : Type*) :=
(K L M : α)

structure Circle (α : Type*) :=
(radius : ℝ) (K touches_LM_at_B intersects_KL_at_A : α)

variable {α : Type*} [n : NormedAddCommGroup α] [m : InnerProductSpace ℝ α]
open Real

def segment_bisector (K L : α) (B : α) := true  -- Placeholder definition to express that B is the bisector.

-- Given conditions
def conditions (triangle : Triangle α) (circle : Circle α) : Prop :=
  let (Triangle.mk K L M) := triangle in
  let (Circle.mk radius K touches_LM_at_B intersects_KL_at_A) := circle in
  segment_bisector K L touches_LM_at_B ∧ -- KB is the bisector
  radius = 5 ∧                                     -- Circle's radius is 5
  let B := touches_LM_at_B in
  let A := intersects_KL_at_A in
  let M_to_L := dist M L in
  M_to_L = 9 * sqrt 3 ∧                            -- ML = 9√3
  let K_to_A := 5 in
  let L_to_B := 6 in
  K_to_A / L_to_B = 5 / 6                          -- KA : LB = 5 : 6

-- Proof problem
theorem find_angle_and_area (triangle : Triangle α) (circle : Circle α) (h : conditions triangle circle):
  let (Triangle.mk K L M) := triangle in
  -- Prove the angle ∠MKL
  ∠MKL = π / 3 ∧
  -- Prove the area of triangle KLM
  let area := (81 * sqrt 3) / 16 in
  true := sorry -- Function to compute area of triangle will be a placeholder here.

end find_angle_and_area_l76_76038


namespace tetrahedron_has_four_vertices_l76_76093

noncomputable def tetrahedron := Type -- Define a tetrahedron as a type

-- Hypothesis to express that a tetrahedron has exactly 4 vertices
axiom vertices_count : ∀ T : tetrahedron, fintype.card (vertex T) = 4

theorem tetrahedron_has_four_vertices (T : tetrahedron) : fintype.card (vertex T) = 4 :=
vertices_count T

end tetrahedron_has_four_vertices_l76_76093


namespace mary_needs_more_sugar_l76_76436

def recipe_sugar := 14
def sugar_already_added := 2
def sugar_needed := recipe_sugar - sugar_already_added

theorem mary_needs_more_sugar : sugar_needed = 12 := by
  sorry

end mary_needs_more_sugar_l76_76436


namespace trapezoid_bisectors_intersect_midpoint_l76_76442

variable {a b : ℝ}
variable (A B C D K M : Type*)

structure IsTrapezoid (A B C D : Type*) :=
(base1 base2 : ℝ)
(leg : ℝ)
(leg_eq_sum_bases : ∀ (a b : ℝ), base1 = a ∧ base2 = b ∧ leg = a + b)

structure AngleBisectorsIntersectMidpoint (A B C D : Type*) (K : Type*) :=
(is_bisector_A : K ∈ bisector_of ∠ A)
(is_bisector_B : K ∈ bisector_of ∠ B)
(intersects_mid_C_D : K ∈ midpoint C D)

theorem trapezoid_bisectors_intersect_midpoint :
  ∀ (A B C D K : Type*) (a b : ℝ), (IsTrapezoid A B C D ∧ IsLegEqSumBases A B C D a b) → AngleBisectorsIntersectMidpoint A B C D K :=
by
  intro A B C D K a b h
  cases' h with ht he
  sorry

end trapezoid_bisectors_intersect_midpoint_l76_76442


namespace find_a1_l76_76292

-- Definitions used in the conditions
variables {a : ℕ → ℝ} -- Sequence a(n)
variable (n : ℕ) -- Number of terms
noncomputable def arithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def arithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m + (n - m) * (a 2 - a 1)

theorem find_a1 (h_seq : arithmeticSeq a)
  (h_sum_first_100 : arithmeticSum a 100 = 100)
  (h_sum_last_100 : arithmeticSum (λ i => a (i + 900)) 100 = 1000) :
  a 1 = 101 / 200 :=
  sorry

end find_a1_l76_76292


namespace opposite_of_neg_two_l76_76510

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76510


namespace sum_of_fractions_l76_76641

theorem sum_of_fractions : 
  (1/12 + 2/12 + 3/12 + 4/12 + 5/12 + 6/12 + 7/12 + 8/12 + 9/12 + 65/12 + 3/4) = 119 / 12 :=
by
  sorry

end sum_of_fractions_l76_76641


namespace ones_digit_of_prime_p_l76_76692

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l76_76692


namespace minimum_soldiers_to_add_l76_76782

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76782


namespace minimum_soldiers_to_add_l76_76792

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76792


namespace solve_inner_parentheses_l76_76109

theorem solve_inner_parentheses (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 57 ↔ x = 18 := by
  sorry

end solve_inner_parentheses_l76_76109


namespace sum_of_edge_lengths_at_least_3d_l76_76888

-- Condition definitions
variable (P : Type) [polyhedron P]

-- Points A and B with distance d between them
variable (A B : P) (d : ℝ) 
variable (vertices : set P) [finite vertices]
variable (V_A V_B : A ∈ vertices ∧ B ∈ vertices)
variable (edges : set (P × P))
variable (sum_edge_lengths : ℝ)
variable (dist_eq_d : dist A B = d)

-- Definition of polyhedron edges
def sum_of_edge_lengths (edges : set (P × P)) : ℝ :=
  ∑ edge in edges, dist edge.1 edge.2

-- The theorem to prove
theorem sum_of_edge_lengths_at_least_3d :
  sum_of_edge_lengths edges ≥ 3 * d := 
sorry

end sum_of_edge_lengths_at_least_3d_l76_76888


namespace goats_more_than_pigs_l76_76571

theorem goats_more_than_pigs : 
  let G := 66 in
  let C := 2 * G in
  let D := (1 / 2) * (G + C) in
  let P := (1 / 3) * D in
  G - P = 33 := 
by
  sorry

end goats_more_than_pigs_l76_76571


namespace distance_between_trees_l76_76601

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (h1 : yard_length = 434) (h2 : num_trees = 32) :
  yard_length / (num_trees - 1) = 14 :=
by {
  rw [h1, h2],
  norm_num,
}


end distance_between_trees_l76_76601


namespace ones_digit_of_prime_p_l76_76690

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l76_76690


namespace minimum_soldiers_to_add_l76_76799

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76799


namespace minimum_soldiers_to_add_l76_76788

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76788


namespace max_discount_rate_l76_76159

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76159


namespace find_lambda_l76_76329

noncomputable def vector_a (x : Real) : Real × Real := (Real.cos (3 / 2 * x), Real.sin (3 / 2 * x))
noncomputable def vector_b (x : Real) : Real × Real := (Real.cos (x / 2), -Real.sin (x / 2))

def dot_product (a b : Real × Real) : Real := a.1 * b.1 + a.2 * b.2
def magnitude (a : Real × Real) : Real := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

def f (x λ : Real) : Real := 
  dot_product (vector_a x) (vector_b x) - 2 * λ * magnitude (vector_a x + vector_b x)

theorem find_lambda (x : Real) (h : 0 < x ∧ x < Real.pi / 2) (h_min : f x (1 / 2) = -3 / 2) : λ = 1 / 2 := by
  sorry

end find_lambda_l76_76329


namespace ascending_order_l76_76870

noncomputable theory

open Int

def a := 5^2019 - 10 * (floor ((5^2019 : ℝ) / 10))
def b := 7^2020 - 10 * (floor ((7^2020 : ℝ) / 10))
def c := 13^2021 - 10 * (floor ((13^2021 : ℝ) / 10))

theorem ascending_order : b < c < a := sorry

end ascending_order_l76_76870


namespace expression_I_evaluation_expression_II_evaluation_l76_76251

theorem expression_I_evaluation :
  ( (3 / 2) ^ (-2: ℤ) - (49 / 81) ^ (0.5: ℝ) + (0.008: ℝ) ^ (-2 / 3: ℝ) * (2 / 25) ) = (5 / 3) := 
by
  sorry

theorem expression_II_evaluation :
  ( (Real.logb 2 2) ^ 2 + (Real.logb 10 20) * (Real.logb 10 5) ) = (17 / 9) := 
by
  sorry

end expression_I_evaluation_expression_II_evaluation_l76_76251


namespace math_problem_l76_76346

theorem math_problem
  (x : ℝ)
  (h : x + sqrt (x^2 + 2) + (1 / (x - sqrt (x^2 + 2))) = 15) :
  (x^2 + sqrt (x^4 + 2) + (1 / (x^2 + sqrt (x^4 + 2)))) = 47089 / 1800 := by
  sorry

end math_problem_l76_76346


namespace range_of_t_l76_76313

-- Define the complex number z and the condition given in the problem
def complex_condition (z : ℂ) : Prop := 
  4 * z^(-2011) - 3 * complex.I * z^(-2010) - 3 * complex.I * z^(-1) - 4 = 0

-- Define the target expression t
def t (z : ℂ) : ℂ :=
  complex.conj ((3 - 4 * complex.I) / z) + complex.conj ((3 + 4 * complex.I) * z)

-- Main theorem statement
theorem range_of_t (z : ℂ) (hz : complex_condition z) : 
  abs (complex.re (t z)) ≤ 10 :=
sorry

end range_of_t_l76_76313


namespace repeating_decimal_to_fraction_l76_76963

theorem repeating_decimal_to_fraction : 
  (x : ℝ) (h : x = 0.4 + 36 / (10^1 + 10^2 + 10^3 + ...)) : x = 24 / 55 :=
sorry

end repeating_decimal_to_fraction_l76_76963


namespace max_discount_rate_l76_76154

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76154


namespace height_of_cylinder_l76_76210

theorem height_of_cylinder (r_hemisphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) :
  r_hemisphere = 7 → r_cylinder = 3 → h_cylinder = 2 * Real.sqrt 10 :=
by
  intro r_hemisphere_eq r_cylinder_eq
  sorry

end height_of_cylinder_l76_76210


namespace m_range_l76_76282

variable (x m : ℝ)

def p : Prop := |1 - (x - 1) / 3| ≤ 2
def q : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem m_range (h : (¬ p) → (¬ q ∧ q ≠ false)) : 0 < m ∧ m ≤ 3 := by
  sorry

end m_range_l76_76282


namespace problem_part1_problem_part2_l76_76321

open Real

noncomputable def f (x : ℝ) : ℝ := exp(x) - x

theorem problem_part1 :
  (∃ a : ℝ, ∀ x : ℝ, f x = exp(x) + a * x ∧ (f'(0) = 1 → a = -1)) ∧
  (∀ x : ℝ, x > 0 → f'(x) > 0) ∧
  (∀ x : ℝ, x < 0 → f'(x) < 0) ∧
  (∀ x : ℝ, f(x) is monotonic_increasing on (0, +∞) ∧ f(x) is monotonic_decreasing on (-∞, 0))
 :=
by
  sorry

theorem problem_part2 (b c : ℝ) (hb : b > 0)
  (h : ∀ x : ℝ, f(x) ≥ b * (b - 1) * x + c) :
  b^2 * c ≤ (1 / 3) * exp(2) :=
by
  sorry

end problem_part1_problem_part2_l76_76321


namespace tangent_lines_through_M_l76_76256

-- Define the point M and the circle
def point_M : ℝ × ℝ := (3, 1)
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the lines we need to prove
def line1 (x y : ℝ) : Prop := x = 3
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 5 = 0

-- Statement of the proof problem
theorem tangent_lines_through_M : 
  (∀ (x y : ℝ), circle x y → line1 x y ∨ line2 x y) :=
sorry

end tangent_lines_through_M_l76_76256


namespace Sams_age_is_10_l76_76021

theorem Sams_age_is_10 (S M : ℕ) (h1 : M = S + 7) (h2 : S + M = 27) : S = 10 := 
by
  sorry

end Sams_age_is_10_l76_76021


namespace average_score_is_67_l76_76450

def scores : List ℕ := [55, 67, 76, 82, 55]
def num_of_subjects : ℕ := List.length scores
def total_score : ℕ := List.sum scores
def average_score : ℕ := total_score / num_of_subjects

theorem average_score_is_67 : average_score = 67 := by
  sorry

end average_score_is_67_l76_76450


namespace valentines_left_l76_76878

theorem valentines_left (initial_valentines given_away : ℕ) (h_initial : initial_valentines = 30) (h_given : given_away = 8) :
  initial_valentines - given_away = 22 :=
by {
  sorry
}

end valentines_left_l76_76878


namespace opposite_of_neg_two_l76_76542

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76542


namespace meaningful_iff_x_ne_1_l76_76755

theorem meaningful_iff_x_ne_1 (x : ℝ) : (x - 1) ≠ 0 ↔ (x ≠ 1) :=
by 
  sorry

end meaningful_iff_x_ne_1_l76_76755


namespace no_values_satisfy_eqn_l76_76492

theorem no_values_satisfy_eqn : ∀ x : ℝ, 
  (x ≠ 0 ∧ x ≠ 5) → ¬ (3 * x^2 - 15 * x) / (x^2 - 5 * x) = x - 2 := 
by
  intro x
  intro h
  have h₁ : x * (x - 5) ≠ 0 := by
    cases h
    intro hx
    cases hx 
    case inl => contradiction
    case inr => contradiction
  have h₂ : x ≠ 0 := h.1
  have h₃ : x ≠ 5 := h.2
  sorry

end no_values_satisfy_eqn_l76_76492


namespace find_n_l76_76261

theorem find_n (n : ℤ) (h : n * 1296 / 432 = 36) : n = 12 :=
sorry

end find_n_l76_76261


namespace octagon_perimeter_correct_l76_76973

def octagon_perimeter (n : ℕ) (side_length : ℝ) : ℝ :=
  n * side_length

theorem octagon_perimeter_correct :
  octagon_perimeter 8 3 = 24 :=
by
  sorry

end octagon_perimeter_correct_l76_76973


namespace minimum_soldiers_to_add_l76_76787

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76787


namespace largest_lcm_value_is_60_l76_76971

-- Define the conditions
def lcm_values : List ℕ := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 9, Nat.lcm 15 12, Nat.lcm 15 10, Nat.lcm 15 15]

-- State the proof problem
theorem largest_lcm_value_is_60 : lcm_values.maximum = some 60 :=
by
  repeat { sorry }

end largest_lcm_value_is_60_l76_76971


namespace smallest_prime_dividing_4_pow_11_plus_6_pow_13_l76_76974

-- Definition of the problem
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem smallest_prime_dividing_4_pow_11_plus_6_pow_13 :
  ∃ p : ℕ, is_prime p ∧ p ∣ (4^11 + 6^13) ∧ ∀ q : ℕ, is_prime q ∧ q ∣ (4^11 + 6^13) → p ≤ q :=
by {
  sorry
}

end smallest_prime_dividing_4_pow_11_plus_6_pow_13_l76_76974


namespace find_A_l76_76268

theorem find_A (A : ℝ) (h : 4 * A + 5 = 33) : A = 7 :=
  sorry

end find_A_l76_76268


namespace new_person_weight_l76_76987

theorem new_person_weight 
    (W : ℝ) -- total weight of original 8 people
    (x : ℝ) -- weight of the new person
    (increase_by : ℝ) -- average weight increases by 2.5 kg
    (replaced_weight : ℝ) -- weight of the replaced person (55 kg)
    (h1 : increase_by = 2.5)
    (h2 : replaced_weight = 55)
    (h3 : x = replaced_weight + (8 * increase_by)) : x = 75 := 
by
  sorry

end new_person_weight_l76_76987


namespace max_area_of_equilateral_triangle_in_rectangle_l76_76937

theorem max_area_of_equilateral_triangle_in_rectangle :
  ∃ a b c : ℕ, a * Real.sqrt b - c = maximum_triangle_area 12 13 ∧ b = 3 ∧ Nat.prime_square_free b ∧ a + b + c = 1251 := by
  sorry

noncomputable def maximum_triangle_area (length width : ℕ) : ℝ :=
  -- this definition can be properly elaborated based on geometric constraints
  312 * Real.sqrt 3 - 936

end max_area_of_equilateral_triangle_in_rectangle_l76_76937


namespace pentagon_parallelogram_l76_76060

structure Pentagon (P : Type) :=
(F1 F2 F3 F4 F5 : P)
(A : P)
(convex : True)

variable {P : Type} [AffinePlane P] (pentagon : Pentagon P)

def Midpoint (p q : P) : P := sorry
def IsParallelogram (p q r s : P) : Prop := sorry

theorem pentagon_parallelogram (h : IsParallelogram P pentagon.F2 pentagon.F3 pentagon.F4) :
  IsParallelogram P (Midpoint pentagon.A pentagon.F1) pentagon.F5 pentagon.A (Midpoint pentagon.A pentagon.F1) :=
sorry

end pentagon_parallelogram_l76_76060


namespace opposite_of_neg_2_is_2_l76_76493

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76493


namespace geometric_sequence_first_term_l76_76674

variable (a y z : ℕ)
variable (r : ℕ)
variable (h₁ : 16 = a * r^2)
variable (h₂ : 128 = a * r^4)

theorem geometric_sequence_first_term 
  (h₃ : r = 2) : a = 4 :=
by
  sorry

end geometric_sequence_first_term_l76_76674


namespace total_cost_proof_l76_76092

def sandwich_cost : ℝ := 2.49
def soda_cost : ℝ := 1.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 12.46

theorem total_cost_proof : (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = total_cost :=
by
  sorry

end total_cost_proof_l76_76092


namespace toy_ratio_problem_l76_76437

variable (total_toys : ℕ) (elder_son_toys : ℕ) (younger_son_toys : ℕ) (ratio : ℕ)

-- Conditions
def condition_total_toys : Prop := total_toys = 240
def condition_elder_son_toys : Prop := elder_son_toys = 60
def calculate_younger_son_toys : Prop := younger_son_toys = total_toys - elder_son_toys
def calculate_ratio : Prop := ratio = younger_son_toys / elder_son_toys

-- Question to prove
def problem_statement : Prop := calculate_ratio ∧ ratio = 3

theorem toy_ratio_problem : 
  condition_total_toys → 
  condition_elder_son_toys → 
  calculate_younger_son_toys → 
  problem_statement :=
sorry

end toy_ratio_problem_l76_76437


namespace probability_without_replacement_expectation_and_variance_l76_76612

-- Define the bag of balls
def bag : List ℕ := [1, 1, 2, 2, 3]

-- Define probability of drawing a ball numbered 3 without replacement
def prob_3_without_replacement (draws : Finset (Fin 5)) : ℚ :=
  (Finset.card (draws.filter (λ x, bag[x] = 3))) / (Finset.card draws)

-- Define combination function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Statement (1): Prove the probability without replacement
 theorem probability_without_replacement :
   (comb 4 2) / (comb 5 3) = (3 / 5) :=
 sorry

-- Define the binomial distribution parameters
def n : ℕ := 10
def p : ℚ := 3 / 5

-- Define the expectation of a binomial random variable
def expectation := n * p

-- Define the variance of a binomial random variable
def variance := n * p * (1 - p)

-- Statement (2): Prove the expectation and variance with replacement
 theorem expectation_and_variance :
   expectation = 6 ∧ variance = (12 / 5) :=
 sorry

end probability_without_replacement_expectation_and_variance_l76_76612


namespace length_of_PR_l76_76396

theorem length_of_PR {P Q R : Type} [Inhabited P] [Inhabited Q] [Inhabited R] 
  (h1 : ∠PQR = ∠PRQ) (h2 : QR = 8) (h3 : area (triangle P Q R) = 24)
  (h4 : height_from P (line QR) = midpoint Q R) : 
  PR = 2 * (sqrt 13) :=
by
  sorry

end length_of_PR_l76_76396


namespace trains_meeting_time_l76_76956

-- Definitions derived from conditions
def length_Train_A : ℝ := 120
def speed_Train_A_kmh : ℝ := 90
def speed_Train_A_ms : ℝ := speed_Train_A_kmh * 1000 / 3600

def length_Train_B : ℝ := 150
def speed_Train_B_kmh : ℝ := 72
def speed_Train_B_ms : ℝ := speed_Train_B_kmh * 1000 / 3600

def length_platform : ℝ := 180

-- Total relative speed when the two trains are approaching each other
def relative_speed : ℝ := speed_Train_A_ms + speed_Train_B_ms

-- Total distance to be covered by both trains to meet
def total_distance : ℝ := length_Train_A + length_Train_B + length_platform

-- Proof statement
theorem trains_meeting_time : (total_distance / relative_speed) = 10 := sorry

end trains_meeting_time_l76_76956


namespace minimum_soldiers_to_add_l76_76784

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76784


namespace initial_amounts_l76_76947

theorem initial_amounts (x y z : ℕ) (h1 : x + y + z = 24)
  (h2 : z = 24 - x - y)
  (h3 : x - (y + z) = 8)
  (h4 : y - (x + z) = 12) :
  x = 13 ∧ y = 7 ∧ z = 4 :=
by
  sorry

end initial_amounts_l76_76947


namespace find_setC_l76_76299

def setA := {x : ℝ | x^2 - 3 * x + 2 = 0}
def setB (a : ℝ) := {x : ℝ | a * x - 2 = 0}
def union_condition (a : ℝ) : Prop := (setA ∪ setB a) = setA
def setC := {a : ℝ | union_condition a}

theorem find_setC : setC = {0, 1, 2} :=
by
  sorry

end find_setC_l76_76299


namespace find_interest_rate_l76_76982

variable (P A1 A2 r : ℝ) (t1 t2 : ℕ)

-- Given conditions
def condition1 : Prop := A1 = P * (1 + r) ^ t1
def condition2 : Prop := A2 = P * (1 + r) ^ t2

-- Main theorem stating the interest rate is 0.2 (20%)
theorem find_interest_rate (h1 : condition1 P 3000 3 r t1)
                          (h2 : condition2 P 3600 4 r t2) :
    r = 0.2 := 
by
  sorry

end find_interest_rate_l76_76982


namespace roots_polynomial_l76_76865

theorem roots_polynomial (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + c * a = 19) (h3 : a * b * c = 8) : 
  (1 + a) * (1 + b) * (1 + c) = 46 :=
by
  sorry

end roots_polynomial_l76_76865


namespace number_of_ways_l76_76386

constant Grid : Type
constant cell : Type
constant cells_2_or_5 : Finset cell
constant cells_3_or_4 : Finset cell
constant cells_1_or_6_one : Finset cell
constant cells_1_or_6_two : Finset cell

-- The problem grid setup
axiom six_by_six_grid (grid : Grid) : 
  ∀ r c, 0 ≤ r < 6 → 0 ≤ c < 6 → ∃! n, 1 ≤ n ∧ n ≤ 6 ∧ ∃ filled, filled n r c

-- The constraints for cells to be filled
axiom constraints (grid : Grid) : 
  Finset.card cells_2_or_5 = 4 →
  Finset.card cells_3_or_4 = 4 →
  Finset.card cells_1_or_6_one = 4 →
  Finset.card cells_1_or_6_two = 4 →

-- The actual cells can only be filled by the specific conditions
  (∀ cell ∈ cells_2_or_5, filled_with_two_or_five grid cell) →
  (∀ cell ∈ cells_3_or_4, filled_with_three_or_four grid cell) →
  (∀ cell ∈ cells_1_or_6_one, filled_with_one_or_six grid cell) →
  (∀ cell ∈ cells_1_or_6_two, filled_with_one_or_six grid cell)

-- The final proof statement that the number of fills equals 16
theorem number_of_ways (grid : Grid) : number_of_ways_to_fill (grid) = 16 := 
sorry

end number_of_ways_l76_76386


namespace min_distance_curve_line_l76_76872

-- Definitions
def curve (x : ℝ) : ℝ := x^2 - log x

noncomputable def tangent_line_slope (x : ℝ) : ℝ := 2 * x - 1 / x

def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Problem statement
theorem min_distance_curve_line : 
    let P := (1, curve 1) in
    let Q := (1, 1) in
    ∃ P Q, curve P.1 = P.2 ∧ line Q.1 Q.2 ∧ dist P Q = sqrt 2 :=
by sorry

end min_distance_curve_line_l76_76872


namespace gumball_total_l76_76202

variable R B G : ℕ

axiom condition1 : B = R / 2
axiom condition2 : G = 4 * B
axiom condition3 : R = 16

theorem gumball_total : R + B + G = 56 := by
  sorry

end gumball_total_l76_76202


namespace num_intersecting_chords_on_circle_l76_76027

theorem num_intersecting_chords_on_circle (points : Fin 20 → Prop) : 
  ∃ num_chords : ℕ, num_chords = 156180 :=
by
  sorry

end num_intersecting_chords_on_circle_l76_76027


namespace find_point_M_l76_76294

def point := ℝ × ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2

def A : point := (3, 2, 0)
def B : point := (2, -1, 2)

noncomputable def findM : point :=
  let Mx := 2 in (Mx, 0, 0)

theorem find_point_M :
  ∃ M : point, M = (findM.1, 0, 0) ∧ dist M A = dist M B := by
  let M := findM
  use M
  split
  -- M is on the x-axis and has the coordinates found
  { rw findM, simp, }
  -- Distances from M to A and B are equal
  { sorry, }

end find_point_M_l76_76294


namespace minimum_soldiers_to_add_l76_76778

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76778


namespace prob_ξ_eq_2_prob_ξ_ge_2_l76_76721

-- Probability of generating a successful trial
def probSuccess : ℚ := 1 / 3

-- Independent trials
def independentTrials (n : ℕ) : Prop := True -- Assuming independence is defined elsewhere or naturally holds

-- Number of trials
def numTrials : ℕ := 4

-- Absolute value of the difference between the number of successes and failures
def ξ (successes failures : ℕ) : ℕ := abs (successes - failures)

-- Calculate the probability of exactly k successes in n trials
noncomputable def binomialProb (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Problem 1: Probability that ξ = 2
theorem prob_ξ_eq_2 : 
  (binomialProb numTrials 3 probSuccess) + (binomialProb numTrials 1 probSuccess) = 40 / 81 := 
sorry

-- Problem 2: Probability that ξ >= 2
theorem prob_ξ_ge_2 : 
  1 - (binomialProb numTrials 2 probSuccess) = 57 / 81 := 
sorry

end prob_ξ_eq_2_prob_ξ_ge_2_l76_76721


namespace find_positive_number_l76_76566

theorem find_positive_number (n : ℕ) (h : n^2 + n = 156) : n = 12 :=
begin
  sorry
end

end find_positive_number_l76_76566


namespace opposite_of_neg_two_l76_76511

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76511


namespace maximum_discount_rate_l76_76174

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76174


namespace minimum_soldiers_to_add_l76_76796

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76796


namespace opposite_of_neg_two_l76_76543

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76543


namespace sum_of_edge_lengths_at_least_3d_l76_76889

-- Condition definitions
variable (P : Type) [polyhedron P]

-- Points A and B with distance d between them
variable (A B : P) (d : ℝ) 
variable (vertices : set P) [finite vertices]
variable (V_A V_B : A ∈ vertices ∧ B ∈ vertices)
variable (edges : set (P × P))
variable (sum_edge_lengths : ℝ)
variable (dist_eq_d : dist A B = d)

-- Definition of polyhedron edges
def sum_of_edge_lengths (edges : set (P × P)) : ℝ :=
  ∑ edge in edges, dist edge.1 edge.2

-- The theorem to prove
theorem sum_of_edge_lengths_at_least_3d :
  sum_of_edge_lengths edges ≥ 3 * d := 
sorry

end sum_of_edge_lengths_at_least_3d_l76_76889


namespace total_digits_l76_76909

theorem total_digits (n S S6 S4 : ℕ) 
  (h1 : S = 80 * n)
  (h2 : S6 = 6 * 58)
  (h3 : S4 = 4 * 113)
  (h4 : S = S6 + S4) : 
  n = 10 :=
by 
  sorry

end total_digits_l76_76909


namespace max_discount_rate_l76_76168

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76168


namespace fraction_subtraction_simplification_l76_76230

/-- Given that 57 equals 19 times 3, we want to prove that (8/19) - (5/57) equals 1/3. -/
theorem fraction_subtraction_simplification :
  8 / 19 - 5 / 57 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l76_76230


namespace opposite_of_neg2_l76_76505

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76505


namespace find_angle_BXY_l76_76833

variables (AB CD : line) (AXE CYX BXY : angle)

-- Condition definitions
def parallel_AB_CD : Prop := AB ∥ CD
def angle_relationship1 : Prop := AXE = 4 * CYX - 90
def angle_relationship2 : Prop := AXE = CYX

-- Proof statement to be verified
theorem find_angle_BXY (h1 : parallel_AB_CD AB CD) (h2 : angle_relationship1 AXE CYX) (h3 : angle_relationship2 AXE CYX) :
  BXY = 30 :=
sorry

end find_angle_BXY_l76_76833


namespace max_discount_rate_l76_76132

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76132


namespace smallest_positive_period_of_f_max_min_values_f_sum_of_roots_l76_76696

def vector_ac (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2) + Real.sin (x / 2), Real.sin (x / 2))
def vector_bc (x : ℝ) : ℝ × ℝ := (Real.sin (x / 2) - Real.cos (x / 2), 2 * Real.cos (x / 2))
def f (x : ℝ) : ℝ := (vector_ac x).1 * (vector_bc x).1 + (vector_ac x).2 * (vector_bc x).2

theorem smallest_positive_period_of_f : ∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x := sorry

theorem max_min_values_f :
  (∃ max_val, ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val) ∧
  (∃ min_val, ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), min_val ≤ f x) :=
sorry

theorem sum_of_roots :
  ∀ x1 x2 ∈ Set.Ioo (Real.pi : ℝ) (3 * Real.pi),
    f x1 = Real.sqrt 6 / 2 → f x2 = Real.sqrt 6 / 2 → x1 + x2 = 11 * Real.pi / 2 :=
sorry

end smallest_positive_period_of_f_max_min_values_f_sum_of_roots_l76_76696


namespace triangle_ABC_is_isosceles_right_find_k_l76_76326

noncomputable def A : ℝ × ℝ × ℝ := (-1, -1, 2)
noncomputable def B : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def C : ℝ × ℝ × ℝ := (-2, 3, 1)

noncomputable def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

noncomputable def BA := vec_sub B A -- (1, 2, -2)
noncomputable def BC := vec_sub C B -- (-2, 2, 1)
noncomputable def AC := vec_sub C A -- (-1, 4, -1)

-- Prove that ∆ABC is an isosceles right triangle
theorem triangle_ABC_is_isosceles_right : 
  let a := BA
  let b := BC
  let c := AC
  (|∥a∥^2 + |∥b∥^2 = |∥c∥^2) → 
  (∥a∥ = ∥b∥) := sorry

-- Prove the value of k such that (-2BA + kBC) ∥ AC
theorem find_k (k : ℝ) :
  let a := BA
  let b := BC
  let c := AC
  (-2 * a + k * b) ∥ c → k = 2 := sorry

end triangle_ABC_is_isosceles_right_find_k_l76_76326


namespace length_of_second_train_l76_76955

-- Defining the parameters
def speed_train1_kmph : ℝ := 60
def speed_train2_kmph : ℝ := 40
def length_train1_m : ℝ := 260
def crossing_time_s : ℝ := 16.918646508279338

-- Converting speeds to m/s
def speed_train1_mps : ℝ := speed_train1_kmph * 1000 / 3600
def speed_train2_mps : ℝ := speed_train2_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Total distance covered when crossing each other
def total_distance_m : ℝ := relative_speed_mps * crossing_time_s

-- Lean 4 statement
theorem length_of_second_train : 
  ∃ L : ℝ, (length_train1_m + L = total_distance_m ∧ L = 210) := 
by {
  -- Using placeholder for the proof
  sorry
}

end length_of_second_train_l76_76955


namespace tan_alpha_minus_beta_l76_76362

theorem tan_alpha_minus_beta (α β : ℝ) (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_l76_76362


namespace minimum_soldiers_to_add_l76_76783

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76783


namespace zero_condition_l76_76724

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log2 x

variables (a b c d : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < b ∧ b < c
axiom h2 : f a * f b * f c < 0
axiom h3 : f d = 0

-- Target statement
theorem zero_condition (ha : 0 < a) (hab : a < b) (hbc : b < c)
    (h_prod : f a * f b * f c < 0) (h_zero : f d = 0) : 
    d > a ∧ d > b ∧ d < c := 
    sorry

end zero_condition_l76_76724


namespace max_discount_rate_l76_76163

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76163


namespace cream_ratio_l76_76882

variable (servings : ℕ) (fat_per_serving : ℕ) (fat_per_cup : ℕ)
variable (h_servings : servings = 4) (h_fat_per_serving : fat_per_serving = 11) (h_fat_per_cup : fat_per_cup = 88)

theorem cream_ratio (total_fat : ℕ) (h_total_fat : total_fat = fat_per_serving * servings) :
  (total_fat : ℚ) / fat_per_cup = 1 / 2 :=
by
  sorry

end cream_ratio_l76_76882


namespace triangle_area_l76_76820

noncomputable def area_of_triangle (a b c : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * a * c * Real.sin B

theorem triangle_area
  (a b c : ℝ) (B : ℝ)
  (h_b : b = 7)
  (h_c : c = 5)
  (h_B : B = (2 * Real.pi) / 3)
  (h_a : a = 3) :
  area_of_triangle a b c B = (15 * Real.sqrt 3) / 4 :=
by
  rw [area_of_triangle, h_b, h_c, h_B, h_a]
  sorry

end triangle_area_l76_76820


namespace proof_main_l76_76620

def final_position (distances : List Int) : Int :=
  distances.sum

def total_distance (distances : List Int) : Nat :=
  distances.map (Int.natAbs).sum

def fuel_consumption (distances : List Int) (rate : Float) : Float :=
  rate * (Float.ofNat (total_distance distances) + Float.ofInt (Int.natAbs (final_position distances)))

theorem proof_main : 
  (final_position [2, -8, 5, -7, -8, 6, -7, 13] = -4) ∧ 
  (total_distance [2, -8, 5, -7, -8, 6, -7, 13] = 56) ∧
  (fuel_consumption [2, -8, 5, -7, -8, 6, -7, 13] 0.3 = 18) :=
by
  sorry

end proof_main_l76_76620


namespace cubic_eq_root_nature_l76_76653

-- Definitions based on the problem statement
def cubic_eq (x : ℝ) : Prop := x^3 + 3 * x^2 - 4 * x - 12 = 0

-- The main theorem statement
theorem cubic_eq_root_nature :
  (∃ p n₁ n₂ : ℝ, cubic_eq p ∧ cubic_eq n₁ ∧ cubic_eq n₂ ∧ p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧ p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂) :=
sorry

end cubic_eq_root_nature_l76_76653


namespace minimum_soldiers_to_add_l76_76794

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76794


namespace count_non_representable_as_diff_of_squares_l76_76333

theorem count_non_representable_as_diff_of_squares :
  let count := (Finset.filter (fun n => ∃ k, n = 4 * k + 2 ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card in
  count = 250 :=
by
  sorry

end count_non_representable_as_diff_of_squares_l76_76333


namespace vector_perpendicular_to_a_l76_76631

theorem vector_perpendicular_to_a :
  let a := (4, 3)
  let b := (3, -4)
  a.1 * b.1 + a.2 * b.2 = 0 := by
  let a := (4, 3)
  let b := (3, -4)
  sorry

end vector_perpendicular_to_a_l76_76631


namespace solve_for_x_l76_76040

theorem solve_for_x (x : ℝ) (h₁ : x ≠ -3) :
  (7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3) ↔ x = 1 ∨ x = -1 := 
sorry

end solve_for_x_l76_76040


namespace ferris_wheel_cost_l76_76106

theorem ferris_wheel_cost (roller_coaster_cost log_ride_cost zach_initial_tickets zach_additional_tickets total_tickets ferris_wheel_cost : ℕ) 
  (h1 : roller_coaster_cost = 7)
  (h2 : log_ride_cost = 1)
  (h3 : zach_initial_tickets = 1)
  (h4 : zach_additional_tickets = 9)
  (h5 : total_tickets = zach_initial_tickets + zach_additional_tickets)
  (h6 : total_tickets - (roller_coaster_cost + log_ride_cost) = ferris_wheel_cost) :
  ferris_wheel_cost = 2 := 
by
  sorry

end ferris_wheel_cost_l76_76106


namespace talia_total_distance_l76_76468

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end talia_total_distance_l76_76468


namespace shaded_non_shaded_ratio_l76_76075

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

noncomputable def ratio_shaded_to_non_shaded (s : ℝ) : ℝ :=
  let area_ABC := area_equilateral_triangle s
  let area_DEF := area_equilateral_triangle (s / 2)
  let area_shaded_triangle := area_equilateral_triangle (s / 4)
  let total_shaded_area := 3 * area_shaded_triangle
  let non_shaded_area := area_ABC - total_shaded_area
  total_shaded_area / non_shaded_area

theorem shaded_non_shaded_ratio (s : ℝ) (s_pos : 0 < s) :
  ratio_shaded_to_non_shaded s = 3 / 13 :=
by
  sorry

end shaded_non_shaded_ratio_l76_76075


namespace monotonic_decreasing_interval_l76_76490

noncomputable def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval :
  ∃ a b : ℝ, a < b ∧ (∀ x : ℝ, a < x ∧ x < b → f' x < 0) ∧ a = -1 ∧ b = 11 :=
sorry

end monotonic_decreasing_interval_l76_76490


namespace Rob_has_three_dimes_l76_76036

theorem Rob_has_three_dimes (quarters dimes nickels pennies : ℕ) 
                            (val_quarters val_nickels val_pennies : ℚ)
                            (total_amount : ℚ) :
  quarters = 7 →
  nickels = 5 →
  pennies = 12 →
  val_quarters = 0.25 →
  val_nickels = 0.05 →
  val_pennies = 0.01 →
  total_amount = 2.42 →
  (7 * 0.25 + 5 * 0.05 + 12 * 0.01 + dimes * 0.10 = total_amount) →
  dimes = 3 :=
by sorry

end Rob_has_three_dimes_l76_76036


namespace solution_set_of_inequality_l76_76938

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l76_76938


namespace opposite_of_neg_two_is_two_l76_76537

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76537


namespace points_deducted_for_incorrect_answer_is_5_l76_76392

-- Define the constants and variables used in the problem
def total_questions : ℕ := 30
def points_per_correct_answer : ℕ := 20
def correct_answers : ℕ := 19
def incorrect_answers : ℕ := total_questions - correct_answers
def final_score : ℕ := 325

-- Define a function that models the total score calculation
def calculate_final_score (points_deducted_per_incorrect : ℕ) : ℕ :=
  (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect)

-- The theorem that states the problem and expected solution
theorem points_deducted_for_incorrect_answer_is_5 :
  ∃ (x : ℕ), calculate_final_score x = final_score ∧ x = 5 :=
by
  sorry

end points_deducted_for_incorrect_answer_is_5_l76_76392


namespace math_scores_120_or_higher_l76_76199

-- Define the normal distribution with mean 110 and variance 100
def examScoresDistribution : ProbabilityMassFunction ℝ :=
  ProbabilityMassFunction.normal (110 : ℝ) (10 : ℝ)

-- Given condition
axiom condition1 : ∀ (x : ℝ), 100 ≤ x ∧ x ≤ 110 → examScoresDistribution.prob_mass x = 0.34

noncomputable def numberOfStudents (total_students: ℕ) : ℕ :=
  let P_x_ge_120 := (1 - 0.34) / 2
  P_x_ge_120 * total_students

-- We need to show that if the conditions hold, then the number of students scoring 120 or higher is 8
theorem math_scores_120_or_higher :
  let total_students := 50 in
  numberOfStudents total_students = 8 :=
by sorry

end math_scores_120_or_higher_l76_76199


namespace maximum_discount_rate_l76_76176

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76176


namespace extraordinary_stack_card_count_l76_76625

theorem extraordinary_stack_card_count :
  ∃ (n : ℕ), 2 * n = 198 ∧ 
    (∀ k : ℕ, k < n → (2 * k + 1 = 57 → k = 28)) ∧
    (∀ j : ℕ, n ≤ j < 2 * n → (2 * (j - n) + 2 = 200 → j = 199)) :=
begin
  -- give a proof here
  sorry
end

end extraordinary_stack_card_count_l76_76625


namespace minimum_soldiers_to_add_l76_76803

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76803


namespace total_outcomes_l76_76999

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of events
def num_events : ℕ := 3

-- Theorem statement: asserting the total number of different outcomes
theorem total_outcomes : num_students ^ num_events = 125 :=
by
  sorry

end total_outcomes_l76_76999


namespace find_number_l76_76581

theorem find_number (x : ℝ) (h : 0.62 * x - 50 = 43) : x = 150 :=
sorry

end find_number_l76_76581


namespace median_of_set_is_89_5_l76_76489

noncomputable def set_nums : Set ℝ := {92, 90, 86, 89, 91}
noncomputable def y : ℝ := 89.5 * 6 - (92 + 90 + 86 + 89 + 91)

theorem median_of_set_is_89_5 : 
  let full_set := set_nums ∪ {y}
  let sorted_list := List.sort (Set.toList full_set)
  let median := (sorted_list.nth 2 + sorted_list.nth 3) / 2
  median = 89.5 := 
by
  sorry

end median_of_set_is_89_5_l76_76489


namespace max_discount_rate_l76_76196

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76196


namespace maximum_discount_rate_l76_76175

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76175


namespace max_discount_rate_l76_76150

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76150


namespace opposite_of_neg_two_is_two_l76_76533

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76533


namespace f_at_1_eq_25_l76_76813

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_at_1_eq_25 {m : ℝ} (h1 : ∀ x ≥ -2, deriv (f x m) x ≥ 0)
  (h2 : ∀ x ≤ -2, deriv (f x m) x ≤ 0) (h3 : deriv (f (-2 : ℝ) m) (-2) = 0)
  (h4 : deriv (f x m) x = 8 * x - m) : f 1 (-16) = 25 := by
  sorry

end f_at_1_eq_25_l76_76813


namespace number_of_common_tangents_l76_76929

noncomputable def center (a b c: ℝ) : ℝ × ℝ := (-a/2, -b/2)
noncomputable def radius (a b c: ℝ) : ℝ := real.sqrt (a^2 + b^2 - 4 * c) / 2
noncomputable def distance (p1 p2: ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (a1 b1 c1 a2 b2 c2 : ℝ) (r1 r2 d: ℝ)
  (h1: a1=4) (h2: b1=4) (h3: c1=4)
  (h4: a2=-4) (h5: b2=-2) (h6: c2=-4) 
  (center1: center a1 b1 c1 = (-2, -2))
  (center2: center a2 b2 c2 = (2, 1))
  (r1: radius a1 b1 c1 = 2)
  (r2: radius a2 b2 c2 = 3)
  (dist_eq: distance (center a1 b1 c1) (center a2 b2 c2) = 5)
  : 3 := 
sorry

end number_of_common_tangents_l76_76929


namespace exists_irrational_in_interval_l76_76412

noncomputable theory

open Real

/-- Definitions -/
def is_in_set_M (M : set ℝ) (x : ℝ) : Prop := x ∈ M
def is_in_set_M_star (M : set ℝ) (x : ℝ) : Prop := x ∈ M ∧ irrational x

/-- Main statement -/
theorem exists_irrational_in_interval
  (a b : ℚ) 
  (M : set ℝ)
  (h1 : 0 < a ∧ a < b)
  (h2 : is_in_set_M M a)
  (h3 : is_in_set_M M b)
  (h4 : ∀ x y, is_in_set_M M x → is_in_set_M M y → is_in_set_M M (sqrt (x * y))) :
  ∀ c d : ℝ, a < c → c < d → d < b → ∃ m ∈ M, irrational m ∧ c < m ∧ m < d :=
by 
  intros c d h_ac h_cd h_db
  sorry

end exists_irrational_in_interval_l76_76412


namespace four_op_two_l76_76004

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end four_op_two_l76_76004


namespace max_discount_rate_l76_76156

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76156


namespace smallest_n_l76_76597

theorem smallest_n (n : ℕ) (h : 10 - n ≥ 0) : 
  (9 / 10) * (8 / 9) * (7 / 8) * (6 / 7) * (5 / 6) * (4 / 5) < 0.5 → n = 6 :=
by
  sorry

end smallest_n_l76_76597


namespace opposite_of_neg_2_is_2_l76_76500

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76500


namespace find_original_fraction_l76_76985

theorem find_original_fraction (x y : ℕ) (hxy : Nat.gcd x y = 1) (h : 12 * x = y) :
  let original_fraction := (x, y) in 
  original_fraction = (1, 12) :=
by
  -- The proof will go here
  sorry

end find_original_fraction_l76_76985


namespace sum_series_eq_one_l76_76259

theorem sum_series_eq_one :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2*n + 1))) = 1 :=
by
  have h1 : ∀ n : ℕ, 1 + 3^n + 3^(n + 1) + 3^(2*n + 1) = (1 + 3^n) * (1 + 3^(n + 1)) := sorry
  have h2 : ∀ n : ℕ, (3^n) / ((1 + 3^n) * (1 + 3^(n + 1))) = (1 / (1 + 3^n)) - (1 / (1 + 3^(n + 1))) := sorry
  have h3 : (∑' n : ℕ, (3^n) / ((1 + 3^n) * (1 + 3^(n + 1)))) = (3^0) / ((1 + 3^0) * (1 + 3^(0 + 1))) := 
    calc
      (∑' n : ℕ, (3^n) / ((1 + 3^n) * (1 + 3^(n + 1)))) = (3^0) / (1 + 3^0) - lim (∑ m : ℕ, 1 / (1 + 3^(m + 1))) := sorry
  have h4 : lim (∑ m : ℕ, 1 / (1 + 3^(m + 1))) = 0 := sorry
  exact h3 - h4

end sum_series_eq_one_l76_76259


namespace van_helsing_removed_percentage_l76_76080

theorem van_helsing_removed_percentage :
  ∀ (V W : ℕ), 
  (5 * V / 2 + 10 * 8 = 105) →
  (W = 4 * V) →
  8 / W * 100 = 20 := 
by
  sorry

end van_helsing_removed_percentage_l76_76080


namespace problem_statement_l76_76695

theorem problem_statement (a b : ℝ) (h1 : 1 / a + 1 / b = Real.sqrt 5) (h2 : a ≠ b) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 :=
by
  sorry

end problem_statement_l76_76695


namespace opposite_of_neg2_l76_76502

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76502


namespace real_number_condition_imaginary_number_condition_pure_imaginary_condition_l76_76275

def isReal (z : ℂ) : Prop :=
  z.im = 0

def isImaginary (z : ℂ) : Prop :=
  z.re = 0

def isPureImaginary (z : ℂ) : Prop :=
  ¬ isReal z ∧ isImaginary z

def complex_number (m : ℝ) : ℂ :=
  ⟨(m^2 + m - 6) / m, m^2 - 2 * m⟩

theorem real_number_condition (m : ℝ) (h : m = 2) : isReal (complex_number m) :=
sorry

theorem imaginary_number_condition (m : ℝ) (h₁ : m ≠ 2) (h₂ : m ≠ 0) : ¬ isReal (complex_number m) :=
sorry

theorem pure_imaginary_condition (m : ℝ) (h : m = -3) : isPureImaginary (complex_number m) :=
sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_condition_l76_76275


namespace new_avg_is_6_8_l76_76603

noncomputable theory
open_locale classical

-- Define the original set S with 10 elements and given average
def avg_of_set (S : fin 10 → ℝ) : ℝ :=
  (∑ i, S i) / 10

-- Define the condition that the average of set S is 6.2
def avg6_2 (S : fin 10 → ℝ) : Prop :=
  avg_of_set S = 6.2

-- Define the new set S' after increasing one element by 6
def new_set (S : fin 10 → ℝ) (i : fin 10) : fin 10 → ℝ :=
  λ j, if j = i then S j + 6 else S j

-- Define the theorem that the new average is 6.8
theorem new_avg_is_6_8 (S : fin 10 → ℝ) (i : fin 10) (h : avg6_2 S) : avg_of_set (new_set S i) = 6.8 :=
sorry

end new_avg_is_6_8_l76_76603


namespace area_of_MBCN_l76_76397

def area_rectangle (length width : ℝ) : ℝ := length * width

def area_trapezoid (base1 base2 height : ℝ) : ℝ := 1 / 2 * (base1 + base2) * height

theorem area_of_MBCN 
    (ABCD_area : ℝ)
    (AB : ℝ)
    (BC : ℝ)
    (M : Point ℝ)
    (N : Point ℝ)
    (C : Point ℝ)
    (conditions : ABCD_area = 40 ∧ AB = 8 ∧ area_rectangle AB BC = ABCD_area ∧ AB + BC = 8 + 5)
    : area_trapezoid 2 4 5 = 15 :=
sorry

end area_of_MBCN_l76_76397


namespace solve_for_x_l76_76458

theorem solve_for_x (x : ℝ) (hx : x = Real.log ((Real.log 2) / (Real.log 3)) / Real.log (9 / 8)) : 3^(8^x) = 8^(3^x) :=
by
  sorry

end solve_for_x_l76_76458


namespace MN_eq_l76_76657

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}
def operation (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

theorem MN_eq : operation M N = {1, 4} :=
sorry

end MN_eq_l76_76657


namespace probability_Q_eq_i_l76_76068

noncomputable def vertices : set ℂ := 
  { complex.sqrt 2 * complex.I, -complex.sqrt 2 * complex.I, complex.sqrt 2, -complex.sqrt 2,
    (1 + complex.I) / complex.sqrt 8, (-1 + complex.I) / complex.sqrt 8, 
    (1 - complex.I) / complex.sqrt 8, (-1 - complex.I) / complex.sqrt 8 }

def selected_vertices : ℕ → ℂ
| k := if h : k < 16 then classical.some (classical.some_spec $ finset.exists_mem vertices.to_finset) else 0

def Q : ℂ := ∏ k in finset.range 16, selected_vertices k

theorem probability_Q_eq_i :
  ∃ c d q : ℕ, nat.prime q ∧ ¬ q ∣ c ∧ Q = complex.I →
  ∃ c d q : ℕ, nat.prime q ∧ ¬ q ∣ c ∧ (c + d + q = 6452) ∧ 
  (∃ (p : ℚ), p = 6435 / 32768) :=
begin
  sorry,
end

end probability_Q_eq_i_l76_76068


namespace smallest_faces_l76_76629

def tetrahedron_faces := 4
def quadrangular_pyramid_faces := 5
def triangular_prism_faces := 5
def triangular_pyramid_faces := 4

theorem smallest_faces (shapes : List Nat):
  shapes = [tetrahedron_faces, quadrangular_pyramid_faces, triangular_prism_faces, triangular_pyramid_faces] →
  List.minimum shapes = tetrahedron_faces :=
by
  intros h
  rw h
  simp [tetrahedron_faces, quadrangular_pyramid_faces, triangular_prism_faces, triangular_pyramid_faces]
  -- After these simplifications Lean can prove it, but we leave it as sorry for now
  sorry

end smallest_faces_l76_76629


namespace find_oranges_l76_76408

def A : ℕ := 3
def B : ℕ := 1

theorem find_oranges (O : ℕ) : A + B + O + (A + 4) + 10 * B + 2 * (A + 4) = 39 → O = 4 :=
by 
  intros h
  sorry

end find_oranges_l76_76408


namespace min_value_of_a2_plus_b2_l76_76661

theorem min_value_of_a2_plus_b2 
  (a b : ℝ) 
  (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4 := 
sorry

end min_value_of_a2_plus_b2_l76_76661


namespace tan_alpha_minus_beta_l76_76351

variable (α β : ℝ)

theorem tan_alpha_minus_beta
  (h : sin (α + β) + cos (α + β) = 2 * real.sqrt 2 * cos (α + π/4) * sin β) : 
  real.tan (α - β) = -1 :=
sorry

end tan_alpha_minus_beta_l76_76351


namespace smallest_n_mushrooms_l76_76114

theorem smallest_n_mushrooms (n : ℕ) (h_gatherers : n ≥ 1) 
  (h_total : ∀ (m : ℝ → ℝ), (∀ i : ℕ, 1 ≤ i ∧ i <= n → 1 ≤ m i) → (∑ i in finset.range n, m i) = 450)
  : n ≥ 30 :=
begin
  sorry,
end

end smallest_n_mushrooms_l76_76114


namespace max_discount_rate_l76_76169

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76169


namespace other_team_members_points_l76_76840

theorem other_team_members_points :
  ∃ (x : ℕ), ∃ (y : ℕ), (y ≤ 9 * 3) ∧ (x = y + 18 + x / 3 + x / 5) ∧ y = 24 :=
by
  sorry

end other_team_members_points_l76_76840


namespace Talia_total_distance_l76_76470

variable (Talia : Type)
variable (house park store : Talia)

-- Define the distances given in the conditions
variable (distance : Talia → Talia → ℕ)
variable (h2p : distance house park = 5)
variable (p2s : distance park store = 3)
variable (s2h : distance store house = 8)

-- Define the total distance function
def total_distance (t : Talia) : ℕ :=
  distance house park + distance park store + distance store house

-- Lean 4 theorem statement
theorem Talia_total_distance : total_distance Talia house park store distance = 16 :=
by
  simp [total_distance, h2p, p2s, s2h]
  sorry

end Talia_total_distance_l76_76470


namespace inheritance_amount_l76_76879

-- Definitions
def interest_eqns (x y : ℝ) : Prop := 0.06 * x + 0.08 * y = 860
def total_investment (x y : ℝ) : Prop := x + y = 12000

-- Problem Statement
theorem inheritance_amount :
  (∃ x y : ℝ, interest_eqns x y ∧ 
  ((x = 5000) ∨ (y = 5000))) → total_investment 5000 7000 :=
begin
  sorry
end

end inheritance_amount_l76_76879


namespace gcf_180_240_300_l76_76967

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end gcf_180_240_300_l76_76967


namespace sum_marked_sides_ge_one_l76_76624

theorem sum_marked_sides_ge_one (N : ℕ) (x : Fin N → ℝ) (y : Fin N → ℝ)
  (hx : ∀ n, x n ≤ 1) (hy : ∀ n, y n ≤ 1) (h_area : (Finset.univ.sum (λ n, x n * y n)) = 1) :
  (Finset.univ.sum (λ n, x n)) ≥ 1 :=
by
  sorry

end sum_marked_sides_ge_one_l76_76624


namespace intersecting_rectangles_shaded_area_l76_76074

theorem intersecting_rectangles_shaded_area 
  (a_w : ℕ) (a_l : ℕ) (b_w : ℕ) (b_l : ℕ) (c_w : ℕ) (c_l : ℕ)
  (overlap_ab_w : ℕ) (overlap_ab_h : ℕ)
  (overlap_ac_w : ℕ) (overlap_ac_h : ℕ)
  (overlap_bc_w : ℕ) (overlap_bc_h : ℕ)
  (triple_overlap_w : ℕ) (triple_overlap_h : ℕ) :
  a_w = 4 → a_l = 12 →
  b_w = 5 → b_l = 10 →
  c_w = 3 → c_l = 6 →
  overlap_ab_w = 4 → overlap_ab_h = 5 →
  overlap_ac_w = 3 → overlap_ac_h = 4 →
  overlap_bc_w = 3 → overlap_bc_h = 3 →
  triple_overlap_w = 3 → triple_overlap_h = 3 →
  ((a_w * a_l) + (b_w * b_l) + (c_w * c_l)) - 
  ((overlap_ab_w * overlap_ab_h) + (overlap_ac_w * overlap_ac_h) + (overlap_bc_w * overlap_bc_h)) + 
  (triple_overlap_w * triple_overlap_h) = 84 :=
by 
  sorry

end intersecting_rectangles_shaded_area_l76_76074


namespace interval_monotonically_decreasing_triangle_properties_l76_76736

def f (x : Real) : Real :=
  let m := (Real.sin x, -1)
  let n := (Real.sqrt 3 * Real.cos x, -1 / 2)
  let dot_prod := (m.1 + n.1) * m.1 + (m.2 + n.2) * m.2
  dot_prod

theorem interval_monotonically_decreasing (k : ℤ) :
  ∃ I : Set ℝ, I = {(x : ℝ) | k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 6} ∧
  ∀ x ∈ I, derivative (f x) < 0 :=
sorry

theorem triangle_properties (A a c f : Real) (b : Real) (S : Real) (ha : a = 2 * Real.sqrt 3) (hc : c = 4) (Aacute : A < Real.pi / 2) (hf : f = fun x => 2 + Real.sin (2 * x - Real.pi / 6)) :
  A = Real.pi / 3 ∧ b = 2 ∧ S = 2 * Real.sqrt 3 :=
sorry

end interval_monotonically_decreasing_triangle_properties_l76_76736


namespace fox_initial_coins_l76_76076

theorem fox_initial_coins :
  ∃ x : ℤ, x - 10 = 0 ∧ 2 * (x - 10) - 50 = 0 ∧ 2 * (2 * (x - 10) - 50) - 50 = 0 ∧
  2 * (2 * (2 * (x - 10) - 50) - 50) - 50 = 0 ∧ 2 * (2 * (2 * (2 * (x - 10) - 50) - 50) - 50) - 50 = 0 ∧
  x = 56 := 
by
  -- we skip the proof here
  sorry

end fox_initial_coins_l76_76076


namespace arctan_sum_l76_76650

theorem arctan_sum : 
  let x := (3 : ℝ) / 7
  let y := 7 / 3
  x * y = 1 → (Real.arctan x + Real.arctan y = Real.pi / 2) :=
by
  intros x y h
  -- Proof goes here
  sorry

end arctan_sum_l76_76650


namespace simplify_cosine_sum_l76_76897

theorem simplify_cosine_sum :
  cos (2 * π / 11) + cos (6 * π / 11) + cos (8 * π / 11) = (-1 + sqrt (-11)) / 4 :=
begin
  sorry
end

end simplify_cosine_sum_l76_76897


namespace opposite_of_neg_2_is_2_l76_76495

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76495


namespace ellipse_standard_equation_point_on_circle_l76_76722

-- Part 1: Proving the standard equation of the ellipse
theorem ellipse_standard_equation (a b c: ℝ) (h1: a > b) (h2: b > 0) (h3: c = a * (√(3) / 2)) (h4: 2 * b = 2):
  (a = 2) ∧ (b = 1) ∧ (c = √3) ∧ (∀ x y: ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by
  sorry

-- Part 2: Proving the point (m, k) lies on the circle
theorem point_on_circle (k m: ℝ) (h1: (x1 y1 x2 y2: ℝ) (hx1: y1 = k * x1 + m) (hx2: y2 = k * x2 + m) 
    (h3: x1 + x2 = -8 * k * m / (4 * k^2 + 1)) (h4: x1 * x2 = (4 * m^2 - 4) / (4 * k^2 + 1)) 
    (h5: k * (y1 * y2 / x1 * x2) = k * (5 / 4)) : m^2 + k^2 = 5 / 4) :=
by
  sorry

end ellipse_standard_equation_point_on_circle_l76_76722


namespace solve_for_z_l76_76433

theorem solve_for_z (z : ℂ) (h : z * complex.I = 2 + 3 * complex.I) : 
  z = 3 - 2 * complex.I :=
sorry

end solve_for_z_l76_76433


namespace maximum_sin_ABC_l76_76377

theorem maximum_sin_ABC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = π) :
  sin A + sin B + sin C ≤ (3 * real.sqrt 3) / 2 := 
sorry

end maximum_sin_ABC_l76_76377


namespace find_value_of_expression_l76_76241

theorem find_value_of_expression
  (k m : ℕ)
  (hk : 3^(k - 1) = 9)
  (hm : 4^(m + 2) = 64) :
  2^(3*k + 2*m) = 2^11 :=
by 
  sorry

end find_value_of_expression_l76_76241


namespace ellipse_equation_hyperbola_equation_l76_76116

theorem ellipse_equation (e : ℝ) (d : ℝ) (h_e : e = sqrt 7 / 4) (h_d : d = 4) : 
    ∃ (a b : ℝ), (a > b > 0) ∧ (frac a 4) ∧ (frac b 3) ∧ (frac (x^2)(a^2) + frac (y^2)(b^2) = 1) :=
sorry

theorem hyperbola_equation (A : ℝ × ℝ) (F : ℝ × ℝ) (h_A : A = (6, -5)) (h_F : F = (-6, 0)) 
    : ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ (frac (6^2) / (a^2) - frac (25) / (b^2) = 1) ∧ (c^2 = a^2 + b^2)
    : ∃ (x y : ℝ), ∃ a b := 
sorry

end ellipse_equation_hyperbola_equation_l76_76116


namespace gcf_180_240_300_l76_76968

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end gcf_180_240_300_l76_76968


namespace quadratic_roots_l76_76933

theorem quadratic_roots {a : ℝ} :
  (4 < a ∧ a < 6) ∨ (a > 12) → 
  (∃ x1 x2 : ℝ, x1 = a + Real.sqrt (18 * (a - 4)) ∧ x2 = a - Real.sqrt (18 * (a - 4)) ∧ x1 > 0 ∧ x2 > 0) :=
by sorry

end quadratic_roots_l76_76933


namespace log2_80_not_determinable_l76_76280

variable {α : Type*}

def given_conditions (log2_16 : ℝ) (log2_3 : ℝ) : Prop :=
  log2_16 = 4 ∧ log2_3 ≈ 1.585

theorem log2_80_not_determinable (log2_16 : ℝ) (log2_3 : ℝ) 
  (h : given_conditions log2_16 log2_3) :
  ∀ log2_80 : ℝ, ¬(log2_80 = log2_16 + log2_5 (log2_5 formula) simplified) :=
sorry

end log2_80_not_determinable_l76_76280


namespace eval_ceiling_fraction_l76_76250

theorem eval_ceiling_fraction :
  (⌈(19: ℚ) / 11 - ⌈(35: ℚ) / 22⌉⌉ / ⌈(35: ℚ) / 11 + ⌈((11 * 22): ℚ) / 35⌉⌉) = 1 / 10 := 
by {
  have h1 : ⌈(35: ℚ) / 22⌉ = 2,  from sorry,
  have h2 : ⌈((11 * 22): ℚ) / 35⌉ = 7,  from sorry,
  have h3 : ⌈(19: ℚ) / 11 - 2⌉ = 0,  from sorry,
  have h4 : ⌈(35: ℚ) / 11 + 7⌉ = 10,  from sorry,
  show (⌈(19: ℚ) / 11 - ⌈(35: ℚ) / 22⌉⌉ / ⌈(35: ℚ) / 11 + ⌈((11 * 22): ℚ) / 35⌉⌉) = 1 / 10,
  rw [h1, h2, h3, h4],
  norm_num,
}

end eval_ceiling_fraction_l76_76250


namespace solution_set_of_inequality_l76_76065

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end solution_set_of_inequality_l76_76065


namespace max_sundays_in_84_days_l76_76086

-- Define constants
def days_in_week : ℕ := 7
def total_days : ℕ := 84

-- Theorem statement
theorem max_sundays_in_84_days : (total_days / days_in_week) = 12 :=
by sorry

end max_sundays_in_84_days_l76_76086


namespace soldiers_to_add_l76_76766

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76766


namespace number_of_elements_in_set_l76_76910

-- We define the conditions in terms of Lean definitions.
variable (n : ℕ) (S : ℕ)

-- Define the initial wrong average condition
def wrong_avg_condition : Prop := (S + 26) / n = 18

-- Define the corrected average condition
def correct_avg_condition : Prop := (S + 36) / n = 19

-- The main theorem to be proved
theorem number_of_elements_in_set (h1 : wrong_avg_condition n S) (h2 : correct_avg_condition n S) : n = 10 := 
sorry

end number_of_elements_in_set_l76_76910


namespace standard_equation_of_ellipse_l76_76633

theorem standard_equation_of_ellipse (a b : ℝ) (e : ℝ) (x y : ℝ) 
  (h1 : (3:ℝ), 0)
  (h2 : (e = Real.sqrt 6 / 3)) :
  (a^2 = 9 ∧ b^2 = 3 ∧ (x^2 / 9 + y^2 / 3 = 1))
  ∨ (a^2 = 9 ∧ b^2 = 27 ∧ (x^2 / 9 + y^2 / 27 = 1)) := 
sorry

end standard_equation_of_ellipse_l76_76633


namespace problem_statement_l76_76730

theorem problem_statement (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : ℝ) :
  ( ( ∀ x : ℝ, (x^2 - x + 1)^6 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 +
                    a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^{10} +
                    a_{11} * x^{11} + a_{12} * x^{12} ) ) →
  ((a + a_2 + a_4 + a_6 + a_8 + a_{10} + a_{12})^2 - (a_1 + a_3 + a_5 + a_7 + a_9 + a_{11})^2 = 729) := 
sorry

end problem_statement_l76_76730


namespace opposite_of_neg_two_l76_76553

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76553


namespace find_x_y_sum_l76_76735

theorem find_x_y_sum (x y : ℝ) (h1 : ∃ m, (2, -4, y) = m • (-1, x, 3)) : x + y = -4 :=
by
  sorry

end find_x_y_sum_l76_76735


namespace expression_remainder_l76_76590

theorem expression_remainder (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 :=
by
  sorry

end expression_remainder_l76_76590


namespace distribute_balls_l76_76748

open Nat

theorem distribute_balls : 
  (∑ (c : ℕ × ℕ × ℕ) in {x : ℕ × ℕ × ℕ | x.1 + x.2 + x.3 = 6 ∧ x.1 ≤ 2 ∧ x.2 ≤ 2 ∧ x.3 ≤ 2}.to_finset, 
    if c.1 = 2 ∧ c.2 = 2 ∧ c.3 = 2 then 
      choose 6 2 * choose 4 2 * choose 2 2 / 6
    else if c.1 = 3 ∧ c.2 = 3 ∧ c.3 = 0 then
      choose 6 3 * choose 3 3 / 2
    else if c.1 = 4 ∧ c.2 = 2 ∧ c.3 = 0 then
      choose 6 4 * choose 2 2
    else if c.1 = 3 ∧ c.2 = 2 ∧ c.3 = 1 then
      choose 6 3 * choose 3 2 * choose 1 1
    else 0) = 100 := by
  sorry

end distribute_balls_l76_76748


namespace sum_zero_product_n_if_and_only_if_divisible_by_4_l76_76883

theorem sum_zero_product_n_if_and_only_if_divisible_by_4 (n : ℕ) (hn : n ≠ 0) :
  (∃ (a : Fin n → ℤ), (∑ i, a i) = 0 ∧ (∏ i, a i) = n) ↔ 4 ∣ n := by
  sorry

end sum_zero_product_n_if_and_only_if_divisible_by_4_l76_76883


namespace example_satisfies_condition_x_is_perfect_square_l76_76932

-- Part (a): Specific example verification
theorem example_satisfies_condition : 
  (∃ (x y : ℕ+), x = 4 ∧ y = 2 ∧ ((x : ℕ) ^ 2019 + x + y ^ 2) % (x * y) = 0) := 
by {
  use [4, 2],
  split,
  { refl },
  split,
  { refl },
  {
    norm_cast,
    -- Calculating the expression (4 ^ 2019 + 4 + 2^2) % (4 * 2)
    sorry
  }
}

-- Part (b): Prove x is necessarily a perfect square given the condition
theorem x_is_perfect_square (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x^2019 + x + y^2) % (x * y) = 0) : ∃ (k : ℕ), x = k^2 :=
by {
  sorry
}

end example_satisfies_condition_x_is_perfect_square_l76_76932


namespace value_of_x_plus_y_l76_76697

theorem value_of_x_plus_y (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l76_76697


namespace maximum_discount_rate_l76_76135

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76135


namespace maximum_discount_rate_l76_76143

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76143


namespace find_num_c_atoms_l76_76617

def atomic_weight_c : ℝ := 12.01
def atomic_weight_h : ℝ := 1.008
def atomic_weight_o : ℝ := 16.00

def num_h_atoms : ℕ := 8
def num_o_atoms : ℕ := 6
def molecular_weight : ℝ := 176

noncomputable def weight_h_atoms : ℝ := num_h_atoms * atomic_weight_h
noncomputable def weight_o_atoms : ℝ := num_o_atoms * atomic_weight_o
noncomputable def weight_c_atoms : ℝ := molecular_weight - (weight_h_atoms + weight_o_atoms)
noncomputable def num_c_atoms : ℝ := weight_c_atoms / atomic_weight_c

theorem find_num_c_atoms : num_c_atoms ≈ 6 := sorry

end find_num_c_atoms_l76_76617


namespace star_3_4_equals_8_l76_76375

def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

theorem star_3_4_equals_8 : star 3 4 = 8 := by
  sorry

end star_3_4_equals_8_l76_76375


namespace sum_of_1_over_1004_array_l76_76120

theorem sum_of_1_over_1004_array (k : ℕ) :
  let term := (λ (r c : ℕ), 1 / ((2 * 1004)^r * 1004^c))
  let row_sum := (λ r : ℕ, sum (λ c : ℕ, term r c))
  let sum_of_first_k_rows := sum (row_sum r) (Finset.range k)
  sum_of_first_k_rows = 1004 * (1 - (1/2008)^k) / (2007 * 1003) :=
by sorry

end sum_of_1_over_1004_array_l76_76120


namespace sum_of_radii_of_inscribed_circles_equals_original_radius_l76_76648

variables {r r1 r2 r3 : ℝ}
variables {A B C : Type*} [metric_space A]

-- Given conditions in Lean code:
def circle_inscribed_in_triangle (r : ℝ) : Prop :=
  ∃ (ABC : triangle A), ABC.inscribed_circle.radius = r

def tangent_lines_cut_small_triangles_with_radii (ABC : triangle A) (r1 r2 r3 : ℝ) : Prop :=
  ∃ (triangles : list (triangle A)), ∀ (t : triangle A), t ∈ triangles → t.inscribed_circle.radius ∈ [r1, r2, r3]

-- Main theorem in Lean code:
theorem sum_of_radii_of_inscribed_circles_equals_original_radius
  (r r1 r2 r3 : ℝ)
  (h₀ : circle_inscribed_in_triangle r)
  (h₁ : ∃ ABC, tangent_lines_cut_small_triangles_with_radii ABC r1 r2 r3) :
  r1 + r2 + r3 = r :=
sorry

end sum_of_radii_of_inscribed_circles_equals_original_radius_l76_76648


namespace opposite_of_neg2_l76_76501

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76501


namespace f_eq_n_l76_76010

open Nat

def f : ℕ+ → ℕ+  -- ℕ+ denotes positive integers

theorem f_eq_n (f : ℕ+ → ℕ+) (h : ∀ (n : ℕ+), f (n + 1) > f (f (n))) : ∀ (n : ℕ+), f n = n :=
sorry

end f_eq_n_l76_76010


namespace curves_intersect_at_three_points_l76_76097

theorem curves_intersect_at_three_points (b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = b^2 ∧ y = 2 * x^2 - b) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₁^2 + y₁^2 = b^2) ∧ (x₂^2 + y₂^2 = b^2) ∧ (x₃^2 + y₃^2 = b^2) ∧
    (y₁ = 2 * x₁^2 - b) ∧ (y₂ = 2 * x₂^2 - b) ∧ (y₃ = 2 * x₃^2 - b)) ↔ b > 1 / 4 :=
by
  sorry

end curves_intersect_at_three_points_l76_76097


namespace scientific_notation_123000_l76_76831

theorem scientific_notation_123000 : (123000 : ℝ) = 1.23 * 10^5 := by
  sorry

end scientific_notation_123000_l76_76831


namespace smallest_positive_m_l76_76582

theorem smallest_positive_m {m p q : ℤ} (h_eq : 12 * p^2 - m * p - 360 = 0) (h_pq : p * q = -30) :
  (m = 12 * (p + q)) → 0 < m → m = 12 :=
by
  sorry

end smallest_positive_m_l76_76582


namespace infinite_either_interval_exists_rational_infinite_elements_l76_76237

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end infinite_either_interval_exists_rational_infinite_elements_l76_76237


namespace opposite_of_neg_two_l76_76551

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76551


namespace lemonade_quarts_water_l76_76573

-- Definitions derived from the conditions
def total_parts := 6 + 2 + 1 -- Sum of all ratio parts
def parts_per_gallon : ℚ := 1.5 / total_parts -- Volume per part in gallons
def parts_per_quart : ℚ := parts_per_gallon * 4 -- Volume per part in quarts
def water_needed : ℚ := 6 * parts_per_quart -- Quarts of water needed

-- Statement to prove
theorem lemonade_quarts_water : water_needed = 4 := 
by sorry

end lemonade_quarts_water_l76_76573


namespace find_v_l76_76673

noncomputable def v : ℝ :=
  sorry -- This definition aligns with using v as real number in solving logarithms.

lemma log_eq_v :
  log 8 (v + 24) = 7 / 3 :=
  sorry -- The given condition.

theorem find_v : v = 104 :=
  sorry -- The goal is to prove that v = 104.

end find_v_l76_76673


namespace max_right_angles_in_convex_polygon_l76_76926

theorem max_right_angles_in_convex_polygon (sum_exterior_angles : ∀ n, n > 2 → (∑ i in (range n), exterior_angle i) = 360)
  (right_angle := 90) : ∃ n, maximum_right_angles_in_convex_polygon n = 4 :=
by
  -- Sorry to skip the actual proof
  sorry

end max_right_angles_in_convex_polygon_l76_76926


namespace abc_product_le_two_l76_76430

theorem abc_product_le_two (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end abc_product_le_two_l76_76430


namespace no_valid_bases_l76_76462

theorem no_valid_bases
  (x y : ℕ)
  (h1 : 4 * x + 9 = 4 * y + 1)
  (h2 : 4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9)
  (hx : x > 1)
  (hy : y > 1)
  : false :=
by
  sorry

end no_valid_bases_l76_76462


namespace opposite_of_neg_two_l76_76514

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76514


namespace proof_expression_equals_60_times_10_power_1501_l76_76643

noncomputable def expression_equals_60_times_10_power_1501 : Prop :=
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501

theorem proof_expression_equals_60_times_10_power_1501 :
  expression_equals_60_times_10_power_1501 :=
by 
  sorry

end proof_expression_equals_60_times_10_power_1501_l76_76643


namespace Craig_initial_apples_l76_76243

variable (j : ℕ) (shared : ℕ) (left : ℕ)

theorem Craig_initial_apples (HJ : j = 11) (HS : shared = 7) (HL : left = 13) :
  shared + left = 20 := by
  sorry

end Craig_initial_apples_l76_76243


namespace number_divisible_by_33_l76_76957

theorem number_divisible_by_33 (x y : ℕ) 
  (h1 : (x + y) % 3 = 2) 
  (h2 : (y - x) % 11 = 8) : 
  (27850 + 1000 * x + y) % 33 = 0 := 
sorry

end number_divisible_by_33_l76_76957


namespace product_calculation_l76_76652

theorem product_calculation :
  12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end product_calculation_l76_76652


namespace count_divisible_by_4_6_10_l76_76742

theorem count_divisible_by_4_6_10 :
  (card {n : ℕ | n < 300 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0}) = 4 :=
by 
  sorry

end count_divisible_by_4_6_10_l76_76742


namespace find_roots_l76_76416

noncomputable def zeta : ℂ := sorry
noncomputable def gamma : ℂ := zeta + zeta^2 + zeta^4
noncomputable def delta : ℂ := zeta^3 + zeta^5 + zeta^6 + zeta^7

theorem find_roots :
  (zeta^9 = 1) ∧ (zeta ≠ 1) → (gamma + delta = -1) ∧ (gamma * delta = 3) → 
  ∃ (c d : ℝ), (γ δ are the roots of x^2 + cx + d = 0) ∧ (c = 1) ∧ (d = 3) :=
by
  sorry

end find_roots_l76_76416


namespace hyperbola_parabola_focus_l76_76729

open Classical

theorem hyperbola_parabola_focus :
  ∃ a : ℝ, (a > 0) ∧ (∃ c > 0, (c = 2) ∧ (a^2 + 3 = c^2)) → a = 1 :=
sorry

end hyperbola_parabola_focus_l76_76729


namespace minimum_soldiers_to_add_l76_76802

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76802


namespace expression_for_y_l76_76123

noncomputable def operation_sequence (x : ℝ) (n : ℕ) :ℝ :=
if even n then 3 * x ^ (4^n)
else (3 * x ^ (4^n))⁻¹

theorem expression_for_y (x : ℝ) (n : ℕ) (h : x ≠ 0) : ∃ k, 
  (k = 0 → even n) ∧ (k = 1 → odd n) ∧
  operation_sequence x n = 3^(1 - 2 * k) * x ^ (4^n) := 
sorry

end expression_for_y_l76_76123


namespace meeting_probability_l76_76204

def time_bounds := (0 : ℝ) ≤ x ∧ x ≤ 4 ∧ (0 : ℝ) ≤ y ∧ y ≤ 4 ∧ (0 : ℝ) ≤ z ∧ z ≤ 4
def meeting_possible (x y z : ℝ) := abs (x - z) ≤ 0.5 ∧ abs (y - z) ≤ 0.5
def volume_total := (4 : ℝ) ^ 3
def volume_meeting := (1 : ℝ) * 4 * 4

theorem meeting_probability : 
  (1 / volume_total * volume_meeting) = 0.25 :=
by
  sorry

end meeting_probability_l76_76204


namespace probability_at_least_50_cents_l76_76614

-- Define the types for coins
inductive Coin
| penny
| nickel
| dime

open Coin

-- Given conditions
def box : List Coin := List.replicate 2 penny ++ List.replicate 4 nickel ++ List.replicate 6 dime
def num_coins := 6

-- Define the value of each coin
def value (c : Coin) : ℕ :=
  match c with
  | penny   => 1
  | nickel  => 5
  | dime    => 10

-- Function to calculate the total value of a list of coins
def total_value (coins : List Coin) : ℕ :=
  coins.map value |>.sum

-- Total number of ways to draw 6 coins out of 12 coins
def total_outcomes := nat.choose 12 6

-- Number of successful outcomes
def successful_outcomes := 
  ((List.replicate 1 penny ++ List.replicate 5 dime) :: 
   (List.replicate 2 nickel ++ List.replicate 4 dime) ::
   (List.replicate 1 nickel ++ List.replicate 5 dime) ::
   [List.replicate 6 dime])
  .count (λ coins, total_value coins >= 50)

-- Calculate the probability
def probability := (successful_outcomes.toRat / total_outcomes.toRat)

-- Prove the probability is as expected
theorem probability_at_least_50_cents : probability = 127 / 924 := by
  sorry

end probability_at_least_50_cents_l76_76614


namespace virus_diameter_scientific_notation_l76_76456

theorem virus_diameter_scientific_notation :
  (0.000000103 : ℝ) = 1.03 * (10:ℝ) ^ (-7) := 
sorry

end virus_diameter_scientific_notation_l76_76456


namespace bus_stops_duration_per_hour_l76_76602

def speed_without_stoppages : ℝ := 90
def speed_with_stoppages : ℝ := 84
def distance_covered_lost := speed_without_stoppages - speed_with_stoppages

theorem bus_stops_duration_per_hour :
  distance_covered_lost / speed_without_stoppages * 60 = 4 :=
by
  sorry

end bus_stops_duration_per_hour_l76_76602


namespace right_triangle_sides_l76_76630

theorem right_triangle_sides : ∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = Real.sqrt 2 ∧ a^2 + b^2 = c^2 :=
by
  use [1, 1, Real.sqrt 2]
  split
  · refl
  split
  · refl
  split
  · refl
  · sorry

end right_triangle_sides_l76_76630


namespace average_extra_chores_l76_76645

-- Define the conditions
variables (fixed_allowance : ℕ) (extra_earn_per_chore : ℚ) (weeks : ℕ) (total_money : ℚ)
variables (average_extra_chores_per_week : ℚ)

-- Set the given values
constants (allowance : fixed_allowance = 20)
           (extra_per_chore : extra_earn_per_chore = 1.5)
           (num_weeks : weeks = 10)
           (total_savings : total_money = 425)

-- Define the property to prove
theorem average_extra_chores :
  average_extra_chores_per_week = (total_money / weeks - fixed_allowance) / extra_earn_per_chore := by
  sorry

end average_extra_chores_l76_76645


namespace opposite_of_neg_two_is_two_l76_76519

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76519


namespace gcd_1443_999_l76_76923

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end gcd_1443_999_l76_76923


namespace solve_olympics_problem_max_large_sets_l76_76048

-- Definitions based on the conditions
variables (x y : ℝ)

-- Condition 1: 2 small sets cost $20 less than 1 large set
def condition1 : Prop := y - 2 * x = 20

-- Condition 2: 3 small sets and 2 large sets cost $390
def condition2 : Prop := 3 * x + 2 * y = 390

-- Finding unit prices
def unit_prices : Prop := x = 50 ∧ y = 120

-- Condition 3: Budget constraint for purchasing sets
def budget_constraint (m : ℕ) : Prop := m ≤ 7

-- Prove unit prices and purchasing constraints
theorem solve_olympics_problem :
  condition1 x y ∧ condition2 x y → unit_prices x y :=
by
  sorry

theorem max_large_sets :
  budget_constraint 7 :=
by
  sorry

end solve_olympics_problem_max_large_sets_l76_76048


namespace keanu_needs_refills_l76_76407

def distance_sa := 80
def distance_ab := 120
def distance_bc := 160
def distance_cs := 100

def consumption_sa := 8 / 35.0
def consumption_ab := 8 / 55.0
def consumption_bc := 8 / 45.0
def consumption_cs := 8 / 60.0

def tank_capacity := 8

def total_gas_needed :=
  (distance_sa * consumption_sa) +
  (distance_ab * consumption_ab) +
  (distance_bc * consumption_bc) +
  (distance_cs * consumption_cs)

def number_of_refills := (total_gas_needed / tank_capacity).ceil

theorem keanu_needs_refills : number_of_refills = 10 :=
by
  -- Insert proof here
  sorry

end keanu_needs_refills_l76_76407


namespace abs_diff_of_slopes_l76_76733

theorem abs_diff_of_slopes (k1 k2 b : ℝ) (h : k1 * k2 < 0) (area_cond : (1 / 2) * 3 * |k1 - k2| * 3 = 9) :
  |k1 - k2| = 2 :=
by
  sorry

end abs_diff_of_slopes_l76_76733


namespace train_speed_l76_76215

variable (length : ℕ) (time : ℕ)
variable (h_length : length = 120)
variable (h_time : time = 6)

theorem train_speed (length time : ℕ) (h_length : length = 120) (h_time : time = 6) :
  length / time = 20 := by
  sorry

end train_speed_l76_76215


namespace carson_clouds_l76_76233

theorem carson_clouds (C D : ℕ) (h1 : D = 3 * C) (h2 : C + D = 24) : C = 6 :=
by
  sorry

end carson_clouds_l76_76233


namespace tan_alpha_minus_beta_l76_76363

theorem tan_alpha_minus_beta (α β : ℝ) (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_l76_76363


namespace third_test_point_l76_76400

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point_l76_76400


namespace opposite_of_neg_2_is_2_l76_76498

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76498


namespace min_value_2xy_minus_2x_minus_y_l76_76295

theorem min_value_2xy_minus_2x_minus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 2/y = 1) :
  2 * x * y - 2 * x - y ≥ 8 :=
sorry

end min_value_2xy_minus_2x_minus_y_l76_76295


namespace min_soldiers_needed_l76_76773

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76773


namespace tan_alpha_minus_beta_l76_76349

variable (α β : ℝ)

theorem tan_alpha_minus_beta
  (h : sin (α + β) + cos (α + β) = 2 * real.sqrt 2 * cos (α + π/4) * sin β) : 
  real.tan (α - β) = -1 :=
sorry

end tan_alpha_minus_beta_l76_76349


namespace neither_play_sports_l76_76843

theorem neither_play_sports (total_students : ℕ) (table_tennis : ℕ) (chess : ℕ) (both : ℕ) : 
  total_students = 12 → table_tennis = 5 → chess = 8 → both = 3 → 
  (total_students - (table_tennis + chess - both)) = 2 := by
  intros h_total h_tt h_ch h_both
  rw [h_total, h_tt, h_ch, h_both]
  simp

end neither_play_sports_l76_76843


namespace proj_v_on_w_l76_76269

def v : ℝ × ℝ × ℝ := (3, -2, 4)
def w : ℝ × ℝ × ℝ := (2, -1, 2)
noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(c * v.1, c * v.2, c * v.3)

noncomputable def projection (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
scalar_mult (dot_product v w / dot_product w w) w

theorem proj_v_on_w :
projection v w = (⟨32 / 9, -16 / 9, 32 / 9⟩ : ℝ × ℝ × ℝ) :=
by
  sorry

end proj_v_on_w_l76_76269


namespace part1_part2_l76_76432

-- Defining set A
def A : Set ℝ := {x | x^2 + 4 * x = 0}

-- Defining set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

-- Problem 1: Prove that if A ∩ B = A ∪ B, then a = 1
theorem part1 (a : ℝ) : (A ∩ (B a) = A ∪ (B a)) → a = 1 := by
  sorry

-- Problem 2: Prove the range of values for a if A ∩ B = B
theorem part2 (a : ℝ) : (A ∩ (B a) = B a) → a ∈ Set.Iic (-1) ∪ {1} := by
  sorry

end part1_part2_l76_76432


namespace find_a_l76_76244

def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (h : star a 4 = 17) : a = 49 / 3 :=
by sorry

end find_a_l76_76244


namespace solve_quadratic_eq_1_solve_quadratic_eq_2_l76_76459

-- Proof for Equation 1
theorem solve_quadratic_eq_1 : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

-- Proof for Equation 2
theorem solve_quadratic_eq_2 : ∀ x : ℝ, 5 * x - 2 = (2 - 5 * x) * (3 * x + 4) ↔ (x = 2 / 5 ∨ x = -5 / 3) :=
by sorry

end solve_quadratic_eq_1_solve_quadratic_eq_2_l76_76459


namespace max_discount_rate_l76_76189

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76189


namespace wendy_tooth_extraction_cost_eq_290_l76_76084

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l76_76084


namespace find_least_four_digit_integer_divisible_by_2_3_5_7_l76_76972

theorem find_least_four_digit_integer_divisible_by_2_3_5_7 :
  ∃ n, 1000 ≤ n ∧ n < 10000 ∧ (∀ (k : ℕ), k ∣ n ↔ k ∈ {2, 3, 5, 7}) ∧ ∀ m, 1000 ≤ m ∧ m < n → ¬ (∀ k, k ∈ {2, 3, 5, 7} → k ∣ m) → n = 1050 :=
by
  sorry

end find_least_four_digit_integer_divisible_by_2_3_5_7_l76_76972


namespace sum_of_digits_divisible_by_7_l76_76579

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

def sum_digits (n : ℕ) : ℕ :=
  n.digits.foldl (λ acc d, acc + d) 0

theorem sum_of_digits_divisible_by_7 (n : ℕ) (h : is_divisible_by_7 n) : sum_digits n ≥ 2 :=
by
  sorry

end sum_of_digits_divisible_by_7_l76_76579


namespace opposite_of_neg_two_l76_76512

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76512


namespace reggie_money_left_l76_76452

theorem reggie_money_left:
  ∀ (initial_money cost_per_book : ℕ) (books_bought : ℕ),
    initial_money = 48 →
    cost_per_book = 2 →
    books_bought = 5 →
    initial_money - (books_bought * cost_per_book) = 38 :=
by
  intros initial_money cost_per_book books_bought h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end reggie_money_left_l76_76452


namespace max_rectangles_in_step_triangle_l76_76881

theorem max_rectangles_in_step_triangle (n : ℕ) (h : n = 6) : 
  let num_rectangles := 
    ∑ i in finset.range n, ∑ j in finset.range (n - i), (n - i) * (n - j) in
  num_rectangles = 126 := 
by
  sorry

end max_rectangles_in_step_triangle_l76_76881


namespace opposite_of_neg_two_is_two_l76_76534

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76534


namespace values_range_l76_76871

noncomputable def possible_values (x y : ℝ) : Set ℝ :=
{t | ∃ (hx : x > 0) (hy : y > 0) (hxy : x + y = 2), t = 1/x + 1/y}

theorem values_range : possible_values = { z : ℝ | ∃ (t ≥ 2), z = t } :=
by
  sorry

end values_range_l76_76871


namespace tan_alpha_minus_beta_l76_76359

theorem tan_alpha_minus_beta (α β : ℝ) (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_l76_76359


namespace proof_statement_l76_76812

-- Definition of the complex conjugate
def conjugate (z : ℂ) : ℂ := complex.conj z

-- Conditions
axiom condition (z : ℂ) : (1 + complex.I) * z = 2 * complex.I ^ 5

-- Statements to prove
def problem_statement (z : ℂ) := 
  conjugate z = 1 - complex.I 
  ∧ z + (conjugate z) = 2 
  ∧ z.re > 0 ∧ z.im > 0 
  ∧ z^2 = 2 * complex.I

-- Prove the final statement under the given condition
theorem proof_statement (z : ℂ) (h : (1 + complex.I) * z = 2 * complex.I ^ 5) : 
  problem_statement z :=
sorry

end proof_statement_l76_76812


namespace alice_wins_probability_l76_76223

/-- Alice rolls a standard 6-sided die. Bob rolls a second standard 6-sided die.
Alice wins if the values shown differ by 1. -/
def die_rolls := set (ℕ × ℕ) :=
  { (a, b) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} }

theorem alice_wins_probability :
  let favorable_outcomes := { (a, b) | (a, b) ∈ die_rolls ∧ |a - b| = 1 },
      total_outcomes := { (a, b) | (a, b) ∈ die_rolls },
      probability := favorable_outcomes.card.to_rat / total_outcomes.card.to_rat
  in probability = 5 / 18 := by
  sorry

end alice_wins_probability_l76_76223


namespace tan_alpha_minus_beta_l76_76350

variable (α β : ℝ)

theorem tan_alpha_minus_beta
  (h : sin (α + β) + cos (α + β) = 2 * real.sqrt 2 * cos (α + π/4) * sin β) : 
  real.tan (α - β) = -1 :=
sorry

end tan_alpha_minus_beta_l76_76350


namespace number_of_solutions_of_system_l76_76340

theorem number_of_solutions_of_system :
  let S : Set (ℝ × ℝ) := {p | let x := p.1 in let y := p.2 in x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1} in
  S.card = 2 :=
by
  sorry

end number_of_solutions_of_system_l76_76340


namespace sum_first_2017_l76_76713

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (OA OB OC : α)

-- Arithmetic sequence condition
axiom arithSeq (n : ℕ) : a (n + 1) = a n + d

-- Sum of first n terms condition
def sumTerms (n : ℕ) : ℝ := n * (a 1 + a n) / 2

-- Given vector condition
axiom vectorCond : OC = (a 17 - 3) • OA + (a 2001) • OB

-- Collinearity condition (implies a₁ + a₁₇ = 4)
axiom collinear : (a 17 - 3) + (a 2001) = 1

-- Proving the final statement
theorem sum_first_2017 : S 2017 = 4034 :=
by
  -- Translate arithSeq and sumTerms definitions into usage
  have h1 : a 1 + a 2017 = 4 := sorry,
  show S 2017 = 4034 := sorry

end sum_first_2017_l76_76713


namespace sum_and_product_of_roots_sum_of_reciprocal_roots_of_specific_quadratic_l76_76421

theorem sum_and_product_of_roots (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b^2 - 4 * a * c ≥ 0) :
  let disc := real.sqrt (b^2 - 4 * a * c) in
  let x1 := (-b + disc) / (2 * a) in
  let x2 := (-b - disc) / (2 * a) in
  (x1 + x2 = -b / a) ∧ (x1 * x2 = c / a) :=
by
  sorry

theorem sum_of_reciprocal_roots_of_specific_quadratic :
  let a := 1 in
  let b := 13 in
  let c := -real.sqrt 17 in
  let disc := real.sqrt (b^2 - 4 * a * c) in
  let y1 := (-b + disc) / (2 * a) in
  let y2 := (-b - disc) / (2 * a) in
  (1/y1 + 1/y2 = (13 * real.sqrt 17) / 17) :=
by
  sorry

end sum_and_product_of_roots_sum_of_reciprocal_roots_of_specific_quadratic_l76_76421


namespace trains_crossing_time_l76_76953

-- Definitions based on the problem conditions
def length_of_train : ℝ := 120 -- Length of each train in meters
def time_train1 : ℝ := 8 -- Time for the first train to cross a telegraph post in seconds
def time_train2 : ℝ := 15 -- Time for the second train to cross a telegraph post in seconds

-- Given the lengths and times, prove the time taken for the trains to cross each other
theorem trains_crossing_time : 
  let v1 := length_of_train / time_train1 in
  let v2 := length_of_train / time_train2 in
  let relative_speed := v1 + v2 in
  let total_distance := 2 * length_of_train in
  let crossing_time := total_distance / relative_speed in
  crossing_time = 240 / 23 :=
by
  sorry -- Omit the proof steps

end trains_crossing_time_l76_76953


namespace opposite_of_neg_two_l76_76554

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76554


namespace max_discount_rate_l76_76126

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76126


namespace tournament_matches_l76_76569

theorem tournament_matches (n : ℕ) (h : n > 0) : (number_of_matches n = n - 1) :=
sorry

end tournament_matches_l76_76569


namespace length_CD_l76_76398

-- Definitions from the problem conditions
variable (x y : ℝ) -- x = AB = AC, y = CB = CD

-- Conditions given in the problem
axiom is_isosceles_ABC : x = AB = AC
axiom is_isosceles_CBD : y = CB = CD
axiom perimeter_ABC : 2 * x + y = 20
axiom perimeter_CBD : 2 * y + 9 = 25

theorem length_CD : y = 8 := by
  -- Here we would provide proof based on the given conditions
  sorry

end length_CD_l76_76398


namespace valid_house_orderings_l76_76081

-- Definitions for the conditions of the problem
def ordering_condition1 (houses : List Char) : Prop :=
  houses.indexOf 'O' < houses.indexOf 'R'

def ordering_condition2 (houses : List Char) : Prop :=
  houses.indexOf 'B' < houses.indexOf 'Y'

def ordering_condition3 (houses : List Char) : Prop :=
  (houses.indexOf 'B' + 1 < houses.indexOf 'Y') ∨ (houses.indexOf 'B' > houses.indexOf 'Y' + 1)

def ordering_condition4 (houses : List Char) : Prop :=
  houses.indexOf 'O' < houses.indexOf 'G'

-- The main theorem to be proved
theorem valid_house_orderings :
  {houses : List Char // ordering_condition1 houses ∧ 
                      ordering_condition2 houses ∧ 
                      ordering_condition3 houses ∧ 
                      ordering_condition4 houses}.card = 7 :=
sorry

end valid_house_orderings_l76_76081


namespace tangent_bisects_angle_l76_76394

variables {A B C D E : Type*}
variables [triangle_ABC : right_triangle A B C]
variables [circle : circle (diameter BC)]
variables [hypotenuse : hypotenuse AB]
variables [tangent_circle : tangent_point circle D (line AB)]
variables [tangent_intersect : tangent_intersect_at tangent_circle (line BC) E]

theorem tangent_bisects_angle :
  bisects_angle (ray (line DE)) (∠BDE) = true := 
sorry

end tangent_bisects_angle_l76_76394


namespace sum_of_first_n_terms_is_a_geometric_l76_76290

variable {R : Type} [Real R]

section
  -- Part (Ⅰ) Definitions
  variable {a : ℕ → R} (b : ℕ → R)
  variable (t : R) (h₀ : t ≠ 0)
  variable (h₁ : a 1 = 1) (h₂ : a 2 = t)
  variable (h₃ : ∀ n : ℕ, b n = a n * a (n + 1))

  -- (Ⅰ) Theorem: Sum of the first n terms
  theorem sum_of_first_n_terms (n : ℕ) : 
    let T_n := Σ i in Finset.range n, b i
    T_n = if t = -1 then - (n : R)
          else if t = 1 then (n : R)
          else (t - t ^ (2 * n + 1)) / (1 - t ^ 2) := 
  sorry
end

section
  -- Part (Ⅱ) Definitions
  variable {a : ℕ → R} (b : ℕ → R)
  variable (q : R)
  variable (h₀ : ∀ n : ℕ, b (n + 1) / b n = q)
  variable (h₁ : ∀ n : ℕ, b n = a n * a (n + 1))
  variable (h₂ : a 1 = 1) (h₃ : a 2 = t)

  -- (Ⅱ) Theorem: Whether a_n is geometric
  theorem is_a_geometric : 
    if q = t^2 then (∃ r : ℝ, ∀ n : ℕ, a n = a 0 * r ^ n)
    else true := 
  sorry
end

end sum_of_first_n_terms_is_a_geometric_l76_76290


namespace circular_pipes_equivalence_l76_76247

/-- Determine how many circular pipes with an inside diameter 
of 2 inches are required to carry the same amount of water as 
one circular pipe with an inside diameter of 8 inches. -/
theorem circular_pipes_equivalence 
  (d_small d_large : ℝ)
  (h1 : d_small = 2)
  (h2 : d_large = 8) :
  (d_large / 2) ^ 2 / (d_small / 2) ^ 2 = 16 :=
by
  sorry

end circular_pipes_equivalence_l76_76247


namespace point_in_second_quadrant_l76_76717

def imaginary_unit : ℂ := complex.i

def compute_z : ℂ := (1 + 2 * imaginary_unit) * imaginary_unit

theorem point_in_second_quadrant (z : ℂ) (h : z = compute_z) : (z.re < 0 ∧ z.im > 0) :=
by {
  rw [h, complex.mul_re, complex.mul_im],
  simp,
  split,
  { linarith, },
  { linarith, }
  sorry
}

end point_in_second_quadrant_l76_76717


namespace school_anniversary_problem_l76_76902

theorem school_anniversary_problem
    (total_cost : ℕ)
    (cost_commemorative_albums cost_bone_china_cups : ℕ)
    (num_commemorative_albums num_bone_china_cups : ℕ)
    (price_commemorative_album price_bone_china_cup : ℕ)
    (H1 : total_cost = 312000)
    (H2 : cost_commemorative_albums + cost_bone_china_cups = total_cost)
    (H3 : cost_commemorative_albums = 3 * cost_bone_china_cups)
    (H4 : price_commemorative_album = 3 / 2 * price_bone_china_cup)
    (H5 : num_bone_china_cups = 4 * num_commemorative_albums + 1600) :
    (cost_commemorative_albums = 72000 ∧ cost_bone_china_cups = 240000) ∧
    (price_commemorative_album = 45 ∧ price_bone_china_cup = 30) :=
by
  sorry

end school_anniversary_problem_l76_76902


namespace min_soldiers_to_add_l76_76807

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76807


namespace maximum_discount_rate_l76_76142

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l76_76142


namespace largest_fraction_l76_76098

theorem largest_fraction : (A B C D E : ℚ) 
  (hA : A = 1 / 3) 
  (hB : B = 1 / 4) 
  (hC : C = 3 / 8) 
  (hD : D = 5 / 12) 
  (hE : E = 7 / 24) : D > A ∧ D > B ∧ D > C ∧ D > E := by
  sorry

end largest_fraction_l76_76098


namespace equation1_solutions_equation2_solutions_l76_76460

theorem equation1_solutions (x : ℝ) : x^2 - 4 = 0 → (x = 2 ∨ x = -2) :=
sorry

theorem equation2_solutions (x : ℝ) : x^2 - 2x = 3 → (x = -1 ∨ x = 3) :=
sorry

end equation1_solutions_equation2_solutions_l76_76460


namespace arithmetic_sqrt_of_linear_combination_l76_76311

theorem arithmetic_sqrt_of_linear_combination (a b : ℝ)
  (h1 : real.sqrt (a + 9) = -5)
  (h2 : real.cbrt (2 * b - a) = -2) :
  real.sqrt (2 * a + b) = 6 :=
by
  sorry

end arithmetic_sqrt_of_linear_combination_l76_76311


namespace num_possible_sequences_l76_76287

theorem num_possible_sequences :
  ∃ (C : ℕ → ℕ → ℕ), 
    let seqs := {a : fin 6 → fin 10 // ∀ i j, i ≠ j → a i ≠ a j ∧ a 0 > a 1 ∧ a 1 > a 2 ∧ a 3 < a 4 ∧ a 4 < a 5} in
    seqs.card = C 10 3 * C 7 3 :=
sorry

end num_possible_sequences_l76_76287


namespace polyhedron_edge_sum_leq_l76_76886

-- Define distance between two points in a polyhedron
variable {P : Type} [polyhedron P]
variable (d : P → P → ℝ)

-- Define vertices and edges of the polyhedron
variable (A B : P)
variable (edges : List (P × P))

-- Assume A and B are the points at the greatest distance in P
variable (h1 : ∀ x y : P, d x y ≤ d A B)

-- Define a function to compute the length of an edge
def edge_length : (P × P) → ℝ := λ (x y), d x y

-- Sum of lengths of all edges in polyhedron P
def sum_edge_lengths : ℝ := edges.map edge_length |>.sum

theorem polyhedron_edge_sum_leq : sum_edge_lengths edges ≥ 3 * d A B :=
  sorry

end polyhedron_edge_sum_leq_l76_76886


namespace minimum_soldiers_to_add_l76_76801

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76801


namespace profit_percentage_is_25_l76_76476

variables (C S : ℝ)
variables (profit_percentage : ℝ)

-- Conditions as given
axiom cost_price_equals_selling_price : 40 * C = 32 * S

-- Definition of profit percentage
def profit_percentage_calculated (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

-- Proof that the profit percentage is 25%
theorem profit_percentage_is_25 (C S : ℝ) (h : 40 * C = 32 * S) : profit_percentage_calculated C S = 25 :=
by
  sorry

end profit_percentage_is_25_l76_76476


namespace horner_method_v2_l76_76283

def f(x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.reverse.foldl (λ acc c => acc * x + c) 0

theorem horner_method_v2 :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  sorry

end horner_method_v2_l76_76283


namespace max_crosses_4x10_proof_l76_76600

def max_crosses_4x10 (table : Matrix ℕ ℕ Bool) : ℕ :=
  sorry -- Placeholder for actual function implementation

theorem max_crosses_4x10_proof (table : Matrix ℕ ℕ Bool) (h : ∀ i < 4, ∃ j < 10, table i j = tt) :
  max_crosses_4x10 table = 30 :=
sorry

end max_crosses_4x10_proof_l76_76600


namespace albert_needs_more_money_l76_76220

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l76_76220


namespace area_difference_is_correct_l76_76475

noncomputable def circumference_1 : ℝ := 264
noncomputable def circumference_2 : ℝ := 352

noncomputable def radius_1 : ℝ := circumference_1 / (2 * Real.pi)
noncomputable def radius_2 : ℝ := circumference_2 / (2 * Real.pi)

noncomputable def area_1 : ℝ := Real.pi * radius_1^2
noncomputable def area_2 : ℝ := Real.pi * radius_2^2

noncomputable def area_difference : ℝ := area_2 - area_1

theorem area_difference_is_correct :
  abs (area_difference - 4305.28) < 1e-2 :=
by
  sorry

end area_difference_is_correct_l76_76475


namespace opposite_of_neg_two_l76_76509

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l76_76509


namespace gcf_of_180_240_300_l76_76969

def prime_factors (n : ℕ) : ℕ → Prop :=
λ p, p ^ (nat.factorization n p)

def gcf (n1 n2 n3 : ℕ) : ℕ :=
nat.gcd n1 (nat.gcd n2 n3)

theorem gcf_of_180_240_300 : gcf 180 240 300 = 60 := by
  sorry

end gcf_of_180_240_300_l76_76969


namespace cot_identity_1_cot_identity_2_l76_76983

-- Definitions for the cotangents
def cot (x : ℝ) : ℝ := 1 / tan x

-- Condition for the first proof
theorem cot_identity_1 (α β γ : ℝ) :
  cot α * cot β + cot β * cot γ + cot α * cot γ = 1 :=
sorry

-- Condition for the second proof
theorem cot_identity_2 (α β γ : ℝ) :
  cot α * cot β + cot β * cot γ + cot α * cot γ = 1 →
  (cot α + cot β + cot γ - cot α * cot β * cot γ = 1 / (sin α * sin β * sin γ)) :=
sorry

end cot_identity_1_cot_identity_2_l76_76983


namespace angles_sum_l76_76622

noncomputable def quadrilateral (A B C D : Type) : Type := sorry

variables {A B C D : Type} [quadrilateral A B C D]

-- Given conditions as definitions
def inscribed_angle_ACB : ℝ := 50
def inscribed_angle_CAD : ℝ := 40

-- Statement to prove
theorem angles_sum (h : quadrilateral A B C D) 
  (hACB : inscribed_angle_ACB = 50)
  (hCAD : inscribed_angle_CAD = 40) :
  let angle_sum := 90 in
  angle_sum = 90 := by
  sorry

end angles_sum_l76_76622


namespace max_discount_rate_l76_76134

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76134


namespace part1_part2_part3_l76_76420

-- Problem 1
theorem part1 (a b: ℝ) (f g: ℝ → ℝ) (m: ℝ):
  (∀ x, f x = log x - a * x) →
  (∀ x, g x = b / x) →
  (f 1 = m ∧ g 1 = m) →
  (f' 1 = g' 1) →
  (∀ x, tangent_line_at f 1 = tangent_line_at g 1) →
  (∀ y, x - 2 * y - 2 = 0)  := 
by sorry

-- Problem 2
theorem part2 (a: ℝ) (f: ℝ → ℝ):
  (∀ x, f x = log x - a * x) →
  ((∃ c, ∀ x, f' x = c) ∧ (∀ x, f x ≠ 0)) →
  (a > 1 / real.exp 1) := 
by sorry

-- Problem 3
theorem part3 (a: ℝ) (b: ℝ) (F: ℝ → ℝ):
  (0 < a) →
  (b = 1) →
  (∀ x, F x = log x - a * x - 1 / x) →
  (0 < a ∧ a < log 2 + 1 / 2 → ∀ x ∈ set.Icc 1 2, F x ≥ -a - 1) ∧ 
  (a ≥ log 2 + 1 / 2 → ∀ x ∈ set.Icc 1 2, F x ≥ log 2 - 1 / 2 - 2 * a) := 
by sorry

end part1_part2_part3_l76_76420


namespace minimum_soldiers_to_add_l76_76776

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76776


namespace younger_son_age_in_30_years_l76_76914

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l76_76914


namespace determinant_transformation_l76_76279

theorem determinant_transformation 
  (p q r s : ℝ)
  (h : Matrix.det ![![p, q], ![r, s]] = 6) :
  Matrix.det ![![p, 9 * p + 4 * q], ![r, 9 * r + 4 * s]] = 24 := 
sorry

end determinant_transformation_l76_76279


namespace infinite_seq_partition_infinite_seq_within_interval_l76_76240

open Mathlib

variable (x : ℕ → ℝ) (ε : ℝ)

-- Condition: Infinite sequence of real numbers in [0, 1)
constant seq_bounded : ∀ n, 0 ≤ x n ∧ x n < 1

-- Condition: ε is strictly between 0 and 1/2
constant eps_cond : ε > 0 ∧ ε < 1/2

-- Problem 1: Prove that either [0, 1/2) or [1/2, 1) contains infinitely many elements
theorem infinite_seq_partition :
  (∃∞ n, 0 ≤ x n ∧ x n < 1/2) ∨ (∃∞ n, 1/2 ≤ x n ∧ x n < 1) := sorry

-- Problem 2: Prove that there exists a rational number α ∈ [0, 1] such that infinitely many elements are within [α - ε, α + ε]
theorem infinite_seq_within_interval :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃∞ n, (α.toReal - ε ≤ x n ∧ x n ≤ α.toReal + ε) := sorry

end infinite_seq_partition_infinite_seq_within_interval_l76_76240


namespace sum_of_digits_of_greatest_prime_divisor_of_4095_l76_76583

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def greatest_prime_divisor_of_4095 : ℕ := 13

theorem sum_of_digits_of_greatest_prime_divisor_of_4095 :
  sum_of_digits greatest_prime_divisor_of_4095 = 4 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_4095_l76_76583


namespace albert_needs_more_money_l76_76222

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l76_76222


namespace range_of_a_l76_76814

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 2 * (a - 1) * x + 2) → 
  (∀ x, x ≤ 4 → (derivative f x) ≤ 0) → 
  a ≤ -3 := 
by 
  intros hf hdec 
  have haxis : (derivative f (1 - a)) = 0 := sorry
  have hineq : 1 - a ≤ 4 := sorry
  linarith

end range_of_a_l76_76814


namespace find_AD_l76_76401

variable (A B C D : Type)
variable [InnerProductSpace ℝ A]
variables {AB AC BD CD AD : ℝ}

-- Given conditions
axiom AB_eq_12 : AB = 12
axiom AC_eq_20 : AC = 20
axiom D_perpendicular : ∀ A B C D, InnerProductGeometry.IsPerpendicular (line[A, D]) (line[B, C])
axiom BD_CD_ratio : BD / CD = 3 / 4

-- Prove that AD = (36 * sqrt(14)) / 7
theorem find_AD (h : ℝ) (AB AC BD CD AD : ℝ) :
  AB = 12 → AC = 20 → BD / CD = 3 / 4 → AD = (36 * sqrt 14) / 7 :=
by
  intros
  sorry

end find_AD_l76_76401


namespace range_of_f_l76_76841

-- Definitions from conditions
def f (x y : ℝ) : ℝ := (x + y) / (Real.floor x * Real.floor y + Real.floor x + Real.floor y + 1)

variable {x y : ℝ}

-- Condition that x and y are greater than 0 and their product is 1
axiom h1 : x > 0
axiom h2 : y > 0
axiom h3 : x * y = 1

-- Required proof statement
theorem range_of_f : 
  {z : ℝ | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ f x y = z} 
  = {1/2} ∪ set.Icc (5/6) (5/4) :=
sorry

end range_of_f_l76_76841


namespace min_distance_from_origin_to_line_l76_76013

-- Definitions of the conditions
def P (x y : ℝ) := x + y = 4
def O := (0 : ℝ, 0 : ℝ)

-- The statement of the problem
theorem min_distance_from_origin_to_line : 
  ∀ (x y : ℝ), P x y → dist (x, y) O = 2 * Real.sqrt 2 :=
by
  sorry

end min_distance_from_origin_to_line_l76_76013


namespace ada_original_seat_l76_76264

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end ada_original_seat_l76_76264


namespace find_base_l76_76564

theorem find_base (x b : ℝ) (h1 : 9^(x + 8) = 16^x) (h2 : x = Real.log 9^8 / Real.log b) : b = 16 / 9 := 
sorry

end find_base_l76_76564


namespace james_total_payment_l76_76854

noncomputable def first_pair_cost : ℝ := 40
noncomputable def second_pair_cost : ℝ := 60
noncomputable def discount_applied_to : ℝ := min first_pair_cost second_pair_cost
noncomputable def discount_amount := discount_applied_to / 2
noncomputable def total_before_extra_discount := first_pair_cost + (second_pair_cost - discount_amount)
noncomputable def extra_discount := total_before_extra_discount / 4
noncomputable def final_amount := total_before_extra_discount - extra_discount

theorem james_total_payment : final_amount = 60 := by
  sorry

end james_total_payment_l76_76854


namespace find_d_l76_76836

variables (A B C P : Type) [LinearOrderedField B]

variables (AB AC d c : B)
variables (AP_bisects_angle : ∀ α β : B, α = β)
variables (AB_eq_c : AB = c)
variables (BP_eq_d : ∀ α : B, α = d)
variables (PC_eq_75 : ∀ α : B, α = 75)
variables (AC_eq_150 : ∀ α : B, α = 150)

theorem find_d (c_value : B)(h : c_value = 100) : BP_eq_d d 50 :=
by 
  have c := c_value;
  have _ := AP_bisects_angle _ _;
  have _ := AB_eq_c;
  have _ := BP_eq_d;
  have _ := PC_eq_75;
  have _ := AC_eq_150;
  have d := 50;
  sorry

end find_d_l76_76836


namespace proposition_contrapositive_same_truth_value_l76_76595

variable {P : Prop}

theorem proposition_contrapositive_same_truth_value (P : Prop) :
  (P → P) = (¬P → ¬P) := 
sorry

end proposition_contrapositive_same_truth_value_l76_76595


namespace committee_selection_l76_76636

theorem committee_selection:
  let total_candidates := 15
  let former_members := 6
  let positions := 4
  let total_combinations := Nat.choose total_candidates positions
  let non_former_candidates := total_candidates - former_members
  let no_former_combinations := Nat.choose non_former_candidates positions
  total_combinations - no_former_combinations = 1239 :=
by
  let total_candidates := 15
  let former_members := 6
  let positions := 4
  let total_combinations := Nat.choose total_candidates positions
  let non_former_candidates := total_candidates - former_members
  let no_former_combinations := Nat.choose non_former_candidates positions
  sorry

end committee_selection_l76_76636


namespace san_antonio_to_austin_buses_l76_76638

theorem san_antonio_to_austin_buses 
  (sa_to_austin_departure_intervals : ∀ t : ℕ, t % 2 = 0 → Bus t "SanAntonio" "Austin")
  (austin_to_sa_departure_intervals : ∀ t : ℕ, t % 2 = 1 → Bus t "Austin" "SanAntonio")
  (travel_time : ∀ c1 c2 : City, Journey c1 c2 = 7)
  (same_route : ∀ b1 b2 : Bus, Route b1 = Route b2) :
  ∀ sa_bound : Bus, sa_bound.destination = "SanAntonio" → count_encounters sa_bound = 4 := 
sorry

end san_antonio_to_austin_buses_l76_76638


namespace chess_group_unique_pairings_l76_76943

theorem chess_group_unique_pairings:
  ∀ (players games : ℕ), players = 50 → games = 1225 →
  (∃ (games_per_pair : ℕ), games_per_pair = 1 ∧ (∀ p: ℕ, p < players → (players - 1) * games_per_pair = games)) :=
by
  sorry

end chess_group_unique_pairings_l76_76943


namespace find_r_l76_76837

theorem find_r 
  (r RB QC : ℝ)
  (angleA : ℝ)
  (h0 : RB = 6)
  (h1 : QC = 4)
  (h2 : angleA = 90) :
  (r + 6) ^ 2 + (r + 4) ^ 2 = 10 ^ 2 → r = 2 := 
by 
  sorry

end find_r_l76_76837


namespace common_ratio_q_l76_76704

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => geom_seq a q n * q

def sum_geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => sum_geom_seq a q n + geom_seq a q (n + 1)

theorem common_ratio_q (a q : α) (hq : 0 < q) (h_inc : ∀ n, geom_seq a q n < geom_seq a q (n + 1))
  (h1 : geom_seq a q 1 = 2)
  (h2 : sum_geom_seq a q 2 = 7) :
  q = 2 :=
sorry

end common_ratio_q_l76_76704


namespace finite_discrete_points_3_to_15_l76_76465

def goldfish_cost (n : ℕ) : ℕ := 18 * n

theorem finite_discrete_points_3_to_15 : 
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → 
  ∃ (C : ℕ), C = goldfish_cost n ∧ ∃ (x : ℕ), (n, C) = (x, goldfish_cost x) :=
by
  sorry

end finite_discrete_points_3_to_15_l76_76465


namespace not_diff_of_squares_count_l76_76336

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l76_76336


namespace maximum_discount_rate_l76_76172

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76172


namespace tan_alpha_minus_beta_neg_one_l76_76368

theorem tan_alpha_minus_beta_neg_one 
  (α β : ℝ)
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_neg_one_l76_76368


namespace no_i_satisfies_condition_l76_76818

def sumOfDivisors (n : ℕ) : ℕ := ∑ d in (finset.range (n + 1)).filter (λ d, n % d = 0), d

theorem no_i_satisfies_condition :
  ∀ i : ℕ, (1 ≤ i ∧ i ≤ 10000) → sumOfDivisors i ≠ 1 + 2 * nat.sqrt i + i :=
by {
  intros i hi,
  sorry
}

end no_i_satisfies_condition_l76_76818


namespace permutation_reversal_cycle_l76_76111

def is_reversal {α : Type*} (n : ℕ) (a b : list α) (k : ℕ) : Prop :=
  k ≤ n ∧ ((∀ i, 1 ≤ i ∧ i ≤ k → b[i] = a[k+1-i]) ∧ 
            ∀ i, k+1 ≤ i ∧ i ≤ n → b[i] = a[i])

theorem permutation_reversal_cycle (n : ℕ) (hn : n ≥ 2) :
  ∃ (P : ℕ → list (ℕ)), 
    (∀ i, P i ∈ list.permutations (list.range (n+1))) ∧
    (∀ i, ∃ k, is_reversal n (P i) (P (i+1)) k) ∧
    P (n! + 1) = P 1 := sorry

end permutation_reversal_cycle_l76_76111


namespace usual_time_is_60_l76_76989

variable (S T T' D : ℝ)

-- Defining the conditions
axiom condition1 : T' = T + 12
axiom condition2 : D = S * T
axiom condition3 : D = (5 / 6) * S * T'

-- The theorem to prove
theorem usual_time_is_60 (S T T' D : ℝ) 
  (h1 : T' = T + 12)
  (h2 : D = S * T)
  (h3 : D = (5 / 6) * S * T') : T = 60 := 
sorry

end usual_time_is_60_l76_76989


namespace arithmetic_sequence_properties_l76_76227

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a1 d : ℤ) : ℕ → ℤ
| 0     := a1
| (n+1) := arithmetic_sequence a1 d n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

theorem arithmetic_sequence_properties (a1 d : ℤ) (h1 : d > 0 ∧ d ≤ 6) :
  (∀ d ≤ 6, ∃ a1, ∃ n, arithmetic_sequence a1 d n = 99) ∧
  (∀ d ≤ 6, ¬ (∃ a1, ∃ n, arithmetic_sequence a1 d n = 30)) ∧
  (∃ a1 d, d = 2 * a1 ∧ ∀ n : ℕ, sum_first_n_terms a1 d (2 * n) = 4 * sum_first_n_terms a1 d n) := sorry

end arithmetic_sequence_properties_l76_76227


namespace fully_filled_boxes_l76_76858

-- Define the total number of cards Joe has
def magic_cards := 33
def rare_cards := 28
def common_cards := 33

-- Define the maximum cards per box for each type
def max_magic_per_box := 8
def max_rare_per_box := 10
def max_common_per_box := 12

-- Calculate the number of fully filled boxes for each type
def full_magic_boxes := magic_cards / max_magic_per_box
def full_rare_boxes := rare_cards / max_rare_per_box
def full_common_boxes := common_cards / max_common_per_box

-- Total number of fully filled boxes
def total_filled_boxes := full_magic_boxes.floor + full_rare_boxes.floor + full_common_boxes.floor

theorem fully_filled_boxes : total_filled_boxes = 8 := 
by sorry

end fully_filled_boxes_l76_76858


namespace cats_owners_percentage_l76_76828

noncomputable def percentage_of_students_owning_cats (total_students : ℕ) (cats_owners : ℕ) : ℚ :=
  (cats_owners : ℚ) / (total_students : ℚ) * 100

theorem cats_owners_percentage (total_students : ℕ) (cats_owners : ℕ)
  (dogs_owners : ℕ) (birds_owners : ℕ)
  (h_total_students : total_students = 400)
  (h_cats_owners : cats_owners = 80)
  (h_dogs_owners : dogs_owners = 120)
  (h_birds_owners : birds_owners = 40) :
  percentage_of_students_owning_cats total_students cats_owners = 20 :=
by {
  -- We state the proof but leave it as sorry so it's an incomplete placeholder.
  sorry
}

end cats_owners_percentage_l76_76828


namespace chebyshev_substitution_even_chebyshev_substitution_odd_l76_76096

def T (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the first kind
def U (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the second kind

theorem chebyshev_substitution_even (k : ℕ) (α : ℝ) :
  T (2 * k) (Real.sin α) = (-1)^k * Real.cos ((2 * k) * α) ∧
  U ((2 * k) - 1) (Real.sin α) = (-1)^(k + 1) * (Real.sin ((2 * k) * α) / Real.cos α) :=
by
  sorry

theorem chebyshev_substitution_odd (k : ℕ) (α : ℝ) :
  T (2 * k + 1) (Real.sin α) = (-1)^k * Real.sin ((2 * k + 1) * α) ∧
  U (2 * k) (Real.sin α) = (-1)^k * (Real.cos ((2 * k + 1) * α) / Real.cos α) :=
by
  sorry

end chebyshev_substitution_even_chebyshev_substitution_odd_l76_76096


namespace valid_pairs_l76_76659

theorem valid_pairs :
  {p : ℕ × ℕ | let m := p.1, n := p.2 in m > 0 ∧ n > 0 ∧ (n^3 + 1) % (m^2 - 1) = 0} =
    {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (1, 2), (1, 3)} :=
by
  sorry

end valid_pairs_l76_76659


namespace sum_of_digits_of_N_l76_76218

theorem sum_of_digits_of_N (N : ℕ) (hN : N * (N + 1) / 2 = 3003) :
  (Nat.digits 10 N).sum = 14 :=
sorry

end sum_of_digits_of_N_l76_76218


namespace work_ratio_l76_76760

theorem work_ratio (M B : ℝ) 
  (h1 : 5 * (12 * M + 16 * B) = 1)
  (h2 : 4 * (13 * M + 24 * B) = 1) : 
  M / B = 2 := 
  sorry

end work_ratio_l76_76760


namespace range_of_x_range_of_a_l76_76012

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_of_x (h1 : a = 1) (h2 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem range_of_a (h : ∀ x, p x a → q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l76_76012


namespace complex_division_1_complex_division_2_l76_76640

theorem complex_division_1 :
  (1 - complex.i) * (1 + 2 * complex.i) / (1 + complex.i) = 2 - complex.i := 
by 
  -- Proof omitted
  sorry

theorem complex_division_2 :
  ((1 + 2 * complex.i) ^ 2 + 3 * (1 - complex.i)) / (2 + complex.i) = 
  3 - (6 / 5) * complex.i :=
by 
  -- Proof omitted
  sorry

end complex_division_1_complex_division_2_l76_76640


namespace soldiers_to_add_l76_76762

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76762


namespace max_discount_rate_l76_76129

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76129


namespace simplify_expression_l76_76898

theorem simplify_expression (x y : ℝ) : 
    3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := 
by 
    sorry

end simplify_expression_l76_76898


namespace linear_regression_eq_l76_76487

noncomputable def x_vals : List ℝ := [3, 7, 11]
noncomputable def y_vals : List ℝ := [10, 20, 24]

theorem linear_regression_eq :
  ∃ a b : ℝ, (a = 5.75) ∧ (b = 1.75) ∧ (∀ x, ∃ y, y = a + b * x) := sorry

end linear_regression_eq_l76_76487


namespace proof_problem_l76_76829

noncomputable def triangle_problem (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] 
  (angle : A -> B -> C -> ℝ)
  (length : A -> B -> ℝ) : Prop :=
  let angleC := angle A B C in
  let angleA := angle C B A in
  let angleEBC := angle E B C in
  let angleEDC := angle E D C in
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  D ≠ B ∧ B ≠ E ∧ E ≠ C ∧
  angle C = angle A + 90 ∧
  length A C = length A D ∧
  ∠ E B C = angle A ∧
  ∠ E D C = 1 / 2 * angle A

theorem proof_problem {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (angle : A → B → C → ℝ)
  (length : A → B → ℝ)
  (hAB : A ≠ B) (hBC : B ≠ C) (hCA : C ≠ A)
  (hDB : D ≠ B) (hBE : B ≠ E) (hEC : E ≠ C)
  (h1 : angle C = angle A + 90)
  (h2 : length A C = length A D)
  (h3 : angle E B C = angle A)
  (h4 : angle E D C = 1 / 2 * angle A) :
  angle C E D = angle A B C :=
sorry

end proof_problem_l76_76829


namespace number_of_centroid_positions_l76_76046

def has_equally_spaced_points (length width : ℕ) (num_points : ℕ) : Prop :=
  ∃ (points : list (ℕ × ℕ)), 
    length = 15 ∧ width = 10 ∧ num_points = 50 ∧ 
    points.length = 50 ∧
    (∀ (p : (ℕ × ℕ)), p ∈ points → p.1 ≤ 15 ∧ p.2 ≤ 10)

theorem number_of_centroid_positions (length width : ℕ) (num_points : ℕ) 
  (points : list (ℕ × ℕ)) 
  (h : has_equally_spaced_points length width num_points)
  : ∃ (num_centroids : ℕ), num_centroids = 1426 :=
by
  use 1426
  sorry

end number_of_centroid_positions_l76_76046


namespace binomial_coefficient_largest_imp_n_eq_10_l76_76395

theorem binomial_coefficient_largest_imp_n_eq_10 (n : ℕ) (h : 0 < n) 
    (h1 : ∀ r, 0 ≤ r → r ≠ 5 → binom n 5 > binom n r):
  n = 10 :=
sorry

end binomial_coefficient_largest_imp_n_eq_10_l76_76395


namespace quadratic_symmetry_l76_76815

theorem quadratic_symmetry (a : ℝ) : (∃ x y : ℝ, y = a * x^2 + 1 ∧ (x, y) = (1, 2)) → (∃ x y : ℝ, y = a * x^2 + 1 ∧ (x, y) = (-1, 2)) :=
by
sory

end quadratic_symmetry_l76_76815


namespace binomial_probability_l76_76309

noncomputable def binomial_distribution (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem binomial_probability :
  let ξ : ℕ → ℝ := binomial_distribution 3 1 (1 / 3)
  in ξ = 4 / 9 :=
by
  sorry

end binomial_probability_l76_76309


namespace trigonometric_identity_l76_76358

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 :=
sorry

end trigonometric_identity_l76_76358


namespace count_multiples_4_6_10_less_300_l76_76746

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l76_76746


namespace cube_face_expression_l76_76229

theorem cube_face_expression (a b c : ℤ) (h1 : 3 * a + 2 = 17) (h2 : 7 * b - 4 = 10) (h3 : a + 3 * b - 2 * c = 11) : 
  a - b * c = 5 :=
by sorry

end cube_face_expression_l76_76229


namespace opposite_of_neg_2_is_2_l76_76497

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l76_76497


namespace at_least_1996_collinear_l76_76698

-- Define the collinear relationship
def collinear (points : Set (fin 1997 → ℝ)) : Prop :=
∀ p1 p2 p3 : (fin 1997 → ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points 
→ ∃ l : ℝ, ∀ p : (fin 1997 → ℝ), p ∈ points → (p = p1) ∨ (p = p2) ∨ (p = p3) ∨ (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * p 1 + b * p 2 = c)

-- Define the main proposition
theorem at_least_1996_collinear (points : Set (fin 1997 → ℝ))
(h : ∀ s ⊆ points, card s = 4 → ∃ s' ⊆ s, card s' = 3 ∧ collinear s') :
  ∃ l : ℝ, ∃ large_subset : Set (fin 1997 → ℝ), card large_subset ≥ 1996 ∧ large_subset ⊆ points ∧ collinear large_subset :=
by {
  sorry,
}

end at_least_1996_collinear_l76_76698


namespace section_area_ratios_l76_76605

structure HexagonalPyramid :=
  (a : ℝ)
  (S A B C D E F : Point) -- Points defining the hexagonal pyramid

def divided_points (a : ℝ) (S A D : Point) :=
  let AD := 2 * a
  let P := AD / 4
  let Q := AD / 2
  let R := (3 * AD) / 4
  (P, Q, R)

theorem section_area_ratios (a : ℝ) (S A D : Point) :
  let (P, Q, R) := divided_points a S A D
  let h := 1 -- Assume height of pyramid is normalized to 1
  let area1 := (3/4)^2
  let area2 := (1/2)^2
  let area3 := (1/4)^2
  let ratio := [area1, area2, area3].map (λ x => x * 16)
  ratio = [9, 4, 1] :=
by
  sorry

end section_area_ratios_l76_76605


namespace disqualified_team_participants_l76_76838

theorem disqualified_team_participants
  (initial_teams : ℕ) (initial_avg : ℕ) (final_teams : ℕ) (final_avg : ℕ)
  (total_initial : ℕ) (total_final : ℕ) :
  initial_teams = 9 →
  initial_avg = 7 →
  final_teams = 8 →
  final_avg = 6 →
  total_initial = initial_teams * initial_avg →
  total_final = final_teams * final_avg →
  total_initial - total_final = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end disqualified_team_participants_l76_76838


namespace number_of_red_parrots_l76_76025

-- Defining the conditions from a)
def fraction_yellow_parrots : ℚ := 2 / 3
def total_birds : ℕ := 120

-- Stating the theorem we want to prove
theorem number_of_red_parrots (H1 : fraction_yellow_parrots = 2 / 3) (H2 : total_birds = 120) : 
  (1 - fraction_yellow_parrots) * total_birds = 40 := 
by 
  sorry

end number_of_red_parrots_l76_76025


namespace opposite_of_neg_two_is_two_l76_76518

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76518


namespace min_soldiers_to_add_l76_76805

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76805


namespace max_f_on_0_pi_div_2_l76_76113

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x)^2 + 2 * cos x

theorem max_f_on_0_pi_div_2 : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ (Real.pi / 2) → f x ≤ 1 + (3 * Real.sqrt 3) / 2 := 
sorry

end max_f_on_0_pi_div_2_l76_76113


namespace total_delivery_time_l76_76217

def initial_coal_cars : Nat := 6
def initial_iron_cars : Nat := 12
def initial_wood_cars : Nat := 2

def distance_between_stations : Nat := 6
def travel_time_between_stations : Nat := 25

def max_coal_cars_per_station : Nat := 2
def max_iron_cars_per_station : Nat := 3
def max_wood_cars_per_station : Nat := 1

theorem total_delivery_time :
  let coal_stations := (initial_coal_cars + max_coal_cars_per_station - 1) / max_coal_cars_per_station in
  let iron_stations := (initial_iron_cars + max_iron_cars_per_station - 1) / max_iron_cars_per_station in
  let wood_stations := (initial_wood_cars + max_wood_cars_per_station - 1) / max_wood_cars_per_station in
  let max_stations := max (max coal_stations iron_stations) wood_stations in
  max_stations * travel_time_between_stations = 100 := by
  sorry

end total_delivery_time_l76_76217


namespace tan_alpha_minus_beta_neg_one_l76_76366

theorem tan_alpha_minus_beta_neg_one 
  (α β : ℝ)
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_neg_one_l76_76366


namespace range_of_a_l76_76322

noncomputable def sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then (1 / 3 - a) * n + 2 else a ^ (n - 7)

theorem range_of_a (a : ℝ) : (∀ n : ℕ, n > 0 → sequence a n > sequence a (n + 1)) → 1 / 2 < a ∧ a < 1 := by
  sorry

end range_of_a_l76_76322


namespace opposite_of_neg_two_l76_76549

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l76_76549


namespace teachers_in_the_middle_teachers_together_not_ends_teachers_not_ends_not_adjacent_l76_76276

variable (students : Fin 4) (teachers : Fin 2)

-- (1) Prove the number of seating arrangements when teachers must sit in the middle is 48
theorem teachers_in_the_middle : (4.perm * 2.perm = 48) :=
  sorry

-- (2) Prove the number of seating arrangements when teachers cannot sit at the ends, but must sit together is 144
theorem teachers_together_not_ends : (4.perm * 2.perm * 3 = 144) :=
  sorry

-- (3) Prove the number of seating arrangements when teachers cannot sit at the ends, nor can they sit adjacent to each other is 144
theorem teachers_not_ends_not_adjacent : (4.perm * (3.factorial / 2.factorial) = 144) :=
  sorry

end teachers_in_the_middle_teachers_together_not_ends_teachers_not_ends_not_adjacent_l76_76276


namespace cube_sum_is_integer_l76_76885

theorem cube_sum_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^3 + 1/a^3 = m :=
sorry

end cube_sum_is_integer_l76_76885


namespace area_of_triangle_ABC_l76_76845

theorem area_of_triangle_ABC (A B C D E F G: Point) (β : Real)
  (h_isosceles: AB = BC)
  (h_parallel: Parallel (Line.extend A C) (Line.extend D E))
  (h_segments: dist A D = 2 ∧ dist E C = 2 ∧ dist D E = 2)
  (h_midpoint_F: Midpoint F A C)
  (h_midpoint_G: Midpoint G E C)
  (h_angle: angle G F C = β):
  area ABC = (1 + 2 * cos (2 * β))^2 * tan (2 * β) :=
sorry

end area_of_triangle_ABC_l76_76845


namespace max_discount_rate_l76_76186

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76186


namespace length_of_BC_l76_76297

theorem length_of_BC (AD BD CD : ℝ) (BC : ℝ) 
  (h1 : AD = 20) 
  (h2 : BD = 15) 
  (h3 : CD = 29) 
  (right_triangle_ABD : AD^2 = BD^2 + AB^2)
  (right_triangle_ABC : AC^2 = AB^2 + BC^2) : 
  BC = 2 * Real.sqrt 254 :=
by
  have AB := Real.sqrt (AD^2 - BD^2)
  rw [h1, h2] at AB
  have AC := CD
  rw [h3] at AC
  sorry

end length_of_BC_l76_76297


namespace arithmetic_sequence_term_l76_76832

theorem arithmetic_sequence_term {a : ℕ → ℤ} 
  (h1 : a 4 = -4) 
  (h2 : a 8 = 4) : 
  a 12 = 12 := 
by 
  sorry

end arithmetic_sequence_term_l76_76832


namespace apple_in_B_l76_76945

-- Define the boxes and their statements
def box := ℕ
def A : box := 0
def B : box := 1
def C : box := 2

-- Define the statements on the boxes
def statement_A (apple_in : box) : Prop := apple_in = A
def statement_B (apple_in : box) : Prop := apple_in ≠ B
def statement_C (apple_in : box) : Prop := apple_in ≠ A

-- The main theorem to be proved
theorem apple_in_B (apple_in : box) :
  (statement_A apple_in ∧ ¬statement_B apple_in ∧ ¬statement_C apple_in) ∨
  (¬statement_A apple_in ∧ statement_B apple_in ∧ ¬statement_C apple_in) ∨
  (¬statement_A apple_in ∧ ¬statement_B apple_in ∧ statement_C apple_in) →
  apple_in = B :=
sorry

end apple_in_B_l76_76945


namespace opposite_of_neg_two_is_two_l76_76535

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76535


namespace negative_remainder_l76_76757

theorem negative_remainder (a : ℤ) (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end negative_remainder_l76_76757


namespace twelve_pirates_l76_76201

theorem twelve_pirates {x : ℕ} (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 -> (k * x / 12 ^ k) % 1 = 0) :
  (1925 : ℕ) = 1 / 12 * 2 / 12 * ... * 11 / 12 * x :=
sorry

end twelve_pirates_l76_76201


namespace stage_performance_median_mode_better_stage_performance_optimal_selection_l76_76200

-- Definition of the heights of the students
def heights : List ℕ := [161, 162, 162, 164, 165, 165, 165, 166, 166, 167, 168, 168, 170, 172, 172, 175]
def median_heights (l : List ℕ) := (l.nth (l.length / 2 - 1) + l.nth (l.length / 2)) / 2
def mode_heights (l : List ℕ) := l.groupBy id |> List.maxBy (·.length) |> List.head

-- Group definitions
def groupA : List ℕ := [162, 165, 165, 166, 166]
def groupB : List ℕ := [161, 162, 164, 165, 175]

-- Variance calculation helper
def variance (l : List ℕ) :=
  let avg := l.sum / l.length
  (l.map (λ x => (x - avg) ^ 2)).sum / l.length

-- The preselected students and variance limit
def selected_students : List ℕ := [168, 168, 172]
def variance_limit : ℚ := 32 / 9

-- Lean 4 Statement
theorem stage_performance_median_mode : 
  median_heights heights = 166 ∧ mode_heights heights = 165 :=
  sorry

theorem better_stage_performance : 
  variance groupA < variance groupB :=
  sorry

theorem optimal_selection : ∃ (height1 height2 : ℕ),
  let new_group := selected_students ++ [height1, height2]
  variance new_group < variance_limit ∧
  (new_group.sum / new_group.length) = 170 :=
  sorry

end stage_performance_median_mode_better_stage_performance_optimal_selection_l76_76200


namespace total_earnings_l76_76592

theorem total_earnings (zachary_games : ℕ) (price_per_game : ℝ) (jason_percentage_increase : ℝ) (ryan_extra : ℝ)
  (h1 : zachary_games = 40) (h2 : price_per_game = 5) (h3 : jason_percentage_increase = 0.30) (h4 : ryan_extra = 50) :
  let z_earnings := zachary_games * price_per_game,
      j_earnings := z_earnings * (1 + jason_percentage_increase),
      r_earnings := j_earnings + ryan_extra
  in z_earnings + j_earnings + r_earnings = 770 := by
  sorry

end total_earnings_l76_76592


namespace invested_sum_l76_76198

/-- Given conditions -/
variables {P R1 R2 T diff : ℕ}

/-- Rates of interest and time period -/
def interest_data (R1 R2 T diff : ℕ) : Prop :=
  R1 = 18 ∧ R2 = 12 ∧ T = 2 ∧ diff = 840

/-- Simple interest calculations -/
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

/-- Proof that the invested sum P is 7000 -/
theorem invested_sum (h: interest_data R1 R2 T diff) : P = 7000 :=
by
  sorry

end invested_sum_l76_76198


namespace sqrt_seven_l76_76939

theorem sqrt_seven (x : ℝ) : x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end sqrt_seven_l76_76939


namespace maximum_expr_value_l76_76835

theorem maximum_expr_value :
  ∃ (x y e f : ℕ), (e = 4 ∧ x = 3 ∧ y = 2 ∧ f = 0) ∧
  (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4) ∧
  (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ∧
  (y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4) ∧
  (f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4) ∧
  (e ≠ x ∧ e ≠ y ∧ e ≠ f ∧ x ≠ y ∧ x ≠ f ∧ y ≠ f) ∧
  (e * x^y - f = 36) :=
by
  sorry

end maximum_expr_value_l76_76835


namespace max_discount_rate_l76_76181

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76181


namespace probability_A_and_B_same_last_hour_l76_76575
open Classical

-- Define the problem conditions
def attraction_count : ℕ := 6
def total_scenarios : ℕ := attraction_count * attraction_count
def favorable_scenarios : ℕ := attraction_count

-- Define the probability calculation
def probability_same_attraction : ℚ := favorable_scenarios / total_scenarios

-- The proof problem statement
theorem probability_A_and_B_same_last_hour : 
  probability_same_attraction = 1 / 6 :=
sorry

end probability_A_and_B_same_last_hour_l76_76575


namespace pyramid_volume_l76_76211

noncomputable def area_of_square_base (surface_area : ℝ) (triangle_area_ratio : ℝ) : ℝ :=
  let x := (surface_area * 3) / 7 in x

noncomputable def side_length_of_square (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def height_of_pyramid (side_length : ℝ) (triangle_area : ℝ) : ℝ :=
  let h := (triangle_area * 2) / side_length in h

noncomputable def vertical_height (slant_height : ℝ) (half_diagonal : ℝ) : ℝ :=
  real.sqrt (slant_height ^ 2 - half_diagonal ^ 2)

noncomputable def volume_of_pyramid (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem pyramid_volume :
  let total_surface_area := 486
  let triangle_area_ratio := 1 / 3
  let base_area := area_of_square_base total_surface_area triangle_area_ratio
  let side_length := side_length_of_square base_area
  let triangle_area := base_area * triangle_area_ratio
  let slant_height := height_of_pyramid side_length triangle_area
  let half_diagonal := side_length / 2 * real.sqrt 2
  let height := vertical_height slant_height half_diagonal
  volume_of_pyramid base_area height = 310.5 * real.sqrt 207 :=
by sorry

end pyramid_volume_l76_76211


namespace triangle_ABC_problem_l76_76718

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ := a + b + c

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = 3) 
  (h2 : B = π / 3) 
  (area : ℝ)
  (h3 : (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3) :

  perimeter_of_triangle a b c = 18 ∧ 
  Real.sin (2 * A) = 39 * Real.sqrt 3 / 98 := 
by 
  sorry

end triangle_ABC_problem_l76_76718


namespace pentagon_area_and_irregularity_l76_76011

theorem pentagon_area_and_irregularity
  (ABCDE : convex_pentagon)
  (hABC : ABCDE.area_of_triangle ABC = 1)
  (hBCD : ABCDE.area_of_triangle BCD = 1)
  (hCDE : ABCDE.area_of_triangle CDE = 1)
  (hDEA : ABCDE.area_of_triangle DEA = 1)
  (hEAB : ABCDE.area_of_triangle EAB = 1) :
  ABCDE.total_area = (5 + Real.sqrt 5) / 2 ∧ ¬ ABCDE.is_regular := 
sorry

end pentagon_area_and_irregularity_l76_76011


namespace number_of_integer_values_of_n_l76_76271

theorem number_of_integer_values_of_n (n : ℤ) : 
  { n : ℤ | ∃ m : ℤ, 5400 * (3^(n : ℕ)) * (2^(-2 * (n : ℕ))) = m }.to_finset.card = 2 :=
by
  sorry

end number_of_integer_values_of_n_l76_76271


namespace area_of_DEF_l76_76621

variable (t4_area t5_area t6_area : ℝ) (a_DEF : ℝ)

def similar_triangles_area := (t4_area = 1) ∧ (t5_area = 16) ∧ (t6_area = 36)

theorem area_of_DEF 
  (h : similar_triangles_area t4_area t5_area t6_area) :
  a_DEF = 121 := sorry

end area_of_DEF_l76_76621


namespace part1_part2_l76_76708

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l76_76708


namespace series_sum_equals_102_l76_76649

theorem series_sum_equals_102 :
  (\sum k in Finset.range 50, (2 + 4 * (k + 1)) / (3^(50 - k))) = 102 :=
by
  sorry

end series_sum_equals_102_l76_76649


namespace count_divisible_by_4_6_10_l76_76743

theorem count_divisible_by_4_6_10 :
  (card {n : ℕ | n < 300 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0}) = 4 :=
by 
  sorry

end count_divisible_by_4_6_10_l76_76743


namespace rank_abc_l76_76003

noncomputable def a := 6 ^ 0.4
noncomputable def b := real.log 0.5 / real.log 0.4
noncomputable def c := real.log 0.4 / real.log 8

theorem rank_abc : c < b ∧ b < a :=
by
  -- proof steps go here
  sorry

end rank_abc_l76_76003


namespace probability_last_digit_7_l76_76418

def lastDigit (n: ℕ) : ℕ := n % 10

theorem probability_last_digit_7 (a : ℕ) 
  (h1 : a ∈ finset.range 101 \ finset.singleton 0) :
  let prob := (finset.card (finset.filter (λ x, lastDigit (3^x) = 7) (finset.range 101))) / (finset.card (finset.range 101)) in
  prob = (1 / 4 : ℚ) :=
by
  sorry

end probability_last_digit_7_l76_76418


namespace area_square_II_is_correct_l76_76478

variable (a b : ℝ)

-- The conditions given in the problem
def diagonal_square_I : ℝ := 2 * a + 3 * b
def area_square_I : ℝ := (diagonal_square_I a b / Real.sqrt 2)^2
def area_square_II : ℝ := (area_square_I a b)^3

-- The theorem stating our problem
theorem area_square_II_is_correct : area_square_II a b = (2 * a + 3 * b)^6 / 8 := by sorry

end area_square_II_is_correct_l76_76478


namespace babylon_game_proof_l76_76994

section BabylonGame

-- Defining the number of holes on the sphere
def number_of_holes : Nat := 26

-- The number of 45° angles formed by the pairs of rays
def num_45_degree_angles : Nat := 40

-- The number of 60° angles formed by the pairs of rays
def num_60_degree_angles : Nat := 48

-- The other angles that can occur between pairs of rays
def other_angles : List Real := [31.4, 81.6, 90]

-- Constructs possible given the conditions
def constructible (shape : String) : Bool :=
  shape = "regular tetrahedron" ∨ shape = "regular octahedron"

-- Constructs not possible given the conditions
def non_constructible (shape : String) : Bool :=
  shape = "joined regular tetrahedrons"

-- Proof problem statement
theorem babylon_game_proof :
  (number_of_holes = 26) →
  (num_45_degree_angles = 40) →
  (num_60_degree_angles = 48) →
  (other_angles = [31.4, 81.6, 90]) →
  (constructible "regular tetrahedron" = True) →
  (constructible "regular octahedron" = True) →
  (non_constructible "joined regular tetrahedrons" = True) :=
  by
    sorry

end BabylonGame

end babylon_game_proof_l76_76994


namespace ones_digit_of_p_is_3_l76_76687

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l76_76687


namespace tangent_identity_l76_76372

theorem tangent_identity (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tangent_identity_l76_76372


namespace opposite_of_neg_two_l76_76526

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76526


namespace fibonacci_pos_int_range_l76_76862

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib (n + 1) + fib n

-- Problem statement in Lean
theorem fibonacci_pos_int_range :
  ∀ n : ℕ, (n > 0 → (∃ a : ℕ, 0 < a ∧ fib n ≤ a ∧ a ≤ fib (n + 1)
    ∧ a * (∑ i in finset.range n,
            1 / (finset.prod (finset.range i.succ) fib)) ∈ ℤ)) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by
  sorry

end fibonacci_pos_int_range_l76_76862


namespace derivative_of_f_l76_76477

-- Define the function f(x) = (2 * π * x)^2
def f (x : ℝ) : ℝ := (2 * real.pi * x) ^ 2

-- Statement: Prove that the derivative of f(x) is 8 * π^2 * x
theorem derivative_of_f (x : ℝ) : deriv f x = 8 * real.pi^2 * x := sorry

end derivative_of_f_l76_76477


namespace canal_cross_section_area_l76_76053

/-- Definitions of the conditions -/
def top_width : Real := 6
def bottom_width : Real := 4
def depth : Real := 257.25

/-- Proof statement -/
theorem canal_cross_section_area : 
  (1 / 2) * (top_width + bottom_width) * depth = 1286.25 :=
by
  sorry

end canal_cross_section_area_l76_76053


namespace conditional_probability_of_events_l76_76049

/-- Define the three routes for the schools -/
inductive Routes
| WanhuaRock
| Wangxianling
| Xiongying

open Routes

/-- Define the events A and B -/
def EventA (route_A route_B : Routes) : Prop :=
  route_A = Wangxianling ∨ route_B = Wangxianling

def EventB (route_A route_B : Routes) : Prop :=
  route_A ≠ route_B

/-- The proof problem to prove P(B|A) = 4/5 -/
theorem conditional_probability_of_events :
  (∑ route_A route_B, ((if EventA route_A route_B ∧ EventB route_A route_B then 1 else 0) / 9.0)) /
  (∑ route_A route_B, ((if EventA route_A route_B then 1 else 0) / 9.0)) = 4 / 5 :=
by
  sorry

end conditional_probability_of_events_l76_76049


namespace sum_sqrt_reciprocal_geq_l76_76863

theorem sum_sqrt_reciprocal_geq (n : ℕ) (x : Fin n → ℝ) 
    (h1 : ∀ i, 0 < x i) 
    (h2 : 2 ≤ n) 
    (h3 : (Finset.univ.sum x) = 1) :
    (Finset.univ.sum (λ i, 1 / Real.sqrt (1 - x i))) ≥ n * Real.sqrt (n / (n - 1)) :=
by
  sorry

end sum_sqrt_reciprocal_geq_l76_76863


namespace sqrt_cos_eq_sin_half_l76_76376

theorem sqrt_cos_eq_sin_half (α : ℝ) (h : π < α ∧ α < 3 * π / 2) :
  sqrt (1 / 2 + 1 / 2 * sqrt (1 / 2 + 1 / 2 * cos (2 * α))) = sin (α / 2) :=
sorry

end sqrt_cos_eq_sin_half_l76_76376


namespace meeting_time_is_two_hours_l76_76443

-- Define the conditions
def distance := 65  -- Distance in kilometers
def total_speed := 32.5  -- Sum of speeds in kilometers per hour

-- Define a statement to prove the time it takes for them to meet
def meet_time (d : ℝ) (v : ℝ) : ℝ := d / v

-- Prove that given the conditions, the meeting time is equal to 2 hours
theorem meeting_time_is_two_hours : meet_time distance total_speed = 2 := by
  sorry

end meeting_time_is_two_hours_l76_76443


namespace min_soldiers_to_add_l76_76806

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76806


namespace leon_older_than_aivo_in_months_l76_76683

theorem leon_older_than_aivo_in_months
    (jolyn therese aivo leon : ℕ)
    (h1 : jolyn = therese + 2)
    (h2 : therese = aivo + 5)
    (h3 : jolyn = leon + 5) :
    leon = aivo + 2 := 
sorry

end leon_older_than_aivo_in_months_l76_76683


namespace player_A_wins_l76_76429

-- Define the game conditions
def game_conditions (n : ℕ) : Prop :=
  n ≥ 3 

-- Define the strategy winning condition
def player_A_winning_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ) → (ℕ → ℕ), 
    (∀ state : ℕ, ∀ moveA, ∀ moveB,
      (strategy moveA moveB) state = true)

-- The theorem statement for proving Player A always wins
theorem player_A_wins (n : ℕ) (hn : game_conditions n) : player_A_winning_strategy n := sorry

end player_A_wins_l76_76429


namespace max_n_eq_6_l76_76056

-- Define the max_n property as the maximum value such that ((n!)!)! divides (2021!)!
def max_n (n max_n : ℕ) : Prop := ((factorial (factorial n))!) ∣ (factorial (factorial max_n))

theorem max_n_eq_6 : ∃ (n : ℕ), max_n n 2021 ∧ (∀ m : ℕ, max_n m 2021 → m ≤ n) ∧ n = 6 :=
by
  sorry

end max_n_eq_6_l76_76056


namespace equal_share_candy_l76_76751

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l76_76751


namespace find_principal_l76_76599

variables (SI : ℝ) (R : ℝ) (T : ℝ)

def principal (SI R T : ℝ) : ℝ := SI / (R * T / 100)

theorem find_principal (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : principal SI R T = 8925 := by
  rw [hSI, hR, hT, principal]
  norm_num
  sorry

end find_principal_l76_76599


namespace minimum_soldiers_to_add_l76_76795

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76795


namespace opposite_of_neg_two_is_two_l76_76539

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l76_76539


namespace count_irrationals_l76_76632

theorem count_irrationals : 
  let nums := [22 / 7, Real.sqrt 5, - Real.cbrt 8, Real.pi, 2023]
  nums.count (λ x, ¬ IsRational x) = 2 := by
  sorry

end count_irrationals_l76_76632


namespace initial_bacteria_count_is_one_l76_76050

noncomputable theory

-- Define initial conditions
def bacteria_growth (n : ℕ) (t : ℕ) : ℕ := n * 5^t

-- Define the constants and relationship in the problem
def number_of_intervals : ℕ := 8
def final_bacteria_count : ℕ := 590490

-- The main theorem to prove
theorem initial_bacteria_count_is_one : ∃ n : ℕ, bacteria_growth n number_of_intervals = final_bacteria_count ∧ n = 1 :=
sorry

end initial_bacteria_count_is_one_l76_76050


namespace min_soldiers_to_add_l76_76808

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76808


namespace count_divisible_by_4_6_10_l76_76744

theorem count_divisible_by_4_6_10 :
  (card {n : ℕ | n < 300 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0}) = 4 :=
by 
  sorry

end count_divisible_by_4_6_10_l76_76744


namespace simplify_cube_root_8000_l76_76587

theorem simplify_cube_root_8000 : ∃ c d : ℕ, c * (d^(1/3: ℚ)) = (8000^(1/3: ℚ)) ∧ (d = 1) ∧ (c + d = 21) :=
by
  use 20
  use 1
  split
  -- Proof that 20 * (1^(1/3)) = (8000^(1/3))
  sorry
  split
  -- Proof that d = 1
  rfl
  -- Proof that c + d = 21
  rfl

end simplify_cube_root_8000_l76_76587


namespace solve_inequality_range_of_a_l76_76324

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define the set A
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- First part: Solve the inequality f(x) ≤ 3a^2 + 1 when a ≠ 0
-- Solution would be translated in a theorem
theorem solve_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x a ≤ 3 * a^2 + 1 → if a > 0 then -a ≤ x ∧ x ≤ 3 * a else -3 * a ≤ x ∧ x ≤ a :=
sorry

-- Second part: Find the range of a if there exists no x0 ∈ A such that f(x0) ≤ A is false
theorem range_of_a (a : ℝ) :
  (∀ x ∈ A, f x a > 0) ↔ a < 1 :=
sorry

end solve_inequality_range_of_a_l76_76324


namespace find_x_for_orthogonal_vectors_l76_76252

theorem find_x_for_orthogonal_vectors :
  ∃ x : ℚ, let v1 := (⟨3, 4, -1⟩ : ℚ × ℚ × ℚ) in
           let v2 := (⟨x, -2, 5⟩ : ℚ × ℚ × ℚ) in
           (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0) → x = 13/3 :=
by
  sorry

end find_x_for_orthogonal_vectors_l76_76252


namespace count_S_below_2008_l76_76426

def S : Set ℕ := 
  {x | let rec is_in (x : ℕ) : Bool := 
    match x with
    | 0       => true
    | (3 * n) => is_in n
    | (3 * n + 1) => is_in n
    | _ => false
  in is_in x}

theorem count_S_below_2008 : 
  {n : ℕ | n ∈ S ∧ n < 2008}.card = 128 := 
by
  sorry

end count_S_below_2008_l76_76426


namespace opposite_of_neg2_l76_76506

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l76_76506


namespace smallest_number_increased_by_nine_divisible_by_8_11_24_l76_76990

theorem smallest_number_increased_by_nine_divisible_by_8_11_24 :
  ∃ x : ℕ, (x + 9) % 8 = 0 ∧ (x + 9) % 11 = 0 ∧ (x + 9) % 24 = 0 ∧ x = 255 :=
by
  sorry

end smallest_number_increased_by_nine_divisible_by_8_11_24_l76_76990


namespace real_part_of_complex_product_l76_76561

noncomputable def i : ℂ := complex.I -- define the imaginary unit in complex numbers

-- defining the problem statement
theorem real_part_of_complex_product : 
  (complex.re ((1 - i) * (2 + 3 * i)) = 5) :=
by
  sorry

end real_part_of_complex_product_l76_76561


namespace school_visits_arrangement_l76_76634

theorem school_visits_arrangement : 
  ∃ (arrangement : Nat), 
  (let total_days := 7 in 
  let consecutive_days_A := 2 in 
  let days_remain := total_days - consecutive_days_A in 
  let ways_to_arrange_A := total_days - consecutive_days_A + 1 in 
  let ways_to_arrange_BC := ways_to_arrange_A * (days_remain.choose 2) in 
  arrangement = ways_to_arrange_BC) ∧ arrangement = 120 := 
sorry

end school_visits_arrangement_l76_76634


namespace sum_reciprocal_squares_lt_half_l76_76891

theorem sum_reciprocal_squares_lt_half (n : ℕ) (h : 0 < n) :
  (∑ k in finset.Ico (n + 1) (2 * n + 1), (1 : ℝ) / k) ^ 2 < 1 / 2 :=
sorry

end sum_reciprocal_squares_lt_half_l76_76891


namespace opposite_of_neg_two_l76_76525

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76525


namespace distances_sum_to_one_l76_76884

variables {A B C M : Type}
variables {a b c x y z h_a h_b h_c : ℝ}
variables [Triangle ABC] [Point_inside_triangle M ABC]

theorem distances_sum_to_one 
  (h₁ : altitude_from A = h_a)
  (h₂ : altitude_from B = h_b)
  (h₃ : altitude_from C = h_c)
  (d₁ : distance_from M to BC = x)
  (d₂ : distance_from M to AC = y)
  (d₃ : distance_from M to AB = z) :
  (x / h_a) + (y / h_b) + (z / h_c) = 1 :=
begin
  sorry
end

end distances_sum_to_one_l76_76884


namespace solve_for_x_l76_76662

theorem solve_for_x (x : ℝ) (hx : x^(1/10) * (x^(3/2))^(1/10) = 3) : x = 9 :=
sorry

end solve_for_x_l76_76662


namespace correct_calculated_value_l76_76103

theorem correct_calculated_value (n : ℕ) (h : n + 9 = 30) : n + 7 = 28 :=
by
  sorry

end correct_calculated_value_l76_76103


namespace correct_statements_l76_76596

-- Define the problem's elements and conditions
noncomputable def Plane (α : Type*) := α → α → Prop
variable {α : Type*}

-- Define properties of planes
def parallel_to_line (P : Plane α) (l : α → Prop) : Prop := ∀ {x y}, l x → l y → P x y
def parallel_to_plane (P1 P2 : Plane α) := ∀ {x y}, P1 x y → P2 x y
def perpendicular_to_line (P : Plane α) (l : α → Prop) : Prop := ∀ {x}, l x → ∀ {y}, not (P x y)
def equal_angles_with_line (P : Plane α) (l : α → Prop) (angle : α → ℝ) : Prop := ∀ {x}, l x → angle x = angle y

-- Statements to prove as per the problem
theorem correct_statements :
  (∀ (P1 P2 : Plane α) (l : α → Prop), parallel_to_line P1 l → parallel_to_line P2 l → parallel_to_plane P1 P2) ∧
  (∀ (P1 P2 : Plane α), parallel_to_plane P1 P2 → parallel_to_plane P2 P1) ∧
  ¬ (∀ (P1 P2 : Plane α) (l : α → Prop), perpendicular_to_line P1 l → perpendicular_to_line P2 l → parallel_to_plane P1 P2) ∧
  ¬ (∀ (P1 P2 : Plane α) (l : α → Prop) (angle : α → ℝ), equal_angles_with_line P1 l angle → equal_angles_with_line P2 l angle → parallel_to_plane P1 P2) :=
by sorry

end correct_statements_l76_76596


namespace angle_CDE_in_quadrilateral_l76_76826

theorem angle_CDE_in_quadrilateral 
  (A B C D : Type) [AD : A 'orthogonal' D] [BC : B 'orthogonal' C]
  [angle_AEB: 'angle' A E B = 50] 
  [angle_BED: 'angle' B E D = 60] 
  [angle_BDE: 'angle' B D E = 30] : 
  'angle' C D E = 40 :=
begin
  sorry
end

end angle_CDE_in_quadrilateral_l76_76826


namespace center_value_is_18_l76_76880

-- Define a type for the grid
def grid := Matrix (Fin 3) (Fin 3) ℕ

-- Define the conditions for the grid transformation
def adjacent_sum (m : grid) (i j : Fin 3) : ℕ :=
  (if i > 0 then m (i - 1) j else 0) + 
  (if i < 2 then m (i + 1) j else 0) +
  (if j > 0 then m i (j - 1) else 0) + 
  (if j < 2 then m i (j + 1) else 0)

-- The transformed grid
def transformed_grid (m : grid) : grid :=
  λ i j, adjacent_sum m i j

noncomputable def original_center_value : Prop :=
  ∀ (m : grid), 
  (∀ i j, i ≠ j → m i j ≠ m i j) → -- distinct positive integers
  (∑ i j, m i j = 74) →             -- summing to 74
  (∃ (count_23 : ℕ), count_23 = 4 ∧ (∀ i j, (transformed_grid m) i j = 23 → count_23 = count_23 + 1)) → 
  (m 1 1 = 18)

-- The theorem statement
theorem center_value_is_18 : original_center_value :=
sorry

end center_value_is_18_l76_76880


namespace translation_of_sin_is_cos_add_2_l76_76484

theorem translation_of_sin_is_cos_add_2 :
  ∀ x : ℝ, let g := λ x : ℝ, cos x + 2 in 
  (∃ f : ℝ → ℝ, f = sin ∧ (λ (x' y' : ℝ), (x' = x - π / 2 ∧ y' = f x + 2)) = (x, g x)) :=
by
  intro x
  let g := λ x : ℝ, cos x + 2
  existsi (λ x, sin x)
  split
  { 
    refl 
  }
  {
    ext
    split
    {
      intro h
      cases h
      split
      { 
        intro
        refl 
      }
      { 
        intro
        refl 
      }
    }
    {
      intro h
      cases h
      split
      { 
        exact h_left 
      }
      { 
        exact h_right 
      }
    }
  }

end translation_of_sin_is_cos_add_2_l76_76484


namespace ones_digit_of_p_is_3_l76_76688

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l76_76688


namespace problem_statement_l76_76644

def sin_60 := Real.sin (60 * Real.pi / 180)
def inv_neg_half_sq := (-1 / 2 : ℝ) ^ (-2)
def abs_two_minus_sqrt_three := abs (2 - Real.sqrt 3)
def sqrt_12 := Real.sqrt 12
def expression := 2 * sin_60 + inv_neg_half_sq - abs_two_minus_sqrt_three - sqrt_12

theorem problem_statement : expression = 2 := by
  -- Forgoing the proof for template adherence
  sorry

end problem_statement_l76_76644


namespace simplify_expression_l76_76457

theorem simplify_expression : (625:ℝ)^(1/4) * (256:ℝ)^(1/2) = 80 := 
by 
  sorry

end simplify_expression_l76_76457


namespace integer_root_polynomial_l76_76666

theorem integer_root_polynomial (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 7 = 0) ↔ (a = -71 ∨ a = -27 ∨ a = -11 ∨ a = 9) :=
begin
  sorry,
end

end integer_root_polynomial_l76_76666


namespace sin_2α_plus_pi_over_3_l76_76304

variable {α : ℝ}

-- α is an acute angle
axiom acute_angle (h : 0 < α ∧ α < π / 2)

-- Given condition
axiom cos_condition : cos (α + π / 4) = sqrt 5 / 5

-- Statement to prove
theorem sin_2α_plus_pi_over_3 :
  sin (2 * α + π / 3) = (4 * sqrt 3 + 3) / 10 := sorry

end sin_2α_plus_pi_over_3_l76_76304


namespace nails_per_plank_l76_76273

theorem nails_per_plank {total_nails planks : ℕ} (h1 : total_nails = 4) (h2 : planks = 2) :
  total_nails / planks = 2 := by
  sorry

end nails_per_plank_l76_76273


namespace spoiled_apples_l76_76613

theorem spoiled_apples (S G : ℕ) (h1 : S + G = 8) (h2 : (G * (G - 1)) / 2 = 21) : S = 1 :=
by
  sorry

end spoiled_apples_l76_76613


namespace maximize_product_l76_76577

-- Define the range of digits to be used.
def digits : List ℕ := [1, 5, 6, 8, 9]

-- Define the condition that the digits should be used exactly once.
def use_digits_exactly_once (num1 num2 : ℕ) : Prop := 
  (num1.digits ++ num2.digits).perm digits

-- Define the three-digit number and the two-digit number.
def three_digit_number := 851
def two_digit_number := 96

-- The main theorem: 851 * 96 gives the maximum product when digits are used exactly once.
theorem maximize_product : ∃ (a b : ℕ), three_digit_number = 851 ∧ two_digit_number = 96 
                             ∧ use_digits_exactly_once three_digit_number two_digit_number 
                             := by
  existsi 851
  existsi 96
  split
  · rfl  -- a equals 851
  split
  · rfl  -- b equals 96
  · sorry -- Proof that the digits are used exactly once and maximize the product

end maximize_product_l76_76577


namespace minimum_soldiers_to_add_l76_76789

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76789


namespace conic_curve_equations_l76_76977

noncomputable def is_ellipse (m n : ℝ) : Prop :=
  (m * (3 / 2)^2 + n * (-√3)^2 = 1) ∧ (m * (-√2)^2 + n * 2^2 = 1)

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 12 = 1

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

theorem conic_curve_equations :
  (∃ m n : ℝ, is_ellipse m n ∧ ellipse_equation = (λ x y, x^2 / (1 / 3) + y^2 / (1 / 12) = 1)) ∧
  (hyperbola_equation = (λ x y, x^2 / 1 - y^2 / 4 = 1)) :=
sorry

end conic_curve_equations_l76_76977


namespace vector_dot_product_solution_l76_76327

variable (x : ℝ)
def a := (2, -1)
def b := (3, x)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem vector_dot_product_solution (h : dot_product a b = 3) : x = 3 :=
by
  sorry

end vector_dot_product_solution_l76_76327


namespace number_of_lattice_points_l76_76203

def is_lattice_point (x y : ℤ) : Prop := x^2 - y^2 = 53

theorem number_of_lattice_points : {p : ℤ × ℤ | is_lattice_point p.1 p.2}.to_finset.card = 4 :=
sorry

end number_of_lattice_points_l76_76203


namespace max_discount_rate_l76_76164

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76164


namespace solve_equation_l76_76993

noncomputable def f (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs x - 8) - 4) - 2) - 1)

noncomputable def g (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs x - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1)

theorem solve_equation : ∀ (x : ℝ), f x = g x :=
by
  sorry -- The proof will be inserted here

end solve_equation_l76_76993


namespace problem1_problem2_problem3_problem4_l76_76639

-- Proving the given mathematical equalities

theorem problem1 (x : ℝ) : 
  (x^4)^3 + (x^3)^4 - 2 * x^4 * x^8 = 0 := 
  sorry

theorem problem2 (x y : ℝ) : 
  (-2 * x^2 * y^3)^2 * (x * y)^3 = 4 * x^7 * y^9 := 
  sorry

theorem problem3 (a : ℝ) : 
  (-2 * a)^6 - (-3 * a^3)^2 + (-(2 * a)^2)^3 = -9 * a^6 := 
  sorry

theorem problem4 : 
  abs (- 1/8) + (Real.pi - 3)^0 + (- 1/2)^3 - (1/3)^(-2) = 8/9 := 
  sorry

end problem1_problem2_problem3_problem4_l76_76639


namespace coeff_x3_l76_76254

theorem coeff_x3 :
  let expr := 2 * (fun x => x^2 - 2 * x^3 + x)
           + 4 * (fun x => x + 3 * x^3 - 2 * x^2 + 2 * x^5 + x^3)
           - 6 * (fun x => 2 + x - 5 * x^3 - x^2) in
  (expr.coeff 3) = 42 := 
sorry

end coeff_x3_l76_76254


namespace probability_before_third_ring_l76_76559

-- Definitions of the conditions
def prob_first_ring : ℝ := 0.2
def prob_second_ring : ℝ := 0.3

-- Theorem stating that the probability of being answered before the third ring is 0.5
theorem probability_before_third_ring : prob_first_ring + prob_second_ring = 0.5 :=
by
  sorry

end probability_before_third_ring_l76_76559


namespace mr_haj_pays_1800_for_orders_l76_76950
   
   variable (total_operation_costs : ℕ)
   variable (employees_salary_fraction : ℚ)
   variable (delivery_costs_fraction : ℚ)

   def money_paid_for_orders (total_operation_costs : ℕ) (employees_salary_fraction : ℚ) (delivery_costs_fraction : ℚ) : ℕ :=
     let employees_salaries := (employees_salary_fraction * total_operation_costs)
     let remaining_amount := total_operation_costs - employees_salaries
     let delivery_costs := (delivery_costs_fraction * remaining_amount)
     total_operation_costs - (employees_salaries + delivery_costs)
   
   theorem mr_haj_pays_1800_for_orders :
     total_operation_costs = 4000 →
     employees_salary_fraction = 2/5 →
     delivery_costs_fraction = 1/4 →
     money_paid_for_orders total_operation_costs employees_salary_fraction delivery_costs_fraction = 1800 :=
   by
     intro h1 h2 h3
     rw [h1, h2, h3]
     -- detailed proof would go here (omitted)
     sorry
   
end mr_haj_pays_1800_for_orders_l76_76950


namespace max_discount_rate_l76_76153

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76153


namespace determine_k_l76_76700

theorem determine_k (S : ℕ → ℝ) (k : ℝ)
  (hSn : ∀ n, S n = k + 2 * (1 / 3)^n)
  (a1 : ℝ := S 1)
  (a2 : ℝ := S 2 - S 1)
  (a3 : ℝ := S 3 - S 2)
  (geom_property : a2^2 = a1 * a3) :
  k = -2 := 
by
  sorry

end determine_k_l76_76700


namespace opposite_of_neg_two_is_two_l76_76524

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76524


namespace max_discount_rate_l76_76152

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76152


namespace parallel_line_slope_l76_76089

theorem parallel_line_slope (x y : ℝ) :
  (∃ k b : ℝ, 3 * x + 6 * y = k * x + b) ∧ (∃ a b, y = a * x + b) ∧ 3 * x + 6 * y = -24 → 
  ∃ m : ℝ, m = -1/2 :=
by
  sorry

end parallel_line_slope_l76_76089


namespace min_diff_composite_sum_95_l76_76121

-- Definition of composite number
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

-- The problem statement in Lean
theorem min_diff_composite_sum_95 :
  ∀ a b : ℕ, is_composite a → is_composite b → a + b = 95 → 
  (∀ c d : ℕ, is_composite c → is_composite d → c + d = 95 → (abs (a - b) ≤ abs (c - d))) → 
  abs (a - b) = 3 :=
sorry

end min_diff_composite_sum_95_l76_76121


namespace minimal_n_and_initial_set_l76_76069

-- Definitions of the conditions
def operation_add (a b : ℕ) : ℤ := (a + b) / (a - b)

-- The main statement
theorem minimal_n_and_initial_set :
  ∃ (n : ℕ) (initial_set : list ℕ),
    n = 2 ∧ (initial_set = [1, 2] ∨ initial_set = [1, 3]) ∧
    (∀ k : ℕ, k > 0 → ∃ a b ∈ initial_set, operation_add a b = k) := sorry

end minimal_n_and_initial_set_l76_76069


namespace chopstick_length_l76_76979

theorem chopstick_length (wetted_part : ℝ) (dry_part : ℝ) (L : ℝ) 
  (h1 : wetted_part = 8) 
  (h2 : dry_part = wetted_part / 2) 
  (h3 : L = 2 * wetted_part + dry_part) : 
  L = 24 := 
by 
  rw [h1] at h2 h3
  rw [h2] at h3
  norm_num at h3
  exact h3

end chopstick_length_l76_76979


namespace min_soldiers_needed_l76_76770

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76770


namespace ones_digit_of_prime_sequence_l76_76684

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l76_76684


namespace limit_expression_l76_76759

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem limit_expression (h : ℝ) (hf : ContinuousAt f x₀) (hx₀ : f x₀ = -3) : 
  (∃ L, tendsto (λ h, (f (x₀ + h) - f (x₀ - 3 * h)) / h) (𝓝 0) (𝓝 L) ∧ L = -12) :=
begin
  sorry
end

end limit_expression_l76_76759


namespace maximize_sum_of_arithmetic_sequence_l76_76301

variable {a : ℕ → ℝ}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (S : ℕ → ℝ) (n : ℕ)
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in List.range n, a (i + 1)

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

theorem maximize_sum_of_arithmetic_sequence
  (h_arith_seq : arithmetic_sequence a)
  (h_condition1 : a 1 + a 3 + a 5 = 105)
  (h_condition2 : a 2 + a 4 + a 6 = 99)
  (h_sum_def : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n_max, ∀ n, S n ≤ S n_max ∧ n_max = 20 := sorry

end maximize_sum_of_arithmetic_sequence_l76_76301


namespace max_min_trig_expression_correct_l76_76417

noncomputable def max_min_trig_expression (a b : ℝ) : ℝ × ℝ :=
(let max_value := Real.sqrt (a^2 + b^2) in
 let min_value := - Real.sqrt (a^2 + b^2) in
 (max_value, min_value))

theorem max_min_trig_expression_correct (a b : ℝ) :
  max_min_trig_expression a b = (Real.sqrt (a^2 + b^2), -Real.sqrt (a^2 + b^2)) :=
by
  sorry

end max_min_trig_expression_correct_l76_76417


namespace quadratic_cubic_inequalities_l76_76873

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x : ℝ) : ℝ := -x ^ 3 + 5 * x - 3

variable (x : ℝ)

theorem quadratic_cubic_inequalities (h : 0 < x) : 
  (f x ≥ 2 * x - 1) ∧ (g x ≤ 2 * x - 1) := 
sorry

end quadratic_cubic_inequalities_l76_76873


namespace time_solution_l76_76589

-- Define the condition as a hypothesis
theorem time_solution (x : ℝ) (h : x / 4 + (24 - x) / 2 = x) : x = 9.6 :=
by
  -- Proof skipped
  sorry

end time_solution_l76_76589


namespace tan_alpha_minus_beta_l76_76360

theorem tan_alpha_minus_beta (α β : ℝ) (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_l76_76360


namespace opposite_of_neg_two_l76_76548

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76548


namespace value_of_a_minus_b_l76_76819

theorem value_of_a_minus_b (a b : ℤ) 
  (h₁ : |a| = 7) 
  (h₂ : |b| = 5) 
  (h₃ : a < b) : 
  a - b = -12 ∨ a - b = -2 := 
sorry

end value_of_a_minus_b_l76_76819


namespace largest_volume_sold_in_august_is_21_l76_76480

def volumes : List ℕ := [13, 15, 16, 17, 19, 21]

theorem largest_volume_sold_in_august_is_21
  (sold_volumes_august : List ℕ)
  (sold_volumes_september : List ℕ) :
  sold_volumes_august.length = 3 ∧
  sold_volumes_september.length = 2 ∧
  2 * (sold_volumes_september.sum) = sold_volumes_august.sum ∧
  (sold_volumes_august ++ sold_volumes_september).sum = volumes.sum →
  21 ∈ sold_volumes_august :=
sorry

end largest_volume_sold_in_august_is_21_l76_76480


namespace coin_and_die_probability_l76_76345

theorem coin_and_die_probability : 
  let p_heads := 1 / 2 in
  let p_2 := 1 / 4 in
  let p_4 := 1 / 4 in
  (p_heads * (p_2 + p_4) = 1 / 4) :=
by
  sorry

end coin_and_die_probability_l76_76345


namespace coin_and_dice_probability_l76_76980

-- define the probability of events in a uniform sample space
noncomputable def probability (event : set ℕ) (sample_space : set ℕ) : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

-- Define the coin flip outcomes and their probability
def coin_outcomes : set ℕ := {0, 1} -- 0 represents Tails, 1 represents Heads
def heads : set ℕ := {1}

-- Define the dice outcomes and event sets
def roll_dice : set (ℕ × ℕ) := {p | p.1 ∈ finset.range 1 7 ∧ p.2 ∈ finset.range 1 7}
def sum_fives : set (ℕ × ℕ) := {(1, 4), (2, 3), (3, 2), (4, 1)}
def sum_nines : set (ℕ × ℕ) := {(3, 6), (4, 5), (5, 4), (6, 3)}
def sum_fives_or_nines : set (ℕ × ℕ) := sum_fives ∪ sum_nines

-- Calculate the probability of the combined event
theorem coin_and_dice_probability :
  probability heads coin_outcomes * probability sum_fives_or_nines roll_dice = 1 / 9 :=
by
  sorry

end coin_and_dice_probability_l76_76980


namespace opposite_sides_line_range_a_l76_76720

theorem opposite_sides_line_range_a (a : ℝ) :
  (3 * 2 - 2 * 1 + a) * (3 * -1 - 2 * 3 + a) < 0 → -4 < a ∧ a < 9 := by
  sorry

end opposite_sides_line_range_a_l76_76720


namespace beta_sum_l76_76931

open Complex

noncomputable def Q (x : ℂ) := (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19)^2 - x^19

theorem beta_sum :
  let zeros : List ℂ := (19.map (λ n, ζ (n / 19)))  ++ (21.map (λ n, ζ (n / 21)))
  (0 < (zeros sum (λ k, k.β)) ∧ (zeros.sum (λ k, k.β)) == 5 ) :
  (∑ i in (finset.range 5), zeros[i] = (158 / 399)) :=
sorry

end beta_sum_l76_76931


namespace opposite_of_neg_two_l76_76530

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l76_76530


namespace work_days_l76_76574

/-- A needs 20 days to complete the work alone. B needs 10 days to complete the work alone.
    The total work must be completed in 12 days. We need to find how many days B must work 
    before A continues, such that the total work equals the full task. -/
theorem work_days (x : ℝ) (h0 : 0 ≤ x ∧ x ≤ 12) (h1 : 1 / 10 * x + 1 / 20 * (12 - x) = 1) : x = 8 := by
  sorry

end work_days_l76_76574


namespace num_solutions_f_f_x_eq_0_l76_76431

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x + 3 else -3 * x + 6

theorem num_solutions_f_f_x_eq_0 :
  {x : ℝ | f (f x) = 0}.finite.to_finset.card = 3 :=
sorry

end num_solutions_f_f_x_eq_0_l76_76431


namespace ellipse_equation_exists_slopes_product_l76_76293

-- Definitions of given conditions
def ellipse_eq(a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity(c a : ℝ) := c / a
def focus_vertex_distance(a c d : ℝ) := a - c = d

-- Correct answer statements
theorem ellipse_equation_exists (a b x y : ℝ) (h1 : eccentricity 1 2 = 1 / 2) (h2 : focus_vertex_distance 2 1 1) :
  ellipse_eq 2 (sqrt 3) x y = ellipse_eq 2 (sqrt 3) x y := by sorry

theorem slopes_product (x y : ℝ) (h1 : ellipse_eq 2 (sqrt 3) x y) :
  let K_PA := y / (x + 2), K_PB := y / (x - 2) in K_PA * K_PB = -3 / 4 := by sorry

end ellipse_equation_exists_slopes_product_l76_76293


namespace soldiers_to_add_l76_76763

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76763


namespace opposite_of_neg_two_l76_76546

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76546


namespace max_discount_rate_l76_76160

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76160


namespace min_soldiers_needed_l76_76772

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76772


namespace employed_females_distribution_l76_76822

-- Conditions
def total_population_employed := 0.96
def age_group_20_35 := 0.40
def age_group_36_50 := 0.50
def age_group_51_65 := 0.10
def employed_males := 0.24
def high_school_education := 0.45
def college_degree := 0.35
def postgraduate_degree := 0.20

-- Proof Statement
theorem employed_females_distribution :
  let employed_females := total_population_employed - employed_males in
  let employed_females_20_35 := age_group_20_35 * employed_females in
  let employed_females_36_50 := age_group_36_50 * employed_females in
  let employed_females_51_65 := age_group_51_65 * employed_females in
  let employed_females_high_school := high_school_education * employed_females in
  let employed_females_college := college_degree * employed_females in
  let employed_females_postgrad := postgraduate_degree * employed_females in
  employed_females_20_35 = 0.288 ∧
  employed_females_36_50 = 0.36 ∧
  employed_females_51_65 = 0.072 ∧
  employed_females_high_school = 0.324 ∧
  employed_females_college = 0.252 ∧
  employed_females_postgrad = 0.144 :=
by
  -- Calculation of employed females
  let employed_females := total_population_employed - employed_males
  -- Calculation per age group
  let employed_females_20_35 := age_group_20_35 * employed_females
  let employed_females_36_50 := age_group_36_50 * employed_females
  let employed_females_51_65 := age_group_51_65 * employed_females
  -- Calculation per educational level
  let employed_females_high_school := high_school_education * employed_females
  let employed_females_college := college_degree * employed_females
  let employed_females_postgrad := postgraduate_degree * employed_females
  -- Concluding the proof
  exact ⟨by norm_num [employed_females, employed_females_20_35], 
         by norm_num [employed_females, employed_females_36_50], 
         by norm_num [employed_females, employed_females_51_65], 
         by norm_num [employed_females, employed_females_high_school], 
         by norm_num [employed_females, employed_females_college], 
         by norm_num [employed_females, employed_females_postgrad]⟩


end employed_females_distribution_l76_76822


namespace max_discount_rate_l76_76147

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76147


namespace hyperbola_eccentricity_correct_l76_76608

noncomputable def hyperbola_eccentricity : ℝ :=
  let P := (x: ℝ, y: ℝ) in
  let a := 4 in
  let b := 3 in
  let c := Real.sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_correct :
  let P := (x: ℝ, y: ℝ) in
  let a := 4 in
  let b := 3 in
  let c := Real.sqrt (a^2 + b^2) in
  c / a = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_correct_l76_76608


namespace minimum_soldiers_to_add_l76_76790

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76790


namespace price_difference_l76_76108

noncomputable def original_price (discounted_price : ℝ) : ℝ :=
  discounted_price / 0.85

noncomputable def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.25

theorem price_difference (discounted_price : ℝ) (h : discounted_price = 71.4) : 
  (final_price discounted_price) - (original_price discounted_price) = 5.25 := 
by
  sorry

end price_difference_l76_76108


namespace digits_difference_l76_76466

theorem digits_difference (d A B : ℕ) (h1 : d > 6) (h2 : (B + A) * d + 2 * A = d^2 + 7 * d + 2)
  (h3 : B + A = 10) (h4 : 2 * A = 8) : A - B = 3 :=
by 
  sorry

end digits_difference_l76_76466


namespace gcd_7854_13843_l76_76231

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := 
  sorry

end gcd_7854_13843_l76_76231


namespace mr_haj_pays_1800_for_orders_l76_76949
   
   variable (total_operation_costs : ℕ)
   variable (employees_salary_fraction : ℚ)
   variable (delivery_costs_fraction : ℚ)

   def money_paid_for_orders (total_operation_costs : ℕ) (employees_salary_fraction : ℚ) (delivery_costs_fraction : ℚ) : ℕ :=
     let employees_salaries := (employees_salary_fraction * total_operation_costs)
     let remaining_amount := total_operation_costs - employees_salaries
     let delivery_costs := (delivery_costs_fraction * remaining_amount)
     total_operation_costs - (employees_salaries + delivery_costs)
   
   theorem mr_haj_pays_1800_for_orders :
     total_operation_costs = 4000 →
     employees_salary_fraction = 2/5 →
     delivery_costs_fraction = 1/4 →
     money_paid_for_orders total_operation_costs employees_salary_fraction delivery_costs_fraction = 1800 :=
   by
     intro h1 h2 h3
     rw [h1, h2, h3]
     -- detailed proof would go here (omitted)
     sorry
   
end mr_haj_pays_1800_for_orders_l76_76949


namespace incorrect_proposition_example_l76_76976

theorem incorrect_proposition_example (p q : Prop) (h : ¬ (p ∧ q)) : ¬ (¬p ∧ ¬q) :=
by
  sorry

end incorrect_proposition_example_l76_76976


namespace square_perimeter_increase_l76_76062

theorem square_perimeter_increase (s : ℝ) : (4 * (s + 2) - 4 * s) = 8 := 
by
  sorry

end square_perimeter_increase_l76_76062


namespace find_CD_l76_76414

-- Definitions for the right triangle and circle intersection
variables (A B C D : Type) [RightAngleTriangle A B C] [CircleWithDiameter B C] [IntersectsAt AC D] 

-- Given conditions
variable (AD BD : ℝ)
axiom AD_eq_3 : AD = 3
axiom BD_eq_2 : BD = 2

-- Proof problem statement
theorem find_CD (AD BD : ℝ) (h1: AD = 3) (h2: BD = 2) : CD = 4 / 3 :=
by {
  sorry
}

end find_CD_l76_76414


namespace max_discount_rate_l76_76191

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76191


namespace minimum_soldiers_to_add_l76_76785

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76785


namespace Sierra_Crest_Trail_Length_l76_76856

theorem Sierra_Crest_Trail_Length (a b c d e : ℕ) 
(h1 : a + b + c = 36) 
(h2 : b + d = 30) 
(h3 : d + e = 38) 
(h4 : a + d = 32) : 
a + b + c + d + e = 74 := by
  sorry

end Sierra_Crest_Trail_Length_l76_76856


namespace probability_blue_is_correct_l76_76615
open nat

-- Define the set of tiles
def tiles : finset ℕ := finset.range 71 -- tiles numbered 1 through 70

-- Define a function to get blue tiles
def is_blue (n : ℕ) : Prop := n % 7 = 3

-- Define the set of blue tiles
def blue_tiles : finset ℕ := tiles.filter is_blue

-- Count the blue tiles
def blue_tile_count : ℕ := blue_tiles.card

-- Count the total number of tiles
def total_tile_count : ℕ := tiles.card

-- Compute the probability as a fraction
def probability_blue : ℚ := blue_tile_count / total_tile_count

-- The main statement we are to prove: the probability that the tile is blue is 1/7
theorem probability_blue_is_correct : probability_blue = 1 / 7 := by 
  sorry

end probability_blue_is_correct_l76_76615


namespace trigonometric_identity_l76_76356

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 :=
sorry

end trigonometric_identity_l76_76356


namespace price_of_computer_and_desk_l76_76948

theorem price_of_computer_and_desk (x y : ℕ) 
  (h1 : 10 * x + 200 * y = 90000)
  (h2 : 12 * x + 120 * y = 90000) : 
  x = 6000 ∧ y = 150 :=
by
  sorry

end price_of_computer_and_desk_l76_76948


namespace correct_options_l76_76312

def circle_eq (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 4

def line_eq (x y m : ℝ) := x + m * y - m - 2 = 0

theorem correct_options (m : ℝ) :
  (∀ x y, circle_eq x y → x = 1 ∧ y = 2) ∧
  (∀ x y, line_eq x y m → (x = 2 ∧ y = 1)) ∧
  (∃ x y, circle_eq x y ∧ line_eq x y m ∧ (2 * Real.sqrt 2)) :=
sorry

end correct_options_l76_76312


namespace distance_moved_by_P_l76_76618

-- Assume the original circle with radius 4 centered at B(3,3)
def B := (3, 3)
def radius_original := 4

-- Assume the dilated circle with radius 6 centered at B'(7, 9)
def B' := (7, 9)
def radius_dilated := 6

-- Assume the point P(1,1)
def P := (1, 1)

-- We need to compute the distance the point P moves under the transformation
theorem distance_moved_by_P : 
  let k := radius_dilated / radius_original in
  let center_dilation := (-5, -7) in
  let d0 := Real.sqrt (((center_dilation.1 - P.1) ^ 2) + ((center_dilation.2 - P.2) ^ 2)) in
  let d1 := k * d0 in
  d1 - d0 = 5 :=
sorry

end distance_moved_by_P_l76_76618


namespace jackson_money_l76_76850

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l76_76850


namespace reflection_line_properties_l76_76924

theorem reflection_line_properties : 
  ∃ (m b : ℚ), (∀ (x y x' y' : ℚ), 
    ((x, y) = (2, 3)) → ((x', y') = (4, 9)) → 
    (y' = y + m * (x' - x) + b) →
    m + b = 20 / 3) :=
by {
  use [-1/3, 7],
  intros x y x' y' hx hy hr,
  simp only [hx, hy] at hr,
  sorry -- The actual proof would go here.
}

end reflection_line_properties_l76_76924


namespace max_discount_rate_l76_76144

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76144


namespace simplify_fraction_l76_76896

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := 
by 
  sorry

end simplify_fraction_l76_76896


namespace min_soldiers_needed_l76_76771

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l76_76771


namespace first_number_eventually_one_l76_76485

/-- Given a sequence of integers from 1 to 1993 written in some order, and an operation
 that reverses the first k numbers if the first number is k, prove that the first number 
 will eventually become 1 after a finite number of operations. -/
theorem first_number_eventually_one (a : Fin 1993 → ℕ) (h : ∀ i, 1 ≤ a i ∧ a i ≤ 1993) :
  ∃ n : ℕ, ∀ a' : Fin 1993 → ℕ, (operation_sequence a n = a' ∧ a' 0 = 1) :=
sorry

/-- Define the operation that reverses the first k numbers if the first number is k. -/
def operation (a : Fin 1993 → ℕ) : Fin 1993 → ℕ :=
λ i, if h : (a 0) ≤ i.val.succ then a ⟨i.val.succ - (a 0), sorry⟩ else a i

/-- Define the sequence of operations performed n times. -/
def operation_sequence (a : Fin 1993 → ℕ) : ℕ → (Fin 1993 → ℕ)
| 0       := a
| (n + 1) := operation (operation_sequence a n)

end first_number_eventually_one_l76_76485


namespace triangle_third_side_length_l76_76045

-- Definitions of angles and sides
variables {A B C : ℝ} {a b c : ℝ}

-- The conditions as described
def triangle_conditions :=
  (sin (2 * A) + sin (2 * B) + sin (2 * C) = 3 / 2) ∧
  (a = 8) ∧
  (b = 15)

-- The maximum length of the third side
def max_third_side (a b : ℝ) :=
  (a^2 + b^2 - 2 * a * b * cos (π / 3)) = 169

-- The problem statement
theorem triangle_third_side_length (A B C : ℝ) (a b : ℝ) 
  (h : triangle_conditions A B C a b) :
  c = 13 :=
sorry

end triangle_third_side_length_l76_76045


namespace find_heaviest_three_l76_76104

-- Given a set of weights
def weights : Type := fin 36

-- Define a balance that can order any 6 weights
axiom balance : Π (w : fin 6 → weights), {a // ∀ i j, i < j → a i < a j}

-- Main theorem based on the problem
theorem find_heaviest_three :
  ∃ (w1 w2 w3 : weights),
    (∀ (c1 c2 : weights), (c1 = w1 ∨ c1 = w2 ∨ c1 = w3) → (c2 = w1 ∨ c2 = w2 ∨ c2 = w3) → c1 ≤ c2 → c1 = w1 → c2 = w2 → c2 = w3) :=
sorry

end find_heaviest_three_l76_76104


namespace max_discount_rate_l76_76190

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76190


namespace root_power_division_l76_76055

noncomputable def root4 (a : ℝ) : ℝ := a^(1/4)
noncomputable def root6 (a : ℝ) : ℝ := a^(1/6)

theorem root_power_division : 
  (root4 7) / (root6 7) = 7^(1/12) :=
by sorry

end root_power_division_l76_76055


namespace minimum_soldiers_to_add_l76_76786

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l76_76786


namespace total_time_spent_l76_76020

theorem total_time_spent :
  let mac_download := 10 in
  let win_download := 3 * mac_download in
  let ny_audio_glitch := 2 * 6 in
  let ny_video_glitch := 8 in
  let ny_glitch_time := ny_audio_glitch + ny_video_glitch in
  let ny_non_glitch_time := 3 * ny_glitch_time in
  let ny_total_time := ny_glitch_time + ny_non_glitch_time in
  let berlin_audio_glitch := 3 * 4 in
  let berlin_video_glitch := 2 * 5 in
  let berlin_glitch_time := berlin_audio_glitch + berlin_video_glitch in
  let berlin_non_glitch_time := 2 * berlin_glitch_time in
  let berlin_total_time := berlin_glitch_time + berlin_non_glitch_time in
  let total_download_time := mac_download + win_download in
  let grand_total_time := total_download_time + ny_total_time + berlin_total_time in
  grand_total_time = 186 :=
by
  sorry

end total_time_spent_l76_76020


namespace tan_alpha_minus_beta_neg_one_l76_76364

theorem tan_alpha_minus_beta_neg_one 
  (α β : ℝ)
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tan_alpha_minus_beta_neg_one_l76_76364


namespace E_divides_ABC_half_l76_76424

-- Given Conditions
variables {α : Type*} [linear_ordered_field α]

-- Definition: Point D is the midpoint of arc AC
def D_midpoint_arc (A C D : α) : Prop := D = (A + C) / 2

-- Definition: Point B on arc A D different from D
def B_on_arc (A D B : α) : Prop := A < B ∧ B < D 

-- Definition: Point E is the foot of perpendicular from D to A B C
def E_perpendicular (A B C D E : α) : Prop :=
  ∃ m : α, E = m * (A + B + C) / 2

-- The theorem to prove
theorem E_divides_ABC_half (A B C D E : α)
  (hD : D_midpoint_arc A C D)
  (hB : B_on_arc A D B)
  (hE : E_perpendicular A B C D E) :
  2 * A + 2 * B + 2 * C = 4 * E :=
begin
  sorry
end

end E_divides_ABC_half_l76_76424


namespace integral_of_quarter_circle_l76_76642

noncomputable def quarter_circle_integral : ℝ := ∫ x in 0..5, real.sqrt (25 - x^2)

theorem integral_of_quarter_circle : quarter_circle_integral = (25 * real.pi / 4) :=
by
  sorry

end integral_of_quarter_circle_l76_76642


namespace tangent_identity_l76_76373

theorem tangent_identity (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) :
  tan (α - β) = -1 := 
sorry

end tangent_identity_l76_76373


namespace tan_alpha_minus_beta_l76_76353

variable (α β : ℝ)

theorem tan_alpha_minus_beta
  (h : sin (α + β) + cos (α + β) = 2 * real.sqrt 2 * cos (α + π/4) * sin β) : 
  real.tan (α - β) = -1 :=
sorry

end tan_alpha_minus_beta_l76_76353


namespace circle_radius_5_l76_76274

theorem circle_radius_5 (k : ℝ) : 
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ↔ k = -40 :=
by
  sorry

end circle_radius_5_l76_76274


namespace max_discount_rate_l76_76131

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l76_76131


namespace max_discount_rate_l76_76157

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76157


namespace find_length_PT_l76_76445

-- Definitions and conditions
variables (p q RH RT : ℝ) (areaPQRS areaTriangleRHT : ℝ)

-- Rectangle's area and triangle's area
def is_rectangle_PQRS (p q areaPQRS : ℝ) : Prop := p * q = areaPQRS
def is_triangle_RHT (RH RT areaTriangleRHT : ℝ) : Prop := (1 / 2) * RH * RT = areaTriangleRHT

-- Given values
def given_values : Prop :=
  areaPQRS = 144 ∧ areaTriangleRHT = 64

-- Proof problem
theorem find_length_PT
  (hpqrs : is_rectangle_PQRS p q areaPQRS)
  (hrht : is_triangle_RHT RH RT areaTriangleRHT)
  (givens : given_values) :
  (RH = 8 * Real.sqrt 2) →
  (q = 8 * Real.sqrt 2) →
  (p = 9 * Real.sqrt 2) →
  (PT = 17 * Real.sqrt 2) :=
begin
  sorry
end

end find_length_PT_l76_76445


namespace int_as_sum_of_squares_l76_76894

theorem int_as_sum_of_squares (n : ℤ) : ∃ a b c : ℤ, n = a^2 + b^2 - c^2 :=
sorry

end int_as_sum_of_squares_l76_76894


namespace power_function_condition_l76_76263

theorem power_function_condition (m : ℤ) :
  (∀ x : ℝ, x ≠ 0 → x^(m^2 - 2 * m - 3) ≠ 0) ∧
  (0^(m^2 - 2 * m - 3) = 0 → (m^2 - 2 * m - 3) ≠ 0) ∧
  (m % 2 = 0) → m = 1 :=
sorry

end power_function_condition_l76_76263


namespace max_expr_value_l76_76930

theorem max_expr_value (a b c d : ℝ) (h_a : -8.5 ≤ a ∧ a ≤ 8.5)
                       (h_b : -8.5 ≤ b ∧ b ≤ 8.5)
                       (h_c : -8.5 ≤ c ∧ c ≤ 8.5)
                       (h_d : -8.5 ≤ d ∧ d ≤ 8.5) :
                       a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 306 :=
sorry

end max_expr_value_l76_76930


namespace sarah_jamie_julien_ratio_l76_76855

theorem sarah_jamie_julien_ratio (S J : ℕ) (R : ℝ) :
  -- Conditions
  (J = S + 20) ∧
  (S = R * 50) ∧
  (7 * (J + S + 50) = 1890) ∧
  -- Prove the ratio
  R = 2 := by
  sorry

end sarah_jamie_julien_ratio_l76_76855


namespace sum_of_digits_of_product_is_18108_l76_76671

-- Define the numbers in the problem
def num1 : Nat := 4444...4 (2012 times)
def num2 : Nat := 9999...9 (2012 times)
def ten_pow_2012 : Nat := 10^2012

-- Define the given condition
def cond_num2_eq : num2 = ten_pow_2012 - 1 := by sorry
def N : Nat := num1

-- Define the assertion
theorem sum_of_digits_of_product_is_18108 : 
  digitSum (N * num2) = 18108 := by sorry

end sum_of_digits_of_product_is_18108_l76_76671


namespace spadesuit_eval_l76_76006

def spadesuit (a b : ℝ) : ℝ := (3 * a / b) * (b / a)

theorem spadesuit_eval : ((spadesuit 7 (spadesuit 4 9)) ∉ {0}) → spadesuit (spadesuit 7 (spadesuit 4 9)) 2 = 3 :=
by
  admit

end spadesuit_eval_l76_76006


namespace smallest_percent_increase_l76_76064

def percent_increase (a b : ℝ) : ℝ := (b - a) / a * 100

theorem smallest_percent_increase :
  let q1 := 200
  let q2 := 500
  let q3 := 1500
  let q4 := 3000
  let q5 := 10000
  percent_increase q1 q2 = 150 ∧
  percent_increase q2 q3 = 200 ∧
  percent_increase q3 q4 = 100 ∧
  percent_increase q4 q5 ≈ 233.33 →
  percent_increase q3 q4 = 100 :=
by
  intros q1 q2 q3 q4 q5
  rw [percent_increase]
  sorry

end smallest_percent_increase_l76_76064


namespace minimum_soldiers_to_add_l76_76780

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76780


namespace taxi_fare_ride_distance_l76_76067

theorem taxi_fare_ride_distance (fare_first: ℝ) (first_mile: ℝ) (additional_fare_rate: ℝ) (additional_distance: ℝ) (total_amount: ℝ) (tip: ℝ) (x: ℝ) :
  fare_first = 3.00 ∧ first_mile = 0.75 ∧ additional_fare_rate = 0.25 ∧ additional_distance = 0.1 ∧ total_amount = 15 ∧ tip = 3 ∧
  (total_amount - tip) = fare_first + additional_fare_rate * (x - first_mile) / additional_distance → x = 4.35 :=
by
  intros
  sorry

end taxi_fare_ride_distance_l76_76067


namespace orthocenter_projections_l76_76033

-- Begin with necessary definitions
variables {A B C D : Type} [Tetrahedron A B C D] -- define a tetrahedron/triangular pyramid

-- Define orthogonal projection and orthocenter
class OrthogonalProjection (A B C D : Type) :=
  (proj : A → B)
  (orthocenter : A → C)

-- Assume conditions
variable [hp : OrthogonalProjection A (Plane B C) D]

-- Main theorem statement
theorem orthocenter_projections (A B C D : Type) [Tetrahedron A B C D] [OrthogonalProjection A (Plane B C) D]
  (h1 : hp.proj D = hp.orthocenter (Plane B C))
  : ∀ (V : Type), V ∈ {A, B, C} → hp.proj V = hp.orthocenter (Plane _ (opposite_face V)) :=
sorry

end orthocenter_projections_l76_76033


namespace maximum_perimeter_triangle_area_l76_76402

-- Part 1: Maximum Perimeter
theorem maximum_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h_c : c = 2) 
  (h_C : C = Real.pi / 3) :
  (a + b + c) ≤ 6 :=
sorry

-- Part 2: Area under given trigonometric condition
theorem triangle_area (A B C a b c : ℝ) 
  (h_c : 2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C) :
  (1/2 * a * b * Real.sin C) = (2 * Real.sqrt 6) / 3 :=
sorry

end maximum_perimeter_triangle_area_l76_76402


namespace Pooja_speed_3_l76_76453

variable (Roja_speed Pooja_speed : ℝ)
variable (t d : ℝ)

theorem Pooja_speed_3
  (h1 : Roja_speed = 6)
  (h2 : t = 4)
  (h3 : d = 36)
  (h4 : d = t * (Roja_speed + Pooja_speed)) :
  Pooja_speed = 3 :=
by
  sorry

end Pooja_speed_3_l76_76453


namespace find_percentage_l76_76118

theorem find_percentage (P : ℝ) (n : ℝ) (m : ℝ) (c : ℝ) (h : 0.50 * n = (P / 100) * m + c) :
  P = 40 :=
by
  -- Given terms
  let n := 456
  let m := 120
  let c := 180
  
  -- Equations from the problem
  have h₁ : 0.50 * 456 = 228, by sorry
  have h₂ : ∀ (P : ℝ), 228 = (P / 100) * 120 + 180, by sorry
  
  -- Given that h is true
  have h := h₁.trans (h₂ 40)

  -- Therefore, P = 40
  exact (by sorry : P = 40)

end find_percentage_l76_76118


namespace find_smallest_number_l76_76940

theorem find_smallest_number (x y z : ℝ) 
  (h1 : x + y + z = 150) 
  (h2 : y = 3 * x + 10) 
  (h3 : z = x^2 - 5) 
  : x = 10.21 :=
sorry

end find_smallest_number_l76_76940


namespace num_of_functions_with_two_distinct_elements_in_range_l76_76014

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- The function space from A to B
def func_space := {f : ℕ → ℕ // ∀ x ∈ A, f x ∈ B}

-- Define the property of having exactly two distinct elements in the range
def has_two_distinct_elements_in_range (f : ℕ → ℕ) : Prop :=
  ∃ x1 x2 ∈ B, x1 ≠ x2 ∧ ∀ y ∈ A, f y = x1 ∨ f y = x2

-- The number of functions with the given property
def num_functions_with_property : ℕ :=
  {f : func_space // has_two_distinct_elements_in_range f.val}.to_finset.card

-- The proof goal
theorem num_of_functions_with_two_distinct_elements_in_range : num_functions_with_property = 6 :=
  sorry

end num_of_functions_with_two_distinct_elements_in_range_l76_76014


namespace parallel_BF_CE_l76_76286

variables {A B C D E F : Type*}
variables [convex_hexagon ABCDEF] [cyclic_quad ABDF] [cyclic_quad ACDE]

theorem parallel_BF_CE
  (h1 : ∠ F A E = ∠ B D C)
  (h2 : cyclic ABDF)
  (h3 : cyclic ACDE)
  : B F ∥ C E :=
sorry

end parallel_BF_CE_l76_76286


namespace a3_min_value_l76_76935

variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 3, a n = (setOf (λ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 1).toFinset.image (λ (i, j), |a i - a j|)).min')

-- The main statement to prove
theorem a3_min_value (h : a_sequence a) (h10 : a 10 = 1) : a 3 = 21 :=
sorry

end a3_min_value_l76_76935


namespace soldiers_to_add_l76_76767

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76767


namespace maximum_discount_rate_l76_76178

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l76_76178


namespace max_discount_rate_l76_76188

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76188


namespace train_cross_time_l76_76984

-- Definitions from the conditions
def length_of_train : ℤ := 600
def speed_of_man_kmh : ℤ := 2
def speed_of_train_kmh : ℤ := 56

-- Conversion factors and speed conversion
def kmh_to_mph_factor : ℤ := 1000 / 3600 -- 1 km/hr = 0.27778 m/s approximately

def speed_of_man_ms : ℤ := speed_of_man_kmh * kmh_to_mph_factor -- Convert speed of man to m/s
def speed_of_train_ms : ℤ := speed_of_train_kmh * kmh_to_mph_factor -- Convert speed of train to m/s

-- Calculating relative speed
def relative_speed_ms : ℤ := speed_of_train_ms - speed_of_man_ms

-- Calculating the time taken to cross
def time_to_cross : ℤ := length_of_train / relative_speed_ms 

-- The theorem to prove
theorem train_cross_time : time_to_cross = 40 := 
by sorry

end train_cross_time_l76_76984


namespace convert_101110_to_decimal_l76_76656

def binary_to_decimal (s : String) : ℕ :=
  let bits := s.toList.reverse
  List.foldl (λ acc ⟨bit, idx⟩ => 
    if bit = '1' then acc + 2^idx else acc) 0 (bits.enum)

theorem convert_101110_to_decimal : binary_to_decimal "101110" = 46 :=
by
  sorry

end convert_101110_to_decimal_l76_76656


namespace units_digit_l76_76260

theorem units_digit (a b c d : ℕ) (units_a : a % 10 = 7) (units_b : b % 10 = 7) (units_c : c % 10 = 7) : 
  (a * b * c - d ^ 3) % 10 = 0 := 
by
  have h1 : (7 * 7 * 7) % 10 = 3 :=
    sorry
  have h2 : (7 ^ 3) % 10 = 3 :=
    sorry
  show (a * b * c - d ^ 3) % 10 = 0 from
    sorry

end units_digit_l76_76260


namespace younger_son_age_30_years_later_eq_60_l76_76918

variable (age_diff : ℕ) (elder_age : ℕ) (younger_age_30_years_later : ℕ)

-- Conditions
axiom h1 : age_diff = 10
axiom h2 : elder_age = 40

-- Definition of younger son's current age
def younger_age : ℕ := elder_age - age_diff

-- Definition of younger son's age 30 years from now
def younger_age_future : ℕ := younger_age + 30

-- Proving the required statement
theorem younger_son_age_30_years_later_eq_60 (h_age_diff : age_diff = 10) (h_elder_age : elder_age = 40) :
  younger_age_future elder_age age_diff = 60 :=
by
  unfold younger_age
  unfold younger_age_future
  rw [h_age_diff, h_elder_age]
  sorry

end younger_son_age_30_years_later_eq_60_l76_76918


namespace binomial_sum_l76_76651

theorem binomial_sum (n k : ℕ) (h : n = 10) (hk : k = 3) :
  Nat.choose n k + Nat.choose n (n - k) = 240 :=
by
  -- placeholder for actual proof
  sorry

end binomial_sum_l76_76651


namespace total_earnings_l76_76591

theorem total_earnings (zachary_games : ℕ) (price_per_game : ℝ) (jason_percentage_increase : ℝ) (ryan_extra : ℝ)
  (h1 : zachary_games = 40) (h2 : price_per_game = 5) (h3 : jason_percentage_increase = 0.30) (h4 : ryan_extra = 50) :
  let z_earnings := zachary_games * price_per_game,
      j_earnings := z_earnings * (1 + jason_percentage_increase),
      r_earnings := j_earnings + ryan_extra
  in z_earnings + j_earnings + r_earnings = 770 := by
  sorry

end total_earnings_l76_76591


namespace max_discount_rate_l76_76194

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76194


namespace gcd_of_two_powers_l76_76966

-- Define the expressions
def two_pow_1015_minus_1 : ℤ := 2^1015 - 1
def two_pow_1024_minus_1 : ℤ := 2^1024 - 1

-- Define the gcd function and the target value
noncomputable def gcd_expr : ℤ := Int.gcd (2^1015 - 1) (2^1024 - 1)
def target : ℤ := 511

-- The statement we want to prove
theorem gcd_of_two_powers : gcd_expr = target := by 
  sorry

end gcd_of_two_powers_l76_76966


namespace pasha_meets_katya_at_exactly_5_hours_after_noon_l76_76995

def meet_hours_after_noon (a_to_b_speed : ℝ) (b_to_a_speed : ℝ)
  (vitya_meet_masha : ℝ) (pasha_meet_masha : ℝ) (vitya_meet_katya : ℝ) : ℝ :=
5

theorem pasha_meets_katya_at_exactly_5_hours_after_noon :
  ∀ (a_to_b_speed b_to_a_speed : ℝ)
    (vitya_meet_masha : ℝ)
    (pasha_meet_masha : ℝ)
    (vitya_meet_katya : ℝ),
    a_to_b_speed = b_to_a_speed →
    vitya_meet_masha = 12 →
    pasha_meet_masha = 15 →
    vitya_meet_katya = 14 →
    meet_hours_after_noon a_to_b_speed b_to_a_speed vitya_meet_masha pasha_meet_masha vitya_meet_katya = 5 :=
begin
  intros a_to_b_speed b_to_a_speed 
         vitya_meet_masha pasha_meet_masha 
         vitya_meet_katya,
  assume h_speed_eq hmasha_vitya 
         hpasha_masha hkatya_vitya,
  -- the rest of the proof goes here
  sorry
end

end pasha_meets_katya_at_exactly_5_hours_after_noon_l76_76995


namespace find_sin_phi_l76_76001

variables (a b d : ℝ → ℝ → ℝ → ℝ)
variables (φ : ℝ)
variables (norm_a norm_b norm_d : ℝ)
variables (cross : (ℝ → ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ → ℝ))

-- Given conditions
axiom norm_a_eq_2 : norm_a = 2
axiom norm_b_eq_4 : norm_b = 4
axiom norm_d_eq_6 : norm_d = 6
axiom cross_product_condition : cross a (cross a b) = d

-- Definition of norms and cross product
axiom norms_def : norm_a = (a 0 0 1)^2 + (a 0 1 0)^2 + (a 1 0 0)^2
axiom norms_def_b : norm_b = (b 0 0 1)^2 + (b 0 1 0)^2 + (b 1 0 0)^2
axiom norms_def_d : norm_d = (d 0 0 1)^2 + (d 0 1 0)^2 + (d 1 0 0)^2
axiom cross_product_def : ∀ u v, cross u v = ((u 1 0 0) * (v 0 1 0) - (u 0 1 0) * (v 1 0 0), 
                                             (u 0 1 0) * (v 0 0 1) - (u 0 0 1) * (v 0 1 0), 
                                             (u 0 0 1) * (v 1 0 0) - (u 1 0 0) * (v 0 0 1))

-- Goal to prove
theorem find_sin_phi : (norm_d = Mathlib.sqrt (cross a b).norm) → φ = Math.asin (3 / 8) :=
by sorry

end find_sin_phi_l76_76001


namespace soldiers_to_add_l76_76768

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l76_76768


namespace hayes_loads_per_week_l76_76737

-- Constants and conditions
constant podsPerPack : Nat := 39
constant packsPerYear : Nat := 4
constant weeksPerYear : Nat := 52

-- Statement to prove
theorem hayes_loads_per_week : (packsPerYear * podsPerPack) / weeksPerYear = 3 := by
  sorry

end hayes_loads_per_week_l76_76737


namespace Cary_height_is_72_l76_76646

variable (Cary_height Bill_height Jan_height : ℕ)

-- Conditions
axiom Bill_height_is_half_Cary_height : Bill_height = Cary_height / 2
axiom Jan_height_is_6_inches_taller_than_Bill : Jan_height = Bill_height + 6
axiom Jan_height_is_42 : Jan_height = 42

-- Theorem statement
theorem Cary_height_is_72 : Cary_height = 72 := 
by
  sorry

end Cary_height_is_72_l76_76646


namespace children_marbles_problem_l76_76234

theorem children_marbles_problem (n x N : ℕ) 
  (h1 : N = n * x)
  (h2 : 1 + (N - 1) / 10 = x) :
  n = 9 ∧ x = 9 :=
by
  sorry

end children_marbles_problem_l76_76234


namespace count_valid_numbers_l76_76341

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (4 < a) ∧ (4 < b) ∧ a + b + 9 = 9 * (n / 100 + (n % 100) / 10 + (n % 10)) ∧ n = 100 * a + 10 * b + 9

theorem count_valid_numbers : Finset.card {n | n ∈ (Finset.range 1000).filter is_valid_number} = 5 := sorry

end count_valid_numbers_l76_76341


namespace max_discount_rate_l76_76185

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l76_76185


namespace Sheela_monthly_income_l76_76893

theorem Sheela_monthly_income (M : ℝ) (h : 0.22 * M = 3800) : M = 17272.73 :=
begin
  sorry -- Proof to be filled in
end

end Sheela_monthly_income_l76_76893


namespace length_BC_l76_76635

-- Define the points and triangle
variables {A B C E F : Type}

-- Assume conditions provided
variables {ABC : Triangle A B C}
variables (is_midpoint_AE : Midpoint A B E)
variables (is_midpoint_AF : Midpoint A C F)
variables (length_EF : length EF = 2)

theorem length_BC : length BC = 4 :=
by sorry

end length_BC_l76_76635


namespace f_sin_cos_1_f_cos_sin_2pi_over_3_l76_76921

def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (1 + x) = -f(1 - x)
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom f_interval : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 1 - x

theorem f_sin_cos_1 : f (Real.sin 1) < f (Real.cos 1) :=
sorry

theorem f_cos_sin_2pi_over_3 : f (Real.cos (2 * Real.pi / 3)) > f (Real.sin (2 * Real.pi / 3)) :=
sorry

end f_sin_cos_1_f_cos_sin_2pi_over_3_l76_76921


namespace five_equilateral_triangles_total_area_l76_76675

theorem five_equilateral_triangles_total_area :
  let s := 3
  let single_triangle_area := (sqrt 3 / 4) * s ^ 2
  let total_area := 5 * single_triangle_area
  total_area = 45 * (sqrt 3) / 4 := by
  sorry

end five_equilateral_triangles_total_area_l76_76675


namespace reggie_money_left_l76_76451

theorem reggie_money_left:
  ∀ (initial_money cost_per_book : ℕ) (books_bought : ℕ),
    initial_money = 48 →
    cost_per_book = 2 →
    books_bought = 5 →
    initial_money - (books_bought * cost_per_book) = 38 :=
by
  intros initial_money cost_per_book books_bought h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end reggie_money_left_l76_76451


namespace water_dispenser_capacity_l76_76922

theorem water_dispenser_capacity :
  ∀ (x : ℝ), (0.25 * x = 60) → x = 240 :=
by
  intros x h
  sorry

end water_dispenser_capacity_l76_76922


namespace sum_of_solutions_zero_l76_76306

variable (f : ℝ → ℝ)
variable (h_symm : ∀ x : ℝ, f(-x) = f(x))
variable (h_solutions : ∃ s : Finset ℝ, s.card = 2009 ∧ ∀ x ∈ s, f(x) = 0)

theorem sum_of_solutions_zero : ∃ s : Finset ℝ, s.card = 2009 ∧ (∀ x ∈ s, f(x) = 0) ∧ s.sum id = 0 :=
by
  obtain ⟨s, h_card, h_f⟩ := h_solutions
  have : ∀ x ∈ s, -x ∈ s := 
  begin 
    intros x hx,
    rw [mem_finset, mem_finset] at hx,
    exact ⟨h_f _ hx, h_symm x⟩,
  end
  sorry

end sum_of_solutions_zero_l76_76306


namespace problem1_problem2_l76_76710

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l76_76710


namespace solve_log_eqn_l76_76899

-- Define the constraints for the logarithmic and fractional expressions
def valid_domain (y : ℝ) : Prop :=
  (2 * y + 8) / (3 * y - 2) > 0 ∧ (3 * y - 2) / (2 * y - 5) > 0

theorem solve_log_eqn (y : ℝ) (h : valid_domain y) :
  log 4 ((2 * y + 8) / (3 * y - 2)) + log 4 ((3 * y - 2) / (2 * y - 5)) = 2 → y = 44 / 15 :=
by
  sorry

end solve_log_eqn_l76_76899


namespace total_yield_l76_76846

theorem total_yield (x y z : ℝ)
  (h1 : 0.4 * z + 0.2 * x = 1)
  (h2 : 0.1 * y - 0.1 * z = -0.5)
  (h3 : 0.1 * x + 0.2 * y = 4) :
  x + y + z = 15 :=
sorry

end total_yield_l76_76846


namespace max_discount_rate_l76_76161

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l76_76161


namespace max_discount_rate_l76_76165

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76165


namespace min_value_tan_of_triangle_l76_76715

open Real

-- Define the circumcenter property
variables {A B C O : Point}
variables {vecAO vecBO vecCO vecBC vecCA vecAB : Vector3}

-- Conditions given in the problem
axiom circumcenter_condition :
  (vecAO • vecBC) + 2 * (vecBO • vecCA) + 3 * (vecCO • vecAB) = 0

theorem min_value_tan_of_triangle :
  ∀ {A B C : ℝ}, (∃ O : Point, 
    let vecAO := vector_from O A in
    let vecBO := vector_from O B in
    let vecCO := vector_from O C in
    let vecBC := vector_from B C in
    let vecCA := vector_from C A in
    let vecAB := vector_from A B in
    circumcenter_condition
  → ∃ (x : ℝ), x = (1/tan A) + (1/tan C) ∧ x ≥ (2*sqrt(3))/3
  :=
sorry

end min_value_tan_of_triangle_l76_76715


namespace max_discount_rate_l76_76167

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l76_76167


namespace effective_decontamination_time_for_4_units_minimum_a_for_effective_decontamination_l76_76823

-- Define the concentration function
def concentration (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 then 16 / (8 - x) - 1
  else if 4 < x ∧ x ≤ 10 then 5 - x / 2
  else 0

-- Define the condition for effective concentration
def effective_concentration (c : ℝ) : Prop := c ≥ 4

-- (I) Part (1)
theorem effective_decontamination_time_for_4_units : 
  (∀ x : ℝ, 0 ≤ x → x ≤ 8 → effective_concentration (4 * concentration x)) :=
sorry

-- (II) Part (2a)
noncomputable def combined_concentration (x a : ℝ) : ℝ :=
  let c1 := 2 * (5 - x / 2)
  let c2 := a * (16 / (8 - (x - 6)) - 1)
  c1 + c2

-- (II) Part (2b)
theorem minimum_a_for_effective_decontamination :
  (∀ x : ℝ, 6 ≤ x → x ≤ 10 → effective_concentration ((combined_concentration x (24 - 16 * Real.sqrt 2)))) :=
sorry

noncomputable def min_a: ℝ := 24 - 16 * Real.sqrt 2

lemma effective_min_a: min_a = 1.6 :=
sorry

end effective_decontamination_time_for_4_units_minimum_a_for_effective_decontamination_l76_76823


namespace f_odd_max_value_l76_76245

-- Definitions based on conditions
noncomputable def f (x : ℝ) : ℝ := if x ∈ Icc (-1:ℝ) 0 then (1 / 4^x - 1 / 2^x) else 0

-- Lean statement using conditions and answers
theorem f_odd_max_value : 
  (∀ x ∈ Icc (0 : ℝ) 1, f(x) = 2^x - 4^x) ∧ 
  (∀ x ∈ Icc (0 : ℝ) 1, f x ≤ 0) :=
by 
  sorry

end f_odd_max_value_l76_76245


namespace used_car_percentage_l76_76960

-- Define the variables and conditions
variables (used_car_price original_car_price : ℕ) (h_used_car_price : used_car_price = 15000) (h_original_price : original_car_price = 37500)

-- Define the statement to prove the percentage
theorem used_car_percentage (h : used_car_price / original_car_price * 100 = 40) : true :=
sorry

end used_car_percentage_l76_76960


namespace actual_plot_area_in_acres_l76_76207

-- Define the conditions
def base1_cm := 18
def base2_cm := 12
def height_cm := 8
def scale_cm_to_miles := 5
def sq_mile_to_acres := 640

-- Prove the question which is to find the actual plot area in acres
theorem actual_plot_area_in_acres : 
  (1/2 * (base1_cm + base2_cm) * height_cm * (scale_cm_to_miles ^ 2) * sq_mile_to_acres) = 1920000 :=
by
  sorry

end actual_plot_area_in_acres_l76_76207


namespace plane_eq_of_proj_l76_76002

-- Define the vectors w and the projection constraint
def w : ℝ × ℝ × ℝ := (1, 3, -1)
def proj_w_v : ℝ × ℝ × ℝ := (-3, -9, 3)

-- Assert the vector v and utilize the projection condition to deduce the plane equation.
theorem plane_eq_of_proj (v : ℝ × ℝ × ℝ) 
  (h : proj_w_v = ((v.1 + 3 * v.2 - v.3) / 11) • w) : 
  v.1 + 3 * v.2 - v.3 + 33 = 0 := sorry

end plane_eq_of_proj_l76_76002


namespace num_solutions_l76_76272

def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem num_solutions : 
  let valid_n (n : ℤ) := (n ≠ 25) ∧ (n ≥ 0 ∧ n ≤ 24) in
  (∑ n in (finset.range 25).filter (λ n, is_square (n / (25 - n))), 1) = 2 :=
sorry

end num_solutions_l76_76272


namespace no_solutions_f_eq_f_neg_l76_76482

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_not_defined_at_zero : ¬(∃ (x : ℝ), x = 0 ∧ f x = f 0) := 
sorry

lemma functional_equation (x : ℝ) (hx : x ≠ 0) : 
  f(x) + 2 * f(1 / x) = 3 * x := sorry

theorem no_solutions_f_eq_f_neg (x : ℝ) (hx : x ≠ 0) : ¬(f x = f (-x)) :=
sorry

end no_solutions_f_eq_f_neg_l76_76482


namespace no_elimination_method_l76_76095

theorem no_elimination_method
  (x y : ℤ)
  (h1 : x + 3 * y = 4)
  (h2 : 2 * x - y = 1) :
  ¬ (∀ z : ℤ, z = x + 3 * y - 3 * (2 * x - y)) →
  ∃ x y : ℤ, x + 3 * y - 3 * (2 * x - y) ≠ 0 := sorry

end no_elimination_method_l76_76095


namespace total_books_l76_76037

def sam_books := 110
def joan_books := 102
def tom_books := 125
def alice_books := 97

theorem total_books : sam_books + joan_books + tom_books + alice_books = 434 :=
by
  sorry

end total_books_l76_76037


namespace least_number_of_plates_to_get_matching_pair_l76_76572

noncomputable def plates_in_cabinet := {white : ℕ // white > 0 ∨ white = 0}

def green_plates : ℕ := 6
def red_plates : ℕ := 8
def pink_plates : ℕ := 4
def purple_plates : ℕ := 10

theorem least_number_of_plates_to_get_matching_pair :
  ∃ n, n ≥ 6 ∧ ∀ plates_in_cabinet, ∃ (p1 p2 : {x // 0 ≤ x ∧ x ≤ n}), p1 ≠ p2 → color p1 = color p2 :=
begin
  sorry
end

end least_number_of_plates_to_get_matching_pair_l76_76572


namespace percent_increase_in_movie_length_is_16_l76_76405

def previousMovieLength : ℕ := 120
def previousMovieCostPerMinute : ℕ := 50
def newestMovieCostPerMinute : ℕ := previousMovieCostPerMinute * 2
def totalCostNewestMovie : ℕ := 1920

theorem percent_increase_in_movie_length_is_16 :
  let L := totalCostNewestMovie / newestMovieCostPerMinute in
  let increaseInLength := L - previousMovieLength in
  let P := (increaseInLength * 100) / previousMovieLength in
  P = 16 := by
  sorry

end percent_increase_in_movie_length_is_16_l76_76405


namespace shaded_area_percentage_is_100_l76_76586

-- Definitions and conditions
def square_side := 6
def square_area := square_side * square_side

def rect1_area := 2 * 2
def rect2_area := (5 * 5) - (3 * 3)
def rect3_area := 6 * 6

-- Percentage shaded calculation
def shaded_area := square_area
def percentage_shaded := (shaded_area / square_area) * 100

-- Lean 4 statement for the problem
theorem shaded_area_percentage_is_100 :
  percentage_shaded = 100 :=
by
  sorry

end shaded_area_percentage_is_100_l76_76586


namespace integer_root_is_neg_six_l76_76063

-- Define the context and conditions.
variable (p q : Rat)
variable (root1 root2 root3 : Rat)
variable [IsRational root1]
variable [IsRational root2]
variable [IsRational root3]

-- Define roots.
axiom h_root1 : root1 = 3 - Real.sqrt 5
axiom h_root2 : root2 = 3 + Real.sqrt 5

-- Third root is an integer.
axiom root3_int : ∃ n : Int, root3 = n

-- Polynomial equation and Vieta's formulas.
axiom polynomial_eq : (root1 + root2 + root3 = 0) ∧ (root1 * root2 * root3 = q)

-- Lean statement to prove the integer root is -6.
theorem integer_root_is_neg_six : root3 = -6 := by
  sorry

end integer_root_is_neg_six_l76_76063


namespace smallest_largest_sum_l76_76864

theorem smallest_largest_sum (a b c : ℝ) (m M : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : m = (1/3))
  (h4 : M = 1) :
  (m + M) = 4 / 3 := by
sorry

end smallest_largest_sum_l76_76864


namespace circle_area_proof_l76_76031

noncomputable def area_of_circle (A B : ℝ × ℝ) (y_intercept : ℝ) :=
  let y := 4 in -- Given y-intercept is 4
  let area := 125 * π / 8 in
  sorry

theorem circle_area_proof
  (A B : ℝ × ℝ)
  (hA : A = (4, 9))
  (hB : B = (10, 7))
  (y_intercept : ℝ)
  (hY : y_intercept = 4) :
  area_of_circle A B y_intercept = (125 * π / 8) :=
by
  sorry

end circle_area_proof_l76_76031


namespace ratio_of_tshirts_l76_76017

def spending_on_tshirts (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ) : Prop :=
  Lisa_tshirts = 40 ∧
  Lisa_jeans = Lisa_tshirts / 2 ∧
  Lisa_coats = 2 * Lisa_tshirts ∧
  Carly_jeans = 3 * Lisa_jeans ∧
  Carly_coats = Lisa_coats / 4 ∧
  Lisa_tshirts + Lisa_jeans + Lisa_coats + Carly_tshirts + Carly_jeans + Carly_coats = 230

theorem ratio_of_tshirts 
  (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ)
  (h : spending_on_tshirts Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats)
  : Carly_tshirts / Lisa_tshirts = 1 / 4 := 
sorry

end ratio_of_tshirts_l76_76017


namespace minimum_soldiers_to_add_l76_76797

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76797


namespace sum_of_reciprocals_of_transformed_roots_l76_76427

theorem sum_of_reciprocals_of_transformed_roots :
  ∀ (a b c : ℂ), (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -2) :=
by
  intros a b c ha hb hc habc
  sorry

end sum_of_reciprocals_of_transformed_roots_l76_76427


namespace sum_a5_a8_eq_six_l76_76390

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∀ {m n : ℕ}, a (m + 1) / a m = a (n + 1) / a n

theorem sum_a5_a8_eq_six (h_seq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36) :
  a 5 + a 8 = 6 := 
sorry

end sum_a5_a8_eq_six_l76_76390


namespace domain_of_f_eq_real_f_monotonically_increasing_on_interval_l76_76320

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log 3 (a * x^2 + 3 * x + a + 5/4)

theorem domain_of_f_eq_real (a : ℝ) :
  (∀ x, a * x^2 + 3 * x + a + 5/4 > 0) ↔ (a > 1) :=
sorry

theorem f_monotonically_increasing_on_interval (a : ℝ) :
  (∀ x y, -1 / 4 < x → x < -1 / 8 → -1 / 4 < y → y < -1 / 8 → x ≤ y → f a x ≤ f a y) ↔ 
  (-8 / 17 ≤ a ∧ a ≤ 6) :=
sorry

end domain_of_f_eq_real_f_monotonically_increasing_on_interval_l76_76320


namespace trigonometric_identity_l76_76357

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : sin (α + β) + cos (α + β) = 2 * sqrt 2 * cos (α + π / 4) * sin β) : 
  tan (α - β) = -1 :=
sorry

end trigonometric_identity_l76_76357


namespace probability_multiple_of_3_or_5_l76_76077

theorem probability_multiple_of_3_or_5 :
  let count_multiples := (finset.filter (λ n, n % 3 = 0 ∨ n % 5 = 0) (finset.range 21)).card
  let total_tickets := 20
  (count_multiples : ℚ) / total_tickets = 9 / 20 :=
by
  sorry

end probability_multiple_of_3_or_5_l76_76077


namespace not_diff_of_squares_count_l76_76334

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l76_76334


namespace number_of_partitions_indistinguishable_balls_into_boxes_l76_76344

/-- The number of distinct ways to partition 6 indistinguishable balls into 3 indistinguishable boxes is 7. -/
theorem number_of_partitions_indistinguishable_balls_into_boxes :
  ∃ n : ℕ, n = 7 := sorry

end number_of_partitions_indistinguishable_balls_into_boxes_l76_76344


namespace max_discount_rate_l76_76193

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l76_76193


namespace quadratic_root_and_a_value_l76_76816

theorem quadratic_root_and_a_value (a : ℝ) (h1 : (a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) (h2 : a + 3 ≠ 0) : a = 3 :=
by
  sorry

end quadratic_root_and_a_value_l76_76816


namespace city_visited_by_B_is_A_l76_76463

variable (City : Type) (A B C : City)

-- Definitions for the conditions
variable (visited_by_A visited_by_B visited_by_C : City → Prop)
variable (has_visited_more_cities : ∀ A, (∃ x, visited_by_A x → ¬ visited_by_B x) ∧ (∃ x, visited_by_B x → ¬ visited_by_A x) → Prop)

-- Conditions
axiom condition1 : has_visited_more_cities visited_by_A visited_by_B
axiom condition2 : ¬ visited_by_A B
axiom condition3 : ¬ visited_by_B C
axiom condition4 : ∀ x, visited_by_A x ↔ visited_by_B x ↔ visited_by_C x

-- Theorem stating the city visited by B
theorem city_visited_by_B_is_A : visited_by_B A :=
by
  sorry

end city_visited_by_B_is_A_l76_76463


namespace preferred_dividend_rate_l76_76438

noncomputable def dividend_rate_on_preferred_shares
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℕ)
  (semi_annual_dividend_common : ℚ)
  (total_annual_dividend : ℚ)
  (dividend_rate_preferred : ℚ) : Prop :=
  preferred_shares * par_value * (dividend_rate_preferred / 100) +
  2 * (common_shares * par_value * (semi_annual_dividend_common / 100)) =
  total_annual_dividend

theorem preferred_dividend_rate
  (h1 : 1200 = 1200)
  (h2 : 3000 = 3000)
  (h3 : 50 = 50)
  (h4 : 3.5 = 3.5)
  (h5 : 16500 = 16500) :
  dividend_rate_on_preferred_shares 1200 3000 50 3.5 16500 10 :=
by sorry

end preferred_dividend_rate_l76_76438


namespace polynomials_equal_if_coeff_conditions_l76_76413

theorem polynomials_equal_if_coeff_conditions (f g : ℕ[X]) (m : ℕ) 
  (a b : ℕ) (hfa : f.eval a = g.eval a) (hfb : f.eval b = g.eval b)
  (hcoeff : ∀ k, f.coeff k = 0 ∨ f.coeff k ≤ m) (hb_gt_m : b > m) :
  f = g := 
sorry

end polynomials_equal_if_coeff_conditions_l76_76413


namespace cost_per_dozen_l76_76859

theorem cost_per_dozen (total_cost : ℝ) (total_rolls dozens : ℝ) (cost_per_dozen : ℝ) (h₁ : total_cost = 15) (h₂ : total_rolls = 36) (h₃ : dozens = total_rolls / 12) (h₄ : cost_per_dozen = total_cost / dozens) : cost_per_dozen = 5 :=
by
  sorry

end cost_per_dozen_l76_76859


namespace length_increase_percentage_l76_76486

variables {L B : ℝ} -- Original length and breadth
variables (x : ℂ) -- Percentage increase in length

def original_area (L B : ℝ) : ℝ := L * B
def increased_breadth (B : ℝ) : ℝ := B * 1.06
def increased_length (L : ℝ) (x : ℂ) : ℝ := L * (1 + x)

def new_area (L B : ℝ) (x : ℂ) : ℝ := increased_length L x * increased_breadth B

theorem length_increase_percentage :
  (new_area L B x = original_area L B * 1.0918) → x = 0.03 :=
begin
  sorry
end

end length_increase_percentage_l76_76486


namespace opposite_of_neg_two_l76_76547

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l76_76547


namespace count_diff_squares_not_representable_1_to_1000_l76_76337

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l76_76337


namespace min_soldiers_to_add_l76_76804

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76804


namespace talia_total_distance_l76_76467

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end talia_total_distance_l76_76467


namespace probability_of_less_than_6_l76_76860

theorem probability_of_less_than_6 {p : ℚ} (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → p = 1/6) : 
  (∑ k in (finset.range 5).image (λ n, n + 1), p) = 5/6 :=
by
  sorry

end probability_of_less_than_6_l76_76860


namespace number_of_students_l76_76023

def candiesPerStudent : ℕ := 2
def totalCandies : ℕ := 18
def expectedStudents : ℕ := 9

theorem number_of_students :
  totalCandies / candiesPerStudent = expectedStudents :=
sorry

end number_of_students_l76_76023


namespace quadrilateral_area_l76_76844

noncomputable def area_quadrilateral_ABCD 
  (A B C D E : ℝ) 
  (hBAC : ∠ BAC = π / 2)
  (h_AB : AB = 15)
  (h_BC : BC = 20) 
  (h_DC : DC = 26)
  (h_midE : E = (A + C) / 2): ℝ := 
195

theorem quadrilateral_area (A B C D E : ℝ)
  (hBAC : ∠ BAC = π / 2)
  (h_AB : AB = 15)
  (h_BC : BC = 20) 
  (h_DC : DC = 26)
  (h_midE : E = (A + C) / 2) :
  area_quadrilateral_ABCD A B C D E hBAC h_AB h_BC h_DC h_midE = 195 :=
sorry

end quadrilateral_area_l76_76844


namespace exists_perpendicular_line_through_point_l76_76285

open EuclideanGeometry

structure Circle (V : Type*) [InnerProductSpace ℝ V] extends DiskV :=
  (radius : ℝ)

noncomputable def construct_perpendicular (V : Type*) [InnerProductSpace ℝ V]
  (k : Circle V) (e : Line V) (P : V)
  (center_k : e.contains k.center)
  (not_on_k : ¬k.on_circle P)
  (not_on_e : ¬P ∈ e) : Line V :=
  sorry -- Construction logic here

theorem exists_perpendicular_line_through_point
  (V : Type*) [InnerProductSpace ℝ V]
  (k : Circle V) (e : Line V) (P : V)
  (center_k : e.contains k.center)
  (not_on_k : ¬k.on_circle P)
  (not_on_e : ¬P ∈ e) :
  ∃ l : Line V, l.perpendicular e ∧ l.contains P :=
begin
  use construct_perpendicular V k e P center_k not_on_k not_on_e,
  sorry -- Proof that the construct_perpendicular function indeed creates the required line
end

end exists_perpendicular_line_through_point_l76_76285


namespace natural_number_solution_condition_l76_76901

variable (C : ℝ) -- constant C
variable (b : ℝ) -- parameter b
variable (a : ℝ) -- parameter a
def equation_has_nat_solution (x : ℕ) : Prop :=
  C - x = 2 * b - 2 * a * x

theorem natural_number_solution_condition (C : ℝ) (x : ℕ) :
  ∀ a b : ℝ, b = 7 → (∃ x : ℕ, equation_has_nat_solution C b a x) ↔ (∃ k : ℕ, a = (k + 1) / 2 ∧ a > 1 / 2 := by
  sorry

end natural_number_solution_condition_l76_76901


namespace total_games_played_l76_76277

variable (CarlaGames : ℕ)
variable (FrankieGames : ℕ)
variable (TotalGames : ℕ)

axiom Carla_won_20 : CarlaGames = 20
axiom Frankie_won_half : FrankieGames = CarlaGames / 2

theorem total_games_played
  (h1 : CarlaGames = 20)
  (h2 : FrankieGames = CarlaGames / 2) :
  TotalGames = CarlaGames + FrankieGames :=
by 
  unfold TotalGames
  rw [h1, h2]
  sorry

end total_games_played_l76_76277


namespace probability_not_choose_own_dress_l76_76598

theorem probability_not_choose_own_dress :
  let total_ways := factorial 3
  let favorable_ways := 2
  let probability := favorable_ways / total_ways
  probability = 1 / 3 :=
by
  sorry

end probability_not_choose_own_dress_l76_76598


namespace minimum_soldiers_to_add_l76_76779

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l76_76779


namespace log_equal_products_l76_76668

theorem log_equal_products (x : ℝ) :
  (log x (x - 3 / 2) = log (x - 3 / 2) (x - 3) * log (x - 3) x ∨
   log (x - 3 / 2) (x - 3) = log x (x - 3 / 2) * log (x - 3) x ∨
   log (x - 3) x = log x (x - 3 / 2) * log (x - 3 / 2) (x - 3))
  → (x = 7 / 2 ∨ x = (3 + sqrt 13) / 2) :=
begin
  sorry
end

end log_equal_products_l76_76668


namespace interview_passing_probability_l76_76391

def probability_of_passing_interview (p : ℝ) : ℝ :=
  p + (1 - p) * p + (1 - p) * (1 - p) * p

theorem interview_passing_probability : probability_of_passing_interview 0.7 = 0.973 :=
by
  -- proof steps to be filled
  sorry

end interview_passing_probability_l76_76391


namespace min_soldiers_to_add_l76_76809

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l76_76809


namespace opposite_of_neg_two_is_two_l76_76523

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l76_76523


namespace ones_digit_of_prime_sequence_l76_76685

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l76_76685


namespace misha_cards_digit_example_l76_76876

-- Define variables representing the digits
variables {L O M H C : ℕ}

-- Define the main equation
def main_equation : Prop := L + O / M + O + H + O / C = 20

-- Conditions
def digit_conditions: Prop :=
  M > O ∧ C > O

theorem misha_cards_digit_example
  (digit_conditions)
  : main_equation :=
sorry

end misha_cards_digit_example_l76_76876


namespace euro_operation_example_l76_76986

def euro_operation (x y : ℕ) : ℕ := 3 * x * y

theorem euro_operation_example : euro_operation 3 (euro_operation 4 5) = 540 :=
by sorry

end euro_operation_example_l76_76986


namespace number_of_solutions_l76_76669

theorem number_of_solutions : 
  (∃ (x y : ℤ), x * y + 5 * x + 7 * y = 29) ∧
  (card (set_of (λ p : ℤ × ℤ, p.1 * p.2 + 5 * p.1 + 7 * p.2 = 29)) = 14) :=
sorry

end number_of_solutions_l76_76669


namespace younger_son_age_after_30_years_l76_76915

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l76_76915


namespace football_complex_product_l76_76018

/-- A representation of a football made of stitched polygonal panels, each edge stitched with a specific color.
    We need to prove that it is possible to place a complex number (not equal to 1) at each vertex 
    such that the product of the complex numbers at the vertices of any polygonal panel is equal to 1. --/
theorem football_complex_product :
  ∃ (assign_comp_num : vertex → ℂ),
    (∀ (v : vertex), assign_comp_num v ≠ 1) ∧
    (∀ (p : polygonal_panel), (∏ v in vertices_of p, assign_comp_num v) = 1) := 
sorry

end football_complex_product_l76_76018


namespace metro_travel_interval_l76_76992

theorem metro_travel_interval :
  let p := 7 / 12 in 
  let E_to := 17 - 6 * p in
  let E_back := 11 + 6 * p in
  let avg_difference := E_back - E_to = 1 in
  let interval := 5 / 4 / (1 - p) in
  avg_difference -> interval = 3 :=
by sorry

end metro_travel_interval_l76_76992


namespace part2_l76_76725

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - x

theorem part2 (a : ℝ) (h : a > 0) (x : ℝ) : f a x < (a - 1) * Real.log a + a^2 := 
  sorry

end part2_l76_76725


namespace average_side_lengths_of_squares_l76_76474

theorem average_side_lengths_of_squares (A1 A2 A3 : ℝ) (hA1 : A1 = 25) (hA2 : A2 = 64) (hA3 : A3 = 121) : 
  (real.sqrt A1 + real.sqrt A2 + real.sqrt A3) / 3 = 8 := 
by 
  sorry

end average_side_lengths_of_squares_l76_76474


namespace max_discount_rate_l76_76149

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l76_76149


namespace hyperdeficient_count_l76_76422

def sum_of_divisors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum id

def is_hyperdeficient (n : ℕ) : Prop := sum_of_divisors (sum_of_divisors n) = n + 3

theorem hyperdeficient_count : (Finset.range 1000).filter is_hyperdeficient = {3} := by sorry

end hyperdeficient_count_l76_76422


namespace sequence_solution_l76_76253

theorem sequence_solution (a : ℕ → ℝ) (h : ∀ n, (∑ i in Finset.range n.succ, (a i) ^ 3) = (∑ i in Finset.range n.succ, a i) ^ 2) (h_pos : ∀ n, 0 < a n) : ∀ n, a n = n := 
by 
  sorry

end sequence_solution_l76_76253


namespace tennis_players_meeting_l76_76117

theorem tennis_players_meeting
  (M J : Type)
  (masters : fin 30 → M) (juniors : fin 30 → J)
  (plays : M → M → Prop)
  (playsJ : J → J → Prop)
  (playsMJ : M → J → Prop) :
  -- Each master plays with one other master and 15 juniors
  (∀ m : M, ∃ m' : M, plays m m' ∧ ∃ js : fin 16 → J, ∀ j, (js j) ∈ juniors ∧ playsMJ m (js j)) →
  -- Each junior plays with one other junior and 15 masters
  (∀ j : J, ∃ j' : J, playsJ j j' ∧ ∃ ms : fin 16 → M, ∀ m, (ms m) ∈ masters ∧ playsMJ (ms m) j) →
  -- Exist two masters and two juniors satisfying the problem conditions
  (∃ (m1 m2 : M) (j1 j2 : J), 
    plays m1 m2 ∧ playsJ j1 j2 ∧ 
    (playsMJ m1 j1 ∨ playsMJ m1 j2 ∨ playsMJ m2 j1 ∨ playsMJ m2 j2) ∧ 
    (playsMJ j1 m1 ∨ playsMJ j1 m2 ∨ playsMJ j2 m1 ∨ playsMJ j2 m2)) :=
sorry

end tennis_players_meeting_l76_76117


namespace parcel_cost_l76_76052

theorem parcel_cost (P : ℤ) (hP : P ≥ 1) : 
  (P ≤ 5 → C = 15 + 4 * (P - 1)) ∧ (P > 5 → C = 15 + 4 * (P - 1) - 10) :=
sorry

end parcel_cost_l76_76052


namespace part1_part2_l76_76716

-- Define variables and conditions
variables (m n d : ℕ)

-- Given conditions
def gcd_condition : d = Nat.gcd m n := sorry
def pos_ints : m > 0 ∧ n > 0 := sorry
def x : ℕ := 2^m - 1
def y : ℕ := 2^n + 1

-- Part 1: If m / d is odd, then x and y are coprime
theorem part1 (h : (m / d) % 2 = 1) : Nat.gcd x y = 1 :=
by
  rw [x, y]
  have hDiv : d ∣ m := sorry
  have hDiv' : d ∣ n := sorry
  sorry

-- Part 2: If m / d is even, then the GCD of x and y is 2^d + 1
theorem part2 (h : (m / d) % 2 = 0) : Nat.gcd x y = 2^d + 1 :=
by
  rw [x, y]
  have hDiv : d ∣ m := sorry
  have hDiv' : d ∣ n := sorry
  sorry

end part1_part2_l76_76716


namespace probability_two_girls_l76_76825

theorem probability_two_girls (total_students girls boys : ℕ) (htotal : total_students = 6) (hg : girls = 4) (hb : boys = 2) :
  (Nat.choose girls 2 / Nat.choose total_students 2 : ℝ) = 2 / 5 := by
  sorry

end probability_two_girls_l76_76825


namespace correct_number_of_pairings_l76_76647

noncomputable def number_of_pairings :
  (bowls : Finset String) (glasses : Finset String) (green_fixed : Bool) → Nat :=
sorry

theorem correct_number_of_pairings :
  number_of_pairings
    { "red", "blue", "yellow", "green", "orange", "purple" }.to_finset
    { "red", "blue", "yellow", "green" }.to_finset
    true
    = 16 :=
sorry

end correct_number_of_pairings_l76_76647
