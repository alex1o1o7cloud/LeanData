import Mathlib
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.SField
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Combinatorial
import Mathlib.Combinatorics.CombinatorialChoice
import Mathlib.Combinatorics.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.CircumscribedCircle
import Mathlib.Geometry.Euclidean.InscribedCircle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Stats.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace financial_calculations_correct_l348_348521

noncomputable def revenue : ℝ := 2500000
noncomputable def expenses : ℝ := 1576250
noncomputable def loan_payment_per_month : ℝ := 25000
noncomputable def number_of_shares : ℕ := 1600
noncomputable def ceo_share_percentage : ℝ := 0.35

theorem financial_calculations_correct :
  let net_profit := (revenue - expenses) - (revenue - expenses) * 0.2 in
  let total_loan_payment := loan_payment_per_month * 12 in
  let dividends_per_share := (net_profit - total_loan_payment) / number_of_shares in
  let ceo_dividends := dividends_per_share * ceo_share_percentage * number_of_shares in
  net_profit = 739000 ∧
  total_loan_payment = 300000 ∧
  dividends_per_share = 274 ∧
  ceo_dividends = 153440 :=
begin
  sorry
end

end financial_calculations_correct_l348_348521


namespace max_distinct_integers_sum_100_or_101_l348_348060

theorem max_distinct_integers_sum_100_or_101 :
  ∃ (seq : ℕ → ℤ), (∀ k, (∑ i in range 11, seq (k + i)) = 100 ∨ (∑ i in range 11, seq (k + i)) = 101) → (# (set.of $ range 0 23) = 22) :=
begin
  sorry
end

end max_distinct_integers_sum_100_or_101_l348_348060


namespace product_of_values_l348_348164

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := |2 * x| + 4 = 38

-- State the theorem
theorem product_of_values : ∃ x1 x2 : ℝ, satisfies_eq x1 ∧ satisfies_eq x2 ∧ x1 * x2 = -289 := 
by
  sorry

end product_of_values_l348_348164


namespace shaded_area_concentric_circles_l348_348386

theorem shaded_area_concentric_circles 
  (CD_length : ℝ) (r1 r2 : ℝ) 
  (h_tangent : CD_length = 100) 
  (h_radius1 : r1 = 60) 
  (h_radius2 : r2 = 40) 
  (tangent_condition : CD_length = 2 * real.sqrt (r1^2 - r2^2)) :
  ∃ area : ℝ, area = π * (r1^2 - r2^2) ∧ area = 2000 * π :=
by
  use π * (r1^2 - r2^2)
  have h1 : r1^2 = 3600 := by { rw h_radius1, norm_num }
  have h2 : r2^2 = 1600 := by { rw h_radius2, norm_num }
  rw [h1, h2]
  simp
  sorry

end shaded_area_concentric_circles_l348_348386


namespace find_k_l348_348249

theorem find_k (x k : ℝ) (h : ((x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) ∧ k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l348_348249


namespace new_lengths_proof_l348_348831

-- Define the initial conditions
def initial_lengths : List ℕ := [15, 20, 24, 26, 30]
def original_average := initial_lengths.sum / initial_lengths.length
def average_decrease : ℕ := 5
def median_unchanged : ℕ := 24
def range_unchanged : ℕ := 15
def new_average := original_average - average_decrease

-- Assume new lengths
def new_lengths : List ℕ := [9, 9, 24, 24, 24]

-- Proof statement
theorem new_lengths_proof :
  (new_lengths.sorted.nth 2 = initial_lengths.sorted.nth 2) ∧
  (new_lengths.maximum - new_lengths.minimum = range_unchanged) ∧
  (new_lengths.sum / new_lengths.length = new_average) :=
by
  sorry

end new_lengths_proof_l348_348831


namespace sum_of_n_over_a_n_l348_348585

noncomputable def a : ℕ → ℝ
| 0     := 2 / 3
| (n+1) := 2 * a n / (a n + 1)

lemma geometric_sequence_1_div_an_minus_1 (n : ℕ) :
  let s := (λ n, 1 / (a n) - 1) in
  s 0 = 1 / (2 / 3) - 1 ∧ ∀ n, s (n + 1) = (s n) / 2 :=
by sorry

theorem sum_of_n_over_a_n (n : ℕ) :
  let seq := (λ n, n / (a n)) in
  ∑ i in finset.range n, seq i = (n^2 + n + 4) / 2 - (2 + n) / 2^n :=
by sorry

end sum_of_n_over_a_n_l348_348585


namespace range_of_c_l348_348231

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem range_of_c :
  ∀ (c : ℝ), (∀ (x : ℝ), 0 < x ∧ x ≤ Real.exp 1 → f(x) - f 1 ≥ c * (x - 1)) ↔ (c ∈ set.Icc (-1) ((Real.exp 1 - 1)⁻¹)) :=
by
  sorry

end range_of_c_l348_348231


namespace minimum_value_l348_348706

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
    6 ≤ (x^2 + 2*y^2) / (x + y) + (x^2 + 2*z^2) / (x + z) + (y^2 + 2*z^2) / (y + z) :=
by
  sorry

end minimum_value_l348_348706


namespace units_digit_sum_l348_348167

theorem units_digit_sum : 
  let fact_sum := (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10!)
  let pow2_sum := (2^1 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 + 2^10)
  units_digit (fact_sum + pow2_sum) = 9 :=
by
  sorry

end units_digit_sum_l348_348167


namespace area_of_triangle_l348_348922

-- Define points A, B, and C
def A := (3, 4)
def B := (-1, 2)
def C := (5, -6)

-- Area function for a triangle given its vertices
def triangle_area (P Q R : (ℤ × ℤ)) : ℚ :=
  (1 / 2 : ℚ) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- The statement we want to prove
theorem area_of_triangle : triangle_area A B C = 22 := 
  by sorry

end area_of_triangle_l348_348922


namespace pyramid_center_to_plane_distance_l348_348667

noncomputable def distance_from_center_to_plane (l : ℝ) (angle : ℝ) : ℝ :=
  if h_angle : angle = 60 then (Real.sqrt 21) / 14 else sorry

theorem pyramid_center_to_plane_distance 
  (P A B C D E F O : Type) 
  (slant_height : P → ℝ) 
  (angle_between_face_and_base : P → ℝ) 
  (is_midpoint_E : P → B → C → Prop) 
  (is_midpoint_F : P → C → D → Prop) 
  (is_center_O : P → ℝ)
  (Pyramids : Type)
  (in_pyramid : P → Pyramids)
  (PEF : P → Prop) 
  (h1 : slant_height P = l)
  (h2 : angle_between_face_and_base P = 60)
  (h3 : is_midpoint_E P B C) 
  (h4 : is_midpoint_F P C D)  
  (h5 : is_center_O P) : 
  distance_from_center_to_plane l 60 = (Real.sqrt 21) / 14 := 
begin
  sorry
end

end pyramid_center_to_plane_distance_l348_348667


namespace apples_mass_left_l348_348114

theorem apples_mass_left (initial_kidney golden canada fuji granny : ℕ)
                         (sold_kidney golden canada fuji granny : ℕ)
                         (left_kidney golden canada fuji granny : ℕ) :
  initial_kidney = 26 → sold_kidney = 15 → left_kidney = 11 →
  initial_golden = 42 → sold_golden = 28 → left_golden = 14 →
  initial_canada = 19 → sold_canada = 12 → left_canada = 7 →
  initial_fuji = 35 → sold_fuji = 20 → left_fuji = 15 →
  initial_granny = 22 → sold_granny = 18 → left_granny = 4 →
  left_kidney = initial_kidney - sold_kidney ∧
  left_golden = initial_golden - sold_golden ∧
  left_canada = initial_canada - sold_canada ∧
  left_fuji = initial_fuji - sold_fuji ∧
  left_granny = initial_granny - sold_granny := by sorry

end apples_mass_left_l348_348114


namespace determine_head_start_l348_348472

def head_start (v : ℝ) (s : ℝ) : Prop :=
  let a_speed := 2 * v
  let distance := 142
  distance / a_speed = (distance - s) / v

theorem determine_head_start (v : ℝ) : head_start v 71 :=
  by
    sorry

end determine_head_start_l348_348472


namespace number_of_sets_of_four_teams_l348_348295

theorem number_of_sets_of_four_teams
  (n : ℕ)
  (h1 : ∀ t, t < n → (wins t = 15 ∧ losses t = 6))
  (h2 : ∀ t, t < n → wins t + losses t = 21)
  : n = 22 →
    (card {s : set (fin n) | s.card = 4 ∧ (∀ (A B C D : fin n), A ∈ s ∧ B ∈ s ∧ C ∈ s ∧ D ∈ s → A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A beats B ∧ B beats C ∧ C beats D ∧ D beats A)}) = 1643 :=
by
  sorry

end number_of_sets_of_four_teams_l348_348295


namespace seating_arrangements_l348_348028

theorem seating_arrangements :
  let boys := {B1, B2, B3}
  let girls := {G1, G2, G3}
  let seats := [B1, G1, B2, G2, B3, G3]

  (∀ (i : ℕ), i < 5 →
    ((seats[i] ∈ boys ∧ seats[i + 1] ∈ girls) ∨ 
     (seats[i] ∈ girls ∧ seats[i + 1] ∈ boys)) ∧
    B1 ∈ seats ∧ G1 ∈ seats ∧ ((seats.indexOf B1) + 1 = seats.indexOf G1 ∨ (seats.indexOf B1) - 1 = seats.indexOf G1)) →
  s = 40 :=
sorry

end seating_arrangements_l348_348028


namespace axis_of_symmetry_of_parabola_l348_348792

-- Definitions (from conditions):
def quadratic_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root_of_quadratic (a b c x : ℝ) : Prop := quadratic_equation a b c x = 0

-- Given conditions
variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)
variable (h_root1 : is_root_of_quadratic a b c 1)
variable (h_root2 : is_root_of_quadratic a b c 5)

-- Problem statement
theorem axis_of_symmetry_of_parabola : (3 : ℝ) = (1 + 5) / 2 :=
by
  -- proof omitted
  sorry

end axis_of_symmetry_of_parabola_l348_348792


namespace find_f_7_5_l348_348333

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_2  : ∀ x, f (x + 2) = -f x
axiom initial_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- Proof goes here
  sorry

end find_f_7_5_l348_348333


namespace jim_age_l348_348319

variable (J F S : ℕ)

theorem jim_age (h1 : J = 2 * F) (h2 : F = S + 9) (h3 : J - 6 = 5 * (S - 6)) : J = 46 := 
by
  sorry

end jim_age_l348_348319


namespace max_sin_a_l348_348709

theorem max_sin_a (a b : ℝ) (h : sin (a - b) = sin a - sin b) : 
  sin a ≤ 1 :=
sorry

end max_sin_a_l348_348709


namespace polyhedron_volume_l348_348027

/-- Each 12 cm × 12 cm square is cut into two right-angled isosceles triangles by joining the midpoints of two adjacent sides. 
    These six triangles are attached to a regular hexagon to form a polyhedron.
    Prove that the volume of the resulting polyhedron is 864 cubic cm. -/
theorem polyhedron_volume :
  let s : ℝ := 12
  let volume_of_cube := s^3
  let volume_of_polyhedron := volume_of_cube / 2
  volume_of_polyhedron = 864 := 
by
  sorry

end polyhedron_volume_l348_348027


namespace probability_of_rolling_five_on_six_sided_die_l348_348103

theorem probability_of_rolling_five_on_six_sided_die :
  let S := {1, 2, 3, 4, 5, 6}
  let |S| := 6
  let A := {5}
  let |A| := 1
  probability A S = 1 / 6 := by
  -- Proof goes here
  sorry

end probability_of_rolling_five_on_six_sided_die_l348_348103


namespace bob_final_amount_l348_348507

-- Define initial amount
def initial_amount : ℝ := 80

-- Define money spent on Monday
def spent_monday : ℝ := initial_amount / 2

-- Define money left after Monday
def left_after_monday : ℝ := initial_amount - spent_monday

-- Define money spent on Tuesday
def spent_tuesday : ℝ := left_after_monday / 5

-- Define money left after Tuesday
def left_after_tuesday : ℝ := left_after_monday - spent_tuesday

-- Define money spent on Wednesday
def spent_wednesday : ℝ := (3 / 8) * left_after_tuesday

-- Define money left after Wednesday
def left_after_wednesday : ℝ := left_after_tuesday - spent_wednesday

-- The theorem stating the final amount of money left
theorem bob_final_amount : left_after_wednesday = 20 := by
  have h0 : initial_amount = 80 := rfl
  have h1 : spent_monday = 40 := by 
    simp [initial_amount, spent_monday]
  have h2 : left_after_monday = 40 := by 
    simp [left_after_monday, initial_amount, spent_monday]
    norm_num
  have h3 : spent_tuesday = 8 := by
    simp [spent_tuesday, left_after_monday]
    norm_num
  have h4 : left_after_tuesday = 32 := by
    simp [left_after_tuesday, left_after_monday, spent_tuesday]
    norm_num
  have h5 : spent_wednesday = 12 := by
    simp [spent_wednesday, left_after_tuesday]
    norm_num
  have h6 : left_after_wednesday = 20 := by
    simp [left_after_wednesday, left_after_tuesday, spent_wednesday]
    norm_num
  exact h6

end bob_final_amount_l348_348507


namespace find_principal_l348_348072

theorem find_principal 
    (SI : ℤ) (R : ℤ) (T : ℤ) (H_SI : SI = 2700) (H_R : R = 6) (H_T : T = 3) :
    (P : ℤ) := 
  P = SI * 100 / (R * T) :=
by {
  have H1 : P * 6 * 3 = 2700 * 100, {
    calc
      P * 6 * 3   = SI * 100    
                 : by sorry,
  },
  show P = 15000, {
    calc
      P = 2700 * 100 / (6 * 3)  
        : by sorry,
  }
}

end find_principal_l348_348072


namespace necessary_but_not_sufficient_l348_348223

variable {x m : ℝ}
variable (p : |x| > 1) (q : x < m)

theorem necessary_but_not_sufficient (hp : ¬p → ¬q) (hns : ¬(¬q → ¬p)) : m ≤ -1 := 
sorry

end necessary_but_not_sufficient_l348_348223


namespace inscribed_sphere_surface_area_l348_348196

theorem inscribed_sphere_surface_area (r : ℝ) (h_eq_triangle : ∀ r, r = (sqrt 3) / 3) :
    4 * π * r^2 = 4 * π / 3 := by
  have r_eq : r = (sqrt 3) / 3 := h_eq_triangle r
  rw [r_eq]
  sorry

end inscribed_sphere_surface_area_l348_348196


namespace ratio_of_smaller_to_trapezoid_l348_348502

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

def ratio_of_areas : ℝ :=
  let side_large := 10
  let side_small := 5
  let area_large := area_equilateral_triangle side_large
  let area_small := area_equilateral_triangle side_small
  let area_trapezoid := area_large - area_small
  area_small / area_trapezoid

theorem ratio_of_smaller_to_trapezoid :
  ratio_of_areas = 1 / 3 :=
sorry

end ratio_of_smaller_to_trapezoid_l348_348502


namespace total_percentage_of_profit_no_discount_is_20_09_l348_348100

-- Define the conditions
def cost_price_item_a : ℝ := 15
def cost_price_item_b : ℝ := 10
def profit_percentage_item_a : ℝ := 0.32
def profit_percentage_item_b : ℝ := 0.20
def units_sold_item_a : ℕ := 100
def units_sold_item_b : ℕ := 200

-- Calculate the expected value
def total_sales_revenue_without_discount : ℝ :=
  (cost_price_item_a * units_sold_item_a + cost_price_item_a * units_sold_item_a * profit_percentage_item_a) +
  (cost_price_item_b * units_sold_item_b + cost_price_item_b * units_sold_item_b * profit_percentage_item_b)

def total_profit_without_discount : ℝ :=
  (cost_price_item_a * units_sold_item_a * profit_percentage_item_a) +
  (cost_price_item_b * units_sold_item_b * profit_percentage_item_b)

def total_percentage_profit_without_discount : ℝ :=
  (total_profit_without_discount / total_sales_revenue_without_discount) * 100

-- The statement we seek to prove
theorem total_percentage_of_profit_no_discount_is_20_09 :
  total_percentage_profit_without_discount ≈ 20.09 :=
by
  sorry

end total_percentage_of_profit_no_discount_is_20_09_l348_348100


namespace function_expression_l348_348613

theorem function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 - 2x :=
sorry

end function_expression_l348_348613


namespace find_a_l348_348616

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 16) = 1

def left_focus : ℝ × ℝ := (-3, 0)

def is_line_through (p : ℝ × ℝ) (k x y : ℝ) : Prop :=
  y = k * (x + p.1)

def point_on_line (x y : ℝ) (k : ℝ) (x1 x2 : ℝ) : Prop :=
  y = k * (x + 3)

def directrix (x : ℝ) := x + 5 + sqrt(9 - 25 * 4 / 5) -- Approximation for directrix

theorem find_a (a : ℝ) : 
  (∀ (k x1 x2 : ℝ) (y1 := k * (x1 + 3)) (y2 := k * (x2 + 3)) (m n := (directrix (a-x1), directrix (a-x2))),
  ellipse x1 y1 →
  ellipse x2 y2 →
  (x1 + 3) * (x2 + 3) - (15 * k * m * n / (3 * (a - x1) * (a - x2))) ∧ a > -3 →
  circle (m, y1) (n, y2) through left_focus) → 
  a = 5 :=
sorry

end find_a_l348_348616


namespace shaded_region_area_l348_348759

open Real

structure Point where
  x : ℝ
  y : ℝ

def E := Point.mk 0 0
def F := Point.mk 2 0
def A := Point.mk 2 5
def D := Point.mk 0 5
def C := Point.mk 0 3
def B := Point.mk 2 3

-- Define the function to calculate the area of a triangle using Shoelace Theorem
def area_2d (P1 P2 P3 : Point) : ℝ :=
  abs (P1.x * P2.y + P2.x * P3.y + P3.x * P1.y - P1.y * P2.x - P2.y * P3.x - P3.y * P1.x) / 2

def shaded_area : ℝ := area_2d E C F + area_2d E F B

theorem shaded_region_area : shaded_area = 6 := sorry

end shaded_region_area_l348_348759


namespace probability_of_same_suit_top_four_cards_l348_348897

-- Definitions based on the conditions.
def standard_deck : Set (Fin 52) := {i | 0 ≤ i ∧ i < 52}

def same_suit_four_cards (cards : Set (Fin 52)) : Prop :=
  ∃ s : Fin 4, ∃ subset : Set (Fin 13), subset.card = 4 ∧ ⊆ {j | ⟨s.val * 13 + j, (s.val * 13 + j) ∈ standard_deck⟩}

-- The theorem to be proved.
theorem probability_of_same_suit_top_four_cards :
  let favorable_outcomes := 4 * (Nat.choose 13 4)
      total_outcomes := Nat.choose 52 4
  in favorable_outcomes %/ total_outcomes = 2860 %/ 270725 :=
sorry

end probability_of_same_suit_top_four_cards_l348_348897


namespace solve_equation_l348_348368

theorem solve_equation : ∃ x : ℝ, (1 + x) / (2 - x) - 1 = 1 / (x - 2) ↔ x = 0 := 
by
  sorry

end solve_equation_l348_348368


namespace smallest_number_divisible_by_15_and_36_l348_348062

theorem smallest_number_divisible_by_15_and_36 : 
  ∃ x, (∀ y, (y % 15 = 0 ∧ y % 36 = 0) → y ≥ x) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_15_and_36_l348_348062


namespace find_length_DF_l348_348296

-- Define the context of parallelogram ABCD and given conditions
variables (A B C D E F : Type*)
variables [parallelogram A B C D]
variables (DE_alt_AB : altitude DE A B)
variables (DF_alt_BC : altitude DF B C)
variables (DC_len : segment_length D C = 15)
variables (EB_len : segment_length E B = 3)
variables (DE_len : segment_length D E = 6)

-- Goal: Prove the length of DF given the conditions
theorem find_length_DF : segment_length D F = 6 :=
sorry

end find_length_DF_l348_348296


namespace rectangular_prism_edges_vertices_faces_sum_l348_348098

theorem rectangular_prism_edges_vertices_faces_sum (a b c : ℕ) (h1: a = 2) (h2: b = 3) (h3: c = 4) : 
  12 + 8 + 6 = 26 :=
by
  sorry

end rectangular_prism_edges_vertices_faces_sum_l348_348098


namespace slope_of_line_through_intersecting_points_of_circles_l348_348554

theorem slope_of_line_through_intersecting_points_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 6*x + 4*y - 5 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 10*x + 16*y + 24 = 0
  ∀ (C D : ℝ × ℝ), circle1 C.1 C.2 → circle2 C.1 C.2 → circle1 D.1 D.2 → circle2 D.1 D.2 → 
  let dx := D.1 - C.1
  let dy := D.2 - C.2
  dx ≠ 0 → dy / dx = 1 / 3 :=
by
  intros
  sorry

end slope_of_line_through_intersecting_points_of_circles_l348_348554


namespace general_term_arithmetic_sequence_l348_348657

theorem general_term_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a4 : a 4 = 10)
  (h_geom : a 3 * a 10 = (a 6) ^ 2) :
  ∀ n, a n = n + 6 :=
begin
  sorry
end

end general_term_arithmetic_sequence_l348_348657


namespace time_for_first_three_workers_l348_348191

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l348_348191


namespace domain_of_f_l348_348783

noncomputable def f (x : ℝ) : ℝ := (Real.log (4 - x)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | 4 - x > 0 ∧ x ≠ 2} = {x : ℝ | x ∈ set.Ioo -∞ 2 ∪ set.Ioo 2 4} :=
by
  sorry

end domain_of_f_l348_348783


namespace alcohol_percentage_after_evaporation_l348_348451

def initial_solution_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def added_alcohol_volume : ℝ := 3.5
def added_water_volume : ℝ := 6.5
def alcohol_evaporation_rate : ℝ := 0.02

theorem alcohol_percentage_after_evaporation :
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_concentration in
  let total_volume_after_addition := initial_solution_volume + added_alcohol_volume + added_water_volume in
  let total_alcohol_before_evaporation := initial_alcohol_amount + added_alcohol_volume in
  let evaporated_alcohol := alcohol_evaporation_rate * total_alcohol_before_evaporation in
  let remaining_alcohol := total_alcohol_before_evaporation - evaporated_alcohol in
  let alcohol_percentage := (remaining_alcohol / total_volume_after_addition) * 100 in
  alcohol_percentage = 10.78 :=
by {
  sorry
}

end alcohol_percentage_after_evaporation_l348_348451


namespace average_of_remaining_two_l348_348778

theorem average_of_remaining_two (S S3 : ℚ) (h1 : S / 5 = 6) (h2 : S3 / 3 = 4) : (S - S3) / 2 = 9 :=
by
  sorry

end average_of_remaining_two_l348_348778


namespace fastest_hike_is_faster_by_one_hour_l348_348680

def first_trail_time (d₁ s₁ : ℕ) : ℕ :=
  d₁ / s₁

def second_trail_time (d₂ s₂ break_time : ℕ) : ℕ :=
  (d₂ / s₂) + break_time

theorem fastest_hike_is_faster_by_one_hour 
(h₁ : first_trail_time 20 5 = 4)
(h₂ : second_trail_time 12 3 1 = 5) :
  5 - 4 = 1 :=
by
  exact Nat.sub_self 4.symm
  sorry

end fastest_hike_is_faster_by_one_hour_l348_348680


namespace bob_final_amount_l348_348508

-- Define initial amount
def initial_amount : ℝ := 80

-- Define money spent on Monday
def spent_monday : ℝ := initial_amount / 2

-- Define money left after Monday
def left_after_monday : ℝ := initial_amount - spent_monday

-- Define money spent on Tuesday
def spent_tuesday : ℝ := left_after_monday / 5

-- Define money left after Tuesday
def left_after_tuesday : ℝ := left_after_monday - spent_tuesday

-- Define money spent on Wednesday
def spent_wednesday : ℝ := (3 / 8) * left_after_tuesday

-- Define money left after Wednesday
def left_after_wednesday : ℝ := left_after_tuesday - spent_wednesday

-- The theorem stating the final amount of money left
theorem bob_final_amount : left_after_wednesday = 20 := by
  have h0 : initial_amount = 80 := rfl
  have h1 : spent_monday = 40 := by 
    simp [initial_amount, spent_monday]
  have h2 : left_after_monday = 40 := by 
    simp [left_after_monday, initial_amount, spent_monday]
    norm_num
  have h3 : spent_tuesday = 8 := by
    simp [spent_tuesday, left_after_monday]
    norm_num
  have h4 : left_after_tuesday = 32 := by
    simp [left_after_tuesday, left_after_monday, spent_tuesday]
    norm_num
  have h5 : spent_wednesday = 12 := by
    simp [spent_wednesday, left_after_tuesday]
    norm_num
  have h6 : left_after_wednesday = 20 := by
    simp [left_after_wednesday, left_after_tuesday, spent_wednesday]
    norm_num
  exact h6

end bob_final_amount_l348_348508


namespace find_charge_first_week_per_day_l348_348780

def charge_first_week_per_day (x : ℝ) : Prop :=
  let charge_additional_days_per_day := 11
  let total_days := 23
  let total_cost := 302
  let first_week_days := 7
  let additional_days := total_days - first_week_days
  let cost_first_week := first_week_days * x
  let cost_additional_days := additional_days * charge_additional_days_per_day
  let total_calculated_cost := cost_first_week + cost_additional_days
  total_calculated_cost = total_cost

theorem find_charge_first_week_per_day :
  ∃ (x : ℝ), charge_first_week_per_day x ∧ x = 18 :=
by
  exists 18
  split
  unfold charge_first_week_per_day
  -- Provide justification
  sorry

end find_charge_first_week_per_day_l348_348780


namespace bob_remaining_money_l348_348511

noncomputable def remaining_money_after_wednesday : ℕ :=
let monday := 80 - (1 / 2) * 80 in
let tuesday := monday - (1 / 5) * monday in
let wednesday := tuesday - (3 / 8) * tuesday in
wednesday

theorem bob_remaining_money : remaining_money_after_wednesday = 20 :=
by sorry

end bob_remaining_money_l348_348511


namespace find_teacher_age_l348_348441

noncomputable def teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_people : ℕ): ℕ :=
  avg_age_with_teacher * num_people - avg_age_students * num_students

theorem find_teacher_age:
  ∀ (total_students total_avg_age total_people teacher_avg_age : ℕ),
  total_avg_age = 21 → total_students = 22 → teacher_avg_age = 22 → total_people = 23 → teacher_age (total_avg_age) (total_students) (teacher_avg_age) (total_people) = 44 :=
by {
  intros,
  simp [teacher_age],
  sorry
}

end find_teacher_age_l348_348441


namespace prime_product_decomposition_l348_348014

theorem prime_product_decomposition :
  let n := 989 * 1001 * 1007 + 320
  in n = 991 * 997 * 1009 :=
by
  let n := 989 * 1001 * 1007 + 320
  have h : n = 991 * 997 * 1009 := sorry
  exact h

end prime_product_decomposition_l348_348014


namespace Clarence_oranges_l348_348545

theorem Clarence_oranges : 
  ∀ (initial_oranges : ℕ) (extra_oranges : ℕ), 
    initial_oranges = 5 → 
    extra_oranges = 3 → 
    initial_oranges + extra_oranges = 8 :=
by
  intros initial_oranges extra_oranges h_initial h_extra
  rw [h_initial, h_extra]
  exact rfl

end Clarence_oranges_l348_348545


namespace projection_of_b_on_a_l348_348998

variables (a b : ℝ^3) -- Assuming a and b are vectors in ℝ^3
variables (h1 : ∥a∥ = 2) (h2 : a • (b - a) = -3) -- Conditions

theorem projection_of_b_on_a :
  (a • b) / ∥a∥ = 1 / 2 :=
by sorry -- Proof skipped

end projection_of_b_on_a_l348_348998


namespace range_of_a_l348_348989

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → -3 * x^2 + a ≥ 0) → a ≥ 3 := 
by
  sorry

end range_of_a_l348_348989


namespace arrangment_count_l348_348246

def number_of_arrangements (r b : ℕ) :=
  if r = 5 ∧ b = 5 then
    126
  else
    0

theorem arrangment_count (r b : ℕ) (h1 : r = 5) (h2 : b = 5) :
  number_of_arrangements r b = 126 :=
begin
  unfold number_of_arrangements,
  rw [h1, h2],
  exact if_pos (and.intro h1 h2),
end

end arrangment_count_l348_348246


namespace compare_fractions_compare_integers_l348_348546

-- First comparison: Prove -4/7 > -2/3
theorem compare_fractions : - (4 : ℚ) / 7 > - (2 : ℚ) / 3 := 
by sorry

-- Second comparison: Prove -(-7) > -| -7 |
theorem compare_integers : -(-7) > -abs (-7) := 
by sorry

end compare_fractions_compare_integers_l348_348546


namespace tommy_paint_cost_l348_348041

def tommy_spends_on_paint : ℕ :=
  let width := 5 in
  let height := 4 in
  let sides := 2 in
  let cost_per_quart := 2 in
  let coverage_per_quart := 4 in
  let area_per_side := width * height in
  let total_area := sides * area_per_side in
  let quarts_needed := total_area / coverage_per_quart in
  let total_cost := quarts_needed * cost_per_quart in
  total_cost

theorem tommy_paint_cost : tommy_spends_on_paint = 20 := by
  sorry

end tommy_paint_cost_l348_348041


namespace trapezoid_side_lengths_l348_348863

theorem trapezoid_side_lengths
  (isosceles : ∀ (A B C D : ℝ) (height BE : ℝ), height = 2 → BE = 2 → A = 2 * Real.sqrt 2 → D = A → 12 = 0.5 * (B + C) * BE → A = D)
  (area : ∀ (BC AD : ℝ), 12 = 0.5 * (BC + AD) * 2)
  (height : ∀ (BE : ℝ), BE = 2)
  (intersect_right_angle : ∀ (A B C D : ℝ), 90 = 45 + 45) :
  ∃ A B C D, A = 2 * Real.sqrt 2 ∧ B = 4 ∧ C = 8 ∧ D = 2 * Real.sqrt 2 :=
by
  sorry

end trapezoid_side_lengths_l348_348863


namespace sum_of_first_16_terms_l348_348970

variable {α : Type*} [AddGroup α] [Module ℝ α]
variable {a : ℕ → α} -- the arithmetic sequence
variable {S : ℕ → α} -- the sum of first n terms

def arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a n = a 0 + n • d 

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) :=
  ∀ n : ℕ, S n = (n + 1) • a 0 + (n * (n + 1) • ds) / 2

theorem sum_of_first_16_terms (a_4 a_13 : α) (S : ℕ → α) :
  arithmetic_sequence a →
  sum_of_first_n_terms a S →
  a 4 + a 13 = 1 →
  S 16 = 8 :=
by
  sorry

end sum_of_first_16_terms_l348_348970


namespace ellipse_product_major_minor_l348_348756

theorem ellipse_product_major_minor 
  (O : Point) (A B C D F : Point)
  (hO : O = (center_of_ellipse (ellipse_with_major_minor_axis O A B C D)))
  (hF : F = (focus_of_ellipse O A B))
  (hOF : dist O F = 6)
  (h_inscribed_circle_diameter : 2 * inradius (triangle O C F) = 2)
  : (2 * dist O A) * (2 * dist O C) = 65 := 
sorry

end ellipse_product_major_minor_l348_348756


namespace tom_initial_amount_l348_348417

variables (t s j : ℝ)

theorem tom_initial_amount :
  t + s + j = 1200 →
  t - 200 + 3 * s + 2 * j = 1800 →
  t = 400 :=
by
  intros h1 h2
  sorry

end tom_initial_amount_l348_348417


namespace problem_l348_348332
-- Importing all necessary modules

-- Definitions based on the conditions given
variables {a b c : ℝ}
variables (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)

def P := a + b - c
def Q := b + c - a
def R := c + a - b

-- The proof problem statement
theorem problem (hPQR : P * Q * R > 0) : P > 0 ∧ Q > 0 ∧ R > 0 ↔ P * Q * R > 0 :=
begin
  sorry
end

end problem_l348_348332


namespace ball_center_distance_traveled_l348_348877

theorem ball_center_distance_traveled (d : ℝ) (r1 r2 r3 r4 : ℝ) (R1 R2 R3 R4 : ℝ) :
  d = 6 → 
  R1 = 120 → 
  R2 = 50 → 
  R3 = 90 → 
  R4 = 70 → 
  r1 = R1 - 3 → 
  r2 = R2 + 3 → 
  r3 = R3 - 3 → 
  r4 = R4 + 3 → 
  (1/2) * 2 * π * r1 + (1/2) * 2 * π * r2 + (1/2) * 2 * π * r3 + (1/2) * 2 * π * r4 = 330 * π :=
by
  sorry

end ball_center_distance_traveled_l348_348877


namespace problem_expr1_problem_expr2_l348_348538

noncomputable def eval_expr1 : ℝ := (8 : ℝ) ^ (2 / 3) + (0.01 : ℝ) ^ (-1 / 2) + (1 / 27 : ℝ) ^ (-1 / 3)
noncomputable def eval_expr2 : ℝ := 2 * Real.log 5 + (2 / 3) * Real.log 8 + Real.log 5 * Real.log 20 + (Real.log 2) ^ 2

theorem problem_expr1 : eval_expr1 = 17 := by
  sorry

theorem problem_expr2 : eval_expr2 = 3 := by
  sorry

end problem_expr1_problem_expr2_l348_348538


namespace unique_n_l348_348065

theorem unique_n : ∃ n : ℕ, 0 < n ∧ n^3 % 1000 = n ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 1000 = m → m = n :=
by
  sorry

end unique_n_l348_348065


namespace range_of_a_l348_348617

-- Define a predicate to determine if the quadratic equation has roots with the given constraint
def has_roots_between (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (x1 < 2) ∧ (2 < x2) 

-- Define the polynomial equation
noncomputable def quadratic_eqn (a : ℝ) :=
  polynomial.C (-9) + polynomial.C (2 * a) * polynomial.X + polynomial.C 1 * polynomial.X^2

-- Lean statement representing the proof problem
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h : (quadratic_eqn a).roots = {x1, x2}) :
  has_roots_between a x1 x2 → a < 5 / 4 :=
by sorry

end range_of_a_l348_348617


namespace max_sum_abs_diff_l348_348701

theorem max_sum_abs_diff (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (∑ i j in Finset.univ.filter (λ p, p.1 < p.2), (|x i - x j|)) ≤ ⌊n^2 / 4⌋ := 
sorry

end max_sum_abs_diff_l348_348701


namespace simplify_fraction_multiplication_l348_348761

theorem simplify_fraction_multiplication :
  (15/35) * (28/45) * (75/28) = 5/7 :=
by
  sorry

end simplify_fraction_multiplication_l348_348761


namespace problem_statement_l348_348156

-- Definitions of the sets
def S1 := {77, 78}
def inverse (n : Nat) : Nat :=
  match n with
  | 7 => 8
  | 8 => 7
  | _ => 0 -- In case any unexpected digit, default to 0.

def inverse_set (s : Set Nat) : Set Nat :=
  s.map inverse

noncomputable def S (n : Nat) : Set Nat :=
  if n = 1 then S1 else sorry -- Skip the inductive step definition for simplicity.

noncomputable def T (n : Nat) : Set Nat :=
  S n ∪ inverse_set (S n)

-- The main statement
theorem problem_statement : 
  ∃ T2005 : Set Nat,
  (T2005 = T 2005 ) ∧
  (T2005.card = 2 ^ 2006) ∧
  (∀ (a ∈ T2005), Nat.digits 10 a).length = 2 ^ 2005) ∧
  (∀ (a ∈ T2005), ∀ d, d ∈ Nat.digits 10 a → d = 7 ∨ d = 8) ∧
  (∀ (a b ∈ T2005), a ≠ b → (Nat.same_half_digits a b)) := sorry

end problem_statement_l348_348156


namespace john_has_200_cards_l348_348687

theorem john_has_200_cards :
  let deck_size := 52
  let half_full_decks := 3
  let full_decks := 3
  let discarded_cards := 34
  let cards_per_half_full_deck := deck_size / 2
  let total_cards_half_full := half_full_decks * cards_per_half_full_deck
  let total_cards_full := full_decks * deck_size
  let total_cards_before_discard := total_cards_half_full + total_cards_full
  let total_cards_after_discard := total_cards_before_discard - discarded_cards
  total_cards_after_discard = 200 := 
by 
  let deck_size := 52
  let half_full_decks := 3
  let full_decks := 3
  let discarded_cards := 34
  let cards_per_half_full_deck := deck_size / 2
  let total_cards_half_full := half_full_decks * cards_per_half_full_deck
  let total_cards_full := full_decks * deck_size
  let total_cards_before_discard := total_cards_half_full + total_cards_full
  let total_cards_after_discard := total_cards_before_discard - discarded_cards
  show total_cards_after_discard = 200 from sorry

end john_has_200_cards_l348_348687


namespace solution_set_unique_line_l348_348747

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l348_348747


namespace curry_pepper_count_l348_348885

theorem curry_pepper_count :
  ∀ (very_spicy_curried_peppers spicy_curried_peppers mild_curried_peppers 
      prev_very_spicy_curry prev_spicy_curry prev_mild_curry new_spicy_curry),
  (very_spicy_curried_peppers = 3) → 
  (spicy_curried_peppers = 2) → 
  (mild_curried_peppers = 1) →
  (prev_very_spicy_curry = 30) →
  (prev_spicy_curry = 30) →
  (prev_mild_curry = 10) →
  (new_spicy_curry = 15) →
  let prev_total_peppers := very_spicy_curried_peppers * prev_very_spicy_curry + 
                            spicy_curried_peppers * prev_spicy_curry + 
                            mild_curried_peppers * prev_mild_curry in
  let new_total_peppers := prev_total_peppers - 40 in
  let mild := new_total_peppers - spicy_curried_peppers * new_spicy_curry in
  mild = 90 :=
begin
  intros,
  sorry
end

end curry_pepper_count_l348_348885


namespace system_solution_unique_l348_348161

theorem system_solution_unique (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x ^ 3 + 2 * y ^ 2 + 1 / (4 * z) = 1)
  (eq2 : y ^ 3 + 2 * z ^ 2 + 1 / (4 * x) = 1)
  (eq3 : z ^ 3 + 2 * x ^ 2 + 1 / (4 * y) = 1) :
  (x, y, z) = ( ( (-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2) ) := 
by
  sorry

end system_solution_unique_l348_348161


namespace fraction_sequence_product_l348_348064

noncomputable def fraction_sequence : List (ℚ) := [
  1/4, 8/1, 1/32, 64/1,
  1/256, 512/1, 1/2048, 4096/1
]

theorem fraction_sequence_product : (fraction_sequence.product) = 32 :=
by
  sorry

end fraction_sequence_product_l348_348064


namespace solution_set_l348_348743

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l348_348743


namespace solution_set_l348_348738

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l348_348738


namespace domain_of_g_l348_348921

theorem domain_of_g :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 49) ↔ (0 ≤ 3 - real.sqrt (7 - real.sqrt x)) :=
begin
  intro x, 
  split,
  { 
    intro hx,
    have h1 : 0 ≤ real.sqrt x := real.sqrt_nonneg x,
    have h2 : x ≤ 49 := hx.2,
    have h3 : 0 ≤ real.sqrt (7 - real.sqrt x) ∧ real.sqrt (7 - real.sqrt x) ≤ 3,
    { 
      split,
      { 
        apply real.sqrt_nonneg,
      },
      {
        apply le_of_sub_nonneg,
        rw sub_self_add_eq_add,
        apply sub_nonneg_of_le,
        rw sub_add_eq_sub_sub,
        rw sub_le_iff,
        rw real.sqrt_mul_self_eq_abs,
        apply abs_le,
        exact hx.2,
        exact (by norm_num : 0 ≤ 3)
      }
    },
    exact h3.2,
  },
  {
    intro h4,
    split, 
    {
      linarith,
    },
    { 
      norm_num,
    }
  }
end

end domain_of_g_l348_348921


namespace ribbon_proof_l348_348817

theorem ribbon_proof :
  ∃ (ribbons : List ℝ), 
    ribbons = [9, 9, 24, 24, 24] ∧
    (∃ (initial_ribbons : List ℝ) (cuts : List ℝ),
      initial_ribbons = [15, 20, 24, 26, 30] ∧
      ((List.sum initial_ribbons / initial_ribbons.length) - (List.sum ribbons / ribbons.length) = 5) ∧
      (List.median ribbons = List.median initial_ribbons) ∧
      (List.range ribbons = List.range initial_ribbons)) :=
sorry

end ribbon_proof_l348_348817


namespace sqrt_expression_simplification_l348_348365

theorem sqrt_expression_simplification:
  (\(sqrt 8 - sqrt (4 + 1/2))^2 = 1/2 :=
by
  sorry

end sqrt_expression_simplification_l348_348365


namespace BQP_eq_BOP_l348_348698

-- Definition of the points and conditions
variables (a b : ℝ) (h : 0 < b ∧ b < a)

def O := (0, 0) : ℝ × ℝ
def A := (0, a) : ℝ × ℝ
def B := (0, b) : ℝ × ℝ
def Γ : set (ℝ × ℝ) := { P | (P.1 - O.1)^2 + (P.2 - (a + b) / 2)^2 = ((a - b) / 2)^2 }

-- Angle between points function (a utility function to calculate angles or could represent the same result)
def angle (P Q R : ℝ × ℝ) : ℝ := sorry -- Placeholder, as implementing angle calculation is non-trivial

-- Given any point P on the circle Γ, line PA intersects the x-axis at Q.
variables (P : ℝ × ℝ) (hP : P ∈ Γ) (Q : ℝ × ℝ) (hQ : Q.2 = 0 ∧ ∃ k : ℝ, P = (k, a.k))

-- The conjecture to prove
theorem BQP_eq_BOP : angle B Q P = angle B O P := sorry

end BQP_eq_BOP_l348_348698


namespace binary_to_decimal_l348_348918

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 := by
  sorry

end binary_to_decimal_l348_348918


namespace sum_of_y_neg_l348_348400

-- Define the conditions from the problem
def condition1 (x y : ℝ) : Prop := x + y = 7
def condition2 (x z : ℝ) : Prop := x * z = -180
def condition3 (x y z : ℝ) : Prop := (x + y + z)^2 = 4

-- Define the main theorem to prove
theorem sum_of_y_neg (x y z : ℝ) (S : ℝ) :
  (condition1 x y) ∧ (condition2 x z) ∧ (condition3 x y z) →
  (S = (-29) + (-13)) →
  -S = 42 :=
by
  sorry

end sum_of_y_neg_l348_348400


namespace finite_set_solution_l348_348965

def finite_set_problem (n : ℕ) (a : Fin n → ℤ) : Prop :=
  (n ≥ 3) → 
  (a 0 = 0) →
  (a (n-1) = 0) →
  (∀ k : Fin n, 2 ≤ k.val → k.val < n - 1 → a (k - 1) + a (k + 1) ≥ 2 * a k) →
  (∀ i : Fin n, a i ≤ 0)

theorem finite_set_solution (n : ℕ) (a : Fin n → ℤ) : finite_set_problem n a :=
by
  intros h_n h_a1 h_an h_condition
  sorry

end finite_set_solution_l348_348965


namespace ceo_dividends_correct_l348_348528

-- Definitions of parameters
def revenue := 2500000
def expenses := 1576250
def tax_rate := 0.2
def monthly_loan_payment := 25000
def months := 12
def number_of_shares := 1600
def ceo_ownership := 0.35

-- Calculation functions based on conditions
def net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
def loan_payments := monthly_loan_payment * months
def dividends_per_share := (net_profit - loan_payments) / number_of_shares
def ceo_dividends := dividends_per_share * ceo_ownership * number_of_shares

-- Statement to prove
theorem ceo_dividends_correct : ceo_dividends = 153440 :=
by 
  -- skipping the proof
  sorry

end ceo_dividends_correct_l348_348528


namespace percent_increase_area_squares_l348_348896

theorem percent_increase_area_squares :
  let s₁ := 3
  let s₂ := 1.25 * s₁
  let s₃ := 1.25 * s₂
  let s₄ := 1.25 * s₃
  let A₁ := s₁^2
  let A₄ := s₄^2
  let percent_increase := ((A₄ - A₁) / A₁) * 100
  percent_increase ≈ 281.4 := by
  sorry

end percent_increase_area_squares_l348_348896


namespace part_a_part_b_l348_348301

-- Definitions for Part (a)
def A (n : ℕ) : ℝ × ℝ := (n, n^3)

-- Statement for Part (a)
theorem part_a (k j i : ℕ) (hk : k > j) (hj : j > i) (hi : i ≥ 1) :
  ¬ collinear {A i, A j, A k} :=
sorry

-- Definitions for Part (b)
def B : ℝ × ℝ := (0, 1)
def angle (A B : ℝ × ℝ) : ℝ -- This would define the angle ∠AOB
  := sorry

-- Statement for Part (b)
theorem part_b (i : ℕ → ℕ) (h : strict_mono i) (hn : ∀ j, 1 ≤ i j) :
  (∑ j in Finset.range k, angle (A (i j)) B) < π / 2 :=
sorry

end part_a_part_b_l348_348301


namespace first_three_workers_time_l348_348183

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l348_348183


namespace prime_saturated_two_digit_max_is_98_l348_348465

def is_prime_saturated (z : ℕ) : Prop :=
  ∃ p, (z > 1) ∧ (Nat.factors z = p) ∧ (List.prod p < Real.sqrt z)

def greatest_prime_saturated_two_digit : ℕ :=
  98

theorem prime_saturated_two_digit_max_is_98 :
  greatest_prime_saturated_two_digit = 98 ∧ is_prime_saturated greatest_prime_saturated_two_digit :=
by
  -- We need to prove the greatest two-digit prime saturated integer is 98
  sorry

end prime_saturated_two_digit_max_is_98_l348_348465


namespace triangle_area_is_4_point_5_l348_348391

-- Define the sides of the triangle
def a : ℝ := 3
def b : ℝ := 5

-- Define the cosine value as being a root of the equation 5x^2 - 7x - 6 = 0
def cos_theta : ℝ := -3/5

-- Define the area calculation using the absolute value of the cosine
def triangle_area : ℝ := 1/2 * a * b * abs(cos_theta)

-- The theorem we want to prove
theorem triangle_area_is_4_point_5 :
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 ∧ triangle_area = 4.5 :=
by
  sorry

end triangle_area_is_4_point_5_l348_348391


namespace not_divisible_67_l348_348726

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l348_348726


namespace log_tan_sum_zero_l348_348929

theorem log_tan_sum_zero : 
  ∑ k in Finset.range 44, Real.logb 5 (Real.tan (2 * (k + 1) * Real.pi / 180)) = 0 :=
sorry

end log_tan_sum_zero_l348_348929


namespace area_ABC_calculation_l348_348356

-- Geometric setup for the problem
def Point := ℝ × ℝ

-- Define points B, C, D, and E based on given conditions
def B : Point := (4, 0)
def C : Point := (-3, 0)
def D : Point := (0, 0)
def E : Point := (3, 0)

-- Midpoint function for calculating midpoints of circles
def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Circle equation function
def circle_eq (center : Point) (radius : ℝ) (P : Point) : Prop :=
  (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

-- Given the angles BAD and EAC are 90 degrees, A lies on the circles
def angle_BAD_90 (A : Point) : Prop :=
  circle_eq (midpoint B D) 2 A

def angle_EAC_90 (A : Point) : Prop :=
  circle_eq (midpoint E C) 3 A

-- Assumptions
axiom angle_BAD : ∀ A, angle_BAD_90 A
axiom angle_EAC : ∀ A, angle_EAC_90 A

-- Calculate the area of triangle ABC
def triangle_area (A B C : Point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Point A lies on intersection of two circles, considering positive y for simplicity
def A : Point := (9 / 4, real.sqrt(63) / 4)

-- Final statement for the proof
theorem area_ABC_calculation : 
  triangle_area A B C = 21 * real.sqrt 7 / 8 → 36 :=
by
  sorry

end area_ABC_calculation_l348_348356


namespace not_arithmetic_seq_sum_first_n_terms_l348_348630

def seq (n : ℕ) (h : n > 0) : ℤ :=
  if h' : n ≤ 7 then n + 1
  else n - 1

def sum (n : ℕ) (h : n > 0) : ℤ :=
  if h' : n ≤ 7 then (n * n + 3 * n) / 2
  else (n * n - n) / 2 + 14

theorem not_arithmetic_seq (n m : ℕ) (h₁ : n > 0) (h₂ : m > 0) (h₃ : n ≠ m) :
  seq n h₁ - seq (n-1) (nat.sub_pos_of_lt (nat.lt_of_le_of_ne h₁ (ne.symm h₃))) 
  ≠ seq m h₂ - seq (m-1) (nat.sub_pos_of_lt (nat.lt_of_le_of_ne h₂ h₃)) :=
sorry

theorem sum_first_n_terms (n : ℕ) (h : n > 0) : ∑ i in range n, seq (i+1) (nat.succ_pos i) = sum n h :=
sorry

end not_arithmetic_seq_sum_first_n_terms_l348_348630


namespace probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l348_348423

noncomputable def total_cassettes : ℕ := 30
noncomputable def disco_cassettes : ℕ := 12
noncomputable def classical_cassettes : ℕ := 18

-- Part (a): DJ returns the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_returned :
  (disco_cassettes / total_cassettes) * (disco_cassettes / total_cassettes) = 4 / 25 :=
by
  sorry

-- Part (b): DJ does not return the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_not_returned :
  (disco_cassettes / total_cassettes) * ((disco_cassettes - 1) / (total_cassettes - 1)) = 22 / 145 :=
by
  sorry

end probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l348_348423


namespace true_propositions_count_l348_348486

theorem true_propositions_count :
  let prop1 := ¬(sqrt (sqrt 16) = 2)
  let prop2 := ¬∀ (a b c d : ℝ) (α : ℝ), let tri₁ := (a, b, α) in let tri₂ := (c, d, α) in (a = c) → (b = d) → tri₁ = tri₂
  let prop3 := ∀ (q : Quadrilateral), is_parallelogram (midpoint_quadrilateral q)
  let prop4 := ∀ (A B C : Angle), let angle_ratio := (1, 2, 3) in (angle_sum (A, B, C) = 180) → is_right_triangle (triangle A B C)
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) ↔ true :=
by
  sorry

end true_propositions_count_l348_348486


namespace valid_q_cardinality_l348_348958

noncomputable def num_valid_q : ℕ :=
  {q : ℝ | abs (abs (abs (q - 5) - 10) - 5) = 2}.to_finset.card

theorem valid_q_cardinality : num_valid_q = 4 :=
  sorry

end valid_q_cardinality_l348_348958


namespace arithmetic_sequence_sum_4_l348_348969

-- Definitions based on conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

def a : ℕ → ℕ
| 1 := 4
| 2 := 6
| (n + 1) := a n + (a 2 - a 1)

def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1)

-- The actual proof problem
theorem arithmetic_sequence_sum_4 :
  S 4 = 28 :=
by
  sorry

end arithmetic_sequence_sum_4_l348_348969


namespace math_problem_l348_348267

noncomputable def proof_problem (n : ℕ) (a : Fin n → ℝ) (d s : ℝ) : Prop :=
  (∀ i j : Fin n, i ≠ j → ∑ (i < j), abs (a i - a j) = s) ∧ 
  (d = (finset.univ.sup' (nonempty_fin (n)) a id) - (finset.univ.inf' (nonempty_fin (n)) a id)) →
  (n - 1) * d ≤ s ∧ s ≤ (n * n / 4) * d

theorem math_problem (n : ℕ) (a : Fin n → ℝ) (d s : ℝ) :
  proof_problem n a d s :=
begin
  sorry
end

end math_problem_l348_348267


namespace polar_to_cartesian_l348_348008

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = -4 * cos θ) :
    (x y : ℝ) (hx : x = ρ * cos θ) (hy : y = ρ * sin θ) (hρ2 : ρ^2 = x^2 + y^2) :
    (x + 2)^2 + y^2 = 4 :=
sorry

end polar_to_cartesian_l348_348008


namespace geom_arith_sequence_l348_348605

theorem geom_arith_sequence (a b c m n : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : m = (a + b) / 2) 
  (h3 : n = (b + c) / 2) : 
  a / m + c / n = 2 := 
by 
  sorry

end geom_arith_sequence_l348_348605


namespace loss_percentage_first_book_l348_348638

theorem loss_percentage_first_book (C1 C2 : ℝ) 
    (total_cost : ℝ) 
    (gain_percentage : ℝ)
    (S1 S2 : ℝ)
    (cost_first_book : C1 = 175)
    (total_cost_condition : total_cost = 300)
    (gain_condition : gain_percentage = 0.19)
    (same_selling_price : S1 = S2)
    (second_book_cost : C2 = total_cost - C1)
    (selling_price_second_book : S2 = C2 * (1 + gain_percentage)) :
    (C1 - S1) / C1 * 100 = 15 :=
by
  sorry

end loss_percentage_first_book_l348_348638


namespace g_inv_90_eq_3_l348_348641

variable {R : Type} [Real]

def g (x : ℝ) : ℝ := 3 * x^3 + 9

theorem g_inv_90_eq_3 : g⁻¹' 90 = 3 := sorry

end g_inv_90_eq_3_l348_348641


namespace not_consecutive_terms_arithmetic_sequence_l348_348758

theorem not_consecutive_terms_arithmetic_sequence:
  ∀ (a d : ℝ) (m n k : ℤ),
    1 = a + (m-1 : ℤ) * d →
    sqrt 2 = a + (n-1 : ℤ) * d →
    3 = a + (k-1 : ℤ) * d →
    False :=
by
  sorry

end not_consecutive_terms_arithmetic_sequence_l348_348758


namespace largest_number_le_1_1_from_set_l348_348279

def is_largest_le (n : ℚ) (l : List ℚ) (bound : ℚ) : Prop :=
  (n ∈ l ∧ n ≤ bound) ∧ ∀ m ∈ l, m ≤ bound → m ≤ n

theorem largest_number_le_1_1_from_set : 
  is_largest_le (9/10) [14/10, 9/10, 12/10, 5/10, 13/10] (11/10) :=
by 
  sorry

end largest_number_le_1_1_from_set_l348_348279


namespace product_uvw_l348_348383

theorem product_uvw (a x y c : ℝ) (u v w : ℤ) :
  (a^u * x - a^v) * (a^w * y - a^3) = a^5 * c^5 → 
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1) → 
  u * v * w = 6 :=
by
  intros h1 h2
  -- Proof will go here
  sorry

end product_uvw_l348_348383


namespace number_of_subsets_of_S_is_8_l348_348790

def S := {x : ℤ | -1 <= x ∧ x <= 1}

theorem number_of_subsets_of_S_is_8 : (2 ^ S.to_finset.card) = 8 :=
by sorry

end number_of_subsets_of_S_is_8_l348_348790


namespace bob_remaining_money_l348_348509

noncomputable def remaining_money_after_wednesday : ℕ :=
let monday := 80 - (1 / 2) * 80 in
let tuesday := monday - (1 / 5) * monday in
let wednesday := tuesday - (3 / 8) * tuesday in
wednesday

theorem bob_remaining_money : remaining_money_after_wednesday = 20 :=
by sorry

end bob_remaining_money_l348_348509


namespace bridge_length_proof_l348_348834

-- Definitions
def train_length : ℝ := 148
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Unit conversion from km/h to m/s
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Total distance travelled by train in the given time
def total_distance : ℝ := train_speed_ms * crossing_time_seconds

-- Length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem bridge_length_proof : bridge_length = 227 := by
  -- We are only providing the statement of the proof, not the proof itself.
  sorry

end bridge_length_proof_l348_348834


namespace find_n_for_eq_l348_348168

theorem find_n_for_eq :
  (∃ n : ℕ, (∑ k in Finset.range n, 1 / (Real.sqrt k + Real.sqrt (k + 1))) = 2010) → n = 4044120 :=
by
  sorry

end find_n_for_eq_l348_348168


namespace incenter_bisects_BB1_l348_348390

noncomputable theory
open_locale euclidean_geometry

variables {a b c d : ℝ} (A B C O B1 : Point)

-- Condition 1: The sides of the triangle form an arithmetic progression
variables (h_arith_prog : a + d = b ∧ b + d = c)

-- Condition 2: The bisector of angle B intersects the circumcircle at point B1
variables (h_bisector : angle_bisector B A C B1)
variables [incircle_configuration A B C O] (h3 : O = incircle_center A B C) (h4 : concyclic_points [A, B, C, B1])

-- Conclusion: O bisects BB1.
theorem incenter_bisects_BB1 (h_arith_prog : a + d = b ∧ b + d = c)
(h_bisector : angle_bisector B A C B1)
(h3 : O = incircle_center A B C)
(h4 : concyclic_points [A, B, C, B1]):
line_segment O B = line_segment O B1 :=
begin
  sorry
end

end incenter_bisects_BB1_l348_348390


namespace bottle_cap_cost_l348_348151

-- Define the conditions given in the problem.
def caps_cost (n : ℕ) (cost : ℝ) : Prop := n * cost = 12

-- Prove that the cost of each bottle cap is $2 given 6 bottle caps cost $12.
theorem bottle_cap_cost (h : caps_cost 6 cost) : cost = 2 :=
sorry

end bottle_cap_cost_l348_348151


namespace quadratic_solution_linear_factor_solution_l348_348767

theorem quadratic_solution (x : ℝ) : (5 * x^2 + 2 * x - 1 = 0) ↔ (x = (-1 + Real.sqrt 6) / 5 ∨ x = (-1 - Real.sqrt 6) / 5) := by
  sorry

theorem linear_factor_solution (x : ℝ) : (x * (x - 3) - 4 * (3 - x) = 0) ↔ (x = 3 ∨ x = -4) := by
  sorry

end quadratic_solution_linear_factor_solution_l348_348767


namespace sales_in_second_month_l348_348886

-- Given conditions:
def sales_first_month : ℕ := 6400
def sales_third_month : ℕ := 6800
def sales_fourth_month : ℕ := 7200
def sales_fifth_month : ℕ := 6500
def sales_sixth_month : ℕ := 5100
def average_sales : ℕ := 6500

-- Statement to prove:
theorem sales_in_second_month :
  ∃ (sales_second_month : ℕ), 
    average_sales * 6 = sales_first_month + sales_second_month + sales_third_month 
    + sales_fourth_month + sales_fifth_month + sales_sixth_month 
    ∧ sales_second_month = 7000 :=
  sorry

end sales_in_second_month_l348_348886


namespace sum_of_squares_l348_348362

theorem sum_of_squares (a d : Int) : 
  ∃ y1 y2 : Int, a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (3*a + y1*d)^2 + (a + y2*d)^2 :=
by
  sorry

end sum_of_squares_l348_348362


namespace calculation_eq_l348_348865

theorem calculation_eq : -2^2 + Real.sqrt 4 + (-1:ℤ)^2023 - Real.cbrt (-8:ℤ) + |1 - Real.sqrt 3| = Real.sqrt 3 - 2 :=
by
  -- proof goes here
  sorry

end calculation_eq_l348_348865


namespace largest_digit_sum_in_24_hour_format_l348_348090

theorem largest_digit_sum_in_24_hour_format : ∃ (h m : ℕ), (0 ≤ h < 24) ∧ (0 ≤ m < 60) ∧ (digit_sum h + digit_sum m) = 24 :=
sorry

-- Here, digit_sum is a helper function you should define that calculates the sum of the digits of a number.
def digit_sum (n : ℕ) : ℕ :=
(n / 10) + (n % 10)

end largest_digit_sum_in_24_hour_format_l348_348090


namespace sum_of_two_integers_l348_348006
noncomputable theory

variables {x y : ℤ}

theorem sum_of_two_integers (hx : x > 0) (hy : y > 0) (hxy1 : x - y = 16) (hxy2 : x * y = 162) : x + y = 30 :=
sorry

end sum_of_two_integers_l348_348006


namespace power_mod_l348_348430

theorem power_mod :
  ∀ (a n p : ℕ), p.prime → (a : ℤ) ^ p ∣ a → (a : ℤ) ^ n ≡ 0 [MOD 17] :=
by
  sorry

end power_mod_l348_348430


namespace julia_short_money_l348_348413

theorem julia_short_money 
  (price_rock : ℕ := 7)
  (price_pop : ℕ := 12)
  (price_dance : ℕ := 5)
  (price_country : ℕ := 9)
  (discount : ℕ := 15)
  (threshold : ℕ := 3)
  (wanted_rock : ℕ := 5)
  (wanted_pop : ℕ := 3)
  (wanted_dance : ℕ := 6)
  (wanted_country : ℕ := 4)
  (available_rock : ℕ := 4)
  (available_dance : ℕ := 5)
  (budget : ℕ := 80)
  : 
  let total_cost := (min wanted_rock available_rock * price_rock + wanted_pop * price_pop + 
                     min wanted_dance available_dance * price_dance + wanted_country * price_country)
      discount_rock := if min wanted_rock available_rock >= threshold then (discount * min wanted_rock available_rock * price_rock / 100) else 0
      discount_pop := if wanted_pop >= threshold then (discount * wanted_pop * price_pop / 100) else 0
      discount_dance := if min wanted_dance available_dance >= threshold then (discount * min wanted_dance available_dance * price_dance / 100) else 0
      discount_country := if wanted_country >= threshold then (discount * wanted_country * price_country / 100) else 0
      total_discount := discount_rock + discount_pop + discount_dance + discount_country
      final_cost := total_cost - total_discount
  in final_cost - budget = 26.25 := 
by 
  sorry

end julia_short_money_l348_348413


namespace system_solution_l348_348769

theorem system_solution {x y : ℝ} 
  (h1 : 4 ^ (abs (x^2 - 8 * x + 12) - Real.log 7 / Real.log 4) = 7 ^ (2 * y - 1))
  (h2 : abs (y - 3) - 3 * abs y - 2 * (y + 1)^2 ≥ 1) : 
  (x = 2 ∧ y = 0) ∨ (x = 6 ∧ y = 0) :=
by
  sorry

end system_solution_l348_348769


namespace incorrect_zero_vector_has_no_direction_l348_348434

-- Conditions
variable {V : Type} [AddCommGroup V] [Module ℝ V]
def zero_vector (v : V) := v = 0
def is_zero_vector (v : V) : Prop := v = 0
def magnitude_zero (v : V) : Prop := ∥v∥ = 0
def collinear (u v : V) : Prop := ∃ (a : ℝ), u = a • v

-- Problem Statement
theorem incorrect_zero_vector_has_no_direction (v : V) 
  (hv_zero : is_zero_vector v)
  (hv_mag_zero : magnitude_zero v)
  (hv_collinear : ∀ u, collinear v u) :
  ¬ (∀ v, zero_vector v → ∃ d, d ≠ d) :=
sorry

end incorrect_zero_vector_has_no_direction_l348_348434


namespace max_value_of_A_l348_348566

theorem max_value_of_A (α : ℝ) (h₁ : 0 ≤ α) (h₂ : α ≤ π / 2) :
  (∃ α, 0 ≤ α ∧ α ≤ π / 2 ∧ 2 = max {A | A = 1 / (sin α ^ 4 + cos α ^ 4)}) :=
sorry

end max_value_of_A_l348_348566


namespace ratio_of_areas_l348_348497

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l348_348497


namespace fraction_of_15_smaller_by_20_l348_348057

/-- Define 80% of 40 -/
def eighty_percent_of_40 : ℝ := 0.80 * 40

/-- Define the fraction of 15 that we are looking for -/
def fraction_of_15 (x : ℝ) : ℝ := x * 15

/-- Define the problem statement -/
theorem fraction_of_15_smaller_by_20 : ∃ x : ℝ, fraction_of_15 x = eighty_percent_of_40 - 20 ∧ x = 4 / 5 :=
by
  sorry

end fraction_of_15_smaller_by_20_l348_348057


namespace original_price_of_article_l348_348489

theorem original_price_of_article (selling_price : ℝ) (loss_percent : ℝ) (P : ℝ) 
  (h1 : selling_price = 450)
  (h2 : loss_percent = 25)
  : selling_price = (1 - loss_percent / 100) * P → P = 600 :=
by
  sorry

end original_price_of_article_l348_348489


namespace square_711th_position_l348_348475

def square_transformation (n : Nat) : List Char :=
  [ ['A', 'B', 'C', 'D'], -- Starting position
    ['A', 'D', 'C', 'B'], -- After 1st transformation (ADCB)
    ['B', 'A', 'D', 'C'], -- After 2nd transformation (BADC)
    ['C', 'D', 'A', 'B'], -- After 3rd transformation (CDAB)
    ['D', 'B', 'C', 'A']  -- After 4th transformation (DBCA)
  ].cycle.take (n + 1)

theorem square_711th_position : square_transformation 710 = ['A', 'D', 'C', 'B'] :=
by
  -- Giving this theorem statement to describe the solution.
  -- Transforming 710 times (since we start at 0) gives us the 711th position.
  sorry

end square_711th_position_l348_348475


namespace angle_B_is_180_l348_348716

variables {l k : Line} {A B C: Point}

def parallel (l k : Line) : Prop := sorry 
def angle (A B C : Point) : ℝ := sorry

theorem angle_B_is_180 (h1 : parallel l k) (h2 : angle A = 110) (h3 : angle C = 70) :
  angle B = 180 := 
by
  sorry

end angle_B_is_180_l348_348716


namespace work_done_by_forces_l348_348610

-- Definitions of given forces and displacement
noncomputable def F1 : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F2 : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Statement of the theorem
theorem work_done_by_forces :
  let F := (F1.1 + F2.1, F1.2 + F2.2)
  let W := F.1 * S.1 + F.2 * S.2
  W = 2 :=
by
  sorry

end work_done_by_forces_l348_348610


namespace seed_total_after_trading_l348_348849

theorem seed_total_after_trading :
  ∀ (Bom Gwi Yeon Eun : ℕ),
  Yeon = 3 * Gwi →
  Gwi = Bom + 40 →
  Eun = 2 * Gwi →
  Bom = 300 →
  Yeon_gives = 20 * Yeon / 100 →
  Bom_gives = 50 →
  let Yeon_after := Yeon - Yeon_gives
  let Gwi_after := Gwi + Yeon_gives
  let Bom_after := Bom - Bom_gives
  let Eun_after := Eun + Bom_gives
  Bom_after + Gwi_after + Yeon_after + Eun_after = 2340 :=
by
  intros Bom Gwi Yeon Eun hYeon hGwi hEun hBom hYeonGives hBomGives Yeon_after Gwi_after Bom_after Eun_after
  sorry

end seed_total_after_trading_l348_348849


namespace rectangle_width_l348_348033

theorem rectangle_width (w : ℝ)
    (h₁ : 5 > 0) (h₂ : 6 > 0) (h₃ : 3 > 0) 
    (area_relation : w * 5 = 3 * 6 + 2) : w = 4 :=
by
  sorry

end rectangle_width_l348_348033


namespace girls_with_no_pets_l348_348289

-- Define the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def fraction_girls : ℚ := 1 - fraction_boys
def girls_with_dogs_fraction : ℚ := 40 / 100
def girls_with_cats_fraction : ℚ := 20 / 100
def girls_with_no_pets_fraction : ℚ := 1 - (girls_with_dogs_fraction + girls_with_cats_fraction)

-- Calculate the number of girls
def total_girls : ℕ := total_students * fraction_girls.to_nat
def number_girls_with_no_pets : ℕ := total_girls * girls_with_no_pets_fraction.to_nat

-- Theorem statement
theorem girls_with_no_pets : number_girls_with_no_pets = 8 :=
by sorry

end girls_with_no_pets_l348_348289


namespace hundredths_digit_of_power_l348_348833

theorem hundredths_digit_of_power (n : ℕ) (h : n % 20 = 14) : 
  (8 ^ n % 1000) / 100 = 1 :=
by sorry

lemma test_power_hundredths_digit : (8 ^ 1234 % 1000) / 100 = 1 :=
hundredths_digit_of_power 1234 (by norm_num)

end hundredths_digit_of_power_l348_348833


namespace dexter_total_cards_l348_348555

theorem dexter_total_cards 
  (boxes_basketball : ℕ) 
  (cards_per_basketball_box : ℕ) 
  (boxes_football : ℕ) 
  (cards_per_football_box : ℕ) 
   (h1 : boxes_basketball = 15)
   (h2 : cards_per_basketball_box = 20)
   (h3 : boxes_football = boxes_basketball - 7)
   (h4 : cards_per_football_box = 25) 
   : boxes_basketball * cards_per_basketball_box + boxes_football * cards_per_football_box = 500 := by 
sorry

end dexter_total_cards_l348_348555


namespace equal_distribution_l348_348122

namespace MoneyDistribution

def Ann_initial := 777
def Bill_initial := 1111
def Charlie_initial := 1555
def target_amount := 1148
def Bill_to_Ann := 371
def Charlie_to_Bill := 408

theorem equal_distribution :
  (Bill_initial - Bill_to_Ann + Charlie_to_Bill = target_amount) ∧
  (Ann_initial + Bill_to_Ann = target_amount) ∧
  (Charlie_initial - Charlie_to_Bill = target_amount) :=
by
  sorry

end MoneyDistribution

end equal_distribution_l348_348122


namespace fraction_mangoes_taken_l348_348875

-- Define the initial counts of fruits
def initial_apples : ℕ := 7
def initial_oranges : ℕ := 8
def initial_mangoes : ℕ := 15

-- Define the actions Luisa takes
def apples_taken : ℕ := 2
def oranges_taken : ℕ := 2 * apples_taken
def total_remaining_fruits : ℕ := 14

-- Define the remaining counts of fruits
def remaining_apples : ℕ := initial_apples - apples_taken
def remaining_oranges : ℕ := initial_oranges - oranges_taken

-- Prove that Luisa took out 2/3 of the mangoes
theorem fraction_mangoes_taken : 
  let remaining_mangoes := total_remaining_fruits - (remaining_apples + remaining_oranges) in
  let mangoes_taken := initial_mangoes - remaining_mangoes in
  (mangoes_taken : ℚ) / initial_mangoes = 2 / 3 := by
  sorry

end fraction_mangoes_taken_l348_348875


namespace ryan_caught_30_fish_l348_348685

-- Definitions for the conditions
def Jason : Type := Nat
def Ryan : Type := Nat
def Jeffery : Type := Nat

-- Given conditions
axiom fishRyan (J : Jason) : Ryan := 3 * J
axiom fishJeffery : Jeffery := 60
axiom totalFish (J : Jason) (R : Ryan) (F : Jeffery) : J + R + F = 100

-- The theorem to prove
theorem ryan_caught_30_fish (J : Jason) (R : Ryan) (F : Jeffery) (h1 : R = 3 * J)
  (h2 : F = 60) (h3 : J + R + F = 100) : R = 30 :=
by
  sorry

end ryan_caught_30_fish_l348_348685


namespace negation_of_all_adults_good_cooks_l348_348225

-- Definitions of statements
def all_adults_good_cooks : Prop := ∀ a : Adult, GoodCook a
def at_least_one_adult_bad_cook : Prop := ∃ a : Adult, ¬GoodCook a

-- Theorem statement
theorem negation_of_all_adults_good_cooks :
  ¬all_adults_good_cooks ↔ at_least_one_adult_bad_cook := by
  sorry

end negation_of_all_adults_good_cooks_l348_348225


namespace cable_intersections_l348_348152

def combination (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem cable_intersections :
  let side1_houses := 6
  let side2_houses := 8
  let pairwise_intersections := combination side1_houses 2 * combination side2_houses 2
  pairwise_intersections = 420 :=
by
  let side1_houses := 6
  let side2_houses := 8
  let pairwise_intersections := combination side1_houses 2 * combination side2_houses 2
  have h1 : combination side1_houses 2 = 15 := by sorry
  have h2 : combination side2_houses 2 = 28 := by sorry
  have h3 : pairwise_intersections = 15 * 28 := by
    rw [h1, h2]
  show pairwise_intersections = 420 := by
    rw [h3]
    norm_num

end cable_intersections_l348_348152


namespace new_lengths_proof_l348_348829

-- Define the initial conditions
def initial_lengths : List ℕ := [15, 20, 24, 26, 30]
def original_average := initial_lengths.sum / initial_lengths.length
def average_decrease : ℕ := 5
def median_unchanged : ℕ := 24
def range_unchanged : ℕ := 15
def new_average := original_average - average_decrease

-- Assume new lengths
def new_lengths : List ℕ := [9, 9, 24, 24, 24]

-- Proof statement
theorem new_lengths_proof :
  (new_lengths.sorted.nth 2 = initial_lengths.sorted.nth 2) ∧
  (new_lengths.maximum - new_lengths.minimum = range_unchanged) ∧
  (new_lengths.sum / new_lengths.length = new_average) :=
by
  sorry

end new_lengths_proof_l348_348829


namespace sequence_sum_bound_l348_348404

noncomputable def a_seq : ℕ → ℕ
| 0     := 2
| (n+1) := (a_seq n)^2 - (a_seq n) + 1

theorem sequence_sum_bound :
  1 - 1 / (2003^2003 : ℝ) < (Finset.range 2003).sum (λ n => 1 / (a_seq n : ℝ)) ∧
  (Finset.range 2003).sum (λ n => 1 / (a_seq n : ℝ)) < 1 :=
sorry

end sequence_sum_bound_l348_348404


namespace Tn_less_than_16_over_9_l348_348219

noncomputable def a_n (n : ℕ) : ℝ := 1 / 4 ^ (n - 1)

def c_n (n : ℕ) : ℕ → ℝ := λ n, n * a_n n

def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, c_n (k + 1))

theorem Tn_less_than_16_over_9 (n : ℕ) : T_n n < 16 / 9 := 
  sorry

end Tn_less_than_16_over_9_l348_348219


namespace possible_solutions_l348_348337

noncomputable def sequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
(a 0 > a 1) ∧ (a 1 > a 2) ∧ (∀ i, i ≥ 2 → a i > 1) ∧
(∀ i, i ≤ n → a i > a (i + 1)) ∧
((1 - (1 / (a 1 : ℝ))) + (1 - (1 / (a 2 : ℝ))) + 
  ∑ i in finset.range (n - 1), (1 - (1 / (a (i + 2) : ℝ)))) = 
  2 * (1 - (1 / (a 0 : ℝ)))

theorem possible_solutions (n : ℕ) (a : ℕ → ℕ) :
  sequence n a → 
  (a 0 = 24 ∧ a 1 = 4 ∧ a 2 = 3 ∧ a 3 = 2) ∨ 
  (a 0 = 60 ∧ a 1 = 5 ∧ a 2 = 3 ∧ a 3 = 2) :=
by
  sorry

end possible_solutions_l348_348337


namespace distribution_y_value_l348_348381

theorem distribution_y_value :
  ∀ (x y : ℝ),
  (x + 0.1 + 0.3 + y = 1) →
  (7 * x + 8 * 0.1 + 9 * 0.3 + 10 * y = 8.9) →
  y = 0.4 :=
by
  intros x y h1 h2
  sorry

end distribution_y_value_l348_348381


namespace sum_coefficients_eq_nine_l348_348396

noncomputable def f (x : ℂ) (a b c d : ℝ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem sum_coefficients_eq_nine
  (a b c d : ℝ)
  (h1 : f (2 * Complex.I) a b c d = 0)
  (h2 : f (2 + Complex.I) a b c d = 0)
  (coeff_real : ∀ z : ℂ, f z a b c d ∈ ℝ) :
  a + b + c + d = 9 := by
  sorry

end sum_coefficients_eq_nine_l348_348396


namespace graduation_photos_l348_348282

-- Conditions
variables (x : ℕ)
variable (photos_given : ℕ)
hypothesis (h : photos_given = 2550)

-- Lean statement
theorem graduation_photos (h : photos_given = 2550) : x * (x - 1) = 2550 :=
sorry

end graduation_photos_l348_348282


namespace triangle_area_inscribed_in_circle_l348_348111

noncomputable def triangle_area_heron (a b c : ℝ) (s : ℝ) : ℝ :=
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_inscribed_in_circle (a b c : ℝ) (r : ℝ) :
  a = 6 ∧ b = 8 ∧ c = 10 ∧ r = 5 → 
  triangle_area_heron a b c ((a + b + c) / 2) = 24 :=
by
  sorry

end triangle_area_inscribed_in_circle_l348_348111


namespace average_cost_correct_l348_348092

-- Defining the conditions
def groups_of_4_oranges := 11
def cost_of_4_oranges_bundle := 15
def groups_of_7_oranges := 2
def cost_of_7_oranges_bundle := 25

-- Calculating the relevant quantities as per the conditions
def total_cost : ℕ := (groups_of_4_oranges * cost_of_4_oranges_bundle) + (groups_of_7_oranges * cost_of_7_oranges_bundle)
def total_oranges : ℕ := (groups_of_4_oranges * 4) + (groups_of_7_oranges * 7)
def average_cost_per_orange := (total_cost:ℚ) / (total_oranges:ℚ)

-- Proving the average cost per orange matches the correct answer
theorem average_cost_correct : average_cost_per_orange = 215 / 58 := by
  sorry

end average_cost_correct_l348_348092


namespace parabola_directrix_l348_348009

theorem parabola_directrix (a : ℝ) (h : a ≠ 0) : directrix_of_parabola_x_eq_ay2 a = - (1 / (4 * a)) :=
sorry

end parabola_directrix_l348_348009


namespace number_of_rectangles_containing_cell_l348_348947

theorem number_of_rectangles_containing_cell (m n p q : ℕ) (hp : 1 ≤ p ∧ p ≤ m) (hq : 1 ≤ q ∧ q ≤ n) :
    ∃ count : ℕ, count = p * q * (m - p + 1) * (n - q + 1) := 
    sorry

end number_of_rectangles_containing_cell_l348_348947


namespace conic_section_is_parabola_l348_348844

theorem conic_section_is_parabola (x y : ℝ) :
  abs (y - 3) = sqrt ((x + 4)^2 + y^2) →
  ∃ (A B C D : ℝ), A = 1 ∧ C = 0 ∧ (A * x^2 + B * x + C * y + D = 0) :=
by
  sorry

end conic_section_is_parabola_l348_348844


namespace smallest_positive_angle_l348_348913

theorem smallest_positive_angle (y : ℝ) (h : 10 * sin y * (cos y)^3 - 10 * (sin y)^3 * cos y = sqrt 2) : 
  y = 11.25 :=
by sorry

end smallest_positive_angle_l348_348913


namespace product_fractions_equals_23205_l348_348548

theorem product_fractions_equals_23205 :
  (\prod n in Finset.range 15 | n + 1, (n + 5) / (n + 1)) = 23205 := by
  sorry

end product_fractions_equals_23205_l348_348548


namespace find_f_prime_zero_l348_348213

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition given in the problem.
def f_def : ∀ x : ℝ, f x = x^2 + 2 * x * f' 1 := 
sorry

-- Statement we want to prove.
theorem find_f_prime_zero : f' 0 = -4 := 
sorry

end find_f_prime_zero_l348_348213


namespace intersection_points_count_length_of_AB_l348_348302

open Real

-- Definitions based on the problem's conditions
def param_line (t : ℝ) : ℝ × ℝ := (1/2 * t, 1 - (sqrt 3 / 2) * t)

def polar_eq_circle (θ : ℝ) : ℝ := 2 * sin θ

-- Cartesian forms derived from the conditions
def line_eq (x y : ℝ) : Prop := (sqrt 3) * x + y - 1 = 0

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Theorems to be proven
theorem intersection_points_count : ∃ (t1 t2 : ℝ), 
  let (x1, y1) := param_line t1 in 
  let (x2, y2) := param_line t2 in
  line_eq x1 y1 ∧ circle_eq x1 y1 ∧
  line_eq x2 y2 ∧ circle_eq x2 y2 ∧
  t1 ≠ t2 := 
sorry

theorem length_of_AB : 
  (∃ (t1 t2 : ℝ), 
    let (x1, y1) := param_line t1 in 
    let (x2, y2) := param_line t2 in
    circle_eq x1 y1 ∧ circle_eq x2 y2) → 
  dist (param_line t1) (param_line t2) = 2 := 
sorry

end intersection_points_count_length_of_AB_l348_348302


namespace polynomial_remainder_l348_348338

noncomputable def h (x : ℕ) := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder :
  ∀ (x : ℕ), h(x) ≠ 0 → 
  let h_x_10 := h(x^10) in (h_x_10 % h(x) = 7) :=
by
  intro x h_x_nonzero,
  sorry

end polynomial_remainder_l348_348338


namespace solve_log_eq_l348_348851

theorem solve_log_eq (x : ℝ) (h : x ^ 2 > 19) :
    \log_{2} \log_{3}(x^2 - 16) - \log_{\frac{1}{2}} \log_{\frac{1}{3}} \left(\frac{1}{x^2 - 16}\right) = 2
    ↔ (x = 5 ∨ x = -5) := by
  sorry

end solve_log_eq_l348_348851


namespace problem_solution_l348_348071

def data : List ℕ := [95, 91, 93, 95, 97, 99, 95, 98, 90, 99, 96, 94, 95, 97, 96, 92, 94, 95, 96, 98]

def interval : ℕ := 2

def num_classes : ℕ := 5

def class_range : ℕ × ℕ := (945, 965)

def frequency_945_965 : ℕ := 8

def relative_frequency_945_965 : ℚ := 8 / 20

theorem problem_solution :
  (data ∀ x, x ∈ data) ∧ 
  (interval = 2) ∧ 
  (num_classes = 5) ∧ 
  (frequency_945_965 = 8) ∧ 
  (relative_frequency_945_965 = 0.4) := 
by
  sorry

end problem_solution_l348_348071


namespace number_of_elements_in_M_l348_348015

noncomputable def M : set ℂ := { x | ∃ n : ℕ, x = complex.I^n + complex.I^(-n) }

theorem number_of_elements_in_M : (M.to_finset.card = 3) := sorry

end number_of_elements_in_M_l348_348015


namespace ingrid_tax_rate_l348_348690

def john_income : ℝ := 57000
def ingrid_income : ℝ := 72000
def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_tax_rate :
  let john_tax := john_tax_rate * john_income
  let combined_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * combined_income
  let ingrid_tax := total_tax - john_tax
  let ingrid_tax_rate := ingrid_tax / ingrid_income
  ingrid_tax_rate = 0.40 :=
by
  sorry

end ingrid_tax_rate_l348_348690


namespace distribution_schemes_l348_348150

-- Define the conditions: six teachers, four neighborhoods, each neighborhood receiving at least one teacher.
def num_teachers : ℕ := 6
def num_neighborhoods : ℕ := 4

theorem distribution_schemes :
  ∃ (n : ℕ), n = 1560 :=
by 
  -- We state that there exists an n, which is the number of distribution schemes, and it equals 1560
  use 1560
  sorry

end distribution_schemes_l348_348150


namespace disproving_statement_B_l348_348920

structure Vector2D (ℝ : Type) :=
  (m : ℝ)
  (n : ℝ)

def operation1 (v1 v2 : Vector2D ℝ) : ℝ :=
  v1.m * v2.n - v1.n * v2.m

def operation2 (v1 v2 : Vector2D ℝ) : ℝ :=
  v1.m * v2.m + v1.n * v2.n

theorem disproving_statement_B (a b : Vector2D ℝ) : 
  operation1 a b ≠ operation1 b a := by
  sorry

end disproving_statement_B_l348_348920


namespace new_ribbon_lengths_correct_l348_348820

noncomputable def ribbon_lengths := [15, 20, 24, 26, 30]
noncomputable def new_average_change := 5
noncomputable def new_lengths := [9, 9, 24, 24, 24]

theorem new_ribbon_lengths_correct :
  let new_length_list := [9, 9, 24, 24, 24]
  ribbon_lengths.length = 5 ∧ -- we have 5 ribbons
  new_length_list.length = 5 ∧ -- the new list also has 5 ribbons
  list.average new_length_list = list.average ribbon_lengths - new_average_change ∧ -- new average decreased by 5
  list.median new_length_list = list.median ribbon_lengths ∧ -- median unchanged
  list.range new_length_list = list.range ribbon_lengths -- range unchanged
  :=
by {
  sorry
}

end new_ribbon_lengths_correct_l348_348820


namespace solution_set_l348_348742

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l348_348742


namespace sum_of_other_endpoint_coordinates_l348_348395

theorem sum_of_other_endpoint_coordinates (x y : ℤ) :
  (7 + x) / 2 = 5 ∧ (4 + y) / 2 = -8 → x + y = -17 :=
by 
  sorry

end sum_of_other_endpoint_coordinates_l348_348395


namespace find_integer_pairs_l348_348005

theorem find_integer_pairs (x y : ℤ) (t : ℤ) :
  x ≠ y ∧ (x * y - (x + y) = Int.gcd x y + Int.lcm x y) ↔ 
  ((x, y) = (6, 3) ∨ (x, y) = (6, 4) ∨ ∃ t, (x, y) = (1 + t, -t) ∨ ∃ t, (x, y) = (2, -2t)) :=
by sorry

end find_integer_pairs_l348_348005


namespace largest_whole_number_less_than_100_l348_348788

theorem largest_whole_number_less_than_100 (x : ℕ) (h1 : 7 * x < 100) (h_max : ∀ y : ℕ, 7 * y < 100 → y ≤ x) :
  x = 14 := 
sorry

end largest_whole_number_less_than_100_l348_348788


namespace bob_remaining_money_l348_348510

noncomputable def remaining_money_after_wednesday : ℕ :=
let monday := 80 - (1 / 2) * 80 in
let tuesday := monday - (1 / 5) * monday in
let wednesday := tuesday - (3 / 8) * tuesday in
wednesday

theorem bob_remaining_money : remaining_money_after_wednesday = 20 :=
by sorry

end bob_remaining_money_l348_348510


namespace angle_between_lines_l348_348266

theorem angle_between_lines (a : ℝ) (θ : ℝ) (hθ : θ = Real.arccos (√5 / 5))
  (h_angle : θ = Real.arccos (√5 / 5)) :
  (a = -3 / 4) ↔
  (Real.tan θ = |((1 / 2) - a) / (1 + (1 / 2) * a)|) :=
begin
  sorry
end

end angle_between_lines_l348_348266


namespace solve_equation_solve_inequality_system_l348_348866

theorem solve_equation :
  ∃ x, 2 * x^2 - 4 * x - 1 = 0 ∧ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
sorry

theorem solve_inequality_system : 
  ∀ x, (2 * x + 3 > 1 → -1 < x) ∧
       (x - 2 ≤ (1 / 2) * (x + 2) → x ≤ 6) ∧ 
       (2 * x + 3 > 1 ∧ x - 2 ≤ (1 / 2) * (x + 2) ↔ (-1 < x ∧ x ≤ 6)) :=
sorry

end solve_equation_solve_inequality_system_l348_348866


namespace hyperbola_problem_l348_348583

theorem hyperbola_problem :
    (∃ (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ),
      a = b ∧
      c = 2 * real.sqrt 3 ∧
      F₁ = (-2 * real.sqrt 3, 0) ∧
      F₂ = (2 * real.sqrt 3, 0) ∧
      P = (4, -real.sqrt 10) ∧
      (∀ M : ℝ × ℝ, 
        let x₁ := M.1 in let y₁ := M.2 in
        M.1^2 - M.2^2 = 6 → 
        ∀ M : ℝ × ℝ, 
          let x₁ := M.1 in let y₁ := M.2 in 
          (y₁^2 = -6 + x₁^2) → 
          (x₁ - real.sqrt 3)^2 * 2 ≥ 18 - 12 * real.sqrt 2)) :=
  sorry

end hyperbola_problem_l348_348583


namespace sugar_per_bag_l348_348346

theorem sugar_per_bag :
  ∀ (cups_at_home : ℕ) (num_bags : ℕ) (sugar_per_bag : ℕ) (batter_sugar : ℕ) (frosting_sugar : ℕ) (dozen_cupcakes : ℕ),
  cups_at_home = 3 →
  num_bags = 2 →
  batter_sugar = 1 →
  frosting_sugar = 2 →
  dozen_cupcakes = 5 →
  (batter_sugar * dozen_cupcakes + frosting_sugar * dozen_cupcakes - cups_at_home) / num_bags = sugar_per_bag →
  sugar_per_bag = 6 := 
by
  intros cups_at_home num_bags sugar_per_bag batter_sugar frosting_sugar dozen_cupcakes
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  have eq1 : (1 * 5 + 2 * 5 - 3) / 2 = sugar_per_bag := by
    rw [h6]
  norm_num at eq1
  exact eq1
sorry

end sugar_per_bag_l348_348346


namespace sum_of_three_digit_products_of_four_distinct_primes_l348_348950

-- Define a function to check if a number is a product of four distinct primes
def isProductOfFourDistinctPrimes (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = a * b * c * d

-- Define the range of three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the set of numbers that meet the conditions
def validNumbers : List ℕ :=
  [210, 330, 390, 510, 570, 690, 870, 930, 462, 546, 714, 798, 966, 770, 910, 858]

-- Main theorem stating that the sum of all valid numbers is 10494
theorem sum_of_three_digit_products_of_four_distinct_primes :
  (validNumbers.filter isThreeDigitNumber).filter isProductOfFourDistinctPrimes |> List.sum = 10494 :=
by
  sorry

end sum_of_three_digit_products_of_four_distinct_primes_l348_348950


namespace janna_wrote_more_words_than_yvonne_l348_348437

theorem janna_wrote_more_words_than_yvonne :
  ∃ (janna_words_written yvonne_words_written : ℕ), 
    yvonne_words_written = 400 ∧
    janna_words_written > yvonne_words_written ∧
    ∃ (removed_words added_words : ℕ),
      removed_words = 20 ∧
      added_words = 2 * removed_words ∧
      (janna_words_written + yvonne_words_written - removed_words + added_words + 30 = 1000) ∧
      (janna_words_written - yvonne_words_written = 130) :=
by
  sorry

end janna_wrote_more_words_than_yvonne_l348_348437


namespace sum_of_numbers_l348_348399

theorem sum_of_numbers (x : ℝ) (h1 : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) : x + 2 * x + 4 * x = 63 :=
sorry

end sum_of_numbers_l348_348399


namespace XT_value_l348_348868

noncomputable def AB := 15
noncomputable def BC := 20
noncomputable def height_P := 30
noncomputable def volume_ratio := 9

theorem XT_value 
  (AB BC height_P : ℕ)
  (volume_ratio : ℕ)
  (h1 : AB = 15)
  (h2 : BC = 20)
  (h3 : height_P = 30)
  (h4 : volume_ratio = 9) : 
  ∃ (m n : ℕ), m + n = 97 ∧ m.gcd n = 1 :=
by sorry

end XT_value_l348_348868


namespace magnitude_AF_l348_348987

noncomputable def ellipse := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

noncomputable def right_focus : ℝ × ℝ := (1, 0)

noncomputable def right_directrix := {p : ℝ × ℝ | p.1 = 2}

variable A : ℝ × ℝ
variable B : ℝ × ℝ

-- Condition: Point A lies on the right directrix
axiom A_on_directrix : A ∈ right_directrix

-- Condition: AF intersects the ellipse C at point B
axiom AF_intersects_ellipse_at_B : line_segment (right_focus, A) ∩ ellipse = {B}

-- Condition: FA = 3 * FB
axiom FA_eq_3_FB : ∥right_focus - A∥ = 3 * ∥right_focus - B∥

-- Proof of the magnitude of segment AF
theorem magnitude_AF : ∥right_focus - A∥ = sqrt 2 :=
sorry

end magnitude_AF_l348_348987


namespace arrange_descending_order_l348_348979

theorem arrange_descending_order (m : ℝ) (h1 : 1 < m) (h2 : m < 2) :
  let a := 0.3 ^ m
  let b := Real.log m / Real.log 0.3
  let c := m ^ 0.3
  c > a ∧ a > b := by
  sorry

end arrange_descending_order_l348_348979


namespace tire_circumference_l348_348076

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) (h_rpm : rpm = 400) (h_speed_kmh : speed_kmh = 48) :
  (C = 2) :=
by
  -- sorry statement to assume the solution for now
  sorry

end tire_circumference_l348_348076


namespace choose_distinct_integers_l348_348130

theorem choose_distinct_integers :
  ∃ (T : Set ℕ), (T.card = 1983) ∧ (∀ t ∈ T, t ≤ 10^5) ∧ (∀ t1 t2 t3 ∈ T, t1 < t2 → t2 < t3 → 2 * t2 ≠ t1 + t3) :=
sorry

end choose_distinct_integers_l348_348130


namespace evaluate_expression_correct_l348_348155

noncomputable def evaluate_expression : ℤ :=
  6 - 8 * (9 - 4 ^ 2) * 5 + 2

theorem evaluate_expression_correct : evaluate_expression = 288 := by
  sorry

end evaluate_expression_correct_l348_348155


namespace divisor_of_1104_l348_348429

theorem divisor_of_1104 : ∃ d, d ∣ 1104 ∧ d = 8 := by
  exists 8
  -- d ∣ 1104 means that there's no remainder when you divide 1104 by d.
  -- Here we need to show that 8 divides 1104.
  simp
  sorry

end divisor_of_1104_l348_348429


namespace find_days_l348_348173

theorem find_days (n : ℕ) (P : ℕ) (d : ℕ → ℕ)
  (havg : P = 50 * n)
  (hgeo : (∏ i in finset.range n, d i) = 45 ^ n)
  (hlimit : ∀ i, i < n → 20 ≤ d i ∧ d i ≤ 120)
  (hnew_avg : (P + 105) / (n + 1) = 55) :
  n = 10 :=
by
  sorry

end find_days_l348_348173


namespace value_of_ON_l348_348466

-- Definitions of the geometric objects and properties
variable (M F1 F2 N O : Point)
variable (dist_MF1 : ℝ)
-- Ellipse equation
def ellipse : Prop := M.x^2 / 25 + M.y^2 / 9 = 1
-- Distance conditions
def distance_foci : Prop := dist(M, F1) = 2
def midpoint_N : Prop := N = midpoint(M, F1)
def origin_O : Prop := O = (0, 0)
-- Target length |ON| == 4
def O_N_length : Prop := dist(O, N) = 4

-- The theorem stating the proof goal
theorem value_of_ON 
  (M_on_ellipse : ellipse)
  (dist_MF1_condition : distance_foci)
  (midpoint_N_condition : midpoint_N)
  (origin_O_condition : origin_O) :
  O_N_length :=
by sorry

end value_of_ON_l348_348466


namespace value_of_x_plus_y_l348_348255

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l348_348255


namespace equilateral_triangle_and_square_AD_div_BC_l348_348664

noncomputable def triangle_side_length : Type := ℝ
noncomputable def square_side_length : Type := ℝ

-- Definitions of the conditions
def is_equilateral_triangle (A B C : ℝ × ℝ) (BC : triangle_side_length) : Prop :=
  ∃ s : ℝ, BC = s ∧ distance A B = s ∧ distance B C = s ∧ distance C A = s

def is_square_with_side_BC (B C D E : ℝ × ℝ) (BC : square_side_length) : Prop :=
  ∃ s : ℝ, BC = s ∧ distance B C = s ∧ distance C D = s ∧ distance D E = s ∧ distance E B = s ∧
           ∠ B C D = π/2 ∧ ∠ C D E = π/2 ∧ ∠ D E B = π/2 ∧ ∠ E B C = π/2

-- Problem statement
theorem equilateral_triangle_and_square_AD_div_BC (A B C D E : ℝ × ℝ) (BC : square_side_length) :
  is_equilateral_triangle A B C BC →
  is_square_with_side_BC B C D E BC →
  (distance A D) / (distance B C) = (real.sqrt 2 + real.sqrt 3 / 2) :=
by
  intros h_eq_triangle h_square_with_side_BC
  sorry

end equilateral_triangle_and_square_AD_div_BC_l348_348664


namespace age_of_second_replaced_man_l348_348374

theorem age_of_second_replaced_man (avg_age_increase : ℕ) (avg_new_men_age : ℕ) (first_replaced_age : ℕ) (total_men : ℕ) (new_age_sum : ℕ) :
  avg_age_increase = 1 →
  avg_new_men_age = 34 →
  first_replaced_age = 21 →
  total_men = 12 →
  new_age_sum = 2 * avg_new_men_age →
  47 - (new_age_sum - (first_replaced_age + x)) = 12 →
  x = 35 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end age_of_second_replaced_man_l348_348374


namespace men_absent_l348_348461

/-- 
A group of men decided to do a work in 20 days, but some of them became absent. 
The rest of the group did the work in 40 days. The original number of men was 20. 
Prove that 10 men became absent. 
--/
theorem men_absent 
    (original_men : ℕ) (absent_men : ℕ) (planned_days : ℕ) (actual_days : ℕ)
    (h1 : original_men = 20) (h2 : planned_days = 20) (h3 : actual_days = 40)
    (h_work : original_men * planned_days = (original_men - absent_men) * actual_days) : 
    absent_men = 10 :=
    by 
    rw [h1, h2, h3] at h_work
    -- Proceed to manually solve the equation, but here we add sorry
    sorry

end men_absent_l348_348461


namespace diagonals_in_dodecagon_l348_348636

theorem diagonals_in_dodecagon : ∀ (n : ℕ), n = 12 → (n * (n - 3)) / 2 = 54 := by
  intros n h
  rw [h]
  norm_num
  sorry

end diagonals_in_dodecagon_l348_348636


namespace concyclic_points_iff_concyclic_prime_points_l348_348052

noncomputable def circles_concyclic (ω1 ω2 : Set (Set ℝ)) (P A A' B B' C C' D D' : ℝ × ℝ) :=
  ∃ γ : Set (ℝ × ℝ), circle γ ∧ ({A, B, C, D} ⊆ γ) ↔ ∃ γ' : Set (ℝ × ℝ), circle γ' ∧ ({A', B', C', D'} ⊆ γ')

variables {ω1 ω2 : Set (Set ℝ)}
variables {P A A' B B' C C' D D' : ℝ × ℝ}

theorem concyclic_points_iff_concyclic_prime_points :
  (two_non_intersecting_circles ω1 ω2) →
  (point_outside_circles P ω1 ω2) →
  (first_line_through_P P A A' B B' ω1 ω2) →
  (second_line_through_P P C C' D D' ω1 ω2) →
  circles_concyclic ω1 ω2 P A A' B B' C C' D D' :=
sorry

end concyclic_points_iff_concyclic_prime_points_l348_348052


namespace car_moving_speed_mph_l348_348453

-- Definitions
def car_efficiency_km_per_liter : ℝ := 72
def fuel_decrease_gallons : ℝ := 3.9
def time_hours : ℝ := 5.7
def gallon_to_liters : ℝ := 3.8
def km_to_miles : ℝ := 1.6

def convert_gallons_to_liters (g: ℝ) : ℝ :=
  g * gallon_to_liters

def total_distance_km (fuel_liters: ℝ) : ℝ :=
  fuel_liters * car_efficiency_km_per_liter

def convert_km_to_miles (km: ℝ) : ℝ :=
  km / km_to_miles

def car_speed_mph (distance_miles time_hours: ℝ) : ℝ :=
  distance_miles / time_hours

-- Theorem to prove the car speed in mph
theorem car_moving_speed_mph : car_speed_mph (convert_km_to_miles (total_distance_km (convert_gallons_to_liters fuel_decrease_gallons))) time_hours = 117.04 :=
  by
    sorry

end car_moving_speed_mph_l348_348453


namespace exponential_function_passes_through_01_l348_348388

theorem exponential_function_passes_through_01 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^0 = 1) :=
by
  sorry

end exponential_function_passes_through_01_l348_348388


namespace greatest_c_not_in_range_l348_348426

theorem greatest_c_not_in_range (c : ℤ) : ∀ (x : ℝ), x^2 + c * x + 15 ≠ -9 → c ≤ 9 :=
by
  sorry

end greatest_c_not_in_range_l348_348426


namespace lesser_number_l348_348796

theorem lesser_number (x y : ℕ) (h1 : x + y = 58) (h2 : x - y = 6) : y = 26 :=
by
  sorry

end lesser_number_l348_348796


namespace tan_alpha_eq_one_implies_f_alpha_eq_one_l348_348871

theorem tan_alpha_eq_one_implies_f_alpha_eq_one (α : ℝ) (h : Real.tan α = 1) :
  let f : ℝ → ℝ := λ α, (Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) / (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))
  f α = 1 :=
by
  sorry

end tan_alpha_eq_one_implies_f_alpha_eq_one_l348_348871


namespace survey_support_percentage_l348_348467

theorem survey_support_percentage 
  (num_men : ℕ) (percent_men_support : ℝ)
  (num_women : ℕ) (percent_women_support : ℝ)
  (h_men : num_men = 200)
  (h_percent_men_support : percent_men_support = 0.7)
  (h_women : num_women = 500)
  (h_percent_women_support : percent_women_support = 0.75) :
  (num_men * percent_men_support + num_women * percent_women_support) / (num_men + num_women) * 100 = 74 := 
by
  sorry

end survey_support_percentage_l348_348467


namespace exists_small_area_triangle_l348_348210

structure Point :=
(x : ℝ)
(y : ℝ)

def is_valid_point (p : Point) : Prop :=
(|p.x| ≤ 2) ∧ (|p.y| ≤ 2)

def no_three_collinear (points : List Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  (p1 ≠ p2) → (p1 ≠ p3) → (p2 ≠ p3) →
  ((p1.y - p2.y) * (p1.x - p3.x) ≠ (p1.y - p3.y) * (p1.x - p2.x))

noncomputable def triangle_area (p1 p2 p3: Point) : ℝ :=
(abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) / 2

theorem exists_small_area_triangle (points : List Point)
  (h_valid : ∀ p ∈ points, is_valid_point p)
  (h_no_collinear : no_three_collinear points)
  (h_len : points.length = 6) :
  ∃ (p1 p2 p3: Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  triangle_area p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l348_348210


namespace determine_ordered_pair_l348_348923

theorem determine_ordered_pair (a b : ℝ) 
  (h1 : cos θ = 4 / 5)
  (h2 : sin θ = 3 / 5)
  (h3 : ∃ (θ : ℝ), True):
  ((a = 4 / 5) ∧ (b = -3 / 5)) :=
by
  sorry

end determine_ordered_pair_l348_348923


namespace first_recipe_cups_l348_348771

-- Definitions based on the given conditions
def ounces_per_bottle : ℕ := 16
def ounces_per_cup : ℕ := 8
def cups_second_recipe : ℕ := 1
def cups_third_recipe : ℕ := 3
def total_bottles : ℕ := 3
def total_ounces : ℕ := total_bottles * ounces_per_bottle
def total_cups_needed : ℕ := total_ounces / ounces_per_cup

-- Proving the amount of cups of soy sauce needed for the first recipe
theorem first_recipe_cups : 
  total_cups_needed - (cups_second_recipe + cups_third_recipe) = 2 
:= by 
-- Proof omitted
  sorry

end first_recipe_cups_l348_348771


namespace num_trips_l348_348724

theorem num_trips (t d : ℕ) (h_t: t = 120) (h_d: d = 30) : t / d = 4 :=
by
  rw [h_t, h_d]
  simp
  sorry

end num_trips_l348_348724


namespace part_a_part_b_l348_348890

-- Define what it means for a pair (a, p) to be good
def is_good (a p : ℕ) : Prop :=
  (a^3 + p^3) % (a^2 - p^2) = 0 ∧ a > p

-- Define the list of prime numbers less than 18
def prime_list : List ℕ := [2, 3, 5, 7, 11, 13, 17]

-- Statement 1: Finding a specific a for p = 17
theorem part_a : ∃ a, is_good a 17 ∧ (a = 18 ∨ a = 34 ∨ a = 306) :=
sorry

-- Statement 2: Count the number of good pairs (a, p) where p is prime and less than 18
theorem part_b : (finset.univ.filter (λ p : ℕ, p ∈ prime_list).sum 
                  (λ p, finset.card (finset.filter (λ a, is_good a p) finset.univ))) = 21 :=
sorry

end part_a_part_b_l348_348890


namespace intersection_A_B_l348_348715

def setA : Set ℝ := {x | 1 ≤ 3^x ∧ 3^x ≤ 81}
def setB : Set ℝ := {x | log 2 (x^2 - x) > 1}

theorem intersection_A_B :
  setA ∩ setB = {x | 2 < x ∧ x ≤ 4} :=
sorry

end intersection_A_B_l348_348715


namespace number_of_girls_with_no_pets_l348_348283

-- Define total number of students
def total_students : ℕ := 30

-- Define the fraction of boys in the class
def fraction_boys : ℚ := 1 / 3

-- Define the percentages of girls with pets
def percentage_girls_with_dogs : ℚ := 0.40
def percentage_girls_with_cats : ℚ := 0.20

-- Calculate the number of boys
def number_of_boys : ℕ := (fraction_boys * total_students).toNat

-- Calculate the number of girls
def number_of_girls : ℕ := total_students - number_of_boys

-- Calculate the number of girls who own dogs
def number_of_girls_with_dogs : ℕ := (percentage_girls_with_dogs * number_of_girls).toNat

-- Calculate the number of girls who own cats
def number_of_girls_with_cats : ℕ := (percentage_girls_with_cats * number_of_girls).toNat

-- Define the statement to be proved
theorem number_of_girls_with_no_pets : number_of_girls - (number_of_girls_with_dogs + number_of_girls_with_cats) = 8 := by
  sorry

end number_of_girls_with_no_pets_l348_348283


namespace correct_answer_l348_348631

def M : Set ℤ := {x | |x| < 5}

theorem correct_answer : {0} ⊆ M := by
  sorry

end correct_answer_l348_348631


namespace pairC_opposite_numbers_l348_348904

-- Definitions used in the conditions
def setA := (abs (- (2 / 3)), - (- (2 / 3)))
def setB := (abs (- (2 / 3)), - abs (- (3 / 2)))
def setC := (abs (- (2 / 3)), + (- (2 / 3)))
def setD := (abs (- (2 / 3)), abs (- (3 / 2)))

-- The statement to prove which pair are opposite numbers
theorem pairC_opposite_numbers : 
  (setC.1 = abs (- (2 / 3)) ∧ setC.2 = + (- (2 / 3)) ∧ setC.1 = - setC.2) ∧ 
  (¬ (setA.1 = - setA.2)) ∧ 
  (¬ (setB.1 = - setB.2)) ∧ 
  (¬ (setD.1 = - setD.2)) := 
by 
  -- Proof to be provided
  sorry

end pairC_opposite_numbers_l348_348904


namespace v_is_82_875_percent_of_z_l348_348277

theorem v_is_82_875_percent_of_z (x y z w v : ℝ) 
  (h1 : x = 1.30 * y)
  (h2 : y = 0.60 * z)
  (h3 : w = 1.25 * x)
  (h4 : v = 0.85 * w) : 
  v = 0.82875 * z :=
by
  sorry

end v_is_82_875_percent_of_z_l348_348277


namespace sum_first_10_terms_arithmetic_sequence_l348_348207

open Nat

-- Defining the arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_first_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) : Prop :=
  S n = ∑ i in range n, a i

theorem sum_first_10_terms_arithmetic_sequence :
  ∃ a S : ℕ → ℕ, 
    arithmetic_sequence a 2 ∧ 
    a 4 = 10 ∧ 
    sum_first_n a S 10 ∧ 
    S 10 = 110 :=
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l348_348207


namespace minimum_spending_for_20_oranges_l348_348355

theorem minimum_spending_for_20_oranges :
  ∀ (cost_bag cost_box : ℕ) (bag_size box_size total_oranges : ℕ),
  cost_bag = 12 → cost_box = 25 →
  bag_size = 4 → box_size = 6 →
  total_oranges = 20 →
  0 ≤ total_oranges → 
  (∀ (x y : ℕ), x * bag_size + y * box_size = total_oranges → 
  x * cost_bag + y * cost_box ≥ 60) :=
begin
  intros cost_bag cost_box bag_size box_size total_oranges hc_bag hc_box hc_b_size hc_b_size hc_t_oranges h_oranges_pos,
  sorry
end

end minimum_spending_for_20_oranges_l348_348355


namespace circumscribed_circle_radius_l348_348651

noncomputable def radius_of_circumscribed_circle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
  let R := a / (2 * Real.sin A)
  R

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (hb : b = 4) (hc : c = 2) (hA : A = Real.pi / 3) :
  radius_of_circumscribed_circle b c A = 2 := by
  sorry

end circumscribed_circle_radius_l348_348651


namespace area_triangle_AEF_is_correct_l348_348297

structure Rectangle :=
  (BE EC CF FD : ℝ)
  (h_BE : BE = 5)
  (h_EC : EC = 4)
  (h_CF : CF = 4)
  (h_FD : FD = 1)

def area_of_triangle_AEF (rect : Rectangle) : ℝ :=
  let AD := rect.BE + rect.EC in
  let AB := rect.CF + rect.FD in
  let area_ABCD := AB * AD in
  let area_EFC := 0.5 * rect.EC * rect.CF in
  let area_ABE := 0.5 * AB * rect.BE in
  let area_ADF := 0.5 * AD * rect.FD in
  area_ABCD - area_EFC - area_ABE - area_ADF

theorem area_triangle_AEF_is_correct (rect : Rectangle) : area_of_triangle_AEF rect = 20 := by
  sorry

end area_triangle_AEF_is_correct_l348_348297


namespace find_radius_probability_l348_348093

theorem find_radius_probability (r: ℝ) (h: \(\pi r^2 = \frac{1}{3}\)) : r = 0.325 :=
sorry

end find_radius_probability_l348_348093


namespace shaded_area_concentric_circles_l348_348385

theorem shaded_area_concentric_circles 
  (CD_length : ℝ) (r1 r2 : ℝ) 
  (h_tangent : CD_length = 100) 
  (h_radius1 : r1 = 60) 
  (h_radius2 : r2 = 40) 
  (tangent_condition : CD_length = 2 * real.sqrt (r1^2 - r2^2)) :
  ∃ area : ℝ, area = π * (r1^2 - r2^2) ∧ area = 2000 * π :=
by
  use π * (r1^2 - r2^2)
  have h1 : r1^2 = 3600 := by { rw h_radius1, norm_num }
  have h2 : r2^2 = 1600 := by { rw h_radius2, norm_num }
  rw [h1, h2]
  simp
  sorry

end shaded_area_concentric_circles_l348_348385


namespace coefficient_of_x2_in_polynomial_is_neg13_l348_348640

theorem coefficient_of_x2_in_polynomial_is_neg13 :
  let f := (λ x : ℝ, (x-1)^5 + (x-1)^3 + (x-1)) in
  (polynomial.coeff f.polynomialize 2) = -13 :=
by
  sorry

end coefficient_of_x2_in_polynomial_is_neg13_l348_348640


namespace speed_of_first_train_l348_348815

noncomputable def speed_of_second_train : ℝ := 40 -- km/h
noncomputable def length_of_first_train : ℝ := 125 -- m
noncomputable def length_of_second_train : ℝ := 125.02 -- m
noncomputable def time_to_pass_each_other : ℝ := 1.5 / 60 -- hours (converted from minutes)

theorem speed_of_first_train (V1 V2 : ℝ) 
  (h1 : V2 = speed_of_second_train)
  (h2 : 125 + 125.02 = 250.02) 
  (h3 : 1.5 / 60 = 0.025) :
  V1 - V2 = 10.0008 → V1 = 50 :=
by 
  sorry

end speed_of_first_train_l348_348815


namespace final_single_digit_is_three_l348_348070

/-- Theorem: Starting with the number consisting of 100 nines (999...999, 100 times),
and applying the allowed operations any number of times, results in a final single-digit number
which is 3. -/
theorem final_single_digit_is_three
  (initial_number : ℕ) 
  (h : initial_number = 999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999_999) 
  (allowed_operations : ℕ → ℕ)
  (operations_conserve_modulo : ∀ n k, k ≤ initial_number → allowed_operations (n % 7) = (allowed_operations n) % 7) : 
  allowed_operations initial_number % 7 = 3 :=
begin
  sorry
end

end final_single_digit_is_three_l348_348070


namespace train_length_l348_348478

def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

def length_of_train (speed_ms time_s : ℝ) : ℝ :=
  speed_ms * time_s

theorem train_length 
  (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 144)
  (h2 : time_s = 2.9997600191984644) :
  length_of_train (speed_kmh_to_ms speed_kmh) time_s = 119.99040076793858 :=
by
  sorry

end train_length_l348_348478


namespace tommy_paint_cost_l348_348042

def tommy_spends_on_paint : ℕ :=
  let width := 5 in
  let height := 4 in
  let sides := 2 in
  let cost_per_quart := 2 in
  let coverage_per_quart := 4 in
  let area_per_side := width * height in
  let total_area := sides * area_per_side in
  let quarts_needed := total_area / coverage_per_quart in
  let total_cost := quarts_needed * cost_per_quart in
  total_cost

theorem tommy_paint_cost : tommy_spends_on_paint = 20 := by
  sorry

end tommy_paint_cost_l348_348042


namespace range_of_b_for_increasing_f_l348_348647

noncomputable def f (b x : ℝ) : ℝ :=
  if x > 1 then (2 * b - 1) / x + b + 3 else -x^2 + (2 - b) * x

theorem range_of_b_for_increasing_f :
  ∀ b : ℝ, (∀ x1 x2 : ℝ, x1 < x2 → f b x1 ≤ f b x2) ↔ -1/4 ≤ b ∧ b ≤ 0 := 
sorry

end range_of_b_for_increasing_f_l348_348647


namespace initial_people_on_train_l348_348481

theorem initial_people_on_train {x y z u v w : ℤ} 
  (h1 : y = 29) (h2 : z = 17) (h3 : u = 27) (h4 : v = 35) (h5 : w = 116) :
  x - (y - z) + (v - u) = w → x = 120 := 
by sorry

end initial_people_on_train_l348_348481


namespace multiple_of_5_digits_B_l348_348956

theorem multiple_of_5_digits_B (B : ℕ) : B = 0 ∨ B = 5 ↔ 23 * 10 + B % 5 = 0 :=
by
  sorry

end multiple_of_5_digits_B_l348_348956


namespace train_passing_time_approximately_12_seconds_l348_348073

-- Define the conditions
def length_of_train : ℝ := 200 -- in meters
def speed_of_train : ℝ := 68 -- in kmph
def speed_of_man : ℝ := 8 -- in kmph

-- Define the conversion factors and computations
def relative_speed_kmph := speed_of_train - speed_of_man
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600
def relative_speed_mps := kmph_to_mps relative_speed_kmph
def time_to_pass := length_of_train / relative_speed_mps

-- The main theorem to prove
theorem train_passing_time_approximately_12_seconds :
  abs (time_to_pass - 12) < 0.01 :=
by
  sorry

end train_passing_time_approximately_12_seconds_l348_348073


namespace trench_dig_time_l348_348185

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l348_348185


namespace sum_of_x_and_y_l348_348257

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l348_348257


namespace time_for_first_three_workers_l348_348194

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l348_348194


namespace find_k_value_l348_348915

def f (x : ℝ) : ℝ := 5 * x^2 - 1 / x + 3
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k_value (k : ℝ) (h : f 2 - g 2 k = 2) : k = -25 / 2 := by
  have f2 := f 2
  have g2 := g 2 k
  sorry

end find_k_value_l348_348915


namespace polygon_area_is_12_l348_348556

def polygon_vertices := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

def area_of_polygon (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to compute the area (stub here for now)
  sorry

theorem polygon_area_is_12 :
  area_of_polygon polygon_vertices = 12 :=
by
  sorry

end polygon_area_is_12_l348_348556


namespace bus_speed_excluding_stoppages_l348_348561

theorem bus_speed_excluding_stoppages (s_including_stops : ℕ) (stop_time_minutes : ℕ) (s_excluding_stops : ℕ) (v : ℕ) : 
  (s_including_stops = 45) ∧ (stop_time_minutes = 24) ∧ (v = s_including_stops * 5 / 3) → s_excluding_stops = 75 := 
by {
  sorry
}

end bus_speed_excluding_stoppages_l348_348561


namespace milk_pumping_rate_l348_348806

theorem milk_pumping_rate (
    initial_milk : ℕ,
    pump_time : ℕ,
    add_rate : ℕ,
    add_time : ℕ,
    final_milk : ℕ
) (h_initial : initial_milk = 30000)
  (h_pump : pump_time = 4)
  (h_rate : add_rate = 1500)
  (h_add_time : add_time = 7)
  (h_final : final_milk = 28980) :
  (initial_milk + add_rate * add_time - final_milk) / pump_time = 2880 :=
by
  sorry

end milk_pumping_rate_l348_348806


namespace minimum_odd_correct_answers_l348_348456

theorem minimum_odd_correct_answers (students : Fin 50 → Fin 5) :
  (∀ S : Finset (Fin 50), S.card = 40 → 
    (∃ x ∈ S, students x = 3) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, students x₁ = 2 ∧ x₁ ≠ x₂ ∧ students x₂ = 2) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, students x₁ = 1 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ students x₂ = 1 ∧ students x₃ = 1) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, ∃ x₄ ∈ S, students x₁ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ students x₂ = 0 ∧ students x₃ = 0 ∧ students x₄ = 0)) →
  (∃ S : Finset (Fin 50), (∀ x ∈ S, (students x = 1 ∨ students x = 3)) ∧ S.card = 23) :=
by
  sorry

end minimum_odd_correct_answers_l348_348456


namespace average_visitors_per_day_l348_348438

/-- The average number of visitors per day in a month of 30 days that begins with a Sunday is 188, 
given that the library has 500 visitors on Sundays and 140 visitors on other days. -/
theorem average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) 
   (starts_on_sunday : Bool) (sundays : ℕ) 
   (visitors_sunday_eq_500 : visitors_sunday = 500)
   (visitors_other_eq_140 : visitors_other = 140)
   (days_in_month_eq_30 : days_in_month = 30)
   (starts_on_sunday_eq_true : starts_on_sunday = true)
   (sundays_eq_4 : sundays = 4) :
   (visitors_sunday * sundays + visitors_other * (days_in_month - sundays)) / days_in_month = 188 := 
by {
  sorry
}

end average_visitors_per_day_l348_348438


namespace sum_of_x_and_y_l348_348256

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l348_348256


namespace solution_set_unique_line_l348_348746

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l348_348746


namespace number_of_girls_with_no_pet_l348_348287

-- Definitions based on the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def percentage_girls_own_dogs : ℚ := 40 / 100
def percentage_girls_own_cats : ℚ := 20 / 100

-- Prove that the number of girls with no pets is 8
theorem number_of_girls_with_no_pet :
  let girls := total_students * (1 - fraction_boys),
      percentage_girls_no_pets := 1 - percentage_girls_own_dogs - percentage_girls_own_cats,
      girls_with_no_pets := girls * percentage_girls_no_pets
  in girls_with_no_pets = 8 := by
{
  sorry
}

end number_of_girls_with_no_pet_l348_348287


namespace reduce_to_one_l348_348954

theorem reduce_to_one (n : ℕ) (h : n > 0) : 
  ∃ N : ℕ, (∀ m, m < N → (∀ k, (m + k) > 0 → (k <= N) → (k > 0 → if k % 2 = 0 then (k / 2) else ((k + 1) / 2) < k))) → (n = 1) :=
  sorry

end reduce_to_one_l348_348954


namespace infinite_sum_converges_l348_348560

theorem infinite_sum_converges :
  ∑' n : ℕ, (n^2 : ℝ) / (n^6 + 5) = 1 / 10 :=
begin
  sorry
end

end infinite_sum_converges_l348_348560


namespace different_books_l348_348044

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l348_348044


namespace bob_remaining_money_l348_348512

def initial_amount : ℕ := 80
def monday_spent (initial : ℕ) : ℕ := initial / 2
def tuesday_spent (remaining_monday : ℕ) : ℕ := remaining_monday / 5
def wednesday_spent (remaining_tuesday : ℕ) : ℕ := remaining_tuesday * 3 / 8

theorem bob_remaining_money : 
  let remaining_monday := initial_amount - monday_spent initial_amount
  let remaining_tuesday := remaining_monday - tuesday_spent remaining_monday
  let final_remaining := remaining_tuesday - wednesday_spent remaining_tuesday
  in final_remaining = 20 := 
by
  -- Proof goes here
  sorry

end bob_remaining_money_l348_348512


namespace solve_absolute_value_equation_l348_348766

theorem solve_absolute_value_equation (x : ℝ) :
  |2 * x - 3| = x + 1 → (x = 4 ∨ x = 2 / 3) := by
  sorry

end solve_absolute_value_equation_l348_348766


namespace ratio_seniors_to_children_l348_348086

-- Definitions for the conditions
def numAdults : ℕ := 58
def numChildren : ℕ := numAdults - 35
def totalGuests : ℕ := 127
def numSeniors : ℕ := totalGuests - numAdults - numChildren

-- The proof goal: ratio of seniors to children is 2:1
theorem ratio_seniors_to_children : numSeniors / numChildren = 2 := by
  -- We start from the conditions
  have children_val : numChildren = 23 := by
    unfold numChildren
    rw [numAdults]
    norm_num
  have seniors_val : numSeniors = 46 := by
    unfold numSeniors
    rw [totalGuests, numAdults]
    norm_num
  -- Now apply the values to the ratio
  rw [children_val, seniors_val]
  norm_num
  done

end ratio_seniors_to_children_l348_348086


namespace range_of_a_l348_348268

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_f : ∀ x, f x = x^3 - 3*x + a) 
  (has_3_distinct_roots : ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) : 
  -2 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l348_348268


namespace magician_trick_l348_348347

noncomputable def largest_arc {α : Type*} [linear_order α] (points : finset α) : α × α :=
sorry -- This function finds the endpoints of the largest arc formed by the given points

theorem magician_trick
  (points : finset ℝ)
  (h_points_card : points.card = 2007)
  (erase_point : ℝ)
  (h_erase_point : erase_point ∈ points)
  (remaining_points := points.erase erase_point)
  (h_remaining_card : remaining_points.card = 2006)
  (largest_pre_erase := largest_arc points)
  (largest_post_erase := largest_arc remaining_points):
  ∃ (semicircle : set ℝ), erase_point ∈ semicircle ∧ ∀ x ∈ largest_post_erase, x ∈ semicircle :=
begin
  sorry -- This proof shows that the magician can determine the semicircle containing the erased point
end

end magician_trick_l348_348347


namespace pow_mod_eq_l348_348837

theorem pow_mod_eq : (6 ^ 2040) % 50 = 26 := by
  sorry

end pow_mod_eq_l348_348837


namespace johnny_payment_correct_l348_348730

variable (cost_per_ball : ℝ) (num_balls : ℕ) (discount_rate : ℝ)
#check cost_per_ball
#check num_balls
#check discount_rate
open Lean.Parser.Tactic
noncomputable def johnny_payment := 
  sorry




variable h1 : cost_per_ball = 0.10
variable h2 : num_balls = 10000
variable h3 : discount_rate = 0.30

theorem johnny_payment_correct : 
  let initial_cost := num_balls * cost_per_ball in
  let discount := initial_cost * discount_rate in
  let final_amount := initial_cost - discount in
  final_amount = 700 := by
  sorry

end johnny_payment_correct_l348_348730


namespace selene_total_payment_l348_348360

-- Definitions based on the conditions given:

def cost_instant_cameras : ℝ := 2 * 110
def cost_digital_frames : ℝ := 3 * 120
def cost_memory_cards : ℝ := 4 * 30

def discount_instant_cameras : ℝ := 0.07
def discount_digital_frames : ℝ := 0.05
def discount_memory_cards : ℝ := 0.10

def sales_tax_cameras_and_frames : ℝ := 0.06
def sales_tax_memory_cards : ℝ := 0.04

-- Initial cost calculations
def initial_total_cost : ℝ := cost_instant_cameras + cost_digital_frames + cost_memory_cards

-- Discounts calculations
def discount_amount_instant_cameras : ℝ := cost_instant_cameras * discount_instant_cameras
def discount_amount_digital_frames : ℝ := cost_digital_frames * discount_digital_frames
def discount_amount_memory_cards : ℝ := cost_memory_cards * discount_memory_cards

-- Total after discount
def total_after_discount_instant_cameras : ℝ := cost_instant_cameras - discount_amount_instant_cameras
def total_after_discount_digital_frames : ℝ := cost_digital_frames - discount_amount_digital_frames
def total_after_discount_memory_cards : ℝ := cost_memory_cards - discount_amount_memory_cards

-- Sales tax calculations after discount
def sales_tax_total_cameras_and_frames : ℝ := (total_after_discount_instant_cameras + total_after_discount_digital_frames) * sales_tax_cameras_and_frames
def sales_tax_total_memory_cards : ℝ := total_after_discount_memory_cards * sales_tax_memory_cards

-- Total after tax
def total_with_tax_cameras_and_frames : ℝ := (total_after_discount_instant_cameras + total_after_discount_digital_frames) + sales_tax_total_cameras_and_frames
def total_with_tax_memory_cards : ℝ := total_after_discount_memory_cards + sales_tax_total_memory_cards

-- Overall total cost
def overall_total_cost : ℝ := total_with_tax_cameras_and_frames + total_with_tax_memory_cards

-- Rounding to nearest cent
def rounded_overall_cost : ℝ := Real.round (overall_total_cost * 100) / 100

theorem selene_total_payment : rounded_overall_cost = 691.72 :=
by sorry

end selene_total_payment_l348_348360


namespace find_b_for_continuous_l348_348322

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4*x^2 + 5 else b*x + 3

theorem find_b_for_continuous : 
  ∃ b : ℝ, (∀ x : ℝ, f x b = (if x ≤ 2 then 4*x^2 + 5 else b*x + 3)) ∧
           (∀ x : ℝ, f x b ≤ 2 → f x b = 21) ∧
           (∀ x : ℝ, 2 < f x b → f 2 b = 21) ∧
           (∀ x y : ℝ, f 2 b = 21 → 2*b + 3 = 21 → b = 9) :=
begin
  use 9,
  split,
  { intro x,
    simp [f],
    split_ifs,
    { refl, },
    { refl, }},
  split,
  { intros x hx,
    simp [f],
    split_ifs;
    { linarith }},
  split,
  { intros x hx,
    simp [f],
    split_ifs;
    { linarith }},
  { intros x y h₁ h₂,
    exact eq_of_sub_eq_zero (sub_eq_zero.mpr h₂) }
end

end find_b_for_continuous_l348_348322


namespace Jordan_length_is_8_l348_348132

-- Definitions of the conditions given in the problem
def Carol_length := 5
def Carol_width := 24
def Jordan_width := 15

-- Definition to calculate the area of Carol's rectangle
def Carol_area : ℕ := Carol_length * Carol_width

-- Definition to calculate the length of Jordan's rectangle
def Jordan_length (area : ℕ) (width : ℕ) : ℕ := area / width

-- Proposition to prove the length of Jordan's rectangle
theorem Jordan_length_is_8 : Jordan_length Carol_area Jordan_width = 8 :=
by
  -- skipping the proof
  sorry

end Jordan_length_is_8_l348_348132


namespace financial_calculations_correct_l348_348522

noncomputable def revenue : ℝ := 2500000
noncomputable def expenses : ℝ := 1576250
noncomputable def loan_payment_per_month : ℝ := 25000
noncomputable def number_of_shares : ℕ := 1600
noncomputable def ceo_share_percentage : ℝ := 0.35

theorem financial_calculations_correct :
  let net_profit := (revenue - expenses) - (revenue - expenses) * 0.2 in
  let total_loan_payment := loan_payment_per_month * 12 in
  let dividends_per_share := (net_profit - total_loan_payment) / number_of_shares in
  let ceo_dividends := dividends_per_share * ceo_share_percentage * number_of_shares in
  net_profit = 739000 ∧
  total_loan_payment = 300000 ∧
  dividends_per_share = 274 ∧
  ceo_dividends = 153440 :=
begin
  sorry
end

end financial_calculations_correct_l348_348522


namespace trains_crossing_time_l348_348813

theorem trains_crossing_time :
  let L1 := 500 -- length of the first train in meters
  let L2 := 600 -- length of the second train in meters
  let S1 := 54 * 1000 / 3600 -- speed of the first train in meters per second
  let S2 := 27 * 1000 / 3600 -- speed of the second train in meters per second
  let relative_speed := S1 + S2 -- relative speed because trains are in opposite directions
  let total_distance := L1 + L2 -- total distance to be covered
  let T := total_distance / relative_speed -- time to cross each other
  T ≈ 48.89 := 
by
  let L1 := 500
  let L2 := 600
  let S1 := 54 * 1000 / 3600
  let S2 := 27 * 1000 / 3600
  let relative_speed := S1 + S2
  let total_distance := L1 + L2
  let T := total_distance / relative_speed
  have T_val : T = 1100 / 22.5 := rfl
  have approx_T : 1100 / 22.5 ≈ 48.89 := 
  sorry
  exact approx_T

end trains_crossing_time_l348_348813


namespace problem_statement_l348_348260

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 :=
begin
  sorry
end

end problem_statement_l348_348260


namespace first_three_workers_dig_time_l348_348178

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l348_348178


namespace perimeter_shaded_area_is_942_l348_348124

-- Definition involving the perimeter of the shaded area of the circles
noncomputable def perimeter_shaded_area (s : ℝ) : ℝ := 
  4 * 75 * 3.14

-- Main theorem stating that if the side length of the octagon is 100 cm,
-- then the perimeter of the shaded area is 942 cm.
theorem perimeter_shaded_area_is_942 :
  perimeter_shaded_area 100 = 942 := 
  sorry

end perimeter_shaded_area_is_942_l348_348124


namespace ribbon_lengths_after_cut_l348_348825

theorem ribbon_lengths_after_cut {
  l1 l2 l3 l4 l5 l1' l2' l3' l4' l5' : ℕ
  (initial_lengths : multiset ℕ)
  (new_lengths : multiset ℕ)
  (hl : initial_lengths = {15, 20, 24, 26, 30})
  (hl' : new_lengths = {l1', l2', l3', l4', l5'})
  (h_average_decrease : (∑ x in initial_lengths, x) / 5 - 5 = (∑ x in new_lengths, x) / 5)
  (h_median : ∀ x ∈ new_lengths, x = 24 ∨ 24 ∈ new_lengths)
  (h_range : multiset.range new_lengths = 15)
  (h_lengths : l1' ≤ l2' ≤ l3' = 24 ∧ l3' ≤ l4' ≤ l5') :
  new_lengths = {9, 9, 24, 24, 24} := sorry

end ribbon_lengths_after_cut_l348_348825


namespace each_person_gets_9_apples_l348_348316

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end each_person_gets_9_apples_l348_348316


namespace determine_statistical_measures_l348_348883

theorem determine_statistical_measures 
  (total_students : ℕ)
  (age_13_count : ℕ)
  (age_14_count : ℕ)
  (unknown_age_count : ℕ)
  (h_total : total_students = 50)
  (h_age_13 : age_13_count = 5)
  (h_age_14 : age_14_count = 23)
  (h_unknown : unknown_age_count = total_students - age_13_count - age_14_count) :
  (median_age : ℕ) × (mode_age : ℕ) :=
by
  have h : 5 + 23 ≤ total_students := by linarith,
  have median_age := 14,
  have mode_age := 14,
  exact ⟨median_age, mode_age⟩

end determine_statistical_measures_l348_348883


namespace trapezoid_is_isosceles_l348_348409

variable (A B C D P Q : Type)
variable (PQ_parallel_AD : ℝ)
variable (is_angle_bisector : (A → A → P → P → P → P → Prop))
variable (is_parallel : P → Q → Prop)

theorem trapezoid_is_isosceles
  (h1 : is_angle_bisector A C P P)
  (h2 : is_angle_bisector B D Q Q)
  (h3 : PQ_parallel_AD) :
  is_parallel A D :=
sorry

end trapezoid_is_isosceles_l348_348409


namespace book_costs_l348_348075

theorem book_costs (C1 C2 : ℝ) (h1 : C1 + C2 = 450) (h2 : 0.85 * C1 = 1.19 * C2) : C1 = 262.5 := 
sorry

end book_costs_l348_348075


namespace probability_of_rolling_five_l348_348106

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end probability_of_rolling_five_l348_348106


namespace angle_between_l348_348239

open Real

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

def c : ℝ × ℝ := (3, -1) -- calculate 2a - b

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cos_theta : ℝ :=
  dot_product c a / (magnitude c * magnitude a)

theorem angle_between : acos cos_theta = π / 4 := by
  sorry

end angle_between_l348_348239


namespace bus_capacity_percentage_l348_348812

theorem bus_capacity_percentage (x : ℕ) (h1 : 150 * x / 100 + 150 * 70 / 100 = 195) : x = 60 :=
by
  sorry

end bus_capacity_percentage_l348_348812


namespace problem_inequality_l348_348772

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_pos : ∀ x : ℝ, x > 0 → f x > 0

axiom f_increasing : ∀ x y : ℝ, x > 0 → y > 0 → x ≤ y → (f x / x) ≤ (f y / y)

theorem problem_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
    3 * (f a + f b + f c) / (a + b + c) + (f a / a + f b / b + f c / c) :=
sorry

end problem_inequality_l348_348772


namespace conic_section_is_parabola_l348_348845

theorem conic_section_is_parabola (x y : ℝ) :
  abs (y - 3) = sqrt ((x + 4)^2 + y^2) → (∃ a b c A B : ℝ, (a * x^2 + b * x + c = A * y + B) ∧ (a ≠ 0 ∧ A ≠ 0)) :=
by
  sorry

end conic_section_is_parabola_l348_348845


namespace solution_set_of_x_abs_x_lt_x_l348_348406

theorem solution_set_of_x_abs_x_lt_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} :=
by
  sorry

end solution_set_of_x_abs_x_lt_x_l348_348406


namespace sum_of_number_and_reverse_l348_348001

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l348_348001


namespace set_relationship_l348_348975

def set_M : Set ℚ := {x : ℚ | ∃ m : ℤ, x = m + 1/6}
def set_N : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n/2 - 1/3}
def set_P : Set ℚ := {x : ℚ | ∃ p : ℤ, x = p/2 + 1/6}

theorem set_relationship : set_M ⊆ set_N ∧ set_N = set_P := by
  sorry

end set_relationship_l348_348975


namespace mouse_cannot_eat_all_cubes_l348_348892

theorem mouse_cannot_eat_all_cubes :
  ∀ (cheese : ℕ → ℕ → ℕ → bool),   -- cheese layout
  (∀ x y z, 0 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 3 ∧ 0 ≤ z ∧ z < 3 → cheese x y z) →      -- cheese dimensions
  (cheese 1 1 1 = false) →  -- central cube removed
  (∀ x y z, (cheese x y z = true) → (∃ dx dy dz, abs dx + abs dy + abs dz = 1 ∧ cheese (x + dx) (y + dy) (z + dz) = true)) →  -- mouse movement rule
  false :=   -- impossible for the mouse to eat the whole cheese
sorry

end mouse_cannot_eat_all_cubes_l348_348892


namespace ribbon_lengths_after_cut_l348_348824

theorem ribbon_lengths_after_cut {
  l1 l2 l3 l4 l5 l1' l2' l3' l4' l5' : ℕ
  (initial_lengths : multiset ℕ)
  (new_lengths : multiset ℕ)
  (hl : initial_lengths = {15, 20, 24, 26, 30})
  (hl' : new_lengths = {l1', l2', l3', l4', l5'})
  (h_average_decrease : (∑ x in initial_lengths, x) / 5 - 5 = (∑ x in new_lengths, x) / 5)
  (h_median : ∀ x ∈ new_lengths, x = 24 ∨ 24 ∈ new_lengths)
  (h_range : multiset.range new_lengths = 15)
  (h_lengths : l1' ≤ l2' ≤ l3' = 24 ∧ l3' ≤ l4' ≤ l5') :
  new_lengths = {9, 9, 24, 24, 24} := sorry

end ribbon_lengths_after_cut_l348_348824


namespace ratio_of_areas_l348_348494

theorem ratio_of_areas 
  (s1 s2 : ℝ)
  (A_large A_small A_trapezoid : ℝ)
  (h1 : s1 = 10)
  (h2 : s2 = 5)
  (h3 : A_large = (sqrt 3 / 4) * s1^2)
  (h4 : A_small = (sqrt 3 / 4) * s2^2)
  (h5 : A_trapezoid = A_large - A_small) :
  (A_small / A_trapezoid = 1 / 3) :=
sorry

end ratio_of_areas_l348_348494


namespace solution_of_system_l348_348754

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l348_348754


namespace mul_inv_301_mod_401_l348_348135

theorem mul_inv_301_mod_401 : ∃ (b : ℕ), 0 ≤ b ∧ b ≤ 400 ∧ (301 * b ≡ 1 [MOD 401]) :=
by
  let b := 397
  use b
  split
  · show 0 ≤ b, exact dec_trivial
  split
  · show b ≤ 400, exact dec_trivial
  · show 301 * b ≡ 1 [MOD 401], exact dec_trivial
  sorry

end mul_inv_301_mod_401_l348_348135


namespace dividends_CEO_2018_l348_348524

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l348_348524


namespace cosine_difference_formula_l348_348607

theorem cosine_difference_formula
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < (Real.pi / 2))
  (h3 : Real.tan α = 2) :
  Real.cos (α - (Real.pi / 4)) = (3 * Real.sqrt 10) / 10 := 
by
  sorry

end cosine_difference_formula_l348_348607


namespace definite_integral_eq_zero_l348_348148

noncomputable def f : ℝ → ℝ :=
  λ x, x^3 + tan x + x^2 * sin x

theorem definite_integral_eq_zero :
  ∫ x in -1..1, f x = 0 :=
by
  sorry

end definite_integral_eq_zero_l348_348148


namespace infinite_primes_divide_sequence_exists_prime_not_divide_sequence_l348_348859

open Nat

theorem infinite_primes_divide_sequence (a : ℕ) (h_a_pos : 0 < a) :
  ∃∞ p, Prime p ∧ ∃ n, e_n a n % p = 0 :=
by sorry

theorem exists_prime_not_divide_sequence (a : ℕ) (h_a_pos : 0 < a) :
  ∃ p, Prime p ∧ ∀ n, e_n a n % p ≠ 0 :=
by sorry

noncomputable def e_n (a : ℕ) : ℕ → ℕ
| 0       := 1
| (n + 1) := a + (List.prod (List.map (λ k => e_n a k) (List.range (n + 1))))

end infinite_primes_divide_sequence_exists_prime_not_divide_sequence_l348_348859


namespace perpendicular_lines_l348_348963

variables {L : Type} [LinearOrder L]
variables (a b : L) (alpha beta : Set L)

def parallel (α β : Set L) := ∀ x ∈ α, ∀ y ∈ β, x ∥ y
def perpendicular (α β : Set L) := ∀ x ∈ α, ∀ y ∈ β, x ⊥ y

theorem perpendicular_lines 
  (ha_alpha : a ∈ alpha)
  (hb_beta : b ∈ beta)
  (a_perp_alpha : perpendicular {a} alpha)
  (b_perp_beta : perpendicular {b} beta)
  (alpha_perp_beta : perpendicular alpha beta) :
  perpendicular {a} {b} :=
sorry

end perpendicular_lines_l348_348963


namespace probability_each_bin_has_one_idea_l348_348838

theorem probability_each_bin_has_one_idea
  (p : ℝ) (hp : 0.5 < p) :
  (1 - 2 * p^5 + p^10) = (1 - (p^5)^2 - (1 - p^5)^2 + (1 - p^5 + p^5)^2) :=
begin
  sorry
end

end probability_each_bin_has_one_idea_l348_348838


namespace find_ellipse_eq_area_triangle_MNA_l348_348592

-- Define the ellipse and given conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def condition_ellipse (a b c : ℝ) (x y : ℝ) :=
  a > b ∧ b > 0 ∧ (b / c = Real.sqrt 3) ∧ (a^2 = b^2 + c^2)

noncomputable def point_D : (ℝ × ℝ) := (1, 3 / 2)
def condition_D (a b : ℝ) : Prop := ellipse a b (point_D.1) (point_D.2)

-- Prove the equation of the ellipse
theorem find_ellipse_eq :
  ∃ a b c : ℝ, condition_ellipse a b c 1 (3 / 2) ∧ condition_D a b ∧ ellipse 2 (Real.sqrt 3) 1 (3 / 2) :=
by
  sorry

-- Define the area of a triangle formed under specific conditions
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

noncomputable def point_A : (ℝ × ℝ) := (0, Real.sqrt 3)
noncomputable def point_F1 : (ℝ × ℝ) := (-1, 0)

def slope_AF1 : ℝ := Real.sqrt 3
def slope_MN : ℝ := -1 / (Real.sqrt 3)
def line_MN_through_F1 (x y : ℝ) : Prop := y = slope_MN * (x + 1)

theorem area_triangle_MNA :
  ∃ (points_MN : ((ℝ × ℝ) × (ℝ × ℝ))),
    (line_MN_through_F1 (points_MN.1.1) (points_MN.1.2) ∧ line_MN_through_F1 (points_MN.2.1) (points_MN.2.2) ∧ 
    (1 + 1/3) * ((points_MN.1.1 + points_MN.2.1)^2 + 4 * 32 / 13) = 48 / 13 ∧ 
    (Real.sqrt (1^2 + (Real.sqrt 3)^2) = 2) ∧ 
     (1/2 * 2 * 48 / 13 = 48 / 13)) :=
by
  sorry

end find_ellipse_eq_area_triangle_MNA_l348_348592


namespace sum_median_mode_eq_l348_348795

open Classical
noncomputable theory

variable {A B : List ℝ} 

def median (l : List ℝ) : ℝ := 
  if l = [] then 0 
  else if l.length % 2 = 1 then l.nthLe (l.length / 2) sorry 
  else (l.nthLe (l.length / 2 - 1) sorry + l.nthLe (l.length / 2) sorry) / 2

def mode (l : List ℝ) : ℝ :=
  l.foldr
    (λ x acc, if count x l > count acc l then x else acc)
    0

theorem sum_median_mode_eq (medA modeB : ℝ) (h : medA + modeB = 51) : 
  ∃ A B : List ℝ, median A = medA ∧ mode B = modeB :=
by 
  refine ⟨A, B, _, _⟩
  sorry

end sum_median_mode_eq_l348_348795


namespace tax_raise_expectation_l348_348856

noncomputable section

variables 
  (x y : ℝ) -- x: fraction of liars, y: fraction of economists
  (p1 p2 p3 p4 : ℝ) -- percentages of affirmative answers
  (expected_taxes : ℝ) -- expected fraction for taxes

-- Given conditions
def given_conditions (x y p1 p2 p3 p4 : ℝ) :=
  p1 = 0.4 ∧ p2 = 0.3 ∧ p3 = 0.5 ∧ p4 = 0.0 ∧
  y = 1 - x ∧ -- fraction of economists
  3 * x + y = 1.2 -- sum of affirmative answers

-- The statement to prove
theorem tax_raise_expectation (x y p1 p2 p3 p4 : ℝ) : 
  given_conditions x y p1 p2 p3 p4 →
  expected_taxes = 0.4 - x →
  expected_taxes = 0.3 :=
begin
  intro h, intro h_exp,
  sorry -- proof to be filled in
end

end tax_raise_expectation_l348_348856


namespace YZ_squared_eq_YK_YB_l348_348480

-- Define the triangle and points structure
structure TrianglePoint (α : Type) :=
(A B C X Y Z K : α) 
(Y_on_CA : Y ∈ line A C)
(Z_on_AB : Z ∈ line A B)
(X_on_BC : X ∈ line B C)

-- Conditions: equilateral triangles
axiom equilateral_AYZ (α : Type) [inner_product_space ℝ α] (t : TrianglePoint α) : 
  is_equilateral_triangle α t.A t.Y t.Z

axiom equilateral_XYZ (α : Type) [inner_product_space ℝ α] (t : TrianglePoint α) : 
  is_equilateral_triangle α t.X t.Y t.Z

-- Condition: Intersection point K of BY and CZ
axiom intersection_K (α : Type) [inner_product_space ℝ α] (t : TrianglePoint α) :
  ∃ K, K ∈ line t.B t.Y ∧ K ∈ line t.C t.Z

-- Goal: Prove the required identity
theorem YZ_squared_eq_YK_YB (α : Type) [inner_product_space ℝ α] (t : TrianglePoint α) :
  ∃ K, intersection_K α t → 
  ∥t.Y - t.Z∥^2 = ∥t.Y - K∥ * ∥t.Y - t.B∥ :=
by sorry

end YZ_squared_eq_YK_YB_l348_348480


namespace common_difference_is_three_l348_348654

noncomputable theory

def arithmetic_sequence_common_difference (a a_n S_n : ℕ) (hₐ : a = 2) (hₐₙ : a_n = 50) (hₛₙ : S_n = 442) : ℕ :=
  by
  sorry

theorem common_difference_is_three : arithmetic_sequence_common_difference 2 50 442 rfl rfl rfl = 3 :=
  by
  sorry

end common_difference_is_three_l348_348654


namespace not_divisible_67_l348_348725

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l348_348725


namespace perpendicular_vectors_l348_348049

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l348_348049


namespace sector_area_l348_348984

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end sector_area_l348_348984


namespace num_teams_of_4_l348_348242

theorem num_teams_of_4 (n k : ℕ) (h₁ : n = 6) (h₂ : k = 4) : nat.choose n k = 15 := by
  rw [h₁, h₂]
  norm_num
  sorry

end num_teams_of_4_l348_348242


namespace diamond_value_l348_348552

-- Define the diamond operation
def diamond (a b : ℝ) := a^2 + a^2 / b

-- State and prove the main theorem (proof will be omitted with sorry)
theorem diamond_value : diamond 5 2 = 37.5 :=
by
-- Proof is omitted
sorry

end diamond_value_l348_348552


namespace solution_set_line_l348_348732

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l348_348732


namespace problem_EF_fraction_of_GH_l348_348357

theorem problem_EF_fraction_of_GH (E F G H : Type) 
  (GE EH GH GF FH EF : ℝ) 
  (h1 : GE = 3 * EH) 
  (h2 : GF = 8 * FH)
  (h3 : GH = GE + EH)
  (h4 : GH = GF + FH) : 
  EF = 5 / 36 * GH :=
by
  sorry

end problem_EF_fraction_of_GH_l348_348357


namespace new_ribbon_lengths_correct_l348_348821

noncomputable def ribbon_lengths := [15, 20, 24, 26, 30]
noncomputable def new_average_change := 5
noncomputable def new_lengths := [9, 9, 24, 24, 24]

theorem new_ribbon_lengths_correct :
  let new_length_list := [9, 9, 24, 24, 24]
  ribbon_lengths.length = 5 ∧ -- we have 5 ribbons
  new_length_list.length = 5 ∧ -- the new list also has 5 ribbons
  list.average new_length_list = list.average ribbon_lengths - new_average_change ∧ -- new average decreased by 5
  list.median new_length_list = list.median ribbon_lengths ∧ -- median unchanged
  list.range new_length_list = list.range ribbon_lengths -- range unchanged
  :=
by {
  sorry
}

end new_ribbon_lengths_correct_l348_348821


namespace num_tangent_lines_with_equal_intercepts_to_circle_is_four_l348_348016

-- Definitions
def circle : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 + (y - 2)^2 = 2 }

def is_tangent (L : Set (ℝ × ℝ)) : Prop :=
  ∃ p ∈ circle, ∃ v ∈ L, p = v ∧ ∀ q ∈ L, q ≠ p → (p.1 - q.1) * (v.1 - q.1) + (p.2 - q.2) * (v.2 - q.2) = 0

def has_equal_intercepts (L : Set (ℝ × ℝ)) : Prop :=
  ∃ a : ℝ, L = { p : ℝ × ℝ | let (x, y) := p in x + y = a } ∨ L = { p : ℝ × ℝ | let (x, y) := p in y = x * a }

-- Main statement
theorem num_tangent_lines_with_equal_intercepts_to_circle_is_four :
  ∃ Ls : Set (Set (ℝ × ℝ)), (∀ L ∈ Ls, is_tangent L ∧ has_equal_intercepts L) ∧ Ls.size = 4 :=
sorry

end num_tangent_lines_with_equal_intercepts_to_circle_is_four_l348_348016


namespace smallest_possible_sum_of_digits_l348_348689

-- Define the problem in Lean
noncomputable def smallest_sum_of_digits : ℕ :=
  @classical.some ℕ (∃ P x y : ℕ, 
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    ∀ d ∈ [x / 10, x % 10, y / 10, y % 10], d ∈ finset.Icc 0 9 ∧
    finset.card (finset {x / 10, x % 10, y / 10, y % 10}) = 4 ∧
    P = x * y ∧
    1000 ≤ P ∧ P < 10000 ∧
    finset.card (finset P.digits) = 4 ∧
    (∃ S, S = P.digits.sum ∧ P.digits.sum = 12))

theorem smallest_possible_sum_of_digits :
  smallest_sum_of_digits = 12 :=
by sorry

end smallest_possible_sum_of_digits_l348_348689


namespace television_advertiser_loss_l348_348797

theorem television_advertiser_loss :
  let in_store_price := 10999 -- price in cents
  let payment_per_installment := 2499 -- cost per installment in cents
  let shipping_and_handling := 1498 -- shipping and handling cost in cents
  let total_television_price := 4 * payment_per_installment + shipping_and_handling in
  total_television_price - in_store_price = 495 := 
by
  sorry

end television_advertiser_loss_l348_348797


namespace find_pairs_l348_348935

-- Definitions and conditions from the problem
def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pairs (a b : ℕ) : 
  (∃ a b : ℕ, 
    (is_divisible (a^2 + 4*a + 3) b ↔ (a, b) = (6, 1) ∨ (a, b) = (18, 7)) ∧
    (a^2 + ab - 6*b^2 - 2*a - 16*b - 8 = 0) ∧
    (a + 2*b + 1 ≡ 0 [MOD 4] ↔ (a, b) = (6, 1) ∨ (a, b) = (18, 7)) ∧
    (is_prime (a + 6*b + 1) ↔ (a, b) = (6, 1) ∨ (a, b) = (18, 7)))
    ↔ (a, b) = (6, 1) ∨ (a, b) = (18, 7) :=
begin
  sorry
end

end find_pairs_l348_348935


namespace tires_in_parking_lot_l348_348275

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l348_348275


namespace min_value_f_l348_348990

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * sin ((1 / 2) * x + (π / 4)) - 1

theorem min_value_f (x : ℝ) :
  ∃ k : ℤ, x = 4 * k * π - (3 * π / 2) ∧ f x = -4 :=
sorry

end min_value_f_l348_348990


namespace find_y_l348_348563

theorem find_y (y : ℝ) (h : log y 8 = log 64 4 + 1) : y = 2^(9/4) :=
sorry

end find_y_l348_348563


namespace fastest_hike_is_faster_by_one_hour_l348_348681

def first_trail_time (d₁ s₁ : ℕ) : ℕ :=
  d₁ / s₁

def second_trail_time (d₂ s₂ break_time : ℕ) : ℕ :=
  (d₂ / s₂) + break_time

theorem fastest_hike_is_faster_by_one_hour 
(h₁ : first_trail_time 20 5 = 4)
(h₂ : second_trail_time 12 3 1 = 5) :
  5 - 4 = 1 :=
by
  exact Nat.sub_self 4.symm
  sorry

end fastest_hike_is_faster_by_one_hour_l348_348681


namespace field_area_l348_348482

-- Define the given conditions and prove the area of the field
theorem field_area (x y : ℕ) 
  (h1 : 2*(x + 20) + 2*y = 2*(2*x + 2*y))
  (h2 : 2*x + 2*(2*y) = 2*x + 2*y + 18) : x * y = 99 := by 
{
  sorry
}

end field_area_l348_348482


namespace find_asking_price_l348_348133

variable {P : ℝ} -- Define asking price P as a real number

def earnings_buyer1 (P : ℝ) : ℝ := P - (P / 10)
def earnings_buyer2 (P : ℝ) : ℝ := P - (80 + 3 * 80)

theorem find_asking_price (h : earnings_buyer1 P - earnings_buyer2 P = 200) : P = 1200 :=
by
  simp [earnings_buyer1, earnings_buyer2] at h
  sorry

end find_asking_price_l348_348133


namespace probability_red_odd_blue_special_l348_348048

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def successful_outcomes_red := {1, 3, 5, 7}
def successful_outcomes_blue := {1, 2, 3, 4, 5, 7, 9}

noncomputable def probability (successful : Nat) (total : Nat) : ℚ :=
  (successful : ℚ) / (total : ℚ)

theorem probability_red_odd_blue_special :
  probability (successful_outcomes_red.card * successful_outcomes_blue.card) (8 * 10) = 7 / 20 := by
  sorry

end probability_red_odd_blue_special_l348_348048


namespace third_smallest_four_digit_number_in_pascals_triangle_l348_348063

def pascal_triangle : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 1
| n, k := if h : k ≤ n then pascal_triangle (n - 1) (k - 1) + pascal_triangle (n - 1) k else 0

theorem third_smallest_four_digit_number_in_pascals_triangle :
  ∃ (n k : ℕ), pascal_triangle n k = 1002 ∧ n = 1001 ∧ k = 3 :=
begin
  sorry
end

end third_smallest_four_digit_number_in_pascals_triangle_l348_348063


namespace complex_abs_prod_conj_l348_348934

noncomputable def complex_abs_product : ℂ := 3 - 2 * complex.I
noncomputable def complex_abs_conjugate : ℂ := 3 + 2 * complex.I

theorem complex_abs_prod_conj : |complex_abs_product| * |complex_abs_conjugate| = 13 := by
  sorry

end complex_abs_prod_conj_l348_348934


namespace variance_changes_when_adding_point_l348_348586

theorem variance_changes_when_adding_point (D : List ℕ) 
  (hD : D = [1, 3, 3, 5]) (new_point : ℕ) (h_new_point : new_point = 3) :
  let new_data := D ++ [new_point] in
  variance new_data ≠ variance D :=
by
  sorry

end variance_changes_when_adding_point_l348_348586


namespace arccos_sin_three_pi_over_two_eq_pi_l348_348134

theorem arccos_sin_three_pi_over_two_eq_pi : 
  Real.arccos (Real.sin (3 * Real.pi / 2)) = Real.pi :=
by
  sorry

end arccos_sin_three_pi_over_two_eq_pi_l348_348134


namespace joe_marshmallow_ratio_l348_348686

theorem joe_marshmallow_ratio (J : ℕ) (h1 : 21 / 3 = 7) (h2 : 1 / 2 * J = 49 - 7) : J / 21 = 4 :=
by
  sorry

end joe_marshmallow_ratio_l348_348686


namespace polynomial_coefficient_sum_l348_348447

/-- Given the polynomial expansion of (x^2+1)(2x+1)^9, prove that the sum of the 
coefficients a_0, a_1, ..., a_11 in the expansion in powers of (x + 2) is -2. -/
theorem polynomial_coefficient_sum : 
  (∃ (a : Fin 12 → ℝ), (λ x : ℝ, (x^2 + 1) * (2 * x + 1)^9) = (λ x : ℝ, ∑ i in Finset.range 12, a i * (x + 2) ^ i)) →
  (∑ i in Finset.range 12, a i) = -2 :=
by
  intro h_exists_a
  cases h_exists_a with a ha
  have h_sum_coeff := ha (-1)
  simp at h_sum_coeff
  exact h_sum_coeff

end polynomial_coefficient_sum_l348_348447


namespace geometric_sequence_property_l348_348665

variable {a_n : ℕ → ℝ}

theorem geometric_sequence_property (h1 : ∀ m n p q : ℕ, m + n = p + q → a_n m * a_n n = a_n p * a_n q)
    (h2 : a_n 4 * a_n 5 * a_n 6 = 27) : a_n 1 * a_n 9 = 9 := by
  sorry

end geometric_sequence_property_l348_348665


namespace find_r_l348_348158

theorem find_r (r : ℝ) (h : log 27 (3 * r - 2) = -1 / 3) : r = 7 / 9 :=
sorry

end find_r_l348_348158


namespace first_three_workers_dig_time_l348_348179

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l348_348179


namespace sum_of_fractions_approx_equals_1_832_l348_348911

noncomputable def sum_of_fractions_approx : ℝ :=
  ∑ n in finset.range 2023, (3 / (n + 1) / (n + 4))

theorem sum_of_fractions_approx_equals_1_832 :
  |sum_of_fractions_approx - 1.832| < 0.001 :=
sorry

end sum_of_fractions_approx_equals_1_832_l348_348911


namespace solution_set_l348_348740

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l348_348740


namespace find_f_of_2_l348_348971

-- Define the conditions of the problem
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def f (x : ℝ) : ℝ := if x < 0 then x^2 - 1 else 0

-- State the theorem
theorem find_f_of_2 (h1 : is_odd_function f) (h2 : ∀ x, x < 0 → f x = x^2 - 1) : f 2 = -3 :=
by
  sorry

end find_f_of_2_l348_348971


namespace geometric_seq_a5_l348_348024

theorem geometric_seq_a5 : ∃ (a₁ q : ℝ), 0 < q ∧ a₁ + 2 * a₁ * q = 4 ∧ (a₁ * q^3)^2 = 4 * (a₁ * q^2) * (a₁ * q^6) ∧ (a₅ = a₁ * q^4) := 
  by
    sorry

end geometric_seq_a5_l348_348024


namespace middle_number_of_consecutive_squares_l348_348854

theorem middle_number_of_consecutive_squares (x : ℕ ) (h : x^2 + (x+1)^2 + (x+2)^2 = 2030) : x + 1 = 26 :=
sorry

end middle_number_of_consecutive_squares_l348_348854


namespace first_three_workers_time_l348_348180

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l348_348180


namespace right_triangle_congruence_l348_348069

theorem right_triangle_congruence (A B C D : Prop) :
  (A → true) → (C → true) → (D → true) → (¬ B) → B :=
by
sorry

end right_triangle_congruence_l348_348069


namespace chinese_character_symmetry_l348_348668

-- Definitions of the characters and their symmetry properties
def is_symmetric (ch : String) : Prop :=
  ch = "喜"

-- Hypotheses (conditions)
def option_A := "喜"
def option_B := "欢"
def option_C := "数"
def option_D := "学"

-- Lean statement to prove the symmetry
theorem chinese_character_symmetry :
  is_symmetric option_A ∧ 
  ¬ is_symmetric option_B ∧ 
  ¬ is_symmetric option_C ∧ 
  ¬ is_symmetric option_D :=
by
  sorry

end chinese_character_symmetry_l348_348668


namespace area_quadrilateral_GHCD_l348_348672

theorem area_quadrilateral_GHCD
  (AB CD: ℝ)
  (altitude_ABCD: ℝ)
  (altitude_GHCD: ℝ)
  (GH CD: ℝ)
  (G H: Point)
  (midpoint_G: G.midpoint = AD)
  (midpoint_H: H.midpoint = BC)
  (length_AB: AB = 10)
  (length_CD: CD = 24)
  (altitude_ABCD_val: altitude_ABCD = 15)
  (altitude_GHCD_val: altitude_GHCD = altitude_ABCD / 2)
  (length_GH_val: GH = (AB + CD) / 2) :
  area GHCD = 153.75 := by
  sorry

end area_quadrilateral_GHCD_l348_348672


namespace distance_from_X_to_CD_l348_348676

theorem distance_from_X_to_CD (s : ℝ) (h : 0 < s) :
  let A := (0, 0)
  let B := (2 * s, 0)
  let C := (2 * s, 2 * s)
  let D := (0, 2 * s)
  let X := (s, s * Real.sqrt 3)
  let CD := 2 * s
  let distance := 2 * s - s * Real.sqrt 3
  (X.2 - CD.2) * s = distance := by
  sorry

end distance_from_X_to_CD_l348_348676


namespace evaluate_sum_l348_348930

theorem evaluate_sum : 
  let S := (1 / 4^1) + (2 / 4^2) + (3 / 4^3) + (4 / 4^4) + (5 / 4^5)
  in S = (4 / 3) * (1 - 1 / 4096) :=
by
  let S := (1 / 4^1) + (2 / 4^2) + (3 / 4^3) + (4 / 4^4) + (5 / 4^5)
  have : S = (4 / 3) * (1 - 1 / 4096) := sorry
  exact this

end evaluate_sum_l348_348930


namespace greatest_x_plus_z_l348_348215

theorem greatest_x_plus_z (x y z c d : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 1 ≤ z ∧ z ≤ 9)
  (h4 : 700 - c = 700)
  (h5 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 693)
  (h6 : x > z) :
  d = 11 :=
by
  sorry

end greatest_x_plus_z_l348_348215


namespace sum_of_sequence_l348_348924

noncomputable def sum_sequence (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (∑ k in Finset.range (i + 1), 10^k)

theorem sum_of_sequence (n : ℕ) : sum_sequence n = (10^(n + 1) - 10 - 9 * n) / 81 := by
  sorry

end sum_of_sequence_l348_348924


namespace apples_distribution_l348_348313

theorem apples_distribution (total_apples : ℕ) (given_to_father : ℕ) (total_people : ℕ) : total_apples = 55 → given_to_father = 10 → total_people = 5 → (total_apples - given_to_father) / total_people = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end apples_distribution_l348_348313


namespace count_zeros_in_quotient_R18_R6_l348_348136

def R_k (k : ℕ) : ℕ := (10^k - 1) / 9

/- Problem statement: Given conditions, we need to prove that the number of zeros in quotient Q is 10 -/
theorem count_zeros_in_quotient_R18_R6 :
  let R18 := R_k 18
  let R6 := R_k 6
  let Q := R18 / R6
  Q = 1 + 10^6 + 10^12 → 
  10 := count_zeros Q := 
  sorry

end count_zeros_in_quotient_R18_R6_l348_348136


namespace parabola_is_8x_l348_348577

noncomputable section

-- Define the conditions
constant p : ℝ
constant P : ℝ × ℝ
constant PF M : ℝ
constant focus : ℝ × ℝ
constant directrix_distance : ℝ

axiom parabola_eqn : (P.snd ^ 2 = 2 * p * P.fst)
axiom p_gt_zero : p > 0
axiom PF_eq_4 : |PF| = 4
axiom PM_eq_PF : PF = PF
axiom distance_M_to_PF : directrix_distance = 4
axiom P_coordinates : P = (2, 4)
constant parabola_eq : ℝ × ℝ → Prop

-- Define the parabola property
def parabola_property (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq (x, y)

-- State the proof goal
theorem parabola_is_8x : parabola_property p → parabola_eq P := 
begin
  sorry
end

end parabola_is_8x_l348_348577


namespace base9_is_decimal_l348_348141

-- Define the base-9 number and its decimal equivalent
def base9_to_decimal (n : Nat := 85) (base : Nat := 9) : Nat := 8 * base^1 + 5 * base^0

-- Theorem stating the proof problem
theorem base9_is_decimal : base9_to_decimal 85 9 = 77 := by
  unfold base9_to_decimal
  simp
  sorry

end base9_is_decimal_l348_348141


namespace negation_relation_l348_348600

open Real

-- Definitions of p and q
def p (x : ℝ) : Prop := x^2 - x < 1
def q (x : ℝ) : Prop := log 2 (x^2 - x) < 0

-- Theorem statement
theorem negation_relation (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x ∨ True) := sorry

end negation_relation_l348_348600


namespace number_of_triangles_l348_348578

theorem number_of_triangles (points : Finset ℕ) (h_points : points.card = 12) : 
  ∃ n, n = 200 ∧ 
  (∀ (a b c : ℕ), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → a ≠ c → 
  (¬ (collinear a b c)) → n = 200) :=
by {
  let n := (points.card.choose 3 - (number of collinear subsets of 3)),
  have h : n = 200,
  exact ⟨200, h, sorry⟩
}

end number_of_triangles_l348_348578


namespace net_profit_loan_payments_dividends_per_share_director_dividends_l348_348517

theorem net_profit (revenue expenses : ℕ) (tax_rate : ℚ) 
  (h_rev : revenue = 2500000)
  (h_exp : expenses = 1576250)
  (h_tax : tax_rate = 0.2) :
  ((revenue - expenses) - (revenue - expenses) * tax_rate).toNat = 739000 := by
  sorry

theorem loan_payments (monthly_payment : ℕ) 
  (h_monthly : monthly_payment = 25000) :
  (monthly_payment * 12) = 300000 := by
  sorry

theorem dividends_per_share (net_profit loan_payments : ℕ) (total_shares : ℕ)
  (h_net_profit : net_profit = 739000)
  (h_loan_payments : loan_payments = 300000)
  (h_shares : total_shares = 1600) :
  ((net_profit - loan_payments) / total_shares) = 274 := by
  sorry

theorem director_dividends (dividend_per_share : ℕ) (share_percentage : ℚ) (total_shares : ℕ)
  (h_dividend_per_share : dividend_per_share = 274)
  (h_percentage : share_percentage = 0.35)
  (h_shares : total_shares = 1600) :
  (dividend_per_share * share_percentage * total_shares).toNat = 153440 := by
  sorry

end net_profit_loan_payments_dividends_per_share_director_dividends_l348_348517


namespace southton_capsule_depth_l348_348370

theorem southton_capsule_depth :
  ∃ S : ℕ, 4 * S + 12 = 48 ∧ S = 9 :=
by
  sorry

end southton_capsule_depth_l348_348370


namespace each_friend_receives_one_grape_lollipop_l348_348131

noncomputable def total_lollipops := 60
noncomputable def num_friends := 6

noncomputable def percent_cherry := 30 / 100
noncomputable def percent_watermelon := 20 / 100
noncomputable def percent_sour_apple := 15 / 100
noncomputable def percent_blue_raspberry_grape := (100 / 100) - (percent_cherry + percent_watermelon + percent_sour_apple)
noncomputable def percent_each_remaining_flavor := percent_blue_raspberry_grape / 2

noncomputable def cherry_lollipops := total_lollipops * percent_cherry
noncomputable def watermelon_lollipops := total_lollipops * percent_watermelon
noncomputable def sour_apple_lollipops := total_lollipops * percent_sour_apple
noncomputable def grape_lollipops := total_lollipops * percent_each_remaining_flavor

noncomputable def lollipops_per_person_grape : ℕ := (grape_lollipops / num_friends).toNat

theorem each_friend_receives_one_grape_lollipop
    (total_lollipops = 60)
    (num_friends = 6)
    (percent_cherry = 0.30)
    (percent_watermelon = 0.20)
    (percent_sour_apple = 0.15)
    (percent_each_remaining_flavor := (1 - (0.30 + 0.20 + 0.15)) / 2)
    (grape_lollipops := (total_lollipops * percent_each_remaining_flavor).toNat) :
  lollipops_per_person_grape = 1 := by
  sorry

end each_friend_receives_one_grape_lollipop_l348_348131


namespace different_books_read_l348_348046

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l348_348046


namespace tommy_paint_cost_l348_348037

theorem tommy_paint_cost :
  ∀ (width height : ℕ) (cost_per_quart coverage_per_quart : ℕ),
    width = 5 →
    height = 4 →
    cost_per_quart = 2 →
    coverage_per_quart = 4 →
    2 * width * height / coverage_per_quart * cost_per_quart = 20 :=
by
  intros width height cost_per_quart coverage_per_quart
  intros hwidth hheight hcost hcoverage
  rw [hwidth, hheight, hcost, hcoverage]
  simp
  sorry

end tommy_paint_cost_l348_348037


namespace rhombus_perimeter_l348_348000

-- Define the conditions
def is_rhombus (d1 d2 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ (d1 / 2)^2 + (d2 / 2)^2 = (√((d1 / 2)^2 + (d2 / 2)^2))^2

-- The theorem statement that we need to prove
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : is_rhombus d1 d2) (hd1 : d1 = 8) (hd2 : d2 = 30) : 
  4 * √((d1 / 2)^2 + (d2 / 2)^2) = 4 * √241 := 
by
  sorry

end rhombus_perimeter_l348_348000


namespace h_at_neg_eight_l348_348326

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + x + 1

noncomputable def h (x : ℝ) (a b c : ℝ) : ℝ := (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_neg_eight (a b c : ℝ) (hf : f a = 0) (hf_b : f b = 0) (hf_c : f c = 0) :
  h (-8) a b c = -115 :=
  sorry

end h_at_neg_eight_l348_348326


namespace number_of_Bs_l348_348023

theorem number_of_Bs 
  (normal_recess : ℕ)
  (extra_min_per_A : ℕ)
  (extra_min_per_B : ℕ)
  (extra_min_per_C : ℕ)
  (extra_min_per_D : ℤ)
  (num_As : ℕ)
  (num_Cs : ℕ)
  (num_Ds : ℕ)
  (total_recess : ℕ)
  (normal_recess = 20)
  (extra_min_per_A = 2)
  (extra_min_per_B = 1)
  (extra_min_per_C = 0)
  (extra_min_per_D = -1)
  (num_As = 10)
  (num_Cs = 14)
  (num_Ds = 5)
  (total_recess = 47) : 
  ∃ num_Bs : ℕ, num_Bs = 12 :=
by
  sorry

end number_of_Bs_l348_348023


namespace exists_collinear_B_points_l348_348937

noncomputable def intersection (A B C D : Point) : Point :=
sorry

noncomputable def collinearity (P Q R S T : Point) : Prop :=
sorry

def convex_pentagon (A1 A2 A3 A4 A5 : Point) : Prop :=
-- Condition ensuring A1, A2, A3, A4, A5 form a convex pentagon, to be precisely defined
sorry

theorem exists_collinear_B_points :
  ∃ (A1 A2 A3 A4 A5 : Point),
    convex_pentagon A1 A2 A3 A4 A5 ∧
    collinearity
      (intersection A1 A4 A2 A3)
      (intersection A2 A5 A3 A4)
      (intersection A3 A1 A4 A5)
      (intersection A4 A2 A5 A1)
      (intersection A5 A3 A1 A2) :=
sorry

end exists_collinear_B_points_l348_348937


namespace union_M_N_l348_348599

open Set Real

def M : Set ℝ := {x | exp (x - 1) > 1}
def N : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

theorem union_M_N :
  M ∪ N = {x | -1 < x ∧ x ∈ Iio ∞} :=
by
  sorry

end union_M_N_l348_348599


namespace ratio_of_smaller_to_trapezoid_l348_348501

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

def ratio_of_areas : ℝ :=
  let side_large := 10
  let side_small := 5
  let area_large := area_equilateral_triangle side_large
  let area_small := area_equilateral_triangle side_small
  let area_trapezoid := area_large - area_small
  area_small / area_trapezoid

theorem ratio_of_smaller_to_trapezoid :
  ratio_of_areas = 1 / 3 :=
sorry

end ratio_of_smaller_to_trapezoid_l348_348501


namespace sqrt_inequality_l348_348580

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
by 
  sorry

end sqrt_inequality_l348_348580


namespace saline_drip_duration_l348_348096

theorem saline_drip_duration (rate_drops_per_minute : ℕ) (drop_to_ml_rate : ℕ → ℕ → Prop)
  (ml_received : ℕ) (time_hours : ℕ) :
  rate_drops_per_minute = 20 ->
  drop_to_ml_rate 100 5 ->
  ml_received = 120 ->
  time_hours = 2 :=
by {
  sorry
}

end saline_drip_duration_l348_348096


namespace mr_mcpherson_needs_to_raise_840_l348_348723

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end mr_mcpherson_needs_to_raise_840_l348_348723


namespace eq_of_line_through_points_l348_348392

noncomputable def line_eqn (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem eq_of_line_through_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = -1 → y1 = 2 → x2 = 2 → y2 = 5 → 
    line_eqn (x1 + y1 - x2) (y2 - y1) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  rw [hx1, hy1, hx2, hy2]
  sorry -- Proof steps would go here.

end eq_of_line_through_points_l348_348392


namespace compare_f_sin_cos_tan_l348_348612

def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(-x) = f(x)

def is_increasing_on (f : ℝ → ℝ) (s: set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f(a) < f(b)

def f : ℝ → ℝ := sorry

theorem compare_f_sin_cos_tan (h_even : is_even f) (h_incr : is_increasing_on f (set.Ici 0)) :
    let a := f (real.sin (50 * real.pi / 180))
        b := f (real.cos (50 * real.pi / 180))
        c := f (real.tan (50 * real.pi / 180))
    in b < a ∧ a < c :=
by
  let a := f (real.sin (50 * real.pi / 180))
  let b := f (real.cos (50 * real.pi / 180))
  let c := f (real.tan (50 * real.pi / 180))
  sorry

end compare_f_sin_cos_tan_l348_348612


namespace net_profit_loan_payments_dividends_per_share_director_dividends_l348_348519

theorem net_profit (revenue expenses : ℕ) (tax_rate : ℚ) 
  (h_rev : revenue = 2500000)
  (h_exp : expenses = 1576250)
  (h_tax : tax_rate = 0.2) :
  ((revenue - expenses) - (revenue - expenses) * tax_rate).toNat = 739000 := by
  sorry

theorem loan_payments (monthly_payment : ℕ) 
  (h_monthly : monthly_payment = 25000) :
  (monthly_payment * 12) = 300000 := by
  sorry

theorem dividends_per_share (net_profit loan_payments : ℕ) (total_shares : ℕ)
  (h_net_profit : net_profit = 739000)
  (h_loan_payments : loan_payments = 300000)
  (h_shares : total_shares = 1600) :
  ((net_profit - loan_payments) / total_shares) = 274 := by
  sorry

theorem director_dividends (dividend_per_share : ℕ) (share_percentage : ℚ) (total_shares : ℕ)
  (h_dividend_per_share : dividend_per_share = 274)
  (h_percentage : share_percentage = 0.35)
  (h_shares : total_shares = 1600) :
  (dividend_per_share * share_percentage * total_shares).toNat = 153440 := by
  sorry

end net_profit_loan_payments_dividends_per_share_director_dividends_l348_348519


namespace ratio_of_areas_l348_348498

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l348_348498


namespace sum_of_number_and_reverse_l348_348002

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l348_348002


namespace solve_x_l348_348195

theorem solve_x (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) := 
by 
  sorry

end solve_x_l348_348195


namespace product_successive_numbers_l348_348020

theorem product_successive_numbers (n : ℝ) (h : n = 88.49858755935034) :
  n * (n + 1) ≈ 7913 := by
  sorry

end product_successive_numbers_l348_348020


namespace solution_set_l348_348741

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l348_348741


namespace f_conjecture_l348_348705

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_conjecture (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 :=
by
  have h₀ : f x = 1 / (3^x + Real.sqrt 3), from rfl
  have h₁ : f (1 - x) = 1 / (3^(1 - x) + Real.sqrt 3), from rfl
  sorry

end f_conjecture_l348_348705


namespace lean_proof_l348_348127

noncomputable def problem1 := (2 + 1 / 4)^(1 / 2) - (-2018)^0 - (3 + 3 / 8)^(2 / 3) + (2 / 3)^(-2) = 1 / 2

noncomputable def problem2 := 100^(1 / 2 * log 9 - log 2) + Real.log (e^(3 / 4)) - (logBase 9 8 * logBase 4 (3^(1 / 3))) = 11 / 4

theorem lean_proof : problem1 ∧ problem2 := by
  sorry

end lean_proof_l348_348127


namespace value_of_x_is_4_l348_348671

variable {A B C D E F G H P : ℕ}

theorem value_of_x_is_4 (h1 : 5 + A + B = 19)
                        (h2 : A + B + C = 19)
                        (h3 : C + D + E = 19)
                        (h4 : D + E + F = 19)
                        (h5 : F + x + G = 19)
                        (h6 : x + G + H = 19)
                        (h7 : H + P + 10 = 19) :
                        x = 4 :=
by
  sorry

end value_of_x_is_4_l348_348671


namespace range_of_x_l348_348985

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 
  (if x > 0 then f x = Real.log x 
  else if x < 0 then f x = - Real.log (-x) 
  else f x = 0)

theorem range_of_x (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_def : f_definition f) :
  {x : ℝ | (x - 1) * f x < 0} = Ioo (-1:ℝ) (0:ℝ) :=
sorry

end range_of_x_l348_348985


namespace tommy_paint_cost_l348_348040

def tommy_spends_on_paint : ℕ :=
  let width := 5 in
  let height := 4 in
  let sides := 2 in
  let cost_per_quart := 2 in
  let coverage_per_quart := 4 in
  let area_per_side := width * height in
  let total_area := sides * area_per_side in
  let quarts_needed := total_area / coverage_per_quart in
  let total_cost := quarts_needed * cost_per_quart in
  total_cost

theorem tommy_paint_cost : tommy_spends_on_paint = 20 := by
  sorry

end tommy_paint_cost_l348_348040


namespace main_inequality_l348_348568

theorem main_inequality (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ m = -4 := by
  sorry

end main_inequality_l348_348568


namespace max_correct_answers_l348_348848

variable (x y z : ℕ)

theorem max_correct_answers
  (h1 : x + y + z = 100)
  (h2 : x - 3 * y - 2 * z = 50) :
  x ≤ 87 := by
    sorry

end max_correct_answers_l348_348848


namespace ratio_AB_CD_lengths_AB_CD_l348_348078

theorem ratio_AB_CD 
  (AM MD BN NC : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  : (AM / MD) / (BN / NC) = 5 / 6 :=
by
  sorry

theorem lengths_AB_CD
  (AM MD BN NC AB CD : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  (AB_div_CD : (AM / MD) / (BN / NC) = 5 / 6)
  (h_touch : true)  -- A placeholder condition indicating circles touch each other
  : AB = 5 ∧ CD = 6 :=
by
  sorry

end ratio_AB_CD_lengths_AB_CD_l348_348078


namespace max_set_with_property_P_l348_348967

-- Define property P
def has_property_P (M : Set ℕ) : Prop :=
  ∀ a b c ∈ M, a * b ≠ c

-- Define maximum size of sets with property P
def max_size_set_with_P : ℕ :=
  1968

theorem max_set_with_property_P :
  ∃ M : Set ℕ, M ⊆ {n | n ≤ 2011} ∧ has_property_P M ∧ M.card = max_size_set_with_P :=
by {
  sorry
}

end max_set_with_property_P_l348_348967


namespace truncated_cone_volume_l348_348397

-- Define the radii and height of the truncated cone
def R : ℝ := 10
def r : ℝ := 3
def H : ℝ := 8

-- Define the original height of the larger cone (h) and smaller cone (h') based on proportionality
noncomputable def h : ℝ := 80 / 7
noncomputable def h' : ℝ := 24 / 7

-- Define the volumes of the original cone (V0) and the smaller cone (V1)
noncomputable def V0 : ℝ := (1 / 3) * (Math.pi) * (R^2) * h
noncomputable def V1 : ℝ := (1 / 3) * (Math.pi) * (r^2) * h'

-- Calculate the final volume of the truncated cone
noncomputable def V : ℝ := V0 - V1

-- Statement of the theorem to prove
theorem truncated_cone_volume :
  V = (7784 * Math.pi) / 21 :=
by
  sorry

end truncated_cone_volume_l348_348397


namespace point_P_coordinates_l348_348986

theorem point_P_coordinates (α : ℝ) (P : ℝ × ℝ) (x y : ℝ) 
(h1 : α = π / 4) 
(h2 : P = (x, y)) 
(h3 : sqrt (x ^ 2 + y ^ 2) = sqrt 2) 
(h4 : sin (α) = y / sqrt 2) 
(h5 : cos (α) = x / sqrt 2) :
P = (1, 1) := 
by 
  sorry

end point_P_coordinates_l348_348986


namespace rectangle_difference_l348_348214

theorem rectangle_difference (A d x y : ℝ) (h1 : x * y = A) (h2 : x^2 + y^2 = d^2) :
  x - y = 2 * Real.sqrt A := 
sorry

end rectangle_difference_l348_348214


namespace min_value_f_l348_348623

noncomputable def f (x : ℝ) : ℝ := sin x + sin (x + π / 3)

theorem min_value_f :
  ∃ (m : ℝ) (S : set ℝ), m = -√3 ∧ (∀ x ∈ S, f x = m) ∧ 
  (S = { x | ∃ k : ℤ, x = 2 * k * π - 2 * π / 3 }) :=
sorry

end min_value_f_l348_348623


namespace sqrt_54_sub_sqrt_6_l348_348540

theorem sqrt_54_sub_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end sqrt_54_sub_sqrt_6_l348_348540


namespace solution_of_system_l348_348751

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l348_348751


namespace sum_of_fully_paintable_numbers_l348_348278

def is_fully_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, (∀ k1 : ℕ, n ≠ 1 + k1 * h) ∧ (∀ k2 : ℕ, n ≠ 3 + k2 * t) ∧ (∀ k3 : ℕ, n ≠ 2 + k3 * u)) → False

theorem sum_of_fully_paintable_numbers :  ∃ L : List ℕ, (∀ x ∈ L, ∃ (h t u : ℕ), is_fully_paintable h t u ∧ 100 * h + 10 * t + u = x) ∧ L.sum = 944 :=
sorry

end sum_of_fully_paintable_numbers_l348_348278


namespace range_of_a_l348_348785

def is_zero_point_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f x = 0

def function_f (a : ℝ) : ℝ → ℝ := λ x, a * x + 1 - 2 * a

theorem range_of_a (a : ℝ) : is_zero_point_in_interval (function_f a) a ↔ (1/3 : ℝ) < a ∧ a < 1 := by
  sorry

end range_of_a_l348_348785


namespace ratio_of_sums_l348_348107

theorem ratio_of_sums (total_sums : ℕ) (correct_sums : ℕ) (incorrect_sums : ℕ)
  (h1 : total_sums = 75)
  (h2 : incorrect_sums = 2 * correct_sums)
  (h3 : total_sums = correct_sums + incorrect_sums) :
  incorrect_sums / correct_sums = 2 :=
by
  -- Proof placeholder
  sorry

end ratio_of_sums_l348_348107


namespace cos_identity_150_deg_l348_348549

theorem cos_identity_150_deg :
  cos 150 + cos (-150) = -real.sqrt 3 := 
by 
  sorry

end cos_identity_150_deg_l348_348549


namespace value_of_f_neg_t_l348_348010

def f (x : ℝ) : ℝ := 3 * x + Real.sin x + 1

theorem value_of_f_neg_t (t : ℝ) (h : f t = 2) : f (-t) = 0 :=
by
  sorry

end value_of_f_neg_t_l348_348010


namespace necessary_but_not_sufficient_condition_l348_348081

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∃ x, 0 < x ∧ x < 1 ∧ f(x) = 0 → a < 1) ∧ (¬ (a < 1 → ∃ x, 0 < x ∧ x < 1 ∧ f(x) = 0)) :=
by
  have f : ℝ → ℝ := fun x => x - a
  sorry

end necessary_but_not_sufficient_condition_l348_348081


namespace cyclic_sum_inequality_l348_348575

open BigOperators

theorem cyclic_sum_inequality {n : ℕ} (h : 0 < n) (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i) :
  (∑ k in Finset.range n, a k / (a (k+1) + a (k+2))) > n / 4 := by
  sorry

end cyclic_sum_inequality_l348_348575


namespace part1_part2_l348_348222

-- Given that the sum of the first n terms of the sequence {a_n} is S_n, and 3S_n + a_n = 4.
variable {n : ℕ}
def S (n : ℕ) : ℝ
def a (n : ℕ) : ℝ

axiom sum_cond (n : ℕ) : 3 * S n + a n = 4

-- Let c_n = n * a_n and the sum of the first n terms of the sequence {c_n} is T_n.
def c (n : ℕ) : ℝ := n * a n
def T (n : ℕ) : ℝ := (∑ i in finset.range n + 1, c i)

-- Prove the general formula for {a_n}
def a_gen (n : ℕ) : Prop := a n = 1 / (4 ^ (n - 1))

-- Prove that T_n < 16/9
def T_ineq (n : ℕ) : Prop := T n < 16 / 9

-- Main theorem statements
theorem part1 (H : sum_cond n) : a_gen n := sorry

theorem part2 (H_gen : ∀ n, a n = 1 / (4 ^ (n - 1))) : T_ineq n := sorry

end part1_part2_l348_348222


namespace y_intercept_exists_l348_348787

def line_eq (x y : ℝ) : Prop := x + 2 * y + 2 = 0

theorem y_intercept_exists : ∃ y : ℝ, line_eq 0 y ∧ y = -1 :=
by
  sorry

end y_intercept_exists_l348_348787


namespace helen_owes_more_l348_348635

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value_semiannually : ℝ :=
  future_value 8000 0.10 2 3

noncomputable def future_value_annually : ℝ :=
  8000 * (1 + 0.10) ^ 3

noncomputable def difference : ℝ :=
  future_value_semiannually - future_value_annually

theorem helen_owes_more : abs (difference - 72.80) < 0.01 :=
by
  sorry

end helen_owes_more_l348_348635


namespace trench_dig_time_l348_348186

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l348_348186


namespace find_radius_l348_348087

-- Definitions based on conditions
def circle_radius (r : ℝ) : Prop := r = 2

-- Specification based on the question and conditions
theorem find_radius (r : ℝ) : circle_radius r :=
by
  -- Skip the proof
  sorry

end find_radius_l348_348087


namespace measure_of_arc_CD_l348_348961

variables (m n : ℝ)
variables (angle_DMC angle_DNC : ℝ)

-- Given the degree measure of arc AB is m
def arc_AB := m

-- Given angle DMC formed by the intersection of chords AC and BD
def angle_DMC := (arc_AB + n) / 2

-- Given angle DNC is an inscribed angle subtending arc CD
def angle_DNC := n / 2

-- Given angle DMC is equal to angle DNC
def angle_DMC_eq_angle_DNC := angle_DMC = angle_DNC

/-- The measure of arc CD n is equal to 180 degrees minus half of m -/
theorem measure_of_arc_CD (h : angle_DMC_eq_angle_DNC) :
  n = 180 - m / 2 := sorry

end measure_of_arc_CD_l348_348961


namespace price_of_plain_lemonade_l348_348123

variables (P : ℝ) (glasses_sold : ℝ) (revenue_strawberry : ℝ) (extra_revenue_plain : ℝ)
variables (revenue_plain : ℝ)

-- Conditions
def condition1 : glasses_sold = 36 := by sorry
def condition2 : revenue_strawberry = 16 := by sorry
def condition3 : extra_revenue_plain = 11 := by sorry
def condition4 : revenue_plain = revenue_strawberry + extra_revenue_plain := by sorry

-- Theorem
theorem price_of_plain_lemonade
  (h1 : glasses_sold = 36)
  (h2 : revenue_strawberry = 16)
  (h3 : extra_revenue_plain = 11)
  (h4 : revenue_plain = revenue_strawberry + extra_revenue_plain)
  : P = 0.75 :=
begin
  -- Definitions based on conditions
  have h5 : revenue_plain = 27, from calc
    revenue_plain = 16 + 11 : by rw [h2, h3, h4]
    ... = 27 : by norm_num,
  have h6 : 36 * P = 27, from calc
    glasses_sold * P = revenue_plain : by sorry
    ... = 27 : by rw [h1, h5],
  -- Solve for P
  solve_by_elim = P,
  sorry,
end

end price_of_plain_lemonade_l348_348123


namespace irreducible_fraction_l348_348363

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by sorry

end irreducible_fraction_l348_348363


namespace find_smallest_n_l348_348330

def greatest_integer_least (x : ℝ) : ℤ := int.floor x

theorem find_smallest_n :
  ∃ (n : ℕ), n > 0 ∧
    greatest_integer_least (
      (sqrt (n+1) - sqrt (n))
      / (sqrt (4 * n^2 + 4 * n + 1) - sqrt (4 * n^2 + 4 * n))
    ) = greatest_integer_least (sqrt (4 * n + 2018)) ∧ 
    n = 252253 := sorry

end find_smallest_n_l348_348330


namespace tommy_paint_cost_l348_348039

theorem tommy_paint_cost :
  ∀ (width height : ℕ) (cost_per_quart coverage_per_quart : ℕ),
    width = 5 →
    height = 4 →
    cost_per_quart = 2 →
    coverage_per_quart = 4 →
    2 * width * height / coverage_per_quart * cost_per_quart = 20 :=
by
  intros width height cost_per_quart coverage_per_quart
  intros hwidth hheight hcost hcoverage
  rw [hwidth, hheight, hcost, hcoverage]
  simp
  sorry

end tommy_paint_cost_l348_348039


namespace sum_of_first_ten_terms_l348_348629

theorem sum_of_first_ten_terms (S : ℕ → ℕ) (h : ∀ n, S n = n^2 - 4 * n + 1) : S 10 = 61 :=
by
  sorry

end sum_of_first_ten_terms_l348_348629


namespace log_tan_sum_l348_348559

theorem log_tan_sum :
  ∑ i in finset.range 89, real.logb 10 (real.tan ((i + 1) * (real.pi / 360))) = 0 :=
by
  sorry

end log_tan_sum_l348_348559


namespace total_number_of_pets_l348_348800

open Nat

theorem total_number_of_pets (d f c : Nat) (hd : d = 43) (hf : f = 72) (hc : c = 34) : d + f + c = 149 :=
by
  rw [hd, hf, hc]
  rfl

end total_number_of_pets_l348_348800


namespace converse_of_x_eq_one_implies_x_squared_eq_one_l348_348379

theorem converse_of_x_eq_one_implies_x_squared_eq_one (x : ℝ) : x^2 = 1 → x = 1 := 
sorry

end converse_of_x_eq_one_implies_x_squared_eq_one_l348_348379


namespace degree_difference_l348_348371

variable (S J : ℕ)

theorem degree_difference :
  S = 150 → S + J = 295 → S - J = 5 :=
by
  intros h₁ h₂
  sorry

end degree_difference_l348_348371


namespace James_balloons_correct_l348_348317

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct_l348_348317


namespace time_for_first_three_workers_l348_348190

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l348_348190


namespace percentage_profit_l348_348891

theorem percentage_profit (cp sp : ℝ) (h1 : cp = 1200) (h2 : sp = 1680) : ((sp - cp) / cp) * 100 = 40 := 
by 
  sorry

end percentage_profit_l348_348891


namespace range_of_a_l348_348626

noncomputable def f (x: ℝ) : ℝ := -2^x

noncomputable def g (a: ℝ) (x: ℝ) : ℝ := log (10, a * x^2 - 2 * x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g a x2) ↔ a ∈ Set.Iic 1 := sorry

end range_of_a_l348_348626


namespace new_ribbon_lengths_correct_l348_348822

noncomputable def ribbon_lengths := [15, 20, 24, 26, 30]
noncomputable def new_average_change := 5
noncomputable def new_lengths := [9, 9, 24, 24, 24]

theorem new_ribbon_lengths_correct :
  let new_length_list := [9, 9, 24, 24, 24]
  ribbon_lengths.length = 5 ∧ -- we have 5 ribbons
  new_length_list.length = 5 ∧ -- the new list also has 5 ribbons
  list.average new_length_list = list.average ribbon_lengths - new_average_change ∧ -- new average decreased by 5
  list.median new_length_list = list.median ribbon_lengths ∧ -- median unchanged
  list.range new_length_list = list.range ribbon_lengths -- range unchanged
  :=
by {
  sorry
}

end new_ribbon_lengths_correct_l348_348822


namespace coeff_of_x_sqrt_x_in_expansion_l348_348662

theorem coeff_of_x_sqrt_x_in_expansion :
  (∃ c : ℕ, (∑ k in finset.range 6, (nat.choose 5 k) * (-1)^k * (x^(5 - k)/2)) = c * x^(3/2)) ∧ c = 10 :=
  sorry

end coeff_of_x_sqrt_x_in_expansion_l348_348662


namespace additional_people_proof_l348_348557

variable (initialPeople additionalPeople mowingHours trimmingRate totalNewPeople totalMowingPeople requiredPersonHours totalPersonHours: ℕ)

noncomputable def mowingLawn (initialPeople mowingHours : ℕ) : ℕ :=
  initialPeople * mowingHours

noncomputable def mowingRate (requiredPersonHours : ℕ) (mowingHours : ℕ) : ℕ :=
  (requiredPersonHours / mowingHours)

noncomputable def trimmingEdges (totalMowingPeople trimmingRate : ℕ) : ℕ :=
  (totalMowingPeople / trimmingRate)

noncomputable def totalPeople (mowingPeople trimmingPeople : ℕ) : ℕ :=
  (mowingPeople + trimmingPeople)

noncomputable def additionalPeopleNeeded (totalPeople initialPeople : ℕ) : ℕ :=
  (totalPeople - initialPeople)

theorem additional_people_proof :
  initialPeople = 8 →
  mowingHours = 3 →
  totalPersonHours = mowingLawn initialPeople mowingHours →
  totalMowingPeople = mowingRate totalPersonHours 2 →
  trimmingRate = 3 →
  requiredPersonHours = totalPersonHours →
  totalNewPeople = totalPeople totalMowingPeople (trimmingEdges totalMowingPeople trimmingRate) →
  additionalPeople = additionalPeopleNeeded totalNewPeople initialPeople →
  additionalPeople = 8 :=
by
  sorry

end additional_people_proof_l348_348557


namespace triangle_area_scaled_l348_348646

theorem triangle_area_scaled (a b θ : ℝ) :
  let A := (a * b * Real.sin θ) / 2 in
  let A' := (3 * a * 2 * b * Real.sin θ) / 2 in
  A' = 6 * A :=
by
  sorry

end triangle_area_scaled_l348_348646


namespace fastest_hike_is_1_hour_faster_l348_348683

-- Given conditions
def distance_trail_A : ℕ := 20
def speed_trail_A : ℕ := 5
def distance_trail_B : ℕ := 12
def speed_trail_B : ℕ := 3
def mandatory_break_time : ℕ := 1

-- Lean statement to prove the problem
theorem fastest_hike_is_1_hour_faster :
  let time_trail_A := distance_trail_A / speed_trail_A in
  let hiking_time_trail_B := distance_trail_B / speed_trail_B in
  let total_time_trail_B := hiking_time_trail_B + mandatory_break_time in
  total_time_trail_B - time_trail_A = 1 :=
by
  let time_trail_A := distance_trail_A / speed_trail_A
  let hiking_time_trail_B := distance_trail_B / speed_trail_B
  let total_time_trail_B := hiking_time_trail_B + mandatory_break_time
  have h : total_time_trail_B - time_trail_A = 1, sorry
  exact h

end fastest_hike_is_1_hour_faster_l348_348683


namespace connectivity_queries_lower_bound_l348_348292

-- Define the conditions
def cities : Type := Fin 64
def road (c1 c2 : cities) : Prop 

noncomputable def number_of_queries_needed : Nat := 2016

-- Define the theorem
theorem connectivity_queries_lower_bound :
  ¬∃ alg : (cities → cities → Prop) → Prop, ∀ c1 c2 : cities, alg road → road c1 c2 → road c1 c2 := 
begin
  -- Here we need to construct the proof that no such algorithm exists
  sorry
end

end connectivity_queries_lower_bound_l348_348292


namespace alice_ride_average_speed_l348_348115

theorem alice_ride_average_speed
    (d1 d2 : ℝ) 
    (s1 s2 : ℝ)
    (h_d1 : d1 = 40)
    (h_d2 : d2 = 20)
    (h_s1 : s1 = 8)
    (h_s2 : s2 = 40) :
    (d1 + d2) / (d1 / s1 + d2 / s2) = 10.909 :=
by
  simp [h_d1, h_d2, h_s1, h_s2]
  norm_num
  sorry

end alice_ride_average_speed_l348_348115


namespace basic_salary_third_year_annual_growth_rate_l348_348085

-- Define the conditions
def basic_salary_first_year : ℝ := 1
def housing_allowance_first_year : ℝ := 0.04
def medical_expenses_per_year : ℝ := 0.1384
def housing_allowance_increase_per_year : ℝ := 0.04

-- Define the parameter for the growth rate of the basic salary
variable (x : ℝ)

-- 1. Statement: Expression for the basic salary in the third year
theorem basic_salary_third_year : basic_salary_first_year * (1 + x) * (1 + x) = (1 + x)^2 :=
by sorry

-- Define the summation condition
def housing_allowance_sum := housing_allowance_first_year + 2 * housing_allowance_increase_per_year + 3 * housing_allowance_increase_per_year
def medical_expenses_sum := medical_expenses_per_year * 3
def total_sum := housing_allowance_sum + medical_expenses_sum
def total_basic_salary_3_years := 1 + (1 + x) + (1 + x)^2

-- 2. Statement: Calculation of the annual growth rate of the basic salary
theorem annual_growth_rate : 0.18 * total_basic_salary_3_years = total_sum → x = 0.2 :=
by sorry

end basic_salary_third_year_annual_growth_rate_l348_348085


namespace sum_a1_to_a5_max_S_n_l348_348604

-- Definitions for part 1
variable {a : ℕ → ℝ}
variable {d : ℝ}
axiom arithmetic_seq (n : ℕ) : a (n+1) = a n + d
axiom geo_condition : (a 1 + d)^2 = (a 1) * (a 1 + 4 * d)
axiom sum_condition : a 3 + a 4 = 12

-- Proof statement for part 1
theorem sum_a1_to_a5 : a 1 + a 2 + a 3 + a 4 + a 5 = 25 ∨ a 1 + a 2 + a 3 + a 4 + a 5 = 30 :=
sorry

-- Definitions for part 2
variable {b : ℕ → ℝ}
axiom b_definition (n : ℕ) : b n = 10 - a n
axiom b1_ne_b2 : b 1 ≠ b 2

-- Sum of first n terms of b
noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Proof statement for part 2
theorem max_S_n : (∃ n, S n = 25 ∧ ∀ m, S m ≤ S n) :=
sorry

end sum_a1_to_a5_max_S_n_l348_348604


namespace sequence_bounded_l348_348021

open Classical

noncomputable def bounded_sequence (a : ℕ → ℝ) (M : ℝ) :=
  ∀ n : ℕ, n > 0 → a n < M

theorem sequence_bounded {a : ℕ → ℝ} (h0 : 0 ≤ a 1 ∧ a 1 ≤ 2)
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = a n + (a n)^2 / n^3) :
  ∃ M : ℝ, 0 < M ∧ bounded_sequence a M :=
by
  sorry

end sequence_bounded_l348_348021


namespace sum_of_first_five_terms_l348_348968

noncomputable -- assuming non-computable for general proof involving sums
def arithmetic_sequence_sum (a_n : ℕ → ℤ) := ∃ d m : ℤ, ∀ n : ℕ, a_n = m + n * d

theorem sum_of_first_five_terms 
(a_n : ℕ → ℤ) 
(h_arith : arithmetic_sequence_sum a_n)
(h_cond : a_n 5 + a_n 8 - a_n 10 = 2)
: ((a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) = 10) := 
by 
  sorry

end sum_of_first_five_terms_l348_348968


namespace sum_of_x_and_y_l348_348258

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l348_348258


namespace cos_alpha_third_quadrant_l348_348609

theorem cos_alpha_third_quadrant (α : ℝ) (hα1 : π < α ∧ α < 3 * π / 2) (hα2 : Real.tan α = 4 / 3) :
  Real.cos α = -3 / 5 :=
sorry

end cos_alpha_third_quadrant_l348_348609


namespace prove_half_lives_l348_348445

variable (m₀ t λ₁ λ₂ : ℝ)

-- Initial masses
def initial_mass_A := m₀
def initial_mass_B := 2 * m₀

-- Mass after t years
def mass_A (t : ℝ) := m₀ * 2^(-λ₁ * t)
def mass_B (t : ℝ) := 2 * m₀ * 2^(-λ₂ * t)

-- Half-life condition
def half_life_condition_A (t half_life_A : ℝ) := mass_A t = initial_mass_A / 2 → t = half_life_A
def half_life_condition_B (t half_life_B : ℝ) := mass_B t = initial_mass_B / 2 → t = half_life_B

-- Given relationship between half-lives and decay rates
def decay_rate_A (half_life_A : ℝ) := λ₁ = 1 / (2 * half_life_A)
def decay_rate_B (half_life_B : ℝ) := λ₂ = 1 / half_life_B

-- Condition after 20 years
def total_mass_condition (half_life_A half_life_B : ℝ) := 
  mass_A 20 + mass_B 20 = (initial_mass_A + initial_mass_B) / 8

-- Main theorem to prove the half-lives
theorem prove_half_lives : ∃ (half_life_A half_life_B : ℝ), 
  decay_rate_A half_life_A ∧ 
  decay_rate_B half_life_B ∧ 
  half_life_condition_A half_life_A 10 ∧ 
  half_life_condition_B half_life_B 5 ∧ 
  total_mass_condition half_life_A half_life_B :=
sorry -- Proof skipped

end prove_half_lives_l348_348445


namespace ratio_of_areas_is_eight_fifths_l348_348553

variables (r : ℝ)

-- s₁ is the side length of the square inscribed in the semicircle with radius 2r
def s₁ := (16 * r^2 / 5).sqrt

-- s₂ is the side length of the square inscribed in the circle with radius r
def s₂ := (2 * r^2).sqrt

-- The ratio of the areas of the two squares
def ratio_of_areas := (s₁^2) / (s₂^2)

theorem ratio_of_areas_is_eight_fifths : 
  ratio_of_areas r = 8 / 5 :=
by
  -- Leaving the proof as an exercise (or to be completed later)
  sorry

end ratio_of_areas_is_eight_fifths_l348_348553


namespace polynomial_roots_l348_348936

noncomputable def z_polynomial : Polynomial ℂ :=
  Polynomial.C 1 * (Polynomial.X + 2) *
  (Polynomial.X - 1) *
  (Polynomial.X - Complex.i * Complex.sqrt 7) *
  (Polynomial.X + Complex.i * Complex.sqrt 7)

theorem polynomial_roots :
  (\( -2 : ℂ \) :=
  ∀ z : ℂ, (z = -2 ∨ z = 1 ∨ z = Complex.i * Complex.sqrt 7 ∨ z = -Complex.i * Complex.sqrt 7) → z ^ 4 - 6 * z ^ 2 + z + 8 = 0 := 
  sorry

end polynomial_roots_l348_348936


namespace decompose_vector_l348_348847

def vec_a : ℝ × ℝ × ℝ := (-1, -4, -2)
def vec_p : ℝ × ℝ × ℝ := (1, 2, 4)
def vec_q : ℝ × ℝ × ℝ := (1, -1, 1)
def vec_r : ℝ × ℝ × ℝ := (2, 2, 4)

theorem decompose_vector :
  vec_a = (1 : ℝ) • vec_p + (2 : ℝ) • vec_q + (-2 : ℝ) • vec_r := 
by 
  -- proof body
  sorry

end decompose_vector_l348_348847


namespace y_eq_fraction_x_l348_348444

theorem y_eq_fraction_x (p : ℝ) (x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) :=
sorry

end y_eq_fraction_x_l348_348444


namespace solution_set_line_l348_348731

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l348_348731


namespace part_1_a_eq_1_part_1_a_neq_1_part_2_part_3_l348_348201

variable {a : ℝ} (a_pos : a > 0)

def a_seq : ℕ → ℝ
| 1 := 1
| 2 := a
| n := sorry -- Recursive definition depends on additional conditions

def b_seq : ℕ → ℝ
| n := a_seq n * a_seq (n + 1)

theorem part_1_a_eq_1 (n : ℕ) (h_geom : ∀ n, a_seq n = 1) :
  (finset.range n).sum b_seq = n :=
sorry

theorem part_1_a_neq_1 (n : ℕ) (h_geom : ∀ n, a_seq n = a ^ (n - 1)) :
  (finset.range n).sum b_seq = (a * (1 - a^(2 * n))) / (1 - a^2) :=
sorry

theorem part_2 (b_eq_three_power : ∀ n, b_seq n = 3 ^ n) :
  ∀ n, a_seq n =
    if (n % 2 = 1) then 3 ^ ((n - 1) / 2)
    else a * 3 ^ ((n - 2) / 2) :=
sorry

theorem part_3 (n : ℕ) (b_eq_n_plus_2 : ∀ n, b_seq n = n + 2) :
  ∑ i in finset.range n, 1 / a_seq i > 2 * real.sqrt (n + 2) - 3 :=
sorry

end part_1_a_eq_1_part_1_a_neq_1_part_2_part_3_l348_348201


namespace problem_statement_l348_348976

noncomputable theory
open_locale big_operators

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

-- Conditions from the problem
def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * (2:ℝ) ^ n

def cond_2 (a : ℕ → ℝ) : Prop :=
(1 / a 0) - (1 / a 1) = (2 / a 2)

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(0.add n).sum a

def cond_S6 (a : ℕ → ℝ) : Prop :=
S_n a 5 = 63

-- Definition of b_n
def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(Real.log (a n * a (n+1)) / (Real.log 2)) / 2

-- Definition of T_n
def T_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
(∑ k in (range n).filter odd, (-1)^k*(b k)^2) + 
(∑ k in (range n).filter even, (b k)^2)

-- Proof statements
theorem problem_statement (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_geom: geom_seq a)
  (h_cond2: cond_2 a)
  (h_S6: cond_S6 a) :
  (∀ n : ℕ, a n = 2 ^ (n - 1)) ∧
  (∀ n : ℕ, T n = 2 * n^2) :=
sorry

end problem_statement_l348_348976


namespace curve_not_ellipse_l348_348899

theorem curve_not_ellipse (m : ℝ) :
  m ∈ (Set.Icc (-∞ : ℝ) 1) ∪ {2} ∪ (Set.Icc 3 (+∞ : ℝ)) →
  ¬ (∃ (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0),
    (m - 1) ≠ 0 ∧ (3 - m) ≠ 0 ∧
    ∀ (x y : ℝ), (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) ↔
    (x / a)^2 + (y / b)^2 = 1) :=
begin
  sorry
end

end curve_not_ellipse_l348_348899


namespace taxi_fare_function_l348_348281

theorem taxi_fare_function (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, y = 2 * x + 4 :=
by
  sorry

end taxi_fare_function_l348_348281


namespace cylinder_height_relationship_l348_348053

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_equal : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relationship : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l348_348053


namespace interval_of_increase_l348_348387

noncomputable def power_function_passing_through : ℝ → ℝ := λ x, x ^ (-2)

theorem interval_of_increase :
  (∃ f : ℝ → ℝ, (∀ x, f x = x ^ (-2)) ∧ (f 2 = 1/4)) →
  ∀ x y : ℝ, x < y → 0 < y → y < 0 →
  interval_increase (-∞, 0) := sorry

end interval_of_increase_l348_348387


namespace n_prime_of_divisors_l348_348351

theorem n_prime_of_divisors (n k : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ d : ℕ, d ∣ n → (d + k ∣ n) ∨ (d - k ∣ n)) : Prime n :=
  sorry

end n_prime_of_divisors_l348_348351


namespace polar_to_rectangular_and_line_equation_l348_348628

noncomputable def circle_O1_polar : Prop :=
  ∀ (ρ θ : ℝ), ρ = 2 → (ρ^2 = 4 ∧ (x^2 + y^2 = 4))

noncomputable def circle_O2_polar : Prop :=
  ∀ (ρ θ : ℝ), ρ^2 - 2 * sqrt 2 * ρ * cos (θ - π / 4) = 2 → (x^2 + y^2 - 2 * x - 2 * y - 2 = 0)

noncomputable def line_through_intersections: Prop :=
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x^2 + y^2 - 2 * x - 2 * y - 2 = 0)) →
  (ρ sin (θ + π / 4) = sqrt 2)

theorem polar_to_rectangular_and_line_equation :
  circle_O1_polar ∧ circle_O2_polar ∧ line_through_intersections :=
sorry

end polar_to_rectangular_and_line_equation_l348_348628


namespace find_plane_through_points_and_perpendicular_l348_348941

-- Definitions for points and plane conditions
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def point1 : Point3D := ⟨2, -2, 2⟩
def point2 : Point3D := ⟨0, 2, -1⟩

def normal_vector_of_given_plane : Point3D := ⟨2, -1, 2⟩

-- Lean 4 statement
theorem find_plane_through_points_and_perpendicular :
  ∃ (A B C D : ℤ), 
  (∀ (p : Point3D), (p = point1 ∨ p = point2) → A * p.x + B * p.y + C * p.z + D = 0) ∧
  (A * 2 + B * -1 + C * 2 = 0) ∧ 
  A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (A = 5 ∧ B = -2 ∧ C = 6 ∧ D = -26) :=
by
  sorry

end find_plane_through_points_and_perpendicular_l348_348941


namespace triangle_IGF_similar_to_ABC_l348_348864

-- Define points, segments, and relevant conditions for the problem
variables {A B C E F H I G : Point}
variables {triangle_ABC : Triangle}
variables (tri_sim : Similar triangle_ABC (Triangle.mk I G F))
variable (angle_C_90 : ∠ C = 90)
variable (point_E_on_AC : E ∈ AC)
variable (midpoint_F_of_EC : Midpoint F E C)
variable (altitude_CH : Altitude C H)
variable (circumcenter_I_of_AHE : Circumcenter I A H E)
variable (midpoint_G_of_BC : Midpoint G B C)

-- Statement that needs to be proved
theorem triangle_IGF_similar_to_ABC
  (h₁ : RightAngle (Angle.mk C))
  (h₂ : OnLine E AC)
  (h₃ : IsMidpoint F E C)
  (h₄ : IsAltitude H C)
  (h₅ : IsCircumcenter I A H E)
  (h₆ : IsMidpoint G B C) :
  Similar (Triangle.mk A B C) (Triangle.mk I G F) :=
sorry

end triangle_IGF_similar_to_ABC_l348_348864


namespace no_fixed_points_range_l348_348573

-- Define the function and fixed points conditions
def quadratic (x a : ℝ) : ℝ := x^2 + a*x + 1
def fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = x

-- State the main theorem
theorem no_fixed_points_range (a : ℝ) : 
  (¬ ∃ x : ℝ, fixed_point (quadratic x a)) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end no_fixed_points_range_l348_348573


namespace decagon_diagonal_intersections_l348_348116

theorem decagon_diagonal_intersections : 
  ∀ (n : ℕ), n = 10 → 
  finset.card (finset.choose 4 (finset.range n)) = 210 :=
by
  intros n h
  sorry

end decagon_diagonal_intersections_l348_348116


namespace cats_eat_fish_l348_348080

theorem cats_eat_fish (c d: ℕ) (h1 : 1 < c) (h2 : c < 10) (h3 : c * d = 91) : c + d = 20 := by
  sorry

end cats_eat_fish_l348_348080


namespace first_three_workers_time_l348_348182

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l348_348182


namespace part1_part2_l348_348221

-- Given that the sum of the first n terms of the sequence {a_n} is S_n, and 3S_n + a_n = 4.
variable {n : ℕ}
def S (n : ℕ) : ℝ
def a (n : ℕ) : ℝ

axiom sum_cond (n : ℕ) : 3 * S n + a n = 4

-- Let c_n = n * a_n and the sum of the first n terms of the sequence {c_n} is T_n.
def c (n : ℕ) : ℝ := n * a n
def T (n : ℕ) : ℝ := (∑ i in finset.range n + 1, c i)

-- Prove the general formula for {a_n}
def a_gen (n : ℕ) : Prop := a n = 1 / (4 ^ (n - 1))

-- Prove that T_n < 16/9
def T_ineq (n : ℕ) : Prop := T n < 16 / 9

-- Main theorem statements
theorem part1 (H : sum_cond n) : a_gen n := sorry

theorem part2 (H_gen : ∀ n, a n = 1 / (4 ^ (n - 1))) : T_ineq n := sorry

end part1_part2_l348_348221


namespace symmetric_point_coordinates_l348_348303

theorem symmetric_point_coordinates :
  ∀ (M N : ℝ × ℝ), M = (3, -4) ∧ M.fst = -N.fst ∧ M.snd = N.snd → N = (-3, -4) :=
by
  intro M N h
  sorry

end symmetric_point_coordinates_l348_348303


namespace solution_set_unique_line_l348_348748

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l348_348748


namespace ribbon_lengths_after_cut_l348_348826

theorem ribbon_lengths_after_cut {
  l1 l2 l3 l4 l5 l1' l2' l3' l4' l5' : ℕ
  (initial_lengths : multiset ℕ)
  (new_lengths : multiset ℕ)
  (hl : initial_lengths = {15, 20, 24, 26, 30})
  (hl' : new_lengths = {l1', l2', l3', l4', l5'})
  (h_average_decrease : (∑ x in initial_lengths, x) / 5 - 5 = (∑ x in new_lengths, x) / 5)
  (h_median : ∀ x ∈ new_lengths, x = 24 ∨ 24 ∈ new_lengths)
  (h_range : multiset.range new_lengths = 15)
  (h_lengths : l1' ≤ l2' ≤ l3' = 24 ∧ l3' ≤ l4' ≤ l5') :
  new_lengths = {9, 9, 24, 24, 24} := sorry

end ribbon_lengths_after_cut_l348_348826


namespace isosceles_perimeter_l348_348209

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_perimeter
  (k : ℝ)
  (a b : ℝ)
  (h1 : 4 = a)
  (h2 : k * b^2 - (k + 8) * b + 8 = 0)
  (h3 : k ≠ 0)
  (h4 : is_triangle 4 a a) : a + 4 + a = 9 :=
sorry

end isosceles_perimeter_l348_348209


namespace f_comp_f_one_over_four_l348_348966

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 4 * x else Real.log 4 x

theorem f_comp_f_one_over_four : f (f (1 / 4)) = -4 := by
  sorry

end f_comp_f_one_over_four_l348_348966


namespace sum_of_reciprocals_is_five_l348_348782

theorem sum_of_reciprocals_is_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = 3 * x * y) : 
  (1 / x) + (1 / y) = 5 :=
sorry

end sum_of_reciprocals_is_five_l348_348782


namespace g_increasing_on_minus_infty_one_l348_348611

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 + (2 * x) / (1 - x)

theorem g_increasing_on_minus_infty_one : (∀ x y : ℝ, x < y → x < 1 → y ≤ 1 → g x < g y) :=
sorry

end g_increasing_on_minus_infty_one_l348_348611


namespace leak_empties_tank_in_12_hours_l348_348464

theorem leak_empties_tank_in_12_hours 
  (capacity : ℕ) (inlet_rate : ℕ) (net_emptying_time : ℕ) (leak_rate : ℤ) (leak_emptying_time : ℕ) :
  capacity = 5760 →
  inlet_rate = 4 →
  net_emptying_time = 8 →
  (inlet_rate - leak_rate : ℤ) = (capacity / (net_emptying_time * 60)) →
  leak_emptying_time = (capacity / leak_rate) →
  leak_emptying_time = 12 * 60 / 60 :=
by sorry

end leak_empties_tank_in_12_hours_l348_348464


namespace problem_statement_l348_348325

theorem problem_statement (x y : ℝ) (m n : ℕ) (hmn : Nat.coprime m n) (hx : x = ↑m / ↑n)
  (h₁ : 129 - x^2 = x * y) (h₂ : 195 - y^2 = x * y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  100 * m + n = 4306 := by
  sorry

end problem_statement_l348_348325


namespace election_votes_and_deposit_l348_348658

theorem election_votes_and_deposit (V : ℕ) (A B C D E : ℕ) (hA : A = 40 * V / 100) 
  (hB : B = 28 * V / 100) (hC : C = 20 * V / 100) (hDE : D + E = 12 * V / 100)
  (win_margin : A - B = 500) :
  V = 4167 ∧ (15 * V / 100 ≤ A) ∧ (15 * V / 100 ≤ B) ∧ (15 * V / 100 ≤ C) ∧ 
  ¬ (15 * V / 100 ≤ D) ∧ ¬ (15 * V / 100 ≤ E) :=
by 
  sorry

end election_votes_and_deposit_l348_348658


namespace correct_calculation_l348_348432

theorem correct_calculation :
  (∀ x : ℤ, x^5 + x^3 ≠ x^8) ∧
  (∀ x : ℤ, x^5 - x^3 ≠ x^2) ∧
  (∀ x : ℤ, x^5 * x^3 = x^8) ∧
  (∀ x : ℤ, (-3 * x)^3 ≠ -9 * x^3) :=
by
  sorry

end correct_calculation_l348_348432


namespace variances_are_equal_thirtieth_percentile_is_neg13_l348_348587

namespace SampleData

def x (i : ℕ) : ℕ := if 1 ≤ i ∧ i ≤ 10 then 2 * i else 0
def y (i : ℕ) : ℤ := x i - 20

theorem variances_are_equal : (variance (finset.filter (λ i, x i > 0) (finset.range 11) (λ i, (x i : ℝ))) = 
                              variance (finset.filter (λ i, y i > -20) (finset.range 11) (λ i, (y i : ℝ)))) :=
sorry

theorem thirtieth_percentile_is_neg13 : percentile (finset.filter (λ i, y i > -20) (finset.range 11) 
                                                (λ i, (y i : ℚ))) 0.3 = -13 :=
sorry

end SampleData

end variances_are_equal_thirtieth_percentile_is_neg13_l348_348587


namespace sum_of_squares_of_roots_l348_348166

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) (h_eq : 6 * a = a) (h_b : b = -5) (h_c : c = -12) :
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
      x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) in
  x1^2 + x2^2 = 169 / 36 := 
by
  sorry

end sum_of_squares_of_roots_l348_348166


namespace proof_number_of_subsets_l348_348393

open Finset

-- Definition of the main problem statement
theorem proof_number_of_subsets (S : Finset ℕ) (T : Finset ℕ) 
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (hT : T.card = 4) 
  (h_diff : ∀ (x y ∈ T), x ≠ y → (x - y).nat_abs ≠ 1) :
  T.card = 35 := sorry

end proof_number_of_subsets_l348_348393


namespace solve_eq1_solve_eq2_l348_348768

-- Define the equations
def eq1 (x : ℝ) : Prop := x^2 - 2 * x = 1
def eq2 (x : ℝ) : Prop := x * (x - 3) = 7 * (3 - x)

-- State the theorem for equation 1
theorem solve_eq1 : ∀ (x : ℝ), eq1 x → (x = 1 + real.sqrt 2) ∨ (x = 1 - real.sqrt 2) :=
by
  intro x
  intro h
  -- The given condition is eq1 x
  sorry

-- State the theorem for equation 2
theorem solve_eq2 : ∀ (x : ℝ), eq2 x → (x = 3) ∨ (x = -7) :=
by
  intro x
  intro h
  -- The given condition is eq2 x
  sorry

end solve_eq1_solve_eq2_l348_348768


namespace min_box_value_l348_348639

theorem min_box_value (a b : ℤ) (h : a * b = 32) : a^2 + b^2 ≥ 80 :=
begin
  sorry
end

end min_box_value_l348_348639


namespace bob_remaining_money_l348_348513

def initial_amount : ℕ := 80
def monday_spent (initial : ℕ) : ℕ := initial / 2
def tuesday_spent (remaining_monday : ℕ) : ℕ := remaining_monday / 5
def wednesday_spent (remaining_tuesday : ℕ) : ℕ := remaining_tuesday * 3 / 8

theorem bob_remaining_money : 
  let remaining_monday := initial_amount - monday_spent initial_amount
  let remaining_tuesday := remaining_monday - tuesday_spent remaining_monday
  let final_remaining := remaining_tuesday - wednesday_spent remaining_tuesday
  in final_remaining = 20 := 
by
  -- Proof goes here
  sorry

end bob_remaining_money_l348_348513


namespace order_xyz_l348_348235

theorem order_xyz (x : ℝ) (h1 : 0.8 < x) (h2 : x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y :=
by
  sorry

end order_xyz_l348_348235


namespace total_complaints_proof_l348_348373

def base_complaints := 120

def inc_staff_shortage (percent : ℕ) : ℕ :=
  match percent with
  | 20 => base_complaints * 1 / 3
  | 40 => base_complaints * 2 / 3
  | _  => 0

def inc_self_checkout (status : ℕ) : ℕ :=
  match status with
  | 1 => base_complaints * 10 / 100
  | 2 => base_complaints * 20 / 100
  | _ => 0

def inc_weather (weather : ℕ) : ℕ :=
  match weather with
  | 1 => base_complaints * 5 / 100
  | 2 => base_complaints * 15 / 100
  | _ => 0

def inc_events (event : ℕ) : ℕ :=
  match event with
  | 1 => base_complaints * 25 / 100
  | 2 => base_complaints * 10 / 100
  | _ => 0

def total_complaints (staff : ℕ) (checkout : ℕ) (weather : ℕ) (event : ℕ) : ℕ :=
  base_complaints + inc_staff_shortage(staff) + inc_self_checkout(checkout) + inc_weather(weather) + inc_events(event)

def day1_complaints := total_complaints 20 2 1 2
def day2_complaints := total_complaints 40 1 0 1
def day3_complaints := total_complaints 40 2 2 0
def day4_complaints := total_complaints 0 0 1 2
def day5_complaints := total_complaints 20 2 0 1

def total_five_days_complaints := day1_complaints + day2_complaints + day3_complaints + day4_complaints + day5_complaints

theorem total_complaints_proof : total_five_days_complaints = 1038 := by
  unfold total_five_days_complaints day1_complaints day2_complaints day3_complaints day4_complaints day5_complaints total_complaints
  rw [add_assoc, add_assoc, add_assoc, add_assoc]
  sorry

end total_complaints_proof_l348_348373


namespace tires_in_parking_lot_l348_348272

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l348_348272


namespace dividends_CEO_2018_l348_348525

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l348_348525


namespace net_profit_loan_payments_dividends_per_share_director_dividends_l348_348518

theorem net_profit (revenue expenses : ℕ) (tax_rate : ℚ) 
  (h_rev : revenue = 2500000)
  (h_exp : expenses = 1576250)
  (h_tax : tax_rate = 0.2) :
  ((revenue - expenses) - (revenue - expenses) * tax_rate).toNat = 739000 := by
  sorry

theorem loan_payments (monthly_payment : ℕ) 
  (h_monthly : monthly_payment = 25000) :
  (monthly_payment * 12) = 300000 := by
  sorry

theorem dividends_per_share (net_profit loan_payments : ℕ) (total_shares : ℕ)
  (h_net_profit : net_profit = 739000)
  (h_loan_payments : loan_payments = 300000)
  (h_shares : total_shares = 1600) :
  ((net_profit - loan_payments) / total_shares) = 274 := by
  sorry

theorem director_dividends (dividend_per_share : ℕ) (share_percentage : ℚ) (total_shares : ℕ)
  (h_dividend_per_share : dividend_per_share = 274)
  (h_percentage : share_percentage = 0.35)
  (h_shares : total_shares = 1600) :
  (dividend_per_share * share_percentage * total_shares).toNat = 153440 := by
  sorry

end net_profit_loan_payments_dividends_per_share_director_dividends_l348_348518


namespace factoring_difference_of_squares_l348_348484

theorem factoring_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := 
sorry

end factoring_difference_of_squares_l348_348484


namespace largest_integer_satisfying_conditions_l348_348944

theorem largest_integer_satisfying_conditions (n : ℤ) (m : ℤ) :
  n^2 = (m + 1)^3 - m^3 ∧ ∃ k : ℤ, 2 * n + 103 = k^2 → n = 313 := 
by 
  sorry

end largest_integer_satisfying_conditions_l348_348944


namespace ratio_of_areas_l348_348496

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l348_348496


namespace problem_l348_348252

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l348_348252


namespace domain_of_f_2x_l348_348217

theorem domain_of_f_2x (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x < 0 → ∃ y, f(x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x < (1 : ℝ)/2 → ∃ y, f(2 * x) = y) :=
by
  sorry

end domain_of_f_2x_l348_348217


namespace number_of_girls_with_no_pets_l348_348284

-- Define total number of students
def total_students : ℕ := 30

-- Define the fraction of boys in the class
def fraction_boys : ℚ := 1 / 3

-- Define the percentages of girls with pets
def percentage_girls_with_dogs : ℚ := 0.40
def percentage_girls_with_cats : ℚ := 0.20

-- Calculate the number of boys
def number_of_boys : ℕ := (fraction_boys * total_students).toNat

-- Calculate the number of girls
def number_of_girls : ℕ := total_students - number_of_boys

-- Calculate the number of girls who own dogs
def number_of_girls_with_dogs : ℕ := (percentage_girls_with_dogs * number_of_girls).toNat

-- Calculate the number of girls who own cats
def number_of_girls_with_cats : ℕ := (percentage_girls_with_cats * number_of_girls).toNat

-- Define the statement to be proved
theorem number_of_girls_with_no_pets : number_of_girls - (number_of_girls_with_dogs + number_of_girls_with_cats) = 8 := by
  sorry

end number_of_girls_with_no_pets_l348_348284


namespace B_knit_time_l348_348452

theorem B_knit_time (x : ℕ) (hA : 3 > 0) (h_combined_rate : 1/3 + 1/x = 1/2) : x = 6 := sorry

end B_knit_time_l348_348452


namespace minimum_employees_needed_l348_348781

-- Conditions
def water_monitors : ℕ := 95
def air_monitors : ℕ := 80
def soil_monitors : ℕ := 45
def water_and_air : ℕ := 30
def air_and_soil : ℕ := 20
def water_and_soil : ℕ := 15
def all_three : ℕ := 10

-- Theorems/Goals
theorem minimum_employees_needed 
  (water : ℕ := water_monitors)
  (air : ℕ := air_monitors)
  (soil : ℕ := soil_monitors)
  (water_air : ℕ := water_and_air)
  (air_soil : ℕ := air_and_soil)
  (water_soil : ℕ := water_and_soil)
  (all_3 : ℕ := all_three) :
  water + air + soil - water_air - air_soil - water_soil + all_3 = 165 :=
by
  sorry

end minimum_employees_needed_l348_348781


namespace dividends_CEO_2018_l348_348527

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l348_348527


namespace paint_cost_l348_348035

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l348_348035


namespace find_x_l348_348993

-- Definitions for the problem
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem find_x (x : ℝ) (h : ∃ k : ℝ, a x = k • b) : x = -1/2 := by
  sorry

end find_x_l348_348993


namespace k_value_opposite_solutions_l348_348066

theorem k_value_opposite_solutions (k x1 x2 : ℝ) 
  (h1 : 3 * (2 * x1 - 1) = 1 - 2 * x1)
  (h2 : 8 - k = 2 * (x2 + 1))
  (opposite : x2 = -x1) :
  k = 7 :=
by sorry

end k_value_opposite_solutions_l348_348066


namespace distance_from_blast_site_l348_348439

-- Declaring the conditions as constants
constant t_heard : Nat := 30 * 60 + 24  -- Time when the man heard the second blast in seconds
constant t_actual : Nat := 30 * 60      -- Time when the second blast actually occurred in seconds
constant speed_of_sound : Nat := 330    -- Speed of sound in meters per second

-- Main theorem statement
theorem distance_from_blast_site (t_heard t_actual speed_of_sound : Nat) : 
  t_heard = t_actual + 24 → 
  (speed_of_sound > 0) → 
  let distance := speed_of_sound * (t_heard - t_actual) in
  distance = 7920 :=
by
  sorry

end distance_from_blast_site_l348_348439


namespace ceo_dividends_correct_l348_348531

-- Definitions of parameters
def revenue := 2500000
def expenses := 1576250
def tax_rate := 0.2
def monthly_loan_payment := 25000
def months := 12
def number_of_shares := 1600
def ceo_ownership := 0.35

-- Calculation functions based on conditions
def net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
def loan_payments := monthly_loan_payment * months
def dividends_per_share := (net_profit - loan_payments) / number_of_shares
def ceo_dividends := dividends_per_share * ceo_ownership * number_of_shares

-- Statement to prove
theorem ceo_dividends_correct : ceo_dividends = 153440 :=
by 
  -- skipping the proof
  sorry

end ceo_dividends_correct_l348_348531


namespace selling_price_after_reductions_l348_348394

variable (a : ℝ)

theorem selling_price_after_reductions (a : ℝ) : 
  let p := (a - 10) * 0.90 in 
  p = (a - 10) * 0.90 :=
by
  sorry

end selling_price_after_reductions_l348_348394


namespace find_f_of_2_l348_348972

-- Define the conditions of the problem
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def f (x : ℝ) : ℝ := if x < 0 then x^2 - 1 else 0

-- State the theorem
theorem find_f_of_2 (h1 : is_odd_function f) (h2 : ∀ x, x < 0 → f x = x^2 - 1) : f 2 = -3 :=
by
  sorry

end find_f_of_2_l348_348972


namespace expected_value_of_Z_l348_348700

variable (δ : ℝ) (Z : ℕ)
variable (x : ℕ → ℝ)

-- Conditions
axiom x0_def : x 0 = 1
axiom δ_def : 0 < δ ∧ δ < 1
axiom x_iter : ∀ n : ℕ, x (n + 1) ∈ set.Icc 0 (x n)
axiom Z_def : Z = Nat.find (λ n, x n < δ)

-- Question: expected value of Z
theorem expected_value_of_Z : 
  (∑' (n : ℕ), (1 / (real.log (n + 1))) / (n + 1) ) / (1 - real.log δ) = 1 - real.log δ :=
sorry

end expected_value_of_Z_l348_348700


namespace common_difference_is_7_l348_348660

-- Define the arithmetic sequence with common difference d
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the conditions
variables (a1 d : ℕ)

-- Define the conditions provided in the problem
def condition1 := (arithmetic_seq a1 d 3) + (arithmetic_seq a1 d 6) = 11
def condition2 := (arithmetic_seq a1 d 5) + (arithmetic_seq a1 d 8) = 39

-- Prove that the common difference d is 7
theorem common_difference_is_7 : condition1 a1 d → condition2 a1 d → d = 7 :=
by
  intros cond1 cond2
  sorry

end common_difference_is_7_l348_348660


namespace quadratic_has_two_distinct_real_roots_find_m_and_other_root_l348_348584

-- Part (1)
theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
    let Δ := (m-3)^2 + 4 * m in
    Δ > 0 :=
by
  sorry

-- Part (2)
theorem find_m_and_other_root (x : ℝ) (m : ℝ) (hx : x = 1)
    (h : x^2 - (m-3)*x - m = 0) : 
    m = 2 ∧ ∃ y : ℝ, y ≠ x ∧ y^2 - (m-3)*y - m = 0 :=
by
  have hm : m = 2,
  sorry
  have hy : ∃ y : ℝ, y ≠ x ∧ y^2 - (m-3)*y - m = 0,
  sorry
  exact ⟨hm, hy⟩

end quadratic_has_two_distinct_real_roots_find_m_and_other_root_l348_348584


namespace partition_exists_iff_l348_348145

def partitionable (X : Finset ℕ) : Prop :=
  ∃ A B : Finset ℕ, A ∪ B = X ∧ A ∩ B = ∅ ∧ (A.sum id) = (B.sum id)

theorem partition_exists_iff (k : ℕ) (h : 0 < k) :
  partitionable (Finset.range (k + 1) + 1990) ↔ (k % 4 = 0 ∨ k % 4 = 3) := 
sorry

end partition_exists_iff_l348_348145


namespace solution_set_line_l348_348735

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l348_348735


namespace power_func_value_at_neg_one_l348_348389

-- Let f be a power function (x^alpha) that passes through (2, 8).
def f (x : ℝ) : ℝ := x ^ 3 -- Definition based on solving for α in the solution

theorem power_func_value_at_neg_one : f (-1) = -1 :=
by
  -- Assume f(x) = x^alpha and f(2) = 8 implies 2^alpha = 8, thus, α = 3 from the solution.
  -- So f(x) = x^3, therefore f(-1) = (-1)^3 = -1.
  rw [f, (-1 : ℝ) ^ 3]
  exact neg_one_pow 3
  sorry

end power_func_value_at_neg_one_l348_348389


namespace cylinder_radius_eq_3_l348_348928

theorem cylinder_radius_eq_3 (r : ℝ) : 
  (π * (r + 4)^2 * 3 = π * r^2 * 11) ∧ (r >= 0) → r = 3 :=
by 
  sorry

end cylinder_radius_eq_3_l348_348928


namespace simplified_radical_formula_l348_348534

theorem simplified_radical_formula (y : ℝ) (hy : 0 ≤ y):
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) :=
by
  sorry

end simplified_radical_formula_l348_348534


namespace dividends_CEO_2018_l348_348526

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l348_348526


namespace paperclips_volume_75_l348_348088

noncomputable def paperclips (v : ℝ) : ℝ := 60 / Real.sqrt 27 * Real.sqrt v

theorem paperclips_volume_75 :
  paperclips 75 = 100 :=
by
  sorry

end paperclips_volume_75_l348_348088


namespace length_EQ_l348_348305

-- Define the square EFGH with side length 8
def square_EFGH (a : ℝ) (b : ℝ): Prop := a = 8 ∧ b = 8

-- Define the rectangle IJKL with IL = 12 and JK = 8
def rectangle_IJKL (l : ℝ) (w : ℝ): Prop := l = 12 ∧ w = 8

-- Define the perpendicularity of EH and IJ
def perpendicular_EH_IJ : Prop := true

-- Define the shaded area condition
def shaded_area_condition (area_IJKL : ℝ) (shaded_area : ℝ): Prop :=
  shaded_area = (1/3) * area_IJKL

-- Theorem to prove
theorem length_EQ (a b l w area_IJKL shaded_area EH HG HQ EQ : ℝ):
  square_EFGH a b →
  rectangle_IJKL l w →
  perpendicular_EH_IJ →
  shaded_area_condition area_IJKL shaded_area →
  HQ * HG = shaded_area →
  EQ = EH - HQ →
  EQ = 4 := by
  intros hSquare hRectangle hPerpendicular hShadedArea hHQHG hEQ
  sorry

end length_EQ_l348_348305


namespace solve_for_x_l348_348765

theorem solve_for_x (x : ℤ) : (16 : ℝ) ^ (3 * x - 5) = ((1 : ℝ) / 4) ^ (2 * x + 6) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l348_348765


namespace fraction_water_by_volume_l348_348655

theorem fraction_water_by_volume
  (A W : ℝ) 
  (h1 : A / W = 0.5)
  (h2 : A / (A + W) = 1/7) : 
  W / (A + W) = 2/7 :=
by
  sorry

end fraction_water_by_volume_l348_348655


namespace percentage_enclosed_by_pentagons_l348_348018

-- Define the condition for the large square and smaller squares.
def large_square_area (b : ℝ) : ℝ := (4 * b) ^ 2

-- Define the condition for the number of smaller squares forming pentagons.
def pentagon_small_squares : ℝ := 10

-- Define the total number of smaller squares within a large square.
def total_small_squares : ℝ := 16

-- Prove that the percentage of the plane enclosed by pentagons is 62.5%.
theorem percentage_enclosed_by_pentagons :
  (pentagon_small_squares / total_small_squares) * 100 = 62.5 :=
by 
  -- The proof is left as an exercise.
  sorry

end percentage_enclosed_by_pentagons_l348_348018


namespace vanyas_password_valid_l348_348055

-- Define properties of the password and constraints
def has_no_repeating_digits (s : String) : Prop :=
  s.toList.nodup

def has_no_zeros (s : String) : Prop :=
  '0' ∉ s.toList

def no_invalid_jumps (s : String) : Prop :=
  ∀ (i j : ℕ), i < j → j < s.length →
  let d_i := s.get i
      d_j := s.get j in 
  ∀ (m : ℕ), (d_i, d_j) ≠ ('1', '3') ∧ (d_i, d_j) ≠ ('3', '1') ∧
              (d_i, d_j) ≠ ('1', '7') ∧ (d_i, d_j) ≠ ('7', '1') ∧ -- and so on for all invalid connections
              (d_i, d_j) ≠ ('3', '9') ∧ (d_i, d_j) ≠ ('9', '3')

def no_self_intersections (s : String) : Prop :=
  true -- Placeholder for a more detailed intersection check if necessary

def unique_sequence_except_reverse (s : String) : Prop :=
  ∀ (t : String), t ≠ s.reverse → ∀ (i : ℕ), (i < s.length ∧ i < t.length) → s.get i ≠ t.get i

-- Define Vanya's problem
def vanyas_password (s : String) : Prop :=
  has_no_repeating_digits s ∧
  has_no_zeros s ∧
  no_invalid_jumps s ∧
  no_self_intersections s ∧
  unique_sequence_except_reverse s

-- State the proof problem
theorem vanyas_password_valid : vanyas_password "12769" :=
by {
  -- proof steps go here
  sorry -- Placeholder for the actual proof
}

end vanyas_password_valid_l348_348055


namespace dummies_remainder_l348_348931

/-
  Prove that if the number of Dummies in one bag is such that when divided among 10 kids, 3 pieces are left over,
  then the number of Dummies in four bags when divided among 10 kids leaves 2 pieces.
-/
theorem dummies_remainder (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := 
by {
  sorry
}

end dummies_remainder_l348_348931


namespace trench_dig_time_l348_348187

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l348_348187


namespace interest_rate_approximation_l348_348264

-- Define the conditions
def initial_investment : ℝ := 8000
def final_investment : ℝ := 32000
def years : ℝ := 36

-- The rule of 70 for doubling time based on interest rate r
def doubling_time_by_rule_of_70 (r : ℝ) : ℝ := 70 / r

-- Prove that the interest rate r is approximately 3.89 percent under the given conditions.
theorem interest_rate_approximation (r : ℝ) (H : final_investment = initial_investment * 2^2 ∧ doubling_time_by_rule_of_70 r = years / 2) : r ≈ 3.89 :=
by 
  sorry

end interest_rate_approximation_l348_348264


namespace arithmetic_sequence_sum_n_l348_348408

noncomputable def find_n (Sn : ℕ → ℕ) (S4 : ℕ) (Sn_minus_4 : ℕ) (Sn_n : ℕ) : ℕ :=
  let S4 := 20
  let Sn_minus_4 := 60
  let Sn_n := 120
  let n := 12
  n

theorem arithmetic_sequence_sum_n (Sn : ℕ → ℕ) (S4 Sn_minus_4 Sn_n n : ℕ) :
  S4 = 20 → Sn_minus_4 = 60 → Sn_n = 120 → find_n Sn S4 Sn_minus_4 Sn_n = 12 :=
by {
  intros h1 h2 h3,
  sorry
}

end arithmetic_sequence_sum_n_l348_348408


namespace ratio_of_areas_l348_348493

theorem ratio_of_areas 
  (s1 s2 : ℝ)
  (A_large A_small A_trapezoid : ℝ)
  (h1 : s1 = 10)
  (h2 : s2 = 5)
  (h3 : A_large = (sqrt 3 / 4) * s1^2)
  (h4 : A_small = (sqrt 3 / 4) * s2^2)
  (h5 : A_trapezoid = A_large - A_small) :
  (A_small / A_trapezoid = 1 / 3) :=
sorry

end ratio_of_areas_l348_348493


namespace remainder_1488_1977_mod_500_l348_348836

theorem remainder_1488_1977_mod_500 :
  (1488 * 1977) % 500 = 276 :=
by
  have h1 : 1488 % 500 = -12 % 500, from sorry,
  have h2 : 1977 % 500 = -23 % 500, from sorry,
  calc
    (1488 * 1977) % 500
        = ((-12 % 500) * (-23 % 500)) % 500 : by rw [h1, h2]
    ... = (276 % 500) : by norm_num
    ... = 276 : by norm_num

end remainder_1488_1977_mod_500_l348_348836


namespace trench_dig_time_l348_348188

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l348_348188


namespace simplify_expression_l348_348870

theorem simplify_expression :
  (1 / 2^2 + (2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107 / 84 :=
by
  -- Skip the proof
  sorry

end simplify_expression_l348_348870


namespace value_of_x_plus_y_l348_348254

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l348_348254


namespace angle_ACE_eq_2_angle_EDB_l348_348079

noncomputable def rectangle_ABCD (A B C D E : Point) :=
  (E ∈ Segment A B) ∧ (Angle BCA = 90) ∧ (Angle BAD = 90)

open Real

theorem angle_ACE_eq_2_angle_EDB {A B C D E : Point}
  (h1 : distance A B = (sqrt 5 + 1) / 2)
  (h2 : distance B C = 1)
  (h3 : distance A E = 1)
  (h4 : E ∈ Segment A B)
  (h5 : rectangle_ABCD A B C D E) : angle A C E = 2 * angle E D B :=
sorry

end angle_ACE_eq_2_angle_EDB_l348_348079


namespace first_cube_weight_l348_348089

-- Given definitions of cubes and their relationships
def weight_of_cube (s : ℝ) (weight : ℝ) : Prop :=
  ∃ v : ℝ, v = s^3 ∧ weight = v

def cube_relationship (s1 s2 weight2 : ℝ) : Prop :=
  s2 = 2 * s1 ∧ weight2 = 32

-- The proof problem
theorem first_cube_weight (s1 s2 weight1 weight2 : ℝ) (h1 : cube_relationship s1 s2 weight2) : weight1 = 4 :=
  sorry

end first_cube_weight_l348_348089


namespace chord_intersection_probability_is_small_l348_348169

-- Given five distinct points chosen from 2023 evenly spaced points on a circle,
-- we aim to show that the probability that the chord AB intersects the chord CD,
-- but neither intersects with the chord DE, is very small but non-zero.

noncomputable def probability_intersection (n : ℕ) [Fact (n = 2023)] : ℚ :=
    let points := finset.range n
    let quintuples := finset.powersetLen 5 points
    -- Additional definitions would go here, such as chosen chords and intersections

theorem chord_intersection_probability_is_small (h : Fact (2023 % 2 = 1)) :
    0 < probability_intersection 2023 ∧ probability_intersection 2023 < 1 / 1000 := -- Arbitrarily small threshold
    sorry

end chord_intersection_probability_is_small_l348_348169


namespace cupcakes_count_l348_348425

-- Definitions of given conditions as Lean variables.
variables (C : ℕ) -- Number of cupcakes Wendy baked.
variables (cookies : ℕ) (pastries_left : ℕ) (pastries_sold : ℕ)

-- Assigning values according to the conditions in problem statement
def cookies := 29
def pastries_left := 24
def pastries_sold := 9

-- Stating the theorem
theorem cupcakes_count :
  (C + cookies - pastries_sold = pastries_left) →
  C = 4 :=
by
  intros h1
  sorry

end cupcakes_count_l348_348425


namespace shortest_wire_length_l348_348420

/--
Two cylindrical poles with diameters of 8 inches and 24 inches respectively, are placed side by side and bound together with a wire. Prove that the length of the shortest wire that will go around both poles is \(16\sqrt{3} + \frac{28\pi}{3}\) inches.
-/
theorem shortest_wire_length {d1 d2 : ℝ} (h1 : d1 = 8) (h2 : d2 = 24) :
  let r1 := d1 / 2;
  let r2 := d2 / 2;
  let distance_centres := r1 + r2;
  let distance_radii := r2 - r1;
  let straight_section := 2 * real.sqrt (distance_centres^2 - distance_radii^2);
  let total_curved_section := (2 * real.pi * r1 / 6) + (2 * real.pi * r2 / 3)
  in straight_section + total_curved_section = 16 * real.sqrt 3 + 28 * real.pi / 3 := 
sorry

end shortest_wire_length_l348_348420


namespace cos_alpha_value_cos_beta_value_l348_348608

open Real

-- Define the conditions and the corresponding Lean 4 proof problems.
theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Ioo (π / 2) π)
  (h2 : sin (α / 2) + cos (α / 2) = sqrt 6 / 2) : 
  cos α = -sqrt 3 / 2 := 
sorry

theorem cos_beta_value (α β : ℝ) (h1 : α ∈ Ioo (π / 2) π)
  (h2 : sin (α / 2) + cos (α / 2) = sqrt 6 / 2)
  (h3 : sin (α - β) = -3 / 5)
  (h4 : β ∈ Ioo (π / 2) π) : 
  cos β = -(4 * sqrt 3 + 3) / 10 := 
sorry

end cos_alpha_value_cos_beta_value_l348_348608


namespace number_of_good_subsets_l348_348914

-- Define the set S
def S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define what constitutes a "good subset"
def is_good_subset (subset : Set ℕ) : Prop :=
  subset.Nonempty ∧ (subset.filter Even).card ≥ (subset.filter Odd).card

-- The main statement that we want to prove
theorem number_of_good_subsets :
  Fintype.card {subset : Set ℕ // subset ⊆ S ∧ is_good_subset subset} = 637 :=
sorry

end number_of_good_subsets_l348_348914


namespace number_of_real_solutions_l348_348946

theorem number_of_real_solutions : 
  (∃ x : ℝ, 2^(x^2 - 3*x + 2) - 1 = 0) ∧ 
  (∃ y : ℝ, 2^(y^2 - 3*y + 2) - 1 = 0) ∧ 
  (x ≠ y) ∧ 
  ∀ z : ℝ, 2^(z^2 - 3*z + 2) - 1 = 0 → (z = x ∨ z = y) :=
sorry

end number_of_real_solutions_l348_348946


namespace solution_set_of_inequality_l348_348405

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = set.Ioo (-2) 1 :=
sorry

end solution_set_of_inequality_l348_348405


namespace perpendicular_vectors_dot_product_zero_l348_348240

theorem perpendicular_vectors_dot_product_zero (m : ℤ) : 
  let a := (3 : ℤ, 1 : ℤ)
  let b := (2 : ℤ, m)
  (a.1 * b.1 + a.2 * b.2 = 0) -> (m = -6) :=
by 
  let a := (3, 1)
  let b := (2, m)
  intro h
  have : 3 * 2 + 1 * m = 0 := h
  simp at this
  linarith this

end perpendicular_vectors_dot_product_zero_l348_348240


namespace find_possible_k_l348_348596

def f (a : ℝ) (x : ℝ) : ℝ := a * 4^x - a * 2^(x+1) + 1
def g (k : ℝ) (x : ℝ) : ℝ := k * 2^x

theorem find_possible_k (a k : ℝ) (h_a : a > 0) (b : ℝ) :
  (∀ x ∈ set.Icc 1 2, f a x = k * 2^x) →
  (∃ x ∈ set.Icc 1 2, f a x = 9) →
  (∃ x ∈ set.Icc 1 2, f a x = 1) →
  (∃ x1 x2 ∈ set.Icc (-1) 2, f a x1 = g k x1 ∧ f a x2 = g k x2 ∧ x1 ≠ x2) →
  (k = 1/4 ∨ k = 1/2) :=
sorry

end find_possible_k_l348_348596


namespace mall_entry_exit_l348_348102

theorem mall_entry_exit (n : ℕ) (h : n = 4) : 
  (∃ ways_to_enter ways_to_exit, ways_to_enter = n ∧ ways_to_exit = n - 1 ∧ ways_to_enter * ways_to_exit = 12) := 
by {
  use [4, 3],
  exact ⟨rfl, rfl, rfl⟩
}

end mall_entry_exit_l348_348102


namespace product_remainder_mod_5_l348_348536

theorem product_remainder_mod_5 : 
  let seq := list.range 20 in 
  ∀ (seq' : list (ℕ)) (h : seq' = seq.map (λ n, 10 * n + 4)), 
  ((seq'.prod) % 5) = 1 := 
by 
  let seq := list.range 20
  assume seq' h
  rw h
  sorry

end product_remainder_mod_5_l348_348536


namespace johns_new_weekly_earnings_l348_348688

-- Definition of the initial weekly earnings
def initial_weekly_earnings := 40

-- Definition of the percent increase in earnings
def percent_increase := 100

-- Definition for the final weekly earnings after the raise
def final_weekly_earnings (initial_earnings : Nat) (percentage : Nat) := 
  initial_earnings + (initial_earnings * percentage / 100)

-- Theorem stating John’s final weekly earnings after the raise
theorem johns_new_weekly_earnings : final_weekly_earnings initial_weekly_earnings percent_increase = 80 :=
  by
  sorry

end johns_new_weekly_earnings_l348_348688


namespace isosceles_triangle_angle_D_l348_348120

theorem isosceles_triangle_angle_D (y : ℝ) (h1 : ∠D = 3*y)
  (h2 : ∠E = 3*y) (h3 : ∠F = y)
  (h4 : ∠E = ∠D) (h5 : ∠D + ∠E + ∠F = 180) : ∠D = 540 / 7 :=
by
sory

end isosceles_triangle_angle_D_l348_348120


namespace largest_sphere_radius_on_torus_l348_348588

theorem largest_sphere_radius_on_torus
  (inner_radius outer_radius : ℝ)
  (torus_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (sphere_radius : ℝ)
  (sphere_center : ℝ × ℝ × ℝ) :
  inner_radius = 3 →
  outer_radius = 5 →
  torus_center = (4, 0, 1) →
  circle_radius = 1 →
  sphere_center = (0, 0, sphere_radius) →
  sphere_radius = 4 :=
by
  intros h_inner_radius h_outer_radius h_torus_center h_circle_radius h_sphere_center
  sorry

end largest_sphere_radius_on_torus_l348_348588


namespace problem_l348_348251

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l348_348251


namespace angle_ratios_l348_348361

variables {A B C D : Type} [EuclideanGeometry A B C D]

noncomputable def AD_is_median (A B C D : Type) [EuclideanGeometry A B C D] (triangle : Triangle A B C) (D_is_midpoint : Midpoint B C D) : Prop :=
  Segment A D = MedianSegment A B C

theorem angle_ratios (A B C D : Type) [EuclideanGeometry A B C D]
  (triangle : Triangle A B C) (D_is_midpoint : Midpoint B C D) (length_ratio : Length AC = 2 * Length AD) :
  IsObtuse (Angle CAB) ∧ IsAcute (Angle DAB) :=
  by
    sorry

end angle_ratios_l348_348361


namespace relationship_f_pow_l348_348344

variable {a b c : ℝ}
variable {x : ℝ}

def f (x : ℝ) := a * x^2 + b * x + c

theorem relationship_f_pow (h_pos : a > 0) (h_sym : ∀ x, f (1 - x) = f (1 + x)) (h_x_pos : x > 0) :
  f (3 ^ x) > f (2 ^ x) :=
sorry

end relationship_f_pow_l348_348344


namespace limit_r_as_m_zero_eq_1_over_2sqrt3_l348_348938

-- Define the function L (intersections points) and r as given in the conditions
def L (m : ℝ) : ℝ := -Real.sqrt ((m + 4) / 3)
def r (m : ℝ) : ℝ := (L (-m) - L (m)) / m

-- State the problem as a theorem in Lean 4
theorem limit_r_as_m_zero_eq_1_over_2sqrt3 : 
  filter.tendsto (r) (nhds 0) (nhds (1 / (2 * Real.sqrt 3))) :=
sorry

end limit_r_as_m_zero_eq_1_over_2sqrt3_l348_348938


namespace sequence_formula_l348_348403

open Nat

def sequence : ℕ → ℕ
| 0       => 1
| (n + 1) => 3 * (sequence n) + 2^n

theorem sequence_formula (n : ℕ) : sequence n = 3^n - 2^n :=
sorry

end sequence_formula_l348_348403


namespace length_of_pipe_is_correct_l348_348110

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end length_of_pipe_is_correct_l348_348110


namespace sara_picked_peaches_l348_348359

theorem sara_picked_peaches (initial_peaches : ℕ) (final_peaches : ℕ) (initial_pears : ℕ) 
  (h_initial_peaches : initial_peaches = 24) 
  (h_initial_pears : initial_pears = 37) 
  (h_final_peaches : final_peaches = 61) : 
  final_peaches - initial_peaches = 37 :=
by
  rw [h_initial_peaches, h_final_peaches]
  norm_num

end sara_picked_peaches_l348_348359


namespace number_of_girls_with_no_pet_l348_348288

-- Definitions based on the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def percentage_girls_own_dogs : ℚ := 40 / 100
def percentage_girls_own_cats : ℚ := 20 / 100

-- Prove that the number of girls with no pets is 8
theorem number_of_girls_with_no_pet :
  let girls := total_students * (1 - fraction_boys),
      percentage_girls_no_pets := 1 - percentage_girls_own_dogs - percentage_girls_own_cats,
      girls_with_no_pets := girls * percentage_girls_no_pets
  in girls_with_no_pets = 8 := by
{
  sorry
}

end number_of_girls_with_no_pet_l348_348288


namespace salary_for_may_l348_348376

theorem salary_for_may
  (J F M A May : ℝ)
  (h1 : J + F + M + A = 32000)
  (h2 : F + M + A + May = 34400)
  (h3 : J = 4100) :
  May = 6500 := 
by 
  sorry

end salary_for_may_l348_348376


namespace trench_dig_time_l348_348189

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l348_348189


namespace negative_rearrangement_l348_348543

theorem negative_rearrangement {n k : ℕ} (arrangement_n: fin n → ℤ) (arrangement_2n1: fin (2^n - 1) → ℤ)
  (h_n: ∀ i, arrangement_n i = -1) (h_return: (steps: ℕ) (steps = k) → (∀ j, arrangement_n j = -1)) 
  : (steps_2n1: ℕ) (steps_2n1 = (2^k - 1)) → (∀ l, arrangement_2n1 l = -1) := 
begin
  sorry
end

end negative_rearrangement_l348_348543


namespace problem1_problem2_l348_348532

-- Define the real number domain
variables (a : ℝ)

-- Problem 1: Prove {{(-5)}^{-2}} + 2{{(\pi -1)}^{0}} = 2 \frac{1}{25}
theorem problem1 : (-5 : ℝ) ^ (-2) + 2 * ((real.pi - 1 : ℝ) ^ (0 : ℕ)) = 2 * (1 / 25) :=
by sorry

-- Problem 2: Prove {{(2a+1)}^{2}} - (2a+1)(-1+2a) = 4a + 2
theorem problem2 : (2 * a + 1) ^ 2 - (2 * a + 1) * (-1 + 2 * a) = 4 * a + 2 :=
by sorry

end problem1_problem2_l348_348532


namespace min_students_l348_348774

noncomputable def num_boys_min (students : ℕ) (girls : ℕ) : Prop :=
  ∃ (boys : ℕ), boys > (3 * girls / 2) ∧ students = boys + girls

theorem min_students (girls : ℕ) (h_girls : girls = 5) : ∃ n, num_boys_min n girls ∧ n = 13 :=
by
  use 13
  unfold num_boys_min
  use 8
  sorry

end min_students_l348_348774


namespace cannot_form_right_triangle_l348_348309

theorem cannot_form_right_triangle :
  ¬(∃ (a b c : ℝ), {a, b, c} = {4, 5, 7} ∧ a^2 + b^2 = c^2) :=
by
  sorry

end cannot_form_right_triangle_l348_348309


namespace difference_of_fractions_l348_348058

theorem difference_of_fractions (a b : ℕ) (h₁ : a = 280) (h₂ : b = 99) : a - b = 181 := by
  have h₁ : (7 / 8 : ℝ) * 320 = 280 := by norm_num
  have h₂ : (11 / 16 : ℝ) * 144 = 99 := by norm_num
  calc
    a - b = 280 - 99 := by rw [h₁, h₂]
        ... = 181 := by norm_num

end difference_of_fractions_l348_348058


namespace annual_decrease_rate_l348_348019

theorem annual_decrease_rate (r : ℝ) 
  (h1 : 15000 * (1 - r / 100)^2 = 9600) : 
  r = 20 := 
sorry

end annual_decrease_rate_l348_348019


namespace average_age_all_groups_l348_348029

theorem average_age_all_groups 
  (a b c : ℕ) 
  (A B C : ℕ) 
  (hA : A = 34 * a) 
  (hB : B = 25 * b) 
  (hC : C = 45 * c) 
  (hAB : A + B = 30 * (a + b)) 
  (hAC : A + C = 42 * (a + c)) 
  (hBC : B + C = 35 * (b + c)) :
  (A + B + C) / (a + b + c) = 26.07 := by
  sorry

end average_age_all_groups_l348_348029


namespace area_BCD_l348_348328

variables {A B C D : Type*}
variables (b c d θ x y z : Real)
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]

-- Assuming points in Cartesian coordinates
-- Let A = (0, 0, 0), B = (b, 0, 0), C = (0, c, 0), D = (b * cos θ, c * cos θ, d * sin θ)
-- Representing these as vectors
noncomputable def pointA : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def pointB : ℝ × ℝ × ℝ := (b, 0, 0)
noncomputable def pointC : ℝ × ℝ × ℝ := (0, c, 0)
noncomputable def pointD : ℝ × ℝ × ℝ := (b * Real.cos θ, c * Real.cos θ, d * Real.sin θ)

-- Areas of triangles ABC, ACD, ADB
noncomputable def area_ABC := x
noncomputable def area_ACD := y
noncomputable def area_ADB := z

theorem area_BCD (h1 : b * c = 2 * x) 
                 (h2 : y = b * d * Real.sin θ / 2) 
                 (h3 : z = c * d * Real.sin θ / 2) 
                 : ∃ (a_BCD : Real), a_BCD = Real.sqrt (x^2 + y^2 + z^2) :=
begin
  sorry
end

end area_BCD_l348_348328


namespace y_satisfies_equation1_l348_348860

noncomputable def y (x : ℝ) : ℝ := (2 * x / (x^3 + 1)) + (1 / x)

theorem y_satisfies_equation1 (x : ℝ) (h : x ≠ 0) :
  x * (x^3 + 1) * (derivative y x) + (2 * x^3 - 1) * (y x) = (x^3 - 2) / x := sorry

end y_satisfies_equation1_l348_348860


namespace dot_product_result_l348_348983

variable (a b : EuclideanSpace ℝ (Fin 2))

noncomputable def angle_between (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.arccos ((a • b) / (∥a∥ * ∥b∥))

theorem dot_product_result
  (h_angle : angle_between a b = 2/3 * Real.pi)
  (h_norm_a : ∥a∥ = 1)
  (h_norm_b : ∥b∥ = 2) :
  a • (a + b) = 0 :=
  sorry

end dot_product_result_l348_348983


namespace area_of_triangle_AEB_l348_348300

-- Definitions for the problem
def right_triangle (A B D : Type) [is_right_angle A B D] : Prop := 
    dist A B = 8 ∧ dist A D = 6

def points_on_BD (B D F G: Type) : Prop := 
    dist D F = 2 ∧ dist B G = 3

noncomputable def area_triangle_AEB (A B E: Type) : ℝ := 
    1 / 2 * dist A E * dist B E

-- Theorem statement
theorem area_of_triangle_AEB 
    (A B D F G E : Type)
    [right_triangle A B D] 
    [points_on_BD B D F G] 
    [intersection AF AG = E] :
    area_triangle_AEB A B E = 24 := 
by 
  sorry

end area_of_triangle_AEB_l348_348300


namespace count_cyclic_quadrilaterals_l348_348957

def is_cyclic {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] (points : set α) : Prop :=
∃ (O : α) (r : ℝ), ∀ (P ∈ points), dist O P = r

def square {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] : set α := sorry
def rectangle_not_square {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] : set α := sorry
def kite_not_rhombus {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] : set α := sorry
def general_quadrilateral {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] : set α := sorry
def equilateral_trapezoid_not_parallelogram {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] : set α := sorry

theorem count_cyclic_quadrilaterals :
  (∃ (f : (nat → (set point))),
    ∀ n, 
    (f n = square ∨ f n = rectangle_not_square ∨ 
     f n = kite_not_rhombus ∨ f n = general_quadrilateral ∨ 
     f n = equilateral_trapezoid_not_parallelogram) ∧ 
    (is_cyclic (f n) → n < 3)) :=
sorry

end count_cyclic_quadrilaterals_l348_348957


namespace arithmetic_series_sum_l348_348951

theorem arithmetic_series_sum :
  let a1 := -39
  let d := 2
  let an := -1
  let n := ((an - a1) / d).to_nat + 1
  let sum := (n * (a1 + an)) / 2
  sum = -400 := by
  sorry

end arithmetic_series_sum_l348_348951


namespace solution_set_l348_348739

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l348_348739


namespace rhombus_diagonal_length_l348_348380

-- Define a rhombus with one diagonal of 10 cm and a perimeter of 52 cm.
theorem rhombus_diagonal_length (d : ℝ) 
  (h1 : ∃ a b c : ℝ, a = 10 ∧ b = d ∧ c = 13) -- The diagonals and side of rhombus.
  (h2 : 52 = 4 * c) -- The perimeter condition.
  (h3 : c^2 = (d/2)^2 + (10/2)^2) -- The relationship from Pythagorean theorem.
  : d = 24 :=
by
  sorry

end rhombus_diagonal_length_l348_348380


namespace angle_UWY_in_regular_octagon_l348_348832

theorem angle_UWY_in_regular_octagon
  (U W Y : Type)
  (is_regular_octagon : ∀ (x y : U), x ≠ y → x = W ∧ y = Y)
  : ∠ U W Y = 135 := 
sorry

end angle_UWY_in_regular_octagon_l348_348832


namespace general_formula_a_find_m_l348_348202
open Nat Real

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 1 else (sequence_a (n - 1) * (n / (n - 1)))

-- Add additional definitions based on conditions
def sum_sequence (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum S

def b_sequence (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (sum_sequence S n)

def sum_b_sequence (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum b

theorem general_formula_a (n : ℕ) :
  sequence_a n = n := sorry

theorem find_m :
  ∃ m : ℕ, (∀ n : ℕ, sum_b_sequence (b_sequence sequence_a) n < m / 10) ∧ m = 20 := sorry

end general_formula_a_find_m_l348_348202


namespace units_digit_sum_of_sequence_l348_348428

theorem units_digit_sum_of_sequence :
  let sequence := (List.range 10 ).map (λ n, (n+1)! + (n+1)) in
  let units_digit := λ n : ℕ, n % 10 in
  units_digit (sequence.sum) = 8 :=
by
  -- Definitions based on problem conditions
  let sequence := (List.range 10 ).map (λ n, (n+1)! + (n+1))
  let units_digit := λ n : ℕ, n % 10
  let individual_units_digits := [2, 4, 9, 8]
  let sum_units_digit_1_to_4 := (individual_units_digits.sum : ℕ) % 10
  have h1 : sum_units_digit_1_to_4 = 3 := by norm_num
  
  let sum_5_to_10 := (5 + 6 + 7 + 8 + 9 + 10) % 10
  have h2 : sum_5_to_10 = 5 := by norm_num
  
  have final_units_digit := (sum_units_digit_1_to_4 + sum_5_to_10) % 10
  have h3 : final_units_digit = 8 := by norm_num
  
  show units_digit (sequence.sum) = 8 from h3

end units_digit_sum_of_sequence_l348_348428


namespace ken_and_kendra_brought_home_l348_348696

-- Define the main variables
variables (ken_caught kendra_caught ken_brought_home : ℕ)

-- Define the conditions as hypothesis
def conditions :=
  kendra_caught = 30 ∧
  ken_caught = 2 * kendra_caught ∧
  ken_brought_home = ken_caught - 3

-- Define the problem to prove
theorem ken_and_kendra_brought_home :
  (ken_caught + kendra_caught = 87) :=
begin
  -- Unpacking the conditions for readability
  unfold conditions at *,
  sorry -- Proof will go here
end

end ken_and_kendra_brought_home_l348_348696


namespace cart_speed_l348_348879

theorem cart_speed :
  (diameter : ℝ) (clicks_per_second : ℕ) (π_approx : ℝ) (1 ≤ clicks_per_second) (diameter = 1) (π_approx = 3.14) :
  (speed : ℝ) (speed = π_approx * 3.6) → speed ≈ 11.3 :=
by
  intros diameter clicks_per_second π_approx h_clicks_per_second h_diameter h_π_approx speed h_speed
  rw [h_π_approx, h_speed]
  norm_num
  sorry

end cart_speed_l348_348879


namespace equal_distribution_possible_l348_348799

noncomputable def pitchers := Fin 10
def milk_distribution (m : pitchers → ℝ) : Prop := ∀ i : pitchers, 0 ≤ m i ∧ m i ≤ 0.1 * (∑ i, m i)
def operation (m : pitchers → ℝ) (i : pitchers) (k : ℝ) : pitchers → ℝ := 
  λ j, if j = i then (1 - k) * m i else m j + k / 9 * m i

theorem equal_distribution_possible 
  (m : pitchers → ℝ)
  (h : milk_distribution m) :
  ∃ ops : List pitchers, ops.length ≤ 10 ∧ 
  (∀ i, let new_m := List.foldl (λ m op, operation m op 1/10) m ops in
        new_m i = (∑ i, m i) / 10) :=
sorry

end equal_distribution_possible_l348_348799


namespace line_stabs_discs_l348_348572

def satisfies_conditions (n : ℕ) (D : Finset (Set (ℝ × ℝ))) (O : ℝ × ℝ) : Prop :=
  n ≥ 3 ∧
  (∀ d ∈ D, ¬ (O ∈ d)) ∧
  (∀ k : ℕ, 0 < k ∧ k < n → Finset.card (Finset.filter (λ d, (dist O (Set.center d) ≤ k + 1)) D) ≥ k)

noncomputable def number_of_discs_stabbed (n : ℕ) : ℝ :=
  2 * Real.log (n + 1) / π

theorem line_stabs_discs (n : ℕ) (D : Finset (Set (ℝ × ℝ))) (O : ℝ × ℝ) :
  satisfies_conditions n D O → ∃ ℓ : ℝ × ℝ → Prop, ∀ d ∈ D, ℓ O → Set.countable (ℓ ∩ d) ≥ number_of_discs_stabbed n :=
by sorry

end line_stabs_discs_l348_348572


namespace Bruce_grape_purchase_l348_348910

theorem Bruce_grape_purchase
  (G : ℕ)
  (total_paid : ℕ)
  (cost_per_kg_grapes : ℕ)
  (kg_mangoes : ℕ)
  (cost_per_kg_mangoes : ℕ)
  (total_mango_cost : ℕ)
  (total_grape_cost : ℕ)
  (total_amount : ℕ)
  (h1 : cost_per_kg_grapes = 70)
  (h2 : kg_mangoes = 10)
  (h3 : cost_per_kg_mangoes = 55)
  (h4 : total_paid = 1110)
  (h5 : total_mango_cost = kg_mangoes * cost_per_kg_mangoes)
  (h6 : total_grape_cost = G * cost_per_kg_grapes)
  (h7 : total_amount = total_mango_cost + total_grape_cost)
  (h8 : total_amount = total_paid) :
  G = 8 := by
  sorry

end Bruce_grape_purchase_l348_348910


namespace distance_from_E_to_plane_l348_348862

def point := ℝ × ℝ × ℝ

def A : point := (0, 4 / 3, 0)
def B : point := (0, 2, 2 / 3)
def G1 : point := (2, 0, 2)
def E : point := (0, 0, 0)

theorem distance_from_E_to_plane (A B G1 E : point)
  (hA : A = (0, 4 / 3, 0))
  (hB : B = (0, 2, 2 / 3))
  (hG1 : G1 = (2, 0, 2))
  (hE : E = (0, 0, 0)) :
  ∃ (d : ℝ), d = 2 * sqrt (2 / 11) := by
  sorry

end distance_from_E_to_plane_l348_348862


namespace correlated_relationships_l348_348903

-- Definitions for the conditions are arbitrary
-- In actual use cases, these would be replaced with real mathematical conditions
def great_teachers_produce_outstanding_students : Prop := sorry
def volume_of_sphere_with_radius : Prop := sorry
def apple_production_climate : Prop := sorry
def height_and_weight : Prop := sorry
def taxi_fare_distance_traveled : Prop := sorry
def crows_cawing_bad_omen : Prop := sorry

-- The final theorem statement
theorem correlated_relationships : 
  great_teachers_produce_outstanding_students ∧
  apple_production_climate ∧
  height_and_weight ∧
  ¬ volume_of_sphere_with_radius ∧ 
  ¬ taxi_fare_distance_traveled ∧ 
  ¬ crows_cawing_bad_omen :=
sorry

end correlated_relationships_l348_348903


namespace equation_has_three_solutions_l348_348960

theorem equation_has_three_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, f = λ x, x^3 + 6 * x^2 + a * x + 8 ∧
   (∀ f' : ℝ → ℝ, f' = λ x, 3 * x^2 + 12 * x + a ∧
   (∀ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3))) ↔
  a ∈ Iio (-15) :=
sorry

end equation_has_three_solutions_l348_348960


namespace solve_for_x_l348_348763

theorem solve_for_x (x : ℂ) (i : ℂ) (h : i ^ 2 = -1) (eqn : 3 + i * x = 5 - 2 * i * x) : x = i / 3 :=
sorry

end solve_for_x_l348_348763


namespace sum_of_ns_satisfying_log_condition_l348_348582

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 + 3 * x + 2) ^ real.cos (real.pi * x)

-- Define the problem
theorem sum_of_ns_satisfying_log_condition :
  let sum_log_condition (n : ℕ) := ∑ k in finset.range n, real.log10 (f k)
  ∑ n in finset.range 100, (if (abs (sum_log_condition n) = 1) then 1 else 0) = 2 ∧
  (finset.filter (λ n, abs (sum_log_condition n) = 1) (finset.range 100)).sum id = 21 :=
by
  sorry

end sum_of_ns_satisfying_log_condition_l348_348582


namespace integral_correct_l348_348126

noncomputable def integral_value : ℝ :=
  ∫ x in 0..2, (4 * sqrt (2 - x) - sqrt (x + 2)) / ((sqrt (x + 2) + 4 * sqrt (2 - x)) * (x + 2) ^ 2)

theorem integral_correct : integral_value = (1 / 2) * log 5 :=
by
  sorry

end integral_correct_l348_348126


namespace findNumberOfStudentsInGroupA_l348_348803

variable (A : ℕ) -- Number of students in group A

-- Defining the known conditions
def studentsInGroupB : ℕ := 50
def percentForgotHomeworkA : ℚ := 0.20
def percentForgotHomeworkB : ℚ := 0.12
def percentForgotHomeworkTotal : ℚ := 0.15

-- Defining the total number of students
def totalStudents : ℕ := A + studentsInGroupB

-- Number of students in group A who forgot their homework
def forgotHomeworkA : ℚ := percentForgotHomeworkA * A

-- Number of students in group B who forgot their homework
def forgotHomeworkB : ℚ := percentForgotHomeworkB * studentsInGroupB

-- Total number of students who forgot their homework
def forgotHomeworkTotal : ℚ := percentForgotHomeworkTotal * totalStudents

-- The main theorem to prove: find A such that the total number of students who forgot homework matches
theorem findNumberOfStudentsInGroupA (h : forgotHomeworkA + forgotHomeworkB = forgotHomeworkTotal) :
  A = 30 :=
sorry

end findNumberOfStudentsInGroupA_l348_348803


namespace flour_quantity_l348_348876

-- Define the recipe ratio of eggs to flour
def recipe_ratio : ℚ := 3 / 2

-- Define the number of eggs needed
def eggs_needed := 9

-- Prove that the number of cups of flour needed is 6
theorem flour_quantity (r : ℚ) (n : ℕ) (F : ℕ) 
  (hr : r = 3 / 2) (hn : n = 9) : F = 6 :=
by
  sorry

end flour_quantity_l348_348876


namespace roger_cookie_price_l348_348757

noncomputable def price_per_roger_cookie (A_cookies: ℕ) (A_price_per_cookie: ℕ) (A_area_per_cookie: ℕ) (R_cookies: ℕ) (R_area_per_cookie: ℕ): ℕ :=
  by
  let A_total_earnings := A_cookies * A_price_per_cookie
  let R_total_area := A_cookies * A_area_per_cookie
  let price_per_R_cookie := A_total_earnings / R_cookies
  exact price_per_R_cookie
  
theorem roger_cookie_price {A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie : ℕ}
  (h1 : A_cookies = 12)
  (h2 : A_price_per_cookie = 60)
  (h3 : A_area_per_cookie = 12)
  (h4 : R_cookies = 18) -- assumed based on area calculation 144 / 8 (we need this input to match solution context)
  (h5 : R_area_per_cookie = 8) :
  price_per_roger_cookie A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie = 40 :=
  by
  sorry

end roger_cookie_price_l348_348757


namespace problem_f_2023_l348_348504

noncomputable def f : ℝ → ℝ := λ x, if hx : 0 ≤ x ∧ x ≤ 1 then x^3 + 3*x else sorry

theorem problem_f_2023 : 
  (∀ x, f (1 + x) = f (1 - x)) → 
  (∀ x, f (-x) = -f x) → 
  f 2023 = -4 :=
by 
  intro h1 h2
  sorry

end problem_f_2023_l348_348504


namespace circle_equation_l348_348218

theorem circle_equation :
  ∃ x y : ℝ, x = 2 ∧ y = 0 ∧ ∀ (p q : ℝ), ((p - x)^2 + q^2 = 4) ↔ (p^2 + q^2 - 4 * p = 0) :=
sorry

end circle_equation_l348_348218


namespace find_x_l348_348995

def A : Set ℝ := {1, 5}
def B (x : ℝ) : Set ℝ := {x^2 - 3*x + 1, x^2 - 4*x + 5}

theorem find_x (x : ℝ) : (A ∪ B x).card = 3 ↔ x = -1 ∨ x = 2 ∨ x = 3 := 
by
  sorry

end find_x_l348_348995


namespace y_work_time_l348_348443

theorem y_work_time (x_days : ℕ) (x_work_time : ℕ) (y_work_time : ℕ) :
  x_days = 40 ∧ x_work_time = 8 ∧ y_work_time = 20 →
  let x_rate := 1 / 40
  let work_done_by_x := 8 * x_rate
  let remaining_work := 1 - work_done_by_x
  let y_rate := remaining_work / 20
  y_rate * 25 = 1 :=
by {
  sorry
}

end y_work_time_l348_348443


namespace abs_less_than_zero_impossible_l348_348902

theorem abs_less_than_zero_impossible (x : ℝ) : |x| < 0 → false :=
by
  sorry

end abs_less_than_zero_impossible_l348_348902


namespace radius_of_third_circle_l348_348419

theorem radius_of_third_circle (R r : ℝ) (hR_pos : 0 < R) (hr_pos : 0 < r) :
  ∃ (K : ℝ), K = 2 * r * R / (R + r) :=
by {
  use (2 * r * R / (R + r)),
  sorry
}

end radius_of_third_circle_l348_348419


namespace mr_mcpherson_needs_to_raise_840_l348_348722

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end mr_mcpherson_needs_to_raise_840_l348_348722


namespace total_numbers_l348_348375

theorem total_numbers (n : ℕ) (a : ℕ → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 25)
  (h2 : (a (n - 3) + a (n - 2) + a (n - 1)) / 3 = 35)
  (h3 : a 3 = 25)
  (h4 : (Finset.sum (Finset.range n) a) / n = 30) :
  n = 6 :=
sorry

end total_numbers_l348_348375


namespace base9_to_decimal_l348_348139

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end base9_to_decimal_l348_348139


namespace solution_set_line_l348_348734

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l348_348734


namespace problem1_inequality_problem2_inequality_l348_348369

theorem problem1_inequality (x : ℝ) (h1 : 2 * x + 10 ≤ 5 * x + 1) (h2 : 3 * (x - 1) > 9) : x > 4 := sorry

theorem problem2_inequality (x : ℝ) (h1 : 3 * (x + 2) ≥ 2 * x + 5) (h2 : 2 * x - (3 * x + 1) / 2 < 1) : -1 ≤ x ∧ x < 3 := sorry

end problem1_inequality_problem2_inequality_l348_348369


namespace probability_of_rolling_five_l348_348105

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end probability_of_rolling_five_l348_348105


namespace sum_of_distinct_products_82G34061H5_sum_of_distinct_products_82G34061H5_l348_348571

theorem sum_of_distinct_products_82G34061H5 (G H : ℕ) (key: G ≤ 9 ∧ H ≤ 9 ∧ 
  (∃ G H, 82G34061H5_mod_Eight = 0 ∧ 82G34061H5_mod_Nine = 0)) : 
  let products := { 0, 10, 6, 64 }
##  (∃ set.products:has_sum) 

-- this phrase might be wrong,I am not skilled.validator input. The ultimate product of this step is called sum_of_products_in_G.



theorem sum_of_distinct_products_82G34061H5 :
  let valid_pairs := { (7, 0), (5, 2), (1, 6), (8, 8) },
  let products := valid_pairs.map (λ p : ℕ×ℕ, p.1 * p.2),
  let distinct_products := products.to_finset,
  finset.sum distinct_products = 80 :=
begin
  sorry
end

end sum_of_distinct_products_82G34061H5_sum_of_distinct_products_82G34061H5_l348_348571


namespace longest_side_of_polygonal_region_is_2sqrt5_l348_348137

noncomputable def longest_side_length (x y : ℝ) : ℝ := 
if h : (x + 2 * y ≤ 4) ∧ (3 * x + 2 * y ≥ 6) ∧ (x ≥ 0) ∧ (y ≥ 0) 
then 2 * Real.sqrt 5
else 0

theorem longest_side_of_polygonal_region_is_2sqrt5 :
  ∀ x y : ℝ, (x + 2 * y ≤ 4) ∧ (3 * x + 2 * y ≥ 6) ∧ (x ≥ 0) ∧ (y ≥ 0) → longest_side_length x y = 2 * Real.sqrt 5 := 
by {
  intros,
  sorry
}

end longest_side_of_polygonal_region_is_2sqrt5_l348_348137


namespace conic_section_is_parabola_l348_348843

theorem conic_section_is_parabola (x y : ℝ) :
  abs (y - 3) = sqrt ((x + 4)^2 + y^2) →
  ∃ (A B C D : ℝ), A = 1 ∧ C = 0 ∧ (A * x^2 + B * x + C * y + D = 0) :=
by
  sorry

end conic_section_is_parabola_l348_348843


namespace junk_mail_per_house_l348_348095

theorem junk_mail_per_house (total_junk_mail : ℕ) (houses_per_block : ℕ) 
  (h1 : total_junk_mail = 14) (h2 : houses_per_block = 7) : 
  (total_junk_mail / houses_per_block) = 2 :=
by 
  sorry

end junk_mail_per_house_l348_348095


namespace find_higher_interest_rate_l348_348477

theorem find_higher_interest_rate :
  ∃ r : ℝ, 
  let total_investment := 20000,
      fraction_higher_rate := 0.55,
      higher_rate_invested := fraction_higher_rate * total_investment,
      lower_rate := 0.06,
      lower_rate_invested := total_investment - higher_rate_invested,
      total_interest := 1440,
      lower_rate_interest := lower_rate_invested * lower_rate,
      higher_rate_interest := total_interest - lower_rate_interest
  in higher_rate_interest = higher_rate_invested * r ∧ r ≈ 0.081818 :=
by
  have : ∃ r : ℝ,
    let higher_rate_invested := 0.55 * 20000,
        lower_rate_invested := 20000 - higher_rate_invested,
        lower_rate_interest := 0.06 * lower_rate_invested,
        higher_rate_interest := 1440 - lower_rate_interest
    in higher_rate_interest = higher_rate_invested * r ∧ r ≈ 0.081818 :=
  sorry
  exact this

end find_higher_interest_rate_l348_348477


namespace transformed_quadratic_equation_l348_348567

theorem transformed_quadratic_equation (u v: ℝ) :
  (u + v = -5 / 2) ∧ (u * v = 3 / 2) ↔ (∃ y : ℝ, y^2 - y + 6 = 0) := sorry

end transformed_quadratic_equation_l348_348567


namespace plant_stopped_growing_30_percent_on_10th_day_l348_348881

def plant_growth (initial_length : ℝ) (daily_growth : ℝ) (growth_target_ratio : ℝ) (target_day : ℕ) : Prop :=
  ∃ day : ℕ,
    let growth_three_days := daily_growth * 3
    let length_fourth_day := initial_length + growth_three_days
    let final_length := length_fourth_day * (1 + growth_target_ratio)
    let growth_needed := final_length - length_fourth_day
    let days_needed := growth_needed / daily_growth
    day = 4 + days_needed.to_nat ∧
    day = target_day

theorem plant_stopped_growing_30_percent_on_10th_day :
  plant_growth 11 0.6875 0.30 10 :=
by
  sorry

end plant_stopped_growing_30_percent_on_10th_day_l348_348881


namespace distance_travelled_downstream_in_12_minutes_l348_348442

noncomputable def speed_boat_still : ℝ := 15 -- in km/hr
noncomputable def rate_current : ℝ := 3 -- in km/hr
noncomputable def time_downstream : ℝ := 12 / 60 -- in hr (since 12 minutes is 12/60 hours)
noncomputable def effective_speed_downstream : ℝ := speed_boat_still + rate_current -- in km/hr
noncomputable def distance_downstream := effective_speed_downstream * time_downstream -- in km

theorem distance_travelled_downstream_in_12_minutes :
  distance_downstream = 3.6 := 
by
  sorry

end distance_travelled_downstream_in_12_minutes_l348_348442


namespace bounds_of_F_and_G_l348_348597

noncomputable def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem bounds_of_F_and_G {a b c : ℝ}
  (hF0 : |F a b c 0| ≤ 1)
  (hF1 : |F a b c 1| ≤ 1)
  (hFm1 : |F a b c (-1)| ≤ 1) :
  (∀ x, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by
  sorry

end bounds_of_F_and_G_l348_348597


namespace arithmetic_sequence_common_difference_l348_348211

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = 2)
  (h3 : ∃ r, a 2 = r * a 1 ∧ a 5 = r * a 2) :
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l348_348211


namespace jennifer_cards_left_l348_348318

-- Define the initial number of cards and the number of cards eaten
def initial_cards : ℕ := 72
def eaten_cards : ℕ := 61

-- Define the final number of cards
def final_cards (initial_cards eaten_cards : ℕ) : ℕ :=
  initial_cards - eaten_cards

-- Proposition stating that Jennifer has 11 cards left
theorem jennifer_cards_left : final_cards initial_cards eaten_cards = 11 :=
by
  -- Proof here
  sorry

end jennifer_cards_left_l348_348318


namespace joey_initial_ice_creams_l348_348077

theorem joey_initial_ice_creams :
  ∃ n : ℕ, (let cond1 := λ (n : ℕ), (n - 1 / 2) * 2 = 1 ∧ (n - 3 / 2) * 2 = 3 ∧ (n - 7 / 2) * 2 = 7 ∧ (n - 15 / 2) * 2 = 15
  in cond1 (n)) :=
sorry

end joey_initial_ice_creams_l348_348077


namespace series_diverges_1_series_converges_2_series_diverges_3_series_converges_4_l348_348677

-- 1. Prove that the series ∑ (1 / (√n)) from 1 to ∞ diverges
theorem series_diverges_1 :
  (Real.Summable (fun n : Nat => (1 / Real.sqrt n)) = False) :=
by
  sorry

-- 2. Prove that the series ∑ (1 / (n^n)) from 1 to ∞ converges
theorem series_converges_2 :
  (Real.Summable (fun n : Nat => (1 / (n^n)))) :=
by
  sorry

-- 3. Prove that the series ∑ (1 / (ln n)) from 2 to ∞ diverges
theorem series_diverges_3 :
  (Real.Summable (fun n : Nat => (1 / Real.log n)) ∧ n ≥ 2 = False) :=
by
  sorry

-- 4. Prove that the series ∑ (1 / ((n+1) * 3^n)) from 1 to ∞ converges
theorem series_converges_4 :
  (Real.Summable (fun n : Nat => (1 / ((n + 1) * (3^n))))) :=
by
  sorry

end series_diverges_1_series_converges_2_series_diverges_3_series_converges_4_l348_348677


namespace minimize_sum_of_first_n_terms_l348_348208

open Nat

noncomputable def arithmetic_sequence : ℕ → ℤ
| 1 => -11
| n => -11 + 2 * (n - 1)

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ :=
  n * (-11 + arithmetic_sequence n) / 2

theorem minimize_sum_of_first_n_terms 
  (H1 : arithmetic_sequence 1 = -11)
  (H2 : arithmetic_sequence 4 + arithmetic_sequence 6 = -6) :
  ∃ n, n = 6 ∧ sum_of_first_n_terms n = min ((λ n => sum_of_first_n_terms n) <$> (range 100)) :=
begin
  sorry
end

end minimize_sum_of_first_n_terms_l348_348208


namespace no_grasshopper_at_fourth_vertex_l348_348807

-- Definitions based on given conditions
def is_vertex_of_square (x : ℝ) (y : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1)

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

def leapfrog_jump (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2)

-- Problem statement
theorem no_grasshopper_at_fourth_vertex (a b c : ℝ × ℝ) :
  is_vertex_of_square a.1 a.2 ∧ is_vertex_of_square b.1 b.2 ∧ is_vertex_of_square c.1 c.2 →
  ∃ d : ℝ × ℝ, is_vertex_of_square d.1 d.2 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c →
  ∀ (n : ℕ) (pos : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), (pos 0 a b = leapfrog_jump a b) ∧
    (pos n a b = leapfrog_jump (pos (n-1) a b) (pos (n-1) b c)) →
    (pos n a b).1 ≠ (d.1) ∨ (pos n a b).2 ≠ (d.2) :=
sorry

end no_grasshopper_at_fourth_vertex_l348_348807


namespace volume_relationship_l348_348470

variable (r : ℝ)

def volume_cone (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
def volume_cylinder (r : ℝ) : ℝ := 2 * Real.pi * r^3
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relationship (r : ℝ) :
  2 * volume_cone r + 2 * volume_cylinder r = 3 * volume_sphere r := by
  sorry

end volume_relationship_l348_348470


namespace horizontal_coordinate_of_impact_l348_348416

variable (R g α : ℝ)

def V : ℝ := sqrt (2 * g * R * cos α)
def x (t : ℝ) : ℝ := R * sin α + V * cos α * t
def y (t : ℝ) : ℝ := R * (1 - cos α) + V * sin α * t - (g * t ^ 2) / 2
def T : ℝ := sqrt (2 * R / g) * (sin α * sqrt (cos α) + sqrt (1 - cos α ^ 3))

theorem horizontal_coordinate_of_impact : 
  x T = R * (sin α + sin (2 * α) + sqrt (cos α * (1 - cos α^3))) := by
  sorry

end horizontal_coordinate_of_impact_l348_348416


namespace probability_red_ball_fourth_draw_l348_348804

theorem probability_red_ball_fourth_draw :
  ∀ (white red : ℕ), white = 8 → red = 2 →
  let total_probability := (2 / 10) * (8 / 9) * (9 / 10) * (1 / 10) +
                           (8 / 10) * (2 / 10) * (8 / 10) * (1 / 10) * (3 / 9) +
                           (8 / 10) * (8 / 10) * (2 / 10) * (8 / 9) * (1 / 10) in
  total_probability = 0.0434 :=
begin
  sorry
end

end probability_red_ball_fourth_draw_l348_348804


namespace number_of_pencils_l348_348398

-- Definitions based on the conditions
def ratio_pens_pencils (P L : ℕ) : Prop := P * 6 = 5 * L
def pencils_more_than_pens (P L : ℕ) : Prop := L = P + 4

-- Statement to prove the number of pencils
theorem number_of_pencils : ∃ L : ℕ, (∃ P : ℕ, ratio_pens_pencils P L ∧ pencils_more_than_pens P L) ∧ L = 24 :=
by
  sorry

end number_of_pencils_l348_348398


namespace rectangular_to_polar_l348_348142

def point_rectangular := (-2 : ℝ, 2 * Real.sqrt 3 : ℝ)
def point_polar := (4 : ℝ, 2 * Real.pi / 3 : ℝ)

theorem rectangular_to_polar :
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  point_polar = (r, θ) :=
by
  sorry

end rectangular_to_polar_l348_348142


namespace four_digit_number_perfect_square_l348_348091

theorem four_digit_number_perfect_square (abcd : ℕ) (h1 : abcd ≥ 1000 ∧ abcd < 10000) (h2 : ∃ k : ℕ, k^2 = 4000000 + abcd) :
  abcd = 4001 ∨ abcd = 8004 :=
sorry

end four_digit_number_perfect_square_l348_348091


namespace solve_inequality_l348_348570

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | (1 / 2)^(x - x^2) < Log.log 3 81}

theorem solve_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solve_inequality_l348_348570


namespace find_w_e_l348_348884

theorem find_w_e : 
  ∃ (w e : ℚ),
    let V := 20 in 
    0.20 * V = 4 ∧ 
    0.15 * V = 4 ∧ 
    e = 0.25 * V ∧ 
    w = 0.40 * V ∧ 
    V ≤ 30 :=
by
  sorry

end find_w_e_l348_348884


namespace wolf_sheep_problem_l348_348412

theorem wolf_sheep_problem (n : ℕ) : ∃ m, m = 2^(n-1) ∧ 
  (∃ (friendship : finset (ℕ × ℕ)), 
   (∀ i j, i < n → j < n → (i, j) ∈ friendship ↔ (j, i) ∈ friendship) ∧ -- Mutual friendships
   (∀ initial_friends: finset ℕ, (∀ i, i ∈ initial_friends → i < n) → -- Initial friends condition
    -- The wolf can eat all sheep starting from these initial friends
    ... -- Need to precisely define the eating and friendship toggling process here
   )
) := sorry

end wolf_sheep_problem_l348_348412


namespace neither_5_nice_nor_6_nice_count_l348_348945

def is_k_nice (N k : ℕ) : Prop :=
  N % k = 1

def count_5_nice (N : ℕ) : ℕ :=
  (N - 1) / 5 + 1

def count_6_nice (N : ℕ) : ℕ :=
  (N - 1) / 6 + 1

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_30_nice (N : ℕ) : ℕ :=
  (N - 1) / 30 + 1

theorem neither_5_nice_nor_6_nice_count : 
  ∀ N < 200, 
  (N - (count_5_nice 199 + count_6_nice 199 - count_30_nice 199)) = 133 := 
by
  sorry

end neither_5_nice_nor_6_nice_count_l348_348945


namespace symm_about_origin_max_deriv_value_l348_348436

def f (x : ℝ) : ℝ := ∑ i in {1, 2, 3}, (sin ((2 * i - 1) * x)) / (2 * i - 1)

theorem symm_about_origin : ∀ x, f (-x) = - f x :=
by
  -- The proof will go here
  sorry

theorem max_deriv_value : ∃ x, (f' x = 3) ∧ ∀ y, f' y ≤ 3 :=
by
  -- The proof will go here
  sorry

-- A note: We assume here 'sum', 'sin' and exploration of 'f' are already appropriately defined and imported
-- f' would denote the derivative of the function f

end symm_about_origin_max_deriv_value_l348_348436


namespace bill_shelves_needed_l348_348909

noncomputable def shelves_needed (pots : ℕ) (per_shelf : ℕ) : ℕ :=
  (pots + per_shelf - 1) / per_shelf  -- division with ceiling

theorem bill_shelves_needed :
  let large_pots := 25
      medium_pots := 20
      small_pots := 15
      large_pots_per_shelf := 3 * 2
      medium_pots_per_shelf := 5 * 3
      small_pots_per_shelf := 7 * 4

  shelves_needed large_pots large_pots_per_shelf +
  shelves_needed medium_pots medium_pots_per_shelf +
  shelves_needed small_pots small_pots_per_shelf = 8 :=
by
  let large_pots := 25
  let medium_pots := 20
  let small_pots := 15
  let large_pots_per_shelf := 3 * 2
  let medium_pots_per_shelf := 5 * 3
  let small_pots_per_shelf := 7 * 4

  calc
  shelves_needed large_pots large_pots_per_shelf +
  shelves_needed medium_pots medium_pots_per_shelf +
  shelves_needed small_pots small_pots_per_shelf
    = (large_pots + large_pots_per_shelf - 1) / large_pots_per_shelf +
      (medium_pots + medium_pots_per_shelf - 1) / medium_pots_per_shelf +
      (small_pots + small_pots_per_shelf - 1) / small_pots_per_shelf : by rfl
... = (25 + 6 - 1) / 6 + (20 + 15 - 1) / 15 + (15 + 28 - 1) / 28 : by rfl
... = 5 + 2 + 1 : by norm_num
... = 8 : by norm_num

end bill_shelves_needed_l348_348909


namespace different_books_l348_348043

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l348_348043


namespace triangle_AFG_isosceles_l348_348702

-- Let ABC be a triangle
variables {A B C D E F G : Type*}
variables [Triangle A B C] -- We assume there's a Triangle type with vertices A, B, C

-- The angle bisectors of ∠B and ∠C intersect the circumcircle at points D and E respectively
variable (hBisectorBD : isAngleBisector B A D)
variable (hBisectorCE : isAngleBisector C A E)
variable (hDOnCircumcircle : OnCircumcircle D A B C)
variable (hEOnCircumcircle : OnCircumcircle E A B C)

-- DE intersects AB at F and AC at G
variable (hDEF : LineIntersection D E A B F)
variable (hDEG : LineIntersection D E A C G)

-- Show that triangle AFG is isosceles
theorem triangle_AFG_isosceles :
  IsIsoscelesTriangle A F G :=
sorry

end triangle_AFG_isosceles_l348_348702


namespace find_base_b_l348_348642

theorem find_base_b (b : ℕ) : ( (2 * b + 5) ^ 2 = 6 * b ^ 2 + 5 * b + 5 ) → b = 9 := 
by 
  sorry  -- Proof is not required as per instruction

end find_base_b_l348_348642


namespace minimize_distance_sum_l348_348997

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

noncomputable def f (x y a b c d : ℝ) : ℝ := 
  y + distance (x, y) (a, b) + distance (x, y) (c, d)

theorem minimize_distance_sum (a b c d : ℝ) (h1 : d > b) :
  ∃ (x y : ℝ), f x y a b c d = min_value (λ (p : ℝ × ℝ), f p.1 p.2 a b c d) :=
sorry

end minimize_distance_sum_l348_348997


namespace passengers_landed_in_virginia_l348_348905

theorem passengers_landed_in_virginia
  (P_start : ℕ) (D_Texas : ℕ) (C_Texas : ℕ) (D_NC : ℕ) (C_NC : ℕ) (C : ℕ)
  (hP_start : P_start = 124)
  (hD_Texas : D_Texas = 58)
  (hC_Texas : C_Texas = 24)
  (hD_NC : D_NC = 47)
  (hC_NC : C_NC = 14)
  (hC : C = 10) :
  P_start - D_Texas + C_Texas - D_NC + C_NC + C = 67 := by
  sorry

end passengers_landed_in_virginia_l348_348905


namespace find_x_plus_y_l348_348261

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 16) :
  x + y = 4 := 
by
  sorry

end find_x_plus_y_l348_348261


namespace Tn_less_than_16_over_9_l348_348220

noncomputable def a_n (n : ℕ) : ℝ := 1 / 4 ^ (n - 1)

def c_n (n : ℕ) : ℕ → ℝ := λ n, n * a_n n

def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, c_n (k + 1))

theorem Tn_less_than_16_over_9 (n : ℕ) : T_n n < 16 / 9 := 
  sorry

end Tn_less_than_16_over_9_l348_348220


namespace ellipse_equation_line_orthocenter_l348_348590

-- Definitions corresponding to the conditions
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  vertex_B : ℝ × ℝ
  eccentricity : ℝ

theorem ellipse_equation (C : Ellipse)
  (C_center : C.center = (0, 0))
  (C_foci_on_x_axis : C.foci_on_x_axis = true)
  (C_vertex_B : C.vertex_B = (0, 1))
  (C_eccentricity : C.eccentricity = sqrt 2 / 2) :
  (∀ x y : ℝ, (x^2 / 2 + y^2 = 1)) :=
sorry

noncomputable def line_eq (F : ℝ × ℝ) (B : ℝ × ℝ) (M N : ℝ × ℝ) : ℝ → ℝ :=
λ m, x - 4 / 3

theorem line_orthocenter 
  (C : Ellipse) (F : ℝ × ℝ) (B : ℝ × ℝ) (M N : ℝ × ℝ)
  (C_center : C.center = (0, 0))
  (C_foci_on_x_axis : C.foci_on_x_axis = true)
  (C_vertex_B : C.vertex_B = (0, 1))
  (C_eccentricity : C.eccentricity = sqrt 2 / 2)
  (F_position: F = (1, 0))
  (B_position: B = (0, 1))
  (F_is_orthocenter: true) :
  (∀ x y : ℝ, y = x - 4 / 3) :=
sorry

end ellipse_equation_line_orthocenter_l348_348590


namespace sin_identity_example_l348_348925

theorem sin_identity_example :
  sin (10 * (Real.pi / 180)) * cos (20 * (Real.pi / 180)) + cos (10 * (Real.pi / 180)) * sin (20 * (Real.pi / 180)) = 1 / 2 := 
by
  sorry

end sin_identity_example_l348_348925


namespace minimize_sum_l348_348948

noncomputable def objective_function (x : ℝ) : ℝ := x + x^2

theorem minimize_sum : ∃ x : ℝ, (objective_function x = x + x^2) ∧ (∀ y : ℝ, objective_function y ≥ objective_function (-1/2)) :=
by
  sorry

end minimize_sum_l348_348948


namespace sum_sequence_l348_348624

noncomputable section

open scoped BigOperators

def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

def a_n (n : ℕ) (f : ℝ → ℝ) : ℝ := 1 / (f (n + 1) + f n)

def S (n : ℕ) (a_n : ℕ → ℝ) : ℝ := ∑ k in Finset.range n, a_n (k + 1)

theorem sum_sequence :
  (∃ a : ℝ, f 4 a = 2) → S 2018 (a_n f) = Real.sqrt 2019 - 1 := by
  sorry

end sum_sequence_l348_348624


namespace number_of_nickels_is_3_l348_348850

-- Defining the problem conditions
def total_coins := 8
def total_value := 53 -- in cents
def at_least_one_penny := 1
def at_least_one_nickel := 1
def at_least_one_dime := 1

-- Stating the proof problem
theorem number_of_nickels_is_3 : ∃ (pennies nickels dimes : Nat), 
  pennies + nickels + dimes = total_coins ∧ 
  pennies ≥ at_least_one_penny ∧ 
  nickels ≥ at_least_one_nickel ∧ 
  dimes ≥ at_least_one_dime ∧ 
  pennies + 5 * nickels + 10 * dimes = total_value ∧ 
  nickels = 3 := sorry

end number_of_nickels_is_3_l348_348850


namespace seq_a_minus_2n_is_arithmetic_seq_b_general_formula_l348_348670

noncomputable def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else seq_a (n - 1) + 2^(n - 1) + 1

def seq_diff (n : ℕ) := seq_a n - 2^n

theorem seq_a_minus_2n_is_arithmetic : ∀ n : ℕ, seq_diff (n + 1) - seq_diff n = 1 :=
by
  intros n
  -- sorry can be used here to skip the proof
  sorry

noncomputable def seq_b (n : ℕ) := 2 * (seq_a n + 1 - n).log 2

theorem seq_b_general_formula : ∀ n : ℕ, seq_b n = 2 * n :=
by
  intros n
  -- sorry can be used here to skip the proof
  sorry

end seq_a_minus_2n_is_arithmetic_seq_b_general_formula_l348_348670


namespace complex_power_sum_l348_348154

theorem complex_power_sum : 
  (Complex.i ^ 15732) + (Complex.i ^ 15733) + (Complex.i ^ 15734) + (Complex.i ^ 15735) = 0 :=
by 
  sorry

end complex_power_sum_l348_348154


namespace regression_difference_is_residual_sum_of_squares_l348_348298

theorem regression_difference_is_residual_sum_of_squares (goodness_of_fit : ℝ -> ℝ)
    (residual_sum_of_squares : ℝ -> ℝ) :
  (∀ (data_points regression_line : ℝ), goodness_of_fit (data_points - regression_line) = residual_sum_of_squares (data_points - regression_line)) →
  (∀ (data_points regression_line : ℝ), goodness_of_fit (data_points - regression_line) = residual_sum_of_squares (data_points - regression_line)) :=
begin
  sorry
end

end regression_difference_is_residual_sum_of_squares_l348_348298


namespace probability_of_25_l348_348460

-- Declare the custom conditions for the two dice
def die1_faces : Finset ℕ := (Finset.range 20).erase 0 -- 1 to 19 and a blank face
def die2_faces : Finset ℕ := ((Finset.range 8).erase 0 ∪ (Finset.range 21).erase 8).erase 0 -- 1 to 7, 9 to 20, and a blank face

noncomputable def probability_sum_25 : ℚ :=
  let valid_combinations := 
    (die1_faces.product die2_faces).filter (λ (p : ℕ × ℕ), p.1 + p.2 = 25)
  in valid_combinations.card / (die1_faces.card * die2_faces.card : ℚ)

theorem probability_of_25 : probability_sum_25 = 7 / 200 := by
  sorry

end probability_of_25_l348_348460


namespace volume_of_alcohol_correct_l348_348459

noncomputable def radius := 3 / 2 -- radius of the tank
noncomputable def total_height := 9 -- total height of the tank
noncomputable def full_solution_height := total_height / 3 -- height of the liquid when the tank is one-third full
noncomputable def volume := Real.pi * radius^2 * full_solution_height -- volume of liquid in the tank
noncomputable def alcohol_ratio := 1 / 6 -- ratio of alcohol to the total solution
noncomputable def volume_of_alcohol := volume * alcohol_ratio -- volume of alcohol in the tank

theorem volume_of_alcohol_correct : volume_of_alcohol = (9 / 8) * Real.pi :=
by
  -- Proof would go here
  sorry

end volume_of_alcohol_correct_l348_348459


namespace number_of_valid_pairs_l348_348013

theorem number_of_valid_pairs:
  (∀ (m n : ℕ), 
    1 ≤ m ∧ m ≤ 2398 ∧
    7^1234 > 3^2399 ∧ 7^1234 < 3^2400 ∧
    7^n < 3^m ∧ 3^m < 3^(m+1) ∧ 3^(m+1) < 7^(n+1)
  ) ↔ (count_pairs 2468 := sorry)

def count_pairs : ℕ := sorry

end number_of_valid_pairs_l348_348013


namespace jacket_price_after_discounts_and_taxes_l348_348101

theorem jacket_price_after_discounts_and_taxes :
    let original_price := 150
    let discount_rate := 0.30
    let coupon_amount := 10
    let tax_rate := 0.10
    let discounted_price := original_price * (1 - discount_rate)
    let price_after_coupon := discounted_price - coupon_amount
    let final_price := price_after_coupon * (1 + tax_rate)
    final_price = 104.5 := by
    unfold original_price discount_rate coupon_amount tax_rate discounted_price price_after_coupon final_price
    sorry

end jacket_price_after_discounts_and_taxes_l348_348101


namespace complex_problem_l348_348259

noncomputable def z : ℂ := 4 + 3 * complex.I

-- Define the statement to prove
theorem complex_problem :
  complex.conj(z) / complex.abs(z) = (4 / 5) - (3 / 5) * complex.I :=
sorry

end complex_problem_l348_348259


namespace delivery_and_tip_cost_l348_348248

def original_order_cost : ℝ := 25
def new_tomatoes_cost : ℝ := 2.20
def original_tomatoes_cost : ℝ := 0.99
def new_lettuce_cost : ℝ := 1.75
def original_lettuce_cost : ℝ := 1.00
def new_celery_cost : ℝ := 2.00
def original_celery_cost : ℝ := 1.96
def new_total_bill : ℝ := 35

theorem delivery_and_tip_cost : 
  original_order_cost + (new_tomatoes_cost - original_tomatoes_cost) + 
  (new_lettuce_cost - original_lettuce_cost) + 
  (new_celery_cost - original_celery_cost) + delivery_and_tip_cost = new_total_bill → 
  delivery_and_tip_cost = 8 :=
by 
  sorry

end delivery_and_tip_cost_l348_348248


namespace lambda_companion_function_correct_count_l348_348953

noncomputable def lambda_companion_function (λ : ℝ) (f : ℝ → ℝ) : Prop := 
∀ x : ℝ, f(x + λ) + λ * f(x) = 0

theorem lambda_companion_function_correct_count : 
  let f_1 := (λ (x : ℝ), if x = 0 then 1 else 0),
      g_1 := ![0, 1],
      f_2 := (λ (λ : ℝ) (a : ℝ) (x : ℝ), a^x),
      f_3 := (λ (x : ℝ), if x = 0 then 1 else -1),
      f_4 := (λ (x : ℝ), x^2),
      n_correct := 3
  in 
  (λ (f : ℝ → ℝ), f =! f_1 ∨ f =! f_2 ∨ f =! f_3 ∨ f =! f_4) 
  := sorry

end lambda_companion_function_correct_count_l348_348953


namespace girls_with_no_pets_l348_348291

-- Define the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def fraction_girls : ℚ := 1 - fraction_boys
def girls_with_dogs_fraction : ℚ := 40 / 100
def girls_with_cats_fraction : ℚ := 20 / 100
def girls_with_no_pets_fraction : ℚ := 1 - (girls_with_dogs_fraction + girls_with_cats_fraction)

-- Calculate the number of girls
def total_girls : ℕ := total_students * fraction_girls.to_nat
def number_girls_with_no_pets : ℕ := total_girls * girls_with_no_pets_fraction.to_nat

-- Theorem statement
theorem girls_with_no_pets : number_girls_with_no_pets = 8 :=
by sorry

end girls_with_no_pets_l348_348291


namespace solution_of_system_l348_348752

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l348_348752


namespace woman_worked_days_l348_348479

-- Define variables and conditions
variables (W I : ℕ)

-- Conditions
def total_days : Prop := W + I = 25
def net_earnings : Prop := 20 * W - 5 * I = 450

-- Main theorem statement
theorem woman_worked_days (h1 : total_days W I) (h2 : net_earnings W I) : W = 23 :=
sorry

end woman_worked_days_l348_348479


namespace ecology_club_probability_l348_348382

theorem ecology_club_probability :
  let total_ways := nat.choose 30 4,
      all_boys := nat.choose 12 4,
      all_girls := nat.choose 18 4,
      probability_all_boys_or_all_girls := (all_boys + all_girls) / total_ways,
      probability_at_least_one_boy_and_one_girl := 1 - probability_all_boys_or_all_girls in
  probability_at_least_one_boy_and_one_girl = 530 / 609 :=
by
  let total_ways := nat.choose 30 4
  let all_boys := nat.choose 12 4
  let all_girls := nat.choose 18 4
  let probability_all_boys_or_all_girls := (all_boys + all_girls) / total_ways
  let probability_at_least_one_boy_and_one_girl := 1 - probability_all_boys_or_all_girls
  have h1 : total_ways = 27405 := by norm_num
  have h2 : all_boys = 495 := by norm_num
  have h3 : all_girls = 3060 := by norm_num
  have h4 : probability_all_boys_or_all_girls = (495 + 3060) / 27405 := by norm_num
  have h5 : (495 + 3060) / 27405 = 395 / 3045 := by norm_num
  have h6 : 1 - 395 / 3045 = 2650 / 3045 := by norm_num
  have h7 : 2650 / 3045 = 530 / 609 := by norm_num
  show 1 - probability_all_boys_or_all_girls = 530 / 609
  exact congr_arg (λ x, 1 - x) h5 ▸ h7

end ecology_club_probability_l348_348382


namespace a_in_union_l348_348236

variable (M N : Set ℕ) (a : ℕ)
hypothesis h1 : M ∪ N = {1, 2, 3}
hypothesis h2 : M ∩ N = {a}

theorem a_in_union : a ∈ M ∪ N := 
by {
  sorry
}

end a_in_union_l348_348236


namespace length_of_second_train_l348_348814

-- Define given conditions
def speed_train1 : ℝ := 42 -- speed in km/h
def speed_train2 : ℝ := 30 -- speed in km/h
def length_train1 : ℝ := 220 -- length in meters
def clear_time : ℝ := 24.998 -- time in seconds

-- Convert speed from km/h to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Relative speed in m/s
def relative_speed : ℝ := kmph_to_mps speed_train1 + kmph_to_mps speed_train2

-- Total distance covered in clear_time seconds
def total_distance : ℝ := relative_speed * clear_time

-- Calculate the length of the second train
def length_train2 : ℝ := total_distance - length_train1

-- The theorem statement: the length of the second train is 279.96 meters
theorem length_of_second_train : length_train2 = 279.96 := by
  sorry

end length_of_second_train_l348_348814


namespace each_person_gets_9_apples_l348_348315

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end each_person_gets_9_apples_l348_348315


namespace b_uniform_on_minus3_to_3_l348_348703

def is_uniform (X : ℝ) (a b : ℝ) : Prop := sorry -- Placeholder definition

variable (b1 : ℝ) (b : ℝ)
variable h_b1_uniform : is_uniform b1 0 1
variable h_b_def : b = 6 * (b1 - 0.5)

theorem b_uniform_on_minus3_to_3 : is_uniform b (-3) 3 :=
by
  sorry

end b_uniform_on_minus3_to_3_l348_348703


namespace polynomial_horner_value_l348_348422

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end polynomial_horner_value_l348_348422


namespace bob_remaining_money_l348_348514

def initial_amount : ℕ := 80
def monday_spent (initial : ℕ) : ℕ := initial / 2
def tuesday_spent (remaining_monday : ℕ) : ℕ := remaining_monday / 5
def wednesday_spent (remaining_tuesday : ℕ) : ℕ := remaining_tuesday * 3 / 8

theorem bob_remaining_money : 
  let remaining_monday := initial_amount - monday_spent initial_amount
  let remaining_tuesday := remaining_monday - tuesday_spent remaining_monday
  let final_remaining := remaining_tuesday - wednesday_spent remaining_tuesday
  in final_remaining = 20 := 
by
  -- Proof goes here
  sorry

end bob_remaining_money_l348_348514


namespace planar_vector_lambda_l348_348627

open Real

theorem planar_vector_lambda (λ : ℝ) : 
  let a := (0, 1 : ℝ × ℝ)
  let b := (2, 1 : ℝ × ℝ)
  (∥λ • a + b∥ = 2) → (λ = -1) :=
by
  sorry

end planar_vector_lambda_l348_348627


namespace triangle_statements_l348_348674

-- Define the fundamental properties of the triangle
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a = 45 ∧ a = 2 ∧ b = 2 * Real.sqrt 2 ∧ 
  (a - b = c * Real.cos B - c * Real.cos A)

-- Statement A
def statement_A (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  ∃ B, Real.sin B = 1

-- Statement B
def statement_B (A B C : ℝ) (v_AC v_AB : ℝ) : Prop :=
  v_AC * v_AB > 0 → Real.cos A > 0

-- Statement C
def statement_C (A B : ℝ) (a b : ℝ) : Prop :=
  Real.sin A > Real.sin B → a > b

-- Statement D
def statement_D (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  (a - b = c * Real.cos B - c * Real.cos A) →
  (a = b ∨ c^2 = a^2 + b^2)

-- Final proof statement
theorem triangle_statements (A B C a b c : ℝ) (v_AC v_AB : ℝ) 
  (h_triangle : triangle A B C a b c) :
  (statement_A A B C a b c h_triangle) ∧
  ¬(statement_B A B C v_AC v_AB) ∧
  (statement_C A B a b) ∧
  (statement_D A B C a b c h_triangle) :=
by sorry

end triangle_statements_l348_348674


namespace alex_correct_operations_l348_348483

theorem alex_correct_operations:
  ∃ x : ℝ, (x / 9 - 21 = 24) → (x * 9 + 21 = 3666) :=
by
  intros x h
  have H1 : x / 9 = 45 := by linarith
  have H2 : x = 405 := by linarith
  have H3 : 405 * 9 + 21 = 3666 := by norm_num
  show x * 9 + 21 = 3666 from H3
  sorry

end alex_correct_operations_l348_348483


namespace area_ratio_PQR_ABC_l348_348310

-- Definitions of the points and triangle
variables {A B C D E F P Q R : Type}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables [inner_product_space ℝ D] [inner_product_space ℝ E] [inner_product_space ℝ F]
variables [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]

-- Definition of segments and the ratios
variable (points_on_sides : (D ∈ line_segment B C) ∧ (E ∈ line_segment C A) ∧ (F ∈ line_segment A B))
variable (ratios : line_ratio B D C = 2 / 3 ∧ line_ratio C E A = 1 / 4 ∧ line_ratio A F B = 3 / 2)
variable (intersections : (AD ∩ BE = P) ∧ (BE ∩ CF = Q) ∧ (CF ∩ AD = R))

-- Lean 4 statement to prove the ratio of areas
theorem area_ratio_PQR_ABC :
  ∀ (A B C D E F P Q R : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] 
  [inner_product_space ℝ C] [inner_product_space ℝ D] [inner_product_space ℝ E] 
  [inner_product_space ℝ F] [inner_product_space ℝ P] [inner_product_space ℝ Q] 
  [inner_product_space ℝ R],
  (D ∈ line_segment B C ∧ E ∈ line_segment C A ∧ F ∈ line_segment A B) →
  (line_ratio B D C = 2 / 3 ∧ line_ratio C E A = 1 / 4 ∧ line_ratio A F B = 3 / 2) →
  (AD ∩ BE = P ∧ BE ∩ CF = Q ∧ CF ∩ AD = R) →
  area_ratio P Q R A B C = 9 / 380 :=
by sorry

end area_ratio_PQR_ABC_l348_348310


namespace find_p_l348_348017

def delta (a b : ℝ) : ℝ := a * b + a + b

theorem find_p (p : ℝ) (h : delta p 3 = 39) : p = 9 :=
by
  sorry

end find_p_l348_348017


namespace gill_walk_distance_l348_348579

theorem gill_walk_distance :
  ∃ d, d ∈ {x | 15 ≤ x ∧ x ≤ 20} ∧ d = 19 :=
by
  use 19
  split
  { split
    { linarith }
    { linarith }
  }
  { refl }

end gill_walk_distance_l348_348579


namespace max_value_of_q_l348_348653

-- Definitions and conditions
def q (a : ℕ) : ℚ :=
  if a >= 4 ∧ a <= 52 then
    (binomial (a - 4) 2 + binomial (52 - a) 2 : ℚ) / 1225
  else 0

-- The statement to prove
theorem max_value_of_q :
  ∃ a : ℕ, q(a) ≥ 2 / 3 :=
by
  sorry

end max_value_of_q_l348_348653


namespace general_term_formula_l348_348786

-- Define the sequence given the conditions
def sequence (n : ℕ) : ℚ := (2^n) / (2 * n + 1)

-- State the theorem
theorem general_term_formula (n : ℕ) : 
  sequence n = (2^n) / (2 * n + 1) :=
by 
  rfl -- The sequence definition is exactly the formula we want to prove

end general_term_formula_l348_348786


namespace net_displacement_after_30_moves_l348_348656

theorem net_displacement_after_30_moves : 
  let primes := { n | nat.prime n ∧ 2 ≤ n ∧ n ≤ 30 }
  let composites := { n | ¬ nat.prime n ∧ 2 ≤ n ∧ n ≤ 30 }
  let forward_steps := 2 * primes.card
  let backward_steps := 3 * composites.card
  forward_steps - backward_steps = -37 :=
by {
  sorry
}

end net_displacement_after_30_moves_l348_348656


namespace ken_and_kendra_brought_home_l348_348695

-- Define the main variables
variables (ken_caught kendra_caught ken_brought_home : ℕ)

-- Define the conditions as hypothesis
def conditions :=
  kendra_caught = 30 ∧
  ken_caught = 2 * kendra_caught ∧
  ken_brought_home = ken_caught - 3

-- Define the problem to prove
theorem ken_and_kendra_brought_home :
  (ken_caught + kendra_caught = 87) :=
begin
  -- Unpacking the conditions for readability
  unfold conditions at *,
  sorry -- Proof will go here
end

end ken_and_kendra_brought_home_l348_348695


namespace probability_four_different_numbers_l348_348801

theorem probability_four_different_numbers :
  let total_balls := 10
  let total_ways_to_choose_four := nat.choose total_balls 4
  let ways_to_choose_four_unique_numbers := nat.choose 5 4
  let ways_to_choose_colors := 2 ^ 4
  let successful_outcomes := ways_to_choose_four_unique_numbers * ways_to_choose_colors
  let probability : ℚ := successful_outcomes / total_ways_to_choose_four
  probability = 8 / 21 :=
by
  sorry

end probability_four_different_numbers_l348_348801


namespace problem_l348_348159

def f : ℕ → ℕ → ℕ := sorry

theorem problem (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) :
  2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1) ∧
  (f m 0 = 0) ∧ (f 0 n = 0) → f m n = m * n :=
by sorry

end problem_l348_348159


namespace new_person_weight_is_90_l348_348779

-- Define the data given in conditions
def original_average_weight_of_8 (avg_increase: ℝ) := avg_increase = 2.5
def weight_of_replaced_person (w_replaced: ℝ) := w_replaced = 70

-- Define the calculation that needs to be proven
def weight_of_new_person (avg_increase: ℝ) (w_replaced: ℝ) : ℝ :=
  w_replaced + (8 * avg_increase)

-- Formal statement to be proven
theorem new_person_weight_is_90 {avg_increase w_replaced: ℝ} :
  original_average_weight_of_8 avg_increase →
  weight_of_replaced_person w_replaced →
  weight_of_new_person avg_increase w_replaced = 90 :=
by
  intros h_avg h_wr
  rw [original_average_weight_of_8, weight_of_replaced_person] at h_avg h_wr
  rw [h_avg, h_wr]
  sorry

end new_person_weight_is_90_l348_348779


namespace apples_distribution_l348_348314

theorem apples_distribution (total_apples : ℕ) (given_to_father : ℕ) (total_people : ℕ) : total_apples = 55 → given_to_father = 10 → total_people = 5 → (total_apples - given_to_father) / total_people = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end apples_distribution_l348_348314


namespace yacht_arrangement_l348_348802

theorem yacht_arrangement (n k : ℕ) (h_tourists : n = 5) (h_yachts : k = 2) (h_min_tourists_per_yacht : ∀ (a b : ℕ), a + b = n → a ≥ 2 → b ≥ 2) :
  ∃(arrangements : ℕ), arrangements = 20 :=
by
  sorry

end yacht_arrangement_l348_348802


namespace time_for_first_three_workers_l348_348192

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l348_348192


namespace calculate_fraction_l348_348129

theorem calculate_fraction: (1 / (2 + 1 / (3 + 1 / 4))) = 13 / 30 := by
  sorry

end calculate_fraction_l348_348129


namespace pear_juice_percentage_is_correct_l348_348718

-- Definitions from problem conditions
def dozen_oranges : ℕ := 12
def dozen_pears : ℕ := 12
def pear_juice_per_4_pears : ℕ := 12
def orange_juice_per_3_oranges : ℕ := 6

def pears_used : ℕ := 8
def oranges_used : ℕ := 6

-- Target Express
def percent_pear_juice_in_blend : ℚ := 2 / 3

-- Proof statement
theorem pear_juice_percentage_is_correct : 
  let pear_juice_per_pear : ℚ := pear_juice_per_4_pears / 4,
      orange_juice_per_orange : ℚ := orange_juice_per_3_oranges / 3,
      total_pear_juice : ℚ := pears_used * pear_juice_per_pear,
      total_orange_juice : ℚ := oranges_used * orange_juice_per_orange,
      total_juice : ℚ := total_pear_juice + total_orange_juice,
      percent_pear_juice : ℚ := total_pear_juice / total_juice
  in
  percent_pear_juice = percent_pear_juice_in_blend
:=
by
  sorry

end pear_juice_percentage_is_correct_l348_348718


namespace problem_1_problem_2_l348_348603

noncomputable def general_term_formula : ℕ → ℕ
| n => 1 + 2 * (n - 1)

theorem problem_1 (d : ℕ) (h_pos : d > 0) (h_eq1 : 2 * (1 + 2 * (1 - 1)) + 7 * d = 16) (h_eq2 : (1 + 2 * (3 - 1)) * (1 + 2 * (6 - 1)) = 55) :
  ∀ n, general_term_formula n = 2 * n - 1 :=
sorry -- proof goes here

noncomputable def b_n (n : ℕ) : ℚ :=
1 / (general_term_formula n * general_term_formula (n + 1))

noncomputable def sum_first_n_terms (n : ℕ) : ℚ :=
(1 / 2) * ∑ i in finset.range n, (b_n i)

theorem problem_2 (n : ℕ) :
  sum_first_n_terms n = n / (2 * n + 1) :=
sorry -- proof goes here

end problem_1_problem_2_l348_348603


namespace cookie_sales_l348_348440

theorem cookie_sales (n : ℕ) (h1 : 1 ≤ n - 11) (h2 : 1 ≤ n - 2) (h3 : (n - 11) + (n - 2) < n) : n = 12 :=
sorry

end cookie_sales_l348_348440


namespace sum_of_divisors_correct_product_of_divisors_correct_l348_348889

-- Define natural number a with exactly 109 distinct divisors
def is_natural_with_109_divisors (a : ℕ) : Prop :=
  ∃ (p : ℕ) (e : ℕ), (prime p) ∧ (e + 1 = 109) ∧ (a = p^e)

-- Define the sum of divisors given the conditions
def sum_of_divisors (a : ℕ) : ℕ :=
  ∃ (p : ℕ) (e : ℕ), (prime p) ∧ (e = 108) ∧ (a = p^e) ∧ 
  (sum_divisors = (p^{109} - 1) / (p - 1))

-- Define the product of divisors given the conditions
def product_of_divisors (a : ℕ) : ℕ :=
  ∃ (p : ℕ) (e : ℕ), (prime p) ∧ (e = 108) ∧ (a = p^e) ∧ 
  (product_divisors = p^5886)

-- Prove the sum of divisors is correct under the given conditions
theorem sum_of_divisors_correct (a : ℕ) (h : is_natural_with_109_divisors a) :
  sum_of_divisors a = (p^{109} - 1) / (p - 1) :=
by
  sorry

-- Prove the product of divisors is correct under the given conditions
theorem product_of_divisors_correct (a : ℕ) (h : is_natural_with_109_divisors a) :
  product_of_divisors a = p^5886 :=
by
  sorry

end sum_of_divisors_correct_product_of_divisors_correct_l348_348889


namespace solution_l348_348764

noncomputable def proof_problem (x : ℝ) : Prop :=
  log x ^ 2 + log (x ^ 2) = 0

theorem solution : ∀ (x : ℝ), proof_problem x → (x = 1 ∨ x = 1/10^2) :=
by
  intros x h
  sorry

end solution_l348_348764


namespace locus_of_z_is_ellipse_l348_348012

theorem locus_of_z_is_ellipse (z : ℂ) (h : 2 * complex.abs (z - (2 - complex.I)) + 2 * complex.abs (z - (-3 + complex.I)) = 12) :
  ∃ a b c d : ℝ, ∀ z : ℂ, (complex.abs (z - (a + b * complex.I)) + complex.abs (z - (c + d * complex.I)) = 6) :=
sorry

end locus_of_z_is_ellipse_l348_348012


namespace expenditure_recording_l348_348349

theorem expenditure_recording (income_expenditure_rule : ∀ x : ℝ, x > 0 → record x = "+" ++ toString x ∧ x < 0 → record x = "-" ++ toString (-x)) 
    (income_condition : record 50 = "+50") : 
    record (-16) = "-16" :=
by 
  sorry

end expenditure_recording_l348_348349


namespace smallest_k_subsets_cover_sum_l348_348853

theorem smallest_k_subsets_cover_sum 
  (n r : ℕ) (hn : 0 < n) (hr : 0 < r) : 
  ∃ k : ℕ, (k = ⌈n ^ (1 / r : ℝ)⌉) ∧
    ∀ (m : ℕ) (hm : m < n), ∃ (A : Finset (Fin n) → Finset (Fin n)), 
    (∀ i : Fin r, A i.card = k) ∧ 
    (∃ (a : Fin r → ℕ), ∑ i, a i = m ∧ ∀ i, a i ∈ A i) :=
by
  sorry

end smallest_k_subsets_cover_sum_l348_348853


namespace log_sum_geom_seq_l348_348197

noncomputable def geom_seq (n : ℕ) : ℕ → ℕ := sorry
noncomputable def a_5 := geom_seq 5
noncomputable def a_6 := geom_seq 6

-- Define the given condition
axiom positive_terms : ∀ n, 0 < geom_seq n
axiom product_condition : a_5 * a_6 = 81

-- Define the theorem to prove
theorem log_sum_geom_seq : 
    log (1 / 3) (geom_seq 1) + 
    log (1 / 3) (geom_seq 2) + 
    log (1 / 3) (geom_seq 3) + 
    log (1 / 3) (geom_seq 4) + 
    log (1 / 3) (geom_seq 5) + 
    log (1 / 3) (geom_seq 6) + 
    log (1 / 3) (geom_seq 7) + 
    log (1 / 3) (geom_seq 8) + 
    log (1 / 3) (geom_seq 9) + 
    log (1 / 3) (geom_seq 10) = -20 := 
sorry

end log_sum_geom_seq_l348_348197


namespace sum_of_possible_values_of_M_l348_348791

theorem sum_of_possible_values_of_M :
  (∑ M in (finset.filter (λ M, M * (M - 4) = -7) (finset.range 100)), M)
  = 4 := 
sorry

end sum_of_possible_values_of_M_l348_348791


namespace probability_hare_given_statements_l348_348932

open scoped Classical
open Probability

-- Define events and probabilities
def event_is_hare : Prop := sorry  -- The event that the creature is a hare
def event_claims_not_hare : Prop := sorry  -- The event that the creature claims "I am not a hare"
def event_claims_not_rabbit : Prop := sorry  -- The event that the creature claims "I am not a rabbit"

#check event_is_hare
#check event_claims_not_hare
#check event_claims_not_rabbit

-- Given probabilities
def P_event_is_hare : ℚ := 1 / 2
def P_event_is_rabbit : ℚ := 1 / 2
def P_event_claims_not_hare_given_hare : ℚ := 1 / 4
def P_event_claims_not_rabbit_given_hare : ℚ := 3 / 4
def P_event_claims_not_hare_given_rabbit : ℚ := 2 / 3
def P_event_claims_not_rabbit_given_rabbit : ℚ := 1 / 3

-- Independent events
def P_event_claims_not_hare_and_not_rabbit_given_hare : ℚ := P_event_claims_not_hare_given_hare * P_event_claims_not_rabbit_given_hare
def P_event_claims_not_hare_and_not_rabbit_given_rabbit : ℚ := P_event_claims_not_hare_given_rabbit * P_event_claims_not_rabbit_given_rabbit

-- Total probability
def P_event_claims_not_hare_and_not_rabbit : ℚ := 
  (P_event_claims_not_hare_and_not_rabbit_given_hare * P_event_is_hare) + 
  (P_event_claims_not_hare_and_not_rabbit_given_rabbit * P_event_is_rabbit)

-- Conditional probability
def P_hare_given_claims_not_hare_and_not_rabbit : ℚ :=
  (P_event_claims_not_hare_and_not_rabbit_given_hare * P_event_is_hare) / P_event_claims_not_hare_and_not_rabbit

theorem probability_hare_given_statements :
  P_hare_given_claims_not_hare_and_not_rabbit = (27 / 59) :=
by sorry  -- prove it follows from the given conditions

end probability_hare_given_statements_l348_348932


namespace money_left_after_purchase_l348_348143

def initial_money : ℕ := 7
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := 3

def total_spent : ℕ := cost_candy_bar + cost_chocolate
def money_left : ℕ := initial_money - total_spent

theorem money_left_after_purchase : 
  money_left = 2 := by
  sorry

end money_left_after_purchase_l348_348143


namespace product_mod_23_l348_348427

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 :=
by 
  sorry

end product_mod_23_l348_348427


namespace ellipse_properties_line_perpendicular_PQ_OT_l348_348591

-- Define initial conditions
def ellipse_equation (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (x y : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

def left_vertex (a : ℝ) : ℝ × ℝ := (-a, 0)
def top_vertex (b : ℝ) : ℝ × ℝ := (0, b)
def right_focus : ℝ × ℝ := (1, 0)
def origin : ℝ × ℝ := (0, 0)
def midpoint_OA (a : ℝ) : ℝ × ℝ := (-a / 2, 0)

-- Prove that given b^2 = a + 1
theorem ellipse_properties (a b : ℝ) (h1 : b^2 = a + 1) (h2 : a^2 = b^2 + 1) :
  ellipse_equation 2 (sqrt 3) (by norm_num) (by norm_num) x y := 
sorry

-- Prove that line PQ is perpendicular to line OT
theorem line_perpendicular_PQ_OT (m n : ℝ) (hmn : m * n + 4 = 0) :
  let P := (2 * (12 - m^2) / (m^2 + 12), 12 * m / (m^2 + 12)),
      Q := (2 * (12 - n^2) / (n^2 + 12), 12 * n / (n^2 + 12)),
      T := (2, (m + n) / 2) in 
  (P.1 - Q.1) * 2 + (P.2 - Q.2) * ((m + n) / 2) = 0 := 
sorry

end ellipse_properties_line_perpendicular_PQ_OT_l348_348591


namespace probability_multiple_of_144_l348_348649

-- Definitions
def set : Finset ℕ := {2, 4, 6, 8, 12, 18, 36}
def is_multiple_of_144 (n : ℕ) : Prop := 144 ∣ n

-- Function to count valid pairs
def count_pairs (s : Finset ℕ) (p : ℕ → ℕ → Prop) : ℕ :=
  s.card / 2 -- Placeholder; replace with the actual counting of pairs satisfying the property p

-- Probability calculation
noncomputable def probability : ℚ :=
  (count_pairs set (λ x y, x ≠ y ∧ is_multiple_of_144 (x * y))) / (set.card * (set.card - 1) / 2)

-- Theorem statement
theorem probability_multiple_of_144 : probability = 1 / 3 :=
by sorry

end probability_multiple_of_144_l348_348649


namespace part_a_part_b_l348_348418
open Function

-- Part (a)
theorem part_a :
  ∀ (C1 C2 : set Point) (A B I : Point),
  circles_intersect_at_two_pts C1 C2 A B → midpoint_segment I A B →
  (¬ ∃ l : set Line, lines_concurrent_at_I l I ∧ intersection_points C1 C2 l = 2017) ∧
  (∃ l : set Line, lines_concurrent_at_I l I ∧ intersection_points C1 C2 l = 2018) :=
sorry

-- Part (b)
theorem part_b :
  ∀ (C1 C2 : set Point) (A B I : Point),
  circles_intersect_at_two_pts C1 C2 A B → point_within_intersection I C1 C2 A B →
  (¬ ∃ l : set Line, lines_concurrent_at_I l I ∧ intersection_points C1 C2 l = 2017) ∧
  (∃ l : set Line, lines_concurrent_at_I l I ∧ intersection_points C1 C2 l = 2018) :=
sorry

end part_a_part_b_l348_348418


namespace problem1_problem2_l348_348542

theorem problem1 :
  Real.sqrt 27 - (Real.sqrt 2 * Real.sqrt 6) + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := 
  by sorry

theorem problem2 :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 := 
  by sorry

end problem1_problem2_l348_348542


namespace find_vector_at_t_3_l348_348094

open Matrix

def vector_at_t (t : ℝ) := 
  Matrix.colVec 3 (λ i : Fin 3, vector_at_t_elem t i)

def given_conditions : Prop :=
  vector_at_t (-1) = ![1, 3, 8] ∧
  vector_at_t 2 = ![0, -2, -4]

def target_vector : Vector ℝ 3 := ![-1/3, -11/3, -8]

theorem find_vector_at_t_3 :
  given_conditions → 
  vector_at_t 3 = target_vector :=
by
  intro hc
  sorry

end find_vector_at_t_3_l348_348094


namespace first_three_workers_time_l348_348181

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l348_348181


namespace water_formed_l348_348939

theorem water_formed (n_HCl : ℕ) (n_CaCO3: ℕ) (n_H2O: ℕ) 
  (balance_eqn: ∀ (n : ℕ), 
    (2 * n_CaCO3) ≤ n_HCl ∧
    n_H2O = n_CaCO3 ):
  n_HCl = 4 ∧ n_CaCO3 = 2 → n_H2O = 2 :=
by
  intros h0
  obtain ⟨h1, h2⟩ := h0
  sorry

end water_formed_l348_348939


namespace count_irrat_numbers_l348_348487

-- Defining the list of numbers.
def number1 : ℝ := Real.sqrt 36
def number2 : ℝ := Real.cbrt 2
def number3 : ℝ := 22 / 7
def number4 : ℝ := (Real.rpower 0.1010010001)
def number5 : ℝ := Real.pi / 3
def number6 : ℝ := Real.sqrt 5

-- Defining the problem statement.
theorem count_irrat_numbers :
  (¬ (∃ p q : ℤ, q ≠ 0 ∧ number1 = p / q) ∨ 
   ¬ (∃ p q : ℤ, q ≠ 0 ∧ number2 = p / q) ∨ 
   ¬ (∃ p q : ℤ, q ≠ 0 ∧ number3 = p / q) ∨ 
   ¬ (∃ p q : ℤ, q ≠ 0 ∧ number4 = p / q) ∨ 
   ¬ (∃ p q : ℤ, q ≠ 0 ∧ number5 = p / q) ∨ 
   ¬ (∃ p q : ℤ, q ≠ 0 ∧ number6 = p / q)) = 4 :=  by sorry

end count_irrat_numbers_l348_348487


namespace first_player_winning_strategy_l348_348280

theorem first_player_winning_strategy :
  ∀ (grid : Fin 50 × Fin 70), 
    (∀ s : Finset (Fin 50 × Fin 70 × Fin 50 × Fin 70), 
      (∀ (x y : Fin 50 × Fin 70), 
        (x ≠ y) → 
        (((x, y) ∈ s) → 
          (∀ z : Fin 50 × Fin 70, (z ≠ x) → (z ≠ y) → (x, z) ∉ s ∧ (y, z) ∉ s))) → 
        ∃ f : (Fin 50 × Fin 70 × Fin 50 × Fin 70) → Fin 2, 
          (∑ i in s, (f i).1 - (f i).2 = 0 ∧ ∑ i in s, (f i).2 - (f i).1 = 0)) → 
            (∃ strategy : Nat → option (Fin 50 × Fin 70 × Fin 50 × Fin 70), 
              (∀ n, 
                strategy n ≠ none → 
                (∀ m < n, strategy n = strategy m → false))) :=
sorry

end first_player_winning_strategy_l348_348280


namespace sequence_increasing_l348_348204

theorem sequence_increasing (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ∀ n : ℕ, a^n / n^b < a^(n+1) / (n+1)^b :=
by sorry

end sequence_increasing_l348_348204


namespace natural_numbers_fitting_description_l348_348113

theorem natural_numbers_fitting_description (n : ℕ) (h : 1 / (n : ℚ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) : n = 2 ∨ n = 3 :=
by
  sorry

end natural_numbers_fitting_description_l348_348113


namespace total_cost_l348_348684

theorem total_cost (frames_cost lenses_cost coatings_cost : ℝ) 
                   (insurance_cover : ℝ)
                   (coupon_amount loyalty_discount : ℝ)
                   (tax_rate : ℝ)
                   (frames_discounted_cost : ℝ)
                   (insurance_paid : ℝ)
                   (total_lenses_coatings_cost pre_tax_total : ℝ)
                   (total_tax : ℝ)
                   (final_cost : ℝ) :
  frames_cost = 200 →
  lenses_cost = 500 →
  coatings_cost = 150 →
  insurance_cover = 0.8 →
  coupon_amount = 50 →
  loyalty_discount = 0.1 →
  tax_rate = 0.07 →
  frames_discounted_cost = frames_cost - coupon_amount - (frames_cost - coupon_amount) * loyalty_discount →
  insurance_paid = lenses_cost * insurance_cover →
  total_lenses_coatings_cost = (lenses_cost - insurance_paid) + coatings_cost →
  pre_tax_total = frames_discounted_cost + total_lenses_coatings_cost →
  total_tax = pre_tax_total * tax_rate →
  final_cost = pre_tax_total + total_tax →
  final_cost = 411.95 :=
by
  intros,
  sorry

end total_cost_l348_348684


namespace train_length_proof_l348_348898

def speed_kmph : ℝ := 54
def time_seconds : ℝ := 54.995600351971845
def bridge_length_m : ℝ := 660
def train_length_approx : ℝ := 164.93

noncomputable def speed_m_s : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_m_s * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length_m

theorem train_length_proof :
  abs (train_length - train_length_approx) < 0.01 :=
by
  sorry

end train_length_proof_l348_348898


namespace ratio_of_areas_l348_348907

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 12) (h2 : s2 = 6) :
  let A_small := area_of_equilateral_triangle s2,
      A_large := area_of_equilateral_triangle s1,
      A_trapezoid := A_large - A_small
  in (A_small / A_trapezoid) = (1 / 3) :=
by
  intros
  sorry

end ratio_of_areas_l348_348907


namespace find_a_l348_348634

theorem find_a (a : ℝ) (h₁ : ¬ (a = 0)) (h_perp : (∀ x y : ℝ, (a * x + 1 = 0) 
  -> (a - 2) * x + y + a = 0 -> ∀ x₁ y₁, (a * x₁ + 1 = 0) -> y = y₁)) : a = 2 := 
by 
  sorry

end find_a_l348_348634


namespace ratio_of_areas_l348_348495

theorem ratio_of_areas 
  (s1 s2 : ℝ)
  (A_large A_small A_trapezoid : ℝ)
  (h1 : s1 = 10)
  (h2 : s2 = 5)
  (h3 : A_large = (sqrt 3 / 4) * s1^2)
  (h4 : A_small = (sqrt 3 / 4) * s2^2)
  (h5 : A_trapezoid = A_large - A_small) :
  (A_small / A_trapezoid = 1 / 3) :=
sorry

end ratio_of_areas_l348_348495


namespace graph_equal_abs_l348_348648

theorem graph_equal_abs (f : ℝ → ℝ) : 
  (∀ x, f x = |f x|) ↔ (f = λ x, 2^x) :=
begin
  sorry
end

end graph_equal_abs_l348_348648


namespace count_odd_3digit_integers_l348_348243

variable (condition : ℕ → Bool)

theorem count_odd_3digit_integers (h_condition : ∀ n, 100 ≤ n ∧ n < 500 ∧ n % 2 = 1 ∧ condition n → n ∈ {x | x < 500}) :
  { n : ℕ | 100 ≤ n ∧ n < 500 ∧ n % 2 = 1 ∧ condition n }.card = 144 :=
sorry

end count_odd_3digit_integers_l348_348243


namespace solution_set_l348_348745

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l348_348745


namespace max_value_complex_expr_l348_348973

theorem max_value_complex_expr (x y : ℂ) : 
  (|3 * x + 4 * y| / Real.sqrt (|x| ^ 2 + |y| ^ 2 + |x ^ 2 + y ^ 2|)) ≤ (5 * Real.sqrt 2 / 2) :=
by 
  sorry

end max_value_complex_expr_l348_348973


namespace paint_cost_l348_348036

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l348_348036


namespace l_perp_β_l348_348449

variables {Point : Type} [MetricSpace Point]

-- Definitions of plane, line, perpendicular, and parallel relations
variable (α β : set Point) -- α and β are planes
variable (l : set Point)  -- l is a line

-- Conditions
def is_plane (P : set Point) : Prop := ∃a b c : Point, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ P = affine_span A ({a, b, c} : set Point)
def is_line (l : set Point) : Prop := ∃a b : Point, a ≠ b ∧ l = line_through a b

def perpendicular (P : set Point) (l : set Point) : Prop := 
∀ p ∈ P, ∀ q ∈ l, ∃ r ∈ P, ∃ s ∈ l, dist p q = dist r s

def parallel (P1 P2 : set Point) : Prop := 
∀ p ∈ P1, ∃ q ∈ P2, ∀ r ∈ P2, ∃ s ∈ P1, dist p r = dist q s
def parallel_line (l : set Point) (P : set Point) : Prop := 
∀ p ∈ P, ∃ q ∈ l, ∃ r ∈ P, ∀ s ∈ l, dist p q = dist r s

def perpendicular_line (l1 l2 : set Point) : Prop := 
∀ p ∈ l1, ∀ q ∈ l2, ∃ r ∈ l1, ∃ s ∈ l2, dist p q = dist r s

-- Given Conditions
axiom α_plane : is_plane α
axiom β_plane : is_plane β
axiom l_line : is_line l

axiom l_perp_α : perpendicular α l
axiom α_para_β : parallel α β

-- Conclusion
theorem l_perp_β : perpendicular_line l β := 
sorry

end l_perp_β_l348_348449


namespace Ken_and_Kendra_fish_count_l348_348692

def Ken_and_Kendra_bring_home (kendra_fish_caught : ℕ) (ken_ratio : ℕ) (ken_releases : ℕ) : ℕ :=
  let ken_fish_caught := ken_ratio * kendra_fish_caught
  let ken_fish_brought_home := ken_fish_caught - ken_releases
  ken_fish_brought_home + kendra_fish_caught

theorem Ken_and_Kendra_fish_count :
  let kendra_fish_caught := 30 in
  let ken_ratio := 2 in
  let ken_releases := 3 in
  Ken_and_Kendra_bring_home kendra_fish_caught ken_ratio ken_releases = 87 :=
by
  sorry

end Ken_and_Kendra_fish_count_l348_348692


namespace smallest_a_n_l348_348082

theorem smallest_a_n (n : ℕ) (h_pos : 0 < n) (x : ℝ) :
  ∃ a_n : ℝ, (∀ x : ℝ, real.sqrt (x ^ (2 * 2^n) + 1) / real.sqrt 2 ≤ a_n * (x - 1) ^ 2 + x)
    ∧ (∀ b : ℝ, (∀ x : ℝ, real.sqrt (x ^ (2 * 2^n) + 1) / real.sqrt 2 ≤ b * (x - 1) ^ 2 + x) → a_n ≤ b) :=
begin
  use 2^(n-1),
  sorry -- proof to be provided
end

end smallest_a_n_l348_348082


namespace twelve_percent_greater_l348_348276

theorem twelve_percent_greater :
  ∃ x : ℝ, x = 80 + (12 / 100) * 80 := sorry

end twelve_percent_greater_l348_348276


namespace ellipse_rolls_condition_l348_348491

variables {a b c : ℝ} (h_ellipse : ∀ x : ℝ, x ∈ (0..2 * Real.pi * a) → c = real.sqrt (b^2 - a^2))
  (h_roll_without_slip : b ≥ a)
  (h_curv : ∀ x : ℝ, c < a ^ 1.5 / b^2)

theorem ellipse_rolls_condition : b ≥ a ∧ c = real.sqrt (b^2 - a^2) ∧ c < a ^ 1.5 / b^2 :=
by
  apply and.intro h_roll_without_slip
  apply and.intro (h_ellipse _ _)
  apply h_curv
  done
end

end ellipse_rolls_condition_l348_348491


namespace fastest_hike_is_1_hour_faster_l348_348682

-- Given conditions
def distance_trail_A : ℕ := 20
def speed_trail_A : ℕ := 5
def distance_trail_B : ℕ := 12
def speed_trail_B : ℕ := 3
def mandatory_break_time : ℕ := 1

-- Lean statement to prove the problem
theorem fastest_hike_is_1_hour_faster :
  let time_trail_A := distance_trail_A / speed_trail_A in
  let hiking_time_trail_B := distance_trail_B / speed_trail_B in
  let total_time_trail_B := hiking_time_trail_B + mandatory_break_time in
  total_time_trail_B - time_trail_A = 1 :=
by
  let time_trail_A := distance_trail_A / speed_trail_A
  let hiking_time_trail_B := distance_trail_B / speed_trail_B
  let total_time_trail_B := hiking_time_trail_B + mandatory_break_time
  have h : total_time_trail_B - time_trail_A = 1, sorry
  exact h

end fastest_hike_is_1_hour_faster_l348_348682


namespace tan_expression_value_l348_348539

open Real

theorem tan_expression_value :
  let tan := Real.tan in
  tan (42 * Real.pi / 180) + tan (78 * Real.pi / 180) - sqrt 3 * tan (42 * Real.pi / 180) * tan (78 * Real.pi / 180) = -sqrt 3 :=
by
  have h1 : 42 + 78 = 120 := by norm_num
  have h2 : tan (120 * Real.pi / 180) = -sqrt 3 := by sorry 
  have h3 : tan (42 * Real.pi / 180) + tan (78 * Real.pi / 180) = tan (120 * Real.pi / 180) * (1 - tan (42 * Real.pi / 180) * tan (78 * Real.pi / 180)) := by sorry 
  exact sorry

end tan_expression_value_l348_348539


namespace find_fraction_l348_348729

-- Define the variables
variables {N F : ℚ}

-- Define the conditions
def condition1 : Prop := (1/4) * F * (2/5) * N = 35
def condition2 : Prop := (40/100) * N = 420

theorem find_fraction (h₁ : condition1) (h₂ : condition2) : F = 2/3 :=
sorry

end find_fraction_l348_348729


namespace meeting_number_l348_348867

-- Definitions based on the conditions
section
variable (n : ℕ)

def A1 := n < 12
def A2 := ¬ (7 ∣ n)
def A3 := 5 * n < 70

def B1 := 12 * n > 1000
def B2 := 10 ∣ n
def B3 := n > 100

def C1 := 4 ∣ n
def C2 := 11 * n < 1000
def C3 := 9 ∣ n

def D1 := n < 20
def D2 := Nat.Prime n
def D3 := 7 ∣ n

-- The theorem to be proved
theorem meeting_number : 
  (∃ (A1_true : A1 ∨ A2 ∨ A3) (A2_false : ¬A2), n = 89) :=
by
  sorry
end

end meeting_number_l348_348867


namespace dartboard_central_angle_l348_348454

-- Let's define the problem in Lean 4 statement
theorem dartboard_central_angle
  (A : ℝ)  -- Total area of the dartboard
  (x : ℝ)  -- The central angle we want to find
  (h1 : A > 0)  -- Total area is positive
  (h2 : 1 / 4 = (x / 360))  -- Probability condition
  : x = 90 :=
begin
  sorry -- proof to be constructed
end

end dartboard_central_angle_l348_348454


namespace two_consecutive_lucky_tickets_l348_348345

def sum_of_digits (n : ℕ) : ℕ :=
  (to_string n).clamp.mk_iterator.foldl (λ acc c => acc + c.toNat.digit) 0

def is_lucky (n : ℕ) : Prop :=
  sum_of_digits n % 7 = 0

theorem two_consecutive_lucky_tickets : ∃ (a b : ℕ), b = a + 1 ∧ is_lucky a ∧ is_lucky b ∧ 99999 ≤ a ∧ a < 1000000 :=
by {
  use (429999, 430000),
  simp [is_lucky, sum_of_digits] at *,
  sorry
}

end two_consecutive_lucky_tickets_l348_348345


namespace sum_of_sequence_l348_348574

-- Given conditions
def floor (x : ℝ) := x.to_floor

noncomputable def f (x : ℝ) : ℤ := floor x

def a (n : ℕ) : ℤ := f (n / 2)

noncomputable def sequence_term (n : ℕ) : ℤ := 2 ^ a n

noncomputable def S (n : ℕ) : ℤ :=
  let terms := list.range (2 * n) in
  list.sum (terms.map (λ x, sequence_term (x + 1)))

-- Prove that S 2n = 3 * 2^n - 3
theorem sum_of_sequence (n : ℕ) : S n = 3 * 2^n - 3 := by {
  sorry
}

end sum_of_sequence_l348_348574


namespace hyperbola_eccentricity_l348_348233

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) (H : (2:ℝ) * (|sqrt (4 + m^2)|) / 2 = sqrt(3)) : m = 2 * sqrt(2) :=
sorry

end hyperbola_eccentricity_l348_348233


namespace population_after_four_years_l348_348908

def initial_population : ℕ := 21

def update_population (b : ℕ) : ℕ := 4 * b - 9

theorem population_after_four_years :
  let b0 := initial_population in
  let b1 := update_population b0 in
  let b2 := update_population b1 in
  let b3 := update_population b2 in
  let b4 := update_population b3 in
  b4 = 4611 :=
by
  let b0 := initial_population
  let b1 := update_population b0
  let b2 := update_population b1
  let b3 := update_population b2
  let b4 := update_population b3
  show b4 = 4611 from sorry

end population_after_four_years_l348_348908


namespace smallest_k_l348_348165

theorem smallest_k (
  x : Fin 51 → ℝ
) (h_mean_zero : ∑ i, x i = 0) :
  let A := (1 / 51) * ∑ i, |x i|
  in ∑ i, (x i)^2 ≥ 51 * A^2 :=
by
  sorry

end smallest_k_l348_348165


namespace calc_expr_l348_348541

theorem calc_expr : ((Real.pi - Real.sqrt 3) ^ 0) - (2 * Real.sin (Real.pi / 4)) + Real.abs (- Real.sqrt 2) + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := 
by
  sorry

end calc_expr_l348_348541


namespace least_number_of_tiles_l348_348471

/-- A room of 544 cm long and 374 cm broad is to be paved with square tiles. 
    Prove that the least number of square tiles required to cover the floor is 176. -/
theorem least_number_of_tiles (length breadth : ℕ) (h1 : length = 544) (h2 : breadth = 374) :
  let gcd_length_breadth := Nat.gcd length breadth
  let num_tiles_length := length / gcd_length_breadth
  let num_tiles_breadth := breadth / gcd_length_breadth
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end least_number_of_tiles_l348_348471


namespace complex_modulus_pow_eight_l348_348533

theorem complex_modulus_pow_eight :
  ∀ (z : ℂ) (n : ℕ), z = 1 - 2 * Complex.I → n = 8 → |z^n| = 625 :=
by
  intros z n h_z h_n
  sorry

end complex_modulus_pow_eight_l348_348533


namespace rugged_terrain_distance_ratio_l348_348348

theorem rugged_terrain_distance_ratio (D k : ℝ) 
  (hD : D > 0) 
  (hk : k > 0) 
  (v_M v_P : ℝ) 
  (hm : v_M = 2 * k) 
  (hp : v_P = 3 * k)
  (v_Mr v_Pr : ℝ) 
  (hmr : v_Mr = k) 
  (hpr : v_Pr = 3 * k / 2) :
  ∀ (x y a b : ℝ), (x + y = D / 2) → (a + b = D / 2) → (y + b = 2 * D / 3) →
  (x / (2 * k) + y / k = a / (3 * k) + 2 * b / (3 * k)) → 
  (y / b = 1 / 3) := 
sorry

end rugged_terrain_distance_ratio_l348_348348


namespace tan_sum_identity_l348_348606

theorem tan_sum_identity :
  ∀ (α : ℝ), 0 < α ∧ α < π / 2 ∧ cos α = sqrt 5 / 5 -> tan (π / 4 + α) = -3 :=
by
  intros α h
  sorry

end tan_sum_identity_l348_348606


namespace no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l348_348794

theorem no_real_solution_x_squared_minus_2x_plus_3_eq_zero :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≠ 0 :=
by
  sorry

end no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l348_348794


namespace tetrahedron_intersection_ratio_l348_348713

theorem tetrahedron_intersection_ratio
  (tetrahedron : Type)
  (O G A B C D A1 B1 C1 D1 : tetrahedron)
  (inside_tetrahedron : inside_tetrahedron O A B C D)
  (centroid : is_centroid G A B C D)
  (intersect_A : intersection (line O G) (face A B C) A1)
  (intersect_B : intersection (line O G) (face B C D) B1)
  (intersect_C : intersection (line O G) (face C D A) C1)
  (intersect_D : intersection (line O G) (face D A B) D1)
  :
  (segment_ratio O A1 G) + (segment_ratio O B1 G) + (segment_ratio O C1 G) + (segment_ratio O D1 G) = 4 :=
  sorry

end tetrahedron_intersection_ratio_l348_348713


namespace area_GKL_l348_348306

noncomputable def point := (ℝ × ℝ)

def J : point := (0, 8)
def K : point := (0, 0)
def L : point := (10, 0)
def G : point := (0, (8 + 0) / 2)
def H : point := ((0 + 10) / 2, 0)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def base : ℝ := distance K L

def height : ℝ := G.2

def area_of_triangle (base : ℝ) (height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem area_GKL : area_of_triangle base height = 20 :=
by 
  unfold base height area_of_triangle;
  sorry

end area_GKL_l348_348306


namespace perpendicular_vectors_l348_348051

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l348_348051


namespace correct_function_is_f4_l348_348840

-- Definitions of functions
def f1 (x : ℝ) : ℝ := Real.log x
def f2 (x : ℝ) : ℝ := x^3
def f3 (x : ℝ) : ℝ := abs (Real.tan x)
def f4 (x : ℝ) : ℝ := (2 : ℝ) ^ abs x

-- Conditions for even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Conditions for monotonic increase on interval (0, +∞)
def is_monotonically_increasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, (0 < x ∧ x < y) → f x < f y

-- Main statement to prove
theorem correct_function_is_f4 :
  (is_even f1 ∧ is_monotonically_increasing_on_positive f1) ∨
  (is_even f2 ∧ is_monotonically_increasing_on_positive f2) ∨
  (is_even f3 ∧ is_monotonically_increasing_on_positive f3) ∨
  (is_even f4 ∧ is_monotonically_increasing_on_positive f4) :=
by {
  -- the proof would involve proving the conditions for each function.
  sorry
}

end correct_function_is_f4_l348_348840


namespace ratio_of_volumes_is_1_over_36_l348_348949

noncomputable def cone_cylinder_volume_ratio (r h : ℝ) : ℝ :=
  (1 / 3) * π * (r / 2)^2 * (h / 3) / (π * r^2 * h)

theorem ratio_of_volumes_is_1_over_36 (r h : ℝ) :
  cone_cylinder_volume_ratio r h = 1 / 36 :=
by
  unfold cone_cylinder_volume_ratio
  sorry

end ratio_of_volumes_is_1_over_36_l348_348949


namespace mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l348_348720

theorem mrs_mcpherson_contributes (rent : ℕ) (percentage : ℕ) (mrs_mcp_contribution : ℕ) : 
  mrs_mcp_contribution = (percentage * rent) / 100 := by
  sorry

theorem mr_mcpherson_raises (rent : ℕ) (mrs_mcp_contribution : ℕ) : 
  mr_mcp_contribution = rent - mrs_mcp_contribution := by
  sorry

theorem mr_mcpherson_complete_rent : 
  let rent := 1200
  let percentage := 30
  let mrs_mcp_contribution := (percentage * rent) / 100
  let mr_mcp_contribution := rent - mrs_mcp_contribution
  mr_mcp_contribution = 840 := by
  have mrs_contribution : mrs_mcp_contribution = (30 * 1200) / 100 := by
    exact mrs_mcpherson_contributes 1200 30 ((30 * 1200) / 100)
  have mr_contribution : mr_mcp_contribution = 1200 - ((30 * 1200) / 100) := by
    exact mr_mcpherson_raises 1200 ((30 * 1200) / 100)
  show 1200 - 360 = 840 from by
    rw [mrs_contribution, mr_contribution]
    rfl

end mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l348_348720


namespace ratio_of_areas_l348_348499

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l348_348499


namespace number_of_girls_with_no_pet_l348_348286

-- Definitions based on the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def percentage_girls_own_dogs : ℚ := 40 / 100
def percentage_girls_own_cats : ℚ := 20 / 100

-- Prove that the number of girls with no pets is 8
theorem number_of_girls_with_no_pet :
  let girls := total_students * (1 - fraction_boys),
      percentage_girls_no_pets := 1 - percentage_girls_own_dogs - percentage_girls_own_cats,
      girls_with_no_pets := girls * percentage_girls_no_pets
  in girls_with_no_pets = 8 := by
{
  sorry
}

end number_of_girls_with_no_pet_l348_348286


namespace financial_calculations_correct_l348_348520

noncomputable def revenue : ℝ := 2500000
noncomputable def expenses : ℝ := 1576250
noncomputable def loan_payment_per_month : ℝ := 25000
noncomputable def number_of_shares : ℕ := 1600
noncomputable def ceo_share_percentage : ℝ := 0.35

theorem financial_calculations_correct :
  let net_profit := (revenue - expenses) - (revenue - expenses) * 0.2 in
  let total_loan_payment := loan_payment_per_month * 12 in
  let dividends_per_share := (net_profit - total_loan_payment) / number_of_shares in
  let ceo_dividends := dividends_per_share * ceo_share_percentage * number_of_shares in
  net_profit = 739000 ∧
  total_loan_payment = 300000 ∧
  dividends_per_share = 274 ∧
  ceo_dividends = 153440 :=
begin
  sorry
end

end financial_calculations_correct_l348_348520


namespace solve_fraction_equation_l348_348367

theorem solve_fraction_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : x = -2 / 3 :=
by
  sorry

end solve_fraction_equation_l348_348367


namespace tommy_paint_cost_l348_348038

theorem tommy_paint_cost :
  ∀ (width height : ℕ) (cost_per_quart coverage_per_quart : ℕ),
    width = 5 →
    height = 4 →
    cost_per_quart = 2 →
    coverage_per_quart = 4 →
    2 * width * height / coverage_per_quart * cost_per_quart = 20 :=
by
  intros width height cost_per_quart coverage_per_quart
  intros hwidth hheight hcost hcoverage
  rw [hwidth, hheight, hcost, hcoverage]
  simp
  sorry

end tommy_paint_cost_l348_348038


namespace find_phi_intervals_of_increase_max_min_values_l348_348227

/-- Lean code to prove the mathematically equivalent proof problem --/

--(1) Proving that \(\phi = \frac{\pi}{3}\)
theorem find_phi (f : ℝ → ℝ) (h : ∀ x, f x = 1/2 * sin(2*x*sin φ + cos^2 x * cos φ - 1/2 * sin(π/2+φ))) :
  f (π/6) = 1/2 → (0 < φ) ∧ (φ < π) → φ = π/3 :=
sorry

--(2) Proving the intervals of increase of \( f(x) = \frac{1}{2}\cos(2x-\frac{\pi}{3}) \)
theorem intervals_of_increase (f : ℝ → ℝ) (φ := π/3) (h : ∀ x, f x = 1/2 * cos (2*x - π/3)) :
  ∀ k : ℤ, (k*π - π / 3) ≤ x ∧ x ≤ (k*π + π / 6) :=
sorry

--(3) Proving the maximum and minimum values of \( g(x) = \frac{1}{2}\cos(4x-\frac{\pi}{3}) \)
theorem max_min_values (g : ℝ → ℝ) (h : ∀ x, g x = 1/2 * cos (4*x - π/3)) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π/4 → -1/2 ≤ g x ∧ g x ≤ 1 :=
sorry

end find_phi_intervals_of_increase_max_min_values_l348_348227


namespace base9_is_decimal_l348_348140

-- Define the base-9 number and its decimal equivalent
def base9_to_decimal (n : Nat := 85) (base : Nat := 9) : Nat := 8 * base^1 + 5 * base^0

-- Theorem stating the proof problem
theorem base9_is_decimal : base9_to_decimal 85 9 = 77 := by
  unfold base9_to_decimal
  simp
  sorry

end base9_is_decimal_l348_348140


namespace ellipse_standard_equation_l348_348593

-- Definitions from the problem
def major_axis : ℝ := 8
def eccentricity : ℝ := 3 / 4

-- Statement to prove that the given conditions lead to the expected equations
theorem ellipse_standard_equation (a : ℝ) (b : ℝ) (c : ℝ) :
  2 * a = major_axis ∧ c = eccentricity * a ∧ b^2 = a^2 - c^2 →
  (4 * a^2 = 16 ∧ b^2 = 7)
  ∨ (4 * a^2 = 7 ∧ b^2 = 16) :=
sorry

end ellipse_standard_equation_l348_348593


namespace find_a_l348_348625

def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 1 / (2^x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

theorem find_a (a : ℝ) (h : is_odd_function (f a)) : a = 1 / 2 :=
  sorry

end find_a_l348_348625


namespace number_of_insects_l348_348125

-- Conditions
def total_legs : ℕ := 30
def legs_per_insect : ℕ := 6

-- Theorem statement
theorem number_of_insects (total_legs legs_per_insect : ℕ) : 
  total_legs / legs_per_insect = 5 := 
by
  sorry

end number_of_insects_l348_348125


namespace probability_arithmetic_sequence_l348_348030

theorem probability_arithmetic_sequence : 
  let dice_outcomes := finset.product (finset.range 1 7) (finset.product (finset.range 1 7) (finset.range 1 7)),
      valid_sequences := [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)],
      number_of_valid_sequences := 4,
      arrangements_per_sequence := 6,
      total_arrangements := 6 * 6 * 6
  in
    finset.card (finset.filter (λ triplet, ∃ s ∈ valid_sequences, s = triplet) dice_outcomes) * arrangements_per_sequence / total_arrangements = 1 / 9 := 
by
  sorry

end probability_arithmetic_sequence_l348_348030


namespace min_dot_product_Q_l348_348974

def point (α : Type*) := α × α × α

def O : point ℝ := (0, 0, 0)
def A : point ℝ := (1, 2, 2)
def B : point ℝ := (2, 1, 1)
def P : point ℝ := (1, 0, 2)

def Q (t : ℝ) : point ℝ := (t, 0, 2 * t)

def vector_sub (p1 p2 : point ℝ) : point ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : point ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def minimize_dot_product : point ℝ :=
  let t := 9 / 10
  Q t

theorem min_dot_product_Q :
  minimize_dot_product = (9 / 10, 0, 9 / 5) :=
sorry

end min_dot_product_Q_l348_348974


namespace fateful_number_probability_is_half_l348_348108
noncomputable def is_fateful_number (a b c : ℕ) : Prop :=
  (a + b = c ∨ a + c = b ∨ b + c = a)

def fateful_probability : ℚ :=
  let digits := {1, 2, 3, 4}
  let three_digit_numbers := digits.to_list.permutations.filter (λ l, l.length = 3)
  let fateful_numbers := three_digit_numbers.filter (λ l, is_fateful_number l.nth_le 0 (by sorry) l.nth_le 1 (by sorry) l.nth_le 2 (by sorry))
  fateful_numbers.length / three_digit_numbers.length

theorem fateful_number_probability_is_half : fateful_probability = 1 / 2 :=
  sorry

end fateful_number_probability_is_half_l348_348108


namespace complement_of_M_is_34_l348_348798

open Set

noncomputable def U : Set ℝ := univ
def M : Set ℝ := {x | (x - 3) / (4 - x) < 0}
def complement_M (U : Set ℝ) (M : Set ℝ) : Set ℝ := U \ M

theorem complement_of_M_is_34 : complement_M U M = {x | 3 ≤ x ∧ x ≤ 4} := 
by sorry

end complement_of_M_is_34_l348_348798


namespace arctan_sum_l348_348200

theorem arctan_sum (θ₁ θ₂ : ℝ) (h₁ : θ₁ = Real.arctan (1/2))
                              (h₂ : θ₂ = Real.arctan 2) :
  θ₁ + θ₂ = Real.pi / 2 :=
by
  have : θ₁ + θ₂ + Real.pi / 2 = Real.pi := sorry
  linarith

end arctan_sum_l348_348200


namespace tax_raise_expectation_l348_348855

noncomputable section

variables 
  (x y : ℝ) -- x: fraction of liars, y: fraction of economists
  (p1 p2 p3 p4 : ℝ) -- percentages of affirmative answers
  (expected_taxes : ℝ) -- expected fraction for taxes

-- Given conditions
def given_conditions (x y p1 p2 p3 p4 : ℝ) :=
  p1 = 0.4 ∧ p2 = 0.3 ∧ p3 = 0.5 ∧ p4 = 0.0 ∧
  y = 1 - x ∧ -- fraction of economists
  3 * x + y = 1.2 -- sum of affirmative answers

-- The statement to prove
theorem tax_raise_expectation (x y p1 p2 p3 p4 : ℝ) : 
  given_conditions x y p1 p2 p3 p4 →
  expected_taxes = 0.4 - x →
  expected_taxes = 0.3 :=
begin
  intro h, intro h_exp,
  sorry -- proof to be filled in
end

end tax_raise_expectation_l348_348855


namespace sum_reciprocals_eq_one_l348_348789

theorem sum_reciprocals_eq_one (n : ℕ) (d : Fin n → ℕ) (h : ∀ (n : ℕ), ∃ (i : Fin n), n % d i = 0) :
  (Finset.univ.sum (λ i, (d i)⁻¹ : ℚ)) = 1 :=
sorry

end sum_reciprocals_eq_one_l348_348789


namespace same_domain_intervals_l348_348576

def same_domain_function (f : ℝ → ℝ) (A : Set ℝ) : Prop :=
  ∀ (x ∈ A), f x ∈ A ∧ ∀ (y ∈ A), y ∈ Set.range f

def f1 (x : ℝ) : ℝ := Real.cos (π / 2 * x)
def f2 (x : ℝ) : ℝ := x^2 - 1
def f3 (x : ℝ) : ℝ := abs (x^2 - 1)
def f4 (x : ℝ) : ℝ := Real.log2 (x - 1)

theorem same_domain_intervals :
  (∃ A: Set ℝ, A = Set.Icc 0 1 ∧ same_domain_function f1 A) ∧
  (∃ A: Set ℝ, A = Set.Icc (-1) 0 ∧ same_domain_function f2 A) ∧
  (∃ A: Set ℝ, A = Set.Icc 0 1 ∧ same_domain_function f3 A) ∧
  ¬ (∃ A: Set ℝ, same_domain_function f4 A) :=
sorry

end same_domain_intervals_l348_348576


namespace correct_operation_l348_348433

theorem correct_operation :
  (∃ (A B C D : Prop), 
    A ↔ (sqrt(2023) + sqrt(23)) * (sqrt(2023) - sqrt(23)) = 2000 ∧
    B ↔ (1 / (sqrt(5) - 2) = 2 - sqrt(5)) ∧
    C ↔ (-ab * (ab^3) = -a^4b^4) ∧
    D ↔ ((-1)^(-1) = 1)) →
  A :=
by
  sorry

end correct_operation_l348_348433


namespace probability_log_condition_l348_348632

noncomputable def setA : set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def setB : set ℕ := {1, 2, 3}

def log_condition (a b : ℕ) : Prop := log a (2 * b) = 1

def favorable_pairs : set (ℕ × ℕ) :=
  { (a, b) | a ∈ setA ∧ b ∈ setB ∧ log_condition a b }

def total_outcomes : ℕ := setA.card * setB.card
def favorable_outcome_count : ℕ := favorable_pairs.to_finset.card

theorem probability_log_condition : 
  (favorable_outcome_count : ℚ) / total_outcomes = 1 / 6 := 
by
  -- The proof would go here
  sorry

end probability_log_condition_l348_348632


namespace trapezoid_longer_parallel_side_length_l348_348468

def rectangle_length : ℝ := 3
def rectangle_width : ℝ := 1
def center_rect (length width : ℝ) : ℝ × ℝ := (length / 2, width / 2)

theorem trapezoid_longer_parallel_side_length :
  let x := 1.25 in
  let O := center_rect rectangle_length rectangle_width in
  let Px := x in
  let Ox := 0.75 in
  let area_trapezoid := (Px + Ox) / 2 * rectangle_width in
  let total_area := rectangle_length * rectangle_width in
  let subshape_area := total_area / 3 in
  area_trapezoid = subshape_area →
  Px = x := sorry

end trapezoid_longer_parallel_side_length_l348_348468


namespace football_championship_min_games_l348_348293

theorem football_championship_min_games :
  (∃ (teams : Finset ℕ) (games : Finset (ℕ × ℕ)),
    teams.card = 20 ∧
    (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → c ≠ a →
      (a, b) ∈ games ∨ (b, c) ∈ games ∨ (c, a) ∈ games) ∧
    games.card = 90) :=
sorry

end football_championship_min_games_l348_348293


namespace percentage_of_indian_men_is_correct_l348_348652

-- Definitions for the conditions given in the problem
def total_people : ℕ := 500 + 300 + 500
def indian_women : ℕ := 0.60 * 300
def indian_children : ℕ := 0.70 * 500
def non_indian_percentage : ℝ := 55.38461538461539 / 100
def non_indian_people : ℕ := non_indian_percentage * total_people

-- Definition for the correct answer percentage of men who are Indians
def men_indian_percentage : ℝ := 10

-- The statement to be proven in Lean
theorem percentage_of_indian_men_is_correct :
  let total_people := total_people,
      indian_women := indian_women,
      indian_children := indian_children,
      non_indian_people := non_indian_people,
      indian_people := total_people - non_indian_people,
      men := 500,
      eqn := indian_people = (men_indian_percentage * men / 100) + indian_women + indian_children
  in eqn ∧ men_indian_percentage = 10 := by
  sorry

end percentage_of_indian_men_is_correct_l348_348652


namespace new_lengths_proof_l348_348828

-- Define the initial conditions
def initial_lengths : List ℕ := [15, 20, 24, 26, 30]
def original_average := initial_lengths.sum / initial_lengths.length
def average_decrease : ℕ := 5
def median_unchanged : ℕ := 24
def range_unchanged : ℕ := 15
def new_average := original_average - average_decrease

-- Assume new lengths
def new_lengths : List ℕ := [9, 9, 24, 24, 24]

-- Proof statement
theorem new_lengths_proof :
  (new_lengths.sorted.nth 2 = initial_lengths.sorted.nth 2) ∧
  (new_lengths.maximum - new_lengths.minimum = range_unchanged) ∧
  (new_lengths.sum / new_lengths.length = new_average) :=
by
  sorry

end new_lengths_proof_l348_348828


namespace problem1_solution_problem2_solution_problem3_solution_l348_348216

-- Definitions for transformations
def φ (f : ℝ → ℝ) (t : ℝ) (x : ℝ) : ℝ := f(x) - f(x - t)
def ω (f : ℝ → ℝ) (t : ℝ) (x : ℝ) : ℝ := |f(x + t) - f(x)|

-- Problem 1 Proof
theorem problem1_solution : ∀ (x : ℝ), φ (λ x, 2^x) 1 x = 2 ↔ x = 2 := sorry

-- Problem 2 Proof
theorem problem2_solution : 
  ∀ (x t : ℝ), t > 0 → 
  (λ x : ℝ, x^2) x ≥ ω (λ x : ℝ, x^2) t x ↔ 
  x ∈ (-∞, (1-√2)*t] ∪ [(1+√2)*t, ∞) := sorry

-- Problem 3 Proof
theorem problem3_solution : 
  ∀ (f : ℝ → ℝ), 
  (∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y) → 
  (∀ x t : ℝ, t > 0 → 
  ω (λ y, φ f t y) t x = φ (λ y, ω f t y) t x) → 
  ∀ x y : ℝ, x < y → f x < f y := sorry

end problem1_solution_problem2_solution_problem3_solution_l348_348216


namespace time_for_first_three_workers_l348_348193

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l348_348193


namespace matrix_and_inverse_proof_l348_348964

-- Definitions based on the conditions
def a : ℝ := 2
def b : ℝ := 3
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![1, 4]]
def alpha1 : Fin 2 → ℝ := ![3, -1]
def alpha2 : Fin 2 → ℝ := ![1, 1]
def lambda1 : ℝ := 1
def lambda2 : ℝ := 5

-- The proof statement
theorem matrix_and_inverse_proof :
  A = ![![2, 3], ![1, 4]] ∧
  (A.det ≠ 0) ∧
  A⁻¹ = ![![4 / 5, -( 3 / 5)], [-(1 / 5), 2 / 5]] ∧
  (A.mul_vec alpha1 = lambda1 • alpha1) ∧
  (A.mul_vec alpha2 = lambda2 • alpha2) :=
by {
  sorry
}

end matrix_and_inverse_proof_l348_348964


namespace symmetric_codes_count_l348_348099

def isSymmetric (grid : List (List Bool)) : Prop :=
  -- condition for symmetry: rotational and reflectional symmetry
  sorry

def isValidCode (grid : List (List Bool)) : Prop :=
  -- condition for valid scanning code with at least one black and one white
  sorry

noncomputable def numberOfSymmetricCodes : Nat :=
  -- function to count the number of symmetric valid codes
  sorry

theorem symmetric_codes_count :
  numberOfSymmetricCodes = 62 := 
  sorry

end symmetric_codes_count_l348_348099


namespace probability_correct_l348_348421

noncomputable def probability_same_or_one_diff_rolls : ℚ :=
  8 / 33

theorem probability_correct :
  let P := probability_same_or_one_diff_rolls
  in  P = 8 / 33 :=
by
  sorry

end probability_correct_l348_348421


namespace equation_has_seven_real_solutions_l348_348620

def f (x : ℝ) : ℝ := abs (x^2 - 1) - 1

theorem equation_has_seven_real_solutions (b c : ℝ) : 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) ↔ 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
  x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
  x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
  x₆ ≠ x₇ ∧
  f x₁ ^ 2 - b * f x₁ + c = 0 ∧ f x₂ ^ 2 - b * f x₂ + c = 0 ∧
  f x₃ ^ 2 - b * f x₃ + c = 0 ∧ f x₄ ^ 2 - b * f x₄ + c = 0 ∧
  f x₅ ^ 2 - b * f x₅ + c = 0 ∧ f x₆ ^ 2 - b * f x₆ + c = 0 ∧
  f x₇ ^ 2 - b * f x₇ + c = 0 :=
sorry

end equation_has_seven_real_solutions_l348_348620


namespace volume_ratio_l348_348893

noncomputable def V_D (s : ℝ) := (15 + 7 * Real.sqrt 5) * s^3 / 4
noncomputable def a (s : ℝ) := s / 2 * (1 + Real.sqrt 5)
noncomputable def V_I (a : ℝ) := 5 * (3 + Real.sqrt 5) * a^3 / 12

theorem volume_ratio (s : ℝ) (h₁ : 0 < s) :
  V_I (a s) / V_D s = (5 * (3 + Real.sqrt 5) * (1 + Real.sqrt 5)^3) / (12 * 2 * (15 + 7 * Real.sqrt 5)) :=
by
  sorry

end volume_ratio_l348_348893


namespace slope_of_line_slope_of_line_eq_sqrt3_l348_348614

theorem slope_of_line (θ : Real) (hθ : θ = Real.pi / 3) : Real :=
begin
  have h : θ = Real.pi / 3 := hθ,
  sorry
end

theorem slope_of_line_eq_sqrt3 : slope_of_line (Real.pi / 3) (by rfl) = Real.sqrt 3 :=
begin
  sorry
end

end slope_of_line_slope_of_line_eq_sqrt3_l348_348614


namespace number_solution_l348_348645

variable (a : ℝ) (x : ℝ)

theorem number_solution :
  (a^(-x) + 25^(-2*x) + 5^(-4*x) = 11) ∧ (x = 0.25) → a = 625 / 7890481 :=
by 
  sorry

end number_solution_l348_348645


namespace correct_average_mark_l348_348455

theorem correct_average_mark (
  num_students : ℕ := 50)
  (incorrect_avg : ℚ := 85.4)
  (wrong_mark_A : ℚ := 73.6) (correct_mark_A : ℚ := 63.5)
  (wrong_mark_B : ℚ := 92.4) (correct_mark_B : ℚ := 96.7)
  (wrong_mark_C : ℚ := 55.3) (correct_mark_C : ℚ := 51.8) :
  (incorrect_avg*num_students + 
   (correct_mark_A - wrong_mark_A) + 
   (correct_mark_B - wrong_mark_B) + 
   (correct_mark_C - wrong_mark_C)) / 
   num_students = 85.214 :=
sorry

end correct_average_mark_l348_348455


namespace other_root_of_quadratic_eq_l348_348916

namespace QuadraticEquation

variables {a b c : ℝ}

theorem other_root_of_quadratic_eq
  (h : ∀ x, a * (b + c) * x^2 - b * (c + a) * x - c * (a + b) = 0)
  (root1 : -1) :
  ∃ root2, root2 = (c * (a + b)) / (a * (b + c)) := by
  sorry

end QuadraticEquation

end other_root_of_quadratic_eq_l348_348916


namespace girls_with_no_pets_l348_348290

-- Define the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def fraction_girls : ℚ := 1 - fraction_boys
def girls_with_dogs_fraction : ℚ := 40 / 100
def girls_with_cats_fraction : ℚ := 20 / 100
def girls_with_no_pets_fraction : ℚ := 1 - (girls_with_dogs_fraction + girls_with_cats_fraction)

-- Calculate the number of girls
def total_girls : ℕ := total_students * fraction_girls.to_nat
def number_girls_with_no_pets : ℕ := total_girls * girls_with_no_pets_fraction.to_nat

-- Theorem statement
theorem girls_with_no_pets : number_girls_with_no_pets = 8 :=
by sorry

end girls_with_no_pets_l348_348290


namespace exists_at_most_three_planes_through_P_l348_348198

variable {α : Type*} [euclidean_space α]
variable (P : α) (T : α ≃ₑ α) -- T is an isometry (congruence transformation)

theorem exists_at_most_three_planes_through_P (hP : T P = P) :
  ∃ (S1 S2 S3 : affine_subspace α), P ∈ S1 ∧ P ∈ S2 ∧ P ∈ S3 ∧
     (S1.reflection.trans S2.reflection.trans S3.reflection = T ∨ S1.reflection.trans S2.reflection = T) :=
sorry

end exists_at_most_three_planes_through_P_l348_348198


namespace altitudes_ratio_eq_one_l348_348402

variable {α : Type*} [LinearOrderedField α]

-- Definitions of points, triangle and altitudes
structure Point (α : Type*) := (x y : α)
structure Triangle := (A B C : Point α) (acute_angle : Prop)
structure Altitude (α : Type*) := (start end : Point α)

noncomputable def areAltitudes 
  (T : Triangle) (AA1 BB1 CC1 : Altitude α) : Prop := 
  -- This should define the conditions when AA1, BB1, and CC1 are altitudes 
  sorry 

noncomputable def ratio_eq 
  (p1 q1 r1 p2 q2 r2 : α) : Prop := 
  (p1 / q1) * (r1 / p2) * (q2 / r2) = 1

theorem altitudes_ratio_eq_one 
  (T : Triangle) (AA1 BB1 CC1 : Altitude α) 
  (h : areAltitudes T AA1 BB1 CC1) 
  (AB1 AC1 CA1 CB1 BC1 BA1 : α) :
  ratio_eq AB1 AC1 CA1 CB1 BC1 BA1 :=
by
  sorry

end altitudes_ratio_eq_one_l348_348402


namespace equal_points_per_person_l348_348321

theorem equal_points_per_person :
  let blue_eggs := 12
  let blue_points := 2
  let pink_eggs := 5
  let pink_points := 3
  let golden_eggs := 3
  let golden_points := 5
  let total_people := 4
  (blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points) / total_people = 13 :=
by
  -- place the steps based on the conditions and calculations
  sorry

end equal_points_per_person_l348_348321


namespace base9_to_decimal_l348_348138

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end base9_to_decimal_l348_348138


namespace solution_set_l348_348736

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l348_348736


namespace find_n_l348_348414

-- Define the two numbers a and b
def a : ℕ := 4665
def b : ℕ := 6905

-- Calculate their difference
def diff : ℕ := b - a

-- Define that n is the greatest common divisor of the difference
def n : ℕ := Nat.gcd (b - a) 0 -- 0 here because b - a is non-zero and gcd(n, 0) = n

-- Define a function to calculate the sum of the digits of a number
def digit_sum (x : ℕ) : ℕ :=
  (Nat.digits 10 x).sum

-- Define our required properties
def has_property : Prop :=
  Nat.gcd (b - a) 0 = n ∧ digit_sum n = 4

-- Finally, state the theorem we need to prove
theorem find_n (h : has_property) : n = 1120 :=
sorry

end find_n_l348_348414


namespace harkamal_grapes_purchase_l348_348241

-- Define the conditions as parameters and constants
def cost_per_kg_grapes := 70
def kg_mangoes := 9
def cost_per_kg_mangoes := 45
def total_payment := 965

-- The theorem stating Harkamal purchased 8 kg of grapes
theorem harkamal_grapes_purchase : 
  ∃ G : ℕ, (cost_per_kg_grapes * G + cost_per_kg_mangoes * kg_mangoes = total_payment) ∧ G = 8 :=
by
  use 8
  unfold cost_per_kg_grapes cost_per_kg_mangoes kg_mangoes total_payment
  show 70 * 8 + 45 * 9 = 965 ∧ 8 = 8
  sorry

end harkamal_grapes_purchase_l348_348241


namespace range_of_f_value_of_reciprocal_l348_348618

variable {x : ℝ}
variable {m A B a b : ℝ}
variable (R : ℝ := Real.sqrt 3)
variable (f : ℝ → ℝ := λ x, m * Real.sin x + Real.sqrt 2 * Real.cos x)

noncomputable def maximum_value_condition (m : ℝ) (f : ℝ → ℝ) : Prop := 
  m > 0 ∧ (∀ x, f x ≤ 2)

noncomputable def circumcircle_condition (R : ℝ) (A B : ℝ) (f : ℝ → ℝ) : Prop := 
  R = Real.sqrt 3 ∧ f(A - Real.pi / 4) + f(B - Real.pi / 4) = 4 * Real.sqrt 6 * Real.sin A * Real.sin B

theorem range_of_f (h1 : maximum_value_condition m f) : 
  Set.range (λ x, f x) ⊆ {y | (y ∈ Icc (-2) (-1)) ∨ (y ∈ Ioc 2 6)} :=
sorry

theorem value_of_reciprocal (h1 : maximum_value_condition m f) 
  (h2 : circumcircle_condition R A B f) (h3 : Real.sin A + Real.sin B = 2 * Real.sqrt 6 * Real.sin A * Real.sin B) : 
  1 / a + 1 / b = Real.sqrt 2 :=
sorry

end range_of_f_value_of_reciprocal_l348_348618


namespace fill_entire_dish_l348_348679

theorem fill_entire_dish (n : ℕ) (h : n = 26) : 
  let days_to_fill : ℕ := 22 in 
  ∀ (doubling_factor : ℕ → ℕ) (initial_fraction : ℕ),
  doubling_factor 1 = 2 ∧ doubling_factor (initial_fraction) = 2 * initial_fraction → 
  ∃ days_to_fill, n - log2 16 = days_to_fill := 
by
  sorry

end fill_entire_dish_l348_348679


namespace findPositiveRealSolutions_l348_348162

noncomputable def onlySolutions (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a^2 - b * d) / (b + 2 * c + d) +
  (b^2 - c * a) / (c + 2 * d + a) +
  (c^2 - d * b) / (d + 2 * a + b) +
  (d^2 - a * c) / (a + 2 * b + c) = 0

theorem findPositiveRealSolutions :
  ∀ a b c d : ℝ,
  onlySolutions a b c d →
  ∃ k m : ℝ, k > 0 ∧ m > 0 ∧ a = k ∧ b = m ∧ c = k ∧ d = m :=
by
  intros a b c d h
  -- proof steps (if required) go here
  sorry

end findPositiveRealSolutions_l348_348162


namespace constant_in_expression_l348_348171

theorem constant_in_expression :
  ∃ n_const : ℤ, 
    ∀ n : ℤ, 
      (1 < 4 * n + 7) ∧ (4 * n + 7 < 40) 
        → (4 * n + n_const = 4 * n + 7) ∧ (∃ finset_condition : ℕ, finset_condition = 10 ∧  finset.card (finset.range (8 + 2) \ finset.range(-2)) = finset_condition) :=
by
  sorry

end constant_in_expression_l348_348171


namespace perfect_square_divisors_product_factorial_up_to_10_l348_348244

theorem perfect_square_divisors_product_factorial_up_to_10 :
  let product_of_factorials := (List.prod (List.map Nat.factorial (List.range' 1 (10 + 1)))) in
  let num_divisors := (fun n => ∏ p in (Nat.factors_multiset n).to_finset,
                            (Nat.factors_multiset n).count p + 1) / 2 in
  num_divisors product_of_factorials = 1920 :=
sorry

end perfect_square_divisors_product_factorial_up_to_10_l348_348244


namespace smallest_multiple_of_6_and_15_l348_348569

theorem smallest_multiple_of_6_and_15 : ∃ a : ℕ, a > 0 ∧ a % 6 = 0 ∧ a % 15 = 0 ∧ ∀ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 → a ≤ b :=
  sorry

end smallest_multiple_of_6_and_15_l348_348569


namespace original_price_shoes_l348_348153

theorem original_price_shoes (P : ℝ) (P >= 0) : 
  let total_cost := 0.95 * (0.70 * P + 160) in
  total_cost = 285 → P = 200 :=
by
  sorry

end original_price_shoes_l348_348153


namespace karen_piggy_bank_total_l348_348691

theorem karen_piggy_bank_total (a r n : ℕ) (h1 : a = 2) (h2 : r = 3) (h3 : n = 7) :
  (a * ((1 - r^n) / (1 - r))) = 2186 := by
  sorry

end karen_piggy_bank_total_l348_348691


namespace largest_binomial_term_l348_348562

-- Define the problem's conditions
def A_k (k : ℕ) : ℝ := (Nat.choose 500 k : ℝ) * (0.1 : ℝ) ^ k

-- State the main theorem
theorem largest_binomial_term : ∃ k, k = 45 ∧ (∀ j, A_k j ≤ A_k 45) :=
by
  use 45
  split
  . rfl
  . intro j
    sorry

end largest_binomial_term_l348_348562


namespace range_of_a_l348_348228

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, -2 < x → -2 < y → x < y → f a x < f a y) → (a > 1/2) :=
by
  sorry

end range_of_a_l348_348228


namespace conjugate_z_l348_348707

def complex.z : ℂ := (2 + 4 * complex.i) / (1 + complex.i)

theorem conjugate_z : complex.conj (complex.z) = 3 - complex.i := 
by sorry

end conjugate_z_l348_348707


namespace new_ribbon_lengths_correct_l348_348823

noncomputable def ribbon_lengths := [15, 20, 24, 26, 30]
noncomputable def new_average_change := 5
noncomputable def new_lengths := [9, 9, 24, 24, 24]

theorem new_ribbon_lengths_correct :
  let new_length_list := [9, 9, 24, 24, 24]
  ribbon_lengths.length = 5 ∧ -- we have 5 ribbons
  new_length_list.length = 5 ∧ -- the new list also has 5 ribbons
  list.average new_length_list = list.average ribbon_lengths - new_average_change ∧ -- new average decreased by 5
  list.median new_length_list = list.median ribbon_lengths ∧ -- median unchanged
  list.range new_length_list = list.range ribbon_lengths -- range unchanged
  :=
by {
  sorry
}

end new_ribbon_lengths_correct_l348_348823


namespace paint_cost_l348_348034

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l348_348034


namespace solution_set_l348_348737

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l348_348737


namespace min_fraction_sum_l348_348232

open Real

noncomputable theory

variable {a m n : ℝ}

def valid_log_function_constraints (a m n : ℝ) := a > 0 ∧ a ≠ 1 ∧ m > 0 ∧ n > 0

theorem min_fraction_sum : valid_log_function_constraints a m n → (3 - m - 2n = 1) → 
  min ((1 / (m + 1)) + (1 / (2 * n))) = 4 / 3 :=
by
  intros h₁ h₂
  sorry

end min_fraction_sum_l348_348232


namespace smallest_period_of_f_symmetry_center_of_f_intervals_of_monotonically_increasing_max_min_values_of_f_in_interval_l348_348714

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * cos x * cos x + (sqrt 3) / 2

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f(x) = f(x + T) :=
by {
  use π, 
  intros,
  sorry
}

theorem symmetry_center_of_f : ∃ k : ℤ, (∀ x, f(x) = f(2 * (k * π / 2 + π / 6) - x)) :=
by {
  sorry
}

theorem intervals_of_monotonically_increasing :
  ∃ k : ℤ, ∀ x, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → (∀ y1 y2, y1 ≤ y2 → f(y1) ≤ f(y2)) :=
by {
  sorry
}

theorem max_min_values_of_f_in_interval :
  ∃ max min, (max = 1) ∧ (min = -sqrt 3 / 2) ∧ ∀ x, 0 ≤ x ∧ x ≤ π / 2 → (f(x) ≤ max ∧ f(x) ≥ min) :=
by {
  use 1,
  use -sqrt 3 / 2,
  sorry
}

end smallest_period_of_f_symmetry_center_of_f_intervals_of_monotonically_increasing_max_min_values_of_f_in_interval_l348_348714


namespace distance_from_P_to_AB_l348_348031

theorem distance_from_P_to_AB
  (A B C P : Point)
  (h2 : altitude_to_AB A B C = 2)
  (line_through_P_parallel_to_AB : ∃ P, is_point_on_line P (line_parallel_to A B))
  (area_ratio : area_smaller_triangle_over_total_area = 1 / 3) :
  distance_P_to_AB P = 1 / 2 := 
sorry

end distance_from_P_to_AB_l348_348031


namespace sequence_is_geometric_not_arithmetic_l348_348917

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b / a = c / b

theorem sequence_is_geometric_not_arithmetic :
  ∀ (a₁ a₂ an : ℕ), a₁ = 3 ∧ a₂ = 9 ∧ an = 729 →
    ¬ is_arithmetic_sequence a₁ a₂ an ∧ is_geometric_sequence a₁ a₂ an :=
by
  intros a₁ a₂ an h
  sorry

end sequence_is_geometric_not_arithmetic_l348_348917


namespace total_wages_of_12_men_l348_348637

variable {M W B x y : Nat}
variable {total_wages : Nat}

-- Condition 1: 12 men do the work equivalent to W women
axiom work_equivalent_1 : 12 * M = W

-- Condition 2: 12 men do the work equivalent to 20 boys
axiom work_equivalent_2 : 12 * M = 20 * B

-- Condition 3: All together earn Rs. 450
axiom total_earnings : (12 * M) + (x * (12 * M / W)) + (y * (12 * M / (20 * B))) = 450

-- The theorem to prove
theorem total_wages_of_12_men : total_wages = 12 * M → false :=
by sorry

end total_wages_of_12_men_l348_348637


namespace middle_book_price_l348_348174

theorem middle_book_price (p : ℤ) :
  let price (n : ℤ) := p + 5 * (n - 1) in
  price 49 = 2 * (price 24 + price 25 + price 26) →
  price 25 = 24 :=
by
  sorry

end middle_book_price_l348_348174


namespace angle_ABC_arccos_11_by_16_l348_348675

theorem angle_ABC_arccos_11_by_16 (A B C : ℝ) (sin_A sin_B sin_C : ℝ)
  (h1 : sin_ratios_eq : sin_A / sin_B = 2 / 3)
  (h2 : sin_ratios_eq : sin_B / sin_C = 3 / 4) :
  ∠ ABC = arccos (11 / 16) :=
sorry

end angle_ABC_arccos_11_by_16_l348_348675


namespace number_of_girls_with_no_pets_l348_348285

-- Define total number of students
def total_students : ℕ := 30

-- Define the fraction of boys in the class
def fraction_boys : ℚ := 1 / 3

-- Define the percentages of girls with pets
def percentage_girls_with_dogs : ℚ := 0.40
def percentage_girls_with_cats : ℚ := 0.20

-- Calculate the number of boys
def number_of_boys : ℕ := (fraction_boys * total_students).toNat

-- Calculate the number of girls
def number_of_girls : ℕ := total_students - number_of_boys

-- Calculate the number of girls who own dogs
def number_of_girls_with_dogs : ℕ := (percentage_girls_with_dogs * number_of_girls).toNat

-- Calculate the number of girls who own cats
def number_of_girls_with_cats : ℕ := (percentage_girls_with_cats * number_of_girls).toNat

-- Define the statement to be proved
theorem number_of_girls_with_no_pets : number_of_girls - (number_of_girls_with_dogs + number_of_girls_with_cats) = 8 := by
  sorry

end number_of_girls_with_no_pets_l348_348285


namespace gcd_and_base5_of_187_and_119_l348_348943

theorem gcd_and_base5_of_187_and_119 :
    ∃ g : ℕ, g = Nat.gcd 187 119 ∧ g = 17 ∧ Nat.toDigits 5 g = [3, 2] :=
by
  use 17
  split
  · sorry
  split
  · rfl
  · sorry

end gcd_and_base5_of_187_and_119_l348_348943


namespace solution_set_line_l348_348733

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l348_348733


namespace max_activities_four_l348_348025

-- Definitions based on given problem conditions
inductive Gender where
  | Boy
  | Girl

def initial_children : List Gender := [Gender.Girl, Gender.Girl, Gender.Boy, Gender.Girl]

def transition (g1 g2 : Gender) : Gender :=
  match g1, g2 with
  | Gender.Boy, Gender.Boy => Gender.Boy
  | Gender.Girl, Gender.Girl => Gender.Boy
  | _, _ => Gender.Girl

def next_state (state : List Gender) : List Gender :=
  match state with
  | [g1, g2, g3, g4] => 
    [transition g1 g2, transition g2 g3, transition g3 g4, transition g4 g1]
  | _ => []

def is_all_boys (state : List Gender) : Bool :=
  state = [Gender.Boy, Gender.Boy, Gender.Boy, Gender.Boy]

-- Defining the maximum number of activities according to problem conditions
def max_activities (initial_state : List Gender) : Nat :=
  let rec count_activities (state : List Gender) (n : Nat) : Nat :=
    if is_all_boys state then n
    else count_activities (next_state state) (n + 1)
  count_activities initial_state 0

theorem max_activities_four : max_activities initial_children = 4 := by
  sorry

end max_activities_four_l348_348025


namespace anagram_pair_count_l348_348901

-- Condition1: Define the set of letters
inductive Letter
| T | U | R | N | I | P

open Letter

-- Condition2: Define the sequence type (a list of 5 letters)
def Sequence := vector Letter 5

-- Condition3: Define anagram equivalence
def is_anagram (s1 s2 : Sequence) : Prop := multiset.of_list s1.val = multiset.of_list s2.val

-- Problem statement: There are zero pairs of anagrams with exactly 100 sequences between them.
theorem anagram_pair_count : 
  ∀ s1 s2 : Sequence, 
    is_anagram s1 s2 → 
    (count_sequences_between (s1, s2) = 100) → 
    false :=
by 
  intros s1 s2 h_anagram h_count
  sorry

end anagram_pair_count_l348_348901


namespace train_crossing_time_l348_348450

theorem train_crossing_time
  (l : ℝ) (t_pole : ℝ) (p : ℝ)
  (h_train_length : l = 300)
  (h_time_pole : t_pole = 18)
  (h_platform_length : p = 450) :
  ∃ t_platform : ℝ, 
  t_platform = 45 :=
by
  -- Assign the speed based on train length and time to cross the pole
  let v := l / t_pole,
  -- Calculate the total distance to cross platform
  let D := l + p,
  -- Calculate the time to cross the platform
  let t_platform := D / v,
  -- Conclude the proof by showing that t_platform equals 45
  have : t_platform = 750 / (300 / 18),
  rw [h_train_length, h_time_pole, h_platform_length] at this,
  simp only [<-div_eq_mul_inv, div_self] at this,
  exact ⟨t_platform, sorry⟩

end train_crossing_time_l348_348450


namespace no_solution_for_p_eq_7_l348_348147

theorem no_solution_for_p_eq_7 : ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ( (x-3)/(x-4) = (x-7)/(x-8) ) → false := by
  intro x h1 h2 h
  sorry

end no_solution_for_p_eq_7_l348_348147


namespace probability_of_two_fours_is_correct_l348_348927

noncomputable def prob_exactly_two_fours : ℝ :=
  let comb := (8.choose 2 : ℝ)
  let prob_comb := (1 / 6) ^ 2 * (5 / 6) ^ 6
  let total_prob := comb * prob_comb
  total_prob

theorem probability_of_two_fours_is_correct :
  (Float.round(prob_exactly_two_fours * 1000) / 1000 = 0.094) :=
by
  sorry

end probability_of_two_fours_is_correct_l348_348927


namespace remainder_hx10_div_hx_l348_348340

noncomputable def h (x : ℝ) : ℝ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_hx10_div_hx : 
  ∀ x : ℝ, polynomial.remainder (h (x^10)) (h x) = 7 :=
by
  sorry

end remainder_hx10_div_hx_l348_348340


namespace find_number_l348_348068

theorem find_number (x : ℤ) (h : 7 * x + 37 = 100) : x = 9 :=
by
  sorry

end find_number_l348_348068


namespace a_sq_gt_b_sq_necessary_but_not_sufficient_ln_a_gt_ln_b_l348_348601

theorem a_sq_gt_b_sq_necessary_but_not_sufficient_ln_a_gt_ln_b :
  (∀ (a b : ℝ), (ln a > ln b) → (a^2 > b^2)) ∧ 
  (∃ (a b : ℝ), (a^2 > b^2) ∧ ¬ (ln a > ln b)) :=
by
  sorry

end a_sq_gt_b_sq_necessary_but_not_sufficient_ln_a_gt_ln_b_l348_348601


namespace prob_two_consecutive_tails_after_eight_heads_l348_348719

/-- Molly flips a fair coin eight times and gets heads each time. 
What is the probability of flipping two tails consecutively 
on the next two flips? --/
def probability_two_consecutive_tails_after_eight_heads 
  (fair_coin : Prop) 
  (independent_flips : Prop) 
  (prob_head : ℝ) 
  (prob_tail : ℝ) : ℝ :=
  if fair_coin ∧ independent_flips ∧ prob_head = 1/2 ∧ prob_tail = 1/2 then 1/4 else 0

/-- Assertion verifying that the probability of flipping two tails consecutively 
after flipping eight heads is 1/4, given that the coin is fair and flips are independent. --/
theorem prob_two_consecutive_tails_after_eight_heads
  (h_fair_coin : fair_coin)
  (h_independent_flips : independent_flips)
  (h_prob_tail : prob_tail = 1/2) :
  probability_two_consecutive_tails_after_eight_heads fair_coin independent_flips 1/2 1/2 = 1/4 :=
by {
  -- Proof steps will go here, fulfilling the requirements of the theorem.
  sorry
}

end prob_two_consecutive_tails_after_eight_heads_l348_348719


namespace unique_integer_in_ranges_l348_348119

theorem unique_integer_in_ranges {x : ℤ} :
  1 < x ∧ x < 9 → 
  2 < x ∧ x < 15 → 
  -1 < x ∧ x < 7 → 
  0 < x ∧ x < 4 → 
  x + 1 < 5 → 
  x = 3 := by
  intros _ _ _ _ _
  sorry

end unique_integer_in_ranges_l348_348119


namespace equivalent_operation_l348_348842

theorem equivalent_operation (x : ℚ) :
  (x / (5 / 6) * (4 / 7)) = x * (24 / 35) :=
by
  sorry

end equivalent_operation_l348_348842


namespace balls_meet_at_time_and_height_and_speed_l348_348463

-- Define the given values
def c : ℝ := 14.56  -- initial velocity in m/s
def g : ℝ := 9.808  -- acceleration due to gravity in m/s^2
def m : ℝ := 0.7    -- interval delay between the throws in seconds
def t_meet : ℝ := 1.84  -- the time when the balls meet in seconds
def s_meet : ℝ := 10.17  -- the height where the balls meet in meters
def v_collide : ℝ := 3.43  -- the speed at which the balls collide in m/s

theorem balls_meet_at_time_and_height_and_speed :
  (∃ t s v, 
    t = (c / g + m / 2) ∧ 
    s = c * t - (1/2) * g * t^2 ∧
    v = c - g * t ∧
    t ≈ t_meet ∧ 
    s ≈ s_meet ∧ 
    v ≈ v_collide) :=
begin
  use [t_meet, s_meet, v_collide],
  split,
  { exact ((14.56 / 9.808) + (0.7 / 2)) },
  split,
  { exact (14.56 * t_meet - (1 / 2) * 9.808 * t_meet^2) },
  { exact (14.56 - 9.808 * t_meet) },
  { exact sorry },
  { exact sorry },
  { exact sorry }
end

end balls_meet_at_time_and_height_and_speed_l348_348463


namespace second_child_birth_year_l348_348458

theorem second_child_birth_year :
  ∃ X, X = 1992 ∧ (∃ (first_child_birth_year marriage_year current_year : ℕ),
    marriage_year = 1980 ∧ 
    first_child_birth_year = 1982 ∧ 
    current_year = 1986 ∧
    (current_year - marriage_year) +
    (current_year - first_child_birth_year) +
    (current_year - X) = current_year) :=
begin
  sorry
end

end second_child_birth_year_l348_348458


namespace complex_equal_to_z_l348_348784

section proof_problem

variables (a b : ℝ) (x : ℝ) (i : ℂ)
noncomputable def quadratic_root_condition := x^2 + (4 + i) * x + (4 + a * i) = 0
noncomputable def complex_number (a b : ℝ) := (a : ℂ) + (b : ℂ) * i

theorem complex_equal_to_z :
  (quadratic_root_condition x i a) → (x = b) → (b = -2) → (a = 2) → (complex_number a b = 2 - 2 * i) :=
by
  intros h1 h2 h3 h4
  sorry

end proof_problem

end complex_equal_to_z_l348_348784


namespace sin_ratio_in_triangle_l348_348311

theorem sin_ratio_in_triangle 
  (A B C D : Type) 
  (B45 : ∠ B = 45)
  (C60 : ∠ C = 60)
  (D_ratio : divides D B C 2 1) :
  (sin (∠ BAD) / sin (∠ CAD)) = sqrt 6 := 
sorry

end sin_ratio_in_triangle_l348_348311


namespace first_three_workers_dig_time_l348_348177

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l348_348177


namespace mean_inequality_mean_equality_l348_348711

noncomputable def mean_r (r : ℝ) (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  (Real.exp (r⁻¹ * Real.log (∑ i, x i ^ r) / n)) ^ r

theorem mean_inequality {α β : ℝ} {n : ℕ} {x : Fin n → ℝ} (hαβ : α > β) (hx : ∀ i, 0 < x i) :
  mean_r α n x ≥ mean_r β n x := sorry

theorem mean_equality {α β : ℝ} {n : ℕ} {x : Fin n → ℝ} (hαβ : α > β) (hx : ∀ i, 0 < x i) :
  mean_r α n x = mean_r β n x ↔ ∀ i j, x i = x j := sorry

end mean_inequality_mean_equality_l348_348711


namespace original_faculty_theorem_l348_348895

-- Define the conditions of the problem
variable (original_faculty : Int) -- The original number of faculty members
variable (reduced_faculty : Int) -- The number of faculty members after reduction
variable (reduction_percentage : Float) -- The percentage reduction

-- State the given conditions
def conditions : Prop := 
  reduction_percentage = 0.30 ∧
  reduced_faculty = 250 ∧
  reduced_faculty = (original_faculty:Float) * (1.0 - reduction_percentage)

-- State the theorem to be proven
theorem original_faculty_theorem : conditions original_faculty reduced_faculty reduction_percentage → original_faculty = 357 :=
by
  sorry

end original_faculty_theorem_l348_348895


namespace min_right_triangles_cover_equilateral_triangle_l348_348061

theorem min_right_triangles_cover_equilateral_triangle :
  let side_length_equilateral := 12
  let legs_right_triangle := 1
  let area_equilateral := (Real.sqrt 3 / 4) * side_length_equilateral ^ 2
  let area_right_triangle := (1 / 2) * legs_right_triangle * legs_right_triangle
  let triangles_needed := area_equilateral / area_right_triangle
  triangles_needed = 72 * Real.sqrt 3 := 
by 
  sorry

end min_right_triangles_cover_equilateral_triangle_l348_348061


namespace prob1_prob2_l348_348415

-- Definitions of vectors a, b, c, d
def vec2 := ℝ × ℝ

def a : vec2 := (3, 2)
def b : vec2 := (-1, 2)
def c : vec2 := (4, 1)

-- Definitions of m and n solving the linear system
axiom m : ℝ
axiom n : ℝ
axiom eq_m : m = (5 / 9)
axiom eq_n : n = (8 / 9)

-- Definition of d and possible solutions
def d1 : vec2 := (20 / 5 + 2 * real.sqrt 5 / 5, 5 / 5 + 4 * real.sqrt 5 / 5)
def d2 : vec2 := (20 / 5 - 2 * real.sqrt 5 / 5, 5 / 5 - 4 * real.sqrt 5 / 5)

-- The proof to check m and n satisfy the vector equation
theorem prob1 (a b c : vec2) (m n : ℝ) (h1 : -m + 4 * n = 3) (h2 : 2 * m + n = 2) :
    a = (3, 2) ∧ b = (-1, 2) ∧ c = (4, 1) ∧ m = 5 / 9 ∧ n = 8 / 9 →
    a = m * b + n * c := sorry

-- The proof to find the vector d
theorem prob2 (a b c : vec2) (d1 d2 : vec2) :
    a = (3, 2) ∧ b = (-1, 2) ∧ c = (4, 1) →
    (∃ k : ℝ, d1 = (4 + 2 * k, 1 + 4 * k) ∧ (4 + 2 * k - 4)^2 + (1 + 4 * k - 1)^2 = 1) ∧ 
    (∃ k : ℝ, d2 = (4 - 2 * k, 1 - 4 * k) ∧ (4 - 2 * k - 4)^2 + (1 - 4 * k - 1)^2 = 1) := sorry

end prob1_prob2_l348_348415


namespace value_range_sinx_sinabsx_l348_348410

theorem value_range_sinx_sinabsx : 
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi → 
    (∃ y : ℝ, y = Real.sin x + Real.sin (|x|) ∧ y ∈ Set.Icc 0 2) := 
begin
  -- We do not need to include the proof here.  
  sorry
end

end value_range_sinx_sinabsx_l348_348410


namespace length_of_pipe_is_correct_l348_348109

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end length_of_pipe_is_correct_l348_348109


namespace sum_of_ODD_and_EVEN_l348_348331

def num_divisors (m : ℕ) : ℕ :=
  (Finset.range m).filter (λ x, x > 0 ∧ m % x = 0).card

theorem sum_of_ODD_and_EVEN (n : ℕ) :
  ∑ k in Finset.range n, num_divisors (2 * k + 1) ≤ ∑ k in Finset.range n, num_divisors (2 * k) :=
sorry

end sum_of_ODD_and_EVEN_l348_348331


namespace find_multiplier_l348_348097

noncomputable def x : ℝ := 2.3333333333333335

-- Defining the main hypothesis
def main_hypothesis (n : ℝ) : Prop :=
  (x * n / 3 = x^2)

-- Defining the proof problem
theorem find_multiplier (h : x > 0) : ∃ n, main_hypothesis n ∧ n = 7 :=
by
  use 7
  have x_sq : x^2 = 5.444444444444444 := by sorry
  have main_eq : x * 7 / 3 = 5.444444444444444 := by sorry
  exact ⟨main_eq, rfl⟩

end find_multiplier_l348_348097


namespace necessary_and_sufficient_condition_l348_348212

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) ↔ (a * b > 0) :=
sorry

end necessary_and_sufficient_condition_l348_348212


namespace fraction_flowering_plants_on_porch_l348_348717

theorem fraction_flowering_plants_on_porch (total_plants : ℕ) (percent_flowering : ℕ) (flowers_per_plant : ℕ) (total_flowers_on_porch : ℕ) :
 (total_plants = 80) →
 (percent_flowering = 40) →
 (flowers_per_plant = 5) →
 (total_flowers_on_porch = 40) →
 let total_flowering := (percent_flowering * total_plants) / 100 in
 let plants_on_porch := total_flowers_on_porch / flowers_per_plant in
 let fraction_on_porch := plants_on_porch * 1 / total_flowering in
 fraction_on_porch = 1 / 4 :=
by {
  intros h1 h2 h3 h4,
  let total_flowering := (percent_flowering * total_plants) / 100,
  let plants_on_porch := total_flowers_on_porch / flowers_per_plant,
  let fraction_on_porch := plants_on_porch * 1 / total_flowering,
  have ht1 : total_flowering = 32, from by simp [h1, h2, total_flowering],
  have hp1 : plants_on_porch = 8, from by simp [h3, h4, total_flowers_on_porch, flowers_per_plant],
  have frac_calc : fraction_on_porch = 1 / 4, from by simp [ht1, hp1, fraction_on_porch],
  exact frac_calc
}

end fraction_flowering_plants_on_porch_l348_348717


namespace two_digit_number_sum_l348_348004

theorem two_digit_number_sum (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : |(10 * a + b) - (10 * b + a)| = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end two_digit_number_sum_l348_348004


namespace DE_plus_FG_equals_5_l348_348047

noncomputable def equilateral_triangle_ABC : Triangle :=
{ A := (0, 0),
  B := (2, 0),
  C := (1, sqrt 3) }

-- Given points E and G on AC, and D, F on AB such that DE and FG are parallel to BC
noncomputable def points_on_lines : Prop :=
  ∃ E G D F, 
  E ∈ Line_AC ∧ G ∈ Line_AC ∧ D ∈ Line_AB ∧ F ∈ Line_AB ∧ 
  Parallel(Line_DE, Line_BC) ∧ Parallel(Line_FG, Line_BC)

-- Perimeters condition
noncomputable def equal_perimeters : Prop :=
  ∀ E G D F x t,
  (AD = x) ∧ (DE = x) ∧ (AF = t) ∧ (FG = t / 2) →
  (perimeter (triangle ADE) = perimeter (trapezoid DFGE)) ∧
  (perimeter (triangle ADE) = perimeter (trapezoid FBCG))

-- Main theorem
theorem DE_plus_FG_equals_5 :
  points_on_lines ∧ equal_perimeters →
  ∃ E G D F x t, DE + FG = 5 :=
by
  intro h
  sorry

end DE_plus_FG_equals_5_l348_348047


namespace triangle_is_isosceles_l348_348673

theorem triangle_is_isosceles {A B C : Type} [triangle A B C] (b c : ℝ) (h : b * cos C = c * cos B) :
  is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l348_348673


namespace rudy_second_run_rate_l348_348760

theorem rudy_second_run_rate (distance1 distance2 time1 total_time : ℝ) (rate1 rate2 : ℝ) :
  distance1 = 5 ∧ rate1 = 10 ∧ distance2 = 4 ∧ total_time = 88 ∧ time1 = distance1 * rate1 →
  rate2 = (total_time - time1) / distance2 →
  rate2 = 9.5 :=
begin
  sorry
end

end rudy_second_run_rate_l348_348760


namespace part_a_part_b_l348_348323

open Complex

theorem part_a (n : ℕ) (hn : n ≥ 3) : 
  ∃ (z : Fin n → ℂ), 
    (∑ i in Finset.range n, z i / z ((i + 1) % n)) = n * Complex.I :=
sorry

theorem part_b (n : ℕ) (hn : n ≥ 3)
  (h_modulus: ∃ r : ℝ, r > 0 ∧ (∀ i, ∃ θ : ℝ, (z i = r * Complex.exp (Complex.I * θ)))) 
  (h_sum: (∑ i in Finset.range n, z i / z ((i + 1) % n)) = n * Complex.I) : 
  n % 4 = 0 :=
sorry

end part_a_part_b_l348_348323


namespace value_of_x_plus_y_l348_348253

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l348_348253


namespace total_employees_l348_348888

theorem total_employees (part_time full_time : ℕ) (h₀ : part_time = 2041) (h₁ : full_time = 63093) : 
  part_time + full_time = 65134 := 
by {
  rw [h₀, h₁],
  refl, -- This is where the calculation of 2041 + 63093 = 65134 happens
  sorry -- placeholder for skipping additional proof details if required
}

end total_employees_l348_348888


namespace odd_function_m_monotonically_increasing_m_symmetry_about_point_m_l348_348229

def f (x m : ℝ) : ℝ := 2^x + m*2^(1-x)

-- Question 1
theorem odd_function_m (m : ℝ) : (∀ x, f (-x) m = -f x m) ↔ m = -1/2 :=
begin
  sorry
end

-- Question 2
theorem monotonically_increasing_m (m : ℝ) : (∀ x1 x2, 1 < x1 → x1 < x2 → f x1 m < f x2 m) ↔ m ≤ 2 :=
begin
  sorry
end

-- Question 3 - Symmetry about the point A(a, 0)
theorem symmetry_about_point_m (m a : ℝ) : 
  (∃ a, ∀ x1, f x1 m + f (2*a - x1) m = 0) ↔ 
  (m < 0 ∧ a = (1 / 2) * log2 (-2 * m)) ∨ (m ≥ 0 ∧ false) :=
begin
  sorry
end

end odd_function_m_monotonically_increasing_m_symmetry_about_point_m_l348_348229


namespace first_three_workers_dig_time_l348_348175

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l348_348175


namespace sunzi_classic_l348_348446

noncomputable def length_of_rope : ℝ := sorry
noncomputable def length_of_wood : ℝ := sorry
axiom first_condition : length_of_rope - length_of_wood = 4.5
axiom second_condition : length_of_wood - (1 / 2) * length_of_rope = 1

theorem sunzi_classic : 
  (length_of_rope - length_of_wood = 4.5) ∧ (length_of_wood - (1 / 2) * length_of_rope = 1) := 
by 
  exact ⟨first_condition, second_condition⟩

end sunzi_classic_l348_348446


namespace ribbon_proof_l348_348819

theorem ribbon_proof :
  ∃ (ribbons : List ℝ), 
    ribbons = [9, 9, 24, 24, 24] ∧
    (∃ (initial_ribbons : List ℝ) (cuts : List ℝ),
      initial_ribbons = [15, 20, 24, 26, 30] ∧
      ((List.sum initial_ribbons / initial_ribbons.length) - (List.sum ribbons / ribbons.length) = 5) ∧
      (List.median ribbons = List.median initial_ribbons) ∧
      (List.range ribbons = List.range initial_ribbons)) :=
sorry

end ribbon_proof_l348_348819


namespace veronica_initial_marbles_l348_348926

variable {D M P V : ℕ}

theorem veronica_initial_marbles (hD : D = 14) (hM : M = 20) (hP : P = 19)
  (h_total : D + M + P + V = 60) : V = 7 :=
by
  sorry

end veronica_initial_marbles_l348_348926


namespace number_of_grade11_students_l348_348887

-- Define the total number of students in the high school.
def total_students : ℕ := 900

-- Define the total number of students selected in the sample.
def sample_students : ℕ := 45

-- Define the number of Grade 10 students in the sample.
def grade10_students_sample : ℕ := 20

-- Define the number of Grade 12 students in the sample.
def grade12_students_sample : ℕ := 10

-- Prove the number of Grade 11 students in the school is 300.
theorem number_of_grade11_students :
  (sample_students - grade10_students_sample - grade12_students_sample) * (total_students / sample_students) = 300 :=
by
  sorry

end number_of_grade11_students_l348_348887


namespace laptop_weight_l348_348727

-- Defining the weights
variables (B U L P : ℝ)
-- Karen's tote weight
def K := 8

-- Conditions from the problem
axiom tote_eq_two_briefcase : K = 2 * B
axiom umbrella_eq_half_briefcase : U = B / 2
axiom full_briefcase_eq_double_tote : B + L + P + U = 2 * K
axiom papers_eq_sixth_full_briefcase : P = (B + L + P) / 6

-- Theorem stating the weight of Kevin's laptop is 7.67 pounds
theorem laptop_weight (hB : B = 4) (hU : U = 2) (hL : L = 7.67) : 
  L - K = -0.33 :=
by
  sorry

end laptop_weight_l348_348727


namespace new_lengths_proof_l348_348830

-- Define the initial conditions
def initial_lengths : List ℕ := [15, 20, 24, 26, 30]
def original_average := initial_lengths.sum / initial_lengths.length
def average_decrease : ℕ := 5
def median_unchanged : ℕ := 24
def range_unchanged : ℕ := 15
def new_average := original_average - average_decrease

-- Assume new lengths
def new_lengths : List ℕ := [9, 9, 24, 24, 24]

-- Proof statement
theorem new_lengths_proof :
  (new_lengths.sorted.nth 2 = initial_lengths.sorted.nth 2) ∧
  (new_lengths.maximum - new_lengths.minimum = range_unchanged) ∧
  (new_lengths.sum / new_lengths.length = new_average) :=
by
  sorry

end new_lengths_proof_l348_348830


namespace divisor_of_425904_l348_348424

theorem divisor_of_425904 :
  ∃ (d : ℕ), d = 7 ∧ ∃ (n : ℕ), n = 425897 + 7 ∧ 425904 % d = 0 :=
by
  sorry

end divisor_of_425904_l348_348424


namespace percentage_tax_raise_expecting_population_l348_348858

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end percentage_tax_raise_expecting_population_l348_348858


namespace cookies_per_bag_l348_348650

theorem cookies_per_bag (b T : ℕ) (h1 : b = 37) (h2 : T = 703) : (T / b) = 19 :=
by
  -- Placeholder for proof
  sorry

end cookies_per_bag_l348_348650


namespace set_M_properties_l348_348234

theorem set_M_properties (m : ℝ) : 
  (∃ M : set ℝ, M = {x | x ^ 2 - m * x + 6 = 0} ∧ M ∩ {1, 2, 3, 6} = M) →
  (∀ M, (M ⊆ {1, 2, 3, 6}) ∧ (M = {1, 6} ∨ M = {2, 3} ∨ M = ∅) →
    (m = 5 ∨ m = 7 ∨ m ∈ set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6))) :=
begin
  by {
    assume h,
    cases h with M hM,
    sorry, -- Proof goes here
  }
end

end set_M_properties_l348_348234


namespace proof_problem_l348_348861

/-
Definitions of the distributions and masses as lists.
The distributions and lists should match one-to-one positions as presented in the problem.
-/
def X := [1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 8]
def Y := [5, 3, 2, 4, 5, 5, 6, 7, 8, 8, 8, 7, 8, 10]
def Z := [-2, -4, -1, -3, -3, -4, -4, -6, -7, -6, -9, -8, -8, -10]
def m := [1, 2, 3, 3, 2, 4, 3, 2, 3, 1, 2, 1, 2, 1]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length.toFloat

noncomputable def variance (l : List ℝ) (mean_l : ℝ) : ℝ :=
  (l.map (λ x => (x - mean_l)^2)).sum / l.length.toFloat

noncomputable def covariance (l1 l2 : List ℝ) (mean_l1 mean_l2 : ℝ) : ℝ :=
  (List.zipWith (λ x y => (x - mean_l1) * (y - mean_l2)) l1 l2).sum / l1.length.toFloat

noncomputable def correlation (l1 l2 : List ℝ) (mean_l1 mean_l2 variance_l1 variance_l2 : ℝ) : ℝ :=
  covariance l1 l2 mean_l1 mean_l2 / (Math.sqrt variance_l1 * Math.sqrt variance_l2)

/- Declaration of the means for X, Y, Z -/
noncomputable def mean_X := mean X
noncomputable def mean_Y := mean Y
noncomputable def mean_Z := mean Z

/- Declaration of the variances for X', Y', Z' -/
noncomputable def var_X := variance X mean_X
noncomputable def var_Y := variance Y mean_Y
noncomputable def var_Z := variance Z mean_Z

/- Declaration of the correlations r(X, Y), r(X, Z), r(Y, Z) -/
noncomputable def r_XY := correlation X Y mean_X mean_Y var_X var_Y
noncomputable def r_XZ := correlation X Z mean_X mean_Z var_X var_Z
noncomputable def r_YZ := correlation Y Z mean_Y mean_Z var_Y var_Z

/- Declaration of the partial correlation coefficients -/
noncomputable def r_X_YZ := (r_YZ - r_XY * r_XZ) / 
                         (Math.sqrt (1 - r_XY ^ 2) * Math.sqrt (1 - r_XZ ^ 2))

noncomputable def r_Y_XZ := (r_XZ - r_XY * r_YZ) / 
                         (Math.sqrt (1 - r_XY ^ 2) * Math.sqrt (1 - r_YZ ^ 2))

/- Declaration of the collective linear correlation coefficient R_{Z, X Y} -/
noncomputable def R_Z_XY := Math.sqrt (r_XZ ^ 2 + r_YZ ^ 2 - 2 * r_XZ * r_YZ * r_XY)

/- Proof problem statement skipping the proof -/
theorem proof_problem : R_Z_XY = 0.935 ∧ r_X_YZ = -0.163 ∧ r_Y_XZ = -0.570 := 
by 
  sorry

end proof_problem_l348_348861


namespace line_tangent_to_circle_l348_348401

noncomputable def circle_eq : ℝ → ℝ → Prop := 
λ x y, x^2 + y^2 = 1

noncomputable def line_eq (θ : ℝ) : ℝ → ℝ → Prop := 
λ x y, x * real.cos θ + y * real.sin θ = 1

theorem line_tangent_to_circle (θ : ℝ) : 
  (∀ x y : ℝ, circle_eq x y → line_eq θ x y → False) ∧
  (∃ x y : ℝ, circle_eq x y ∧ line_eq θ x y) :=
sorry

end line_tangent_to_circle_l348_348401


namespace polar_to_rectangular_l348_348551

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 4) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 :=
by
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l348_348551


namespace true_propositions_are_3_and_4_l348_348026

-- Define the four propositions
def proposition_1 : Prop :=
  ∀ (K X Y : Type), (K² → ∀ (x: X, y: Y), correlation K x y) -> ∃ (x: X, y: Y), correlation K x y

def proposition_2 : Prop :=
  ∀ (x y : ℝ), (regression_line x y = 2.347 * x - 6.423) → negatively_correlated x y

def proposition_3 : Prop :=
  ∀ (a b : ℝ), ¬(a = 0 → a * b = 0) → (a ≠ 0 → a * b ≠ 0)

def proposition_4 : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ), (q > 1 → ∀ n, a (n+1) = a(n) * q) → ∀ n, a(n)^2 < a(n+1)^2

-- Define the statement that asserts ③ and ④ are the true propositions
theorem true_propositions_are_3_and_4 :
  (proposition_3 ∧ proposition_4) ∧ (¬proposition_1) ∧ (¬proposition_2) :=
by
  sorry

end true_propositions_are_3_and_4_l348_348026


namespace mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l348_348721

theorem mrs_mcpherson_contributes (rent : ℕ) (percentage : ℕ) (mrs_mcp_contribution : ℕ) : 
  mrs_mcp_contribution = (percentage * rent) / 100 := by
  sorry

theorem mr_mcpherson_raises (rent : ℕ) (mrs_mcp_contribution : ℕ) : 
  mr_mcp_contribution = rent - mrs_mcp_contribution := by
  sorry

theorem mr_mcpherson_complete_rent : 
  let rent := 1200
  let percentage := 30
  let mrs_mcp_contribution := (percentage * rent) / 100
  let mr_mcp_contribution := rent - mrs_mcp_contribution
  mr_mcp_contribution = 840 := by
  have mrs_contribution : mrs_mcp_contribution = (30 * 1200) / 100 := by
    exact mrs_mcpherson_contributes 1200 30 ((30 * 1200) / 100)
  have mr_contribution : mr_mcp_contribution = 1200 - ((30 * 1200) / 100) := by
    exact mr_mcpherson_raises 1200 ((30 * 1200) / 100)
  show 1200 - 360 = 840 from by
    rw [mrs_contribution, mr_contribution]
    rfl

end mrs_mcpherson_contributes_mr_mcpherson_raises_mr_mcpherson_complete_rent_l348_348721


namespace no_adjacent_beads_probability_l348_348808

theorem no_adjacent_beads_probability :
  let total_arrangements := nat.factorial 7 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 2)
  let valid_arrangements := 12
  let probability := valid_arrangements / total_arrangements
  probability = 2 / 35 :=
by
  sorry

end no_adjacent_beads_probability_l348_348808


namespace expected_value_greater_than_median_l348_348457

noncomputable def density_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, x < a → f x = 0) ∧
  (∀ x, x ≥ b → f x = 0) ∧
  (∀ x, a ≤ x ∧ x < b → 0 < f x) ∧
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y < b → f x ≥ f y) ∧
  continuous_on f (set.Ico a b)

theorem expected_value_greater_than_median
  (X : Type*)
  [probability_space X]
  (a b : ℝ)
  (f : ℝ → ℝ)
  (hf : density_function f a b) :
  E(X) > median(X) :=
by
  sorry

end expected_value_greater_than_median_l348_348457


namespace net_profit_loan_payments_dividends_per_share_director_dividends_l348_348516

theorem net_profit (revenue expenses : ℕ) (tax_rate : ℚ) 
  (h_rev : revenue = 2500000)
  (h_exp : expenses = 1576250)
  (h_tax : tax_rate = 0.2) :
  ((revenue - expenses) - (revenue - expenses) * tax_rate).toNat = 739000 := by
  sorry

theorem loan_payments (monthly_payment : ℕ) 
  (h_monthly : monthly_payment = 25000) :
  (monthly_payment * 12) = 300000 := by
  sorry

theorem dividends_per_share (net_profit loan_payments : ℕ) (total_shares : ℕ)
  (h_net_profit : net_profit = 739000)
  (h_loan_payments : loan_payments = 300000)
  (h_shares : total_shares = 1600) :
  ((net_profit - loan_payments) / total_shares) = 274 := by
  sorry

theorem director_dividends (dividend_per_share : ℕ) (share_percentage : ℚ) (total_shares : ℕ)
  (h_dividend_per_share : dividend_per_share = 274)
  (h_percentage : share_percentage = 0.35)
  (h_shares : total_shares = 1600) :
  (dividend_per_share * share_percentage * total_shares).toNat = 153440 := by
  sorry

end net_profit_loan_payments_dividends_per_share_director_dividends_l348_348516


namespace monotonic_intervals_k_eq_1_max_k_for_f_gt_zero_l348_348619

-- Given function definition f(x)
def f (x k: ℝ) := x * log x + (1 - k) * x + k

-- Part 1: Monotonic intervals for k = 1
theorem monotonic_intervals_k_eq_1 :
  ∀ x : ℝ, f(x, 1) = (x * log x) + 1 → 
  (strict_mono_on (f x 1) (frac 1 (real.exp 1), ∞) ∧
   strict_anti_on (f x 1) (0, frac 1 (real.exp 1))) := 
by 
  sorry

-- Part 2: Maximum integer value of k for x > 1 such that f(x) > 0
theorem max_k_for_f_gt_zero :
  ∀ x: ℝ, x > 1 → (∀ k: ℝ, f x k > 0) → k < 4 :=
by 
  sorry

end monotonic_intervals_k_eq_1_max_k_for_f_gt_zero_l348_348619


namespace first_three_workers_dig_time_l348_348176

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l348_348176


namespace find_lambda_l348_348996

-- Definitions of vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (3, 4)

-- Condition: ( a + λ b ) ∥ c
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ μ : ℝ, ∀ i : ℕ, i < 2 → u.1 = μ • v.1 ∧ u.2 = μ • v.2

theorem find_lambda (λ : ℝ) :
  parallel (1 + λ, 2) c → λ = 1 / 2 :=
sorry

end find_lambda_l348_348996


namespace min_students_l348_348776

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end min_students_l348_348776


namespace total_chips_l348_348999

theorem total_chips (h_prepared : ℕ) (f_prepared : ℕ) (b_prepared : ℕ) :
  h_prepared = 350 → f_prepared = 268 → b_prepared = 182 → 
  h_prepared + f_prepared + b_prepared = 800 :=
by
  intros h_prepared_eq f_prepared_eq b_prepared_eq
  rw [h_prepared_eq, f_prepared_eq, b_prepared_eq]
  rfl

end total_chips_l348_348999


namespace sum_of_floors_arith_seq_l348_348547

/-- The sum of floor values of an arithmetic sequence with an initial term of 0.5, 
    common difference of 0.6, and last term 99.9 is 8316. -/
theorem sum_of_floors_arith_seq :
  let a := 0.5
  let d := 0.6
  let n := 167 -- derived from (99.9 - 0.5)/0.6 + 1
  (∑ k in finset.range n, floor (a + k * d)) = 8316 :=
by
  sorry

end sum_of_floors_arith_seq_l348_348547


namespace number_and_sum_of_f3_l348_348704

def f : ℤ → ℤ

axiom fun_eq : ∀ m n : ℤ, f(m + n) + f(m * n + 1) = f(m) * f(n) + 1

theorem number_and_sum_of_f3 :
  (∃ p t : ℕ, 
   (p = (set.univ.filter (λ y, y = f 3)).size) ∧ 
   (t = (set.univ.filter (λ y, y = f 3)).sum) ∧ 
   (p * t = 1)) :=
sorry

end number_and_sum_of_f3_l348_348704


namespace real_a_of_z_l348_348615

noncomputable def z (a : ℝ) : ℂ := (a + complex.I) / complex.I

theorem real_a_of_z (a : ℝ) (hz : z(a) ∈ ℝ) : a = 0 :=
by 
  sorry

end real_a_of_z_l348_348615


namespace no_consistently_composite_expressions_l348_348839

theorem no_consistently_composite_expressions (n : ℕ) (hn : Nat.Prime n ∧ n > 3) :
  ¬(∀ k ∈ {40, 55, 81, 117, 150}, ¬Nat.Prime (n^2 + k)) :=
sorry

end no_consistently_composite_expressions_l348_348839


namespace smallest_pawns_needed_l348_348770

theorem smallest_pawns_needed (n : ℕ) (hn : n = 1999) : 
  ∃ pawns : ℕ, ∀ (r c : fin (n + 1)), pawns >= n*r + n*c - (r*c) ∧ pawns = 1998001 :=
sorry

end smallest_pawns_needed_l348_348770


namespace part_I_part_II_l348_348621

-- Definitions and problem statements
def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2018 * π - x) * sin (3 * π / 2 + x) - cos x ^ 2 + 1

theorem part_I (k : ℤ) :
  ∀ x : ℝ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 ↔ f x is increasing := sorry

noncomputable def triangle_angle_bisector (A B C : ℝ) (a b c : ℝ) (AD BD : ℝ) :=
  A = B + C → 
  AD = sqrt 2 * BD → 
  AD = 2 → 
  f A = 3 / 2 →
  cos C = (sqrt 6 - sqrt 2) / 4

theorem part_II (A B C a b c AD BD : ℝ) (h : triangle_angle_bisector A B C a b c AD BD) :
  cos C = (sqrt 6 - sqrt 2) / 4 := sorry

end part_I_part_II_l348_348621


namespace solution_set_l348_348744

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l348_348744


namespace cookies_per_tray_needed_l348_348515

-- Definitions for conditions
def trays := 3
def cookies_per_box := 60
def box_cost := 3.50
def total_cost := 14

-- Statement to prove
theorem cookies_per_tray_needed : 
  (total_cost / box_cost * cookies_per_box) / trays = 80 := 
by
  sorry

end cookies_per_tray_needed_l348_348515


namespace different_books_read_l348_348045

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l348_348045


namespace cosine_angle_between_vectors_l348_348940

structure Point3D (α : Type) :=
(x : α) (y : α) (z : α)

def vector_sub {α : Type} [Add α] [Neg α] (P Q : Point3D α) : Point3D α := 
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def dot_product {α : Type} [Mul α] [Add α] (u v : Point3D α) : α :=
  u.x * v.x + u.y * v.y + u.z * v.z

def magnitude {α : Type} [Mul α] [Add α] [Pow α ℕ] [HasToReal α] (u : Point3D α) : Real :=
  sqrt (u.x ^ 2 + u.y ^ 2 + u.z ^ 2).toReal

def cosine_of_angle {α : Type} [Mul α] [Add α] [Neg α] [Pow α ℕ] [HasToReal α] [HasDiv Real ℝ] 
  (u v : Point3D α) : Real :=
  (dot_product u v).toReal / (magnitude u * magnitude v)

theorem cosine_angle_between_vectors (A B C : Point3D ℤ) :
  A = ⟨1, -1, 0⟩ ∧ B = ⟨-2, -1, 4⟩ ∧ C = ⟨8, -1, -1⟩ →
  cosine_of_angle (vector_sub A B) (vector_sub A C) = -1 / sqrt 2 :=
by
  sorry

end cosine_angle_between_vectors_l348_348940


namespace supplement_of_complementary_angle_l348_348602

theorem supplement_of_complementary_angle (α β : ℝ) 
  (h1 : α + β = 90) (h2 : α = 30) : 180 - β = 120 :=
by sorry

end supplement_of_complementary_angle_l348_348602


namespace count_numbers_with_digit_sum_25_l348_348270

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_sum_eq (n : ℕ) (s : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let units := n % 10
  (hundreds + tens + units) = s

theorem count_numbers_with_digit_sum_25 :
  {n : ℕ | is_three_digit_number n ∧ digit_sum_eq n 25}.to_finset.card = 6 := by
  sorry

end count_numbers_with_digit_sum_25_l348_348270


namespace circle_area_l348_348265

theorem circle_area (r : ℝ) (h1 : 6 * (1 / (2 * π * r)) = 2 * r) : 
  let area := π * r^2 in
  area = 3 / 2 :=
by
  -- proof omitted as instructed
  sorry

end circle_area_l348_348265


namespace unique_perpendicular_through_point_l348_348358

-- Definition of perpendicular lines, point, and line
variables (Point Line : Type)

-- Definitions of necessary conditions
def is_perpendicular (l₁ l₂ : Line) : Prop := sorry
def contains_point (l : Line) (p : Point) : Prop := sorry

-- Given conditions
variables (A : Point) (l : Line)

-- Theorem statement
theorem unique_perpendicular_through_point : 
  ∃! m : Line, contains_point m A ∧ is_perpendicular m l :=
sorry

end unique_perpendicular_through_point_l348_348358


namespace two_digit_number_sum_l348_348003

theorem two_digit_number_sum (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : |(10 * a + b) - (10 * b + a)| = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end two_digit_number_sum_l348_348003


namespace expression_value_l348_348149

theorem expression_value :
  3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := 
sorry

end expression_value_l348_348149


namespace probability_between2and5_l348_348773

-- Define the random variable X with a normal distribution
variables (μ σ : ℝ)

-- State the conditions given in the problem
def condition1 : Prop := ∀ X : ℝ, X ∼ Normal μ σ^2
def condition2 : Prop := P(X > 5) = 0.2
def condition3 : Prop := P(X < -1) = 0.2

-- The main problem statement
theorem probability_between2and5 (μ σ : ℝ) (h1 : condition1 μ σ) (h2 : condition2 (Normal μ σ^2)) (h3 : condition3 (Normal μ σ^2)) :
  P(2 < X < 5) = 0.3 :=
sorry

end probability_between2and5_l348_348773


namespace continuous_2D_random_variable_l348_348565

-- Definitions of probability density functions for X and Y
def f1 (x : ℝ) : ℝ := if 0 < x ∧ x < 2 then x / 2 else 0
def f2 (y : ℝ) (xi : ℝ) : ℝ := if xi < y ∧ y < xi + 3 then 1 / 3 else 0

-- Definitions of CDFs derived in the problem
def FX (x : ℝ) : ℝ := x^2 / 4

noncomputable def X (r : ℝ) : ℝ := 2 * Real.sqrt r
noncomputable def Y (r' : ℝ) (r : ℝ) : ℝ := 3 * r' + 2 * Real.sqrt r

-- Lean theorem statement
theorem continuous_2D_random_variable :
  ∀ (r r' : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ 0 ≤ r' ∧ r' ≤ 1 → 
    (X r = 2 * Real.sqrt r ∧ Y r' r = 3 * r' + 2 * Real.sqrt r) :=
by
  intros r r' h
  exact sorry

end continuous_2D_random_variable_l348_348565


namespace solve_problem_l348_348144

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem solve_problem : spadesuit 3 (spadesuit 5 (spadesuit 8 11)) = 1 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l348_348144


namespace eventually_in_lowest_form_l348_348352

-- Definitions and assumptions based on the problem statement
def fraction := {p q : ℕ // Nat.coprime p q}

def next_fraction (f1 f2 : fraction) : fraction :=
  ⟨f1.val.1 + f2.val.1, f1.val.2 + f2.val.2, sorry⟩ -- proof of coprimality is omitted for brevity

-- Statement of the theorem in Lean 4
theorem eventually_in_lowest_form (f1 f2 : fraction) :
  ∃ n : ℕ, ∀ m ≥ n, let f := if m % 2 = 0 then next_fraction f1 f2 else next_fraction f2 f1 in Nat.coprime f.val.1 f.val.2 :=
by sorry

end eventually_in_lowest_form_l348_348352


namespace monotonic_intervals_b_neg1_maximum_value_on_interval_l348_348622

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (x^2 + b*x + b) * Real.sqrt(1 - 2*x)

theorem monotonic_intervals_b_neg1 :
  ∀ (x : ℝ), 
    (x ∈ Set.Iic 0 → MonotoneDecreasing (f x (-1))) ∧ 
    (x ∈ Set.Icc 0 (1/2) → MonotoneIncreasing (f x (-1))) := 
sorry

theorem maximum_value_on_interval :
  ∀ (x : ℝ) (b : ℝ),
    x ∈ Set.Icc (-1) 0 →
    (f x b ≤ f 0 b ∨ f x b ≤ Real.sqrt 3) :=
sorry

end monotonic_intervals_b_neg1_maximum_value_on_interval_l348_348622


namespace min_value_sin_cos_expr_l348_348263

open Real

theorem min_value_sin_cos_expr (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ min_val : ℝ, min_val = 3 * sqrt 2 ∧ ∀ β, (0 < β ∧ β < π / 2) → 
    sin β + cos β + (2 * sqrt 2) / sin (β + π / 4) ≥ min_val :=
by
  sorry

end min_value_sin_cos_expr_l348_348263


namespace find_omega_l348_348644

theorem find_omega (f : ℝ → ℝ) (ω : ℝ) (h1 : ∀ x, f x = 2 * Real.sin (ω * x))
  (h2 : 0 < ω ∧ ω < 1)
  (h3 : ∀ x ∈ Set.Icc 0 (π / 3), f x ≤ sqrt 2 ∧ ∃ y ∈ Set.Icc 0 (π / 3), f y = sqrt 2) :
  ω = 3 / 4 :=
by
  sorry

end find_omega_l348_348644


namespace smallest_sum_of_two_3_digit_numbers_l348_348054

theorem smallest_sum_of_two_3_digit_numbers : 
  (∃ a b c d e f :ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                    d ≠ e ∧ d ≠ f ∧
                    e ≠ f ∧
                    {a, b, c, d, e, f} = {1, 2, 3, 7, 8, 9} ∧
                    (100 * a + 10 * b + c) + (100 * d + 10 * e + f) = 912) :=
sorry

end smallest_sum_of_two_3_digit_numbers_l348_348054


namespace eggs_needed_for_recipe_l348_348112

noncomputable section

theorem eggs_needed_for_recipe 
  (total_eggs : ℕ) 
  (rotten_eggs : ℕ) 
  (prob_all_rotten : ℝ)
  (h_total : total_eggs = 36)
  (h_rotten : rotten_eggs = 3)
  (h_prob : prob_all_rotten = 0.0047619047619047615) 
  : (2 : ℕ) = 2 :=
by
  sorry

end eggs_needed_for_recipe_l348_348112


namespace sum_of_coefficients_factors_l348_348384

theorem sum_of_coefficients_factors :
  ∃ (a b c d e : ℤ), 
    (343 * (x : ℤ)^3 + 125 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 51) :=
sorry

end sum_of_coefficients_factors_l348_348384


namespace triangle_area_orthogonality_condition_l348_348659

section Hyperbola_Triangle_Area

def hyperbola_eq (x y : ℝ) := 2 * x^2 - y^2 = 1
def left_vertex : ℝ × ℝ := (-sqrt 2 / 2, 0)
def asymptote1 (x : ℝ) := sqrt 2 * x
def line_through_vertex (x : ℝ) := sqrt 2 * (x + sqrt 2 / 2)

theorem triangle_area : 
  let A := left_vertex
  let line := line_through_vertex
  let intersect_x : ℝ := -sqrt 2 / 4
  let intersect_y : ℝ := 1 / 2
  let area := 1 / 2 * abs (fst A) * abs intersect_y 
  in area = sqrt 2 / 8 := 
sorry

end Hyperbola_Triangle_Area

section Hyperbola_Orthogonality

def hyperbola_eq (x y : ℝ) := 2 * x^2 - y^2 = 1
def circle_eq (x y : ℝ) := x^2 + y^2 = 1
def line_eq (x b : ℝ) := x + b

theorem orthogonality_condition (b : ℝ) : 
  b^2 = 2 → (∀ x1 y1 x2 y2 : ℝ, 
  hyperbola_eq x1 y1 → hyperbola_eq x2 y2 →
  circle_eq (x1 + x2) (y1 + y2) → 
  line_eq x1 b = -1 / b ∧ line_eq x2 b = -1 / b → 
  x1 * x2 + y1 * y2 = 0) := 
sorry

end Hyperbola_Orthogonality

end triangle_area_orthogonality_condition_l348_348659


namespace central_angle_of_cone_lateral_surface_l348_348377

   -- Define the base radius and height of the cone
   def base_radius : ℝ := 1
   def height : ℝ := 2

   -- Define the slant height using the Pythagorean theorem
   def slant_height : ℝ := Real.sqrt (base_radius ^ 2 + height ^ 2)

   -- The formula for the central angle α of the cone's lateral surface when it is unfolded
   def central_angle : ℝ := (2 * Real.sqrt 5 / 5) * Real.pi

   -- Statement to be proven
   theorem central_angle_of_cone_lateral_surface :
     ∀ (r h : ℝ), r = base_radius ∧ h = height →
     2 * Real.pi = central_angle * slant_height := by
     intros r h H
     sorry
   
end central_angle_of_cone_lateral_surface_l348_348377


namespace minimum_value_l348_348978

variable (a b : ℝ)

-- Assume a and b are positive real numbers
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)

-- Given the condition a + b = 2
variable (h₂ : a + b = 2)

theorem minimum_value : (1 / a) + (2 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end minimum_value_l348_348978


namespace length_of_common_chord_l348_348633

theorem length_of_common_chord (x y : ℝ) :
  (x + 1)^2 + (y - 3)^2 = 9 ∧ x^2 + y^2 - 4 * x + 2 * y - 11 = 0 → 
  ∃ l : ℝ, l = 24 / 5 :=
by
  sorry

end length_of_common_chord_l348_348633


namespace exists_closed_hemisphere_with_four_points_l348_348594

-- Define the existence of a closed hemisphere containing at least four out of five points on a sphere
theorem exists_closed_hemisphere_with_four_points (points : Fin 5 → Point) (sphere : Sphere):
  ∃ (hemisphere : Hemisphere), ∃ (subset : Set Point), 
    subset ⊆ points.to_set ∧ subset.card ≥ 4 ∧ subset ⊆ hemisphere.points :=
sorry

end exists_closed_hemisphere_with_four_points_l348_348594


namespace quadratic_discriminant_l348_348059

-- Define the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -11
def c : ℤ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- assert the discriminant for given coefficients
theorem quadratic_discriminant : discriminant a b c = 81 :=
by
  sorry

end quadratic_discriminant_l348_348059


namespace solve_for_r_l348_348157

theorem solve_for_r (r : ℚ) (h : 4 * (r - 10) = 3 * (3 - 3 * r) + 9) : r = 58 / 13 :=
by
  sorry

end solve_for_r_l348_348157


namespace printing_presses_equivalence_l348_348312

theorem printing_presses_equivalence :
  ∃ P : ℕ, (500000 / 12) / P = (500000 / 14) / 30 ∧ P = 26 :=
by
  sorry

end printing_presses_equivalence_l348_348312


namespace conditional_prob_B_given_A_and_C_l348_348335

variable (A B C : Type) [ProbabilitySpace A] [ProbabilitySpace B] [ProbabilitySpace C]

variable (p_A : Probability A)
variable (p_B : Probability B)
variable (p_C : Probability C)
variable (p_A_and_B : Probability (A ∩ B))
variable (p_A_and_C : Probability (A ∩ C))
variable (p_B_and_C : Probability (B ∩ C))
variable (p_A_and_B_and_C : Probability (A ∩ B ∩ C))

axiom pA : p_A = 8 / 15
axiom pB : p_B = 4 / 15
axiom pC : p_C = 3 / 15
axiom pA_andB : p_A_and_B = 2 / 15
axiom pA_andC : p_A_and_C = 6 / 15
axiom pB_andC : p_B_and_C = 1 / 15
axiom pA_andB_andC : p_A_and_B_and_C = 1 / 30

def conditional_probability (p_inter : Probability (A ∩ B ∩ C)) (p_single : Probability (A ∩ C)) : Probability B :=
  p_inter / p_single

theorem conditional_prob_B_given_A_and_C : conditional_probability p_A_and_B_and_C p_A_and_C = 1 / 12 :=
by
  sorry

end conditional_prob_B_given_A_and_C_l348_348335


namespace largest_share_l348_348170

theorem largest_share (total_profit : ℕ) (ratio : List ℕ) 
  (h_total : total_profit = 42000) (h_ratio : ratio = [3, 3, 4, 5, 6]) : 
  ∃ (largest_share : ℕ), largest_share = 12000 := 
by
  have total_parts : ℕ := ratio.sum
  have h_parts : total_parts = 21 := by sorry
  let value_per_part := total_profit / total_parts
  have h_value_per_part : value_per_part = 2000 := by sorry
  let largest_ratio := ratio.maximum.get_or_else 0
  have h_largest_ratio : largest_ratio = 6 := by sorry
  have largest_share := largest_ratio * value_per_part
  exists largest_share
  have h_largest_share : largest_share = 12000 := by sorry
  exact h_largest_share

end largest_share_l348_348170


namespace a_n_formula_l348_348535

open Nat 

noncomputable def a_n (n : ℕ) : ℚ :=
  2 * ∏ k in finset.range (n + 1).succ \ {0, 1}, (1 - 1 / (k : ℚ)^2)

theorem a_n_formula (n : ℕ) (hn : n > 0) : a_n n = (n + 2) / (n + 1) := 
by 
  sorry

end a_n_formula_l348_348535


namespace max_sum_of_arithmetic_sequence_l348_348977

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (d : ℤ) (h_a : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 2 + a 3 = 156)
  (h2 : a 2 + a 3 + a 4 = 147) :
  ∃ n : ℕ, n = 19 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end max_sum_of_arithmetic_sequence_l348_348977


namespace average_speed_l348_348407

theorem average_speed
    (distance1 distance2 : ℕ)
    (time1 time2 : ℕ)
    (h1 : distance1 = 100)
    (h2 : distance2 = 80)
    (h3 : time1 = 1)
    (h4 : time2 = 1) :
    (distance1 + distance2) / (time1 + time2) = 90 :=
by
  sorry

end average_speed_l348_348407


namespace fx_eq_neg_one_l348_348226

theorem fx_eq_neg_one (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^(2-m) ∧ ∀ y, f (-y) = -f y) ∧ 
  Interval (-3 - m) (m^2 - m) ⊆ {x : ℝ | ∃ y, f x = y}) → f m = -1 :=
sorry

end fx_eq_neg_one_l348_348226


namespace rectangle_length_35_l348_348366

theorem rectangle_length_35
  (n_rectangles : ℕ) (area_abcd : ℝ) (rect_length_multiple : ℕ) (rect_width_multiple : ℕ) 
  (n_rectangles_eq : n_rectangles = 6)
  (area_abcd_eq : area_abcd = 4800)
  (rect_length_multiple_eq : rect_length_multiple = 3)
  (rect_width_multiple_eq : rect_width_multiple = 2) :
  ∃ y : ℝ, round y = 35 ∧ y^2 * (4/3) = area_abcd :=
by
  sorry


end rectangle_length_35_l348_348366


namespace omega_range_l348_348230

noncomputable def f (ω x : ℝ) := sin (ω * x) + cos (ω * x)

theorem omega_range (ω : ℝ) (hω : ω > 0) 
  (h_monotonic : ∀ x y : ℝ, (π / 2 < x ∧ x < y ∧ y < π) → f ω x ≥ f ω y) : 
  ω ∈ set.Icc (1 / 2 : ℝ) (5 / 4 : ℝ) :=
sorry

end omega_range_l348_348230


namespace complex_expression_evaluation_l348_348710

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := x / 3
def h (x : ℝ) : ℝ := x^2
def f_inv (x : ℝ) : ℝ := x - 2
def g_inv (x : ℝ) : ℝ := 3 * x
def h_inv (x : ℝ) : ℝ := Real.sqrt x

theorem complex_expression_evaluation :
    f (h (g_inv (f_inv (h_inv (g (f (f 5))))))) = 191 - 36 * Real.sqrt 17 := by
  sorry

end complex_expression_evaluation_l348_348710


namespace conic_section_is_parabola_l348_348846

theorem conic_section_is_parabola (x y : ℝ) :
  abs (y - 3) = sqrt ((x + 4)^2 + y^2) → (∃ a b c A B : ℝ, (a * x^2 + b * x + c = A * y + B) ∧ (a ≠ 0 ∧ A ≠ 0)) :=
by
  sorry

end conic_section_is_parabola_l348_348846


namespace finding_m_l348_348981

-- Given definitions representing points P, M, and N with their coordinates
variables {m : ℝ}
def P := (3, m)
def M := (2, -1)
def N := (-3, 4)

-- Proving that point P lies on the line passing through M and N means m = -2
theorem finding_m 
  (h1 : ∃ (l : ℝ), ∀ t : ℝ, P = (2 + t * (-5), -1 + t * 5)) : 
  m = -2 :=
sorry

end finding_m_l348_348981


namespace number_of_true_props_l348_348485

def prop1 : Prop := 
  ∀ (R2 : ℝ), R2 ≥ 0 → R2 ≤ 1 → (R2 determines goodness of fit)

def prop2 : Prop := 
  ∀ (r : ℝ), r ≥ -1 → r ≤ 1 → (higher linear correlation implies |r| ≈ 1)

def prop3 : Prop := 
  ∀ (x : ℕ → ℝ), var x = 1 → var (2 • x) = 4

def prop4 : Prop := 
  (∀ (k0 : ℝ) (x y : Type), smaller k0 implies greater certainty of the relationship between x and y)

theorem number_of_true_props : 
  (if prop1 then 1 else 0) + 
  (if prop2 then 1 else 0) + 
  (if prop3 then 1 else 0) + 
  (if prop4 then 1 else 0) = 1 :=
sorry

end number_of_true_props_l348_348485


namespace exists_m_with_at_least_2015_solutions_l348_348712

noncomputable def φ (n : ℕ) : ℕ := (finset.range n).filter (nat.coprime n).card

theorem exists_m_with_at_least_2015_solutions :
  ∃ m, (finset.range 2015.succ).card ≤ (finset.filter (λ n, φ n = m) (finset.range (n : ℕ).succ)).card :=
sorry

end exists_m_with_at_least_2015_solutions_l348_348712


namespace max_value_of_g_l348_348955

def g (x : ℝ) : ℝ :=
  min (3 * x + 3) (min ((1 / 3) * x + 2) (- (1 / 2) * x + 8))

theorem max_value_of_g : ∃ x, g x = 22 / 5 := by
  sorry

end max_value_of_g_l348_348955


namespace simplify_fraction_l348_348364

theorem simplify_fraction : (7 + 14 * Complex.i) / (3 - 4 * Complex.i) = (77 / 25) + (70 / 25) * Complex.i :=
by
  -- Skipping the proof for now.
  sorry

end simplify_fraction_l348_348364


namespace min_students_l348_348777

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end min_students_l348_348777


namespace solution_of_system_l348_348753

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l348_348753


namespace problem_solution_l348_348959

theorem problem_solution (n : ℕ) (h : 4 * 6 * 3 * n = 8!) : n = 560 :=
by {
  sorry -- solution is omitted as only the statement is required.
}

end problem_solution_l348_348959


namespace number_of_intersections_l348_348550

theorem number_of_intersections : 
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + 4 * y = 6
  let line3 (x y : ℝ) := 6 * x - 9 * y = 8
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    ((∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1 ∧ p1 = (x1, y1) ∧
               (∃ x2 y2, line1 x2 y2 ∧ line3 x2 y2 ∧ p1 = (x2, y2) )) ∨
     (∃ x1 y1, line2 x1 y1 ∧ line3 x1 y1 ∧ p1 = (x1, y1))) ∧
    ((∃ x1 y1, line2 x1 y1 ∧ line3 x1 y1 ∧ p2 = (x1, y1) ∧
               (∃ x2 y2, line1 x2 y2 ∧ line3 x2 y2 ∧ p2 = (x2, y2) )) ∨
     (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1 ∧ p2 = (x1, y1))) ∧
    ¬(∃ x y, p1 = p2 := (x, y)) :=
by sorry

end number_of_intersections_l348_348550


namespace angle_is_60_degrees_l348_348906

-- Definitions
def angle_is_twice_complementary (x : ℝ) : Prop := x = 2 * (90 - x)

-- Theorem statement
theorem angle_is_60_degrees (x : ℝ) (h : angle_is_twice_complementary x) : x = 60 :=
by sorry

end angle_is_60_degrees_l348_348906


namespace acute_angle_at_3_37_l348_348835

-- Definitions based on the conditions
def degrees_per_hour_mark := 360 / 12
def minute_angle (m : ℕ) := (m / 60) * 360
def hour_angle (h m : ℕ) := (h * degrees_per_hour_mark) + ((m / 60) * degrees_per_hour_mark)

theorem acute_angle_at_3_37 :
  let m := 37 -- minute
  let h := 3  -- hour
  ∃ (angle : ℝ), angle = 113.5 ∧ 
                  let angle_difference := minute_angle m - hour_angle h m in
                  min angle_difference (360 - angle_difference) = angle :=
by {
  let m := 37
  let h := 3
  let degrees_per_hour_mark := 360 / 12
  let minute_angle (m : ℕ) := (m / 60) * 360
  let hour_angle (h m : ℕ) := (h * degrees_per_hour_mark) + ((m / 60) * degrees_per_hour_mark)
  use 113.5,
  split,
  { refl },
  sorry -- The actual proof goes here
}

end acute_angle_at_3_37_l348_348835


namespace original_cost_price_l348_348852

theorem original_cost_price (sp_friend : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) (final_sp : ℝ) :
  sp_friend = 0.85 * sp_friend / (1 + gain_percent) →
  loss_percent = 0.15 →
  gain_percent = 0.20 →
  final_sp = 54000 →
  let x := sp_friend / (1 - loss_percent) in
  x = 52941.18 :=
  sorry

end original_cost_price_l348_348852


namespace solution_of_system_l348_348755

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l348_348755


namespace sum_of_997_lemons_l348_348304

-- Define x and y as functions of k
def x (k : ℕ) := 1 + 9 * k
def y (k : ℕ) := 110 - 7 * k

-- The theorem we need to prove
theorem sum_of_997_lemons :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ 15 ∧ 7 * (x k) + 9 * (y k) = 997 := 
by
  sorry -- Proof to be filled in

end sum_of_997_lemons_l348_348304


namespace trip_total_charge_l348_348320

noncomputable def initial_fee : ℝ := 2.25
noncomputable def additional_charge_per_increment : ℝ := 0.25
noncomputable def increment_length : ℝ := 2 / 5
noncomputable def trip_length : ℝ := 3.6

theorem trip_total_charge :
  initial_fee + (trip_length / increment_length) * additional_charge_per_increment = 4.50 :=
by
  sorry

end trip_total_charge_l348_348320


namespace value_of_2a_plus_b_l348_348271

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let f' := (fun x => (1 : ℝ) / x - a)
  let slope_perpendicular_line := - (1/3 : ℝ)
  f' 1 * slope_perpendicular_line = -1 

def point_on_function (a b : ℝ) : Prop :=
  f a 1 = b

theorem value_of_2a_plus_b (a b : ℝ) 
  (h_tangent_perpendicular : is_tangent_perpendicular a b)
  (h_point_on_function : point_on_function a b) : 
  2 * a + b = -2 := sorry

end value_of_2a_plus_b_l348_348271


namespace count_vars_in_denominator_l348_348117

theorem count_vars_in_denominator :
  let fractions := [6 / 1, 4 / y, y / 4, 6 / (x + 1), y / Real.pi, (x + y) / 2] in
  (fractions.count (λ f, match f with
                        | 6 / 1 => false
                        | 4 / y => true
                        | y / 4 => false
                        | 6 / (x + 1) => true
                        | y / Real.pi => false
                        | (x + y) / 2 => false
                        | _ => false
                        end)) = 2 :=
by sorry

end count_vars_in_denominator_l348_348117


namespace problem_l348_348250

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l348_348250


namespace calc_value_l348_348537

theorem calc_value (a : ℝ) (h1 : a ≠ 0) : 
  (1/8) * (a^0) + ((1 / (8 * a))^0) - (32^(-1/2)) - ((-16)^(-3/5)) = 1 + (3/16) - (1 / (4 * real.sqrt 2)) := 
by
  sorry

end calc_value_l348_348537


namespace shortest_distance_translation_correct_l348_348809

noncomputable def shortest_distance_to_tangent_line : Real :=
  let line (c : ℝ) : ℝ → ℝ → Prop := λ x y, x - y + c = 0
  let circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1
  let distance (c : ℝ) : ℝ := abs (1 + c) / Real.sqrt 2
  let c_val : Set ℝ := {c | distance c = 1}
  let shortest_distance_translation : ℝ := abs (2 - Real.sqrt 2) / Real.sqrt 2
  shortest_distance_translation

theorem shortest_distance_translation_correct :
  shortest_distance_to_tangent_line = Real.sqrt 2 - 1 := sorry

end shortest_distance_translation_correct_l348_348809


namespace max_value_x1_x2_l348_348991

noncomputable def f (x : ℝ) := 1 - Real.sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) := 2 * Real.log x

theorem max_value_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≤ 2 / 3) (h2 : x2 > 0) (h3 : x1 - x2 = (1 - Real.sqrt (2 - 3 * x1)) - (2 * Real.log x2)) :
  x1 - x2 ≤ -25 / 48 :=
sorry

end max_value_x1_x2_l348_348991


namespace cost_price_of_radio_l348_348469

theorem cost_price_of_radio (SP : ℝ) (profit_percent : ℝ) (overhead : ℝ) : 
  SP = 300 → profit_percent = 0.25 → overhead = 15 → 
  let CP := SP / (1 + profit_percent) 
  in CP + overhead = 255 := 
by
  intros hSP hprofit hoverhead
  let CP := SP / (1 + profit_percent)
  have : CP + overhead = (300 / (1 + 0.25)) + 15 := by sorry
  simp [hSP, hprofit, hoverhead] at this
  exact this

end cost_price_of_radio_l348_348469


namespace x_intercepts_sin_3_over_x_l348_348163

theorem x_intercepts_sin_3_over_x :
  let y (x : ℝ) := Real.sin (3 / x)
  let interval := Ioo 0.01 0.1
  let x_intercepts := {x ∈ interval | y x = 0}
  (x_intercepts.card = 86) :=
by
  let y (x : ℝ) := Real.sin (3 / x)
  let interval := Set.Ioo (0.01 : ℝ) (0.1 : ℝ)
  let x_intercepts := {x ∈ interval | y x = 0}
  have h : (x_intercepts.toFinset.card = 86) := sorry
  exact h

end x_intercepts_sin_3_over_x_l348_348163


namespace chi_square_test_probability_not_attended_l348_348669

-- Definitions for Part 1
def n := 100
def a := 40
def b := 10
def c := 20
def d := 30

noncomputable def K_squared := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
def k_0 := 6.635

-- Definitions for Part 2
def total_students := 6
def attended_training := 4
def not_attended_training := 2

noncomputable def probability_at_least_one_not_attended := 9 / 15 -- Because 9 favorable outcomes out of 15 possible.

-- Lean theorem statements:

-- Part 1: We can be 99% certain
theorem chi_square_test :
  K_squared > k_0 :=
by
  sorry

-- Part 2: Probability of selecting at least one student who did not attend training
theorem probability_not_attended :
  probability_at_least_one_not_attended = 3 / 5 :=
by
  sorry

end chi_square_test_probability_not_attended_l348_348669


namespace ribbon_proof_l348_348816

theorem ribbon_proof :
  ∃ (ribbons : List ℝ), 
    ribbons = [9, 9, 24, 24, 24] ∧
    (∃ (initial_ribbons : List ℝ) (cuts : List ℝ),
      initial_ribbons = [15, 20, 24, 26, 30] ∧
      ((List.sum initial_ribbons / initial_ribbons.length) - (List.sum ribbons / ribbons.length) = 5) ∧
      (List.median ribbons = List.median initial_ribbons) ∧
      (List.range ribbons = List.range initial_ribbons)) :=
sorry

end ribbon_proof_l348_348816


namespace polynomial_remainder_l348_348339

noncomputable def h (x : ℕ) := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder :
  ∀ (x : ℕ), h(x) ≠ 0 → 
  let h_x_10 := h(x^10) in (h_x_10 % h(x) = 7) :=
by
  intro x h_x_nonzero,
  sorry

end polynomial_remainder_l348_348339


namespace min_arithmetic_mean_value_l348_348205

noncomputable def min_arithmetic_mean (a_1 a_2 a_3 : ℝ) (h_pos : a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0) 
  (h_cond : 2 * a_1 + 3 * a_2 + a_3 = 1) : ℝ :=
  (1 / (a_1 + a_2) + 1 / (a_2 + a_3)) / 2

theorem min_arithmetic_mean_value : 
  ∀ (a_1 a_2 a_3 : ℝ), 
  (a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0) →
  2 * a_1 + 3 * a_2 + a_3 = 1 →
  min_arithmetic_mean a_1 a_2 a_3 (by repeat { split; assumption }) (by assumption) = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_arithmetic_mean_value_l348_348205


namespace decreasing_sequence_b_l348_348203

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a n * a (n + 1) = (a n)^2 + 1

def b_n (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = (a n - 1) / (a n + 1)

theorem decreasing_sequence_b {a b : ℕ → ℝ} (h1 : seq_a a) (h2 : b_n a b) :
  ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end decreasing_sequence_b_l348_348203


namespace largest_zip_code_l348_348793

/-- The seven digits of Lisa's phone number 465-3271 -/
def lisa_phone_number : List ℕ := [4, 6, 5, 3, 2, 7, 1]

/-- The condition that the sum of the digits in Lisa's phone number is 28 -/
def phone_number_sum : Prop := lisa_phone_number.sum = 28

/-- A function to compute the sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The main theorem to prove the largest possible value of Lisa's zip code -/
theorem largest_zip_code :
  phone_number_sum →
  (∃ zip_code : ℕ, digit_sum zip_code = 28 ∧ zip_code.digits 10.Nodup ∧ zip_code = 9865) :=
by
  sorry

end largest_zip_code_l348_348793


namespace probability_of_rolling_five_on_six_sided_die_l348_348104

theorem probability_of_rolling_five_on_six_sided_die :
  let S := {1, 2, 3, 4, 5, 6}
  let |S| := 6
  let A := {5}
  let |A| := 1
  probability A S = 1 / 6 := by
  -- Proof goes here
  sorry

end probability_of_rolling_five_on_six_sided_die_l348_348104


namespace Ken_and_Kendra_fish_count_l348_348693

def Ken_and_Kendra_bring_home (kendra_fish_caught : ℕ) (ken_ratio : ℕ) (ken_releases : ℕ) : ℕ :=
  let ken_fish_caught := ken_ratio * kendra_fish_caught
  let ken_fish_brought_home := ken_fish_caught - ken_releases
  ken_fish_brought_home + kendra_fish_caught

theorem Ken_and_Kendra_fish_count :
  let kendra_fish_caught := 30 in
  let ken_ratio := 2 in
  let ken_releases := 3 in
  Ken_and_Kendra_bring_home kendra_fish_caught ken_ratio ken_releases = 87 :=
by
  sorry

end Ken_and_Kendra_fish_count_l348_348693


namespace tires_in_parking_lot_l348_348273

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l348_348273


namespace alien_socks_and_shoes_l348_348488

theorem alien_socks_and_shoes :
  let total_feet := 4 in 
  let total_items := total_feet * 2 in
  let total_ways := Nat.choose total_items total_feet in
  total_ways = 70 :=
by sorry

end alien_socks_and_shoes_l348_348488


namespace angle_bisector_angle_BMC_l348_348327

variable {A B C A1 B1 C1 M : Type}
variable [Nonempty A] [Points A B C] [Concurrent {A1, B1, C1}]
           [Projection M A1 (Line B1 C1)]

theorem angle_bisector_angle_BMC :
  is_angle_bisector (Line M A1) (Angle ∠ B M C) :=
sorry

end angle_bisector_angle_BMC_l348_348327


namespace vanessa_ribbon_length_l348_348378

theorem vanessa_ribbon_length (area : ℕ) (pi_estimate : ℚ) (extra_percentage : ℚ) (circumference_ribbon : ℕ) :
  area = 616 → 
  pi_estimate = 22 / 7 → 
  extra_percentage = 1 / 10 → 
  circumference_ribbon = 97 := 
by
  intro h_area h_pi h_extra
  have h_r_squared : ℚ := (7 * 616) / 22
  have h_r : ℚ := real.sqrt h_r_squared
  have h_circumference : ℚ := 2 * pi_estimate * h_r
  have h_total_ribbon := h_circumference * (1 + extra_percentage)
  have h_approximation : circumference_ribbon = h_total_ribbon.ceil.to_nat
  exact h_approximation
  sorry

end vanessa_ribbon_length_l348_348378


namespace village_population_l348_348874

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 :=
sorry

end village_population_l348_348874


namespace ratio_of_smaller_to_trapezoid_l348_348503

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

def ratio_of_areas : ℝ :=
  let side_large := 10
  let side_small := 5
  let area_large := area_equilateral_triangle side_large
  let area_small := area_equilateral_triangle side_small
  let area_trapezoid := area_large - area_small
  area_small / area_trapezoid

theorem ratio_of_smaller_to_trapezoid :
  ratio_of_areas = 1 / 3 :=
sorry

end ratio_of_smaller_to_trapezoid_l348_348503


namespace percentage_tax_raise_expecting_population_l348_348857

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end percentage_tax_raise_expecting_population_l348_348857


namespace most_likely_dissatisfied_passengers_correct_expected_dissatisfied_passengers_correct_variance_dissatisfied_passengers_correct_l348_348411

noncomputable def most_likely_dissatisfied_passengers (n : ℕ) : ℕ :=
  1

theorem most_likely_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  most_likely_dissatisfied_passengers n = 1 :=
by
  sorry

noncomputable def expected_dissatisfied_passengers (n : ℕ) : ℝ :=
  sqrt (n / π)

theorem expected_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in 
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  expected_dissatisfied_passengers n ≈ sqrt (n / π) :=
by
  sorry

noncomputable def variance_dissatisfied_passengers (n : ℕ) : ℝ :=
  0.182 * n

theorem variance_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  variance_dissatisfied_passengers n ≈ 0.182 * n :=
by
  sorry

end most_likely_dissatisfied_passengers_correct_expected_dissatisfied_passengers_correct_variance_dissatisfied_passengers_correct_l348_348411


namespace quadratic_inequality_solution_l348_348269

theorem quadratic_inequality_solution (a b: ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) ∧
  (2 + 3 = -a) ∧
  (2 * 3 = b) →
  ∀ x : ℝ, (b * x^2 + a * x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end quadratic_inequality_solution_l348_348269


namespace find_t_l348_348811

noncomputable def point_A := (0 : ℝ, 7 : ℝ)
noncomputable def point_B := (3 : ℝ, 0 : ℝ)
noncomputable def point_C := (9 : ℝ, 0 : ℝ)

def horizontal_line (t : ℝ) := { p : ℝ × ℝ | p.2 = t }

def line_eq_AB (x : ℝ) : ℝ := -7/3 * x + 7
def line_eq_AC (x : ℝ) : ℝ := -7/9 * x + 7

def point_T (t : ℝ) := (3 / 7 * (7 - t), t)
def point_U (t : ℝ) := (9 / 7 * (7 - t), t)

def length_TU (t : ℝ) : ℝ := abs ((9 / 7) * (7 - t) - (3 / 7) * (7 - t))
def height_A_TU (t : ℝ) : ℝ := 7 - t

def area_ΔATU (t : ℝ) : ℝ := (1 / 2) * length_TU t * height_A_TU t

theorem find_t : ∃ t : ℝ, area_ΔATU t = 18 ∧ t = 7 - sqrt 42 :=
by
  sorry

end find_t_l348_348811


namespace maxCars_l348_348505

-- Define the triangular grid properties and conditions
def isValidCell (x y z : ℕ) : Prop := (x + y + z <= 10)

-- Define the car control conditions
def controlArea (x y z : ℕ) : set (ℕ × ℕ × ℕ) :=
  {c | c.1 + c.2 + c.3 <= 10 ∧ (c.1 = x ∨ c.2 = y ∨ c.3 = z)}

-- Define the function that checks if a car can be placed without overlap
def noOverlap (cars : list (ℕ × ℕ × ℕ)) : Prop :=
  ∀ i j, i ≠ j → controlArea (cars.nth_le i sorry).1 (cars.nth_le i sorry).2 (cars.nth_le i sorry).3
             ∩ controlArea (cars.nth_le j sorry).1 (cars.nth_le j sorry).2 (cars.nth_le j sorry).3 = ∅

-- Define the chessboard side length
def sideLength : ℕ := 10

-- The main theorem stating the maximum number of cars that can be placed
theorem maxCars (k : ℕ) (cars : list (ℕ × ℕ × ℕ)) :
  length cars = k →
  ∀ car ∈ cars, isValidCell car.1 car.2 car.3 →
  noOverlap cars →
  k ≤ 7 :=
by
  sorry

end maxCars_l348_348505


namespace problem_proof_l348_348869

noncomputable def problem_statement (a a1 b b1 c c1 S S1 : ℝ) : Prop :=
  a1^2 * (-a^2 + b^2 + c^2) + b1^2 * (a^2 - b^2 + c^2) + c1^2 * (a^2 + b^2 - c^2) ≥ 16 * S * S1

theorem problem_proof (a a1 b b1 c c1 S S1 : ℝ) (hS : S = 1/2 * abs a * abs b * sin (some_angle)) (hS1 : S1 = 1/2 * abs a1 * abs b1 * sin (some_angle1)) :
  problem_statement a a1 b b1 c c1 S S1 :=
begin
  sorry
end

end problem_proof_l348_348869


namespace ceo_dividends_correct_l348_348529

-- Definitions of parameters
def revenue := 2500000
def expenses := 1576250
def tax_rate := 0.2
def monthly_loan_payment := 25000
def months := 12
def number_of_shares := 1600
def ceo_ownership := 0.35

-- Calculation functions based on conditions
def net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
def loan_payments := monthly_loan_payment * months
def dividends_per_share := (net_profit - loan_payments) / number_of_shares
def ceo_dividends := dividends_per_share * ceo_ownership * number_of_shares

-- Statement to prove
theorem ceo_dividends_correct : ceo_dividends = 153440 :=
by 
  -- skipping the proof
  sorry

end ceo_dividends_correct_l348_348529


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l348_348982

-- Definitions and conditions
def square (x : ℝ) : ℝ := x * x
def sqrt (x : ℝ) : ℝ := Real.sqrt x

variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a + 2 * Real.sqrt b > 0) (haminb : a - 2 * Real.sqrt b > 0)

-- The problems translated into Lean statements
theorem problem_1 : sqrt (5 - 2 * sqrt 6) = sqrt 3 - sqrt 2 :=
sorry

theorem problem_2 : sqrt (12 + 2 * sqrt 35) = sqrt 7 + sqrt 5 :=
sorry

theorem problem_3 : sqrt (9 + 6 * sqrt 2) = sqrt 6 + sqrt 3 :=
sorry

theorem problem_4 : sqrt (16 - 4 * sqrt 15) = sqrt 10 - sqrt 6 :=
sorry

theorem problem_5 : sqrt (3 - sqrt 5) + sqrt (2 + sqrt 3) = (sqrt 10 + sqrt 6) / 2 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l348_348982


namespace max_value_expression_l348_348663

theorem max_value_expression : 
  ∃ (a b c d ∈ {1, 2, 5, 6}), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a - b) ^ 2 + c * d = 35 :=
by sorry

end max_value_expression_l348_348663


namespace vector_collinear_k_values_l348_348962

theorem vector_collinear_k_values (k : ℝ) :
  let a := (2, k)
  let b := (k - 1, k * (k + 1))
  in (2 * (k * (k + 1))) = (k * (k - 1)) → k = -3 ∨ k = 0 :=
by
  intros a b
  simp [a, b]
  sorry

end vector_collinear_k_values_l348_348962


namespace ceo_dividends_correct_l348_348530

-- Definitions of parameters
def revenue := 2500000
def expenses := 1576250
def tax_rate := 0.2
def monthly_loan_payment := 25000
def months := 12
def number_of_shares := 1600
def ceo_ownership := 0.35

-- Calculation functions based on conditions
def net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
def loan_payments := monthly_loan_payment * months
def dividends_per_share := (net_profit - loan_payments) / number_of_shares
def ceo_dividends := dividends_per_share * ceo_ownership * number_of_shares

-- Statement to prove
theorem ceo_dividends_correct : ceo_dividends = 153440 :=
by 
  -- skipping the proof
  sorry

end ceo_dividends_correct_l348_348530


namespace total_money_l348_348900

theorem total_money (A B C : ℕ) (h1 : A + C = 400) (h2 : B + C = 750) (hC : C = 250) :
  A + B + C = 900 :=
sorry

end total_money_l348_348900


namespace bus_driver_overtime_percentage_increase_l348_348084

theorem bus_driver_overtime_percentage_increase :
  let regular_rate := 18
  let total_earnings := 976
  let total_hours := 48.12698412698413
  let regular_hours := 40
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := total_earnings - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  (overtime_rate - regular_rate) / regular_rate * 100 = 75 :=
by
  let regular_rate := 18
  let total_earnings := 976
  let total_hours := 48.12698412698413
  let regular_hours := 40
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := total_earnings - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  have overtime_increase : (overtime_rate - regular_rate) / regular_rate * 100 = 75
  sorry

end bus_driver_overtime_percentage_increase_l348_348084


namespace greatest_integer_function_of_pi_plus_3_l348_348128

noncomputable def pi_plus_3 : Real := Real.pi + 3

theorem greatest_integer_function_of_pi_plus_3 : Int.floor pi_plus_3 = 6 := 
by
  -- sorry is used to skip the proof
  sorry

end greatest_integer_function_of_pi_plus_3_l348_348128


namespace tires_in_parking_lot_l348_348274

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l348_348274


namespace value_of_f_neg_t_l348_348011

def f (x : ℝ) : ℝ := 3 * x + Real.sin x + 1

theorem value_of_f_neg_t (t : ℝ) (h : f t = 2) : f (-t) = 0 :=
by
  sorry

end value_of_f_neg_t_l348_348011


namespace price_of_larger_jar_l348_348462

-- Define the context and problem
theorem price_of_larger_jar 
  (d1 h1 : ℝ) 
  (price1 : ℝ) 
  (d2 h2 : ℝ) 
  (V₁ : ℝ) 
  (V₂ : ℝ) 
  (price_per_volume : ℝ) 
  (r1 r2 : ℝ) :
  (V₁ = π * r1^2 * h1) ->
  (V₂ = π * r2^2 * h2) ->
  (price_per_volume = price1 / V₁) ->
  (price2 = price_per_volume * V₂) ->
  (d1 = 3) ->
  (h1 = 4) ->
  (price1 = 0.60) ->
  (d2 = 6) ->
  (h2 = 6) ->
  (r1 = d1 / 2) ->
  (r2 = d2 / 2) ->
  (price2 = 3.60) :=
begin
  sorry
end

end price_of_larger_jar_l348_348462


namespace desired_alcohol_percentage_l348_348083

-- Conditions given in the problem
def initial_volume : ℝ := 6
def initial_percentage_alcohol : ℝ := 0.30
def added_pure_alcohol : ℝ := 2.4

-- Definitions derived from the conditions
def initial_alcohol_amount : ℝ := initial_percentage_alcohol * initial_volume
def final_alcohol_amount : ℝ := initial_alcohol_amount + added_pure_alcohol
def final_volume : ℝ := initial_volume + added_pure_alcohol
def desired_percentage : ℝ := (final_alcohol_amount / final_volume) * 100

-- Proof that desired percentage is 50%
theorem desired_alcohol_percentage : desired_percentage = 50 := sorry

end desired_alcohol_percentage_l348_348083


namespace ribbon_lengths_after_cut_l348_348827

theorem ribbon_lengths_after_cut {
  l1 l2 l3 l4 l5 l1' l2' l3' l4' l5' : ℕ
  (initial_lengths : multiset ℕ)
  (new_lengths : multiset ℕ)
  (hl : initial_lengths = {15, 20, 24, 26, 30})
  (hl' : new_lengths = {l1', l2', l3', l4', l5'})
  (h_average_decrease : (∑ x in initial_lengths, x) / 5 - 5 = (∑ x in new_lengths, x) / 5)
  (h_median : ∀ x ∈ new_lengths, x = 24 ∨ 24 ∈ new_lengths)
  (h_range : multiset.range new_lengths = 15)
  (h_lengths : l1' ≤ l2' ≤ l3' = 24 ∧ l3' ≤ l4' ≤ l5') :
  new_lengths = {9, 9, 24, 24, 24} := sorry

end ribbon_lengths_after_cut_l348_348827


namespace not_collinear_implies_m_ne_half_right_angle_at_a_implies_m_eq_7_over_4_l348_348237

-- Definitions of the vectors
def vector_oa : ℝ × ℝ := (3, -4)
def vector_ob : ℝ × ℝ := (6, -3)
def vector_oc (m : ℝ) : ℝ × ℝ := (5 - m, -(3 + m))

-- Problem 1: Prove that if points A, B, and C can form a triangle, then m ≠ 1/2
theorem not_collinear_implies_m_ne_half (m : ℝ) :
  let ab := (3, 1),
      ac := (2 - m, 1 - m) in
  ¬ (3 * (1 - m) = 2 - m) → m ≠ 1 / 2 :=
sorry

-- Problem 2: Prove that if ΔABC is a right-angled triangle with ∠A as a right angle, then m = 7/4
theorem right_angle_at_a_implies_m_eq_7_over_4 (m : ℝ) :
  let ab := (3, 1),
      ac := (2 - m, 1 - m) in
  (3 * (2 - m) + 1 * (-m) = 0) → m = 7 / 4 :=
sorry

end not_collinear_implies_m_ne_half_right_angle_at_a_implies_m_eq_7_over_4_l348_348237


namespace depth_of_lost_ship_l348_348894

theorem depth_of_lost_ship (rate_of_descent : ℕ) (time_taken : ℕ) (h1 : rate_of_descent = 60) (h2 : time_taken = 60) :
  rate_of_descent * time_taken = 3600 :=
by {
  /-
  Proof steps would go here.
  -/
  sorry
}

end depth_of_lost_ship_l348_348894


namespace geom_arith_seq_l348_348666

theorem geom_arith_seq (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_arith : 2 * a 3 - (a 5 / 2) = (a 5 / 2) - 3 * a 1) (hq : q > 0) :
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 :=
by
  sorry

end geom_arith_seq_l348_348666


namespace probability_l348_348372

def is_multiple (x y : ℕ) : Prop := x % y = 0

def num_ways : ℕ := 12 * 11 * 10

def valid_ways : ℕ :=
  List.length [(a, b, c) | a ← List.range (12 + 1), b ← List.range (12 + 1), 
     c ← List.range (12 + 1), a ≠ 0, b ≠ 0, c ≠ 0, 
     a ≠ b, b ≠ c, a ≠ c, is_multiple a b, is_multiple a c]

theorem probability (A B C : ℕ) (hA: A ∈ List.range (12+1)) (hB: B ∈ List.range (12+1)) (hC: C ∈ List.range (12+1))
  (h_distinct: A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  valid_ways = 35 ∧ num_ways = 1320 → (35 / 1320 : ℚ) = 35 / 1320 :=
by 
  sorry

end probability_l348_348372


namespace area_of_rhombus_l348_348343

variable (a b θ : ℝ)
variable (h_a : 0 < a) (h_b : 0 < b)

theorem area_of_rhombus (h : true) : (2 * a) * (2 * b) / 2 = 2 * a * b := by
  sorry

end area_of_rhombus_l348_348343


namespace probability_each_delegate_next_to_another_country_l348_348762

theorem probability_each_delegate_next_to_another_country
  (total_delegates : ℕ)
  (delegates_per_country : ℕ)
  (countries : ℕ)
  (seats : ℕ)
  (h1 : total_delegates = 16)
  (h2 : delegates_per_country = 4)
  (h3 : countries = 4)
  (h4 : seats = 16)
  : ∃ m n : ℕ, m.gcd n = 1 ∧ (m + n) = ? := 
sorry

end probability_each_delegate_next_to_another_country_l348_348762


namespace solution_set_unique_line_l348_348750

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l348_348750


namespace michael_passes_donovan_after_laps_l348_348074

/-- The length of the track in meters -/
def track_length : ℕ := 400

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℕ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℕ := 36

/-- The number of laps that Michael will have to complete in order to pass Donovan -/
theorem michael_passes_donovan_after_laps : 
  ∃ (laps : ℕ), laps = 5 ∧ (∃ t : ℕ, 400 * t / 36 = 5 ∧ 400 * t / 45 < 5) :=
sorry

end michael_passes_donovan_after_laps_l348_348074


namespace girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l348_348873

-- Definition of the primary condition
def girls := 3
def boys := 5

-- Statement for each part of the problem
theorem girls_together (A : ℕ → ℕ → ℕ) : 
  A (girls + boys - 1) girls * A girls girls = 4320 := 
sorry

theorem girls_separated (A : ℕ → ℕ → ℕ) : 
  A boys boys * A (girls + boys - 1) girls = 14400 := 
sorry

theorem girls_not_both_ends (A : ℕ → ℕ → ℕ) : 
  A boys 2 * A (girls + boys - 2) (girls + boys - 2) = 14400 := 
sorry

theorem girls_not_both_ends_simul (P : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ) : 
  P (girls + boys) (girls + boys) - A girls 2 * A (girls + boys - 2) (girls + boys - 2) = 36000 := 
sorry

end girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l348_348873


namespace lattice_points_count_l348_348294

def is_lattice_point (x y : ℤ) : Prop :=
  (|x| - 1)^2 + (|y| - 1)^2 < 2

theorem lattice_points_count : 
  {p : ℤ × ℤ | is_lattice_point p.fst p.snd}.to_finset.card = 16 :=
by
  sorry

end lattice_points_count_l348_348294


namespace tangent_line_at_A_l348_348942

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def f (x : ℝ) : ℝ := x * ln x

def derivative_f (x : ℝ) : ℝ := 1 + ln x

def is_tangent_line (p : ℝ × ℝ) (l : ℝ → ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, l x = (derivative_f p.1) * (x - p.1) + p.2

theorem tangent_line_at_A :
  is_tangent_line (1, 0) (λ x, x - 1) f :=
sorry

end tangent_line_at_A_l348_348942


namespace maximum_whole_number_of_cars_l348_348353

noncomputable def maxCarsPassing (speed : ℕ) (carLength : ℕ) (duration : ℕ) : ℕ :=
  let n := speed / 10
  let distanceBetweenCars := 5 * (n + 1)
  let numCars := (speed * 500 * duration) / distanceBetweenCars
  numCars

theorem maximum_whole_number_of_cars :
    let M := maxCarsPassing 10 5 30 in
    M = 6000 ∧ (M / 10) = 600 :=
by 
  sorry

end maximum_whole_number_of_cars_l348_348353


namespace vector_magnitude_l348_348238

theorem vector_magnitude (m : ℝ) 
  (h : (6 : ℝ) + (-3) * m = 0)
  (a : ℝ × ℝ := (1, -3))
  (b : ℝ × ℝ := (6, m)) :
  |2 • a - b| = 4 * real.sqrt 5 :=
by
  /-
  We have vectors a and b defined, and the condition of orthogonality 
  provides a relationship for the scalar m. The goal now is to prove 
  the magnitude of the resulting vector expression is as stated.
  -/
  sorry

end vector_magnitude_l348_348238


namespace first_three_workers_time_l348_348184

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l348_348184


namespace angle_DNP_l348_348544

/-- Circle Ω is the incircle of triangle DEF and is also the circumcircle of triangle MNP.
The point M is on segment EF, point N is on segment DE, and point P is on segment DF.
Given that ∠D = 50°, ∠E = 70°, and ∠F = 60°, we prove that ∠DNP = 120°. -/
theorem angle_DNP (Ω : Circle) (D E F M N P : Point) :
  Incircle Ω (Triangle DEF) ∧ Circumcircle Ω (Triangle MNP) ∧
  OnSegment E F M ∧ OnSegment D E N ∧ OnSegment D F P ∧
  Angle D = 50 ∧ Angle E = 70 ∧ Angle F = 60 →
  Angle D N P = 120 :=
by
  sorry

end angle_DNP_l348_348544


namespace blue_tile_fraction_l348_348118

theorem blue_tile_fraction :
  let num_tiles := 8 * 8
  let corner_blue_tiles := 4 * 4 - 2 * 2
  let total_blue_tiles := 4 * corner_blue_tiles
  total_blue_tiles / num_tiles = (3 : ℚ) / 4 := 
by 
  let num_tiles := 8 * 8
  let corner_blue_tiles := 4 * 4 - 2 * 2
  let total_blue_tiles := 4 * corner_blue_tiles
  have frac_eq : total_blue_tiles / num_tiles = 48 / 64 := by sorry
  rw frac_eq,
  norm_num
  sorry

end blue_tile_fraction_l348_348118


namespace Ken_and_Kendra_fish_count_l348_348694

def Ken_and_Kendra_bring_home (kendra_fish_caught : ℕ) (ken_ratio : ℕ) (ken_releases : ℕ) : ℕ :=
  let ken_fish_caught := ken_ratio * kendra_fish_caught
  let ken_fish_brought_home := ken_fish_caught - ken_releases
  ken_fish_brought_home + kendra_fish_caught

theorem Ken_and_Kendra_fish_count :
  let kendra_fish_caught := 30 in
  let ken_ratio := 2 in
  let ken_releases := 3 in
  Ken_and_Kendra_bring_home kendra_fish_caught ken_ratio ken_releases = 87 :=
by
  sorry

end Ken_and_Kendra_fish_count_l348_348694


namespace combined_co2_to_o2_ratio_l348_348146

noncomputable def combined_co2_volume : ℕ := 6 + 2
noncomputable def oxygen_consumed : ℕ := 6

theorem combined_co2_to_o2_ratio : combined_co2_volume / oxygen_consumed = 4 / 3 := by
  have h1 : combined_co2_volume = 8 := rfl
  have h2 : oxygen_consumed = 6 := rfl
  rw [h1, h2]
  norm_num
  exact rfl

end combined_co2_to_o2_ratio_l348_348146


namespace solve_problem_l348_348334

theorem solve_problem (p q r : ℝ) (h1 : p + q + r = 8) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = 3) :
  \frac{p}{q * r + 1} + \frac{q}{p * r + 1} + \frac{r}{p * q + 1} = \frac{59}{22} := 
sorry

end solve_problem_l348_348334


namespace sum_of_vectors_zero_in_even_gon_l348_348199

variables (n : ℕ) (h : 2 * n ≥ 3) 

theorem sum_of_vectors_zero_in_even_gon (h : 2 * n ≥ 3) :
    ∃ (f : Fin (2 * n) → Fin (2 * n) → ℝ × ℝ), (∀ (i j : Fin (2 * n)), i ≠ j → 
    f i j = - f j i) ∧ ∑ (i : Fin (2 * n)) (j : Fin (2 * n)), f i j = (0, 0) :=
sorry

end sum_of_vectors_zero_in_even_gon_l348_348199


namespace find_quotient_l348_348728

theorem find_quotient (dividend : ℝ) (remainder : ℝ) (divisor : ℝ) (quotient : ℝ) 
                      (h1 : dividend = 17698)
                      (h2 : remainder = 14)
                      (h3 : divisor = 198.69662921348313) 
                      (h4 : quotient = (dividend - remainder) / divisor) :
                      quotient ≈ 89 := 
by
  -- Proof would go here
  sorry

end find_quotient_l348_348728


namespace volume_relationship_l348_348206

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
(1/6) * abs (det (matrix.of [A - D, B - D, C - D]))

theorem volume_relationship (A B C D : ℝ × ℝ × ℝ) (D1 : ℝ × ℝ × ℝ) (A1 B1 C1 : ℝ × ℝ × ℝ)
  (hD1 : D1 = (1/3) • A + (1/3) • B + (1/3) • C)
  (hA1 : ∃ k, A1 = A + k • (D1 - D))
  (hB1 : ∃ k, B1 = B + k • (D1 - D))
  (hC1 : ∃ k, C1 = C + k • (D1 - D)) :
  volume_tetrahedron A B C D = (1/3) * volume_tetrahedron A1 B1 C1 D1 :=
sorry

end volume_relationship_l348_348206


namespace log_identity_l348_348262

theorem log_identity (x : ℝ) (h : log 16 (x - 3) = 1 / 2) : log 64 x = log 2 7 / 6 :=
by
  sorry

end log_identity_l348_348262


namespace domain_of_f_l348_348007

noncomputable def f (x : ℝ) := Real.log (1 - x)

theorem domain_of_f : ∀ x, f x = Real.log (1 - x) → (1 - x > 0) →  x < 1 :=
by
  intro x h₁ h₂
  exact lt_of_sub_pos h₂

end domain_of_f_l348_348007


namespace equal_sum_of_distances_l348_348448

variable {n : ℕ}

theorem equal_sum_of_distances
  (n_ge_1 : n ≥ 1)
  (P : ℝ × ℝ)
  (circumcircle : ℝ × ℝ → bool)
  (A : Fin (2*n + 1) → ℝ × ℝ)
  (on_circumcircle_A : ∀ (i : Fin (2*n + 1)), circumcircle (A i))
  (on_circumcircle_P : circumcircle P) :
  (Finset.range (2*n+1)).filter (λ i, i % 2 = 0).sum (λ i, dist P (A ⟨i, Fin.is_lt _⟩))
  =
  (Finset.range (2*n)).filter (λ i, i % 2 = 1).sum (λ i, dist P (A ⟨i, Fin.is_lt _⟩)) := sorry

end equal_sum_of_distances_l348_348448


namespace isosceles_right_triangle_l348_348299

theorem isosceles_right_triangle (XYZ : Type) (X Y Z : XYZ)
  (h_right_triangle : right_triangle X Y Z)
  (h_equal_angles : ∠X = ∠Y)
  (h_hypotenuse : hypotenuse X Y Z = 10 * sqrt 2):
  (circumradius X Y Z = 5 * sqrt 2) ∧ (area X Y Z = 50) := 
by
  sorry

end isosceles_right_triangle_l348_348299


namespace ribbon_proof_l348_348818

theorem ribbon_proof :
  ∃ (ribbons : List ℝ), 
    ribbons = [9, 9, 24, 24, 24] ∧
    (∃ (initial_ribbons : List ℝ) (cuts : List ℝ),
      initial_ribbons = [15, 20, 24, 26, 30] ∧
      ((List.sum initial_ribbons / initial_ribbons.length) - (List.sum ribbons / ribbons.length) = 5) ∧
      (List.median ribbons = List.median initial_ribbons) ∧
      (List.range ribbons = List.range initial_ribbons)) :=
sorry

end ribbon_proof_l348_348818


namespace divisors_form_l348_348435

theorem divisors_form (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < n) :
  ∃ k : ℕ, (p^n - 1 = 2^k - 1 ∨ p^n - 1 ∣ 48) :=
sorry

end divisors_form_l348_348435


namespace truck_stops_l348_348067

variable (a : ℕ → ℕ)
variable (sum_1 : ℕ)
variable (sum_2 : ℕ)

-- Definition for the first sequence with a common difference of -10
def first_sequence : ℕ → ℕ
| 0       => 40
| (n + 1) => first_sequence n - 10

-- Definition for the second sequence with a common difference of -5
def second_sequence : ℕ → ℕ 
| 0       => 10
| (n + 1) => second_sequence n - 5

-- Summing the first sequence elements before the condition change:
def sum_first_sequence : ℕ → ℕ 
| 0       => 40
| (n + 1) => sum_first_sequence n + first_sequence (n + 1)

-- Summing the second sequence elements after the condition change:
def sum_second_sequence : ℕ → ℕ 
| 0       => second_sequence 0
| (n + 1) => sum_second_sequence n + second_sequence (n + 1)

-- Final sum of distances
def total_distance : ℕ :=
  sum_first_sequence 3 + sum_second_sequence 1

theorem truck_stops (sum_1 sum_2 : ℕ) (h1 : sum_1 = sum_first_sequence 3)
 (h2 : sum_2 = sum_second_sequence 1) : 
  total_distance = 115 := by
  sorry


end truck_stops_l348_348067


namespace diamonds_to_bullets_l348_348952

variable {α : Type*}

variables (Δ : α) (diamondsuit : α) (bullet : α)
variables [Add α] [Sub α] [Mul α] [Div α] [Neg α] [OfNat α] [One α]
variables [Neg α] [One α]

def balance (a b c : α) : Prop :=
  (5 * a + 2 * b = 12 * c) ∧ (a = b + 3 * c)

theorem diamonds_to_bullets 
  (a b c : α) 
  (h : balance a b c) : 
  4 * b = - (12 / 7) * c :=
by
  sorry

end diamonds_to_bullets_l348_348952


namespace valid_triples_l348_348564

theorem valid_triples :
  ∀ (a b c : ℕ), 1 ≤ a → 1 ≤ b → 1 ≤ c →
  (∃ k : ℕ, 32 * a + 3 * b + 48 * c = 4 * k * a * b * c) ↔ 
  (a = 1 ∧ b = 20 ∧ c = 1) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 3 ∧ b = 4 ∧ c = 1) := 
by
  sorry

end valid_triples_l348_348564


namespace ratio_of_areas_l348_348492

theorem ratio_of_areas 
  (s1 s2 : ℝ)
  (A_large A_small A_trapezoid : ℝ)
  (h1 : s1 = 10)
  (h2 : s2 = 5)
  (h3 : A_large = (sqrt 3 / 4) * s1^2)
  (h4 : A_small = (sqrt 3 / 4) * s2^2)
  (h5 : A_trapezoid = A_large - A_small) :
  (A_small / A_trapezoid = 1 / 3) :=
sorry

end ratio_of_areas_l348_348492


namespace result_after_divide_and_add_l348_348805

theorem result_after_divide_and_add (n : ℕ) (h : n = 72) : (n / 6) + 5 = 17 := by
  have h1 : n / 6 = 12, by
    rw h
    norm_num
  rw h1
  norm_num
  done

end result_after_divide_and_add_l348_348805


namespace sin_shift_left_l348_348032

theorem sin_shift_left :
  (∀ x, sin (2 * (x - π/6)) = sin (2 * x + π/3)) :=
by
  intros
  sorry

end sin_shift_left_l348_348032


namespace probability_of_exactly_two_dice_showing_3_is_257_l348_348933

noncomputable def probability_two_dice_show_three : ℝ :=
  let num_ways := (Nat.choose 15 2 : ℝ)
  let prob_two_show_three := (1 / 6) ^ 2
  let prob_thirteen_not_show_three := (5 / 6) ^ 13
  (num_ways * prob_two_show_three * prob_thirteen_not_show_three)

theorem probability_of_exactly_two_dice_showing_3_is_257 :
  Real.round (probability_two_dice_show_three * 1000) / 1000 = 0.257 :=
sorry

end probability_of_exactly_two_dice_showing_3_is_257_l348_348933


namespace ellipse_eq_standard_l348_348022

noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem ellipse_eq_standard : 
  ∃ (a b : ℝ), 
    a + b = 3 ∧ 
    (a^2 = b^2 + sqrt3^2) ∧ 
    (eq (x y : ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1) →
    (a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2) / 4 + y^2 = 1)) :=
sorry

end ellipse_eq_standard_l348_348022


namespace minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l348_348878

-- Definition for Problem Part (a)
def box_dimensions := (3, 5, 7)
def initial_cockchafers := 3 * 5 * 7 -- or 105

-- Defining the theorem for part (a)
theorem minimum_empty_cells_face_move (d : (ℕ × ℕ × ℕ)) (n : ℕ) :
  d = box_dimensions →
  n = initial_cockchafers →
  ∃ k ≥ 1, k = 1 :=
by
  intros hdim hn
  sorry

-- Definition for Problem Part (b)
def row_odd_cells := 2 * 5 * 7  
def row_even_cells := 1 * 5 * 7  

-- Defining the theorem for part (b)
theorem minimum_empty_cells_diagonal_move (r_odd r_even : ℕ) :
  r_odd = row_odd_cells →
  r_even = row_even_cells →
  ∃ m ≥ 35, m = 35 :=
by
  intros ho he
  sorry

end minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l348_348878


namespace power_sum_constant_l348_348329

universe u
open Real EuclideanGeometry

variables {A B C P : Point} 
variables {a b c R R_Q : ℝ}

noncomputable def H := orthocenter A B C
noncomputable def Q := centroid A B C

theorem power_sum_constant :
  ∀ (P : Point), P ∈ circumcircle A B C →
  PA^2 + PB^2 + PC^2 + QA^2 + QB^2 + QC^2 - PH^2 - QH^2 = a^2 + b^2 + c^2 - 4R^2 :=
by sorry

end power_sum_constant_l348_348329


namespace complete_quadrilateral_theorem_l348_348595

variables {A B C D P Q R K L : Type}
variables [projective_geometry A]
variables (intersect_AB_CD : P = Locus_intersect AB CD)
variables (intersect_AD_BC : Q = Locus_intersect AD BC)
variables (intersect_AC_BD : R = Locus_intersect AC BD)
variables (intersect_QR_AB : K = Locus_intersect QR AB)
variables (intersect_QR_CD : L = Locus_intersect QR CD)

theorem complete_quadrilateral_theorem :
  cross_ratio Q R K L = - 1 :=
sorry

end complete_quadrilateral_theorem_l348_348595


namespace new_shoes_cost_percent_greater_l348_348880

-- Conditions
def cost_repair : ℝ := 13.50
def duration_repair : ℝ := 1
def cost_new : ℝ := 32.00
def duration_new : ℝ := 2

-- Average cost per year calculations
def avg_cost_repair : ℝ := cost_repair / duration_repair
def avg_cost_new : ℝ := cost_new / duration_new

-- Difference in average costs
def difference_cost : ℝ := avg_cost_new - avg_cost_repair

-- Percentage increase calculation
def percentage_increase : ℝ := (difference_cost / avg_cost_repair) * 100

-- Theorem statement
theorem new_shoes_cost_percent_greater :
  percentage_increase = 18.52 := 
sorry

end new_shoes_cost_percent_greater_l348_348880


namespace concyclicity_of_EFGH_l348_348661

-- Definitions of a cyclic quadrilateral
structure CyclicQuadrilateral (A B C D : Type) :=
  (cyclic : ∀ P Q R S : Type, ∠ PAC + ∠ PQB = 180) -- This is very simplistic; adapt as needed to be rigorously correct.

-- Supporting definitions for points and angle bisectors
structure Point (P : Type) :=
  (coord : Type) -- coordinates can be specified in more detail if necessary

noncomputable def angleBisector (P Q R : Point) : Point := sorry -- This should compute the bisector of an angle

-- Problem statement in Lean
theorem concyclicity_of_EFGH 
  (A B C D E F G H : Point)
  (ABCD_cyclic : CyclicQuadrilateral A B C D)
  (E_on_AB : angleBisector A D B = E)
  (F_on_AB : angleBisector A C B = F)
  (G_on_CD : angleBisector C B D = G)
  (H_on_CD : angleBisector C A D = H)
  : ∃ k : Type, Circle k ∧ (E ∈ k ∧ F ∈ k ∧ G ∈ k ∧ H ∈ k) :=
sorry

end concyclicity_of_EFGH_l348_348661


namespace probability_green_ball_l348_348121

theorem probability_green_ball 
  (total_balls : ℕ) 
  (green_balls : ℕ) 
  (white_balls : ℕ) 
  (h_total : total_balls = 9) 
  (h_green : green_balls = 7)
  (h_white : white_balls = 2)
  (h_total_eq : total_balls = green_balls + white_balls) : 
  (green_balls / total_balls : ℚ) = 7 / 9 := 
by
  sorry

end probability_green_ball_l348_348121


namespace solution_set_unique_line_l348_348749

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l348_348749


namespace exists_distinct_block_similar_polynomials_no_distinct_block_similar_polynomials_of_degree_n_l348_348324

noncomputable def exists_block_similar_polynomials (n : ℕ) (hn : 2 ≤ n) : Prop :=
  ∃ (P : ℝ → ℝ) (Q : ℝ → ℝ), 
  (degree P = n + 1 ∧ degree Q = n + 1 ∧ P ≠ Q ∧ 
    ∀ i ∈ finset.range n,
    (finset.range 2015).perm (λ j, P (2015 * i + j)) (λ j, Q (2015 * i + j)))

noncomputable def no_block_similar_polynomials_of_degree_n (n : ℕ) (hn : 2 ≤ n) : Prop :=
  ∀ (P : ℝ → ℝ) (Q : ℝ → ℝ), 
    (degree P = n → degree Q = n → P ≠ Q →
      ∃ i ∈ finset.range n,
      (λ j, P (2015 * i + j)) ≠ (λ j, Q (2015 * i + j)))

-- The statements to prove:
theorem exists_distinct_block_similar_polynomials (n : ℕ) (hn : 2 ≤ n) :
  exists_block_similar_polynomials n hn := sorry

theorem no_distinct_block_similar_polynomials_of_degree_n (n : ℕ) (hn : 2 ≤ n) :
  no_block_similar_polynomials_of_degree_n n hn := sorry

end exists_distinct_block_similar_polynomials_no_distinct_block_similar_polynomials_of_degree_n_l348_348324


namespace area_isosceles_triangle_bde_l348_348308

-- Define the lengths of sides and the special points
theorem area_isosceles_triangle_bde (A B C D E : Type) [metric_space A] 
  [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (AB AC BC BE BD DE : ℝ)
  (h1: AB = 12)
  (h2: AC = 24)
  (h3: BC = 12 * sqrt 5)
  (h4: BE = 8 * sqrt 5)
  (h5: BD = BE)
  (h6: DE = BE)
  (isosceles: BE = BD = DE):
  1/2 * BE * BE = 80 :=
by
  sorry

end area_isosceles_triangle_bde_l348_348308


namespace jake_weight_l348_348643

variable (J S : ℕ)

theorem jake_weight (h1 : J - 15 = 2 * S) (h2 : J + S = 132) : J = 93 := by
  sorry

end jake_weight_l348_348643


namespace find_b_and_asymptotes_l348_348980

-- Definitions based on the given conditions
def hyperbola_eq (x y b : ℝ) : Prop :=
  x^2 - (y^2 / b^2) = 1

def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (2, 0)

-- The main theorem to prove
theorem find_b_and_asymptotes (b : ℝ) (h_b_pos : b > 0) (h_hyperbola : ∀ x y, hyperbola_eq x y b) (h_focus : is_focus (2, 0)) :
  b = real.sqrt 3 ∧ (∀ x, y = real.sqrt 3 * x ∨ y = -real.sqrt 3 * x) :=
by
  sorry  -- The proof will be filled in here.

end find_b_and_asymptotes_l348_348980


namespace angle_XPY_120_degrees_l348_348708

theorem angle_XPY_120_degrees
  (XYZ : Type)
  [triangle : Triangle XYZ]
  (angle_X : triangle.angle X = 60)
  (angle_Y : triangle.angle Y = 45)
  (circle_P : circle XYZ.center P)
  (passes_through_A : circle_P.passes_through A) 
  (passes_through_B : circle_P.passes_through B)
  (passes_through_C : circle_P.passes_through C)
  (passes_through_D : circle_P.passes_through D)
  (passes_through_E : circle_P.passes_through E)
  (passes_through_F : circle_P.passes_through F)
  (AB_length : segment_length AB = CD_length)
  (CD_length : segment_length CD = EF_length)
  (EF_length : segment_length EF = AB_length) :
  triangle.angle XPY = 120 := 
sorry

end angle_XPY_120_degrees_l348_348708


namespace partition_7_students_into_groups_of_2_or_3_l348_348247

def binomial (n k : ℕ) : ℕ :=
nat.choose n k

def partitions_of_7_students : ℕ :=
  let ways_to_choose_3 := binomial 7 3
  let ways_to_partition_remaining := binomial 4 2 / 2
  ways_to_choose_3 * ways_to_partition_remaining

theorem partition_7_students_into_groups_of_2_or_3 :
    partitions_of_7_students = 105 := by
  sorry

end partition_7_students_into_groups_of_2_or_3_l348_348247


namespace car_a_traveled_distance_l348_348912

def CarAStartSpeed := 40 -- in km/h
def CarBStartSpeed := 50 -- in km/h
def CarATurnSpeed := 50 -- in km/h
def CarBTurnSpeed := 40 -- in km/h
def initialDistance := 900 -- in km
def meetingNumber := 2016

theorem car_a_traveled_distance : 
  ∃ distance_traveled : ℝ, -- Define the variable for the traveled distance
  distance_traveled = 1813900 ∧
  CarAStartSpeed = 40 ∧ CarBStartSpeed = 50 ∧
  CarATurnSpeed = 50 ∧ CarBTurnSpeed = 40 ∧
  initialDistance = 900 ∧
  meetingNumber = 2016 :=
begin
  -- Statement without providing the proof
  sorry
end

end car_a_traveled_distance_l348_348912


namespace hyperbola_asymptote_perpendicular_to_AM_l348_348992

theorem hyperbola_asymptote_perpendicular_to_AM :
  ∀ (p m a : ℝ),
    (m^2 = 2 * p) ∧ (p > 0) ∧ (real.dist ((1 : ℝ), m) ((p / 2), (0 : ℝ)) = 5) ∧ 
    (∃ (x y : ℝ), x^2 - y^2 / a = 1) ∧ 
    ((-√a * 2 = -1) ∨ (2 * -√a = -1)) → 
    (a = 1 / 4) := 
by
  sorry

end hyperbola_asymptote_perpendicular_to_AM_l348_348992


namespace snail_reaches_B_l348_348473

open Real

/-- Will the snail eventually reach point B, given the initial conditions? -/
theorem snail_reaches_B :
  ∃ t : ℝ, t > 0 ∧ S(t) = 0
  (S : ℝ → ℝ) -- S(t) denotes the snail's distance from B over time t
  (initial_length : ℝ := 10) -- initial length of the rope in meters
  (snail_speed : ℝ := 0.01) -- snail's speed in meters per second (1 cm/s)
  (gnome_speed : ℝ := 10) -- gnome's speed in meters per second
  (S_equation : ∀ t : ℝ, S(t) = initial_length * exp(-10 * t) + (snail_speed * t)) -- a simplified model equation for S(t)
  : sorry

end snail_reaches_B_l348_348473


namespace percent_of_whole_l348_348872

theorem percent_of_whole (Part Whole : ℝ) (Percent : ℝ) (hPart : Part = 160) (hWhole : Whole = 50) :
  Percent = (Part / Whole) * 100 → Percent = 320 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_whole_l348_348872


namespace remainder_hx10_div_hx_l348_348341

noncomputable def h (x : ℝ) : ℝ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_hx10_div_hx : 
  ∀ x : ℝ, polynomial.remainder (h (x^10)) (h x) = 7 :=
by
  sorry

end remainder_hx10_div_hx_l348_348341


namespace radius_ratio_l348_348474

noncomputable def ratio_of_radii (V1 V2 : ℝ) (R : ℝ) : ℝ := 
  (V2 / V1)^(1/3) * R 

theorem radius_ratio (V1 V2 : ℝ) (π : ℝ) (R r : ℝ) :
  V1 = 450 * π → 
  V2 = 36 * π → 
  (4 / 3) * π * R^3 = V1 →
  (4 / 3) * π * r^3 = V2 →
  r / R = 1 / (12.5)^(1/3) :=
by {
  sorry
}

end radius_ratio_l348_348474


namespace polynomial_zero_or_nested_zero_l348_348699

variable {R : Type} [CommRing R]

theorem polynomial_zero_or_nested_zero (P : R[X]) :
  (∀ k : ℕ, (Nat.iterate P.eval 0 k) = 0) → (P.eval 0 = 0 ∨ P.eval (P.eval 0) = 0) :=
begin
  intro h,
  sorry
end

end polynomial_zero_or_nested_zero_l348_348699


namespace compare_surface_areas_l348_348336

theorem compare_surface_areas (p : ℝ) (PQ MN : ℝ → ℝ → Prop) 
  (hyp_chord: ∀ P Q : ℝ × ℝ, PQ P Q → P.1 * Q.1 ≥ (p / 2) ^ 2)
  (hyp_proj: ∀ P Q M N : ℝ × ℝ, PQ P Q → MN (P.1, 0) (Q.1, 0))
  (S1 S2 : ℝ)
  (hyp_S1: S1 = π * (p + (sqrt p)^2)^2)
  (hyp_S2: S2 = π * (p + (sqrt p)^2 * sin (p / 2))^2) :
  S1 ≥ S2 :=
sorry

end compare_surface_areas_l348_348336


namespace range_of_m_l348_348988

/-- The range of the real number m such that the equation x^2/m + y^2/(2m - 1) = 1 represents an ellipse with foci on the x-axis is (1/2, 1). -/
theorem range_of_m (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, x^2 / m + y^2 / (2 * m - 1) = 1 → x^2 / a^2 + y^2 / b^2 = 1 ∧ b^2 < a^2))
  ↔ 1 / 2 < m ∧ m < 1 :=
sorry

end range_of_m_l348_348988


namespace sampling_survey_suitability_l348_348841

-- Define the conditions
def OptionA := "Understanding the effectiveness of a certain drug"
def OptionB := "Understanding the vision status of students in this class"
def OptionC := "Organizing employees of a unit to undergo physical examinations at a hospital"
def OptionD := "Inspecting components of artificial satellite"

-- Mathematical statement
theorem sampling_survey_suitability : OptionA = "Understanding the effectiveness of a certain drug" → 
  ∃ (suitable_for_sampling_survey : String), suitable_for_sampling_survey = OptionA :=
by
  sorry

end sampling_survey_suitability_l348_348841


namespace inradius_formula_l348_348589

variable (r : ℝ) (α β γ : ℝ)

-- tg α = 1 / 3
axiom tg_α : Mathlib.tan α = 1 / 3

-- sin β sin γ = 1 / √10
axiom sin_β_sin_γ : Mathlib.sin β * Mathlib.sin γ = 1 / Mathlib.sqrt 10

-- Definition of varrho (inradius) as derived in the problem
def varrho (r : ℝ) (α β γ : ℝ) :=
  (2 * r * Mathlib.sin α * Mathlib.sin β * Mathlib.sin γ) / (Mathlib.sin α + Mathlib.sin β + Mathlib.sin γ)

-- Proof Statement
theorem inradius_formula :
  let sin_α := 1 / (Mathlib.sqrt (1 + 1 / 9)) in
  varrho r α β γ = (r * sin_α * Mathlib.sqrt 10 / (sin_α + sin_α + Mathlib.sqrt 10)) :=
sorry

end inradius_formula_l348_348589


namespace ratio_of_smaller_to_trapezoid_l348_348500

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

def ratio_of_areas : ℝ :=
  let side_large := 10
  let side_small := 5
  let area_large := area_equilateral_triangle side_large
  let area_small := area_equilateral_triangle side_small
  let area_trapezoid := area_large - area_small
  area_small / area_trapezoid

theorem ratio_of_smaller_to_trapezoid :
  ratio_of_areas = 1 / 3 :=
sorry

end ratio_of_smaller_to_trapezoid_l348_348500


namespace problem_1_problem_2_l348_348307

-- Definitions for the problem conditions
structure Triangle :=
  (A B C : Point)
  (side_length : ℝ)
  (equilateral : ∀ P Q R : Point, P ≠ Q → Q ≠ R → P ≠ R → dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = side_length)
  (midpoint : Point)
  (E F : Point)
  (on_sides : ∃ u v : ℝ, 0 ≤ u ∧ u ≤ side_length ∧ 0 ≤ v ∧ v ≤ side_length ∧ E = u • (B - A) + A ∧ F = v • (C - A) + A)
  (D : Point)
  (D_is_midpoint : D = (B + C) / 2)

-- Problem (1)
theorem problem_1 (T : Triangle) (angle_DEF := 120) : 
  ∃ AE AF : ℝ, AE + AF = 3 := 
sorry

-- Problem (2)
theorem problem_2 (T : Triangle) (angle_DEF := 60) : 
  ∃ AE AF : ℝ, AE + AF ∈ set.Icc (3/2) 2 :=
sorry

end problem_1_problem_2_l348_348307


namespace exists_triangle_three_colors_l348_348810

/-
Given a large triangle ΔGRB dissected into 25 smaller triangles, with vertices on:
- G colored green,
- R colored red,
- B colored blue,
- Side GR vertices colored either green or red,
- Side RB vertices colored either red or blue,
- Side GB vertices colored either green or blue,
- Interior vertices colored arbitrarily.

Prove that there's at least one small triangle with vertices of three different colors.
-/

structure Triangle :=
  (vertices : Finset ( Finset (Finset color)))
  (conditions : verts_exist )

def g := color.green
def r := color.red
def b := color.blue

axiom verts_exist : ∃ (s ∈ verts), colored_correctly s

theorem exists_triangle_three_colors (T : Triangle) : ( ∃ t ∈ T.t, colors_mixed t ) :=
sorry

end exists_triangle_three_colors_l348_348810


namespace find_a_l348_348558

-- Representing the parametric curve C_1
def C1_param (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (a * Real.cos t, 1 + a * Real.sin t)

-- Polar equation of C_2
def C2_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Polar equation of C_3
def C3_polar (θ : ℝ) : Prop :=
  θ = Real.arctan 2

-- Proof statement: Prove that a = 1 under the given conditions
theorem find_a (a : ℝ) (t θ : ℝ) (h : a > 0) :
  (C1_param t a).1^2 + ((C1_param t a).2 - 1)^2 = a^2 ∧
  (C2_polar θ)^2 = 4 * C2_polar θ * Real.cos θ ∧
  C3_polar θ →
  a = 1 :=
sorry

end find_a_l348_348558


namespace min_students_l348_348775

noncomputable def num_boys_min (students : ℕ) (girls : ℕ) : Prop :=
  ∃ (boys : ℕ), boys > (3 * girls / 2) ∧ students = boys + girls

theorem min_students (girls : ℕ) (h_girls : girls = 5) : ∃ n, num_boys_min n girls ∧ n = 13 :=
by
  use 13
  unfold num_boys_min
  use 8
  sorry

end min_students_l348_348775


namespace perpendicular_vectors_l348_348050

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l348_348050


namespace solve_z_squared_eq_l348_348160

open Complex

theorem solve_z_squared_eq : 
  ∀ z : ℂ, z^2 = -100 - 64 * I → (z = 4 - 8 * I ∨ z = -4 + 8 * I) :=
by
  sorry

end solve_z_squared_eq_l348_348160


namespace financial_calculations_correct_l348_348523

noncomputable def revenue : ℝ := 2500000
noncomputable def expenses : ℝ := 1576250
noncomputable def loan_payment_per_month : ℝ := 25000
noncomputable def number_of_shares : ℕ := 1600
noncomputable def ceo_share_percentage : ℝ := 0.35

theorem financial_calculations_correct :
  let net_profit := (revenue - expenses) - (revenue - expenses) * 0.2 in
  let total_loan_payment := loan_payment_per_month * 12 in
  let dividends_per_share := (net_profit - total_loan_payment) / number_of_shares in
  let ceo_dividends := dividends_per_share * ceo_share_percentage * number_of_shares in
  net_profit = 739000 ∧
  total_loan_payment = 300000 ∧
  dividends_per_share = 274 ∧
  ceo_dividends = 153440 :=
begin
  sorry
end

end financial_calculations_correct_l348_348523


namespace equilateral_triangle_negation_l348_348598

theorem equilateral_triangle_negation (P : Prop) (hP : ∃ (T : Type) [∀ x : T, is_equilateral_triangle x]) :
  ¬P ↔ ∀ (T : Type) [∀ x : T, is_triangle x], ¬is_equilateral_triangle T := sorry

end equilateral_triangle_negation_l348_348598


namespace find_m_minus_n_l348_348581

noncomputable def m_abs := 4
noncomputable def n_abs := 6

theorem find_m_minus_n (m n : ℝ) (h1 : |m| = m_abs) (h2 : |n| = n_abs) (h3 : |m + n| = m + n) : m - n = -2 ∨ m - n = -10 :=
sorry

end find_m_minus_n_l348_348581


namespace problem_part1_increase_sales_volume_problem_part1_profit_per_unit_problem_part2_price_reduction_l348_348882

open Real

variables (x : ℝ)

-- Assuming initial conditions
def initial_sales := 30
def initial_profit := 50
def sales_increase_per_unit := 2

-- Statements for parts (1) and (2)
theorem problem_part1_increase_sales_volume : 
  true -> 
  initial_sales + sales_increase_per_unit * x = 30 + 2 * x := 
begin
  sorry
end

theorem problem_part1_profit_per_unit : 
  true -> 
  initial_profit - x = 50 - x := 
begin
  sorry
end

theorem problem_part2_price_reduction (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 :=
begin
  sorry
end

end problem_part1_increase_sales_volume_problem_part1_profit_per_unit_problem_part2_price_reduction_l348_348882


namespace A_three_two_l348_348919

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m+1, 0 => A m 2
| m+1, n+1 => A m (A (m + 1) n)

theorem A_three_two : A 3 2 = 5 := 
by 
  sorry

end A_three_two_l348_348919


namespace exists_divisor_in_interval_l348_348342

open Nat

theorem exists_divisor_in_interval {n k : ℕ} (hn : n % 2 = 1) (hd : ∃m, odd_divisors_le_k 2n k = 2 * m + 1) :
  ∃ d, d ∣ (2 * n) ∧ k < d ∧ d ≤ 2 * k := sorry

-- Auxiliary definitions and conditions

-- Definition for odd numbers
def odd (n : ℕ) : Prop := n % 2 = 1

-- Definition of the number of odd divisors of 2n that are less than or equal to k
def odd_divisors_le_k (n k : ℕ) : ℕ :=
  (finset.filter (λ d, d ∣ n ∧ d ≤ k) (finset.range (n + 1))).card

end exists_divisor_in_interval_l348_348342


namespace non_existence_of_non_planar_closed_chain_l348_348678

noncomputable def length (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

def perpendicular (p q r : ℝ × ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (r.1 - q.1) + (q.2 - p.2) * (r.2 - q.2) + (q.3 - p.3) * (r.3 - q.3) = 0

theorem non_existence_of_non_planar_closed_chain :
  ¬ ∃ (A B C D E : ℝ × ℝ × ℝ),
    length A B = 1 ∧
    length B C = 1 ∧
    length C D = 1 ∧
    length D E = 1 ∧
    length E A = 1 ∧
    perpendicular A B C ∧
    perpendicular B C D ∧
    perpendicular C D E ∧
    perpendicular D E A ∧
    (¬ ∃ (a b c : ℝ), A.3 = 0 ∧ B.3 = 0 ∧ C.3 = 0) := 
sorry

end non_existence_of_non_planar_closed_chain_l348_348678


namespace mr_johnson_fencing_l348_348350

variable (Length Width : ℕ)

def perimeter_of_rectangle (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem mr_johnson_fencing
  (hLength : Length = 25)
  (hWidth : Width = 15) :
  perimeter_of_rectangle Length Width = 80 := by
  sorry

end mr_johnson_fencing_l348_348350


namespace correct_sum_l348_348476

variable (A B : ℚ[X])
variable (student_result : ℚ[X])

-- Conditions from the problem statement
axiom h1 : A - B = 9 * X ^ 2 - 2 * X + 7
axiom h2 : B = X ^ 2 + 3 * X - 2

-- Goal is to prove the correct result for A + B
theorem correct_sum (A B : ℚ[X]) (student_result : ℚ[X]) (h1 : A - B = 9 * X ^ 2 - 2 * X + 7) (h2 : B = X ^ 2 + 3 * X - 2) :
  A + B = 11 * X ^ 2 + 4 * X + 3 :=
by sorry

end correct_sum_l348_348476


namespace area_of_isosceles_right_triangle_max_area_of_triangle_l348_348224

-- Part (Ⅰ)
theorem area_of_isosceles_right_triangle (t : ℝ) (h : -2 < t ∧ t < 2) :
  ∃ (area : ℝ), area = (9 / 2) ∧ 
  (∀ y : ℝ, y = 4 - t^2 → y = t + 2) :=
sorry

-- Part (Ⅱ)
theorem max_area_of_triangle (t : ℝ) (h : -2 < t ∧ t < 2) :
  ∃ (S : ℝ → ℝ), 
  S t = (1 / 2) * ((t + 2) * (4 - t^2)) ∧ 
  (∀ t, S(t) ≤ 128 / 27) :=
sorry

end area_of_isosceles_right_triangle_max_area_of_triangle_l348_348224


namespace heptagon_side_intersection_points_l348_348245

theorem heptagon_side_intersection_points : 
  (number_of_intersection_points sides_of_convex_heptagon) = 21 :=
sorry

end heptagon_side_intersection_points_l348_348245


namespace sum_of_h_and_k_l348_348490

theorem sum_of_h_and_k (foci1 foci2 : ℝ × ℝ) (pt : ℝ × ℝ) (a b h k : ℝ) 
  (h_positive : a > 0) (b_positive : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = if (x, y) = pt then 1 else sorry)
  (foci_eq : foci1 = (1, 2) ∧ foci2 = (4, 2))
  (pt_eq : pt = (-1, 5)) :
  h + k = 4.5 :=
sorry

end sum_of_h_and_k_l348_348490


namespace bob_final_amount_l348_348506

-- Define initial amount
def initial_amount : ℝ := 80

-- Define money spent on Monday
def spent_monday : ℝ := initial_amount / 2

-- Define money left after Monday
def left_after_monday : ℝ := initial_amount - spent_monday

-- Define money spent on Tuesday
def spent_tuesday : ℝ := left_after_monday / 5

-- Define money left after Tuesday
def left_after_tuesday : ℝ := left_after_monday - spent_tuesday

-- Define money spent on Wednesday
def spent_wednesday : ℝ := (3 / 8) * left_after_tuesday

-- Define money left after Wednesday
def left_after_wednesday : ℝ := left_after_tuesday - spent_wednesday

-- The theorem stating the final amount of money left
theorem bob_final_amount : left_after_wednesday = 20 := by
  have h0 : initial_amount = 80 := rfl
  have h1 : spent_monday = 40 := by 
    simp [initial_amount, spent_monday]
  have h2 : left_after_monday = 40 := by 
    simp [left_after_monday, initial_amount, spent_monday]
    norm_num
  have h3 : spent_tuesday = 8 := by
    simp [spent_tuesday, left_after_monday]
    norm_num
  have h4 : left_after_tuesday = 32 := by
    simp [left_after_tuesday, left_after_monday, spent_tuesday]
    norm_num
  have h5 : spent_wednesday = 12 := by
    simp [spent_wednesday, left_after_tuesday]
    norm_num
  have h6 : left_after_wednesday = 20 := by
    simp [left_after_wednesday, left_after_tuesday, spent_wednesday]
    norm_num
  exact h6

end bob_final_amount_l348_348506


namespace fill_time_both_pipes_l348_348354

variables (T : ℝ) (t : ℝ) (faster slower : ℝ → ℝ)

-- Conditions
def faster_pipe_fill_time := 45
def slower_pipe_fill_time := 180
def combined_fill_time := 36

-- Definitions
def faster := 1 / faster_pipe_fill_time
def slower := 1 / slower_pipe_fill_time
def combined := faster + slower

-- Theorem stating that both pipes together fill in 36 minutes
theorem fill_time_both_pipes :
  (combined_fill_time = 1 / combined) :=
sorry

end fill_time_both_pipes_l348_348354


namespace ken_and_kendra_brought_home_l348_348697

-- Define the main variables
variables (ken_caught kendra_caught ken_brought_home : ℕ)

-- Define the conditions as hypothesis
def conditions :=
  kendra_caught = 30 ∧
  ken_caught = 2 * kendra_caught ∧
  ken_brought_home = ken_caught - 3

-- Define the problem to prove
theorem ken_and_kendra_brought_home :
  (ken_caught + kendra_caught = 87) :=
begin
  -- Unpacking the conditions for readability
  unfold conditions at *,
  sorry -- Proof will go here
end

end ken_and_kendra_brought_home_l348_348697


namespace correct_calculation_l348_348431

theorem correct_calculation : 
  ∃ (option : (ℕ → ℕ) → Prop), 
    (option (λ x, (x^3)^2 = x^5) = false) ∧
    (option (λ x, x^2 + x^2 = x^4) = false) ∧
    (option (λ x, x^8 / x^2 = x^6) = true) ∧
    (option (λ x, (3 * x)^2 = 6 * x^2) = false) :=
by {
  existsi (λ f, f),
  split, {
    intros x,
    rw pow_mul,
    exact (by contradiction), -- indicates the term should be non-evaluated and false
  },
  split, {
    intros x,
    linarith, -- Verify the addition of exponents
  },
  split, {
    intros x,
    exact (by norm_num),
  },
  {
    intros x,
    norm_num,
    exact (by contradiction), -- indicates the term should be non-evaluated and false
  }
}

end correct_calculation_l348_348431


namespace pentagon_rotation_l348_348056

theorem pentagon_rotation (P : Type) [regular_pentagon P] : rotate P (252 : ℝ) = P':=
    begin
        sorry -- Proof omitted
    end

end pentagon_rotation_l348_348056


namespace find_a_l348_348994

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

def A_def : A = {1, 2, 5} :=
  by
    sorry

def B_def : B = {a + 4, a} :=
  by
    sorry

def intersection_property : A ∩ B = B :=
  by
    sorry

theorem find_a : a = 1 :=
  by
    have hA : A = {1, 2, 5} := A_def
    have hB : B = {a + 4, a} := B_def
    have hIntersect : A ∩ B = B := intersection_property
    sorry

end find_a_l348_348994


namespace local_odd_func_1_local_odd_func_2_l348_348172

-- Part (1)
theorem local_odd_func_1 (m : ℝ) :
  (∀ x ∈ set.Icc (-2:ℝ) (2:ℝ), 2^(-x) + m = - (2^x + m)) → m ∈ set.Icc (-17/8) (-1) :=
sorry

-- Part (2)
theorem local_odd_func_2 (m : ℝ) :
  (∀ x : ℝ, 4^(-x) + m * 2^(1 - x) + m^2 - 4 = - (4^x + m * 2^(1 + x) + m^2 - 4)) → m ∈ set.Icc (-1) (Real.sqrt 10) :=
sorry

end local_odd_func_1_local_odd_func_2_l348_348172
