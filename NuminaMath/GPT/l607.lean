import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.FiniteSets
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LinearEquiv
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Catalan
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Polynomial
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Polar
import Mathlib.Geometry.Triangle.Basic
import Mathlib.LinearAlgebra.Determinant
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independent
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace number_of_nonempty_subsets_l607_607818

theorem number_of_nonempty_subsets : 
  let B := {x ∈ (Finset.range 11) | true} in
  (B.powerset.filter (λ A, A.nonempty)).card = 1023 :=
by
  let B := {x ∈ (Finset.range 11) | true}
  have h₁ : B.card = 11 := by rw [Finset.card_range]
  have h₂ : B.powerset.card = 2^11 := by rw [Finset.card_powerset]

  have h₃ : (B.powerset.filter (λ A, A.nonempty)).card 
           = B.powerset.card - 1 :=
    Finset.card_filter_sub (λ A, A.nonempty) _ (sorry : finset B.subsets ⟹ finset B.subsets - finset singleton)

  rw [h₂] at h₃
  exact eq.rec rfl (by simp only [B.powerset.filter (λ A, A.nonempty), B.powerset.card_pred h₃])

end number_of_nonempty_subsets_l607_607818


namespace binomial_7_4_eq_35_l607_607716

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607716


namespace arithmetic_sequence_properties_l607_607411

noncomputable def a_n (n : ℕ) : ℤ := 9 - n

theorem arithmetic_sequence_properties :
  (∃ d : ℤ, d < 0 ∧
    ∃ a_2 a_6 : ℤ, a_2 = a_n 2 ∧ a_6 = a_n 6 ∧
    a_2 + a_6 = 10 ∧ a_2 * a_6 = 21) →
    (∀ n : ℕ, a_n n = 9 - n) ∧
    (∃ n : ℕ, max_T_n n = 2 ^ 36) :=
by
  sorry

end arithmetic_sequence_properties_l607_607411


namespace gardener_tree_arrangement_l607_607295

theorem gardener_tree_arrangement :
  let maple_trees := 4
  let oak_trees := 5
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)
  let valid_slots := 9  -- as per slots identified in the solution
  let valid_arrangements := 1 * Nat.choose valid_slots oak_trees
  let probability := valid_arrangements / total_arrangements
  probability = 1 / 75075 →
  (1 + 75075) = 75076 := by {
    sorry
  }

end gardener_tree_arrangement_l607_607295


namespace circle_radius_is_zero_l607_607757

-- Define the condition: the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 10 * y + 41 = 0

-- Define the statement to be proved: the radius of the circle described by the equation is 0
theorem circle_radius_is_zero : ∀ x y : ℝ, circle_eq x y → (∃ r : ℝ, r = 0) :=
begin
  sorry
end

end circle_radius_is_zero_l607_607757


namespace greatest_divisor_l607_607626

theorem greatest_divisor (n : ℕ) :
  (n ∣ (60 - 6)) ∧ (n ∣ (190 - 10)) ∧ ∀ m : ℕ, (m ∣ (60 - 6)) ∧ (m ∣ (190 - 10)) → m ≤ n :=
  n = 18 :=
sorry

end greatest_divisor_l607_607626


namespace find_coordinates_sum_l607_607941

def point (ℝ : Type*) := (ℝ × ℝ)

variables {a b : ℝ}
def A : point ℝ := (2, 8)
def B : point ℝ := (2, 2)
def C : point ℝ := (6, 2)
def D : point ℝ := (a, b)

def midpoint (p1 p2 : point ℝ) : point ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def is_square (m1 m2 m3 m4 : point ℝ) : Prop :=
  let d1 := (m1.1 - m2.1)^2 + (m1.2 - m2.2)^2 in
  let d2 := (m2.1 - m3.1)^2 + (m2.2 - m3.2)^2 in
  let d3 := (m3.1 - m4.1)^2 + (m3.2 - m4.2)^2 in
  let d4 := (m4.1 - m1.1)^2 + (m4.2 - m1.2)^2 in
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

def M1 : point ℝ := midpoint A B
def M2 : point ℝ := midpoint B C
def M3 : point ℝ := (7, 4)  -- precomputed result from solution steps
def M4 : point ℝ := midpoint D A  -- must form a square with other midpoints

theorem find_coordinates_sum :
  M4 = midpoint C D →
  is_square M1 M2 M3 M4 →
  a + b = 14 :=
begin
  -- Proof omitted
  sorry,
end

end find_coordinates_sum_l607_607941


namespace max_prob_diff_of_two_dice_l607_607259

theorem max_prob_diff_of_two_dice : 
  let d_vals := {-2, -1, 0, 1, 2}
  let total_outcomes := 36
  let diff_counts := [4, 5, 6, 5, 4] -- counts for d = -2, -1, 0, 1, 2 respectively
  let max_favorable := 6 -- highest count from diff_counts
  let probability := max_favorable / total_outcomes
  in probability = 1 / 6 :=
by
  sorry

end max_prob_diff_of_two_dice_l607_607259


namespace geometry_problem_l607_607880

theorem geometry_problem
  (PQ_parallel_RS : parallel PQ RS)
  (PST_straight_line : straight_line P S T)
  (angle_QPR : angle Q P R = 65)
  (angle_PRS : angle P R S = 95)
  (angle_QSR : angle Q S R = 130) :
  y = 20 :=
sorry

end geometry_problem_l607_607880


namespace problem_r_minus_s_l607_607161

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l607_607161


namespace coordinates_of_D_l607_607597

-- Coordinates of vertices A, B, and C
def A : ℝ × ℝ × ℝ := (3, -1, 2)
def B : ℝ × ℝ × ℝ := (1, 2, -4)
def C : ℝ × ℝ × ℝ := (-1, 1, 2)

-- Coordinate of vertex D that we want to prove
def D : ℝ × ℝ × ℝ := (1, -2, 8)

-- Definition to verify that the diagonals of the parallelogram bisect each other
def midpoint (P Q: ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

-- Proposition that AB and CD are diagonals of the parallelogram that intersect B
theorem coordinates_of_D : midpoint A C = midpoint B D :=
  by
    rw [A, B, C, D]
    dsimp [midpoint]
    apply prod.ext
    { norm_num }
    { apply prod.ext
      { norm_num }
      { norm_num } }

end coordinates_of_D_l607_607597


namespace tiles_19_not_feasible_tiles_20_not_feasible_l607_607264

-- Definition of the quadratic equation in Lean
def feasible_tiles (n : ℕ) : Prop :=
  ∃ (ℓ : ℤ), 2 * ℓ^2 + ℓ = n

-- Prove that 19 is not feasible
theorem tiles_19_not_feasible : ¬ feasible_tiles 19 :=
by {
  intro h,
  obtain ⟨ℓ, h⟩ := h,
  have h_discriminant : ∆ = 1 + 4 * 2 * 19 := by norm_num,
  have not_int_sqrt : ¬ ∃ k : ℤ, k^2 = 153 := by norm_num,
  cases (h : 2 * ℓ^2 + ℓ = 19),
  sorry
}

-- Prove that 20 is not feasible
theorem tiles_20_not_feasible : ¬ feasible_tiles 20 := 
by {
  intro h,
  obtain ⟨ℓ, h⟩ := h,
  have h_discriminant : ∆ = 1 + 4 * 2 * 20 := by norm_num,
  have not_int_sqrt : ¬ ∃ k : ℤ, k^2 = 161 := by norm_num,
  cases (h : 2 * ℓ^2 + ℓ = 20),
  sorry
}

end tiles_19_not_feasible_tiles_20_not_feasible_l607_607264


namespace find_shorter_side_length_l607_607311

open Nat

noncomputable def carpet_square_cost : ℕ := 15
noncomputable def total_cost : ℕ := 225
noncomputable def one_side_length : ℕ := 10
noncomputable def carpet_square_side : ℕ := 2

theorem find_shorter_side_length 
  (carpet_square_cost : ℕ) 
  (total_cost : ℕ) 
  (one_side_length : ℕ) 
  (carpet_square_side : ℕ) :
  let number_of_carpet_squares := total_cost / carpet_square_cost,
      area_per_carpet_square := carpet_square_side * carpet_square_side,
      total_carpet_area := number_of_carpet_squares * area_per_carpet_square,
      shorter_side := total_carpet_area / one_side_length
  in shorter_side = 6 := 
  by
  sorry

end find_shorter_side_length_l607_607311


namespace meeting_time_proof_l607_607131

variable (distance : ℝ) (speed_John : ℝ) (speed_Bob : ℝ)
variable (time_hours : ℝ) (time_minutes : ℝ)

-- Conditions
def John_speed := speed_John = 4
def Bob_speed := speed_Bob = 6
def initial_distance := distance = 7

-- Goal: Prove that the time for Bob and John to meet is 42 minutes.
theorem meeting_time_proof (h1 : John_speed) (h2 : Bob_speed) (h3 : initial_distance) :
    time_minutes = 42 :=
by
  sorry

end meeting_time_proof_l607_607131


namespace cos2_2x_sin2_2x_period_l607_607218

def min_positive_period (f : ℝ → ℝ) : ℝ := sorry

theorem cos2_2x_sin2_2x_period :
  min_positive_period (fun x => cos (2 * x) ^ 2 - sin (2 * x) ^ 2) = π / 2 :=
sorry

end cos2_2x_sin2_2x_period_l607_607218


namespace find_middle_number_l607_607240

theorem find_middle_number (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 12.5)
  (h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h3 : (a + b + c) / 3 = 11.6)
  (h4 : (c + d + e) / 3 = 13.5) : c = 12.8 :=
sorry

end find_middle_number_l607_607240


namespace solution_set_of_inequality_l607_607998

theorem solution_set_of_inequality (x : ℝ) :
  |x^2 - 2| < 2 ↔ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l607_607998


namespace min_value_of_x_plus_y_l607_607844

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy: 0 < y) (h: 9 * x + y = x * y) : x + y ≥ 16 := 
sorry

end min_value_of_x_plus_y_l607_607844


namespace sufficient_but_not_necessary_l607_607417

-- Define relevant predicates and concepts
def line (ℝ : Type*) := ℝ × ℝ
def plane (ℝ : Type*) := ℝ × ℝ × ℝ

-- Predicate that defines perpendicular relationship
def perp (n : line ℝ) (a : plane ℝ) : Prop := sorry

-- Predicate that defines a line being a subset of a plane
def subset (m : line ℝ) (a : plane ℝ) : Prop := sorry

-- Main Statement
theorem sufficient_but_not_necessary (m n : line ℝ) (α : plane ℝ) : 
  perp n α → ((subset m α → perp n m) ∧ ¬(perp n m → subset m α)) :=
by
  sorry

end sufficient_but_not_necessary_l607_607417


namespace triangle_area_l607_607789

noncomputable def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  in (x^2) / 9 - (y^2) / 16 = 1

theorem triangle_area
  (P F1 F2 : ℝ × ℝ)
  (h1: is_on_hyperbola P)
  (h2: F1 = (-5, 0))
  (h3: F2 = (5, 0))
  (h4: (dist P F1) * (dist P F2) = 32) :
  1/2 * (dist P F1) * (dist P F2) = 16 :=
begin
  sorry
end

end triangle_area_l607_607789


namespace range_of_intersection_distance_l607_607824

open Real EuclideanSpace

variables {A B F M N : EuclideanSpace ℝ (Fin 2)}
variables {C : A → B → Prop}
variables {x_1 x_2 : ℝ}
variables {y_1 y_2 : ℝ}

-- Define the parabola condition
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the points lie on the parabola
def points_on_parabola (x_1 y_1 x_2 y_2 : ℝ) : Prop :=
  parabola (x_1) (y_1) ∧ parabola (x_2) (y_2)

-- Define the sum of x-coordinates condition
def sum_x_coor (x_1 x_2 : ℝ) : Prop := x_1 + x_2 = 4

-- Define the final intersection points distance condition
def intersection_distance {M N : EuclideanSpace ℝ (Fin 2)} (MN_distance : ℝ) : Prop :=
  6 < MN_distance ∧ MN_distance < 12 ∨ MN_distance > 12

-- Main theorem statement
theorem range_of_intersection_distance 
  (h1 : parabola (1) (0)) -- Focus
  (h2 : points_on_parabola x_1 y_1 x_2 y_2) -- Points on parabola C
  (h3 : sum_x_coor x_1 x_2) -- Sum of x-coordinates

  : ∃ MN_distance : ℝ, intersection_distance MN_distance :=
sorry

end range_of_intersection_distance_l607_607824


namespace direct_proportion_m_n_l607_607851

theorem direct_proportion_m_n (m n : ℤ) (h₁ : m - 2 = 1) (h₂ : n + 1 = 0) : m + n = 2 :=
by
  sorry

end direct_proportion_m_n_l607_607851


namespace inequality_of_cubic_powers_l607_607031

theorem inequality_of_cubic_powers 
  (a b: ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h_cond : a * |a| > b * |b|) : 
  a^3 > b^3 := by
  sorry

end inequality_of_cubic_powers_l607_607031


namespace perimeter_of_square_l607_607859

theorem perimeter_of_square
  (length_rect : ℕ) (width_rect : ℕ) (area_rect : ℕ)
  (area_square : ℕ) (side_square : ℕ) (perimeter_square : ℕ) :
  (length_rect = 32) → (width_rect = 10) → 
  (area_rect = length_rect * width_rect) →
  (area_square = 5 * area_rect) →
  (side_square * side_square = area_square) →
  (perimeter_square = 4 * side_square) →
  perimeter_square = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof would go here
  sorry

end perimeter_of_square_l607_607859


namespace greatest_possible_sum_of_roots_l607_607463

noncomputable def quadratic_roots (c b : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ α + β = c ∧ α * β = b ∧ |α - β| = 1

theorem greatest_possible_sum_of_roots :
  ∃ (c : ℝ), ( ∃ b : ℝ, quadratic_roots c b) ∧
             ( ∀ (d : ℝ), ( ∃ b : ℝ, quadratic_roots d b) → d ≤ 11 ) ∧ c = 11 :=
sorry

end greatest_possible_sum_of_roots_l607_607463


namespace alice_guarantees_victory_l607_607191

theorem alice_guarantees_victory :
  ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 2020) →
  let m := (2021 - k) in (1 ≤ m ∧ m ≤ 2020) →
  ∃ alice_strategy : ℕ → ℕ, ∀ n, (0 ≤ n → n < 1010) →
  let erased_numbers := (finset.range 2020).product (finset.range 2020).erase n
  in erased_numbers.card = 2 →
  (finset.image (λ x, x^2) erased_numbers).sum (k, m) ∈ (set.range (0, 2020)) →
  (k^2 - m^2) % 2021 = 0 :=
sorry

end alice_guarantees_victory_l607_607191


namespace problem_statement_l607_607164

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l607_607164


namespace initial_amount_is_3_l607_607282

-- Define the initial amount of water in the bucket
def initial_water_amount (total water_added : ℝ) : ℝ :=
  total - water_added

-- Define the variables
def total : ℝ := 9.8
def water_added : ℝ := 6.8

-- State the problem
theorem initial_amount_is_3 : initial_water_amount total water_added = 3 := 
  by
    sorry

end initial_amount_is_3_l607_607282


namespace geometric_seq_arithmetic_triplet_l607_607883

-- Definition of being in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Condition that a_5, a_4, and a_6 form an arithmetic sequence
def is_arithmetic_triplet (a : ℕ → ℝ) (n : ℕ) : Prop :=
  2 * a n = a (n+1) + a (n+2)

-- Our specific problem translated into a Lean statement
theorem geometric_seq_arithmetic_triplet {a : ℕ → ℝ} (q : ℝ) :
  is_geometric_sequence a q →
  is_arithmetic_triplet a 4 →
  q = 1 ∨ q = -2 :=
by
  intros h_geo h_arith
  -- Proof here is omitted
  sorry

end geometric_seq_arithmetic_triplet_l607_607883


namespace OilBillJanuary_l607_607581

theorem OilBillJanuary (J F : ℝ) (h1 : F / J = 5 / 4) (h2 : (F + 30) / J = 3 / 2) : J = 120 := by
  sorry

end OilBillJanuary_l607_607581


namespace find_length_of_segment_CD_l607_607126

variables (A B C D E : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E]
variables (base height_area_BE_CABE trapezoid_Area_ABE_Alt : ℝ)

def is_isosceles_triangle (A B E : Type) : Prop := sorry

def line_segment_divides (A B E C D : Type) : Prop := sorry

def is_isosceles_trapezoid (A C E D : Type) : Prop := sorry

def is_smaller_isosceles_triangle (A C D : Type) : Prop := sorry

def length_of_segment_CD (CD length_CD_be : ℝ) := 
is_isosceles_triangle A B E ∧ 
(∃ (length_of_BE_ca : ℝ) (length_of_height : ℝ) (total_area : ℝ), 
  total_area = 180 ∧ length_of_height = 30 ∧ 
  ((1 / 2) * length_of_BE_ca * length_of_height) = total_area ∧ 
  line_segment_divides A B E C D ∧ 
  is_isosceles_trapezoid A C E D ∧ 
  ((1 / 2) * (length_of_BE_ca + length_CD_be * length_of_height) = length_of_CD ∧ 
  length_of_CD = 6)

theorem find_length_of_segment_CD (h₁ : is_isosceles_triangle A B E)
(h₂ : line_segment_divides A B E C D)
(h₃ : is_isosceles_trapezoid A C E D)
(h₄ : is_smaller_isosceles_triangle A B E)
(h₅ : height_area_BE_CABE = 180)
(h₆ : trapezoid_Area_ABE_Alt = 135)
(h₇ : height_area_BE_CABE = 30) :
  length_of_segment_CD 6 :=
sorry

end find_length_of_segment_CD_l607_607126


namespace nelly_earns_per_night_l607_607183

/-- 
  Nelly wants to buy pizza for herself and her 14 friends. Each pizza costs $12 and can feed 3 
  people. Nelly has to babysit for 15 nights to afford the pizza. We need to prove that Nelly earns 
  $4 per night babysitting.
--/
theorem nelly_earns_per_night 
  (total_people : ℕ) (people_per_pizza : ℕ) 
  (cost_per_pizza : ℕ) (total_nights : ℕ) (total_cost : ℕ) 
  (total_pizzas : ℕ) (cost_per_night : ℕ)
  (h1 : total_people = 15)
  (h2 : people_per_pizza = 3)
  (h3 : cost_per_pizza = 12)
  (h4 : total_nights = 15)
  (h5 : total_pizzas = total_people / people_per_pizza)
  (h6 : total_cost = total_pizzas * cost_per_pizza)
  (h7 : cost_per_night = total_cost / total_nights) :
  cost_per_night = 4 := sorry

end nelly_earns_per_night_l607_607183


namespace distinct_positive_integers_factors_PQ_RS_l607_607636

theorem distinct_positive_integers_factors_PQ_RS (P Q R S : ℕ) (hP : P > 0) (hQ : Q > 0) (hR : R > 0) (hS : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDistinctPQ : P ≠ Q) (hDistinctRS : R ≠ S) (hPQR_S : P + Q = R - S) :
  P = 4 :=
by
  sorry

end distinct_positive_integers_factors_PQ_RS_l607_607636


namespace max_cursed_roads_l607_607116

-- Define the structure of the problem
structure Empire where
  cities : ℕ
  roads : ℕ
  initialConnected : Bool
  
def curse_roads (G : Empire) (k : ℕ) : Empire :=
  { G with roads := G.roads - k }

-- Initial conditions
def initial_empire : Empire :=
  { cities := 1000, roads := 2017, initialConnected := True }

-- Function to determine if the graph is divided into N components after cursing k roads
def divides_into (G : Empire) (k N : ℕ) : Prop :=
  (k = N - 1) ∧ (N = 7)

-- Maximum value of roads that can be cursed while satisfying the conditions
theorem max_cursed_roads : 
  ∃ N, 
    divides_into (curse_roads initial_empire N) 6 7 ∧ 
    N = 2011 :=
begin
  sorry
end

end max_cursed_roads_l607_607116


namespace value_of_g_3_l607_607455

def g (x : ℚ) : ℚ := (x^2 + x + 1) / (5*x - 3)

theorem value_of_g_3 : g 3 = 13 / 12 :=
by
  -- Proof goes here
  sorry

end value_of_g_3_l607_607455


namespace min_overlap_l607_607184

variable (P : Set ℕ → ℝ)
variable (B M : Set ℕ)

-- Conditions
def P_B_def : P B = 0.95 := sorry
def P_M_def : P M = 0.85 := sorry

-- To Prove
theorem min_overlap : P (B ∩ M) = 0.80 := sorry

end min_overlap_l607_607184


namespace find_y_l607_607076

-- Definition of the function G
def G (a b c d : ℕ) : ℕ := a ^ b + c * d

-- Statement of the problem
theorem find_y (y : ℝ) : G 3 y 6 7 = 800 ↔ y = real.log 758 / real.log 3 :=
by
  -- Proof goes here
  sorry

end find_y_l607_607076


namespace cyclic_SNDM_l607_607197

-- Definitions and assumptions from conditions
variables {A B C D K L M N S : Type*}
variables [is_circumscribed_quadrilateral A B C D K L M N]
variables {Gamma : Circle}

-- Definition of cyclic quadrilateral SKBL
def is_cyclic_quadrilateral_SKBL : Prop :=
  ∠ K S L + ∠ K B L = 180

-- The main theorem statement
theorem cyclic_SNDM (h1 : ∠ K S L + ∠ K B L = 180) :
  ∠ M S N + ∠ M D N = 180 :=
sorry

end cyclic_SNDM_l607_607197


namespace number_of_equilateral_triangles_l607_607215

-- Definitions based on conditions
def line1 (k : ℤ) : ℝ → ℝ := fun x => k
def line2 (k : ℤ) : ℝ → ℝ := fun x => sqrt 3 * x + 3 * k
def line3 (k : ℤ) : ℝ → ℝ := fun x => -sqrt 3 * x + 3 * k

-- Define the range of k
def k_range := (-10 : ℤ) ≤ k ∧ k ≤ 10

-- Constants
def side_length_small_triangle : ℝ := 3 / sqrt 3

noncomputable def area_large_hexagon : ℝ := 1800 * sqrt 3
noncomputable def area_small_triangle : ℝ := (sqrt 3 / 4) * 9

-- Theorem to prove
theorem number_of_equilateral_triangles (k : ℤ) (h : k_range) : 
  (area_large_hexagon / area_small_triangle = 800) := by
  sorry

end number_of_equilateral_triangles_l607_607215


namespace factorize_expr_l607_607748

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l607_607748


namespace leak_empties_tank_in_18_hours_l607_607625

theorem leak_empties_tank_in_18_hours :
  let A : ℚ := 1 / 6
  let L : ℚ := 1 / 6 - 1 / 9
  (1 / L) = 18 := by
    sorry

end leak_empties_tank_in_18_hours_l607_607625


namespace range_of_a_l607_607799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x - 3 * a + 3 
  else Real.log x / Real.log a

-- Main statement to prove
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (5 / 4 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l607_607799


namespace angle_CHX_69_l607_607322

-- Define the basic setup: acute triangle, altitudes, orthocenter, and given angles
variables (A B C H X Y : Type) -- Define the vertices and points
variables [h1 : Triangle A B C] [h2 : Orthocenter H A B C]

-- Given conditions
variables (hAX : Altitude AX A B C) (hBY : Altitude BY A B C)
variables (angleBAC : ∠BAC = 58) (angleABC : ∠ABC = 69)

-- Theorem statement
theorem angle_CHX_69 : ∠CHX = 69 := 
by
  sorry

end angle_CHX_69_l607_607322


namespace equivalent_math_problem_l607_607007

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := - (Real.sqrt 1011 + Real.sqrt 1012)
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem equivalent_math_problem :
  (P * Q)^2 * R * S = 8136957 :=
by
  sorry

end equivalent_math_problem_l607_607007


namespace coordinates_va_coordinates_vb_parallel_condition_orthogonal_condition_l607_607063

variable (λ : ℝ)
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

def va (λ : ℝ) : ℝ × ℝ := (3 + 4*λ, 2 + λ)
def vb : ℝ × ℝ := (-5, 2)

-- Prove the coordinates:
theorem coordinates_va (λ : ℝ) : ((a.1 + λ * c.1), (a.2 + λ * c.2)) = (3 + 4*λ, 2 + λ) := 
by 
  sorry

theorem coordinates_vb : (2 * b.1 - a.1, 2 * b.2 - a.2) = (-5, 2) := 
by 
  sorry

-- Prove the parallel condition:
theorem parallel_condition (λ : ℝ) : (va λ = vb) → λ = -16 / 13 :=
by
  sorry

-- Prove the orthogonal condition:
theorem orthogonal_condition (λ : ℝ) : 
  (va λ.1 * vb.1 + va λ.2 * vb.2 = 0) → λ = 11 / 18 :=
by 
  sorry

end coordinates_va_coordinates_vb_parallel_condition_orthogonal_condition_l607_607063


namespace area_of_triangle_l607_607905

open Matrix

def a : Vector ℝ 2 := ![5.0, 1.0]
def b : Vector ℝ 2 := ![-3.0, 6.0]

theorem area_of_triangle :
  let det := Matrix.det ![![5, 1], ![-3, 6]] in
  ∃ (area : ℝ), area = 0.5 * det ∧ area = 16.5 :=
by
  -- Proof of the theorem will go here
  sorry

end area_of_triangle_l607_607905


namespace sequence_bounded_by_a_l607_607409

theorem sequence_bounded_by_a
  (c : ℝ) (hc : 0 < c)
  (f : ℝ → ℝ) (hf : ∀ x, f(x) = Real.sqrt (x + c))
  (a b : ℝ) (ha : a = (1 + Real.sqrt(1 + 4 * c)) / 2)
  (h0b : 0 < b) (hb : b < a)
  (x : ℕ → ℝ) (hx1 : x 1 = b) (hxn : ∀ n, x (n + 1) = f (x n))
  (n : ℕ) :
  x n < a := 
sorry

end sequence_bounded_by_a_l607_607409


namespace domain_f_l607_607732

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

theorem domain_f :
  (∀ x : ℝ, (x ≠ -6) ∧ (x ≠ -3) → ∃ y : ℝ, y = f x) ∧
  (∀ x : ℝ, x = -6 ∨ x = -3 → ¬(∃ y : ℝ, y = f x)) :=
sorry

end domain_f_l607_607732


namespace reciprocal_of_fraction_subtraction_l607_607376

theorem reciprocal_of_fraction_subtraction : (1 / ((2 / 3) - (3 / 4))) = -12 := by
  sorry

end reciprocal_of_fraction_subtraction_l607_607376


namespace base_length_of_parallelogram_l607_607566

-- Definitions and conditions
def parallelogram_area (base altitude : ℝ) : ℝ := base * altitude
def altitude (base : ℝ) : ℝ := 2 * base

-- Main theorem to prove
theorem base_length_of_parallelogram (A : ℝ) (base : ℝ)
  (hA : A = 200) 
  (h_altitude : altitude base = 2 * base) 
  (h_area : parallelogram_area base (altitude base) = A) : 
  base = 10 := 
sorry

end base_length_of_parallelogram_l607_607566


namespace total_perimeter_of_compound_shape_l607_607881

-- Definitions of the conditions from the original problem
def triangle1_side : ℝ := 10
def triangle2_side : ℝ := 6
def shared_side : ℝ := 6

-- A theorem to represent the mathematically equivalent proof problem
theorem total_perimeter_of_compound_shape 
  (t1s : ℝ := triangle1_side) 
  (t2s : ℝ := triangle2_side)
  (ss : ℝ := shared_side) : 
  t1s = 10 ∧ t2s = 6 ∧ ss = 6 → 3 * t1s + 3 * t2s - ss = 42 := 
by
  sorry

end total_perimeter_of_compound_shape_l607_607881


namespace number_of_trapezoid_solutions_l607_607209

theorem number_of_trapezoid_solutions
  (A h : ℕ)
  (H_A : A = 1800)
  (H_h : h = 60)
  (H_multiple_of_6 : ∀ b_1 b_2, (b_1 + b_2 = 60) → (∃ m n : ℕ, b_1 = 6 * m ∧ b_2 = 6 * n)) :
  ({(b_1, b_2) : ℕ × ℕ | b_1 + b_2 = 60 ∧ ∃ m n : ℕ, b_1 = 6 * m ∧ b_2 = 6 * n }.size > 3) := 
sorry

end number_of_trapezoid_solutions_l607_607209


namespace Osborn_dressing_time_on_Wednesday_l607_607935

variable (t_M t_T t_Th t_F t_avg_old t_W : ℕ)

def time_on_days : ℕ := t_M + t_T + t_Th + t_F
def avg_time_on_days : ℕ := time_on_days / 4

theorem Osborn_dressing_time_on_Wednesday :
  t_M = 2 ∧ t_T = 4 ∧ t_Th = 4 ∧ t_F = 2 ∧ t_avg_old = 3 → 
  t_W = 3 :=
by
  sorry

end Osborn_dressing_time_on_Wednesday_l607_607935


namespace value_of_a1_over_d_l607_607792

noncomputable def arithmetic_sequence_a_n (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

theorem value_of_a1_over_d 
  (a_1 d : ℤ) 
  (h_arith_seq : ∀ n : ℕ, arithmetic_sequence_a_n a_1 d n = a_1 + (n - 1) * d)
  (h_non_zero_diff : d ≠ 0)
  (h_geom_seq : (arithmetic_sequence_a_n a_1 d 1)
                ∘ (arithmetic_sequence_a_n a_1 d 3)
                ∘ (arithmetic_sequence_a_n a_1 d 7)) :
  (a_1 / d) = 2 :=
begin
  sorry -- proof omitted
end

end value_of_a1_over_d_l607_607792


namespace probability_x_square_lt_4_l607_607198

theorem probability_x_square_lt_4 :
  let interval : Set ℝ := Set.Icc (-2) 5 in
  let favorable : Set ℝ := {x | -2 < x ∧ x < 2} in
  let total_length := 5 - (-2) in
  let favorable_length := 2 - (-2) in
  (Set.measure_theory.volume favorable).toReal / (Set.measure_theory.volume interval).toReal = 4 / 7 :=
by
  -- Provide the necessary calculation in the final proof
  sorry

end probability_x_square_lt_4_l607_607198


namespace min_value_l607_607400

noncomputable def min_value_of_expression (a b: ℝ) :=
    a > 0 ∧ b > 0 ∧ a + b = 1 → (∃ (m : ℝ), (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2)

theorem min_value (a b: ℝ) (h₀: a > 0) (h₁: b > 0) (h₂: a + b = 1) :
    ∃ m, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2 := 
by
    sorry

end min_value_l607_607400


namespace coeff_a3b4c3_expansion_l607_607383

theorem coeff_a3b4c3_expansion :
  ∑ (k1 k2 k3 : ℕ) in finset.filter (λ k, k.1 + k.2 + k.3 = 10 ∧ k.1 = 3 ∧ k.2 = 4 ∧ k.3 = 3) 
  (finset.product (finset.product (finset.range (10 + 1)) (finset.range (10 + 1))) (finset.range (10 + 1))), 
  (nat.choose 10 k1) * (nat.choose (10 - k1) k2) * (nat.choose (10 - k1 - k2) k3) * 2^k2 * (-3)^k3 = 
  - nat.choose 10 3 * nat.choose 7 4 * 16 * 27 :=
by 
  sorry

end coeff_a3b4c3_expansion_l607_607383


namespace shaded_fraction_in_fifth_diagram_l607_607089

-- Definitions for conditions
def geometric_sequence (a₀ r n : ℕ) : ℕ := a₀ * r^n

def total_triangles (n : ℕ) : ℕ := n^2

-- Lean theorem statement
theorem shaded_fraction_in_fifth_diagram 
  (a₀ r n : ℕ) 
  (h_geometric : a₀ = 1) 
  (h_ratio : r = 2)
  (h_step_number : n = 4):
  (geometric_sequence a₀ r n) / (total_triangles (n + 1)) = 16 / 25 :=
by
  sorry

end shaded_fraction_in_fifth_diagram_l607_607089


namespace arithmetic_sequence_a9_l607_607106

variable {α : Type*} [AddMonoid α] [HasSmul ℕ α]

-- Define the arithmetic sequence
variable (a : ℕ → α)

-- Define the conditions
def condition1 := a 5 + a 7 = 16
def condition2 := a 3 = 4

-- State the theorem to be proven
theorem arithmetic_sequence_a9 (a : ℕ → ℕ) (h1: condition1 a) (h2: condition2 a) : a 9 = 12 := 
by 
  sorry

end arithmetic_sequence_a9_l607_607106


namespace Peter_Basil_Equal_l607_607641

-- Definitions based on conditions (a):
def PeterNums (a b c : ℕ) : set ℕ :=
  {Int.gcd a b, Int.gcd b c, Int.gcd c a}

def BasilNums (x y z : ℕ) : set ℕ :=
  {Int.lcm x y, Int.lcm y z, Int.lcm z x}

-- Proof goal statement:
theorem Peter_Basil_Equal (a b c x y z : ℕ) :
  PeterNums a b c = BasilNums x y z → a = b ∧ b = c :=
by
  sorry

end Peter_Basil_Equal_l607_607641


namespace andy_time_correct_l607_607325

-- Define the conditions
def time_dawn_wash_dishes : ℕ := 20
def time_andy_put_laundry : ℕ := 2 * time_dawn_wash_dishes + 6

-- The theorem to prove
theorem andy_time_correct : time_andy_put_laundry = 46 :=
by
  -- Proof goes here
  sorry

end andy_time_correct_l607_607325


namespace graph_is_finite_set_of_distinct_points_l607_607450

def cost_function (n : ℕ) : ℕ :=
  20 * n + 500

theorem graph_is_finite_set_of_distinct_points :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 →
  ∃ (points : set (ℕ × ℕ)), points = { (n, cost_function n) | n ∈ finset.range 20 + 1 } ∧
  (∀ p ∈ points, ∃ n', p = (n', cost_function n')) ∧
  (∀ p1 p2 ∈ points, p1 = p2 → n = n') :=
by
  sorry

end graph_is_finite_set_of_distinct_points_l607_607450


namespace range_of_k_l607_607431

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x + a / x
noncomputable def h (x : ℝ) : ℝ := f x - g x (-1)

theorem range_of_k (k : ℝ) :
  (∀ x1 x2 ∈ Icc (1/Real.exp 1) 3, (f x1 - g x2 (-1)) / (k - 1) ≤ 1)
  → k ∈ Icc (←∞) (-34/3 + 2*Real.log 3) ∪ Ioo 1 ∞ :=
sorry

end range_of_k_l607_607431


namespace correct_propositions_l607_607816

-- Define necessary conditions
def cond1 (l1 l2 : Prop) : Prop := ¬(∃ p, p ∈ l1 ∧ p ∈ l2) → ∥l1 ∥ l2
def cond2 (l1 l2 : Prop) : Prop := (∃ p, p ∈ l1 ∧ p ∈ l2) ∧ α(l1, l2) = 90 → ∃ p, p ∈ l1 ∧ p ∈ l2
def cond3 (l1 l2 : Prop) : Prop := ¬∥l1 ∧ ¬(∃ p, p ∈ l1 ∧ p ∈ l2) → skew l1 l2
def cond4 (l1 l2 : Prop) : Prop := ¬∃ plane, l1 ∈ plane ∧ l2 ∈ plane → skew l1 l2

-- Define the propositions
def proposition1 (l1 l2 : Prop) : Prop := cond1 l1 l2
def proposition2 (l1 l2 : Prop) : Prop := cond2 l1 l2
def proposition3 (l1 l2 : Prop) : Prop := cond3 l1 l2
def proposition4 (l1 l2 : Prop) : Prop := cond4 l1 l2

-- Statement to prove
theorem correct_propositions (l1 l2 : Prop) (h1 : ¬proposition1 l1 l2) (h2 : ¬proposition2 l1 l2) (h3 : proposition3 l1 l2) (h4 : proposition4 l1 l2) : 
  (h3 ∧ h4) :=
by
  sorry

end correct_propositions_l607_607816


namespace derivative_at_x0_l607_607913

variables {f : ℝ → ℝ} {x0 : ℝ}

theorem derivative_at_x0 (h_lim : (𝓝[≠] 0).lim (λ h, (f x0 - f (x0 - h)) / h) 6) : deriv f x0 = 6 :=
sorry

end derivative_at_x0_l607_607913


namespace problem_solution_l607_607018

theorem problem_solution (x : ℝ) (h : ∃ (A B : Set ℝ), A = {0, 1, 2, 4, 5} ∧ B = {x-2, x, x+2} ∧ A ∩ B = {0, 2}) : x = 0 :=
sorry

end problem_solution_l607_607018


namespace percentage_students_on_trip_l607_607082

variable (total_students : ℕ)
variable (students_more_than_100 : ℕ)
variable (students_on_trip : ℕ)
variable (percentage_more_than_100 : ℝ)
variable (percentage_not_more_than_100 : ℝ)

-- Given conditions
def condition_1 := percentage_more_than_100 = 0.16
def condition_2 := percentage_not_more_than_100 = 0.75

-- The final proof statement
theorem percentage_students_on_trip :
  percentage_more_than_100 * (total_students : ℝ) /
  ((1 - percentage_not_more_than_100)) / (total_students : ℝ) * 100 = 64 :=
by
  sorry

end percentage_students_on_trip_l607_607082


namespace max_a_sum_plus_b_sum_l607_607509

theorem max_a_sum_plus_b_sum (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n, a n ≤ a (n + 1)) ∧ -- a non-decreasing sequence
  (∀ m, b m = Nat.find (λ n, a n ≥ m)) ∧ -- definition of b
  a 19 = 85 → -- given condition a₁₉ = 85
  (Finset.range 19).sum a + (Finset.range 85).sum b = 1700 := -- prove the maximum sum is 1700
sorry

end max_a_sum_plus_b_sum_l607_607509


namespace area_relationship_hexagon_triangle_l607_607728

theorem area_relationship_hexagon_triangle
  (A B C D E F O : Point)
  (h1 : IsInscribedHexagon A B C D E F O)
  (h2 : IsDiameter A D O)
  (h3 : IsDiameter B E O)
  (h4 : IsDiameter C F O) :
  2 * AreaTriangle A C E = AreaHexagon A B C D E F :=
sorry

end area_relationship_hexagon_triangle_l607_607728


namespace problem_r_minus_s_l607_607158

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l607_607158


namespace distinct_solutions_diff_l607_607155

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l607_607155


namespace election_max_k_1002_l607_607482

/-- There are 2002 candidates initially. 
In each round, one candidate with the least number of votes is eliminated unless a candidate receives more than half the votes.
Determine the highest possible value of k if Ostap Bender is elected in the 1002nd round. -/
theorem election_max_k_1002 
  (number_of_candidates : ℕ)
  (number_of_rounds : ℕ)
  (k : ℕ)
  (h1 : number_of_candidates = 2002)
  (h2 : number_of_rounds = 1002)
  (h3 : k ≤ number_of_candidates - 1)
  (h4 : ∀ n : ℕ, n < number_of_rounds → (k + n) % (number_of_candidates - n) ≠ 0) : 
  k = 2001 := sorry

end election_max_k_1002_l607_607482


namespace term_50_is_12_l607_607863

-- Define the sequence generation rules
def next_term (n : ℕ) : ℕ :=
if n < 20 then 3 * n
else if n ≥ 20 ∧ n ≤ 60 ∧ n % 2 = 0 then n + 10
else if n > 60 ∧ n % 2 = 0 then n / 5
else if n > 20 ∧ n % 2 = 1 then n - 7
else n -- unspecified behavior, should not be reached given problem constraints

-- Define the sequence as an infinite sequence
def sequence : ℕ → ℕ
| 0 := 120
| (k + 1) := next_term (sequence k)

-- Theorem statement for the 50th term
theorem term_50_is_12 : sequence 50 = 12 :=
sorry

end term_50_is_12_l607_607863


namespace proof_problem_l607_607424

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

def a_n (n : ℕ) : ℝ := -2 * (1/3) ^ n

def S (n : ℕ) : ℝ := n^2

def b_n (n : ℕ) : ℕ := 2 * n - 1

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (1 : ℝ) / (b_n i * b_n (i + 1))

theorem proof_problem :
  (∀ n, a_n n = -2 * (1/3) ^ n) ∧
  (∀ n, S n - S (n - 1) = (S n + S (n + 1)) ^ (1/2) + (S (n + 1)) ^ (1/2)) ∧
  (∀ n, b_n n = 2 * n - 1) ∧
  (∃ n : ℕ, n > 0 ∧ T n > 1000/2009) :=
begin
  sorry
end

end proof_problem_l607_607424


namespace jerry_reaches_five_probability_l607_607497

noncomputable def probability_move_reaches_five_at_some_point : ℚ :=
  let num_heads_needed := 7
  let num_tails_needed := 3
  let total_tosses := 10
  let num_ways_to_choose_heads := Nat.choose total_tosses num_heads_needed
  let total_possible_outcomes : ℚ := 2^total_tosses
  let prob_reach_4 := num_ways_to_choose_heads / total_possible_outcomes
  let prob_reach_5_at_some_point := 2 * prob_reach_4
  prob_reach_5_at_some_point

theorem jerry_reaches_five_probability :
  probability_move_reaches_five_at_some_point = 15 / 64 := by
  sorry

end jerry_reaches_five_probability_l607_607497


namespace find_divisor_value_l607_607670

theorem find_divisor_value (x : ℝ) (h : 63 / x = 63 - 42) : x = 3 :=
by
  sorry

end find_divisor_value_l607_607670


namespace factorize_expression_l607_607741

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l607_607741


namespace fern_total_payment_l607_607378

theorem fern_total_payment (price_high_heels : ℕ) (price_ballet_slippers : ℕ) (num_ballet_slippers : ℕ) : 
  price_high_heels = 60 → 
  price_ballet_slippers = (2 * price_high_heels) / 3 → 
  num_ballet_slippers = 5 →
  price_high_heels + num_ballet_slippers * price_ballet_slippers = 260 :=
by {
  intro h1,
  intro h2,
  intro h3,
  sorry -- proof would go here
}

end fern_total_payment_l607_607378


namespace find_a_parallel_tangent_l607_607521

-- Definition 1: The curve equation at point x
def curve (x : ℝ) : ℝ := (1 + Real.cos x) / Real.sin x

-- Condition: The point on the curve
def point_x : ℝ := Real.pi / 2
def point_y : ℝ := 1

-- Condition: The tangential parallel line equation form
def line_slope (a : ℝ) : ℝ := 1 / a

-- The theorem statement
theorem find_a_parallel_tangent :
  (∃ (a : ℝ), ∀ (x : ℝ), curve x = point_y → 
  (derivative curve x) = -1 ∧ (line_slope a = -1) ) :=
sorry

end find_a_parallel_tangent_l607_607521


namespace solve_fraction_eq_zero_l607_607000

theorem solve_fraction_eq_zero (a : ℝ) (h : a ≠ -1) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by {
  sorry
}

end solve_fraction_eq_zero_l607_607000


namespace percentage_of_rotten_oranges_l607_607675

-- Define the conditions
def total_oranges : ℕ := 600
def total_bananas : ℕ := 400
def rotten_bananas_percentage : ℝ := 0.08
def good_fruits_percentage : ℝ := 0.878

-- Define the proof problem
theorem percentage_of_rotten_oranges :
  let total_fruits := total_oranges + total_bananas
  let number_of_rotten_bananas := rotten_bananas_percentage * total_bananas
  let number_of_good_fruits := good_fruits_percentage * total_fruits
  let number_of_rotten_fruits := total_fruits - number_of_good_fruits
  let number_of_rotten_oranges := number_of_rotten_fruits - number_of_rotten_bananas
  let percentage_of_rotten_oranges := (number_of_rotten_oranges / total_oranges) * 100
  percentage_of_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l607_607675


namespace Malou_average_is_correct_l607_607530

def quiz1_score : ℕ := 91
def quiz2_score : ℕ := 90
def quiz3_score : ℕ := 92
def total_score : ℕ := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes : ℕ := 3

def Malous_average_score : ℕ := total_score / number_of_quizzes

theorem Malou_average_is_correct : Malous_average_score = 91 := by
  sorry

end Malou_average_is_correct_l607_607530


namespace num_functions_with_given_range_l607_607989

theorem num_functions_with_given_range : 
  ∃ (X : Set ℤ) (f : ℤ → ℤ), (∀ x ∈ X, f x = x^2 + 1) ∧ (Set.Range f = {5, 10}) ∧ (X.toFinset.card = 9) := sorry

end num_functions_with_given_range_l607_607989


namespace problem_equivalent_l607_607454

theorem problem_equivalent (a b : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^2 * x^2)/2 + (a^3 * x^3)/6 + (a^4 * x^4)/24 + (a^5 * x^5)/120) : 
  a - b = -38 :=
sorry

end problem_equivalent_l607_607454


namespace zoo_feeding_ways_l607_607318

theorem zoo_feeding_ways :
  let males := fin 6
  let females := fin 6
  (n: ℕ)
  (start_with_lion: males)
  (next_feeding: ∀ (k: ℕ) (males left males right females right: fin 6), k < n →
    ((k % 2 = 0) → 
      (males ≠ start_with_lion) ∧
      males ≠ left ∧ 
      males ≠ females)
    ∧
    ((k % 2 = 1) → 
      (females ≠ start_with_lion) ∧ 
      females ≠ males right))
  (cond: ∀ (k: ℕ) (males_right males_left females_left females right: fin 6), k < n → (males_right ≠ males_left) ∧ (females_right ≠ females_left)
  n = 14400 := 
      sorry

end zoo_feeding_ways_l607_607318


namespace max_total_length_of_cuts_l607_607276

theorem max_total_length_of_cuts :
  ∀ (n : ℕ), n = 30 →
  ∀ (p : ℕ), p = 225 →
  (∃ k : ℕ, n * n = p * k) →
  1065 = ((p * 10 - 4 * n) / 2) :=
by
  intros n hn p hp h
  rw [hn, hp]
  have : 30 * 30 = 225 * 4 := by simp
  exact sorry

end max_total_length_of_cuts_l607_607276


namespace trains_to_or_from_jena_l607_607101

theorem trains_to_or_from_jena
  (total_trains : ℕ)
  (trains_freiburg trains_göttingen trains_hamburg trains_ingolstadt : ℕ)
  (h₁ : total_trains = 40)
  (h₂ : trains_freiburg = 10)
  (h₃ : trains_göttingen = 10)
  (h₄ : trains_hamburg = 10)
  (h₅ : trains_ingolstadt = 10)
  : (2 * total_trains - (trains_freiburg + trains_göttingen + trains_hamburg + trains_ingolstadt)) = 40 := 
by 
  have h₆ : 2 * total_trains = 80, from congr_arg (λ x, 2 * x) h₁
  rw [h₂, h₃, h₄, h₅] at *
  simp only [Nat.add_sub_cancel_right, zero_add] at h₆
  exact h₆ 

end trains_to_or_from_jena_l607_607101


namespace even_product_probability_l607_607966

-- Define the spinners C and D
def spinner_C := {1, 1, 2, 3, 5, 5}
def spinner_D := {1, 2, 3, 4}

-- Define the event of interest: the product is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probability calculations
def total_outcomes := card spinner_C * card spinner_D
def even_outcomes := ∑ i in spinner_C, ∑ j in spinner_D, if is_even (i * j) then 1 else 0

-- The theorem stating the required probability
theorem even_product_probability : (even_outcomes : ℚ) / total_outcomes = 1 / 2 :=
sorry

end even_product_probability_l607_607966


namespace sum_of_coordinates_l607_607354

def g (x : ℝ) : ℝ := sorry

def h (x : ℝ) : ℝ := (g x) ^ 3

theorem sum_of_coordinates :
  g 2 = -5 → (g 2) = -5 →
  (h 2 = (g 2) ^ 3) → h 2 = -125 →
  (2 + -125) = -123 :=
by
  intros h1 h2 h3 h4
  rw [h2] at h4
  rw [h3] at h4
  rw [h2]
  exact h4

end sum_of_coordinates_l607_607354


namespace total_cost_of_purchases_l607_607128

def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

theorem total_cost_of_purchases : cost_cat_toy + cost_cage = 21.95 := by
  -- skipping the proof
  sorry

end total_cost_of_purchases_l607_607128


namespace find_a_value_l607_607522

/-- Given the distribution of the random variable ξ as p(ξ = k) = a (1/3)^k for k = 1, 2, 3, 
    prove that the value of a that satisfies the probabilities summing to 1 is 27/13. -/
theorem find_a_value (a : ℝ) :
  (a * (1 / 3) + a * (1 / 3)^2 + a * (1 / 3)^3 = 1) → a = 27 / 13 :=
by 
  intro h
  sorry

end find_a_value_l607_607522


namespace drink_all_tea_l607_607541

noncomputable def can_drink_all_tea : Prop :=
  ∀ (initial_hare_cup : ℕ) (initial_dormouse_cup : ℕ), 
  0 ≤ initial_hare_cup ∧ initial_hare_cup < 30 ∧ 
  0 ≤ initial_dormouse_cup ∧ initial_dormouse_cup < 30 ∧ 
  initial_hare_cup ≠ initial_dormouse_cup →
  ∃ (rotation : ℕ → ℕ), 
  (∀ (n : ℕ), 
    (rotation n) % 30 = (initial_hare_cup + n) % 30 ∧ 
    (rotation n + x) % 30 ≠ initial_hare_cup % 30 ∧ 
    (∀ m, m < n → 
      (rotation (m+1)) % 30 ≠ (rotation m) % 30)) ∧ 
    set.range rotation = {0,1,2,...,29} 

theorem drink_all_tea : can_drink_all_tea :=
  by sorry

end drink_all_tea_l607_607541


namespace find_ellipse_equation_max_segment_length_l607_607412

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def conditions (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  let e := real.sqrt 2 / 2 in
  let c := a * e in
  let latus_rectum := 2 * b^2 / a in
  (latus_rectum = real.sqrt 2) ∧ (a^2 = b^2 + c^2)

theorem find_ellipse_equation :
  ∃ (a b : ℝ), a = real.sqrt 2 ∧ b = 1 ∧ (conditions a b ⟨by norm_num, by norm_num⟩) ∧ (ellipse_equation a b ⟨by norm_num, by norm_num⟩) :=
sorry

noncomputable def line_passing_point (k : ℝ) : ℝ → ℝ := λ x, k * x + 2

noncomputable def area_of_triangle (k : ℝ) (x1 x2 : ℝ) : ℝ :=
  let y1 := k * x1 + 2 in
  let y2 := k * x2 + 2 in
  abs (x1 * y2 - x2 * y1) / 2

theorem max_segment_length :
  ∀ (k : ℝ), k^2 = 7 / 2 → ∃ (x1 x2 : ℝ), let l := line_passing_point k in ∃ (a b : ℝ), 
  a = real.sqrt 2 ∧ b = 1 ∧ (ellipse_equation a b ⟨by norm_num, by norm_num⟩) ∧ 
  (area_of_triangle k x1 x2 = real.sqrt 2 / 2) ∧ (abs (x1 - x2) = 3 / 2) :=
sorry

end find_ellipse_equation_max_segment_length_l607_607412


namespace slope_tangent_line_at_one_l607_607233

open Real

theorem slope_tangent_line_at_one (f : ℝ → ℝ) (x : ℝ) (h : f = fun x => x * exp x) (hx : x = 1) :
  deriv f 1 = 2 * exp 1 :=
by 
  sorry

end slope_tangent_line_at_one_l607_607233


namespace binomial_7_4_eq_35_l607_607718

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607718


namespace problem_1_range_and_interval_problem_2_find_sin_value_l607_607061

open Real

-- Define the function f
def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x + 1

-- Problem (1)
theorem problem_1_range_and_interval :
  (∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ k : ℤ, ∀ x : ℝ, - 5 * π / 6 + 2 * k * π ≤ x ∧ x ≤ π / 6 + 2 * k * π → 
  ∀ ε > 0, x < x + ε → f x < f (x + ε)) :=
sorry

-- Problem (2)
theorem problem_2_find_sin_value (α : ℝ) :
  f α = 13 / 5 ∧ π / 6 < α ∧ α < 2 * π / 3 → sin (2 * α + 2 * π / 3) = -24 / 25 :=
sorry

end problem_1_range_and_interval_problem_2_find_sin_value_l607_607061


namespace percentage_deposited_to_wife_is_33_l607_607310

-- Definitions based on the conditions
def total_income : ℝ := 800000
def children_distribution_rate : ℝ := 0.20
def number_of_children : ℕ := 3
def donation_rate : ℝ := 0.05
def final_amount : ℝ := 40000

-- We can compute the intermediate values to use them in the final proof
def amount_distributed_to_children : ℝ := total_income * children_distribution_rate * number_of_children
def remaining_after_distribution : ℝ := total_income - amount_distributed_to_children
def donation_amount : ℝ := remaining_after_distribution * donation_rate
def remaining_after_donation : ℝ := remaining_after_distribution - donation_amount
def deposited_to_wife : ℝ := remaining_after_donation - final_amount

-- The statement to prove
theorem percentage_deposited_to_wife_is_33 :
  (deposited_to_wife / total_income) * 100 = 33 := by
  sorry

end percentage_deposited_to_wife_is_33_l607_607310


namespace angle_between_lines_is_equal_to_angle_between_circles_l607_607351

-- Define the geometrical entities and the conditions
variables {S₁ S₂ : Type} [circle S₁] [circle S₂]
variables {A B P₁ Q₁ P₂ Q₂ : point}

-- Define the intersections and line definitions
axiom intersects_at_A_and_B : intersect S₁ S₂ A
axiom also_intersects_at_B : intersect S₁ S₂ B
axiom line_passes_through_A_p : line_through p A
axiom line_passes_through_A_q : line_through q A
axiom line_p_intersects_S₁_P₁ : line_intersects_circle p S₁ P₁
axiom line_q_intersects_S₁_Q₁ : line_intersects_circle q S₁ Q₁
axiom line_p_intersects_S₂_P₂ : line_intersects_circle p S₂ P₂
axiom line_q_intersects_S₂_Q₂ : line_intersects_circle q S₂ Q₂

-- The statement we need to prove
theorem angle_between_lines_is_equal_to_angle_between_circles :
  angle_between_lines P₁ Q₁ = angle_between_lines P₂ Q₂ ↔ angle_between_circles S₁ S₂ :=
sorry -- Proof to be provided

end angle_between_lines_is_equal_to_angle_between_circles_l607_607351


namespace quadratic_two_distinct_real_roots_l607_607231

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ - k^2 = 0) ∧ (x₂^2 - x₂ - k^2 = 0) :=
by
  -- The proof is omitted as requested.
  sorry

end quadratic_two_distinct_real_roots_l607_607231


namespace north_southland_population_increase_l607_607118

noncomputable def births_per_hour : ℝ := 1 / 6
noncomputable def deaths_per_hour : ℝ := 1 / 36
noncomputable def hours_per_day : ℝ := 24
noncomputable def days_per_year : ℝ := 365

-- Calculate yearly net population increase based on given birth and death rates.
theorem north_southland_population_increase :
  let net_rate_per_hour := births_per_hour - deaths_per_hour in
  let net_rate_per_day := net_rate_per_hour * hours_per_day in
  let annual_increase := net_rate_per_day * days_per_year in
  round annual_increase.toReal = 1200 :=
by
  sorry

end north_southland_population_increase_l607_607118


namespace midpoint_probability_l607_607139

def T : Set (ℤ × ℤ × ℤ) := 
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2.1 ∧ p.2.1 ≤ 4 ∧ 0 ≤ p.2.2 ∧ p.2.2 ≤ 5}

noncomputable def count_valid_pairs (T : Set (ℤ × ℤ × ℤ)) : ℕ := 2880

noncomputable def total_pairs_count (T : Set (ℤ × ℤ × ℤ)) : ℕ := (∑ i in (Finset.range (4 * 5 * 6)), i) - (4 * 5 * 6)

theorem midpoint_probability : 
  ∃ p q : ℕ, p + q = 167 ∧ gcd p q = 1 ∧ (count_valid_pairs T).toRat / (total_pairs_count T).toRat = p / q := 
sorry

end midpoint_probability_l607_607139


namespace right_triangle_inequality_equality_condition_l607_607196

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b ≤ 5 * c :=
by 
  sorry

theorem equality_condition (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b = 5 * c ↔ a / b = 3 / 4 :=
by
  sorry

end right_triangle_inequality_equality_condition_l607_607196


namespace initial_markup_percentage_l607_607308

theorem initial_markup_percentage (C M : ℝ) 
  (h1 : C > 0) 
  (h2 : (1 + M) * 1.25 * 0.92 = 1.38) :
  M = 0.2 :=
sorry

end initial_markup_percentage_l607_607308


namespace number_of_intersections_l607_607111

theorem number_of_intersections (A B: ℝ × ℝ) (r: ℝ) :
  A = (-1, 0) → B = (0, 1) → r = 4 →
  (∃ P : ℝ × ℝ, (P.1 + 1)^2 + P.2^2 - (P.1^2 + (P.2 - 1)^2) = 4 ∧
                 P.1^2 + P.2^2 = r) → 
  ∃! (P Q : ℝ × ℝ), (P.1 + P.2 - 2 = 0) ∧ (Q.1 + Q.2 - 2 = 0) ∧ (P.1^2 + P.2^2 = r) ∧ (Q.1^2 + Q.2^2 = r) :=
begin
  intros hA hB hr hP_exists,
  sorry -- Proof steps are not required
end

end number_of_intersections_l607_607111


namespace sum_of_transformed_roots_l607_607177

theorem sum_of_transformed_roots:
  ∀ (z1 z2 z3 z4 : ℂ),
  let P := polynomial.C (-1) + polynomial.X * (polynomial.C 2) +
           polynomial.X^2 * (polynomial.C (-4)) + polynomial.X^3 * (polynomial.C (-3)) +
           polynomial.X^4,
      g := λ z : ℂ, 3 * complex.I * complex.conj z + 2,
      R_roots := [g z1, g z2, g z3, g z4],
      R := polynomial.C (R_roots.product) + polynomial.X * (R_roots.sum) +
           polynomial.X^2 * (R_roots.paired_sum_products) + polynomial.X^3 * (g_roots.sum) +
           polynomial.X^4
  in (P.roots = [z1, z2, z3, z4] ∧ g_roots = R_roots) →
  R.coeff 2 + R.coeff 0 = -45 :=
by {
  sorry
}

end sum_of_transformed_roots_l607_607177


namespace log_sine_sum_l607_607374

theorem log_sine_sum
  : ∑ x in Finset.range 89, Real.log10 (Real.sin (Real.pi * (x + 1) / 180)) = - 45 * Real.log10 2 :=
by
  sorry

end log_sine_sum_l607_607374


namespace problem_l607_607402

noncomputable def p (m : ℝ) : Prop :=
∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0

noncomputable def q (m : ℝ) : Prop :=
∃ x : ℝ, x ∈ set.Icc (1 : ℝ) 2 → real.log (x^2 - m*x + 1) / real.log (1/2) < -1

theorem problem (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 
  m < 1/2 ∨ m = 3/2 := 
sorry

end problem_l607_607402


namespace pure_imaginary_complex_number_l607_607087

theorem pure_imaginary_complex_number (a : ℝ) (ha : (∃ (b : ℂ), b = (2 - a * complex.I) / (1 + complex.I) ∧ ∃ (y : ℝ), b = y * complex.I)) : a = 2 :=
sorry

end pure_imaginary_complex_number_l607_607087


namespace trigonometric_identity_l607_607008

theorem trigonometric_identity (α : ℝ) 
  (h : (sin α + 3 * cos α) / (3 * cos α - sin α) = 5) :
  cos α ^ 2 + 1 / 2 * sin (2 * α) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l607_607008


namespace proof_Bill_age_is_24_l607_607696

noncomputable def Bill_is_24 (C : ℝ) (Bill_age : ℝ) (Daniel_age : ℝ) :=
  (Bill_age = 2 * C - 1) ∧ 
  (Daniel_age = C - 4) ∧ 
  (C + Bill_age + Daniel_age = 45) → 
  (Bill_age = 24)

theorem proof_Bill_age_is_24 (C Bill_age Daniel_age : ℝ) : 
  Bill_is_24 C Bill_age Daniel_age :=
by
  sorry

end proof_Bill_age_is_24_l607_607696


namespace min_value_a_l607_607524

theorem min_value_a :
  ∃ (a : ℝ), a = 5/4 ∧ ∀ (x1 x2 x3 x4 : ℝ), ∃ (k1 k2 k3 k4 : ℤ),
    (∑ i j in ({1, 2, 3, 4} : Finset ℕ).sup' ⟨1, by simp⟩, (x1 - k1 - (x2 - k2))^2) ≤ a :=
begin
  sorry
end

end min_value_a_l607_607524


namespace quadratic_polynomial_correct_l607_607360

variables {a b c : ℝ}
-- Given conditions: a, b, c are distinct numbers
axiom h : a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Define p(x) and the conditions on p(x)
def p (x : ℝ) : ℝ := (a^2 + b^2 + c^2 + ab + bc + ac) * x^2 - (a + b) * (b + c) * (c + a) * x + abc * (a + b + c)

theorem quadratic_polynomial_correct (x : ℝ) : 
  p(a) = a^4 ∧ p(b) = b^4 ∧ p(c) = c^4 :=
sorry

end quadratic_polynomial_correct_l607_607360


namespace largest_consecutive_semi_primes_l607_607178

/-- 
A natural number is semi-prime if it is greater than 25 and can be expressed as 
the sum of two distinct prime numbers.
-/
def is_semi_prime (x : ℕ) : Prop :=
  x > 25 ∧ ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p ≠ q ∧ x = p + q

/--
Prove that the largest number of consecutive natural numbers 
that can be semi-prime is 5.
-/
theorem largest_consecutive_semi_primes : ∃ n, (∀ k, (k ≥ 25) → (∀ m, (m < n) → is_semi_prime (k+m))) ∧ (¬ ∃ n', n' > n ∧ ∀ k', (k' ≥ 25) → (∀ m', (m' < n'), is_semi_prime (k'+m')))
:= by
  sorry

end largest_consecutive_semi_primes_l607_607178


namespace mary_earns_per_home_l607_607181

noncomputable def earnings_per_home (T : ℕ) (n : ℕ) : ℕ := T / n

theorem mary_earns_per_home :
  ∀ (T n : ℕ), T = 276 → n = 6 → earnings_per_home T n = 46 := 
by
  intros T n h1 h2
  -- Placeholder proof step
  sorry

end mary_earns_per_home_l607_607181


namespace mildred_initial_blocks_l607_607556

theorem mildred_initial_blocks
  (final_blocks : Nat)
  (found_blocks : Nat)
  (h_final : final_blocks = 86)
  (h_found : found_blocks = 84) :
  final_blocks - found_blocks = 2 :=
by
  rw [h_final, h_found]
  rfl

end mildred_initial_blocks_l607_607556


namespace distance_between_stripes_l607_607314

def street_conditions (curb_distance angle_degrees stripe_length curb_length true_distance : ℝ) :=
  60 = curb_distance ∧ 
  30 = angle_degrees ∧ 
  65 = stripe_length ∧ 
  20 = curb_length ∧ 
  abs (true_distance - 18.462) < 0.001 

theorem distance_between_stripes (curb_distance angle_degrees stripe_length curb_length : ℝ) 
  (h : street_conditions curb_distance angle_degrees stripe_length curb_length 18.462) : 
  abs (true_distance - 18.462) < 0.001 :=
  sorry

end distance_between_stripes_l607_607314


namespace pure_imaginary_m_eq_two_l607_607911

open Complex

theorem pure_imaginary_m_eq_two (m : ℝ) : (∃ (z : ℂ), z = (2 - (m : ℂ) * Complex.i) / (1 + Complex.i) ∧ Im z ≠ 0 ∧ Re z = 0) → m = 2 :=
by
  sorry

end pure_imaginary_m_eq_two_l607_607911


namespace baron_observation_correct_l607_607933

theorem baron_observation_correct :
  ∀ (t: Fin (12 * 60 * 60)),
  (8 * 60 * 60) ≤ t ∧ t < (8 * 60 * 60) + (12 * 60 * 60) →
  ¬ ∃ d ∈ Fin (12 * 60 * 60), 
  d ≠ 0 ∧
  (t + d) % (12 * 60 * 60) = t ∧
  (t + d) % (60 * 60) = t % (60 * 60) ∧
  (t + d) % 60 = t % 60 :=
begin
  sorry
end

end baron_observation_correct_l607_607933


namespace f_monotonically_decreasing_intervals_l607_607783

def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * (cos x) ^ 2

theorem f_monotonically_decreasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, (π / 12 + real.pi * (k : ℝ)) ≤ x ∧ x ≤ (7 * π / 12 + real.pi * (k : ℝ)) →
  ∃ (I : set ℝ), (I = set.Icc (π / 12 + real.pi * (k : ℝ)) (7 * π / 12 + real.pi * (k : ℝ))) ∧ monotone_decreasing_on f I :=
sorry

end f_monotonically_decreasing_intervals_l607_607783


namespace at_least_one_5_in_three_rolls_expected_value_of_profit_l607_607660

-- Define the probability of rolling at least one 5 in three dice rolls
theorem at_least_one_5_in_three_rolls : 
    let p := 1 - (5 / 6)^3 in 
    p = 91 / 216 := 
by
  sorry

-- Define the expected value of the profit in the game
theorem expected_value_of_profit (m : ℝ) : 
    let P_xi_1 := 75 / 216 in
    let P_xi_2 := 15 / 216 in
    let P_xi_3 := 1 / 216 in
    let P_xi_neg_1 := 125 / 216 in
    let E_xi := m * (P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3 - P_xi_neg_1) in
    E_xi = - (17 / 216) * m :=
by
  sorry

end at_least_one_5_in_three_rolls_expected_value_of_profit_l607_607660


namespace ellipse_properties_l607_607793

-- Define the conditions of the problem
def center_origin (Γ : Type) := true
def focal_length (Γ : Type) (f : ℝ) := f = 2
def major_minor_axis_ratio (Γ : Type) (ratio : ℝ) := ratio = Real.sqrt 2

-- Standard equation of the ellipse
def ellipse_standard_eq (Γ : Type) (x y : ℝ) := (x^2) / 2 + y^2 = 1

-- Define the scenario for line intersection and the focal point
def line_passing_through_focus (Γ : Type) (P F : Type) := true
def intersection_points (l : Type) (A B : Type) := true

-- Define the vector dot product inequality
def vector_dot_product (PA PB : ℝ × ℝ) := (PA.1 - 2) * (PB.1 - 2) + PA.2 * PB.2 ≤ 17/2

-- Rewrite the proof problem
theorem ellipse_properties (Γ : Type) (x y : ℝ) (P F A B : Type) (λ : ℝ) 
  [center_origin Γ]
  [focal_length Γ 2]
  [major_minor_axis_ratio Γ (Real.sqrt 2)]
  [line_passing_through_focus Γ P F]
  [intersection_points F A B] :
  ellipse_standard_eq Γ x y ∧ vector_dot_product (x, y) (x, y) :=
begin
  split,
  { -- Prove standard equation of the ellipse
    sorry,
  },
  { -- Prove the vector dot product inequality
    sorry,
  }

end ellipse_properties_l607_607793


namespace polynomial_coefficients_l607_607205

noncomputable def P (n : ℕ) : ℝ[X] :=
  X^n - n * X^(n-1) + (n^2 - n : ℝ)/2 * X^(n-2) + ∑ i in range (n-3), (coeffs i) * X^i
  where coeffs := sorry -- Placeholder for the remaining coefficients

theorem polynomial_coefficients (n : ℕ) (hn : 2 < n) (hroots : ∀ z : ℂ, IsRoot (P n) z → z.im = 0) :
  ∀ i ≤ n, coeff (P n) i = (-1)^(n-i) * binom n i :=
  sorry

end polynomial_coefficients_l607_607205


namespace tan_addition_l607_607077

variable (x y : ℝ)

def tan (θ : ℝ) := Real.tan θ
def cot (θ : ℝ) := 1 / (Real.tan θ)

theorem tan_addition
  (h1 : tan x + tan y = 25)
  (h2 : cot x + cot y = 30) : tan (x + y) = 150 := by
  sorry

end tan_addition_l607_607077


namespace range_of_a_l607_607090

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l607_607090


namespace midpoint_probability_l607_607140

def T : Set (ℤ × ℤ × ℤ) := 
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2.1 ∧ p.2.1 ≤ 4 ∧ 0 ≤ p.2.2 ∧ p.2.2 ≤ 5}

noncomputable def count_valid_pairs (T : Set (ℤ × ℤ × ℤ)) : ℕ := 2880

noncomputable def total_pairs_count (T : Set (ℤ × ℤ × ℤ)) : ℕ := (∑ i in (Finset.range (4 * 5 * 6)), i) - (4 * 5 * 6)

theorem midpoint_probability : 
  ∃ p q : ℕ, p + q = 167 ∧ gcd p q = 1 ∧ (count_valid_pairs T).toRat / (total_pairs_count T).toRat = p / q := 
sorry

end midpoint_probability_l607_607140


namespace math_problem_equivalence_l607_607026

def ellipse_equation_and_pseudo_circle 
  (a b : ℝ) (h_ab : a > b) (h_bpos : b > 0)
  (c : ℝ) (h_c : c = sqrt 2) 
  (distance_to_focus : ℝ) (h_distance : distance_to_focus = sqrt 3) :
  Prop :=
  let ellipse_eq := "x^2 / 3 + y^2 = 1"
  let pseudo_circle_eq := "x^2 + y^2 = 4"
  ellipse_eq ∧ pseudo_circle_eq

def range_of_dot_product
  (A B D : ℝ × ℝ)
  (h_A : A = (2, 0))
  (on_ellipse_C : B.1^2 / 3 + B.2^2 = 1 ∧ D.1^2 / 3 + D.2^2 = 1)
  (BD_perp_x : B.2 ≠ D.2 ∧ B.1 = D.1) :
  set.Ico 0 (7 + 4 * sqrt 3) :=
sorry

theorem math_problem_equivalence :
  ∃ a b (h_ab : a > b) (h_bpos : b > 0)
    c (h_c : c = sqrt 2)
    distance_to_focus (h_distance : distance_to_focus = sqrt 3),
  ellipse_equation_and_pseudo_circle a b h_ab h_bpos c h_c distance_to_focus h_distance ∧
  ∃ (A B D : ℝ × ℝ)
    (h_A: A = (2, 0))
    (on_ellipse_C : B.1^2 / 3 + B.2^2 = 1 ∧ D.1^2 / 3 + D.2^2 = 1)
    (BD_perp_x: B.2 ≠ D.2 ∧ B.1 = D.1),
  range_of_dot_product A B D h_A on_ellipse_C BD_perp_x
:=
sorry

end math_problem_equivalence_l607_607026


namespace sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l607_607585

theorem sqrt_of_16_eq_4 : Real.sqrt 16 = 4 := 
by sorry

theorem sqrt_of_364_eq_pm19 : Real.sqrt 364 = 19 ∨ Real.sqrt 364 = -19 := 
by sorry

theorem opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2 : -(2 - Real.sqrt 6) = Real.sqrt 6 - 2 := 
by sorry

end sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l607_607585


namespace find_x_l607_607984

-- Defining the elements of the problem
def area_of_smaller_square (x : ℝ) := (3 * x) ^ 2
def area_of_larger_square (x : ℝ) := (7 * x) ^ 2
def area_of_triangle (x : ℝ) := 1 / 2 * (3 * x) * (7 * x)
def total_area (x : ℝ) := area_of_smaller_square x + area_of_larger_square x + area_of_triangle x

-- Stating the theorem that x = sqrt(4400 / 137) is the solution
theorem find_x (x : ℝ) (h : total_area x = 2200) : x = Real.sqrt (4400 / 137) :=
sorry

end find_x_l607_607984


namespace ratio_of_boys_to_girls_l607_607103

theorem ratio_of_boys_to_girls (total_students : ℕ) (girls : ℕ) (boys : ℕ)
  (h_total : total_students = 1040)
  (h_girls : girls = 400)
  (h_boys : boys = total_students - girls) :
  (boys / Nat.gcd boys girls = 8) ∧ (girls / Nat.gcd boys girls = 5) :=
sorry

end ratio_of_boys_to_girls_l607_607103


namespace binomial_seven_four_l607_607722

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607722


namespace arrangement_of_programs_l607_607239

-- Define the set of programs and the condition that A and B must be consecutive
constant Programs : Type
constants A B C D E : Programs
constants rest : Finset Programs
noncomputable def AB_unit : Finset (Finset Programs) := {{A, B}, {B, A}}
noncomputable def total_units := ({C, D, E} : Finset Programs) ∪ subset_union (AB_unit)

-- Problem statement: Prove that the number of ways to arrange the programs such that A and B are consecutive is 48
theorem arrangement_of_programs : (total_units.card * 2) = 48 := sorry

end arrangement_of_programs_l607_607239


namespace intersection_of_M_and_N_l607_607176

-- Definitions of sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℕ := {0, 1, 2}

-- Proof Goal: M ∩ N = {0, 1}
theorem intersection_of_M_and_N : (M ∩ N) = {0, 1} := by
  sorry

end intersection_of_M_and_N_l607_607176


namespace expression_equals_log_expression_l607_607638

/-- Proof problem 1: Simplified expression equals 13/2 --/
theorem expression_equals :
  (0.25 : ℝ)^0 - (1 / 16) - 0.75 + 4 * (1 - real.sqrt 2)^4 + real.sqrt (6 - 4 * real.sqrt 2) + real.log (real.sqrt real.exp) + 2^(2 + real.log 2 3) = 13 / 2 :=
by
  sorry

/-- Proof problem 2: Logarithm expression in terms of a and b --/
theorem log_expression (a b : ℝ) (ha : real.log 14 6 = a) (hb : real.log 14 7 = b) :
  real.log 42 56 = (3 - 2 * b) / (a + b) :=
by
  sorry

end expression_equals_log_expression_l607_607638


namespace part1_part2_l607_607060

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - 1| + |2 * x - a|

theorem part1 (x : ℝ) : (f x 2 < 2) ↔ (1/4 < x ∧ x < 5/4) := by
  sorry
  
theorem part2 (a : ℝ) (hx : ∀ x : ℝ, f x a ≥ 3 * a + 2) :
  (-3/2 ≤ a ∧ a ≤ -1/4) := by
  sorry

end part1_part2_l607_607060


namespace total_handshakes_l607_607339

theorem total_handshakes (team_size : ℕ) (num_referees : ℕ) :
  team_size = 6 → num_referees = 3 →
  (team_size * team_size) + (team_size * (team_size - 1) / 2 * 2) + ((team_size * 2) * num_referees) = 102 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_handshakes_l607_607339


namespace toys_per_rabbit_l607_607894

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end toys_per_rabbit_l607_607894


namespace isochoric_heating_ratio_l607_607476

theorem isochoric_heating_ratio 
  (T_max : ℝ) (T_min : ℝ) (η : ℝ) 
  (h1 : T_max = 900) (h2 : T_min = 350) (h3 : η = 0.4) 
  : (T_max / T_min) * (1 - η) ≈ 1.54 :=
by
  have k := (T_max / T_min) * (1 - η)
  have h4 : k = 1.5428571428571428
  sorry

end isochoric_heating_ratio_l607_607476


namespace calc_problem_system_of_equations_l607_607634

-- For calculation problem
theorem calc_problem : 
  sqrt 12 + (pi - 203) ^ 0 + (1 / 2: ℝ) ^ (-1) - 6 * tan (real.pi / 6) = 3 := by sorry

-- For the system of equations problem
theorem system_of_equations :
  ∃ (x y : ℝ), 
  (x + 2 * y = 4) ∧ (x + 3 * y = 5) ∧ (x = 2) ∧ (y = 1) := by sorry

end calc_problem_system_of_equations_l607_607634


namespace max_acute_triangles_correct_l607_607771

noncomputable def max_acute_triangles (P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 ∧ ∀ p1 p2 p3 ∈ P, LinearIndependent ℝ ![p1, p2, p3] 
  then sorry 
  else 0

theorem max_acute_triangles_correct (P : Finset (ℝ × ℝ)) :
  (∃ p1 p2 p3 p4 p5, 
    {p1, p2, p3, p4, p5} = P ∧
    ∀ p1 p2 p3 ∈ P, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬Collinear ℝ ![p1, p2, p3]) →
  max_acute_triangles P = 7 :=
begin
  sorry
end

end max_acute_triangles_correct_l607_607771


namespace mary_money_left_l607_607536

def initial_amount : Float := 150
def game_cost : Float := 60
def discount_percent : Float := 15 / 100
def remaining_percent_for_goggles : Float := 20 / 100
def tax_on_goggles : Float := 8 / 100

def money_left_after_shopping_trip (initial_amount : Float) (game_cost : Float) (discount_percent : Float) (remaining_percent_for_goggles : Float) (tax_on_goggles : Float) : Float :=
  let discount := game_cost * discount_percent
  let discounted_price := game_cost - discount
  let remainder_after_game := initial_amount - discounted_price
  let goggles_cost_before_tax := remainder_after_game * remaining_percent_for_goggles
  let tax := goggles_cost_before_tax * tax_on_goggles
  let final_goggles_cost := goggles_cost_before_tax + tax
  let remainder_after_goggles := remainder_after_game - final_goggles_cost
  remainder_after_goggles

#eval money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles -- expected: 77.62

theorem mary_money_left (initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles : Float) : 
  money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles = 77.62 :=
by sorry

end mary_money_left_l607_607536


namespace correct_propositions_l607_607810

def Z : ℂ := 2 / (-1 + complex.i)

def p1 : Prop := |Z| = 2
def p2 : Prop := Z ^ 2 = 2 * complex.i
def p3 : Prop := complex.conj Z = 1 + complex.i
def p4 : Prop := Z.im = -1

theorem correct_propositions : ¬p1 ∧ p2 ∧ ¬p3 ∧ p4 :=
  by sorry

end correct_propositions_l607_607810


namespace unit_circles_cover_parallelogram_l607_607902

variables {α : ℝ} {a : ℝ}

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := 
  (B.1 - A.1 = D.1 - C.1) ∧ (B.2 - A.2 = D.2 - C.2) ∧ (C.1 - B.1 = A.1 - D.1) ∧ (C.2 - B.2 = A.2 - D.2)

def is_acute_triangle (A B D : ℝ × ℝ) : Prop := 
  ∠ A B D < π / 2 ∧ ∠ B D A < π / 2 ∧ ∠ D A B < π / 2

theorem unit_circles_cover_parallelogram (A B C D : ℝ × ℝ) (h_parallelogram : is_parallelogram A B C D) 
    (h_AB_length : |B.1 - A.1| = a) 
    (h_AD_length : |D.1 - A.1| = 1) 
    (h_angle_DAB : ∠ D A B = α) 
    (h_acute_triangle : is_acute_triangle A B D) : 
  a ≤ cos α + sqrt 3 * sin α :=
sorry

end unit_circles_cover_parallelogram_l607_607902


namespace z_sixth_power_l607_607047

variables (a : ℝ) (z : ℂ)
def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = (0 : ℝ) + (b : ℝ) * complex.I

theorem z_sixth_power (ha : z = a + (a + 1) * complex.I) (hpure : is_pure_imaginary z) : z^6 = -1 := 
  by sorry

end z_sixth_power_l607_607047


namespace second_expression_l607_607976

variable (a b : ℕ)

theorem second_expression (h : 89 = ((2 * a + 16) + b) / 2) (ha : a = 34) : b = 94 :=
by
  sorry

end second_expression_l607_607976


namespace factorize_expr_l607_607745

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l607_607745


namespace family_age_problem_l607_607965

theorem family_age_problem (T y : ℕ)
  (h1 : T = 5 * 17)
  (h2 : (T + 5 * y + 2) = 6 * 17)
  : y = 3 := by
  sorry

end family_age_problem_l607_607965


namespace polygon_sides_of_interior_angle_l607_607460

theorem polygon_sides_of_interior_angle (n : ℕ) (h : ∀ i : Fin n, (∃ (x : ℝ), x = (180 - 144) / 1) → (360 / (180 - 144)) = n) : n = 10 :=
sorry

end polygon_sides_of_interior_angle_l607_607460


namespace pow_comparison_l607_607260

theorem pow_comparison : 2^700 > 5^300 :=
by sorry

end pow_comparison_l607_607260


namespace geom_seq_formula_abs_seq_formula_l607_607486

noncomputable def geom_seq (n : ℕ) := 2 ^ (n - 1)

theorem geom_seq_formula :
  ∀ (a1 a4 : ℕ), a1 = 1 → a4 = 8 → geom_seq 1 = 1 ∧ geom_seq 4 = 8 :=
by
  intros a1 a4 h1 h4
  rw [h1, h4]
  unfold geom_seq
  split
  · simp
  · simp

def arith_seq (n : ℕ) := 6 * n - 32

def abs_sum_b_n (n : ℕ) : ℤ :=
if n ≤ 5 then -3 * (n * n) + 29 * n
else 3 * (n * n) - 29 * n + 140

theorem abs_seq_formula (n : ℕ) :
  ∀ (a3 a5 : ℕ), a3 = 4 → a5 = 16 → abs_sum_b_n n =
    if n ≤ 5 then -3 * (n * n) + 29 * n
    else 3 * (n * n) - 29 * n + 140 :=
by
  intros a3 a5 h3 h5
  unfold abs_sum_b_n
  split_ifs
  · reflexivity
  · reflexivity

end geom_seq_formula_abs_seq_formula_l607_607486


namespace car_distance_l607_607285

/-- A car takes 4 hours to cover a certain distance. We are given that the car should maintain a speed of 90 kmph to cover the same distance in (3/2) of the previous time (which is 6 hours). We need to prove that the distance the car needs to cover is 540 km. -/
theorem car_distance (time_initial : ℝ) (speed : ℝ) (time_new : ℝ) (distance : ℝ) 
  (h1 : time_initial = 4) 
  (h2 : speed = 90)
  (h3 : time_new = (3/2) * time_initial)
  (h4 : distance = speed * time_new) : 
  distance = 540 := 
sorry

end car_distance_l607_607285


namespace intersection_is_correct_l607_607523

noncomputable def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x < 4}

theorem intersection_is_correct : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end intersection_is_correct_l607_607523


namespace domain_g_l607_607206

def f (x : ℝ) : ℝ := sorry -- assume some definition for f

def domain_f := set.Icc (-8 : ℝ) (4 : ℝ) -- domain of f is [-8, 4]

def g (x : ℝ) : ℝ := f (-3 * x)

theorem domain_g : set.Icc (-4 / 3 : ℝ) (8 / 3 : ℝ) = {x : ℝ | -8 ≤ -3 * x ∧ -3 * x ≤ 4} :=
by
suffices h : set.Icc (-4 / 3 : ℝ) (8 / 3 : ℝ) = {x : ℝ | -3 * x ∈ set.Icc (-8 : ℝ) (4 : ℝ)}, from h,
calc
set.Icc (-4 / 3 : ℝ) (8 / 3 : ℝ)
= {x : ℝ | -4 / 3 ≤ x ∧ x ≤ 8 / 3} : sorry
... = {x : ℝ | -8 / -3 ≤ x ∧ x ≤ 4 / -3} : by norm_num
... = {x : ℝ | 8 / 3 ≥ x ∧ x ≥ -4 / 3} : by norm_num
... = {x : ℝ | -8 ≤ -3 * x ∧ -3 * x ≤ 4} : sorry

end domain_g_l607_607206


namespace man_buys_article_for_20_l607_607301

variable (SP : ℝ) (G : ℝ) (CP : ℝ)

theorem man_buys_article_for_20 (hSP : SP = 25) (hG : G = 0.25) (hEquation : SP = CP * (1 + G)) : CP = 20 :=
by
  sorry

end man_buys_article_for_20_l607_607301


namespace slope_of_line_l607_607232

theorem slope_of_line (m : ℤ) (hm : (3 * m - 6) / (1 + m) = 12) : m = -2 := 
sorry

end slope_of_line_l607_607232


namespace monochromatic_triangles_min_n_l607_607784

theorem monochromatic_triangles_min_n (n : ℕ)
  (h : ∀ (c : Fin (n * (n - 1) / 2) → bool),
    ∃ (T : Fin (n.choose 3) → Fin 3 → Fin n),
      ∀ (i : Fin (n.choose 3)), 
        (T i).1 ≠ (T i).2 ∧ (T i).1 ≠ (T i).3 ∧ (T i).2 ≠ (T i).3 ∧ 
        (c (T i).1) = c (T i).2 ∧ (c (T i).1) = c (T i).3 ∧ 
        (12 ≤ Finset.count (λ x, c (T x).1 = c (T x).2 ∧ c (T x).1 = c (T x).3) 
           (Finset.univ : Finset (Fin (n.choose 3))))) :
  n ≥ 9 := sorry

end monochromatic_triangles_min_n_l607_607784


namespace radius_of_circle_zero_l607_607759

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end radius_of_circle_zero_l607_607759


namespace expected_accidents_no_overtime_l607_607623

noncomputable def accidents_with_no_overtime_hours 
    (hours1 hours2 : ℕ) (accidents1 accidents2 : ℕ) : ℕ :=
  let slope := (accidents2 - accidents1) / (hours2 - hours1)
  let intercept := accidents1 - slope * hours1
  intercept

theorem expected_accidents_no_overtime : 
    accidents_with_no_overtime_hours 1000 400 8 5 = 3 :=
by
  sorry

end expected_accidents_no_overtime_l607_607623


namespace volume_correct_l607_607270

-- Define the structure and conditions
structure Point where
  x : ℝ
  y : ℝ

def is_on_circle (C : Point) (P : Point) : Prop :=
  (P.x - C.x)^2 + (P.y - C.y)^2 = 25

def volume_of_solid_of_revolution (P A B : Point) : ℝ := sorry

noncomputable def main : ℝ :=
  volume_of_solid_of_revolution {x := 2, y := -8} {x := 4.58, y := -1.98} {x := -3.14, y := -3.91}

theorem volume_correct :
  main = 672.1 := by
  -- Proof skipped
  sorry

end volume_correct_l607_607270


namespace hare_wins_l607_607866

def hare_wins_race : Prop :=
  let hare_speed := 10
  let hare_run_time := 30
  let hare_nap_time := 30
  let tortoise_speed := 4
  let tortoise_delay := 10
  let total_race_time := 60
  let hare_distance := hare_speed * hare_run_time
  let tortoise_total_time := total_race_time - tortoise_delay
  let tortoise_distance := tortoise_speed * tortoise_total_time
  hare_distance > tortoise_distance

theorem hare_wins : hare_wins_race := by
  -- Proof here
  sorry

end hare_wins_l607_607866


namespace binomial_7_4_eq_35_l607_607707
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607707


namespace probability_intersection_of_diagonals_hendecagon_l607_607274

-- Definition statements expressing the given conditions and required probability

def total_diagonals (n : ℕ) : ℕ := (Nat.choose n 2) - n

def ways_to_choose_2_diagonals (n : ℕ) : ℕ := Nat.choose (total_diagonals n) 2

def ways_sets_of_intersecting_diagonals (n : ℕ) : ℕ := Nat.choose n 4

def probability_intersection_lies_inside (n : ℕ) : ℚ :=
  ways_sets_of_intersecting_diagonals n / ways_to_choose_2_diagonals n

theorem probability_intersection_of_diagonals_hendecagon :
  probability_intersection_lies_inside 11 = 165 / 473 := 
by
  sorry

end probability_intersection_of_diagonals_hendecagon_l607_607274


namespace math_proof_l607_607052

variable {a : ℝ} (h1 : a > 1)
def f (x : ℝ) : ℝ := Real.logBase a (|x|)

theorem math_proof :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f h1 x < f h1 y) →
  (∀ x : ℝ, f h1 x = f h1 (-x)) →
  f h1 1 < f h1 (-2) ∧ f h1 (-2) < f h1 3 :=
by
  intros h_mono h_even
  -- Proof is omitted
  sorry

end math_proof_l607_607052


namespace geometry_problem_l607_607493
noncomputable theory

variables {A B C K A1 B1 C1 : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space K] [metric_space A1] [metric_space B1] [metric_space C1]
variables [preorder A] [preorder B] [preorder C] [preorder K] [preorder A1] [preorder B1] [preorder C1]

-- Given conditions
def triangle (A B C : Type) : Prop := 
  ∃ (T : Type), is_triangle T ∧ (A ∈ T) ∧ (B ∈ T) ∧ (C ∈ T)

def point_in_triangle (K : Type) (A B C : Type) : Prop :=
  ∀ (T : Type), is_triangle T → (A ∈ T) ∧ (B ∈ T) ∧ (C ∈ T) → (K ∈ interior T)

def intersection_point (p q r : Type) (K : Type) : Type := 
  some point_at_intersection : Type where
  p q r intersection point

-- Mathematical statement to be proved
theorem geometry_problem 
  (h_triangle : triangle A B C)
  (h_point : point_in_triangle K A B C)
  (h_A1 : intersection_point A B C K = A1)
  (h_B1 : intersection_point B A C K = B1)
  (h_C1 : intersection_point C A B K = C1) :
  (dist A K / dist K A1 = dist A B1 / dist B1 C + dist A C1 / dist C1 B) :=
sorry

end geometry_problem_l607_607493


namespace solution_set_log_inequality_l607_607038

open Real

theorem solution_set_log_inequality (b : ℝ) (x : ℝ) 
  (h1 : 0 < b ∧ b < 1)
  (h2 : ∀ x ∈ Icc 0 1, log b (2 - b * x) <= log b (2 - b * (x + 1))) :
  (log b (abs (x + 2)) > log b (abs (x - 4))) ↔ (x < 1 ∧ x ≠ -2) :=
by
  sorry

end solution_set_log_inequality_l607_607038


namespace two_digit_number_l607_607680

noncomputable theory

def is_valid_number(a b: ℕ) : bool :=
  a ≠ 0 ∧ b < 10 ∧ 2.6 < (10 * a + b : ℝ) / (a + b : ℝ) ∧ (10 * a + b : ℝ) / (a + b : ℝ) < 2.7

theorem two_digit_number: ∃ (a b : ℕ), is_valid_number a b ∧ 10 * a + b = 29 :=
by {
  existsi 2,
  existsi 9,
  simp [is_valid_number],
  norm_num,
  split,
  exact dec_trivial,
  split,
  exact dec_trivial,
  split,
  { norm_num1, linarith, },
  { norm_num1, linarith, }
}

end two_digit_number_l607_607680


namespace length_of_crease_l607_607671

theorem length_of_crease (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) : 
  ∃ d : ℝ, d = 5 :=
by
  -- Definitions of the sides of the triangle
  let A := (0 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 5)
  let C := (12, 0)
  
  -- Midpoint of AC
  let D := ((0 + 12) / 2, (0 + 0) / 2) -- (6, 0)
  
  -- Coordinates of E as intersection of perpendicular bisector with BC
  let E := (6, 5)
  
  -- Distance DE
  let DE := Real.sqrt((6 - 6)^2 + (5 - 0)^2)
  have h5 : DE = 5 := sorry
  
  -- The length of the crease is 5
  exact ⟨DE, h5⟩

end length_of_crease_l607_607671


namespace malou_average_score_l607_607528

def quiz1_score := 91
def quiz2_score := 90
def quiz3_score := 92

def sum_of_scores := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes := 3

theorem malou_average_score : sum_of_scores / number_of_quizzes = 91 :=
by
  sorry

end malou_average_score_l607_607528


namespace BC_length_correct_l607_607601

noncomputable def length_of_BC {A B C : Type} (radius_A radius_B : ℝ) (AB_dist : ℝ) (tangent_point_C : ℝ) 
  (externally_tangent : Boolean) (tangent_intersect : Boolean) : ℝ :=
if externally_tangent ∧ tangent_intersect ∧ radius_A = 7 ∧ radius_B = 4 ∧ AB_dist = (radius_A + radius_B) 
then (4 * 11) / 3 else 0

theorem BC_length_correct : length_of_BC 7 4 11 (44 / 3) true true = 44 / 3 :=
by {
  sorry
}

end BC_length_correct_l607_607601


namespace circle_symmetric_equation_line_through_points_l607_607065

/-- Given that circle C2: (x + 1)^2 + (y + 2)^2 = 4 and circle C1 is symmetric to C2 
with respect to the line x - y + 1 = 0, prove that the equation of circle C1 is (x + 3)^2 + y^2 = 4. -/
theorem circle_symmetric_equation :
  (∀ x y: ℝ, (x + 1)^2 + (y + 2)^2 = 4 ↔ C2 (x, y)) →
  (∀ x y: ℝ, C1 (x, y)) →
  ∀ x y: ℝ, (x + 3)^2 + y^2 = 4 :=
begin
  sorry
end

/-- Given the line passing through the point A(0, 3) intersects circle C1 at points M and N,
prove that the equation of the line passing through these points is y = 2x + 3 or y = 3x + 3
when OM • ON = 7 / 5. -/
theorem line_through_points :
  ∀ x y: ℝ, (x + 3)^2 + y^2 = 4 →
  (M, N : ℝ × ℝ) →
  let O := (0, 0) in
  (OM • ON = 7 / 5) →
  (line : ℝ → ℝ) →
  ((line (0, 3)) = 2 * x + 3 ∨ (line (0, 3)) = 3 * x + 3) :=
begin
  sorry
end

end circle_symmetric_equation_line_through_points_l607_607065


namespace problem1_problem2_l607_607891

-- Define that a quadratic is a root-multiplying equation if one root is twice the other
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 ≠ 0 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)

-- Problem 1: Prove that x^2 - 3x + 2 = 0 is a root-multiplying equation
theorem problem1 : is_root_multiplying 1 (-3) 2 :=
  sorry

-- Problem 2: Given ax^2 + bx - 6 = 0 is a root-multiplying equation with one root being 2, determine a and b
theorem problem2 (a b : ℝ) : is_root_multiplying a b (-6) → (∃ x1 x2 : ℝ, x1 = 2 ∧ x1 ≠ 0 ∧ a * x1^2 + b * x1 - 6 = 0 ∧ a * x2^2 + b * x2 - 6 = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)) →
( (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
  sorry

end problem1_problem2_l607_607891


namespace computer_price_decrease_l607_607615

theorem computer_price_decrease 
  (initial_price : ℕ) 
  (decrease_factor : ℚ)
  (years : ℕ) 
  (final_price : ℕ) 
  (h1 : initial_price = 8100)
  (h2 : decrease_factor = 1/3)
  (h3 : years = 6)
  (h4 : final_price = 2400) : 
  initial_price * (1 - decrease_factor) ^ (years / 2) = final_price :=
by
  sorry

end computer_price_decrease_l607_607615


namespace infinite_t_exists_l607_607621

theorem infinite_t_exists (t : ℕ) : (∀ t : ℕ, ∃ (a b : ℕ), 2012 * t + 1 = a ^ 2 ∧ 2013 * t + 1 = b ^ 2) → ∞ := 
sorry

end infinite_t_exists_l607_607621


namespace secant_slope_tangent_eq_l607_607043

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the secant slope between points (1, f 1) and (4, f 4)
def slope_secant : ℝ := (f 4 - f 1) / (4 - 1)

-- Prove that the slope at the given points is 3
theorem secant_slope : slope_secant = 3 := by
  unfold slope_secant
  unfold f
  calc 
    (5 - (-4)) / 3 = 9 / 3 := by norm_num
    ... = 3 := by norm_num

-- Find the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- Prove that the tangent line at x = 1 is y + 4 = 0
theorem tangent_eq : ∃ m b, f' 1 = m ∧ (f 1 - m * 1) = b ∧ (y + 4 = 0) := by
  unfold f'
  unfold f
  calc 
    deriv f 1 = 2 * 1 - 2 := by norm_num
  use 0 -- slope is zero
  use (-4) -- intercept comes from the point P (1, -4)
  split
  . norm_num
  . split
    . norm_num
    . norm_num

end secant_slope_tangent_eq_l607_607043


namespace distinct_solutions_difference_l607_607149

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l607_607149


namespace solve_equation1_solve_equation2_l607_607560

theorem solve_equation1 (x : ℝ) (h1 : 3 * x^3 - 15 = 9) : x = 2 :=
sorry

theorem solve_equation2 (x : ℝ) (h2 : 2 * (x - 1)^2 = 72) : x = 7 ∨ x = -5 :=
sorry

end solve_equation1_solve_equation2_l607_607560


namespace find_range_for_a_l607_607921

noncomputable def f : ℝ → ℝ := sorry  -- The function f is given but not defined here

theorem find_range_for_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Ioi (-2) → x2 ∈ Ioi (-2) → x1 < x2 → f x1 < f x2) → a > 1 / 2 :=
begin
  intro h,
  -- Proof is skipped with sorry
  sorry
end

end find_range_for_a_l607_607921


namespace cynthia_more_miles_l607_607854

open Real

noncomputable def david_speed : ℝ := 55 / 5
noncomputable def cynthia_speed : ℝ := david_speed + 3

theorem cynthia_more_miles (t : ℝ) (ht : t = 5) :
  (cynthia_speed * t) - (david_speed * t) = 15 :=
by
  sorry

end cynthia_more_miles_l607_607854


namespace length_AB_and_h_other_intersection_and_h_range_area_triangle_ABC_value_a_constant_sum_l607_607119

-- Definitions and given conditions
variable {a h k : ℝ}
variable {m n : ℝ}
variable (A B C D P Q : ℝ × ℝ)
variable (x1 x2 x3 y1 y2 y3 : ℝ)

-- Conditions based on the problem
def quadratic_form (x : ℝ) := a*(x-h)^2 + k
def point_A := (-3, m)
def point_B := (-1, n)
def point_C := (1, 0)
def A_and_B_on_parabola := quadratic_form (-3) = m ∧ quadratic_form (-1) = n
def C_on_parabola := quadratic_form 1 = 0
def m_lt_0_lt_n := m < 0 ∧ n > 0

-- Lean Proof Statements
theorem length_AB_and_h (ha : a < 0) (hmn : m = n) (hab : A_and_B_on_parabola) :
  (abs ((-3) - (-1)) = 2) ∧ (h = -2) := by {
  sorry
}

theorem other_intersection_and_h_range 
  (ha : a < 0) (h_mlt0ltn : m_lt_0_lt_n) (hC : C_on_parabola) :
  (∃ x, quadratic_form x = 0 ∧ x ≠ 1) ∧ (-1 < h ∧ h < 0) := by {
  sorry
}

theorem area_triangle_ABC (ha : a = -1) (h_mlt0ltn : m_lt_0_lt_n) (hC : C_on_parabola) :
  (area_of_triangle (-3, m) (-1, n) (1, 0) = 8) := by {
  sorry
}

theorem value_a_constant_sum
  (ha : a = -1/4) (hC : C_on_parabola) (hD : D = (0, h^2)) :
  (∃ a, (x1 + x2 - x3) = -1) := by {
  sorry
}

end length_AB_and_h_other_intersection_and_h_range_area_triangle_ABC_value_a_constant_sum_l607_607119


namespace triangle_ABC_angles_l607_607395

theorem triangle_ABC_angles (A B C M : Type)
  [triangle_ABC : triangle A B C]
  (h1 : angle C = 90)
  (h2 : median CM from C to M)
  (h3 : midpoint M on hypotenuse AB)
  (h4 : incircle_ACM_touches_midpoint_CM)
  : angles_of_triangle_ABC = (30, 60, 90) := sorry

end triangle_ABC_angles_l607_607395


namespace possible_remainder_degrees_l607_607258

theorem possible_remainder_degrees (f g : Polynomial ℝ) (h : g = 2 * X^7 - 5 * X^3 + 4 * X^2 - 9) :
  ∃ (r : Polynomial ℝ), r.degree < g.degree →
  r.degree ∈ {0, 1, 2, 3, 4, 5, 6} := by
  sorry

end possible_remainder_degrees_l607_607258


namespace Mina_digits_l607_607201

theorem Mina_digits (Carlos Sam Mina : ℕ) 
  (h1 : Sam = Carlos + 6) 
  (h2 : Mina = 6 * Carlos) 
  (h3 : Sam = 10) : 
  Mina = 24 := 
sorry

end Mina_digits_l607_607201


namespace binomial_7_4_eq_35_l607_607710
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607710


namespace S₄_is_28_l607_607860

noncomputable def geometric_sequence_sum {q : ℝ} (a₁ : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

variables {a₁ q : ℝ} (hq : q ≠ 1) (hq2 : q^2 = 3)

-- S_2 and S_6 definitions based on given conditions
def S₂ := geometric_sequence_sum a₁ 2
def S₆ := geometric_sequence_sum a₁ 6

axiom hS₂ : S₂ = 7
axiom hS₆ : S₆ = 91

-- Theorem: S_4 = 28
theorem S₄_is_28 : geometric_sequence_sum a₁ 4 = 28 :=
sorry

end S₄_is_28_l607_607860


namespace evaluate_g_h_2_l607_607078

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l607_607078


namespace constant_exists_l607_607024

theorem constant_exists
  (x : ℕ → ℝ)
  (h_pos : ∀ n, x n > 0)
  (h_S : ∀ n ≥ 2, (∑ i in finset.range (n+1), x i) ≥ 2 * (∑ i in finset.range n, x i)) :
  ∃ C > 0, ∀ n, x n ≥ C * 2^n :=
sorry

end constant_exists_l607_607024


namespace cow_total_spots_l607_607946

theorem cow_total_spots : 
  let left_spots := 16 in 
  let right_spots := 3 * left_spots + 7 in
  left_spots + right_spots = 71 :=
by
  let left_spots := 16
  let right_spots := 3 * left_spots + 7
  show left_spots + right_spots = 71
  sorry

end cow_total_spots_l607_607946


namespace inequality_proof_l607_607415

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 1)
noncomputable def g (x : ℝ) : ℝ := real.sin x

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > b) 
  (h2 : b ≥ 1)
  (h3 : c > d)
  (h4 : d > 0)
  (h5 : f a - f b = real.pi)
  (h6 : g c - g d = real.pi / 10) :
  a + d - b - c < 9 * real.pi / 10 :=
by {
  sorry
}

end inequality_proof_l607_607415


namespace third_consecutive_odd_integer_l607_607275

theorem third_consecutive_odd_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : x + 4 = 15 :=
sorry

end third_consecutive_odd_integer_l607_607275


namespace determine_coordinates_of_a_l607_607066

theorem determine_coordinates_of_a
  (a b : ℝ × ℝ)
  (ha : ∥a∥ = 3)
  (hb : b = (1, 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 0) :
  (a = (-6 * real.sqrt 5 / 5, 3 * real.sqrt 5 / 5) ∨ 
   a = (6 * real.sqrt 5 / 5, -3 * real.sqrt 5 / 5)) :=
by
  sorry

end determine_coordinates_of_a_l607_607066


namespace john_initial_amount_l607_607130

theorem john_initial_amount (num_barbells : ℕ) (cost_per_barbell : ℕ) (change_received : ℕ) :
  num_barbells = 3 →
  cost_per_barbell = 270 →
  change_received = 40 →
  (num_barbells * cost_per_barbell + change_received) = 850 :=
by
  intros h_barbells h_cost h_change
  rw [h_barbells, h_cost, h_change]
  sorry

end john_initial_amount_l607_607130


namespace sticker_distribution_l607_607069

theorem sticker_distribution (sheets stickers : ℕ) (at_least_one_empty : sheets ≥ 1) : 
  (sheets = 5) ∧ (stickers = 10) → 
  ∃ (n : ℕ), (n = 23) ∧ 
  -- The methodology to count valid ways to distribute stickers ensuring at least one sheet remains empty
  sorry

end sticker_distribution_l607_607069


namespace log_n_ge_k_log_2_l607_607193

-- Definition of distinct prime factors and base-10 logarithm
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (nat.factors n).erase_dup.length

theorem log_n_ge_k_log_2 (n : ℕ) (hn : n > 0) :
  let k := num_distinct_prime_factors n in
  log (n : ℝ) ≥ k * log (2 : ℝ) := 
by
  sorry

end log_n_ge_k_log_2_l607_607193


namespace chord_ratio_l607_607244

theorem chord_ratio {FQ HQ : ℝ} (h : EQ * FQ = GQ * HQ) (h_eq : EQ = 5) (h_gq : GQ = 12) : 
  FQ / HQ = 12 / 5 :=
by
  rw [h_eq, h_gq] at h
  sorry

end chord_ratio_l607_607244


namespace arithmetic_sequence_seventh_term_l607_607236

theorem arithmetic_sequence_seventh_term (a d : ℝ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 14) 
  (h2 : a + 4 * d = 9) : 
  a + 6 * d = 13.4 := 
sorry

end arithmetic_sequence_seventh_term_l607_607236


namespace function_domain_is_minus_one_to_one_l607_607211

def domain_of_function : Set ℝ :=
  {x : ℝ | x + 1 > 0 ∧ -x^2 - 3x + 4 > 0}

theorem function_domain_is_minus_one_to_one :
  domain_of_function = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end function_domain_is_minus_one_to_one_l607_607211


namespace solve_exponential_equation_l607_607957

theorem solve_exponential_equation (x : ℝ) (h : (16 ^ x) * (16 ^ x) * (16 ^ x) * (16 ^ x) = 256 ^ 4) : 
  x = 2 := by 
sorry

end solve_exponential_equation_l607_607957


namespace sum_of_p_squares_l607_607506

theorem sum_of_p_squares (p n : ℕ) (hp : p.prime) (hn : 0 < n) (hsq : ∃ a : ℕ, 1 + n * p = a^2) :
  ∃ (s : Fin p → ℕ), (n + 1) = (∑ i, s i)^2 :=
begin
  sorry
end

end sum_of_p_squares_l607_607506


namespace circle_property_l607_607871

noncomputable theory

variables {A B C M K O : Type*}
variables (triangle : Type*) [acute_angles : Π (A B C : triangle), angle A + angle B + angle C = 180]

def is_orthocenter (M : triangle) : Prop := sorry
def is_incenter (K : triangle) : Prop := sorry
def is_circumcenter (O : triangle) : Prop := sorry

def passes_through (circle : set (point)) (p : point) : Prop := sorry

theorem circle_property (h_acute : acute_angles A B C)
  (h_diff_sides : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_orthocenter : is_orthocenter M)
  (h_incenter : is_incenter K)
  (h_circumcenter : is_circumcenter O)
  (h_circle_pass : ∀ {P : point}, P = A ∨ P = B ∨ P = C → passes_through K P ∧ passes_through O P ∧ passes_through M P) :
  ∃ Q : triangle, (Q ≠ A ∧ Q ≠ B ∧ Q ≠ C) ∧ passes_through K Q ∧ passes_through O Q ∧ passes_through M Q :=
sorry

end circle_property_l607_607871


namespace identical_lines_pairs_l607_607912

theorem identical_lines_pairs :
  let m := {a : ℝ // ∃ (d : ℝ), d = 3 * a ∧ (∀ x y, 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0)} 
  in m.to_finset.card = 2 :=
by
  sorry

end identical_lines_pairs_l607_607912


namespace greatest_possible_y_l607_607562

theorem greatest_possible_y (x y : ℤ) (h : 2 * x * y + 8 * x + 2 * y = -14) : y ≤ 5 :=
begin
  sorry
end

noncomputable def greatest_y (x y : ℤ) (h : 2 * x * y + 8 * x + 2 * y = -14) : ℤ :=
if y' ≤ 5 then y' else y sorry

example : ∃ x y : ℤ, 2 * x * y + 8 * x + 2 * y = -14 ∧ greatest_y x y = 5 :=
begin
  use [0, 5], -- example values satisfying the condition
  split,
  { norm_num }, -- proof that 2 * 0 * 5 + 8 * 0 + 2 * 5 = -14
  { norm_num }  -- confirming the greatest possible value of y is 5
end

end greatest_possible_y_l607_607562


namespace min_value_expr_l607_607190

theorem min_value_expr (a b : ℕ) (ha : a > 0) (hb : b > 0) (ha7 : a < 8) (hb7 : b < 8) :
  3 * a - 2 * a * b ≥ -77 :=
by
  sorry

end min_value_expr_l607_607190


namespace first_number_less_than_twice_second_l607_607586

theorem first_number_less_than_twice_second (x y z : ℕ) : 
  x + y = 50 ∧ y = 19 ∧ x = 2 * y - z → z = 7 :=
by sorry

end first_number_less_than_twice_second_l607_607586


namespace norm_a_minus_b_eq_2_angle_a_plus_b_and_a_minus_b_eq_2pi_over_3_l607_607017

variables (a b : ℝ × ℝ)

def condition1 : Prop := ∥a∥ = 1
def condition2 : Prop := ∥b∥ = sqrt 3
def condition3 : Prop := a + b = (sqrt 3, 1)

-- First proof problem: ∥a - b∥ = 2 given the conditions
theorem norm_a_minus_b_eq_2 (h1 : condition1 a) (h2: condition2 b) (h3 : condition3 a b) : ∥a - b∥ = 2 := by
  sorry

-- Second proof problem: the angle between a + b and a - b is 2π/3 given the conditions
noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := real.sqrt (u.1 ^ 2 + u.2 ^ 2)
  let norm_v := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  real.arccos (dot_product / (norm_u * norm_v))

theorem angle_a_plus_b_and_a_minus_b_eq_2pi_over_3 (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  angle_between_vectors (a + b) (a - b) = 2 * real.pi / 3 := by
  sorry

end norm_a_minus_b_eq_2_angle_a_plus_b_and_a_minus_b_eq_2pi_over_3_l607_607017


namespace probability_of_forming_triangle_l607_607593

def length_segments : List ℕ := [2, 4, 6, 8, 10]

def combinations (l : List ℕ) (k : ℕ) : List (List ℕ) :=
  (l.combinations k).filter (λ s, k ≤ l.length)

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def valid_triangle_combinations (l : List ℕ) : List (List ℕ) :=
  (combinations l 3).filter (λ s, match s with
    | [a, b, c] := can_form_triangle a b c
    | _ := false)

theorem probability_of_forming_triangle :
  (valid_triangle_combinations length_segments).length.toRat / (combinations length_segments 3).length.toRat = 3 / 10 :=
by
  sorry

end probability_of_forming_triangle_l607_607593


namespace line_passes_through_fixed_point_max_area_difference_l607_607690

-- Define the ellipse Γ
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 9) = 1

-- Define points A and B as the left and right vertices of the ellipse respectively
def A : (ℝ × ℝ) := (-5, 0)
def B : (ℝ × ℝ) := (5, 0)

-- Define line l intersecting the ellipse Γ at points M and N
variables (M N : ℝ × ℝ)

-- Define slopes k1 and k2
variables (k1 k2 : ℝ)

-- Condition that k1 : k2 = 1 : 9
def slope_ratio : Prop :=
  k2 = 9 * k1

-- Question 1: Prove that the line l passes through a fixed point (0,0)
theorem line_passes_through_fixed_point (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2)
  (h_slopes : slope_ratio k1 k2) : 
  ∃ C : ℝ × ℝ, C = (0, 0) ∧ line_through_points M N C :=
sorry

-- Define areas S1 and S2
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

def S1 : ℝ :=
  area_triangle A M N

def S2 : ℝ :=
  area_triangle B M N

-- Question 2: Find maximum value of S1 - S2
theorem max_area_difference (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2)
  (h_slopes : slope_ratio k1 k2) : 
  ∃ max_val : ℝ, max_val = 13/6 ∧ max_val = (S1 - S2) :=
sorry

end line_passes_through_fixed_point_max_area_difference_l607_607690


namespace triangle_area_ratio_l607_607353

open Real

variables {x z y : ℝ} (hxz : 0 < z < x) (n m : ℕ) (P : ℝ × ℝ)
          (P_def : P = (1/3 * (x + z), 1/3 * y))

theorem triangle_area_ratio (hxz : 0 < z < x) (n m : ℕ)
  (P : ℝ × ℝ) (P_def : P = (1/3 * (x + z), 1/3 * y)) :
  let area_A := 1/2 * (y / n) * (x + z) / 3,
      area_B := 1/2 * (x / m) * (y / 3)
  in area_A / area_B = m * (x + z) / (n * x) := 
sorry

end triangle_area_ratio_l607_607353


namespace cow_spots_total_l607_607949

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end cow_spots_total_l607_607949


namespace cow_spots_total_l607_607948

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end cow_spots_total_l607_607948


namespace a_plus_b_eq_14_l607_607907

theorem a_plus_b_eq_14 (a b : ℝ) (h : (X ^ 3 + a * X + b).is_root (2 + complex.I * real.sqrt 2)) : a + b = 14 :=
sorry

end a_plus_b_eq_14_l607_607907


namespace line_through_point_perpendicular_l607_607985

theorem line_through_point_perpendicular (C : Point) (x y : ℝ) :
  C = (2, -1) →
  (∀ p : Point, on_line p (Line.mk (-1, 1) (-3))) →
  perpendicular (Line.mk (-1, 1) (-3)) (Line.mk (1, -1) 3) →
  on_line C (Line.mk (1, -1) 3) :=
by
  sorry

end line_through_point_perpendicular_l607_607985


namespace smallest_subtracted_from_largest_two_digit_number_l607_607611

def max_two_digit_number (s : set ℕ) : ℕ :=
  s.to_finset.powerset.filter (λ x, x.card = 2)
    .image (λ x, 10 * x.max' (by simp [ne_of_gt (finset.nonempty_of_card_eq_two.mpr ⟨1, ⟨(by simp), by simp [finset.card_eq_two.mp x.card_exact]⟩)])) 
                    + x.erase (x.max' (by simp [ne_of_gt (finset.nonempty_of_card_eq_two.mpr ⟨1, ⟨(by simp), by simp [finset.card_eq_two.mp x.card_exact]⟩)]))).max' (by simp)).max' sorry

def min_two_digit_number (s : set ℕ) : ℕ :=
  s.to_finset.powerset.filter (λ x, x.card = 2)
    .image (λ x, 10 * x.min' (by simp [ne_of_gt (finset.nonempty_of_card_eq_two.mpr ⟨1, ⟨(by simp), by simp [finset.card_eq_two.mp x.card_exact]⟩)])) 
                    + x.erase (x.min' (by simp [ne_of_gt (finset.nonempty_of_card_eq_two.mpr ⟨1, ⟨(by simp), by simp [finset.card_eq_two.mp x.card_exact]⟩)]))).min' sorry

theorem smallest_subtracted_from_largest_two_digit_number : 
  max_two_digit_number {1, 4, 7, 9} - min_two_digit_number {1, 4, 7, 9} = 83 := 
by
  sorry

end smallest_subtracted_from_largest_two_digit_number_l607_607611


namespace smallest_number_satisfying_conditions_l607_607186

def erase_last_digit (N : ℕ) : ℕ := N / 10
def erase_first_digit (N : ℕ) : ℕ :=
  let digits := N.toDigits 10
  digits.tail.foldr (λ d n, 10 * n + d) 0

theorem smallest_number_satisfying_conditions :
  ∃ (N : ℕ), (erase_last_digit N % 20 = 0) ∧ (erase_first_digit N % 21 = 0) ∧ (N.toDigits 10).length > 1 ∧ (N.toDigits 10).nth 1 ≠ 0 ∧ N = 1609 := sorry

end smallest_number_satisfying_conditions_l607_607186


namespace bike_tractor_speed_ratio_l607_607568

/-- The speeds of various vehicles and the relationship between them as described in the problem statement --/
variables (speed_car speed_bike speed_tractor : ℝ)

-- Conditions provided in the problem
def condition1 : Prop := speed_car = (9/5) * speed_bike
def condition2 : Prop := speed_tractor = 575 / 23
def condition3 : Prop := speed_car = 360 / 4

-- The goal of the problem
theorem bike_tractor_speed_ratio (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  speed_bike / speed_tractor = 2 :=
by sorry

end bike_tractor_speed_ratio_l607_607568


namespace n_is_power_of_three_l607_607899

theorem n_is_power_of_three {n : ℕ} (hn_pos : 0 < n) (p : Nat.Prime (4^n + 2^n + 1)) :
  ∃ (a : ℕ), n = 3^a :=
by
  sorry

end n_is_power_of_three_l607_607899


namespace toys_per_rabbit_l607_607895

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end toys_per_rabbit_l607_607895


namespace rationalize_denominator_l607_607944

theorem rationalize_denominator : (14 / Real.sqrt 14) = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l607_607944


namespace sqrt_gt_iff_ln_gt_needed_sqrt_gt_not_sufficient_ln_gt_l607_607033

variable (a b : ℝ)

theorem sqrt_gt_iff_ln_gt_needed (ha : 0 < a) (hb : 0 < b) : 
  (ln a > ln b) → (sqrt a > sqrt b) := 
sorry

theorem sqrt_gt_not_sufficient_ln_gt (h : sqrt a > sqrt b) : 
  ¬ (ln a > ln b) := 
sorry

end sqrt_gt_iff_ln_gt_needed_sqrt_gt_not_sufficient_ln_gt_l607_607033


namespace factorize_expression_l607_607740

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l607_607740


namespace investment_return_l607_607647

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l607_607647


namespace triangle_side_lengths_log_l607_607228

theorem triangle_side_lengths_log (m : ℕ) (h : m > 0) :
  log 2 15 + log 2 50 > log 2 m ∧ log 2 15 + log 2 m > log 2 50 ∧ log 2 50 + log 2 m > log 2 15 → 
  (4 ≤ m ∧ m < 750) → (746 : ℕ) :=
sorry

end triangle_side_lengths_log_l607_607228


namespace problem1_problem2_l607_607434

noncomputable def f (m : ℝ) (x : ℝ) :=
  (Real.cos x) * (Real.sqrt 3 * (Real.sin x) - (Real.cos x)) + m

noncomputable def g (m : ℝ) (x : ℝ) :=
  f m (x + π/6)

theorem problem1 (m : ℝ) :
  (∀ x ∈ set.Icc (π / 4) (π / 3), g m x ≥ g m (π / 4) ∧ g m x ≥ g m (π / 3)) →
  (g m (π / 3) = sqrt 3 / 2) →
  m = sqrt 3 / 2 :=
sorry

theorem problem2 (A B C : ℝ) (m : ℝ) :
  (π / 3 < A ∧ A < π / 2) →
  (0 < A ∧ A < π / 2 ∧ 0 < (5 * π / 6) - A ∧ (5 * π / 6 - A) < π / 2) →
  (g (sqrt 3 / 2) (C / 2) = -1 / 2 + sqrt 3) →
  (∀ x, sin A + cos x = (sqrt 3) * (sin (A - π / 6))) →
  (π / 6 < A - π / 6 ∧ A - π / 6 < π / 3) →
  (∀ y, 1 / 2 < (sin y) ∧ (sin y) < sqrt 3 / 2) →
  (sqrt 3 / 2 < sqrt 3 * sin (A - π / 6) ∧ sqrt 3 * sin (A - π / 6) < 3 / 2) :=
sorry

end problem1_problem2_l607_607434


namespace largest_possible_difference_l607_607706

-- Definition of the conditions
def clara_estimate : ℕ := 40000
def houston_actual_range : Set ℕ := {h | 32000 ≤ h ∧ h ≤ 48000}

def daniel_estimate : ℕ := 70000
def denver_actual_range : Set ℕ := {d | 60870 ≤ d ∧ d ≤ 82353}

-- Definition of the question as proving the largest possible difference to the nearest 1,000
theorem largest_possible_difference :
  ∃ diff : ℕ, diff = 50000 ∧ 
    ∀ h ∈ houston_actual_range, ∀ d ∈ denver_actual_range, 
      abs (d - h) ≤ diff :=
sorry

end largest_possible_difference_l607_607706


namespace remainder_distinct_values_l607_607915

open Nat

theorem remainder_distinct_values 
  (a : Fin 100 → Fin 100) 
  (ha : ∀ i j : Fin 100, a i = a j → i = j)   -- a is a permutation
  (h_sum : ∀ i : Fin 100, ∃ b : ℕ, b = (∑ k in finRange i, a k))
  (h_rem : ∀ i, ∃ r : Fin 100, r = (h_sum i).1 % 100) : 
  ∃ n, 11 ≤ n ∧ ∀ S : Finset (Fin 100), S.card = 11 → ∀ i, i ∈ S → (h_rem i).1 ≠ (h_rem i+1).1 := 
begin
  sorry
end

end remainder_distinct_values_l607_607915


namespace proof_of_expression_l607_607323

def set_of_digits : Set ℕ := {8, 3, 6, 5, 0, 7}

def largest_2_digit_number (digits : Set ℕ) : ℕ :=
  let max1 := digits.max' sorry  -- Use max' to get the largest element, need non-empty proof
  let digits_without_max1 := digits.erase max1
  let max2 := digits_without_max1.max' sorry
  max1 * 10 + max2

def smallest_2_digit_number (digits : Set ℕ) : ℕ :=
  let min1 := (digits.erase 0).min' sorry  -- Use min' to get the smallest non-zero element, need non-empty proof
  let min2 := digits.min' sorry  -- Use min' to get the smallest element, need non-empty proof
  min1 * 10 + min2

theorem proof_of_expression (digits : Set ℕ)
  (h_digits : digits = {8, 3, 6, 5, 0, 7}) :
  ((largest_2_digit_number digits - smallest_2_digit_number digits) * 6 = 342) :=
by
  sorry

end proof_of_expression_l607_607323


namespace derivative_even_implies_a_eq_3_l607_607910

theorem derivative_even_implies_a_eq_3 (a α : ℝ) :
  (∀ x : ℝ, deriv (λ x, x^3 + (a - 3) * x^2 + α * x) x = deriv (λ x, x^3 + (a - 3) * x^2 + α * x) (-x)) →
  a = 3 :=
by
  -- This is where the proof would go.
  sorry

end derivative_even_implies_a_eq_3_l607_607910


namespace icepop_selling_price_l607_607674

noncomputable def selling_price_of_icepop (cost_per_pop pencil_cost total_pops total_pencils : ℝ) : ℝ :=
  let total_revenue_needed := pencil_cost * total_pencils in
  let total_cost_to_make_pops := cost_per_pop * total_pops in
  let total_profit_needed := total_revenue_needed - total_cost_to_make_pops in
  let profit_per_pop := total_profit_needed / total_pops in
  cost_per_pop + profit_per_pop

theorem icepop_selling_price 
  (cost_per_pop : ℝ)
  (pencil_cost : ℝ)
  (total_pops : ℕ)
  (total_pencils : ℕ)
  (h1 : cost_per_pop = 0.90)
  (h2 : pencil_cost = 1.80)
  (h3 : total_pops = 300)
  (h4 : total_pencils = 100) : 
  selling_price_of_icepop cost_per_pop pencil_cost total_pops total_pencils = 1.20 :=
by
  sorry

end icepop_selling_price_l607_607674


namespace sonya_falls_l607_607370

theorem sonya_falls
  (steven_falls : ℕ)
  (stephanie_falls_diff : ℕ)
  (sonya_falls_diff : ℕ)
  (sam_falls : ℕ)
  (sophie_falls_diff : ℕ)
  (h1 : steven_falls = 3)
  (h2 : stephanie_falls_diff = 13)
  (h3 : sonya_falls_diff = 2)
  (h4 : sam_falls = 1)
  (h5 : sophie_falls_diff = 4) :
  let stephanie_falls := steven_falls + stephanie_falls_diff
  let sonya_falls := (stephanie_falls / 2) - sonya_falls_diff in
  sonya_falls = 6 :=
by
  sorry

end sonya_falls_l607_607370


namespace laser_beam_path_distance_l607_607664

theorem laser_beam_path_distance :
  let A := (2, 3)
  let F := (6, 3)
  let distance (P Q : (ℝ × ℝ)) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  distance A (-6, -3) = 10 :=
sorry

end laser_beam_path_distance_l607_607664


namespace hyperbola_equation_l607_607753

-- Given ellipse
def ellipse := ∀ (x y : ℝ), x^2 / 4 + y^2 = 1

-- Point P(2,1) which hyperbola passes through
def point_P := (2 : ℝ, 1 : ℝ)

-- Hyperbola equation
def hyperbola (x y : ℝ) := x^2 / 2 - y^2 = 1

-- Prove that the equation of the hyperbola that shares the same foci
-- with the given ellipse and passes through the point P(2,1) is hyperbola.
theorem hyperbola_equation (x y : ℝ) (h_ellipse : ellipse x y) (h_point : point_P = (2, 1)) :
  hyperbola x y :=
sorry

end hyperbola_equation_l607_607753


namespace concyclic_B_E_F_H_l607_607691

variables (A B C D E F G H : Type) 
variables [circle Geometry A] [cyclic_quadrilateral ABCD]
variables [AC_diameter : diameter AC] [BD_perp_AC : ⟂ BD AC E] [F_on_extension_DA : extension_of F DA]
variables [DG_parallel_BF : parallel DG BF] [CH_perp_SF : ⟂ CH SF]

theorem concyclic_B_E_F_H :
  cyclic_quad B E F H :=
sorry

end concyclic_B_E_F_H_l607_607691


namespace Malou_average_is_correct_l607_607531

def quiz1_score : ℕ := 91
def quiz2_score : ℕ := 90
def quiz3_score : ℕ := 92
def total_score : ℕ := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes : ℕ := 3

def Malous_average_score : ℕ := total_score / number_of_quizzes

theorem Malou_average_is_correct : Malous_average_score = 91 := by
  sorry

end Malou_average_is_correct_l607_607531


namespace lean4_math_problem_l607_607518

noncomputable def triangle := sorry   -- Placeholder definition

variables {A B C I D P Q : triangle}

-- Conditions
axiom triangle_with_incenter 
  (ABC: triangle) 
  (I: triangle) 
  (D: triangle) 
  (P: triangle)
  (Q: triangle)
  (BI: ℝ)
  (CI: ℝ)
  (DI: ℝ)
  (BC: triangle) 
  (CIRCLE_DIAMETER_AI: triangle)
  (TOUCH: BC)
  (LINES: BI)
  : CI = 5 ∧ DI = 3 ∧ BI = 6 
    ∧ incircle (ABC) (TOUCH)
    ∧ (P ∈ CIRCLE_DIAMETER_AI) 
    ∧ (Q ∈ CIRCLE_DIAMETER_AI)

-- Theorem Statement
theorem lean4_math_problem :
  ∀ {A B C I D P Q : triangle}
  (h: triangle_with_incenter ABC I D P Q 6 5 3 BC CIRCLE_DIAMETER_AI)
, \left(\frac{DP}{DQ}\right)^2 = \frac{75}{64} := sorry

end lean4_math_problem_l607_607518


namespace sin_graph_symmetry_l607_607575

theorem sin_graph_symmetry :
  let f := λ x : ℝ, Real.sin (2 * x + Real.pi / 3)
  in (f (Real.pi / 3) = 0) ∧
     (f (Real.pi / 4) ≠ 0.5) ∧
     (f (Real.pi / 4) ≠ 0) ∧
     (f (Real.pi / 12) = 1) :=
by
  let f := λ x : ℝ, Real.sin (2 * x + Real.pi / 3)
  sorry

end sin_graph_symmetry_l607_607575


namespace card_distribution_l607_607693

theorem card_distribution (n : ℕ) (players : Fin n → ℕ) (cards : Fin n → ℕ) :
  (∑ i, cards i = n - 1) → 
  (∀ i, (cards i ≥ 2 → ∃! j k, j ≠ k ∧ (players j, players k) are_neighbours_in_circular_table_with_i) →
    eventually (∀ i, cards i ≤ 1)) :=
sorry

end card_distribution_l607_607693


namespace certain_number_d_sq_l607_607459

theorem certain_number_d_sq (d n m : ℕ) (hd : d = 14) (h : n * d = m^2) : n = 14 :=
by
  sorry

end certain_number_d_sq_l607_607459


namespace proof_ellipse_and_distance_l607_607795

-- Definitions of ellipse
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b e : ℝ) : Prop := e = Real.sqrt(1 - (b^2 / a^2))

-- Given conditions
variables (x y m : ℝ)
constant a : ℝ := 2
constant b : ℝ := 1
constant e : ℝ := Real.sqrt(3) / 2

-- Ellipse E passes through the point (0, 1)
axiom point_on_ellipse : ellipse 0 1 a b

-- Eccentricity condition
axiom eccentricity_condition : eccentricity a b e

-- Line l satisfies the equation
def line_l (m x y : ℝ) : Prop := y = x / 2 + m

-- Intersection of line with ellipse
def intersection_with_ellipse (m x y : ℝ) : Prop := ellipse x y a b ∧ line_l m x y

-- Distance BN calculation
def distance_BN (m : ℝ) : ℝ := Real.sqrt 10 / 2

-- Proof statement
theorem proof_ellipse_and_distance :
  (∀ (x y m : ℝ), (a > b ∧ b > 0 ∧ eccentricity a b e ∧ ellipse x y a b ∧ intersection_with_ellipse m x y) →
        (ellipse x y 2 1) ∧ (∀ (B N : ℝ × ℝ), distance_BN m = Real.sqrt 10 / 2)) := 
sorry

end proof_ellipse_and_distance_l607_607795


namespace solve_equation_l607_607962

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8) ∨ (x = 5) :=
by
  sorry

end solve_equation_l607_607962


namespace no_integer_k_such_that_f_k_eq_8_l607_607406

noncomputable def polynomial_with_integer_coefficients (n : ℕ) : Type :=
  {f : Polynomial ℤ // f.degree = n}

theorem no_integer_k_such_that_f_k_eq_8
  (f : polynomial_with_integer_coefficients)
  (a b c d : ℤ)
  (h0 : a ≠ b)
  (h1 : a ≠ c)
  (h2 : a ≠ d)
  (h3 : b ≠ c)
  (h4 : b ≠ d)
  (h5 : c ≠ d)
  (h6 : f.val.eval a = 5)
  (h7 : f.val.eval b = 5)
  (h8 : f.val.eval c = 5)
  (h9 : f.val.eval d = 5)
  : ¬ ∃ k : ℤ, f.val.eval k = 8 :=
sorry

end no_integer_k_such_that_f_k_eq_8_l607_607406


namespace complementary_event_probability_l607_607040

-- Define A and B as events such that B is the complement of A.
section
variables (A B : Prop) -- A and B are propositions representing events.
variable (P : Prop → ℝ) -- P is a function that gives the probability of an event.

-- Define the conditions for the problem.
variable (h_complementary : ∀ A B, A ∧ B = false ∧ A ∨ B = true) 
variable (h_PA : P A = 1 / 5)

-- The statement to be proved.
theorem complementary_event_probability : P B = 4 / 5 :=
by
  -- Here we would provide the proof, but for now, we use 'sorry' to bypass it.
  sorry
end

end complementary_event_probability_l607_607040


namespace magnitude_of_z_l607_607563

-- Define complex numbers w and z
variables {w z : ℂ}

-- Given conditions
def cond1 := (w * z = 20 - 15 * complex.i)
def cond2 := (complex.abs w = 5)

-- Proof statement
theorem magnitude_of_z (h1 : cond1) (h2 : cond2) : complex.abs z = 5 :=
sorry

end magnitude_of_z_l607_607563


namespace binomial_7_4_eq_35_l607_607713

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607713


namespace problem_statement_l607_607167

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l607_607167


namespace find_f7_l607_607422

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2 * x^2 else
    let y := x % 4 in
      if y < 0 then f (-y)
      else f y

theorem find_f7 : (f 7 = 2) :=
  sorry

end find_f7_l607_607422


namespace problem_statement_l607_607055

def f (x : ℝ) : ℝ := 2 * Real.sin ((1/3) * x - Real.pi / 6)

theorem problem_statement (x : ℝ) (α β : ℝ) (hx : x = 5 * Real.pi / 4)
    (hα : α ∈ Set.Icc 0 (Real.pi / 2))
    (hβ : β ∈ Set.Icc 0 (Real.pi / 2))
    (hα_condition : f (3 * α + Real.pi / 2) = 10 / 13)
    (hβ_condition : f (3 * β + 2 * Real.pi) = 6 / 5) :
    f x = Real.sqrt 2 ∧ Real.cos (α + β) = 16 / 65 :=
by
  sorry

end problem_statement_l607_607055


namespace michelle_has_total_crayons_l607_607532

noncomputable def michelle_crayons : ℕ :=
  let type1_crayons_per_box := 5
  let type2_crayons_per_box := 12
  let type1_boxes := 4
  let type2_boxes := 3
  let missing_crayons := 2
  (type1_boxes * type1_crayons_per_box - missing_crayons) + (type2_boxes * type2_crayons_per_box)

theorem michelle_has_total_crayons : michelle_crayons = 54 :=
by
  -- The proof step would go here, but it is omitted according to instructions.
  sorry

end michelle_has_total_crayons_l607_607532


namespace problem_solution_l607_607782

def f (x : ℝ) : ℝ :=
if x > 0 then 2 * x else f (x + 1)

theorem problem_solution : f (4 / 3) + f (-4 / 3) = 4 := 
by
  sorry

end problem_solution_l607_607782


namespace two_lines_perpendicular_to_same_line_are_parallel_l607_607570

/- Define what it means for two lines to be perpendicular -/
def perpendicular (l m : Line) : Prop :=
  -- A placeholder definition for perpendicularity, replace with the actual definition
  sorry

/- Define what it means for two lines to be parallel -/
def parallel (l m : Line) : Prop :=
  -- A placeholder definition for parallelism, replace with the actual definition
  sorry

/- Given: Two lines l1 and l2 that are perpendicular to the same line l3 -/
variables (l1 l2 l3 : Line)
variable (h1 : perpendicular l1 l3)
variable (h2 : perpendicular l2 l3)

/- Prove: l1 and l2 are parallel to each other -/
theorem two_lines_perpendicular_to_same_line_are_parallel :
  parallel l1 l2 :=
  sorry

end two_lines_perpendicular_to_same_line_are_parallel_l607_607570


namespace standard_eq_of_parabola_l607_607462

-- Define a point on the parabola
def Point (x y : ℝ) := (x, y)

-- Define the equation of the parabola
def parabola_equation (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the distance from the point to the directrix
def distance_to_directrix (p x : ℝ) : ℝ := p / 2 + x

-- Given assumptions
variables (y0 p : ℝ) (hp : p > 0)

-- Given point on parabola where x coordinate is 2
def point_on_parabola := Point 2 y0

-- Given distance from point to directrix is 4
def distance_condition : Prop := distance_to_directrix p 2 = 4

-- Prove the standard equation of the parabola is y^2 = 8x
theorem standard_eq_of_parabola (h : parabola_equation p 2 y0) (hd : distance_condition) : y^2 = 8 * x :=
sorry

end standard_eq_of_parabola_l607_607462


namespace john_gifts_total_l607_607498

/-- John received 20 gifts on his 12th birthday. On his 13th birthday, he received 25% fewer
    gifts than on his 12th birthday. How many total gifts did he receive between those two birthdays? -/
theorem john_gifts_total (gifts_12_bday : ℕ) (percent_fewer_13_bday : ℚ) 
  (gifts_13_bday : ℕ) (total_gifts : ℕ) 
  (h1 : gifts_12_bday = 20) 
  (h2 : percent_fewer_13_bday = 0.25)
  (h3 : gifts_13_bday = gifts_12_bday - nat.floor (percent_fewer_13_bday * gifts_12_bday))
  (h4 : total_gifts = gifts_12_bday + gifts_13_bday) : 
  total_gifts = 35 := 
sorry

end john_gifts_total_l607_607498


namespace total_buyers_l607_607655

-- Definitions based on conditions
def C : ℕ := 50
def M : ℕ := 40
def B : ℕ := 19
def pN : ℝ := 0.29  -- Probability that a random buyer purchases neither

-- The theorem statement
theorem total_buyers :
  ∃ T : ℝ, (T = (C + M - B) + pN * T) ∧ T = 100 :=
by
  sorry

end total_buyers_l607_607655


namespace total_points_other_7_members_is_15_l607_607099

variable (x y : ℕ)
variable (h1 : y ≤ 21)
variable (h2 : y = x * 7 / 15 - 18)
variable (h3 : (1 / 3) * x + (1 / 5) * x + 18 + y = x)

theorem total_points_other_7_members_is_15 (h : x * 7 % 15 = 0) : y = 15 :=
by
  sorry

end total_points_other_7_members_is_15_l607_607099


namespace sqrt_2_minus_x_meaningful_l607_607466

theorem sqrt_2_minus_x_meaningful (x : ℝ) : 
  x = -1 ↔ (√(2 - x)).IsReal ∧ (x = 4 ∨ x = π ∨ x = -1 ∨ x = 3) :=
by 
  sorry

end sqrt_2_minus_x_meaningful_l607_607466


namespace employee_Y_payment_l607_607628

variable (x y : ℝ)

theorem employee_Y_payment (hx : x = 1.2 * y) (h_total : x + y = 800) :
  y ≈ 363.64 :=
by {
  -- proof goes here
  sorry
}

end employee_Y_payment_l607_607628


namespace length_of_segment_l607_607294

theorem length_of_segment (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi / 2)
  (h₁ : 6 * Real.cos x = 5 * Real.tan x) :
  ∃ P_1 P_2 : ℝ, P_1 = 0 ∧ P_2 = (1 / 2) * Real.sin x ∧ abs (P_2 - P_1) = 1 / 3 :=
by
  sorry

end length_of_segment_l607_607294


namespace permuation_calculation_l607_607344

def permutation (n k : Nat) : Nat :=
  Nat.factorial n / Nat.factorial (n - k)

theorem permuation_calculation :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 2.4 := 
by
  -- Defining the conditions
  let A_8_4 := permutation 8 4
  let A_8_5 := permutation 8 5
  let A_8_6 := permutation 8 6
  let A_9_5 := permutation 9 5
  let zero_factorial := 1 -- 0! = 1
  -- Using the conditions to prove the equality
  have h1 : A_8_4 = permutation 8 4 := rfl
  have h2 : A_8_5 = permutation 8 5 := rfl
  have h3 : A_8_6 = permutation 8 6 := rfl
  have h4 : A_9_5 = permutation 9 5 := rfl
  have h5 : zero_factorial = 1 := by rfl
  sorry

end permuation_calculation_l607_607344


namespace trig_identity_proof_l607_607365

theorem trig_identity_proof :
  (sin (15 * Real.pi / 180) * cos (10 * Real.pi / 180) + cos (165 * Real.pi / 180) * cos (105 * Real.pi / 180)) /
  (sin (19 * Real.pi / 180) * cos (11 * Real.pi / 180) + cos (161 * Real.pi / 180) * cos (101 * Real.pi / 180))
  = sin (5 * Real.pi / 180) / sin (8 * Real.pi / 180) :=
by
  sorry

end trig_identity_proof_l607_607365


namespace donald_final_payment_l607_607369

-- Define the original costs, discount rates, and sales tax rate
def orig_laptop_price : ℝ := 800
def laptop_discount_rate : ℝ := 0.15
def orig_accessories_price : ℝ := 200
def accessories_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.07

-- Define the expected final amount
def expected_final_amount : ℝ := 920.20

-- Prove the final amount given the conditions
theorem donald_final_payment :
  let laptop_discount := laptop_discount_rate * orig_laptop_price,
      discounted_laptop_price := orig_laptop_price - laptop_discount,
      accessories_discount := accessories_discount_rate * orig_accessories_price,
      discounted_accessories_price := orig_accessories_price - accessories_discount,
      total_before_tax := discounted_laptop_price + discounted_accessories_price,
      sales_tax := sales_tax_rate * total_before_tax,
      final_amount := total_before_tax + sales_tax
  in final_amount = expected_final_amount :=
by
  let laptop_discount := laptop_discount_rate * orig_laptop_price
  let discounted_laptop_price := orig_laptop_price - laptop_discount
  let accessories_discount := accessories_discount_rate * orig_accessories_price
  let discounted_accessories_price := orig_accessories_price - accessories_discount
  let total_before_tax := discounted_laptop_price + discounted_accessories_price
  let sales_tax := sales_tax_rate * total_before_tax
  let final_amount := total_before_tax + sales_tax
  have : final_amount = 920.20 := sorry
  exact this

end donald_final_payment_l607_607369


namespace angela_finished_9_problems_l607_607328

def martha_problems : Nat := 2

def jenna_problems : Nat := 4 * martha_problems - 2

def mark_problems : Nat := jenna_problems / 2

def total_problems : Nat := 20

def total_friends_problems : Nat := martha_problems + jenna_problems + mark_problems

def angela_problems : Nat := total_problems - total_friends_problems

theorem angela_finished_9_problems : angela_problems = 9 := by
  -- Placeholder for proof steps
  sorry

end angela_finished_9_problems_l607_607328


namespace trapezoid_bd_length_l607_607877

theorem trapezoid_bd_length
  (AB CD AC BD : ℝ)
  (tanC tanB : ℝ)
  (h1 : AB = 24)
  (h2 : CD = 15)
  (h3 : AC = 30)
  (h4 : tanC = 2)
  (h5 : tanB = 1.25)
  (h6 : AC ^ 2 = AB ^ 2 + (CD - AB) ^ 2) :
  BD = 9 * Real.sqrt 11 := by
  sorry

end trapezoid_bd_length_l607_607877


namespace puzzle_solution_l607_607382

-- Definitions for the digits
def K : ℕ := 3
def O : ℕ := 2
def M : ℕ := 4
def R : ℕ := 5
def E : ℕ := 6

-- The main proof statement
theorem puzzle_solution : (10 * K + O : ℕ) + (M / 10 + K / 10 + O / 100) = (10 * K + R : ℕ) + (O / 10 + M / 100) := 
  by 
  sorry

end puzzle_solution_l607_607382


namespace inequality_satisfaction_l607_607548

theorem inequality_satisfaction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y + 1 / x + y ≥ y / x + 1 / y + x) ↔ 
  ((x = y) ∨ (x = 1 ∧ y ≠ 0) ∨ (y = 1 ∧ x ≠ 0)) ∧ (x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end inequality_satisfaction_l607_607548


namespace aquarium_height_l607_607929

theorem aquarium_height (h : ℝ) (V : ℝ) (final_volume : ℝ) :
  let length := 4
  let width := 6
  let halfway_volume := (length * width * h) / 2
  let spilled_volume := halfway_volume / 2
  let tripled_volume := 3 * spilled_volume
  tripled_volume = final_volume →
  final_volume = 54 →
  h = 3 := by
  intros
  sorry

end aquarium_height_l607_607929


namespace exists_x0_f_leq_one_tenth_l607_607435

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3*x))^2 - 2*a*x - 6*a*(Real.log (3*x)) + 10*a^2

theorem exists_x0_f_leq_one_tenth (a : ℝ) : (∃ x₀, f x₀ a ≤ 1/10) ↔ a = 1/30 := by
  sorry

end exists_x0_f_leq_one_tenth_l607_607435


namespace adjacent_probability_is_2_over_7_l607_607972

variable (n : Nat := 5) -- number of student performances
variable (m : Nat := 2) -- number of teacher performances

/-- Total number of ways to insert two performances
    (ignoring adjacency constraints) into the program list. -/
def total_insertion_ways : Nat :=
  Fintype.card (Fin (n + m))

/-- Number of ways to insert two performances such that they are adjacent. -/
def adjacent_insertion_ways : Nat :=
  Fintype.card (Fin (n + 1))

/-- Probability that two specific performances are adjacent in a program list. -/
def adjacent_probability : ℚ :=
  adjacent_insertion_ways / total_insertion_ways

theorem adjacent_probability_is_2_over_7 :
  adjacent_probability = (2 : ℚ) / 7 := by
  sorry

end adjacent_probability_is_2_over_7_l607_607972


namespace eccentricity_range_of_ellipse_l607_607813

noncomputable def range_of_eccentricity (a b: ℝ) (h_ab: a > b) :=
  let e := sqrt (1 - (b^2 / a^2)) in
  e >= (sqrt 5 - 1) / 2 ∧ e < 1

theorem eccentricity_range_of_ellipse (a b c: ℝ) (h_ab: a > b) (h_c: c = sqrt (a^2 - b^2)):
  ∃ e, 
  let e := sqrt (1 - (b / a)^2) in
  e >= (sqrt 5 - 1) / 2 ∧ e < 1 :=
sorry

end eccentricity_range_of_ellipse_l607_607813


namespace circle_radius_is_zero_l607_607758

-- Define the condition: the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 10 * y + 41 = 0

-- Define the statement to be proved: the radius of the circle described by the equation is 0
theorem circle_radius_is_zero : ∀ x y : ℝ, circle_eq x y → (∃ r : ℝ, r = 0) :=
begin
  sorry
end

end circle_radius_is_zero_l607_607758


namespace binomial_7_4_eq_35_l607_607708
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607708


namespace possible_to_form_larger_rectangle_l607_607890

/-- 
  Assumes a sequence of rectangles with dimensions 1 x 1, 1 x 3, ... up to 1 x 2019.
  Proves it is possible to arrange these rectangles to form a larger rectangle with each side greater than 1.
-/
theorem possible_to_form_larger_rectangle : 
  ∃ (L W : ℕ), L > 1 ∧ W > 1 ∧ (∑ i in range(1010), (2 * i + 1) = L * W) :=
sorry

end possible_to_form_larger_rectangle_l607_607890


namespace area_of_given_rhombus_side_length_of_given_rhombus_l607_607022

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

noncomputable def side_length_of_rhombus (d1 d2 angle_deg : ℝ) : ℝ :=
  let half_d1 := d1 / 2
  let cos_angle := Real.cos (angle_deg * Real.pi / 180)
  half_d1 / cos_angle

theorem area_of_given_rhombus : area_of_rhombus 62 80 = 2480 := by
  sorry

theorem side_length_of_given_rhombus : side_length_of_rhombus 62 80 37 ≈ 38.83 := by
  sorry

end area_of_given_rhombus_side_length_of_given_rhombus_l607_607022


namespace interest_rate_proof_l607_607665

noncomputable def interest_rate (P : ℕ) (r_c : Rational) (T : ℕ) (gain_B : ℕ) : Rational :=
  let interest_from_C := P * r_c * T
  let interest_to_A := interest_from_C - gain_B
  interest_to_A / (P * T)

theorem interest_rate_proof :
  interest_rate 3500 (115 / 1000) 3 157.5 = 0.1 :=
by
  -- The proof steps would go here
  sorry

end interest_rate_proof_l607_607665


namespace range_of_product_of_roots_l607_607430

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem range_of_product_of_roots :
  ∀ m : ℝ,
  (∀ x : ℝ, f x = m → x ∈ {x₁, x₂, x₃, x₄}) →
  (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₄) →
  x₁ * x₂ * x₃ * x₄ ∈ set.Ioo (-3 : ℝ) (0 : ℝ) :=
by
  sorry

end range_of_product_of_roots_l607_607430


namespace distinct_solutions_difference_l607_607148

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l607_607148


namespace roys_cat_finishes_food_on_sunday_l607_607554

/--
Roy's cat eats 2/5 of a can of cat food every morning and 1/5 of a can every evening.
Starting Monday morning, Roy opens a new box containing 8 cans of cat food.
Prove that the day of the week Roy's cat finishes eating all the cat food in the box is Sunday (14 days later).
-/
theorem roys_cat_finishes_food_on_sunday
  (morning_consumption : ℚ := 2/5)
  (evening_consumption : ℚ := 1/5)
  (total_cans : ℚ := 8)
  (days_per_week : ℕ := 7) :
  let daily_consumption := morning_consumption + evening_consumption in
  total_cans / daily_consumption = 14 := 
by
  let _daily_consumption := morning_consumption + evening_consumption;
  have h_div : 8 / _daily_consumption = 14,
  {
    sorry
  };
  exact h_div

end roys_cat_finishes_food_on_sunday_l607_607554


namespace round_nearest_tenth_45_26384_l607_607552

theorem round_nearest_tenth_45_26384 : 
  let num := 45.26384
  let tenths := 2 
  let hundredths := 6 
  (hundredths >= 5) → (tenths + 1 = 3) → (num ≈ 45.3) := 
by
  sorry

end round_nearest_tenth_45_26384_l607_607552


namespace fraction_of_income_to_taxes_l607_607286

noncomputable def joe_income : ℕ := 2120
noncomputable def joe_taxes : ℕ := 848

theorem fraction_of_income_to_taxes : (joe_taxes / gcd joe_taxes joe_income) / (joe_income / gcd joe_taxes joe_income) = 106 / 265 := sorry

end fraction_of_income_to_taxes_l607_607286


namespace count_N_with_two_four_digit_numbers_l607_607841

theorem count_N_with_two_four_digit_numbers :
  ∃ N : ℕ, (N > 900) ∧ exactly_two_are_four_digit (3 * N, N - 900, N + 15, 2 * N) = 5069
:= by
  // proof goes here
  sorry

noncomputable def exactly_two_are_four_digit (a b c d : ℕ) : Prop := 
  let four_digit(x) := (1000 ≤ x ∧ x < 10000) in
  (four_digit(a) ∧ ¬four_digit(b) ∧ four_digit(c) ∧ ¬four_digit(d)) ∨
  (four_digit(a) ∧ four_digit(b) ∧ ¬four_digit(c) ∧ ¬four_digit(d)) ∨
  (four_digit(a) ∧ ¬four_digit(b) ∧ ¬four_digit(c) ∧ four_digit(d)) ∨
  (¬four_digit(a) ∧ four_digit(b) ∧ four_digit(c) ∧ ¬four_digit(d)) ∨
  (¬four_digit(a) ∧ four_digit(b) ∧ ¬four_digit(c) ∧ four_digit(d)) ∨
  (¬four_digit(a) ∧ ¬four_digit(b) ∧ four_digit(c) ∧ four_digit(d))

end count_N_with_two_four_digit_numbers_l607_607841


namespace speed_of_car_B_l607_607701

-- Problem Definitions
def speed_car_B : ℝ := 1500 / 38
def speed_car_A (v_B : ℝ) : ℝ := 3 * v_B
def speed_car_C (v_B : ℝ) : ℝ := 1.5 * (speed_car_A v_B + v_B)
def distance_car_A (v_B : ℝ) : ℝ := speed_car_A v_B * 6
def distance_car_B (v_B : ℝ) : ℝ := v_B * 2
def distance_car_C (v_B : ℝ) : ℝ := speed_car_C v_B * 3

-- Theorem statement to prove
theorem speed_of_car_B (v_B : ℝ) : 
  distance_car_A v_B + distance_car_B v_B + distance_car_C v_B = 1500 → 
  v_B = speed_car_B :=
by
  sorry

end speed_of_car_B_l607_607701


namespace distribution_plans_count_l607_607368

def students := {S1, S2, S3, S4}
def classes := {A, B, C}

-- Given conditions
def cannot_be_assigned (s : students) (c : classes) : Prop := 
  (s = S1) → (c ≠ A)

-- Main theorem to state the problem and the solution
theorem distribution_plans_count : ∃ f : (students → classes), 
  (∀ (s : students), ∃ c : classes, f s = c) ∧ 
  (∀ (c : classes), ∃ s, f s = c) ∧ 
  cannot_be_assigned S1 (f S1) ∧
  (f S1 = B ∨ f S1 = C) ∧ 
  (∑ c : classes, (∑ s in students, (if f s = c then 1 else 0)) = 3) :=
sorry

end distribution_plans_count_l607_607368


namespace penny_identified_whales_l607_607371

theorem penny_identified_whales (sharks eels total : ℕ)
  (h_sharks : sharks = 35)
  (h_eels   : eels = 15)
  (h_total  : total = 55) :
  total - (sharks + eels) = 5 :=
by
  sorry

end penny_identified_whales_l607_607371


namespace angela_finished_9_problems_l607_607326

def martha_problems : Nat := 2

def jenna_problems : Nat := 4 * martha_problems - 2

def mark_problems : Nat := jenna_problems / 2

def total_problems : Nat := 20

def total_friends_problems : Nat := martha_problems + jenna_problems + mark_problems

def angela_problems : Nat := total_problems - total_friends_problems

theorem angela_finished_9_problems : angela_problems = 9 := by
  -- Placeholder for proof steps
  sorry

end angela_finished_9_problems_l607_607326


namespace log_simplification_l607_607202

noncomputable def log := Real.log10

theorem log_simplification : 
  log (5^2) + log 2 * log (50) + (log 2)^2 = 2 :=
by
  sorry

end log_simplification_l607_607202


namespace convex_polygon_triangle_count_l607_607416

theorem convex_polygon_triangle_count {n : ℕ} (h : n ≥ 5) :
  ∃ T : ℕ, T ≤ n * (2 * n - 5) / 3 :=
by
  sorry

end convex_polygon_triangle_count_l607_607416


namespace alex_fourth_test_score_l607_607502

theorem alex_fourth_test_score :
  ∃ (s₁ s₂ s₃ s₄ s₅ : ℤ),
  85 ≤ s₁ ∧ s₁ ≤ 95 ∧
  85 ≤ s₂ ∧ s₂ ≤ 95 ∧
  85 ≤ s₃ ∧ s₃ ≤ 95 ∧
  85 ≤ s₄ ∧ s₄ ≤ 95 ∧
  85 ≤ s₅ ∧ s₅ ≤ 95 ∧
  s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₁ ≠ s₅ ∧
  s₂ ≠ s₃ ∧ s₂ ≠ s₄ ∧ s₂ ≠ s₅ ∧
  s₃ ≠ s₄ ∧ s₃ ≠ s₅ ∧
  s₄ ≠ s₅ ∧
  ((s₁ + s₂) / 2) ∈ ℤ ∧
  ((s₁ + s₂ + s₃) / 3) ∈ ℤ ∧
  ((s₁ + s₂ + s₃ + s₄) / 4) ∈ ℤ ∧
  ((s₁ + s₂ + s₃ + s₄ + s₅) / 5) ∈ ℤ ∧
  s₅ = 90 ∧
  s₄ = 95 :=
sorry

end alex_fourth_test_score_l607_607502


namespace monotonic_interval_1_exactly_three_zeros_inequality_of_zeros_l607_607820

-- Part 1
theorem monotonic_interval_1 (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x > 0, f x = (x - 1) / (x + 1) - log x)
  (ha : a = 1):
  ∀ x1 x2, 0 < x1 ∧ 0 < x2 ∧ x1 < x2 → f x1 > f x2 :=
sorry

-- Part 2 (i)
theorem exactly_three_zeros (f : ℝ → ℝ) (a : ℝ) 
  (ha1 : 0 < a) (ha2 : a < 1/2) 
  (h : ∀ x > 0, f x = (x - 1) / (x + 1) - a * log x) :
  ∃ x1 x2 x3, (x1 < x2 ∧ x2 < x3) ∧ (f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0):=
sorry

-- Part 2 (ii)
theorem inequality_of_zeros (a : ℝ) (x1 x2 x3 : ℝ)
  (h1 : x1 < x2 ∧ x2 < x3)
  (h2 : 0 < a ∧ a < 1/2) 
  (h3 : ∀ x > 0, f x = (x - 1) / (x + 1) - a * log x)
  (hzeros : f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0):
  x1^2 * (1 - x3) > a * (x1^2 - 1) :=
sorry

end monotonic_interval_1_exactly_three_zeros_inequality_of_zeros_l607_607820


namespace ratio_of_side_length_to_brush_width_l607_607281

theorem ratio_of_side_length_to_brush_width (s w : ℝ) (h : (w^2 + ((s - w)^2) / 2) = s^2 / 3) : s / w = 3 :=
by
  sorry

end ratio_of_side_length_to_brush_width_l607_607281


namespace payal_finished_fraction_l607_607938

-- Define the conditions
variables (x : ℕ)

-- Given conditions
-- 1. Total pages in the book
def total_pages : ℕ := 60
-- 2. Payal has finished 20 more pages than she has yet to read.
def pages_yet_to_read (x : ℕ) : ℕ := x - 20

-- Main statement to prove: the fraction of the pages finished is 2/3
theorem payal_finished_fraction (h : x + (x - 20) = 60) : (x : ℚ) / 60 = 2 / 3 :=
sorry

end payal_finished_fraction_l607_607938


namespace distinct_solution_difference_l607_607172

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l607_607172


namespace B_can_complete_work_in_30_days_l607_607283

theorem B_can_complete_work_in_30_days :
  (A_work_rate : ℝ) (B_work_rate : ℝ) (x : ℝ) 
  (hA1 : A_work_rate = 1 / 20) 
  (hB1 : 1 / x = 1 / 30) 
  (hWork_A : 10 * A_work_rate = 1 / 2) 
  (hWork_B : B_work_rate = 1 / x) 
  (hB_days : B_work_rate * 15 = 1 / 2) :
  x = 30 :=
begin
  sorry
end

end B_can_complete_work_in_30_days_l607_607283


namespace urea_formation_l607_607755

-- Define the chemical reaction and the stoichiometry as conditions
def reaction (NH3 CO2 NH2CONH2 H2O : ℝ) : Prop :=
  2 * NH3 + CO2 = NH2CONH2 + H2O

-- Define the conditions
def conditions (NH3 CO2 NH2CONH2 formation : ℝ) : Prop :=
  NH3 = 2 ∧ formation = 1 ∧ ∃ urea, urea = NH2CONH2

-- The theorem that needs to be proved
theorem urea_formation (NH3 CO2 NH2CONH2 H2O formation : ℝ) :
  conditions NH3 CO2 NH2CONH2 formation →
  reaction NH3 CO2 NH2CONH2 H2O →
  NH3 = 2 ∧ CO2 = 1 ∧ formation = 1 :=
begin
  sorry
end

end urea_formation_l607_607755


namespace find_a_for_parallel_lines_l607_607527

-- Definition of slope for a line given two points
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Given conditions
def line_l_points := ((-2 : ℝ), (0 : ℝ), (0 : ℝ), (a : ℝ))
def line_ll_points := ((4 : ℝ), (0 : ℝ), (6 : ℝ), (2 : ℝ))

noncomputable def slope_l := slope (-2) 0 0 a
noncomputable def slope_ll := slope 4 0 6 2

theorem find_a_for_parallel_lines (a : ℝ) :
  slope_l a = slope_ll → a = 2 :=
by
  intros h
  sorry

end find_a_for_parallel_lines_l607_607527


namespace intersecting_line_circle_l607_607002

theorem intersecting_line_circle (a b : ℝ) 
  (h1 : b < 2)
  (h2 : ∀ x y : ℝ, ax + y + a + 1 = 0 → x = -1 ∧ y = -1)
  (h3 : 2 - b - 6 > 0) : ∀ b : ℝ, b < -6 := 
begin
  sorry
end

end intersecting_line_circle_l607_607002


namespace new_rectangle_area_half_l607_607727

variable (A B C D O : Type)

-- Define that A, B, C, D form a rectangle
def rectangle (A B C D : Type) : Prop :=
  -- Assuming that we can define properties of the rectangle here

-- Define that the area of rectangle ABCD is 1
def area_rectangle_one (A B C D : Type) [rect : rectangle A B C D] : Prop :=
  -- Area ABCD = 1 unit
  sorry

-- Define incenter O of the triangle ABC
def is_incenter (O : Type) (A B C : Type) : Prop :=
  -- Assuming that we can define properties of incenter here
  
  sorry

-- Define the new rectangle (ODEF)
def new_rectangle (O D : Type) : Prop :=
  -- Assuming properties that define the new rectangle ODEF
  
  sorry

-- Goal: area of new rectangle ODEF = 1 / 2
theorem new_rectangle_area_half (A B C D O : Type) 
  (h1 : rectangle A B C D)
  (h2 : area_rectangle_one A B C D)
  (h3 : is_incenter O A B C)
  (h4 : new_rectangle O D) : 
  area O D = 1/2 :=
sorry

end new_rectangle_area_half_l607_607727


namespace binomial_theorem_expansion_l607_607612

theorem binomial_theorem_expansion (n k : ℤ) (a b c : ℤ) : 
  n ≥ 2 ∧ a ≠ b ∧ ab ≠ 0 ∧ a = kb + c ∧ k > 0 ∧ c ≠ 0 ∧ c ≠ b * (k - 1) ∧ 
  (b * (k-1) + c)^n + n * (b * (k-1))^(n-1) * c = 0 :=
by
  sorry

end binomial_theorem_expansion_l607_607612


namespace find_n_l607_607904

def alpha (n : ℕ) : ℚ := ((n - 2) * 180) / n
def alpha_plus_3 (n : ℕ) : ℚ := ((n + 1) * 180) / (n + 3)
def alpha_minus_2 (n : ℕ) : ℚ := ((n - 4) * 180) / (n - 2)

theorem find_n (n : ℕ) (h : alpha_plus_3 n - alpha n = alpha n - alpha_minus_2 n) : n = 12 :=
by
  -- The proof will be added here
  sorry

end find_n_l607_607904


namespace number_of_drawn_games_is_54_l607_607102

-- Definitions based on the conditions provided in the problem

def participants : ℕ := 12

def lists (n : ℕ) : fin 13 → set ℕ
| ⟨0, _⟩ := {n}
| ⟨k + 1, hk⟩ := lists ⟨k, sorry⟩ ∪ ⋃ m ∈ lists ⟨k, sorry⟩, {x | participant_defeated m x}

noncomputable def final_list_differs (n : ℕ) : Prop :=
  ∃ x, x ∈ lists n ⟨12, sorry⟩ ∧ x ∉ lists n ⟨11, sorry⟩

axiom participants_have_different_final_lists : ∀ n, final_list_differs n

-- Main theorem to prove the number of drawn games
noncomputable def number_of_drawn_games : ℕ :=
  66 - 12

theorem number_of_drawn_games_is_54 : number_of_drawn_games = 54 :=
by
  sorry

end number_of_drawn_games_is_54_l607_607102


namespace part1_part2_l607_607428

-- Define the quadratic equation
def quadratic_eq (a b c : ℂ) := λ x : ℂ, a * x^2 + b * x + c = 0

-- Part 1: Given p = 8, find x₁ and x₂
theorem part1 (x₁ x₂ : ℂ) (h₁ : quadratic_eq 1 (-8) 25 x₁) (h₂ : quadratic_eq 1 (-8) 25 x₂) :
  x₁ = 4 + 3 * complex.I ∧ x₂ = 4 - 3 * complex.I :=
sorry

-- Part 2: Given x₁ = 3 + 4i, find the value of p
theorem part2 (x₁ : ℂ) (h₁ : x₁ = 3 + 4 * complex.I) (h₂ : ∃ p : ℝ, quadratic_eq 1 (-p) 25 x₁) :
  ∃ p : ℝ, p = 6 :=
sorry

end part1_part2_l607_607428


namespace wendy_made_money_l607_607736

-- Given conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 9
def bars_sold : ℕ := total_bars - 3

-- Statement to prove: Wendy made $18
theorem wendy_made_money : bars_sold * price_per_bar = 18 := by
  sorry

end wendy_made_money_l607_607736


namespace Sn_100_l607_607141

noncomputable def floor (x : ℝ) : ℤ :=
  int.floor x

def a (x : ℝ) (n : ℕ) : ℤ :=
  floor (10^n * x) - 10 * floor (10^(n-1) * x)

def b (x : ℝ) (k : ℕ) (n : ℕ) : ℤ :=
  floor ((a x n + 1) / (k + 1)) - floor ((a x n + 1) / (k + 1.01))

def S (x : ℝ) (k n : ℕ) : ℤ :=
  (finset.range n).sum (b x k)

theorem Sn_100 : S (1/7) 7 100 = 16 :=
  by
  sorry

end Sn_100_l607_607141


namespace complex_modulus_squared_l607_607980

noncomputable def modulus (w : ℂ) : ℝ := complex.abs w

theorem complex_modulus_squared (w : ℂ) (h : w + 3 * modulus w = -1 + 12 * complex.I) : modulus w ^ 2 = 2545 :=
  by
  sorry

end complex_modulus_squared_l607_607980


namespace radius_of_circle_zero_l607_607760

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end radius_of_circle_zero_l607_607760


namespace convex_polyhedron_triangular_face_or_tetrahedral_angle_l607_607551

/-- In any convex polyhedron, there exists either a triangular face or a tetrahedral angle (where exactly three edges meet). -/
theorem convex_polyhedron_triangular_face_or_tetrahedral_angle 
    (W : Polyhedron) 
    (h_convex : W.convex) : 
    (∃ face ∈ W.faces, face.is_triangle) ∨ (∃ vertex ∈ W.vertices, vertex.degree = 3) := 
sorry

end convex_polyhedron_triangular_face_or_tetrahedral_angle_l607_607551


namespace probability_yellow_ball_l607_607855

theorem probability_yellow_ball (x : ℕ) (h1 : 8 = 8) (h2 : 10 = 10)
    (h3 : (10 : ℚ) / (8 + 10 + x) = 1 / 4) :
  (x = 22) → (22 / (8 + 10 + 22 : ℚ) = 11 / 20) :=
by
  assume hx : x = 22
  rwa [hx]
  sorry

end probability_yellow_ball_l607_607855


namespace distinct_solutions_diff_l607_607153

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l607_607153


namespace distinct_elements_in_T_l607_607507

open Finset

noncomputable def a_k (k : ℕ) : ℕ := 3 * k - 1
noncomputable def b_l (l : ℕ) : ℕ := 7 * l
noncomputable def c_m (m : ℕ) : ℕ := 10 * m

noncomputable def A : Finset ℕ := (finset.range 1500).image a_k
noncomputable def B : Finset ℕ := (finset.range 1500).image b_l
noncomputable def C : Finset ℕ := (finset.range 1500).image c_m
noncomputable def T : Finset ℕ := A ∪ B ∪ C

theorem distinct_elements_in_T : T.card = 4061 := 
by {
  sorry
}

end distinct_elements_in_T_l607_607507


namespace problem_statement_l607_607508

noncomputable def a : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 1
| n + 1 := if (a n)^2 - (a (n - 1)^2 + a (n - 2)) < 0 then 0
           else if (a n)^2 - (a (n - 1)^2 + a (n - 2)) = 0 then if a n > 0 then 2 else 1
           else 4

theorem problem_statement :
  ∑ n in finset.range 100, a n = 105 :=
by
  sorry

end problem_statement_l607_607508


namespace quadratic_discriminant_positive_find_other_root_l607_607443

variable {m : ℝ}

theorem quadratic_discriminant_positive {a b c : ℝ} (h_eq : 1 = a) (h_b : b = m) (h_c : c = -3) :
  b^2 - 4 * a * c > 0 := by
  calc
    b^2 - 4 * a * c = m^2 - 4 * 1 * (-3) : by rw [h_eq, h_b, h_c]
    ... = m^2 + 12 : by norm_num
    ... > 0 : by apply add_pos_of_nonneg_of_pos; norm_num; exact pow_two_nonneg m

theorem find_other_root {a b c : ℝ} (h_eq : 1 = a) (h_b : b = m) (h_c : c = -3) 
  (root_3 : 3^2 + m * 3 - 3 = 0) : -1 = (-m - 3) := by
  calc
    3^2 + 3 * m - 3 = 0 : by exact root_3
    ... = 3m + 6 - 9 : by ring
    ... = (-m - 3) : by sorry

end quadratic_discriminant_positive_find_other_root_l607_607443


namespace fraction_of_seniors_study_japanese_l607_607692

-- Definitions for the problem
variables (J S : ℕ) (x : ℝ)
def juniors_study_japanese := (3 / 4 : ℝ) * J
def seniors_study_japanese := x * S
def total_students := (J + S : ℕ)
def total_study_japanese := (0.4375 : ℝ) * total_students

-- Main statement to prove
theorem fraction_of_seniors_study_japanese
  (h1 : S = 3 * J)
  (h2 : total_study_japanese = juniors_study_japanese + seniors_study_japanese) :
  x = 1 / 3 :=
by
  simp [total_study_japanese, juniors_study_japanese, seniors_study_japanese, *]
  sorry

end fraction_of_seniors_study_japanese_l607_607692


namespace general_form_equation_of_line_l_l607_607788

variable {A : Point}
variable {l : Line}
variable {y_intercept_line : ℝ}

-- Assume point A is (-2, 2)
def A := (-2, 2) : Point

-- Assume y_intercept_line is the y-intercept of the line y = x + 6, hence y_intercept_line = 6
def y_intercept_line := 6

-- Assume line l passes through point A and has the same y-intercept as the line y = x + 6
axiom passes_through_A : ∃ l : Line, l.contains A
axiom same_y_intercept : ∃ l : Line, l.y_intercept = y_intercept_line

-- Prove that the general form equation of the line l is 2x - y + 6 = 0
theorem general_form_equation_of_line_l :
  ∀ l : Line, l.contains A ∧ l.y_intercept = y_intercept_line → l.equation = 2 * x - y + 6 := 
sorry

end general_form_equation_of_line_l_l607_607788


namespace rank_percentage_changes_correct_l607_607324

noncomputable def percentage_change (old_price new_price: ℕ) : ℝ :=
  ((new_price - old_price) / old_price.to_real) * 100

def price_changes_conditions : Prop :=
  percentage_change 150 120 = -20 ∧
  percentage_change 120 130 = 8.33 ∧
  percentage_change 130 100 = -23.08 ∧
  percentage_change 100 80 = -20

def ranking_correct : Prop :=
  ∀ (changes: List ℝ), changes = [-23.08, -20, -20, 8.33] → 
     List.sort changes (λ x y, abs x > abs y) = [-23.08, -20, -20, 8.33]

theorem rank_percentage_changes_correct :
  price_changes_conditions → ranking_correct :=
by
  sorry

end rank_percentage_changes_correct_l607_607324


namespace transformed_mean_l607_607827

theorem transformed_mean (n : ℕ) (x : Fin n → ℝ) (h : (Finset.univ.sum x) / n = 5) :
  (Finset.univ.sum (λ i, 2 * x i + 1)) / n = 11 :=
sorry

end transformed_mean_l607_607827


namespace parabola_no_intersection_l607_607775

noncomputable def parabola_line_intersection (x0 y0 : ℝ) (h : y0^2 < 4 * x0) :
  Nat :=
if 4 * (y0^2 - 4 * x0) < 0
then 0
else sorry

theorem parabola_no_intersection (x0 y0: ℝ) (h: y0^2 < 4 * x0) :
  parabola_line_intersection x0 y0 h = 0 :=
begin
  sorry
end

end parabola_no_intersection_l607_607775


namespace min_distance_slope_range_l607_607019

-- Given curve C in polar coordinates
def curve_C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  in (x^2 + y^2 = 2)

-- Given line l in parametric equations, with α as an angle parameter
def line_l (Q : ℝ × ℝ) (α : ℝ) (t : ℝ) : Prop :=
  let (x, y) := Q
  in (x = 2 + t * (Real.cos α)) ∧ (y = 2 + t * (Real.sin α))

-- The angle α for which to find minimum |PQ|
def alpha := (3 / 4) * Real.pi

-- Minimum value of the line segment |PQ|
theorem min_distance (P Q : ℝ × ℝ) (t : ℝ) :
  curve_C P ∧ line_l Q alpha t → dist P Q = sqrt 2 :=
sorry

-- Range of the slope k when line l and curve C have two distinct intersection points
theorem slope_range (k : ℝ) (t : ℝ) :
  (∃ Q : ℝ × ℝ, line_l Q (Real.atan k) t) ∧ (∃ P1 P2 : ℝ × ℝ, curve_C P1 ∧ curve_C P2 ∧ P1 ≠ P2 ∧ dist P1 P2 ≠ 0) →
  k ∈ (Set.Ioo (2 - Real.sqrt 3) (2 + Real.sqrt 3)) :=
sorry

end min_distance_slope_range_l607_607019


namespace conic_section_is_hyperbola_l607_607364

theorem conic_section_is_hyperbola :
  ∀ (x y : ℝ), x^2 - 16 * y^2 - 8 * x + 16 * y + 32 = 0 → 
               (∃ h k a b : ℝ, h = 4 ∧ k = 0.5 ∧ a = b ∧ a^2 = 2 ∧ b^2 = 2) :=
by
  sorry

end conic_section_is_hyperbola_l607_607364


namespace product_of_factors_l607_607346

theorem product_of_factors :
  (∏ (n : ℕ) in finset.range 11 \{0}, (1 - (1 / (n + 2) : ℝ))) = (1 / 12 : ℝ) :=
by
  sorry

end product_of_factors_l607_607346


namespace investment_return_l607_607645

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l607_607645


namespace roots_of_equation_l607_607226

theorem roots_of_equation :
  ∀ x : ℝ, x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 :=
by {
  sorry
}

end roots_of_equation_l607_607226


namespace distinct_solution_difference_l607_607170

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l607_607170


namespace statement_A_statement_B_statement_C_statement_D_correct_statements_l607_607263

variable {α : Type*} {a b : α}

-- Statement A
theorem statement_A : ¬ (0 ∈ (∅ : set ℕ)) :=
by simp

-- Statement B
theorem statement_B : (∅ : set ℕ) ⊆ {0} :=
by simp

-- Statement C
theorem statement_C : {a, b} ⊆ ({b, a} : set α) :=
by simp

-- Statement D
theorem statement_D : ¬ (∅ ∈ ({0} : set (set ℕ))) :=
by simp

-- Combine B and C as correct statements
theorem correct_statements : (statement_B ∧ statement_C) = true :=
by exact ⟨statement_B, statement_C⟩

end statement_A_statement_B_statement_C_statement_D_correct_statements_l607_607263


namespace polynomial_horner_eval_v3_l607_607348

theorem polynomial_horner_eval_v3 :
  let f (x : ℤ) := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6 in
  let x := (-4 : ℤ) in
  let v0 := 2 in
  let v1 := 2 * x + 6 in
  let v2 := v1 * x + 1 in
  let v3 := v2 * x + 0 in
  v3 = -36 :=
by
  -- Proof will be here
  sorry

end polynomial_horner_eval_v3_l607_607348


namespace find_a_minus_b_l607_607994

noncomputable def initial_point := (0, 9)

def rotate_180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := center
  (2*h - x, 2*k - y)

def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem find_a_minus_b (hf : initial_point = (0, 9)) (final_pos : ℝ × ℝ) (h : final_pos = (-1, 4)) :
  let pr := reflect_y_neg_x final_pos in
  let pa := rotate_180 pr (2, 4) in
  pa = initial_point → (fst initial_point - snd initial_point) = -9 :=
by
  intros pr pa hpr hpa
  sorry

end find_a_minus_b_l607_607994


namespace number_of_buses_l607_607297

-- Definitions based on the given conditions
def vans : ℕ := 6
def people_per_van : ℕ := 6
def people_per_bus : ℕ := 18
def total_people : ℕ := 180

-- Theorem to prove the number of buses
theorem number_of_buses : 
  ∃ buses : ℕ, buses = (total_people - (vans * people_per_van)) / people_per_bus ∧ buses = 8 :=
by
  sorry

end number_of_buses_l607_607297


namespace fractional_eq_solution_l607_607583

theorem fractional_eq_solution (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) :
  (1 / (x - 1) = 2 / (x - 2)) → (x = 2) :=
by
  sorry

end fractional_eq_solution_l607_607583


namespace pipe_volume_equivalence_l607_607649

theorem pipe_volume_equivalence :
  ∀ (h : ℝ), let V12 := π * (6 ^ 2) * h, V3 := π * (1.5 ^ 2) * h in V12 / V3 = 16 :=
by
  intro h
  let V12 := π * (6 ^ 2) * h
  let V3 := π * (1.5 ^ 2) * h
  have hV : V12 / V3 = 36π h / 2.25π h := sorry
  have hDivide : 36 / 2.25 = 16 := sorry
  exact hV.symm.trans hDivide

end pipe_volume_equivalence_l607_607649


namespace lisa_time_calculation_l607_607320

-- Definitions based on the conditions
def sam_initial_distance : ℝ := 200
def sam_distance_moved : ℝ := 100
def sam_time : ℝ := 40
def lisa_behind_sam : ℝ := 10
def lisa_distance_moved : ℝ := 130

-- Additional calculations based on conditions
def sam_walking_rate : ℝ := sam_distance_moved / sam_time -- Sam's walking rate in feet per minute
def lisa_initial_distance : ℝ := sam_initial_distance + lisa_behind_sam -- Initial distance for Lisa
def lisa_walking_rate : ℝ := lisa_distance_moved / sam_time -- Lisa's walking rate in feet per minute

-- Remaining and half-distance calculations
def lisa_remaining_distance : ℝ := lisa_initial_distance - lisa_distance_moved
def lisa_half_remaining_distance : ℝ := lisa_remaining_distance / 2

-- Time required for Lisa to cover half the remaining distance
def lisa_time_to_cover_half_remaining : ℝ := lisa_half_remaining_distance / lisa_walking_rate

-- Prove the correct time is 12.31 minutes
theorem lisa_time_calculation : lisa_time_to_cover_half_remaining = 12.31 :=
sorry

end lisa_time_calculation_l607_607320


namespace g_monotonic_decreasing_F_min_value_l607_607512

noncomputable def f(x : ℝ) : ℝ := sorry -- Placeholder for the odd function that is decreasing on [-7, -3] and has max value -5

def g (a : ℝ) (x : ℝ) : ℝ := (ax + 1) / (x + 2)

theorem g_monotonic_decreasing (a : ℝ) (h : a < 0.5) : ∀ x1 x2 : ℝ, -2 < x1 → x1 < x2 → g a x2 < g a x1 :=
by 
  -- Proof omitted
  sorry

noncomputable def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := f(x) + g(x)

theorem F_min_value (a : ℝ) (h : a < 0.5) : ∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ F f (g a) x = (7*a + 46) / 9 :=
by 
  -- Proof omitted
  sorry

end g_monotonic_decreasing_F_min_value_l607_607512


namespace perfect_squares_less_than_100_l607_607074

theorem perfect_squares_less_than_100 :
  (∃ (n : ℕ), n = (finset.card (finset.filter (λ x, ∃ k : ℕ, x = k^2) (finset.range 100))) ∧ n = 9) :=
sorry

end perfect_squares_less_than_100_l607_607074


namespace exists_sequence_for_sum_l607_607504

theorem exists_sequence_for_sum (d S : ℤ) (hd : d > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ (epsilon : fin n → ℤ), (∀ i, epsilon i = 1 ∨ epsilon i = -1) ∧
    S = ∑ i in finset.range n, epsilon ⟨i, nat.lt_of_lt_succ (finset.mem_range_succ.mp (finset.mem_range.mpr (nat.lt_of_lt_of_le (nat.lt_succ_self _) (nat.succ_le_of_lt hd))))⟩ * (1 + i * d) ^ 2 := sorry

end exists_sequence_for_sum_l607_607504


namespace sum_of_ages_l607_607590

theorem sum_of_ages (youngest_age : ℕ) (interval : ℕ) (num_children : ℕ)
  (h1 : youngest_age = 8)
  (h2 : interval = 3)
  (h3 : num_children = 5) :
  let ages := (List.range num_children).map (λ i, youngest_age + i * interval) in
  ages.sum = 70 :=
by
  sorry

end sum_of_ages_l607_607590


namespace round_robin_tournament_10_players_3_draws_l607_607868

theorem round_robin_tournament_10_players_3_draws :
  ∀ (players : ℕ) (draws : ℕ),
  players = 10 → draws = 3 →
  (players * (players - 1)) / 2 = 45 :=
by
  intros players draws h_players h_draws
  rw [h_players, h_draws]
  sorry

end round_robin_tournament_10_players_3_draws_l607_607868


namespace smallest_multiplier_for_perfect_square_l607_607765

theorem smallest_multiplier_for_perfect_square (n : ℕ) (h1 : n = 1152) 
  (h2 : ∀ a, 1152 = 2^7 * 3^2) :
  ∃ m, m = 2 ∧ ∃ k, (n * m = k^2) :=
by
  sorry

end smallest_multiplier_for_perfect_square_l607_607765


namespace polynomial_factor_l607_607995

theorem polynomial_factor (b a: ℝ) 
  (h₁ : 3x^4 + b * x^3 + 45x^2 - 21x + 8 = (2x^2 - 3x + 2) * (a * x^2 + (-9) * x + 4)) : 
  (a = 3) ∧ (b = -27) :=
sorry

end polynomial_factor_l607_607995


namespace intersecting_lines_l607_607025

-- We declare the points and the triangle
variables (A B C M : Point)
variables (A1 A2 B1 B2 C1 C2 : Point)

-- Conditions based on the problem description
def angle_bisector_intersection_on_AB (C1 C2 : Point) : Prop :=
  are_bisectors_intersecting (angle A M B) A B C1 C2
  
def angle_bisector_intersection_on_BC (A1 A2 : Point) : Prop :=
  are_bisectors_intersecting (angle B M C) B C A1 A2
  
def angle_bisector_intersection_on_CA (B1 B2 : Point) : Prop :=
  are_bisectors_intersecting (angle C M A) C A B1 B2

-- The final proof problem
theorem intersecting_lines :
  (∀ M A B C A1 A2 B1 B2 C1 C2 : Point,
    angle_bisector_intersection_on_AB C1 C2 →
    angle_bisector_intersection_on_BC A1 A2 →
    angle_bisector_intersection_on_CA B1 B2 →
    ∃ l1 l2 l3 l4 : Line,
      (A1 ∈ l1 ∧ A2 ∈ l1 ∧ B1 ∈ l1) ∧
      (B2 ∈ l2 ∧ C1 ∈ l2 ∧ C2 ∈ l2) ∧
      (A1 ∈ l3 ∧ C2 ∈ l3 ∧ B2 ∈ l3) ∧
      (A2 ∈ l4 ∧ B1 ∈ l4 ∧ C1 ∈ l4)) :=
by {
  sorry
}

end intersecting_lines_l607_607025


namespace stan_words_per_minute_l607_607561

theorem stan_words_per_minute :
  (let pages := 5 in
   let words_per_page := 400 in
   let total_words := pages * words_per_page in
   let water_needed := 10 in
   let water_per_hour := 15 in
   let hours := water_needed / water_per_hour in
   let minutes := hours * 60 in
   total_words / minutes = 50) :=
sorry

end stan_words_per_minute_l607_607561


namespace basketball_player_practices_2_hours_each_day_l607_607650

axiom basketball_practice
  (average_hours_per_day : ℝ)
  (days_in_week : ℕ)
  (weekend_hours : ℝ)
  : average_hours_per_day = 3 →
    days_in_week = 7 →
    weekend_hours = 11 →
    ∃ weekday_hours_per_day : ℝ, weekday_hours_per_day = 2 ∧
    (average_hours_per_day * days_in_week = weekend_hours + (weekday_hours_per_day * 5))

-- Equivalent Problem rephrased statement
theorem basketball_player_practices_2_hours_each_day :
  ∀ (average_hours_per_day : ℝ)
    (days_in_week : ℕ)
    (weekend_hours : ℝ),
    average_hours_per_day = 3 →
    days_in_week = 7 →
    weekend_hours = 11 →
    ∃ weekday_hours_per_day : ℝ, weekday_hours_per_day = 2 ∧
    (average_hours_per_day * days_in_week = weekend_hours + (weekday_hours_per_day * 5)) := by
  intro average_hours_per_day days_in_week weekend_hours
  intros h1 h2 h3
  use 2
  split
  case right =>
    calc
      average_hours_per_day * days_in_week = 3 * 7 : by rw [h1, h2]
                                        _ = 21   : by norm_num
      _ = 11 + 10   : by rw [h3, add_comm]
      _ = 11 + (2 * 5) : by norm_num
      _ = 11 + 2 * 5 : by rw [mul_comm 2 5]
      _ = weekend_hours + 2 * 5 : by rw h3

end basketball_player_practices_2_hours_each_day_l607_607650


namespace polynomial_unique_R_l607_607381

noncomputable def identify_polynomial (P R : ℝ → ℝ) : Prop :=
∃ P : (ℝ → ℝ), ∀ t : ℝ, 
  7 * (sin t)^31 + 8 * (sin t)^18 - 5 * (sin t)^5 * (cos t)^4 - 10 * (sin t)^2 + 5 * (sin t)^5 - 2 =
  P (sin t) * ( (sin t)^4 - (1 + sin t) * ( (cos t)^2 - 2 )) + R (sin t)

theorem polynomial_unique_R :
  ∀ (R : ℝ → ℝ),
  (∀ x : ℝ, polynomial.degree (polynomial.C x • polynomial.X ≤ 3)) →
  (identify_polynomial (λ x, 13 * x^3 + 5 * x^2 + 12 * x + 3)) →
  R = (λ x, 13 * x^3 + 5 * x^2 + 12 * x + 3) :=
by
  intros R h_deg h_id
  sorry

end polynomial_unique_R_l607_607381


namespace distinct_solution_difference_l607_607169

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l607_607169


namespace magnitude_of_b_perpendicular_l607_607449

-- Definitions and conditions
def a (x : ℝ) : ℝ × ℝ := (x + 1, 2)
def b (x : ℝ) : ℝ × ℝ := (-1, x)

-- Theorem: Given the conditions, the magnitude of b for perpendicular vectors a and b
theorem magnitude_of_b_perpendicular (x : ℝ) (h : (x + 1) * (-1) + 2 * x = 0) :
  ∥b x∥ = Real.sqrt 2 :=
by
  sorry

end magnitude_of_b_perpendicular_l607_607449


namespace range_of_f_l607_607049

theorem range_of_f :
  ∀ x : ℝ, -π / 3 < x ∧ x < π / 3 → 
    (let f := λ x, (3 * Real.cos x + 1) / (2 - Real.cos x) 
     in ∃ y, f x = y ∧ (5/3 < y ∧ y ≤ 4)) := 
by
  sorry

end range_of_f_l607_607049


namespace number_of_integer_pairs_satisfying_conditions_l607_607073

theorem number_of_integer_pairs_satisfying_conditions
  : (∃ S : set (ℤ × ℤ), 
        S = { (a, b) | a^2 + b^2 < 25 ∧ a^2 + b^2 < 10 * a ∧ a^2 + b^2 < 10 * b } ∧ 
        fintype.card S = 20) :=
sorry

end number_of_integer_pairs_satisfying_conditions_l607_607073


namespace magnitude_of_sum_l607_607446

variable (a b : ℝ^3) -- Assuming vectors are in R^3 for simplicity

-- Condition: norms of vectors
def norm_a : ℝ := Real.sqrt 2
def norm_b : ℝ := 2

-- Condition: perpendicularity of (a - b) and a, implying orthogonality
def orthogonal_cond : (a - b) • a = 0

theorem magnitude_of_sum (ha : ‖a‖ = norm_a) (hb : ‖b‖ = norm_b) (horth : orthogonal_cond) : 
  ‖a + b‖ = Real.sqrt 10 :=
by
  sorry

end magnitude_of_sum_l607_607446


namespace vertex_of_quadratic_l607_607571

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, 3 * (x + 5) ^ 2 - 2 = 3 * (x - h) ^ 2 + k) ∧ (h, k) = (-5, -2) :=
by
  use -5, -2
  split
  · intro x
    sorry
  · refl

end vertex_of_quadratic_l607_607571


namespace cookies_left_l607_607936

def initial_cookies : ℕ := 93
def eaten_cookies : ℕ := 15

theorem cookies_left : initial_cookies - eaten_cookies = 78 := by
  sorry

end cookies_left_l607_607936


namespace students_not_opt_for_math_l607_607857

theorem students_not_opt_for_math (total_students S E both_subjects M : ℕ) 
    (h1 : total_students = 40) 
    (h2 : S = 15) 
    (h3 : E = 2) 
    (h4 : both_subjects = 7) 
    (h5 : total_students - both_subjects = M + S - E) : M = 20 := 
  by
  sorry

end students_not_opt_for_math_l607_607857


namespace tank_capacity_l607_607659

theorem tank_capacity (w c : ℕ) (h1 : w = c / 3) (h2 : w + 7 = 2 * c / 5) : c = 105 :=
sorry

end tank_capacity_l607_607659


namespace sum_of_z_values_l607_607517

/--
Let \( f \) be a function such that \( f\left(\frac{x}{2}\right) = x^2 + x + 2 \).
Find the sum of all values of \( z \) for which \( f(2z) = 10 \).
-/
theorem sum_of_z_values (f : ℝ → ℝ)
  (h : ∀ x, f (x / 2) = x^2 + x + 2) :
  (∑ z in { z : ℝ | f (2 * z) = 10 }, z) = -1/4 :=
by 
  sorry

end sum_of_z_values_l607_607517


namespace total_coins_are_correct_l607_607924

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l607_607924


namespace count_sequences_l607_607943

-- Problem statement: 
theorem count_sequences (n m : ℕ) :
  ∃ (k : ℕ), k = Nat.choose (n+1) (2*m+1) ∧
  let seq_count := λ (l : List ℕ), l.length = n ∧
                                    (∀ i, i < n → l.nth i ∈ (some 0, some 1)) ∧
                                    (list.countp (λ (i : ℕ), ∃ j < n-1, i = l.nth j + l.nth (j+1)) 1 1) = m in
                                    seq_count = k :=
by
  sorry

end count_sequences_l607_607943


namespace distinct_solutions_difference_l607_607151

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l607_607151


namespace pedal_triangle_angle_pedal_triangle_angle_equality_l607_607124

variables {A B C T_A T_B T_C: Type*}
variables {α β γ : Real}
variables {triangle : ∀ (A B C : Type*) (α β γ : Real), α ≤ β ∧ β ≤ γ ∧ γ < 90}

theorem pedal_triangle_angle
  (h : α ≤ β ∧ β ≤ γ ∧ γ < 90)
  (angles : 180 - 2 * α ≥ γ) :
  true :=
sorry

theorem pedal_triangle_angle_equality
  (h : α = β)
  (angles : (45 < α ∧ α = β ∧ α ≤ 60) ∧ (60 ≤ γ ∧ γ < 90)) :
  true :=
sorry

end pedal_triangle_angle_pedal_triangle_angle_equality_l607_607124


namespace sam_total_points_l607_607471

theorem sam_total_points (x y z : ℕ) (hx : 0 ≤ 0.25 * x) (hy : 0 ≤ 0.40 * y) (hz : 0 ≤ 0.80 * z)
  (h_total : x + y + z = 50) : 
  0.75 * x + 0.8 * y + 0.8 * z = 39 :=
by
  sorry

end sam_total_points_l607_607471


namespace all_permissible_triangles_generated_l607_607408

noncomputable def angle_form (m : ℕ) (p : ℕ) : ℝ := (m / p : ℝ) * 180

structure PermissibleTriangle (p : ℕ) :=
(angle1 angle2 angle3 : ℝ)
(sum_angles : angle1 + angle2 + angle3 = 180)
(angle_form1 : ∀ m : ℕ, angle1 = angle_form m p)
(angle_form2 : ∀ m : ℕ, angle2 = angle_form m p)
(angle_form3 : ∀ m : ℕ, angle3 = angle_form m p)

def initial_triangle (p : ℕ) : PermissibleTriangle p :=
{ angle1 := angle_form 1 p,
  angle2 := angle_form 1 p,
  angle3 := angle_form (p - 2) p,
  sum_angles := by simp [angle_form, mul_comm, (Nat.cast_add 1 1 : ℝ)],
  angle_form1 := by simp,
  angle_form2 := by simp,
  angle_form3 := by simp }

def divisible (T : PermissibleTriangle) (p : ℕ) : Prop :=
∃ m1 m2 m3 : ℕ, ∀ {n : ℕ}, T.angle1 = angle_form m1 p ∨ 
                     T.angle2 = angle_form m2 p ∨ 
                     T.angle3 = angle_form m3 p

theorem all_permissible_triangles_generated (p : ℕ) [fact (Nat.Prime p)] :
  ∀ (triangles : list (PermissibleTriangle p)) 
    (h₀ : triangles = [initial_triangle p])
    (h₁ : ∀ T ∈ triangles, divisible T p),
    (∀ T ∈ triangles, ∃ T' ∈ triangles, ¬(divisible T' p)) →
    ∀ T' : PermissibleTriangle p, T' ∈ triangles := 
  sorry

end all_permissible_triangles_generated_l607_607408


namespace slope_angle_tangent_line_at_x_eq_1_l607_607997

noncomputable def f (x : ℝ) := - (Real.sqrt 3 / 3) * x^3 + 2

theorem slope_angle_tangent_line_at_x_eq_1 :
  let slope := deriv f 1 in
  let angle := Real.angleOfSlope slope in
  angle = (2 / 3) * Real.pi :=
by
  sorry

end slope_angle_tangent_line_at_x_eq_1_l607_607997


namespace ellipse_with_foci_on_x_axis_l607_607013

variable (m : ℝ)

def curve (m : ℝ) : Prop := (2 - m) * x^2 + (m + 1) * y^2 = 1

theorem ellipse_with_foci_on_x_axis
  (hm : m ∈ set.Ioo (1/2 : ℝ) 2) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (curve m = (x^2 / a^2) + (y^2 / b^2) = 1)) :=
sorry

end ellipse_with_foci_on_x_axis_l607_607013


namespace equiangular_polygons_unique_solution_l607_607359

theorem equiangular_polygons_unique_solution :
  ∃! (n1 n2 : ℕ), (n1 ≠ 0 ∧ n2 ≠ 0) ∧ (180 / n1 + 360 / n2 = 90) :=
by
  sorry

end equiangular_polygons_unique_solution_l607_607359


namespace ellipse_equation_max_area_triangle_l607_607811

-- Definitions of conditions
def ellipse_eq (x y : ℝ) (a : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / 3 = 1

def eccentricity (a c : ℝ) : Prop :=
  c = a * (sqrt 2 / 2)

def focus_left (a : ℝ) : ℝ × ℝ :=
  (-sqrt (a * a - 3), 0)

def line_through_focus (m : ℝ) : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), p.1 = m * p.2 - sqrt 3

-- The questions to prove
theorem ellipse_equation (a : ℝ) (h : a > sqrt 3) (c : ℝ) (h_ell : eccentricity a c) :
  ellipse_eq 1 0 sqrt 6 :=
sorry

theorem max_area_triangle (a : ℝ) (h : a > sqrt 3) (c : ℝ) (h_ell : eccentricity a c)
  (F_1 := focus_left a) (line_eq := line_through_focus 0) :
  let l := λ (y : ℝ), (0 * y - sqrt 3, y) in
  ∃ (y1 y2 : ℝ), y1 + y2 = 0 → y1 * y2 = -3 / 1 →
  area (1, 0) F_1 (1, y1) + area (1, 0) F_1 (1, y2) = 3 + 3 * sqrt 2 / 2 ∧
  (0 * y1 - sqrt 3) = (0 * y2 - sqrt 3) :=
sorry

end ellipse_equation_max_area_triangle_l607_607811


namespace bird_speed_undetermined_l607_607280

-- Definitions for points, speed, and distance
def Point := Type
def B : Point := sorry
def R : Point := sorry
def train_speed := 60 -- Speed of each train in km/hr
def bird_distance := 120 -- Distance traveled by bird in km

-- Define the statement, implying we cannot determine the bird's speed with given data alone
theorem bird_speed_undetermined (bird_distance: nat) (train_speed: nat) (bird_speed: nat) : 
  bird_distance ≠ 0 → train_speed ≠ 0 → bird_speed = bird_speed :=
by
  sorry
  
-- The given conditions imply the impossibility of determining the bird's speed since 
-- the requisite time information is not provided.

end bird_speed_undetermined_l607_607280


namespace students_moved_outside_correct_l607_607592

noncomputable def students_total : ℕ := 90
noncomputable def students_cafeteria_initial : ℕ := (2 * students_total) / 3
noncomputable def students_outside_initial : ℕ := students_total - students_cafeteria_initial
noncomputable def students_ran_inside : ℕ := students_outside_initial / 3
noncomputable def students_cafeteria_now : ℕ := 67
noncomputable def students_moved_outside : ℕ := students_cafeteria_initial + students_ran_inside - students_cafeteria_now

theorem students_moved_outside_correct : students_moved_outside = 3 := by
  sorry

end students_moved_outside_correct_l607_607592


namespace point_label_is_T_l607_607485

-- Let us define the coordinates for points P, Q, R, S, T
def P : (ℝ × ℝ) := (x₁, y₁)
def Q : (ℝ × ℝ) := (x₂, y₂)
def R : (ℝ × ℝ) := (x₃, y₃)
def S : (ℝ × ℝ) := (x₄, y₄)
def T : (ℝ × ℝ) := (3, -4)

-- The proof goal is to show that the point (3, -4) corresponds to the label T
theorem point_label_is_T : (3, -4) = T := 
by 
    sorry

end point_label_is_T_l607_607485


namespace find_a_plus_b_l607_607908

theorem find_a_plus_b (a b : ℝ) (h : Polynomial.root (Polynomial.X^3 + Polynomial.C a * Polynomial.X + Polynomial.C b) (2 + complex.I * real.sqrt 2)) : 
  a + b = 14 := 
sorry

end find_a_plus_b_l607_607908


namespace minimalYellowFraction_l607_607658

-- Definitions
def totalSurfaceArea (sideLength : ℕ) : ℕ := 6 * (sideLength * sideLength)

def minimalYellowExposedArea : ℕ := 15

theorem minimalYellowFraction (sideLength : ℕ) (totalYellow : ℕ) (totalBlue : ℕ) 
    (totalCubes : ℕ) (yellowExposed : ℕ) :
    sideLength = 4 → totalYellow = 16 → totalBlue = 48 →
    totalCubes = 64 → yellowExposed = minimalYellowExposedArea →
    (yellowExposed / (totalSurfaceArea sideLength) : ℚ) = 5 / 32 :=
by
  sorry

end minimalYellowFraction_l607_607658


namespace problem_solution_l607_607377

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n % 100) / 10) * 8 + (n % 10)

def base3_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 3 + (n % 10)

def base7_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 49 + ((n % 100) / 10) * 7 + (n % 10)

def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

def expression_in_base10 : ℕ :=
  (base8_to_base10 254) / (base3_to_base10 13) + (base7_to_base10 232) / (base5_to_base10 32)

theorem problem_solution : expression_in_base10 = 35 :=
by
  sorry

end problem_solution_l607_607377


namespace velocity_at_t_eq_2_is_13_over_4_l607_607815

-- 1. Define the equation of motion.
def equation_of_motion (t : ℝ) : ℝ := t^2 + 3/t

-- 2. Define the velocity as the time derivative of the equation of motion.
noncomputable def velocity (t : ℝ) : ℝ := (deriv equation_of_motion) t

-- 3. State the theorem: The velocity at t=2 is 13/4.
theorem velocity_at_t_eq_2_is_13_over_4 : velocity 2 = 13/4 :=
sorry

end velocity_at_t_eq_2_is_13_over_4_l607_607815


namespace billy_cherries_l607_607342

def initial_cherries : ℝ := 3682.5
def eaten_cherries : ℝ := 2147.25
def gift_cherries : ℝ := 128.5
def fraction_given_away : ℝ := 3 / 5

theorem billy_cherries : 
  initial_cherries - eaten_cherries - fraction_given_away * (initial_cherries - eaten_cherries) + gift_cherries = 742.6 := 
by
  sorry

end billy_cherries_l607_607342


namespace four_lines_circumcircles_common_point_l607_607776

open EuclideanGeometry

theorem four_lines_circumcircles_common_point
  (A B C D E F: Point)
  (l1 l2 l3 l4 : Line)
  (h1 : A ∈ l1) (h2 : B ∈ l1)
  (h3 : D ∈ l2) (h4 : E ∈ l2)
  (h5 : C ∈ l3) (h6 : F ∈ l3)
  (h7 : A ∈ l4) (h8 : E ∈ l4)
  (h9 : B ∈ l4) (h10 : D ∈ l4)
  (h11 : ∃ P, P ∈ l1 ∧ P ∈ l2)
  (h12 : ∃ Q, Q ∈ l3 ∧ Q ∈ l4) :
  ∃ P, 
    P ∈ circumcircle (triangle A E C) ∧ 
    P ∈ circumcircle (triangle B D C) ∧ 
    P ∈ circumcircle (triangle A B F) ∧ 
    P ∈ circumcircle (triangle E D F) :=
sorry

end four_lines_circumcircles_common_point_l607_607776


namespace find_g_function_l607_607214

noncomputable def g : ℝ → ℝ :=
  sorry

theorem find_g_function (x y : ℝ) (h1 : g 1 = 2) (h2 : ∀ (x y : ℝ), g (x + y) = 5^y * g x + 3^x * g y) :
  g x = 5^x - 3^x :=
by
  sorry

end find_g_function_l607_607214


namespace find_a_b_find_min_g_l607_607441

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - x^2 + a * x + b

theorem find_a_b (a b : ℝ) (h_tangent : ∀ x, f x a b = -x + 1 → x = 0) : a = -1 ∧ b = 1 :=
by
  have h₀ : f 0 a b = b := by simp [f]
  have h₁ : -0 + 1 = 1 := by norm_num
  have h₂ : b = 1 := by rw [←h_tangent 0 h₁]
  have hf' : ∀ x, deriv (f x a b) = 3 * x^2 - 2 * x + a := by simp [f, deriv]
  have h₃ : deriv (f 0 a b) = a := by simp [hf']
  have h_slope : ∀ x, deriv (λ x, -x + 1) x = -1 := by simp [deriv]
  have h₄ : a = -1 := by rw [←h_slope 0, h₃]
  exact ⟨h₄, h₂⟩

noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem find_min_g : ∃ x ∈ Icc (-2 : ℝ) 2, g x = -9 :=
by
  have eval_g : ∀ x, g x = x^3 - x^2 - x + 1 := by simp [g]
  have h_g_neg2 : g (-2) = -9 := by norm_num [g, eval_g]
  use -2
  split
  · norm_num
  · exact h_g_neg2

end find_a_b_find_min_g_l607_607441


namespace number_of_students_l607_607624

theorem number_of_students (N : ℕ) (h1 : (1/5 : ℚ) * N + (1/4 : ℚ) * N + (1/2 : ℚ) * N + 5 = N) : N = 100 :=
by
  sorry

end number_of_students_l607_607624


namespace width_of_tank_is_4_l607_607605

-- Variables and conditions
variables (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ)

-- Given values
def rate : ℝ := 4 -- cubic feet per hour
def time : ℝ := 18 -- hours
def length : ℝ := 6 -- feet
def depth : ℝ := 3 -- feet

-- We need to prove that the width is 4 feet
theorem width_of_tank_is_4 : 
  let volume := rate * time in 
  let width := volume / (length * depth) in 
  width = 4 := 
by
  -- Theorem proof
  sorry

end width_of_tank_is_4_l607_607605


namespace registered_voters_democrats_l607_607100

variables (D R : ℝ)

theorem registered_voters_democrats :
  (D + R = 100) →
  (0.80 * D + 0.30 * R = 65) →
  D = 70 :=
by
  intros h1 h2
  sorry

end registered_voters_democrats_l607_607100


namespace smallest_even_in_sequence_sum_400_l607_607970

theorem smallest_even_in_sequence_sum_400 :
  ∃ (n : ℤ), (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 400 ∧ (n - 6) % 2 = 0 ∧ n - 6 = 52 :=
sorry

end smallest_even_in_sequence_sum_400_l607_607970


namespace zou_mei_competition_l607_607874

theorem zou_mei_competition (n : ℕ) (h1 : 271 = n^2 + 15) (h2 : n^2 + 33 = (n + 1)^2) : 
  ∃ n, 271 = n^2 + 15 ∧ n^2 + 33 = (n + 1)^2 :=
by
  existsi n
  exact ⟨h1, h2⟩

end zou_mei_competition_l607_607874


namespace valid_9_digit_numbers_count_l607_607451

def is_valid_digit (d : Nat) : Prop :=
  d < 10 ∧ d ≠ 5

def is_valid_number (n : Nat) : Prop :=
  ∃ digits : List Nat, 
    digits.length = 9 ∧
    (∀ d ∈ digits, is_valid_digit d) ∧
    n = digits.foldl (λ acc d, acc * 10 + d) 0

noncomputable def count_valid_numbers : Nat :=
  9 ^ 9

theorem valid_9_digit_numbers_count : 
  ∃ count, count = count_valid_numbers ∧ count_valid_numbers = 387420489 :=
by
  use count_valid_numbers
  split
  { rfl }
  { sorry }

end valid_9_digit_numbers_count_l607_607451


namespace logarithmic_problem_l607_607010

theorem logarithmic_problem (x : ℝ) (h : log 7 (log 3 (log 2 x)) = 0) : x^(-1/2) = (Real.sqrt 2) / 4 :=
by
  sorry

end logarithmic_problem_l607_607010


namespace interest_rate_correct_l607_607250

namespace InterestProblem

variable (P : ℤ) (SI : ℤ) (T : ℤ)

def rate_of_interest (P : ℤ) (SI : ℤ) (T : ℤ) : ℚ :=
  (SI * 100) / (P * T)

theorem interest_rate_correct :
  rate_of_interest 400 140 2 = 17.5 := by
  sorry

end InterestProblem

end interest_rate_correct_l607_607250


namespace sum_of_powers_divisibility_l607_607048

noncomputable def s_n (x1 x2 : ℝ) (n : ℕ) : ℝ :=
  x1^n + x2^n

theorem sum_of_powers_divisibility (p q r k n : ℕ) 
  (hpqgcd : Nat.gcd p q = r) (hrgt1 : r > 1) 
  (hnat : n ∈ (Nat.range (n + 1))) 
  (hpowers : k ≤ Nat.floor ((n + 1) / 2)) :
  ∀ (x1 x2 : ℝ), 
  (x1^2 + p * x1 + q = 0) ∧ (x2^2 + p * x2 + q = 0) → 
  r^k ∣ s_n x1 x2 n :=
by
  sorry

end sum_of_powers_divisibility_l607_607048


namespace particle_speed_properties_l607_607306

-- Definition of the motion law
def motion_law (t : ℝ) : ℝ := 2 * t^3

-- Definition of the average speed function
def average_speed (s : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  (s t2 - s t1) / (t2 - t1)

-- Definition of the instantaneous speed function
def instantaneous_speed (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  deriv s t

-- The theorem we aim to prove (statement only, no proof)
theorem particle_speed_properties :
  average_speed motion_law 1 2 = 14 ∧ instantaneous_speed motion_law 1 = 6 :=
by
  -- Placeholder for the proof
  sorry

end particle_speed_properties_l607_607306


namespace find_a_plus_b_l607_607909

theorem find_a_plus_b (a b : ℝ) (h : Polynomial.root (Polynomial.X^3 + Polynomial.C a * Polynomial.X + Polynomial.C b) (2 + complex.I * real.sqrt 2)) : 
  a + b = 14 := 
sorry

end find_a_plus_b_l607_607909


namespace pounds_of_sugar_l607_607663

theorem pounds_of_sugar (x p : ℝ) (h1 : x * p = 216) (h2 : (x + 3) * (p - 1) = 216) : x = 24 :=
sorry

end pounds_of_sugar_l607_607663


namespace uniforms_needed_for_heights_163_183_l607_607299

noncomputable def num_uniforms_needed 
  (mean : ℝ) (std_dev : ℝ) (total_employees : ℕ) (lower_bound : ℝ) (upper_bound : ℝ) : ℕ :=
  let variance := std_dev^2
  let distribution := measure_theory.normal mean variance
  let prop_within_bounds := measure_theory.measure.within_Icc lower_bound upper_bound distribution
  (prop_within_bounds * total_employees).round

theorem uniforms_needed_for_heights_163_183 :
  num_uniforms_needed 173 5 10000 163 183 = 9544 :=
sorry

end uniforms_needed_for_heights_163_183_l607_607299


namespace non_perfect_power_probability_l607_607580

-- Definition: A perfect power is an integer x^y where x is an integer and y > 1.
def is_perfect_power (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ x^y = n

-- Set of numbers from 1 to 200.
def numbers := finset.range 201

-- Count the perfect powers and non-perfect powers in the given range.
def count_perfect_powers (s : finset ℕ) : ℕ :=
  (s.filter is_perfect_power).card

def count_non_perfect_powers (s : finset ℕ) : ℕ :=
  s.card - count_perfect_powers s

-- Calculate the probability of selecting a non-perfect power.
def probability_non_perfect_power (s : finset ℕ) : ℚ :=
  count_non_perfect_powers s / s.card

-- The main theorem statement.
theorem non_perfect_power_probability :
  probability_non_perfect_power numbers = 9 / 10 :=
by sorry

end non_perfect_power_probability_l607_607580


namespace regular_polygon_sides_l607_607865

theorem regular_polygon_sides (n : ℕ) (h : 2 < n)
  (interior_angle : ∀ n, (n - 2) * 180 / n = 144) : n = 10 :=
sorry

end regular_polygon_sides_l607_607865


namespace oil_already_put_in_engine_l607_607396

def oil_per_cylinder : ℕ := 8
def cylinders : ℕ := 6
def additional_needed_oil : ℕ := 32

theorem oil_already_put_in_engine :
  (oil_per_cylinder * cylinders) - additional_needed_oil = 16 := by
  sorry

end oil_already_put_in_engine_l607_607396


namespace unique_zero_of_f_l607_607821

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (-x + 1))

theorem unique_zero_of_f (a : ℝ) : (∃! x, f x a = 0) ↔ a = 1 / 2 := sorry

end unique_zero_of_f_l607_607821


namespace sum_of_104th_group_l607_607689

theorem sum_of_104th_group :
  let seq := λ n : ℕ, (2 * n + 1)
  let group_sizes := list.cycle ([1, 2, 3, 4, 1])
  let group_start := ∑ i in (list.fin_range 103), group_sizes.get_or_else i 0
  let first_in_group := seq (group_start + 1)
  let group := list.init (group_sizes.get_or_else 103 0) (λ i, first_in_group + 2 * i)
  (list.sum group) = 2464 :=
by
  sorry

end sum_of_104th_group_l607_607689


namespace area_of_triangle_ABC_l607_607558

theorem area_of_triangle_ABC : 
  let side_length := 1
  let circumradius := side_length
  let centers_triangle_side := 2 * circumradius
  let perimeter := 3 * centers_triangle_side
  let semiperimeter := perimeter / 2
  let inradius := circumradius
  let triangle_area := inradius * semiperimeter
  triangle_area = 3 * Real.sqrt 3 :=
by
  let side_length := 1
  let circumradius := side_length
  let centers_triangle_side := 2 * circumradius
  let perimeter := 3 * centers_triangle_side
  let semiperimeter := perimeter / 2
  let inradius := circumradius
  let triangle_area := inradius * semiperimeter
  have h: triangle_area = 3 * Real.sqrt 3
  exact h

end area_of_triangle_ABC_l607_607558


namespace problem_equivalent_proof_l607_607414

-- Condition definitions
def circleM (x y : ℝ) := 2 * x^2 + 2 * y^2 - 8 * x - 8 * y - 1 = 0
def circleN (x y : ℝ) := x^2 + y^2 + 2 * x + 2 * y - 6 = 0
def lineL (x y : ℝ) := x + y - 9 = 0
def origin := (0, 0)

-- Conclusion definitions
def equationOfCircle (x y : ℝ) := x^2 + y^2 - (50 / 11) * x - (50 / 11) * y = 0

def pointA_onLineL (x y : ℝ) := x = 4 ∧ y = 5
def centerM := (2, 2)

def lineAC_case1 (x y : ℝ) := 5 * x + y - 25 = 0
def lineAC_case2 (x y : ℝ) := x - 5 * y + 21 = 0

def range_x_coordinates_for_A := 3 ≤ x ∧ x ≤ 6

-- Complete statement
theorem problem_equivalent_proof :
  (∀ x y : ℝ, circleM x y → circleN x y → equationOfCircle x y) ∧
  pointA_onLineL 4 5 ∧
  (∀ x y : ℝ, (pointA_onLineL x y → 
    ( lineAC_case1 x y ∨ lineAC_case2 x y))) ∧
  (∀ x, lineL x (9 - x) → range_x_coordinates_for_A x) :=
by sorry

end problem_equivalent_proof_l607_607414


namespace next_term_in_geometric_sequence_l607_607608

theorem next_term_in_geometric_sequence : 
  ∀ (x : ℕ), (∃ (a : ℕ), a = 768 * x^4) :=
by
  sorry

end next_term_in_geometric_sequence_l607_607608


namespace find_a_l607_607051

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x^2 + 3)

def derivative_at_one (f' : ℝ) : Prop := f' = 1/2

theorem find_a (a : ℝ) (h1 : derivative_at_one 
  (deriv (λ x : ℝ, f a x) 1)) : a = 4 :=
by {
  sorry
}

end find_a_l607_607051


namespace angela_problems_l607_607330

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end angela_problems_l607_607330


namespace graphs_symmetric_about_origin_l607_607367

noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := -2^(-x)

theorem graphs_symmetric_about_origin :
  (∀ x : ℝ, f (-x) = -g (x)) →
  (∀ x y, y = f x ↔ -y = g (-x)) :=
begin
  intros h x y,
  split,
  { intro h1,
    rw h1,
    apply h, },
  { intro h2,
    rw ←h at h2,
    rw neg_neg at h2,
    exact h2, }
end

end graphs_symmetric_about_origin_l607_607367


namespace total_fruits_correct_l607_607496

def total_fruits 
  (Jason_watermelons : Nat) (Jason_pineapples : Nat)
  (Mark_watermelons : Nat) (Mark_pineapples : Nat)
  (Sandy_watermelons : Nat) (Sandy_pineapples : Nat) : Nat :=
  Jason_watermelons + Jason_pineapples +
  Mark_watermelons + Mark_pineapples +
  Sandy_watermelons + Sandy_pineapples

theorem total_fruits_correct :
  total_fruits 37 56 68 27 11 14 = 213 :=
by
  sorry

end total_fruits_correct_l607_607496


namespace fifteenth_number_is_5319_l607_607594

def digits : List ℕ := [1, 3, 5, 9]

def digit_permutations := digits.permutations.filter (λ l, l.length = 4)

def sorted_permutations := digit_permutations.sort (λ l₁ l₂, l₁ < l₂)

def find_15th_number : List ℕ := sorted_permutations.get (15 - 1) -- list indices are 0-based

theorem fifteenth_number_is_5319 :
  find_15th_number = [5, 3, 1, 9] := 
sorry

end fifteenth_number_is_5319_l607_607594


namespace divisors_of_10_factorial_greater_than_9_factorial_l607_607072

def is_divisor (n d : ℕ) : Prop :=
  d ∣ n

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

def number_of_divisors_greater_than (n m : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ d => (is_divisor (factorial n) d) ∧ (d > m)).length

theorem divisors_of_10_factorial_greater_than_9_factorial : number_of_divisors_greater_than 10 (factorial 9) = 9 := 
sorry

end divisors_of_10_factorial_greater_than_9_factorial_l607_607072


namespace binomial_7_4_eq_35_l607_607712
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607712


namespace dartboard_points_proof_l607_607362

variable (points_one points_two points_three points_four : ℕ)

theorem dartboard_points_proof
  (h1 : points_one = 30)
  (h2 : points_two = 38)
  (h3 : points_three = 41)
  (h4 : 2 * points_four = points_one + points_two) :
  points_four = 34 :=
by {
  sorry
}

end dartboard_points_proof_l607_607362


namespace smallest_n_for_violet_candy_l607_607737

theorem smallest_n_for_violet_candy (p y o n : Nat) (h : 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) :
  n = 8 :=
by 
  sorry

end smallest_n_for_violet_candy_l607_607737


namespace abs_val_of_2_l607_607208

theorem abs_val_of_2 : abs 2 = 2 :=
by
  sorry

end abs_val_of_2_l607_607208


namespace base_height_calculation_l607_607702

noncomputable def height_of_sculpture : ℚ := 2 + 5/6 -- 2 feet 10 inches in feet
noncomputable def total_height : ℚ := 3.5
noncomputable def height_of_base : ℚ := 2/3

theorem base_height_calculation (h1 : height_of_sculpture = 17/6) (h2 : total_height = 21/6):
  height_of_base = total_height - height_of_sculpture := by
  sorry

end base_height_calculation_l607_607702


namespace ξ_distribution_and_expectation_probability_harvested_by_team_B_l607_607973

/-- Define the teams and their respective probabilities -/
def p_team_A : ℚ := 3 / 10
def p_team_B : ℚ := 3 / 10
def p_team_C : ℚ := 2 / 5

/-- Define the utilization rates for each team -/
def p_use_A : ℚ := 8 / 10
def p_use_B : ℚ := 75 / 100
def p_use_C : ℚ := 6 / 10

/-- Random variable ξ following a binomial distribution -/
def ξ_distribution (k : ℕ) : ℚ :=
nat.choose 3 k * (p_team_C^k) * ((1 - p_team_C)^(3 - k))

/-- Distribution and expectation of ξ -/
theorem ξ_distribution_and_expectation :
  ∀ k : ℕ, k ∈ [0, 1, 2, 3] →
  ξ_distribution k =
    match k with
    | 0 => 27 / 125
    | 1 => 54 / 125
    | 2 => 36 / 125
    | 3 => 8 / 125
    | _ => 0
    end
∧
(Eξ : ℚ) = 6 / 5 :=
sorry

/-- Probability a block was harvested by Team B and can be used -/
def P_B := p_team_A * p_use_A + p_team_B * p_use_B + p_team_C * p_use_C

/-- Probability a block was harvested by Team B given it can be used -/
theorem probability_harvested_by_team_B :
  (p_team_B * p_use_B) / P_B = 15 / 47 :=
sorry

end ξ_distribution_and_expectation_probability_harvested_by_team_B_l607_607973


namespace tangent_parallel_line_l607_607427

theorem tangent_parallel_line (a : ℝ) : 
  (∀ x : ℝ, y x = (x + 3) / (x - 1)) →
  (∀ x : ℝ, point_on_curve x := (2, 5)) →
  tangent_at (2, 5) is_parallel_to (ax + y - 1 = 0) →
  a = 4 := 
  by
  -- Definitions
  let tangent_slope := -4
  let line_slope := -a
  sorry -- Proof goes here

end tangent_parallel_line_l607_607427


namespace computer_price_decrease_l607_607616

theorem computer_price_decrease (P₀ : ℝ) (years : ℕ) (decay_rate : ℝ) (P₆ : ℝ) :
  P₀ = 8100 →
  decay_rate = 2 / 3 →
  years = 6 →
  P₆ = P₀ * decay_rate ^ (years / 2) →
  P₆ = 2400 :=
begin
  intros h₀ h₁ h₂ h₃,
  sorry
end

end computer_price_decrease_l607_607616


namespace interest_rate_example_l607_607340

noncomputable def annual_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  (A / P) ^ (1 / (n * t)) - 1

theorem interest_rate_example :
  annual_interest_rate 25000 45000 1 10 ≈ 0.059463094 :=
by
  sorry

end interest_rate_example_l607_607340


namespace eccentricity_of_ellipse_G_equation_of_line_l_l607_607797

-- Definition of an ellipse
structure Ellipse (a b : ℝ) :=
(eq : ∀ (x y : ℝ), x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)

-- Points on the ellipse
structure Point on (E : Ellipse) :=
(x : ℝ)
(y : ℝ)
(on_ellipse : E.eq x y)

-- Define the specific ellipse G given points A and B
noncomputable def G : Ellipse (2 * Real.sqrt 3) 2 :=
{ eq := λ x y, x ^ 2 / (2 * Real.sqrt 3) ^ 2 + y ^ 2 / 2 ^ 2 = 1 }

-- Define points A and B on the ellipse
def A : Point on G := { x := 0, y := 2, on_ellipse := by { dsimp [G], norm_num } }
def B : Point on G := { x := 3, y := 1, on_ellipse := by { dsimp [G], norm_num, ring } }

-- Verify the eccentricity of the ellipse G
theorem eccentricity_of_ellipse_G : (Real.sqrt 6) / 3 = Real.sqrt ((2 * Real.sqrt 2) ^ 2 / (2 * Real.sqrt 3) ^ 2) :=
begin
  sorry -- proof required here
end

-- Define lines passing through point B and their possible equations
def line1_through_B (l : ℝ → ℝ → Prop) : Prop := ∀ x y, l x y ↔ y = (-1 / 2) * (x - 3) + 1
def line2_through_B (l : ℝ → ℝ → Prop) : Prop := ∀ x y, l x y ↔ y = (1 / 9) * (x - 3) + 1

-- Prove that lines passing through point B and satisfying given circle condition
theorem equation_of_line_l (l : ℝ → ℝ → Prop) 
  (h1 : line1_through_B l ∨ line2_through_B l) : 
  l = (λ x y, x + 2 * y - 5 = 0) ∨ l = (λ x y, x - 9 * y + 6 = 0) :=
begin
  sorry -- proof required here
end

end eccentricity_of_ellipse_G_equation_of_line_l_l607_607797


namespace intersects_at_fixed_point_l607_607062

noncomputable def parabola_eq (x y : ℝ) : Prop := y = x^2
noncomputable def curve_eq (x y k : ℝ) : Prop := y = k * |x| - 2

theorem intersects_at_fixed_point (k x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (hk : k > 0) (hparabola1 : parabola_eq x1 y1) (hparabola2 : parabola_eq x2 y2)
  (hparabola3 : parabola_eq x3 y3) (hparabola4 : parabola_eq x4 y4)
  (hcurve1 : curve_eq x1 y1 k) (hcurve2 : curve_eq x2 y2 k)
  (hcurve3 : curve_eq x3 y3 k) (hcurve4 : curve_eq x4 y4 k)
  (h_parallel : y2 = y1 ∧ y4 = y3 ∧ (x1 - x2) ≠ 0 ∧ (x3 - x4) ≠ 0)
  (h_less : |x1 - x2| < |x3 - x4|) (h_ad_right : x1 > 0) :
  -- Proving the range of k
  k > 2 * real.sqrt 2 ∧
  -- Proving the fixed intersection point E
  ∃ E : ℝ × ℝ, E = (0, 2) := sorry

end intersects_at_fixed_point_l607_607062


namespace smallest_whole_number_greater_than_triangle_perimeter_l607_607252

theorem smallest_whole_number_greater_than_triangle_perimeter 
  (a b : ℝ) (h_a : a = 7) (h_b : b = 23) :
  ∀ c : ℝ, 16 < c ∧ c < 30 → ⌈a + b + c⌉ = 60 :=
by
  intros c h
  rw [h_a, h_b]
  sorry

end smallest_whole_number_greater_than_triangle_perimeter_l607_607252


namespace triangle_similarity_l607_607516

open EuclideanGeometry

/-- Given that G is the centroid of triangle ABC, M is the midpoint of BC,
X is on AB, Y is on AC, X, Y, G are collinear, XY is parallel to BC,
XC and GB intersect at Q, YB and GC intersect at P, show that triangle MPQ is similar to triangle ABC. -/
theorem triangle_similarity (A B C G M X Y Q P : Point) :
  is_centroid G A B C →
  midpoint M B C →
  online X A B →    -- X line on AB
  online Y A C →    -- Y line on AC
  collinear {X, Y, G} →  -- Collinearity of points X, Y, G
  parallel XY BC → 
  intersection Q XC GB →
  intersection P YB GC →
  similar (triangle M P Q) (triangle A B C) :=
sorry

end triangle_similarity_l607_607516


namespace problem_statement_l607_607401

-- Define the variables based on the given conditions
variables (a b c : ℝ)
def a_def : ℝ := 2^(0.3)
def b_def : ℝ := Real.log 0.3 / Real.log 2
def c_def : ℝ := 0.3^2

-- Specify the proof goal
theorem problem_statement : b < c ∧ c < a :=
by
  unfold a_def b_def c_def
  have ha : a_def = 2^(0.3), by refl
  have hb : b_def = Real.log 0.3 / Real.log 2, by refl
  have hc : c_def = 0.3^2, by refl
  sorry

end problem_statement_l607_607401


namespace exists_a_with_2018_distinct_terms_l607_607361

noncomputable def x_seq (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := if x_seq n = 0 then 0 else (x_seq n ^ 2 - 1) / (2 * x_seq n)

theorem exists_a_with_2018_distinct_terms :
  ∃ a : ℝ, 
    let seq := x_seq a 
    in (∀ i ≤ 2017, seq i ≠ 0) ∧ (seq 2018 = 0) ∧ 
       (∀ i j ≤ 2017, i ≠ j → seq i ≠ seq j) :=
begin
  use real.cot (real.pi / 2^2018),
  -- Proof goes here
  sorry,
end

end exists_a_with_2018_distinct_terms_l607_607361


namespace AI_tangent__l607_607117

variables {A B C I N P Q R : Type}

-- Assume triangle ABC is an acute triangle
variables [acute_triangle ABC]

-- Assume I is the incenter
variables (I : incenter ABC)

-- Assume N is the midpoint of the arc BAC on the circumcircle of ABC
variables [circumcircle_midpoint N A B C]

-- Assume P such that ABPC is a parallelogram
variables (P : parallelogram AB P C)

-- Assume Q is the reflection of A over N
variables (Q : reflection A N)

-- Assume R is the projection of A onto QI
variables (R : projection A (line_segment Q I))

theorem AI_tangent_ circumcircle_PQR :
  tangent (line_segment A I) (circumcircle P Q R) :=
sorry

end AI_tangent__l607_607117


namespace mechanical_pencils_fraction_l607_607861

theorem mechanical_pencils_fraction (total_pencils : ℕ) (frac_mechanical : ℚ)
    (mechanical_pencils : ℕ) (standard_pencils : ℕ) (new_total_pencils : ℕ) 
    (new_standard_pencils : ℕ) (new_frac_mechanical : ℚ):
  total_pencils = 120 →
  frac_mechanical = 1 / 4 →
  mechanical_pencils = frac_mechanical * total_pencils →
  standard_pencils = total_pencils - mechanical_pencils →
  new_standard_pencils = 3 * standard_pencils →
  new_total_pencils = mechanical_pencils + new_standard_pencils →
  new_frac_mechanical = mechanical_pencils / new_total_pencils →
  new_frac_mechanical = 1 / 10 :=
by
  sorry

end mechanical_pencils_fraction_l607_607861


namespace determine_price_reduction_l607_607287

noncomputable def initial_cost_price : ℝ := 220
noncomputable def initial_selling_price : ℝ := 280
noncomputable def initial_daily_sales_volume : ℕ := 30
noncomputable def price_reduction_increase_rate : ℝ := 3

variable (x : ℝ)

noncomputable def daily_sales_volume (x : ℝ) : ℝ := initial_daily_sales_volume + price_reduction_increase_rate * x
noncomputable def profit_per_item (x : ℝ) : ℝ := (initial_selling_price - x) - initial_cost_price

theorem determine_price_reduction (x : ℝ) 
    (h1 : daily_sales_volume x = initial_daily_sales_volume + price_reduction_increase_rate * x)
    (h2 : profit_per_item x = 60 - x) : 
    (30 + 3 * x) * (60 - x) = 3600 → x = 30 :=
by 
  sorry

end determine_price_reduction_l607_607287


namespace determine_OP_l607_607955

theorem determine_OP
  (a b c d e : ℝ)
  (h_dist_OA : a > 0)
  (h_dist_OB : b > 0)
  (h_dist_OC : c > 0)
  (h_dist_OD : d > 0)
  (h_dist_OE : e > 0)
  (h_c_le_d : c ≤ d)
  (P : ℝ)
  (hP : c ≤ P ∧ P ≤ d)
  (h_ratio : ∀ (P : ℝ) (hP : c ≤ P ∧ P ≤ d), (a - P) / (P - e) = (c - P) / (P - d)) :
  P = (ce - ad) / (a - c + e - d) :=
sorry

end determine_OP_l607_607955


namespace factorize_expression_l607_607743

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l607_607743


namespace lines_parallel_l607_607421

theorem lines_parallel (a b c : Line) (α β : Plane) (l : Line) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : α ≠ β) (h5 : l ∥ α) (h6 : a ∥ c) (h7 : b ∥ c) : a ∥ b := 
by
  sorry

end lines_parallel_l607_607421


namespace triangle_shape_isosceles_or_right_l607_607123

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

theorem triangle_shape_isosceles_or_right (h1 : a^2 + b^2 ≠ 0) (h2 : 
  (a^2 + b^2) * Real.sin (A - B) 
  = (a^2 - b^2) * Real.sin (A + B))
  (h3 : ∀ (A B C : ℝ), A + B + C = π) :
  ∃ (isosceles : Bool), (isosceles = true) ∨ (isosceles = false ∧ A + B = π / 2) :=
sorry

end triangle_shape_isosceles_or_right_l607_607123


namespace bd_ad_ratio_l607_607094

noncomputable def mass_point_geometry_bd_ad : ℚ := 
  let AT_OVER_ET := 5
  let DT_OVER_CT := 2
  let mass_A := 1
  let mass_D := 3 * mass_A
  let mass_B := mass_A + mass_D
  mass_B / mass_D

theorem bd_ad_ratio (h1 : AT/ET = 5) (h2 : DT/CT = 2) : BD/AD = 4 / 3 :=
by
  have mass_A := 1
  have mass_D := 3
  have mass_B := 4
  have h := mass_B / mass_D
  sorry

end bd_ad_ratio_l607_607094


namespace find_b_value_l607_607969

theorem find_b_value : 
  ∀ (a b : ℝ), 
    (a^3 * b^4 = 2048) ∧ (a = 8) → b = Real.sqrt 2 := 
by 
sorry

end find_b_value_l607_607969


namespace prob_GPA_geq_3_5_l607_607337

-- Define the points for each grade
def points (grade : Char) : ℕ :=
  match grade with
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | 'D' => 1
  | _ => 0

-- Probability distributions for grades in English and History
def probEnglish (grade : Char) : ℚ :=
  match grade with
  | 'A' => 1 / 6
  | 'B' => 1 / 4
  | 'C' => 7 / 12
  | _ => 0

def probHistory (grade : Char) : ℚ :=
  match grade with
  | 'A' => 1 / 4
  | 'B' => 1 / 3
  | 'C' => 5 / 12
  | _ => 0

-- GPA calculation function
def GPA (grades : List Char) : ℚ :=
  (grades.map points).sum / 4

-- All combinations of grades in English and History that would result in a GPA ≥ 3.5
def successfulCombos : List (Char × Char) :=
  [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'B'), ('C', 'A')]

-- Calculate the probability of each successful combination
def probSuccessfulCombo : (Char × Char) → ℚ
  | (e, h) => probEnglish e * probHistory h

-- Sum the probabilities of the successful combinations
def totalProb : ℚ :=
  (successfulCombos.map probSuccessfulCombo).sum

theorem prob_GPA_geq_3_5 : totalProb = 11 / 24 :=
  sorry

end prob_GPA_geq_3_5_l607_607337


namespace sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l607_607095

theorem sin_C_eq_sqrt14_div_8 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  sinC = Real.sqrt 14 / 8 := 
by
  -- Proof is omitted
  sorry

theorem area_triangle_eq_sqrt7_div_4 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  let cosC := Real.sqrt (1 - sinC^2)
  let sinA := sinB * cosC + cosB * sinC
  let area := 1 / 2 * b * c * sinA
  area = Real.sqrt 7 / 4 := 
by
  -- Proof is omitted
  sorry

end sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l607_607095


namespace isabella_hair_length_l607_607125

-- Define the conditions and the question in Lean
def current_length : ℕ := 9
def length_cut_off : ℕ := 9

-- Main theorem statement
theorem isabella_hair_length 
  (current_length : ℕ) 
  (length_cut_off : ℕ) 
  (H1 : current_length = 9) 
  (H2 : length_cut_off = 9) : 
  current_length + length_cut_off = 18 :=
  sorry

end isabella_hair_length_l607_607125


namespace no_harmonic_point_on_reciprocal_find_a_and_c_range_of_m_l607_607112

def harmonic_point (P : ℝ × ℝ) : Prop :=
P.1 = P.2

theorem no_harmonic_point_on_reciprocal :
  ¬ ∃ (P : ℝ × ℝ), harmonic_point P ∧ P.2 = -4 / P.1 :=
sorry

theorem find_a_and_c :
  ∀ (a c : ℝ), harmonic_point (5/2, 5/2) →
  5/2 = a * (5/2)^2 + 6 * (5/2) + c →
  (a = -1 ∧ c = -25/4) :=
sorry

theorem range_of_m :
  ∀ (a c m : ℝ), a = -1 ∧ c = -25/4 ∧
  (-a * (m - a * m^2 + 6 * m + c + 1/4) + m ∈ set.Icc (1:ℝ) m) ∧
  (∃ (y : ℝ → ℝ), y = λ x, -1 * x^2 + 6 * x - 6) →
  (-1 * 1^2 + 6 * 1 - 6 = -1) ∧
  (-1 * 5^2 + 6 * 5 - 6 = -1) ∧
  (1 ≤ m ↔ m ∈ set.Icc (3:ℝ) 5) :=
sorry

end no_harmonic_point_on_reciprocal_find_a_and_c_range_of_m_l607_607112


namespace find_remainder_l607_607538

-- Given conditions
def dividend : ℕ := 144
def divisor : ℕ := 11
def quotient : ℕ := 13

-- Theorem statement
theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = divisor * quotient + 1):
  ∃ r, r = dividend % divisor := 
by 
  exists 1
  sorry

end find_remainder_l607_607538


namespace sector_perimeter_minimized_l607_607806

theorem sector_perimeter_minimized
  (S : ℝ) (r l α C : ℝ)
  (h_area : S = 1/2 * r^2 * α)
  (h_arc : l = r * α) :
  α = 2 ∧ C = 4 * Real.sqrt S :=
begin
  sorry,
end

end sector_perimeter_minimized_l607_607806


namespace binomial_7_4_eq_35_l607_607711
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607711


namespace equilateral_triangle_l607_607519

-- Definitions and given conditions
variables {A B C : ℝ} -- Angles of the triangle ABC
variables {AA1 BB1 CC1 : ℝ} -- Distances from vertices to the points of tangency
variables (A1 B1 C1 : ℝ) -- In-termed points where the incircle touches the sides

-- Given condition of equal segment lengths
constant h : AA1 = BB1 ∧ BB1 = CC1

-- Definition for proving triangle is equilateral
theorem equilateral_triangle (h : AA1 = BB1 ∧ BB1 = CC1) : A = B ∧ B = C :=
sorry

end equilateral_triangle_l607_607519


namespace FG_passes_through_midpoint_of_AC_l607_607187

open EuclideanGeometry

noncomputable def midpoint_of_AC (A C : Point) : Point := midpoint A C

theorem FG_passes_through_midpoint_of_AC 
  (ω : Circle)
  (A C B : Point)
  (ω1 ω2: Circle) 
  (O1 O2: Point)
  (D E F G: Point)
  (h1 : is_chord ω A C)
  (h2 : B ∈ seg A C)
  (h3 : center ω1 = O1 ∧ O1 ∈ seg A B ∧ D ∈ ω1 ∧ D ∈ ω ∧ D ≠ A ∧ D ≠ B)
  (h4 : center ω2 = O2 ∧ O2 ∈ seg B C ∧ E ∈ ω2 ∧ E ∈ ω ∧ E ≠ B ∧ E ≠ C)
  (h5 : incident O1 D F)
  (h6 : incident O2 E F)
  (h7 : incident A D G)
  (h8 : incident C E G)
: incident F G (midpoint_of_AC A C) := 
sorry

end FG_passes_through_midpoint_of_AC_l607_607187


namespace bacteria_initial_count_l607_607657

theorem bacteria_initial_count (n : ℕ) :
  (∀ t : ℕ, t % 30 = 0 → n * 2^(t / 30) = 262144 → t = 240) → n = 1024 :=
by sorry

end bacteria_initial_count_l607_607657


namespace sum_S_15_28_39_l607_607699

/-- Definition of S_n based on given conditions -/
def S : ℕ → ℤ
| n := ∑ i in range n, (-1) ^ i * (i + 1)

theorem sum_S_15_28_39 :
  S 15 + S 28 + S 39 = 14 :=
sorry

end sum_S_15_28_39_l607_607699


namespace coin_exchange_impossible_l607_607290

theorem coin_exchange_impossible :
  ∀ (n : ℕ), (n % 4 = 1) → (¬ (∃ k : ℤ, n + 4 * k = 26)) :=
by
  intros n h
  sorry

end coin_exchange_impossible_l607_607290


namespace g_at_2_l607_607843

def g (x : ℝ) : ℝ := x^3 - x

theorem g_at_2 : g 2 = 6 :=
by
  sorry

end g_at_2_l607_607843


namespace product_华_杯_赛_eq_120_l607_607577

variables (华 杯 赛 英 数 学 : ℕ)
-- additional assumptions from problem conditions
variable (田_condition : ∀ (a b c d : ℕ), (a + b + c + d = sum) → (sum)) 
variable (cond2 : 华 * 华 = 英 * 英 + 赛 * 赛)
variable (cond3 : 数 > 学)

theorem product_华_杯_赛_eq_120 :
  华 * 杯 * 赛 = 120 :=
sorry

end product_华_杯_赛_eq_120_l607_607577


namespace solve_for_x_l607_607960

-- Definitions based on conditions
def base_relation (x : ℝ) : Prop :=
  16 ^ x * 16 ^ x * 16 ^ x * 16 ^ x = 256 ^ 4

def relation_256_eq_16_square : Prop :=
  256 = 16 ^ 2

-- Theorem statement based on the mathematically equivalent proof problem
theorem solve_for_x (x : ℝ) (h1 : base_relation x) (h2 : relation_256_eq_16_square) : x = 2 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l607_607960


namespace subset_123_12_false_l607_607262

-- Definitions derived from conditions
def is_int (x : ℤ) := true
def subset_123_12 (A B : Set ℕ) := A = {1, 2, 3} ∧ B = {1, 2}
def intersection_empty {A B : Set ℕ} (hA : A = {1, 2}) (hB : B = ∅) := (A ∩ B = ∅)
def union_nat_real {A B : Set ℝ} (hA : Set.univ ⊆ A) (hB : Set.univ ⊆ B) := (A ∪ B)

-- The mathematically equivalent proof problem
theorem subset_123_12_false (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 2}):
  ¬ (A ⊆ B) :=
by
  sorry

end subset_123_12_false_l607_607262


namespace quadratic_roots_p_eq_l607_607212

theorem quadratic_roots_p_eq (b c p q r s : ℝ)
  (h1 : r + s = -b)
  (h2 : r * s = c)
  (h3 : r^2 + s^2 = -p)
  (h4 : r^2 * s^2 = q):
  p = 2 * c - b^2 :=
by sorry

end quadratic_roots_p_eq_l607_607212


namespace stock_quote_correctness_l607_607278

theorem stock_quote_correctness 
    (dividend_rate : ℝ) (yield_rate : ℝ) 
    (tax_rate : ℝ) (inflation_rate : ℝ) (risk_premium : ℝ) 
    (face_value : ℝ) (quote : ℝ) : 
    dividend_rate = 0.16 → 
    yield_rate = 0.14 → 
    tax_rate = 0.20 → 
    inflation_rate = 0.03 → 
    risk_premium = 0.02 → 
    face_value = 100 → 
    let after_tax_yield := yield_rate * (1 - tax_rate) in 
    let real_yield := (1 + after_tax_yield) / (1 + inflation_rate) - 1 in 
    let required_yield := real_yield + risk_premium in 
    let dividend_per_share := face_value * dividend_rate in 
    let computed_quote := (dividend_per_share / required_yield) * 100 in 
    abs (computed_quote - quote) < 0.01 → 
    quote = 160.64 :=
by
  intros
  sorry

end stock_quote_correctness_l607_607278


namespace circular_pond_area_l607_607672

noncomputable def plank_length_ab : ℝ := 20 -- 20 foot plank from point A to point B
noncomputable def plank_length_fe : ℝ := 18 -- 18 foot plank from point F to point E
noncomputable def midpoint_ab (x : ℝ) : ℝ := x / 2 -- F is the midpoint of AB

theorem circular_pond_area :
  let r := real.sqrt (plank_length_fe^2 - midpoint_ab plank_length_ab^2) in
  r^2 = 224 -> 
  ∃ π A, A = 224 * π :=
by
  sorry

end circular_pond_area_l607_607672


namespace scalene_triangle_angles_l607_607473

theorem scalene_triangle_angles (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x ≠ y) (h4 : x ≠ 2 * y) (h5 : y ≠ 2 * x) (h6 : x + 2 * x + y = 180)
  (hx : x = 36 ∨ 2 * x = 36 ∨ y = 36): 
  (x = 36 ∧ 2 * x = 72 ∧ y = 72) → false ∨ 
  (x = 36 ∧ 2 * x = 72 ∧ y = 72) ∨ 
  (x = 48 ∧ 2 * x = 96 ∧ y = 36) ∨ 
  (x = 18 ∧ 2 * x = 36 ∧ y = 126) :=
begin
  -- Proof goes here
  sorry
end

end scalene_triangle_angles_l607_607473


namespace radius_of_third_tangent_circle_l607_607245

theorem radius_of_third_tangent_circle :
  ∃ (r : ℝ), r = 5 / 7 ∧ 
  let C := (circle (2: ℝ)), 
      D := (circle (5: ℝ)), 
      E := (circle r) in 
    is_tangent C D ∧ is_tangent C E ∧ is_tangent D E ∧ is_tangent E (line_through C D) :=
sorry

end radius_of_third_tangent_circle_l607_607245


namespace calculate_sacks_l607_607704

theorem calculate_sacks : 
  let
    wood_per_sack := 20
    wood_father := 80
    wood_senior_ranger := 80
    wood_worker := 120
    total_wood := wood_father + wood_senior_ranger + 2 * wood_worker
  in
  total_wood / wood_per_sack = 20 := 
  sorry

end calculate_sacks_l607_607704


namespace value_of_expression_l607_607456

theorem value_of_expression (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 4 / 3) :
  (1 / 3 * x^7 * y^6) * 4 = 1 :=
by
  sorry

end value_of_expression_l607_607456


namespace arithmetic_sequence_general_term_l607_607481

theorem arithmetic_sequence_general_term
  (a : ℕ → ℤ)
  (h1 : a 3 = 7)
  (h2 : a 5 = a 2 + 6)
  (∀ n : ℕ, n > 0 → ∃ d : ℤ, a (n + 1) = a n + d) :
  ∃ (c : ℤ) (d : ℤ), (∀ n : ℕ, a n = c + n * d) ∧ c = 1 ∧ d = 2 :=
by
  sorry

end arithmetic_sequence_general_term_l607_607481


namespace max_lateral_area_l607_607981

theorem max_lateral_area (r h : ℝ) (hr : 0 ≤ r) (hh : 0 ≤ h) (hp : 2 * r + h = 2) :
  2 * (π * r * h) ≤ π :=
begin
  sorry
end

end max_lateral_area_l607_607981


namespace a2_plus_a4_sum_first_n_terms_is_n_squared_l607_607490

section sequence_proof

variable {a_n : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℤ) :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + 2

def a_n_initial_condition (a_n : ℕ → ℤ) :=
  a_n 3 = 5

-- Prove that a₂ + a₄ = 10
theorem a2_plus_a4 (a_n : ℕ → ℤ) [is_arithmetic_sequence a_n] [a_n_initial_condition a_n] :
    a_n 2 + a_n 4 = 10 :=
sorry

-- Prove that the sum of the first n terms Sₙ = n²
noncomputable def sum_first_n_terms (a_n : ℕ → ℤ) (n : ℕ): ℤ :=
  (∑ i in Finset.range n, a_n i)

theorem sum_first_n_terms_is_n_squared (a_n : ℕ → ℤ) [is_arithmetic_sequence a_n] [a_n_initial_condition a_n] :
    ∀ n : ℕ, sum_first_n_terms a_n n = n^2 :=
sorry

end sequence_proof

end a2_plus_a4_sum_first_n_terms_is_n_squared_l607_607490


namespace solve_for_x_l607_607559

theorem solve_for_x : (∃ x : ℝ, (1/2 - 1/3 = 1/x)) ↔ (x = 6) := sorry

end solve_for_x_l607_607559


namespace find_a_minus_b_plus_c_l607_607489

def a_n (n : ℕ) : ℕ := 4 * n - 3

def S_n (a b c n : ℕ) : ℕ := 2 * a * n ^ 2 + b * n + c

theorem find_a_minus_b_plus_c
  (a b c : ℕ)
  (h : ∀ n : ℕ, n > 0 → S_n a b c n = 2 * n ^ 2 - n)
  : a - b + c = 2 :=
by
  sorry

end find_a_minus_b_plus_c_l607_607489


namespace base_for_four_digit_even_l607_607393

theorem base_for_four_digit_even (b : ℕ) : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0 → b = 6 :=
by
  sorry

end base_for_four_digit_even_l607_607393


namespace area_of_EFCD_l607_607887

theorem area_of_EFCD (A B C D E F : Type*) [LinearOrderedField ℝ] 
  (AB CD : ℝ) (hAB : AB = 10) (hCD : CD = 26) (altitude_ABCD : ℝ) 
  (hAltitude : altitude_ABCD = 15) (E_midpoint_AD : E) (F_midpoint_BC : F) :
  (let EF := (AB + CD) / 2;
       altitude_EFCD := altitude_ABCD / 2 in
       (altitude_EFCD * (EF + CD) / 2) = 165) :=
by
  sorry

end area_of_EFCD_l607_607887


namespace cylindrical_shape_is_line_l607_607774

def cylindrical_shape (c k : ℝ) : Type :=
  {p : ℝ × ℝ × ℝ // p.2.1 = c ∧ p.2.2 = k}

theorem cylindrical_shape_is_line (c k : ℝ) : ∀ (p1 p2 : cylindrical_shape c k), 
  (p1.val.1 = p2.val.1 ∧ p1.val.2.1 = p2.val.2.1 ∧ p1.val.2.2 = p2.val.2.2) → 
  (\exists l : set (ℝ × ℝ × ℝ), ∀ p ∈ l, p.2.1 = c ∧ p.2.2 = k ∧ (l = {(x, y, z) | x = c * cos(c) + y * sin(c) ∧ z = k})) := 
sorry

end cylindrical_shape_is_line_l607_607774


namespace investment_return_l607_607646

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l607_607646


namespace problem_l607_607780

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 9 / Real.log 4
def c : ℝ := 3 / 2

theorem problem : a < c ∧ c < b := by
  sorry

end problem_l607_607780


namespace lcm_of_18_and_36_l607_607251

theorem lcm_of_18_and_36 : Nat.lcm 18 36 = 36 := 
by 
  sorry

end lcm_of_18_and_36_l607_607251


namespace roxy_garden_problem_l607_607553

variable (initial_flowering : ℕ)
variable (multiplier : ℕ)
variable (bought_flowering : ℕ)
variable (bought_fruiting : ℕ)
variable (given_flowering : ℕ)
variable (given_fruiting : ℕ)

def initial_fruiting (initial_flowering : ℕ) (multiplier : ℕ) : ℕ :=
  initial_flowering * multiplier

def saturday_flowering (initial_flowering : ℕ) (bought_flowering : ℕ) : ℕ :=
  initial_flowering + bought_flowering

def saturday_fruiting (initial_fruiting : ℕ) (bought_fruiting : ℕ) : ℕ :=
  initial_fruiting + bought_fruiting

def sunday_flowering (saturday_flowering : ℕ) (given_flowering : ℕ) : ℕ :=
  saturday_flowering - given_flowering

def sunday_fruiting (saturday_fruiting : ℕ) (given_fruiting : ℕ) : ℕ :=
  saturday_fruiting - given_fruiting

def total_plants_remaining (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  sunday_flowering + sunday_fruiting

theorem roxy_garden_problem 
  (h1 : initial_flowering = 7)
  (h2 : multiplier = 2)
  (h3 : bought_flowering = 3)
  (h4 : bought_fruiting = 2)
  (h5 : given_flowering = 1)
  (h6 : given_fruiting = 4) :
  total_plants_remaining 
    (sunday_flowering 
      (saturday_flowering initial_flowering bought_flowering) 
      given_flowering) 
    (sunday_fruiting 
      (saturday_fruiting 
        (initial_fruiting initial_flowering multiplier) 
        bought_fruiting) 
      given_fruiting) = 21 := 
  sorry

end roxy_garden_problem_l607_607553


namespace jasmine_pies_l607_607127

-- Definitions based on the given conditions
def total_pies : Nat := 30
def raspberry_part : Nat := 2
def peach_part : Nat := 5
def plum_part : Nat := 3
def total_parts : Nat := raspberry_part + peach_part + plum_part

-- Calculate pies per part
def pies_per_part : Nat := total_pies / total_parts

-- Prove the statement
theorem jasmine_pies :
  (plum_part * pies_per_part = 9) :=
by
  -- The statement and proof will go here, but we are skipping the proof part.
  sorry

end jasmine_pies_l607_607127


namespace binomial_7_4_eq_35_l607_607709
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l607_607709


namespace part_one_part_two_l607_607831

def universal_set : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

noncomputable def C_R_A : Set ℝ := { x | x < 1 ∨ x ≥ 7 }
noncomputable def C_R_A_union_B : Set ℝ := C_R_A ∪ B

theorem part_one : C_R_A_union_B = { x | x < 1 ∨ x > 2 } :=
sorry

theorem part_two (a : ℝ) (h : A ⊆ C a) : a ≥ 7 :=
sorry

end part_one_part_two_l607_607831


namespace stratified_sampling_group_C_l607_607654

open Nat

theorem stratified_sampling_group_C
  (total_sample_size : ℕ)
  (group_A_ratio group_B_ratio group_C_ratio : ℕ)
  (h_ratio : group_A_ratio = 5 ∧ group_B_ratio = 4 ∧ group_C_ratio = 1)
  (h_sample_size : total_sample_size = 20) :
  let total_ratio := group_A_ratio + group_B_ratio + group_C_ratio in
  let group_C_samples := total_sample_size * group_C_ratio / total_ratio in
  group_C_samples = 2 := by
  sorry

end stratified_sampling_group_C_l607_607654


namespace proof_of_f_x0_l607_607819

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) / (Real.exp x - 1) + x

-- Define f' (the derivative of f)
def f' (x : ℝ) : ℝ := (Real.exp x * (Real.exp x - x - 2)) / (Real.exp x - 1)^2

-- Define the minimum point x0 and its condition
def x0 := some (filter (λ x, f' x = 0) {x | 0 < x})

-- The corresponding condition for f(x0)
def f_x0 : Prop := f x0.val = x0.val + 1 ∧ f x0.val < 3

-- Statement to prove
theorem proof_of_f_x0 : f_x0 :=
by
  sorry

end proof_of_f_x0_l607_607819


namespace part_1_part_2_l607_607804

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ} 
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Given conditions
def condition_1 (n : ℕ) (h : 2 ≤ n) : Prop := 
  (a n + 1) ^ 2 = a (n - 1) ^ 2 + 4 * a n + 2 * a (n - 1) + 1
  
def condition_2 (n : ℕ) : Prop := 
  n * a (n + 1) = (n + 1) * a n + 1
  
def condition_3 (n : ℕ) : Prop := 
  S n ≠ 0 → (n * S (n + 1)) / (S n + n) = n + 1

def initial_condition : Prop := 
  a 1 = 1

-- To be proven
theorem part_1 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) (h3 : condition_3 n) (h_initial : initial_condition) : 
  a n = 2 * n - 1 :=
sorry

theorem part_2 (n : ℕ) (h_partition : ∀ n, a n = 2 * n - 1) : 
  T n = ∑ k in finset.range n, b k → T n = (2 * n) / (2 * n + 1) :=
sorry


end part_1_part_2_l607_607804


namespace find_value_of_a_l607_607807

theorem find_value_of_a :
  ∀ (a : ℝ), (let T_r := λ r : ℕ, (nat.choose 7 r) * 2^r * a^(7-r) * x^(2*r-7),
                  coeff := 84,
                  r := 2 in
               (T_r r).coeff (x^(-3)) = coeff) → a = -1 :=
by
  intro a
  sorry

end find_value_of_a_l607_607807


namespace dihedral_angle_B1EF_A1B1C1D1_l607_607483

theorem dihedral_angle_B1EF_A1B1C1D1
    (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ) -- Vertices of the cube
    (E : ℝ × ℝ × ℝ) -- Midpoint of BC
    (F : ℝ × ℝ × ℝ) -- Point on AA1 such that A1F : FA = 1 : 2
    (side_length : ℝ)
    (h_cube : 
        (A = (0, 0, 0)) ∧ 
        (B = (0, side_length, 0)) ∧ 
        (C = (side_length, side_length, 0)) ∧ 
        (D = (side_length, 0, 0)) ∧
        (A1 = (0, 0, side_length)) ∧ 
        (B1 = (0, side_length, side_length)) ∧ 
        (C1 = (side_length, side_length, side_length)) ∧ 
        (D1 = (side_length, 0, side_length))
    )
    (h_midpoint : E = (side_length / 2, side_length, 0))
    (h_ratio : ∃ s t, F = s * A1 + t * A ∧ s + t = 1 ∧ s / t = 1 / 2)
    (h_side_length : side_length = 6) :
    dihedral_angle (B1, E, F) (A1, B1, C1, D1) = arctan (sqrt 37 / 3) := by
sorry

end dihedral_angle_B1EF_A1B1C1D1_l607_607483


namespace φ_is_inner_product_ψ_is_inner_product_θ_is_not_inner_product_l607_607953

open Matrix
open Finset

noncomputable def φ (n : ℕ) (a b : Fin n → ℝ) : ℝ :=
∑ i in range(n), i.succ^2 * a i * b i

noncomputable def ψ (n : ℕ) (a b : Fin n → ℝ) : ℝ :=
φ n a b + (a 0 + a 1) * (b 0 + b 1)

noncomputable def θ (n : ℕ) (a b : Fin n → ℝ) : ℝ :=
∑ i in range(n), ∑ j in range(n), if i ≠ j then a i * b j else 0

theorem φ_is_inner_product (n : ℕ) : 
  ∀ (a b : Fin n → ℝ), ∀ (α : ℝ), 
    φ n a b = φ n b a 
    ∧ φ n (λ i, α * a i) b = α * φ n a b
    ∧ φ n (λ i, a i + b i) c = φ n a c + φ n b c
    ∧ (φ n a a ≥ 0 ∧ (φ n a a = 0 → a = 0)) := 
sorry

theorem ψ_is_inner_product (n : ℕ) :
  ∀ (a b : Fin n → ℝ), ∀ (α : ℝ), 
    ψ n a b = ψ n b a
    ∧ ψ n (λ i, α * a i) b = α * ψ n a b
    ∧ ψ n (λ i, a i + b i) c = ψ n a c + ψ n b c
    ∧ (ψ n a a ≥ 0 ∧ (ψ n a a = 0 → a = 0)) :=
sorry

theorem θ_is_not_inner_product (n : ℕ) :
  ∃ (a : Fin n → ℝ), θ n a a = 0 ∧ a ≠ 0 := 
sorry

end φ_is_inner_product_ψ_is_inner_product_θ_is_not_inner_product_l607_607953


namespace intersection_of_sets_l607_607445

def A (x : ℝ) : Prop := x > -2
def B (x : ℝ) : Prop := 1 - x > 0

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | x > -2 ∧ x < 1} := by
  sorry

end intersection_of_sets_l607_607445


namespace obtuse_angle_between_hypotenuses_l607_607213

-- Define the angles in degrees
def deg (n : ℕ) := n

-- Define the triangles and their properties
structure Triangle := 
  (angle1 : ℕ) 
  (angle2 : ℕ) 
  (angle3 : ℕ)
  (sum_angles : angle1 + angle2 + angle3 = 180)

-- Assuming these two triangles are congruent and right-angled
def triangle1 : Triangle := {angle1 := 40, angle2 := 50, angle3 := 90, sum_angles := by norm_num}
def triangle2 : Triangle := {angle1 := 40, angle2 := 50, angle3 := 90, sum_angles := by norm_num}

-- Define the measure of the obtuse angle formed by the hypotenuses
def alpha : ℕ := 170

-- The theorem to be proven
theorem obtuse_angle_between_hypotenuses : 
  let α := 170
  in α = 180 - 50 + 40 :=
by
  sorry

end obtuse_angle_between_hypotenuses_l607_607213


namespace carol_can_invite_friends_l607_607703

-- Definitions based on the problem's conditions
def invitations_per_pack := 9
def packs_bought := 5

-- Required proof statement
theorem carol_can_invite_friends :
  invitations_per_pack * packs_bought = 45 :=
by
  sorry

end carol_can_invite_friends_l607_607703


namespace sum_a_up_to_1000_l607_607772

def a (n : ℕ) : ℕ := 
  if n % 15 = 0 then 15 
  else if n % 10 = 0 then 10 
  else if n % 6 = 0 then 6 
  else 0

theorem sum_a_up_to_1000 : (∑ n in Finset.range 1000, a (n + 1)) = 2026 := 
  sorry

end sum_a_up_to_1000_l607_607772


namespace smallest_nat_satisfying_conditions_l607_607391

theorem smallest_nat_satisfying_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 2) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 2) ∧ 
  (x % 12 = 2) ∧ 
  (∀ y : ℕ, (y % 4 = 2) ∧ (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 12 = 2) → x ≤ y) :=
  sorry

end smallest_nat_satisfying_conditions_l607_607391


namespace problem_part1_problem_part2_l607_607410

variables {a b c : ℝ} (α β γ : ℝ) (B : ℝ)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def triangle_area (a b c B : ℝ) : ℝ :=
  (1/2) * a * b * real.sin(B)

theorem problem_part1
  (h1 : is_parallel (a - c, a - b) (a + b, c)) :
  B = real.pi / 3 := 
begin
  sorry
end

theorem problem_part2
  (h1 : is_parallel (1 - 3, 1 - sqrt(7)) (1 + sqrt(7), 3))
  (h2 : a = 1)
  (h3 : b = sqrt(7))
  (h4 : B = real.pi / 3) :
  triangle_area 1 3 (real.pi / 3) = (3 * real.sqrt 3) / 4 :=
begin
  sorry
end

end problem_part1_problem_part2_l607_607410


namespace tan_theta_value_l607_607039

-- Define the conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ θ ∉ set.Icc Real.pi (3 * Real.pi / 2)

noncomputable def condition1 (θ : ℝ) : Prop := sin θ + 3 * cos θ = 1

-- State the theorem
theorem tan_theta_value (θ : ℝ) (h1 : is_in_fourth_quadrant θ) (h2 : condition1 θ) : tan θ = -4 / 3 :=
sorry

end tan_theta_value_l607_607039


namespace find_d_k_l607_607987

variable (d k : ℚ)  -- Use rational numbers for precision over floats
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 3], ![4, d]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  (d - 12)⁻¹ • ![![d, -3], ![-4, 1]]

theorem find_d_k (h : A⁻¹ = k • A) : d = 6 ∧ k = 1 / 6 :=
by
  have hA_inv : A⁻¹ = A_inv d := by sorry
  have h_eq : A_inv d = k • A := by sorry
  sorry -- Proof goes here.

end find_d_k_l607_607987


namespace Points_On_Same_Sphere_l607_607584

structure Point (α : Type*) :=
(x : α) (y : α) (z : α)

structure Sphere (α : Type*) :=
(center : Point α) (radius : α)

structure Pyramid (α : Type*) :=
(S : Point α) (A : Point α) (B : Point α) (C : Point α)

structure Conditions (α : Type*) :=
(pyramid : Pyramid α)
(A₁ B₁ C₁ A₂ B₂ C₂ : Point α)
(ω : Sphere α)
(Ω : Sphere α)
(passes_through_S : ω.center = pyramid.S)
(intersects_edges_SA_SB_SC : ω ∋ pyramid.A ∧ ω ∋ pyramid.B ∧ ω ∋ pyramid.C)
(intersects_again_SA₁_SB₁_SC₁ : ω ∋ A₁ ∧ ω ∋ B₁ ∧ ω ∋ C₁)
(intersects_plane_ABC_parallel : ∃ (P_plane : Plane α), (@is_parallel_to α (Plane α) P_plane (Plane.mk pyramid.A pyramid.B pyramid.C)))
(spheres_intersection_in_circle : ∃ (circle : Circle α), (Ω ∩ ω) = circle)
(symmetric_points : A₂ = midpoint A₁ (midpoint pyramid.S pyramid.A)
                  ∧ B₂ = midpoint B₁ (midpoint pyramid.S pyramid.B)
                  ∧ C₂ = midpoint C₁ (midpoint pyramid.S pyramid.C))

theorem Points_On_Same_Sphere {α : Type*} [Field α] (cond : Conditions α) :
  ∃ (sphere : Sphere α), cond.pyramid.A ∈ sphere ∧ cond.pyramid.B ∈ sphere ∧ cond.pyramid.C ∈ sphere 
  ∧ cond.A₂ ∈ sphere ∧ cond.B₂ ∈ sphere ∧ cond.C₂ ∈ sphere := by
  sorry

end Points_On_Same_Sphere_l607_607584


namespace externally_tangent_circles_l607_607029

theorem externally_tangent_circles (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (x - a)^2 + y^2 = 4) →
  (∀ x y : ℝ, x^2 + (y - sqrt 5)^2 = 9) →
  (sqrt (a^2 + 5) = 5) →
  a = 2 * sqrt 5 := by
  sorry

end externally_tangent_circles_l607_607029


namespace min_value_of_y_on_interval_l607_607578

noncomputable def y (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_y_on_interval :
  ∃ (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi), y x = (5 * Real.pi / 6) - Real.sqrt 3 :=
begin
  sorry
end

end min_value_of_y_on_interval_l607_607578


namespace sasha_can_eat_minimum_32_l607_607313

-- Definitions for the conditions
def cell := (ℕ × ℕ)
def grid := fin 7 × fin 7
def color := bool -- true for black, false for white

structure candy (c : color) := 
(coord : cell)

-- Each cell contains a candy of either black or white
def contains_candy (c : color) (g : grid) : Prop := 
  ∃ (cd : candy c), cd.coord = g

-- Sasha can eat two candies if they are of the same color and lie in neighboring cells
def can_eat (g : grid → color) (p q : grid) : Prop := 
  (p.1 ≤ 6 ∧ p.2 ≤ 6) ∧ (q.1 ≤ 6 ∧ q.2 ≤ 6) ∧ 
  (g p = g q) ∧ 
  ((p.1 = q.1 ∧ abs (p.2 - q.2) = 1) ∨ (p.2 = q.2 ∧ abs (p.1 - q.1) = 1) ∨ (abs (p.1 - q.1) = 1 ∧ abs (p.2 - q.2) = 1))

-- The main theorem to prove
theorem sasha_can_eat_minimum_32 : ∀ (arrangement : grid → color),
  ∃ (eaten_candies : fin 33 → (grid × grid)), 
  (∀ (i : fin 33), can_eat arrangement (eaten_candies i).1 (eaten_candies i).2) :=
sorry

end sasha_can_eat_minimum_32_l607_607313


namespace solve_system_l607_607964

theorem solve_system (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + y * z + z * x = 11) (h3 : x * y * z = 6) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end solve_system_l607_607964


namespace max_truthful_students_l607_607204

def count_students (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem max_truthful_students : count_students 2015 = 2031120 :=
by sorry

end max_truthful_students_l607_607204


namespace asymptote_of_hyperbola_l607_607442

-- Define the hyperbola and conditions
variables {a b c : ℝ} (a_pos : a > 0) (b_pos : b > 0)
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the eccentricity condition
def eccentricity : Prop := c = 2 * a

-- Define the condition where the line coincides with the parabola's directrix
def directrix_line : Prop := a^2 / c = 1

-- Define the requirement to prove
theorem asymptote_of_hyperbola (h_hyp : ∀ x y, hyperbola a b x y)
  (h_ecc : eccentricity a c) (h_dir : directrix_line a c) : 
  ∀ x : ℝ, ∃ y : ℝ, y = √3 * x ∨ y = -√3 * x :=
sorry

end asymptote_of_hyperbola_l607_607442


namespace stratified_sampling_students_systematic_sampling_faculty_staff_l607_607317

/-- Given a university with the following structure:
    - 8000 total students
    - 1600 freshmen
    - 3200 sophomores
    - 2000 juniors
    - 1200 seniors

    And using stratified sampling to draw a sample of size 400, 
    prove that the number of students drawn from each grade are:
    - Freshmen: 80
    - Sophomores: 160
    - Juniors: 100
    - Seniors: 60
-/
theorem stratified_sampling_students :
  ∀ (total_students : ℕ) (freshmen sophomores juniors seniors : ℕ) (sample_size : ℕ),
  total_students = 8000 →
  freshmen = 1600 →
  sophomores = 3200 →
  juniors = 2000 →
  seniors = 1200 →
  sample_size = 400 →
  -- Total number of students
  (sample_size / total_students) *
  -- Freshmen
  freshmen = 80 ∧
  -- Sophomores
  (sample_size / total_students) *
  sophomores = 160 ∧
  -- Juniors
  (sample_size / total_students) *
  juniors = 100 ∧
  -- Seniors
  (sample_size / total_students) *
  seniors = 60 :=
by {
  intro total_students,
  intro freshmen,
  intro sophomores,
  intro juniors,
  intro seniors,
  intro sample_size,
  intros h1 h2 h3 h4 h5 h6,
  sorry
}

/-- Given a set of 505 faculty and staff members and a required sample size of 50, 
    prove that the systematic sampling method can be used to draw the sample.
-/
theorem systematic_sampling_faculty_staff :
  ∀ (total_faculty_staff : ℕ) (sample_size : ℕ),
  total_faculty_staff = 505 →
  sample_size = 50 →
  -- The proof for the usage of systematic sampling method.
  True := -- only proving the type
by {
  intro total_faculty_staff,
  intro sample_size,
  intros h1 h2,
  trivial -- since we are proving "True" for the method.
}

end stratified_sampling_students_systematic_sampling_faculty_staff_l607_607317


namespace find_special_n_l607_607380

def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

theorem find_special_n (n : ℤ) (h1 : is_prime (5 * n - 7)) (h2 : is_prime (6 * n + 1)) (h3 : is_prime (20 - 3 * n)) : 
  n = 6 := sorry

end find_special_n_l607_607380


namespace merchant_marked_price_percentage_l607_607667

variables (L S M C : ℝ)
variable (h1 : C = 0.7 * L)
variable (h2 : C = 0.75 * S)
variable (h3 : S = 0.9 * M)

theorem merchant_marked_price_percentage : M = 1.04 * L :=
by
  sorry

end merchant_marked_price_percentage_l607_607667


namespace cos_double_angle_l607_607399

theorem cos_double_angle (θ : ℝ) 
  (h : sin (θ / 2) + cos (θ / 2) = 1 / 2) : 
  cos (2 * θ) = -1 / 8 := sorry

end cos_double_angle_l607_607399


namespace unique_element_a_values_set_l607_607085

open Set

theorem unique_element_a_values_set :
  {a : ℝ | ∃! x : ℝ, a * x^2 + 2 * x - a = 0} = {0} :=
by
  sorry

end unique_element_a_values_set_l607_607085


namespace find_d_value_l607_607096

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value_l607_607096


namespace interval_notations_meaning_l607_607606

noncomputable theory

variable (a b : ℝ)

def interval_open_a_infinity := { x : ℝ | x > a }
def interval_closed_a_infinity := { x : ℝ | x >= a }
def interval_open_neg_infinity_b := { x : ℝ | x < b }
def interval_closed_neg_infinity_b := { x : ℝ | x <= b }

theorem interval_notations_meaning :
  (interval_open_a_infinity a = { x : ℝ | x > a }) ∧
  (interval_closed_a_infinity a = { x : ℝ | x >= a }) ∧
  (interval_open_neg_infinity_b b = { x : ℝ | x < b }) ∧
  (interval_closed_neg_infinity_b b = { x : ℝ | x <= b }) :=
by sorry

end interval_notations_meaning_l607_607606


namespace sum_of_solutions_l607_607734

theorem sum_of_solutions (x : ℝ) : 
  let equation := x^2 - 6 * x - 22 = 4 * x + 24
  in
  (∑ (roots : x), x) = 10 :=
sorry

end sum_of_solutions_l607_607734


namespace intersect_y_axis_l607_607694

theorem intersect_y_axis (x1 y1 x2 y2 : ℝ)
    (h1 : (x1, y1) = (2 : ℝ, 10 : ℝ)) 
    (h2 : (x2, y2) = (5 : ℝ, 16 : ℝ)) : 
    ∃ y : ℝ, (0, y) ∈ {(x, m*x + c) | x : ℝ, m c : ℝ ∧ m = (y2 - y1) / (x2 - x1) ∧ c = y1 - m * x1} 
    ∧ y = 6 :=
by
  sorry

end intersect_y_axis_l607_607694


namespace equal_segments_of_tangents_l607_607971

-- Definitions and conditions from the problem

structure Circle (α : Type u) where
  center : α
  radius : ℝ

variables {α : Type u} [metric_space α]

/-- Given a circle, point A external to the circle, T1 and T2 are points of tangency to the circle
    from A, and M is any point on the circle. T1Q1 and T2Q2 are segments formed by lines from M to the
    tangents AT2 and AT1 respectively. -/
theorem equal_segments_of_tangents
  (c : Circle α)
  (A T1 T2 Q1 Q2 M : α)
  (angle_AT1_AT2 : angle A T1 T2 = 60)
  (M_on_circle : dist c.center M = c.radius)
  (T1_on_tangent : tangent_point c A T1)
  (T2_on_tangent : tangent_point c A T2)
  (Q1_intersection: Line T1 M ∩ Line A T2 = {Q1})
  (Q2_intersection: Line T2 M ∩ Line A T1 = {Q2}) :
  dist T1 Q1 = dist T2 Q2 := sorry

end equal_segments_of_tangents_l607_607971


namespace employee_salary_amount_l607_607179

theorem employee_salary_amount (total_revenue : ℝ) (ratio_salary : ℝ) (ratio_stock : ℝ) (total_parts : ℝ) (salary_ratio_fraction : ℝ) :
  total_revenue = 3000 →
  ratio_salary = 4 →
  ratio_stock = 11 →
  total_parts = ratio_salary + ratio_stock →
  salary_ratio_fraction = ratio_salary / total_parts →
  salary_ratio_fraction * total_revenue = 800 :=
by
  intros h_total_revenue h_ratio_salary h_ratio_stock h_total_parts h_salary_ratio_fraction
  rw [h_total_revenue, h_ratio_salary, h_ratio_stock, h_total_parts, h_salary_ratio_fraction]
  sorry

end employee_salary_amount_l607_607179


namespace exists_point_equal_distances_l607_607595

theorem exists_point_equal_distances
  (O O1 : Point) (r r1 : ℝ) (A : Point)
  (M N : ℝ → Point) -- M and N are functions of time representing the moving points
  (h1 : M 0 = A ∧ N 0 = A)
  (h2 : ∀ t, dist M t O = r ∧ dist N t O1 = r1)
  (h3 : ∃ T, M T = A ∧ N T = A) :
  ∃ Q : Point, ∀ t : ℝ, dist Q (M t) = dist Q (N t) :=
sorry

end exists_point_equal_distances_l607_607595


namespace problem_l607_607505

theorem problem (n : ℕ) (k : ℕ) (h : 125 * n + 22 = 3 ^ k) : ∃ p : ℕ, p.prime ∧ p > 100 ∧ p ∣ (125 * n + 29) :=
by
  sorry

end problem_l607_607505


namespace distinct_solutions_diff_l607_607154

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l607_607154


namespace find_b_find_area_l607_607105

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (b_is_4 : b = 4)
variable (h1 : a - 4 * Real.cos C = c * Real.cos B)
variable (h2 : a^2 + b^2 + c^2 = 2 * Real.sqrt 3 * a * b * Real.sin C)

theorem find_b (h1 : a - 4 * Real.cos C = c * Real.cos B) : b = 4 := by
  sorry

theorem find_area (h2 : a^2 + b^2 + c^2 = 2 * sqrt 3 * a * b * sin C) (b : ℝ) (hb : b = 4) : 
area (a b c C : ℝ) := 
1/2 * a * b * sin C = 4 * sqrt 3 := by
  sorry

end find_b_find_area_l607_607105


namespace computer_price_decrease_l607_607617

theorem computer_price_decrease (P₀ : ℝ) (years : ℕ) (decay_rate : ℝ) (P₆ : ℝ) :
  P₀ = 8100 →
  decay_rate = 2 / 3 →
  years = 6 →
  P₆ = P₀ * decay_rate ^ (years / 2) →
  P₆ = 2400 :=
begin
  intros h₀ h₁ h₂ h₃,
  sorry
end

end computer_price_decrease_l607_607617


namespace sqrt_condition_l607_607465

theorem sqrt_condition (x : ℝ) (h : 2 - x ≥ 0) (hx : x ∈ {4, real.pi, -1, 3}) : x = -1 :=
by {
  simp [set.mem_insert_iff, set.mem_singleton_iff] at hx,
  cases hx with h4 hrest,
  { exfalso, linarith },
  cases hrest with hpi hrest,
  { exfalso, linarith [real.pi_pos] },
  cases hrest with hm1 h3,
  { exact hm1 },
  { exfalso, linarith },
}

end sqrt_condition_l607_607465


namespace taxi_ride_cost_l607_607677

theorem taxi_ride_cost (base_fare : ℝ) (first_three_miles_rate : ℝ) (additional_miles_rate : ℝ)
                       (first_miles : ℕ) (total_miles : ℕ) (result : ℝ)
                       (h0 : base_fare = 2.00)
                       (h1 : first_three_miles_rate = 0.30)
                       (h2 : additional_miles_rate = 0.40)
                       (h3 : first_miles = 3)
                       (h4 : total_miles = 8)
                       (h_result : result = 4.90):
  let first_part_fare := first_miles * first_three_miles_rate in
  let remaining_miles := total_miles - first_miles in
  let second_part_fare := remaining_miles * additional_miles_rate in
  base_fare + first_part_fare + second_part_fare = result :=
by
  sorry

end taxi_ride_cost_l607_607677


namespace minimum_value_of_function_l607_607389

theorem minimum_value_of_function :
  ∃ (y : ℝ), y > 0 ∧
  (∀ z : ℝ, z > 0 → y^2 + 10 * y + 100 / y^3 ≤ z^2 + 10 * z + 100 / z^3) ∧ 
  y^2 + 10 * y + 100 / y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 := 
sorry

end minimum_value_of_function_l607_607389


namespace even_three_digit_numbers_less_than_300_count_l607_607604

theorem even_three_digit_numbers_less_than_300_count :
  let digits := {1, 2, 3, 4, 5, 6}
  ∃ count : Nat, count = 36 ∧
    (∀ n : Nat, n < 300 ∧ n % 2 = 0 ∧
      (∀ d ∈ to_list n.digits, d ∈ digits) → True) :=
begin
  let digits := {1, 2, 3, 4, 5, 6},
  use 36,
  split,
  { refl, },
  { intros n hn,
    unfold has_mem.mem to_list set.Sep set.mem digits,
    sorry, -- proof omitted
  }
end

end even_three_digit_numbers_less_than_300_count_l607_607604


namespace distinct_solutions_difference_l607_607152

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l607_607152


namespace move_point_right_l607_607876

theorem move_point_right (x y : ℤ) (units : ℤ) (new_x : ℤ) (new_y : ℤ) :
  x = -1 → y = 3 → units = 5 → new_x = x + units → new_y = y → (new_x, new_y) = (4, 3) :=
by
  intros hx hy hu hnewx hnewy
  rw [hx, hy, hu] at hnewx hnewy
  exact (by simp [hnewx, hnewy])

end move_point_right_l607_607876


namespace intersection_A_B_l607_607829

-- Define the sets A and B
def set_A : Set ℝ := { x | x^2 ≤ 1 }
def set_B : Set ℝ := { -2, -1, 0, 1, 2 }

-- The goal is to prove that the intersection of A and B is {-1, 0, 1}
theorem intersection_A_B : set_A ∩ set_B = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l607_607829


namespace infinite_series_sum_eq_l607_607767

noncomputable def infinite_series_sum : Rat :=
  ∑' n : ℕ, (2 * n + 1) * (2000⁻¹) ^ n

theorem infinite_series_sum_eq : infinite_series_sum = (2003000 / 3996001) := by
  sorry

end infinite_series_sum_eq_l607_607767


namespace no_solution_intervals_l607_607752

theorem no_solution_intervals (a : ℝ) :
  (a < -13 ∨ a > 0) → ¬ ∃ x : ℝ, 6 * abs (x - 4 * a) + abs (x - a^2) + 5 * x - 3 * a = 0 :=
by sorry

end no_solution_intervals_l607_607752


namespace mean_equals_l607_607988

theorem mean_equals (z : ℝ) :
    (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := 
by
    intro h
    sorry

end mean_equals_l607_607988


namespace hiker_rate_l607_607662

theorem hiker_rate : 
  ∀ (R : ℝ), 
  (∀ t : ℝ, t = 2) → -- Both routes take 2 days
  (∀ down_distance : ℝ, down_distance = 24) → -- Descent route is 24 miles long
  (∀ down_rate : ℝ, down_rate = 1.5 * R) → -- Rate of descent is 1.5 times the ascent rate
  (2 * down_rate = down_distance) → -- Condition derived from the descent details
  R = 8 := -- Prove the rate of ascent R is 8 miles per day given all conditions
by
  intros R h_t h_down_distance h_down_rate h_eq,
  sorry

end hiker_rate_l607_607662


namespace inequality_l607_607900

-- Define the real variables p, q, r and the condition that their product is 1
variables {p q r : ℝ} (h : p * q * r = 1)

-- State the theorem
theorem inequality (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := 
sorry

end inequality_l607_607900


namespace simplify_fraction_l607_607347

theorem simplify_fraction :
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
  sorry

end simplify_fraction_l607_607347


namespace solve_system_l607_607494

noncomputable def solution (C1 C2 : ℝ) (t : ℝ) : ℝ × ℝ :=
  let x := C1 * Real.exp t + C2 * Real.exp (-t) - 1
  let y := C1 * Real.exp t - C2 * Real.exp (-t) - 1
  (x, y)

theorem solve_system (C1 C2 : ℝ) (t : ℝ) :
  let (x, y) := solution C1 C2 t
  let dx_dt := y + 1
  let dy_dt := x + 1
  (deriv x t = dx_dt) ∧ (deriv y t = dy_dt) :=
by 
  sorry

end solve_system_l607_607494


namespace cow_total_spots_l607_607947

theorem cow_total_spots : 
  let left_spots := 16 in 
  let right_spots := 3 * left_spots + 7 in
  left_spots + right_spots = 71 :=
by
  let left_spots := 16
  let right_spots := 3 * left_spots + 7
  show left_spots + right_spots = 71
  sorry

end cow_total_spots_l607_607947


namespace triangle_exists_and_perimeter_l607_607309

def triangle_sides : Prop :=
  ∃ (a b c : ℕ), a = 24 ∧ b = 30 ∧ c = 39 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  (a + b + c = 93)

theorem triangle_exists_and_perimeter :
  triangle_sides :=
begin
  sorry
end

end triangle_exists_and_perimeter_l607_607309


namespace avg_remaining_score_l607_607858

open Real

theorem avg_remaining_score (n : ℕ) (hn : n > 15) 
    (avg_class : ℝ := 10) (avg_subset : ℝ := 17) :
    (∑ i in (range 15), avg_subset + ∑ i in (range (n - 15)), avg_subset) / n = avg_class → 
    (∑ i in (range (n - 15)), (avg_class * n - 255) / (n - 15)) / (n - 15) :=
by sorry

end avg_remaining_score_l607_607858


namespace power_eq_l607_607942

open Real

-- Define the main properties: radius, distance, point, and power
variables (P O: Point) (R d: ℝ)
  (h1: d > R) -- P is outside of circle implies distance to center is greater than radius
  (h2: dist P O = d) -- Distance from point P to center O is d
  (h3: r O P = R) -- Radius of circle S with center O is R

-- Definition of power of a point with respect to a circle
def power (P O: Point) (R d: ℝ) : ℝ :=
  let PA := d + R in
  let PB := |d - R| in
  PA * PB

-- Theorem: Power of a point is equal to d^2 - R^2
theorem power_eq (P O: Point) (R d: ℝ) (h1: d > R) (h2: dist P O = d) (h3: r O P = R) : power P O R d = d^2 - R^2 := 
begin
  sorry
end

end power_eq_l607_607942


namespace equal_sides_of_convex_ngon_l607_607785

theorem equal_sides_of_convex_ngon {n : ℕ} 
  (h_convex : convex n) 
  (h_equal_angles : ∀ i j, interior_angle i = interior_angle j) 
  (h_side_len_order : ∀ i j (hij : i ≤ j), a_i ≥ a_j) : 
  ∀ i j, a_i = a_j :=
by
  sorry

end equal_sides_of_convex_ngon_l607_607785


namespace distinct_solution_difference_l607_607168

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l607_607168


namespace fifteenth_prime_is_forty_seven_l607_607242

-- Define the condition that the ninth prime number is 23.
def ninth_prime_is_twenty_three : ℕ := 23

-- Prove that the fifteenth prime number is 47.
theorem fifteenth_prime_is_forty_seven (h : ninth_prime_is_twenty_three = 23) : Nat.prime 47 :=
by
  sorry

end fifteenth_prime_is_forty_seven_l607_607242


namespace right_triangle_third_side_length_l607_607867

theorem right_triangle_third_side_length (x y z : ℝ) (h1 : ∀ a b c : ℝ, a^2 + b^2 = c^2) 
(h2 : abs (x - 3) + real.sqrt (2 * y - 8) = 0) :
  z = real.sqrt 7 ∨ z = 5 :=
by
  sorry

end right_triangle_third_side_length_l607_607867


namespace num_people_in_group_l607_607978

theorem num_people_in_group 
  (new_weight replaced_weight : ℕ) 
  (avg_increase : ℚ) 
  (weight_increase : ℚ := new_weight - replaced_weight)
  (n : ℚ := weight_increase / avg_increase)
  (condition : new_weight = 93 ∧ replaced_weight = 65 ∧ avg_increase = 3.5): 
  n = 8 :=
by
  have condition_new_weight : new_weight = 93 := condition.1
  have condition_replaced_weight : replaced_weight = 65 := (condition.2).1
  have condition_avg_increase : avg_increase = 3.5 := (condition.2).2
  have weight_increase_calc : weight_increase = new_weight - replaced_weight := rfl
  have weight_increase_value : weight_increase = 28 := by
    rw [condition_new_weight, condition_replaced_weight, weight_increase_calc]
    norm_num
  have n_value : n = weight_increase / avg_increase := rfl
  rw [weight_increase_value, condition_avg_increase] at n_value
  norm_num at n_value
  exact n_value

end num_people_in_group_l607_607978


namespace binomial_7_4_eq_35_l607_607714

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607714


namespace problem_statement_l607_607163

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l607_607163


namespace f_increasing_on_Ioo_l607_607028

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_increasing_on_Ioo : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
by sorry

end f_increasing_on_Ioo_l607_607028


namespace fifth_element_is_17_l607_607379

-- Define the sequence pattern based on given conditions
def seq : ℕ → ℤ 
| 0 => 5    -- first element
| 1 => -8   -- second element
| n + 2 => seq n + 3    -- each following element is calculated by adding 3 to the two positions before

-- Additional condition: the sign of sequence based on position
def seq_sign : ℕ → ℤ
| n => if n % 2 = 0 then 1 else -1

-- The final adjusted sequence based on the above observations
def final_seq (n : ℕ) : ℤ := seq n * seq_sign n

-- Assert the expected outcome for the 5th element
theorem fifth_element_is_17 : final_seq 4 = 17 :=
by
  sorry

end fifth_element_is_17_l607_607379


namespace nobody_but_angela_finished_9_problems_l607_607332

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end nobody_but_angela_finished_9_problems_l607_607332


namespace modulus_of_conjugate_l607_607420

variable {z : ℂ}

theorem modulus_of_conjugate (h : (1 + complex.i) * conj z = 4) : complex.abs z = 2 * real.sqrt 2 :=
sorry

end modulus_of_conjugate_l607_607420


namespace problem1_problem2_problem3_l607_607143

-- Define the function f(x) = x^3 - b * x
def f (x : ℝ) (b : ℝ) : ℝ := x^3 - b * x

-- Conditions derived: a = 0, c = 0, b ≤ 3
def conditions (a c b : ℝ) : Prop :=
a = 0 ∧ c = 0 ∧ b ≤ 3

-- Prove f is odd and monotonically increasing on [1, +∞)
theorem problem1 (a c b : ℝ) : 
  conditions a c b → 
  ∀ x : ℝ, f(-x, b) = -f(x, b) → 
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ x2 → f(x1, b) ≤ f(x2, b)) :=
  sorry

-- Find the monotonic intervals of the function f(x) given a = 0, c = 0, b ≤ 3
theorem problem2 (b : ℝ) (h : conditions 0 0 b) : 
  -- Monotonic intervals 
  (if b ≤ 0 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1, b) ≤ f(x2, b)
   else ∀ x : ℝ, f(x, b) is increasing on (-∞, -sqrt(b/3)) ∪ (sqrt(b/3), ∞)) :=
  sorry

-- Prove f(x0) = x0 given x0 ≥ 1, f(x0) ≥ 1, and f(f(x0)) = x0
theorem problem3 (b x0 : ℝ) (h : conditions 0 0 b) :
  x0 ≥ 1 → f(x0, b) ≥ 1 → f(f(x0, b), b) = x0 → f(x0, b) = x0 :=
  sorry

end problem1_problem2_problem3_l607_607143


namespace boat_speed_in_still_water_l607_607651

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 6) (h2 : B - S = 4) : B = 5 := by
  sorry

end boat_speed_in_still_water_l607_607651


namespace total_coins_are_correct_l607_607925

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l607_607925


namespace sprouted_percentage_l607_607500

-- Define the initial conditions
def cherryPits := 80
def saplingsSold := 6
def saplingsLeft := 14

-- Define the calculation of the total saplings that sprouted
def totalSaplingsSprouted := saplingsSold + saplingsLeft

-- Define the percentage calculation
def percentageSprouted := (totalSaplingsSprouted / cherryPits) * 100

-- The theorem to be proved
theorem sprouted_percentage : percentageSprouted = 25 := by
  sorry

end sprouted_percentage_l607_607500


namespace count_diff_two_primes_l607_607071

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def can_be_written_as_diff_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 - p2

theorem count_diff_two_primes :
  (finset.filter (λ n, can_be_written_as_diff_of_two_primes n) (finset.range 100).map (λ k, 4 + 10 * k)).card = 1 :=
sorry

end count_diff_two_primes_l607_607071


namespace numberOfFlowerbeds_l607_607545

def totalSeeds : ℕ := 32
def seedsPerFlowerbed : ℕ := 4

theorem numberOfFlowerbeds : totalSeeds / seedsPerFlowerbed = 8 :=
by
  sorry

end numberOfFlowerbeds_l607_607545


namespace cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l607_607277

/-- A 4x4 chessboard is entirely white except for one square which is black.
The allowed operations are flipping the colors of all squares in a column or in a row.
Prove that it is impossible to have all the squares the same color regardless of the position of the black square. -/
theorem cannot_all_white_without_diagonals :
  ∀ (i j : Fin 4), False :=
by sorry

/-- If diagonal flips are also allowed, prove that 
it is impossible to have all squares the same color if the black square is at certain positions. -/
theorem cannot_all_white_with_diagonals :
  ∀ (i j : Fin 4), (i, j) ≠ (0, 1) ∧ (i, j) ≠ (0, 2) ∧
                   (i, j) ≠ (1, 0) ∧ (i, j) ≠ (1, 3) ∧
                   (i, j) ≠ (2, 0) ∧ (i, j) ≠ (2, 3) ∧
                   (i, j) ≠ (3, 1) ∧ (i, j) ≠ (3, 2) → False :=
by sorry

end cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l607_607277


namespace find_certain_number_l607_607602

-- Define the conditions
variable (m : ℕ)
variable (h_lcm : Nat.lcm 24 m = 48)
variable (h_gcd : Nat.gcd 24 m = 8)

-- State the theorem to prove
theorem find_certain_number (h_lcm : Nat.lcm 24 m = 48) (h_gcd : Nat.gcd 24 m = 8) : m = 16 :=
sorry

end find_certain_number_l607_607602


namespace connor_sleep_duration_l607_607352

variables {Connor_sleep Luke_sleep Puppy_sleep : ℕ}

def sleeps_two_hours_longer (Luke_sleep Connor_sleep : ℕ) : Prop :=
  Luke_sleep = Connor_sleep + 2

def sleeps_twice_as_long (Puppy_sleep Luke_sleep : ℕ) : Prop :=
  Puppy_sleep = 2 * Luke_sleep

def sleeps_sixteen_hours (Puppy_sleep : ℕ) : Prop :=
  Puppy_sleep = 16

theorem connor_sleep_duration 
  (h1 : sleeps_two_hours_longer Luke_sleep Connor_sleep)
  (h2 : sleeps_twice_as_long Puppy_sleep Luke_sleep)
  (h3 : sleeps_sixteen_hours Puppy_sleep) :
  Connor_sleep = 6 :=
by {
  sorry
}

end connor_sleep_duration_l607_607352


namespace triangle_angle_inradius_l607_607888

variable (A B C : ℝ) 
variable (a b c R : ℝ)

theorem triangle_angle_inradius 
    (h1: 0 < A ∧ A < Real.pi)
    (h2: a * Real.cos C + (1/2) * c = b)
    (h3: a = 1):

    A = Real.pi / 3 ∧ R ≤ Real.sqrt 3 / 6 := 
by
  sorry

end triangle_angle_inradius_l607_607888


namespace find_a6_l607_607726

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem find_a6 :
  ∀ (a b : ℕ → ℕ),
    a 1 = 3 →
    b 1 = 2 →
    b 3 = 6 →
    is_arithmetic_sequence b →
    (∀ n, b n = a (n + 1) - a n) →
    a 6 = 33 :=
by
  intros a b h_a1 h_b1 h_b3 h_arith h_diff
  sorry

end find_a6_l607_607726


namespace average_reading_days_l607_607735

theorem average_reading_days :
  let days_participated := [2, 3, 4, 5, 6]
  let students := [5, 4, 7, 3, 6]
  let total_days := List.zipWith (· * ·) days_participated students |>.sum
  let total_students := students.sum
  let average := total_days / total_students
  average = 4.04 := sorry

end average_reading_days_l607_607735


namespace deepak_wife_speed_l607_607185

-- Definitions and conditions
def track_circumference_km : ℝ := 0.66
def deepak_speed_kmh : ℝ := 4.5
def time_to_meet_hr : ℝ := 0.08

-- Theorem statement
theorem deepak_wife_speed
  (track_circumference_km : ℝ)
  (deepak_speed_kmh : ℝ)
  (time_to_meet_hr : ℝ)
  (deepak_distance : ℝ := deepak_speed_kmh * time_to_meet_hr)
  (wife_distance : ℝ := track_circumference_km - deepak_distance)
  (wife_speed_kmh : ℝ := wife_distance / time_to_meet_hr) : 
  wife_speed_kmh = 3.75 :=
sorry

end deepak_wife_speed_l607_607185


namespace circle_range_of_m_l607_607849

theorem circle_range_of_m (m : ℝ) :
  (∃ h k r : ℝ, (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ x ^ 2 + y ^ 2 - x + y + m = 0)) ↔ (m < 1/2) :=
by
  sorry

end circle_range_of_m_l607_607849


namespace circle_tangent_line_center_l607_607572

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 10

theorem circle_tangent_line_center :
  ∀ (x y : ℝ),
    let center : ℝ × ℝ := (1, -2) in
    let tangent_line (x y : ℝ) : Prop := x - 3 * y + 3 = 0 in
    tangent_line x y → equation_of_circle x y :=
begin
  intro x,
  intro y,
  intro center,
  intro tangent_line,
  assume h,
  sorry
end

end circle_tangent_line_center_l607_607572


namespace problem_statement_l607_607273

theorem problem_statement (f : ℕ → ℕ) (h1 : f 1 = 4) (h2 : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4) :
  f 2 + f 5 = 125 :=
by
  sorry

end problem_statement_l607_607273


namespace pole_break_height_l607_607293

/-- Define the flagpole problem --/
def flagpole_problem (h: ℝ) (d: ℝ) : ℝ :=
  (h ^ 2 + d ^ 2).sqrt / 2

/-- Prove the height above the ground where the pole breaks --/
theorem pole_break_height (h d: ℝ) (h_eq : h = 8) (d_eq : d = 3) :
  flagpole_problem h d = (Real.sqrt 73) / 2 :=
by
  -- Using the definition of the problem
  rw [flagpole_problem]
  -- Substitute h and d with given conditions
  rw [h_eq, d_eq]
  -- Perform calculations
  norm_num
  rw [Real.sqrt_eq_rpow]
  -- Handwave the square root and power calculations
  simp
  rw [show 64 + 9 = 73, by norm_num]
  sorry

end pole_break_height_l607_607293


namespace length_of_floor_l607_607268

theorem length_of_floor (b l : ℝ) (h1 : l = 3 * b) (h2 : 3 * b^2 = 484 / 3) : l ≈ 21.99 :=
by
  sorry

end length_of_floor_l607_607268


namespace vector_addition_l607_607068

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (6, 2)
def vector_b : ℝ × ℝ := (-2, 4)

-- Theorem statement to prove the sum of vector_a and vector_b equals (4, 6)
theorem vector_addition :
  vector_a + vector_b = (4, 6) :=
sorry

end vector_addition_l607_607068


namespace fib_100_mod_5_l607_607565

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem fib_100_mod_5 : fib 100 % 5 = 0 := 
by sorry

end fib_100_mod_5_l607_607565


namespace find_subtracted_number_l607_607086

-- Given conditions
def t : ℕ := 50
def k : ℕ := 122
def eq_condition (n : ℤ) : Prop := t = (5 / 9 : ℚ) * (k - n)

-- The proof problem proving the number subtracted from k is 32
theorem find_subtracted_number : eq_condition 32 :=
by
  -- implementation here will demonstrate that t = 50 implies the number is 32
  sorry

end find_subtracted_number_l607_607086


namespace jose_to_haylee_ratio_l607_607838

variable (J : ℕ)

def haylee_guppies := 36
def charliz_guppies := J / 3
def nicolai_guppies := 4 * (J / 3)
def total_guppies := haylee_guppies + J + charliz_guppies + nicolai_guppies

theorem jose_to_haylee_ratio :
  haylee_guppies = 36 ∧ total_guppies = 84 →
  J / haylee_guppies = 1 / 2 :=
by
  intro h
  sorry

end jose_to_haylee_ratio_l607_607838


namespace helium_cost_per_ounce_l607_607836

theorem helium_cost_per_ounce (total_money : ℕ) (cost_sheet : ℕ) (cost_rope : ℕ) (cost_propane : ℕ)
  (height_per_ounce : ℕ) (max_height : ℕ) (helium_left : ℕ) :
  total_money = 200 → cost_sheet = 42 → cost_rope = 18 → cost_propane = 14 →
  height_per_ounce = 113 → max_height = 9492 → 
  helium_left = total_money - (cost_sheet + cost_rope + cost_propane) →
  helium_left = 126 →
  (max_height / height_per_ounce) = 84 →
  (helium_left / (max_height / height_per_ounce)) = 1.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end helium_cost_per_ounce_l607_607836


namespace min_value_of_quadratic_l607_607609

theorem min_value_of_quadratic : ∀ x : ℝ, (x^2 + 6*x + 5) ≥ -4 :=
by 
  sorry

end min_value_of_quadratic_l607_607609


namespace number_of_people_entered_organizers_problem_l607_607934

-- Define the conditions
def placards_per_person : ℕ := 2
def total_placards : ℕ := 823

-- Define the question and the proof goal
theorem number_of_people_entered (t : ℕ) (pp : ℕ) (tp : ℕ) (h_pp : pp = placards_per_person) (h_tp : tp = total_placards) : ℕ :=
  (tp / pp)

-- The final statement that encapsulates the conditions and the proof goal
theorem organizers_problem : number_of_people_entered 39 placards_per_person total_placards = 411 :=
by
  sorry

end number_of_people_entered_organizers_problem_l607_607934


namespace rectangle_circles_l607_607826

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬ q) : p ∨ q :=
by sorry

end rectangle_circles_l607_607826


namespace rectangle_circle_radii_inequality_l607_607798

variables {A B C D : Type} [metric_space A] -- Rectangle vertices
variables (ABCD : rectangle A B C D) -- Rectangle ABCD
variables {k1 k2 : Type} [metric_space k1] [metric_space k2] -- Circles k1 and k2
variables (r1 r2 : ℝ) -- Radii of circles k1 and k2
variables {a b : ℝ} -- Sides of rectangle

/-- Given the conditions -/
def circle_passing_conditions (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
[k1 : metric_space A] [k2 : metric_space A] (ABCD : rectangle A B C D) 
(passes_A_B_tangent_CD : k1 passes_through A B ∧ tangent_to_line k1 CD) 
(passes_A_D_tangent_BC : k2 passes_through A D ∧ tangent_to_line k2 BC) 
(r1 r2 : ℝ) (sides : a = side AB ∧ b = side AD) : Prop :=
  true

theorem rectangle_circle_radii_inequality (A B C D : Type) [metric_space A] 
[metric_space B] [metric_space C] [metric_space D] 
[k1 : metric_space A] [k2 : metric_space A] 
(ABCD : rectangle A B C D) (passes_A_B_tangent_CD : k1 passes_through A B ∧ tangent_to_line k1 CD)
(passes_A_D_tangent_BC : k2 passes_through A D ∧ tangent_to_line k2 BC) 
(r1 r2 : ℝ) (sides : a = side AB ∧ b = side AD) : 
  r1 + r2 ≥ (5 / 8) * (a + b) :=
by sorry

end rectangle_circle_radii_inequality_l607_607798


namespace distinct_solutions_diff_l607_607156

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l607_607156


namespace expected_value_of_win_l607_607292

theorem expected_value_of_win : 
  let p := (1 : ℚ) / 8;
  let win n := (n + 1) * (n + 1);
  (∑ n in finset.range 8, p * win (n + 1)) = 35.5 := 
by
  sorry

end expected_value_of_win_l607_607292


namespace perimeter_of_region_l607_607567

theorem perimeter_of_region (A : ℝ) (n : ℕ) (rows : ℕ) (cols : ℕ) 
  (hA : A = 392) 
  (hn : n = 8)
  (hrows : rows = 2)
  (hcols : cols = 4)
  (h_area : A / n = (7 : ℝ)^2) : 
  ∃ (p : ℝ), p = 126 :=
by {
  use 126,
  sorry
}

end perimeter_of_region_l607_607567


namespace purple_sequins_each_row_l607_607495

theorem purple_sequins_each_row (x : ℕ) : 
  (6 * 8) + (9 * 6) + (5 * x) = 162 → x = 12 :=
by 
  sorry

end purple_sequins_each_row_l607_607495


namespace exists_nat_digit_sum_1990_l607_607892

-- Define the digit sum function
def digitSum (n : ℕ) : ℕ :=
  n.digits.sum

theorem exists_nat_digit_sum_1990 : ∃ (m : ℕ), digitSum m = 1990 ∧ digitSum (m * m) = digitSum (1990 * 1990) :=
by
  sorry

end exists_nat_digit_sum_1990_l607_607892


namespace sum_squared_residuals_correct_l607_607685

-- Define sum of squares for different expressions
def sum_squared_residuals (y : ℕ → ℝ) (ŷ : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (y i - ŷ i)^2)

def expression_A (y : ℕ → ℝ) (ȳ : ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (y i - ȳ)^2)

def expression_C (ŷ_avg : ℝ) (ȳ : ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (ŷ_avg - ȳ)^2)

def expression_D (y : ℕ → ℝ) (ȳ : ℝ) (ŷ : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, (y i - ȳ)^2) + (Finset.range n).sum (λ i, (ŷ i - ȳ)^2)

-- The theorem to prove the correct expression for sum of squared residuals
theorem sum_squared_residuals_correct (y : ℕ → ℝ) (ŷ : ℕ → ℝ) (n : ℕ) :
  (sum_squared_residuals y ŷ n = 
    ∑ i in Finset.range n, (y i - ŷ i) ^ 2) :=
sorry

end sum_squared_residuals_correct_l607_607685


namespace quadratic_real_roots_range_l607_607392

theorem quadratic_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + 2 * x + 1 = 0 → 
    (∃ x1 x2 : ℝ, x = x1 ∧ x = x2 ∧ x1 = x2 → true)) → 
    m ≤ 2 ∧ m ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l607_607392


namespace matrix_inverse_solution_l607_607823

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -1], ![1, 1]]

def b : ℚ × ℚ := (0, 3)

def x : ℚ := 1 -- Solution of the system from part b
def y : ℚ := 2 -- Solution of the system from part b

theorem matrix_inverse_solution (A : Matrix (Fin 2) (Fin 2) ℚ) (b : ℚ × ℚ) (x y : ℚ) 
  (h : A⁻¹ ⬝ ![b.1, b.2] = ![x, y]) : x + y = 3 := by
  sorry

end matrix_inverse_solution_l607_607823


namespace paula_routes_count_l607_607884

-- Definitions based on the problem conditions
def city_map : Type := ℕ   -- Representing 15 cities with numbers from 1 to 15
def road : city_map → city_map → Prop -- 20 roads interconnecting the cities
def paula_route : list (city_map × city_map) -- List of 15 roads Paula will travel through

variable {A M C J : city_map}
variable {route : paula_route}

-- Conditions translation
axiom city_map_prop : ∀ c : city_map, c ≤ 15
axiom road_prop : ∀ (c1 c2 : city_map), road c1 c2 → (c1 ≠ c2 ∧ c1 > 0 ∧ c2 > 0 ∧ c1 ≤ 15 ∧ c2 ≤ 15)
axiom unique_roads : ∀ (r r' : city_map × city_map), r ∈ route → r' ∈ route → r = r' → r = r'
axiom route_length : route.length = 15
axiom start_end_cities : (route.head = (A, C) ∧ route.last = (J, M))
axiom mandatory_cities : (∃ r1 r2, r1 ∈ route ∧ r2 ∈ route ∧ r1.2 = C ∧ r2.2 = J)

-- Theorem to prove the required number of distinct routes
theorem paula_routes_count : ∃ R : ℕ, R = 8 :=
begin
    sorry
end

end paula_routes_count_l607_607884


namespace cut_square_to_form_octagon_cut_triangle_to_form_20_sided_polygon_l607_607630

theorem cut_square_to_form_octagon (a : ℝ) :
  ∃ (triangles : List (Triangle ℝ)), (∀ t ∈ triangles, is_right_angle t) ∧ rearrange_to_octagon triangles := 
sorry

theorem cut_triangle_to_form_20_sided_polygon (L : ℝ) :
  ∃ (isosceles_triangles : List (Triangle ℝ)), number_of_sides (rearrange_to_polygon isosceles_triangles) = 20 :=
sorry

end cut_square_to_form_octagon_cut_triangle_to_form_20_sided_polygon_l607_607630


namespace annie_overtakes_bonnie_l607_607687

theorem annie_overtakes_bonnie
  (v : ℝ)
  (track_length : ℝ := 400)
  (annie_speed : ℝ := 1.20 * v) :
  let t := 2000 / v,
      bonnie_distance := v * t,
      bonnie_laps := bonnie_distance / track_length,
      annie_distance := annie_speed * t,
      annie_laps := annie_distance / track_length
  in annie_laps = 6 := by
  sorry

end annie_overtakes_bonnie_l607_607687


namespace arithmetic_mean_of_scores_is_89_3_l607_607893

def scores : List ℤ := [87, 94, 85, 92, 90, 88]

def mean (lst : List ℤ) : ℚ :=
  lst.sum / lst.length

theorem arithmetic_mean_of_scores_is_89_3 :
  round (mean scores).toReal = 89.3 := 
sorry

end arithmetic_mean_of_scores_is_89_3_l607_607893


namespace slope_tangent_line_l607_607520

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem slope_tangent_line (h₁ : Differentiable ℝ f) 
                           (h₂ : filter.tendsto (λ x, (f (x + 1) - f 1) / (2 * x)) (nhds 0) (nhds 3)) :
  deriv f 1 = 6 :=
sorry

end slope_tangent_line_l607_607520


namespace sum_largest_odd_factors_l607_607035

/-- Define f(n) as the largest odd factor of n -/
def f (n : ℕ) : ℕ := 
if n = 0 then 0 
else n.div2.pow (n.bits.to_list.filter (λ b, b = tt).length)

/-- Lean statement of the problem -/
theorem sum_largest_odd_factors :
  (∑ i in (finset.range 100).filter (λ x, 51 ≤ x + 1), f (i + 1)) -
  (∑ i in (finset.range 50).filter (λ x, 1 ≤ x + 1), f (i + 1)) = 1656 :=
by sorry

end sum_largest_odd_factors_l607_607035


namespace radii_inequality_l607_607914

variable {R1 R2 R3 r : ℝ}

/-- Given that R1, R2, and R3 are the radii of three circles passing through a vertex of a triangle 
and touching the opposite side, and r is the radius of the incircle of this triangle,
prove that 1 / R1 + 1 / R2 + 1 / R3 ≤ 1 / r. -/
theorem radii_inequality (h_ge : ∀ i : Fin 3, 0 < [R1, R2, R3][i]) (h_incircle : 0 < r) :
  (1 / R1) + (1 / R2) + (1 / R3) ≤ 1 / r :=
  sorry

end radii_inequality_l607_607914


namespace car_X_travel_distance_l607_607622

def car_distance_problem (speed_X speed_Y : ℝ) (delay : ℝ) : ℝ :=
  let t := 7 -- duration in hours computed in the provided solution
  speed_X * t

theorem car_X_travel_distance
  (speed_X speed_Y : ℝ) (delay : ℝ)
  (h_speed_X : speed_X = 35) (h_speed_Y : speed_Y = 39) (h_delay : delay = 48 / 60) :
  car_distance_problem speed_X speed_Y delay = 245 :=
by
  rw [h_speed_X, h_speed_Y, h_delay]
  -- compute the given car distance problem using the values provided
  sorry

end car_X_travel_distance_l607_607622


namespace speed_of_boat_in_still_water_l607_607302

variable (V_b V_s t_up t_down : ℝ)

theorem speed_of_boat_in_still_water (h1 : t_up = 2 * t_down)
  (h2 : V_s = 18) 
  (h3 : ∀ d : ℝ, d = (V_b - V_s) * t_up ∧ d = (V_b + V_s) * t_down) : V_b = 54 :=
sorry

end speed_of_boat_in_still_water_l607_607302


namespace ratio_simple_compound_interest_l607_607229

-- Define the simple interest function
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

-- Define the compound interest function
def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ( (1 + R / 100) ^ T - 1 )

-- Prove the ratio of simple interest to compound interest is 1 / 2 with given conditions
theorem ratio_simple_compound_interest :
  simple_interest 1750 8 3 / compound_interest 4000 10 2 = 1 / 2 :=
by
  have SI := simple_interest 1750 8 3
  have CI := compound_interest 4000 10 2
  sorry

end ratio_simple_compound_interest_l607_607229


namespace find_number_multiplied_l607_607769

theorem find_number_multiplied (m : ℕ) (h : 9999 * m = 325027405) : m = 32505 :=
by {
  sorry
}

end find_number_multiplied_l607_607769


namespace f_pi_over_4_l607_607054

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem f_pi_over_4 (ω φ : ℝ) (h : ω ≠ 0) 
  (symm : ∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) : 
  f ω φ (π / 4) = 2 ∨ f ω φ (π / 4) = -2 := 
by 
  sorry

end f_pi_over_4_l607_607054


namespace a_2_9_and_a_3_25_lambda_neg1_and_an_sn_n_1_only_for_Sn_an_in_Nplus_l607_607791

-- Define the sequence a_n and the sum S_n
def a : ℕ+ → ℕ
| ⟨1, _⟩ := 3
| ⟨n + 1, h⟩ := 2 * a ⟨n, Nat.succ_pos n⟩ + 2^(n + 1) - 1

def S : ℕ+ → ℕ
| ⟨1, _⟩ := a ⟨1, Nat.succ_pos 0⟩
| ⟨n + 1, h⟩ := S ⟨n, Nat.succ_pos n⟩ + a ⟨n + 1, h⟩

-- Proof goals:
theorem a_2_9_and_a_3_25 : a ⟨2, Nat.succ_pos 1⟩ = 9 ∧ a ⟨3, Nat.succ_pos 2⟩ = 25 :=
sorry

theorem lambda_neg1_and_an_sn (lambda : ℝ) (h : λ = -1) : 
  (∀ n : ℕ+, (a n + λ)/2^n ∈ ArithmeticSeq) ∧
  (λ = -1) ∧
  (∃ (an_formula : ℕ → ℕ) (sn_formula : ℕ → ℕ), 
    (∀ n : ℕ, a ⟨n + 1, Nat.succ_pos n⟩ = an_formula n) ∧ 
    (∀ n : ℕ, S ⟨n + 1, Nat.succ_pos n⟩ = sn_formula n) ∧ 
    (an_formula n = n * 2^n + 1 ∧ sn_formula n = (n - 1) * 2^(n + 1) + 2 + n)) :=
sorry

theorem n_1_only_for_Sn_an_in_Nplus : 
  ∀ n : ℕ+, (S n / a n ∈ ℕ+) ↔ n = 1 :=
sorry

end a_2_9_and_a_3_25_lambda_neg1_and_an_sn_n_1_only_for_Sn_an_in_Nplus_l607_607791


namespace find_y_l607_607879

open Locale.Real.Basic

/-- Prove y = 25° given the conditions -/
theorem find_y
  (PQ_parallel_RS : is_parallel PQ RS)
  (PRS_straight : is_straight PRS)
  (angle_PSQ : measure_of_angle PSQ = 85)
  (angle_PRS : measure_of_angle PRS = 115) :
  y = 25 := sorry

end find_y_l607_607879


namespace probability_three_heads_in_five_tosses_l607_607682

theorem probability_three_heads_in_five_tosses :
  let p_head := 1 / 2,
      n := 5,
      k := 3
  in (finset.card (finset.icombine (finset.range n) k) * p_head^k * (1 - p_head)^(n - k)) = 5 / 16 :=
by sorry

end probability_three_heads_in_five_tosses_l607_607682


namespace pin_squares_l607_607950

-- Definition of the problem:
-- There are several identical squares laid out on a rectangular table.
-- We aim to show that we can pin each square to the table with exactly one pin.

theorem pin_squares (n : ℕ) (table : ℝ × ℝ) (squares : fin n → (ℝ × ℝ) × ℝ)
  (identical_squares : ∀ i j, (squares i).snd = (squares j).snd)
  (aligned_with_table : ∀ i, ∃ a b, (squares i).fst = (a, b))
  (may_overlap : ∀ i j, i ≠ j → overlaps (squares i) (squares j)) :
  ∃ pins : fin n → (ℝ × ℝ), (∀ i, ∃ j, pins j ∈ square (squares i).fst (squares i).snd) :=
sorry

end pin_squares_l607_607950


namespace maximum_pq_qr_rs_sp_l607_607220

open Finset

def pq_qr_rs_sp (p q r s : ℕ) : ℕ := p * q + q * r + r * s + s * p

theorem maximum_pq_qr_rs_sp :
  ∃ (p q r s : ℕ), {p, q, r, s} = {2, 4, 6, 8} ∧ pq_qr_rs_sp p q r s = 100 :=
by
  use 8, 4, 2, 6
  simp [pq_qr_rs_sp]
  split
  { norm_num }
  { sorry }

end maximum_pq_qr_rs_sp_l607_607220


namespace problem_one_problem_two_l607_607397

-- Define the given vectors
def vector_oa : ℝ × ℝ := (-1, 3)
def vector_ob : ℝ × ℝ := (3, -1)
def vector_oc (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the subtraction of two 2D vectors
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Define the parallel condition (u and v are parallel if u = k*v for some scalar k)
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1  -- equivalent to u = k*v

-- Define the dot product in 2D
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Problem 1
theorem problem_one (m : ℝ) :
  is_parallel (vector_sub vector_ob vector_oa) (vector_oc m) ↔ m = -1 :=
by
-- Proof omitted
sorry

-- Problem 2
theorem problem_two (m : ℝ) :
  dot_product (vector_sub (vector_oc m) vector_oa) (vector_sub (vector_oc m) vector_ob) = 0 ↔
  m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2 :=
by
-- Proof omitted
sorry

end problem_one_problem_two_l607_607397


namespace westeros_max_cursed_roads_l607_607114

theorem westeros_max_cursed_roads :
  ∀ (V E N : ℕ), V = 1000 ∧ E = 2017 ∧ connected_graph V E ∧ (N = E - 6) →
  7_connected_components_after_removal V E N ∧ N = 2011 :=
by
  intros
  sorry

end westeros_max_cursed_roads_l607_607114


namespace graph_translation_l607_607852

theorem graph_translation (f : ℝ → ℝ) :
  ∀ x, (f (x + 1) - 2) = f (x - 1) - 2 + 3 := 
begin
  sorry
end

end graph_translation_l607_607852


namespace sum_of_x_values_proof_l607_607253

noncomputable def sum_of_x_values : ℝ := 
  (-(-4)) / 1 -- Sum of roots of x^2 - 4x - 7 = 0

theorem sum_of_x_values_proof (x : ℝ) (h : 7 = (x^3 - 2 * x^2 - 8 * x) / (x + 2)) : sum_of_x_values = 4 :=
sorry

end sum_of_x_values_proof_l607_607253


namespace monotonicity_intervals_range_m_real_bounds_for_a_l607_607053

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Define the function g(x) when a = e
def g (x : ℝ) : ℝ := (2 - Real.exp 1) * x

-- Define the function h(x) as f(x) - g(x)
def h (x : ℝ) : ℝ := Real.exp x - 2 * x - 1

-- Monotonicity intervals for h(x)
theorem monotonicity_intervals : 
    (∀ x : ℝ, x < Real.log 2 → deriv h x < 0) ∧ 
    (∀ x : ℝ, x > Real.log 2 → deriv h x > 0) := 
sorry

-- Range of m for the piecewise function F(x) = f(x) for x ≤ m and g(x) for x > m
def F (m : ℝ) (x : ℝ) : ℝ := if x ≤ m then f (Real.exp 1) x else g x

theorem range_m_real : 
    ∀ m : ℝ, 
    (m ∈ Icc 0 (1 / (Real.exp 1 - 2)) ↔ (∀ y : ℝ, ∃ x : ℝ, F m x = y)) := 
sorry

-- Bound for a if f(x1) = f(x2) and |x1 - x2| ≥ 1, for x1, x2 ∈ [0, 2]
theorem bounds_for_a (a x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) 
    (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) (h5 : |x1 - x2| ≥ 1) 
    (h6 : f a x1 = f a x2) : 
    Real.exp 1 - 1 ≤ a ∧ a ≤ (Real.exp 1)^2 - Real.exp 1 := 
sorry

end monotonicity_intervals_range_m_real_bounds_for_a_l607_607053


namespace find_f_neg_one_l607_607145

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then exp x - 1 else -(exp (-x) - 1)

theorem find_f_neg_one : f (-1) = 1 - exp 1 := by
  sorry

end find_f_neg_one_l607_607145


namespace minimum_omega_l607_607144

theorem minimum_omega (ω : ℝ) (ϕ : ℝ) (T : ℝ)
  (hω : ω > 0) (hϕ_gt : 0 < ϕ) (hϕ_lt : ϕ < π)
  (hT_period : T = 2 * π / ω)
  (hT_value : cos (ω * T + ϕ) = sqrt 3 / 2)
  (hx_zero: cos (ω * π / 9 + ϕ) = 0) :
  ω = 3 :=
by sorry

end minimum_omega_l607_607144


namespace find_a_b_and_min_val_l607_607438

-- Given function f(x) = x^3 - x^2 + ax + b and tangent line condition at (0, f(0))
variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + b

-- The conditions
theorem find_a_b_and_min_val 
  (h₁ : f 0 = 1) 
  (h₂ : ∂ (λ x, x^3 - x^2 + a * x + b) / ∂ x | (0 : ℝ) = -1) : 
  a = -1 ∧ b = 1 ∧ ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), 
  (x = -2 → f x = -9) ∧ (x = 1 → f x ≥ f(ℂ.find_val -2 0 (λ x, -x^2 -x + 1))) := 
by 
  sorry

end find_a_b_and_min_val_l607_607438


namespace sum_of_reciprocal_squares_lt_arithmetic_fraction_l607_607930

theorem sum_of_reciprocal_squares_lt_arithmetic_fraction (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range (n+1), (1 : ℚ) / (k+1)^2) < (2*n + 1 : ℚ) / (n + 1) :=
sorry

end sum_of_reciprocal_squares_lt_arithmetic_fraction_l607_607930


namespace range_of_m_l607_607036

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (Set.Ioc 0 2) then 2^x - 1 else sorry

def g (x m : ℝ) : ℝ :=
x^2 - 2*x + m

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2:ℝ) 2, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioc (0:ℝ) 2, f x = 2^x - 1) ∧
  (∀ x1 ∈ Set.Icc (-2:ℝ) 2, ∃ x2 ∈ Set.Icc (-2:ℝ) 2, g x2 m = f x1) 
  → -5 ≤ m ∧ m ≤ -2 :=
sorry

end range_of_m_l607_607036


namespace product_of_real_roots_l607_607756

theorem product_of_real_roots : 
  let f (x : ℝ) := x ^ Real.log x / Real.log 2 
  ∃ r1 r2 : ℝ, (f r1 = 16 ∧ f r2 = 16) ∧ (r1 * r2 = 1) := 
by
  sorry

end product_of_real_roots_l607_607756


namespace inequality_always_holds_l607_607773

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 - m * x - 1 < 0) → -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_always_holds_l607_607773


namespace angle_KDA_l607_607945

theorem angle_KDA {A B C D M K : Type} 
  (rectangle : Rectangle A B C D) 
  (AD_eq_2AB : rectangle.AD = 2 * rectangle.AB)
  (M_mid_AD : M = rectangle.midpoint_AD) 
  (angle_AMK_eq_80 : ∠(A, M, K) = 80)
  (KD_bisects_MKC : isAngleBisector (K, D) (M, K, C)) : 
  ∠(K, D, A) = 35 :=
sorry

end angle_KDA_l607_607945


namespace schedule_lectures_l607_607666

-- Conditions of the problem: Dr. X after Dr. Y, Dr. L after Dr. N
def valid_ordering (lecturers : List String) : Prop :=
  ∃ i j k l, 
    lecturers[i] = "Dr. Y" ∧ lecturers[j] = "Dr. X" ∧ i < j ∧
    lecturers[k] = "Dr. N" ∧ lecturers[l] = "Dr. L" ∧ k < l

theorem schedule_lectures : 
  ∃ (lecturers : List String) (orders : Finset (List String)),
    (orders.card = 180) ∧
    (∀ o ∈ orders, valid_ordering o) :=
sorry

end schedule_lectures_l607_607666


namespace distinct_solutions_difference_l607_607150

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l607_607150


namespace problem1_problem2_l607_607632

-- Problem 1
theorem problem1
  : √12 + (π - 203)^0 + (1 / 2)^(-1) - 6 * Real.tan (Real.pi / 6) = 3 := 
sorry

-- Problem 2
theorem problem2 
  (x y : ℝ) 
  (h1 : x + 2 * y = 4) 
  (h2 : x + 3 * y = 5) 
  : x = 2 ∧ y = 1 := 
sorry

end problem1_problem2_l607_607632


namespace find_meeting_lamp_post_l607_607684

-- Declare the conditions as a structure
structure WalkingProblem :=
  (n : ℕ) -- number of lamps
  (a_start : ℕ) -- initial position of Alla
  (b_start : ℕ) -- initial position of Boris
  (a_position_t : ℕ) -- position of Alla at time t
  (b_position_t : ℕ) -- position of Boris at time t)

-- Example to initialize the structure with our conditions
def example : WalkingProblem :=
  { n := 400,
    a_start := 1,
    b_start := 400,
    a_position_t := 55,
    b_position_t := 321 }

-- Define the meeting lamp post proof problem
theorem find_meeting_lamp_post (w : WalkingProblem)
    (h1 : w.a_position_t = 55)
    (h2 : w.b_position_t = 321) :
    (∃ k : ℕ, k ≤ w.n ∧
      w.a_start + ((k - w.a_start) * 54) / (w.a_position_t - w.a_start) = 
      w.b_start - ((w.b_start - k) * 79) / (w.b_start - w.b_position_t) ∧
      k = 163) := sorry

end find_meeting_lamp_post_l607_607684


namespace factorize_expr_l607_607747

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l607_607747


namespace number_of_integers_in_set_y_l607_607091

open Set

variables (x y : Set ℤ)

theorem number_of_integers_in_set_y
  (hx : x.finite)
  (hy : y.finite)
  (h₁ : x.card = 12)
  (h₂ : (x ∩ y).card = 6)
  (h₃ : ((x \ y) ∪ (y \ x)).card = 18) :
  y.card = 18 :=
sorry

end number_of_integers_in_set_y_l607_607091


namespace molecular_weight_of_1_mole_l607_607249

theorem molecular_weight_of_1_mole (m : ℝ) (w : ℝ) (h : 7 * m = 420) : m = 60 :=
by
  sorry

end molecular_weight_of_1_mole_l607_607249


namespace three_letter_words_with_vowels_equals_189_l607_607070

-- Define the letters and vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'I'}

def vowels := {'A', 'E', 'I'}

-- Define the total number of unrestricted 3-letter words
def total_3_letter_words := (6: ℕ)^3

-- Define the total number of 3-letter words without vowels
def non_vowel_3_letter_words := (3: ℕ)^3

-- Define the number of 3-letter words with at least one vowel
def words_with_at_least_one_vowel := total_3_letter_words - non_vowel_3_letter_words

-- The main theorem statement
theorem three_letter_words_with_vowels_equals_189 : words_with_at_least_one_vowel = 189 := by
  -- Replace with the actual proof
  sorry

end three_letter_words_with_vowels_equals_189_l607_607070


namespace LanceCents_l607_607897

noncomputable def MargaretCents : ℕ := 75
noncomputable def GuyCents : ℕ := 60
noncomputable def BillCents : ℕ := 60
noncomputable def TotalCents : ℕ := 265

theorem LanceCents (lanceCents : ℕ) :
  MargaretCents + GuyCents + BillCents + lanceCents = TotalCents → lanceCents = 70 :=
by
  intros
  sorry

end LanceCents_l607_607897


namespace triangle_cut_l607_607316

theorem triangle_cut (x : ℝ) (hx : x ≥ 6) : 
  ¬ («12 - x, 18 - x, 24 - x» form a triangle) :=
by
  sorry

end triangle_cut_l607_607316


namespace transformed_vertex_l607_607681

def quadratic_func (x : ℝ) : ℝ :=
  (1/2) * x^2 + 3 * x + 5 / 2

def shift_right (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ :=
  λ x, f (x - c)

def shift_up (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x, f x + k

theorem transformed_vertex :
  let f := quadratic_func
  let g := shift_up (shift_right f 2) 3
  ∃ x y : ℝ, 
    g x = y ∧ 
    (∀ z, g z ≥ y) := 
  ∃ x y : ℝ, 
    g (-1) = 1 ∧ 
    (∀ z, g z ≥ 1) :=
sorry

end transformed_vertex_l607_607681


namespace union_A_B_l607_607418

def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem union_A_B : A ∪ B = {x | x > 0} :=
by
  sorry

end union_A_B_l607_607418


namespace solve_lambda_l607_607011

variable (a b : ℝ × ℝ)
variable (lambda : ℝ)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

axiom a_def : a = (-3, 2)
axiom b_def : b = (-1, 0)
axiom perp_def : perpendicular (a.1 + lambda * b.1, a.2 + lambda * b.2) b

theorem solve_lambda : lambda = -3 :=
by
  sorry

end solve_lambda_l607_607011


namespace find_figure_with_2112_gray_squares_l607_607992

/--
Each figure in the series of figures consists of a grid of black, white, and gray squares.
The number of gray squares in the n-th figure is given by:

(2 * n - 1)^2 - n^2 - (n - 1)^2

We want to find the value of n for which the number of gray squares is exactly 2112.
-/
theorem find_figure_with_2112_gray_squares :
  ∃ n : ℕ, (2 * n - 1)^2 - n^2 - (n - 1)^2 = 2112 ∧ n = 33 :=
begin
  use 33,
  -- Simplification and proof go here
  sorry
end

end find_figure_with_2112_gray_squares_l607_607992


namespace train_length_l607_607315

-- Problem definition: Given conditions
def initial_speed := 60 * (1000 / 3600)  -- initial speed in m/s
def acceleration := 2  -- acceleration in m/s^2
def time := 6  -- time in seconds

-- Prove that the length of the train is 136.02 meters
theorem train_length : (initial_speed * time + 1/2 * acceleration * time^2) = 136.02 := by
  sorry

end train_length_l607_607315


namespace maximal_sum_of_xy_l607_607237

theorem maximal_sum_of_xy (x y : ℤ) (h : x^2 + y^2 = 100) : ∃ (s : ℤ), s = 14 ∧ ∀ (u v : ℤ), u^2 + v^2 = 100 → u + v ≤ s :=
by sorry

end maximal_sum_of_xy_l607_607237


namespace ellipse_eccentricity_is_fifth_l607_607573

noncomputable def ellipse_eccentricity (a b c : ℝ) (h : a > b ∧ b > 0) (ellipse_eqn : ∀ x y : ℝ, 
  (x^2 / a^2) + (y^2 / b^2) = 1 → True)
  (vertices_and_foci : A = (-a, 0) ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ D = (0, b))
  (vector_relation : 3 * (D - F₁) = (D - A) + 2 * (D - F₂)) : ℝ :=
  let e : ℝ := c / a in
  e

theorem ellipse_eccentricity_is_fifth (a b c : ℝ) 
  (h : a > b ∧ b > 0) 
  (ellipse_eqn : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 → True)
  (vertices_and_foci : A = (-a, 0) ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ D = (0, b)) 
  (vector_relation : 3 * (D - F₁) = (D - A) + 2 * (D - F₂)) : 
  ellipse_eccentricity a b c h ellipse_eqn vertices_and_foci vector_relation = 1 / 5 :=
sorry

end ellipse_eccentricity_is_fifth_l607_607573


namespace abs_diff_is_correct_l607_607731

-- Definitions based on given conditions
def C : ℕ
def D : ℕ
def abs_diff (C D : ℕ) : ℕ := abs (C - D)

-- Conditions from the problem
axiom single_digit_C : C < 3
axiom single_digit_D : D < 3
axiom base3_sum : (D*27 + 2*9 + D*3 + C) + (C*27 + 3*9 + 2*3 + 4) = C*27 + 2*9 + 4*3 + 1

-- Final proof goal
theorem abs_diff_is_correct : abs_diff C D = 1 := sorry

end abs_diff_is_correct_l607_607731


namespace cos_double_angle_l607_607779

theorem cos_double_angle (α : ℝ) (h : Real.sin (π/6 - α) = 1/3) :
  Real.cos (2 * (π/3 + α)) = -7/9 :=
by
  sorry

end cos_double_angle_l607_607779


namespace sum_of_translated_parabolas_l607_607305

noncomputable def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := - (a * x^2 + b * x + c)

noncomputable def translated_right (a b c : ℝ) (x : ℝ) : ℝ := parabola_equation a b c (x - 3)

noncomputable def translated_left (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 3)

theorem sum_of_translated_parabolas (a b c x : ℝ) : 
  (translated_right a b c x) + (translated_left a b c x) = -12 * a * x - 6 * b :=
sorry

end sum_of_translated_parabolas_l607_607305


namespace binomial_seven_four_l607_607720

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607720


namespace S_greater_than_one_l607_607398

noncomputable def S : ℝ :=
  (π / 200000) * (∑ k in finset.range 100000, real.sin ((k + 1) * π / 200000))

theorem S_greater_than_one :
  S > 1 :=
sorry

end S_greater_than_one_l607_607398


namespace find_C_marks_l607_607729

theorem find_C_marks :
  let english := 90
  let math := 92
  let physics := 85
  let biology := 85
  let avg_marks := 87.8
  let total_marks := avg_marks * 5
  let other_marks := english + math + physics + biology
  ∃ C : ℝ, total_marks - other_marks = C ∧ C = 87 :=
by
  sorry

end find_C_marks_l607_607729


namespace calc_problem_system_of_equations_l607_607635

-- For calculation problem
theorem calc_problem : 
  sqrt 12 + (pi - 203) ^ 0 + (1 / 2: ℝ) ^ (-1) - 6 * tan (real.pi / 6) = 3 := by sorry

-- For the system of equations problem
theorem system_of_equations :
  ∃ (x y : ℝ), 
  (x + 2 * y = 4) ∧ (x + 3 * y = 5) ∧ (x = 2) ∧ (y = 1) := by sorry

end calc_problem_system_of_equations_l607_607635


namespace domain_of_p_l607_607968

theorem domain_of_p (h : ℝ → ℝ) (h_domain : ∀ x, -10 ≤ x → x ≤ 6 → ∃ y, h x = y) :
  ∀ x, -1.2 ≤ x ∧ x ≤ 2 → ∃ y, h (-5 * x) = y :=
by
  sorry

end domain_of_p_l607_607968


namespace paperback_copies_sold_l607_607627

theorem paperback_copies_sold
  (H : ℕ) (P : ℕ)
  (h1 : H = 36000)
  (h2 : P = 9 * H)
  (h3 : H + P = 440000) :
  P = 360000 := by
  sorry

end paperback_copies_sold_l607_607627


namespace intersection_of_sets_l607_607833
  
   def A := {x : ℕ | x - 3 ≤ 0}
   def B := {x : ℤ | x^2 + x - 2 ≤ 0}
  
   theorem intersection_of_sets : A ∩ (B ∩ {x : ℤ | x ≥ 0}) = {0, 1} :=
   by {
     sorry
   }
   
end intersection_of_sets_l607_607833


namespace ratio_OR_OQ_intersection_NR_Parabola_l607_607405

variables {m : ℝ} (h_pos : 0 < m)

def Circle : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 3 * m ^ 2}
def Parabola : set (ℝ × ℝ) := {p | p.2 ^ 2 = 2 * m * p.1}

def P : ℝ × ℝ := (m, -real.sqrt (2 * m))
def N : ℝ × ℝ := (0, -real.sqrt (3) * m)
def R : ℝ × ℝ := (2 * m, (real.sqrt (3) - 2 * real.sqrt (2)) * m)
def Q : ℝ × ℝ := (8 * m / ((real.sqrt (3) - 2 * real.sqrt (2)) ^ 2), 4 * m / (real.sqrt (3) - 2 * real.sqrt (2)))

theorem ratio_OR_OQ : 
  ∀ (O : ℝ × ℝ), 
  ∀ (OR_len OQ_len : ℝ), 
  OR_len = dist O R → 
  OQ_len = dist O Q → 
  O = (0, 0) →
  |OR_len / OQ_len| = (11 - 4 * real.sqrt (6)) / 4 :=
sorry

theorem intersection_NR_Parabola : 
  ∀ (NR_slope : ℝ),
  NR_slope = (real.sqrt (3) - real.sqrt (2)) → 
  ∃ (points : list (ℝ × ℝ)), 
  length points = 2 ∧ 
  (∀ q ∈ points, q ∈ Parabola ∧ ∃ k : ℝ, q.2 = NR_slope * q.1 - real.sqrt (3) * m) :=
sorry

end ratio_OR_OQ_intersection_NR_Parabola_l607_607405


namespace find_a_l607_607037

def imaginary_unit := Complex.i
def line_eq (x y : ℝ) := x - y + 1 = 0

theorem find_a (a : ℝ) :
  line_eq ((a - 1) / 2) (-(a + 1) / 2) → a = -1 :=
by
  intro h
  sorry

end find_a_l607_607037


namespace solution_y_chemical_A_percentage_l607_607956

def percent_chemical_A_in_x : ℝ := 0.30
def percent_chemical_A_in_mixture : ℝ := 0.32
def percent_solution_x_in_mixture : ℝ := 0.80
def percent_solution_y_in_mixture : ℝ := 0.20

theorem solution_y_chemical_A_percentage
  (P : ℝ) 
  (h : percent_solution_x_in_mixture * percent_chemical_A_in_x + percent_solution_y_in_mixture * P = percent_chemical_A_in_mixture) :
  P = 0.40 :=
sorry

end solution_y_chemical_A_percentage_l607_607956


namespace parabola_transformation_correct_l607_607487

-- Definitions and conditions
def original_parabola (x : ℝ) : ℝ := 2 * x^2

def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 3)^2 - 4

-- Theorem to prove that the above definition is correct
theorem parabola_transformation_correct : 
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 3)^2 - 4 :=
by
  intros x
  rfl -- This uses the definition of 'transformed_parabola' directly

end parabola_transformation_correct_l607_607487


namespace alice_can_prevent_bob_winning_l607_607591

noncomputable def minPebbles (N : ℕ) : ℕ :=
  let k := N / 2
  k * (k + 1) - 1

theorem alice_can_prevent_bob_winning :
  let N := 60
  minPebbles N = 960 :=
by
  let k := 60 / 2
  have := nat.div_eq_of_lt nat.succ_pos'
  simp [minPebbles, this]
  sorry

end alice_can_prevent_bob_winning_l607_607591


namespace find_second_number_l607_607977

theorem find_second_number :
  let avg1 := (24 + 35 + 58) / 3 in
  let avg2 := (19 + x + 29) / 3 in
  avg1 = avg2 + 6 →
  x = 51 :=
by
  intros avg1 avg2 h
  have h1 : avg1 = 39 := by sorry
  have h2 : avg2 = 33 := by sorry
  rw [h1, h2] at h
  linarith

end find_second_number_l607_607977


namespace positive_integers_satisfy_eq_l607_607363

theorem positive_integers_satisfy_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + 1 = c! → (a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end positive_integers_satisfy_eq_l607_607363


namespace exists_k_sum_Nq_gt_one_l607_607903

/-- Let α be an irrational number. For every positive integer q, N_q(α) is the distance 
to the nearest fraction with denominator q (not necessarily in reduced form). -/
theorem exists_k_sum_Nq_gt_one (α : ℝ) (hα : irrational α) :
  ∃ k : ℕ, ∑ q in finset.range (k + 1), (λ (q : ℕ), 
    if q = 0 then 0 else 
    let N_q := (finset.min' (finset.image (λ p : ℤ, abs (α - p / q)) 
      (finset.Icc (-q) q)) (by simp [hα.ne])) in N_q) q > 1 :=
begin
  sorry
end

end exists_k_sum_Nq_gt_one_l607_607903


namespace sum_of_g_49_l607_607511

def f (x : ℝ) := 4 * x^2 - 3
def g (y : ℝ) := y^2 + 2 * y + 2

theorem sum_of_g_49 : (g 49) = 30 :=
  sorry

end sum_of_g_49_l607_607511


namespace P_Q_R_S_concyclic_l607_607786

variables {A B C D P Q R S O : Type} [AffineSpace ℝ A]
variables {AB BC CD DA PR QS : AffineSubspace ℝ A}

-- Conditions
def convex_quadrilateral (A B C D : A) : Prop := sorry -- convex quadrilateral definition
def on_segment (P : A) (AB : AffineSubspace ℝ A) : Prop := sorry -- point on segment definition
def intersect (PR QS : AffineSubspace ℝ A) (O : A) : Prop := sorry -- intersecting definition
def divide_into_four_convex_quadrilaterals (ABCD PR QS : AffineSubspace ℝ A) : Prop := sorry -- dividing into four convex quads definition
def diagonals_perpendicular (PR QS : AffineSubspace ℝ A) : Prop := sorry -- perpendicular diagonals definition
def concyclic (P Q R S : A) : Prop := sorry -- concyclic definition

-- Given conditions translated to definitions
axiom h1 : convex_quadrilateral A B C D
axiom h2 : on_segment P AB
axiom h3 : on_segment Q BC
axiom h4 : on_segment R CD
axiom h5 : on_segment S DA
axiom h6 : intersect PR QS O
axiom h7 : divide_into_four_convex_quadrilaterals ABCD PR QS
axiom h8 : diagonals_perpendicular PR QS

-- Statement to prove
theorem P_Q_R_S_concyclic : concyclic P Q R S :=
by
  sorry

end P_Q_R_S_concyclic_l607_607786


namespace find_ab_l607_607009

variable (a b m n : ℝ)

theorem find_ab (h1 : (a + b)^2 = m) (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 :=
by
  sorry

end find_ab_l607_607009


namespace concyclic_circumcircle_concentric_l607_607175

open EuclideanGeometry
open Triangle
open Circle

theorem concyclic_circumcircle_concentric
  (ABC : Triangle)
  (D E F : Point)
  (hD : tangent_circle ABC.D BC = D)
  (hE : tangent_circle ABC.E CA = E)
  (hF : tangent_circle ABC.F AB = F)
  (K L M N U V : Point)
  (hK : is_midpoint_of K (segment AE))
  (hL : is_midpoint_of L (segment AF))
  (hM : is_midpoint_of M (segment BF))
  (hN : is_midpoint_of N (segment BD))
  (hU : is_midpoint_of U (segment CD))
  (hV : is_midpoint_of V (segment CE)) :
  concentric (circumcircle_of (triangle_of_lines K L M))
             (circumcircle_of ABC) :=
sorry

end concyclic_circumcircle_concentric_l607_607175


namespace combined_work_rate_l607_607284

-- Define the problem conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 2
def C_work_rate := 1 / 8

-- Problem statement: The combined work rate of A, B, and C is 7/8
theorem combined_work_rate :
  A_work_rate + B_work_rate + C_work_rate = 7 / 8 :=
begin
  sorry
end

end combined_work_rate_l607_607284


namespace product_of_points_l607_607683

def g (n : ℕ) : ℕ :=
  if n % 12 = 0 then 8
  else if n % 3 = 0 then 3
  else if n % 4 = 0 then 1
  else 0

def allie_points := g 6 + g 3 + g 4 + g 1
def betty_points := g 12 + g 9 + g 4 + g 2

theorem product_of_points : allie_points * betty_points = 84 :=
by
  have h_allie : allie_points = 7 := by sorry
  have h_betty : betty_points = 12 := by sorry
  rw [h_allie, h_betty]
  norm_num

end product_of_points_l607_607683


namespace midpoint_probability_l607_607137

theorem midpoint_probability (T : set (ℤ × ℤ × ℤ)) 
  (hT : T = { p | ∃ (x y z : ℤ), 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5 ∧ p = (x, y, z) }) :
  ∃ p q : ℕ, nat.coprime p q ∧ (p + q) = 156 ∧ (p / q : ℚ) = ((10 * 13 * 18 - 120) : ℚ) / (7140 - 120) :=
by
  sorry -- proof omitted

end midpoint_probability_l607_607137


namespace intersection_A_B_l607_607832

def A : Set ℤ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 2, 3} :=
by sorry

end intersection_A_B_l607_607832


namespace distance_from_negative_two_is_three_l607_607543

theorem distance_from_negative_two_is_three (x : ℝ) : abs (x + 2) = 3 → (x = -5) ∨ (x = 1) :=
  sorry

end distance_from_negative_two_is_three_l607_607543


namespace f_pow2_l607_607781

noncomputable def f (n : ℕ) : ℚ := (List.range n).map (λ i => (1 : ℚ) / (i + 1)).sum

theorem f_pow2 (n : ℕ) (hn : n > 0) :
  f (2^n) ≥ (n + 2) / 2 := by
  sorry

end f_pow2_l607_607781


namespace min_area_circle_tangent_l607_607385

theorem min_area_circle_tangent (h : ∀ (x : ℝ), x > 0 → y = 2 / x) : 
  ∃ (a b r : ℝ), (∀ (x : ℝ), x > 0 → 2 * a + b = 2 + 2 / x) ∧
  (∀ (x : ℝ), x > 0 → (x - 1)^2 + (y - 2)^2 = 5) :=
sorry

end min_area_circle_tangent_l607_607385


namespace a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l607_607845

variable (a0 a1 a2 a3 a4 a5 : ℝ)

noncomputable def polynomial (x : ℝ) : ℝ :=
  a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5

theorem a3_is_neg_10 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a3 = -10 :=
sorry

theorem a1_a3_a5_sum_is_neg_16 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a1 + a3 + a5 = -16 :=
sorry

end a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l607_607845


namespace employee_salary_amount_l607_607180

theorem employee_salary_amount (total_revenue : ℝ) (ratio_salary : ℝ) (ratio_stock : ℝ) (total_parts : ℝ) (salary_ratio_fraction : ℝ) :
  total_revenue = 3000 →
  ratio_salary = 4 →
  ratio_stock = 11 →
  total_parts = ratio_salary + ratio_stock →
  salary_ratio_fraction = ratio_salary / total_parts →
  salary_ratio_fraction * total_revenue = 800 :=
by
  intros h_total_revenue h_ratio_salary h_ratio_stock h_total_parts h_salary_ratio_fraction
  rw [h_total_revenue, h_ratio_salary, h_ratio_stock, h_total_parts, h_salary_ratio_fraction]
  sorry

end employee_salary_amount_l607_607180


namespace find_original_number_l607_607304

theorem find_original_number (x : ℝ)
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 :=
sorry

end find_original_number_l607_607304


namespace leo_average_speed_last_segment_l607_607898

theorem leo_average_speed_last_segment :
  let total_distance := 135
  let total_time_hr := 135 / 60.0
  let segment_time_hr := 45 / 60.0
  let first_segment_distance := 55 * segment_time_hr
  let second_segment_distance := 70 * segment_time_hr
  let last_segment_distance := total_distance - (first_segment_distance + second_segment_distance)
  last_segment_distance / segment_time_hr = 55 :=
by
  sorry

end leo_average_speed_last_segment_l607_607898


namespace black_cubes_multiple_of_4_l607_607291

def edge_length := 10
def total_cubes := 1000
def black_cubes := 500
def white_cubes := 500
def removed_cubes := 100
def rods := 300

-- Condition: A cube with edge length 10 is constructed from 500 black unit cubes and 500 white unit cubes
def constructed_cube := (total_cubes = black_cubes + white_cubes) ∧ (black_cubes = 500) ∧ (white_cubes = 500)

-- Condition: The unit cubes are arranged such that every two adjacent faces are of different colors
def proper_arrangement := ∀ unit_cube1 unit_cube2, adjacent_faces unit_cube1 unit_cube2 → diff_colors unit_cube1 unit_cube2

-- Condition: From this cube, 100 unit cubes are removed
def removed_unit_cubes := removed_cubes = 100

-- Condition: Each of the 300 \(1 \times 1 \times 10\) rods parallel to the edges of the cube is missing exactly 1 unit cube
def rods_conditions := ∀ rod, rod ∈ rods → missing_unit_cubes rod = 1

theorem black_cubes_multiple_of_4 :
  constructed_cube ∧ proper_arrangement ∧ removed_unit_cubes ∧ rods_conditions → ∃ n, n ∈ ℕ ∧ (4 * n) = (removed_black_cubes) := 
sorry

end black_cubes_multiple_of_4_l607_607291


namespace midpoint_probability_l607_607138

theorem midpoint_probability (T : set (ℤ × ℤ × ℤ)) 
  (hT : T = { p | ∃ (x y z : ℤ), 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5 ∧ p = (x, y, z) }) :
  ∃ p q : ℕ, nat.coprime p q ∧ (p + q) = 156 ∧ (p / q : ℚ) = ((10 * 13 * 18 - 120) : ℚ) / (7140 - 120) :=
by
  sorry -- proof omitted

end midpoint_probability_l607_607138


namespace possible_values_of_k_l607_607990

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3)
  (moves : list (ℕ × ℕ → ℕ × ℕ)) :
  ∃ k, (∀ a b ∈ (list.range (n + 1)).tail, k = (a + b).uniform_nat (abs (a - b)).uniform_nat) ∧ (∃ m : ℕ, k = 2 ^ m ∧ k ≥ n) :=
sorry

end possible_values_of_k_l607_607990


namespace angela_finished_9_problems_l607_607327

def martha_problems : Nat := 2

def jenna_problems : Nat := 4 * martha_problems - 2

def mark_problems : Nat := jenna_problems / 2

def total_problems : Nat := 20

def total_friends_problems : Nat := martha_problems + jenna_problems + mark_problems

def angela_problems : Nat := total_problems - total_friends_problems

theorem angela_finished_9_problems : angela_problems = 9 := by
  -- Placeholder for proof steps
  sorry

end angela_finished_9_problems_l607_607327


namespace center_of_symmetry_of_cos_3x_minus_pi_over_4_l607_607574

noncomputable def is_center_of_symmetry (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
∀ (x : ℝ), g (2 * p.1 - x) = g x

theorem center_of_symmetry_of_cos_3x_minus_pi_over_4 :
  is_center_of_symmetry (λ x : ℝ, cos (3 * x - π / 4)) (-π / 12, 0) :=
by
  sorry

end center_of_symmetry_of_cos_3x_minus_pi_over_4_l607_607574


namespace find_a_b_find_min_g_l607_607439

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - x^2 + a * x + b

theorem find_a_b (a b : ℝ) (h_tangent : ∀ x, f x a b = -x + 1 → x = 0) : a = -1 ∧ b = 1 :=
by
  have h₀ : f 0 a b = b := by simp [f]
  have h₁ : -0 + 1 = 1 := by norm_num
  have h₂ : b = 1 := by rw [←h_tangent 0 h₁]
  have hf' : ∀ x, deriv (f x a b) = 3 * x^2 - 2 * x + a := by simp [f, deriv]
  have h₃ : deriv (f 0 a b) = a := by simp [hf']
  have h_slope : ∀ x, deriv (λ x, -x + 1) x = -1 := by simp [deriv]
  have h₄ : a = -1 := by rw [←h_slope 0, h₃]
  exact ⟨h₄, h₂⟩

noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem find_min_g : ∃ x ∈ Icc (-2 : ℝ) 2, g x = -9 :=
by
  have eval_g : ∀ x, g x = x^3 - x^2 - x + 1 := by simp [g]
  have h_g_neg2 : g (-2) = -9 := by norm_num [g, eval_g]
  use -2
  split
  · norm_num
  · exact h_g_neg2

end find_a_b_find_min_g_l607_607439


namespace train_length_l607_607678

theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) (total_distance : ℝ) (train_length : ℝ) 
  (h1 : speed = 48) (h2 : time = 45) (h3 : bridge_length = 300)
  (h4 : total_distance = speed * time) (h5 : train_length = total_distance - bridge_length) : 
  train_length = 1860 :=
sorry

end train_length_l607_607678


namespace responses_needed_l607_607461

theorem responses_needed (p : ℝ) (q : ℕ) (r : ℕ) : 
  p = 0.6 → q = 370 → r = 222 → 
  q * p = r := 
by
  intros hp hq hr
  rw [hp, hq] 
  sorry

end responses_needed_l607_607461


namespace initial_markup_percentage_l607_607307

theorem initial_markup_percentage (C : ℝ) (M : ℝ) :
  (C > 0) →
  (1 + M) * 1.25 * 0.90 = 1.35 →
  M = 0.2 :=
by
  intros
  sorry

end initial_markup_percentage_l607_607307


namespace distance_between_P_and_Q_is_correct_l607_607136

noncomputable def distance_PQ : ℝ :=
  let x₁ := 240 / 41
  let y₁ := 17 * x₁ / 6
  let x₂ := 560 / 41
  let y₂ := 5 * x₂ / 8
  real.sqrt ((x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2)

theorem distance_between_P_and_Q_is_correct :
  let R := (10, 5)
  let P := (240 / 41, 17 * (240 / 41) / 6)
  let Q := (560 / 41, 5 * (560 / 41) / 8)
  P.1 + Q.1 = 20 →
  (P.2 + Q.2) / 2 = R.2 →
  distance_PQ = real.sqrt ((-320 / 41) ^ 2 + (-2810 / 123) ^ 2) :=
by
  intros
  sorry

end distance_between_P_and_Q_is_correct_l607_607136


namespace evaluate_g_h_2_l607_607079

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l607_607079


namespace julia_average_speed_l607_607499

theorem julia_average_speed :
  let distance1 := 45
  let speed1 := 15
  let distance2 := 15
  let speed2 := 45
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 18 := by
sorry

end julia_average_speed_l607_607499


namespace intersection_point_product_l607_607413

open Real

noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}

def line (α : ℝ) (P : ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = P.1 + t * cos α ∧ p.2 = P.2 + t * sin α}

def pointP : ℝ × ℝ := (2, 2)
def angle : ℝ := π / 3

theorem intersection_point_product (A B : ℝ × ℝ)
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (hA' : A ∈ line angle pointP)
  (hB' : B ∈ line angle pointP) :
  (dist pointP A) * (dist pointP B) = 8 :=
sorry

end intersection_point_product_l607_607413


namespace trip_distance_l607_607870

theorem trip_distance (D : ℝ) (t1 t2 : ℝ) :
  (30 / 60 = t1) →
  (70 / 35 = t2) →
  (t1 + t2 = 2.5) →
  (40 = D / (t1 + t2)) →
  D = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end trip_distance_l607_607870


namespace sum_of_solutions_eq_4_l607_607967

def t (x : ℝ) : ℝ :=
  (1 / 2) ^ x + (2 / 3) ^ x + (5 / 6) ^ x

theorem sum_of_solutions_eq_4 :
  let solutions := {x : ℝ | (t x) = 1 ∨ (t x) = 2 ∨ (t x) = 3}
  ∑ x in solutions, x = 4 :=
sorry

end sum_of_solutions_eq_4_l607_607967


namespace ellipse_with_foci_on_x_axis_l607_607014

variable (m : ℝ)

def curve (m : ℝ) : Prop := (2 - m) * x^2 + (m + 1) * y^2 = 1

theorem ellipse_with_foci_on_x_axis
  (hm : m ∈ set.Ioo (1/2 : ℝ) 2) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (curve m = (x^2 / a^2) + (y^2 / b^2) = 1)) :=
sorry

end ellipse_with_foci_on_x_axis_l607_607014


namespace a_plus_b_eq_14_l607_607906

theorem a_plus_b_eq_14 (a b : ℝ) (h : (X ^ 3 + a * X + b).is_root (2 + complex.I * real.sqrt 2)) : a + b = 14 :=
sorry

end a_plus_b_eq_14_l607_607906


namespace westeros_max_cursed_roads_l607_607113

theorem westeros_max_cursed_roads :
  ∀ (V E N : ℕ), V = 1000 ∧ E = 2017 ∧ connected_graph V E ∧ (N = E - 6) →
  7_connected_components_after_removal V E N ∧ N = 2011 :=
by
  intros
  sorry

end westeros_max_cursed_roads_l607_607113


namespace gcd_689_1021_l607_607386

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 :=
by sorry

end gcd_689_1021_l607_607386


namespace range_eccentricity_l607_607812

noncomputable def semiLatusRectum (a b : ℝ) : ℝ := b^2 / a

noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 - (b^2 / a^2))

theorem range_eccentricity 
  (a b : ℝ) (h : a > b) (c : ℝ) 
  (heq : c = semiLatusRectum a b) 
  (ineq : b < (b / 2 + c) ∧ (b / 2 + c) < a) :
  (real.sqrt 5 / 5) < eccentricity a b ∧ eccentricity a b < (3 / 5) :=
sorry

end range_eccentricity_l607_607812


namespace sin_2012_is_negative_l607_607587

theorem sin_2012_is_negative : ∃ θ : ℝ, θ = 2012 * (π / 180) ∧ sin θ < 0 :=
by
  have h : 2012 * (π / 180) = (212 + 360 * 5) * (π / 180) := by sorry
  -- Using θ == 212+(5 * 360) degrees, a transformation we showed is possible
  let θ := (212 * (π / 180));
  use 212 * (π / 180)
  split
  { rw ← h
    exact rfl }
  { -- here, we would complete the proof showing sin is negative in 3rd quadrant
    sorry }

end sin_2012_is_negative_l607_607587


namespace product_of_roots_eq_neg9_l607_607390

theorem product_of_roots_eq_neg9 :
  ∀ x, x^2 + 6 * x - 9 = 0 → (∏ r in ({x | (x^2 + 6 * x - 9) = 0}.to_finset), r) = -9 := 
sorry

end product_of_roots_eq_neg9_l607_607390


namespace black_car_overtakes_red_car_in_one_hour_l607_607243

-- Define the speeds of the cars
def red_car_speed := 30 -- in miles per hour
def black_car_speed := 50 -- in miles per hour

-- Define the initial distance between the cars
def initial_distance := 20 -- in miles

-- Calculate the time required for the black car to overtake the red car
theorem black_car_overtakes_red_car_in_one_hour : initial_distance / (black_car_speed - red_car_speed) = 1 := by
  sorry

end black_car_overtakes_red_car_in_one_hour_l607_607243


namespace all_numbers_in_table_are_even_l607_607303

noncomputable def table_10x10 := Fin 10 → Fin 10 → ℕ

def all_even (tbl : table_10x10) : Prop :=
  ∀ i j, tbl i j % 2 = 0

def sum_in_subtable (tbl : table_10x10) (rows cols : Fin 10 → Fin 5) : ℕ :=
  ∑ i : Fin 5, ∑ j : Fin 5, tbl (rows i) (cols j)

def condition (tbl : table_10x10) : Prop :=
  ∀ (rows cols : Fin 10 → Fin 5), sum_in_subtable tbl rows cols % 2 = 0

theorem all_numbers_in_table_are_even
  (tbl : table_10x10)
  (h_condition : condition tbl) :
  all_even tbl := sorry

end all_numbers_in_table_are_even_l607_607303


namespace age_ratio_l607_607582

theorem age_ratio (V A : ℕ) (h1 : V - 5 = 16) (h2 : V * 2 = 7 * A) :
  (V + 4) * 2 = (A + 4) * 5 := 
sorry

end age_ratio_l607_607582


namespace matrix_G_relation_G_1002_G_1004_minus_G_1003_squared_l607_607513

noncomputable def matrix_power (A : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  A ^ n

def G : ℕ → ℤ
| 0     => 0
| 1     => 1
| (n+2) => (G (n+1)) - (G n)

theorem matrix_G_relation (n : ℕ) (h : n > 0) :
  matrix_power (Matrix.vecCons (Matrix.vecCons 1 -1) (Matrix.vecCons 1 0)) n =
  Matrix.vecCons (Matrix.vecCons (G (n + 1)) (G n))
                 (Matrix.vecCons (G n)       (G (n - 1))) :=
sorry

theorem G_1002_G_1004_minus_G_1003_squared :
  G 1002 * G 1004 - (G 1003)^2 = 1 :=
sorry

end matrix_G_relation_G_1002_G_1004_minus_G_1003_squared_l607_607513


namespace houston_firewood_l607_607501

theorem houston_firewood (k e h : ℕ) (k_collected : k = 10) (e_collected : e = 13) (total_collected : k + e + h = 35) : h = 12 :=
by
  sorry

end houston_firewood_l607_607501


namespace radian_measure_of_sector_l607_607425

theorem radian_measure_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  let α := 2 * S / r^2 
  in α = 2 :=
by
  sorry

end radian_measure_of_sector_l607_607425


namespace acute_triangle_inequality_l607_607550

section AcuteAngledTriangle

variables {A B C : Point} 
variable (ABC : Triangle A B C)

-- Condition: Triangle is acute-angled
axiom acute_angled_triangle (h : ∀ (θ : Angle A B C), θ < (π / 2)) : True

-- Definitions of semi-perimeter and circumradius
def semi_perimeter (ABC : Triangle A B C) := (Triangle.perimeter ABC) / 2
def circumradius (ABC : Triangle A B C) := Triangle.circumradius ABC

-- Hypothesis: Triangle ABC is acute-angled
variable (h_acute: acute_angled_triangle ABC)

-- The theorem to prove
theorem acute_triangle_inequality (ABC : Triangle A B C) (h_acute: acute_angled_triangle ABC) :
  semi_perimeter ABC > 2 * circumradius ABC := 
sorry

end AcuteAngledTriangle

end acute_triangle_inequality_l607_607550


namespace sin_plus_cos_of_acute_angle_l607_607846

variable {θ : Real} (b : Real)

theorem sin_plus_cos_of_acute_angle (h1 : 0 < θ ∧ θ < Real.pi / 2) (h2 : Real.cos (2 * θ) = b) :
  Real.sin θ + Real.cos θ = sqrt (1 + sqrt ((1 + b) / 2)) :=
by
  sorry

end sin_plus_cos_of_acute_angle_l607_607846


namespace move_point_right_l607_607875

theorem move_point_right 
  (x y : ℤ)
  (h : (x, y) = (2, -1)) :
  (x + 3, y) = (5, -1) := 
by
  sorry

end move_point_right_l607_607875


namespace function_domain_l607_607384

noncomputable def domain := {x : ℝ | 0 < x ∧ x ≤ 3 ∧ ∃ k : ℤ, - real.pi / 6 + k * real.pi < x ∧ x < real.pi / 6 + k * real.pi}

theorem function_domain : domain = ((set.Ioo 0 (real.pi / 6)) ∪ (set.Ioc (5 * real.pi / 6) 3)) :=
sorry

end function_domain_l607_607384


namespace sqrt_2_minus_x_meaningful_l607_607467

theorem sqrt_2_minus_x_meaningful (x : ℝ) : 
  x = -1 ↔ (√(2 - x)).IsReal ∧ (x = 4 ∨ x = π ∨ x = -1 ∨ x = 3) :=
by 
  sorry

end sqrt_2_minus_x_meaningful_l607_607467


namespace problem1_problem2_l607_607633

-- Problem 1
theorem problem1
  : √12 + (π - 203)^0 + (1 / 2)^(-1) - 6 * Real.tan (Real.pi / 6) = 3 := 
sorry

-- Problem 2
theorem problem2 
  (x y : ℝ) 
  (h1 : x + 2 * y = 4) 
  (h2 : x + 3 * y = 5) 
  : x = 2 ∧ y = 1 := 
sorry

end problem1_problem2_l607_607633


namespace lines_without_common_point_are_not_nececssarily_skew_l607_607246

-- Definitions based on conditions:
def are_skew (l1 l2 : line) : Prop := ¬ ∃ p : point, p ∈ l1 ∧ p ∈ l2 ∧ l1 ∦ l2
def are_parallel (l1 l2 : line) : Prop := ∀ p q : point, (p ∈ l1 ∧ q ∈ l2) → (p - q) ∥ (q - p)

-- The mathematical equivalent proof problem translated into Lean 4:
theorem lines_without_common_point_are_not_nececssarily_skew (l1 l2 : line) :
  (¬ ∃ p : point, p ∈ l1 ∧ p ∈ l2) → (are_parallel l1 l2 ∨ are_skew l1 l2) := 
by
  sorry

end lines_without_common_point_are_not_nececssarily_skew_l607_607246


namespace integral_cos_over_sqrt_sin_squared_integral_sin_to_the_fifth_integral_one_over_tan_squared_integral_cot_to_the_sixth_l607_607387

-- Integrals problem proof statements in Lean 4

-- 1. Prove the integral ∫ (cos x) / sqrt(3 + (sin x)^2) dx = ln (sin x + sqrt(3 + (sin x)^2)) + C
theorem integral_cos_over_sqrt_sin_squared (C : ℝ) :
  ∫ (x : ℝ) in a..b, (cos x) / sqrt(3 + (sin x)^2) = 
    ln (sin x + sqrt(3 + (sin x)^2)) + C := sorry

-- 2. Prove the integral ∫ (sin x)^5 dx = -cos x + (2/3) cos^3 x - (1/5) cos^5 x + C
theorem integral_sin_to_the_fifth (C : ℝ) :
  ∫ (x : ℝ) in a..b, (sin x)^5 = 
    -cos x + (2/3) cos^3 x - (1/5) cos^5 x + C := sorry

-- 3. Prove the integral ∫ (1 / (3 (tan x)^2 + 5)) * (dx / (cos x)^2) = (1 / sqrt(15)) arctan (sqrt(3) (tan x) / sqrt(5)) + C
theorem integral_one_over_tan_squared (C : ℝ) :
  ∫ (x : ℝ) in a..b, (1 / (3 * (tan x)^2 + 5)) * (1 / (cos x)^2) = 
    (1 / sqrt(15)) * arctan (sqrt(3) * tan x / sqrt(5)) + C := sorry

-- 4. Prove the integral ∫ (cot x)^6 dx = - (cot^5 x / 5 + cot^3 x / 3 - cot x - arctan(cot x)) + C
theorem integral_cot_to_the_sixth (C : ℝ) :
  ∫ (x : ℝ) in a..b, (cot x)^6 = 
    - ((cot x)^5 / 5 - (cot x)^3 / 3 + cot x - (arctan (cot x))) + C := sorry

end integral_cos_over_sqrt_sin_squared_integral_sin_to_the_fifth_integral_one_over_tan_squared_integral_cot_to_the_sixth_l607_607387


namespace max_people_no_5_consecutive_max_people_no_4_consecutive_l607_607599

-- First Problem: Maximum 18 people with no 5 consecutive seats
theorem max_people_no_5_consecutive (rows cols : ℕ) (total_seats : ℕ) (dist_between : ℕ)
  (restriction_5 : ∀ r c : ℕ, r < rows → c < cols → c + 5 ≤ cols → ∑ i in finset.range 5, (seat_occ r (c + i)) = 0)
  : rows = 6 → cols = 6 → dist_between > 1 → total_seats = 36 → 
    ∃ max_people : ℕ, max_people = 18 ∧ ∀ arrangement, valid_arrangement arrangement → 
    count_occupied arrangement ≤ max_people := sorry

-- Second Problem: Maximum 28 people with no 4 consecutive seats
theorem max_people_no_4_consecutive (rows cols : ℕ) (total_seats : ℕ) (dist_between : ℕ)
  (restriction_4 : ∀ r c : ℕ, r < rows → c < cols → c + 4 ≤ cols → ∑ i in finset.range 4, (seat_occ r (c + i)) = 0)
  : rows = 6 → cols = 6 → dist_between > 1 → total_seats = 36 → 
    ∃ max_people : ℕ, max_people = 28 ∧ ∀ arrangement, valid_arrangement arrangement → 
    count_occupied arrangement ≤ max_people := sorry

end max_people_no_5_consecutive_max_people_no_4_consecutive_l607_607599


namespace coeff_x3_in_expansion_l607_607979

theorem coeff_x3_in_expansion : 
  let p := (1 - x) * (1 + x) ^ 8 in 
  (p.coeff 3) = 28 :=
begin
  sorry
end

end coeff_x3_in_expansion_l607_607979


namespace non_convex_polygon_odd_sides_non_convex_polygon_even_sides_l607_607700

theorem non_convex_polygon_odd_sides (n : Nat) : 
  ¬ ∃ (L : Set (ℝ × ℝ)), (∀ (p : ℕ), p ≤ 2 * n + 1 → intersects_all_sides (non_convex_polygon (2 * n + 1)) L) :=
sorry

theorem non_convex_polygon_even_sides (n : Nat) : 
  ∃ (L : Set (ℝ × ℝ)), (∀ (p : ℕ), p ≤ 2 * n → intersects_all_sides (non_convex_polygon (2 * n)) L) :=
sorry

end non_convex_polygon_odd_sides_non_convex_polygon_even_sides_l607_607700


namespace number_of_students_playing_soccer_l607_607882

variable (total_students boys playing_soccer_girls not_playing_soccer_girls : ℕ)
variable (percentage_boys_playing_soccer : ℕ)

-- Conditions
axiom h1 : total_students = 470
axiom h2 : boys = 300
axiom h3 : not_playing_soccer_girls = 135
axiom h4 : percentage_boys_playing_soccer = 86
axiom h5 : playing_soccer_girls = 470 - 300 - not_playing_soccer_girls

-- Question: Prove that the number of students playing soccer is 250
theorem number_of_students_playing_soccer : 
  (playing_soccer_girls * 100) / (100 - percentage_boys_playing_soccer) = 250 :=
sorry

end number_of_students_playing_soccer_l607_607882


namespace prime_gt_five_condition_l607_607147

theorem prime_gt_five_condition (p : ℕ) [Fact (Nat.Prime p)] (h : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - a^2) ∣ (p - b)^2 := 
sorry

end prime_gt_five_condition_l607_607147


namespace nobody_but_angela_finished_9_problems_l607_607333

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end nobody_but_angela_finished_9_problems_l607_607333


namespace proof_problem1_proof_problem2_l607_607794

noncomputable def ellipse_condition1 (a b : ℝ) : Prop :=
  a > b ∧ b > 0

noncomputable def ellipse_condition2 (x y : ℝ) : Prop :=
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ x = 2 ∧ y = sqrt 2 ∧ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / 8) + (y^2 / 4) = 1

noncomputable def line_chord_condition (x1 y1 x2 y2 x y : ℝ) : Prop :=
  (x1^2 / 8 + y1^2 / 4 = 1) ∧ (x2^2 / 8 + y2^2 / 4 = 1) ∧
  (y1 - y2) / (x1 - x2) = -1 ∧
  x = 2 ∧ y = 1

def problem_statement1 : Prop :=
  ∃ (a b : ℝ), ellipse_condition1 a b ∧ ellipse 2 (sqrt 2)

def problem_statement2 : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line_chord_condition x1 y1 x2 y2 2 1 ∧ (x1 + x2 ≠ 4 ∨ y1 + y2 ≠ 2 → (x + y) = 3)

theorem proof_problem1 : problem_statement1 :=
by
  sorry

theorem proof_problem2 : problem_statement2 :=
by
  sorry

end proof_problem1_proof_problem2_l607_607794


namespace pizza_dough_milk_needed_l607_607241

variable (milk_per_300 : ℕ) (flour_per_batch : ℕ) (total_flour : ℕ)

-- Definitions based on problem conditions
def milk_per_batch := milk_per_300
def batch_size := flour_per_batch
def used_flour := total_flour

-- The target proof statement
theorem pizza_dough_milk_needed (h1 : milk_per_batch = 60) (h2 : batch_size = 300) (h3 : used_flour = 1500) : 
  (used_flour / batch_size) * milk_per_batch = 300 :=
by
  rw [h1, h2, h3]
  sorry -- proof steps

end pizza_dough_milk_needed_l607_607241


namespace a_is_perfect_square_l607_607544

open Real

-- Definitions to set up the conditions
def integers := ℕ+
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem a_is_perfect_square (a b c d : integers) 
  (h1 : a < b) (h2 : b ≤ c) (h3 : c < d)
  (h4 : a * d = b * c)
  (h5 : sqrt ↑d - sqrt ↑a ≤ 1) : 
  is_perfect_square a :=
by sorry

end a_is_perfect_square_l607_607544


namespace certain_event_among_13_students_l607_607261

-- Definition: A month in a year
def Month : Type := Fin 12

-- Definition: A student assigned a birth month
def Student : Type := {birthMonth : Month}

-- Definition: A set of students
def Students (n : Nat) := Fin n → Student

-- Theorem: Among 13 students, at least two have the same birth month.
theorem certain_event_among_13_students (students : Students 13) :
  ∃ (a b : Fin 13), a ≠ b ∧ students a = students b :=
sorry

end certain_event_among_13_students_l607_607261


namespace max_cursed_roads_l607_607115

-- Define the structure of the problem
structure Empire where
  cities : ℕ
  roads : ℕ
  initialConnected : Bool
  
def curse_roads (G : Empire) (k : ℕ) : Empire :=
  { G with roads := G.roads - k }

-- Initial conditions
def initial_empire : Empire :=
  { cities := 1000, roads := 2017, initialConnected := True }

-- Function to determine if the graph is divided into N components after cursing k roads
def divides_into (G : Empire) (k N : ℕ) : Prop :=
  (k = N - 1) ∧ (N = 7)

-- Maximum value of roads that can be cursed while satisfying the conditions
theorem max_cursed_roads : 
  ∃ N, 
    divides_into (curse_roads initial_empire N) 6 7 ∧ 
    N = 2011 :=
begin
  sorry
end

end max_cursed_roads_l607_607115


namespace range_of_f_l607_607366

noncomputable def f (x : ℝ) : ℝ := 3^(-x^2)

theorem range_of_f : set.Ioo 0 1 ∪ {1} = (set.range f) :=
by sorry

end range_of_f_l607_607366


namespace original_cost_of_statue_l607_607129

theorem original_cost_of_statue
  (SP : ℝ) (profit_rate : ℝ) (C : ℝ)
  (h1 : SP = 670)
  (h2 : SP = C * (1 + profit_rate)) :
  C = 496.3 :=
by {
  have h3 : SP = C * 1.35, {
    rw h2,
    have h4 : profit_rate = 0.35 := by sorry,
    rw h4,
    ring,
  },
  rw h1 at h3,
  linarith,
  sorry,
}

end original_cost_of_statue_l607_607129


namespace problem_r_minus_s_l607_607160

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l607_607160


namespace solve_for_x_l607_607961

theorem solve_for_x : ∃ x : ℝ, (9^x + 3^x - 6 = 0) ↔ (x = Real.log 2 / Real.log 3) :=
by
  use Real.log 2 / Real.log 3
  sorry

end solve_for_x_l607_607961


namespace patrícia_157th_number_l607_607937

theorem patrícia_157th_number :
  (∃ seq : ℕ → ℕ, (∀ n, seq n > 0 ∧ ∀ d ∈ seq n.digits, d ∈ {1, 3, 5, 7, 9}) ∧ (seq 156 = 1113)) :=
sorry

end patrícia_157th_number_l607_607937


namespace complement_set_l607_607834

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem complement_set :
  U = univ ∧ M = {x | x^2 - 2 * x ≤ 0} → (U \ M) = {x | x < 0 ∨ x > 2} :=
by
  intros
  sorry

end complement_set_l607_607834


namespace units_digit_M_M12_l607_607526

def modifiedLucas (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | 1     => 2
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem units_digit_M_M12 (n : ℕ) (H : modifiedLucas 12 = 555) : 
  (modifiedLucas (modifiedLucas 12) % 10) = 1 := by
  sorry

end units_digit_M_M12_l607_607526


namespace population_net_increase_l607_607267

theorem population_net_increase
  (birth_rate_s5 : ℕ := 6) (birth_rate_t5 : ℕ := 2)
  (death_rate_s5 : ℕ := 2) (death_rate_t5 : ℕ := 2) :
  let net_increase_per_second := (birth_rate_s5 / birth_rate_t5) - (death_rate_s5 / death_rate_t5) in
  let seconds_in_a_day := 24 * 60 * 60 in
  let net_increase_in_one_day := net_increase_per_second * seconds_in_a_day in
  net_increase_in_one_day = 172800 :=
by
  sorry

end population_net_increase_l607_607267


namespace sum_of_possible_values_of_x_l607_607963

theorem sum_of_possible_values_of_x :
  ∑ x in {x | (2 : ℝ) ^ (x^2 + 5 * x + 6) = (8 : ℝ) ^ (x + 3)}, x = -2 :=
by
  sorry

end sum_of_possible_values_of_x_l607_607963


namespace find_function_satisfying_condition_l607_607751

theorem find_function_satisfying_condition :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → 
                          (∀ x : ℝ, f x = 2 * x + c) :=
sorry

end find_function_satisfying_condition_l607_607751


namespace find_ratio_EG_ES_l607_607873

variables (EF GH EH EG ES QR : ℝ) -- lengths of the segments
variables (x y : ℝ) -- unknowns for parts of the segments
variables (Q R S : Point) -- points

-- Define conditions based on the problem
def parallelogram_EFGH (EF GH EH EG : ℝ) : Prop :=
  ∀ (x y : ℝ), EF = 8 * x ∧ EH = 9 * y

def point_on_segment_Q (Q : Point) (EF EQ : ℝ) : Prop :=
  ∃ x : ℝ, EQ = (1 / 8) * EF

def point_on_segment_R (R : Point) (EH ER : ℝ) : Prop :=
  ∃ y : ℝ, ER = (1 / 9) * EH

def intersection_at_S (EG QR ES : ℝ) : Prop :=
  ∃ x y : ℝ, ES = (1 / 8) * EG + (1 / 9) * EG

theorem find_ratio_EG_ES :
  parallelogram_EFGH EF GH EH EG →
  point_on_segment_Q Q EF (1/8 * EF) →
  point_on_segment_R R EH (1/9 * EH) →
  intersection_at_S EG QR ES →
  EG / ES = 72 / 17 :=
by
  intros h_parallelogram h_pointQ h_pointR h_intersection
  sorry

end find_ratio_EG_ES_l607_607873


namespace number_of_students_from_second_department_is_17_l607_607121

noncomputable def students_selected_from_second_department 
  (total_students : ℕ)
  (num_departments : ℕ)
  (students_per_department : List (ℕ × ℕ))
  (sample_size : ℕ)
  (starting_number : ℕ) : ℕ :=
-- This function will compute the number of students selected from the second department.
sorry

theorem number_of_students_from_second_department_is_17 : 
  students_selected_from_second_department 600 3 
    [(1, 300), (301, 495), (496, 600)] 50 3 = 17 :=
-- Proof is left as an exercise.
sorry

end number_of_students_from_second_department_is_17_l607_607121


namespace initial_food_weight_l607_607896

-- Definitions based on conditions
def initial_water : ℝ := 20
def initial_gear : ℝ := 20
def water_consumption_rate : ℝ := 2
def food_consumption_rate (water_rate : ℝ) : ℝ := (1 / 3) * water_rate
def time_hiking : ℝ := 6
def remaining_weight : ℝ := 34

-- Computing the consumed water and food
def consumed_water (rate : ℝ) (time : ℝ) : ℝ := rate * time
def consumed_food (rate : ℝ) (time : ℝ) : ℝ := rate * time

-- Proof statement: given initial conditions, prove the initial food weight
theorem initial_food_weight (F : ℝ) :
  initial_water + F + initial_gear - consumed_water water_consumption_rate time_hiking - consumed_food (food_consumption_rate water_consumption_rate) time_hiking = remaining_weight ->
  F = 10 :=
by
  sorry

end initial_food_weight_l607_607896


namespace tangent_line_equation_l607_607983

noncomputable def curve (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x

def point_x : ℝ := 1

def point_y : ℝ := -3 / 2

theorem tangent_line_equation :
  ∃ (a b c : ℝ), a * 2 + b * 2 + c = 0 ∧ curve 1 = point_y ∧ deriv (λ x, (1 / 2) * x^2 - 2 * x) 1 = -1 :=
sorry

end tangent_line_equation_l607_607983


namespace axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l607_607056

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt 3 * Real.sin (Real.pi - x) + 5 * Real.sin (Real.pi / 2 + x) + 5

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi / 3 + k * Real.pi) :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi - Real.pi / 6) = 5 :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x) - 8

theorem g_max_value :
  ∀ x : ℝ, g x ≤ 2 :=
sorry

theorem g_increasing_intervals :
  ∀ k : ℤ, -Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≤ g (x + 1) :=
sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≥ g (x + 1) :=
sorry

end axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l607_607056


namespace sum_of_imaginary_parts_l607_607589

theorem sum_of_imaginary_parts (x y u v w z : ℝ) (h1 : y = 5) 
  (h2 : w = -x - u) (h3 : (x + y * I) + (u + v * I) + (w + z * I) = 4 * I) :
  v + z = -1 :=
by
  sorry

end sum_of_imaginary_parts_l607_607589


namespace factorize_expression_l607_607739

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l607_607739


namespace Connie_correct_result_l607_607725

theorem Connie_correct_result :
  ∀ x: ℝ, (200 - x = 100) → (200 + x = 300) :=
by
  intros x h
  have h1 : x = 100 := by linarith [h]
  rw [h1]
  linarith

end Connie_correct_result_l607_607725


namespace prob_x_lt_1_l607_607041

noncomputable def normalDist : Distribution := NormalDistribution.mk 2 1

axiom P_1_le_x_le_3: ℝ := 0.6826

theorem prob_x_lt_1 : ∀ (x : ℝ), ProbabilityDensityFunction normalDist x < 1 = 0.1587 :=
by
  intro x
  sorry

end prob_x_lt_1_l607_607041


namespace binomial_7_4_eq_35_l607_607715

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607715


namespace sum_fractions_le_one_l607_607030

theorem sum_fractions_le_one (x : Fin 5 → ℝ) (h0 : ∀ i, 0 ≤ x i)
  (h1 : ∑ i, 1 / (1 + x i) = 1) :
  ∑ i, x i / (4 + (x i)^2) ≤ 1 := sorry

end sum_fractions_le_one_l607_607030


namespace express_train_numbers_l607_607569

theorem express_train_numbers:
  ∃ (n k : ℕ), (∑ i in Finset.range (n + 1), i) = 111 ∧ k ≤ n ∧ k > 0 ∧ n = 15 ∧ k = 6 :=
by
  sorry

end express_train_numbers_l607_607569


namespace count_solutions_abs_inequality_l607_607840

theorem count_solutions_abs_inequality :
  let S := {x : ℤ | |4 * x + 1| ≤ 9} in
  S.card = 5 :=
by
  sorry

end count_solutions_abs_inequality_l607_607840


namespace remainder_x_plus_3uy_plus_u_div_y_l607_607407

theorem remainder_x_plus_3uy_plus_u_div_y (x y u v : ℕ) (hx : x = u * y + v) (hu : 0 ≤ v) (hv : v < y) (huv : u + v < y) : 
  (x + 3 * u * y + u) % y = u + v :=
by
  sorry

end remainder_x_plus_3uy_plus_u_div_y_l607_607407


namespace light_path_total_distance_l607_607901

-- Define the conditions
def edge_length := 10
def distance_from_BG := 6
def distance_from_BC := 4

-- Define the total distance statement
theorem light_path_total_distance (ABCD BCFG : Type) (A B C D F G : ABCD) (P : BCFG)
  (h_AB : edge_length = 10) (h_P_BG : distance_from_BG = 6) (h_P_BC : distance_from_BC = 4) :
  ∃ m n (Hn : ¬ exists p : ℕ, nat.prime p ∧ p^2 ∣ n), m * real.sqrt n = 10 * real.sqrt 152 :=
by
  use 10
  use 152
  split
  . sorry -- Proof that n = 152 is not divisible by the square of any prime
  . sorry -- Proof that 10 * sqrt(152) is indeed the total distance

end light_path_total_distance_l607_607901


namespace exists_convex_polygon_l607_607404

theorem exists_convex_polygon (M : set (ℝ × ℝ)) (n : ℕ) (h_card : M.to_finset.card = n) 
  (h_n : n ≥ 3) (h_non_collinear : ¬ ∀ p1 p2 p3 ∈ M, collinear ℝ {p1, p2, p3}) : 
  ∃ P : set (ℝ × ℝ), convex ℝ P ∧ (∀ x ∈ M, x ∈ P) ∧ (∀ v ∈ P, v ∈ M) :=
sorry

end exists_convex_polygon_l607_607404


namespace johnny_ran_4_times_l607_607533

-- Block length is 200 meters
def block_length : ℕ := 200

-- Distance run by Johnny is Johnny's running times times the block length
def johnny_distance (J : ℕ) : ℕ := J * block_length

-- Distance run by Mickey is half of Johnny's running times times the block length
def mickey_distance (J : ℕ) : ℕ := (J / 2) * block_length

-- Average distance run by Johnny and Mickey is 600 meters
def average_distance_condition (J : ℕ) : Prop :=
  ((johnny_distance J + mickey_distance J) / 2) = 600

-- We are to prove that Johnny ran 4 times based on the condition
theorem johnny_ran_4_times (J : ℕ) (h : average_distance_condition J) : J = 4 :=
sorry

end johnny_ran_4_times_l607_607533


namespace ratio_of_boxes_sold_l607_607132

-- Definitions for conditions
variables (T W Tu : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  W = 2 * T ∧
  Tu = 2 * W ∧
  T = 1200

-- The statement to prove the ratio Tu / W = 2
theorem ratio_of_boxes_sold (T W Tu : ℕ) (h : conditions T W Tu) :
  Tu / W = 2 :=
by
  sorry

end ratio_of_boxes_sold_l607_607132


namespace positive_value_of_t_l607_607005

noncomputable def t_value : ℝ :=
  let t := 2 * Real.sqrt 5 in
  t

theorem positive_value_of_t : ∃ t : ℝ, 0 < t ∧ |(8 : ℂ) + (2 * t * Complex.I)| = 12 ∧ t = 2 * Real.sqrt 5 :=
by
  use 2 * Real.sqrt 5
  split
  · exact Real.sqrt_pos.mpr (by norm_num)
  split
  · sorry
  · rfl

end positive_value_of_t_l607_607005


namespace find_side2_l607_607221

-- Define the given conditions
def perimeter : ℕ := 160
def side1 : ℕ := 40
def side3 : ℕ := 70

-- Define the second side as a variable
def side2 : ℕ := perimeter - side1 - side3

-- State the theorem to be proven
theorem find_side2 : side2 = 50 := by
  -- We skip the proof here with sorry
  sorry

end find_side2_l607_607221


namespace relationship_of_y_l607_607939

theorem relationship_of_y (b y1 y2 : ℝ) (h1 : y1 = -3 * (-3) + b) (h2 : y2 = -3 * (4) + b) : 
  y1 > y2 :=
begin
  rw [h1, h2],
  linarith,
end

end relationship_of_y_l607_607939


namespace triangle_PQ_l607_607093

theorem triangle_PQ (A B C D E H Q P : Point) (AB BC CA : Length) 
  (h1 : distance A B = 9) (h2 : distance B C = 10) (h3 : distance C A = 11)
  (AH : Line) (h4 : isAltitude A H B C AH)
  (BD CE : Line) (hBD1 : onLine D AC) (hCE1 : onLine E AB)
  (hBD2 : isAngleBisector B D AC) (hCE2 : isAngleBisector C E AB)
  (intersect_BD_AH : intersection BD AH = Q) (intersect_CE_AH : intersection CE AH = P) :
  distance P Q = 5 * sqrt 2 / 6 := sorry

end triangle_PQ_l607_607093


namespace segments_inequality_l607_607885

variables {n : ℕ} {a : Fin n → ℝ} {h : Fin n → ℝ}

theorem segments_inequality (h_pos : ∀ i, h i > 0)
  (segment_cond : ∀ θ : ℝ, ∃ i, ∃ C, C ∈ [0, a i] ∧ ∃ (ray_O_to_C : ℝ → ℝ), ray_O_to_C θ = 0 ∧ (∀ x ≥ 0, ray_O_to_C x = segment_cond x)) :
  (Finset.univ.sum (λ i, a i / h i)) ≥ 2 * Real.pi :=
by
  sorry

end segments_inequality_l607_607885


namespace normal_probability_l607_607044

noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (Real.pi * 2).sqrt⁻¹ / σ * Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2))

noncomputable def P (X : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, X t

variable (X : ℝ → ℝ)
variable (μ σ : ℝ := 0) (P_X_le_1 : ℝ := 0.8413)

axiom pdf_normal : ∀ x, X x = normal_pdf μ 1 x

theorem normal_probability : P (λ x, normal_pdf 0 1 x) (-1) 0 = 0.3413 :=
by
  sorry

end normal_probability_l607_607044


namespace circle_arrangement_rel_prime_l607_607452

/--
Prove that the number of ways to arrange the numbers {1, 2, 3, 4, 5, 6, 7, 8} in a circle such that every two adjacent elements are relatively prime, considering rotations and reflectional symmetries, is 36.
-/
theorem circle_arrangement_rel_prime : 
  let nums := {1, 2, 3, 4, 5, 6, 7, 8} in 
  let rel_prime (a b : ℕ) := Nat.gcd a b = 1 in 
  ∃ l : List ℕ, 
    l.Perm (List.formPermutation nums) ∧ 
    (∀ i, rel_prime (l.nth i) (l.nth ((i + 1) % l.length))) ∧ 
    l.CombinationsDistinctByRotationsReflections (List.formPermutation nums) ∧ 
    CountValidArrangements(nums) = 36 := sorry

end circle_arrangement_rel_prime_l607_607452


namespace new_car_distance_in_same_time_l607_607668

-- Define the given conditions and the distances
variable (older_car_distance : ℝ := 150)
variable (new_car_speed_factor : ℝ := 1.30)  -- Since the new car is 30% faster, its speed factor is 1.30
variable (time : ℝ)

-- Define the older car's distance as a function of time and speed
def older_car_distance_covered (t : ℝ) (distance : ℝ) : ℝ := distance

-- Define the new car's distance as a function of time and speed factor
def new_car_distance_covered (t : ℝ) (distance : ℝ) (speed_factor : ℝ) : ℝ := speed_factor * distance

theorem new_car_distance_in_same_time
  (older_car_distance : ℝ)
  (new_car_speed_factor : ℝ)
  (time : ℝ)
  (h1 : older_car_distance = 150)
  (h2 : new_car_speed_factor = 1.30) :
  new_car_distance_covered time older_car_distance new_car_speed_factor = 195 := by
  sorry

end new_car_distance_in_same_time_l607_607668


namespace toothpick_grid_boundary_and_total_l607_607600

theorem toothpick_grid_boundary_and_total (height : ℕ) (width : ℕ)
  (h_height : height = 15) 
  (h_width : width = 12) :
  let total_horizontal_toothpicks := (height + 1) * width,
      total_vertical_toothpicks := (width + 1) * height,
      total_toothpicks := total_horizontal_toothpicks + total_vertical_toothpicks,
      boundary_horizontal := 2 * width,
      boundary_vertical := 2 * height,
      boundary_toothpicks := boundary_horizontal + boundary_vertical in
  (boundary_toothpicks = 54) ∧ (total_toothpicks = 387) := by
  sorry

end toothpick_grid_boundary_and_total_l607_607600


namespace simplify_expression_l607_607918

theorem simplify_expression
  (a b c : ℝ) 
  (hnz_a : a ≠ 0) 
  (hnz_b : b ≠ 0) 
  (hnz_c : c ≠ 0) 
  (h_sum : a + b + c = 0) :
  (1 / (b^3 + c^3 - a^3)) + (1 / (a^3 + c^3 - b^3)) + (1 / (a^3 + b^3 - c^3)) = 1 / (a * b * c) :=
by
  sorry

end simplify_expression_l607_607918


namespace throws_to_return_to_start_l607_607749

def next_position (pos: ℕ) (skip: ℕ) (n: ℕ) : ℕ :=
  ((pos + skip) % n) + 1

theorem throws_to_return_to_start :
  ∃ n, n > 0 ∧ n ≤ 15 ∧ 
  (∀ k, end_position 1 (6 * k) 15 = 1 ↔ 6 * k % 15 + 1 = 1) :=
sorry

end throws_to_return_to_start_l607_607749


namespace simplify_polynomial_l607_607557

def poly1 : ℕ → ℝ 
| 6 := 2
| 5 := 3
| 4 := 1
| 3 := 3
| 2 := 0
| 1 := 2
| 0 := 15
| _ := 0

def poly2 : ℕ → ℝ 
| 6 := 1
| 5 := 4
| 4 := 0
| 3 := 2
| 2 := -1
| 1 := 0
| 0 := 5
| _ := 0

def result_poly : ℕ → ℝ 
| 6 := 1
| 5 := -1
| 4 := 1
| 3 := 1
| 2 := 1
| 1 := 2
| 0 := 10
| _ := 0

theorem simplify_polynomial : 
  ∀ x : ℝ, 
  (poly1 6 * x^6 + poly1 5 * x^5 + poly1 4 * x^4 + poly1 3 * x^3 + poly1 2 * x^2 + poly1 1 * x + poly1 0)
  - (poly2 6 * x^6 + poly2 5 * x^5 + poly2 4 * x^4 + poly2 3 * x^3 + poly2 2 * x^2 + poly2 1 * x + poly2 0)
  = result_poly 6 * x^6 + result_poly 5 * x^5 + result_poly 4 * x^4 + result_poly 3 * x^3 + result_poly 2 * x^2 + result_poly 1 * x + result_poly 0 :=
by 
  sorry

end simplify_polynomial_l607_607557


namespace polynomial_identity_evaluation_l607_607822

theorem polynomial_identity_evaluation :
  (let f := (x : ℤ) ↦ x^2 - x + 1
    in (f 1)^6 - (f (-1))^6 = 729) :=
sorry

end polynomial_identity_evaluation_l607_607822


namespace flatville_additional_plates_l607_607097

theorem flatville_additional_plates : 
  let initial_plates :=
    (5 * 3 * 5 * 5 : ℕ),
  new_plates :=
    (5 * 4 * 7 * 5 : ℕ)
  in new_plates - initial_plates = 325 := 
by
  let initial_plates := 5 * 3 * 5 * 5
  let new_plates := 5 * 4 * 7 * 5
  show new_plates - initial_plates = 325
  from sorry

end flatville_additional_plates_l607_607097


namespace variance_linear_transformation_of_binomial_l607_607403

open ProbabilityTheory

variables (p q : ℝ)
variables (X : ℕ → ℝ)

theorem variance_linear_transformation_of_binomial :
  (∃ p q : ℝ, (distribution X = binomial 5 p) ∧ (𝔼[X] = 2)) → variance(2 * X + q) = 4.8 :=
sorry

end variance_linear_transformation_of_binomial_l607_607403


namespace find_opposite_endpoint_l607_607350

/-- A utility function to model coordinate pairs as tuples -/
def coord_pair := (ℝ × ℝ)

-- Define the center and one endpoint
def center : coord_pair := (4, 6)
def endpoint1 : coord_pair := (2, 1)

-- Define the expected endpoint
def expected_endpoint2 : coord_pair := (6, 11)

/-- Definition of the opposite endpoint given the center and one endpoint -/
def opposite_endpoint (c : coord_pair) (p : coord_pair) : coord_pair :=
  let dx := c.1 - p.1
  let dy := c.2 - p.2
  (c.1 + dx, c.2 + dy)

/-- The proof statement for the problem -/
theorem find_opposite_endpoint :
  opposite_endpoint center endpoint1 = expected_endpoint2 :=
sorry

end find_opposite_endpoint_l607_607350


namespace inequality_proof_l607_607916

variable {ι : Type*}

theorem inequality_proof (n : ℕ) (t : Fin n → ℝ)
  (h_cond1 : ∀ i, 0 < t i)
  (h_cond2 : ∀ i j, i ≤ j → t i ≤ t j)
  (h_cond3 : ∀ i, t i < 1) :
  (1 - t (n-1)) ^ 2 *
  ∑ i in Finset.range n, t i ^ i / (1 - t i ^ (i+1)) ^ 2 < 1 :=
by
  sorry

end inequality_proof_l607_607916


namespace smallest_n_is_2022_l607_607733

theorem smallest_n_is_2022 :
  ∀ (n : ℕ), (∀ (x : Fin n → ℝ), (∀ i, x i > -1 ∧ x i < 1) → (∑ i, x i = 0) → (∑ i, (x i)^2 = 2020) → n ≥ 2022) :=
by
  sorry

end smallest_n_is_2022_l607_607733


namespace number_of_possible_m_values_l607_607216

theorem number_of_possible_m_values : 
  ∃ n : ℕ, n = 897 ∧ ∀ m : ℕ, (∃ m' : ℤ, m = Int.ofNat m') → 
    (log 45 + log m > log 20) ∧ (log 20 + log 45 > log m) ∧ (log 20 + log m > log 45) ↔ 
    (3 ≤ m ∧ m ≤ 899) :=
by
  sorry

end number_of_possible_m_values_l607_607216


namespace circle_properties_l607_607886

noncomputable def polarToCartesian (rho theta : ℝ) : (ℝ × ℝ) :=
  let x := rho * cos theta
  let y := rho * sin theta
  (x, y)

theorem circle_properties :
  ∀ (θ : ℝ), 
    let ρ := 6 * cos θ + 8 * sin θ in
    let (x, y) := polarToCartesian ρ θ in
    (x - 3) ^ 2 + (y - 4) ^ 2 = 25 ∧
    (x + y ≤ 7 + 5 * sqrt 2) ∧ 
    (x + y = 7 + 5 * sqrt 2 → x = 3 + 5 * sqrt 2 / 2 ∧ y = 4 + 5 * sqrt 2 / 2) := 
by
  sorry

end circle_properties_l607_607886


namespace part1_part2_l607_607432

-- Conditions and function definitions
def f (x : ℝ) (m : ℝ) := -x ^ 3 + m * x ^ 2 - m

-- Part 1: Prove the intervals of decreasing for f(x) when m = 1
theorem part1 (x : ℝ) : 
  f x 1 = -x ^ 3 + x ^ 2 - 1 → 
  (f' 1 x < 0 → x < 0 ∨ x > 2 / 3) :=
sorry

-- Part 2: Prove the maximum value of g(x) on [0, m]
theorem part2 (m : ℝ) (hm : 0 < m) : 
  let g (x : ℝ) := |f x m|
  in (∀ x, 0 ≤ x ∧ x ≤ m → 
    (g x ≤ if m ≥ (3 * Real.sqrt 6) / 2 then (4 / 27) * m ^ 3 - m else m)) :=
sorry

end part1_part2_l607_607432


namespace optimal_post_office_location_l607_607588

-- Define the coordinates for houses
def house_coordinates (n : ℕ) (i : ℕ) : ℝ :=
  if 1 ≤ i ∧ i ≤ n then x i else 0 -- dummy implementation for placeholder

-- Define the total distance function
def total_distance (t : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, |house_coordinates n (i + 1) - t|

-- Define the median function for a sorted list of house coordinates
def median (n : ℕ) (x : ℕ → ℝ) : Set ℝ :=
  if n % 2 = 1 then {x (n / 2 + 1)}
  else (Icc (x (n / 2)) (x (n / 2 + 1)))

theorem optimal_post_office_location (n : ℕ) (x : ℕ → ℝ) (h : ∀ i, x i < x (i + 1)) :
  ∃ t ∈ median n x, ∀ t', total_distance t n ≤ total_distance t' n :=
begin
  sorry
end

end optimal_post_office_location_l607_607588


namespace problem_statement_l607_607166

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l607_607166


namespace find_number_of_students_l607_607579

open Nat

theorem find_number_of_students :
  ∃ n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 :=
by
  use 57
  sorry

end find_number_of_students_l607_607579


namespace douglas_vote_percentage_is_66_l607_607107

noncomputable def percentDouglasVotes (v : ℝ) : ℝ :=
  let votesX := 0.74 * (2 * v)
  let votesY := 0.5000000000000002 * v
  let totalVotes := 3 * v
  let totalDouglasVotes := votesX + votesY
  (totalDouglasVotes / totalVotes) * 100

theorem douglas_vote_percentage_is_66 :
  ∀ v : ℝ, percentDouglasVotes v = 66 := 
by
  intros v
  unfold percentDouglasVotes
  sorry

end douglas_vote_percentage_is_66_l607_607107


namespace bob_fencing_needed_l607_607697

/-
  Given:
  - A rectangular plot with dimensions 225 feet by 125 feet.
  - An irregularly shaped area with side lengths: 75 feet, 150 feet, 45 feet, and 120 feet.
  - A circular tree base with a diameter of 6 feet.
  - An elliptical pond with a major axis of 20 feet and a minor axis of 12 feet.
  - Gates with widths: 3 feet, 10 feet, 4 feet, 7 feet, 2.5 feet, and 5 feet.

  Prove:
  The total amount of fencing required is 1191.4 feet.
-/

noncomputable def perimeter_rectangular : ℝ := 2 * (225 + 125)
noncomputable def perimeter_irregular : ℝ := 75 + 150 + 45 + 120
noncomputable def circumference_tree : ℝ := Real.pi * 6
noncomputable def perimeter_pond : ℝ := 
  let a := 10
  let b := 6
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))
noncomputable def total_gate_width : ℝ := 3 + 10 + 4 + 7 + 2.5 + 5

noncomputable def total_fencing : ℝ :=
  perimeter_rectangular + perimeter_irregular + circumference_tree + perimeter_pond + total_gate_width

theorem bob_fencing_needed : total_fencing ≈ 1191.4 := 
by
  unfold total_fencing
  unfold perimeter_rectangular
  unfold perimeter_irregular
  unfold circumference_tree
  unfold perimeter_pond
  unfold total_gate_width
  sorry

end bob_fencing_needed_l607_607697


namespace smallest_multiple_divisors_l607_607146

theorem smallest_multiple_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∀ m : ℕ, m % 75 = 0 ∧ nat.totient m = 75 → n ≤ m) : n / 75 = 432 := by
  sorry

end smallest_multiple_divisors_l607_607146


namespace sin_double_angle_value_l607_607777

open Real

theorem sin_double_angle_value (α : ℝ) (h : α ∈ Ioo 0 π) (h1 : 3 * cos (2 * α) - 4 * cos α + 1 = 0) : 
  sin (2 * α) = -((4 * sqrt 2) / 9) := 
by
  sorry

end sin_double_angle_value_l607_607777


namespace coefficient_of_y_in_first_equation_is_minus_1_l607_607429

variable (x y z : ℝ)

def equation1 : Prop := 6 * x - y + 3 * z = 22 / 5
def equation2 : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_xyz : Prop := x + y + z = 10

theorem coefficient_of_y_in_first_equation_is_minus_1 :
  equation1 x y z → equation2 x y z → equation3 x y z → sum_xyz x y z → (-1 : ℝ) = -1 :=
by
  sorry

end coefficient_of_y_in_first_equation_is_minus_1_l607_607429


namespace percentage_design_black_is_57_l607_607862

noncomputable def circleRadius (n : ℕ) : ℝ :=
  3 * (n + 1)

noncomputable def circleArea (n : ℕ) : ℝ :=
  Real.pi * (circleRadius n) ^ 2

noncomputable def totalArea : ℝ :=
  circleArea 6

noncomputable def blackAreas : ℝ :=
  circleArea 0 + (circleArea 2 - circleArea 1) +
  (circleArea 4 - circleArea 3) +
  (circleArea 6 - circleArea 5)

noncomputable def percentageBlack : ℝ :=
  (blackAreas / totalArea) * 100

theorem percentage_design_black_is_57 :
  percentageBlack = 57 := 
by
  sorry

end percentage_design_black_is_57_l607_607862


namespace collinear_A_I_A_circumcenter_A_l607_607932

-- Definitions for A, B, C, and their properties
variables {A B C : Type} [EuclideanGeometry A] [Circumcircle B C A A'] [Incenter B C A I]

-- Midpoint of arc not containing A
def midpoint_of_arc (A' : A) (B C : B) (circumcircle : A') : Prop :=
  midpoint B C A' ∧ ¬contains A circumcircle

-- Showing A, I, A' are collinear
theorem collinear_A_I_A' (h1 : midpoint_of_arc A' B C (circumcircle)) 
                        (h2 : incenter_of_triangle I (triangle A B C)) :
                        collinear A' I A :=
by 
  sorry

-- Showing A' is the circumcenter of BIC
theorem circumcenter_A' (h1 : midpoint_of_arc A' B C (circumcircle)) 
                        (h2 : incenter_of_triangle I (triangle A B C)) : 
                        circumcenter A' B I C :=
by 
  sorry

end collinear_A_I_A_circumcenter_A_l607_607932


namespace number_of_elements_in_union_l607_607805

open Set

-- Define natural numbers excluding zero
def N_star : Set ℕ := {n | n > 0}

-- Define the elements of sets A and B
variables (A B : Set ℕ)

-- Conditions for A and B
def conditions (A B : Set ℕ) : Prop :=
  (∀ x y ∈ A, x ≠ y → x * y ∈ B) ∧
  (∀ x y ∈ B, x < y → (y / x) ∈ A) ∧
  (A ⊆ N_star) ∧ (B ⊆ N_star) ∧
  (2 ≤ size A) ∧ (4 = size A)

-- The proof problem
theorem number_of_elements_in_union (A B : Set ℕ) 
  (h : conditions A B) : size (A ∪ B) = 7 :=
sorry

end number_of_elements_in_union_l607_607805


namespace length_of_side_AB_l607_607491

theorem length_of_side_AB
  (ABC : Type) [triangle ABC]
  (A B C : ABC)
  (M N G : ABC) -- Assume M, N, G are points with G as centroid
  (AM BN : line ABC)
  (h1 : perpendicular AM BN)
  (h2 : length AM = 15)
  (h3 : length BN = 20)
  (h4 : divides_ratio G AM 2 1)
  (h5 : divides_ratio G BN 2 1)
  : length (segment A B) = 50 / 3 := by
  sorry

end length_of_side_AB_l607_607491


namespace find_X_l607_607864

theorem find_X 
  (X Y : ℕ)
  (h1 : 6 + X = 13)
  (h2 : Y = 7) :
  X = 7 := by
  sorry

end find_X_l607_607864


namespace angela_problems_l607_607329

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end angela_problems_l607_607329


namespace even_function_a_eq_4_l607_607842

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_a_eq_4 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = f x a) : a = 4 := by
  sorry

end even_function_a_eq_4_l607_607842


namespace find_a_b_and_min_val_l607_607436

-- Given function f(x) = x^3 - x^2 + ax + b and tangent line condition at (0, f(0))
variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + b

-- The conditions
theorem find_a_b_and_min_val 
  (h₁ : f 0 = 1) 
  (h₂ : ∂ (λ x, x^3 - x^2 + a * x + b) / ∂ x | (0 : ℝ) = -1) : 
  a = -1 ∧ b = 1 ∧ ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), 
  (x = -2 → f x = -9) ∧ (x = 1 → f x ≥ f(ℂ.find_val -2 0 (λ x, -x^2 -x + 1))) := 
by 
  sorry

end find_a_b_and_min_val_l607_607436


namespace find_SD_l607_607122

open Real

variables {A B C D P Q T S : Point}
variable {PA AQ QP AP PT TQ TS TP QS SD : ℝ}
variable {angle_APD_right triangle_PQA_similar_triangle_TPQ triangle_PQT_similar_triangle_DQS trapezoid}

axiom eq_parallel : AB ∥ CD
axiom angle_APD : ∠ APD = π / 2
axiom perpendicular_TS_AB : TS ⟂ AB
axiom AP_PT : AP = PT
axiom intersect_PQ : PD ∩ TS = Q
axiom RA_through_Q : RA passes through Q
axiom in_triangle_PQA : PA = 24 ∧ AQ = 30 ∧ QP = 18
axiom relations : 
  ∀ (TQ TP QS : ℝ), 
    (AQ / QP) = 5 / 3 → (TQ = 5 / 3 * 18) → 
    (TS = TP) → (TS = 36) → (QS = TS - TQ) →
    (TP = 2 * 18) → ∀ (SD : ℝ), (QS / TQ) = 1 / 5 → 
    (SD = TP * 1 / 5)

theorem find_SD : ∀ (SD : ℝ), SD = 7.2 := by 
  sorry

end find_SD_l607_607122


namespace polar_to_cartesian_and_distance_l607_607120

-- Define the main problem
theorem polar_to_cartesian_and_distance :
  (∀ θ : ℝ, (ρ θ = 6 * real.sin θ) → ∀ (x y : ℝ), (x = 1 ∧ y = 1) →
  ∀ (t1 t2 : ℝ), (|t1 - t2| = 3 * real.sqrt 2) →
  (t1 * t2 = -4 ∧ t1 = -2 * t2 ∨ t1 = 2 * t2) →
  |2 * t2 + t2| = 3 * real.sqrt 2) :=
begin
  sorry
end

end polar_to_cartesian_and_distance_l607_607120


namespace sufficient_and_necessary_condition_l607_607272

theorem sufficient_and_necessary_condition (a : ℝ) : (0 < a ∧ a < 1) ↔ (a < real.sqrt a) :=
sorry

end sufficient_and_necessary_condition_l607_607272


namespace find_lambda_l607_607067

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -2)
def c (λ : ℝ) : ℝ × ℝ := (1, λ)
def d : ℝ × ℝ := (3, 0)  -- This is a + b

theorem find_lambda (λ : ℝ) (h : c λ = (1, λ) ∧ c λ ∥ d) : λ = 0 :=
by
  sorry

end find_lambda_l607_607067


namespace quadratic_equation_unique_l607_607613

/-- Prove that among the given options, the only quadratic equation in \( x \) is \( x^2 - 3x = 0 \). -/
theorem quadratic_equation_unique (A B C D : ℝ → ℝ) :
  A = (3 * x + 2) →
  B = (x^2 - 3 * x) →
  C = (x + 3 * x * y - 1) →
  D = (1 / x - 4) →
  ∃! (eq : ℝ → ℝ), eq = B := by
  sorry

end quadratic_equation_unique_l607_607613


namespace sequence_sum_l607_607768

theorem sequence_sum : 
  2 * ((-1 : ℤ) ^ -11 + (-1 : ℤ) ^ -10) 
  + 3 * ((-1 : ℤ) ^ -9 + (-1 : ℤ) ^ -8) 
  + 4 * ((-1 : ℤ) ^ -7 + (-1 : ℤ) ^ -6)
  + 5 * ((-1 : ℤ) ^ -5 + (-1 : ℤ) ^ -4)
  + 6 * ((-1 : ℤ) ^ -3 + (-1 : ℤ) ^ -2)
  + 7 * ((-1 : ℤ) ^ -1 + (-1 : ℤ) ^ 0)
  + 8 * ((-1 : ℤ) ^ 1 + (-1 : ℤ) ^ 2)
  + 9 * ((-1 : ℤ) ^ 3 + (-1 : ℤ) ^ 4)
  + 10 * ((-1 : ℤ) ^ 5 + (-1 : ℤ) ^ 6)
  + 11 * ((-1 : ℤ) ^ 7 + (-1 : ℤ) ^ 8)
  + 12 * ((-1 : ℤ) ^ 9 + (-1 : ℤ) ^ 10)
  + 13 * ((-1 : ℤ) ^ 11) = 43 := 
sorry

end sequence_sum_l607_607768


namespace find_a_l607_607470

-- Assume the conditions as definitions
def c : ℝ := Real.sqrt 3
def A : ℝ := Real.pi / 4  -- 45 degrees in radians
def C : ℝ := Real.pi / 3  -- 60 degrees in radians

-- Main statement to prove
theorem find_a :
  ∃ a : ℝ, a = Real.sqrt 2 :=
by
  sorry

end find_a_l607_607470


namespace planes_parallel_l607_607484

-- defining the points A, B, C, D, A1, B1, C1, and D1 as points in 3D space
variables (A B C D A1 B1 C1 D1 : Point)

-- given conditions
def is_cube : Prop := 
  ∃ (u v w : Vector), 
  (u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0) ∧ 
  (u ⊥ v ∧ u ⊥ w ∧ v ⊥ w) ∧ 
  (B = A + u ∧ D = A + v ∧ 
   A1 = A + w ∧ B1 = B + w ∧ 
   C = A + u + v ∧ C1 = C + w ∧ 
   D1 = D + w ∧ B1 = A1 + u ∧ C1 = A1 + u + v)

-- defining plans as sets of points satisfying linear equations
def plane (p q r : Point) : set Point := 
  {s | ∃ a b, s = a * p + b * q + r}

-- defining the parallel relationship
def parallel_planes (P Q : set Point) : Prop := 
  ∀ p1 p2 ∈ P, ∃ q1 q2 ∈ Q, 
  (p1 - p2) = (q1 - q2)

-- The theorem to prove
theorem planes_parallel (h₁ : is_cube A B C D A1 B1 C1 D1) : 
  parallel_planes (plane A B1 D1) (plane B C1 D) :=
sorry

end planes_parallel_l607_607484


namespace units_digit_of_3_pow_2011_l607_607698

theorem units_digit_of_3_pow_2011 : (3 ^ 2011) % 10 = 7 := 
by
  -- Pattern of the units digits of successive powers of 3: 3, 9, 7, 1
  have units_digits : ℕ → ℕ := λ n, [3, 9, 7, 1].nth! (n % 4)
  -- Units digit repeats every 4 terms
  have cycle_repeats : (3 ^ 2011) % 10 = units_digits 2010 := by sorry
  -- Remainder of 2011 divided by 4 is 3
  have remainder : 2011 % 4 = 3 := by sorry
  -- Thus, the units digit of 3^2011 is the same as 3^3
  show (3 ^ 2011) % 10 = 7
  sorry

end units_digit_of_3_pow_2011_l607_607698


namespace colored_graph_exists_isolation_edges_parity_l607_607372

-- Definitions and conditions for the problem
def is_connected_graph (G : Type) (n : ℕ) : Prop := 
  ∀ (v : G), ∃ (e₁ e₂ e₃ : G), 
    e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₂ ≠ e₃ ∧ 
    (edge_colored e₁ = red ∧ edge_colored e₂ = blue ∧ edge_colored e₃ = green)

def is_edge_colored (G : Type) (e : G) : Prop :=
  edge_colored e ∈ {red, blue, green}

-- Problem statement: Show that n must be even and the colored graph is possible for any even n > 2
theorem colored_graph_exists (n : ℕ) (H1 : is_connected_graph G n) (H2 : ∀ v ∈ G, degree v = 3) : 
  ∃ G : Type, is_connected_graph G n ∧ is_edge_colored G ∧ (n % 2 = 0 ∧ n > 2) :=
sorry

-- Problem statement: Show that for a subset X, R, B, G are all even or all odd
theorem isolation_edges_parity (n : ℕ) (k : ℕ) (G : Type) (X : set G) (H1 : is_connected_graph G n) 
  (H2 : 1 < k < n) (H3 : X ⊆ G) :
  ∃ (R B G : ℕ), (X.size = k) ∧ (R % 2 = B % 2 ∧ B % 2 = G % 2) :=
sorry

end colored_graph_exists_isolation_edges_parity_l607_607372


namespace sequence_sum_l607_607254

theorem sequence_sum :
  27^2 - 25^2 + 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 392 :=
by
  -- By pairing and factoring as differences of squares
  have h₀ : 27^2 - 25^2 = 2 * (27 + 25) := sorry,
  have h₁ : 23^2 - 21^2 = 2 * (23 + 21) := sorry,
  have h₂ : 19^2 - 17^2 = 2 * (19 + 17) := sorry,
  have h₃ : 15^2 - 13^2 = 2 * (15 + 13) := sorry,
  have h₄ : 11^2 - 9^2 = 2 * (11 + 9) := sorry,
  have h₅ : 7^2 - 5^2 = 2 * (7 + 5) := sorry,
  have h₆ : 3^2 - 1^2 = 2 * (3 + 1) := sorry,

  -- Combining all
  calc
  27^2 - 25^2 + 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2
      = (27^2 - 25^2) + (23^2 - 21^2) + (19^2 - 17^2) + (15^2 - 13^2) + (11^2 - 9^2) + (7^2 - 5^2) + (3^2 - 1^2) : by simp 
  ... = 2 * (27 + 25) + 2 * (23 + 21) + 2 * (19 + 17) + 2 * (15 + 13) + 2 * (11 + 9) + 2 * (7 + 5) + 2 * (3 + 1) : by rw [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  ... = 2 * (27 + 25 + 23 + 21 + 19 + 17 + 15 + 13 + 11 + 9 + 7 + 5 + 3 + 1) : by ring
  ... = 2 * 196 : by norm_num 
  ... = 392 : by norm_num 

end sequence_sum_l607_607254


namespace compute_a2004_l607_607084

def recurrence_sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 0
  else sorry -- We'll define recurrence operations in the proofs

theorem compute_a2004 : recurrence_sequence 2004 = -2^1002 := 
sorry -- Proof omitted

end compute_a2004_l607_607084


namespace range_of_varphi_l607_607050

theorem range_of_varphi (φ : ℝ) (h1 : abs φ < π) (h2 : 
  ∀ x y, π / 5 < x ∧ x < 5 * π / 8 ∧ π / 5 < y ∧ y < 5 * π / 8 ∧ x < y → 
  -2 * sin (2 * x + φ) < -2 * sin (2 * y + φ)) :
  π / 10 ≤ φ ∧ φ ≤ π / 4 :=
sorry

end range_of_varphi_l607_607050


namespace total_pupils_correct_l607_607474

def number_of_girls : ℕ := 868
def difference_girls_boys : ℕ := 281
def number_of_boys : ℕ := number_of_girls - difference_girls_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

theorem total_pupils_correct : total_pupils = 1455 := by
  sorry

end total_pupils_correct_l607_607474


namespace find_b_l607_607222

-- Define the problem conditions
def polynomial := 3 * x ^ 3 + b * x + 8
def factor := x ^ 2 + p * x + 2

theorem find_b (p : ℚ) (b : ℚ) (c : ℚ) : (factor * (3 * x + c) = polynomial) → b = 2 / 3 := 
sorry

end find_b_l607_607222


namespace range_of_a_l607_607224

theorem range_of_a (a : ℝ) : (a^3 > (-3)^3) ↔ (a ∈ set.Ioi (-3)) :=
by
  sorry

end range_of_a_l607_607224


namespace coordinates_of_P_l607_607042

def angle : ℝ := 4 * Real.pi / 3
def distance : ℝ := 4

def point_P : ℝ × ℝ := (distance * Real.cos angle, distance * Real.sin angle)

theorem coordinates_of_P :
  point_P = (-2, -2 * Real.sqrt 3) :=
by
  sorry

end coordinates_of_P_l607_607042


namespace pascals_triangle_20th_in_25_row_l607_607247

theorem pascals_triangle_20th_in_25_row : nat.choose 24 19 = 4252 :=
by
  sorry

end pascals_triangle_20th_in_25_row_l607_607247


namespace fractional_equation_solution_l607_607234

noncomputable def problem_statement (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 = 2 / (x^2 - 1))

theorem fractional_equation_solution :
  ∀ x : ℝ, problem_statement x → x = -2 :=
by
  intro x hx
  sorry

end fractional_equation_solution_l607_607234


namespace proven_smallest_m_is_correct_l607_607001

-- Define divisors and related sets
def divisors (n : ℕ) : set ℕ := {d | d ∣ n}

def F_i (n : ℕ) (i : ℕ) : set ℕ := {a ∈ divisors n | a % 4 = i}

def f (n : ℕ) (i : ℕ) : ℕ := (F_i n i).toFinset.card

-- Define the smallest positive integer m such that 2f_1(m) - f_2(m) = 2017
noncomputable def smallest_m : ℕ :=
  Nat.find (λ m, 2 * f m 1 - f m 2 = 2017)

theorem proven_smallest_m_is_correct :
  smallest_m = 2 * 5^2016 :=
sorry

end proven_smallest_m_is_correct_l607_607001


namespace knight_reachability_in_2n_moves_l607_607931

-- Define the chessboard as an infinite grid of integers and coordinate positions.
variables {n : ℕ} (x y : ℤ)

def is_black (x y : ℤ) : Prop := (x + y) % 2 = 0

def reachable_by_knight (start_x start_y : ℤ) (kx ky : ℤ) (moves : ℕ) : Prop :=
  ∃ L, L.length = moves ∧ L.head = (start_x, start_y) ∧ L.last = (kx, ky) ∧ 
  ∀ i < moves, abs (L[i+1].fst - L[i].fst) = 2 ∧ abs (L[i+1].snd - L[i].snd) = 1 ∨
                abs (L[i+1].fst - L[i].fst) = 1 ∧ abs (L[i+1].snd - L[i].snd) = 2

def within_bounds (x y n : ℤ) : Prop :=
  abs x ≤ 8 * n + 1 ∧ abs y ≤ 8 * n + 1 ∧
  (abs x > 2 * n ∨ abs y > 2 * n)

theorem knight_reachability_in_2n_moves (n : ℕ) :
  ∀ (start_x start_y : ℤ), is_black start_x start_y →
    ∀ (kx ky : ℤ), reachable_by_knight start_x start_y kx ky (2 * n) →
      within_bounds kx ky n → is_black kx ky :=
by {
  sorry
}

end knight_reachability_in_2n_moves_l607_607931


namespace find_sum_of_terms_l607_607803

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def given_conditions (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ (a 4 + a 7 = 2) ∧ (a 5 * a 6 = -8)

theorem find_sum_of_terms (a : ℕ → ℝ) (h : given_conditions a) : a 1 + a 10 = -7 :=
sorry

end find_sum_of_terms_l607_607803


namespace probability_sum_eq_l607_607194

noncomputable def probability_space (Ω : Type*) := (Ω → Prop) → ℝ

variable {Ω : Type*} (P : probability_space Ω)
variable {A : ℕ → (Ω → Prop)}

-- Axiom: probability of the union of two mutually exclusive events is the sum of their probabilities
axiom ProbabilityUnion (A B : Ω → Prop) (h : ∀ ω, ¬(A ω ∧ B ω)) : P (λ ω, A ω ∨ B ω) = P A + P B

-- Definition of pairwise mutually exclusive events
def pairwise_mutually_exclusive (A : ℕ → (Ω → Prop)) :=
  ∀ i j, i ≠ j → ∀ ω, ¬(A i ω ∧ A j ω)

-- Theorem: Prove the given equality for pairwise mutually exclusive events
theorem probability_sum_eq (h : pairwise_mutually_exclusive A):
  P (λ ω, ∃ i, A i ω) = ∑' i, P (A i) :=
sorry

end probability_sum_eq_l607_607194


namespace product_of_roots_l607_607761

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end product_of_roots_l607_607761


namespace average_age_correct_l607_607104

def ratio (m w : ℕ) : Prop := w * 8 = m * 9

def average_age_of_group (m w : ℕ) (avg_men avg_women : ℕ) : ℚ :=
  (avg_men * m + avg_women * w) / (m + w)

/-- The average age of the group is 32 14/17 given that the ratio of the number of women to the number of men is 9 to 8, 
    the average age of the women is 30 years, and the average age of the men is 36 years. -/
theorem average_age_correct
  (m w : ℕ)
  (h_ratio : ratio m w)
  (h_avg_women : avg_age_women = 30)
  (h_avg_men : avg_age_men = 36) :
  average_age_of_group m w avg_age_men avg_age_women = 32 + (14 / 17) := 
by
  sorry

end average_age_correct_l607_607104


namespace m_not_in_P_l607_607923

def P := {x : ℝ | x^2 - real.sqrt 2 * x ≤ 0}
def m := real.sqrt 3

theorem m_not_in_P : m ∉ P :=
by
  sorry

end m_not_in_P_l607_607923


namespace find_a_b_find_min_g_l607_607440

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - x^2 + a * x + b

theorem find_a_b (a b : ℝ) (h_tangent : ∀ x, f x a b = -x + 1 → x = 0) : a = -1 ∧ b = 1 :=
by
  have h₀ : f 0 a b = b := by simp [f]
  have h₁ : -0 + 1 = 1 := by norm_num
  have h₂ : b = 1 := by rw [←h_tangent 0 h₁]
  have hf' : ∀ x, deriv (f x a b) = 3 * x^2 - 2 * x + a := by simp [f, deriv]
  have h₃ : deriv (f 0 a b) = a := by simp [hf']
  have h_slope : ∀ x, deriv (λ x, -x + 1) x = -1 := by simp [deriv]
  have h₄ : a = -1 := by rw [←h_slope 0, h₃]
  exact ⟨h₄, h₂⟩

noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - x + 1

theorem find_min_g : ∃ x ∈ Icc (-2 : ℝ) 2, g x = -9 :=
by
  have eval_g : ∀ x, g x = x^3 - x^2 - x + 1 := by simp [g]
  have h_g_neg2 : g (-2) = -9 := by norm_num [g, eval_g]
  use -2
  split
  · norm_num
  · exact h_g_neg2

end find_a_b_find_min_g_l607_607440


namespace intersection_on_circumcircle_l607_607271

variables {A B C P Q H M N : Type}
variable [Triangle A B C]

-- Conditions as hypotheses:
variables (hABC : acute_triangle A B C)
variables (hABltAC : AB < AC)
variables (hPerpBisectorPQ : perp_bisector BC intersects AB at P ∧ perp_bisector BC intersects AC at Q)
variables (hH : orthocenter H A B C)
variables (hM : midpoint M B C)
variables (hN : midpoint N P Q)

-- The goal to prove:
theorem intersection_on_circumcircle (hABC : acute_triangle A B C) (hABltAC : AB < AC)
    (hPerpBisectorPQ : perp_bisector BC intersects AB at P ∧ perp_bisector BC intersects AC at Q)
    (hH : orthocenter H A B C) (hM : midpoint M B C) (hN : midpoint N P Q) :
    let Intersection := line_intersection (line H M) (line A N) in
    lies_on_circumcircle Intersection A B C :=
sorry

end intersection_on_circumcircle_l607_607271


namespace ellipse_equation_l607_607355

theorem ellipse_equation
  (x y t : ℝ)
  (h1 : x = (3 * (Real.sin t - 2)) / (3 - Real.cos t))
  (h2 : y = (4 * (Real.cos t - 6)) / (3 - Real.cos t))
  (h3 : ∀ t : ℝ, (Real.cos t)^2 + (Real.sin t)^2 = 1) :
  ∃ (A B C D E F : ℤ), (9 * x^2 + 36 * x * y + 9 * y^2 + 216 * x + 432 * y + 1440 = 0) ∧ 
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2142) :=
sorry

end ellipse_equation_l607_607355


namespace Bexy_bicycle_speed_correct_l607_607341

-- Definitions based on the conditions
def BexyWalkSpeed : ℕ := 5 -- Bexy walks 5 miles in one hour
def BexyWalkTime : ℕ := 1 -- in one hour
def BenRoundTripTimeMin : ℕ := 160 -- Ben spends 160 minutes on his round trip
def BenRelativeSpeedRatio : ℚ := 1 / 2 -- Ben’s average speed is half of Bexy’s average speed

-- Noncomputable definition for Bexy's bicycle speed to state the target property
noncomputable def BexyBikeSpeed : ℚ := 7.5 -- the correct answer based on the conditions and proof

theorem Bexy_bicycle_speed_correct :
  ∀ (walk_speed walk_time : ℕ) (round_trip_time_min : ℕ) (relative_speed_ratio : ℚ),
  walk_speed = BexyWalkSpeed →
  walk_time = BexyWalkTime →
  round_trip_time_min = BenRoundTripTimeMin →
  relative_speed_ratio = BenRelativeSpeedRatio →
  let BenRoundTripTimeHours := (round_trip_time_min : ℚ) / 60 in
  let BenWalkTime := 2 * walk_time in
  let BenBikeTime := BenRoundTripTimeHours - BenWalkTime in
  let BenBikeSpeed := walk_speed / (2 * BenBikeTime) in
  BexyBikeSpeed = 2 * BenBikeSpeed :=
begin
  intros,
  sorry,
end

end Bexy_bicycle_speed_correct_l607_607341


namespace relationship_f1_f_l607_607809

theorem relationship_f1_f'1 (f : ℝ → ℝ) (m : ℝ) 
  (h_tangent : ∀ x, f x = 2 * x - 1)
  (h_derivative : ∀ x, deriv f x = f' x) : 
  f(1) < f'(1) := 
by 
  sorry

end relationship_f1_f_l607_607809


namespace find_y_coordinate_l607_607300

-- Given conditions
def slope : ℝ := 3 / 4
def x_intercept : ℝ × ℝ := (400, 0)
def point_x : ℝ := -12

-- Y-coordinate calculation given the conditions
theorem find_y_coordinate : 
  ∃ y : ℝ, 
  y = slope * point_x - (x_intercept.snd - slope * x_intercept.fst) :=
by
  use -309
  sorry

end find_y_coordinate_l607_607300


namespace complex_mult_trig_identities_l607_607426

noncomputable def z1 : ℂ := complex.cos (23 * real.pi / 180) + complex.sin (23 * real.pi / 180) * complex.I
noncomputable def z2 : ℂ := complex.cos (37 * real.pi / 180) + complex.sin (37 * real.pi / 180) * complex.I
noncomputable def z_result : ℂ := (1/2 : ℂ) + (complex.I) * (real.sqrt 3 / 2)

theorem complex_mult_trig_identities :
  z1 * z2 = z_result := by
  sorry

end complex_mult_trig_identities_l607_607426


namespace median_pets_l607_607853

theorem median_pets {pets : List ℕ} (h_len : pets.length = 15): 
  pets.sorted.nth (15 / 2) = some 3 :=
by
  sorry

end median_pets_l607_607853


namespace problem_statement_l607_607919

noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + ...))

theorem problem_statement : x = 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + ...)) →
  |((3 : ℤ) : ℝ)| + |((3 : ℤ) : ℝ)| + |((-6 : ℤ) : ℝ)| = 12 :=
by
  intro hx
  -- remainder of proof would be here
  sorry

end problem_statement_l607_607919


namespace binomial_seven_four_l607_607723

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607723


namespace exists_short_distance_l607_607688

noncomputable def minDistanceInSquare (points : Fin 7 → (ℝ × ℝ)) : ℝ :=
  let maxSideLength : ℝ := 2
  let minB : ℝ := Real.sqrt 2
  if ∀ p1 p2 : Fin 7, p1 ≠ p2 → (dist points[p1] points[p2] ≤ minB) then minB else 0

theorem exists_short_distance (points : Fin 7 → (ℝ × ℝ))
  (inside_square : ∀ i, fst (points i) ≥ 0 ∧ fst (points i) ≤ 2 ∧ snd (points i) ≥ 0 ∧ snd (points i) ≤ 2) :
  ∃ (p1 p2 : Fin 7), p1 ≠ p2 ∧ dist points[p1] points[p2] ≤ Real.sqrt 2 := 
  sorry

end exists_short_distance_l607_607688


namespace probability_one_copresident_l607_607238

noncomputable def binom (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

theorem probability_one_copresident (clubs : fin 4 → ℕ) (h : clubs = ![6, 8, 9, 10])
  (copres : ∀ c, 3 ≤ clubs c) :
  let prob_6 := (3 * binom 3 3) / binom 6 4,
      prob_8 := (3 * binom 5 3) / binom 8 4,
      prob_9 := (3 * binom 6 3) / binom 9 4,
      prob_10 := (3 * binom 7 3) / binom 10 4 in
  (1 / 4) * (prob_6 + prob_8 + prob_9 + prob_10) = 101 / 420 :=
by
  sorry

end probability_one_copresident_l607_607238


namespace hyperbola_equation_foci_shared_l607_607423

theorem hyperbola_equation_foci_shared :
  ∃ m : ℝ, (∃ c : ℝ, c = 2 * Real.sqrt 2 ∧ 
              ∃ a b : ℝ, a^2 = 12 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2) ∧ 
    (c = 2 * Real.sqrt 2 → (∃ a b : ℝ, a^2 = m ∧ b^2 = m - 8 ∧ c^2 = a^2 + b^2)) → 
  (∃ m : ℝ, m = 7) := 
sorry

end hyperbola_equation_foci_shared_l607_607423


namespace length_of_AB_l607_607982

theorem length_of_AB : 
  let circle := {p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 1) ^ 2 = 1} in
  let line := {p : ℝ × ℝ | p.1 + p.2 = 1} in
  let intersections := circle ∩ line in
  ∃ A B : ℝ × ℝ, A ∈ intersections ∧ B ∈ intersections ∧ dist A B = 2 :=
by 
  sorry

end length_of_AB_l607_607982


namespace probability_distribution_X_expectation_X_maximize_f_l607_607598

-- Definitions for part 1
def trialProducedProducts := {0, 1, 2, 3, 4, 5}  -- 6 products
def defectiveProducts : Finset ℕ := {a, b}  -- exactly 2 defective (arbitrary a, b ∈ {0, ..., 5})

-- Random variable X: number of inspections until both defective products are found
def X : ℕ → ProbabilityTheory.Measure ℕ := sorry  -- Lean definition of X as random variable (detailed construction is skipped)

theorem probability_distribution_X :
  (ProbabilityTheory.Prob (X = 2) = 1/15) ∧
  (ProbabilityTheory.Prob (X = 3) = 2/15) ∧
  (ProbabilityTheory.Prob (X = 4) = 4/15) ∧
  (ProbabilityTheory.Prob (X = 5) = 8/15) :=
sorry

theorem expectation_X : ProbabilityTheory.Measure_Theory.Expectation X = 64/15 := sorry

-- Definitions for part 2
noncomputable def f (p : ℝ) : ℝ :=
  Nat.choose 50 2 * p^2 * (1 - p)^(48)

theorem maximize_f (h : 0 < p ∧ p < 1) : 
  (∃ p, p = 1/25 ∧ ∀ q, f(q) ≤ f(1/25)) :=
sorry

end probability_distribution_X_expectation_X_maximize_f_l607_607598


namespace swap_correct_l607_607109

variable (a b c : ℕ)

noncomputable def swap_and_verify (a : ℕ) (b : ℕ) : Prop :=
  let c := b
  let b := a
  let a := c
  a = 2012 ∧ b = 2011

theorem swap_correct :
  ∀ a b : ℕ, a = 2011 → b = 2012 → swap_and_verify a b :=
by
  intros a b ha hb
  sorry

end swap_correct_l607_607109


namespace exists_parallel_line_l607_607637

variable (Plane Line : Type)
variable (perp : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)

theorem exists_parallel_line
  (α β γ : Plane)
  (a b : Line) :
  perp β γ →
  intersect α γ ∧ ¬perp α γ →
  ∃ a, subset a α ∧ parallel a γ :=
by
  sorry

end exists_parallel_line_l607_607637


namespace second_oldest_brother_age_l607_607453

theorem second_oldest_brother_age
  (y s o : ℕ)
  (h1 : y + s + o = 34)
  (h2 : o = 3 * y)
  (h3 : s = 2 * y - 2) :
  s = 10 := by
  sorry

end second_oldest_brother_age_l607_607453


namespace problem_r_minus_s_l607_607162

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l607_607162


namespace medals_awarded_correctly_l607_607108

def totalWaysToAwardMedals (totalAthletes europeans asians: ℕ) :=
  if totalAthletes = 10 ∧ europeans = 4 ∧ asians = 6 then
    let case1 := 6 * 5 * 4
    let case2 := 4 * 3 * (6 * 5)
    let case3 := (Nat.choose 4 2) * 3 * 6
    case1 + case2 + case3
  else 0

theorem medals_awarded_correctly :
  totalWaysToAwardMedals 10 4 6 = 588 := by
    simp [totalWaysToAwardMedals]
    sorry

end medals_awarded_correctly_l607_607108


namespace factorize_negative_quadratic_l607_607738

theorem factorize_negative_quadratic (x y : ℝ) : 
  -4 * x^2 + y^2 = (y - 2 * x) * (y + 2 * x) :=
by 
  sorry

end factorize_negative_quadratic_l607_607738


namespace sum_of_k_values_l607_607610

theorem sum_of_k_values :
  (∑ k in {k | ∃ (p q : ℤ), p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧ pq = 8 ∧ k = 2 * (p + q)}, k) = 0 :=
by sorry

end sum_of_k_values_l607_607610


namespace pole_height_l607_607669

theorem pole_height (AC AD DE : ℝ) (hAC : AC = 5) (hAD : AD = 3) (hDE : DE = 1.75) : 
  let DC := AC - AD in
  DC = 2 →
  let AB := 5 * (1.75 / 2) in
  AB = 4.375 :=
by
  intros DC_eq AB_eq
  simp only []
  sorry

end pole_height_l607_607669


namespace g_h_2_equals_584_l607_607080

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l607_607080


namespace quadratic_polynomial_value_q10_l607_607514

theorem quadratic_polynomial_value_q10 (q : ℝ → ℝ) (h_q : ∃ a b c : ℝ, q = λ x, a * x^2 + b * x + c)
  (h_div : ∀ x : ℝ, x = 2 ∨ x = -2 ∨ x = 5 → ([q x]^3 + x = 0)) :
  q 10 = -139 * real.cbrt 2 :=
by sorry

end quadratic_polynomial_value_q10_l607_607514


namespace marks_lost_per_wrong_answer_l607_607475

variable (x : ℕ)
variable (correctAnswers wrongAnswers : ℕ)
variable (scorePerCorrectAnswer : ℕ)
variable (totalQuestions totalScore : ℕ)

def exam_condition_1 := scorePerCorrectAnswer = 3
def exam_condition_2 := totalQuestions = 120
def exam_condition_3 := totalScore = 180
def exam_condition_4 := correctAnswers = 75
def exam_condition_5 := wrongAnswers = totalQuestions - correctAnswers

theorem marks_lost_per_wrong_answer :
  exam_condition_1 →
  exam_condition_2 →
  exam_condition_3 →
  exam_condition_4 →
  exam_condition_5 →
  correctAnswers * scorePerCorrectAnswer - wrongAnswers * x = totalScore →
  x = 1 :=
by
  intros
  sorry

end marks_lost_per_wrong_answer_l607_607475


namespace minimum_radius_circle_eqn_l607_607480

theorem minimum_radius_circle_eqn (A B : ℝ × ℝ) 
  (hA : 3 * A.1 + A.2 - 10 = 0) 
  (hB : 3 * B.1 + B.2 - 10 = 0) 
  (M_center : ℝ × ℝ) 
  (hM : ∀ (A B : ℝ × ℝ), (3 * A.1 + A.2 - 10 = 0) → (3 * B.1 + B.2 - 10 = 0) → (circle_center A B = M_center ∧ (distance_from_center_to_line M_center (3, 1, -10) = sqrt 10))) 
  : ∃ C : ℝ × ℝ, ∃ R : ℝ, circle_eqn C R = (x - 3)^2 + (y - 1)^2 - 10 := 
sorry

end minimum_radius_circle_eqn_l607_607480


namespace find_z_l607_607802

noncomputable theory

def is_imaginary_unit (i : ℂ) : Prop := i = complex.I

def satisfies_condition (i z : ℂ) : Prop := (1 + i) * z = 1 + 3 * i

theorem find_z (i z : ℂ) (h1 : is_imaginary_unit i) (h2 : satisfies_condition i z) : z = 2 + i :=
by
  sorry

end find_z_l607_607802


namespace binomial_7_4_eq_35_l607_607717

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l607_607717


namespace evaluation_of_expression_l607_607142

variable {a b x y : ℝ}

noncomputable def expression : ℝ :=
  (ax + y / b)⁻¹ * ((ax)⁻¹ + (y / b)⁻¹)

theorem evaluation_of_expression
    (ha : a ≠ 0) (hb : b ≠ 0) (h : ax + y / b ≠ 0) :
  expression = (a * x * y)⁻¹ :=
by
  sorry

end evaluation_of_expression_l607_607142


namespace average_value_l607_607032

theorem average_value 
  (x : Fin 10 → ℝ) 
  (h1 : ∑ i, |x i - 1| ≤ 4) 
  (h2 : ∑ i, |x i - 2| ≤ 6) : 
  (∑ i, x i) / 10 = 1.4 :=
by
  sorry

end average_value_l607_607032


namespace geometric_sequence_common_ratio_l607_607661

/--
  Given a geometric sequence with the first three terms:
  a₁ = 27,
  a₂ = 54,
  a₃ = 108,
  prove that the common ratio is r = 2.
-/
theorem geometric_sequence_common_ratio :
  let a₁ := 27
  let a₂ := 54
  let a₃ := 108
  ∃ r : ℕ, (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ r = 2 := by
  sorry

end geometric_sequence_common_ratio_l607_607661


namespace least_q_l607_607003

def gcd (a b : ℕ) : ℕ := nat.gcd a b
def f (p q i : ℕ) : ℕ := i % q

theorem least_q (p q a b : ℕ) (hpq : gcd p q = 1) 
  (ha : a = finset.filter (λ i, f p q i = q-1) (finset.range q).card) 
  (hb : b = finset.filter (λ i, f p q i = 0) (finset.range q).card) : 
  q = a * b + 1 := 
sorry

end least_q_l607_607003


namespace number_of_intervals_l607_607618

theorem number_of_intervals (maxHeight minHeight classInterval : ℝ) 
  (h1 : maxHeight = 173) (h2 : minHeight = 140) (h3 : classInterval = 5) : 
  Nat.ceil ((maxHeight - minHeight) / classInterval) = 7 := by
  sorry

end number_of_intervals_l607_607618


namespace least_max_points_on_circle_l607_607644

theorem least_max_points_on_circle 
    (P : Fin 10 → ℝ × ℝ) 
    (h : ∀ (S : Finset (Fin 10)), S.card = 5 → ∃ C : Finset (Fin 10), C.card ≥ 4 ∧ ∃ (O : ℝ × ℝ) (r : ℝ), ∀ p ∈ C, dist p.val O = r) :
    ∃ C : Finset (Fin 10), C.card ≥ 9 ∧ ∃ (O : ℝ × ℝ) (r : ℝ), ∀ p ∈ C, dist p.val O = r :=
begin
    -- solution goes here
    sorry
end

end least_max_points_on_circle_l607_607644


namespace binomial_seven_four_l607_607721

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607721


namespace minimum_n_constant_term_l607_607850

noncomputable def binomial_constant_term (n : ℕ) : ℚ :=
  let term := λ r : ℕ, ((-1/2)^r) * 3^(n-r) * (Nat.choose n r : ℚ)
  if h : ∃ r, 2 * n = 5 * r then
    let r := Nat.find h in
    term r
  else 0

theorem minimum_n_constant_term :
  (binomial_constant_term 5) = 135 / 2 :=
by
  sorry

end minimum_n_constant_term_l607_607850


namespace parabola_area_correct_l607_607345

noncomputable def parabola_area : ℝ :=
  let f : ℝ → ℝ := λ x, -x^2 + 3 * x - 2
  (∫ x in 0..1, f x).abs + (∫ x in 1..2, f x).abs

theorem parabola_area_correct : parabola_area = 4 / 3 :=
sorry

end parabola_area_correct_l607_607345


namespace increasing_interval_of_f_l607_607034

-- Assuming f is a real-valued function and a is a real number
variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Conditions
axiom deriv_f : ∀ x, deriv f x = (a - 1) * x^2 + a * x + 1
axiom even_f' : ∀ x, deriv f (-x) = deriv f x

-- Theorem statement
theorem increasing_interval_of_f : (a = 0) → Ioo (-1 : ℝ) 1 ⊆ {x | ∀ x, deriv f x > 0} :=
by
  intros ha0
  rw [ha0, deriv_f]
  sorry

end increasing_interval_of_f_l607_607034


namespace g_h_2_equals_584_l607_607081

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l607_607081


namespace polynomial_satisfies_condition_l607_607750

def floor (v : ℝ) : ℤ := Int.floor v

theorem polynomial_satisfies_condition :
  ∃ P : ℤ → ℤ → ℤ, (∀ a : ℝ, P (floor a) (floor (2 * a)) = 0) :=
by
  let P : ℤ → ℤ → ℤ := λ x y, (y - 2 * x) * (y - 2 * x - 1)
  use P
  intro a
  sorry

end polynomial_satisfies_condition_l607_607750


namespace distance_AB_l607_607027

noncomputable theory

open Real

/-- Define the ellipse centered at the origin with eccentricity 1/2 -/
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 12) = 1

/-- Define the directrix of the given parabola y^2 = 8x -/
def directrix (x : ℝ) : Prop := x = -2

/-- The focus of the parabola y^2 = 8x is at (2, 0) -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The distance between the intersection points of the ellipse and the directrix of the parabola -/
theorem distance_AB : ∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ directrix A.1 ∧ directrix B.1 ∧ |A.2 - B.2| = 6 :=
by
  sorry

end distance_AB_l607_607027


namespace shari_total_distance_l607_607951

theorem shari_total_distance (speed : ℝ) (time_1 : ℝ) (rest : ℝ) (time_2 : ℝ) (distance : ℝ) :
  speed = 4 ∧ time_1 = 2 ∧ rest = 0.5 ∧ time_2 = 1 ∧ distance = speed * time_1 + speed * time_2 → distance = 12 :=
by
  sorry

end shari_total_distance_l607_607951


namespace total_votes_poled_l607_607872

theorem total_votes_poled (V : ℕ) : 
  (∀ (VA VB : ℕ), VA = VB + (15 / 100) * (8 / 10) * V ∧ 
                  VB = 1859 ∧ 
                  VA + VB = 8 / 10 * V) → 
  V = 5468 := 
begin
  sorry
end

end total_votes_poled_l607_607872


namespace gasoline_price_increase_l607_607576

theorem gasoline_price_increase 
  (highest_price : ℝ) (lowest_price : ℝ) 
  (h_high : highest_price = 17) 
  (h_low : lowest_price = 10) : 
  (highest_price - lowest_price) / lowest_price * 100 = 70 := 
by
  /- proof can go here -/
  sorry

end gasoline_price_increase_l607_607576


namespace trigonometric_identity_theorem_l607_607006

noncomputable def trigonometric_identity_proof (α: ℝ) : Prop :=
  (sin (π / 2 + α) = 1 / 3) → (cos (π + 2 * α) = 7 / 9)

theorem trigonometric_identity_theorem (α: ℝ) : trigonometric_identity_proof α :=
  by
    intro h
    sorry

end trigonometric_identity_theorem_l607_607006


namespace find_point_D_l607_607488

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2 + (p.z - q.z) ^ 2)

theorem find_point_D :
  ∃ (D : Point3D), D.x = 0 ∧ D.y = 0 ∧ distance {x := 2, y := 0, z := 0} D = distance {x := 0, y := 2, z := 10} D ∧ D.z = 5 :=
by
  use {x := 0, y := 0, z := 5}
  split
  { refl }
  split
  { refl }
  split
  { sorry }
  { refl }

end find_point_D_l607_607488


namespace fans_standing_l607_607537

-- Definitions of the problem conditions
def fans : List String := ["A", "B", "C", "D", "E"]

def left_end_restriction (perm : List String) : Bool :=
  ¬(perm.head = "A" ∨ perm.head = "B")

def ac_together (perm : List String) : Bool :=
  let idx_A := perm.indexOf "A"
  let idx_C := perm.indexOf "C"
  (idx_A ≠ -1) ∧ (idx_C ≠ -1) ∧ (abs (idx_A - idx_C) = 1)

-- Main statement for the proof
theorem fans_standing : ∃ perm : List (List String), 
  (perm.all left_end_restriction) ∧ 
  (perm.all ac_together) ∧ 
  (perm.length = 30) :=
by
  sorry

end fans_standing_l607_607537


namespace related_events_l607_607256

noncomputable def K_squared : ℝ := 4.0  -- Example value, to use symbolic value use a variable

def condition (K_squared : ℝ) : Prop := K_squared > 3.841

theorem related_events (K_squared : ℝ) (h : condition K_squared) : 
  ∀ (A B : Type), 95% certain to be related :=
sorry

end related_events_l607_607256


namespace binomial_seven_four_l607_607724

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607724


namespace radius_of_circle_centered_at_l607_607656

def center : ℝ × ℝ := (3, 4)

def intersects_axes_at_three_points (A : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r = 0 ∨ A.1 + r = 0) ∧ (A.2 - r = 0 ∨ A.2 + r = 0)

theorem radius_of_circle_centered_at (A : ℝ × ℝ) : 
  (intersects_axes_at_three_points A 4) ∨ (intersects_axes_at_three_points A 5) :=
by
  sorry

end radius_of_circle_centered_at_l607_607656


namespace cans_equilibrium_possible_l607_607631

theorem cans_equilibrium_possible (d : ℕ → ℕ) :
  (∀ d, (∀ j, d j = 0) ∨ (∀ j, d j = j) ∨ (∀ j, d j = 2016 - j)) →
  (∃ N, ∀ j, N = d j) :=
by
  assume initial_configs
  apply sorry

end cans_equilibrium_possible_l607_607631


namespace problem_r_minus_s_l607_607159

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l607_607159


namespace product_of_roots_l607_607762

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end product_of_roots_l607_607762


namespace range_of_x_range_of_m_l607_607639

-- Statement for Question 1
theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (ineq : |2 * a + b| + |2 * a - b| ≥ |a| * (|2 + x| + |2 - x|)) : x ∈ Icc (-2 : ℝ) 2 :=
sorry

-- Statement for Question 2
theorem range_of_m (m : ℝ) (ineq : ∀ x ∈ Icc 4 16, (2 * log x / log 4 - 1/2) ≥ m * (log x / log 4)) : m ≤ 3/2 :=
sorry

end range_of_x_range_of_m_l607_607639


namespace part_a_part_b_part_c_part_d_part_e_l607_607535

-- Define the Catalan numbers recursively
def catalan : ℕ → ℕ 
| 0       := 1
| (n + 1) := ∑ i in Finset.range (n + 1), catalan i * catalan (n - i)

-- Given the recursive definition of Catalan numbers
def catalan_recurrence (n : ℕ) : ℕ :=
∑ i in Finset.range (n + 1), catalan i * catalan (n - i)

-- Part (a)
theorem part_a (n : ℕ) (h : n ≥ 3) : N_n = catalan n := sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) : N_n = catalan (n - 1) := sorry

-- Part (c)
theorem part_c (n : ℕ) (h : n > 1) 
  (C₀ : ℕ) (h₀ : C₀ = 0) 
  (C₁ : ℕ) (h₁ : C₁ = 1) 
  (C : ℕ → ℕ) 
  (hC : ∀ n > 1, C n = ∑ i in Finset.range (n - 1), C i * C (n - i)) : 
  ∀ n > 1, C n = ∑ i in Finset.range (n - 1), C i * C (n - i) := 
sorry

-- Part (d)
noncomputable def generating_function (x : ℝ) : ℝ := 
∑ n in Finset.range ∞, (catalan n) * x ^ n

theorem part_d (x : ℝ) : generating_function x = x + (generating_function x) ^ 2 := sorry

-- Part (e)
noncomputable def generating_function_explicit (x : ℝ) : ℝ := 
(1 - (1 - 4 * x) ^ (1 / 2)) / 2

theorem part_e 
  (h₀ : generating_function_explicit 0 = 0) 
  (x : ℝ) 
  (hx : |x| ≤ 1 / 4) :
  generating_function x = generating_function_explicit x ∧ 
  ∀ n : ℕ, (finsupp.effective_support (generating_function x)).coeff n = catalan n := sorry

end part_a_part_b_part_c_part_d_part_e_l607_607535


namespace drink_all_tea_l607_607539

theorem drink_all_tea (cups : Fin 30 → Prop) (red blue : Fin 30 → Prop)
  (h₀ : ∀ n, cups n ↔ (red n ↔ ¬ blue n))
  (h₁ : ∃ a b, a ≠ b ∧ red a ∧ blue b)
  (h₂ : ∀ n, red n → red (n + 2))
  (h₃ : ∀ n, blue n → blue (n + 2)) :
  ∃ sequence : ℕ → Fin 30, (∀ n, cups (sequence n)) ∧ (sequence 0 ≠ sequence 1) 
  ∧ (∀ n, cups (sequence (n+1))) :=
by
  sorry

end drink_all_tea_l607_607539


namespace ratio_of_fifth_terms_l607_607064

theorem ratio_of_fifth_terms (a_n b_n : ℕ → ℕ) (S T : ℕ → ℕ)
  (hs : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (ht : ∀ n, T n = n * (b_n 1 + b_n n) / 2)
  (h : ∀ n, S n / T n = (7 * n + 2) / (n + 3)) :
  a_n 5 / b_n 5 = 65 / 12 :=
by
  sorry

end ratio_of_fifth_terms_l607_607064


namespace min_distance_point_to_line_l607_607217

-- Lean 4 statement to prove the equivalence
theorem min_distance_point_to_line (P : Point) (line : Line) 
  (h : ¬(P ∈ line)) : 
  ∃ A ∈ line, ∀ B ∈ line, distance P B ≥ distance P A := 
sorry

end min_distance_point_to_line_l607_607217


namespace all_coefficients_of_Q_are_good_numbers_l607_607629

def is_good_number (x : ℝ) : Prop :=
  ∃ (a b : ℤ), x = a + b * (real.sqrt 2)

structure polynomial_with_good_coeffs (R : Type*) [ring R] :=
  (coeffs : nat → R)
  (is_good : ∀ n, is_good_number (coeffs n))

variables {A B Q : polynomial_with_good_coeffs ℝ}
  (hA : ∀ n, is_good_number (A.coeffs n))
  (hB : ∀ n, is_good_number (B.coeffs n))
  (b0 : B.coeffs 0 = 1)
  (hQ : ∀ n, Q.coeffs n = B.coeffs n * Q.coeffs n)

theorem all_coefficients_of_Q_are_good_numbers :
  ∀ n, is_good_number (Q.coeffs n) :=
sorry

end all_coefficients_of_Q_are_good_numbers_l607_607629


namespace cone_volume_l607_607223

noncomputable def volume_of_cone (R : ℝ) : ℝ :=
  (π * R^3 * Real.sqrt 15) / 3

theorem cone_volume (R : ℝ) (h_sector_angle : ∠ sector = π / 2) : 
  volume_of_cone R = (π * R^3 * Real.sqrt 15) / 3 :=
by
  sorry

end cone_volume_l607_607223


namespace seq_v13_eq_b_l607_607357

noncomputable def seq (v : ℕ → ℝ) (b : ℝ) : Prop :=
v 1 = b ∧ ∀ n ≥ 1, v (n + 1) = -1 / (v n + 2)

theorem seq_v13_eq_b (b : ℝ) (hb : 0 < b) (v : ℕ → ℝ) (hs : seq v b) : v 13 = b := by
  sorry

end seq_v13_eq_b_l607_607357


namespace ratio_12_minutes_to_1_hour_l607_607269

theorem ratio_12_minutes_to_1_hour : 
  let one_hour_in_minutes : ℕ := 60 
  in (12 / gcd 12 one_hour_in_minutes = 1) ∧ (one_hour_in_minutes / gcd 12 one_hour_in_minutes = 5) :=
by
  let one_hour_in_minutes := 60
  have h1 : gcd 12 one_hour_in_minutes = 12 := by sorry
  have h2 : 12 / gcd 12 one_hour_in_minutes = 1 := by sorry
  have h3 : one_hour_in_minutes / gcd 12 one_hour_in_minutes = 5 := by sorry
  exact ⟨h2, h3⟩

end ratio_12_minutes_to_1_hour_l607_607269


namespace probability_at_least_one_boy_and_one_girl_l607_607336

theorem probability_at_least_one_boy_and_one_girl :
  (∀ n : ℕ, (P (X n = Boy) = 1/2) ∧ (P (X n = Girl) = 1/2)) →
  ∃ P_boys : ℝ, ∃ P_girls : ℝ,
    (P_boys = (1/2)^4) ∧ (P_girls = (1/2)^4) →
    1 - P_boys - P_girls = 7/8 :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l607_607336


namespace product_of_roots_l607_607763

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem product_of_roots (x : ℝ) : 
  (4 * f (3 - x) - f x = 3 * x^2 - 4 * x - 3) →
  (Exists (λ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5)) :=
by
  sorry

end product_of_roots_l607_607763


namespace equal_circumradii_imp_ratio_l607_607098

theorem equal_circumradii_imp_ratio {A B C N : Type} 
  [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space N]
  (AB BC AC : ℝ)
  (h_AB : AB = 15) (h_BC : BC = 14) (h_AC : AC = 16) 
  (x : ℝ) (AN NC : ℝ)
  (h_AN : AN = x) (h_NC : NC = 16 - x)
  (h_circumradii : circumradius A B N = circumradius B N C) :
  x / (16 - x) = 15 / 14 := 
sorry

end equal_circumradii_imp_ratio_l607_607098


namespace quadratic_point_ellipse_l607_607021

theorem quadratic_point_ellipse
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (vertex_condition : (-b / (2 * a), -1 / (4 * a)) = (-b / (2 * a), -1 / (4 * a)))
  (circle_condition : (⟨-b / (2 * a), 0⟩ : ℝ × ℝ) = ⟨-b / (2 * a), 0⟩)
  (intersection_condition : |(((-b / (2 * a))^2 + 16) = (b^2 - 4 * a * c) / (4 * a^2))|)
  (simplified_condition : (b^2 - 4 * a * c = 1)) :
  b^2 + (c^2 / 4) = 1 :=
sorry

end quadratic_point_ellipse_l607_607021


namespace plug_counts_l607_607835

theorem plug_counts
(h_mittens_a : 100)
(h_mittens_b : 50)
(h_plugs_total : (100 + 50) + 20 = 170)
(h_x_y_relation : ∀ X Y : ℕ, (X + 30) = 2 * Y)
(h_z_plugs : ∀ Z : ℕ, Z = 100 / 2):
∃ X Y Z : ℕ, X = 100 ∧ Y = 50 ∧ Z = 50 :=
by
  sorry

end plug_counts_l607_607835


namespace sum_of_coeffs_zero_l607_607356

theorem sum_of_coeffs_zero (A B C D E F : ℝ) :
  (1 : ℝ) = A*(x + 1)*(x + 2)*(x + 3)*(x + 4)*(x + 5) +
            B*x*(x + 2)*(x + 3)*(x + 4)*(x + 5) +
            C*x*(x + 1)*(x + 3)*(x + 4)*(x + 5) +
            D*x*(x + 1)*(x + 2)*(x + 4)*(x + 5) +
            E*x*(x + 1)*(x + 2)*(x + 3)*(x + 5) +
            F*x*(x + 1)*(x + 2)*(x + 3)*(x + 4) :=
  A + B + C + D + E + F = 0 :=
begin
  sorry
end

end sum_of_coeffs_zero_l607_607356


namespace range_of_a_l607_607825

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ∧ ¬proposition_q a ↔ (1 / 4 < a ∧ a < 4) :=
begin
  sorry
end

end range_of_a_l607_607825


namespace nobody_but_angela_finished_9_problems_l607_607334

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end nobody_but_angela_finished_9_problems_l607_607334


namespace eugene_payment_correct_l607_607869

noncomputable def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (original_price * discount_rate)

noncomputable def total_cost (quantity : ℕ) (price : ℝ) : ℝ :=
  quantity * price

noncomputable def eugene_total_cost : ℝ :=
  let tshirt_price := discounted_price 20 0.10
  let pants_price := discounted_price 80 0.10
  let shoes_price := discounted_price 150 0.15
  let hat_price := discounted_price 25 0.05
  let jacket_price := discounted_price 120 0.20
  let total_cost_before_tax := 
    total_cost 4 tshirt_price + 
    total_cost 3 pants_price + 
    total_cost 2 shoes_price + 
    total_cost 3 hat_price + 
    total_cost 1 jacket_price
  total_cost_before_tax + (total_cost_before_tax * 0.06)

theorem eugene_payment_correct : eugene_total_cost = 752.87 := by
  sorry

end eugene_payment_correct_l607_607869


namespace compare_f_values_l607_607787

-- Function defined on ℝ that is increasing on the interval [-4, +∞)
def is_increasing {α β : Type*} [Preorder α] (f : α → β) (s : Set α) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Even function condition for y = f(x - 4)
def is_even {α β : Type*} [Preorder α] {f : α → β} : Prop :=
  ∀ x, f (-x - 4) = f (x - 4)

-- The primary theorem we want to prove
theorem compare_f_values
  (f : ℝ → ℝ)
  (h1 : is_increasing f {x : ℝ | -4 ≤ x})
  (h2 : is_even f) :
  f (-4) < f (-6) ∧ f (-6) < f (0) :=
by
  sorry

end compare_f_values_l607_607787


namespace value_of_f_l607_607012

noncomputable def f : ℝ → ℝ := λ x, if x < 1 then Real.cos (Real.pi * x) else f (x - 1)

theorem value_of_f :
  f (1 / 3) + f (4 / 3) = 1 := by
  sorry

end value_of_f_l607_607012


namespace milk_left_over_yesterday_l607_607695

theorem milk_left_over_yesterday (M_m M_e M_s M_l : ℕ) (H1 : M_m = 365) (H2 : M_e = 380) (H3 : M_s = 612) (H4 : M_l = 148) :
  M_l - (M_m + M_e - M_s) = 15 :=
by
  rw [H1, H2, H3, H4]
  sorry

end milk_left_over_yesterday_l607_607695


namespace smallest_f1_value_l607_607766

noncomputable def polynomial := 
  fun (f : ℝ → ℝ) (r s : ℝ) => 
    f = λ x => (x - r) * (x - s) * (x - ((r + s)/2))

def distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ polynomial f r s ∧ 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (f ∘ f) a = 0 ∧ (f ∘ f) b = 0 ∧ (f ∘ f) c = 0)

theorem smallest_f1_value
  (f : ℝ → ℝ)
  (hf : distinct_real_roots f) :
  ∃ r s : ℝ, r ≠ s ∧ f 1 = 3/8 :=
sorry

end smallest_f1_value_l607_607766


namespace alice_price_per_acorn_l607_607457

theorem alice_price_per_acorn
  (num_acorns : ℕ) (bob_paid : ℕ) (alice_factor : ℕ) (num_acorns_alice : ℕ) (total_alice_paid : ℕ) :
  num_acorns = 3600 →
  bob_paid = 6000 →
  alice_factor = 9 →
  num_acorns_alice = num_acorns →
  total_alice_paid = alice_factor * bob_paid →
  total_alice_paid / num_acorns_alice = 15 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end alice_price_per_acorn_l607_607457


namespace parabola_vertex_range_l607_607814

theorem parabola_vertex_range (
  a b m : ℝ,
  e : ℝ,
  h1 : a = 1,
  h2 : (1 : ℝ) / a² + (0 : ℝ) / b² = 1,
  h3 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → |y| = x,
  h4 : 0 < b ∧ b < sqrt 3 / 3,
  h5 : sqrt (2 / 3) < e ∧ e < 1,
  h6 : e = sqrt (1 - b^2)
) : 1 < m ∧ m < (3 + sqrt 2) / 4 :=
by
  sorry

end parabola_vertex_range_l607_607814


namespace min_a2005_l607_607922

-- Definitions based on conditions
def strictly_increasing_seq (a : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → a i < a j

def distinct_products_property (a : ℕ → ℕ) : Prop :=
  ∀ i j k, i < j ∧ j < k ∧ i < 2005 ∧ j < 2005 ∧ k < 2005 → a i * a j ≠ a k

-- The theorem to be proven
theorem min_a2005
  (a: ℕ → ℕ)
  (h_inc : strictly_increasing_seq a)
  (h_distinct : distinct_products_property a)
  : a 2004 ≥ 2048 :=
begin
  sorry
end

end min_a2005_l607_607922


namespace circumscribed_quadrilateral_implies_parallel_l607_607917

-- Definitions for the elements of the problem
structure Quadrilateral :=
(A B C D : Point)

def is_convex (quad : Quadrilateral) : Prop := sorry
def non_parallel (p1 p2 : Line) : Prop := sorry
def is_circumscribed (quad : Quadrilateral) : Prop := sorry
def is_parallel (line1 line2 : Line) : Prop := sorry

-- Statement of the problem
theorem circumscribed_quadrilateral_implies_parallel 
  (ABCD : Quadrilateral)
  (E : Point)
  (h_convex : is_convex ABCD)
  (h_non_parallel_BC_AD : non_parallel (Line.mk ABCD.B ABCD.C) (Line.mk ABCD.A ABCD.D))
  (h_E_on_BC : E ∈ segment ABCD.B ABCD.C)
  (h_circumscribed_ABED : is_circumscribed (Quadrilateral.mk ABCD.A ABCD.B E ABCD.D))
  (h_circumscribed_AECD : is_circumscribed (Quadrilateral.mk ABCD.A E ABCD.C ABCD.D)) :
  (∃ F : Point, F ∈ segment ABCD.A ABCD.D ∧ 
    is_circumscribed (Quadrilateral.mk ABCD.A ABCD.B ABCD.C F) ∧
    is_circumscribed (Quadrilateral.mk ABCD.B ABCD.C ABCD.D F)) ↔ 
    is_parallel (Line.mk ABCD.A ABCD.B) (Line.mk ABCD.C ABCD.D) := 
sorry

end circumscribed_quadrilateral_implies_parallel_l607_607917


namespace number_of_registration_methods_l607_607083

theorem number_of_registration_methods
  (students : ℕ) (groups : ℕ) (registration_methods : ℕ)
  (h_students : students = 4) (h_groups : groups = 3) :
  registration_methods = groups ^ students :=
by
  rw [h_students, h_groups]
  exact sorry

end number_of_registration_methods_l607_607083


namespace proof_f2_2002_l607_607510

noncomputable def f : ℝ → ℝ := sorry -- Define f as a non-constant polynomial

axiom h_polynomial: ¬is_constant f -- f is non-constant

axiom h_condition: ∀ x : ℝ, x ≠ 0 → f (x - 1) + f (x + 1) = (f x)^2 / (1001 * x)

theorem proof_f2_2002 : f 2 = 2002 := by
  sorry

end proof_f2_2002_l607_607510


namespace cos_product_identity_l607_607192

theorem cos_product_identity :
  (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 7 → cos (i * π / 15)) = 1 / 128 :=
begin
  sorry
end

end cos_product_identity_l607_607192


namespace rectangle_lines_product_l607_607986

theorem rectangle_lines_product (b : ℤ) :
  (∃ b, (∃ x : ℤ, x = 2 - 5 ∨ x = 2 + 5) ∧
          (b = x) ∧
           (8 - 3 = 5)) →
  (b = -3 ∨ b = 7) →
  (b * if b = -3 then 7 else -3) = -21 :=
by
  intro h1 h2
  obtain ⟨b_exists, hexists, hb, _⟩ := h1
  cases hexists with x hx
  rw [hb] at h2
  cases h2
  .rw h2
  .use (-3)
  .intro _
  .exact rfl
  rw h
  .use 7
  .exact rfl
  sorry

end rectangle_lines_product_l607_607986


namespace range_of_a_l607_607444

-- Definition of sets A and B
def set_A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def set_B (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}

-- Statement that if A ⊆ B, then a > 3
theorem range_of_a (a : ℝ) (h : set_A ⊆ set_B a) : 3 < a :=
by sorry

end range_of_a_l607_607444


namespace angle_A_is_pi_over_3_triangle_area_l607_607469

-- Define the conditions
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Given conditions
def conditions :=
  (a = 5) ∧ (b = 4) ∧ (c ≠ 0) ∧ (A + B + C = π) ∧ 
  (cos (B - C) - 2 * sin B * sin C = -1 / 2)

-- Part 1: Prove that A = π / 3 given the conditions
theorem angle_A_is_pi_over_3 : conditions → A = π / 3 :=
sorry

-- Part 2: Prove the area of ΔABC is 2√3 + √39
theorem triangle_area : conditions → (1 / 2) * b * c * sin A = 2 * sqrt 3 + sqrt 39 :=
sorry

end angle_A_is_pi_over_3_triangle_area_l607_607469


namespace find_angle_B_l607_607889

def angle_A (B : ℝ) : ℝ := B + 21
def angle_C (B : ℝ) : ℝ := B + 36
def is_triangle_sum (A B C : ℝ) : Prop := A + B + C = 180

theorem find_angle_B (B : ℝ) 
  (hA : angle_A B = B + 21) 
  (hC : angle_C B = B + 36) 
  (h_sum : is_triangle_sum (angle_A B) B (angle_C B) ) : B = 41 :=
  sorry

end find_angle_B_l607_607889


namespace infinite_odd_prime_factors_multiple_of_3_l607_607195

def f (m : ℕ) : ℕ := m * (m + 3)

def d (m : ℕ) : ℕ :=
  Nat.card (Nat.factorization (f m)).support \ {2}

theorem infinite_odd_prime_factors_multiple_of_3 :
  ∃∞ m, d m % 3 = 0 :=
sorry

end infinite_odd_prime_factors_multiple_of_3_l607_607195


namespace find_smaller_number_l607_607188

theorem find_smaller_number (a b : ℕ) (h1 : b = 2 * a - 3) (h2 : a + b = 39) : a = 14 :=
by
  -- Sorry to skip the proof
  sorry

end find_smaller_number_l607_607188


namespace quadrilateral_is_symmetric_trapezoid_l607_607525

-- Define the circle and inscribed square
variables {O : Point} {R : ℝ}
variables {A B C D M : Point}

-- Define the relationships and properties given by the problem
def circle (O : Point) (R : ℝ) (P : Point) : Prop := dist O P = R
def perpendicular (P Q R : Point) : Prop := (P - Q) ⬝ (R - Q) = 0
def inscribed_square (O : Point) (R : ℝ) (A B C D : Point) : Prop :=
  circle O R A ∧ circle O R B ∧ circle O R C ∧ circle O R D ∧
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  perpendicular A B O ∧ perpendicular B C O ∧ perpendicular C D O ∧ perpendicular D A O

-- Statement of the problem
theorem quadrilateral_is_symmetric_trapezoid 
  (h_circle_A : circle O R A) 
  (h_circle_B : circle O R B) 
  (h_perpendicular_AD_BC : perpendicular A M D ∧ perpendicular B M C)
  (h_inscribed_square : inscribed_square O R A B C D) 
  : is_isosceles_trapezoid A B C D :=
sorry

end quadrilateral_is_symmetric_trapezoid_l607_607525


namespace solve_for_x_l607_607203

theorem solve_for_x (x : ℝ) : 4^x * 4^x * 4^x = 16^5 → x = 10 / 3 :=
by
  intro h
  sorry

end solve_for_x_l607_607203


namespace maximize_profit_l607_607279

noncomputable def deposit_amount (x : ℝ) := 10000 * x^2

noncomputable def interest_expense (x : ℝ) := x * deposit_amount x

noncomputable def profit (x : ℝ) := 480 * x^2 - interest_expense x

theorem maximize_profit (h : ∀ x, 0 < x ∧ x < 0.048 → profit x ≤ profit 0.032) : ∃ x, 0 < x ∧ x < 0.048 ∧ profit x = profit 0.032 :=
by 
   use [0.032]
   split
   sorry -- prove 0 < 0.032 < 0.048
   sorry -- prove the profit at 0.032 is the max

end maximize_profit_l607_607279


namespace computer_price_decrease_l607_607614

theorem computer_price_decrease 
  (initial_price : ℕ) 
  (decrease_factor : ℚ)
  (years : ℕ) 
  (final_price : ℕ) 
  (h1 : initial_price = 8100)
  (h2 : decrease_factor = 1/3)
  (h3 : years = 6)
  (h4 : final_price = 2400) : 
  initial_price * (1 - decrease_factor) ^ (years / 2) = final_price :=
by
  sorry

end computer_price_decrease_l607_607614


namespace problem_statement_l607_607796

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 4) = -f x
axiom increasing_on_0_2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Theorem to prove
theorem problem_statement : f (-10) < f 40 ∧ f 40 < f 3 :=
by
  sorry

end problem_statement_l607_607796


namespace replaced_person_age_is_40_l607_607210

def average_age_decrease_replacement (T age_of_replaced: ℕ) : Prop :=
  let original_average := T / 10
  let new_total_age := T - age_of_replaced + 10
  let new_average := new_total_age / 10
  original_average - 3 = new_average

theorem replaced_person_age_is_40 (T : ℕ) (h : average_age_decrease_replacement T 40) : Prop :=
  ∀ age_of_replaced, age_of_replaced = 40 → average_age_decrease_replacement T age_of_replaced

-- To actually formalize the proof, you can use the following structure:
-- proof by calculation omitted
lemma replaced_person_age_is_40_proof (T : ℕ) (h : average_age_decrease_replacement T 40) : 
  replaced_person_age_is_40 T h :=
by
  sorry

end replaced_person_age_is_40_l607_607210


namespace xiao_yun_age_l607_607620

theorem xiao_yun_age (x : ℕ) (h1 : ∀ x, x + 25 = Xiao_Yun_fathers_current_age)
                     (h2 : ∀ x, Xiao_Yun_fathers_age_in_5_years = 2 * (x+5) - 10) :
  x = 30 := by
  sorry

end xiao_yun_age_l607_607620


namespace number_of_outfits_l607_607092

-- Define the number of shirts, pants, and jacket options.
def shirts : Nat := 8
def pants : Nat := 5
def jackets : Nat := 3

-- The theorem statement for the total number of outfits.
theorem number_of_outfits : shirts * pants * jackets = 120 := 
by
  sorry

end number_of_outfits_l607_607092


namespace largest_integer_k_l607_607730

def S : ℕ → ℝ
| 1 := 3
| (n + 1) := 3 ^ S n

def C := (S 5) ^ 3

def D := (S 5) ^ C

theorem largest_integer_k : ∃ k : ℕ, ∀ m : ℕ, m ≤ k → ∀ x : ℝ, x = ((nat.rec_on m (λ x, S 5) (λ _ f x, log 3 x)) D) → x > 0 ∧ k = 6 :=
sorry

end largest_integer_k_l607_607730


namespace length_of_train_l607_607679

theorem length_of_train (speed_kmph : ℕ) (time_sec : ℕ) (h_speed : speed_kmph = 72) (h_time : time_sec = 5) : 
  let speed_mps := (speed_kmph * 1000) / 3600 in
  speed_mps * time_sec = 100 := 
by
  unfold speed_mps
  rw [h_speed, h_time]
  norm_num

end length_of_train_l607_607679


namespace sqrt_condition_l607_607464

theorem sqrt_condition (x : ℝ) (h : 2 - x ≥ 0) (hx : x ∈ {4, real.pi, -1, 3}) : x = -1 :=
by {
  simp [set.mem_insert_iff, set.mem_singleton_iff] at hx,
  cases hx with h4 hrest,
  { exfalso, linarith },
  cases hrest with hpi hrest,
  { exfalso, linarith [real.pi_pos] },
  cases hrest with hm1 h3,
  { exact hm1 },
  { exfalso, linarith },
}

end sqrt_condition_l607_607464


namespace chord_length_l607_607288

theorem chord_length (radius : ℝ) (distance_to_chord : ℝ) (EF_length : ℝ) 
  (h_radius : radius = 5) (h_distance_to_chord : distance_to_chord = 4) 
  (h_chord_length : EF_length = 2 * sqrt (radius^2 - distance_to_chord^2)) : 
  EF_length = 6 :=
by {
  have h_radius_nonneg : radius ≥ 0 := by linarith [h_radius],
  have h_distance_nonneg : distance_to_chord ≥ 0 := by linarith [h_distance_to_chord],
  exact h_chord_length
}

end chord_length_l607_607288


namespace ming_wins_inequality_l607_607619

variables (x : ℕ)

def remaining_distance (x : ℕ) : ℕ := 10000 - 200 * x
def ming_remaining_distance (x : ℕ) : ℕ := remaining_distance x - 200

-- Ensure that Xiao Ming's winning inequality holds:
theorem ming_wins_inequality (h1 : 0 < x) :
  (ming_remaining_distance x) / 250 > (remaining_distance x) / 300 :=
sorry

end ming_wins_inequality_l607_607619


namespace volume_tetrahedron_4D_eq_l607_607174

noncomputable def volume_tetrahedron_4D (a b c d : ℝ) (angle_BAC angle_CAD : ℝ) : ℝ := 
  (1 / 6) * a * b * c * Real.sqrt (1 - (Real.cos angle_BAC)^2 - (Real.cos angle_BAC)^2 - (Real.cos angle_CAD)^2)

theorem volume_tetrahedron_4D_eq :
  let O := (0, 0, 0, 0 : ℝ×ℝ×ℝ×ℝ) in
  let A := (√(4√144), 0, 0, 0) in
  let B := (0, √(4√144), 0, 0) in
  let C := (0, 0, √(4√144), 0) in
  let D := (0, 0, 0, √(4√144)) in
  let angle_BAC := Real.pi / 4 in
  let angle_CAD := Real.pi / 6 in
  volume_tetrahedron_4D (√(4√144)) (√(4√144)) (√(4√144)) (√(4√144)) angle_BAC angle_CAD = 8 * Real.sqrt 6 / 3 :=
sorry

end volume_tetrahedron_4D_eq_l607_607174


namespace factorize_expr_l607_607744

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l607_607744


namespace matrix_power_50_l607_607503

-- Defining the matrix A.
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 1], 
    ![-12, -3]]

-- Statement of the theorem
theorem matrix_power_50 :
  A ^ 50 = ![![301, 50], 
               ![-900, -301]] :=
by
  sorry

end matrix_power_50_l607_607503


namespace triangle_max_distance_l607_607754

/-- Given a triangle ABC, the sum of the distances from A and B to the line through C is maximized
when the line through C is considered, resulting in the maximum being either the distance AB
or the distance CD where D is the projection of C onto line AB. -/
theorem triangle_max_distance (A B C : EuclideanSpace ℝ 2) (D : EuclideanSpace ℝ 2)
    [is_projection D A B C] :
  (∃ ℓ : LineOn ℝ, passes_through C ℓ ∧ 
    (sum_distances_to_line A B ℓ = max (dist A B) (dist C D)) :=
sorry

end triangle_max_distance_l607_607754


namespace missing_number_l607_607847

theorem missing_number 
  (a : ℕ) (b : ℕ) (x : ℕ)
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * x * b) 
  (h3 : b = 147) : 
  x = 3 :=
sorry

end missing_number_l607_607847


namespace polygonal_chain_length_bound_maximum_polygon_area_l607_607266

-- Definition for Problem a)
theorem polygonal_chain_length_bound 
  {A B : Point} (P : ℕ → Point) (distance_AB : dist A B = 1) 
  (convex : convex_polygonal_chain A P B)
  (alpha : ℝ) (angle_bound : α < real.pi) : 
  length_of_chain A P B ≤ 1 / real.cos (α / 2) :=
sorry

-- Definition for Problem b)
theorem maximum_polygon_area 
  (a : ℝ) (convex_polygon : ∀ (P : list Point), polygon_has_side P a ∧ 
    sum_external_angles P = 120°) : 
  ∃ (P : list Point), area_of_polygon P = a^2 * real.sqrt 3 / 4 :=
sorry

end polygonal_chain_length_bound_maximum_polygon_area_l607_607266


namespace max_elements_in_S_l607_607555

theorem max_elements_in_S : ∀ (S : Finset ℕ), 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → 
    (∃ c ∈ S, Nat.Coprime c a ∧ Nat.Coprime c b) ∧
    (∃ d ∈ S, ∃ x y : ℕ, x ∣ a ∧ x ∣ b ∧ x ∣ d ∧ y ∣ a ∧ y ∣ b ∧ y ∣ d)) →
  S.card ≤ 72 :=
by sorry

end max_elements_in_S_l607_607555


namespace A_alone_days_l607_607653

variable (x : ℝ) -- Number of days A takes to do the work alone
variable (B_rate : ℝ := 1 / 12) -- Work rate of B
variable (Together_rate : ℝ := 1 / 4) -- Combined work rate of A and B

theorem A_alone_days :
  (1 / x + B_rate = Together_rate) → (x = 6) := by
  intro h
  sorry

end A_alone_days_l607_607653


namespace distinct_solution_difference_l607_607171

theorem distinct_solution_difference :
  let f := λ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3 
  ∃ r s : ℝ, f r ∧ f s ∧ r ≠ s ∧ r > s ∧ (r - s = 3) :=
begin
  -- Placeholder for proof
  sorry
end

end distinct_solution_difference_l607_607171


namespace shirts_per_day_l607_607952

theorem shirts_per_day (total_shirts : ℕ) (days : ℕ) (shirts_each_day : ℕ) :
  total_shirts = 11 → days = 5 → shirts_each_day = 2 :=
begin
  assume h1 : total_shirts = 11,
  assume h2 : days = 5,
  have h3: total_shirts - 1 = days * shirts_each_day,
  sorry
end

end shirts_per_day_l607_607952


namespace value_of_m_unique_tangent_perpendicular_l607_607848

def curve (x : ℝ) (m : ℝ) := (1/3) * x^3 + x^2 + m * x

def derivative (x : ℝ) (m : ℝ) := x^2 + 2 * x + m

noncomputable def condition (m : ℝ) := ∃ a : ℝ, (a^2 + 2 * a + m = 1)

theorem value_of_m_unique_tangent_perpendicular
  (h1 : ∀ m : ℝ, ∃! t : ℝ, t = m -> condition m) :
  m = 2 := 
sorry

end value_of_m_unique_tangent_perpendicular_l607_607848


namespace product_of_roots_l607_607764

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem product_of_roots (x : ℝ) : 
  (4 * f (3 - x) - f x = 3 * x^2 - 4 * x - 3) →
  (Exists (λ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5)) :=
by
  sorry

end product_of_roots_l607_607764


namespace correct_statement_l607_607015

def is_ellipse (m : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (m ∈ (1/2, 2)) ∧ C = λ x y, a * x^2 + b * y^2 = 1

theorem correct_statement (m : ℝ) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, C x y ↔ (2 - m) * x^2 + (m + 1) * y^2 = 1) →
  is_ellipse m C :=
by
  intro hC
  sorry

end correct_statement_l607_607015


namespace investment_return_l607_607648

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l607_607648


namespace find_principal_l607_607230

def comp_interest (P : ℕ) (r : ℕ) (n : ℕ) : ℕ := P * (1 + r / 100)^n

def simp_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℕ := P * r * t / 100

theorem find_principal :
  let P₁ := 8000
  let r₁ := 15
  let n₁ := 2
  let r₂ := 8
  let t₂ := 5
  let CI := comp_interest P₁ r₁ n₁ - P₁
  let SI := CI / 2
  let P₂ := simp_interest P x r₂ t₂ = SI
  in P₂ = 3225 := 
sorry

end find_principal_l607_607230


namespace sum_of_arithmetic_sequence_l607_607023

theorem sum_of_arithmetic_sequence :
  (∑ k in {1, 3, 5, 7, 9}, (λ n : ℕ, n^2 + n - (n - 1)^2 - (n - 1)) k) = 50 :=
by sorry

end sum_of_arithmetic_sequence_l607_607023


namespace cosine_of_angle_between_vectors_l607_607447

variable {ℝ : Type*} [IsROrC ℝ]
variable {E : Type*} [InnerProductSpace ℝ E] [NormedSpace ℝ E] [CompleteSpace E]

theorem cosine_of_angle_between_vectors
  {a b : E} (ha : ∥a∥ = 1) (hb : ∥b∥ = 2 * Real.sqrt 2) (hab : ∥a - b∥ = 2) :
  Real.cos ((inner_product_space.angle ⟨a, ha⟩ ⟨b, hb⟩) : ℝ) = 5 * Real.sqrt 2 / 8 :=
by
  sorry

end cosine_of_angle_between_vectors_l607_607447


namespace inverse_function_value_l607_607388

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9 + 3 * real.sin(x) + 2 * real.exp(x)

theorem inverse_function_value : ∃ y : ℝ, f(y) = -3.5 :=
by 
  sorry

end inverse_function_value_l607_607388


namespace evaluate_expression_l607_607375

theorem evaluate_expression : 3 - (-3)^(3 - (-3) + 1) = 2190 := by
  sorry

end evaluate_expression_l607_607375


namespace meet_at_centroid_l607_607837

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid_l607_607837


namespace tank_capacity_percentage_l607_607564

-- Define the data for tank M
def tankM_height : ℝ := 10
def tankM_circumference : ℝ := 8

-- Define the data for tank B
def tankB_height : ℝ := 8
def tankB_circumference : ℝ := 10

-- Define the volume of a right circular cylinder
noncomputable def volume (h : ℝ) (c : ℝ) : ℝ :=
  let r := c / (2 * Real.pi) in
  Real.pi * r^2 * h

-- Define the volumes for tank M and tank B
noncomputable def volumeM : ℝ := volume tankM_height tankM_circumference
noncomputable def volumeB : ℝ := volume tankB_height tankB_circumference

-- Define the function to calculate the capacity percentage
noncomputable def percentage_capacity_m_b : ℝ := (volumeM / volumeB) * 100

-- Prove that the capacity of tank M as a percentage of the capacity of tank B is 80%.
theorem tank_capacity_percentage : percentage_capacity_m_b = 80 := 
  sorry

end tank_capacity_percentage_l607_607564


namespace log_a_b_squared_l607_607801

/-- Given the roots of a quadratic polynomial 2x^2 - 4x + 1 are log a and log b, prove that (log (a/b))^2 = 2. -/
theorem log_a_b_squared (a b : ℝ) (h₁: (∃ x : ℝ, x^2 - (4/2)x + 1/2 = 0 ∧ ((∃ log_a : ℝ, log_a = Real.log a) ∧ (∃ log_b : ℝ, log_b = Real.log b) ∧ x = log_a ∨ x = log_b))) :
  (Real.log (a / b))^2 = 2 :=
sorry

end log_a_b_squared_l607_607801


namespace distinct_solutions_diff_l607_607157

theorem distinct_solutions_diff {r s : ℝ} (h_eq_r : (6 * r - 18) / (r^2 + 3 * r - 18) = r + 3)
  (h_eq_s : (6 * s - 18) / (s^2 + 3 * s - 18) = s + 3)
  (h_distinct : r ≠ s) (h_r_gt_s : r > s) : r - s = 11 := 
sorry

end distinct_solutions_diff_l607_607157


namespace infinitely_many_natural_numbers_satisfy_equation_l607_607134

noncomputable def α := (1989 + Real.sqrt (1989^2 + 4)) / 2

theorem infinitely_many_natural_numbers_satisfy_equation :
  ∃ᶠ (n : ℕ) in Filter.atTop, 
    let k := ⌊α * n⌋ in 
    ⌊α * n + 1989 * α * k⌋ = 1989 * n + (1989^2 + 1) * k :=
by
  sorry

end infinitely_many_natural_numbers_satisfy_equation_l607_607134


namespace problem_statement_l607_607165

noncomputable def distinct_solutions (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ r s : ℝ, r ≠ s ∧ r > s ∧ f r = 0 ∧ f s = 0 ∧ a = r - s

theorem problem_statement : distinct_solutions (λ x : ℝ, x^2 + 9 * x + 12) 1 :=
by
  sorry

end problem_statement_l607_607165


namespace no_three_by_three_red_prob_l607_607373

theorem no_three_by_three_red_prob : 
  ∃ (m n : ℕ), 
  Nat.gcd m n = 1 ∧ 
  m / n = 340 / 341 ∧ 
  m + n = 681 :=
by
  sorry

end no_three_by_three_red_prob_l607_607373


namespace adjacent_probability_l607_607652

def box := {1, 2, 3, 4, 5}

def total_outcomes := { (a, b) | a ∈ box ∧ b ∈ box ∧ a ≠ b }.card

def adjacent_events := { (a, b) | a ∈ box ∧ b ∈ box ∧ |a - b| = 1 }.card

theorem adjacent_probability : 
  total_outcomes = 10 ∧ adjacent_events = 4 → 
  (adjacent_events / total_outcomes) = 2 / 5 := 
  by
    sorry

end adjacent_probability_l607_607652


namespace simplify_trigonometric_expression_l607_607954

noncomputable def simplification_problem : Prop :=
  let t20 := Real.tan (20 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  let t60 := Real.tan (60 * Real.pi / 180)
  let t70 := Real.tan (70 * Real.pi / 180)
  let c40 := Real.cos (40 * Real.pi / 180)
  S = (t20 + t30 + t60 + t70) / c40

theorem simplify_trigonometric_expression : simplification_problem := by
  sorry

end simplify_trigonometric_expression_l607_607954


namespace drink_all_tea_l607_607540

theorem drink_all_tea (cups : Fin 30 → Prop) (red blue : Fin 30 → Prop)
  (h₀ : ∀ n, cups n ↔ (red n ↔ ¬ blue n))
  (h₁ : ∃ a b, a ≠ b ∧ red a ∧ blue b)
  (h₂ : ∀ n, red n → red (n + 2))
  (h₃ : ∀ n, blue n → blue (n + 2)) :
  ∃ sequence : ℕ → Fin 30, (∀ n, cups (sequence n)) ∧ (sequence 0 ≠ sequence 1) 
  ∧ (∀ n, cups (sequence (n+1))) :=
by
  sorry

end drink_all_tea_l607_607540


namespace total_people_in_class_l607_607472

def likes_both (n : ℕ) := n = 5
def likes_only_baseball (n : ℕ) := n = 2
def likes_only_football (n : ℕ) := n = 3
def likes_neither (n : ℕ) := n = 6

theorem total_people_in_class
  (h1 : likes_both n1)
  (h2 : likes_only_baseball n2)
  (h3 : likes_only_football n3)
  (h4 : likes_neither n4) :
  n1 + n2 + n3 + n4 = 16 :=
by 
  sorry

end total_people_in_class_l607_607472


namespace negation_exists_l607_607219

theorem negation_exists (x : ℝ) (h : x ≥ 0) : (¬ (∀ x : ℝ, (x ≥ 0) → (2^x > x^2))) ↔ (∃ x₀ : ℝ, (x₀ ≥ 0) ∧ (2 ^ x₀ ≤ x₀^2)) := by
  sorry

end negation_exists_l607_607219


namespace part1_part2_l607_607058

def f (a x : ℝ) : ℝ := a * x ^ 2 - (2 * a + 1) * x - 1

theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≤ -3 / 4) ↔ (a ∈ set.Icc (-1 : ℝ) (-1 / 4)) :=
sorry

theorem part2 (a : ℝ) :
  (a ≤ 0 ∧ ∀ x : ℝ, 0 < x → x * f a x ≤ 1) ↔ (a ∈ set.Icc (-3 : ℝ) 0) :=
sorry

end part1_part2_l607_607058


namespace sampling_is_systematic_l607_607856

-- Define the conditions as a structure
structure SchoolSampling :=
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (student_id_to_keep : ℕ)

-- Define the instance representing the problem
def specificSampling : SchoolSampling :=
  { num_classes := 30,
    students_per_class := 56,
    student_id_to_keep := 16 }

-- Define the theorem to prove the sampling method is systematic sampling
theorem sampling_is_systematic (s : SchoolSampling) 
  (h1 : s.num_classes = 30)
  (h2 : s.students_per_class = 56)
  (h3 : s.student_id_to_keep = 16) : 
  (sampling_method s = "Systematic Sampling") :=
sorry

end sampling_is_systematic_l607_607856


namespace find_omega_min_value_after_shift_l607_607057

noncomputable section

def f (x : ℝ) (ω : ℝ) (φ : ℝ) := sqrt 2 * cos (ω * x + φ)

theorem find_omega 
  (ω : ℝ) (φ : ℝ)
  (hω_pos : ω > 0)
  (hφ_bound : |φ| ≤ π / 2)
  (x₁ x₂ : ℝ)
  (hf_condition : f x₁ ω φ * f x₂ ω φ = -2)
  (hx_condition : abs (x₁ - x₂) = π / 2) :
  ω = 2 :=
sorry

theorem min_value_after_shift 
  (φ : ℝ)
  (hφ_bound : |φ| ≤ π / 2)
  (hf_sym : ∀ x : ℝ, √2 * cos (2 * (x + π / 6) + φ) = √2 * cos (2 * (7 * π / 12 - x) + φ))
  (a b : ℝ)
  (ha : a = π / 6)
  (hb : b = π / 3) :
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ (∀ y : ℝ, a ≤ y ∧ y ≤ b → f y 2 φ ≥ f x 2 φ) ∧ f x 2 φ = -sqrt 6 / 2 :=
sorry

end find_omega_min_value_after_shift_l607_607057


namespace find_measure_of_A_find_area_of_triangle_l607_607468

-- Define the conditions
def triangle_ABC (A B C : ℝ) :=
  A + B + C = π

def given_equation (A : ℝ) :=
  sin A + sqrt 3 * cos A = 2

def solve_for_angle (A : ℝ) :=
  A = π / 6

def area_of_triangle (A B C a b c : ℝ) :=
  A = π / 6 ∧ B = π / 4 ∧ a = 2 ∧ b = a * sin B / sin A ∧
  0 < C ∧ C = π - A - B ∧ (1/2 * a * b * sin C) = 1 + sqrt 3

-- Rewrite the proof problems
theorem find_measure_of_A (A : ℝ) : given_equation A → solve_for_angle A :=
by sorry

theorem find_area_of_triangle (A B C a b : ℝ) :
  triangle_ABC A B C ∧ given_equation A ∧ a = 2 ∧ B = π / 4 ∧ A = π / 6 → area_of_triangle A B C a b (sqrt ((a^2) + (b^2) - 2 * a * b * cos C)) :=
by sorry

end find_measure_of_A_find_area_of_triangle_l607_607468


namespace solve_for_x_l607_607959

-- Definitions based on conditions
def base_relation (x : ℝ) : Prop :=
  16 ^ x * 16 ^ x * 16 ^ x * 16 ^ x = 256 ^ 4

def relation_256_eq_16_square : Prop :=
  256 = 16 ^ 2

-- Theorem statement based on the mathematically equivalent proof problem
theorem solve_for_x (x : ℝ) (h1 : base_relation x) (h2 : relation_256_eq_16_square) : x = 2 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l607_607959


namespace women_reseating_l607_607207

def T : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 4
| 3     := 7
| (n+3) := T (n+2) + T (n+1) + T n

theorem women_reseating : T 10 = 480 :=
sorry

end women_reseating_l607_607207


namespace problem_1_problem_2i_problem_2ii_l607_607059

-- Conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x - a

def f_n (n : ℕ) (x : ℝ) : ℝ := n * x^3 + 2 * x - n

def b_n (x_n x_n1 : ℝ) : ℝ := (1 - x_n) * (1 - x_n1)

def S (n : ℕ) (x : ℕ → ℝ) : ℝ := (Finset.range n).sum (λ i, b_n (x i) (x (i + 1)))

-- Proof statements
theorem problem_1 (a : ℝ) : 
  (a ≥ 0 → ∀ x : ℝ, 3 * a * x^2 + 2 > 0) ∧ 
  (a < 0 → ∀ x : ℝ, x > real.sqrt (2 / (3 * a)) ∨ x < -real.sqrt (2 / (3 * a)) → 3 * a * x^2 + 2 > 0) :=
sorry

theorem problem_2i (n : ℕ) (n_ge_two : 2 ≤ n) :
  ∃! x : ℝ, f_n n x = 0 ∧ x ∈ set.Ioo (n / (n + 1)) 1 :=
sorry

theorem problem_2ii (n : ℕ) (n_ge_two : 2 ≤ n) (x : ℕ → ℝ) : 
  (∀ k : ℕ, k < n → x k ∈ set.Ioo (k / (k + 1)) 1) →
  S n x < 1 :=
sorry

end problem_1_problem_2i_problem_2ii_l607_607059


namespace sum_of_distances_l607_607289

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_2 = d_1 + 5) (h2 : d_1 + d_2 = 13) :
  d_1 + d_2 = 13 :=
by sorry

end sum_of_distances_l607_607289


namespace second_machine_finishes_in_10_minutes_l607_607603

-- Definitions for the conditions:
def time_to_clear_by_first_machine (t : ℝ) : Prop := t = 1
def time_to_clear_by_second_machine (t : ℝ) : Prop := t = 3 / 4
def time_first_machine_works (t : ℝ) : Prop := t = 1 / 3
def remaining_time (t : ℝ) : Prop := t = 1 / 6

-- Theorem statement:
theorem second_machine_finishes_in_10_minutes (t₁ t₂ t₃ t₄ : ℝ) 
  (h₁ : time_to_clear_by_first_machine t₁) 
  (h₂ : time_to_clear_by_second_machine t₂) 
  (h₃ : time_first_machine_works t₃) 
  (h₄ : remaining_time t₄) 
  : t₄ = 1 / 6 → t₄ * 60 = 10 := 
by
  -- here we can provide the proof steps, but the task does not require the proof
  sorry

end second_machine_finishes_in_10_minutes_l607_607603


namespace speed_ratio_male_to_female_summit_distance_correct_point_B_distance_less_than_360_l607_607974

-- Condition definitions:
def time_ratio_male_to_female := (2:ℚ) / (3:ℚ)
def distance_to_summit := 600

-- Statement and proof goals:
theorem speed_ratio_male_to_female : ∃ (r:ℚ), r = 3 / 2 :=
by {
  existsi (3:ℚ) / (2:ℚ),
  refl
}

noncomputable def distance_foot_to_summit := 
let x := 1800 in x

theorem summit_distance_correct : ∃ (x:ℚ), x - distance_to_summit = 600 ∧ x = distance_foot_to_summit :=
by {
  existsi (1800:ℚ),
  split;
  { norm_num }
}

theorem point_B_distance_less_than_360 : ∀ (k:ℚ) (a:ℚ), (a / (3*k)) < (600 - a) / (2*k) → a < 360 :=
by {
  intros k a h,
  linarith only [h]
}

end speed_ratio_male_to_female_summit_distance_correct_point_B_distance_less_than_360_l607_607974


namespace least_D_for_n_diverse_and_n_reachable_l607_607770

def n_diverse (n : ℕ) (coins : List ℕ) : Prop :=
  ∀x, list.count x coins ≤ n

def n_reachable (n S : ℕ) (coins : List ℕ) : Prop :=
  ∃ sublist, list.sublist sublist coins ∧ list.sum sublist = S

theorem least_D_for_n_diverse_and_n_reachable:
  ∀ (n k : ℕ), n ≥ k → k ≥ 2 →
  ∃ (D : ℕ), (∀ (coins : List ℕ), 
    n_diverse n coins → list.length coins = D → 
    ∃ (reachable_S_vals : List ℕ), list.length reachable_S_vals ≥ k ∧ 
    ∀ S ∈ reachable_S_vals, n_reachable n S coins) ∧
  (∀ (d : ℕ), d < D → 
    ¬(∀ (coins : List ℕ), 
      n_diverse n coins → list.length coins = d → 
      ∃ (reachable_S_vals : List ℕ), list.length reachable_S_vals ≥ k ∧ 
      ∀ S ∈ reachable_S_vals, n_reachable n S coins)) :=
by
  sorry

end least_D_for_n_diverse_and_n_reachable_l607_607770


namespace drink_all_tea_l607_607542

noncomputable def can_drink_all_tea : Prop :=
  ∀ (initial_hare_cup : ℕ) (initial_dormouse_cup : ℕ), 
  0 ≤ initial_hare_cup ∧ initial_hare_cup < 30 ∧ 
  0 ≤ initial_dormouse_cup ∧ initial_dormouse_cup < 30 ∧ 
  initial_hare_cup ≠ initial_dormouse_cup →
  ∃ (rotation : ℕ → ℕ), 
  (∀ (n : ℕ), 
    (rotation n) % 30 = (initial_hare_cup + n) % 30 ∧ 
    (rotation n + x) % 30 ≠ initial_hare_cup % 30 ∧ 
    (∀ m, m < n → 
      (rotation (m+1)) % 30 ≠ (rotation m) % 30)) ∧ 
    set.range rotation = {0,1,2,...,29} 

theorem drink_all_tea : can_drink_all_tea :=
  by sorry

end drink_all_tea_l607_607542


namespace solution_to_fractional_equation_l607_607999

theorem solution_to_fractional_equation (x : ℝ) (h₁ : 2 / (x - 3) = 1 / x) (h₂ : x ≠ 3) (h₃ : x ≠ 0) : x = -3 :=
sorry

end solution_to_fractional_equation_l607_607999


namespace three_digit_numbers_with_tens_at_least_twice_units_l607_607839

theorem three_digit_numbers_with_tens_at_least_twice_units :
  ∃ (n : ℕ), (n = 270) ∧
  (∀ (h t u : ℕ), 
    (100 ≤ h * 100 + t * 10 + u) ∧ (h * 100 + t * 10 + u < 1000) ∧ 
    (t ≥ 2 * u) → 
    (h ∈ {1,2,3,4,5,6,7,8,9})) := sorry

end three_digit_numbers_with_tens_at_least_twice_units_l607_607839


namespace semi_gloss_saves_most_l607_607705

-- Define the paint problem parameters
def pints_per_door : ℕ := 1
def doors_to_paint : ℕ := 8
def pints_per_window : ℕ := 0.5
def windows_to_paint : ℕ := 4
def high_gloss_pint_price : ℕ := 800 -- prices are multiplied by 100 to avoid decimals
def semi_gloss_pint_price : ℕ := 650 -- prices are multiplied by 100 to avoid decimals
def matte_finish_pint_price : ℕ := 500 -- prices are multiplied by 100 to avoid decimals
def high_gloss_gallon_price : ℕ := 5500 -- prices are multiplied by 100 to avoid decimals
def semi_gloss_gallon_price : ℕ := 4800 -- prices are multiplied by 100 to avoid decimals
def matte_finish_gallon_price : ℕ := 4000 -- prices are multiplied by 100 to avoid decimals
def pints_per_gallon : ℕ := 8

-- Calculate total pints required
def total_pints_needed : ℕ := (doors_to_paint * pints_per_door) + (windows_to_paint * pints_per_window)

-- Calculate costs in pints and gallons for each type
def high_gloss_pints_cost : ℕ := total_pints_needed * high_gloss_pint_price
def semi_gloss_pints_cost : ℕ := total_pints_needed * semi_gloss_pint_price
def matte_finish_pints_cost : ℕ := total_pints_needed * matte_finish_pint_price

def high_gloss_gallons_cost : ℕ := ((2 : ℕ) * high_gloss_gallon_price)
def semi_gloss_gallons_cost : ℕ := ((2 : ℕ) * semi_gloss_gallon_price)
def matte_finish_gallons_cost : ℕ := ((2 : ℕ) * matte_finish_gallon_price)

-- Calculate savings for each type of paint
def high_gloss_savings : ℕ := high_gloss_gallons_cost - high_gloss_pints_cost
def semi_gloss_savings : ℕ := semi_gloss_gallons_cost - semi_gloss_pints_cost
def matte_finish_savings : ℕ := matte_finish_gallons_cost - matte_finish_pints_cost

-- Prove that semi-gloss saves the most money
theorem semi_gloss_saves_most : semi_gloss_savings = 3100 ∧ semi_gloss_savings > high_gloss_savings ∧ semi_gloss_savings > matte_finish_savings :=
by sorry

end semi_gloss_saves_most_l607_607705


namespace binomial_seven_four_l607_607719

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l607_607719


namespace number_of_sets_of_segments_l607_607515

theorem number_of_sets_of_segments : 
  let points := {A, B, C, D, E} : Finset Point,
      pairs := points.pairs,
      conditions (segments : Finset (Point × Point)) :=
        (∀ (p1 p2 : Point), ∃ path : List (Point × Point), isConnected segments path p1 p2) ∧
        ∃ (S T : Finset Point), S ∪ T = points ∧ S ∩ T = ∅ ∧
                               (∀ ⦃x y : Point⦄, (x, y) ∈ segments → (x ∈ S ∧ y ∈ T) ∨ (x ∈ T ∧ y ∈ S)),
                               (∃! segments : Finset (Point × Point), conditions segments)
  in (∑ s in ((Finset.filter conditions pairs.powerset)), 1) = 195 :=
sorry

end number_of_sets_of_segments_l607_607515


namespace quadrant_of_point_l607_607075

theorem quadrant_of_point (α : ℝ) (h1 : - (π / 2) < α) (h2 : α < 0) : 
  0 < cot α ∧ cos α > 0 := 
by sorry

end quadrant_of_point_l607_607075


namespace handshake_problem_l607_607321

theorem handshake_problem (x y : ℕ) 
  (H : (x * (x - 1)) / 2 + y = 159) : 
  x = 18 ∧ y = 6 := 
sorry

end handshake_problem_l607_607321


namespace ratio_a_b_l607_607358

variables {x y a b : ℝ}

theorem ratio_a_b (h1 : 8 * x - 6 * y = a)
                  (h2 : 12 * y - 18 * x = b)
                  (hx : x ≠ 0)
                  (hy : y ≠ 0)
                  (hb : b ≠ 0) :
  a / b = -4 / 9 :=
sorry

end ratio_a_b_l607_607358


namespace curry_probability_l607_607182

def P_make : ℚ := 2 / 5
def P_miss : ℚ := 3 / 5

theorem curry_probability :
  let prob_two_consecutive_makes := 1 - ((3/5)^4 + 4 * (2/5) * (3/5)^3 + 3 * (2/5)^2 * (3/5)^2) in
    prob_two_consecutive_makes = 44 / 125 :=
by
  sorry

end curry_probability_l607_607182


namespace reconstruct_parallelogram_l607_607199

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}

def scalar_mult (k : ℝ) (P : Point) : Point :=
{x := k * P.x, y := k * P.y}

def subtract (P Q : Point) : Point :=
{x := P.x - Q.x, y := P.y - Q.y}

def add_points (P Q : Point) : Point :=
{x := P.x + Q.x, y := P.y + Q.y}

def R (M N P : Point) : Point :=
let Q := midpoint M N in subtract (scalar_mult 2 Q) P

-- The theorem will assert that given points M, N, and P, 
-- if M and N are midpoints of opposite sides of a parallelogram,
-- then M, N, P, and R (the point computed above) form the vertices of a parallelogram.
theorem reconstruct_parallelogram (M N P : Point) (H : M ≠ N ∧ M ≠ P ∧ N ≠ P) :
  ∃ R, ∀ Q, Q = midpoint M N → R = add_points (scalar_mult 2 Q) (scalar_mult (-1) P) → 
          parallelogram M N P R :=
sorry

end reconstruct_parallelogram_l607_607199


namespace sum_first_n_terms_l607_607004

theorem sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 2) →
  (∀ n, a (n + 1) - a n = 2^n) →
  (∀ n, S n = ∑ i in finset.range n, a (i + 1)) →
  (∀ n, S n = 2^{n+1} - 2) :=
by
  intros h1 h2 h3
  sorry

end sum_first_n_terms_l607_607004


namespace pens_sold_promotion_l607_607996

theorem pens_sold_promotion (pen_profit bear_cost total_profit : ℕ) (bundle_pens bundle_bears : ℕ)
  (h1 : pen_profit = 9)
  (h2 : bear_cost = 2)
  (h3 : bundle_pens = 4)
  (h4 : bundle_bears = 1)
  (h5 : total_profit = 1922) :
  let net_profit_per_bundle := (bundle_pens * pen_profit) - bear_cost,
      bundles_sold := total_profit / net_profit_per_bundle,
      remaining_profit := total_profit % net_profit_per_bundle,
      total_pens_from_bundles := bundles_sold * bundle_pens,
      additional_pens := remaining_profit / pen_profit,
      total_pens := total_pens_from_bundles + additional_pens in
  total_pens = 226 :=
by
  sorry

end pens_sold_promotion_l607_607996


namespace even_function_a_value_l607_607088

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x - 1) = ((-x)^2 + a * (-x) - 1)) ↔ a = 0 :=
by
  sorry

end even_function_a_value_l607_607088


namespace card_visibility_l607_607547

noncomputable def visibility_problem (n : ℕ) : Prop :=
  ∃ arrangement : { l : list (fin (2 * n)) // l.nodup },
  ∀ i : fin (2 * n),
  i ∈ arrangement.val

theorem card_visibility (n : ℕ) : visibility_problem n :=
sorry

end card_visibility_l607_607547


namespace problem1_l607_607643

theorem problem1 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2 * α) + Real.cos α ^ 2 = 3 / 2 := 
sorry

end problem1_l607_607643


namespace committee_of_8_choose_4_is_70_l607_607296

theorem committee_of_8_choose_4_is_70 :
  nat.choose 8 4 = 70 :=
sorry

end committee_of_8_choose_4_is_70_l607_607296


namespace max_f_value_sides_difference_l607_607433

noncomputable def f (x : ℝ) : ℝ := 2 * sin(x + π / 4) ^ 2 - sqrt 3 * cos (2 * x)

theorem max_f_value :
  let α := (5 * π) / 12
  x ∈ set.Icc (π / 4) (π / 2) →
  (∀ y ∈ set.Icc (π / 4) (π / 2), f y ≤ f α) ∧ f α = 3 :=
by sorry

variables {A B C a b c : ℝ}

theorem sides_difference :
  let α := (5 * π) / 12 in
  let A := α - (π / 12) in
  sin B * sin C = (sin A) ^ 2 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) →
  b - c = 0 :=
by sorry

end max_f_value_sides_difference_l607_607433


namespace find_a_b_and_min_val_l607_607437

-- Given function f(x) = x^3 - x^2 + ax + b and tangent line condition at (0, f(0))
variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + b

-- The conditions
theorem find_a_b_and_min_val 
  (h₁ : f 0 = 1) 
  (h₂ : ∂ (λ x, x^3 - x^2 + a * x + b) / ∂ x | (0 : ℝ) = -1) : 
  a = -1 ∧ b = 1 ∧ ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), 
  (x = -2 → f x = -9) ∧ (x = 1 → f x ≥ f(ℂ.find_val -2 0 (λ x, -x^2 -x + 1))) := 
by 
  sorry

end find_a_b_and_min_val_l607_607437


namespace total_beads_correct_l607_607927

-- Definitions of the problem conditions
def blue_beads : ℕ := 5
def red_beads : ℕ := 2 * blue_beads
def white_beads : ℕ := blue_beads + red_beads
def silver_beads : ℕ := 10

-- Definition of the total number of beads
def total_beads : ℕ := blue_beads + red_beads + white_beads + silver_beads

-- The main theorem statement
theorem total_beads_correct : total_beads = 40 :=
by 
  sorry

end total_beads_correct_l607_607927


namespace max_tuesdays_in_45_days_l607_607607

theorem max_tuesdays_in_45_days : 
  ∀ (start_day : ℕ), start_day ∈ {0, 1, 2, 3, 4, 5, 6} → 
  (∃ tuesdays : ℕ, tuesdays ≤ 7 ∧ 
  (tuesdays = 6 + if start_day ≤ 2 then 1 else 0) ∧ 
  tuesdays = 7) :=
by
  intro start_day
  intro start_day_range
  use 7
  sorry

end max_tuesdays_in_45_days_l607_607607


namespace factorize_expr_l607_607746

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l607_607746


namespace correct_option_B_l607_607686

axiom mitosis_stability : ∀ (cell : Type), (cell → Prop) → (somatic : cell → Prop) → (late_mitosis : cell → Prop) → diploid (cell : Type) → 
  (∀ c, late_mitosis c → ¬ ∀ p, homologous_chromosomes p → pole_of_cell p c)

axiom random_distribution : ∀ (cell : Type), diploid cell →  (∀ c, cytoplasm c → random_unequal_distribution c)

axiom meiotic_division : ∀ (cell : Type), diploid cell → (∀ c, late_meiotic_II c → 1/2 chromosome_number c = somatic_chromosome_number c)

axiom allele_sorting : ∀ (cell : Type), diploid cell → 
  (∀ c, meiotic_I c → ∃ s, separation_of_alleles s ∧ meiotic_II s → independent_assortment s)

theorem correct_option_B : ∀ (cell: Type), diploid cell → (∀ c, division c → random_unequal_distribution c) :=
by 
  intros,
  exact random_distribution

end correct_option_B_l607_607686


namespace students_journals_l607_607642

theorem students_journals :
  ∃ u v : ℕ, 
    u + v = 75000 ∧ 
    (7 * u + 2 * v = 300000) ∧ 
    (∃ b g : ℕ, b = u * 7 / 300 ∧ g = v * 2 / 300 ∧ b = 700 ∧ g = 300) :=
by {
  -- The proving steps will go here
  sorry
}

end students_journals_l607_607642


namespace factorize_expression_l607_607742

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l607_607742


namespace attic_current_junk_items_l607_607926

def initial_useful_items : ℕ := 20
def percentage_useful_items : ℝ := 0.20
def percentage_valuable_heirlooms : ℝ := 0.10
def percentage_junk : ℝ := 0.70
def given_away_useful_items : ℕ := 4
def sold_valuable_items : ℕ := 20
def remaining_useful_items : ℕ := 16

noncomputable def total_initial_items : ℝ := initial_useful_items / percentage_useful_items

noncomputable def initial_valuable_heirlooms : ℝ := total_initial_items * percentage_valuable_heirlooms

noncomputable def initial_junk_items : ℝ := total_initial_items * percentage_junk

noncomputable def extra_valuable_items_sold : ℕ := sold_valuable_items - initial_valuable_heirlooms

def current_junk_items : ℕ := initial_junk_items - extra_valuable_items_sold

theorem attic_current_junk_items : current_junk_items = 60 := by
  sorry

end attic_current_junk_items_l607_607926


namespace sum_of_valid_m_values_l607_607676

theorem sum_of_valid_m_values :
  (∀ m : ℤ, 6 + 10 > m ∧ 6 + m > 10 ∧ 10 + m > 6 → m ∈ {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) →
  ∑ m in finset.Icc 5 15, m = 110 :=
by
  sorry

end sum_of_valid_m_values_l607_607676


namespace fraction_of_married_men_l607_607338

def num_women (n : ℕ) := n
def prob_single : ℚ := 3 / 5

theorem fraction_of_married_men (n : ℕ) (hn : n > 0) :
  let single_women := n * prob_single in
  let married_women := n - single_women in
  let married_men := married_women in
  let total_people := n + married_men in
  (married_men / total_people) = (2 / 7) :=
by
  let single_women := n * prob_single
  let married_women := n - single_women
  let married_men := married_women
  let total_people := n + married_men
  have h_single_women : single_women = 3 / 5 * n := sorry
  have h_married_women : married_women = n - single_women := sorry
  have h_married_men : married_men = married_women := sorry
  have h_total_people : total_people = n + married_men := sorry
  have h_fraction : (married_men / total_people) = (2 / 7) := sorry
  exact h_fraction

end fraction_of_married_men_l607_607338


namespace angela_problems_l607_607331

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end angela_problems_l607_607331


namespace equal_power_intersection_l607_607940

noncomputable def power_of_point (A : Point) (O : Circle) : ℝ :=
  let ⟨M, N⟩ := secant_intersections_through A O in
  M.distance_to(A) * N.distance_to(A)

theorem equal_power_intersection (A B : Point) (O : Circle)
  (M N P Q E F: Point)
  (h1 : power_of_point A O = power_of_point B O)
  (h2 : line_through A M.intersect_circle O = {M, N})
  (h3 : line_through B P.intersect_circle O = {P, Q})
  (h4 : MP_intersects_AB E)
  (h5 : NQ_intersects_AB F)
  : power_of_point E O = power_of_point F O := sorry

end equal_power_intersection_l607_607940


namespace average_and_variance_of_remaining_scores_l607_607110

-- Define the original scores
def original_scores : List ℕ := [90, 89, 90, 95, 93, 94, 93]

-- Define the function to remove the highest and lowest scores
def remove_extremes (scores : List ℕ) : List ℕ :=
  let sorted_scores := List.sort scores
  List.drop 1 (List.take (scores.length - 1) sorted_scores)

-- Define the remaining scores after removing the highest and lowest
def remaining_scores : List ℕ := remove_extremes original_scores

-- Define the mean of the remaining scores
def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Define the variance of the remaining scores
def variance (l : List ℕ) (μ : ℚ) : ℚ :=
  (l.map (λ x => (x : ℚ - μ)^2)).sum / l.length

-- The final proof problem
theorem average_and_variance_of_remaining_scores :
  mean remaining_scores = 92 ∧ variance remaining_scores 92 = 2.8 :=
by
  sorry

end average_and_variance_of_remaining_scores_l607_607110


namespace minimize_sum_of_squares_of_roots_l607_607394

theorem minimize_sum_of_squares_of_roots :
  ∀ m : ℝ, x^2 - (m + 1) * x + (m - 1) = 0 → 
    (∃ m : ℝ, m = 0 ∧ ∀ m', (m' ≠ 0 → (m^2 + 3 ≤ m'^2 + 3)),
sorry

end minimize_sum_of_squares_of_roots_l607_607394


namespace math_proof_equivalent_problem_l607_607790

section Proof

-- Given the conditions in the problem
variables {a b c : ℝ}
def f (x : ℝ) := a * x^2 + b * x + c

lemma problem_inequality_1 (x : ℝ) : 2 * x ≤ f x :=
sorry

lemma problem_inequality_2 (x : ℝ) : f x ≤ (1 / 2) * (x + 1)^2 :=
sorry

lemma part1_range_f_neg1 : (-2 : ℝ) < f (-1) ∧ f (-1) ≤ 0 :=
sorry

lemma part2_range_a : (1 / 4 : ℝ) ≤ a ∧ a ≤ (9 + Real.sqrt 17) / 32 :=
sorry

-- Now we state the combined theorem
theorem math_proof_equivalent_problem :
  ((∀ x : ℝ, 2 * x ≤ f x) ∧ (∀ x : ℝ, f x ≤ (1 / 2) * (x + 1)^2) →
  (∃ y : ℝ, (-2 : ℝ) < y ∧ y ≤ 0 ∧ y = f (-1))) ∧
  ((∀ x1 x2 ∈ Icc (-3 : ℝ) (-1), abs (f x1 - f x2) ≤ 1) →
  ((1 / 4 : ℝ) ≤ a ∧ a ≤ (9 + Real.sqrt 17) / 32)) :=
by
  constructor
  { intro h
    cases h with h1 h2
    exact part1_range_f_neg1 h1 h2 }
  { intro h
    exact part2_range_a h }

end Proof

end math_proof_equivalent_problem_l607_607790


namespace range_of_x_l607_607808

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f to satisfy given conditions later

theorem range_of_x (hf_odd : ∀ x : ℝ, f (-x) = - f x)
                   (hf_inc_mono_neg : ∀ x y : ℝ, x ≤ y → y ≤ 0 → f x ≤ f y)
                   (h_ineq : f 1 + f (Real.log x - 2) < 0) : (0 < x) ∧ (x < 10) :=
by
  sorry

end range_of_x_l607_607808


namespace solve_exponential_equation_l607_607958

theorem solve_exponential_equation (x : ℝ) (h : (16 ^ x) * (16 ^ x) * (16 ^ x) * (16 ^ x) = 256 ^ 4) : 
  x = 2 := by 
sorry

end solve_exponential_equation_l607_607958


namespace young_employees_l607_607225

theorem young_employees (ratio_young : ℕ)
                        (ratio_middle : ℕ)
                        (ratio_elderly : ℕ)
                        (sample_selected : ℕ)
                        (prob_selection : ℚ)
                        (h_ratio : ratio_young = 10 ∧ ratio_middle = 8 ∧ ratio_elderly = 7)
                        (h_sample : sample_selected = 200)
                        (h_prob : prob_selection = 0.2) :
                        10 * (sample_selected / prob_selection) / 25 = 400 :=
by {
  sorry
}

end young_employees_l607_607225


namespace miles_reads_129_pages_l607_607928

def hours_per_day : ℝ := 24
def fraction_of_day_reading : ℝ := 1/6
def total_reading_hours : ℝ := hours_per_day * fraction_of_day_reading

def genres : List String := ["novels", "graphic_novels", "comic_books", "non_fiction", "biographies"]
def fraction_per_genre : ℝ := 1/5

def time_per_genre : ℝ := total_reading_hours * fraction_per_genre
def fraction_per_focus : ℝ := 1/3

def time_per_focus : ℝ := time_per_genre * fraction_per_focus

def reading_speeds : List (String × List ℝ) :=
  [ ("novels", [21, 25, 30])
  , ("graphic_novels", [30, 36, 42])
  , ("comic_books", [45, 54, 60])
  , ("non_fiction", [18, 22, 28])
  , ("biographies", [20, 24, 29])
  ]

def pages_read (time : ℝ) (speeds : List ℝ) : ℝ :=
  time * speeds[0] + time * speeds[1] + time * speeds[2]

def total_pages_read (time : ℝ) (speeds : List (String × List ℝ)) : ℝ :=
  speeds.foldl (λ acc x, acc + pages_read time x.2) 0

noncomputable def miles_total_pages : ℝ :=
  total_pages_read time_per_focus reading_speeds

theorem miles_reads_129_pages : miles_total_pages = 129 :=
by
  sorry

end miles_reads_129_pages_l607_607928


namespace oc_times_ab_lt_oa_times_bc_add_ob_times_ac_l607_607335

variables {A B C O : Type} [Points : MetricSpace]
variables (oAB : IsOnSegment O A B) (neAO : A ≠ O) (neBO : B ≠ O)

theorem oc_times_ab_lt_oa_times_bc_add_ob_times_ac :
  dist O C * dist A B < dist O A * dist B C + dist O B * dist A C := sorry

end oc_times_ab_lt_oa_times_bc_add_ob_times_ac_l607_607335


namespace find_x7_plus_32x2_l607_607173

theorem find_x7_plus_32x2 (x : ℝ) (h : x^3 + 2 * x = 4) : x^7 + 32 * x^2 = 64 :=
sorry

end find_x7_plus_32x2_l607_607173


namespace hospitalization_fee_l607_607189

-- Definitions for the problem conditions
def reimbursement_rate : ℝ → ℝ 
| x := if x ≤ 500 then 0
       else if x ≤ 1000 then 0.6
       else if x ≤ 3000 then 0.8
       else 0 -- assumption for simplification, extending the table pattern

-- The total reimbursement
noncomputable def total_reimbursement (x : ℝ) : ℝ :=
  let r1 := 0.6 * min (x - 500) 500
  let r2 := 0.8 * max (x - 1000) 0
  r1 + r2

-- Main theorem statement
theorem hospitalization_fee (h : total_reimbursement 2000 = 1100) : x = 2000 := 
by {
  have h1 : 0.6 * 500 = 300 := by norm_num,
  have h2 : 0.8 * (2000 - 1000) = 800 := by norm_num,
  have h3 : total_reimbursement 2000 = 300 + 800 := by simp [total_reimbursement, h1, h2],
  exact eq_of_eq_true h3,
}

end hospitalization_fee_l607_607189


namespace vector_magnitude_l607_607046

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude : 
  let AB := (-1, 2)
  let BC := (x, -5)
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  dot_product AB BC = -7 → magnitude AC = 5 :=
by sorry

end vector_magnitude_l607_607046


namespace orthocenter_of_triangle_ABC_l607_607479

def point : Type := ℝ × ℝ × ℝ

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)

def orthocenter (A B C : point) : point := sorry -- We'll skip the function implementation here

theorem orthocenter_of_triangle_ABC :
  orthocenter A B C = (13/7, 41/14, 55/7) :=
sorry

end orthocenter_of_triangle_ABC_l607_607479


namespace correct_relationship_l607_607020

variables (f : ℝ → ℝ) (f' : ℝ → ℝ) (H_diff : ∀ x, differentiable_at ℝ f x) 

-- Given conditions as assumptions
axiom H1 : ∀ x, f'' x + f x < 0

noncomputable def proof_problem : Prop :=
  f 1 < f 0 / real.exp 1 ∧ f 0 / real.exp 1 < f (-1) / real.exp 2

theorem correct_relationship :
  proof_problem f f' H_diff :=
begin
  sorry,
end

end correct_relationship_l607_607020


namespace number_of_students_l607_607975

theorem number_of_students (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 18 → n = 424 := by
  intro h
  cases h with n_lt_600 h'
  cases h' with n_mod_25_24 n_mod_19_18
  sorry

end number_of_students_l607_607975


namespace fourth_throw_probability_l607_607257

-- Define a fair dice where each face has an equal probability.
def fair_dice (n : ℕ) : Prop := (n >= 1 ∧ n <= 6)

-- Define the probability of rolling a 6 on a fair dice.
noncomputable def probability_of_6 : ℝ := 1 / 6

/-- 
  Prove that the probability of getting a "6" on the 4th throw is 1/6 
  given that the dice is fair and the first three throws result in "6".
-/
theorem fourth_throw_probability : 
  (∀ (n1 n2 n3 : ℕ), fair_dice n1 ∧ fair_dice n2 ∧ fair_dice n3 ∧ n1 = 6 ∧ n2 = 6 ∧ n3 = 6) 
  → (probability_of_6 = 1 / 6) :=
by 
  sorry

end fourth_throw_probability_l607_607257


namespace supplement_of_supplement_of_58_l607_607248

theorem supplement_of_supplement_of_58 (α : ℝ) (h : α = 58) : 180 - (180 - α) = 58 :=
by
  sorry

end supplement_of_supplement_of_58_l607_607248


namespace complex_rhombus_abs_l607_607920

variables (z : ℂ) (a b : ℝ)
def is_rhombus (A B C D : ℂ) : Prop := sorry -- fill in based on geometric definitions

theorem complex_rhombus_abs (hz : z = a + b * complex.I)
  (hz_norm : complex.abs z = 2)
  (h_points : let A := z, B := -z, C := z^2 - z + 1, D := z^2 + z + 1 in is_rhombus A B C D) :
  |a| + |b| = (real.sqrt 7 + 3) / 2 :=
by sorry

end complex_rhombus_abs_l607_607920


namespace shaded_area_percentage_in_checkerboard_grid_is_50_percent_l607_607255

theorem shaded_area_percentage_in_checkerboard_grid_is_50_percent :
  let n := 6 in
  let total_squares := n * n in
  let shaded_squares := total_squares / 2 in
  (shaded_squares / total_squares : ℚ) * 100 = 50 :=
by
  sorry

end shaded_area_percentage_in_checkerboard_grid_is_50_percent_l607_607255


namespace find_radius_of_semicircular_plot_l607_607298

noncomputable def radius_of_semicircular_plot (π : ℝ) : ℝ :=
  let total_fence_length := 33
  let opening_length := 3
  let effective_fence_length := total_fence_length - opening_length
  let r := effective_fence_length / (π + 2)
  r

theorem find_radius_of_semicircular_plot 
  (π : ℝ) (Hπ : π = Real.pi) :
  radius_of_semicircular_plot π = 30 / (Real.pi + 2) :=
by
  unfold radius_of_semicircular_plot
  rw [Hπ]
  sorry

end find_radius_of_semicircular_plot_l607_607298


namespace range_of_a_l607_607817

-- Given conditions
variables (a b : ℝ)
def f (x : ℝ) : ℝ := -x^3 + a * x^2 + b

-- Statement to be proved
theorem range_of_a (h : ∀ x : ℝ, (-3 * x^2 + 2 * a * x) < 1) : 
  -real.sqrt 3 < a ∧ a < real.sqrt 3 :=
sorry

end range_of_a_l607_607817


namespace segments_can_form_triangle_l607_607312

noncomputable def can_form_triangle (a b c : ℝ) : Prop :=
  a + b + c = 2 ∧ a + b > 1 ∧ a + c > b ∧ b + c > a

theorem segments_can_form_triangle (a b c : ℝ) (h : a + b + c = 2) : (a + b > 1) ↔ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end segments_can_form_triangle_l607_607312


namespace largest_number_folders_in_package_l607_607477

def gcd_folders (a b : ℕ) : ℕ := Nat.gcd a b

theorem largest_number_folders_in_package (a b : ℕ) (h₁ : a = 60) (h₂ : b = 90) :
  gcd_folders a b = 30 :=
by
  rw [h₁, h₂]
  exact Nat.gcd_comm 60 90
  sorry

end largest_number_folders_in_package_l607_607477


namespace quadratic_real_solution_probability_l607_607478

-- Definition of quadratic equation coefficients from the problem conditions
def is_real_solution (a c : ℕ) : Prop :=
  a * c ≤ 4

-- Total sample space for (a, c) selections
def total_combinations : List (ℕ × ℕ) := 
  [ (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3) ]

-- Valid pairs satisfying the discriminant condition
def real_solution_combinations (l : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  l.filter (λ x => is_real_solution x.1 x.2)

-- The probability of obtaining a real solution
def probability_real_solution : ℚ :=
  (real_solution_combinations total_combinations).length / total_combinations.length

theorem quadratic_real_solution_probability : probability_real_solution = 2 / 3 :=
  sorry

end quadratic_real_solution_probability_l607_607478


namespace angle_value_l607_607878

theorem angle_value (EF GH : ℝ) (L M N : ℕ) 
  (h1: ∠ ELF = 110) (h2: ∠ MLN = 70) (h3: ∠ NLM = 40) : 
  ∠ x = 40 := 
by sorry

end angle_value_l607_607878


namespace carpet_dimensions_l607_607349

variable (q : ℕ) (k : ℚ)
variables (a b x y : ℕ)

def carpet_conditions (q : ℕ) (k : ℚ) :=
  ∃ a b : ℕ, 
    x = a ∧ 
    y = b ∧ 
    (a + b * k = 50) ∧ 
    (a * k + b = q) ∧ 
    (x^2 = a^2 + b^2 = (q * k - 50)^2 / (k^2 - 1)^2 + (50 * k - q)^2 / (k^2 - 1)^2) ∧
    (x^2 = (q * k - 38)^2 / (k^2 - 1)^2 + (38 * k - q)^2 / (k^2 - 1)^2)

theorem carpet_dimensions : carpet_conditions q k → (x = 25) ∧ (y = 50) := by
  sorry

end carpet_dimensions_l607_607349


namespace alice_price_per_acorn_l607_607458

theorem alice_price_per_acorn
  (num_acorns : ℕ) (bob_paid : ℕ) (alice_factor : ℕ) (num_acorns_alice : ℕ) (total_alice_paid : ℕ) :
  num_acorns = 3600 →
  bob_paid = 6000 →
  alice_factor = 9 →
  num_acorns_alice = num_acorns →
  total_alice_paid = alice_factor * bob_paid →
  total_alice_paid / num_acorns_alice = 15 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end alice_price_per_acorn_l607_607458


namespace value_of_a10_l607_607828

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ 
  (∀ n, a (2 * n) - a (2 * n - 1) = (2 ^ (2 * n - 1))) ∧
  (∀ n, a (2 * n + 1) - a (2 * n) = (2 ^ (2 * n)))

theorem value_of_a10 (a : ℕ → ℤ) (h : sequence a) : 
  a 10 = 1021 := 
sorry

end value_of_a10_l607_607828


namespace find_investment_sum_l607_607265

variable (P : ℝ)

def simple_interest (rate time : ℝ) (principal : ℝ) : ℝ :=
  principal * rate * time

theorem find_investment_sum (h : simple_interest 0.18 2 P - simple_interest 0.12 2 P = 240) :
  P = 2000 :=
by
  sorry

end find_investment_sum_l607_607265


namespace sin_cos_sum_l607_607045

theorem sin_cos_sum (α : ℝ) (x y r : ℝ) (hx : x = 3) (hy : y = -4) (hr : r = 5) 
  (har : r = Real.sqrt (x^2 + y^2)) : 
  (Real.sin α + Real.cos α) = -1/5 :=
by
  rw [hx, hy, hr] at har
  have ha : r = 5 := by rw har -- using given information r = 5
  rw [Real.sin, Real.cos]
  sorry

end sin_cos_sum_l607_607045


namespace n_is_power_of_p_l607_607135

theorem n_is_power_of_p 
  (x y p n k : ℕ) 
  (hxy : x > 0 ∧ y > 0) 
  (hp : p > 0) 
  (hn : n > 1 ∧ odd n) 
  (hpx : Prime p) 
  (heq : x ^ n + y ^ n = p ^ k) 
  (hpodd: ∃ q : ℕ, p = 2 * q + 1) : ∃ m : ℕ, n = p ^ m :=
by sorry

end n_is_power_of_p_l607_607135


namespace malou_average_score_l607_607529

def quiz1_score := 91
def quiz2_score := 90
def quiz3_score := 92

def sum_of_scores := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes := 3

theorem malou_average_score : sum_of_scores / number_of_quizzes = 91 :=
by
  sorry

end malou_average_score_l607_607529


namespace find_m_l607_607448

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vec_c : ℝ × ℝ := (3, -2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m (m : ℝ) (h : perpendicular (vec_a.1 - vec_b m.1, vec_a.2 - vec_b m.2) vec_c) : m = -3 :=
by sorry

end find_m_l607_607448


namespace lg_45_eq_l607_607778

variable (m n : ℝ)
axiom lg_2 : Real.log 2 = m
axiom lg_3 : Real.log 3 = n

theorem lg_45_eq : Real.log 45 = 1 - m + 2 * n := by
  -- proof to be filled in
  sorry

end lg_45_eq_l607_607778


namespace mat_length_on_round_table_l607_607673

theorem mat_length_on_round_table
    (radius : ℝ)
    (num_mats : ℕ)
    (width x : ℝ)
    (corner_touch_edge : ∀ mat, mat ∈ set.range (λ n, (n : ℝ) / num_mats * 2 * real.pi) → 
                              ∃ θ : ℝ, cos θ = 1 ∧ sin θ = 0)
    (adjacent_mats_inner_corners_touch : ∀ n : ℕ, n < num_mats → ∃ θ : ℝ, cos θ * radius = radius)
    (radius_eq : radius = 5)
    (num_mats_eq : num_mats = 8)
    (width_eq : width = 1) :
  x = 5 * real.sqrt(2 - real.sqrt 2) :=
sorry

end mat_length_on_round_table_l607_607673


namespace mitchell_total_pages_read_l607_607534

def num_pages_read_before_4 := [30, 35, 45, 28, 50, 32, 38, 47, 29, 40]

def length_eleventh_chapter := 60
def length_twelfth_chapter (length_tenth_chapter : ℕ) := (75 * length_tenth_chapter) / 100
def length_thirteenth_chapter (length_sixth_chapter : ℕ) := (length_sixth_chapter / 2) + 20

theorem mitchell_total_pages_read :
  let pages_before_4 := num_pages_read_before_4.sum in
  let pages_after_4 := (length_eleventh_chapter / 2) +
                       length_twelfth_chapter 40 +
                       length_thirteenth_chapter 32 in
  pages_before_4 + pages_after_4 = 470 :=
by
  sorry

end mitchell_total_pages_read_l607_607534


namespace transportation_charges_l607_607200

-- Define the conditions
def cost_of_machine := 13000
def cost_of_repair := 5000
def selling_price := 28500
def profit_rate := 1.5

-- Define the transportation charges (unknown)
variables (T : ℝ)

-- Calculate the total cost including transportation charges
def total_cost_including_transportation := cost_of_machine + cost_of_repair + T

-- Calculate the selling price based on the total cost and profit rate
def calculated_selling_price := profit_rate * total_cost_including_transportation

-- Theorem to prove the transportation charges are Rs 1000
theorem transportation_charges :
  calculated_selling_price = selling_price → T = 1000 :=
by
  sorry

end transportation_charges_l607_607200


namespace parabola_zeros_difference_l607_607991

theorem parabola_zeros_difference (a b c : ℝ) (h₁ : (3 : ℝ), -9 = (a * (3 : ℝ)^2 + b * (3 : ℝ) + c)) (h₂ : (5 : ℝ), 7 = (a * (5 : ℝ)^2 + b * (5 : ℝ) + c)) (m n : ℝ) (h₃ : m > n) (h₄ : 0 = a * m^2 + b * m + c) (h₅ : 0 = a * n^2 + b * n + c) : m - n = 3 :=
sorry

end parabola_zeros_difference_l607_607991


namespace bryson_shoes_l607_607343

theorem bryson_shoes (pairs: ℕ) (shoes_per_pair: ℕ) (h1: pairs = 2) (h2: shoes_per_pair = 2) : pairs * shoes_per_pair = 4 :=
by {
  rw [h1, h2],
  norm_num,
  sorry -- proof step is skipped
}

end bryson_shoes_l607_607343


namespace part1_part2_l607_607640

-- Part 1: Prove the equivalence of the given function definition
theorem part1 (f : ℝ → ℝ) (x : ℝ) 
  (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  f x = x^2 - 4 * x + 3 :=
sorry

-- Part 2: Maximum value of the given function
theorem part2 (x : ℝ) :
  (∃ y, (∀ x, (g(x) = 1 / (1 - x * (1 - x))) → y <= g x) ∧
   (∀ x, x = 1/2 → g x = y)) :=
begin
  let g : ℝ → ℝ := λ x, 1 / (1 - x * (1 - x)),
  use (4 / 3),
  split,
  { intros x h1,
    rw h1,
    sorry },
  { intros x h2,
    rw h2,
    let num : ℝ := 1,
    let den : ℝ := (x - 1/2)^2 + 3/4,
    have : x = 1/2 := by assumption,
    rw this,
    have : den = 3/4 := by finish,
    have : g (1 / (den)) = 4/3 := by finish,
    exact 1 / 3/4 = 4/3 }
end


end part1_part2_l607_607640


namespace lunch_break_length_l607_607546

theorem lunch_break_length (p h : ℝ) (L : ℝ) (hp : p > 0) (hh : h > 0)
    (condition1 : (9 - L) * (p + h) = 0.6)
    (condition2 : (7 - L) * h = 0.3)
    (condition3 : (12 - L) * p = 0.1) : L = 1 :=
by
  sorry

end lunch_break_length_l607_607546


namespace triangle_XYZ_MN_eq_l607_607492

theorem triangle_XYZ_MN_eq (
  (XY XZ YZ : ℝ) (L K M N : ℝ) :
  XY = 156 → XZ = 143 → YZ = 150 →
  ∃L L∈(YZ) ∧ 
  ∃K K∈(XZ) ∧  
  ∃ (M N : ℝ) (M∈(perp_to(BK)) ∧ N∈(perp_to(AL))) → 
  MN = 137 / 2 :=
begin
  sorry
end

end triangle_XYZ_MN_eq_l607_607492


namespace equal_segment_sums_impossible_l607_607596

-- Definitions corresponding to the given conditions
def int_list : list ℕ := [2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

def total_sum (l : list ℕ) : ℕ := l.sum  -- defining the sum of all integers in the list

def double_total_sum (l : list ℕ) : ℕ := 2 * total_sum l

-- Determine if it is possible for the numbers to be arranged such that each of the 7 segments (composed of 4 circles) has the same sum
def check_equal_segment_sums (l : list ℕ) : Prop :=
  let total := double_total_sum l in  -- total sum to be distributed among 7 segments (each segment being counted twice)
  total % 7 = 0

theorem equal_segment_sums_impossible : ¬ check_equal_segment_sums int_list := 
by
  sorry

end equal_segment_sums_impossible_l607_607596


namespace correct_statement_l607_607016

def is_ellipse (m : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (m ∈ (1/2, 2)) ∧ C = λ x y, a * x^2 + b * y^2 = 1

theorem correct_statement (m : ℝ) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, C x y ↔ (2 - m) * x^2 + (m + 1) * y^2 = 1) →
  is_ellipse m C :=
by
  intro hC
  sorry

end correct_statement_l607_607016


namespace tyson_age_l607_607133

noncomputable def age_proof : Prop :=
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  t = 20           -- Statement that needs to be proved

theorem tyson_age : age_proof :=
by
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  show t = 20
  sorry

end tyson_age_l607_607133


namespace geometric_locus_midpoints_of_sides_and_centers_l607_607319

variables {A B C D : Type} [EuclideanGeometry A B C D]
variables {O1 O2 K1 K2 : Type} [EuclideanGeometry O1 O2 K1 K2]

theorem geometric_locus_midpoints_of_sides_and_centers
  (ABCD_is_quadrilateral : is_quadrilateral A B C D)
  (rectangles_perpendicular : ∀ rect : Rectangle, rectangle_sides_perpendicular_to_quadrilateral_sides rect A B C D)
  (K1_is_midpoint_AC : is_midpoint K1 A C)
  (K2_is_midpoint_BD : is_midpoint K2 B D)
  (O_is_center_rectangle : ∀ rect : Rectangle, is_center O rect) :
  (∀ O1 O2, is_circle_with_diameter O1 A C ∧ is_midpoint O1 side1 ∧ is_midpoint O2 side2) ∧
  (is_circle_with_diameter O K1 K2) :=
sorry

end geometric_locus_midpoints_of_sides_and_centers_l607_607319


namespace superior_points_in_Omega_l607_607830

-- Define the set Omega
def Omega : Set (ℝ × ℝ) := { p | let (x, y) := p; x^2 + y^2 ≤ 2008 }

-- Definition of the superior relation
def superior (P P' : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x', y') := P'
  x ≤ x' ∧ y ≥ y'

-- Definition of the set of points Q such that no other point in Omega is superior to Q
def Q_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p; x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 }

theorem superior_points_in_Omega :
  { p | p ∈ Omega ∧ ¬ (∃ q ∈ Omega, superior q p) } = Q_set :=
by
  sorry

end superior_points_in_Omega_l607_607830


namespace no_valid_weights_l607_607993

theorem no_valid_weights (w_1 w_2 w_3 w_4 : ℝ) : 
  w_1 + w_2 + w_3 = 100 → w_1 + w_2 + w_4 = 101 → w_2 + w_3 + w_4 = 102 → 
  w_1 < 90 → w_2 < 90 → w_3 < 90 → w_4 < 90 → False :=
by 
  intros h1 h2 h3 hl1 hl2 hl3 hl4
  sorry

end no_valid_weights_l607_607993


namespace evaluate_expression_l607_607419

variable (x y z : ℝ)

theorem evaluate_expression (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := 
sorry

end evaluate_expression_l607_607419


namespace parallelogram_diagonal_property_l607_607549

-- Definition of points and segments
variables {A B C D E F : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]

-- Functions to represent our conditions
def parallelogram (A B C D : Type) : Prop := sorry
def longer_diagonal (A C : Type) : Prop := sorry
def projection (C : Type) (A B : Type) (E : Type) : Prop := sorry
def projection_C_on_AB : Prop := projection C A B E
def projection_C_on_AD : Prop := projection C A D F

-- Main theorem
theorem parallelogram_diagonal_property
  (h1 : parallelogram A B C D)
  (h2 : longer_diagonal A C)
  (h3 : projection_C_on_AB)
  (h4 : projection_C_on_AD) :
  AB * AE + AD * AF = AC * AC :=
sorry

end parallelogram_diagonal_property_l607_607549


namespace sin_alpha_proof_l607_607800

theorem sin_alpha_proof (alpha beta : ℝ) (h1 : 0 < alpha ∧ alpha < π / 2)
  (h2 : 2 * tan (π - alpha) - 3 * cos (π / 2 + beta) + 5 = 0)
  (h3 : tan (π + alpha) + 6 * sin (π + beta) = 1) :
  sin alpha = 3 * sqrt 10 / 10 := 
sorry

end sin_alpha_proof_l607_607800


namespace find_correct_numbers_l607_607235

def find_numbers (x y : ℚ) : Prop :=
  38 + 2 * x = 124 ∧
  x + 3 * y = 47 ∧
  x = 43 ∧
  y = 4 / 3

theorem find_correct_numbers : ∃ (x y : ℚ), 38 + 2 * x = 124 ∧ x + 3 * y = 47 ∧ x = 43 ∧ y = 4 / 3 :=
by
  use 43, 4 / 3
  simp [find_numbers]
  split; refl; split; ring; split; refl; refl
  sorry

end find_correct_numbers_l607_607235


namespace sum_first_100_terms_l607_607227

-- Define the sequence based on the given conditions.
def seq : ℕ → ℝ
| 1     := 1
| (2 * k) := 2 * seq (2 * k - 1)
| (2 * k + 1) := 3 * seq (2 * k)

-- Define the sum of the first 100 terms of the sequence.
def S_100 : ℝ := ∑ i in finset.range 100, seq (i + 1)

-- The theorem stating the sum of the first 100 terms.
theorem sum_first_100_terms : S_100 = (3 / 5) * (6 ^ 50 - 1) := by
  sorry

end sum_first_100_terms_l607_607227
